pub mod protocol;
pub mod light_protocol;
use light_protocol::{LightRequest, LightResponse, LIGHT_PROTOCOL};

pub use protocol::{Message, MidstateCodec, MIDSTATE_PROTOCOL, MAX_GETBATCHES_COUNT, MAX_GETHEADERS_COUNT};

use anyhow::Result;
use futures::StreamExt;
use libp2p::{
    autonat,
    dcutr,
    identify, kad,
    noise,
    relay,
    request_response::{
        self, Config as RequestResponseConfig, OutboundRequestId, ProtocolSupport,
        ResponseChannel,
    },
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
    identity::Keypair,
    Multiaddr, PeerId, Swarm,
    core::ConnectedPoint,
    Transport,
};

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::Duration;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

/// Max addresses to send in a single PEX Addr message.
pub const MAX_PEX_ADDRS: usize = 50;

// ── Light Client Protection Constants ────────────────────────────────────────

/// Maximum concurrent WebRTC light client connections.
const MAX_LIGHT_PEERS: usize = 30;
/// Maximum concurrent streams (requests in flight) per light peer.
const MAX_LIGHT_STREAMS_PER_PEER: usize = 5;
/// Rate limit window in seconds.
const LIGHT_RATE_WINDOW_SECS: u64 = 60;

/// Number of rate-limit violations before temporary ban.
const LIGHT_BAN_THRESHOLD: u32 = 3;
/// Duration of a temporary ban in seconds.
const LIGHT_BAN_DURATION_SECS: u64 = 300;
/// Timeout waiting for a light client to send its request data.
const LIGHT_READ_TIMEOUT_SECS: u64 = 10;
/// Timeout waiting for the node to produce a response.
const LIGHT_RESPONSE_TIMEOUT_SECS: u64 = 30;

// ── Light Client Rate Limiter ────────────────────────────────────────────────

struct LightPeerState {
    request_count: u32,
    expensive_count: u32,
    window_start: Instant,
    active_streams: u32,
    violations: u32,
    banned_until: Option<Instant>,
    
    // --- BAYESIAN REPUTATION (Beta Distribution) ---
    /// Prior: successful cryptographic proofs (Honest)
    alpha: u32, 
    /// Prior: failed proofs, spam, or timeouts (Adversarial)
    beta: u32,  
}

impl LightPeerState {
    fn new() -> Self {
        Self {
            request_count: 0,
            expensive_count: 0,
            window_start: Instant::now(),
            active_streams: 0,
            violations: 0,
            banned_until: None,
            // Uniform prior: We start neutral, assuming 1 good and 1 bad interaction
            alpha: 1, 
            beta: 1,
        }
    }

/// Calculate dynamic rate limit based on the Expected Value of the Beta distribution
    fn current_rate_limit(&self) -> u32 {
        // E[X] = alpha / (alpha + beta). 
        // Approaches 1.0 for honest peers, approaches 0.0 for malicious peers.
        let probability_honest = self.alpha as f32 / (self.alpha + self.beta) as f32;

        // If the probability they are honest drops below 10%, throttle them to the floor
        if probability_honest < 0.1 {
            return 5; // 5 requests per minute maximum
        }

        // Base rate for unknown peers (0.5 prob) is ~275. Highly trusted peers scale up to ~500.
        (50.0 + (450.0 * probability_honest)) as u32
    }

    fn current_expensive_limit(&self) -> u32 {
        let probability_honest = self.alpha as f32 / (self.alpha + self.beta) as f32;
        if probability_honest < 0.2 { return 5; }
        // Base 20, max 200. A new peer (0.5 prob) gets 120 requests per minute.
        // 120 requests * 1000 blocks = 120,000 blocks synced per minute over WebRTC.
        (20.0 + (200.0 * probability_honest)) as u32
    }

    /// Reset counters if the window has elapsed.
    fn maybe_reset_window(&mut self) {
        if self.window_start.elapsed().as_secs() >= LIGHT_RATE_WINDOW_SECS {
            self.request_count = 0;
            self.expensive_count = 0;
            self.window_start = Instant::now();
        }
    }

    fn is_banned(&self) -> bool {
        self.banned_until.map_or(false, |t| Instant::now() < t)
    }
}

/// Thread-safe light client rate limiter shared between the event loop
/// and spawned stream-handling tasks.
#[derive(Clone)]
struct LightGuard {
    inner: Arc<tokio::sync::Mutex<LightGuardInner>>,
}

struct LightGuardInner {
    peers: HashMap<PeerId, LightPeerState>,
}

impl LightGuard {
    fn new() -> Self {
        Self {
            inner: Arc::new(tokio::sync::Mutex::new(LightGuardInner {
                peers: HashMap::new(),
            })),
        }
    }

    /// Check if a new stream should be allowed. Returns Err(reason) if denied.
    async fn try_open_stream(&self, peer: PeerId) -> Result<(), &'static str> {
        let mut guard = self.inner.lock().await;
        let state = guard.peers.entry(peer).or_insert_with(LightPeerState::new);

        if state.is_banned() {
            return Err("peer is temporarily banned");
        }

        state.maybe_reset_window();

        if state.active_streams >= MAX_LIGHT_STREAMS_PER_PEER as u32 {
            state.violations += 1;
            if state.violations >= LIGHT_BAN_THRESHOLD {
                state.banned_until = Some(Instant::now() + Duration::from_secs(LIGHT_BAN_DURATION_SECS));
                return Err("banned: too many concurrent streams");
            }
            return Err("too many concurrent streams");
        }

        if state.request_count >= state.current_rate_limit() {
            state.violations += 1;
            if state.violations >= LIGHT_BAN_THRESHOLD {
                state.banned_until = Some(Instant::now() + Duration::from_secs(LIGHT_BAN_DURATION_SECS));
                return Err("banned: rate limit exceeded");
            }
            return Err("rate limit exceeded");
        }

        state.active_streams += 1;
        state.request_count += 1;
        Ok(())
    }

    /// Check if an expensive request (BlockTemplate) is allowed.
    /// Call AFTER try_open_stream succeeds.
    async fn check_expensive(&self, peer: PeerId) -> bool {
        let mut guard = self.inner.lock().await;
        if let Some(state) = guard.peers.get_mut(&peer) {
            state.maybe_reset_window();
            if state.expensive_count >= state.current_expensive_limit() {
                state.violations += 1;
                if state.violations >= LIGHT_BAN_THRESHOLD {
                    state.banned_until = Some(Instant::now() + Duration::from_secs(LIGHT_BAN_DURATION_SECS));
                }
                return false;
            }
            state.expensive_count += 1;
        }
        true
    }

    /// Decrement the active stream counter when a stream finishes.
    async fn close_stream(&self, peer: PeerId) {
        let mut guard = self.inner.lock().await;
        if let Some(state) = guard.peers.get_mut(&peer) {
            state.active_streams = state.active_streams.saturating_sub(1);
        }
    }

    /// Remove all state for a disconnected peer.
    async fn remove_peer(&self, peer: &PeerId) {
        let mut guard = self.inner.lock().await;
        guard.peers.remove(peer);
    }

    /// Check if a peer is currently banned.
    async fn is_banned(&self, peer: &PeerId) -> bool {
        let guard = self.inner.lock().await;
        guard.peers.get(peer).map_or(false, |s| s.is_banned())
    }

  
/// BAYESIAN INFERENCE: Observe an honest interaction
    async fn observe_honest(&self, peer: PeerId) {
        let mut guard = self.inner.lock().await;
        if let Some(state) = guard.peers.get_mut(&peer) {
            // Cap at 10,000 exactly like your FinalityEstimator to prevent overflow
            state.alpha = state.alpha.saturating_add(1).min(10_000);
        }
    }

    /// BAYESIAN INFERENCE: Observe an adversarial interaction
    async fn observe_adversarial(&self, peer: PeerId) {
        let mut guard = self.inner.lock().await;
        if let Some(state) = guard.peers.get_mut(&peer) {
            // Penalize faster than we reward (1 bad act = 10 good acts)
            state.beta = state.beta.saturating_add(10).min(10_000);
            
            // Auto-ban if the probability of honesty collapses completely
            if state.beta > state.alpha * 10 {
                state.banned_until = Some(Instant::now() + Duration::from_secs(LIGHT_BAN_DURATION_SECS));
            }
        }
    }
    
    /// Garbage-collect stale peer entries that have no active streams and
    /// whose rate-limit window hasn't been touched in over an hour.
    /// Prevents unbounded memory growth from peers that connect once and vanish.
    async fn gc_stale(&self) {
        let mut guard = self.inner.lock().await;
        let stale_threshold = Duration::from_secs(3600);
        guard.peers.retain(|_, state| {
            // Keep peers with active streams, active bans, or recent activity
            state.active_streams > 0
                || state.is_banned()
                || state.window_start.elapsed() < stale_threshold
        });
    }
}



// ── Behaviour ───────────────────────────────────────────────────────────────

#[derive(NetworkBehaviour)]
pub struct MidstateBehaviour {
    pub rr: request_response::Behaviour<MidstateCodec>,
    pub light: libp2p_stream::Behaviour,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub relay_server: relay::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub autonat: autonat::Behaviour,
    pub connection_limits: libp2p::connection_limits::Behaviour, 
}

// ── NAT status ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatStatus {
    Unknown,
    Public,
    Private,
}

// ── Events ──────────────────────────────────────────────────────────────────

pub enum NetworkEvent {
    MessageReceived {
        peer: PeerId,
        message: Message,
        channel: Option<ResponseChannel<Message>>,
    },
    LightRequest {
        peer: PeerId,
        request: LightRequest,
        /// Oneshot sender: drop it or send the response; the stream-writing
        /// task in next_event() will write it back to the browser.
        respond: tokio::sync::oneshot::Sender<LightResponse>,
    },
    PeerConnected(PeerId, String),
    PeerDisconnected(PeerId),
    OutgoingConnectionFailed(String),
}

// ── Network API ─────────────────────────────────────────────────────────────

    pub fn is_routable(addr: &Multiaddr) -> bool {
        for proto in addr.iter() {
            match proto {
                libp2p::multiaddr::Protocol::Ip4(ip) => {
                    // Reject Loopback, Private (RFC 1918), and Link-local
                    if ip.is_loopback() || ip.is_private() || ip.is_link_local() { return false; }
                }
                libp2p::multiaddr::Protocol::Ip6(ip) => {
                    // Reject Loopback and Link-local
                    if ip.is_loopback() || (ip.segments()[0] & 0xff00 == 0xfe00) { return false; }
                }
                _ => {}
            }
        }
        true
    }

pub struct MidstateNetwork {
    swarm: Swarm<MidstateBehaviour>,
    /// Incoming raw streams from browser light clients.
    light_incoming: libp2p_stream::IncomingStreams,
    /// Light requests parsed in spawned tasks (so swarm keeps being polled).
    light_rx: tokio::sync::mpsc::UnboundedReceiver<(
        PeerId,
        LightRequest,
        tokio::sync::oneshot::Sender<LightResponse>,
    )>,
    light_tx: tokio::sync::mpsc::UnboundedSender<(
        PeerId,
        LightRequest,
        tokio::sync::oneshot::Sender<LightResponse>,
    )>,
    /// Rate limiter / abuse protection for light protocol.
    light_guard: LightGuard,
    /// Peers that connected via WebRTC (light-only, don't speak binary protocol).
    light_peers: HashSet<PeerId>,
    connected: HashMap<PeerId, ConnectedPoint>,
    pending_requests: HashMap<OutboundRequestId, PeerId>,
    nat_status: NatStatus,
    relay_reservations: HashSet<PeerId>,
    listen_addrs: Vec<Multiaddr>,
    external_addrs: Vec<Multiaddr>,
    subnet_peers: HashMap<IpAddr, HashSet<PeerId>>,
}

impl MidstateNetwork {
    pub async fn new(
        keypair: Keypair,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
    ) -> Result<Self> {
        let peer_id = keypair.public().to_peer_id();
        tracing::info!("Local peer id: {}", peer_id);

        let mut swarm = libp2p::SwarmBuilder::with_existing_identity(keypair.clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_quic()
.with_other_transport(|keypair| {
                let certificate = libp2p_webrtc::tokio::Certificate::generate(&mut rand::rngs::OsRng)
                    .expect("WebRTC certificate generation");
                    
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                    libp2p_webrtc::tokio::Transport::new(keypair.clone(), certificate)
                        .map(|(peer_id, conn), _| (peer_id, libp2p::core::muxing::StreamMuxerBox::new(conn)))
                )
            })?
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key: &libp2p::identity::Keypair, relay_client| {
                let local_peer = key.public().to_peer_id();

            // --- Increase timeout from 60s to 120s to give weaker boards a chance ---
                let rr_config = RequestResponseConfig::default()
                    .with_request_timeout(Duration::from_secs(120));

                let rr = request_response::Behaviour::new(
                    [(MIDSTATE_PROTOCOL, ProtocolSupport::Full)],
                    rr_config,
                );
                // -------------------------------------------------

                let light = libp2p_stream::Behaviour::new();
                let kad_store = kad::store::MemoryStore::new(local_peer);
                let mut kademlia = kad::Behaviour::new(local_peer, kad_store);
                kademlia.set_mode(Some(kad::Mode::Client));

                let identify = identify::Behaviour::new(
                    identify::Config::new(
                        "/midstate/id/1.0.0".to_string(),
                        key.public(),
                    )
                    .with_push_listen_addr_updates(true)
                    .with_interval(Duration::from_secs(60)),
                );

// RELAY CHOKE: Protect our outbound bandwidth from free-riders.
                // We provide just enough bandwidth for light clients to submit
                // transactions, but cut off heavy data streaming.
                let mut relay_config = relay::Config::default();
                relay_config.max_circuits = 16;                 // Max 16 total relayed connections
                relay_config.max_circuits_per_peer = 2;         // Prevent 1 peer taking all 16 slots
                relay_config.max_circuit_duration = std::time::Duration::from_secs(2 * 60); // 2 min max
                relay_config.max_circuit_bytes = 1_048_576;     // 1 MB data limit per circuit

                let relay_server = relay::Behaviour::new(
                    local_peer,
                    relay_config,
                );

                let dcutr = dcutr::Behaviour::new(local_peer);

let autonat = autonat::Behaviour::new(
                    local_peer,
                    autonat::Config {
                        boot_delay: Duration::from_secs(10),
                        refresh_interval: Duration::from_secs(120),
                        retry_interval: Duration::from_secs(60),
                        throttle_server_period: Duration::from_secs(15),
                        only_global_ips: true,
                        ..Default::default()
                    },
                );

                // --- EFFICIENCY FIX: Prevent File Descriptor / Socket Exhaustion ---
                // In libp2p v0.52+, ConnectionLimits is a Behaviour, not a Swarm Config.
                let limits = libp2p::connection_limits::ConnectionLimits::default()
                    .with_max_established_per_peer(Some(2)) // Prevent buggy peers from spamming parallel connections
                    .with_max_pending_incoming(Some(50))
                    .with_max_established_incoming(Some(200)); // Hard cap to prevent OS error 24
                let connection_limits = libp2p::connection_limits::Behaviour::new(limits);
                // -------------------------------------------------------------------

                MidstateBehaviour {
                    rr,
                    light,
                    kademlia,
                    identify,
                    relay_client,
                    relay_server,
                    dcutr,
                    autonat,
                    connection_limits, 
                }
            })?
            .with_swarm_config(|c: libp2p::swarm::Config| {
                c.with_idle_connection_timeout(Duration::from_secs(120))
            })
            .build();

        let (light_tx, light_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut net = Self {
            light_incoming: swarm
                .behaviour_mut()
                .light
                .new_control()
                .accept(LIGHT_PROTOCOL)
                .expect("light protocol already registered"),
            swarm,
            light_rx,
            light_tx,
            light_guard: LightGuard::new(),
            light_peers: HashSet::new(),
            connected: HashMap::new(),
            pending_requests: HashMap::new(),
            nat_status: NatStatus::Unknown,
            relay_reservations: HashSet::new(),
            listen_addrs: Vec::new(),
            external_addrs: Vec::new(),
            subnet_peers: HashMap::new(),
        };

        net.swarm.listen_on(listen_addr.clone())?;

        if let Some(quic_addr) = tcp_to_quic(&listen_addr) {
            match net.swarm.listen_on(quic_addr.clone()) {
                Ok(_) => tracing::info!("Also listening on QUIC: {}", quic_addr),
                Err(e) => tracing::debug!("QUIC listen failed (non-fatal): {}", e),
            }
        }
        // Listen for WebRTC direct connections from browsers
        if let Some(webrtc_addr) = tcp_to_webrtc(&listen_addr) {
            match net.swarm.listen_on(webrtc_addr.clone()) {
                Ok(_) => tracing::info!("WebRTC listening on {}", webrtc_addr),
                Err(e) => tracing::debug!("WebRTC listen failed (non-fatal): {}", e),
            }
        }
        

        for addr in &bootstrap_peers {
            if let Some(peer) = extract_peer_id(addr) {
                net.swarm.behaviour_mut().kademlia.add_address(&peer, addr.clone());
                let relay_addr = addr.clone()
                    .with(libp2p::multiaddr::Protocol::P2pCircuit);
                match net.swarm.listen_on(relay_addr.clone()) {
                    Ok(_) => tracing::info!("Relay-listening through {}", addr),
                    Err(e) => tracing::debug!("Relay listen failed (non-fatal): {}", e),
                }
            }
            if let Err(e) = net.swarm.dial(addr.clone()) {
                tracing::warn!("Failed to dial {}: {}", addr, e);
            }
        }

        if !bootstrap_peers.is_empty() {
            if let Err(e) = net.swarm.behaviour_mut().kademlia.bootstrap() {
                tracing::debug!("Kademlia bootstrap not ready: {}", e);
            }
        }

        Ok(net)
    }

    // ── Public API ──────────────────────────────────────────────────────

    pub fn local_peer_id(&self) -> PeerId {
        *self.swarm.local_peer_id()
    }

    pub fn nat_status(&self) -> NatStatus {
        self.nat_status
    }

    pub fn send(&mut self, peer: PeerId, msg: Message) {
        let req_id = self.swarm.behaviour_mut().rr.send_request(&peer, msg);
        self.pending_requests.insert(req_id, peer);
    }

    pub fn broadcast(&mut self, msg: Message) {
        let peers: Vec<PeerId> = self.connected.keys()
            .filter(|p| !self.light_peers.contains(p))
            .copied().collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
    }

    pub fn broadcast_except(&mut self, exclude: Option<PeerId>, msg: Message) {
        let peers: Vec<PeerId> = self.connected
            .keys()
            .filter(|&&p| Some(p) != exclude && !self.light_peers.contains(&p))
            .copied()
            .collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
    }
pub async fn observe_honest_light_peer(&self, peer: PeerId) {
        self.light_guard.observe_honest(peer).await;
    }
    pub async fn observe_adversarial_light_peer(&self, peer: PeerId) {
        self.light_guard.observe_adversarial(peer).await;
    }
    pub async fn peer_honesty_probability(&self, peer: &PeerId) -> f32 {
        let guard = self.light_guard.inner.lock().await;
        if let Some(state) = guard.peers.get(peer) {
            state.alpha as f32 / (state.alpha + state.beta) as f32
        } else {
            0.5 // Unknown peers start at 50% probability of honesty
        }
    }
    /// Returns true if this peer is a light-only client (WebRTC browser).
    pub fn is_light_peer(&self, peer: &PeerId) -> bool {
        self.light_peers.contains(peer)
    }

    pub fn respond(&mut self, channel: ResponseChannel<Message>, msg: Message) {
        if let Err(_) = self.swarm.behaviour_mut().rr.send_response(channel, msg) {
            tracing::warn!("Failed to send response (channel closed)");
        }
    }

    pub fn peer_count(&self) -> usize {
        self.connected.len()
    }

    pub fn connected_peers(&self) -> Vec<PeerId> {
        self.connected.keys().copied().collect()
    }

    pub fn peer_addrs(&self) -> Vec<String> {
        self.connected.keys().map(|p| p.to_string()).collect()
    }

    pub fn disconnect_peer(&mut self, peer: PeerId) {
        let _ = self.swarm.disconnect_peer_id(peer);
    }

    /// Extract the subnet of a connected peer to facilitate network-level bans
    pub fn peer_subnet(&self, peer: &PeerId) -> Option<IpAddr> {
        self.connected.get(peer).and_then(|endpoint| {
            let addr = endpoint.get_remote_address();
            extract_subnet(addr)
        })
    }

    pub fn add_kad_address(&mut self, peer: PeerId, addr: Multiaddr) {
        self.swarm.behaviour_mut().kademlia.add_address(&peer, addr);
    }

    pub fn random_peer(&self) -> Option<PeerId> {
        use rand::seq::IteratorRandom;
        self.connected
            .keys()
            .filter(|p| !self.light_peers.contains(p))
            .copied()
            .choose(&mut rand::thread_rng())
    }

    // ── PEX (Peer Exchange) ─────────────────────────────────────────

    /// Our own externally-reachable addresses (for advertising to peers).
    /// Prefers confirmed external addrs; falls back to listen addrs.
    pub fn advertisable_addrs(&self) -> Vec<String> {
        let local_id = *self.swarm.local_peer_id();
        let p2p_suffix = libp2p::multiaddr::Protocol::P2p(local_id);

        // Get the confirmed external IP from external_addrs if available
        let external_ip = self.external_addrs.iter()
            .find_map(|a| extract_ip(a));

        let mut addrs: Vec<String> = self.listen_addrs.iter()
            .filter_map(|a| {
                // Only include webrtc-direct addresses
                if !a.to_string().contains("webrtc-direct") { return None; }
                
                let a_str = a.to_string();
                if let Some(ip) = external_ip {
                    let replaced = replace_ip(a, ip);
                    let rep_str = replaced.to_string();
                    if rep_str.contains("/p2p/") {
                        Some(rep_str)
                    } else {
                        Some(replaced.with(p2p_suffix.clone()).to_string())
                    }
                } else {
                    // Fall back to non-localhost listen addrs
                    if is_localhost(a) { return None; }
                    if a_str.contains("/p2p/") {
                        Some(a_str)
                    } else {
                        Some(a.clone().with(p2p_suffix.clone()).to_string())
                    }
                }
            })
            .collect();

        addrs.dedup();
        addrs
    }

    /// Multiaddrs of connected peers (from Kademlia routing table).
    /// These are candidates to send in PEX Addr messages.
    pub fn connected_peer_addrs(&mut self) -> Vec<String> {
        let mut addrs = Vec::new();
        for bucket in self.swarm.behaviour_mut().kademlia.kbuckets() {
            for entry in bucket.iter() {
                let peer = entry.node.key.preimage();
                if !self.connected.contains_key(peer) {
                    continue;
                }
                for addr in entry.node.value.iter() {
                    if is_localhost(addr) {
                        continue;
                    }
                    let a_str = addr.to_string();
                    if a_str.contains("/p2p/") {
                        addrs.push(a_str);
                    } else {
                        let full = addr.clone().with(libp2p::multiaddr::Protocol::P2p(*peer));
                        addrs.push(full.to_string());
                    }
                }
            }
        }
        addrs.truncate(MAX_PEX_ADDRS);
        addrs
    }

    /// Build the PEX addr list: our own addrs + known peer addrs.
    pub fn pex_addrs(&mut self) -> Vec<String> {
        let mut all = self.advertisable_addrs();
        all.extend(self.connected_peer_addrs());
        all.sort();
        all.dedup();
        all.truncate(MAX_PEX_ADDRS);
        all
    }



    /// Try to dial a multiaddr string. Ignores bad parses or dial failures.
    pub fn dial_addr(&mut self, addr_str: &str) {
        match addr_str.parse::<Multiaddr>() {
            Ok(addr) => {
            // NEW: Add the is_routable check here
            if !is_routable(&addr) {
                tracing::debug!("PEX ignoring non-routable address: {}", addr_str);
                return;
            }

            if let Some(peer) = extract_peer_id(&addr) {
                    if self.connected.contains_key(&peer) || peer == *self.swarm.local_peer_id() {
                        return; 
                    }

                    // --- NEW: Prevent dialing saturated subnets ---
                    if let Some(subnet) = extract_subnet(&addr) {
                        if let Some(peers) = self.subnet_peers.get(&subnet) {
                            // Only reject if there are 4+ peers AND this specific peer isn't one of them
                            if peers.len() >= 4 && !peers.contains(&peer) {
                                tracing::debug!("PEX ignoring {}: subnet limit reached", addr_str);
                                return;
                            }
                        }
                    }
                    // ----------------------------------------------

                    
                    if let Err(e) = self.swarm.dial(addr) {
                        tracing::debug!("PEX dial {} failed: {}", addr_str, e);
                    }
                } else {
                    tracing::debug!("Ignoring PEX address without PeerId: {}", addr_str);
                }
            }
            Err(e) => {
                tracing::debug!("PEX ignoring bad multiaddr '{}': {}", addr_str, e);
            }
        }
    }

    // ── Event loop ──────────────────────────────────────────────────────

    pub async fn next_event(&mut self) -> NetworkEvent {
        loop {
            tokio::select! {
                // ── Incoming raw stream from a browser light client ──────
                Some((peer, stream)) = self.light_incoming.next() => {
                    // ── Gate 1: Is peer banned? ──
                    if self.light_guard.is_banned(&peer).await {
                        tracing::debug!("Light stream from banned peer {}, dropping", peer);
                        continue;
                    }

                    // ── Gate 2: Rate limit / concurrent stream check ──
                    match self.light_guard.try_open_stream(peer).await {
                        Ok(()) => {}
                        Err(reason) => {
                            tracing::warn!("Light stream from {} denied: {}", peer, reason);
                            // If banned, disconnect them entirely
                            if reason.starts_with("banned") {
                                let _ = self.swarm.disconnect_peer_id(peer);
                            }
                            continue;
                        }
                    }

                    let tx = self.light_tx.clone();
                    let guard = self.light_guard.clone();
                    tokio::spawn(async move {
                        use light_protocol::{read_request_raw, write_response_raw, LightResponse};
                        let mut stream = stream;

                        // ── Read with timeout ──
                        let read_result = tokio::time::timeout(
                            Duration::from_secs(LIGHT_READ_TIMEOUT_SECS),
                            read_request_raw(&mut stream),
                        ).await;

                        let request = match read_result {
                            Ok(Ok(req)) => req,
                            Ok(Err(e)) => {
                                tracing::debug!("Light read error from {}: {}", peer, e);
                                guard.close_stream(peer).await;
                                return;
                            }
                            Err(_) => {
                                tracing::debug!("Light read timeout from {}", peer);
                                guard.close_stream(peer).await;
                                return;
                            }
                        };

                        // ── Gate 3: Expensive request throttle ──
                        let is_expensive = matches!(
                            request, 
                            LightRequest::BlockTemplate { .. } | LightRequest::GetFilters { .. }
                        );
                        if is_expensive && !guard.check_expensive(peer).await {
                            tracing::warn!("Light {}: BlockTemplate rate limit hit", peer);
                            let _ = write_response_raw(
                                &mut stream,
                                LightResponse::error("rate limit: too many template requests"),
                            ).await;
                            guard.close_stream(peer).await;
                            return;
                        }

                        tracing::debug!("Light request from {}: {:?}", peer, request);

                        // ── Send to node for processing ──
                        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel::<LightResponse>();
                        if tx.send((peer, request, resp_tx)).is_err() {
                            guard.close_stream(peer).await;
                            return;
                        }

                        // ── Await response with timeout ──
                        let resp = match tokio::time::timeout(
                            Duration::from_secs(LIGHT_RESPONSE_TIMEOUT_SECS),
                            resp_rx,
                        ).await {
                            Ok(Ok(resp)) => resp,
                            Ok(Err(_)) => LightResponse::error("internal error"),
                            Err(_) => {
                                tracing::warn!("Light {}: node response timeout", peer);
                                LightResponse::error("server timeout")
                            }
                        };

                        // ── Write response ──
                        if let Err(e) = write_response_raw(&mut stream, resp).await {
                            tracing::debug!("Light {}: write failed: {}", peer, e);
                        }

                        guard.close_stream(peer).await;
                    });
                    continue;
                }

                // ── Light requests parsed by spawned tasks ──────────────
                Some((peer, request, respond)) = self.light_rx.recv() => {
                    return NetworkEvent::LightRequest { peer, request, respond };
                }

                event = self.swarm.select_next_some() => match event {
                // ── Request-Response ────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::Message { peer, message },
                )) => match message {
                    request_response::Message::Request {
                        request, channel, ..
                    } => {
                        return NetworkEvent::MessageReceived {
                            peer,
                            message: request,
                            channel: Some(channel),
                        };
                    }
                    request_response::Message::Response {
                        request_id,
                        response,
                    } => {
                        self.pending_requests.remove(&request_id);
                        return NetworkEvent::MessageReceived {
                            peer,
                            message: response,
                            channel: None,
                        };
                    }
                },
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::OutboundFailure {
                        peer,
                        request_id,
                        error,
                    },
                )) => {
                    self.pending_requests.remove(&request_id);
                    tracing::warn!("Outbound request to {} failed: {}", peer, error);
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::InboundFailure { peer, error, .. },
                )) => {
                    tracing::warn!("Inbound request from {} failed: {}", peer, error);
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Rr(
                    request_response::Event::ResponseSent { .. },
                )) => {}

                // ── Identify ────────────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. },
                )) => {
                    for addr in &info.listen_addrs {
                        self.swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, addr.clone());
                    }
                    self.swarm
                        .behaviour_mut()
                        .autonat
                        .add_server(peer_id, Some(info.observed_addr.clone()));
                }

                // ── libp2p_stream produces no swarm events for light ─────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Light(_)) => {}


                // ── AutoNAT ─────────────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Autonat(
                    autonat::Event::StatusChanged { old, new },
                )) => {
                    tracing::info!("AutoNAT status: {:?} → {:?}", old, new);
                    match new {
                        autonat::NatStatus::Public(_addr) => {
                            self.nat_status = NatStatus::Public;
                            self.swarm
                                .behaviour_mut()
                                .kademlia
                                .set_mode(Some(kad::Mode::Server));
                            tracing::info!(
                                "Node is PUBLIC — Kademlia server mode, serving as relay"
                            );
                        }
                        autonat::NatStatus::Private => {
                            self.nat_status = NatStatus::Private;
                            self.swarm
                                .behaviour_mut()
                                .kademlia
                                .set_mode(Some(kad::Mode::Client));
                            tracing::info!(
                                "Node is behind NAT — using relays and hole-punching"
                            );
                        }
                        autonat::NatStatus::Unknown => {
                            self.nat_status = NatStatus::Unknown;
                        }
                    }
                }

                // ── Relay client ────────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::RelayClient(
                    relay::client::Event::ReservationReqAccepted {
                        relay_peer_id, ..
                    },
                )) => {
                    self.relay_reservations.insert(relay_peer_id);
                    tracing::info!(
                        "Relay reservation accepted by {} (total: {})",
                        relay_peer_id,
                        self.relay_reservations.len()
                    );
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::RelayClient(event)) => {
                    tracing::debug!("Relay client event: {:?}", event);
                }

                // ── DCUtR (hole-punch) ──────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Dcutr(event)) => {
                    tracing::info!("DCUtR event: {:?}", event);
                }

                // ── Relay server ────────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::RelayServer(event)) => {
                    tracing::debug!("Relay server event: {:?}", event);
                }

                // ── Kademlia ────────────────────────────────────────
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Kademlia(
                    kad::Event::RoutingUpdated { peer, .. },
                )) => {
                    tracing::debug!("Kademlia routing updated: {}", peer);
                }
                SwarmEvent::Behaviour(MidstateBehaviourEvent::Kademlia(_)) => {}

                // ── Catch-all ───────────────────────────────────────
                SwarmEvent::Behaviour(_) => {}

                // ── Connection lifecycle ─────────────────────────────
                SwarmEvent::ConnectionEstablished { peer_id, endpoint, num_established, .. } => {
                    // --- Ignore closures of self-connections ---
                    if peer_id == *self.swarm.local_peer_id() {
                        tracing::debug!("Ignoring self-connection");
                        let _ = self.swarm.disconnect_peer_id(peer_id);
                        continue;
                    }

                    // --- DEDUPLICATION GATE ---
                    // If we already have 1 or more connections to this PeerID, 
                    // ignore this event. This prevents "flapping" when a peer 
                    // connects via multiple protocols (QUIC + TCP + WebRTC).
                    if num_established.get() > 1 {
                        tracing::debug!("Redundant connection established from {}; ignoring event.", peer_id);
                        continue;
                    }
                    // --------------------------

                    let remote_addr = endpoint.get_remote_address();
                    let is_webrtc = remote_addr.to_string().contains("webrtc-direct");

                    // --- Light peer cap ---
                    if is_webrtc && self.light_peers.len() >= MAX_LIGHT_PEERS {
                        tracing::warn!("Max light peers ({}) reached, dropping {}", MAX_LIGHT_PEERS, peer_id);
                        let _ = self.swarm.disconnect_peer_id(peer_id);
                        continue;
                    }

                    // --- Check if this light peer is banned ---
                    if is_webrtc && self.light_guard.is_banned(&peer_id).await {
                        tracing::warn!("Banned light peer {} reconnected, dropping", peer_id);
                        let _ = self.swarm.disconnect_peer_id(peer_id);
                        continue;
                    }

                    // --- Subnet Limit Defense ---
                    if let Some(subnet) = extract_subnet(remote_addr) {
                        let peers = self.subnet_peers.entry(subnet).or_default();
                        if !peers.contains(&peer_id) {
                            if peers.len() >= 4 { // Max 4 distinct peers per subnet
                                tracing::warn!("Eclipse Defense: Rejecting {}, subnet {} limit reached", peer_id, subnet);
                                let _ = self.swarm.disconnect_peer_id(peer_id);
                                continue;
                            }
                            peers.insert(peer_id);
                        }
                    }

                    // Eclipse Protection: Enforce inbound limit manually
                    if endpoint.is_listener() {
                        let inbound_count = self.connected.values().filter(|e| e.is_listener()).count();
                        if inbound_count >= 40 {
                            tracing::warn!("Max inbound peers (40) reached, dropping {}", peer_id);
                            let _ = self.swarm.disconnect_peer_id(peer_id);
                            continue; 
                        }
                    }

                    // Track WebRTC peers separately (don't add to Kademlia)
                    if is_webrtc {
                        self.light_peers.insert(peer_id);
                    } else {
                        self.swarm.behaviour_mut().kademlia.add_address(&peer_id, remote_addr.clone());
                    }

                    self.connected.insert(peer_id, endpoint.clone());
                    tracing::info!(
                        "Peer connected: {} via {:?} (total: {}, light: {})",
                        peer_id,
                        remote_addr,
                        self.connected.len(),
                        self.light_peers.len()
                    );
                    
                    // --- BAYESIAN ECLIPSE DEFENSE ---
                    // Construct the full PEX address string to pass to the node
                    let remote_str = remote_addr.to_string();
                    let full_addr = if remote_str.contains("/p2p/") {
                        remote_str
                    } else {
                        remote_addr.clone().with(libp2p::multiaddr::Protocol::P2p(peer_id)).to_string()
                    };
                    
                    return NetworkEvent::PeerConnected(peer_id, full_addr);
                }
                SwarmEvent::ConnectionClosed {
                    peer_id,
                    endpoint,
                    num_established,
                    ..
                } => {
                    // --- Ignore closures of self-connections ---
                    if peer_id == *self.swarm.local_peer_id() {
                        continue;
                    }
                    // ------------------------------------------------
                    
                    if num_established == 0 {
                        self.connected.remove(&peer_id);

                        // --- Light peer cleanup ---
                        if self.light_peers.remove(&peer_id) {
                            self.light_guard.remove_peer(&peer_id).await;
                        }
                        
                        // --- Subnet Limit Defense Cleanup ---
                        if let Some(subnet) = extract_subnet(endpoint.get_remote_address()) {
                            if let std::collections::hash_map::Entry::Occupied(mut entry) = self.subnet_peers.entry(subnet) {
                                entry.get_mut().remove(&peer_id);
                                if entry.get().is_empty() {
                                    entry.remove();
                                }
                            }
                        }
                        // -----------------------------------------

                        tracing::info!(
                            "Peer disconnected: {} (total: {}, light: {})",
                            peer_id,
                            self.connected.len(),
                            self.light_peers.len()
                        );
                        return NetworkEvent::PeerDisconnected(peer_id);
                    }
                }
                SwarmEvent::NewListenAddr { address, .. } => {
                    tracing::info!("Listening on {}", address);
                    self.listen_addrs.push(address);
                }
                SwarmEvent::OutgoingConnectionError { error, .. } => {
                    // Extract the failed multiaddr based on the type of dial error
                    if let libp2p::swarm::DialError::Transport(failed_addrs) = error {
                        // Just grab the first failed address from the transport error
                        if let Some((multiaddr, _)) = failed_addrs.first() {
                            return NetworkEvent::OutgoingConnectionFailed(multiaddr.to_string());
                        }
                    } else if let libp2p::swarm::DialError::WrongPeerId { endpoint, .. } = error {
                        // The peer answered, but lied about its PeerId (malicious/misconfigured)
                        return NetworkEvent::OutgoingConnectionFailed(endpoint.get_remote_address().to_string());
                    }
                }
                SwarmEvent::ExternalAddrConfirmed { address } => {
                    tracing::info!("External address confirmed: {}", address);
                    if !self.external_addrs.contains(&address) {
                        self.external_addrs.push(address);
                    }
                }
                _ => {}
            } // match event
            } // tokio::select!
        } // loop
    }
    
    pub fn outbound_peer_count(&self) -> usize {
        self.connected.values().filter(|e| e.is_dialer()).count()
    }
    // respond_light is superseded: use the oneshot sender in NetworkEvent::LightRequest.
    // Kept as a stub so any stale call site produces a compile error rather than
    // a silent behaviour change.
    #[deprecated(note = "Send the response via the `respond` oneshot in NetworkEvent::LightRequest")]
    pub fn respond_light(&mut self, _resp: LightResponse) {
        unimplemented!("use NetworkEvent::LightRequest::respond oneshot")
    }

    /// Garbage-collect stale light client rate-limiter entries.
    /// Call periodically (e.g. every 60s) to prevent unbounded growth.
    pub async fn gc_stale_light_peers(&self) {
        self.light_guard.gc_stale().await;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

pub fn socket_to_multiaddr(addr: SocketAddr) -> Multiaddr {
    use std::net::IpAddr;
    let mut ma = Multiaddr::empty();
    match addr.ip() {
        IpAddr::V4(ip) => ma.push(libp2p::multiaddr::Protocol::Ip4(ip)),
        IpAddr::V6(ip) => ma.push(libp2p::multiaddr::Protocol::Ip6(ip)),
    }
    ma.push(libp2p::multiaddr::Protocol::Tcp(addr.port()));
    ma
}

/// Convert a TCP multiaddr to its WebRTC-direct equivalent.
/// /ip4/0.0.0.0/tcp/9333 → /ip4/0.0.0.0/udp/9091/webrtc-direct
fn tcp_to_webrtc(addr: &Multiaddr) -> Option<Multiaddr> {
    let mut components = addr.iter().collect::<Vec<_>>();
    let tcp_idx = components.iter().position(|p| matches!(p, libp2p::multiaddr::Protocol::Tcp(_)))?;
    let port = match components[tcp_idx] {
        libp2p::multiaddr::Protocol::Tcp(p) => p,
        _ => return None,
    };
    // Use port + 2 to avoid colliding with QUIC (port + 0)
    let webrtc_port = port.checked_add(2).unwrap_or(port);
    components[tcp_idx] = libp2p::multiaddr::Protocol::Udp(webrtc_port);
    components.insert(tcp_idx + 1, libp2p::multiaddr::Protocol::WebRTCDirect);
    let mut ma = Multiaddr::empty();
    for c in components {
        ma.push(c);
    }
    Some(ma)
}

/// Convert a TCP multiaddr to its QUIC-v1 equivalent.
/// /ip4/0.0.0.0/tcp/9333 → /ip4/0.0.0.0/udp/9333/quic-v1
fn tcp_to_quic(addr: &Multiaddr) -> Option<Multiaddr> {
    let mut components = addr.iter().collect::<Vec<_>>();
    let tcp_idx = components.iter().position(|p| matches!(p, libp2p::multiaddr::Protocol::Tcp(_)))?;
    let port = match components[tcp_idx] {
        libp2p::multiaddr::Protocol::Tcp(p) => p,
        _ => return None,
    };
    components[tcp_idx] = libp2p::multiaddr::Protocol::Udp(port);
    components.insert(tcp_idx + 1, libp2p::multiaddr::Protocol::QuicV1);
    let mut ma = Multiaddr::empty();
    for c in components {
        ma.push(c);
    }
    Some(ma)
}

/// Extract PeerId from a multiaddr like /ip4/.../tcp/.../p2p/<peer_id>
fn extract_peer_id(addr: &Multiaddr) -> Option<PeerId> {
    addr.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::P2p(peer_id) => Some(peer_id),
        _ => None,
    })
}

/// Extracts the /24 (IPv4) or /32 (IPv6) subnet prefix from a multiaddr.
/// Returns None for localhost addresses so local testing isn't restricted.
fn extract_subnet(addr: &Multiaddr) -> Option<IpAddr> {
    if is_localhost(addr) {
        return None;
    }
    // Relayed connections share the relay's IP. Do not rate-limit the relay itself.
    if addr.iter().any(|p| p == libp2p::multiaddr::Protocol::P2pCircuit) {
        return None;
    }
    addr.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => {
            let octets = ip.octets();
            // Mask to /24
            Some(IpAddr::V4(std::net::Ipv4Addr::new(octets[0], octets[1], octets[2], 0)))
        }
        libp2p::multiaddr::Protocol::Ip6(ip) => {
            let segs = ip.segments();
            // Mask to /32: A single /32 represents a standard ISP/LIR allocation.
            // This prevents an attacker with a single leased block from spoofing
            // thousands of independent subnets.
            Some(IpAddr::V6(std::net::Ipv6Addr::new(segs[0], segs[1], 0, 0, 0, 0, 0, 0)))
        }
        _ => None,
    })
}

/// Check if a multiaddr points to localhost/loopback.
fn is_localhost(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => ip.is_loopback(),
        libp2p::multiaddr::Protocol::Ip6(ip) => ip.is_loopback(),
        _ => false,
    })
}

fn extract_ip(addr: &Multiaddr) -> Option<std::net::IpAddr> {
    for proto in addr.iter() {
        match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => return Some(std::net::IpAddr::V4(ip)),
            libp2p::multiaddr::Protocol::Ip6(ip) => return Some(std::net::IpAddr::V6(ip)),
            _ => {}
        }
    }
    None
}

fn replace_ip(addr: &Multiaddr, new_ip: std::net::IpAddr) -> Multiaddr {
    addr.iter().map(|proto| match proto {
        libp2p::multiaddr::Protocol::Ip4(_) => match new_ip {
            std::net::IpAddr::V4(ip) => libp2p::multiaddr::Protocol::Ip4(ip),
            std::net::IpAddr::V6(ip) => libp2p::multiaddr::Protocol::Ip6(ip),
        },
        libp2p::multiaddr::Protocol::Ip6(_) => match new_ip {
            std::net::IpAddr::V4(ip) => libp2p::multiaddr::Protocol::Ip4(ip),
            std::net::IpAddr::V6(ip) => libp2p::multiaddr::Protocol::Ip6(ip),
        },
        other => other,
    }).collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper function tests ───────────────────────────────────────

    #[test]
    fn tcp_to_quic_basic() {
        let tcp: Multiaddr = "/ip4/0.0.0.0/tcp/9333".parse().unwrap();
        let quic = tcp_to_quic(&tcp).unwrap();
        assert_eq!(quic.to_string(), "/ip4/0.0.0.0/udp/9333/quic-v1");
    }

    #[test]
    fn tcp_to_quic_with_ip6() {
        let tcp: Multiaddr = "/ip6/::/tcp/9333".parse().unwrap();
        let quic = tcp_to_quic(&tcp).unwrap();
        assert_eq!(quic.to_string(), "/ip6/::/udp/9333/quic-v1");
    }

    #[test]
    fn tcp_to_quic_no_tcp_returns_none() {
        let udp: Multiaddr = "/ip4/0.0.0.0/udp/9333".parse().unwrap();
        assert!(tcp_to_quic(&udp).is_none());
    }

    #[test]
    fn tcp_to_quic_preserves_peer_id() {
        let tcp: Multiaddr = "/ip4/1.2.3.4/tcp/9333/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN"
            .parse().unwrap();
        let quic = tcp_to_quic(&tcp).unwrap();
        let quic_str = quic.to_string();
        assert!(quic_str.contains("/udp/9333/quic-v1"));
        assert!(quic_str.contains("/p2p/12D3KooW"));
    }

    #[test]
    fn extract_peer_id_present() {
        let addr: Multiaddr = "/ip4/1.2.3.4/tcp/9333/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN"
            .parse().unwrap();
        assert!(extract_peer_id(&addr).is_some());
    }

    #[test]
    fn extract_peer_id_absent() {
        let addr: Multiaddr = "/ip4/1.2.3.4/tcp/9333".parse().unwrap();
        assert!(extract_peer_id(&addr).is_none());
    }

    #[test]
    fn is_localhost_ipv4_loopback() {
        let addr: Multiaddr = "/ip4/127.0.0.1/tcp/9333".parse().unwrap();
        assert!(is_localhost(&addr));
    }

    #[test]
    fn is_localhost_ipv6_loopback() {
        let addr: Multiaddr = "/ip6/::1/tcp/9333".parse().unwrap();
        assert!(is_localhost(&addr));
    }

    #[test]
    fn is_localhost_public_ip() {
        let addr: Multiaddr = "/ip4/203.0.113.10/tcp/9333".parse().unwrap();
        assert!(!is_localhost(&addr));
    }

    #[test]
    fn socket_to_multiaddr_ipv4() {
        let sa: SocketAddr = "1.2.3.4:9333".parse().unwrap();
        let ma = socket_to_multiaddr(sa);
        assert_eq!(ma.to_string(), "/ip4/1.2.3.4/tcp/9333");
    }

    #[test]
    fn socket_to_multiaddr_ipv6() {
        let sa: SocketAddr = "[::1]:9333".parse().unwrap();
        let ma = socket_to_multiaddr(sa);
        assert_eq!(ma.to_string(), "/ip6/::1/tcp/9333");
    }

    // ── NAT status tests ────────────────────────────────────────────

    #[test]
    fn nat_status_default() {
        // Can't construct MidstateNetwork without async, but we can
        // test the enum directly
        let status = NatStatus::Unknown;
        assert_eq!(status, NatStatus::Unknown);
        assert_ne!(status, NatStatus::Public);
        assert_ne!(status, NatStatus::Private);
    }

    // ── PEX constant tests ──────────────────────────────────────────

    #[test]
    fn max_pex_addrs_is_reasonable() {
        // Sanity: the limit should be between 10 and 1000
        assert!(MAX_PEX_ADDRS >= 10);
        assert!(MAX_PEX_ADDRS <= 1000);
    }
}
