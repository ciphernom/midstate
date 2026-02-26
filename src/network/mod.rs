pub mod protocol;

pub use protocol::{Message, MidstateCodec, MIDSTATE_PROTOCOL, MAX_GETBATCHES_COUNT};

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
    core::ConnectedPoint
};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::Duration;

/// Max addresses to send in a single PEX Addr message.
pub const MAX_PEX_ADDRS: usize = 50;

// ── Behaviour ───────────────────────────────────────────────────────────────

#[derive(NetworkBehaviour)]
pub struct MidstateBehaviour {
    pub rr: request_response::Behaviour<MidstateCodec>,
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub relay_server: relay::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub autonat: autonat::Behaviour,
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
    PeerConnected(PeerId),
    PeerDisconnected(PeerId),
    OutgoingConnectionFailed(String),
}

// ── Network API ─────────────────────────────────────────────────────────────

pub struct MidstateNetwork {
    swarm: Swarm<MidstateBehaviour>,
    connected: HashMap<PeerId, ConnectedPoint>,
    pending_requests: HashMap<OutboundRequestId, PeerId>,
    nat_status: NatStatus,
    relay_reservations: HashSet<PeerId>,
    listen_addrs: Vec<Multiaddr>,
    external_addrs: Vec<Multiaddr>,
}

impl MidstateNetwork {
    pub async fn new(
        keypair: Keypair,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
    ) -> Result<Self> {
        let peer_id = keypair.public().to_peer_id();
        tracing::info!("Local peer id: {}", peer_id);

        let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair.clone())
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_quic()
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay_client| {
                let local_peer = key.public().to_peer_id();

            // --- Increase timeout from 10s to 60s ---
                let rr_config = RequestResponseConfig::default()
                    .with_request_timeout(Duration::from_secs(60));

                let rr = request_response::Behaviour::new(
                    [(MIDSTATE_PROTOCOL, ProtocolSupport::Full)],
                    rr_config,
                );
                // -------------------------------------------------

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

                let relay_server = relay::Behaviour::new(
                    local_peer,
                    relay::Config::default(),
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

                MidstateBehaviour {
                    rr,
                    kademlia,
                    identify,
                    relay_client,
                    relay_server,
                    dcutr,
                    autonat,
                }
            })?
            .with_swarm_config(|c| {
                c.with_idle_connection_timeout(Duration::from_secs(120))
            })
            .build();

        let mut net = Self {
            swarm,
            connected: HashMap::new(),
            pending_requests: HashMap::new(),
            nat_status: NatStatus::Unknown,
            relay_reservations: HashSet::new(),
            listen_addrs: Vec::new(),
            external_addrs: Vec::new(),
        };

        net.swarm.listen_on(listen_addr.clone())?;

        if let Some(quic_addr) = tcp_to_quic(&listen_addr) {
            match net.swarm.listen_on(quic_addr.clone()) {
                Ok(_) => tracing::info!("Also listening on QUIC: {}", quic_addr),
                Err(e) => tracing::debug!("QUIC listen failed (non-fatal): {}", e),
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
        let peers: Vec<PeerId> = self.connected.keys().copied().collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
    }

    pub fn broadcast_except(&mut self, exclude: Option<PeerId>, msg: Message) {
        let peers: Vec<PeerId> = self.connected
            .keys()
            .filter(|&&p| Some(p) != exclude)
            .copied()
            .collect();
        for peer in peers {
            self.send(peer, msg.clone());
        }
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

    pub fn add_kad_address(&mut self, peer: PeerId, addr: Multiaddr) {
        self.swarm.behaviour_mut().kademlia.add_address(&peer, addr);
    }

    pub fn random_peer(&self) -> Option<PeerId> {
        use rand::seq::IteratorRandom;
        self.connected
            .keys()
            .copied()
            .choose(&mut rand::thread_rng())
    }

    // ── PEX (Peer Exchange) ─────────────────────────────────────────

    /// Our own externally-reachable addresses (for advertising to peers).
    /// Prefers confirmed external addrs; falls back to listen addrs.
    pub fn advertisable_addrs(&self) -> Vec<String> {
        let local_id = *self.swarm.local_peer_id();
        let p2p_suffix = libp2p::multiaddr::Protocol::P2p(local_id);

        let base = if !self.external_addrs.is_empty() {
            &self.external_addrs
        } else {
            &self.listen_addrs
        };

        base.iter()
            .filter(|a| !is_localhost(a))
            .map(|a| {
                // Append /p2p/<our_id> if not already present
                if extract_peer_id(a).is_some() {
                    a.to_string()
                } else {
                    a.clone().with(p2p_suffix.clone()).to_string()
                }
            })
            .collect()
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
                    let full = addr.clone()
                        .with(libp2p::multiaddr::Protocol::P2p(*peer));
                    addrs.push(full.to_string());
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
                if let Some(peer) = extract_peer_id(&addr) {
                    // Don't dial if we're already connected, or if it's our own PeerId
                    if self.connected.contains_key(&peer) || peer == *self.swarm.local_peer_id() {
                        return; 
                    }
                    // Feed into Kademlia
                    self.swarm.behaviour_mut().kademlia.add_address(&peer, addr.clone());
                    
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
            match self.swarm.select_next_some().await {
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
                SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                    // --- Prevent self-connections from poisoning the peer list ---
                    if peer_id == *self.swarm.local_peer_id() {
                        tracing::debug!("Ignoring self-connection");
                        let _ = self.swarm.disconnect_peer_id(peer_id);
                        continue;
                    }
                    // ------------------------------------------------------------------

                    // Eclipse Protection: Enforce inbound limit manually
                    if endpoint.is_listener() {
                        let inbound_count = self.connected.values().filter(|e| e.is_listener()).count();
                        if inbound_count >= 40 {
                            tracing::warn!("Max inbound peers (40) reached, dropping {}", peer_id);
                            let _ = self.swarm.disconnect_peer_id(peer_id);
                            continue; // Skip further processing, let them disconnect
                        }
                    }

                    self.connected.insert(peer_id, endpoint.clone());
                    tracing::info!(
                        "Peer connected: {} via {:?} (total: {})",
                        peer_id,
                        endpoint.get_remote_address(),
                        self.connected.len()
                    );
                    return NetworkEvent::PeerConnected(peer_id);
                }
                SwarmEvent::ConnectionClosed {
                    peer_id,
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
                        tracing::info!(
                            "Peer disconnected: {} (total: {})",
                            peer_id,
                            self.connected.len()
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
            }
        }
    }
    
    pub fn outbound_peer_count(&self) -> usize {
        self.connected.values().filter(|e| e.is_dialer()).count()
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

/// Check if a multiaddr points to localhost/loopback.
fn is_localhost(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => ip.is_loopback(),
        libp2p::multiaddr::Protocol::Ip6(ip) => ip.is_loopback(),
        _ => false,
    })
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
