use super::protocol::{Message, PROTOCOL_VERSION};
use super::discovery::AddressBook;
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

pub type PeerIndex = u64;

// Peer limits
pub const MAX_INBOUND: usize = 32;
pub const MAX_OUTBOUND: usize = 8;
pub const MAX_PEERS: usize = 40;
pub const MAX_GETBATCHES_COUNT: u64 = 100;

// Rate limiting
const RATE_WINDOW: Duration = Duration::from_secs(60);
const RATE_MAX_MSGS: u64 = 500;

pub struct PeerConnection {
    index: Option<PeerIndex>,
    addr: SocketAddr,
    writer: Option<WriteHalf<TcpStream>>,
    msg_rx: Option<mpsc::UnboundedReceiver<Result<Message>>>,
    last_ping: SystemTime,
    last_pong: SystemTime,
    handshake_complete: bool,
    inbound: bool,
    // Rate limiting
    msg_count: u64,
    rate_window_start: Instant,
}

impl PeerConnection {
    pub async fn connect(addr: SocketAddr, _our_addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        tracing::info!("Connected to peer: {}", addr);

        let (reader, writer) = tokio::io::split(stream);
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        tokio::spawn(Self::read_loop(reader, msg_tx));

        Ok(Self {
            index: None,
            addr,
            writer: Some(writer),
            msg_rx: Some(msg_rx),
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
            inbound: false,
            msg_count: 0,
            rate_window_start: Instant::now(),
        })
    }

    pub fn from_stream(stream: TcpStream, addr: SocketAddr) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        tokio::spawn(Self::read_loop(reader, msg_tx));

        Self {
            index: None,
            addr,
            writer: Some(writer),
            msg_rx: Some(msg_rx),
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
            inbound: true,
            msg_count: 0,
            rate_window_start: Instant::now(),
        }
    }

    async fn read_loop(mut reader: ReadHalf<TcpStream>, tx: mpsc::UnboundedSender<Result<Message>>) {
        loop {
            let mut len_bytes = [0u8; 4];
            if let Err(e) = reader.read_exact(&mut len_bytes).await {
                let _ = tx.send(Err(e.into()));
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;

            if len > 10_000_000 {
                let _ = tx.send(Err(anyhow::anyhow!("Message too large: {} bytes", len)));
                break;
            }

            let mut msg_bytes = vec![0u8; len];
            if let Err(e) = reader.read_exact(&mut msg_bytes).await {
                let _ = tx.send(Err(e.into()));
                break;
            }

            match Message::deserialize(&msg_bytes) {
                Ok(msg) => {
                    if tx.send(Ok(msg)).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        }
    }

    pub async fn complete_handshake(&mut self, our_addr: SocketAddr) -> Result<()> {
        if self.handshake_complete {
            return Ok(());
        }

        let version = Message::Version {
            version: PROTOCOL_VERSION,
            services: 1,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs(),
            addr_recv: self.addr,
            addr_from: our_addr,
        };

        self.send_message(&version).await?;

        let msg = self.receive_message().await?;
        match msg {
            Message::Version { version, .. } => {
                if version != PROTOCOL_VERSION {
                    bail!("Protocol version mismatch");
                }
                self.send_message(&Message::Verack).await?;

                let msg2 = self.receive_message().await?;
                match msg2 {
                    Message::Verack => {
                        self.handshake_complete = true;
                        tracing::info!("Handshake complete with {}", self.addr);
                        Ok(())
                    }
                    _ => bail!("Expected Verack, got {:?}", msg2),
                }
            }
            _ => bail!("Expected Version, got {:?}", msg),
        }
    }

    pub async fn send_message(&mut self, msg: &Message) -> Result<()> {
        let writer = self.writer.as_mut().ok_or_else(|| anyhow::anyhow!("Not connected"))?;
        let bytes = msg.serialize();
        let len = bytes.len() as u32;
        writer.write_all(&len.to_le_bytes()).await?;
        writer.write_all(&bytes).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Receive a message directly (used during handshake and standalone sync).
    /// After the peer is registered with PeerManager, messages arrive via the shared channel instead.
    pub async fn receive_message(&mut self) -> Result<Message> {
        let rx = self.msg_rx.as_mut().ok_or_else(|| anyhow::anyhow!("msg_rx taken"))?;
        match rx.recv().await {
            Some(Ok(msg)) => Ok(msg),
            Some(Err(e)) => Err(e),
            None => Err(anyhow::anyhow!("Connection closed")),
        }
    }

    /// Take the message receiver so PeerManager can forward it to the shared channel.
    pub fn take_msg_rx(&mut self) -> Option<mpsc::UnboundedReceiver<Result<Message>>> {
        self.msg_rx.take()
    }

    pub async fn send_ping(&mut self) -> Result<()> {
        let nonce: u64 = rand::random();
        self.send_message(&Message::Ping { nonce }).await?;
        self.last_ping = SystemTime::now();
        Ok(())
    }

    pub fn handle_pong(&mut self) {
        self.last_pong = SystemTime::now();
    }

    pub fn is_alive(&self) -> bool {
        SystemTime::now()
            .duration_since(self.last_pong)
            .map(|d| d < Duration::from_secs(60))
            .unwrap_or(false)
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn is_connected(&self) -> bool {
        self.writer.is_some() && self.handshake_complete
    }

    pub fn disconnect(&mut self) {
        self.writer = None;
        self.handshake_complete = false;
    }

    /// Record an incoming message for rate limiting. Returns false if limit exceeded.
    pub fn record_message(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.rate_window_start) > RATE_WINDOW {
            self.msg_count = 0;
            self.rate_window_start = now;
        }
        self.msg_count += 1;
        self.msg_count <= RATE_MAX_MSGS
    }
}

/// Forward messages from a peer's local channel to the shared channel.
async fn forward_messages(
    idx: PeerIndex,
    mut local_rx: mpsc::UnboundedReceiver<Result<Message>>,
    shared_tx: mpsc::UnboundedSender<(PeerIndex, Result<Message>)>,
) {
    loop {
        match local_rx.recv().await {
            Some(Ok(msg)) => {
                if shared_tx.send((idx, Ok(msg))).is_err() {
                    return;
                }
            }
            Some(Err(e)) => {
                let _ = shared_tx.send((idx, Err(e)));
                return;
            }
            None => break,
        }
    }
    let _ = shared_tx.send((idx, Err(anyhow::anyhow!("peer reader closed"))));
}

pub struct PeerManager {
    peers: HashMap<PeerIndex, PeerConnection>,
    next_index: PeerIndex,
    our_addr: SocketAddr,
    peer_msg_tx: mpsc::UnboundedSender<(PeerIndex, Result<Message>)>,
    address_book: AddressBook,
    address_book_path: Option<PathBuf>,
    inbound_count: usize,
    outbound_count: usize,
}

impl PeerManager {
    pub fn new(our_addr: SocketAddr) -> (Self, mpsc::UnboundedReceiver<(PeerIndex, Result<Message>)>) {
        let (peer_msg_tx, peer_msg_rx) = mpsc::unbounded_channel();
        let mgr = Self {
            peers: HashMap::new(),
            next_index: 0,
            our_addr,
            peer_msg_tx,
            address_book: AddressBook::new(),
            address_book_path: None,
            inbound_count: 0,
            outbound_count: 0,
        };
        (mgr, peer_msg_rx)
    }

    /// Create a PeerManager that persists its address book to `path`.
    pub fn with_persistence(
        our_addr: SocketAddr,
        peers_path: PathBuf,
    ) -> (Self, mpsc::UnboundedReceiver<(PeerIndex, Result<Message>)>) {
        let (peer_msg_tx, peer_msg_rx) = mpsc::unbounded_channel();
        let address_book = AddressBook::load(&peers_path);
        let mgr = Self {
            peers: HashMap::new(),
            next_index: 0,
            our_addr,
            peer_msg_tx,
            address_book,
            address_book_path: Some(peers_path),
            inbound_count: 0,
            outbound_count: 0,
        };
        (mgr, peer_msg_rx)
    }

    /// Flush the address book to disk (call periodically or on shutdown).
    pub fn save_address_book(&self) {
        if let Some(ref path) = self.address_book_path {
            if let Err(e) = self.address_book.save(path) {
                tracing::warn!("Failed to save address book: {}", e);
            }
        }
    }

    pub async fn connect_to_peer(&mut self, addr: SocketAddr) -> Result<PeerIndex> {
        if self.outbound_count >= MAX_OUTBOUND {
            bail!("outbound peer limit reached ({}/{})", self.outbound_count, MAX_OUTBOUND);
        }
        if self.peers.len() >= MAX_PEERS {
            bail!("total peer limit reached ({}/{})", self.peers.len(), MAX_PEERS);
        }
        if self.address_book.is_banned(addr) {
            bail!("peer {} is banned", addr);
        }

        let mut peer = PeerConnection::connect(addr, self.our_addr).await?;
        peer.complete_handshake(self.our_addr).await?;
        peer.inbound = false;

        let idx = self.register_peer(peer);
        self.outbound_count += 1;
        self.address_book.mark_connected(addr);
        self.address_book.mark_tried(addr);
        tracing::info!("Connected to outbound peer {}: {} (total: {})", idx, addr, self.peers.len());
        Ok(idx)
    }

    /// Add an already-handshaked inbound peer.
    pub fn add_inbound_peer(&mut self, peer: PeerConnection) -> Result<PeerIndex> {
        if self.inbound_count >= MAX_INBOUND {
            bail!("inbound peer limit reached ({}/{})", self.inbound_count, MAX_INBOUND);
        }
        if self.peers.len() >= MAX_PEERS {
            bail!("total peer limit reached ({}/{})", self.peers.len(), MAX_PEERS);
        }
        let addr = peer.addr();
        if self.address_book.is_banned(addr) {
            bail!("peer {} is banned", addr);
        }

        let mut peer = peer;
        peer.inbound = true;
        let idx = self.register_peer(peer);
        self.inbound_count += 1;
        self.address_book.mark_connected(addr);
        tracing::info!("Added inbound peer {}: {} (total: {})", idx, addr, self.peers.len());
        Ok(idx)
    }

    /// Assign an index, take the local msg_rx, start forwarding to shared channel.
    fn register_peer(&mut self, mut peer: PeerConnection) -> PeerIndex {
        let idx = self.next_index;
        self.next_index += 1;
        peer.index = Some(idx);

        if let Some(local_rx) = peer.take_msg_rx() {
            let shared_tx = self.peer_msg_tx.clone();
            tokio::spawn(forward_messages(idx, local_rx, shared_tx));
        }

        self.peers.insert(idx, peer);
        idx
    }

    /// Send a message to a specific peer (unicast).
    pub async fn send_to(&mut self, idx: PeerIndex, msg: &Message) {
        if let Some(peer) = self.peers.get_mut(&idx) {
            if let Err(e) = peer.send_message(msg).await {
                tracing::warn!("Failed to send to peer {} ({}): {}", idx, peer.addr(), e);
                peer.disconnect();
            }
        }
    }

    /// Broadcast to all connected peers.
    pub async fn broadcast(&mut self, msg: &Message) {
        self.broadcast_except(None, msg).await;
    }

    /// Broadcast to all connected peers except `exclude`.
    pub async fn broadcast_except(&mut self, exclude: Option<PeerIndex>, msg: &Message) {
        tracing::warn!("DEBUG: broadcast_except called, peers={}, connected={}", 
        self.peers.len(),
        self.peers.values().filter(|p| p.is_connected()).count());
        
        let mut dead = Vec::new();

        for (&idx, peer) in self.peers.iter_mut() {
            if Some(idx) == exclude {
                tracing::warn!("DEBUG: Skipping peer {} (excluded)", idx);
                continue;
            }
            if !peer.is_connected() {
                tracing::warn!("DEBUG: Skipping peer {} (not connected): writer={} handshake={}", 
                    idx, peer.writer.is_some(), peer.handshake_complete);
                continue;
            }
            
            tracing::warn!("DEBUG: Sending to peer {}", idx);
            if let Err(e) = peer.send_message(msg).await {
                tracing::warn!("DEBUG: Send failed to peer {}: {}", idx, e);
                peer.disconnect();
                dead.push(idx);
            } else {
                tracing::warn!("DEBUG: Successfully sent to peer {}", idx);
            }
        }

        for idx in dead {
            self.cleanup_peer(idx);
        }
    }

    pub async fn send_pings(&mut self) {
        let mut dead = Vec::new();
        for (&idx, peer) in self.peers.iter_mut() {
            if peer.is_connected() {
                if let Err(_) = peer.send_ping().await {
                    dead.push(idx);
                }
            }
        }
        for idx in dead {
            if let Some(peer) = self.peers.get_mut(&idx) {
                peer.disconnect();
            }
            self.cleanup_peer(idx);
        }
    }

    pub fn remove_dead_peers(&mut self) {
        let dead: Vec<PeerIndex> = self.peers.iter()
            .filter(|(_, p)| !p.is_alive())
            .map(|(&idx, _)| idx)
            .collect();

        for idx in dead {
            tracing::info!("Removing dead peer {}", idx);
            self.cleanup_peer(idx);
        }
    }

    /// Remove a peer by index (e.g. on disconnect or error).
    pub fn remove_peer(&mut self, idx: PeerIndex) {
        if let Some(mut peer) = self.peers.remove(&idx) {
            tracing::info!("Removed peer {} ({})", idx, peer.addr());
            let addr = peer.addr();
            peer.disconnect();
            self.address_book.mark_disconnected(addr);
            if peer.inbound {
                self.inbound_count = self.inbound_count.saturating_sub(1);
            } else {
                self.outbound_count = self.outbound_count.saturating_sub(1);
            }
        }
    }

    fn cleanup_peer(&mut self, idx: PeerIndex) {
        self.remove_peer(idx);
    }

    /// Check rate limit for a peer. Returns false if exceeded.
    pub fn check_rate(&mut self, idx: PeerIndex) -> bool {
        match self.peers.get_mut(&idx) {
            Some(peer) => peer.record_message(),
            None => true,
        }
    }

    /// Ban a peer: remove and add to address book ban list.
    pub fn ban_peer(&mut self, idx: PeerIndex) {
        if let Some(peer) = self.peers.get(&idx) {
            let addr = peer.addr();
            tracing::warn!("Banning peer {} ({})", idx, addr);
            self.address_book.ban_peer(addr);
        }
        self.remove_peer(idx);
    }

    /// Record misbehavior. Bans at threshold.
    pub fn record_misbehavior(&mut self, idx: PeerIndex, score: u32) {
        if let Some(peer) = self.peers.get(&idx) {
            let addr = peer.addr();
            self.address_book.mark_misbehavior(addr, score);
            if self.address_book.is_banned(addr) {
                self.remove_peer(idx);
            }
        }
    }

    pub fn peer_addrs(&self) -> Vec<SocketAddr> {
        self.peers.values().map(|p| p.addr()).collect()
    }

    pub fn handle_pong(&mut self, idx: PeerIndex) {
        if let Some(peer) = self.peers.get_mut(&idx) {
            peer.handle_pong();
        }
    }

    pub fn connected_count(&self) -> usize {
        self.peers.values().filter(|p| p.is_connected()).count()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Return a random connected peer index (for requesting missing batches).
    pub fn random_peer(&self) -> Option<PeerIndex> {
        use rand::seq::IteratorRandom;
        self.peers.iter()
            .filter(|(_, p)| p.is_connected())
            .map(|(&idx, _)| idx)
            .choose(&mut rand::thread_rng())
    }
}
