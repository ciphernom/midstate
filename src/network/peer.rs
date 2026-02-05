// ========================================
// --- FILE: src/network/peer.rs
// ========================================
use super::protocol::{Message, PROTOCOL_VERSION};
use anyhow::Result;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time;

pub struct PeerConnection {
    addr: SocketAddr,
    // We only keep the writer. The reader is moved to a background task.
    writer: Option<WriteHalf<TcpStream>>, 
    // Channel to receive messages from the background reader task
    msg_rx: mpsc::UnboundedReceiver<Result<Message>>,
    reconnect_attempts: u32,
    last_ping: SystemTime,
    last_pong: SystemTime,
    handshake_complete: bool,
}

impl PeerConnection {
    pub async fn connect(addr: SocketAddr, _our_addr: SocketAddr) -> Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        tracing::info!("Connected to peer: {}", addr);
        
        // Spawn the reader task immediately
        let (reader, writer) = tokio::io::split(stream);
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        
        tokio::spawn(Self::read_loop(reader, msg_tx));
        
        Ok(Self {
            addr,
            writer: Some(writer),
            msg_rx,
            reconnect_attempts: 0,
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
        })
    }
    
    pub fn from_stream(stream: TcpStream, addr: SocketAddr) -> Self {
        let (reader, writer) = tokio::io::split(stream);
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        
        tokio::spawn(Self::read_loop(reader, msg_tx));
        
        Self {
            addr,
            writer: Some(writer),
            msg_rx,
            reconnect_attempts: 0,
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
        }
    }
    
    /// Background task to read messages from the stream
    async fn read_loop(mut reader: ReadHalf<TcpStream>, tx: mpsc::UnboundedSender<Result<Message>>) {
        loop {
            let mut len_bytes = [0u8; 4];
            // Read length prefix
            if let Err(e) = reader.read_exact(&mut len_bytes).await {
                // If stream closed (unexpected EOF) or error, send error and exit
                let _ = tx.send(Err(e.into()));
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            
            if len > 10_000_000 {
                let _ = tx.send(Err(anyhow::anyhow!("Message too large: {} bytes", len)));
                break;
            }
            
            let mut msg_bytes = vec![0u8; len];
            // Read body
            if let Err(e) = reader.read_exact(&mut msg_bytes).await {
                let _ = tx.send(Err(e.into()));
                break;
            }
            
            // Deserialize
            match Message::deserialize(&msg_bytes) {
                Ok(msg) => {
                    if tx.send(Ok(msg)).is_err() {
                        break; // Receiver dropped
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
        
        tracing::info!("Starting handshake with {}", self.addr);

        let version = Message::Version {
            version: PROTOCOL_VERSION,
            services: 1,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs(),
            addr_recv: self.addr,
            addr_from: our_addr,
        };
        
        // 1. Send our Version
        self.send_message(&version).await?;
        
        // 2. Receive their Version
        let msg = self.receive_message().await?;
        
        match msg {
            Message::Version { version, .. } => {
                if version != PROTOCOL_VERSION {
                    anyhow::bail!("Protocol version mismatch");
                }
                
                // 3. Send Verack
                self.send_message(&Message::Verack).await?;
                
                // 4. Wait for their Verack
                let msg2 = self.receive_message().await?;
                match msg2 {
                    Message::Verack => {
                        self.handshake_complete = true;
                        tracing::info!("Handshake complete with {}", self.addr);
                        Ok(())
                    }
                    _ => anyhow::bail!("Expected Verack, got {:?}", msg2),
                }
            }
            _ => anyhow::bail!("Expected Version, got {:?}", msg),
        }
    }
    
    pub async fn reconnect(&mut self, our_addr: SocketAddr) -> Result<()> {
        if self.reconnect_attempts >= 5 {
            anyhow::bail!("Max reconnection attempts reached");
        }
        
        let backoff = Duration::from_secs(2u64.pow(self.reconnect_attempts));
        tracing::info!("Reconnecting to {} in {:?}", self.addr, backoff);
        
        time::sleep(backoff).await;
        
        // Re-connect logic needs to reset the reader loop
        let stream = TcpStream::connect(self.addr).await?;
        let (reader, writer) = tokio::io::split(stream);
        let (msg_tx, msg_rx) = mpsc::unbounded_channel();
        
        tokio::spawn(Self::read_loop(reader, msg_tx));
        
        self.writer = Some(writer);
        self.msg_rx = msg_rx;
        self.handshake_complete = false;
        
        tracing::info!("TCP Reconnected to {}", self.addr);
        
        // FIX: Actually perform the handshake!
        // This uses 'our_addr', fixing the warning and the protocol logic.
        self.complete_handshake(our_addr).await?;
        
        Ok(())
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
    
    pub async fn receive_message(&mut self) -> Result<Message> {
        // Safe to cancel: just awaits the channel
        match self.msg_rx.recv().await {
            Some(Ok(msg)) => Ok(msg),
            Some(Err(e)) => Err(e),
            None => Err(anyhow::anyhow!("Connection closed")),
        }
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
        // Dropping msg_rx will eventually close the sender in the background task
        self.handshake_complete = false;
    }
}

pub struct PeerManager {
    peers: Vec<PeerConnection>,
    pub broadcast_tx: mpsc::UnboundedSender<Message>,
    broadcast_rx: mpsc::UnboundedReceiver<Message>,
    our_addr: SocketAddr,
}

impl PeerManager {
    pub fn new(our_addr: SocketAddr) -> Self {
        let (broadcast_tx, broadcast_rx) = mpsc::unbounded_channel();
        Self {
            peers: Vec::new(),
            broadcast_tx,
            broadcast_rx,
            our_addr,
        }
    }
    
    pub async fn connect_to_peer(&mut self, addr: SocketAddr) -> Result<()> {
        let mut peer = PeerConnection::connect(addr, self.our_addr).await?;
        peer.complete_handshake(self.our_addr).await?;
        self.peers.push(peer);
        Ok(())
    }
    
    pub async fn add_peer(&mut self, mut peer: PeerConnection) -> Result<()> {
        peer.complete_handshake(self.our_addr).await?;
        tracing::info!("Added peer: {}", peer.addr());
        self.peers.push(peer);
        Ok(())
    }
    
    pub fn broadcast(&self, msg: Message) {
        let _ = self.broadcast_tx.send(msg);
    }
    
    pub async fn process_broadcasts(&mut self) {
        while let Ok(msg) = self.broadcast_rx.try_recv() {
            let mut disconnected = Vec::new();
            
            for (idx, peer) in self.peers.iter_mut().enumerate() {
                if !peer.is_connected() {
                    if let Err(e) = peer.reconnect(self.our_addr).await {
                        tracing::warn!("Failed to reconnect to {}: {}", peer.addr(), e);
                        disconnected.push(idx);
                        continue;
                    }
                }
                
                if let Err(e) = peer.send_message(&msg).await {
                    tracing::warn!("Failed to send to {}: {}", peer.addr(), e);
                    peer.disconnect();
                }
            }
            
            for idx in disconnected.into_iter().rev() {
                let peer = self.peers.remove(idx);
                tracing::warn!("Removed peer {}", peer.addr());
            }
        }
    }
    
    pub async fn send_pings(&mut self) {
        for peer in &mut self.peers {
            if peer.is_connected() {
                let _ = peer.send_ping().await;
            }
        }
    }
    
    pub fn remove_dead_peers(&mut self) {
        self.peers.retain(|p| p.is_alive());
    }
    
    pub fn peers_mut(&mut self) -> &mut [PeerConnection] {
        &mut self.peers
    }
    
    pub fn peer_addrs(&self) -> Vec<SocketAddr> {
        self.peers.iter().map(|p| p.addr()).collect()
    }
    
    pub fn connected_count(&self) -> usize {
        self.peers.iter().filter(|p| p.is_connected()).count()
    }
}
