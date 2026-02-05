use super::protocol::{Message, PROTOCOL_VERSION};
use anyhow::Result;
use std::net::SocketAddr;
use std::time::{Duration, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time;

pub struct PeerConnection {
    addr: SocketAddr,
    stream: Option<TcpStream>,
    reconnect_attempts: u32,
    last_ping: SystemTime,
    last_pong: SystemTime,
    handshake_complete: bool,
}

impl PeerConnection {
    pub async fn connect(addr: SocketAddr, our_addr: SocketAddr) -> Result<Self> {
        let mut stream = TcpStream::connect(addr).await?;
        tracing::info!("Connected to peer: {}", addr);
        
        let version = Message::Version {
            version: PROTOCOL_VERSION,
            services: 1,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs(),
            addr_recv: addr,
            addr_from: our_addr,
        };
        
        Self::send_msg(&mut stream, &version).await?;
        
        Ok(Self {
            addr,
            stream: Some(stream),
            reconnect_attempts: 0,
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
        })
    }
    
    pub fn from_stream(stream: TcpStream, addr: SocketAddr) -> Self {
        Self {
            addr,
            stream: Some(stream),
            reconnect_attempts: 0,
            last_ping: SystemTime::now(),
            last_pong: SystemTime::now(),
            handshake_complete: false,
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
        
        // Send version first
        self.send_message(&version).await?;
        
        // Then receive their version
        match self.receive_message().await? {
            Message::Version { version, .. } => {
                if version != PROTOCOL_VERSION {
                    anyhow::bail!("Protocol version mismatch");
                }
                
                // Send verack
                self.send_message(&Message::Verack).await?;
                
                // Wait for their verack
                match self.receive_message().await? {
                    Message::Verack => {
                        self.handshake_complete = true;
                        tracing::info!("Handshake complete with {}", self.addr);
                        Ok(())
                    }
                    _ => anyhow::bail!("Expected Verack"),
                }
            }
            _ => anyhow::bail!("Expected Version"),
        }
    }
    
    pub async fn reconnect(&mut self, our_addr: SocketAddr) -> Result<()> {
        if self.reconnect_attempts >= 5 {
            anyhow::bail!("Max reconnection attempts reached");
        }
        
        let backoff = Duration::from_secs(2u64.pow(self.reconnect_attempts));
        tracing::info!("Reconnecting to {} in {:?}", self.addr, backoff);
        
        time::sleep(backoff).await;
        
        match Self::connect(self.addr, our_addr).await {
            Ok(new_conn) => {
                *self = new_conn;
                tracing::info!("Reconnected to {}", self.addr);
                Ok(())
            }
            Err(e) => {
                self.reconnect_attempts += 1;
                Err(e)
            }
        }
    }
    
    async fn send_msg(stream: &mut TcpStream, msg: &Message) -> Result<()> {
        let bytes = msg.serialize();
        let len = bytes.len() as u32;
        
        stream.write_all(&len.to_le_bytes()).await?;
        stream.write_all(&bytes).await?;
        stream.flush().await?;
        
        Ok(())
    }
    
    pub async fn send_message(&mut self, msg: &Message) -> Result<()> {
        let stream = self.stream.as_mut().ok_or_else(|| anyhow::anyhow!("Not connected"))?;
        Self::send_msg(stream, msg).await
    }
    
    pub async fn receive_message(&mut self) -> Result<Message> {
        let stream = self.stream.as_mut().ok_or_else(|| anyhow::anyhow!("Not connected"))?;
        
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes).await?;
        let len = u32::from_le_bytes(len_bytes) as usize;
        
        if len > 10_000_000 {
            anyhow::bail!("Message too large: {} bytes", len);
        }
        
        let mut msg_bytes = vec![0u8; len];
        stream.read_exact(&mut msg_bytes).await?;
        
        Message::deserialize(&msg_bytes)
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
        self.stream.is_some() && self.handshake_complete
    }
    
    pub fn disconnect(&mut self) {
        self.stream = None;
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
        
        match peer.receive_message().await? {
            Message::Verack => {
                peer.handshake_complete = true;
                self.peers.push(peer);
                Ok(())
            }
            _ => anyhow::bail!("Handshake failed"),
        }
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
