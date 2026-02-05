use crate::core::*;
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::mine_extension;
use crate::core::transaction::{apply_transaction, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::network::{Message, PeerConnection, PeerManager};
use crate::storage::Storage;
use anyhow::Result;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time;

pub struct Node {
    state: State,
    mempool: Mempool,
    storage: Storage,
    peer_manager: PeerManager,
    metrics: Metrics,
    is_mining: bool,
    our_addr: SocketAddr,
    recent_states: Vec<State>,
    incoming_peers_rx: tokio::sync::mpsc::UnboundedReceiver<PeerConnection>,
    incoming_peers_tx: tokio::sync::mpsc::UnboundedSender<PeerConnection>,
}

#[derive(Clone)]
pub struct NodeHandle {
    state: Arc<RwLock<State>>,
    mempool_size: Arc<RwLock<usize>>,
    mempool_txs: Arc<RwLock<Vec<Transaction>>>,
    peer_addrs: Arc<RwLock<Vec<SocketAddr>>>,
    tx_sender: tokio::sync::mpsc::UnboundedSender<NodeCommand>,
}

pub enum NodeCommand {
    SendTransaction(Transaction),
}

impl NodeHandle {
    pub async fn get_state(&self) -> State {
        self.state.read().await.clone()
    }

    pub async fn check_coin(&self, coin: [u8; 32]) -> bool {
        self.state.read().await.coins.contains(&coin)
    }

    pub async fn get_mempool_info(&self) -> (usize, Vec<Transaction>) {
        let size = *self.mempool_size.read().await;
        let txs = self.mempool_txs.read().await.clone();
        (size, txs)
    }

    pub async fn get_peers(&self) -> Vec<SocketAddr> {
        self.peer_addrs.read().await.clone()
    }

    pub async fn send_transaction(&self, tx: Transaction) -> Result<()> {
        // Validation Logic Added Here
        let state_guard = self.state.read().await;
        // Validate against current state (rejects bad nonces, insufficient funds, etc.)
        validate_transaction(&state_guard, &tx)?; 
        drop(state_guard);

        self.tx_sender.send(NodeCommand::SendTransaction(tx))?;
        Ok(())
    }
}

impl Node {
    pub fn new(data_dir: PathBuf, is_mining: bool, our_addr: SocketAddr) -> Result<Self> {
        std::fs::create_dir_all(&data_dir)?;

        let storage = Storage::open(data_dir.join("db"))?;
        let state = storage.load_state()?.unwrap_or_else(|| {
            tracing::info!("No saved state, using genesis");
            State::genesis()
        });

        tracing::info!(
            "Loaded state: height={} depth={} coins={} commitments={}",
            state.height,
            state.depth,
            state.coins.len(),
            state.commitments.len()
        );

        let mempool = Mempool::new(data_dir.join("mempool"))?;

        let (incoming_peers_tx, incoming_peers_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            state,
            mempool,
            storage,
            peer_manager: PeerManager::new(our_addr),
            metrics: Metrics::new(),
            is_mining,
            our_addr,
            recent_states: Vec::new(),
            incoming_peers_rx,
            incoming_peers_tx,
        })
    }

    pub fn our_addr(&self) -> SocketAddr {
        self.our_addr
    }

    pub fn create_handle(&self) -> (NodeHandle, tokio::sync::mpsc::UnboundedReceiver<NodeCommand>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        let handle = NodeHandle {
            state: Arc::new(RwLock::new(self.state.clone())),
            mempool_size: Arc::new(RwLock::new(self.mempool.len())),
            mempool_txs: Arc::new(RwLock::new(self.mempool.transactions().to_vec())),
            peer_addrs: Arc::new(RwLock::new(Vec::new())),
            tx_sender: tx,
        };

        (handle, rx)
    }

    pub async fn connect_to_peer(&mut self, addr: SocketAddr) -> Result<()> {
        self.peer_manager.connect_to_peer(addr).await
    }

    pub async fn listen(&mut self, bind_addr: SocketAddr) -> Result<()> {
        let listener = TcpListener::bind(bind_addr).await?;
        tracing::info!("Listening on {}", bind_addr);

        let incoming_tx = self.incoming_peers_tx.clone();
        let our_addr = self.our_addr;

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        tracing::info!("Accepted connection from {}", addr);

                        let incoming_tx = incoming_tx.clone();
                        let our_addr_clone = our_addr;

                        tokio::spawn(async move {
                            let mut peer = PeerConnection::from_stream(stream, addr);

                            // Complete handshake first
                            if let Err(e) = peer.complete_handshake(our_addr_clone).await {
                                tracing::warn!("Handshake failed with {}: {}", addr, e);
                                return;
                            }

                            tracing::info!("Handshake complete with {}", addr);

                            // Send the completed peer connection back to the main loop
                            if incoming_tx.send(peer).is_err() {
                                tracing::warn!("Failed to register peer {}", addr);
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("Accept error: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

pub async fn run(
        mut self,
        handle: NodeHandle,
        mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<NodeCommand>,
    ) -> Result<()> {
        let mut mine_interval = time::interval(Duration::from_secs(5));
        let mut save_interval = time::interval(Duration::from_secs(10));
        let mut ui_interval = time::interval(Duration::from_secs(1)); 
        let mut metrics_interval = time::interval(Duration::from_secs(30));
        let mut peer_maintenance = time::interval(Duration::from_secs(60));
        let mut ping_interval = time::interval(Duration::from_secs(30));

        loop {
            tokio::select! {
                _ = mine_interval.tick() => {
                    if self.is_mining && self.mempool.len() > 0 {
                        if let Err(e) = self.try_mine().await {
                            tracing::error!("Mining error: {}", e);
                        }
                    }
                }

                _ = save_interval.tick() => {
                    if let Err(e) = self.storage.save_state(&self.state) {
                        tracing::error!("Failed to save state: {}", e);
                    }
                }

                _ = ui_interval.tick() => {
                    *handle.state.write().await = self.state.clone();
                    *handle.mempool_size.write().await = self.mempool.len();
                    *handle.mempool_txs.write().await = self.mempool.transactions().to_vec();
                    *handle.peer_addrs.write().await = self.peer_manager.peer_addrs();
                }

                _ = metrics_interval.tick() => {
                    self.metrics.report();
                }

                _ = peer_maintenance.tick() => {
                    self.mempool.prune_invalid(&self.state);
                    self.peer_manager.remove_dead_peers();
                }

                _ = ping_interval.tick() => {
                    self.peer_manager.send_pings().await;
                }

                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        NodeCommand::SendTransaction(tx) => {
                            if let Err(e) = self.handle_new_transaction(tx) {
                                tracing::error!("Failed to handle transaction: {}", e);
                            }
                        }
                    }
                }

                Some(peer) = self.incoming_peers_rx.recv() => {
                    tracing::info!("Adding incoming peer: {}", peer.addr());
                    if let Err(e) = self.peer_manager.add_peer(peer).await {
                        tracing::warn!("Failed to add incoming peer: {}", e);
                    }
                }

                _ = time::sleep(Duration::from_millis(100)) => {
                    if let Err(e) = self.process_peer_messages().await {
                        tracing::error!("Error processing peer messages: {}", e);
                    }
                    self.peer_manager.process_broadcasts().await;
                }
            }
        }
    }

    async fn process_peer_messages(&mut self) -> Result<()> {
        let mut messages = Vec::new();

        for peer in self.peer_manager.peers_mut() {
            if !peer.is_connected() {
                continue;
            }

            tokio::select! {
                msg_result = peer.receive_message() => {
                    match msg_result {
                        Ok(msg) => {
                            messages.push(msg);
                        }
                        Err(e) => {
                            tracing::warn!("Error receiving from {}: {}", peer.addr(), e);
                            peer.disconnect();
                        }
                    }
                }
                _ = time::sleep(Duration::from_millis(10)) => {
                    continue;
                }
            }
        }

        for msg in messages {
            if let Err(e) = self.handle_message(msg).await {
                tracing::warn!("Error handling message: {}", e);
            }
        }

        Ok(())
    }

    async fn handle_message(&mut self, msg: Message) -> Result<()> {
        match msg {
            Message::Transaction(tx) => {
                self.handle_new_transaction(tx)?;
            }

            Message::Batch(batch) => {
                self.handle_new_batch(batch)?;
            }

            Message::GetState => {
                let response = Message::StateInfo {
                    height: self.state.height,
                    depth: self.state.depth,
                    midstate: self.state.midstate,
                };
                self.peer_manager.broadcast(response);
            }

            Message::StateInfo { height, depth, midstate } => {
                tracing::debug!(
                    "Peer state: height={} depth={} midstate={}",
                    height,
                    depth,
                    hex::encode(midstate)
                );
            }

            Message::Ping { nonce } => {
                self.peer_manager.broadcast(Message::Pong { nonce });
            }

            Message::Pong { .. } => {}

            Message::GetAddr => {
                let addrs = self.peer_manager.peer_addrs();
                self.peer_manager.broadcast(Message::Addr(addrs));
            }

            Message::Addr(addrs) => {
                tracing::debug!("Received {} peer addresses", addrs.len());
            }

            Message::Version { .. } | Message::Verack => {}

            Message::GetBatches { start_height, count } => {
                tracing::info!("Peer requesting batches {}-{}", start_height, start_height + count - 1);

                let end = (start_height + count).min(self.state.height);
                let batches = self.storage.load_batches(start_height, end)?;

                self.peer_manager.broadcast(Message::Batches(batches));
            }

            Message::Batches(_batches) => {}
        }

        Ok(())
    }

    fn handle_new_transaction(&mut self, tx: Transaction) -> Result<()> {
        match &tx {
            Transaction::Commit { commitment } => {
                tracing::info!("Received commit: {}", hex::encode(commitment));
            }
            Transaction::Reveal { .. } => {
                let inputs: Vec<String> = tx.input_coins().iter().map(|c| hex::encode(c)).collect();
                tracing::info!("Received reveal: inputs={}", inputs.join(", "));
            }
        }

        match self.mempool.add(tx.clone(), &self.state) {
            Ok(_) => {
                self.metrics.inc_transactions_processed();
                self.peer_manager.broadcast(Message::Transaction(tx));
                Ok(())
            }
            Err(e) => {
                self.metrics.inc_invalid_transactions();
                tracing::warn!("Invalid transaction: {}", e);
                Err(e)
            }
        }
    }

    fn handle_new_batch(&mut self, batch: Batch) -> Result<()> {
        tracing::info!("Received batch with {} txs", batch.transactions.len());

        let mut candidate_state = self.state.clone();
        match apply_batch(&mut candidate_state, &batch) {
            Ok(_) => {
                let best = choose_best_state(&self.state, &candidate_state);

                if best.height > self.state.height {
                    tracing::info!("Accepted batch: new height={}", best.height);

                    if best.midstate != candidate_state.midstate {
                        self.metrics.inc_reorgs();
                        tracing::warn!("Chain reorganization detected!");
                    }

                    self.recent_states.push(self.state.clone());
                    if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                        self.recent_states.remove(0);
                    }

                    self.state = candidate_state;
                    self.storage.save_batch(self.state.height - 1, &batch)?;

                    self.state.target = adjust_difficulty(&self.state, &self.recent_states);

                    self.metrics.inc_batches_processed();
                    self.mempool.prune_invalid(&self.state);
                    self.peer_manager.broadcast(Message::Batch(batch));
                }

                Ok(())
            }
            Err(e) => {
                self.metrics.inc_invalid_batches();
                tracing::warn!("Invalid batch: {}", e);
                Err(e)
            }
        }
    }

    async fn try_mine(&mut self) -> Result<()> {
        tracing::info!("Mining batch with {} transactions...", self.mempool.len());

        let transactions = self.mempool.drain(MAX_BATCH_SIZE);
        let mut candidate_state = self.state.clone();

        for tx in &transactions {
            apply_transaction(&mut candidate_state, tx)?;
        }

        let midstate = candidate_state.midstate;
        let target = self.state.target;

        let extension = tokio::task::spawn_blocking(move || {
            mine_extension(midstate, target)
        })
        .await?;

        let batch = Batch {
            transactions,
            extension,
        };

        self.recent_states.push(self.state.clone());
        if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
            self.recent_states.remove(0);
        }

        apply_batch(&mut self.state, &batch)?;
        self.storage.save_batch(self.state.height - 1, &batch)?;

        self.state.target = adjust_difficulty(&self.state, &self.recent_states);

        self.metrics.inc_batches_mined();
        self.peer_manager.broadcast(Message::Batch(batch));

        tracing::info!("Mined batch! height={} target={}", self.state.height, hex::encode(self.state.target));

        Ok(())
    }
}
