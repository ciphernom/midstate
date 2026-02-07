use crate::core::*;
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::mine_extension;
use crate::core::transaction::{apply_transaction, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::network::{Message, PeerConnection, PeerManager, PeerIndex, MAX_GETBATCHES_COUNT};
use crate::storage::Storage;
use crate::wallet::coinbase_seed;
use anyhow::Result;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time;
use rayon::prelude::*;

const MAX_ORPHAN_BATCHES: usize = 256;

pub struct Node {
    state: State,
    mempool: Mempool,
    storage: Storage,
    peer_manager: PeerManager,
    peer_msg_rx: tokio::sync::mpsc::UnboundedReceiver<(PeerIndex, Result<Message>)>,
    metrics: Metrics,
    is_mining: bool,
    our_addr: SocketAddr,
    recent_states: Vec<State>,
    incoming_peers_rx: tokio::sync::mpsc::UnboundedReceiver<PeerConnection>,
    incoming_peers_tx: tokio::sync::mpsc::UnboundedSender<PeerConnection>,
    orphan_batches: HashMap<u64, Batch>,
    sync_in_progress: bool,
    sync_requested_up_to: u64,
    /// Persistent mining seed for deterministic coinbase keygen.
    mining_seed: [u8; 32],
    /// Data directory for coinbase log.
    data_dir: PathBuf,
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
        let state_guard = self.state.read().await;
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
            state.height, state.depth, state.coins.len(), state.commitments.len()
        );

        // Load or generate mining seed
        let mining_seed = match storage.load_mining_seed()? {
            Some(seed) => {
                tracing::info!("Loaded mining seed");
                seed
            }
            None => {
                let seed: [u8; 32] = rand::random();
                storage.save_mining_seed(&seed)?;
                tracing::info!("Generated new mining seed");
                seed
            }
        };

        let mempool = Mempool::new(data_dir.join("mempool"))?;

        let peers_path = data_dir.join("peers.json");
        let (peer_manager, peer_msg_rx) = PeerManager::with_persistence(our_addr, peers_path);
        let (incoming_peers_tx, incoming_peers_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            state,
            mempool,
            storage,
            peer_manager,
            peer_msg_rx,
            metrics: Metrics::new(),
            is_mining,
            our_addr,
            recent_states: Vec::new(),
            incoming_peers_rx,
            incoming_peers_tx,
            orphan_batches: HashMap::new(),
            sync_in_progress: false,
            sync_requested_up_to: 0,
            mining_seed,
            data_dir,
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
        self.peer_manager.connect_to_peer(addr).await?;
        Ok(())
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
                            if let Err(e) = peer.complete_handshake(our_addr_clone).await {
                                tracing::warn!("Handshake failed with {}: {}", addr, e);
                                return;
                            }
                            let _ = incoming_tx.send(peer);
                        });
                    }
                    Err(e) => tracing::error!("Accept error: {}", e),
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
                if self.is_mining {
                        if let Err(e) = self.try_mine().await {
                            tracing::error!("Mining error: {}", e);
                        }
                    }
                }
                _ = save_interval.tick() => {
                    if let Err(e) = self.storage.save_state(&self.state) {
                        tracing::error!("Failed to save state: {}", e);
                    }
                    self.peer_manager.save_address_book();
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
                            if let Err(e) = self.handle_new_transaction(tx, None).await {
                                tracing::error!("Failed to handle transaction: {}", e);
                            }
                        }
                    }
                }
                Some(peer) = self.incoming_peers_rx.recv() => {
                    if let Err(e) = self.peer_manager.add_inbound_peer(peer) {
                        tracing::warn!("Failed to add incoming peer: {}", e);
                    }
                }
                Some((peer_idx, msg_result)) = self.peer_msg_rx.recv() => {
                    match msg_result {
                        Ok(msg) => {
                            if !self.peer_manager.check_rate(peer_idx) {
                                tracing::warn!("Rate limit exceeded for peer {}", peer_idx);
                                self.peer_manager.ban_peer(peer_idx);
                                continue;
                            }
                            if let Err(e) = self.handle_message(peer_idx, msg).await {
                                tracing::warn!("Error from peer {}: {}", peer_idx, e);
                                self.peer_manager.record_misbehavior(peer_idx, 10);
                            }
                        }
                        Err(e) => {
                            tracing::debug!("Peer {} disconnected: {}", peer_idx, e);
                            self.peer_manager.remove_peer(peer_idx);
                        }
                    }
                }
            }
        }
    }

    async fn handle_message(&mut self, from: PeerIndex, msg: Message) -> Result<()> {
        match msg {
            Message::Transaction(tx) => {
                self.handle_new_transaction(tx, Some(from)).await?;
            }
            Message::Batch(batch) => {
                self.handle_new_batch(batch, Some(from)).await?;
            }
            Message::GetState => {
                let response = Message::StateInfo {
                    height: self.state.height,
                    depth: self.state.depth,
                    midstate: self.state.midstate,
                };
                self.peer_manager.send_to(from, &response).await;
            }
            Message::StateInfo { height, depth, midstate: _ } => {
                tracing::debug!("Peer {} state: height={} depth={}", from, height, depth);
                if height > self.state.height && !self.sync_in_progress {
                    self.request_missing_batches(from, height).await;
                }
            }
            Message::Ping { nonce } => {
                self.peer_manager.send_to(from, &Message::Pong { nonce }).await;
            }
            Message::Pong { .. } => {
                self.peer_manager.handle_pong(from);
            }
            Message::GetAddr => {
                let addrs = self.peer_manager.peer_addrs();
                self.peer_manager.send_to(from, &Message::Addr(addrs)).await;
            }
            Message::Addr(_addrs) => {}
            Message::Version { .. } | Message::Verack => {}
            Message::GetBatches { start_height, count } => {
                let count = count.min(MAX_GETBATCHES_COUNT);
                let end = (start_height + count).min(self.state.height);
                match self.storage.load_batches(start_height, end) {
                    Ok(batches) => {
                        self.peer_manager.send_to(from, &Message::Batches(batches)).await;
                    }
                    Err(e) => tracing::warn!("Failed to load batches: {}", e),
                }
            }
            Message::Batches(batches) => {
                self.handle_batches_response(batches, from).await?;
            }
        }
        Ok(())
    }

    async fn handle_new_transaction(&mut self, tx: Transaction, from: Option<PeerIndex>) -> Result<()> {
        match self.mempool.add(tx.clone(), &self.state) {
            Ok(_) => {
                self.metrics.inc_transactions_processed();
                self.peer_manager.broadcast_except(from, &Message::Transaction(tx)).await;
                Ok(())
            }
            Err(e) => {
                self.metrics.inc_invalid_transactions();
                Err(e)
            }
        }
    }

    async fn handle_new_batch(&mut self, batch: Batch, from: Option<PeerIndex>) -> Result<()> {
        let mut candidate_state = self.state.clone();
        match apply_batch(&mut candidate_state, &batch) {
            Ok(_) => {
                let best = choose_best_state(&self.state, &candidate_state);
                if best.height > self.state.height {
                    tracing::info!("Accepted batch: new height={}", best.height);
                    if best.midstate != candidate_state.midstate {
                        self.metrics.inc_reorgs();
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
                    self.peer_manager.broadcast_except(from, &Message::Batch(batch)).await;
                    self.try_apply_orphans().await;
                }
                Ok(())
            }
            Err(_e) => {
                if let Some(peer_idx) = from {
                    self.peer_manager.send_to(peer_idx, &Message::GetState).await;
                }
                Ok(())
            }
        }
    }

    async fn request_missing_batches(&mut self, from: PeerIndex, peer_height: u64) {
        let gap = peer_height - self.state.height;
        if gap == 0 { return; }
        let start = self.state.height;
        let count = gap.min(MAX_GETBATCHES_COUNT);
        if start + count <= self.sync_requested_up_to { return; }

        self.sync_in_progress = true;
        self.sync_requested_up_to = start + count;
        self.peer_manager
            .send_to(from, &Message::GetBatches { start_height: start, count })
            .await;
    }

    async fn handle_batches_response(&mut self, batches: Vec<Batch>, from: PeerIndex) -> Result<()> {
        let mut applied = 0u64;
        for batch in batches {
            let mut candidate = self.state.clone();
            match apply_batch(&mut candidate, &batch) {
                Ok(_) => {
                    self.recent_states.push(self.state.clone());
                    if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                        self.recent_states.remove(0);
                    }
                    self.storage.save_batch(candidate.height - 1, &batch)?;
                    self.state = candidate;
                    self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                    self.metrics.inc_batches_processed();
                    applied += 1;
                }
                Err(e) => {
                    tracing::warn!("Gap-fill batch failed at height {}: {}", self.state.height, e);
                    self.peer_manager.record_misbehavior(from, 20);
                    break;
                }
            }
        }
        self.sync_in_progress = false;
        if applied > 0 {
            self.mempool.prune_invalid(&self.state);
            self.try_apply_orphans().await;
        }
        Ok(())
    }

    async fn try_apply_orphans(&mut self) {
        loop {
            let height = self.state.height;
            let batch = match self.orphan_batches.remove(&height) {
                Some(b) => b,
                None => break,
            };
            let mut candidate = self.state.clone();
            match apply_batch(&mut candidate, &batch) {
                Ok(_) => {
                    self.recent_states.push(self.state.clone());
                    if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                        self.recent_states.remove(0);
                    }
                    self.storage.save_batch(candidate.height - 1, &batch).ok();
                    self.state = candidate;
                    self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                    self.metrics.inc_batches_processed();
                    self.mempool.prune_invalid(&self.state);
                }
                Err(_) => break,
            }
        }
        while self.orphan_batches.len() > MAX_ORPHAN_BATCHES {
            if let Some(&oldest) = self.orphan_batches.keys().min() {
                self.orphan_batches.remove(&oldest);
            }
        }
    }

    /// Generate coinbase coins for a block at the given height.
    fn generate_coinbase(&self, height: u64, total_fees: u64) -> Vec<[u8; 32]> {
        let reward = block_reward(height);
        let count = reward + total_fees;
        (0..count).into_par_iter() 
            .map(|i| {
                let seed = coinbase_seed(&self.mining_seed, height, i);
                wots::keygen(&seed)
            })
            .collect()
    }

    /// Log coinbase seeds to a JSONL file for later wallet import.
    fn log_coinbase(&self, height: u64, total_fees: u64) {
        let reward = block_reward(height);
        let count = reward + total_fees;
        let log_path = self.data_dir.join("coinbase_seeds.jsonl");

        // STEP 1: Generate the JSON strings in PARALLEL (Fast)
        let entries: Vec<String> = (0..count).into_par_iter()
            .map(|i| {
                let seed = coinbase_seed(&self.mining_seed, height, i);
                let coin = wots::keygen(&seed);
                format!(
                    r#"{{"height":{},"index":{},"seed":"{}","coin":"{}"}}"#,
                    height,
                    i,
                    hex::encode(seed),
                    hex::encode(coin)
                )
            })
            .collect();

        // STEP 2: Write to file SEQUENTIALLY (Fast enough for I/O)
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path) 
        {
            use std::io::Write;
            for entry in entries {
                let _ = writeln!(file, "{}", entry);
            }
        }
    }

    async fn try_mine(&mut self) -> Result<()> {
        tracing::info!("Mining batch with {} transactions...", self.mempool.len());

        let transactions = self.mempool.drain(MAX_BATCH_SIZE);
        let pre_mine_height = self.state.height;
        let pre_mine_midstate = self.state.midstate;

        let mut candidate_state = self.state.clone();

        let mut total_fees: u64 = 0;
        for tx in &transactions {
            total_fees += tx.fee() as u64;
            apply_transaction(&mut candidate_state, tx)?;
        }

        // Generate coinbase coins
        let coinbase = self.generate_coinbase(pre_mine_height, total_fees);

        // Add coinbase coins to candidate state and fold into midstate
        for coin in &coinbase {
            candidate_state.coins.insert(*coin);
            candidate_state.midstate = hash_concat(&candidate_state.midstate, coin);
        }

        let midstate = candidate_state.midstate;
        let target = self.state.target;

        let extension = tokio::task::spawn_blocking(move || {
            mine_extension(midstate, target)
        })
        .await?;

        // Check staleness
        if self.state.height != pre_mine_height || self.state.midstate != pre_mine_midstate {
            tracing::warn!("State advanced during mining. Restoring transactions.");
            self.mempool.re_add(transactions, &self.state);
            return Ok(());
        }

        let batch = Batch {
            transactions,
            extension,
            coinbase: coinbase.clone(),
        };

        self.recent_states.push(self.state.clone());
        if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
            self.recent_states.remove(0);
        }

        match apply_batch(&mut self.state, &batch) {
            Ok(_) => {
                self.storage.save_batch(self.state.height - 1, &batch)?;
                self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                self.metrics.inc_batches_mined();
                self.log_coinbase(pre_mine_height, total_fees);
                self.peer_manager.broadcast(&Message::Batch(batch)).await;
                tracing::info!(
                    "Mined batch! height={} coinbase={} target={}",
                    self.state.height,
                    coinbase.len(),
                    hex::encode(self.state.target)
                );
            }
            Err(e) => {
                tracing::error!("Failed to apply our own mined batch: {}", e);
                self.mempool.re_add(batch.transactions, &self.state);
            }
        }

        Ok(())
    }
}
