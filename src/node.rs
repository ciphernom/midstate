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
    chain_history: Vec<(u64, [u8; 32], Batch)>, // (height, midstate_after, batch)
    max_reorg_depth: u64,
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
            chain_history: Vec::new(),
            max_reorg_depth: 100, // Allow reorgs up to 100 blocks deep
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
        let mut sync_poll_interval = time::interval(Duration::from_secs(30));

        // ===== ADD THIS BLOCK =====
        // Initial sync: ask all peers for their height before mining
        if self.peer_manager.peer_count() > 0 {
            tracing::info!("Requesting chain state from {} peer(s)...", self.peer_manager.peer_count());
            
            // Set sync flag to block mining during initial sync
            self.sync_in_progress = true;
            
            // Ask all peers for their state
            for idx in 0..self.peer_manager.peer_count() {
                self.peer_manager.send_to(idx as u64, &Message::GetState).await;
            }
            
            // Wait a bit for responses
            tokio::time::sleep(Duration::from_secs(2)).await;
            
            tracing::info!("Initial sync requests sent, starting node loop");
        }
        // ===== END ADD =====

        loop {
            tokio::select! {
                _ = mine_interval.tick() => {
                    if self.is_mining {
                        // Block mining if we're significantly behind
                        if self.sync_in_progress {
                            tracing::debug!("Skipping mining (sync in progress)");
                            continue;
                        }
                        
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
                _ = sync_poll_interval.tick() => {
                    // Periodically ask peers for their state to detect if we're behind
                    if self.peer_manager.peer_count() > 0 {
                        if let Some(peer_idx) = self.peer_manager.random_peer() {
                            self.peer_manager.send_to(peer_idx, &Message::GetState).await;
                        }
                    }
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
            tracing::warn!("DEBUG: Received message from peer {}: {:?}", from, 
        match &msg {
            Message::Transaction(_) => "Transaction",
            Message::Batch(_) => "Batch",
            Message::GetState => "GetState",
            Message::StateInfo { .. } => "StateInfo",
            Message::Ping { .. } => "Ping",
            Message::Pong { .. } => "Pong",
            Message::GetAddr => "GetAddr",
            Message::Addr(_) => "Addr",
            Message::Version { .. } => "Version",
            Message::Verack => "Verack",
            Message::GetBatches { .. } => "GetBatches",
            Message::Batches(_) => "Batches",
        });
        match msg {
            Message::Transaction(tx) => {
                self.handle_new_transaction(tx, Some(from)).await?;
            }
            Message::Batch(batch) => {
                tracing::warn!("DEBUG: Handling batch from peer {}", from);
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
            Message::StateInfo { height, depth, midstate } => {
                tracing::debug!("Peer {} state: height={} depth={}", from, height, depth);
                
                // Check if peer has a competing chain with more work
                if depth > self.state.depth && midstate != self.state.midstate {
                    tracing::warn!(
                        "Peer {} has competing chain with more work! depth {} > {} (height {} vs {})",
                        from, depth, self.state.depth, height, self.state.height
                    );
                    
                    // Find potential fork point (simplified: request from genesis or recent checkpoint)
                    let fork_start = self.state.height.saturating_sub(self.max_reorg_depth).max(0);
                    let count = (height - fork_start).min(MAX_GETBATCHES_COUNT);
                    
                    tracing::info!("Requesting alternative chain from height {} (count {})", fork_start, count);
                    self.request_missing_batches(from, height).await;
                } else if height > self.state.height {
                    // Normal sync (same chain, we're just behind)
                    if !self.sync_in_progress {
                        self.request_missing_batches(from, height).await;
                    }
                } else {
                    // We're caught up or ahead
                    self.sync_in_progress = false;
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
                // Check if this is a competing chain (fork)
                if !batches.is_empty() {
                    // Assume batches are sequential starting from some height
                    // We need to detect if they fork from our current chain
                    
                    let first_batch_height = self.state.height; // Simplified: assume they start where we requested
                    
                    // If the first batch doesn't apply on our current state, it might be a fork
                    let mut test_state = self.state.clone();
                    if apply_batch(&mut test_state, &batches[0]).is_err() {
                        // This might be a fork, evaluate the alternative chain
                        tracing::info!("Received alternative chain, evaluating for reorg...");
                        
                        match self.evaluate_alternative_chain(first_batch_height, &batches, from).await {
                            Ok(Some((new_state, new_history))) => {
                                // Perform the reorg!
                                tracing::warn!(
                                    "PERFORMING REORG: height {} -> {}, depth {} -> {}",
                                    self.state.height, new_state.height,
                                    self.state.depth, new_state.depth
                                );
                                
                                // Replace our state and history
                                self.state = new_state;
                                
                                // Replace recent chain history
                                let keep_old = self.chain_history.iter()
                                    .take_while(|(h, _, _)| *h < first_batch_height)
                                    .cloned()
                                    .collect::<Vec<_>>();
                                
                                self.chain_history = keep_old;
                                self.chain_history.extend(new_history);
                                
                                // Prune mempool
                                self.mempool.prune_invalid(&self.state);
                                
                                self.metrics.inc_reorgs();
                                
                                // Save the new state
                                self.storage.save_state(&self.state)?;
                                
                                return Ok(());
                            }
                            Ok(None) => {
                                // Alternative chain rejected, fall through to normal handling
                            }
                            Err(e) => {
                                tracing::error!("Error evaluating alternative chain: {}", e);
                                // Fall through to normal handling
                            }
                        }
                    }
                }
                
                // Try normal sequential sync
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


/// Check if an alternative chain starting from `fork_height` is better than our chain.
/// Returns (should_reorg, new_state, new_history) if the alternative is better.
async fn evaluate_alternative_chain(
    &mut self,
    fork_height: u64,
    alternative_batches: &[Batch],
    from: PeerIndex,
) -> Result<Option<(State, Vec<(u64, [u8; 32], Batch)>)>> {
    
    // Find the fork point in our history
    let fork_state = if fork_height == 0 {
        State::genesis()
    } else {
        // Find the state at fork_height - 1
        match self.chain_history.iter().find(|(h, _, _)| *h == fork_height - 1) {
            Some((_, _midstate, _)) => {
                // Rebuild state at fork point by replaying from genesis
                // (For now, simplified: just check if we have it in history)
                if fork_height > self.state.height.saturating_sub(self.max_reorg_depth) {
                    // We have this in history, need to rebuild state
                    self.rebuild_state_at_height(fork_height - 1)?
                } else {
                    // Fork is too deep, reject
                    tracing::warn!("Rejecting reorg: fork at {} is deeper than max_reorg_depth", fork_height);
                    return Ok(None);
                }
            }
            None => {
                tracing::warn!("Cannot find fork point at height {}", fork_height);
                return Ok(None);
            }
        }
    };
    
    // Try applying the alternative chain
    let mut candidate_state = fork_state;
    let mut new_history = Vec::new();
    
    for (i, batch) in alternative_batches.iter().enumerate() {
        match apply_batch(&mut candidate_state, batch) {
            Ok(_) => {
                new_history.push((
                    fork_height + i as u64,
                    candidate_state.midstate,
                    batch.clone()
                ));
            }
            Err(e) => {
                tracing::warn!("Alternative chain invalid at height {}: {}", fork_height + i as u64, e);
                self.peer_manager.record_misbehavior(from, 50);
                return Ok(None);
            }
        }
    }
    
    // Compare chains: choose based on depth (total work)
    let our_depth = self.state.depth;
    let their_depth = candidate_state.depth;
    
    if their_depth > our_depth {
        tracing::warn!(
            "REORG DETECTED: Alternative chain has more work (depth {} > {})",
            their_depth, our_depth
        );
        Ok(Some((candidate_state, new_history)))
    } else {
        tracing::debug!(
            "Rejecting alternative chain: insufficient work (depth {} <= {})",
            their_depth, our_depth
        );
        Ok(None)
    }
}

/// Rebuild state at a specific height by replaying from genesis or nearest checkpoint
fn rebuild_state_at_height(&self, target_height: u64) -> Result<State> {
    let mut state = State::genesis();
    
    // Replay batches from storage up to target_height
    for h in 0..target_height {
        if let Some(batch) = self.storage.load_batch(h)? {
            apply_batch(&mut state, &batch)?;
            state.target = adjust_difficulty(&state, &self.recent_states);
        } else {
            anyhow::bail!("Missing batch at height {} needed for reorg", h);
        }
    }
    
    Ok(state)
}

 async fn handle_new_batch(&mut self, batch: Batch, from: Option<PeerIndex>) -> Result<()> {
    tracing::warn!("DEBUG: handle_new_batch called, from={:?}, current_height={}", 
        from, self.state.height);
    
    let mut candidate_state = self.state.clone();
    match apply_batch(&mut candidate_state, &batch) {
        Ok(_) => {
            tracing::warn!("DEBUG: Batch applied successfully, new height would be={}", 
                candidate_state.height);
            
            // Choose best between current and candidate (handles competing batches at same height)
            let best = choose_best_state(&self.state, &candidate_state);
            
            // Detect reorg: same height but different midstate
            let is_reorg = best.height == self.state.height && 
                           best.midstate != self.state.midstate;
            
            if best.height > self.state.height || is_reorg {
                if is_reorg {
                    tracing::warn!("REORG at height {}", self.state.height);
                    self.metrics.inc_reorgs();
                }
                
                tracing::info!("DEBUG: Accepting batch! height {} -> {}", 
                    self.state.height, best.height);
                
                self.recent_states.push(self.state.clone());
                if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                    self.recent_states.remove(0);
                }
                self.state = candidate_state;
                self.storage.save_batch(self.state.height - 1, &batch)?;
                self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                self.metrics.inc_batches_processed();
                self.mempool.prune_invalid(&self.state);
                
                // Store in chain history BEFORE broadcasting (which moves batch)
                self.chain_history.push((
                    self.state.height - 1, 
                    self.state.midstate,
                    batch.clone()
                ));
                
                // Prune old history
                if self.chain_history.len() > self.max_reorg_depth as usize {
                    self.chain_history.remove(0);
                }
                
                // Now broadcast (this moves batch)
                self.peer_manager.broadcast_except(from, &Message::Batch(batch)).await;
                self.try_apply_orphans().await;
                
            } else {
                tracing::warn!("DEBUG: Rejecting batch, not better than current state");
            }
            Ok(())
        }
        Err(e) => {
            tracing::error!("DEBUG: Batch apply_batch() failed: {}", e);
            
            // Check if this is a future batch (gap in chain)
            // We detect this by checking if the batch extends beyond our current height
            // This happens when apply_batch fails because commitments/coins don't exist yet
            if e.to_string().contains("not found") || e.to_string().contains("No matching commitment") {
                // Store as orphan for later application
                let expected_height = self.state.height; // Next batch should be at current height
                if self.orphan_batches.len() < MAX_ORPHAN_BATCHES {
                    self.orphan_batches.insert(expected_height + 1, batch);
                    tracing::info!("Stored orphan batch, total orphans: {}", self.orphan_batches.len());
                    
                    // Request missing batches from this peer
                    if let Some(peer_idx) = from {
                        self.peer_manager.send_to(peer_idx, &Message::GetState).await;
                    }
                }
            } else {
                // Actually invalid batch
                if let Some(peer_idx) = from {
                    self.peer_manager.record_misbehavior(peer_idx, 10);
                }
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
    
    // Don't re-request if we already asked for these batches
    if self.sync_in_progress && start + count <= self.sync_requested_up_to { 
        return; 
    }
    
    
    // Set sync flag to block mining
    self.sync_in_progress = true;
    self.sync_requested_up_to = start + count;
    
    tracing::info!("Requesting {} batches from peer {} (height {} -> {})", 
        count, from, start, start + count - 1);
    
    self.peer_manager
        .send_to(from, &Message::GetBatches { start_height: start, count })
        .await;
}

async fn handle_batches_response(&mut self, batches: Vec<Batch>, from: PeerIndex) -> Result<()> {
    let mut applied = 0u64;
    let batch_count = batches.len();
    
    tracing::info!("Received {} batch(es) from peer {}", batch_count, from);
    
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
    
    if applied > 0 {
        tracing::info!("Synced {} batch(es), now at height {}", applied, self.state.height);
        self.mempool.prune_invalid(&self.state);
        self.try_apply_orphans().await;
        
        // Check if we need more batches by asking for peer state
        self.peer_manager.send_to(from, &Message::GetState).await;
    } else {
        // No progress made, clear sync flag
        self.sync_in_progress = false;
    }
    
    Ok(())
}

async fn try_apply_orphans(&mut self) {
    let mut applied = 0;
    
    loop {
        let height = self.state.height; // The height we're looking to fill next
        let batch = match self.orphan_batches.remove(&height) {
            Some(b) => b,
            None => break, // No orphan at this height
        };
        
        tracing::info!("Applying orphan batch at height {}", height);
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
                applied += 1;
            }
            Err(e) => {
                tracing::warn!("Orphan batch at {} still invalid: {}", height, e);
                break; // Stop trying, still missing dependencies
            }
        }
    }
    
    if applied > 0 {
        tracing::info!("Applied {} orphan batch(es)", applied);
    }
    
    // Prune orphans that are too old (more than 10 blocks behind)
    let cutoff = self.state.height.saturating_sub(10);
    self.orphan_batches.retain(|&h, _| h > cutoff);
    
    // Enforce max orphans limit
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
                self.peer_manager.broadcast(&Message::Batch(batch)).await;
                self.log_coinbase(pre_mine_height, total_fees);
                               
                
                tracing::warn!("DEBUG: Broadcasting batch at height={} to {} peers", 
                    self.state.height - 1,
                    self.peer_manager.peer_count());
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
