use crate::core::*;
use crate::core::types::CoinbaseOutput;
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::{mine_extension, create_extension};

use crate::core::transaction::{apply_transaction, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::network::{Message, PeerConnection, PeerManager, PeerIndex, MAX_GETBATCHES_COUNT};
use crate::storage::Storage;
use crate::wallet::{coinbase_seed, coinbase_salt};

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
        let mut state = storage.load_state()?.unwrap_or_else(|| {
            tracing::info!("No saved state, using genesis");
            State::genesis().0
        });

        tracing::info!(
            "Loaded state: height={} depth={} coins={} commitments={}",
            state.height, state.depth, state.coins.len(), state.commitments.len()
        );

        if state.height == 0 {
            match storage.load_batch(0)? {
                None => {
                    tracing::info!("Creating genesis batch (batch_0)");
                    
                 let genesis_coinbase = State::genesis().1;

                    // --- LOGGING START ---
                    let mut mining_midstate = state.midstate;
                    for cb in &genesis_coinbase {
                        mining_midstate = hash_concat(&mining_midstate, &cb.coin_id());
                    }
                    tracing::error!("GENESIS DEBUG:");
                    tracing::error!("  State Midstate: {}", hex::encode(state.midstate));
                    tracing::error!("  Mining Midstate (State + Coinbase): {}", hex::encode(mining_midstate));
                    // --- LOGGING END ---

                    // Deterministic mining loop
                    let mut nonce = 0u64;
                    let extension = loop {
                        let ext = create_extension(mining_midstate, nonce);
                        
                        if ext.final_hash < state.target {
                            tracing::info!("Found deterministic genesis nonce: {}", nonce);
                            // --- LOGGING START ---
                            tracing::error!("  Nonce Found: {}", nonce);
                            tracing::error!("  Generated Checkpoint[0]: {}", hex::encode(ext.checkpoints[0]));
                            // --- LOGGING END ---
                            break ext;
                        }
                        nonce += 1;
                    };
                    
                    let genesis_batch = Batch {
                        prev_midstate: state.midstate,
                        transactions: vec![],
                        extension: extension,
                        coinbase: genesis_coinbase,
                        timestamp: state.timestamp,
                        target: state.target,
                    };
                    storage.save_batch(0, &genesis_batch)?;
                    
                    
                    // apply it so mining builds on top
                    apply_batch(&mut state, &genesis_batch)?;
                    storage.save_state(&state)?;
                    tracing::info!("Genesis batch applied, height now {}", state.height);
                }
                Some(batch) => {
                    // Also apply on reload if state is still at height 0
                    if state.height == 0 {
                        apply_batch(&mut state, &batch)?;
                        storage.save_state(&state)?;
                    }
                }
            }
        }

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

        let mempool = Mempool::new();

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
            Message::Batches { .. } => "Batches",
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
                
                if midstate != self.state.midstate {
                    if depth > self.state.depth {
                        // Competing chain - request with overlap to find fork
                        let rewind = 100.min(self.state.height);
                        let start = self.state.height.saturating_sub(rewind);
                        let count = (height - start).min(MAX_GETBATCHES_COUNT);
                        
                        self.sync_in_progress = true;
                        
                        tracing::warn!(
                            "Peer {} has competing chain! Requesting {} batches from {} (with overlap)",
                            from, count, start
                        );
                        
                        self.peer_manager
                            .send_to(from, &Message::GetBatches { start_height: start, count })
                            .await;
                    }
                } else if height > self.state.height {
                    // Same chain, just behind - normal sync
                    self.request_missing_batches(from, height).await;
                } else {
                    // We're up to date
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
                    Ok(tagged) => {
                        let actual_start = tagged.first().map(|(h, _)| *h).unwrap_or(start_height);
                        let batches: Vec<Batch> = tagged.into_iter().map(|(_, b)| b).collect();
                        self.peer_manager.send_to(from, &Message::Batches {
                            start_height: actual_start,
                            batches,
                        }).await;
                    }
                    Err(e) => tracing::warn!("Failed to load batches: {}", e),
                }
            }
            Message::Batches { start_height: batch_start, batches } => {
                if !batches.is_empty() {
                    let mut test_state = self.state.clone();

                    // 1. Fast path: Does the first batch extend our tip?
                    if apply_batch(&mut test_state, &batches[0]).is_ok() {
                        self.handle_batches_response(batch_start, batches, from).await?;

                    } else {
                        // 2. Overlap Scan: Does ANY batch in the list extend our tip?
                        // This fixes the "Gap-fill batch failed" error when syncing with overlap.
                        let mut found_extension = false;
                        for (i, batch) in batches.iter().enumerate() {
                            let mut candidate = self.state.clone();
                            if apply_batch(&mut candidate, batch).is_ok() {
                                tracing::info!("Found batch extending current chain at index {}", i);
                                // We found the point where their history meets our tip.
                                // Apply everything from this point onward.
                                let valid_batches = batches[i..].to_vec();
                                self.handle_batches_response(batch_start + i as u64, valid_batches, from).await?;

                                found_extension = true;
                                break;
                            }
                        }

                        // 3. Reorg Check: If no linear extension was found, it might be a fork.
                        if !found_extension {
                            // This is a competing chain. Assume fork is somewhere in the overlap window.
                            let fork_height = batch_start;


                            tracing::info!(
                                "Received alternative chain, evaluating from height {}...",
                                fork_height
                            );

                            match self.evaluate_alternative_chain(fork_height, &batches, from).await {
                                Ok(Some((new_state, new_history))) => {
                                    tracing::warn!(
                                        "PERFORMING REORG: height {} -> {}, depth {} -> {}",
                                        self.state.height,
                                        new_state.height,
                                        self.state.depth,
                                        new_state.depth
                                    );

                                    let fork_height = new_history.first().map(|(h, _, _)| *h).unwrap_or(0);

                                    let abandoned_history: Vec<_> = self
                                        .chain_history
                                        .iter()
                                        .skip_while(|(h, _, _)| *h < fork_height)
                                        .cloned()
                                        .collect();

                                    // ─── NEW: Persist the new batches (critical fix) ─────────────────────
                                    for (height, _, batch) in &new_history {
                                        if let Err(e) = self.storage.save_batch(*height, batch) {
                                            tracing::error!("Failed to save reorg batch at height {}: {}", height - 1, e);
                                        }
                                    }

                                    // ─── Replace state and chain history (unchanged) ─────────────────────
                                    self.state = new_state;

                                    let keep_old = self
                                        .chain_history
                                        .iter()
                                        .take_while(|(h, _, _)| *h < fork_height)
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    self.chain_history = keep_old;
                                    self.chain_history.extend(new_history);

                                    // ─── FIXED: Correctly rebuild recent_states (uses saved batches) ────
                                    self.recent_states.clear();
                                    let window_start = self.state.height.saturating_sub((DIFFICULTY_ADJUSTMENT_INTERVAL * 2) as u64);

                                    // Base state at the start of the window (height == window_start)
                                    let mut current_state = self.rebuild_state_at_height(window_start)?;

                                    self.recent_states.push(current_state.clone());

                                    for pre_height in window_start..self.state.height {
                                        let batch = match self.storage.load_batch(pre_height)? {
                                            Some(b) => b,
                                            None => {
                                                tracing::error!("Missing batch {} after reorg (should not happen)", pre_height);
                                                continue;
                                            }
                                        };
                                        // These applies cannot fail — the chain was already validated
                                        if apply_batch(&mut current_state, &batch).is_err() {
                                            tracing::error!("Invalid batch {} during recent_states rebuild", pre_height);
                                            break;
                                        }
                                        self.recent_states.push(current_state.clone());
                                    }

                                    // ─── NEW: Final difficulty adjustment using the correct window ──────
                                    self.state.target = adjust_difficulty(&self.state, &self.recent_states);

                                    // ─── Resubmit abandoned transactions (unchanged) ────────────────────
                                    for (_, _, batch) in abandoned_history {
                                        self.mempool.re_add(batch.transactions, &self.state);
                                    }
                                    self.mempool.prune_invalid(&self.state);

                                    self.metrics.inc_reorgs();
                                    self.storage.save_state(&self.state)?;

                                    return Ok(());
                                }
                                Ok(None) => {} // Not better, ignore
                                Err(e) => tracing::error!("Error evaluating reorg: {}", e),
                            }
                        }
                    }
                }
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
         State::genesis().0
    } else {
        // Find the state at fork_height - 1
        match self.chain_history.iter().find(|(h, _, _)| *h == fork_height - 1) {
            Some((_, _midstate, _)) => {
                if fork_height > self.state.height.saturating_sub(self.max_reorg_depth) {
                    self.rebuild_state_at_height(fork_height - 1)?
                } else {
                    tracing::warn!("Rejecting reorg: fork at {} is deeper than max_reorg_depth", fork_height);
                    return Ok(None);
                }
            }
            None => {
                // PATCH: Fallback to storage if not in memory (e.g. after restart)
                if fork_height > self.state.height.saturating_sub(self.max_reorg_depth) {
                    // Try to rebuild from storage. rebuild_state_at_height handles loading batches.
                    match self.rebuild_state_at_height(fork_height - 1) {
                        Ok(s) => s,
                        Err(_) => {
                            tracing::warn!("Cannot find fork point at height {}", fork_height);
                            return Ok(None);
                        }
                    }
                } else {
                    tracing::warn!("Cannot find fork point at height {}", fork_height);
                    return Ok(None);
                }
            }
        }
    };

    println!("DEBUG: Evaluating reorg from height {}. Batches: {}", fork_height, alternative_batches.len());
    
    // Try applying the alternative chain
    let mut candidate_state = fork_state;
    let mut new_history = Vec::new();
    let mut recent_states = Vec::new();

    // Only rebuild recent_states for the window we actually need
    // We only need DIFFICULTY_ADJUSTMENT_INTERVAL * 2 blocks before fork_height
    let window_size = (DIFFICULTY_ADJUSTMENT_INTERVAL as usize) * 2;
    let start_height = fork_height.saturating_sub(window_size as u64);

    for h in start_height..fork_height {
        if let Some(batch) = self.storage.load_batch(h)? {
            let mut temp_state = if h == start_height {
                // For the first block in our window, rebuild its state
                if start_height == 0 {
                     State::genesis().0
                } else {
                    self.rebuild_state_at_height(start_height)?
                }
            } else {
                recent_states.last().cloned().unwrap()
            };
            
            apply_batch(&mut temp_state, &batch)?;
            temp_state.target = adjust_difficulty(&temp_state, &recent_states);
            recent_states.push(temp_state);
        }
    }

    for (i, batch) in alternative_batches.iter().enumerate() {
        // Track state before this batch for difficulty
        recent_states.push(candidate_state.clone());
        if recent_states.len() > window_size {
            recent_states.remove(0);
        }
        
        // CRITICAL FIX: The batch's prev_midstate MUST match candidate_state.midstate
        // We perform this check manually here to provide better error logging
        if batch.prev_midstate != candidate_state.midstate {
            // --- DEBUG PRINT ---
            println!("DEBUG: Reorg Broken Link at index {}!", i);
            println!("DEBUG:   Candidate Midstate: {}", hex::encode(candidate_state.midstate));
            println!("DEBUG:   Batch Prev Midstate: {}", hex::encode(batch.prev_midstate));
            tracing::warn!(
                "Alternative chain broken at batch index {} (height {}). Expected parent {}, got {}", 
                i, fork_height + i as u64,
                hex::encode(candidate_state.midstate),
                hex::encode(batch.prev_midstate)
            );
            return Ok(None); // Chain is invalid, discard
        }       
        
        
        match apply_batch(&mut candidate_state, batch) {
            Ok(_) => {
                // Update difficulty based on alternative chain's history
                candidate_state.target = adjust_difficulty(&candidate_state, &recent_states);
                
                new_history.push((
                    fork_height + i as u64,
                    candidate_state.midstate,
                    batch.clone()
                ));
            }
            Err(e) => {
                println!("DEBUG: Reorg Apply Failed at index {}: {}", i, e);
                tracing::warn!("Alternative chain invalid at height {}: {}", fork_height + i as u64, e);
                self.peer_manager.record_misbehavior(from, 50);
                return Ok(None);
            }
        }
    }
    
    // Compare chains: choose based on depth (total work)
    let our_depth = self.state.depth;
    let their_depth = candidate_state.depth;
    
    if their_depth > our_depth { //previously: || (their_depth == our_depth && candidate_state.midstate < self.state.midstate)
        
        tracing::warn!(
            "REORG DETECTED: Alternative chain has more work (depth {} > {})",
            their_depth, our_depth
        );
        Ok(Some((candidate_state, new_history)))
    } else {
        println!("DEBUG: Reorg rejected due to depth. Theirs: {}, Ours: {}", their_depth, our_depth);
        tracing::debug!(
            "Rejecting alternative chain: insufficient work (depth {} <= {})",
            their_depth, our_depth
        );
        Ok(None)
    }
}

/// Rebuild state at a specific height by replaying from genesis or nearest checkpoint
fn rebuild_state_at_height(&self, target_height: u64) -> Result<State> {
      let mut state = State::genesis().0;
    let mut recent_states = Vec::new();
    
    // Replay batches from storage up to target_height
    for h in 0..target_height {
        if let Some(batch) = self.storage.load_batch(h)? {
            // Track recent states for difficulty adjustment
            recent_states.push(state.clone());
            if recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                recent_states.remove(0);
            }
            
            apply_batch(&mut state, &batch)?;
            state.target = adjust_difficulty(&state, &recent_states);
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
                let pre_height = self.state.height;
                self.state = candidate_state;
                self.storage.save_batch(pre_height, &batch)?;
                self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                self.metrics.inc_batches_processed();
                self.mempool.prune_invalid(&self.state);
                
                // Store in chain history BEFORE broadcasting
                self.chain_history.push((
                    pre_height, 
                    self.state.midstate,
                    batch.clone()
                ));
                
                // Prune old history
                if self.chain_history.len() > self.max_reorg_depth as usize {
                    self.chain_history.remove(0);
                }
                
                // Now broadcast
                self.peer_manager.broadcast_except(from, &Message::Batch(batch)).await;
                
                // Try to apply any orphans that might now connect
                self.try_apply_orphans().await;
                
            } else {
                tracing::warn!("DEBUG: Rejecting batch, not better than current state");
            }
            Ok(())
        }
        Err(e) => {
            let err_str = e.to_string();
            tracing::error!("DEBUG: Batch apply_batch() failed: {}", err_str);
            
            // ─── ORPHANAGE SETUP ─────────────────────────────────────────────
            // If the parent midstate doesn't match, this is a future block or fork.
            if err_str.contains("Block parent mismatch") || 
               err_str.contains("not found") || 
               err_str.contains("No matching commitment") 
            {
                tracing::info!("Received orphan/fork block (parent mismatch). Storing for reorg evaluation.");

                // 1. Safety Limit: Don't hoard too many orphans (Limit to ~6 blocks / 1 min)
                const ORPHAN_LIMIT: usize = 64;
                if self.orphan_batches.len() >= ORPHAN_LIMIT {
                    tracing::warn!("Orphan limit reached ({}), clearing buffer to force fresh sync", ORPHAN_LIMIT);
                    self.orphan_batches.clear();
                }

                // 2. Store the orphan
                // Since `Batch` doesn't have an explicit height field, we guess it belongs 
                // to the next slot. If it's a deep fork, the Sync request below will fix it.
                let estimated_height = self.state.height + 1;
                self.orphan_batches.insert(estimated_height, batch);
                
                tracing::info!("Stored orphan batch at est. height {}, total orphans: {}", 
                    estimated_height, self.orphan_batches.len());

                // 3. Trigger Sync/Reorg Evaluation
                // We ask the peer for their state. If they are ahead or on a fork, 
                // `handle_message` -> `StateInfo` will trigger `evaluate_alternative_chain`.
                // Request history overlap directly instead of GetState
                if let Some(peer_idx) = from {
                    // Ask for the last 50 blocks to find where we diverged significantly faster
                    let start_h = self.state.height.saturating_sub(50).max(0);
                    tracing::info!("Requesting overlap history from peer {} starting at {}", peer_idx, start_h);
                    
                    self.peer_manager.send_to(peer_idx, &Message::GetBatches {
                        start_height: start_h,
                        count: 100 // Fetch 100 to cover the gap and extend
                    }).await;
                }
            }
            // ─────────────────────────────────────────────────────────────────
            else {
                // Genuine invalid batch (bad signature, bad PoW, etc.)
                if let Some(peer_idx) = from {
                    self.peer_manager.record_misbehavior(peer_idx, 10);
                }
            }
            Ok(())
        }
    }
}

async fn request_missing_batches(&mut self, from: PeerIndex, peer_height: u64) {
    if peer_height <= self.state.height { return; }
    let gap = peer_height - self.state.height;
    
    let start = self.state.height;
    let count = gap.min(MAX_GETBATCHES_COUNT);
    
    if self.sync_in_progress && start + count <= self.sync_requested_up_to { 
        return; 
    }
    
    self.sync_in_progress = true;
    self.sync_requested_up_to = start + count;
    
    tracing::info!("Requesting {} batches from peer {} (height {} -> {})", 
        count, from, start, start + count - 1);
    
    self.peer_manager
        .send_to(from, &Message::GetBatches { start_height: start, count })
        .await;
}

async fn handle_batches_response(&mut self, batch_start_height: u64, batches: Vec<Batch>, from: PeerIndex) -> Result<()> {
    if batches.is_empty() { return Ok(()); }

    let batch_count = batches.len();
    tracing::info!("Received {} batch(es) from peer {}", batch_count, from);

    // 1. Fast Path
    let mut test_state = self.state.clone();
    if apply_batch(&mut test_state, &batches[0]).is_ok() {
        return self.process_linear_extension(batches, from).await;
    } else {
        // --- DEBUG PRINT ---
        println!("DEBUG: Fast path failed for batch 0 (height unknown)");
    }

    // 2. Overlap Scan
    for (i, batch) in batches.iter().enumerate() {
        let mut candidate = self.state.clone();
        match apply_batch(&mut candidate, batch) {
            Ok(_) => {
                tracing::info!("Found linear extension starting at batch index {}", i);
                return self.process_linear_extension(batches[i..].to_vec(), from).await;
            }
            Err(e) => {
                // --- DEBUG PRINT ---
                // Only print for first few to avoid spam
                if i < 3 {
                    println!("DEBUG: Overlap scan failed at index {}: {}", i, e);
                    println!("DEBUG:   My State Midstate: {}", hex::encode(self.state.midstate));
                    println!("DEBUG:   Batch Prev Midstate: {}", hex::encode(batch.prev_midstate));
                }
            }
        }
    }

    // 3. Reorg Check: Find the Common Ancestor
    // We cannot guess the fork height. We must find where these batches attach to our history.
    
    let mut connection_found = false;

    // We scan the incoming batches to see if any of them link to a block we already have.
    for (i, batch) in batches.iter().enumerate() {
        // We know batch[i] is at height batch_start_height + i.
        // Its prev_midstate tells us it wants to attach at height (batch_start_height + i - 1).
        // Try to evaluate the fork from there.
        let fork_height = batch_start_height + i as u64;

        // Verify the batch actually connects to something we know about
        let connects = batch.prev_midstate == State::genesis().0.midstate
            || self.chain_history.iter().any(|(_, mid, _)| *mid == batch.prev_midstate)
            || self.storage.load_batch(fork_height.saturating_sub(1))
                .ok()
                .flatten()
                .is_some();

        if connects {
            let attach_height = fork_height;
                        tracing::info!("Found fork connection point at height {} (batch index {})", attach_height, i);
            
            // We found the zipper point! 
            // Try to evaluate the chain starting from this specific batch.
            let relevant_batches = &batches[i..];
            
            match self.evaluate_alternative_chain(attach_height, relevant_batches, from).await {
                Ok(Some((new_state, new_history))) => {
                    tracing::warn!(
                        "PERFORMING REORG: height {} -> {}, depth {} -> {}",
                        self.state.height, new_state.height,
                        self.state.depth, new_state.depth
                    );

                    let fork_height = new_history.first().map(|(h, _, _)| *h).unwrap_or(0);
                    
                    // 1. Recover transactions from abandoned chain
                    let abandoned_history: Vec<_> = self.chain_history.iter()
                        .skip_while(|(h, _, _)| *h < fork_height)
                        .cloned()
                        .collect();

                    // 2. Persist new batches so reorgs survive restarts
                    for (height, _, batch) in &new_history {
                        if let Err(e) = self.storage.save_batch(*height, batch) {
                            tracing::error!("Failed to save reorg batch at height {}: {}", height, e);
                        }
                    }

                    // 3. Rewrite History
                    self.chain_history.retain(|(h, _, _)| *h < fork_height);
                    self.chain_history.extend(new_history);

                    // 3. Rebuild Recent States (Critical for difficulty adjustment)
                    self.recent_states.clear();
                    let window_start = new_state.height.saturating_sub((DIFFICULTY_ADJUSTMENT_INTERVAL * 2) as u64);
                    
                    // Replay history to rebuild recent_states window
                    // (We can optimize this later, but this ensures correctness)
                    for h in window_start..new_state.height {
                        if let Ok(Some(b)) = self.storage.load_batch(h) {
                            let mut temp_state = if h == 0 { State::genesis().0 } else {
                                self.recent_states.last().cloned().unwrap_or(State::genesis().0)
                            };
                            if apply_batch(&mut temp_state, &b).is_ok() {
                                self.recent_states.push(temp_state);
                            }
                        }
                    }
                    self.recent_states.push(new_state.clone());

                    // 4. Update Global State
                    self.state = new_state;
                    self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                    
                    // 5. Restore Mempool
                    for (_, _, batch) in abandoned_history {
                        self.mempool.re_add(batch.transactions, &self.state);
                    }
                    self.mempool.prune_invalid(&self.state);
                    
                    // 6. Save & Metrics
                    self.storage.save_state(&self.state)?;
                    self.metrics.inc_reorgs();
                    
                    connection_found = true;
                    break; // Stop scanning, we successfully reorged
                }
                Ok(None) => {
                    // This specific attachment point didn't result in a better chain.
                    // Continue scanning in case a later batch attaches to a later point?
                    // Usually unlikely, but safe to continue.
                }
                Err(e) => {
                    tracing::warn!("Failed to evaluate fork at height {}: {}", attach_height, e);
                }
            }
        }
    }

    if !connection_found {
        tracing::debug!("Orphan batch chain received (could not find attachment point in history)");
        // Optional: Trigger a deeper GetState or just wait
    }

    Ok(())
}

// Helper to dedup the linear extension logic
async fn process_linear_extension(&mut self, batches: Vec<Batch>, from: PeerIndex) -> Result<()> {
    let mut applied = 0;
    for batch in batches {
        let mut candidate = self.state.clone();
        if apply_batch(&mut candidate, &batch).is_ok() {
            // .. (Standard apply logic you already had) ..
            self.recent_states.push(self.state.clone());
            if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                self.recent_states.remove(0);
            }
            self.storage.save_batch(candidate.height - 1, &batch)?;
            self.state = candidate;
            self.state.target = adjust_difficulty(&self.state, &self.recent_states);
            self.metrics.inc_batches_processed();
            
            // Add to history
            self.chain_history.push((self.state.height, self.state.midstate, batch.clone()));
            if self.chain_history.len() > self.max_reorg_depth as usize {
                self.chain_history.remove(0);
            }
            
            applied += 1;
        } else {
            break;
        }
    }
    
    if applied > 0 {
        tracing::info!("Synced {} batch(es), now at height {}", applied, self.state.height);
        self.mempool.prune_invalid(&self.state);
        self.try_apply_orphans().await;
        
        // Check if we need more
        self.peer_manager.send_to(from, &Message::GetState).await;
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
    fn generate_coinbase(&self, height: u64, total_fees: u64) -> Vec<CoinbaseOutput> {
        let reward = block_reward(height);
        let total_value = reward + total_fees;
        let denominations = decompose_value(total_value);

        denominations.into_par_iter()
            .enumerate()
            .map(|(i, value)| {
                let seed = coinbase_seed(&self.mining_seed, height, i as u64);
                let owner_pk = wots::keygen(&seed);
                let salt = coinbase_salt(&self.mining_seed, height, i as u64);
                CoinbaseOutput { owner_pk, value, salt }
            })
            .collect()
    }

    /// Log coinbase seeds to a JSONL file for later wallet import.
    fn log_coinbase(&self, height: u64, total_fees: u64) {
        let reward = block_reward(height);
        let total_value = reward + total_fees;
        let denominations = decompose_value(total_value);
        let log_path = self.data_dir.join("coinbase_seeds.jsonl");

        let entries: Vec<String> = denominations.into_par_iter()
            .enumerate()
            .map(|(i, value)| {
                let seed = coinbase_seed(&self.mining_seed, height, i as u64);
                let owner_pk = wots::keygen(&seed);
                let salt = coinbase_salt(&self.mining_seed, height, i as u64);
                let coin_id = compute_coin_id(&owner_pk, value, &salt);
                format!(
                    r#"{{"height":{},"index":{},"seed":"{}","coin":"{}","value":{},"salt":"{}"}}"#,
                    height, i,
                    hex::encode(seed),
                    hex::encode(coin_id),
                    value,
                    hex::encode(salt)
                )
            })
            .collect();

        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true).append(true).open(&log_path)
        {
            use std::io::Write;
            for entry in entries {
                let _ = writeln!(file, "{}", entry);
            }
        }
    }

    async fn try_mine(&mut self) -> Result<()> {
    // 1. pause mine if sync in progress
        if self.sync_in_progress {
            tracing::debug!("Skipping mining to allow sync to complete.");
            return Ok(());
        }
        tracing::info!("Mining batch with {} transactions...", self.mempool.len());

        let transactions = self.mempool.drain(MAX_BATCH_SIZE);
        let pre_mine_height = self.state.height;
        let pre_mine_midstate = self.state.midstate;

        let mut candidate_state = self.state.clone();

        let mut total_fees: u64 = 0;
        for tx in &transactions {
            total_fees += tx.fee();
            apply_transaction(&mut candidate_state, tx)?;
        }

        // Generate coinbase coins
        let coinbase = self.generate_coinbase(pre_mine_height, total_fees);

        // Add coinbase coin IDs to candidate state and fold into midstate
        for cb in &coinbase {
            let coin_id = cb.coin_id();
            candidate_state.coins.insert(coin_id);
            candidate_state.midstate = hash_concat(&candidate_state.midstate, &coin_id);
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

        // NEW: Set timestamp to max(current_time, prev_timestamp + 1)
        let current_time = state::current_timestamp();
        let block_timestamp = current_time.max(self.state.timestamp + 1);

        let batch = Batch {
            prev_midstate: pre_mine_midstate,
            transactions,
            extension,
            coinbase: coinbase.clone(),
            timestamp: block_timestamp,  // NEW: Include timestamp in batch
            target: self.state.target,
        };

        self.recent_states.push(self.state.clone());
        if self.recent_states.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
            self.recent_states.remove(0);
        }

        match apply_batch(&mut self.state, &batch) {
            Ok(_) => {
                self.storage.save_batch(pre_mine_height, &batch)?;
                self.storage.save_state(&self.state)?;
                self.state.target = adjust_difficulty(&self.state, &self.recent_states);
                self.metrics.inc_batches_mined();
                self.peer_manager.broadcast(&Message::Batch(batch)).await;
                self.log_coinbase(pre_mine_height, total_fees);
                               
                
                tracing::warn!("DEBUG: Broadcasting batch at height={} to {} peers", 
                    self.state.height,
                    self.peer_manager.peer_count());
                let coinbase_value: u64 = coinbase.iter().map(|cb| cb.value).sum();
                tracing::info!(
                    "Mined batch! height={} coinbase_value={} outputs={} target={}",
                    self.state.height,
                    coinbase_value,
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
