use crate::core::*;
use crate::core::types::compute_address;
use crate::core::types::CoinbaseOutput;
use crate::core::types::BatchHeader;
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::{mine_extension, create_extension};
use crate::core::transaction::{apply_transaction, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::network::{Message, MidstateNetwork, NetworkEvent, MAX_GETBATCHES_COUNT};
use crate::storage::Storage;
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::mss;
use crate::core::wots;
use anyhow::Result;
use libp2p::{request_response::ResponseChannel, PeerId, Multiaddr, identity::Keypair};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time;
use rayon::prelude::*;

const MAX_ORPHAN_BATCHES: usize = 256;

pub struct Node {
    state: State,
    mempool: Mempool,
    storage: Storage,
    network: MidstateNetwork,
    metrics: Metrics,
    is_mining: bool,
    recent_headers: Vec<BatchHeader>,
    orphan_batches: HashMap<u64, Batch>,
    sync_in_progress: bool,
    sync_requested_up_to: u64,
    mining_seed: [u8; 32],
    data_dir: PathBuf,
    chain_history: Vec<(u64, [u8; 32], Batch)>,
    max_reorg_depth: u64,
}

#[derive(Clone)]
pub struct NodeHandle {
    state: Arc<RwLock<State>>,
    mempool_size: Arc<RwLock<usize>>,
    mempool_txs: Arc<RwLock<Vec<Transaction>>>,
    peer_addrs: Arc<RwLock<Vec<String>>>,
    tx_sender: tokio::sync::mpsc::UnboundedSender<NodeCommand>,
    batches_path: PathBuf,
}

pub enum NodeCommand {
    SendTransaction(Transaction),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScannedCoin {
    pub address: [u8; 32],
    pub value: u64,
    pub salt: [u8; 32],
    pub coin_id: [u8; 32],
    pub height: u64,
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

    pub async fn get_peers(&self) -> Vec<String> {
        self.peer_addrs.read().await.clone()
    }

    pub async fn send_transaction(&self, tx: Transaction) -> Result<()> {
        let state_guard = self.state.read().await;
        validate_transaction(&state_guard, &tx)?;
        drop(state_guard);
        self.tx_sender.send(NodeCommand::SendTransaction(tx))?;
        Ok(())
    }

    pub fn scan_addresses(&self, addresses: &[[u8; 32]], start: u64, end: u64) -> Result<Vec<ScannedCoin>> {
        let store = crate::storage::BatchStore::new(&self.batches_path)?;
        let mut found = Vec::new();
        for height in start..end {
            if let Some(batch) = store.load(height)? {
                for tx in &batch.transactions {
                    if let Transaction::Reveal { outputs, .. } = tx {
                        for out in outputs {
                            if addresses.contains(&out.address) {
                                found.push(ScannedCoin {
                                    address: out.address,
                                    value: out.value,
                                    salt: out.salt,
                                    coin_id: out.coin_id(),
                                    height,
                                });
                            }
                        }
                    }
                }
                for cb in &batch.coinbase {
                    if addresses.contains(&cb.address) {
                        found.push(ScannedCoin {
                            address: cb.address,
                            value: cb.value,
                            salt: cb.salt,
                            coin_id: cb.coin_id(),
                            height,
                        });
                    }
                }
            }
        }
        Ok(found)
    }
    pub fn scan_mss_index(&self, master_pk: &[u8; 32], height: u64) -> Result<u64> {
        let store = crate::storage::BatchStore::new(&self.batches_path)?;
        let mut max_idx: u64 = 0;
        for h in 0..height {
            if let Some(batch) = store.load(h)? {
                max_idx = max_idx.max(scan_txs_for_mss_index(&batch.transactions, master_pk));
            }
        }
        Ok(max_idx)
    }
    
}

pub fn scan_txs_for_mss_index(txs: &[Transaction], master_pk: &[u8; 32]) -> u64 {
    let mut max_idx: u64 = 0;
    for tx in txs {
        if let Transaction::Reveal { inputs, signatures, .. } = tx {
            for (input, sig_bytes) in inputs.iter().zip(signatures.iter()) {
                if input.owner_pk == *master_pk && sig_bytes.len() > wots::SIG_SIZE {
                    if let Ok(mss_sig) = mss::MssSignature::from_bytes(sig_bytes) {
                        // leaf_index is 0-based, so next usable = leaf_index + 1
                        max_idx = max_idx.max(mss_sig.leaf_index + 1);
                    }
                }
            }
        }
    }
    max_idx
}

impl Node {
    pub async fn new(
        data_dir: PathBuf,
        is_mining: bool,
        listen_addr: Multiaddr,
        bootstrap_peers: Vec<Multiaddr>,
    ) -> Result<Self> {
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

                    let mut mining_midstate = state.midstate;
                    for cb in &genesis_coinbase {
                        mining_midstate = hash_concat(&mining_midstate, &cb.coin_id());
                    }

                    let mut nonce = 0u64;
                    let extension = loop {
                        let ext = create_extension(mining_midstate, nonce);
                        if ext.final_hash < state.target {
                            tracing::info!("Found deterministic genesis nonce: {}", nonce);
                            break ext;
                        }
                        nonce += 1;
                    };

                    let genesis_batch = Batch {
                        prev_midstate: state.midstate,
                        transactions: vec![],
                        extension,
                        coinbase: genesis_coinbase,
                        timestamp: state.timestamp,
                        target: state.target,
                    };
                    storage.save_batch(0, &genesis_batch)?;
                    apply_batch(&mut state, &genesis_batch)?;
                    storage.save_state(&state)?;
                    tracing::info!("Genesis batch applied, height now {}", state.height);
                }
                Some(batch) => {
                    if state.height == 0 {
                        apply_batch(&mut state, &batch)?;
                        storage.save_state(&state)?;
                    }
                }
            }
        }

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

        // Load or generate libp2p keypair
        let keypair = match load_keypair(&data_dir) {
            Some(kp) => {
                tracing::info!("Loaded peer keypair");
                kp
            }
            None => {
                let kp = Keypair::generate_ed25519();
                save_keypair(&data_dir, &kp);
                tracing::info!("Generated new peer keypair");
                kp
            }
        };

        let network = MidstateNetwork::new(keypair, listen_addr, bootstrap_peers).await?;

        let mut recent_headers = Vec::new();
        let window = DIFFICULTY_ADJUSTMENT_INTERVAL as u64 * 2;
        let start_height = state.height.saturating_sub(window);

        for h in start_height..state.height {
            if let Some(batch) = storage.load_batch(h)? {
                recent_headers.push(BatchHeader {
                    midstate: batch.extension.final_hash,
                    timestamp: batch.timestamp,
                    target: batch.target,
                    height: h + 1,
                });
            }
        }

        Ok(Self {
            state,
            mempool: Mempool::new(),
            storage,
            network,
            metrics: Metrics::new(),
            is_mining,
            recent_headers, // Updated field
            orphan_batches: HashMap::new(),
            sync_in_progress: false,
            sync_requested_up_to: 0,
            mining_seed,
            data_dir,
            chain_history: Vec::new(),
            max_reorg_depth: 100,
        })
    }

    pub fn create_handle(&self) -> (NodeHandle, tokio::sync::mpsc::UnboundedReceiver<NodeCommand>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = NodeHandle {
            state: Arc::new(RwLock::new(self.state.clone())),
            mempool_size: Arc::new(RwLock::new(self.mempool.len())),
            mempool_txs: Arc::new(RwLock::new(self.mempool.transactions().to_vec())),
            peer_addrs: Arc::new(RwLock::new(Vec::new())),
            tx_sender: tx,
            batches_path: self.data_dir.join("db").join("batches"),
        };
        (handle, rx)
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
        let mut sync_poll_interval = time::interval(Duration::from_secs(30));
        let mut mempool_prune_interval = time::interval(Duration::from_secs(60));

        // Initial sync: ask all peers for their height
        if self.network.peer_count() > 0 {
            tracing::info!("Requesting chain state from {} peer(s)...", self.network.peer_count());
            self.sync_in_progress = true;
            for peer in self.network.connected_peers() {
                self.network.send(peer, Message::GetState);
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        loop {
            tokio::select! {
                _ = mine_interval.tick() => {
                    if self.is_mining && !self.sync_in_progress {
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
                    *handle.peer_addrs.write().await = self.network.peer_addrs();
                }
                _ = metrics_interval.tick() => {
                    self.metrics.report();
                }
                _ = mempool_prune_interval.tick() => {
                    self.mempool.prune_invalid(&self.state);
                }
                _ = sync_poll_interval.tick() => {
                    if let Some(peer) = self.network.random_peer() {
                        self.network.send(peer, Message::GetState);
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
                event = self.network.next_event() => {
                    match event {
                        NetworkEvent::MessageReceived { peer, message, channel } => {
                            if let Err(e) = self.handle_message(peer, message, channel).await {
                                tracing::warn!("Error from peer {}: {}", peer, e);
                            }
                        }
                        NetworkEvent::PeerConnected(peer) => {
                            tracing::info!("Peer connected: {}", peer);
                            self.network.send(peer, Message::GetState);
                        }
                        NetworkEvent::PeerDisconnected(peer) => {
                            tracing::info!("Peer disconnected: {}", peer);
                        }
                    }
                }
            }
        }
    }

    fn send_response(&mut self, channel: Option<ResponseChannel<Message>>, msg: Message) {
        if let Some(ch) = channel {
            self.network.respond(ch, msg);
        }
    }

    fn ack(&mut self, channel: Option<ResponseChannel<Message>>) {
        self.send_response(channel, Message::Pong { nonce: 0 });
    }

    async fn handle_message(
        &mut self,
        from: PeerId,
        msg: Message,
        channel: Option<ResponseChannel<Message>>,
    ) -> Result<()> {
        match msg {
            Message::Transaction(tx) => {
                self.ack(channel);
                self.handle_new_transaction(tx, Some(from)).await?;
            }
            Message::Batch(batch) => {
                self.ack(channel);
                self.handle_new_batch(batch, Some(from)).await?;
            }
            Message::GetState => {
                let response = Message::StateInfo {
                    height: self.state.height,
                    depth: self.state.depth,
                    midstate: self.state.midstate,
                };
                self.send_response(channel, response);
            }
            // ── Fix C (Fix 1): Replaced StateInfo handler ────────────────
            Message::StateInfo { height, depth, midstate } => {
                self.ack(channel);
                tracing::debug!("Peer {} state: height={} depth={}", from, height, depth);

                if midstate == self.state.midstate && height == self.state.height {
                    self.sync_in_progress = false;
                } else if depth > self.state.depth || height > self.state.height {
                    let start = self.state.height.saturating_sub(self.max_reorg_depth.min(self.state.height));
                    let count = (height.saturating_sub(start) + 1).min(MAX_GETBATCHES_COUNT);
                    tracing::info!("Syncing from peer {} (requesting {} batches from {})", from, count, start);
                    self.sync_in_progress = true;
                    self.sync_requested_up_to = height;
                    self.network.send(from, Message::GetBatches { start_height: start, count });
                }
            }
            Message::Ping { nonce } => {
                self.send_response(channel, Message::Pong { nonce });
            }
            Message::Pong { .. } => {
                self.ack(channel);
            }
            Message::GetAddr => {
                let addrs = self.network.peer_addrs();
                // Convert to SocketAddr for protocol compat — send empty if can't parse
                let socket_addrs: Vec<std::net::SocketAddr> = addrs
                    .iter()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                self.send_response(channel, Message::Addr(socket_addrs));
            }
            Message::Addr(_addrs) => {
                self.ack(channel);
            }
            Message::GetBatches { start_height, count } => {
                let count = count.min(MAX_GETBATCHES_COUNT);
                let end = (start_height + count).min(self.state.height);
                match self.storage.load_batches(start_height, end) {
                    Ok(tagged) => {
                        let actual_start = tagged.first().map(|(h, _)| *h).unwrap_or(start_height);
                        let batches: Vec<Batch> = tagged.into_iter().map(|(_, b)| b).collect();
                        self.send_response(channel, Message::Batches {
                            start_height: actual_start,
                            batches,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load batches: {}", e);
                        self.send_response(channel, Message::Batches {
                            start_height,
                            batches: vec![],
                        });
                    }
                }
            }
            Message::Batches { start_height: batch_start, batches } => {
                self.ack(channel);
                if !batches.is_empty() {
                    self.handle_batches_response(batch_start, batches, from).await?;
                }
            }
        }
        Ok(())
    }

    async fn handle_new_transaction(&mut self, tx: Transaction, from: Option<PeerId>) -> Result<()> {
        match self.mempool.add(tx.clone(), &self.state) {
            Ok(_) => {
                self.metrics.inc_transactions_processed();
                self.network.broadcast_except(from, Message::Transaction(tx));
                Ok(())
            }
            Err(e) => {
                self.metrics.inc_invalid_transactions();
                Err(e)
            }
        }
    }

    // ── Fix A: New find_fork_point method ────────────────────────────────
    /// Find the height where our chain and the alternative chain diverge
    /// by comparing batches side-by-side.
    fn find_fork_point(&self, alternative_batches: &[Batch], alt_start_height: u64) -> Result<u64> {
        for (i, alt_batch) in alternative_batches.iter().enumerate() {
            let height = alt_start_height + i as u64;
            match self.storage.load_batch(height)? {
                Some(our_batch) => {
                    if our_batch.extension.final_hash != alt_batch.extension.final_hash {
                        tracing::info!(
                            "Fork point: height {} — our final_hash={} alt final_hash={}",
                            height,
                            hex::encode(&our_batch.extension.final_hash[..8]),
                            hex::encode(&alt_batch.extension.final_hash[..8])
                        );
                        return Ok(height);
                    }
                }
                None => {
                    tracing::info!("Fork point: height {} — we have no batch here (alt extends us)", height);
                    return Ok(height);
                }
            }
        }

        anyhow::bail!("No divergence found — chains are identical over the received range")
    }

    // ── Fix D: Simplified evaluate_alternative_chain ─────────────────────
    async fn evaluate_alternative_chain(
        &mut self,
        fork_height: u64,
        alternative_batches: &[Batch],
        _from: PeerId,
    ) -> Result<Option<(State, Vec<(u64, [u8; 32], Batch)>)>> {
        // Simplified fork_state derivation
        let fork_state = if fork_height == 0 {
            State::genesis().0
        } else if fork_height <= self.state.height.saturating_sub(self.max_reorg_depth) {
            tracing::warn!("Fork at {} exceeds max reorg depth, rejecting", fork_height);
            return Ok(None);
        } else {
            self.rebuild_state_at_height(fork_height)?
        };

        let mut candidate_state = fork_state;
        let mut new_history = Vec::new();
        
        // CHANGED: Use headers instead of states
        let mut recent_headers = Vec::new();

        let window_size = (DIFFICULTY_ADJUSTMENT_INTERVAL as usize) * 2;
        let start_height = fork_height.saturating_sub(window_size as u64);

        for h in start_height..fork_height {
            if let Some(batch) = self.storage.load_batch(h)? {
                recent_headers.push(BatchHeader {
                    midstate: batch.extension.final_hash,
                    timestamp: batch.timestamp,
                    target: batch.target,
                    height: h + 1,
                });
            }
        }

        for (i, batch) in alternative_batches.iter().enumerate() {
            // Push the *current* state's header before applying the new batch
            recent_headers.push(candidate_state.header());
            if recent_headers.len() > window_size {
                recent_headers.remove(0);
            }

            if batch.prev_midstate != candidate_state.midstate {
                tracing::warn!(
                    "Alternative chain broken at batch index {} (height {})",
                    i, fork_height + i as u64
                );
                return Ok(None);
            }

            match apply_batch(&mut candidate_state, batch) {
                Ok(_) => {
                    // Pass headers to adjust_difficulty
                    candidate_state.target = adjust_difficulty(&candidate_state, &recent_headers);
                    new_history.push((
                        fork_height + i as u64,
                        candidate_state.midstate,
                        batch.clone(),
                    ));
                }
                Err(e) => {
                    tracing::warn!("Alternative chain invalid at height {}: {}", fork_height + i as u64, e);
                    return Ok(None);
                }
            }
        }

        if candidate_state.depth > self.state.depth {
            tracing::warn!(
                "REORG DETECTED: Alternative chain has more work (depth {} > {})",
                candidate_state.depth, self.state.depth
            );
            Ok(Some((candidate_state, new_history)))
        } else {
            tracing::debug!(
                "Rejecting alternative chain: insufficient work (depth {} <= {})",
                candidate_state.depth, self.state.depth
            );
            Ok(None)
        }
    }

    // ── Fix C (Fix 3): Added sync_in_progress = false at end ────────────
    fn perform_reorg(
        &mut self,
        new_state: State,
        new_history: Vec<(u64, [u8; 32], Batch)>,
    ) -> Result<()> {
        tracing::warn!(
            "PERFORMING REORG: height {} -> {}, depth {} -> {}",
            self.state.height, new_state.height,
            self.state.depth, new_state.depth
        );

        let fork_height = new_history.first().map(|(h, _, _)| *h).unwrap_or(0);

        let abandoned_history: Vec<_> = self.chain_history
            .iter()
            .skip_while(|(h, _, _)| *h < fork_height)
            .cloned()
            .collect();

        for (height, _, batch) in &new_history {
            if let Err(e) = self.storage.save_batch(*height, batch) {
                tracing::error!("Failed to save reorg batch at height {}: {}", height, e);
            }
        }

        self.state = new_state;
        self.chain_history.retain(|(h, _, _)| *h < fork_height);
        self.chain_history.extend(new_history);

        // Rebuild headers cache from disk
        self.recent_headers.clear();
        let window = (DIFFICULTY_ADJUSTMENT_INTERVAL * 2) as u64;
        let start = self.state.height.saturating_sub(window);
        
        for h in start..self.state.height {
             if let Some(batch) = self.storage.load_batch(h)? {
                 self.recent_headers.push(BatchHeader {
                     midstate: batch.extension.final_hash,
                     timestamp: batch.timestamp,
                     target: batch.target,
                     height: h + 1,
                 });
             }
        }

        self.state.target = adjust_difficulty(&self.state, &self.recent_headers);

        for (_, _, batch) in abandoned_history {
            self.mempool.re_add(batch.transactions, &self.state);
        }
        self.mempool.prune_invalid(&self.state);
        self.metrics.inc_reorgs();
        self.storage.save_state(&self.state)?;

        self.sync_in_progress = false;

        Ok(())
    }

    fn rebuild_state_at_height(&self, target_height: u64) -> Result<State> {
        let mut state = State::genesis().0;
        let mut recent_headers = Vec::new();

        for h in 0..target_height {
            if let Some(batch) = self.storage.load_batch(h)? {
                recent_headers.push(state.header());
                if recent_headers.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                    recent_headers.remove(0);
                }
                apply_batch(&mut state, &batch)?;
                state.target = adjust_difficulty(&state, &recent_headers);
            } else {
                anyhow::bail!("Missing batch at height {} needed for reorg", h);
            }
        }

        Ok(state)
    }

    async fn handle_new_batch(&mut self, batch: Batch, from: Option<PeerId>) -> Result<()> {
        let mut candidate_state = self.state.clone();
        match apply_batch(&mut candidate_state, &batch) {
            Ok(_) => {
                let best = choose_best_state(&self.state, &candidate_state);
                let is_reorg = best.height == self.state.height &&
                               best.midstate != self.state.midstate;

                if best.height > self.state.height || is_reorg {
                    if is_reorg {
                        tracing::warn!("REORG at height {}", self.state.height);
                        self.metrics.inc_reorgs();
                    }

                    // Use recent_headers and header()
                    self.recent_headers.push(self.state.header());
                    if self.recent_headers.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                        self.recent_headers.remove(0);
                    }
                    let pre_height = self.state.height;
                    self.state = candidate_state;
                    self.storage.save_batch(pre_height, &batch)?;
                    
                    // Pass recent_headers
                    self.state.target = adjust_difficulty(&self.state, &self.recent_headers);
                    
                    self.metrics.inc_batches_processed();
                    self.mempool.prune_invalid(&self.state);

                    self.chain_history.push((
                        pre_height,
                        self.state.midstate,
                        batch.clone(),
                    ));
                    if self.chain_history.len() > self.max_reorg_depth as usize {
                        self.chain_history.remove(0);
                    }

                    self.network.broadcast_except(from, Message::Batch(batch));
                    tracing::info!("Applied new batch from peer, height now {}", self.state.height);
                    self.try_apply_orphans().await;
                }
                Ok(())
            }
            Err(e) => {
                let err_str = e.to_string();

                if err_str.contains("Block parent mismatch") ||
                   err_str.contains("not found") ||
                   err_str.contains("No matching commitment")
                {
                    tracing::info!("Received orphan/fork block (parent mismatch).");

                    const ORPHAN_LIMIT: usize = 64;
                    if self.orphan_batches.len() >= ORPHAN_LIMIT {
                        self.orphan_batches.clear();
                    }

                    let estimated_height = self.state.height + 1;
                    self.orphan_batches.insert(estimated_height, batch);

                    if !self.sync_in_progress {
                        if let Some(peer) = from {
                            self.sync_in_progress = true;
                            self.network.send(peer, Message::GetState);
                        }
                    }
                }
                Ok(())
            }
        }
    }

    // ── Fix B: Rewritten handle_batches_response ─────────────────────────
    async fn handle_batches_response(&mut self, batch_start_height: u64, batches: Vec<Batch>, from: PeerId) -> Result<()> {
        if batches.is_empty() { return Ok(()); }
        tracing::info!("Received {} batch(es) starting at height {} from peer {}", batches.len(), batch_start_height, from);

        // Try 1: Do they extend our current chain directly?
        let mut test_state = self.state.clone();
        if apply_batch(&mut test_state, &batches[0]).is_ok() {
            return self.process_linear_extension(batches, from).await;
        }

        // Try 2: Do any of them extend our chain? (we might already have some)
        for (i, batch) in batches.iter().enumerate() {
            let mut candidate = self.state.clone();
            if apply_batch(&mut candidate, batch).is_ok() {
                tracing::info!("Found linear extension at batch index {}", i);
                return self.process_linear_extension(batches[i..].to_vec(), from).await;
            }
        }

        // Try 3: This is a fork. Find the fork point.
        match self.find_fork_point(&batches, batch_start_height) {
            Ok(fork_height) => {
                tracing::info!("Fork detected at height {}", fork_height);
                let offset = fork_height.saturating_sub(batch_start_height) as usize;
                let relevant = if offset < batches.len() { &batches[offset..] } else { &batches };

                match self.evaluate_alternative_chain(fork_height, relevant, from).await {
                    Ok(Some((new_state, new_history))) => {
                        self.perform_reorg(new_state, new_history)?;
                    }
                    Ok(None) => {
                        tracing::debug!("Alternative chain rejected (insufficient work)");
                    }
                    Err(e) => {
                        tracing::warn!("Error evaluating fork: {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::debug!("Could not find fork point: {}", e);
            }
        }

        // Always clear sync flag after processing batch response
        self.sync_in_progress = false;
        Ok(())
    }

    // ── Fix C (Fix 2): Added sync_in_progress clear at end ──────────────
    async fn process_linear_extension(&mut self, batches: Vec<Batch>, from: PeerId) -> Result<()> {
        let mut applied = 0;
        for batch in batches {
            let mut candidate = self.state.clone();
            if apply_batch(&mut candidate, &batch).is_ok() {
                // CHANGED: recent_headers
                self.recent_headers.push(self.state.header());
                if self.recent_headers.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                    self.recent_headers.remove(0);
                }
                self.storage.save_batch(candidate.height - 1, &batch)?;
                self.state = candidate;
                
                // CHANGED: recent_headers
                self.state.target = adjust_difficulty(&self.state, &self.recent_headers);
                
                self.metrics.inc_batches_processed();

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

            if self.state.height >= self.sync_requested_up_to {
                self.sync_in_progress = false;
            } else {
                // Still behind — request more batches from same peer
                let start = self.state.height;
                let count = (self.sync_requested_up_to.saturating_sub(start) + 1).min(MAX_GETBATCHES_COUNT);
                tracing::info!("Continuing sync from peer {} (requesting {} batches from {})", from, count, start);
                self.network.send(from, Message::GetBatches { start_height: start, count });
            }
        } else {
            self.sync_in_progress = false;
        }

        Ok(())
    }

    async fn try_apply_orphans(&mut self) {
        let mut applied = 0;
        loop {
            let height = self.state.height;
            let batch = match self.orphan_batches.remove(&height) {
                Some(b) => b,
                None => break,
            };

            let mut candidate = self.state.clone();
            match apply_batch(&mut candidate, &batch) {
                Ok(_) => {
                    // recent_headers
                    self.recent_headers.push(self.state.header());
                    if self.recent_headers.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
                        self.recent_headers.remove(0);
                    }
                    self.storage.save_batch(candidate.height - 1, &batch).ok();
                    self.state = candidate;
                    
                    // recent_headers
                    self.state.target = adjust_difficulty(&self.state, &self.recent_headers);
                    
                    self.metrics.inc_batches_processed();
                    self.mempool.prune_invalid(&self.state);
                    applied += 1;
                }
                Err(e) => {
                    tracing::warn!("Orphan batch at {} still invalid: {}", height, e);
                    break;
                }
            }
        }

        if applied > 0 {
            tracing::info!("Applied {} orphan batch(es)", applied);
        }

        let cutoff = self.state.height.saturating_sub(10);
        self.orphan_batches.retain(|&h, _| h > cutoff);

        while self.orphan_batches.len() > MAX_ORPHAN_BATCHES {
            if let Some(&oldest) = self.orphan_batches.keys().min() {
                self.orphan_batches.remove(&oldest);
            }
        }
    }

    fn generate_coinbase(&self, height: u64, total_fees: u64) -> Vec<CoinbaseOutput> {
        let reward = block_reward(height);
        let total_value = reward + total_fees;
        let denominations = decompose_value(total_value);

        let mining_seed = self.mining_seed; // Extract seed here

        denominations.into_par_iter()
            .enumerate()
            .map(move |(i, value)| { // Add move
                let seed = coinbase_seed(&mining_seed, height, i as u64); // Use local variable
                let owner_pk = wots::keygen(&seed);
                let address = compute_address(&owner_pk);
                let salt = coinbase_salt(&mining_seed, height, i as u64); // Use local variable
                CoinbaseOutput { address, value, salt }
            })
            .collect()
    }

    fn log_coinbase(&self, height: u64, total_fees: u64) {
        let reward = block_reward(height);
        let total_value = reward + total_fees;
        let denominations = decompose_value(total_value);
        let log_path = self.data_dir.join("coinbase_seeds.jsonl");

        let mining_seed = self.mining_seed; // Extract seed here

        let entries: Vec<String> = denominations.into_par_iter()
            .enumerate()
            .map(move |(i, value)| { // Add move
                let seed = coinbase_seed(&mining_seed, height, i as u64);
                let owner_pk = wots::keygen(&seed);
                let address = compute_address(&owner_pk);
                let salt = coinbase_salt(&mining_seed, height, i as u64);
                let coin_id = compute_coin_id(&address, value, &salt);
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
        if self.sync_in_progress {
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

        let coinbase = self.generate_coinbase(pre_mine_height, total_fees);
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

        if self.state.height != pre_mine_height || self.state.midstate != pre_mine_midstate {
            tracing::warn!("State advanced during mining. Restoring transactions.");
            self.mempool.re_add(transactions, &self.state);
            return Ok(());
        }

        let current_time = state::current_timestamp();
        let block_timestamp = current_time.max(self.state.timestamp + 1);

        let batch = Batch {
            prev_midstate: pre_mine_midstate,
            transactions,
            extension,
            coinbase: coinbase.clone(),
            timestamp: block_timestamp,
            target: self.state.target,
        };

        self.recent_headers.push(self.state.header());
        if self.recent_headers.len() > DIFFICULTY_ADJUSTMENT_INTERVAL as usize * 2 {
            self.recent_headers.remove(0);
        }

        match apply_batch(&mut self.state, &batch) {
            Ok(_) => {
                self.storage.save_batch(pre_mine_height, &batch)?;
                self.storage.save_state(&self.state)?;
                self.state.target = adjust_difficulty(&self.state, &self.recent_headers);
                self.metrics.inc_batches_mined();
                self.network.broadcast(Message::Batch(batch));
                self.log_coinbase(pre_mine_height, total_fees);

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

// ── Keypair persistence ─────────────────────────────────────────────────────

fn load_keypair(data_dir: &PathBuf) -> Option<Keypair> {
    let path = data_dir.join("peer_key");
    let mut bytes = std::fs::read(&path).ok()?;
    let ed_kp = libp2p::identity::ed25519::Keypair::try_from_bytes(&mut bytes).ok()?;
    Some(Keypair::from(ed_kp))
}

fn save_keypair(data_dir: &PathBuf, keypair: &Keypair) {
    let path = data_dir.join("peer_key");
    if let Ok(ed_kp) = keypair.clone().try_into_ed25519() {
        let _ = std::fs::write(&path, ed_kp.to_bytes());
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::core::extension::create_extension;
    use crate::core::mss;
    use crate::core::types::hash;
    
    // Helper to create a bare-bones node for testing internal logic
    pub(crate) async fn create_test_node(dir: &std::path::Path) -> Node {
        let _keypair = libp2p::identity::Keypair::generate_ed25519();
        // Bind to port 0 to let OS assign a random available port
        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        // Initialize node (this will create genesis if needed)
        Node::new(dir.to_path_buf(), false, listen, vec![]).await.unwrap()
    }

    #[tokio::test]
    async fn test_find_fork_point_logic() {
        // 1. Setup
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        // 2. Build a local chain: Genesis (0) -> B1 -> B2
        // Note: Node::new already created Genesis at height 0.
        let genesis_batch = node.storage.load_batch(0).unwrap().unwrap();
        let midstate_0 = genesis_batch.extension.final_hash;

        // Create Batch 1
        let ext1 = create_extension(midstate_0, 100);
        let midstate_1 = ext1.final_hash;
        let batch1 = Batch {
            prev_midstate: midstate_0,
            transactions: vec![],
            extension: ext1,
            coinbase: vec![],
            timestamp: 1000,
            target: node.state.target,
        };
        node.storage.save_batch(1, &batch1).unwrap();

        // Create Batch 2
        let ext2 = create_extension(midstate_1, 200);
        let midstate_2 = ext2.final_hash;
        let batch2 = Batch {
            prev_midstate: midstate_1,
            transactions: vec![],
            extension: ext2,
            coinbase: vec![],
            timestamp: 1010,
            target: node.state.target,
        };
        node.storage.save_batch(2, &batch2).unwrap();

        // Manually update node state to reflect tip is at height 2
        node.state.height = 2;
        node.state.midstate = midstate_2;

        // ---------------------------------------------------------
        // Case A: Linear Extension (No Fork)
        // ---------------------------------------------------------
        // Remote sends Batch 3, which builds on Batch 2.
        let ext3 = create_extension(midstate_2, 300);
        let batch3 = Batch {
            prev_midstate: midstate_2,
            transactions: vec![],
            extension: ext3,
            coinbase: vec![],
            timestamp: 1020,
            target: node.state.target,
        };

        // We ask: "Where does this batch attach?"
        // Since batch3.prev_midstate == node.midstate (midstate_2),
        // find_fork_point should identify it attaches at height 3.
        let fork_h = node.find_fork_point(&[batch3.clone()], 3).unwrap();
        assert_eq!(fork_h, 3, "Linear extension should attach at height 3");

        // ---------------------------------------------------------
        // Case B: Deep Fork
        // ---------------------------------------------------------
        // Remote sends Batch 2', which builds on Batch 1 (forks off before Batch 2).
        // prev_midstate = midstate_1.
        let ext2_prime = create_extension(midstate_1, 999); // Different nonce -> different hash
        let batch2_prime = Batch {
            prev_midstate: midstate_1,
            transactions: vec![],
            extension: ext2_prime,
            coinbase: vec![],
            timestamp: 1011,
            target: node.state.target,
        };

        // We ask: "Where does this batch attach?"
        // It connects to Batch 1 (height 1).
        // So the fork point (the first new block) is at height 2.
        let fork_h = node.find_fork_point(&[batch2_prime], 2).unwrap();
        assert_eq!(fork_h, 2, "Deep fork should be detected at height 2");

        // ---------------------------------------------------------
        // Case C: Genesis Fork
        // ---------------------------------------------------------
        // Remote sends a batch that builds on Genesis (height 0), replacing Batch 1.
        let ext1_prime = create_extension(midstate_0, 555);
        let batch1_prime = Batch {
            prev_midstate: midstate_0,
            transactions: vec![],
            extension: ext1_prime,
            coinbase: vec![],
            timestamp: 1001,
            target: node.state.target,
        };

        // It connects to Genesis (height 0).
        // Fork point is height 1.
        let fork_h = node.find_fork_point(&[batch1_prime], 1).unwrap();
        assert_eq!(fork_h, 1, "Genesis fork should be detected at height 1");
    }
    
    #[test]
    fn scan_txs_for_mss_index_finds_max() {
        // Create an MSS keypair and sign a few messages
        let seed = hash(b"test mss scan seed");
        let mut keypair = mss::keygen(&seed, 4).unwrap();
        let master_pk = keypair.public_key();

        // Sign 5 messages (uses leaves 0-4)
        let mut txs = Vec::new();
        for i in 0..5u8 {
            let msg = hash(&[i]);
            let sig = keypair.sign(&msg).unwrap();
            let sig_bytes = sig.to_bytes();

            // Build a minimal Reveal tx with this MSS signature
            let tx = Transaction::Reveal {
                inputs: vec![InputReveal {
                    owner_pk: master_pk,
                    value: 1,
                    salt: [i; 32],
                }],
                signatures: vec![sig_bytes],
                outputs: vec![OutputData {
                    address: [0xAA; 32],
                    value: 1,
                    salt: [i; 32],
                }],
                salt: [0; 32],
            };
            txs.push(tx);
        }

        // scan should find max index = 5 (leaf 4 used, so next = 5)
        let max_idx = scan_txs_for_mss_index(&txs, &master_pk);
        assert_eq!(max_idx, 5);
    }

    #[test]
    fn scan_txs_for_mss_index_ignores_other_keys() {
        let seed1 = hash(b"key1");
        let seed2 = hash(b"key2");
        let mut kp1 = mss::keygen(&seed1, 4).unwrap();
        let kp2 = mss::keygen(&seed2, 4).unwrap();

        let msg = hash(b"msg");
        let sig = kp1.sign(&msg).unwrap();

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                owner_pk: kp1.public_key(),
                value: 1,
                salt: [0; 32],
            }],
            signatures: vec![sig.to_bytes()],
            outputs: vec![OutputData {
                address: [0xAA; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        // Scanning for kp2's key should find nothing
        assert_eq!(scan_txs_for_mss_index(&[tx.clone()], &kp2.public_key()), 0);
        // Scanning for kp1's key should find index 1
        assert_eq!(scan_txs_for_mss_index(&[tx], &kp1.public_key()), 1);
    }

    #[test]
    fn scan_txs_mss_recovery_simulation() {
        // Simulates: use indices 0-4, then "restore backup" to 0,
        // scan should report 5 so wallet can jump ahead
        let seed = hash(b"recovery sim");
        let mut keypair = mss::keygen(&seed, 4).unwrap();
        let master_pk = keypair.public_key();

        let mut txs = Vec::new();
        for i in 0..5u8 {
            let msg = hash(&[i]);
            let sig = keypair.sign(&msg).unwrap();
            txs.push(Transaction::Reveal {
                inputs: vec![InputReveal {
                    owner_pk: master_pk,
                    value: 1,
                    salt: [i; 32],
                }],
                signatures: vec![sig.to_bytes()],
                outputs: vec![OutputData {
                    address: [0xBB; 32],
                    value: 1,
                    salt: [i; 32],
                }],
                salt: [0; 32],
            });
        }

        let chain_max = scan_txs_for_mss_index(&txs, &master_pk);
        assert_eq!(chain_max, 5, "should find highest used index + 1");

        // Simulate backup restore: keypair at index 0
        let mut restored = mss::keygen(&seed, 4).unwrap();
        assert_eq!(restored.next_leaf, 0);

        // Apply recovery logic (same as wallet sync)
        const SAFETY_MARGIN: u64 = 20;
        if chain_max >= restored.next_leaf {
            restored.set_next_leaf(chain_max + SAFETY_MARGIN);
        }
        assert_eq!(restored.next_leaf, 25, "should be 5 + 20 safety margin");
    }

    #[test]
    fn scan_txs_mss_mempool_race() {
        // Simulate: tx with index 10 in mempool (unmined)
        let seed = hash(b"mempool race");
        let mut keypair = mss::keygen(&seed, 5).unwrap(); // height 5 = 32 leaves

        // Advance to leaf 10 by signing 10 messages
        for i in 0..10u8 {
            keypair.sign(&hash(&[i])).unwrap();
        }

        // Sign one more (leaf index 10) — this is the "mempool tx"
        let msg = hash(b"mempool tx");
        let sig = keypair.sign(&msg).unwrap();
        assert_eq!(sig.leaf_index, 10);

        let mempool_tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                owner_pk: keypair.public_key(),
                value: 1,
                salt: [0; 32],
            }],
            signatures: vec![sig.to_bytes()],
            outputs: vec![OutputData {
                address: [0xCC; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        let mempool_max = scan_txs_for_mss_index(&[mempool_tx], &keypair.public_key());
        assert_eq!(mempool_max, 11, "should account for leaf 10 → next = 11");

        // Simulate restore from backup at index 0
        let mut restored = mss::keygen(&seed, 5).unwrap();

        const SAFETY_MARGIN: u64 = 20;
        let remote_idx = mempool_max; // in real code: max(chain_max, mempool_max)
        if remote_idx >= restored.next_leaf {
            restored.set_next_leaf(remote_idx + SAFETY_MARGIN);
        }
        assert_eq!(restored.next_leaf, 31, "should be 11 + 20 safety margin");
    }
    
    #[test]
    fn scan_txs_skips_wots_signatures() {
        // WOTS sigs are exactly 576 bytes — scanner should ignore them
        let seed = hash(b"wots not mss");
        let pk = wots::keygen(&seed);
        let msg = hash(b"test");
        let sig = wots::sign(&seed, &msg);
        let sig_bytes = wots::sig_to_bytes(&sig);
        assert_eq!(sig_bytes.len(), wots::SIG_SIZE);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                owner_pk: pk,
                value: 1,
                salt: [0; 32],
            }],
            signatures: vec![sig_bytes],
            outputs: vec![OutputData {
                address: [0xAA; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        // Should return 0 — no MSS signatures found
        assert_eq!(scan_txs_for_mss_index(&[tx], &pk), 0);
    }

    #[test]
    fn scan_txs_empty_returns_zero() {
        let pk = hash(b"any key");
        assert_eq!(scan_txs_for_mss_index(&[], &pk), 0);
    }

    #[test]
    fn mss_recovery_does_not_roll_back() {
        // If local index is ahead of remote, should NOT decrease
        let seed = hash(b"no rollback");
        let mut keypair = mss::keygen(&seed, 4).unwrap();

        // Advance local to leaf 10
        for i in 0..10u8 {
            keypair.sign(&hash(&[i])).unwrap();
        }
        assert_eq!(keypair.next_leaf, 10);

        // Remote only knows about 3 usages
        let remote_idx: u64 = 3;

        // Recovery logic should NOT roll back
        const SAFETY_MARGIN: u64 = 20;
        if remote_idx >= keypair.next_leaf {
            keypair.set_next_leaf(remote_idx + SAFETY_MARGIN);
        }
        // Should still be 10, not 23
        assert_eq!(keypair.next_leaf, 10);
    }

    #[test]
    fn scan_txs_corrupt_mss_sig_no_panic() {
        // Garbage bytes longer than WOTS SIG_SIZE should not crash
        let pk = hash(b"corrupt test");
        let garbage = vec![0xFFu8; wots::SIG_SIZE + 100]; // longer than WOTS → tries MSS parse

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                owner_pk: pk,
                value: 1,
                salt: [0; 32],
            }],
            signatures: vec![garbage],
            outputs: vec![OutputData {
                address: [0xAA; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        // Should not panic, just return 0
        assert_eq!(scan_txs_for_mss_index(&[tx], &pk), 0);
    }
}

// ── Complex Integration Tests ───────────────────────────────────────────────
#[cfg(test)]
mod complex_tests {
    use super::*;
    use tempfile::tempdir;
    use crate::core::types::hash;

    // --- Test Helpers ---

    /// Creates a node instance isolated in a temp directory.
    /// 
    /// Note: Node::new() automatically creates and applies the genesis block (Height 0).
    /// Therefore, any node returned by this function is sitting at Height 1, ready 
    /// to mine or receive Block 1.
    async fn create_test_node(dir: &std::path::Path) -> Node {
        let _keypair = libp2p::identity::Keypair::generate_ed25519();
        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        
        // Initialize node (creates Genesis internally)
        Node::new(dir.to_path_buf(), false, listen, vec![]).await.unwrap()
    }

    /// specific helper to manually construct a valid batch structure.
    /// 
    /// This bypasses the main mining loop but performs all necessary cryptographic 
    /// operations to create a valid block:
    /// 1. Simulates transaction application to calculate the correct `midstate`.
    /// 2. Generates the correct deterministic Coinbase outputs.
    /// 3. Finds a valid Proof-of-Work nonce for the *real* target of the previous state.
    fn make_valid_batch(prev_state: &State, timestamp_offset: u64, transactions: Vec<Transaction>) -> Batch {
        let timestamp = prev_state.timestamp + timestamp_offset;
        let mut midstate_after_txs = prev_state.midstate;
        let mut tx_fees = 0;

        // 1. Simulate applying transactions to get the post-tx midstate
        for tx in &transactions {
            tx_fees += tx.fee();
            if let Transaction::Commit { commitment, .. } = tx {
                midstate_after_txs = hash_concat(&midstate_after_txs, commitment);
            } else if let Transaction::Reveal { inputs, outputs, salt, .. } = tx {
                let mut hasher = blake3::Hasher::new();
                for i in inputs { hasher.update(&i.coin_id()); }
                for o in outputs { hasher.update(&o.coin_id()); }
                hasher.update(salt);
                let tx_hash = *hasher.finalize().as_bytes();
                midstate_after_txs = hash_concat(&midstate_after_txs, &tx_hash);
            }
        }

        // 2. Generate Coinbase outputs (deterministic based on previous midstate/height)
        let reward = crate::core::block_reward(prev_state.height);
        let total_value = reward + tx_fees;
        let mining_seed = prev_state.midstate; 
        
        let denominations = crate::core::types::decompose_value(total_value);
        let coinbase: Vec<CoinbaseOutput> = denominations.iter().enumerate().map(|(i, &val)| {
             let seed = coinbase_seed(&mining_seed, prev_state.height, i as u64);
             let pk = wots::keygen(&seed);
             let addr = compute_address(&pk);
             let salt = coinbase_salt(&mining_seed, prev_state.height, i as u64);
             CoinbaseOutput { address: addr, value: val, salt }
        }).collect();

        // 3. Update midstate with coinbase
        for cb in &coinbase {
            midstate_after_txs = hash_concat(&midstate_after_txs, &cb.coin_id());
        }

        // 4. Mine a valid nonce.
        let target = prev_state.target;
        let mut nonce = 0u64;
        let extension = loop {
            let ext = create_extension(midstate_after_txs, nonce);
            if ext.final_hash < target {
                break ext;
            }
            nonce += 1;
        };

        Batch {
            prev_midstate: prev_state.midstate,
            transactions,
            extension,
            coinbase,
            timestamp,
            target,
        }
    }

    // --- Tests ---

    /// Verifies that the node stops mining activities when a sync is triggered.
    /// This prevents wasting resources extending a chain that might be obsolete.
    #[tokio::test]
    async fn mining_pauses_during_sync() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;
        
        // Node starts at Height 1 (Genesis applied)
        assert_eq!(node.state.height, 1);

        // 1. Verify mining works normally
        // The node is in "mining mode" but we call try_mine() manually to step through it.
        node.try_mine().await.expect("Mining should succeed");
        assert_eq!(node.state.height, 2);

        // 2. Set sync flag manually (simulating a "GetBatches" request sent to a peer)
        node.sync_in_progress = true;

        // 3. Try mine again. It should return Ok() immediately without doing work.
        node.try_mine().await.expect("Should return Ok implicitly");
        
        // 4. Assert height did NOT increase
        assert_eq!(node.state.height, 2, "Mining should be paused during sync");

        // 5. Unset flag (simulating sync completion) and verify mining resumes
        node.sync_in_progress = false;
        node.try_mine().await.expect("Mining should succeed");
        assert_eq!(node.state.height, 3, "Mining should resume after sync");
    }

    /// Verifies the critical safety mechanism of Reorgs:
    /// When switching to a longer chain, transactions in the abandoned blocks 
    /// must be returned to the mempool so they are not lost.
    #[tokio::test]
    async fn reorg_restores_mempool_transactions() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;
        // Start at H=1

        // --- Chain A: H=1 -> B1(H=2) -> B2(H=3) ---
        // This is the "local" chain the node initially follows.
        
        // Block 1 (Height 1->2)
        let b1 = make_valid_batch(&node.state, 10, vec![]);
        node.handle_new_batch(b1.clone(), None).await.unwrap();
        assert_eq!(node.state.height, 2);
        
        // Create a unique transaction that will be mined into Block 2 of Chain A.
        let commit_hash = hash(b"tx_on_chain_a");
        let mut nonce = 0u64;
        loop {
            let h = hash_concat(&commit_hash, &nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            nonce += 1;
        }
        let valid_tx = Transaction::Commit { commitment: commit_hash, spam_nonce: nonce };

        // Block 2 (Height 2->3) contains the transaction
        let b2 = make_valid_batch(&node.state, 10, vec![valid_tx.clone()]);
        node.handle_new_batch(b2.clone(), None).await.unwrap();

        assert_eq!(node.state.height, 3);
        assert!(node.state.commitments.contains(&commit_hash), "Tx confirmed in Chain A");
        assert_eq!(node.mempool.len(), 0, "Mempool empty after mining");

        // --- Chain B: Fork at H=2 ---
        // Chain B: H=1 -> B1(H=2) -> B2'(H=3) -> B3'(H=4)
        // This is the "competitor" chain. It is longer, so it should trigger a reorg.
        // It forks *after* B1.
        
        // We need to reconstruct the state at H=1 (Genesis) to build the fork.
        let mut state_at_2 = State::genesis().0; 
        
        // REASONING: We cannot simply use State::genesis() because we need to load 
        // the *exact* genesis batch saved by the node to ensure midstates align.
        let genesis_batch = node.storage.load_batch(0).unwrap().unwrap();
        apply_batch(&mut state_at_2, &genesis_batch).unwrap(); // H=1
        
        // Apply B1 (shared history)
        apply_batch(&mut state_at_2, &b1).unwrap(); // H=2

        // B2' (Alternative block at H=3). Empty, does NOT have the transaction.
        let b2_prime = make_valid_batch(&state_at_2, 20, vec![]); 
        let mut state_at_3_prime = state_at_2.clone();
        apply_batch(&mut state_at_3_prime, &b2_prime).unwrap();

        // B3' (extends B2', making Chain B longer)
        let b3_prime = make_valid_batch(&state_at_3_prime, 10, vec![]);

        // --- Submit Chain B ---
        let peer = PeerId::random();
        
        // We simulate receiving the fork batches.
        // The node detects the fork at H=2, evaluates Chain B, sees it is longer (Len 4 vs 3),
        // and performs the reorg.
        node.handle_batches_response(2, vec![b2_prime, b3_prime], peer).await.unwrap();

        // --- Assertions ---
        assert_eq!(node.state.height, 4, "Node should have switched to longer chain (H=4)");
        
        // The transaction `commit_hash` was in Block 2 (Chain A), but NOT in Block 2' (Chain B).
        // Therefore, it is no longer in the confirmed state.
        assert!(!node.state.commitments.contains(&commit_hash), "Tx from abandoned chain should be gone from state");
        
        // However, the reorg logic should have rescued it from the abandoned block 
        // and placed it back into the mempool.
        assert_eq!(node.mempool.len(), 1, "Mempool should have 1 restored tx");
        let mempool_txs = node.mempool.transactions();
        if let Transaction::Commit { commitment, .. } = mempool_txs[0] {
            assert_eq!(commitment, commit_hash);
        } else {
            panic!("Restored transaction mismatch");
        }
    }

    /// Verifies that the node can ingest a linear sequence of blocks during sync.
    #[tokio::test]
    async fn sync_ingests_batches_and_completes() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;
        
        // Snapshot the start state (H=1) so we can reset the node later to simulate being behind.
        let start_state = node.state.clone();

        // Generate 50 blocks extending H=1.
        // We use the node's internal state to generate them validly, then revert.
        let mut batches = Vec::new();
        for _ in 0..50 {
            let b = make_valid_batch(&node.state, 10, vec![]);
            apply_batch(&mut node.state, &b).unwrap();
            
            // Note: Storage keys are usually (height-1) for batches? 
            // Actually storage saves batch X at index X. 
            // If Genesis is batch_0 (H=0), apply_batch makes H=1.
            // Next block is batch_1.
            // We save mainly to ensure `make_valid_batch` has consistent history if it looked at storage.
            node.storage.save_batch(node.state.height - 1, &b).unwrap();
            batches.push(b);
        }
        
        assert_eq!(node.state.height, 51);

        // Reset node to H=1 (the "behind" node).
        node.state = start_state;
        
        // Simulate sync state
        node.sync_in_progress = true;
        node.sync_requested_up_to = 51;
        
        // Feed the 50 batches we generated.
        let peer = PeerId::random();
        node.handle_batches_response(1, batches, peer).await.unwrap();

        // Verify the node caught up.
        assert_eq!(node.state.height, 51);
        assert_eq!(node.sync_in_progress, false, "Sync should complete automatically");
    }
}
