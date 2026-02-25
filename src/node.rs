use crate::core::*;
use crate::core::types::compute_address;
use crate::core::types::{CoinbaseOutput, BatchHeader};
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::{mine_extension, create_extension};
use crate::core::transaction::{apply_transaction, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::mix::{MixManager, MixPhase, MixStatusSnapshot};
use crate::network::{Message, MidstateNetwork, NetworkEvent, MAX_GETBATCHES_COUNT};
use crate::storage::Storage;
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::mss;
use crate::core::wots;
use crate::sync::Syncer;
use anyhow::{bail, Result};
use libp2p::{request_response::ResponseChannel, PeerId, Multiaddr, identity::Keypair};
use std::collections::{HashMap, HashSet, VecDeque};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time;
use rayon::prelude::*;

const MAX_ORPHAN_BATCHES: usize = 256;
const SYNC_TIMEOUT_SECS: u64 = 300;
const CATCH_UP_THRESHOLD: u64 = 20;

/// Non-blocking sync session driven by the main event loop.
/// Replaces the old blocking `Syncer::sync_via_network` which hijacked the
/// network and dropped unrelated messages.
struct SyncSession {
    peer: PeerId,
    peer_height: u64,
    peer_depth: u64,
    phase: SyncPhase,
    started_at: std::time::Instant,
}

enum SyncPhase {
    /// Waiting for a requested state snapshot
    WaitingForSnapshot {
        target_height: u64,
    },
    /// Downloading headers. If fast-forwarding, it holds the snapshot to verify against.
    Headers {
        accumulated: Vec<BatchHeader>,
        cursor: u64,
        snapshot: Option<Box<State>>,
    },
    VerifyingHeaders,
    /// Headers verified, now downloading batches from fork_height forward.
    Batches {
        headers: Vec<BatchHeader>,
        fork_height: u64,
        candidate_state: State,
        cursor: u64,
        new_history: Vec<(u64, [u8; 32], Batch)>,
        is_fast_forward: bool,
    },
}

pub struct Node {
    state: State,
    mempool: Mempool,
    storage: Storage,
    network: MidstateNetwork,
    syncer: Syncer,
    metrics: Metrics,
    mining_threads: Option<usize>,
    recent_headers: VecDeque<u64>,
    orphan_batches: HashMap<[u8; 32], Vec<Batch>>,
    orphan_order: VecDeque<[u8; 32]>,
    sync_in_progress: bool,
    sync_requested_up_to: u64,
    sync_session: Option<SyncSession>,
    mining_seed: [u8; 32],
    data_dir: PathBuf,
    chain_history: VecDeque<(u64, [u8; 32])>,
    finality: crate::core::finality::FinalityEstimator,
    known_pex_addrs: HashSet<String>,
    connected_peers: HashSet<PeerId>,
    // Background mining concurrency
    mining_cancel: Option<Arc<AtomicBool>>,
    mined_batch_rx: tokio::sync::mpsc::UnboundedReceiver<Batch>,
    mined_batch_tx: tokio::sync::mpsc::UnboundedSender<Batch>,
    // CoinJoin mix coordinator
    mix_manager: Arc<RwLock<MixManager>>,
    /// Reveals waiting for their Commit to be mined.
    /// Key: commitment hash, Value: (mix_id, Reveal transaction)
    pending_mix_reveals: HashMap<[u8; 32], ([u8; 32], Transaction)>,
    cmd_tx: Option<tokio::sync::mpsc::UnboundedSender<NodeCommand>>,
}

#[derive(Clone)]
pub struct NodeHandle {
    state: Arc<RwLock<State>>,
    safe_depth: Arc<RwLock<u64>>,
    mempool_size: Arc<RwLock<usize>>,
    mempool_txs: Arc<RwLock<Vec<Transaction>>>,
    peer_addrs: Arc<RwLock<Vec<String>>>,
    tx_sender: tokio::sync::mpsc::UnboundedSender<NodeCommand>,
    pub batches_path: PathBuf,
    pub mix_manager: Arc<RwLock<MixManager>>,
}

pub enum NodeCommand {
    SendTransaction(Transaction),
    SubmitMixTransaction { mix_id: [u8; 32], tx: Transaction },
    // --- P2P Mix Coordination Commands ---
    BroadcastMixAnnounce { mix_id: [u8; 32], denomination: u64 },
    SendMixJoin { coordinator: PeerId, mix_id: [u8; 32], input: InputReveal, output: OutputData, signature: Vec<u8> },
    SendMixFee { coordinator: PeerId, mix_id: [u8; 32], input: InputReveal },
    SendMixSign { coordinator: PeerId, mix_id: [u8; 32], input_index: usize, signature: Vec<u8> },
    BroadcastMixProposal { mix_id: [u8; 32], proposal: crate::wallet::coinjoin::MixProposal, peers: Vec<PeerId> },
    FinishSyncHeaders { peer: PeerId, headers: Vec<BatchHeader>, is_valid: bool, snapshot: Option<Box<State>> },
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

    /// Returns the current dynamic safe depth calculated by the Bayesian finality estimator.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use midstate::node::NodeHandle;
    /// # async fn example(handle: NodeHandle) {
    /// let depth = handle.get_safe_depth().await;
    /// println!("Transactions older than {} blocks are final.", depth);
    /// # }
    /// ```
    pub async fn get_safe_depth(&self) -> u64 {
        *self.safe_depth.read().await
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
                            if addresses.contains(&out.address()) {
                                if let Some(c_id) = out.coin_id() {
                                    found.push(ScannedCoin {
                                        address: out.address(),
                                        value: out.value(),
                                        salt: out.salt(),
                                        coin_id: c_id,
                                        height,
                                    });
                                }
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

    // ── CoinJoin mix helpers ────────────────────────────────────────────

    pub async fn mix_create(&self, denomination: u64, min_participants: usize) -> Result<[u8; 32]> {
        let mut mgr = self.mix_manager.write().await;
        let mix_id = mgr.create_session(denomination, min_participants)?;
        drop(mgr); // Drop lock before sending over channel
        
        // Broadcast the announcement to the network
        self.tx_sender.send(NodeCommand::BroadcastMixAnnounce { mix_id, denomination })?;
        Ok(mix_id)
    }

    pub async fn mix_register(
        &self, mix_id: [u8; 32], input: InputReveal, output: OutputData, signature: Vec<u8> // <-- Added signature
    ) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        // Pass the signature to the MixManager
        mgr.register(&mix_id, input.clone(), output.clone(), &signature, None)?;

        if !is_coord {
            if let Some(peer) = coord_peer {
                self.tx_sender.send(NodeCommand::SendMixJoin { 
                    coordinator: peer, 
                    mix_id, 
                    input, 
                    output, 
                    signature // <-- Added signature
                })?;
            }
        } else if let Some(proposal) = mgr.try_finalize(&mix_id)? {
            // We are the coordinator and the mix is full. Broadcast proposal!
            let peers = mgr.remote_participants(&mix_id);
            self.tx_sender.send(NodeCommand::BroadcastMixProposal { mix_id, proposal, peers })?;
        }
        Ok(())
    }

    pub async fn mix_set_fee(&self, mix_id: [u8; 32], input: InputReveal) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        mgr.set_fee_input(&mix_id, input.clone(), None)?;

        if !is_coord {
            if let Some(peer) = coord_peer {
                self.tx_sender.send(NodeCommand::SendMixFee { coordinator: peer, mix_id, input })?;
            }
        } else if let Some(proposal) = mgr.try_finalize(&mix_id)? {
            let peers = mgr.remote_participants(&mix_id);
            self.tx_sender.send(NodeCommand::BroadcastMixProposal { mix_id, proposal, peers })?;
        }
        Ok(())
    }

    pub async fn mix_sign(&self, mix_id: [u8; 32], input_index: usize, signature: Vec<u8>) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        mgr.add_signature(&mix_id, input_index, signature.clone())?;

        if !is_coord {
            if let Some(peer) = coord_peer {
                self.tx_sender.send(NodeCommand::SendMixSign { coordinator: peer, mix_id, input_index, signature })?;
            }
        } else if let Some(tx) = mgr.try_build_transaction(&mix_id)? {
            mgr.set_phase(&mix_id, MixPhase::CommitSubmitted);
            self.tx_sender.send(NodeCommand::SubmitMixTransaction { mix_id, tx })?;
        }
        Ok(())
    }

    pub async fn mix_status(&self, mix_id: [u8; 32]) -> Option<MixStatusSnapshot> {
        let mgr = self.mix_manager.read().await;
        mgr.status(&mix_id)
    }

    pub async fn mix_list(&self) -> Vec<MixStatusSnapshot> {
        let mgr = self.mix_manager.read().await;
        mgr.list_sessions()
    }

    pub async fn mix_find_input_index(&self, mix_id: [u8; 32], coin_id: [u8; 32]) -> Option<usize> {
        let mgr = self.mix_manager.read().await;
        mgr.find_input_index(&mix_id, &coin_id)
    }
    
}

pub fn scan_txs_for_mss_index(txs: &[Transaction], master_pk: &[u8; 32]) -> u64 {
    let mut max_idx: u64 = 0;
    for tx in txs {
        if let Transaction::Reveal { inputs, witnesses, .. } = tx {
            for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                if let Some(owner_pk) = input.predicate.owner_pk() {
                    if &owner_pk == master_pk {
                        let Witness::ScriptInputs(wit_inputs) = witness; 
                            if let Some(sig_bytes) = wit_inputs.first() {
                                if sig_bytes.len() > wots::SIG_SIZE {
                                    if let Ok(mss_sig) = mss::MssSignature::from_bytes(sig_bytes) {
                                        max_idx = max_idx.max(mss_sig.leaf_index.saturating_add(1));
                                    }
                                }
                            }                        
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
        mining_threads: Option<usize>,
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
                    let mut temp_coins = state.coins.clone();
                    for cb in &genesis_coinbase {
                        let coin_id = cb.coin_id();
                        mining_midstate = hash_concat(&mining_midstate, &coin_id);
                        temp_coins.insert(coin_id);
                    }

                    // --- NEW: Calculate Genesis State Root ---
                    let state_root = hash_concat(&temp_coins.root(), &state.chain_mmr.root());
                    mining_midstate = hash_concat(&mining_midstate, &state_root);
                    // -----------------------------------------

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
                        state_root, // NEW
                    };
                    storage.save_batch(0, &genesis_batch)?;
                    apply_batch(&mut state, &genesis_batch, &[])?;
                    state.target = adjust_difficulty(&state);
                    storage.save_state(&state)?;
                    tracing::info!("Genesis batch applied, height now {}", state.height);
                }
                Some(batch) => {
                    if state.height == 0 {
                        apply_batch(&mut state, &batch, &[])?;
                        state.target = adjust_difficulty(&state);
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

        let mut recent_headers = VecDeque::new();
        let window = DIFFICULTY_LOOKBACK as u64;
        let start_height = state.height.saturating_sub(window);

        for h in start_height..state.height {
            if let Some(batch) = storage.load_batch(h)? {
                recent_headers.push_back(batch.timestamp);
            }
        }

        let (mined_batch_tx, mined_batch_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            state,
            mempool: Mempool::new(),
            storage: storage.clone(),
            syncer: Syncer::new(storage),
            network,
            metrics: Metrics::new(),
            mining_threads,
            recent_headers,
            orphan_batches: HashMap::new(),
            orphan_order: VecDeque::new(),
            sync_in_progress: false,
            sync_requested_up_to: 0,
            mining_seed,
            data_dir,
            chain_history: VecDeque::new(),
            //lets assume a hostile environment where 80% of new connections are malicious. 
            finality: crate::core::finality::FinalityEstimator::new(2, 8), 
            sync_session: None,
            known_pex_addrs: HashSet::new(),
            connected_peers: HashSet::new(),
            mining_cancel: None,
            mined_batch_rx,
            mined_batch_tx,
            mix_manager: Arc::new(RwLock::new(MixManager::new())),
            pending_mix_reveals: HashMap::new(),
            cmd_tx: None,
        })
    }

    pub fn create_handle(&self) -> (NodeHandle, tokio::sync::mpsc::UnboundedReceiver<NodeCommand>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = NodeHandle {
            state: Arc::new(RwLock::new(self.state.clone())),
            safe_depth: Arc::new(RwLock::new(self.finality.calculate_safe_depth(1e-6))),
            mempool_size: Arc::new(RwLock::new(self.mempool.len())),
            mempool_txs: Arc::new(RwLock::new(self.mempool.transactions_cloned())),
            peer_addrs: Arc::new(RwLock::new(Vec::new())),
            tx_sender: tx,
            batches_path: self.data_dir.join("db").join("batches"),
            mix_manager: Arc::clone(&self.mix_manager),
        };
        (handle, rx)
    }

    /// Abort any active background mining task so the event loop can adopt a new chain.
    fn cancel_mining(&mut self) {
        if let Some(cancel) = self.mining_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
            // Drain any batch the thread may have sent before seeing the flag
            while self.mined_batch_rx.try_recv().is_ok() {}
            tracing::debug!("Cancelled active mining task for network update.");
        }
    }

    pub async fn run(
        mut self,
        handle: NodeHandle,
        mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<NodeCommand>,
    ) -> Result<()> {
        self.cmd_tx = Some(handle.tx_sender.clone());
        let mut mine_interval = time::interval(Duration::from_secs(5));
        let mut save_interval = time::interval(Duration::from_secs(10));
        let mut ui_interval = time::interval(Duration::from_secs(1));
        let mut metrics_interval = time::interval(Duration::from_secs(30));
        let mut sync_poll_interval = time::interval(Duration::from_secs(30));
        let mut mempool_prune_interval = time::interval(Duration::from_secs(60));
        let mut sync_timeout_interval = time::interval(Duration::from_secs(5));
        let mut pex_interval = time::interval(Duration::from_secs(120));
        let mut connection_maintenance = time::interval(Duration::from_secs(15));
        const TARGET_OUTBOUND_PEERS: usize = 8;
        
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
                    // Check mining_threads.is_some() instead of is_mining
                    if self.mining_threads.is_some() && !self.sync_in_progress && self.mining_cancel.is_none() {
                        if let Err(e) = self.spawn_mining_task() {
                            tracing::error!("Mining error: {}", e);
                        }
                    }
                }
                Some(batch) = self.mined_batch_rx.recv() => {
                    if let Err(e) = self.handle_mined_batch(batch).await {
                        tracing::error!("Failed to process mined batch: {}", e);
                    }
                }
                _ = save_interval.tick() => {
                    if let Err(e) = self.storage.save_state(&self.state) {
                        tracing::error!("Failed to save state: {}", e);
                    }
                    
                    // Periodically save a snapshot for fast-sync peers
                    if self.state.height > 0 && self.state.height % PRUNE_DEPTH == 0 {
                        if let Err(e) = self.storage.save_state_snapshot(self.state.height, &self.state) {
                            tracing::warn!("Failed to save state snapshot: {}", e);
                        }
                    }
                    
                    self.maybe_prune_checkpoints();
                }
                _ = ui_interval.tick() => {
                    *handle.state.write().await = self.state.clone();
                    *handle.safe_depth.write().await = self.finality.calculate_safe_depth(1e-6);
                    *handle.mempool_size.write().await = self.mempool.len();
                    *handle.mempool_txs.write().await = self.mempool.transactions_cloned();
                    *handle.peer_addrs.write().await = self.network.peer_addrs();
                }
                _ = metrics_interval.tick() => {
                    self.metrics.report();
                }
                _ = mempool_prune_interval.tick() => {
                    // CoinJoin: clean up stale mix sessions
                    self.mix_manager.write().await.cleanup();
                }
                _ = sync_poll_interval.tick() => {
                    if let Some(peer) = self.network.random_peer() {
                        self.network.send(peer, Message::GetState);
                    }
                }
                _ = sync_timeout_interval.tick() => {
                    if let Some(session) = &self.sync_session {
                        if session.started_at.elapsed().as_secs() > SYNC_TIMEOUT_SECS {
                            self.abort_sync_session("timed out");
                        }
                    }
                }
                _ = pex_interval.tick() => {
                    if let Some(peer) = self.network.random_peer() {
                        tracing::debug!("PEX: requesting addrs from {}", peer);
                        self.network.send(peer, Message::GetAddr);
                    }
                }
                
                _ = connection_maintenance.tick() => {
                    let current_outbound = self.network.outbound_peer_count();
                    if current_outbound < TARGET_OUTBOUND_PEERS {
                        let needed = TARGET_OUTBOUND_PEERS - current_outbound;
                        
                        // Pick random addresses from our known pool
                        use rand::seq::IteratorRandom;
                        let mut rng = rand::thread_rng();
                        let to_dial: Vec<String> = self.known_pex_addrs
                            .iter()
                            .choose_multiple(&mut rng, needed)
                            .into_iter()
                            .cloned()
                            .collect();

                        for addr in to_dial {
                            //tracing::info!("Maintenance: Dialing {} to maintain outbound ratio", addr); //spams output
                            self.network.dial_addr(&addr);
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
                        NodeCommand::SubmitMixTransaction { mix_id, tx } => {
                            if let Err(e) = self.handle_mix_transaction(mix_id, tx).await {
                                tracing::error!("Failed to submit mix transaction: {}", e);
                                let mut mgr = self.mix_manager.write().await;
                                mgr.set_phase(&mix_id, MixPhase::Failed(e.to_string()));
                            }
                        }
                        NodeCommand::BroadcastMixAnnounce { mix_id, denomination } => {
                            self.network.broadcast(Message::MixAnnounce { mix_id, denomination });
                        }
                        NodeCommand::SendMixJoin { coordinator, mix_id, input, output, signature } => {
                            self.network.send(coordinator, Message::MixJoin { mix_id, input, output, signature });
                        }
                        NodeCommand::SendMixFee { coordinator, mix_id, input } => {
                            self.network.send(coordinator, Message::MixFee { mix_id, input });
                        }
                        NodeCommand::SendMixSign { coordinator, mix_id, input_index, signature } => {
                            self.network.send(coordinator, Message::MixSign { mix_id, input_index, signature });
                        }
                        
                        NodeCommand::FinishSyncHeaders { peer, headers, is_valid, snapshot } => {
                            if let Err(e) = self.process_verified_headers(peer, headers, is_valid, snapshot).await {
                                tracing::warn!("Failed to process verified headers: {}", e);
                            }
                        }
                        NodeCommand::BroadcastMixProposal { mix_id, proposal, peers } => {
                            for peer in peers {
                                self.network.send(peer, Message::MixProposal {
                                    mix_id,
                                    inputs: proposal.inputs.clone(),
                                    outputs: proposal.outputs.clone(),
                                    salt: proposal.salt,
                                    commitment: proposal.commitment,
                                });
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
                            if !self.connected_peers.insert(peer) {
                                // Already connected via another transport — skip
                                continue;
                            }
                            tracing::info!("Peer connected: {}", peer);
                            self.network.send(peer, Message::GetState);
                            self.network.send(peer, Message::GetAddr);
                        }
                        NetworkEvent::PeerDisconnected(peer) => {
                            self.connected_peers.remove(&peer);
                            tracing::info!("Peer disconnected: {}", peer);
                            if self.sync_session.as_ref().map_or(false, |s| s.peer == peer) {
                                self.abort_sync_session("sync peer disconnected");
                            }
                            
                            // Fail any mixes relying on this peer
                            self.mix_manager.write().await.handle_peer_disconnect(peer);
                        }
                        // Eclipse Defense Purge 
                        NetworkEvent::OutgoingConnectionFailed(addr_str) => {
                            if self.known_pex_addrs.remove(&addr_str) {
                                tracing::info!("Eclipse Defense: Purged unreachable PEX address: {}", addr_str);
                            }
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

            Message::StateInfo { height, depth, midstate } => {
                self.ack(channel);
                tracing::debug!("Peer {} state: height={} depth={}", from, height, depth);

                if midstate == self.state.midstate && height == self.state.height {
                    self.sync_in_progress = false;
                    self.sync_session = None;
                } else if depth > self.state.depth || height > self.state.height
                    || (depth == self.state.depth && midstate < self.state.midstate)
                {
                    // Don't restart sync if we're already syncing from this peer
                    if self.sync_session.as_ref().map_or(false, |s| s.peer == from) {
                        tracing::debug!("Already syncing from this peer, ignoring duplicate StateInfo");
                    } else {
                        let gap = height.saturating_sub(self.state.height);
                        if gap > 0 && gap <= CATCH_UP_THRESHOLD {
                            tracing::info!(
                                "Peer {} blocks ahead (h={} vs h={}), requesting batches directly",
                                gap, height, self.state.height
                            );
                            self.cancel_mining();
                            self.sync_in_progress = true;
                            self.sync_requested_up_to = height;
                            let count = gap.min(MAX_GETBATCHES_COUNT);
                            self.network.send(from, Message::GetBatches {
                                start_height: self.state.height,
                                count,
                            });
                        } else if gap > PRUNE_DEPTH {
                            // The gap is massive -> Trigger Fast-Forward
                            self.start_fast_forward_sync(from, height, depth);
                        } else {
                            // Standard sync
                            self.start_sync_session(from, height, depth);
                        }
                    }
                } else {
                    tracing::debug!(
                        "Peer {} at equal/lower depth (h={}, d={}) with different chain, resuming mining",
                        from, height, depth
                    );
                    self.sync_in_progress = false;
                }
            }
            
            Message::Ping { nonce } => {
                self.send_response(channel, Message::Pong { nonce });
            }
            Message::Pong { .. } => {
                self.ack(channel);
            }
            Message::GetAddr => {
                let addrs = self.network.pex_addrs();
                tracing::debug!("PEX: sending {} addrs to {}", addrs.len(), from);
                self.send_response(channel, Message::Addr(addrs));
            }
            Message::Addr(addrs) => {
                self.ack(channel);
                let mut new_count = 0;
                
                // 1. Cap intake: Do not let one peer flood the table in a single message
                for addr_str in addrs.into_iter().take(20) {
                    if !self.known_pex_addrs.contains(&addr_str) {
                        
                        // 2. Churn: If the table is full, evict a random address
                        // to prevent deterministic eclipse attacks.
                        if self.known_pex_addrs.len() >= 1_000 {
                            let skip = rand::random::<usize>() % self.known_pex_addrs.len();
                            let victim = self.known_pex_addrs.iter().nth(skip).cloned().unwrap();
                            self.known_pex_addrs.remove(&victim);
                        }
                        
                        self.known_pex_addrs.insert(addr_str);
                        new_count += 1;
                    }
                }
                
                if new_count > 0 {
                    tracing::debug!("PEX: saved {} new addrs from {}", new_count, from);
                }
            }
            Message::GetBatches { start_height, count } => {
                let count = count.min(MAX_GETBATCHES_COUNT);
                let end = (start_height + count).min(self.state.height);
                match self.storage.load_batches(start_height, end) {
                    Ok(tagged) => {
                        let actual_start = tagged.first().map(|(h, _)| *h).unwrap_or(start_height);
                        
                        // Byte-bounded packing to prevent MAX_MSG_SIZE crashes ---
                        let mut batches = Vec::new();
                        let mut current_size = 0u64;
                        
                        // Target 8 MB to leave 2 MB of safety margin for bincode/network overhead
                        // against the 10 MB MAX_MSG_SIZE limit.
                        const MAX_PAYLOAD_BYTES: u64 = 8_000_000; 

                        for (_, batch) in tagged {
                            // bincode::serialized_size is very fast, it doesn't allocate
                            let batch_size = bincode::serialized_size(&batch).unwrap_or(0);
                            
                            // Always include at least 1 batch to prevent stalling the sync,
                            // otherwise break if adding this batch exceeds our safe limit.
                            if !batches.is_empty() && current_size + batch_size > MAX_PAYLOAD_BYTES {
                                tracing::debug!(
                                    "Truncating GetBatches response to {} blocks ({} bytes) to fit message limits.", 
                                    batches.len(), current_size
                                );
                                break;
                            }
                            
                            batches.push(batch);
                            current_size += batch_size;
                        }

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
                    // If we have an active sync session in the Batches phase from this
                    // peer, route through the sync state machine instead of the normal
                    // batch handler.
                    let is_sync_batches = self.sync_session.as_ref().map_or(false, |s| {
                        s.peer == from && matches!(s.phase, SyncPhase::Batches { .. })
                    });
                    if is_sync_batches {
                        if let Err(e) = self.handle_sync_batches(from, batches).await {
                            tracing::warn!("Error processing sync batches: {}", e);
                            self.abort_sync_session("batch processing error");
                        }
                    } else {
                        self.handle_batches_response(batch_start, batches, from).await?;
                    }
                }
            }
            Message::GetHeaders { start_height, count } => {
                let count = count.min(MAX_GETBATCHES_COUNT);
                let end = (start_height + count).min(self.state.height + 1);
                
                match self.storage.batches.load_headers(start_height, end) {
                    Ok(headers) => {
                        self.send_response(channel, Message::Headers { 
                            start_height, 
                            headers 
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load headers: {}", e);
                    }
                }
            }
            Message::Headers { start_height: _, headers } => {
                self.ack(channel);
                if let Err(e) = self.handle_sync_headers(from, headers).await {
                    tracing::warn!("Error processing sync headers: {}", e);
                    self.abort_sync_session("header processing error");
                }
            }

            // ── CoinJoin mix messages ───────────────────────────────────

            Message::MixAnnounce { mix_id, denomination } => {
                self.ack(channel);
                let mut mgr = self.mix_manager.write().await;
                if mgr.get_session_info(&mix_id).is_none() {
                    match mgr.create_joining_session(mix_id, denomination, from) {
                        Ok(()) => tracing::info!(
                            "Joined mix session {} (denom={}) from peer {}",
                            hex::encode(mix_id), denomination, from
                        ),
                        Err(e) => tracing::debug!("Ignoring MixAnnounce: {}", e),
                    }
                }
            }

            Message::MixJoin { mix_id, input, output, signature } => { // <-- Added signature
                self.ack(channel);
                let mut mgr = self.mix_manager.write().await;
                // Pass the signature reference to the MixManager
                match mgr.register(&mix_id, input, output, &signature, Some(from)) {
                    Ok(()) => {
                        tracing::info!("Peer {} joined mix {}", from, hex::encode(mix_id));
                        // Auto-finalize if ready
                        if let Ok(Some(proposal)) = mgr.try_finalize(&mix_id) {
                            let peers = mgr.remote_participants(&mix_id);
                            drop(mgr);
                            // Broadcast proposal to all participants
                            for peer in peers {
                                self.network.send(peer, Message::MixProposal {
                                    mix_id,
                                    inputs: proposal.inputs.clone(),
                                    outputs: proposal.outputs.clone(),
                                    salt: proposal.salt,
                                    commitment: proposal.commitment,
                                });
                            }
                        }
                    }
                    Err(e) => tracing::debug!("MixJoin rejected: {}", e),
                }
            }
            
            Message::MixFee { mix_id, input } => {
                self.ack(channel);
                let mut mgr = self.mix_manager.write().await;
                match mgr.set_fee_input(&mix_id, input, Some(from)) {
                    Ok(()) => {
                        tracing::info!("Peer {} provided fee for mix {}", from, hex::encode(mix_id));
                        // Auto-finalize if ready
                        if let Ok(Some(proposal)) = mgr.try_finalize(&mix_id) {
                            let peers = mgr.remote_participants(&mix_id);
                            drop(mgr);
                            for peer in peers {
                                self.network.send(peer, Message::MixProposal {
                                    mix_id,
                                    inputs: proposal.inputs.clone(),
                                    outputs: proposal.outputs.clone(),
                                    salt: proposal.salt,
                                    commitment: proposal.commitment,
                                });
                            }
                        }
                    }
                    Err(e) => tracing::debug!("MixFee rejected: {}", e),
                }
            }
            
            Message::MixProposal { mix_id, inputs, outputs, salt, commitment } => {
                self.ack(channel);
                let mut mgr = self.mix_manager.write().await;
                match mgr.apply_remote_proposal(&mix_id, inputs, outputs, salt, commitment) {
                    Ok(()) => tracing::info!(
                        "Applied remote mix proposal for {}",
                        hex::encode(mix_id)
                    ),
                    Err(e) => tracing::debug!("MixProposal rejected: {}", e),
                }
            }

            Message::MixSign { mix_id, input_index, signature } => {
                self.ack(channel);
                let mut mgr = self.mix_manager.write().await;
                if let Err(e) = mgr.add_signature(&mix_id, input_index, signature) {
                    tracing::debug!("MixSign rejected: {}", e);
                } else {
                    // Auto-build if all sigs collected
                    if let Ok(Some(tx)) = mgr.try_build_transaction(&mix_id) {
                        tracing::info!("Mix {} complete from p2p signatures", hex::encode(mix_id));
                        mgr.set_phase(&mix_id, MixPhase::CommitSubmitted);
                        drop(mgr);
                        if let Err(e) = self.handle_mix_transaction(mix_id, tx).await {
                            tracing::error!("Failed to submit p2p mix tx: {}", e);
                            self.mix_manager.write().await
                                .set_phase(&mix_id, MixPhase::Failed(e.to_string()));
                        }
                    }
                }
            }
            // ── Fast-Forward Sync Messages ──────────────────────────────
            
            Message::GetStateSnapshot { height } => {
                match self.storage.load_state_snapshot(height) {
                    Ok(Some(state)) => {
                        self.send_response(channel, Message::StateSnapshot {
                            height,
                            state: Box::new(state),
                        });
                    }
                    Ok(None) => {
                        tracing::debug!("Peer {} requested snapshot for height {}, but we don't have it", from, height);
                        self.ack(channel);
                    }
                    Err(e) => {
                        tracing::error!("Failed to load snapshot for height {}: {}", height, e);
                        self.ack(channel);
                    }
                }
            }

            Message::StateSnapshot { height, mut state } => {
                self.ack(channel);

                // --- Rebuild the SMT caches after network deserialization! ---
                // Without this, the node cache is empty due to #[serde(skip)],
                // and the state root validation will fail on the very next block.
                state.coins.rebuild_tree();
                state.commitments.rebuild_tree();
                // -----------------------------------------------------------------

                if let Some(session) = self.sync_session.take() {
                    if session.peer == from {
                        if let SyncPhase::WaitingForSnapshot { target_height } = session.phase {
                            if height == target_height {
                                tracing::info!("Received snapshot at height {}. Downloading {} subsequent headers to verify PoW...", height, session.peer_height - height);
                                
                                self.sync_session = Some(SyncSession {
                                    peer: from,
                                    peer_height: session.peer_height,
                                    peer_depth: session.peer_depth,
                                    phase: SyncPhase::Headers {
                                        accumulated: Vec::new(),
                                        cursor: height,
                                        snapshot: Some(state), // This now holds the rebuilt tree!
                                    },
                                    started_at: std::time::Instant::now(),
                                });

                                let count = 100.min(session.peer_height - height);
                                self.network.send(from, Message::GetHeaders { start_height: height, count });
                                return Ok(());
                            }
                        }
                    }
                    self.sync_session = Some(session); // Put session back if it wasn't a match
                }
            }
        }
        Ok(())
    }

    // ── Non-blocking sync state machine ─────────────────────────────────

    fn start_sync_session(&mut self, peer: PeerId, peer_height: u64, peer_depth: u64) {
        self.cancel_mining();
        tracing::info!(
            "Starting headers-first sync: peer(h={}, d={}) vs us(h={}, d={})",
            peer_height, peer_depth, self.state.height, self.state.depth
        );
        self.sync_in_progress = true;
        self.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::Headers {
                accumulated: Vec::new(),
                cursor: 0,
                snapshot: None, // Added field
            },
            started_at: std::time::Instant::now(),
        });
        // Request first chunk of headers from genesis
        let count = 100.min(peer_height);
        self.network.send(peer, Message::GetHeaders { start_height: 0, count });
    }

    fn start_fast_forward_sync(&mut self, peer: PeerId, peer_height: u64, peer_depth: u64) {
        self.cancel_mining();
        // Snapshots are saved every PRUNE_DEPTH blocks, so we round down to the nearest multiple.
        let snap_height = (peer_height / PRUNE_DEPTH) * PRUNE_DEPTH;
        
        tracing::info!("Fast-forward sync: requesting snapshot at height {}", snap_height);
        self.sync_in_progress = true;
        self.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::WaitingForSnapshot { target_height: snap_height },
            started_at: std::time::Instant::now(),
        });
        self.network.send(peer, Message::GetStateSnapshot { height: snap_height });
    }

    fn abort_sync_session(&mut self, reason: &str) {
        if self.sync_session.is_some() {
            tracing::warn!("Aborting sync session: {}", reason);
        }
        self.sync_session = None;
        self.sync_in_progress = false;
    }

    pub async fn handle_sync_headers(&mut self, from: PeerId, headers: Vec<BatchHeader>) -> Result<()> {
        // Extract state from the session — only accept headers from the sync peer
        let (peer_height, cursor, snapshot) = match &mut self.sync_session {
            Some(s) if s.peer == from => {
                match &mut s.phase {
                    SyncPhase::Headers { cursor, snapshot, .. } => (s.peer_height, *cursor, snapshot.take()),
                    _ => return Ok(()), 
                }
            }
            _ => return Ok(()), 
        };

        if headers.is_empty() {
            self.abort_sync_session("peer sent empty headers");
            return Ok(());
        }

        // Accumulate headers
        let new_cursor = cursor + headers.len() as u64;
        match &mut self.sync_session {
            Some(s) => {
                if let SyncPhase::Headers { accumulated, cursor: c, snapshot: snap_ref } = &mut s.phase {
                    
                    // --- OOM Defense ---
                    if accumulated.len() + headers.len() > 100_000 {
                        self.abort_sync_session("Peer attempted OOM DoS with too many headers");
                        return Ok(());
                    }
                    accumulated.extend(headers);
                    *c = new_cursor;
                    if snapshot.is_some() {
                        *snap_ref = snapshot; // Put the snapshot back for the next iteration
                    }
                }
            }
            _ => unreachable!(),
        }

        if new_cursor < peer_height {
            // Need more headers — request next chunk
            let count = 100.min(peer_height - new_cursor);
            self.network.send(from, Message::GetHeaders { start_height: new_cursor, count });

            //  Reset idle timeout because we are making progress
            if let Some(s) = &mut self.sync_session {
                s.started_at = std::time::Instant::now();
            }
            // ----------------

            return Ok(());
        }


        // All headers received — take ownership of the session data
        let session = self.sync_session.take().unwrap();
        let (all_headers, snapshot) = match session.phase {
            SyncPhase::Headers { accumulated, snapshot, .. } => (accumulated, snapshot),
            _ => unreachable!(),
        };

        let total_headers = all_headers.len(); 
        tracing::info!("Downloaded {} headers, offloading PoW verification...", total_headers);

        let tx = self.cmd_tx.as_ref().unwrap().clone();
        tokio::spawn(async move {
            let mut is_valid = true;

            // 1. Fast sequential check: linkage + difficulty target validation
            for i in 1..all_headers.len() {
                if all_headers[i].prev_midstate != all_headers[i - 1].extension.final_hash {
                    tracing::warn!("Header linkage broken at index {}", i);
                    is_valid = false;
                    break;
                }
                let expected_target = crate::core::state::calculate_target(
                    all_headers[i - 1].height + 1,
                    all_headers[i - 1].timestamp,
                );
                if all_headers[i].target != expected_target {
                    tracing::warn!("Invalid difficulty target at header index {}", i);
                    is_valid = false;
                    break;
                }
            }

            // 2. Heavy PoW check in chunks, yielding to prevent CPU starvation
            if is_valid {
                let mut processed = 0; // <--- Track progress
                
                for chunk in all_headers.chunks(100) {
                    let chunk_owned = chunk.to_vec();
                    
                    let chunk_valid = tokio::task::spawn_blocking(move || {
                        use rayon::prelude::*;
                        use crate::core::extension::verify_extension;
                        
                        chunk_owned.par_iter().all(|header| {
                            verify_extension(
                                header.post_tx_midstate,
                                &header.extension,
                                &header.target,
                            ).is_ok()
                        })
                    }).await.expect("Header verification task panicked");

                    if !chunk_valid {
                        is_valid = false;
                        break;
                    }

                    processed += chunk.len();

                    // --- NEW: Progress Logging ---
                    // Log progress every 1,000 headers, or when we hit 100%
                    if processed % 1000 == 0 || processed == total_headers {
                        let pct = (processed as f64 / total_headers as f64) * 100.0;
                        tracing::info!("Initial Headers Download Progress: Verified {}/{} headers ({:.1}%)", processed, total_headers, pct);
                    }
                    // -----------------------------

                    // Suspend this task for exactly one cycle of the Tokio executor,
                    // letting it answer network pings before Rayon locks the CPUs again.
                    tokio::task::yield_now().await;
                }
            }

        let _ = tx.send(NodeCommand::FinishSyncHeaders {
                peer: from,
                headers: all_headers,
                is_valid,
                snapshot,
            });
        });

        self.sync_session = Some(SyncSession {
            peer: session.peer,
            peer_height: session.peer_height,
            peer_depth: session.peer_depth,
            phase: SyncPhase::VerifyingHeaders,
            started_at: session.started_at,
        });

        Ok(())
    }

    async fn process_verified_headers(
        &mut self,
        peer: PeerId,
        all_headers: Vec<BatchHeader>,
        is_valid: bool,
        snapshot: Option<Box<State>>,
    ) -> Result<()> {
        let session = match self.sync_session.take() {
            Some(s) if s.peer == peer && matches!(s.phase, SyncPhase::VerifyingHeaders) => s,
            other => {
                self.sync_session = other; 
                return Ok(());
            }
        };

        if !is_valid {
            tracing::warn!("Peer header chain invalid (PoW or linkage failed)");
            self.sync_in_progress = false;
            return Ok(());
        }

        // --- Fast-Forward Validation Logic ---
        let (fork_height, candidate_state, is_fast_forward) = if let Some(snap) = snapshot {
            tracing::info!("Headers verified. Cryptographically validating snapshot at height {}...", snap.height);
            
            // THE CRITICAL CHECK: The downloaded headers represent thousands of valid blocks of PoW. 
            // If the first header builds exactly on top of the snapshot's midstate, the snapshot is authentic.
            if all_headers.is_empty() || all_headers[0].prev_midstate != snap.midstate {
                tracing::warn!("Snapshot midstate mismatch! Peer sent fraudulent snapshot. Aborting sync.");
                self.sync_in_progress = false;
                return Ok(());
            }

            // SECURITY: Require a minimum number of post-snapshot headers before
            // trusting a snapshot. Without this, an attacker with modest hash power
            // could forge a snapshot for a fresh node by mining only a handful of
            // valid blocks on top of a fabricated state. PRUNE_DEPTH headers
            // (1,000 blocks × 1M iterations each = 1 billion sequential hashes)
            // makes this prohibitively expensive.
            let min_headers = PRUNE_DEPTH as usize;
            if all_headers.len() < min_headers {
                tracing::warn!(
                    "Fast-forward rejected: only {} headers on top of snapshot (need >= {}). \
                     Peer may be attempting state injection.",
                    all_headers.len(), min_headers
                );
                self.sync_in_progress = false;
                return Ok(());
            }
            
            // Trust the snapshot!
            (snap.height, *snap, true)
        } else {
            // Standard sync path
            let fh = self.syncer.find_fork_point(&all_headers, self.state.height)?;
            let cand = if fh == 0 {
                State::genesis().0
            } else if fh <= self.state.height {
                self.rebuild_state_at_height(fh)?
            } else {
                self.state.clone()
            };
            (fh, cand, false)
        };

        tracing::info!(
            "Fork point/Snapshot at height {}. Will download batches {}..{}",
            fork_height, fork_height, session.peer_height
        );

        if fork_height >= session.peer_height {
            tracing::info!("Already in sync with peer");
            self.sync_in_progress = false;
            return Ok(());
        }

        self.sync_session = Some(SyncSession {
            peer,
            peer_height: session.peer_height,
            peer_depth: session.peer_depth,
            phase: SyncPhase::Batches {
                headers: all_headers,
                fork_height,
                candidate_state,
                cursor: fork_height,
                new_history: Vec::new(),
                is_fast_forward,
            },
            started_at: std::time::Instant::now(),
        });

        let count = (session.peer_height - fork_height).min(MAX_GETBATCHES_COUNT);
        self.network.send(peer, Message::GetBatches { start_height: fork_height, count });

        Ok(())
    }

    async fn handle_sync_batches(&mut self, from: PeerId, batches: Vec<Batch>) -> Result<()> {
        if batches.is_empty() {
            self.abort_sync_session("peer sent empty batches");
            return Ok(());
        }

        // Take the session to work with it
        let mut session = match self.sync_session.take() {
            Some(s) if s.peer == from => s,
            other => {
                self.sync_session = other; // put it back
                return Ok(());
            }
        };

        let (headers, fork_height, candidate_state, cursor, new_history, is_fast_forward) = match &mut session.phase {
            SyncPhase::Batches { headers, fork_height, candidate_state, cursor, new_history, is_fast_forward } => {
                (headers, *fork_height, candidate_state, cursor, new_history, *is_fast_forward)
            }
            _ => {
                self.sync_session = Some(session); // wrong phase, put it back
                return Ok(());
            }
        };

        // Build a bounded sliding window of recent timestamps (mirrors
        // evaluate_alternative_chain).  Both validate_timestamp and
        // adjust_difficulty uses ASERT anchored to genesis,
        // entries, so there is no need to collect every timestamp from genesis.
        let window_size = DIFFICULTY_LOOKBACK as usize;
        let mut recent_ts: VecDeque<u64> = VecDeque::new();


        let header_start_height = if is_fast_forward { fork_height } else { 0 };
        let start_height = cursor.saturating_sub(window_size as u64);

        for h in start_height..*cursor {
            // 1. If we have the header in our downloaded array, use it
            if h >= header_start_height {
                let idx = (h - header_start_height) as usize;
                if let Some(hdr) = headers.get(idx) {
                    recent_ts.push_back(hdr.timestamp);
                    continue;
                }
            }
            
            // 2. Try local storage (for standard sync resolving a local fork)
            if let Ok(Some(batch)) = self.storage.load_batch(h) {
                recent_ts.push_back(batch.timestamp);
                continue;
            }

            // 3. Fallback for fast-forward gaps: use the snapshot's timestamp
            recent_ts.push_back(candidate_state.timestamp);
        }

        // Apply each batch, verifying against the already-validated headers
        for batch in &batches {
            let height = *cursor;
            
            // Map absolute height to relative array index
            let hdr_idx = (height - header_start_height) as usize;

            if hdr_idx >= headers.len() {
                tracing::warn!("Batch height {} exceeds header count {}", height, headers.len());
                self.sync_in_progress = false;
                return Ok(());
            }

            let header = &headers[hdr_idx];

            // Integrity: batch PoW must match the already-verified header
            if batch.extension.final_hash != header.extension.final_hash {
                tracing::warn!("Batch at height {} does not match verified header PoW", height);
                self.sync_in_progress = false;
                return Ok(());
            }
            let calc = batch.header();
            if calc.post_tx_midstate != header.post_tx_midstate {
                tracing::warn!("Batch at height {} tx commitment does not match header", height);
                self.sync_in_progress = false;
                return Ok(());
            }

            // Apply to candidate state (do NOT save to disk yet — if sync
            // aborts before the reorg is committed, premature writes would
            // leave a Frankenstein chain on disk: some heights from the peer,
            // some from our old chain.  Batches are persisted atomically in
            // perform_reorg only after we decide to adopt.)
            recent_ts.push_back(candidate_state.timestamp);
            if recent_ts.len() > window_size {
                recent_ts.pop_front();
            }
            apply_batch(candidate_state, batch, recent_ts.make_contiguous())?;
            candidate_state.target = adjust_difficulty(candidate_state);
            new_history.push((height, candidate_state.midstate, batch.clone()));
            *cursor += 1;
            tokio::task::yield_now().await;
        }

        let current_cursor = *cursor;
        let peer_height = session.peer_height;

        tracing::info!("Applied sync batches up to height {}/{}", current_cursor, peer_height);

        if current_cursor < peer_height {
            // Need more batches — request next chunk
            let count = (peer_height - current_cursor).min(MAX_GETBATCHES_COUNT);
            self.network.send(from, Message::GetBatches { start_height: current_cursor, count });
            // Reset idle timeout
            session.started_at = std::time::Instant::now();
            self.sync_session = Some(session); // put session back
            return Ok(());
        }

        // All batches applied — check if we should adopt this chain
        let (final_state, final_history, is_ff) = match session.phase {
            SyncPhase::Batches { candidate_state, new_history, is_fast_forward, .. } => {
                (candidate_state, new_history, is_fast_forward)
            }
            _ => unreachable!(),
        };

        if final_state.depth > self.state.depth
            || (final_state.depth == self.state.depth && final_state.midstate < self.state.midstate)
        {
            tracing::info!(
                "✓ Sync complete! Adopting chain: height {} -> {}, depth {} -> {}",
                self.state.height, final_state.height,
                self.state.depth, final_state.depth
            );
            self.perform_reorg(final_state, final_history, is_ff)?;
            self.try_apply_orphans().await;
        } else {
            tracing::info!(
                "Sync complete but peer chain has less work (depth {} <= {}), keeping ours",
                final_state.depth, self.state.depth
            );
        }

        self.sync_in_progress = false;
        // sync_session is already None (we took it above)

        // Immediately check if the peer has mined more blocks while we were
        // syncing.  Without this, we'd have to wait for the next broadcast
        // (which will fail because we missed intermediate blocks) before
        // discovering we're still behind — creating a catch-up death spiral
        // where each cycle takes ~5 s and the miner advances by ~1 block.
        self.network.send(from, Message::GetState);

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

    // ── New find_fork_point method ────────────────────────────────
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

    // ── Simplified evaluate_alternative_chain ─────────────────────
    async fn evaluate_alternative_chain(
        &mut self,
        fork_height: u64,
        alternative_batches: &[Batch],
        _from: PeerId,
    ) -> Result<Option<(State, Vec<(u64, [u8; 32], Batch)>)>> {
        // Simplified fork_state derivation
        let fork_state = if fork_height == 0 {
            State::genesis().0
        } else if fork_height <= self.state.height.saturating_sub(self.finality.calculate_safe_depth(1e-6)) {
            tracing::warn!("Fork at {} exceeds statistical safe depth, rejecting", fork_height);
            return Ok(None);
        } else {
            self.rebuild_state_at_height(fork_height)?
        };

        let mut candidate_state = fork_state;
        let mut new_history = Vec::new();
        
        // Use headers instead of states
        let mut recent_headers: VecDeque<u64> = VecDeque::new();


        let window_size = DIFFICULTY_LOOKBACK as usize;
        let start_height = fork_height.saturating_sub(window_size as u64);

        for h in start_height..fork_height {
            if let Some(batch) = self.storage.load_batch(h)? {
                recent_headers.push_back(batch.timestamp);
            }
        }

        for (i, batch) in alternative_batches.iter().enumerate() {
            recent_headers.push_back(candidate_state.timestamp);
            if recent_headers.len() > window_size {
                recent_headers.pop_front();
            }


            if batch.prev_midstate != candidate_state.midstate {
                tracing::warn!(
                    "Alternative chain broken at batch index {} (height {})",
                    i, fork_height + i as u64
                );
                return Ok(None);
            }

            match apply_batch(&mut candidate_state, batch, recent_headers.make_contiguous()) {

                Ok(_) => {
                    // Adjust difficulty via ASERT
                    candidate_state.target = adjust_difficulty(&candidate_state);
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

    // ── Added sync_in_progress = false at end ────────────
fn perform_reorg(
        &mut self,
        new_state: State,
        new_history: Vec<(u64, [u8; 32], Batch)>,
        is_fast_forward: bool,
    ) -> Result<()> {
        self.cancel_mining();

        let fork_height = new_history.first().map(|(h, _, _)| *h).unwrap_or(0);
        let is_actual_reorg = fork_height < self.state.height && !is_fast_forward;

        if is_fast_forward {
            tracing::warn!("FAST-FORWARD SYNC COMPLETE: Jumped from {} to {}", self.state.height, new_state.height);
            self.chain_history.clear();
            self.recent_headers.clear();
        } else if is_actual_reorg {
            tracing::warn!(
                "CHAIN REORG at fork height {}: replacing blocks {}..{} with new chain to {}",
                fork_height, fork_height, self.state.height, new_state.height
            );
            self.finality.observe_adversarial();
        } else {
            tracing::info!(
                "Chain extension via sync: height {} -> {}",
                self.state.height, new_state.height
            );
        }

        // Load abandoned batches from disk BEFORE overwriting them
        let mut abandoned_txs = Vec::new();
        if is_actual_reorg {
            for (h, _) in self.chain_history.iter().filter(|(h, _)| *h >= fork_height) {
                if let Ok(Some(batch)) = self.storage.load_batch(*h) {
                    abandoned_txs.extend(batch.transactions);
                }
            }
        }

        for (height, _, batch) in &new_history {
            if let Err(e) = self.storage.save_batch(*height, batch) {
                tracing::error!("Failed to save reorg batch at height {}: {}", height, e);
            }
        }

        self.state = new_state;
        while self.chain_history.back().map_or(false, |&(h, _)| h >= fork_height) {
            self.chain_history.pop_back();
        }
        self.chain_history.extend(new_history.iter().map(|(h, ms, _)| (*h, *ms)));

        // Rebuild headers cache from disk
        self.recent_headers.clear();
        let window = DIFFICULTY_LOOKBACK as u64;
        let start = self.state.height.saturating_sub(window);
        
        for h in start..self.state.height {
            if let Some(batch) = self.storage.load_batch(h)? {
                self.recent_headers.push_back(batch.timestamp);
            }
        }

        self.state.target = adjust_difficulty(&self.state);

        self.mempool.re_add(abandoned_txs, &self.state);

        self.mempool.prune_invalid(&self.state);
        if is_actual_reorg {
            self.metrics.inc_reorgs();
        }
        self.storage.save_state(&self.state)?;

        self.sync_in_progress = false;

        Ok(())
    }

    fn rebuild_state_at_height(&self, target_height: u64) -> Result<State> {
        // Find the nearest snapshot at or below target_height so we only
        // replay blocks from that point instead of from genesis.
        // Snapshots are written every PRUNE_DEPTH blocks, so round down.
        let snap_height = (target_height / PRUNE_DEPTH) * PRUNE_DEPTH;
        let (mut state, replay_from) = if snap_height > 0 {
            match self.storage.load_state_snapshot(snap_height) {
                Ok(Some(snap)) => {
                    tracing::debug!(
                        "rebuild_state_at_height: using snapshot at {} (target {})",
                        snap_height, target_height
                    );
                    (snap, snap_height)
                }
                _ => {
                    tracing::debug!(
                        "rebuild_state_at_height: no snapshot at {}, replaying from genesis",
                        snap_height
                    );
                    (State::genesis().0, 0)
                }
            }
        } else {
            (State::genesis().0, 0)
        };

        let mut recent_headers: VecDeque<u64> = VecDeque::new();
        for h in replay_from..target_height {
            if let Some(batch) = self.storage.load_batch(h)? {
                recent_headers.push_back(state.timestamp);
                if recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                    recent_headers.pop_front();
                }
                apply_batch(&mut state, &batch, recent_headers.make_contiguous())?;
                state.target = adjust_difficulty(&state);
            } else {
                anyhow::bail!("Missing batch at height {} needed for state rebuild", h);
            }
        }

        Ok(state)
    }

    /// Prune checkpoints from deeply finalized batches.
    ///
    /// Safe to call after every state advancement — internally tracks what's
    /// already pruned and skips redundant work.
    fn maybe_prune_checkpoints(&self) {
        if self.state.height > PRUNE_DEPTH {
            let prune_below = self.state.height - PRUNE_DEPTH;
            if let Err(e) = self.storage.prune_checkpoints(prune_below) {
                tracing::warn!("Checkpoint pruning failed: {}", e);
            }
        }
    }

    /// Handle a completed CoinJoin mix: submit Commit, queue Reveal.
    async fn handle_mix_transaction(&mut self, mix_id: [u8; 32], reveal_tx: Transaction) -> Result<()> {
        // Extract the commitment from the reveal tx
        let (input_ids, output_ids, salt) = match &reveal_tx {
            Transaction::Reveal { inputs, outputs, salt, .. } => {
                let ins: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
                (ins, outs, *salt)
            }
            _ => bail!("expected Reveal transaction"),
        };

        let commitment = compute_commitment(&input_ids, &output_ids, &salt);

        // Mine spam nonce for the Commit (respecting dynamic mempool difficulty)
        let required_pow = self.mempool.required_commit_pow();
        let spam_nonce = tokio::task::spawn_blocking(move || {
            let mut n = 0u64;
            loop {
                let h = hash_concat(&commitment, &n.to_le_bytes());
                if crate::core::types::count_leading_zeros(&h) >= required_pow {
                    return n;
                }
                n += 1;
            }
        }).await?;

        let commit_tx = Transaction::Commit { commitment, spam_nonce };
        tracing::info!(
            "CoinJoin mix {}: submitting Commit ({})",
            hex::encode(mix_id), hex::encode(commitment)
        );

        // Submit Commit to mempool
        self.handle_new_transaction(commit_tx, None).await?;

        // Queue Reveal for when the Commit gets mined
        self.pending_mix_reveals.insert(commitment, (mix_id, reveal_tx));
        Ok(())
    }

    /// Check if any pending CoinJoin Commits have been mined, and if so, submit their Reveals.
    async fn check_pending_mix_reveals(&mut self) {
        if self.pending_mix_reveals.is_empty() {
            return;
        }

        let mut to_reveal = Vec::new();
        for (commitment, _) in &self.pending_mix_reveals {
            // A commitment is "mined" when it's in the state accumulator
            if self.state.commitments.contains(commitment) {
                to_reveal.push(*commitment);
            }
        }

        for commitment in to_reveal {
            if let Some((mix_id, reveal_tx)) = self.pending_mix_reveals.remove(&commitment) {
                tracing::info!(
                    "CoinJoin mix {}: Commit mined, submitting Reveal",
                    hex::encode(mix_id)
                );
                match self.handle_new_transaction(reveal_tx, None).await {
                    Ok(()) => {
                        self.mix_manager.write().await
                            .set_phase(&mix_id, MixPhase::Complete);
                        tracing::info!("CoinJoin mix {} complete!", hex::encode(mix_id));
                    }
                    Err(e) => {
                        tracing::error!("CoinJoin Reveal failed for mix {}: {}", hex::encode(mix_id), e);
                        self.mix_manager.write().await
                            .set_phase(&mix_id, MixPhase::Failed(format!("reveal failed: {}", e)));
                    }
                }
            }
        }
    }

    async fn handle_new_batch(&mut self, batch: Batch, from: Option<PeerId>) -> Result<()> {
        // Fast pre-checks BEFORE cloning state (which copies the entire SMT).
        if batch.prev_midstate != self.state.midstate {
            tracing::debug!("Received orphan block (parent mismatch), queuing for later.");

            const ORPHAN_LIMIT: usize = 64;
            
            if self.orphan_order.len() >= ORPHAN_LIMIT {
                // Evict oldest half via FIFO order
                let to_evict = ORPHAN_LIMIT / 2;
                for _ in 0..to_evict {
                    if let Some(key) = self.orphan_order.pop_front() {
                        self.orphan_batches.remove(&key);
                    }
                }
            }

            self.orphan_order.push_back(batch.prev_midstate);
            self.orphan_batches.entry(batch.prev_midstate).or_default().push(batch);

            if !self.sync_in_progress {
                if let Some(peer) = from {
                    self.sync_in_progress = true;
                    self.network.send(peer, Message::GetState);
                }
            }
            return Ok(());
        }

        if batch.target != self.state.target {
            tracing::debug!("Batch target mismatch, ignoring");
            return Ok(());
        }

        // Checks passed — now clone state and apply fully.
        let mut candidate_state = self.state.clone();
        match apply_batch(&mut candidate_state, &batch, self.recent_headers.make_contiguous()) {
            Ok(_) => {
                let best = choose_best_state(&self.state, &candidate_state);
                let is_reorg = best.height == self.state.height &&
                               best.midstate != self.state.midstate;

                if best.height > self.state.height || is_reorg {
                    self.cancel_mining();

                    if is_reorg {
                        tracing::warn!("REORG at height {}", self.state.height);
                        self.metrics.inc_reorgs();
                    }

                    self.recent_headers.push_back(self.state.timestamp);
                    if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                        self.recent_headers.pop_front();
                    }
                    let pre_height = self.state.height;
                    self.state = candidate_state;
                    self.storage.save_batch(pre_height, &batch)?;
                    
                    self.state.target = adjust_difficulty(&self.state);
                    
                    self.metrics.inc_batches_processed();

                    let mut spent_inputs = Vec::new();
                    let mut mined_commits = Vec::new();
                    for tx in &batch.transactions {
                        match tx {
                            Transaction::Commit { commitment, .. } => mined_commits.push(*commitment),
                            Transaction::Reveal { inputs, .. } => {
                                for input in inputs { spent_inputs.push(input.coin_id()); }
                            }
                        }
                    }
                    self.mempool.prune_on_new_block(&self.state, &spent_inputs, &mined_commits);

                    self.chain_history.push_back((pre_height, self.state.midstate));
                    self.finality.observe_honest();
                    let safe_depth = self.finality.calculate_safe_depth(1e-6);
                    let cutoff_height = self.state.height.saturating_sub(safe_depth);
                    while self.chain_history.front().map_or(false, |&(h, _)| h < cutoff_height) {
                        self.chain_history.pop_front();
                    }

                    self.network.broadcast_except(from, Message::Batch(batch));
                    tracing::info!("Applied new batch from peer, height now {}", self.state.height);
                    self.try_apply_orphans().await;
                    self.check_pending_mix_reveals().await;
                }
                Ok(())
            }
            Err(e) => {
                tracing::debug!("Batch rejected after full validation: {}", e);
                // NOTE: We intentionally do NOT call observe_adversarial() here.
                // A rejected batch could be simple spam (garbage sigs, no PoW).
                // The finality estimator models adversarial *hashpower* — only
                // actual chain reorgs (which require real PoW) should shift the
                // estimate. See perform_reorg() for the legitimate call site.
                Ok(())
            }
        }
    }

    // ── Rewritten handle_batches_response ─────────────────────────
    async fn handle_batches_response(&mut self, batch_start_height: u64, batches: Vec<Batch>, from: PeerId) -> Result<()> {
        if batches.is_empty() { return Ok(()); }
        tracing::info!("Received {} batch(es) starting at height {} from peer {}", batches.len(), batch_start_height, from);

        // Try 1: Do they extend our current chain directly?
        // Cheap midstate check before expensive state clone.
        if batches[0].prev_midstate == self.state.midstate {
            let mut test_state = self.state.clone();
            if apply_batch(&mut test_state, &batches[0], self.recent_headers.make_contiguous()).is_ok() {
                return self.process_linear_extension(batches, from).await;
            }
        }

        // Try 2: Do any of them extend our chain? (we might already have some)
        for (i, batch) in batches.iter().enumerate() {
            if batch.prev_midstate != self.state.midstate { continue; }
            let mut candidate = self.state.clone();
            if apply_batch(&mut candidate, batch, self.recent_headers.make_contiguous()).is_ok() {
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
                        self.perform_reorg(new_state, new_history, false)?;
                        self.try_apply_orphans().await;
                        // Check if peer has even more blocks
                        self.network.send(from, Message::GetState);
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

    // ── Added sync_in_progress clear at end ──────────────
    async fn process_linear_extension(&mut self, batches: Vec<Batch>, from: PeerId) -> Result<()> {
        self.cancel_mining();
        let mut applied = 0;
        for batch in batches {
            // Cheap check before expensive clone
            if batch.prev_midstate != self.state.midstate { break; }
            let mut candidate = self.state.clone();
            if apply_batch(&mut candidate, &batch, self.recent_headers.make_contiguous()).is_ok() {
                self.recent_headers.push_back(self.state.timestamp);
                if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                    self.recent_headers.pop_front();
                }
                self.storage.save_batch(candidate.height - 1, &batch)?;
                self.state = candidate;
                
                self.state.target = adjust_difficulty(&self.state);
                self.metrics.inc_batches_processed();

                self.chain_history.push_back((self.state.height, self.state.midstate));
                self.finality.observe_honest();
                let safe_depth = self.finality.calculate_safe_depth(1e-6);
                let cutoff = self.state.height.saturating_sub(safe_depth);
                while self.chain_history.front().map_or(false, |&(h, _)| h < cutoff) {
                    self.chain_history.pop_front();
                }

                applied += 1;
            } else {
                break;
            }
        }

        if applied > 0 {
            tracing::info!("Synced {} batch(es), now at height {}", applied, self.state.height);
            // During bulk sync, a single O(N) sweep at the end is fine
            self.mempool.prune_invalid(&self.state);
            self.try_apply_orphans().await;
            self.check_pending_mix_reveals().await;

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

        while let Some(mut batches) = self.orphan_batches.remove(&self.state.midstate) {
            // Also remove all entries for this key from the order tracker
            self.orphan_order.retain(|k| k != &self.state.midstate);

            let mut matched = false;
            for batch in batches.drain(..) {
                let mut candidate = self.state.clone();
                if apply_batch(&mut candidate, &batch, self.recent_headers.make_contiguous()).is_ok() {
                    self.cancel_mining();
                    self.recent_headers.push_back(self.state.timestamp);
                    if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                        self.recent_headers.pop_front();
                    }
                    self.storage.save_batch(candidate.height - 1, &batch).ok();
                    self.state = candidate;

                    self.state.target = adjust_difficulty(&self.state);
                    self.metrics.inc_batches_processed();

                    let mut spent_inputs = Vec::new();
                    let mut mined_commits = Vec::new();
                    for tx in &batch.transactions {
                        match tx {
                            Transaction::Commit { commitment, .. } => mined_commits.push(*commitment),
                            Transaction::Reveal { inputs, .. } => {
                                for input in inputs { spent_inputs.push(input.coin_id()); }
                            }
                        }
                    }
                    self.mempool.prune_on_new_block(&self.state, &spent_inputs, &mined_commits);

                    applied += 1;
                    matched = true;
                    break; // State advanced — re-enter while loop for next height
                }
            }

            if !matched {
                break; // All candidates for this midstate failed validation
            }
        }

        if applied > 0 {
            tracing::info!("Applied {} orphan batch(es)", applied);
        }

        // Evict if over limit (FIFO: oldest first)
        while self.orphan_order.len() > MAX_ORPHAN_BATCHES {
            if let Some(key) = self.orphan_order.pop_front() {
                self.orphan_batches.remove(&key);
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

    /// Prepare a batch template and spawn a non-blocking background mining task.
    /// Returns immediately — the result arrives via mined_batch_rx.
    fn spawn_mining_task(&mut self) -> Result<()> {
        let threads = match self.mining_threads {
            Some(t) => t,
            None => return Ok(()),
        };
        
        if self.sync_in_progress || self.mining_cancel.is_some() {
            return Ok(());
        }
        tracing::info!("Mining batch with {} transactions...", self.mempool.len());

        // Clone only valid transactions. If any became stale since entering the
        // mempool, skip them silently instead of aborting the entire mining attempt.
        let pre_mine_height = self.state.height;
        let pre_mine_midstate = self.state.midstate;
        let mut candidate_state = self.state.clone();
        let mut total_fees: u64 = 0;
        let mut transactions = Vec::new();
        
        // Pull commits and reveals separately so commits can't be starved.
        // Reserve up to 20% of block space for commits; fills the rest with
        // fee-paying reveals. This is a local mining policy — not consensus —
        // but rational miners should follow it: including commits enables future
        // reveals (which pay fees), and a miner who starves commits starves
        // their own future revenue.
        let max_commits = MAX_BATCH_SIZE / 5;  // 20 slots reserved
        let max_reveals = MAX_BATCH_SIZE - max_commits;

        let (pending_commits, pending_reveals) = self.mempool.transactions_split();

        for arc_tx in pending_commits.into_iter().take(max_commits) {
            let tx = Arc::unwrap_or_clone(arc_tx);
            match apply_transaction(&mut candidate_state, &tx) {
                Ok(_) => {
                    total_fees += tx.fee();
                    transactions.push(tx);
                }
                Err(e) => {
                    tracing::debug!("Skipping stale mempool tx during mining: {}", e);
                }
            }
        }

        for arc_tx in pending_reveals.into_iter().take(max_reveals) {
            let tx = Arc::unwrap_or_clone(arc_tx);
            match apply_transaction(&mut candidate_state, &tx) {
                Ok(_) => {
                    total_fees += tx.fee();
                    transactions.push(tx);
                }
                Err(e) => {
                    tracing::debug!("Skipping stale mempool tx during mining: {}", e);
                }
            }
        }

        let coinbase = self.generate_coinbase(pre_mine_height, total_fees);
        for cb in &coinbase {
            let coin_id = cb.coin_id();
            candidate_state.coins.insert(coin_id);
            candidate_state.midstate = hash_concat(&candidate_state.midstate, &coin_id);
        }

        // --- NEW: Calculate state root ---
        let state_root = hash_concat(&candidate_state.coins.root(), &candidate_state.chain_mmr.root());
        candidate_state.midstate = hash_concat(&candidate_state.midstate, &state_root);
        // ---------------------------------

        let midstate = candidate_state.midstate;
        let target = self.state.target;

        let current_time = crate::core::state::current_timestamp();
        let block_timestamp = current_time.max(self.state.timestamp + 1);

        let mut template = Batch {
            prev_midstate: pre_mine_midstate,
            transactions,
            extension: Extension { nonce: 0, final_hash: [0; 32], checkpoints: vec![] },
            coinbase,
            timestamp: block_timestamp,
            target: self.state.target,
            state_root, // NEW
        };

        // Spawn background mining with cancellation token
        let cancel = Arc::new(AtomicBool::new(false));
        self.mining_cancel = Some(cancel.clone());
        let tx = self.mined_batch_tx.clone();

        tokio::task::spawn_blocking(move || {
            // Pass the `threads` variable into mine_extension
            if let Some(extension) = mine_extension(midstate, target, threads, cancel) {
                template.extension = extension;
                let _ = tx.send(template);
            }
            // If cancelled or channel closed, silently drop — the main loop already moved on
        });

        Ok(())
    }

    /// Process a successfully mined batch received from the background task.
    async fn handle_mined_batch(&mut self, batch: Batch) -> Result<()> {
        // If state advanced while we were mining, this batch is stale.
        // Don't clear mining_cancel — a new task may already be running.
        if self.state.midstate != batch.prev_midstate {
            tracing::warn!("State advanced during mining. Discarding stale mined block.");
            return Ok(());
        }

        self.mining_cancel = None; // This batch is current — task is done

        let pre_mine_height = self.state.height;

        self.recent_headers.push_back(self.state.timestamp);
        if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
            self.recent_headers.pop_front();
        }

        match apply_batch(&mut self.state, &batch, self.recent_headers.make_contiguous()) {
            Ok(_) => {
                self.storage.save_batch(pre_mine_height, &batch)?;
                self.storage.save_state(&self.state)?;
                self.state.target = adjust_difficulty(&self.state);
                self.metrics.inc_batches_mined();
                self.network.broadcast(Message::Batch(batch.clone()));

                let total_fees: u64 = batch.transactions.iter().map(|tx| tx.fee()).sum();
                self.log_coinbase(pre_mine_height, total_fees);

                let coinbase_value: u64 = batch.coinbase.iter().map(|cb| cb.value).sum();
                tracing::info!(
                    "Mined batch! height={} coinbase_value={} outputs={} target={}",
                    self.state.height,
                    coinbase_value,
                    batch.coinbase.len(),
                    hex::encode(self.state.target)
                );

                let mut spent_inputs = Vec::new();
                let mut mined_commits = Vec::new();
                for tx in &batch.transactions {
                    match tx {
                        Transaction::Commit { commitment, .. } => mined_commits.push(*commitment),
                        Transaction::Reveal { inputs, .. } => {
                            for input in inputs { spent_inputs.push(input.coin_id()); }
                        }
                    }
                }
                self.mempool.prune_on_new_block(&self.state, &spent_inputs, &mined_commits);
                self.check_pending_mix_reveals().await;
            }
            Err(e) => {
                tracing::error!("Failed to apply our own mined batch: {}", e);
            }
        }

        Ok(())
    }

    /// Synchronous test wrapper — spawns mining then blocks until it finishes.
    /// Preserves identical behavior for all existing tests.
    #[cfg(test)]
    pub async fn try_mine(&mut self) -> Result<()> {
        // Automatically enable mining for tests if it wasn't enabled
        if self.mining_threads.is_none() {
            self.mining_threads = Some(0); 
        }
        // Short-circuit if a sync is happening so the test doesn't hang forever
        if self.sync_in_progress || self.mining_cancel.is_some() {
            return Ok(());
        }
        self.spawn_mining_task()?;
        if let Some(batch) = self.mined_batch_rx.recv().await {
            self.handle_mined_batch(batch).await?;
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
    use crate::core::mss;
    use crate::core::types::hash;
    
    // Helper to create a bare-bones node for testing internal logic
    pub(crate) async fn create_test_node(dir: &std::path::Path) -> Node {
        let _keypair = libp2p::identity::Keypair::generate_ed25519();
        // Bind to port 0 to let OS assign a random available port
        let listen: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
        // Initialize node (this will create genesis if needed)
        Node::new(dir.to_path_buf(), None, listen, vec![]).await.unwrap()
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
            state_root: [0u8; 32],
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
            state_root: [0u8; 32],
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
            state_root: [0u8; 32],
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
            state_root: [0u8; 32],
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
            state_root: [0u8; 32],
        };

        // It connects to Genesis (height 0).
        // Fork point is height 1.
        let fork_h = node.find_fork_point(&[batch1_prime], 1).unwrap();
        assert_eq!(fork_h, 1, "Genesis fork should be detected at height 1");
    }
    
    #[test]
    fn scan_txs_for_mss_index_finds_max() {
        let seed = hash(b"test mss scan seed");
        let mut keypair = mss::keygen(&seed, 4).unwrap();
        let master_pk = keypair.public_key();

        let mut txs = Vec::new();
        for i in 0..5u8 {
            let msg = hash(&[i]);
            let sig = keypair.sign(&msg).unwrap();
            let sig_bytes = sig.to_bytes();

            let tx = Transaction::Reveal {
                inputs: vec![InputReveal {
                    predicate: Predicate::p2pk(&master_pk),
                    value: 1,
                    salt: [i; 32],
                }],
                witnesses: vec![Witness::sig(sig_bytes)],
                outputs: vec![OutputData::Standard {
                    address: [0xAA; 32],
                    value: 1,
                    salt: [i; 32],
                }],
                salt: [0; 32],
            };
            txs.push(tx);
        }

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
                predicate: Predicate::p2pk(&kp1.public_key()),
                value: 1,
                salt: [0; 32],
            }],
            witnesses: vec![Witness::sig(sig.to_bytes())],
            outputs: vec![OutputData::Standard {
                address: [0xAA; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        assert_eq!(scan_txs_for_mss_index(&[tx.clone()], &kp2.public_key()), 0);
        assert_eq!(scan_txs_for_mss_index(&[tx], &kp1.public_key()), 1);
    }

    #[test]
    fn scan_txs_mss_recovery_simulation() {
        let seed = hash(b"recovery sim");
        let mut keypair = mss::keygen(&seed, 4).unwrap();
        let master_pk = keypair.public_key();

        let mut txs = Vec::new();
        for i in 0..5u8 {
            let msg = hash(&[i]);
            let sig = keypair.sign(&msg).unwrap();
            txs.push(Transaction::Reveal {
                inputs: vec![InputReveal {
                    predicate: Predicate::p2pk(&master_pk),
                    value: 1,
                    salt: [i; 32],
                }],
                witnesses: vec![Witness::sig(sig.to_bytes())],
                outputs: vec![OutputData::Standard {
                    address: [0xBB; 32],
                    value: 1,
                    salt: [i; 32],
                }],
                salt: [0; 32],
            });
        }

        let chain_max = scan_txs_for_mss_index(&txs, &master_pk);
        assert_eq!(chain_max, 5, "should find highest used index + 1");

        let mut restored = mss::keygen(&seed, 4).unwrap();
        assert_eq!(restored.next_leaf, 0);

        const SAFETY_MARGIN: u64 = 20;
        if chain_max >= restored.next_leaf {
            restored.set_next_leaf(chain_max + SAFETY_MARGIN);
        }
        assert_eq!(restored.next_leaf, 25, "should be 5 + 20 safety margin");
    }

    #[test]
    fn scan_txs_mss_mempool_race() {
        let seed = hash(b"mempool race");
        let mut keypair = mss::keygen(&seed, 5).unwrap();

        for i in 0..10u8 {
            keypair.sign(&hash(&[i])).unwrap();
        }

        let msg = hash(b"mempool tx");
        let sig = keypair.sign(&msg).unwrap();
        assert_eq!(sig.leaf_index, 10);

        let mempool_tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&keypair.public_key()),
                value: 1,
                salt: [0; 32],
            }],
            witnesses: vec![Witness::sig(sig.to_bytes())],
            outputs: vec![OutputData::Standard {
                address: [0xCC; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

        let mempool_max = scan_txs_for_mss_index(&[mempool_tx], &keypair.public_key());
        assert_eq!(mempool_max, 11, "should account for leaf 10 → next = 11");

        let mut restored = mss::keygen(&seed, 5).unwrap();

        const SAFETY_MARGIN: u64 = 20;
        let remote_idx = mempool_max; 
        if remote_idx >= restored.next_leaf {
            restored.set_next_leaf(remote_idx + SAFETY_MARGIN);
        }
        assert_eq!(restored.next_leaf, 31, "should be 11 + 20 safety margin");
    }
    
    #[test]
    fn scan_txs_skips_wots_signatures() {
        let seed = hash(b"wots not mss");
        let pk = wots::keygen(&seed);
        let msg = hash(b"test");
        let sig = wots::sign(&seed, &msg);
        let sig_bytes = wots::sig_to_bytes(&sig);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&pk),
                value: 1,
                salt: [0; 32],
            }],
            witnesses: vec![Witness::sig(sig_bytes)],
            outputs: vec![OutputData::Standard {
                address: [0xAA; 32],
                value: 1,
                salt: [0; 32],
            }],
            salt: [0; 32],
        };

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
                predicate: Predicate::p2pk(&pk),
                value: 1,
                salt: [0; 32],
            }],
            witnesses: vec![Witness::sig(garbage)],
            outputs: vec![OutputData::Standard {
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
        Node::new(dir.to_path_buf(), None, listen, vec![]).await.unwrap()
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
        let mut candidate_state = prev_state.clone();
        
        // 1. Apply txs to candidate state
        let mut tx_fees = 0;
        for tx in &transactions {
            tx_fees += tx.fee();
            crate::core::transaction::apply_transaction(&mut candidate_state, tx).unwrap();
        }

        // 2. Generate coinbase
        let reward = crate::core::block_reward(prev_state.height);
        let total_value = reward + tx_fees;
        let mining_seed = prev_state.midstate; 
        
        let denominations = crate::core::types::decompose_value(total_value);
        let coinbase: Vec<CoinbaseOutput> = denominations.iter().enumerate().map(|(i, &val)| {
             let seed = crate::wallet::coinbase_seed(&mining_seed, prev_state.height, i as u64);
             let pk = crate::core::wots::keygen(&seed);
             let addr = crate::core::types::compute_address(&pk);
             let salt = crate::wallet::coinbase_salt(&mining_seed, prev_state.height, i as u64);
             CoinbaseOutput { address: addr, value: val, salt }
        }).collect();

        for cb in &coinbase {
            let coin_id = cb.coin_id();
            candidate_state.coins.insert(coin_id);
            candidate_state.midstate = hash_concat(&candidate_state.midstate, &coin_id);
        }

        // 3. Compute State Root
        let state_root = hash_concat(&candidate_state.coins.root(), &candidate_state.chain_mmr.root());
        candidate_state.midstate = hash_concat(&candidate_state.midstate, &state_root);

        let target = prev_state.target;
        let mut nonce = 0u64;
        let extension = loop {
            let ext = create_extension(candidate_state.midstate, nonce);
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
            state_root,
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
        apply_batch(&mut state_at_2, &genesis_batch, &[]).unwrap(); // H=1
        state_at_2.target = adjust_difficulty(&state_at_2);
        
        // Apply B1 (shared history)
       let ts_at_1 = vec![state_at_2.timestamp];
        apply_batch(&mut state_at_2, &b1, &ts_at_1).unwrap(); // H = 2
        state_at_2.target = adjust_difficulty(&state_at_2);

        // B2' (Alternative block at H=3). Empty, does NOT have the transaction.
        let b2_prime = make_valid_batch(&state_at_2, 20, vec![]); 
        let mut state_at_3_prime = state_at_2.clone();
        let ts_at_2 = vec![state_at_2.timestamp];
        apply_batch(&mut state_at_3_prime, &b2_prime, &ts_at_2).unwrap();
        state_at_3_prime.target = adjust_difficulty(&state_at_3_prime);
        
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
        let mempool_txs = node.mempool.transactions_cloned();
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
        // IMPORTANT: must call adjust_difficulty after each batch, exactly as
        // process_linear_extension does, so the target carried inside each batch
        // matches what the replaying node will expect. recent_headers are still
        // needed for timestamp validation (MTP) in apply_batch.
        let mut batches = Vec::new();
        let mut recent_headers: Vec<u64> = vec![node.state.timestamp];
        let window_size = DIFFICULTY_LOOKBACK as usize;
        for _ in 0..50 {
            let b = make_valid_batch(&node.state, 10, vec![]);
            apply_batch(&mut node.state, &b, &recent_headers).unwrap();
            recent_headers.push(node.state.timestamp);
            if recent_headers.len() > window_size { recent_headers.remove(0); }
            node.state.target = adjust_difficulty(&node.state);
            node.storage.save_batch(node.state.height - 1, &b).unwrap();
            batches.push(b);
        }
        
        assert_eq!(node.state.height, 51);

        // Reset node to H=1 (the "behind" node).
        node.state = start_state;
        node.recent_headers = VecDeque::from(vec![node.state.timestamp]);
        
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

    // ── Crash Recovery ──────────────────────────────────────────────────

    /// Verify that a node can be killed and restarted from the same data dir
    /// and resume at the correct height with the correct state.
    #[tokio::test]
    async fn crash_recovery_preserves_state() {
        let dir = tempdir().unwrap();

        // 1. Create a node, mine 5 blocks, save state
        let height_before;
        let midstate_before;
        let coins_before;
        {
            let mut node = create_test_node(dir.path()).await;
            assert_eq!(node.state.height, 1);

            for _ in 0..5 {
                node.try_mine().await.unwrap();
            }
            assert_eq!(node.state.height, 6);

            // Save state (normally happens on interval)
            node.storage.save_state(&node.state).unwrap();

            height_before = node.state.height;
            midstate_before = node.state.midstate;
            coins_before = node.state.coins.len();
            // node is dropped here — simulates crash
        }

        // 2. Recreate node from same data dir
        let node2 = create_test_node(dir.path()).await;

        // 3. Verify state matches
        assert_eq!(node2.state.height, height_before, "height must survive restart");
        assert_eq!(node2.state.midstate, midstate_before, "midstate must survive restart");
        assert_eq!(node2.state.coins.len(), coins_before, "coin set must survive restart");
    }

    /// Verify that after crash recovery, mining can resume and produce valid blocks.
    #[tokio::test]
    async fn crash_recovery_can_resume_mining() {
        let dir = tempdir().unwrap();

        {
            let mut node = create_test_node(dir.path()).await;
            for _ in 0..3 {
                node.try_mine().await.unwrap();
            }
            node.storage.save_state(&node.state).unwrap();
        }

        let mut node2 = create_test_node(dir.path()).await;
        assert_eq!(node2.state.height, 4);

        // Mining should work after restart
        node2.mining_threads = Some(0);
        node2.try_mine().await.unwrap();
        assert_eq!(node2.state.height, 5, "mining must resume after crash recovery");
    }

    // ── Sync From Scratch ───────────────────────────────────────────────

    /// A fresh node with no history receives the full chain and catches up.
    #[tokio::test]
    async fn fresh_node_syncs_full_chain() {
        let dir_miner = tempdir().unwrap();
        let dir_fresh = tempdir().unwrap();

        // 1. Build a chain of 20 blocks on the "miner" node
        let mut miner = create_test_node(dir_miner.path()).await;
        for _ in 0..20 {
            miner.try_mine().await.unwrap();
        }
        assert_eq!(miner.state.height, 21);

        // 2. Collect all batches from storage
        let mut batches = Vec::new();
        for h in 1..21 {
            let batch = miner.storage.load_batch(h).unwrap().unwrap();
            batches.push(batch);
        }

        // 3. Create a fresh node
        let mut fresh = create_test_node(dir_fresh.path()).await;
        assert_eq!(fresh.state.height, 1);

        // 4. Feed batches as if received from a peer
        fresh.sync_in_progress = true;
        fresh.sync_requested_up_to = 21;
        let peer = PeerId::random();
        fresh.handle_batches_response(1, batches, peer).await.unwrap();

        // 5. Verify state matches
        assert_eq!(fresh.state.height, miner.state.height);
        assert_eq!(fresh.state.midstate, miner.state.midstate);
        assert_eq!(fresh.state.coins.len(), miner.state.coins.len());
        assert!(!fresh.sync_in_progress);
    }

    /// A node that is partially synced receives remaining blocks.
    #[tokio::test]
    async fn partial_sync_completes() {
        let dir_miner = tempdir().unwrap();
        let dir_behind = tempdir().unwrap();

        let mut miner = create_test_node(dir_miner.path()).await;
        for _ in 0..15 {
            miner.try_mine().await.unwrap();
        }

        // Collect batches 1..15
        let mut all_batches = Vec::new();
        for h in 1..16 {
            all_batches.push(miner.storage.load_batch(h).unwrap().unwrap());
        }

        // Fresh node syncs first 5 blocks
        let mut behind = create_test_node(dir_behind.path()).await;
        behind.sync_in_progress = true;
        behind.sync_requested_up_to = 16;
        let peer = PeerId::random();
        behind.handle_batches_response(1, all_batches[..5].to_vec(), peer).await.unwrap();
        assert_eq!(behind.state.height, 6);

        // Now feed remaining blocks
        behind.handle_batches_response(6, all_batches[5..].to_vec(), peer).await.unwrap();
        assert_eq!(behind.state.height, 16);
        assert_eq!(behind.state.midstate, miner.state.midstate);
    }

    // ── Full Commit-Reveal Transaction Cycle ────────────────────────────

    /// Exercises the complete transaction lifecycle through the node:
    /// mine → get coins → commit → mine commit → reveal → mine reveal → verify
    #[tokio::test]
    async fn full_commit_reveal_cycle() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;
        let mining_seed = node.mining_seed;

        let pre_mine_height = node.state.height; // = 1
        node.try_mine().await.unwrap();
        assert_eq!(node.state.height, 2);

        let cb_seed = coinbase_seed(&mining_seed, pre_mine_height, 0);
        let cb_owner_pk = wots::keygen(&cb_seed);
        let cb_address = compute_address(&cb_owner_pk);
        let cb_salt = coinbase_salt(&mining_seed, pre_mine_height, 0);
        let cb_value = block_reward(pre_mine_height);
        let denominations = decompose_value(cb_value);
        let first_denom = denominations[0];
        let cb_coin_id = compute_coin_id(&cb_address, first_denom, &cb_salt);
        assert!(node.state.coins.contains(&cb_coin_id), "coinbase coin must be in UTXO set");

        let recipient_seed: [u8; 32] = hash(b"recipient seed");
        let recipient_pk = wots::keygen(&recipient_seed);
        let recipient_addr = compute_address(&recipient_pk);
        let output_salt: [u8; 32] = hash(b"output salt");

        let send_value = first_denom / 2;
        assert!(send_value > 0 && send_value.is_power_of_two());

        let change_seed: [u8; 32] = hash(b"change seed");
        let change_pk = wots::keygen(&change_seed);
        let change_addr = compute_address(&change_pk);
        let change_salt: [u8; 32] = hash(b"change salt");
        let change_value = first_denom / 4; 
        assert!(send_value + change_value < first_denom, "must leave fee");

        let outputs = vec![
            OutputData::Standard { address: recipient_addr, value: send_value, salt: output_salt },
            OutputData::Standard { address: change_addr, value: change_value, salt: change_salt },
        ];

        let input_coin_ids = vec![cb_coin_id];
        let output_coin_ids: Vec<[u8; 32]> = outputs.iter().filter_map(|o| o.coin_id()).collect();
        let tx_salt: [u8; 32] = hash(b"tx salt");
        let commitment = compute_commitment(&input_coin_ids, &output_coin_ids, &tx_salt);

        let mut spam_nonce = 0u64;
        loop {
            let h = hash_concat(&commitment, &spam_nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            spam_nonce += 1;
        }

        let commit_tx = Transaction::Commit { commitment, spam_nonce };
        let commit_batch = make_valid_batch(&node.state, 10, vec![commit_tx]);
        node.handle_new_batch(commit_batch, None).await.unwrap();
        assert!(node.state.commitments.contains(&commitment), "commitment must be in state");

        let sig = wots::sign(&cb_seed, &commitment);
        let reveal_tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&cb_owner_pk),
                value: first_denom,
                salt: cb_salt,
            }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs,
            salt: tx_salt,
        };

        let reveal_batch = make_valid_batch(&node.state, 10, vec![reveal_tx]);
        node.handle_new_batch(reveal_batch, None).await.unwrap();

        assert!(!node.state.coins.contains(&cb_coin_id), "spent coin must be removed");
        let recipient_coin_id = compute_coin_id(&recipient_addr, send_value, &output_salt);
        let change_coin_id = compute_coin_id(&change_addr, change_value, &change_salt);
        assert!(node.state.coins.contains(&recipient_coin_id), "recipient coin must exist");
        assert!(node.state.coins.contains(&change_coin_id), "change coin must exist");
    }

    // ── Wallet Send Flow End-to-End ─────────────────────────────────────

    /// Full wallet lifecycle: create → mine → scan → send → verify
    #[tokio::test]
    async fn wallet_send_flow_end_to_end() {
        use crate::wallet::{Wallet, coinbase_seed, coinbase_salt};

        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;
        let mining_seed = node.mining_seed;

        let mine_height = node.state.height;
        node.try_mine().await.unwrap();

        let wallet_path = dir.path().join("test_wallet.dat");
        let mut wallet = Wallet::create(&wallet_path, b"pass").unwrap();

        let denominations = decompose_value(block_reward(mine_height));
        let cb_seed = coinbase_seed(&mining_seed, mine_height, 0);
        let cb_salt = coinbase_salt(&mining_seed, mine_height, 0);
        let coin_id = wallet.import_coin(cb_seed, denominations[0], cb_salt, Some("coinbase".into())).unwrap();
        assert!(node.state.coins.contains(&coin_id));

        let recv_addr = wallet.generate_key(Some("recv".into())).unwrap();
        let send_value = denominations[0] / 2;
        let send_denoms = decompose_value(send_value);

        let live_coins: Vec<[u8; 32]> = wallet.coins().iter().map(|c| c.coin_id).collect();
        let selected = wallet.select_coins(send_value + 1, &live_coins).unwrap();
        assert_eq!(selected.len(), 1);

        let in_value: u64 = selected.iter()
            .filter_map(|id| wallet.find_coin(id))
            .map(|c| c.value)
            .sum();
            
        let (outputs, change_seeds) = wallet.build_outputs(&recv_addr, &send_denoms, 0).unwrap();
        let out_sum: u64 = outputs.iter().map(|o| o.value()).sum();
        assert!(in_value > out_sum, "fee must be positive");

        let (commitment, _salt) = wallet.prepare_commit(&selected, &outputs, change_seeds, false).unwrap();

        let mut spam_nonce = 0u64;
        loop {
            let h = hash_concat(&commitment, &spam_nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            spam_nonce += 1;
        }

        let commit_tx = Transaction::Commit { commitment, spam_nonce };
        let commit_batch = make_valid_batch(&node.state, 10, vec![commit_tx]);
        node.handle_new_batch(commit_batch, None).await.unwrap();
        assert!(node.state.commitments.contains(&commitment));

        let pending = wallet.find_pending(&commitment).unwrap().clone();
        let (input_reveals, witnesses) = wallet.sign_reveal(&pending).unwrap();
        
        // This is the specific fix for wallet_send_flow_end_to_end
        let reveal_tx = Transaction::Reveal {
            inputs: input_reveals,
            witnesses,
            outputs: pending.outputs.clone(),
            salt: pending.salt,
        };

        let reveal_batch = make_valid_batch(&node.state, 10, vec![reveal_tx]);
        node.handle_new_batch(reveal_batch, None).await.unwrap();

        assert!(!node.state.coins.contains(&coin_id), "spent coin removed");
        for out in &pending.outputs {
            assert!(node.state.coins.contains(&out.coin_id().unwrap()), "output coin must exist in UTXO set");
        }

        wallet.complete_reveal(&commitment).unwrap();
    }

    // ── Reorg Under Concurrent Mining ───────────────────────────────────

    /// Two miners find blocks at the same height. The node initially follows one,
    /// then switches when the other chain becomes longer.
    #[tokio::test]
    async fn concurrent_miners_reorg() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        // Mine 3 blocks for common history
        for _ in 0..3 {
            node.try_mine().await.unwrap();
        }
        assert_eq!(node.state.height, 4);
        let fork_state = node.state.clone();

        // Chain A: node mines 1 more block (height 5)
        node.try_mine().await.unwrap();
        assert_eq!(node.state.height, 5);

        // Chain B: built offline from fork_state, 3 blocks (heights 5, 6, 7)
        let mut chain_b_state = fork_state.clone();
        let mut chain_b_batches = Vec::new();
        let mut recent_headers: Vec<u64> = vec![chain_b_state.timestamp];

        for i in 0..3 {
            let b = make_valid_batch(&chain_b_state, 10 + i, vec![]);
            apply_batch(&mut chain_b_state, &b, &recent_headers).unwrap();
            recent_headers.push(chain_b_state.timestamp);
            if recent_headers.len() > 11 { recent_headers.remove(0); }
            chain_b_state.target = adjust_difficulty(&chain_b_state);
            chain_b_batches.push(b);
        }
        assert_eq!(chain_b_state.height, 7);

        // Feed Chain B to the node — should trigger reorg
        let peer = PeerId::random();
        node.handle_batches_response(4, chain_b_batches, peer).await.unwrap();

        assert_eq!(node.state.height, 7, "should switch to longer chain");
        assert_eq!(node.state.midstate, chain_b_state.midstate, "must adopt chain B's midstate");
    }
    
    // ── Interrupted Sync / Frankenstein Chain Tests ────────────────────

    /// Build a divergent chain from a fork point, returning the batches
    /// and the headers the peer would serve.
fn build_divergent_chain(
        fork_state: &State,
        length: usize,
    ) -> (Vec<Batch>, Vec<BatchHeader>) {
        let mut state = fork_state.clone();
        let mut batches = Vec::new();
        let mut headers = Vec::new();
        
        let mut recent_ts = vec![state.timestamp];
        
        for i in 0..length {
            let batch = make_valid_batch(&state, 50 + i as u64, vec![]);
            
            let mut hdr = batch.header();
            // FIX 1: The height of the new block is the parent's height 
            hdr.height = state.height; 
            headers.push(hdr);

            // FIX 2: Push the PARENT state's timestamp into the history window,
            // NOT the new batch's timestamp!
            recent_ts.push(state.timestamp);
            if recent_ts.len() > 11 { recent_ts.remove(0); }
            
            crate::core::state::apply_batch(&mut state, &batch, &recent_ts).unwrap();
            state.target = crate::core::state::adjust_difficulty(&state);
            batches.push(batch);
        }
        (batches, headers)
    }

    /// Core invariant: after an aborted sync, on-disk headers must still
    /// form a valid chain (no Frankenstein mixing of two chains).
    ///
    /// Before the fix, handle_sync_batches called save_batch() for each
    /// incoming peer batch *during* sync.  If the session was then aborted
    /// (timeout, disconnect, new session), some heights on disk held the
    /// peer's blocks while others still held the original chain's blocks.
    /// A subsequent load_headers() would produce a chain with broken
    /// prev_midstate linkage — the "Frankenstein chain" bug.
    #[tokio::test]
    async fn aborted_sync_does_not_corrupt_disk() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        // 1. Mine 10 blocks on the "main" chain so we have something on disk
        for _ in 0..10 {
            node.try_mine().await.unwrap();
        }
        assert_eq!(node.state.height, 11);

        // Snapshot — this is what disk should look like after abort
        let pre_sync_state = node.state.clone();

        // 2. Build a divergent chain that forks at height 5
        //    (longer than ours so the node would want to adopt it)
        let fork_state = node.rebuild_state_at_height(5).unwrap();
        let (alt_batches, alt_headers) = build_divergent_chain(&fork_state, 15);

        // We need the full header chain from genesis for the sync session.
        // Heights 0..5 are shared, 5..20 are the alt chain.
        let mut full_headers = node.storage.batches.load_headers(0, 5).unwrap();
        full_headers.extend(alt_headers);

        // 3. Manually construct a sync session in the Batches phase
        let peer = PeerId::random();
        let peer_height = 5 + alt_batches.len() as u64; // = 20
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth: peer_height * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 5,
                candidate_state: fork_state.clone(),
                cursor: 5,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
        });

        // 4. Feed only half the alt batches (simulating partial download)
        let half = alt_batches.len() / 2;
        node.handle_sync_batches(peer, alt_batches[..half].to_vec())
            .await
            .unwrap();

        // Session should still be in progress (waiting for more batches)
        assert!(node.sync_session.is_some(), "Session should still be active");

        // 5. ABORT the sync (simulates timeout or peer disconnect)
        node.abort_sync_session("simulated timeout");

        // 6. THE KEY ASSERTION: on-disk headers must still be a valid chain.
        //    Load all headers from disk and verify linkage.
        let disk_headers = node.storage.batches.load_headers(0, pre_sync_state.height).unwrap();
        assert!(
            !disk_headers.is_empty(),
            "Should have headers on disk"
        );
        assert!(
            Syncer::verify_header_chain(&disk_headers).is_ok(),
            "On-disk header chain must have valid linkage after aborted sync. \
             If this fails, sync wrote peer batches to disk before committing the reorg."
        );

        // 7. The node's in-memory state should be unchanged
        assert_eq!(node.state.height, pre_sync_state.height);
        assert_eq!(node.state.midstate, pre_sync_state.midstate);
    }

    /// Verify that a node which aborted a sync can still serve valid
    /// headers to other peers.  This is the downstream consequence of
    /// the Frankenstein bug: even if the node's own state is fine, any
    /// peer trying to sync FROM it would see broken header linkage and
    /// abort its own sync.
    #[tokio::test]
    async fn headers_served_after_aborted_sync_are_valid() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        // Mine 8 blocks
        for _ in 0..8 {
            node.try_mine().await.unwrap();
        }
        let original_height = node.state.height; // 9

        // Build alt chain forking at height 3, longer than ours
        let fork_state = node.rebuild_state_at_height(3).unwrap();
        let (alt_batches, alt_headers) = build_divergent_chain(&fork_state, 12);

        let mut full_headers = node.storage.batches.load_headers(0, 3).unwrap();
        full_headers.extend(alt_headers);

        // Set up sync session and feed some batches
        let peer = PeerId::random();
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer,
            peer_height: 3 + alt_batches.len() as u64,
            peer_depth: (3 + alt_batches.len() as u64) * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 3,
                candidate_state: fork_state,
                cursor: 3,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
        });

        // Feed 4 of 12 alt batches, then abort
        node.handle_sync_batches(peer, alt_batches[..4].to_vec())
            .await
            .unwrap();
        node.abort_sync_session("peer disconnected");

        // Now simulate what happens when another peer asks us for headers.
        // This is exactly what GetHeaders does internally.
        let served_headers = node.storage.batches
            .load_headers(0, original_height)
            .unwrap();

        assert!(
            Syncer::verify_header_chain(&served_headers).is_ok(),
            "Headers served to peers must have valid linkage. \
             Broken linkage here means peers cannot sync from us."
        );
    }

    /// Verify that a COMPLETED sync (not aborted) does persist the new
    /// chain to disk correctly and the headers are valid.
    #[tokio::test]
    async fn completed_sync_persists_valid_chain() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        // Mine 5 blocks
        for _ in 0..5 {
            node.try_mine().await.unwrap();
        }
        assert_eq!(node.state.height, 6);

        // Build alt chain forking at genesis (height 1), longer than ours
        let fork_state = node.rebuild_state_at_height(1).unwrap();
        let (alt_batches, alt_headers) = build_divergent_chain(&fork_state, 10);

        let mut full_headers = node.storage.batches.load_headers(0, 1).unwrap();
        full_headers.extend(alt_headers);

        let peer = PeerId::random();
        let peer_height = 1 + alt_batches.len() as u64; // 11
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth: peer_height * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 1,
                candidate_state: fork_state,
                cursor: 1,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
        });

        // Feed ALL batches — sync should complete and adopt the chain
        node.handle_sync_batches(peer, alt_batches.clone())
            .await
            .unwrap();

        assert_eq!(node.state.height, peer_height);
        assert!(!node.sync_in_progress);

        // Verify that the NEW chain on disk is coherent
        let disk_headers = node.storage.batches
            .load_headers(0, node.state.height)
            .unwrap();
        assert_eq!(disk_headers.len(), (node.state.height) as usize);
        assert!(
            Syncer::verify_header_chain(&disk_headers).is_ok(),
            "After a completed sync+reorg, on-disk headers must be a valid chain"
        );
    }

    /// A second sync replacing a first mid-flight must not leave the disk
    /// in a mixed state.  This simulates: start syncing from peer A,
    /// receive some batches, then peer B shows up with a better chain
    /// and we start a new session (dropping the old one).
    #[tokio::test]
    async fn replaced_sync_session_does_not_corrupt_disk() {
        let dir = tempdir().unwrap();
        let mut node = create_test_node(dir.path()).await;

        for _ in 0..8 {
            node.try_mine().await.unwrap();
        }
        let original_state = node.state.clone();

        // Build two different alt chains forking at different points
        let fork_a_state = node.rebuild_state_at_height(3).unwrap();
        let (alt_a_batches, alt_a_headers) = build_divergent_chain(&fork_a_state, 10);

        let mut full_headers_a = node.storage.batches.load_headers(0, 3).unwrap();
        full_headers_a.extend(alt_a_headers);

        // Start sync session with peer A
        let peer_a = PeerId::random();
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer: peer_a,
            peer_height: 13,
            peer_depth: 13_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers_a,
                fork_height: 3,
                candidate_state: fork_a_state,
                cursor: 3,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
        });

        // Feed some of A's batches
        node.handle_sync_batches(peer_a, alt_a_batches[..3].to_vec())
            .await
            .unwrap();

        // Now peer B shows up — start_sync_session replaces the session
        let peer_b = PeerId::random();
        node.start_sync_session(peer_b, 20, 20_000_000);

        // Disk should still be coherent (old session's writes should not exist)
        let disk_headers = node.storage.batches
            .load_headers(0, original_state.height)
            .unwrap();
        assert!(
            Syncer::verify_header_chain(&disk_headers).is_ok(),
            "Replacing a sync session mid-flight must not leave corrupted headers on disk"
        );
    }

    // ── Database Lock Retry Test ────────────────────────────────────────

    /// Verify that Storage::open retries when the database lock is held,
    /// rather than failing immediately.
    #[tokio::test]
    async fn storage_open_retries_on_lock() {
        use crate::storage::Storage;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");

        // Open the database (acquires the exclusive lock)
        let _storage1 = Storage::open(&db_path).unwrap();

        // Spawn a thread that will drop storage1 after a short delay,
        // simulating a previous process releasing the lock
        let db_path2 = db_path.clone();
        let handle = std::thread::spawn(move || {
            // Hold the lock for 300ms then drop
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(_storage1);
            // Now try to open — should succeed after retry
            Storage::open(&db_path2)
        });

        // The spawned thread should eventually acquire the lock
        let result = handle.join().unwrap();
        assert!(result.is_ok(), "Storage::open should succeed after lock is released");
    }

    /// Verify that Storage::open still fails if the lock is permanently held.
    #[test]
    fn storage_open_fails_after_max_retries() {
        use crate::storage::Storage;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("db");

        // Hold the lock for the duration of the test
        let _storage1 = Storage::open(&db_path).unwrap();

        // Second open should fail after exhausting retries
        let result = Storage::open(&db_path);
        assert!(result.is_err(), "Should fail when lock is permanently held");
    }
    
}
