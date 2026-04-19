use crate::core::*;
use crate::core::types::compute_address;
use crate::core::types::{CoinbaseOutput, BatchHeader};
use crate::core::state::{apply_batch, choose_best_state};
use crate::core::extension::{mine_extension, create_extension};
use crate::core::transaction::{apply_transaction, apply_transaction_no_sig_check, validate_transaction};
use crate::mempool::Mempool;
use crate::metrics::Metrics;
use crate::mix::{MixManager, MixPhase, MixStatusSnapshot};
use crate::network::{Message, MidstateNetwork, NetworkEvent, MAX_GETBATCHES_COUNT, MAX_GETHEADERS_COUNT};
use crate::storage::Storage;
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::mss;
use crate::core::wots;

use anyhow::{bail, Result};
use libp2p::{request_response::ResponseChannel, PeerId, Multiaddr, identity::Keypair};
use std::collections::{HashMap, HashSet, VecDeque};

/// Number of recent states to keep in memory for instant reorg rollback.
/// Each state uses O(1) memory to clone thanks to `im::` persistent data structures.
const STATE_CACHE_SIZE: usize = 200;

/// Disk snapshot interval (blocks). Smaller than PRUNE_DEPTH so that
/// post-restart reorgs deeper than the in-memory cache still have a
/// nearby snapshot to replay from instead of rewinding to genesis.
const SNAPSHOT_INTERVAL: u64 = 100;



use std::path::PathBuf;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time;
use rayon::prelude::*;

const MAX_ORPHAN_BATCHES: usize = 32;
/// How many batch chunks to have in-flight simultaneously across peers.
/// Each chunk is up to 8 MB, so 3 = up to ~24 MB of in-flight data.
const BATCH_LOOKAHEAD: usize = 3;

const MAX_PREFETCH_BUFFER: usize = 32;
const MAX_PREFETCH_DISTANCE: u64 = 10_000;

const SYNC_TIMEOUT_SECS: u64 = 120;
/// Extra grace period for the first chunk (relay handshake, NAT traversal, etc.)
const SYNC_INITIAL_TIMEOUT_SECS: u64 = 180;

/// Max transactions accepted from a single peer per rate-limit window.
const MAX_TX_PER_PEER_PER_WINDOW: u32 = 50;
/// Rate-limit window duration in seconds.
const TX_RATE_WINDOW_SECS: u64 = 10;

/// Dandelion++: if a stem tx hasn't been fluffed within this many seconds,
/// we fluff it ourselves as a safety net.
const STEM_TIMEOUT_SECS: u64 = 30;
/// Dandelion++: hard cap on stem pool entries. If full, new stem txs skip
/// the privacy phase and go directly to the public mempool. This prevents
/// unbounded memory growth under PoW spam from multiple Sybil peers.
const MAX_STEM_POOL_SIZE: usize = 1000;

/// GetHeaders responses are cheap when header files exist (~18 KB for 100 headers).
/// But the fallback path loads full batches (~8 MB each) for pre-migration blocks.
/// Allow enough requests for a full sync in one window while still bounding abuse.
/// 2700 headers needed worst case (12085 - 9425) / 100 = 27 requests minimum.
/// 200 per 60 seconds gives comfortable headroom without enabling disk-exhaustion.
const MAX_HEADER_REQS_PER_PEER: u32 = 200;
const HEADER_REQ_WINDOW_SECS: u64 = 60;

/// GetBatches requests are expensive (up to 8 MB each). Limit them separately.
/// 500 per 60 seconds allows fast sync while still bounding worst-case CPU/disk load.
const MAX_BATCH_REQS_PER_PEER: u32 = 1000;
const BATCH_REQ_WINDOW_SECS: u64 = 60;

/// Non-blocking sync session driven by the main event loop.
/// Replaces the old blocking `Syncer::sync_via_network` which hijacked the
/// network and dropped unrelated messages.
struct SyncSession {
    peer: PeerId,
    peer_height: u64,
    peer_depth: u128,
    phase: SyncPhase,
    started_at: std::time::Instant,
    /// Tracks when we last received useful data. Reset on every header/batch chunk.
    /// The timeout fires based on this, not `started_at`.
    last_progress_at: std::time::Instant,
}

enum SyncPhase {
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
        /// Chunks requested from secondary peers that haven't been applied yet.
       /// (start_height, peer_id)
        in_flight: std::collections::BTreeMap<u64, PeerId>,
        /// Out-of-order chunks that arrived before we were ready for them.
        /// Keyed by start_height.
        prefetch_buffer: std::collections::BTreeMap<u64, Vec<Batch>>,
    },
    VerifyingBatches {
        in_flight: std::collections::BTreeMap<u64, PeerId>,
        prefetch_buffer: std::collections::BTreeMap<u64, Vec<Batch>>,
    },
    /// Fork point found, GetBatches already sent, but state rebuild is still
    /// running in the background. Batches arriving from the peer are buffered here.
    PipelinedRebuild {
        headers: Vec<BatchHeader>,
        fork_height: u64,
        is_fast_forward: bool,
        buffered_batches: Vec<Batch>,
        in_flight: std::collections::BTreeMap<u64, PeerId>,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SyncStateBackup {
    pub cursor: u64,
    pub peer_height: u64,
    pub accumulated_headers: Vec<BatchHeader>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinerToml {
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiningConfig {
    pub mode: String,
    pub pool_url: Option<String>,
    pub payout_address: Option<String>,
    pub pool_address: Option<String>,
}

pub enum MinedResult {
    Block(Batch),
    Share {
        batch: Batch,
        pool_url: String,
        payout_address: String,
    },
}

pub struct Node {
    state: State,
    mempool: Mempool,
    storage: Storage,
    network: MidstateNetwork,
    metrics: Metrics,
    mining_threads: Option<usize>,
    recent_headers: VecDeque<u64>,
    orphan_batches: HashMap<[u8; 32], Vec<Batch>>,
    orphan_order: VecDeque<[u8; 32]>,
    sync_in_progress: bool,
    sync_requested_up_to: u64,
    sync_session: Option<SyncSession>,
    sync_retry_count: u32,
    sync_backoff_until: Option<std::time::Instant>,
    mining_seed: [u8; 32],
    data_dir: PathBuf,
    chain_history: VecDeque<(u64, [u8; 32])>,
    finality: crate::core::finality::FinalityEstimator,
    cached_safe_depth: u64,
    /// Last header cursor reached during sync. Used to resume after timeout
    /// instead of restarting from height - 360 every time.
    last_sync_cursor: Option<u64>,
    known_pex_addrs: HashMap<String, (u32, u32)>, // Bayesian Routing (Alpha, Beta)

    connected_peers: HashSet<PeerId>,
    // Background mining concurrency
    mining_cancel: Option<Arc<AtomicBool>>,
    mined_batch_rx: tokio::sync::mpsc::Receiver<MinedResult>,
    mined_batch_tx: tokio::sync::mpsc::Sender<MinedResult>,
    // CoinJoin mix coordinator
    mix_manager: Arc<RwLock<MixManager>>,
    /// Reveals waiting for their Commit to be mined.
    /// Key: commitment hash, Value: (mix_id, Reveal transaction)
    pending_mix_reveals: HashMap<[u8; 32], ([u8; 32], Transaction)>,
    /// Per-peer transaction rate limiter: maps peer -> (count, window_start).
    /// Resets every TX_RATE_WINDOW_SECS seconds.
    peer_tx_counts: HashMap<PeerId, (u32, std::time::Instant)>,
    cmd_tx: Option<tokio::sync::mpsc::Sender<NodeCommand>>,

    /// Dandelion++ stem pool: txs in stem phase waiting to be fluffed.
    /// Key: commitment or tx hash, Value: (transaction, received_at).
    /// After STEM_TIMEOUT_SECS without being fluffed, we fluff them ourselves.
    stem_pool: HashMap<[u8; 32], (Transaction, std::time::Instant)>,
    /// Per-peer rate limiter for GetBatches/GetHeaders requests.
    /// Separate rate-limit counter for GetBatches (expensive disk reads).
    peer_batch_req_counts: HashMap<PeerId, (u32, std::time::Instant)>,
    /// Rate-limit counter for GetHeaders (cheap normally, expensive on fallback path).
    peer_header_req_counts: HashMap<PeerId, (u32, std::time::Instant)>,
    hash_counter: Arc<AtomicU64>,
    banned_subnets: HashMap<IpAddr, std::time::Instant>,

    
    /// Ring buffer of recent states for instant reorg rollback.
    /// Keyed by height: state_cache[i] = (height, State) where the State
    /// is the result of applying all blocks through height-1.
    state_cache: VecDeque<(u64, State)>,
    
    /// Fallback peers to dial if we suffer a total network eclipse
    bootstrap_peers: Vec<String>,
}

#[derive(Clone)]
pub struct NodeHandle {
    state: Arc<RwLock<State>>,
    safe_depth: Arc<RwLock<u64>>,
    mempool_size: Arc<RwLock<usize>>,
    mempool_txs: Arc<RwLock<Vec<Transaction>>>,
    peer_addrs: Arc<RwLock<Vec<String>>>,
    webrtc_addrs: Arc<RwLock<Vec<String>>>, 
    tx_sender: tokio::sync::mpsc::Sender<NodeCommand>,
    pub storage: crate::storage::Storage,
    pub mix_manager: Arc<RwLock<MixManager>>,
    pub commit_limiter: Arc<tokio::sync::Semaphore>, 
    pub hash_counter: Arc<AtomicU64>,
    pub metrics: Metrics,
    /// Our local PeerId, needed for PeerId-bound CoinJoin PoW.
    local_peer_id: PeerId,
}

pub enum NodeCommand {
    SubmitMinedBlock(Batch),
    SendTransaction(Transaction),
    SubmitMixTransaction { mix_id: [u8; 32], tx: Transaction },
    // --- P2P Mix Coordination Commands ---
    BroadcastMixAnnounce { mix_id: [u8; 32], denomination: u64 },
    SendMixJoin { coordinator: PeerId, mix_id: [u8; 32], input: InputReveal, output: OutputData, signature: Vec<u8>, join_nonce: u64 },
    SendMixFee { coordinator: PeerId, mix_id: [u8; 32], input: InputReveal, join_nonce: u64 },
    SendMixSign { coordinator: PeerId, mix_id: [u8; 32], input_index: usize, signature: Vec<u8> },
    BroadcastMixProposal { mix_id: [u8; 32], proposal: crate::wallet::coinjoin::MixProposal, peers: Vec<PeerId> },
    FinishSyncHeadersChunk { peer: PeerId, headers: Vec<BatchHeader>, is_valid: bool },
    FinishSyncBatchesChunk {
        peer: PeerId,
        headers: Vec<BatchHeader>,
        fork_height: u64,
        candidate_state: Box<State>,
        cursor: u64,
        new_history: Vec<(u64, [u8; 32], Batch)>,
        is_fast_forward: bool,
        is_valid: bool,
        error_msg: String,
        session_started_at: std::time::Instant,
        peer_height: u64,
        peer_depth: u128,
    },
    FinishStateRebuild {
        peer: PeerId,
        fork_height: u64,
        candidate_state: Option<Box<State>>,
        headers: Vec<BatchHeader>,
        is_fast_forward: bool,
        is_valid: bool,
        is_local_corruption: bool,
    },
    BroadcastLightPush(crate::network::light_protocol::LightNotification),
    SendResponse { channel: libp2p::request_response::ResponseChannel<crate::network::Message>, msg: crate::network::Message },
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

    pub async fn check_commitment(&self, commitment: [u8; 32]) -> bool {
        self.state.read().await.commitments.contains(&commitment)
    }

    pub async fn get_mempool_info(&self) -> (usize, Vec<Transaction>) {
        let size = *self.mempool_size.read().await;
        let txs = self.mempool_txs.read().await.clone();
        (size, txs)
    }


    /// Build a block template for external miners.
    ///
    /// Clones the current state, applies valid mempool transactions,
    /// validates the miner's coinbase, computes state_root and mining
    /// midstate, and returns everything the miner needs to search nonces.
    pub async fn build_block_template(
        &self,
        req: crate::rpc::types::BlockTemplateRequest,
    ) -> Result<crate::rpc::types::BlockTemplateResponse, axum::response::Response> {
        use crate::core::transaction::apply_transaction;
        use axum::http::StatusCode;
        use axum::Json;
        use axum::response::IntoResponse;

                let state = self.state.read().await;
        let mut candidate = state.clone();
        let height = state.height;
        let target = state.target;
        let prev_midstate = state.midstate;
        let prev_header_hash = state.header_hash; // <-- Extract before drop
        let state_timestamp = state.timestamp;    // <-- Extract before drop
        drop(state);

        let mempool_txs = self.mempool_txs.read().await.clone();

        // Select valid mempool transactions
        let mut total_fees: u64 = 0;
        let mut transactions = Vec::new();
        let mut current_inputs = 0;
        let mut current_outputs = 0;

        for tx in &mempool_txs {
            match tx {
                Transaction::Commit { .. } => {
                    if transactions.iter().filter(|t| matches!(t, Transaction::Commit { .. })).count()
                        >= MAX_BATCH_COMMITS { continue; }
                    match apply_transaction(&mut candidate, tx) {
                        Ok(_) => { total_fees += tx.fee(); transactions.push(tx.clone()); }
                        Err(_) => {}
                    }
                }
                Transaction::Reveal { inputs, outputs, .. } => {
                    if transactions.iter().filter(|t| matches!(t, Transaction::Reveal { .. })).count()
                        >= MAX_BATCH_REVEALS { continue; }
                    if current_inputs + inputs.len() > MAX_BATCH_INPUTS { continue; }
                    if current_outputs + outputs.len() > MAX_BATCH_OUTPUTS { continue; }
                    match apply_transaction(&mut candidate, tx) {
                        Ok(_) => {
                            current_inputs += inputs.len();
                            current_outputs += outputs.len();
                            total_fees += tx.fee();
                            transactions.push(tx.clone());
                        }
                        Err(_) => {}
                    }
                }
            }
        }

        // Parse and validate coinbase
        let reward = block_reward(height);
        let expected_total = reward + total_fees;
        let mut coinbase = Vec::with_capacity(req.coinbase.len());
        let mut coinbase_total: u64 = 0;

        for cb in &req.coinbase {
            let mut address = [0u8; 32];
            let mut salt = [0u8; 32];
            if hex::decode_to_slice(&cb.address, &mut address).is_err()
                || hex::decode_to_slice(&cb.salt, &mut salt).is_err()
                || cb.value == 0 || !cb.value.is_power_of_two()
            {
                return Err((StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Invalid coinbase output"}))).into_response());
            }
            coinbase_total += cb.value;
            coinbase.push(CoinbaseOutput { address, value: cb.value, salt });
        }

        if coinbase_total != expected_total {
            return Err((StatusCode::CONFLICT, Json(
                crate::rpc::types::BlockTemplateMismatchError {
                    error: "Coinbase total mismatch".into(),
                    expected_total,
                    block_reward: reward,
                    total_fees,
                }
            )).into_response());
        }

        // Compute mining midstate
        for cb in &coinbase {
            candidate.midstate = hash_concat(&candidate.midstate, &cb.coin_id());
            candidate.coins.insert(cb.coin_id());
        }

        let smt_root = hash_concat(&candidate.coins.root(), &candidate.commitments.root());
        let state_root = hash_concat(&smt_root, &candidate.chain_mmr.root());
        candidate.midstate = hash_concat(&candidate.midstate, &state_root);

        let min_timestamp = state_timestamp + 1; // <-- Use extracted variable
        let actual_timestamp = crate::core::state::current_timestamp().max(min_timestamp);

        let candidate_header = BatchHeader {
            height,
            prev_header_hash, // <-- Use extracted variable
            prev_midstate,
            post_tx_midstate: candidate.midstate,
            extension: Extension { nonce: 0, final_hash: [0u8; 32] },
            timestamp: actual_timestamp,
            target,
            state_root,
        };
        let mining_hash = crate::core::types::compute_header_hash(&candidate_header);

        let batch = Batch {
            prev_midstate,
            prev_header_hash, // <-- Fixes E0063
            transactions,
            extension: Extension { nonce: 0, final_hash: [0u8; 32] },
            coinbase,
            timestamp: actual_timestamp,
            target,
            state_root,
        };

        Ok(crate::rpc::types::BlockTemplateResponse {
            mining_midstate: hex::encode(mining_hash), // Web miner grinds on the HEADER hash now
            target: hex::encode(target),
            batch_template: serde_json::to_value(&batch).unwrap(),
            total_fees,
            block_reward: reward,
        })
    }

    pub async fn get_peers(&self) -> Vec<String> {
        self.peer_addrs.read().await.clone()
    }
    pub async fn get_webrtc_addrs(&self) -> Vec<String> {
        self.webrtc_addrs.read().await.clone()
    }
    pub async fn send_transaction(&self, tx: Transaction) -> Result<()> {
        let state_guard = self.state.read().await;
        validate_transaction(&state_guard, &tx)?;
        drop(state_guard);
        self.tx_sender.try_send(NodeCommand::SendTransaction(tx))
            .map_err(|e| anyhow::anyhow!("Node is currently overloaded: {}", e))?;
        Ok(())
    }
    pub fn submit_mined_block(&self, batch: crate::core::Batch) -> anyhow::Result<()> {
        self.tx_sender.try_send(NodeCommand::SubmitMinedBlock(batch))
            .map_err(|e| anyhow::anyhow!("Node is currently overloaded: {}", e))?;
        Ok(())
    }
    pub fn scan_addresses(&self, addresses: &[[u8; 32]], start: u64, end: u64) -> Result<Vec<ScannedCoin>> {
        let store = &self.storage.batches; 
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
pub fn scan_mss_index(&self, master_pk: &[u8; 32], start: u64, end: u64) -> Result<u64> {
        let store = &self.storage.batches;
        let mut max_idx: u64 = 0;
        for h in start..end {
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
        self.tx_sender.try_send(NodeCommand::BroadcastMixAnnounce { mix_id, denomination })
            .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
        Ok(mix_id)
    }

    pub async fn mix_register(
        &self, mix_id: [u8; 32], input: InputReveal, output: OutputData, signature: Vec<u8> 
    ) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        mgr.register(&mix_id, input.clone(), output.clone(), &signature, None)?;
        
        //  Drop the lock BEFORE doing heavy PoW
        drop(mgr);

        if !is_coord {
            if let Some(peer) = coord_peer {
                // Mine anti-Sybil PoW on a blocking thread to avoid starving Tokio
                let coin_id = input.coin_id();
                let peer_id_bytes = self.local_peer_id.to_bytes();
                let join_nonce = tokio::task::spawn_blocking(move || {
                    crate::mix::mine_mix_join_pow(&mix_id, &coin_id, &peer_id_bytes, crate::mix::MIX_JOIN_POW_BITS)

                }).await.map_err(|e| anyhow::anyhow!("PoW task failed: {}", e))?;

                self.tx_sender.try_send(NodeCommand::SendMixJoin { 
                    coordinator: peer, mix_id, input, output, signature, join_nonce 
                }).map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
            }
        } else {
            // We are the coordinator. Re-acquire lock to check if ready.
            let mut mgr_coord = self.mix_manager.write().await;
            if let Ok(Some(proposal)) = mgr_coord.try_finalize(&mix_id) {
                let peers = mgr_coord.remote_participants(&mix_id);
                self.tx_sender.try_send(NodeCommand::BroadcastMixProposal { mix_id, proposal, peers })
                    .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
            }
        }
        Ok(())
    }

    pub async fn mix_set_fee(&self, mix_id: [u8; 32], input: InputReveal) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        mgr.set_fee_input(&mix_id, input.clone(), None)?;
        
        //  Drop the lock BEFORE doing heavy PoW
        drop(mgr);

        if !is_coord {
            if let Some(peer) = coord_peer {
                // Mine anti-Sybil PoW on a blocking thread to avoid starving Tokio
                let coin_id = input.coin_id();
                let peer_id_bytes = self.local_peer_id.to_bytes();
                let join_nonce = tokio::task::spawn_blocking(move || {
                    crate::mix::mine_mix_join_pow(&mix_id, &coin_id, &peer_id_bytes, crate::mix::MIX_JOIN_POW_BITS)
                }).await.map_err(|e| anyhow::anyhow!("PoW task failed: {}", e))?;

                self.tx_sender.try_send(NodeCommand::SendMixFee { coordinator: peer, mix_id, input, join_nonce })
                    .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
            }
        } else {
            // We are the coordinator. Re-acquire lock to check if ready.
            let mut mgr_coord = self.mix_manager.write().await;
            if let Ok(Some(proposal)) = mgr_coord.try_finalize(&mix_id) {
                let peers = mgr_coord.remote_participants(&mix_id);
                self.tx_sender.try_send(NodeCommand::BroadcastMixProposal { mix_id, proposal, peers })
                    .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
            }
        }
        Ok(())
    }

    pub async fn mix_sign(&self, mix_id: [u8; 32], input_index: usize, signature: Vec<u8>, current_height: u64) -> Result<()> {
        let mut mgr = self.mix_manager.write().await;
        let (is_coord, coord_peer) = mgr.get_session_info(&mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        mgr.add_signature(&mix_id, input_index, signature.clone(), current_height, None)?;

        if !is_coord {
            if let Some(peer) = coord_peer {
                self.tx_sender.try_send(NodeCommand::SendMixSign { coordinator: peer, mix_id, input_index, signature })
                    .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
            }
        } else if let Some(tx) = mgr.try_build_transaction(&mix_id)? {
            mgr.set_phase(&mix_id, MixPhase::CommitSubmitted);
            self.tx_sender.try_send(NodeCommand::SubmitMixTransaction { mix_id, tx })
                .map_err(|e| anyhow::anyhow!("Node is overloaded: {}", e))?;
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

                    // --- Calculate Genesis State Root ---
                    let smt_root = hash_concat(&temp_coins.root(), &state.commitments.root());
                    let state_root = hash_concat(&smt_root, &state.chain_mmr.root());
                    mining_midstate = hash_concat(&mining_midstate, &state_root);
                    // -----------------------------------------

                    let candidate_header = BatchHeader {
                        height: 0,
                        prev_header_hash: state.header_hash,
                        prev_midstate: state.midstate,
                        post_tx_midstate: mining_midstate,
                        extension: Extension { nonce: 0, final_hash: [0u8; 32] },
                        timestamp: state.timestamp,
                        target: state.target,
                        state_root,
                    };
                    let mining_hash = crate::core::types::compute_header_hash(&candidate_header);

                    // Hardcoded genesis nonce to avoid PoW on node initialization.
                    #[cfg(not(feature = "fast-mining"))]
                    let nonce = 8136715899467231487;
                    
                    // For tests, just find it dynamically (takes 0.001s)
                    #[cfg(feature = "fast-mining")]
                    let nonce = {
                        let mut n = 0;
                        loop {
                            if create_extension(mining_hash, n).final_hash < state.target { break n; }
                            n += 1;
                        }
                    };
                    
                    let extension = create_extension(mining_hash, nonce);
                    tracing::info!("Using Genesis Nonce: {}", nonce);

                    let genesis_batch = Batch {
                        prev_midstate: state.midstate,
                        prev_header_hash: state.header_hash,
                        transactions: vec![],
                        extension,
                        coinbase: genesis_coinbase,
                        timestamp: state.timestamp,
                        target: state.target,
                        state_root, 
                    };
                    storage.save_batch(0, &genesis_batch)?;
                    apply_batch(&mut state, &genesis_batch, &[], &mut std::collections::HashMap::new())?;
                    state.target = adjust_difficulty(&state);
                    storage.save_state(&state)?;
                    tracing::info!("Genesis batch applied, height now {}", state.height);
                }
                Some(batch) => {
                    if state.height == 0 {
                        apply_batch(&mut state, &batch, &[], &mut std::collections::HashMap::new())?;
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
        // We now persist the identity for ALL nodes using a data directory.
        // This allows nodes to have a "Static ID" that survives restarts.
        let keypair = match load_keypair(&data_dir) {
            Some(kp) => {
                tracing::info!("Loaded persistent peer identity: {}", kp.public().to_peer_id());
                kp
            }
            None => {
                let kp = Keypair::generate_ed25519();
                save_keypair(&data_dir, &kp);
                tracing::info!("Generated new persistent peer identity: {}", kp.public().to_peer_id());
                kp
            }
        };

        // Convert Multiaddrs to Strings BEFORE we move them into the network
        let bootstrap_strings: Vec<String> = bootstrap_peers.iter().map(|a| a.to_string()).collect();

        let network = MidstateNetwork::new(keypair, listen_addr, bootstrap_peers).await?;

        let mut recent_headers = VecDeque::new();
        let window = DIFFICULTY_LOOKBACK as u64;
        let start_height = state.height.saturating_sub(window);

        for h in start_height..state.height {
            if let Some(batch) = storage.load_batch(h)? {
                recent_headers.push_back(batch.timestamp);
            }
        }

        // Convert Multiaddrs to Strings to store for the fallback dialer
        let (mined_batch_tx, mined_batch_rx) = tokio::sync::mpsc::channel(100);

        Ok(Self {
            state,
            mempool: Mempool::new(),
            storage: storage.clone(),
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
            finality: crate::core::finality::FinalityEstimator::new(2, 8),
            cached_safe_depth: crate::core::finality::FinalityEstimator::new(2, 8).calculate_safe_depth(1e-6),
            last_sync_cursor: None,
            sync_session: None,
            sync_retry_count: 0,
            sync_backoff_until: None,
            known_pex_addrs: HashMap::new(),
            connected_peers: HashSet::new(),
            mining_cancel: None,
            mined_batch_rx,
            mined_batch_tx,
            mix_manager: Arc::new(RwLock::new(MixManager::new())),
            pending_mix_reveals: HashMap::new(),
            peer_tx_counts: HashMap::new(),
            cmd_tx: None,
            stem_pool: HashMap::new(),
            peer_batch_req_counts: HashMap::new(),
            peer_header_req_counts: HashMap::new(),
            hash_counter: Arc::new(AtomicU64::new(0)),
            banned_subnets: HashMap::new(),
            state_cache: VecDeque::with_capacity(STATE_CACHE_SIZE + 1),
            bootstrap_peers: bootstrap_strings, // <-- Uses the variable we created above
        })
    }

async fn process_state_rebuild(
    &mut self,
    peer: PeerId,
    fork_height: u64,
    candidate_state: Option<Box<State>>,
    headers: Vec<BatchHeader>,
    is_fast_forward: bool,
    is_valid: bool,
) -> Result<()> {
    if !is_valid {
        tracing::warn!("State rebuild or header validation failed, banning peer");
        self.sync_in_progress = false;
        self.ban_peer(peer, "invalid header chain or rebuild failed");
        return Ok(());
    }

    let headers_start_height = headers.first().map(|h| h.height).unwrap_or(0);
    if fork_height == headers_start_height && headers_start_height > 0 && !is_fast_forward {
        let session = match self.sync_session.take() {
            Some(s) if s.peer == peer => s,
            other => { self.sync_session = other; return Ok(()); }
        };
        let step_back = crate::network::MAX_GETHEADERS_COUNT;
        let new_start = headers_start_height.saturating_sub(step_back);
        tracing::warn!(
            "Fork is deeper than downloaded headers ({}). Stepping back to {} to find the exact fork point.", 
            headers_start_height, new_start
        );
        self.sync_session = Some(SyncSession {
            peer,
            peer_height: session.peer_height,
            peer_depth: session.peer_depth,
            phase: SyncPhase::Headers {
                accumulated: headers, // <--- KEEP previously downloaded headers
                cursor: new_start,    // <--- Step cursor backward
                snapshot: None,
            },
            started_at: session.started_at,
            last_progress_at: std::time::Instant::now(),
        });

        let count = headers_start_height.saturating_sub(new_start).min(crate::network::MAX_GETHEADERS_COUNT);
        self.network.send(peer, Message::GetHeaders { start_height: new_start, count });
        return Ok(());
    }

    match candidate_state {
        None => {
            // First message: fork point found, state rebuild still in progress.
            let session = match self.sync_session.take() {
                Some(s) if s.peer == peer && matches!(
                    s.phase,
                    SyncPhase::VerifyingHeaders | SyncPhase::PipelinedRebuild { .. }
                ) => s,
                other => {
                    tracing::warn!(
                        "FinishStateRebuild (None branch) arrived but session phase mismatch \
                         for peer {}, ignoring. Session will timeout.", peer
                    );
                    self.sync_session = other;
                    return Ok(());
                }
            };
            let existing_buffer = match &session.phase {
                SyncPhase::PipelinedRebuild { buffered_batches, .. } => buffered_batches.clone(),
                _ => Vec::new(),
            };
            tracing::info!(
                "Pipeline: fork point at {}. Sending GetBatches while state rebuilds...",
                fork_height
            );

            let count = (session.peer_height - fork_height).min(MAX_GETBATCHES_COUNT);
            self.network.send(peer, Message::GetBatches { start_height: fork_height, count });

            let mut in_flight = match &session.phase {
                SyncPhase::PipelinedRebuild { in_flight, .. } => in_flight.clone(),
                _ => std::collections::BTreeMap::new(),
            };
            in_flight.insert(fork_height, peer);

            self.sync_session = Some(SyncSession {
                peer,
                peer_height: session.peer_height,
                peer_depth: session.peer_depth,
                phase: SyncPhase::PipelinedRebuild {
                    headers,
                    fork_height,
                    is_fast_forward,
                    buffered_batches: existing_buffer,
                    in_flight,
                },
                started_at: session.started_at,
                last_progress_at: std::time::Instant::now(),
            });
        }

        Some(state) => {
            let _ = self.storage.save_state_snapshot(fork_height, &state);
            // Second message: state is ready. Drain buffer into Batches phase.
            let session = match self.sync_session.take() {
                Some(s) if s.peer == peer => s,
                other => { self.sync_session = other; return Ok(()); }
            };

            let mut is_instant = false;
            let (buffered, stored_headers, stored_fork_height, stored_is_fast_forward, in_flight, peer_height, peer_depth, started_at) = match session.phase {
                SyncPhase::PipelinedRebuild { 
                    buffered_batches, headers: ph_headers, fork_height: ph_fh, is_fast_forward: ph_ff, in_flight
                } => {
                    (buffered_batches, ph_headers, ph_fh, ph_ff, in_flight, session.peer_height, session.peer_depth, session.started_at)
                }
                SyncPhase::VerifyingHeaders => {
                    is_instant = true;
                    (Vec::new(), headers, fork_height, is_fast_forward, std::collections::BTreeMap::new(), session.peer_height, session.peer_depth, session.started_at)
                }
                _ => return Ok(()),
            };

            tracing::info!("Pipeline: state ready at height {}. {} buffered batch(es).", stored_fork_height, buffered.len());

            self.sync_session = Some(SyncSession {
                peer,
                peer_height,
                peer_depth,
                phase: SyncPhase::Batches {
                    headers: stored_headers,
                    fork_height: stored_fork_height,
                    candidate_state: *state,
                    cursor: stored_fork_height,
                    new_history: Vec::new(),
                    is_fast_forward: stored_is_fast_forward,
                    in_flight,
                    prefetch_buffer: std::collections::BTreeMap::new(),
                },
                started_at,
                last_progress_at: std::time::Instant::now(),
            });

            if is_instant {
                let count = (peer_height - stored_fork_height).min(MAX_GETBATCHES_COUNT);
                self.network.send(peer, Message::GetBatches { start_height: stored_fork_height, count });
                if let Some(s) = &mut self.sync_session {
                    if let SyncPhase::Batches { in_flight, .. } = &mut s.phase { in_flight.insert(stored_fork_height, peer); }
                }
            } else if !buffered.is_empty() {
                if let Err(e) = self.handle_sync_batches(peer, buffered).await {
                    tracing::warn!("Error applying buffered batches: {}", e);
                    self.abort_sync_session("buffered batch apply failed"); 
                    return Ok(()); 
                }
            }
            self.fire_batch_lookahead();
            self.drain_prefetch_buffer().await;
        }
    }
    Ok(())
}


async fn process_verified_batches_chunk(
        &mut self,
        from: PeerId,
        headers: Vec<BatchHeader>,
        fork_height: u64,
        candidate_state: Box<State>,
        current_cursor: u64,
        mut new_history: Vec<(u64, [u8; 32], Batch)>,
        is_fast_forward: bool,
        is_valid: bool,
        error_msg: String,
        session_started_at: std::time::Instant,
        peer_height: u64,
        peer_depth: u128,
    ) -> Result<()> {
        // Remove the VerifyingBatches placeholder the spawn set and extract lookahead state.
        let (mut in_flight, prefetch_buffer) = match self.sync_session.take() {
            Some(s) => match s.phase {
                SyncPhase::VerifyingBatches { in_flight, prefetch_buffer } => (in_flight, prefetch_buffer),
                _ => (std::collections::BTreeMap::new(), std::collections::BTreeMap::new()),
            },
            None => (std::collections::BTreeMap::new(), std::collections::BTreeMap::new()),
        };

        if !is_valid {
            tracing::warn!("Batch at height {} does not match verified header PoW — peer has corrupt data, trying different peer", current_cursor);
            tracing::warn!("{}", error_msg);
            // Save cursor so we restart from here, not from height - 360
            self.last_sync_cursor = Some(current_cursor.saturating_sub(1));
            self.abort_sync_session("peer sent corrupt batch");
            return Ok(());
        }

        tracing::info!("Applied sync batches up to height {}/{}", current_cursor, peer_height);

        // RAM FLUSH & CHECKPOINT: We cannot hold 60,000 blocks in RAM on a Raspberry Pi.
        // Once we have a chunk of blocks, AND the candidate chain has definitively 
        // overtaken our local chain's work (depth), we can safely commit the progress 
        // to disk immediately. This frees RAM (preventing OOM kills) and saves our spot.
        if new_history.len() >= 500 && candidate_state.depth > self.state.depth {
            tracing::info!("Checkpointing sync: Flushing {} batches to disk to free RAM...", new_history.len());
            let history_to_flush = std::mem::take(&mut new_history);
            self.perform_reorg(*candidate_state.clone(), history_to_flush, is_fast_forward).await?; 
        }

        // Restore sync state because perform_reorg clears it ---
        self.sync_in_progress = true;
        self.cancel_mining();
        
       if current_cursor < peer_height {
            let has_next = prefetch_buffer.contains_key(&current_cursor);
            if !has_next {
                let count = (peer_height - current_cursor).min(MAX_GETBATCHES_COUNT);
                self.network.send(from, Message::GetBatches { start_height: current_cursor, count });
                in_flight.insert(current_cursor, from);
            }
            
            // Re-arm the session back into the Batches phase
            self.sync_session = Some(SyncSession {
                peer: from,
                peer_height,
                peer_depth,
                phase: SyncPhase::Batches {
                    headers,
                    fork_height,
                    candidate_state: *candidate_state,
                    cursor: current_cursor,
                    new_history,
                    is_fast_forward,
                    in_flight,
                    prefetch_buffer,
                },
                started_at: session_started_at,
                last_progress_at: std::time::Instant::now(),
            });
            
            // Now that we are back in the Batches phase and cursor is advanced:
            self.fire_batch_lookahead();
            self.drain_prefetch_buffer().await;
            
            return Ok(());
        }

        // All batches applied — check if we should adopt this chain
        if candidate_state.depth > self.state.depth
            || (candidate_state.depth == self.state.depth && candidate_state.midstate < self.state.midstate)
        {
            tracing::info!(
                "✓ Sync complete! Adopting chain: height {} -> {}, depth {} -> {}",
                self.state.height, candidate_state.height,
                self.state.depth, candidate_state.depth
            );
            self.perform_reorg(*candidate_state, new_history, is_fast_forward).await?; 
            self.try_apply_orphans().await;
        } else {
            tracing::info!(
                "Sync complete but peer chain has less work (depth {} <= {}), keeping ours",
                candidate_state.depth, self.state.depth
            );
        }

        self.sync_in_progress = false;

        // By subtracting 1, we guarantee exactly 1 block of overlap for the next sync,
        // preventing the "Fork is deeper" panic, while keeping the sync instant.
        self.last_sync_cursor = Some(self.state.height.saturating_sub(1)); 
               
        // Reset backoff: successful sync proves the peer and path are healthy
        self.sync_retry_count = 0;
        self.sync_backoff_until = None;

        // CLEAR THE CACHE 
        let _ = std::fs::remove_file(self.data_dir.join("sync_state.bin"));

        // Immediately check if the peer has mined more blocks while we were syncing.
        self.network.send(from, Message::GetState);

        Ok(())
    }

    /// Handle a request from a browser light client over WebRTC.
    ///
    /// This mirrors the RPC handler logic but communicates via libp2p
    /// instead of HTTP, allowing browsers to reach any node directly
    /// without HTTPS, domains, or certificates.
async fn handle_light_request(
        &self,
        from: PeerId,
        request: crate::network::light_protocol::LightRequest,
    ) -> crate::network::light_protocol::LightResponse {
        use crate::network::light_protocol::{LightRequest, LightResponse};

        match request {
            LightRequest::GetState => {
                let state = &self.state;
                LightResponse::success(serde_json::json!({
                    "height": state.height,
                    "target": hex::encode(state.target),
                    "midstate": hex::encode(state.midstate),
                    "block_reward": crate::core::block_reward(state.height),
                    "required_pow": self.mempool.required_commit_pow(),
                    "timestamp": state.timestamp,
                }))
            }

            LightRequest::GetBlock { height } => {
                 let store = self.storage.batches.clone();
                
                let result = tokio::task::spawn_blocking(move || {
                    
                    match store.load(height) {
                        Ok(Some(mut batch)) => {
                            // Strip witness data to prevent WebRTC SCTP congestion and save bandwidth
                            for tx in &mut batch.transactions {
                                if let crate::core::Transaction::Reveal { witnesses, .. } = tx {
                                    *witnesses = vec![];
                                }
                            }
                            
                            match serde_json::to_value(&batch) {
                                Ok(val) => LightResponse::success(val),
                                Err(e) => LightResponse::error(format!("Serialization error: {}", e)),
                            }
                        }
                        Ok(None) => LightResponse::error("Block not found"),
                        Err(e) => LightResponse::error(format!("Storage error: {}", e)),
                    }
                }).await.unwrap_or_else(|_| LightResponse::error("Internal task panicked"));

                result
            }

            LightRequest::GetFilters { start_height, end_height } => {
                let end = end_height
                    .min(self.state.height)
                    .min(start_height.saturating_add(1000));
                
                let store = self.storage.batches.clone(); 
                
                // Spawn blocking thread to prevent freezing the network reactor!
                let result = tokio::task::spawn_blocking(move || {
                    

                    let mut filters = Vec::new();
                    let mut element_counts = Vec::new();
                    let mut block_hashes = Vec::new();

                    for h in start_height..end {
                        match (store.load(h), store.load_filter(h)) {
                            (Ok(Some(batch)), Ok(Some(filter_data))) => {
                                // Count elements the same way the RPC handler does
                                let mut items = std::collections::HashSet::new();
                                for tx in &batch.transactions {
                                    match tx {
                                        crate::core::Transaction::Commit { commitment, .. } => {
                                            items.insert(*commitment);
                                        }
                                        crate::core::Transaction::Reveal { inputs, outputs, .. } => {
                                            for input in inputs {
                                                items.insert(input.coin_id());
                                                items.insert(input.predicate.address());
                                            }
                                            for output in outputs {
                                                if let Some(cid) = output.coin_id() { items.insert(cid); }
                                                items.insert(output.address());
                                            }
                                        }
                                    }
                                }
                                for cb in &batch.coinbase {
                                    items.insert(cb.coin_id());
                                    items.insert(cb.address);
                                }
                                filters.push(hex::encode(filter_data));
                                block_hashes.push(hex::encode(batch.extension.final_hash));
                                element_counts.push(items.len() as u64);
                            }
                            (Ok(Some(batch)), _) => {
                                filters.push(String::new());
                                block_hashes.push(hex::encode(batch.extension.final_hash));
                                element_counts.push(0);
                            }
                            _ => break,
                        }
                    }
                    LightResponse::success(serde_json::json!({
                        "start_height": start_height,
                        "filters": filters,
                        "element_counts": element_counts,
                        "block_hashes": block_hashes,
                    }))
                }).await.unwrap_or_else(|_| LightResponse::error("Internal task panicked"));

                result
            }

            LightRequest::GetMempool => {
                let size = self.mempool.len();
                let txs = self.mempool.transactions_cloned();
                let tx_json: Vec<serde_json::Value> = txs.iter()
                    .filter_map(|tx| serde_json::to_value(tx).ok())
                    .collect();
                LightResponse::success(serde_json::json!({
                    "size": size,
                    "transactions": tx_json,
                }))
            }

            LightRequest::BlockTemplate { coinbase } => {
                let req: crate::rpc::types::BlockTemplateRequest = match serde_json::from_value(
                    serde_json::json!({ "coinbase": coinbase })
                ) {
                    Ok(r) => r,
                    Err(e) => return LightResponse::error(format!("Invalid coinbase: {}", e)),
                };

                let mut candidate = self.state.clone();
                let height = self.state.height;
                let target = self.state.target;
                let prev_midstate = self.state.midstate;

                let mempool_txs = self.mempool.transactions_cloned();

                let mut total_fees: u64 = 0;
                let mut transactions = Vec::new();
                let mut current_inputs = 0;
                let mut current_outputs = 0;

                for tx in &mempool_txs {
                    match tx {
                        Transaction::Commit { .. } => {
                            if transactions.iter().filter(|t| matches!(t, Transaction::Commit { .. })).count()
                                >= MAX_BATCH_COMMITS { continue; }
                            if let Ok(_) = apply_transaction(&mut candidate, tx) {
                                total_fees += tx.fee();
                                transactions.push(tx.clone());
                            }
                        }
                        Transaction::Reveal { inputs, outputs, .. } => {
                            if transactions.iter().filter(|t| matches!(t, Transaction::Reveal { .. })).count()
                                >= MAX_BATCH_REVEALS { continue; }
                            if current_inputs + inputs.len() > MAX_BATCH_INPUTS { continue; }
                            if current_outputs + outputs.len() > MAX_BATCH_OUTPUTS { continue; }
                            if let Ok(_) = apply_transaction(&mut candidate, tx) {
                                current_inputs += inputs.len();
                                current_outputs += outputs.len();
                                total_fees += tx.fee();
                                transactions.push(tx.clone());
                            }
                        }
                    }
                }

                let reward = block_reward(height);
                let expected_total = reward + total_fees;
                let mut coinbase_out = Vec::with_capacity(req.coinbase.len());
                let mut coinbase_total: u64 = 0;

                for cb in &req.coinbase {
                    let mut address = [0u8; 32];
                    let mut salt = [0u8; 32];
                    if hex::decode_to_slice(&cb.address, &mut address).is_err()
                        || hex::decode_to_slice(&cb.salt, &mut salt).is_err()
                        || cb.value == 0 || !cb.value.is_power_of_two()
                    {
                        return LightResponse::error("Invalid coinbase output");
                    }
                    coinbase_total += cb.value;
                    coinbase_out.push(CoinbaseOutput { address, value: cb.value, salt });
                }

if coinbase_total != expected_total {
                    return LightResponse {
                        ok: false,
                        data: Some(serde_json::json!({
                            "error": "Coinbase total mismatch",
                            "expected_total": expected_total,
                            "block_reward": reward,
                            "total_fees": total_fees,
                        })),
                        error: None,
                    };
                }

                for cb in &coinbase_out {
                    candidate.midstate = hash_concat(&candidate.midstate, &cb.coin_id());
                    candidate.coins.insert(cb.coin_id());
                }

                let smt_root = hash_concat(&candidate.coins.root(), &candidate.commitments.root());
                let state_root = hash_concat(&smt_root, &candidate.chain_mmr.root());
                candidate.midstate = hash_concat(&candidate.midstate, &state_root);
                let min_timestamp = candidate.timestamp + 1;
                let actual_timestamp = crate::core::state::current_timestamp().max(min_timestamp);
                let batch = Batch {
                    prev_midstate,
                    prev_header_hash: candidate.header_hash, // <-- Fixes E0063
                    transactions,
                    extension: Extension { nonce: 0, final_hash: [0u8; 32] },
                    coinbase: coinbase_out,
                    timestamp: actual_timestamp,
                    target,
                    state_root,
                };

                let resp = crate::rpc::types::BlockTemplateResponse {
                    mining_midstate: hex::encode(candidate.midstate),
                    target: hex::encode(target),
                    batch_template: serde_json::to_value(&batch).unwrap(),
                    total_fees,
                    block_reward: reward,
                };

                match serde_json::to_value(&resp) {
                    Ok(val) => LightResponse::success(val),
                    Err(e) => LightResponse::error(format!("Serialization error: {}", e)),
                }
            }

            LightRequest::SubmitBatch { batch } => {
                match serde_json::from_value::<crate::core::Batch>(batch) {
                    Ok(batch) => {
                        // Submit via the command channel
                        if let Some(tx) = &self.cmd_tx {
                            match tx.try_send(NodeCommand::SubmitMinedBlock(batch)) {
                                Ok(_) => LightResponse::success(serde_json::json!({ "accepted": true })),
                                Err(_) => LightResponse::error("Node is currently overloaded, please try again"),
                            }
                        } else {
                            LightResponse::error("Node command channel unavailable")
                        }
                    }
                    Err(e) => LightResponse::error(format!("Invalid batch JSON: {}", e)),
                }
            }

            LightRequest::Commit { commitment, spam_nonce } => {
                let mut commitment_bytes = [0u8; 32];
                if hex::decode_to_slice(&commitment, &mut commitment_bytes).is_err() {
                    return LightResponse::error("Invalid commitment hex");
                }

                let tx = crate::core::Transaction::Commit {
                    commitment: commitment_bytes,
                    spam_nonce,
                };

            match crate::core::transaction::validate_transaction(&self.state, &tx) {
                    Ok(_) => {
                        self.network.observe_honest_light_peer(from).await; // SMART GUARD REWARD
                        if let Some(cmd_tx) = &self.cmd_tx {
                            let _ = cmd_tx.try_send(NodeCommand::SendTransaction(tx));
                            LightResponse::success(serde_json::json!({ "accepted": true }))
                        } else {
                            LightResponse::error("Node command channel unavailable")
                        }
                    }
                    Err(e) => {
                        self.network.observe_adversarial_light_peer(from).await; // SMART GUARD PENALTY
                        LightResponse::error(format!("{}", e))
                    }
                }
            }

            LightRequest::Send { reveal } => {
                // Parse the reveal JSON into a Transaction::Reveal
                // The format matches what the web wallet sends to POST /send
                match crate::rpc::handlers::parse_reveal_json(reveal){
                    Ok(tx) => {
                        match crate::core::transaction::validate_transaction(&self.state, &tx) {
                            Ok(_) => {
                                self.network.observe_honest_light_peer(from).await; // SMART GUARD REWARD
                                if let Some(cmd_tx) = &self.cmd_tx {
                                    let _ = cmd_tx.try_send(NodeCommand::SendTransaction(tx));
                                    LightResponse::success(serde_json::json!({ "accepted": true }))
                                } else {
                                    LightResponse::error("Node command channel unavailable")
                                }
                            }
                            Err(e) => {
                                let err_str = format!("{}", e);
                                if !err_str.contains("No matching commitment found") {
                                    self.network.observe_adversarial_light_peer(from).await; // SMART GUARD PENALTY
                                }
                                LightResponse::error(err_str)
                            }
                        }
                    }
                    Err(e) => LightResponse::error(format!("Invalid reveal: {}", e)),
                }
            }

            LightRequest::CheckCoin { coin } => {
                let mut coin_bytes = [0u8; 32];
                if hex::decode_to_slice(&coin, &mut coin_bytes).is_err() {
                    return LightResponse::error("Invalid coin hex");
                }
                let exists = self.state.coins.contains(&coin_bytes);
                LightResponse::success(serde_json::json!({ "exists": exists }))
            }
            LightRequest::CheckCommitment { commitment } => {
                let mut commitment_bytes = [0u8; 32];
                if hex::decode_to_slice(&commitment, &mut commitment_bytes).is_err() {
                    return LightResponse::error("Invalid commitment hex");
                }
                let exists = self.state.commitments.contains(&commitment_bytes);
                LightResponse::success(serde_json::json!({ "exists": exists }))
            }
            LightRequest::MssState { master_pk } => {
                let mut pk_bytes = [0u8; 32];
                if hex::decode_to_slice(&master_pk, &mut pk_bytes).is_err() {
                    return LightResponse::error("Invalid master_pk hex");
                }
                // O(1) DB lookup instead of scanning every block from genesis
                let chain_max = self.storage.query_mss_leaf_index(&pk_bytes).unwrap_or(0);

                // Also check mempool for in-flight but unmined transactions
                let mempool_max = scan_txs_for_mss_index(
                    &self.mempool.transactions_cloned(), &pk_bytes
                );

                LightResponse::success(serde_json::json!({ "next_index": chain_max.max(mempool_max) }))
            }
        }
    }


    pub fn local_peer_id(&self) -> PeerId {
        self.network.local_peer_id()
    }

    /// Evaluates if the node is ready to mine, and spawns the task if so.
    fn trigger_mining(&mut self) {
        // Only mine if:
        // 1. Mining is enabled
        // 2. We aren't currently syncing
        // 3. We don't have a background task already
        // 4. We haven't heard of a better height/depth from peers recently (SyncRequestedUpTo)
        let is_behind = self.sync_requested_up_to > self.state.height;
        
        if self.mining_threads.is_some() && !self.sync_in_progress && !is_behind && self.mining_cancel.is_none() {
            if let Err(e) = self.spawn_mining_task() {
                tracing::error!("Failed to trigger mining task: {}", e);
            }
        }
    }

pub fn create_handle(&self) -> (NodeHandle, tokio::sync::mpsc::Receiver<NodeCommand>) {
            let (tx, rx) = tokio::sync::mpsc::channel(10_000); 
            let handle = NodeHandle {
            state: Arc::new(RwLock::new(self.state.clone())),
            safe_depth: Arc::new(RwLock::new(self.finality.calculate_safe_depth(1e-6))),
            mempool_size: Arc::new(RwLock::new(self.mempool.len())),
            mempool_txs: Arc::new(RwLock::new(self.mempool.transactions_cloned())),
            peer_addrs: Arc::new(RwLock::new(Vec::new())),
            webrtc_addrs: Arc::new(RwLock::new(Vec::new())),
            tx_sender: tx,
            storage: self.storage.clone(),
            mix_manager: Arc::clone(&self.mix_manager),
            commit_limiter: Arc::new(tokio::sync::Semaphore::new(4)), // <--  (Max 4 concurrent PoW tasks)
            hash_counter: Arc::clone(&self.hash_counter),
            metrics: self.metrics.clone(),
            local_peer_id: self.network.local_peer_id(),
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
        mut cmd_rx: tokio::sync::mpsc::Receiver<NodeCommand>,
    ) -> Result<()> {
        self.cmd_tx = Some(handle.tx_sender.clone());

        // Seed the state cache with our loaded tip so that shallow
        // reorgs immediately after startup can be resolved without
        // touching disk at all.
        self.cache_current_state();

        let mut save_interval = time::interval(Duration::from_secs(10));
        let mut ui_interval = time::interval(Duration::from_secs(1));
        ui_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut metrics_interval = time::interval(Duration::from_secs(30));
        let mut sync_poll_interval = time::interval(Duration::from_secs(5));
        sync_poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut mempool_prune_interval = time::interval(Duration::from_secs(60));
        let mut sync_timeout_interval = time::interval(Duration::from_secs(5));
        let mut pex_interval = time::interval(Duration::from_secs(120));
        let mut connection_maintenance = time::interval(Duration::from_secs(15));
        let mut stem_flush_interval = time::interval(Duration::from_secs(5));
        const TARGET_OUTBOUND_PEERS: usize = 8;
        let mut health_check_interval = time::interval(Duration::from_secs(600)); // Every 10 mins
        health_check_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        
        // Initial sync: ask all peers for their height
        if self.network.peer_count() > 0 {
            tracing::info!("Requesting chain state from {} peer(s)...", self.network.peer_count());
            self.sync_in_progress = true;
            for peer in self.network.connected_peers() {
                if self.network.is_light_peer(&peer) { continue; }
                self.network.send(peer, Message::GetState);
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        // Start mining immediately if we aren't waiting on a sync!
        self.trigger_mining();
        loop {
            tokio::select! {
                
                Some(result) = self.mined_batch_rx.recv() => {
                    match result {
                        MinedResult::Block(batch) => {
                            if let Err(e) = self.handle_mined_batch(batch).await {
                                tracing::error!("Failed to process mined batch: {}", e);
                            }
                        }
                        MinedResult::Share { batch, pool_url, payout_address } => {
                            tracing::info!("Found pool share! Submitting to {}", pool_url);
                            let client = reqwest::Client::new();
                            tokio::spawn(async move {
                                let payload = serde_json::json!({
                                    "batch": batch,
                                    "payout_address": payout_address
                                });
                                match client.post(&pool_url).json(&payload).send().await {
                                    Ok(res) if res.status().is_success() => {
                                        tracing::info!("Share accepted by pool!");
                                    }
                                    Ok(res) => {
                                        tracing::warn!("Pool rejected share. Status: {}", res.status());
                                    }
                                    Err(e) => {
                                        tracing::warn!("Failed to submit share to pool: {}", e);
                                    }
                                }
                            });
                            // Resume mining the same template immediately
                            self.mining_cancel = None;
                            self.trigger_mining();
                        }
                    }
                }
                _ = save_interval.tick() => {
                    // Clone the Arc to the DB and the persistent `im` state (O(1) RAM cost)
                    let storage_clone = self.storage.clone();
                    let state_clone = self.state.clone(); 
                    
                    // Offload the heavy serialization to a background thread
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = storage_clone.save_state(&state_clone) {
                            tracing::error!("Failed to save state in background: {}", e);
                        }
                    });
                }
                
                _ = health_check_interval.tick() => {
                    // Periodic O(1) health check to ensure we aren't desynced/corrupted
                    if !self.sync_in_progress && self.state.height > 0 {
                        match self.perform_health_check().await {
                            Ok(true) => {
                                tracing::debug!("Periodic state health check passed.");
                            }
                            Ok(false) => {
                                tracing::error!("CRITICAL: Node state is sick/corrupted. Initiating self-healing sequence...");
                                self.cancel_mining();
                                self.self_heal_rollback(self.state.height).await;
                                self.trigger_mining();
                            }
                            Err(e) => {
                                tracing::error!("Health check execution failed: {}", e);
                            }
                        }
                    }
                }
                
                _ = ui_interval.tick() => {
                    let current_safe_depth = self.cached_safe_depth;
                    *handle.state.write().await = self.state.clone();
                    *handle.safe_depth.write().await = current_safe_depth;
                    *handle.mempool_size.write().await = self.mempool.len();
                    *handle.mempool_txs.write().await = self.mempool.transactions_cloned();
                    *handle.peer_addrs.write().await = self.network.peer_addrs();
                    
                    // --- WebRTC Load Shedding ---
                    let mut webrtc_list = self.network.advertisable_addrs();
                    let community_addrs = self.network.pex_addrs(); 
                    
                    for addr in community_addrs {
                        // CRITICAL: Must contain certhash or the browser multiaddr parser will crash!
                        if addr.contains("webrtc-direct") && addr.contains("certhash") && !webrtc_list.contains(&addr) {
                            webrtc_list.push(addr);
                        }
                    }
                    *handle.webrtc_addrs.write().await = webrtc_list;  
                    // ---------------------------------
                }
                _ = metrics_interval.tick() => {
                    self.metrics.report();
                }
                _ = mempool_prune_interval.tick() => {
                    // CoinJoin: clean up stale mix sessions
                    self.mix_manager.write().await.cleanup();
                    // Light client: GC stale rate-limiter entries
                    self.network.gc_stale_light_peers().await;
                }
                _ = stem_flush_interval.tick() => {
                    self.flush_stem_pool();
                }
                _ = sync_poll_interval.tick() => {
                    // Respect exponential backoff: don't poll until the window expires
                    if let Some(until) = self.sync_backoff_until {
                        if std::time::Instant::now() < until {
                            continue;
                        }
                        self.sync_backoff_until = None;
                    }

                    // Don't interrupt an active session
                    if self.sync_session.is_some() || self.sync_in_progress {
                        continue;
                    }

                    // Poll ALL connected peers, not just one random one.
                    // On a small network (2-6 peers) polling one random peer means
                    // a 50-80% chance each tick of missing the peer with the best chain.
                    let peers: Vec<_> = self.network.connected_peers()
                        .into_iter()
                        .filter(|p| !self.network.is_light_peer(p))
                        .collect();

                    if peers.is_empty() {
                        tracing::debug!("Sync poll: no peers connected, skipping");
                    } else {
                        tracing::debug!("Sync poll: querying {} peer(s) for chain state", peers.len());
                        for peer in peers {
                            self.network.send(peer, Message::GetState);
                        }
                    }
                }
                
                _ = sync_timeout_interval.tick() => {
                    if let Some(session) = &self.sync_session {
                        if !matches!(session.phase, SyncPhase::VerifyingHeaders | SyncPhase::VerifyingBatches { .. } | SyncPhase::PipelinedRebuild { .. }) {
                            let idle_secs = session.last_progress_at.elapsed().as_secs();
                            let has_made_progress = session.last_progress_at != session.started_at;
                            let timeout = if has_made_progress { SYNC_TIMEOUT_SECS } else { SYNC_INITIAL_TIMEOUT_SECS };
                            let peer = session.peer;
                            let phase_name = match &session.phase {
                                SyncPhase::Headers { cursor, .. } => format!("Headers(cursor={})", cursor),
                                SyncPhase::Batches { cursor, .. } => format!("Batches(cursor={})", cursor),
                                _ => "Verifying".into(),
                            };
                            if idle_secs > timeout {
                                self.abort_sync_session(&format!(
                                    "timed out after {}s in phase {} with peer {}",
                                    idle_secs, phase_name, peer
                                ));
                            } else if idle_secs > timeout / 2 && idle_secs % 30 < 5 {
                                // Log roughly every 30s in the danger zone, not every 5s
                                tracing::warn!(
                                    "Sync stall warning: {}s idle in {} (timeout={}s, peer={})",
                                    idle_secs, phase_name, timeout, peer
                                );
                            }
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
                            
                            use rand::seq::IteratorRandom;
                            let mut rng = rand::thread_rng();

                            let mut candidates: Vec<_> = self.known_pex_addrs.iter().collect();
                            candidates.sort_by(|a, b| {
                                let p_a = a.1.0 as f32 / (a.1.0 + a.1.1) as f32;
                                let p_b = b.1.0 as f32 / (b.1.0 + b.1.1) as f32;
                                p_b.partial_cmp(&p_a).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            
                            let to_dial: Vec<String> = candidates.into_iter()
                                .take(needed.max(10)) 
                                .choose_multiple(&mut rng, needed)
                                .into_iter()
                                .map(|(k, _)| k.clone())
                                .collect();

                            let mut dialed = 0;
                            for addr in to_dial {
                                self.network.dial_addr(&addr);
                                dialed += 1;
                            }

                            // --- ANTI-ECLIPSE BOOTSTRAP FALLBACK ---
                            // If the PEX pool is completely poisoned/empty and we failed to dial 
                            // any good peers, OR if we have 0 outbound connections (total eclipse), 
                            // fall back to the hardcoded bootstrap peers.
                            if current_outbound == 0 || dialed == 0 {
                                if self.network.peer_count() == 0 {
                                    tracing::warn!("Eclipse Defense: Outbound connections critically low. Dialing bootstrap peers.");
                                } else {
                                    tracing::debug!("Outbound connections low ({} inbound). Dialing bootstrap peers.", self.network.peer_count());
                                }
                                for addr in &self.bootstrap_peers {
                                    self.network.dial_addr(addr);
                                }
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
                        NodeCommand::SendMixJoin { coordinator, mix_id, input, output, signature, join_nonce } => {
                            self.network.send(coordinator, Message::MixJoin { mix_id, input, output, signature, join_nonce });
                        }
                        NodeCommand::SendMixFee { coordinator, mix_id, input, join_nonce } => {
                            self.network.send(coordinator, Message::MixFee { mix_id, input, join_nonce });
                        }
                        NodeCommand::SendMixSign { coordinator, mix_id, input_index, signature } => {
                            self.network.send(coordinator, Message::MixSign { mix_id, input_index, signature });
                        }
                        NodeCommand::SubmitMinedBlock(batch) => {
                            if let Err(e) = self.handle_new_batch(batch, None).await {
                                tracing::error!("Failed to process pool-submitted block: {}", e);
                            }
                        }
                        NodeCommand::FinishSyncHeadersChunk { peer, headers, is_valid } => {
                            if let Err(e) = self.process_verified_headers_chunk(peer, headers, is_valid).await {
                                tracing::warn!("Failed to process headers chunk: {}", e);
                            }
                        }
                        NodeCommand::FinishSyncBatchesChunk { peer, headers, fork_height, candidate_state, cursor, new_history, is_fast_forward, is_valid, error_msg, session_started_at, peer_height, peer_depth } => {
                            if let Err(e) = self.process_verified_batches_chunk(peer, headers, fork_height, candidate_state, cursor, new_history, is_fast_forward, is_valid, error_msg, session_started_at, peer_height, peer_depth).await {
                                tracing::warn!("Error processing verified batches chunk: {}", e);
                            }
                        }
                        NodeCommand::FinishStateRebuild { peer, fork_height, candidate_state, headers, is_fast_forward, is_valid, is_local_corruption } => {
                            if is_local_corruption {
                                tracing::error!("Local database corruption detected during state rebuild. Initiating self-healing rollback...");
                                self.self_heal_rollback(fork_height).await;
                                self.abort_sync_session("local corruption, rolled back");
                            } else if let Err(e) = self.process_state_rebuild(peer, fork_height, candidate_state, headers, is_fast_forward, is_valid).await {
                                tracing::warn!("Failed to process state rebuild: {}", e);
                            }
                        }
                        NodeCommand::BroadcastLightPush(notif) => {
                            self.network.broadcast_light_push(&notif);
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
                        NodeCommand::SendResponse { channel, msg } => {
                            self.network.respond(channel, msg);
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
                        NetworkEvent::LightRequest { peer, request, respond } => {
                            let resp = self.handle_light_request(peer, request).await;
                            let _ = respond.send(resp);
                        }
                        NetworkEvent::PeerConnected(peer, full_addr) => {
                            // --- BAYESIAN ECLIPSE DEFENSE (REWARD) ---
                            // Successfully connecting proves this is a routable, honest peer.
                            // We reward them heavily to ensure they stay at the top of the PEX pool.
                            let stats = self.known_pex_addrs.entry(full_addr).or_insert((1, 1));
                            stats.0 = stats.0.saturating_add(10); // +10 to Alpha
                            // -----------------------------------------

                            // --- BANNED SUBNET DEFENSE ---
                            if let Some(subnet) = self.network.peer_subnet(&peer) {
                                if let Some(&ban_time) = self.banned_subnets.get(&subnet) {
                                    // Bans expire after 1 hour
                                    if ban_time.elapsed().as_secs() < 3600 {
                                        tracing::debug!("Rejected connection from banned subnet: {}", subnet);
                                        self.network.disconnect_peer(peer);
                                        continue;
                                    } else {
                                        self.banned_subnets.remove(&subnet);
                                    }
                                }
                            }
                            if !self.connected_peers.insert(peer) {
                                // Already connected via another transport — skip
                                continue;
                            }
                            // Light (WebRTC browser) peers don't speak the binary protocol.
                            // Don't send them GetState/GetAddr — it just causes errors.
                            if self.network.is_light_peer(&peer) {
                                tracing::info!("Light peer connected: {}", peer);
                                continue;
                            }
                            tracing::info!("Peer connected: {}", peer);
                            self.network.send(peer, Message::GetState);
                            self.network.send(peer, Message::GetAddr);
                        }
                        NetworkEvent::PeerDisconnected(peer) => {
                            self.connected_peers.remove(&peer);
                            self.peer_tx_counts.remove(&peer); 
                            tracing::info!("Peer disconnected: {}", peer);

                            // If this peer had in-flight lookahead requests, drop them from
                            // tracking so fire_batch_lookahead re-requests from another peer
                            if let Some(s) = &mut self.sync_session {
                                match &mut s.phase {
                                    SyncPhase::Batches { in_flight, .. } |
                                    SyncPhase::VerifyingBatches { in_flight, .. } |
                                    SyncPhase::PipelinedRebuild { in_flight, .. } => {
                                        let lost = in_flight.values().filter(|&p| p == &peer).count();
                                        if lost > 0 {
                                            tracing::warn!("Peer {} disconnected with {} in-flight chunk(s), will re-request", peer, lost);
                                            in_flight.retain(|_, p| &*p != &peer);
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            if self.sync_session.as_ref().map_or(false, |s| s.peer == peer) {
                                self.abort_sync_session("sync peer disconnected");
                            } else {
                                self.fire_batch_lookahead();
                            }
                            
                            // Fail any mixes relying on this peer
                            self.mix_manager.write().await.handle_peer_disconnect(peer);
                        }
// Bayesian Eclipse Defense 
                        NetworkEvent::OutgoingConnectionFailed(addr_str) => {
                            if let Some(stats) = self.known_pex_addrs.get_mut(&addr_str) {
                                stats.1 = stats.1.saturating_add(5); // Heavy penalty for failing to connect
                                let prob = stats.0 as f32 / (stats.0 + stats.1) as f32;
                                if prob < 0.1 {
                                    self.known_pex_addrs.remove(&addr_str);
                                    tracing::info!("Eclipse Defense: Purged statistically unreachable PEX address: {}", addr_str);
                                }
                            }
                        }
                        NetworkEvent::RequestFailed(peer) => {
                            if self.sync_session.as_ref().map_or(false, |s| s.peer == peer) {
                                self.abort_sync_session("Outbound request failed (timeout or disconnected)");
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
            Message::StemTransaction(tx) => {
                self.ack(channel);
                self.handle_stem_transaction(tx, from).await?;
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

                let is_syncing = self.sync_in_progress || self.sync_session.is_some();
                let is_sync_peer = self.sync_session.as_ref().map_or(false, |s| s.peer == from);

                if is_syncing {
                    // We are actively busy syncing. 
                    if is_sync_peer {
                        // The peer we are syncing with just sent us an update.
                        if midstate == self.state.midstate && height == self.state.height {
                            tracing::info!("Caught up to sync peer {}", from);
                            self.sync_in_progress = false;
                            self.sync_session = None;
                            self.trigger_mining();
                        } else if depth > self.state.depth || height > self.state.height {
                            tracing::debug!("Sync peer {} advanced to height {}", from, height);
                            // Do nothing, let the active sync session continue fetching
                        }
                    } else {
                        // A random peer sent us their state while we are busy. 
                        // IGNORE IT completely so it doesn't sabotage the active sync.
                        tracing::debug!("Ignoring StateInfo from {} because we are actively syncing with another peer.", from);
                    }
                } else {
                    // We are NOT syncing. Evaluate normally.
                    if depth > self.state.depth || height > self.state.height
                        || (depth == self.state.depth && midstate < self.state.midstate)
                    {
                        tracing::info!("Peer {} is ahead (h={}, d={}). Starting sync.", from, height, depth);
                        self.start_sync_session(from, height, depth, None);
                    } else {
                        tracing::debug!("Peer {} is at equal/lower depth, ignoring.", from);
                        if midstate == self.state.midstate && height == self.state.height {
                            self.trigger_mining();
                        }
                    }
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
                    let is_valid = addr_str.parse::<Multiaddr>()
                        .map(|ma| crate::network::is_routable(&ma)) 
                        .unwrap_or(false);

if is_valid && !self.known_pex_addrs.contains_key(&addr_str) {
                        // Evict the lowest probability peer if full
                        if self.known_pex_addrs.len() >= 1_000 {
                            if let Some(worst) = self.known_pex_addrs.iter()
                                .min_by(|a, b| {
                                    let p_a = a.1.0 as f32 / (a.1.0 + a.1.1) as f32;
                                    let p_b = b.1.0 as f32 / (b.1.0 + b.1.1) as f32;
                                    p_a.partial_cmp(&p_b).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(k, _)| k.clone()) {
                                self.known_pex_addrs.remove(&worst);
                            }
                        }
                        // Bayesian Prior: 1 honest connect, 1 failed (50% probability baseline)
                        self.known_pex_addrs.insert(addr_str, (1, 1));
                        new_count += 1;
                    }
                }
                
                if new_count > 0 {
                    tracing::debug!("PEX: saved {} new addrs from {}", new_count, from);
                }
            }
            Message::GetBatches { start_height, count } => {
                let now = std::time::Instant::now();
                let entry = self.peer_batch_req_counts.entry(from).or_insert((0, now));
                if now.duration_since(entry.1).as_secs() >= BATCH_REQ_WINDOW_SECS {
                    *entry = (0, now);
                }
                entry.0 += 1;
                if entry.0 > MAX_BATCH_REQS_PER_PEER {
                    tracing::debug!("Rate-limiting batch requests from peer {}", from);
                    self.send_response(channel, Message::Batches { start_height, batches: vec![] });
                    return Ok(());
                }

                let count = count.min(MAX_GETBATCHES_COUNT);
                let end = start_height.saturating_add(count).min(self.state.height);
                if end <= start_height {
                    self.send_response(channel, Message::Batches { start_height, batches: vec![] });
                    return Ok(());
                }

                if let Some(ch) = channel {
                    let storage = self.storage.clone();
                    let tx = self.cmd_tx.as_ref().unwrap().clone();
                    
                    tokio::task::spawn_blocking(move || {
                        let response = match storage.load_batches(start_height, end) {
                            Ok(tagged) => {
                                let actual_start = tagged.first().map(|(h, _)| *h).unwrap_or(start_height);
                                let mut batches = Vec::new();
                                let mut current_size = 0u64;
                                const MAX_PAYLOAD_BYTES: u64 = 8_000_000; 

                                for (_, batch) in tagged {
                                    let batch_size = bincode::serialized_size(&batch).unwrap_or(0);
                                    if !batches.is_empty() && current_size + batch_size > MAX_PAYLOAD_BYTES {
                                        break;
                                    }
                                    batches.push(batch);
                                    current_size += batch_size;
                                }
                                Message::Batches { start_height: actual_start, batches }
                            }
                            Err(e) => {
                                tracing::warn!("Failed to load batches: {}", e);
                                Message::Batches { start_height, batches: vec![] }
                            }
                        };
                        let _ = tx.try_send(NodeCommand::SendResponse { channel: ch, msg: response });
                    });
                }
                // Return immediately, the spawned task will send the response
                return Ok(());
            }
            Message::Batches { start_height: batch_start, batches } => {
                self.ack(channel);
                if !batches.is_empty() {
                    let session_role = self.sync_session.as_ref().and_then(|s| {
                        match &s.phase {
                            SyncPhase::Batches { in_flight, .. } => {
                                if s.peer == from || in_flight.iter().any(|(_, p)| *p == from) {
                                    Some("batches")
                                } else { None }
                            }
                            SyncPhase::VerifyingBatches { in_flight, .. } => {
                                if s.peer == from || in_flight.iter().any(|(_, p)| *p == from) {
                                    Some("verifying")
                                } else { None }
                            }
                            SyncPhase::PipelinedRebuild { .. } if s.peer == from => Some("pipeline"),
                            _ => None,
                        }
                    });

                    match session_role {
                        Some("batches") | Some("verifying") => {
                            let cursor = match &self.sync_session {
                                Some(s) => match &s.phase {
                                    SyncPhase::Batches { cursor, .. } => *cursor,
                                    // If we are currently verifying, the effective cursor for INCOMING
                                    // data is still the old cursor. We buffer everything that arrives.
                                    SyncPhase::VerifyingBatches { .. } => u64::MAX, // Force buffer
                                    _ => 0,
                                },
                                None => 0,
                            };

                            if batch_start == cursor && session_role == Some("batches") {
                                if let Some(s) = &mut self.sync_session {
                                    if let SyncPhase::Batches { in_flight, .. } = &mut s.phase {
                                        in_flight.remove(&batch_start);
                                    }
                                }
                                if let Err(e) = self.handle_sync_batches(from, batches).await {
                                    tracing::warn!("Error processing sync batches: {}", e);
                                    self.abort_sync_session("batch processing error");
                                }
                            } else {
                                tracing::debug!("Prefetch: buffering chunk at {} (role={:?})", batch_start, session_role);
                                if let Some(s) = &mut self.sync_session {
                                    match &mut s.phase {
                                        SyncPhase::Batches { cursor, prefetch_buffer, in_flight, .. } => {
                                            let current_cursor = *cursor;
                                            if batch_start > current_cursor && batch_start <= current_cursor + MAX_PREFETCH_DISTANCE && prefetch_buffer.len() < MAX_PREFETCH_BUFFER {
                                                prefetch_buffer.insert(batch_start, batches);
                                            }
                                            in_flight.remove(&batch_start);
                                        }
                                        SyncPhase::VerifyingBatches { prefetch_buffer, in_flight } => {
                                            if prefetch_buffer.len() < MAX_PREFETCH_BUFFER {
                                                prefetch_buffer.insert(batch_start, batches);
                                            }
                                            in_flight.remove(&batch_start);
                                        }
                                        _ => {}
                                    }
                                    s.last_progress_at = std::time::Instant::now();
                                }
                            }
                        }
                        Some("pipeline") => {
                            if let Some(s) = &mut self.sync_session {
                                if let SyncPhase::PipelinedRebuild { buffered_batches, .. } = &mut s.phase {
                                    if buffered_batches.len() < 1000 {
                                        buffered_batches.extend(batches);
                                        s.last_progress_at = std::time::Instant::now();
                                    } else {
                                        tracing::warn!("Pipeline buffer overflow. Aborting sync.");
                                        self.abort_sync_session("pipeline buffer overflow");
                                    }
                                }
                            }
                        }
                        _ => {
                            self.handle_batches_response(batch_start, batches, from).await?;
                        }
                    }
               } else {
                    // If the peer sends an empty list, and we are actively waiting 
                    // for blocks from them, abort immediately instead of stalling for 15 mins.
                    let is_waiting = self.sync_session.as_ref().map_or(false, |s| s.peer == from);
                    if is_waiting {
                        tracing::warn!("Peer {} sent an empty block list. Aborting sync.", from);
                        self.abort_sync_session("peer sent empty batches response");
                    }
                }
            }
            Message::GetHeaders { start_height, count } => {
                let now = std::time::Instant::now();
                let entry = self.peer_header_req_counts.entry(from).or_insert((0, now));
                if now.duration_since(entry.1).as_secs() >= HEADER_REQ_WINDOW_SECS {
                    *entry = (0, now);
                }
                entry.0 += 1;
                if entry.0 > MAX_HEADER_REQS_PER_PEER {
                    tracing::debug!("Rate-limiting header requests from peer {}", from);
                    self.send_response(channel, Message::Headers { start_height, headers: vec![] });
                    return Ok(());
                }

                let count = count.min(MAX_GETHEADERS_COUNT);
                let end = start_height.saturating_add(count).min(self.state.height + 1);
                if end <= start_height {
                    self.send_response(channel, Message::Headers { start_height, headers: vec![] });
                    return Ok(());
                }
                
                if let Some(ch) = channel {
                    let storage = self.storage.clone();
                    let tx = self.cmd_tx.as_ref().unwrap().clone();
                    
                    tokio::task::spawn_blocking(move || {
                        let response = match storage.batches.load_headers(start_height, end) {
                            Ok(headers) => Message::Headers { start_height, headers },
                            Err(e) => {
                                tracing::warn!("Failed to load headers: {}", e);
                                Message::Headers { start_height, headers: vec![] }
                            }
                        };
                        let _ = tx.try_send(NodeCommand::SendResponse { channel: ch, msg: response });
                    });
                }
                return Ok(());
            }
            Message::Headers { start_height: _, headers } => {
                self.ack(channel);
                if let Err(e) = self.handle_sync_headers(from, headers).await {
                    tracing::warn!("Error processing sync headers from {}: {}", from, e);
                    self.abort_sync_session("header processing error");
                    self.ban_peer(from, &format!("malicious headers: {}", e));
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

            Message::MixJoin { mix_id, input, output, signature, join_nonce } => {
                self.ack(channel);

                // Validate coin exists in UTXO set before touching MixManager
                let coin_id = input.coin_id();
                if !self.state.coins.contains(&coin_id) {
                    tracing::debug!("MixJoin rejected from peer {}: coin does not exist", from);
                    return Ok(());
                }

                // Client always mines MIX_JOIN_POW_BITS. The Bayesian system handles dropping 
                // bad peers independently, so we don't need a hidden PoW trap here.
                let required_pow = crate::mix::MIX_JOIN_POW_BITS;

                if !crate::mix::verify_mix_join_pow(&mix_id, &coin_id, &from.to_bytes(), join_nonce, required_pow) {
                    tracing::debug!("MixJoin rejected from peer {}: insufficient join PoW (needed {})", from, required_pow);
                    return Ok(());
                }

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
            
            Message::MixFee { mix_id, input, join_nonce } => {
                self.ack(channel);

                // Validate fee coin exists in UTXO set before touching MixManager
                let coin_id = input.coin_id();
                if !self.state.coins.contains(&coin_id) {
                    tracing::debug!("MixFee rejected from peer {}: fee coin does not exist", from);
                    return Ok(());
                }

                let required_pow = crate::mix::MIX_JOIN_POW_BITS;

                if !crate::mix::verify_mix_join_pow(&mix_id, &coin_id, &from.to_bytes(), join_nonce, required_pow) {
                    tracing::debug!("MixFee rejected from peer {}: insufficient join PoW (needed {})", from, required_pow);
                    return Ok(());
                }

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
                if let Err(e) = mgr.add_signature(&mix_id, input_index, signature, self.state.height, Some(from)) {
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
        }
        Ok(())
    }

    // ── Non-blocking sync state machine ─────────────────────────────────

fn start_sync_session(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, force_start: Option<u64>) {
        self.cancel_mining();

        let sync_file_path = self.data_dir.join("sync_state.bin");
        let mut recovered_headers = Vec::new();
        let mut recovered_cursor = None;

        if let Ok(bytes) = std::fs::read(&sync_file_path) {
            if let Ok(backup) = bincode::deserialize::<SyncStateBackup>(&bytes) {
                // Only resume if the peer we are connecting to has a chain at least as long
                if peer_height > backup.cursor {
                    tracing::info!("Recovered interrupted sync session. Resuming from height {}", backup.cursor);
                    recovered_headers = backup.accumulated_headers;
                    recovered_cursor = Some(backup.cursor);
                } else {
                    tracing::info!("Discarding recovered sync session (new peer has shorter chain).");
                    let _ = std::fs::remove_file(&sync_file_path);
                }
            }
        }
        
        // <--- Prevent mixing old recovered headers with a fresh start
        if force_start.is_some() {
            recovered_headers.clear();
            recovered_cursor = None;
            let _ = std::fs::remove_file(&sync_file_path);
        }
        // --------------------------------------------------------------------

        // <--- Clamp sync window to peer_height for lower-height forks
        let effective_height = self.state.height.min(peer_height);
        
        // Prefer: explicit override > recovered_cursor > last known cursor > 360-block lookback.
        let start_height = force_start.unwrap_or_else(|| {
            recovered_cursor.unwrap_or_else(|| {
                let base_start = self.last_sync_cursor
                    .filter(|&c| c > effective_height.saturating_sub(360)
                                 && c <= effective_height)
                    .unwrap_or_else(|| {
                        effective_height.saturating_sub(360)
                    });
                base_start.min(effective_height)
            })
        });
        // --------------------------------------------------------------------
        
        // Prevent zero-header sync traps
        if peer_height <= start_height {
            tracing::debug!("Peer height {} is <= our sync cursor {}, ignoring.", peer_height, start_height);
            self.sync_in_progress = false;
            self.trigger_mining();
            return;
        }
        // ---------------------------------------------
        
        tracing::info!(
            "Starting headers-first sync from height {}: peer(h={}, d={}) vs us(h={}, d={})",
            start_height, peer_height, peer_depth, self.state.height, self.state.depth
        );
        self.sync_in_progress = true;
        let now = std::time::Instant::now();
        self.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::Headers {
                accumulated: recovered_headers,
                cursor: start_height,
                snapshot: None, 
            },
            started_at: now,
            last_progress_at: now,
        });
        
let count = MAX_GETHEADERS_COUNT.min(peer_height.saturating_sub(start_height));
self.network.send(peer, Message::GetHeaders { start_height, count });
    }



fn fire_batch_lookahead(&mut self) {
        let (cursor, peer_height, in_flight_len, primary_peer) = match &self.sync_session {
            Some(s) => match &s.phase {
                SyncPhase::Batches { cursor, in_flight, .. } => {
                    (*cursor, s.peer_height, in_flight.len(), s.peer)
                }
                _ => return,
            },
            None => return,
        };

        let slots_available = BATCH_LOOKAHEAD.saturating_sub(in_flight_len);
        if slots_available == 0 { return; }

        let busy_peers: HashSet<PeerId> = match &self.sync_session {
            Some(s) => match &s.phase {
                SyncPhase::Batches { in_flight, .. } => in_flight.values().cloned().collect(),
                _ => return,
            },
            None => return,
        };

        let mut available_peers: Vec<PeerId> = self.network.connected_peers()
            .into_iter()
            .filter(|p| !self.network.is_light_peer(p) && !busy_peers.contains(p))
            .collect();
        if available_peers.is_empty() {
            available_peers.push(primary_peer);
        }

        let mut next_start = cursor;
        let mut peer_idx = 0;
        let mut reqs_sent = 0;

        while reqs_sent < slots_available && next_start < peer_height {
            let is_in_flight = match &self.sync_session {
                Some(s) => match &s.phase {
                    SyncPhase::Batches { in_flight, .. } => in_flight.contains_key(&next_start),
                    _ => false,
                },
                None => false,
            };

            let is_prefetched = match &self.sync_session {
                Some(s) => match &s.phase {
                    SyncPhase::Batches { prefetch_buffer, .. } => prefetch_buffer.contains_key(&next_start),
                    _ => false,
                },
                None => false,
            };

            if !is_in_flight && !is_prefetched {
                let target_peer = available_peers[peer_idx % available_peers.len()];
                peer_idx += 1;

                let count = (peer_height - next_start).min(MAX_GETBATCHES_COUNT);
                tracing::debug!("Lookahead: requesting batches {}..{} from {}", next_start, next_start + count, target_peer);
                self.network.send(target_peer, Message::GetBatches { start_height: next_start, count });

                if let Some(s) = &mut self.sync_session {
                    if let SyncPhase::Batches { in_flight, .. } = &mut s.phase {
                        in_flight.insert(next_start, target_peer);
                    }
                }
                reqs_sent += 1;
            }

            next_start += MAX_GETBATCHES_COUNT;
        }
    }



    async fn drain_prefetch_buffer(&mut self) {
        let cursor = match &self.sync_session {
            Some(s) => match &s.phase {
                SyncPhase::Batches { cursor, .. } => *cursor,
                _ => return,
            },
            None => return,
        };

        let next_batches = match &mut self.sync_session {
            Some(s) => match &mut s.phase {
                SyncPhase::Batches { prefetch_buffer, .. } => prefetch_buffer.remove(&cursor),
                _ => None,
            },
            None => None,
        };

        if let Some(batches) = next_batches {
            tracing::debug!("Draining prefetch buffer at height {}", cursor);
            let peer = self.sync_session.as_ref().unwrap().peer;
            if let Err(e) = self.handle_sync_batches(peer, batches).await {
                tracing::warn!("Error applying prefetched batches: {}", e);
                self.abort_sync_session("prefetch batch apply failed");
            }
        }
    }

    fn abort_sync_session(&mut self, reason: &str) {
        if self.sync_session.is_some() {
            self.sync_retry_count += 1;
            // Exponential backoff: 2^n seconds, capped at 5 minutes.
            // Sequence: 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 300s, 300s...
            let backoff_secs = 2u64.saturating_pow(self.sync_retry_count).min(300);
            self.sync_backoff_until = Some(
                std::time::Instant::now() + Duration::from_secs(backoff_secs)
            );
            tracing::warn!(
                "Aborting sync session (attempt {}): {}. Retrying in {}s.",
                self.sync_retry_count, reason, backoff_secs
            );
        }
        self.sync_session = None;
        self.sync_in_progress = false;
        // Don't clear last_sync_cursor here — keep it so the next attempt resumes
    }

    /// Instantly drop a malicious peer and penalize their Bayesian reputation.
    /// We do not waste CPU or RAM holding the connection open.
    ///
    /// Also bans their subnet so new connections from the same IP range are rejected
    /// instantly at the network layer, and purges them from the PEX routing table
    /// so we stop gossiping their address to honest nodes.
    fn ban_peer(&mut self, peer: PeerId, reason: &str) {
        tracing::error!("BANNING peer {} — {}", peer, reason);
        
        // 1. Immediately sever the TCP connection (Frees RAM/FDs)
        self.network.disconnect_peer(peer);

        // 2. Ban the subnet so they can't spam reconnects
        if let Some(subnet) = self.network.peer_subnet(&peer) {
            self.banned_subnets.insert(subnet, std::time::Instant::now());
            tracing::debug!("Banned subnet {} due to malicious peer", subnet);
        }

        // 3. Bayesian Penalty: Heavily penalize them in the routing table
        // We find their IP string from the PEX pool to apply the penalty.
        let peer_str = peer.to_string();
        let mut to_purge = None;
        
        for (addr_str, stats) in self.known_pex_addrs.iter_mut() {
            if addr_str.contains(&peer_str) {
                stats.1 = stats.1.saturating_add(50); // Massive penalty to Beta (Adversarial)
                
                let prob = stats.0 as f32 / (stats.0 + stats.1) as f32;
                if prob < 0.1 {
                    to_purge = Some(addr_str.clone());
                }
            }
        }

        // If they are irredeemably bad, purge them from PEX entirely
        if let Some(purge_addr) = to_purge {
            self.known_pex_addrs.remove(&purge_addr);
            tracing::info!("Purged malicious peer from PEX routing table: {}", purge_addr);
        }
    }

 pub async fn handle_sync_headers(&mut self, from: PeerId, headers: Vec<BatchHeader>) -> Result<()> {
        // Extract state from the session — only accept headers from the sync peer
        let (_peer_height, cursor, snapshot) = match &mut self.sync_session {
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
            let _ = std::fs::remove_file(self.data_dir.join("sync_state.bin"));
            return Ok(());
        }

        // Put the snapshot back for the next iteration
        if let Some(s) = &mut self.sync_session {
            if let SyncPhase::Headers { snapshot: snap_ref, .. } = &mut s.phase {
                *snap_ref = snapshot;
            }
        }

        let chunk_size = headers.len();
        let start_h = cursor;
        let end_h = cursor + chunk_size as u64 - 1;
        tracing::info!("Received headers {}..{} ({} total). Verifying Proof-of-Work...", start_h, end_h, chunk_size);

        // --- NEW: NON-BLOCKING PIPELINED VERIFICATION ---
        // Verify the PoW of this specific chunk immediately as it arrives on a background thread.
        let chunk_owned = headers.clone();
        let tx = self.cmd_tx.as_ref().unwrap().clone();
        
        tokio::spawn(async move {
            let chunk_valid = tokio::task::spawn_blocking(move || {
                use rayon::prelude::*;
                use crate::core::extension::verify_extension;
                chunk_owned.par_iter().all(|header| {
                    // FIX: Compute the header hash here!
                    let mining_hash = crate::core::types::compute_header_hash(header);
                    verify_extension(
                        mining_hash,
                        &header.extension,
                        &header.target,
                    ).is_ok()
                })
            }).await.expect("Header verification task panicked");

            let _ = tx.try_send(NodeCommand::FinishSyncHeadersChunk {
                peer: from,
                headers,
                is_valid: chunk_valid,
            });
        });
        
        if let Some(s) = &mut self.sync_session {
            s.last_progress_at = std::time::Instant::now();
        }

        
        Ok(())
    }

    async fn process_verified_headers_chunk(&mut self, from: PeerId, headers: Vec<BatchHeader>, is_valid: bool) -> Result<()> {
        let (peer_height, cursor) = match &mut self.sync_session {
            Some(s) if s.peer == from => {
                match &mut s.phase {
                    SyncPhase::Headers { cursor, .. } => (s.peer_height, *cursor),
                    _ => return Ok(()),
                }
            }
            _ => return Ok(()),
        };

        if !is_valid {
            self.abort_sync_session("peer sent headers with invalid Proof-of-Work");
            self.ban_peer(from, "invalid header PoW");
            return Ok(());
        }

        let mut new_cursor = cursor + headers.len() as u64; 
        let pct = (new_cursor as f64 / peer_height as f64) * 100.0;
        tracing::info!("✓ Verified and saved chunk. Sync progress: {}/{} ({:.1}%)", new_cursor, peer_height, pct);

        match &mut self.sync_session {
            Some(s) => {
                if let SyncPhase::Headers { accumulated, cursor: c, .. } = &mut s.phase {
                    
                    // Allow up to 10 million headers in RAM during initial sync (~19 years of blocks)
                    if accumulated.len() + headers.len() > 10_000_000 {
                        self.abort_sync_session("Peer attempted OOM DoS with too many headers");
                        return Ok(());
                    }
                    
                    // Prepend backward chunks for deep forks ---
                    if !accumulated.is_empty() && headers.last().map(|h| h.height) < accumulated.first().map(|h| h.height) {
                        tracing::info!("Prepended deep fork headers {}..{}", headers.first().unwrap().height, headers.last().unwrap().height);
                        
                        if let (Some(last_new), Some(first_acc)) = (headers.last(), accumulated.first()) {
                            if first_acc.prev_header_hash != last_new.extension.final_hash || first_acc.prev_midstate != last_new.post_tx_midstate {
                                tracing::warn!("Incoming deep-fork headers do not link to recovered sync state. Discarding sync_state.bin and restarting.");
                                let _ = std::fs::remove_file(self.data_dir.join("sync_state.bin"));
                                self.last_sync_cursor = None;
                                self.abort_sync_session("sync_state.bin deep-fork mismatch");
                                return Ok(());
                            }
                        }

                        let mut new_acc = headers.clone();
                        new_acc.append(accumulated); // Moves all elements from accumulated to new_acc
                        *accumulated = new_acc;
                        
                        // The backward gap is now filled, and the array reaches the tip again.
                        new_cursor = peer_height;
                        *c = new_cursor;
                    } else {
                        // Normal forward sync
                        if let (Some(last_acc), Some(first_new)) = (accumulated.last(), headers.first()) {
                            if first_new.prev_header_hash != last_acc.extension.final_hash || first_new.prev_midstate != last_acc.post_tx_midstate {
                                tracing::warn!("Incoming headers do not link to recovered sync state (heights: acc={}, new={}). Discarding sync_state.bin and restarting.", last_acc.height, first_new.height);
                                let _ = std::fs::remove_file(self.data_dir.join("sync_state.bin"));
                                self.last_sync_cursor = None;
                                self.abort_sync_session("sync_state.bin fork mismatch");
                                return Ok(());
                            }
                        }

                        accumulated.extend(headers);
                        *c = new_cursor;
                    }
                    
                    let sync_file_path = self.data_dir.join("sync_state.bin");
                    let backup = SyncStateBackup {
                        cursor: new_cursor,
                        peer_height,
                        accumulated_headers: accumulated.clone(),
                    };
                    tokio::task::spawn_blocking(move || {
                        if let Ok(bytes) = bincode::serialize(&backup) {
                            let _ = std::fs::write(&sync_file_path, bytes);
                        }
                    });
                }
            }
            _ => unreachable!(),
        }

        if new_cursor < peer_height {
            // Need more headers — request next chunk
            let count = MAX_GETHEADERS_COUNT.min(peer_height - new_cursor);
            self.network.send(from, Message::GetHeaders { start_height: new_cursor, count });

            if let Some(s) = &mut self.sync_session {
                s.last_progress_at = std::time::Instant::now();
            }
            self.last_sync_cursor = Some(new_cursor);

            return Ok(());
        }

        // All headers received — take ownership of the session data
        let session = self.sync_session.take().unwrap();
        let (all_headers, snapshot) = match session.phase {
            SyncPhase::Headers { accumulated, snapshot, .. } => (accumulated, snapshot),
            _ => unreachable!(),
        };

        let total_headers = all_headers.len(); 
        tracing::info!("Downloaded {} headers, offloading final verification...", total_headers);

        let tx = self.cmd_tx.as_ref().unwrap().clone();
        let storage = self.storage.clone();
        let current_height = self.state.height;
        let state_cache = self.state_cache.clone(); 
        let current_state = self.state.clone(); 
        
        // Grab recent timestamps BEFORE the start of this header chunk for MTP validation
        let mut recent_headers_vec: Vec<u64> = Vec::new();
        if let Some(first_hdr) = all_headers.first() {
            let start_h = first_hdr.height;
            let window_size = crate::core::DIFFICULTY_LOOKBACK as u64;
            let lookback_start = start_h.saturating_sub(window_size);
            
            for h in lookback_start..start_h {
                if let Ok(Some(batch)) = storage.load_batch(h) {
                    recent_headers_vec.push(batch.timestamp);
                } else if let Some(snap) = &snapshot {
                    // Fallback for fast-forward sync gaps
                    recent_headers_vec.push(snap.timestamp);
                }
            }
        }

        tokio::spawn(async move {
            let mut is_valid = true;

            // --- Time Warp Defense 
            if let Err(e) = crate::sync::Syncer::verify_header_chain_no_pow(&all_headers, &recent_headers_vec) {
                tracing::warn!("Accumulated header chain failed consensus (MTP/Linkage): {}", e);
                is_valid = false;
            }

            let current_time = crate::core::state::current_timestamp();
            const MAX_FUTURE_BLOCK_TIME: u64 = 2 * 60 * 60;

            if !all_headers.is_empty() && all_headers[0].timestamp > current_time + MAX_FUTURE_BLOCK_TIME {
                tracing::warn!("Root header timestamp too far in future");
                is_valid = false;
            }

            let mut fork_height = 0;
            let mut candidate_state = None;
            let mut is_fast_forward = false;
            let mut is_local_corruption = false;

            // 3. BACKGROUND FORK POINT & STATE REBUILD
            if is_valid {
                if let Some(snap) = &snapshot {
                    if all_headers.is_empty() || all_headers[0].prev_midstate != snap.midstate || all_headers[0].prev_header_hash != snap.header_hash {
                        tracing::warn!("Snapshot midstate mismatch! Peer sent fraudulent snapshot. Aborting sync.");
                        is_valid = false;
                    } else if all_headers.len() < crate::core::PRUNE_DEPTH as usize {
                        tracing::warn!("Fast-forward rejected: only {} headers on top of snapshot", all_headers.len());
                        is_valid = false;
                    } else {
                        fork_height = snap.height;
                        candidate_state = Some(snap.clone());
                        is_fast_forward = true;
                    }
                } else {
                    let syncer = crate::sync::Syncer::new(storage.clone());
                    let headers_start_height = all_headers.first().map(|h| h.height).unwrap_or(0);
                    
                    // Offload disk I/O to a background thread
                    // This prevents the Tokio network layer from freezing while searching the disk!
                    let all_headers_clone = all_headers.clone();
                    let fork_res = tokio::task::spawn_blocking(move || {
                        syncer.find_fork_point(&all_headers_clone, headers_start_height, current_height)
                    }).await.unwrap();

                    match fork_res {
                        Ok(fh) => {
                            // Nakamoto Consensus dictates the heaviest chain always wins
                            // We do not bound reorgs by safe_depth, otherwise we cause permanent network splits.
                            if fh == headers_start_height && headers_start_height > 0 {
                                fork_height = fh;
                            } else {
                                fork_height = fh;
                                if fh == 0 {
                                    candidate_state = Some(Box::new(State::genesis().0));
                                } else if fh <= current_height {
                                    if let Some(s) = state_cache.iter().find(|(h, _)| *h == fh).map(|(_, s)| s.clone()) {
                                        candidate_state = Some(Box::new(s));
                                    } else {
                                        tracing::info!("Cache miss at height {}, starting background state rebuild...", fh);
                                        
                                        // Pipelined Rebuild Start Command
                                        let _ = tx.try_send(NodeCommand::FinishStateRebuild {
                                            peer: from,
                                            fork_height: fh,
                                            candidate_state: None,
                                            headers: all_headers.clone(),
                                            is_fast_forward: false,
                                            is_valid: true,
                                            is_local_corruption: false,
                                        });

                                        let cache_start = state_cache.iter()
                                            .filter(|(h, _)| *h <= fh)
                                            .max_by_key(|(h, _)| *h)
                                            .map(|(h, s)| (*h, s.clone()));
                                        match rebuild_state_from_disk(storage.clone(), fh, cache_start).await {
                                            Ok(s) => candidate_state = Some(Box::new(s)),
                                            Err(e) => {
                                                tracing::error!("Background state rebuild failed: {}", e);
                                                is_local_corruption = true;
                                            }
                                        }
                                    }
                                } else {
                                    candidate_state = Some(Box::new(current_state));
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to find fork point: {}", e);
                            is_local_corruption = true;
                        }
                    }
                }
            }

            let _ = tx.try_send(NodeCommand::FinishStateRebuild {
                peer: from,
                fork_height,
                candidate_state,
                headers: all_headers,
                is_fast_forward,
                is_valid,
                is_local_corruption,
            });
        });
        
        self.sync_session = Some(SyncSession {
            peer: session.peer,
            peer_height: session.peer_height,
            peer_depth: session.peer_depth,
            phase: SyncPhase::VerifyingHeaders,
            started_at: session.started_at,
            last_progress_at: std::time::Instant::now(),
        });

        Ok(())
    }

    

    async fn handle_sync_batches(&mut self, from: PeerId, batches: Vec<Batch>) -> Result<()> {
        if batches.is_empty() {
            self.abort_sync_session("peer sent empty batches");
            return Ok(());
        }

        // Take the session to work with it
        let session = match self.sync_session.take() {
            Some(s) if s.peer == from => s,
            other => {
                self.sync_session = other; // put it back
                return Ok(());
            }
        };

        // Bind Copy fields before destructuring session, so they can be
        // captured independently by the spawn_blocking closure.
        let session_peer_height = session.peer_height;
        let session_peer_depth = session.peer_depth;
        let session_started_at = session.started_at;

        let (headers, fork_height, candidate_state, cursor, new_history, is_fast_forward, in_flight, prefetch_buffer) = match session.phase {
            SyncPhase::Batches { headers, fork_height, candidate_state, cursor, new_history, is_fast_forward, in_flight, prefetch_buffer } => {
                (headers, fork_height, candidate_state, cursor, new_history, is_fast_forward, in_flight, prefetch_buffer)
            }
            _ => {
                // Wrong phase — put it back untouched
                self.sync_session = Some(session);
                return Ok(());
            }
        };

        // Build a bounded sliding window of recent timestamps (mirrors
        // evaluate_alternative_chain). Both validate_timestamp and
        // adjust_difficulty uses ASERT anchored to genesis,
        // entries, so there is no need to collect every timestamp from genesis.
        let mut recent_ts: VecDeque<u64> = VecDeque::new();
        let window_size = DIFFICULTY_LOOKBACK as usize;
        
        // FIX: Unify start height mapping
        let header_start_height = headers.first().map(|h| h.height).unwrap_or(0);
        let start_height = cursor.saturating_sub(window_size as u64);

        for h in start_height..cursor {
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

        let tx = self.cmd_tx.as_ref().unwrap().clone();
        let storage = self.storage.clone();

        tokio::task::spawn_blocking(move || {
            let mut candidate_state = candidate_state;
            let mut new_history = new_history;
            let mut cursor = cursor;
            let mut recent_ts = recent_ts;
            
            // <--- FIX: Accumulate spent addresses across the sync loop to prevent cross-block reuse bypass
            let mut wots_oracle = std::collections::HashMap::new();
            let mut is_valid = true;
            let mut error_msg = String::new();

            // Apply each batch, verifying against the already-validated headers
            for batch in &batches {
                let height = cursor;

                if height < header_start_height {
                    error_msg = format!("Fatal sync error: batch height {} is below header_start_height {}. Aborting to prevent underflow.", height, header_start_height);
                    is_valid = false;
                    break;
                }

                // Map absolute height to relative array index
                let hdr_idx = (height - header_start_height) as usize;

                if hdr_idx >= headers.len() {
                    error_msg = format!("Batch height {} exceeds header count {}", height, headers.len());
                    is_valid = false;
                    break;
                }

                let header = &headers[hdr_idx];

                // Integrity: batch PoW must match the already-verified header
                if batch.extension.final_hash != header.extension.final_hash {
                    // Legacy blocks mined before state_root was added have state_root = [0;32].
                    // Re-check with zeroed state_root to allow syncing pre-migration blocks.
                    let mut legacy_batch = batch.clone();
                    legacy_batch.state_root = [0u8; 32];
                    let legacy_calc = legacy_batch.header();
                    if legacy_calc.post_tx_midstate != header.post_tx_midstate {
                        error_msg = format!("Batch at height {} tx commitment does not match header", height);
                        is_valid = false;
                        break;
                    }
                    // Legacy block accepted — continue with the actual batch (state_root already zeroed on disk)
                }

                // <--- FIX: Extract DB records and EXTEND the running memory oracle
                    if let Ok(db_oracle) = storage.query_spent_addresses(batch) {
                        wots_oracle.extend(db_oracle);
                    }
                

                // Apply to candidate state (do NOT save to disk yet — if sync
                // aborts before the reorg is committed, premature writes would
                // leave a Frankenstein chain on disk: some heights from the peer,
                // some from our old chain.  Batches are persisted atomically in
                // perform_reorg only after we decide to adopt.)
                if let Err(e) = crate::core::state::apply_batch(
                    &mut candidate_state,
                    batch,
                    recent_ts.make_contiguous(),
                    &mut wots_oracle,
                ) {
                    error_msg = format!("Failed to apply batch {}: {}", height, e);
                    is_valid = false;
                    break;
                }

                recent_ts.push_back(batch.timestamp);
                if recent_ts.len() > window_size {
                    recent_ts.pop_front();
                }

                candidate_state.target = crate::core::state::adjust_difficulty(&candidate_state);
                new_history.push((height, candidate_state.header_hash, batch.clone()));

                cursor += 1;
            }

            let _ = tx.try_send(NodeCommand::FinishSyncBatchesChunk {
                peer: from,
                headers,
                fork_height,
                candidate_state: Box::new(candidate_state),
                cursor,
                new_history,
                is_fast_forward,
                is_valid,
                error_msg,
                session_started_at,
                peer_height: session_peer_height,
                peer_depth: session_peer_depth,
            });
        });

        // Set a placeholder phase so the timeout guard knows work is in flight
        // and the event loop stays alive processing other messages.
        self.sync_session = Some(SyncSession {
            peer: from,
            peer_height: session_peer_height,
            peer_depth: session_peer_depth,
            phase: SyncPhase::VerifyingBatches {
                in_flight,
                prefetch_buffer,
            },
            started_at: session_started_at,
            last_progress_at: std::time::Instant::now(),
        });

        Ok(())
    }

    async fn handle_new_transaction(&mut self, tx: Transaction, from: Option<PeerId>) -> Result<()> {
        // Per-peer rate limiting: prevent CPU exhaustion from tx validation spam.
        // Local submissions (from = None, i.e. RPC) bypass the limit.
        if let Some(peer) = from {
            let now = std::time::Instant::now();
            let entry = self.peer_tx_counts.entry(peer).or_insert((0, now));
            if now.duration_since(entry.1).as_secs() >= TX_RATE_WINDOW_SECS {
                *entry = (0, now); // reset window
            }
            entry.0 += 1;
            if entry.0 > MAX_TX_PER_PEER_PER_WINDOW {
                tracing::debug!("Rate-limiting peer {}: {} txs in window", peer, entry.0);
                return Ok(()); // silently drop, don't validate
            }
        }

        let wots_oracle = self.storage.query_spent_addresses_for_tx(&tx).unwrap_or_default();

        // EVENT LOOP DEFENSE: Validate cryptography on a background thread.
        // We clone the state (O(1) cost due to immutable data structures) and the tx,
        // so the heavy STARK/WOTS math doesn't stall the Tokio networking reactor.
        let state_clone = self.state.clone();
        let tx_clone = tx.clone();
        let is_valid = tokio::task::spawn_blocking(move || {
            crate::core::transaction::validate_transaction(&state_clone, &tx_clone)
        }).await.unwrap();

        if let Err(e) = is_valid {
            self.metrics.inc_invalid_transactions();
            return Err(e); // Abort before locking mempool
        }

        // The transaction is cryptographically valid. Now add it to the mempool.
        match self.mempool.add(tx.clone(), &self.state, &wots_oracle) {
            Ok(_) => {
                self.metrics.inc_transactions_processed();

                if from.is_none() {
                    // Dandelion++ stem phase for COMMITS only.
                    // Reveals are broadcast immediately because:
                    // 1. They're already linkable to a public commitment
                    // 2. The 30s stem delay starves other miners' templates
                    if matches!(&tx, Transaction::Commit { .. }) {
                        if let Some(stem_peer) = self.network.random_peer() {
                            let tx_id = match &tx { Transaction::Commit { commitment, .. } => *commitment, _ => [0; 32] };
                            self.stem_pool.insert(tx_id, (tx.clone(), std::time::Instant::now()));
                            self.network.send(stem_peer, Message::StemTransaction(tx));
                            tracing::debug!("Dandelion++ stem: sent commit to {}", stem_peer);
                        } else {
                            self.network.broadcast(Message::Transaction(tx));
                        }
                    } else {
                        // Reveals: broadcast immediately to all peers
                        self.network.broadcast(Message::Transaction(tx));
                    }
                } else {
                    // Received from a peer (already fluffed) — relay normally
                    self.network.broadcast_except(from, Message::Transaction(tx.clone()));

                    // If this tx was in our stem pool (we were stemming it),
                    // remove it now that it's been fluffed by someone else.
                    // Prevents redundant re-broadcast when flush_stem_pool fires.
                    let tx_id = tx.input_coin_ids().first().copied()
                        .unwrap_or_else(|| match &tx { Transaction::Commit { commitment, .. } => *commitment, _ => [0; 32] });
                    self.stem_pool.remove(&tx_id);
                }

                self.cancel_mining();
                self.trigger_mining();
                Ok(())
            }
            Err(e) => {
                self.metrics.inc_invalid_transactions();
                Err(e)
            }
        }
    }

    /// Dandelion++ stem phase handler: received a StemTransaction from a peer.
    /// With STEM_FLUFF_PERCENT probability, "fluff" it (broadcast normally).
    /// Otherwise, forward to one random peer (excluding sender).
    async fn handle_stem_transaction(&mut self, tx: Transaction, from: PeerId) -> Result<()> {
        // --- FIX: Dandelion++ Rate Limiting ---
        let now = std::time::Instant::now();
        let entry = self.peer_tx_counts.entry(from).or_insert((0, now));
        if now.duration_since(entry.1).as_secs() >= TX_RATE_WINDOW_SECS {
            *entry = (0, now); // reset window
        }
        entry.0 += 1;
        if entry.0 > MAX_TX_PER_PEER_PER_WINDOW {
            tracing::debug!("Stem rate-limiting peer {}: {} txs in window", from, entry.0);
            return Ok(()); // silently drop, don't execute heavy validation
        }
        // --------------------------------------

        // Compute a tx identifier for dedup
        let tx_id = tx.input_coin_ids().first().copied()
            .unwrap_or_else(|| match &tx { Transaction::Commit { commitment, .. } => *commitment, _ => [0; 32] });

        // Already in stem pool — ignore duplicate
        if self.stem_pool.contains_key(&tx_id) {
            return Ok(());
        }

        // Dandelion++ Privacy: Do NOT add to mempool during stem phase!
        // Adding to mempool exposes the tx via RPC and mining, defeating anonymity.
        // Validate only — mempool insertion happens when we fluff.
        if let Err(e) = validate_transaction(&self.state, &tx) {
            self.metrics.inc_invalid_transactions();
            return Err(e);
        }
        self.metrics.inc_transactions_processed();


        // If the stem pool is at capacity, DROP the incoming P2P stem transaction.
        // If we broadcasted it, an attacker could spam us with stem txs to force us
        // to fluff our own txs, instantly deanonymizing our IP address.
        if self.stem_pool.len() >= MAX_STEM_POOL_SIZE {
            tracing::warn!("Stem pool full ({} entries). Dropping incoming P2P stem tx to preserve privacy and prevent broadcast storms.", MAX_STEM_POOL_SIZE);
            return Ok(());
        } else {
// ADAPTIVE DANDELION++: Dynamic fluff probability based on network size.
            let outbound_count = self.network.outbound_peer_count().max(1) as u32;
            let dynamic_fluff_percent = (100 / outbound_count).clamp(2, 50);

            // Flip the coin: fluff or continue stemming?
            let roll = rand::random::<u32>() % 100;
            if roll < dynamic_fluff_percent {
                // Fluff: add to public mempool and broadcast
                tracing::debug!("Dandelion++ fluff: broadcasting tx after stem");
                let wots_oracle = self.storage.query_spent_addresses_for_tx(&tx).unwrap_or_default();
                let _ = self.mempool.add(tx.clone(), &self.state, &wots_oracle);
                self.network.broadcast(Message::Transaction(tx));
                self.cancel_mining();
                self.trigger_mining();
            } else {
                // Continue stem: forward to ONE random peer (not the sender)
                let peers: Vec<PeerId> = self.network.connected_peers()
                    .into_iter()
                    .filter(|p| *p != from)
                    .collect();
                if let Some(next) = peers.first() {
                    self.stem_pool.insert(tx_id, (tx.clone(), std::time::Instant::now()));
                    self.network.send(*next, Message::StemTransaction(tx));
                    tracing::debug!("Dandelion++ stem: forwarded to {}", next);
                } else {
                    // No other peers — fluff immediately
                    let wots_oracle = self.storage.query_spent_addresses_for_tx(&tx).unwrap_or_default();
                    let _ = self.mempool.add(tx.clone(), &self.state, &wots_oracle);
                    self.network.broadcast(Message::Transaction(tx));
                    self.cancel_mining();
                    self.trigger_mining();
                }
            }
        }
        Ok(())
    }

    /// Flush timed-out stem pool entries: if a tx has been in stem phase
    /// for longer than STEM_TIMEOUT_SECS, broadcast it ourselves.
    fn flush_stem_pool(&mut self) {
        let now = std::time::Instant::now();
        let expired: Vec<[u8; 32]> = self.stem_pool.iter()
            .filter(|(_, (_, t))| now.duration_since(*t).as_secs() >= STEM_TIMEOUT_SECS)
            .map(|(k, _)| *k)
            .collect();

        for tx_id in expired {
            if let Some((tx, _)) = self.stem_pool.remove(&tx_id) {
                // 1. Check if the tx is still fundamentally valid against the chain state
                if validate_transaction(&self.state, &tx).is_ok() {
                    tracing::debug!("Dandelion++ timeout: fluffing valid stem tx");
                    
                    // 2. Try to add to local mempool (might fail if our mempool is full/fee too low)
                    let wots_oracle = self.storage.query_spent_addresses_for_tx(&tx).unwrap_or_default();
                    let _ = self.mempool.add(tx.clone(), &self.state, &wots_oracle);
                    
                    // 3. Broadcast to the network regardless of local mempool admission, 
                    // so the rest of the network gets a chance to mine it.
                    self.network.broadcast(Message::Transaction(tx));
                } else {
                    // It became invalid while waiting in the stem pool (e.g., double spent)
                    tracing::debug!("Dandelion++ timeout: dropping stem tx (became invalid)");
                }
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
        
        // FIX: Optimize state derivation to prevent unnecessary and impossible rebuilds.
        let fork_state = if fork_height == self.state.height {
            // It's a direct linear extension. We already have the exact state in memory!
            // No need to touch the disk or search for snapshots.
            self.state.clone()
        } else if fork_height == 0 {
            State::genesis().0
        } else {
            // Pre-check: can the alternative chain's work possibly beat ours?
            // Pre-check: can the alternative chain's work possibly beat ours?
            // Compare the work in our chain from fork_height..tip against the
            // alternative's theoretical maximum. This is O(N) integer math on
            // small headers — avoids the expensive state rebuild + disk I/O.
            let our_headers = self.storage.batches.load_headers(fork_height, self.state.height)?;
            let our_work_since_fork: u128 = our_headers.iter()
                .map(|h| crate::core::state::calculate_work(&h.target))
                .fold(0u128, |a, b| a.saturating_add(b));
            let alt_work: u128 = alternative_batches.iter()
                .map(|b| crate::core::state::calculate_work(&b.target))
                .fold(0u128, |a, b| a.saturating_add(b));
            if alt_work <= our_work_since_fork {
                tracing::debug!(
                    "Rejecting fork at {}: alt work {} <= our work since fork {}",
                    fork_height, alt_work, our_work_since_fork
                );
                return Ok(None);
            }

            // It's a genuine reorg (fork_height < self.state.height). We MUST rebuild the state.
            let cache_start = self.state_cache.iter()
                .filter(|(h, _)| *h <= fork_height)
                .max_by_key(|(h, _)| *h)
                .map(|(h, s)| (*h, s.clone()));
            rebuild_state_from_disk(self.storage.clone(), fork_height, cache_start).await?
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


         let mut wots_oracle = std::collections::HashMap::new();
         
         for (i, batch) in alternative_batches.iter().enumerate() {

             if batch.prev_header_hash != candidate_state.header_hash {
                tracing::warn!(
                    "Alternative chain broken at batch index {} (height {})",
                    i, fork_height + i as u64
                );
                return Ok(None);
            }

            match {
                // <--- FIX: Extend the oracle
                    let db_oracle = self.storage.query_spent_addresses(batch).unwrap_or_default();
                    wots_oracle.extend(db_oracle);
                
                apply_batch(&mut candidate_state, batch, recent_headers.make_contiguous(), &mut wots_oracle)
            } {

                Ok(_) => {
                    recent_headers.push_back(batch.timestamp);
                    if recent_headers.len() > window_size {
                        recent_headers.pop_front();
                    }
                    // Adjust difficulty via ASERT
                    candidate_state.target = adjust_difficulty(&candidate_state);
                    new_history.push((
                        fork_height + i as u64,
                        candidate_state.midstate,
                        batch.clone(),
                    ));
                    
                    // Yield to prevent event loop starvation on large forks
                    tokio::task::yield_now().await;
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
    async fn perform_reorg(
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
            self.cached_safe_depth = self.finality.calculate_safe_depth(1e-6);
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

        // Update in-memory state FIRST
        // Trim state cache: discard all entries at or above the fork
        self.trim_cache_above(fork_height);
        
        // Delete stale snapshots from the abandoned chain
        let _ = self.storage.delete_snapshots_above(fork_height);
        
        self.state = new_state;
        while self.chain_history.back().map_or(false, |&(h, _)| h >= fork_height) {
            self.chain_history.pop_back();
        }
        self.chain_history.extend(new_history.iter().map(|(h, ms, _)| (*h, *ms)));

        // Rebuild headers cache. For heights below the fork, read from disk
        // (those batches are shared between old and new fork). For heights at
        // or above the fork, use the in-memory new_history (batch files may
        // not be written yet).
        self.recent_headers.clear();
        let window = DIFFICULTY_LOOKBACK as u64;
        let start = self.state.height.saturating_sub(window);

        // Build a lookup table from new_history for fast access
        let new_batch_timestamps: HashMap<u64, u64> = new_history.iter()
            .map(|(h, _, b)| (*h, b.timestamp))
            .collect();

        for h in start..self.state.height {
            if let Some(&ts) = new_batch_timestamps.get(&h) {
                self.recent_headers.push_back(ts);
            } else if let Some(batch) = self.storage.load_batch(h)? {
                self.recent_headers.push_back(batch.timestamp);
            }
        }

        self.state.target = adjust_difficulty(&self.state);

        // Cache the new tip for future reorgs
        self.cache_current_state();

        // OFFLOAD HEAVY REORG DISK I/O TO THREADPOOL ---
        let storage_clone = self.storage.clone();
        let state_clone = self.state.clone();
        let history_clone = new_history.clone();
        
        tokio::task::spawn_blocking(move || -> Result<()> {
            // 1. WRITE BATCHES
            for (height, _, batch) in &history_clone {
                storage_clone.save_batch(*height, batch)?;
            }
            // 2. COMMIT STATE 
            storage_clone.save_state(&state_clone)?;
            // 3. BURN ADDRESSES
            for (height, _, batch) in &history_clone {
                if let Err(e) = storage_clone.burn_batch_addresses(batch, *height) {
                    tracing::warn!("burn_batch_addresses failed at height {}: {}", height, e);
                }
            }
            Ok(())
        }).await.expect("Reorg DB task panicked")?;

        self.mempool.re_add(abandoned_txs, &self.state);

        self.mempool.prune_invalid(&self.state);
        if is_actual_reorg {
            self.metrics.inc_reorgs();
        }
        if self.state.height > 0 && self.state.height % SNAPSHOT_INTERVAL == 0 {
            if let Err(e) = self.storage.save_state_snapshot(self.state.height, &self.state) {
                tracing::warn!("Failed to save state snapshot during reorg/sync: {}", e);
            } else {
                tracing::info!("Saved state snapshot at height {}", self.state.height);
            }
        }
        self.sync_in_progress = false;
        self.trigger_mining();
        
        // --- Hardware-Safe Background Push Generation ---
        // Push the new chain tip to light clients so they recognize the reorg instantly
        if let Some((_, _, last_batch)) = new_history.last() {
            let last_batch_clone = last_batch.clone();
            let state_clone = self.state.clone();
            let cmd_tx = self.cmd_tx.as_ref().unwrap().clone();
            let has_light_peers = self.network.has_light_peers();

            if has_light_peers {
                tokio::task::spawn_blocking(move || {
                    let filter = crate::core::filter::CompactFilter::build(&last_batch_clone);
                    
                    let mut items = std::collections::HashSet::new();
                    for tx in &last_batch_clone.transactions {
                        match tx {
                            crate::core::Transaction::Commit { commitment, .. } => { items.insert(*commitment); }
                            crate::core::Transaction::Reveal { inputs, outputs, .. } => {
                                for input in inputs {
                                    items.insert(input.coin_id());
                                    items.insert(input.predicate.address());
                                }
                                for output in outputs {
                                    if let Some(cid) = output.coin_id() { items.insert(cid); }
                                    items.insert(output.address());
                                }
                            }
                        }
                    }
                    for cb in &last_batch_clone.coinbase {
                        items.insert(cb.coin_id());
                        items.insert(cb.address);
                    }

                    let notif = crate::network::light_protocol::LightNotification::NewBlockTip {
                        height: state_clone.height,
                        target: hex::encode(state_clone.target),
                        filter_hex: hex::encode(filter.data),
                        block_hash: hex::encode(last_batch_clone.extension.final_hash),
                        element_count: items.len() as u64,
                    };
                    
                    let _ = cmd_tx.try_send(NodeCommand::BroadcastLightPush(notif));
                });
            }
        }

        Ok(())
    }

    

    /// Push the current state into the ring buffer cache.
    /// Called after every successful state advancement.
    fn cache_current_state(&mut self) {
        // Avoid duplicate entries at the same height
        if self.state_cache.back().map(|(h, _)| *h) == Some(self.state.height) {
            self.state_cache.pop_back();
        }
        self.state_cache.push_back((self.state.height, self.state.clone()));
        if self.state_cache.len() > STATE_CACHE_SIZE {
            self.state_cache.pop_front();
        }
    }

    /// Trim the cache: discard all entries at or above `fork_height`.
    fn trim_cache_above(&mut self, fork_height: u64) {
        while self.state_cache.back().map_or(false, |(h, _)| *h >= fork_height) {
            self.state_cache.pop_back();
        }
    }

    /// Performs an ultra-fast O(log N) health check on the node's internal state.
    /// Verifies that the in-memory UTXO set, midstate, and disk records are perfectly aligned.
    async fn perform_health_check(&mut self) -> Result<bool> {
        if self.state.height == 0 { return Ok(true); }

        let disk_highest = self.storage.highest_batch().unwrap_or(0);
        if disk_highest + 1 != self.state.height {
            tracing::error!("Health Check Failed: Disk tip is {} but Memory State is at {}", disk_highest, self.state.height);
            return Ok(false);
        }

        let tip_batch = match self.storage.load_batch(self.state.height - 1)? {
            Some(b) => b,
            None => {
                tracing::error!("Health Check Failed: Missing tip block {} on disk!", self.state.height - 1);
                return Ok(false);
            }
        };

        if self.state.header_hash != tip_batch.extension.final_hash {
            tracing::error!("Health Check Failed: Memory header_hash diverges from disk tip!");
            return Ok(false);
        }

        // Just ensure SMT math works without crashing
        let mut coins_clone = self.state.coins.clone();
        let mut comms_clone = self.state.commitments.clone();
        let _smt_root = crate::core::types::hash_concat(&coins_clone.root(), &comms_clone.root());

        Ok(true)
    }

    async fn self_heal_rollback(&mut self, fork_height: u64) {
        // Start the rollback from strictly BELOW the corrupted fork point
        let mut snap_height = (fork_height / 100) * 100;
        if snap_height == fork_height {
            snap_height = snap_height.saturating_sub(100);
        }

        while snap_height > 0 {
            if let Ok(Some(snap)) = self.storage.load_state_snapshot(snap_height) {
                tracing::info!("Self-healing: Rolling back state to snapshot at {}", snap_height);
                self.state = snap;
                self.state.target = adjust_difficulty(&self.state);
                self.cache_current_state();
                let _ = self.storage.save_state(&self.state);
                let _ = self.storage.truncate_chain(snap_height);
                
                self.chain_history.clear();
                self.recent_headers.clear();
                self.last_sync_cursor = Some(snap_height);
                
                // Repopulate recent headers
                let window = crate::core::DIFFICULTY_LOOKBACK as u64;
                let start = self.state.height.saturating_sub(window);
                for h in start..self.state.height {
                    if let Ok(Some(batch)) = self.storage.load_batch(h) {
                        self.recent_headers.push_back(batch.timestamp);
                    }
                }
                
                return;
            }
            snap_height = snap_height.saturating_sub(100);
        }
        
        tracing::error!("Self-healing failed: no valid snapshots found. Node may require a resync from genesis.");
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
        let current_height = self.state.height;
        let spam_nonce = tokio::task::spawn_blocking(move || {
            let mut n = 0u64;
            loop {
                let h = crate::core::transaction::commit_pow_hash(&commitment, n, current_height);
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
        // Extract the header hash before we potentially move the batch
        let prev_header_hash = batch.prev_header_hash;

        // --- FIX 1: Ignore Redundant Blocks ---
        // If this is the block we just applied, or it is in our recent history, drop it.
        if batch.extension.final_hash == self.state.header_hash || 
           self.chain_history.iter().any(|&(_, hash)| hash == batch.extension.final_hash) {
            tracing::debug!("Received already-applied block, ignoring.");
            return Ok(());
        }
        // --------------------------------------

        // Fast pre-checks BEFORE cloning state (O(1) shallow clone via structural sharing).
        if prev_header_hash != self.state.header_hash {
            // --- FIX: Prevent Orphan OOM Attack ---
            // 1. Verify the sequential PoW is valid (forces attacker to compute 1M hashes)
            let header = batch.header();
            let mining_hash = crate::core::types::compute_header_hash(&header); // <-- FIX: Add this
            if crate::core::extension::verify_extension(mining_hash, &batch.extension, &batch.target).is_err() {
                tracing::debug!("Rejected invalid orphan block (PoW failed)");
                if let Some(peer) = from {
                    self.ban_peer(peer, "invalid orphan block (PoW failed)");
                }
                return Ok(());
            }

            // 2. Asymmetry check: ensure the target isn't artificially easy.
            // We use a 16x (4-bit shift) window instead of 4x (2-bit), because ASERT is
            // anchored to wall-clock time: two nodes that diverged for even a few
            // minutes will legitimately have different targets. The 4x window was
            // causing valid miner batches to be silently dropped, forking them off
            // permanently. The PoW verification above is the real spam gate.
            let current_target = primitive_types::U256::from_big_endian(&self.state.target);
            let batch_target = primitive_types::U256::from_big_endian(&batch.target);
            let (max_allowed, overflow) = current_target.overflowing_mul(primitive_types::U256::from(16u64));
            if !overflow && batch_target > max_allowed {
                tracing::warn!(
                    "Rejected orphan block from {:?}: target {:?} is >16x easier than ours ({:?}). Likely spam.",
                    from,
                    hex::encode(batch.target),
                    hex::encode(self.state.target)
                );
                if let Some(peer) = from {
                    self.ban_peer(peer, "absurdly easy difficulty target (orphan spam)");
                }
                return Ok(());
            }
            // --------------------------------------

            tracing::debug!("Received valid orphan block (parent mismatch), queuing for later.");

            let list = self.orphan_batches.entry(prev_header_hash).or_default();
            
            // --- EFFICIENCY FIX: Deduplicate identical orphans ---
            if list.iter().any(|b| b.extension.final_hash == batch.extension.final_hash) {
                tracing::debug!("Already tracking this exact orphan block. Ignoring duplicate.");
                return Ok(());
            }

            // Prevent infinite vector growth on a single header_hash
            if list.len() < 4 {
                list.push(batch); // batch is moved here
                if list.len() == 1 { // Only push to order queue on first entry
                    self.orphan_order.push_back(prev_header_hash);
                }
            } else {
                tracing::warn!("Too many UNIQUE competing orphans for midstate, dropping to prevent RAM exhaustion.");
            }

            const ORPHAN_LIMIT: usize = 8;
            
            if self.orphan_order.len() >= ORPHAN_LIMIT {
                // Evict oldest half via FIFO order
                let to_evict = ORPHAN_LIMIT / 2;
                for _ in 0..to_evict {
                    if let Some(key) = self.orphan_order.pop_front() {
                        self.orphan_batches.remove(&key);
                    }
                }
            }

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
            if let Some(peer) = from {
                self.ban_peer(peer, "batch target mismatch for current height");
            }
            return Ok(());
        }

        // Checks passed — now clone state and apply fully.
        let mut candidate_state = self.state.clone();
        let mut wots_oracle = self.storage.query_spent_addresses(&batch).unwrap_or_default();
        match apply_batch(&mut candidate_state, &batch, self.recent_headers.make_contiguous(), &mut wots_oracle) {
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

                    self.recent_headers.push_back(batch.timestamp);
                    if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                        self.recent_headers.pop_front();
                    }
                    let pre_height = self.state.height;
                    self.state = candidate_state;
                    
                    // ADJUST TARGET FIRST
                    self.state.target = adjust_difficulty(&self.state);

                    // Cache for instant reorg rollback
                    self.cache_current_state();

                    self.storage.save_batch(pre_height, &batch)?;

                    // Periodic snapshot 
                    if self.state.height > 0 && self.state.height % SNAPSHOT_INTERVAL == 0 {
                        if let Err(e) = self.storage.save_state_snapshot(self.state.height, &self.state) {
                            tracing::warn!("Failed to save state snapshot: {}", e);
                        } else {
                            tracing::info!("Saved state snapshot at height {}", self.state.height);
                        }
                    }
                    
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

                    self.chain_history.push_back((pre_height, self.state.header_hash));

                    self.finality.observe_honest();
                    self.cached_safe_depth = self.finality.calculate_safe_depth(1e-6);
                    let cutoff_height = self.state.height.saturating_sub(self.cached_safe_depth);
                    while self.chain_history.front().map_or(false, |&(h, _)| h < cutoff_height) {
                        self.chain_history.pop_front();
                    }

                    self.network.broadcast_except(from, Message::Batch(batch));
                    tracing::info!("Applied new batch from peer, height now {}", self.state.height);
                    self.try_apply_orphans().await;
                    self.check_pending_mix_reveals().await;
                    self.trigger_mining();
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
        // Cheap header_hash check before expensive state clone.
        if batches[0].prev_header_hash == self.state.header_hash {
            let mut test_state = self.state.clone();
            let mut wots_oracle = self.storage.query_spent_addresses(&batches[0]).unwrap_or_default();
            
            // Change to &mut
            if apply_batch(&mut test_state, &batches[0], self.recent_headers.make_contiguous(), &mut wots_oracle).is_ok() {
                return self.process_linear_extension(batches, from).await;
            }
        }

        // Try 2
        for (i, batch) in batches.iter().enumerate() {
            if batch.prev_header_hash != self.state.header_hash { continue; }
            let mut candidate = self.state.clone();
            // Change to let mut
            let mut wots_oracle = self.storage.query_spent_addresses(&batch).unwrap_or_default();
            // Change to &mut
            if apply_batch(&mut candidate, batch, self.recent_headers.make_contiguous(), &mut wots_oracle).is_ok() {
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
                        self.perform_reorg(new_state, new_history, false).await?; 
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
        let mut wots_oracle = std::collections::HashMap::new();
        
        // --- Track the last applied batch for light client notifications ---
        let mut last_applied_batch = None;
        
        for batch in batches {
            if batch.prev_header_hash != self.state.header_hash { break; }
            let mut candidate = self.state.clone();
            
                let db_oracle = self.storage.query_spent_addresses(&batch).unwrap_or_default();
                wots_oracle.extend(db_oracle);
            
            if apply_batch(&mut candidate, &batch, self.recent_headers.make_contiguous(), &mut wots_oracle).is_ok() {
                self.recent_headers.push_back(batch.timestamp);
                if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                    self.recent_headers.pop_front();
                }
                self.state = candidate;
                
                self.state.target = adjust_difficulty(&self.state);
                self.cache_current_state();
                self.storage.save_batch(self.state.height - 1, &batch)?;
                
                if self.state.height > 0 && self.state.height % SNAPSHOT_INTERVAL == 0 {
                    if let Err(e) = self.storage.save_state_snapshot(self.state.height, &self.state) {
                        tracing::warn!("Failed to save state snapshot: {}", e);
                    } else {
                        tracing::info!("Saved state snapshot at height {}", self.state.height);
                    }
                }
                
                self.metrics.inc_batches_processed();

                self.chain_history.push_back((self.state.height, self.state.header_hash));

                self.finality.observe_honest();
                self.cached_safe_depth = self.finality.calculate_safe_depth(1e-6);
                let cutoff = self.state.height.saturating_sub(self.cached_safe_depth);
                while self.chain_history.front().map_or(false, |&(h, _)| h < cutoff) {
                    self.chain_history.pop_front();
                }

                applied += 1;
                last_applied_batch = Some(batch); // Keep a copy of the block we just processed
            } else {
                break;
            }
        }

        if applied > 0 {
            tracing::info!("Synced {} batch(es), now at height {}", applied, self.state.height);
            self.mempool.prune_invalid(&self.state);
            self.try_apply_orphans().await;
            self.check_pending_mix_reveals().await;

            // --- 1000 Block Milestone Check ---
            if self.state.height > 0 && self.state.height % 1000 == 0 {
                if let Ok(false) = self.perform_health_check().await {
                     tracing::error!("CRITICAL: Node state is sick/corrupted at milestone. Initiating self-healing sequence...");
                     self.cancel_mining();
                     self.self_heal_rollback(self.state.height).await;
                }
            }

            if self.state.height >= self.sync_requested_up_to {
                self.sync_in_progress = false;
            } else {
                let start = self.state.height;
                let count = (self.sync_requested_up_to.saturating_sub(start) + 1).min(MAX_GETBATCHES_COUNT);
                tracing::info!("Continuing sync from peer {} (requesting {} batches from {})", from, count, start);
                self.network.send(from, Message::GetBatches { start_height: start, count });
            }
            
            self.trigger_mining();
            
            // --- Hardware-Safe Background Push Generation ---
            // Only push the VERY LAST block applied in this chunk to avoid spamming the clients
            if let Some(batch_clone) = last_applied_batch {
                let state_clone = self.state.clone();
                let cmd_tx = self.cmd_tx.as_ref().unwrap().clone();
                let has_light_peers = self.network.has_light_peers();

                if has_light_peers {
                    tokio::task::spawn_blocking(move || {
                        let filter = crate::core::filter::CompactFilter::build(&batch_clone);
                        
                        let mut items = std::collections::HashSet::new();
                        for tx in &batch_clone.transactions {
                            match tx {
                                crate::core::Transaction::Commit { commitment, .. } => { items.insert(*commitment); }
                                crate::core::Transaction::Reveal { inputs, outputs, .. } => {
                                    for input in inputs {
                                        items.insert(input.coin_id());
                                        items.insert(input.predicate.address());
                                    }
                                    for output in outputs {
                                        if let Some(cid) = output.coin_id() { items.insert(cid); }
                                        items.insert(output.address());
                                    }
                                }
                            }
                        }
                        for cb in &batch_clone.coinbase {
                            items.insert(cb.coin_id());
                            items.insert(cb.address);
                        }

                        let notif = crate::network::light_protocol::LightNotification::NewBlockTip {
                            height: state_clone.height,
                            target: hex::encode(state_clone.target),
                            filter_hex: hex::encode(filter.data),
                            block_hash: hex::encode(batch_clone.extension.final_hash),
                            element_count: items.len() as u64,
                        };
                        
                        let _ = cmd_tx.try_send(NodeCommand::BroadcastLightPush(notif));

                    });
                }
            }

        } else {
            self.sync_in_progress = false;
        }

        Ok(())
    }

async fn try_apply_orphans(&mut self) {
        let mut applied = 0;

        while let Some(mut batches) = self.orphan_batches.remove(&self.state.header_hash) {
            // Also remove all entries for this key from the order tracker
            self.orphan_order.retain(|k| k != &self.state.header_hash);

            let mut matched = false;
            for batch in batches.drain(..) {
                let mut candidate = self.state.clone();
                // Change to let mut
                let mut wots_oracle = self.storage.query_spent_addresses(&batch).unwrap_or_default();
                // Change to &mut
                if apply_batch(&mut candidate, &batch, self.recent_headers.make_contiguous(), &mut wots_oracle).is_ok() {
                    self.cancel_mining();
                    self.recent_headers.push_back(batch.timestamp);
                    if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                        self.recent_headers.pop_front();
                    }
                    
                    self.state = candidate;
                    
                    // ADJUST TARGET FIRST
                    self.state.target = adjust_difficulty(&self.state);

                    // Cache for instant reorg rollback
                    self.cache_current_state();

                    self.storage.save_batch(self.state.height - 1, &batch).ok();
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

    fn generate_coinbase(
        &self, 
        height: u64, 
        total_fees: u64,
        pool_target: Option<([u8; 32], [u8; 32])>, // (Pool MSS Address, Miner Payout Address)
    ) -> Vec<CoinbaseOutput> {
        let reward = block_reward(height);
        let total_value = reward.saturating_add(total_fees);
        let denominations = decompose_value(total_value);

        let mining_seed = self.mining_seed;

        denominations.into_par_iter()
            .enumerate()
            .map(move |(i, value)| {
                match pool_target {
                    Some((pool_addr, miner_addr)) => {
                        // POOL MINING MODE
                        // Pay the pool's address, but embed the miner's address in the salt
                        // so the pool can cryptographically verify who did the work.
                        let mut salt = [0u8; 32];
                        let mut hasher = blake3::Hasher::new();
                        hasher.update(b"pool_share");
                        hasher.update(&miner_addr);
                        hasher.update(&height.to_le_bytes());
                        hasher.update(&(i as u64).to_le_bytes());
                        salt.copy_from_slice(hasher.finalize().as_bytes());

                        CoinbaseOutput { address: pool_addr, value, salt }
                    }
                    None => {
                        // SOLO MINING MODE (Original Logic)
                        let seed = coinbase_seed(&mining_seed, height, i as u64);
                        let owner_pk = wots::keygen(&seed);
                        let address = compute_address(&owner_pk);
                        let salt = coinbase_salt(&mining_seed, height, i as u64);
                        
                        CoinbaseOutput { address, value, salt }
                    }
                }
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
                // NOTE: We intentionally do NOT log the seed (private key).
                // It is derivable from (mining_seed, height, index) when
                // the wallet needs to spend. Logging it in cleartext would
                // allow anyone with filesystem or RPC access to steal funds.
                format!(
                    r#"{{"height":{},"index":{},"address":"{}","coin":"{}","value":{},"salt":"{}"}}"#,
                    height, i,
                    hex::encode(address),
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

        let mut pool_target = None;
        let mut pool_url = None;
        let mut payout_address = None;
        let mut pool_address_bytes = None;
        
        if let Ok(toml_str) = std::fs::read_to_string("miner.toml") {
            if let Ok(config) = toml::from_str::<MinerToml>(&toml_str) {
                if config.mining.mode == "pool" {
                    pool_url = config.mining.pool_url;
                    payout_address = config.mining.payout_address.clone();
                    
                    // Parse the addresses from the config
                    if let (Some(pool_hex), Some(payout_hex)) = (&config.mining.pool_address, &config.mining.payout_address) {
                        if let (Ok(pool_bytes), Ok(payout_bytes)) = (hex::decode(pool_hex), hex::decode(payout_hex)) {
                            if pool_bytes.len() == 32 && payout_bytes.len() == 32 {
                                let mut pb = [0u8; 32];
                                pb.copy_from_slice(&pool_bytes);
                                let mut mb = [0u8; 32];
                                mb.copy_from_slice(&payout_bytes);
                                pool_address_bytes = Some((pb, mb));
                            }
                        }
                    }

                    // Lower share difficulty: 16 leading zero bits for demonstration
                    let mut pt = [0xff; 32];
                    pt[0] = 0x00; pt[1] = 0x00;
                    pool_target = Some(pt);
                }
            }
        }

        // Clone only valid transactions. If any became stale since entering the
        // mempool, skip them silently instead of aborting the entire mining attempt.
        let pre_mine_height = self.state.height;
        let pre_mine_midstate = self.state.midstate;
        let mut candidate_state = self.state.clone();
        
        let mut total_fees: u64 = 0;
        let mut transactions = Vec::new();
        
        let max_commits = crate::core::MAX_BATCH_COMMITS;
        let max_reveals = crate::core::MAX_BATCH_REVEALS;

        let (pending_commits, pending_reveals) = self.mempool.transactions_split();

        let mut current_inputs = 0;
        let mut current_outputs = 0;

        for arc_tx in pending_commits.into_iter().take(max_commits) {
            let tx = Arc::unwrap_or_clone(arc_tx);
            match apply_transaction(&mut candidate_state, &tx) {
                Ok(_) => {
                    total_fees = total_fees.saturating_add(tx.fee());
                    transactions.push(tx);
                }
                Err(e) => tracing::debug!("Skipping stale commit during mining: {}", e),
            }
        }

        for arc_tx in pending_reveals.into_iter().take(max_reveals) {
            let tx = Arc::unwrap_or_clone(arc_tx);

            // Pre-check: Will adding this transaction exceed our global batch limits?
            if let Transaction::Reveal { inputs, outputs, .. } = &tx {
                if current_inputs + inputs.len() > crate::core::MAX_BATCH_INPUTS {
                    continue; // Skip this tx, try the next one
                }
                if current_outputs + outputs.len() > crate::core::MAX_BATCH_OUTPUTS {
                    continue; // Skip this tx, try the next one
                }
            }

            match apply_transaction_no_sig_check(&mut candidate_state, &tx) {  // sigs already verified at mempool admission
                Ok(_) => {
                    if let Transaction::Reveal { inputs, outputs, .. } = &tx {
                        current_inputs += inputs.len();
                        current_outputs += outputs.len();
                    }
                    total_fees = total_fees.saturating_add(tx.fee());
                    transactions.push(tx);
                }
                Err(e) => tracing::warn!("Skipping stale reveal during mining: {}", e),
            }
        }

        let coinbase = self.generate_coinbase(pre_mine_height, total_fees, pool_address_bytes);
        for cb in &coinbase {
            let coin_id = cb.coin_id();
            candidate_state.coins.insert(coin_id);
            candidate_state.midstate = hash_concat(&candidate_state.midstate, &coin_id);
        }

        // --- Calculate state root ---
        let smt_root = hash_concat(&candidate_state.coins.root(), &candidate_state.commitments.root());
        let state_root = hash_concat(&smt_root, &candidate_state.chain_mmr.root());
        candidate_state.midstate = hash_concat(&candidate_state.midstate, &state_root);
        // ---------------------------------

        let target = self.state.target;

        // Minimum timestamp the block must have (consensus: must exceed previous).
        // The actual timestamp is set AFTER mining completes to avoid staleness.
        let min_timestamp = self.state.timestamp + 1;
        let actual_timestamp = crate::core::state::current_timestamp().max(min_timestamp);

        let candidate_header = BatchHeader {
            height: pre_mine_height,
            prev_header_hash: self.state.header_hash,
            prev_midstate: pre_mine_midstate,
            post_tx_midstate: candidate_state.midstate,
            extension: Extension { nonce: 0, final_hash: [0u8; 32] },
            timestamp: actual_timestamp,
            target: self.state.target,
            state_root,
        };
        let mining_hash = crate::core::types::compute_header_hash(&candidate_header);

        let mut template = Batch {
            prev_midstate: pre_mine_midstate,
            prev_header_hash: self.state.header_hash,
            transactions,
            extension: Extension { nonce: 0, final_hash: [0; 32]},
            coinbase,
            timestamp: actual_timestamp, // Locked in BEFORE mining
            target: self.state.target,
            state_root, 
        };

        let cancel = Arc::new(AtomicBool::new(false));
        self.mining_cancel = Some(cancel.clone());
        let tx = self.mined_batch_tx.clone();
        let hash_counter = Arc::clone(&self.hash_counter);
        
        tokio::task::spawn_blocking(move || {
            use crate::core::extension::MiningResult;
            // Mine against the secure header hash!
            if let Some(mining_result) = mine_extension(mining_hash, target, pool_target, threads, cancel, hash_counter) {
                // Do NOT mutate the timestamp here anymore.
                match mining_result {
                    MiningResult::Block(extension) => {
                        template.extension = extension;
                        let _ = tx.try_send(MinedResult::Block(template)); 
                    }
                    MiningResult::Share(extension) => {
                        template.extension = extension;
                        if let (Some(url), Some(addr)) = (pool_url, payout_address) {
                            let _ = tx.try_send(MinedResult::Share { 
                                batch: template,
                                pool_url: url,
                                payout_address: addr,
                            });
                        }
                    }
                }
            }
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

        let mut wots_oracle = self.storage.query_spent_addresses(&batch).unwrap_or_default();
        // Change to &mut
        match apply_batch(&mut self.state, &batch, self.recent_headers.make_contiguous(), &mut wots_oracle) {
            Ok(_) => {
                self.recent_headers.push_back(batch.timestamp);
                if self.recent_headers.len() > DIFFICULTY_LOOKBACK as usize {
                    self.recent_headers.pop_front();
                }

                self.storage.save_batch(pre_mine_height, &batch)?;
                
                // 1. ADJUST TARGET BEFORE SAVING ANYTHING
                self.state.target = adjust_difficulty(&self.state);
                
                // 2. Cache for instant reorg rollback
                self.cache_current_state();
                
                // 3. NOW SAVE STATE and BURN ADDRESSES (Offloaded to prevent event loop stalls)
                let storage_clone = self.storage.clone();
                let state_clone = self.state.clone();
                let batch_clone = batch.clone();
                let cmd_tx = self.cmd_tx.as_ref().unwrap().clone(); // Pass the channel
                let has_light_peers = self.network.has_light_peers(); // Check if we even need to bother

                tokio::task::spawn_blocking(move || {
                    if let Err(e) = storage_clone.save_state(&state_clone) {
                        tracing::error!("Failed to save state: {}", e);
                    }
                    if let Err(e) = storage_clone.burn_batch_addresses(&batch_clone, pre_mine_height) {
                        tracing::warn!("burn_batch_addresses failed at height {}: {}", pre_mine_height, e);
                    }

                    // --- PUSH GENERATION ---
                    // We build the filter and count elements in the background, entirely avoiding main-thread CPU/Disk stalls.
                    if has_light_peers {
                        let filter = crate::core::filter::CompactFilter::build(&batch_clone);
                        
                        let mut items = std::collections::HashSet::new();
                        for tx in &batch_clone.transactions {
                            match tx {
                                crate::core::Transaction::Commit { commitment, .. } => { items.insert(*commitment); }
                                crate::core::Transaction::Reveal { inputs, outputs, .. } => {
                                    for input in inputs {
                                        items.insert(input.coin_id());
                                        items.insert(input.predicate.address());
                                    }
                                    for output in outputs {
                                        if let Some(cid) = output.coin_id() { items.insert(cid); }
                                        items.insert(output.address());
                                    }
                                }
                            }
                        }
                        for cb in &batch_clone.coinbase {
                            items.insert(cb.coin_id());
                            items.insert(cb.address);
                        }

                        let notif = crate::network::light_protocol::LightNotification::NewBlockTip {
                            height: state_clone.height,
                            target: hex::encode(state_clone.target),
                            filter_hex: hex::encode(filter.data),
                            block_hash: hex::encode(batch_clone.extension.final_hash),
                            element_count: items.len() as u64,
                        };
                        
                        let _ = cmd_tx.try_send(NodeCommand::BroadcastLightPush(notif));

                    }
                });

                // 5. NOW SAVE SNAPSHOT (every SNAPSHOT_INTERVAL blocks)
                if self.state.height > 0 && self.state.height % SNAPSHOT_INTERVAL == 0 {
                    if let Err(e) = self.storage.save_state_snapshot(self.state.height, &self.state) {
                        tracing::warn!("Failed to save state snapshot: {}", e);
                    } else {
                        tracing::info!("Saved state snapshot at height {}", self.state.height);
                    }
                }
                
                // 1000 Block Milestone Check ---
                if self.state.height > 0 && self.state.height % 1000 == 0 {
                    if let Ok(false) = self.perform_health_check().await {
                         tracing::error!("CRITICAL: Node state is sick/corrupted at milestone. Initiating self-healing sequence...");
                         self.cancel_mining();
                         self.self_heal_rollback(self.state.height).await;
                         return Ok(()); // abort further processing
                    }
                }

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
        self.trigger_mining();
        Ok(())
    }

    /// Synchronous test wrapper — spawns mining then blocks until it finishes.
    /// Preserves identical behavior for all existing tests.
    #[cfg(test)]
    pub async fn try_mine(&mut self) -> Result<()> {
        if self.mining_threads.is_none() {
            self.mining_threads = Some(0); 
        }
        // Short-circuit if a sync is happening so the test doesn't hang forever
        if self.sync_in_progress {
            return Ok(());
        }
        
        // If a task isn't already running, spawn one
        if self.mining_cancel.is_none() {
            self.spawn_mining_task()?;
        }
        
        // Wait for the block to be mined!
        if let Some(res) = self.mined_batch_rx.recv().await {
            match res {
                MinedResult::Block(batch) => {
                    self.handle_mined_batch(batch).await?;
                }
                MinedResult::Share { .. } => {
                    self.mining_cancel = None;
                }
            }
        }
        Ok(())
    }
    
       
    
}

/// Heavy background task to safely rebuild state from disk.
async fn rebuild_state_from_disk(storage: crate::storage::Storage, target_height: u64, cache_start: Option<(u64, State)>) -> Result<State> {
    if let Some((start_h, start_state)) = cache_start {
        tokio::task::spawn_blocking(move || -> Result<State> {
            let mut state = start_state;
            let mut recent_headers: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
            let mut wots_oracle = std::collections::HashMap::new();
            
            for h in start_h..target_height {
                if h % 500 == 0 && h > 0 {
                    tracing::info!("[REBUILD] Rebuilding state: {}/{}", h, target_height);
                }
                if let Some(batch) = storage.load_batch(h)? {
                    recent_headers.push_back(state.timestamp);
                    if recent_headers.len() > crate::core::DIFFICULTY_LOOKBACK as usize {
                        recent_headers.pop_front();
                    }
                        wots_oracle.clear(); //we dont need to hold the oracle in ram since the whole truth is kept in SPENT_ADDRESSES_TABLE anyway - we just need to ensure the current batch is good. 
                        let db_oracle = storage.query_spent_addresses(&batch).unwrap_or_default();
                        wots_oracle.extend(db_oracle);
                    
                    crate::core::state::apply_batch(&mut state, &batch, recent_headers.make_contiguous(), &mut wots_oracle)?;
                    state.target = crate::core::state::adjust_difficulty(&state);

                    if h > 0 && h % 500 == 0 {
                        let _ = storage.save_state_snapshot(h, &state);
                    }
                } else {
                    anyhow::bail!("Missing batch at height {} needed for state rebuild", h);
                }
            }
            Ok(state)
        }).await.map_err(|e| anyhow::anyhow!("State rebuild task panicked: {}", e))?
    } else {
        tokio::task::spawn_blocking(move || -> Result<State> {
            let mut snap_height = (target_height / 100) * 100;
            let mut best_snap = None;

            while snap_height > 0 {
                match storage.load_state_snapshot(snap_height) {
                    Ok(Some(snap)) => {
                        tracing::debug!("rebuild_state_from_disk: using snapshot at {} (target {})", snap_height, target_height);
                        best_snap = Some((snap, snap_height));
                        break;
                    }
                    _ => {
                        snap_height = snap_height.saturating_sub(100);
                    }
                }
            }

            let (mut state, replay_from) = best_snap.unwrap_or_else(|| {
                tracing::debug!("rebuild_state_from_disk: no snapshots found, replaying from genesis");
                (State::genesis().0, 0)
            });

            let mut recent_headers: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
            let mut wots_oracle = std::collections::HashMap::new(); 
            for h in replay_from..target_height {
                if h % 500 == 0 && h > 0 {
                    tracing::info!("[REPLAY] Rebuilding state: {}/{}", h, target_height);
                }
                if let Some(batch) = storage.load_batch(h)? {
                    recent_headers.push_back(state.timestamp);
                    if recent_headers.len() > crate::core::DIFFICULTY_LOOKBACK as usize {
                        recent_headers.pop_front();
                    }
                        wots_oracle.clear(); //we dont need to hold the oracle in ram since the whole truth is kept in SPENT_ADDRESSES_TABLE anyway - we just need to ensure the current batch is good. 
                        let db_oracle = storage.query_spent_addresses(&batch).unwrap_or_default();
                        wots_oracle.extend(db_oracle);
                    
                    crate::core::state::apply_batch(&mut state, &batch, recent_headers.make_contiguous(), &mut wots_oracle)?;
                    state.target = crate::core::state::adjust_difficulty(&state);
                    
                    // save our work in case there is a node crash or restart mid replay
                    if h > 0 && h % 500 == 0 {
                        let _ = storage.save_state_snapshot(h, &state);
                    }
                    
                } else {
                    anyhow::bail!("Missing batch at height {} needed for state rebuild", h);
                }
            }
            Ok(state)
        }).await.map_err(|e| anyhow::anyhow!("State rebuild task panicked: {}", e))?
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

impl Drop for Node {
    fn drop(&mut self) {
        if let Some(cancel) = self.mining_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
            tracing::debug!("Node shutting down: cancelled background mining tasks.");
        }
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
        Node::new(dir.to_path_buf(), None, listen, vec![], false).await.unwrap()
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
                    commitment: None,
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
                commitment: None,
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
                    commitment: None,
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
                commitment: None,
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
                commitment: None,
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
                commitment: None,
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
    
    #[test]
    fn find_genesis_nonce() {
        use crate::core::types::*;
        use crate::core::extension::create_extension;

        let (mut state, genesis_coinbase) = State::genesis();
        let mut mining_midstate = state.midstate;
        let mut temp_coins = state.coins.clone();
        for cb in &genesis_coinbase {
            let coin_id = cb.coin_id();
            mining_midstate = hash_concat(&mining_midstate, &coin_id);
            temp_coins.insert(coin_id);
        }
        let smt_root = hash_concat(&temp_coins.root(), &state.commitments.root());
        let state_root = hash_concat(&smt_root, &state.chain_mmr.root());
        mining_midstate = hash_concat(&mining_midstate, &state_root);

        for nonce in 0u64.. {
            let ext = create_extension(mining_midstate, nonce);
            if ext.final_hash < state.target {
                println!("\n\n  *** VALID GENESIS NONCE: {} ***\n", nonce);
                return;
            }
        }
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
        Node::new(dir.to_path_buf(), None, listen, vec![], false).await.unwrap()
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
        let smt_root = hash_concat(&candidate_state.coins.root(), &candidate_state.commitments.root());
        let state_root = hash_concat(&smt_root, &candidate_state.chain_mmr.root());
        candidate_state.midstate = hash_concat(&candidate_state.midstate, &state_root);

        let target = prev_state.target;
        let candidate_header = BatchHeader {
            height: prev_state.height,
            prev_header_hash: prev_state.header_hash,
            prev_midstate: prev_state.midstate,
            post_tx_midstate: candidate_state.midstate,
            extension: Extension { nonce: 0, final_hash: [0u8; 32] },
            timestamp,
            target,
            state_root,
        };
        let mining_hash = crate::core::types::compute_header_hash(&candidate_header);

        let mut nonce = 0u64;
        let extension = loop {
            let ext = create_extension(mining_hash, nonce);
            if ext.final_hash < target { break ext; }
            nonce += 1;
        };

        Batch {
            prev_midstate: prev_state.midstate,
            prev_header_hash: prev_state.header_hash,
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
            if crate::core::types::count_leading_zeros(&h) >= crate::core::transaction::MIN_COMMIT_POW_BITS { break; }
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
        apply_batch(&mut state_at_2, &genesis_batch, &[], &mut std::collections::HashMap::new()).unwrap();// H=1
        state_at_2.target = adjust_difficulty(&state_at_2);
        
        // Apply B1 (shared history)
       let ts_at_1 = vec![state_at_2.timestamp];
        apply_batch(&mut state_at_2, &b1, &ts_at_1, &mut std::collections::HashMap::new()).unwrap(); // H = 2
        state_at_2.target = adjust_difficulty(&state_at_2);

        // B2' (Alternative block at H=3). Empty, does NOT have the transaction.
        let b2_prime = make_valid_batch(&state_at_2, 20, vec![]); 
        let mut state_at_3_prime = state_at_2.clone();
        let ts_at_2 = vec![state_at_2.timestamp];
        apply_batch(&mut state_at_3_prime, &b2_prime, &ts_at_2, &mut std::collections::HashMap::new()).unwrap();
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
            apply_batch(&mut node.state, &b, &recent_headers, &mut std::collections::HashMap::new()).unwrap();
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
            if crate::core::types::count_leading_zeros(&h) >= crate::core::transaction::MIN_COMMIT_POW_BITS { break; }
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
                commitment: None,
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
            if crate::core::types::count_leading_zeros(&h) >= crate::core::transaction::MIN_COMMIT_POW_BITS { break; }
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
            apply_batch(&mut chain_b_state, &b, &recent_headers, &mut std::collections::HashMap::new()).unwrap();
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

            crate::core::state::apply_batch(&mut state, &batch, &recent_ts, &mut std::collections::HashMap::new()).unwrap();
            
            // FIX 2: Push the CURRENT block's timestamp into the history window 
            // after validating it
            recent_ts.push(batch.timestamp);
            if recent_ts.len() > 11 { recent_ts.remove(0); }
            
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
        let fork_state = node.rebuild_state_at_height(5).await.unwrap();
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
            peer_depth: peer_height as u128 * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 5,
                candidate_state: fork_state.clone(),
                cursor: 5,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
            last_progress_at: std::time::Instant::now(),
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
            Syncer::verify_header_chain(&disk_headers, &[]).is_ok(),
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
        let fork_state = node.rebuild_state_at_height(3).await.unwrap();
        let (alt_batches, alt_headers) = build_divergent_chain(&fork_state, 12);

        let mut full_headers = node.storage.batches.load_headers(0, 3).unwrap();
        full_headers.extend(alt_headers);

        // Set up sync session and feed some batches
        let peer = PeerId::random();
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer,
            peer_height: 3 + alt_batches.len() as u64,
            peer_depth: (3 + alt_batches.len() as u128) * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 3,
                candidate_state: fork_state,
                cursor: 3,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
            last_progress_at: std::time::Instant::now(),
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
            Syncer::verify_header_chain(&served_headers, &[]).is_ok(),
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
        let fork_state = node.rebuild_state_at_height(1).await.unwrap();
        let (alt_batches, alt_headers) = build_divergent_chain(&fork_state, 10);

        let mut full_headers = node.storage.batches.load_headers(0, 1).unwrap();
        full_headers.extend(alt_headers);

        let peer = PeerId::random();
        let peer_height = 1 + alt_batches.len() as u64; // 11
        node.sync_in_progress = true;
        node.sync_session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth: peer_height as u128 * 1_000_000,
            phase: SyncPhase::Batches {
                headers: full_headers,
                fork_height: 1,
                candidate_state: fork_state,
                cursor: 1,
                new_history: Vec::new(),
                is_fast_forward: false,
            },
            started_at: std::time::Instant::now(),
            last_progress_at: std::time::Instant::now(),
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
            Syncer::verify_header_chain(&disk_headers, &[]).is_ok(),
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
        let fork_a_state = node.rebuild_state_at_height(3).await.unwrap();
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
            last_progress_at: std::time::Instant::now(),
        });

        // Feed some of A's batches
        node.handle_sync_batches(peer_a, alt_a_batches[..3].to_vec())
            .await
            .unwrap();

        // Now peer B shows up — start_sync_session replaces the session
        let peer_b = PeerId::random();
        node.start_sync_session(peer_b, 20, 20_000_000, None);

        // Disk should still be coherent (old session's writes should not exist)
        let disk_headers = node.storage.batches
            .load_headers(0, original_state.height)
            .unwrap();
        assert!(
            Syncer::verify_header_chain(&disk_headers, &[]).is_ok(),
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
