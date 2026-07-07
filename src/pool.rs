//! # Provably Fair Stratum Pool
//!
//! This module implements a decentralized-auditable mining pool. Unlike traditional 
//! Stratum pools where miners must blindly trust the operator to report shares and 
//! distribute rewards fairly, this pool embeds an SPV-style **Merkle Precommitment** 
//! into every block template.
//!
//! Miners can query the HTTP Audit API (`/api/proof`) to receive a cryptographic 
//! proof that their exact accumulated score was included in the block's coinbase 
//! transaction *before* they begin hashing. If the pool operator lies or omits them, 
//! the miner's local client instantly detects the mismatch and disconnects.
//!
//! ## Security Mitigations Implemented
//! 1. **Replay Protection (`valid_shares`)**: Prevents "Infinite Money" glitches where 
//!    a miner resubmits the same valid nonce millions of times per second.
//! 2. **CPU Exhaustion Defense (`spawn_blocking`)**: Offloads the 1,000,000-iteration 
//!    BLAKE3 VDF from the async reactor, preventing remote DoS attacks.
//! 3. **Conditional Score Deduction**: Prevents "Orphan Theft" by waiting for the 
//!    network to explicitly `HTTP 200 OK` the block before wiping the miners' shares.
//! 4. **Tandem Port Binding**: Binds the Stratum TCP port and Audit HTTP port simultaneously
//!    to guarantee they never desync due to ghost processes holding TCP sockets open.
//! 5. **Checksum-Agnostic Ingestion**: Strips 4-byte UI checksums from user-supplied 
//!    addresses before hashing to prevent silent HTTP 400 rejection loops from the core node.

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{broadcast, RwLock};
use axum::{extract::{State, Query}, routing::get, Json, Router};

use crate::core::types::{hash, hash_concat, Batch};
use crate::core::extension::create_extension;

/// The database table storing the cumulative scores of all miners.
/// Key: 32-byte cryptographic address hash. Value: u64 share count.
const SHARES_TABLE: TableDefinition<&[u8; 32], u64> = TableDefinition::new("shares");
/// The database table storing historical blocks found and their exact payouts.
/// Key: Block timestamp (u64). Value: JSON string of payouts.
const BLOCKS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("blocks");
/// The committed per-miner score snapshot for each found block, keyed by the same
/// block timestamp as BLOCKS_TABLE. Stored separately so the frequently-polled
/// /pool/stats payload stays lean; served on demand via /api/block_scores for
/// historical split-verification (proving each payout was proportional to score).
const BLOCK_SCORES_TABLE: TableDefinition<u64, &str> = TableDefinition::new("block_scores");

// ── Stratum Protocol Types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StratumRequest {
    id: Option<u64>,
    method: String,
    params: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StratumResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

/// Represents an active mining job broadcast to all connected Stratum clients.
#[derive(Clone, Debug)]
struct Job {
    job_id: u64,
    /// The 32-byte target that the miner must grind nonces against.
    mining_hash: [u8; 32],
    /// The difficulty threshold to submit a pool share.
    share_target: [u8; 32],
    /// The difficulty threshold to find a full network block.
    network_target: [u8; 32],
    /// The full block template, used to reconstruct the block if a full hash is found.
    batch_template: serde_json::Value,
    /// The chain height this job is mining (the block found from it will have this height).
    height: u64,
    /// The committed (address, score) leaves backing this job's Merkle root. These are
    /// the exact scores the coinbase payout was computed from, captured per job so a
    /// found block can be split-verified against the precommitment it actually used.
    /// Arc so cloning the Job onto the broadcast channel stays cheap.
    committed_scores: Arc<Vec<([u8; 32], u64)>>,
}

// ── Merkle Tree Logic for Share Proofs ──────────────────────────────────────

/// A Merkle tree representing the current state of all miner shares in the pool.
/// 
/// # Reasoning
/// By constructing a Merkle tree of `H(Miner_Address || Score)`, we can compress 
/// the entire state of the pool into a single 32-byte root. This root is embedded 
/// into the salt of the pool's fee coin in the block template. This allows $O(\log N)$ 
/// inclusion proofs for miners to audit their shares.
#[derive(Clone)]
pub struct ShareMerkleTree {
    pub root: [u8; 32],
    pub leaves: Vec<([u8; 32], u64)>, 
    pub layers: Vec<Vec<[u8; 32]>>,
}

impl ShareMerkleTree {
    /// Builds a deterministic Merkle tree from a set of miner shares.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  shares is a valid sequence of (Address, Score) tuples.
    /// Post: The tree is deterministically sorted by Address to prevent malleability.
    ///       root! = The final 32-byte Merkle root.
    /// ```
    ///
    /// ```zed
    ///     BuildTree
    ///     ---------
    ///     shares? : seq (𝔹³² × ℕ₆₄)
    ///     root! : 𝔹³²
    ///
    ///     let sorted_shares = sort_by_address(shares?)
    ///     let L₀ = ⟨ ℋ(addr ⌢ le8(score)) | (addr, score) ∈ sorted_shares ⟩
    ///     let L_{i+1} = ⟨ ℋ(L_i[2k] ⌢ L_i[2k+1]) | k ∈ 0..|L_i|/2 ⟩
    ///     post root! = L_{max}[0]
    /// ```
    fn build(mut shares: Vec<([u8; 32], u64)>) -> Self {
        if shares.is_empty() {
            return Self { root: [0; 32], leaves: vec![], layers: vec![] };
        }
        
        shares.sort_by_key(|&(addr, _)| addr);
        
        let mut current_layer: Vec<[u8; 32]> = shares.iter().map(|(addr, score)| {
            let mut data = [0u8; 40];
            data[0..32].copy_from_slice(addr);
            data[32..40].copy_from_slice(&score.to_le_bytes());
            hash(&data)
        }).collect();

        let mut layers = vec![current_layer.clone()];

        while current_layer.len() > 1 {
            let mut next_layer = Vec::with_capacity((current_layer.len() + 1) / 2);
            for chunk in current_layer.chunks(2) {
                if chunk.len() == 2 {
                    next_layer.push(hash_concat(&chunk[0], &chunk[1]));
                } else {
                    next_layer.push(hash_concat(&chunk[0], &chunk[0]));
                }
            }
            layers.push(next_layer.clone());
            current_layer = next_layer;
        }

        Self { root: current_layer[0], leaves: shares, layers }
    }

    /// Generates an $O(\log N)$ Merkle inclusion proof for a specific miner address.
    /// Returns the leaf index (needed for left/right hashing reconstruction) and the proof array.
    fn generate_proof(&self, address: &[u8; 32]) -> Option<(usize, Vec<[u8; 32]>)> {
        let idx = self.leaves.iter().position(|&(a, _)| a == *address)?;
        let mut proof = Vec::new();
        let mut current_idx = idx;

        for layer in &self.layers[..self.layers.len() - 1] {
            let is_right = current_idx % 2 == 1;
            let sibling_idx = if is_right { current_idx - 1 } else { (current_idx + 1).min(layer.len() - 1) };
            proof.push(layer[sibling_idx]);
            current_idx /= 2;
        }
        Some((idx, proof))
    }
}

// ── App State ───────────────────────────────────────────────────────────────

/// Global state shared across the HTTP Audit API, the Core Polling Task, 
/// and the TCP Stratum Socket Handlers.
struct PoolState {
    db: Arc<Database>,
    current_job: RwLock<Option<Job>>,
    job_notifier: broadcast::Sender<Job>,
    /// The pool's raw 32-byte MSS public key hash.
    pool_address: String,
    share_target: [u8; 32],
    current_tree: RwLock<ShareMerkleTree>,
    /// The Share Replay Cache. Tracks successfully submitted nonces for the *current* job.
    /// Wiped clean every time a new block is detected to prevent OOM.
    valid_shares: RwLock<HashSet<u64>>, 
    /// Dynamic RPC URL of the core node, provided at startup.
    node_rpc_url: String,
    /// The percentage fee the pool takes from block rewards (e.g., 1.0 for 1%).
    pool_fee_percent: f64,
    /// The most recent network block reward seen from the node, relayed to the
    /// dashboard so it can estimate expected payouts / coins-per-day before the
    /// pool has found its first block. Lock-free; updated each polling cycle.
    current_block_reward: std::sync::atomic::AtomicU64,
    /// The most recent confirmed chain height seen from the node, relayed so the
    /// dashboard can show the live height and label blocks. Updated each poll.
    current_height: std::sync::atomic::AtomicU64,
    /// Cumulative per-miner share outcomes: address -> (accepted, rejected). Lifetime
    /// totals (never deducted on block-find, unlike score) so the dashboard can show a
    /// stable accept/reject efficiency per miner.
    share_stats: RwLock<HashMap<[u8; 32], (u64, u64)>>,
    /// Cumulative accepted shares per (address, worker-name). A pure stats layer for
    /// per-rig breakdown; payout accounting stays strictly per-address and is untouched.
    worker_stats: RwLock<HashMap<([u8; 32], String), u64>>,
    /// Set by the block-submission task when a submitted block is REJECTED by the
    /// node or the submission itself FAILS (connection refused / timeout / dropped).
    /// The network tip does not change in either case, so without this flag the
    /// template loop would never rebuild and miners would re-grind the same doomed
    /// template forever. Consumed (cleared) by the polling loop when it rebuilds.
    force_new_job: std::sync::atomic::AtomicBool,
}

#[derive(Deserialize)]
struct ProofQuery {
    address: String,
}

// ── HTTP API for Miner Audits ───────────────────────────────────────────────

/// Serves the SPV-style Merkle inclusion proof to miners via HTTP.
///
/// # Reasoning
/// Stratum clients independently poll this endpoint upon receiving a `mining.notify`
/// event. They use the returned index, score, and sibling hashes to reconstruct the 
/// Merkle root locally. If it doesn't match the `salt` of the fee coin in the template, 
/// the miner knows the pool is lying and disconnects.
///
/// # Security
/// Uses `parse_address_flexible` to allow miners to query using either raw 64-char hex
/// or the 72-char checksummed UI address.
async fn get_proof(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<ProofQuery>,
) -> Json<serde_json::Value> {
    let addr = match crate::core::types::parse_address_flexible(&query.address) {
        Ok(a) => a,
        Err(_) => return Json(serde_json::json!({ "error": "Invalid address" })),
    };
    
    let tree = state.current_tree.read().await;
    let score = tree.leaves.iter().find(|(a, _)| a == &addr).map(|(_, s)| *s).unwrap_or(0);
    
    if let Some((idx, proof)) = tree.generate_proof(&addr) {
        Json(serde_json::json!({
            "root": hex::encode(tree.root),
            "score": score,
            "index": idx, 
            "proof": proof.iter().map(hex::encode).collect::<Vec<_>>()
        }))
    } else {
        Json(serde_json::json!({ "error": "Miner not found in current block precommitment" }))
    }
}

#[derive(Deserialize)]
struct BlockScoresQuery {
    /// Block timestamp (the key the dashboard already has from recent_blocks[].timestamp).
    ts: u64,
}

/// Serves the committed per-miner score snapshot for one found block, so the dashboard
/// can prove every payout in that block was proportional to committed score (the
/// "prove the split" check). Held out of /pool/stats to keep that payload lean, and
/// fetched on demand only when a miner clicks to verify a specific block.
///
/// Returns `{ total_score, scores: [{address, score}] }` where `total_score` is the
/// committed total the coinbase payouts were actually proportioned against.
async fn get_block_scores(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<BlockScoresQuery>,
) -> Json<serde_json::Value> {
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(BLOCK_SCORES_TABLE) {
            if let Ok(Some(v)) = table.get(query.ts) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(v.value()) {
                    return Json(json);
                }
            }
        }
    }
    Json(serde_json::json!({ "error": "No committed score snapshot stored for that block" }))
}

/// Serves the Provably Fair Pool HTML dashboard.
///
/// # Reasoning
/// Providing a built-in dashboard allows pool operators to transparently display
/// current hash weights and historical payouts without needing external infrastructure.
/// By serving this alongside the stratum port offset, it does not conflict with the core node.
///
/// # Formal Specification
/// ```text
/// Pre:  true
/// Post: result is an HTML response containing the dashboard UI
/// ```
async fn pool_ui() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("pool.html"))
}

/// Serves the shared Midstate stylesheet.
///
/// # Reasoning
/// The dashboard links to `/midstate.css` rather than embedding its own palette so it
/// stays visually identical to the Explorer and Chat and inherits the global light/dark
/// theme automatically. The sheet is baked into the binary with `include_str!`, so there
/// is no runtime "file not found" failure mode — if it compiles, it serves.
///
/// Path note: `pool.rs` lives in `src/` while the shared web assets live in `src/rpc/`,
/// so this resolves to the single canonical `src/rpc/midstate.css` (the temporary copy
/// in `src/` can be deleted). `include_str!` is relative to THIS source file.
async fn pool_css() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("rpc/midstate.css"),
    )
}

/// Aggregates and returns current pool statistics as JSON.
///
/// # Reasoning
/// Iterates over the `shares` table to calculate current miner weights, and reads
/// the last 100 entries from the `blocks` table to show recent payouts and to give
/// the dashboard enough history to draw the rolling effort/earnings charts. This
/// provides total transparency to the miners so they can verify their payout equity
/// matches their hash contribution.
///
/// Note: the raw `network_target` and `share_target` are returned verbatim so the
/// browser can do all hashrate/effort math locally with native `BigInt`. The node's
/// own `/state` endpoint is intentionally NOT exposed to the public (it sits behind a
/// firewall/VPN), so the pool relays the only two values the UI needs to stay honest.
///
/// # Formal Specification
/// ```text
/// Pre:  true
/// Post: result contains (pool_fee_percent, total_score, active_miners, miners[], recent_blocks[])
/// ```
async fn get_pool_stats(State(state): State<Arc<PoolState>>) -> Json<serde_json::Value> {
    let mut miners = Vec::new();
    let mut total_score = 0u64;

    // Lifetime accept/reject tallies, snapshotted so we don't hold the lock across the
    // redb scan. Attached per-miner below for an efficiency readout.
    let share_snapshot = state.share_stats.read().await.clone();

    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(SHARES_TABLE) {
            for iter in table.iter().unwrap() {
                let (addr, score) = iter.unwrap();
                let mut a = [0u8; 32];
                a.copy_from_slice(addr.value());
                let s = score.value();
                if s > 0 {
                    let (accepted, rejected) = share_snapshot.get(&a).copied().unwrap_or((0, 0));
                    miners.push(serde_json::json!({
                        "address": crate::core::types::encode_address_with_checksum(&a),
                        "score": s,
                        "accepted": accepted,
                        "rejected": rejected
                    }));
                    total_score += s;
                }
            }
        }
    }

    miners.sort_by_key(|m| std::cmp::Reverse(m["score"].as_u64().unwrap_or(0)));

    // Per-(address, worker) accepted-share tallies for the rig breakdown.
    let mut workers = Vec::new();
    {
        let ws = state.worker_stats.read().await;
        for ((addr, name), count) in ws.iter() {
            workers.push(serde_json::json!({
                "address": crate::core::types::encode_address_with_checksum(addr),
                "worker": name,
                "score": count
            }));
        }
    }

    let mut blocks = Vec::new();
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(BLOCKS_TABLE) {
            for iter in table.iter().unwrap().rev().take(100) {
                let (_, data) = iter.unwrap();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data.value()) {
                    blocks.push(json);
                }
            }
        }
    }

    // --- RAW TARGETS + PRECOMMITMENT FOR CLIENT-SIDE BIGINT MATH ---
    // The browser does ALL hashrate/effort math locally with native BigInt against
    // these raw 32-byte targets, so the server never touches floating point for it.
    let current_job = state.current_job.read().await.clone();
    let share_target_hex = hex::encode(state.share_target);
    let network_target_hex = current_job
        .as_ref()
        .map(|job| hex::encode(job.network_target))
        .unwrap_or_default();
    // The current Merkle precommitment root (what miners audit against this block).
    let merkle_root_hex = hex::encode(state.current_tree.read().await.root);
    const BLOCK_TIME_SECS: u64 = 60;

    // --- OPTIONAL FLOAT FALLBACK (legacy clients / non-BigInt environments) ---
    let mut network_hashrate = 0.0;
    let mut hashes_per_share = 0.0;
    if let Some(job) = current_job {
        // Helper to convert U256 target to a float for math
        fn u256_to_f64(u: primitive_types::U256) -> f64 {
            u.0[0] as f64 +
            (u.0[1] as f64) * 2.0f64.powi(64) +
            (u.0[2] as f64) * 2.0f64.powi(128) +
            (u.0[3] as f64) * 2.0f64.powi(192)
        }

        let net_target = primitive_types::U256::from_big_endian(&job.network_target);
        let share_target = primitive_types::U256::from_big_endian(&state.share_target);
        
        let max_u256 = 2.0f64.powi(256);
        let net_diff_hashes = max_u256 / u256_to_f64(net_target).max(1.0);
        network_hashrate = net_diff_hashes / BLOCK_TIME_SECS as f64;
        
        hashes_per_share = max_u256 / u256_to_f64(share_target).max(1.0);
    }

    Json(serde_json::json!({
        "pool_fee_percent": state.pool_fee_percent,
        "total_score": total_score,
        "active_miners": miners.len(),
        "miners": miners,
        "recent_blocks": blocks,
        // Provably-fair anchor + raw targets for local BigInt math:
        "merkle_root": merkle_root_hex,
        "network_target": network_target_hex,
        "share_target": share_target_hex,
        "block_time_secs": BLOCK_TIME_SECS,
        // Pool fee address (so the UI can label + verify the fee coin in every block)
        // and the live network reward (for payout / coins-per-day estimates):
        "pool_address": state.pool_address,
        "block_reward": state.current_block_reward.load(std::sync::atomic::Ordering::Relaxed),
        // Live confirmed chain height (blocks the pool finds are stamped with their own
        // height), and the per-rig worker breakdown (stats only; payouts are per-address):
        "network_height": state.current_height.load(std::sync::atomic::Ordering::Relaxed),
        "workers": workers,
        // Float fallbacks (kept for backwards compatibility):
        "network_hashrate": network_hashrate,
        "hashes_per_share": hashes_per_share
    }))
}

// ── Main Server Boot ────────────────────────────────────────────────────────

/// Boots the Provably Fair Stratum Server and its companion HTTP Audit API.
///
/// # Architecture
/// This server is designed to run independently from the core Midstate node. 
/// In professional mining setups, the core node is heavily firewalled or hidden 
/// behind a VPN (e.g., Tailscale), while the Stratum server is exposed to the 
/// public internet.
///
/// # Arguments
/// * `pool_address` - The Midstate address where the pool fee will be sent.
/// * `bind_addr` - The `IP:PORT` to bind the Stratum TCP server to (e.g., `0.0.0.0:3333`).
///                 The HTTP Audit API will automatically bind to an offset port (e.g., `8081`).
/// * `node_rpc_url` - The HTTP URL of the backend Midstate node (e.g., `http://10.0.0.5:8545`).
/// * `pool_fee_percent` - The percentage of the block reward taken by the pool (e.g., 1.0).
pub async fn run_stratum_pool(pool_address: String, bind_addr: String, node_rpc_url: String, pool_fee_percent: f64) {
    tracing::info!("starting stratum pool server");
    
    std::fs::create_dir_all("data").unwrap();
    let db = Arc::new(Database::create("data/pool_stratum.redb").unwrap());
    
    let write_txn = db.begin_write().unwrap();
    let _ = write_txn.open_table(SHARES_TABLE).unwrap();
    let _ = write_txn.open_table(BLOCKS_TABLE).unwrap(); 
    write_txn.commit().unwrap();

    let (job_notifier, _) = broadcast::channel(32);
    
    let mut share_target = [0xff; 32];
    share_target[0] = 0x00; share_target[1] = 0x0f; 

    // Strip the UI checksum from the pool address so the backend node accepts it
    // during block template generation.
    let clean_pool_address_bytes = crate::core::types::parse_address_flexible(&pool_address)
        .expect("CRITICAL: Invalid Pool Address provided");
    let clean_pool_address = hex::encode(clean_pool_address_bytes);

    let state = Arc::new(PoolState {
        db,
        current_job: RwLock::new(None),
        job_notifier,
        pool_address: clean_pool_address,
        share_target,
        current_tree: RwLock::new(ShareMerkleTree::build(vec![])),
        valid_shares: RwLock::new(HashSet::new()),
        node_rpc_url,
        pool_fee_percent,
        current_block_reward: std::sync::atomic::AtomicU64::new(0),
        current_height: std::sync::atomic::AtomicU64::new(0),
        share_stats: RwLock::new(HashMap::new()),
        worker_stats: RwLock::new(HashMap::new()),
        force_new_job: std::sync::atomic::AtomicBool::new(false),
    });

    let api_state = state.clone();
    let parts: Vec<&str> = bind_addr.split(':').collect();
    let host = if parts.is_empty() { "0.0.0.0" } else { parts[0] };
    let base_stratum_port: u16 = if parts.len() > 1 { parts[1].parse().unwrap_or(3333) } else { 3333 };

    // ── Tandem Port Binding ──
    // Bind both the Stratum Port and the Audit API port simultaneously.
    // If either fails (e.g., stuck in TIME_WAIT), bump the offset and try the next pair. 
    // This guarantees the miner's offset math always aligns perfectly with the server.
    let mut offset = 0;
    let (api_listener, stratum_listener) = loop {
        let current_stratum = base_stratum_port + offset;
        let current_api = 8081 + offset;

        let a_res = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", current_api)).await;
        let s_res = tokio::net::TcpListener::bind(format!("{}:{}", host, current_stratum)).await;

        match (a_res, s_res) {
            (Ok(a), Ok(s)) => {
                tracing::info!("audit api bound to 0.0.0.0:{}", current_api);
                tracing::info!("stratum pool bound to {}:{}", host, current_stratum);
                break (a, s);
            }
            _ => {
                tracing::warn!("port pair {}/{} in use. trying next pair...", current_stratum, current_api);
                offset += 1;
                if offset > 10 {
                    panic!("fatal: could not find available stratum/api port pairs");
                }
            }
        }
    };

    tokio::spawn(async move {
        let app = Router::new()
            .route("/pool", get(pool_ui))            
            .route("/midstate.css", get(pool_css))   
            .route("/pool/stats", get(get_pool_stats)) 
            .route("/api/proof", get(get_proof))     
            .route("/api/block_scores", get(get_block_scores))
            .with_state(api_state);
        axum::serve(api_listener, app).await.unwrap();
    });
    
    // ── Core Polling & Template Builder Task ──
    let state_clone = state.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut last_network_tip = String::new();
        // Seed job IDs with the Unix timestamp so IDs issued before a pool restart
        // can never collide with IDs issued after it. (A zero-seeded counter resets
        // on restart; a still-connected miner's stale share can then carry a job_id
        // that matches the NEW counter and gets validated against the new midstate,
        // producing an endless "Low difficulty" reject stream for that miner.)
        // The loop below issues at most one job per second, so the counter can never
        // catch up to and overlap a future restart's seed.
        let mut job_counter: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        loop {
            let rpc_url = state_clone.node_rpc_url.clone();
            
            let net_state: serde_json::Value = match client.get(&format!("{}/state", rpc_url)).send().await {
                Ok(res) => res.json().await.unwrap_or_default(),
                Err(_) => { tokio::time::sleep(Duration::from_secs(2)).await; continue; }
            };

            // ── Sync Guard ──
            // While the backend node is bulk-downloading historical blocks, every tip
            // it reports is an already-superseded height: any template built from it
            // burns 100% of the pool's hashpower on obsolete blocks. Drop the current
            // job so miner submissions are ignored, and clear the tip tracker so a
            // fresh template is built on the first poll after the sync completes.
            // (`unwrap_or(false)` keeps this backward compatible with older nodes
            // whose /state payload doesn't include the field.)
            if net_state["is_syncing"].as_bool().unwrap_or(false) {
                if state_clone.current_job.write().await.take().is_some() {
                    tracing::warn!("backend node is syncing historical blocks; pausing job generation");
                }
                last_network_tip.clear();
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }

            let current_tip = net_state["header_hash"].as_str().unwrap_or("").to_string();
            let mut n_target = [0u8; 32];
            if let Some(t_hex) = net_state["target"].as_str() {
                let _ = hex::decode_to_slice(t_hex, &mut n_target);
            }

            // The node's reported height is the confirmed tip; the block a found job
            // produces is the next one (tip + 1). Refreshed every poll so the dashboard
            // can show the live chain height even between our own block finds.
            let tip_height = net_state["height"].as_u64().unwrap_or(0);
            state_clone.current_height.store(tip_height, std::sync::atomic::Ordering::Relaxed);

            // A rejected or failed block submission sets `force_new_job`: the network
            // tip won't change in that case, so a tip-only condition would re-serve
            // the doomed template forever. Consume the flag only when we have a
            // usable tip, so a request that races an empty /state response isn't
            // silently lost. (If template building fails below, `last_network_tip`
            // is cleared, which guarantees a retry on the next poll regardless.)
            let force_new_job = !current_tip.is_empty()
                && state_clone.force_new_job.swap(false, std::sync::atomic::Ordering::SeqCst);

            if (current_tip != last_network_tip || force_new_job) && !current_tip.is_empty() {
                job_counter += 1;
                last_network_tip = current_tip.clone();

                // Clear the replay cache for the new block
                state_clone.valid_shares.write().await.clear();

                let mut shares_vec = Vec::new();
                let mut total_score = 0u128;
                if let Ok(read_txn) = state_clone.db.begin_read() {
                    if let Ok(table) = read_txn.open_table(SHARES_TABLE) {
                        for iter in table.iter().unwrap() {
                            let (addr, score) = iter.unwrap();
                            let mut a = [0u8; 32];
                            a.copy_from_slice(addr.value());
                            shares_vec.push((a, score.value()));
                            total_score += score.value() as u128;
                        }
                    }
                }

                let tree = ShareMerkleTree::build(shares_vec.clone());
                *state_clone.current_tree.write().await = tree.clone();

                let mut expected_total = net_state["block_reward"].as_u64().unwrap_or(0);
                state_clone.current_block_reward.store(expected_total, std::sync::atomic::Ordering::Relaxed);
                
                // ── Proportional Reward Distribution Algorithm ──
                // Calculates the pool fee, then distributes the remaining reward 
                // across all miners strictly proportional to their accumulated scores.
                // Output values MUST be powers of 2. It iteratively assigns the largest 
                // possible power-of-2 denomination to the miner with the highest current score.
                let template_data = loop {
                    let mut coinbase_json = Vec::new();
                    
                    let pool_fee = (expected_total as f64 * (state_clone.pool_fee_percent / 100.0)) as u64;
                    let safe_pool_fee = pool_fee.max(1); 
                    let actual_distributable = expected_total.saturating_sub(safe_pool_fee);
                    
                    let fee_coins = crate::core::types::decompose_value(safe_pool_fee);
                    for (i, coin) in fee_coins.into_iter().enumerate() {
                        // Embed the Merkle Precommitment in the FIRST fee coin's salt.
                        let salt = if i == 0 { 
                            hex::encode(tree.root) 
                        } else { 
                            hex::encode(rand::random::<[u8; 32]>()) 
                        };
                        
                        coinbase_json.push(serde_json::json!({
                            "address": state_clone.pool_address,
                            "value": coin,
                            "salt": salt 
                        }));
                    }

                    if actual_distributable > 0 {
                        if total_score > 0 {
                            let mut scores: HashMap<_, i64> = shares_vec.clone().into_iter().map(|(k,v)| (k, v as i64)).collect();
                            for coin in crate::core::types::decompose_value(actual_distributable).into_iter().rev() {
                                let mut best_miner = [0u8; 32];
                                let mut max_score = i64::MIN;
                                for (addr, &score) in &scores {
                                    if score > max_score { max_score = score; best_miner = *addr; }
                                }
                                coinbase_json.push(serde_json::json!({
                                    "address": hex::encode(best_miner),
                                    "value": coin,
                                    "salt": hex::encode(rand::random::<[u8; 32]>())
                                }));
                                let simulated_drop = ((coin as u128 * total_score) / (actual_distributable as u128)) as i64;
                                *scores.get_mut(&best_miner).unwrap() -= simulated_drop.max(1);
                            }
                        } else {
                            for coin in crate::core::types::decompose_value(actual_distributable) {
                                coinbase_json.push(serde_json::json!({
                                    "address": state_clone.pool_address,
                                    "value": coin,
                                    "salt": hex::encode(rand::random::<[u8; 32]>())
                                }));
                            }
                        }
                    }

                    let req = serde_json::json!({ "coinbase": coinbase_json });
                    if let Ok(res) = client.post(&format!("{}/block_template", rpc_url)).json(&req).send().await {
                        if let Ok(json) = res.json::<serde_json::Value>().await {
                            if let Some(err) = json.get("error") {
                                let err_str = err.as_str().unwrap_or("");
                                // Re-sync mempool fees dynamically if the node rejects our block value
                                if err_str.contains("Expected: ") {
                                    if let Some(num_str) = err_str.split("Expected: ").nth(1) {
                                        if let Ok(new_expected) = num_str.parse::<u64>() {
                                            tracing::info!("Mempool fees detected. Adjusting block value to {}", new_expected);
                                            expected_total = new_expected;
                                            continue; 
                                        }
                                    }
                                }
                                tracing::error!("Node rejected block template request: {}", err_str);
                                break None;
                            }
                            break Some(json);
                        }
                    }
                    break None;
                };

                if let Some(template) = template_data {
                    if let Some(m_hex) = template["mining_midstate"].as_str() {
                        let mut m_hash = [0u8; 32];
                        hex::decode_to_slice(m_hex, &mut m_hash).unwrap();

                        let job = Job {
                            job_id: job_counter,
                            mining_hash: m_hash,
                            share_target: state_clone.share_target,
                            network_target: n_target,
                            batch_template: template["batch_template"].clone(),
                            height: tip_height.saturating_add(1),
                            committed_scores: std::sync::Arc::new(shares_vec.clone()),
                        };

                        *state_clone.current_job.write().await = Some(job.clone());
                        let _ = state_clone.job_notifier.send(job);
                        tracing::info!("new job {}: root {}", job_counter, hex::encode(&tree.root[..8]));
                    }
                } else {
                    last_network_tip.clear();
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    loop {
        let (socket, _) = stratum_listener.accept().await.unwrap();
        let state = state.clone();
        tokio::spawn(async move {
            let _ = handle_miner(socket, state).await;
        });
    }
}

// ── Stratum Connection Handler ──────────────────────────────────────────────

/// Handles an active TCP Stratum session with a miner.
///
/// # Security Mechanisms
/// 1. **Replay Protection**: Identical nonces for the same block are rejected instantly.
/// 2. **CPU Offloading**: PoW hashing runs in `spawn_blocking` to protect the reactor.
/// 3. **Orphan Theft Defense**: Miner scores are only wiped if the core node accepts the block.
async fn handle_miner(mut socket: TcpStream, state: Arc<PoolState>) -> anyhow::Result<()> {
    let (read_half, mut write_half) = socket.split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let mut job_rx = state.job_notifier.subscribe();
    let mut authorized_address = None;
    // Worker name from mining.authorize params[1] (stratum convention). Scopes this
    // connection's accepted shares to a rig for the per-worker breakdown; defaults
    // when a miner omits it (today's reference miner sends "worker1").
    let mut authorized_worker: String = "default".to_string();

    loop {
        tokio::select! {
            res = reader.read_line(&mut line) => {
                if res? == 0 { break; } 
                let req: StratumRequest = match serde_json::from_str(&line) {
                    Ok(r) => r,
                    Err(_) => { line.clear(); continue; }
                };
                line.clear();

                match req.method.as_str() {
                    "mining.subscribe" => {
                        let res = StratumResponse { id: req.id, result: Some(serde_json::json!(true)), error: None };
                        write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                    }
                    "mining.authorize" => {
                        let address = req.params[0].as_str().unwrap_or("").to_string();
                        // params[1] is the worker name by stratum convention (the reference
                        // miner sends "worker1"); record it for this connection's breakdown.
                        if let Some(w) = req.params.get(1).and_then(|v| v.as_str()) {
                            if !w.is_empty() { authorized_worker = w.to_string(); }
                        }
                        // Strip the UI checksum from the miner's address
                        if let Ok(addr_bytes) = crate::core::types::parse_address_flexible(&address) {
                            authorized_address = Some(addr_bytes);
                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(true)), error: None };
                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;

                            if let Some(job) = state.current_job.read().await.clone() {
                                let notif = StratumRequest {
                                    id: None,
                                    method: "mining.notify".into(),
                                    params: vec![
                                        serde_json::json!(job.job_id),
                                        serde_json::json!(hex::encode(job.mining_hash)),
                                        serde_json::json!(job.batch_template) 
                                    ]
                                };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&notif)?).as_bytes()).await?;
                            }
                        } else {
                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Invalid Address".into()) };
                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                        }
                    }
                    "mining.submit" => {
                        if let Some(miner_addr) = authorized_address {
                            let job_id = req.params[1].as_u64().unwrap();
                            let nonce = req.params[2].as_u64().unwrap();

                            if let Some(job) = state.current_job.read().await.clone() {
                                if job.job_id == job_id {
                                    {
                                        let mut cache = state.valid_shares.write().await;
                                        if !cache.insert(nonce) {
                                            drop(cache);
                                            state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).1 += 1;
                                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Duplicate share".into()) };
                                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                                            continue;
                                        }
                                    }

                                    let m_hash = job.mining_hash;
                                    let ext = tokio::task::spawn_blocking(move || {
                                        create_extension(m_hash, nonce)
                                    }).await.unwrap();

                                    if ext.final_hash < job.share_target {
                                        let write_txn = state.db.begin_write()?;
                                        {
                                            let mut table = write_txn.open_table(SHARES_TABLE)?;
                                            let current = table.get(&miner_addr)?.map(|v| v.value()).unwrap_or(0);
                                            table.insert(&miner_addr, current + 1)?;
                                        }
                                        write_txn.commit()?;

                                        // Tally the accepted share: per-miner (efficiency) and
                                        // per-(miner,worker) (rig breakdown). Stats only; payout
                                        // accounting in SHARES_TABLE above is untouched.
                                        state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).0 += 1;
                                        *state.worker_stats.write().await.entry((miner_addr, authorized_worker.clone())).or_insert(0) += 1;

                                        let res = StratumResponse { id: req.id, result: Some(serde_json::json!(true)), error: None };
                                        write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;

                                        if ext.final_hash < job.network_target {
                                            tracing::info!("block found by miner {}. submitting to network.", hex::encode(&miner_addr[..8]));
                                            // Capture block identity for the dashboard BEFORE `ext`/`job` are consumed:
                                            // the PoW hash (for the Explorer hyperlink) and the network target in
                                            // force for this block, so historical "luck" can be computed exactly
                                            // instead of approximated against the *current* difficulty.
                                            let block_hash_hex = hex::encode(ext.final_hash);
                                            let block_net_target_hex = hex::encode(job.network_target);
                                            // Block height and the committed score snapshot this job's payout was
                                            // built from — captured before `batch_template` is consumed below, and
                                            // moved into the spawn so the block can later be split-verified.
                                            let block_height = job.height;
                                            let committed_scores = job.committed_scores.clone();
                                            let mut batch: Batch = serde_json::from_value(job.batch_template).unwrap();
                                            batch.extension = ext;
                                            
                                            let total_reward: u64 = batch.coinbase.iter().map(|cb| cb.value).sum();
                                            let batch_for_node = batch.clone();
                                            let db_clone = state.db.clone();
                                            let rpc_url = state.node_rpc_url.clone();
                                            // Cloned into the submission task so it can flag the template
                                            // loop for a fresh job on rejection/failure (force_new_job).
                                            let submit_state = state.clone();
                                            
                                            tokio::spawn(async move {
                                                let res = reqwest::Client::new().post(&format!("{}/submit_batch", rpc_url))
                                                    .json(&batch_for_node).send().await;
                                                    
                                                match res {
                                                    Ok(resp) if resp.status().is_success() => {
                                                        tracing::info!("block accepted by network. applying score deductions.");
                                                        let write_txn = db_clone.begin_write().unwrap();
                                                        {
                                                            let mut table = write_txn.open_table(SHARES_TABLE).unwrap();
                                                            let mut total_score = 0u128;
                                                            for iter in table.iter().unwrap() { total_score += iter.unwrap().1.value() as u128; }
                                                            
                                                            let mut payouts = Vec::new(); 

                                                            for cb in &batch.coinbase {
                                                                let mut a = [0u8; 32]; a.copy_from_slice(&cb.address);
                                                                let deduction = ((cb.value as u128 * total_score) / (total_reward as u128)) as u64;
                                                                let current = table.get(&a).unwrap().map(|v| v.value()).unwrap_or(0);
                                                                table.insert(&a, current.saturating_sub(deduction)).unwrap();
                                                                
                                                                // <--- Record payout for dashboard
                                                                payouts.push(serde_json::json!({
                                                                    "address": crate::core::types::encode_address_with_checksum(&a),
                                                                    "value": cb.value
                                                                }));
                                                            }

                                                            // <--- Save block payout history to DB
                                                            let mut b_table = write_txn.open_table(BLOCKS_TABLE).unwrap();
                                                            let block_data = serde_json::json!({
                                                                "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                                                                "block_ts": batch.timestamp,
                                                                "hash": block_hash_hex,
                                                                "height": block_height,
                                                                "total_score": total_score as u64,
                                                                "net_target": block_net_target_hex,
                                                                "payouts": payouts
                                                            }).to_string();
                                                            b_table.insert(batch.timestamp, block_data.as_str()).unwrap();

                                                            // <--- Persist the committed score snapshot (split-verification).
                                                            // `committed_total` is the sum of the snapshot — the basis the coinbase
                                                            // payouts were proportioned against, not the accept-time table total
                                                            // used for the deductions above.
                                                            let committed_total: u128 = committed_scores.iter().map(|(_, s)| *s as u128).sum();
                                                            let scores_json: Vec<serde_json::Value> = committed_scores.iter().map(|(a, s)| serde_json::json!({
                                                                "address": crate::core::types::encode_address_with_checksum(a),
                                                                "score": s
                                                            })).collect();
                                                            let scores_data = serde_json::json!({
                                                                "total_score": committed_total as u64,
                                                                "scores": scores_json
                                                            }).to_string();
                                                            let mut s_table = write_txn.open_table(BLOCK_SCORES_TABLE).unwrap();
                                                            s_table.insert(batch.timestamp, scores_data.as_str()).unwrap();
                                                        }
                                                        write_txn.commit().unwrap();
                                                    }
                                                    Ok(resp) => {
                                                        // The node answered but refused the block (stale, invalid,
                                                        // etc). The tip won't change on a rejection, so force the
                                                        // template loop to rebuild instead of letting miners
                                                        // re-grind the doomed template indefinitely.
                                                        let status = resp.status();
                                                        let body = resp.text().await.unwrap_or_default();
                                                        tracing::warn!(
                                                            "block rejected by network ({}): {}. retaining miner scores; requesting fresh job.",
                                                            status, body.trim()
                                                        );
                                                        submit_state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
                                                    }
                                                    Err(e) => {
                                                        // The submission never reached the node (connection refused,
                                                        // timeout, dropped mid-flight). Previously this arm didn't
                                                        // exist and the error was swallowed silently: no log, no
                                                        // corrective action. Scores are retained (deductions only
                                                        // happen on confirmed acceptance) and a fresh job is forced.
                                                        tracing::error!(
                                                            "block submission to node failed: {}. retaining miner scores; requesting fresh job.",
                                                            e
                                                        );
                                                        submit_state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
                                                    }
                                                }
                                            });
                                        }
                                    } else {
                                        state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).1 += 1;
                                        let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Low difficulty".into()) };
                                        write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(job) = job_rx.recv() => {
                let notif = StratumRequest {
                    id: None,
                    method: "mining.notify".into(),
                    params: vec![
                        serde_json::json!(job.job_id),
                        serde_json::json!(hex::encode(job.mining_hash)),
                        serde_json::json!(job.batch_template)
                    ]
                };
                write_half.write_all(format!("{}\n", serde_json::to_string(&notif)?).as_bytes()).await?;
            }
        }
    }
    Ok(())
}
