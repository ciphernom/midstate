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
use tokio::sync::{broadcast, RwLock, Semaphore};
use axum::{extract::{State, Query}, http::Method, routing::{get, post}, Json, Router};
use tower_http::cors::{Any, CorsLayer};

use crate::core::types::{hash, hash_concat, Batch, Extension};
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

/// Tracks blocks waiting for maturity (e.g. 10 blocks) to detect network orphans.
/// Key: Block height (u64). Value: JSON string of `{ hash, block_ts, deductions: [[address_hex, deducted_score]] }`
const PENDING_BLOCKS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("pending_blocks");

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
    /// Bounds how many share verifications may run concurrently.
    ///
    /// One verification is a full `EXTENSION_ITERATIONS` (1,000,000) sequential
    /// BLAKE3 chain — roughly 100–200ms of a pinned core. `/api/submit` is a
    /// public, unauthenticated endpoint, so without a ceiling a trickle of
    /// requests can saturate every core on the box (a ~150ms-of-CPU-per-packet
    /// amplification factor). Permits are sized to the core count and held
    /// across the `spawn_blocking` call by BOTH protocols, so Stratum and HTTP
    /// draw from the same budget and neither can starve the other.
    verify_permits: Arc<Semaphore>,
    /// Per-address submit throttle for the public HTTP path: the instant of that
    /// address's last accepted verification. A browser miner produces a share
    /// every few minutes, so `HTTP_MIN_SUBMIT_INTERVAL` costs honest clients
    /// nothing while capping how fast one address can spend pool CPU. Stratum is
    /// exempt: it is rate-limited implicitly by holding an open TCP session.
    http_throttle: RwLock<HashMap<[u8; 32], std::time::Instant>>,
}

/// Minimum spacing between two verified share submissions from one address.
///
/// Lowered from 250ms: a browser rig runs one miner per core (10 here), and
/// although each finds a share only every ~95s, independent workers do collide.
/// At 250ms a real share was being discarded as "Submitting too fast" — throwing
/// away work a miner actually did. The CPU that this guard exists to protect is
/// already bounded by `verify_permits`, so this only needs to stop a pathological
/// flood, not police normal bursts.
const HTTP_MIN_SUBMIT_INTERVAL: Duration = Duration::from_millis(50);

/// Default share difficulty, in leading zero bits: one share per 2^12 nonces.
/// Matches the previous hardcoded `[0x00, 0x0f, 0xff, …]` target exactly, so
/// existing deployments see no change unless they pass `--share-bits`.
const DEFAULT_SHARE_BITS: u32 = 12;

/// Builds a 256-bit target with `bits` leading zero bits.
///
/// The share target is the pool's only difficulty knob, and it was previously a
/// pair of magic byte writes that silently encoded "12 leading zero bits".
/// Expressing it as bits makes the knob legible and lets an operator match it to
/// the hardware actually pointed at the pool without a recompile.
///
/// ```text
/// bits = 12  →  00 0f ff ff … ff   (1 share per 2^12 nonces)
/// bits = 16  →  00 00 ff ff … ff   (1 share per 2^16 nonces)
/// ```
fn target_from_leading_zero_bits(bits: u32) -> [u8; 32] {
    let bits = bits.min(255);
    let mut t = [0xffu8; 32];
    let full = (bits / 8) as usize;
    let rem = bits % 8;
    for b in t.iter_mut().take(full) {
        *b = 0x00;
    }
    if full < 32 && rem > 0 {
        t[full] = 0xffu8 >> rem;
    }
    t
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
/// * `share_bits` - Share difficulty in leading zero bits. `None` uses
///   [`DEFAULT_SHARE_BITS`]. Every share is worth exactly one point of score
///   regardless of this value, so changing it mid-round rescales what a point
///   costs: raise it and existing score becomes relatively more valuable, lower
///   it and new shares are cheaper than the ones already banked. Change it
///   between blocks, not during one, or set it once and leave it.
/// * `webrtc_port` - UDP port for the browser-facing webrtc-direct listener.
///   `None` disables it, leaving Stratum + HTTP only.
pub async fn run_stratum_pool(
    pool_address: String,
    bind_addr: String,
    node_rpc_url: String,
    pool_fee_percent: f64,
    share_bits: Option<u32>,
    webrtc_port: Option<u16>,
) {
    tracing::info!("starting stratum pool server");
    
    std::fs::create_dir_all("data").unwrap();
    let db = Arc::new(Database::create("data/pool_stratum.redb").unwrap());
    
    let write_txn = db.begin_write().unwrap();
    {
        let mut shares = write_txn.open_table(SHARES_TABLE).unwrap();
        let _ = write_txn.open_table(BLOCKS_TABLE).unwrap();
        let _ = write_txn.open_table(PENDING_BLOCKS_TABLE).unwrap();
        // ── One-off migration: purge stale zero-score rows ──
        // Databases written by affected versions accumulated permanent `score == 0`
        // rows (the deduction path used to write them back instead of removing them).
        // The template builder now filters them on load, but purging heals existing
        // DBs in place so the shares table matches what /pool/stats already shows
        // (`s > 0`), and keeps redb from carrying dead keys forever. Collect first,
        // then remove, to avoid mutating the table while its iterator is live.
        let stale: Vec<[u8; 32]> = shares
            .iter()
            .unwrap()
            .filter_map(|iter| {
                let (addr, score) = iter.unwrap();
                if score.value() == 0 {
                    let mut a = [0u8; 32];
                    a.copy_from_slice(addr.value());
                    Some(a)
                } else {
                    None
                }
            })
            .collect();
        let purged = stale.len();
        for a in stale {
            shares.remove(&a).unwrap();
        }
        if purged > 0 {
            tracing::info!("purged {} stale zero-score address(es) from the shares table", purged);
        }
    }
    write_txn.commit().unwrap();

    let (job_notifier, _) = broadcast::channel(32);
    
    let share_bits = share_bits.unwrap_or(DEFAULT_SHARE_BITS).clamp(1, 240);
    let share_target = target_from_leading_zero_bits(share_bits);
    tracing::info!(
        "share difficulty: {} leading zero bits (~1 share per {} nonces); target {}",
        share_bits,
        1u128.checked_shl(share_bits).map(|v| v.to_string()).unwrap_or_else(|| "2^…".into()),
        hex::encode(&share_target[..8])
    );

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
        // Leave one core for the async reactor, the redb writer and the template
        // poller, so a burst of share verifications can never freeze job delivery.
        verify_permits: Arc::new(Semaphore::new(
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(1).max(1)).unwrap_or(1),
        )),
        http_throttle: RwLock::new(HashMap::new()),
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
        // ── CORS ──
        // The web wallet is a static page served from a different origin than
        // this pool, so every /api/* call it makes is cross-origin. Without
        // these headers the browser blocks the response before the wallet's
        // JS ever sees it — the fetch fails with an opaque "TypeError: Failed
        // to fetch" that looks like the pool being down. `Any` origin is the
        // right call here specifically because these endpoints are public,
        // read-only-or-idempotent, and carry no cookies or ambient authority:
        // a share submission is authenticated by proof-of-work, not by the
        // caller's identity, so there is no cross-site request forgery surface
        // to protect. (Contrast the node's own RPC in `rpc/server.rs`, which
        // pins an explicit origin allow-list because it exposes wallet routes.)
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any);

        let app = Router::new()
            .route("/pool", get(pool_ui))            
            .route("/midstate.css", get(pool_css))   
            .route("/pool/stats", get(get_pool_stats)) 
            .route("/api/proof", get(get_proof))     
            .route("/api/block_scores", get(get_block_scores))
            // HTTP mining protocol (browser miners / web wallet):
            .route("/api/template", get(get_http_template))
            .route("/api/submit", post(post_http_submit))
            .layer(cors)
            .with_state(api_state);
        axum::serve(api_listener, app).await.unwrap();
    });
    
    // ── WebRTC listener (browser miners) ──
    // Started alongside the HTTP audit API rather than replacing it: HTTP stays
    // curl-able for monitoring and native tooling, while browsers get a path
    // that needs no CA certificate and receives pushed jobs instead of polling.
    // Failure here is non-fatal — Stratum and HTTP keep working.
    if let Some(udp_port) = webrtc_port {
        let webrtc_state = state.clone();
        tokio::spawn(async move {
            run_pool_webrtc(webrtc_state, udp_port, std::path::PathBuf::from("data")).await;
        });
    } else {
        tracing::info!("pool WebRTC listener disabled (pass --webrtc-port to enable)");
    }

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
        // One-shot latch for the "share difficulty >= network difficulty" warning.
        let mut warned_share_too_hard = false;
        // Latches so a persistently-down or syncing node logs once, not every 2s.
        let mut node_unreachable_logged = false;
        let mut syncing_logged = false;

        loop {
            let rpc_url = state_clone.node_rpc_url.clone();
            
            let net_state: serde_json::Value = match client.get(&format!("{}/state", rpc_url)).send().await {
                Ok(res) => {
                    if node_unreachable_logged {
                        tracing::info!("backend node reachable again at {}", rpc_url);
                        node_unreachable_logged = false;
                    }
                    match res.json().await {
                        Ok(v) => v,
                        Err(e) => {
                            // The node answered but not with JSON (wrong port, a
                            // proxy, an HTML error page). Previously this became
                            // Value::Null via unwrap_or_default() and the loop
                            // carried on with an empty tip, producing no job and
                            // no explanation.
                            tracing::error!("backend node at {}/state returned unparseable JSON: {}", rpc_url, e);
                            tokio::time::sleep(Duration::from_secs(2)).await;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // Latched so a down node logs once rather than every 2s.
                    if !node_unreachable_logged {
                        node_unreachable_logged = true;
                        tracing::error!(
                            "cannot reach backend node at {} ({}). No jobs will be served until it responds. \
                             Check the node is running and that --rpc-port/--rpc-host match it.",
                            rpc_url, e
                        );
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
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
                let had_job = state_clone.current_job.write().await.take().is_some();
                // Latch on the flag, not on had_job: at a cold start there is no
                // job to drop, so keying the message off `had_job` meant a pool
                // pointed at a syncing node logged absolutely nothing and simply
                // served "no active job" forever with no clue why.
                if had_job || !syncing_logged {
                    syncing_logged = true;
                    tracing::warn!(
                        "backend node is syncing historical blocks (height {}); pausing job generation until it catches up",
                        net_state["height"].as_u64().unwrap_or(0)
                    );
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

            // A share target at or below the network target is a misconfiguration:
            // every share would also be a block, so the pool degrades to solo
            // mining with extra steps and miners accumulate no score between finds.
            // Warned once rather than every poll so it doesn't drown the log.
            if !warned_share_too_hard && n_target != [0u8; 32] && state_clone.share_target <= n_target {
                warned_share_too_hard = true;
                tracing::warn!(
                    "share difficulty is at or above network difficulty (share target {} <= network target {}). \
                     Miners will earn score only when they find actual blocks — lower --share-bits.",
                    hex::encode(&state_clone.share_target[..8]),
                    hex::encode(&n_target[..8])
                );
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
                
                 // Only reconcile if the network tip actually advanced
                if current_tip != last_network_tip {
                    if let Err(e) = reconcile_pending_blocks(&state_clone, &client, tip_height).await {
                        tracing::warn!("Failed to reconcile pending blocks: {}", e);
                    }
                }
                
                job_counter += 1;
                last_network_tip = current_tip.clone();

                // Clear the replay cache for the new block
                state_clone.valid_shares.write().await.clear();

                // Drop throttle entries that are already expired. The map is keyed
                // by address, so an attacker rotating through fresh random addresses
                // would otherwise grow it without bound; pruning on every job keeps
                // it proportional to the miners actually submitting right now.
                {
                    let now = std::time::Instant::now();
                    state_clone
                        .http_throttle
                        .write()
                        .await
                        .retain(|_, last| now.duration_since(*last) < HTTP_MIN_SUBMIT_INTERVAL);
                }

                // Only positive-score rows are eligible for reward. A stale `score == 0`
                // row (e.g. the pool fee address, or a miner whose score was fully
                // consumed and never removed) contributes no work and MUST NOT enter the
                // allocator: the greedy loop below drives active miners' simulated scores
                // negative over a long coin decomposition, at which point a leaf sitting at
                // exactly 0 becomes the highest remaining score and captures the tail of the
                // distribution. Filtering here also keeps zero-score leaves out of the
                // Merkle tree and the committed-score snapshot, so the precommitment only
                // ever commits to addresses that actually worked the round.
                let mut shares_vec = Vec::new();
                let mut total_score = 0u128;
                if let Ok(read_txn) = state_clone.db.begin_read() {
                    if let Ok(table) = read_txn.open_table(SHARES_TABLE) {
                        for iter in table.iter().unwrap() {
                            let (addr, score) = iter.unwrap();
                            let s = score.value();
                            if s == 0 { continue; }
                            let mut a = [0u8; 32];
                            a.copy_from_slice(addr.value());
                            shares_vec.push((a, s));
                            total_score += s as u128;
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
                    let post_res = client.post(&format!("{}/block_template", rpc_url)).json(&req).send().await;
                    if let Err(ref e) = post_res {
                        tracing::error!("block_template request to {} failed: {}", rpc_url, e);
                    }
                    if let Ok(res) = post_res {
                        let parsed = res.json::<serde_json::Value>().await;
                        if let Err(ref e) = parsed {
                            tracing::error!("block_template response was not valid JSON: {}", e);
                        }
                        if let Ok(json) = parsed {
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

// ── HTTP Mining Protocol (browser / web-wallet miners) ──────────────────────
//
// Browsers cannot open raw TCP sockets, so the web wallet can never speak
// Stratum. These two endpoints expose the *same* job and the *same* share
// pipeline over plain request/response HTTP on the audit port:
//
//   GET  /api/template?address=<addr>   → the current job
//   POST /api/submit  {job_id, nonce, address, worker} → share result
//
// The protocol is deliberately poll-based rather than push-based. A browser
// miner runs at single-digit hashes per second (one nonce is a 1,000,000-round
// sequential BLAKE3 chain), so it produces a share every few minutes and a
// one-second template poll is already far finer-grained than it needs.
//
// # Why submissions carry only (job_id, nonce)
// The obvious shape — have the client POST the finished `batch` it just mined —
// is unsafe: the batch contains the coinbase, so a miner could rewrite the
// payout addresses to send the entire block reward to themselves and the pool
// would faithfully relay it to the node. A `job_id` plus a `nonce` is the
// complete proof-of-work claim; the pool rebuilds the block from the template
// it generated and still holds. There is nothing miner-supplied in the block.

/// Query for `GET /api/template`.
#[derive(Deserialize)]
struct HttpTemplateQuery {
    /// Raw 64-char hex or the 72-char checksummed UI address. Used only to
    /// report that miner's current score back to them; it is not registered or
    /// trusted at this point (score is credited on submit, never on request).
    address: String,
}

/// Serves the current job to an HTTP miner.
///
/// `target` is the pool's **share** target, not the network target: browser
/// miners grind toward the (much easier) share threshold and the pool decides
/// server-side whether a given share also happens to clear the network. Both
/// are returned so the client can display real odds.
async fn get_http_template(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<HttpTemplateQuery>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let addr = match crate::core::types::parse_address_flexible(&query.address) {
        Ok(a) => a,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Invalid address: {}", e) })),
            )
        }
    };

    match build_template_payload(&state, addr).await {
        Ok(v) => (axum::http::StatusCode::OK, Json(v)),
        // No job means the backend node is syncing or unreachable. 503 (not
        // 500) so the client treats it as "retry shortly" rather than fatal.
        Err(e) => (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": e })),
        ),
    }
}

/// Reads a miner's current accumulated score.
async fn miner_score(state: &Arc<PoolState>, addr: &[u8; 32]) -> u64 {
    state
        .db
        .begin_read()
        .ok()
        .and_then(|txn| txn.open_table(SHARES_TABLE).ok())
        .and_then(|t| t.get(addr).ok().flatten().map(|v| v.value()))
        .unwrap_or(0)
}

/// Builds the job payload served to a miner, independent of transport.
///
/// Shared verbatim by `/api/template` and the WebRTC `get_template` method, so
/// a browser dialing the pool over libp2p and one polling over HTTP audit the
/// same precommitment against the same fields. Divergence here would mean the
/// two transports could disagree about which root was committed.
async fn build_template_payload(
    state: &Arc<PoolState>,
    addr: [u8; 32],
) -> Result<serde_json::Value, String> {
    let job = match state.current_job.read().await.clone() {
        Some(j) => j,
        None => {
            return Err(
                "Pool has no active job (backend node syncing or unreachable). Retry shortly."
                    .to_string(),
            )
        }
    };

    let shares_recorded = miner_score(state, &addr).await;

    Ok(serde_json::json!({
            // Identifies which template a later submission belongs to. The client
            // must echo this back on submit; it is how a share mined against a
            // superseded template is detected and rejected as stale.
            "job_id":           job.job_id,
            "mining_midstate":  hex::encode(job.mining_hash),
            // The threshold the browser grinds against.
            "target":           hex::encode(job.share_target),
            // Informational: lets the UI show the real block odds behind a share.
            "network_target":   hex::encode(job.network_target),
            // Informational only — the pool rebuilds the block from its own copy.
            // Sent so the wallet can display the tx count for this template.
            "batch_template":   job.batch_template,
            "height":           job.height,
            "shares_recorded":  shares_recorded,
            // Precommitment anchor, so the client can cross-check /api/proof.
            "merkle_root":      hex::encode(state.current_tree.read().await.root),
            // The raw 32-byte pool fee address. The precommitment root is planted
            // in the salt of a coinbase output paying THIS address, so the client
            // needs it to locate the fee coin and confirm the root was actually
            // committed to the block rather than merely reported over the wire.
            "pool_address":     state.pool_address,
            "pool_fee_percent": state.pool_fee_percent,
            "block_reward":     state.current_block_reward.load(std::sync::atomic::Ordering::Relaxed),
    }))
}

/// Body for `POST /api/submit`.
#[derive(Deserialize)]
struct HttpSubmitBody {
    job_id: u64,
    /// Accepted as a JSON string or number. JavaScript's `Number` cannot hold a
    /// u64 without silent precision loss above 2^53, so the web wallet sends the
    /// nonce as a decimal string; `parse_nonce` handles both encodings.
    nonce: serde_json::Value,
    address: String,
    #[serde(default)]
    worker: Option<String>,
}

/// Validates a share and renders the transport-neutral result body.
///
/// Both `/api/submit` and the WebRTC `submit_share` method call this, so the two
/// transports cannot drift into crediting shares differently or reporting
/// different scores for the same work. HTTP additionally maps the outcome onto a
/// status code; WebRTC just returns the body.
async fn submit_share_payload(
    state: &Arc<PoolState>,
    address: &str,
    worker: Option<&str>,
    job_id: u64,
    nonce_raw: &serde_json::Value,
) -> (Option<ShareOutcome>, serde_json::Value) {
    let addr = match crate::core::types::parse_address_flexible(address) {
        Ok(a) => a,
        Err(e) => {
            return (
                None,
                serde_json::json!({ "accepted": false, "error": format!("Invalid address: {}", e) }),
            )
        }
    };

    let nonce = match parse_nonce(nonce_raw) {
        Some(n) => n,
        None => {
            return (
                None,
                serde_json::json!({ "accepted": false, "error": "Nonce must be a u64 (decimal string or number)" }),
            )
        }
    };

    // Per-address throttle, checked before the VDF. This is the only thing
    // standing between a publicly reachable endpoint and an unbounded CPU sink,
    // and it applies to WebRTC too: a libp2p stream is no harder to open in bulk
    // than an HTTP request.
    {
        let now = std::time::Instant::now();
        let mut throttle = state.http_throttle.write().await;
        if let Some(last) = throttle.get(&addr) {
            if now.duration_since(*last) < HTTP_MIN_SUBMIT_INTERVAL {
                return (
                    None,
                    serde_json::json!({ "accepted": false, "reason": "throttled", "error": "Submitting too fast" }),
                );
            }
        }
        throttle.insert(addr, now);
    }

    let worker = worker.filter(|w| !w.is_empty()).unwrap_or("browser");

    let outcome = match process_share(state, addr, worker, job_id, nonce).await {
        Ok(o) => o,
        Err(e) => {
            tracing::error!("share processing failed: {}", e);
            return (
                None,
                serde_json::json!({ "accepted": false, "error": "Internal error validating share" }),
            );
        }
    };

    // Report the miner's post-credit score so the client can show progress
    // without a second round trip.
    let score = miner_score(state, &addr).await;

    let body = match outcome {
        ShareOutcome::Accepted { is_block } => serde_json::json!({
            "accepted": true,
            "is_block": is_block,
            "score": score,
            "message": if is_block {
                "Share accepted — and it solved the block! Submitting to the network."
            } else {
                "Share accepted."
            }
        }),
        // Stale and duplicate are normal, expected races, not client errors, so
        // the client logs them quietly and keeps mining rather than alarming.
        ShareOutcome::StaleJob => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "stale", "message": "Stale share — the pool has moved to a new job."
        }),
        ShareOutcome::Duplicate => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "duplicate", "message": "Duplicate share."
        }),
        ShareOutcome::LowDifficulty => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "low_difficulty", "message": "Share did not meet the pool target."
        }),
        ShareOutcome::Busy => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "busy", "message": "Pool verifier saturated — retry this nonce shortly."
        }),
    };

    (Some(outcome), body)
}

/// Accepts a share from an HTTP miner and routes it through the shared pipeline.
async fn post_http_submit(
    State(state): State<Arc<PoolState>>,
    Json(body): Json<HttpSubmitBody>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    use axum::http::StatusCode;

    let (outcome, payload) = submit_share_payload(
        &state,
        &body.address,
        body.worker.as_deref(),
        body.job_id,
        &body.nonce,
    )
    .await;

    let status = match outcome {
        Some(ShareOutcome::Busy) => StatusCode::SERVICE_UNAVAILABLE,
        Some(_) => StatusCode::OK,
        // No outcome means the request never reached the verifier: bad address,
        // bad nonce, or throttled. Distinguish throttling so a client can back
        // off rather than treating it as a permanent malformed-request error.
        None => {
            if payload.get("reason").and_then(|r| r.as_str()) == Some("throttled") {
                StatusCode::TOO_MANY_REQUESTS
            } else if payload.get("error").is_some()
                && payload["error"].as_str().map(|e| e.starts_with("Internal")).unwrap_or(false)
            {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::BAD_REQUEST
            }
        }
    };

    (status, Json(payload))
}

// ── WebRTC Transport (browser miners, no TLS / no CORS / push-based) ─────────
//
// A second front door onto exactly the same job and share pipeline as Stratum
// and HTTP. Browsers cannot open raw TCP, and reaching the HTTP port from an
// HTTPS-served wallet requires a CA-signed certificate on the pool. libp2p's
// webrtc-direct sidesteps both: the connection is authenticated by a certificate
// hash embedded in the multiaddr, so there is no domain, no CA, and no
// mixed-content block — the same trick `light_client.js` already uses to reach
// full nodes.
//
// The decisive advantage over HTTP is not the transport, it is that streams are
// bidirectional and long-lived. The HTTP path has browsers polling once a second
// and receiving the entire `batch_template` — every transaction in the block —
// even when nothing has changed. Here the pool subscribes each connected miner
// to the *same* `job_notifier` broadcast the Stratum handler uses and pushes
// `notify` the instant a job rotates, which also closes the stale-work window
// (with 60s blocks and 1s polling, miners burn up to ~1.6% of their work on
// templates that are already dead).
//
// Framing and protocol shape deliberately mirror `light_protocol.rs`: 4-byte
// little-endian length prefix followed by JSON, one request/response per stream,
// pushes as one-shot server-initiated streams. The wallet's existing
// `/midstate/light-push/2.0.0` reader works the same way, so the client side is
// a near-copy of code that already works.

/// Request/response protocol: miner opens a stream, sends one request, reads one
/// response, stream closes.
pub const POOL_PROTOCOL: libp2p::StreamProtocol =
    libp2p::StreamProtocol::new("/midstate/pool/1.0.0");
/// Push protocol: the pool opens a stream to a miner and sends one notification.
pub const POOL_PUSH_PROTOCOL: libp2p::StreamProtocol =
    libp2p::StreamProtocol::new("/midstate/pool-push/1.0.0");

/// Cap on a single inbound pool request. Requests are tiny (an address, a job id
/// and a nonce); anything larger is malformed or hostile.
const MAX_POOL_MSG_SIZE: u32 = 64 * 1024;

/// Reads one 4-byte-LE-length-prefixed JSON payload from a libp2p stream.
async fn read_framed<T: futures::AsyncRead + Unpin>(io: &mut T) -> anyhow::Result<Vec<u8>> {
    use futures::AsyncReadExt;
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_POOL_MSG_SIZE {
        anyhow::bail!("pool message too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len as usize];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Writes one 4-byte-LE-length-prefixed JSON payload to a libp2p stream.
async fn write_framed<T: futures::AsyncWrite + Unpin>(
    io: &mut T,
    value: &serde_json::Value,
) -> anyhow::Result<()> {
    use futures::AsyncWriteExt;
    let bytes = serde_json::to_vec(value)?;
    io.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
    io.write_all(&bytes).await?;
    io.flush().await?;
    Ok(())
}

/// Loads the pool's libp2p identity from disk, generating it on first run.
///
/// # Why this must persist
/// The PeerId derived from this key is the `/p2p/<id>` component of the
/// multiaddr miners save in their wallet. Generating a fresh key each boot would
/// silently invalidate every saved pool address on every restart.
fn load_or_create_identity(path: &std::path::Path) -> anyhow::Result<libp2p::identity::Keypair> {
    if path.exists() {
        let bytes = std::fs::read(path)?;
        if let Ok(kp) = libp2p::identity::Keypair::from_protobuf_encoding(&bytes) {
            return Ok(kp);
        }
        tracing::warn!("pool identity at {:?} is unreadable; regenerating", path);
    }
    let kp = libp2p::identity::Keypair::generate_ed25519();
    write_secret(path, &kp.to_protobuf_encoding()?)?;
    tracing::info!("generated new pool libp2p identity at {:?}", path);
    Ok(kp)
}

/// Writes private key material with owner-only permissions.
///
/// Both files this guards hold secrets: the libp2p identity is the pool's
/// signing key, and the WebRTC certificate embeds its private key. A plain
/// `fs::write` creates them under the process umask — commonly world-readable
/// 0644 — which on a shared or containerised host leaks the pool's identity to
/// any local user. Permissions are set before the bytes land, so there is no
/// window during which the file exists with looser modes.
fn write_secret(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        return Ok(());
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// Loads the pool's WebRTC certificate from disk, generating it on first run.
///
/// # Why this must persist
/// webrtc-direct authenticates the server by a hash of this certificate, carried
/// in the multiaddr as `/certhash/<...>`. `network/mod.rs` calls
/// `Certificate::generate` on every node startup, which is fine for a node
/// (Kademlia and bootstrap discovery re-find it) but would be a serious bug for
/// a pool: miners persist the pool multiaddr in `localStorage`, so a rotating
/// certhash means every restart silently breaks every saved address with a
/// connection failure that looks like the pool being offline.
fn load_or_create_certificate(
    path: &std::path::Path,
) -> anyhow::Result<libp2p_webrtc::tokio::Certificate> {
    if path.exists() {
        let pem = std::fs::read_to_string(path)?;
        match libp2p_webrtc::tokio::Certificate::from_pem(&pem) {
            Ok(cert) => return Ok(cert),
            Err(e) => tracing::warn!("pool WebRTC certificate at {:?} is invalid ({}); regenerating — SAVED MINER ADDRESSES WILL BREAK", path, e),
        }
    }
    // NOTE: `Certificate::generate` takes its RNG parameter as `_rng` and never
    // reads it — the key comes from the crate's own entropy. So seeding it to
    // derive a reproducible certificate from a stored seed is not possible; the
    // certificate itself must be saved and reloaded, which is why the `pem`
    // feature on libp2p-webrtc is required rather than optional here.
    let cert = libp2p_webrtc::tokio::Certificate::generate(&mut rand::rngs::OsRng)
        .map_err(|e| anyhow::anyhow!("WebRTC certificate generation failed: {}", e))?;
    write_secret(path, cert.serialize_pem().as_bytes())?;
    tracing::info!("generated new pool WebRTC certificate at {:?}", path);
    Ok(cert)
}

/// Closes the write side, then waits for the peer to close theirs before the
/// stream is dropped.
///
/// # Why this is not just `close()`
/// `close()` only *half*-closes: it signals EOF on our write side while our read
/// side stays open. Dropping the stream immediately afterwards — which is what
/// happens when the handler task returns — sends a RESET to the remote. The
/// browser surfaces that as "The stream has been reset" and discards the
/// response, even though the bytes were already on the wire. Draining to EOF
/// first lets the remote observe our FIN and close cleanly.
///
/// Bounded: a client that never closes its side must not pin a task forever.
async fn graceful_close(mut stream: libp2p::Stream) {
    use futures::{AsyncReadExt, AsyncWriteExt};
    if stream.close().await.is_err() {
        return;
    }
    let _ = tokio::time::timeout(Duration::from_secs(5), async move {
        let mut sink = [0u8; 64];
        loop {
            match stream.read(&mut sink).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    })
    .await;
}

/// Handles one inbound pool stream: read a request, answer it, close.
async fn handle_pool_stream(
    state: Arc<PoolState>,
    mut stream: libp2p::Stream,
) -> anyhow::Result<()> {
    let raw = read_framed(&mut stream).await?;
    let req: serde_json::Value = serde_json::from_slice(&raw)?;
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(serde_json::Value::Null);

    let response = match method {
        "get_template" => {
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            match crate::core::types::parse_address_flexible(address) {
                Ok(addr) => match build_template_payload(&state, addr).await {
                    Ok(v) => serde_json::json!({ "ok": true, "data": v }),
                    Err(e) => serde_json::json!({ "ok": false, "error": e }),
                },
                Err(e) => serde_json::json!({ "ok": false, "error": format!("Invalid address: {}", e) }),
            }
        }
        "submit_share" => {
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            let worker = params.get("worker").and_then(|w| w.as_str());
            let job_id = params.get("job_id").and_then(|j| j.as_u64()).unwrap_or(0);
            let nonce = params.get("nonce").cloned().unwrap_or(serde_json::Value::Null);
            let (_, body) = submit_share_payload(&state, address, worker, job_id, &nonce).await;
            serde_json::json!({ "ok": true, "data": body })
        }
        "get_proof" => {
            // Mirrors GET /api/proof. Serving the audit over the same transport
            // matters: a miner that fetches its job over WebRTC but its proof
            // over HTTPS gains nothing, because the HTTPS hop is exactly the one
            // that needs a certificate the pool may not have.
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            match crate::core::types::parse_address_flexible(address) {
                Ok(addr) => {
                    let tree = state.current_tree.read().await;
                    let score = tree.leaves.iter().find(|(a, _)| a == &addr).map(|(_, s)| *s).unwrap_or(0);
                    match tree.generate_proof(&addr) {
                        Some((idx, proof)) => serde_json::json!({ "ok": true, "data": {
                            "root": hex::encode(tree.root),
                            "score": score,
                            "index": idx,
                            "proof": proof.iter().map(hex::encode).collect::<Vec<_>>()
                        }}),
                        None => serde_json::json!({ "ok": false, "error": "Miner not found in current block precommitment" }),
                    }
                }
                Err(e) => serde_json::json!({ "ok": false, "error": format!("Invalid address: {}", e) }),
            }
        }
        other => serde_json::json!({ "ok": false, "error": format!("Unknown method: {}", other) }),
    };

    write_framed(&mut stream, &response).await?;
    graceful_close(stream).await;
    Ok(())
}

/// Builds the pool's minimal libp2p swarm: webrtc-direct transport plus the
/// stream behaviour, and nothing else — no Kademlia, identify or relay, since
/// miners dial the pool by an address they already hold.
///
/// # Why this is a separate function
/// The `SwarmBuilder` phases return *different* error types:
/// `with_other_transport` yields `Result<_, TransportBuilderError>` while
/// `with_behaviour` yields `Result<_, Infallible>`. `and_then` requires both
/// arms to share one error type, so chaining them that way cannot typecheck no
/// matter how the closures are written. Giving the sequence its own function
/// lets `?` convert each error into one boxed type in turn.
fn build_pool_swarm(
    keypair: libp2p::identity::Keypair,
    certificate: libp2p_webrtc::tokio::Certificate,
) -> Result<libp2p::Swarm<libp2p_stream::Behaviour>, Box<dyn std::error::Error + Send + Sync>> {
    use libp2p_core::{muxing::StreamMuxerBox, Transport};

    Ok(libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_other_transport(move |kp| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                libp2p_webrtc::tokio::Transport::new(kp.clone(), certificate)
                    .map(|(peer_id, conn), _| (peer_id, StreamMuxerBox::new(conn))),
            )
        })?
        .with_behaviour(|_| libp2p_stream::Behaviour::new())?
        .with_swarm_config(|c| {
            // Miners hold a connection open for the whole session and may sit
            // idle between shares for minutes at browser hashrates, so the idle
            // timeout has to be generous or the pool disconnects the very miners
            // it is trying to serve.
            c.with_idle_connection_timeout(Duration::from_secs(600))
        })
        .build())
}

/// Boots the pool's WebRTC listener and its job-push loop.
///
/// Returns immediately; the swarm runs on a spawned task. The dialable multiaddr
/// is logged once the listener binds — that string is what miners paste into the
/// wallet, and it is only fully known after libp2p reports the listen address
/// (it carries the certhash).
async fn run_pool_webrtc(state: Arc<PoolState>, udp_port: u16, data_dir: std::path::PathBuf) {
    use futures::StreamExt;

    let keypair = match load_or_create_identity(&data_dir.join("pool_identity.key")) {
        Ok(k) => k,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not load identity: {}", e);
            return;
        }
    };
    let certificate = match load_or_create_certificate(&data_dir.join("pool_webrtc_cert.pem")) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not load certificate: {}", e);
            return;
        }
    };

    let local_peer = keypair.public().to_peer_id();

    let mut swarm = match build_pool_swarm(keypair, certificate) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: swarm construction failed: {}", e);
            return;
        }
    };

    let mut control = swarm.behaviour_mut().new_control();
    let mut incoming = match control.accept(POOL_PROTOCOL) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not register protocol: {}", e);
            return;
        }
    };

    let listen: libp2p::Multiaddr = match format!("/ip4/0.0.0.0/udp/{}/webrtc-direct", udp_port).parse() {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: bad listen address: {}", e);
            return;
        }
    };
    if let Err(e) = swarm.listen_on(listen) {
        tracing::error!("pool WebRTC disabled: listen failed on udp/{}: {}", udp_port, e);
        return;
    }

    // Miners currently connected, so job pushes go only to live sessions.
    let peers: Arc<RwLock<HashSet<libp2p::PeerId>>> = Arc::new(RwLock::new(HashSet::new()));

    // ── Job push loop ──
    // Subscribes to the SAME broadcast the Stratum handler uses, so a WebRTC
    // miner and a Stratum miner are notified of a new job by the same event and
    // can never be served different templates for the same height.
    {
        let peers = peers.clone();
        let mut job_rx = state.job_notifier.subscribe();
        let push_control = control.clone();
        tokio::spawn(async move {
            loop {
                let job = match job_rx.recv().await {
                    Ok(j) => j,
                    // Lagged means this task fell behind the broadcast ring. The
                    // next job supersedes any it missed, so resubscribing and
                    // waiting is correct — a miner briefly holds a stale template
                    // and the job_id check rejects its shares as stale, which is
                    // exactly the designed behaviour.
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("pool webrtc push lagged {} job(s)", n);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                let notification = serde_json::json!({
                    "method": "notify",
                    "params": {
                        "job_id":          job.job_id,
                        "mining_midstate": hex::encode(job.mining_hash),
                        "target":          hex::encode(job.share_target),
                        "network_target":  hex::encode(job.network_target),
                        "height":          job.height,
                    }
                });

                let targets: Vec<_> = peers.read().await.iter().copied().collect();
                for peer in targets {
                    let mut ctrl = push_control.clone();
                    let payload = notification.clone();
                    let peers = peers.clone();
                    tokio::spawn(async move {
                        // Bounded, like the node's light-push: a browser that
                        // stops reading must not pin a task forever.
                        let ok = tokio::time::timeout(Duration::from_secs(5), async move {
                            let mut stream = ctrl.open_stream(peer, POOL_PUSH_PROTOCOL).await.ok()?;
                            write_framed(&mut stream, &payload).await.ok()?;
                            // Same reset hazard as the request path: the browser
                            // reads until it has the whole framed payload, so
                            // dropping early loses the notification outright.
                            graceful_close(stream).await;
                            Some(())
                        })
                        .await;
                        if !matches!(ok, Ok(Some(()))) {
                            // A failed push means the miner is gone or wedged.
                            // Drop it so the peer set tracks reality.
                            peers.write().await.remove(&peer);
                        }
                    });
                }
            }
        });
    }

    // ── Inbound stream loop ──
    {
        let state = state.clone();
        tokio::spawn(async move {
            while let Some((peer, stream)) = incoming.next().await {
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_pool_stream(state, stream).await {
                        tracing::debug!("pool webrtc stream from {} ended: {}", peer, e);
                    }
                });
            }
        });
    }

    // ── Swarm event loop ──
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                    // This is the string miners paste into the wallet. It only
                    // becomes known here because libp2p appends the certhash.
                    let dialable = address
                        .clone()
                        .with(libp2p::multiaddr::Protocol::P2p(local_peer));
                    tracing::info!("pool WebRTC listening — miners connect to: {}", dialable);
                }
                libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    peers.write().await.insert(peer_id);
                    tracing::debug!("pool webrtc miner connected: {}", peer_id);
                }
                libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id, num_established, .. } => {
                    if num_established == 0 {
                        peers.write().await.remove(&peer_id);
                        tracing::debug!("pool webrtc miner disconnected: {}", peer_id);
                    }
                }
                _ => {}
            }
        }
    });
}

/// Parses a nonce supplied as either a JSON number or a decimal string.
fn parse_nonce(v: &serde_json::Value) -> Option<u64> {
    match v {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

// ── Shared Share Pipeline (Stratum + HTTP) ──────────────────────────────────

/// The result of validating one submitted share.
#[derive(Debug, Clone, Copy)]
enum ShareOutcome {
    /// Met the pool's share target and was credited. `is_block` is true when the
    /// same hash also cleared the full network target, in which case the block
    /// has been handed to the background submission task.
    Accepted { is_block: bool },
    /// This nonce was already credited for this job.
    Duplicate,
    /// Genuine work, but above the share target.
    LowDifficulty,
    /// The `job_id` doesn't match the job currently being served.
    StaleJob,
    /// The verifier is saturated; the miner should retry the same nonce.
    Busy,
}

/// Validates and credits one share. This is the single entry point for **both**
/// the Stratum socket handler and the HTTP `/api/submit` endpoint.
///
/// # Reasoning
/// Replay protection, scoring, the network-target check and the block-submission
/// handoff are all consensus- and payout-critical. Implementing them twice — once
/// per transport — is how the two paths silently drift apart (one gets a bug fix,
/// the other doesn't, and the pool starts paying two different rates depending on
/// which port a miner connected to). Everything transport-specific stays in the
/// callers; everything that touches score lives here.
///
/// # Ordering note (replay cache vs. proof-of-work)
/// The nonce is inserted into `valid_shares` only **after** the work is proven.
/// Claiming it up front — as the original inline Stratum path did — lets anyone
/// with a socket burn the nonce space: submitting garbage nonces poisons the
/// cache so that an honest miner who later finds a real share on one of those
/// nonces is rejected as a "duplicate", and the set grows without bound on
/// unvalidated input until the next block clears it. Probing for membership
/// first (cheap) and inserting after (proven) keeps the fast path fast while
/// making the cache hold only real work.
///
/// # Formal Specification
///
/// ```text
/// Let  H(m, n)   = create_extension(m, n).final_hash        (the 1e6-round VDF)
///      score     : Address ⇸ ℕ                              (SHARES_TABLE)
///      seen      : ℙ ℕ                                      (valid_shares)
///      job       = current_job
///
/// Pre:
///   - true  (total: every rejection path is a normal return, never an error)
///
/// Post — exactly one outcome, and score changes on Accepted alone:
///
///   outcome! = Accepted{is_block} ⇔
///        job ≠ ⊥ ∧ job.job_id = job_id?
///      ∧ nonce? ∉ seen
///      ∧ H(job.mining_hash, nonce?) < job.share_target
///      ∧ seen'   = seen ∪ {nonce?}
///      ∧ score'  = score ⊕ {addr? ↦ score(addr?) + 1}
///      ∧ is_block = H(job.mining_hash, nonce?) < job.network_target
///
///   outcome! ∈ {StaleJob, Duplicate, LowDifficulty, Busy} ⇒
///        score' = score                       (no credit, ever)
///      ∧ seen'  = seen                        (no nonce burned)
///      ∧ rejected'(addr?) = rejected(addr?) + 1
///
/// Credit is exactly one point per accepted share, independent of how far
/// H fell below share_target. See `HTTP_MIN_SUBMIT_INTERVAL` and `--share-bits`:
/// changing the share target rescales what a point costs.
/// ```
///
/// ```zed
///     ProcessShare
///     ----------------
///     ΔShares
///     ΔValidShares
///     ΞCurrentJob
///     addr?    : Address
///     job_id?  : ℕ
///     nonce?   : ℕ
///     outcome! : ShareOutcome
///
///     pre  true
///
///     post outcome! = Accepted ⇒
///            job ≠ ⊥ ∧ job.job_id = job_id?
///            ∧ nonce? ∉ seen
///            ∧ H(job.mining_hash, nonce?) < job.share_target
///            ∧ seen' = seen ∪ {nonce?}
///            ∧ score' = score ⊕ {addr? ↦ score addr? + 1}
///
///     post outcome! ≠ Accepted ⇒ score' = score ∧ seen' = seen
///
///     post job' = job          (this operation never rotates the template)
/// ```
///
/// # Safety / Invariants
///
/// - **`seen` only ever holds proven nonces.** Insertion happens strictly after
///   the target comparison. This is what stops an unauthenticated caller from
///   burning nonce space or growing the set on garbage input.
/// - **A rejected share must leave `seen` untouched.** The late-stale path
///   therefore *rolls back* its insertion: `seen` is keyed by nonce alone and
///   cleared per **block**, while jobs rotate more often than blocks (any
///   template rebuild on the same tip). A burned entry would reject that nonce
///   for the remainder of the block even though the new job's midstate makes it
///   legitimately re-mineable.
/// - **Score is credited at most once per (job, nonce).** Enforced by `seen`,
///   which is why its correctness is a payout property and not just a
///   performance one.
/// - **`is_block` is decided from the same hash that cleared `share_target`,**
///   never recomputed, so a share and the block it may also solve can never
///   disagree.
/// - **This is the single entry point for both transports.** Any invariant above
///   holds for Stratum, HTTP and WebRTC identically, because none of them may
///   credit score by another route.
async fn process_share(
    state: &Arc<PoolState>,
    miner_addr: [u8; 32],
    worker_name: &str,
    job_id: u64,
    nonce: u64,
) -> anyhow::Result<ShareOutcome> {
    // Snapshot the job. Everything below validates against THIS job, and the
    // job_id is re-checked after the VDF so a template rotation mid-verify can't
    // credit work against a template that is no longer being paid out.
    let job = match state.current_job.read().await.clone() {
        Some(j) if j.job_id == job_id => j,
        _ => return Ok(ShareOutcome::StaleJob),
    };

    // Cheap probe before spending ~150ms of CPU on a nonce we already hold.
    // The guard is scoped and dropped before the `.await` below: holding a read
    // lock across an await point is how a write-preferring RwLock deadlocks.
    let seen = { state.valid_shares.read().await.contains(&nonce) };
    if seen {
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::Duplicate);
    }

    // Hold a permit across the blocking VDF so the two transports share one
    // CPU budget. `try_acquire` rather than `acquire`: a queued HTTP request
    // just holds a connection open and lets the backlog grow, whereas telling
    // the miner "busy, retry" keeps the work on their side.
    let _permit = match state.verify_permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return Ok(ShareOutcome::Busy),
    };

    let m_hash = job.mining_hash;
    let ext = tokio::task::spawn_blocking(move || create_extension(m_hash, nonce)).await?;
    drop(_permit);

    if ext.final_hash >= job.share_target {
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::LowDifficulty);
    }

    // Claim the nonce now that the work behind it is proven. Losing this race
    // means another connection submitted the same proven nonce first.
    {
        let mut cache = state.valid_shares.write().await;
        if !cache.insert(nonce) {
            drop(cache);
            bump_rejected(state, miner_addr).await;
            return Ok(ShareOutcome::Duplicate);
        }
    }

    // The VDF above is slow enough that a new block can land mid-verify. If the
    // job rotated, the share is against a dead template: don't credit it, and
    // don't try to submit it as a block.
    let still_current = {
        matches!(state.current_job.read().await.as_ref(), Some(j) if j.job_id == job_id)
    };
    if !still_current {
        // Release the nonce we just claimed. `valid_shares` is keyed by nonce
        // alone and is cleared per BLOCK, not per job — several jobs share one
        // cache whenever the template is rebuilt on the same tip. Leaving a
        // burned entry behind would therefore reject that nonce for the rest of
        // the block, even though the rotated job has a different midstate and the
        // nonce is legitimately re-mineable against it. Rolling back keeps the
        // postcondition "outcome ≠ Accepted ⇒ valid_shares' = valid_shares" true.
        state.valid_shares.write().await.remove(&nonce);
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::StaleJob);
    }

    {
        let write_txn = state.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SHARES_TABLE)?;
            let current = table.get(&miner_addr)?.map(|v| v.value()).unwrap_or(0);
            table.insert(&miner_addr, current + 1)?;
        }
        write_txn.commit()?;
    }

    state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).0 += 1;
    *state
        .worker_stats
        .write()
        .await
        .entry((miner_addr, worker_name.to_string()))
        .or_insert(0) += 1;

    let is_block = ext.final_hash < job.network_target;
    if is_block {
        tracing::info!(
            "block found by miner {}. submitting to network.",
            hex::encode(&miner_addr[..8])
        );
        spawn_block_submission(state, job, ext);
    }

    Ok(ShareOutcome::Accepted { is_block })
}

/// Records a rejected share against a miner's lifetime efficiency tally.
async fn bump_rejected(state: &Arc<PoolState>, miner_addr: [u8; 32]) {
    state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).1 += 1;
}

/// Reconstructs the full block from the pool's own stored template, submits it
/// to the core node, and — only on a confirmed `HTTP 200` — applies the score
/// deductions and writes the payout/committed-score history.
///
/// # Security
/// The batch is rebuilt from `job.batch_template`, the template the pool itself
/// generated and broadcast. Nothing block-shaped supplied by a miner is ever
/// used: a share submission carries only a `job_id` and a `nonce`, so a miner
/// cannot rewrite the coinbase to redirect the reward to themselves.
fn spawn_block_submission(state: &Arc<PoolState>, job: Job, ext: Extension) {
    // Capture block identity for the dashboard BEFORE `job` is consumed: the PoW
    // hash (for the Explorer hyperlink) and the network target in force for this
    // block, so historical "luck" is computed exactly rather than approximated
    // against the *current* difficulty.
    let block_hash_hex = hex::encode(ext.final_hash);
    let block_net_target_hex = hex::encode(job.network_target);
    let block_height = job.height;
    let committed_scores = job.committed_scores.clone();

    let mut batch: Batch = match serde_json::from_value(job.batch_template) {
        Ok(b) => b,
        Err(e) => {
            // Previously an `.unwrap()`. A malformed template is the node's or
            // the pool's fault, not the miner's, and panicking here would kill
            // the connection task holding a perfectly good block.
            tracing::error!("could not rebuild block from stored template: {}. dropping block.", e);
            state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            return;
        }
    };
    batch.extension = ext;

    let total_reward: u64 = batch.coinbase.iter().map(|cb| cb.value).sum();
    let batch_for_node = batch.clone();
    let db_clone = state.db.clone();
    let rpc_url = state.node_rpc_url.clone();
    let submit_state = state.clone();

    tokio::spawn(async move {
        let res = reqwest::Client::new()
            .post(&format!("{}/submit_batch", rpc_url))
            .json(&batch_for_node)
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!("block accepted by network. applying score deductions.");
                let write_txn = db_clone.begin_write().unwrap();
                {
                    let mut table = write_txn.open_table(SHARES_TABLE).unwrap();
                    let mut total_score = 0u128;
                    for iter in table.iter().unwrap() { total_score += iter.unwrap().1.value() as u128; }

                    let mut payouts = Vec::new();

                    // Track exact deductions to allow refunds on orphan ---
                    let mut actual_deductions = Vec::new();

                    for cb in &batch.coinbase {
                        let mut a = [0u8; 32]; a.copy_from_slice(&cb.address);
                        let deduction = ((cb.value as u128 * total_score) / (total_reward as u128)) as u64;
                        // Deduct ONLY from rows that already exist, and delete a row once its
                        // remaining score hits 0. Writing back `0.saturating_sub(d) = 0` for an
                        // absent address (the pool fee address, or any coinbase output not in
                        // SHARES_TABLE) is what previously seeded permanent zero-score rows; those
                        // rows then leaked into the allocator and captured block rewards. redb has
                        // no delete-if-absent, so guard the remove on the row existing.
                        if let Some(current) = table.get(&a).unwrap().map(|v| v.value()) {
                            let remaining = current.saturating_sub(deduction);
                            if remaining > 0 {
                                table.insert(&a, remaining).unwrap();
                            } else {
                                table.remove(&a).unwrap();
                                // Record deduction ---
                                actual_deductions.push((hex::encode(a), deduction));
                            }
                        }

                        payouts.push(serde_json::json!({
                            "address": crate::core::types::encode_address_with_checksum(&a),
                            "value": cb.value
                        }));
                    }

                    let mut b_table = write_txn.open_table(BLOCKS_TABLE).unwrap();
                    let block_data = serde_json::json!({
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                        "block_ts": batch.timestamp,
                        "hash": block_hash_hex,
                        "height": block_height,
                        "total_score": total_score as u64,
                        "net_target": block_net_target_hex,
                        "payouts": payouts,
                        "status": "pending" 
                    }).to_string();
                    b_table.insert(batch.timestamp, block_data.as_str()).unwrap();

                    // Save to Pending Blocks for Maturity Pipeline ---
                    let pending_data = serde_json::json!({
                        "hash": block_hash_hex,
                        "block_ts": batch.timestamp,
                        "deductions": actual_deductions
                    }).to_string();
                    let mut p_table = write_txn.open_table(PENDING_BLOCKS_TABLE).unwrap();
                    p_table.insert(block_height, pending_data.as_str()).unwrap();
                    // ---------------------------------------------------------

                    // Persist the committed score snapshot (split-verification).
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
                // The node answered but refused the block (stale, invalid, etc).
                // The tip won't change on a rejection, so force the template loop
                // to rebuild instead of letting miners re-grind the doomed template.
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
                // timeout, dropped mid-flight). Scores are retained (deductions
                // only happen on confirmed acceptance) and a fresh job is forced.
                tracing::error!(
                    "block submission to node failed: {}. retaining miner scores; requesting fresh job.",
                    e
                );
                submit_state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
    });
}

/// Reconciles pending blocks against the canonical chain to detect orphans and refund shares.
///
/// # Reasoning
/// When the pool submits a block and the node accepts it (HTTP 200 OK), the pool deducts 
/// shares from miners to prevent double-paying if they continue mining. However, due to 
/// standard network latency and Nakamoto Consensus tie-breakers (e.g., midstate comparison), 
/// the local node might reorganize and abandon the pool's block seconds later.
/// 
/// Previously, the pool never checked the chain again. If a block was orphaned, miners 
/// permanently lost their shares for a block that yielded no actual rewards, destroying 
/// trust in the pool. This function introduces a maturity pipeline: it verifies block 
/// hashes at `MATURITY_DEPTH` and refunds all deducted shares if the block was orphaned.
///
/// # Formal Specification
///
/// ```text
/// Pre:
///   - MATURITY_DEPTH = 10
///   - For every pending block b in PENDING_BLOCKS_TABLE where b.height <= current_network_height - MATURITY_DEPTH
///   - The core node is reachable via RPC
///
/// Post:
///   result = Ok(()) ⇒
///     ∀ b ∈ evaluated_blocks:
///       if network_hash(b.height) == b.hash then
///         PENDING_BLOCKS_TABLE' = PENDING_BLOCKS_TABLE \ {b}
///         BLOCKS_TABLE'[b.ts].status = "confirmed"
///       else (Orphaned)
///         PENDING_BLOCKS_TABLE' = PENDING_BLOCKS_TABLE \ {b}
///         BLOCKS_TABLE'[b.ts].status = "orphaned"
///         ∀ (addr, amt) ∈ b.deductions:
///           SHARES_TABLE'[addr] = SHARES_TABLE[addr] + amt
/// ```
///
/// ```zed
///     ReconcilePendingBlocks
///     ----------------------
///     ΔShares
///     ΔPendingBlocks
///     ΔBlocks
///     current_height? : ℕ
///     network : RPC
///
///     let mature = { b ∈ PendingBlocks | b.height ≤ current_height? - 10 }
///
///     pre  true
///     post ∀ b ∈ mature •
///            (network.hash_at(b.height) = b.hash ⇒ 
///               PendingBlocks' = PendingBlocks \ {b} ∧
///               Blocks'[b.ts].status = "confirmed")
///          ∧ (network.hash_at(b.height) ≠ b.hash ⇒
///               PendingBlocks' = PendingBlocks \ {b} ∧
///               Blocks'[b.ts].status = "orphaned" ∧
///               Shares' = Shares ⊕ { d.addr ↦ Shares(d.addr) + d.amount | d ∈ b.deductions })
/// ```
///
/// # Safety / Invariants
/// - **No Double Refunds**: Blocks are removed from `PENDING_BLOCKS_TABLE` in the same
///   atomic database transaction that restores the shares.
/// - **Idempotent network reads**: Network failures skip reconciliation for that tick;
///   the block remains pending until a clean read confirms its status.
async fn reconcile_pending_blocks(state: &Arc<PoolState>, client: &reqwest::Client, current_height: u64) -> Result<(), String> {
    const MATURITY_DEPTH: u64 = 10;
    let maturity_threshold = current_height.saturating_sub(MATURITY_DEPTH);

    // 1. Gather mature pending blocks
    let mut mature_blocks = Vec::new();
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(PENDING_BLOCKS_TABLE) {
            let range = 0..=maturity_threshold;
            if let Ok(iter) = table.range(range) {
                for entry in iter {
                    if let Ok((k, v)) = entry {
                        mature_blocks.push((k.value(), v.value().to_string()));
                    }
                }
            }
        }
    }

    if mature_blocks.is_empty() {
        return Ok(());
    }

    let mut confirmed = Vec::new();
    let mut orphaned = Vec::new();

    // 2. Query the node for the canonical block at each mature height
    for (height, json_str) in mature_blocks {
        let pending_data: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
        let expected_hash = pending_data["hash"].as_str().unwrap_or("");
        
        let url = format!("{}/block/{}", state.node_rpc_url, height);
        match client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(batch) = resp.json::<serde_json::Value>().await {
                        let canonical_hash = batch["extension"]["final_hash"].as_str().unwrap_or("");
                        if canonical_hash == expected_hash {
                            confirmed.push((height, pending_data));
                        } else {
                            orphaned.push((height, pending_data));
                        }
                    }
                } else if resp.status() == 404 || resp.status() == 400 || resp.status() == 500 {
                    // Block not found or invalid height on node -> definitively orphaned
                    orphaned.push((height, pending_data));
                }
            }
            Err(e) => {
                tracing::warn!("Failed to query node for block {}: {}. Will retry reconciliation later.", height, e);
                // Network error, skip reconciliation for this block until next tick
            }
        }
    }

    if confirmed.is_empty() && orphaned.is_empty() {
        return Ok(());
    }

    // 3. Atomically apply resolutions
    let write_txn = state.db.begin_write().map_err(|e| e.to_string())?;
    {
        let mut p_table = write_txn.open_table(PENDING_BLOCKS_TABLE).map_err(|e| e.to_string())?;
        let mut b_table = write_txn.open_table(BLOCKS_TABLE).map_err(|e| e.to_string())?;
        let mut s_table = write_txn.open_table(SHARES_TABLE).map_err(|e| e.to_string())?;

        // Process Confirmed
        for (height, data) in confirmed {
            p_table.remove(height).map_err(|e| e.to_string())?;
            
            let block_ts = data["block_ts"].as_u64().unwrap_or(0);
            
            // Isolate the read to drop the AccessGuard before writing
            let mut b_json = None;
            if let Ok(Some(b_val)) = b_table.get(block_ts) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(b_val.value()) {
                    b_json = Some(parsed);
                }
            }

            if let Some(mut json) = b_json {
                json["status"] = serde_json::json!("confirmed");
                b_table.insert(block_ts, json.to_string().as_str()).map_err(|e| e.to_string())?;
            }
            
            tracing::info!("Block at height {} reached maturity (Confirmed).", height);
        }

        // Process Orphans and Refund Shares
        for (height, data) in orphaned {
            p_table.remove(height).map_err(|e| e.to_string())?;
            
            let block_ts = data["block_ts"].as_u64().unwrap_or(0);
            
            // Isolate the read to drop the AccessGuard before writing
            let mut b_json = None;
            if let Ok(Some(b_val)) = b_table.get(block_ts) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(b_val.value()) {
                    b_json = Some(parsed);
                }
            }

            if let Some(mut json) = b_json {
                json["status"] = serde_json::json!("orphaned");
                b_table.insert(block_ts, json.to_string().as_str()).map_err(|e| e.to_string())?;
            }

            // Refund the shares
            if let Some(deductions) = data["deductions"].as_array() {
                for d in deductions {
                    let addr_hex = d[0].as_str().unwrap_or("");
                    let amount = d[1].as_u64().unwrap_or(0);
                    
                    if let Ok(addr_bytes) = hex::decode(addr_hex) {
                        if let Ok(addr_array) = <[u8; 32]>::try_from(addr_bytes.as_slice()) {
                            let current = s_table.get(&addr_array).map_err(|e| e.to_string())?.map(|v| v.value()).unwrap_or(0);
                            s_table.insert(&addr_array, current + amount).map_err(|e| e.to_string())?;
                        }
                    }
                }
            }
            tracing::warn!("🚨 ORPHAN DETECTED at height {}. Shares have been fully refunded to miners.", height);
        }
    }
    write_txn.commit().map_err(|e| e.to_string())?;

    Ok(())
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
                        // params: [worker_name, job_id, nonce]. Missing/!numeric
                        // params previously hit `.unwrap()` and killed the task,
                        // dropping the whole connection on one malformed frame.
                        let job_id = req.params.get(1).and_then(|v| v.as_u64());
                        let nonce = req.params.get(2).and_then(parse_nonce);

                        match (authorized_address, job_id, nonce) {
                            (Some(miner_addr), Some(job_id), Some(nonce)) => {
                                let outcome = process_share(
                                    &state, miner_addr, &authorized_worker, job_id, nonce,
                                ).await?;

                                let (ok, err) = match outcome {
                                    ShareOutcome::Accepted { .. } => (true, None),
                                    ShareOutcome::Duplicate      => (false, Some("Duplicate share")),
                                    ShareOutcome::LowDifficulty  => (false, Some("Low difficulty")),
                                    ShareOutcome::StaleJob       => (false, Some("Stale job")),
                                    ShareOutcome::Busy           => (false, Some("Verifier busy, retry")),
                                };
                                let res = StratumResponse {
                                    id: req.id,
                                    result: Some(serde_json::json!(ok)),
                                    error: err.map(|e| e.to_string()),
                                };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            }
                            (None, _, _) => {
                                let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Not authorized".into()) };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            }
                            _ => {
                                let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Malformed submit params".into()) };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
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
