use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitRequest {
    /// Client-computed commitment hash (hex, 32 bytes)
    pub commitment: String,
    /// Client-mined PoW nonce
    pub spam_nonce: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitResponse {
    pub commitment: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputRevealJson {
    pub bytecode: String,
    pub value: u64,
    pub salt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commitment: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputDataJson {
    Standard {
        address: String, 
        value: u64,
        salt: String,
    },
    Confidential {
        address: String,
        commitment: String,
        salt: String,
    },
    DataBurn {
        payload: String,
        value_burned: u64,
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionRequest {
    pub inputs: Vec<InputRevealJson>,
    pub signatures: Vec<String>,
    pub outputs: Vec<OutputDataJson>,
    /// Commitment salt (hex)
    pub salt: String,
    /// Flag to indicate this is a dust-sweeping Consolidate transaction
    #[serde(default)]
    pub is_consolidate: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionResponse {
    pub input_coins: Vec<String>,
    pub output_coins: Vec<String>,
    pub fee: u64,
    pub status: String,
}
/// A client's proof of work for a gated endpoint.
///
/// Used by BOTH `/scan` and `/search` — the name predates the second caller and
/// is now a mild misnomer, but renaming it buys nothing and touches two files.
///
/// The preimage, which both sides must build identically:
///
/// ```text
/// blake3(format!("{}:{}:{}:{}", subject, height, timestamp, nonce))
/// ```
///
/// where `subject` is the address (`/scan`) or the query (`/search`), and
/// `height` is the request's `start_height` (`/scan`) or `0` (`/search`). The
/// digest must carry at least the leading HEX zeros the governor demands for the
/// caller's IP — ask `/pow_params`, don't guess. Difficulty is server-owned.
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ScanPow {
    pub nonce: u64,
    /// Unix seconds. The server rejects anything outside -600s / +120s of its
    /// own clock, so a skewed client fails before it has mined anything useful.
    pub timestamp: u64,
    /// Lowercase hex digest the client claims to have found.
    ///
    /// Redundant — the server recomputes it — but worth sending: it separates
    /// "you hashed the wrong thing" (Invalid PoW hash) from "you didn't work
    /// hard enough" (difficulty too low). Those are very different client bugs,
    /// and collapsing them cost real debugging time on the explorer.
    pub hash: String,
}

/// Reply from `GET /pow_params`: what this caller owes for its next expensive
/// request.
///
/// Free to call, by necessity — a client cannot discover the price without
/// asking, so charging for the quote would deadlock. The server re-derives the
/// requirement when it verifies, so a stale quote can't buy a discount; a client
/// outbid mid-solve is rejected, re-quotes, and retries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowParamsResponse {
    /// Leading HEX zeros required right now, for this caller's IP.
    pub zeros: u32,
    /// The floor every caller pays.
    pub min_zeros: u32,
    /// The ceiling, so a client can show an honest worst case up front.
    pub max_zeros: u32,
    /// Counter window in seconds; difficulty decays back to `min_zeros` after it.
    pub window_secs: u64,
    /// Server unix time. Lets a client detect its own clock skew before it
    /// spends a solve on a timestamp that will be rejected as stale.
    pub server_time: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanRequest {
    pub addresses: Vec<String>,
    pub start_height: u64,
    pub end_height: u64,
    #[serde(default)]
    pub pow: Option<ScanPow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanResponse {
    pub coins: Vec<ScanCoin>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScanCoin {
    pub address: String,
    pub value: u64,
    pub salt: String,
    pub coin_id: String,
    pub height: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetStateResponse {
    pub height: u64,
    pub depth: u128,
    /// Number of blocks required for 1e-6 finality risk.
    pub safe_depth: u64,
    pub midstate: String,
    pub num_coins: usize,
    pub num_commitments: usize,
    pub target: String,
    pub block_reward: u64,
    pub required_pow: u32,
    pub webrtc_addrs: Vec<String>, 
    pub header_hash: String, 
    /// True while the node is bulk-syncing historical blocks. Pools and miners
    /// should pause template generation while this is set — every tip reported
    /// during a sync is an already-superseded height. `#[serde(default)]` so
    /// clients deserializing responses from older nodes (field absent) don't fail.
    #[serde(default)]
    pub is_syncing: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct GetMssStateRequest {
    pub master_pk: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetMssStateResponse {
    pub next_index: u64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckCoinRequest {
    pub coin: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckCoinResponse {
    pub exists: bool,
    pub coin: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckCommitmentRequest {
    pub commitment: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckCommitmentResponse {
    pub exists: bool,
    pub commitment: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetMempoolResponse {
    pub size: usize,
    pub transactions: Vec<TransactionInfo>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TransactionInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_coins: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_coins: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u64>,
    // ── Detail fields (additive: older clients ignore them, and every field
    //    is an Option so newer clients tolerate older nodes omitting them) ──
    /// "commit" | "reveal" | "consolidate"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tx_type: Option<String>,
    /// Seconds this entry has been waiting in the mempool. Computed
    /// server-side (now − arrival), so it is immune to client clock skew.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_secs: Option<u64>,
    /// Commits: achieved PoW in leading zero bits — the commit queue's actual
    /// priority key (weakest is evicted first).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pow_zeros: Option<u32>,
    /// Reveals: serialized transaction size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Reveals: integer fee rate (fee × 1024 / size_bytes) — mirrors the fee
    /// market's actual priority key (cheapest per byte is evicted first).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee_per_kb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_in: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_out: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<MempoolInputInfo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<MempoolOutputInfo>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MempoolInputInfo {
    pub coin_id: String,
    pub value: u64,
    /// The Phase-1 commitment this input opens, when present — lets the
    /// explorer link a pending Reveal back to its Commit in the other queue.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MempoolOutputInfo {
    /// "standard" | "confidential" | "data_burn"
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    /// Standard value or DataBurn value_burned; None for Confidential
    /// outputs, whose value is hidden by design.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coin_id: Option<String>,
    /// DataBurn: hex payload preview, capped at 64 bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_preview: Option<String>,
    /// DataBurn: full payload length in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload_len: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateKeyResponse {
    pub seed: String,
    pub address: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetPeersResponse {
    pub peers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

// ── CoinJoin Mix Types ──────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct MixCreateRequest {
    pub denomination: u64,
    #[serde(default = "default_min_participants")]
    pub min_participants: usize,
}
fn default_min_participants() -> usize { 2 }

#[derive(Debug, Serialize, Deserialize)]
pub struct MixCreateResponse {
    pub mix_id: String,
    pub denomination: u64,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixRegisterRequest {
    pub mix_id: String,
    /// Hex coin_id of the input being mixed
    pub coin_id: String,
    /// Input reveal data
    pub input: InputRevealJson,
    /// Output data for the mixed coin
    pub output: OutputDataJson,
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetFiltersRequest {
    pub start_height: u64,
    pub end_height: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetFiltersResponse {
    pub start_height: u64,
    pub filters: Vec<String>, // Hex-encoded filter data
    /// Block hashes (final_hash) keying each filter — needed for client-side matching
    #[serde(default)]
    pub block_hashes: Vec<String>,
    /// Number of elements in each filter — needed for Golomb-Rice decoding
    #[serde(default)]
    pub element_counts: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixFeeRequest {
    pub mix_id: String,
    /// Input reveal for the denomination-1 fee coin
    pub input: InputRevealJson,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixSignRequest {
    pub mix_id: String,
    /// Hex coin_id the wallet is signing for (used to find input_index)
    pub coin_id: String,
    /// Hex-encoded signature
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixStatusResponse {
    pub mix_id: String,
    pub denomination: u64,
    pub participants: usize,
    pub phase: String,
    /// Set when phase == "signing"
    pub commitment: Option<String>,
    /// Input coin IDs in canonical proposal order (for the wallet to find its index)
    pub input_coin_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixListResponse {
    pub sessions: Vec<MixStatusResponse>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MixActionResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_index: Option<usize>,
}

// ── Explorer Search Types ───────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Hex-encoded 32-byte hash to search for
    pub query: String,
    /// Proof of work. Required in practice — /search loads and linearly scans up
    /// to 5000 batches per call, more disk I/O than /scan, which was already
    /// gated while this was wide open.
    ///
    /// Kept `Option` + `serde(default)` deliberately: an absent proof should
    /// produce a readable "Search requires proof-of-work — see /pow_params",
    /// not a serde 422 that a client can't act on.
    ///
    /// Reuses `ScanPow` rather than declaring a second identical struct — the
    /// preimage is the same shape for both endpoints, and one type is one fewer
    /// thing to drift. (The name is now a slight misnomer; see the note there.)
    #[serde(default)]
    pub pow: Option<ScanPow>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub result_type: String, // "block_hash", "commitment", "address", "coin_id", "salt"
    pub height: u64,
    pub tx_index: Option<usize>,
    pub detail: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct AxeStatsResponse {
    pub temp_c: f32,
    pub uptime_s: u64,
    pub total_nonces: u64,
    pub is_axe_hardware: bool,
    /// Live CPU frequency in MHz (from cpufreq), when readable. Lets the
    /// dashboard show the *actual* clock instead of a hardcoded guess.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freq_mhz: Option<u32>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct AxeWifiRequest {
    pub ssid: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AxeConfigRequest {
    pub mode: String, // "solo" or "pool"
    pub pool_url: Option<String>,
    pub payout_address: Option<String>,
    pub pool_address: Option<String>, // <-- NEW: The pool's MSS address
    /// Optional rig name reported to the pool for per-worker stats.
    /// `default` keeps requests from older dashboards (no field) parsing.
    #[serde(default)]
    pub worker: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AxeOverclockRequest {
    pub freq: u32,
    pub overvoltage: u32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct GetMetricsResponse {
    pub transactions_processed: u64,
    pub batches_processed: u64,
    pub batches_mined: u64,
    pub invalid_batches: u64,
    pub invalid_transactions: u64,
    pub reorgs: u64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct CheckOutputRequest {
    pub address: String,
    pub value: u64,
    pub salt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckOutputResponse {
    pub coin_id: String,
    pub exists: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct GetTxByInputRequest {
    /// The 32-byte hex ID of the coin being spent (the HTLC coin)
    pub coin_id: String,
}

// ── Block Template Types (web miner) ────────────────────────────────────

/// Request from a web miner to build a block template.
///
/// The miner provides its coinbase outputs (with pre-derived WOTS addresses).
/// The node validates the total, selects mempool transactions, computes the
/// state root and mining midstate, and returns a ready-to-mine template.
///
/// # Example
///
/// ```json
/// {
///   "coinbase": [
///     { "address": "ab01...ff", "value": 512, "salt": "cd23...ee" },
///     { "address": "ef45...aa", "value": 256, "salt": "01ab...99" }
///   ]
/// }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockTemplateRequest {
    pub coinbase: Vec<CoinbaseOutputJson>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CoinbaseOutputJson {
    pub address: String,
    pub value: u64,
    pub salt: String,
}

/// Response containing a complete block template ready for nonce searching.
///
/// The web miner searches nonces against `mining_midstate`, then fills in
/// the `extension` field of `batch_template` with `{ nonce, final_hash }`
/// and submits to `/api/internal/submit_batch`.
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockTemplateResponse {
    /// Post-tx, post-coinbase, post-state-root midstate to mine against (hex).
    pub mining_midstate: String,
    /// Current difficulty target (hex).
    pub target: String,
    /// A serialized `Batch` with placeholder extension. Client fills in
    /// `extension` and `timestamp` after finding a valid nonce.
    pub batch_template: serde_json::Value,
    /// Total fees from included mempool transactions.
    pub total_fees: u64,
    /// Block reward at the current height.
    pub block_reward: u64,
}

/// Returned as a 409 when the miner's coinbase total doesn't match
/// `block_reward + mempool_fees`. The client should rebuild its coinbase
/// for `expected_total` and retry.
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockTemplateMismatchError {
    pub error: String,
    pub expected_total: u64,
    pub block_reward: u64,
    pub total_fees: u64,
}

/// HTTP request body for `POST /api/chat`.
///
/// # Validation (in `crate::rpc::handlers::send_chat`)
///
/// ```text
/// (#words ≥ 1 ∨ #attachments ≥ 1)
/// #words ≤ 10
/// ∀ w ∈ ran words • w < #CHAT_DICTIONARY
/// #attachments ≤ MAX_CHAT_ATTACHMENTS
/// ```
///
/// # JSON shape
///
/// ```json
/// {
///   "reply_to": null,
///   "words": [42, 81, 200],
///   "attachments": [
///     { "kind": "address", "value": "ab01...64-hex-chars...ff" }
///   ]
/// }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct SendChatRequest {
    pub reply_to: Option<u64>,
    pub words: Vec<u8>,
    #[serde(default)]
    pub attachments: Vec<crate::node::ChatAttachment>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitChatRequest {
    pub sender: String,
    pub timestamp: u64,
    pub nonce: u64,
    pub reply_to: Option<u64>,
    pub words: Vec<u8>,
    #[serde(default)]
    pub attachments: Vec<crate::node::ChatAttachment>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct GetChatResponse {
    pub messages: Vec<crate::node::ChatMessage>,
    pub dictionary: Vec<String>,
}
// ── Analytics / Chart Types ─────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockStat {
    pub height: u64,
    pub timestamp: u64,
    pub target: String,
    pub tx_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChainStatsResponse {
    pub blocks: Vec<BlockStat>,
}
