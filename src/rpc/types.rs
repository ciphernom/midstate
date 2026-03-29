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
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputDataJson {
    Standard {
        address: String, 
        value: u64,
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionResponse {
    pub input_coins: Vec<String>,
    pub output_coins: Vec<String>,
    pub fee: u64,
    pub status: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ScanRequest {
    pub addresses: Vec<String>,
    pub start_height: u64,
    pub end_height: u64,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct TransactionInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_coins: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_coins: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fee: Option<u64>,
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
