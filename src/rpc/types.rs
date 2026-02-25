use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitRequest {
    /// Input coin IDs being spent (hex, 32 bytes each)
    pub coins: Vec<String>,
    /// Output coin IDs (hex, 32 bytes each)
    pub destinations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitResponse {
    pub commitment: String,
    pub salt: String,
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
    pub depth: u64,
    /// Number of blocks required for 1e-6 finality risk.
    pub safe_depth: u64,
    pub midstate: String,
    pub num_coins: usize,
    pub num_commitments: usize,
    pub target: String,
    pub block_reward: u64,
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
