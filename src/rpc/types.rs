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
    pub owner_pk: String,  
    pub value: u64,
    pub salt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputDataJson {
    pub address: String, 
    pub value: u64,
    pub salt: String,
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
