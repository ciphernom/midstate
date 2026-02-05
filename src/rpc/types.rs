use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitRequest {
    /// Coin IDs being spent (hex)
    pub coins: Vec<String>,
    /// Destination coin commitments (hex, 32 bytes each)
    pub destinations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CommitResponse {
    pub commitment: String,
    pub salt: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionRequest {
    pub secrets: Vec<String>,
    pub destinations: Vec<String>,
    /// Salt from the commit phase (hex)
    pub salt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionResponse {
    pub input_coins: Vec<String>,
    pub output_coins: Vec<String>,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetStateResponse {
    pub height: u64,
    pub depth: u64,
    pub midstate: String,
    pub num_coins: usize,
    pub num_commitments: usize,
    pub target: String,
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateKeyResponse {
    pub secret: String,
    pub coin: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetPeersResponse {
    pub peers: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}
