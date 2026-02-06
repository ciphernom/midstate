use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionRequest {
    pub secrets: Vec<String>,  // Changed from single secret
    pub destinations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendTransactionResponse {
    pub input_coins: Vec<String>,  // Changed from single input
    pub output_coins: Vec<String>,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetStateResponse {
    pub height: u64,
    pub depth: u64,
    pub midstate: String,
    pub num_coins: usize,
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
    pub input_coins: Vec<String>,  // Changed from single input
    pub output_coins: Vec<String>,
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
