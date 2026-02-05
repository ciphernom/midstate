use super::types::*;
use crate::core::{hash, Transaction};
use crate::node::NodeHandle;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

type AppState = NodeHandle;

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(self)).into_response()
    }
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn get_state(State(node): State<AppState>) -> Json<GetStateResponse> {
    let state = node.get_state().await;
    
    Json(GetStateResponse {
        height: state.height,
        depth: state.depth,
        midstate: hex::encode(state.midstate),
        num_coins: state.coins.len(),
        target: hex::encode(state.target),
    })
}

pub async fn send_transaction(
    State(node): State<AppState>,
    Json(req): Json<SendTransactionRequest>,
) -> Result<Json<SendTransactionResponse>, ErrorResponse> {
    if req.secrets.is_empty() {
        return Err(ErrorResponse {
            error: "Must provide at least one secret".to_string(),
        });
    }
    
    let mut secrets = Vec::new();
    for secret_hex in &req.secrets {
        let secret = hex::decode(secret_hex)
            .map_err(|e| ErrorResponse { error: format!("Invalid secret hex: {}", e) })?;
        secrets.push(secret);
    }
    
    let mut destinations = Vec::new();
    for dest_hex in &req.destinations {
        let dest = hex::decode(dest_hex)
            .map_err(|e| ErrorResponse { error: format!("Invalid destination hex: {}", e) })?;
        
        if dest.len() != 32 {
            return Err(ErrorResponse {
                error: "Destination must be 32 bytes".to_string(),
            });
        }
        
        let dest: [u8; 32] = dest.try_into().unwrap();
        destinations.push(dest);
    }
    
    let tx = Transaction {
        secrets,
        new_coins: destinations.clone(),
    };
    
    let input_coins = tx.input_coins();
    
    node.send_transaction(tx)
        .await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;
    
    Ok(Json(SendTransactionResponse {
        input_coins: input_coins.iter().map(|c| hex::encode(c)).collect(),
        output_coins: destinations.iter().map(|d| hex::encode(d)).collect(),
        status: "submitted".to_string(),
    }))
}

pub async fn check_coin(
    State(node): State<AppState>,
    Json(req): Json<CheckCoinRequest>,
) -> Result<Json<CheckCoinResponse>, ErrorResponse> {
    let coin = hex::decode(&req.coin)
        .map_err(|e| ErrorResponse { error: format!("Invalid coin hex: {}", e) })?;
    
    if coin.len() != 32 {
        return Err(ErrorResponse {
            error: "Coin must be 32 bytes".to_string(),
        });
    }
    
    let coin: [u8; 32] = coin.try_into().unwrap();
    let exists = node.check_coin(coin).await;
    
    Ok(Json(CheckCoinResponse {
        exists,
        coin: hex::encode(coin),
    }))
}

pub async fn get_mempool(State(node): State<AppState>) -> Json<GetMempoolResponse> {
    let (size, transactions) = node.get_mempool_info().await;
    
    let tx_info: Vec<_> = transactions
        .iter()
        .map(|tx| TransactionInfo {
            input_coins: tx.input_coins().iter().map(|c| hex::encode(c)).collect(),
            output_coins: tx.new_coins.iter().map(|c| hex::encode(c)).collect(),
        })
        .collect();
    
    Json(GetMempoolResponse {
        size,
        transactions: tx_info,
    })
}

pub async fn generate_key() -> Json<GenerateKeyResponse> {
    let secret: [u8; 32] = rand::random();
    let coin = hash(&secret);
    
    Json(GenerateKeyResponse {
        secret: hex::encode(secret),
        coin: hex::encode(coin),
    })
}

pub async fn get_peers(State(node): State<AppState>) -> Json<GetPeersResponse> {
    let peers = node.get_peers().await;
    
    Json(GetPeersResponse {
        peers: peers.iter().map(|p| p.to_string()).collect(),
    })
}
