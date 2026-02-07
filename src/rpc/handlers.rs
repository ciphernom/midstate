use super::types::*;
use crate::core::{compute_commitment, wots, block_reward, Transaction};
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
        num_commitments: state.commitments.len(),
        target: hex::encode(state.target),
        block_reward: block_reward(state.height),
    })
}

fn parse_hex32(hex_str: &str, label: &str) -> Result<[u8; 32], ErrorResponse> {
    let bytes = hex::decode(hex_str)
        .map_err(|e| ErrorResponse { error: format!("Invalid {} hex: {}", label, e) })?;
    if bytes.len() != 32 {
        return Err(ErrorResponse { error: format!("{} must be 32 bytes", label) });
    }
    Ok(<[u8; 32]>::try_from(bytes).unwrap())
}

/// Phase 1: Register a commitment
pub async fn commit_transaction(
    State(node): State<AppState>,
    Json(req): Json<CommitRequest>,
) -> Result<Json<CommitResponse>, ErrorResponse> {
    if req.coins.is_empty() {
        return Err(ErrorResponse { error: "Must provide at least one coin".into() });
    }
    if req.destinations.is_empty() {
        return Err(ErrorResponse { error: "Must provide at least one destination".into() });
    }

    let input_coins: Vec<[u8; 32]> = req.coins.iter()
        .map(|h| parse_hex32(h, "coin"))
        .collect::<Result<_, _>>()?;

    let destinations: Vec<[u8; 32]> = req.destinations.iter()
        .map(|h| parse_hex32(h, "destination"))
        .collect::<Result<_, _>>()?;

    let salt: [u8; 32] = rand::random();
    let commitment = compute_commitment(&input_coins, &destinations, &salt);

    let tx = Transaction::Commit { commitment };
    node.send_transaction(tx)
        .await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(CommitResponse {
        commitment: hex::encode(commitment),
        salt: hex::encode(salt),
        status: "committed".to_string(),
    }))
}

/// Phase 2: Reveal and execute the spend
pub async fn send_transaction(
    State(node): State<AppState>,
    Json(req): Json<SendTransactionRequest>,
) -> Result<Json<SendTransactionResponse>, ErrorResponse> {
    if req.input_coins.is_empty() {
        return Err(ErrorResponse { error: "Must provide at least one input coin".into() });
    }
    if req.signatures.len() != req.input_coins.len() {
        return Err(ErrorResponse {
            error: "Signature count must match input coin count".into(),
        });
    }

    let input_coins: Vec<[u8; 32]> = req.input_coins.iter()
        .map(|h| parse_hex32(h, "input_coin"))
        .collect::<Result<_, _>>()?;

    let mut signatures = Vec::new();
    for sig_hex in &req.signatures {
        let sig_bytes = hex::decode(sig_hex)
            .map_err(|e| ErrorResponse { error: format!("Invalid signature hex: {}", e) })?;
        signatures.push(sig_bytes);
    }
    
    let destinations: Vec<[u8; 32]> = req.destinations.iter()
        .map(|h| parse_hex32(h, "destination"))
        .collect::<Result<_, _>>()?;

    let salt = parse_hex32(&req.salt, "salt")?;

    let tx = Transaction::Reveal {
        input_coins: input_coins.clone(),
        signatures,
        new_coins: destinations.clone(),
        salt,
    };

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
    let coin = parse_hex32(&req.coin, "coin")?;
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
        .map(|tx| match tx {
            Transaction::Commit { commitment } => TransactionInfo {
                commitment: Some(hex::encode(commitment)),
                input_coins: None,
                output_coins: None,
            },
            Transaction::Reveal { input_coins, new_coins, .. } => TransactionInfo {
                commitment: None,
                input_coins: Some(input_coins.iter().map(|c| hex::encode(c)).collect()),
                output_coins: Some(new_coins.iter().map(|c| hex::encode(c)).collect()),
            },
        })
        .collect();

    Json(GetMempoolResponse { size, transactions: tx_info })
}

pub async fn generate_key() -> Json<GenerateKeyResponse> {
    let seed: [u8; 32] = rand::random();
    let coin = wots::keygen(&seed);

    Json(GenerateKeyResponse {
        seed: hex::encode(seed),
        coin: hex::encode(coin),
    })
}

pub async fn get_peers(State(node): State<AppState>) -> Json<GetPeersResponse> {
    let peers = node.get_peers().await;
    Json(GetPeersResponse {
        peers: peers.iter().map(|p| p.to_string()).collect(),
    })
}
