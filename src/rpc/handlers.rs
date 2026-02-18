use super::types::*;
use crate::core::{compute_commitment, compute_address, hash_concat, wots,
                  block_reward, Transaction, InputReveal, OutputData};
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
pub async fn get_mss_state(
    State(node): State<AppState>,
    Json(req): Json<GetMssStateRequest>,
) -> Result<Json<GetMssStateResponse>, ErrorResponse> {
    let master_pk = parse_hex32(&req.master_pk, "master_pk")?;
    let state = node.get_state().await;

    // Scan chain history
    let chain_max = node.scan_mss_index(&master_pk, state.height)
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    // Scan mempool
    let (_, mempool_txs) = node.get_mempool_info().await;
    let mempool_max = crate::node::scan_txs_for_mss_index(&mempool_txs, &master_pk);

    let next_index = chain_max.max(mempool_max);

    Ok(Json(GetMssStateResponse { next_index }))
}
pub async fn get_state(State(node): State<AppState>) -> Json<GetStateResponse> {
    let state = node.get_state().await;
    let safe_depth = node.get_safe_depth().await;

    Json(GetStateResponse {
        height: state.height,
        depth: state.depth,
        safe_depth,
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

    // Mine PoW nonce for anti-spam
    let spam_nonce = {
        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 {
                break n;
            }
            n += 1;
        }
    };

    let tx = Transaction::Commit { commitment, spam_nonce };
    
    node.send_transaction(tx)
        .await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(CommitResponse {
        commitment: hex::encode(commitment),
        salt: hex::encode(salt),
        status: "committed".to_string(),
    }))
}

pub async fn send_transaction(
    State(node): State<AppState>,
    Json(req): Json<SendTransactionRequest>,
) -> Result<Json<SendTransactionResponse>, ErrorResponse> {
    if req.inputs.is_empty() {
        return Err(ErrorResponse { error: "Must provide at least one input".into() });
    }
    if req.signatures.len() != req.inputs.len() {
        return Err(ErrorResponse {
            error: "Signature count must match input count".into(),
        });
    }

    let inputs: Vec<InputReveal> = req.inputs.iter().map(|i| {
        Ok(InputReveal {
            owner_pk: parse_hex32(&i.owner_pk, "owner_pk")?,
            value: i.value,
            salt: parse_hex32(&i.salt, "input_salt")?,
        })
    }).collect::<Result<_, ErrorResponse>>()?;

    let mut signatures = Vec::new();
    for sig_hex in &req.signatures {
        let sig_bytes = hex::decode(sig_hex)
            .map_err(|e| ErrorResponse { error: format!("Invalid signature hex: {}", e) })?;
        signatures.push(sig_bytes);
    }

    let outputs: Vec<OutputData> = req.outputs.iter().map(|o| {
        Ok(OutputData {
            address: parse_hex32(&o.address, "address")?,
            value: o.value,
            salt: parse_hex32(&o.salt, "output_salt")?,
        })
    }).collect::<Result<_, ErrorResponse>>()?;

    let salt = parse_hex32(&req.salt, "salt")?;

    let input_coin_ids: Vec<String> = inputs.iter().map(|i| hex::encode(i.coin_id())).collect();
    let output_coin_ids: Vec<String> = outputs.iter().map(|o| hex::encode(o.coin_id())).collect();
    let fee = {
        let in_sum: u64 = inputs.iter().map(|i| i.value).sum();
        let out_sum: u64 = outputs.iter().map(|o| o.value).sum();
        in_sum.saturating_sub(out_sum)
    };

    let tx = Transaction::Reveal { inputs, signatures, outputs, salt };

    node.send_transaction(tx)
        .await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(SendTransactionResponse {
        input_coins: input_coin_ids,
        output_coins: output_coin_ids,
        fee,
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
            Transaction::Commit { commitment, .. } => TransactionInfo  {
                commitment: Some(hex::encode(commitment)),
                input_coins: None,
                output_coins: None,
                fee: None,
            },
            Transaction::Reveal { inputs, outputs, .. } => TransactionInfo {
                commitment: None,
                input_coins: Some(inputs.iter().map(|i| hex::encode(i.coin_id())).collect()),
                output_coins: Some(outputs.iter().map(|o| hex::encode(o.coin_id())).collect()),
                fee: Some(tx.fee()),
            },
        })
        .collect();

    Json(GetMempoolResponse { size, transactions: tx_info })
}
pub async fn scan_addresses(
    State(node): State<AppState>,
    Json(req): Json<ScanRequest>,
) -> Result<Json<ScanResponse>, ErrorResponse> {
    let addresses: Vec<[u8; 32]> = req.addresses.iter()
        .map(|h| parse_hex32(h, "address"))
        .collect::<Result<_, _>>()?;

    let coins = node.scan_addresses(&addresses, req.start_height, req.end_height)
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(ScanResponse {
        coins: coins.into_iter().map(|c| ScanCoin {
            address: hex::encode(c.address),
            value: c.value,
            salt: hex::encode(c.salt),
            coin_id: hex::encode(c.coin_id),
            height: c.height,
        }).collect(),
    }))
}

pub async fn generate_key() -> Json<GenerateKeyResponse> {
    let seed: [u8; 32] = rand::random();
    let owner_pk = wots::keygen(&seed);
    let address = compute_address(&owner_pk);

    Json(GenerateKeyResponse {
        seed: hex::encode(seed),
        address: hex::encode(address),
    })
}

pub async fn get_peers(State(node): State<AppState>) -> Json<GetPeersResponse> {
    let peers = node.get_peers().await;
    Json(GetPeersResponse { peers })
}
