use super::types::*;
use crate::core::{compute_commitment, compute_address, hash_concat, wots,
                  block_reward, Transaction, InputReveal, OutputData, Predicate, Witness};
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
    
    // --- Rate Limit Check using Semaphore on NodeHandle ---
    let _permit = node.commit_limiter.try_acquire().map_err(|_| ErrorResponse {
        error: "Server is under heavy load computing PoW. Try again later.".into()
    })?;

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

    // Determine dynamic PoW requirement based on mempool congestion
    let (_, txs) = node.get_mempool_info().await;
    let pending_commits = txs.iter().filter(|t| matches!(t, Transaction::Commit { .. })).count();
    
    // Use the dynamic threshold from Mempool
    let required_pow = crate::mempool::Mempool::calculate_required_pow(pending_commits);

    // Mine PoW nonce for anti-spam
    let spam_nonce = tokio::task::spawn_blocking(move || {
        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if crate::core::types::count_leading_zeros(&h) >= required_pow {
                return n;
            }
            n += 1;
        }
    }).await.map_err(|_| ErrorResponse { error: "PoW task failed".into() })?;

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
            predicate: Predicate::Script { 
                bytecode: hex::decode(&i.bytecode).map_err(|e| ErrorResponse { error: format!("Invalid bytecode hex: {}", e) })? 
            },
            value: i.value,
            salt: parse_hex32(&i.salt, "input_salt")?,
        })
    }).collect::<Result<_, ErrorResponse>>()?;

    let mut witnesses = Vec::new();
    for sig_string in &req.signatures {
        let stack_items = sig_string.split(',')
            .filter(|s| !s.is_empty())
            .map(hex::decode)
            .collect::<Result<Vec<Vec<u8>>, _>>()
            .map_err(|e| ErrorResponse { error: format!("Invalid hex in witness stack: {}", e) })?;
        witnesses.push(Witness::ScriptInputs(stack_items));
    }

    let outputs: Vec<OutputData> = req.outputs.iter().map(|o| {
        match o {
            OutputDataJson::Standard { address, value, salt } => {
                Ok(OutputData::Standard {
                    address: parse_hex32(address, "address")?,
                    value: *value,
                    salt: parse_hex32(salt, "output_salt")?,
                })
            }
            OutputDataJson::DataBurn { payload, value_burned } => {
                Ok(OutputData::DataBurn {
                    payload: hex::decode(payload).map_err(|_| ErrorResponse { error: "Invalid payload hex".into() })?,
                    value_burned: *value_burned,
                })
            }
        }
    }).collect::<Result<_, ErrorResponse>>()?;

    let salt = parse_hex32(&req.salt, "salt")?;

    let input_coin_ids: Vec<String> = inputs.iter().map(|i| hex::encode(i.coin_id())).collect();
    let output_commit_hashes: Vec<String> = outputs.iter().map(|o| hex::encode(o.hash_for_commitment())).collect();
    
    let fee = {
        let in_sum: u64 = inputs.iter().map(|i| i.value).sum();
        let out_sum: u64 = outputs.iter().map(|o| o.value()).sum();
        in_sum.saturating_sub(out_sum)
    };

    let tx = Transaction::Reveal { inputs, witnesses, outputs, salt };

    node.send_transaction(tx)
        .await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(SendTransactionResponse {
        input_coins: input_coin_ids,
        output_coins: output_commit_hashes,
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
                output_coins: Some(outputs.iter().filter_map(|o| o.coin_id().map(hex::encode)).collect()),                
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

// ── CoinJoin Mix Handlers ───────────────────────────────────────────────

pub async fn mix_create(
    State(node): State<AppState>,
    Json(req): Json<MixCreateRequest>,
) -> Result<Json<MixCreateResponse>, ErrorResponse> {
    let mix_id = node.mix_create(req.denomination, req.min_participants).await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(MixCreateResponse {
        mix_id: hex::encode(mix_id),
        denomination: req.denomination,
        status: "collecting".to_string(),
    }))
}

pub async fn mix_register(
    State(node): State<AppState>,
    Json(req): Json<MixRegisterRequest>,
) -> Result<Json<serde_json::Value>, ErrorResponse> {
    let mix_id = parse_hex32(&req.mix_id, "mix_id")?;

let input = InputReveal {
        predicate: Predicate::Script { 
            bytecode: hex::decode(&req.input.bytecode).map_err(|e| ErrorResponse { error: format!("Invalid bytecode hex: {}", e) })? 
        },
        value: req.input.value,
        salt: parse_hex32(&req.input.salt, "input_salt")?,
    };
    let output = match req.output {
        OutputDataJson::Standard { address, value, salt } => OutputData::Standard {
            address: parse_hex32(&address, "address")?,
            value,
            salt: parse_hex32(&salt, "output_salt")?,
        },
        OutputDataJson::DataBurn { .. } => {
            return Err(ErrorResponse { error: "DataBurn payloads are not allowed in CoinJoin mixes".into() })
        }
    };

    // --- Validate coin exists in UTXO set before touching MixManager ---
    let coin_id = input.coin_id();
    {
        let state = node.get_state().await;
        if !state.coins.contains(&coin_id) {
            return Err(ErrorResponse {
                error: "Input coin does not exist or is already spent".into(),
            });
        }
    }
    // -------------------------------------------------------------------

    // --- Decode the signature from the RPC request ---
    let signature = hex::decode(&req.signature)
        .map_err(|e| ErrorResponse { error: format!("Invalid signature hex: {}", e) })?;
    // ------------------------------------------------------

    // Pass the signature into the node handle
    node.mix_register(mix_id, input, output, signature).await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

        Ok(Json(serde_json::json!({ "status": "registered" })))
}

pub async fn mix_fee(
    State(node): State<AppState>,
    Json(req): Json<MixFeeRequest>,
) -> Result<Json<serde_json::Value>, ErrorResponse> {
    let mix_id = parse_hex32(&req.mix_id, "mix_id")?;

let input = InputReveal {
        predicate: Predicate::Script { 
            bytecode: hex::decode(&req.input.bytecode).map_err(|e| ErrorResponse { error: format!("Invalid bytecode hex: {}", e) })? 
        },
        value: req.input.value,
        salt: parse_hex32(&req.input.salt, "input_salt")?,
    };
    // --- Validate fee coin exists in UTXO set before touching MixManager ---
    {
        let coin_id = input.coin_id();
        let state = node.get_state().await;
        if !state.coins.contains(&coin_id) {
            return Err(ErrorResponse {
                error: "Fee input coin does not exist or is already spent".into(),
            });
        }
    }
    // ---------------------------------------------------------------------

    node.mix_set_fee(mix_id, input).await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(serde_json::json!({ "status": "fee_set" })))
}

pub async fn mix_sign(
    State(node): State<AppState>,
    Json(req): Json<MixSignRequest>,
) -> Result<Json<serde_json::Value>, ErrorResponse> {
    let mix_id = parse_hex32(&req.mix_id, "mix_id")?;
    let coin_id = parse_hex32(&req.coin_id, "coin_id")?;

    let input_index = node.mix_find_input_index(mix_id, coin_id).await
        .ok_or_else(|| ErrorResponse {
            error: format!("coin {} not found in mix proposal", req.coin_id),
        })?;

    let signature = hex::decode(&req.signature)
        .map_err(|e| ErrorResponse { error: format!("invalid signature hex: {}", e) })?;

    let height = node.get_state().await.height;
    node.mix_sign(mix_id, input_index, signature, height).await
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    Ok(Json(serde_json::json!({ "status": "signed", "input_index": input_index })))
}

pub async fn mix_status(
    State(node): State<AppState>,
    axum::extract::Path(mix_id_hex): axum::extract::Path<String>,
) -> Result<Json<MixStatusResponse>, ErrorResponse> {
    let mix_id = parse_hex32(&mix_id_hex, "mix_id")?;

    let snapshot = node.mix_status(mix_id).await
        .ok_or_else(|| ErrorResponse { error: "mix session not found".to_string() })?;

    Ok(Json(snapshot_to_response(snapshot)))
}

pub async fn mix_list(
    State(node): State<AppState>,
) -> Json<MixListResponse> {
    let sessions = node.mix_list().await;
    Json(MixListResponse {
        sessions: sessions.into_iter().map(snapshot_to_response).collect(),
    })
}

pub async fn get_filters(
    State(node): State<AppState>,
    Json(req): Json<GetFiltersRequest>,
) -> Result<Json<GetFiltersResponse>, ErrorResponse> {
    // Limit to 1000 filters per request to prevent abuse
    let count = (req.end_height.saturating_sub(req.start_height)).min(1000);
    let end = req.start_height + count;

    let store = crate::storage::BatchStore::new(&node.batches_path)
        .map_err(|e| ErrorResponse { error: format!("Storage error: {}", e) })?;

    let mut filters = Vec::new();
    let mut block_hashes = Vec::new();
    let mut element_counts = Vec::new();

    for h in req.start_height..end {
        let filter_data = match store.load_filter(h) {
            Ok(Some(data)) => data,
            _ => break,
        };

        // Load batch to get the block hash (final_hash) and element count.
        // Needed for client-side Golomb-Rice filter matching.
        let batch = match store.load(h) {
            Ok(Some(b)) => b,
            _ => break,
        };

        // Count unique identifiable elements the same way CompactFilter::build does
        let mut items = std::collections::HashSet::new();
        for tx in &batch.transactions {
            match tx {
                crate::core::Transaction::Commit { commitment, .. } => {
                    items.insert(*commitment);
                }
                crate::core::Transaction::Reveal { inputs, outputs, .. } => {
                    for input in inputs {
                        items.insert(input.coin_id());
                        items.insert(input.predicate.address());
                    }
                    for output in outputs {
                        if let Some(cid) = output.coin_id() { items.insert(cid); }
                        items.insert(output.address());
                    }
                }
            }
        }
        for cb in &batch.coinbase {
            items.insert(cb.coin_id());
            items.insert(cb.address);
        }

        filters.push(hex::encode(filter_data));
        block_hashes.push(hex::encode(batch.extension.final_hash));
        element_counts.push(items.len() as u64);
    }

    Ok(Json(GetFiltersResponse {
        start_height: req.start_height,
        filters,
        block_hashes,
        element_counts,
    }))
}

fn snapshot_to_response(s: crate::mix::MixStatusSnapshot) -> MixStatusResponse {
    let phase_str = match &s.phase {
        crate::mix::MixPhase::Collecting => "collecting",
        crate::mix::MixPhase::Signing => "signing",
        crate::mix::MixPhase::CommitSubmitted => "commit_submitted",
        crate::mix::MixPhase::Complete => "complete",
        crate::mix::MixPhase::Failed(_) => "failed",
    };
    MixStatusResponse {
        mix_id: s.mix_id,
        denomination: s.denomination,
        participants: s.participants,
        phase: phase_str.to_string(),
        commitment: s.commitment,
        input_coin_ids: s.input_coin_ids,
    }
}

// ── Explorer Endpoints ──────────────────────────────────────────────────

/// Return a batch with witnesses stripped (they're ~40KB each and useless
/// for display). Keeps the explorer snappy.
pub async fn get_batch(
    State(node): State<AppState>,
    axum::extract::Path(height): axum::extract::Path<u64>,
) -> Result<Json<serde_json::Value>, ErrorResponse> {
    let store = crate::storage::BatchStore::new(&node.batches_path)
        .map_err(|e| ErrorResponse { error: format!("Storage error: {}", e) })?;

    let batch = store.load(height)
        .map_err(|e| ErrorResponse { error: e.to_string() })?
        .ok_or_else(|| ErrorResponse { error: format!("Batch at height {} not found", height) })?;

    // Serialize to JSON, then strip the witness arrays to save bandwidth
    let mut val = serde_json::to_value(&batch)
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    // Strip witnesses from each Reveal transaction
    if let Some(txs) = val.get_mut("transactions").and_then(|v| v.as_array_mut()) {
        for tx in txs {
            if let Some(reveal) = tx.get_mut("Reveal") {
                if let Some(witnesses) = reveal.get_mut("witnesses") {
                    // Replace with just the count so the UI knows how many inputs
                    let count = witnesses.as_array().map(|a| a.len()).unwrap_or(0);
                    *witnesses = serde_json::json!(format!("{} witness(es) stripped", count));
                }
            }
        }
    }

    // Inject height into the response
    val.as_object_mut().map(|o| o.insert("height".to_string(), serde_json::json!(height)));

    Ok(Json(val))
}

/// Universal search: find any 32-byte hash across blocks, txs, addresses.
/// Searches the last `limit` blocks (default 1000) server-side.
pub async fn search(
    State(node): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ErrorResponse> {
    let query = parse_hex32(&req.query, "query")?;

    let store = crate::storage::BatchStore::new(&node.batches_path)
        .map_err(|e| ErrorResponse { error: format!("Storage error: {}", e) })?;

    let tip = store.highest()
        .map_err(|e| ErrorResponse { error: e.to_string() })?;

    let search_start = tip.saturating_sub(5000);
    let mut results = Vec::new();

    for height in (search_start..=tip).rev() {
        let batch = match store.load(height) {
            Ok(Some(b)) => b,
            _ => continue,
        };

        // Check block-level hashes
        if batch.extension.final_hash == query {
            results.push(SearchResult {
                result_type: "block_hash".into(),
                height,
                tx_index: None,
                detail: "Block hash match".into(),
            });
        }
        if batch.prev_midstate == query {
            results.push(SearchResult {
                result_type: "prev_midstate".into(),
                height,
                tx_index: None,
                detail: "Parent midstate match".into(),
            });
        }
        if batch.state_root == query {
            results.push(SearchResult {
                result_type: "state_root".into(),
                height,
                tx_index: None,
                detail: "State root match".into(),
            });
        }

        // Check coinbase
        for cb in &batch.coinbase {
            if cb.address == query {
                results.push(SearchResult {
                    result_type: "coinbase_address".into(),
                    height,
                    tx_index: None,
                    detail: format!("Coinbase output, value {}", cb.value),
                });
            }
            if cb.coin_id() == query {
                results.push(SearchResult {
                    result_type: "coinbase_coin_id".into(),
                    height,
                    tx_index: None,
                    detail: format!("Coinbase coin, value {}", cb.value),
                });
            }
        }

        // Check transactions
        for (i, tx) in batch.transactions.iter().enumerate() {
            match tx {
                Transaction::Commit { commitment, .. } => {
                    if *commitment == query {
                        results.push(SearchResult {
                            result_type: "commitment".into(),
                            height,
                            tx_index: Some(i),
                            detail: "Commitment hash match".into(),
                        });
                    }
                }
                Transaction::Reveal { inputs, outputs, salt, .. } => {
                    if *salt == query {
                        results.push(SearchResult {
                            result_type: "reveal_salt".into(),
                            height,
                            tx_index: Some(i),
                            detail: "Reveal commitment salt".into(),
                        });
                    }
                    for (j, inp) in inputs.iter().enumerate() {
                        if inp.predicate.address() == query {
                            results.push(SearchResult {
                                result_type: "input_address".into(),
                                height,
                                tx_index: Some(i),
                                detail: format!("Input {} spent, value {}", j, inp.value),
                            });
                        }
                        if inp.coin_id() == query {
                            results.push(SearchResult {
                                result_type: "input_coin_id".into(),
                                height,
                                tx_index: Some(i),
                                detail: format!("Input {} coin spent, value {}", j, inp.value),
                            });
                        }
                    }
                    for (j, out) in outputs.iter().enumerate() {
                        if out.address() == query {
                            results.push(SearchResult {
                                result_type: "output_address".into(),
                                height,
                                tx_index: Some(i),
                                detail: format!("Output {} created, value {}", j, out.value()),
                            });
                        }
                        if let Some(cid) = out.coin_id() {
                            if cid == query {
                                results.push(SearchResult {
                                    result_type: "output_coin_id".into(),
                                    height,
                                    tx_index: Some(i),
                                    detail: format!("Output {} coin created, value {}", j, out.value()),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Stop after finding results (return all matches at first matching height,
        // then keep scanning for more)
        if results.len() >= 50 {
            break;
        }
    }

    Ok(Json(SearchResponse { results }))
}

/// Check whether a coin exists in the current UTXO set (GET version for explorer)
pub async fn check_coin_get(
    State(node): State<AppState>,
    axum::extract::Path(coin_hex): axum::extract::Path<String>,
) -> Result<Json<CheckCoinResponse>, ErrorResponse> {
    let coin = parse_hex32(&coin_hex, "coin")?;
    let exists = node.check_coin(coin).await;
    Ok(Json(CheckCoinResponse {
        exists,
        coin: hex::encode(coin),
    }))
}

/// Return a raw block for private wallet scanning.
/// Unlike /batch/:height, this returns the full Batch struct including
/// output addresses and salts but with witnesses stripped for bandwidth.
/// The wallet parses this client-side so it never has to send addresses
/// to the node — preserving Neutrino-level scan privacy.
pub async fn get_block_raw(
    State(node): State<AppState>,
    axum::extract::Path(height): axum::extract::Path<u64>,
) -> Result<Json<crate::core::Batch>, ErrorResponse> {
    let store = crate::storage::BatchStore::new(&node.batches_path)
        .map_err(|e| ErrorResponse { error: format!("Storage error: {}", e) })?;

    let mut batch = store.load(height)
        .map_err(|e| ErrorResponse { error: e.to_string() })?
        .ok_or_else(|| ErrorResponse { error: format!("Block at height {} not found", height) })?;

    // Strip witness data to save bandwidth (~40KB per input).
    // The wallet only needs inputs (for address) and outputs (for address, value, salt).
    for tx in &mut batch.transactions {
        if let Transaction::Reveal { witnesses, .. } = tx {
            *witnesses = vec![];
        }
    }

    Ok(Json(batch))
}

pub async fn explorer_ui() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("explorer.html"))
}
