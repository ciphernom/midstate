use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use tower_http::cors::{CorsLayer, Any};
use axum::http::{Method, HeaderValue};

// ── Pool Configuration ─────────────────────────────────────────────────────

const NODE_RPC_URL: &str = "http://127.0.0.1:8545";
const NODE_TCP_PORT: &str = "127.0.0.1:9333"; // Light Protocol Port
// If no miner has shares yet, burn the block reward (or set to your dev fund)
const DEFAULT_POOL_ADDRESS: &str = "0000000000000000000000000000000000000000000000000000000000000000"; 

// Database Tables
const SCORES_TABLE: TableDefinition<&[u8; 32], i64> = TableDefinition::new("scores");

// ── Types ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    db: Arc<Database>,
    /// The target difficulty required to accept a pool share (e.g., 20 bits)
    share_target: [u8; 32],
    network_height: Arc<AtomicU64>,
    network_reward: Arc<AtomicU64>,
    network_target: Arc<std::sync::RwLock<[u8; 32]>>,
    /// Cache the block template so we don't hammer the node or cause template churn
    cached_template: Arc<RwLock<Option<(std::time::Instant, PoolTemplateResponse)>>>,
}

#[derive(Deserialize)]
struct SubmitShareRequest {
    batch: midstate::core::Batch,
    payout_address: String,
}

#[derive(Clone, Serialize)]
struct PoolTemplateResponse {
    mining_midstate: String,
    target: String,
    batch_template: serde_json::Value,
}

// ── TCP Light Protocol Client ──────────────────────────────────────────────

async fn get_template(client: &reqwest::Client, coinbase_json: Vec<serde_json::Value>) -> anyhow::Result<serde_json::Value> {
    let req = serde_json::json!({ "coinbase": coinbase_json });
    let res = client.post(&format!("{}/block_template", NODE_RPC_URL))
        .json(&req)
        .send().await?;
        
    if res.status().is_success() {
        Ok(res.json().await?)
    } else {
        anyhow::bail!("Node error: {:?}", res.text().await?)
    }
}

async fn submit_block(client: &reqwest::Client, batch: &midstate::core::Batch) -> anyhow::Result<()> {
    let res = client.post(&format!("{}/submit_batch", NODE_RPC_URL))
        .json(batch)
        .send().await?;
        
    if res.status().is_success() {
        Ok(())
    } else {
        anyhow::bail!("Node rejected block: {:?}", res.text().await?)
    }
}


// ── The Deficit Round-Robin Apportionment ──────────────────────────────────

/// Reads scores from the database and assigns the natural power-of-2 denominations
/// of the block reward to the miners with the highest scores.
fn build_coinbase_distribution(
    db: &Database,
    total_reward: u64,
) -> Vec<serde_json::Value> {
    let mut outputs = Vec::new();
    
    // 1. Load scores from DB
    let mut scores = HashMap::new();
    let mut total_positive_score = 0u128;
    
    if let Ok(read_txn) = db.begin_read() {
        if let Ok(table) = read_txn.open_table(SCORES_TABLE) {
            for iter in table.iter().unwrap() {
                let (addr_bytes, score) = iter.unwrap();
                let score_val = score.value();
                scores.insert(addr_bytes.value().clone(), score_val);
                if score_val > 0 {
                    total_positive_score += score_val as u128;
                }
            }
        }
    }

    if total_positive_score == 0 {
        // No one has done work. Burn it to the default address.
        for denom in midstate::core::types::decompose_value(total_reward) {
            outputs.push(serde_json::json!({
                "address": DEFAULT_POOL_ADDRESS,
                "value": denom,
                "salt": hex::encode(rand::random::<[u8; 32]>())
            }));
        }
        return outputs;
    }

    // 2. Decompose the reward naturally (e.g. 2^30 reward = exactly 1 coin)
    // plus any extra small coins from mempool transaction fees.
    let coins = midstate::core::types::decompose_value(total_reward);
    
    // 3. Greedily assign coins
    for coin in coins.into_iter().rev() { // Largest to smallest
        // Find miner with highest score
        let mut best_miner = [0u8; 32];
        let mut max_score = i64::MIN;
        
        for (addr, &score) in &scores {
            if score > max_score {
                max_score = score;
                best_miner = *addr;
            }
        }
        
        outputs.push(serde_json::json!({
            "address": hex::encode(best_miner),
            "value": coin,
            "salt": hex::encode(rand::random::<[u8; 32]>())
        }));

        // Simulate the deduction so they don't get ALL the small fee coins too
        let simulated_drop = ((coin as u128 * total_positive_score) / (total_reward as u128)) as i64;
        *scores.get_mut(&best_miner).unwrap() -= simulated_drop.max(1);
    }

    outputs
}

// ── Boot & Main Loop ───────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Force the logger to always show INFO level messages
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    tracing::info!("Starting Zero-Fee Trustless Smart Pool (Deficit Round-Robin)...");
    std::fs::create_dir_all("data").unwrap();
    let db = Arc::new(Database::create("data/smart_pool.redb").expect("Failed to open database"));
    let write_txn = db.begin_write().unwrap();
    let _ = write_txn.open_table(SCORES_TABLE).unwrap();
    write_txn.commit().unwrap();

    // Set share difficulty to 20 leading zero bits (~1 M hashes)
    let mut share_target = [0xff; 32];
    share_target[0] = 0x00;
    share_target[1] = 0x0f; 

    let state = Arc::new(AppState {
        db: db.clone(),
        share_target,
        network_height: Arc::new(AtomicU64::new(0)),
        network_reward: Arc::new(AtomicU64::new(0)),
        network_target: Arc::new(std::sync::RwLock::new([0u8; 32])),
        cached_template: Arc::new(RwLock::new(None)),
    });

    // Background Task: Poll Node State
    let bg_state = Arc::clone(&state);
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            if let Ok(res) = client.get(&format!("{}/state", NODE_RPC_URL)).send().await {
                if let Ok(json) = res.json::<serde_json::Value>().await {
                    bg_state.network_height.store(json["height"].as_u64().unwrap_or(0), Ordering::Relaxed);
                    bg_state.network_reward.store(json["block_reward"].as_u64().unwrap_or(0), Ordering::Relaxed);
                    
                    if let Some(t_hex) = json["target"].as_str() {
                        if let Ok(t_bytes) = hex::decode(t_hex) {
                            let mut arr = [0u8; 32];
                            arr.copy_from_slice(&t_bytes);
                            *bg_state.network_target.write().unwrap() = arr;
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });

    let allowed_origins = [
        "http://localhost:8080".parse::<HeaderValue>().unwrap(),
        "https://ciphernom.github.io".parse::<HeaderValue>().unwrap(), 
        "https://cypherpunk.gold".parse::<HeaderValue>().unwrap(), 
        "https://www.cypherpunk.gold".parse::<HeaderValue>().unwrap(), 
    ];
    
    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/template", get(handle_get_template))
        .route("/api/submit", post(handle_submit))
        .with_state(state);

    // Defensively try ports starting from 8080 up to 8090
    let mut port = 8080;
    let listener = loop {
        match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
            Ok(l) => {
                tracing::info!("✅ Smart Pool successfully bound to 0.0.0.0:{}", port);
                break l;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                tracing::warn!("Port {} is already in use. Trying next port...", port);
                port += 1;
                if port > 8090 {
                    panic!("CRITICAL: Could not find any available ports in the range 8080-8090!");
                }
            }
            Err(e) => panic!("CRITICAL: Failed to bind to network interface: {}", e),
        }
    };

    axum::serve(listener, app).await.unwrap();
}

// ── HTTP Handlers ──────────────────────────────────────────────────────────

/// Miners call this to get the block template. 
/// It caches the template for 5 seconds to completely eliminate template churn.
async fn handle_get_template(
    State(state): State<Arc<AppState>>,
) -> Result<Json<PoolTemplateResponse>, (StatusCode, String)> {
    let reward = state.network_reward.load(Ordering::Relaxed);
    if reward == 0 {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Node not synced yet".into()));
    }

    // 1. Check cache to prevent churn
    {
        let cache = state.cached_template.read().await;
        if let Some((timestamp, template)) = &*cache {
            if timestamp.elapsed().as_secs() < 5 {
                return Ok(Json(template.clone()));
            }
        }
    }

    // 2. Cache expired. Calculate new perfectly sized distribution based on scores.
    let coinbase_json = build_coinbase_distribution(&state.db, reward);

    // Ask the node for a block template using the perfectly split coinbase
    let client = reqwest::Client::new();
    match get_template(&client, coinbase_json).await {
        Ok(template_data) => {
            let resp = PoolTemplateResponse {
                mining_midstate: template_data["mining_midstate"].as_str().unwrap().to_string(),
                target: template_data["target"].as_str().unwrap().to_string(),
                batch_template: template_data["batch_template"].clone(),
            };
            
            let mut cache = state.cached_template.write().await;
            *cache = Some((std::time::Instant::now(), resp.clone()));
            
            Ok(Json(resp))
        },
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Node error: {}", e))),
    }
}

/// Miners submit their PoW here.
async fn handle_submit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitShareRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    
    let batch = req.batch;
    let header = batch.header();
    let mining_hash = midstate::core::types::compute_header_hash(&header);

    // 1. Parse Miner Address
    let miner_addr_bytes = hex::decode(&req.payout_address)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid payout address hex".into()))?;
    if miner_addr_bytes.len() != 32 {
        return Err((StatusCode::BAD_REQUEST, "Payout address must be 32 bytes".into()));
    }
    let mut miner_addr = [0u8; 32];
    miner_addr.copy_from_slice(&miner_addr_bytes);

    // 2. Verify Pool Share PoW
    if midstate::core::extension::verify_extension(mining_hash, &batch.extension, &state.share_target).is_err() {
        return Err((StatusCode::BAD_REQUEST, "Insufficient Proof of Work for pool share".into()));
    }

    // 3. Add to Score! (1 Share = 1,000,000 points for precision)
    {
        let write_txn = state.db.begin_write().unwrap();
        {
            let mut table = write_txn.open_table(SCORES_TABLE).unwrap();
            let current = table.get(&miner_addr).unwrap().map(|v| v.value()).unwrap_or(0);
            table.insert(&miner_addr, current + 1_000_000).unwrap();
        }
        write_txn.commit().unwrap();
    }

    // 4. Check if it's a global block winner
    let actual_target = *state.network_target.read().unwrap();
    let is_full_block = midstate::core::extension::verify_extension(mining_hash, &batch.extension, &actual_target).is_ok();

    if is_full_block {
        tracing::info!("🥇 FULL BLOCK FOUND by miner {}!", &req.payout_address[..8]);
        
        let db_clone = state.db.clone();
        
        // Use the new helper
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            if let Ok(_) = submit_block(&client, &batch).await {
                tracing::info!("Block successfully accepted by network! Applying score deductions...");
                
                // Block was accepted! Deduct scores to balance the scales.
                let mut total_positive_score = 0u128;
                let total_reward: u64 = batch.coinbase.iter().map(|cb| cb.value).sum();

                if let Ok(write_txn) = db_clone.begin_write() {
                    {
                        let mut table = write_txn.open_table(SCORES_TABLE).unwrap();
                        
                        // Sum positive scores
                        for iter in table.iter().unwrap() {
                            let (_, score) = iter.unwrap();
                            let v = score.value();
                            if v > 0 { total_positive_score += v as u128; }
                        }

                        // Deduct proportionally
                        for cb in &batch.coinbase {
                            let mut addr = [0u8; 32];
                            addr.copy_from_slice(&cb.address);
                            
                            let deduction = ((cb.value as u128 * total_positive_score) / (total_reward as u128)) as i64;
                            let current = table.get(&addr).unwrap().map(|v| v.value()).unwrap_or(0);
                            
                            table.insert(&addr, current - deduction).unwrap();
                            tracing::info!("Paid {} units to {}. Score deduction: {}", cb.value, hex::encode(&addr[..8]), deduction);
                        }
                    }
                    
                    write_txn.commit().unwrap();
                }
            } else {
                tracing::error!("Node rejected the block (Orphan or Invalid). Scores not deducted.");
            }
        });

        return Ok(Json(serde_json::json!({ "status": "success", "message": "FULL BLOCK MINED!" })));
    }

    Ok(Json(serde_json::json!({ "status": "success", "message": "Share accepted" })))
}
