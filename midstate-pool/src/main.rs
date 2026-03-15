use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use moka::future::Cache;
use redb::{Database, ReadableTable, TableDefinition}; // <-- Added ReadableTable
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;

use midstate::core::extension::verify_extension;
use midstate::core::types::{Batch, hash_concat}; // <-- Removed unused types
use midstate::core::mss::{self, MssKeypair};

// ── Database Schema ────────────────────────────────────────────────────────

const SHARES_TABLE: TableDefinition<&[u8; 32], u64> = TableDefinition::new("shares");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");

// ── API Payloads ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PoolInfoResponse {
    pool_address: String,
    share_target: String,
    pool_fee_percent: f64,
}

#[derive(Deserialize)]
struct SubmitShareRequest {
    batch: Batch,
    payout_address: String,
    /// Optional: height the miner believes they're mining at.
    /// If absent (current node clients don't send this), falls back
    /// to the server's known current height.
    #[serde(default)]
    height: Option<u64>,
}

#[derive(Serialize)]
struct SubmitShareResponse {
    status: String,
    message: String,
}

// ── Internal Types ─────────────────────────────────────────────────────────

/// A verified share sent from the HTTP handler to the DB writer thread
struct VerifiedShare {
    miner_address: [u8; 32],
    is_full_block: bool,
    batch: Option<Batch>, // Only populated if we need to broadcast a full block
}

struct AppState {
    /// The pool's reusable MSS keypair
    mss_keypair: std::sync::RwLock<MssKeypair>,
    /// Target required for a share
    share_target: [u8; 32],
    /// LRU Cache to prevent replay attacks (stores hash of shares seen in the last hour)
    seen_shares: Cache<[u8; 32], ()>,
    /// Channel to send verified shares to the DB writer thread
    db_tx: mpsc::Sender<VerifiedShare>,
    /// Shared atomic tracker for the current network height
    network_height: Arc<AtomicU64>,
    /// The real network mining target, fetched from the local node every 5s.
    /// Initialized to all-zeros (impossible target) so shares are rejected
    /// until the node has been contacted at least once.
    network_target: Arc<std::sync::RwLock<[u8; 32]>>,
}

// ── Boot & Main Loop ───────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // 1. Open or create the persistent database
    std::fs::create_dir_all("data").unwrap();
    let db = Arc::new(Database::create("data/pool.redb").expect("Failed to open database"));

    // Ensure tables exist
    let write_txn = db.begin_write().unwrap();
    {
        let _ = write_txn.open_table(SHARES_TABLE).unwrap();
        let _ = write_txn.open_table(METADATA_TABLE).unwrap();
    }
    write_txn.commit().unwrap();

    // 2. Load or Generate the Pool's MSS Keypair
    let mss_keypair = load_or_generate_mss_key(&db);
    tracing::info!("Pool Address (MSS): {}", hex::encode(mss_keypair.master_pk));

    // 3. Define the pool share difficulty
    // e.g., 20 leading zero bits for shares.
    let mut share_target = [0xff; 32];
    share_target[0] = 0x00;
    share_target[1] = 0x0f; 

    // 4. Setup Lock-Free MPSC Channel for Database Writes
    // Bounded to 100,000 pending shares to prevent OOM under extreme load/DDoS
    let (db_tx, db_rx) = mpsc::channel::<VerifiedShare>(100_000);
    
    // Spawn the robust DB Writer Task
    let db_clone = Arc::clone(&db);
    tokio::spawn(async move {
        database_writer_task(db_clone, db_rx).await;
    });

    let network_height = Arc::new(AtomicU64::new(0));
    let network_target = Arc::new(std::sync::RwLock::new([0u8; 32])); // all-zeros = reject all until synced
    
    // 5. Setup Chain Syncer (polls local full node)
    let height_clone = Arc::clone(&network_height);
    let target_clone = Arc::clone(&network_target);
    let share_target_check = share_target;
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut warned_target = false;
        loop {
            if let Ok(res) = client.get("http://127.0.0.1:8545/state").send().await {
                if let Ok(json) = res.json::<serde_json::Value>().await {
                    if let Some(h) = json["height"].as_u64() {
                        height_clone.store(h, Ordering::Relaxed);
                    }
                    if let Some(t_hex) = json["target"].as_str() {
                        if let Ok(t_bytes) = hex::decode(t_hex) {
                            if t_bytes.len() == 32 {
                                let mut arr = [0u8; 32];
                                arr.copy_from_slice(&t_bytes);
                                // Sanity check: share_target must be EASIER (larger) than network target.
                                // If not, every share would need to be a full block, defeating the purpose.
                                if !warned_target && arr > share_target_check {
                                    tracing::error!(
                                        "MISCONFIGURED: share_target ({}) is HARDER than network target ({}). No shares will be accepted!",
                                        hex::encode(share_target_check), hex::encode(arr)
                                    );
                                    warned_target = true;
                                }
                                *target_clone.write().unwrap() = arr;
                            }
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    // 6. Build App State
    let state = Arc::new(AppState {
        mss_keypair: std::sync::RwLock::new(mss_keypair),
        share_target,
        seen_shares: Cache::builder()
            .max_capacity(1_000_000)
            .time_to_live(Duration::from_secs(3600))
            .build(),
        db_tx,
        network_height,
        network_target,  // <-- add this
    });

    // 7. Start HTTP Server
    let app = Router::new()
        .route("/api/info", get(handle_info))
        .route("/api/submit", post(handle_submit))
        .with_state(state);

    tracing::info!("Pool Server running on 0.0.0.0:8080");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ── Key Management ─────────────────────────────────────────────────────────

fn load_or_generate_mss_key(db: &Database) -> MssKeypair {
    let read_txn = db.begin_read().unwrap();
    let table = read_txn.open_table(METADATA_TABLE).unwrap();
    
    if let Some(bytes) = table.get("mss_keypair").unwrap() {
        bincode::deserialize(bytes.value()).expect("Corrupted pool MSS key")
    } else {
        drop(table);
        drop(read_txn);
        
        tracing::warn!("No pool key found. Generating new MSS tree (Height 20). This may take a minute...");
        let seed: [u8; 32] = rand::random();
        let keypair = mss::keygen(&seed, 20).expect("MSS Keygen failed");
        
        let write_txn = db.begin_write().unwrap();
        {
            let mut w_table = write_txn.open_table(METADATA_TABLE).unwrap();
            w_table.insert("mss_keypair", bincode::serialize(&keypair).unwrap().as_slice()).unwrap();
        }
        write_txn.commit().unwrap();
        
        keypair
    }
}

// ── HTTP Handlers ──────────────────────────────────────────────────────────

async fn handle_info(State(state): State<Arc<AppState>>) -> Json<PoolInfoResponse> {
    let pk = state.mss_keypair.read().unwrap().master_pk;
    Json(PoolInfoResponse {
        pool_address: hex::encode(pk),
        share_target: hex::encode(state.share_target),
        pool_fee_percent: 1.0, // 1% fee
    })
}

async fn handle_submit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitShareRequest>,
) -> Result<Json<SubmitShareResponse>, (StatusCode, String)> {
    
    let batch = req.batch;
    let header = batch.header();

    // 1. Replay Protection (O(1) Memory Cache)
    // A share is uniquely identified by the parent midstate and the nonce.
    let share_id = hash_concat(&batch.prev_midstate, &batch.extension.nonce.to_le_bytes());
    if state.seen_shares.contains_key(&share_id) {
        return Err((StatusCode::BAD_REQUEST, "Duplicate share (Replay Attack)".into()));
    }

    // 2. Parse Miner Address
    let miner_addr_bytes = hex::decode(&req.payout_address)
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid payout address hex".into()))?;
    if miner_addr_bytes.len() != 32 {
        return Err((StatusCode::BAD_REQUEST, "Payout address must be 32 bytes".into()));
    }
    let mut miner_addr = [0u8; 32];
    miner_addr.copy_from_slice(&miner_addr_bytes);

    // 3. Stale Share Check
    let current_net_height = state.network_height.load(Ordering::Relaxed);
    let claimed_height = req.height.unwrap_or(current_net_height);
    if claimed_height < current_net_height.saturating_sub(2) || claimed_height > current_net_height + 2 {
        return Err((StatusCode::BAD_REQUEST, "Stale share (Chain moved on)".into()));
    }

    // 4. Proof of Work Check
    if verify_extension(header.post_tx_midstate, &batch.extension, &state.share_target).is_err() {
        return Err((StatusCode::BAD_REQUEST, "Insufficient Proof of Work for pool share".into()));
    }

    // 5. Cryptographic Watermark Verification
    let pool_pk = state.mss_keypair.read().unwrap().master_pk;
    let mut verified = false;

    for (i, cb) in batch.coinbase.iter().enumerate() {
        if cb.address == pool_pk {
            // Verify the salt correctly embeds the miner's identity
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"pool_share");
            hasher.update(&miner_addr);
            hasher.update(&claimed_height.to_le_bytes());
            hasher.update(&(i as u64).to_le_bytes());
            
            if cb.salt == *hasher.finalize().as_bytes() {
                verified = true;
                break;
            }
        }
    }

    if !verified {
        return Err((StatusCode::BAD_REQUEST, "Invalid Coinbase watermark or Pool Address".into()));
    }

    // 6. Check if it's a global block winner using the REAL network target.
    let actual_target = *state.network_target.read().unwrap();
    if actual_target == [0u8; 32] {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Pool not yet synced with node".into()));
    }
    let is_full_block = verify_extension(
        header.post_tx_midstate,
        &batch.extension,
        &actual_target,
    ).is_ok();

    // 7. Accept Share — queue to DB worker first, then mark as seen.
    // Order matters: if try_send fails, we must NOT insert to seen_shares,
    // otherwise the share is permanently replay-blocked but never credited.
    let msg = VerifiedShare {
        miner_address: miner_addr,
        is_full_block,
        batch: if is_full_block { Some(batch) } else { None },
    };

    if state.db_tx.try_send(msg).is_err() {
        tracing::error!("Pool is overloaded! Dropping share.");
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Pool overloaded".into()));
    }

    // Insert AFTER successful queue — share is now guaranteed to be credited.
    state.seen_shares.insert(share_id, ()).await;

    Ok(Json(SubmitShareResponse {
        status: "success".into(),
        message: if is_full_block { "FULL BLOCK FOUND!".into() } else { "Share accepted".into() },
    }))
}

// ── Database Worker Task ───────────────────────────────────────────────────

/// Runs infinitely in the background. Receives fully verified shares and writes
/// them to the ACID database sequentially, maximizing IO throughput and preventing DB locks.
async fn database_writer_task(db: Arc<Database>, mut rx: mpsc::Receiver<VerifiedShare>) {
    let client = reqwest::Client::new();
    
    // We batch DB writes to maximize disk throughput
    let mut batch_buffer = Vec::new();

    loop {
        // Collect shares available in the channel (up to 1000 per DB transaction)
        let mut got_messages = false;
        while let Ok(msg) = rx.try_recv() {
            batch_buffer.push(msg);
            got_messages = true;
            if batch_buffer.len() >= 1000 { break; }
        }

        if !got_messages {
            // Block until we get at least one message
            if let Some(msg) = rx.recv().await {
                batch_buffer.push(msg);
            } else {
                break; // Channel closed, server shutting down
            }
        }

        // ACID Commit all shares in this buffer
        if let Ok(write_txn) = db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(SHARES_TABLE) {
                for share in &batch_buffer {
                    let current_shares = table.get(&share.miner_address)
                        .unwrap()
                        .map(|v| v.value())
                        .unwrap_or(0);
                    
                    table.insert(&share.miner_address, current_shares + 1).unwrap();

                    if share.is_full_block {
                        tracing::info!("🥇 Committing FULL BLOCK win by miner {}!", hex::encode(&share.miner_address[..8]));
                        
                        //we immediately forward this to the node
                        if let Some(ref b) = share.batch {
                            let _ = client.post("http://127.0.0.1:8545/api/internal/submit_batch")
                                .json(b)
                                .send()
                                .await;
                        }
                    }
                }
            }
            if let Err(e) = write_txn.commit() {
                tracing::error!("CRITICAL: Failed to commit shares to database: {}", e);
            }
        }

        batch_buffer.clear();
    }
}
