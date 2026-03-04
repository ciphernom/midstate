use anyhow::{Context, Result};
use midstate::core::types::{decompose_value, InputReveal, OutputData, Predicate}; // <-- Removed unused types
use midstate::core::mss::MssKeypair;
use midstate::rpc::{CommitRequest, CommitResponse, InputRevealJson, OutputDataJson, SendTransactionRequest};
use redb::{Database, ReadableTable, TableDefinition}; // <-- Added ReadableTable
use std::collections::HashMap;
use std::time::Duration;

const SHARES_TABLE: TableDefinition<&[u8; 32], u64> = TableDefinition::new("shares");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");

const POOL_FEE_PERCENT: f64 = 0.01; // 1% pool fee
const NODE_RPC_URL: &str = "http://127.0.0.1:8545";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    tracing::info!("Starting Midstate Pool Payout Engine...");

    let db = Database::create("data/pool.redb").context("Failed to open database")?;
    
    // Guard: if a previous run completed the send but crashed before wiping
    // shares, finish the cleanup now rather than paying miners twice.
    {
        let read_txn = db.begin_read()?;
        let meta = read_txn.open_table(METADATA_TABLE)?;
        let already_sent = meta.get("payout_send_complete")?
            .map(|v| v.value() == b"1".as_slice())
            .unwrap_or(false);
        drop(meta);
        drop(read_txn);

        if already_sent {
            tracing::warn!("Prior payout send was completed but shares were not wiped. Finishing cleanup.");
            let write_txn = db.begin_write()?;
            {
                let mut table = write_txn.open_table(SHARES_TABLE)?;
                while table.pop_last()?.is_some() {}
                let mut meta_w = write_txn.open_table(METADATA_TABLE)?;
                meta_w.remove("payout_send_complete")?;
            }
            write_txn.commit()?;
            tracing::info!("Cleanup complete. Exiting.");
            return Ok(());
        }
    }
    
    let client = reqwest::Client::new();

    // 1. Load the Pool's MSS Keypair
    let mut mss_keypair = load_mss_key(&db)?;
    let pool_address = mss_keypair.master_pk;
    tracing::info!("Pool Address: {}", hex::encode(pool_address));

    // 2. Scan the local node for the pool's available UTXOs
    // Load the last successfully paid-out scan height from DB.
    let last_scan_height: u64 = {
        let read_txn = db.begin_read()?;
        let meta = read_txn.open_table(METADATA_TABLE)?;
        meta.get("last_scan_height")?
            .and_then(|v| v.value().try_into().ok())
            .map(u64::from_le_bytes)
            .unwrap_or(0)
    };

    tracing::info!("Scanning node for pool UTXOs from height {}...", last_scan_height);
    let scan_req = midstate::rpc::ScanRequest {
        addresses: vec![hex::encode(pool_address)],
        start_height: last_scan_height,
        end_height: 999_999_999,
    };

    let res: midstate::rpc::ScanResponse = client
        .post(&format!("{}/scan", NODE_RPC_URL))
        .json(&scan_req)
        .send().await?.json().await?;

    if res.coins.is_empty() {
        tracing::info!("Pool has no balance since height {}. Exiting.", last_scan_height);
        return Ok(());
    }

let mut total_balance = 0u64;
    let mut inputs_to_spend = Vec::new();

    for c in &res.coins {
        let salt_bytes: [u8; 32] = hex::decode(&c.salt)?.try_into()
            .map_err(|_| anyhow::anyhow!("invalid salt length"))?;
        inputs_to_spend.push(InputReveal {
            predicate: Predicate::p2pk(&pool_address),
            value: c.value,
            salt: salt_bytes,
        });
        total_balance += c.value;
    }

    tracing::info!("Pool Balance: {} units across {} UTXOs", total_balance, inputs_to_spend.len());


    // 3. Tally Shares from the Database
    let read_txn = db.begin_read()?;
    let table = read_txn.open_table(SHARES_TABLE)?;
    let mut total_shares = 0u64;
    let mut miner_shares = HashMap::new();

    for iter in table.iter()? {
        let (addr, shares) = iter?;
        total_shares += shares.value();
        let mut addr_bytes = [0u8; 32];
        addr_bytes.copy_from_slice(addr.value());
        miner_shares.insert(addr_bytes, shares.value());
    }
    drop(table);
    drop(read_txn);

    if total_shares == 0 {
        tracing::info!("No shares submitted. Exiting.");
        return Ok(());
    }

    // 4. Calculate Payouts
    let pool_fee_cut = (total_balance as f64 * POOL_FEE_PERCENT) as u64;
    let distributable = total_balance.saturating_sub(pool_fee_cut);
    
    // Safety check: leave at least 1 unit to pay the network transaction fee
    let distributable = distributable.saturating_sub(1); 

    tracing::info!("Total Shares: {}", total_shares);
    tracing::info!("Distributing {} units (Pool fee: {})", distributable, pool_fee_cut);

    let mut outputs = Vec::new();

    // Generate miner outputs (decomposed into powers of 2)
    for (miner_addr, shares) in &miner_shares {
        let share_percent = *shares as f64 / total_shares as f64;
        let payout = (distributable as f64 * share_percent) as u64;

        if payout == 0 { continue; } // Dust

        for denom in decompose_value(payout) {
            outputs.push(OutputData::Standard {
                address: *miner_addr,
                value: denom,
                salt: rand::random(),
            });
        }
    }

    // Generate pool fee outputs (back to the pool's own address)
    for denom in decompose_value(pool_fee_cut) {
        outputs.push(OutputData::Standard {
            address: pool_address,
            value: denom,
            salt: rand::random(),
        });
    }

    // 5. Phase 1: Submit the COMMIT
    tracing::info!("Submitting Commit transaction...");
    let input_ids: Vec<String> = inputs_to_spend.iter().map(|i| hex::encode(i.coin_id())).collect();
    let dest_hashes: Vec<String> = outputs.iter().map(|o| hex::encode(o.hash_for_commitment())).collect();

    let commit_req = CommitRequest {
        coins: input_ids.clone(),
        destinations: dest_hashes,
    };

    let commit_res: CommitResponse = client.post(&format!("{}/commit", NODE_RPC_URL))
        .json(&commit_req)
        .send().await?.json().await?;
        
    let commitment = hex::decode(&commit_res.commitment)?.try_into().unwrap();
    
    tracing::info!("Commit accepted: {}. Waiting for it to be mined...", commit_res.commitment);

    // Poll until commit is mined
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let mp: serde_json::Value = client.get(&format!("{}/mempool", NODE_RPC_URL)).send().await?.json().await?;
        let is_pending = mp["transactions"].as_array().unwrap().iter().any(|tx| {
            tx.get("commitment").and_then(|c| c.as_str()) == Some(&commit_res.commitment)
        });
        
        if !is_pending {
            break; // It left the mempool, assuming it mined!
        }
        tracing::info!("Still waiting for commit to mine...");
    }

    // 6. Phase 2: Sign and REVEAL
    tracing::info!("Commit mined! Syncing MSS state from node before signing...");

    // SAFETY: Query the node for the highest leaf index it has seen, including
    // mempool. If it's ahead of our local state (prior crash mid-payout), we
    // fast-forward before signing, preventing any leaf from being reused.
    let mss_resp: midstate::rpc::GetMssStateResponse = client
        .post(&format!("{}/mss_state", NODE_RPC_URL))
        .json(&midstate::rpc::GetMssStateRequest {
            master_pk: hex::encode(pool_address),
        })
        .send()
        .await
        .context("CRITICAL: Cannot reach node to verify MSS state. Aborting to prevent key reuse.")?
        .json()
        .await
        .context("CRITICAL: Invalid response from /mss_state.")?;

    const SAFETY_MARGIN: u64 = 20;
    if mss_resp.next_index > mss_keypair.next_leaf {
        let safe_index = mss_resp.next_index + SAFETY_MARGIN;
        tracing::warn!(
            "MSS state mismatch: node={}, local={}. Fast-forwarding to {}.",
            mss_resp.next_index, mss_keypair.next_leaf, safe_index
        );
        mss_keypair.set_next_leaf(safe_index);
        save_mss_key(&db, &mss_keypair)?;
    }

    tracing::info!("MSS state verified. Generating {} signature(s)...", inputs_to_spend.len());
    let mut signatures = Vec::new();
    for _ in &inputs_to_spend {
        if mss_keypair.remaining() == 0 {
            anyhow::bail!("CRITICAL: MSS key capacity exhausted during payout!");
        }
        // Save the incremented leaf index to disk BEFORE signing.
        // If the process crashes between save and sign, the next run's
        // /mss_state check will see the gap and fast-forward past it.
        // This is strictly safer than saving after: crash-after-sign = reuse.
        mss_keypair.set_next_leaf(mss_keypair.next_leaf + 1);
        save_mss_key(&db, &mss_keypair)?;
        let sig = mss_keypair.sign(&commitment)?;
        signatures.push(hex::encode(sig.to_bytes()));
    }

    let reveal_req = SendTransactionRequest {
        inputs: inputs_to_spend.iter().map(|i| InputRevealJson {
            bytecode: hex::encode(match &i.predicate { Predicate::Script { bytecode } => bytecode }),
            value: i.value,
            salt: hex::encode(i.salt),
        }).collect(),
        signatures, // One signature per input
        outputs: outputs.iter().map(|o| OutputDataJson::Standard {
            address: hex::encode(o.address()),
            value: o.value(),
            salt: hex::encode(o.salt()),
        }).collect(),
        salt: commit_res.salt,
    };

    tracing::info!("Submitting Reveal transaction...");
    let reveal_res = client.post(&format!("{}/send", NODE_RPC_URL))
        .json(&reveal_req)
        .send().await?;

    if !reveal_res.status().is_success() {
        let err = reveal_res.text().await?;
        anyhow::bail!("Failed to submit Reveal: {}", err);
    }

    // 7. Atomically mark the send as complete BEFORE wiping shares.
    // If we crash between this write and the wipe below, the startup
    // guard above will finish the cleanup on the next invocation.
    {
        let write_txn = db.begin_write()?;
        {
            let mut meta = write_txn.open_table(METADATA_TABLE)?;
            meta.insert("payout_send_complete", b"1".as_slice())?;
        }
        write_txn.commit()?;
    }

    tracing::info!("Payout successful! Wiping shares table.");
    {
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(SHARES_TABLE)?;
            while table.pop_last()?.is_some() {}
            let mut meta = write_txn.open_table(METADATA_TABLE)?;
            meta.remove("payout_send_complete")?;
        }
        write_txn.commit()?;
    }
    
    // Persist the scan tip so the next payout only scans new blocks.
    let state_resp: midstate::rpc::GetStateResponse = client
        .get(&format!("{}/state", NODE_RPC_URL))
        .send().await?.json().await?;
    {
        let write_txn = db.begin_write()?;
        {
            let mut meta = write_txn.open_table(METADATA_TABLE)?;
            meta.insert("last_scan_height", state_resp.height.to_le_bytes().as_slice())?;
        }
        write_txn.commit()?;
    }

    tracing::info!("Payout complete.");
    Ok(())
}

fn load_mss_key(db: &Database) -> Result<MssKeypair> {
    let read_txn = db.begin_read()?;
    let table = read_txn.open_table(METADATA_TABLE)?;
    if let Some(bytes) = table.get("mss_keypair")? {
        Ok(bincode::deserialize(bytes.value())?)
    } else {
        anyhow::bail!("Pool MSS key not found in database! Must start the pool server first.")
    }
}

fn save_mss_key(db: &Database, keypair: &MssKeypair) -> Result<()> {
    let write_txn = db.begin_write()?;
    {
        let mut table = write_txn.open_table(METADATA_TABLE)?;
        table.insert("mss_keypair", bincode::serialize(keypair)?.as_slice())?;
    }
    write_txn.commit()?;
    Ok(())
}
