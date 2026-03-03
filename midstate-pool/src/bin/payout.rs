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
    let client = reqwest::Client::new();

    // 1. Load the Pool's MSS Keypair
    let mut mss_keypair = load_mss_key(&db)?;
    let pool_address = mss_keypair.master_pk;
    tracing::info!("Pool Address: {}", hex::encode(pool_address));

    // 2. Scan the local node for the pool's available UTXOs
    tracing::info!("Scanning node for pool UTXOs...");
    let scan_req = serde_json::json!({
        "addresses": [hex::encode(pool_address)],
        "start_height": 0,
        "end_height": 999_999_999 // Scan to tip
    });
    
    let res = client.post(&format!("{}/scan", NODE_RPC_URL))
        .json(&scan_req)
        .send().await?.json::<serde_json::Value>().await?;
        
    let coins = res["coins"].as_array().context("No coins returned")?;
    if coins.is_empty() {
        tracing::info!("Pool has no balance. Exiting.");
        return Ok(());
    }

    let mut total_balance = 0u64;
    let mut inputs_to_spend = Vec::new();
    
    for c in coins {
        let val = c["value"].as_u64().unwrap();
        total_balance += val;
        inputs_to_spend.push(InputReveal {
            predicate: Predicate::p2pk(&pool_address),
            value: val,
            salt: hex::decode(c["salt"].as_str().unwrap())?.try_into().unwrap(),
        });
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
    tracing::info!("Commit mined! Generating MSS signatures...");
    
    let mut signatures = Vec::new();
    for _ in &inputs_to_spend {
        if mss_keypair.remaining() == 0 {
            anyhow::bail!("CRITICAL: MSS Key capacity exhausted during payout!");
        }
        // Generate the post-quantum signature using the pool's tree
        let sig = mss_keypair.sign(&commitment)?;
        signatures.push(hex::encode(sig.to_bytes()));
    }

    // Save the updated MSS key state (next_leaf index) to the DB immediately
    save_mss_key(&db, &mss_keypair)?;

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

    // 7. Success! Wipe the shares table for the next round.
    tracing::info!("Payout successful! Wiping shares table.");
    let write_txn = db.begin_write()?;
    {
        let mut table = write_txn.open_table(SHARES_TABLE)?;
        // redb doesn't have a truncate(), so we pop everything
        while table.pop_last()?.is_some() {}
    }
    write_txn.commit()?;

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
