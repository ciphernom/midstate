//! Mining coordination for the full node.
//!
//! This module was extracted from the `node` god module during the refactor.
//! It owns the types for miner config, the result of a mining attempt (block or pool share),
//! and helpers for coinbase generation and logging.
//!
//! The CPU-intensive extension mining itself lives in `crate::core::extension`.
//! Template building for *external* miners (RPC/light) remains in `node.rs` as
//! `build_block_template_inner` (due to its dependency on `rpc::types`).

use crate::core::types::*;
use crate::core::{block_reward, decompose_value, CoinbaseOutput};
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::wots;
use std::path::PathBuf;
use rayon::prelude::*;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinerToml {
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiningConfig {
    pub mode: String,
    pub pool_url: Option<String>,
    pub payout_address: Option<String>,
    pub pool_address: Option<String>,
}

pub enum MinedResult {
    Block(crate::core::Batch),
    Share {
        batch: crate::core::Batch,
        pool_url: String,
        payout_address: String,
    },
}

/// Generate coinbase outputs for a new block being mined.
///
/// In solo mode: uses the node's mining seed to derive WOTS keys/addresses/salts.
/// In pool mode: pays the pool's MSS address and watermarks the miner's payout
/// address into the salt (so the pool can later prove who earned the share).
pub fn generate_coinbase(
    mining_seed: &[u8; 32],
    height: u64,
    total_fees: u64,
    pool_target: Option<([u8; 32], [u8; 32])>, // (Pool MSS Address, Miner Payout Address)
) -> Vec<CoinbaseOutput> {
    let reward = block_reward(height);
    let total_value = reward.saturating_add(total_fees);
    let denominations = decompose_value(total_value);

    denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            match pool_target {
                Some((pool_addr, miner_addr)) => {
                    // POOL MINING MODE
                    // Pay the pool's address, but embed the miner's address in the salt
                    // so the pool can cryptographically verify who did the work.
                    let mut salt = [0u8; 32];
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"pool_share");
                    hasher.update(&miner_addr);
                    hasher.update(&height.to_le_bytes());
                    hasher.update(&(i as u64).to_le_bytes());
                    salt.copy_from_slice(hasher.finalize().as_bytes());

                    CoinbaseOutput { address: pool_addr, value, salt }
                }
                None => {
                    // SOLO MINING MODE (Original Logic)
                    let seed = coinbase_seed(mining_seed, height, i as u64);
                    let owner_pk = wots::keygen(&seed);
                    let address = compute_address(&owner_pk);
                    let salt = coinbase_salt(mining_seed, height, i as u64);
                    
                    CoinbaseOutput { address, value, salt }
                }
            }
        })
        .collect()
}

/// Append a JSONL entry for every coinbase output created at this height.
/// The seed itself is deliberately NOT logged (it is derivable from the
/// node's persistent mining_seed + height + index).
pub fn log_coinbase(
    mining_seed: &[u8; 32],
    data_dir: &PathBuf,
    height: u64,
    total_fees: u64,
) {
    let reward = block_reward(height);
    let total_value = reward + total_fees;
    let denominations = decompose_value(total_value);
    let log_path = data_dir.join("coinbase_seeds.jsonl");

    let entries: Vec<String> = denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            let seed = coinbase_seed(mining_seed, height, i as u64);
            let owner_pk = wots::keygen(&seed);
            let address = compute_address(&owner_pk);
            let salt = coinbase_salt(mining_seed, height, i as u64);
            let coin_id = compute_coin_id(&address, value, &salt);
            // NOTE: We intentionally do NOT log the seed (private key).
            // It is derivable from (mining_seed, height, index) when
            // the wallet needs to spend. Logging it in cleartext would
            // allow anyone with filesystem or RPC access to steal funds.
            format!(
                r#"{{"height":{},"index":{},"address":"{}","coin":"{}","value":{},"salt":"{}"}}"#,
                height, i,
                hex::encode(address),
                hex::encode(coin_id),
                value,
                hex::encode(salt)
            )
        })
        .collect();

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true).append(true).open(&log_path)
    {
        use std::io::Write;
        for entry in entries {
            let _ = writeln!(file, "{}", entry);
        }
    }
}

/// Lightweight coordinator for the node's autonomous mining.
/// Holds the configuration that is stable across mining attempts.
/// The heavy per-block preparation + delegation to core::extension::mine_extension
/// is still driven from Node (to keep access to live mempool/state), but
/// this gives the mining logic a named home outside the god module.
#[derive(Clone)]
pub struct MiningCoordinator {
    pub threads: Option<usize>,
    seed: [u8; 32],
    data_dir: PathBuf,
}

impl MiningCoordinator {
    pub fn new(threads: Option<usize>, seed: [u8; 32], data_dir: PathBuf) -> Self {
        Self {
            threads,
            seed,
            data_dir,
        }
    }

    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    /// Convenience wrapper around the free function, using the coordinator's seed.
    pub fn generate_coinbase(
        &self,
        height: u64,
        total_fees: u64,
        pool_target: Option<([u8; 32], [u8; 32])>,
    ) -> Vec<CoinbaseOutput> {
        generate_coinbase(&self.seed, height, total_fees, pool_target)
    }

    /// Convenience wrapper for logging using the coordinator's seed + data_dir.
    pub fn log_coinbase(&self, height: u64, total_fees: u64) {
        log_coinbase(&self.seed, &self.data_dir, height, total_fees);
    }
}