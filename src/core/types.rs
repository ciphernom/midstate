use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Hash a byte slice with SHA-256
pub fn hash(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// Concatenate two byte slices and hash them
pub fn hash_concat(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(a);
    hasher.update(b);
    hasher.finalize().into()
}

/// Compute a commitment hash that binds inputs to outputs
///
/// commitment = SHA256(coin_id_1 || coin_id_2 || ... || new_coin_1 || new_coin_2 || ... || salt)
pub fn compute_commitment(
    input_coins: &[[u8; 32]],
    new_coins: &[[u8; 32]],
    salt: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for coin in input_coins {
        hasher.update(coin);
    }
    for coin in new_coins {
        hasher.update(coin);
    }
    hasher.update(salt);
    hasher.finalize().into()
}

/// The global consensus state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    /// Cumulative hash of all history
    pub midstate: [u8; 32],

    /// Set of unspent coin commitments
    pub coins: HashSet<[u8; 32]>,

    /// Set of registered commitments (pending reveals)
    pub commitments: HashSet<[u8; 32]>,

    /// Cumulative sequential work (number of hash iterations)
    pub depth: u64,

    /// Current difficulty target
    pub target: [u8; 32],

    /// Number of batches processed
    pub height: u64,

    /// Unix timestamp of this state (for difficulty adjustment)
    pub timestamp: u64,
}

impl State {
    /// Create genesis state
    pub fn genesis() -> Self {
        let genesis_coins = vec![
            hash(b"genesis_coin_1"),
            hash(b"genesis_coin_2"),
            hash(b"genesis_coin_3"),
        ];

        // Initial difficulty: ~1 in 10 (easy for testing)
        let target = [
            0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];

        Self {
            midstate: hash(b"midstate_genesis_2026"),
            coins: genesis_coins.into_iter().collect(),
            commitments: HashSet::new(),
            depth: 0,
            target,
            height: 0,
            timestamp: 0,
        }
    }
}

/// A transaction is either a Commit (register intent) or a Reveal (execute spend)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Transaction {
    /// Phase 1: Register a commitment binding inputs to outputs.
    /// The commitment is opaque — it hides which coins and destinations are involved.
    Commit {
        commitment: [u8; 32],
    },

    /// Phase 2: Reveal secrets and destinations, proving they match a prior commitment.
    /// The commitment must already exist in state (from a previous batch).
    Reveal {
        /// The secret preimages that unlock the old coins
        secrets: Vec<Vec<u8>>,
        /// New coin commitments to create
        new_coins: Vec<[u8; 32]>,
        /// Salt used when computing the commitment
        salt: [u8; 32],
    },
}

impl Transaction {
    /// Get the coins this transaction is spending (empty for Commit)
    pub fn input_coins(&self) -> Vec<[u8; 32]> {
        match self {
            Transaction::Commit { .. } => vec![],
            Transaction::Reveal { secrets, .. } => {
                secrets.iter().map(|s| hash(s)).collect()
            }
        }
    }
}

/// Proof of sequential work
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Extension {
    /// Mining nonce
    pub nonce: u64,

    /// Result of sequential hashing
    pub final_hash: [u8; 32],
}

/// A batch of transactions plus proof of work
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub transactions: Vec<Transaction>,
    pub extension: Extension,
}

/// Protocol constants
#[cfg(not(feature = "fast-mining"))]
pub const EXTENSION_ITERATIONS: u64 = 1_000_000; // ~1 second @ 1 GHz (faster for testing)

// TESTING: Only 100 iterations (10,000x faster)
#[cfg(feature = "fast-mining")]
pub const EXTENSION_ITERATIONS: u64 = 100;

pub const MAX_BATCH_SIZE: usize = 100;

/// Difficulty adjustment parameters
pub const TARGET_BLOCK_TIME: u64 = 10; // seconds
pub const DIFFICULTY_ADJUSTMENT_INTERVAL: u64 = 10; // blocks
pub const MAX_ADJUSTMENT_FACTOR: u64 = 4; // max 4x change per adjustment
