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

/// The global consensus state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    /// Cumulative hash of all history
    pub midstate: [u8; 32],
    
    /// Set of unspent coin commitments
    pub coins: HashSet<[u8; 32]>,
    
    /// Cumulative sequential work (number of hash iterations)
    pub depth: u64,
    
    /// Current difficulty target
    pub target: [u8; 32],
    
    /// Number of batches processed
    pub height: u64,
}

impl State {
    /// Create genesis state
    pub fn genesis() -> Self {
        let genesis_coins = vec![
            hash(b"genesis_coin_1"),
            hash(b"genesis_coin_2"),
            hash(b"genesis_coin_3"),
        ];
        
        Self {
            midstate: hash(b"midstate_genesis_2026"),
            coins: genesis_coins.into_iter().collect(),
            depth: 0,
            target: [0xff; 32], // Easy difficulty for testing
            height: 0,
        }
    }
}

/// A transaction spends multiple coins and creates new coins
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Transaction {
    /// The secret preimages that unlock the old coins
    pub secrets: Vec<Vec<u8>>,
    
    /// New coin commitments to create
    pub new_coins: Vec<[u8; 32]>,
}

impl Transaction {
    /// Get the coins this transaction is spending
    pub fn input_coins(&self) -> Vec<[u8; 32]> {
        self.secrets.iter().map(|s| hash(s)).collect()
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
pub const EXTENSION_ITERATIONS: u64 = 1_000_000; // ~1 second @ 1 GHz (faster for testing)
pub const MAX_BATCH_SIZE: usize = 100;
