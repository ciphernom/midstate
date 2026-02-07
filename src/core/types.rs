use serde::{Deserialize, Serialize};
use super::mmr::UtxoAccumulator;


/// Hash a byte slice with BLAKE3 (truncated to 32 bytes — BLAKE3 native output).
pub fn hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Concatenate two byte slices and hash them with BLAKE3.
pub fn hash_concat(a: &[u8], b: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(a);
    hasher.update(b);
    *hasher.finalize().as_bytes()
}

/// Compute a commitment hash that binds inputs to outputs.
///
/// commitment = BLAKE3(coin_id_1 || ... || new_coin_1 || ... || salt)
pub fn compute_commitment(
    input_coins: &[[u8; 32]],
    new_coins: &[[u8; 32]],
    salt: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for coin in input_coins {
        hasher.update(coin);
    }
    for coin in new_coins {
        hasher.update(coin);
    }
    hasher.update(salt);
    *hasher.finalize().as_bytes()
}

/// The global consensus state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub midstate: [u8; 32],
    pub coins: UtxoAccumulator,
    pub commitments: UtxoAccumulator,
    pub depth: u64,
    pub target: [u8; 32],
    pub height: u64,
    pub timestamp: u64,
}

impl State {
    pub fn genesis() -> Self {
        use super::wots;

        let seeds: [[u8; 32]; 3] = [
            hash(b"genesis_coin_1"),
            hash(b"genesis_coin_2"),
            hash(b"genesis_coin_3"),
        ];
        let genesis_coins: Vec<[u8; 32]> = seeds.iter().map(|s| wots::keygen(s)).collect();

        let target = [
            0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];

        Self {
            midstate: hash(b"midstate_genesis_v2_blake3"),
            coins: UtxoAccumulator::from_set(genesis_coins),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target,
            height: 0,
            timestamp: 0,
        }
    }
}

/// A transaction is either a Commit or a Reveal
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Transaction {
    /// Phase 1: Register a commitment binding inputs to outputs.
    Commit {
        commitment: [u8; 32],
    },

    /// Phase 2: Reveal and execute the spend with WOTS signatures.
    Reveal {
        /// The coin IDs being spent (public, in the UTXO set).
        input_coins: Vec<[u8; 32]>,
        /// WOTS signatures proving ownership of each input coin (one per input).
        signatures: Vec<Vec<u8>>,
        /// New coin commitments to create
        new_coins: Vec<[u8; 32]>,
        /// Salt used when computing the commitment
        salt: [u8; 32],
    },
}

impl Transaction {
    /// Get the coins this transaction is spending (empty for Commit).
    pub fn input_coins(&self) -> Vec<[u8; 32]> {
        match self {
            Transaction::Commit { .. } => vec![],
            Transaction::Reveal { input_coins, .. } => input_coins.clone(),
        }
    }

    /// Fee = number of inputs - number of outputs. Zero for Commit.
    pub fn fee(&self) -> usize {
        match self {
            Transaction::Commit { .. } => 0,
            Transaction::Reveal { input_coins, new_coins, .. } => {
                input_coins.len().saturating_sub(new_coins.len())
            }
        }
    }
}

/// Proof of sequential work with checkpoint witnesses
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Extension {
    pub nonce: u64,
    pub final_hash: [u8; 32],
    pub checkpoints: Vec<[u8; 32]>,
}

/// A batch of transactions plus proof of work
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    pub transactions: Vec<Transaction>,
    pub extension: Extension,
    /// Coinbase coins created as mining reward + fees
    #[serde(default)]
    pub coinbase: Vec<[u8; 32]>,
}

// ── Protocol constants ──────────────────────────────────────────────────────

#[cfg(not(feature = "fast-mining"))]
pub const EXTENSION_ITERATIONS: u64 = 1_000_000;
#[cfg(feature = "fast-mining")]
pub const EXTENSION_ITERATIONS: u64 = 100;

#[cfg(not(feature = "fast-mining"))]
pub const CHECKPOINT_INTERVAL: u64 = 1_000;
#[cfg(feature = "fast-mining")]
pub const CHECKPOINT_INTERVAL: u64 = 10;

#[cfg(not(feature = "fast-mining"))]
pub const SPOT_CHECK_COUNT: usize = 16;
#[cfg(feature = "fast-mining")]
pub const SPOT_CHECK_COUNT: usize = 3;

pub const MAX_BATCH_SIZE: usize = 100;

// ── Difficulty adjustment ───────────────────────────────────────────────────

pub const TARGET_BLOCK_TIME: u64 = 10;
pub const DIFFICULTY_ADJUSTMENT_INTERVAL: u64 = 10;
pub const MAX_ADJUSTMENT_FACTOR: u64 = 4;

// ── Economics ───────────────────────────────────────────────────────────────

/// Blocks per year at TARGET_BLOCK_TIME seconds per block.
pub const BLOCKS_PER_YEAR: u64 = 365 * 24 * 3600 / TARGET_BLOCK_TIME; // 3_153_600

/// Initial block reward in coins.
pub const INITIAL_REWARD: u64 = 256;

/// Block reward at a given height. Halves every BLOCKS_PER_YEAR, minimum 1.
pub fn block_reward(height: u64) -> u64 {
    let halvings = height / BLOCKS_PER_YEAR;
    if halvings >= 8 {
        1
    } else {
        (INITIAL_REWARD >> halvings).max(1)
    }
}

const _: () = assert!(
    EXTENSION_ITERATIONS % CHECKPOINT_INTERVAL == 0,
    "EXTENSION_ITERATIONS must be divisible by CHECKPOINT_INTERVAL"
);
