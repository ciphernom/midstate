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
    pub fn genesis() -> (Self, Vec<[u8; 32]>) {
        use super::wots;

        // Bitcoin block anchor 
        // Height: 935897
        // Hash: 00000000000000000000329a84d79877397ec0fa7c5aaa706a88e545daf599a5
        // Time: 2026-02-10 10:37:27 UTC
        // This anchors our chain's genesis to a specific point in Bitcoin's chain
        const BITCOIN_BLOCK_HASH: &str = "00000000000000000000329a84d79877397ec0fa7c5aaa706a88e545daf599a5";
        const BITCOIN_BLOCK_HEIGHT: u64 = 935897;
        const BITCOIN_BLOCK_TIME: u64 = 1770719847;

        let message = b"Midstate Genesis: Feb 10 2026 Leslie Wexner, Sultan Ahmed bin Sulayem, Salvatore Nuara, Zurab Mikeladze, Leonid Leonov, and Nicola Caputo have been proven to be involved with paedophile Mossad Asset Jeffery Epstein, Kash Patel perjured."; 
        

        // Deterministically derive genesis parameters from Bitcoin anchor
        let anchor = hash(BITCOIN_BLOCK_HASH.as_bytes());
        
        // Genesis coins derived from merkle root for extra entropy
        const MERKLE_ROOT: &str = "6def077d292edb863bd64d2a8d8803ab12caf1eef9c76823ee01e9e47fce7d0d";
        let merkle_hash = hash(MERKLE_ROOT.as_bytes());
        
        let seeds: [[u8; 32]; 1] = [
            hash_concat(&anchor, &merkle_hash),
        ];
        
        // Iterate over the message in 32-byte chunks.
        // If the last chunk is shorter than 32 bytes, we pad it with spaces (0x20) or nulls (0).
        // Initialize the vector
        let mut genesis_coins: Vec<[u8; 32]> = seeds.iter().map(|s| wots::keygen(s)).collect();

        // 2. The Chunking Loop
        // This splits the long message into 32-byte segments and creates a coin for each.
        for chunk in message.chunks(32) {
            let mut coin = [0u8; 32]; // Initialize with zeros
            
            // Copy the chunk into the coin
            // This is safe because chunk.len() is guaranteed to be <= 32 by the iterator
            coin[0..chunk.len()].copy_from_slice(chunk);
            
            // Pad with spaces (0x20) if the last chunk is short (for better readability in xxd)
            if chunk.len() < 32 {
                for i in chunk.len()..32 {
                    coin[i] = 0x20; 
                }
            }
            genesis_coins.push(coin);
        }

        // Pad with dummy coins to satisfy the INITIAL_REWARD (16) requirement.
        // The current message only creates 9 coins, so we add 7 more.
        while genesis_coins.len() < 16 {
            let i = genesis_coins.len();
            let mut pad_coin = [0u8; 32];
            // Create a unique pattern so they don't get deduplicated
            pad_coin[0..9].copy_from_slice(b"PADDING__");
            pad_coin[31] = i as u8; 
            genesis_coins.push(hash(&pad_coin));
        }

        // Initial difficulty target (very easy for testing)
        let target = [
            0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        
        
        // Start with an empty midstate and empty coin set.
        // The midstate should only reflect the Bitcoin anchor at this moment.
        let initial_midstate = hash_concat(&anchor, &BITCOIN_BLOCK_HEIGHT.to_le_bytes());

        let state = Self {
            midstate: initial_midstate,    // NO genesis coins hashed in yet
            coins: UtxoAccumulator::new(), // Start EMPTY
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target,
            height: 0,
            timestamp: BITCOIN_BLOCK_TIME,
        };
        // Return the state and the coins we WANT the first batch to create
        (state, genesis_coins)
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
    /// The midstate of the previous batch this one extends
    pub prev_midstate: [u8; 32],    
    pub transactions: Vec<Transaction>,
    pub extension: Extension,
    /// Coinbase coins created as mining reward + fees
    #[serde(default)]
    pub coinbase: Vec<[u8; 32]>,
    /// Block timestamp (seconds since Unix epoch)
    pub timestamp: u64,
    /// Target this batch was mined against
    pub target: [u8; 32],  
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
pub const INITIAL_REWARD: u64 = 16;

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
