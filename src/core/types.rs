use serde::{Deserialize, Deserializer, Serialize};
use super::mmr::UtxoAccumulator;


/// A stateless spending condition that governs a UTXO.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// A compiled MidstateScript bytecode payload.
    Script { bytecode: Vec<u8> },
}

impl Predicate {
    /// Address = BLAKE3(bytecode). Every address is Pay-to-Script-Hash.
    pub fn address(&self) -> [u8; 32] {
        match self {
            Predicate::Script { bytecode } => hash(bytecode),
        }
    }

    /// Convenience: build a standard P2PK predicate.
    pub fn p2pk(owner_pk: &[u8; 32]) -> Self {
        Predicate::Script { bytecode: super::script::compile_p2pk(owner_pk) }
    }

    /// Extract owner_pk from a standard P2PK script, if it is one.
    pub fn owner_pk(&self) -> Option<[u8; 32]> {
        match self {
            Predicate::Script { bytecode } => {
                // P2PK: PUSH_DATA(32) + CHECKSIGVERIFY + PUSH_DATA(1) = 40 bytes
                if bytecode.len() == 40
                    && bytecode[0] == 0x01
                    && bytecode[1] == 32
                    && bytecode[2] == 0
                    && bytecode[35] == 0x32
                {
                    let mut pk = [0u8; 32];
                    pk.copy_from_slice(&bytecode[3..35]);
                    Some(pk)
                } else {
                    None
                }
            }
        }
    }
}

/// The proof provided to satisfy a Predicate.
#[derive(Clone, Debug, Serialize, PartialEq, Eq, Hash)]
pub enum Witness {
    /// Raw byte arrays pushed onto the stack before script execution.
    ScriptInputs(Vec<Vec<u8>>),
}

/// Maximum size of a single witness stack item at deserialization time.
/// Matches MAX_ITEM_SIZE from the script VM (131,072 bytes / 128 KB).
/// Rejects oversized items before they reach the script engine, preventing
/// multi-megabyte heap allocations from malicious P2P messages.
const MAX_WITNESS_ITEM_SIZE: usize = 131_072;

impl<'de> Deserialize<'de> for Witness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum WitnessHelper {
            ScriptInputs(Vec<Vec<u8>>),
        }

        let helper = WitnessHelper::deserialize(deserializer)?;
        match helper {
            WitnessHelper::ScriptInputs(items) => {
                for (i, item) in items.iter().enumerate() {
                    if item.len() > MAX_WITNESS_ITEM_SIZE {
                        return Err(serde::de::Error::custom(format!(
                            "Witness stack item {} is {} bytes (max {})",
                            i, item.len(), MAX_WITNESS_ITEM_SIZE
                        )));
                    }
                }
                Ok(Witness::ScriptInputs(items))
            }
        }
    }
}

impl Witness {
    /// Convenience: P2PK witness from a single signature.
    pub fn sig(sig_bytes: Vec<u8>) -> Self {
        Witness::ScriptInputs(vec![sig_bytes])
    }
}

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

/// Count the number of leading zero bits in a 32-byte hash
pub fn count_leading_zeros(hash: &[u8; 32]) -> u32 {
    let mut zeros = 0;
    for &byte in hash {
        if byte == 0 {
            zeros += 8;
        } else {
            zeros += byte.leading_zeros();
            break;
        }
    }
    zeros
}

/// Compute a standard P2PK address: BLAKE3(compile_p2pk(owner_pk)).
pub fn compute_address(owner_pk: &[u8; 32]) -> [u8; 32] {
    Predicate::p2pk(owner_pk).address()
}

/// Compute a coin ID that commits to address, value, and salt.
/// CoinID = BLAKE3(address || value_le_bytes || salt)
/// The UTXO set stores ONLY this 32-byte hash.
pub fn compute_coin_id(address: &[u8; 32], value: u64, salt: &[u8; 32]) -> [u8; 32] {
     let mut hasher = blake3::Hasher::new();
     hasher.update(address);
     hasher.update(&value.to_le_bytes());
     hasher.update(salt);
     *hasher.finalize().as_bytes()
}

/// Create a confidential value commitment: Blake3(value_le || blinding_factor).
/// The blinding_factor must be 32 bytes of cryptographic randomness.
pub fn compute_value_commitment(value: u64, blinding: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&value.to_le_bytes());
    hasher.update(blinding);
    *hasher.finalize().as_bytes()
}

/// Compute a commitment hash that binds inputs to outputs.
///
/// commitment = BLAKE3(NETWORK_MAGIC || coin_id_1 || ... || new_coin_id_1 || ... || salt)
pub fn compute_commitment(
    input_coins: &[[u8; 32]],
    new_coins: &[[u8; 32]],
    salt: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NETWORK_MAGIC); 
    // Length-prefix both arrays to prevent boundary ambiguity:
    // without this, [A,B]+[C] and [A]+[B,C] hash identically.
    hasher.update(&(input_coins.len() as u32).to_le_bytes());
    for coin in input_coins {
        hasher.update(coin);
    }
    hasher.update(&(new_coins.len() as u32).to_le_bytes());
    for coin in new_coins {
        hasher.update(coin);
    }
    hasher.update(salt);
    *hasher.finalize().as_bytes()
}

/// Decompose a value into power-of-2 denominations (its binary representation).
pub fn decompose_value(mut value: u64) -> Vec<u64> {
    let mut parts = Vec::new();
    let mut bit = 1u64;
    while value > 0 {
        if value & 1 == 1 {
            parts.push(bit);
        }
        value >>= 1;
        if value > 0 {
            bit <<= 1;
        }
    }
    parts
}

/// Encodes a 32-byte address into a 72-character hex string with a 4-byte checksum.
pub fn encode_address_with_checksum(address: &[u8; 32]) -> String {
    let checksum_hash = hash(address);
    let checksum = &checksum_hash[0..4];

    let mut payload = Vec::with_capacity(36);
    payload.extend_from_slice(address);
    payload.extend_from_slice(checksum);

    hex::encode(payload)
}

/// Safely parses an address, automatically handling both legacy (64-char) 
/// and new checksummed (72-char) formats.
pub fn parse_address_flexible(s: &str) -> Result<[u8; 32], String> {
    if s.len() == 64 {
        // Legacy Format (No Checksum)
        let decoded = hex::decode(s).map_err(|e| format!("Invalid hex: {}", e))?;
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&decoded);
        Ok(addr)
    } else if s.len() == 72 {
        // New Format (With Checksum)
        let decoded = hex::decode(s).map_err(|e| format!("Invalid hex: {}", e))?;
        
        let address: [u8; 32] = decoded[0..32].try_into().unwrap();
        let expected_checksum = &decoded[32..36];

        let actual_checksum_hash = hash(&address);
        let actual_checksum = &actual_checksum_hash[0..4];

        if expected_checksum != actual_checksum {
            return Err("Checksum mismatch! The address contains a typo.".to_string());
        }
        Ok(address)
    } else {
        Err(format!("Invalid address length: expected 64 or 72 characters, got {}", s.len()))
    }
}

/// The global consensus state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub midstate: [u8; 32],
    pub coins: UtxoAccumulator,
    pub commitments: UtxoAccumulator,
    pub depth: u128,
    pub target: [u8; 32],
    pub height: u64,
    pub timestamp: u64,
    #[serde(default)]
    pub commitment_heights: im::HashMap<[u8; 32], u64>,
    /// Append-only log of historical block hashes for light client proofs
    #[serde(default)]
    pub chain_mmr: crate::core::mmr::MerkleMountainRange,
}

impl State {
    pub fn genesis() -> (Self, Vec<CoinbaseOutput>) {
        // Bitcoin block anchor
        // Height: 938708
        // Hash: 000000000000000000018f5ad5625d43356136c2e50c6dc18967a90a18f0af2e
        const BITCOIN_BLOCK_HASH: &str = "000000000000000000018f5ad5625d43356136c2e50c6dc18967a90a18f0af2e";
        const BITCOIN_BLOCK_HEIGHT: u64 = 938708;
        const BITCOIN_BLOCK_TIME: u64 = 1772274770;

        let anchor = hash(BITCOIN_BLOCK_HASH.as_bytes());

        // --- The Genesis Inscription ---
        // We embed this plaintext directly into the address and salt fields of 
        // the genesis coinbase outputs. This places the message permanently 
        // into the blockchain data on disk, while mathematically ensuring the 
        // initial coins are unspendable (burned).
        
        let mut chunk0 = [0u8; 32];
        chunk0.copy_from_slice(b"Harvest Now, Decrypt Later: The ");
        
        let mut chunk1 = [0u8; 32];
        chunk1.copy_from_slice(b"Quantum Era's Encryption Challen");
        
        let mut chunk2 = [0u8; 32];
        chunk2.copy_from_slice(b"ge (Published Feb 24, 2026, by R");
        
        let mut chunk3 = [0u8; 32];
        let last_part = b"BC Disruptors)";
        chunk3[..last_part.len()].copy_from_slice(last_part);

        // INITIAL_REWARD is 16. We use two outputs of 8 (both are powers of 2).
        let genesis_coinbase = vec![
            CoinbaseOutput {
                address: chunk0, // Plaintext!
                value: 536_870_912,
                salt: chunk1,    // Plaintext!
            },
            CoinbaseOutput {
                address: chunk2, // Plaintext!
                value: 536_870_912,
                salt: chunk3,    // Plaintext!
            }
        ];

        // Initial difficulty target (calibrated for ~60s blocks on baseline hardware)
        let target = [
            0x00, 0x11, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        ];
        let initial_midstate = hash_concat(&anchor, &BITCOIN_BLOCK_HEIGHT.to_le_bytes());

        let state = Self {
            midstate: initial_midstate,
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target,
            height: 0,
            timestamp: BITCOIN_BLOCK_TIME,
            commitment_heights: im::HashMap::new(),
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
        };
        (state, genesis_coinbase)
    }
pub fn header(&self) -> BatchHeader {
        // Clone coins because root() requires &mut self
        let mut coins_clone = self.coins.clone();
        let mut commitments_clone = self.commitments.clone();
        let smt_root = hash_concat(&coins_clone.root(), &commitments_clone.root());
        let state_root = hash_concat(&smt_root, &self.chain_mmr.root());

        BatchHeader {
            height: self.height,
            prev_midstate: [0u8; 32],
            post_tx_midstate: self.midstate,
            extension: Extension {
                nonce: 0,
                final_hash: self.midstate,
            },
            timestamp: self.timestamp,
            target: self.target,
            state_root, 
        }
    }
}
impl Batch {
    /// Compute header from full batch by replaying transaction commitments
    pub fn header(&self) -> BatchHeader {
        let mut midstate = self.prev_midstate;

        // 1. Replay transaction commitments
        for tx in &self.transactions {
            match tx {
                Transaction::Commit { commitment, .. } => {
                    midstate = hash_concat(&midstate, commitment);
                }
                Transaction::Reveal { inputs, outputs, salt, .. } => {
                    let mut hasher = blake3::Hasher::new();
                    for i in inputs { hasher.update(&i.coin_id()); }
                    for o in outputs { hasher.update(&o.hash_for_commitment()); }
                    hasher.update(salt);
                    let tx_hash = *hasher.finalize().as_bytes();
                    midstate = hash_concat(&midstate, &tx_hash);
                }
            }
        }

        // 2. Replay coinbase
        for cb in &self.coinbase {
            midstate = hash_concat(&midstate, &cb.coin_id());
        }

// 3. Hash in the state root (Bypass for legacy blocks!)
        if self.state_root != [0u8; 32] {
            midstate = hash_concat(&midstate, &self.state_root);
        }

        BatchHeader {
            height: 0, // Caller assigns this
            prev_midstate: self.prev_midstate,
            post_tx_midstate: midstate,
            extension: self.extension.clone(),
            timestamp: self.timestamp,
            target: self.target,
            state_root: self.state_root, 
        }
    }
}

// ── Value-bearing data structures ───────────────────────────────────────────


/// Cleartext output data carried in a transaction.
#[derive(Clone, Debug, serde::Serialize, PartialEq, Eq, Hash)] // <-- Removed serde::Deserialize here
pub enum OutputData {
    Standard {
        address: [u8; 32],
        value: u64,
        salt: [u8; 32],
    },
    /// A confidential output whose value is hidden behind a Blake3 commitment.
    /// Balance correctness is proven via OP_VERIFY_STARK with the
    /// `confidential_transfer` program. The node does not perform arithmetic
    /// balance checking — the STARK proof covers it.
    Confidential {
        address: [u8; 32],
        commitment: [u8; 32],
        salt: [u8; 32],
    },
    /// A provably unspendable data payload that is ignored by the UTXO SMT
    DataBurn {
        payload: Vec<u8>,
        value_burned: u64,
    }
}

// Manually implement Deserialize to enforce the payload size limit
impl<'de> Deserialize<'de> for OutputData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum OutputDataHelper {
            Standard { address: [u8; 32], value: u64, salt: [u8; 32] },
            Confidential { address: [u8; 32], commitment: [u8; 32], salt: [u8; 32] },
            DataBurn { payload: Vec<u8>, value_burned: u64 },
        }

        let helper = OutputDataHelper::deserialize(deserializer)?;
        match helper {
            OutputDataHelper::Standard { address, value, salt } => {
                Ok(OutputData::Standard { address, value, salt })
            }
            OutputDataHelper::Confidential { address, commitment, salt } => {
                Ok(OutputData::Confidential { address, commitment, salt })
            }
            OutputDataHelper::DataBurn { payload, value_burned } => {
                if payload.len() > crate::core::MAX_BURN_DATA_SIZE {
                    return Err(serde::de::Error::custom(format!(
                        "DataBurn payload exceeds max size of {}", 
                        crate::core::MAX_BURN_DATA_SIZE
                    )));
                }
                Ok(OutputData::DataBurn { payload, value_burned })
            }
        }
    }
}

impl OutputData {
    /// Returns the Coin ID if this is a spendable UTXO (Standard or Confidential).
    pub fn coin_id(&self) -> Option<[u8; 32]> {
        match self {
            OutputData::Standard { address, value, salt } => {
                Some(compute_coin_id(address, *value, salt))
            }
            OutputData::Confidential { address, commitment, salt } => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"CONFIDENTIAL");
                hasher.update(address);
                hasher.update(commitment);
                hasher.update(salt);
                Some(*hasher.finalize().as_bytes())
            }
            OutputData::DataBurn { .. } => None,
        }
    }

    /// The hash used when computing the transaction commitment. 
    /// Burns must be committed to so they cannot be tampered with in the mempool.
    pub fn hash_for_commitment(&self) -> [u8; 32] {
        match self {
            OutputData::Standard { address, value, salt } => {
                compute_coin_id(address, *value, salt)
            }
            OutputData::Confidential { address, commitment, salt } => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"CONFIDENTIAL");
                hasher.update(address);
                hasher.update(commitment);
                hasher.update(salt);
                *hasher.finalize().as_bytes()
            }
            OutputData::DataBurn { payload, value_burned } => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"DATABURN");
                hasher.update(&value_burned.to_le_bytes());
                hasher.update(payload);
                *hasher.finalize().as_bytes()
            }
        }
    }

    /// Returns the visible value. Confidential outputs return 0 —
    /// their real value is proven via STARK, not visible on-chain.
    pub fn value(&self) -> u64 {
        match self {
            OutputData::Standard { value, .. } => *value,
            OutputData::Confidential { .. } => 0,
            OutputData::DataBurn { value_burned, .. } => *value_burned,
        }
    }

    pub fn address(&self) -> [u8; 32] {
        match self {
            OutputData::Standard { address, .. } => *address,
            OutputData::Confidential { address, .. } => *address,
            OutputData::DataBurn { .. } => [0u8; 32],
        }
    }

    pub fn salt(&self) -> [u8; 32] {
        match self {
            OutputData::Standard { salt, .. } => *salt,
            OutputData::Confidential { salt, .. } => *salt,
            OutputData::DataBurn { .. } => [0u8; 32],
        }
    }

    /// Returns the commitment hash for confidential outputs, None otherwise.
    pub fn commitment(&self) -> Option<[u8; 32]> {
        match self {
            OutputData::Confidential { commitment, .. } => Some(*commitment),
            _ => None,
        }
    }

    /// True if this output hides its value behind a STARK-verified commitment.
    pub fn is_confidential(&self) -> bool {
        matches!(self, OutputData::Confidential { .. })
    }
}

/// Cleartext input preimage carried in a reveal transaction.
/// Proves what value a coin holds by revealing the preimage of its coin_id.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InputReveal {
    pub predicate: Predicate,
    pub value: u64,
    pub salt: [u8; 32],
    /// FIX 2: For spending a Confidential UTXO, the spender must provide the
    /// commitment hash so the node can reconstruct the correct coin_id.
    #[serde(default)]
    pub commitment: Option<[u8; 32]>,
}

impl InputReveal {
    pub fn coin_id(&self) -> [u8; 32] {
        match self.commitment {
            Some(ref c) => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"CONFIDENTIAL");
                hasher.update(&self.predicate.address());
                hasher.update(c);
                hasher.update(&self.salt);
                *hasher.finalize().as_bytes()
            }
            None => compute_coin_id(&self.predicate.address(), self.value, &self.salt),
        }
    }
}

/// Coinbase output carried in a Batch. Same validation rules as OutputData.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoinbaseOutput {
    pub address: [u8; 32],
    pub value: u64,
    pub salt: [u8; 32],
}

impl CoinbaseOutput {
    pub fn coin_id(&self) -> [u8; 32] {
        compute_coin_id(&self.address, self.value, &self.salt)
    }
}

// ── Transaction ─────────────────────────────────────────────────────────────

/// A transaction is either a Commit or a Reveal
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Transaction {
    /// Phase 1: Register a commitment binding inputs to outputs.
    Commit {
        commitment: [u8; 32],
        spam_nonce: u64,
    },

    /// Phase 2: Reveal and execute the spend with signatures.
    Reveal {
        /// Preimages proving what each input coin contains.
        inputs: Vec<InputReveal>,
        /// Matches 1:1 with inputs, replacing the old `signatures: Vec<Vec<u8>>`
        witnesses: Vec<Witness>,
        /// New coins to create. Value + salt revealed for validation, then discarded.
        outputs: Vec<OutputData>,
        /// Salt used when computing the commitment.
        salt: [u8; 32],
    },
}

impl Transaction {
    /// Coin IDs this transaction spends.
    pub fn input_coin_ids(&self) -> Vec<[u8; 32]> {
        match self {
            Transaction::Commit { .. } => vec![],
            Transaction::Reveal { inputs, .. } => inputs.iter().map(|i| i.coin_id()).collect(),
        }
    }

    /// Output coin IDs this transaction creates.
    pub fn output_coin_ids(&self) -> Vec<[u8; 32]> {
        match self {
            Transaction::Commit { .. } => vec![],
            Transaction::Reveal { outputs, .. } => outputs.iter().filter_map(|o| o.coin_id()).collect(),
        }
    }

    /// Fee = sum(input values) - sum(output values). Zero for Commit.
    pub fn fee(&self) -> u64 {
        match self {
            Transaction::Commit { .. } => 0,
            Transaction::Reveal { inputs, outputs, .. } => {
                let in_sum = inputs.iter()
                    .try_fold(0u64, |acc, i| acc.checked_add(i.value))
                    .unwrap_or(u64::MAX);
                let out_sum = outputs.iter()
                    .try_fold(0u64, |acc, o| acc.checked_add(o.value()))
                    .unwrap_or(u64::MAX);
                in_sum.saturating_sub(out_sum)
            }
        }
    }
}

/// Proof of sequential work with checkpoint witnesses
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Extension {
    pub nonce: u64,
    pub final_hash: [u8; 32],
}

/// A batch of transactions plus proof of work
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    /// The midstate of the previous batch this one extends
    pub prev_midstate: [u8; 32],
    pub transactions: Vec<Transaction>,
    pub extension: Extension,
    /// Coinbase outputs with revealed values. Each must be a power of 2.
    #[serde(default)]
    pub coinbase: Vec<CoinbaseOutput>,
    /// Block timestamp (seconds since Unix epoch)
    pub timestamp: u64,
    /// Target this batch was mined against
    pub target: [u8; 32],
    /// Explicitly commit to the state
    #[serde(default)]
    pub state_root: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchHeader {
    pub height: u64,
    pub prev_midstate: [u8; 32],
    pub post_tx_midstate: [u8; 32],
    pub extension: Extension,
    pub timestamp: u64,
    pub target: [u8; 32],
    #[serde(default)]
    pub state_root: [u8; 32],
}



// ── Protocol constants ──────────────────────────────────────────────────────
pub const MAX_BURN_DATA_SIZE: usize = 80; //OP_RETURN analog
pub const NETWORK_MAGIC: &[u8] = b"MIDSTATE_MAINNET_V1";
#[cfg(not(feature = "fast-mining"))]
pub const EXTENSION_ITERATIONS: u64 = 1_000_000;
#[cfg(feature = "fast-mining")]
pub const EXTENSION_ITERATIONS: u64 = 100;

/// Maximum number of Reveals (actual value transfers) per batch
pub const MAX_BATCH_REVEALS: usize = 500;
/// Maximum number of Commits per batch. High limit to prevent bottlenecks.
pub const MAX_BATCH_COMMITS: usize = 2_000;

// ── Difficulty adjustment ───────────────────────────────────────────────────

pub const TARGET_BLOCK_TIME: u64 = 60;
/// ASERT half-life in seconds. Difficulty halves (or doubles) for every
/// half-life of drift between actual and ideal elapsed time since genesis.
pub const ASERT_HALF_LIFE: i64 = 4 * 60 * 60; // 4 hours (240 blocks)
/// Maximum number of recent timestamps kept for median-time-past validation
/// and timestamp window management in the node layer.
pub const DIFFICULTY_LOOKBACK: u64 = 60;
pub const MEDIAN_TIME_PAST_WINDOW: usize = 11;
/// Commitment time-to-live in blocks. A commitment must be revealed within
/// this window or it expires and is garbage-collected from state.
///
/// 1000 blocks ≈ 16.7 hours at 60s target. Long enough to survive transient
/// censorship or network partitions; short enough to bound state bloat
/// (max ~20K live commitments at 20 commits/block).
pub const COMMITMENT_TTL: u64 = 1000; 

/// Blocks behind tip before checkpoints are pruned from stored batches.
/// Pruning reclaims ~32 KB per block (~98% of batch storage). Pruned batches
/// remain fully verifiable via full-chain recomputation in verify_extension.
pub const PRUNE_DEPTH: u64 = 1000;

/// Block height at which WOTS address-reuse is enforced at consensus level.
/// Blocks below this height are grandfathered (network was live before this rule).
pub const WOTS_REUSE_ACTIVATION_HEIGHT: u64 = 18_000;

/// Block height at which MSS leaf reuse is enforced at consensus level.
/// Uses the leaf's WOTS public key as a nullifier in the spent oracle.
pub const MSS_REUSE_ACTIVATION_HEIGHT: u64 = 25_000; 

/// Block height at which state_root in batches becomes mandatory.
/// Before this height, blocks may omit the state root (legacy blocks).
/// After this height, state_root must be non-zero and match the expected value.
pub const STATE_ROOT_ACTIVATION_HEIGHT: u64 = 30_000;

/// Block height at which ZK remnants are repurposed into Zero-Value State Threads.
pub const STATE_THREAD_ACTIVATION_HEIGHT: u64 = 65_000;

pub const RECENT_POW_ACTIVATION_HEIGHT: u64 = 80_000;
pub const COMMIT_POW_WINDOW: u64 = 1000;

// ── Economics ───────────────────────────────────────────────────────────────

/// Blocks per year at TARGET_BLOCK_TIME seconds per block.
pub const BLOCKS_PER_YEAR: u64 = 365 * 24 * 3600 / TARGET_BLOCK_TIME; 

/// Initial block reward in value units (2^30).
pub const INITIAL_REWARD: u64 = 1_073_741_824;

pub const MAX_TX_INPUTS: usize = 256;
pub const MAX_TX_OUTPUTS: usize = 256;
/// Maximum size of a single signature in bytes.
/// WOTS = 576, MSS(height=20) = 1280. Pad for safety.
pub const MAX_SIGNATURE_SIZE: usize = 1_536;
/// Cap on total signature verifications per batch to prevent CPU exhaustion.
/// 100 txs × 256 inputs = 25,600 WOTS verifications ≈ 15B hashes — too much.
/// 1,024 total inputs keeps verification under ~600M hashes (~0.5s).
pub const MAX_BATCH_INPUTS: usize = 1_024;

/// Cap on total outputs per batch to strictly bound worst-case block size 
/// and prevent exceeding the 10MB P2P message limit.
pub const MAX_BATCH_OUTPUTS: usize = 10_000;

/// Block reward value at a given height. Halves every BLOCKS_PER_YEAR, minimum 1.
pub fn block_reward(height: u64) -> u64 {
    let halvings = height / BLOCKS_PER_YEAR;
    INITIAL_REWARD >> halvings.min(30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        assert_eq!(hash(b"hello"), hash(b"hello"));
    }

    #[test]
    fn hash_different_inputs_differ() {
        assert_ne!(hash(b"hello"), hash(b"world"));
    }

    #[test]
    fn hash_empty_input() {
        let h = hash(b"");
        assert_ne!(h, [0u8; 32]); // BLAKE3 of empty is defined, not zero
    }

    #[test]
    fn hash_concat_not_commutative() {
        let a = b"alpha";
        let b = b"beta";
        assert_ne!(hash_concat(a, b), hash_concat(b, a));
    }

    #[test]
    fn hash_concat_vs_manual() {
        // hash_concat(a,b) should equal BLAKE3(a || b)
        let a = b"foo";
        let b = b"bar";
        let expected = {
            let mut h = blake3::Hasher::new();
            h.update(a);
            h.update(b);
            *h.finalize().as_bytes()
        };
        assert_eq!(hash_concat(a, b), expected);
    }

    // ── compute_address ─────────────────────────────────────────────────

    #[test]
    fn compute_address_deterministic() {
        let pk = [0xAA; 32];
        assert_eq!(compute_address(&pk), compute_address(&pk));
    }

    #[test]
    fn compute_address_is_hash_of_script() {
        let pk = [0xBB; 32];
        assert_eq!(compute_address(&pk), hash(&crate::core::script::compile_p2pk(&pk)));
    }

    // ── compute_coin_id ─────────────────────────────────────────────────

    #[test]
    fn coin_id_deterministic() {
        let addr = [1u8; 32];
        let salt = [2u8; 32];
        assert_eq!(
            compute_coin_id(&addr, 16, &salt),
            compute_coin_id(&addr, 16, &salt),
        );
    }

    #[test]
    fn coin_id_differs_by_value() {
        let addr = [1u8; 32];
        let salt = [2u8; 32];
        assert_ne!(
            compute_coin_id(&addr, 8, &salt),
            compute_coin_id(&addr, 16, &salt),
        );
    }

    #[test]
    fn coin_id_differs_by_salt() {
        let addr = [1u8; 32];
        assert_ne!(
            compute_coin_id(&addr, 8, &[0u8; 32]),
            compute_coin_id(&addr, 8, &[1u8; 32]),
        );
    }

    #[test]
    fn coin_id_differs_by_address() {
        let salt = [2u8; 32];
        assert_ne!(
            compute_coin_id(&[0u8; 32], 8, &salt),
            compute_coin_id(&[1u8; 32], 8, &salt),
        );
    }

    // ── compute_commitment ──────────────────────────────────────────────

    #[test]
    fn commitment_deterministic() {
        let inputs = vec![[1u8; 32]];
        let outputs = vec![[2u8; 32]];
        let salt = [3u8; 32];
        assert_eq!(
            compute_commitment(&inputs, &outputs, &salt),
            compute_commitment(&inputs, &outputs, &salt),
        );
    }

    #[test]
    fn commitment_differs_with_different_salt() {
        let inputs = vec![[1u8; 32]];
        let outputs = vec![[2u8; 32]];
        assert_ne!(
            compute_commitment(&inputs, &outputs, &[0u8; 32]),
            compute_commitment(&inputs, &outputs, &[1u8; 32]),
        );
    }

    // ── decompose_value ─────────────────────────────────────────────────

    #[test]
    fn decompose_zero() {
        assert!(decompose_value(0).is_empty());
    }

    #[test]
    fn decompose_power_of_two() {
        assert_eq!(decompose_value(16), vec![16]);
        assert_eq!(decompose_value(1), vec![1]);
    }

    #[test]
    fn decompose_sums_correctly() {
        for v in [1, 7, 15, 100, 255, 1023, 65535] {
            let parts = decompose_value(v);
            assert_eq!(parts.iter().sum::<u64>(), v);
            for &p in &parts {
                assert!(p.is_power_of_two());
            }
        }
    }

    #[test]
    fn decompose_all_unique() {
        let parts = decompose_value(255);
        let mut sorted = parts.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(parts.len(), sorted.len(), "decomposition should have unique denominations");
    }

    // ── block_reward ────────────────────────────────────────────────────

    #[test]
    fn block_reward_initial() {
        assert_eq!(block_reward(0), INITIAL_REWARD);
    }

    #[test]
    fn block_reward_first_halving() {
        assert_eq!(block_reward(BLOCKS_PER_YEAR), INITIAL_REWARD / 2);
    }

    #[test]
    fn block_reward_floor_at_one() {
        assert_eq!(block_reward(u64::MAX), 1);
        assert_eq!(block_reward(BLOCKS_PER_YEAR * 100), 1);
    }

    #[test]
    fn block_reward_monotonically_decreasing() {
        let mut prev = block_reward(0);
        for era in 1..=10 {
            let r = block_reward(BLOCKS_PER_YEAR * era);
            assert!(r <= prev);
            prev = r;
        }
    }

    // ── OutputData / InputReveal / CoinbaseOutput coin_id ───────────────

    #[test]
    fn output_data_coin_id() {
        let o = OutputData::Standard { address: [1u8; 32], value: 8, salt: [2u8; 32] };
        assert_eq!(o.coin_id(), Some(compute_coin_id(&[1u8; 32], 8, &[2u8; 32])));
    }

    #[test]
    fn confidential_coin_id_differs_from_standard() {
        let addr = [1u8; 32];
        let salt = [2u8; 32];
        let commitment = compute_value_commitment(8, &[0xAA; 32]);
        let std = OutputData::Standard { address: addr, value: 8, salt };
        let conf = OutputData::Confidential { address: addr, commitment, salt };
        assert_ne!(std.coin_id(), conf.coin_id());
        assert!(conf.coin_id().is_some());
    }

    #[test]
    fn confidential_value_is_zero() {
        let o = OutputData::Confidential {
            address: [1u8; 32],
            commitment: [2u8; 32],
            salt: [3u8; 32],
        };
        assert_eq!(o.value(), 0);
        assert!(o.is_confidential());
        assert!(o.commitment().is_some());
    }

    #[test]
    fn confidential_address_and_salt() {
        let addr = [0xAA; 32];
        let salt = [0xBB; 32];
        let o = OutputData::Confidential { address: addr, commitment: [0; 32], salt };
        assert_eq!(o.address(), addr);
        assert_eq!(o.salt(), salt);
    }

    #[test]
    fn confidential_hash_for_commitment_matches_coin_id() {
        let o = OutputData::Confidential {
            address: [1u8; 32],
            commitment: [2u8; 32],
            salt: [3u8; 32],
        };
        assert_eq!(o.hash_for_commitment(), o.coin_id().unwrap());
    }

    #[test]
    fn fee_zero_with_confidential_output() {
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&[0u8; 32]), value: 100, salt: [0u8; 32] , commitment: None }],
            witnesses: vec![Witness::sig(vec![])],
            outputs: vec![OutputData::Confidential {
                address: [0u8; 32],
                commitment: [0u8; 32],
                salt: [0u8; 32],
            }],
            salt: [0u8; 32],
        };
        assert_eq!(tx.fee(), 0);
    }

    #[test]
    fn standard_not_confidential() {
        let o = OutputData::Standard { address: [0; 32], value: 1, salt: [0; 32] };
        assert!(!o.is_confidential());
        assert!(o.commitment().is_none());
    }

    #[test]
    fn compute_value_commitment_deterministic() {
        let c1 = compute_value_commitment(100, &[0xAA; 32]);
        let c2 = compute_value_commitment(100, &[0xAA; 32]);
        assert_eq!(c1, c2);
    }

    #[test]
    fn compute_value_commitment_differs_by_value() {
        assert_ne!(
            compute_value_commitment(100, &[0xAA; 32]),
            compute_value_commitment(101, &[0xAA; 32]),
        );
    }

    #[test]
    fn compute_value_commitment_differs_by_blinding() {
        assert_ne!(
            compute_value_commitment(100, &[0xAA; 32]),
            compute_value_commitment(100, &[0xBB; 32]),
        );
    }

    #[test]
    fn input_reveal_coin_id_uses_address() {
        let pk = [0xAA; 32];
        let ir = InputReveal { predicate: Predicate::p2pk(&pk), value: 4, salt: [0u8; 32] , commitment: None };
        let expected_addr = Predicate::p2pk(&pk).address();
        assert_eq!(ir.coin_id(), compute_coin_id(&expected_addr, 4, &[0u8; 32]));
    }

    #[test]
    fn coinbase_output_coin_id() {
        let cb = CoinbaseOutput { address: [5u8; 32], value: 16, salt: [6u8; 32] };
        assert_eq!(cb.coin_id(), compute_coin_id(&[5u8; 32], 16, &[6u8; 32]));
    }

    // ── Transaction methods ─────────────────────────────────────────────

    #[test]
    fn commit_fee_is_zero() {
        let tx = Transaction::Commit { commitment: [0u8; 32], spam_nonce: 0 };
        assert_eq!(tx.fee(), 0);
        assert!(tx.input_coin_ids().is_empty());
        assert!(tx.output_coin_ids().is_empty());
    }

    #[test]
    fn reveal_fee_computed() {
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&[0u8; 32]), value: 10, salt: [0u8; 32] , commitment: None }],
            witnesses: vec![Witness::sig(vec![])],
            outputs: vec![OutputData::Standard { address: [0u8; 32], value: 8, salt: [0u8; 32] }],
            salt: [0u8; 32],
        };
        assert_eq!(tx.fee(), 2);
    }

    #[test]
    fn reveal_input_output_coin_ids() {
        let input = InputReveal { predicate: Predicate::p2pk(&[1u8; 32]), value: 8, salt: [2u8; 32] , commitment: None };
        let output = OutputData::Standard { address: [3u8; 32], value: 4, salt: [4u8; 32] };
        let tx = Transaction::Reveal {
            inputs: vec![input.clone()],
            witnesses: vec![Witness::sig(vec![])],
            outputs: vec![output.clone()],
            salt: [0u8; 32],
        };
        assert_eq!(tx.input_coin_ids(), vec![input.coin_id()]);
        assert_eq!(tx.output_coin_ids(), vec![output.coin_id().unwrap()]);
    }

    // ── State::genesis ──────────────────────────────────────────────────

    #[test]
    fn genesis_deterministic() {
        let (s1, cb1) = State::genesis();
        let (s2, cb2) = State::genesis();
        assert_eq!(s1.midstate, s2.midstate);
        assert_eq!(s1.height, 0);
        assert_eq!(cb1.len(), cb2.len());
        for (a, b) in cb1.iter().zip(cb2.iter()) {
            assert_eq!(a.coin_id(), b.coin_id());
        }
    }

    #[test]
    fn genesis_coinbase_values_sum_to_initial_reward() {
        let (_, cb) = State::genesis();
        let total: u64 = cb.iter().map(|c| c.value).sum();
        assert_eq!(total, INITIAL_REWARD);
    }

    #[test]
    fn genesis_coinbase_all_power_of_two() {
        let (_, cb) = State::genesis();
        for c in &cb {
            assert!(c.value.is_power_of_two(), "coinbase value {} not power of 2", c.value);
        }
    }
}
