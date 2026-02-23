use serde::{Deserialize, Serialize};
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Witness {
    /// Raw byte arrays pushed onto the stack before script execution.
    ScriptInputs(Vec<Vec<u8>>),
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

/// Compute a commitment hash that binds inputs to outputs.
///
/// commitment = BLAKE3(coin_id_1 || ... || new_coin_id_1 || ... || salt)
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
    #[serde(default)]
    pub commitment_heights: im::HashMap<[u8; 32], u64>,
}

impl State {
    pub fn genesis() -> (Self, Vec<CoinbaseOutput>) {
        use super::wots;

        // Bitcoin block anchor
        // Height: 935897
        // Hash: 00000000000000000000329a84d79877397ec0fa7c5aaa706a88e545daf599a5
        // Time: 2026-02-10 10:37:27 UTC
        const BITCOIN_BLOCK_HASH: &str = "00000000000000000000329a84d79877397ec0fa7c5aaa706a88e545daf599a5";
        const BITCOIN_BLOCK_HEIGHT: u64 = 935897;
        const BITCOIN_BLOCK_TIME: u64 = 1770719847;

        let anchor = hash(BITCOIN_BLOCK_HASH.as_bytes());

        const MERKLE_ROOT: &str = "6def077d292edb863bd64d2a8d8803ab12caf1eef9c76823ee01e9e47fce7d0d";
        let merkle_hash = hash(MERKLE_ROOT.as_bytes());

        // Genesis coinbase: INITIAL_REWARD decomposed into power-of-2 outputs.
        // Each output gets a deterministic seed and salt.
        let base_seed = hash_concat(&anchor, &merkle_hash);
        let denominations = decompose_value(INITIAL_REWARD);

        let genesis_coinbase: Vec<CoinbaseOutput> = denominations
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let seed = hash_concat(&base_seed, &(i as u64).to_le_bytes());
                let owner_pk = wots::keygen(&seed);
                let address = compute_address(&owner_pk);
                let salt = hash_concat(&seed, &[0xCBu8; 32]);
                CoinbaseOutput { address, value, salt }
            })
            .collect();

        // Initial difficulty target (very easy for testing)
        let target = [
            0x1f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
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
        };

        (state, genesis_coinbase)
    }
    pub fn header(&self) -> BatchHeader {
        BatchHeader {
            height: self.height,
            prev_midstate: [0u8; 32],
            post_tx_midstate: self.midstate,
            extension: Extension {
                nonce: 0,
                final_hash: self.midstate,
                checkpoints: vec![],
            },
            timestamp: self.timestamp,
            target: self.target,
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
                    for i in inputs {
                        hasher.update(&i.coin_id());
                    }
                    for o in outputs {
                        hasher.update(&o.coin_id());
                    }
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
 
        BatchHeader {
            height: 0, // Caller assigns this
            prev_midstate: self.prev_midstate,
            post_tx_midstate: midstate,
            extension: self.extension.clone(),
            timestamp: self.timestamp,
            target: self.target,
        }
    }
}

// ── Value-bearing data structures ───────────────────────────────────────────


/// Cleartext output data carried in a transaction.
/// Transmitted in the block, validated (value is power of 2), then discarded from state.
/// Only the resulting coin_id is stored in the UTXO set.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct OutputData {
    pub address: [u8; 32],
    pub value: u64,
    pub salt: [u8; 32],
}

impl OutputData {
    pub fn coin_id(&self) -> [u8; 32] {
        compute_coin_id(&self.address, self.value, &self.salt)
    }
}

/// Cleartext input preimage carried in a reveal transaction.
/// Proves what value a coin holds by revealing the preimage of its coin_id.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct InputReveal {
    pub predicate: Predicate,
    pub value: u64,
    pub salt: [u8; 32],
}

impl InputReveal {
    pub fn coin_id(&self) -> [u8; 32] {
        // The coin_id now commits to the Predicate's address hash
        compute_coin_id(&self.predicate.address(), self.value, &self.salt)
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
            Transaction::Reveal { outputs, .. } => outputs.iter().map(|o| o.coin_id()).collect(),
        }
    }

    /// Fee = sum(input values) - sum(output values). Zero for Commit.
    pub fn fee(&self) -> u64 {
        match self {
            Transaction::Commit { .. } => 0,
            Transaction::Reveal { inputs, outputs, .. } => {
                let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).unwrap_or(u64::MAX);
                let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value)).unwrap_or(u64::MAX);
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
    pub checkpoints: Vec<[u8; 32]>,
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchHeader {
    pub height: u64,
    pub prev_midstate: [u8; 32],
    pub post_tx_midstate: [u8; 32],
    pub extension: Extension,
    pub timestamp: u64,
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

pub const TARGET_BLOCK_TIME: u64 = 60;
/// ASERT half-life in seconds. Difficulty halves (or doubles) for every
/// half-life of drift between actual and ideal elapsed time since genesis.
pub const ASERT_HALF_LIFE: i64 = 4 * 60 * 60; // 4 hours (240 blocks)
/// Maximum number of recent timestamps kept for median-time-past validation
/// and timestamp window management in the node layer.
pub const DIFFICULTY_LOOKBACK: u64 = 60;
pub const MEDIAN_TIME_PAST_WINDOW: usize = 11;
pub const COMMITMENT_TTL: u64 = 100; 

/// Blocks behind tip before checkpoints are pruned from stored batches.
/// Pruning reclaims ~32 KB per block (~98% of batch storage). Pruned batches
/// remain fully verifiable via full-chain recomputation in verify_extension.
pub const PRUNE_DEPTH: u64 = 1000;
// ── Economics ───────────────────────────────────────────────────────────────

/// Blocks per year at TARGET_BLOCK_TIME seconds per block.
pub const BLOCKS_PER_YEAR: u64 = 365 * 24 * 3600 / TARGET_BLOCK_TIME; // 3_153_600

/// Initial block reward in value units.
pub const INITIAL_REWARD: u64 = 16;

pub const MAX_TX_INPUTS: usize = 256;
pub const MAX_TX_OUTPUTS: usize = 256;
/// Maximum size of a single signature in bytes.
/// WOTS = 576, MSS(height=20) = 1280. Pad for safety.
pub const MAX_SIGNATURE_SIZE: usize = 1_536;
/// Cap on total signature verifications per batch to prevent CPU exhaustion.
/// 100 txs × 256 inputs = 25,600 WOTS verifications ≈ 15B hashes — too much.
/// 1,024 total inputs keeps verification under ~600M hashes (~0.5s).
pub const MAX_BATCH_INPUTS: usize = 1_024;

/// Block reward value at a given height. Halves every BLOCKS_PER_YEAR, minimum 1.
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
// ============================================================
// ADD THIS ENTIRE BLOCK at the bottom of src/core/types.rs
// ============================================================

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
        let o = OutputData { address: [1u8; 32], value: 8, salt: [2u8; 32] };
        assert_eq!(o.coin_id(), compute_coin_id(&[1u8; 32], 8, &[2u8; 32]));
    }

    #[test]
    fn input_reveal_coin_id_uses_address() {
        let pk = [0xAA; 32];
        let ir = InputReveal { predicate: Predicate::p2pk(&pk), value: 4, salt: [0u8; 32] };
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&[0u8; 32]), value: 10, salt: [0u8; 32] }],
            witnesses: vec![Witness::sig(vec![])],
            outputs: vec![OutputData { address: [0u8; 32], value: 8, salt: [0u8; 32] }],
            salt: [0u8; 32],
        };
        assert_eq!(tx.fee(), 2);
    }

    #[test]
    fn reveal_input_output_coin_ids() {
        let input = InputReveal { predicate: Predicate::p2pk(&[1u8; 32]), value: 8, salt: [2u8; 32] };
        let output = OutputData { address: [3u8; 32], value: 4, salt: [4u8; 32] };
        let tx = Transaction::Reveal {
            inputs: vec![input.clone()],
            witnesses: vec![Witness::sig(vec![])],
            outputs: vec![output.clone()],
            salt: [0u8; 32],
        };
        assert_eq!(tx.input_coin_ids(), vec![input.coin_id()]);
        assert_eq!(tx.output_coin_ids(), vec![output.coin_id()]);
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
