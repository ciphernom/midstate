// ── Winternitz parameter w=16 ───────────────────────────────────────────────
//
// The message (32 bytes = 256 bits) is parsed as 16-bit digits:
//   256 / 16 = 16 message chains
//
// Checksum:
//   max_sum = 16 * 65535 = 1,048,560  (fits in 20 bits)
//   Encoded as 2 × 16-bit digits → 2 checksum chains
//
// Total: 16 + 2 = 18 chains
// Chain depth: 0..65535
// Signature size: 18 × 32 = 576 bytes  (was 1,088 at w=8)

pub const W: usize = 16;               // bits per digit
pub const MSG_CHAINS: usize = 16;      // 256 / W
pub const CHECKSUM_CHAINS: usize = 2;  // ceil(20 / 16)
pub const CHAINS: usize = MSG_CHAINS + CHECKSUM_CHAINS; // 18
pub const MAX_DIGIT: u32 = (1 << W) - 1; // 65_535
pub const SIG_SIZE: usize = CHAINS * 32; // 576 bytes

/// Generate a coin ID sequentially. 
/// Use this inside outer parallel loops (like MSS tree generation) to avoid thread thrashing.
pub fn keygen_seq(seed: &[u8; 32]) -> [u8; 32] {
    let mut inputs = [[0u8; 32]; CHAINS];
    for i in 0..CHAINS {
        inputs[i] = chain_sk(seed, i);
    }
    let targets = [MAX_DIGIT as usize; CHAINS];
    
    // Process all 18 chains simultaneously using SIMD
    let results = crate::core::wots_simd::process_wots_batch(&inputs, &targets);
    
    let mut endpoints = [[0u8; 32]; CHAINS];
    endpoints.copy_from_slice(&results);
    compress(&endpoints)
}
/// Derive chain secret key element: sk[i] = BLAKE3(seed || i)
fn chain_sk(seed: &[u8; 32], i: usize) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(seed);
    hasher.update(&(i as u32).to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Compress all chain endpoints into a single 32-byte coin ID.
fn compress(endpoints: &[[u8; 32]; CHAINS]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for ep in endpoints {
        hasher.update(ep);
    }
    *hasher.finalize().as_bytes()
}

/// Parse a 32-byte message into 16 × 16-bit digits (big-endian).
fn message_digits(msg: &[u8; 32]) -> [u32; MSG_CHAINS] {
    let mut digits = [0u32; MSG_CHAINS];
    for i in 0..MSG_CHAINS {
        digits[i] = u16::from_be_bytes([msg[i * 2], msg[i * 2 + 1]]) as u32;
    }
    digits
}

/// Compute the 2-digit checksum over the message digits.
///
/// checksum = Σ (MAX_DIGIT - d_i)  for all message digits
///
/// Max value: 16 × 65535 = 1,048,560 (0x000F_FFF0), fits in 20 bits.
/// Encoded big-endian into 2 × 16-bit digits.
fn checksum_digits(msg_digits: &[u32; MSG_CHAINS]) -> [u32; CHECKSUM_CHAINS] {
    let sum: u32 = msg_digits.iter().map(|&d| MAX_DIGIT - d).sum();
    [
        (sum >> 16) & 0xFFFF, // high 16 bits
        sum & 0xFFFF,         // low 16 bits
    ]
}

/// Combine message + checksum digits into the full digit vector.
fn all_digits(msg: &[u8; 32]) -> [u32; CHAINS] {
    let md = message_digits(msg);
    let cd = checksum_digits(&md);
    let mut digits = [0u32; CHAINS];
    digits[..MSG_CHAINS].copy_from_slice(&md);
    digits[MSG_CHAINS..].copy_from_slice(&cd);
    digits
}

/// Generate a coin ID (public key) from a seed (private key).
/// Because SIMD processing is so fast, spawning Rayon threads for 
/// only 18 items adds latency. We route directly to the SIMD sequence.
pub fn keygen(seed: &[u8; 32]) -> [u8; 32] {
    keygen_seq(seed)
}

/// Sign a 32-byte message with the given seed.
///
/// For each digit d_i, reveals hash^{d_i}(sk_i).
/// The verifier can hash the remaining (MAX_DIGIT - d_i) times to reach the endpoint.
pub fn sign(seed: &[u8; 32], message: &[u8; 32]) -> [[u8; 32]; CHAINS] {
    let digits = all_digits(message);
    
    let mut inputs = [[0u8; 32]; CHAINS];
    let mut targets = [0usize; CHAINS];
    
    for i in 0..CHAINS {
        inputs[i] = chain_sk(seed, i);
        targets[i] = digits[i] as usize;
    }
    
    // Compute the signature via the variable-masking SIMD processor
    let results = crate::core::wots_simd::process_wots_batch(&inputs, &targets);
    
    let mut sig = [[0u8; 32]; CHAINS];
    sig.copy_from_slice(&results);
    sig
}

/// Verify a WOTS signature against a message and coin ID.
///
/// For each digit d_i, hashes sig[i] exactly (MAX_DIGIT - d_i) times
/// and checks that all endpoints compress to the coin ID.
///
/// Average verification cost: CHAINS × (MAX_DIGIT / 2) ≈ 590K hashes.
/// With BLAKE3: ~0.5–1 ms on modern hardware.
pub fn verify(sig: &[[u8; 32]; CHAINS], message: &[u8; 32], coin_id: &[u8; 32]) -> bool {
    let digits = all_digits(message);
    let mut targets = [0usize; CHAINS];
    
    for i in 0..CHAINS {
        targets[i] = (MAX_DIGIT - digits[i]) as usize;
    }
    
    // Finish the hash chains simultaneously using SIMD
    let results = crate::core::wots_simd::process_wots_batch(sig, &targets);
    
    let mut endpoints = [[0u8; 32]; CHAINS];
    endpoints.copy_from_slice(&results);
    compress(&endpoints) == *coin_id
}

/// Serialize signature to bytes (18 × 32 = 576 bytes).
pub fn sig_to_bytes(sig: &[[u8; 32]; CHAINS]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SIG_SIZE);
    for chunk in sig {
        out.extend_from_slice(chunk);
    }
    out
}

/// Deserialize signature from bytes.
pub fn sig_from_bytes(bytes: &[u8]) -> Option<[[u8; 32]; CHAINS]> {
    if bytes.len() != SIG_SIZE {
        return None;
    }
    let mut sig = [[0u8; 32]; CHAINS];
    for (i, chunk) in bytes.chunks_exact(32).enumerate() {
        sig[i].copy_from_slice(chunk);
    }
    Some(sig)
}

// ── Key Reuse Punishment Burn Protocol ──────────────────────────────────────

use rayon::prelude::*;

/// A partially recovered WOTS secret key, extracted by observing key reuse.
///
/// # Reasoning
/// WOTS signatures reveal intermediate hashes in a hash chain. If a key signs
/// two different messages, different depths of the chain are exposed. By taking
/// the minimum (shallowest) depth exposed across all chains, an observer can
/// reconstruct a partial secret key. This struct holds that leaked key material.
#[derive(Clone, Debug)]
pub struct PartialSecretKey {
    /// Array of (revealed_depth, intermediate_hash) for each of the 18 chains.
    pub chains: [(u32, [u8; 32]); CHAINS],
}

impl PartialSecretKey {
    /// Initializes a partial secret key from a single intercepted signature.
    pub fn from_signature(sig: &[[u8; 32]; CHAINS], message: &[u8; 32]) -> Self {
        let digits = all_digits(message);
        let mut chains = [(0, [0u8; 32]); CHAINS];
        for i in 0..CHAINS {
            chains[i] = (digits[i], sig[i]);
        }
        Self { chains }
    }

    /// Merges a second intercepted signature, keeping the shallowest (most powerful)
    /// revealed hash for each chain.
    pub fn merge_signature(&mut self, sig: &[[u8; 32]; CHAINS], message: &[u8; 32]) {
        let digits = all_digits(message);
        for i in 0..CHAINS {
            if digits[i] < self.chains[i].0 {
                self.chains[i] = (digits[i], sig[i]);
            }
        }
    }

    /// Evaluates if this partial key contains enough depth to forge a signature
    /// for the given target message.
    pub fn can_sign(&self, message: &[u8; 32]) -> bool {
        let digits = all_digits(message);
        for i in 0..CHAINS {
            // We can only sign if the target message digit requires hashing
            // *down* the chain from our known starting point.
            if digits[i] < self.chains[i].0 {
                return false;
            }
        }
        true
    }

    /// Forges a valid signature for the target message using the leaked partial key.
    ///
    /// # Panics
    /// Panics if `can_sign(message)` is false.
    pub fn sign(&self, message: &[u8; 32]) -> [[u8; 32]; CHAINS] {
        let digits = all_digits(message);
        let mut sig = [[0u8; 32]; CHAINS];
        for i in 0..CHAINS {
            let (stored_depth, mut val) = self.chains[i];
            let target_depth = digits[i];
            assert!(target_depth >= stored_depth, "Attempted to forge signature above known depth");
            let iters = target_depth - stored_depth;
            for _ in 0..iters {
                val = crate::core::types::hash(&val);
            }
            sig[i] = val;
        }
        sig
    }

    /// Grinds a transaction salt until it produces a commitment hash that falls
    /// entirely within the known bounds of this partial secret key. 
    ///
    /// # Reasoning
    /// To punish an attacker who reuses a key, we must construct a valid transaction
    /// spending their funds to a burn address. Because we only control the `salt`,
    /// we use `rayon` to parallel-grind salts until `ℋ(inputs ⌢ burn_output ⌢ salt)`
    /// produces a digit sequence we can forge.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - psk contains the minimum revealed hash depths from multiple signatures.
    ///   - unspent_inputs are valid inputs corresponding to the leaked key.
    ///
    /// Post:
    ///   result = Some((commit_tx, reveal_tx)) ⇒
    ///     commit_tx.commitment = ℋ(inputs ⌢ burn_output ⌢ salt)
    ///     psk.can_sign(commit_tx.commitment) = true
    ///     reveal_tx.witness = valid forged signature over commitment
    ///   result = None ⇒
    ///     no salt found within search bounds (10,000,000 iterations)
    /// ```
    ///
    /// ```zed
    ///     ForgeBurnTransaction
    ///     --------------------
    ///     psk? : PartialSecretKey
    ///     inputs? : seq InputReveal
    ///     req_pow? : ℕ₃₂
    ///     commit_tx!, reveal_tx! : Transaction
    ///
    ///     pre  #inputs? > 0
    ///     post result = Some(commit_tx!, reveal_tx!) ⇒
    ///            ∃ salt ∈ 𝔹³² •
    ///              commitment = ℋ(inputs? ⌢ burn_output ⌢ salt) ∧
    ///              psk?.can_sign(commitment) = true ∧
    ///              reveal_tx!.witness = psk?.sign(commitment)
    ///     post result = None ⇒ true
    /// ```
    ///
    /// # Safety / Invariants
    /// Caps the search space to 10M iterations to guarantee bounded execution time
    /// and prevent the node's CPU from hanging indefinitely.
    pub fn forge_burn_transaction(
        &self,
        unspent_inputs: Vec<crate::core::types::InputReveal>,
        is_mss: bool,
        auth_path: Vec<[u8; 32]>,
        leaf_index: u64,
        wots_pk: [u8; 32],
        required_pow: u32,
        current_height: u64,
        header_hash: [u8; 32],
    ) -> Option<(crate::core::types::Transaction, crate::core::types::Transaction)> {
        let output = crate::core::types::OutputData::DataBurn {
            payload: b"PUNISHED FOR KEY REUSE".to_vec(),
            // 100% of the UTXO value is implicitly given to the miner as the transaction fee
            value_burned: 0, 
        };

        let input_ids: Vec<[u8; 32]> = unspent_inputs.iter().map(|i| i.coin_id()).collect();
        let output_ids = vec![output.hash_for_commitment()];

        // Parallel salt grinding to quickly find a forgeable commitment
        let found_salt = (0..10_000_000u64).into_par_iter().find_map_any(|i| {
            let mut salt = [0u8; 32];
            salt[0..8].copy_from_slice(&i.to_le_bytes());
            let commitment = crate::core::compute_commitment(&input_ids, &output_ids, &salt);
            if self.can_sign(&commitment) {
                Some((salt, commitment))
            } else {
                None
            }
        });

        let (salt, commitment) = found_salt?;
        let forged_wots = self.sign(&commitment);
        
        let witness = if is_mss {
            let mss_sig = crate::core::mss::MssSignature {
                leaf_index,
                wots_pk,
                wots_sig: forged_wots,
                auth_path,
            };
            crate::core::types::Witness::sig(mss_sig.to_bytes())
        } else {
            crate::core::types::Witness::sig(sig_to_bytes(&forged_wots))
        };

        let reveal_tx = if unspent_inputs.len() > 1 {
            crate::core::types::Transaction::Consolidate {
                inputs: unspent_inputs,
                witness,
                outputs: vec![output],
                salt,
            }
        } else {
            crate::core::types::Transaction::Reveal {
                inputs: unspent_inputs,
                witnesses: vec![witness],
                outputs: vec![output],
                salt,
            }
        };

        let spam_nonce = crate::core::transaction::mine_pow(&commitment, required_pow, current_height, header_hash);
        let commit_tx = crate::core::types::Transaction::Commit {
            commitment,
            spam_nonce,
        };

        Some((commit_tx, reveal_tx))
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash;

    #[test]
    fn sign_verify_round_trip() {
        let seed: [u8; 32] = [0x42; 32];
        let coin = keygen(&seed);
        let msg = hash(b"test message");
        let sig = sign(&seed, &msg);
        assert!(verify(&sig, &msg, &coin));
    }

    #[test]
    fn wrong_message_fails() {
        let seed: [u8; 32] = [0x42; 32];
        let coin = keygen(&seed);
        let msg = hash(b"test message");
        let sig = sign(&seed, &msg);
        let bad_msg = hash(b"wrong message");
        assert!(!verify(&sig, &bad_msg, &coin));
    }

    #[test]
    fn wrong_key_fails() {
        let seed: [u8; 32] = [0x42; 32];
        let msg = hash(b"test message");
        let sig = sign(&seed, &msg);
        let other_seed: [u8; 32] = [0x43; 32];
        let other_coin = keygen(&other_seed);
        assert!(!verify(&sig, &msg, &other_coin));
    }

    #[test]
    fn ser_deser_round_trip() {
        let seed: [u8; 32] = [0x42; 32];
        let msg = hash(b"test");
        let sig = sign(&seed, &msg);
        let bytes = sig_to_bytes(&sig);
        assert_eq!(bytes.len(), SIG_SIZE);
        assert_eq!(bytes.len(), 576);
        let sig2 = sig_from_bytes(&bytes).unwrap();
        assert_eq!(sig, sig2);
    }

    #[test]
    fn signature_size_is_576() {
        assert_eq!(CHAINS, 18);
        assert_eq!(SIG_SIZE, 576);
    }

    #[test]
    fn checksum_prevents_forgery() {
        let msg1 = [0u8; 32];
        let msg2 = {
            let mut m = [0u8; 32];
            m[0] = 1; 
            m
        };
        let d1 = all_digits(&msg1);
        let d2 = all_digits(&msg2);

        assert!(d2[0] > d1[0]);

        let cs_decreased = (MSG_CHAINS..CHAINS).any(|i| d2[i] < d1[i]);
        assert!(cs_decreased, "checksum must decrease when a message digit increases");
    }

    #[test]
    fn digit_extraction() {
        let mut msg = [0u8; 32];
        msg[0] = 0x01;
        msg[1] = 0x00;
        let digits = message_digits(&msg);
        assert_eq!(digits[0], 256);
        assert_eq!(digits[1], 0);
    }

    #[test]
    fn max_checksum_fits() {
        let msg = [0u8; 32];
        let md = message_digits(&msg);
        let cd = checksum_digits(&md);
        let sum: u32 = md.iter().map(|&d| MAX_DIGIT - d).sum();
        assert_eq!(sum, 16 * 65535); 
        assert!(cd[0] <= MAX_DIGIT);
        assert!(cd[1] <= MAX_DIGIT);
    }

    #[test]
    fn all_ff_message() {
        let msg = [0xff; 32];
        let md = message_digits(&msg);
        for &d in &md {
            assert_eq!(d, 65535);
        }
        let cd = checksum_digits(&md);
        assert_eq!(cd[0], 0);
        assert_eq!(cd[1], 0);
    }

    #[test]
    fn sig_from_bytes_wrong_length() {
        assert!(sig_from_bytes(&[0u8; 100]).is_none());
        assert!(sig_from_bytes(&[0u8; SIG_SIZE + 1]).is_none());
        assert!(sig_from_bytes(&[]).is_none());
    }

    #[test]
    fn keygen_deterministic() {
        let seed = [0x42u8; 32];
        assert_eq!(keygen(&seed), keygen(&seed));
    }

    #[test]
    fn different_seeds_different_keys() {
        assert_ne!(keygen(&[1u8; 32]), keygen(&[2u8; 32]));
    }
}
