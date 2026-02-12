use super::types::hash;

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

/// Iteratively hash `data` exactly `n` times using BLAKE3.
fn hash_n(data: &[u8; 32], n: u32) -> [u8; 32] {
    let mut x = *data;
    for _ in 0..n {
        x = hash(&x);
    }
    x
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
        (sum >> 16) & 0xFFFF, // high 16 bits (at most 0x000F = 15)
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
///
/// Each chain element is hashed MAX_DIGIT (65535) times to reach the endpoint.
/// Cost: CHAINS × MAX_DIGIT = 18 × 65535 ≈ 1.18M hashes.
/// With BLAKE3: ~1–2 ms on modern hardware.
pub fn keygen(seed: &[u8; 32]) -> [u8; 32] {
    let mut endpoints = [[0u8; 32]; CHAINS];
    for i in 0..CHAINS {
        let sk_i = chain_sk(seed, i);
        endpoints[i] = hash_n(&sk_i, MAX_DIGIT);
    }
    compress(&endpoints)
}

/// Sign a 32-byte message with the given seed.
///
/// For each digit d_i, reveals hash^{d_i}(sk_i).
/// The verifier can hash the remaining (MAX_DIGIT - d_i) times to reach the endpoint.
pub fn sign(seed: &[u8; 32], message: &[u8; 32]) -> Vec<[u8; 32]> {
    let digits = all_digits(message);
    let mut sig = Vec::with_capacity(CHAINS);
    for (i, &d) in digits.iter().enumerate() {
        let sk_i = chain_sk(seed, i);
        sig.push(hash_n(&sk_i, d));
    }
    sig
}

/// Verify a WOTS signature against a message and coin ID.
///
/// For each digit d_i, hashes sig[i] exactly (MAX_DIGIT - d_i) times
/// and checks that all endpoints compress to the coin ID.
///
/// Average verification cost: CHAINS × (MAX_DIGIT / 2) ≈ 590K hashes.
/// With BLAKE3: ~0.5–1 ms on modern hardware.
pub fn verify(sig: &[[u8; 32]], message: &[u8; 32], coin_id: &[u8; 32]) -> bool {
    if sig.len() != CHAINS {
        return false;
    }

    let digits = all_digits(message);
    let mut endpoints = [[0u8; 32]; CHAINS];
    for (i, &d) in digits.iter().enumerate() {
        let remaining = MAX_DIGIT - d;
        endpoints[i] = hash_n(&sig[i], remaining);
    }

    compress(&endpoints) == *coin_id
}

/// Serialize signature to bytes (18 × 32 = 576 bytes).
pub fn sig_to_bytes(sig: &[[u8; 32]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sig.len() * 32);
    for chunk in sig {
        out.extend_from_slice(chunk);
    }
    out
}

/// Deserialize signature from bytes.
pub fn sig_from_bytes(bytes: &[u8]) -> Option<Vec<[u8; 32]>> {
    if bytes.len() != SIG_SIZE {
        return None;
    }
    let mut sig = Vec::with_capacity(CHAINS);
    for chunk in bytes.chunks_exact(32) {
        sig.push(<[u8; 32]>::try_from(chunk).unwrap());
    }
    Some(sig)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Verify that increasing a message digit forces a checksum digit to decrease,
        // which requires hashing *forward* on the checksum chain (infeasible).
        let msg1 = [0u8; 32];
        let msg2 = {
            let mut m = [0u8; 32];
            m[0] = 1; // increase first digit
            m
        };
        let d1 = all_digits(&msg1);
        let d2 = all_digits(&msg2);

        // Message digit increased
        assert!(d2[0] > d1[0]);

        // At least one checksum digit decreased
        let cs_decreased = (MSG_CHAINS..CHAINS).any(|i| d2[i] < d1[i]);
        assert!(cs_decreased, "checksum must decrease when a message digit increases");
    }

    #[test]
    fn digit_extraction() {
        let mut msg = [0u8; 32];
        // Set first two bytes to 0x0100 = 256
        msg[0] = 0x01;
        msg[1] = 0x00;
        let digits = message_digits(&msg);
        assert_eq!(digits[0], 256);
        assert_eq!(digits[1], 0);
    }

    #[test]
    fn max_checksum_fits() {
        // All-zero message → max checksum
        let msg = [0u8; 32];
        let md = message_digits(&msg);
        let cd = checksum_digits(&md);
        let sum: u32 = md.iter().map(|&d| MAX_DIGIT - d).sum();
        assert_eq!(sum, 16 * 65535); // 1,048,560
        // Must fit in 2 × 16-bit digits
        assert!(cd[0] <= MAX_DIGIT);
        assert!(cd[1] <= MAX_DIGIT);
    }

    #[test]
    fn all_ff_message() {
        // All-0xFF message → all digits = 65535 → checksum = 0
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

    #[test]
    fn verify_wrong_length_sig_fails() {
        let seed = [0x42u8; 32];
        let coin = keygen(&seed);
        let msg = hash(b"test");
        // Too few chunks
        let short_sig: Vec<[u8; 32]> = vec![[0u8; 32]; CHAINS - 1];
        assert!(!verify(&short_sig, &msg, &coin));
    }



}
