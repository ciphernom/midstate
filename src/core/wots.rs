use super::types::hash;
use sha2::{Digest, Sha256};

pub const CHAINS: usize = 34; // 32 message bytes + 2 checksum bytes
pub const SIG_SIZE: usize = CHAINS * 32; // 1088 bytes

/// Iteratively hash `data` exactly `n` times.
fn hash_n(data: &[u8; 32], n: u8) -> [u8; 32] {
    let mut x = *data;
    for _ in 0..n {
        x = hash(&x);
    }
    x
}

/// Derive chain secret key element: sk[i] = hash(seed || i)
fn chain_sk(seed: &[u8; 32], i: usize) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed);
    hasher.update((i as u32).to_le_bytes());
    hasher.finalize().into()
}

/// Compress 34 chain endpoints into a single 32-byte coin ID.
fn compress(endpoints: &[[u8; 32]; CHAINS]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for ep in endpoints {
        hasher.update(ep);
    }
    hasher.finalize().into()
}

/// Compute checksum bytes (2 big-endian bytes) for a 32-byte message.
fn checksum(msg: &[u8; 32]) -> [u8; 2] {
    let sum: u32 = msg.iter().map(|&b| 255u32 - b as u32).sum();
    [(sum >> 8) as u8, sum as u8]
}

/// Generate a coin ID (public key) from a seed (private key).
pub fn keygen(seed: &[u8; 32]) -> [u8; 32] {
    let mut endpoints = [[0u8; 32]; CHAINS];
    for i in 0..CHAINS {
        let sk_i = chain_sk(seed, i);
        endpoints[i] = hash_n(&sk_i, 255);
    }
    compress(&endpoints)
}

/// Sign a 32-byte message with the given seed.
/// Returns CHAINS chain elements (each 32 bytes).
pub fn sign(seed: &[u8; 32], message: &[u8; 32]) -> Vec<[u8; 32]> {
    let cs = checksum(message);
    let mut digits = Vec::with_capacity(CHAINS);
    for &b in message.iter() {
        digits.push(b);
    }
    digits.push(cs[0]);
    digits.push(cs[1]);

    let mut sig = Vec::with_capacity(CHAINS);
    for (i, &d) in digits.iter().enumerate() {
        let sk_i = chain_sk(seed, i);
        sig.push(hash_n(&sk_i, d));
    }
    sig
}

/// Verify a WOTS signature against a message and coin ID.
pub fn verify(sig: &[[u8; 32]], message: &[u8; 32], coin_id: &[u8; 32]) -> bool {
    if sig.len() != CHAINS {
        return false;
    }

    let cs = checksum(message);
    let mut digits = Vec::with_capacity(CHAINS);
    for &b in message.iter() {
        digits.push(b);
    }
    digits.push(cs[0]);
    digits.push(cs[1]);

    let mut endpoints = [[0u8; 32]; CHAINS];
    for (i, &d) in digits.iter().enumerate() {
        let remaining = 255u8 - d;
        endpoints[i] = hash_n(&sig[i], remaining);
    }

    compress(&endpoints) == *coin_id
}

/// Serialize signature to bytes.
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
        let sig2 = sig_from_bytes(&bytes).unwrap();
        assert_eq!(sig, sig2);
    }
}
