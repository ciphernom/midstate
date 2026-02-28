//! Hierarchical Deterministic key derivation for post-quantum keys.
//!
//! Uses BIP39 mnemonics for the human-readable backup layer, then derives
//! an infinite tree of WOTS and MSS seeds using BLAKE3 with domain separation.
//!
//! ```text
//! 24 words  ──BIP39──▶  64-byte seed  ──BLAKE3──▶  32-byte master_seed
//!                                                       │
//!                                       ┌───────────────┼───────────────┐
//!                                       ▼               ▼               ▼
//!                                   WOTS/0          WOTS/1          MSS/0
//!                                  (receive)        (change)       (reusable)
//! ```
//!
//! # Security properties
//!
//! - **Domain separation**: WOTS and MSS derivation use different prefixes,
//!   so `derive_wots(0) ≠ derive_mss(0)` even from the same master seed.
//! - **Forward secrecy**: Revealing seed N does not compromise seed N+1
//!   (BLAKE3 is a PRF).
//! - **No BIP32 needed**: EC scalar tweaking is irrelevant for hash-based
//!   signatures. Direct `BLAKE3(prefix || master || index)` is simpler and
//!   equally secure.

use anyhow::Result;

/// Domain separation constants. These MUST never change across versions —
/// changing them silently derives different keys from the same mnemonic,
/// making old funds unrecoverable.
const WOTS_DOMAIN: &[u8] = b"midstate/wots/v1";
const MSS_DOMAIN: &[u8] = b"midstate/mss/v1";

/// Derives the master seed from a BIP39 mnemonic phrase.
///
/// Returns the 32-byte master seed. The mnemonic itself should be shown
/// to the user once at creation and never stored on disk — only the
/// derived master_seed is persisted (encrypted).
pub fn master_seed_from_mnemonic(phrase: &str) -> Result<[u8; 32]> {
    let mnemonic = bip39::Mnemonic::parse_in(bip39::Language::English, phrase)
        .map_err(|e| anyhow::anyhow!("Invalid seed phrase: {}", e))?;

    // BIP39 seed is 64 bytes; compress to 32 via BLAKE3
    let bip39_seed = mnemonic.to_seed("");
    Ok(*blake3::hash(&bip39_seed).as_bytes())
}

/// Generates a new 24-word BIP39 mnemonic and returns (master_seed, phrase).
///
/// The phrase MUST be shown to the user for backup. It is not stored anywhere.
pub fn generate_mnemonic() -> Result<([u8; 32], String)> {
    // 24 words = 256 bits of entropy
    let entropy: [u8; 32] = rand::random();
    let mnemonic = bip39::Mnemonic::from_entropy_in(bip39::Language::English, &entropy)
        .map_err(|e| anyhow::anyhow!("Mnemonic generation failed: {}", e))?;
    let phrase = mnemonic.to_string();
    let seed = master_seed_from_mnemonic(&phrase)?;
    Ok((seed, phrase))
}

/// Derive the Nth WOTS seed from a master seed.
///
/// Used for both receiving keys and change outputs. Every call to
/// `Wallet::generate_key()` or change-seed generation increments a
/// persistent counter to ensure no index is ever reused.
pub fn derive_wots_seed(master_seed: &[u8; 32], index: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(WOTS_DOMAIN);
    hasher.update(master_seed);
    hasher.update(&index.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// Derive the Nth MSS tree seed from a master seed.
pub fn derive_mss_seed(master_seed: &[u8; 32], index: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MSS_DOMAIN);
    hasher.update(master_seed);
    hasher.update(&index.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_wots_deterministic() {
        let master = [0xAA; 32];
        assert_eq!(derive_wots_seed(&master, 0), derive_wots_seed(&master, 0));
    }

    #[test]
    fn derive_wots_varies_by_index() {
        let master = [0xAA; 32];
        assert_ne!(derive_wots_seed(&master, 0), derive_wots_seed(&master, 1));
    }

    #[test]
    fn derive_wots_varies_by_master() {
        assert_ne!(
            derive_wots_seed(&[0xAA; 32], 0),
            derive_wots_seed(&[0xBB; 32], 0)
        );
    }

    #[test]
    fn wots_and_mss_domains_differ() {
        let master = [0xAA; 32];
        assert_ne!(
            derive_wots_seed(&master, 0),
            derive_mss_seed(&master, 0),
        );
    }

    #[test]
    fn mnemonic_round_trip() {
        let (seed1, phrase) = generate_mnemonic().unwrap();
        let seed2 = master_seed_from_mnemonic(&phrase).unwrap();
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn mnemonic_is_24_words() {
        let (_, phrase) = generate_mnemonic().unwrap();
        assert_eq!(phrase.split_whitespace().count(), 24);
    }

    #[test]
    fn invalid_mnemonic_rejected() {
        assert!(master_seed_from_mnemonic("not a valid mnemonic").is_err());
    }

    #[test]
    fn different_phrases_different_seeds() {
        // Two independently generated mnemonics must produce different seeds
        let (seed1, _) = generate_mnemonic().unwrap();
        let (seed2, _) = generate_mnemonic().unwrap();
        assert_ne!(seed1, seed2);
    }
}
