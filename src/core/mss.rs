//! # Merkle Signature Scheme (MSS)
//!
//! Wraps WOTS one-time keys in a binary Merkle tree so that a single
//! **master public key** (the tree root, 32 bytes) can authorise up to
//! `2^H` signatures.
//!
//! ```text
//!             root  ←  master public key
//!            /    \
//!          h1      h2
//!         /  \    /  \
//!       pk0  pk1 pk2  pk3   ← WOTS public keys
//! ```
//!
//! ## Signing (stateful)
//!
//! Each call to `sign()` consumes the next unused leaf.  The signer
//! **must** persist `next_leaf` — reusing a WOTS leaf is catastrophic.
//!
//! ## Signature contents
//!
//! `MssSignature` = WOTS sig (576 B) + WOTS pk (32 B) + leaf index (8 B)
//!                + auth path (H × 33 B).
//! At height 10: ~950 bytes total — compact for post-quantum.
//!
//! ## Integration with midstate
//!
//! The master public key **is** the coin ID.  On-chain, the verifier:
//!   1. Checks the WOTS sig against `sig.wots_pk`.
//!   2. Checks the Merkle path from `sig.wots_pk` to the coin ID.
//!
//! This means `Transaction::Reveal` signatures can carry either a raw
//! WOTS sig (legacy, one-time) or an `MssSignature` (reusable address).
//! The verifier distinguishes them by length.

use super::types::hash_concat;
use super::wots;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write}; 
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::path::PathBuf;
use rayon::prelude::*; 
// ── Config ──────────────────────────────────────────────────────────────────

/// Default tree height. 2^10 = 1024 signatures per master key.
pub const DEFAULT_HEIGHT: u32 = 10;

/// Max supported height. 2^20 ≈ 1M keys.
pub const MAX_HEIGHT: u32 = 20;

// ── Types ───────────────────────────────────────────────────────────────────

/// Full MSS keypair (private — stored in wallet).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MssKeypair {
    pub height: u32,
    /// All WOTS seeds derive from this: seed_i = BLAKE3(master_seed || i).
    pub master_seed: [u8; 32],
    /// 1-indexed binary tree.  tree[1] = root, leaves at [2^H .. 2^{H+1}).
    pub tree: Vec<[u8; 32]>,
    /// Next unused leaf (0-based among leaves).
    pub next_leaf: u64,
    /// Cached root = tree[1].
    pub master_pk: [u8; 32],
}

pub type MasterPublicKey = [u8; 32];

/// An MSS signature: WOTS sig + Merkle auth path to root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MssSignature {
    pub leaf_index: u64,
    /// WOTS public key for this leaf (Merkle-verified against master PK).
    pub wots_pk: [u8; 32],
    /// The WOTS signature 
    pub wots_sig: [[u8; 32]; wots::CHAINS],
    /// Auth path: H sibling hashes from leaf to root.
    pub auth_path: Vec<AuthNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthNode {
    pub hash: [u8; 32],
    /// true → sibling is on the right (we are the left child).
    pub is_right: bool,
}

// ── Key generation ──────────────────────────────────────────────────────────

/// Derive WOTS seed for leaf `i`.
fn derive_wots_seed(master_seed: &[u8; 32], index: u64) -> [u8; 32] {
    hash_concat(master_seed, &index.to_le_bytes())
}

/// Generate an MSS keypair.
///
/// Cost: 2^height WOTS key generations (~1-2 ms each at w=16).
/// Height 10 ≈ 1-2 s, height 16 ≈ 1-2 min.
/// Generate an MSS keypair with progress tracking and chunked checkpointing.
pub fn keygen(master_seed: &[u8; 32], height: u32) -> Result<MssKeypair> {
    if height == 0 || height > MAX_HEIGHT {
        bail!("height must be in 1..={}", MAX_HEIGHT);
    }

    let num_leaves = 1u64 << height;
    let tree_size = (num_leaves * 2) as usize;
    let mut tree = vec![[0u8; 32]; tree_size];
    let leaf_start = num_leaves as usize;

    #[cfg(target_arch = "wasm32")]
    {
        // Wasm-safe: Pure sequential math, no OS features, no threading
        for i in 0..num_leaves as usize {
            let global_idx = i as u64;
            let seed = derive_wots_seed(master_seed, global_idx);
            tree[leaf_start + i] = wots::keygen_seq(&seed);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Native: Multi-threaded with Rayon, checkpoints, and time tracking
        let checkpoint_path = std::path::PathBuf::from(format!("mss_h{}.checkpoint", height));
        let mut leaves_completed = 0usize;

        if checkpoint_path.exists() {
            if let Ok(file) = std::fs::File::open(&checkpoint_path) {
                let reader = std::io::BufReader::new(file);
                if let Ok(saved_leaves) = bincode::deserialize_from::<_, Vec<[u8; 32]>>(reader) {
                    if saved_leaves.len() == num_leaves as usize {
                        tree[leaf_start..].copy_from_slice(&saved_leaves);
                        leaves_completed = saved_leaves.iter().filter(|&&l| l != [0u8; 32]).count();
                        tracing::info!("Loaded checkpoint for height {}. Resuming from {} leaves.", height, leaves_completed);
                    }
                }
            }
        }

        let start_time = std::time::Instant::now();
        let total_finished = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(leaves_completed));
        let chunk_size = 4096;

        tracing::info!("Starting MSS Leaf Generation (Height {})...", height);

        for chunk_start_idx in (leaves_completed..num_leaves as usize).step_by(chunk_size) {
            let chunk_end_idx = (chunk_start_idx + chunk_size).min(num_leaves as usize);
            
            {
                let chunk = &mut tree[leaf_start + chunk_start_idx .. leaf_start + chunk_end_idx];
                use rayon::prelude::*;
                chunk.par_iter_mut().enumerate().for_each(|(i, leaf)| {
                    let global_idx = (chunk_start_idx + i) as u64;
                    let seed = derive_wots_seed(master_seed, global_idx);
                    *leaf = wots::keygen_seq(&seed);
                    total_finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                });
            }

            let count = total_finished.load(std::sync::atomic::Ordering::Relaxed);
            let pct = (count as f64 / num_leaves as f64) * 100.0;
            let leaves_this_session = count - leaves_completed;
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { leaves_this_session as f64 / elapsed } else { 0.0 };
            let eta = if rate > 0.0 { (num_leaves as usize - count) as f64 / rate } else { 0.0 };
            
            use std::io::Write;
            print!(
                "\r[MSS] Progress: {:.2}% ({}/{}) | Rate: {:.1} leaf/s | ETA: {:.1}m    ",
                pct, count, num_leaves, rate, eta / 60.0
            );
            let _ = std::io::stdout().flush();

            if (chunk_start_idx / chunk_size) % 10 == 0 || count == num_leaves as usize {
                save_checkpoint(&checkpoint_path, &tree[leaf_start..]);
            }
        }
        println!();

        if checkpoint_path.exists() {
            let _ = std::fs::remove_file(&checkpoint_path);
        }
        tracing::info!("MSS Keypair generated in {:?}", start_time.elapsed());
    }

    // Build Internal Nodes
    for i in (1..leaf_start).rev() {
        tree[i] = hash_concat(&tree[2 * i], &tree[2 * i + 1]);
    }

    Ok(MssKeypair {
        height,
        master_seed: *master_seed,
        master_pk: tree[1],
        tree,
        next_leaf: 0,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn save_checkpoint(path: &std::path::Path, leaves: &[[u8; 32]]) {
    if let Ok(file) = std::fs::File::create(path) {
        let writer = std::io::BufWriter::new(file);
        let _ = bincode::serialize_into(writer, leaves);
    }
}

/// Generate a random MSS keypair with default height.
pub fn keygen_random() -> Result<MssKeypair> {
    let seed: [u8; 32] = rand::random();
    keygen(&seed, DEFAULT_HEIGHT)
}

// ── Signing ─────────────────────────────────────────────────────────────────

impl MssKeypair {
    pub fn remaining(&self) -> u64 { (1u64 << self.height) - self.next_leaf }
    pub fn used(&self) -> u64 { self.next_leaf }
    pub fn public_key(&self) -> MasterPublicKey { self.master_pk }

    /// Sign a 32-byte message, consuming the next leaf.
    pub fn sign(&mut self, message: &[u8; 32]) -> Result<MssSignature> {
        let num_leaves = 1u64 << self.height;
        if self.next_leaf >= num_leaves {
            bail!("MSS tree exhausted: all {} leaves used", num_leaves);
        }

        let leaf_idx = self.next_leaf;
        self.next_leaf += 1;

        let wots_seed = derive_wots_seed(&self.master_seed, leaf_idx);
        let wots_pk = wots::keygen(&wots_seed);
        let wots_sig = wots::sign(&wots_seed, message);
        let auth_path = self.auth_path(leaf_idx);

        Ok(MssSignature { leaf_index: leaf_idx, wots_pk, wots_sig, auth_path })
    }

    /// Merkle auth path for leaf `leaf_idx`.
    fn auth_path(&self, leaf_idx: u64) -> Vec<AuthNode> {
        let num_leaves = 1u64 << self.height;
        let mut path = Vec::with_capacity(self.height as usize);
        let mut node = (num_leaves + leaf_idx) as usize; // 1-indexed

        for _ in 0..self.height {
            let (sibling_hash, is_right) = if node % 2 == 0 {
                (self.tree[node + 1], true) // we're left, sibling right
            } else {
                (self.tree[node - 1], false) // we're right, sibling left
            };
            path.push(AuthNode { hash: sibling_hash, is_right });
            node /= 2;
        }
        path
    }

    pub fn peek_next_leaf(&self) -> u64 { self.next_leaf }

    /// Force leaf index (recovery only — reuse breaks security).
    pub fn set_next_leaf(&mut self, idx: u64) { self.next_leaf = idx; }
}

// ── Verification ────────────────────────────────────────────────────────────

/// Verify MSS signature: check WOTS sig, then Merkle path to master PK.
pub fn verify(sig: &MssSignature, message: &[u8; 32], master_pk: &MasterPublicKey) -> bool {
    // SECURITY FIX: Prevent high-bit malleability
    let capacity = 1u64.checked_shl(sig.auth_path.len() as u32).unwrap_or(u64::MAX);
    if sig.leaf_index >= capacity {
        return false; 
    }
    // 1. WOTS
    if !wots::verify(&sig.wots_sig, message, &sig.wots_pk) {
        return false;
    }

    // 2. Merkle auth path - STRICTLY ENFORCED BY LEAF INDEX
    let mut current = sig.wots_pk;
    let mut current_idx = sig.leaf_index;

    for node in &sig.auth_path {
        // If current_idx is even, we are the left child, so sibling is on the right.
        let expected_is_right = current_idx % 2 == 0;
        
        // Prevent forged paths and leaf_index poisoning
        if node.is_right != expected_is_right {
            return false;
        }

        current = if node.is_right {
            hash_concat(&current, &node.hash)
        } else {
            hash_concat(&node.hash, &current)
        };
        
        current_idx /= 2;
    }
    current == *master_pk
}

// ── Serialization ───────────────────────────────────────────────────────────

impl MssSignature {
    /// Layout: leaf_index(8) || wots_pk(32) || wots_sig(576) ||
    ///         auth_len(4) || auth_path(len × 33)
    pub fn to_bytes(&self) -> Vec<u8> {
        let wots_bytes = wots::sig_to_bytes(&self.wots_sig);
        let auth_len = self.auth_path.len();
        let mut buf = Vec::with_capacity(8 + 32 + wots_bytes.len() + 4 + auth_len * 33);

        buf.extend_from_slice(&self.leaf_index.to_le_bytes());
        buf.extend_from_slice(&self.wots_pk);
        buf.extend_from_slice(&wots_bytes);
        buf.extend_from_slice(&(auth_len as u32).to_le_bytes());
        for node in &self.auth_path {
            buf.extend_from_slice(&node.hash);
            buf.push(if node.is_right { 1 } else { 0 });
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let min = 8 + 32 + wots::SIG_SIZE + 4;
        if data.len() < min { bail!("MSS signature too short"); }

        let leaf_index = u64::from_le_bytes(data[..8].try_into().unwrap());
        let wots_pk: [u8; 32] = data[8..40].try_into().unwrap();
        let wots_sig = wots::sig_from_bytes(&data[40..40 + wots::SIG_SIZE])
            .ok_or_else(|| anyhow::anyhow!("invalid WOTS sig in MSS"))?;

        let ao = 40 + wots::SIG_SIZE;
        let auth_len = u32::from_le_bytes(data[ao..ao + 4].try_into().unwrap()) as usize;
        if auth_len > MAX_HEIGHT as usize {
            bail!("MSS auth path too long: {} > {}", auth_len, MAX_HEIGHT);
        }
        let ps = ao + 4;
        if data.len() < ps + auth_len * 33 { bail!("MSS signature truncated"); }

        let auth_path = (0..auth_len).map(|i| {
            let o = ps + i * 33;
            AuthNode {
                hash: data[o..o + 32].try_into().unwrap(),
                is_right: data[o + 32] != 0,
            }
        }).collect();

        Ok(Self { leaf_index, wots_pk, wots_sig, auth_path })
    }

    pub fn size(&self) -> usize {
        8 + 32 + wots::SIG_SIZE + 4 + self.auth_path.len() * 33
    }
}


// --- special web wallet function
#[cfg(target_arch = "wasm32")]
pub fn keygen_with_progress<F>(master_seed: &[u8; 32], height: u32, mut progress_cb: F) -> Result<MssKeypair>
where
    F: FnMut(u32, u32),
{
    use anyhow::bail;
    if height == 0 || height > MAX_HEIGHT {
        bail!("height must be in 1..={}", MAX_HEIGHT);
    }

    let num_leaves = 1u64 << height;
    let tree_size = (num_leaves * 2) as usize;
    let mut tree = vec![[0u8; 32]; tree_size];
    let leaf_start = num_leaves as usize;

    for i in 0..num_leaves as usize {
        let global_idx = i as u64;
        let seed = derive_wots_seed(master_seed, global_idx);
        tree[leaf_start + i] = wots::keygen_seq(&seed);
        
        // Fire the callback every 2 leaves, and on the final leaf
        if i % 2 == 0 || i == (num_leaves as usize - 1) {
            progress_cb((i + 1) as u32, num_leaves as u32);
        }
    }

    // Build Internal Nodes
    for i in (1..leaf_start).rev() {
        tree[i] = hash_concat(&tree[2 * i], &tree[2 * i + 1]);
    }

    Ok(MssKeypair {
        height,
        master_seed: *master_seed,
        master_pk: tree[1],
        tree,
        next_leaf: 0,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::hash;
    
    fn test_seed() -> [u8; 32] { hash(b"test mss master seed") }

    #[test]
    fn keygen_valid() {
        let kp = keygen(&test_seed(), 4).unwrap();
        assert_eq!(kp.height, 4);
        assert_eq!(kp.remaining(), 16);
        assert_ne!(kp.master_pk, [0u8; 32]);
    }

    #[test]
    fn sign_verify() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let msg = hash(b"hello");
        let sig = kp.sign(&msg).unwrap();
        assert!(verify(&sig, &msg, &kp.public_key()));
        assert_eq!(kp.remaining(), 15);
    }

    #[test]
    fn sign_all_leaves() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let pk = kp.public_key();
        for i in 0..16u8 {
            let msg = hash(&[i]);
            let sig = kp.sign(&msg).unwrap();
            assert!(verify(&sig, &msg, &pk));
        }
        assert_eq!(kp.remaining(), 0);
    }

    #[test]
    fn exhausted_errors() {
        let mut kp = keygen(&test_seed(), 2).unwrap();
        for _ in 0..4 { kp.sign(&hash(b"m")).unwrap(); }
        assert!(kp.sign(&hash(b"one more")).is_err());
    }

    #[test]
    fn wrong_message_fails() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let sig = kp.sign(&hash(b"correct")).unwrap();
        assert!(!verify(&sig, &hash(b"wrong"), &kp.public_key()));
    }

    #[test]
    fn wrong_master_pk_fails() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let sig = kp.sign(&hash(b"test")).unwrap();
        let other = keygen(&hash(b"other"), 4).unwrap();
        assert!(!verify(&sig, &hash(b"test"), &other.public_key()));
    }

    #[test]
    fn ser_deser_round_trip() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let msg = hash(b"serialize");
        let sig = kp.sign(&msg).unwrap();
        let bytes = sig.to_bytes();
        let sig2 = MssSignature::from_bytes(&bytes).unwrap();
        assert!(verify(&sig2, &msg, &kp.public_key()));
    }

    #[test]
    fn deterministic_keygen() {
        let s = test_seed();
        assert_eq!(keygen(&s, 4).unwrap().master_pk, keygen(&s, 4).unwrap().master_pk);
    }

    #[test]
    fn different_leaves_both_verify() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let msg = hash(b"same");
        let s1 = kp.sign(&msg).unwrap();
        let s2 = kp.sign(&msg).unwrap();
        assert_ne!(s1.leaf_index, s2.leaf_index);
        let pk = kp.public_key();
        assert!(verify(&s1, &msg, &pk));
        assert!(verify(&s2, &msg, &pk));
    }

    #[test]
    fn sig_size_reasonable() {
        let mut kp = keygen(&test_seed(), 2).unwrap();
        let sig = kp.sign(&hash(b"t")).unwrap();
        // 8 + 32 + 576 + 4 + 2*33 = 686
        assert_eq!(sig.size(), 686);
    }
    #[test]
    fn keygen_rejects_zero_height() {
        assert!(keygen(&test_seed(), 0).is_err());
    }

    #[test]
    fn keygen_rejects_excessive_height() {
        assert!(keygen(&test_seed(), MAX_HEIGHT + 1).is_err());
    }

    #[test]
    fn from_bytes_truncated_fails() {
        let mut kp = keygen(&test_seed(), 4).unwrap();
        let sig = kp.sign(&hash(b"test")).unwrap();
        let bytes = sig.to_bytes();
        assert!(MssSignature::from_bytes(&bytes[..bytes.len() - 10]).is_err());
    }

    #[test]
    fn from_bytes_too_short_fails() {
        assert!(MssSignature::from_bytes(&[0u8; 10]).is_err());
    }
    
}
