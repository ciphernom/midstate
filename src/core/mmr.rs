//! # Merkle Mountain Range (MMR) + UTXO Accumulator
//!
//! Two structures for replacing the `HashSet<[u8; 32]>` coin set:
//!
//! 1. **MMR** — append-only log of state transitions for light-client proofs.
//!    O(1) amortised append, O(log n) inclusion proofs, O(log n) root via peaks.
//!
//! 2. **UtxoAccumulator** — Merkle-committed mutable UTXO set.
//!    Sorted-vec backed today (O(n) insert/remove, trivially correct).
//!    Drop-in replacement: swap `State.coins: HashSet` → `UtxoAccumulator`,
//!    keep the same `.contains()` / `.insert()` / `.remove()` API.
//!    For millions of coins, swap internals for a Sparse Merkle Tree or Utreexo.

use super::types::hash_concat;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
//  MMR
// ═══════════════════════════════════════════════════════════════════════════

/// Height of the node at MMR position `pos` (0-indexed).
/// Leaf = height 0.
fn pos_height(mut pos: u64) -> u32 {
    // Find the height of the smallest perfect tree that contains 'pos'
    let mut h = 0;
    while pos >= (1 << (h + 1)) - 1 {
        h += 1;
    }
    
    // We iterate down from the top of that tree to find the height of 'pos'
    let mut cur_h = h;
    let mut cur_size = (1 << (cur_h + 1)) - 1;
    
    loop {
        // If pos is the root of the current subtree, return its height
        if pos == cur_size - 1 {
            return cur_h;
        }
        
        // Otherwise, descend
        cur_h -= 1;
        let left_size = (1 << (cur_h + 1)) - 1;
        
        // If pos is in the right child, shift it relative to the right child
        if pos >= left_size {
            pos -= left_size;
        }
        // If pos is in the left child, we just process it with the reduced height
        
        cur_size = left_size;
    }
}

/// Total nodes in an MMR with `n` leaves: `2n − popcount(n)`.
pub fn mmr_size(n: u64) -> u64 {
    if n == 0 { 0 } else { 2 * n - (n.count_ones() as u64) }
}

/// Peak positions in an MMR of `size` nodes.
pub fn peaks(size: u64) -> Vec<u64> {
    let mut result = Vec::new();
    let mut remaining = size;
    let mut offset = 0u64;

    while remaining > 0 {
        let mut h = 1u32;
        while (1u64 << (h + 1)) - 1 <= remaining {
            h += 1;
        }
        let tree_size = (1u64 << h) - 1;
        if tree_size > remaining { break; }
        result.push(offset + tree_size - 1);
        offset += tree_size;
        remaining -= tree_size;
    }
    result
}

/// A Merkle Mountain Range backed by a flat vec of hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerkleMountainRange {
    nodes: Vec<[u8; 32]>,
    leaf_count: u64,
}

impl MerkleMountainRange {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), leaf_count: 0 }
    }

    pub fn leaf_count(&self) -> u64 { self.leaf_count }
    pub fn size(&self) -> u64 { self.nodes.len() as u64 }

    /// Append a leaf, auto-merging complete pairs. Returns its MMR position.
    pub fn append(&mut self, leaf_hash: &[u8; 32]) -> u64 {
        let pos = self.nodes.len() as u64;
        self.nodes.push(*leaf_hash);
        self.leaf_count += 1;

        let mut current_pos = pos;
        let mut current_height = 0u32;

        loop {
            let left_sibling_size = (1u64 << (current_height + 1)) - 1;
            if current_pos < left_sibling_size { break; }
            let left_pos = current_pos - left_sibling_size;
            if pos_height(left_pos) != current_height { break; }

            let parent_hash = hash_concat(
                &self.nodes[left_pos as usize],
                &self.nodes[current_pos as usize],
            );
            let parent_pos = self.nodes.len() as u64;
            self.nodes.push(parent_hash);
            current_pos = parent_pos;
            current_height += 1;
        }
        pos
    }

    /// Bag peaks right-to-left into a single root.
    pub fn root(&self) -> [u8; 32] {
        let peak_positions = peaks(self.nodes.len() as u64);
        if peak_positions.is_empty() { return [0u8; 32]; }

        let mut root = self.nodes[*peak_positions.last().unwrap() as usize];
        for &pos in peak_positions.iter().rev().skip(1) {
            root = hash_concat(&self.nodes[pos as usize], &root);
        }
        root
    }

    /// Inclusion proof for the leaf at `leaf_pos`.
    pub fn prove(&self, leaf_pos: u64) -> Result<MmrProof> {
        let sz = self.nodes.len() as u64;
        if leaf_pos >= sz { bail!("position {} out of range (size {})", leaf_pos, sz); }
        if pos_height(leaf_pos) != 0 { bail!("position {} is not a leaf", leaf_pos); }

        let peak_positions = peaks(sz);
        let mut siblings = Vec::new();
        let mut pos = leaf_pos;
        let mut height = 0u32;

        loop {
            if peak_positions.contains(&pos) { break; }

            let right_sibling = pos + (1u64 << (height + 1)) - 1;
            if right_sibling < sz && pos_height(right_sibling) == height {
                siblings.push(ProofElement {
                    hash: self.nodes[right_sibling as usize],
                    is_right: true,
                });
                pos = right_sibling + 1;
            } else {
                let left_sibling = pos - ((1u64 << (height + 1)) - 1);
                siblings.push(ProofElement {
                    hash: self.nodes[left_sibling as usize],
                    is_right: false,
                });
                pos += 1;
            }
            height += 1;
        }

        let our_peak = pos;
        let peak_index = peak_positions.iter().position(|&p| p == our_peak)
            .ok_or_else(|| anyhow::anyhow!("internal error: peak not found"))?;

        Ok(MmrProof {
            leaf_pos,
            siblings,
            peak_hashes: peak_positions.iter().map(|&p| self.nodes[p as usize]).collect(),
            peak_index,
            mmr_size: sz,
        })
    }

    pub fn get(&self, pos: u64) -> Option<&[u8; 32]> {
        self.nodes.get(pos as usize)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofElement {
    pub hash: [u8; 32],
    pub is_right: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MmrProof {
    pub leaf_pos: u64,
    pub siblings: Vec<ProofElement>,
    pub peak_hashes: Vec<[u8; 32]>,
    pub peak_index: usize,
    pub mmr_size: u64,
}

/// Verify an MMR inclusion proof.
pub fn verify_mmr_proof(leaf_hash: &[u8; 32], proof: &MmrProof, expected_root: &[u8; 32]) -> bool {
    let mut current = *leaf_hash;
    for elem in &proof.siblings {
        current = if elem.is_right {
            hash_concat(&current, &elem.hash)
        } else {
            hash_concat(&elem.hash, &current)
        };
    }

    if proof.peak_index >= proof.peak_hashes.len() { return false; }
    if current != proof.peak_hashes[proof.peak_index] { return false; }

    if proof.peak_hashes.is_empty() { return false; }
    let mut root = *proof.peak_hashes.last().unwrap();
    for peak in proof.peak_hashes.iter().rev().skip(1) {
        root = hash_concat(peak, &root);
    }
    root == *expected_root
}

// ═══════════════════════════════════════════════════════════════════════════
//  UTXO Accumulator  (Sparse Merkle Tree)
// ═══════════════════════════════════════════════════════════════════════════

static EMPTY_HASHES: std::sync::OnceLock<Vec<[u8; 32]>> = std::sync::OnceLock::new();

fn get_empty_hash(height: usize) -> [u8; 32] {
    let hashes = EMPTY_HASHES.get_or_init(|| {
        let mut h = Vec::with_capacity(257);
        h.push([0u8; 32]);
        for i in 0..257 {
            h.push(hash_concat(&h[i], &h[i]));
        }
        h
    });
    hashes[height]
}

/// Sparse Merkle Tree backed UTXO accumulator.
/// Sorted vec kept in parallel for iteration/contains.
/// SMT nodes stored as (height, path_key) -> hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtxoAccumulator {
    coins: Vec<[u8; 32]>,
    #[serde(skip)]
    nodes: std::collections::HashMap<(u16, [u8; 32]), [u8; 32]>,
}

impl PartialEq for UtxoAccumulator {
    fn eq(&self, other: &Self) -> bool {
        self.coins == other.coins
    }
}

impl Eq for UtxoAccumulator {}

impl UtxoAccumulator {
    pub fn new() -> Self {
        Self { coins: Vec::new(), nodes: std::collections::HashMap::new() }
    }

    /// Rebuild the SMT node cache from the coin list (after deserialization).
    pub fn rebuild_tree(&mut self) {
        self.nodes.clear();
        for i in 0..self.coins.len() {
            let coin = self.coins[i];
            self.update_path(coin, true);
        }
    }

    pub fn from_set(coins: impl IntoIterator<Item = [u8; 32]>) -> Self {
        let mut acc = Self::new();
        for c in coins { acc.insert(c); }
        acc
    }

    pub fn len(&self) -> usize { self.coins.len() }
    pub fn is_empty(&self) -> bool { self.coins.is_empty() }

    pub fn contains(&self, coin: &[u8; 32]) -> bool {
        self.coins.binary_search(coin).is_ok()
    }

    pub fn insert(&mut self, coin: [u8; 32]) -> bool {
        if self.contains(&coin) { return false; }
        let idx = self.coins.binary_search(&coin).unwrap_err();
        self.coins.insert(idx, coin);
        self.update_path(coin, true);
        true
    }

    pub fn remove(&mut self, coin: &[u8; 32]) -> bool {
        if let Ok(idx) = self.coins.binary_search(coin) {
            self.coins.remove(idx);
            self.update_path(*coin, false);
            true
        } else {
            false
        }
    }

    pub fn root(&mut self) -> [u8; 32] {
        self.get_node(256, [0u8; 32])
    }

    pub fn prove(&self, coin: &[u8; 32]) -> Result<UtxoProof> {
        if !self.contains(coin) {
            bail!("coin not in accumulator");
        }

        let mut siblings = Vec::with_capacity(256);
        let mut current_path = *coin;

        for h in 0usize..256 {
            let bit = get_bit(coin, h);
            let mut sibling_path = current_path;
            flip_bit(&mut sibling_path, h);
            mask_lower_bits(&mut sibling_path, h);

            let sibling_hash = self.get_node(h as u16, sibling_path);
            siblings.push(ProofElement {
                hash: sibling_hash,
                is_right: bit == 0,
            });

            mask_lower_bits(&mut current_path, h + 1);
        }

        Ok(UtxoProof {
            leaf_index: 0,
            leaf_count: self.coins.len(),
            siblings,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8; 32]> { self.coins.iter() }
    pub fn into_vec(self) -> Vec<[u8; 32]> { self.coins }

    fn get_node(&self, height: u16, path: [u8; 32]) -> [u8; 32] {
        self.nodes.get(&(height, path))
            .copied()
            .unwrap_or_else(|| get_empty_hash(height as usize))
    }

    fn update_path(&mut self, coin: [u8; 32], inserting: bool) {
        if inserting {
            self.nodes.insert((0u16, coin), coin);
        } else {
            self.nodes.remove(&(0u16, coin));
        }

        let mut current_path = coin;

        for h in 0usize..256 {
            let bit = get_bit(&coin, h);

            let mut sibling_path = current_path;
            flip_bit(&mut sibling_path, h);
            mask_lower_bits(&mut sibling_path, h);

            let current_hash = self.get_node(h as u16, current_path);
            let sibling_hash = self.get_node(h as u16, sibling_path);

            let parent_hash = if bit == 0 {
                hash_concat(&current_hash, &sibling_hash)
            } else {
                hash_concat(&sibling_hash, &current_hash)
            };

            mask_lower_bits(&mut current_path, h + 1);

            let empty = get_empty_hash(h + 1);
            if parent_hash == empty {
                self.nodes.remove(&((h + 1) as u16, current_path));
            } else {
                self.nodes.insert(((h + 1) as u16, current_path), parent_hash);
            }
        }
    }
}

fn get_bit(bytes: &[u8; 32], bit_index: usize) -> u8 {
    let byte_idx = 31 - (bit_index / 8);
    let bit_offset = bit_index % 8;
    (bytes[byte_idx] >> bit_offset) & 1
}

fn flip_bit(bytes: &mut [u8; 32], bit_index: usize) {
    let byte_idx = 31 - (bit_index / 8);
    let bit_offset = bit_index % 8;
    bytes[byte_idx] ^= 1 << bit_offset;
}

fn mask_lower_bits(path: &mut [u8; 32], height: usize) {
    for i in 0..height {
        let byte_idx = 31 - (i / 8);
        let bit_offset = i % 8;
        path[byte_idx] &= !(1 << bit_offset);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtxoProof {
    pub leaf_index: usize,
    pub leaf_count: usize,
    pub siblings: Vec<ProofElement>,
}

/// Verify a UTXO inclusion proof against the SMT root.
pub fn verify_utxo_proof(coin: &[u8; 32], proof: &UtxoProof, expected_root: &[u8; 32]) -> bool {
    if proof.siblings.len() != 256 { return false; }

    let mut current = *coin;
    for (h, elem) in proof.siblings.iter().enumerate() {
        let bit = get_bit(coin, h);
        let should_be_right = bit == 0;
        if elem.is_right != should_be_right { return false; }

        current = if elem.is_right {
            hash_concat(&current, &elem.hash)
        } else {
            hash_concat(&elem.hash, &current)
        };
    }
    current == *expected_root
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::hash;

    // ── MMR tests ───────────────────────────────────────────────────────

    #[test]
    fn mmr_append_and_root() {
        let mut mmr = MerkleMountainRange::new();
        let h1 = hash(b"leaf1");
        let h2 = hash(b"leaf2");

        mmr.append(&h1);
        assert_eq!(mmr.root(), h1);

        mmr.append(&h2);
        assert_ne!(mmr.root(), h1);

        mmr.append(&hash(b"leaf3"));
        assert_eq!(mmr.leaf_count(), 3);
    }

    #[test]
    fn mmr_proof_round_trip() {
        let mut mmr = MerkleMountainRange::new();
        let leaves: Vec<[u8; 32]> = (0..8u8).map(|i| hash(&[i])).collect();
        for leaf in &leaves { mmr.append(leaf); }
        let root = mmr.root();

        let positions = [0u64, 1, 3, 4, 7, 8, 10, 11];
        for (i, leaf) in leaves.iter().enumerate() {
            let proof = mmr.prove(positions[i]).unwrap();
            assert!(verify_mmr_proof(leaf, &proof, &root), "proof failed for leaf {}", i);
        }
    }

    #[test]
    fn mmr_size_formula() {
        assert_eq!(mmr_size(0), 0);
        assert_eq!(mmr_size(1), 1);
        assert_eq!(mmr_size(2), 3);
        assert_eq!(mmr_size(4), 7);
        assert_eq!(mmr_size(8), 15);
    }

    #[test]
    fn peaks_correctness() {
        assert_eq!(peaks(1), vec![0]);
        assert_eq!(peaks(3), vec![2]);
        assert_eq!(peaks(4), vec![2, 3]);
        assert_eq!(peaks(7), vec![6]);
    }

    // ── UTXO Accumulator (SMT) tests ────────────────────────────────────

    #[test]
    fn utxo_accumulator_basics() {
        let mut acc = UtxoAccumulator::new();
        let c1 = hash(b"coin1");
        let c2 = hash(b"coin2");
        let c3 = hash(b"coin3");

        assert!(acc.insert(c1));
        assert!(acc.insert(c2));
        assert!(acc.insert(c3));
        assert!(!acc.insert(c1)); // dup

        assert_eq!(acc.len(), 3);
        assert!(acc.contains(&c1));

        let r1 = acc.root();
        assert!(acc.remove(&c2));
        assert_ne!(r1, acc.root());
    }

    #[test]
    fn utxo_insert_remove_reinsert_same_root() {
        let mut acc = UtxoAccumulator::new();
        let c1 = hash(b"coin1");
        let c2 = hash(b"coin2");

        acc.insert(c1);
        acc.insert(c2);
        let root_before = acc.root();

        acc.remove(&c1);
        assert_ne!(root_before, acc.root());

        acc.insert(c1);
        assert_eq!(root_before, acc.root());
    }

    #[test]
    fn utxo_empty_root_is_deterministic() {
        let mut a = UtxoAccumulator::new();
        let mut b = UtxoAccumulator::new();
        assert_eq!(a.root(), b.root());
    }

    #[test]
    fn utxo_proof_round_trip() {
        let mut acc = UtxoAccumulator::new();
        let coins: Vec<[u8; 32]> = (0..10u8).map(|i| hash(&[i])).collect();
        for c in &coins { acc.insert(*c); }
        let root = acc.root();

        for c in &coins {
            let proof = acc.prove(c).unwrap();
            assert!(verify_utxo_proof(c, &proof, &root));
        }
    }

    #[test]
    fn utxo_wrong_coin_fails() {
        let mut acc = UtxoAccumulator::new();
        acc.insert(hash(b"coin1"));
        acc.insert(hash(b"coin2"));
        let root = acc.root();

        let proof = acc.prove(&hash(b"coin1")).unwrap();
        assert!(!verify_utxo_proof(&hash(b"fake"), &proof, &root));
    }

    #[test]
    fn utxo_proof_against_wrong_root_fails() {
        let mut acc = UtxoAccumulator::new();
        let c = hash(b"coin1");
        acc.insert(c);
        let proof = acc.prove(&c).unwrap();
        let fake_root = hash(b"not the root");
        assert!(!verify_utxo_proof(&c, &proof, &fake_root));
    }

    #[test]
    fn utxo_from_set() {
        let coins: Vec<[u8; 32]> = (0..5u8).map(|i| hash(&[i])).collect();
        let mut acc = UtxoAccumulator::from_set(coins.clone());
        assert_eq!(acc.len(), 5);
        for c in &coins { assert!(acc.contains(c)); }

        // Compare root with manual insert
        let mut acc2 = UtxoAccumulator::new();
        for c in &coins { acc2.insert(*c); }
        assert_eq!(acc.root(), acc2.root());
    }
    #[test]
    fn utxo_accumulator_large_set() {
        let mut acc = UtxoAccumulator::new();
        let coins: Vec<[u8; 32]> = (0..200u32).map(|i| {
            let mut h = blake3::Hasher::new();
            h.update(&i.to_le_bytes());
            *h.finalize().as_bytes()
        }).collect();
        for c in &coins { acc.insert(*c); }
        assert_eq!(acc.len(), 200);
        let root_with_all = acc.root();
        for c in &coins { assert!(acc.contains(c)); }
        for c in &coins[..100] { acc.remove(c); }
        assert_eq!(acc.len(), 100);
        assert_ne!(root_with_all, acc.root());
    }

    #[test]
    fn utxo_remove_all_returns_to_empty_root() {
        let mut acc = UtxoAccumulator::new();
        let empty_root = acc.root();
        let coins: Vec<[u8; 32]> = (0..5u8).map(|i| hash(&[i])).collect();
        for c in &coins { acc.insert(*c); }
        for c in &coins { acc.remove(c); }
        assert_eq!(acc.root(), empty_root);
        assert!(acc.is_empty());
    }

    #[test]
    fn utxo_proof_non_member_fails() {
        let acc = UtxoAccumulator::new();
        let coin = hash(b"not in set");
        assert!(acc.prove(&coin).is_err());
    }

    #[test]
    fn mmr_proof_invalid_position() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(&hash(b"a"));
        assert!(mmr.prove(999).is_err());
    }
}
