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
//  UTXO Accumulator  (drop-in replacement for HashSet<[u8;32]>)
// ═══════════════════════════════════════════════════════════════════════════

/// Merkle-committed UTXO set.  Sorted vec for correctness;
/// swap internals for Sparse Merkle Tree / Utreexo at scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtxoAccumulator {
    coins: Vec<[u8; 32]>,
    #[serde(skip)]
    cached_root: Option<[u8; 32]>,
}

impl PartialEq for UtxoAccumulator {
    fn eq(&self, other: &Self) -> bool {
        // Only compare the coins (the persistent state), ignoring the cached_root
        self.coins == other.coins
    }
}

impl Eq for UtxoAccumulator {}

impl UtxoAccumulator {
    pub fn new() -> Self { Self { coins: Vec::new(), cached_root: None } }

    pub fn from_set(coins: impl IntoIterator<Item = [u8; 32]>) -> Self {
        let mut v: Vec<[u8; 32]> = coins.into_iter().collect();
        v.sort();
        v.dedup();
        Self { coins: v, cached_root: None }
    }

    pub fn len(&self) -> usize { self.coins.len() }
    pub fn is_empty(&self) -> bool { self.coins.is_empty() }

    pub fn contains(&self, coin: &[u8; 32]) -> bool {
        self.coins.binary_search(coin).is_ok()
    }

    pub fn insert(&mut self, coin: [u8; 32]) -> bool {
        match self.coins.binary_search(&coin) {
            Ok(_) => false,
            Err(idx) => { self.coins.insert(idx, coin); self.cached_root = None; true }
        }
    }

    pub fn remove(&mut self, coin: &[u8; 32]) -> bool {
        match self.coins.binary_search(coin) {
            Ok(idx) => { self.coins.remove(idx); self.cached_root = None; true }
            Err(_) => false,
        }
    }

    /// Balanced Merkle root over sorted coins.
    pub fn root(&mut self) -> [u8; 32] {
        if let Some(r) = self.cached_root { return r; }
        let r = merkle_root(&self.coins);
        self.cached_root = Some(r);
        r
    }

    /// Merkle inclusion proof for `coin`.
    pub fn prove(&self, coin: &[u8; 32]) -> Result<UtxoProof> {
        let idx = self.coins.binary_search(coin)
            .map_err(|_| anyhow::anyhow!("coin not in accumulator"))?;
        Ok(UtxoProof {
            leaf_index: idx,
            leaf_count: self.coins.len(),
            siblings: merkle_proof(&self.coins, idx),
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8; 32]> { self.coins.iter() }
    pub fn into_vec(self) -> Vec<[u8; 32]> { self.coins }
}

fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() { return [0u8; 32]; }
    if leaves.len() == 1 { return leaves[0]; }

    let mut layer: Vec<[u8; 32]> = leaves.to_vec();
    let n = layer.len().next_power_of_two();
    layer.resize(n, [0u8; 32]);

    while layer.len() > 1 {
        layer = layer.chunks(2)
            .map(|pair| hash_concat(&pair[0], &pair[1]))
            .collect();
    }
    layer[0]
}

fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<ProofElement> {
    if leaves.len() <= 1 { return vec![]; }

    let mut layer: Vec<[u8; 32]> = leaves.to_vec();
    let n = layer.len().next_power_of_two();
    layer.resize(n, [0u8; 32]);

    let mut proof = Vec::new();
    let mut idx = index;

    while layer.len() > 1 {
        let sib = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        proof.push(ProofElement {
            hash: layer[sib],
            is_right: idx % 2 == 0,
        });
        layer = layer.chunks(2)
            .map(|pair| hash_concat(&pair[0], &pair[1]))
            .collect();
        idx /= 2;
    }
    proof
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtxoProof {
    pub leaf_index: usize,
    pub leaf_count: usize,
    pub siblings: Vec<ProofElement>,
}

/// Verify a UTXO inclusion proof.
pub fn verify_utxo_proof(coin: &[u8; 32], proof: &UtxoProof, expected_root: &[u8; 32]) -> bool {
    let mut current = *coin;
    for elem in &proof.siblings {
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

        // Leaf positions for 8-leaf MMR: 0,1,3,4,7,8,10,11
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
}
