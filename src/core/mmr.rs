//! # Merkle Mountain Range and UTXO Sparse Merkle Tree
//!
//! This module provides two cryptographic accumulators used to commit to chain
//! state.
//!
//! ## Structures
//!
//! 1. **[`MerkleMountainRange`]** — an append-only log of fixed-size hashes,
//!    used here for the chain history (block hashes). Supports O(1) amortised
//!    append, O(log n) inclusion proofs, and a deterministic O(log n) root.
//!
//! 2. **[`UtxoAccumulator`]** — a mutable set of fixed-size hashes (UTXOs),
//!    implemented as a Sparse Merkle Tree over the full 2^256 keyspace. Backed
//!    by `im::OrdSet` for persistent O(log n) clones. Supports
//!    insert/remove/contains in O(log n) amortised, plus a verifiable Merkle
//!    root and inclusion proofs.
//!
//! ## V1 / V2 hashing modes
//!
//! Every interior-node hash function in this module accepts a boolean
//! `is_v2` parameter that selects between two hashing rules:
//!
//! ```text
//! hash_node(L, R, is_v2 = false)  ≜  BLAKE3(L ‖ R)              (V1: legacy)
//! hash_node(L, R, is_v2 = true )  ≜  BLAKE3(0x01 ‖ L ‖ R)       (V2: domain-separated)
//! ```
//!
//! V1 is bit-identical to the original pre-domain-separation accumulator and
//! is retained so existing chains continue to verify. V2 adds a one-byte
//! domain separator [`NODE_TAG`] that prevents an attacker from constructing
//! a leaf value that collides with an internal node hash. Without the tag,
//! a 64-byte attacker-chosen leaf could in principle equal `BLAKE3(L ‖ R)` for
//! some attacker-chosen `(L, R)` pair, which would let them lie about either
//! the leaf's existence or the subtree's structure. With the tag, the only
//! way to find such a collision is a second-preimage attack on BLAKE3 over
//! the entire `0x01 ‖ L ‖ R` shape — ≈ 2^256 work.
//!
//! Activation of V2 is a chain-level concern: callers pass `is_v2 = true`
//! once the chain has crossed its `V2_ACTIVATION_HEIGHT` and `false` before
//! that. The single source of truth for the comparison lives in
//! `crate::core::types::is_v2_at(height)`.
//!
//! Crucially, `is_v2` is **not** stored inside any struct in this module. The
//! on-disk wire format of [`MerkleMountainRange`] and [`UtxoAccumulator`] is
//! therefore the same as it was before the V2 work began, and every existing
//! database loads without migration. The trade-off is that the SMT cache
//! (the `nodes` field of [`UtxoAccumulator`]) is correct for exactly one
//! hashing mode at a time — whichever mode produced the most recent
//! mutation. Mixing modes within a single accumulator's lifetime corrupts
//! the cache. In practice this is fine because every code path in this
//! crate derives `is_v2` from `state.height` and applies it consistently
//! within a single block; the boundary block at `V2_ACTIVATION_HEIGHT`
//! triggers an explicit [`UtxoAccumulator::rebuild_tree`].
//!
//! Leaves are stored verbatim (no leaf tag) in either mode. In V2, security
//! against leaf/node confusion comes from BLAKE3 second-preimage resistance
//! over the entire `0x01 ‖ L ‖ R` shape; in V1 it is inherited from the
//! original construction (and the fact that all leaves in this codebase are
//! themselves BLAKE3 outputs, making collisions a 2^128 problem). The
//! "root of a one-leaf MMR equals the leaf" identity is preserved in both
//! modes.
//!
//! ## Cache reconstruction after deserialisation
//!
//! [`UtxoAccumulator`] stores two derived caches (`nodes`, `buckets`) marked
//! `#[serde(skip)]`. After bincode-deserialising an accumulator, callers
//! **must** call [`UtxoAccumulator::rebuild_tree`] with the correct `is_v2`
//! flag for the chain height before invoking [`UtxoAccumulator::root`] or
//! [`UtxoAccumulator::prove`]; otherwise both will return the empty-tree
//! hash regardless of the actual coin set. The storage layer
//! (`crate::storage::Storage`) is responsible for this; if you add a new
//! deserialisation site, you must rebuild there too.
//!
//! ## Reading these docs
//!
//! Public functions carry a *Formal specification* block written in Z-style
//! notation. The conventions are:
//!
//! ```text
//!   Hash               ≜ BLAKE3 output ([u8; 32])
//!   seq T              ≜ finite sequence of T
//!   ⟨a, b, c⟩          ≜ literal sequence
//!   s ⌢ t              ≜ sequence concatenation
//!   #s                 ≜ length / cardinality
//!   ℙ T                ≜ power set of T
//!   pre / post         ≜ pre- and postcondition
//!   x'                 ≜ value of x AFTER the operation
//!   x?                 ≜ input parameter named x
//!   y!                 ≜ output named y
//!   𝔹                 ≜ {true, false}
//! ```

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║   SECTION 1 ─ Domain-separated hashing                                   ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Domain separator prepended to every internal Merkle node hash in V2 mode.
///
/// The choice of `0x01` (rather than e.g. `0x00`) is deliberate: leaves passed
/// to either accumulator are 32-byte BLAKE3 outputs, so a leaf cannot equal
/// `BLAKE3(0x01 ‖ … )` without breaking BLAKE3 second-preimage resistance.
/// Using `0x00` would not be incorrect, but it interacts badly with all the
/// places elsewhere in the codebase that use the all-zero hash as a sentinel
/// (empty MMR root, empty SMT leaf), and would tempt future readers into
/// believing a leaf-tag scheme was intended.
const NODE_TAG: u8 = 0x01;

/// Hash of an internal Merkle node given its two children, under the
/// specified hashing mode.
///
/// # Formal specification
///
/// ```text
///   hash_node : Hash × Hash × 𝔹 → Hash
///   hash_node(L, R, false) ≜ BLAKE3(L ⌢ R)
///   hash_node(L, R, true ) ≜ BLAKE3(⟨0x01⟩ ⌢ L ⌢ R)
/// ```
///
/// # Properties (both modes)
///
/// * **Determinism**: equal inputs in the same mode produce equal outputs.
/// * **Non-commutativity**: `hash_node(L, R, m) ≠ hash_node(R, L, m)`
///   whenever `L ≠ R`. The left/right ordering of children is significant
///   because BLAKE3 ingests them sequentially.
/// * **Cross-mode independence**: `hash_node(L, R, true) ≠ hash_node(L, R, false)`
///   except with negligible (≈ 2^-256) probability. This is the property
///   that makes V1 and V2 distinct hash universes — a V2 root over the same
///   underlying set is a different value from the V1 root.
///
/// # Properties (V2 only)
///
/// * **Domain separation**: an attacker cannot construct a 32-byte string
///   `x` in the BLAKE3 codomain together with a pair `(L, R)` such that
///   `x = hash_node(L, R, true)`, except by computing it forward from the
///   pair. In particular, an attacker who can choose a leaf value cannot
///   make that leaf collide with any internal-node hash, because internal
///   hashes always begin with `0x01` under BLAKE3 and the BLAKE3 input space
///   for valid leaves never starts with `0x01`-prefixed data of length 65
///   that the attacker controls.
///
/// # Implementation note
///
/// We construct a single `blake3::Hasher` and feed it the tag (if any),
/// the left child, and the right child as three separate `update` calls.
/// This is equivalent to one big concatenation but avoids allocating a
/// 65-byte temporary buffer on every call.
#[inline]
fn hash_node(left: &[u8; 32], right: &[u8; 32], is_v2: bool) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    if is_v2 {
        h.update(&[NODE_TAG]);
    }
    h.update(left);
    h.update(right);
    *h.finalize().as_bytes()
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║   SECTION 2 ─ Bit-level helpers (used by the SMT)                        ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// Bit indexing convention (CRITICAL — the rest of the SMT depends on this):
//
//   * `bytes` is a 32-byte big-endian path.
//   * Bit 0 is the **least-significant bit of the last byte** (`bytes[31]`).
//   * Bit 255 is the **most-significant bit of the first byte** (`bytes[0]`).
//
// The SMT root is at height 256. Descending one level uses bit 255, then 254,
// down to bit 0 at the leaf level. Equivalently, the path at height `h` is
// determined by bits `h..256` and bits `0..h` are conventionally zero.
//
// This convention means that **lexicographic ordering of 32-byte paths
// (big-endian, as `Ord` for `[u8; 32]` provides) matches a left-to-right
// traversal of the SMT at every level**. Within any subtree at height `h`, all
// paths share the top `256 − h` bits; sorting by the remaining bits is exactly
// sorting by the bit values from bit `h−1` down to bit `0`. This is the
// invariant that makes the `partition_point` split inside
// `compute_sparse_subtree` correct, and the reason `im::OrdSet<[u8; 32]>` is
// the right choice for the canonical coin set: its iteration order is already
// the in-tree leaf order.

/// Return bit `bit_index` of `bytes`, as 0 or 1.
///
/// # Preconditions
/// * `bit_index < 256`
///
/// # Reasoning
///
/// `bit_index >> 3` gives the byte distance from the LSB end (bit 0 lives in
/// `bytes[31]`, bit 8 in `bytes[30]`, …). Subtracting from 31 converts that
/// to the actual array index. `bit_index & 7` is the position within that
/// byte, again counted from the LSB. The final `& 1` isolates the result.
#[inline]
fn get_bit(bytes: &[u8; 32], bit_index: usize) -> u8 {
    debug_assert!(bit_index < 256);
    let byte_idx = 31 - (bit_index >> 3);
    let bit_offset = bit_index & 7;
    (bytes[byte_idx] >> bit_offset) & 1
}

/// Toggle bit `bit_index` of `bytes` in place.
///
/// # Preconditions
/// * `bit_index < 256`
///
/// # Postconditions
/// * The returned `bytes` differs from the input in exactly one bit position.
/// * Calling `flip_bit` twice on the same `bit_index` is the identity.
#[inline]
fn flip_bit(bytes: &mut [u8; 32], bit_index: usize) {
    debug_assert!(bit_index < 256);
    let byte_idx = 31 - (bit_index >> 3);
    let bit_offset = bit_index & 7;
    bytes[byte_idx] ^= 1 << bit_offset;
}

/// Clear all bits in positions `0..height` of `path` in place.
///
/// # Formal specification
///
/// ```text
///   pre:   height ≤ 256
///   post:  ∀ i ∈ 0..height       · bit(path', i) = 0
///          ∀ i ∈ height..256     · bit(path', i) = bit(path, i)
/// ```
///
/// # Reasoning
///
/// We split `height` into a whole-byte count (`full_bytes = height / 8`) and
/// a remainder bit count (`partial_bits = height % 8`). The low-order
/// `full_bytes` bytes of `path` (which sit at the high-numbered end of the
/// array, since the array is big-endian) are zeroed via a single slice fill.
/// The byte just above them (if it exists) has its low `partial_bits` bits
/// masked out; bits above `partial_bits` within that byte are preserved.
/// All bytes above receive no change.
///
/// # Complexity
///
/// O(1) — constant work regardless of `height`, since the slice fill is at
/// most 32 bytes wide. Earlier versions of this code looped `for i in
/// 0..height { clear bit i }`, which is O(height). The masking version
/// matters because `prove()` calls it inside a 240-iteration loop, and the
/// inner-loop cost dominates on large bucket sizes.
#[inline]
fn mask_lower_bits(path: &mut [u8; 32], height: usize) {
    debug_assert!(height <= 256);
    if height == 0 {
        return;
    }
    let full_bytes = height >> 3;
    let partial_bits = height & 7;

    // The last `full_bytes` bytes (the low-order ones) are zeroed entirely.
    let start = 32 - full_bytes;
    if full_bytes > 0 {
        path[start..].fill(0);
    }

    // The byte just above the cleared region, if any, has its bottom
    // `partial_bits` bits masked out; the upper bits are preserved.
    if partial_bits > 0 && start > 0 {
        let mask = !((1u8 << partial_bits) - 1);
        path[start - 1] &= mask;
    }
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║   SECTION 3 ─ Merkle Mountain Range                                      ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// ## Theory
//
// An MMR is a sequence of perfect binary trees ("peaks") whose heights
// correspond to the bits set in the leaf count `n`. After every leaf append,
// we eagerly merge any two adjacent peaks of equal height, cascading upward.
// Conceptually:
//
//   leaf_count = 1   →  peaks heights ⟨0⟩
//   leaf_count = 2   →  peaks heights ⟨1⟩
//   leaf_count = 3   →  peaks heights ⟨1, 0⟩
//   leaf_count = 4   →  peaks heights ⟨2⟩
//   leaf_count = 5   →  peaks heights ⟨2, 0⟩
//   leaf_count = 6   →  peaks heights ⟨2, 1⟩
//   leaf_count = 7   →  peaks heights ⟨2, 1, 0⟩
//   leaf_count = 8   →  peaks heights ⟨3⟩
//
// In general the heights of the peaks read left-to-right are the bit positions
// of `n` from MSB to LSB.
//
// ## Position numbering
//
// All nodes (leaves AND internal) are assigned a single contiguous index in
// **post-order**:
//
//   positions:   0  1  2   3  4  5   6   7  8  9   10
//   heights:     0  0  1   0  0  1   2   0  0  1    0   …
//
// Two leaves at positions `p` and `p+1` (both height 0) have parent at
// position `p+2` (height 1). Two height-`h` siblings at positions `p` and
// `p + 2^(h+1) − 1` have parent at position `p + 2^(h+1)` (height `h+1`).
//
// ## Total node count
//
// `mmr_size(n) = 2n − popcount(n)`, where `popcount(n)` is the number of
// 1-bits in `n`. Each leaf contributes one node directly, plus one internal
// node for each cascading merge, and the merges that happen on appending leaf
// `i` correspond exactly to the trailing 1-bits of `i`.
//
// ## V1 vs V2
//
// The post-order layout, position numbering, and merge cascade are
// hashing-mode-agnostic. Only the function used to compute parent hashes
// (i.e. `hash_node`) differs. Public methods that hash anything take an
// `is_v2: bool` argument and forward it; the struct itself is stateless
// w.r.t. mode.

/// Compute the height of the MMR node at the given post-order position.
///
/// Heights start at 0 for leaves and increase upward.
///
/// # Formal specification
///
/// ```text
///   pre:    pos < 2^63
///   post:   result = h ⇔ the node at position `pos` is the root of a
///                         perfect binary subtree of height h
/// ```
///
/// # Reasoning
///
/// We first locate the smallest perfect binary tree (in the post-order
/// numbering) that contains position `pos`. A perfect tree of height `h` has
/// `2^(h+1) − 1` nodes, occupying positions `0..2^(h+1) − 1`. We find the
/// smallest such `h` for which `pos < 2^(h+1) − 1`.
///
/// Then we descend through that tree:
/// * If `pos == 2^(h+1) − 2` (the post-order root of the current subtree), we
///   are done — the answer is the current height.
/// * Otherwise we are in either the left or right subtree. The left subtree
///   occupies positions `0..2^h − 1`, the right subtree occupies positions
///   `2^h − 1..2^(h+1) − 2`. We adjust `pos` to be relative to the chosen
///   subtree and decrement the height.
///
/// The descent terminates because the height strictly decreases each step and
/// we eventually reach the unique node at the current subtree's root.
///
/// # Complexity
///
/// O(log pos) time, O(1) space. There are O(1) bit-twiddling tricks that
/// do this in true O(1), but the descent form is easier to read and only
/// matters under microbenchmarks.
fn pos_height(mut pos: u64) -> u32 {
    // Find the smallest h such that pos < 2^(h+1) − 1.
    let mut h: u32 = 0;
    while (1u64 << (h + 1)) - 1 <= pos {
        h += 1;
    }
    // Descend.
    loop {
        let subtree_root_pos = (1u64 << (h + 1)) - 2;
        if pos == subtree_root_pos {
            return h;
        }
        // Left subtree of height h has 2^h − 1 nodes, occupying positions
        // 0..2^h − 1 (relative). The right subtree's positions start at 2^h − 1.
        let left_subtree_size = (1u64 << h) - 1;
        h -= 1;
        if pos >= left_subtree_size {
            pos -= left_subtree_size;
        }
    }
}

/// Total number of MMR nodes (leaves and internal) in an MMR with `n` leaves.
///
/// # Formal specification
///
/// ```text
///   mmr_size : ℕ → ℕ
///   mmr_size(n) = 2·n − popcount(n)
///   mmr_size(0) = 0
/// ```
///
/// # Reasoning
///
/// Each appended leaf produces exactly one new leaf node plus one new
/// internal node for every merge it triggers. The number of merges triggered
/// by appending leaf number `i` (0-indexed) equals the number of trailing
/// 1-bits in `i+1` written in binary — the same count that the binary
/// counter "carry" operation performs. Summing across all leaves gives
/// `n + (n - popcount(n)) = 2n - popcount(n)`.
///
/// # Examples
///
/// | n  | mmr_size(n) |
/// |----|-------------|
/// | 0  | 0           |
/// | 1  | 1           |
/// | 2  | 3           |
/// | 3  | 4           |
/// | 4  | 7           |
/// | 5  | 8           |
/// | 7  | 11          |
/// | 8  | 15          |
pub fn mmr_size(n: u64) -> u64 {
    if n == 0 {
        0
    } else {
        2 * n - (n.count_ones() as u64)
    }
}

/// Positions of the peaks in an MMR with `size` total nodes (post-order).
///
/// Peaks are listed **left to right** (largest to smallest in height).
///
/// # Formal specification
///
/// ```text
///   peaks : ℕ → seq ℕ
///   pre:    size = mmr_size(n) for some n ≥ 0
///   post:   #result = popcount(n)
///           ∀ i · result[i] is the root position of the i-th leftmost
///                  perfect subtree
/// ```
///
/// # Reasoning
///
/// Greedily carve off the largest possible perfect subtree from the front of
/// the MMR; repeat with what remains. This is equivalent to scanning the bits
/// of `n` from MSB to LSB: each set bit corresponds to one peak whose height
/// equals that bit's position. The post-order root of a height-`h` perfect
/// subtree starting at offset `o` lives at position `o + 2^(h+1) − 2`.
///
/// # Complexity
///
/// O(log n) time and space (one entry per peak, popcount(n) ≤ ⌈log₂(n+1)⌉).
pub fn peaks(size: u64) -> Vec<u64> {
    let mut result = Vec::new();
    let mut remaining = size;
    let mut offset = 0u64;

    while remaining > 0 {
        // Find the largest perfect tree (height h, 2^(h+1) − 1 nodes) that fits
        // into `remaining`.
        let mut h = 1u32;
        while (1u64 << (h + 1)) - 1 <= remaining {
            h += 1;
        }
        let tree_size = (1u64 << h) - 1;
        if tree_size > remaining {
            break;
        }
        // The peak is the post-order root of this subtree.
        result.push(offset + tree_size - 1);
        offset += tree_size;
        remaining -= tree_size;
    }
    result
}

/// An append-only Merkle Mountain Range over 32-byte hashes.
///
/// # Abstract state
///
/// ```text
///   MMR ≜ ⟨L : seq Hash⟩
///   inv:  #L < 2^63
/// ```
///
/// The internal representation also holds the post-order array of all node
/// hashes (leaves and internal merges) so that proofs can be built without
/// recomputation. The `nodes` field is fully serializable; deserialisation
/// restores the structure verbatim with no rebuild step required.
///
/// `nodes` is an [`im::Vector`] so that `clone()` is O(1) — important for the
/// chain state, which is forked frequently during reorg evaluation.
///
/// # Hashing version
///
/// `MerkleMountainRange` is *stateless* with respect to V1/V2 hashing: the
/// `is_v2` flag is supplied to [`append`](Self::append),
/// [`root`](Self::root), and [`verify_mmr_proof`] at call time. The struct
/// itself stores only post-order node hashes — historical values that were
/// computed under whichever mode was active at append time. Mixing modes
/// across appends produces an MMR whose interior hashes are inconsistent
/// with each other; it is the caller's responsibility to apply a single
/// `is_v2` value consistently within any one block.
///
/// In normal operation that responsibility lives in `apply_batch_internal`
/// in `state.rs`, which derives `is_v2` from the current chain height once
/// per block and threads it through all relevant calls.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MerkleMountainRange {
    /// Post-order array of all node hashes (leaves and internal merges).
    nodes: im::Vector<[u8; 32]>,
    /// Number of leaves appended so far. Tracked explicitly for O(1) access;
    /// also derivable from `nodes.len()` via the inverse of `mmr_size`, but
    /// that's more work than just maintaining a counter.
    leaf_count: u64,
}

impl MerkleMountainRange {
    /// Construct an empty MMR.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result.L = ⟨⟩
    /// ```
    pub fn new() -> Self {
        Self {
            nodes: im::Vector::new(),
            leaf_count: 0,
        }
    }

    /// Reconstruct an MMR from its raw post-order node array and leaf count.
    ///
    /// # Formal specification
    /// ```text
    ///   pre:   #nodes = mmr_size(leaf_count) = 2·leaf_count − popcount(leaf_count)
    ///          all interior nodes were hashed under a single, consistent
    ///                  is_v2 mode at the time `nodes` was produced
    ///   post:  result.L has #leaf_count leaves and #nodes total nodes
    /// ```
    ///
    /// Intended for storage migration code that has already validated the
    /// inputs are consistent. The caller is responsible for the structural
    /// invariant; this constructor performs no checks because the only
    /// caller is `storage::deserialize_state`'s legacy-format path, which
    /// loads bytes that were themselves produced by this same MMR layout.
    #[doc(hidden)]
    pub fn from_raw_parts(nodes: im::Vector<[u8; 32]>, leaf_count: u64) -> Self {
        Self { nodes, leaf_count }
    }

    /// Number of leaves currently in the MMR.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result = #L
    /// ```
    pub fn leaf_count(&self) -> u64 {
        self.leaf_count
    }

    /// Total number of nodes (leaves plus internal merges).
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result = mmr_size(#L) = 2·#L − popcount(#L)
    /// ```
    pub fn size(&self) -> u64 {
        self.nodes.len() as u64
    }

    /// Append a leaf and cascade-merge any sibling pairs of equal height.
    ///
    /// Returns the post-order position of the newly appended leaf.
    ///
    /// # Formal specification
    ///
    /// ```text
    ///   pre:   #L < 2^63 − 1
    ///   post:  L' = L ⌢ ⟨leaf_hash?⟩
    ///          result! = mmr_size(#L)              (the leaf's position)
    ///          all interior merges hashed under hash_node(_, _, is_v2?)
    /// ```
    ///
    /// # Reasoning
    ///
    /// The append is always at the next available position. After placing the
    /// leaf at height 0, we examine the immediately-preceding node: if it sits
    /// at the same height, the two are siblings whose parent is the very next
    /// position, computed as `hash_node(left, right, is_v2)`. Repeating this
    /// rule upward terminates because (a) heights strictly increase each
    /// iteration, and (b) at each height we either find a same-height sibling
    /// (merge and continue) or we don't (stop). The number of merges equals
    /// the number of trailing 1-bits in the new leaf index — the same count
    /// the binary-counter "carry" operation performs on each increment.
    ///
    /// # Complexity
    ///
    /// Worst case O(log #L) — an append where every height up to `log #L`
    /// merges. Amortised O(1) over a sequence of appends, because the total
    /// number of merges over all appends `0..n` is `n − popcount(n) < n`.
    pub fn append(&mut self, leaf_hash: &[u8; 32], is_v2: bool) -> u64 {
        let leaf_pos = self.nodes.len() as u64;
        self.nodes.push_back(*leaf_hash);
        self.leaf_count += 1;

        let mut current_pos = leaf_pos;
        let mut current_height: u32 = 0;

        // Cascade: as long as the node we just placed has a same-height left
        // sibling, merge them into a parent and continue with that parent.
        loop {
            // For a same-height sibling pair at height `current_height`, the
            // left sibling sits `2^(h+1) − 1` positions before the right.
            let sibling_distance = (1u64 << (current_height + 1)) - 1;
            if current_pos < sibling_distance {
                // No room for a left sibling — current_pos is already a peak.
                break;
            }
            let left_pos = current_pos - sibling_distance;
            // The candidate left sibling exists in the array, but it is only
            // *our* sibling if it is at the same height. Otherwise, the MMR
            // construction guarantees that no merge is owed at this level.
            if pos_height(left_pos) != current_height {
                break;
            }

            let parent_hash = hash_node(
                &self.nodes[left_pos as usize],
                &self.nodes[current_pos as usize],
                is_v2,
            );
            let parent_pos = self.nodes.len() as u64;
            self.nodes.push_back(parent_hash);
            current_pos = parent_pos;
            current_height += 1;
        }
        leaf_pos
    }

    /// Compute the MMR root by **bagging the peaks right-to-left**.
    ///
    /// # Formal specification
    ///
    /// ```text
    ///   pre:   true
    ///   post:  let p = peaks(size())   in
    ///          if p = ⟨⟩          then result = 0^32
    ///          else                    result = bag(p, is_v2?)
    ///   where  bag(⟨x⟩, _)        = nodes[x]
    ///          bag(p₁ ⌢ p₂, m)    = hash_node(nodes[p₁], bag(p₂, m), m)
    /// ```
    ///
    /// In words: take the rightmost peak, fold each peak to its left into it
    /// using `hash_node(left_peak, accumulator, is_v2)`. For a single peak
    /// the root is just that peak's hash; for an empty MMR the root is the
    /// all-zero hash by convention.
    ///
    /// # Reasoning
    ///
    /// Right-to-left bagging is the canonical order for MMRs because peaks
    /// list left-to-right from large to small height; folding right-to-left
    /// places the smaller (younger) subtrees deeper in the resulting hash
    /// chain, mirroring the order in which they would have been merged had
    /// the MMR happened to grow into a single perfect tree. This means the
    /// MMR root over any prefix `L'` of `L` is computable from a small
    /// number of peak hashes regardless of `#L'`, which is the property
    /// light clients rely on.
    ///
    /// # Complexity
    ///
    /// O(log #L) — one `hash_node` call per peak, and `popcount(#L) ≤ log₂(#L+1)`.
    pub fn root(&self, is_v2: bool) -> [u8; 32] {
        let peak_positions = peaks(self.nodes.len() as u64);
        if peak_positions.is_empty() {
            return [0u8; 32];
        }
        // Start with the rightmost (smallest) peak, then fold each
        // larger-height peak into the accumulator from the right.
        let mut acc = self.nodes[*peak_positions.last().unwrap() as usize];
        for &pos in peak_positions.iter().rev().skip(1) {
            acc = hash_node(&self.nodes[pos as usize], &acc, is_v2);
        }
        acc
    }

    /// Build an inclusion proof for the leaf at post-order position `leaf_pos`.
    ///
    /// # Formal specification
    ///
    /// ```text
    ///   pre:   leaf_pos < size() ∧ pos_height(leaf_pos) = 0
    ///   post:  ∀ is_v2 : 𝔹 ·
    ///            verify_mmr_proof(L[idx(leaf_pos)], result, root(is_v2), is_v2) = true
    ///          where the LHS root and RHS verifier use the same is_v2.
    /// ```
    ///
    /// In other words, the proof itself is mode-agnostic — it carries
    /// sibling hashes and peak hashes as the MMR currently stores them, and
    /// is verified against whichever root the caller supplies. The caller
    /// is responsible for using a consistent `is_v2` on both sides.
    ///
    /// The proof contains:
    /// * the `leaf_pos`,
    /// * the chain of sibling hashes from the leaf up to the containing peak,
    /// * the full list of peak hashes (so the verifier can bag them), and
    /// * the index of *our* peak within that list, plus the MMR's total size.
    ///
    /// # Reasoning for the climb
    ///
    /// Starting at `leaf_pos` with height 0, on each iteration we either
    /// * find a right sibling at the same height (the next sibling lives at
    ///   `pos + 2^(h+1) − 1`) — emit it as `is_right = true` and jump to the
    ///   parent at `right_sibling + 1`; or
    /// * have no right sibling, which means the right sibling subtree never
    ///   formed (we are the right child) — locate the left sibling at
    ///   `pos − (2^(h+1) − 1)`, emit it as `is_right = false`, and jump to the
    ///   parent at `pos + 1`.
    ///
    /// We stop when the current position is itself one of the MMR's peaks.
    /// Termination follows from the fact that height strictly increases each
    /// step and there are only O(log #L) heights below the tallest peak.
    ///
    /// # Complexity
    /// O(log #L) time and space.
    pub fn prove(&self, leaf_pos: u64) -> Result<MmrProof> {
        let sz = self.nodes.len() as u64;
        if leaf_pos >= sz {
            bail!("position {} out of range (size {})", leaf_pos, sz);
        }
        if pos_height(leaf_pos) != 0 {
            bail!("position {} is not a leaf", leaf_pos);
        }

        let peak_positions = peaks(sz);
        let mut siblings = Vec::new();
        let mut pos = leaf_pos;
        let mut height: u32 = 0;

        loop {
            if peak_positions.contains(&pos) {
                break;
            }

            let right_sibling = pos + (1u64 << (height + 1)) - 1;
            if right_sibling < sz && pos_height(right_sibling) == height {
                // We are the LEFT child; the right sibling exists.
                siblings.push(ProofElement {
                    hash: self.nodes[right_sibling as usize],
                    is_right: true,
                });
                pos = right_sibling + 1; // parent (post-order)
            } else {
                // We are the RIGHT child; the left sibling is `sibling_distance`
                // positions back.
                let left_sibling = pos - ((1u64 << (height + 1)) - 1);
                siblings.push(ProofElement {
                    hash: self.nodes[left_sibling as usize],
                    is_right: false,
                });
                pos += 1; // parent
            }
            height += 1;
        }

        let our_peak_pos = pos;
        let peak_index = peak_positions
            .iter()
            .position(|&p| p == our_peak_pos)
            .ok_or_else(|| {
                anyhow::anyhow!("internal invariant violated: climbed to non-peak")
            })?;

        Ok(MmrProof {
            leaf_pos,
            siblings,
            peak_hashes: peak_positions
                .iter()
                .map(|&p| self.nodes[p as usize])
                .collect(),
            peak_index,
            mmr_size: sz,
        })
    }

    /// Read a node hash by post-order position. Returns `None` if out of range.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result = Some(nodes[pos]) if pos < size()
    ///                 = None             otherwise
    /// ```
    pub fn get(&self, pos: u64) -> Option<&[u8; 32]> {
        self.nodes.get(pos as usize)
    }
}

/// One step of an inclusion proof: a sibling hash plus which side it sits on.
///
/// `is_right = true` means the sibling is to the right of the path node, i.e.
/// the verifier should compute `hash_node(current, sibling, is_v2)`.
/// `is_right = false` means the sibling is to the left, i.e. compute
/// `hash_node(sibling, current, is_v2)`.
///
/// This struct is shared between MMR proofs and SMT proofs because the
/// climb logic is identical at the level of "which order do I hash the
/// pair in" — only the way the climb is *navigated* differs between the
/// two structures.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofElement {
    pub hash: [u8; 32],
    pub is_right: bool,
}

/// MMR inclusion proof.
///
/// The verifier reconstructs the path hash from the leaf upward to the peak,
/// then verifies that bagging the supplied peaks reproduces the expected root.
///
/// # Wire format stability
///
/// This struct is `Serialize + Deserialize` and may travel over the network
/// to light clients. The on-wire layout is exactly the field order shown
/// here; do not reorder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MmrProof {
    /// Post-order position of the leaf being proved.
    pub leaf_pos: u64,
    /// Chain of sibling hashes from the leaf up to (but not including) the
    /// containing peak. Ordered bottom-up.
    pub siblings: Vec<ProofElement>,
    /// All peak hashes of the MMR at the moment of proof generation,
    /// left-to-right.
    pub peak_hashes: Vec<[u8; 32]>,
    /// Index into `peak_hashes` identifying the peak that contains the leaf.
    pub peak_index: usize,
    /// MMR total size at proof generation. Informational; the verifier does
    /// not strictly need this to check correctness against `expected_root`,
    /// but light clients use it to detect "proof from a future MMR" cases.
    pub mmr_size: u64,
}

/// Verify an MMR inclusion proof against a known root, under the given
/// hashing-mode flag.
///
/// # Formal specification
///
/// ```text
///   verify_mmr_proof : Hash × MmrProof × Hash × 𝔹 → 𝔹
///   post:  result = true ⇒ ∃ M : MMR ·
///                          M.root(is_v2?) = expected_root  ∧
///                          M.L[idx(proof.leaf_pos)] = leaf_hash
///          (soundness: assuming BLAKE3 collision resistance)
/// ```
///
/// # Algorithm
///
/// 1. Walk the path: starting from `leaf_hash`, fold in each `siblings[i]`
///    according to its `is_right` flag, producing the proof's claimed peak
///    value.
/// 2. Confirm that value matches `peak_hashes[peak_index]`. This is the
///    step that defeats a forged-leaf attack: even if the climb produces
///    *some* hash, it has to land on the specific peak the proof claims.
/// 3. Bag `peak_hashes` right-to-left, exactly as
///    [`MerkleMountainRange::root`] would.
/// 4. Compare the bagged result with `expected_root`.
///
/// Returns `false` on any mismatch or out-of-range index.
///
/// # Cross-mode behaviour
///
/// A proof generated with the MMR in V1 mode (i.e. interior hashes computed
/// under `hash_node(_, _, false)`) will only verify when this function is
/// called with `is_v2 = false`. Calling it with `is_v2 = true` against a V1
/// MMR will fail at step 2 because the climb produces a `BLAKE3(0x01 ‖ L ‖ R)`
/// chain that cannot equal the V1 peak. The same logic applies in reverse.
/// This is the property tested by the `utxo_v2_proof_does_not_verify_against_v1_root`
/// test (the SMT analogue of this check).
pub fn verify_mmr_proof(
    leaf_hash: &[u8; 32],
    proof: &MmrProof,
    expected_root: &[u8; 32],
    is_v2: bool,
) -> bool {
    // Defensive bounds check.
    if proof.peak_hashes.is_empty() || proof.peak_index >= proof.peak_hashes.len() {
        return false;
    }

    // 1. Reconstruct the climb to the peak.
    let mut current = *leaf_hash;
    for elem in &proof.siblings {
        current = if elem.is_right {
            hash_node(&current, &elem.hash, is_v2)
        } else {
            hash_node(&elem.hash, &current, is_v2)
        };
    }

    // 2. The reconstructed climb must hit the claimed peak.
    if current != proof.peak_hashes[proof.peak_index] {
        return false;
    }

    // 3. Bag peaks right-to-left, matching MerkleMountainRange::root.
    let mut acc = *proof.peak_hashes.last().unwrap();
    for peak in proof.peak_hashes.iter().rev().skip(1) {
        acc = hash_node(peak, &acc, is_v2);
    }

    // 4. Final equality check.
    acc == *expected_root
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║   SECTION 4 ─ UTXO Sparse Merkle Tree                                    ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// ## Theory
//
// An SMT of height H is a Merkle tree spanning the entire 2^H leaf keyspace;
// here `H = 256` so the leaf index is exactly a 32-byte coin ID. Most leaves
// are absent. We define an "empty leaf" sentinel `[0; 32]` and a precomputed
// table of "all-empty subtree" hashes (one per height), so any subtree
// containing zero coins has a known constant hash. This is what makes the
// "sparse" part viable: of the 2^256 leaf positions, only the few thousand
// that actually hold UTXOs need any real work; everything else collapses to
// a precomputed constant.
//
// ## Bit ordering
//
// Bit 255 of the path picks the child at the root (height 256), bit 254 picks
// the next, …, bit 0 picks the leaf (height 1 → height 0). Combined with the
// big-endian convention from Section 2, **lexicographic order on coin IDs
// matches the left-to-right ordering of leaves in the tree at every level**.
// This invariant powers the partition-based recursion below: an `im::OrdSet`
// of coin IDs is, when iterated, already in tree-leaf order, so a
// `partition_point` on a single bit splits the set into the corresponding
// left and right subtree-resident subsets in O(log n).
//
// ## Storage strategy: bucketed two-tier
//
// A naïve SMT stores every interior node along the path of every coin, which
// gives `O(N · 256)` memory. We do better:
//
// * **Buckets** (`im::HashMap<u16, im::OrdSet<[u8; 32]>>`): coins are grouped
//   by their top 16 bits. The bottom 240 levels of the tree under each bucket
//   are computed *on the fly* from the sorted coin set (see
//   [`compute_sparse_subtree`]). Because `compute_sparse_subtree` is
//   `O(K log K)` for `K` coins in the bucket, this is cheap as long as
//   buckets stay small — which they do, statistically: with `N` coins
//   distributed uniformly over 16-bit prefixes, expected bucket size is
//   `N / 65_536`, comfortably small for any realistic UTXO set in the
//   millions.
//
// * **Cached top 16 levels** (`im::HashMap<(u16, u16), [u8; 32]>`): node
//   hashes at heights 240..=256 are cached, keyed by `(height,
//   top-16-bit prefix)`. This is a tight upper bound: at height `h`, only
//   `2^(256 − h)` distinct paths exist, so the total number of cached nodes
//   is bounded by `Σ_{h=240..=256} 2^(256 − h) = 2^17 − 1 = 131_071`.
//   Crucially, this bound is independent of `N` — even a billion-UTXO set
//   uses no more than 131K cache entries, eliminating any memory blowup
//   when the chain grows.
//
// ## Insertion / deletion
//
// On every change to a bucket, we recompute the bucket's height-240 hash from
// scratch (cheap because `compute_sparse_subtree` does only `O(K log K)` work
// for `K` coins in the bucket) and ripple it up through the cached 16 levels,
// updating each parent on the way to the root. Because the cache is kept
// continuously consistent with the coin set, [`UtxoAccumulator::root`] is
// O(1).
//
// ## Cache eviction
//
// Empty subtrees are *never* stored in the cache. Whenever a ripple
// computes a node hash equal to `get_empty_hash(height, is_v2)`, the entry
// is removed instead of inserted. This keeps the cache size proportional to
// the number of *non-empty* subtree positions, not to the keyspace, and
// guarantees that an accumulator that has been emptied and rebuilt holds
// no dangling cache entries.
//
// ## V1 vs V2 in the SMT
//
// The hashing-mode boundary affects every interior-node hash and every
// empty-subtree hash. Two precomputed empty-subtree tables exist
// (`EMPTY_HASHES_V1` and `EMPTY_HASHES_V2`); the right one is selected by
// `get_empty_hash(_, is_v2)`. All of `compute_sparse_subtree`,
// `extract_path_siblings`, and the [`UtxoAccumulator`] methods that hash
// anything take an `is_v2: bool` parameter and propagate it.
//
// **Critical caller responsibility**: the in-memory `nodes` cache is
// consistent with exactly *one* hashing mode at a time — whichever mode
// produced the most recent `update_cache_for_bucket` call. A caller that
// alternates between V1 and V2 mutations on the same accumulator without
// rebuilding will end up with a cache containing a mixture of V1 and V2
// hashes, none of which agree on the root. In practice the only places
// this matters are the V2 activation boundary (handled by an explicit
// `rebuild_tree(true)` in `apply_batch_internal`) and post-deserialisation
// (handled by `storage.rs::deserialize_state`).

/// Height of the SMT (number of edge levels from leaf to root).
const SMT_HEIGHT: usize = 256;

/// The lowest height at which interior node hashes are cached. Heights below
/// this are computed on demand from the bucket's sorted coin list.
///
/// 240 is chosen so that the cached portion holds exactly 16 levels — enough
/// to map cleanly onto the top 16 bits of the path (used as the bucket key)
/// while keeping the bound on cache size at the convenient value 2^17 - 1.
const CACHE_MIN_HEIGHT: usize = 240;

/// Number of distinct buckets (one per top-16-bit prefix). Currently 2^16.
#[allow(dead_code)]
const NUM_BUCKETS: usize = 1 << (SMT_HEIGHT - CACHE_MIN_HEIGHT);

/// Lazily-initialised tables of empty-subtree hashes, indexed by height.
/// Separated by hashing mode so both V1 and V2 chains have O(1) lookups.
///
/// `EMPTY_HASHES_V1[h]` is the hash of the all-empty subtree of height `h`
/// under V1 hashing (no domain separator). `EMPTY_HASHES_V2[h]` is the same
/// under V2. Each table holds exactly `SMT_HEIGHT + 1 = 257` entries
/// (indices `0..=256`).
///
/// Initialisation cost is 257 BLAKE3 calls per mode, run once on first use
/// of that mode; thereafter every lookup is a vector index.
static EMPTY_HASHES_V1: std::sync::OnceLock<Vec<[u8; 32]>> = std::sync::OnceLock::new();
static EMPTY_HASHES_V2: std::sync::OnceLock<Vec<[u8; 32]>> = std::sync::OnceLock::new();

/// Build the empty-hash table for one mode.
///
/// # Formal specification
/// ```text
///   pre:   true
///   post:  result[0] = 0^32
///          ∀ h ∈ 1..=SMT_HEIGHT ·
///              result[h] = hash_node(result[h-1], result[h-1], is_v2?)
///          #result = SMT_HEIGHT + 1
/// ```
fn init_empty_hashes(is_v2: bool) -> Vec<[u8; 32]> {
    let mut h = Vec::with_capacity(SMT_HEIGHT + 1);
    h.push([0u8; 32]);
    for i in 0..SMT_HEIGHT {
        let parent = hash_node(&h[i], &h[i], is_v2);
        h.push(parent);
    }
    debug_assert_eq!(h.len(), SMT_HEIGHT + 1);
    h
}

/// Return the hash of an all-empty subtree of the given height, under the
/// specified hashing mode.
///
/// # Formal specification
///
/// ```text
///   get_empty_hash : 0..=256 × 𝔹 → Hash
///   get_empty_hash(0,    _    ) = 0^32
///   get_empty_hash(h,    is_v2) = hash_node(get_empty_hash(h−1, is_v2),
///                                            get_empty_hash(h−1, is_v2),
///                                            is_v2)         for h > 0
/// ```
///
/// # Preconditions
/// * `height ≤ SMT_HEIGHT`
fn get_empty_hash(height: usize, is_v2: bool) -> [u8; 32] {
    let table = if is_v2 {
        EMPTY_HASHES_V2.get_or_init(|| init_empty_hashes(true))
    } else {
        EMPTY_HASHES_V1.get_or_init(|| init_empty_hashes(false))
    };
    table[height]
}

/// Recursively compute the hash of an SMT subtree of the given height,
/// containing exactly the given (sorted) set of coins, under the specified
/// hashing mode.
///
/// # Formal specification
///
/// ```text
///   pre:   coins sorted lexicographically (ascending)        ∧
///          ∀ c ∈ coins · c shares its top (256 − height) bits with the
///                         subtree's path (the caller's responsibility)  ∧
///          height ≤ SMT_HEIGHT
///
///   post:  result = the hash of the subtree of height `height` that
///                   contains exactly the leaves listed in `coins`,
///                   hashed according to the `is_v2` rule.
/// ```
///
/// # Reasoning
///
/// At height 0 the subtree is a single leaf: it is either empty (return the
/// sentinel `0^32`) or contains exactly one coin (return that coin's hash —
/// no leaf domain separator; see module docs for justification).
///
/// At height `h > 0` we split on bit `h − 1`. Because the input slice is
/// lexicographically sorted **and** all coins in the slice agree on the top
/// `256 − h` bits, all coins with bit `h − 1` equal to 0 form a contiguous
/// prefix and all coins with bit `h − 1` equal to 1 form the corresponding
/// suffix. We find the split via [`partition_point`], a binary-search
/// returning the index of the first coin whose bit-`(h−1)` is 1.
///
/// The two halves are recursively hashed and combined via [`hash_node`]. If
/// either half is empty, the recursion bottoms out at the empty-subtree
/// sentinel, which is what allows this routine to be efficient even for very
/// sparse inputs: the total work is `O(K log K)` for `K = #coins`, regardless
/// of the height, because each coin's contribution propagates through at most
/// `log₂ K + h_eff` levels where `h_eff` is the depth at which it is finally
/// alone in its subtree.
///
/// # Examples
///
/// ```
/// # use midstate::core::mmr::*;
/// # use midstate::core::types::hash;
/// let coins = vec![hash(b"coin1")];
/// let root = compute_sparse_subtree(240, &coins, false);
/// assert_ne!(root, [0u8; 32]);
/// ```
///
/// [`partition_point`]: slice::partition_point
pub fn compute_sparse_subtree(height: usize, coins: &[[u8; 32]], is_v2: bool) -> [u8; 32] {
    // Empty subtree → constant hash.
    if coins.is_empty() {
        return get_empty_hash(height, is_v2);
    }
    // Single leaf at the bottom.
    if height == 0 {
        return coins[0];
    }

    let bit_idx = height - 1;
    // Sorted-input invariant: bit-0 coins precede bit-1 coins.
    let split_idx = coins.partition_point(|c| get_bit(c, bit_idx) == 0);
    let (left_coins, right_coins) = coins.split_at(split_idx);

    let left_hash = compute_sparse_subtree(height - 1, left_coins, is_v2);
    let right_hash = compute_sparse_subtree(height - 1, right_coins, is_v2);

    hash_node(&left_hash, &right_hash, is_v2)
}

/// Walk the SMT subtree along the path of `target`, **emitting all off-path
/// sibling hashes** at heights `height − 1, height − 2, …, 0` in *bottom-up*
/// order (level 0 first, level `height − 1` last).
///
/// # Formal specification
///
/// ```text
///   pre:   coins sorted ascending                                         ∧
///          ∀ c ∈ coins · c shares the top (256 − height) bits of `target` ∧
///          height ≤ SMT_HEIGHT
///
///   post:  #out' − #out = height                                          ∧
///          if reconstruct_path(target, out'[#out..]) is folded with
///          hash_node(_, _, is_v2?), the result equals
///          compute_sparse_subtree(height, coins, is_v2?)
/// ```
///
/// # Reasoning
///
/// We perform a single recursive descent. At each level `h ∈ {height, …, 1}`
/// we split `coins` on bit `h − 1`. Exactly one half (the one whose bit-(h−1)
/// matches `target`'s bit-(h−1)) contains, or could contain, the target; the
/// other half is the off-path sibling subtree. We hash the sibling subtree
/// once via [`compute_sparse_subtree`], then **recurse first** into the
/// on-path half, and only push the sibling element after the recursion
/// returns. This post-order push order yields siblings in bottom-up order,
/// which is the convention the verifier expects.
///
/// The reason this single-descent design matters is that an earlier version
/// of the prove path called `compute_sparse_subtree` 240 times (once per
/// level, each at full bucket scope), giving `O(240 · K log K)` work per
/// proof. The current single descent is `O(K log K)` total — a ~240x
/// speedup at proof generation time on full buckets.
///
/// # Complexity
///
/// O(K · log K) total over `height` levels, where `K = #coins`. Each coin
/// participates in at most `log₂ K + 1` non-empty sibling-side hashings.
fn extract_path_siblings(
    height: usize,
    coins: &[[u8; 32]],
    target: &[u8; 32],
    out: &mut Vec<ProofElement>,
    is_v2: bool,
) {
    if height == 0 {
        return;
    }
    let bit_idx = height - 1;
    let split_idx = coins.partition_point(|c| get_bit(c, bit_idx) == 0);
    let (left_coins, right_coins) = coins.split_at(split_idx);

    let target_bit = get_bit(target, bit_idx);
    if target_bit == 0 {
        // Target is in the LEFT subtree; the sibling is the RIGHT subtree.
        let sibling_hash = compute_sparse_subtree(height - 1, right_coins, is_v2);
        // Recurse FIRST so deeper siblings get pushed before this one.
        extract_path_siblings(height - 1, left_coins, target, out, is_v2);
        out.push(ProofElement {
            hash: sibling_hash,
            is_right: true,
        });
    } else {
        // Target is in the RIGHT subtree; the sibling is the LEFT subtree.
        let sibling_hash = compute_sparse_subtree(height - 1, left_coins, is_v2);
        extract_path_siblings(height - 1, right_coins, target, out, is_v2);
        out.push(ProofElement {
            hash: sibling_hash,
            is_right: false,
        });
    }
}

/// Convert a 32-byte path (with bits `0..h` zero) into the compact
/// `(height, top-16-bits)` cache key used by [`UtxoAccumulator::nodes`].
///
/// The top 16 bits of any path at height ≥ 240 fully determine the path
/// (because all lower bits are zero), so we can store keys in 4 bytes
/// instead of the 33 bytes we'd need for `(u8, [u8; 32])`. Across a full
/// 131K-entry cache that's ~3.7 MB saved.
///
/// # Preconditions
/// * `height ∈ CACHE_MIN_HEIGHT..=SMT_HEIGHT`
/// * `path` has all bits below `CACHE_MIN_HEIGHT` cleared (caller's job).
#[inline]
fn cache_key(height: usize, path: &[u8; 32]) -> (u16, u16) {
    debug_assert!((CACHE_MIN_HEIGHT..=SMT_HEIGHT).contains(&height));
    let top16 = u16::from_be_bytes([path[0], path[1]]);
    (height as u16, top16)
}

/// Build a 32-byte path with only the top-16-bit prefix set (low bits zero).
/// Used as the canonical "path at height 240" representative for a bucket.
#[inline]
fn path_from_top16(top16: u16) -> [u8; 32] {
    let mut p = [0u8; 32];
    let bytes = top16.to_be_bytes();
    p[0] = bytes[0];
    p[1] = bytes[1];
    p
}

/// A Sparse Merkle Tree (SMT) backed UTXO accumulator over the 2^256
/// keyspace.
///
/// Provides a cryptographically verifiable commitment to the current set of
/// unspent transaction outputs. Supports O(log N) insert / remove / contains
/// and O(1) root lookup.
///
/// # Abstract state
///
/// ```text
///   SMT ≜ ⟨ C : ℙ Hash ⟩
///   inv:  ∀ c ∈ C · |c| = 32
/// ```
///
/// # Concrete representation
///
/// The accumulator holds three fields:
/// 1. `coins` — the canonical `im::OrdSet` of all UTXO IDs. Cryptographic
///    state is fully determined by this field; everything else is derived.
///    `im::OrdSet` gives O(1) clones for chain reorgs and O(log N)
///    `insert`/`remove`/`contains`, while iterating in lexicographic
///    (i.e. tree-leaf) order.
/// 2. `buckets` — an `im::HashMap` grouping coins by their top 16 bits.
///    Used for O(K log K) on-the-fly subtree hashing of the bottom 240
///    levels in [`prove`](Self::prove) and `update_cache_for_bucket`.
/// 3. `nodes` — the cached top-16-level interior node hashes, keyed by
///    `(height, top_16_bits)`. Strictly bounded to ≤ 131,071 entries.
///
/// `buckets` and `nodes` are both `#[serde(skip)]` and reconstructible from
/// `coins` alone via [`rebuild_tree`](Self::rebuild_tree).
///
/// # Hashing version (V1 vs V2)
///
/// `UtxoAccumulator` is *stateless* with respect to V1/V2 hashing: the
/// `is_v2` flag is supplied to every method that hashes anything
/// ([`insert`](Self::insert), [`remove`](Self::remove),
/// [`root`](Self::root), [`prove`](Self::prove),
/// [`rebuild_tree`](Self::rebuild_tree), and the implementation-private
/// `update_cache_for_bucket` / `get_node`).
///
/// **A consequence of statelessness**: the `nodes` cache is correct for
/// exactly one hashing mode at a time — whichever mode produced the most
/// recent `update_cache_for_bucket` call. Mixing `is_v2 = true` and
/// `is_v2 = false` calls within the same accumulator's lifetime corrupts
/// the cache. Callers must therefore use a single, height-derived `is_v2`
/// value for any one accumulator instance, and call
/// [`rebuild_tree`](Self::rebuild_tree) when the chain crosses the V2
/// activation boundary. In this codebase that responsibility lives in two
/// places only:
/// * `apply_batch_internal` in `state.rs` rebuilds when
///   `state.height == V2_ACTIVATION_HEIGHT`.
/// * `deserialize_state` in `storage.rs` rebuilds with whatever mode the
///   loaded state's height implies.
///
/// # Cache reconstruction after deserialisation
///
/// `nodes` and `buckets` are `#[serde(skip)]`. After deserialisation they
/// are empty; callers **must** invoke
/// [`rebuild_tree`](Self::rebuild_tree) with the chain's current `is_v2`
/// flag before calling [`root`](Self::root) or [`prove`](Self::prove).
/// Otherwise both will return the empty-tree hash regardless of the actual
/// coin set — a consensus-corrupting bug. `Default::default()` likewise
/// yields an accumulator whose caches need no rebuild only if `coins` is
/// also empty (which it is, since `Default` is the zero value).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct UtxoAccumulator {
    /// The canonical set of all unspent coin IDs. Cryptographic state of
    /// the accumulator is fully determined by this field.
    coins: im::OrdSet<[u8; 32]>,

    /// Top 16 levels of the SMT cache, keyed by `(height, top_16_bits)` to
    /// eliminate 30 bytes of zero-padding per entry. Empty subtrees are
    /// never stored; absence in the map means "use `get_empty_hash(height,
    /// is_v2)`". This keeps the cache size proportional to the number of
    /// *occupied* buckets rather than the full 2^16 keyspace.
    #[serde(skip)]
    nodes: im::HashMap<(u16, u16), [u8; 32]>,

    /// Coins grouped by their top 16 bits. Empty buckets are removed.
    /// Used for instant O(1) bucket extraction during the dynamic folding
    /// of the bottom 240 levels.
    #[serde(skip)]
    buckets: im::HashMap<u16, im::OrdSet<[u8; 32]>>,
}

/// Equality is defined purely on the abstract state (the coin set). The
/// derived caches are deliberately ignored: two accumulators are equal iff
/// they hold the same coins, regardless of which order those coins were
/// inserted in or which hashing mode their caches reflect.
impl PartialEq for UtxoAccumulator {
    fn eq(&self, other: &Self) -> bool {
        self.coins == other.coins
    }
}
impl Eq for UtxoAccumulator {}

impl UtxoAccumulator {
    /// Construct an empty accumulator.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result.coins = ∅
    /// ```
    pub fn new() -> Self {
        Self {
            coins: im::OrdSet::new(),
            nodes: im::HashMap::new(),
            buckets: im::HashMap::new(),
        }
    }

    /// Construct an accumulator from any iterable of coins, populated under
    /// the given hashing mode.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result.coins = { c | c ∈ coins? }
    ///          result.nodes consistent with result.coins under is_v2?
    /// ```
    ///
    /// Equivalent to constructing an empty accumulator and inserting each
    /// coin under `is_v2`. Two implementation choices were considered:
    /// (a) collect all coins first then call `rebuild_tree(is_v2)` once,
    /// or (b) call `insert(c, is_v2)` per coin. Option (a) is faster
    /// asymptotically (O(N) bucket grouping, B cache ripples) but has a
    /// large constant from the `BTreeSet → im::OrdSet` conversion; option
    /// (b) has a per-call cost but reuses already-warm cache entries when
    /// many coins land in the same bucket. We use (b) here because the
    /// difference doesn't matter at the call sites that use this method
    /// (mostly tests and the genesis block).
    pub fn from_set(coins: impl IntoIterator<Item = [u8; 32]>, is_v2: bool) -> Self {
        let mut acc = Self::new();
        for c in coins {
            acc.insert(c, is_v2);
        }
        acc
    }

    /// Reconstruct an accumulator from a pre-existing canonical coin set,
    /// rebuilding the SMT cache in one bulk pass under the given hashing mode.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result.coins = coins?
    ///          result.nodes, result.buckets consistent with result.coins
    ///                                       under is_v2?
    /// ```
    ///
    /// Faster than [`from_set`](Self::from_set) when the canonical set is
    /// already in `im::OrdSet` form, because it avoids the per-insert
    /// `im::*` copy-on-write and runs the bulk-staging path of
    /// [`rebuild_tree`](Self::rebuild_tree) directly. Intended for storage
    /// migration code where the caller already deserialised the canonical
    /// `OrdSet` from disk.
    #[doc(hidden)]
    pub fn from_canonical_coins(coins: im::OrdSet<[u8; 32]>, is_v2: bool) -> Self {
        let mut acc = Self {
            coins,
            nodes: im::HashMap::new(),
            buckets: im::HashMap::new(),
        };
        acc.rebuild_tree(is_v2);
        acc
    }

    /// Number of coins currently in the accumulator.
    pub fn len(&self) -> usize {
        self.coins.len()
    }

    /// `true` iff the coin set is empty.
    pub fn is_empty(&self) -> bool {
        self.coins.is_empty()
    }

    /// Membership test (does not depend on hashing mode).
    ///
    /// # Formal specification
    /// ```text
    ///   post:  result = (coin? ∈ coins)
    /// ```
    ///
    /// O(log N).
    pub fn contains(&self, coin: &[u8; 32]) -> bool {
        self.coins.contains(coin)
    }

    /// Iterator over coins in ascending lexicographic order. Because of the
    /// big-endian bit convention, this is also tree-leaf order.
    pub fn iter(&self) -> impl Iterator<Item = &[u8; 32]> {
        self.coins.iter()
    }

    /// Consume the accumulator, returning all coins as a `Vec` in ascending
    /// lexicographic order.
    pub fn into_vec(self) -> Vec<[u8; 32]> {
        self.coins.into_iter().collect()
    }

    /// Completely rebuild the `buckets` and `nodes` caches from `coins`,
    /// under the given hashing mode.
    ///
    /// # Formal specification
    /// ```text
    ///   pre:   true
    ///   post:  buckets' and nodes' satisfy the cache derivability invariant
    ///          for self.coins under is_v2?
    ///          self.coins unchanged
    /// ```
    ///
    /// # When to call
    ///
    /// * Immediately after deserialisation, before any `root()` or `prove()`
    ///   call. Without this step, both will return the empty-tree hash even
    ///   though `coins` is populated.
    /// * On the V2 activation block, to switch the cache from V1 hashing to
    ///   V2 hashing without changing the coin set.
    /// * After any out-of-band mutation of `coins` (the public API does not
    ///   expose any such mutation; this is mostly defensive).
    ///
    /// # Reasoning
    ///
    /// The two-pass design (stage in `std::HashMap`+`BTreeSet`, then convert
    /// to `im::*`) avoids a quadratic blow-up that the naïve "insert each
    /// coin into `buckets`" loop suffers from: every insert into an
    /// `im::OrdSet` cloned out of `im::HashMap` and reinserted is O(log N)
    /// plus a copy-on-write fan, and over N coins that gives roughly
    /// `O(N · log N · log N)`. Staging in `std` first avoids the
    /// copy-on-write entirely, so the total cost is `O(N + B · log B + B · 16)`
    /// where `B` is the number of occupied buckets.
    ///
    /// # Complexity
    ///
    /// `O(N + B · log B + B · 16)` where `N = #coins`, `B = #occupied buckets`.
    /// The bucket-build pass is O(N) hash-map inserts; the cache pass is one
    /// 16-level ripple per occupied bucket.
    pub fn rebuild_tree(&mut self, is_v2: bool) {
        self.nodes.clear();
        self.buckets.clear();

        // First pass: group coins by 16-bit prefix using a temporary std
        // HashMap of BTreeSets to avoid the per-insert clone-on-write cost
        // of im::*. The BTreeSet keeps each bucket sorted, so the eventual
        // im::OrdSet construction is a single ordered insertion sweep.
        let mut staging: std::collections::HashMap<u16, std::collections::BTreeSet<[u8; 32]>> =
            std::collections::HashMap::new();
        for &coin in self.coins.iter() {
            let prefix = u16::from_be_bytes([coin[0], coin[1]]);
            staging.entry(prefix).or_default().insert(coin);
        }

        // Second pass: move staging into the persistent `buckets` map and
        // ripple each bucket's hash into the cache.
        for (prefix, set) in staging {
            let mut bucket = im::OrdSet::new();
            for c in set {
                bucket.insert(c);
            }
            self.buckets.insert(prefix, bucket.clone());
            self.update_cache_for_bucket(prefix, &bucket, is_v2);
        }
    }

    /// Insert a coin under the given hashing mode. Returns `true` if newly
    /// added, `false` if the coin was already present.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  coins' = coins ∪ {coin?}
    ///          result = (coin? ∉ coins)
    ///          buckets' and nodes' remain consistent with coins' under is_v2?
    /// ```
    ///
    /// # Reasoning
    ///
    /// Insert into the canonical set first; if the coin was already there,
    /// no further work is needed (idempotent set semantics, returns
    /// `false`). Otherwise locate the coin's bucket, insert, and ripple the
    /// updated bucket hash up through the 16 cached levels via
    /// `update_cache_for_bucket`.
    ///
    /// # Complexity
    ///
    /// O(K log K + 16) where K is the bucket population — i.e. O(log N)
    /// amortised for random coins.
    pub fn insert(&mut self, coin: [u8; 32], is_v2: bool) -> bool {
        if self.coins.insert(coin).is_some() {
            return false;
        }

        let prefix = u16::from_be_bytes([coin[0], coin[1]]);
        let mut bucket = self.buckets.remove(&prefix).unwrap_or_default();
        bucket.insert(coin);
        self.buckets.insert(prefix, bucket.clone());
        self.update_cache_for_bucket(prefix, &bucket, is_v2);
        true
    }

    /// Remove a coin under the given hashing mode. Returns `true` if it was
    /// present, `false` otherwise.
    ///
    /// # Formal specification
    /// ```text
    ///   post:  coins' = coins ∖ {coin?}
    ///          result = (coin? ∈ coins)
    ///          buckets' and nodes' remain consistent with coins' under is_v2?
    /// ```
    ///
    /// # Reasoning
    ///
    /// Remove from the canonical set first; if absent, no further work is
    /// needed (returns `false`). Otherwise locate the coin's bucket, remove,
    /// and either delete the bucket entirely (if it has emptied) or update
    /// it in place. In both cases the cache is rippled accordingly.
    /// Removal of an empty bucket triggers a ripple with an empty bucket,
    /// which causes the cached parent entries to be removed if they
    /// collapse to the empty-subtree hash — preventing stale cache entries.
    pub fn remove(&mut self, coin: &[u8; 32], is_v2: bool) -> bool {
        if self.coins.remove(coin).is_none() {
            return false;
        }

        let prefix = u16::from_be_bytes([coin[0], coin[1]]);
        let mut bucket = self.buckets.remove(&prefix).unwrap_or_default();
        bucket.remove(coin);

        if bucket.is_empty() {
            // Bucket gone entirely — leave it out of `buckets` and ripple
            // an empty hash through the cache so any cached ancestors that
            // collapse to empty-subtree hashes get evicted.
            self.update_cache_for_bucket(prefix, &im::OrdSet::new(), is_v2);
        } else {
            self.buckets.insert(prefix, bucket.clone());
            self.update_cache_for_bucket(prefix, &bucket, is_v2);
        }
        true
    }

    /// The current SMT root under the given hashing mode.
    ///
    /// # Formal specification
    /// ```text
    ///   pre:   true
    ///   post:  result = the canonical SMT root for self.coins under is_v2?
    ///          self unchanged
    /// ```
    ///
    /// # Caller responsibility
    ///
    /// The caller is responsible for ensuring the cache was last updated
    /// under the same `is_v2` value. Calling `root(true)` on an
    /// accumulator whose cache was built under V1 will return a
    /// nonsensical mixture-of-modes hash. See the struct-level docs for
    /// when rebuilds happen.
    ///
    /// # Complexity
    ///
    /// O(1) — reads the cached node at `(SMT_HEIGHT, 0)`, falling back to
    /// the empty-tree hash if the cache holds no entry there (i.e. the
    /// accumulator is empty).
    pub fn root(&self, is_v2: bool) -> [u8; 32] {
        self.get_node(SMT_HEIGHT as u16, &[0u8; 32], is_v2)
    }

    /// Generate an inclusion proof for `coin` under the given hashing mode.
    ///
    /// # Formal specification
    /// ```text
    ///   pre:   coin? ∈ coins
    ///   post:  #result.siblings = 256
    ///          verify_utxo_proof(coin?, result, root(is_v2?), is_v2?) = true
    ///          self unchanged
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Locate the coin's bucket (top 16 bits of `coin`).
    /// 2. Use [`extract_path_siblings`] to walk the bottom 240 levels in a
    ///    single recursive descent over the bucket, emitting siblings
    ///    bottom-up. This produces siblings 0..240.
    /// 3. For the top 16 levels (240..256), look up siblings directly from
    ///    the cache. Empty-subtree positions return the empty-tree hash
    ///    via `get_node`'s fallback.
    ///
    /// # Complexity
    ///
    /// O(K log K + 16) where K is the bucket population. The single-descent
    /// approach replaces an earlier "rebuild every level independently"
    /// pattern that was O(K log K · 240); see `extract_path_siblings`'s
    /// docs for detail.
    pub fn prove(&self, coin: &[u8; 32], is_v2: bool) -> Result<UtxoProof> {
        if !self.contains(coin) {
            bail!("coin not in accumulator");
        }

        let mut siblings = Vec::with_capacity(SMT_HEIGHT);

        // ── Bottom 240 levels: single recursive descent through the bucket ──
        let prefix = u16::from_be_bytes([coin[0], coin[1]]);
        let bucket = self.buckets.get(&prefix).cloned().unwrap_or_default();
        let bucket_coins: Vec<[u8; 32]> = bucket.iter().copied().collect();
        extract_path_siblings(CACHE_MIN_HEIGHT, &bucket_coins, coin, &mut siblings, is_v2);

        // After the descent, `siblings` holds entries for h ∈ {0, …, 239} in
        // bottom-up order, exactly the convention the verifier expects.
        debug_assert_eq!(siblings.len(), CACHE_MIN_HEIGHT);

        // ── Top 16 levels: cache reads ──
        //
        // We start with `current_path` equal to the path of the coin's
        // height-240 ancestor — i.e. just the top 16 bits with everything
        // below cleared. At each level h ∈ {240, …, 255} the sibling lives
        // at `current_path` with bit h flipped, then we move up by clearing
        // bit h to obtain the parent's path.
        let mut current_path = path_from_top16(prefix);

        for h in CACHE_MIN_HEIGHT..SMT_HEIGHT {
            let bit = get_bit(coin, h);

            let mut sibling_path = current_path;
            flip_bit(&mut sibling_path, h);
            // The sibling is itself at height h, so its path has bits 0..h
            // cleared. `current_path` already satisfies that (we cleared bit
            // (h-1) on the previous iteration, and bits below 240 were never
            // set), so flipping bit h preserves the invariant.

            let sibling_hash = self.get_node(h as u16, &sibling_path, is_v2);
            siblings.push(ProofElement {
                hash: sibling_hash,
                is_right: bit == 0,
            });

            // Move up to the parent at height h+1 by clearing bit h.
            mask_lower_bits(&mut current_path, h + 1);
        }

        debug_assert_eq!(siblings.len(), SMT_HEIGHT);

        Ok(UtxoProof {
            leaf_index: 0,                // unused for SMT; reserved
            leaf_count: self.coins.len(), // informational
            siblings,
        })
    }

    /// Read a cached interior-node hash at `(height, path)`. Falls back to
    /// the canonical empty-subtree hash for the appropriate mode if the
    /// entry is absent (which by the cache eviction rule means the subtree
    /// is fully empty).
    ///
    /// # Preconditions
    /// * `height ∈ CACHE_MIN_HEIGHT..=SMT_HEIGHT`
    /// * `path` is a valid height-`h` SMT path (low `h` bits cleared).
    fn get_node(&self, height: u16, path: &[u8; 32], is_v2: bool) -> [u8; 32] {
        debug_assert!((CACHE_MIN_HEIGHT as u16..=SMT_HEIGHT as u16).contains(&height));
        let key = cache_key(height as usize, path);
        self.nodes
            .get(&key)
            .copied()
            .unwrap_or_else(|| get_empty_hash(height as usize, is_v2))
    }

    /// Recompute the cache contributions for one bucket under the given
    /// hashing mode.
    ///
    /// Computes the bucket's height-240 hash from scratch (cheap because
    /// `compute_sparse_subtree` does only `O(K log K)` work for `K` coins
    /// in the bucket), then ripples it up through the 16 cached levels,
    /// updating each parent according to the sibling already in the cache.
    ///
    /// # Cache eviction
    ///
    /// At every level we compare the new hash against the canonical empty-
    /// subtree hash for that level under the current mode; if they match,
    /// we *remove* the entry instead of inserting it. This keeps the cache
    /// proportional to the number of occupied subtrees, not the keyspace,
    /// and ensures that an accumulator that has been emptied holds no
    /// stale cache entries.
    ///
    /// # Preconditions
    /// * `bucket` contains exactly the coins whose top-16 bits equal
    ///   `prefix` in `self.coins` (or is empty if the bucket has just been
    ///   emptied).
    fn update_cache_for_bucket(
        &mut self,
        prefix: u16,
        bucket: &im::OrdSet<[u8; 32]>,
        is_v2: bool,
    ) {
        let bucket_coins: Vec<[u8; 32]> = bucket.iter().copied().collect();
        let bucket_hash = compute_sparse_subtree(CACHE_MIN_HEIGHT, &bucket_coins, is_v2);

        let mut current_path = path_from_top16(prefix);

        // Insert / evict the height-240 entry.
        let key_240 = cache_key(CACHE_MIN_HEIGHT, &current_path);
        if bucket_hash == get_empty_hash(CACHE_MIN_HEIGHT, is_v2) {
            self.nodes.remove(&key_240);
        } else {
            self.nodes.insert(key_240, bucket_hash);
        }

        let mut hash_to_ripple = bucket_hash;

        for h in CACHE_MIN_HEIGHT..SMT_HEIGHT {
            let bit = get_bit(&current_path, h);

            // Sibling at this level: flip bit h.
            let mut sibling_path = current_path;
            flip_bit(&mut sibling_path, h);
            let sibling_hash = self.get_node(h as u16, &sibling_path, is_v2);

            // Compose the parent. `bit` tells us which side WE are on.
            let parent_hash = if bit == 0 {
                hash_node(&hash_to_ripple, &sibling_hash, is_v2)
            } else {
                hash_node(&sibling_hash, &hash_to_ripple, is_v2)
            };

            // Move up: clear bit h to obtain the parent's path.
            mask_lower_bits(&mut current_path, h + 1);

            // Insert / evict the parent.
            let parent_key = cache_key(h + 1, &current_path);
            if parent_hash == get_empty_hash(h + 1, is_v2) {
                self.nodes.remove(&parent_key);
            } else {
                self.nodes.insert(parent_key, parent_hash);
            }

            hash_to_ripple = parent_hash;
        }
    }
}

/// SMT inclusion proof.
///
/// The proof is a fixed-length sequence of 256 sibling hashes plus side
/// indicators. The verifier folds them bottom-up against the coin to
/// reconstruct the SMT root.
///
/// `leaf_index` and `leaf_count` are *informational only*: they describe
/// the state of the accumulator at the moment of proof generation but are
/// not strictly required to verify against the expected root. They are
/// retained for compatibility and diagnostics.
///
/// # Wire format stability
///
/// This struct is `Serialize + Deserialize` and may travel over the
/// network to light clients. The on-wire layout is exactly the field
/// order shown here; do not reorder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtxoProof {
    /// Reserved; always `0` for SMT proofs (the SMT path is determined by
    /// the coin itself, not by an arbitrary leaf index).
    pub leaf_index: usize,
    /// Number of coins in the accumulator at the moment of proof
    /// generation. Informational; ignored by [`verify_utxo_proof`].
    pub leaf_count: usize,
    /// Exactly 256 sibling elements, in bottom-up order: index `h` holds
    /// the sibling at SMT height `h`.
    pub siblings: Vec<ProofElement>,
}

/// Verify an SMT inclusion proof against a known root, under the given
/// hashing mode.
///
/// # Formal specification
///
/// ```text
///   verify_utxo_proof : Hash × UtxoProof × Hash × 𝔹 → 𝔹
///   post:  result = true ⇒ ∃ U : UtxoAccumulator ·
///                          U.root(is_v2?) = expected_root  ∧  coin? ∈ U.coins
///          (soundness: assuming BLAKE3 collision resistance)
/// ```
///
/// # Algorithm
///
/// 1. Sanity-check the proof shape: exactly `SMT_HEIGHT = 256` siblings.
/// 2. For each level `h ∈ 0..256`, verify that `siblings[h].is_right` is
///    consistent with bit `h` of `coin` (a defensive structural check
///    that prevents an attacker from supplying siblings in a plausible-
///    looking but path-inconsistent order), then fold the sibling into
///    the running hash via [`hash_node`] under the supplied mode.
/// 3. Compare the final hash with `expected_root`.
///
/// # Cross-mode behaviour
///
/// A proof generated under mode `M` will only verify when this function
/// is called with the same `M`. A V2-mode proof verified with `is_v2 =
/// false` will produce a chain of V1 hashes that cannot reconstruct the
/// V2 root (and vice versa) — this is the property tested by
/// `utxo_v2_proof_does_not_verify_against_v1_root` in the test module.
pub fn verify_utxo_proof(
    coin: &[u8; 32],
    proof: &UtxoProof,
    expected_root: &[u8; 32],
    is_v2: bool,
) -> bool {
    if proof.siblings.len() != SMT_HEIGHT {
        return false;
    }

    let mut current = *coin;
    for (h, elem) in proof.siblings.iter().enumerate() {
        // Structural sanity: bit `h` of the coin determines which side WE
        // are on at level h, hence which side the sibling is on. This
        // guards against a malformed proof that happens to reconstruct
        // the right hash by accident if the order were ambiguous.
        let bit = get_bit(coin, h);
        let expected_is_right = bit == 0;
        if elem.is_right != expected_is_right {
            return false;
        }

        current = if elem.is_right {
            hash_node(&current, &elem.hash, is_v2)
        } else {
            hash_node(&elem.hash, &current, is_v2)
        };
    }
    current == *expected_root
}

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                                                                          ║
// ║   SECTION 5 ─ Tests                                                      ║
// ║                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// Test strategy: most behavioural properties (round-trip proof verification,
// insert/remove/reinsert idempotence, deserialise-then-rebuild correctness,
// etc.) should hold identically under V1 and V2 hashing. Those tests use the
// `for_each_mode` helper, which runs the test body twice — once with
// `is_v2 = false` and once with `is_v2 = true`. A second category of tests
// asserts that V1 and V2 are *distinct* hash universes (different roots, no
// cross-mode proof verification); those tests pin the modes explicitly.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::hash;

    /// Run a closure under both V1 and V2 hashing modes. Use this for any
    /// test whose correctness should be invariant to hashing mode.
    fn for_each_mode<F: Fn(bool)>(f: F) {
        f(false);
        f(true);
    }

    // ── MMR ────────────────────────────────────────────────────────────────

    #[test]
    fn mmr_append_and_root() {
        for_each_mode(|v2| {
            let mut mmr = MerkleMountainRange::new();
            let h1 = hash(b"leaf1");
            let h2 = hash(b"leaf2");

            mmr.append(&h1, v2);
            // For a one-leaf MMR there is exactly one peak (the leaf itself),
            // and bagging a one-element peak list is the identity. Hence
            // root == leaf, regardless of mode.
            assert_eq!(mmr.root(v2), h1);

            mmr.append(&h2, v2);
            assert_ne!(mmr.root(v2), h1);

            mmr.append(&hash(b"leaf3"), v2);
            assert_eq!(mmr.leaf_count(), 3);
        });
    }

    #[test]
    fn mmr_proof_round_trip() {
        for_each_mode(|v2| {
            let mut mmr = MerkleMountainRange::new();
            let leaves: Vec<[u8; 32]> = (0..8u8).map(|i| hash(&[i])).collect();
            for leaf in &leaves {
                mmr.append(leaf, v2);
            }
            let root = mmr.root(v2);

            // Post-order positions of the eight leaves in an MMR of leaf_count=8:
            //   leaf 0 → pos  0
            //   leaf 1 → pos  1
            //   leaf 2 → pos  3
            //   leaf 3 → pos  4
            //   leaf 4 → pos  7
            //   leaf 5 → pos  8
            //   leaf 6 → pos 10
            //   leaf 7 → pos 11
            let positions = [0u64, 1, 3, 4, 7, 8, 10, 11];
            for (i, leaf) in leaves.iter().enumerate() {
                let proof = mmr.prove(positions[i]).unwrap();
                assert!(
                    verify_mmr_proof(leaf, &proof, &root, v2),
                    "v2={} proof failed for leaf {}",
                    v2,
                    i
                );
            }
        });
    }

    #[test]
    fn mmr_v1_v2_roots_differ() {
        // Same set of leaves, different hashing modes → different roots
        // except with negligible probability. This is the property that
        // makes V1 and V2 distinct hash universes.
        let mut mmr = MerkleMountainRange::new();
        for i in 0..5u8 {
            mmr.append(&hash(&[i]), false);
        }
        let v1_root = mmr.root(false);

        // The V2 MMR must have its leaves appended under V2 to be meaningful;
        // mixing modes within a single MMR's interior hashes is undefined.
        let mut mmr2 = MerkleMountainRange::new();
        for i in 0..5u8 {
            mmr2.append(&hash(&[i]), true);
        }
        let v2_root = mmr2.root(true);

        assert_ne!(v1_root, v2_root);
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
    fn pos_height_spot_checks() {
        // Heights of the first 11 positions:
        //   pos     0 1 2 3 4 5 6 7 8 9 10
        //   height  0 0 1 0 0 1 2 0 0 1  0
        let expected = [0u32, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0];
        for (pos, &h) in expected.iter().enumerate() {
            assert_eq!(pos_height(pos as u64), h, "pos_height({})", pos);
        }
    }

    #[test]
    fn mmr_proof_invalid_position() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(&hash(b"a"), false);
        // Out-of-range position must fail with a clean error, not a panic.
        assert!(mmr.prove(999).is_err());
    }

    #[test]
    fn mmr_proof_for_internal_node_rejected() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(&hash(b"a"), true);
        mmr.append(&hash(b"b"), true);
        // After two appends, position 2 is the merged internal node, not a leaf.
        // Asking for a proof of an internal-node position must be rejected.
        assert!(mmr.prove(2).is_err());
    }

    // ── UTXO Accumulator ───────────────────────────────────────────────────

    #[test]
    fn utxo_accumulator_basics() {
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            let c1 = hash(b"coin1");
            let c2 = hash(b"coin2");
            let c3 = hash(b"coin3");

            assert!(acc.insert(c1, v2));
            assert!(acc.insert(c2, v2));
            assert!(acc.insert(c3, v2));
            assert!(!acc.insert(c1, v2)); // duplicate

            assert_eq!(acc.len(), 3);
            assert!(acc.contains(&c1));

            let r1 = acc.root(v2);
            assert!(acc.remove(&c2, v2));
            assert_ne!(r1, acc.root(v2));
        });
    }

    #[test]
    fn utxo_insert_remove_reinsert_same_root() {
        // Set membership is idempotent: inserting → removing → reinserting
        // the same coin must produce the same root the original insert did.
        // This catches subtle bugs where remove leaves cache state inconsistent.
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            let c1 = hash(b"coin1");
            let c2 = hash(b"coin2");

            acc.insert(c1, v2);
            acc.insert(c2, v2);
            let root_before = acc.root(v2);

            acc.remove(&c1, v2);
            assert_ne!(root_before, acc.root(v2));

            acc.insert(c1, v2);
            assert_eq!(root_before, acc.root(v2));
        });
    }

    #[test]
    fn utxo_empty_root_is_deterministic() {
        for_each_mode(|v2| {
            let a = UtxoAccumulator::new();
            let b = UtxoAccumulator::new();
            assert_eq!(a.root(v2), b.root(v2));
        });
    }

    #[test]
    fn utxo_root_takes_shared_reference() {
        // Compile-time check: root() should be callable on a shared reference.
        // Protects against an accidental future regression where someone adds
        // `&mut self` to root() and breaks every read-only call site.
        let acc = UtxoAccumulator::new();
        let _r: [u8; 32] = acc.root(false);
        let r2 = (&acc).root(false);
        assert_eq!(_r, r2);
    }

    #[test]
    fn utxo_v1_v2_roots_differ_for_nonempty_set() {
        // Same coin set, different hashing modes → different roots.
        let mut a = UtxoAccumulator::new();
        let mut b = UtxoAccumulator::new();
        for i in 0..5u8 {
            a.insert(hash(&[i]), false);
            b.insert(hash(&[i]), true);
        }
        assert_ne!(
            a.root(false),
            b.root(true),
            "V1 and V2 must produce distinct roots over the same coin set"
        );
    }

    #[test]
    fn utxo_proof_round_trip() {
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            let coins: Vec<[u8; 32]> = (0..10u8).map(|i| hash(&[i])).collect();
            for c in &coins {
                acc.insert(*c, v2);
            }
            let root = acc.root(v2);

            for c in &coins {
                let proof = acc.prove(c, v2).unwrap();
                assert!(verify_utxo_proof(c, &proof, &root, v2));
            }
        });
    }

    #[test]
    fn utxo_v2_proof_does_not_verify_against_v1_root() {
        // A proof generated under V2 must not verify when the verifier is
        // told it's V1. This is the property that makes V1/V2 a true
        // hard fork rather than a rebrand.
        let mut acc = UtxoAccumulator::new();
        let c = hash(b"x");
        acc.insert(c, true);
        let v2_root = acc.root(true);
        let v2_proof = acc.prove(&c, true).unwrap();
        assert!(!verify_utxo_proof(&c, &v2_proof, &v2_root, false));
    }

    #[test]
    fn utxo_accumulator_memory_safety_check() {
        let mut acc = UtxoAccumulator::new();
        // 1,000 random coins. Theoretical worst-case for cache occupancy is
        // 2^17 - 1 = 131,071 entries; with random 16-bit prefixes and
        // N=1000 we expect roughly N · 17 ≈ 17,000 actual entries.
        for i in 0..1000u32 {
            let mut h = blake3::Hasher::new();
            h.update(&i.to_le_bytes());
            acc.insert(*h.finalize().as_bytes(), true);
        }

        // Strict mathematical upper bound on cache size, regardless of input.
        assert!(
            acc.nodes.len() < 131_072,
            "Cache exceeded its tight upper bound!"
        );

        // Root and a sample proof must still verify cleanly.
        let root = acc.root(true);
        let mut h = blake3::Hasher::new();
        h.update(&42u32.to_le_bytes());
        let coin = *h.finalize().as_bytes();

        let proof = acc.prove(&coin, true).unwrap();
        assert!(verify_utxo_proof(&coin, &proof, &root, true));
    }

    #[test]
    fn utxo_wrong_coin_fails() {
        let mut acc = UtxoAccumulator::new();
        acc.insert(hash(b"coin1"), true);
        acc.insert(hash(b"coin2"), true);
        let root = acc.root(true);

        // A proof for `coin1` must not verify if the verifier substitutes
        // a different coin value (this would otherwise be a soundness break).
        let proof = acc.prove(&hash(b"coin1"), true).unwrap();
        assert!(!verify_utxo_proof(&hash(b"fake"), &proof, &root, true));
    }

    #[test]
    fn utxo_proof_against_wrong_root_fails() {
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            let c = hash(b"coin1");
            acc.insert(c, v2);
            let proof = acc.prove(&c, v2).unwrap();
            let fake_root = hash(b"not the root");
            assert!(!verify_utxo_proof(&c, &proof, &fake_root, v2));
        });
    }

    #[test]
    fn utxo_from_set_matches_iterative_inserts() {
        // Two different construction paths over the same coin set must
        // produce equal roots — ensures `from_set` doesn't accidentally
        // diverge from the canonical `new + insert+` construction.
        for_each_mode(|v2| {
            let coins: Vec<[u8; 32]> = (0..5u8).map(|i| hash(&[i])).collect();
            let acc1 = UtxoAccumulator::from_set(coins.clone(), v2);
            let mut acc2 = UtxoAccumulator::new();
            for c in &coins {
                acc2.insert(*c, v2);
            }
            assert_eq!(acc1.root(v2), acc2.root(v2));
        });
    }

    #[test]
    fn utxo_remove_all_returns_to_empty_root() {
        // After inserting and then removing every coin, the root must
        // collapse exactly back to the empty-tree hash. This catches any
        // residual cache state that survives removal.
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            let empty_root = acc.root(v2);
            let coins: Vec<[u8; 32]> = (0..5u8).map(|i| hash(&[i])).collect();
            for c in &coins {
                acc.insert(*c, v2);
            }
            for c in &coins {
                acc.remove(c, v2);
            }
            assert_eq!(acc.root(v2), empty_root);
            assert!(acc.is_empty());
        });
    }

    #[test]
    fn utxo_proof_non_member_fails() {
        // prove() must error for any coin not in the set, in both modes.
        let acc = UtxoAccumulator::new();
        let coin = hash(b"not in set");
        assert!(acc.prove(&coin, true).is_err());
        assert!(acc.prove(&coin, false).is_err());
    }

    #[test]
    fn utxo_insertion_order_independence() {
        // The abstract state is a *set*, not a sequence — inserting the
        // same coins in different orders must yield identical roots.
        for_each_mode(|v2| {
            let coins: Vec<[u8; 32]> = (0..32u8).map(|i| hash(&[i])).collect();

            let mut a = UtxoAccumulator::new();
            for c in &coins {
                a.insert(*c, v2);
            }

            let mut b = UtxoAccumulator::new();
            for c in coins.iter().rev() {
                b.insert(*c, v2);
            }

            assert_eq!(a.root(v2), b.root(v2));
            assert_eq!(a, b);
        });
    }

    #[test]
    fn utxo_deserialize_then_rebuild_recovers_root() {
        // Critical regression test for the deserialisation footgun: a
        // round-trip through bincode must produce an accumulator that
        // — once the storage layer calls `rebuild_tree` — computes the
        // SAME root, NOT the empty-tree root. This is the contract the
        // `storage.rs::deserialize_state` function relies on.
        for_each_mode(|v2| {
            let mut acc = UtxoAccumulator::new();
            for i in 0..50u8 {
                acc.insert(hash(&[i]), v2);
            }
            let original_root = acc.root(v2);
            assert_ne!(original_root, get_empty_hash(SMT_HEIGHT, v2));

            let bytes = bincode::serialize(&acc).expect("serialize");
            let mut restored: UtxoAccumulator =
                bincode::deserialize(&bytes).expect("deserialize");
            // Caches start empty post-deserialise; rebuild before use.
            // (This is what the storage layer does on every load.)
            restored.rebuild_tree(v2);

            assert_eq!(restored.len(), acc.len());
            assert_eq!(
                restored.root(v2),
                original_root,
                "deserialised accumulator computed the wrong root after rebuild"
            );

            // Proofs from the restored accumulator must also verify.
            let probe = hash(&[7u8]);
            let proof = restored.prove(&probe, v2).unwrap();
            assert!(verify_utxo_proof(&probe, &proof, &original_root, v2));
        });
    }

    #[test]
    fn utxo_proof_length_guard() {
        // Malformed proofs with the wrong sibling count must be rejected
        // before any hashing happens. This guards against a denial-of-
        // service vector where a peer feeds the verifier a million-element
        // proof and forces a million BLAKE3 calls.
        let coin = hash(b"x");
        let bad_proof = UtxoProof {
            leaf_index: 0,
            leaf_count: 1,
            siblings: vec![
                ProofElement {
                    hash: [0u8; 32],
                    is_right: true,
                };
                100
            ], // wrong length (must be 256)
        };
        assert!(!verify_utxo_proof(&coin, &bad_proof, &[0u8; 32], false));
        assert!(!verify_utxo_proof(&coin, &bad_proof, &[0u8; 32], true));
    }

    // ── Bit helpers ────────────────────────────────────────────────────────

    #[test]
    fn mask_lower_bits_byte_aligned() {
        let mut p = [0xFFu8; 32];
        mask_lower_bits(&mut p, 8);
        // The last byte should be cleared, all others untouched.
        assert_eq!(p[31], 0);
        assert!(p[0..31].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn mask_lower_bits_non_aligned() {
        let mut p = [0xFFu8; 32];
        mask_lower_bits(&mut p, 9);
        // Last byte fully cleared, byte 30 has its LSB cleared.
        assert_eq!(p[31], 0);
        assert_eq!(p[30], 0xFE);
        assert!(p[0..30].iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn mask_lower_bits_full() {
        let mut p = [0xFFu8; 32];
        mask_lower_bits(&mut p, 256);
        assert_eq!(p, [0u8; 32]);
    }

    #[test]
    fn get_and_flip_bit_round_trip() {
        // Every bit position must round-trip cleanly: get → flip → get → flip → get.
        let mut p = [0u8; 32];
        for i in [0usize, 1, 7, 8, 9, 127, 128, 200, 254, 255] {
            assert_eq!(get_bit(&p, i), 0);
            flip_bit(&mut p, i);
            assert_eq!(get_bit(&p, i), 1);
            flip_bit(&mut p, i);
            assert_eq!(get_bit(&p, i), 0);
        }
    }

    #[test]
    fn empty_hash_table_consistency() {
        // The recursive construction property of the empty-hash table:
        //     empty[h] = hash_node(empty[h-1], empty[h-1])
        // — must hold for every level in both modes. If this ever broke,
        // every SMT root in the system would be wrong.
        for_each_mode(|v2| {
            for h in 1..=SMT_HEIGHT {
                let lower = get_empty_hash(h - 1, v2);
                assert_eq!(get_empty_hash(h, v2), hash_node(&lower, &lower, v2));
            }
        });
    }
}
