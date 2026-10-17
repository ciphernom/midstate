//! # Provably Fair Stratum Pool
//!
//! This module implements a decentralized-auditable mining pool. Unlike traditional 
//! Stratum pools where miners must blindly trust the operator to report shares and 
//! distribute rewards fairly, this pool embeds an SPV-style **Merkle Precommitment** 
//! into every block template.
//!
//! Miners can query the HTTP Audit API (`/api/proof`) to receive a cryptographic 
//! proof that their exact accumulated score was included in the block's coinbase 
//! transaction *before* they begin hashing. If the pool operator lies or omits them, 
//! the miner's local client instantly detects the mismatch and disconnects.
//!
//! ## Security Mitigations Implemented
//! 1. **Replay Protection (`valid_shares`)**: Prevents "Infinite Money" glitches where 
//!    a miner resubmits the same valid nonce millions of times per second.
//! 2. **CPU Exhaustion Defense (`spawn_blocking`)**: Offloads the 1,000,000-iteration 
//!    BLAKE3 VDF from the async reactor, preventing remote DoS attacks.
//! 3. **Conditional Score Deduction**: Prevents "Orphan Theft" by waiting for the 
//!    network to explicitly `HTTP 200 OK` the block before wiping the miners' shares.
//! 4. **Tandem Port Binding**: Binds the Stratum TCP port and Audit HTTP port simultaneously
//!    to guarantee they never desync due to ghost processes holding TCP sockets open.
//! 5. **Checksum-Agnostic Ingestion**: Strips 4-byte UI checksums from user-supplied 
//!    addresses before hashing to prevent silent HTTP 400 rejection loops from the core node.

use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{broadcast, RwLock, Semaphore};
use axum::{extract::{State, Query}, http::Method, routing::{get, post}, Json, Router};
use tower_http::cors::{Any, CorsLayer};

use crate::core::types::{hash, hash_concat, Batch, Extension};
use crate::core::extension::create_extension;

/// The database table storing the cumulative scores of all miners.
/// Key: 32-byte cryptographic address hash. Value: u64 share count.
const SHARES_TABLE: TableDefinition<&[u8; 32], u64> = TableDefinition::new("shares");
/// The database table storing historical blocks found and their exact payouts.
/// Key: Block timestamp (u64). Value: JSON string of payouts.
const BLOCKS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("blocks");
/// The committed per-miner score snapshot for each found block, keyed by the same
/// block timestamp as BLOCKS_TABLE. Stored separately so the frequently-polled
/// /pool/stats payload stays lean; served on demand via /api/block_scores for
/// historical split-verification (proving each payout was proportional to score).
const BLOCK_SCORES_TABLE: TableDefinition<u64, &str> = TableDefinition::new("block_scores");

/// Tracks blocks waiting for maturity (e.g. 10 blocks) to detect network orphans.
/// Key: Block height (u64). Value: JSON string of `{ hash, block_ts, deductions: [[address_hex, deducted_score]] }`
const PENDING_BLOCKS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("pending_blocks");

// ── Stratum Protocol Types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StratumRequest {
    id: Option<u64>,
    method: String,
    params: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct StratumResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

/// What a job commits to on midwimble, when this pool merge-mines. The
/// commitment rides in a coinbase output's salt, beside the pool's own score
/// root in another output; neither knows about the other. See
/// `docs/POOL_MERGED_MINING.md` in midwimble.
#[derive(Clone, Debug)]
struct MergeCommit {
    /// The midwimble template this job's coinbase commits to.
    mining_hash: String,
    /// A share below this target is also a midwimble block.
    target: [u8; 32],
    /// Which coinbase output carries the commitment in its salt.
    commit_index: usize,
    /// The midwimble scores this job's block pays, taken out of
    /// `MW_SHARES_TABLE` when one of its shares becomes that block. Empty when
    /// the block pays only the pool's own address as a fallback.
    paid: Arc<Vec<(String, u64)>>,
    /// Set by the first share that wins this job's midwimble block. A second
    /// would only be a rival block at the same height, and would take the same
    /// scores twice.
    claimed: Arc<std::sync::atomic::AtomicBool>,
    /// The midwimble height of the block this job makes, as the node said.
    height: Option<u64>,
    /// What this job's midwimble block pays, as (address, weight): each
    /// address's share is its weight out of their sum.
    payouts: Arc<Vec<(String, u64)>>,
    /// Why the block pays only the pool's own address, when it does.
    note: Option<String>,
}

/// What the pool is doing about midwimble, for the dashboard. Set each time a
/// job is built.
#[derive(Clone, Debug, Default, PartialEq)]
struct MergeStatus {
    /// Whether the current job commits to a midwimble block.
    active: bool,
    /// Why not; or, while active, why its block pays only the pool.
    note: Option<String>,
    /// The midwimble height of the block the current job makes.
    height: Option<u64>,
    /// When this was set (unix seconds); 0 before the first job.
    updated: u64,
}

impl MergeStatus {
    /// The status of a job just built with `merge` (its midwimble commitment,
    /// if it has one) after planning failed with `error`, if it did.
    fn of_job(merge: Option<&MergeCommit>, error: Option<&str>, now: u64) -> Self {
        match merge {
            Some(m) => MergeStatus { active: true, note: m.note.clone(), height: m.height, updated: now },
            None => MergeStatus {
                active: false,
                note: Some(error.unwrap_or("this job's coinbase had no room for the commitment").to_string()),
                height: None,
                updated: now,
            },
        }
    }
}

/// Represents an active mining job broadcast to all connected Stratum clients.
#[derive(Clone, Debug)]
struct Job {
    job_id: u64,
    /// The 32-byte target that the miner must grind nonces against.
    mining_hash: [u8; 32],
    /// The difficulty threshold to submit a pool share.
    share_target: [u8; 32],
    /// The difficulty threshold to find a full network block.
    network_target: [u8; 32],
    /// The full block template, used to reconstruct the block if a full hash is found.
    batch_template: serde_json::Value,
    /// The chain height this job is mining (the block found from it will have this height).
    height: u64,
    /// The committed (address, score) leaves backing this job's Merkle root. These are
    /// the exact scores the coinbase payout was computed from, captured per job so a
    /// found block can be split-verified against the precommitment it actually used.
    /// Arc so cloning the Job onto the broadcast channel stays cheap.
    committed_scores: Arc<Vec<([u8; 32], u64)>>,
    /// Set when this pool is merge-mining midwimble.
    midwimble: Option<MergeCommit>,
}

// ── Merkle Tree Logic for Share Proofs ──────────────────────────────────────

/// A Merkle tree representing the current state of all miner shares in the pool.
/// 
/// # Reasoning
/// By constructing a Merkle tree of `H(Miner_Address || Score)`, we can compress 
/// the entire state of the pool into a single 32-byte root. This root is embedded 
/// into the salt of the pool's fee coin in the block template. This allows $O(\log N)$ 
/// inclusion proofs for miners to audit their shares.
#[derive(Clone)]
pub struct ShareMerkleTree {
    pub root: [u8; 32],
    pub leaves: Vec<([u8; 32], u64)>, 
    pub layers: Vec<Vec<[u8; 32]>>,
}

impl ShareMerkleTree {
    /// Builds a deterministic Merkle tree from a set of miner shares.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  shares is a valid sequence of (Address, Score) tuples.
    /// Post: The tree is deterministically sorted by Address to prevent malleability.
    ///       root! = The final 32-byte Merkle root.
    /// ```
    ///
    /// ```zed
    ///     BuildTree
    ///     ---------
    ///     shares? : seq (𝔹³² × ℕ₆₄)
    ///     root! : 𝔹³²
    ///
    ///     let sorted_shares = sort_by_address(shares?)
    ///     let L₀ = ⟨ ℋ(addr ⌢ le8(score)) | (addr, score) ∈ sorted_shares ⟩
    ///     let L_{i+1} = ⟨ ℋ(L_i[2k] ⌢ L_i[2k+1]) | k ∈ 0..|L_i|/2 ⟩
    ///     post root! = L_{max}[0]
    /// ```
    fn build(mut shares: Vec<([u8; 32], u64)>) -> Self {
        if shares.is_empty() {
            return Self { root: [0; 32], leaves: vec![], layers: vec![] };
        }
        
        shares.sort_by_key(|&(addr, _)| addr);
        
        let mut current_layer: Vec<[u8; 32]> = shares.iter().map(|(addr, score)| {
            let mut data = [0u8; 40];
            data[0..32].copy_from_slice(addr);
            data[32..40].copy_from_slice(&score.to_le_bytes());
            hash(&data)
        }).collect();

        let mut layers = vec![current_layer.clone()];

        while current_layer.len() > 1 {
            let mut next_layer = Vec::with_capacity((current_layer.len() + 1) / 2);
            for chunk in current_layer.chunks(2) {
                if chunk.len() == 2 {
                    next_layer.push(hash_concat(&chunk[0], &chunk[1]));
                } else {
                    next_layer.push(hash_concat(&chunk[0], &chunk[0]));
                }
            }
            layers.push(next_layer.clone());
            current_layer = next_layer;
        }

        Self { root: current_layer[0], leaves: shares, layers }
    }

    /// Generates an $O(\log N)$ Merkle inclusion proof for a specific miner address.
    /// Returns the leaf index (needed for left/right hashing reconstruction) and the proof array.
    fn generate_proof(&self, address: &[u8; 32]) -> Option<(usize, Vec<[u8; 32]>)> {
        let idx = self.leaves.iter().position(|&(a, _)| a == *address)?;
        let mut proof = Vec::new();
        let mut current_idx = idx;

        for layer in &self.layers[..self.layers.len() - 1] {
            let is_right = current_idx % 2 == 1;
            let sibling_idx = if is_right { current_idx - 1 } else { (current_idx + 1).min(layer.len() - 1) };
            proof.push(layer[sibling_idx]);
            current_idx /= 2;
        }
        Some((idx, proof))
    }
}

// ── Midwimble Payouts (merged mining) ───────────────────────────────────────
//
// When this pool merge-mines midwimble, every accepted share also earns one
// point of midwimble score for the midwimble address its miner gave: the
// third `mining.authorize` parameter, or `midwimble_address` over HTTP and
// WebRTC. A share from a miner who gave none counts toward the pool's own
// `--midwimble-address`, which is where every midwimble reward went before.
//
// Each job asks the midwimble node for a block paying the score table, and
// when one of the job's shares becomes that block, the scores it paid come
// out of the table: proportional payment, one midwimble block at a time.
//
// Score is kept per midwimble address and credited by the connection that did
// the work. Naming somebody else's midstate address therefore gains nothing of
// theirs: the shares you submit pay your midwimble address, theirs pay theirs.

/// Midwimble score per midwimble address (canonical, lowercase).
const MW_SHARES_TABLE: TableDefinition<&str, u64> = TableDefinition::new("midwimble_shares");

/// Midwimble blocks this pool found, for the dashboard, keyed by when (unix
/// milliseconds): JSON `{ height, timestamp, payouts: [{ address, weight }] }`.
const MW_BLOCKS_TABLE: TableDefinition<u64, &str> = TableDefinition::new("midwimble_blocks");

/// How many found midwimble blocks `/pool/stats` lists.
const MW_RECENT_BLOCKS: usize = 10;

/// A midwimble coinbase pays at most this many outputs (its `MAX_GROUP_OUTPUTS`).
const MW_MAX_OUTPUTS: usize = 32;

/// Payout weights sent to midwimble are fractions of this.
const MW_PAYOUT_SCALE: u128 = 1 << 50;

/// Checks a midwimble address the way midwimble decodes one (bech32m, `mw`,
/// version 1, scan and spend keys that are valid, non-identity Ristretto
/// points) and returns it in canonical lowercase.
///
/// Checking here matters: the midwimble node refuses a whole job if it can't
/// decode one payout, so a single bad address would stop the pool paying
/// anybody in midwimble.
pub fn parse_midwimble_address(s: &str) -> Result<String, String> {
    use curve25519_dalek::ristretto::CompressedRistretto;
    use curve25519_dalek::traits::IsIdentity;

    let s = s.trim();
    let lower = s.to_ascii_lowercase();
    if s != lower && s != s.to_ascii_uppercase() {
        return Err("mixed case".into());
    }
    let (hrp, data) = bech32m_decode(&lower)?;
    if hrp != "mw" {
        return Err("a midwimble address starts with mw1".into());
    }
    if data.len() != 97 || data[0] != 1 {
        return Err("not a version 1 midwimble address".into());
    }
    for key in [&data[1..33], &data[33..65]] {
        let point = CompressedRistretto::from_slice(key)
            .ok()
            .and_then(|c| c.decompress())
            .ok_or_else(|| "invalid key".to_string())?;
        if point.is_identity() {
            return Err("invalid key".into());
        }
    }
    Ok(lower)
}

/// Decodes a lowercase bech32m string (BIP 350) into its prefix and bytes.
fn bech32m_decode(s: &str) -> Result<(String, Vec<u8>), String> {
    const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    const BECH32M: u32 = 0x2bc8_30a3;

    let sep = s.rfind('1').ok_or_else(|| "not bech32".to_string())?;
    let (hrp, rest) = (&s[..sep], &s[sep + 1..]);
    if hrp.is_empty() || rest.len() < 6 || s.len() > 1023 || !hrp.bytes().all(|c| (33..=126).contains(&c)) {
        return Err("not bech32".into());
    }
    let mut values = Vec::with_capacity(rest.len());
    for c in rest.bytes() {
        let v = CHARSET.iter().position(|&x| x == c).ok_or_else(|| "invalid character".to_string())?;
        values.push(v as u8);
    }
    let mut checked: Vec<u8> = hrp.bytes().map(|c| c >> 5).collect();
    checked.push(0);
    checked.extend(hrp.bytes().map(|c| c & 31));
    checked.extend_from_slice(&values);
    if bech32_polymod(&checked) != BECH32M {
        return Err("bad checksum".into());
    }
    // Five-bit groups back to bytes; leftover bits must be zero padding.
    let (mut acc, mut bits, mut out) = (0u32, 0u32, Vec::new());
    for &v in &values[..values.len() - 6] {
        acc = ((acc << 5) | v as u32) & 0xfff;
        bits += 5;
        while bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    if bits >= 5 || (acc << (8 - bits)) & 0xff != 0 {
        return Err("bad padding".into());
    }
    Ok((hrp.to_string(), out))
}

fn bech32_polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [0x3b6a_57b2, 0x2650_8e6d, 0x1ea1_19fa, 0x3d42_33dd, 0x2a14_62b3];
    let mut chk = 1u32;
    for &v in values {
        let top = chk >> 25;
        chk = ((chk & 0x01ff_ffff) << 5) ^ v as u32;
        for (i, g) in GEN.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

/// Coins smaller than 1/2^this of the block are not split any further by
/// [`split_with_floor`]: that is as close to proportional as a block gets.
const SPLIT_GRANULARITY_SHIFT: u32 = 8;
/// The most coinbase outputs [`split_with_floor`] makes by splitting coins.
const MAX_SPLIT_OUTPUTS: usize = 1_000;

/// Splits `distributable` among `shares` (address, score) in power-of-2 coins,
/// paying every miner with a positive score at least one coin.
///
/// # Reasoning
/// The greedy split handed out only `popcount(distributable)` coins (about 17
/// for a block reward), largest score first, so a miner with a few shares next
/// to larger miners often got no coin at all. Its own audit then reports
/// "omitted from payout array despite score" and disconnects, and while
/// disconnected it can never earn the score that would get it paid: it stays
/// locked out until the pool's next block. Paying everyone in the committed
/// tree something also keeps the audit's promise: score means a payout.
///
/// 1. Each miner's entitlement is `distributable * score / total`, rounded
///    down, the units left over going to the largest fractional parts (exact
///    largest-remainder apportionment). A miner whose share rounds to nothing
///    is lent one unit by the miner owed the most.
/// 2. Each miner first gets the largest power-of-2 coin within its
///    entitlement, so every miner is in the coinbase with at least half of what
///    it is owed.
/// 3. The rest goes out largest coin first, each to the miner still owed the
///    most. A coin worth more than that miner is owed is split in two (one more
///    output) until it fits, or until it is under 1/2^[`SPLIT_GRANULARITY_SHIFT`]
///    of the block. Every payout ends within that of its exact share (give or
///    take a few units of rounding). The coinbase holds one output per miner,
///    up to 64 more for the rest, and one per split; splitting stops at
///    [`MAX_SPLIT_OUTPUTS`].
///
/// Score deductions after a found block follow the coins actually paid, so
/// what rounding remains still evens out over later blocks.
///
/// Returns `None` (the caller keeps the greedy split) only when there is
/// nothing to split, or more miners than units to pay, which no real block
/// reward reaches.
fn split_with_floor(distributable: u64, shares: &[([u8; 32], u64)]) -> Option<Vec<([u8; 32], u64)>> {
    let miners: Vec<([u8; 32], u64)> = shares.iter().copied().filter(|(_, s)| *s > 0).collect();
    if miners.is_empty() || distributable == 0 || miners.len() as u64 > distributable {
        return None;
    }
    let d = distributable as u128;
    let total: u128 = miners.iter().map(|(_, s)| *s as u128).sum();

    // 1. Entitlements: the floor of each exact share, remainder by largest fraction.
    let mut owed: Vec<u128> = miners.iter().map(|(_, s)| d * *s as u128 / total).collect();
    let mut spare = d - owed.iter().sum::<u128>();
    let mut by_fraction: Vec<usize> = (0..miners.len()).collect();
    by_fraction.sort_by(|&a, &b| {
        let fa = d * miners[a].1 as u128 % total;
        let fb = d * miners[b].1 as u128 % total;
        fb.cmp(&fa).then(miners[a].0.cmp(&miners[b].0))
    });
    for &i in &by_fraction {
        if spare == 0 {
            break;
        }
        owed[i] += 1;
        spare -= 1;
    }
    // Ties go to the lower address, so the split is the same on every run.
    let most_owed = |owed: &[u128]| {
        (0..miners.len())
            .max_by(|&a, &b| owed[a].cmp(&owed[b]).then(miners[b].0.cmp(&miners[a].0)))
            .expect("miners is not empty")
    };
    // There are no more miners than units, so while anyone is owed nothing,
    // someone else is owed at least two.
    for i in 0..miners.len() {
        if owed[i] == 0 {
            let j = most_owed(&owed);
            owed[j] -= 1;
            owed[i] = 1;
        }
    }

    // 2. A floor coin each: the largest power of two within the entitlement.
    let mut coins = Vec::with_capacity(2 * miners.len() + 64);
    let mut left = d;
    for (i, (addr, _)) in miners.iter().enumerate() {
        let coin = 1u128 << (127 - owed[i].leading_zeros());
        coins.push((*addr, coin as u64));
        left -= coin;
        owed[i] -= coin;
    }

    // 3. The rest, largest coin first, to whoever is still owed the most.
    let small = (d >> SPLIT_GRANULARITY_SHIFT).max(1);
    let mut pending: std::collections::BinaryHeap<u128> = crate::core::types::decompose_value(left as u64)
        .into_iter()
        .map(u128::from)
        .collect();
    while let Some(coin) = pending.pop() {
        let i = most_owed(&owed);
        if coin > owed[i] && coin > small && coins.len() + pending.len() + 2 <= MAX_SPLIT_OUTPUTS {
            pending.push(coin / 2);
            pending.push(coin / 2);
            continue;
        }
        coins.push((miners[i].0, coin as u64));
        owed[i] = owed[i].saturating_sub(coin);
    }
    Some(coins)
}

/// Splits `total` among `owners` (sorted by address) in proportion to score,
/// paying at most `slots_total` of them. An owner with at least
/// 1/`slots_total` of the score is paid its share exactly. The rest share the
/// remaining outputs by systematic sampling: laid end to end, each spanning
/// its score, with the outputs at evenly spaced points from `offset`, so each
/// one's *expected* payment is exactly its share. Returns the payments and
/// what rounding left over. (Midwimble's own pool pays the same way.)
fn allocate<'a>(
    total: u128,
    owners: &[(&'a str, u64)],
    offset: u128,
    slots_total: usize,
) -> (Vec<(&'a str, u128)>, u128) {
    let weight: u128 = owners.iter().map(|(_, w)| *w as u128).sum();
    if weight == 0 {
        return (Vec::new(), total);
    }
    let slots_total = slots_total as u128;
    let (mut paid, mut small, mut exact) = (Vec::new(), Vec::new(), 0u128);
    for (key, w) in owners {
        let w = *w as u128;
        if w * slots_total >= weight {
            let amount = total * w / weight;
            exact += amount;
            paid.push((*key, amount));
        } else if w > 0 {
            small.push((*key, w));
        }
    }
    let rest = total - exact;
    if small.is_empty() {
        return (paid, rest);
    }
    // At least one output is left: owners of 1/slots_total or more can fill
    // every output only by holding all the score, and then `small` is empty.
    let slots = slots_total - paid.len() as u128;
    let small_weight: u128 = small.iter().map(|(_, w)| *w).sum();
    let offset = offset % small_weight;
    let first_slot_at = |x: u128| -> u128 {
        if x <= offset {
            0
        } else {
            (x - offset).div_ceil(small_weight).min(slots)
        }
    };
    let slot_value = rest / slots;
    let mut before = 0u128;
    for (key, w) in small {
        let from = first_slot_at(slots * before);
        before += w;
        let to = first_slot_at(slots * before);
        if to > from {
            paid.push((key, (to - from) * slot_value));
        }
    }
    paid.sort_by(|a, b| a.0.cmp(b.0));
    (paid, rest - slots * slot_value)
}

/// What a job's midwimble block pays, as `(address, weight)` out of
/// [`MW_PAYOUT_SCALE`]: the pool's fee and any rounding to the pool's own
/// address, and the rest by midwimble score (see [`allocate`]), at most
/// [`MW_MAX_OUTPUTS`] in all. With no score yet, everything goes to the pool.
fn merge_payouts(fee_percent: f64, pool: &str, scores: &[(String, u64)], seed: &[u8; 32]) -> Vec<(String, u64)> {
    let fee_ppm = if fee_percent.is_finite() {
        ((fee_percent.clamp(0.0, 100.0) * 10_000.0).round() as u128).min(999_999)
    } else {
        0
    };
    let fee = MW_PAYOUT_SCALE * fee_ppm / 1_000_000;
    let mut owners: Vec<(&str, u64)> = scores
        .iter()
        .filter(|(_, s)| *s > 0)
        .map(|(a, s)| (a.as_str(), *s))
        .collect();
    owners.sort();
    let mut offset = [0u8; 16];
    offset.copy_from_slice(&seed[..16]);
    let (miners, dust) = allocate(MW_PAYOUT_SCALE - fee, &owners, u128::from_be_bytes(offset), MW_MAX_OUTPUTS - 1);
    let mut payouts: Vec<(String, u64)> = miners.into_iter().map(|(a, w)| (a.to_string(), w as u64)).collect();
    let to_pool = (fee + dust) as u64;
    if to_pool > 0 {
        match payouts.iter_mut().find(|(a, _)| a == pool) {
            Some((_, w)) => *w += to_pool,
            None => payouts.push((pool.to_string(), to_pool)),
        }
    }
    payouts
}

/// The midwimble score table's positive entries.
fn midwimble_scores(db: &Database) -> Vec<(String, u64)> {
    let Ok(txn) = db.begin_read() else { return Vec::new() };
    let Ok(table) = txn.open_table(MW_SHARES_TABLE) else { return Vec::new() };
    let Ok(rows) = table.iter() else { return Vec::new() };
    rows.flatten()
        .map(|(a, s)| (a.value().to_string(), s.value()))
        .filter(|(_, s)| *s > 0)
        .collect()
}

/// Takes the scores a midwimble block paid out of the table, or, with
/// `restore`, puts them back when that block wasn't accepted after all.
fn settle_midwimble_scores(db: &Database, paid: &[(String, u64)], restore: bool) -> anyhow::Result<()> {
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(MW_SHARES_TABLE)?;
        for (address, score) in paid {
            let current = table.get(address.as_str())?.map(|v| v.value()).unwrap_or(0);
            let next = if restore { current.saturating_add(*score) } else { current.saturating_sub(*score) };
            if next == 0 {
                table.remove(address.as_str())?;
            } else {
                table.insert(address.as_str(), next)?;
            }
        }
    }
    txn.commit()?;
    Ok(())
}

/// The midwimble node's answer to `POST /merge/job`.
struct MergeJobReply {
    commitment: String,
    mining_hash: String,
    target: [u8; 32],
    height: Option<u64>,
}

/// Asks the midwimble node for a block paying `payouts`.
async fn request_merge_job(
    client: &reqwest::Client,
    rpc: &str,
    payouts: &[(String, u64)],
) -> Result<MergeJobReply, String> {
    let payouts: Vec<serde_json::Value> = payouts
        .iter()
        .map(|(address, weight)| serde_json::json!({ "address": address, "weight": weight }))
        .collect();
    let res = client
        .post(format!("{rpc}/merge/job"))
        .timeout(Duration::from_secs(5))
        .json(&serde_json::json!({ "payouts": payouts }))
        .send()
        .await
        .map_err(|e| format!("midwimble node unreachable: {e}"))?;
    let status = res.status();
    let job: serde_json::Value = res.json().await.map_err(|e| format!("unreadable answer: {e}"))?;
    if !status.is_success() {
        return Err(job["error"].as_str().map(str::to_string).unwrap_or_else(|| format!("HTTP {status}")));
    }
    let field = |k: &str| job[k].as_str().unwrap_or_default().to_string();
    let (commitment, mining_hash) = (field("commitment"), field("mining_hash"));
    let mut target = [0u8; 32];
    if commitment.len() != 64 || mining_hash.len() != 64 || hex::decode_to_slice(field("target"), &mut target).is_err() {
        return Err(format!("unusable job: {job}"));
    }
    Ok(MergeJobReply { commitment, mining_hash, target, height: job["height"].as_u64() })
}

/// Gets this job's midwimble block from the midwimble node, paying the score
/// table, and returns the commitment to plant in a coinbase salt with the
/// job's merge details (`commit_index` is set where it is planted).
///
/// Every failure is soft. If the node refuses the miners' payouts, the job
/// pays the pool's own address, as before per-miner payouts, and says why in
/// its `note`; if it refuses that too, or can't be reached, the reason: the
/// pool mines midstate alone.
async fn plan_merge(
    db: &Database,
    client: &reqwest::Client,
    rpc: &str,
    pool: &str,
    fee_percent: f64,
    seed: &[u8; 32],
) -> Result<(String, MergeCommit), String> {
    let scores = midwimble_scores(db);
    let payouts = merge_payouts(fee_percent, pool, &scores, seed);
    let pool_only = payouts.len() == 1 && payouts[0].0 == pool;
    let (reply, paid, payouts, note) = match request_merge_job(client, rpc, &payouts).await {
        Ok(reply) => (reply, scores, payouts, None),
        Err(e) if pool_only => {
            tracing::warn!("midwimble job refused ({e}); mining midstate alone");
            return Err(e);
        }
        Err(e) => {
            tracing::warn!("midwimble refused this job's payouts ({e}); paying the pool's own address instead");
            let fallback = vec![(pool.to_string(), 1)];
            match request_merge_job(client, rpc, &fallback).await {
                Ok(reply) => {
                    let note = format!("the node refused the miners' payouts ({e}), so this block pays the pool's own address");
                    (reply, Vec::new(), fallback, Some(note))
                }
                Err(e) => {
                    tracing::warn!("midwimble job refused ({e}); mining midstate alone");
                    return Err(e);
                }
            }
        }
    };
    Ok((
        reply.commitment,
        MergeCommit {
            mining_hash: reply.mining_hash,
            target: reply.target,
            commit_index: 0,
            paid: Arc::new(paid),
            claimed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            height: reply.height,
            payouts: Arc::new(payouts),
            note,
        },
    ))
}

/// Records a midwimble block this pool found, at `height`, for the dashboard.
fn record_midwimble_block(db: &Database, height: Option<u64>, merge: &MergeCommit) -> anyhow::Result<()> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let payouts: Vec<serde_json::Value> = merge
        .payouts
        .iter()
        .map(|(address, weight)| serde_json::json!({ "address": address, "weight": weight }))
        .collect();
    let row = serde_json::json!({ "height": height, "timestamp": now.as_secs(), "payouts": payouts }).to_string();
    let txn = db.begin_write()?;
    {
        let mut table = txn.open_table(MW_BLOCKS_TABLE)?;
        let mut key = now.as_millis() as u64;
        while table.get(key)?.is_some() {
            key += 1;
        }
        table.insert(key, row.as_str())?;
    }
    txn.commit()?;
    Ok(())
}

/// The dashboard's midwimble section: `null` unless this pool merge-mines.
async fn midwimble_stats(state: &PoolState) -> serde_json::Value {
    let (Some(_), Some(pool_mw)) = (&state.midwimble_rpc, &state.midwimble_address) else {
        return serde_json::Value::Null;
    };
    let status = state.merge_status.read().await.clone();
    let mut scores = midwimble_scores(&state.db);
    scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let total_score: u64 = scores.iter().map(|(_, s)| *s).sum();
    let (mut blocks_found, mut recent) = (0usize, Vec::new());
    if let Ok(txn) = state.db.begin_read() {
        if let Ok(table) = txn.open_table(MW_BLOCKS_TABLE) {
            if let Ok(rows) = table.iter() {
                for row in rows.rev().flatten() {
                    blocks_found += 1;
                    if recent.len() < MW_RECENT_BLOCKS {
                        if let Ok(block) = serde_json::from_str::<serde_json::Value>(row.1.value()) {
                            recent.push(block);
                        }
                    }
                }
            }
        }
    }
    serde_json::json!({
        "pool_address": pool_mw,
        "active": status.active,
        "note": status.note,
        "height": status.height,
        "updated": status.updated,
        "scores": scores
            .iter()
            .map(|(address, score)| serde_json::json!({ "address": address, "score": score }))
            .collect::<Vec<_>>(),
        "total_score": total_score,
        "blocks_found": blocks_found,
        "recent_blocks": recent,
    })
}

// ── App State ───────────────────────────────────────────────────────────────

/// Global state shared across the HTTP Audit API, the Core Polling Task, 
/// and the TCP Stratum Socket Handlers.
struct PoolState {
    db: Arc<Database>,
    current_job: RwLock<Option<Job>>,
    job_notifier: broadcast::Sender<Job>,
    /// The pool's raw 32-byte MSS public key hash.
    pool_address: String,
    share_target: [u8; 32],
    current_tree: RwLock<ShareMerkleTree>,
    /// The Share Replay Cache. Tracks successfully submitted nonces for the *current* job.
    /// Wiped clean every time a new block is detected to prevent OOM.
    valid_shares: RwLock<HashSet<u64>>, 
    /// Dynamic RPC URL of the core node, provided at startup.
    node_rpc_url: String,
    /// A midwimble node with a mining bond, to merge-mine against.
    midwimble_rpc: Option<String>,
    /// Where midwimble pays this pool's blocks.
    midwimble_address: Option<String>,
    /// Whether the current job merge-mines midwimble, and why not if it
    /// doesn't, for the dashboard.
    merge_status: RwLock<MergeStatus>,
    /// The percentage fee the pool takes from block rewards (e.g., 1.0 for 1%).
    pool_fee_percent: f64,
    /// The most recent network block reward seen from the node, relayed to the
    /// dashboard so it can estimate expected payouts / coins-per-day before the
    /// pool has found its first block. Lock-free; updated each polling cycle.
    current_block_reward: std::sync::atomic::AtomicU64,
    /// The most recent confirmed chain height seen from the node, relayed so the
    /// dashboard can show the live height and label blocks. Updated each poll.
    current_height: std::sync::atomic::AtomicU64,
    /// Cumulative per-miner share outcomes: address -> (accepted, rejected). Lifetime
    /// totals (never deducted on block-find, unlike score) so the dashboard can show a
    /// stable accept/reject efficiency per miner.
    share_stats: RwLock<HashMap<[u8; 32], (u64, u64)>>,
    /// Cumulative accepted shares per (address, worker-name). A pure stats layer for
    /// per-rig breakdown; payout accounting stays strictly per-address and is untouched.
    worker_stats: RwLock<HashMap<([u8; 32], String), u64>>,
    /// Set by the block-submission task when a submitted block is REJECTED by the
    /// node or the submission itself FAILS (connection refused / timeout / dropped).
    /// The network tip does not change in either case, so without this flag the
    /// template loop would never rebuild and miners would re-grind the same doomed
    /// template forever. Consumed (cleared) by the polling loop when it rebuilds.
    force_new_job: std::sync::atomic::AtomicBool,
    /// Bounds how many share verifications may run concurrently.
    ///
    /// One verification is a full `EXTENSION_ITERATIONS` (1,000,000) sequential
    /// BLAKE3 chain — roughly 100–200ms of a pinned core. `/api/submit` is a
    /// public, unauthenticated endpoint, so without a ceiling a trickle of
    /// requests can saturate every core on the box (a ~150ms-of-CPU-per-packet
    /// amplification factor). Permits are sized to the core count and held
    /// across the `spawn_blocking` call by BOTH protocols, so Stratum and HTTP
    /// draw from the same budget and neither can starve the other.
    verify_permits: Arc<Semaphore>,
    /// Per-address submit throttle for the public HTTP path: the instant of that
    /// address's last accepted verification. A browser miner produces a share
    /// every few minutes, so `HTTP_MIN_SUBMIT_INTERVAL` costs honest clients
    /// nothing while capping how fast one address can spend pool CPU. Stratum is
    /// exempt: it is rate-limited implicitly by holding an open TCP session.
    http_throttle: RwLock<HashMap<[u8; 32], std::time::Instant>>,
}

/// Minimum spacing between two verified share submissions from one address.
///
/// Lowered from 250ms: a browser rig runs one miner per core (10 here), and
/// although each finds a share only every ~95s, independent workers do collide.
/// At 250ms a real share was being discarded as "Submitting too fast" — throwing
/// away work a miner actually did. The CPU that this guard exists to protect is
/// already bounded by `verify_permits`, so this only needs to stop a pathological
/// flood, not police normal bursts.
const HTTP_MIN_SUBMIT_INTERVAL: Duration = Duration::from_millis(50);

/// Default share difficulty, in leading zero bits: one share per 2^12 nonces.
/// Matches the previous hardcoded `[0x00, 0x0f, 0xff, …]` target exactly, so
/// existing deployments see no change unless they pass `--share-bits`.
const DEFAULT_SHARE_BITS: u32 = 12;

/// Builds a 256-bit target with `bits` leading zero bits.
///
/// The share target is the pool's only difficulty knob, and it was previously a
/// pair of magic byte writes that silently encoded "12 leading zero bits".
/// Expressing it as bits makes the knob legible and lets an operator match it to
/// the hardware actually pointed at the pool without a recompile.
///
/// ```text
/// bits = 12  →  00 0f ff ff … ff   (1 share per 2^12 nonces)
/// bits = 16  →  00 00 ff ff … ff   (1 share per 2^16 nonces)
/// ```
fn target_from_leading_zero_bits(bits: u32) -> [u8; 32] {
    let bits = bits.min(255);
    let mut t = [0xffu8; 32];
    let full = (bits / 8) as usize;
    let rem = bits % 8;
    for b in t.iter_mut().take(full) {
        *b = 0x00;
    }
    if full < 32 && rem > 0 {
        t[full] = 0xffu8 >> rem;
    }
    t
}

#[derive(Deserialize)]
struct ProofQuery {
    address: String,
}

// ── HTTP API for Miner Audits ───────────────────────────────────────────────

/// Serves the SPV-style Merkle inclusion proof to miners via HTTP.
///
/// # Reasoning
/// Stratum clients independently poll this endpoint upon receiving a `mining.notify`
/// event. They use the returned index, score, and sibling hashes to reconstruct the 
/// Merkle root locally. If it doesn't match the `salt` of the fee coin in the template, 
/// the miner knows the pool is lying and disconnects.
///
/// # Security
/// Uses `parse_address_flexible` to allow miners to query using either raw 64-char hex
/// or the 72-char checksummed UI address.
async fn get_proof(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<ProofQuery>,
) -> Json<serde_json::Value> {
    let addr = match crate::core::types::parse_address_flexible(&query.address) {
        Ok(a) => a,
        Err(_) => return Json(serde_json::json!({ "error": "Invalid address" })),
    };
    
    let tree = state.current_tree.read().await;
    let score = tree.leaves.iter().find(|(a, _)| a == &addr).map(|(_, s)| *s).unwrap_or(0);
    
    if let Some((idx, proof)) = tree.generate_proof(&addr) {
        Json(serde_json::json!({
            "root": hex::encode(tree.root),
            "score": score,
            "index": idx, 
            "proof": proof.iter().map(hex::encode).collect::<Vec<_>>()
        }))
    } else {
        Json(serde_json::json!({ "error": "Miner not found in current block precommitment" }))
    }
}

#[derive(Deserialize)]
struct BlockScoresQuery {
    /// Block timestamp (the key the dashboard already has from recent_blocks[].timestamp).
    ts: u64,
}

/// Serves the committed per-miner score snapshot for one found block, so the dashboard
/// can prove every payout in that block was proportional to committed score (the
/// "prove the split" check). Held out of /pool/stats to keep that payload lean, and
/// fetched on demand only when a miner clicks to verify a specific block.
///
/// Returns `{ total_score, scores: [{address, score}] }` where `total_score` is the
/// committed total the coinbase payouts were actually proportioned against.
async fn get_block_scores(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<BlockScoresQuery>,
) -> Json<serde_json::Value> {
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(BLOCK_SCORES_TABLE) {
            if let Ok(Some(v)) = table.get(query.ts) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(v.value()) {
                    return Json(json);
                }
            }
        }
    }
    Json(serde_json::json!({ "error": "No committed score snapshot stored for that block" }))
}

/// Serves the Provably Fair Pool HTML dashboard.
///
/// # Reasoning
/// Providing a built-in dashboard allows pool operators to transparently display
/// current hash weights and historical payouts without needing external infrastructure.
/// By serving this alongside the stratum port offset, it does not conflict with the core node.
///
/// # Formal Specification
/// ```text
/// Pre:  true
/// Post: result is an HTML response containing the dashboard UI
/// ```
async fn pool_ui() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("pool.html"))
}

/// Serves the shared Midstate stylesheet.
///
/// # Reasoning
/// The dashboard links to `/midstate.css` rather than embedding its own palette so it
/// stays visually identical to the Explorer and Chat and inherits the global light/dark
/// theme automatically. The sheet is baked into the binary with `include_str!`, so there
/// is no runtime "file not found" failure mode — if it compiles, it serves.
///
/// Path note: `pool.rs` lives in `src/` while the shared web assets live in `src/rpc/`,
/// so this resolves to the single canonical `src/rpc/midstate.css` (the temporary copy
/// in `src/` can be deleted). `include_str!` is relative to THIS source file.
async fn pool_css() -> impl axum::response::IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("rpc/midstate.css"),
    )
}

/// Aggregates and returns current pool statistics as JSON.
///
/// # Reasoning
/// Iterates over the `shares` table to calculate current miner weights, and reads
/// the last 100 entries from the `blocks` table to show recent payouts and to give
/// the dashboard enough history to draw the rolling effort/earnings charts. This
/// provides total transparency to the miners so they can verify their payout equity
/// matches their hash contribution.
///
/// Note: the raw `network_target` and `share_target` are returned verbatim so the
/// browser can do all hashrate/effort math locally with native `BigInt`. The node's
/// own `/state` endpoint is intentionally NOT exposed to the public (it sits behind a
/// firewall/VPN), so the pool relays the only two values the UI needs to stay honest.
///
/// # Formal Specification
/// ```text
/// Pre:  true
/// Post: result contains (pool_fee_percent, total_score, active_miners, miners[], recent_blocks[])
/// ```
async fn get_pool_stats(State(state): State<Arc<PoolState>>) -> Json<serde_json::Value> {
    let mut miners = Vec::new();
    let mut total_score = 0u64;

    // Lifetime accept/reject tallies, snapshotted so we don't hold the lock across the
    // redb scan. Attached per-miner below for an efficiency readout.
    let share_snapshot = state.share_stats.read().await.clone();

    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(SHARES_TABLE) {
            for iter in table.iter().unwrap() {
                let (addr, score) = iter.unwrap();
                let mut a = [0u8; 32];
                a.copy_from_slice(addr.value());
                let s = score.value();
                if s > 0 {
                    let (accepted, rejected) = share_snapshot.get(&a).copied().unwrap_or((0, 0));
                    miners.push(serde_json::json!({
                        "address": crate::core::types::encode_address_with_checksum(&a),
                        "score": s,
                        "accepted": accepted,
                        "rejected": rejected
                    }));
                    total_score += s;
                }
            }
        }
    }

    miners.sort_by_key(|m| std::cmp::Reverse(m["score"].as_u64().unwrap_or(0)));

    // Per-(address, worker) accepted-share tallies for the rig breakdown.
    let mut workers = Vec::new();
    {
        let ws = state.worker_stats.read().await;
        for ((addr, name), count) in ws.iter() {
            workers.push(serde_json::json!({
                "address": crate::core::types::encode_address_with_checksum(addr),
                "worker": name,
                "score": count
            }));
        }
    }

    let mut blocks = Vec::new();
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(BLOCKS_TABLE) {
            for iter in table.iter().unwrap().rev().take(100) {
                let (_, data) = iter.unwrap();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(data.value()) {
                    blocks.push(json);
                }
            }
        }
    }

    // --- RAW TARGETS + PRECOMMITMENT FOR CLIENT-SIDE BIGINT MATH ---
    // The browser does ALL hashrate/effort math locally with native BigInt against
    // these raw 32-byte targets, so the server never touches floating point for it.
    let current_job = state.current_job.read().await.clone();
    let share_target_hex = hex::encode(state.share_target);
    let network_target_hex = current_job
        .as_ref()
        .map(|job| hex::encode(job.network_target))
        .unwrap_or_default();
    // The current Merkle precommitment root (what miners audit against this block).
    let merkle_root_hex = hex::encode(state.current_tree.read().await.root);
    const BLOCK_TIME_SECS: u64 = 60;

    // --- OPTIONAL FLOAT FALLBACK (legacy clients / non-BigInt environments) ---
    let mut network_hashrate = 0.0;
    let mut hashes_per_share = 0.0;
    if let Some(job) = current_job {
        // Helper to convert U256 target to a float for math
        fn u256_to_f64(u: primitive_types::U256) -> f64 {
            u.0[0] as f64 +
            (u.0[1] as f64) * 2.0f64.powi(64) +
            (u.0[2] as f64) * 2.0f64.powi(128) +
            (u.0[3] as f64) * 2.0f64.powi(192)
        }

        let net_target = primitive_types::U256::from_big_endian(&job.network_target);
        let share_target = primitive_types::U256::from_big_endian(&state.share_target);
        
        let max_u256 = 2.0f64.powi(256);
        let net_diff_hashes = max_u256 / u256_to_f64(net_target).max(1.0);
        network_hashrate = net_diff_hashes / BLOCK_TIME_SECS as f64;
        
        hashes_per_share = max_u256 / u256_to_f64(share_target).max(1.0);
    }

    Json(serde_json::json!({
        "pool_fee_percent": state.pool_fee_percent,
        "total_score": total_score,
        "active_miners": miners.len(),
        "miners": miners,
        "recent_blocks": blocks,
        // Merge-mining: status, the midwimble score table and found blocks;
        // null when this pool doesn't merge-mine.
        "midwimble": midwimble_stats(&state).await,
        // Provably-fair anchor + raw targets for local BigInt math:
        "merkle_root": merkle_root_hex,
        "network_target": network_target_hex,
        "share_target": share_target_hex,
        "block_time_secs": BLOCK_TIME_SECS,
        // Pool fee address (so the UI can label + verify the fee coin in every block)
        // and the live network reward (for payout / coins-per-day estimates):
        "pool_address": state.pool_address,
        "block_reward": state.current_block_reward.load(std::sync::atomic::Ordering::Relaxed),
        // Live confirmed chain height (blocks the pool finds are stamped with their own
        // height), and the per-rig worker breakdown (stats only; payouts are per-address):
        "network_height": state.current_height.load(std::sync::atomic::Ordering::Relaxed),
        "workers": workers,
        // Float fallbacks (kept for backwards compatibility):
        "network_hashrate": network_hashrate,
        "hashes_per_share": hashes_per_share
    }))
}

// ── Main Server Boot ────────────────────────────────────────────────────────

/// Boots the Provably Fair Stratum Server and its companion HTTP Audit API.
///
/// # Architecture
/// This server is designed to run independently from the core Midstate node. 
/// In professional mining setups, the core node is heavily firewalled or hidden 
/// behind a VPN (e.g., Tailscale), while the Stratum server is exposed to the 
/// public internet.
///
/// # Arguments
/// * `pool_address` - The Midstate address where the pool fee will be sent.
/// * `bind_addr` - The `IP:PORT` to bind the Stratum TCP server to (e.g., `0.0.0.0:3333`).
///                 The HTTP Audit API will automatically bind to an offset port (e.g., `8081`).
/// * `node_rpc_url` - The HTTP URL of the backend Midstate node (e.g., `http://10.0.0.5:8545`).
/// * `pool_fee_percent` - The percentage of the block reward taken by the pool (e.g., 1.0).
/// * `share_bits` - Share difficulty in leading zero bits. `None` uses
///   [`DEFAULT_SHARE_BITS`]. Every share is worth exactly one point of score
///   regardless of this value, so changing it mid-round rescales what a point
///   costs: raise it and existing score becomes relatively more valuable, lower
///   it and new shares are cheaper than the ones already banked. Change it
///   between blocks, not during one, or set it once and leave it.
/// * `webrtc_port` - UDP port for the browser-facing webrtc-direct listener.
///   `None` disables it, leaving Stratum + HTTP only.
pub async fn run_stratum_pool(
    pool_address: String,
    bind_addr: String,
    node_rpc_url: String,
    pool_fee_percent: f64,
    share_bits: Option<u32>,
    webrtc_port: Option<u16>,
    midwimble_rpc: Option<String>,
    midwimble_address: Option<String>,
) {
    tracing::info!("starting stratum pool server");
    
    std::fs::create_dir_all("data").unwrap();
    let db = Arc::new(Database::create("data/pool_stratum.redb").unwrap());
    
    let write_txn = db.begin_write().unwrap();
    {
        let mut shares = write_txn.open_table(SHARES_TABLE).unwrap();
        let _ = write_txn.open_table(BLOCKS_TABLE).unwrap();
        let _ = write_txn.open_table(PENDING_BLOCKS_TABLE).unwrap();
        let _ = write_txn.open_table(MW_SHARES_TABLE).unwrap();
        let _ = write_txn.open_table(MW_BLOCKS_TABLE).unwrap();
        // ── One-off migration: purge stale zero-score rows ──
        // Databases written by affected versions accumulated permanent `score == 0`
        // rows (the deduction path used to write them back instead of removing them).
        // The template builder now filters them on load, but purging heals existing
        // DBs in place so the shares table matches what /pool/stats already shows
        // (`s > 0`), and keeps redb from carrying dead keys forever. Collect first,
        // then remove, to avoid mutating the table while its iterator is live.
        let stale: Vec<[u8; 32]> = shares
            .iter()
            .unwrap()
            .filter_map(|iter| {
                let (addr, score) = iter.unwrap();
                if score.value() == 0 {
                    let mut a = [0u8; 32];
                    a.copy_from_slice(addr.value());
                    Some(a)
                } else {
                    None
                }
            })
            .collect();
        let purged = stale.len();
        for a in stale {
            shares.remove(&a).unwrap();
        }
        if purged > 0 {
            tracing::info!("purged {} stale zero-score address(es) from the shares table", purged);
        }
    }
    write_txn.commit().unwrap();

    let (job_notifier, _) = broadcast::channel(32);
    
    let share_bits = share_bits.unwrap_or(DEFAULT_SHARE_BITS).clamp(1, 240);
    let share_target = target_from_leading_zero_bits(share_bits);
    tracing::info!(
        "share difficulty: {} leading zero bits (~1 share per {} nonces); target {}",
        share_bits,
        1u128.checked_shl(share_bits).map(|v| v.to_string()).unwrap_or_else(|| "2^…".into()),
        hex::encode(&share_target[..8])
    );

    // Strip the UI checksum from the pool address so the backend node accepts it
    // during block template generation.
    let clean_pool_address_bytes = crate::core::types::parse_address_flexible(&pool_address)
        .expect("CRITICAL: Invalid Pool Address provided");
    let clean_pool_address = hex::encode(clean_pool_address_bytes);

    // The pool's own midwimble address takes the fee and the shares of miners
    // who gave no midwimble address. Checked now, since the midwimble node
    // refuses a job with any payout it can't decode.
    let midwimble_address = midwimble_address.map(|a| {
        parse_midwimble_address(&a)
            .unwrap_or_else(|e| panic!("CRITICAL: invalid --midwimble-address ({e})"))
    });
    match (&midwimble_rpc, &midwimble_address) {
        (Some(rpc), Some(_)) => tracing::info!(
            "merge-mining midwimble against {rpc}; miners are paid in midwimble when they give a midwimble address"
        ),
        (Some(_), None) | (None, Some(_)) => tracing::warn!(
            "merge-mining needs both --midwimble-rpc and --midwimble-address; mining midstate alone"
        ),
        (None, None) => {}
    }

    let state = Arc::new(PoolState {
        db,
        current_job: RwLock::new(None),
        job_notifier,
        pool_address: clean_pool_address,
        share_target,
        current_tree: RwLock::new(ShareMerkleTree::build(vec![])),
        valid_shares: RwLock::new(HashSet::new()),
        node_rpc_url,
        midwimble_rpc: midwimble_rpc.map(|u| {
            let u = u.trim_end_matches('/').to_string();
            if u.starts_with("http") { u } else { format!("http://{u}") }
        }),
        midwimble_address,
        pool_fee_percent,
        current_block_reward: std::sync::atomic::AtomicU64::new(0),
        current_height: std::sync::atomic::AtomicU64::new(0),
        share_stats: RwLock::new(HashMap::new()),
        worker_stats: RwLock::new(HashMap::new()),
        force_new_job: std::sync::atomic::AtomicBool::new(false),
        // Leave one core for the async reactor, the redb writer and the template
        // poller, so a burst of share verifications can never freeze job delivery.
        verify_permits: Arc::new(Semaphore::new(
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(1).max(1)).unwrap_or(1),
        )),
        http_throttle: RwLock::new(HashMap::new()),
        merge_status: RwLock::new(MergeStatus::default()),
    });

    let api_state = state.clone();
    let parts: Vec<&str> = bind_addr.split(':').collect();
    let host = if parts.is_empty() { "0.0.0.0" } else { parts[0] };
    let base_stratum_port: u16 = if parts.len() > 1 { parts[1].parse().unwrap_or(3333) } else { 3333 };

    // ── Tandem Port Binding ──
    // Bind both the Stratum Port and the Audit API port simultaneously.
    // If either fails (e.g., stuck in TIME_WAIT), bump the offset and try the next pair. 
    // This guarantees the miner's offset math always aligns perfectly with the server.
    let mut offset = 0;
    let (api_listener, stratum_listener) = loop {
        let current_stratum = base_stratum_port + offset;
        let current_api = 8081 + offset;

        let a_res = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", current_api)).await;
        let s_res = tokio::net::TcpListener::bind(format!("{}:{}", host, current_stratum)).await;

        match (a_res, s_res) {
            (Ok(a), Ok(s)) => {
                tracing::info!("audit api bound to 0.0.0.0:{}", current_api);
                tracing::info!("stratum pool bound to {}:{}", host, current_stratum);
                break (a, s);
            }
            _ => {
                tracing::warn!("port pair {}/{} in use. trying next pair...", current_stratum, current_api);
                offset += 1;
                if offset > 10 {
                    panic!("fatal: could not find available stratum/api port pairs");
                }
            }
        }
    };

    tokio::spawn(async move {
        // ── CORS ──
        // The web wallet is a static page served from a different origin than
        // this pool, so every /api/* call it makes is cross-origin. Without
        // these headers the browser blocks the response before the wallet's
        // JS ever sees it — the fetch fails with an opaque "TypeError: Failed
        // to fetch" that looks like the pool being down. `Any` origin is the
        // right call here specifically because these endpoints are public,
        // read-only-or-idempotent, and carry no cookies or ambient authority:
        // a share submission is authenticated by proof-of-work, not by the
        // caller's identity, so there is no cross-site request forgery surface
        // to protect. (Contrast the node's own RPC in `rpc/server.rs`, which
        // pins an explicit origin allow-list because it exposes wallet routes.)
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any);

        let app = Router::new()
            .route("/pool", get(pool_ui))            
            .route("/midstate.css", get(pool_css))   
            .route("/pool/stats", get(get_pool_stats)) 
            .route("/api/proof", get(get_proof))     
            .route("/api/block_scores", get(get_block_scores))
            // HTTP mining protocol (browser miners / web wallet):
            .route("/api/template", get(get_http_template))
            .route("/api/submit", post(post_http_submit))
            .layer(cors)
            .with_state(api_state);
        axum::serve(api_listener, app).await.unwrap();
    });
    
    // ── WebRTC listener (browser miners) ──
    // Started alongside the HTTP audit API rather than replacing it: HTTP stays
    // curl-able for monitoring and native tooling, while browsers get a path
    // that needs no CA certificate and receives pushed jobs instead of polling.
    // Failure here is non-fatal — Stratum and HTTP keep working.
    if let Some(udp_port) = webrtc_port {
        let webrtc_state = state.clone();
        tokio::spawn(async move {
            run_pool_webrtc(webrtc_state, udp_port, std::path::PathBuf::from("data")).await;
        });
    } else {
        tracing::info!("pool WebRTC listener disabled (pass --webrtc-port to enable)");
    }

    // ── Core Polling & Template Builder Task ──
    let state_clone = state.clone();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut last_network_tip = String::new();
        // Midwimble's tip when the current job was built. A commitment names
        // one midwimble block, so once that tip moves no share can win
        // midwimble until the job is rebuilt.
        let mut last_mw_tip = String::new();
        // Seed job IDs with the Unix timestamp so IDs issued before a pool restart
        // can never collide with IDs issued after it. (A zero-seeded counter resets
        // on restart; a still-connected miner's stale share can then carry a job_id
        // that matches the NEW counter and gets validated against the new midstate,
        // producing an endless "Low difficulty" reject stream for that miner.)
        // The loop below issues at most one job per second, so the counter can never
        // catch up to and overlap a future restart's seed.
        let mut job_counter: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // One-shot latch for the "share difficulty >= network difficulty" warning.
        let mut warned_share_too_hard = false;
        // Latches so a persistently-down or syncing node logs once, not every 2s.
        let mut node_unreachable_logged = false;
        let mut syncing_logged = false;

        loop {
            let rpc_url = state_clone.node_rpc_url.clone();
            
            let net_state: serde_json::Value = match client.get(&format!("{}/state", rpc_url)).send().await {
                Ok(res) => {
                    if node_unreachable_logged {
                        tracing::info!("backend node reachable again at {}", rpc_url);
                        node_unreachable_logged = false;
                    }
                    match res.json().await {
                        Ok(v) => v,
                        Err(e) => {
                            // The node answered but not with JSON (wrong port, a
                            // proxy, an HTML error page). Previously this became
                            // Value::Null via unwrap_or_default() and the loop
                            // carried on with an empty tip, producing no job and
                            // no explanation.
                            tracing::error!("backend node at {}/state returned unparseable JSON: {}", rpc_url, e);
                            tokio::time::sleep(Duration::from_secs(2)).await;
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // Latched so a down node logs once rather than every 2s.
                    if !node_unreachable_logged {
                        node_unreachable_logged = true;
                        tracing::error!(
                            "cannot reach backend node at {} ({}). No jobs will be served until it responds. \
                             Check the node is running and that --rpc-port/--rpc-host match it.",
                            rpc_url, e
                        );
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };

            // ── Sync Guard ──
            // While the backend node is bulk-downloading historical blocks, every tip
            // it reports is an already-superseded height: any template built from it
            // burns 100% of the pool's hashpower on obsolete blocks. Drop the current
            // job so miner submissions are ignored, and clear the tip tracker so a
            // fresh template is built on the first poll after the sync completes.
            // (`unwrap_or(false)` keeps this backward compatible with older nodes
            // whose /state payload doesn't include the field.)
            if net_state["is_syncing"].as_bool().unwrap_or(false) {
                let had_job = state_clone.current_job.write().await.take().is_some();
                // Latch on the flag, not on had_job: at a cold start there is no
                // job to drop, so keying the message off `had_job` meant a pool
                // pointed at a syncing node logged absolutely nothing and simply
                // served "no active job" forever with no clue why.
                if had_job || !syncing_logged {
                    syncing_logged = true;
                    tracing::warn!(
                        "backend node is syncing historical blocks (height {}); pausing job generation until it catches up",
                        net_state["height"].as_u64().unwrap_or(0)
                    );
                }
                last_network_tip.clear();
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }

            let current_tip = net_state["header_hash"].as_str().unwrap_or("").to_string();
            let mut n_target = [0u8; 32];
            if let Some(t_hex) = net_state["target"].as_str() {
                let _ = hex::decode_to_slice(t_hex, &mut n_target);
            }

            // A share target at or below the network target is a misconfiguration:
            // every share would also be a block, so the pool degrades to solo
            // mining with extra steps and miners accumulate no score between finds.
            // Warned once rather than every poll so it doesn't drown the log.
            if !warned_share_too_hard && n_target != [0u8; 32] && state_clone.share_target <= n_target {
                warned_share_too_hard = true;
                tracing::warn!(
                    "share difficulty is at or above network difficulty (share target {} <= network target {}). \
                     Miners will earn score only when they find actual blocks — lower --share-bits.",
                    hex::encode(&state_clone.share_target[..8]),
                    hex::encode(&n_target[..8])
                );
            }

            // The node's reported height is the confirmed tip; the block a found job
            // produces is the next one (tip + 1). Refreshed every poll so the dashboard
            // can show the live chain height even between our own block finds.
            let tip_height = net_state["height"].as_u64().unwrap_or(0);
            state_clone.current_height.store(tip_height, std::sync::atomic::Ordering::Relaxed);

            // A rejected or failed block submission sets `force_new_job`: the network
            // tip won't change in that case, so a tip-only condition would re-serve
            // the doomed template forever. Consume the flag only when we have a
            // usable tip, so a request that races an empty /state response isn't
            // silently lost. (If template building fails below, `last_network_tip`
            // is cleared, which guarantees a retry on the next poll regardless.)
            let force_new_job = !current_tip.is_empty()
                && state_clone.force_new_job.swap(false, std::sync::atomic::Ordering::SeqCst);

            // When merge-mining, a new midwimble tip needs a new job too. The
            // short timeout keeps a hung midwimble node from slowing midstate.
            let mw_tip = match (state_clone.midwimble_rpc.as_deref(), state_clone.midwimble_address.as_deref()) {
                (Some(rpc), Some(_)) => match client
                    .get(format!("{rpc}/state"))
                    .timeout(Duration::from_secs(1))
                    .send()
                    .await
                {
                    Ok(r) => r
                        .json::<serde_json::Value>()
                        .await
                        .ok()
                        .and_then(|s| s["tip"].as_str().map(str::to_string))
                        .unwrap_or_default(),
                    Err(_) => String::new(),
                },
                _ => String::new(),
            };
            let mw_moved = !mw_tip.is_empty() && mw_tip != last_mw_tip;

            if (current_tip != last_network_tip || force_new_job || mw_moved) && !current_tip.is_empty() {
                if !mw_tip.is_empty() {
                    last_mw_tip = mw_tip.clone();
                }

                 // Only reconcile if the network tip actually advanced
                if current_tip != last_network_tip {
                    if let Err(e) = reconcile_pending_blocks(&state_clone, &client, tip_height).await {
                        tracing::warn!("Failed to reconcile pending blocks: {}", e);
                    }
                }
                
                job_counter += 1;
                last_network_tip = current_tip.clone();

                // Clear the replay cache for the new block
                state_clone.valid_shares.write().await.clear();

                // Drop throttle entries that are already expired. The map is keyed
                // by address, so an attacker rotating through fresh random addresses
                // would otherwise grow it without bound; pruning on every job keeps
                // it proportional to the miners actually submitting right now.
                {
                    let now = std::time::Instant::now();
                    state_clone
                        .http_throttle
                        .write()
                        .await
                        .retain(|_, last| now.duration_since(*last) < HTTP_MIN_SUBMIT_INTERVAL);
                }

                // Only positive-score rows are eligible for reward. A stale `score == 0`
                // row (e.g. the pool fee address, or a miner whose score was fully
                // consumed and never removed) contributes no work and MUST NOT enter the
                // allocator: the greedy loop below drives active miners' simulated scores
                // negative over a long coin decomposition, at which point a leaf sitting at
                // exactly 0 becomes the highest remaining score and captures the tail of the
                // distribution. Filtering here also keeps zero-score leaves out of the
                // Merkle tree and the committed-score snapshot, so the precommitment only
                // ever commits to addresses that actually worked the round.
                let mut shares_vec = Vec::new();
                let mut total_score = 0u128;
                if let Ok(read_txn) = state_clone.db.begin_read() {
                    if let Ok(table) = read_txn.open_table(SHARES_TABLE) {
                        for iter in table.iter().unwrap() {
                            let (addr, score) = iter.unwrap();
                            let s = score.value();
                            if s == 0 { continue; }
                            let mut a = [0u8; 32];
                            a.copy_from_slice(addr.value());
                            shares_vec.push((a, s));
                            total_score += s as u128;
                        }
                    }
                }

                let tree = ShareMerkleTree::build(shares_vec.clone());
                *state_clone.current_tree.write().await = tree.clone();

                let mut expected_total = net_state["block_reward"].as_u64().unwrap_or(0);
                state_clone.current_block_reward.store(expected_total, std::sync::atomic::Ordering::Relaxed);
                
                // ── Proportional Reward Distribution Algorithm ──
                // Calculates the pool fee, then distributes the remaining reward
                // across all miners proportional to their accumulated scores.
                // Output values MUST be powers of 2. `split_with_floor` gives every
                // miner with score at least one coin and splits coins until each
                // payout is within 1/256 of the block of its exact share.
                // Merged mining: one midwimble job per template, paying the
                // midwimble score table, its commitment planted in the salt of
                // the last coinbase output (the score root keeps the first fee
                // coin's). Every failure here is soft — the pool simply mines
                // midstate alone. The draw that picks which small miners a
                // midwimble block pays is seeded by both tips and the score
                // root, none of which the pool can re-roll.
                let merging = state_clone.midwimble_rpc.is_some() && state_clone.midwimble_address.is_some();
                let mut merge_error: Option<String> = None;
                let merge_plan = match (state_clone.midwimble_rpc.as_deref(), state_clone.midwimble_address.as_deref()) {
                    (Some(mw_rpc), Some(mw_pool)) => {
                        let mut seed_data = b"midstate.pool.midwimble-draw.v1".to_vec();
                        seed_data.extend_from_slice(current_tip.as_bytes());
                        seed_data.extend_from_slice(last_mw_tip.as_bytes());
                        seed_data.extend_from_slice(&tree.root);
                        plan_merge(
                            &state_clone.db,
                            &client,
                            mw_rpc,
                            mw_pool,
                            state_clone.pool_fee_percent,
                            &hash(&seed_data),
                        )
                        .await
                        .map_err(|e| merge_error = Some(e))
                        .ok()
                    }
                    _ => None,
                };
                let mut merge_commit: Option<MergeCommit> = None;
                let template_data = loop {
                    let mut coinbase_json = Vec::new();
                    
                    let pool_fee = (expected_total as f64 * (state_clone.pool_fee_percent / 100.0)) as u64;
                    let safe_pool_fee = pool_fee.max(1); 
                    let actual_distributable = expected_total.saturating_sub(safe_pool_fee);
                    
                    let fee_coins = crate::core::types::decompose_value(safe_pool_fee);
                    for (i, coin) in fee_coins.into_iter().enumerate() {
                        // Embed the Merkle Precommitment in the FIRST fee coin's salt.
                        let salt = if i == 0 { 
                            hex::encode(tree.root) 
                        } else { 
                            hex::encode(rand::random::<[u8; 32]>()) 
                        };
                        
                        coinbase_json.push(serde_json::json!({
                            "address": state_clone.pool_address,
                            "value": coin,
                            "salt": salt 
                        }));
                    }

                    if actual_distributable > 0 {
                        // Every miner in the committed tree gets at least one
                        // coin: a miner's audit disconnects it when it has score
                        // but no payout, which locked small miners out.
                        let floor_split = if total_score > 0 {
                            split_with_floor(actual_distributable, &shares_vec)
                        } else {
                            None
                        };
                        if let Some(coins) = floor_split {
                            for (addr, coin) in coins {
                                coinbase_json.push(serde_json::json!({
                                    "address": hex::encode(addr),
                                    "value": coin,
                                    "salt": hex::encode(rand::random::<[u8; 32]>())
                                }));
                            }
                        } else if total_score > 0 {
                            let mut scores: HashMap<_, i64> = shares_vec.clone().into_iter().map(|(k,v)| (k, v as i64)).collect();
                            for coin in crate::core::types::decompose_value(actual_distributable).into_iter().rev() {
                                let mut best_miner = [0u8; 32];
                                let mut max_score = i64::MIN;
                                for (addr, &score) in &scores {
                                    if score > max_score { max_score = score; best_miner = *addr; }
                                }
                                coinbase_json.push(serde_json::json!({
                                    "address": hex::encode(best_miner),
                                    "value": coin,
                                    "salt": hex::encode(rand::random::<[u8; 32]>())
                                }));
                                let simulated_drop = ((coin as u128 * total_score) / (actual_distributable as u128)) as i64;
                                *scores.get_mut(&best_miner).unwrap() -= simulated_drop.max(1);
                            }
                        } else {
                            for coin in crate::core::types::decompose_value(actual_distributable) {
                                coinbase_json.push(serde_json::json!({
                                    "address": state_clone.pool_address,
                                    "value": coin,
                                    "salt": hex::encode(rand::random::<[u8; 32]>())
                                }));
                            }
                        }
                    }

                    // A retry rebuilds the coinbase, so last attempt's commitment
                    // must not survive into a job whose salt no longer carries it.
                    merge_commit.take();
                    if let Some((commitment, plan)) = &merge_plan {
                        // The first output's salt is the score root that miners
                        // audit; with one output there is nowhere else to put it.
                        if coinbase_json.len() > 1 {
                            let last = coinbase_json.len() - 1;
                            coinbase_json[last]["salt"] = serde_json::json!(commitment);
                            merge_commit = Some(MergeCommit { commit_index: last, ..plan.clone() });
                        }
                    }

                    let req = serde_json::json!({ "coinbase": coinbase_json });
                    let post_res = client.post(&format!("{}/block_template", rpc_url)).json(&req).send().await;
                    if let Err(ref e) = post_res {
                        tracing::error!("block_template request to {} failed: {}", rpc_url, e);
                    }
                    if let Ok(res) = post_res {
                        let parsed = res.json::<serde_json::Value>().await;
                        if let Err(ref e) = parsed {
                            tracing::error!("block_template response was not valid JSON: {}", e);
                        }
                        if let Ok(json) = parsed {
                            if let Some(err) = json.get("error") {
                                let err_str = err.as_str().unwrap_or("");
                                // Re-sync mempool fees dynamically if the node rejects our block value
                                if err_str.contains("Expected: ") {
                                    if let Some(num_str) = err_str.split("Expected: ").nth(1) {
                                        if let Ok(new_expected) = num_str.parse::<u64>() {
                                            tracing::info!("Mempool fees detected. Adjusting block value to {}", new_expected);
                                            expected_total = new_expected;
                                            continue; 
                                        }
                                    }
                                }
                                tracing::error!("Node rejected block template request: {}", err_str);
                                break None;
                            }
                            break Some(json);
                        }
                    }
                    break None;
                };

                if let Some(template) = template_data {
                    if let Some(m_hex) = template["mining_midstate"].as_str() {
                        let mut m_hash = [0u8; 32];
                        hex::decode_to_slice(m_hex, &mut m_hash).unwrap();

                        let job = Job {
                            job_id: job_counter,
                            mining_hash: m_hash,
                            share_target: state_clone.share_target,
                            network_target: n_target,
                            midwimble: merge_commit.clone(),
                            batch_template: template["batch_template"].clone(),
                            height: tip_height.saturating_add(1),
                            committed_scores: std::sync::Arc::new(shares_vec.clone()),
                        };

                        if merging {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map_or(0, |d| d.as_secs());
                            *state_clone.merge_status.write().await =
                                MergeStatus::of_job(merge_commit.as_ref(), merge_error.as_deref(), now);
                        }
                        *state_clone.current_job.write().await = Some(job.clone());
                        let _ = state_clone.job_notifier.send(job);
                        tracing::info!("new job {}: root {}", job_counter, hex::encode(&tree.root[..8]));
                    }
                } else {
                    last_network_tip.clear();
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    loop {
        let (socket, _) = stratum_listener.accept().await.unwrap();
        let state = state.clone();
        tokio::spawn(async move {
            let _ = handle_miner(socket, state).await;
        });
    }
}

// ── HTTP Mining Protocol (browser / web-wallet miners) ──────────────────────
//
// Browsers cannot open raw TCP sockets, so the web wallet can never speak
// Stratum. These two endpoints expose the *same* job and the *same* share
// pipeline over plain request/response HTTP on the audit port:
//
//   GET  /api/template?address=<addr>   → the current job
//   POST /api/submit  {job_id, nonce, address, worker} → share result
//
// The protocol is deliberately poll-based rather than push-based. A browser
// miner runs at single-digit hashes per second (one nonce is a 1,000,000-round
// sequential BLAKE3 chain), so it produces a share every few minutes and a
// one-second template poll is already far finer-grained than it needs.
//
// # Why submissions carry only (job_id, nonce)
// The obvious shape — have the client POST the finished `batch` it just mined —
// is unsafe: the batch contains the coinbase, so a miner could rewrite the
// payout addresses to send the entire block reward to themselves and the pool
// would faithfully relay it to the node. A `job_id` plus a `nonce` is the
// complete proof-of-work claim; the pool rebuilds the block from the template
// it generated and still holds. There is nothing miner-supplied in the block.

/// Query for `GET /api/template`.
#[derive(Deserialize)]
struct HttpTemplateQuery {
    /// Raw 64-char hex or the 72-char checksummed UI address. Used only to
    /// report that miner's current score back to them; it is not registered or
    /// trusted at this point (score is credited on submit, never on request).
    address: String,
}

/// Serves the current job to an HTTP miner.
///
/// `target` is the pool's **share** target, not the network target: browser
/// miners grind toward the (much easier) share threshold and the pool decides
/// server-side whether a given share also happens to clear the network. Both
/// are returned so the client can display real odds.
async fn get_http_template(
    State(state): State<Arc<PoolState>>,
    Query(query): Query<HttpTemplateQuery>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    let addr = match crate::core::types::parse_address_flexible(&query.address) {
        Ok(a) => a,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Invalid address: {}", e) })),
            )
        }
    };

    match build_template_payload(&state, addr).await {
        Ok(v) => (axum::http::StatusCode::OK, Json(v)),
        // No job means the backend node is syncing or unreachable. 503 (not
        // 500) so the client treats it as "retry shortly" rather than fatal.
        Err(e) => (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "error": e })),
        ),
    }
}

/// Reads a miner's current accumulated score.
async fn miner_score(state: &Arc<PoolState>, addr: &[u8; 32]) -> u64 {
    state
        .db
        .begin_read()
        .ok()
        .and_then(|txn| txn.open_table(SHARES_TABLE).ok())
        .and_then(|t| t.get(addr).ok().flatten().map(|v| v.value()))
        .unwrap_or(0)
}

/// Builds the job payload served to a miner, independent of transport.
///
/// Shared verbatim by `/api/template` and the WebRTC `get_template` method, so
/// a browser dialing the pool over libp2p and one polling over HTTP audit the
/// same precommitment against the same fields. Divergence here would mean the
/// two transports could disagree about which root was committed.
async fn build_template_payload(
    state: &Arc<PoolState>,
    addr: [u8; 32],
) -> Result<serde_json::Value, String> {
    let job = match state.current_job.read().await.clone() {
        Some(j) => j,
        None => {
            return Err(
                "Pool has no active job (backend node syncing or unreachable). Retry shortly."
                    .to_string(),
            )
        }
    };

    let shares_recorded = miner_score(state, &addr).await;

    Ok(serde_json::json!({
            // Identifies which template a later submission belongs to. The client
            // must echo this back on submit; it is how a share mined against a
            // superseded template is detected and rejected as stale.
            "job_id":           job.job_id,
            "mining_midstate":  hex::encode(job.mining_hash),
            // The threshold the browser grinds against.
            "target":           hex::encode(job.share_target),
            // Informational: lets the UI show the real block odds behind a share.
            "network_target":   hex::encode(job.network_target),
            // Informational only — the pool rebuilds the block from its own copy.
            // Sent so the wallet can display the tx count for this template.
            "batch_template":   job.batch_template,
            "height":           job.height,
            "shares_recorded":  shares_recorded,
            // Precommitment anchor, so the client can cross-check /api/proof.
            "merkle_root":      hex::encode(state.current_tree.read().await.root),
            // The raw 32-byte pool fee address. The precommitment root is planted
            // in the salt of a coinbase output paying THIS address, so the client
            // needs it to locate the fee coin and confirm the root was actually
            // committed to the block rather than merely reported over the wire.
            "pool_address":     state.pool_address,
            "pool_fee_percent": state.pool_fee_percent,
            "block_reward":     state.current_block_reward.load(std::sync::atomic::Ordering::Relaxed),
    }))
}

/// Body for `POST /api/submit`.
#[derive(Deserialize)]
struct HttpSubmitBody {
    job_id: u64,
    /// Accepted as a JSON string or number. JavaScript's `Number` cannot hold a
    /// u64 without silent precision loss above 2^53, so the web wallet sends the
    /// nonce as a decimal string; `parse_nonce` handles both encodings.
    nonce: serde_json::Value,
    address: String,
    #[serde(default)]
    worker: Option<String>,
    /// Where this share earns midwimble when the pool merge-mines (optional).
    #[serde(default)]
    midwimble_address: Option<String>,
}

/// Validates a share and renders the transport-neutral result body.
///
/// Both `/api/submit` and the WebRTC `submit_share` method call this, so the two
/// transports cannot drift into crediting shares differently or reporting
/// different scores for the same work. HTTP additionally maps the outcome onto a
/// status code; WebRTC just returns the body.
async fn submit_share_payload(
    state: &Arc<PoolState>,
    address: &str,
    worker: Option<&str>,
    midwimble_address: Option<&str>,
    job_id: u64,
    nonce_raw: &serde_json::Value,
) -> (Option<ShareOutcome>, serde_json::Value) {
    let addr = match crate::core::types::parse_address_flexible(address) {
        Ok(a) => a,
        Err(e) => {
            return (
                None,
                serde_json::json!({ "accepted": false, "error": format!("Invalid address: {}", e) }),
            )
        }
    };
    let midwimble = match midwimble_address.map(str::trim).filter(|a| !a.is_empty()) {
        None => None,
        Some(a) => match parse_midwimble_address(a) {
            Ok(a) => Some(a),
            Err(e) => {
                return (
                    None,
                    serde_json::json!({ "accepted": false, "error": format!("Invalid midwimble address: {}", e) }),
                )
            }
        },
    };

    let nonce = match parse_nonce(nonce_raw) {
        Some(n) => n,
        None => {
            return (
                None,
                serde_json::json!({ "accepted": false, "error": "Nonce must be a u64 (decimal string or number)" }),
            )
        }
    };

    // Per-address throttle, checked before the VDF. This is the only thing
    // standing between a publicly reachable endpoint and an unbounded CPU sink,
    // and it applies to WebRTC too: a libp2p stream is no harder to open in bulk
    // than an HTTP request.
    {
        let now = std::time::Instant::now();
        let mut throttle = state.http_throttle.write().await;
        if let Some(last) = throttle.get(&addr) {
            if now.duration_since(*last) < HTTP_MIN_SUBMIT_INTERVAL {
                return (
                    None,
                    serde_json::json!({ "accepted": false, "reason": "throttled", "error": "Submitting too fast" }),
                );
            }
        }
        throttle.insert(addr, now);
    }

    let worker = worker.filter(|w| !w.is_empty()).unwrap_or("browser");

    let outcome = match process_share(state, addr, worker, midwimble.as_deref(), job_id, nonce).await {
        Ok(o) => o,
        Err(e) => {
            tracing::error!("share processing failed: {}", e);
            return (
                None,
                serde_json::json!({ "accepted": false, "error": "Internal error validating share" }),
            );
        }
    };

    // Report the miner's post-credit score so the client can show progress
    // without a second round trip.
    let score = miner_score(state, &addr).await;

    let body = match outcome {
        ShareOutcome::Accepted { is_block } => serde_json::json!({
            "accepted": true,
            "is_block": is_block,
            "score": score,
            "message": if is_block {
                "Share accepted — and it solved the block! Submitting to the network."
            } else {
                "Share accepted."
            }
        }),
        // Stale and duplicate are normal, expected races, not client errors, so
        // the client logs them quietly and keeps mining rather than alarming.
        ShareOutcome::StaleJob => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "stale", "message": "Stale share — the pool has moved to a new job."
        }),
        ShareOutcome::Duplicate => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "duplicate", "message": "Duplicate share."
        }),
        ShareOutcome::LowDifficulty => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "low_difficulty", "message": "Share did not meet the pool target."
        }),
        ShareOutcome::Busy => serde_json::json!({
            "accepted": false, "is_block": false, "score": score,
            "reason": "busy", "message": "Pool verifier saturated — retry this nonce shortly."
        }),
    };

    (Some(outcome), body)
}

/// Accepts a share from an HTTP miner and routes it through the shared pipeline.
async fn post_http_submit(
    State(state): State<Arc<PoolState>>,
    Json(body): Json<HttpSubmitBody>,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    use axum::http::StatusCode;

    let (outcome, payload) = submit_share_payload(
        &state,
        &body.address,
        body.worker.as_deref(),
        body.midwimble_address.as_deref(),
        body.job_id,
        &body.nonce,
    )
    .await;

    let status = match outcome {
        Some(ShareOutcome::Busy) => StatusCode::SERVICE_UNAVAILABLE,
        Some(_) => StatusCode::OK,
        // No outcome means the request never reached the verifier: bad address,
        // bad nonce, or throttled. Distinguish throttling so a client can back
        // off rather than treating it as a permanent malformed-request error.
        None => {
            if payload.get("reason").and_then(|r| r.as_str()) == Some("throttled") {
                StatusCode::TOO_MANY_REQUESTS
            } else if payload.get("error").is_some()
                && payload["error"].as_str().map(|e| e.starts_with("Internal")).unwrap_or(false)
            {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::BAD_REQUEST
            }
        }
    };

    (status, Json(payload))
}

// ── WebRTC Transport (browser miners, no TLS / no CORS / push-based) ─────────
//
// A second front door onto exactly the same job and share pipeline as Stratum
// and HTTP. Browsers cannot open raw TCP, and reaching the HTTP port from an
// HTTPS-served wallet requires a CA-signed certificate on the pool. libp2p's
// webrtc-direct sidesteps both: the connection is authenticated by a certificate
// hash embedded in the multiaddr, so there is no domain, no CA, and no
// mixed-content block — the same trick `light_client.js` already uses to reach
// full nodes.
//
// The decisive advantage over HTTP is not the transport, it is that streams are
// bidirectional and long-lived. The HTTP path has browsers polling once a second
// and receiving the entire `batch_template` — every transaction in the block —
// even when nothing has changed. Here the pool subscribes each connected miner
// to the *same* `job_notifier` broadcast the Stratum handler uses and pushes
// `notify` the instant a job rotates, which also closes the stale-work window
// (with 60s blocks and 1s polling, miners burn up to ~1.6% of their work on
// templates that are already dead).
//
// Framing and protocol shape deliberately mirror `light_protocol.rs`: 4-byte
// little-endian length prefix followed by JSON, one request/response per stream,
// pushes as one-shot server-initiated streams. The wallet's existing
// `/midstate/light-push/2.0.0` reader works the same way, so the client side is
// a near-copy of code that already works.

/// Request/response protocol: miner opens a stream, sends one request, reads one
/// response, stream closes.
pub const POOL_PROTOCOL: libp2p::StreamProtocol =
    libp2p::StreamProtocol::new("/midstate/pool/1.0.0");
/// Push protocol: the pool opens a stream to a miner and sends one notification.
pub const POOL_PUSH_PROTOCOL: libp2p::StreamProtocol =
    libp2p::StreamProtocol::new("/midstate/pool-push/1.0.0");

/// Cap on a single inbound pool request. Requests are tiny (an address, a job id
/// and a nonce); anything larger is malformed or hostile.
const MAX_POOL_MSG_SIZE: u32 = 64 * 1024;

/// Reads one 4-byte-LE-length-prefixed JSON payload from a libp2p stream.
async fn read_framed<T: futures::AsyncRead + Unpin>(io: &mut T) -> anyhow::Result<Vec<u8>> {
    use futures::AsyncReadExt;
    let mut len_bytes = [0u8; 4];
    io.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_POOL_MSG_SIZE {
        anyhow::bail!("pool message too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len as usize];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Writes one 4-byte-LE-length-prefixed JSON payload to a libp2p stream.
async fn write_framed<T: futures::AsyncWrite + Unpin>(
    io: &mut T,
    value: &serde_json::Value,
) -> anyhow::Result<()> {
    use futures::AsyncWriteExt;
    let bytes = serde_json::to_vec(value)?;
    io.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
    io.write_all(&bytes).await?;
    io.flush().await?;
    Ok(())
}

/// Loads the pool's libp2p identity from disk, generating it on first run.
///
/// # Why this must persist
/// The PeerId derived from this key is the `/p2p/<id>` component of the
/// multiaddr miners save in their wallet. Generating a fresh key each boot would
/// silently invalidate every saved pool address on every restart.
fn load_or_create_identity(path: &std::path::Path) -> anyhow::Result<libp2p::identity::Keypair> {
    if path.exists() {
        let bytes = std::fs::read(path)?;
        if let Ok(kp) = libp2p::identity::Keypair::from_protobuf_encoding(&bytes) {
            return Ok(kp);
        }
        tracing::warn!("pool identity at {:?} is unreadable; regenerating", path);
    }
    let kp = libp2p::identity::Keypair::generate_ed25519();
    write_secret(path, &kp.to_protobuf_encoding()?)?;
    tracing::info!("generated new pool libp2p identity at {:?}", path);
    Ok(kp)
}

/// Writes private key material with owner-only permissions.
///
/// Both files this guards hold secrets: the libp2p identity is the pool's
/// signing key, and the WebRTC certificate embeds its private key. A plain
/// `fs::write` creates them under the process umask — commonly world-readable
/// 0644 — which on a shared or containerised host leaks the pool's identity to
/// any local user. Permissions are set before the bytes land, so there is no
/// window during which the file exists with looser modes.
fn write_secret(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        return Ok(());
    }
    #[cfg(not(unix))]
    {
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// Loads the pool's WebRTC certificate from disk, generating it on first run.
///
/// # Why this must persist
/// webrtc-direct authenticates the server by a hash of this certificate, carried
/// in the multiaddr as `/certhash/<...>`. `network/mod.rs` calls
/// `Certificate::generate` on every node startup, which is fine for a node
/// (Kademlia and bootstrap discovery re-find it) but would be a serious bug for
/// a pool: miners persist the pool multiaddr in `localStorage`, so a rotating
/// certhash means every restart silently breaks every saved address with a
/// connection failure that looks like the pool being offline.
fn load_or_create_certificate(
    path: &std::path::Path,
) -> anyhow::Result<libp2p_webrtc::tokio::Certificate> {
    if path.exists() {
        let pem = std::fs::read_to_string(path)?;
        match libp2p_webrtc::tokio::Certificate::from_pem(&pem) {
            Ok(cert) => return Ok(cert),
            Err(e) => tracing::warn!("pool WebRTC certificate at {:?} is invalid ({}); regenerating — SAVED MINER ADDRESSES WILL BREAK", path, e),
        }
    }
    // NOTE: `Certificate::generate` takes its RNG parameter as `_rng` and never
    // reads it — the key comes from the crate's own entropy. So seeding it to
    // derive a reproducible certificate from a stored seed is not possible; the
    // certificate itself must be saved and reloaded, which is why the `pem`
    // feature on libp2p-webrtc is required rather than optional here.
    let cert = libp2p_webrtc::tokio::Certificate::generate(&mut rand::rngs::OsRng)
        .map_err(|e| anyhow::anyhow!("WebRTC certificate generation failed: {}", e))?;
    write_secret(path, cert.serialize_pem().as_bytes())?;
    tracing::info!("generated new pool WebRTC certificate at {:?}", path);
    Ok(cert)
}

/// Closes the write side, then waits for the peer to close theirs before the
/// stream is dropped.
///
/// # Why this is not just `close()`
/// `close()` only *half*-closes: it signals EOF on our write side while our read
/// side stays open. Dropping the stream immediately afterwards — which is what
/// happens when the handler task returns — sends a RESET to the remote. The
/// browser surfaces that as "The stream has been reset" and discards the
/// response, even though the bytes were already on the wire. Draining to EOF
/// first lets the remote observe our FIN and close cleanly.
///
/// Bounded: a client that never closes its side must not pin a task forever.
async fn graceful_close(mut stream: libp2p::Stream) {
    use futures::{AsyncReadExt, AsyncWriteExt};
    if stream.close().await.is_err() {
        return;
    }
    let _ = tokio::time::timeout(Duration::from_secs(5), async move {
        let mut sink = [0u8; 64];
        loop {
            match stream.read(&mut sink).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    })
    .await;
}

/// Handles one inbound pool stream: read a request, answer it, close.
async fn handle_pool_stream(
    state: Arc<PoolState>,
    mut stream: libp2p::Stream,
) -> anyhow::Result<()> {
    let raw = read_framed(&mut stream).await?;
    let req: serde_json::Value = serde_json::from_slice(&raw)?;
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req.get("params").cloned().unwrap_or(serde_json::Value::Null);

    let response = match method {
        "get_template" => {
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            match crate::core::types::parse_address_flexible(address) {
                Ok(addr) => match build_template_payload(&state, addr).await {
                    Ok(v) => serde_json::json!({ "ok": true, "data": v }),
                    Err(e) => serde_json::json!({ "ok": false, "error": e }),
                },
                Err(e) => serde_json::json!({ "ok": false, "error": format!("Invalid address: {}", e) }),
            }
        }
        "submit_share" => {
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            let worker = params.get("worker").and_then(|w| w.as_str());
            let midwimble = params.get("midwimble_address").and_then(|a| a.as_str());
            let job_id = params.get("job_id").and_then(|j| j.as_u64()).unwrap_or(0);
            let nonce = params.get("nonce").cloned().unwrap_or(serde_json::Value::Null);
            let (_, body) = submit_share_payload(&state, address, worker, midwimble, job_id, &nonce).await;
            serde_json::json!({ "ok": true, "data": body })
        }
        "get_proof" => {
            // Mirrors GET /api/proof. Serving the audit over the same transport
            // matters: a miner that fetches its job over WebRTC but its proof
            // over HTTPS gains nothing, because the HTTPS hop is exactly the one
            // that needs a certificate the pool may not have.
            let address = params.get("address").and_then(|a| a.as_str()).unwrap_or("");
            match crate::core::types::parse_address_flexible(address) {
                Ok(addr) => {
                    let tree = state.current_tree.read().await;
                    let score = tree.leaves.iter().find(|(a, _)| a == &addr).map(|(_, s)| *s).unwrap_or(0);
                    match tree.generate_proof(&addr) {
                        Some((idx, proof)) => serde_json::json!({ "ok": true, "data": {
                            "root": hex::encode(tree.root),
                            "score": score,
                            "index": idx,
                            "proof": proof.iter().map(hex::encode).collect::<Vec<_>>()
                        }}),
                        None => serde_json::json!({ "ok": false, "error": "Miner not found in current block precommitment" }),
                    }
                }
                Err(e) => serde_json::json!({ "ok": false, "error": format!("Invalid address: {}", e) }),
            }
        }
        other => serde_json::json!({ "ok": false, "error": format!("Unknown method: {}", other) }),
    };

    write_framed(&mut stream, &response).await?;
    graceful_close(stream).await;
    Ok(())
}

/// Builds the pool's minimal libp2p swarm: webrtc-direct transport plus the
/// stream behaviour, and nothing else — no Kademlia, identify or relay, since
/// miners dial the pool by an address they already hold.
///
/// # Why this is a separate function
/// The `SwarmBuilder` phases return *different* error types:
/// `with_other_transport` yields `Result<_, TransportBuilderError>` while
/// `with_behaviour` yields `Result<_, Infallible>`. `and_then` requires both
/// arms to share one error type, so chaining them that way cannot typecheck no
/// matter how the closures are written. Giving the sequence its own function
/// lets `?` convert each error into one boxed type in turn.
fn build_pool_swarm(
    keypair: libp2p::identity::Keypair,
    certificate: libp2p_webrtc::tokio::Certificate,
) -> Result<libp2p::Swarm<libp2p_stream::Behaviour>, Box<dyn std::error::Error + Send + Sync>> {
    use libp2p_core::{muxing::StreamMuxerBox, Transport};

    Ok(libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_other_transport(move |kp| {
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                libp2p_webrtc::tokio::Transport::new(kp.clone(), certificate)
                    .map(|(peer_id, conn), _| (peer_id, StreamMuxerBox::new(conn))),
            )
        })?
        .with_behaviour(|_| libp2p_stream::Behaviour::new())?
        .with_swarm_config(|c| {
            // Miners hold a connection open for the whole session and may sit
            // idle between shares for minutes at browser hashrates, so the idle
            // timeout has to be generous or the pool disconnects the very miners
            // it is trying to serve.
            c.with_idle_connection_timeout(Duration::from_secs(600))
        })
        .build())
}

/// Boots the pool's WebRTC listener and its job-push loop.
///
/// Returns immediately; the swarm runs on a spawned task. The dialable multiaddr
/// is logged once the listener binds — that string is what miners paste into the
/// wallet, and it is only fully known after libp2p reports the listen address
/// (it carries the certhash).
async fn run_pool_webrtc(state: Arc<PoolState>, udp_port: u16, data_dir: std::path::PathBuf) {
    use futures::StreamExt;

    let keypair = match load_or_create_identity(&data_dir.join("pool_identity.key")) {
        Ok(k) => k,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not load identity: {}", e);
            return;
        }
    };
    let certificate = match load_or_create_certificate(&data_dir.join("pool_webrtc_cert.pem")) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not load certificate: {}", e);
            return;
        }
    };

    let local_peer = keypair.public().to_peer_id();

    let mut swarm = match build_pool_swarm(keypair, certificate) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: swarm construction failed: {}", e);
            return;
        }
    };

    let mut control = swarm.behaviour_mut().new_control();
    let mut incoming = match control.accept(POOL_PROTOCOL) {
        Ok(i) => i,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: could not register protocol: {}", e);
            return;
        }
    };

    let listen: libp2p::Multiaddr = match format!("/ip4/0.0.0.0/udp/{}/webrtc-direct", udp_port).parse() {
        Ok(a) => a,
        Err(e) => {
            tracing::error!("pool WebRTC disabled: bad listen address: {}", e);
            return;
        }
    };
    if let Err(e) = swarm.listen_on(listen) {
        tracing::error!("pool WebRTC disabled: listen failed on udp/{}: {}", udp_port, e);
        return;
    }

    // Miners currently connected, so job pushes go only to live sessions.
    let peers: Arc<RwLock<HashSet<libp2p::PeerId>>> = Arc::new(RwLock::new(HashSet::new()));

    // ── Job push loop ──
    // Subscribes to the SAME broadcast the Stratum handler uses, so a WebRTC
    // miner and a Stratum miner are notified of a new job by the same event and
    // can never be served different templates for the same height.
    {
        let peers = peers.clone();
        let mut job_rx = state.job_notifier.subscribe();
        let push_control = control.clone();
        tokio::spawn(async move {
            loop {
                let job = match job_rx.recv().await {
                    Ok(j) => j,
                    // Lagged means this task fell behind the broadcast ring. The
                    // next job supersedes any it missed, so resubscribing and
                    // waiting is correct — a miner briefly holds a stale template
                    // and the job_id check rejects its shares as stale, which is
                    // exactly the designed behaviour.
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("pool webrtc push lagged {} job(s)", n);
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                let notification = serde_json::json!({
                    "method": "notify",
                    "params": {
                        "job_id":          job.job_id,
                        "mining_midstate": hex::encode(job.mining_hash),
                        "target":          hex::encode(job.share_target),
                        "network_target":  hex::encode(job.network_target),
                        "height":          job.height,
                    }
                });

                let targets: Vec<_> = peers.read().await.iter().copied().collect();
                for peer in targets {
                    let mut ctrl = push_control.clone();
                    let payload = notification.clone();
                    let peers = peers.clone();
                    tokio::spawn(async move {
                        // Bounded, like the node's light-push: a browser that
                        // stops reading must not pin a task forever.
                        let ok = tokio::time::timeout(Duration::from_secs(5), async move {
                            let mut stream = ctrl.open_stream(peer, POOL_PUSH_PROTOCOL).await.ok()?;
                            write_framed(&mut stream, &payload).await.ok()?;
                            // Same reset hazard as the request path: the browser
                            // reads until it has the whole framed payload, so
                            // dropping early loses the notification outright.
                            graceful_close(stream).await;
                            Some(())
                        })
                        .await;
                        if !matches!(ok, Ok(Some(()))) {
                            // A failed push means the miner is gone or wedged.
                            // Drop it so the peer set tracks reality.
                            peers.write().await.remove(&peer);
                        }
                    });
                }
            }
        });
    }

    // ── Inbound stream loop ──
    {
        let state = state.clone();
        tokio::spawn(async move {
            while let Some((peer, stream)) = incoming.next().await {
                let state = state.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_pool_stream(state, stream).await {
                        tracing::debug!("pool webrtc stream from {} ended: {}", peer, e);
                    }
                });
            }
        });
    }

    // ── Swarm event loop ──
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                    // This is the string miners paste into the wallet. It only
                    // becomes known here because libp2p appends the certhash.
                    let dialable = address
                        .clone()
                        .with(libp2p::multiaddr::Protocol::P2p(local_peer));
                    tracing::info!("pool WebRTC listening — miners connect to: {}", dialable);
                }
                libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                    peers.write().await.insert(peer_id);
                    tracing::debug!("pool webrtc miner connected: {}", peer_id);
                }
                libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id, num_established, .. } => {
                    if num_established == 0 {
                        peers.write().await.remove(&peer_id);
                        tracing::debug!("pool webrtc miner disconnected: {}", peer_id);
                    }
                }
                _ => {}
            }
        }
    });
}

/// Parses a nonce supplied as either a JSON number or a decimal string.
fn parse_nonce(v: &serde_json::Value) -> Option<u64> {
    match v {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

// ── Shared Share Pipeline (Stratum + HTTP) ──────────────────────────────────

/// The result of validating one submitted share.
#[derive(Debug, Clone, Copy)]
enum ShareOutcome {
    /// Met the pool's share target and was credited. `is_block` is true when the
    /// same hash also cleared the full network target, in which case the block
    /// has been handed to the background submission task.
    Accepted { is_block: bool },
    /// This nonce was already credited for this job.
    Duplicate,
    /// Genuine work, but above the share target.
    LowDifficulty,
    /// The `job_id` doesn't match the job currently being served.
    StaleJob,
    /// The verifier is saturated; the miner should retry the same nonce.
    Busy,
}

/// Validates and credits one share. This is the single entry point for **both**
/// the Stratum socket handler and the HTTP `/api/submit` endpoint.
///
/// # Reasoning
/// Replay protection, scoring, the network-target check and the block-submission
/// handoff are all consensus- and payout-critical. Implementing them twice — once
/// per transport — is how the two paths silently drift apart (one gets a bug fix,
/// the other doesn't, and the pool starts paying two different rates depending on
/// which port a miner connected to). Everything transport-specific stays in the
/// callers; everything that touches score lives here.
///
/// # Ordering note (replay cache vs. proof-of-work)
/// The nonce is inserted into `valid_shares` only **after** the work is proven.
/// Claiming it up front — as the original inline Stratum path did — lets anyone
/// with a socket burn the nonce space: submitting garbage nonces poisons the
/// cache so that an honest miner who later finds a real share on one of those
/// nonces is rejected as a "duplicate", and the set grows without bound on
/// unvalidated input until the next block clears it. Probing for membership
/// first (cheap) and inserting after (proven) keeps the fast path fast while
/// making the cache hold only real work.
///
/// # Formal Specification
///
/// ```text
/// Let  H(m, n)   = create_extension(m, n).final_hash        (the 1e6-round VDF)
///      score     : Address ⇸ ℕ                              (SHARES_TABLE)
///      seen      : ℙ ℕ                                      (valid_shares)
///      job       = current_job
///
/// Pre:
///   - true  (total: every rejection path is a normal return, never an error)
///
/// Post — exactly one outcome, and score changes on Accepted alone:
///
///   outcome! = Accepted{is_block} ⇔
///        job ≠ ⊥ ∧ job.job_id = job_id?
///      ∧ nonce? ∉ seen
///      ∧ H(job.mining_hash, nonce?) < job.share_target
///      ∧ seen'   = seen ∪ {nonce?}
///      ∧ score'  = score ⊕ {addr? ↦ score(addr?) + 1}
///      ∧ is_block = H(job.mining_hash, nonce?) < job.network_target
///
///   outcome! ∈ {StaleJob, Duplicate, LowDifficulty, Busy} ⇒
///        score' = score                       (no credit, ever)
///      ∧ seen'  = seen                        (no nonce burned)
///      ∧ rejected'(addr?) = rejected(addr?) + 1
///
/// Credit is exactly one point per accepted share, independent of how far
/// H fell below share_target. See `HTTP_MIN_SUBMIT_INTERVAL` and `--share-bits`:
/// changing the share target rescales what a point costs.
/// ```
///
/// ```zed
///     ProcessShare
///     ----------------
///     ΔShares
///     ΔValidShares
///     ΞCurrentJob
///     addr?    : Address
///     job_id?  : ℕ
///     nonce?   : ℕ
///     outcome! : ShareOutcome
///
///     pre  true
///
///     post outcome! = Accepted ⇒
///            job ≠ ⊥ ∧ job.job_id = job_id?
///            ∧ nonce? ∉ seen
///            ∧ H(job.mining_hash, nonce?) < job.share_target
///            ∧ seen' = seen ∪ {nonce?}
///            ∧ score' = score ⊕ {addr? ↦ score addr? + 1}
///
///     post outcome! ≠ Accepted ⇒ score' = score ∧ seen' = seen
///
///     post job' = job          (this operation never rotates the template)
/// ```
///
/// # Safety / Invariants
///
/// - **`seen` only ever holds proven nonces.** Insertion happens strictly after
///   the target comparison. This is what stops an unauthenticated caller from
///   burning nonce space or growing the set on garbage input.
/// - **A rejected share must leave `seen` untouched.** The late-stale path
///   therefore *rolls back* its insertion: `seen` is keyed by nonce alone and
///   cleared per **block**, while jobs rotate more often than blocks (any
///   template rebuild on the same tip). A burned entry would reject that nonce
///   for the remainder of the block even though the new job's midstate makes it
///   legitimately re-mineable.
/// - **Score is credited at most once per (job, nonce).** Enforced by `seen`,
///   which is why its correctness is a payout property and not just a
///   performance one.
/// - **`is_block` is decided from the same hash that cleared `share_target`,**
///   never recomputed, so a share and the block it may also solve can never
///   disagree.
/// - **This is the single entry point for both transports.** Any invariant above
///   holds for Stratum, HTTP and WebRTC identically, because none of them may
///   credit score by another route.
async fn process_share(
    state: &Arc<PoolState>,
    miner_addr: [u8; 32],
    worker_name: &str,
    midwimble_addr: Option<&str>,
    job_id: u64,
    nonce: u64,
) -> anyhow::Result<ShareOutcome> {
    // Snapshot the job. Everything below validates against THIS job, and the
    // job_id is re-checked after the VDF so a template rotation mid-verify can't
    // credit work against a template that is no longer being paid out.
    let job = match state.current_job.read().await.clone() {
        Some(j) if j.job_id == job_id => j,
        _ => return Ok(ShareOutcome::StaleJob),
    };

    // Cheap probe before spending ~150ms of CPU on a nonce we already hold.
    // The guard is scoped and dropped before the `.await` below: holding a read
    // lock across an await point is how a write-preferring RwLock deadlocks.
    let seen = { state.valid_shares.read().await.contains(&nonce) };
    if seen {
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::Duplicate);
    }

    // Hold a permit across the blocking VDF so the two transports share one
    // CPU budget. `try_acquire` rather than `acquire`: a queued HTTP request
    // just holds a connection open and lets the backlog grow, whereas telling
    // the miner "busy, retry" keeps the work on their side.
    let _permit = match state.verify_permits.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return Ok(ShareOutcome::Busy),
    };

    let m_hash = job.mining_hash;
    let ext = tokio::task::spawn_blocking(move || create_extension(m_hash, nonce)).await?;
    drop(_permit);

    if ext.final_hash >= job.share_target {
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::LowDifficulty);
    }

    // Claim the nonce now that the work behind it is proven. Losing this race
    // means another connection submitted the same proven nonce first.
    {
        let mut cache = state.valid_shares.write().await;
        if !cache.insert(nonce) {
            drop(cache);
            bump_rejected(state, miner_addr).await;
            return Ok(ShareOutcome::Duplicate);
        }
    }

    // The VDF above is slow enough that a new block can land mid-verify. If the
    // job rotated, the share is against a dead template: don't credit it, and
    // don't try to submit it as a block.
    let still_current = {
        matches!(state.current_job.read().await.as_ref(), Some(j) if j.job_id == job_id)
    };
    if !still_current {
        // Release the nonce we just claimed. `valid_shares` is keyed by nonce
        // alone and is cleared per BLOCK, not per job — several jobs share one
        // cache whenever the template is rebuilt on the same tip. Leaving a
        // burned entry behind would therefore reject that nonce for the rest of
        // the block, even though the rotated job has a different midstate and the
        // nonce is legitimately re-mineable against it. Rolling back keeps the
        // postcondition "outcome ≠ Accepted ⇒ valid_shares' = valid_shares" true.
        state.valid_shares.write().await.remove(&nonce);
        bump_rejected(state, miner_addr).await;
        return Ok(ShareOutcome::StaleJob);
    }

    {
        let write_txn = state.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SHARES_TABLE)?;
            let current = table.get(&miner_addr)?.map(|v| v.value()).unwrap_or(0);
            table.insert(&miner_addr, current + 1)?;
        }
        // Merge-mining: the same share earns a point of midwimble score for
        // the midwimble address this miner gave, or the pool's own if none.
        if let (Some(_), Some(pool_mw)) = (&state.midwimble_rpc, &state.midwimble_address) {
            let key = midwimble_addr.unwrap_or(pool_mw.as_str());
            let mut table = write_txn.open_table(MW_SHARES_TABLE)?;
            let current = table.get(key)?.map(|v| v.value()).unwrap_or(0);
            table.insert(key, current + 1)?;
        }
        write_txn.commit()?;
    }

    state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).0 += 1;
    *state
        .worker_stats
        .write()
        .await
        .entry((miner_addr, worker_name.to_string()))
        .or_insert(0) += 1;

    let is_block = ext.final_hash < job.network_target;
    if is_block {
        tracing::info!(
            "block found by miner {}. submitting to network.",
            hex::encode(&miner_addr[..8])
        );
        spawn_block_submission(state, job.clone(), ext.clone());
    }
    if let Some(merge) = job.midwimble.clone() {
        if ext.final_hash < merge.target && !merge.claimed.swap(true, std::sync::atomic::Ordering::SeqCst) {
            tracing::info!("that share is also a midwimble block; submitting it");
            // Taken now, before the block can move midwimble's tip, so the
            // next job is never built on scores this block already pays.
            let taken = match settle_midwimble_scores(&state.db, &merge.paid, false) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!("could not take the scores this midwimble block pays: {e}");
                    false
                }
            };
            spawn_midwimble_submission(state, &job, merge, &ext, taken);
        }
    }

    Ok(ShareOutcome::Accepted { is_block })
}

/// Records a rejected share against a miner's lifetime efficiency tally.
async fn bump_rejected(state: &Arc<PoolState>, miner_addr: [u8; 32]) {
    state.share_stats.write().await.entry(miner_addr).or_insert((0, 0)).1 += 1;
}

/// A share that also cleared midwimble's target is a midwimble block. The
/// midwimble node rebuilds the parent proof from this template, seals the
/// block and applies it; the pool never touches midwimble's block format.
///
/// The scores the block pays were taken out of the table before this was
/// called (`scores_taken`). If the node doesn't accept the block they are put
/// back, and the job may submit another winning share.
fn spawn_midwimble_submission(
    state: &Arc<PoolState>,
    job: &Job,
    merge: MergeCommit,
    ext: &Extension,
    scores_taken: bool,
) {
    let Some(rpc) = state.midwimble_rpc.clone() else {
        return;
    };
    let body = serde_json::json!({
        "mining_hash": merge.mining_hash,
        "batch_template": job.batch_template,
        "commit_index": merge.commit_index,
        "nonce": ext.nonce,
        "final_hash": hex::encode(ext.final_hash),
    });
    let state = state.clone();
    tokio::spawn(async move {
        let answer = match reqwest::Client::new()
            .post(format!("{rpc}/merge/found"))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => {
                let ok = r.status().is_success();
                let reply: serde_json::Value = r.json().await.unwrap_or_default();
                if ok && reply["accepted"].as_bool() == Some(true) {
                    Ok(reply)
                } else {
                    Err(reply.to_string())
                }
            }
            Err(e) => Err(e.to_string()),
        };
        match answer {
            Ok(reply) => {
                tracing::info!("midwimble block accepted at height {}", reply["height"]);
                if let Err(e) = record_midwimble_block(&state.db, reply["height"].as_u64().or(merge.height), &merge) {
                    tracing::warn!("could not record the midwimble block for the dashboard: {e}");
                }
                // Midwimble's tip has moved, so this job's commitment is spent.
                state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Err(e) => {
                tracing::warn!("midwimble block not accepted: {e}");
                if scores_taken {
                    if let Err(e) = settle_midwimble_scores(&state.db, &merge.paid, true) {
                        tracing::error!("could not restore the scores of a refused midwimble block: {e}");
                    }
                }
                merge.claimed.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
    });
}

/// Reconstructs the full block from the pool's own stored template, submits it
/// to the core node, and — only on a confirmed `HTTP 200` — applies the score
/// deductions and writes the payout/committed-score history.
///
/// # Security
/// The batch is rebuilt from `job.batch_template`, the template the pool itself
/// generated and broadcast. Nothing block-shaped supplied by a miner is ever
/// used: a share submission carries only a `job_id` and a `nonce`, so a miner
/// cannot rewrite the coinbase to redirect the reward to themselves.
fn spawn_block_submission(state: &Arc<PoolState>, job: Job, ext: Extension) {
    // Capture block identity for the dashboard BEFORE `job` is consumed: the PoW
    // hash (for the Explorer hyperlink) and the network target in force for this
    // block, so historical "luck" is computed exactly rather than approximated
    // against the *current* difficulty.
    let block_hash_hex = hex::encode(ext.final_hash);
    let block_net_target_hex = hex::encode(job.network_target);
    let block_height = job.height;
    let committed_scores = job.committed_scores.clone();

    let mut batch: Batch = match serde_json::from_value(job.batch_template) {
        Ok(b) => b,
        Err(e) => {
            // Previously an `.unwrap()`. A malformed template is the node's or
            // the pool's fault, not the miner's, and panicking here would kill
            // the connection task holding a perfectly good block.
            tracing::error!("could not rebuild block from stored template: {}. dropping block.", e);
            state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            return;
        }
    };
    batch.extension = ext;

    let total_reward: u64 = batch.coinbase.iter().map(|cb| cb.value).sum();
    let batch_for_node = batch.clone();
    let db_clone = state.db.clone();
    let rpc_url = state.node_rpc_url.clone();
    let submit_state = state.clone();

    tokio::spawn(async move {
        let res = reqwest::Client::new()
            .post(&format!("{}/submit_batch", rpc_url))
            .json(&batch_for_node)
            .send()
            .await;

        match res {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!("block accepted by network. applying score deductions.");
                let write_txn = db_clone.begin_write().unwrap();
                {
                    let mut table = write_txn.open_table(SHARES_TABLE).unwrap();
                    let mut total_score = 0u128;
                    for iter in table.iter().unwrap() { total_score += iter.unwrap().1.value() as u128; }

                    let mut payouts = Vec::new();

                    // Track exact deductions to allow refunds on orphan ---
                    let mut actual_deductions = Vec::new();

                    for cb in &batch.coinbase {
                        let mut a = [0u8; 32]; a.copy_from_slice(&cb.address);
                        let deduction = ((cb.value as u128 * total_score) / (total_reward as u128)) as u64;
                        // Deduct ONLY from rows that already exist, and delete a row once its
                        // remaining score hits 0. Writing back `0.saturating_sub(d) = 0` for an
                        // absent address (the pool fee address, or any coinbase output not in
                        // SHARES_TABLE) is what previously seeded permanent zero-score rows; those
                        // rows then leaked into the allocator and captured block rewards. redb has
                        // no delete-if-absent, so guard the remove on the row existing.
                        if let Some(current) = table.get(&a).unwrap().map(|v| v.value()) {
                            let remaining = current.saturating_sub(deduction);
                            if remaining > 0 {
                                table.insert(&a, remaining).unwrap();
                            } else {
                                table.remove(&a).unwrap();
                                // Record deduction ---
                                actual_deductions.push((hex::encode(a), deduction));
                            }
                        }

                        payouts.push(serde_json::json!({
                            "address": crate::core::types::encode_address_with_checksum(&a),
                            "value": cb.value
                        }));
                    }

                    let mut b_table = write_txn.open_table(BLOCKS_TABLE).unwrap();
                    let block_data = serde_json::json!({
                        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                        "block_ts": batch.timestamp,
                        "hash": block_hash_hex,
                        "height": block_height,
                        "total_score": total_score as u64,
                        "net_target": block_net_target_hex,
                        "payouts": payouts,
                        "status": "pending" 
                    }).to_string();
                    b_table.insert(batch.timestamp, block_data.as_str()).unwrap();

                    // Save to Pending Blocks for Maturity Pipeline ---
                    let pending_data = serde_json::json!({
                        "hash": block_hash_hex,
                        "block_ts": batch.timestamp,
                        "deductions": actual_deductions
                    }).to_string();
                    let mut p_table = write_txn.open_table(PENDING_BLOCKS_TABLE).unwrap();
                    p_table.insert(block_height, pending_data.as_str()).unwrap();
                    // ---------------------------------------------------------

                    // Persist the committed score snapshot (split-verification).
                    // `committed_total` is the sum of the snapshot — the basis the coinbase
                    // payouts were proportioned against, not the accept-time table total
                    // used for the deductions above.
                    let committed_total: u128 = committed_scores.iter().map(|(_, s)| *s as u128).sum();
                    let scores_json: Vec<serde_json::Value> = committed_scores.iter().map(|(a, s)| serde_json::json!({
                        "address": crate::core::types::encode_address_with_checksum(a),
                        "score": s
                    })).collect();
                    let scores_data = serde_json::json!({
                        "total_score": committed_total as u64,
                        "scores": scores_json
                    }).to_string();
                    let mut s_table = write_txn.open_table(BLOCK_SCORES_TABLE).unwrap();
                    s_table.insert(batch.timestamp, scores_data.as_str()).unwrap();
                }
                write_txn.commit().unwrap();
            }
            Ok(resp) => {
                // The node answered but refused the block (stale, invalid, etc).
                // The tip won't change on a rejection, so force the template loop
                // to rebuild instead of letting miners re-grind the doomed template.
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                tracing::warn!(
                    "block rejected by network ({}): {}. retaining miner scores; requesting fresh job.",
                    status, body.trim()
                );
                submit_state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            Err(e) => {
                // The submission never reached the node (connection refused,
                // timeout, dropped mid-flight). Scores are retained (deductions
                // only happen on confirmed acceptance) and a fresh job is forced.
                tracing::error!(
                    "block submission to node failed: {}. retaining miner scores; requesting fresh job.",
                    e
                );
                submit_state.force_new_job.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
    });
}

/// Reconciles pending blocks against the canonical chain to detect orphans and refund shares.
///
/// # Reasoning
/// When the pool submits a block and the node accepts it (HTTP 200 OK), the pool deducts 
/// shares from miners to prevent double-paying if they continue mining. However, due to 
/// standard network latency and Nakamoto Consensus tie-breakers (e.g., midstate comparison), 
/// the local node might reorganize and abandon the pool's block seconds later.
/// 
/// Previously, the pool never checked the chain again. If a block was orphaned, miners 
/// permanently lost their shares for a block that yielded no actual rewards, destroying 
/// trust in the pool. This function introduces a maturity pipeline: it verifies block 
/// hashes at `MATURITY_DEPTH` and refunds all deducted shares if the block was orphaned.
///
/// # Formal Specification
///
/// ```text
/// Pre:
///   - MATURITY_DEPTH = 10
///   - For every pending block b in PENDING_BLOCKS_TABLE where b.height <= current_network_height - MATURITY_DEPTH
///   - The core node is reachable via RPC
///
/// Post:
///   result = Ok(()) ⇒
///     ∀ b ∈ evaluated_blocks:
///       if network_hash(b.height) == b.hash then
///         PENDING_BLOCKS_TABLE' = PENDING_BLOCKS_TABLE \ {b}
///         BLOCKS_TABLE'[b.ts].status = "confirmed"
///       else (Orphaned)
///         PENDING_BLOCKS_TABLE' = PENDING_BLOCKS_TABLE \ {b}
///         BLOCKS_TABLE'[b.ts].status = "orphaned"
///         ∀ (addr, amt) ∈ b.deductions:
///           SHARES_TABLE'[addr] = SHARES_TABLE[addr] + amt
/// ```
///
/// ```zed
///     ReconcilePendingBlocks
///     ----------------------
///     ΔShares
///     ΔPendingBlocks
///     ΔBlocks
///     current_height? : ℕ
///     network : RPC
///
///     let mature = { b ∈ PendingBlocks | b.height ≤ current_height? - 10 }
///
///     pre  true
///     post ∀ b ∈ mature •
///            (network.hash_at(b.height) = b.hash ⇒ 
///               PendingBlocks' = PendingBlocks \ {b} ∧
///               Blocks'[b.ts].status = "confirmed")
///          ∧ (network.hash_at(b.height) ≠ b.hash ⇒
///               PendingBlocks' = PendingBlocks \ {b} ∧
///               Blocks'[b.ts].status = "orphaned" ∧
///               Shares' = Shares ⊕ { d.addr ↦ Shares(d.addr) + d.amount | d ∈ b.deductions })
/// ```
///
/// # Safety / Invariants
/// - **No Double Refunds**: Blocks are removed from `PENDING_BLOCKS_TABLE` in the same
///   atomic database transaction that restores the shares.
/// - **Idempotent network reads**: Network failures skip reconciliation for that tick;
///   the block remains pending until a clean read confirms its status.
async fn reconcile_pending_blocks(state: &Arc<PoolState>, client: &reqwest::Client, current_height: u64) -> Result<(), String> {
    const MATURITY_DEPTH: u64 = 10;
    let maturity_threshold = current_height.saturating_sub(MATURITY_DEPTH);

    // 1. Gather mature pending blocks
    let mut mature_blocks = Vec::new();
    if let Ok(read_txn) = state.db.begin_read() {
        if let Ok(table) = read_txn.open_table(PENDING_BLOCKS_TABLE) {
            let range = 0..=maturity_threshold;
            if let Ok(iter) = table.range(range) {
                for entry in iter {
                    if let Ok((k, v)) = entry {
                        mature_blocks.push((k.value(), v.value().to_string()));
                    }
                }
            }
        }
    }

    if mature_blocks.is_empty() {
        return Ok(());
    }

    let mut confirmed = Vec::new();
    let mut orphaned = Vec::new();

    // 2. Query the node for the canonical block at each mature height
    for (height, json_str) in mature_blocks {
        let pending_data: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();
        let expected_hash = pending_data["hash"].as_str().unwrap_or("");
        
        let url = format!("{}/block/{}", state.node_rpc_url, height);
        match client.get(&url).send().await {
            Ok(resp) => {
                if resp.status().is_success() {
                    if let Ok(batch) = resp.json::<serde_json::Value>().await {
                        let canonical_hash = batch["extension"]["final_hash"].as_str().unwrap_or("");
                        if canonical_hash == expected_hash {
                            confirmed.push((height, pending_data));
                        } else {
                            orphaned.push((height, pending_data));
                        }
                    }
                } else if resp.status() == 404 || resp.status() == 400 || resp.status() == 500 {
                    // Block not found or invalid height on node -> definitively orphaned
                    orphaned.push((height, pending_data));
                }
            }
            Err(e) => {
                tracing::warn!("Failed to query node for block {}: {}. Will retry reconciliation later.", height, e);
                // Network error, skip reconciliation for this block until next tick
            }
        }
    }

    if confirmed.is_empty() && orphaned.is_empty() {
        return Ok(());
    }

    // 3. Atomically apply resolutions
    let write_txn = state.db.begin_write().map_err(|e| e.to_string())?;
    {
        let mut p_table = write_txn.open_table(PENDING_BLOCKS_TABLE).map_err(|e| e.to_string())?;
        let mut b_table = write_txn.open_table(BLOCKS_TABLE).map_err(|e| e.to_string())?;
        let mut s_table = write_txn.open_table(SHARES_TABLE).map_err(|e| e.to_string())?;

        // Process Confirmed
        for (height, data) in confirmed {
            p_table.remove(height).map_err(|e| e.to_string())?;
            
            let block_ts = data["block_ts"].as_u64().unwrap_or(0);
            
            // Isolate the read to drop the AccessGuard before writing
            let mut b_json = None;
            if let Ok(Some(b_val)) = b_table.get(block_ts) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(b_val.value()) {
                    b_json = Some(parsed);
                }
            }

            if let Some(mut json) = b_json {
                json["status"] = serde_json::json!("confirmed");
                b_table.insert(block_ts, json.to_string().as_str()).map_err(|e| e.to_string())?;
            }
            
            tracing::info!("Block at height {} reached maturity (Confirmed).", height);
        }

        // Process Orphans and Refund Shares
        for (height, data) in orphaned {
            p_table.remove(height).map_err(|e| e.to_string())?;
            
            let block_ts = data["block_ts"].as_u64().unwrap_or(0);
            
            // Isolate the read to drop the AccessGuard before writing
            let mut b_json = None;
            if let Ok(Some(b_val)) = b_table.get(block_ts) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(b_val.value()) {
                    b_json = Some(parsed);
                }
            }

            if let Some(mut json) = b_json {
                json["status"] = serde_json::json!("orphaned");
                b_table.insert(block_ts, json.to_string().as_str()).map_err(|e| e.to_string())?;
            }

            // Refund the shares
            if let Some(deductions) = data["deductions"].as_array() {
                for d in deductions {
                    let addr_hex = d[0].as_str().unwrap_or("");
                    let amount = d[1].as_u64().unwrap_or(0);
                    
                    if let Ok(addr_bytes) = hex::decode(addr_hex) {
                        if let Ok(addr_array) = <[u8; 32]>::try_from(addr_bytes.as_slice()) {
                            let current = s_table.get(&addr_array).map_err(|e| e.to_string())?.map(|v| v.value()).unwrap_or(0);
                            s_table.insert(&addr_array, current + amount).map_err(|e| e.to_string())?;
                        }
                    }
                }
            }
            tracing::warn!("🚨 ORPHAN DETECTED at height {}. Shares have been fully refunded to miners.", height);
        }
    }
    write_txn.commit().map_err(|e| e.to_string())?;

    Ok(())
}

// ── Stratum Connection Handler ──────────────────────────────────────────────

/// Handles an active TCP Stratum session with a miner.
///
/// # Security Mechanisms
/// 1. **Replay Protection**: Identical nonces for the same block are rejected instantly.
/// 2. **CPU Offloading**: PoW hashing runs in `spawn_blocking` to protect the reactor.
/// 3. **Orphan Theft Defense**: Miner scores are only wiped if the core node accepts the block.
async fn handle_miner(mut socket: TcpStream, state: Arc<PoolState>) -> anyhow::Result<()> {
    let (read_half, mut write_half) = socket.split();
    let mut reader = BufReader::new(read_half);
    let mut line = String::new();
    let mut job_rx = state.job_notifier.subscribe();
    let mut authorized_address = None;
    // Worker name from mining.authorize params[1] (stratum convention). Scopes this
    // connection's accepted shares to a rig for the per-worker breakdown; defaults
    // when a miner omits it (today's reference miner sends "worker1").
    let mut authorized_worker: String = "default".to_string();
    // Optional midwimble address from mining.authorize params[2]: where this
    // connection's shares earn midwimble when the pool merge-mines.
    let mut authorized_midwimble: Option<String> = None;

    loop {
        tokio::select! {
            res = reader.read_line(&mut line) => {
                if res? == 0 { break; } 
                let req: StratumRequest = match serde_json::from_str(&line) {
                    Ok(r) => r,
                    Err(_) => { line.clear(); continue; }
                };
                line.clear();

                match req.method.as_str() {
                    "mining.subscribe" => {
                        let res = StratumResponse { id: req.id, result: Some(serde_json::json!(true)), error: None };
                        write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                    }
                    "mining.authorize" => {
                        let address = req.params.get(0).and_then(|v| v.as_str()).unwrap_or("").to_string();
                        // params[1] is the worker name by stratum convention (the reference
                        // miner sends "worker1"); record it for this connection's breakdown.
                        if let Some(w) = req.params.get(1).and_then(|v| v.as_str()) {
                            if !w.is_empty() { authorized_worker = w.to_string(); }
                        }
                        // params[2], optional: a midwimble address to be paid
                        // in midwimble. A bad one refuses the login, so the
                        // miner finds out now rather than never being paid.
                        let midwimble = match req.params.get(2).and_then(|v| v.as_str()).map(str::trim) {
                            None | Some("") => Ok(None),
                            Some(a) => parse_midwimble_address(a).map(Some),
                        };
                        if let Err(e) = &midwimble {
                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some(format!("Invalid midwimble address: {e}")) };
                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            continue;
                        }
                        // Strip the UI checksum from the miner's address
                        if let Ok(addr_bytes) = crate::core::types::parse_address_flexible(&address) {
                            authorized_address = Some(addr_bytes);
                            authorized_midwimble = midwimble.unwrap_or(None);
                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(true)), error: None };
                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;

                            if let Some(job) = state.current_job.read().await.clone() {
                                let notif = StratumRequest {
                                    id: None,
                                    method: "mining.notify".into(),
                                    params: vec![
                                        serde_json::json!(job.job_id),
                                        serde_json::json!(hex::encode(job.mining_hash)),
                                        serde_json::json!(job.batch_template),
                                        serde_json::json!(hex::encode(job.share_target)),
                                    ]
                                };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&notif)?).as_bytes()).await?;
                            }
                        } else {
                            let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Invalid Address".into()) };
                            write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                        }
                    }
                    "mining.submit" => {
                        // params: [worker_name, job_id, nonce]. Missing/!numeric
                        // params previously hit `.unwrap()` and killed the task,
                        // dropping the whole connection on one malformed frame.
                        let job_id = req.params.get(1).and_then(|v| v.as_u64());
                        let nonce = req.params.get(2).and_then(parse_nonce);

                        match (authorized_address, job_id, nonce) {
                            (Some(miner_addr), Some(job_id), Some(nonce)) => {
                                let outcome = process_share(
                                    &state, miner_addr, &authorized_worker,
                                    authorized_midwimble.as_deref(), job_id, nonce,
                                ).await?;

                                let (ok, err) = match outcome {
                                    ShareOutcome::Accepted { .. } => (true, None),
                                    ShareOutcome::Duplicate      => (false, Some("Duplicate share")),
                                    ShareOutcome::LowDifficulty  => (false, Some("Low difficulty")),
                                    ShareOutcome::StaleJob       => (false, Some("Stale job")),
                                    ShareOutcome::Busy           => (false, Some("Verifier busy, retry")),
                                };
                                let res = StratumResponse {
                                    id: req.id,
                                    result: Some(serde_json::json!(ok)),
                                    error: err.map(|e| e.to_string()),
                                };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            }
                            (None, _, _) => {
                                let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Not authorized".into()) };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            }
                            _ => {
                                let res = StratumResponse { id: req.id, result: Some(serde_json::json!(false)), error: Some("Malformed submit params".into()) };
                                write_half.write_all(format!("{}\n", serde_json::to_string(&res)?).as_bytes()).await?;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(job) = job_rx.recv() => {
                let notif = StratumRequest {
                    id: None,
                    method: "mining.notify".into(),
                    params: vec![
                        serde_json::json!(job.job_id),
                        serde_json::json!(hex::encode(job.mining_hash)),
                        serde_json::json!(job.batch_template),
                        serde_json::json!(hex::encode(job.share_target)),
                    ]
                };
                write_half.write_all(format!("{}\n", serde_json::to_string(&notif)?).as_bytes()).await?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    /// Two real addresses, printed by `midwimble wallet address` for throwaway
    /// wallets, so the checks below are held to midwimble's own encoder.
    const ALICE: &str = "mw1q960ls6efz2kjcfp6t62djd408kjp2ff48ljdmtjumsgpdnk0xez4axmugrxsgp7tr6pdc7jg4fdgmung293gcyqql4pud3c0qat8usvfx0mmkexn3vqntw2rsmcxjpzkgysqv7q7jgnqzzg0qxwf60ncqkqvj9ses";
    const BOB: &str = "mw1q9sgyyn4zutnff5g89cz8pjeqwmc38xk367qqq86apatz4h6xcl3nqxzqf8l9r4etm25sgnvxu89v7zp9tt73t7lztegvjx5mtmt472j78njfuazanww53hmjpxlc607nhw7zzfyj9c868htve3svkn5fltse6kg2z";

    fn bech32m_encode(hrp: &str, data: &[u8]) -> String {
        const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
        let (mut values, mut acc, mut bits) = (Vec::new(), 0u32, 0u32);
        for &b in data {
            acc = (acc << 8) | b as u32;
            bits += 8;
            while bits >= 5 {
                bits -= 5;
                values.push(((acc >> bits) & 31) as u8);
            }
            acc &= (1 << bits) - 1;
        }
        if bits > 0 {
            values.push(((acc << (5 - bits)) & 31) as u8);
        }
        let mut checked: Vec<u8> = hrp.bytes().map(|c| c >> 5).collect();
        checked.push(0);
        checked.extend(hrp.bytes().map(|c| c & 31));
        checked.extend_from_slice(&values);
        checked.extend_from_slice(&[0; 6]);
        let pm = bech32_polymod(&checked) ^ 0x2bc8_30a3;
        for i in 0..6 {
            values.push(((pm >> (5 * (5 - i))) & 31) as u8);
        }
        let body: String = values.iter().map(|&v| CHARSET[v as usize] as char).collect();
        format!("{hrp}1{body}")
    }

    /// ALICE's payload, edited by `edit` and encoded again.
    fn alice_with(edit: impl FnOnce(&mut Vec<u8>)) -> String {
        let (_, mut data) = bech32m_decode(ALICE).unwrap();
        edit(&mut data);
        bech32m_encode("mw", &data)
    }

    /// A third valid address: ALICE with her scan and spend keys swapped.
    fn pool_mw() -> String {
        alice_with(|d| {
            let scan: Vec<u8> = d[1..33].to_vec();
            d.copy_within(33..65, 1);
            d[33..65].copy_from_slice(&scan);
        })
    }

    #[test]
    fn midwimble_addresses_are_checked_as_midwimble_decodes_them() {
        // Midwimble's own output round-trips, so decoding matches its format.
        assert_eq!(alice_with(|_| {}), ALICE);
        assert_eq!(parse_midwimble_address(ALICE).unwrap(), ALICE);
        assert_eq!(parse_midwimble_address(BOB).unwrap(), BOB);
        assert_eq!(parse_midwimble_address(&pool_mw()).unwrap(), pool_mw());
        // Canonical lowercase out, whatever case (and spacing) came in.
        assert_eq!(parse_midwimble_address(&format!(" {} ", ALICE.to_uppercase())).unwrap(), ALICE);

        let mut mixed = ALICE.to_string();
        mixed.replace_range(10..11, &ALICE[10..11].to_uppercase());
        assert!(parse_midwimble_address(&mixed).is_err(), "mixed case");
        let typo = format!("{}{}", &ALICE[..40], if &ALICE[40..41] == "q" { "p" } else { "q" }) + &ALICE[41..];
        assert!(parse_midwimble_address(&typo).is_err(), "one changed character fails the checksum");
        assert!(parse_midwimble_address(&ALICE[..ALICE.len() - 1]).is_err(), "truncated");
        let (_, data) = bech32m_decode(ALICE).unwrap();
        assert!(parse_midwimble_address(&bech32m_encode("mx", &data)).is_err(), "another prefix");
        assert!(parse_midwimble_address(&alice_with(|d| d[0] = 2)).is_err(), "another version");
        assert!(parse_midwimble_address(&alice_with(|d| { d.pop(); })).is_err(), "short payload");
        assert!(parse_midwimble_address(&alice_with(|d| d[1..33].fill(0))).is_err(), "identity scan key");
        assert!(parse_midwimble_address(&alice_with(|d| d[33..65].fill(0xff))).is_err(), "spend key off the curve");
        assert!(parse_midwimble_address("").is_err());
        assert!(parse_midwimble_address("deadbeef").is_err(), "a midstate-style address");
    }

    fn total(payouts: &[(String, u64)]) -> u128 {
        payouts.iter().map(|(_, w)| *w as u128).sum()
    }

    #[test]
    fn with_no_midwimble_score_the_pool_is_paid_everything() {
        let payouts = merge_payouts(1.0, &pool_mw(), &[], &[0; 32]);
        assert_eq!(payouts, vec![(pool_mw(), MW_PAYOUT_SCALE as u64)]);
    }

    #[test]
    fn larger_miners_are_paid_exactly_and_the_fee_goes_to_the_pool() {
        let scores = vec![(ALICE.to_string(), 60), (BOB.to_string(), 40)];
        let payouts = merge_payouts(1.0, &pool_mw(), &scores, &[9; 32]);
        assert_eq!(total(&payouts), MW_PAYOUT_SCALE, "every unit is paid to someone");
        let fee = MW_PAYOUT_SCALE / 100;
        let weight = |a: &str| payouts.iter().find(|(x, _)| x == a).map(|(_, w)| *w as u128).unwrap();
        assert_eq!(weight(ALICE), (MW_PAYOUT_SCALE - fee) * 60 / 100);
        assert_eq!(weight(BOB), (MW_PAYOUT_SCALE - fee) * 40 / 100);
        assert!(weight(&pool_mw()) >= fee && weight(&pool_mw()) < fee + 2);
    }

    #[test]
    fn the_pools_own_score_and_fee_share_one_output() {
        // Shares from miners who gave no midwimble address count for the pool.
        let scores = vec![(ALICE.to_string(), 50), (pool_mw(), 50)];
        let payouts = merge_payouts(2.0, &pool_mw(), &scores, &[1; 32]);
        assert_eq!(payouts.len(), 2);
        assert_eq!(total(&payouts), MW_PAYOUT_SCALE);
    }

    #[test]
    fn many_small_miners_fit_one_coinbase_and_are_paid_fairly_on_average() {
        // 40 miners, all under 1/31 of the score: they share 31 outputs.
        let scores: Vec<(String, u64)> = (0..40u64).map(|i| (format!("mw1miner{i:02}"), 1 + i % 5)).collect();
        let sum: u128 = scores.iter().map(|(_, s)| *s as u128).sum();
        let mut earned: HashMap<String, u128> = HashMap::new();
        let draws = 4000u32;
        for i in 0..draws {
            let seed = hash(&i.to_le_bytes());
            let payouts = merge_payouts(0.0, "mw1pool", &scores, &seed);
            assert!(payouts.len() <= MW_MAX_OUTPUTS);
            assert_eq!(payouts.len(), MW_MAX_OUTPUTS, "31 miners and the pool's rounding");
            assert_eq!(total(&payouts), MW_PAYOUT_SCALE);
            for (a, w) in payouts {
                *earned.entry(a).or_default() += w as u128;
            }
        }
        for (address, score) in &scores {
            let expected = MW_PAYOUT_SCALE * *score as u128 / sum;
            let average = earned.get(address).copied().unwrap_or(0) / draws as u128;
            let off = expected.abs_diff(average) as f64 / expected as f64;
            assert!(off < 0.08, "{address}: average {average}, expected {expected}");
        }
    }

    fn test_db() -> (tempfile::TempDir, Arc<Database>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("pool.redb")).unwrap();
        let txn = db.begin_write().unwrap();
        {
            txn.open_table(SHARES_TABLE).unwrap();
            txn.open_table(MW_SHARES_TABLE).unwrap();
        }
        txn.commit().unwrap();
        (dir, Arc::new(db))
    }

    fn sorted(mut v: Vec<(String, u64)>) -> Vec<(String, u64)> {
        v.sort();
        v
    }

    #[test]
    fn a_found_block_takes_its_scores_and_a_refused_one_puts_them_back() {
        let (_dir, db) = test_db();
        settle_midwimble_scores(&db, &[(ALICE.into(), 5), (BOB.into(), 2)], true).unwrap();
        let paid = vec![(ALICE.to_string(), 3), (BOB.to_string(), 2)];
        settle_midwimble_scores(&db, &paid, false).unwrap();
        assert_eq!(midwimble_scores(&db), vec![(ALICE.to_string(), 2)], "BOB's row is gone, not zero");
        settle_midwimble_scores(&db, &paid, true).unwrap();
        assert_eq!(sorted(midwimble_scores(&db)), sorted(vec![(ALICE.into(), 5), (BOB.into(), 2)]));
    }

    /// What a stand-in midwimble node has been asked, and how it answers.
    #[derive(Default)]
    struct Mock {
        jobs: Vec<Vec<(String, u64)>>,
        found: Vec<serde_json::Value>,
        refuse_miners: bool,
        refuse_blocks: bool,
    }
    type Shared = Arc<std::sync::Mutex<Mock>>;

    /// A stand-in midwimble node serving `/merge/job` and `/merge/found`.
    async fn stand_in_midwimble(pool: String, mock: Shared) -> String {
        use axum::http::StatusCode;
        async fn job(
            State((pool, mock)): State<(String, Shared)>,
            Json(body): Json<serde_json::Value>,
        ) -> (StatusCode, Json<serde_json::Value>) {
            let payouts: Vec<(String, u64)> = body["payouts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p["address"].as_str().unwrap().to_string(), p["weight"].as_u64().unwrap()))
                .collect();
            let mut m = mock.lock().unwrap();
            let refuse = m.refuse_miners && payouts.iter().any(|(a, _)| *a != pool);
            m.jobs.push(payouts);
            if refuse {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "invalid scan key" })));
            }
            (StatusCode::OK, Json(serde_json::json!({
                "commitment": "11".repeat(32), "target": "ff".repeat(32),
                "mining_hash": "22".repeat(32), "height": 7,
            })))
        }
        async fn found(
            State((_, mock)): State<(String, Shared)>,
            Json(body): Json<serde_json::Value>,
        ) -> (StatusCode, Json<serde_json::Value>) {
            let mut m = mock.lock().unwrap();
            m.found.push(body);
            if m.refuse_blocks {
                return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "stale tip" })));
            }
            (StatusCode::OK, Json(serde_json::json!({ "accepted": true, "height": 7, "merged_mined": true })))
        }
        let app = Router::new()
            .route("/merge/job", post(job))
            .route("/merge/found", post(found))
            .with_state((pool, mock));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn a_job_pays_the_miners_or_falls_back_to_the_pool() {
        let (_dir, db) = test_db();
        let pool = pool_mw();
        let mock: Shared = Default::default();
        let rpc = stand_in_midwimble(pool.clone(), mock.clone()).await;
        let client = reqwest::Client::new();
        settle_midwimble_scores(&db, &[(ALICE.into(), 3), (BOB.into(), 1)], true).unwrap();

        let (commitment, plan) = plan_merge(&db, &client, &rpc, &pool, 1.0, &[0; 32]).await.unwrap();
        assert_eq!(commitment, "11".repeat(32));
        assert_eq!(sorted(plan.paid.to_vec()), sorted(vec![(ALICE.into(), 3), (BOB.into(), 1)]));
        let asked: Vec<String> = mock.lock().unwrap().jobs[0].iter().map(|(a, _)| a.clone()).collect();
        assert!(asked.contains(&ALICE.to_string()) && asked.contains(&BOB.to_string()) && asked.contains(&pool));
        // The job remembers what its block pays, for the dashboard.
        assert_eq!(plan.payouts.to_vec(), mock.lock().unwrap().jobs[0]);
        assert_eq!((plan.height, plan.note.clone()), (Some(7), None));

        // The node refuses the miners' payouts: the job pays the pool alone,
        // as before per-miner payouts, and takes nobody's score.
        mock.lock().unwrap().refuse_miners = true;
        let (_, plan) = plan_merge(&db, &client, &rpc, &pool, 1.0, &[0; 32]).await.unwrap();
        assert!(plan.paid.is_empty());
        assert_eq!(mock.lock().unwrap().jobs.last().unwrap(), &vec![(pool.clone(), 1)]);
        assert_eq!(plan.payouts.to_vec(), vec![(pool.clone(), 1)]);
        assert!(plan.note.as_deref().is_some_and(|n| n.contains("invalid scan key")), "{:?}", plan.note);

        // No node at all: no merge-mining this job, and nothing breaks.
        let refused = plan_merge(&db, &client, "http://127.0.0.1:9", &pool, 1.0, &[0; 32]).await;
        assert!(refused.as_ref().is_err_and(|e| e.contains("unreachable")), "{:?}", refused.map(|r| r.0));
    }

    fn test_state(db: Arc<Database>, rpc: String, pool: String) -> Arc<PoolState> {
        test_state_with(db, Some(rpc), Some(pool))
    }

    fn test_state_with(db: Arc<Database>, rpc: Option<String>, pool: Option<String>) -> Arc<PoolState> {
        let (job_notifier, _) = broadcast::channel(4);
        Arc::new(PoolState {
            db,
            current_job: RwLock::new(None),
            job_notifier,
            pool_address: hex::encode([7u8; 32]),
            share_target: [0xff; 32],
            current_tree: RwLock::new(ShareMerkleTree::build(vec![])),
            valid_shares: RwLock::new(HashSet::new()),
            node_rpc_url: "http://127.0.0.1:9".into(),
            midwimble_rpc: rpc,
            midwimble_address: pool,
            merge_status: RwLock::new(MergeStatus::default()),
            pool_fee_percent: 1.0,
            current_block_reward: AtomicU64::new(0),
            current_height: AtomicU64::new(0),
            share_stats: RwLock::new(HashMap::new()),
            worker_stats: RwLock::new(HashMap::new()),
            force_new_job: AtomicBool::new(false),
            verify_permits: Arc::new(Semaphore::new(2)),
            http_throttle: RwLock::new(HashMap::new()),
        })
    }

    /// Serves a job whose every share is a midwimble block and none a
    /// midstate one, paying `paid`.
    async fn serve_winning_job(state: &Arc<PoolState>, job_id: u64, paid: Vec<(String, u64)>) -> MergeCommit {
        let merge = MergeCommit {
            mining_hash: "22".repeat(32),
            target: [0xff; 32],
            commit_index: 1,
            payouts: Arc::new(paid.clone()),
            paid: Arc::new(paid),
            claimed: Arc::new(AtomicBool::new(false)),
            height: Some(7),
            note: None,
        };
        *state.current_job.write().await = Some(Job {
            job_id,
            mining_hash: [3; 32],
            share_target: [0xff; 32],
            network_target: [0; 32],
            batch_template: serde_json::json!({ "stand_in": true }),
            height: 8,
            committed_scores: Arc::new(Vec::new()),
            midwimble: Some(merge.clone()),
        });
        merge
    }

    async fn settled(mock: &Shared, submissions: usize) {
        for _ in 0..200 {
            if mock.lock().unwrap().found.len() >= submissions {
                tokio::time::sleep(Duration::from_millis(100)).await;
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("the midwimble block was never submitted");
    }

    #[tokio::test]
    async fn a_share_that_wins_midwimble_pays_by_score_once() {
        let (_dir, db) = test_db();
        let pool = pool_mw();
        let mock: Shared = Default::default();
        let rpc = stand_in_midwimble(pool.clone(), mock.clone()).await;
        let state = test_state(db.clone(), rpc, pool.clone());
        settle_midwimble_scores(&db, &[(ALICE.into(), 3), (BOB.into(), 1)], true).unwrap();
        let merge = serve_winning_job(&state, 1, midwimble_scores(&db)).await;

        // ALICE's share wins: it earns its point, then the block takes the
        // scores it pays. BOB's row empties and goes.
        let outcome = process_share(&state, [1; 32], "rig", Some(ALICE), 1, 11).await.unwrap();
        assert!(matches!(outcome, ShareOutcome::Accepted { is_block: false }));
        assert_eq!(midwimble_scores(&db), vec![(ALICE.to_string(), 1)]);
        settled(&mock, 1).await;
        assert!(state.force_new_job.load(Ordering::SeqCst), "a spent commitment asks for a new job");
        assert_eq!(mock.lock().unwrap().found[0]["mining_hash"], "22".repeat(32));
        assert_eq!(mock.lock().unwrap().found[0]["commit_index"], 1);

        // Another winning share on the same job is credited but not submitted:
        // it would be a rival block taking the same scores again.
        let outcome = process_share(&state, [2; 32], "rig", None, 1, 12).await.unwrap();
        assert!(matches!(outcome, ShareOutcome::Accepted { .. }));
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(mock.lock().unwrap().found.len(), 1);
        assert!(merge.claimed.load(Ordering::SeqCst));
        assert_eq!(
            sorted(midwimble_scores(&db)),
            sorted(vec![(ALICE.into(), 1), (pool.clone(), 1)]),
            "a share with no midwimble address counts for the pool"
        );
    }

    #[tokio::test]
    async fn a_refused_midwimble_block_gives_the_scores_back() {
        let (_dir, db) = test_db();
        let pool = pool_mw();
        let mock: Shared = Default::default();
        mock.lock().unwrap().refuse_blocks = true;
        let rpc = stand_in_midwimble(pool.clone(), mock.clone()).await;
        let state = test_state(db.clone(), rpc, pool.clone());
        settle_midwimble_scores(&db, &[(ALICE.into(), 3), (BOB.into(), 1)], true).unwrap();
        let merge = serve_winning_job(&state, 1, midwimble_scores(&db)).await;

        process_share(&state, [1; 32], "rig", Some(BOB), 1, 21).await.unwrap();
        settled(&mock, 1).await;
        assert_eq!(sorted(midwimble_scores(&db)), sorted(vec![(ALICE.into(), 3), (BOB.into(), 2)]));
        assert!(!merge.claimed.load(Ordering::SeqCst), "the job may still win with another share");
        assert!(!state.force_new_job.load(Ordering::SeqCst));
    }

    // ── The dashboard's midwimble section ───────────────────────────────

    #[tokio::test]
    async fn the_dashboard_lists_found_midwimble_blocks_and_the_score_table() {
        let (_dir, db) = test_db();
        let pool = pool_mw();
        let mock: Shared = Default::default();
        let rpc = stand_in_midwimble(pool.clone(), mock.clone()).await;
        let state = test_state(db.clone(), rpc, pool.clone());
        settle_midwimble_scores(&db, &[(ALICE.into(), 3), (BOB.into(), 1)], true).unwrap();

        let before = midwimble_stats(&state).await;
        assert_eq!(before["pool_address"], pool.as_str());
        assert_eq!(before["active"], false, "no job built yet");
        assert_eq!(before["blocks_found"], 0);
        assert_eq!(before["total_score"], 4);
        assert_eq!(before["scores"][0]["address"], ALICE, "highest score first");

        serve_winning_job(&state, 1, midwimble_scores(&db)).await;
        process_share(&state, [1; 32], "rig", Some(ALICE), 1, 11).await.unwrap();
        settled(&mock, 1).await;

        let after = midwimble_stats(&state).await;
        assert_eq!(after["blocks_found"], 1);
        let block = &after["recent_blocks"][0];
        assert_eq!(block["height"], 7);
        assert!(block["timestamp"].as_u64().unwrap() > 0);
        let paid: Vec<(String, u64)> = block["payouts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p["address"].as_str().unwrap().to_string(), p["weight"].as_u64().unwrap()))
            .collect();
        assert_eq!(sorted(paid), sorted(vec![(ALICE.into(), 3), (BOB.into(), 1)]));
        // The block took the scores it paid; ALICE's winning share is still hers.
        assert_eq!(after["total_score"], 1);
        assert_eq!(after["scores"][0]["address"], ALICE);
    }

    #[tokio::test]
    async fn a_refused_midwimble_block_is_not_listed() {
        let (_dir, db) = test_db();
        let pool = pool_mw();
        let mock: Shared = Default::default();
        mock.lock().unwrap().refuse_blocks = true;
        let rpc = stand_in_midwimble(pool.clone(), mock.clone()).await;
        let state = test_state(db.clone(), rpc, pool.clone());
        settle_midwimble_scores(&db, &[(ALICE.into(), 3)], true).unwrap();
        serve_winning_job(&state, 1, midwimble_scores(&db)).await;
        process_share(&state, [1; 32], "rig", Some(ALICE), 1, 11).await.unwrap();
        settled(&mock, 1).await;
        let stats = midwimble_stats(&state).await;
        assert_eq!(stats["blocks_found"], 0);
        assert_eq!(stats["recent_blocks"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn a_pool_that_does_not_merge_mine_has_no_midwimble_section() {
        let (_dir, db) = test_db();
        let state = test_state_with(db, None, None);
        assert_eq!(midwimble_stats(&state).await, serde_json::Value::Null);
    }

    #[test]
    fn merge_status_says_why_a_job_does_not_merge_mine() {
        let refused = MergeStatus::of_job(None, Some("bonded mining is required from height 1"), 5);
        assert_eq!(
            refused,
            MergeStatus { active: false, note: Some("bonded mining is required from height 1".into()), height: None, updated: 5 }
        );
        let no_room = MergeStatus::of_job(None, None, 6);
        assert!(!no_room.active && no_room.note.is_some());

        let merge = MergeCommit {
            mining_hash: "22".repeat(32),
            target: [0xff; 32],
            commit_index: 1,
            paid: Arc::new(Vec::new()),
            claimed: Arc::new(AtomicBool::new(false)),
            height: Some(9),
            payouts: Arc::new(vec![(pool_mw(), 1)]),
            note: Some("the node refused the miners' payouts".into()),
        };
        let live = MergeStatus::of_job(Some(&merge), None, 7);
        assert_eq!(
            live,
            MergeStatus { active: true, note: merge.note.clone(), height: Some(9), updated: 7 }
        );
    }

    // ── split_with_floor ────────────────────────────────────────────────

    fn miner(i: u64) -> [u8; 32] {
        let mut a = [0u8; 32];
        a[..8].copy_from_slice(&i.to_be_bytes());
        a[31] = 0xa5;
        a
    }

    /// Checks the split's guarantees and returns the worst miner's distance
    /// from its exact share, as a fraction of the block.
    fn check_split(d: u64, shares: &[([u8; 32], u64)]) -> f64 {
        let coins = split_with_floor(d, shares).expect("a split");
        assert_eq!(coins.iter().map(|(_, c)| *c as u128).sum::<u128>(), d as u128, "pays the block exactly");
        assert!(coins.iter().all(|(_, c)| c.is_power_of_two()), "power-of-2 coins only");
        assert_eq!(split_with_floor(d, shares), Some(coins.clone()), "the same split every time");
        let total: f64 = shares.iter().map(|(_, s)| *s as f64).sum();
        let mut worst = 0f64;
        for (a, s) in shares {
            let paid: u64 = coins.iter().filter(|(x, _)| x == a).map(|(_, c)| *c).sum();
            if *s == 0 {
                assert_eq!(paid, 0, "no score, no coin");
                continue;
            }
            assert!(paid > 0, "a miner with score {s} got no coin");
            worst = worst.max((paid as f64 - d as f64 * *s as f64 / total).abs() / d as f64);
        }
        assert!(coins.iter().all(|(a, _)| shares.iter().any(|(x, s)| x == a && *s > 0)), "only miners are paid");
        worst
    }

    #[test]
    fn a_small_miner_next_to_large_ones_is_still_paid() {
        // The greedy split paid these 17 coins to the two large miners, and the
        // third miner's audit disconnected it for being left out.
        let d = 1_063_009_584;
        let shares = [(miner(1), 5_000), (miner(2), 300), (miner(3), 2)];
        assert!(check_split(d, &shares) < 1.0 / 256.0);
    }

    #[test]
    fn split_holds_its_guarantees_on_random_pools() {
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for round in 0..2_000u64 {
            let n = 1 + (next() % 60) as usize;
            let d = 1_063_009_584 + next() % 50_000;
            let shares: Vec<_> = (0..n as u64)
                .map(|i| {
                    let score = match next() % 5 {
                        0 => 0,
                        1 => 1 + next() % 3,
                        2 => 1 + next() % 100,
                        3 => 1 + next() % 10_000,
                        _ => 1 + next() % 5_000_000,
                    };
                    (miner(round * 100 + i), score)
                })
                .collect();
            if shares.iter().all(|(_, s)| *s == 0) {
                assert!(split_with_floor(d, &shares).is_none());
                continue;
            }
            // Within 1/256 of the block of each exact share, give or take a
            // unit per miner of rounding.
            let worst = check_split(d, &shares) * d as f64;
            let bound = (d >> SPLIT_GRANULARITY_SHIFT) as f64 + n as f64 + 1.0;
            assert!(worst <= bound, "round {round}: {worst} units off, bound {bound}");
            let outputs = split_with_floor(d, &shares).unwrap().len();
            assert!(outputs <= 3 * n + 64, "round {round}: {outputs} outputs for {n} miners");
        }
    }

    #[test]
    fn shares_that_round_to_nothing_still_get_a_coin() {
        let mut shares = vec![(miner(0), 10_000_000)];
        shares.extend((1..=10).map(|i| (miner(i), 1)));
        check_split(1_000, &shares);
        check_split(11, &shares);
    }

    #[test]
    fn split_declines_only_without_units_to_pay() {
        let shares: Vec<_> = (0..10).map(|i| (miner(i), 1)).collect();
        assert!(split_with_floor(9, &shares).is_none());
        assert!(split_with_floor(0, &shares).is_none());
        assert!(split_with_floor(1_000, &[]).is_none());
        assert!(split_with_floor(1_000, &[(miner(1), 0)]).is_none());
        let coins = split_with_floor(1_063_009_584, &[(miner(1), 7)]).unwrap();
        assert!(coins.iter().all(|(a, _)| *a == miner(1)));
    }
}
