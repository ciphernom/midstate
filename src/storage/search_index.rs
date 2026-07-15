//! Search acceleration: a block-level hash index and per-height filter metadata.
//!
//! # The problem
//!
//! `search()` walked 5,000 heights and fully deserialised **every non-empty
//! batch** to compare a handful of 32-byte fields. The compact filter looked
//! like the fix and isn't:
//!
//!   * The filter covers commitments, coin_ids and addresses — see
//!     [`CompactFilter::items_in`]. It does **not** cover `block_hash`,
//!     `prev_midstate`, `state_root` or reveal `salt`, all of which `search()`
//!     matches. A filter miss therefore cannot skip a block without silently
//!     losing those result types.
//!   * `match_any` needs the element count `n`, which was only obtainable by
//!     loading the batch — the very cost we were trying to avoid.
//!
//! # The fix
//!
//! Two tables, both written on `save()` and torn down on `truncate()`/
//! `prune_tail()`:
//!
//! * [`FILTER_META_TABLE`]: `height → block_hash ‖ n`. 40 bytes. Supplies
//!   `match_any`'s two missing arguments, so a filter miss can skip the batch
//!   load entirely. This is what makes the filter finally worth consulting.
//! * [`SEARCH_INDEX_TABLE`]: `hash → [heights]`, covering exactly the fields the
//!   filter does not. Turns block-level lookups into O(1) over ALL history,
//!   rather than a linear walk of a 5,000-block window.
//!
//! Together: batch loads drop from "every non-empty block in the window" to
//! "real hits plus the filter's false positives".
//!
//! # Why not just widen the filter?
//!
//! Adding block hashes and salts to `CompactFilter::build` would invalidate
//! every filter on disk and, worse, every light client's `match_any` — the
//! element set is a wire contract. A separate index costs disk instead.
//!
//! # Deliberately not indexed
//!
//! coin_ids and addresses. There are ~50 per block; indexing them across 200k
//! blocks is ~10M entries for a lookup the filter already answers at 1/784 false
//! positive rate. The filter + meta path handles those.

use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::sync::Arc;

use crate::core::types::Batch;
use crate::core::Transaction;   // re-exported from core::types via `pub use types::*`

/// `height → block_hash(32) ‖ element_count(8, LE)`.
///
/// The two arguments `filter::match_any` needs and the filter file doesn't
/// carry. Without this, consulting the filter requires loading the batch, which
/// defeats the point.
pub const FILTER_META_TABLE: TableDefinition<u64, &[u8]> =
    TableDefinition::new("filter_meta");

/// `32-byte hash → packed u64 LE heights`.
///
/// Covers ONLY what the compact filter does not: block hashes, parent midstates,
/// state roots and reveal salts. A packed value rather than a multimap table so
/// this depends on nothing beyond the `TableDefinition` API the rest of the
/// store already uses.
pub const SEARCH_INDEX_TABLE: TableDefinition<&[u8; 32], &[u8]> =
    TableDefinition::new("search_index");

/// `key → u64`. Holds `search_index_height`: the highest height indexed, so a
/// backfill interrupted at block 140,000 of 200,000 resumes there.
pub const INDEX_META_TABLE: TableDefinition<&str, u64> =
    TableDefinition::new("index_meta");

/// Progress key. Stores highest_indexed + 1, so 0 means "nothing indexed" and
/// there is no ambiguity with genesis.
pub const PROGRESS_KEY: &str = "search_index_next_height";

/// Cap on heights recorded per hash.
///
/// Block hashes and state roots are unique; a salt colliding across blocks is
/// astronomically unlikely. This exists so a hostile or degenerate chain cannot
/// grow one value without bound — 64 hits is already far past useful.
const MAX_HEIGHTS_PER_HASH: usize = 64;

/// Every hash in a batch that `search()` matches but the compact filter does not
/// cover.
///
/// The counterpart of [`CompactFilter::items_in`], and deliberately shaped like
/// it: one function, called by both the writer and the backfill, so the index
/// cannot drift from what search expects to find in it. (Hand-copying that
/// function into a handler is a bug this codebase has already shipped once.)
pub fn block_level_items(batch: &Batch) -> Vec<[u8; 32]> {
    let mut out = Vec::with_capacity(4 + batch.transactions.len());
    out.push(batch.extension.final_hash);
    out.push(batch.prev_midstate);
    out.push(batch.state_root);
    for tx in &batch.transactions {
        match tx {
            Transaction::Reveal { salt, .. } | Transaction::Consolidate { salt, .. } => {
                out.push(*salt);
            }
            Transaction::Commit { .. } => {}   // commitments ARE in the filter
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Pack `block_hash ‖ n` for [`FILTER_META_TABLE`].
pub fn encode_meta(block_hash: &[u8; 32], n: u64) -> [u8; 40] {
    let mut buf = [0u8; 40];
    buf[..32].copy_from_slice(block_hash);
    buf[32..].copy_from_slice(&n.to_le_bytes());
    buf
}

/// Unpack it. Returns `None` on anything malformed rather than panicking: a
/// truncated value means "no metadata", and the caller falls back to loading the
/// batch — slow, but correct.
pub fn decode_meta(raw: &[u8]) -> Option<([u8; 32], u64)> {
    if raw.len() != 40 {
        return None;
    }
    let mut h = [0u8; 32];
    h.copy_from_slice(&raw[..32]);
    let n = u64::from_le_bytes(raw[32..].try_into().ok()?);
    Some((h, n))
}

fn unpack_heights(raw: &[u8]) -> Vec<u64> {
    raw.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn pack_heights(heights: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(heights.len() * 8);
    for h in heights {
        out.extend_from_slice(&h.to_le_bytes());
    }
    out
}

/// Index one batch inside an existing write transaction.
///
/// Takes the transaction rather than opening its own so the caller can make the
/// index update atomic with the batch write. A batch on disk without its index
/// entries is a silently wrong search result.
pub fn index_batch_in_txn(
    txn: &redb::WriteTransaction,
    height: u64,
    batch: &Batch,
) -> Result<()> {
    {
        let mut meta = txn.open_table(FILTER_META_TABLE)?;
        let n = crate::core::filter::CompactFilter::items_in(batch).len() as u64;
        let packed = encode_meta(&batch.extension.final_hash, n);
        meta.insert(height, packed.as_slice())?;
    }
    {
        let mut idx = txn.open_table(SEARCH_INDEX_TABLE)?;
        for item in block_level_items(batch) {
            let mut heights = match idx.get(&item)? {
                Some(g) => unpack_heights(g.value()),
                None => Vec::new(),
            };
            if heights.contains(&height) {
                continue;                       // idempotent: re-indexing must not duplicate
            }
            if heights.len() >= MAX_HEIGHTS_PER_HASH {
                continue;
            }
            heights.push(height);
            heights.sort_unstable();
            idx.insert(&item, pack_heights(&heights).as_slice())?;
        }
    }
    Ok(())
}

/// Remove one height's index entries inside an existing write transaction.
///
/// Needs the `batch` because the index is hash→heights with no reverse map:
/// the only way to know what to remove is to recompute it. Callers MUST read the
/// batch before deleting it — hence the explicit argument rather than a lookup
/// here, which would race with the deletion in the same transaction.
pub fn unindex_batch_in_txn(
    txn: &redb::WriteTransaction,
    height: u64,
    batch: &Batch,
) -> Result<()> {
    {
        let mut meta = txn.open_table(FILTER_META_TABLE)?;
        meta.remove(height)?;
    }
    {
        let mut idx = txn.open_table(SEARCH_INDEX_TABLE)?;
        for item in block_level_items(batch) {
            let remaining: Vec<u64> = match idx.get(&item)? {
                Some(g) => unpack_heights(g.value()).into_iter().filter(|h| *h != height).collect(),
                None => continue,
            };
            if remaining.is_empty() {
                idx.remove(&item)?;
            } else {
                idx.insert(&item, pack_heights(&remaining).as_slice())?;
            }
        }
    }
    Ok(())
}

/// Heights where this hash appears as a block hash, parent midstate, state root
/// or reveal salt. O(1), and covers all history — not just a recent window.
pub fn lookup(db: &Arc<Database>, hash: &[u8; 32]) -> Result<Vec<u64>> {
    let read_txn = db.begin_read()?;
    let table = match read_txn.open_table(SEARCH_INDEX_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(Vec::new()),        // table absent → not yet backfilled
    };
    // Bind the owned Vec before the guard drops: redb's AccessGuard borrows the
    // table, which borrows the transaction, so `Ok(match ...)` would keep a
    // temporary alive past both.
    let heights = match table.get(hash)? {
        Some(g) => unpack_heights(g.value()),
        None => Vec::new(),
    };
    Ok(heights)
}

/// `(block_hash, n)` for a height, or `None` if unindexed.
///
/// `None` must mean "fall back to loading the batch", never "no match" — an
/// unbackfilled range would otherwise silently return empty results.
pub fn filter_meta(db: &Arc<Database>, height: u64) -> Result<Option<([u8; 32], u64)>> {
    let read_txn = db.begin_read()?;
    let table = match read_txn.open_table(FILTER_META_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let meta = match table.get(height)? {
        Some(g) => decode_meta(g.value()),
        None => None,
    };
    Ok(meta)
}


/// `(block_hash, n)` for every indexed height in `start..=end`, in one pass.
///
/// The point-lookup version opens a fresh read transaction and re-opens the
/// table for EVERY height — ~17 µs each, so a 5000-block window burns ~85 ms
/// before it has looked at a single filter. One transaction and one ordered scan
/// is the same data for a fraction of the cost, and search walks a contiguous
/// range by definition.
///
/// Heights with no metadata are simply absent from the map; the caller must
/// treat absence as "unindexed → load the batch", never as "no match".
pub fn filter_meta_range(
    db: &Arc<Database>,
    start: u64,
    end: u64,
) -> Result<std::collections::BTreeMap<u64, ([u8; 32], u64)>> {
    let mut out = std::collections::BTreeMap::new();
    if end < start {
        return Ok(out);
    }
    let read_txn = db.begin_read()?;
    let table = match read_txn.open_table(FILTER_META_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(out),               // not yet backfilled
    };
    for entry in table.range(start..=end)? {
        let (k, v) = entry?;
        if let Some(meta) = decode_meta(v.value()) {
            out.insert(k.value(), meta);
        }
    }
    Ok(out)
}

/// Next height the backfill should process. 0 = nothing indexed yet.
pub fn progress(db: &Arc<Database>) -> Result<u64> {
    let read_txn = db.begin_read()?;
    let table = match read_txn.open_table(INDEX_META_TABLE) {
        Ok(t) => t,
        Err(_) => return Ok(0),
    };
    let next = match table.get(PROGRESS_KEY)? {
        Some(g) => g.value(),
        None => 0,
    };
    Ok(next)
}

fn set_progress_in_txn(txn: &redb::WriteTransaction, next: u64) -> Result<()> {
    let mut table = txn.open_table(INDEX_META_TABLE)?;
    table.insert(PROGRESS_KEY, next)?;
    Ok(())
}

/// Rewind progress after a reorg, so truncated heights are re-indexed on the
/// next pass.
pub fn rewind_progress(db: &Arc<Database>, new_next: u64) -> Result<()> {
    let cur = progress(db)?;
    if new_next >= cur {
        return Ok(());
    }
    let txn = db.begin_write()?;
    set_progress_in_txn(&txn, new_next)?;
    txn.commit()?;
    Ok(())
}

/// How many heights per write transaction during the backfill.
///
/// One commit per block across 200k blocks is unusably slow (an fsync each);
/// one commit for the whole run means an interruption loses everything and the
/// transaction grows without bound. 1000 is ~a second of work to redo.
const BACKFILL_CHUNK: u64 = 1000;

/// Populate the index from `progress()` to `tip`.
///
/// Resumable and idempotent: progress advances in the SAME transaction as the
/// data, so a kill -9 mid-run resumes at the last committed chunk and re-indexing
/// an already-indexed height is a no-op.
///
/// Existing nodes need a full pass from genesis on first run — 200k blocks of
/// deserialisation, minutes, not hours. `on_progress` is called per chunk so the
/// operator can see it is alive rather than assume it has hung.
pub fn backfill<F>(
    db: &Arc<Database>,
    load_batch: impl Fn(u64) -> Result<Option<Batch>>,
    tip: u64,
    mut on_progress: F,
) -> Result<u64>
where
    F: FnMut(u64, u64),
{
    let start = progress(db)?;
    if start > tip {
        return Ok(0);                            // already current
    }

    let mut indexed = 0u64;
    let mut height = start;

    while height <= tip {
        let chunk_end = (height + BACKFILL_CHUNK - 1).min(tip);
        let txn = db.begin_write()?;
        {
            for h in height..=chunk_end {
                match load_batch(h) {
                    Ok(Some(batch)) => {
                        index_batch_in_txn(&txn, h, &batch)?;
                        indexed += 1;
                    }
                    // A pruned or missing height is normal (prune_tail removes
                    // old history). Skip it and keep going: aborting here would
                    // leave every later block unindexed forever.
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!("search index: skipping height {} ({})", h, e);
                    }
                }
            }
            set_progress_in_txn(&txn, chunk_end + 1)?;
        }
        txn.commit()?;

        on_progress(chunk_end, tip);
        height = chunk_end + 1;
    }

    Ok(indexed)
}
