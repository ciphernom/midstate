//! Header checkpoints: what lets a new node skip re-checking proof of work
//! on the part of the chain a release already vouches for.
//!
//! # Reasoning
//! Before the Midstate Walk (height 371,500), proof of work was a chain of a
//! million sequential BLAKE3 hashes per header. A node syncing from genesis
//! re-ran every one of them: about an hour of all-cores work on a desktop,
//! the better part of a day's CPU on a Raspberry Pi 5. Those headers can't
//! change, so a release ships a digest of them instead, one per segment of
//! `SPAN` heights, and a header in a segment whose digest matches skips the
//! PoW check.
//!
//! A digest covers every field of every header in its segment
//! (`BatchHeader::to_record`). Unlike a checkpoint on a PoW hash alone,
//! which says nothing about the fields beside it, a match means the peer
//! sent exactly the headers the release was built from, field for field.
//! Checking one costs a cheap BLAKE3 over its ~108 KB.
//!
//! This only ever skips work. A header that doesn't match is checked in
//! full as before, so no chain is accepted or refused because of the table,
//! and the rest of header validation (linkage, difficulty, timestamps) runs
//! on every header either way. An empty table, the default until one is
//! generated, changes nothing at all.
//!
//! The table is built from a synced node's own database with
//! `midstate checkpoints`, and anyone with a synced node can rebuild it and
//! compare the digest the command prints.

use crate::core::types::BatchHeader;
use std::borrow::Cow;

/// Heights per segment. 371,500 = 743 × 500, so the table can end exactly
/// at the Walk activation, where the costly proof of work stops.
pub const SPAN: u64 = 500;

/// One 32-byte digest per segment from height 0: segment `k` holds heights
/// `k*SPAN .. (k+1)*SPAN`. Written by `midstate checkpoints`.
static TABLE: &[u8] = include_bytes!("checkpoints.bin");

const DIGEST_CONTEXT: &str = "midstate 2026-11-16 header checkpoint v1";

/// The digest of segment `k`: its index and every header in it, in order.
pub fn segment_digest<'a>(k: u64, headers: impl IntoIterator<Item = &'a BatchHeader>) -> [u8; 32] {
    let mut hasher = segment_hasher(k);
    for h in headers {
        hasher.update(&h.to_record());
    }
    *hasher.finalize().as_bytes()
}

fn segment_hasher(k: u64) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new_derive_key(DIGEST_CONTEXT);
    hasher.update(&k.to_le_bytes());
    hasher
}

#[derive(Clone, Debug)]
pub struct Checkpoints {
    span: u64,
    table: Cow<'static, [u8]>,
}

impl Default for Checkpoints {
    /// The table this release was built with.
    fn default() -> Self {
        Self { span: SPAN, table: Cow::Borrowed(TABLE) }
    }
}

impl Checkpoints {
    /// A table of `span`-height segments, e.g. from `build_table`.
    pub fn new(span: u64, table: Vec<u8>) -> Self {
        Self { span: span.max(1), table: Cow::Owned(table) }
    }

    pub fn segments(&self) -> u64 {
        (self.table.len() / 32) as u64
    }

    /// Heights below this are covered by the table.
    pub fn covered_end(&self) -> u64 {
        self.segments() * self.span
    }

    pub fn span(&self) -> u64 {
        self.span
    }

    fn expected(&self, k: u64) -> Option<&[u8]> {
        let at = usize::try_from(k).ok()?.checked_mul(32)?;
        self.table.get(at..at + 32)
    }

    /// Which headers of `chunk` the table vouches for.
    ///
    /// `chunk` is a run of consecutive heights whose PoW is unchecked.
    /// `before` holds the same chain's headers from the start of the chunk's
    /// first segment up to the chunk (empty when the chunk starts a segment),
    /// so that segment can be checked too. Only the chunk's own headers are
    /// ever vouched for, and only in whole segments that match. O(n) cheap
    /// hashing, O(1) memory besides the answer.
    pub fn vouched(&self, before: &[BatchHeader], chunk: &[BatchHeader]) -> Vec<bool> {
        let mut out = vec![false; chunk.len()];
        let covered = self.covered_end();
        let Some(first) = chunk.first() else { return out };
        if first.height >= covered {
            return out;
        }
        let mut k = first.height / self.span;
        let lead = first.height - k * self.span;
        let lead_ok = before.len() as u64 == lead
            && before.iter().enumerate().all(|(i, h)| h.height == k * self.span + i as u64);

        let mut hasher = segment_hasher(k);
        if lead_ok {
            for h in before {
                hasher.update(&h.to_record());
            }
        }
        let mut usable = lead_ok;
        let mut segment_from = 0;
        for (i, h) in chunk.iter().enumerate() {
            if h.height != first.height + i as u64 {
                break;
            }
            hasher.update(&h.to_record());
            if (h.height + 1) % self.span == 0 {
                let digest = *hasher.finalize().as_bytes();
                if usable && self.expected(k) == Some(&digest[..]) {
                    out[segment_from..=i].iter_mut().for_each(|v| *v = true);
                }
                k += 1;
                if k * self.span >= covered {
                    break;
                }
                hasher = segment_hasher(k);
                usable = true;
                segment_from = i + 1;
            }
        }
        out
    }

    /// How many headers to ask for from `cursor`, toward `end`, at most
    /// `max`. Below `covered_end` a request stops where a segment does, so
    /// each chunk can be checked against whole segments; past it, nothing
    /// changes.
    pub fn request_count(&self, cursor: u64, end: u64, max: u64) -> u64 {
        let mut stop = cursor.saturating_add(max).min(end);
        let covered = self.covered_end();
        if cursor < covered {
            let aligned = stop.min(covered) / self.span * self.span;
            if aligned > cursor {
                stop = aligned;
            }
        }
        stop.saturating_sub(cursor)
    }
}

/// The checkpoint table for heights `0..to`, rounded down to whole
/// segments, from a node's own database. Fails on a gap or a header that
/// doesn't follow the one before: a table is forever.
pub fn build_table(storage: &crate::storage::Storage, to: u64, span: u64) -> anyhow::Result<Vec<u8>> {
    let span = span.max(1);
    let segments = to / span;
    let mut table = Vec::with_capacity(segments as usize * 32);
    let mut prev: Option<BatchHeader> = None;
    for k in 0..segments {
        let (from, until) = (k * span, (k + 1) * span);
        let headers = storage.batches.load_headers(from, until)?;
        anyhow::ensure!(headers.len() as u64 == span, "headers {}..{} aren't all stored", from, until);
        for (i, h) in headers.iter().enumerate() {
            anyhow::ensure!(h.height == from + i as u64, "header at {} says height {}", from + i as u64, h.height);
            if let Some(p) = &prev {
                anyhow::ensure!(p.is_followed_by(h), "header {} doesn't follow header {}", h.height, p.height);
            }
            prev = Some(h.clone());
        }
        table.extend_from_slice(&segment_digest(k, &headers));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Extension;

    /// `n` linked headers from height 0. Different seeds give different chains.
    fn chain(n: u64, seed: u8) -> Vec<BatchHeader> {
        let mut prev_header_hash = [0u8; 32];
        let mut prev_midstate = [0u8; 32];
        (0..n)
            .map(|height| {
                let tag = [&height.to_le_bytes()[..], &[seed][..]].concat();
                let final_hash = *blake3::hash(&[&tag[..], b"h"].concat()).as_bytes();
                let post_tx_midstate = *blake3::hash(&[&tag[..], b"m"].concat()).as_bytes();
                let h = BatchHeader {
                    height,
                    prev_midstate,
                    post_tx_midstate,
                    extension: Extension { nonce: height, final_hash },
                    timestamp: 1_700_000_000 + height,
                    target: [0xff; 32],
                    state_root: [seed; 32],
                    prev_header_hash,
                };
                prev_header_hash = final_hash;
                prev_midstate = post_tx_midstate;
                h
            })
            .collect()
    }

    fn table_for(headers: &[BatchHeader], span: u64) -> Checkpoints {
        let table = headers
            .chunks_exact(span as usize)
            .enumerate()
            .flat_map(|(k, seg)| segment_digest(k as u64, seg))
            .collect();
        Checkpoints::new(span, table)
    }

    #[test]
    fn matching_segments_are_vouched_for() {
        let all = chain(40, 1);
        let cp = table_for(&all, 10);
        assert_eq!(cp.covered_end(), 40);
        assert!(cp.vouched(&[], &all[..20]).iter().all(|v| *v));
        assert!(cp.vouched(&[], &all[20..]).iter().all(|v| *v), "a chunk starting a segment needs nothing before it");
    }

    #[test]
    fn one_changed_field_unvouches_only_its_segment() {
        let all = chain(40, 1);
        let cp = table_for(&all, 10);
        let mut tampered = all.clone();
        tampered[15].timestamp += 1; // a field the PoW hash would also catch
        let v = cp.vouched(&[], &tampered);
        assert!(v[..10].iter().all(|v| *v));
        assert!(v[10..20].iter().all(|v| !*v), "segment 1 doesn't match");
        assert!(v[20..].iter().all(|v| *v));

        let mut tampered = all.clone();
        tampered[25].prev_midstate[0] ^= 1; // one the PoW hash leaves out
        assert!(cp.vouched(&[], &tampered)[20..30].iter().all(|v| !*v));
    }

    #[test]
    fn a_chunk_starting_mid_segment_uses_the_headers_before_it() {
        let all = chain(40, 1);
        let cp = table_for(&all, 10);
        let v = cp.vouched(&all[10..13], &all[13..30]);
        assert!(v.iter().all(|v| *v), "the rest of segment 1, and segment 2");

        // Without them, or with another chain's, segment 1 can't be checked.
        let v = cp.vouched(&[], &all[13..30]);
        assert!(v[..7].iter().all(|v| !*v));
        assert!(v[7..].iter().all(|v| *v));
        let other = chain(40, 2);
        assert!(cp.vouched(&other[10..13], &all[13..30])[..7].iter().all(|v| !*v));
    }

    #[test]
    fn nothing_past_the_table_or_short_of_a_whole_segment_is_vouched_for() {
        let all = chain(40, 1);
        let cp = table_for(&all[..20], 10);
        let v = cp.vouched(&[], &all);
        assert!(v[..20].iter().all(|v| *v));
        assert!(v[20..].iter().all(|v| !*v));
        // A chunk ending part-way into a segment.
        assert!(cp.vouched(&[], &all[..15])[10..].iter().all(|v| !*v));
    }

    #[test]
    fn an_empty_table_changes_nothing() {
        let all = chain(30, 1);
        let cp = Checkpoints::new(SPAN, Vec::new());
        assert!(cp.vouched(&[], &all).iter().all(|v| !*v));
        assert_eq!(cp.request_count(100, 50_000, 5_000), 5_000);
        assert_eq!(cp.request_count(100, 2_000, 5_000), 1_900);
    }

    #[test]
    fn requests_stop_where_segments_do_inside_the_table() {
        let cp = Checkpoints::new(500, vec![0u8; 32 * 743]); // up to 371,500
        assert_eq!(cp.covered_end(), 371_500);
        // From mid-segment: up to the last boundary in reach.
        assert_eq!(364_059 + cp.request_count(364_059, 380_435, 5_000), 369_000);
        assert_eq!(369_000 + cp.request_count(369_000, 380_435, 5_000), 371_500, "stops at the table's end");
        assert_eq!(cp.request_count(371_500, 380_435, 5_000), 5_000, "past it, unchanged");
        // A peer whose tip is mid-segment: the whole segments, then the rest.
        assert_eq!(cp.request_count(370_000, 370_700, 5_000), 500);
        assert_eq!(cp.request_count(370_500, 370_700, 5_000), 200);
    }

    #[test]
    fn the_table_this_build_ships_is_whole_digests() {
        assert_eq!(TABLE.len() % 32, 0);
    }
}
