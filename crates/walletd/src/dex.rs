//! The cross-chain order book, assembled from two chains at once.
//!
//! Neither side requires trusting a counterparty or an indexer:
//!
//! * **Bids** (someone offering ETH for MDS) escrow their ETH inside
//!   `MidstateAtomicSwap` at creation. A `BidCreated` with no matching
//!   `BidClaimed`/`BidCancelled` is therefore *funded* liquidity, verifiable
//!   from contract logs alone.
//! * **Asks** (someone offering MDS for ETH) publish an MDXA announcement as
//!   zero-value burns in the very transaction that funds the HTLC, so an ask
//!   is verifiable by checking those coins are still unspent.
//!
//! This module is read-only. It never signs anything and never moves value,
//! which makes it safe to run against mainnet while the execution paths are
//! still being built.

use crate::base::{BaseClient, Event};
use anyhow::Result;
use midstate::core::dex::{Announcement, FragmentPool, MakerAnnouncement, TakerAnnouncement};
use midstate::node::NodeHandle;
use std::collections::HashMap;

/// A resting buy order living in the Base contract.
#[derive(Clone, Debug)]
pub struct Bid {
    pub bid_id: [u8; 32],
    pub maker: [u8; 20],
    pub hashlock: [u8; 32],
    /// Wei escrowed for the fill.
    pub amount: u128,
    /// Bond a filler must post, and forfeits by reserving and not delivering.
    pub fill_bond: u128,
    pub mds_amount: u64,
    pub maker_mds_addr: [u8; 32],
    pub expiry: u64,
    /// Set once someone holds the exclusive right to fill.
    pub reserved_by: Option<[u8; 20]>,
    pub fill_deadline: Option<u64>,
    pub block: u64,
}

impl Bid {
    /// Price in wei per MDS unit — the only cross-comparable figure between
    /// orders of different sizes.
    pub fn wei_per_unit(&self) -> f64 {
        if self.mds_amount == 0 {
            return 0.0;
        }
        self.amount as f64 / self.mds_amount as f64
    }

    /// A bid already reserved cannot be taken, and one reservation is all a
    /// bid ever gets — after a lapsed fill the preimage may be public, so
    /// re-reserving would let a stranger collect without delivering.
    pub fn is_takeable(&self, now: u64) -> bool {
        self.reserved_by.is_none() && self.expiry > now
    }
}

/// A maker's sell order, announced on midstate.
#[derive(Clone, Debug)]
pub struct Ask {
    pub announcement: MakerAnnouncement,
    /// Height of the block carrying the announcement.
    pub height: u64,
    /// Units whose funding coin is still unspent — the live part of the order.
    pub live_units: Vec<usize>,
}

impl Ask {
    pub fn live_value(&self) -> u64 {
        self.live_units
            .iter()
            .filter_map(|i| self.announcement.units.get(*i))
            .map(|u| u.value)
            .sum()
    }
    pub fn live_wei(&self) -> u128 {
        self.live_units
            .iter()
            .filter_map(|i| self.announcement.units.get(*i))
            .map(|u| u.wei_amount)
            .sum()
    }
    pub fn wei_per_unit(&self) -> f64 {
        let v = self.live_value();
        if v == 0 {
            return 0.0;
        }
        self.live_wei() as f64 / v as f64
    }
}

/// What a scan actually decoded. Without this an empty book is ambiguous —
/// it could mean no orders exist, or that the event ABI is wrong and every
/// log is being silently ignored. These counts tell the two apart.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanStats {
    pub bids_created: usize,
    pub bids_reserved: usize,
    pub bids_closed: usize,
    /// `Locked` events from the direct lock/claim flow.
    pub locks: usize,
    /// `Claimed` events — each one published a preimage.
    pub claims: usize,
    pub refunds: usize,
    /// Logs from the contract that decoded to nothing recognised.
    pub undecoded: usize,
    pub announcements: usize,
}

impl ScanStats {
    pub fn total_events(&self) -> usize {
        self.bids_created + self.bids_reserved + self.bids_closed + self.locks + self.claims + self.refunds
    }
}

#[derive(Default)]
pub struct OrderBook {
    pub bids: HashMap<[u8; 32], Bid>,
    pub asks: Vec<Ask>,
    pub taker_locks: Vec<(u64, TakerAnnouncement)>,
    /// Highest Base block folded in, so scanning resumes rather than restarts.
    pub base_cursor: u64,
    /// Highest midstate height scanned for announcements.
    pub mds_cursor: u64,
    pub stats: ScanStats,
    frags: FragmentPool,
}

impl OrderBook {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold Base contract history into the bid book. Events are applied in
    /// order, so a claim or cancellation always supersedes the creation it
    /// follows regardless of how the range was chunked.
    pub async fn sync_base(&mut self, client: &BaseClient, to: u64) -> Result<usize> {
        if to <= self.base_cursor {
            return Ok(0);
        }
        let from = self.base_cursor.saturating_add(1);
        let (events, raw_logs) = client.scan_events_counted(from, to).await?;
        let n = events.len();
        self.stats.undecoded += raw_logs.saturating_sub(n);
        for (block, e) in events {
            match e {
                Event::BidCreated {
                    bid_id, maker, hashlock, amount, fill_bond, mds_amount, maker_mds_addr, expiry,
                } => {
                    self.stats.bids_created += 1;
                    self.bids.insert(
                        bid_id,
                        Bid {
                            bid_id, maker, hashlock, amount, fill_bond, mds_amount,
                            maker_mds_addr, expiry,
                            reserved_by: None, fill_deadline: None, block,
                        },
                    );
                }
                Event::BidReserved { bid_id, filler, fill_deadline } => {
                    self.stats.bids_reserved += 1;
                    if let Some(b) = self.bids.get_mut(&bid_id) {
                        b.reserved_by = Some(filler);
                        b.fill_deadline = Some(fill_deadline);
                    }
                }
                // Settled or withdrawn: the escrow is gone either way, so the
                // order leaves the book.
                Event::BidClaimed { bid_id, .. } | Event::BidCancelled { bid_id } => {
                    self.stats.bids_closed += 1;
                    self.bids.remove(&bid_id);
                }
                Event::Locked { .. } => self.stats.locks += 1,
                Event::Claimed { .. } => self.stats.claims += 1,
                Event::Refunded { .. } => self.stats.refunds += 1,
            }
        }
        self.base_cursor = to;
        Ok(n)
    }

    /// Scan midstate blocks for announcement burns. Fragments of one
    /// announcement ride in the same transaction, so they normally complete
    /// within a single block.
    pub async fn sync_midstate(&mut self, node: &NodeHandle, to: u64) -> Result<usize> {
        if to <= self.mds_cursor {
            return Ok(0);
        }
        let from = self.mds_cursor;
        let found = scan_announcements(node, from, to, &mut self.frags).await?;
        let n = found.len();
        self.stats.announcements += n;
        for (height, ann) in found {
            match ann {
                Announcement::Maker(m) => {
                    // Re-announcements of the same group replace the old copy.
                    let gid = m.group_id;
                    self.asks.retain(|a| a.announcement.group_id != gid);
                    let live = (0..m.units.len()).collect();
                    self.asks.push(Ask { announcement: m, height, live_units: live });
                }
                Announcement::Taker(t) => self.taker_locks.push((height, t)),
            }
        }
        self.mds_cursor = to;
        Ok(n)
    }

    /// Check every announced ask unit against the live UTXO set. An ask whose
    /// coins are gone has been filled or reclaimed and is no longer offered.
    pub async fn refresh_ask_liveness(&mut self, node: &NodeHandle) -> Result<()> {
        let state = node.get_state().await;
        for ask in &mut self.asks {
            let a = &ask.announcement;
            let mut live = Vec::new();
            for (i, u) in a.units.iter().enumerate() {
                // A maker ask is locked behind the LIMIT ORDER covenant, not a
                // plain HTLC: the receiver is unknown when the order is posted,
                // so the script commits to a max claim instead of a receiver
                // key, and pays the whole unit to whoever supplies the preimage.
                let script = midstate::core::script::compile_limit_order_covenant(
                    &u.secret_hash,
                    u.value,
                    a.timeout_height,
                    &a.maker_mds_pk,
                );
                let cov = midstate::core::types::hash(&script);
                let coin = midstate::core::compute_coin_id(&cov, u.value, &u.salt);
                if state.coins.contains(&coin) {
                    live.push(i);
                }
            }
            ask.live_units = live;
        }
        self.asks.retain(|a| !a.live_units.is_empty());
        Ok(())
    }

    /// Best price first: bids paying the most per unit, asks charging least.
    pub fn sorted_bids(&self, now: u64) -> Vec<&Bid> {
        let mut v: Vec<&Bid> = self.bids.values().filter(|b| b.is_takeable(now)).collect();
        v.sort_by(|a, b| b.wei_per_unit().total_cmp(&a.wei_per_unit()));
        v
    }

    pub fn sorted_asks(&self) -> Vec<&Ask> {
        let mut v: Vec<&Ask> = self.asks.iter().filter(|a| !a.live_units.is_empty()).collect();
        v.sort_by(|a, b| a.wei_per_unit().total_cmp(&b.wei_per_unit()));
        v
    }
}

/// Pull every `DataBurn` payload out of a height range and classify it.
async fn scan_announcements(
    node: &NodeHandle,
    from: u64,
    to: u64,
    pool: &mut FragmentPool,
) -> Result<Vec<(u64, Announcement)>> {
    let mut out = Vec::new();
    for height in from..to {
        let Some(batch) = node.storage.batches.load(height)? else {
            continue;
        };
        for tx in &batch.transactions {
            for payload in burn_payloads(tx) {
                if let Some(a) = midstate::core::dex::ingest(&payload, pool) {
                    out.push((height, a));
                }
            }
        }
    }
    Ok(out)
}

/// Announcements ride as zero-value `DataBurn` outputs on ordinary reveals.
fn burn_payloads(tx: &midstate::core::types::Transaction) -> Vec<Vec<u8>> {
    use midstate::core::types::{OutputData, Transaction};
    let outputs = match tx {
        Transaction::Reveal { outputs, .. } => outputs,
        _ => return Vec::new(),
    };
    outputs
        .iter()
        .filter_map(|o| match o {
            OutputData::DataBurn { payload, .. } => Some(payload.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::dex::AnnUnit;

    fn ann(units: &[(u64, u128)]) -> MakerAnnouncement {
        MakerAnnouncement {
            maker_evm_addr: [1; 20],
            maker_mds_pk: [2; 32],
            timeout_height: 1000,
            group_id: [9; 6],
            units: units
                .iter()
                .enumerate()
                .map(|(i, (v, w))| AnnUnit {
                    secret_hash: [i as u8; 32],
                    salt: [i as u8 ^ 0x55; 32],
                    value: *v,
                    wei_amount: *w,
                })
                .collect(),
        }
    }

    fn bid(id: u8, wei: u128, mds: u64) -> Bid {
        Bid {
            bid_id: [id; 32],
            maker: [0; 20],
            hashlock: [0; 32],
            amount: wei,
            fill_bond: 0,
            mds_amount: mds,
            maker_mds_addr: [0; 32],
            expiry: 10_000,
            reserved_by: None,
            fill_deadline: None,
            block: 1,
        }
    }

    #[test]
    fn bids_rank_by_price_not_size() {
        let mut b = OrderBook::new();
        // A small order paying well should outrank a large one paying poorly.
        b.bids.insert([1; 32], bid(1, 1_000, 100)); // 10 wei/unit
        b.bids.insert([2; 32], bid(2, 90_000, 30_000)); // 3 wei/unit
        let sorted = b.sorted_bids(0);
        assert_eq!(sorted[0].bid_id, [1; 32]);
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn reserved_and_expired_bids_are_not_takeable() {
        let mut taken = bid(1, 100, 10);
        taken.reserved_by = Some([7; 20]);
        assert!(!taken.is_takeable(0));

        let expired = bid(2, 100, 10);
        assert!(!expired.is_takeable(20_000));
        assert!(expired.is_takeable(0));
    }

    #[test]
    fn asks_price_only_their_live_units() {
        let mut a = Ask { announcement: ann(&[(1024, 5_000), (2048, 20_000)]), height: 1, live_units: vec![0, 1] };
        assert_eq!(a.live_value(), 3072);
        assert_eq!(a.live_wei(), 25_000);

        // Once the expensive unit is filled the ask reprices to what is left.
        a.live_units = vec![0];
        assert_eq!(a.live_value(), 1024);
        assert!((a.wei_per_unit() - 5_000.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn asks_sort_cheapest_first_and_empty_ones_drop_out() {
        let mut b = OrderBook::new();
        b.asks.push(Ask { announcement: ann(&[(1024, 10_240)]), height: 1, live_units: vec![0] }); // 10/unit
        b.asks.push(Ask { announcement: ann(&[(1024, 2_048)]), height: 2, live_units: vec![0] }); // 2/unit
        b.asks.push(Ask { announcement: ann(&[(1024, 1)]), height: 3, live_units: vec![] }); // fully filled
        let s = b.sorted_asks();
        assert_eq!(s.len(), 2);
        assert!(s[0].wei_per_unit() < s[1].wei_per_unit());
    }
}
