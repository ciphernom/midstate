//! MDS <-> ETH and MDS <-> USDC trading on Base: sell orders, selling into
//! buy orders, and the decisions that take each swap from its locks to its
//! claims.
//!
//! Both currencies run the same machine. ETH trades go through
//! `MidstateAtomicSwap.sol`; USDC trades through `MidstateTokenSwap.sol`,
//! the same protocol for one ERC-20, whose events read the same. A trade's
//! [`Asset`] says which, and every amount in it (`wei`, `wei_per_gmds`,
//! `fill_bond`) is counted in that asset's smallest unit: wei, or a
//! millionth of a dollar. A lock on the other contract is never the lock a
//! trade waits for, whatever its numbers say.
//!
//! # Sell orders: the seller locks first
//!
//! The order of the protocol `MidstateAtomicSwap.sol` documents:
//!
//! 1. The buyer asks (`eth_wire::Take`) with a fresh one-time key.
//! 2. The seller locks the MDS on midstate behind `compile_htlc(H, buyer's
//!    key, T, seller's key)`: only a spend signed by the buyer's key and
//!    showing the secret can take it before `T`; after `T`, only the seller.
//! 3. The buyer sees that lock confirmed, then locks ETH on Base with the
//!    same `H`, paying the seller, refundable well before `T`.
//! 4. The seller claims the ETH, which publishes the secret on Base.
//! 5. The buyer claims the MDS with the secret and its signature.
//!
//! The secret is useless to anyone but the buyer, so a bot that reads it on
//! Base gains nothing, and the seller, who knows it all along, cannot take
//! the MDS back before `T`. The seller reveals nothing until ETH is locked
//! to it for at least the price, from the buyer that asked.
//!
//! # Selling into a buy order
//!
//! A buy order is ETH escrowed in the contract (`createBid`) against the
//! bidder's hash. The seller reserves it (`reserveBid`, with the bidder's
//! bond), locks MDS behind `compile_covenant_htlc`, which pays the bid's
//! amount to the bidder's address and nowhere else, and announces the lock
//! (`dex::FillAnnouncement`). The bidder claims it, publishing the secret on
//! midstate; the seller uses it to collect the escrow (`claimBid`), which
//! the contract pays only to the reservation's holder. The lock's timeout
//! ends well before the reservation, so after the latest possible claim the
//! seller still has time to collect.
//!
//! # Deadlines
//!
//! Midstate deadlines are heights; Base's are unix times. Every comparison
//! converts heights to seconds the way that hurts the party it protects
//! (`swap::blocks_to_secs_pessimistic` / `_optimistic`), with the margins
//! below on top.

use crate::base::BidState;
use crate::swap::{blocks_to_secs_optimistic, blocks_to_secs_pessimistic, secs_to_blocks_generous, SETTLE_MARGIN_SECS};
use anyhow::{anyhow, bail, ensure, Result};
use midstate::core::dex::FillAnnouncement;
use midstate::core::script::{compile_covenant_htlc, compile_htlc};
use midstate::core::types::{compute_coin_id, compute_commitment, hash, InputReveal, OutputData, Predicate, Witness};
use midstate::core::{decompose_value, wots, Transaction as MdsTx};
use serde::{Deserialize, Serialize};

/// Midstate blocks on top of a lock before the other side pays against it.
pub const CONFIRMATIONS: u64 = 2;
/// Midstate blocks between adverts of an open order.
pub const ADVERT_EVERY: u64 = 5;
/// A heard advert not repeated in this many midstate blocks has gone.
pub const ADVERT_STALE: u64 = 20;
/// Order lifetimes a seller may choose, in midstate blocks.
pub const MIN_ORDER_LIFE: u64 = 60;
pub const MAX_ORDER_LIFE: u64 = 43_200;
/// Smallest trade: every lock coin is spent by a claim whose fee comes out
/// of the lock.
pub const MIN_MDS_AMOUNT: u64 = 4_096;
/// Most coins one lock may be (one input each in its claim).
pub const MAX_MDS_COINS: usize = 16;
/// The contract's shortest refund delay.
pub const MIN_ETH_DELAY: u64 = 600;
/// How long a buyer's ETH stays locked when the seller's lock allows it.
pub const DEFAULT_ETH_DELAY: u64 = 3_600;
/// Time the buyer keeps in hand, between seeing the seller's lock and its
/// own ETH lock being mined and seen.
pub const LOCK_SLACK_SECS: u64 = 1_200;
/// Seconds allowed for an ETH lock to be mined.
pub const ETH_MINE_SECS: u64 = 300;
/// The seller claims only while at least this long remains on the buyer's
/// ETH lock: a claim must land well before the buyer can refund.
pub const CLAIM_MARGIN_SECS: u64 = 300;
/// Seconds after an ETH deadline before the refund is sent: clocks differ.
pub const REFUND_GRACE_SECS: u64 = 60;
/// A buyer stops asking after this long without an answer.
pub const TAKE_GIVE_UP_SECS: u64 = 300;
/// A buyer stops waiting for an announced lock to confirm after this long.
pub const LOCK_WAIT_SECS: u64 = 1_800;
/// Seconds between repeats of a take, or of a seller's answer.
pub const RESEND_SECS: u64 = 20;
/// Locks a seller keeps per order that are still waiting for ETH.
pub const MAX_UNFUNDED: usize = 2;
/// How long a seller's reservation of a buy order lasts (the contract
/// allows 2 hours to 3 days).
pub const FILL_WINDOW_SECS: u64 = 4 * 3_600;
/// Time a filler keeps between its lock's timeout and the reservation's end,
/// beyond the settle margin.
pub const FILL_SLACK_SECS: u64 = 1_200;
/// Fewest midstate blocks a fill lock may run.
pub const MIN_FILL_BLOCKS: u64 = 40;
/// The bidder claims a fill only while its lock has this many blocks left.
pub const BIDDER_CLAIM_BLOCKS: u64 = 6;
/// The bidder publishes its secret only while this many blocks are left:
/// a claim revealed later could lose to the filler's refund.
pub const REVEAL_MIN_BLOCKS: u64 = 3;
/// How far apart our clock and Base's may be, on top of every Base margin.
pub const CLOCK_SKEW_SECS: u64 = 60;
/// A filler accepts a bond of at most this share of the escrow (1/n): a
/// bidder who never claims keeps the bond, so it bounds that loss.
pub const MAX_BOND_SHARE: u128 = 10;
/// Units in 1 MDS, as prices are quoted.
pub const GMDS: u128 = 1 << 30;

// Midstate's relay floor (`fee * 1024 >= 10 * size`), as `swap.rs` repeats it.
const MIN_FEE_PER_KB: u64 = 10;
const FEE_RATE_SCALE: u64 = 1_024;
const FEE_MARGIN: u64 = 20;

mod wei {
    //! u128 as a decimal string: JSON numbers above u64 do not survive.
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &u128, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&v.to_string())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

// ── What the Base side pays in ──────────────────────────────────────────────

/// The currency of a trade's Base side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Asset {
    /// Ether, in wei, through `MidstateAtomicSwap`.
    #[default]
    Eth,
    /// USDC, in millionths of a dollar, through `MidstateTokenSwap`.
    Usdc,
}

impl Asset {
    pub const ALL: [Asset; 2] = [Asset::Eth, Asset::Usdc];

    pub fn symbol(self) -> &'static str {
        match self {
            Asset::Eth => "ETH",
            Asset::Usdc => "USDC",
        }
    }

    /// Decimal places between the smallest unit and one whole coin.
    pub fn decimals(self) -> u32 {
        match self {
            Asset::Eth => 18,
            Asset::Usdc => 6,
        }
    }

    /// `ETH` or `USDC`, in any case.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_uppercase().as_str() {
            "" | "ETH" => Ok(Asset::Eth),
            "USDC" => Ok(Asset::Usdc),
            other => bail!("{other} is not traded here: ETH or USDC"),
        }
    }

    /// `amount` smallest units as whole coins, to `places` decimals at most,
    /// trailing zeros dropped: `1.5 USDC`.
    pub fn format(self, amount: u128, places: u32) -> String {
        let one = 10u128.pow(self.decimals());
        let (whole, frac) = (amount / one, amount % one);
        let places = places.min(self.decimals());
        let frac = frac / 10u128.pow(self.decimals() - places);
        let mut f = format!("{frac:0width$}", width = places as usize);
        while f.ends_with('0') {
            f.pop();
        }
        if f.is_empty() {
            format!("{whole} {}", self.symbol())
        } else {
            format!("{whole}.{f} {}", self.symbol())
        }
    }
}

// ── Roles and phases ────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Our sell order was taken: we lock MDS, then claim the buyer's ETH.
    Seller,
    /// We took a sell order: we lock ETH, then claim the seller's MDS.
    Buyer,
    /// We sell into someone's buy order: reserve, lock MDS, collect the escrow.
    Filler,
    /// Our buy order was filled: we claim the seller's MDS.
    Bidder,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Buyer: the take is out; waiting for the seller's answer.
    Asking,
    /// Filler: the reservation is out; waiting for it to be mined.
    Reserving,
    /// Buyer, bidder: waiting for the other side's MDS lock to confirm.
    WaitingTheirLock,
    /// Our lock is going out: MDS for a seller or filler, ETH for a buyer.
    LockingMine,
    /// Our lock is in place; waiting on the other side.
    MineLocked,
    /// Claiming the other side's lock.
    Claiming,
    /// Taking our own lock back.
    Refunding,
    Done,
    Refunded,
    Failed(String),
}

impl Phase {
    pub fn settled(&self) -> bool {
        matches!(self, Phase::Done | Phase::Refunded | Phase::Failed(_))
    }
    pub fn name(&self) -> &'static str {
        match self {
            Phase::Asking => "asking",
            Phase::Reserving => "reserving",
            Phase::WaitingTheirLock => "waiting",
            Phase::LockingMine => "locking",
            Phase::MineLocked => "locked",
            Phase::Claiming => "claiming",
            Phase::Refunding => "refunding",
            Phase::Done => "done",
            Phase::Refunded => "refunded",
            Phase::Failed(_) => "failed",
        }
    }
}

// ── Orders ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Open,
    Cancelled,
    /// Everything sold.
    Filled,
}

/// One of our sell orders. It lives only in this wallet and on the bus:
/// nothing is locked until a buyer asks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EthOrder {
    pub id: [u8; 16],
    /// MDS still for sale: neither sold nor locked for a buyer.
    pub mds: u64,
    pub min: u64,
    #[serde(with = "wei")]
    pub wei_per_gmds: u128,
    pub expires: u64,
    pub seller_evm: [u8; 20],
    pub status: OrderStatus,
    pub created: u64,
    #[serde(default)]
    pub last_advert: u64,
    /// MDS sold so far.
    #[serde(default)]
    pub sold: u64,
    /// What it sells for; its price is in this asset's smallest unit.
    #[serde(default)]
    pub asset: Asset,
}

impl EthOrder {
    pub fn takeable(&self, tip: u64) -> bool {
        self.status == OrderStatus::Open && tip < self.expires && self.mds >= self.min.max(MIN_MDS_AMOUNT)
    }
    pub fn advert(&self) -> crate::eth_wire::Advert {
        crate::eth_wire::Advert {
            order_id: self.id,
            seller_evm: self.seller_evm,
            mds: self.mds,
            min: self.min.max(MIN_MDS_AMOUNT),
            wei_per_gmds: self.wei_per_gmds,
            expires: self.expires,
            asset: self.asset,
        }
    }
}

/// What `mds` costs at `wei_per_gmds`, rounded up: a buyer never pays less
/// than the price.
pub fn price_of(mds: u64, wei_per_gmds: u128) -> Option<u128> {
    (mds as u128).checked_mul(wei_per_gmds).map(|v| v.div_ceil(GMDS))
}

/// `mds`, rounded down to the most it can be while splitting into at most
/// [`MAX_MDS_COINS`] coins.
pub fn tradeable_amount(mds: u64) -> u64 {
    let mut v = mds;
    while v.count_ones() as usize > MAX_MDS_COINS {
        v &= v - 1; // drop the lowest set bit
    }
    v
}

/// Why a take of `order` for `mds` and `wei`, paid in `asset`, cannot
/// stand, if it cannot. A take in another currency asks for an order that
/// is not on offer.
pub fn check_take(order: &EthOrder, tip: u64, mds: u64, wei: u128, asset: Asset) -> std::result::Result<(), crate::eth_wire::Decline> {
    use crate::eth_wire::Decline;
    if !order.takeable(tip) || asset != order.asset {
        return Err(Decline::Gone);
    }
    if mds < order.min.max(MIN_MDS_AMOUNT) || mds > order.mds || tradeable_amount(mds) != mds {
        return Err(Decline::Amount);
    }
    match price_of(mds, order.wei_per_gmds) {
        Some(p) if wei >= p => Ok(()),
        _ => Err(Decline::Price),
    }
}

/// A heard advert.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BookEntry {
    pub advert: crate::eth_wire::Advert,
    pub heard: u64,
}

// ── The swap journal ────────────────────────────────────────────────────────

/// One swap, whichever role. Fields a role does not use stay at their zero.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EthSwap {
    /// Sell orders: the buyer's take id. Buy orders: the bid id's first half.
    pub id: [u8; 16],
    pub role: Role,
    pub phase: Phase,
    /// Sell orders: the order taken.
    pub order_id: [u8; 16],
    /// Buy orders: the bid.
    pub bid_id: [u8; 32],
    /// MDS the buyer or bidder receives.
    pub mds_amount: u64,
    /// MDS locked: the amount, plus for a fill the bidder's claim fee.
    pub lock_value: u64,
    #[serde(with = "wei")]
    pub wei: u128,
    /// Buy orders: the reservation's bond.
    #[serde(with = "wei")]
    pub fill_bond: u128,
    /// Who the ETH pays: the seller, or the filler.
    pub seller_evm: [u8; 20],
    /// Sell orders: where the ETH lock comes from and refunds to.
    pub buyer_evm: [u8; 20],
    /// Sell orders: the buyer's one-time key, the only one that can claim.
    pub claim_pk: [u8; 32],
    /// Buy orders: the address the fill lock must pay.
    pub receiver_addr: [u8; 32],
    /// The key that takes the lock back after `timeout`.
    pub refund_pk: [u8; 32],
    /// Our one-time key's seed: the buyer's claim key, or the seller's or
    /// filler's refund key. It signs exactly one message.
    pub key_seed: Option<[u8; 32]>,
    pub secret: Option<[u8; 32]>,
    pub secret_hash: [u8; 32],
    /// The MDS lock's refund height.
    pub timeout: u64,
    /// The wallet send that funds our MDS lock.
    pub lock_commitment: Option<[u8; 32]>,
    /// When the MDS lock was first seen complete.
    pub lock_seen: Option<u64>,
    /// Midstate height the search for the lock's spend resumes from.
    #[serde(default)]
    pub scan_from: u64,
    /// Our latest Base transaction.
    pub eth_tx: Option<[u8; 32]>,
    /// The contract's id for the ETH lock (sell orders).
    pub base_swap: Option<[u8; 32]>,
    /// Sell orders: when the ETH lock may be refunded. Fills: when the
    /// reservation ends.
    pub eth_deadline: u64,
    /// Base block the swap's events are looked for from.
    #[serde(default)]
    pub base_from: u64,
    /// A claim of the other side's lock has gone out: from then on our own
    /// lock is never taken back.
    #[serde(default)]
    pub claim_sent: bool,
    /// Where our claim or refund of the MDS lock pays.
    pub dest: Option<[u8; 32]>,
    pub spend_commitment: Option<[u8; 32]>,
    pub created: u64,
    pub updated: u64,
    /// Unix time of our last message for this swap.
    #[serde(default)]
    pub last_msg: u64,
    #[serde(default)]
    pub note: String,
    /// The wallet has done its bookkeeping for this swap's end.
    #[serde(default)]
    pub booked: bool,
    /// The lock coins our refund spends, one bit per output index. Fixed
    /// before its first commit (its one-time key signs once); 0 until then.
    #[serde(default)]
    pub refund_mask: u32,
    /// Bidder: the claim's reveal, which publishes the secret, has gone out.
    #[serde(default)]
    pub reveal_sent: bool,
    /// Our value-carrying Base transaction (ETH lock or reservation), signed
    /// and journalled before it is sent, so a lost answer cannot hide a
    /// payment. Its hash is `eth_tx`.
    #[serde(default)]
    pub eth_pending: Option<EthPending>,
    /// What the Base side pays in: every amount here is in its unit, and
    /// only its contract's locks and bids are this swap's.
    #[serde(default)]
    pub asset: Asset,
}

/// A signed Base transaction that may yet be mined.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EthPending {
    /// The signed transaction, hex.
    pub raw: String,
    pub nonce: u64,
    /// When it was signed: it is sent again only within `ETH_MINE_SECS`.
    pub at: u64,
}

impl EthSwap {
    pub fn new(id: [u8; 16], role: Role, phase: Phase, now: u64) -> Self {
        Self {
            id,
            role,
            phase,
            order_id: [0; 16],
            bid_id: [0; 32],
            mds_amount: 0,
            lock_value: 0,
            wei: 0,
            fill_bond: 0,
            seller_evm: [0; 20],
            buyer_evm: [0; 20],
            claim_pk: [0; 32],
            receiver_addr: [0; 32],
            refund_pk: [0; 32],
            key_seed: None,
            secret: None,
            secret_hash: [0; 32],
            timeout: 0,
            lock_commitment: None,
            lock_seen: None,
            scan_from: 0,
            eth_tx: None,
            base_swap: None,
            eth_deadline: 0,
            base_from: 0,
            claim_sent: false,
            dest: None,
            spend_commitment: None,
            created: now,
            updated: now,
            last_msg: 0,
            note: String::new(),
            booked: false,
            refund_mask: 0,
            reveal_sent: false,
            eth_pending: None,
            asset: Asset::Eth,
        }
    }

    /// Whether this swap's MDS lock is a sell order's (signature) HTLC, not
    /// a fill's covenant.
    pub fn sell_order(&self) -> bool {
        matches!(self.role, Role::Seller | Role::Buyer)
    }

    /// Whether we hold the lock on midstate (and the other side's is on Base).
    pub fn mds_is_mine(&self) -> bool {
        matches!(self.role, Role::Seller | Role::Filler)
    }

    /// The fill this swap's lock is, for a buy order.
    pub fn fill(&self) -> FillAnnouncement {
        FillAnnouncement {
            taker_mds_pk: self.refund_pk,
            secret_hash: self.secret_hash,
            receiver_addr: self.receiver_addr,
            timeout_height: self.timeout,
            min_payout: self.mds_amount,
            total: self.lock_value,
            wei_amount: self.wei,
        }
    }

    pub fn lock_script(&self) -> Vec<u8> {
        if self.sell_order() {
            compile_htlc(&self.secret_hash, &self.claim_pk, self.timeout, &self.refund_pk)
        } else {
            compile_covenant_htlc(&self.secret_hash, &self.receiver_addr, self.mds_amount, self.timeout, &self.refund_pk)
        }
    }

    pub fn lock_address(&self) -> [u8; 32] {
        hash(&self.lock_script())
    }

    /// The lock's coins, `(value, salt)`, in output order. Both sides derive
    /// them from public terms, so no salt crosses the wire.
    pub fn lock_coins(&self) -> Vec<(u64, [u8; 32])> {
        if self.sell_order() {
            decompose_value(self.lock_value)
                .into_iter()
                .enumerate()
                .map(|(i, v)| (v, lock_salt(&self.id, &self.secret_hash, i as u32)))
                .collect()
        } else {
            self.fill().coins()
        }
    }

    pub fn lock_coin_ids(&self) -> Vec<[u8; 32]> {
        let addr = self.lock_address();
        self.lock_coins().iter().map(|(v, s)| compute_coin_id(&addr, *v, s)).collect()
    }

    pub fn lock_outputs(&self) -> Vec<OutputData> {
        let address = self.lock_address();
        self.lock_coins().into_iter().map(|(value, salt)| OutputData::Standard { address, value, salt }).collect()
    }

    /// The secret, from the items of an input spending the lock.
    ///
    /// Our wallets claim with `[sig, secret, 1]` (sell orders) or
    /// `[secret, 1]` (fills), but the script takes any true selector and
    /// any items beneath, so every 32-byte item is tried against the hash.
    /// A refund carries no secret.
    pub fn secret_from_witness(&self, items: &[Vec<u8>]) -> Option<[u8; 32]> {
        items.iter().find_map(|it| {
            let s: [u8; 32] = it.as_slice().try_into().ok()?;
            (hash(&s) == self.secret_hash).then_some(s)
        })
    }

    /// The value of the lock coins in `mask`.
    pub fn value_in(&self, mask: u32) -> u64 {
        self.lock_coins().iter().enumerate().filter(|(i, _)| mask & (1 << i) != 0).map(|(_, c)| c.0).sum()
    }

    /// Whether a refund of the lock coins in `mask` can pay its own fee.
    pub fn refundable(&self, mask: u32) -> bool {
        mask != 0 && build_spend(&EthSwap { refund_mask: mask, ..self.clone() }, SpendKind::Refund, &[0; 32]).is_ok()
    }
}

/// Whether a buy order's bond is one a filler should risk: a bidder that
/// never claims keeps it.
pub fn bond_acceptable(bond: u128, escrow: u128) -> bool {
    bond <= escrow / MAX_BOND_SHARE
}

/// Salt of coin `i` of a sell order's lock.
pub fn lock_salt(take_id: &[u8; 16], secret_hash: &[u8; 32], i: u32) -> [u8; 32] {
    let mut m = Vec::with_capacity(32 + 52);
    m.extend_from_slice(b"midstate.eth-swap.lock-salt.v1");
    m.extend_from_slice(take_id);
    m.extend_from_slice(secret_hash);
    m.extend_from_slice(&i.to_le_bytes());
    hash(&m)
}

/// The refund height a seller gives a new lock: room for the buyer to see
/// it and lock, a full ETH lock, and the settle margin after it.
pub fn seller_timeout(tip: u64) -> u64 {
    tip + secs_to_blocks_generous(LOCK_SLACK_SECS + DEFAULT_ETH_DELAY + SETTLE_MARGIN_SECS)
}

/// How long the buyer's ETH lock should run against a lock refundable at
/// `timeout`, or `None` if there is no longer room for the shortest.
///
/// # Formal Specification
///
/// ```text
/// Post: Some(d) ⇒ MIN_ETH_DELAY ≤ d ≤ DEFAULT_ETH_DELAY
///                 ∧ now + ETH_MINE_SECS + d + SETTLE_MARGIN ≤ now + pessimistic(timeout − tip)
/// ```
/// so even if midstate's blocks come fast, the buyer has the settle margin
/// after its ETH lock's deadline to claim before the seller can refund.
pub fn buyer_delay(tip: u64, timeout: u64) -> Option<u64> {
    let left = blocks_to_secs_pessimistic(timeout.saturating_sub(tip));
    let room = left.checked_sub(SETTLE_MARGIN_SECS + ETH_MINE_SECS)?;
    let d = room.min(DEFAULT_ETH_DELAY);
    (d >= MIN_ETH_DELAY).then_some(d)
}

/// The refund height for a fill lock when the reservation ends at
/// `fill_deadline`: the latest at which, even with slow blocks, a claim at
/// the timeout leaves the settle margin and slack before the deadline.
pub fn filler_timeout(tip: u64, now: u64, fill_deadline: u64) -> Option<u64> {
    let room = fill_deadline.checked_sub(now + SETTLE_MARGIN_SECS + FILL_SLACK_SECS)?;
    // The largest n with blocks_to_secs_optimistic(n) ≤ room.
    let n = room / blocks_to_secs_optimistic(1);
    (n >= MIN_FILL_BLOCKS).then_some(tip + n)
}

/// What a fill of `mds` must lock so the bidder's claim can pay its own fee
/// and still deliver `mds`, in at most [`MAX_MDS_COINS`] coins.
pub fn fill_total(mds: u64, template: &EthSwap) -> Result<u64> {
    let mut total = mds + 64;
    for _ in 0..64 {
        while total.count_ones() as usize > MAX_MDS_COINS {
            total += total & total.wrapping_neg();
        }
        let mut sw = template.clone();
        sw.mds_amount = mds;
        sw.lock_value = total;
        let fee = build_spend(&sw, SpendKind::Claim, &sw.receiver_addr)?.fee;
        if total >= mds + fee {
            return Ok(total);
        }
        total = mds + fee;
    }
    bail!("the fill's size did not settle")
}

// ── Spending the MDS lock ───────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpendKind {
    /// The buyer's claim of a sell order's lock, or the bidder's of a fill.
    Claim,
    /// The seller's or filler's refund after the timeout.
    Refund,
}

/// A claim or refund of the MDS lock, as committed and as revealed.
#[derive(Clone, Debug)]
pub struct Spend {
    pub inputs: Vec<InputReveal>,
    pub outputs: Vec<OutputData>,
    pub salt: [u8; 32],
    pub commitment: [u8; 32],
    pub fee: u64,
}

impl Spend {
    pub fn output_ids(&self) -> Vec<[u8; 32]> {
        self.outputs.iter().filter_map(|o| o.coin_id()).collect()
    }
}

/// One input's witness items for `kind`, around `sig`.
fn witness(sw: &EthSwap, kind: SpendKind, sig: &[u8], secret: &[u8; 32]) -> Vec<Vec<u8>> {
    match (kind, sw.sell_order()) {
        (SpendKind::Claim, true) => vec![sig.to_vec(), secret.to_vec(), vec![0x01]],
        (SpendKind::Claim, false) => vec![secret.to_vec(), vec![0x01]],
        (SpendKind::Refund, _) => vec![sig.to_vec(), vec![0u8; 32], vec![0x00]],
    }
}

fn reveal(sp: &Spend, items: &[Vec<u8>]) -> MdsTx {
    MdsTx::Reveal {
        inputs: sp.inputs.clone(),
        witnesses: sp.inputs.iter().map(|_| Witness::ScriptInputs(items.to_vec())).collect(),
        outputs: sp.outputs.clone(),
        salt: sp.salt,
    }
}

/// Builds the claim or refund of the MDS lock, paying `dest`.
///
/// # Formal Specification
///
/// ```text
/// Post: a pure function of the journal, so the commit and the reveal
///       rebuild the same transaction and a one-time key signs one message
///       ∧ fee · 1024 ≥ 10 · |bincode(reveal)| + margin, the relay floor
///       ∧ fill claim ⇒ Σ outputs to receiver_addr ≥ mds_amount
/// ```
pub fn build_spend(sw: &EthSwap, kind: SpendKind, dest: &[u8; 32]) -> Result<Spend> {
    // A claim takes the whole lock; a refund the coins planned for it.
    let coins: Vec<(u64, [u8; 32])> = sw
        .lock_coins()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| kind == SpendKind::Claim || sw.refund_mask == 0 || sw.refund_mask & (1 << i) != 0)
        .map(|(_, c)| c)
        .collect();
    ensure!(!coins.is_empty(), "the lock has no coins");
    ensure!(coins.len() <= MAX_MDS_COINS, "the lock is {} coins, more than {MAX_MDS_COINS}", coins.len());
    let total: u64 = coins.iter().map(|c| c.0).sum();
    let bytecode = sw.lock_script();
    let inputs: Vec<InputReveal> = coins
        .iter()
        .map(|&(value, salt)| InputReveal { predicate: Predicate::Script { bytecode: bytecode.clone() }, value, salt, commitment: None })
        .collect();
    let label: &[u8] = match kind {
        SpendKind::Claim => b"claim",
        SpendKind::Refund => b"refund",
    };
    let derive = |what: &[u8], i: u32| {
        let mut h = Vec::with_capacity(96);
        h.extend_from_slice(b"midstate.eth-swap.spend.v1");
        h.extend_from_slice(&sw.id);
        h.extend_from_slice(&sw.secret_hash);
        h.extend_from_slice(label);
        h.extend_from_slice(what);
        h.extend_from_slice(&i.to_le_bytes());
        hash(&h)
    };
    let salt = derive(b"tx", 0);
    let shape = witness(sw, kind, &vec![0u8; wots::SIG_SIZE], &[0u8; 32]);
    let floor = if kind == SpendKind::Claim && !sw.sell_order() { sw.mds_amount } else { 0 };
    let mut fee = 0u64;
    for _ in 0..16 {
        ensure!(fee < total, "the lock cannot pay its own {kind:?} fee");
        ensure!(total - fee >= floor, "the lock leaves {} after its claim fee, under the {floor} it must deliver", total - fee);
        let outputs: Vec<OutputData> = decompose_value(total - fee)
            .into_iter()
            .enumerate()
            .map(|(i, value)| OutputData::Standard { address: *dest, value, salt: derive(b"out", i as u32) })
            .collect();
        let sp = Spend { inputs: inputs.clone(), outputs, salt, commitment: [0; 32], fee };
        let size = bincode::serialized_size(&reveal(&sp, &shape))?;
        let need = (size * MIN_FEE_PER_KB).div_ceil(FEE_RATE_SCALE) + FEE_MARGIN;
        if need <= fee {
            let ins: Vec<[u8; 32]> = sp.inputs.iter().map(|i| i.coin_id()).collect();
            let outs: Vec<[u8; 32]> = sp.outputs.iter().map(|o| o.hash_for_commitment()).collect();
            return Ok(Spend { commitment: compute_commitment(&ins, &outs, &salt), ..sp });
        }
        fee = need;
    }
    bail!("the {kind:?} fee did not settle")
}

/// The reveal of a committed claim or refund, signed with our one-time key
/// where the script asks for a signature.
pub fn sign_spend(sw: &EthSwap, kind: SpendKind, sp: &Spend) -> Result<MdsTx> {
    let secret = match kind {
        SpendKind::Claim => sw.secret.ok_or_else(|| anyhow!("the secret is not known yet"))?,
        SpendKind::Refund => [0; 32],
    };
    let needs_sig = sw.sell_order() || kind == SpendKind::Refund;
    let sig = if needs_sig {
        let seed = sw.key_seed.ok_or_else(|| anyhow!("this wallet holds no key for that spend"))?;
        ensure!(wots::keygen(&seed) == if kind == SpendKind::Claim { sw.claim_pk } else { sw.refund_pk }, "the stored key is not the lock's");
        wots::sig_to_bytes(&wots::sign(&seed, &sp.commitment))
    } else {
        Vec::new()
    };
    Ok(reveal(sp, &witness(sw, kind, &sig, &secret)))
}

pub fn pk_of(seed: &[u8; 32]) -> [u8; 32] {
    wots::keygen(seed)
}

// ── Observing ───────────────────────────────────────────────────────────────

/// A lock on Base, as a contract's `Locked` event reported it: ETH from
/// `MidstateAtomicSwap`, USDC from `MidstateTokenSwap`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseLock {
    pub swap_id: [u8; 32],
    pub beneficiary: [u8; 20],
    pub refund_to: [u8; 20],
    /// In `asset`'s smallest unit.
    pub amount: u128,
    pub timeout: u64,
    pub block: u64,
    /// Which contract emitted it.
    pub asset: Asset,
}

/// What a swap reads from the two chains. The service answers from its
/// node and its Base scan; tests from simulated chains.
pub trait Chains {
    fn mds_tip(&self) -> u64;
    fn mds_coin(&self, id: &[u8; 32]) -> bool;
    /// The witnesses of every input spending one of `coins`, in the mempool
    /// and in blocks from `from` up; and the height the block search reached
    /// (every block below it was read).
    fn mds_spends(&self, coins: &[[u8; 32]], from: u64) -> (Vec<Vec<Vec<u8>>>, u64);
    /// Whether our lock's send is still under way: the wallet still holds
    /// the send `commitment`, or the mempool holds a transaction creating
    /// `coin`.
    fn mds_lock_pending(&self, commitment: &[u8; 32], coin: &[u8; 32]) -> bool;
    fn now(&self) -> u64;
    /// Base's latest block time as last read; 0 if not read yet.
    fn base_now(&self) -> u64;
    /// ETH locks seen with this hashlock.
    fn base_locks(&self, hashlock: &[u8; 32]) -> Vec<BaseLock>;
    /// The secret a claim of this ETH lock published, once seen.
    fn base_claim(&self, swap_id: &[u8; 32]) -> Option<[u8; 32]>;
    fn base_refunded(&self, swap_id: &[u8; 32]) -> bool;
    /// The bid as the contract holds it, as last read.
    fn bid(&self, bid_id: &[u8; 32]) -> Option<BidState>;
    /// Whether the bid's escrow was claimed (`BidClaimed`), which pays only
    /// the holder of its reservation.
    fn bid_claimed(&self, bid_id: &[u8; 32]) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lock {
    Absent,
    Unconfirmed,
    Confirmed,
    /// On chain once, now (partly) spent.
    Spent,
}

/// What one tick saw, for [`decide`].
#[derive(Clone, Debug)]
pub struct Obs {
    pub tip: u64,
    pub now: u64,
    /// Base's clock where it is known and ahead of ours.
    pub base_now: u64,
    pub lock: Lock,
    /// The lock coins still unspent, one bit per output index.
    pub left: u32,
    /// Our lock's send may still land.
    pub lock_pending: bool,
    /// The search for spends of the lock has read every block (filler).
    pub search_done: bool,
    /// Our committed spend is on chain.
    pub spend_landed: bool,
    /// The ETH lock this swap is about: for a seller, the first that meets
    /// the take's terms; for a buyer, its own.
    pub base_lock: Option<BaseLock>,
    /// Locks with our hash that a seller turned down, for the note.
    pub ignored_locks: usize,
    pub base_claimed: bool,
    pub base_refunded: bool,
    pub bid: Option<BidState>,
    pub bid_claimed: bool,
}

impl Obs {
    /// The later of our clock and Base's: Base deadlines are judged by it.
    pub fn base_time(&self) -> u64 {
        self.now.max(self.base_now)
    }
}

fn depth_ok(tip: u64, since: u64) -> bool {
    tip >= since + CONFIRMATIONS
}

/// Whether `l` is what a seller agreed to: the trade's currency, enough of
/// it, from the buyer that asked, to the seller, with time left to claim it
/// by Base's clock.
pub fn lock_meets_terms(sw: &EthSwap, l: &BaseLock, base_time: u64) -> bool {
    l.asset == sw.asset
        && l.beneficiary == sw.seller_evm
        && l.refund_to == sw.buyer_evm
        && l.amount >= sw.wei
        && l.timeout >= base_time + CLAIM_MARGIN_SECS + CLOCK_SKEW_SECS
}

/// The lock coins unspent now, one bit per output index.
pub fn coins_left(sw: &EthSwap, c: &impl Chains) -> u32 {
    if sw.timeout == 0 || sw.lock_value == 0 {
        return 0;
    }
    sw.lock_coin_ids().iter().enumerate().filter(|(_, id)| c.mds_coin(id)).fold(0, |m, (i, _)| m | 1 << i)
}

/// Reads the chains for one swap, recording in it what only needs reading
/// once: when the lock appeared, where the spend search got to, the secret.
///
/// # Formal Specification
///
/// ```text
/// Filler: secret = None after this call ⇒ no input spending any lock coin,
///         in blocks [first scan_from, scan_from) or the mempool, has a
///         32-byte witness item hashing to secret_hash
/// ```
pub fn observe(sw: &mut EthSwap, c: &impl Chains) -> Obs {
    let (tip, now) = (c.mds_tip(), c.now());
    let ids = if sw.timeout == 0 || sw.lock_value == 0 { Vec::new() } else { sw.lock_coin_ids() };
    let left = coins_left(sw, c);
    let whole = !ids.is_empty() && left.count_ones() as usize == ids.len();
    if whole {
        let seen = *sw.lock_seen.get_or_insert(tip);
        if sw.scan_from == 0 {
            sw.scan_from = seen.saturating_sub(1);
        }
    }

    // A filler reads the secret off the bidder's claim: any spend of any
    // lock coin, in any block since the lock could exist, whatever shape its
    // witness takes. Spends without it (our refund) are passed over.
    let mut found_spend = false;
    let mut search_done = true;
    if sw.role == Role::Filler && sw.secret.is_none() && sw.scan_from > 0 && !ids.is_empty() {
        let (ws, reached) = c.mds_spends(&ids, sw.scan_from);
        found_spend = !ws.is_empty();
        sw.secret = ws.iter().find_map(|w| sw.secret_from_witness(w));
        sw.scan_from = sw.scan_from.max(reached);
        search_done = sw.scan_from >= tip;
    }
    // A lock spent before we ever saw it whole was still there once.
    if sw.lock_seen.is_none() && !ids.is_empty() && (found_spend || sw.secret.is_some() && sw.role == Role::Filler) {
        sw.lock_seen = Some(tip);
    }
    let lock = if whole {
        if depth_ok(tip, sw.lock_seen.unwrap_or(tip)) {
            Lock::Confirmed
        } else {
            Lock::Unconfirmed
        }
    } else if sw.lock_seen.is_some() {
        Lock::Spent
    } else {
        Lock::Absent
    };
    let lock_pending = match (sw.lock_commitment, ids.first()) {
        (Some(cm), Some(first)) if lock == Lock::Absent => c.mds_lock_pending(&cm, first),
        _ => false,
    };

    // Our spend's outputs are unique to it, so any one of them on chain
    // means it landed, even if the wallet has since spent others.
    let spend_landed = match (sw.spend_commitment, sw.dest) {
        (Some(_), Some(dest)) => {
            let kind = match sw.role {
                Role::Buyer | Role::Bidder => SpendKind::Claim,
                Role::Seller | Role::Filler => SpendKind::Refund,
            };
            build_spend(sw, kind, &dest).is_ok_and(|sp| sp.output_ids().iter().any(|id| c.mds_coin(id)))
        }
        _ => false,
    };

    let base_now = c.base_now();
    let base_time = now.max(base_now);
    let mut base_lock = None;
    let mut ignored_locks = 0;
    if sw.sell_order() && sw.secret_hash != [0; 32] {
        let locks = c.base_locks(&sw.secret_hash);
        match sw.role {
            Role::Seller => {
                // A lock on the other currency's contract is not about this
                // sale at all, whatever its numbers: not counted, not taken.
                for l in locks.into_iter().filter(|l| l.asset == sw.asset) {
                    if base_lock.is_none() && lock_meets_terms(sw, &l, base_time) {
                        base_lock = Some(l);
                    } else if Some(&l) != base_lock.as_ref() {
                        ignored_locks += 1;
                    }
                }
                // Once claimed, the claimed lock is the swap's, whatever the clock says now.
                if let Some(id) = sw.base_swap {
                    if base_lock.as_ref().map(|l| l.swap_id) != Some(id) {
                        base_lock = c.base_locks(&sw.secret_hash).into_iter().find(|l| l.swap_id == id && l.asset == sw.asset);
                    }
                }
            }
            Role::Buyer => {
                base_lock = locks.into_iter().filter(|l| l.asset == sw.asset).find(|l| match sw.base_swap {
                    Some(id) => l.swap_id == id,
                    None => l.refund_to == sw.buyer_evm && l.beneficiary == sw.seller_evm,
                });
            }
            _ => {}
        }
    }
    let swap_id = sw.base_swap.or(base_lock.as_ref().map(|l| l.swap_id));
    let (mut base_claimed, mut base_refunded) = (false, false);
    if let Some(id) = swap_id {
        if let Some(s) = c.base_claim(&id) {
            base_claimed = true;
            if sw.role == Role::Buyer && sw.secret.is_none() && hash(&s) == sw.secret_hash {
                sw.secret = Some(s);
            }
        }
        base_refunded = c.base_refunded(&id);
    }
    let (bid, bid_claimed) = if sw.bid_id != [0; 32] { (c.bid(&sw.bid_id), c.bid_claimed(&sw.bid_id)) } else { (None, false) };
    Obs {
        tip,
        now,
        base_now,
        lock,
        left,
        lock_pending,
        search_done,
        spend_landed,
        base_lock,
        ignored_locks,
        base_claimed,
        base_refunded,
        bid,
        bid_claimed,
    }
}

// ── Deciding ────────────────────────────────────────────────────────────────

/// What the service should do for a swap now. Sends are idempotent, and
/// the service paces repeats of the same act.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Act {
    /// Seller or filler: fund our MDS lock (filler: with its announcement).
    LockMds,
    /// Seller: tell the buyer the lock's terms.
    Answer,
    /// Buyer: repeat the take.
    Take,
    /// Buyer: lock the ETH, refundable after `delay` seconds.
    LockEth { delay: u64 },
    /// Filler: reserve the bid.
    Reserve,
    /// Seller: claim the buyer's ETH lock, publishing the secret.
    ClaimEth { swap_id: [u8; 32] },
    /// Filler: collect the escrow with the secret.
    ClaimBid,
    /// Buyer or bidder: claim the MDS lock.
    ClaimMds,
    /// Buyer: take the ETH back.
    RefundEth { swap_id: [u8; 32] },
    /// Seller or filler: take the MDS back.
    RefundMds,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    pub phase: Option<Phase>,
    pub acts: Vec<Act>,
    pub note: Option<String>,
}

impl Decision {
    fn wait() -> Self {
        Self::default()
    }
    fn to(phase: Phase) -> Self {
        Self { phase: Some(phase), ..Self::default() }
    }
    fn act(a: Act) -> Self {
        Self { acts: vec![a], ..Self::default() }
    }
    fn fail(why: &str) -> Self {
        Self::to(Phase::Failed(why.into()))
    }
    fn and(mut self, a: Act) -> Self {
        self.acts.push(a);
        self
    }
    fn note(mut self, n: String) -> Self {
        self.note = Some(n);
        self
    }
}

/// The next step for `sw`, from what `o` saw.
pub fn decide(sw: &EthSwap, o: &Obs) -> Decision {
    if sw.phase.settled() {
        return Decision::wait();
    }
    match sw.role {
        Role::Seller => decide_seller(sw, o),
        Role::Buyer => decide_buyer(sw, o),
        Role::Filler => decide_filler(sw, o),
        Role::Bidder => decide_bidder(sw, o),
    }
}

fn mds_secs_left(sw: &EthSwap, tip: u64) -> u64 {
    blocks_to_secs_pessimistic(sw.timeout.saturating_sub(tip))
}

fn decide_seller(sw: &EthSwap, o: &Obs) -> Decision {
    match &sw.phase {
        Phase::LockingMine => match o.lock {
            Lock::Confirmed => Decision::to(Phase::MineLocked).and(Act::Answer),
            Lock::Unconfirmed => Decision::act(Act::Answer),
            Lock::Spent => Decision::to(Phase::MineLocked),
            Lock::Absent if sw.lock_commitment.is_none() => Decision::act(Act::LockMds),
            Lock::Absent if o.tip >= sw.timeout => Decision::fail("our lock never reached the chain"),
            Lock::Absent => Decision::wait(),
        },
        Phase::MineLocked => {
            let ignored = (o.ignored_locks > 0).then(|| {
                format!("ignored {} {} lock(s) that did not meet this sale's terms", o.ignored_locks, sw.asset.symbol())
            });
            // Claim only while the buyer would still have the settle margin's
            // worth of time to claim the MDS before we could take it back.
            if let Some(l) = &o.base_lock {
                if mds_secs_left(sw, o.tip) >= SETTLE_MARGIN_SECS / 2 && o.lock != Lock::Spent {
                    let d = Decision::to(Phase::Claiming).and(Act::ClaimEth { swap_id: l.swap_id });
                    return match ignored {
                        Some(n) => d.note(n),
                        None => d,
                    };
                }
            }
            if o.tip >= sw.timeout && !sw.claim_sent {
                return Decision::to(Phase::Refunding).and(Act::RefundMds);
            }
            let d = if o.base_lock.is_none() { Decision::act(Act::Answer) } else { Decision::wait() };
            match ignored {
                Some(n) => d.note(n),
                None => d,
            }
        }
        Phase::Claiming => {
            if o.base_claimed {
                // Paid. The lock stays for the buyer, however long it takes
                // them: a seller that has been paid never takes it back.
                return Decision::to(Phase::Done);
            }
            if o.base_refunded {
                // Our claim never landed and the buyer has its ETH back. The
                // secret may have been seen in our claim, so the MDS comes
                // back the moment the lock allows.
                if o.tip >= sw.timeout {
                    return Decision::to(Phase::Refunding)
                        .and(Act::RefundMds)
                        .note("our claim did not land before the buyer's refund".into());
                }
                return Decision::wait().note("our claim did not land before the buyer's refund; taking the MDS back at the lock's end".into());
            }
            match &o.base_lock {
                // Retry only while a first sending would still be safe: if
                // the first never reached anyone, this one publishes the
                // secret, and must land before the buyer may refund.
                Some(l) if l.timeout >= o.base_time() + CLAIM_MARGIN_SECS + CLOCK_SKEW_SECS => {
                    Decision::act(Act::ClaimEth { swap_id: l.swap_id })
                }
                _ => Decision::wait(),
            }
        }
        Phase::Refunding => {
            if o.spend_landed {
                Decision::to(Phase::Refunded)
            } else if refund_open(sw, o) {
                Decision::act(Act::RefundMds)
            } else {
                Decision::fail("the lock was spent, but not by our refund")
            }
        }
        _ => Decision::wait(),
    }
}

/// Whether our refund can still land: the coins it was planned for are all
/// there, or, before it is planned, what is left can pay for one.
fn refund_open(sw: &EthSwap, o: &Obs) -> bool {
    if sw.refund_mask != 0 {
        o.left & sw.refund_mask == sw.refund_mask
    } else {
        sw.refundable(o.left)
    }
}

fn decide_buyer(sw: &EthSwap, o: &Obs) -> Decision {
    match &sw.phase {
        Phase::Asking => {
            if sw.secret_hash != [0; 32] && sw.timeout != 0 {
                Decision::to(Phase::WaitingTheirLock)
            } else if o.now >= sw.created + TAKE_GIVE_UP_SECS {
                Decision::fail("the seller did not answer")
            } else {
                Decision::act(Act::Take)
            }
        }
        Phase::WaitingTheirLock => match o.lock {
            Lock::Confirmed => match buyer_delay(o.tip, sw.timeout) {
                Some(delay) => Decision::to(Phase::LockingMine).and(Act::LockEth { delay }),
                None => Decision::fail("the seller's lock has too little time left to pay against safely"),
            },
            Lock::Spent => Decision::fail("the seller's lock was spent before we paid"),
            _ if o.tip >= sw.timeout || o.now >= sw.created + LOCK_WAIT_SECS => {
                Decision::fail("the seller's lock did not appear in time; nothing was paid")
            }
            _ => Decision::wait(),
        },
        Phase::LockingMine => {
            // Our lock is known from its receipt, or found by its event if
            // the receipt was lost.
            if o.base_lock.is_some() || (sw.base_swap.is_some() && sw.eth_deadline > 0) {
                return Decision::to(Phase::MineLocked);
            }
            // Signed: the service finds it by its hash, or shows by its
            // nonce that it can never be mined.
            if sw.eth_tx.is_some() {
                return Decision::wait();
            }
            // Not signed yet: only while paying is still safe.
            match buyer_delay(o.tip, sw.timeout) {
                Some(delay) if o.lock == Lock::Confirmed => Decision::act(Act::LockEth { delay }),
                _ => Decision::fail("the seller's lock has too little time left to pay against safely; nothing was paid"),
            }
        }
        Phase::MineLocked => {
            if sw.secret.is_some() {
                return Decision::to(Phase::Claiming).and(Act::ClaimMds);
            }
            if o.base_refunded {
                return Decision::to(Phase::Refunded);
            }
            let Some((swap_id, deadline)) = buyer_lock(sw, o) else { return Decision::wait() };
            if o.base_time() >= deadline + REFUND_GRACE_SECS {
                return Decision::to(Phase::Refunding).and(Act::RefundEth { swap_id });
            }
            Decision::wait()
        }
        Phase::Claiming => {
            if o.spend_landed {
                Decision::to(Phase::Done)
            } else if o.lock == Lock::Spent && sw.spend_commitment.is_none() {
                Decision::fail("the seller took the MDS back before we claimed it")
            } else {
                Decision::act(Act::ClaimMds)
            }
        }
        Phase::Refunding => {
            if sw.secret.is_some() {
                // The seller claimed after all: the refund was refused and
                // the claim found. Collect the MDS.
                return Decision::to(Phase::Claiming).and(Act::ClaimMds);
            }
            if o.base_refunded {
                return Decision::to(Phase::Refunded);
            }
            match buyer_lock(sw, o) {
                Some((swap_id, _)) => Decision::act(Act::RefundEth { swap_id }),
                None => Decision::wait(),
            }
        }
        _ => Decision::wait(),
    }
}

/// The buyer's ETH lock: its id and refund time, from the event if seen,
/// else from the receipt the journal kept.
fn buyer_lock(sw: &EthSwap, o: &Obs) -> Option<([u8; 32], u64)> {
    match (&o.base_lock, sw.base_swap) {
        (Some(l), _) => Some((l.swap_id, l.timeout)),
        (None, Some(id)) if sw.eth_deadline > 0 => Some((id, sw.eth_deadline)),
        _ => None,
    }
}

fn decide_filler(sw: &EthSwap, o: &Obs) -> Decision {
    match &sw.phase {
        Phase::Reserving => {
            // The service plans the lock once the reservation is ours.
            match &o.bid {
                Some(b) if b.settled => Decision::fail("the buy order was closed before our reservation"),
                Some(b) if b.filler != [0; 20] && b.filler != sw.seller_evm => {
                    Decision::fail("someone else reserved the buy order first; nothing was locked")
                }
                _ if sw.eth_tx.is_none() => Decision::act(Act::Reserve),
                _ => Decision::wait(),
            }
        }
        Phase::LockingMine => {
            if sw.secret.is_some() {
                // Claimed before we ever saw it: collect.
                return Decision::to(Phase::Claiming).and(Act::ClaimBid);
            }
            match o.lock {
                Lock::Confirmed | Lock::Spent => Decision::to(Phase::MineLocked),
                Lock::Unconfirmed => Decision::wait(),
                Lock::Absent if sw.lock_commitment.is_none() => {
                    // Lock only inside the reservation, and only while it
                    // leaves the planned margin.
                    let need = SETTLE_MARGIN_SECS + FILL_SLACK_SECS + blocks_to_secs_optimistic(sw.timeout.saturating_sub(o.tip));
                    if o.base_time() + need > sw.eth_deadline {
                        Decision::fail("the reservation ran too short to lock against; the bond stays with the buyer")
                    } else {
                        Decision::act(Act::LockMds)
                    }
                }
                // Still on its way, or not yet looked for everywhere it
                // could have been spent.
                Lock::Absent if o.lock_pending || !o.search_done => Decision::wait(),
                Lock::Absent if o.tip >= sw.timeout => Decision::fail("our lock never reached the chain; nothing was locked"),
                Lock::Absent => Decision::wait(),
            }
        }
        Phase::MineLocked => {
            if sw.secret.is_some() {
                return Decision::to(Phase::Claiming).and(Act::ClaimBid);
            }
            if o.tip >= sw.timeout && sw.refundable(o.left) {
                return Decision::to(Phase::Refunding).and(Act::RefundMds);
            }
            if o.lock == Lock::Spent {
                return Decision::wait().note("the buyer claimed; reading the secret off its claim".into());
            }
            Decision::wait()
        }
        Phase::Claiming => {
            if o.bid_claimed {
                // Paid. Whatever the bidder left in the lock comes back
                // once it ends.
                if o.spend_landed || !refund_open(sw, o) {
                    return Decision::to(Phase::Done);
                }
                if o.tip >= sw.timeout {
                    return Decision::act(Act::RefundMds).note("paid; taking back the MDS the buyer left in the lock".into());
                }
                return Decision::wait().note("paid; the buyer left part of the lock, which comes back when it ends".into());
            }
            match &o.bid {
                // Settled, but no claim seen yet: ours, read late, or the
                // buyer's cancel after the reservation lapsed.
                Some(b) if b.settled => {
                    if o.base_time() > sw.eth_deadline + 600 {
                        Decision::fail("the reservation lapsed before our claim of the escrow landed")
                    } else {
                        Decision::wait()
                    }
                }
                _ if o.base_time() > sw.eth_deadline + CLOCK_SKEW_SECS => {
                    Decision::fail("the reservation ended before our claim of the escrow landed")
                }
                _ => Decision::act(Act::ClaimBid),
            }
        }
        Phase::Refunding => {
            if sw.secret.is_some() {
                // The bidder claimed at the last moment: collect the escrow.
                return Decision::to(Phase::Claiming).and(Act::ClaimBid);
            }
            if o.spend_landed {
                Decision::to(Phase::Refunded)
            } else if refund_open(sw, o) {
                Decision::act(Act::RefundMds)
            } else if !o.search_done {
                Decision::wait() // a claim took what we planned to refund: its secret is on its way
            } else {
                Decision::fail("the lock was spent, and no claim of it showed the secret")
            }
        }
        _ => Decision::wait(),
    }
}

fn decide_bidder(sw: &EthSwap, o: &Obs) -> Decision {
    // Claiming publishes our secret; do it only while the seller holds the
    // reservation with time to collect, and our claim can land first.
    let reservation = |o: &Obs| -> std::result::Result<(), Decision> {
        let Some(b) = &o.bid else { return Err(Decision::wait()) };
        if b.settled {
            return Err(Decision::fail("the buy order is closed"));
        }
        if b.filler == [0; 20] {
            return Err(Decision::wait().note("waiting for the seller's reservation on Base".into()));
        }
        if b.fill_deadline < o.base_time() + SETTLE_MARGIN_SECS / 2 {
            return Err(Decision::fail("the seller's reservation ends too soon for them to collect; not claiming"));
        }
        Ok(())
    };
    let blocks_left = sw.timeout.saturating_sub(o.tip);
    match &sw.phase {
        Phase::WaitingTheirLock => match o.lock {
            Lock::Spent => Decision::fail("the seller took the MDS back"),
            _ if blocks_left < BIDDER_CLAIM_BLOCKS => Decision::fail("the seller's lock is too close to its end to claim"),
            Lock::Confirmed => match reservation(o) {
                Ok(()) => Decision::to(Phase::Claiming).and(Act::ClaimMds),
                Err(d) => d,
            },
            _ => Decision::wait(),
        },
        Phase::Claiming => {
            if o.spend_landed {
                return Decision::to(Phase::Done);
            }
            if sw.reveal_sent {
                // The secret is out: see the claim through.
                if o.left == 0 {
                    return Decision::fail("the lock was spent, and not by our claim");
                }
                return Decision::act(Act::ClaimMds);
            }
            // Nothing is public yet: stop rather than reveal late.
            if o.lock == Lock::Spent {
                return Decision::fail("the seller took the MDS back");
            }
            if blocks_left < REVEAL_MIN_BLOCKS {
                return Decision::fail("too close to the lock's end to claim safely; the secret was not published");
            }
            match reservation(o) {
                Ok(()) => Decision::act(Act::ClaimMds),
                Err(d) => d,
            }
        }
        _ => Decision::wait(),
    }
}

// ── The store ───────────────────────────────────────────────────────────────

/// Everything walletd keeps about ETH trading, in `WalletData::eth_state`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EthStore {
    pub orders: Vec<EthOrder>,
    pub swaps: Vec<EthSwap>,
    /// Highest Base block whose events have been read.
    #[serde(default)]
    pub base_cursor: u64,
}

impl EthStore {
    pub fn load(v: Option<&serde_json::Value>) -> Result<Self> {
        match v {
            Some(v) => Ok(serde_json::from_value(v.clone())?),
            None => Ok(Self::default()),
        }
    }
    pub fn to_value(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }
    pub fn swap_mut(&mut self, id: &[u8; 16]) -> Option<&mut EthSwap> {
        self.swaps.iter_mut().find(|s| &s.id == id)
    }
    pub fn order_mut(&mut self, id: &[u8; 16]) -> Option<&mut EthOrder> {
        self.orders.iter_mut().find(|o| &o.id == id)
    }
    /// Swaps of `order` whose lock is out but whose ETH has not been seen.
    pub fn unfunded(&self, order: &[u8; 16]) -> usize {
        self.swaps
            .iter()
            .filter(|s| s.role == Role::Seller && &s.order_id == order && matches!(s.phase, Phase::LockingMine | Phase::MineLocked) && !s.claim_sent)
            .count()
    }
    /// Keeps the journal from growing without bound: settled swaps beyond
    /// the newest hundred, and orders that have ended with nothing running.
    pub fn compact(&mut self) {
        let settled: Vec<usize> = self.swaps.iter().enumerate().filter(|(_, s)| s.phase.settled()).map(|(i, _)| i).collect();
        if settled.len() > 100 {
            let drop: std::collections::HashSet<usize> = settled[..settled.len() - 100].iter().copied().collect();
            let mut i = 0;
            self.swaps.retain(|_| {
                let keep = !drop.contains(&i);
                i += 1;
                keep
            });
        }
        let live: std::collections::HashSet<[u8; 16]> = self.swaps.iter().filter(|s| !s.phase.settled()).map(|s| s.order_id).collect();
        self.orders.retain(|o| o.status == OrderStatus::Open || live.contains(&o.id) || o.created + 30 * 86_400 > now_hint());
    }
}

fn now_hint() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order() -> EthOrder {
        EthOrder {
            id: [1; 16],
            mds: 10 << 30,
            min: 1 << 30,
            wei_per_gmds: 3_000_000_000_000_000,
            expires: 500_000,
            seller_evm: [2; 20],
            status: OrderStatus::Open,
            created: 0,
            last_advert: 0,
            sold: 0,
            asset: Asset::Eth,
        }
    }

    #[test]
    fn prices_round_up_and_takes_are_checked() {
        use crate::eth_wire::Decline;
        assert_eq!(price_of(1 << 30, 3), Some(3));
        assert_eq!(price_of(1, 3), Some(1)); // never free
        let o = order();
        let eth = Asset::Eth;
        assert_eq!(check_take(&o, 1, 2 << 30, 6_000_000_000_000_000, eth), Ok(()));
        assert_eq!(check_take(&o, 1, 2 << 30, 5_999_999_999_999_999, eth), Err(Decline::Price));
        assert_eq!(check_take(&o, 1, 11 << 30, u128::MAX, eth), Err(Decline::Amount));
        assert_eq!(check_take(&o, 1, 1 << 20, u128::MAX, eth), Err(Decline::Amount));
        assert_eq!(check_take(&o, 500_000, 2 << 30, u128::MAX, eth), Err(Decline::Gone));
        // A take in the other currency asks for an order that is not on offer,
        // however much it offers.
        assert_eq!(check_take(&o, 1, 2 << 30, u128::MAX, Asset::Usdc), Err(Decline::Gone));
        let usdc = EthOrder { asset: Asset::Usdc, wei_per_gmds: 2_500_000, ..order() };
        assert_eq!(check_take(&usdc, 1, 2 << 30, 5_000_000, Asset::Usdc), Ok(()));
        assert_eq!(check_take(&usdc, 1, 2 << 30, 4_999_999, Asset::Usdc), Err(Decline::Price));
        assert_eq!(check_take(&usdc, 1, 2 << 30, 5_000_000, Asset::Eth), Err(Decline::Gone));
    }

    #[test]
    fn amounts_read_in_their_own_currency() {
        assert_eq!(Asset::Usdc.format(12_500_000, 6), "12.5 USDC");
        assert_eq!(Asset::Usdc.format(1, 6), "0.000001 USDC");
        assert_eq!(Asset::Usdc.format(3_000_000, 2), "3 USDC");
        assert_eq!(Asset::Eth.format(1_500_000_000_000_000_000, 8), "1.5 ETH");
        assert_eq!(Asset::Eth.format(1, 8), "0 ETH", "below what is shown");
        assert_eq!(Asset::parse("usdc").unwrap(), Asset::Usdc);
        assert_eq!(Asset::parse("").unwrap(), Asset::Eth, "requests from before USDC are ETH");
        assert!(Asset::parse("DAI").is_err());
        // Journals from before USDC load as ETH.
        let mut old = serde_json::to_value(order()).unwrap();
        old.as_object_mut().unwrap().remove("asset");
        assert_eq!(serde_json::from_value::<EthOrder>(old).unwrap().asset, Asset::Eth);
        let mut sw = serde_json::to_value(EthSwap::new([1; 16], Role::Buyer, Phase::Asking, 0)).unwrap();
        sw.as_object_mut().unwrap().remove("asset");
        assert_eq!(serde_json::from_value::<EthSwap>(sw).unwrap().asset, Asset::Eth);
    }

    #[test]
    fn a_lock_in_the_other_currency_is_never_the_lock_a_trade_waits_for() {
        let mut sw = EthSwap::new([1; 16], Role::Seller, Phase::MineLocked, 0);
        sw.asset = Asset::Usdc;
        sw.wei = 5_000_000;
        sw.seller_evm = [2; 20];
        sw.buyer_evm = [3; 20];
        let l = BaseLock { swap_id: [9; 32], beneficiary: [2; 20], refund_to: [3; 20], amount: 5_000_000, timeout: 10_000, block: 1, asset: Asset::Usdc };
        assert!(lock_meets_terms(&sw, &l, 0));
        // The same numbers in wei: a few millionths of a cent, not five dollars.
        assert!(!lock_meets_terms(&sw, &BaseLock { asset: Asset::Eth, ..l.clone() }, 0));
        sw.asset = Asset::Eth;
        assert!(!lock_meets_terms(&sw, &l, 0));
    }

    #[test]
    fn a_filler_risks_a_bond_of_at_most_a_tenth() {
        assert!(bond_acceptable(0, 0));
        assert!(bond_acceptable(100, 1_000));
        assert!(!bond_acceptable(101, 1_000));
        assert!(!bond_acceptable(1, 0));
    }

    #[test]
    fn tradeable_amounts_fit_sixteen_coins() {
        let v = tradeable_amount(u64::MAX >> 1);
        assert_eq!(v.count_ones(), 16);
        assert!(v <= u64::MAX >> 1);
        assert_eq!(tradeable_amount(3 << 30), 3 << 30);
    }

    #[test]
    fn deadlines_keep_the_settle_margin_with_fast_blocks() {
        let tip = 400_000;
        let t = seller_timeout(tip);
        let d = buyer_delay(tip, t).unwrap();
        assert_eq!(d, DEFAULT_ETH_DELAY);
        // Fast blocks: the seller's refund arrives at its pessimistic time.
        assert!(ETH_MINE_SECS + d + SETTLE_MARGIN_SECS <= blocks_to_secs_pessimistic(t - tip));
        // Late buyers get shorter locks, then none.
        assert!(buyer_delay(t - 120, t).is_some_and(|d| d < DEFAULT_ETH_DELAY));
        assert_eq!(buyer_delay(t - 90, t), None);
        // A fill's lock ends with the margin to spare even with slow blocks.
        let now = 1_000_000;
        let deadline = now + FILL_WINDOW_SECS;
        let ft = filler_timeout(tip, now, deadline).unwrap();
        assert!(now + blocks_to_secs_optimistic(ft - tip) + SETTLE_MARGIN_SECS <= deadline);
        assert_eq!(filler_timeout(tip, now, now + 3_600), None);
    }

    #[test]
    fn the_journal_round_trips_with_large_amounts() {
        let mut s = EthStore::default();
        let mut o = order();
        o.wei_per_gmds = u128::MAX / 3;
        s.orders.push(o);
        let mut sw = EthSwap::new([3; 16], Role::Buyer, Phase::Asking, 5);
        sw.wei = u128::MAX - 7;
        s.swaps.push(sw);
        let back = EthStore::load(Some(&s.to_value().unwrap())).unwrap();
        assert_eq!(back, s);
    }
}
