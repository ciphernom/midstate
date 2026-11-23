//! The walletd actor's side of trading MDS for ETH and USDC on Base: sell
//! orders on the chat bus, reading Base, and carrying out what `eth_trade`
//! decides. Both contracts are read into one index, every lock tagged with
//! the currency of the contract it came from; each act goes to the contract
//! of its swap's currency. USDC is paid in with a permit for exactly what is
//! paid, in the same transaction, one at a time (a permit takes the token's
//! next nonce, so a second signed before the first is mined would fail).
//!
//! A child of `service`, so it shares the actor's state. Everything that
//! decides lives in `eth_trade`; this module only reads, moves messages and
//! broadcasts.

use super::{now_secs, SendMeta, Service};
use crate::api::*;
use crate::base::{BaseClient, BidState, Event, Settlement};
use crate::eth_trade::*;
use crate::eth_wire::{claim_message, parse, Advert, Answer, Decline, Frame, NameClaim, Take, CLAIM_ORDER, CLAIM_TAKE};
use crate::sendplan;
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use midstate::core::Transaction;
use std::collections::{HashMap, HashSet, VecDeque};

/// Midstate blocks a search for a lock's spend reads in one tick.
const SEARCH_PER_TICK: u64 = 256;
/// Base blocks read per tick at most, so a wallet back after days catches
/// up over a few ticks instead of holding the actor for minutes.
const BASE_PER_TICK: u64 = 20_000;
/// Base blocks read back on first sight, before any swap names a start.
const BASE_LOOKBACK: u64 = 5_000;
/// Seconds between reads of a bid's state.
const BID_REFRESH_SECS: u64 = 30;
/// Seconds between repeats of a Base transaction that has not shown up.
const BASE_RETRY_SECS: u64 = 60;
/// Midstate blocks before an external commit that has not appeared is made again.
const RECOMMIT_BLOCKS: u64 = 20;
/// Events kept for swaps; beyond this, those no live swap needs are dropped.
const INDEX_CAP: usize = 20_000;

/// What has been read off Base for the swaps.
#[derive(Default)]
pub(super) struct BaseIndex {
    locks: HashMap<[u8; 32], Vec<BaseLock>>,
    claims: HashMap<[u8; 32], [u8; 32]>,
    refunds: HashSet<[u8; 32]>,
    bids: HashMap<[u8; 32], (BidState, u64)>,
    /// Bids whose escrow was claimed: paid to their reservation's holder.
    bid_claims: HashSet<[u8; 32]>,
    /// Highest Base block read; 0 until the first read.
    cursor: u64,
    /// The same, for the USDC contract.
    cursor_usdc: u64,
    /// Base's tip at the last read, and its timestamp.
    pub(super) head: u64,
    head_time: u64,
}

impl BaseIndex {
    /// Folds in an event read from `asset`'s contract.
    fn add(&mut self, block: u64, e: &Event, asset: Asset) {
        match e {
            Event::Locked { swap_id, beneficiary, refund_to, amount, timeout, hashlock } => {
                let v = self.locks.entry(*hashlock).or_default();
                if !v.iter().any(|l| &l.swap_id == swap_id) {
                    v.push(BaseLock { swap_id: *swap_id, beneficiary: *beneficiary, refund_to: *refund_to, amount: *amount, timeout: *timeout, block, asset });
                }
            }
            Event::Claimed { swap_id, preimage, .. } => {
                self.claims.insert(*swap_id, *preimage);
            }
            Event::Refunded { swap_id } => {
                self.refunds.insert(*swap_id);
            }
            Event::BidClaimed { bid_id, .. } => {
                self.bid_claims.insert(*bid_id);
            }
            _ => {}
        }
    }

    /// Drops what no unsettled swap can ask about.
    fn prune(&mut self, swaps: &[EthSwap]) {
        if self.locks.len() + self.claims.len() + self.refunds.len() + self.bid_claims.len() < INDEX_CAP {
            return;
        }
        let live: Vec<&EthSwap> = swaps.iter().filter(|s| !s.phase.settled()).collect();
        let hashes: HashSet<[u8; 32]> = live.iter().map(|s| s.secret_hash).collect();
        self.locks.retain(|h, _| hashes.contains(h));
        let ids: HashSet<[u8; 32]> = self.locks.values().flatten().map(|l| l.swap_id).chain(live.iter().filter_map(|s| s.base_swap)).collect();
        self.claims.retain(|id, _| ids.contains(id));
        self.refunds.retain(|id| ids.contains(id));
        let bids: HashSet<[u8; 32]> = live.iter().map(|s| s.bid_id).collect();
        self.bid_claims.retain(|id| bids.contains(id));
    }
}

/// The actor's ETH-trading state.
#[derive(Default)]
pub(super) struct EthLive {
    pub(super) store: EthStore,
    /// Whether `store` was loaded from an unlocked wallet.
    pub(super) loaded: bool,
    pub(super) book: HashMap<[u8; 16], BookEntry>,
    pub(super) index: BaseIndex,
    seen: HashSet<(String, u64, u64)>,
    seen_order: VecDeque<(String, u64, u64)>,
    /// When each swap last did each kind of act (pacing only).
    pushed: HashMap<([u8; 16], u8), u64>,
    /// When each settled-but-unexplained Base swap was last looked up.
    lookups: HashMap<[u8; 32], u64>,
    pub(super) error: Option<String>,
    /// Name claims heard, by order or take id: the name, and the Base
    /// account that signed the claim.
    names: HashMap<[u8; 16], (String, [u8; 20])>,
}

impl EthLive {
    fn first_sight(&mut self, key: (String, u64, u64)) -> bool {
        if !self.seen.insert(key.clone()) {
            return false;
        }
        self.seen_order.push_back(key);
        while self.seen_order.len() > 4_000 {
            if let Some(old) = self.seen_order.pop_front() {
                self.seen.remove(&old);
            }
        }
        true
    }

    /// Whether act `what` of swap `id` is due again, at most once per
    /// `every` seconds; marks it done if so.
    fn due(&mut self, id: [u8; 16], what: u8, now: u64, every: u64) -> bool {
        let last = self.pushed.get(&(id, what)).copied().unwrap_or(0);
        if last != 0 && now < last + every {
            return false;
        }
        self.pushed.insert((id, what), now);
        true
    }

    /// Whether the settlement of Base swap `id` may be looked up again: a
    /// lookup reads a range of logs, so it is paced.
    pub(super) fn lookup_due(&mut self, id: [u8; 32], now: u64) -> bool {
        let last = self.lookups.get(&id).copied().unwrap_or(0);
        if last != 0 && now < last + LOOKUP_SECS {
            return false;
        }
        self.lookups.insert(id, now);
        true
    }
}

/// Seconds between lookups of how one Base swap ended.
const LOOKUP_SECS: u64 = 300;

/// Both chains as one tick sees them.
struct LiveEth<'a> {
    mds: midstate::core::State,
    mempool: Vec<Transaction>,
    node: &'a midstate::node::NodeHandle,
    now: u64,
    index: &'a BaseIndex,
    /// The wallet's sends still under way.
    pending: HashSet<[u8; 32]>,
}

impl Chains for LiveEth<'_> {
    fn mds_tip(&self) -> u64 {
        self.mds.height
    }
    fn mds_coin(&self, id: &[u8; 32]) -> bool {
        self.mds.coins.contains(id)
    }
    fn mds_spends(&self, coins: &[[u8; 32]], from: u64) -> (Vec<Vec<Vec<u8>>>, u64) {
        let mut out = Vec::new();
        let mut take = |tx: &Transaction| {
            let Transaction::Reveal { inputs, witnesses, .. } = tx else { return };
            for (i, input) in inputs.iter().enumerate() {
                if coins.contains(&input.coin_id()) {
                    if let Some(midstate::core::types::Witness::ScriptInputs(items)) = witnesses.get(i) {
                        out.push(items.clone());
                    }
                }
            }
        };
        self.mempool.iter().for_each(&mut take);
        // Blocks are stored at heights below the state's.
        let end = self.mds.height.min(from.saturating_add(SEARCH_PER_TICK));
        for h in from..end {
            match self.node.storage.batches.load(h) {
                Ok(Some(batch)) => batch.transactions.iter().for_each(&mut take),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("midstate block {h} unreadable: {e:#}");
                    return (out, h);
                }
            }
        }
        (out, end.max(from))
    }
    fn mds_lock_pending(&self, commitment: &[u8; 32], coin: &[u8; 32]) -> bool {
        self.pending.contains(commitment)
            || self.mempool.iter().any(|tx| match tx {
                Transaction::Reveal { outputs, .. } => outputs.iter().any(|o| o.coin_id().as_ref() == Some(coin)),
                _ => false,
            })
    }
    fn now(&self) -> u64 {
        self.now
    }
    fn base_now(&self) -> u64 {
        self.index.head_time
    }
    fn base_locks(&self, hashlock: &[u8; 32]) -> Vec<BaseLock> {
        self.index.locks.get(hashlock).cloned().unwrap_or_default()
    }
    fn base_claim(&self, swap_id: &[u8; 32]) -> Option<[u8; 32]> {
        self.index.claims.get(swap_id).copied()
    }
    fn base_refunded(&self, swap_id: &[u8; 32]) -> bool {
        self.index.refunds.contains(swap_id)
    }
    fn bid(&self, bid_id: &[u8; 32]) -> Option<BidState> {
        self.index.bids.get(bid_id).map(|(b, _)| b.clone())
    }
    fn bid_claimed(&self, bid_id: &[u8; 32]) -> bool {
        self.index.bid_claims.contains(bid_id)
    }
}

fn kind_name(r: Role) -> &'static str {
    match r {
        Role::Buyer => "buy",
        Role::Seller => "sell",
        Role::Filler => "fill",
        Role::Bidder => "bid",
    }
}

/// Act kinds, for pacing.
const P_ANSWER: u8 = 1;
const P_TAKE: u8 = 2;
const P_BASE: u8 = 3;
const P_REVEAL: u8 = 4;
const P_COMMIT: u8 = 5;
const P_CANCEL: u8 = 6;

impl Service {
    // ── Lifecycle ───────────────────────────────────────────────────────

    pub(super) fn eth_load(&mut self) {
        let Some(w) = self.wallet.as_ref() else { return };
        self.eth.store = match EthStore::load(w.data.eth_state.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                let _ = self.events.send(WalletEvent::Warning {
                    text: format!("The ETH trading records could not be read ({e:#}); open trades need attention."),
                });
                EthStore::default()
            }
        };
        self.eth.loaded = true;
        // Events are not kept across a restart: read again from where the
        // oldest live swap's start, so none of them misses its lock or claim.
        self.eth.index = BaseIndex::default();
        let from = |asset: Asset| self.eth.store.swaps.iter().filter(|s| !s.phase.settled() && s.base_from > 0 && s.asset == asset).map(|s| s.base_from).min();
        if let Some(f) = from(Asset::Eth) {
            self.eth.index.cursor = f.saturating_sub(1);
        }
        if let Some(f) = from(Asset::Usdc) {
            self.eth.index.cursor_usdc = f.saturating_sub(1);
        }
    }

    pub(super) fn eth_unload(&mut self) {
        self.eth = EthLive::default();
    }

    pub(super) fn eth_save(&mut self) -> Result<()> {
        if !self.eth.loaded {
            return Ok(());
        }
        self.eth.store.compact();
        let v = self.eth.store.to_value()?;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        w.data.eth_state = Some(v);
        w.save()
    }

    /// A client for the contract of `asset`: ETH's, or USDC's.
    fn eth_client(&self, asset: Asset) -> Result<BaseClient> {
        BaseClient::for_asset(self.dex_cfg.clone(), asset)
    }

    /// A USDC payment of ours that may not be mined yet: a lock or a
    /// reservation whose receipt has not come. Its permit took the token's
    /// next nonce, so another permit waits for it.
    pub(super) fn token_payment_in_flight(&self) -> bool {
        self.eth.store.swaps.iter().any(|s| {
            s.asset == Asset::Usdc
                && s.eth_pending.is_some()
                && match (&s.role, &s.phase) {
                    (Role::Buyer, Phase::LockingMine) => s.base_swap.is_none(),
                    (Role::Filler, Phase::Reserving) => true,
                    _ => false,
                }
        }) || self.book.bids.iter().any(|b| b.asset == Asset::Usdc && b.bid_id.is_empty() && !b.tx.is_empty() && !b.cancelled)
    }

    fn eth_address(&self) -> Result<[u8; 20]> {
        Ok(self.evm_key()?.address)
    }

    // ── Commands ────────────────────────────────────────────────────────

    /// Puts `mds` up for sale for `asset` at `wei_per_gmds` (its smallest
    /// unit per 2^30 units), for `life` midstate blocks.
    pub(super) async fn eth_sell(&mut self, asset: Asset, mds: u64, wei_per_gmds: &str, min: u64, life: u64) -> Result<String> {
        ensure!(self.eth.loaded, "wallet is locked");
        ensure!(!self.node.is_syncing(), "the node is still syncing");
        if asset == Asset::Usdc {
            ensure!(self.dex_cfg.usdc_ready(), "USDC trading is not set up: this wallet knows no MidstateTokenSwap contract yet (Trade settings)");
        }
        let price: u128 = wei_per_gmds.trim().parse().map_err(|_| anyhow!("the price is not a whole number of the currency's smallest unit"))?;
        ensure!(price > 0, "the price must be more than zero");
        ensure!(mds >= MIN_MDS_AMOUNT, "the smallest sale is {MIN_MDS_AMOUNT} units");
        ensure!((MIN_ORDER_LIFE..=MAX_ORDER_LIFE).contains(&life), "an order runs {MIN_ORDER_LIFE} to {MAX_ORDER_LIFE} blocks");
        let seller_evm = self.eth_address().context("this wallet has no Base account to be paid at")?;
        let tip = self.node.get_state().await.height;
        let state = self.node.get_state().await;
        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let have: u64 = w.spendable_now(&state, &in_flight).iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum();
        ensure!(have >= mds, "this wallet can spend {have} units now, less than the {mds} for sale");
        let id: [u8; 16] = rand::random();
        self.eth.store.orders.push(EthOrder {
            id,
            mds,
            min: min.max(MIN_MDS_AMOUNT),
            wei_per_gmds: price,
            expires: tip + life,
            seller_evm,
            status: OrderStatus::Open,
            created: now_secs(),
            last_advert: 0,
            sold: 0,
            asset,
        });
        self.eth_save()?;
        self.eth_advertise(tip);
        Ok(hex::encode(id))
    }

    pub(super) fn eth_cancel_sell(&mut self, id: &str) -> Result<()> {
        let id: [u8; 16] = hex::decode(id).ok().and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("not an order id"))?;
        let o = self.eth.store.order_mut(&id).ok_or_else(|| anyhow!("no such order"))?;
        if o.status == OrderStatus::Open {
            o.status = OrderStatus::Cancelled;
        }
        self.eth_save()
    }

    /// Asks to buy `mds` from a heard sell order.
    pub(super) async fn eth_buy(&mut self, order_id: &str, mds: u64) -> Result<String> {
        ensure!(self.eth.loaded, "wallet is locked");
        ensure!(!self.node.is_syncing(), "the node is still syncing");
        let order_id: [u8; 16] = hex::decode(order_id).ok().and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("not an order id"))?;
        let a = self.eth.book.get(&order_id).map(|e| e.advert.clone()).ok_or_else(|| anyhow!("that order is no longer on offer"))?;
        let mds = tradeable_amount(mds);
        ensure!(mds >= a.min && mds <= a.mds, "this order sells between {} and {} units", a.min, a.mds);
        let wei = price_of(mds, a.wei_per_gmds).ok_or_else(|| anyhow!("the price overflows"))?;
        let key = self.evm_key()?;
        // Pay only what we can: the lock costs gas too, in ETH.
        let client = self.eth_client(a.asset)?;
        let have = client.balance(&key.address).await.context("could not read your Base balance")?;
        match a.asset {
            Asset::Eth => ensure!(have > wei, "this needs {} plus gas on Base; your account holds {}", Asset::Eth.format(wei, 8), Asset::Eth.format(have, 8)),
            Asset::Usdc => {
                ensure!(have > 0, "paying in USDC still takes a little ETH for gas on Base; your account holds none");
                let usdc = client.token_balance(&key.address).await.context("could not read your USDC balance")?;
                ensure!(usdc >= wei, "this needs {}; your Base account holds {}", Asset::Usdc.format(wei, 6), Asset::Usdc.format(usdc, 6));
            }
        }
        if let Some(s) = self.eth.store.swaps.iter().find(|s| s.role == Role::Buyer && s.order_id == order_id && !s.phase.settled()) {
            bail!("you are already buying from this order (trade {})", hex::encode(&s.id[..4]));
        }
        let id: [u8; 16] = rand::random();
        let seed: [u8; 32] = rand::random();
        let now = now_secs();
        let mut sw = EthSwap::new(id, Role::Buyer, Phase::Asking, now);
        sw.order_id = order_id;
        sw.mds_amount = mds;
        sw.lock_value = mds;
        sw.wei = wei;
        sw.seller_evm = a.seller_evm;
        sw.buyer_evm = key.address;
        sw.key_seed = Some(seed);
        sw.claim_pk = pk_of(&seed);
        sw.base_from = self.eth.index.head.saturating_sub(100);
        sw.asset = a.asset;
        self.eth.store.swaps.push(sw);
        self.eth_save()?;
        self.eth_send_take(self.eth.store.swaps.len() - 1, now);
        Ok(hex::encode(id))
    }

    /// Sells into a resting buy order: reserves it first.
    pub(super) async fn eth_fill_bid(&mut self, bid_id: &str) -> Result<String> {
        ensure!(self.eth.loaded, "wallet is locked");
        ensure!(!self.node.is_syncing(), "the node is still syncing");
        let bid_id: [u8; 32] = hex::decode(bid_id).ok().and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("not a buy order id"))?;
        let b = self.dex.bids.get(&bid_id).cloned().ok_or_else(|| anyhow!("that buy order is not in the book; refresh and try again"))?;
        let now = now_secs();
        ensure!(b.reserved_by.is_none(), "someone is already filling that buy order");
        ensure!(b.expiry > now + 600, "that buy order is about to expire");
        let key = self.evm_key()?;
        ensure!(b.maker != key.address, "that is your own buy order");
        ensure!(
            bond_acceptable(b.fill_bond, b.amount),
            "that buy order asks a bond of more than a tenth of what it pays; a buyer who never claims keeps the bond, so this wallet will not fill it"
        );
        let client = self.eth_client(b.asset)?;
        let have = client.balance(&key.address).await.context("could not read your Base balance")?;
        match b.asset {
            Asset::Eth => ensure!(
                have > b.fill_bond,
                "reserving needs the {} bond plus gas on Base; your account holds {}",
                Asset::Eth.format(b.fill_bond, 8),
                Asset::Eth.format(have, 8)
            ),
            Asset::Usdc => {
                ensure!(have > 0, "reserving takes a little ETH for gas on Base; your account holds none");
                if b.fill_bond > 0 {
                    let usdc = client.token_balance(&key.address).await.context("could not read your USDC balance")?;
                    ensure!(usdc >= b.fill_bond, "reserving needs the {} bond; your Base account holds {}", Asset::Usdc.format(b.fill_bond, 6), Asset::Usdc.format(usdc, 6));
                }
            }
        }
        if let Some(chain) = client.get_bid(bid_id).await? {
            ensure!(!chain.settled && chain.filler == [0; 20], "that buy order has been taken");
        }
        // The MDS, with room for the buyer's claim fee, must be there now.
        let seed: [u8; 32] = rand::random();
        let mut sw = EthSwap::new(bid_id[..16].try_into().unwrap(), Role::Filler, Phase::Reserving, now);
        sw.bid_id = bid_id;
        sw.mds_amount = b.mds_amount;
        sw.wei = b.amount;
        sw.fill_bond = b.fill_bond;
        sw.seller_evm = key.address;
        sw.buyer_evm = b.maker;
        sw.receiver_addr = b.maker_mds_addr;
        sw.secret_hash = b.hashlock;
        sw.key_seed = Some(seed);
        sw.refund_pk = pk_of(&seed);
        sw.base_from = self.eth.index.head.saturating_sub(100);
        sw.asset = b.asset;
        let tip = self.node.get_state().await.height;
        sw.timeout = tip + 200;
        let need = fill_total(b.mds_amount, &sw)?;
        let state = self.node.get_state().await;
        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let spendable: u64 = w.spendable_now(&state, &in_flight).iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum();
        ensure!(spendable > need, "filling this needs {need} units (the order plus the buyer's claim fee); this wallet can spend {spendable}");
        sw.timeout = 0;
        ensure!(!self.eth.store.swaps.iter().any(|s| s.bid_id == bid_id && !s.phase.settled()), "already filling this buy order");
        self.eth.store.swaps.push(sw);
        self.eth_save()?;
        Ok(hex::encode(bid_id))
    }

    pub(super) async fn eth_market(&self) -> EthMarketView {
        let tip = self.node.get_state().await.height;
        let mut sells: Vec<EthAdvertView> = self
            .eth
            .book
            .values()
            .map(|e| {
                let mut v = advert_view(&e.advert, false);
                v.seller_name = self.verified_name(&e.advert.order_id, &e.advert.seller_evm).unwrap_or_default();
                v.seller_backing = self.seller_backing(&e.advert.seller_evm, &v.seller_name);
                v
            })
            .chain(self.eth.store.orders.iter().filter(|o| o.takeable(tip)).map(|o| {
                let mut v = advert_view(&o.advert(), true);
                v.seller_name = self.own_evm_name(&o.seller_evm).unwrap_or_default();
                v.seller_backing = self.seller_backing(&o.seller_evm, &v.seller_name);
                v
            }))
            .collect();
        // Cheapest first, within each currency.
        sells.sort_by(|a, b| {
            let p = |v: &EthAdvertView| v.wei_per_gmds.parse::<u128>().unwrap_or(u128::MAX);
            a.asset.cmp(&b.asset).then(p(a).cmp(&p(b)))
        });
        let orders = self
            .eth
            .store
            .orders
            .iter()
            .rev()
            .map(|o| EthOrderView {
                id: hex::encode(o.id),
                mds: o.mds,
                sold: o.sold,
                min: o.min,
                wei_per_gmds: o.wei_per_gmds.to_string(),
                expires: o.expires,
                status: match o.status {
                    OrderStatus::Open if tip >= o.expires => "expired",
                    OrderStatus::Open => "open",
                    OrderStatus::Cancelled => "cancelled",
                    OrderStatus::Filled => "filled",
                }
                .into(),
                running: self.eth.store.swaps.iter().filter(|s| s.order_id == o.id && s.role == Role::Seller && !s.phase.settled()).count(),
                asset: o.asset.symbol().into(),
            })
            .collect();
        let swaps = self
            .eth
            .store
            .swaps
            .iter()
            .rev()
            .map(|s| {
                let mut v = swap_view(s);
                let (id, acct) = match s.role {
                    Role::Buyer => (s.order_id, s.seller_evm),
                    Role::Seller => (s.id, s.buyer_evm),
                    _ => ([0; 16], [0; 20]),
                };
                v.counterparty_name = self.verified_name(&id, &acct).unwrap_or_default();
                v
            })
            .collect();
        EthMarketView { sells, orders, swaps, tip, base_error: self.eth.error.clone() }
    }

    // ── The tick ────────────────────────────────────────────────────────

    pub(super) async fn tick_eth(&mut self, tip: u64) {
        if !self.eth.loaded || self.wallet.is_none() || self.node.is_syncing() {
            return;
        }
        let now = now_secs();
        self.eth_read_bus(tip, now).await;
        self.eth_advertise(tip);
        self.eth.book.retain(|_, e| e.advert.expires > tip && tip <= e.heard + ADVERT_STALE);
        self.eth_find_fills();
        // Orders that ended: sold out, or past their height.
        let mut changed = false;
        for o in &mut self.eth.store.orders {
            if o.status == OrderStatus::Open && o.mds < o.min.max(MIN_MDS_AMOUNT) && o.sold > 0 {
                o.status = OrderStatus::Filled;
                changed = true;
            }
        }
        if changed {
            let _ = self.eth_save();
        }
        for i in 0..self.eth.store.swaps.len() {
            if self.eth.store.swaps[i].phase.settled() && !self.eth.store.swaps[i].booked {
                self.eth_settled(i);
            }
        }
        if self.eth.store.swaps.iter().all(|s| s.phase.settled()) {
            return;
        }
        match self.eth_read_base(now).await {
            Ok(()) => self.eth.error = None,
            Err(e) => {
                tracing::warn!("reading Base: {e:#}");
                self.eth.error = Some(format!("{e:#}"));
            }
        }
        if let Err(e) = self.eth_advance(now).await {
            tracing::warn!("ETH swap tick: {e:#}");
        }
    }

    fn eth_send(&self, f: Frame) {
        let (words, attachments) = f.chat();
        if let Err(e) = self.node.send_chat(words, None, attachments) {
            tracing::warn!("trade message not sent: {e:#}");
        }
    }

    fn eth_advertise(&mut self, tip: u64) {
        let mut out = Vec::new();
        let mut claims = Vec::new();
        for o in &mut self.eth.store.orders {
            if o.takeable(tip) && (o.last_advert == 0 || tip >= o.last_advert + ADVERT_EVERY) {
                o.last_advert = tip;
                out.push(Frame::Advert(o.advert()));
                claims.push((o.id, o.seller_evm));
            }
        }
        for f in out {
            self.eth_send(f);
        }
        for (id, acct) in claims {
            self.send_claim(CLAIM_ORDER, id, &acct);
        }
    }

    // ── Names on offers ─────────────────────────────────────────────────

    /// A name this wallet holds whose `evm` record is `acct`.
    fn own_evm_name(&self, acct: &[u8; 20]) -> Option<String> {
        self.names.jobs.iter().filter(|j| j.status == "registered" && !j.renter).find_map(|j| {
            let r = self.live_record(&format!("{}{}", j.name, midstate::core::names::SUFFIX)).ok()?;
            r.records.iter().any(|x| matches!(x, midstate::core::names::Rec::Evm(a) if a == acct)).then(|| j.name.clone())
        })
    }

    /// Says which name an order or take of ours is, signed by its account,
    /// when this wallet holds a name whose `evm` record is that account.
    fn send_claim(&self, subject: u8, id: [u8; 16], acct: &[u8; 20]) {
        let Some(name) = self.own_evm_name(acct) else { return };
        let Ok(key) = self.evm_key() else { return };
        if &key.address != acct {
            return;
        }
        if let Ok(sig) = key.sign_message(&claim_message(subject, &id, &name)) {
            self.eth_send(Frame::Claim(NameClaim { subject, id, name, sig }));
        }
    }

    /// The name to show for the order or take `id` whose account is `acct`:
    /// only when the account signed the claim and the name's `evm` record is
    /// that same account. The account stays what the trade uses.
    fn verified_name(&self, id: &[u8; 16], acct: &[u8; 20]) -> Option<String> {
        let (name, signer) = self.eth.names.get(id)?;
        if signer != acct {
            return None;
        }
        let r = self.live_record(&format!("{name}{}", midstate::core::names::SUFFIX)).ok()?;
        r.records
            .iter()
            .any(|x| matches!(x, midstate::core::names::Rec::Evm(a) if a == acct))
            .then(|| format!("{name}{}", midstate::core::names::SUFFIX))
    }

    fn eth_send_take(&mut self, i: usize, now: u64) {
        let id = self.eth.store.swaps[i].id;
        if !self.eth.due(id, P_TAKE, now, RESEND_SECS) {
            return;
        }
        let sw = &self.eth.store.swaps[i];
        let t = Take { order_id: sw.order_id, take_id: sw.id, mds: sw.mds_amount, wei: sw.wei, claim_pk: sw.claim_pk, buyer_evm: sw.buyer_evm, asset: sw.asset };
        let (id, acct) = (sw.id, sw.buyer_evm);
        self.eth_send(Frame::Take(t));
        self.send_claim(CLAIM_TAKE, id, &acct);
    }

    async fn eth_read_bus(&mut self, tip: u64, now: u64) {
        let hist = self.node.chat_history.read().await.clone();
        for m in hist {
            if !self.eth.first_sight((m.sender.clone(), m.timestamp, m.nonce)) {
                continue;
            }
            let Some(frame) = parse(&m) else { continue };
            match frame {
                Frame::Claim(c) => {
                    if let Some(signer) = c.signer() {
                        if self.eth.names.len() > 4_096 {
                            self.eth.names.clear();
                        }
                        self.eth.names.insert(c.id, (c.name.clone(), signer));
                    }
                }
                Frame::Advert(a) => {
                    let ours = self.eth.store.orders.iter().any(|o| o.id == a.order_id);
                    // A USDC order is shown only where it can be bought.
                    let payable = a.asset == Asset::Eth || self.dex_cfg.usdc_ready();
                    if !ours && payable && a.expires > tip && a.mds >= a.min && a.min >= MIN_MDS_AMOUNT && a.wei_per_gmds > 0 {
                        self.eth.book.insert(a.order_id, BookEntry { advert: a, heard: tip });
                    }
                }
                Frame::Take(t) => {
                    if self.eth.store.orders.iter().any(|o| o.id == t.order_id) {
                        self.eth_on_take(t, tip, now).await;
                    }
                }
                Frame::Answer(a) => {
                    let id = a.take_id();
                    let Some(sw) = self.eth.store.swaps.iter_mut().find(|s| s.id == id && s.role == Role::Buyer) else { continue };
                    if sw.phase != Phase::Asking {
                        continue;
                    }
                    match a {
                        Answer::Lock { secret_hash, timeout, refund_pk, .. } => {
                            sw.secret_hash = secret_hash;
                            sw.timeout = timeout;
                            sw.refund_pk = refund_pk;
                            sw.note = "the seller is locking the MDS for you".into();
                        }
                        Answer::Decline { reason, .. } => {
                            sw.phase = Phase::Failed(format!("the seller declined: {}", reason.says()));
                        }
                    }
                    sw.updated = now;
                    let _ = self.eth_save();
                }
            }
        }
    }

    async fn eth_on_take(&mut self, t: Take, tip: u64, now: u64) {
        // A repeat of a take we already answered: the tick repeats the answer.
        if self.eth.store.swaps.iter().any(|s| s.id == t.take_id) {
            return;
        }
        let decline = |svc: &Self, reason| svc.eth_send(Frame::Answer(Answer::Decline { take_id: t.take_id, reason }));
        let Some(order) = self.eth.store.orders.iter().find(|o| o.id == t.order_id).cloned() else { return };
        if let Err(reason) = check_take(&order, tip, t.mds, t.wei, t.asset) {
            return decline(self, reason);
        }
        let busy_buyer = self.eth.store.swaps.iter().any(|s| s.role == Role::Seller && s.buyer_evm == t.buyer_evm && !s.phase.settled());
        if self.eth.store.unfunded(&order.id) >= MAX_UNFUNDED || busy_buyer {
            return decline(self, Decline::Busy);
        }
        let state = self.node.get_state().await;
        let in_flight = self.in_flight_inputs();
        let have: u64 = match self.wallet.as_ref() {
            Some(w) => w.spendable_now(&state, &in_flight).iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum(),
            None => 0,
        };
        if have < t.mds || self.ensure_restore_done().is_err() {
            return decline(self, Decline::Unable);
        }
        let secret: [u8; 32] = rand::random();
        let seed: [u8; 32] = rand::random();
        let mut sw = EthSwap::new(t.take_id, Role::Seller, Phase::LockingMine, now);
        sw.order_id = order.id;
        sw.mds_amount = t.mds;
        sw.lock_value = t.mds;
        sw.wei = t.wei;
        sw.seller_evm = order.seller_evm;
        sw.buyer_evm = t.buyer_evm;
        sw.claim_pk = t.claim_pk;
        sw.key_seed = Some(seed);
        sw.refund_pk = pk_of(&seed);
        sw.secret = Some(secret);
        sw.secret_hash = midstate::core::types::hash(&secret);
        sw.timeout = seller_timeout(tip);
        sw.base_from = self.eth.index.head.saturating_sub(100);
        sw.asset = order.asset;
        if let Some(o) = self.eth.store.order_mut(&order.id) {
            o.mds -= t.mds;
        }
        self.eth.store.swaps.push(sw);
        if let Err(e) = self.eth_save() {
            tracing::warn!("could not record a sale: {e:#}");
        }
        self.ch_notice(format!("A buyer is taking {} units of your sell order; locking them for that buyer.", t.mds));
    }

    /// Our buy orders' fills, announced as version-2 locks: one swap each.
    fn eth_find_fills(&mut self) {
        let now = now_secs();
        let mut new = Vec::new();
        for (_, f) in &self.dex.fills {
            let h = hex::encode(f.secret_hash);
            let Some(bid) = self.book.bids.iter().find(|b| b.secret_hash == h && !b.cancelled && !b.bid_id.is_empty()) else { continue };
            if hex::encode(f.receiver_addr) != bid.mds_addr || f.min_payout < bid.mds_amount || f.total < f.min_payout {
                continue;
            }
            let Ok(bid_id) = hex::decode(&bid.bid_id) else { continue };
            let Ok(bid_id) = <[u8; 32]>::try_from(bid_id.as_slice()) else { continue };
            let mut idm = Vec::with_capacity(64);
            idm.extend_from_slice(&bid_id);
            idm.extend_from_slice(&f.taker_mds_pk);
            let id: [u8; 16] = midstate::core::types::hash(&idm)[..16].try_into().unwrap();
            if self.eth.store.swaps.iter().any(|s| s.id == id) || new.iter().any(|s: &EthSwap| s.id == id) {
                continue;
            }
            let Ok(secret) = hex::decode(&bid.secret) else { continue };
            let Ok(secret) = <[u8; 32]>::try_from(secret.as_slice()) else { continue };
            let mut sw = EthSwap::new(id, Role::Bidder, Phase::WaitingTheirLock, now);
            sw.bid_id = bid_id;
            sw.mds_amount = f.min_payout;
            sw.lock_value = f.total;
            sw.wei = bid.wei.parse().unwrap_or(0);
            sw.receiver_addr = f.receiver_addr;
            sw.refund_pk = f.taker_mds_pk;
            sw.secret_hash = f.secret_hash;
            sw.secret = Some(secret);
            sw.timeout = f.timeout_height;
            sw.dest = Some(f.receiver_addr);
            sw.base_from = self.eth.index.head.saturating_sub(100);
            sw.asset = bid.asset;
            new.push(sw);
        }
        if !new.is_empty() {
            for sw in &new {
                self.ch_notice(format!("Someone is filling your buy order with {} units; checking their lock.", sw.mds_amount));
            }
            self.eth.store.swaps.extend(new);
            let _ = self.eth_save();
        }
    }

    /// New events of both contracts, receipts of our transactions, and the
    /// bids our swaps are about.
    async fn eth_read_base(&mut self, now: u64) -> Result<()> {
        let client = self.eth_client(Asset::Eth)?;
        let (head, head_time) = client.head().await?;
        self.eth.index.head = head;
        self.eth.index.head_time = head_time;
        let safe = head.saturating_sub(self.dex_cfg.confirmations);
        let mut changed = false;
        // A swap made before Base's head was known reads back from here.
        for s in &mut self.eth.store.swaps {
            if s.base_from == 0 && !s.phase.settled() {
                s.base_from = head.saturating_sub(BASE_LOOKBACK);
                changed = true;
            }
        }
        for asset in Asset::ALL {
            if asset == Asset::Usdc && !self.dex_cfg.usdc_ready() {
                continue;
            }
            let c = self.eth_client(asset)?;
            let mut cursor = match asset {
                Asset::Eth => self.eth.index.cursor,
                Asset::Usdc => self.eth.index.cursor_usdc,
            };
            if cursor == 0 {
                let live = self.eth.store.swaps.iter().filter(|s| !s.phase.settled() && s.asset == asset);
                cursor = live.map(|s| s.base_from).min().unwrap_or(head).min(safe).saturating_sub(1);
            }
            if safe > cursor {
                let to = safe.min(cursor + BASE_PER_TICK);
                let part = c.scan(cursor + 1, to).await;
                for (b, e) in &part.events {
                    self.eth.index.add(*b, e, asset);
                }
                cursor = part.reached.max(cursor);
                if let Some(e) = part.error {
                    tracing::warn!("Base {} events past block {}: {e:#}", asset.symbol(), part.reached);
                }
            }
            match asset {
                Asset::Eth => self.eth.index.cursor = cursor,
                Asset::Usdc => self.eth.index.cursor_usdc = cursor,
            }
        }
        self.eth.index.prune(&self.eth.store.swaps);

        // Our value-carrying transactions, a buyer's lock and a filler's
        // reservation: found by their receipt, or shown by their nonce never
        // to be mined.
        for i in 0..self.eth.store.swaps.len() {
            let sw = self.eth.store.swaps[i].clone();
            let Some(tx) = sw.eth_tx else { continue };
            let waiting = match (&sw.role, &sw.phase) {
                (Role::Buyer, Phase::LockingMine) => sw.base_swap.is_none(),
                (Role::Filler, Phase::Reserving) => true,
                _ => false,
            };
            if !waiting {
                continue;
            }
            let r = match client.receipt(&tx).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("receipt of {}: {e:#}", hex::encode(tx));
                    continue;
                }
            };
            let what = if sw.role == Role::Buyer { format!("{} lock", sw.asset.symbol()) } else { "reservation".to_string() };
            let on = match self.eth_client(sw.asset) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("{what}: {e:#}");
                    continue;
                }
            };
            match r {
                Some(r) if !r.success => {
                    self.eth.store.swaps[i].phase = Phase::Failed(format!("the {what} reverted; nothing was paid"));
                    changed = true;
                }
                Some(r) if sw.role == Role::Buyer => {
                    // Only the swap contract's own event: a token's logs ride
                    // in the same receipt.
                    let locked = r.logs.iter().filter(|l| l.address == on.contract()).filter_map(crate::base::decode_event).find(|e| matches!(e, Event::Locked { .. }));
                    if let Some(Event::Locked { swap_id, timeout, .. }) = locked {
                        let s = &mut self.eth.store.swaps[i];
                        s.base_swap = Some(swap_id);
                        s.eth_deadline = timeout;
                        s.base_from = s.base_from.min(r.block);
                        s.note = format!("{} locked; waiting for the seller to claim it, which releases your MDS", sw.asset.symbol());
                        changed = true;
                    }
                }
                Some(_) => {
                    // A failed read here leaves the receipt for the next tick.
                    let b = match on.get_bid(sw.bid_id).await {
                        Ok(Some(b)) => b,
                        Ok(None) => {
                            tracing::warn!("reserved buy order {} not found", hex::encode(sw.bid_id));
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!("reading a reserved buy order: {e:#}");
                            continue;
                        }
                    };
                    let tip = self.node.get_state().await.height;
                    let s = &mut self.eth.store.swaps[i];
                    if b.filler != s.seller_evm {
                        s.phase = Phase::Failed("someone else reserved the buy order first".into());
                    } else {
                        match filler_timeout(tip, now.max(head_time), b.fill_deadline) {
                            Some(t) => {
                                s.eth_deadline = b.fill_deadline;
                                s.timeout = t;
                                let template = s.clone();
                                match fill_total(s.mds_amount, &template) {
                                    Ok(total) => {
                                        s.lock_value = total;
                                        s.phase = Phase::LockingMine;
                                        s.note = "reserved; locking the MDS for the buyer".into();
                                    }
                                    Err(e) => s.phase = Phase::Failed(format!("could not size the lock: {e}")),
                                }
                            }
                            None => s.phase = Phase::Failed("the reservation is too short to lock against".into()),
                        }
                    }
                    changed = true;
                }
                None => changed |= self.eth_unmined(i, &client, now, &what).await,
            }
        }

        // Bids our swaps depend on, each from its own contract.
        let bids: Vec<([u8; 32], Asset)> = self
            .eth
            .store
            .swaps
            .iter()
            .filter(|s| !s.phase.settled() && matches!(s.role, Role::Filler | Role::Bidder))
            .map(|s| (s.bid_id, s.asset))
            .collect();
        for (id, asset) in bids {
            let fresh = self.eth.index.bids.get(&id).is_some_and(|(_, at)| now < at + BID_REFRESH_SECS);
            if !fresh {
                let read = match self.eth_client(asset) {
                    Ok(c) => c.get_bid(id).await,
                    Err(e) => Err(e),
                };
                match read {
                    Ok(Some(b)) => {
                        self.eth.index.bids.insert(id, (b, now));
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("reading buy order {}: {e:#}", hex::encode(&id[..8])),
                }
            }
        }
        if changed {
            self.eth_save()?;
        }
        Ok(())
    }

    /// A signed value-carrying transaction with no receipt: sent again while
    /// it could still be mined in time, replaced once it could not, and
    /// given up only once its nonce has gone to another transaction.
    /// Returns whether the journal changed.
    async fn eth_unmined(&mut self, i: usize, client: &BaseClient, now: u64, what: &str) -> bool {
        let sw = self.eth.store.swaps[i].clone();
        let Some(p) = &sw.eth_pending else { return false };
        let Ok(raw) = hex::decode(&p.raw) else { return false };
        let Ok(key) = self.evm_key() else { return false };
        match client.nonce_mined(&key.address).await {
            Ok(n) if n > p.nonce => {
                // Mined, but not ours. Give a slow node time to show our
                // receipt before believing it.
                if now >= p.at + ETH_MINE_SECS + 120 {
                    let s = &mut self.eth.store.swaps[i];
                    s.phase = Phase::Failed(format!("the {what} was replaced before it was mined; nothing was paid"));
                    return true;
                }
            }
            Ok(_) if now < p.at + ETH_MINE_SECS => {
                if self.eth.due(sw.id, P_BASE, now, BASE_RETRY_SECS) {
                    if let Err(e) = client.broadcast(&raw).await {
                        tracing::warn!("sending the {what} again: {e:#}");
                    }
                }
            }
            Ok(_) => {
                // Too late to be mined as planned: an ETH lock mined now
                // could outlast the seller's. Replace it so it never is.
                if self.eth.due(sw.id, P_CANCEL, now, BASE_RETRY_SECS) {
                    match client.cancel(&key, &raw).await {
                        Ok(_) => self.eth.store.swaps[i].note = format!("the {what} was not mined in time; cancelling it"),
                        Err(e) => tracing::warn!("cancelling the {what}: {e:#}"),
                    }
                }
            }
            Err(e) => tracing::warn!("reading our Base nonce: {e:#}"),
        }
        false
    }

    async fn eth_advance(&mut self, now: u64) -> Result<()> {
        let (_, mempool) = self.node.get_mempool_info().await;
        let state = self.node.get_state().await;
        let pending: HashSet<[u8; 32]> = self.wallet.as_ref().map(|w| w.data.pending.iter().map(|p| p.commitment).collect()).unwrap_or_default();
        let mut decisions = Vec::new();
        let mut changed = false;
        {
            let chains = LiveEth { mds: state, mempool, node: &self.node, now, index: &self.eth.index, pending };
            for i in 0..self.eth.store.swaps.len() {
                let sw = &mut self.eth.store.swaps[i];
                if sw.phase.settled() {
                    continue;
                }
                let before = (sw.lock_seen, sw.scan_from, sw.secret);
                let o = observe(sw, &chains);
                if before != (sw.lock_seen, sw.scan_from, sw.secret) {
                    changed = true;
                }
                let d = decide(sw, &o);
                if let Some(p) = &d.phase {
                    if *p != sw.phase {
                        sw.phase = p.clone();
                        sw.updated = now;
                        changed = true;
                    }
                }
                if let Some(n) = &d.note {
                    if &sw.note != n {
                        sw.note = n.clone();
                        changed = true;
                    }
                }
                decisions.push((i, d, o.tip));
            }
        }
        if changed {
            self.eth_save()?;
        }
        for (i, d, tip) in decisions {
            if self.eth.store.swaps[i].phase.settled() {
                continue;
            }
            for a in d.acts {
                if let Err(e) = self.eth_act(i, &a, tip, now).await {
                    let sw = &mut self.eth.store.swaps[i];
                    tracing::warn!(trade = %hex::encode(&sw.id[..4]), "trade step deferred: {e:#}");
                    sw.note = format!("retrying: {e}");
                }
            }
        }
        for i in 0..self.eth.store.swaps.len() {
            if self.eth.store.swaps[i].phase.settled() && !self.eth.store.swaps[i].booked {
                self.eth_settled(i);
            }
        }
        Ok(())
    }

    /// Bookkeeping when a swap ends: its order's amount, and a notice.
    fn eth_settled(&mut self, i: usize) {
        let sw = self.eth.store.swaps[i].clone();
        self.eth.store.swaps[i].booked = true;
        if sw.role == Role::Seller {
            if let Some(o) = self.eth.store.order_mut(&sw.order_id) {
                match sw.phase {
                    Phase::Done => o.sold += sw.mds_amount,
                    // Nothing sold: the amount is for sale again.
                    Phase::Refunded => o.mds += sw.mds_amount,
                    Phase::Failed(_) if sw.lock_commitment.is_none() => o.mds += sw.mds_amount,
                    _ => {}
                }
            }
        }
        let what = match (&sw.role, &sw.phase) {
            (Role::Buyer, Phase::Done) => format!("Bought {} units of MDS.", sw.mds_amount),
            (Role::Seller, Phase::Done) => format!("Sold {} units of MDS for {}.", sw.mds_amount, sw.asset.format(sw.wei, 8)),
            (Role::Filler, Phase::Done) => format!("Filled a buy order: {} received.", sw.asset.format(sw.wei + sw.fill_bond, 8)),
            (Role::Bidder, Phase::Done) => format!("Your buy order was filled: {} units received.", sw.mds_amount),
            (_, Phase::Refunded) => "A trade did not complete; your funds came back.".into(),
            (_, Phase::Failed(r)) => format!("A trade stopped: {r}"),
            _ => String::new(),
        };
        if !what.is_empty() {
            self.ch_notice(what);
        }
        let _ = self.eth_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    async fn eth_act(&mut self, i: usize, a: &Act, tip: u64, now: u64) -> Result<()> {
        let id = self.eth.store.swaps[i].id;
        match a {
            Act::LockMds => self.eth_lock_mds(i).await,
            Act::Answer => {
                if self.eth.due(id, P_ANSWER, now, RESEND_SECS) {
                    let sw = &self.eth.store.swaps[i];
                    self.eth_send(Frame::Answer(Answer::Lock { take_id: sw.id, secret_hash: sw.secret_hash, timeout: sw.timeout, refund_pk: sw.refund_pk }));
                }
                Ok(())
            }
            Act::Take => {
                self.eth_send_take(i, now);
                Ok(())
            }
            Act::LockEth { delay } => {
                let sw = self.eth.store.swaps[i].clone();
                match sw.asset {
                    Asset::Eth => {
                        let data = crate::base::lock_call(sw.secret_hash, sw.seller_evm, *delay)?;
                        self.eth_send_value(i, sw.wei, data, now, "ETH lock").await
                    }
                    Asset::Usdc => {
                        let Some(permit) = self.eth_token_permit(i, sw.wei, now).await? else { return Ok(()) };
                        let data = crate::base::token_lock_call(sw.secret_hash, sw.seller_evm, sw.wei, *delay, Some(&permit))?;
                        self.eth_send_value(i, 0, data, now, "USDC lock").await
                    }
                }
            }
            Act::Reserve => {
                let sw = self.eth.store.swaps[i].clone();
                match sw.asset {
                    Asset::Eth => {
                        let data = crate::base::reserve_call(sw.bid_id, FILL_WINDOW_SECS);
                        self.eth_send_value(i, sw.fill_bond, data, now, "reservation").await
                    }
                    // The bond, if the bid asks one, is paid in USDC too.
                    Asset::Usdc if sw.fill_bond > 0 => {
                        let Some(permit) = self.eth_token_permit(i, sw.fill_bond, now).await? else { return Ok(()) };
                        let data = crate::base::token_reserve_call(sw.bid_id, FILL_WINDOW_SECS, Some(&permit));
                        self.eth_send_value(i, 0, data, now, "reservation").await
                    }
                    Asset::Usdc => {
                        let data = crate::base::token_reserve_call(sw.bid_id, FILL_WINDOW_SECS, None);
                        self.eth_send_value(i, 0, data, now, "reservation").await
                    }
                }
            }
            Act::ClaimEth { swap_id } => {
                if !self.eth.due(id, P_BASE, now, BASE_RETRY_SECS) {
                    return Ok(());
                }
                // Journal first: once a claim may be out, our lock stays.
                let s = &mut self.eth.store.swaps[i];
                s.base_swap = Some(*swap_id);
                s.claim_sent = true;
                let secret = s.secret.ok_or_else(|| anyhow!("no secret"))?;
                let asset = s.asset;
                self.eth_save()?;
                let key = self.evm_key()?;
                let client = self.eth_client(asset)?;
                match client.claim(&key, *swap_id, secret).await {
                    Ok(h) => {
                        self.eth.store.swaps[i].eth_tx = Some(h);
                        self.eth_save()
                    }
                    // Claimed already (ours, seen late), or refunded: only
                    // the event says which.
                    Err(e) if format!("{e:#}").contains("settled") => self.eth_settlement(i, *swap_id, now).await,
                    Err(e) => Err(e),
                }
            }
            Act::ClaimBid => {
                if !self.eth.due(id, P_BASE, now, BASE_RETRY_SECS) {
                    return Ok(());
                }
                let sw = self.eth.store.swaps[i].clone();
                let secret = sw.secret.ok_or_else(|| anyhow!("no secret"))?;
                let key = self.evm_key()?;
                let h = self.eth_client(sw.asset)?.claim_bid(&key, sw.bid_id, secret).await?;
                self.eth.store.swaps[i].eth_tx = Some(h);
                self.eth.index.bids.remove(&sw.bid_id); // read it again soon
                self.eth_save()
            }
            Act::RefundEth { swap_id } => {
                if !self.eth.due(id, P_BASE, now, BASE_RETRY_SECS) {
                    return Ok(());
                }
                let key = self.evm_key()?;
                let client = self.eth_client(self.eth.store.swaps[i].asset)?;
                match client.refund(&key, *swap_id).await {
                    Ok(h) => {
                        self.eth.store.swaps[i].eth_tx = Some(h);
                        self.eth_save()
                    }
                    // Claimed after all (the claim is where the secret is,
                    // whatever our scan missed), or our own refund, seen late.
                    Err(e) if format!("{e:#}").contains("settled") => self.eth_settlement(i, *swap_id, now).await,
                    Err(e) => Err(e),
                }
            }
            Act::ClaimMds => self.eth_spend(i, SpendKind::Claim, tip, now).await,
            Act::RefundMds => self.eth_spend(i, SpendKind::Refund, tip, now).await,
        }
    }

    /// A permit for swap `i` to pay `value` USDC into its contract, or `None`
    /// while another USDC payment of ours is still on its way (its permit
    /// holds the token's next nonce; this one waits a tick or two). Only
    /// asked for when the payment is about to be signed.
    async fn eth_token_permit(&mut self, i: usize, value: u128, now: u64) -> Result<Option<crate::evm::Permit>> {
        let (id, sent) = (self.eth.store.swaps[i].id, self.eth.store.swaps[i].eth_tx.is_some());
        if sent || !self.eth.due(id, P_BASE, now, BASE_RETRY_SECS) {
            return Ok(None);
        }
        // Paced above; let the send itself be due again.
        self.eth.pushed.remove(&(id, P_BASE));
        if self.token_payment_in_flight() {
            self.eth.store.swaps[i].note = "waiting for your other USDC payment to land first".into();
            return Ok(None);
        }
        let client = self.eth_client(Asset::Usdc)?;
        let key = self.evm_key()?;
        let have = client.token_balance(&key.address).await.context("could not read your USDC balance")?;
        if have < value {
            let s = &mut self.eth.store.swaps[i];
            s.phase = Phase::Failed(format!("this needs {}, and your Base account holds {}; nothing was paid", Asset::Usdc.format(value, 6), Asset::Usdc.format(have, 6)));
            self.eth_save()?;
            return Ok(None);
        }
        let base_now = now.max(self.eth.index.head_time);
        Ok(Some(client.permit(&key, value, base_now).await?))
    }

    /// Signs a value-carrying call, journals it, then sends it. Once
    /// journalled it is only ever resolved by its receipt or its nonce
    /// (`eth_unmined`): an error from the send does not mean nothing was paid.
    async fn eth_send_value(&mut self, i: usize, value: u128, data: Vec<u8>, now: u64, what: &str) -> Result<()> {
        let id = self.eth.store.swaps[i].id;
        if self.eth.store.swaps[i].eth_tx.is_some() || !self.eth.due(id, P_BASE, now, BASE_RETRY_SECS) {
            return Ok(());
        }
        let key = self.evm_key()?;
        let client = self.eth_client(self.eth.store.swaps[i].asset)?;
        let t = match client.sign_call(&key, value, data).await {
            Ok(t) => t,
            Err(e) => {
                // Nothing signed, so nothing can be mined.
                let s = &mut self.eth.store.swaps[i];
                s.phase = Phase::Failed(format!("the {what} was not sent: {e:#}; nothing was paid"));
                return self.eth_save();
            }
        };
        {
            let s = &mut self.eth.store.swaps[i];
            s.eth_tx = Some(t.hash);
            s.eth_pending = Some(EthPending { raw: hex::encode(&t.raw), nonce: t.nonce, at: now });
            s.updated = now;
        }
        self.eth_save()?;
        let note = match client.broadcast(&t.raw).await {
            Ok(()) => format!("{what} sent"),
            Err(e) => {
                tracing::warn!("sending the {what}: {e:#}");
                format!("{what} signed; confirming it reached Base")
            }
        };
        self.eth.store.swaps[i].note = note;
        self.eth_save()
    }

    /// How a Base swap our call found settled ended, from its own events:
    /// a claim (and its secret) or a refund. Never guessed from silence.
    async fn eth_settlement(&mut self, i: usize, swap_id: [u8; 32], now: u64) -> Result<()> {
        if !self.eth.lookup_due(swap_id, now) {
            return Ok(());
        }
        let from = self.eth.store.swaps[i].base_from;
        let client = self.eth_client(self.eth.store.swaps[i].asset)?;
        match client.find_settlement(swap_id, from, self.eth.index.head).await? {
            Some(Settlement::Claimed(p)) => {
                let s = &mut self.eth.store.swaps[i];
                if s.role == Role::Buyer && s.secret.is_none() && midstate::core::types::hash(&p) == s.secret_hash {
                    s.secret = Some(p);
                    s.note = "the seller claimed; collecting your MDS".into();
                }
                self.eth.index.claims.insert(swap_id, p);
            }
            Some(Settlement::Refunded) => {
                self.eth.index.refunds.insert(swap_id);
            }
            None => bail!("the ETH lock is settled, but neither its claim nor its refund has been found yet"),
        }
        self.eth_save()
    }

    /// Funds our MDS lock through the wallet's send machine: a seller's
    /// lock, or a filler's with its announcement in the same transaction.
    async fn eth_lock_mds(&mut self, i: usize) -> Result<()> {
        let sw = self.eth.store.swaps[i].clone();
        if sw.lock_commitment.is_some() {
            return Ok(());
        }
        self.ensure_restore_done()?;
        self.verify_mss_indices().await?;
        let state = self.node.get_state().await;
        let in_flight = self.in_flight_inputs();
        let mut outputs = sw.lock_outputs();
        if sw.role == Role::Filler {
            let group: [u8; 6] = rand::random();
            for frag in midstate::core::dex::fragment(&sw.fill().encode(), &group)? {
                outputs.push(midstate::core::OutputData::DataBurn { payload: frag, value_burned: 0 });
            }
        }
        let mut journal = self.eth.store.clone();
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let live = w.spendable_now(&state, &in_flight);
        let plan = match sendplan::plan_fixed_outputs(w, &live, outputs) {
            Ok(p) => p,
            Err(e) => {
                let s = &mut self.eth.store.swaps[i];
                s.phase = Phase::Failed(format!("could not lock the MDS: {e:#}"));
                if s.role == Role::Seller {
                    let take_id = s.id;
                    self.eth_send(Frame::Answer(Answer::Decline { take_id, reason: Decline::Unable }));
                }
                return self.eth_save();
            }
        };
        let (commitment, _) = w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false)?;
        // The send and the journal entry naming it, in one write. Spends of
        // the lock are looked for from here, even if it is never seen whole.
        journal.swaps[i].lock_commitment = Some(commitment);
        journal.swaps[i].scan_from = state.height.saturating_sub(1).max(1);
        journal.compact();
        w.data.eth_state = Some(journal.to_value()?);
        w.save()?;
        self.eth.store = journal;
        let what = if sw.role == Role::Filler { "filling a buy order" } else { "locking MDS for a buyer" };
        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: format!("{what} — solving commit proof-of-work"),
                amount: sw.lock_value,
                fee: plan.fee,
                to: "trade lock".into(),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, what);
        self.broadcast_commit(commitment).await
    }

    /// Our claim or refund of the MDS lock: commit once, then reveal once
    /// the commitment is in a block.
    async fn eth_spend(&mut self, i: usize, kind: SpendKind, tip: u64, now: u64) -> Result<()> {
        let dest = match self.eth.store.swaps[i].dest {
            Some(d) => d,
            None => {
                // Chosen once, and journalled before any commitment to it.
                let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
                let d = w.generate_key(Some(if kind == SpendKind::Claim { "bought MDS".into() } else { "trade refund".into() }))?;
                self.eth.store.swaps[i].dest = Some(d);
                self.eth_save()?;
                d
            }
        };
        if kind == SpendKind::Refund && self.eth.store.swaps[i].refund_mask == 0 {
            // The coins still there, fixed before the first commit: the
            // refund key signs one message, so this set never changes.
            let state = self.node.get_state().await;
            let sw = &self.eth.store.swaps[i];
            let left = sw.lock_coin_ids().iter().enumerate().filter(|(_, id)| state.coins.contains(id)).fold(0u32, |m, (j, _)| m | 1 << j);
            ensure!(sw.refundable(left), "nothing left in the lock that can pay for its refund");
            self.eth.store.swaps[i].refund_mask = left;
            self.eth_save()?;
        }
        let sw = self.eth.store.swaps[i].clone();
        let sp = build_spend(&sw, kind, &dest)?;
        match sw.spend_commitment {
            Some(c) if c != sp.commitment => bail!("the spend no longer matches its commitment"),
            Some(c) if self.node.check_commitment(c).await => {
                if !self.eth.due(sw.id, P_REVEAL, now, 60) {
                    return Ok(());
                }
                let tx = sign_spend(&sw, kind, &sp)?;
                if sw.role == Role::Bidder && !sw.reveal_sent {
                    // This reveal publishes the secret: from now on the claim
                    // is seen through, never abandoned.
                    self.eth.store.swaps[i].reveal_sent = true;
                    self.eth_save()?;
                }
                self.node.send_transaction(tx).await.context("the reveal was refused")?;
                Ok(())
            }
            Some(_) => {
                let since = self.eth.pushed.get(&(sw.id, P_COMMIT)).copied().unwrap_or(0);
                if tip >= since + RECOMMIT_BLOCKS {
                    self.eth.pushed.insert((sw.id, P_COMMIT), tip);
                    self.commit_external(sp.commitment).await?;
                }
                Ok(())
            }
            None => {
                self.eth.store.swaps[i].spend_commitment = Some(sp.commitment);
                self.eth_save()?;
                self.eth.pushed.insert((sw.id, P_COMMIT), tip);
                self.commit_external(sp.commitment).await
            }
        }
    }
}

fn advert_view(a: &Advert, mine: bool) -> EthAdvertView {
    EthAdvertView {
        order_id: hex::encode(a.order_id),
        seller: crate::evm::to_checksum_address(&a.seller_evm),
        mds: a.mds,
        min: a.min,
        wei_per_gmds: a.wei_per_gmds.to_string(),
        expires: a.expires,
        mine,
        seller_name: String::new(),
        seller_backing: 0.0,
        asset: a.asset.symbol().into(),
    }
}

fn swap_view(s: &EthSwap) -> EthSwapView {
    let counterparty = match s.role {
        Role::Buyer => s.seller_evm,
        Role::Seller => s.buyer_evm,
        Role::Filler => s.buyer_evm,
        Role::Bidder => [0; 20],
    };
    let note = match &s.phase {
        Phase::Failed(r) => r.clone(),
        Phase::Done => match s.role {
            Role::Buyer | Role::Bidder => "complete: the MDS is yours".into(),
            Role::Seller | Role::Filler => format!("complete: the {} is yours", s.asset.symbol()),
        },
        Phase::Refunded => "did not complete: your funds came back".into(),
        _ if !s.note.is_empty() => s.note.clone(),
        p => default_note(s.role, p).replace("ETH", s.asset.symbol()),
    };
    EthSwapView {
        id: hex::encode(s.id),
        kind: kind_name(s.role).into(),
        phase: s.phase.name().into(),
        mds: s.mds_amount,
        wei: s.wei.to_string(),
        note,
        settled: s.phase.settled(),
        counterparty: if counterparty == [0; 20] { String::new() } else { crate::evm::to_checksum_address(&counterparty) },
        counterparty_name: String::new(),
        eth_deadline: s.eth_deadline,
        timeout: s.timeout,
        base_tx: s.eth_tx.map(|h| format!("0x{}", hex::encode(h))),
        created: s.created,
        updated: s.updated,
        asset: s.asset.symbol().into(),
    }
}

fn default_note(r: Role, p: &Phase) -> &'static str {
    match (r, p) {
        (Role::Buyer, Phase::Asking) => "asking the seller",
        (Role::Buyer, Phase::WaitingTheirLock) => "waiting for the seller's lock to confirm",
        (Role::Buyer, Phase::LockingMine) => "locking your ETH",
        (Role::Buyer, Phase::MineLocked) => "waiting for the seller to claim, which releases your MDS",
        (Role::Buyer, Phase::Claiming) => "collecting your MDS",
        (Role::Buyer, Phase::Refunding) => "taking your ETH back",
        (Role::Seller, Phase::LockingMine) => "locking the MDS for the buyer",
        (Role::Seller, Phase::MineLocked) => "waiting for the buyer's ETH",
        (Role::Seller, Phase::Claiming) => "claiming the ETH",
        (Role::Seller, Phase::Refunding) => "the buyer did not pay; taking the MDS back",
        (Role::Filler, Phase::Reserving) => "reserving the buy order",
        (Role::Filler, Phase::LockingMine) => "locking the MDS for the buyer",
        (Role::Filler, Phase::MineLocked) => "waiting for the buyer to claim",
        (Role::Filler, Phase::Claiming) => "collecting the ETH",
        (Role::Filler, Phase::Refunding) => "the buyer did not claim; taking the MDS back",
        (Role::Bidder, Phase::WaitingTheirLock) => "checking the seller's lock",
        (Role::Bidder, Phase::Claiming) => "collecting your MDS",
        _ => "",
    }
}
