//! The walletd actor's midwimble side: the MDW account, orders on the chat
//! bus, negotiations, and swaps, driven from the embedded midwimble node and
//! midstate's own.
//!
//! A child of `service`, so it shares the actor's state; everything that
//! decides lives in `mwb_trade` and `mwb_negotiate`, and this module only
//! reads the chains, moves messages and broadcasts what they decide.

use super::{now_secs, SendMeta, Service};
use crate::api::*;
use crate::mwb::min_fee;
use crate::mwb_account::{BlockView, MwbAccount, SwapOutputs, LOCK_LABEL, REORG};
use crate::mwb_negotiate::{abort_messages, Ctx, Linger, Negotiation, Step, MAX_RESENDS, NEGOTIATION_SECS, RESEND_SECS};
use crate::mwb_trade::{
    build_mds_spend, check_terms, decide, observe, sign_mds_spend, Act, BookEntry, Chains, Leg,
    MdsSpendKind, MwbOrder, MwbStore, MwbSwap, OrderStatus, Phase, Role, ADVERT_EVERY, ADVERT_STALE, MAX_ORDER_LIFE,
    MIN_ORDER_LIFE, REBROADCAST_BLOCKS,
};
use crate::mwb_wire::{
    maker_session, parse, session_messages, Frame, KeyPair, Msg, Outgoing, Route, Side, To,
};
use crate::sendplan;
use crate::api::SendStage;
use anyhow::{anyhow, bail, ensure, Context as _, Result};
use midstate::core::Transaction;
use midwimble::core::mw::crypto::Point32;
use midwimble::core::mw::stealth::{StealthAddress, WalletKeys};
use midwimble::core::mw::transaction::{Kernel, Transaction as MwbTx};
use std::collections::{HashMap, HashSet, VecDeque};

/// Midwimble blocks scanned per tick.
const MWB_SCAN_CHUNK: u64 = 500;
/// The account is saved at least this often while scanning, so a restart
/// does not rescan from its birth.
const MWB_SAVE_EVERY: u64 = 1_000;
/// Seconds between rebroadcasts of a payment still waiting for a block.
const MWB_REBROADCAST_SECS: u64 = 120;
/// Midstate blocks before an external commit that has not appeared is made
/// again.
const RECOMMIT_BLOCKS: u64 = 20;

/// Where the embedded node keeps its chain, listens, and serves its
/// explorer: fixed at boot.
pub(super) struct MwbPlace {
    pub(super) explorer: Option<String>,
    pub(super) data_dir: String,
    pub(super) p2p_port: u16,
}

/// The actor's midwimble state that is not journalled.
#[derive(Default)]
pub(super) struct MwbLive {
    pub(super) keys: Option<WalletKeys>,
    pub(super) store: MwbStore,
    /// Whether `store` was loaded from an unlocked wallet.
    pub(super) loaded: bool,
    pub(super) negs: HashMap<[u8; 16], Negotiation>,
    pub(super) lingers: HashMap<[u8; 16], Linger>,
    pub(super) book: HashMap<[u8; 16], BookEntry>,
    pub(super) inbox: crate::mwb_wire::Reassembler,
    seen: HashSet<(String, u64, u64)>,
    seen_order: VecDeque<(String, u64, u64)>,
    /// When each swap last pushed a transaction, by (swap, what): rebroadcast
    /// pacing only, so it need not survive a restart.
    pushed: HashMap<([u8; 32], u8), u64>,
}

impl MwbLive {
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
}

fn side_name(s: Side) -> &'static str {
    match s {
        Side::SellMds => "sell_mds",
        Side::SellMdw => "sell_mdw",
    }
}

fn phase_name(p: &Phase) -> &'static str {
    match p {
        Phase::Negotiating => "negotiating",
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

fn coin_name(l: Leg) -> &'static str {
    match l {
        Leg::Mds => "MDS",
        Leg::Mwb => "MDW",
    }
}

/// Errors a midwimble node gives for a transaction it already has, or one a
/// block has already settled: nothing to do.
fn harmless(e: &anyhow::Error) -> bool {
    let s = format!("{e:#}");
    ["already in pool", "already being validated", "not found or already spent", "duplicate"].iter().any(|m| s.contains(m))
}

/// Both chains as one tick sees them.
struct LiveChains<'a> {
    mds: midstate::core::State,
    mds_mempool: Vec<Transaction>,
    mds_node: &'a midstate::node::NodeHandle,
    mwb: midwimble::core::State,
    mwb_mempool: Vec<MwbTx>,
    mwb_node: &'a midwimble::node::NodeHandle,
}

impl Chains for LiveChains<'_> {
    fn mds_tip(&self) -> u64 {
        self.mds.height
    }
    fn mwb_tip(&self) -> u64 {
        self.mwb.height
    }
    fn mds_coin(&self, id: &[u8; 32]) -> bool {
        self.mds.coins.contains(id)
    }
    fn mds_spend(&self, coin: &[u8; 32], from: u64) -> (Option<Vec<Vec<u8>>>, u64) {
        let find = |tx: &Transaction| -> Option<Vec<Vec<u8>>> {
            let Transaction::Reveal { inputs, witnesses, .. } = tx else { return None };
            let i = inputs.iter().position(|i| &i.coin_id() == coin)?;
            let midstate::core::types::Witness::ScriptInputs(items) = witnesses.get(i)?;
            Some(items.clone())
        };
        if let Some(w) = self.mds_mempool.iter().find_map(find) {
            return (Some(w), from);
        }
        let tip = self.mds.height;
        for h in from..tip {
            match self.mds_node.storage.batches.load(h) {
                Ok(Some(batch)) => {
                    if let Some(w) = batch.transactions.iter().find_map(find) {
                        return (Some(w), h);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("midstate block {h} unreadable: {e:#}");
                    return (None, h);
                }
            }
        }
        (None, tip)
    }
    fn mwb_output(&self, c: &Point32) -> Option<(u64, Point32)> {
        self.mwb.utxos.get(c).map(|e| (e.height, e.owner_key))
    }
    fn mwb_kernel(&self, excess: &Point32, owner_excess: &Point32, from: u64) -> (Option<Kernel>, u64) {
        let find = |tx: &MwbTx| tx.body.kernels.iter().find(|k| &k.excess == excess && &k.owner_excess == owner_excess).cloned();
        if let Some(k) = self.mwb_mempool.iter().find_map(find) {
            return (Some(k), from);
        }
        let tip = self.mwb.height;
        for h in from..tip {
            match self.mwb_node.storage().load_batch(h) {
                Ok(Some(b)) => {
                    if let Some(k) = find(&b.body) {
                        return (Some(k), h);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("midwimble block {h} unreadable: {e:#}");
                    return (None, h);
                }
            }
        }
        (None, tip)
    }
}

impl Service {
    // ── Lifecycle ───────────────────────────────────────────────────────

    /// On unlock: the account's keys and the journal, from wallet.dat.
    pub(super) fn mwb_load(&mut self) {
        let Some(w) = self.wallet.as_ref() else { return };
        self.mwb.keys = w.data.mwb_keys.as_ref().and_then(WalletKeys::from_bytes);
        self.mwb.store = match MwbStore::load(w.data.mwb_state.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                let _ = self.events.send(WalletEvent::Warning {
                    text: format!("The MDW account's records could not be read ({e:#}); it will scan midwimble again."),
                });
                MwbStore::default()
            }
        };
        self.mwb.loaded = true;
        // Negotiations are never journalled, so one a restart cut short left
        // only its reservations and its order's status behind.
        let store = &mut self.mwb.store;
        let swaps: HashSet<String> = store.swaps.iter().map(MwbSwap::swap_tag).collect();
        let stale: HashSet<String> = store
            .account
            .coins
            .iter()
            .filter_map(|c| c.reserved.clone())
            .filter(|r| r.starts_with("swap:") && !swaps.contains(r))
            .collect();
        for r in stale {
            store.account.release(&r);
        }
        let ids: HashSet<[u8; 32]> = store.swaps.iter().map(|s| s.id).collect();
        for o in &mut store.orders {
            if let OrderStatus::Taken { swap } = o.status {
                if !ids.contains(&swap) {
                    o.status = OrderStatus::Open;
                }
            }
        }
    }

    /// On lock: nothing of midwimble stays in memory. Negotiations under way
    /// are dropped; nothing was locked for them.
    pub(super) fn mwb_unload(&mut self) {
        self.mwb = MwbLive::default();
    }

    /// The journal into wallet.dat, and wallet.dat to disk.
    pub(super) fn mwb_save(&mut self) -> Result<()> {
        if !self.mwb.loaded {
            return Ok(());
        }
        let v = self.mwb.store.to_value()?;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        w.data.mwb_state = Some(v);
        w.save()
    }

    /// A new wallet's account starts at midwimble's tip: a fresh phrase has
    /// no history to find.
    pub(super) fn mwb_birth_state(&self) -> Option<serde_json::Value> {
        let tip = self.mwb_node.as_ref()?.state().height;
        MwbStore { account: MwbAccount::starting_at(tip), ..MwbStore::default() }.to_value().ok()
    }

    fn mwb_tip(&self) -> Option<u64> {
        self.mwb_node.as_ref().map(|h| h.state().height)
    }

    fn mwb_synced(&self) -> bool {
        self.mwb_node.as_ref().is_some_and(|h| {
            let i = h.info();
            !i.syncing && !i.peers.is_empty()
        })
    }

    /// The account's keys, or why there are none.
    fn mwb_keys(&self) -> Result<WalletKeys> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        self.mwb.keys.clone().ok_or_else(|| {
            anyhow!("this wallet has no MDW account yet: enter your recovery phrase on the Receive tab to add it")
        })
    }

    /// Spending MDW needs midwimble alone: its node in step with the
    /// network, and the account scanned to its tip (else the coins it picks
    /// may be spent already, or a payment may be missed).
    fn mwb_ready_to_spend(&self) -> Result<WalletKeys> {
        let keys = self.mwb_keys()?;
        ensure!(self.mwb_node.is_some(), "the midwimble node is not running");
        ensure!(self.mwb_synced(), "midwimble is still syncing, or has no peers yet");
        let tip = self.mwb_tip().unwrap_or(0);
        ensure!(
            self.mwb.store.account.scanned >= tip,
            "the MDW account is still scanning midwimble (block {} of {tip})",
            self.mwb.store.account.scanned
        );
        Ok(keys)
    }

    /// A swap reads both chains, so midstate must be in step too.
    fn mwb_ready_to_trade(&self) -> Result<WalletKeys> {
        let keys = self.mwb_ready_to_spend()?;
        ensure!(!self.node.is_syncing(), "midstate is still syncing");
        Ok(keys)
    }

    // ── Commands ────────────────────────────────────────────────────────

    pub(super) fn mwb_account_view(&self) -> MwbAccountView {
        let node = self.mwb_node.as_ref().map(|h| {
            let i = h.info();
            MwbNodeView {
                height: i.state.height,
                peers: i.peers.len(),
                syncing: i.syncing,
                explorer: self.mwb_place.as_ref().and_then(|p| p.explorer.clone()),
            }
        });
        let tip = node.as_ref().map_or(0, |n| n.height);
        let account = &self.mwb.store.account;
        let b = account.balance(tip);
        MwbAccountView {
            enabled: self.mwb.keys.is_some(),
            address: self.mwb.keys.as_ref().map(|k| k.address().encode()),
            spendable: b.spendable,
            immature: b.immature,
            reserved: b.reserved,
            scanned: account.scanned,
            node,
            coins: account.coins.iter().filter(|c| c.spent_height.is_none()).count(),
            spendable_coins: account.spendable_values(tip),
        }
    }

    pub(super) fn mwb_node_info(&self) -> Option<MwbNodeInfo> {
        let h = self.mwb_node.as_ref()?;
        let place = self.mwb_place.as_ref();
        let i = h.info();
        let s = &i.state;
        let difficulty_bits = {
            let mut bits = 0u32;
            for b in s.target {
                if b == 0 {
                    bits += 8;
                } else {
                    bits += b.leading_zeros();
                    break;
                }
            }
            bits
        };
        Some(MwbNodeInfo {
            height: s.height,
            tip_timestamp: s.timestamp,
            header_hash: hex::encode(s.header_hash),
            midstate: hex::encode(s.mw_midstate),
            depth: s.depth.to_string(),
            difficulty_bits,
            // `height` counts the blocks applied, so it is the next block's.
            block_reward: midwimble::core::types::block_reward(s.height),
            supply: s.supply,
            utxo_count: s.utxos.len(),
            bonds: s.bonds.len(),
            mempool: i.mempool_count,
            syncing: i.syncing,
            peer_id: i.peer_id.clone(),
            peers: i
                .peer_details
                .iter()
                .map(|p| MwbPeerView { id: p.id.clone(), addr: p.addr.clone(), outbound: p.outbound })
                .collect(),
            data_dir: place.map(|p| p.data_dir.clone()).unwrap_or_default(),
            p2p_port: place.map_or(0, |p| p.p2p_port),
            explorer: place.and_then(|p| p.explorer.clone()),
        })
    }

    /// Coins swaps paid into the account: bought ones (we claimed the MDW
    /// lock) and refunds (our own lock came back).
    fn mwb_swap_outputs(&self) -> SwapOutputs {
        let mut out = SwapOutputs::new();
        for sw in &self.mwb.store.swaps {
            let id = hex::encode(sw.id);
            let (claim, kind) = if sw.funder() { (false, "refund") } else { (true, "swap") };
            if let Ok(c) = sw.mwb_spend_output(claim) {
                out.insert(c, (kind, id));
            }
        }
        out
    }

    pub(super) fn mwb_history_view(&self) -> Vec<MwbHistoryView> {
        if self.mwb.keys.is_none() {
            return Vec::new();
        }
        let tip = self.mwb_tip().unwrap_or(0);
        let mut rows = self.mwb.store.account.history(tip, &self.mwb_swap_outputs());
        // Coins found before the account kept block times: ask the chain.
        if let Some(h) = self.mwb_node.as_ref() {
            for r in rows.iter_mut().filter(|r| r.time == 0 && r.incoming) {
                if let Some(height) = r.height {
                    if let Ok(Some(header)) = h.storage().load_header(height) {
                        r.time = header.timestamp;
                    }
                }
            }
        }
        rows
    }

    pub(super) fn mwb_coins_view(&self) -> Vec<MwbCoinView> {
        if self.mwb.keys.is_none() {
            return Vec::new();
        }
        let mut v = self.mwb.store.account.coin_views(self.mwb_tip().unwrap_or(0));
        if let Some(h) = self.mwb_node.as_ref() {
            for c in v.iter_mut().filter(|c| c.time == 0) {
                if let Ok(Some(header)) = h.storage().load_header(c.height) {
                    c.time = header.timestamp;
                }
            }
        }
        v
    }

    /// Scans midwimble for the account again from `from`. Refused while a
    /// payment waits for a block or a swap runs: a rescan forgets coins and
    /// finds them again, and what holds them is not part of the chain.
    pub(super) fn mwb_rescan(&mut self, from: u64) -> Result<()> {
        self.mwb_keys()?;
        let account = &self.mwb.store.account;
        ensure!(
            !account.payments.iter().any(|p| p.pending()),
            "an MDW payment is still waiting for a block: rescan once it has one"
        );
        ensure!(
            self.mwb.negs.is_empty() && self.mwb.store.swaps.iter().all(|s| s.phase.settled()),
            "a swap is under way: rescan once it has finished"
        );
        self.mwb.store.account.rescan_from(from);
        self.mwb_save()
    }

    /// Adds the MDW account to a wallet made before it existed. The phrase
    /// must be this wallet's: the account is the one midwimble's own wallets
    /// derive from it.
    pub(super) async fn mwb_enable(&mut self, phrase: &str) -> Result<()> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        ensure!(w.data.mwb_keys.is_none(), "this wallet already has its MDW account");
        ensure!(self.verify_phrase(phrase)?, "that is not this wallet's recovery phrase");
        let phrase = phrase.to_string();
        let keys = tokio::task::spawn_blocking(move || crate::mwb::keys_from_mnemonic(&phrase))
            .await
            .context("key derivation panicked")??;
        let w = self.wallet.as_mut().unwrap();
        w.data.mwb_keys = Some(keys.to_bytes());
        // The phrase may have history in midwimble's own wallets: scan it all.
        self.mwb.store = MwbStore::default();
        self.mwb.keys = Some(keys);
        self.mwb.loaded = true;
        self.mwb_save()
    }

    pub(super) async fn mwb_send(&mut self, to: &str, amount: u64) -> Result<String> {
        let keys = self.mwb_ready_to_spend()?;
        let to = StealthAddress::decode(to.trim()).context("not a midwimble address")?;
        let h = self.mwb_node.clone().unwrap();
        let tip = h.state().height;
        let tx = self.mwb.store.account.build_send(&keys, &to, amount, tip)?;
        let kernel = tx.body.kernels[0].id();
        // Recorded (its coins reserved) before it leaves.
        self.mwb_save()?;
        match h.submit_transaction(tx).await {
            Ok(()) => {}
            Err(e) if harmless(&e) => {}
            Err(e) => {
                self.mwb.store.account.discard_payment(&kernel);
                self.mwb_save()?;
                return Err(e.context("the midwimble node refused the payment"));
            }
        }
        self.mwb.store.account.mark_sent(&kernel, now_secs());
        Ok(hex::encode(kernel))
    }

    pub(super) fn mwb_book_view(&self) -> Vec<MwbBookEntryView> {
        let tip = self.node_tip_cached();
        let mine: HashSet<[u8; 16]> = self.mwb.store.orders.iter().map(|o| o.id).collect();
        let mut v: Vec<MwbBookEntryView> = self
            .mwb
            .book
            .values()
            .filter(|e| e.advert.expires > tip)
            .map(|e| MwbBookEntryView {
                order_id: hex::encode(e.advert.order_id),
                side: side_name(e.advert.side).into(),
                mds_amount: e.advert.mds_amount,
                mdw_amount: e.advert.mdw_amount,
                expires: e.advert.expires,
                heard: e.heard,
                mine: mine.contains(&e.advert.order_id),
            })
            .collect();
        for o in self.mwb.store.orders.iter().filter(|o| o.takeable(tip)) {
            if !self.mwb.book.contains_key(&o.id) {
                v.push(MwbBookEntryView {
                    order_id: hex::encode(o.id),
                    side: side_name(o.side).into(),
                    mds_amount: o.mds_amount,
                    mdw_amount: o.mdw_amount,
                    expires: o.expires,
                    heard: o.last_advert,
                    mine: true,
                });
            }
        }
        v
    }

    fn node_tip_cached(&self) -> u64 {
        self.scan_pos.max(self.mwb_last_mds_tip)
    }

    pub(super) async fn mwb_place_order(&mut self, side: &str, mds_amount: u64, mdw_amount: u64, life: u64) -> Result<String> {
        self.mwb_ready_to_trade()?;
        self.ensure_restore_done()?;
        check_terms(mds_amount, mdw_amount)?;
        let side = match side {
            "sell_mds" => Side::SellMds,
            "sell_mdw" => Side::SellMdw,
            _ => bail!("side must be sell_mds or sell_mdw"),
        };
        let tip = self.node.get_state().await.height;
        let mwb_tip = self.mwb_tip().unwrap_or(0);
        match side {
            Side::SellMds => {
                let bal = self.balance().await?;
                let free = bal.confirmed.saturating_sub(bal.in_flight);
                ensure!(
                    free >= mds_amount + 1_000,
                    "not enough confirmed MDS: {free} units free, the order needs {mds_amount} and a fee"
                );
            }
            Side::SellMdw => {
                self.mwb.store.account.select(mdw_amount, mwb_tip, |n| min_fee(n, 2, 1))?;
            }
        }
        let life = life.clamp(MIN_ORDER_LIFE, MAX_ORDER_LIFE);
        let order = MwbOrder {
            id: rand::random(),
            side,
            mds_amount,
            mdw_amount,
            key: KeyPair::random().to_bytes(),
            created: tip,
            expires: tip + life,
            status: OrderStatus::Open,
            last_advert: 0,
        };
        let id = hex::encode(order.id);
        self.mwb.store.orders.push(order);
        self.mwb_save()?;
        self.mwb_advertise(tip);
        Ok(id)
    }

    pub(super) fn mwb_cancel_order(&mut self, id: &str) -> Result<()> {
        let raw: [u8; 16] = hex::decode(id.trim())?.try_into().map_err(|_| anyhow!("an order id is 16 bytes"))?;
        let o = self.mwb.store.order_mut(&raw).ok_or_else(|| anyhow!("no such order"))?;
        match o.status {
            OrderStatus::Open => o.status = OrderStatus::Cancelled,
            OrderStatus::Taken { .. } => bail!("a swap on this order is under way; it cannot be cancelled now"),
            _ => bail!("this order is already closed"),
        }
        self.mwb_save()
    }

    pub(super) fn mwb_orders_view(&self) -> Vec<MwbOrderView> {
        let tip = self.node_tip_cached();
        self.mwb
            .store
            .orders
            .iter()
            .rev()
            .map(|o| {
                let (status, swap) = match &o.status {
                    OrderStatus::Open if tip >= o.expires => ("expired", None),
                    OrderStatus::Open => ("open", None),
                    OrderStatus::Taken { swap } => ("taken", Some(hex::encode(swap))),
                    OrderStatus::Filled { swap } => ("filled", Some(hex::encode(swap))),
                    OrderStatus::Cancelled => ("cancelled", None),
                };
                MwbOrderView {
                    order_id: hex::encode(o.id),
                    side: side_name(o.side).into(),
                    mds_amount: o.mds_amount,
                    mdw_amount: o.mdw_amount,
                    expires: o.expires,
                    status: status.into(),
                    swap,
                }
            })
            .collect()
    }

    pub(super) async fn mwb_take(&mut self, order_id: &str) -> Result<String> {
        let keys = self.mwb_ready_to_trade()?;
        self.ensure_restore_done()?;
        let raw: [u8; 16] = hex::decode(order_id.trim())?.try_into().map_err(|_| anyhow!("an order id is 16 bytes"))?;
        ensure!(!self.mwb.store.orders.iter().any(|o| o.id == raw), "that order is yours");
        let entry = self.mwb.book.get(&raw).cloned().ok_or_else(|| anyhow!("that order is no longer on the bus"))?;
        ensure!(
            !self.mwb.negs.values().any(|n| n.swap.order_id == raw),
            "you are already negotiating this order"
        );
        let mds_tip = self.node.get_state().await.height;
        if entry.advert.side == Side::SellMdw {
            // We will lock the MDS.
            let bal = self.balance().await?;
            let free = bal.confirmed.saturating_sub(bal.in_flight);
            ensure!(
                free >= entry.advert.mds_amount + 1_000,
                "not enough confirmed MDS: {free} units free, the swap needs {} and a fee",
                entry.advert.mds_amount
            );
        }
        let key = KeyPair::random();
        let mwb_tip = self.mwb_tip().unwrap_or(0);
        let mut ctx = Ctx { mds_tip, mwb_tip, keys: &keys, account: &mut self.mwb.store.account, now: now_secs() };
        let (neg, out) = Negotiation::take(&entry.advert, &key, &mut ctx)?;
        let id = hex::encode(neg.swap.id);
        self.mwb_send_chat(out);
        self.mwb.negs.insert(neg.swap.session.tag, neg);
        Ok(id)
    }

    pub(super) fn mwb_swaps_view(&self) -> Vec<MwbSwapView> {
        let view = |s: &MwbSwap| {
            let (reveal_chain, follow_chain) = match s.reveal_leg() {
                Leg::Mwb => ("midwimble", "midstate"),
                Leg::Mds => ("midstate", "midwimble"),
            };
            MwbSwapView {
                id: hex::encode(s.id),
                role: match s.role {
                    Role::Maker => "maker".into(),
                    Role::Taker => "taker".into(),
                },
                side: side_name(s.side).into(),
                mds_amount: s.mds_amount,
                mdw_amount: s.mdw_amount,
                phase: phase_name(&s.phase).into(),
                note: match &s.phase {
                    Phase::Failed(r) => r.clone(),
                    Phase::Done => format!("complete: {} received", coin_name(s.their_leg())),
                    Phase::Refunded => format!("refunded: your {} came back", coin_name(s.my_leg())),
                    _ => s.note.clone(),
                },
                settled: s.phase.settled(),
                gives: coin_name(s.my_leg()).into(),
                gets: coin_name(s.their_leg()).into(),
                reveal_height: s.reveal_height,
                reveal_chain: reveal_chain.into(),
                follow_height: s.follow_height,
                follow_chain: follow_chain.into(),
                created: s.created,
                updated: s.updated,
            }
        };
        let mut v: Vec<MwbSwapView> = self.mwb.negs.values().map(|n| view(&n.swap)).collect();
        v.extend(self.mwb.store.swaps.iter().rev().map(view));
        v
    }

    // ── The tick ────────────────────────────────────────────────────────

    pub(super) async fn tick_mwb(&mut self, mds_tip: u64) {
        self.mwb_last_mds_tip = mds_tip;
        let (Some(h), Some(keys)) = (self.mwb_node.clone(), self.mwb.keys.clone()) else { return };
        if !self.mwb.loaded {
            return;
        }
        if let Err(e) = self.mwb_scan(&h, &keys).await {
            tracing::warn!("midwimble scan: {e:#}");
        }
        self.mwb_backfill_times(&h);
        self.mwb_rebroadcast_sends(&h).await;
        // Orders, negotiations and swaps read midstate's tip; while midstate
        // catches up that tip is stale, so they wait for it.
        if self.node.is_syncing() {
            return;
        }
        let now = now_secs();
        self.mwb_read_bus(mds_tip, &keys, now).await;
        self.mwb_negotiation_timers(now);
        self.mwb_advertise(mds_tip);
        self.mwb.book.retain(|_, e| e.advert.expires > mds_tip && mds_tip <= e.heard + ADVERT_STALE);
        self.mwb.lingers.retain(|_, l| l.until > now);
        if let Err(e) = self.mwb_advance_swaps(&h).await {
            tracing::warn!("swap tick: {e:#}");
        }
    }

    async fn mwb_scan(&mut self, h: &midwimble::node::NodeHandle, keys: &WalletKeys) -> Result<()> {
        let tip = h.state().height;
        let start = self.mwb.store.account.scanned;
        if start >= tip {
            return Ok(());
        }
        let end = tip.min(start + MWB_SCAN_CHUNK);
        let storage = h.storage().clone();
        let blocks = tokio::task::spawn_blocking(move || -> Result<Vec<(u64, midwimble::core::types::Batch)>> {
            let mut out = Vec::new();
            for height in start..end {
                match storage.load_batch(height)? {
                    Some(b) => out.push((height, b)),
                    None => break,
                }
            }
            Ok(out)
        })
        .await
        .context("scan task panicked")??;
        if blocks.is_empty() {
            bail!("midwimble block {start} is not in the node's storage (pruned?)");
        }
        let mut changed = false;
        let mut received = 0u64;
        for (height, b) in &blocks {
            match self.mwb.store.account.scan(keys, &BlockView::of(*height, b)) {
                Ok(d) => {
                    received += d.received.iter().map(|(v, _)| v).sum::<u64>();
                    changed |= d.changed();
                }
                Err(e) if e.to_string().contains(REORG) => {
                    self.mwb.store.account.step_back();
                    changed = true;
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        let crossed = start / MWB_SAVE_EVERY != self.mwb.store.account.scanned / MWB_SAVE_EVERY;
        if changed || crossed {
            self.mwb_save()?;
        }
        if changed {
            let _ = self.events.send(WalletEvent::WalletChanged);
        }
        if received > 0 {
            self.ch_notice(format!("{} MDW arrived in your midwimble account.", midwimble::core::types::format_amount(received)));
        }
        Ok(())
    }

    /// Coins found before the account kept block times get theirs from the
    /// chain, a few hundred a tick, so the history needs no lookups.
    fn mwb_backfill_times(&mut self, h: &midwimble::node::NodeHandle) {
        let mut filled = 0;
        for c in self.mwb.store.account.coins.iter_mut().filter(|c| c.time == 0).take(300) {
            if let Ok(Some(header)) = h.storage().load_header(c.height) {
                c.time = header.timestamp;
                filled += 1;
            }
        }
        if filled > 0 {
            if let Err(e) = self.mwb_save() {
                tracing::warn!("MDW account not saved: {e:#}");
            }
        }
    }

    async fn mwb_rebroadcast_sends(&mut self, h: &midwimble::node::NodeHandle) {
        let now = now_secs();
        let due: Vec<([u8; 32], MwbTx)> = self
            .mwb
            .store
            .account
            .due_for_broadcast(now, MWB_REBROADCAST_SECS)
            .into_iter()
            .filter(|(k, _)| self.mwb.store.account.payments.iter().any(|p| &p.kernel == k && p.label != LOCK_LABEL))
            .collect();
        for (kernel, tx) in due {
            match h.submit_transaction(tx).await {
                Ok(()) => {}
                Err(e) if harmless(&e) => {}
                Err(e) => tracing::warn!("midwimble payment {} not rebroadcast: {e:#}", hex::encode(&kernel[..8])),
            }
            self.mwb.store.account.mark_sent(&kernel, now);
        }
    }

    pub(super) fn mwb_send_chat(&self, out: Vec<Outgoing>) {
        for (words, attachments) in out {
            if let Err(e) = self.node.send_chat(words, None, attachments) {
                tracing::warn!("swap message not sent: {e:#}");
            }
        }
    }

    /// Re-advertises open orders.
    fn mwb_advertise(&mut self, mds_tip: u64) {
        if !self.mwb_synced() || self.node.is_syncing() {
            return;
        }
        let mut out = Vec::new();
        for o in &mut self.mwb.store.orders {
            if o.takeable(mds_tip) && (o.last_advert == 0 || mds_tip >= o.last_advert + ADVERT_EVERY) {
                if let Ok(a) = o.advert() {
                    out.push(a.chat());
                    o.last_advert = mds_tip;
                }
            }
        }
        self.mwb_send_chat(out);
    }

    async fn mwb_read_bus(&mut self, mds_tip: u64, keys: &WalletKeys, now: u64) {
        let hist = self.node.chat_history.read().await.clone();
        for m in hist {
            if !self.mwb.first_sight((m.sender.clone(), m.timestamp, m.nonce)) {
                continue;
            }
            let Some(frame) = parse(&m) else { continue };
            match frame {
                Frame::Advert(a) => {
                    if self.mwb.store.orders.iter().any(|o| o.id == a.order_id) || a.expires <= mds_tip {
                        continue;
                    }
                    if check_terms(a.mds_amount, a.mdw_amount).is_ok() {
                        self.mwb.book.insert(a.order_id, BookEntry { advert: a, heard: mds_tip });
                    }
                }
                Frame::Take { order_id, taker_key, index, count, chunk } => {
                    if !self.mwb.store.orders.iter().any(|o| o.id == order_id) {
                        continue;
                    }
                    let route = Route::Take { order_id, taker_key };
                    if let Some(sealed) = self.mwb.inbox.add(route, index, count, chunk, now) {
                        self.mwb_on_take(order_id, taker_key, sealed, mds_tip, keys, now).await;
                    }
                }
                Frame::Session { tag, to, seq, index, count, chunk } => {
                    let role = self.mwb.negs.get(&tag).map(|n| n.swap.role).or_else(|| self.mwb.lingers.get(&tag).map(|l| l.role));
                    let Some(role) = role else { continue };
                    let ours = match role {
                        Role::Maker => To::Maker,
                        Role::Taker => To::Taker,
                    };
                    if to != ours {
                        continue;
                    }
                    let route = Route::Session { tag, to, seq };
                    if let Some(sealed) = self.mwb.inbox.add(route.clone(), index, count, chunk, now) {
                        self.mwb_on_session(tag, route, seq, sealed, mds_tip, keys, now).await;
                    }
                }
            }
        }
    }

    async fn mwb_on_take(&mut self, order_id: [u8; 16], taker_key: Point32, sealed: Vec<u8>, mds_tip: u64, keys: &WalletKeys, now: u64) {
        let Some(order) = self.mwb.store.orders.iter().find(|o| o.id == order_id).cloned() else { return };
        let Some(kp) = KeyPair::from_bytes(&order.key) else { return };
        let Ok(session) = maker_session(&kp, &order_id, &taker_key) else { return };
        if let Some(n) = self.mwb.negs.get(&session.tag) {
            // The taker did not hear our offer: say it again.
            let out = n.last_out.clone();
            self.mwb_send_chat(out);
            return;
        }
        let route = Route::Take { order_id, taker_key };
        let Ok(Msg::Take(take)) = route.open(&session, &sealed) else { return };
        let refuse = |why: String| session_messages(&session, To::Taker, 1, &Msg::Abort { reason: why }).unwrap_or_default();
        if !order.takeable(mds_tip) {
            self.mwb_send_chat(refuse("the order is no longer open".into()));
            return;
        }
        if let Err(e) = self.mwb_ready_to_trade() {
            self.mwb_send_chat(refuse(format!("the maker cannot trade right now: {e}")));
            return;
        }
        let mwb_tip = self.mwb_tip().unwrap_or(0);
        let mut ctx = Ctx { mds_tip, mwb_tip, keys, account: &mut self.mwb.store.account, now };
        match Negotiation::answer(&order, take, session.clone(), &mut ctx) {
            Ok((neg, out)) => {
                if let Some(o) = self.mwb.store.order_mut(&order_id) {
                    o.status = OrderStatus::Taken { swap: neg.swap.id };
                }
                self.mwb_send_chat(out);
                self.ch_notice(format!(
                    "Someone is taking your order for {} — negotiating the swap.",
                    describe(&order)
                ));
                self.mwb.negs.insert(session.tag, neg);
            }
            Err(e) => {
                tracing::info!("refused a take: {e:#}");
                self.mwb_send_chat(refuse(format!("{e}")));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn mwb_on_session(&mut self, tag: [u8; 16], route: Route, seq: u8, sealed: Vec<u8>, mds_tip: u64, keys: &WalletKeys, now: u64) {
        if let Some(l) = self.mwb.lingers.get(&tag) {
            if !self.mwb.negs.contains_key(&tag) {
                if seq == l.answers && route.open(&l.session, &sealed).is_ok() {
                    let out = l.last_out.clone();
                    self.mwb_send_chat(out);
                }
                return;
            }
        }
        let mwb_tip = self.mwb_tip().unwrap_or(0);
        let Some(neg) = self.mwb.negs.get_mut(&tag) else { return };
        let Ok(msg) = route.open(&neg.swap.session, &sealed) else { return };
        let mut ctx = Ctx { mds_tip, mwb_tip, keys, account: &mut self.mwb.store.account, now };
        match neg.receive(seq, msg, &mut ctx) {
            Ok(Step::Reply(out)) => self.mwb_send_chat(out),
            Ok(Step::Resend) => {
                neg.last_sent = now;
                let out = neg.last_out.clone();
                self.mwb_send_chat(out);
            }
            Ok(Step::Ignore) => {}
            Ok(Step::Complete(out)) => {
                let sent_last = !out.is_empty();
                self.mwb_send_chat(out);
                let neg = self.mwb.negs.remove(&tag).unwrap();
                if sent_last {
                    self.mwb.lingers.insert(tag, Linger::of(&neg, now));
                }
                self.mwb_agreed(neg, now);
            }
            Err(e) => self.mwb_fail_negotiation(&tag, &format!("{e:#}"), true),
        }
    }

    /// A negotiation finished: journal the swap before anything is locked.
    fn mwb_agreed(&mut self, neg: Negotiation, now: u64) {
        let mut sw = neg.swap;
        sw.updated = now;
        sw.note = "agreed; locks next".into();
        if sw.role == Role::Maker {
            if let Some(o) = self.mwb.store.order_mut(&sw.order_id) {
                o.status = OrderStatus::Filled { swap: sw.id };
            }
        }
        let text = format!(
            "Swap agreed: you give {} {} for {} {}. The wallet now watches both chains until it completes.",
            amount_of(&sw, sw.my_leg()),
            coin_name(sw.my_leg()),
            amount_of(&sw, sw.their_leg()),
            coin_name(sw.their_leg()),
        );
        self.mwb.store.swaps.push(sw);
        if let Err(e) = self.mwb_save() {
            tracing::error!("swap journal save failed: {e:#}");
        }
        self.ch_notice(text);
    }

    fn mwb_fail_negotiation(&mut self, tag: &[u8; 16], reason: &str, tell: bool) {
        let Some(neg) = self.mwb.negs.remove(tag) else { return };
        neg.abandon(&mut self.mwb.store.account);
        if tell && !reason.starts_with("the other side stopped") {
            let out = abort_messages(&neg, reason);
            self.mwb_send_chat(out);
        }
        if neg.swap.role == Role::Maker {
            if let Some(o) = self.mwb.store.order_mut(&neg.swap.order_id) {
                if o.status == (OrderStatus::Taken { swap: neg.swap.id }) {
                    o.status = OrderStatus::Open;
                }
            }
        }
        self.ch_notice(format!("A swap negotiation ended before anything was locked: {reason}"));
    }

    fn mwb_negotiation_timers(&mut self, now: u64) {
        let stalled: Vec<[u8; 16]> = self
            .mwb
            .negs
            .iter()
            .filter(|(_, n)| now.saturating_sub(n.progress) > NEGOTIATION_SECS)
            .map(|(t, _)| *t)
            .collect();
        for tag in stalled {
            self.mwb_fail_negotiation(&tag, "the other side went quiet", true);
        }
        let mut out = Vec::new();
        for n in self.mwb.negs.values_mut() {
            if now.saturating_sub(n.last_sent) >= RESEND_SECS && n.resends < MAX_RESENDS {
                n.resends += 1;
                n.last_sent = now;
                out.extend(n.last_out.clone());
            }
        }
        self.mwb_send_chat(out);
    }

    // ── Swaps ───────────────────────────────────────────────────────────

    async fn mwb_advance_swaps(&mut self, h: &midwimble::node::NodeHandle) -> Result<()> {
        if self.mwb.store.swaps.iter().all(|s| s.phase.settled()) {
            return Ok(());
        }
        // Deadlines are read off both tips: act only on chains that are
        // current.
        if self.node.is_syncing() || !self.mwb_synced() {
            return Ok(());
        }
        let (_, mds_mempool) = self.node.get_mempool_info().await;
        let mwb_mempool = h.mempool().await.unwrap_or_default();
        let chains = LiveChains {
            mds: self.node.get_state().await,
            mds_mempool,
            mds_node: &self.node,
            mwb: h.state(),
            mwb_mempool,
            mwb_node: h,
        };
        let now = now_secs();
        let mut decisions = Vec::new();
        let mut journal_changed = false;
        for i in 0..self.mwb.store.swaps.len() {
            if self.mwb.store.swaps[i].phase.settled() {
                continue;
            }
            // A funding transaction that never reached a block in all the
            // time its refund took to open is gone: stop holding its coins.
            let sw = &self.mwb.store.swaps[i];
            if sw.funder()
                && sw.phase == Phase::LockingMine
                && sw.mwb_seen.is_none()
                && chains.mwb.height >= sw.my_refund_height() + 10
            {
                if let Some(k) = sw.funding().ok().and_then(|f| f.body.kernels.first().map(Kernel::id)) {
                    self.mwb.store.account.discard_payment(&k);
                    journal_changed = true;
                }
            }
            let pending = self.mwb_lock_pending(&self.mwb.store.swaps[i]);
            let sw = &mut self.mwb.store.swaps[i];
            let before = (sw.mds_seen, sw.mwb_seen, sw.t, sw.mds_scan_from, sw.mwb_scan_from);
            let o = observe(sw, &chains, pending);
            if before != (sw.mds_seen, sw.mwb_seen, sw.t, sw.mds_scan_from, sw.mwb_scan_from) {
                journal_changed = true;
            }
            let d = decide(sw, &o);
            if d.phase != sw.phase {
                journal_changed = true;
                sw.updated = now;
            }
            sw.phase = d.phase.clone();
            if !d.note.is_empty() {
                sw.note = d.note.clone();
            }
            decisions.push((i, d, o.mds_tip, o.mwb_tip));
        }
        drop(chains);
        if journal_changed {
            self.mwb_save()?;
        }
        for (i, d, mds_tip, mwb_tip) in decisions {
            let settled = self.mwb.store.swaps[i].phase.settled();
            if settled {
                self.mwb_settled(i);
                continue;
            }
            let r = match d.act {
                Act::None => Ok(()),
                Act::LockMine => self.mwb_lock_mine(i, mds_tip, mwb_tip).await,
                Act::Claim => self.mwb_spend(i, true, mds_tip, mwb_tip).await,
                Act::Refund => self.mwb_spend(i, false, mds_tip, mwb_tip).await,
            };
            if let Err(e) = r {
                let sw = &mut self.mwb.store.swaps[i];
                tracing::warn!(swap = %hex::encode(&sw.id[..8]), "swap step deferred: {e:#}");
                sw.note = format!("{} (retrying: {e})", d.note);
            }
        }
        Ok(())
    }

    /// Whether our lock's broadcast is still under way.
    fn mwb_lock_pending(&self, sw: &MwbSwap) -> bool {
        match sw.my_leg() {
            Leg::Mds => sw
                .mds_lock_commitment
                .is_some_and(|c| self.wallet.as_ref().is_some_and(|w| w.find_pending(&c).is_some())),
            Leg::Mwb => sw.funding().ok().and_then(|f| f.body.kernels.first().map(Kernel::id)).is_some_and(|k| {
                self.mwb.store.account.payments.iter().any(|p| p.kernel == k && p.pending())
            }),
        }
    }

    fn mwb_settled(&mut self, i: usize) {
        let sw = &self.mwb.store.swaps[i];
        let text = match &sw.phase {
            Phase::Done => format!(
                "Swap complete: {} {} received.",
                amount_of(sw, sw.their_leg()),
                coin_name(sw.their_leg())
            ),
            Phase::Refunded => format!("Swap refunded: your {} {} came back.", amount_of(sw, sw.my_leg()), coin_name(sw.my_leg())),
            Phase::Failed(r) => format!("Swap ended: {r}"),
            _ => return,
        };
        let tag = sw.swap_tag();
        // Coins set aside for a lock that never left are free again.
        self.mwb.store.account.release(&tag);
        self.mwb.store.compact();
        if let Err(e) = self.mwb_save() {
            tracing::error!("swap journal save failed: {e:#}");
        }
        let _ = self.events.send(WalletEvent::WalletChanged);
        self.ch_notice(text);
    }

    fn pushed_due(&self, id: &[u8; 32], what: u8, tip: u64) -> bool {
        self.mwb.pushed.get(&(*id, what)).is_none_or(|last| tip >= last + REBROADCAST_BLOCKS)
    }

    async fn mwb_lock_mine(&mut self, i: usize, mds_tip: u64, mwb_tip: u64) -> Result<()> {
        let sw = self.mwb.store.swaps[i].clone();
        match sw.my_leg() {
            Leg::Mds => {
                if sw.mds_lock_commitment.is_some() {
                    return Ok(()); // the send machine has it
                }
                self.ensure_restore_done()?;
                self.verify_mss_indices().await?;
                let state = self.node.get_state().await;
                let in_flight = self.in_flight_inputs();
                let journal = {
                    let mut store = self.mwb.store.clone();
                    store.swaps[i].mds_lock_commitment = Some([0; 32]);
                    store
                };
                let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
                let live: Vec<[u8; 32]> = w
                    .coins()
                    .iter()
                    .filter(|c| !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id))
                    .map(|c| c.coin_id)
                    .collect();
                let plan = sendplan::plan_fixed_outputs(w, &live, sw.mds_lock_outputs())?;
                let (commitment, _) = w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false)?;
                // The send and the journal entry naming it, in one write.
                let mut journal = journal;
                journal.swaps[i].mds_lock_commitment = Some(commitment);
                w.data.mwb_state = Some(journal.to_value()?);
                w.save()?;
                self.mwb.store = journal;
                self.sends.insert(
                    commitment,
                    SendMeta {
                        stage: SendStage::Committing,
                        detail: "locking MDS for a swap — solving commit proof-of-work".into(),
                        amount: sw.mds_amount,
                        fee: plan.fee,
                        to: "swap lock".into(),
                        updated_at: now_secs(),
                    },
                );
                self.set_stage(commitment, SendStage::Committing, "locking MDS for a swap");
                self.broadcast_commit(commitment).await
            }
            Leg::Mwb => {
                if !self.pushed_due(&sw.id, 0, mwb_tip) {
                    return Ok(());
                }
                let tx = sw.funding()?;
                let tag = sw.swap_tag();
                let kernel = tx.body.kernels[0].id();
                if !self.mwb.store.account.payments.iter().any(|p| p.kernel == kernel) {
                    self.mwb.store.account.record_payment(&tx, "swap lock", sw.mdw_amount, LOCK_LABEL, Some(&tag))?;
                }
                if self.mwb.store.swaps[i].mwb_broadcast.is_none() {
                    self.mwb.store.swaps[i].mwb_broadcast = Some(mwb_tip);
                    self.mwb_save()?;
                }
                self.mwb.pushed.insert((sw.id, 0), mwb_tip);
                let _ = mds_tip;
                match self.mwb_node.as_ref().unwrap().submit_transaction(tx).await {
                    Ok(()) => Ok(()),
                    Err(e) if harmless(&e) => Ok(()),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// A maker's claim is its reveal: once one has gone out, the secret is
    /// public, and the claim is finished whatever the cutoff says.
    fn mwb_mark_revealed(&mut self, i: usize, claim: bool) -> Result<()> {
        let sw = &mut self.mwb.store.swaps[i];
        if claim && sw.role == Role::Maker && !sw.reveal_sent {
            sw.reveal_sent = true;
            self.mwb_save()?;
        }
        Ok(())
    }

    /// Our claim of their lock, or our refund of ours.
    async fn mwb_spend(&mut self, i: usize, claim: bool, mds_tip: u64, mwb_tip: u64) -> Result<()> {
        let sw = self.mwb.store.swaps[i].clone();
        let leg = if claim { sw.their_leg() } else { sw.my_leg() };
        let what = if claim { 1 } else { 2 };
        match leg {
            Leg::Mwb => {
                if !self.pushed_due(&sw.id, what, mwb_tip) {
                    return Ok(());
                }
                let tx = if claim { sw.claim()? } else { sw.refund()? };
                self.mwb.pushed.insert((sw.id, what), mwb_tip);
                match self.mwb_node.as_ref().unwrap().submit_transaction(tx).await {
                    Ok(()) => {}
                    Err(e) if harmless(&e) => {}
                    Err(e) => return Err(e),
                }
                self.mwb_mark_revealed(i, claim)
            }
            Leg::Mds => {
                let kind = if claim { MdsSpendKind::Claim } else { MdsSpendKind::Refund };
                let dest = match sw.mds_dest {
                    Some(d) => d,
                    None => {
                        // Chosen once, and journalled before any commitment
                        // to it: the one-time key signs one message ever.
                        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
                        let d = w.generate_key(Some("swap proceeds".into()))?;
                        self.mwb.store.swaps[i].mds_dest = Some(d);
                        self.mwb_save()?;
                        d
                    }
                };
                let sw = self.mwb.store.swaps[i].clone();
                let sp = build_mds_spend(&sw, kind, &dest)?;
                match sw.mds_spend_commitment {
                    Some(c) if c != sp.commitment => bail!("the spend no longer matches its commitment"),
                    Some(c) if self.node.check_commitment(c).await => {
                        if !self.pushed_due(&sw.id, what, mds_tip) {
                            return Ok(());
                        }
                        self.mwb.pushed.insert((sw.id, what), mds_tip);
                        let tx = sign_mds_spend(&sw, kind, &sp)?;
                        self.node.send_transaction(tx).await.context("the reveal was refused")?;
                        self.mwb_mark_revealed(i, claim)
                    }
                    Some(_) => {
                        // Committed, not yet in a block: commit again if it
                        // seems lost.
                        let since = self.mwb.pushed.get(&(sw.id, 3)).copied().unwrap_or(0);
                        if mds_tip >= since + RECOMMIT_BLOCKS {
                            self.mwb.pushed.insert((sw.id, 3), mds_tip);
                            self.commit_external(sp.commitment).await?;
                        }
                        Ok(())
                    }
                    None => {
                        self.mwb.store.swaps[i].mds_spend_commitment = Some(sp.commitment);
                        self.mwb_save()?;
                        self.mwb.pushed.insert((sw.id, 3), mds_tip);
                        self.commit_external(sp.commitment).await
                    }
                }
            }
        }
    }
}

fn amount_of(sw: &MwbSwap, leg: Leg) -> String {
    match leg {
        Leg::Mds => sw.mds_amount.to_string(),
        Leg::Mwb => midwimble::core::types::format_amount(sw.mdw_amount),
    }
}

fn describe(o: &MwbOrder) -> String {
    match o.side {
        Side::SellMds => format!("{} MDS → {} MDW", o.mds_amount, midwimble::core::types::format_amount(o.mdw_amount)),
        Side::SellMdw => format!("{} MDW → {} MDS", midwimble::core::types::format_amount(o.mdw_amount), o.mds_amount),
    }
}
