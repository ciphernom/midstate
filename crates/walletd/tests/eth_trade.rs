//! MDS <-> ETH trades end to end: each side runs the swap machine
//! (`observe` → `decide` → act) until it settles, against midstate's
//! consensus code and a simulation of the Base contract that mirrors
//! `MidstateAtomicSwap.sol` rule for rule.
//!
//! Midstate spends go through the relay fee floor and `apply_transaction`
//! (the script VM included); lock coins are placed directly, as the send
//! that funds a lock is the wallet's ordinary send.
//!
//! The simulated Base holds both contracts: ETH's and USDC's, every lock
//! and bid tagged with the one it is on, and each paying in its own coin.
//!
//! Beyond the happy paths, the attacks the old sell orders were open to:
//! a stranger holding the secret, the seller holding it, an underpaid or
//! foreign ETH lock, and a buyer whose wallet never saw the claim. For buy
//! orders: a bidder that claims with a witness of its own shape, claims
//! only part of the lock, or claims before the filler ever saw it, and one
//! whose claim would land too late.

use anyhow::{bail, ensure, Result};
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::transaction::apply_transaction;
use midstate::core::decompose_value;
use midstate::core::script::push_int;
use midstate::core::types::{
    compute_coin_id, compute_commitment, hash, is_v2_at, InputReveal, OutputData, Predicate, State as MdsState, Transaction as MdsTx,
    Witness,
};
use midstate::core::wots;
use midstate_walletd::base::BidState;
use midstate_walletd::eth_trade::*;
use std::collections::{HashMap, HashSet};

const GWEI: u128 = 1_000_000_000;

// ── Base, as the contract behaves ──────────────────────────────────────────

#[derive(Clone)]
struct EthLock {
    beneficiary: [u8; 20],
    refund_to: [u8; 20],
    amount: u128,
    timeout: u64,
    hashlock: [u8; 32],
    settled: bool,
    /// Which contract holds it.
    asset: Asset,
}

#[derive(Default)]
struct SimBase {
    locks: HashMap<[u8; 32], EthLock>,
    order: Vec<[u8; 32]>,
    claims: HashMap<[u8; 32], [u8; 32]>,
    refunded: HashSet<[u8; 32]>,
    bids: HashMap<[u8; 32], BidState>,
    bid_claims: HashSet<[u8; 32]>,
    /// ETH, as each account's balance moved.
    balance: HashMap<[u8; 20], i128>,
    /// USDC, the same.
    usdc: HashMap<[u8; 20], i128>,
    /// Which contract each bid is on.
    bid_assets: HashMap<[u8; 32], Asset>,
    n: u64,
}

impl SimBase {
    fn pay(&mut self, asset: Asset, to: [u8; 20], v: i128) {
        let book = match asset {
            Asset::Eth => &mut self.balance,
            Asset::Usdc => &mut self.usdc,
        };
        *book.entry(to).or_default() += v;
    }
    /// An ETH lock.
    fn lock(&mut self, now: u64, from: [u8; 20], hashlock: [u8; 32], beneficiary: [u8; 20], delay: u64, value: u128) -> Result<[u8; 32]> {
        self.lock_in(Asset::Eth, now, from, hashlock, beneficiary, delay, value)
    }
    fn lock_in(&mut self, asset: Asset, now: u64, from: [u8; 20], hashlock: [u8; 32], beneficiary: [u8; 20], delay: u64, value: u128) -> Result<[u8; 32]> {
        ensure!((600..=604_800).contains(&delay), "bad delay");
        ensure!(value > 0 && hashlock != [0; 32], "bad lock");
        self.n += 1;
        let id = hash(&[&from[..], &hashlock, &self.n.to_le_bytes()].concat());
        self.locks.insert(id, EthLock { beneficiary, refund_to: from, amount: value, timeout: now + delay, hashlock, settled: false, asset });
        self.order.push(id);
        self.pay(asset, from, -(value as i128));
        Ok(id)
    }
    /// Anyone may call; the coin always goes to the beneficiary.
    fn claim(&mut self, id: [u8; 32], preimage: [u8; 32]) -> Result<()> {
        let l = self.locks.get_mut(&id).ok_or_else(|| anyhow::anyhow!("not found"))?;
        ensure!(!l.settled, "settled");
        ensure!(hash(&preimage) == l.hashlock, "bad preimage");
        l.settled = true;
        let (to, v, asset) = (l.beneficiary, l.amount, l.asset);
        self.claims.insert(id, preimage);
        self.pay(asset, to, v as i128);
        Ok(())
    }
    fn refund(&mut self, now: u64, id: [u8; 32]) -> Result<()> {
        let l = self.locks.get_mut(&id).ok_or_else(|| anyhow::anyhow!("not found"))?;
        ensure!(!l.settled, "settled");
        ensure!(now >= l.timeout, "too early");
        l.settled = true;
        let (to, v, asset) = (l.refund_to, l.amount, l.asset);
        self.refunded.insert(id);
        self.pay(asset, to, v as i128);
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn create_bid_in(&mut self, asset: Asset, now: u64, maker: [u8; 20], hashlock: [u8; 32], mds_amount: u64, addr: [u8; 32], ttl: u64, bond: u128, value: u128) -> [u8; 32] {
        self.n += 1;
        let id = hash(&[&maker[..], &hashlock, &self.n.to_le_bytes()].concat());
        self.bids.insert(
            id,
            BidState {
                maker,
                filler: [0; 20],
                amount: value,
                fill_bond: bond,
                mds_amount,
                expiry: now + ttl,
                fill_deadline: 0,
                hashlock,
                maker_mds_addr: addr,
                settled: false,
            },
        );
        self.bid_assets.insert(id, asset);
        self.pay(asset, maker, -(value as i128));
        id
    }
    fn reserve_bid(&mut self, now: u64, id: [u8; 32], filler: [u8; 20], window: u64, bond: u128) -> Result<()> {
        let b = self.bids.get_mut(&id).ok_or_else(|| anyhow::anyhow!("not found"))?;
        ensure!(!b.settled && b.filler == [0; 20], "already reserved");
        ensure!(now < b.expiry, "expired");
        ensure!((7_200..=259_200).contains(&window), "bad window");
        ensure!(bond == b.fill_bond, "bad bond");
        b.filler = filler;
        b.fill_deadline = now + window;
        let asset = self.bid_assets[&id];
        self.pay(asset, filler, -(bond as i128));
        Ok(())
    }
    fn claim_bid(&mut self, now: u64, id: [u8; 32], preimage: [u8; 32]) -> Result<()> {
        let b = self.bids.get_mut(&id).ok_or_else(|| anyhow::anyhow!("not found"))?;
        ensure!(!b.settled, "settled");
        ensure!(b.filler != [0; 20], "not reserved");
        ensure!(now <= b.fill_deadline, "fill lapsed");
        ensure!(hash(&preimage) == b.hashlock, "bad preimage");
        b.settled = true;
        let (to, v) = (b.filler, b.amount + b.fill_bond);
        self.bid_claims.insert(id);
        let asset = self.bid_assets[&id];
        self.pay(asset, to, v as i128);
        Ok(())
    }
}

// ── Midstate, with consensus ───────────────────────────────────────────────

struct Sim {
    mds: MdsState,
    reveals: Vec<(u64, MdsTx)>,
    base: SimBase,
    now: u64,
    /// Whether the buyer's Base scan reports claims (a wallet that missed one).
    hide_claims: bool,
    /// Whose turn it is.
    viewer: Option<Role>,
    /// How far Base's clock runs ahead of the wallets'.
    base_ahead: u64,
}

impl Sim {
    fn new() -> Self {
        Self {
            mds: MdsState {
                midstate: [0; 32],
                coins: UtxoAccumulator::new(),
                commitments: UtxoAccumulator::new(),
                depth: 0,
                target: [0xff; 32],
                height: 450_000,
                timestamp: 0,
                commitment_heights: Default::default(),
                expirations: Default::default(),
                chain_mmr: MerkleMountainRange::new(),
                header_hash: [0; 32],
                burned_wots: UtxoAccumulator::new(),
                coinbase_window: Default::default(),
            },
            reveals: Vec::new(),
            base: SimBase::default(),
            now: 1_800_000_000,
            hide_claims: false,
            viewer: None,
            base_ahead: 0,
        }
    }
    /// The time the contract judges by.
    fn base_time(&self) -> u64 {
        self.now + self.base_ahead
    }
    fn mine(&mut self) {
        self.mds.height += 1;
        self.now += 60;
    }
    fn place(&mut self, ids: &[[u8; 32]]) {
        let v2 = is_v2_at(self.mds.height);
        for id in ids {
            self.mds.coins.insert(*id, v2);
        }
    }
    fn commit(&mut self, c: [u8; 32]) {
        let v2 = is_v2_at(self.mds.height);
        self.mds.commitments.insert(c, v2);
        self.mds.commitment_heights.insert(c, self.mds.height);
    }
    /// The relay floor, then consensus.
    fn reveal(&mut self, tx: MdsTx) -> Result<()> {
        let bytes = bincode::serialized_size(&tx)? as u128;
        ensure!((tx.fee() as u128) * 1_024 >= 10 * bytes, "fee {} below the relay floor for {bytes} bytes", tx.fee());
        let mut next = self.mds.clone();
        apply_transaction(&mut next, &tx)?;
        self.mds = next;
        self.reveals.push((self.mds.height, tx));
        Ok(())
    }
}

impl Chains for Sim {
    fn mds_tip(&self) -> u64 {
        self.mds.height
    }
    fn mds_coin(&self, id: &[u8; 32]) -> bool {
        self.mds.coins.contains(id)
    }
    fn mds_spends(&self, coins: &[[u8; 32]], from: u64) -> (Vec<Vec<Vec<u8>>>, u64) {
        let mut out = Vec::new();
        for (h, tx) in &self.reveals {
            if *h < from {
                continue;
            }
            if let MdsTx::Reveal { inputs, witnesses, .. } = tx {
                for (i, input) in inputs.iter().enumerate() {
                    if coins.contains(&input.coin_id()) {
                        let Witness::ScriptInputs(items) = &witnesses[i];
                        out.push(items.clone());
                    }
                }
            }
        }
        (out, self.mds.height)
    }
    fn mds_lock_pending(&self, _: &[u8; 32], _: &[u8; 32]) -> bool {
        false
    }
    fn now(&self) -> u64 {
        self.now
    }
    fn base_now(&self) -> u64 {
        self.base_time()
    }
    fn base_locks(&self, hashlock: &[u8; 32]) -> Vec<BaseLock> {
        self.base
            .order
            .iter()
            .filter_map(|id| {
                let l = &self.base.locks[id];
                (&l.hashlock == hashlock).then(|| BaseLock {
                    swap_id: *id,
                    beneficiary: l.beneficiary,
                    refund_to: l.refund_to,
                    amount: l.amount,
                    timeout: l.timeout,
                    block: 0,
                    asset: l.asset,
                })
            })
            .collect()
    }
    fn base_claim(&self, swap_id: &[u8; 32]) -> Option<[u8; 32]> {
        if self.hide_claims && self.viewer == Some(Role::Buyer) {
            return None;
        }
        self.base.claims.get(swap_id).copied()
    }
    fn base_refunded(&self, swap_id: &[u8; 32]) -> bool {
        self.base.refunded.contains(swap_id)
    }
    fn bid(&self, bid_id: &[u8; 32]) -> Option<BidState> {
        self.base.bids.get(bid_id).cloned()
    }
    fn bid_claimed(&self, bid_id: &[u8; 32]) -> bool {
        self.base.bid_claims.contains(bid_id)
    }
}

// ── Wallets ────────────────────────────────────────────────────────────────

const SELLER: [u8; 20] = [0x5e; 20];
const BUYER: [u8; 20] = [0xb0; 20];
const STRANGER: [u8; 20] = [0x66; 20];

struct Party {
    sw: EthSwap,
    /// Where its MDS claims and refunds pay.
    dest: [u8; 32],
    /// The service's rule: a refund refused as settled sends it looking for
    /// the claim, by the swap's own topic, whatever the scan missed.
    lazy: bool,
}

fn seller_and_buyer(sim: &Sim, mds: u64, wei: u128) -> (Party, Party) {
    seller_and_buyer_in(sim, Asset::Eth, mds, wei)
}

fn seller_and_buyer_in(sim: &Sim, asset: Asset, mds: u64, wei: u128) -> (Party, Party) {
    let take_id = [7; 16];
    let claim_seed = [11; 32];
    let refund_seed = [12; 32];
    let secret = [13; 32];
    let now = sim.now;

    let mut b = EthSwap::new(take_id, Role::Buyer, Phase::Asking, now);
    b.order_id = [1; 16];
    b.mds_amount = mds;
    b.lock_value = mds;
    b.wei = wei;
    b.seller_evm = SELLER;
    b.buyer_evm = BUYER;
    b.key_seed = Some(claim_seed);
    b.claim_pk = pk_of(&claim_seed);
    b.asset = asset;

    let mut s = EthSwap::new(take_id, Role::Seller, Phase::LockingMine, now);
    s.asset = asset;
    s.order_id = [1; 16];
    s.mds_amount = mds;
    s.lock_value = mds;
    s.wei = wei;
    s.seller_evm = SELLER;
    s.buyer_evm = BUYER;
    s.claim_pk = b.claim_pk;
    s.key_seed = Some(refund_seed);
    s.refund_pk = pk_of(&refund_seed);
    s.secret = Some(secret);
    s.secret_hash = hash(&secret);
    s.timeout = seller_timeout(sim.mds.height);
    (Party { sw: s, dest: [0xa1; 32], lazy: false }, Party { sw: b, dest: [0xa2; 32], lazy: false })
}

/// The bus: the seller's answer reaches the buyer.
fn deliver_answer(seller: &EthSwap, buyer: &mut EthSwap) {
    if buyer.phase == Phase::Asking {
        buyer.secret_hash = seller.secret_hash;
        buyer.timeout = seller.timeout;
        buyer.refund_pk = seller.refund_pk;
    }
}

/// Our MDS claim or refund, as the service makes it: a refund's coins fixed
/// first, then committed and revealed in one go.
fn spend(p: &mut Party, sim: &mut Sim, kind: SpendKind) -> Result<()> {
    let dest = *p.sw.dest.get_or_insert(p.dest);
    if kind == SpendKind::Refund && p.sw.refund_mask == 0 {
        let left = coins_left(&p.sw, &*sim);
        ensure!(p.sw.refundable(left), "nothing left to refund");
        p.sw.refund_mask = left;
    }
    let sp = build_spend(&p.sw, kind, &dest)?;
    if p.sw.spend_commitment.is_none() {
        p.sw.spend_commitment = Some(sp.commitment);
        sim.commit(sp.commitment);
    }
    let tx = sign_spend(&p.sw, kind, &sp)?;
    if p.sw.role == Role::Bidder {
        p.sw.reveal_sent = true;
    }
    sim.reveal(tx)
}

/// One wallet's turn: read, decide, act.
fn turn(p: &mut Party, sim: &mut Sim) -> Result<Decision> {
    sim.viewer = Some(p.sw.role);
    let o = observe(&mut p.sw, &*sim);
    let d = decide(&p.sw, &o);
    if let Some(ph) = &d.phase {
        p.sw.phase = ph.clone();
    }
    for a in &d.acts {
        match a {
            Act::LockMds if p.sw.lock_commitment.is_none() => {
                sim.place(&p.sw.lock_coin_ids());
                p.sw.lock_commitment = Some([1; 32]);
                p.sw.scan_from = sim.mds.height.saturating_sub(1).max(1);
            }
            Act::LockMds | Act::Answer | Act::Take | Act::Reserve => {}
            Act::LockEth { delay } => {
                let id = sim.base.lock_in(p.sw.asset, sim.base_time(), p.sw.buyer_evm, p.sw.secret_hash, p.sw.seller_evm, *delay, p.sw.wei)?;
                p.sw.eth_tx = Some([2; 32]);
                p.sw.base_swap = Some(id);
                p.sw.eth_deadline = sim.base.locks[&id].timeout;
            }
            Act::ClaimEth { swap_id } => {
                p.sw.base_swap = Some(*swap_id);
                p.sw.claim_sent = true;
                sim.base.claim(*swap_id, p.sw.secret.unwrap())?;
            }
            Act::ClaimBid => {
                sim.base.claim_bid(sim.base_time(), p.sw.bid_id, p.sw.secret.unwrap())?;
            }
            Act::ClaimMds => spend(p, sim, SpendKind::Claim)?,
            Act::RefundMds => spend(p, sim, SpendKind::Refund)?,
            Act::RefundEth { swap_id } => match sim.base.refund(sim.base_time(), *swap_id) {
                Ok(()) => {}
                Err(e) if e.to_string().contains("settled") && !p.lazy => {
                    // The service's recovery: the claim, by its topic.
                    p.sw.secret = sim.base.claims.get(swap_id).copied();
                }
                Err(e) => bail!("refund: {e}"),
            },
        }
    }
    Ok(d)
}

fn run(seller: &mut Party, buyer: &mut Party, sim: &mut Sim, blocks: u64, mut each: impl FnMut(&mut Party, &mut Party, &mut Sim)) -> Result<()> {
    for _ in 0..blocks {
        turn(seller, sim)?;
        deliver_answer(&seller.sw, &mut buyer.sw);
        turn(buyer, sim)?;
        each(seller, buyer, sim);
        if seller.sw.phase.settled() && buyer.sw.phase.settled() {
            return Ok(());
        }
        sim.mine();
    }
    Ok(())
}

fn mds_at(sim: &Sim, sw: &EthSwap, kind: SpendKind) -> bool {
    let sp = build_spend(sw, kind, sw.dest.as_ref().unwrap()).unwrap();
    sp.output_ids().iter().all(|id| sim.mds.coins.contains(id))
}

// ── Sell orders ────────────────────────────────────────────────────────────

#[test]
fn a_sell_order_completes_with_both_sides_paid() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 3 << 30, 9_000_000 * GWEI);
    run(&mut s, &mut b, &mut sim, 200, |_, _, _| {})?;
    assert_eq!(s.sw.phase, Phase::Done);
    assert_eq!(b.sw.phase, Phase::Done);
    assert!(mds_at(&sim, &b.sw, SpendKind::Claim), "the buyer holds the MDS");
    assert_eq!(sim.base.balance[&SELLER], 9_000_000 * GWEI as i128);
    assert_eq!(sim.base.balance[&BUYER], -(9_000_000 * GWEI as i128));
    Ok(())
}

#[test]
fn a_silent_seller_costs_the_buyer_nothing() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 1 << 30, GWEI);
    // The seller locks and answers, then goes quiet before the buyer pays.
    for _ in 0..8 {
        if s.sw.phase == Phase::LockingMine {
            turn(&mut s, &mut sim)?;
        }
        deliver_answer(&s.sw, &mut b.sw);
        turn(&mut b, &mut sim)?;
        sim.mine();
    }
    assert_eq!(b.sw.phase, Phase::MineLocked);
    for _ in 0..400 {
        turn(&mut b, &mut sim)?;
        if b.sw.phase.settled() {
            break;
        }
        sim.mine();
    }
    assert_eq!(b.sw.phase, Phase::Refunded);
    assert_eq!(sim.base.balance[&BUYER], 0, "the ETH came back");
    // The seller, back, takes its MDS back after the timeout.
    while sim.mds.height < s.sw.timeout {
        sim.mine();
    }
    for _ in 0..3 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    assert_eq!(s.sw.phase, Phase::Refunded);
    assert!(mds_at(&sim, &s.sw, SpendKind::Refund));
    Ok(())
}

#[test]
fn a_buyer_who_never_pays_leaves_the_seller_whole() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer(&sim, 1 << 30, GWEI);
    for _ in 0..400 {
        turn(&mut s, &mut sim)?;
        if s.sw.phase.settled() {
            break;
        }
        sim.mine();
    }
    assert_eq!(s.sw.phase, Phase::Refunded);
    assert!(mds_at(&sim, &s.sw, SpendKind::Refund));
    Ok(())
}

/// The attack the old sell orders were open to: the secret is public on
/// Base the moment the seller claims, and a bot uses it first.
#[test]
fn a_stranger_holding_the_secret_cannot_take_the_mds() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 2 << 30, GWEI);
    run(&mut s, &mut b, &mut sim, 200, |s, _, sim| {
        if s.sw.phase == Phase::Done && sim.reveals.is_empty() {
            // The secret is out. A stranger signs a claim to itself.
            let mut thief = b_clone(&s.sw);
            thief.key_seed = Some([66; 32]);
            let sp = build_spend(&thief, SpendKind::Claim, &[0x66; 32]).unwrap();
            sim.commit(sp.commitment);
            let sig = wots::sig_to_bytes(&wots::sign(&[66; 32], &sp.commitment));
            let items = vec![sig, s.sw.secret.unwrap().to_vec(), vec![1]];
            let tx = MdsTx::Reveal {
                inputs: sp.inputs.clone(),
                witnesses: sp.inputs.iter().map(|_| Witness::ScriptInputs(items.clone())).collect(),
                outputs: sp.outputs.clone(),
                salt: sp.salt,
            };
            assert!(sim.reveal(tx).is_err(), "only the buyer's key opens the lock");
        }
    })?;
    assert_eq!(b.sw.phase, Phase::Done);
    assert!(mds_at(&sim, &b.sw, SpendKind::Claim));
    Ok(())
}

fn b_clone(sw: &EthSwap) -> EthSwap {
    let mut t = sw.clone();
    t.id = [0x66; 16];
    t.role = Role::Buyer;
    t.secret = sw.secret;
    // Same lock coins as the real swap.
    t.id = sw.id;
    t
}

/// The seller knows the secret from the start, and still cannot take the
/// MDS back before its timeout: the buyer's lock on Base is safe to make.
#[test]
fn the_seller_cannot_take_the_lock_back_early() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 1 << 30, GWEI);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        deliver_answer(&s.sw, &mut b.sw);
        turn(&mut b, &mut sim)?;
        sim.mine();
    }
    assert_eq!(b.sw.phase, Phase::MineLocked, "the buyer has paid");
    // A refund before the timeout,
    let r = spend(&mut s, &mut sim, SpendKind::Refund);
    assert!(r.is_err(), "refused before the timeout");
    // and a claim with the seller's own key.
    let mut cheat = s.sw.clone();
    cheat.key_seed = Some([12; 32]);
    let sp = build_spend(&cheat, SpendKind::Claim, &[0x5e; 32])?;
    sim.commit(sp.commitment);
    let sig = wots::sig_to_bytes(&wots::sign(&[12; 32], &sp.commitment));
    let items = vec![sig, s.sw.secret.unwrap().to_vec(), vec![1]];
    let tx = MdsTx::Reveal {
        inputs: sp.inputs.clone(),
        witnesses: sp.inputs.iter().map(|_| Witness::ScriptInputs(items.clone())).collect(),
        outputs: sp.outputs.clone(),
        salt: sp.salt,
    };
    assert!(sim.reveal(tx).is_err(), "the claim branch takes the buyer's signature");
    Ok(())
}

#[test]
fn a_seller_never_claims_an_underpaid_or_foreign_lock() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer(&sim, 1 << 30, 5 * GWEI);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    assert_eq!(s.sw.phase, Phase::MineLocked);
    let h = s.sw.secret_hash;
    sim.base.lock(sim.now, BUYER, h, SELLER, 3_600, 5 * GWEI - 1)?; // short
    sim.base.lock(sim.now, STRANGER, h, SELLER, 3_600, 5 * GWEI)?; // not the buyer
    sim.base.lock(sim.now, BUYER, h, STRANGER, 3_600, 5 * GWEI)?; // pays someone else
    let d = turn(&mut s, &mut sim)?;
    assert!(!d.acts.iter().any(|a| matches!(a, Act::ClaimEth { .. })), "{d:?}");
    assert!(sim.base.claims.is_empty(), "the secret stays secret");
    assert!(d.note.unwrap().contains("3 ETH lock"));
    Ok(())
}

#[test]
fn a_seller_does_not_claim_when_the_buyer_could_not_collect_in_time() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer(&sim, 1 << 30, GWEI);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    // A lock that arrives with under half the margin left on ours.
    while blocks_to_secs(s.sw.timeout - sim.mds.height) >= 1_800 {
        sim.mine();
    }
    sim.base.lock(sim.now, BUYER, s.sw.secret_hash, SELLER, 3_600, GWEI)?;
    let d = turn(&mut s, &mut sim)?;
    assert!(!d.acts.iter().any(|a| matches!(a, Act::ClaimEth { .. })));
    // A lock about to be refundable is not claimed either.
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer(&sim, 1 << 30, GWEI);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    sim.base.lock(sim.now, BUYER, s.sw.secret_hash, SELLER, 600, GWEI)?;
    sim.now += 600 - 100;
    let d = turn(&mut s, &mut sim)?;
    assert!(!d.acts.iter().any(|a| matches!(a, Act::ClaimEth { .. })));
    Ok(())
}

fn blocks_to_secs(b: u64) -> u64 {
    midstate_walletd::swap::blocks_to_secs_pessimistic(b)
}

#[test]
fn the_buyer_pays_only_against_a_confirmed_lock_with_time_left() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 1 << 30, GWEI);
    turn(&mut s, &mut sim)?; // the lock is placed
    deliver_answer(&s.sw, &mut b.sw);
    turn(&mut b, &mut sim)?; // heard the answer
    assert_eq!(b.sw.phase, Phase::WaitingTheirLock);
    let d = turn(&mut b, &mut sim)?;
    assert!(d.acts.is_empty(), "unconfirmed: {d:?}");
    // A lock announced with almost no time left is never paid against.
    let mut late = b.sw.clone();
    late.timeout = sim.mds.height + 60;
    sim.place(&late.lock_coin_ids());
    sim.mine();
    sim.mine();
    let mut p = Party { sw: late, dest: [9; 32], lazy: false };
    let d = turn(&mut p, &mut sim)?;
    assert!(matches!(d.phase, Some(Phase::Failed(_))), "{d:?}");
    assert!(sim.base.locks.is_empty());
    Ok(())
}

/// The second bug: a wallet that missed the seller's claim. Its refund is
/// refused as settled, and it finds the claim by the swap's topic.
#[test]
fn a_buyer_that_missed_the_claim_still_collects() -> Result<()> {
    let mut sim = Sim::new();
    sim.hide_claims = true;
    let (mut s, mut b) = seller_and_buyer(&sim, 1 << 30, GWEI);
    run(&mut s, &mut b, &mut sim, 400, |_, _, _| {})?;
    assert_eq!(s.sw.phase, Phase::Done, "the seller was paid");
    assert_eq!(b.sw.phase, Phase::Done, "and the buyer collected anyway");
    assert!(mds_at(&sim, &b.sw, SpendKind::Claim));
    Ok(())
}

#[test]
fn every_sell_order_spend_clears_the_relay_floor_at_sixteen_coins() -> Result<()> {
    let mut sim = Sim::new();
    let mds = tradeable_amount(u64::MAX >> 20);
    assert_eq!(mds.count_ones(), 16);
    let (mut s, mut b) = seller_and_buyer(&sim, mds, GWEI);
    run(&mut s, &mut b, &mut sim, 200, |_, _, _| {})?;
    assert_eq!(b.sw.phase, Phase::Done);
    Ok(())
}

// ── Buy orders ─────────────────────────────────────────────────────────────

const BIDDER: [u8; 20] = [0xbd; 20];
const FILLER: [u8; 20] = [0xf1; 20];

/// A buy order on Base, the filler's swap reserving it, and the bidder's
/// swap that appears once the fill is announced.
fn bid_and_filler(sim: &mut Sim, mds: u64, wei: u128, bond: u128) -> (Party, [u8; 32], [u8; 32]) {
    bid_and_filler_in(sim, Asset::Eth, mds, wei, bond)
}

fn bid_and_filler_in(sim: &mut Sim, asset: Asset, mds: u64, wei: u128, bond: u128) -> (Party, [u8; 32], [u8; 32]) {
    let secret = [21; 32];
    let addr = [0xad; 32];
    let id = sim.base.create_bid_in(asset, sim.now, BIDDER, hash(&secret), mds, addr, 86_400, bond, wei);
    let mut f = EthSwap::new([5; 16], Role::Filler, Phase::Reserving, sim.now);
    f.asset = asset;
    f.bid_id = id;
    f.mds_amount = mds;
    f.wei = wei;
    f.fill_bond = bond;
    f.seller_evm = FILLER;
    f.receiver_addr = addr;
    f.secret_hash = hash(&secret);
    f.key_seed = Some([22; 32]);
    f.refund_pk = pk_of(&[22; 32]);
    (Party { sw: f, dest: [0xa3; 32], lazy: false }, secret, addr)
}

/// The service's step after the reservation: plan the lock inside it.
fn reserve(f: &mut Party, sim: &mut Sim) -> Result<()> {
    sim.base.reserve_bid(sim.now, f.sw.bid_id, FILLER, FILL_WINDOW_SECS, f.sw.fill_bond)?;
    let b = sim.base.bids[&f.sw.bid_id].clone();
    f.sw.eth_tx = Some([3; 32]);
    f.sw.eth_deadline = b.fill_deadline;
    f.sw.timeout = filler_timeout(sim.mds.height, sim.now, b.fill_deadline).unwrap();
    f.sw.lock_value = fill_total(f.sw.mds_amount, &f.sw)?;
    f.sw.phase = Phase::LockingMine;
    Ok(())
}

fn bidder_for(f: &EthSwap, secret: [u8; 32], now: u64) -> Party {
    let mut b = EthSwap::new([6; 16], Role::Bidder, Phase::WaitingTheirLock, now);
    let a = f.fill();
    b.bid_id = f.bid_id;
    b.mds_amount = a.min_payout;
    b.lock_value = a.total;
    b.wei = f.wei;
    b.receiver_addr = a.receiver_addr;
    b.refund_pk = a.taker_mds_pk;
    b.secret_hash = a.secret_hash;
    b.timeout = a.timeout_height;
    b.secret = Some(secret);
    b.asset = f.asset;
    Party { sw: b, dest: a.receiver_addr, lazy: false }
}

#[test]
fn a_buy_order_is_filled_and_both_sides_paid() -> Result<()> {
    let mut sim = Sim::new();
    let mds = (5 << 30) + 12_345;
    let (mut f, secret, addr) = bid_and_filler(&mut sim, mds, 7 * GWEI, GWEI);
    reserve(&mut f, &mut sim)?;
    let mut b: Option<Party> = None;
    for _ in 0..200 {
        turn(&mut f, &mut sim)?;
        if b.is_none() && f.sw.lock_commitment.is_some() {
            b = Some(bidder_for(&f.sw, secret, sim.now));
        }
        if let Some(b) = b.as_mut() {
            turn(b, &mut sim)?;
        }
        if f.sw.phase.settled() && b.as_ref().is_some_and(|b| b.sw.phase.settled()) {
            break;
        }
        sim.mine();
    }
    let b = b.unwrap();
    assert_eq!(b.sw.phase, Phase::Done);
    assert_eq!(f.sw.phase, Phase::Done);
    // The bidder got at least what it bid for, at its own address.
    let got: u64 = match &sim.reveals.last().unwrap().1 {
        MdsTx::Reveal { outputs, .. } => outputs.iter().filter(|o| o.address() == addr).map(|o| o.value()).sum(),
        _ => 0,
    };
    assert!(got >= mds, "{got} < {mds}");
    assert_eq!(sim.base.balance[&FILLER], (7 * GWEI) as i128, "escrow plus the bond back");
    Ok(())
}

#[test]
fn a_silent_bidder_leaves_the_filler_its_mds() -> Result<()> {
    let mut sim = Sim::new();
    let (mut f, _, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
    reserve(&mut f, &mut sim)?;
    for _ in 0..400 {
        turn(&mut f, &mut sim)?;
        if f.sw.phase.settled() {
            break;
        }
        sim.mine();
    }
    assert_eq!(f.sw.phase, Phase::Refunded);
    assert!(mds_at(&sim, &f.sw, SpendKind::Refund));
    Ok(())
}

#[test]
fn a_bidder_does_not_claim_without_a_reservation_to_collect_with() -> Result<()> {
    let mut sim = Sim::new();
    let (mut f, secret, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
    // A fill locked without reserving: the bidder must not reveal, or a
    // stranger could reserve and collect with the public secret.
    let b0 = sim.base.bids[&f.sw.bid_id].clone();
    f.sw.eth_deadline = sim.now + FILL_WINDOW_SECS;
    f.sw.timeout = filler_timeout(sim.mds.height, sim.now, f.sw.eth_deadline).unwrap();
    f.sw.lock_value = fill_total(f.sw.mds_amount, &f.sw)?;
    sim.place(&f.sw.lock_coin_ids());
    f.sw.lock_seen = Some(sim.mds.height);
    assert_eq!(b0.filler, [0; 20]);
    let mut b = bidder_for(&f.sw, secret, sim.now);
    for _ in 0..5 {
        sim.mine();
        let d = turn(&mut b, &mut sim)?;
        assert!(d.acts.is_empty(), "{d:?}");
    }
    assert!(sim.reveals.is_empty());
    Ok(())
}

#[test]
fn a_last_moment_bidder_claim_during_the_refund_still_pays_the_filler() -> Result<()> {
    let mut sim = Sim::new();
    let (mut f, secret, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
    reserve(&mut f, &mut sim)?;
    while f.sw.phase != Phase::MineLocked {
        turn(&mut f, &mut sim)?;
        sim.mine();
    }
    while sim.mds.height < f.sw.timeout {
        sim.mine();
    }
    // The filler has committed its refund, not yet revealed, when the
    // bidder's claim lands.
    f.sw.phase = Phase::Refunding;
    f.sw.dest = Some(f.dest);
    f.sw.refund_mask = coins_left(&f.sw, &sim);
    f.sw.spend_commitment = Some(build_spend(&f.sw, SpendKind::Refund, &f.dest)?.commitment);
    let mut b = bidder_for(&f.sw, secret, sim.now);
    b.sw.phase = Phase::Claiming;
    spend(&mut b, &mut sim, SpendKind::Claim)?;
    for _ in 0..5 {
        turn(&mut f, &mut sim)?;
        sim.mine();
    }
    assert_eq!(f.sw.phase, Phase::Done, "the filler collected the escrow");
    assert_eq!(sim.base.balance[&FILLER], (7 * GWEI) as i128);
    Ok(())
}

/// Runs the filler alone until its lock is confirmed.
fn filler_locked(f: &mut Party, sim: &mut Sim) -> Result<()> {
    reserve(f, sim)?;
    while f.sw.phase != Phase::MineLocked {
        turn(f, sim)?;
        sim.mine();
    }
    Ok(())
}

fn filler_until_settled(f: &mut Party, sim: &mut Sim, blocks: u64) -> Result<()> {
    for _ in 0..blocks {
        turn(f, sim)?;
        if f.sw.phase.settled() {
            break;
        }
        sim.mine();
    }
    Ok(())
}

#[test]
fn a_claim_with_a_witness_of_its_own_shape_still_pays_the_filler() -> Result<()> {
    // The script takes any true selector and items beneath the secret; our
    // wallets write `[secret, 1]`, a bidder need not.
    for shape in [&[2u8][..], &[0, 1][..]] {
        let mut sim = Sim::new();
        let (mut f, secret, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
        filler_locked(&mut f, &mut sim)?;
        let b = bidder_for(&f.sw, secret, sim.now);
        let sp = build_spend(&b.sw, SpendKind::Claim, &b.dest)?;
        sim.commit(sp.commitment);
        let items = vec![secret.to_vec(), shape.to_vec()];
        sim.reveal(MdsTx::Reveal {
            inputs: sp.inputs.clone(),
            witnesses: sp.inputs.iter().map(|_| Witness::ScriptInputs(items.clone())).collect(),
            outputs: sp.outputs.clone(),
            salt: sp.salt,
        })?;
        filler_until_settled(&mut f, &mut sim, 10)?;
        assert_eq!(f.sw.phase, Phase::Done, "selector {shape:?}");
        assert_eq!(sim.base.balance[&FILLER], (7 * GWEI) as i128);
    }
    Ok(())
}

#[test]
fn a_claim_of_one_coin_pays_the_filler_and_the_rest_comes_back() -> Result<()> {
    let mut sim = Sim::new();
    let mds = (3 << 30) + 12_345;
    let (mut f, secret, addr) = bid_and_filler(&mut sim, mds, 7 * GWEI, 0);
    filler_locked(&mut f, &mut sim)?;
    let coins = f.sw.lock_coins();
    assert!(coins.len() >= 3, "{coins:?}");
    // The last coin only, its payout made up from a coin of the bidder's
    // own: the covenant counts the whole transaction's outputs.
    let k = coins.len() - 1;
    let mut own = Vec::new();
    push_int(&mut own, 1);
    let (own_value, own_salt) = (1u64 << 33, [0x4d; 32]);
    sim.place(&[compute_coin_id(&hash(&own), own_value, &own_salt)]);
    let inputs = vec![
        InputReveal { predicate: Predicate::Script { bytecode: f.sw.lock_script() }, value: coins[k].0, salt: coins[k].1, commitment: None },
        InputReveal { predicate: Predicate::Script { bytecode: own }, value: own_value, salt: own_salt, commitment: None },
    ];
    let outputs: Vec<OutputData> = decompose_value(coins[k].0 + own_value - 1_000)
        .into_iter()
        .enumerate()
        .map(|(i, value)| OutputData::Standard { address: addr, value, salt: hash(&[i as u8; 32]) })
        .collect();
    let salt = [0x5a; 32];
    let ins: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
    let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
    sim.commit(compute_commitment(&ins, &outs, &salt));
    sim.reveal(MdsTx::Reveal {
        inputs,
        witnesses: vec![Witness::ScriptInputs(vec![secret.to_vec(), vec![1]]), Witness::ScriptInputs(vec![])],
        outputs,
        salt,
    })?;
    filler_until_settled(&mut f, &mut sim, 400)?;
    assert_eq!(f.sw.phase, Phase::Done);
    assert_eq!(sim.base.balance[&FILLER], (7 * GWEI) as i128, "the escrow, collected with the secret off the last coin");
    assert_eq!(f.sw.refund_mask.count_ones() as usize, coins.len() - 1);
    assert!(mds_at(&sim, &f.sw, SpendKind::Refund), "the coins the bidder left came back");
    Ok(())
}

#[test]
fn a_fill_claimed_before_the_filler_ever_saw_its_lock_still_pays_it() -> Result<()> {
    let mut sim = Sim::new();
    let (mut f, secret, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
    reserve(&mut f, &mut sim)?;
    turn(&mut f, &mut sim)?; // commits the lock, which lands at once
    assert!(f.sw.lock_commitment.is_some() && f.sw.lock_seen.is_none());
    // The wallet is away while the lock is confirmed and claimed, and back
    // only after the lock's end.
    let mut b = bidder_for(&f.sw, secret, sim.now);
    b.sw.phase = Phase::Claiming;
    spend(&mut b, &mut sim, SpendKind::Claim)?;
    while sim.mds.height <= f.sw.timeout {
        sim.mine();
    }
    filler_until_settled(&mut f, &mut sim, 10)?;
    assert_eq!(f.sw.phase, Phase::Done);
    assert_eq!(sim.base.balance[&FILLER], (7 * GWEI) as i128);
    Ok(())
}

#[test]
fn a_bidder_never_publishes_its_secret_too_late_for_its_claim_to_land() -> Result<()> {
    let mut sim = Sim::new();
    let (mut f, secret, _) = bid_and_filler(&mut sim, 3 << 30, 7 * GWEI, 0);
    filler_locked(&mut f, &mut sim)?;
    let mut b = bidder_for(&f.sw, secret, sim.now);
    // Its commitment went out in time and was mined late.
    b.sw.phase = Phase::Claiming;
    b.sw.dest = Some(b.dest);
    let sp = build_spend(&b.sw, SpendKind::Claim, &b.dest)?;
    b.sw.spend_commitment = Some(sp.commitment);
    sim.commit(sp.commitment);
    while sim.mds.height + REVEAL_MIN_BLOCKS <= b.sw.timeout {
        sim.mine();
    }
    let d = turn(&mut b, &mut sim)?;
    assert!(matches!(b.sw.phase, Phase::Failed(_)), "{d:?}");
    assert!(sim.reveals.is_empty() && !b.sw.reveal_sent, "the secret was never published");
    Ok(())
}

#[test]
fn a_seller_retries_its_claim_only_while_base_time_leaves_the_margin() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer(&sim, 1 << 30, 5 * GWEI);
    while sim.base.locks.is_empty() {
        turn(&mut s, &mut sim)?;
        deliver_answer(&s.sw, &mut b.sw);
        turn(&mut b, &mut sim)?;
        sim.mine();
    }
    let id = sim.base.order[0];
    // Our first claim went out, and reached no one.
    s.sw.phase = Phase::Claiming;
    s.sw.claim_sent = true;
    s.sw.base_swap = Some(id);
    let left = sim.base.locks[&id].timeout - sim.now;
    // By our clock there is time; by Base's, one second too little.
    sim.base_ahead = left - (CLAIM_MARGIN_SECS + CLOCK_SKEW_SECS) + 1;
    let d = turn(&mut s, &mut sim)?;
    assert!(d.acts.is_empty() && sim.base.claims.is_empty(), "{d:?}");
    sim.base_ahead -= 1;
    let d = turn(&mut s, &mut sim)?;
    assert_eq!(d.acts, vec![Act::ClaimEth { swap_id: id }]);
    Ok(())
}

// ── USDC ───────────────────────────────────────────────────────────────────

/// Five dollars, in USDC's smallest unit.
const FIVE_DOLLARS: u128 = 5_000_000;

#[test]
fn a_usdc_sell_order_completes_with_both_sides_paid_in_usdc() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, mut b) = seller_and_buyer_in(&sim, Asset::Usdc, 3 << 30, FIVE_DOLLARS);
    run(&mut s, &mut b, &mut sim, 200, |_, _, _| {})?;
    assert_eq!((s.sw.phase.clone(), b.sw.phase.clone()), (Phase::Done, Phase::Done));
    assert!(mds_at(&sim, &b.sw, SpendKind::Claim), "the buyer holds the MDS");
    assert_eq!(sim.base.usdc[&SELLER], FIVE_DOLLARS as i128);
    assert_eq!(sim.base.usdc[&BUYER], -(FIVE_DOLLARS as i128));
    assert!(sim.base.balance.values().all(|v| *v == 0), "no ETH moved");
    assert!(sim.base.locks.values().all(|l| l.asset == Asset::Usdc), "the lock went to the USDC contract");
    Ok(())
}

/// A USDC sale's numbers, paid on the ETH contract: five million wei is a
/// few millionths of a cent, not five dollars. The seller never claims it,
/// so the secret never leaves, and takes its MDS back in the end.
#[test]
fn a_usdc_seller_never_claims_an_eth_lock_of_the_same_numbers() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer_in(&sim, Asset::Usdc, 1 << 30, FIVE_DOLLARS);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    assert_eq!(s.sw.phase, Phase::MineLocked);
    let h = s.sw.secret_hash;
    sim.base.lock_in(Asset::Eth, sim.now, BUYER, h, SELLER, 3_600, FIVE_DOLLARS)?;
    sim.base.lock_in(Asset::Eth, sim.now, BUYER, h, SELLER, 3_600, 1_000 * GWEI * GWEI)?; // a thousand ETH
    let d = turn(&mut s, &mut sim)?;
    assert!(!d.acts.iter().any(|a| matches!(a, Act::ClaimEth { .. })), "{d:?}");
    assert!(d.note.is_none(), "another contract's locks are not this sale's to mention: {d:?}");
    assert!(sim.base.claims.is_empty(), "the secret stays secret");
    for _ in 0..400 {
        turn(&mut s, &mut sim)?;
        if s.sw.phase.settled() {
            break;
        }
        sim.mine();
    }
    assert_eq!(s.sw.phase, Phase::Refunded);
    assert!(mds_at(&sim, &s.sw, SpendKind::Refund));
    assert!(sim.base.claims.is_empty());
    Ok(())
}

/// And the other way: an ETH sale is not paid by USDC, however much.
#[test]
fn an_eth_seller_never_claims_a_usdc_lock() -> Result<()> {
    let mut sim = Sim::new();
    let (mut s, _) = seller_and_buyer(&sim, 1 << 30, 5 * GWEI);
    for _ in 0..4 {
        turn(&mut s, &mut sim)?;
        sim.mine();
    }
    sim.base.lock_in(Asset::Usdc, sim.now, BUYER, s.sw.secret_hash, SELLER, 3_600, 1_000_000 * FIVE_DOLLARS)?;
    let d = turn(&mut s, &mut sim)?;
    assert!(!d.acts.iter().any(|a| matches!(a, Act::ClaimEth { .. })), "{d:?}");
    // Paid properly, on its own contract, it goes through.
    sim.base.lock(sim.now, BUYER, s.sw.secret_hash, SELLER, 3_600, 5 * GWEI)?;
    let d = turn(&mut s, &mut sim)?;
    let id = match d.acts.as_slice() {
        [Act::ClaimEth { swap_id }] => *swap_id,
        other => panic!("{other:?}"),
    };
    assert_eq!(sim.base.locks[&id].asset, Asset::Eth);
    Ok(())
}

#[test]
fn a_usdc_buy_order_is_filled_with_its_bond_paid_back_in_usdc() -> Result<()> {
    let mut sim = Sim::new();
    let mds = (2 << 30) + 777;
    let bond = FIVE_DOLLARS / 10;
    let (mut f, secret, addr) = bid_and_filler_in(&mut sim, Asset::Usdc, mds, FIVE_DOLLARS, bond);
    assert!(bond_acceptable(bond, FIVE_DOLLARS), "a tenth, weighed like for like");
    reserve(&mut f, &mut sim)?;
    assert_eq!(sim.base.usdc[&FILLER], -(bond as i128), "the bond is staked in USDC");
    let mut b: Option<Party> = None;
    for _ in 0..200 {
        turn(&mut f, &mut sim)?;
        if b.is_none() && f.sw.lock_commitment.is_some() {
            b = Some(bidder_for(&f.sw, secret, sim.now));
        }
        if let Some(b) = b.as_mut() {
            turn(b, &mut sim)?;
        }
        if f.sw.phase.settled() && b.as_ref().is_some_and(|b| b.sw.phase.settled()) {
            break;
        }
        sim.mine();
    }
    let b = b.unwrap();
    assert_eq!((b.sw.phase.clone(), f.sw.phase.clone()), (Phase::Done, Phase::Done));
    assert_eq!(b.sw.asset, Asset::Usdc);
    let got: u64 = match &sim.reveals.last().unwrap().1 {
        MdsTx::Reveal { outputs, .. } => outputs.iter().filter(|o| o.address() == addr).map(|o| o.value()).sum(),
        _ => 0,
    };
    assert!(got >= mds, "{got} < {mds}");
    assert_eq!(sim.base.usdc[&FILLER], FIVE_DOLLARS as i128, "escrow, and the bond back, in USDC");
    assert_eq!(sim.base.usdc[&BIDDER], -(FIVE_DOLLARS as i128));
    assert!(sim.base.balance.values().all(|v| *v == 0), "no ETH moved");
    Ok(())
}
