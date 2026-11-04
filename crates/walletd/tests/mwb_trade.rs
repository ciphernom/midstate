//! MDS <-> MDW trades end to end: two wallets negotiate over the wire
//! format (sealed, fragmented, reassembled), then each runs the swap
//! machine (`observe` → `decide` → act) against both chains' consensus code
//! until it settles.
//!
//! Midstate spends go through the mempool's fee floor and
//! `apply_transaction`; lock coins are placed directly (the send that funds
//! a lock is the wallet's ordinary send). Midwimble transactions pass the
//! checks its mempool makes (`validate`, the fee floor, the lock height,
//! `apply_body`) before a simulated block applies them.

use anyhow::{bail, ensure, Result};
use midstate::chat::ChatMessage;
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::transaction::apply_transaction;
use midstate::core::types::{is_v2_at, State as MdsState, Transaction as MdsTx, Witness};
use midstate_walletd::mwb_account::{BlockView, MwbAccount};
use midstate_walletd::mwb_negotiate::{Ctx, Negotiation, Step};
use midstate_walletd::mwb_trade::{
    build_mds_spend, decide, observe, Act, Chains, Leg, MdsSpendKind, MwbOrder, MwbSwap, OrderStatus, Phase,
    SELL_MDS_FROM,
};
use midstate_walletd::mwb_wire::{maker_session, parse, Frame, KeyPair, Msg, Outgoing, Reassembler, Route, Side};
use midwimble::core::mw::builder::build_coinbase;
use midwimble::core::mw::crypto::Point32;
use midwimble::core::mw::stealth::{scan_output, WalletKeys};
use midwimble::core::mw::transaction::{Context, Kernel, Transaction as MwbTx};
use midwimble::core::state::apply_body;
use midwimble::core::types::{State as MwbState, COINBASE_MATURITY, MIN_FEE_PER_WEIGHT};

const MDS_AMOUNT: u64 = 1_500_000; // seven coins
const MDW_AMOUNT: u64 = 25_000_000;

// ── The chains ──────────────────────────────────────────────────────────────

struct Sim {
    mds: MdsState,
    /// Midstate reveals by height, witnesses intact.
    mds_reveals: Vec<(u64, MdsTx)>,
    mwb: MwbState,
    mwb_blocks: Vec<(u64, MwbTx)>,
    mwb_mempool: Vec<MwbTx>,
    /// Midwimble blocks over their signer's cap, by height: valid, but next
    /// to nothing in fork choice. `over_cap` mines the next ones so.
    mwb_over_cap: std::collections::BTreeSet<u64>,
    over_cap: bool,
}

impl Sim {
    fn new() -> Self {
        let mut mwb = MwbState::genesis();
        mwb.height = 1_000;
        Self {
            mds: MdsState {
                midstate: [0; 32],
                coins: UtxoAccumulator::new(),
                commitments: UtxoAccumulator::new(),
                depth: 0,
                target: [0xff; 32],
                // OP_MERKLE_ROOT is live from V6_ACTIVATION_HEIGHT.
                height: 450_000,
                timestamp: 0,
                commitment_heights: Default::default(),
                expirations: Default::default(),
                chain_mmr: MerkleMountainRange::new(),
                header_hash: [0; 32],
                burned_wots: UtxoAccumulator::new(),
            },
            mds_reveals: Vec::new(),
            mwb,
            mwb_blocks: Vec::new(),
            mwb_mempool: Vec::new(),
            mwb_over_cap: Default::default(),
            over_cap: false,
        }
    }

    /// A coinbase to `keys`, matured, and the account that finds it.
    fn mint(&mut self, keys: &WalletKeys, value: u64) -> MwbAccount {
        let cb = build_coinbase(&[(keys.address(), value)]).unwrap();
        let h = self.mwb.height;
        self.mwb = apply_body(&self.mwb, &MwbTx::empty(), Some(&cb), 0).unwrap();
        let mut acct = MwbAccount::starting_at(h);
        let block = BlockView {
            height: h,
            id: [1; 32],
            prev: [0; 32],
            timestamp: 0,
            outputs: cb.outputs.outputs.iter().map(|o| (o, true)).collect(),
            spent: Default::default(),
            kernels: Default::default(),
        };
        acct.scan(keys, &block).unwrap();
        self.mwb.height += COINBASE_MATURITY + 1;
        acct
    }

    fn mds_commit(&mut self, commitment: [u8; 32]) {
        let v2 = is_v2_at(self.mds.height);
        self.mds.commitments.insert(commitment, v2);
        self.mds.commitment_heights.insert(commitment, self.mds.height);
    }

    /// The mempool's fee floor, then consensus.
    fn mds_reveal(&mut self, tx: MdsTx) -> Result<()> {
        let bytes = bincode::serialized_size(&tx)? as u128;
        ensure!((tx.fee() as u128) * 1_024 >= 10 * bytes, "fee {} below the relay floor for {bytes} bytes", tx.fee());
        let mut next = self.mds.clone();
        apply_transaction(&mut next, &tx)?;
        self.mds = next;
        self.mds_reveals.push((self.mds.height, tx));
        Ok(())
    }

    /// What midwimble's mempool checks before admitting a transaction.
    fn mwb_submit(&mut self, tx: MwbTx) -> Result<()> {
        tx.validate(Context::Relay)?;
        ensure!(tx.fee()? >= tx.weight() * MIN_FEE_PER_WEIGHT, "fee below the relay minimum");
        ensure!(tx.lock_height() <= self.mwb.height, "time-locked until {}", tx.lock_height());
        apply_body(&self.mwb, &tx, None, 0)?;
        if !self.mwb_mempool.iter().any(|t| t.hash() == tx.hash()) {
            self.mwb_mempool.push(tx);
        }
        Ok(())
    }

    fn mine(&mut self) {
        for tx in std::mem::take(&mut self.mwb_mempool) {
            // A transaction a block before it made invalid drops out, as
            // it would from a real mempool.
            if let Ok(next) = apply_body(&self.mwb, &tx, None, 0) {
                self.mwb = next;
                self.mwb_blocks.push((self.mwb.height, tx));
            }
        }
        if self.over_cap {
            self.mwb_over_cap.insert(self.mwb.height);
        }
        self.mwb.height += 1;
        self.mds.height += 1;
    }

    fn mds_has(&self, id: &[u8; 32]) -> bool {
        self.mds.coins.contains(id)
    }
}

impl Chains for Sim {
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
        for (h, tx) in &self.mds_reveals {
            if *h < from {
                continue;
            }
            if let MdsTx::Reveal { inputs, witnesses, .. } = tx {
                if let Some(i) = inputs.iter().position(|i| &i.coin_id() == coin) {
                    let Witness::ScriptInputs(items) = &witnesses[i];
                    return (Some(items.clone()), self.mds.height);
                }
            }
        }
        (None, self.mds.height)
    }
    fn mwb_output(&self, c: &Point32) -> Option<(u64, Point32)> {
        self.mwb.utxos.get(c).map(|e| (e.height, e.owner_key))
    }
    fn mwb_full_weight(&self, height: u64) -> u64 {
        (height..self.mwb.height).filter(|h| !self.mwb_over_cap.contains(h)).count() as u64
    }
    fn mwb_kernel(&self, excess: &Point32, owner_excess: &Point32, from: u64) -> (Option<Kernel>, u64) {
        let in_blocks = self.mwb_blocks.iter().filter(|(h, _)| *h >= from).map(|(_, t)| t);
        for tx in self.mwb_mempool.iter().chain(in_blocks) {
            if let Some(k) = tx.body.kernels.iter().find(|k| &k.excess == excess && &k.owner_excess == owner_excess) {
                return (Some(k.clone()), self.mwb.height);
            }
        }
        (None, self.mwb.height)
    }
}

// ── The bus ─────────────────────────────────────────────────────────────────

fn chat(o: &Outgoing) -> ChatMessage {
    ChatMessage { sender: "peer".into(), timestamp: 0, nonce: 0, reply_to: None, words: o.0.clone(), attachments: o.1.clone() }
}

/// Carries chat messages to one wallet: each whole message, with its route.
fn deliver(r: &mut Reassembler, out: &[Outgoing]) -> Vec<(Route, Vec<u8>)> {
    let mut whole = Vec::new();
    for o in out {
        let (route, index, count, chunk) = match parse(&chat(o)).expect("a frame of ours") {
            Frame::Take { order_id, taker_key, index, count, chunk } => (Route::Take { order_id, taker_key }, index, count, chunk),
            Frame::Session { tag, to, seq, index, count, chunk } => (Route::Session { tag, to, seq }, index, count, chunk),
            Frame::Advert(_) => continue,
        };
        if let Some(sealed) = r.add(route.clone(), index, count, chunk, 0) {
            whole.push((route, sealed));
        }
    }
    whole
}

struct Wallet {
    keys: WalletKeys,
    account: MwbAccount,
    neg: Option<Negotiation>,
    swap: Option<MwbSwap>,
    inbox: Reassembler,
}

impl Wallet {
    fn new(keys: WalletKeys, account: MwbAccount) -> Self {
        Self { keys, account, neg: None, swap: None, inbox: Reassembler::default() }
    }
    fn ctx<'a>(keys: &'a WalletKeys, account: &'a mut MwbAccount, sim: &Sim) -> Ctx<'a> {
        Ctx { mds_tip: sim.mds.height, mwb_tip: sim.mwb.height, keys, account, now: 0 }
    }
}

/// Runs a negotiation to completion; returns both journal entries.
fn negotiate(side: Side, sim: &Sim, maker: &mut Wallet, taker: &mut Wallet) -> Result<()> {
    let order_key = KeyPair::random();
    let order = MwbOrder {
        id: [9; 16],
        side,
        mds_amount: MDS_AMOUNT,
        mdw_amount: MDW_AMOUNT,
        key: order_key.to_bytes(),
        created: sim.mds.height,
        expires: sim.mds.height + 1_000,
        status: OrderStatus::Open,
        last_advert: 0,
    };
    let advert = order.advert()?;

    let taker_key = KeyPair::random();
    let (n, out) = Negotiation::take(&advert, &taker_key, &mut Wallet::ctx(&taker.keys, &mut taker.account, sim))?;
    taker.neg = Some(n);

    // The maker finds the take and answers it.
    let (route, sealed) = deliver(&mut maker.inbox, &out).pop().expect("the take arrives");
    let Route::Take { order_id, taker_key: tk } = route.clone() else { bail!("not a take") };
    ensure!(order_id == order.id);
    let session = maker_session(&order_key, &order_id, &tk)?;
    let Msg::Take(take) = route.open(&session, &sealed)? else { bail!("not a take") };
    let (n, mut to_taker) = Negotiation::answer(&order, take, session, &mut Wallet::ctx(&maker.keys, &mut maker.account, sim))?;
    maker.neg = Some(n);

    let mut to_maker: Vec<Outgoing> = Vec::new();
    for _ in 0..8 {
        for (w, out, inbox) in [(&mut *taker, std::mem::take(&mut to_taker), 0), (&mut *maker, std::mem::take(&mut to_maker), 1)] {
            for (route, sealed) in deliver(&mut w.inbox, &out) {
                let Route::Session { seq, .. } = route else { bail!("a take mid-session") };
                let Some(neg) = w.neg.as_mut() else { continue };
                let msg = route.open(&neg.swap.session, &sealed)?;
                let step = neg.receive(seq, msg, &mut Wallet::ctx(&w.keys, &mut w.account, sim))?;
                let (reply, done) = match step {
                    Step::Reply(r) => (r, false),
                    Step::Complete(r) => (r, true),
                    Step::Resend | Step::Ignore => (vec![], false),
                };
                if inbox == 0 { to_maker.extend(reply) } else { to_taker.extend(reply) }
                if done {
                    w.swap = Some(w.neg.take().unwrap().swap);
                }
            }
        }
        if maker.swap.is_some() && taker.swap.is_some() {
            return Ok(());
        }
    }
    bail!("the negotiation did not finish")
}

// ── Running a swap ──────────────────────────────────────────────────────────

/// One wallet's turn: read the chains, decide, act.
fn turn(w: &mut Wallet, sim: &mut Sim, lazy: bool) -> Result<()> {
    let sw = w.swap.as_mut().unwrap();
    let o = observe(sw, &*sim, false);
    let d = decide(sw, &o);
    sw.phase = d.phase.clone();
    if lazy && matches!(d.act, Act::LockMine | Act::Claim) {
        return Ok(());
    }
    match d.act {
        Act::None => {}
        Act::LockMine => match sw.my_leg() {
            Leg::Mds if sw.mds_lock_commitment.is_none() => {
                let v2 = is_v2_at(sim.mds.height);
                for id in sw.mds_lock_coin_ids() {
                    sim.mds.coins.insert(id, v2);
                }
                sw.mds_lock_commitment = Some([1; 32]);
            }
            Leg::Mwb if sw.mwb_broadcast.is_none() => {
                sim.mwb_submit(sw.funding()?)?;
                sw.mwb_broadcast = Some(sim.mwb.height);
            }
            _ => {}
        },
        Act::Claim | Act::Refund => {
            let claim = d.act == Act::Claim;
            let leg = if claim { sw.their_leg() } else { sw.my_leg() };
            match leg {
                Leg::Mwb => {
                    let tx = if claim { sw.claim()? } else { sw.refund()? };
                    sim.mwb_submit(tx)?;
                    sw.reveal_sent |= claim;
                }
                Leg::Mds => {
                    let kind = if claim { MdsSpendKind::Claim } else { MdsSpendKind::Refund };
                    let dest = *sw.mds_dest.get_or_insert_with(rand::random);
                    let sp = build_mds_spend(sw, kind, &dest)?;
                    if sw.mds_spend_commitment.is_none() {
                        sim.mds_commit(sp.commitment);
                        sw.mds_spend_commitment = Some(sp.commitment);
                    } else {
                        sim.mds_reveal(midstate_walletd::mwb_trade::sign_mds_spend(sw, kind, &sp)?)?;
                        sw.reveal_sent |= claim;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Runs both wallets until both settle. `lazy` makes that side never lock
/// or claim (a counterparty that walks away).
fn run(sim: &mut Sim, maker: &mut Wallet, taker: &mut Wallet, maker_lazy: bool, taker_lazy: bool) -> Result<()> {
    for _ in 0..600 {
        turn(maker, sim, maker_lazy)?;
        turn(taker, sim, taker_lazy)?;
        if maker.swap.as_ref().unwrap().phase.settled() && taker.swap.as_ref().unwrap().phase.settled() {
            return Ok(());
        }
        sim.mine();
    }
    bail!(
        "not settled: maker {:?}, taker {:?}",
        maker.swap.as_ref().unwrap().phase,
        taker.swap.as_ref().unwrap().phase
    )
}

fn mwb_received(keys: &WalletKeys, sim: &Sim) -> u64 {
    sim.mwb_blocks
        .iter()
        .flat_map(|(_, tx)| tx.outputs().filter_map(|o| scan_output(keys, o)).map(|o| o.value).collect::<Vec<_>>())
        .sum()
}

fn mds_paid(sw: &MwbSwap, kind: MdsSpendKind, sim: &Sim) -> u64 {
    let sp = build_mds_spend(sw, kind, sw.mds_dest.as_ref().unwrap()).unwrap();
    assert!(sp.output_ids().iter().all(|id| sim.mds_has(id)));
    sp.outputs.iter().map(|o| o.value()).sum()
}

fn wallets(side: Side, sim: &mut Sim) -> (Wallet, Wallet) {
    let (mk, tk) = (WalletKeys::random(), WalletKeys::random());
    // The side that funds midwimble needs coins there.
    let (ma, ta) = match side {
        Side::SellMds => (MwbAccount::default(), sim.mint(&tk, MDW_AMOUNT + 1_000_000)),
        Side::SellMdw => (sim.mint(&mk, MDW_AMOUNT + 1_000_000), MwbAccount::default()),
    };
    (Wallet::new(mk, ma), Wallet::new(tk, ta))
}

const MWB_FEE: u64 = 300_000; // well above min_fee(1, 1, 1)

#[test]
fn selling_mds_settles_both_ways_round() -> Result<()> {
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMds, &mut sim);
    negotiate(Side::SellMds, &sim, &mut maker, &mut taker)?;
    assert_eq!(maker.swap.as_ref().unwrap().phase, Phase::LockingMine);
    assert_eq!(taker.swap.as_ref().unwrap().phase, Phase::WaitingTheirLock);
    assert!(taker.account.balance(sim.mwb.height).reserved >= MDW_AMOUNT, "the funding coins are set aside");

    run(&mut sim, &mut maker, &mut taker, false, false)?;
    let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
    assert_eq!((m.phase.clone(), t.phase.clone()), (Phase::Done, Phase::Done));
    // The maker took the MDW, which handed the taker the secret...
    let got = mwb_received(&maker.keys, &sim);
    assert!(got > MDW_AMOUNT - MWB_FEE && got < MDW_AMOUNT, "maker received {got}");
    assert_eq!(t.t, m.t, "the taker read the secret off the maker's kernel");
    // ...with which the taker took the MDS.
    let paid = mds_paid(t, MdsSpendKind::Claim, &sim);
    assert!(paid > MDS_AMOUNT - 1_000 && paid < MDS_AMOUNT, "taker received {paid}");
    Ok(())
}

#[test]
fn selling_mdw_settles_both_ways_round() -> Result<()> {
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMdw, &mut sim);
    negotiate(Side::SellMdw, &sim, &mut maker, &mut taker)?;
    assert_eq!(maker.swap.as_ref().unwrap().phase, Phase::LockingMine);
    assert_eq!(taker.swap.as_ref().unwrap().phase, Phase::WaitingTheirLock);

    run(&mut sim, &mut maker, &mut taker, false, false)?;
    let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
    assert_eq!((m.phase.clone(), t.phase.clone()), (Phase::Done, Phase::Done));
    // The maker took the MDS, publishing every unopened half...
    let paid = mds_paid(m, MdsSpendKind::Claim, &sim);
    assert!(paid > MDS_AMOUNT - 2_000 && paid < MDS_AMOUNT, "maker received {paid}");
    // ...from which the taker recovered the secret and took the MDW.
    assert_eq!(t.t, m.t);
    let got = mwb_received(&taker.keys, &sim);
    assert!(got > MDW_AMOUNT - MWB_FEE && got < MDW_AMOUNT, "taker received {got}");
    Ok(())
}

#[test]
fn a_taker_who_never_locks_costs_the_maker_only_time() -> Result<()> {
    for side in [Side::SellMds, Side::SellMdw] {
        let mut sim = Sim::new();
        let (mut maker, mut taker) = wallets(side, &mut sim);
        negotiate(side, &sim, &mut maker, &mut taker)?;
        run(&mut sim, &mut maker, &mut taker, false, true)?;
        let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
        assert_eq!(m.phase, Phase::Refunded, "{side:?}");
        assert!(matches!(t.phase, Phase::Failed(_)), "{side:?}: {:?}", t.phase);
        // The maker's refund waited for its height, on its own chain.
        let (tip, height) = match side {
            Side::SellMds => (sim.mds.height, m.follow_height),
            Side::SellMdw => (sim.mwb.height, m.follow_height),
        };
        assert!(tip >= height);
        match side {
            Side::SellMds => assert!(mds_paid(m, MdsSpendKind::Refund, &sim) > MDS_AMOUNT - 1_000),
            Side::SellMdw => assert!(mwb_received(&maker.keys, &sim) > MDW_AMOUNT - MWB_FEE),
        }
    }
    Ok(())
}

/// A midwimble lock buried under nothing but over-cap blocks is not
/// confirmed: they add next to nothing to fork choice, and one full-weight
/// block on the block below replaces them, lock and all. Selling MDS, the
/// maker reveals the secret by claiming the taker's MDW lock, so it waits
/// for blocks that counted in full, however many over-cap ones pile up.
#[test]
fn a_midwimble_lock_under_over_cap_blocks_alone_is_not_confirmed() -> Result<()> {
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMds, &mut sim);
    negotiate(Side::SellMds, &sim, &mut maker, &mut taker)?;
    sim.over_cap = true;
    for _ in 0..12 {
        turn(&mut maker, &mut sim, false)?;
        turn(&mut taker, &mut sim, false)?;
        sim.mine();
    }
    let lock = taker.swap.as_ref().unwrap().view.commitment;
    let (height, _) = sim.mwb_output(&lock).expect("the taker's lock is in a block");
    assert!(sim.mwb.height >= height + 5, "buried under over-cap blocks");
    let m = maker.swap.as_ref().unwrap();
    assert_eq!(m.phase, Phase::MineLocked, "the maker waits");
    assert!(!m.reveal_sent, "and has not revealed the secret");

    // Blocks that count in full: the maker claims, and both settle.
    sim.over_cap = false;
    run(&mut sim, &mut maker, &mut taker, false, false)?;
    let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
    assert_eq!((m.phase.clone(), t.phase.clone()), (Phase::Done, Phase::Done));
    Ok(())
}

#[test]
fn a_maker_who_never_claims_leaves_each_side_its_refund() -> Result<()> {
    for side in [Side::SellMds, Side::SellMdw] {
        let mut sim = Sim::new();
        let (mut maker, mut taker) = wallets(side, &mut sim);
        negotiate(side, &sim, &mut maker, &mut taker)?;
        // The maker locks, then goes quiet: never claims. It still locks
        // first, so drive its lock by hand.
        turn(&mut maker, &mut sim, false)?;
        run(&mut sim, &mut maker, &mut taker, true, false)?;
        let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
        assert_eq!((m.phase.clone(), t.phase.clone()), (Phase::Refunded, Phase::Refunded), "{side:?}");
        assert!(t.t.is_none(), "{side:?}: the secret never came out");
        match side {
            Side::SellMds => {
                assert!(mwb_received(&taker.keys, &sim) > MDW_AMOUNT - MWB_FEE);
                assert!(mds_paid(m, MdsSpendKind::Refund, &sim) > MDS_AMOUNT - 1_000);
            }
            Side::SellMdw => {
                assert!(mwb_received(&maker.keys, &sim) > MDW_AMOUNT - MWB_FEE);
                assert!(mds_paid(t, MdsSpendKind::Refund, &sim) > MDS_AMOUNT - 1_000);
            }
        }
    }
    Ok(())
}

#[test]
fn the_maker_does_not_reveal_once_the_cutoff_has_passed() -> Result<()> {
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMds, &mut sim);
    negotiate(Side::SellMds, &sim, &mut maker, &mut taker)?;
    // Both lock...
    for _ in 0..4 {
        turn(&mut maker, &mut sim, false)?;
        turn(&mut taker, &mut sim, false)?;
        sim.mine();
    }
    // ...then the maker is away until 29 blocks before the taker's refund.
    let reveal = maker.swap.as_ref().unwrap().reveal_height;
    while sim.mwb.height < reveal - 29 {
        turn(&mut taker, &mut sim, false)?;
        sim.mine();
    }
    run(&mut sim, &mut maker, &mut taker, false, false)?;
    let (m, t) = (maker.swap.as_ref().unwrap(), taker.swap.as_ref().unwrap());
    assert_eq!((m.phase.clone(), t.phase.clone()), (Phase::Refunded, Phase::Refunded));
    assert!(t.t.is_none(), "a late maker keeps its secret");
    Ok(())
}

#[test]
fn every_message_of_the_negotiation_is_checked() -> Result<()> {
    // The taker's own share replaced by one it cannot prove: the maker
    // refuses before anything else.
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMds, &mut sim);
    let order_key = KeyPair::random();
    let order = MwbOrder {
        id: [9; 16],
        side: Side::SellMds,
        mds_amount: MDS_AMOUNT,
        mdw_amount: MDW_AMOUNT,
        key: order_key.to_bytes(),
        created: 0,
        expires: sim.mds.height + 100,
        status: OrderStatus::Open,
        last_advert: 0,
    };
    let taker_key = KeyPair::random();
    let (_, out) = Negotiation::take(&order.advert()?, &taker_key, &mut Wallet::ctx(&taker.keys, &mut taker.account, &sim))?;
    let (route, sealed) = deliver(&mut maker.inbox, &out).pop().unwrap();
    let session = maker_session(&order_key, &order.id, &taker_key.public)?;
    let Msg::Take(mut take) = route.open(&session, &sealed)? else { bail!("not a take") };
    take.share = midstate_walletd::mwb::OwnerShare::random().public;
    let err = Negotiation::answer(&order, take.clone(), session.clone(), &mut Wallet::ctx(&maker.keys, &mut maker.account, &sim))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("unproven"), "{err}");

    // A take for other terms than the order's.
    let Msg::Take(mut take) = route.open(&session, &sealed)? else { bail!("not a take") };
    take.mdw_amount -= 1;
    assert!(Negotiation::answer(&order, take, session, &mut Wallet::ctx(&maker.keys, &mut maker.account, &sim)).is_err());

    // An expired order cannot be taken.
    let expired = MwbOrder { expires: sim.mds.height, ..order };
    assert!(Negotiation::take(&expired.advert()?, &taker_key, &mut Wallet::ctx(&taker.keys, &mut taker.account, &sim)).is_err());
    Ok(())
}

/// Selling MDS locks behind `OP_MERKLE_ROOT`, which no script may use below
/// `SELL_MDS_FROM`: neither side starts such a swap before then. Selling MDW
/// is open at any height.
#[test]
fn selling_mds_waits_for_its_opcode() -> Result<()> {
    let mut sim = Sim::new();
    let (mut maker, mut taker) = wallets(Side::SellMds, &mut sim);
    let order_key = KeyPair::random();
    let order = |side| MwbOrder {
        id: [9; 16],
        side,
        mds_amount: MDS_AMOUNT,
        mdw_amount: MDW_AMOUNT,
        key: order_key.to_bytes(),
        created: 0,
        expires: SELL_MDS_FROM + 100,
        status: OrderStatus::Open,
        last_advert: 0,
    };
    let sell_mds = order(Side::SellMds);
    let taker_key = KeyPair::random();

    // One block short: the taker will not take it.
    sim.mds.height = SELL_MDS_FROM - 1;
    let err = Negotiation::take(&sell_mds.advert()?, &taker_key, &mut Wallet::ctx(&taker.keys, &mut taker.account, &sim))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("selling MDS opens"), "{err}");

    // A take made at the height (an older taker's, say) reaches a maker one
    // block short: the maker will not answer it.
    sim.mds.height = SELL_MDS_FROM;
    let (_, out) = Negotiation::take(&sell_mds.advert()?, &taker_key, &mut Wallet::ctx(&taker.keys, &mut taker.account, &sim))?;
    let (route, sealed) = deliver(&mut maker.inbox, &out).pop().expect("the take arrives");
    let session = maker_session(&order_key, &sell_mds.id, &taker_key.public)?;
    let Msg::Take(take) = route.open(&session, &sealed)? else { bail!("not a take") };
    sim.mds.height = SELL_MDS_FROM - 1;
    let err = Negotiation::answer(&sell_mds, take.clone(), session.clone(), &mut Wallet::ctx(&maker.keys, &mut maker.account, &sim))
        .err()
        .expect("refused");
    assert!(err.to_string().contains("selling MDS opens"), "{err}");
    // At the height it answers.
    sim.mds.height = SELL_MDS_FROM;
    Negotiation::answer(&sell_mds, take, session, &mut Wallet::ctx(&maker.keys, &mut maker.account, &sim))?;

    // Selling MDW is open below it.
    let mut sim = Sim::new();
    sim.mds.height = SELL_MDS_FROM - 1;
    let (_, mut taker) = wallets(Side::SellMdw, &mut sim);
    Negotiation::take(&order(Side::SellMdw).advert()?, &KeyPair::random(), &mut Wallet::ctx(&taker.keys, &mut taker.account, &sim))?;
    Ok(())
}
