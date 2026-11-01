//! MDS <-> MDW trading: orders, the swap journal, and the decisions that
//! take a swap from its locks to its claims.
//!
//! # Who does what
//!
//! The maker holds the secret in both directions, and locks first, on the
//! *follow leg*; the taker locks second, on the *reveal leg*. The maker
//! claims the taker's lock, which reveals the secret on the reveal leg's
//! chain, and the taker uses it to claim the maker's lock.
//!
//! | Side (what the maker sells) | maker locks           | taker locks         | secret comes out on |
//! |-----------------------------|-----------------------|---------------------|---------------------|
//! | `SellMds`                   | MDS, behind an OR lock | MDW (it funds)     | midwimble           |
//! | `SellMdw`                   | MDW (it funds)        | MDS, behind an AND lock | midstate        |
//!
//! Every signature either side needs is exchanged before the maker locks
//! (`mwb_negotiate`), so the chain part of a swap needs no further messages:
//! a counterparty that vanishes after the first lock leaves each side its
//! claim or its refund, decided here from what the two chains show.
//!
//! # Deadlines
//!
//! `reveal_height` is the reveal leg's refund height, on its chain;
//! `follow_height` the follow leg's (`swap::plan_mwb_timings`). The maker
//! reveals only while `swap::may_reveal`; the taker locks only while enough
//! of the reveal leg remains for the maker to see its lock and reveal.

use crate::mwb::{extract_adaptor_secret, finish_spend, JointView, KernelPlan};
use crate::mwb_account::{scalar, MwbAccount};
use crate::mwb_link::{and_claim_witness, and_preimages_from_witness, or_claim_witness, refund_witness, LinkRecord, LinkSecret, K_AND};
use crate::mwb_wire::{Advert, Session, Side};
use crate::swap::{may_reveal, REVEAL_CUTOFF_BLOCKS};
use anyhow::{anyhow, bail, ensure, Result};
use curve25519_dalek::ristretto::RistrettoPoint;
use midstate::core::types::{compute_coin_id, compute_commitment, hash, InputReveal, OutputData, Predicate, Witness};
use midstate::core::{decompose_value, wots, Transaction as MdsTx};
use midwimble::core::mw::crypto::{compress, Point32};
use midwimble::core::mw::transaction::{Kernel, Transaction as MwbTx};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Blocks on top of a lock before the other side builds on it.
pub const CONFIRMATIONS: u64 = 2;
/// Reveal-leg blocks the taker keeps in hand when it locks: its lock must
/// confirm, and the maker see it, before the reveal cutoff.
pub const TAKER_LOCK_MARGIN: u64 = 20;
/// The maker locks only while the taker would still have this many
/// reveal-leg blocks left to lock in.
pub const MAKER_LOCK_MARGIN: u64 = 15;
/// Midstate blocks between adverts of an open order.
pub const ADVERT_EVERY: u64 = 5;
/// A heard advert not repeated in this many midstate blocks has gone.
pub const ADVERT_STALE: u64 = 20;
/// Order lifetimes a maker may choose, in midstate blocks.
pub const MIN_ORDER_LIFE: u64 = 30;
pub const MAX_ORDER_LIFE: u64 = 10_080;
/// Smallest MDS amount tradeable: every lock coin is spent by a claim whose
/// fee comes out of the lock.
pub const MIN_MDS_AMOUNT: u64 = 4_096;
/// Most denominations an MDS amount may split into (one input each).
pub const MAX_MDS_COINS: usize = 16;
/// Smallest MDW amount: the claim's fee comes out of the lock.
pub const MIN_MDW_AMOUNT: u64 = 100_000;
/// Blocks between rebroadcasts of a transaction that has not confirmed.
pub const REBROADCAST_BLOCKS: u64 = 3;

// Midstate's relay floor (`fee * 1024 >= 10 * size`), as `swap.rs` repeats it.
const MIN_FEE_PER_KB: u64 = 10;
const FEE_RATE_SCALE: u64 = 1_024;
const MDS_FEE_MARGIN: u64 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Maker,
    Taker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leg {
    Mds,
    Mwb,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Signatures are being exchanged; nothing is locked. Held in memory
    /// (`mwb_negotiate`), never journalled.
    Negotiating,
    /// Taker: waiting for the maker's lock to confirm.
    WaitingTheirLock,
    /// Our lock is being broadcast and confirmed.
    LockingMine,
    /// Our lock is confirmed. Maker: waiting for the taker's; taker: waiting
    /// for the maker to claim it, which reveals the secret.
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
}

// ── Orders ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    Open,
    /// A taker is negotiating, or the swap it started is running.
    Taken { swap: [u8; 32] },
    Filled { swap: [u8; 32] },
    Cancelled,
}

/// An order this wallet made.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbOrder {
    pub id: [u8; 16],
    pub side: Side,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    /// The order's session key (Diffie–Hellman secret).
    pub key: [u8; 32],
    pub created: u64,
    /// Midstate height after which it no longer stands.
    pub expires: u64,
    pub status: OrderStatus,
    /// Midstate height of the last advert.
    #[serde(default)]
    pub last_advert: u64,
}

impl MwbOrder {
    pub fn advert(&self) -> Result<Advert> {
        let kp = crate::mwb_wire::KeyPair::from_bytes(&self.key).ok_or_else(|| anyhow!("order key is not canonical"))?;
        Ok(Advert {
            order_id: self.id,
            side: self.side,
            mds_amount: self.mds_amount,
            mdw_amount: self.mdw_amount,
            maker_key: kp.public,
            expires: self.expires,
        })
    }

    pub fn takeable(&self, mds_tip: u64) -> bool {
        self.status == OrderStatus::Open && mds_tip < self.expires
    }
}

/// Checks an order's terms, for the maker placing it and the taker taking
/// it alike.
pub fn check_terms(mds_amount: u64, mdw_amount: u64) -> Result<()> {
    ensure!(
        mds_amount >= MIN_MDS_AMOUNT,
        "the smallest MDS amount that can be swapped is {MIN_MDS_AMOUNT} units: the claim's fee comes out of the lock"
    );
    let coins = decompose_value(mds_amount).len();
    ensure!(
        coins <= MAX_MDS_COINS,
        "{mds_amount} units split into {coins} coins, more than the {MAX_MDS_COINS} one lock may hold; a rounder amount splits into fewer"
    );
    ensure!(
        mdw_amount >= MIN_MDW_AMOUNT,
        "the smallest MDW amount that can be swapped is {} MDW",
        midwimble::core::types::format_amount(MIN_MDW_AMOUNT)
    );
    Ok(())
}

/// An order heard on the bus.
#[derive(Clone, Debug)]
pub struct BookEntry {
    pub advert: Advert,
    /// Midstate tip when last heard.
    pub heard: u64,
}

// ── The journal ─────────────────────────────────────────────────────────────

/// A swap past its negotiation. Everything either side needs to finish it is
/// here, so it survives a restart at any point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbSwap {
    pub id: [u8; 32],
    pub role: Role,
    pub side: Side,
    pub order_id: [u8; 16],
    pub mds_amount: u64,
    pub mdw_amount: u64,
    pub session: Session,
    /// Unix seconds.
    pub created: u64,
    pub updated: u64,
    pub phase: Phase,
    pub reveal_height: u64,
    pub follow_height: u64,
    pub t_point: Point32,
    /// The maker's secret and its splits (`LinkSecret::to_record`).
    #[serde(default)]
    pub link_secret: Vec<[u8; 32]>,
    /// The verified link: which halves were opened, and the lock's hashes.
    pub link: LinkRecord,
    /// The secret, once this side knows it.
    pub t: Option<[u8; 32]>,

    // ── midstate leg ─────────────────────────────────────────────────────
    pub mds_script: Vec<u8>,
    /// One salt per denomination of `mds_amount`, in `decompose_value` order.
    pub mds_salts: Vec<[u8; 32]>,
    /// WOTS seed for this side's one key in the lock (claim or refund). It
    /// signs one message ever: this swap's deterministic claim or refund.
    pub mds_key: [u8; 32],
    /// Where our claim or refund pays; chosen once, before its commitment.
    pub mds_dest: Option<[u8; 32]>,
    /// Our lock's send (when we lock MDS).
    pub mds_lock_commitment: Option<[u8; 32]>,
    /// Our claim's or refund's commitment, once committed.
    pub mds_spend_commitment: Option<[u8; 32]>,
    /// Midstate tip at which every lock coin was first seen.
    pub mds_seen: Option<u64>,
    /// Next midstate height to search for the lock's spend.
    #[serde(default)]
    pub mds_scan_from: u64,
    /// Midstate tip of our last broadcast on this leg.
    pub mds_broadcast: Option<u64>,

    // ── midwimble leg ────────────────────────────────────────────────────
    pub view: JointView,
    pub joint_key: Point32,
    /// Funder: the funding transaction and the refund, complete.
    pub funding_tx: Option<String>,
    pub refund_tx: Option<String>,
    /// Receiver: the claim, but for its kernel signature.
    pub claim_tx: Option<String>,
    pub claim_plan: KernelPlan,
    /// Receiver: the claim kernel's Pedersen half.
    pub claim_s1: Option<[u8; 32]>,
    /// The claim's owner pre-signature: the receiver completes it with `t`;
    /// the funder, when the receiver holds `t`, reads `t` off it.
    pub claim_s2_pre: Option<[u8; 32]>,
    /// Height of the block that holds the lock output.
    pub mwb_seen: Option<u64>,
    /// Next midwimble height to search for the claim kernel.
    pub mwb_scan_from: u64,
    /// Midwimble tip of our last broadcast on this leg.
    pub mwb_broadcast: Option<u64>,

    /// Maker: its claim, which reveals the secret, has been broadcast. Until
    /// then, past the reveal cutoff it keeps the secret and refunds instead.
    #[serde(default)]
    pub reveal_sent: bool,

    /// What the user sees about where the swap stands.
    #[serde(default)]
    pub note: String,
}

impl MwbSwap {
    pub fn my_leg(&self) -> Leg {
        match (self.role, self.side) {
            (Role::Maker, Side::SellMds) | (Role::Taker, Side::SellMdw) => Leg::Mds,
            _ => Leg::Mwb,
        }
    }
    pub fn their_leg(&self) -> Leg {
        match self.my_leg() {
            Leg::Mds => Leg::Mwb,
            Leg::Mwb => Leg::Mds,
        }
    }
    pub fn reveal_leg(&self) -> Leg {
        match self.side {
            Side::SellMds => Leg::Mwb,
            Side::SellMdw => Leg::Mds,
        }
    }
    /// Funds the midwimble lock (and holds its refund).
    pub fn funder(&self) -> bool {
        self.my_leg() == Leg::Mwb
    }
    /// Our lock's refund height, on its chain.
    pub fn my_refund_height(&self) -> u64 {
        match self.role {
            Role::Maker => self.follow_height,
            Role::Taker => self.reveal_height,
        }
    }
    pub fn swap_tag(&self) -> String {
        format!("swap:{}", hex::encode(self.id))
    }
    pub fn link_secret(&self) -> Option<LinkSecret> {
        LinkSecret::from_record(&self.link_secret)
    }
    pub fn secret(&self) -> Option<curve25519_dalek::scalar::Scalar> {
        self.t.as_ref().and_then(|t| scalar(t, "secret").ok())
    }

    // ── midstate ─────────────────────────────────────────────────────────

    pub fn mds_lock_address(&self) -> [u8; 32] {
        Predicate::Script { bytecode: self.mds_script.clone() }.address()
    }

    /// The lock's coins: (value, salt), in `decompose_value` order.
    pub fn mds_lock_coins(&self) -> Vec<(u64, [u8; 32])> {
        decompose_value(self.mds_amount).into_iter().zip(self.mds_salts.iter().copied()).collect()
    }

    pub fn mds_lock_coin_ids(&self) -> Vec<[u8; 32]> {
        let addr = self.mds_lock_address();
        self.mds_lock_coins().iter().map(|(v, s)| compute_coin_id(&addr, *v, s)).collect()
    }

    /// The lock's outputs, for the side that funds it.
    pub fn mds_lock_outputs(&self) -> Vec<OutputData> {
        let address = self.mds_lock_address();
        self.mds_lock_coins().into_iter().map(|(value, salt)| OutputData::Standard { address, value, salt }).collect()
    }

    // ── midwimble ────────────────────────────────────────────────────────

    pub fn funding(&self) -> Result<MwbTx> {
        decode_mwb(self.funding_tx.as_deref().ok_or_else(|| anyhow!("no funding transaction"))?)
    }
    pub fn refund(&self) -> Result<MwbTx> {
        decode_mwb(self.refund_tx.as_deref().ok_or_else(|| anyhow!("no refund transaction"))?)
    }

    /// Receiver: the claim, complete, once `t` is known.
    pub fn claim(&self) -> Result<MwbTx> {
        let t = self.secret().ok_or_else(|| anyhow!("the secret is not known yet"))?;
        let s1 = scalar(self.claim_s1.as_ref().ok_or_else(|| anyhow!("no claim pre-signature"))?, "claim s1")?;
        let s2 = scalar(self.claim_s2_pre.as_ref().ok_or_else(|| anyhow!("no claim pre-signature"))?, "claim s2")?;
        let sig = self.claim_plan.complete(&s1, &s2, Some(&t))?;
        finish_spend(decode_mwb(self.claim_tx.as_deref().ok_or_else(|| anyhow!("no claim skeleton"))?)?, sig)
    }

    /// The commitment of our claim's (receiver) or refund's (funder) output.
    pub fn mwb_spend_output(&self, claim: bool) -> Result<Point32> {
        let tx = if claim {
            decode_mwb(self.claim_tx.as_deref().ok_or_else(|| anyhow!("no claim skeleton"))?)?
        } else {
            self.refund()?
        };
        let first = tx.outputs().next().map(|o| o.commitment);
        first.ok_or_else(|| anyhow!("a spend without an output"))
    }

    /// Funder, selling MDS: the secret from the claim kernel, if `k` is it.
    pub fn secret_from_kernel(&self, k: &Kernel) -> Option<[u8; 32]> {
        if k.excess != self.claim_plan.excess || k.owner_excess != self.claim_plan.owner_excess {
            return None;
        }
        let s2_pre = scalar(self.claim_s2_pre.as_ref()?, "claim s2").ok()?;
        let t = extract_adaptor_secret(&s2_pre, &k.signature)?;
        (compress(&RistrettoPoint::mul_base(&t)) == self.t_point).then(|| t.to_bytes())
    }

    /// Taker, selling MDW: the secret from the witness that spent a lock
    /// coin, if it is the maker's AND claim.
    pub fn secret_from_witness(&self, items: &[Vec<u8>]) -> Option<[u8; 32]> {
        let preimages = and_preimages_from_witness(items, K_AND)?;
        let link = self.link.link().ok()?;
        link.recover_t(&preimages).map(|t| t.to_bytes())
    }
}

fn decode_mwb(hex_tx: &str) -> Result<MwbTx> {
    Ok(bincode::deserialize(&hex::decode(hex_tx)?)?)
}

pub fn encode_mwb(tx: &MwbTx) -> Result<String> {
    Ok(hex::encode(bincode::serialize(tx)?))
}

// ── Spending the midstate lock ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MdsSpendKind {
    Claim,
    Refund,
}

/// A claim or refund of the midstate lock, as committed and as revealed.
#[derive(Clone, Debug)]
pub struct MdsSpend {
    pub inputs: Vec<InputReveal>,
    pub outputs: Vec<OutputData>,
    pub salt: [u8; 32],
    pub commitment: [u8; 32],
    pub fee: u64,
}

impl MdsSpend {
    pub fn output_ids(&self) -> Vec<[u8; 32]> {
        self.outputs.iter().filter_map(|o| o.coin_id()).collect()
    }
}

/// The witness items one input of this kind of spend carries.
fn spend_witness(sw: &MwbSwap, kind: MdsSpendKind, sig: &[u8]) -> Result<Vec<Vec<u8>>> {
    Ok(match (kind, sw.side) {
        (MdsSpendKind::Refund, _) => refund_witness(sig),
        (MdsSpendKind::Claim, Side::SellMds) => {
            // The taker opens the OR lock with the leaf its secret gives.
            let t = sw.secret().ok_or_else(|| anyhow!("the secret is not known yet"))?;
            let link = sw.link.link()?;
            let (j, preimage) = link.leaf_for_t(&t).ok_or_else(|| anyhow!("the secret opens no leaf of this lock"))?;
            or_claim_witness(sig, &link.merkle_proof(j), &preimage)
        }
        (MdsSpendKind::Claim, Side::SellMdw) => {
            // The maker publishes every unopened half.
            let secret = sw.link_secret().ok_or_else(|| anyhow!("only the maker can claim the AND lock"))?;
            let challenge = crate::mwb_link::Challenge(sw.link.challenge.clone());
            and_claim_witness(sig, &secret.unopened(&challenge))
        }
    })
}

fn mds_reveal(sp: &MdsSpend, witness: &[Vec<u8>]) -> MdsTx {
    MdsTx::Reveal {
        inputs: sp.inputs.clone(),
        witnesses: sp.inputs.iter().map(|_| Witness::ScriptInputs(witness.to_vec())).collect(),
        outputs: sp.outputs.clone(),
        salt: sp.salt,
    }
}

/// Builds our claim or refund of the midstate lock, paying `dest`.
///
/// # Formal Specification
///
/// ```text
/// Pre:  dest = sw.mds_dest (persisted before any commitment)
/// Post: a pure function of the journal: the commit and the reveal rebuild
///       the same transaction, so the one-time key signs one message
///       ∧ fee · 1024 ≥ 10 · |bincode(reveal)| + margin, the relay floor
/// ```
pub fn build_mds_spend(sw: &MwbSwap, kind: MdsSpendKind, dest: &[u8; 32]) -> Result<MdsSpend> {
    let total = sw.mds_amount;
    let inputs: Vec<InputReveal> = sw
        .mds_lock_coins()
        .into_iter()
        .map(|(value, salt)| InputReveal {
            predicate: Predicate::Script { bytecode: sw.mds_script.clone() },
            value,
            salt,
            commitment: None,
        })
        .collect();
    let label: &[u8] = match kind {
        MdsSpendKind::Claim => b"claim",
        MdsSpendKind::Refund => b"refund",
    };
    let derive = |what: &[u8], i: u32| {
        let mut h = Vec::with_capacity(96);
        h.extend_from_slice(b"midstate.mwb-swap.spend.v1");
        h.extend_from_slice(&sw.id);
        h.extend_from_slice(label);
        h.extend_from_slice(what);
        h.extend_from_slice(&i.to_le_bytes());
        hash(&h)
    };
    let salt = derive(b"tx", 0);
    let dummy = spend_witness(sw, kind, &vec![0u8; wots::SIG_SIZE]).or_else(|e| match kind {
        // The claim's witness needs the secret; its size does not.
        MdsSpendKind::Claim => claim_witness_shape(sw),
        MdsSpendKind::Refund => Err(e),
    })?;
    let mut fee = 0u64;
    for _ in 0..16 {
        ensure!(fee < total, "the lock cannot pay its own {kind:?} fee");
        let outputs: Vec<OutputData> = decompose_value(total - fee)
            .into_iter()
            .enumerate()
            .map(|(i, value)| OutputData::Standard { address: *dest, value, salt: derive(b"out", i as u32) })
            .collect();
        let sp = MdsSpend { inputs: inputs.clone(), outputs, salt, commitment: [0; 32], fee };
        let size = bincode::serialized_size(&mds_reveal(&sp, &dummy))?;
        let need = (size * MIN_FEE_PER_KB).div_ceil(FEE_RATE_SCALE) + MDS_FEE_MARGIN;
        if need <= fee {
            let ins: Vec<[u8; 32]> = sp.inputs.iter().map(|i| i.coin_id()).collect();
            let outs: Vec<[u8; 32]> = sp.outputs.iter().map(|o| o.hash_for_commitment()).collect();
            return Ok(MdsSpend { commitment: compute_commitment(&ins, &outs, &salt), ..sp });
        }
        fee = need;
    }
    bail!("the {kind:?} fee did not settle")
}

/// A claim witness of the right size, for pricing it before the secret is
/// known: a signature, then the OR lock's proof and leaf, or the AND lock's
/// halves.
fn claim_witness_shape(sw: &MwbSwap) -> Result<Vec<Vec<u8>>> {
    let sig = vec![0u8; wots::SIG_SIZE];
    Ok(match sw.side {
        Side::SellMds => {
            let link = sw.link.link()?;
            or_claim_witness(&sig, &link.merkle_proof(0), &[0u8; 32])
        }
        Side::SellMdw => and_claim_witness(&sig, &vec![[0u8; 32]; sw.link.opened.len()]),
    })
}

/// The reveal of a committed claim or refund, signed with our one-time key.
pub fn sign_mds_spend(sw: &MwbSwap, kind: MdsSpendKind, sp: &MdsSpend) -> Result<MdsTx> {
    let sig = wots::sig_to_bytes(&wots::sign(&sw.mds_key, &sp.commitment));
    Ok(mds_reveal(sp, &spend_witness(sw, kind, &sig)?))
}

pub fn mds_pk(seed: &[u8; 32]) -> [u8; 32] {
    wots::keygen(seed)
}

// ── Observing ───────────────────────────────────────────────────────────────

/// What a swap needs to read from the two chains. The service answers from
/// its embedded nodes; tests from simulated chains.
pub trait Chains {
    fn mds_tip(&self) -> u64;
    fn mwb_tip(&self) -> u64;
    fn mds_coin(&self, id: &[u8; 32]) -> bool;
    /// The witness of the input that spends `coin`, looked for in the
    /// mempool and in blocks from `from` up; and where the block search got to.
    fn mds_spend(&self, coin: &[u8; 32], from: u64) -> (Option<Vec<Vec<u8>>>, u64);
    /// An unspent output: the height of its block, and its owner key.
    fn mwb_output(&self, commitment: &Point32) -> Option<(u64, Point32)>;
    /// A kernel with this excess pair, in the mempool or in blocks from
    /// `from` up; and where the block search got to.
    fn mwb_kernel(&self, excess: &Point32, owner_excess: &Point32, from: u64) -> (Option<Kernel>, u64);
}

fn depth_ok(tip: u64, since: u64) -> bool {
    tip >= since + CONFIRMATIONS
}

/// Reads the chains for one swap, recording in it what only needs reading
/// once (when each lock appeared, where the searches got to, the secret).
///
/// `mine_pending` is the service's: whether our own lock's broadcast is
/// still under way.
pub fn observe(sw: &mut MwbSwap, c: &impl Chains, mine_pending: bool) -> Obs {
    let (mds_tip, mwb_tip) = (c.mds_tip(), c.mwb_tip());

    let ids = sw.mds_lock_coin_ids();
    let present = !ids.is_empty() && !sw.mds_script.is_empty() && ids.iter().all(|id| c.mds_coin(id));
    let mds_lock = if present {
        let seen = *sw.mds_seen.get_or_insert(mds_tip);
        if sw.mds_scan_from == 0 {
            sw.mds_scan_from = seen.saturating_sub(1);
        }
        if depth_ok(mds_tip, seen) {
            Lock::Confirmed
        } else {
            Lock::Unconfirmed
        }
    } else if sw.mds_seen.is_some() {
        Lock::Spent
    } else {
        Lock::Absent
    };

    let mwb_lock = match c.mwb_output(&sw.view.commitment) {
        Some((h, owner)) if owner == sw.joint_key => {
            sw.mwb_seen = Some(h);
            if sw.mwb_scan_from == 0 {
                sw.mwb_scan_from = h;
            }
            if depth_ok(mwb_tip, h) {
                Lock::Confirmed
            } else {
                Lock::Unconfirmed
            }
        }
        _ if sw.mwb_seen.is_some() => Lock::Spent,
        _ => Lock::Absent,
    };

    let (mine, theirs) = match sw.my_leg() {
        Leg::Mds => (mds_lock, mwb_lock),
        Leg::Mwb => (mwb_lock, mds_lock),
    };

    // The taker reads the secret off the maker's claim of the taker's lock.
    if sw.role == Role::Taker && sw.t.is_none() && mine != Lock::Absent {
        match sw.side {
            Side::SellMds => {
                let (k, next) = c.mwb_kernel(&sw.claim_plan.excess, &sw.claim_plan.owner_excess, sw.mwb_scan_from);
                sw.mwb_scan_from = next;
                sw.t = k.and_then(|k| sw.secret_from_kernel(&k));
            }
            Side::SellMdw => {
                if let Some(first) = ids.first() {
                    let (w, next) = c.mds_spend(first, sw.mds_scan_from);
                    sw.mds_scan_from = next;
                    sw.t = w.and_then(|w| sw.secret_from_witness(&w));
                }
            }
        }
    }

    let mds_spend_landed = |kind| {
        sw.mds_dest
            .and_then(|d| build_mds_spend(sw, kind, &d).ok())
            .is_some_and(|sp| sp.output_ids().iter().all(|id| c.mds_coin(id)))
    };
    let mwb_spend_landed = |claim| sw.mwb_spend_output(claim).ok().is_some_and(|o| c.mwb_output(&o).is_some());
    let claimed = match sw.their_leg() {
        Leg::Mds => mds_spend_landed(MdsSpendKind::Claim),
        Leg::Mwb => mwb_spend_landed(true),
    };
    let refunded = match sw.my_leg() {
        Leg::Mds => mds_spend_landed(MdsSpendKind::Refund),
        Leg::Mwb => mwb_spend_landed(false),
    };

    Obs {
        mds_tip,
        mwb_tip,
        mine,
        theirs,
        mine_pending,
        revealed: if sw.role == Role::Taker { sw.t } else { None },
        claimed,
        refunded,
    }
}

// ── Deciding ────────────────────────────────────────────────────────────────

/// A lock, as its chain shows it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lock {
    /// Not on chain.
    #[default]
    Absent,
    /// On chain, not yet `CONFIRMATIONS` deep.
    Unconfirmed,
    Confirmed,
    /// Was on chain; spent since.
    Spent,
}

/// What the two chains show about one swap, gathered by the service.
#[derive(Clone, Debug, Default)]
pub struct Obs {
    pub mds_tip: u64,
    pub mwb_tip: u64,
    pub mine: Lock,
    pub theirs: Lock,
    /// Our lock's broadcast is still under way (a midstate send not yet
    /// abandoned, or a funding transaction not yet refused).
    pub mine_pending: bool,
    /// Taker: the secret, read off the maker's claim and checked against `T`.
    pub revealed: Option<[u8; 32]>,
    /// Our claim of their lock is in a block.
    pub claimed: bool,
    /// Our refund is in a block.
    pub refunded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    LockMine,
    Claim,
    Refund,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub phase: Phase,
    pub act: Act,
    pub note: String,
}

impl Decision {
    fn wait(phase: &Phase, note: impl Into<String>) -> Self {
        Self { phase: phase.clone(), act: Act::None, note: note.into() }
    }
    fn go(phase: Phase, act: Act, note: impl Into<String>) -> Self {
        Self { phase, act, note: note.into() }
    }
}

/// The last reveal-leg height at which the taker may start its lock.
pub fn taker_lock_deadline(reveal_height: u64) -> u64 {
    reveal_height.saturating_sub(REVEAL_CUTOFF_BLOCKS + TAKER_LOCK_MARGIN)
}

/// The last reveal-leg height at which the maker may start its lock.
pub fn maker_lock_deadline(reveal_height: u64) -> u64 {
    taker_lock_deadline(reveal_height).saturating_sub(MAKER_LOCK_MARGIN)
}

fn tip_of(leg: Leg, o: &Obs) -> u64 {
    match leg {
        Leg::Mds => o.mds_tip,
        Leg::Mwb => o.mwb_tip,
    }
}

fn chain(leg: Leg) -> &'static str {
    match leg {
        Leg::Mds => "midstate",
        Leg::Mwb => "midwimble",
    }
}

/// The next step of a swap past its negotiation.
///
/// # Formal Specification
///
/// ```text
/// Pre:  sw.phase ∉ {Negotiating}; o is the chains as they stand
/// Post: act = Claim  ⇒ (maker: their lock confirmed ∧ may_reveal(reveal tip, reveal_height))
///                      ∨ (taker: the secret is known)
///                      ∨ already claiming (a rebroadcast)
///       act = Refund ⇒ our refund height has been reached ∧ ¬ claimed
///                      ∧ (maker: it never revealed, or its reveal lost the race)
///       act = LockMine ⇒ the lock deadline for our role has not passed
///       a maker whose claim confirmed never refunds: the taker holds the
///       secret and is owed the maker's lock
/// ```
pub fn decide(sw: &MwbSwap, o: &Obs) -> Decision {
    let reveal_tip = tip_of(sw.reveal_leg(), o);
    let my_tip = tip_of(sw.my_leg(), o);
    let refund_open = my_tip >= sw.my_refund_height();
    let maker = sw.role == Role::Maker;
    let refund_wait = || {
        format!(
            "our {} lock can be refunded from block {} (now {})",
            chain(sw.my_leg()),
            sw.my_refund_height(),
            my_tip
        )
    };
    match &sw.phase {
        Phase::Negotiating | Phase::Done | Phase::Refunded | Phase::Failed(_) => Decision::wait(&sw.phase, sw.note.clone()),

        Phase::WaitingTheirLock => match o.theirs {
            Lock::Confirmed if reveal_tip <= taker_lock_deadline(sw.reveal_height) => {
                Decision::go(Phase::LockingMine, Act::LockMine, "their lock is confirmed; locking ours")
            }
            _ if reveal_tip > taker_lock_deadline(sw.reveal_height) => Decision::go(
                Phase::Failed("the other side did not lock in time; nothing of ours was locked".into()),
                Act::None,
                "",
            ),
            Lock::Spent => Decision::go(
                Phase::Failed("the other side took its lock back; nothing of ours was locked".into()),
                Act::None,
                "",
            ),
            _ => Decision::wait(&sw.phase, format!("waiting for the other side's {} lock", chain(sw.their_leg()))),
        },

        Phase::LockingMine => match o.mine {
            Lock::Confirmed | Lock::Spent => decide(&MwbSwap { phase: Phase::MineLocked, ..sw.clone() }, o),
            Lock::Unconfirmed => Decision::wait(&sw.phase, format!("our {} lock is confirming", chain(sw.my_leg()))),
            Lock::Absent => {
                let started = match sw.my_leg() {
                    Leg::Mds => sw.mds_lock_commitment.is_some(),
                    Leg::Mwb => sw.mwb_broadcast.is_some(),
                };
                let deadline = if maker { maker_lock_deadline(sw.reveal_height) } else { taker_lock_deadline(sw.reveal_height) };
                if !started {
                    if reveal_tip > deadline {
                        Decision::go(Phase::Failed("too late to lock; nothing was locked".into()), Act::None, "")
                    } else {
                        Decision::go(Phase::LockingMine, Act::LockMine, format!("locking our {}", chain(sw.my_leg())))
                    }
                } else if o.mine_pending {
                    // A lock still on its way may yet confirm, after the
                    // other side has given up: keep it in sight, rebroadcast
                    // only while it is still useful.
                    let act = if reveal_tip <= deadline { Act::LockMine } else { Act::None };
                    Decision::go(Phase::LockingMine, act, format!("our {} lock is on its way", chain(sw.my_leg())))
                } else if refund_open {
                    Decision::go(
                        Phase::Failed("our lock never reached the chain, so nothing was locked".into()),
                        Act::None,
                        "",
                    )
                } else {
                    Decision::wait(&sw.phase, format!("our {} lock has not appeared", chain(sw.my_leg())))
                }
            }
        },

        Phase::MineLocked if maker => {
            if o.claimed {
                Decision::go(Phase::Done, Act::None, "")
            } else if o.theirs == Lock::Confirmed && may_reveal(reveal_tip, sw.reveal_height) {
                Decision::go(Phase::Claiming, Act::Claim, "claiming their lock, which reveals the secret")
            } else if !may_reveal(reveal_tip, sw.reveal_height) {
                let act = if refund_open { Act::Refund } else { Act::None };
                Decision::go(Phase::Refunding, act, format!("they did not lock in time; {}", refund_wait()))
            } else {
                Decision::wait(&sw.phase, format!("waiting for the other side's {} lock", chain(sw.their_leg())))
            }
        }
        Phase::MineLocked => {
            if o.revealed.is_some() {
                Decision::go(Phase::Claiming, Act::Claim, "the secret is out; claiming their lock")
            } else if refund_open {
                Decision::go(Phase::Refunding, Act::Refund, "they did not claim in time; refunding")
            } else {
                Decision::wait(&sw.phase, format!("waiting for the other side to claim; {}", refund_wait()))
            }
        }

        Phase::Claiming => {
            if o.claimed {
                Decision::go(Phase::Done, Act::None, "")
            } else if maker && o.theirs == Lock::Spent {
                // Our reveal lost the race to their refund.
                let act = if refund_open { Act::Refund } else { Act::None };
                Decision::go(Phase::Refunding, act, format!("their refund beat our claim; {}", refund_wait()))
            } else if maker && !sw.reveal_sent && !may_reveal(reveal_tip, sw.reveal_height) {
                // Our claim never went out (a midstate claim's commit can take
                // blocks), and it is now too late to start revealing.
                let act = if refund_open { Act::Refund } else { Act::None };
                Decision::go(Phase::Refunding, act, format!("too late to reveal safely; {}", refund_wait()))
            } else {
                Decision::go(Phase::Claiming, Act::Claim, format!("claiming their {} lock", chain(sw.their_leg())))
            }
        }

        Phase::Refunding => {
            if o.refunded {
                Decision::go(Phase::Refunded, Act::None, "")
            } else if o.claimed {
                Decision::go(Phase::Done, Act::None, "")
            } else if !maker && o.revealed.is_some() {
                // Their claim beat our refund: the secret is ours to use.
                Decision::go(Phase::Claiming, Act::Claim, "their claim came first; claiming theirs")
            } else if o.mine == Lock::Spent {
                if maker {
                    // Only the secret opens our lock to them, and only our
                    // claim, which lost the race to their refund, published it.
                    Decision::go(
                        Phase::Failed("our reveal came too late: their refund beat our claim and they then claimed our lock".into()),
                        Act::None,
                        "",
                    )
                } else {
                    Decision::wait(&sw.phase, "our lock was spent, and not by our refund: looking for the secret in it")
                }
            } else if refund_open && o.mine != Lock::Absent {
                Decision::go(Phase::Refunding, Act::Refund, "refunding our lock")
            } else if refund_open && !o.mine_pending {
                Decision::go(Phase::Failed("our lock never reached the chain, so nothing was locked".into()), Act::None, "")
            } else {
                Decision::wait(&sw.phase, refund_wait())
            }
        }
    }
}

// ── The store ───────────────────────────────────────────────────────────────

/// Everything walletd keeps about midwimble, in `WalletData::mwb_state`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MwbStore {
    pub account: MwbAccount,
    #[serde(default)]
    pub orders: Vec<MwbOrder>,
    #[serde(default)]
    pub swaps: Vec<MwbSwap>,
}

impl MwbStore {
    pub fn load(v: Option<&serde_json::Value>) -> Result<Self> {
        match v {
            None => Ok(Self::default()),
            Some(v) => Ok(serde_json::from_value(v.clone())?),
        }
    }
    pub fn to_value(&self) -> Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }
    pub fn swap_mut(&mut self, id: &[u8; 32]) -> Option<&mut MwbSwap> {
        self.swaps.iter_mut().find(|s| &s.id == id)
    }
    pub fn order_mut(&mut self, id: &[u8; 16]) -> Option<&mut MwbOrder> {
        self.orders.iter_mut().find(|o| &o.id == id)
    }
    /// Settled swaps keep their summary; the secrets and transactions that
    /// only mattered while they ran are dropped, so the journal stays small.
    pub fn compact(&mut self) {
        for s in self.swaps.iter_mut().filter(|s| s.phase.settled()) {
            s.link_secret.clear();
            s.link.opened.clear();
            s.link.unopened_hashes.clear();
            s.link.challenge.clear();
            s.mds_script.clear();
            s.funding_tx = None;
            s.refund_tx = None;
            s.claim_tx = None;
            s.claim_s1 = None;
            s.claim_s2_pre = None;
            s.mds_key = [0; 32];
        }
    }
}

/// Heard orders, by id.
pub type Book = HashMap<[u8; 16], BookEntry>;

#[cfg(test)]
mod tests {
    use super::*;

    fn swap(role: Role, side: Side, phase: Phase) -> MwbSwap {
        MwbSwap {
            id: [1; 32],
            role,
            side,
            order_id: [2; 16],
            mds_amount: 1 << 20,
            mdw_amount: 1_000_000,
            session: Session { key: [0; 32], tag: [0; 16], swap_id: [1; 32] },
            created: 0,
            updated: 0,
            phase,
            reveal_height: 1_120,
            follow_height: 2_280,
            t_point: [0; 32],
            link_secret: vec![],
            link: LinkRecord { t_point: [0; 32], challenge: vec![], opened: vec![], unopened_hashes: vec![] },
            t: None,
            mds_script: vec![],
            mds_salts: vec![],
            mds_key: [0; 32],
            mds_dest: None,
            mds_lock_commitment: None,
            mds_spend_commitment: None,
            mds_seen: None,
            mds_scan_from: 0,
            mds_broadcast: None,
            view: JointView { commitment: [0; 32], owner_key: [0; 32], value: 0, blinding: [0; 32] },
            joint_key: [0; 32],
            funding_tx: None,
            refund_tx: None,
            claim_tx: None,
            claim_plan: KernelPlan {
                fee: 0,
                min_height: 0,
                excess: [0; 32],
                owner_excess: [0; 32],
                r1: [0; 32],
                r2_builder: [0; 32],
                r2_other: [0; 32],
                adaptor: None,
            },
            claim_s1: None,
            claim_s2_pre: None,
            mwb_seen: None,
            mwb_scan_from: 0,
            mwb_broadcast: None,
            reveal_sent: false,
            note: String::new(),
        }
    }

    /// Selling MDS: the reveal leg is midwimble (refund 1,120), the follow
    /// leg midstate (refund 2,280).
    fn obs(mds_tip: u64, mwb_tip: u64) -> Obs {
        Obs { mds_tip, mwb_tip, ..Obs::default() }
    }

    #[test]
    fn the_maker_reveals_only_inside_the_cutoff() {
        let sw = swap(Role::Maker, Side::SellMds, Phase::MineLocked);
        let o = Obs { theirs: Lock::Confirmed, mine: Lock::Confirmed, ..obs(2_000, 1_050) };
        assert_eq!(decide(&sw, &o).act, Act::Claim);
        // 30 blocks before the taker's refund opens, the maker stops.
        let o = Obs { theirs: Lock::Confirmed, mine: Lock::Confirmed, ..obs(2_000, 1_091) };
        let d = decide(&sw, &o);
        assert_eq!((d.phase, d.act), (Phase::Refunding, Act::None), "it waits for its own refund height");
        let o = Obs { theirs: Lock::Confirmed, mine: Lock::Confirmed, ..obs(2_280, 1_200) };
        assert_eq!(decide(&MwbSwap { phase: Phase::Refunding, ..sw.clone() }, &o).act, Act::Refund);
    }

    #[test]
    fn a_maker_does_not_start_revealing_past_the_cutoff() {
        // Selling MDW the maker claims on midstate, commit then reveal.
        let sw = swap(Role::Maker, Side::SellMdw, Phase::Claiming);
        let o = Obs { theirs: Lock::Confirmed, mine: Lock::Confirmed, ..obs(1_089, 2_000) };
        assert_eq!(decide(&sw, &o).act, Act::Claim, "31 blocks left");
        let o = Obs { theirs: Lock::Confirmed, mine: Lock::Confirmed, ..obs(1_091, 2_000) };
        assert_eq!(decide(&sw, &o).phase, Phase::Refunding, "29 left and nothing revealed yet");
        // Once the reveal is out, the secret is public: finish the claim.
        let sent = MwbSwap { reveal_sent: true, ..sw };
        assert_eq!(decide(&sent, &o).act, Act::Claim);
    }

    #[test]
    fn a_maker_whose_claim_landed_never_refunds() {
        let sw = swap(Role::Maker, Side::SellMds, Phase::Claiming);
        let o = Obs { claimed: true, mine: Lock::Confirmed, ..obs(3_000, 2_000) };
        assert_eq!(decide(&sw, &o).phase, Phase::Done);
        // Done is final, however late it gets.
        let done = MwbSwap { phase: Phase::Done, ..sw };
        assert_eq!(decide(&done, &o).act, Act::None);
    }

    #[test]
    fn the_taker_locks_only_while_the_maker_can_still_reveal() {
        let sw = swap(Role::Taker, Side::SellMds, Phase::WaitingTheirLock);
        let deadline = taker_lock_deadline(1_120);
        assert_eq!(deadline, 1_070);
        let o = Obs { theirs: Lock::Confirmed, ..obs(2_000, deadline) };
        assert_eq!(decide(&sw, &o).act, Act::LockMine);
        let o = Obs { theirs: Lock::Confirmed, ..obs(2_000, deadline + 1) };
        assert!(matches!(decide(&sw, &o).phase, Phase::Failed(_)));
        let o = Obs { theirs: Lock::Unconfirmed, ..obs(2_000, 1_000) };
        assert_eq!(decide(&sw, &o).act, Act::None);
    }

    #[test]
    fn the_taker_claims_on_the_reveal_and_refunds_without_one() {
        let sw = swap(Role::Taker, Side::SellMds, Phase::MineLocked);
        let o = Obs { revealed: Some([3; 32]), mine: Lock::Spent, ..obs(2_000, 1_100) };
        assert_eq!(decide(&sw, &o).act, Act::Claim);
        let o = Obs { mine: Lock::Confirmed, ..obs(2_000, 1_119) };
        assert_eq!(decide(&sw, &o).act, Act::None);
        let o = Obs { mine: Lock::Confirmed, ..obs(2_000, 1_120) };
        assert_eq!(decide(&sw, &o).act, Act::Refund);
        // The maker's claim wins the race against the refund: claim theirs.
        let refunding = MwbSwap { phase: Phase::Refunding, ..sw };
        let o = Obs { revealed: Some([3; 32]), mine: Lock::Spent, ..obs(2_000, 1_121) };
        assert_eq!(decide(&refunding, &o).act, Act::Claim);
    }

    #[test]
    fn a_lock_that_never_left_settles_without_a_refund() {
        let sw = swap(Role::Maker, Side::SellMdw, Phase::LockingMine);
        // Selling MDW, the maker's lock is on midwimble with refund 2,280.
        let o = obs(1_000, 1_000);
        assert_eq!(decide(&sw, &o).act, Act::LockMine);
        let started = MwbSwap { mwb_broadcast: Some(1_000), ..sw.clone() };
        assert_eq!(decide(&started, &Obs { mine_pending: true, ..obs(1_000, 1_001) }).act, Act::LockMine);
        assert!(matches!(decide(&started, &obs(1_200, 2_300)).phase, Phase::Failed(_)));
        // Too late to start at all.
        assert!(matches!(decide(&sw, &obs(1_100, 1_100)).phase, Phase::Failed(_)));
    }

    #[test]
    fn the_journal_round_trips_and_compacts_what_settled() {
        let mut store = MwbStore::default();
        let mut live = swap(Role::Taker, Side::SellMds, Phase::MineLocked);
        live.link_secret = vec![[3; 32]; 3];
        live.refund_tx = Some("00".into());
        live.mds_key = [7; 32];
        let done = MwbSwap { id: [2; 32], phase: Phase::Done, ..live.clone() };
        store.swaps = vec![live.clone(), done];
        let back = MwbStore::load(Some(&store.to_value().unwrap())).unwrap();
        assert_eq!(back.swaps.len(), 2);
        assert_eq!(back.swaps[0].mds_key, [7; 32]);
        assert_eq!(back.swaps[0].phase, Phase::MineLocked);

        store.compact();
        let (l, d) = (&store.swaps[0], &store.swaps[1]);
        assert_eq!((l.refund_tx.as_deref(), l.mds_key), (Some("00"), [7; 32]), "a live swap keeps everything");
        assert!(d.refund_tx.is_none() && d.link_secret.is_empty() && d.mds_key == [0; 32], "a settled one keeps no secret");
        assert_eq!((d.mds_amount, d.phase.clone()), (live.mds_amount, Phase::Done), "but keeps its summary");
        // A wallet without midwimble state loads an empty store.
        assert!(MwbStore::load(None).unwrap().swaps.is_empty());
    }

    #[test]
    fn terms_are_checked() {
        assert!(check_terms(MIN_MDS_AMOUNT, MIN_MDW_AMOUNT).is_ok());
        assert!(check_terms(MIN_MDS_AMOUNT - 1, MIN_MDW_AMOUNT).is_err());
        assert!(check_terms(MIN_MDS_AMOUNT, MIN_MDW_AMOUNT - 1).is_err());
        assert!(check_terms((1 << 20) - 1, MIN_MDW_AMOUNT).is_err(), "twenty coins");
    }
}
