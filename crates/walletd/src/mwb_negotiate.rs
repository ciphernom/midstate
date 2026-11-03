//! The two-party negotiation of an MDS <-> MDW swap: everything both sides
//! must hold before the first lock, exchanged in alternating messages.
//!
//! ```text
//!  seq  selling MDS (taker funds MDW)          selling MDW (maker funds MDW)
//!   0   T→M  Take                              T→M  Take
//!   1   M→T  Offer: link proof (Fiat–Shamir)   M→T  Offer: link commitment, lock,
//!                                                         nonce commitments
//!   2   T→M  lock, nonce commitments           T→M  challenge, nonces
//!   3   M→T  nonces                            M→T  opening, nonces, input partial,
//!                                                         refund plan
//!   4   T→M  nonces, input partial,            T→M  input and refund partials,
//!            refund plan                                claim plan
//!   5   M→T  input and refund partials,        M→T  claim partial          (done)
//!            claim plan, own claim partial
//!   6   T→M  claim partial            (done)
//! ```
//!
//! The funder commits to its nonces before the receiver reveals its own, and
//! reveals only after, so neither picks a nonce after seeing the other's.
//! Each side's last check before it is done is that the other's partial
//! signatures verify: the funder ends holding a complete refund, the
//! receiver a claim pre-signature that `t` completes. Nothing is locked
//! until both are done, and from then on the chains decide (`mwb_trade`).
//!
//! Nonces are never written down: a negotiation interrupted by a restart is
//! simply abandoned, which costs nothing because nothing is locked yet.

use crate::mwb::{
    build_funding, joint_key, min_fee, verify_share, InputSigning, JointOutput, JointSpend, KernelPlan, OwnerShare,
    SecretNonce,
};
use crate::mwb_account::{scalar, MwbAccount};
use crate::mwb_link::{verify, verify_proof, Challenge, LinkCommitment, LinkRecord, LinkSecret, K_AND, K_OR};
use crate::mwb_trade::{check_side_open, check_terms, encode_mwb, mds_pk, taker_lock_deadline, MwbOrder, MwbSwap, Phase, Role};
use crate::mwb_wire::{
    point, session_messages, take_messages, taker_session, Advert, FunderCommit, FunderNonces, KeyPair, Msg, Offer,
    OfferLink, Outgoing, ReceiverNonces, ReceiverSigs, Session, Side, Take, To,
};
use crate::swap::{check_mwb_timings, plan_mwb_timings, MWB_REVEAL_BLOCKS};
use anyhow::{anyhow, bail, ensure, Result};
use curve25519_dalek::scalar::Scalar;
use midstate::core::decompose_value;
use midwimble::core::mw::crypto::Point32;
use midwimble::core::mw::stealth::WalletKeys;
use midwimble::core::mw::transaction::Transaction as MwbTx;

/// Seconds between resends of our last message while no answer comes.
pub const RESEND_SECS: u64 = 30;
pub const MAX_RESENDS: u32 = 8;
/// A negotiation that has not moved for this long is abandoned.
pub const NEGOTIATION_SECS: u64 = 300;
/// The longest reveal leg a taker accepts, in blocks: its funds wait that
/// long if the maker vanishes.
pub const MAX_REVEAL_BLOCKS: u64 = 2 * MWB_REVEAL_BLOCKS;
/// Reveal-leg blocks a taker wants in hand beyond its own lock deadline when
/// it accepts an offer: the maker locks first.
const OFFER_SLACK: u64 = 20;
/// The last message of each direction.
pub const FINAL_SEQ_SELL_MDS: u8 = 6;
pub const FINAL_SEQ_SELL_MDW: u8 = 5;

/// What negotiating needs from the wallet.
pub struct Ctx<'a> {
    pub mds_tip: u64,
    pub mwb_tip: u64,
    /// The MDW account: where claims and refunds pay, and change goes.
    pub keys: &'a WalletKeys,
    /// The funder's coins.
    pub account: &'a mut MwbAccount,
    /// Unix seconds.
    pub now: u64,
}

/// One side's four nonces, in the order `mwb_wire` names them.
struct Nonces {
    n: [Option<SecretNonce>; 4],
    points: [Point32; 4],
}

impl Nonces {
    /// Input (G), refund Pedersen (H), refund owner as builder (G), claim
    /// owner as other (G).
    fn funder() -> Self {
        Self::of([SecretNonce::on_g(), SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g()])
    }
    /// Input (G), refund owner as other (G), claim Pedersen (H), claim owner
    /// as builder (G).
    fn receiver() -> Self {
        Self::of([SecretNonce::on_g(), SecretNonce::on_g(), SecretNonce::on_h(), SecretNonce::on_g()])
    }
    fn of(n: [SecretNonce; 4]) -> Self {
        let points = [n[0].point, n[1].point, n[2].point, n[3].point];
        let [a, b, c, d] = n;
        Self { n: [Some(a), Some(b), Some(c), Some(d)], points }
    }
    fn take(&mut self, i: usize) -> Result<SecretNonce> {
        self.n[i].take().ok_or_else(|| anyhow!("nonce {i} already used"))
    }
    fn commits(&self) -> [[u8; 32]; 4] {
        self.points.map(|p| crate::mwb::nonce_commitment(&p))
    }
}

/// What a received message led to.
pub enum Step {
    /// Send this, then wait for the next message.
    Reply(Vec<Outgoing>),
    /// The negotiation is complete; send this (possibly nothing) last.
    Complete(Vec<Outgoing>),
    /// The other side repeated its previous message: ours was lost.
    Resend,
    /// Nothing to do (an old or out-of-order message).
    Ignore,
}

pub struct Negotiation {
    /// The journal entry being filled in.
    pub swap: MwbSwap,
    /// The other side's next message.
    pub expect: u8,
    /// Our last message, verbatim, for resends.
    pub last_out: Vec<Outgoing>,
    pub last_sent: u64,
    pub resends: u32,
    /// Unix seconds of the last step forward.
    pub progress: u64,
    share: OwnerShare,
    their_share: Point32,
    their_mds_pk: [u8; 32],
    secret: Option<LinkSecret>,
    commitment: Option<LinkCommitment>,
    challenge: Option<Challenge>,
    funding: Option<MwbTx>,
    refund_draft: Option<JointSpend>,
    claim_draft: Option<JointSpend>,
    ours: Nonces,
    their_commits: Option<[[u8; 32]; 4]>,
    theirs: Option<[Point32; 4]>,
    input: Option<InputSigning>,
    input_partial: Option<Scalar>,
    refund_plan: Option<KernelPlan>,
    claim_s2_ours: Option<Scalar>,
}

fn our_to(role: Role) -> To {
    match role {
        Role::Maker => To::Taker,
        Role::Taker => To::Maker,
    }
}

fn final_seq(side: Side) -> u8 {
    match side {
        Side::SellMds => FINAL_SEQ_SELL_MDS,
        Side::SellMdw => FINAL_SEQ_SELL_MDW,
    }
}

/// A journal entry with only what is known at the start.
fn draft(role: Role, side: Side, order_id: [u8; 16], mds_amount: u64, mdw_amount: u64, session: Session, mds_key: [u8; 32], now: u64) -> MwbSwap {
    MwbSwap {
        id: session.swap_id,
        role,
        side,
        order_id,
        mds_amount,
        mdw_amount,
        session,
        created: now,
        updated: now,
        phase: Phase::Negotiating,
        reveal_height: 0,
        follow_height: 0,
        t_point: [0; 32],
        link_secret: Vec::new(),
        link: LinkRecord { t_point: [0; 32], challenge: vec![], opened: vec![], unopened_hashes: vec![] },
        t: None,
        mds_script: Vec::new(),
        mds_salts: Vec::new(),
        mds_key,
        mds_dest: None,
        mds_lock_commitment: None,
        mds_spend_commitment: None,
        mds_seen: None,
        mds_scan_from: 0,
        mds_broadcast: None,
        view: crate::mwb::JointView { commitment: [0; 32], owner_key: [0; 32], value: 0, blinding: [0; 32] },
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
        note: "negotiating".into(),
    }
}

fn salts(mds_amount: u64) -> Vec<[u8; 32]> {
    decompose_value(mds_amount).iter().map(|_| rand::random()).collect()
}

/// The refund height of the midwimble lock, which is the reveal leg when
/// selling MDS and the follow leg when selling MDW.
fn mwb_refund_height(sw: &MwbSwap) -> u64 {
    match sw.side {
        Side::SellMds => sw.reveal_height,
        Side::SellMdw => sw.follow_height,
    }
}

impl Negotiation {
    fn new(swap: MwbSwap, share: OwnerShare, their_share: Point32, their_mds_pk: [u8; 32], ours: Nonces, now: u64) -> Self {
        Self {
            swap,
            expect: 0,
            last_out: Vec::new(),
            last_sent: now,
            resends: 0,
            progress: now,
            share,
            their_share,
            their_mds_pk,
            secret: None,
            commitment: None,
            challenge: None,
            funding: None,
            refund_draft: None,
            claim_draft: None,
            ours,
            their_commits: None,
            theirs: None,
            input: None,
            input_partial: None,
            refund_plan: None,
            claim_s2_ours: None,
        }
    }

    /// Taker: opens a negotiation on a heard order.
    pub fn take(advert: &Advert, key: &KeyPair, ctx: &mut Ctx) -> Result<(Self, Vec<Outgoing>)> {
        check_terms(advert.mds_amount, advert.mdw_amount)?;
        check_side_open(advert.side, ctx.mds_tip)?;
        ensure!(ctx.mds_tip < advert.expires, "that order has expired");
        let session = taker_session(key, advert)?;
        if advert.side == Side::SellMds {
            // We fund the midwimble lock: be able to before asking.
            ctx.account.select(advert.mdw_amount, ctx.mwb_tip, |n| min_fee(n, 2, 1))?;
        }
        let share = OwnerShare::random();
        let mds_key: [u8; 32] = rand::random();
        let msg = Msg::Take(Take {
            side: advert.side,
            mds_amount: advert.mds_amount,
            mdw_amount: advert.mdw_amount,
            share: share.public,
            pok: share.prove(&session.swap_id),
            mds_pk: mds_pk(&mds_key),
        });
        let out = take_messages(&session, &advert.order_id, &key.public, &msg)?;
        let swap = draft(Role::Taker, advert.side, advert.order_id, advert.mds_amount, advert.mdw_amount, session, mds_key, ctx.now);
        let ours = match advert.side {
            Side::SellMds => Nonces::funder(),
            Side::SellMdw => Nonces::receiver(),
        };
        let mut n = Self::new(swap, share, [0; 32], [0; 32], ours, ctx.now);
        n.expect = 1;
        n.last_out = out.clone();
        Ok((n, out))
    }

    /// Maker: answers a take of one of its orders.
    pub fn answer(order: &MwbOrder, take: Take, session: Session, ctx: &mut Ctx) -> Result<(Self, Vec<Outgoing>)> {
        ensure!(order.takeable(ctx.mds_tip), "that order is no longer open");
        check_side_open(order.side, ctx.mds_tip)?;
        ensure!(
            take.side == order.side && take.mds_amount == order.mds_amount && take.mdw_amount == order.mdw_amount,
            "the take does not match the order's terms"
        );
        let swap_id = session.swap_id;
        ensure!(verify_share(&take.share, &swap_id, &take.pok), "the taker's key share is unproven");
        point(&take.share, "the taker's key share")?;
        let share = OwnerShare::random();
        let mds_key: [u8; 32] = rand::random();
        let mut sw = draft(Role::Maker, order.side, order.id, order.mds_amount, order.mdw_amount, session, mds_key, ctx.now);
        let (funder_share, receiver_share) = match order.side {
            Side::SellMds => (take.share, share.public),
            Side::SellMdw => (share.public, take.share),
        };
        sw.joint_key = joint_key(&funder_share, &receiver_share)?;
        let timings = match order.side {
            Side::SellMds => plan_mwb_timings(ctx.mwb_tip, ctx.mds_tip, MWB_REVEAL_BLOCKS)?,
            Side::SellMdw => plan_mwb_timings(ctx.mds_tip, ctx.mwb_tip, MWB_REVEAL_BLOCKS)?,
        };
        sw.reveal_height = timings.reveal_height;
        sw.follow_height = timings.follow_height;

        let pok = share.prove(&swap_id);
        let mut n;
        let msg = match order.side {
            Side::SellMds => {
                let secret = LinkSecret::generate(K_OR);
                let proof = secret.prove(&swap_id)?;
                let link = verify_proof(&swap_id, &proof)?;
                sw.t_point = secret.t_point();
                sw.t = Some(secret.t().to_bytes());
                sw.link_secret = secret.to_record();
                sw.link = LinkRecord::of(&link);
                sw.mds_salts = salts(order.mds_amount);
                // OR lock: the taker claims, we refund at the follow height.
                sw.mds_script = link.or_lock(&take.mds_pk, sw.follow_height, &mds_pk(&mds_key));
                let offer = Offer {
                    share: share.public,
                    pok,
                    mds_pk: mds_pk(&mds_key),
                    reveal_height: sw.reveal_height,
                    follow_height: sw.follow_height,
                    link: OfferLink::Proof(proof),
                    mds_salts: sw.mds_salts.clone(),
                    funder: None,
                };
                n = Self::new(sw, share, take.share, take.mds_pk, Nonces::receiver(), ctx.now);
                n.secret = Some(secret);
                Msg::Offer(offer)
            }
            Side::SellMdw => {
                let secret = LinkSecret::generate(K_AND);
                let commitment = secret.commit();
                sw.t_point = secret.t_point();
                sw.t = Some(secret.t().to_bytes());
                sw.link_secret = secret.to_record();
                n = Self::new(sw, share, take.share, take.mds_pk, Nonces::funder(), ctx.now);
                n.fund(ctx)?;
                let offer = Offer {
                    share: n.share.public,
                    pok,
                    mds_pk: mds_pk(&mds_key),
                    reveal_height: n.swap.reveal_height,
                    follow_height: n.swap.follow_height,
                    link: OfferLink::Commitment(commitment.clone()),
                    mds_salts: Vec::new(),
                    funder: Some(FunderCommit { view: n.swap.view.clone(), commits: n.ours.commits() }),
                };
                n.secret = Some(secret);
                n.commitment = Some(commitment);
                Msg::Offer(offer)
            }
        };
        n.expect = 2;
        let out = session_messages(&n.swap.session, To::Taker, 1, &msg)?;
        n.last_out = out.clone();
        Ok((n, out))
    }

    /// Funder: builds the lock, the funding transaction and the refund's
    /// draft, and sets the coins aside.
    fn fund(&mut self, ctx: &mut Ctx) -> Result<()> {
        let sw = &mut self.swap;
        let (spend, fee) = ctx.account.select(sw.mdw_amount, ctx.mwb_tip, |n| min_fee(n, 2, 1))?;
        let joint = JointOutput::new(sw.mdw_amount, sw.joint_key)?;
        let funding = build_funding(&spend, &joint, &ctx.keys.address(), fee)?;
        sw.view = joint.view();
        let refund_height = mwb_refund_height(sw);
        self.refund_draft = Some(JointSpend::new(&sw.view, &self.share, &self.their_share, &ctx.keys.address(), min_fee(1, 1, 1), refund_height)?);
        let coins: Vec<Point32> = spend.iter().map(|s| s.commitment).collect();
        ctx.account.reserve(&coins, &sw.swap_tag())?;
        sw.funding_tx = Some(encode_mwb(&funding)?);
        self.funding = Some(funding);
        Ok(())
    }

    /// Releases what this negotiation set aside (on failure).
    pub fn abandon(&self, account: &mut MwbAccount) {
        account.release(&self.swap.swap_tag());
    }

    /// The other side's message `seq`.
    pub fn receive(&mut self, seq: u8, msg: Msg, ctx: &mut Ctx) -> Result<Step> {
        if let Msg::Abort { reason } = &msg {
            bail!("the other side stopped: {}", crate::mwb_wire::reason(reason));
        }
        if seq != self.expect {
            return Ok(if seq.wrapping_add(2) == self.expect { Step::Resend } else { Step::Ignore });
        }
        let name = msg.name();
        let (role, side) = (self.swap.role, self.swap.side);
        let reply = match (role, side, seq, msg) {
            (Role::Taker, _, 1, Msg::Offer(o)) => Some(self.on_offer(o, ctx)?),
            (Role::Maker, Side::SellMds, 2, Msg::FunderCommit(f)) => Some(self.on_funder_commit(f, ctx)?),
            (Role::Maker, Side::SellMdw, 2, Msg::ReceiverNonces(r)) => Some(self.on_receiver_nonces(r)?),
            (Role::Taker, Side::SellMds, 3, Msg::ReceiverNonces(r)) => Some(self.on_receiver_nonces(r)?),
            (Role::Taker, Side::SellMdw, 3, Msg::FunderNonces(f)) => Some(self.on_funder_nonces(f)?),
            (Role::Maker, Side::SellMds, 4, Msg::FunderNonces(f)) => Some(self.on_funder_nonces(f)?),
            (Role::Maker, Side::SellMdw, 4, Msg::ReceiverSigs(r)) => Some(self.on_receiver_sigs(r)?),
            (Role::Taker, Side::SellMds, 5, Msg::ReceiverSigs(r)) => Some(self.on_receiver_sigs(r)?),
            (Role::Taker, Side::SellMdw, 5, Msg::FunderClaimSig { claim_partial }) => {
                self.on_funder_claim_sig(&claim_partial)?;
                None
            }
            (Role::Maker, Side::SellMds, 6, Msg::FunderClaimSig { claim_partial }) => {
                self.on_funder_claim_sig(&claim_partial)?;
                None
            }
            _ => bail!("unexpected {name} as message {seq}"),
        };
        self.progress = ctx.now;
        self.resends = 0;
        let Some(msg) = reply else {
            return Ok(Step::Complete(Vec::new()));
        };
        let next = seq + 1;
        let out = session_messages(&self.swap.session, our_to(role), next, &msg)?;
        self.last_out = out.clone();
        self.last_sent = ctx.now;
        self.expect = next + 1;
        if next == final_seq(side) {
            Ok(Step::Complete(out))
        } else {
            Ok(Step::Reply(out))
        }
    }

    // ── Taker ────────────────────────────────────────────────────────────

    fn on_offer(&mut self, o: Offer, ctx: &mut Ctx) -> Result<Msg> {
        let swap_id = self.swap.session.swap_id;
        ensure!(verify_share(&o.share, &swap_id, &o.pok), "the maker's key share is unproven");
        point(&o.share, "the maker's key share")?;
        self.their_share = o.share;
        self.their_mds_pk = o.mds_pk;
        let side = self.swap.side;
        let (funder_share, receiver_share) = match side {
            Side::SellMds => (self.share.public, o.share),
            Side::SellMdw => (o.share, self.share.public),
        };
        let jk = joint_key(&funder_share, &receiver_share)?;
        self.swap.joint_key = jk;
        let (reveal_tip, follow_tip) = match side {
            Side::SellMds => (ctx.mwb_tip, ctx.mds_tip),
            Side::SellMdw => (ctx.mds_tip, ctx.mwb_tip),
        };
        check_mwb_timings(reveal_tip, o.reveal_height, follow_tip, o.follow_height)?;
        ensure!(
            o.reveal_height - reveal_tip <= MAX_REVEAL_BLOCKS,
            "the offer would lock our coins for {} blocks; the most we accept is {MAX_REVEAL_BLOCKS}",
            o.reveal_height - reveal_tip
        );
        ensure!(
            reveal_tip + OFFER_SLACK <= taker_lock_deadline(o.reveal_height),
            "the offer leaves too little time to lock"
        );
        self.swap.reveal_height = o.reveal_height;
        self.swap.follow_height = o.follow_height;
        let coins = decompose_value(self.swap.mds_amount).len();
        match (side, o.link, o.funder) {
            (Side::SellMds, OfferLink::Proof(proof), None) => {
                ensure!(proof.opened.len() == K_OR, "the link proof has {} instances, not {K_OR}", proof.opened.len());
                let link = verify_proof(&swap_id, &proof)?;
                ensure!(o.mds_salts.len() == coins, "the lock's salts do not match its {coins} coins");
                self.swap.t_point = link.t_point;
                self.swap.link = LinkRecord::of(&link);
                self.swap.mds_salts = o.mds_salts;
                // OR lock: we claim, the maker refunds at the follow height.
                self.swap.mds_script = link.or_lock(&mds_pk(&self.swap.mds_key), o.follow_height, &o.mds_pk);
                self.fund(ctx)?;
                Ok(Msg::FunderCommit(FunderCommit { view: self.swap.view.clone(), commits: self.ours.commits() }))
            }
            (Side::SellMdw, OfferLink::Commitment(c), Some(f)) => {
                ensure!(c.a_points.len() == K_AND, "the link commitment has {} instances, not {K_AND}", c.a_points.len());
                point(&c.t_point, "the adaptor point")?;
                f.view.check(&jk)?;
                ensure!(f.view.value == self.swap.mdw_amount, "the maker's lock holds {} MDW, not {}", f.view.value, self.swap.mdw_amount);
                self.swap.view = f.view.clone();
                self.swap.t_point = c.t_point;
                self.their_commits = Some(f.commits);
                let challenge = Challenge::random(K_AND);
                self.swap.mds_salts = salts(self.swap.mds_amount);
                self.claim_draft = Some(JointSpend::new(&f.view, &self.share, &self.their_share, &ctx.keys.address(), min_fee(1, 1, 1), 0)?);
                self.commitment = Some(c);
                self.challenge = Some(challenge.clone());
                Ok(Msg::ReceiverNonces(ReceiverNonces {
                    nonces: self.ours.points,
                    challenge: Some(challenge.0),
                    mds_salts: self.swap.mds_salts.clone(),
                }))
            }
            _ => bail!("the offer does not fit the order"),
        }
    }

    // ── Receiver ─────────────────────────────────────────────────────────

    /// Maker selling MDS: the taker's lock and nonce commitments.
    fn on_funder_commit(&mut self, f: FunderCommit, ctx: &mut Ctx) -> Result<Msg> {
        f.view.check(&self.swap.joint_key)?;
        ensure!(f.view.value == self.swap.mdw_amount, "the taker's lock holds {} MDW, not {}", f.view.value, self.swap.mdw_amount);
        self.swap.view = f.view.clone();
        self.their_commits = Some(f.commits);
        self.claim_draft = Some(JointSpend::new(&f.view, &self.share, &self.their_share, &ctx.keys.address(), min_fee(1, 1, 1), 0)?);
        Ok(Msg::ReceiverNonces(ReceiverNonces { nonces: self.ours.points, challenge: None, mds_salts: Vec::new() }))
    }

    /// The funder's nonces (checked against its commitments), its input
    /// partial and refund plan: we co-sign the input and the refund, and
    /// build and pre-sign the claim.
    fn on_funder_nonces(&mut self, f: FunderNonces) -> Result<Msg> {
        let commits = self.their_commits.ok_or_else(|| anyhow!("no nonce commitments to check against"))?;
        for (i, p) in f.nonces.iter().enumerate() {
            ensure!(crate::mwb::nonce_commitment(p) == commits[i], "the funder's nonce {i} is not the one it committed to");
            point(p, "a nonce")?;
        }
        let fp = f.nonces;
        if self.swap.side == Side::SellMdw {
            // Our challenge, answered.
            let (c, ch) = (self.commitment.as_ref().unwrap(), self.challenge.as_ref().unwrap());
            let opening = f.opening.ok_or_else(|| anyhow!("the maker did not open the link"))?;
            let link = verify(c, ch, &opening)?;
            self.swap.link = LinkRecord::of(&link);
            // AND lock: the maker claims, we refund at the reveal height.
            self.swap.mds_script = link.and_lock(&self.their_mds_pk, self.swap.reveal_height, &crate::mwb_trade::mds_pk(&self.swap.mds_key))?;
        }
        let input = InputSigning::new(&self.swap.view.commitment, &self.swap.joint_key, &fp[0], &self.ours.points[0]);
        let sf = scalar(&f.input_partial, "the funder's input partial")?;
        ensure!(input.verify_partial(&sf, &fp[0], &self.their_share)?, "the funder's input partial does not verify");
        let sr = input.partial(self.ours.take(0)?, &self.share)?;
        let input_sig = input.combine(&sf, &sr)?;

        let rp = &f.refund_plan;
        ensure!(
            rp.r1 == fp[1] && rp.r2_builder == fp[2] && rp.r2_other == self.ours.points[1],
            "the refund plan's nonces are not the ones exchanged"
        );
        let refund_partial = rp.sign_as_other(self.ours.take(1)?, &self.share, mwb_refund_height(&self.swap), None)?;

        let claim = self.claim_draft.as_ref().ok_or_else(|| anyhow!("no claim draft"))?;
        let cplan = claim.plan(&self.ours.points[2], &self.ours.points[3], &fp[3], Some(self.swap.t_point));
        let s1 = claim.sign_pedersen(&cplan, self.ours.take(2)?)?;
        let s2r = claim.sign_owner(&cplan, self.ours.take(3)?)?;
        self.swap.claim_tx = Some(encode_mwb(&claim.skeleton(&input_sig, &cplan))?);
        self.swap.claim_plan = cplan.clone();
        self.swap.claim_s1 = Some(s1.to_bytes());
        self.claim_s2_ours = Some(s2r);
        self.theirs = Some(fp);
        Ok(Msg::ReceiverSigs(ReceiverSigs {
            input_partial: sr.to_bytes(),
            refund_partial: refund_partial.to_bytes(),
            claim_plan: cplan,
            // Selling MDS the funder is the taker, who must read the secret
            // off our claim: it needs our half of the owner pre-signature.
            claim_owner_partial: (self.swap.side == Side::SellMds).then(|| s2r.to_bytes()),
        }))
    }

    /// The funder's claim partial: our pre-signature is complete.
    fn on_funder_claim_sig(&mut self, s2f: &[u8; 32]) -> Result<()> {
        let s2f = scalar(s2f, "the funder's claim partial")?;
        let plan = &self.swap.claim_plan;
        let funder = point(&self.their_share, "the funder's key share")?;
        ensure!(plan.verify_owner_partial(&s2f, &plan.r2_other, &funder)?, "the funder's claim partial does not verify");
        let s2_pre = self.claim_s2_ours.ok_or_else(|| anyhow!("no claim partial of ours"))? + s2f;
        let s1 = scalar(self.swap.claim_s1.as_ref().ok_or_else(|| anyhow!("no claim s1"))?, "claim s1")?;
        ensure!(plan.verify_presig(&s1, &s2_pre)?, "the claim pre-signature does not verify");
        self.swap.claim_s2_pre = Some(s2_pre.to_bytes());
        self.swap.phase = match self.swap.role {
            Role::Maker => Phase::LockingMine,
            Role::Taker => Phase::WaitingTheirLock,
        };
        Ok(())
    }

    // ── Funder ───────────────────────────────────────────────────────────

    /// The receiver's nonces: we sign our half of the input and plan the
    /// refund.
    fn on_receiver_nonces(&mut self, r: ReceiverNonces) -> Result<Msg> {
        for p in &r.nonces {
            point(p, "a nonce")?;
        }
        let mut opening = None;
        if self.swap.side == Side::SellMdw {
            let bits = r.challenge.ok_or_else(|| anyhow!("no challenge to the link"))?;
            ensure!(bits.len() == K_AND, "the challenge has {} bits, not {K_AND}", bits.len());
            let coins = decompose_value(self.swap.mds_amount).len();
            ensure!(r.mds_salts.len() == coins, "the lock's salts do not match its {coins} coins");
            let secret = self.secret.as_ref().ok_or_else(|| anyhow!("no link secret"))?;
            let ch = Challenge(bits);
            let o = secret.open(&ch);
            let link = verify(self.commitment.as_ref().unwrap(), &ch, &o)?;
            self.swap.link = LinkRecord::of(&link);
            self.swap.mds_salts = r.mds_salts;
            // AND lock: we claim, the taker refunds at the reveal height.
            self.swap.mds_script = link.and_lock(&crate::mwb_trade::mds_pk(&self.swap.mds_key), self.swap.reveal_height, &self.their_mds_pk)?;
            opening = Some(o);
        }
        let input = InputSigning::new(&self.swap.view.commitment, &self.swap.joint_key, &self.ours.points[0], &r.nonces[0]);
        let sf = input.partial(self.ours.take(0)?, &self.share)?;
        let draft = self.refund_draft.as_ref().ok_or_else(|| anyhow!("no refund draft"))?;
        let plan = draft.plan(&self.ours.points[1], &self.ours.points[2], &r.nonces[1], None);
        self.input = Some(input);
        self.input_partial = Some(sf);
        self.refund_plan = Some(plan.clone());
        self.theirs = Some(r.nonces);
        Ok(Msg::FunderNonces(FunderNonces { nonces: self.ours.points, input_partial: sf.to_bytes(), refund_plan: plan, opening }))
    }

    /// The receiver's partials and claim plan: we complete the refund, check
    /// the claim carries `T`, and co-sign it.
    fn on_receiver_sigs(&mut self, r: ReceiverSigs) -> Result<Msg> {
        let theirs = self.theirs.ok_or_else(|| anyhow!("no receiver nonces"))?;
        let receiver = point(&self.their_share, "the receiver's key share")?;
        let input = self.input.as_ref().ok_or_else(|| anyhow!("no input session"))?;
        let sr = scalar(&r.input_partial, "the receiver's input partial")?;
        ensure!(input.verify_partial(&sr, &theirs[0], &self.their_share)?, "the receiver's input partial does not verify");
        let input_sig = input.combine(&self.input_partial.unwrap(), &sr)?;

        let plan = self.refund_plan.as_ref().unwrap();
        let s2o = scalar(&r.refund_partial, "the receiver's refund partial")?;
        ensure!(plan.verify_owner_partial(&s2o, &plan.r2_other, &receiver)?, "the receiver's refund partial does not verify");
        let draft = self.refund_draft.as_ref().unwrap();
        let s2 = draft.sign_owner(plan, self.ours.take(2)?)? + s2o;
        let s1 = draft.sign_pedersen(plan, self.ours.take(1)?)?;
        let refund = draft.assemble(&input_sig, plan, plan.complete(&s1, &s2, None)?)?;
        ensure!(refund.lock_height() == mwb_refund_height(&self.swap), "the refund is not held to its height");

        let cp = &r.claim_plan;
        ensure!(cp.adaptor == Some(self.swap.t_point), "the claim does not carry the secret's point");
        ensure!(cp.min_height == 0, "the claim is time-locked");
        ensure!(
            cp.r1 == theirs[2] && cp.r2_builder == theirs[3] && cp.r2_other == self.ours.points[3],
            "the claim plan's nonces are not the ones exchanged"
        );
        let s2f = cp.sign_as_other(self.ours.take(3)?, &self.share, 0, Some(&self.swap.t_point))?;
        if self.swap.side == Side::SellMds {
            // We are the taker: the maker's half of the owner pre-signature is
            // what reads the secret off its claim.
            let s2r = scalar(&r.claim_owner_partial.ok_or_else(|| anyhow!("no claim partial from the maker"))?, "the maker's claim partial")?;
            ensure!(
                cp.verify_owner_partial(&s2r, &cp.r2_builder, &cp.builder_owner_excess(&self.share.public)?)?,
                "the maker's claim partial does not verify"
            );
            self.swap.claim_s2_pre = Some((s2r + s2f).to_bytes());
        }
        self.swap.claim_plan = cp.clone();
        self.swap.refund_tx = Some(encode_mwb(&refund)?);
        self.swap.phase = match self.swap.role {
            Role::Maker => Phase::LockingMine,
            Role::Taker => Phase::WaitingTheirLock,
        };
        Ok(Msg::FunderClaimSig { claim_partial: s2f.to_bytes() })
    }
}

/// The message that tells the other side we are stopping, best effort.
pub fn abort_messages(n: &Negotiation, reason: &str) -> Vec<Outgoing> {
    let msg = Msg::Abort { reason: crate::mwb_wire::reason(reason) };
    session_messages(&n.swap.session, our_to(n.swap.role), n.expect.saturating_sub(1).max(1), &msg).unwrap_or_default()
}

/// After its last message the funder keeps the session open for a while, in
/// case that message was lost and the receiver asks again.
pub struct Linger {
    pub swap_id: [u8; 32],
    pub session: Session,
    pub role: Role,
    /// The receiver's message whose repeat we answer.
    pub answers: u8,
    pub last_out: Vec<Outgoing>,
    pub until: u64,
}

impl Linger {
    pub fn of(n: &Negotiation, now: u64) -> Self {
        Self {
            swap_id: n.swap.id,
            session: n.swap.session.clone(),
            role: n.swap.role,
            answers: final_seq(n.swap.side) - 1,
            last_out: n.last_out.clone(),
            until: now + NEGOTIATION_SECS * 2,
        }
    }
}
