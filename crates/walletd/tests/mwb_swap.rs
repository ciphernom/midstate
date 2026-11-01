//! MDS <-> MWB swaps end to end, against both chains' consensus code.
//!
//! Midstate spends go through `apply_transaction` (script VM, coins,
//! denominations, the commitment and its TTL) after the mempool's fee floor;
//! the Commit is recorded as mined rather than mined with its spam PoW.
//! Midwimble transactions pass the checks its mempool makes before admitting
//! one (`validate`, the fee floor, the lock height, `apply_body`) and are then
//! applied with `apply_body`; test coins come from a coinbase whose reward sum
//! `apply_body` does not check (the block validator does).

use anyhow::{bail, ensure, Result};
use curve25519_dalek::scalar::Scalar;
use midstate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use midstate::core::transaction::apply_transaction;
use midstate::core::types::{
    compute_address, compute_coin_id, compute_commitment, is_v2_at, InputReveal, OutputData, Predicate,
    State as MdsState, Transaction as MdsTx, Witness,
};
use midstate::core::wots;
use midstate_walletd::mwb::{
    self, build_funding, joint_key, nonce_commitment, verify_share, InputSigning, JointOutput, JointSpend,
    JointView, KernelPlan, OwnerShare, SecretNonce,
};
use midstate_walletd::mwb_link::{self, Challenge, LinkSecret, K_AND, K_OR};
use midstate_walletd::swap::{may_reveal, plan_mwb_timings, MWB_REVEAL_BLOCKS};
use midwimble::core::mw::builder::{build_coinbase, Spendable};
use midwimble::core::mw::crypto::{decompress, Point32, SchnorrSig};
use midwimble::core::mw::stealth::{scan_output, WalletKeys};
use midwimble::core::mw::transaction::{Context, Transaction as MwbTx};
use midwimble::core::state::apply_body;
use midwimble::core::types::{State as MwbState, COINBASE_MATURITY, MIN_FEE_PER_WEIGHT};

const MDS_AMOUNT: u64 = 1 << 20;
const MDS_FEE: u64 = 64;
const MWB_AMOUNT: u64 = 25_000_000;

// ── midstate ────────────────────────────────────────────────────────────────

struct Mds {
    state: MdsState,
}

impl Mds {
    fn at_height(height: u64) -> Self {
        Self {
            state: MdsState {
                midstate: [0; 32],
                coins: UtxoAccumulator::new(),
                commitments: UtxoAccumulator::new(),
                depth: 0,
                target: [0xff; 32],
                height,
                timestamp: 0,
                commitment_heights: Default::default(),
                expirations: Default::default(),
                chain_mmr: MerkleMountainRange::new(),
                header_hash: [0; 32],
                burned_wots: UtxoAccumulator::new(),
            },
        }
    }

    fn fund(&mut self, address: [u8; 32], value: u64) -> [u8; 32] {
        let salt: [u8; 32] = rand::random();
        let v2 = is_v2_at(self.state.height);
        assert!(self.state.coins.insert(compute_coin_id(&address, value, &salt), v2));
        salt
    }

    fn has_coin(&self, address: &[u8; 32], value: u64, salt: &[u8; 32]) -> bool {
        self.state.coins.contains(&compute_coin_id(address, value, salt))
    }

    /// Phase 1, recorded as mined at the current height, and the reveal on it.
    fn commit(
        &mut self,
        inputs: Vec<InputReveal>,
        outputs: Vec<OutputData>,
        witnesses: impl FnOnce(&[u8; 32]) -> Vec<Vec<Vec<u8>>>,
    ) -> MdsTx {
        let salt: [u8; 32] = rand::random();
        let ins: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let outs: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(&ins, &outs, &salt);
        let v2 = is_v2_at(self.state.height);
        self.state.commitments.insert(commitment, v2);
        self.state.commitment_heights.insert(commitment, self.state.height);
        MdsTx::Reveal {
            inputs,
            witnesses: witnesses(&commitment).into_iter().map(Witness::ScriptInputs).collect(),
            outputs,
            salt,
        }
    }

    /// Phase 2: the mempool's fee floor, then consensus.
    fn reveal(&mut self, tx: &MdsTx) -> Result<()> {
        let bytes = bincode::serialized_size(tx)? as u128;
        if (tx.fee() as u128) * 1_024 < 10 * bytes {
            bail!("fee {} below the relay floor for {} bytes", tx.fee(), bytes);
        }
        let mut next = self.state.clone();
        apply_transaction(&mut next, tx)?;
        self.state = next;
        Ok(())
    }

    fn spend(
        &mut self,
        inputs: Vec<InputReveal>,
        outputs: Vec<OutputData>,
        witnesses: impl FnOnce(&[u8; 32]) -> Vec<Vec<Vec<u8>>>,
    ) -> Result<MdsTx> {
        let tx = self.commit(inputs, outputs, witnesses);
        self.reveal(&tx)?;
        Ok(tx)
    }
}

fn script_input(script: &[u8], value: u64, salt: [u8; 32]) -> InputReveal {
    InputReveal { predicate: Predicate::Script { bytecode: script.to_vec() }, value, salt, commitment: None }
}

fn payout(address: [u8; 32], value: u64) -> Vec<OutputData> {
    midstate::core::decompose_value(value)
        .into_iter()
        .map(|v| OutputData::Standard { address, value: v, salt: rand::random() })
        .collect()
}

fn wots_key() -> ([u8; 32], [u8; 32]) {
    let seed: [u8; 32] = rand::random();
    (seed, wots::keygen(&seed))
}

fn wots_sig(seed: &[u8; 32], commitment: &[u8; 32]) -> Vec<u8> {
    wots::sig_to_bytes(&wots::sign(seed, commitment))
}

fn lock_address(script: &[u8]) -> [u8; 32] {
    Predicate::Script { bytecode: script.to_vec() }.address()
}

// ── midwimble ───────────────────────────────────────────────────────────────

struct Mwb {
    state: MwbState,
}

impl Mwb {
    fn new() -> Self {
        let mut state = MwbState::genesis();
        state.height = 1_000;
        Self { state }
    }

    fn mint(&mut self, keys: &WalletKeys, value: u64) -> Spendable {
        let cb = build_coinbase(&[(keys.address(), value)]).unwrap();
        self.state = apply_body(&self.state, &MwbTx::empty(), Some(&cb), 0).unwrap();
        self.state.height += COINBASE_MATURITY + 1;
        let out = &cb.outputs.outputs[0];
        let owned = scan_output(keys, out).unwrap();
        Spendable { commitment: out.commitment, value, blinding: owned.blinding, owner_secret: owned.owner_secret }
    }

    /// Exactly the checks midwimble's mempool makes before admitting a tx.
    fn admit(&self, tx: &MwbTx) -> Result<()> {
        tx.validate(Context::Relay)?;
        let floor = tx.weight() * MIN_FEE_PER_WEIGHT;
        ensure!(tx.fee()? >= floor, "fee below relay minimum {floor}");
        ensure!(tx.lock_height() <= self.state.height, "time-locked until {}", tx.lock_height());
        apply_body(&self.state, tx, None, 0)?;
        Ok(())
    }

    fn mine(&mut self, tx: &MwbTx) -> Result<()> {
        self.admit(tx)?;
        self.state = apply_body(&self.state, tx, None, 0)?;
        self.state.height += 1;
        Ok(())
    }
}

fn received(keys: &WalletKeys, tx: &MwbTx) -> u64 {
    tx.outputs().filter_map(|o| scan_output(keys, o)).map(|o| o.value).sum()
}

// ── The midwimble negotiation ───────────────────────────────────────────────

struct Leg {
    view: JointView,
    funding: MwbTx,
    refund: MwbTx,
    claim: JointSpend,
    plan: KernelPlan,
    s1: Scalar,
    s2_pre: Scalar,
    /// The funder's copy, kept to read `t` off the chain.
    funder_s2_pre: Option<Scalar>,
    input_sig: SchnorrSig,
}

fn exchange(a: &SecretNonce, b: &SecretNonce) -> Result<()> {
    let (ca, cb) = (a.commitment(), b.commitment()); // sent first
    ensure!(nonce_commitment(&b.point) == cb && nonce_commitment(&a.point) == ca, "nonce does not match its commitment");
    Ok(())
}

/// A funder (who locks its MWB) and a receiver (who will claim it), each
/// checking what it receives.
#[allow(clippy::too_many_arguments)]
fn negotiate(
    swap_id: &[u8; 32],
    funder: &WalletKeys,
    coin: Spendable,
    receiver: &WalletKeys,
    refund_height: u64,
    adaptor: Point32,
    funder_extracts: bool,
) -> Result<Leg> {
    let (f, r) = (OwnerShare::random(), OwnerShare::random());
    ensure!(verify_share(&r.public, swap_id, &r.prove(swap_id)), "receiver's share unproven");
    ensure!(verify_share(&f.public, swap_id, &f.prove(swap_id)), "funder's share unproven");
    let jk = joint_key(&f.public, &r.public)?;

    let joint = JointOutput::new(MWB_AMOUNT, jk)?;
    let funding = build_funding(&[coin], &joint, &funder.address(), mwb::min_fee(1, 2, 1))?;
    let view = joint.view();
    view.check(&jk)?;
    ensure!(funding.outputs().any(|o| o.commitment == view.commitment && o.owner_key == jk), "no lock in funding");

    let (nf, nr) = (SecretNonce::on_g(), SecretNonce::on_g());
    exchange(&nf, &nr)?;
    let ins = InputSigning::new(&view.commitment, &jk, &nf.point, &nr.point);
    let (rf, rr) = (nf.point, nr.point);
    let (sf, sr) = (ins.partial(nf, &f)?, ins.partial(nr, &r)?);
    ensure!(ins.verify_partial(&sr, &rr, &r.public)? && ins.verify_partial(&sf, &rf, &f.public)?, "bad input partial");
    let input_sig = ins.combine(&sf, &sr)?;

    let draft = JointSpend::new(&view, &f, &r.public, &funder.address(), mwb::min_fee(1, 1, 1), refund_height)?;
    let (n1, nb, no) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
    exchange(&nb, &no)?;
    let plan = draft.plan(&n1.point, &nb.point, &no.point, None);
    let s2o = plan.sign_as_other(no, &r, refund_height, None)?;
    ensure!(plan.verify_owner_partial(&s2o, &plan.r2_other, &decompress(&r.public).unwrap())?, "bad refund partial");
    let s2 = draft.sign_owner(&plan, nb)? + s2o;
    let refund = draft.assemble(&input_sig, &plan, plan.complete(&draft.sign_pedersen(&plan, n1)?, &s2, None)?)?;

    let claim = JointSpend::new(&view, &r, &f.public, &receiver.address(), mwb::min_fee(1, 1, 1), 0)?;
    let (m1, mb, mo) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
    exchange(&mb, &mo)?;
    let cplan = claim.plan(&m1.point, &mb.point, &mo.point, Some(adaptor));
    let s2_f = cplan.sign_as_other(mo, &f, 0, Some(&adaptor))?;
    ensure!(cplan.verify_owner_partial(&s2_f, &cplan.r2_other, &decompress(&f.public).unwrap())?, "bad claim partial");
    let r2b = cplan.r2_builder;
    let s2_r = claim.sign_owner(&cplan, mb)?;
    let s1 = claim.sign_pedersen(&cplan, m1)?;
    ensure!(cplan.verify_presig(&s1, &(s2_r + s2_f))?, "pre-signature does not verify");
    let funder_s2_pre = if funder_extracts {
        let e_r = cplan.builder_owner_excess(&f.public)?;
        ensure!(cplan.verify_owner_partial(&s2_r, &r2b, &e_r)?, "bad claim partial");
        Some(s2_r + s2_f)
    } else {
        None
    };
    Ok(Leg { view, funding, refund, claim, plan: cplan, s1, s2_pre: s2_r + s2_f, funder_s2_pre, input_sig })
}

// ── The two directions ──────────────────────────────────────────────────────

/// Maker sells MDS: an OR lock on midstate; the secret comes out on midwimble.
#[test]
fn maker_sells_mds_for_mwb() -> Result<()> {
    let swap_id: [u8; 32] = rand::random();
    let (maker_mwb, taker_mwb) = (WalletKeys::random(), WalletKeys::random());
    let (_, maker_refund_pk) = wots_key();
    let (taker_claim_seed, taker_claim_pk) = wots_key();
    let mut mds = Mds::at_height(450_000);
    let mut mwb = Mwb::new();
    let taker_coin = mwb.mint(&taker_mwb, MWB_AMOUNT + 1_000_000);

    let secret = LinkSecret::generate(K_OR);
    let commitment = secret.commit();
    let opening = secret.open(&Challenge::fiat_shamir(&swap_id, &commitment)?);
    let link = mwb_link::verify(&commitment, &Challenge::fiat_shamir(&swap_id, &commitment)?, &opening)?;

    let t = plan_mwb_timings(mwb.state.height, mds.state.height, MWB_REVEAL_BLOCKS)?;
    let lock = link.or_lock(&taker_claim_pk, t.follow_height, &maker_refund_pk);
    let salt = mds.fund(lock_address(&lock), MDS_AMOUNT);

    let leg = negotiate(&swap_id, &taker_mwb, taker_coin, &maker_mwb, t.reveal_height, link.t_point, true)?;
    mwb.mine(&leg.funding)?;
    assert!(mwb.admit(&leg.refund).is_err(), "the refund waits for its height");

    assert!(may_reveal(mwb.state.height, t.reveal_height));
    let sig = leg.plan.complete(&leg.s1, &leg.s2_pre, Some(&secret.t()))?;
    let claim = leg.claim.assemble(&leg.input_sig, &leg.plan, sig)?;
    mwb.mine(&claim)?;
    assert_eq!(received(&maker_mwb, &claim), MWB_AMOUNT - mwb::min_fee(1, 1, 1));

    let kernel = claim
        .body
        .kernels
        .iter()
        .find(|k| k.excess == leg.plan.excess && k.owner_excess == leg.plan.owner_excess)
        .expect("the claim kernel is on chain");
    let read_t = mwb::extract_adaptor_secret(&leg.funder_s2_pre.unwrap(), &kernel.signature).unwrap();
    let (j, preimage) = link.leaf_for_t(&read_t).expect("t opens a leaf");
    mds.spend(
        vec![script_input(&lock, MDS_AMOUNT, salt)],
        payout(compute_address(&taker_claim_pk), MDS_AMOUNT - MDS_FEE),
        |c| vec![mwb_link::or_claim_witness(&wots_sig(&taker_claim_seed, c), &link.merkle_proof(j), &preimage)],
    )?;
    assert!(!mds.has_coin(&lock_address(&lock), MDS_AMOUNT, &salt));
    Ok(())
}

/// Maker sells MWB: an AND lock on midstate; the secret comes out on midstate.
#[test]
fn maker_sells_mwb_for_mds() -> Result<()> {
    let swap_id: [u8; 32] = rand::random();
    let (maker_mwb, taker_mwb) = (WalletKeys::random(), WalletKeys::random());
    let (maker_claim_seed, maker_claim_pk) = wots_key();
    let (_, taker_refund_pk) = wots_key();
    let mut mds = Mds::at_height(450_000);
    let mut mwb = Mwb::new();
    let maker_coin = mwb.mint(&maker_mwb, MWB_AMOUNT + 1_000_000);

    let secret = LinkSecret::generate(K_AND);
    let commitment = secret.commit();
    let challenge = Challenge::random(K_AND);
    let link = mwb_link::verify(&commitment, &challenge, &secret.open(&challenge))?;

    let t = plan_mwb_timings(mds.state.height, mwb.state.height, MWB_REVEAL_BLOCKS)?;
    let leg = negotiate(&swap_id, &maker_mwb, maker_coin, &taker_mwb, t.follow_height, link.t_point, false)?;
    mwb.mine(&leg.funding)?;
    assert!(mwb.state.utxos.contains_key(&leg.view.commitment), "taker: lock confirmed");

    let lock = link.and_lock(&maker_claim_pk, t.reveal_height, &taker_refund_pk)?;
    let salt = mds.fund(lock_address(&lock), MDS_AMOUNT);

    let unopened = secret.unopened(&challenge);
    let reveal = mds.spend(
        vec![script_input(&lock, MDS_AMOUNT, salt)],
        payout(compute_address(&maker_claim_pk), MDS_AMOUNT - MDS_FEE),
        |c| vec![mwb_link::and_claim_witness(&wots_sig(&maker_claim_seed, c), &unopened)],
    )?;

    let MdsTx::Reveal { witnesses, .. } = &reveal else { unreachable!() };
    let Witness::ScriptInputs(items) = &witnesses[0];
    let preimages = mwb_link::and_preimages_from_witness(items, K_AND).unwrap();
    let read_t = link.recover_t(&preimages).expect("one consistent instance gives t");

    let sig = leg.plan.complete(&leg.s1, &leg.s2_pre, Some(&read_t))?;
    let claim = leg.claim.assemble(&leg.input_sig, &leg.plan, sig)?;
    mwb.mine(&claim)?;
    assert_eq!(received(&taker_mwb, &claim), MWB_AMOUNT - mwb::min_fee(1, 1, 1));
    Ok(())
}

/// Nobody moves after the locks: each side gets its coins back, not a block
/// early, and with one signature per one-time key.
#[test]
fn both_refunds_unlock_at_their_heights_and_not_before() -> Result<()> {
    let swap_id: [u8; 32] = rand::random();
    let (maker_mwb, taker_mwb) = (WalletKeys::random(), WalletKeys::random());
    let (maker_refund_seed, maker_refund_pk) = wots_key();
    let (_, taker_claim_pk) = wots_key();
    let mut mds = Mds::at_height(450_000);
    let mut mwb = Mwb::new();
    let taker_coin = mwb.mint(&taker_mwb, MWB_AMOUNT + 1_000_000);

    let secret = LinkSecret::generate(K_OR);
    let c = secret.commit();
    let ch = Challenge::fiat_shamir(&swap_id, &c)?;
    let link = mwb_link::verify(&c, &ch, &secret.open(&ch))?;
    let t = plan_mwb_timings(mwb.state.height, mds.state.height, MWB_REVEAL_BLOCKS)?;

    let lock = link.or_lock(&taker_claim_pk, t.follow_height, &maker_refund_pk);
    let salt = mds.fund(lock_address(&lock), MDS_AMOUNT);
    let leg = negotiate(&swap_id, &taker_mwb, taker_coin, &maker_mwb, t.reveal_height, link.t_point, true)?;
    mwb.mine(&leg.funding)?;

    mwb.state.height = t.reveal_height - 1;
    let early = apply_body(&mwb.state, &leg.refund, None, 0);
    assert!(early.unwrap_err().to_string().contains("time-locked"));
    mwb.state.height = t.reveal_height;
    mwb.mine(&leg.refund)?;
    assert_eq!(received(&taker_mwb, &leg.refund), MWB_AMOUNT - mwb::min_fee(1, 1, 1));

    mds.state.height = t.follow_height - 1;
    let refund = mds.commit(
        vec![script_input(&lock, MDS_AMOUNT, salt)],
        payout(compute_address(&maker_refund_pk), MDS_AMOUNT - MDS_FEE),
        |c| vec![mwb_link::refund_witness(&wots_sig(&maker_refund_seed, c))],
    );
    assert!(mds.reveal(&refund).is_err(), "midstate refund one block early");
    mds.state.height = t.follow_height;
    mds.reveal(&refund)?;
    Ok(())
}
