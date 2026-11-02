//! The midwimble side of the cross-chain DEX: the account, and the two-party
//! lock an MDS <-> MWB swap puts MWB in.
//!
//! # The account
//!
//! Derived the way midwimble's own wallets derive it, from the raw BIP39 seed
//! (`WalletKeys::from_seed`, with the post-quantum recovery key at
//! `DEFAULT_RECOVERY_HEIGHT`), for the reason `evm.rs` follows BIP44: the
//! same recovery phrase then restores this account in midwimble's CLI and
//! browser wallets. Only the derived keys are stored (`WalletData::mwb_keys`),
//! never the phrase or the seed, which would also open every other account
//! the phrase controls.
//!
//! # The lock
//!
//! Midwimble spends are authorised by the **owner key**, not the blinding
//! factor: an input must carry the recorded owner key and sign for it, and
//! the kernel must prove knowledge of the owner excess
//! (`Σ Ko_in = Σ E' + x·G`). So a 2-of-2 lock needs only a joint owner key
//! `Ko_J = Ko_F + Ko_R`. The funder picks the blinding factor alone, proves
//! the range alone, and hands the opening to the receiver, which grants
//! nothing. No multi-party Bulletproof.
//!
//! An input signs `H(magic, C, Ko)` and nothing else, so one input signature
//! serves every spend of the lock. The kernel is the only signature bound to
//! one spend, and its Pedersen half is the builder's alone, so the adaptor
//! shifts the kernel's owner nonce:
//!
//! ```text
//!   R2 = R2_builder + R2_other + T,      T = t·G
//!   c  = kernel_challenge(E, E', R1, R2, m)
//!   s2 = s2_builder + s2_other + t                     (final)
//!   t  = s2 − (s2_builder + s2_other)                  (extraction)
//! ```
//!
//! Only someone holding the pre-signature can extract `t`; the published
//! kernel is an ordinary kernel and links to nothing.
//!
//! In a spend of the lock, the **builder** pays itself and knows the whole
//! Pedersen excess and its owner share `ko_b − x`; the **other** party signs
//! with its raw share `ko_o`. The funder builds the refund (with
//! `min_height`); the receiver builds the claim (with the adaptor).

use anyhow::{anyhow, bail, ensure, Result};
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::IsIdentity;
use midwimble::core::mw::builder::Spendable;
use midwimble::core::mw::crypto::{
    commit, compress, decompress, gen_h, kernel_challenge, random_scalar, scalar_from_bytes,
    schnorr_challenge, schnorr_sign, schnorr_verify, KernelSig, Point32, SchnorrSig,
};
use midwimble::core::mw::stealth::{new_output, NewOutput, StealthAddress, WalletKeys, PAYLOAD_LEN};
use midwimble::core::mw::transaction::{
    Context, Input, Kernel, KernelFeatures, Output, OutputGroup, Transaction, TxBody,
};
use midwimble::core::recovery::{recovery_public_key, DEFAULT_RECOVERY_HEIGHT};
use midwimble::core::types::{hash_domain, INPUT_WEIGHT, KERNEL_WEIGHT, MIN_FEE_PER_WEIGHT, OUTPUT_WEIGHT};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

// ── The account ─────────────────────────────────────────────────────────────

/// The account midwimble's own wallets show for this phrase.
///
/// # Reasoning
/// A swap's MWB must land in an account the user can recover with the words
/// they already hold. Deriving it exactly as midwimble's CLI and browser
/// wallet do means the same phrase restores it there, which is why `evm.rs`
/// follows BIP44 for the Base account. A walletd-only derivation would leave
/// funds recoverable by this software alone.
///
/// # Formal Specification
///
/// ```text
/// Pre:  phrase is a BIP39 mnemonic (any case, surrounding whitespace)
/// Post: Ok(k)  ⇒ k = WalletKeys::from_seed(seed, R), where
///                  seed = BIP39 seed of phrase, passphrase ""
///                  R    = recovery key of seed at DEFAULT_RECOVERY_HEIGHT
///                ⇒ k.address() = the address midwimble's wallet restores
///       Err(_) ⇒ nothing derived
/// ```
///
/// # Safety / Invariants
/// - The BIP39 seed is zeroized before return; only the derived keys leave,
///   and only those are stored (`WalletData::mwb_keys`).
/// - Costs one MSS tree of `DEFAULT_RECOVERY_HEIGHT` (1,024 WOTS key
///   generations), the same as each MSS key `restore` generates.
pub fn keys_from_mnemonic(phrase: &str) -> Result<WalletKeys> {
    keys_from_mnemonic_at(phrase, DEFAULT_RECOVERY_HEIGHT)
}

fn keys_from_mnemonic_at(phrase: &str, recovery_height: u32) -> Result<WalletKeys> {
    let m = bip39::Mnemonic::parse_normalized(&phrase.trim().to_lowercase())
        .map_err(|e| anyhow!("invalid recovery phrase: {e}"))?;
    let mut seed = m.to_seed("");
    let keys = recovery_public_key(&seed, recovery_height)
        .map(|recovery| WalletKeys::from_seed(&seed, recovery));
    seed.zeroize();
    keys
}

/// The relay fee floor for a transaction of this shape (midwimble's mempool).
pub fn min_fee(inputs: u64, outputs: u64, kernels: u64) -> u64 {
    (inputs * INPUT_WEIGHT + outputs * OUTPUT_WEIGHT + kernels * KERNEL_WEIGHT) * MIN_FEE_PER_WEIGHT
}

/// Whether `s` is a midwimble address (`mw1…`), and why not if it is not:
/// for a send form to say before anything is built.
pub fn check_address(s: &str) -> Result<()> {
    StealthAddress::decode(s.trim()).map(|_| ())
}

fn point(p: &Point32, what: &str) -> Result<RistrettoPoint> {
    match decompress(p) {
        Some(pt) if !pt.is_identity() => Ok(pt),
        _ => bail!("{what} is not a valid non-identity point"),
    }
}

// ── Owner-key shares ────────────────────────────────────────────────────────

/// One party's share of a lock's owner key. Fresh for every swap.
pub struct OwnerShare {
    secret: Scalar,
    pub public: Point32,
}

impl OwnerShare {
    pub fn random() -> Self {
        let secret = random_scalar();
        Self { secret, public: compress(&RistrettoPoint::mul_base(&secret)) }
    }

    /// Proof of knowledge of the share, bound to the swap. Without it a party
    /// could announce `X − Ko_other` for an `X` it controls and own the
    /// "joint" key outright (a rogue-key attack).
    pub fn prove(&self, swap_id: &[u8; 32]) -> SchnorrSig {
        schnorr_sign(&self.secret, &pok_message(swap_id, &self.public))
    }

    /// The secret, for the swap journal: a swap must survive a restart, and
    /// its record is written before anything it authorises is sent.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }

    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        let secret = scalar_from_bytes(bytes)?;
        Some(Self { secret, public: compress(&RistrettoPoint::mul_base(&secret)) })
    }
}

fn pok_message(swap_id: &[u8; 32], public: &Point32) -> [u8; 32] {
    hash_domain(b"midstate.mwb-swap.owner-pok.v1", &[swap_id, public])
}

/// Checks a counterparty's share before any key is combined with it.
pub fn verify_share(public: &Point32, swap_id: &[u8; 32], pok: &SchnorrSig) -> bool {
    schnorr_verify(public, &pok_message(swap_id, public), pok)
}

pub fn joint_key(a: &Point32, b: &Point32) -> Result<Point32> {
    let sum = point(a, "owner share")? + point(b, "owner share")?;
    ensure!(!sum.is_identity(), "owner shares cancel");
    Ok(compress(&sum))
}

// ── Nonces ──────────────────────────────────────────────────────────────────

/// A one-shot signing nonce. Not `Clone`: every partial signature consumes
/// its nonce, so one nonce can never answer two challenges.
pub struct SecretNonce {
    k: Scalar,
    pub point: Point32,
}

impl SecretNonce {
    /// For owner and input signatures (generator `G`).
    pub fn on_g() -> Self {
        let k = random_scalar();
        Self { k, point: compress(&RistrettoPoint::mul_base(&k)) }
    }

    /// For the Pedersen half of a kernel (generator `H`).
    pub fn on_h() -> Self {
        let k = random_scalar();
        Self { k, point: compress(&(k * gen_h())) }
    }

    /// Sent first; the point follows only once both commitments are in, so
    /// neither side can pick its nonce after seeing the other's.
    pub fn commitment(&self) -> [u8; 32] {
        nonce_commitment(&self.point)
    }
}

pub fn nonce_commitment(point: &Point32) -> [u8; 32] {
    hash_domain(b"midstate.mwb-swap.nonce-commit.v1", &[point])
}

// ── The lock output ─────────────────────────────────────────────────────────

/// The lock and its opening (the funder's; shared with the receiver).
pub struct JointOutput {
    pub output: Output,
    pub value: u64,
    pub blinding: Scalar,
}

/// What the receiver needs to build its claim, and checks before relying on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointView {
    pub commitment: Point32,
    pub owner_key: Point32,
    pub value: u64,
    pub blinding: [u8; 32],
}

impl JointOutput {
    /// A 2-of-2 output. Consensus checks only encodings here, so the stealth
    /// fields are random: the lock looks like any other output on chain.
    pub fn new(value: u64, owner_key: Point32) -> Result<Self> {
        point(&owner_key, "joint owner key")?;
        let blinding = random_scalar();
        let mut payload = vec![0u8; PAYLOAD_LEN];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut payload);
        let output = Output {
            commitment: compress(&commit(value, &blinding)),
            owner_key,
            ephemeral_key: compress(&RistrettoPoint::mul_base(&random_scalar())),
            view_tag: rand::random(),
            payload,
            // A dormant post-quantum claim needs one recipient's MSS key; a
            // lock has two owners and lives for hours, so it carries none.
            recovery_commitment: rand::random(),
        };
        Ok(Self { output, value, blinding })
    }

    pub fn view(&self) -> JointView {
        JointView {
            commitment: self.output.commitment,
            owner_key: self.output.owner_key,
            value: self.value,
            blinding: self.blinding.to_bytes(),
        }
    }
}

impl JointView {
    /// The receiver's check: the opening matches, and the owner key is the
    /// joint key it agreed to, so its own share is needed to spend.
    pub fn check(&self, expected_owner_key: &Point32) -> Result<Scalar> {
        let r = scalar_from_bytes(&self.blinding).ok_or_else(|| anyhow!("blinding is not canonical"))?;
        ensure!(compress(&commit(self.value, &r)) == self.commitment, "opening does not match");
        ensure!(&self.owner_key == expected_owner_key, "lock is not on the agreed joint key");
        Ok(r)
    }
}

/// The funder moves its own coins into the lock (plus change). An ordinary
/// single-party transaction, built as midwimble's `build_transaction` does.
pub fn build_funding(
    spend: &[Spendable],
    joint: &JointOutput,
    change_to: &StealthAddress,
    fee: u64,
) -> Result<Transaction> {
    ensure!(!spend.is_empty(), "no inputs");
    let total_in = spend
        .iter()
        .try_fold(0u64, |a, s| a.checked_add(s.value))
        .ok_or_else(|| anyhow!("input value overflow"))?;
    let need = joint.value.checked_add(fee).ok_or_else(|| anyhow!("output value overflow"))?;
    ensure!(total_in >= need, "insufficient funds: have {total_in}, need {need}");

    let mut outs: Vec<(Output, u64, Scalar)> = vec![(joint.output.clone(), joint.value, joint.blinding)];
    if total_in > need {
        let c = new_output(change_to, total_in - need)?;
        outs.push((c.output, c.value, c.blinding));
    }
    // Position inside the group must not reveal which output is the lock.
    outs.shuffle(&mut rand::thread_rng());
    let group = OutputGroup::prove(
        outs.iter().map(|o| o.0.clone()).collect(),
        &outs.iter().map(|o| o.1).collect::<Vec<_>>(),
        &outs.iter().map(|o| o.2).collect::<Vec<_>>(),
    )?;
    let mut inputs: Vec<Input> = spend.iter().map(|s| Input::sign(s.commitment, &s.owner_secret)).collect();
    inputs.sort_by(|a, b| a.commitment.cmp(&b.commitment));

    let sum_out: Scalar = outs.iter().map(|o| o.2).sum();
    let sum_in: Scalar = spend.iter().map(|s| s.blinding).sum();
    let sum_owner: Scalar = spend.iter().map(|s| s.owner_secret).sum();
    let (o, x, e, e_owner) = loop {
        let (o, x) = (random_scalar(), random_scalar());
        let (e, e_owner) = (sum_out - sum_in - o, sum_owner - x);
        if e != Scalar::ZERO && e_owner != Scalar::ZERO {
            break (o, x, e, e_owner);
        }
    };
    let tx = Transaction {
        body: TxBody {
            inputs,
            outputs: vec![group],
            kernels: vec![Kernel::new(KernelFeatures::Plain, fee, 0, &e, &e_owner)],
        },
        kernel_offset: o.to_bytes(),
        owner_offset: x.to_bytes(),
    };
    tx.validate(Context::Relay)?;
    Ok(tx)
}

// ── The lock's input signature ──────────────────────────────────────────────

/// Two-party Schnorr over `Input::message(C_J, Ko_J)`. Made once; both
/// parties keep it, and it serves the claim and the refund alike.
pub struct InputSigning {
    pub joint_key: Point32,
    pub msg: [u8; 32],
    pub r: [Point32; 2],
}

impl InputSigning {
    pub fn new(joint_commitment: &Point32, joint_key: &Point32, r_a: &Point32, r_b: &Point32) -> Self {
        Self { joint_key: *joint_key, msg: Input::message(joint_commitment, joint_key), r: [*r_a, *r_b] }
    }

    fn challenge(&self) -> Result<Scalar> {
        let r = point(&self.r[0], "nonce")? + point(&self.r[1], "nonce")?;
        Ok(schnorr_challenge(&self.joint_key, &compress(&r), &self.msg))
    }

    pub fn partial(&self, nonce: SecretNonce, share: &OwnerShare) -> Result<Scalar> {
        ensure!(self.r.contains(&nonce.point), "nonce is not part of this session");
        Ok(nonce.k + self.challenge()? * share.secret)
    }

    pub fn verify_partial(&self, s: &Scalar, r_i: &Point32, share: &Point32) -> Result<bool> {
        let e = self.challenge()?;
        Ok(RistrettoPoint::mul_base(s) == point(r_i, "nonce")? + e * point(share, "share")?)
    }

    pub fn combine(&self, s_a: &Scalar, s_b: &Scalar) -> Result<SchnorrSig> {
        let sig = SchnorrSig { e: self.challenge()?.to_bytes(), s: (s_a + s_b).to_bytes() };
        ensure!(schnorr_verify(&self.joint_key, &self.msg, &sig), "combined input signature is invalid");
        Ok(sig)
    }
}

// ── Kernels ─────────────────────────────────────────────────────────────────

/// Everything both parties agree on before partial signatures are exchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelPlan {
    pub fee: u64,
    pub min_height: u64,
    /// `E = e·H`; the builder knows `e`.
    pub excess: Point32,
    /// `E' = (ko_b − x)·G + Ko_o`.
    pub owner_excess: Point32,
    /// The builder's Pedersen nonce (on `H`).
    pub r1: Point32,
    pub r2_builder: Point32,
    pub r2_other: Point32,
    /// `T`, for the claim. The refund has none.
    pub adaptor: Option<Point32>,
}

impl KernelPlan {
    pub fn message(&self) -> [u8; 32] {
        Kernel::message(KernelFeatures::Plain, self.fee, self.min_height)
    }

    fn r2(&self) -> Result<RistrettoPoint> {
        let mut r2 = point(&self.r2_builder, "nonce")? + point(&self.r2_other, "nonce")?;
        if let Some(t) = &self.adaptor {
            r2 += point(t, "adaptor point")?;
        }
        Ok(r2)
    }

    /// Midwimble's kernel challenge for this plan.
    pub fn challenge(&self) -> Result<Scalar> {
        Ok(kernel_challenge(&self.excess, &self.owner_excess, &self.r1, &compress(&self.r2()?), &self.message()))
    }

    /// The *other* party's partial over its raw share.
    ///
    /// # Reasoning
    /// This partial is the only authority the other party ever gives over the
    /// lock, so it states what it is signing and refuses any plan that
    /// differs: a refund without its lock height could be taken at once, and
    /// a claim without `T` could be completed without the secret, so without
    /// paying on midstate.
    ///
    /// Nothing else needs checking. Whatever `E'` the builder chose, it can
    /// only complete the kernel if it knows `log(E' − Ko_o)`, so the kernel
    /// can only balance an owner sum that contains our share, which is fresh
    /// for this swap: that means the lock. Where the builder pays itself and
    /// what fee it pays are its own business.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  nonce was committed to for r2_other, and is unused
    /// Post: Ok(s)  ⇒ s = k + c·ko_o, c = kernel_challenge(this plan)
    ///       Err(_) ⇔ min_height ≠ expect_min_height
    ///                ∨ adaptor ≠ expect_adaptor ∨ nonce.point ≠ r2_other
    ///       either way the nonce is consumed
    /// ```
    ///
    /// # Safety / Invariants
    /// - One nonce answers one challenge: `SecretNonce` is moved in, not
    ///   borrowed, and is not `Clone`.
    pub fn sign_as_other(
        &self,
        nonce: SecretNonce,
        share: &OwnerShare,
        expect_min_height: u64,
        expect_adaptor: Option<&Point32>,
    ) -> Result<Scalar> {
        ensure!(nonce.point == self.r2_other, "nonce is not the one committed for this kernel");
        ensure!(
            self.min_height == expect_min_height,
            "kernel min_height {} is not the agreed {}",
            self.min_height,
            expect_min_height
        );
        ensure!(self.adaptor.as_ref() == expect_adaptor, "kernel adaptor point is not the agreed one");
        Ok(nonce.k + self.challenge()? * share.secret)
    }

    /// Checks a partial owner signature against the signer's share of `E'`.
    pub fn verify_owner_partial(&self, s2: &Scalar, r2_i: &Point32, owner_excess_i: &RistrettoPoint) -> Result<bool> {
        let c = self.challenge()?;
        Ok(RistrettoPoint::mul_base(s2) == point(r2_i, "nonce")? + c * owner_excess_i)
    }

    /// The other party's view of the builder's share of `E'`.
    pub fn builder_owner_excess(&self, other_share: &Point32) -> Result<RistrettoPoint> {
        Ok(point(&self.owner_excess, "owner excess")? - point(other_share, "share")?)
    }

    /// Is `(s1, s2_pre)` a valid kernel signature once `t` is added?
    pub fn verify_presig(&self, s1: &Scalar, s2_pre: &Scalar) -> Result<bool> {
        let c = self.challenge()?;
        let r1 = s1 * gen_h() - c * point(&self.excess, "excess")?;
        let mut r2 = RistrettoPoint::mul_base(s2_pre) - c * point(&self.owner_excess, "owner excess")?;
        if let Some(t) = &self.adaptor {
            r2 += point(t, "adaptor point")?;
        }
        Ok(compress(&r1) == self.r1 && r2 == self.r2()?)
    }

    /// The final signature. `t` must be given iff the plan has an adaptor.
    pub fn complete(&self, s1: &Scalar, s2_pre: &Scalar, t: Option<&Scalar>) -> Result<KernelSig> {
        ensure!(self.adaptor.is_some() == t.is_some(), "adaptor secret given for the wrong kind of kernel");
        let s2 = s2_pre + t.copied().unwrap_or(Scalar::ZERO);
        Ok(KernelSig { c: self.challenge()?.to_bytes(), s1: s1.to_bytes(), s2: s2.to_bytes() })
    }
}

/// `t` from a published kernel, given the pre-signature's owner component.
///
/// # Formal Specification
///
/// ```text
/// Pre:  published is the on-chain signature of the kernel s2_pre was made
///       for (found by its excess pair; its c equals the plan's challenge)
/// Post: Some(t) ⇒ t = published.s2 − s2_pre, and t·G = T for that kernel
///       None    ⇔ published.s2 is not a canonical scalar
/// ```
///
/// The caller checks `t·G = T` before relying on `t`; nothing else can make
/// it fail for the agreed kernel, since only that pre-signature completes it.
pub fn extract_adaptor_secret(s2_pre: &Scalar, published: &KernelSig) -> Option<Scalar> {
    Some(scalar_from_bytes(&published.s2)? - s2_pre)
}

// ── Spending the lock ───────────────────────────────────────────────────────

/// The builder's half of a spend of the lock: one input, one output paying
/// the builder, one kernel.
pub struct JointSpend {
    pub view: JointView,
    pub output: NewOutput,
    group: OutputGroup,
    pub fee: u64,
    pub min_height: u64,
    kernel_offset: Scalar,
    owner_offset: Scalar,
    excess_secret: Scalar,
    owner_secret: Scalar,
    pub excess: Point32,
    pub owner_excess: Point32,
}

impl JointSpend {
    pub fn new(
        view: &JointView,
        builder: &OwnerShare,
        other_share: &Point32,
        pay_to: &StealthAddress,
        fee: u64,
        min_height: u64,
    ) -> Result<Self> {
        let joint_blinding = view.check(&joint_key(&builder.public, other_share)?)?;
        ensure!(view.value > fee, "fee exceeds the locked value");
        let output = new_output(pay_to, view.value - fee)?;
        let group = OutputGroup::prove(vec![output.output.clone()], &[output.value], &[output.blinding])?;
        let other = point(other_share, "share")?;
        loop {
            let (o, x) = (random_scalar(), random_scalar());
            let e = output.blinding - joint_blinding - o;
            let e_owner_b = builder.secret - x;
            let owner_excess = RistrettoPoint::mul_base(&e_owner_b) + other;
            if e != Scalar::ZERO && !owner_excess.is_identity() {
                return Ok(Self {
                    view: view.clone(),
                    excess: compress(&(e * gen_h())),
                    owner_excess: compress(&owner_excess),
                    output,
                    group,
                    fee,
                    min_height,
                    kernel_offset: o,
                    owner_offset: x,
                    excess_secret: e,
                    owner_secret: e_owner_b,
                });
            }
        }
    }

    pub fn plan(&self, r1: &Point32, r2_builder: &Point32, r2_other: &Point32, adaptor: Option<Point32>) -> KernelPlan {
        KernelPlan {
            fee: self.fee,
            min_height: self.min_height,
            excess: self.excess,
            owner_excess: self.owner_excess,
            r1: *r1,
            r2_builder: *r2_builder,
            r2_other: *r2_other,
            adaptor,
        }
    }

    /// `s1`: the Pedersen half, the builder's alone.
    pub fn sign_pedersen(&self, plan: &KernelPlan, nonce: SecretNonce) -> Result<Scalar> {
        ensure!(nonce.point == plan.r1 && plan.excess == self.excess, "plan does not match this spend");
        Ok(nonce.k + plan.challenge()? * self.excess_secret)
    }

    /// `s2_b`: the builder's owner partial.
    pub fn sign_owner(&self, plan: &KernelPlan, nonce: SecretNonce) -> Result<Scalar> {
        ensure!(
            nonce.point == plan.r2_builder && plan.owner_excess == self.owner_excess,
            "plan does not match this spend"
        );
        Ok(nonce.k + plan.challenge()? * self.owner_secret)
    }

    pub fn assemble(&self, input_sig: &SchnorrSig, plan: &KernelPlan, sig: KernelSig) -> Result<Transaction> {
        finish_spend(self.skeleton(input_sig, plan), sig)
    }

    /// The whole spend but its kernel signature, for the journal: the
    /// receiver learns `t` (selling MDW) long after the negotiation, and the
    /// offsets and output this spend chose exist nowhere else. Holds no
    /// secret that spends: the kernel still needs the pre-signature and `t`.
    pub fn skeleton(&self, input_sig: &SchnorrSig, plan: &KernelPlan) -> Transaction {
        Transaction {
            body: TxBody {
                inputs: vec![Input {
                    commitment: self.view.commitment,
                    owner_key: self.view.owner_key,
                    signature: input_sig.clone(),
                }],
                outputs: vec![self.group.clone()],
                kernels: vec![Kernel {
                    features: KernelFeatures::Plain,
                    fee: plan.fee,
                    min_height: plan.min_height,
                    excess: plan.excess,
                    owner_excess: plan.owner_excess,
                    signature: KernelSig { c: [0; 32], s1: [0; 32], s2: [0; 32] },
                }],
            },
            kernel_offset: self.kernel_offset.to_bytes(),
            owner_offset: self.owner_offset.to_bytes(),
        }
    }
}

/// Puts the final kernel signature into a [`JointSpend::skeleton`] and
/// checks the result by midwimble's relay rules.
pub fn finish_spend(mut skeleton: Transaction, sig: KernelSig) -> Result<Transaction> {
    let kernel = skeleton.body.kernels.first_mut().ok_or_else(|| anyhow!("a spend without a kernel"))?;
    kernel.signature = sig;
    skeleton.validate(Context::Relay)?;
    Ok(skeleton)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midwimble::core::mw::stealth::scan_output;
    use midwimble::core::recovery::derive_recovery_keypair;

    const PHRASE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon abandon abandon art";

    /// The account is the one midwimble's own wallet derives from the phrase
    /// (its CLI: `to_seed("")`, `derive_recovery_keypair`, `from_seed`). Done
    /// at a small recovery height: the full one is 1,024 WOTS keys, slow in
    /// a debug build, and midwimble's own tests show the two derivations of
    /// the recovery key agree at every height.
    #[test]
    fn the_account_is_the_one_midwimbles_wallet_derives() {
        let seed = bip39::Mnemonic::parse_normalized(PHRASE).unwrap().to_seed("");
        let theirs = WalletKeys::from_seed(&seed, derive_recovery_keypair(&seed, 2).unwrap().public_key());
        let ours = keys_from_mnemonic_at(PHRASE, 2).unwrap();
        assert_eq!(ours.address(), theirs.address());
        // Case and stray whitespace do not change the account.
        let messy = format!("  {}  ", PHRASE.to_uppercase());
        assert_eq!(keys_from_mnemonic_at(&messy, 2).unwrap().address(), theirs.address());
        assert!(keys_from_mnemonic_at("not a phrase", 2).is_err());

        // What the wallet file keeps loads back as the same account.
        let stored = ours.to_bytes();
        assert_eq!(WalletKeys::from_bytes(&stored).unwrap().address(), theirs.address());
    }

    /// One negotiated lock, as two parties would hold it.
    struct Leg {
        view: JointView,
        jk: Point32,
        funding: Transaction,
        refund: Transaction,
        claim: JointSpend,
        plan: KernelPlan,
        s1: Scalar,
        s2_pre: Scalar,
        input_sig: SchnorrSig,
        funder_share: OwnerShare,
        receiver_share: OwnerShare,
        receiver: WalletKeys,
        t: Scalar,
    }

    fn coin(keys: &WalletKeys, value: u64) -> Spendable {
        let created = new_output(&keys.address(), value).unwrap();
        let owned = scan_output(keys, &created.output).unwrap();
        Spendable {
            commitment: created.output.commitment,
            value,
            blinding: owned.blinding,
            owner_secret: owned.owner_secret,
        }
    }

    fn negotiate(refund_height: u64) -> Leg {
        let swap_id = [7u8; 32];
        let (funder, receiver) = (WalletKeys::random(), WalletKeys::random());
        let (f, r) = (OwnerShare::random(), OwnerShare::random());
        assert!(verify_share(&r.public, &swap_id, &r.prove(&swap_id)));
        assert!(verify_share(&f.public, &swap_id, &f.prove(&swap_id)));
        let jk = joint_key(&f.public, &r.public).unwrap();
        let joint = JointOutput::new(25_000_000, jk).unwrap();
        let funding = build_funding(&[coin(&funder, 26_000_000)], &joint, &funder.address(), min_fee(1, 2, 1)).unwrap();
        let view = joint.view();
        view.check(&jk).unwrap();

        let (nf, nr) = (SecretNonce::on_g(), SecretNonce::on_g());
        let ins = InputSigning::new(&view.commitment, &jk, &nf.point, &nr.point);
        let (sf, sr) = (ins.partial(nf, &f).unwrap(), ins.partial(nr, &r).unwrap());
        let input_sig = ins.combine(&sf, &sr).unwrap();

        let draft = JointSpend::new(&view, &f, &r.public, &funder.address(), min_fee(1, 1, 1), refund_height).unwrap();
        let (n1, nb, no) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
        let plan = draft.plan(&n1.point, &nb.point, &no.point, None);
        let s2o = plan.sign_as_other(no, &r, refund_height, None).unwrap();
        let s2 = draft.sign_owner(&plan, nb).unwrap() + s2o;
        let refund = draft
            .assemble(&input_sig, &plan, plan.complete(&draft.sign_pedersen(&plan, n1).unwrap(), &s2, None).unwrap())
            .unwrap();

        let t = random_scalar();
        let t_point = compress(&RistrettoPoint::mul_base(&t));
        let claim = JointSpend::new(&view, &r, &f.public, &receiver.address(), min_fee(1, 1, 1), 0).unwrap();
        let (m1, mb, mo) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
        let cplan = claim.plan(&m1.point, &mb.point, &mo.point, Some(t_point));
        let s2_f = cplan.sign_as_other(mo, &f, 0, Some(&t_point)).unwrap();
        assert!(cplan.verify_owner_partial(&s2_f, &cplan.r2_other, &decompress(&f.public).unwrap()).unwrap());
        let s2_r = claim.sign_owner(&cplan, mb).unwrap();
        assert!(cplan
            .verify_owner_partial(&s2_r, &cplan.r2_builder, &cplan.builder_owner_excess(&f.public).unwrap())
            .unwrap());
        let s1 = claim.sign_pedersen(&cplan, m1).unwrap();
        assert!(cplan.verify_presig(&s1, &(s2_r + s2_f)).unwrap());

        Leg {
            view,
            jk,
            funding,
            refund,
            claim,
            plan: cplan,
            s1,
            s2_pre: s2_r + s2_f,
            input_sig,
            funder_share: f,
            receiver_share: r,
            receiver,
            t,
        }
    }

    #[test]
    fn the_claim_completes_with_t_and_hands_t_to_the_funder() {
        let leg = negotiate(1_120);
        leg.funding.validate(Context::Relay).unwrap();
        assert_eq!(leg.refund.lock_height(), 1_120);
        let sig = leg.plan.complete(&leg.s1, &leg.s2_pre, Some(&leg.t)).unwrap();
        let claim = leg.claim.assemble(&leg.input_sig, &leg.plan, sig.clone()).unwrap();
        let paid: u64 = claim.outputs().filter_map(|o| scan_output(&leg.receiver, o)).map(|o| o.value).sum();
        assert_eq!(paid, 25_000_000 - min_fee(1, 1, 1));
        assert_eq!(extract_adaptor_secret(&leg.s2_pre, &sig), Some(leg.t));
    }

    #[test]
    fn the_claim_cannot_be_completed_with_a_wrong_secret() {
        let leg = negotiate(1_120);
        let sig = leg.plan.complete(&leg.s1, &leg.s2_pre, Some(&(leg.t + Scalar::ONE))).unwrap();
        let err = leg.claim.assemble(&leg.input_sig, &leg.plan, sig).unwrap_err();
        assert!(err.to_string().contains("kernel signature"), "{err}");
    }

    /// The receiver asks for a claim co-signature without `T`: refused. Had
    /// the funder signed it, the receiver would take the MWB without the
    /// secret, so without paying on midstate.
    #[test]
    fn a_claim_without_the_adaptor_point_is_refused() {
        let leg = negotiate(1_120);
        let t_point = compress(&RistrettoPoint::mul_base(&leg.t));
        let stripped =
            JointSpend::new(&leg.view, &leg.receiver_share, &leg.funder_share.public, &leg.receiver.address(), 2_500, 0)
                .unwrap();
        let (n1, nb, no) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
        let plan = stripped.plan(&n1.point, &nb.point, &no.point, None);
        let err = plan.sign_as_other(no, &leg.funder_share, 0, Some(&t_point)).unwrap_err();
        assert!(err.to_string().contains("adaptor"), "{err}");

        // A funder that skipped the check:
        let (n1, nb, no) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
        let plan = stripped.plan(&n1.point, &nb.point, &no.point, None);
        let s2 = stripped.sign_owner(&plan, nb).unwrap() + plan.sign_as_other(no, &leg.funder_share, 0, None).unwrap();
        let sig = plan.complete(&stripped.sign_pedersen(&plan, n1).unwrap(), &s2, None).unwrap();
        assert!(stripped.assemble(&leg.input_sig, &plan, sig).is_ok(), "taken without t");
    }

    #[test]
    fn a_refund_without_its_lock_height_is_refused() {
        let leg = negotiate(1_120);
        let early = JointSpend::new(&leg.view, &leg.funder_share, &leg.receiver_share.public, &leg.receiver.address(), 2_500, 0)
            .unwrap();
        let (n1, nb, no) = (SecretNonce::on_h(), SecretNonce::on_g(), SecretNonce::on_g());
        let plan = early.plan(&n1.point, &nb.point, &no.point, None);
        let err = plan.sign_as_other(no, &leg.receiver_share, 1_120, None).unwrap_err();
        assert!(err.to_string().contains("min_height"), "{err}");
    }

    /// The funder knows the blinding factor, its own share and the input
    /// signature. The owner sum still needs the receiver's share.
    #[test]
    fn the_funder_cannot_spend_the_lock_alone() {
        let leg = negotiate(1_120);
        let fee = min_fee(1, 1, 1);
        let r_j = leg.view.check(&leg.jk).unwrap();
        let ko_f = scalar_from_bytes(&leg.funder_share.to_bytes()).unwrap();
        let out = new_output(&leg.receiver.address(), 25_000_000 - fee).unwrap();
        let group = OutputGroup::prove(vec![out.output.clone()], &[out.value], &[out.blinding]).unwrap();
        let input = Input { commitment: leg.view.commitment, owner_key: leg.jk, signature: leg.input_sig.clone() };
        let (o, x) = (random_scalar(), random_scalar());
        let e = out.blinding - r_j - o;
        let spend = |kernel: Kernel| Transaction {
            body: TxBody { inputs: vec![input.clone()], outputs: vec![group.clone()], kernels: vec![kernel] },
            kernel_offset: o.to_bytes(),
            owner_offset: x.to_bytes(),
        };
        // An owner excess it can sign for leaves the sum short by Ko_R...
        let err = spend(Kernel::new(KernelFeatures::Plain, fee, 0, &e, &(ko_f - x))).validate(Context::Relay).unwrap_err();
        assert!(err.to_string().contains("owner sum"), "{err}");
        // ...and the one that balances, it cannot sign for.
        let mut kernel = Kernel::new(KernelFeatures::Plain, fee, 0, &e, &(ko_f - x));
        kernel.owner_excess = compress(&(decompress(&leg.jk).unwrap() - RistrettoPoint::mul_base(&x)));
        let err = spend(kernel).validate(Context::Relay).unwrap_err();
        assert!(err.to_string().contains("kernel signature"), "{err}");
    }

    #[test]
    fn a_rogue_owner_key_cannot_be_proven() {
        let swap_id = [7u8; 32];
        let funder = OwnerShare::random();
        let x = random_scalar();
        let rogue = compress(&(RistrettoPoint::mul_base(&x) - decompress(&funder.public).unwrap()));
        assert!(!verify_share(&rogue, &swap_id, &schnorr_sign(&x, &[0u8; 32])));
        // What it would have bought: the joint key is X itself.
        assert_eq!(joint_key(&funder.public, &rogue).unwrap(), compress(&RistrettoPoint::mul_base(&x)));
    }

    #[test]
    fn owner_shares_round_trip_through_the_journal() {
        let s = OwnerShare::random();
        let back = OwnerShare::from_bytes(&s.to_bytes()).unwrap();
        assert_eq!(back.public, s.public);
        assert!(OwnerShare::from_bytes(&[0xff; 32]).is_none());
    }
}
