//! The link between the two legs of an MDS <-> MWB swap: a cut-and-choose
//! proof that the hashes in a midstate lock and the adaptor point `T` on the
//! midwimble side hide the same secret `t`.
//!
//! Midwimble releases a secret as a scalar, through an adaptor signature;
//! midstate's VM checks hash preimages and nothing else. The secret therefore
//! takes two forms, and this is what ties them, entirely off chain.
//!
//! # Statement
//!
//! The secret holder knows `t` with `T = t·G`. For each of `k` instances it
//! splits `t = a_j + b_j` at random and commits to both halves twice over:
//!
//! ```text
//!   A_j = a_j·G                 (B_j = T − A_j is implied)
//!   ha_j = BLAKE3(a_j)          hb_j = BLAKE3(b_j)
//! ```
//!
//! The verifier challenges one half per instance; the prover opens it, and
//! the verifier checks it against *both* its point and its hash. The unopened
//! halves' hashes build the midstate lock. A half on its own is uniformly
//! random, so openings reveal nothing about `t`; but whoever learns an
//! unopened half `u_j` and its opened partner `o_j` learns `t = o_j + u_j`,
//! and whoever learns `t` learns every `u_j = t − o_j`.
//!
//! # Soundness, and which lock goes with which direction
//!
//! An instance is *bad* if one half's hash does not match its point. A
//! cheater must leave every bad half unopened, which each challenge bit
//! foils with probability 1/2. Which lock is used decides how many bad
//! instances the cheater needs, and the pairing below makes it all of them:
//!
//! * **OR lock** (`compile_mwb_or_lock`), when the secret comes out on
//!   midwimble: the counterparty needs *one* unopened half whose hash is a
//!   leaf, which it derives from `t` ([`VerifiedLink::leaf_for_t`]).
//! * **AND lock** (`compile_mwb_and_lock`), when the secret comes out on
//!   midstate: the claim publishes *every* unopened half, and the
//!   counterparty needs one consistent with its opened partner
//!   ([`VerifiedLink::recover_t`]).
//!
//! Either way a cheat survives with probability `2^-k`. The opposite
//! pairings let a single bad instance win, so the pairing is part of the
//! protocol.
//!
//! # Challenges
//!
//! [`Challenge::random`] is the interactive verifier's coin, required when
//! `k` is small: the AND lock caps `k` at `MAX_MWB_AND_PREIMAGES` (62), and
//! a Fiat–Shamir challenge over that few bits can be ground.
//! [`Challenge::fiat_shamir`] refuses `k < 128`; the OR lock affords 128
//! because its on-chain cost grows with `log2 k` (seven Merkle levels).

use anyhow::{bail, ensure, Result};
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::IsIdentity;
use midstate::core::script::{compile_mwb_and_lock, compile_mwb_or_lock, MAX_MWB_AND_PREIMAGES};
use midstate::core::types::{hash, hash_concat};
use midwimble::core::mw::crypto::{compress, decompress, random_scalar, scalar_from_bytes, Point32};
use serde::{Deserialize, Serialize};

/// Instances behind an OR lock: enough for a Fiat–Shamir challenge, and
/// seven Merkle levels (231 bytes) on chain.
pub const K_OR: usize = 128;

/// Instances behind an AND lock. Each is a 32-byte witness item; 48 leaves
/// room under `MAX_MWB_AND_PREIMAGES` and costs 2^-48.
pub const K_AND: usize = 48;

/// Smallest `k` for which [`Challenge::fiat_shamir`] may be used.
pub const MIN_FIAT_SHAMIR_K: usize = 128;

const FS_DOMAIN: &[u8] = b"midstate.mwb-swap.link-challenge.v1";

// ── Prover ──────────────────────────────────────────────────────────────────

/// The secret holder's side: `t` and the first half `a_j` of each split.
pub struct LinkSecret {
    t: Scalar,
    a: Vec<Scalar>,
}

impl LinkSecret {
    pub fn generate(k: usize) -> Self {
        Self::from_t(random_scalar(), k)
    }

    pub fn from_t(t: Scalar, k: usize) -> Self {
        assert!(k > 0, "at least one instance");
        Self { t, a: (0..k).map(|_| random_scalar()).collect() }
    }

    pub fn k(&self) -> usize {
        self.a.len()
    }

    /// The adaptor secret. Only the prover holds it until a leg reveals it.
    pub fn t(&self) -> Scalar {
        self.t
    }

    pub fn t_point(&self) -> Point32 {
        compress(&RistrettoPoint::mul_base(&self.t))
    }

    fn half(&self, j: usize, b_side: bool) -> Scalar {
        if b_side {
            self.t - self.a[j]
        } else {
            self.a[j]
        }
    }

    pub fn commit(&self) -> LinkCommitment {
        LinkCommitment {
            t_point: self.t_point(),
            a_points: self.a.iter().map(|a| compress(&RistrettoPoint::mul_base(a))).collect(),
            hash_a: (0..self.k()).map(|j| hash(self.half(j, false).as_bytes())).collect(),
            hash_b: (0..self.k()).map(|j| hash(self.half(j, true).as_bytes())).collect(),
        }
    }

    /// Opens the challenged half of every instance.
    pub fn open(&self, challenge: &Challenge) -> LinkOpening {
        assert_eq!(challenge.0.len(), self.k(), "challenge length");
        LinkOpening {
            scalars: challenge.0.iter().enumerate().map(|(j, &b)| self.half(j, b).to_bytes()).collect(),
        }
    }

    /// The halves the challenge left closed, in instance order. An AND
    /// lock's claim publishes all of them.
    pub fn unopened(&self, challenge: &Challenge) -> Vec<[u8; 32]> {
        challenge.0.iter().enumerate().map(|(j, &b)| self.half(j, !b).to_bytes()).collect()
    }
}

// ── Messages ────────────────────────────────────────────────────────────────

/// Prover → verifier, before the challenge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkCommitment {
    pub t_point: Point32,
    pub a_points: Vec<Point32>,
    pub hash_a: Vec<[u8; 32]>,
    pub hash_b: Vec<[u8; 32]>,
}

/// Which half to open per instance: `false` opens `a_j`, `true` opens `b_j`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Challenge(pub Vec<bool>);

impl Challenge {
    /// The interactive verifier's coin flips, drawn after the commitment
    /// arrives.
    pub fn random(k: usize) -> Self {
        Self((0..k).map(|_| rand::random::<bool>()).collect())
    }

    /// Non-interactive challenge over the swap's context. Refuses `k < 128`:
    /// a prover can re-roll its commitment until a challenge favours it, so
    /// the work that takes must be out of reach.
    pub fn fiat_shamir(context: &[u8], c: &LinkCommitment) -> Result<Self> {
        let k = c.a_points.len();
        ensure!(
            k >= MIN_FIAT_SHAMIR_K,
            "a Fiat-Shamir challenge needs k >= {MIN_FIAT_SHAMIR_K}, got {k}; use an interactive one"
        );
        let mut h = blake3::Hasher::new();
        h.update(&(FS_DOMAIN.len() as u64).to_le_bytes());
        h.update(FS_DOMAIN);
        h.update(&(context.len() as u64).to_le_bytes());
        h.update(context);
        h.update(&(k as u64).to_le_bytes());
        h.update(&c.t_point);
        for j in 0..k {
            h.update(&c.a_points[j]);
            h.update(&c.hash_a[j]);
            h.update(&c.hash_b[j]);
        }
        let mut bits = vec![0u8; k.div_ceil(8)];
        h.finalize_xof().fill(&mut bits);
        Ok(Self((0..k).map(|j| ((bits[j / 8] >> (j % 8)) & 1) == 1).collect()))
    }
}

/// Prover → verifier, after the challenge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkOpening {
    pub scalars: Vec<[u8; 32]>,
}

// ── Verifier ────────────────────────────────────────────────────────────────

/// What the verifier keeps once the proof checks out.
#[derive(Clone, Debug)]
pub struct VerifiedLink {
    pub t_point: Point32,
    pub challenge: Vec<bool>,
    /// The opened half of each instance.
    pub opened: Vec<Scalar>,
    /// BLAKE3 of each unopened half, in instance order: the OR lock's leaves
    /// or the AND lock's chain.
    pub unopened_hashes: Vec<[u8; 32]>,
}

/// Checks every opened half against its point and its hash.
///
/// # Reasoning
/// This is the only thing standing between a counterparty and a midstate lock
/// whose hashes have nothing to do with `T`: lock funds against such a lock
/// and the secret that comes out on one chain opens nothing on the other.
///
/// # Formal Specification
///
/// ```text
/// Pre:  ch was drawn after c arrived (or ch = Challenge::fiat_shamir(ctx, c), k ≥ 128)
/// Post: Ok(v) ⇒ ∀ j < k •
///         v.opened[j]·G = (ch[j] ? T − A_j : A_j)
///         ∧ BLAKE3(v.opened[j]) = (ch[j] ? hb_j : ha_j)
///         ∧ v.unopened_hashes[j] = (ch[j] ? ha_j : hb_j)
///       Err(_) ⇒ do not trade with this counterparty
/// ```
///
/// # Safety / Invariants
/// - A prover that spoiled every instance passes with probability `2^-k`;
///   one that spoiled fewer leaves a good instance, which is all
///   [`VerifiedLink::recover_t`] and [`VerifiedLink::leaf_for_t`] need.
/// - Openings must be canonical scalars, so one value has one encoding and
///   one hash.
pub fn verify(c: &LinkCommitment, ch: &Challenge, o: &LinkOpening) -> Result<VerifiedLink> {
    let k = c.a_points.len();
    ensure!(k > 0, "empty commitment");
    ensure!(
        c.hash_a.len() == k && c.hash_b.len() == k && ch.0.len() == k && o.scalars.len() == k,
        "commitment, challenge and opening disagree on k"
    );
    let t_pt = match decompress(&c.t_point) {
        Some(p) if !p.is_identity() => p,
        _ => bail!("adaptor point is not a valid non-identity point"),
    };
    let mut opened = Vec::with_capacity(k);
    let mut unopened_hashes = Vec::with_capacity(k);
    for j in 0..k {
        let a_pt = decompress(&c.a_points[j]).ok_or_else(|| anyhow::anyhow!("instance {j}: invalid point"))?;
        let (pt, h_open, h_closed) =
            if ch.0[j] { (t_pt - a_pt, c.hash_b[j], c.hash_a[j]) } else { (a_pt, c.hash_a[j], c.hash_b[j]) };
        let s = scalar_from_bytes(&o.scalars[j])
            .ok_or_else(|| anyhow::anyhow!("instance {j}: opening is not a canonical scalar"))?;
        if RistrettoPoint::mul_base(&s) != pt {
            bail!("instance {j}: opened half does not match its point");
        }
        if hash(&o.scalars[j]) != h_open {
            bail!("instance {j}: opened half does not match its hash");
        }
        opened.push(s);
        unopened_hashes.push(h_closed);
    }
    Ok(VerifiedLink { t_point: c.t_point, challenge: ch.0.clone(), opened, unopened_hashes })
}

impl VerifiedLink {
    pub fn k(&self) -> usize {
        self.opened.len()
    }

    /// The OR lock's root. `k` must be a power of two.
    pub fn merkle_root(&self) -> [u8; 32] {
        merkle_root(&self.unopened_hashes)
    }

    pub fn merkle_proof(&self, j: usize) -> Vec<u8> {
        merkle_proof(&self.unopened_hashes, j)
    }

    /// The AND lock's accumulator.
    pub fn chain_accumulator(&self) -> [u8; 32] {
        chain_accumulator(&self.unopened_hashes)
    }

    /// The OR lock this link backs.
    pub fn or_lock(&self, receiver_pk: &[u8; 32], timeout_height: u64, refund_pk: &[u8; 32]) -> Vec<u8> {
        compile_mwb_or_lock(&self.merkle_root(), receiver_pk, timeout_height, refund_pk)
    }

    /// The AND lock this link backs.
    pub fn and_lock(&self, receiver_pk: &[u8; 32], timeout_height: u64, refund_pk: &[u8; 32]) -> Result<Vec<u8>> {
        compile_mwb_and_lock(&self.chain_accumulator(), self.k(), receiver_pk, timeout_height, refund_pk)
            .ok_or_else(|| anyhow::anyhow!("an AND lock takes 1..={MAX_MWB_AND_PREIMAGES} preimages, not {}", self.k()))
    }

    /// AND lock, after the counterparty claimed on midstate: `t` from any
    /// published preimage consistent with its opened partner. `preimages`
    /// are in instance order ([`and_preimages_from_witness`]).
    ///
    /// ```text
    /// Post: Some(t) ⇒ t·G = T
    ///       None    ⇔ ∀ j • (opened[j] + preimages[j])·G ≠ T   (every instance spoiled)
    /// ```
    pub fn recover_t(&self, preimages: &[Vec<u8>]) -> Option<Scalar> {
        let t_pt = decompress(&self.t_point)?;
        preimages.iter().zip(&self.opened).find_map(|(u, o)| {
            let u: [u8; 32] = u.as_slice().try_into().ok()?;
            let t = o + scalar_from_bytes(&u)?;
            (RistrettoPoint::mul_base(&t) == t_pt).then_some(t)
        })
    }

    /// OR lock, after the counterparty claimed on midwimble: an unopened half
    /// whose hash is a leaf, found from `t`. Returns the instance and the
    /// 32-byte preimage the claim witness carries.
    ///
    /// ```text
    /// Post: Some((j, u)) ⇒ u = t − opened[j] ∧ BLAKE3(u) = unopened_hashes[j]
    ///       None         ⇔ t·G ≠ T ∨ every instance is spoiled
    /// ```
    pub fn leaf_for_t(&self, t: &Scalar) -> Option<(usize, [u8; 32])> {
        if compress(&RistrettoPoint::mul_base(t)) != self.t_point {
            return None;
        }
        self.opened.iter().zip(&self.unopened_hashes).enumerate().find_map(|(j, (o, h))| {
            let u = (t - o).to_bytes();
            (hash(&u) == *h).then_some((j, u))
        })
    }
}

// ── Commitments, bit-compatible with midstate's VM ──────────────────────────

/// Root over `leaves` (a power of two), hashing pairs with
/// `hash_concat(left, right)` exactly as `OP_MERKLE_ROOT` folds.
pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(leaves.len().is_power_of_two(), "leaf count must be a power of two");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| hash_concat(&p[0], &p[1])).collect();
    }
    level[0]
}

/// A path in `OP_MERKLE_ROOT`'s format: 33-byte records from the leaf up, a
/// sibling then a flag that is `1` when the sibling sits on the left.
pub fn merkle_proof(leaves: &[[u8; 32]], mut index: usize) -> Vec<u8> {
    assert!(leaves.len().is_power_of_two() && index < leaves.len());
    let mut proof = Vec::new();
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        proof.extend_from_slice(&level[index ^ 1]);
        proof.push((index & 1) as u8);
        level = level.chunks(2).map(|p| hash_concat(&p[0], &p[1])).collect();
        index >>= 1;
    }
    proof
}

/// `acc_0 = h_0`, `acc_j = BLAKE3(acc_{j-1} ‖ h_j)`: what the AND lock's
/// `HASH (SWAP HASH CONCAT HASH)*` computes over its witness.
pub fn chain_accumulator(hashes: &[[u8; 32]]) -> [u8; 32] {
    assert!(!hashes.is_empty());
    hashes[1..].iter().fold(hashes[0], |acc, h| hash_concat(&acc, h))
}

// ── Witnesses ───────────────────────────────────────────────────────────────

pub fn or_claim_witness(sig: &[u8], proof: &[u8], preimage: &[u8; 32]) -> Vec<Vec<u8>> {
    vec![sig.to_vec(), proof.to_vec(), preimage.to_vec(), vec![1]]
}

/// `unopened` in instance order; instance 0 sits on top of the stack, so it
/// is pushed last.
pub fn and_claim_witness(sig: &[u8], unopened: &[[u8; 32]]) -> Vec<Vec<u8>> {
    let mut w = Vec::with_capacity(unopened.len() + 2);
    w.push(sig.to_vec());
    w.extend(unopened.iter().rev().map(|u| u.to_vec()));
    w.push(vec![1]);
    w
}

/// The counterparty's side of an AND claim: the preimages from a witness
/// read off the chain (`/tx/by_input` returns it unstripped), back in
/// instance order. `None` unless it is an AND claim of `k`.
pub fn and_preimages_from_witness(witness: &[Vec<u8>], k: usize) -> Option<Vec<Vec<u8>>> {
    if witness.len() != k + 2 || witness.last()? != &vec![1u8] {
        return None;
    }
    Some(witness[1..=k].iter().rev().cloned().collect())
}

pub fn refund_witness(sig: &[u8]) -> Vec<Vec<u8>> {
    vec![sig.to_vec(), vec![0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honest_link_verifies_and_both_extractions_work() {
        let s = LinkSecret::generate(16);
        let ch = Challenge::random(16);
        let link = verify(&s.commit(), &ch, &s.open(&ch)).unwrap();
        let pre: Vec<Vec<u8>> = s.unopened(&ch).iter().map(|u| u.to_vec()).collect();
        assert_eq!(link.recover_t(&pre), Some(s.t()));
        let (j, u) = link.leaf_for_t(&s.t()).unwrap();
        assert_eq!(hash(&u), link.unopened_hashes[j]);
        assert!(link.leaf_for_t(&(s.t() + Scalar::ONE)).is_none());
    }

    #[test]
    fn openings_are_checked_against_point_and_hash() {
        let s = LinkSecret::generate(8);
        let ch = Challenge(vec![false; 8]);
        let mut o = s.open(&ch);
        o.scalars[3] = random_scalar().to_bytes();
        let err = verify(&s.commit(), &ch, &o).unwrap_err().to_string();
        assert!(err.contains("instance 3"), "{err}");
    }

    #[test]
    fn fiat_shamir_refuses_small_k_and_binds_the_context() {
        assert!(Challenge::fiat_shamir(b"ctx", &LinkSecret::generate(K_AND).commit()).is_err());
        let c = LinkSecret::generate(K_OR).commit();
        assert_ne!(Challenge::fiat_shamir(b"ctx", &c).unwrap(), Challenge::fiat_shamir(b"other", &c).unwrap());
    }

    /// The witnesses built here open the locks midstate's builders compile,
    /// in midstate's own VM.
    #[test]
    fn witnesses_open_the_locks_in_midstates_vm() {
        use midstate::core::script::{execute_script, ExecContext};
        let seed = [5u8; 32];
        let pk = midstate::core::wots::keygen(&seed);
        let c = [9u8; 32];
        let sig = midstate::core::wots::sig_to_bytes(&midstate::core::wots::sign(&seed, &c));
        let ctx = ExecContext {
            commitment: &c,
            height: midstate::core::types::V6_ACTIVATION_HEIGHT,
            outputs: &[],
            input_value: 0,
            input_state: None,
            this_address: [0; 32],
            sum_input_value: 0,
        };

        let s = LinkSecret::generate(K_OR);
        let ch = Challenge::fiat_shamir(b"swap", &s.commit()).unwrap();
        let link = verify(&s.commit(), &ch, &s.open(&ch)).unwrap();
        let (j, u) = link.leaf_for_t(&s.t()).unwrap();
        let w = or_claim_witness(&sig, &link.merkle_proof(j), &u);
        assert!(execute_script(&link.or_lock(&pk, 1, &[0; 32]), &w, &ctx).is_ok());

        let s = LinkSecret::generate(K_AND);
        let ch = Challenge::random(K_AND);
        let link = verify(&s.commit(), &ch, &s.open(&ch)).unwrap();
        let w = and_claim_witness(&sig, &s.unopened(&ch));
        assert!(execute_script(&link.and_lock(&pk, 1, &[0; 32]).unwrap(), &w, &ctx).is_ok());
        let back = and_preimages_from_witness(&w, K_AND).unwrap();
        assert_eq!(link.recover_t(&back), Some(s.t()));
    }

    /// A prover that spoils one half of every instance passes only if the
    /// challenge opens none of them: probability 2^-k. Measured at k = 4.
    #[test]
    fn a_cheating_prover_wins_only_with_probability_two_to_the_minus_k() {
        const K: usize = 4;
        const TRIALS: usize = 4_000;
        let mut wins = 0;
        for _ in 0..TRIALS {
            let t = random_scalar();
            let a: Vec<Scalar> = (0..K).map(|_| random_scalar()).collect();
            let bad: Vec<bool> = (0..K).map(|_| rand::random()).collect();
            let junk: Vec<[u8; 32]> = (0..K).map(|_| rand::random()).collect();
            let half = |j: usize, b: bool| if b { (t - a[j]).to_bytes() } else { a[j].to_bytes() };
            let published = |j: usize, b: bool| if bad[j] == b { junk[j] } else { half(j, b) };
            let c = LinkCommitment {
                t_point: compress(&RistrettoPoint::mul_base(&t)),
                a_points: a.iter().map(|x| compress(&RistrettoPoint::mul_base(x))).collect(),
                hash_a: (0..K).map(|j| hash(&published(j, false))).collect(),
                hash_b: (0..K).map(|j| hash(&published(j, true))).collect(),
            };
            let ch = Challenge::random(K);
            let opening = LinkOpening { scalars: (0..K).map(|j| half(j, ch.0[j])).collect() };
            let Ok(link) = verify(&c, &ch, &opening) else { continue };
            let preimages: Vec<Vec<u8>> = (0..K).map(|j| published(j, !ch.0[j]).to_vec()).collect();
            assert!(link.recover_t(&preimages).is_none(), "an accepted cheat blocks the AND lock");
            assert!(link.leaf_for_t(&t).is_none(), "an accepted cheat blocks the OR lock");
            wins += 1;
        }
        // Expect TRIALS / 2^K = 250; six standard deviations either side.
        assert!((160..=340).contains(&wins), "cheater won {wins} of {TRIALS}, expected about 250");
    }
}
