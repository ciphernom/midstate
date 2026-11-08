//! Inheritance plans and payment streams: two covenants built from opcodes
//! consensus already has (`CHECKSIG`, `CHECKTIMEVERIFY`, `SUM_TO_ADDR`,
//! `SUM_INPUT_VALUE`, `DIV`, `MUL`). Nothing here changes consensus.
//! `docs/design/inheritance-and-streams.md` has the reasoning in full.
//!
//! # Inheritance plans
//!
//! The owner's coins sit at a plan address. The owner spends them with their
//! own key whenever they like. If they leave them untouched past the plan's
//! deadline, *anyone* may move them, but only by paying every heir its share
//! of them. Checking in is moving the coins to the next deadline's address.
//! Deadlines run on a fixed schedule (`start + k·period`), so an heir who holds
//! the plan's parameters can find its coins at whichever deadline they sit.
//!
//! # Payment streams
//!
//! A payer funds a run of tranches, one address each, the j-th payable from
//! height `first + j·period`. From then *anyone* may move it, but only by paying
//! the recipient all of it, bar a fee allowance. A cancellable stream also
//! lets the payer take back any tranche that has not been collected yet.
//!
//! # Why every payout goes to an address of its own
//!
//! `SUM_INPUT_VALUE` totals only the inputs that share the running script, and
//! `SUM_TO_ADDR` totals every output to an address. If two covenants paid the
//! same address, one transaction could spend both and satisfy each with the
//! same outputs, and whoever built it would keep the smaller one. So heirs and
//! recipients are paid at *payout addresses*, `PUSH tag DROP PUSH pk CHECKSIG`,
//! with a tag unique to the plan's deadline and heir, or the stream's tranche:
//! no two covenants ever share one. The heir or recipient sweeps them with its
//! key.
//!
//! # Why the fee is a share
//!
//! The keyless paths cannot be told what fee a transaction needs, so each
//! keeps back `1/SHARE_DENOM` of what it moves (and the rounding), which is
//! what pays for the transaction. Coins are at least [`MIN_COIN`], so that is
//! always at least 512 units: enough for tens of kilobytes. Shares are counted
//! in `1/SHARE_DENOM` parts and add up to all but that one part, so nothing is
//! left unowed for whoever builds the claim.

use crate::core::script::{
    push_data, push_int, OP_CHECKSIG, OP_CHECKTIMEVERIFY, OP_DIV, OP_DROP, OP_DUP, OP_ELSE, OP_ENDIF,
    OP_GREATER_OR_EQUAL, OP_IF, OP_MUL, OP_SUM_INPUT_VALUE, OP_SUM_TO_ADDR, OP_SWAP, OP_VERIFY,
};
use crate::core::types::{compute_coin_id, decompose_value, hash, InputReveal, OutputData, Predicate, Witness};

/// Shares and the fee allowance are counted in parts of this.
pub const SHARE_DENOM: u64 = 1 << 16;
/// The smallest coin a plan or a stream holds.
pub const MIN_COIN_EXP: u32 = 25;
pub const MIN_COIN: u64 = 1 << MIN_COIN_EXP;
/// Heirs a plan may name.
pub const MAX_HEIRS: usize = 8;
/// Coins one deadline's address may hold (the wallet funds plans with at most
/// this many; heirs look for no more).
pub const MAX_PLAN_COINS: u32 = 8;
/// Tranches a stream may have, and coins one tranche may be.
pub const MAX_TRANCHES: u32 = 120;
pub const MAX_TRANCHE_COINS: usize = 4;
/// Outputs a stream's funding may use for its tranches.
pub const MAX_STREAM_OUTPUTS: usize = 200;
/// Deadlines one plan may run through.
pub const MAX_EPOCHS: u64 = 1_000;
/// Blocks a day at one a minute.
pub const DAY: u64 = 1_440;
/// The covenants need `SUM_INPUT_VALUE`, which is on from this height.
pub const ACTIVATION_HEIGHT: u64 = crate::core::types::V6_ACTIVATION_HEIGHT;

fn tagged(domain: &[u8], secret: &[u8; 32], a: u64, b: u32) -> [u8; 32] {
    let mut m = Vec::with_capacity(domain.len() + 44);
    m.extend_from_slice(domain);
    m.extend_from_slice(secret);
    m.extend_from_slice(&a.to_le_bytes());
    m.extend_from_slice(&b.to_le_bytes());
    hash(&m)
}

// ── Payout addresses ────────────────────────────────────────────────────────

/// `PUSH tag DROP PUSH pk CHECKSIG`: `pk`'s coins, at an address no other
/// covenant pays.
pub fn payout_script(tag: &[u8; 32], pk: &[u8; 32]) -> Vec<u8> {
    let mut bc = Vec::with_capacity(72);
    push_data(&mut bc, tag);
    bc.push(OP_DROP);
    push_data(&mut bc, pk);
    bc.push(OP_CHECKSIG);
    bc
}

pub fn payout_address(tag: &[u8; 32], pk: &[u8; 32]) -> [u8; 32] {
    hash(&payout_script(tag, pk))
}

pub fn payout_input(tag: &[u8; 32], pk: &[u8; 32], value: u64, salt: [u8; 32]) -> InputReveal {
    InputReveal { predicate: Predicate::Script { bytecode: payout_script(tag, pk) }, value, salt, commitment: None }
}

/// A payout coin's witness: its key's signature over the commitment.
pub fn payout_witness(sig: Vec<u8>) -> Witness {
    Witness::ScriptInputs(vec![sig])
}

/// `required` as power-of-two coins to `address`, salted by `salt(n)`.
fn coins_to(address: [u8; 32], required: u64, salt: &mut impl FnMut() -> [u8; 32]) -> Vec<OutputData> {
    decompose_value(required).into_iter().map(|value| OutputData::Standard { address, value, salt: salt() }).collect()
}

// ── Inheritance plans ───────────────────────────────────────────────────────

/// An heir: its key and its share in parts of [`SHARE_DENOM`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heir {
    pub pk: [u8; 32],
    pub share: u32,
}

/// An inheritance plan's parameters. Everything an heir needs to find and
/// claim its coins, and nothing that spends them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InheritancePlan {
    /// The owner's key (an MSS tree): it spends the coins at any time.
    pub owner_pk: [u8; 32],
    pub heirs: Vec<Heir>,
    /// The k-th deadline is `start + k·period`, for k from 1.
    pub start: u64,
    pub period: u64,
    /// Names the plan's addresses and salts its coins. Not a key: knowing it
    /// lets one find the plan, never spend it.
    pub secret: [u8; 32],
}

const PLAN_VERSION: u8 = 1;
const STREAM_VERSION: u8 = 1;

impl InheritancePlan {
    /// Whether the terms make a plan heirs can always claim.
    pub fn check(&self) -> Result<(), String> {
        if self.heirs.is_empty() || self.heirs.len() > MAX_HEIRS {
            return Err(format!("a plan names one to {MAX_HEIRS} heirs"));
        }
        if self.heirs.iter().any(|h| h.share == 0) {
            return Err("every heir needs a share".into());
        }
        // Exactly all but the fee's part: a share nobody is owed would be
        // the claimer's to keep.
        let total: u64 = self.heirs.iter().map(|h| h.share as u64).sum();
        if total != SHARE_DENOM - 1 {
            return Err(format!("the shares must add up to {} parts", SHARE_DENOM - 1));
        }
        for (i, h) in self.heirs.iter().enumerate() {
            if self.heirs[..i].iter().any(|o| o.pk == h.pk) {
                return Err("each heir is named once".into());
            }
            if h.pk == self.owner_pk {
                return Err("the owner cannot be an heir".into());
            }
        }
        if self.period < DAY || self.period > 10 * 366 * DAY {
            return Err("a check-in period is between a day and ten years".into());
        }
        if self.start.checked_add(self.period.saturating_mul(MAX_EPOCHS)).is_none() {
            return Err("the schedule runs past the end of time".into());
        }
        if self.script(MAX_EPOCHS).len() > crate::core::script::MAX_SCRIPT_SIZE {
            return Err("the plan's script is too large".into());
        }
        Ok(())
    }

    /// The k-th deadline (k ≥ 1).
    pub fn deadline(&self, k: u64) -> u64 {
        self.start + k * self.period
    }

    /// The deadline a check-in at `height` moves the coins to: the first that
    /// leaves at least one full period.
    pub fn epoch_after(&self, height: u64) -> u64 {
        let want = height.saturating_add(self.period).saturating_sub(self.start);
        want.div_ceil(self.period).max(1)
    }

    /// Every deadline coins of this plan could sit at, as of `height`.
    pub fn epochs_to(&self, height: u64) -> std::ops::RangeInclusive<u64> {
        1..=self.epoch_after(height).min(MAX_EPOCHS)
    }

    /// The tag of heir `i`'s payout address for deadline `k`.
    pub fn payout_tag(&self, k: u64, i: usize) -> [u8; 32] {
        tagged(b"midstate/plan/payout/v1", &self.secret, k, i as u32)
    }

    pub fn payout_address(&self, k: u64, i: usize) -> [u8; 32] {
        payout_address(&self.payout_tag(k, i), &self.heirs[i].pk)
    }

    /// The plan's script for deadline `k`.
    ///
    /// ```text
    /// IF    PUSH owner_pk CHECKSIG                  witness [sig, 1]
    /// ELSE  PUSH deadline CHECKTIMEVERIFY           witness [0]
    ///       SUM_INPUT_VALUE PUSH 65536 DIV          unit
    ///       per heir: DUP PUSH share MUL            unit, owed
    ///                 PUSH payout SUM_TO_ADDR       unit, owed, paid
    ///                 SWAP GREATER_OR_EQUAL VERIFY  unit
    ///       DROP PUSH 1
    /// ENDIF
    /// ```
    pub fn script(&self, k: u64) -> Vec<u8> {
        let mut bc = Vec::with_capacity(64 + 48 * self.heirs.len());
        bc.push(OP_IF);
        push_data(&mut bc, &self.owner_pk);
        bc.push(OP_CHECKSIG);
        bc.push(OP_ELSE);
        push_int(&mut bc, self.deadline(k));
        bc.push(OP_CHECKTIMEVERIFY);
        bc.push(OP_SUM_INPUT_VALUE);
        push_int(&mut bc, SHARE_DENOM);
        bc.push(OP_DIV);
        for (i, h) in self.heirs.iter().enumerate() {
            bc.push(OP_DUP);
            push_int(&mut bc, h.share as u64);
            bc.push(OP_MUL);
            push_data(&mut bc, &self.payout_address(k, i));
            bc.push(OP_SUM_TO_ADDR);
            bc.push(OP_SWAP);
            bc.push(OP_GREATER_OR_EQUAL);
            bc.push(OP_VERIFY);
        }
        bc.push(OP_DROP);
        push_int(&mut bc, 1);
        bc.push(OP_ENDIF);
        bc
    }

    pub fn address(&self, k: u64) -> [u8; 32] {
        hash(&self.script(k))
    }

    /// The salt of the n-th coin at deadline `k`'s address. Fixed by the
    /// plan, so heirs find the coins from the UTXO set without scanning.
    pub fn coin_salt(&self, k: u64, n: u32) -> [u8; 32] {
        tagged(b"midstate/plan/coin/v1", &self.secret, k, n)
    }

    pub fn coin_id(&self, k: u64, n: u32, value: u64) -> [u8; 32] {
        compute_coin_id(&self.address(k), value, &self.coin_salt(k, n))
    }

    pub fn input(&self, k: u64, n: u32, value: u64) -> InputReveal {
        InputReveal {
            predicate: Predicate::Script { bytecode: self.script(k) },
            value,
            salt: self.coin_salt(k, n),
            commitment: None,
        }
    }

    /// Coin `n` at deadline `k`.
    pub fn output(&self, k: u64, n: u32, value: u64) -> OutputData {
        OutputData::Standard { address: self.address(k), value, salt: self.coin_salt(k, n) }
    }

    /// What claiming `total` (every coin of deadline `k` the claim spends)
    /// owes each heir: `(payout address, at least)`.
    pub fn owed(&self, k: u64, total: u64) -> Vec<([u8; 32], u64)> {
        let unit = total / SHARE_DENOM;
        self.heirs.iter().enumerate().map(|(i, h)| (self.payout_address(k, i), unit * h.share as u64)).collect()
    }

    /// A claim's outputs: each heir's share as power-of-two coins. What is
    /// left over is the fee.
    pub fn claim_outputs(&self, k: u64, total: u64, mut salt: impl FnMut() -> [u8; 32]) -> Vec<OutputData> {
        self.owed(k, total).into_iter().flat_map(|(a, v)| coins_to(a, v, &mut salt)).collect()
    }

    pub fn owner_witness(sig: Vec<u8>) -> Witness {
        Witness::ScriptInputs(vec![sig, vec![1]])
    }

    pub fn claim_witness() -> Witness {
        Witness::ScriptInputs(vec![vec![0]])
    }

    /// The plan as bytes: version, owner, start, period, secret, then each
    /// heir's key and share.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(82 + 36 * self.heirs.len());
        b.push(PLAN_VERSION);
        b.extend_from_slice(&self.owner_pk);
        b.extend_from_slice(&self.start.to_le_bytes());
        b.extend_from_slice(&self.period.to_le_bytes());
        b.extend_from_slice(&self.secret);
        b.push(self.heirs.len() as u8);
        for h in &self.heirs {
            b.extend_from_slice(&h.pk);
            b.extend_from_slice(&h.share.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        if r.u8()? != PLAN_VERSION {
            return None;
        }
        let owner_pk = r.arr()?;
        let start = r.u64()?;
        let period = r.u64()?;
        let secret = r.arr()?;
        let n = r.u8()? as usize;
        let mut heirs = Vec::with_capacity(n);
        for _ in 0..n {
            heirs.push(Heir { pk: r.arr()?, share: r.u32()? });
        }
        let plan = InheritancePlan { owner_pk, heirs, start, period, secret };
        (r.0.is_empty() && plan.check().is_ok()).then_some(plan)
    }
}

// ── Payment streams ─────────────────────────────────────────────────────────

/// A payment stream's parameters: everything the recipient needs to find and
/// collect its tranches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaymentStream {
    /// Who may take back tranches not yet collected; `None` for a stream that
    /// cannot be cancelled.
    pub payer_pk: Option<[u8; 32]>,
    pub recipient_pk: [u8; 32],
    /// Tranche j is payable from `first + j·period`.
    pub first: u64,
    pub period: u64,
    /// Each tranche's value, and how many there are.
    pub amount: u64,
    pub count: u32,
    pub secret: [u8; 32],
}

impl PaymentStream {
    pub fn check(&self) -> Result<(), String> {
        if self.count == 0 || self.count > MAX_TRANCHES {
            return Err(format!("a stream has one to {MAX_TRANCHES} payments"));
        }
        let coins = self.tranche_coins();
        if coins.len() > MAX_TRANCHE_COINS || coins.iter().any(|c| *c < MIN_COIN) {
            return Err(format!(
                "each payment must be at most {MAX_TRANCHE_COINS} coins of {MIN_COIN} units or more"
            ));
        }
        if coins.len() * self.count as usize > MAX_STREAM_OUTPUTS {
            return Err("that many payments of that size need too many coins: use fewer or rounder ones".into());
        }
        if self.period == 0 || self.period > 366 * DAY {
            return Err("payments are between one block and a year apart".into());
        }
        if self.amount.checked_mul(self.count as u64).is_none()
            || self.first.checked_add(self.period * self.count as u64).is_none()
        {
            return Err("the stream is too large".into());
        }
        if self.payer_pk == Some(self.recipient_pk) {
            return Err("the payer cannot be the recipient".into());
        }
        Ok(())
    }

    /// Each tranche's coins, largest first.
    pub fn tranche_coins(&self) -> Vec<u64> {
        let mut c = decompose_value(self.amount);
        c.sort_unstable_by(|a, b| b.cmp(a));
        c
    }

    pub fn total(&self) -> u64 {
        self.amount * self.count as u64
    }

    /// When tranche `j` is payable.
    pub fn due(&self, j: u32) -> u64 {
        self.first + j as u64 * self.period
    }

    pub fn payout_tag(&self, j: u32) -> [u8; 32] {
        tagged(b"midstate/stream/payout/v1", &self.secret, j as u64, 0)
    }

    pub fn payout_address(&self, j: u32) -> [u8; 32] {
        payout_address(&self.payout_tag(j), &self.recipient_pk)
    }

    /// Tranche `j`'s script.
    ///
    /// ```text
    /// IF    PUSH payer_pk CHECKSIG                       witness [sig, 1]  (cancellable only)
    /// ELSE  PUSH due CHECKTIMEVERIFY                     witness [0], or [] if not cancellable
    ///       SUM_INPUT_VALUE PUSH 65536 DIV PUSH 65535 MUL
    ///       PUSH payout SUM_TO_ADDR SWAP GREATER_OR_EQUAL
    /// ENDIF
    /// ```
    pub fn script(&self, j: u32) -> Vec<u8> {
        let mut bc = Vec::with_capacity(140);
        if let Some(payer) = &self.payer_pk {
            bc.push(OP_IF);
            push_data(&mut bc, payer);
            bc.push(OP_CHECKSIG);
            bc.push(OP_ELSE);
        }
        push_int(&mut bc, self.due(j));
        bc.push(OP_CHECKTIMEVERIFY);
        bc.push(OP_SUM_INPUT_VALUE);
        push_int(&mut bc, SHARE_DENOM);
        bc.push(OP_DIV);
        push_int(&mut bc, SHARE_DENOM - 1);
        bc.push(OP_MUL);
        push_data(&mut bc, &self.payout_address(j));
        bc.push(OP_SUM_TO_ADDR);
        bc.push(OP_SWAP);
        bc.push(OP_GREATER_OR_EQUAL);
        if self.payer_pk.is_some() {
            bc.push(OP_ENDIF);
        }
        bc
    }

    pub fn address(&self, j: u32) -> [u8; 32] {
        hash(&self.script(j))
    }

    pub fn coin_salt(&self, j: u32, n: u32) -> [u8; 32] {
        tagged(b"midstate/stream/coin/v1", &self.secret, j as u64, n)
    }

    /// Tranche `j`'s coins, as (index, value, coin id).
    pub fn coins(&self, j: u32) -> Vec<(u32, u64, [u8; 32])> {
        let a = self.address(j);
        self.tranche_coins()
            .into_iter()
            .enumerate()
            .map(|(n, v)| (n as u32, v, compute_coin_id(&a, v, &self.coin_salt(j, n as u32))))
            .collect()
    }

    pub fn input(&self, j: u32, n: u32) -> InputReveal {
        InputReveal {
            predicate: Predicate::Script { bytecode: self.script(j) },
            value: self.tranche_coins()[n as usize],
            salt: self.coin_salt(j, n),
            commitment: None,
        }
    }

    /// The funding's outputs: every tranche, at its own address.
    pub fn funding_outputs(&self) -> Vec<OutputData> {
        (0..self.count)
            .flat_map(|j| {
                let a = self.address(j);
                self.tranche_coins()
                    .into_iter()
                    .enumerate()
                    .map(move |(n, value)| (a, value, n as u32, j))
            })
            .map(|(address, value, n, j)| OutputData::Standard { address, value, salt: self.coin_salt(j, n) })
            .collect()
    }

    /// What collecting `total` of tranche `j` must pay the recipient.
    pub fn owed(&self, total: u64) -> u64 {
        total / SHARE_DENOM * (SHARE_DENOM - 1)
    }

    pub fn collect_outputs(&self, j: u32, total: u64, mut salt: impl FnMut() -> [u8; 32]) -> Vec<OutputData> {
        coins_to(self.payout_address(j), self.owed(total), &mut salt)
    }

    pub fn cancel_witness(sig: Vec<u8>) -> Witness {
        Witness::ScriptInputs(vec![sig, vec![1]])
    }

    pub fn collect_witness(&self) -> Witness {
        if self.payer_pk.is_some() {
            Witness::ScriptInputs(vec![vec![0]])
        } else {
            Witness::ScriptInputs(vec![])
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(126);
        b.push(STREAM_VERSION);
        b.push(self.payer_pk.is_some() as u8);
        b.extend_from_slice(&self.payer_pk.unwrap_or([0; 32]));
        b.extend_from_slice(&self.recipient_pk);
        b.extend_from_slice(&self.first.to_le_bytes());
        b.extend_from_slice(&self.period.to_le_bytes());
        b.extend_from_slice(&self.amount.to_le_bytes());
        b.extend_from_slice(&self.count.to_le_bytes());
        b.extend_from_slice(&self.secret);
        b
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader(b);
        if r.u8()? != STREAM_VERSION {
            return None;
        }
        let cancellable = match r.u8()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        let payer: [u8; 32] = r.arr()?;
        let s = PaymentStream {
            payer_pk: cancellable.then_some(payer),
            recipient_pk: r.arr()?,
            first: r.u64()?,
            period: r.u64()?,
            amount: r.u64()?,
            count: r.u32()?,
            secret: r.arr()?,
        };
        (r.0.is_empty() && (cancellable || payer == [0; 32]) && s.check().is_ok()).then_some(s)
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Some(a)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn arr(&mut self) -> Option<[u8; 32]> {
        self.take(32)?.try_into().ok()
    }
}

#[cfg(test)]
mod tests;
