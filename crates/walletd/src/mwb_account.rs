//! The MDW account: coins found on midwimble with the account's keys
//! (`WalletData::mwb_keys`), and the payments made from them.
//!
//! Follows midwimble's own wallet (`midwimble::wallet::Wallet::scan_block`):
//! blocks are scanned in order from the embedded node's storage, a block that
//! does not build on the one scanned before it is a reorganisation, and the
//! account then steps back and scans the new chain. Unlike that wallet it
//! keeps each coin's opening and spend secret, so a spend needs no block
//! lookup; they live in wallet.dat, encrypted with everything else that
//! spends (`mwb_store`).
//!
//! A coin is *reserved* while something it was handed to may still spend it:
//! a payment waiting for a block, or a swap's lock funding. Reserved coins are
//! never selected again. A payment's coins are released when a block spends
//! them (the payment confirmed, or something else won); a swap releases its
//! funding coins only if it never broadcast the funding.

use anyhow::{anyhow, bail, Result};
use curve25519_dalek::scalar::Scalar;
use midwimble::core::mw::builder::{build_transaction, Payment, Spendable};
use midwimble::core::mw::crypto::{scalar_from_bytes, Point32};
use midwimble::core::mw::stealth::{scan_output, StealthAddress, WalletKeys};
use midwimble::core::mw::transaction::{Output, Transaction};
use midwimble::core::types::{Batch, COINBASE_MATURITY};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Recent block ids kept for reorganisation detection, as midwimble's wallet.
pub const REORG_WINDOW: usize = 200;
/// Blocks a scan steps back each time it finds the chain changed under it.
const ROLLBACK_STEP: u64 = 10;
/// What [`MwbAccount::scan`] fails with when a block does not build on the
/// one scanned before it.
pub const REORG: &str = "midwimble reorg detected";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbCoin {
    pub commitment: Point32,
    pub value: u64,
    pub blinding: [u8; 32],
    pub owner_secret: [u8; 32],
    pub height: u64,
    pub coinbase: bool,
    pub spent_height: Option<u64>,
    /// Who may still spend it: `pay:<kernel id>` for a payment waiting for a
    /// block, `swap:<swap id>` for a swap's lock funding.
    pub reserved: Option<String>,
}

impl MwbCoin {
    pub fn spendable_at(&self, tip: u64) -> bool {
        self.spent_height.is_none()
            && self.reserved.is_none()
            && (!self.coinbase || tip >= self.height + COINBASE_MATURITY)
    }

    pub fn to_spendable(&self) -> Result<Spendable> {
        Ok(Spendable {
            commitment: self.commitment,
            value: self.value,
            blinding: scalar_from_bytes(&self.blinding).ok_or_else(|| anyhow!("stored blinding is not canonical"))?,
            owner_secret: scalar_from_bytes(&self.owner_secret)
                .ok_or_else(|| anyhow!("stored owner secret is not canonical"))?,
        })
    }
}

/// A payment from the account (a send, or a swap's lock funding).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbPayment {
    /// Its kernel's id: a block carrying that kernel confirmed it.
    pub kernel: [u8; 32],
    /// The transaction, for broadcasting again until a block takes it.
    pub tx_hex: Option<String>,
    pub inputs: Vec<Point32>,
    pub to: String,
    pub amount: u64,
    pub fee: u64,
    /// Unix seconds.
    pub created: u64,
    /// When the last broadcast was (unix seconds).
    #[serde(default)]
    pub sent: u64,
    pub height: Option<u64>,
    /// A block spent one of its inputs some other way.
    pub failed_at: Option<u64>,
    pub label: String,
}

impl MwbPayment {
    pub fn pending(&self) -> bool {
        self.height.is_none() && self.failed_at.is_none()
    }
    pub fn reservation(&self) -> String {
        format!("pay:{}", hex::encode(self.kernel))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MwbBalance {
    pub spendable: u64,
    /// Coinbase outputs still maturing.
    pub immature: u64,
    /// Set aside for a payment waiting for a block or a swap's lock.
    pub reserved: u64,
}

/// What one scanned block changed.
#[derive(Debug, Default)]
pub struct ScanDelta {
    /// New coins: (value, coinbase), change of our own payments included.
    pub received: Vec<(u64, bool)>,
    pub spent: usize,
    /// Payments a block confirmed (kernel ids).
    pub confirmed: Vec<[u8; 32]>,
}

impl ScanDelta {
    pub fn changed(&self) -> bool {
        !self.received.is_empty() || self.spent > 0 || !self.confirmed.is_empty()
    }
}

/// The parts of a block a scan reads, so tests need not mine one.
pub struct BlockView<'a> {
    pub height: u64,
    pub id: [u8; 32],
    pub prev: [u8; 32],
    pub outputs: Vec<(&'a Output, bool)>,
    pub spent: HashSet<Point32>,
    pub kernels: HashSet<[u8; 32]>,
}

impl<'a> BlockView<'a> {
    pub fn of(height: u64, b: &'a Batch) -> Self {
        let body = b.body.outputs().map(|o| (o, false));
        let reward = b.coinbase.iter().flat_map(|c| c.outputs.outputs.iter()).map(|o| (o, true));
        Self {
            height,
            id: b.extension.final_hash,
            prev: b.prev_header_hash,
            outputs: body.chain(reward).collect(),
            spent: b.body.input_commitments().copied().collect(),
            kernels: b.body.kernel_ids().into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MwbAccount {
    pub coins: Vec<MwbCoin>,
    pub payments: Vec<MwbPayment>,
    /// The next height to scan.
    pub scanned: u64,
    /// Where scanning began, and where a reorganisation deeper than
    /// [`REORG_WINDOW`] starts again from.
    #[serde(default)]
    pub birth: u64,
    /// (height, block id) of recently scanned blocks.
    pub recent: Vec<(u64, [u8; 32])>,
}

impl MwbAccount {
    /// An account that starts scanning at `height` (a fresh phrase has no
    /// history before the chain's tip; a restored one starts at 0).
    pub fn starting_at(height: u64) -> Self {
        Self { scanned: height, birth: height, ..Self::default() }
    }

    fn recorded(&self, height: u64) -> Option<[u8; 32]> {
        self.recent.iter().find(|(h, _)| *h == height).map(|(_, id)| *id)
    }

    /// Applies one block, which must be at `self.scanned`.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  b.height = scanned
    /// Post: Err(REORG)  ⇔ recorded(b.height − 1) = Some(x) ∧ x ≠ b.prev   (nothing changed)
    ///       Ok(d)       ⇒ scanned' = b.height + 1
    ///                     ∧ every output of b that opens with keys is a coin (once)
    ///                     ∧ every coin b spends has spent_height = b.height, unreserved
    ///                     ∧ every payment whose kernel b carries has height = b.height
    ///                     ∧ every other pending payment whose input b spends has failed_at = b.height
    /// ```
    pub fn scan(&mut self, keys: &WalletKeys, b: &BlockView) -> Result<ScanDelta> {
        if b.height != self.scanned {
            bail!("expected midwimble block {}, got {}", self.scanned, b.height);
        }
        if b.height > 0 {
            if let Some(prev) = self.recorded(b.height - 1) {
                if prev != b.prev {
                    bail!("{REORG} below height {}", b.height);
                }
            }
        }
        let mut d = ScanDelta::default();
        for p in &mut self.payments {
            if !p.pending() {
                continue;
            }
            if b.kernels.contains(&p.kernel) {
                p.height = Some(b.height);
                p.tx_hex = None;
                d.confirmed.push(p.kernel);
            } else if p.inputs.iter().any(|c| b.spent.contains(c)) {
                p.failed_at = Some(b.height);
                p.tx_hex = None;
            }
        }
        for c in &mut self.coins {
            if c.spent_height.is_none() && b.spent.contains(&c.commitment) {
                c.spent_height = Some(b.height);
                c.reserved = None;
                d.spent += 1;
            }
        }
        // A payment that failed releases whatever of its coins are left.
        let failed: HashSet<String> = self
            .payments
            .iter()
            .filter(|p| p.failed_at == Some(b.height))
            .map(MwbPayment::reservation)
            .collect();
        for c in &mut self.coins {
            if c.reserved.as_ref().is_some_and(|r| failed.contains(r)) {
                c.reserved = None;
            }
        }
        let known: HashSet<Point32> = self.coins.iter().map(|c| c.commitment).collect();
        for (output, coinbase) in &b.outputs {
            if known.contains(&output.commitment) {
                continue;
            }
            if let Some(owned) = scan_output(keys, output) {
                self.coins.push(MwbCoin {
                    commitment: output.commitment,
                    value: owned.value,
                    blinding: owned.blinding.to_bytes(),
                    owner_secret: owned.owner_secret.to_bytes(),
                    height: b.height,
                    coinbase: *coinbase,
                    spent_height: None,
                    reserved: None,
                });
                d.received.push((owned.value, *coinbase));
            }
        }
        self.recent.push((b.height, b.id));
        if self.recent.len() > REORG_WINDOW {
            self.recent.remove(0);
        }
        self.scanned = b.height + 1;
        Ok(d)
    }

    /// Forgets everything learned from blocks at or above `height`. A
    /// payment those blocks settled waits for a block again, its coins
    /// reserved for it once more.
    pub fn rollback_to(&mut self, height: u64) {
        self.coins.retain(|c| c.height < height);
        for c in &mut self.coins {
            if c.spent_height.is_some_and(|h| h >= height) {
                c.spent_height = None;
            }
        }
        for p in &mut self.payments {
            if p.height.is_some_and(|h| h >= height) || p.failed_at.is_some_and(|h| h >= height) {
                p.height = None;
                p.failed_at = None;
                let tag = p.reservation();
                for c in &mut self.coins {
                    if c.spent_height.is_none() && p.inputs.contains(&c.commitment) {
                        c.reserved = Some(tag.clone());
                    }
                }
            }
        }
        self.recent.retain(|(h, _)| *h < height);
        self.scanned = self.scanned.min(height);
    }

    /// Where to resume after a reorganisation at `scanned`: ten blocks back
    /// while the block below that is one whose id is kept (so the next scan
    /// checks it), else from the account's birth.
    pub fn step_back(&mut self) {
        let to = self.scanned.saturating_sub(ROLLBACK_STEP);
        match self.recent.first() {
            Some((oldest, _)) if to > *oldest => self.rollback_to(to),
            _ => self.rollback_to(self.birth),
        }
    }

    pub fn balance(&self, tip: u64) -> MwbBalance {
        let mut b = MwbBalance::default();
        for c in self.coins.iter().filter(|c| c.spent_height.is_none()) {
            if c.reserved.is_some() {
                b.reserved += c.value;
            } else if c.coinbase && tip < c.height + COINBASE_MATURITY {
                b.immature += c.value;
            } else {
                b.spendable += c.value;
            }
        }
        b
    }

    /// Picks coins, largest first, covering `amount` plus the fee
    /// `fee_for(n)` for however many end up chosen.
    pub fn select(&self, amount: u64, tip: u64, fee_for: impl Fn(u64) -> u64) -> Result<(Vec<Spendable>, u64)> {
        if amount == 0 {
            bail!("amount must be positive");
        }
        let mut candidates: Vec<&MwbCoin> = self.coins.iter().filter(|c| c.spendable_at(tip)).collect();
        candidates.sort_by(|a, b| b.value.cmp(&a.value));
        let mut chosen = Vec::new();
        let mut total = 0u64;
        for c in candidates {
            let need = amount.checked_add(fee_for(chosen.len().max(1) as u64)).ok_or_else(|| anyhow!("amount overflow"))?;
            if total >= need {
                break;
            }
            chosen.push(c.to_spendable()?);
            total += c.value;
        }
        let fee = fee_for(chosen.len().max(1) as u64);
        let need = amount.checked_add(fee).ok_or_else(|| anyhow!("amount overflow"))?;
        if total < need {
            bail!("not enough spendable MDW: have {total}, need {need} (fee included)");
        }
        Ok((chosen, fee))
    }

    /// Sets coins aside for `why`. Fails, changing nothing, if any is
    /// already spent or reserved.
    pub fn reserve(&mut self, commitments: &[Point32], why: &str) -> Result<()> {
        for c in commitments {
            let coin = self
                .coins
                .iter()
                .find(|x| &x.commitment == c)
                .ok_or_else(|| anyhow!("coin {} is not in the account", hex::encode(c)))?;
            if coin.spent_height.is_some() || coin.reserved.is_some() {
                bail!("coin {} is no longer free to spend", hex::encode(c));
            }
        }
        for coin in self.coins.iter_mut().filter(|x| commitments.contains(&x.commitment)) {
            coin.reserved = Some(why.to_string());
        }
        Ok(())
    }

    /// Releases every unspent coin reserved for `why`.
    pub fn release(&mut self, why: &str) {
        for c in &mut self.coins {
            if c.spent_height.is_none() && c.reserved.as_deref() == Some(why) {
                c.reserved = None;
            }
        }
    }

    /// Records a transaction about to be broadcast and moves its inputs'
    /// reservation to it (from `was`, a swap's, if given).
    pub fn record_payment(&mut self, tx: &Transaction, to: &str, amount: u64, label: &str, was: Option<&str>) -> Result<[u8; 32]> {
        let kernel = tx.body.kernels.first().ok_or_else(|| anyhow!("a payment without a kernel"))?.id();
        let inputs: Vec<Point32> = tx.input_commitments().copied().collect();
        let p = MwbPayment {
            kernel,
            tx_hex: Some(hex::encode(bincode::serialize(tx)?)),
            inputs: inputs.clone(),
            to: to.to_string(),
            amount,
            fee: tx.fee()?,
            created: now_secs(),
            sent: 0,
            height: None,
            failed_at: None,
            label: label.to_string(),
        };
        let tag = p.reservation();
        for c in self.coins.iter_mut().filter(|c| inputs.contains(&c.commitment)) {
            if c.spent_height.is_none() && (c.reserved.is_none() || c.reserved.as_deref() == was) {
                c.reserved = Some(tag.clone());
            }
        }
        if !self.payments.iter().any(|x| x.kernel == kernel) {
            self.payments.push(p);
        }
        Ok(kernel)
    }

    /// Builds a payment of `amount` to `to`, paying the relay minimum, and
    /// records it. The caller saves before broadcasting.
    pub fn build_send(&mut self, keys: &WalletKeys, to: &StealthAddress, amount: u64, tip: u64) -> Result<Transaction> {
        let (spend, fee) = self.select(amount, tip, |n| crate::mwb::min_fee(n, 2, 1))?;
        let built = build_transaction(&spend, &[Payment { to: *to, value: amount }], &keys.address(), fee, 0)?;
        self.record_payment(&built.tx, &to.encode(), amount, "send", None)?;
        Ok(built.tx)
    }

    /// Payments waiting for a block that are due to be broadcast again.
    pub fn due_for_broadcast(&self, now: u64, every_secs: u64) -> Vec<([u8; 32], Transaction)> {
        self.payments
            .iter()
            .filter(|p| p.pending() && now.saturating_sub(p.sent) >= every_secs)
            .filter_map(|p| {
                let bytes = hex::decode(p.tx_hex.as_ref()?).ok()?;
                Some((p.kernel, bincode::deserialize(&bytes).ok()?))
            })
            .collect()
    }

    pub fn mark_sent(&mut self, kernel: &[u8; 32], now: u64) {
        if let Some(p) = self.payments.iter_mut().find(|p| &p.kernel == kernel) {
            p.sent = now;
        }
    }

    /// Forgets a payment the node refused outright (it never left this
    /// wallet) and releases its coins.
    pub fn discard_payment(&mut self, kernel: &[u8; 32]) {
        if let Some(p) = self.payments.iter().find(|p| &p.kernel == kernel && p.pending()) {
            let tag = p.reservation();
            self.release(&tag);
        }
        self.payments.retain(|p| !(&p.kernel == kernel && p.height.is_none()));
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Parses a 32-byte scalar a counterparty or the journal handed us.
pub fn scalar(bytes: &[u8; 32], what: &str) -> Result<Scalar> {
    scalar_from_bytes(bytes).ok_or_else(|| anyhow!("{what} is not a canonical scalar"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use midwimble::core::mw::stealth::new_output;

    fn block<'a>(height: u64, id: u8, prev: u8, outs: &'a [Output], spent: &[Point32], kernels: &[[u8; 32]]) -> BlockView<'a> {
        BlockView {
            height,
            id: [id; 32],
            prev: [prev; 32],
            outputs: outs.iter().map(|o| (o, false)).collect(),
            spent: spent.iter().copied().collect(),
            kernels: kernels.iter().copied().collect(),
        }
    }

    #[test]
    fn finds_its_coins_spends_them_and_follows_a_reorg() {
        let keys = WalletKeys::random();
        let stranger = WalletKeys::random();
        let mine = new_output(&keys.address(), 5_000_000).unwrap().output;
        let theirs = new_output(&stranger.address(), 7_000_000).unwrap().output;
        let mut acct = MwbAccount::starting_at(10);

        let outs = [mine.clone(), theirs];
        let d = acct.scan(&keys, &block(10, 1, 0, &outs, &[], &[])).unwrap();
        assert_eq!(d.received, vec![(5_000_000, false)]);
        assert_eq!(acct.balance(11).spendable, 5_000_000);

        // A block that does not build on block 10 is a reorganisation.
        let err = acct.scan(&keys, &block(11, 2, 9, &[], &[], &[])).unwrap_err();
        assert!(err.to_string().contains(REORG));
        assert_eq!(acct.scanned, 11, "nothing changed");

        let d = acct.scan(&keys, &block(11, 2, 1, &[], &[mine.commitment], &[])).unwrap();
        assert_eq!(d.spent, 1);
        assert_eq!(acct.balance(12).spendable, 0);
        for h in 12..30u64 {
            let prev = if h == 12 { 2 } else { h as u8 + 99 };
            acct.scan(&keys, &block(h, h as u8 + 100, prev, &[], &[], &[])).unwrap();
        }

        // A reorganisation at 30: ten blocks back, where the kept ids still
        // check the new chain. The coin's spend at 11 stands.
        acct.step_back();
        assert_eq!(acct.scanned, 20);
        assert_eq!(acct.balance(20).spendable, 0);
        // The block that spent it is lost too: past the oldest kept id the
        // account starts again from its birth, and the coin comes back.
        acct.rollback_to(11);
        assert_eq!(acct.balance(12).spendable, 5_000_000);
        acct.step_back();
        assert_eq!(acct.scanned, 10);
        assert!(acct.coins.is_empty(), "block 10 is scanned again");
    }

    #[test]
    fn a_payment_reserves_its_coins_until_a_block_settles_it() {
        let keys = WalletKeys::random();
        let a = new_output(&keys.address(), 3_000_000).unwrap().output;
        let b = new_output(&keys.address(), 4_000_000).unwrap().output;
        let mut acct = MwbAccount::starting_at(0);
        acct.scan(&keys, &block(0, 1, 0, &[a, b], &[], &[])).unwrap();

        let to = WalletKeys::random().address();
        let tx = acct.build_send(&keys, &to, 4_500_000, 1).unwrap();
        tx.validate(midwimble::core::mw::transaction::Context::Relay).unwrap();
        let bal = acct.balance(1);
        assert_eq!(bal.reserved, 7_000_000, "both coins were needed");
        assert!(acct.select(1, 1, |_| 0).is_err(), "nothing left to select");

        // The block takes the payment: inputs spent, change found.
        let kernel = tx.body.kernels[0].id();
        let spent: Vec<Point32> = tx.input_commitments().copied().collect();
        let outs: Vec<Output> = tx.outputs().cloned().collect();
        let d = acct.scan(&keys, &block(1, 2, 1, &outs, &spent, &[kernel])).unwrap();
        assert_eq!(d.confirmed, vec![kernel]);
        let change = 7_000_000 - 4_500_000 - crate::mwb::min_fee(2, 2, 1);
        assert_eq!(acct.balance(2), MwbBalance { spendable: change, immature: 0, reserved: 0 });

        // Undo that block: the payment waits again, its coins reserved again.
        acct.rollback_to(1);
        assert!(acct.payments[0].pending());
        assert_eq!(acct.balance(2).reserved, 7_000_000);
    }

    #[test]
    fn a_payment_whose_coin_went_elsewhere_fails_and_frees_the_rest() {
        let keys = WalletKeys::random();
        let a = new_output(&keys.address(), 3_000_000).unwrap().output;
        let b = new_output(&keys.address(), 4_000_000).unwrap().output;
        let mut acct = MwbAccount::starting_at(0);
        acct.scan(&keys, &block(0, 1, 0, &[a.clone(), b], &[], &[])).unwrap();
        acct.build_send(&keys, &WalletKeys::random().address(), 4_500_000, 1).unwrap();
        // Another copy of this account spent coin `a` some other way.
        acct.scan(&keys, &block(1, 2, 1, &[], &[a.commitment], &[])).unwrap();
        assert!(acct.payments[0].failed_at.is_some());
        assert_eq!(acct.balance(2), MwbBalance { spendable: 4_000_000, immature: 0, reserved: 0 });
        // A payment one coin covers leaves the other free throughout.
        let c = new_output(&keys.address(), 9_000_000).unwrap().output;
        acct.scan(&keys, &block(2, 3, 2, &[c], &[], &[])).unwrap();
        acct.build_send(&keys, &WalletKeys::random().address(), 1_000_000, 3).unwrap();
        assert_eq!(acct.balance(3).spendable, 4_000_000, "largest first: the 9M coin pays");
    }

    #[test]
    fn reservations_do_not_overlap() {
        let keys = WalletKeys::random();
        let a = new_output(&keys.address(), 3_000_000).unwrap().output;
        let mut acct = MwbAccount::starting_at(0);
        acct.scan(&keys, &block(0, 1, 0, &[a.clone()], &[], &[])).unwrap();
        acct.reserve(&[a.commitment], "swap:1").unwrap();
        assert!(acct.reserve(&[a.commitment], "swap:2").is_err());
        acct.release("swap:2");
        assert_eq!(acct.balance(1).reserved, 3_000_000);
        acct.release("swap:1");
        assert_eq!(acct.balance(1).spendable, 3_000_000);
    }
}
