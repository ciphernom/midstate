//! Renting midwimble mining bonds, from the desktop wallet: offering one of
//! this wallet's coins as a bond for rent, and renting one for this wallet's
//! miner. `midstate::core::rental` holds the covenant and the announcements;
//! neither chain's consensus changed for any of it.
//!
//! # Offering
//!
//! [`Offer`] terms are a lock height, a rent and a bond size. The wallet
//! makes a fresh MSS tree for every offer ([`OFFER_KEY_HEIGHT`]): it is the
//! bond's owner key, the key that cancels, and, at its own address, where
//! the rent arrives, so the rent shows up as ordinary coins of this wallet.
//! One send funds the offer's coin and announces it beside the coin. From
//! then on the owner need not be online: whoever takes it locks the coin as
//! a bond on their mining key and pays the rent in the same transaction. The
//! wallet notices the coin go, finds the take on chain, and keeps a bond
//! record for it ([`crate::bond::OwnerKind::WalletTree`]), so it can unbond
//! the coin when the lock ends.
//!
//! # Renting
//!
//! The wallet reads the offers announced over the last [`MARKET_LOOKBACK`]
//! blocks and lists those whose coin is still there. Taking one is a single
//! send of this wallet's own: its coins pay the rent and the fee, the offer's
//! coin becomes the bond on this wallet's mining key. The bond is recorded
//! at once ([`crate::bond::OwnerKind::Rented`]) and registered with midwimble
//! as soon as it confirms; the miner then signs with it like any other.
//!
//! Two wallets can go for one offer at once, and only one of them gets it.
//! The loser has signed nothing if it finds the offer gone before its reveal,
//! which the wallet checks; the take prefers this wallet's MSS coins, whose
//! keys can sign again, so a loss after that costs one MSS leaf rather than
//! one-time keys.

use crate::bond;
use crate::sendplan::required_fee;
use anyhow::{bail, Context, Result};
use midstate::core::rental::Offer;
use midstate::core::{compute_address, decompose_value, wots, InputReveal, OutputData, Predicate, Transaction, Witness};
use midstate::wallet::Wallet;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Height of an offer's own MSS tree: 32 signatures, for a cancel or an
/// unbond and for spending the rent, with room for retries.
pub const OFFER_KEY_HEIGHT: u32 = 5;
/// The shortest rental the wallet offers, in days of mining.
pub const MIN_DAYS: u64 = 7;
/// How far back the market reads announcements: ten weeks of midstate.
pub const MARKET_LOOKBACK: u64 = 100_800;
/// Blocks the market reads per two-second tick while it catches up: the
/// window in under two minutes, without holding the actor up for long.
pub const SCAN_BUDGET: u64 = 2_000;
/// Blocks after which a take whose offer went to someone else is given up.
pub const LOST_AFTER: u64 = 30;
/// A day of midstate, for registration's wait.
const DAY_SECS: u64 = 86_400;

// ── Records ──────────────────────────────────────────────────────────────

/// An offer this wallet made.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyOffer {
    pub offer: Offer,
    /// The offer coin's salt: it exists nowhere else until the funding send
    /// confirms, so it is written before that send's commit goes out.
    #[serde(with = "hex32")]
    pub salt: [u8; 32],
    pub created_at: u64,
    #[serde(default, with = "opt_hex32")]
    pub fund_commitment: Option<[u8; 32]>,
    /// "funding" | "listed" | "taken" | "cancelling" | "cancelled" | "failed"
    pub status: String,
    #[serde(default, with = "opt_hex32")]
    pub cancel_commitment: Option<[u8; 32]>,
    /// The owner key's signature over `cancel_commitment`, kept so a retried
    /// reveal sends the same one.
    #[serde(default)]
    pub cancel_sig: Option<String>,
    /// Who took it, and what they paid.
    #[serde(default, with = "opt_hex32")]
    pub taken_by: Option<[u8; 32]>,
    #[serde(default)]
    pub rent_paid: u64,
    /// Height from which to look for the transaction that spent the coin.
    #[serde(default)]
    pub scan_from: u64,
    #[serde(default)]
    pub note: String,
}

impl MyOffer {
    pub fn coin_id(&self) -> [u8; 32] {
        self.offer.coin_id(&self.salt)
    }
    pub fn open(&self) -> bool {
        matches!(self.status.as_str(), "funding" | "listed" | "cancelling")
    }
}

/// A bond this wallet rented, from the take until it is settled. The bond
/// itself is in the bond records.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyRental {
    pub offer: Offer,
    #[serde(with = "hex32")]
    pub offer_salt: [u8; 32],
    #[serde(with = "hex32")]
    pub mining_key: [u8; 32],
    #[serde(with = "hex32")]
    pub bond_salt: [u8; 32],
    #[serde(with = "hex32")]
    pub take_commitment: [u8; 32],
    pub created_at: u64,
    /// "taking" | "rented" | "lost"
    pub status: String,
    /// Midstate height at which the offer was first seen gone while ours
    /// was still on its way.
    #[serde(default)]
    pub gone_at: u64,
    #[serde(default)]
    pub note: String,
}

impl MyRental {
    pub fn offer_coin(&self) -> [u8; 32] {
        self.offer.coin_id(&self.offer_salt)
    }
    pub fn bond_coin(&self) -> [u8; 32] {
        midstate::core::compute_coin_id(&self.offer.bond_address(&self.mining_key), self.offer.bond, &self.bond_salt)
    }
}

/// Offers and rentals, persisted beside the wallet.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RentalBook {
    pub offers: Vec<MyOffer>,
    pub rentals: Vec<MyRental>,
}

impl RentalBook {
    pub fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("rentals")
    }

    /// Missing file: nothing yet. An unreadable one is set aside, never
    /// overwritten: it may hold the only copy of an offer coin's salt.
    pub fn load(wallet_path: &Path) -> (Self, Option<String>) {
        let path = Self::path(wallet_path);
        let Ok(raw) = std::fs::read(&path) else { return (Self::default(), None) };
        match serde_json::from_slice(&raw) {
            Ok(book) => (book, None),
            Err(e) => {
                let aside = path.with_extension(format!("rentals.unreadable-{}", bond::now_secs()));
                let moved = std::fs::rename(&path, &aside).is_ok();
                let text = if moved {
                    format!("The bond rental records could not be read ({e}); the file was kept as {}.", aside.display())
                } else {
                    format!("The bond rental records at {} could not be read: {e}", path.display())
                };
                (Self::default(), Some(text))
            }
        }
    }

    /// Through a temporary file, as the bond records are.
    pub fn save(&self, wallet_path: &Path) -> Result<()> {
        let path = Self::path(wallet_path);
        let tmp = path.with_extension("rentals.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?).context("writing bond rental records")?;
        std::fs::rename(&tmp, &path).context("saving bond rental records")?;
        Ok(())
    }

    pub fn offer_by_coin(&self, coin: &[u8; 32]) -> Option<&MyOffer> {
        self.offers.iter().find(|o| &o.coin_id() == coin)
    }
    pub fn by_cancel(&self, commitment: &[u8; 32]) -> Option<&MyOffer> {
        self.offers.iter().find(|o| o.cancel_commitment.as_ref() == Some(commitment))
    }
    pub fn by_take(&self, commitment: &[u8; 32]) -> Option<&MyRental> {
        self.rentals.iter().find(|r| &r.take_commitment == commitment)
    }
}

// ── The market ───────────────────────────────────────────────────────────

/// An offer read off the chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Listing {
    pub offer: Offer,
    pub salt: [u8; 32],
    pub coin_id: [u8; 32],
    /// Where it was announced.
    pub height: u64,
    /// Its coin is unspent.
    pub live: bool,
}

/// Every offer announced in the window the wallet has read.
#[derive(Clone, Debug, Default)]
pub struct Market {
    /// Next height to read.
    pub cursor: u64,
    pub listings: Vec<Listing>,
}

impl Market {
    /// Reads one transaction's announcements.
    pub fn ingest(&mut self, height: u64, tx: &Transaction) {
        for a in midstate::core::rental::announced(tx) {
            self.add(height, a);
        }
    }

    /// Lists one announced offer, once.
    pub fn add(&mut self, height: u64, a: midstate::core::rental::Announced) {
        if !self.listings.iter().any(|l| l.coin_id == a.coin_id) {
            self.listings.push(Listing { offer: a.offer, salt: a.salt, coin_id: a.coin_id, height, live: true });
        }
    }

    pub fn find(&self, coin_id: &[u8; 32]) -> Option<&Listing> {
        self.listings.iter().find(|l| &l.coin_id == coin_id)
    }
}

/// Whole days a bond locked until `until` mines for a renter who takes it
/// at `now`: midwimble counts it once registered, about a day on, until
/// only its unbonding month (and a week's clock margin) is left.
pub fn mining_days(until: u64, now: u64) -> u64 {
    let start = now + DAY_SECS;
    bond::midwimble::counts_until(until, now)
        .filter(|end| *end > start)
        .map_or(0, |end| (end - start) / DAY_SECS)
}

/// Whole days the coin of an offer locked until `until` stays out of its
/// owner's hands, from midstate height `height` at unix time `now`, a block
/// a minute. A node still syncing reports a low height, so the height is
/// never taken for less than the schedule says (less midwimble's week of
/// margin).
pub fn locked_days(until: u64, height: u64, now: u64) -> u64 {
    let scheduled = bond::midwimble::est_midstate_height(now).saturating_sub(7 * 1440);
    until.saturating_sub(height.max(scheduled)).div_ceil(1440)
}

// ── Taking ───────────────────────────────────────────────────────────────

/// A take, planned and unsigned: this wallet's coins pay the rent and the
/// fee, the offer's coin (the last input) funds the bond.
#[derive(Clone, Debug)]
pub struct TakePlan {
    pub input_coin_ids: Vec<[u8; 32]>,
    pub outputs: Vec<OutputData>,
    /// (output index, seed) of the change, as `PendingCommit` keeps it.
    pub change_seeds: Vec<(usize, [u8; 32])>,
    pub bond_salt: [u8; 32],
    pub fee: u64,
}

/// Plans a take of `offer` (its coin `offer_coin`) for `mining_key`,
/// spending from `prefer` if those coins cover it, else from `live`.
pub fn plan_take(
    w: &mut Wallet,
    live: &[[u8; 32]],
    prefer: &[[u8; 32]],
    offer: &Offer,
    offer_coin: [u8; 32],
    mining_key: &[u8; 32],
) -> Result<TakePlan> {
    if offer.rent == 0 {
        bail!("this offer asks no rent; the wallet takes only offers that do");
    }
    let bond_salt: [u8; 32] = rand::random();
    let mut fixed = vec![OutputData::Standard {
        address: offer.bond_address(mining_key),
        value: offer.bond,
        salt: bond_salt,
    }];
    let pay_to = offer.pay_to();
    for value in decompose_value(offer.rent) {
        fixed.push(OutputData::Standard { address: pay_to, value, salt: rand::random() });
    }
    let mut fee = 100u64;
    for _ in 0..32 {
        let needed = offer.rent + fee;
        let selected = w.select_coins(needed, prefer).or_else(|_| w.select_coins(needed, live))?;
        let in_sum: u64 = selected.iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum();
        if in_sum < needed {
            bail!("not enough coins: the rent and fee come to {needed}, and {in_sum} is spendable");
        }
        // The offer's coin is one more input to price.
        let change = in_sum - needed;
        let need = required_fee(selected.len() + 1, fixed.len() + decompose_value(change).len());
        if need <= fee {
            let mut outputs = fixed;
            let mut change_seeds = Vec::new();
            for value in decompose_value(change) {
                let seed = w.allocate_next_wots_seed()?;
                let address = compute_address(&wots::keygen(&seed));
                change_seeds.push((outputs.len(), seed));
                outputs.push(OutputData::Standard { address, value, salt: rand::random() });
            }
            let mut input_coin_ids = selected;
            input_coin_ids.push(offer_coin);
            return Ok(TakePlan { input_coin_ids, outputs, change_seeds, bond_salt, fee });
        }
        fee = need;
    }
    bail!("the take's fee did not settle")
}

/// Signs this wallet's inputs of a take whose commitment is `commitment`,
/// in the transaction's input order; the offer's own input is left to the
/// caller. One MSS signature serves every coin of its tree, as
/// `Wallet::sign_reveal` does; a WOTS key signs once, for all the coins at
/// its address.
pub fn sign_wallet_inputs(
    w: &mut Wallet,
    ids: &[[u8; 32]],
    commitment: &[u8; 32],
) -> Result<Vec<(InputReveal, Witness)>> {
    let mut cache: std::collections::HashMap<[u8; 32], Vec<u8>> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(coin) = w.find_coin(id).cloned() else { bail!("coin {} is not in this wallet", hex::encode(id)) };
        let input = InputReveal {
            predicate: Predicate::p2pk(&coin.owner_pk),
            value: coin.value,
            salt: coin.salt,
            commitment: if coin.value == 0 { coin.commitment } else { None },
        };
        let sig = match cache.get(&coin.owner_pk) {
            Some(sig) => sig.clone(),
            None => {
                let sig = w.sign_mix_input(id, commitment)?;
                cache.insert(coin.owner_pk, sig.clone());
                sig
            }
        };
        out.push((input, Witness::sig(sig)));
    }
    Ok(out)
}

// ── Cancelling ───────────────────────────────────────────────────────────

/// Exact serialized size of a cancel: the offer's coin, opened by an owner
/// signature of `sig_len` bytes, and `n_outputs` standard outputs.
pub fn cancel_reveal_size(offer: &Offer, sig_len: usize, n_outputs: usize) -> u64 {
    let tx = Transaction::Reveal {
        inputs: vec![offer.input([0u8; 32])],
        witnesses: vec![midstate::core::rental::cancel_witness(vec![0u8; sig_len])],
        outputs: (0..n_outputs)
            .map(|_| OutputData::Standard { address: [0u8; 32], value: 0, salt: [0u8; 32] })
            .collect(),
        salt: [0u8; 32],
    };
    bincode::serialized_size(&tx).unwrap_or(u64::MAX)
}

/// The fee of a cancel that pays the offer's coin back less the fee, and
/// the outputs that leaves; `None` if the coin cannot cover it. As
/// `bond::resolve_unbond_fee`.
pub fn resolve_cancel_fee(offer: &Offer, sig_len: usize) -> Option<(u64, Vec<u64>)> {
    let mut n_out = 1usize;
    loop {
        let size = cancel_reveal_size(offer, sig_len, n_out);
        let fee = size.saturating_mul(10).div_ceil(1024) + 20;
        if fee >= offer.bond {
            return None;
        }
        let denoms = decompose_value(offer.bond - fee);
        if denoms.len() <= n_out {
            return Some((fee, denoms));
        }
        n_out = denoms.len();
    }
}

mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(&s).map_err(serde::de::Error::custom)?;
        v.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

mod opt_hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(v) => s.serialize_some(&hex::encode(v)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        let Some(s) = Option::<String>::deserialize(d)? else { return Ok(None) };
        let v = hex::decode(&s).map_err(serde::de::Error::custom)?;
        v.try_into().map(Some).map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::rental::{announced, spent_by, take_witness, OfferSpend};
    use midstate::core::{compute_commitment, hash, script};
    use tempfile::NamedTempFile;

    fn offer() -> Offer {
        Offer { owner_pk: [0xab; 32], bonded_until: 600_000, rent: 1 << 20, bond: 1 << 37 }
    }

    fn wallet() -> (Wallet, NamedTempFile) {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();
        (Wallet::create(&path, b"pass").unwrap(), file)
    }

    /// Gives the wallet a coin it can spend, at a fresh one-time key.
    fn fund(w: &mut Wallet, value: u64, tag: u8) -> [u8; 32] {
        let seed = hash(&[tag; 32]);
        let salt = hash(&[tag ^ 0xff; 32]);
        w.import_coin(seed, value, salt, None).unwrap()
    }

    #[test]
    fn a_take_pays_rent_and_fee_from_the_wallet_and_the_bond_from_the_offer() {
        let (mut w, _f) = wallet();
        let coin = fund(&mut w, 1 << 22, 1);
        let o = offer();
        let offer_coin = o.coin_id(&[5; 32]);
        let key = [0x11; 32];
        let plan = plan_take(&mut w, &[coin], &[], &o, offer_coin, &key).unwrap();
        assert_eq!(plan.input_coin_ids, vec![coin, offer_coin], "the offer's coin goes last");
        let bond_out = OutputData::Standard { address: o.bond_address(&key), value: o.bond, salt: plan.bond_salt };
        assert_eq!(plan.outputs[0], bond_out);
        let rent: u64 = plan.outputs.iter().filter(|x| x.address() == o.pay_to()).map(|x| x.value()).sum();
        assert_eq!(rent, o.rent);
        let out_sum: u64 = plan.outputs.iter().map(|x| x.value()).sum();
        assert_eq!(out_sum + plan.fee, (1 << 22) + o.bond, "everything in goes out, but the fee");
        assert!(plan.outputs.iter().all(|x| x.value().is_power_of_two()));
        for (i, seed) in &plan.change_seeds {
            assert_eq!(plan.outputs[*i].address(), compute_address(&wots::keygen(seed)));
        }

        // Signed, it satisfies the covenant with the take witness.
        let outs: Vec<[u8; 32]> = plan.outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let c = compute_commitment(&plan.input_coin_ids, &outs, &[3; 32]);
        let signed = sign_wallet_inputs(&mut w, &plan.input_coin_ids[..1], &c).unwrap();
        assert_eq!(signed.len(), 1);
        assert!(w.find_coin(&coin).unwrap().wots_signed, "the one-time key is marked as used");
        let tx = Transaction::Reveal {
            inputs: vec![signed[0].0.clone(), o.input([5; 32])],
            witnesses: vec![signed[0].1.clone(), take_witness(&key)],
            outputs: plan.outputs.clone(),
            salt: [3; 32],
        };
        let Witness::ScriptInputs(items) = take_witness(&key);
        let ctx = script::ExecContext {
            commitment: &c,
            height: 354_000,
            outputs: &plan.outputs,
            input_value: o.bond,
            input_state: None,
            this_address: o.address(),
            sum_input_value: o.bond,
        };
        assert_eq!(script::execute_script(&o.script(), &items, &ctx), Ok(()));
        assert_eq!(
            spent_by(&tx, &o, &offer_coin),
            Some(OfferSpend::Taken { mining_key: key, bond_salt: Some(plan.bond_salt), rent_paid: o.rent })
        );
    }

    #[test]
    fn a_take_the_wallet_cannot_cover_is_refused() {
        let (mut w, _f) = wallet();
        let coin = fund(&mut w, 1 << 10, 2);
        let o = offer();
        assert!(plan_take(&mut w, &[coin], &[], &o, o.coin_id(&[5; 32]), &[0x11; 32]).is_err());
    }

    #[test]
    fn the_market_lists_offers_announced_beside_their_coin() {
        let o = offer();
        let salt = [9; 32];
        let funding = Transaction::Reveal {
            inputs: vec![],
            witnesses: vec![],
            outputs: vec![
                OutputData::Standard { address: o.address(), value: o.bond, salt },
                OutputData::DataBurn { payload: o.announcement(), value_burned: 0 },
            ],
            salt: [0; 32],
        };
        let mut m = Market::default();
        m.ingest(100, &funding);
        m.ingest(101, &funding);
        assert_eq!(m.listings.len(), 1, "one listing per coin");
        assert_eq!(m.listings[0].coin_id, o.coin_id(&salt));
        assert_eq!(m.listings[0].height, 100);
        assert_eq!(announced(&funding).len(), 1);

        // An announcement without its coin is nothing.
        let bare = Transaction::Reveal {
            inputs: vec![],
            witnesses: vec![],
            outputs: vec![OutputData::DataBurn { payload: o.announcement(), value_burned: 0 }],
            salt: [0; 32],
        };
        let mut m = Market::default();
        m.ingest(100, &bare);
        assert!(m.listings.is_empty());
    }

    #[test]
    fn a_cancel_pays_the_coin_back_less_its_fee() {
        let o = offer();
        let sig_len = bond::mss_sig_len(OFFER_KEY_HEIGHT);
        let (fee, denoms) = resolve_cancel_fee(&o, sig_len).unwrap();
        assert_eq!(denoms.iter().sum::<u64>() + fee, o.bond);
        let size = cancel_reveal_size(&o, sig_len, denoms.len());
        assert!(fee * 1024 >= 10 * size, "meets the mempool's floor");
        let tiny = Offer { bond: 4, ..o };
        assert!(resolve_cancel_fee(&tiny, sig_len).is_none());
    }

    #[test]
    fn mining_days_count_from_registration_and_stop_short_of_the_lock() {
        // After midwimble's early phase, whose shorter locks end with it.
        let now = 1_794_000_000;
        for days in [MIN_DAYS, 30, 90] {
            let until = bond::midwimble::until_for_days(now, days);
            let got = mining_days(until, now);
            assert!(got == days || got + 1 == days, "{days} days asked, {got} counted");
        }
        assert_eq!(mining_days(1, now), 0);
    }

    #[test]
    fn the_book_survives_a_round_trip() {
        let (w, f) = wallet();
        drop(w);
        let path = f.path().to_path_buf();
        let mut book = RentalBook::default();
        book.offers.push(MyOffer {
            offer: offer(),
            salt: [1; 32],
            created_at: 5,
            fund_commitment: Some([2; 32]),
            status: "funding".into(),
            cancel_commitment: None,
            cancel_sig: None,
            taken_by: None,
            rent_paid: 0,
            scan_from: 0,
            note: String::new(),
        });
        book.save(&path).unwrap();
        let (back, warning) = RentalBook::load(&path);
        assert!(warning.is_none());
        assert_eq!(back.offers.len(), 1);
        assert_eq!(back.offers[0].offer, offer());
        assert_eq!(back.offers[0].fund_commitment, Some([2; 32]));
        let _ = std::fs::remove_file(RentalBook::path(&path));
    }
}
