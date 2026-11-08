//! The walletd actor's side of bond rentals (`crate::rental`): offering this
//! wallet's coins as bonds for rent, taking offers for its miner, and keeping
//! both in step with the chain.
//!
//! A child of `service`, sharing the actor's state, as `service_mining` is.

use super::{now_secs, parse_hex32, Cmd, Internal, SendMeta, Service};
use crate::api::{RentalListingView, RentalOfferView, RentalsView, RentedBondView, SendStage, WalletEvent};
use crate::bond::{self, BondRecord, OwnerKind, RegStatus};
use crate::rental::{self, MyOffer, MyRental, RentalBook};
use crate::sendplan;
use anyhow::{anyhow, bail, Result};
use midstate::core::rental::{announced, cancel_witness, spent_by, take_witness, Offer, OfferSpend};
use midstate::core::{compute_address, compute_commitment, OutputData, Transaction};
use std::collections::HashSet;
use std::time::Duration;

/// Blocks searched per tick for the transaction that spent an offer.
const SPEND_SCAN_BUDGET: u64 = 2_000;

impl Service {
    pub(super) fn rentals_load(&mut self) {
        let (book, warning) = RentalBook::load(&self.wallet_path);
        self.rentals = book;
        if let Some(text) = warning {
            let _ = self.events.send(WalletEvent::Warning { text });
        }
    }

    fn rentals_save(&self) -> Result<()> {
        self.rentals.save(&self.wallet_path)
    }

    /// The owner keys of this wallet's MSS trees: an offer's key is one.
    fn own_tree_keys(&self) -> HashSet<[u8; 32]> {
        self.wallet.as_ref().map_or_else(HashSet::new, |w| w.mss_keys().iter().map(|k| k.master_pk).collect())
    }

    /// The longest rental midwimble would register now, in days of mining.
    fn max_rental_days(now: u64) -> u64 {
        let first = bond::midwimble::until_for_days(now, 0);
        bond::midwimble::horizon(now).saturating_sub(first) / 1440
    }

    // ── Views ────────────────────────────────────────────────────────────

    pub(super) async fn rentals_view(&mut self) -> Result<RentalsView> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let state = self.node.get_state().await;
        let now = now_secs();
        let ours = self.own_tree_keys();
        let min = bond::midwimble::min_bond(now);
        let taking: HashSet<[u8; 32]> =
            self.rentals.rentals.iter().filter(|r| r.status == "taking").map(|r| r.offer_coin()).collect();
        let mut listings: Vec<RentalListingView> = self
            .market
            .listings
            .iter()
            .filter(|l| l.live)
            .map(|l| {
                let o = &l.offer;
                let mining_days = rental::mining_days(o.bonded_until, now);
                let yours = ours.contains(&o.owner_pk);
                let why_not = if yours {
                    "one of your own offers".to_string()
                } else if taking.contains(&l.coin_id) {
                    "you are renting it now".to_string()
                } else if mining_days == 0 {
                    "too little of its lock is left for midwimble to count it".to_string()
                } else if o.bond < min {
                    format!("smaller than the {} midwimble counts", bond::gmds(min))
                } else if o.rent == 0 {
                    "it asks no rent".to_string()
                } else {
                    String::new()
                };
                RentalListingView {
                    id: hex::encode(l.coin_id),
                    bond: o.bond,
                    rent: o.rent,
                    bonded_until: o.bonded_until,
                    mining_days,
                    locked_days: rental::locked_days(o.bonded_until, state.height, now),
                    rent_per_day: o.rent / mining_days.max(1),
                    height: l.height,
                    yours,
                    takeable: why_not.is_empty(),
                    why_not,
                }
            })
            .collect();
        listings.sort_by_key(|l| (!l.takeable, l.rent_per_day, l.height));

        let offers = self
            .rentals
            .offers
            .iter()
            .rev()
            .map(|o| RentalOfferView {
                id: hex::encode(o.coin_id()),
                status: o.status.clone(),
                note: o.note.clone(),
                bond: o.offer.bond,
                rent: o.offer.rent,
                bonded_until: o.offer.bonded_until,
                mining_days: rental::mining_days(o.offer.bonded_until, now),
                taken_by: o.taken_by.map(hex::encode).unwrap_or_default(),
                rent_paid: o.rent_paid,
                can_cancel: o.status == "listed",
            })
            .collect();

        let rentals = self
            .rentals
            .rentals
            .iter()
            .rev()
            .map(|r| {
                let rec = self.bonds.find(&r.bond_coin());
                RentedBondView {
                    id: hex::encode(r.bond_coin()),
                    status: r.status.clone(),
                    note: r.note.clone(),
                    bond: r.offer.bond,
                    rent: r.offer.rent,
                    bonded_until: r.offer.bonded_until,
                    mining_days: rental::mining_days(r.offer.bonded_until, now),
                    reg_phase: rec.map(|b| b.reg.phase.clone()).unwrap_or_default(),
                    reg_note: rec.map(|b| b.reg.note.clone()).unwrap_or_default(),
                }
            })
            .collect();

        let start = state.height.saturating_sub(rental::MARKET_LOOKBACK).max(1);
        Ok(RentalsView {
            listings,
            offers,
            rentals,
            scanned: self.market.cursor.saturating_sub(start).min(rental::MARKET_LOOKBACK),
            window: state.height.saturating_sub(start),
            bond: self.bond_rules().registration_min,
            min_days: rental::MIN_DAYS,
            max_days: Self::max_rental_days(now),
            mining_key: self.mining_key().map(hex::encode).unwrap_or_default(),
        })
    }

    // ── Offering ─────────────────────────────────────────────────────────

    /// Offers one coin of this wallet as a mining bond for `days` of mining,
    /// for `rent`. Returns the offer's id (its coin's).
    pub(super) async fn post_offer(&mut self, days: u64, rent: u64) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — posting against a stale coin set would fail");
        }
        self.ensure_restore_done()?;
        if rent == 0 {
            bail!("set the rent you want for it");
        }
        let now = now_secs();
        let max = Self::max_rental_days(now);
        if days < rental::MIN_DAYS || days > max {
            bail!("rent it out for {} to {max} days of mining", rental::MIN_DAYS);
        }
        let bond = self.bond_rules().registration_min;
        if !bond.is_power_of_two() {
            bail!("midwimble's bond size is not known yet: wait for its node to start");
        }
        let until = bond::midwimble::until_for_days(now, days);
        let state = self.node.get_state().await;
        self.verify_mss_indices().await?;

        // The offer's own key: owner of the bond, the key that cancels, and
        // where the rent arrives.
        let w = self.wallet.as_mut().unwrap();
        let address = w.generate_mss(rental::OFFER_KEY_HEIGHT, Some("bond offer".into()))?;
        let owner_pk = w
            .mss_keys()
            .iter()
            .find(|k| compute_address(&k.master_pk) == address)
            .map(|k| k.master_pk)
            .ok_or_else(|| anyhow!("internal error: the offer's key was not kept"))?;
        let offer = Offer { owner_pk, bonded_until: until, rent, bond };
        let salt: [u8; 32] = rand::random();
        let outputs = vec![
            OutputData::Standard { address: offer.address(), value: bond, salt },
            OutputData::DataBurn { payload: offer.announcement(), value_burned: 0 },
        ];

        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let live: Vec<[u8; 32]> = w
            .coins()
            .iter()
            .filter(|c| !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id))
            .map(|c| c.coin_id)
            .collect();
        let plan = sendplan::plan_fixed_outputs(w, &live, outputs)?;
        let (commitment, _) = w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false)?;
        w.save()?;

        let id = offer.coin_id(&salt);
        self.rentals.offers.push(MyOffer {
            offer,
            salt,
            created_at: now,
            fund_commitment: Some(commitment),
            status: "funding".into(),
            cancel_commitment: None,
            cancel_sig: None,
            taken_by: None,
            rent_paid: 0,
            scan_from: 0,
            note: "Posting: the send that locks it is on its way (see the Send tab).".into(),
        });
        if let Err(e) = self.rentals_save() {
            // Nothing is broadcast yet: undo, rather than lock a coin whose
            // salt would be lost.
            self.rentals.offers.retain(|o| o.coin_id() != id);
            let w = self.wallet.as_mut().unwrap();
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
            return Err(e.context("could not record the offer, so nothing was sent"));
        }
        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "offering a bond for rent — solving commit proof-of-work".into(),
                amount: bond,
                fee: plan.fee,
                to: "bond offer".into(),
                updated_at: now,
            },
        );
        self.set_stage(commitment, SendStage::Committing, "offering a bond for rent");
        self.broadcast_commit(commitment).await?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(hex::encode(id))
    }

    /// Takes a listed offer of this wallet's back, before anyone rents it.
    pub(super) async fn cancel_offer(&mut self, id: &str) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing");
        }
        let coin = parse_hex32(id)?;
        let o = self.rentals.offer_by_coin(&coin).cloned().ok_or_else(|| anyhow!("no such offer in this wallet"))?;
        if o.status != "listed" {
            bail!("only a listed offer can be cancelled (this one is {})", o.status);
        }
        let state = self.node.get_state().await;
        if !state.coins.contains(&coin) {
            bail!("its coin has left the chain: someone has just rented it");
        }
        let w = self.wallet.as_mut().unwrap();
        let height = w
            .mss_keys()
            .iter()
            .find(|k| k.master_pk == o.offer.owner_pk)
            .map(|k| k.height)
            .ok_or_else(|| anyhow!("the offer's key is no longer in this wallet"))?;
        let (fee, denoms) = rental::resolve_cancel_fee(&o.offer, bond::mss_sig_len(height))
            .ok_or_else(|| anyhow!("the offer's coin cannot pay the fee to cancel it"))?;
        let seeds = w.allocate_next_wots_seeds(denoms.len())?;
        let mut outputs = Vec::with_capacity(denoms.len());
        let mut change_seeds = Vec::with_capacity(denoms.len());
        for (i, (value, seed)) in denoms.iter().zip(seeds).enumerate() {
            let address = compute_address(&midstate::core::wots::keygen(&seed));
            outputs.push(OutputData::Standard { address, value: *value, salt: rand::random() });
            change_seeds.push((i, seed));
        }
        let salt: [u8; 32] = rand::random();
        let hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(&[coin], &hashes, &salt);
        let now = now_secs();
        w.data.pending.push(midstate::wallet::PendingCommit {
            commitment,
            salt,
            input_coin_ids: vec![coin],
            outputs,
            change_seeds,
            created_at: now,
            reveal_not_before: now + 5 + rand::random::<u64>() % 11,
            is_consolidate: false,
        });
        w.save()?;
        let r = self.rentals.offers.iter_mut().find(|x| x.coin_id() == coin).expect("offer present");
        r.cancel_commitment = Some(commitment);
        r.cancel_sig = None;
        r.status = "cancelling".into();
        r.note = "Cancelling: the coin is on its way back (see the Send tab).".into();
        if let Err(e) = self.rentals_save() {
            let w = self.wallet.as_mut().unwrap();
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
            return Err(e.context("could not record the cancel, so nothing was sent"));
        }
        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "cancelling a bond offer — solving commit proof-of-work".into(),
                amount: o.offer.bond - fee,
                fee,
                to: "this wallet (offer cancelled)".into(),
                updated_at: now,
            },
        );
        self.set_stage(commitment, SendStage::Committing, "cancelling a bond offer");
        self.broadcast_commit(commitment).await?;
        Ok(hex::encode(commitment))
    }

    /// `try_reveal` for a cancel: the offer coin is a script coin, signed
    /// here with the offer's own key.
    pub(super) async fn reveal_cancel(&mut self, commitment: [u8; 32], pending: midstate::wallet::PendingCommit) {
        let Some(o) = self.rentals.by_cancel(&commitment).cloned() else { return };
        let coin = o.coin_id();
        let hashes: Vec<[u8; 32]> = pending.outputs.iter().map(|x| x.hash_for_commitment()).collect();
        if pending.input_coin_ids != [coin] || compute_commitment(&pending.input_coin_ids, &hashes, &pending.salt) != commitment {
            self.set_stage(commitment, SendStage::Failed, "the cancel's commit does not match its offer");
            return;
        }
        if !self.node.check_coin(coin).await {
            self.cancel_lost(commitment);
            return;
        }
        let sig = match o.cancel_sig.as_deref().and_then(|h| hex::decode(h).ok()) {
            Some(sig) => sig,
            None => {
                if let Err(e) = self.verify_mss_indices().await {
                    self.set_stage(commitment, SendStage::Failed, format!("could not sign the cancel: {e:#}. Nothing was spent."));
                    return;
                }
                let w = self.wallet.as_mut().unwrap();
                let sig = match w.auto_sign(&o.offer.owner_pk, &commitment) {
                    Ok(sig) => sig,
                    Err(e) => {
                        self.set_stage(commitment, SendStage::Failed, format!("could not sign the cancel: {e:#}. Nothing was spent."));
                        return;
                    }
                };
                if let Some(r) = self.rentals.offers.iter_mut().find(|x| x.coin_id() == coin) {
                    r.cancel_sig = Some(hex::encode(&sig));
                }
                if let Err(e) = self.rentals_save() {
                    self.set_stage(commitment, SendStage::Failed, format!("could not record the cancel's signature: {e:#}"));
                    return;
                }
                sig
            }
        };
        let back = pending.outputs[0].coin_id().expect("standard output");
        let tx = Transaction::Reveal {
            inputs: vec![o.offer.input(o.salt)],
            witnesses: vec![cancel_witness(sig)],
            outputs: pending.outputs.clone(),
            salt: pending.salt,
        };
        match self.node.send_transaction(tx).await {
            Ok(()) => {
                self.set_stage(commitment, SendStage::RevealPending, "cancel broadcast — waiting for confirmation");
                // Watch for the coin coming back, not for the offer's coin
                // leaving: a renter can spend that too.
                self.spawn_output_monitor(commitment, back, coin);
            }
            Err(e) => self.set_stage(
                commitment,
                SendStage::Failed,
                format!("cancel broadcast failed: {e}. The signed cancel is kept; retry when the node recovers."),
            ),
        }
    }

    /// A cancel overtaken by a renter: drop it; the watcher records the take.
    fn cancel_lost(&mut self, commitment: [u8; 32]) {
        if let Some(w) = self.wallet.as_mut() {
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
        }
        if let Some(r) = self.rentals.offers.iter_mut().find(|x| x.cancel_commitment == Some(commitment)) {
            if r.status == "cancelling" {
                r.status = "listed".into();
            }
            r.cancel_commitment = None;
        }
        let _ = self.rentals_save();
        self.set_stage(
            commitment,
            SendStage::Failed,
            "someone rented the bond before the cancel went through: it is theirs to mine with, and the rent is yours",
        );
    }

    // ── Renting ──────────────────────────────────────────────────────────

    /// Rents a listed offer for this wallet's miner. Returns the bond's id.
    pub(super) async fn take_offer(&mut self, id: &str) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — renting against a stale coin set would fail");
        }
        self.ensure_restore_done()?;
        let coin = parse_hex32(id)?;
        let listing = self.market.find(&coin).cloned().ok_or_else(|| anyhow!("that offer is not on the market"))?;
        let offer = listing.offer;
        let mining_key = self.mining_key()?;
        if self.own_tree_keys().contains(&offer.owner_pk) {
            bail!("this is one of your own offers");
        }
        if self.rentals.rentals.iter().any(|r| r.offer_coin() == coin && r.status == "taking") {
            bail!("you are renting this one already");
        }
        let now = now_secs();
        if rental::mining_days(offer.bonded_until, now) == 0 {
            bail!("too little of its lock is left for midwimble to count it");
        }
        if offer.bond < bond::midwimble::min_bond(now) {
            bail!("it is smaller than the {} midwimble counts", bond::gmds(bond::midwimble::min_bond(now)));
        }
        let state = self.node.get_state().await;
        if !state.coins.contains(&coin) {
            bail!("someone has just rented it");
        }
        self.verify_mss_indices().await?;

        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let trees: HashSet<[u8; 32]> = w.mss_keys().iter().map(|k| compute_address(&k.master_pk)).collect();
        let usable: Vec<(&midstate::wallet::WalletCoin, bool)> = w
            .coins()
            .iter()
            .filter(|c| !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id))
            .map(|c| (c, trees.contains(&c.address)))
            .collect();
        let live: Vec<[u8; 32]> = usable.iter().map(|(c, _)| c.coin_id).collect();
        let prefer: Vec<[u8; 32]> = usable.iter().filter(|(_, mss)| *mss).map(|(c, _)| c.coin_id).collect();
        let plan = rental::plan_take(w, &live, &prefer, &offer, coin, &mining_key)?;
        let salt: [u8; 32] = rand::random();
        let hashes: Vec<[u8; 32]> = plan.outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(&plan.input_coin_ids, &hashes, &salt);
        w.data.pending.push(midstate::wallet::PendingCommit {
            commitment,
            salt,
            input_coin_ids: plan.input_coin_ids.clone(),
            outputs: plan.outputs.clone(),
            change_seeds: plan.change_seeds.clone(),
            created_at: now,
            reveal_not_before: now + 5 + rand::random::<u64>() % 11,
            is_consolidate: false,
        });
        w.save()?;

        let rec = BondRecord {
            mining_key,
            bonded_until: offer.bonded_until,
            owner_pk: offer.owner_pk,
            owner: OwnerKind::Rented,
            value: offer.bond,
            salt: plan.bond_salt,
            label: "Rented".into(),
            created_at: now,
            fund_commitment: Some(commitment),
            unbond_commitment: None,
            unbond_sig: None,
            owner_next_leaf: 0,
            seen_live: false,
            bond_file: None,
            reg: RegStatus {
                phase: "waiting".into(),
                note: "Registering: the wallet takes a proof once the bond confirms on midstate.".into(),
                ..RegStatus::default()
            },
        };
        let bond_id = rec.coin_id();
        self.bonds.bonds.push(rec);
        self.rentals.rentals.push(MyRental {
            offer,
            offer_salt: listing.salt,
            mining_key,
            bond_salt: plan.bond_salt,
            take_commitment: commitment,
            created_at: now,
            status: "taking".into(),
            gone_at: 0,
            note: "Renting: the send is on its way (see the Send tab).".into(),
        });
        if let Err(e) = self.bonds.save(&self.wallet_path).and_then(|_| self.rentals_save()) {
            self.bonds.bonds.retain(|b| b.coin_id() != bond_id);
            self.rentals.rentals.retain(|r| r.take_commitment != commitment);
            let w = self.wallet.as_mut().unwrap();
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
            return Err(e.context("could not record the rental, so nothing was sent"));
        }
        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "renting a bond — solving commit proof-of-work".into(),
                amount: offer.rent,
                fee: plan.fee,
                to: "bond rent".into(),
                updated_at: now,
            },
        );
        self.set_stage(commitment, SendStage::Committing, "renting a bond");
        self.broadcast_commit(commitment).await?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(hex::encode(bond_id))
    }

    /// `try_reveal` for a take: this wallet's coins are signed here, the
    /// offer's coin opened by the take witness. Nothing is signed if the
    /// offer has gone to someone else meanwhile.
    pub(super) async fn reveal_take(&mut self, commitment: [u8; 32], pending: midstate::wallet::PendingCommit) {
        let Some(r) = self.rentals.by_take(&commitment).cloned() else { return };
        let offer_coin = r.offer_coin();
        let hashes: Vec<[u8; 32]> = pending.outputs.iter().map(|x| x.hash_for_commitment()).collect();
        if pending.input_coin_ids.last() != Some(&offer_coin)
            || pending.input_coin_ids.len() < 2
            || compute_commitment(&pending.input_coin_ids, &hashes, &pending.salt) != commitment
        {
            self.set_stage(commitment, SendStage::Failed, "the rental's commit does not match its offer");
            return;
        }
        if !self.node.check_coin(offer_coin).await {
            self.take_lost(commitment, "someone else rented this bond first. Nothing was signed, and your coins are untouched.");
            return;
        }
        let ours = pending.input_coin_ids[..pending.input_coin_ids.len() - 1].to_vec();

        // As `try_reveal`: never sign for a one-time key the chain has seen
        // sign something else.
        let reused: Vec<[u8; 32]> = {
            let w = self.wallet.as_ref().unwrap();
            let trees: HashSet<[u8; 32]> = w.mss_keys().iter().map(|m| compute_address(&m.master_pk)).collect();
            let mut v: Vec<[u8; 32]> = ours
                .iter()
                .filter_map(|id| w.find_coin(id))
                .filter(|c| !trees.contains(&c.address))
                .map(|c| c.address)
                .filter(|a| matches!(self.node.storage.get_spent_commitment(a), Ok(Some(prev)) if prev != commitment))
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        if !reused.is_empty() {
            let w = self.wallet.as_mut().unwrap();
            for c in w.data.coins.iter_mut() {
                if reused.contains(&c.address) {
                    c.wots_signed = true;
                }
            }
            let _ = w.save();
            self.take_lost(
                commitment,
                "refused to sign: one of the coins it spends has a one-time key that already signed something else \
                 on chain. Those coins are set aside; nothing was spent — rent again to use your other coins.",
            );
            return;
        }
        if let Err(e) = self.verify_mss_indices().await {
            self.set_stage(commitment, SendStage::Failed, format!("could not sign the rental: {e:#}. Nothing was spent."));
            return;
        }
        let w = self.wallet.as_mut().unwrap();
        let signed = match rental::sign_wallet_inputs(w, &ours, &commitment) {
            Ok(s) => s,
            Err(e) => {
                self.set_stage(commitment, SendStage::Failed, format!("could not sign the rental: {e:#}"));
                return;
            }
        };
        // Signature side effects are persisted before anything is broadcast.
        if let Err(e) = w.save() {
            self.set_stage(commitment, SendStage::Failed, format!("could not persist wallet after signing: {e}"));
            return;
        }
        let (mut inputs, mut witnesses): (Vec<_>, Vec<_>) = signed.into_iter().unzip();
        inputs.push(r.offer.input(r.offer_salt));
        witnesses.push(take_witness(&r.mining_key));
        let tx = Transaction::Reveal { inputs, witnesses, outputs: pending.outputs.clone(), salt: pending.salt };
        match self.node.send_transaction(tx).await {
            Ok(()) => {
                self.set_stage(commitment, SendStage::RevealPending, "rental broadcast — waiting for confirmation");
                self.spawn_reveal_monitor(commitment, ours[0]);
            }
            Err(e) => self.set_stage(
                commitment,
                SendStage::Failed,
                format!("rental broadcast failed: {e}. Retry it from the Send tab when the node recovers."),
            ),
        }
    }

    /// A take that cannot happen: the commit is dropped, the bond record
    /// forgotten (its coin never existed), and the rental marked lost.
    fn take_lost(&mut self, commitment: [u8; 32], why: &str) {
        if let Some(w) = self.wallet.as_mut() {
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
        }
        self.bonds.bonds.retain(|b| b.fund_commitment != Some(commitment) || b.owner != OwnerKind::Rented || b.seen_live);
        if let Err(e) = self.bonds.save(&self.wallet_path) {
            tracing::warn!("saving bond records: {e:#}");
        }
        if let Some(r) = self.rentals.rentals.iter_mut().find(|r| r.take_commitment == commitment) {
            r.status = "lost".into();
            r.note = why.to_string();
        }
        let _ = self.rentals_save();
        self.set_stage(commitment, SendStage::Failed, why.to_string());
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    /// After `complete_reveal`: a take or a cancel has confirmed.
    pub(super) fn rental_confirmed(&mut self, commitment: [u8; 32]) {
        let mut changed = false;
        if let Some(r) = self.rentals.rentals.iter_mut().find(|r| r.take_commitment == commitment) {
            r.status = "rented".into();
            r.note = "Rented: once midwimble has registered the bond (about a day), your miner signs with it.".into();
            changed = true;
        }
        if let Some(o) = self.rentals.offers.iter_mut().find(|o| o.cancel_commitment == Some(commitment)) {
            o.status = "cancelled".into();
            o.note = "Cancelled: the coin is back in this wallet.".into();
            changed = true;
        }
        if changed {
            let _ = self.rentals_save();
        }
    }

    /// Confirms a send by one of its outputs appearing, for a spend of a
    /// coin someone else could spend first. Gives up once that coin has been
    /// gone a while without ours appearing.
    fn spawn_output_monitor(&self, commitment: [u8; 32], ours: [u8; 32], contested: [u8; 32]) {
        let node = self.node.clone();
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            let mut gone = 0u32;
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if node.check_coin(ours).await {
                    let _ = tx.send(Cmd::Internal(Internal::RevealConfirmed(commitment))).await;
                    return;
                }
                if !node.check_coin(contested).await {
                    gone += 1;
                    if gone > 600 {
                        return;
                    }
                }
            }
        });
    }

    // ── Keeping up with the chain ────────────────────────────────────────

    pub(super) async fn tick_rentals(&mut self, tip: u64) {
        if self.wallet.is_none() {
            return;
        }
        self.scan_market(tip).await;
        self.watch_offers(tip).await;
        self.watch_rentals(tip);
    }

    /// Reads the next stretch of blocks for announcements, refreshes which
    /// listed coins are still there, and adopts this wallet's own offers
    /// found on chain (after a restore, say).
    async fn scan_market(&mut self, tip: u64) {
        let start = tip.saturating_sub(rental::MARKET_LOOKBACK).max(1);
        if self.market.cursor < start {
            self.market.cursor = start;
        }
        if self.market.cursor <= tip {
            let from = self.market.cursor;
            let to = (from + rental::SCAN_BUDGET).min(tip + 1);
            let node = self.node.clone();
            let found = tokio::task::spawn_blocking(move || {
                let mut out = Vec::new();
                for h in from..to {
                    if let Ok(Some(batch)) = node.storage.batches.load(h) {
                        for tx in &batch.transactions {
                            out.extend(announced(tx).into_iter().map(|a| (h, a)));
                        }
                    }
                }
                out
            })
            .await
            .unwrap_or_default();
            for (h, a) in found {
                self.market.add(h, a);
            }
            self.market.cursor = to;
        }
        let state = self.node.get_state().await;
        for l in &mut self.market.listings {
            l.live = state.coins.contains(&l.coin_id);
        }
        let ours = self.own_tree_keys();
        let mut adopted = false;
        for l in &self.market.listings {
            if ours.contains(&l.offer.owner_pk) && self.rentals.offer_by_coin(&l.coin_id).is_none() {
                self.rentals.offers.push(MyOffer {
                    offer: l.offer,
                    salt: l.salt,
                    created_at: now_secs(),
                    fund_commitment: None,
                    status: "listed".into(),
                    cancel_commitment: None,
                    cancel_sig: None,
                    taken_by: None,
                    rent_paid: 0,
                    scan_from: l.height,
                    note: "Found on chain: one of this wallet's offers.".into(),
                });
                adopted = true;
            }
        }
        if adopted {
            let _ = self.rentals_save();
        }
    }

    /// Moves this wallet's offers along: listed once their coin is on chain;
    /// once it is gone, finds the transaction that spent it. A take becomes
    /// a bond record (this wallet's coin, someone else's mining key) that
    /// unbonds like any other when its lock ends.
    async fn watch_offers(&mut self, tip: u64) {
        let state = self.node.get_state().await;
        let mut changed = false;
        let mut search: Vec<usize> = Vec::new();
        for (i, o) in self.rentals.offers.iter_mut().enumerate() {
            if !o.open() {
                continue;
            }
            let live = state.coins.contains(&o.coin_id());
            if o.status == "funding" {
                let pending = o.fund_commitment.is_some_and(|c| {
                    self.wallet.as_ref().is_some_and(|w| w.find_pending(&c).is_some())
                });
                let failed = o.fund_commitment.and_then(|c| self.sends.get(&c)).is_some_and(|m| m.stage == SendStage::Failed);
                if live || !pending {
                    if failed && !live {
                        o.status = "failed".into();
                        o.note = "The send that posts it did not go through; nothing was locked.".into();
                    } else {
                        o.status = "listed".into();
                        o.note = "Listed: anyone can rent it, and this wallet need not be open.".into();
                        o.scan_from = tip.saturating_sub(if live { 20 } else { 1_000 });
                    }
                    changed = true;
                }
                continue;
            }
            if !live {
                search.push(i);
            }
        }
        for i in search {
            let (offer, coin, from) = {
                let o = &self.rentals.offers[i];
                (o.offer, o.coin_id(), o.scan_from.max(1))
            };
            let to = (from + SPEND_SCAN_BUDGET).min(tip + 1);
            let node = self.node.clone();
            let found = tokio::task::spawn_blocking(move || {
                for h in from..to {
                    if let Ok(Some(batch)) = node.storage.batches.load(h) {
                        for tx in &batch.transactions {
                            if let Some(s) = spent_by(tx, &offer, &coin) {
                                return Some(s);
                            }
                        }
                    }
                }
                None
            })
            .await
            .unwrap_or(None);
            let o = &mut self.rentals.offers[i];
            match found {
                None => {
                    // Not in this stretch; the next tick reads on.
                    o.scan_from = to.min(tip.saturating_sub(20).max(from));
                    changed = true;
                }
                Some(OfferSpend::Cancelled) => {
                    o.status = "cancelled".into();
                    o.note = "Cancelled: the coin is back in this wallet.".into();
                    changed = true;
                }
                Some(OfferSpend::Taken { mining_key, bond_salt, rent_paid }) => {
                    o.status = "taken".into();
                    o.taken_by = Some(mining_key);
                    o.rent_paid = rent_paid;
                    o.note = format!(
                        "Rented out: the rent ({}) is in your balance, and the bond is in your bond list. \
                         Unbond it there when its lock ends.",
                        bond::gmds(rent_paid)
                    );
                    let cancel = o.cancel_commitment.take();
                    changed = true;
                    if let Some(c) = cancel {
                        if self.wallet.as_ref().is_some_and(|w| w.find_pending(&c).is_some()) {
                            self.cancel_lost(c);
                        }
                    }
                    if let Some(salt) = bond_salt {
                        let rec = BondRecord {
                            mining_key,
                            bonded_until: offer.bonded_until,
                            owner_pk: offer.owner_pk,
                            owner: OwnerKind::WalletTree,
                            value: offer.bond,
                            salt,
                            label: "Rented out".into(),
                            created_at: now_secs(),
                            fund_commitment: None,
                            unbond_commitment: None,
                            unbond_sig: None,
                            owner_next_leaf: 0,
                            seen_live: false,
                            bond_file: None,
                            reg: RegStatus::default(),
                        };
                        if self.bonds.find(&rec.coin_id()).is_none() {
                            self.bonds.bonds.push(rec);
                            if let Err(e) = self.bonds.save(&self.wallet_path) {
                                tracing::warn!("saving bond records: {e:#}");
                            }
                        }
                    }
                    let _ = self.events.send(WalletEvent::WalletChanged);
                }
            }
        }
        if changed {
            let _ = self.rentals_save();
        }
    }

    /// Takes on their way: done once the bond is on chain; lost once the
    /// offer has been gone a while without it.
    fn watch_rentals(&mut self, tip: u64) {
        let mut lost = Vec::new();
        let mut changed = false;
        for r in self.rentals.rentals.iter_mut().filter(|r| r.status == "taking") {
            if self.bonds.find(&r.bond_coin()).is_some_and(|b| b.seen_live) {
                r.status = "rented".into();
                r.note = "Rented: once midwimble has registered the bond (about a day), your miner signs with it.".into();
                changed = true;
            } else if self.market.find(&r.offer_coin()).is_some_and(|l| !l.live) {
                if r.gone_at == 0 {
                    r.gone_at = tip;
                    changed = true;
                } else if tip >= r.gone_at + rental::LOST_AFTER {
                    lost.push(r.take_commitment);
                }
            }
        }
        if changed {
            let _ = self.rentals_save();
        }
        for c in lost {
            self.take_lost(
                c,
                "someone else rented this bond first. Coins whose one-time keys signed for it are set aside; \
                 the rest are untouched.",
            );
        }
    }
}
