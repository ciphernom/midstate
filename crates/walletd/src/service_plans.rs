//! The walletd actor's side of inheritance plans and payment streams
//! (`crate::plans`, `midstate::core::plans`). A child of `service`, sharing
//! the actor's state, as `service_rental` is.
//!
//! Funding a plan or a stream is an ordinary send of this wallet's coins.
//! Every other move (a check-in, a close, a claim, a cancel, a collection, a
//! sweep) spends covenant coins only, with the fee taken from them, and is
//! recorded in full as a [`CovSpend`] before its commit leaves, so its reveal
//! is put together the same way after a restart ([`Service::reveal_plan`]).

use super::{now_secs, parse_hex32, SendMeta, Service};
use crate::api::{HeldPlanView, HeldStreamView, OwnedPlanView, PlansView, SendStage, SentStreamView, WalletEvent};
use crate::bond;
use crate::plans::{
    self, available, claim_inputs, decode_payee, decode_plan, decode_stream, encode_payee, encode_plan,
    encode_stream, fee_for, free_indices, id_of, live_plan_coins, live_tranches, owner_inputs, payout_inputs,
    payouts_in, plan_denominations, plan_of, reveal_size, shares_from_bps, stream_of, tranche_inputs, when,
    witnesses, CovSpend, HeldPlan, HeldStream, OwnedPlan, Payee, PayoutCoin, PlanBook, PlanCoin, SentStream,
    Vanished, KEY_HEIGHT, MAX_SPEND_INPUTS, SCAN_BUDGET,
};
use crate::sendplan;
use anyhow::{anyhow, bail, Result};
use midstate::core::plans::{Heir, InheritancePlan, PaymentStream, DAY, MIN_COIN};
use midstate::core::{compute_address, compute_commitment, decompose_value, OutputData, Transaction};
use std::collections::HashSet;

/// Signatures an owner's or a payer's key can make: one per check-in, close
/// or cancel.
const OWNER_KEY_HEIGHT: u32 = 8;
/// Seconds a payout waits before it is swept on its own.
const SWEEP_AFTER: u64 = 6 * 3_600;

impl Service {
    pub(super) fn plans_load(&mut self) {
        let (book, warning) = PlanBook::load(&self.wallet_path);
        self.plans = book;
        if let Some(text) = warning {
            tracing::warn!("{text}");
            let _ = self.events.send(WalletEvent::Warning { text });
        }
    }

    pub(super) fn plans_unload(&mut self) {
        self.plans = PlanBook::default();
    }

    fn plans_save(&self) -> Result<()> {
        self.plans.save(&self.wallet_path)
    }

    fn plans_ready(&self) -> Result<()> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — a send planned now would fail");
        }
        self.ensure_restore_done()
    }

    fn mss_pks(&self) -> HashSet<[u8; 32]> {
        self.wallet.as_ref().map_or_else(HashSet::new, |w| w.mss_keys().iter().map(|k| k.master_pk).collect())
    }

    fn leaves_left(&self, pk: &[u8; 32]) -> u64 {
        self.wallet
            .as_ref()
            .and_then(|w| w.mss_keys().iter().find(|k| &k.master_pk == pk).map(|k| k.remaining()))
            .unwrap_or(0)
    }

    fn key_height(&self, pk: &[u8; 32]) -> u32 {
        self.wallet
            .as_ref()
            .and_then(|w| w.mss_keys().iter().find(|k| &k.master_pk == pk).map(|k| k.height))
            .unwrap_or(KEY_HEIGHT)
    }

    /// A fresh MSS tree of this wallet's; its public key.
    fn new_tree(&mut self, height: u32, label: &str) -> Result<[u8; 32]> {
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let address = w.generate_mss(height, Some(label.into()))?;
        w.mss_keys()
            .iter()
            .find(|k| compute_address(&k.master_pk) == address)
            .map(|k| k.master_pk)
            .ok_or_else(|| anyhow!("internal error: the new key was not kept"))
    }

    /// A new payee key for one code. Each plan or stream that pays this
    /// wallet then has a key of its own, with every signature its sweeps
    /// will ever need.
    fn payee_key(&mut self) -> Result<[u8; 32]> {
        let pk = self.new_tree(KEY_HEIGHT, "plan payee")?;
        self.plans.payee_keys.push(pk);
        self.plans_save()?;
        Ok(pk)
    }

    pub(super) fn payee_code(&mut self, name: &str) -> Result<String> {
        if name.len() > 64 {
            bail!("keep the name to 64 bytes");
        }
        let pk = self.payee_key()?;
        Ok(encode_payee(&Payee { pk, name: name.trim().to_string() }))
    }

    async fn live(&self) -> midstate::core::State {
        self.node.get_state().await
    }

    // ── The view ─────────────────────────────────────────────────────────

    pub(super) async fn plans_view(&mut self) -> Result<PlansView> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let state = self.live().await;
        let tip = state.height;
        let live = |id: &[u8; 32]| state.coins.contains(id);
        let payees: HashSet<[u8; 32]> = self.plans.payee_keys.iter().copied().collect();
        let in_flight: HashSet<[u8; 32]> =
            self.plans.spends.iter().flat_map(|s| s.inputs.iter().map(|i| i.reveal.coin_id())).collect();

        let mut owned = Vec::new();
        for o in &self.plans.owned {
            let Some(p) = plan_of(&o.plan) else { continue };
            let id = id_of(&p.to_bytes());
            let coins = live_plan_coins(&p, tip, live);
            let balance: u64 = coins.iter().map(|c| c.value).sum();
            let deadline = coins.iter().map(|c| p.deadline(c.k)).min();
            let busy = coins.iter().any(|c| in_flight.contains(&c.id(&p)));
            let total_bps = |share: u32| (share as u64 * 10_000 / (midstate::core::plans::SHARE_DENOM - 1)) as u32;
            owned.push(OwnedPlanView {
                id,
                label: o.label.clone(),
                heirs: p.heirs.iter().zip(&o.heir_names).map(|(h, n)| (n.clone(), total_bps(h.share))).collect(),
                balance,
                period_days: p.period / DAY,
                deadline: deadline.unwrap_or(0),
                deadline_text: deadline.map(|d| when(d, tip)).unwrap_or_default(),
                overdue: deadline.is_some_and(|d| d <= tip),
                due_soon: deadline.is_some_and(|d| d > tip && d - tip <= p.period / 10),
                status: if o.status == "closed" {
                    "closed".into()
                } else if busy {
                    "moving".into()
                } else if balance == 0 {
                    "empty".into()
                } else {
                    "active".into()
                },
                note: o.note.clone(),
                code: encode_plan(&p, &o.heir_names),
                can_check_in: !busy && coins.iter().any(|c| c.k < p.epoch_after(tip)),
                can_close: !busy && balance > 0,
                owner_signatures_left: self.leaves_left(&p.owner_pk),
            });
        }
        let mut held = Vec::new();
        for h in &self.plans.held {
            let Some(p) = plan_of(&h.plan) else { continue };
            let id = id_of(&p.to_bytes());
            let coins = live_plan_coins(&p, tip, live);
            let balance: u64 = coins.iter().map(|c| c.value).sum();
            let deadline = coins.iter().map(|c| p.deadline(c.k)).min();
            let mine: u32 = p.heirs.iter().filter(|x| payees.contains(&x.pk)).map(|x| x.share).sum();
            let claimable = coins.iter().any(|c| p.deadline(c.k) <= tip && !in_flight.contains(&c.id(&p)));
            held.push(HeldPlanView {
                id,
                label: h.label.clone(),
                heirs: p.heirs.iter().zip(&h.heir_names).map(|(x, n)| (n.clone(), payees.contains(&x.pk))).collect(),
                my_share_bps: (mine as u64 * 10_000 / (midstate::core::plans::SHARE_DENOM - 1)) as u32,
                balance,
                deadline: deadline.unwrap_or(0),
                deadline_text: deadline.map(|d| when(d, tip)).unwrap_or_default(),
                claimable,
                period_days: p.period / DAY,
                note: h.note.clone(),
            });
        }
        let mut sent = Vec::new();
        for s in &self.plans.sent {
            let Some(st) = stream_of(&s.stream) else { continue };
            let id = id_of(&st.to_bytes());
            let coins = live_tranches(&st, live);
            let left: HashSet<u32> = coins.iter().map(|(j, _, _)| *j).collect();
            let future: Vec<u32> = left.iter().copied().filter(|j| st.due(*j) > tip).collect();
            let next = (0..st.count).filter(|j| left.contains(j)).map(|j| st.due(j)).min();
            sent.push(SentStreamView {
                id,
                label: s.label.clone(),
                recipient: s.recipient_name.clone(),
                amount: st.amount,
                count: st.count,
                period_days: st.period as f64 / DAY as f64,
                cancellable: st.payer_pk.is_some(),
                waiting: left.len() as u32,
                not_yet_due: future.len() as u32,
                next_due_text: next.map(|d| when(d, tip)).unwrap_or_default(),
                last_due_text: when(st.due(st.count - 1), tip),
                status: s.status.clone(),
                note: s.note.clone(),
                code: encode_stream(&st, &s.label),
                can_cancel: st.payer_pk.is_some()
                    && !future.is_empty()
                    && !coins.iter().any(|(j, n, _)| in_flight.contains(&st.input(*j, *n).coin_id())),
            });
        }
        let mut received = Vec::new();
        for r in &self.plans.received {
            let Some(st) = stream_of(&r.stream) else { continue };
            let id = id_of(&st.to_bytes());
            let coins = live_tranches(&st, live);
            let left: HashSet<u32> = coins.iter().map(|(j, _, _)| *j).collect();
            let due_now = left.iter().filter(|j| st.due(**j) <= tip).count() as u32;
            let next = (0..st.count).filter(|j| left.contains(j) && st.due(*j) > tip).map(|j| st.due(j)).min();
            received.push(HeldStreamView {
                id,
                label: r.label.clone(),
                amount: st.amount,
                count: st.count,
                period_days: st.period as f64 / DAY as f64,
                cancellable: st.payer_pk.is_some(),
                waiting: left.len() as u32,
                due_now,
                next_due_text: next.map(|d| when(d, tip)).unwrap_or_default(),
                last_due_text: when(st.due(st.count - 1), tip),
                auto_collect: r.auto_collect,
                note: r.note.clone(),
            });
        }
        let payee = self.plans.payee_keys.last().copied();
        Ok(PlansView {
            blocked: available(tip).err().unwrap_or_default(),
            tip,
            owned,
            held,
            sent,
            received,
            payee_ready: payee.is_some(),
            payee_left: payee.map(|pk| self.leaves_left(&pk)).unwrap_or(0),
            payouts_waiting: self.plans.payouts.iter().filter(|p| p.sweeping.is_none()).map(|p| p.value).sum(),
            payouts_sweeping: self.plans.payouts.iter().filter(|p| p.sweeping.is_some()).map(|p| p.value).sum(),
        })
    }

    // ── Inheritance ──────────────────────────────────────────────────────

    /// Makes a plan for `amount` of this wallet's coins: heirs by payee code
    /// with their shares in hundredths of a percent, checked in every
    /// `period_days`. Returns the plan's id.
    pub(super) async fn create_plan(
        &mut self,
        label: String,
        heirs: Vec<(String, u32)>,
        period_days: u64,
        amount: u64,
    ) -> Result<String> {
        self.plans_ready()?;
        let tip = self.live().await.height;
        available(tip).map_err(anyhow::Error::msg)?;
        let payees: Vec<Payee> = heirs.iter().map(|(c, _)| decode_payee(c)).collect::<Result<_>>()?;
        let shares = shares_from_bps(&heirs.iter().map(|(_, b)| *b).collect::<Vec<_>>())?;
        let ours = self.mss_pks();
        if payees.iter().any(|p| ours.contains(&p.pk)) {
            bail!("an heir's code is one of this wallet's own keys");
        }
        let denoms = plan_denominations(amount);
        if denoms.is_empty() {
            bail!("a plan holds at least {} units", MIN_COIN);
        }
        let probe = InheritancePlan {
            owner_pk: [0xee; 32],
            heirs: payees.iter().zip(&shares).map(|(p, s)| Heir { pk: p.pk, share: *s }).collect(),
            start: tip,
            period: period_days * DAY,
            secret: rand::random(),
        };
        probe.check().map_err(anyhow::Error::msg)?;
        self.verify_mss_indices().await?;
        let plan = InheritancePlan { owner_pk: self.new_tree(OWNER_KEY_HEIGHT, "inheritance plan")?, ..probe };
        plan.check().map_err(anyhow::Error::msg)?;
        let k = plan.epoch_after(tip);
        let outputs: Vec<OutputData> = denoms.iter().enumerate().map(|(n, v)| plan.output(k, n as u32, *v)).collect();
        let put: u64 = denoms.iter().sum();
        let id = id_of(&plan.to_bytes());
        let names: Vec<String> = payees.iter().map(|p| p.name.clone()).collect();
        let live = self.spendable().await;
        let w = self.wallet.as_mut().unwrap();
        let sp = sendplan::plan_fixed_outputs(w, &live, outputs)?;
        let record = hex::encode(plan.to_bytes());
        self.plans.owned.push(OwnedPlan {
            plan: record.clone(),
            label: label.clone(),
            heir_names: names,
            created_at: now_secs(),
            status: "active".into(),
            note: "Funding: the send is on the Send tab.".into(),
        });
        // Recorded before anything leaves: the plan's parameters exist
        // nowhere else yet.
        let committed = self.plans_save().and_then(|_| {
            let w = self.wallet.as_mut().unwrap();
            let (c, _) = w.prepare_commit(&sp.input_coin_ids, &sp.outputs, sp.change_seeds, false, false)?;
            w.save()?;
            Ok(c)
        });
        let commitment = match committed {
            Ok(c) => c,
            Err(e) => {
                self.plans.owned.retain(|o| o.plan != record);
                let _ = self.plans_save();
                return Err(e.context("nothing was sent"));
            }
        };
        self.launch(commitment, &format!("funding the plan “{label}”"), put, sp.fee, "inheritance plan").await?;
        Ok(id)
    }

    /// This wallet's coins on chain and in no send.
    async fn spendable(&self) -> Vec<[u8; 32]> {
        let state = self.live().await;
        let in_flight = self.in_flight_inputs();
        let Some(w) = self.wallet.as_ref() else { return Vec::new() };
        w.coins()
            .iter()
            .filter(|c| !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id))
            .map(|c| c.coin_id)
            .collect()
    }

    fn owned_plan(&self, id: &str) -> Result<(usize, InheritancePlan)> {
        self.plans
            .owned
            .iter()
            .enumerate()
            .find_map(|(i, o)| plan_of(&o.plan).filter(|p| id_of(&p.to_bytes()) == id).map(|p| (i, p)))
            .ok_or_else(|| anyhow!("no such plan of yours"))
    }

    fn held_plan(&self, id: &str) -> Option<InheritancePlan> {
        self.plans.held.iter().find_map(|h| plan_of(&h.plan).filter(|p| id_of(&p.to_bytes()) == id))
    }

    /// The plan's coins that no send in flight touches.
    async fn free_plan_coins(&self, p: &InheritancePlan) -> (u64, Vec<PlanCoin>, Vec<PlanCoin>) {
        let state = self.live().await;
        let all = live_plan_coins(p, state.height, |id| state.coins.contains(id));
        let free = all.iter().copied().filter(|c| !self.plans.spending(&c.id(p))).collect();
        (state.height, all, free)
    }

    /// Moves the plan's coins to the next deadline: a full period from now.
    pub(super) async fn check_in(&mut self, id: &str) -> Result<String> {
        self.plans_ready()?;
        let (_, p) = self.owned_plan(id)?;
        let (tip, all, free) = self.free_plan_coins(&p).await;
        let to = p.epoch_after(tip);
        let moving: Vec<PlanCoin> = free.iter().copied().filter(|c| c.k < to).take(MAX_SPEND_INPUTS).collect();
        if moving.is_empty() {
            bail!("nothing to move: every coin already sits at the next deadline");
        }
        let total: u64 = moving.iter().map(|c| c.value).sum();
        let inputs = owner_inputs(&p, &moving);
        let sig_len = bond::mss_sig_len(self.key_height(&p.owner_pk));
        let fee = fee_for(reveal_size(&inputs, sig_len, plans::MAX_SPEND_INPUTS));
        let keep = total.checked_sub(fee).ok_or_else(|| anyhow!("the plan cannot pay its own fee"))?;
        let denoms = plan_denominations(keep);
        let free_n = free_indices(&all, to, denoms.len());
        if free_n.len() < denoms.len() {
            bail!("the next deadline's address is full: close the plan and make a new one");
        }
        let mut outputs: Vec<OutputData> =
            denoms.iter().zip(&free_n).map(|(v, n)| p.output(to, *n, *v)).collect();
        let back = keep - denoms.iter().sum::<u64>();
        let change = self.change_outputs(back, outputs.len())?;
        outputs.extend(change.0);
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "check-in".into(),
            of: id.into(),
            inputs,
            signer: Some(p.owner_pk),
            sig: None,
            pays_us: vec![],
            returned: back,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, change.1, "checking in", total - fee - back, fee, "plan, next deadline").await?;
        Ok(hex::encode(c))
    }

    /// Ends a plan: every coin comes back to this wallet.
    pub(super) async fn close_plan(&mut self, id: &str) -> Result<String> {
        self.plans_ready()?;
        let (_, p) = self.owned_plan(id)?;
        let (_, _, free) = self.free_plan_coins(&p).await;
        let coins: Vec<PlanCoin> = free.into_iter().take(MAX_SPEND_INPUTS).collect();
        if coins.is_empty() {
            bail!("the plan holds nothing to take back");
        }
        let inputs = owner_inputs(&p, &coins);
        let total: u64 = coins.iter().map(|c| c.value).sum();
        let sig_len = bond::mss_sig_len(self.key_height(&p.owner_pk));
        let (fee, _) = plans::resolve_fee(&inputs, sig_len, 0, total).ok_or_else(|| anyhow!("the plan cannot pay its own fee"))?;
        let (outputs, seeds) = self.change_outputs(total - fee, 0)?;
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "close".into(),
            of: id.into(),
            inputs,
            signer: Some(p.owner_pk),
            sig: None,
            pays_us: vec![],
            returned: total - fee,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, seeds, "closing a plan", total - fee, fee, "this wallet").await?;
        Ok(hex::encode(c))
    }

    /// Pays every heir of a plan its share, from coins past their deadline.
    /// Anyone may; this wallet does it for plans naming it.
    pub(super) async fn claim_plan(&mut self, id: &str) -> Result<String> {
        self.plans_ready()?;
        let p = self.held_plan(id).ok_or_else(|| anyhow!("no such plan names you"))?;
        let (tip, _, free) = self.free_plan_coins(&p).await;
        let due: Vec<PlanCoin> = free.into_iter().filter(|c| p.deadline(c.k) <= tip).take(MAX_SPEND_INPUTS).collect();
        if due.is_empty() {
            bail!("nothing is claimable yet: the owner checked in, or the deadline has not passed");
        }
        let mut ks: Vec<u64> = due.iter().map(|c| c.k).collect();
        ks.sort_unstable();
        ks.dedup();
        let ours: HashSet<[u8; 32]> = self.plans.payee_keys.iter().copied().collect();
        let mut outputs = Vec::new();
        let mut pays_us = Vec::new();
        for k in ks {
            let total: u64 = due.iter().filter(|c| c.k == k).map(|c| c.value).sum();
            for (i, (address, owed)) in p.owed(k, total).into_iter().enumerate() {
                for value in decompose_value(owed) {
                    let salt: [u8; 32] = rand::random();
                    if ours.contains(&p.heirs[i].pk) {
                        pays_us.push(PayoutCoin {
                            tag: p.payout_tag(k, i),
                            pk: p.heirs[i].pk,
                            value,
                            salt,
                            from: "an inheritance".into(),
                            sweeping: None,
                            found_at: 0,
                        });
                    }
                    outputs.push(OutputData::Standard { address, value, salt });
                }
            }
        }
        if outputs.len() > midstate::core::types::MAX_TX_OUTPUTS {
            bail!("the claim needs too many outputs for one transaction");
        }
        let total: u64 = due.iter().map(|c| c.value).sum();
        let paid: u64 = outputs.iter().map(|o| o.value()).sum();
        let inputs = claim_inputs(&p, &due);
        let need = fee_for(reveal_size(&inputs, 0, outputs.len()));
        if total - paid < need {
            bail!("the claim cannot pay its fee");
        }
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "claim".into(),
            of: id.into(),
            inputs,
            signer: None,
            sig: None,
            pays_us,
            returned: 0,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, vec![], "paying a plan's heirs", paid, total - paid, "the heirs").await?;
        Ok(hex::encode(c))
    }

    /// Takes in a plan or stream code that names one of this wallet's keys.
    pub(super) async fn import_code(&mut self, code: &str, label: &str) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let ours = self.mss_pks();
        let payees: HashSet<[u8; 32]> = self.plans.payee_keys.iter().copied().collect();
        let code = code.trim();
        if code.starts_with(plans::PLAN_CODE) {
            let (p, names) = decode_plan(code)?;
            let hex_plan = hex::encode(p.to_bytes());
            let id = id_of(&p.to_bytes());
            if self.plans.owned.iter().any(|o| o.plan == hex_plan) || self.plans.held.iter().any(|h| h.plan == hex_plan) {
                bail!("this wallet has that plan already");
            }
            if ours.contains(&p.owner_pk) {
                self.plans.owned.push(OwnedPlan {
                    plan: hex_plan,
                    label: label.into(),
                    heir_names: names,
                    created_at: now_secs(),
                    status: "active".into(),
                    note: "Restored from its code.".into(),
                });
            } else if p.heirs.iter().any(|h| payees.contains(&h.pk)) {
                self.plans.held.push(HeldPlan {
                    plan: hex_plan,
                    label: label.into(),
                    heir_names: names,
                    imported_at: now_secs(),
                    note: String::new(),
                });
            } else {
                bail!("that plan names none of this wallet's keys: it is someone else's");
            }
            self.plans_save()?;
            return Ok(id);
        }
        if code.starts_with(plans::STREAM_CODE) {
            let (s, l) = decode_stream(code)?;
            let hex_stream = hex::encode(s.to_bytes());
            let id = id_of(&s.to_bytes());
            if self.plans.sent.iter().any(|x| x.stream == hex_stream) || self.plans.received.iter().any(|x| x.stream == hex_stream) {
                bail!("this wallet has that stream already");
            }
            let label = if label.is_empty() { l } else { label.into() };
            if payees.contains(&s.recipient_pk) {
                self.plans.received.push(HeldStream {
                    stream: hex_stream,
                    label,
                    imported_at: now_secs(),
                    auto_collect: true,
                    note: String::new(),
                });
            } else if s.payer_pk.is_some_and(|k| ours.contains(&k)) {
                self.plans.sent.push(SentStream {
                    stream: hex_stream,
                    label,
                    recipient_name: String::new(),
                    created_at: now_secs(),
                    status: "active".into(),
                    fund_commitment: None,
                    note: "Restored from its code.".into(),
                });
            } else {
                bail!("that stream pays none of this wallet's keys");
            }
            self.plans_save()?;
            return Ok(id);
        }
        bail!("that is neither a plan code ({}…) nor a stream code ({}…)", plans::PLAN_CODE, plans::STREAM_CODE)
    }

    // ── Streams ──────────────────────────────────────────────────────────

    /// Pays `amount` to a payee every `period_blocks`, `count` times, the
    /// first after `delay_blocks`. Returns the stream's code.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn create_stream(
        &mut self,
        label: String,
        payee: String,
        amount: u64,
        period_blocks: u64,
        count: u32,
        delay_blocks: u64,
        cancellable: bool,
    ) -> Result<String> {
        self.plans_ready()?;
        let tip = self.live().await.height;
        available(tip).map_err(anyhow::Error::msg)?;
        let to = decode_payee(&payee)?;
        if self.mss_pks().contains(&to.pk) {
            bail!("that payee code is one of this wallet's own keys");
        }
        let probe = PaymentStream {
            payer_pk: cancellable.then_some([0xee; 32]),
            recipient_pk: to.pk,
            first: tip + delay_blocks,
            period: period_blocks,
            amount,
            count,
            secret: rand::random(),
        };
        probe.check().map_err(anyhow::Error::msg)?;
        self.verify_mss_indices().await?;
        let payer_pk = if cancellable { Some(self.new_tree(OWNER_KEY_HEIGHT, "payment stream")?) } else { None };
        let stream = PaymentStream { payer_pk, ..probe };
        stream.check().map_err(anyhow::Error::msg)?;
        let code = encode_stream(&stream, &label);
        let live = self.spendable().await;
        let w = self.wallet.as_mut().unwrap();
        let sp = sendplan::plan_fixed_outputs(w, &live, stream.funding_outputs())?;
        if sp.outputs.len() > midstate::core::types::MAX_TX_OUTPUTS {
            bail!("that stream needs more outputs than one transaction holds: use fewer payments");
        }
        let record = hex::encode(stream.to_bytes());
        self.plans.sent.push(SentStream {
            stream: record.clone(),
            label: label.clone(),
            recipient_name: to.name.clone(),
            created_at: now_secs(),
            status: "funding".into(),
            fund_commitment: None,
            note: "Funding: the send is on the Send tab. Give the recipient the stream's code.".into(),
        });
        let committed = self.plans_save().and_then(|_| {
            let w = self.wallet.as_mut().unwrap();
            let (c, _) = w.prepare_commit(&sp.input_coin_ids, &sp.outputs, sp.change_seeds, false, false)?;
            w.save()?;
            Ok(c)
        });
        let commitment = match committed {
            Ok(c) => c,
            Err(e) => {
                self.plans.sent.retain(|s| s.stream != record);
                let _ = self.plans_save();
                return Err(e.context("nothing was sent"));
            }
        };
        if let Some(s) = self.plans.sent.iter_mut().find(|s| s.stream == record) {
            s.fund_commitment = Some(commitment);
        }
        self.plans_save()?;
        let what = format!("funding the stream “{label}”");
        self.launch(commitment, &what, stream.total(), sp.fee, &format!("stream to {}", to.name)).await?;
        Ok(code)
    }

    fn sent_stream(&self, id: &str) -> Result<PaymentStream> {
        self.plans
            .sent
            .iter()
            .find_map(|s| stream_of(&s.stream).filter(|x| id_of(&x.to_bytes()) == id))
            .ok_or_else(|| anyhow!("no such stream of yours"))
    }

    fn received_stream(&self, id: &str) -> Result<PaymentStream> {
        self.plans
            .received
            .iter()
            .find_map(|s| stream_of(&s.stream).filter(|x| id_of(&x.to_bytes()) == id))
            .ok_or_else(|| anyhow!("no such stream pays you"))
    }

    /// Takes back every payment of a cancellable stream that is not yet due.
    /// Payments already due are the recipient's to collect and are left.
    pub(super) async fn cancel_stream(&mut self, id: &str) -> Result<String> {
        self.plans_ready()?;
        let s = self.sent_stream(id)?;
        let payer = s.payer_pk.ok_or_else(|| anyhow!("this stream cannot be cancelled: it was made that way"))?;
        let state = self.live().await;
        // A margin of blocks, so a payment cannot fall due under the cancel.
        let cutoff = state.height + 10;
        let coins: Vec<(u32, u32, u64)> = live_tranches(&s, |id| state.coins.contains(id))
            .into_iter()
            .filter(|(j, n, _)| s.due(*j) > cutoff && !self.plans.spending(&s.input(*j, *n).coin_id()))
            .take(MAX_SPEND_INPUTS)
            .collect();
        if coins.is_empty() {
            bail!("no payment is left that is not yet due");
        }
        let total: u64 = coins.iter().map(|c| c.2).sum();
        let inputs = tranche_inputs(&s, &coins, true);
        let sig_len = bond::mss_sig_len(self.key_height(&payer));
        let (fee, _) = plans::resolve_fee(&inputs, sig_len, 0, total).ok_or_else(|| anyhow!("the payments cannot pay the fee"))?;
        let (outputs, seeds) = self.change_outputs(total - fee, 0)?;
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "cancel".into(),
            of: id.into(),
            inputs,
            signer: Some(payer),
            sig: None,
            pays_us: vec![],
            returned: total - fee,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, seeds, "cancelling a stream's future payments", total - fee, fee, "this wallet").await?;
        Ok(hex::encode(c))
    }

    /// Collects every payment of a stream to this wallet that is due.
    pub(super) async fn collect_stream(&mut self, id: &str) -> Result<String> {
        self.plans_ready()?;
        let s = self.received_stream(id)?;
        let state = self.live().await;
        let tip = state.height;
        let coins: Vec<(u32, u32, u64)> = live_tranches(&s, |id| state.coins.contains(id))
            .into_iter()
            .filter(|(j, n, _)| s.due(*j) <= tip && !self.plans.spending(&s.input(*j, *n).coin_id()))
            .take(MAX_SPEND_INPUTS)
            .collect();
        if coins.is_empty() {
            bail!("no payment is due yet");
        }
        let mut js: Vec<u32> = coins.iter().map(|c| c.0).collect();
        js.dedup();
        let mut outputs = Vec::new();
        let mut pays_us = Vec::new();
        for j in js {
            let total: u64 = coins.iter().filter(|c| c.0 == j).map(|c| c.2).sum();
            for value in decompose_value(s.owed(total)) {
                let salt: [u8; 32] = rand::random();
                pays_us.push(PayoutCoin {
                    tag: s.payout_tag(j),
                    pk: s.recipient_pk,
                    value,
                    salt,
                    from: "a payment stream".into(),
                    sweeping: None,
                    found_at: 0,
                });
                outputs.push(OutputData::Standard { address: s.payout_address(j), value, salt });
            }
        }
        let total: u64 = coins.iter().map(|c| c.2).sum();
        let paid: u64 = outputs.iter().map(|o| o.value()).sum();
        let inputs = tranche_inputs(&s, &coins, false);
        if total - paid < fee_for(reveal_size(&inputs, 0, outputs.len())) {
            bail!("the payments cannot pay the fee");
        }
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "collect".into(),
            of: id.into(),
            inputs,
            signer: None,
            sig: None,
            pays_us,
            returned: 0,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, vec![], "collecting a stream's payments", paid, total - paid, "you").await?;
        Ok(hex::encode(c))
    }

    pub(super) fn set_auto_collect(&mut self, id: &str, on: bool) -> Result<()> {
        let r = self
            .plans
            .received
            .iter_mut()
            .find(|r| stream_of(&r.stream).is_some_and(|s| id_of(&s.to_bytes()) == id))
            .ok_or_else(|| anyhow!("no such stream pays you"))?;
        r.auto_collect = on;
        self.plans_save()
    }

    /// Sweeps payouts to one of this wallet's payee keys into ordinary coins.
    pub(super) async fn sweep_payouts(&mut self) -> Result<String> {
        self.plans_ready()?;
        let state = self.live().await;
        let mut waiting: Vec<PayoutCoin> = self
            .plans
            .payouts
            .iter()
            .filter(|p| p.sweeping.is_none() && state.coins.contains(&p.id()))
            .cloned()
            .collect();
        let pk = waiting.first().map(|p| p.pk).ok_or_else(|| anyhow!("nothing to sweep"))?;
        waiting.retain(|p| p.pk == pk);
        waiting.truncate(MAX_SPEND_INPUTS);
        let total: u64 = waiting.iter().map(|p| p.value).sum();
        let inputs = payout_inputs(&waiting);
        let sig_len = bond::mss_sig_len(self.key_height(&pk));
        let (fee, _) = plans::resolve_fee(&inputs, sig_len, 0, total).ok_or_else(|| anyhow!("too little to sweep"))?;
        let (outputs, seeds) = self.change_outputs(total - fee, 0)?;
        let spend = CovSpend {
            commitment: [0; 32],
            kind: "sweep".into(),
            of: String::new(),
            inputs,
            signer: Some(pk),
            sig: None,
            pays_us: vec![],
            returned: total - fee,
            created_at: now_secs(),
        };
        let c = self.commit_spend(spend, outputs, seeds, "moving payouts into this wallet", total - fee, fee, "this wallet").await?;
        for p in &mut self.plans.payouts {
            if waiting.iter().any(|w| w.id() == p.id()) {
                p.sweeping = Some(c);
            }
        }
        self.plans_save()?;
        Ok(hex::encode(c))
    }

    // ── Sends with covenant inputs ───────────────────────────────────────

    /// `value` as fresh one-time addresses of this wallet's: outputs, and
    /// their seeds at their places after `offset` earlier outputs.
    fn change_outputs(&mut self, value: u64, offset: usize) -> Result<(Vec<OutputData>, Vec<(usize, [u8; 32])>)> {
        if value == 0 {
            return Ok((vec![], vec![]));
        }
        let denoms = decompose_value(value);
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let seeds = w.allocate_next_wots_seeds(denoms.len())?;
        let mut outputs = Vec::with_capacity(denoms.len());
        let mut change = Vec::with_capacity(denoms.len());
        for (i, (value, seed)) in denoms.iter().zip(seeds).enumerate() {
            let address = compute_address(&midstate::core::wots::keygen(&seed));
            outputs.push(OutputData::Standard { address, value: *value, salt: rand::random() });
            change.push((offset + i, seed));
        }
        Ok((outputs, change))
    }

    /// Records `spend` and writes its commit, then sends it on its way.
    #[allow(clippy::too_many_arguments)]
    async fn commit_spend(
        &mut self,
        mut spend: CovSpend,
        outputs: Vec<OutputData>,
        change_seeds: Vec<(usize, [u8; 32])>,
        what: &str,
        amount: u64,
        fee: u64,
        to: &str,
    ) -> Result<[u8; 32]> {
        let ins: Vec<[u8; 32]> = spend.inputs.iter().map(|i| i.reveal.coin_id()).collect();
        let salt: [u8; 32] = rand::random();
        let hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(&ins, &hashes, &salt);
        spend.commitment = commitment;
        self.plans.spends.push(spend);
        if let Err(e) = self.plans_save() {
            self.plans.spends.retain(|s| s.commitment != commitment);
            return Err(e.context("could not record the send, so nothing was sent"));
        }
        let now = now_secs();
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        w.data.pending.push(midstate::wallet::PendingCommit {
            commitment,
            salt,
            input_coin_ids: ins,
            outputs,
            change_seeds,
            created_at: now,
            reveal_not_before: now + 5 + rand::random::<u64>() % 11,
            is_consolidate: false,
        });
        if let Err(e) = w.save() {
            w.data.pending.retain(|p| p.commitment != commitment);
            self.plans.spends.retain(|s| s.commitment != commitment);
            let _ = self.plans_save();
            return Err(e.context("could not record the send, so nothing was sent"));
        }
        self.launch(commitment, what, amount, fee, to).await?;
        Ok(commitment)
    }

    async fn launch(&mut self, commitment: [u8; 32], what: &str, amount: u64, fee: u64, to: &str) -> Result<()> {
        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: format!("{what} — solving commit proof-of-work"),
                amount,
                fee,
                to: to.into(),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, what.to_string());
        self.broadcast_commit(commitment).await?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    pub(super) fn is_plan_send(&self, commitment: &[u8; 32]) -> bool {
        self.plans.spend(commitment).is_some()
    }

    /// What a plan send brings back into this wallet as plain coins.
    pub(super) fn plans_returned(&self, commitment: &[u8; 32]) -> Option<u64> {
        self.plans.spend(commitment).filter(|s| s.returned > 0).map(|s| s.returned)
    }

    pub(super) fn plans_meta(&self, commitment: &[u8; 32]) -> Option<(u64, u64)> {
        self.plans.spend(commitment).map(|s| {
            let ins: u64 = s.inputs.iter().map(|i| i.reveal.value).sum();
            (s.returned, ins.saturating_sub(s.returned))
        })
    }

    /// `try_reveal` for a plan send: the covenant coins opened as recorded,
    /// signed by this wallet's key where they need it. Nothing is signed if
    /// any of them has gone meanwhile.
    pub(super) async fn reveal_plan(&mut self, commitment: [u8; 32], pending: midstate::wallet::PendingCommit) {
        let Some(spend) = self.plans.spend(&commitment).cloned() else { return };
        let ins: Vec<[u8; 32]> = spend.inputs.iter().map(|i| i.reveal.coin_id()).collect();
        let hashes: Vec<[u8; 32]> = pending.outputs.iter().map(|o| o.hash_for_commitment()).collect();
        if ins != pending.input_coin_ids || compute_commitment(&ins, &hashes, &pending.salt) != commitment {
            self.set_stage(commitment, SendStage::Failed, "the send's commit does not match its record");
            return;
        }
        for id in &ins {
            if !self.node.check_coin(*id).await {
                self.plan_spend_lost(commitment, "one of its coins moved before it could go: nothing was signed");
                return;
            }
        }
        let sig = match (spend.signer, spend.sig.as_deref()) {
            (None, _) => None,
            (Some(_), Some(cached)) => match hex::decode(cached) {
                Ok(s) => Some(s),
                Err(_) => {
                    self.set_stage(commitment, SendStage::Failed, "its kept signature is damaged");
                    return;
                }
            },
            (Some(pk), None) => {
                if let Err(e) = self.verify_mss_indices().await {
                    self.set_stage(commitment, SendStage::Failed, format!("could not sign: {e:#}. Nothing was spent."));
                    return;
                }
                let Some(w) = self.wallet.as_mut() else { return };
                match w.auto_sign(&pk, &commitment) {
                    Ok(s) => {
                        // Kept before it leaves, so a retry sends the same
                        // signature instead of spending another leaf.
                        if let Some(x) = self.plans.spends.iter_mut().find(|x| x.commitment == commitment) {
                            x.sig = Some(hex::encode(&s));
                        }
                        if let Err(e) = self.plans_save() {
                            self.set_stage(commitment, SendStage::Failed, format!("could not record the signature: {e:#}"));
                            return;
                        }
                        Some(s)
                    }
                    Err(e) => {
                        self.set_stage(commitment, SendStage::Failed, format!("could not sign: {e:#}. Nothing was spent."));
                        return;
                    }
                }
            }
        };
        let witnesses = match witnesses(&spend.inputs, sig.as_deref()) {
            Ok(w) => w,
            Err(e) => {
                self.set_stage(commitment, SendStage::Failed, format!("{e:#}"));
                return;
            }
        };
        let Some(ours) = pending.outputs.iter().find_map(|o| o.coin_id()) else {
            self.set_stage(commitment, SendStage::Failed, "the send has no output to watch for");
            return;
        };
        let tx = Transaction::Reveal {
            inputs: spend.inputs.iter().map(|i| i.reveal.clone()).collect(),
            witnesses,
            outputs: pending.outputs.clone(),
            salt: pending.salt,
        };
        match self.node.send_transaction(tx).await {
            Ok(()) => {
                self.set_stage(commitment, SendStage::RevealPending, "broadcast — waiting for confirmation");
                // Confirmed by its own output appearing: someone else may
                // spend the same coins (a claim, a collection, a cancel).
                self.spawn_output_monitor(commitment, ours, spend.contested());
            }
            Err(e) => self.set_stage(
                commitment,
                SendStage::Failed,
                format!("broadcast failed: {e}. The send is kept; retry it from the Send tab."),
            ),
        }
    }

    fn plan_spend_lost(&mut self, commitment: [u8; 32], why: &str) {
        if let Some(w) = self.wallet.as_mut() {
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
        }
        self.plans.spends.retain(|s| s.commitment != commitment);
        for p in &mut self.plans.payouts {
            if p.sweeping == Some(commitment) {
                p.sweeping = None;
            }
        }
        let _ = self.plans_save();
        self.set_stage(commitment, SendStage::Failed, why.to_string());
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    /// After `complete_reveal`: a plan send has confirmed.
    pub(super) fn plans_confirmed(&mut self, commitment: [u8; 32]) {
        let Some(spend) = self.plans.spend(&commitment).cloned() else { return };
        self.plans.spends.retain(|s| s.commitment != commitment);
        if spend.kind == "sweep" {
            self.plans.payouts.retain(|p| p.sweeping != Some(commitment));
        }
        for mut p in spend.pays_us {
            if !self.plans.payouts.iter().any(|x| x.id() == p.id()) {
                p.found_at = now_secs();
                self.plans.payouts.push(p);
            }
        }
        // Coins this wallet moved are not someone else's doing.
        let gone: HashSet<[u8; 32]> = spend.inputs.iter().map(|i| i.reveal.coin_id()).collect();
        if let Some((seen, coins)) = self.plans.live.get(&spend.of).cloned() {
            let ids = self.coin_ids_for(&spend.of, &coins);
            let keep: Vec<PlanCoin> =
                coins.into_iter().zip(ids).filter(|(_, id)| !gone.contains(id)).map(|(c, _)| c).collect();
            self.plans.live.insert(spend.of.clone(), (seen, keep));
        }
        if spend.kind == "close" {
            for o in &mut self.plans.owned {
                if plan_of(&o.plan).is_some_and(|p| id_of(&p.to_bytes()) == spend.of) {
                    o.status = "closed".into();
                    o.note = "Closed: its coins are back in this wallet.".into();
                }
            }
        }
        if spend.kind == "check-in" {
            for o in &mut self.plans.owned {
                if plan_of(&o.plan).is_some_and(|p| id_of(&p.to_bytes()) == spend.of) {
                    o.note = "Checked in.".into();
                }
            }
        }
        let _ = self.plans_save();
    }

    // ── Keeping up with the chain ────────────────────────────────────────

    pub(super) async fn tick_plans(&mut self, tip: u64) {
        if self.wallet.is_none() || available(tip).is_err() {
            return;
        }
        // Once a block: nothing a plan or stream does changes in between.
        if self.plans.ticked_at == tip {
            return;
        }
        self.plans.ticked_at = tip;
        let state = self.live().await;
        let live = |id: &[u8; 32]| state.coins.contains(id);
        let mut changed = false;

        // Funding sends that have landed.
        for s in &mut self.plans.sent {
            if s.status == "funding" {
                if let Some(st) = stream_of(&s.stream) {
                    if !live_tranches(&st, live).is_empty() {
                        s.status = "active".into();
                        s.note = "Funded. Give the recipient the stream's code if you have not.".into();
                        changed = true;
                    }
                }
            }
        }
        for o in &mut self.plans.owned {
            if o.note.starts_with("Funding") {
                if let Some(p) = plan_of(&o.plan) {
                    if !live_plan_coins(&p, tip, live).is_empty() {
                        o.note = "Funded. Give each heir the plan's code.".into();
                        changed = true;
                    }
                }
            }
        }

        // Plans and streams paying us: coins that went since the last look.
        let payees: HashSet<[u8; 32]> = self.plans.payee_keys.iter().copied().collect();
        let mut watched: Vec<(String, Vec<PlanCoin>, Vec<([u8; 32], [u8; 32])>, u64, String)> = Vec::new();
        for h in &self.plans.held {
            let Some(p) = plan_of(&h.plan) else { continue };
            let id = id_of(&p.to_bytes());
            let now = live_plan_coins(&p, tip, live);
            let payouts: Vec<([u8; 32], [u8; 32])> = p
                .heirs
                .iter()
                .enumerate()
                .filter(|(_, x)| payees.contains(&x.pk))
                .flat_map(|(i, x)| p.epochs_to(tip).map(move |k| (k, i, x.pk)))
                .map(|(k, i, pk)| (p.payout_tag(k, i), pk))
                .collect();
            let earliest = now.iter().map(|c| p.deadline(c.k)).min().unwrap_or(tip);
            watched.push((id, now, payouts, earliest, format!("the plan “{}”", h.label)));
        }
        for r in &self.plans.received {
            let Some(s) = stream_of(&r.stream) else { continue };
            let id = id_of(&s.to_bytes());
            let now: Vec<PlanCoin> = live_tranches(&s, live)
                .into_iter()
                .map(|(j, n, value)| PlanCoin { k: j as u64, n, value })
                .collect();
            let payouts = (0..s.count).map(|j| (s.payout_tag(j), s.recipient_pk)).collect();
            watched.push((id, now, payouts, s.first, format!("the stream “{}”", r.label)));
        }
        for (id, now, payouts, _, what) in watched {
            let before = self.plans.live.get(&id).cloned();
            if let Some((seen_at, was)) = before {
                let gone: Vec<PlanCoin> = was.iter().copied().filter(|c| !now.contains(c)).collect();
                if !gone.is_empty() {
                    let ids = self.coin_ids_for(&id, &gone);
                    self.plans.vanished.push(Vanished {
                        of: id.clone(),
                        coins: ids,
                        payouts: payouts.iter().map(|(t, k)| (hex::encode(t), hex::encode(k))).collect(),
                        from: seen_at.saturating_sub(1),
                        what: what.clone(),
                    });
                    changed = true;
                }
            }
            if self.plans.live.get(&id).is_none_or(|(_, c)| c != &now) {
                self.plans.live.insert(id, (tip, now));
                changed = true;
            } else if let Some(e) = self.plans.live.get_mut(&id) {
                e.0 = tip;
            }
        }

        // Find what moved them, a budget of blocks at a time.
        if let Some(v) = self.plans.vanished.first().cloned() {
            let (found, next) = self.find_spend_of(&v.coins, v.from, tip).await;
            match found {
                Some(tx) => {
                    let pairs: Vec<([u8; 32], [u8; 32])> = v
                        .payouts
                        .iter()
                        .filter_map(|(t, k)| Some((parse_hex32(t).ok()?, parse_hex32(k).ok()?)))
                        .collect();
                    for mut p in payouts_in(&tx, &pairs, &v.what) {
                        if !self.plans.payouts.iter().any(|x| x.id() == p.id()) && state.coins.contains(&p.id()) {
                            p.found_at = now_secs();
                            self.plans.payouts.push(p);
                        }
                    }
                    self.plans.vanished.remove(0);
                }
                None if next >= tip => {
                    // Searched to the tip: whatever moved them is not in a
                    // block this node has (a pruned one), so stop looking.
                    self.plans.vanished.remove(0);
                }
                None => self.plans.vanished[0].from = next,
            }
            changed = true;
        }

        // Sends whose coins have gone: confirmed if their own output is on
        // chain (the monitor may have given up), lost if someone else's
        // spend took the coins.
        let gone: Vec<(CovSpend, Option<[u8; 32]>)> = self
            .plans
            .spends
            .iter()
            .filter(|s| s.inputs.iter().any(|i| !state.coins.contains(&i.reveal.coin_id())))
            .map(|s| {
                let ours = self
                    .wallet
                    .as_ref()
                    .and_then(|w| w.find_pending(&s.commitment))
                    .and_then(|p| p.outputs.iter().find_map(|o| o.coin_id()));
                (s.clone(), ours)
            })
            .collect();
        for (s, ours) in gone {
            match ours {
                // Past the time the send's own monitor would have said so.
                Some(o) if state.coins.contains(&o) => {
                    if now_secs().saturating_sub(s.created_at) > 1_800 {
                        let tx = self.self_tx.clone();
                        let c = s.commitment;
                        tokio::spawn(async move {
                            let _ = tx.send(super::Cmd::Internal(super::Internal::RevealConfirmed(c))).await;
                        });
                    }
                }
                Some(_) if now_secs().saturating_sub(s.created_at) > 3_600 => {
                    self.plan_spend_lost(s.commitment, "someone else moved its coins first");
                }
                Some(_) => {}
                // Its commit is no longer pending: completed or dropped.
                None => {
                    self.plans.spends.retain(|x| x.commitment != s.commitment);
                    changed = true;
                }
            }
        }

        if changed {
            let _ = self.plans_save();
            let _ = self.events.send(WalletEvent::WalletChanged);
        }

        if self.node.is_syncing() {
            return;
        }
        // Collect what is due before a payer can take it back.
        let due: Vec<String> = self
            .plans
            .received
            .iter()
            .filter(|r| r.auto_collect)
            .filter_map(|r| stream_of(&r.stream))
            .filter(|s| {
                live_tranches(s, live)
                    .iter()
                    .any(|(j, n, _)| s.due(*j) <= tip && !self.plans.spending(&s.input(*j, *n).coin_id()))
            })
            .map(|s| id_of(&s.to_bytes()))
            .collect();
        for id in due {
            if let Err(e) = self.collect_stream(&id).await {
                tracing::debug!("collecting stream {id}: {e:#}");
            }
        }
        // And move payouts into ordinary coins, a batch at a time: they are
        // safe where they are (only this wallet's key spends them), and every
        // sweep takes one of that key's signatures.
        let waiting: Vec<&PayoutCoin> =
            self.plans.payouts.iter().filter(|p| p.sweeping.is_none() && state.coins.contains(&p.id())).collect();
        let ripe = waiting.len() >= 8 || waiting.iter().any(|p| now_secs().saturating_sub(p.found_at) >= SWEEP_AFTER);
        if ripe && !self.plans.spends.iter().any(|s| s.kind == "sweep") {
            if let Err(e) = self.sweep_payouts().await {
                tracing::debug!("sweeping payouts: {e:#}");
            }
        }
    }

    /// The coin ids of `coins` of the plan or stream `id`.
    fn coin_ids_for(&self, id: &str, coins: &[PlanCoin]) -> Vec<[u8; 32]> {
        if let Some(p) = self.held_plan(id) {
            return coins.iter().map(|c| c.id(&p)).collect();
        }
        if let Ok(s) = self.received_stream(id) {
            return coins.iter().map(|c| s.input(c.k as u32, c.n).coin_id()).collect();
        }
        Vec::new()
    }

    /// The transaction in `from..` (a budget of blocks, to `tip`) whose reveal
    /// spends any of `coins`, and where the search got to.
    async fn find_spend_of(&self, coins: &[[u8; 32]], from: u64, tip: u64) -> (Option<Transaction>, u64) {
        let from = from.max(1);
        let to = (from + SCAN_BUDGET).min(tip + 1);
        let node = self.node.clone();
        let wanted: HashSet<[u8; 32]> = coins.iter().copied().collect();
        let found = tokio::task::spawn_blocking(move || {
            for h in from..to {
                if let Ok(Some(batch)) = node.storage.batches.load(h) {
                    for tx in &batch.transactions {
                        if let Transaction::Reveal { inputs, .. } = tx {
                            if inputs.iter().any(|i| wanted.contains(&i.coin_id())) {
                                return Some(tx.clone());
                            }
                        }
                    }
                }
            }
            None
        })
        .await
        .unwrap_or(None);
        (found, to)
    }
}
