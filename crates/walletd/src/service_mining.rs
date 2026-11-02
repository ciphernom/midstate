//! The walletd actor's side of mining MDW (`mwb_mining`): registering this
//! wallet's bonds with midwimble as the chain allows, and the merged-mining
//! endpoint the local pool signs its midwimble blocks through.
//!
//! A child of `service`, sharing the actor's state, as `service_mwb` is.

use super::{now_secs, Service};
use crate::api::{BondRulesView, MiningView};
use crate::bond::{self, RegStatus};
use crate::mining::MdwMining;
use crate::mwb_mining::{
    bond_coin, miner_bond, mining_key_of, mining_secret, parse_bond_id, register_once, solo_blocker, ChainNow,
    MergeServer, MERGE_POOL,
};
use anyhow::{anyhow, Result};
use curve25519_dalek::scalar::Scalar;
use midwimble::core::bond::MinerBond;

/// Seconds between registration attempts, by where the bond stands. A run
/// reads up to a day of midstate's headers, so it is not repeated lightly;
/// a ready bond is checked now and then for a proof about to expire.
fn retry_after(phase: &str) -> Option<u64> {
    match phase {
        "waiting" | "failed" => Some(FAILED_RETRY),
        "burying" => Some(600),
        "ready" => Some(1_800),
        _ => None,
    }
}

/// Seconds before a failed attempt is retried.
const FAILED_RETRY: u64 = 120;

/// The `checked` time that makes a bond in `phase` due again `secs` from
/// `now`.
fn retry_in(phase: &str, now: u64, secs: u64) -> u64 {
    (now + secs).saturating_sub(retry_after(phase).unwrap_or(0))
}

impl Service {
    /// This wallet's midwimble mining secret, from its recovery phrase.
    pub(super) fn mining_secret(&self) -> Result<Scalar> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let master = w
            .data
            .master_seed
            .ok_or_else(|| anyhow!("mining needs a wallet with a recovery phrase, and this one predates them"))?;
        Ok(mining_secret(&master))
    }

    pub(super) fn mining_key(&self) -> Result<[u8; 32]> {
        Ok(mining_key_of(&self.mining_secret()?))
    }

    /// The bond the wallet mines with now, if one is ready.
    pub(super) fn wallet_miner_bond(&self) -> Option<MinerBond> {
        let secret = self.mining_secret().ok()?;
        miner_bond(&secret, &mining_key_of(&secret), &self.bonds.bonds, now_secs())
    }

    /// Midwimble's bond rules now, at its chain's difficulty.
    pub(super) fn bond_rules(&self) -> BondRulesView {
        let now = now_secs();
        let floor = bond::midwimble::min_bond(now);
        let registration_min = self
            .mwb_node
            .as_ref()
            .map_or(floor, |h| bond::midwimble::registration_min(&h.state().target, now));
        BondRulesView {
            mining_key: self.mining_key().map(hex::encode).unwrap_or_default(),
            min_bond: floor,
            registration_min,
            max_bond: bond::midwimble::MAX_BOND,
            min_lock_blocks: bond::midwimble::min_remaining_lock(now),
            horizon: bond::midwimble::horizon(now),
            est_midstate_height: bond::midwimble::est_midstate_height(now),
            rpc_on: self.rpc_url.is_some(),
        }
    }

    /// Asks the wallet to register bond `id` with midwimble, and makes a
    /// first attempt now if it can.
    pub(super) async fn register_bond(&mut self, id: &str) -> Result<String> {
        let raw = parse_bond_id(id)?;
        let rec = self.bonds.find_mut(&raw).ok_or_else(|| anyhow!("no such bond"))?;
        if rec.reg.phase == "on_chain" {
            return Ok("Midwimble already holds this bond.".into());
        }
        let now = now_secs();
        if bond::midwimble::counts_until(rec.bonded_until, now).is_none() {
            anyhow::bail!(
                "its lock is too short: midwimble counts a bond only while more than {} days of its \
                 lock remain",
                bond::midwimble::min_remaining_lock(now) / 1440
            );
        }
        if rec.value < bond::midwimble::min_bond(now) {
            anyhow::bail!(
                "it holds {}, below the {} midwimble counts",
                bond::gmds(rec.value),
                bond::gmds(bond::midwimble::min_bond(now))
            );
        }
        if !rec.reg.active() {
            rec.reg = RegStatus {
                phase: "waiting".into(),
                note: "Registering: the wallet takes a proof once the lock confirms on midstate.".into(),
                ..RegStatus::default()
            };
        }
        rec.reg.checked = 0;
        self.bonds.save(&self.wallet_path)?;
        self.advance_registrations().await;
        let note = self.bonds.find(&raw).map(|b| b.reg.note.clone()).unwrap_or_default();
        Ok(note)
    }

    /// One step of a registration under way, for the first bond due one: a
    /// bond midwimble now holds is marked so; one whose coin is on chain is
    /// taken through `register_once`.
    pub(super) async fn advance_registrations(&mut self) {
        let Some(h) = self.mwb_node.clone() else { return };
        if !self.bonds.bonds.iter().any(|b| b.reg.active()) {
            return;
        }
        let now = now_secs();
        let state = h.state();
        let mut changed = false;
        for b in self.bonds.bonds.iter_mut().filter(|b| b.reg.active()) {
            if state.bonds.contains_key(&b.coin_id()) {
                b.reg = RegStatus {
                    phase: "on_chain".into(),
                    progress: 100,
                    note: "Registered: midwimble holds this bond, so it signs the blocks this wallet mines."
                        .into(),
                    checked: now,
                };
                changed = true;
            }
        }
        // A step needs both chains in step with their networks, and a bond
        // whose next attempt is due.
        let is_due = |b: &crate::bond::BondRecord| {
            b.reg.active() && retry_after(&b.reg.phase).is_some_and(|wait| now >= b.reg.checked.saturating_add(wait))
        };
        let mut due: Option<[u8; 32]> = None;
        if self.bonds.bonds.iter().any(is_due) && !self.node.is_syncing() && self.mwb_synced() {
            let mds = self.node.get_state().await;
            for b in self.bonds.bonds.iter_mut().filter(|b| is_due(b)) {
                if mds.coins.contains(&b.coin_id()) {
                    due = Some(b.coin_id());
                    break;
                }
                if b.seen_live {
                    // Spent (unbonded): nothing left to register.
                    b.reg = RegStatus::default();
                    changed = true;
                    continue;
                }
                let note = "Waiting for the lock to confirm on midstate.";
                if b.reg.note != note {
                    b.reg.note = note.into();
                    changed = true;
                }
                b.reg.checked = now;
            }
        }
        if let Some(id) = due {
            let rec = self.bonds.find(&id).cloned().expect("just found");
            let chain = ChainNow::of(&state, now);
            let rpc = self.midstate_rpc_addr();
            let rpc_on = self.rpc_url.is_some();
            let outcome = tokio::task::spawn_blocking(move || {
                if !rpc_on {
                    anyhow::bail!(
                        "registering reads midstate through this app's node RPC, which is off this session \
                         (MIDSTATE_RPC_PORT=none)"
                    );
                }
                register_once(bond_coin(&rec), rec.bond_file.clone(), &rpc, &chain)
            })
            .await
            .map_err(|e| anyhow!("registration task failed: {e}"))
            .and_then(|r| r);
            if let Some(b) = self.bonds.find_mut(&id) {
                match outcome {
                    Ok((file, status)) => {
                        b.bond_file = Some(file);
                        b.reg = status;
                    }
                    Err(e) if b.reg.phase == "ready" => {
                        // The registration in hand still mines until its
                        // proof ages out; a failed check only means another
                        // one soon.
                        b.reg.note = format!(
                            "Ready, though the last check of it failed: {e:#}. The wallet tries again shortly."
                        );
                        b.reg.checked = retry_in("ready", now, FAILED_RETRY);
                    }
                    Err(e) => {
                        b.reg.phase = "failed".into();
                        b.reg.note = format!("The last attempt failed: {e:#}. The wallet tries again shortly.");
                        b.reg.checked = now;
                    }
                }
                changed = true;
            }
        }
        if changed {
            if let Err(e) = self.bonds.save(&self.wallet_path) {
                tracing::warn!("saving bond records: {e:#}");
            }
            self.refresh_merge_bond();
            let _ = self.events.send(crate::api::WalletEvent::WalletChanged);
        }
    }

    /// The local pool's jobs are signed with the wallet's bond as it now
    /// stands (none while the wallet is locked).
    pub(super) fn refresh_merge_bond(&mut self) {
        let bond = self.wallet_miner_bond();
        if let Some(m) = &self.merge {
            m.set_bond(bond);
        }
    }

    /// What a mining run about to start does for MDW: the hasher names this
    /// wallet's MDW address; the local pool merge-mines too when a bond is
    /// ready, through an endpoint started for it.
    pub(super) async fn mdw_mining_args(&mut self, pool_url: &str) -> MdwMining {
        let address = self.mwb.keys.as_ref().map(|k| k.address().encode());
        let local = crate::mining::MiningSupervisor::is_local(pool_url);
        let mut merge_rpc = None;
        if local && address.is_some() {
            if let (Some(h), Some(bond)) = (self.mwb_node.clone(), self.wallet_miner_bond()) {
                if self.merge.is_none() {
                    match MergeServer::start(h, None).await {
                        Ok(m) => self.merge = Some(m),
                        Err(e) => tracing::warn!("merged mining of MDW is off: {e:#}"),
                    }
                }
                if let Some(m) = &self.merge {
                    m.set_bond(Some(bond));
                    merge_rpc = Some(m.rpc());
                }
            }
        }
        MdwMining { address, merge_rpc, outdated: false }
    }

    /// The MDW lines of the Mining panel.
    pub(super) fn fill_mdw_view(&self, view: &mut MiningView, mdw: &MdwMining) {
        view.merge_pool = MERGE_POOL.into();
        let address = self.mwb.keys.as_ref().map(|k| k.address().encode());
        view.mdw_address = mdw.address.clone().or(address.clone()).unwrap_or_default();
        let running = view.phase == "running" || view.phase == "starting";
        let bond_id = self.merge.as_ref().and_then(|m| m.bond_id());
        view.mdw_merge = running && view.local_pool && mdw.merge_rpc.is_some() && bond_id.is_some();
        view.mdw_bond = bond_id.map(hex::encode).unwrap_or_default();
        let bond = self.wallet_miner_bond();
        view.mdw_note = if view.mdw_address.is_empty() {
            "No MDW: this wallet has no MDW account yet (add it on the Receive tab).".into()
        } else if running && mdw.outdated {
            "MDW: left out of this run, since the `midstate` binary beside this app predates MDW mining. \
             Update it, then restart mining."
                .into()
        } else if !view.local_pool && running {
            "MDW: this pool pays your MDW account if it merge-mines midwimble.".into()
        } else if view.mdw_merge {
            "MDW: merge-mined with your bond. A share that also clears midwimble's target is an MDW block \
             paying your MDW account."
                .into()
        } else if view.local_pool && running {
            format!(
                "MDW: not mined here, since {}. Restart mining once a bond is ready, or mine on a pool \
                 that merge-mines midwimble.",
                solo_blocker(true, self.mwb_node.is_some(), bond.as_ref()).unwrap_or("this run started without one")
            )
        } else {
            match solo_blocker(true, self.mwb_node.is_some(), bond.as_ref()) {
                None => "MDW: your own pool will merge-mine it with your bond.".into(),
                Some(why) => format!(
                    "MDW: your own pool cannot mine it, since {why}; a pool that merge-mines midwimble \
                     pays your MDW account without one."
                ),
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempts_are_spaced_by_where_a_bond_stands() {
        let now = 1_793_584_000;
        // A ready bond whose check failed is due again as soon as a failed
        // attempt would be, and stays ready meanwhile.
        let checked = retry_in("ready", now, FAILED_RETRY);
        assert_eq!(checked + retry_after("ready").unwrap(), now + FAILED_RETRY);
        assert!(retry_after("waiting") < retry_after("burying") && retry_after("burying") < retry_after("ready"));
        assert_eq!(retry_after("on_chain"), None, "a registered bond is left alone");
        assert_eq!(retry_after(""), None, "as is one never asked to register");
    }
}
