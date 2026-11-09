//! The walletd actor's side of co-bonding with a pool by its address
//! (`crate::co_bond`). A child of `service`, sharing the actor's state, as
//! `service_rental` is.
//!
//! A bond locked for a pool waits in phase "to_pool" until its lock confirms.
//! Then the wallet hands it in, off the actor (an unanswering pool must not
//! hold the wallet up), and it moves to "pool": the pool's midwimble node
//! registers it. While it does, the wallet reads the pool's progress now and
//! then for the bond's card. It becomes "on_chain" as any bond does, once the
//! embedded midwimble node sees it registered (`advance_registrations`).

use super::{now_secs, Cmd, Internal, Service};
use crate::api::WalletEvent;
use crate::bond::RegStatus;
use crate::co_bond::{self, HandedIn};

/// Seconds between attempts to hand a bond in, after one fails.
const HAND_IN_RETRY: u64 = 300;
/// Seconds between reads of a pool's progress with a bond handed in.
const STATUS_EVERY: u64 = 600;

/// What came back from a pool, for the actor.
pub(super) enum PoolReply {
    HandedIn(Result<HandedIn, String>),
    Status(Result<Option<(String, u32, String)>, String>),
}

impl Service {
    /// Starts, off the actor, the next pool call a bond is due: handing in a
    /// bond whose lock has confirmed, or reading the pool's progress with
    /// one handed in. One at a time, each spaced out by `reg.checked`.
    pub(super) async fn tick_co_bonds(&mut self) {
        if self.co_bond_call.is_some() || self.wallet.is_none() {
            return;
        }
        let now = now_secs();
        let state = self.node.get_state().await;
        let due = self.bonds.bonds.iter().position(|b| {
            b.pool.is_some()
                && match b.reg.phase.as_str() {
                    "to_pool" => state.coins.contains(&b.coin_id()) && now >= b.reg.checked + HAND_IN_RETRY,
                    "pool" => now >= b.reg.checked + STATUS_EVERY,
                    _ => false,
                }
        });
        let Some(i) = due else { return };
        let rec = self.bonds.bonds[i].clone();
        let Some(base) = rec.pool.clone() else { return };
        let id = rec.coin_id();
        self.bonds.bonds[i].reg.checked = now;
        self.co_bond_call = Some(id);
        let address = self.mwb.keys.as_ref().map(|k| k.address().encode());
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            let reply = if rec.reg.phase == "to_pool" {
                PoolReply::HandedIn(co_bond::hand_in(&base, &rec, address.as_deref()).await.map_err(|e| format!("{e:#}")))
            } else {
                let id = hex::encode(id);
                PoolReply::Status(co_bond::status(&base, &id).await.map_err(|e| format!("{e:#}")))
            };
            let _ = tx.send(Cmd::Internal(Internal::PoolReply { id, reply })).await;
        });
    }

    /// What a pool said about bond `id`.
    pub(super) fn pool_reply(&mut self, id: [u8; 32], reply: PoolReply) {
        self.co_bond_call = None;
        let now = now_secs();
        let Some(b) = self.bonds.find_mut(&id) else { return };
        let host = b.pool.as_deref().map(co_bond::pool_host).unwrap_or_default();
        match reply {
            PoolReply::HandedIn(Ok(h)) => {
                let mut note = format!("Handed in: {host} registers it, which takes about a day.");
                if h.address.is_none() {
                    note.push_str(" No MDW address went with it, so it earns no rewards.");
                }
                if !h.note.is_empty() {
                    note = format!("{note} {}", h.note);
                }
                b.reg = RegStatus { phase: "pool".into(), progress: 0, note, checked: now };
            }
            PoolReply::HandedIn(Err(e)) => {
                b.reg.note = format!("Handing it in to {host} failed: {e}. The wallet tries again in a few minutes.");
            }
            PoolReply::Status(Ok(Some((phase, _, _)))) if phase == "registered" => {
                // Midwimble holds it. The embedded node confirms as much
                // (`advance_registrations`) when it runs; the pool's word
                // stands when it does not.
                b.reg = RegStatus {
                    phase: "on_chain".into(),
                    progress: 100,
                    note: format!("Registered: {host} signs with it."),
                    checked: now,
                };
            }
            PoolReply::Status(Ok(Some((phase, progress, note)))) => {
                b.reg.progress = progress;
                b.reg.note = match phase.as_str() {
                    "burying" => format!("{host} is registering it: about a day in all, {progress}% done."),
                    "ready" => format!("{host} registers it with its next midwimble block."),
                    "expired" => "Its lock has run down too far to count.".into(),
                    _ if !note.is_empty() => format!("{host}: {note}"),
                    _ => format!("{host} is registering it."),
                };
            }
            PoolReply::Status(Ok(None)) => {
                // The pool lost it (a fresh node, say): hand it in again.
                b.reg = RegStatus {
                    phase: "to_pool".into(),
                    progress: 0,
                    note: format!("{host} no longer lists it, so the wallet hands it in again."),
                    checked: 0,
                };
            }
            PoolReply::Status(Err(_)) => {
                // Asked again later; the bond is safe where it is.
            }
        }
        if let Err(e) = self.bonds.save(&self.wallet_path) {
            tracing::warn!("saving bond records: {e:#}");
        }
        let _ = self.events.send(WalletEvent::WalletChanged);
    }
}
