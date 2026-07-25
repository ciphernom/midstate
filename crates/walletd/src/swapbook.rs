//! Live cross-chain swaps: the state that must survive a restart, and the
//! rules for advancing it.
//!
//! A swap is the one place in this wallet where losing track of state loses
//! money. Between the two legs there is always a window in which one side has
//! committed and the other has not, and the only thing standing between that
//! window and a loss is a wallet that keeps watching. So every transition is
//! written to disk before the action that causes it, and the watcher is
//! written to be resumable from any point.
//!
//! Two roles, mirror images of each other:
//!
//! * **Taker** — takes a published order. Locks ETH against the order's hash,
//!   waits for the maker to claim it (which publishes the preimage), then
//!   sweeps the MDS. If the maker never claims, refunds the ETH.
//! * **Maker** — published the order and holds the secret. Watches for an
//!   incoming lock against one of its hashes, verifies it, then claims the ETH
//!   — which is what releases the preimage the taker needs.
//!
//! The maker is always the one who reveals, and always on Base. That is why
//! the Base escrow must expire first: after the reveal the taker still has to
//! act on midstate.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Taker,
    Maker,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    /// Taker: `lock()` broadcast, waiting for the receipt that carries the
    /// swap id. The id folds in `block.timestamp`, so it cannot be known
    /// before the transaction is mined.
    LockingEth { tx: String },
    /// Taker: ETH is escrowed. Waiting for the maker to claim and reveal.
    EthLocked { swap_id: String },
    /// Maker: an incoming lock has been seen and verified, `claim()` sent.
    ClaimingEth { swap_id: String, tx: String },
    /// Either side: the preimage is known and the MDS leg is being taken.
    SweepingMds { preimage: String, commitment: String },
    /// The sweep reveal has been broadcast, but the coins are not in the UTXO
    /// set yet. Broadcasting only means a node accepted the transaction into
    /// its mempool — it can still fail to mine, so this is deliberately NOT
    /// treated as collected.
    ConfirmingMds { commitment: String, sent_height: u64 },
    /// Taker: the maker went quiet; the escrow has been refunded.
    RefundingEth { swap_id: String, tx: String },
    Done { note: String },
    Failed { reason: String },
}

impl Phase {
    pub fn label(&self) -> &'static str {
        match self {
            Phase::LockingEth { .. } => "locking ETH",
            Phase::EthLocked { .. } => "waiting for the seller",
            Phase::ClaimingEth { .. } => "claiming ETH",
            Phase::SweepingMds { .. } => "collecting MDS",
            Phase::ConfirmingMds { .. } => "confirming",
            Phase::RefundingEth { .. } => "refunding",
            Phase::Done { .. } => "done",
            Phase::Failed { .. } => "failed",
        }
    }
    pub fn settled(&self) -> bool {
        matches!(self, Phase::Done { .. } | Phase::Failed { .. })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Swap {
    /// Stable local id.
    pub id: String,
    pub role: Role,
    /// The hash both legs lock against.
    pub secret_hash: String,
    /// Known to the maker from the start; learned by the taker from the
    /// maker's claim.
    pub preimage: Option<String>,
    pub mds_value: u64,
    pub wei: String,
    /// Order group this unit belongs to, for the maker's bookkeeping.
    pub group_id: String,
    /// Where the MDS sits: covenant parameters needed to rebuild the address.
    pub max_claim: u64,
    pub mds_timeout_height: u64,
    pub refund_pk: String,
    pub salt: String,
    /// Counterparty's Base address — the beneficiary of the escrow.
    pub counterparty_evm: String,
    /// Unix time the Base escrow may be refunded.
    pub eth_deadline: u64,
    /// Where swept MDS lands. Fixed at the moment the sweep is first built:
    /// the commitment covers this address, so regenerating it later would
    /// produce a different transaction than the one already committed.
    #[serde(default)]
    pub sweep_dest: Option<String>,
    /// The exact covenant bytecode holding the MDS, hex.
    ///
    /// Two different scripts can be swept: a limit-order covenant (a maker's
    /// published ask) and a covenant HTLC (a seller filling our bid). They
    /// take the same claim witness but are built from different parameters, so
    /// storing the compiled bytecode is simpler and safer than trying to
    /// re-derive which shape this was.
    #[serde(default)]
    pub covenant_hex: Option<String>,
    pub phase: Phase,
    pub created: u64,
    pub updated: u64,
    /// Unix time before which the sweep should not be retried.
    ///
    /// A transient failure — node unreachable, commitment not yet visible — is
    /// worth retrying, but not at the 2-second tick rate, and not forever
    /// without saying so. `#[serde(default)]` keeps wallets written by earlier
    /// builds loadable; they simply start with no backoff owed.
    #[serde(default)]
    pub sweep_retry_at: u64,
    /// Consecutive failed sweep attempts, for the backoff schedule.
    #[serde(default)]
    pub sweep_attempts: u32,
}

impl Swap {
    /// Whether this swap still needs the watcher to do something.
    pub fn active(&self) -> bool {
        !self.phase.settled()
    }

    /// A taker whose maker has gone quiet should refund rather than wait
    /// forever. Left a little late deliberately: refunding early would forfeit
    /// a swap the maker might still complete.
    pub fn should_refund(&self, now: u64) -> bool {
        matches!(self.phase, Phase::EthLocked { .. }) && now >= self.eth_deadline
    }

    /// Whether the sweep may be attempted on this tick.
    pub fn sweep_due(&self, now: u64) -> bool {
        now >= self.sweep_retry_at
    }

    /// Record a transient failure and push the next attempt out.
    ///
    /// Doubling from 2 seconds, capped at 5 minutes. The cap matters: a sweep
    /// that keeps failing for a recoverable reason still has to be retried
    /// often enough to land inside the claim window.
    pub fn defer_sweep(&mut self, now: u64) {
        self.sweep_attempts = self.sweep_attempts.saturating_add(1);
        let backoff = 2u64
            .saturating_pow(self.sweep_attempts.min(8))
            .min(300);
        self.sweep_retry_at = now + backoff;
    }

    /// Clear the backoff after a successful attempt.
    pub fn sweep_ok(&mut self) {
        self.sweep_attempts = 0;
        self.sweep_retry_at = 0;
    }
}

/// Persisted swap state. Kept beside the wallet, like the channel book.
#[derive(Default, Serialize, Deserialize)]
pub struct SwapBook {
    #[serde(default)]
    pub swaps: Vec<Swap>,
    /// Base block already scanned for events relevant to our swaps.
    #[serde(default)]
    pub base_cursor: u64,
    #[serde(skip)]
    dirty: bool,
}

impl SwapBook {
    pub fn path_for(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("swaps.json")
    }
    pub fn load(wallet_path: &Path) -> Self {
        std::fs::read_to_string(Self::path_for(wallet_path))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    /// Always called before the action a transition describes, never after —
    /// a crash between the two must leave the wallet believing it has done
    /// less than it has, not more.
    pub fn save(&mut self, wallet_path: &Path) {
        if let Ok(s) = serde_json::to_string(self) {
            let _ = std::fs::write(Self::path_for(wallet_path), s);
            self.dirty = false;
        }
    }
    pub fn touch(&mut self) {
        self.dirty = true;
    }
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
    pub fn find_mut(&mut self, id: &str) -> Option<&mut Swap> {
        self.swaps.iter_mut().find(|s| s.id == id)
    }
    pub fn by_hash_mut(&mut self, secret_hash: &str) -> Option<&mut Swap> {
        self.swaps.iter_mut().find(|s| s.secret_hash == secret_hash)
    }
    pub fn active(&self) -> impl Iterator<Item = &Swap> {
        self.swaps.iter().filter(|s| s.active())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn swap(phase: Phase, deadline: u64) -> Swap {
        Swap {
            id: "s1".into(),
            role: Role::Taker,
            secret_hash: "aa".into(),
            preimage: None,
            mds_value: 1024,
            wei: "5000".into(),
            group_id: "g".into(),
            max_claim: 1024,
            mds_timeout_height: 300_000,
            refund_pk: "bb".into(),
            salt: "cc".into(),
            counterparty_evm: "0x1".into(),
            eth_deadline: deadline,
            sweep_dest: None,
            covenant_hex: None,
            phase,
            created: 0,
            updated: 0,
            sweep_retry_at: 0,
            sweep_attempts: 0,
        }
    }

    #[test]
    fn sweep_backoff_grows_but_stays_inside_the_claim_window() {
        let mut s = swap(Phase::SweepingMds { preimage: "p".into(), commitment: String::new() }, 0);
        assert!(s.sweep_due(0), "a fresh swap is due immediately");

        s.defer_sweep(100);
        assert!(!s.sweep_due(101), "a deferred sweep must not retry on the next tick");

        // However long it keeps failing, the gap has to stay short enough that
        // a recovered node still gets many attempts before the lock expires.
        for _ in 0..50 {
            s.defer_sweep(1_000);
        }
        assert!(s.sweep_retry_at - 1_000 <= 300, "backoff must stay capped");

        s.sweep_ok();
        assert_eq!(s.sweep_attempts, 0);
        assert!(s.sweep_due(0));
    }

    #[test]
    fn only_an_escrowed_taker_refunds_and_only_after_the_deadline() {
        let s = swap(Phase::EthLocked { swap_id: "x".into() }, 1_000);
        assert!(!s.should_refund(999), "refunding early forfeits a swap still in play");
        assert!(s.should_refund(1_000));

        // Nothing else should ever trigger a refund — least of all a swap that
        // already completed.
        for p in [
            Phase::LockingEth { tx: "t".into() },
            Phase::SweepingMds { preimage: "p".into(), commitment: "c".into() },
            Phase::ConfirmingMds { commitment: "c".into(), sent_height: 1 },
            Phase::Done { note: "n".into() },
            Phase::Failed { reason: "r".into() },
        ] {
            assert!(!swap(p, 1_000).should_refund(9_999));
        }
    }

    #[test]
    fn a_broadcast_sweep_is_not_a_settled_swap() {
        // Accepting a transaction into a mempool is not the same as mining it,
        // so this phase must keep the watcher engaged.
        let s = swap(Phase::ConfirmingMds { commitment: "c".into(), sent_height: 10 }, 1);
        assert!(s.active(), "a broadcast reveal still needs watching");
        assert!(!s.phase.settled());
        assert!(!s.should_refund(9_999), "the ETH leg is already done here");
    }

    #[test]
    fn settled_swaps_drop_out_of_the_watcher() {
        let mut b = SwapBook::default();
        b.swaps.push(swap(Phase::EthLocked { swap_id: "x".into() }, 1));
        b.swaps.push(swap(Phase::Done { note: "n".into() }, 1));
        b.swaps.push(swap(Phase::Failed { reason: "r".into() }, 1));
        assert_eq!(b.active().count(), 1);
    }

    #[test]
    fn every_phase_reads_as_something_a_person_understands() {
        for p in [
            Phase::LockingEth { tx: "t".into() },
            Phase::EthLocked { swap_id: "x".into() },
            Phase::ClaimingEth { swap_id: "x".into(), tx: "t".into() },
            Phase::SweepingMds { preimage: "p".into(), commitment: "c".into() },
            Phase::ConfirmingMds { commitment: "c".into(), sent_height: 1 },
            Phase::RefundingEth { swap_id: "x".into(), tx: "t".into() },
            Phase::Done { note: "n".into() },
            Phase::Failed { reason: "r".into() },
        ] {
            let l = p.label();
            assert!(!l.is_empty() && !l.contains('_'), "phase label {l:?} leaks internals");
        }
    }

    #[test]
    fn a_covenant_bound_sweep_keeps_its_mandated_destination() {
        // A covenant HTLC only releases if the spend pays its receiver, so the
        // destination is not ours to pick. Losing it across a restart would
        // build a transaction the script rejects.
        let mut s = swap(Phase::SweepingMds { preimage: "p".into(), commitment: String::new() }, 1);
        s.sweep_dest = Some("f00d".into());
        s.covenant_hex = Some("beef".into());
        let restored: Swap = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(restored.sweep_dest.as_deref(), Some("f00d"));
        assert_eq!(restored.covenant_hex.as_deref(), Some("beef"));
    }

    #[test]
    fn a_sweep_destination_is_chosen_once_and_kept() {
        // The commitment covers the destination address, so re-deriving it on
        // a later pass would produce a transaction that no longer matches the
        // commitment already paid for.
        let mut s = swap(Phase::SweepingMds { preimage: "p".into(), commitment: String::new() }, 1);
        assert!(s.sweep_dest.is_none(), "no destination until the sweep begins");
        s.sweep_dest = Some("abcd".into());
        let restored: Swap = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(restored.sweep_dest.as_deref(), Some("abcd"));
    }

    #[test]
    fn lookup_by_hash_is_how_an_incoming_lock_finds_its_swap() {
        let mut b = SwapBook::default();
        let mut s = swap(Phase::EthLocked { swap_id: "x".into() }, 1);
        s.secret_hash = "deadbeef".into();
        b.swaps.push(s);
        assert!(b.by_hash_mut("deadbeef").is_some());
        assert!(b.by_hash_mut("other").is_none());
    }
}
