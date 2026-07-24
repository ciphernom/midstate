//! Cross-chain swap planning, and the checks that make one safe to start.
//!
//! Two rails, chosen deliberately rather than automatically:
//!
//! * **Submarine** — the MDS leg rides a Q-Bolt channel. Instant, because a
//!   channel payment is a signed state handed over the chat bus. The whole
//!   swap then costs roughly two Base blocks.
//! * **On-chain** — the MDS leg is an ordinary HTLC. Works with no channel at
//!   all, but pays for two commit→reveal cycles on a 60-second chain, which is
//!   where essentially all of the old latency came from.
//!
//! ## The two clocks
//!
//! This is the part that loses money if it is wrong. A Q-Bolt HTLC expires at
//! a midstate **block height** (`OP_CHECKTIMEVERIFY`); the Base contract
//! expires at a **unix timestamp** (`block.timestamp + refundDelay`). They are
//! not the same units and there is no oracle between them — heights convert to
//! wall-clock only through the 60-second block target, which drifts.
//!
//! So every comparison here is done in seconds, using a *pessimistic* estimate
//! of the midstate deadline, and the required margin is deliberately large.
//! When the numbers do not fit, the plan is refused rather than narrowed:
//! a swap that starts unsafe cannot be made safe afterwards.

use anyhow::{bail, Result};

/// Midstate's block target. Real spacing wanders around this, which is exactly
/// why heights are never trusted as precise deadlines.
pub const BLOCK_SECS: u64 = 60;

/// How far the real deadline is assumed to arrive ahead of the nominal one, to
/// absorb fast blocks. 25% early means 120 nominal blocks are treated as only
/// 90 blocks of usable time.
pub const DRIFT_NUMER: u64 = 3;
pub const DRIFT_DENOM: u64 = 4;

/// Room the second actor gets after the secret becomes public. An hour is
/// generous on purpose: it must cover noticing the reveal, building a
/// transaction, and getting it mined on a 60-second chain.
pub const SETTLE_MARGIN_SECS: u64 = 3_600;

/// Default lifetime of the Base escrow. The contract permits 600s to 7 days.
pub const DEFAULT_ETH_REFUND_SECS: u64 = 3_600;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Give ETH, receive MDS.
    BuyMds,
    /// Give MDS, receive ETH.
    SellMds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rail {
    /// MDS leg over a payment channel.
    Submarine,
    /// MDS leg as an on-chain HTLC.
    OnChain,
}

/// Deadlines for both legs, already checked against each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timings {
    /// Seconds from now until the Base escrow may be refunded.
    pub eth_refund_secs: u64,
    /// Absolute unix time the Base escrow expires.
    pub eth_deadline: u64,
    /// Midstate height at which the MDS lock reverts.
    pub mds_timeout_height: u64,
    /// Pessimistic wall-clock estimate of that height.
    pub mds_deadline_est: u64,
    /// Slack between the two, in seconds, after the drift discount.
    pub margin_secs: u64,
}

/// Convert a height difference into the *least* wall-clock time it might
/// represent. Used for deadlines we depend on lasting.
pub fn blocks_to_secs_pessimistic(blocks: u64) -> u64 {
    blocks * BLOCK_SECS * DRIFT_NUMER / DRIFT_DENOM
}

/// Convert seconds into a height difference, rounding up so the resulting
/// deadline is never shorter than asked for.
pub fn secs_to_blocks_generous(secs: u64) -> u64 {
    // Undo the drift discount: to be sure of N seconds, buy N/0.75 worth.
    let padded = secs * DRIFT_DENOM / DRIFT_NUMER;
    padded.div_ceil(BLOCK_SECS)
}

/// Build deadlines for a swap.
///
/// In both directions the **maker generates the secret and reveals it on
/// Base**, so the Base escrow must expire first: after the reveal — at the
/// latest, moments before the refund unlocks — the other side still has to act
/// on midstate.
pub fn plan_timings(now: u64, tip_height: u64, eth_refund_secs: u64) -> Result<Timings> {
    if !(600..=604_800).contains(&eth_refund_secs) {
        bail!("the contract only accepts a refund delay between 10 minutes and 7 days");
    }
    let eth_deadline = now + eth_refund_secs;

    // The MDS lock has to outlast the Base escrow by the settle margin, and
    // the height is bought generously so drift cannot eat into it.
    let needed = eth_refund_secs + SETTLE_MARGIN_SECS;
    let mds_timeout_height = tip_height + secs_to_blocks_generous(needed);

    let est = now + blocks_to_secs_pessimistic(mds_timeout_height - tip_height);
    check_ordering(eth_deadline, est)?;
    Ok(Timings {
        eth_refund_secs,
        eth_deadline,
        mds_timeout_height,
        mds_deadline_est: est,
        margin_secs: est.saturating_sub(eth_deadline),
    })
}

/// The single invariant behind both swap directions: whoever reveals the
/// secret must be up against the earlier deadline.
pub fn check_ordering(reveal_deadline: u64, act_deadline: u64) -> Result<()> {
    if reveal_deadline + SETTLE_MARGIN_SECS > act_deadline {
        bail!(
            "unsafe timing: the revealing leg expires at {reveal_deadline} and the other leg \
             at {act_deadline}, leaving under {SETTLE_MARGIN_SECS}s to settle. Whoever moves \
             second could run out of time and lose the swap."
        );
    }
    Ok(())
}

/// A channel must outlive the HTLC riding inside it. If the channel closes
/// first, the hash-locked output settles on-chain and the neat instant path is
/// gone — so refuse rather than rely on that fallback.
pub fn check_channel_lifetime(channel_expiry: u64, mds_timeout_height: u64) -> Result<()> {
    let cushion = secs_to_blocks_generous(SETTLE_MARGIN_SECS);
    if channel_expiry < mds_timeout_height + cushion {
        bail!(
            "the channel expires at height {channel_expiry}, too close to the swap's lock at \
             {mds_timeout_height}. Open a channel with a longer lifetime, or use the on-chain rail."
        );
    }
    Ok(())
}

// ── Readiness ───────────────────────────────────────────────────────────

/// One prerequisite, phrased so someone can act on it.
#[derive(Clone, Debug)]
pub struct Check {
    pub label: String,
    pub ok: bool,
    /// What is true right now.
    pub detail: String,
    /// What to do about it, when it is not satisfied.
    pub fix: Option<String>,
}

impl Check {
    pub fn pass(label: &str, detail: impl Into<String>) -> Self {
        Self { label: label.into(), ok: true, detail: detail.into(), fix: None }
    }
    pub fn fail(label: &str, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { label: label.into(), ok: false, detail: detail.into(), fix: Some(fix.into()) }
    }
}

/// Everything that must hold before a swap can start, gathered so the wallet
/// can explain "not yet, and here is why" instead of failing mid-flight.
pub struct Prereqs {
    pub side: Side,
    pub rail: Rail,
    pub synced: bool,
    pub has_evm_key: bool,
    pub eth_balance_wei: Option<u128>,
    /// Wei the trade itself needs (buying) plus gas.
    pub wei_needed: u128,
    pub mds_spendable: u64,
    pub mds_needed: u64,
    /// Outbound channel capacity toward the counterparty, if a channel exists.
    pub channel_capacity: Option<u64>,
    pub channel_expiry: Option<u64>,
    pub tip_height: u64,
    pub mds_timeout_height: u64,
}

impl Prereqs {
    pub fn evaluate(&self) -> Vec<Check> {
        let mut v = Vec::new();

        v.push(if self.synced {
            Check::pass("Node synced", "Your node is at the chain tip.")
        } else {
            Check::fail(
                "Node synced",
                "Still catching up with the network.",
                "Wait for the sync to finish — swaps depend on seeing locks the moment they land.",
            )
        });

        v.push(if self.has_evm_key {
            Check::pass("Base account", "Derived from your recovery phrase.")
        } else {
            Check::fail(
                "Base account",
                "This wallet has no Base account.",
                "It predates cross-chain support. Restore from your recovery phrase into a new wallet.",
            )
        });

        // Gas is needed on both sides: the buyer locks, the seller claims.
        match self.eth_balance_wei {
            Some(bal) if bal >= self.wei_needed => v.push(Check::pass(
                "ETH available",
                format!("{bal} wei, need about {}", self.wei_needed),
            )),
            Some(bal) => v.push(Check::fail(
                "ETH available",
                format!("{bal} wei, need about {}", self.wei_needed),
                "Send ETH to your Base account above. Both sides of a swap pay gas.",
            )),
            None => v.push(Check::fail(
                "ETH available",
                "Could not reach the Base endpoint.",
                "Check the RPC setting under Connection.",
            )),
        }

        if self.side == Side::SellMds {
            v.push(if self.mds_spendable >= self.mds_needed {
                Check::pass("MDS available", format!("{} spendable", self.mds_spendable))
            } else {
                Check::fail(
                    "MDS available",
                    format!("{} spendable, need {}", self.mds_spendable, self.mds_needed),
                    "Wait for coins to confirm, or defrag on the Coins tab if they are fragmented.",
                )
            });
        }

        if self.rail == Rail::Submarine {
            match (self.channel_capacity, self.channel_expiry) {
                (Some(cap), Some(exp)) => {
                    v.push(if cap >= self.mds_needed {
                        Check::pass("Channel capacity", format!("{cap} units toward this peer"))
                    } else {
                        Check::fail(
                            "Channel capacity",
                            format!("{cap} units available, need {}", self.mds_needed),
                            "Open a larger channel to this peer, or switch to the on-chain rail.",
                        )
                    });
                    v.push(match check_channel_lifetime(exp, self.mds_timeout_height) {
                        Ok(()) => Check::pass(
                            "Channel lifetime",
                            format!("Expires at height {exp}, past the swap's lock."),
                        ),
                        Err(e) => Check::fail("Channel lifetime", e.to_string(), "Open a channel with a longer lifetime."),
                    });
                }
                _ => v.push(Check::fail(
                    "Channel to this peer",
                    "No open channel toward this counterparty.",
                    "Open one on the Channels tab, or switch to the on-chain rail — slower, but it needs no channel.",
                )),
            }
        }

        v
    }

    pub fn ready(&self) -> bool {
        self.evaluate().iter().all(|c| c.ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;
    const TIP: u64 = 250_000;

    #[test]
    fn the_revealing_leg_always_expires_first() {
        let t = plan_timings(NOW, TIP, DEFAULT_ETH_REFUND_SECS).unwrap();
        assert!(t.eth_deadline < t.mds_deadline_est);
        assert!(t.margin_secs >= SETTLE_MARGIN_SECS);
    }

    #[test]
    fn height_conversion_is_pessimistic_in_the_safe_direction() {
        // Buying time must round up...
        let blocks = secs_to_blocks_generous(3_600);
        assert!(blocks * BLOCK_SECS >= 3_600);
        // ...and spending it must round down, so the same span never looks
        // longer than it might really be.
        assert!(blocks_to_secs_pessimistic(blocks) < blocks * BLOCK_SECS);
        // Round-tripping must not lose the guarantee.
        assert!(blocks_to_secs_pessimistic(secs_to_blocks_generous(3_600)) >= 3_600);
    }

    #[test]
    fn inverted_or_tight_orderings_are_refused() {
        assert!(check_ordering(5_000, 1_000).is_err()); // inverted
        assert!(check_ordering(1_000, 1_000 + SETTLE_MARGIN_SECS - 1).is_err()); // too tight
        assert!(check_ordering(1_000, 1_000 + SETTLE_MARGIN_SECS).is_ok()); // exactly enough
    }

    #[test]
    fn contract_bounds_are_enforced_before_anything_is_signed() {
        assert!(plan_timings(NOW, TIP, 60).is_err()); // under the 10-minute floor
        assert!(plan_timings(NOW, TIP, 8 * 86_400).is_err()); // over the 7-day ceiling
        assert!(plan_timings(NOW, TIP, 600).is_ok());
    }

    #[test]
    fn a_channel_must_outlive_the_lock_it_carries() {
        let t = plan_timings(NOW, TIP, DEFAULT_ETH_REFUND_SECS).unwrap();
        // Expiring before the HTLC: refused.
        assert!(check_channel_lifetime(t.mds_timeout_height - 1, t.mds_timeout_height).is_err());
        // Expiring just after, but inside the cushion: still refused.
        assert!(check_channel_lifetime(t.mds_timeout_height + 1, t.mds_timeout_height).is_err());
        // Comfortably past: fine.
        assert!(check_channel_lifetime(t.mds_timeout_height + 10_000, t.mds_timeout_height).is_ok());
    }

    fn prereqs(rail: Rail, side: Side) -> Prereqs {
        Prereqs {
            side,
            rail,
            synced: true,
            has_evm_key: true,
            eth_balance_wei: Some(10_000_000_000_000_000),
            wei_needed: 1_000_000_000_000_000,
            mds_spendable: 100_000,
            mds_needed: 4_096,
            channel_capacity: Some(50_000),
            channel_expiry: Some(TIP + 100_000),
            tip_height: TIP,
            mds_timeout_height: TIP + 120,
        }
    }

    #[test]
    fn every_unmet_prerequisite_carries_a_fix() {
        let mut p = prereqs(Rail::Submarine, Side::SellMds);
        p.synced = false;
        p.eth_balance_wei = Some(0);
        p.channel_capacity = None;
        p.channel_expiry = None;
        let checks = p.evaluate();
        assert!(!p.ready());
        for c in checks.iter().filter(|c| !c.ok) {
            assert!(c.fix.is_some(), "failing check '{}' offers no way forward", c.label);
        }
    }

    #[test]
    fn the_on_chain_rail_needs_no_channel() {
        let mut p = prereqs(Rail::OnChain, Side::SellMds);
        p.channel_capacity = None;
        p.channel_expiry = None;
        assert!(p.ready(), "on-chain swaps must not require channel capacity");

        // The same situation blocks the submarine rail.
        let mut q = prereqs(Rail::Submarine, Side::SellMds);
        q.channel_capacity = None;
        q.channel_expiry = None;
        assert!(!q.ready());
    }

    #[test]
    fn buying_does_not_require_mds_on_hand() {
        let mut p = prereqs(Rail::OnChain, Side::BuyMds);
        p.mds_spendable = 0;
        assert!(p.ready());
    }
}
