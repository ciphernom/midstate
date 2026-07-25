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

// ── Unit economics ──────────────────────────────────────────────────────
//
// An order is advertised as power-of-two units, and *each unit is claimed by
// its own separate transaction*. That transaction pays its own fee, so a unit
// is only worth trading if it is comfortably larger than that fee. Units below
// that threshold are not merely uneconomic — they are unclaimable, and a taker
// who escrows ETH against one loses it outright: by the time the MDS leg is
// swept the maker has already claimed the ETH, which is how the preimage
// became public in the first place. There is no refund branch after that
// point, and there must not be.
//
// So the threshold is enforced in two independent places, and deliberately so:
// the maker refuses to *publish* such a unit, and the taker refuses to *take*
// one. The second check is the load-bearing one, because orders arrive from
// peers whose software we do not control.

/// Smallest unit that may be published or taken.
///
/// Must be a power of two. That is what makes the maker-side check a single
/// remainder test: if `amount % MIN_SWAP_UNIT == 0` then every bit set in
/// `amount` is at or above `MIN_SWAP_UNIT`, so no sub-threshold unit can be
/// produced by the decomposition.
///
/// 1024 leaves roughly 97% of the unit's value after the sweep fee. Chosen to
/// sit alongside `channels::MIN_CAPACITY` rather than to sit just above the
/// fee: a unit that is *technically* claimable but nets a handful of units is
/// a bad trade, not a working one.
pub const MIN_SWAP_UNIT: u64 = 1_024;

// These two MUST match the constants in `midstate::mempool`. They are repeated
// rather than imported for the same reason `coinjoin::recommended_fee_for_mix`
// repeats them: walletd builds transactions the mempool must accept, so the
// duplication is a deliberate, commented coupling.
const MIN_FEE_PER_KB: u64 = 10;
const FEE_RATE_SCALE: u128 = 1_024;

/// Fixed per-transaction overhead.
const SWEEP_BASE_BYTES: u64 = 256;
/// Input envelope: predicate tag, value, salt, coin reference. The covenant
/// bytecode is counted separately because it varies per script.
const SWEEP_INPUT_BYTES: u64 = 128;
/// Claim witness: 32-byte preimage, branch selector, and framing. A covenant
/// spend carries no WOTS signature, which is why this is so much smaller than
/// the per-input figure used for ordinary spends.
const SWEEP_WITNESS_BYTES: u64 = 96;
/// Output envelope: address, value, salt, framing.
const SWEEP_PER_OUTPUT_BYTES: u64 = 160;
/// Slack above the mempool floor so a sweep is not sitting exactly on the
/// admission boundary, where a slightly larger encoding than estimated would
/// tip it into rejection.
const SWEEP_FEE_MARGIN: u64 = 20;

/// Estimated serialised size of a single-input claim reveal.
pub fn sweep_reveal_size(script_bytes: usize, n_outputs: usize) -> u64 {
    SWEEP_BASE_BYTES
        + SWEEP_INPUT_BYTES
        + SWEEP_WITNESS_BYTES
        + script_bytes as u64
        + (n_outputs as u64) * SWEEP_PER_OUTPUT_BYTES
}

/// Fee for a claim reveal of the given shape, using the mempool's own
/// fee-per-byte rule rather than a flat guess.
pub fn sweep_fee(script_bytes: usize, n_outputs: usize) -> u64 {
    let bytes = sweep_reveal_size(script_bytes, n_outputs) as u128;
    let required = (bytes * MIN_FEE_PER_KB as u128).div_ceil(FEE_RATE_SCALE) as u64;
    required + SWEEP_FEE_MARGIN
}

/// Resolve the fee and the resulting output denominations together.
///
/// The fee depends on the output count, which depends on the decomposition of
/// `value - fee`, which depends on the fee. Same fixed-point iteration as
/// `midstate::wallet::defrag::resolve_fee`, and it terminates for the same
/// reason: each round either settles or strictly increases `n_out`, which is
/// bounded by the 64 bits of a `u64`.
///
/// # Determinism
///
/// This is a **pure function of values persisted in the swap record**, and it
/// must stay that way. The claim is commit-then-reveal: phase one publishes a
/// commitment to this exact transaction, and phase two must rebuild it
/// byte-identically or the reveal is rejected *after* the commit has been paid
/// for. Never let this fee depend on live mempool conditions, wall-clock time,
/// or anything else that can change between the two phases.
pub fn resolve_sweep_fee(script_bytes: usize, value: u64) -> Option<(u64, Vec<u64>)> {
    let mut n_out = 1usize;
    loop {
        let fee = sweep_fee(script_bytes, n_out);
        if fee >= value {
            return None;
        }
        let denoms = midstate::core::decompose_value(value - fee);
        if denoms.len() <= n_out {
            return Some((fee, denoms));
        }
        n_out = denoms.len();
    }
}

/// Whether a unit of this size is worth taking at all.
///
/// Checked against the constant rather than against the resolved fee, so that
/// maker and taker agree on the answer without having to agree on a script
/// size they may encode differently.
pub fn unit_is_tradeable(value: u64) -> bool {
    value >= MIN_SWAP_UNIT
}

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

    // ── Unit economics ──────────────────────────────────────────────────

    #[test]
    fn min_swap_unit_is_a_power_of_two() {
        // The maker-side check is a single `% MIN_SWAP_UNIT` test, which is
        // only equivalent to checking every denomination if this holds.
        assert!(MIN_SWAP_UNIT.is_power_of_two());
    }

    #[test]
    fn an_order_aligned_to_the_minimum_produces_no_unclaimable_units() {
        // The property the whole fix rests on: alignment to MIN_SWAP_UNIT is
        // exactly the condition under which the decomposition is safe.
        for k in 1..=64u64 {
            let amount = k * MIN_SWAP_UNIT;
            for denom in midstate::core::decompose_value(amount) {
                assert!(
                    unit_is_tradeable(denom),
                    "{amount} produced an unclaimable unit of {denom}"
                );
            }
        }
    }

    #[test]
    fn unaligned_amounts_are_exactly_the_ones_that_strand_value() {
        // The converse — otherwise the maker-side guard would be rejecting
        // orders that were actually fine.
        for amount in [MIN_SWAP_UNIT + 1, 5_000, 4_097, 60, 63] {
            let strands = midstate::core::decompose_value(amount)
                .into_iter()
                .any(|d| !unit_is_tradeable(d));
            assert_eq!(
                strands,
                amount % MIN_SWAP_UNIT != 0,
                "alignment must predict stranding for {amount}"
            );
        }
    }

    #[test]
    fn every_tradeable_unit_can_actually_pay_its_own_fee() {
        // The guard promises a claimable unit. If a value at the threshold
        // could not resolve a fee, the promise would be empty.
        for k in 1..=64u64 {
            let value = k * MIN_SWAP_UNIT;
            let (fee, denoms) =
                resolve_sweep_fee(512, value).expect("a tradeable unit must resolve a fee");
            assert!(fee < value);
            assert_eq!(
                denoms.iter().sum::<u64>(),
                value - fee,
                "outputs plus fee must account for the whole unit"
            );
            // And it has to be worth doing, not merely possible.
            assert!(
                value - fee > value * 9 / 10,
                "a {value}-unit claim kept only {} after a {fee} fee",
                value - fee
            );
        }
    }

    #[test]
    fn the_resolved_fee_covers_the_outputs_it_implies() {
        // Fixed-point closure: the fee charged must be the fee for the shape
        // actually produced, or the transaction underpays and is rejected.
        for value in [MIN_SWAP_UNIT, 4_096, 65_536, 1_000_448] {
            let (fee, denoms) = resolve_sweep_fee(512, value).unwrap();
            assert!(
                fee >= sweep_fee(512, denoms.len()),
                "fee {fee} does not cover {} outputs",
                denoms.len()
            );
        }
    }

    #[test]
    fn the_fee_is_deterministic_across_commit_and_reveal() {
        // The commitment binds this exact transaction. If the same inputs ever
        // produced two different fees, the reveal would be rejected after the
        // commit had already been paid for.
        let a = resolve_sweep_fee(377, 8_192).unwrap();
        let b = resolve_sweep_fee(377, 8_192).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn sub_minimum_units_are_refused_rather_than_priced() {
        for value in [1, 2, 8, 32, 60, 64, MIN_SWAP_UNIT - 1] {
            assert!(!unit_is_tradeable(value), "{value} must not be tradeable");
        }
    }

    #[test]
    fn the_fee_clears_the_mempool_floor_it_was_derived_from() {
        for n_out in [1usize, 4, 16, 32] {
            let bytes = sweep_reveal_size(512, n_out);
            let fee = sweep_fee(512, n_out);
            // Mirrors the admission check in mempool.rs.
            assert!(
                (fee as u128) * FEE_RATE_SCALE >= (MIN_FEE_PER_KB as u128) * (bytes as u128),
                "a {n_out}-output sweep would be rejected by the mempool"
            );
        }
    }
}
