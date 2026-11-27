// src/wallet/plan.rs
//
// Privacy-aware unilateral CoinJoin (self-PayJoin) transaction planner.
//
// Plans sends that know their own outputs (CLI `wallet send`, walletd, the
// web wallet) by jointly optimizing input selection, consolidation, and
// power-of-two change shaping against the fee fixed-point. `Wallet::select_coins`
// keeps its greedy snowball for callers that build their outputs themselves.
//
// Tier 2 under `docs/CODING.md`: participates in one-time signature discipline
// (WOTS co-spend atomicity) and transaction value conservation.

use super::defrag::SpendBundle;
use super::Wallet;
use crate::core::{decompose_value, MAX_TX_INPUTS, MAX_TX_OUTPUTS};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};

/// Standard Midstate size model (matches `crates/walletd/src/sendplan.rs` and `main.rs`).
pub const BYTES_BASE: u64 = 100;
pub const BYTES_PER_INPUT: u64 = 1636;
pub const BYTES_PER_OUTPUT: u64 = 100;

/// The fee every wallet pays for a transaction of an estimated `bytes`: the
/// mempool floor of 10 units per KiB, plus 10 units of padding. One formula
/// for the CLI, the desktop and the web wallet, so that a fee does not say
/// which wallet made the transaction.
pub fn fee_for_bytes(bytes: u64) -> u64 {
    (bytes * 10) / 1024 + 10
}

/// [`fee_for_bytes`] for `inputs` inputs and `outputs` outputs under the
/// standard size model.
pub fn standard_fee(inputs: usize, outputs: usize) -> u64 {
    fee_for_bytes(estimate_tx_bytes(inputs, outputs))
}

/// The fee for a `Consolidate` of `inputs` coins at one address, the same in
/// every wallet: one MSS witness (~3,000 bytes) and ~125 bytes per input
/// reveal (bincode's length prefixes and tags included), at 10 units per KiB
/// plus 20 units of padding.
pub fn consolidate_fee(inputs: usize) -> u64 {
    let bytes = 600 + 3000 + 100 + inputs as u64 * 125;
    (bytes * 10) / 1024 + 20
}

/// Estimated serialized transaction size in bytes.
pub fn estimate_tx_bytes(inputs: usize, outputs: usize) -> u64 {
    BYTES_BASE + inputs as u64 * BYTES_PER_INPUT + outputs as u64 * BYTES_PER_OUTPUT
}

/// Tuning parameters and hard economic/fragmentation guardrails for the planner.
#[derive(Clone, Copy, Debug)]
pub struct PlanConfig {
    /// Maximum relative fee overhead above the snowball baseline, in basis points
    /// (1500 = +15%).
    pub max_fee_overhead_bps: u64,
    /// Minimum absolute fee overhead allowance above baseline (prevents tiny
    /// base fees from blocking cheap 100-byte output splits).
    pub max_fee_overhead_abs: u64,
    /// Wallet UTXO count below which a single +1 change split is permitted for
    /// privacy. At or above this ceiling, candidates must not worsen net UTXO
    /// delta relative to the snowball baseline.
    pub utxo_comfort_ceiling: usize,
    /// Maximum number of power-of-two change splits (2d -> d + d) per tx.
    pub max_change_splits: usize,
    /// True when each payment denomination will be sent to its own fresh
    /// recipient address (via a multi-address payment request). When false,
    /// an on-chain analyst clusters all payment outputs into one address group.
    pub distinct_payment_addresses: bool,
    /// Soft score penalty for each input beyond the minimal covering prefix.
    /// Prevents pulling extra WOTS inputs (~1,636 B + CIOH linkage) unless
    /// they buy real address-aware ambiguity or needed consolidation.
    pub extra_input_penalty: i64,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_fee_overhead_bps: 1500,
            max_fee_overhead_abs: 25,
            utxo_comfort_ceiling: 32,
            max_change_splits: 2,
            distinct_payment_addresses: false,
            extra_input_penalty: 140,
        }
    }
}

/// What a transaction pays out besides its change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Payment {
    /// Values of the outputs paid to recipient addresses, each a power of two,
    /// in the order the caller builds them. Change splits aim at these values
    /// and ambiguity is scored against them.
    pub denoms: Vec<u64>,
    /// Value leaving through outputs an observer can tell from payments (data
    /// burns): part of the amount, never matched by change.
    pub other_value: u64,
    /// How many such outputs there are, value-bearing or not (burns, state
    /// threads): counted for the fee and the output limit.
    pub other_outputs: usize,
}

impl Payment {
    /// One payment of `amount`, in its power-of-two denominations.
    pub fn of(amount: u64) -> Self {
        Self { denoms: decompose_value(amount), other_value: 0, other_outputs: 0 }
    }

    /// Everything paid out (payments and burns), without change or fee.
    pub fn amount(&self) -> u64 {
        self.denoms.iter().fold(self.other_value, |a, &d| a.saturating_add(d))
    }

    /// Outputs besides change.
    pub fn outputs(&self) -> usize {
        self.denoms.len() + self.other_outputs
    }
}

/// A pure, unsigned transaction plan produced before any WOTS key allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxPlan {
    pub input_coin_ids: Vec<[u8; 32]>,
    pub payment_denoms: Vec<u64>,
    pub change_denoms: Vec<u64>,
    pub in_sum: u64,
    pub fee: u64,
    /// Everything paid out: `Σ payment_denoms` plus any burned value.
    pub amount: u64,
    /// Non-change outputs besides `payment_denoms` (`Payment::other_outputs`).
    pub other_outputs: usize,
}

impl TxPlan {
    /// Net change in wallet UTXO count: `|change_outputs| - |inputs|`.
    /// Negative values indicate consolidation.
    pub fn net_utxo_delta(&self) -> isize {
        self.change_denoms.len() as isize - self.input_coin_ids.len() as isize
    }

    /// Total output count (payment + other + change).
    pub fn output_count(&self) -> usize {
        self.payment_denoms.len() + self.other_outputs + self.change_denoms.len()
    }
}

/// Plans a transaction over pre-grouped `SpendBundle`s using a unilateral
/// CoinJoin (self-PayJoin) optimizer with the greedy snowball as baseline and
/// fallback.
///
/// # Reasoning
/// The legacy `select_coins` snowball only saw `needed = amount + fee_guess`.
/// Because it was blind to the payment's binary decomposition and output
/// construction, it could not create equal-denomination payment/change pairs
/// and forced every wallet frontend (CLI, desktop, web) to duplicate the fee
/// convergence loop.
///
/// On Midstate, every output is a power of two and outputs (100 B) are ~16x
/// cheaper than WOTS inputs (1636 B). Jointly optimizing input selection,
/// consolidation, and controlled power-of-two change splits (`2d -> d + d`)
/// maximizes address-group-aware payment/change indistinguishability while
/// preserving the snowball's UTXO fragmentation bound.
///
/// # Formal Specification
///
/// ```text
/// Pre:
///   - amount > 0
///   - bundles is a valid partition of spendable coins (WOTS siblings atomic,
///     each bundle has 1 <= #coin_ids <= MAX_TX_INPUTS)
///   - fee_fn is monotonically non-decreasing in (inputs, outputs)
///
/// Post (pure — no wallet or HD state mutated):
///   result = Ok(plan) ⇒
///     plan.input_coin_ids = ⋃ { b.coin_ids | b ∈ S } for some S ⊆ bundles
///     1 <= #plan.input_coin_ids <= MAX_TX_INPUTS
///     #(plan.payment_denoms ⌢ plan.change_denoms) <= MAX_TX_OUTPUTS
///     ∀ d ∈ plan.payment_denoms ⌢ plan.change_denoms: d.is_power_of_two()
///     Σ plan.payment_denoms = amount
///     plan.fee ≥ fee_fn(#plan.input_coin_ids, plan.output_count)
///     plan.fee = fee_fn(#plan.input_coin_ids, m) for the least m ≥ plan.output_count
///       at which the change left fits in m outputs (splits: m = plan.output_count)
///     plan.in_sum = amount + Σ plan.change_denoms + plan.fee
///
///   result = Err(_) ⇒ neither the snowball nor the minimal covering prefix
///     can pay its own fee under MAX_TX_INPUTS / fee_fn
/// ```
///
/// ```zed
///     PlanTransaction
///     ---------------
///     bundles? : seq SpendBundle
///     amount? : ℕ₆₄
///     fee_fn? : (ℕ × ℕ) → ℕ₆₄
///     cfg? : PlanConfig
///     plan! : TxPlan
///
///     pre  amount? > 0
///     post ∃ S ⊆ ran bundles? •
///            plan!.input_coin_ids = ⋃ { b.coin_ids | b ∈ S }
///          ∧ 1 ≤ #plan!.input_coin_ids ≤ MAX_TX_INPUTS
///          ∧ #(plan!.payment_denoms ⌢ plan!.change_denoms) ≤ MAX_TX_OUTPUTS
///          ∧ (∀ d ∈ plan!.payment_denoms ⌢ plan!.change_denoms • is_power_of_two(d))
///          ∧ (∑ d ∈ plan!.payment_denoms • d) = amount?
///          ∧ plan!.fee ≥ fee_fn?(#plan!.input_coin_ids, plan!.output_count)
///          ∧ plan!.in_sum = amount? + (∑ d ∈ plan!.change_denoms • d) + plan!.fee
/// ```
///
/// # Safety / Invariants
/// - **WOTS Co-Spend Atomicity**: Inputs are always unions of whole `SpendBundle`s.
/// - **No Key Mutation**: Pure function; callers allocate fresh WOTS change keys
///   only after a plan is finalized (ideally in one `allocate_next_wots_seeds` call).
/// - **Snowball Fallback**: Candidate 0 is the snowball baseline. When the
///   snowball cannot pay its own fee, the minimal covering prefix stands in
///   for it, so owning an extra coin never makes an affordable send fail.
pub fn plan_transaction_from_bundles<F>(
    bundles: &[SpendBundle],
    amount: u64,
    fee_fn: F,
    cfg: &PlanConfig,
) -> Result<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    if amount == 0 {
        bail!("amount must be greater than zero");
    }
    plan_payment_from_bundles(bundles, &Payment::of(amount), fee_fn, cfg)
}

/// [`plan_transaction_from_bundles`] for a send whose outputs are not one
/// payment of `amount`: several recipients, burns, state threads. The fee is
/// counted over the real outputs and change is shaped against the real
/// payment values.
pub fn plan_payment_from_bundles<F>(
    bundles: &[SpendBundle],
    payment: &Payment,
    fee_fn: F,
    cfg: &PlanConfig,
) -> Result<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    let amount = payment.amount();
    if amount == 0 {
        bail!("amount must be greater than zero");
    }
    if payment.denoms.iter().any(|d| !d.is_power_of_two()) {
        bail!("payment outputs must be powers of two");
    }
    if payment.outputs() > MAX_TX_OUTPUTS {
        bail!("payment requires more than MAX_TX_OUTPUTS");
    }
    let payment_denoms = &payment.denoms;

    // Candidate 0: the snowball baseline. Its merges can cost more fee than
    // they bring; when it cannot pay its own way, the minimal covering prefix
    // takes its place rather than failing a send other candidates can make.
    let prefix = select_prefix_indices(bundles, payment, &fee_fn).ok();
    let baseline = match snowball_baseline(bundles, payment, &fee_fn) {
        Ok(plan) => plan,
        Err(e) => match prefix
            .as_deref()
            .and_then(|idx| resolve_candidate(bundles, idx, payment, &fee_fn))
        {
            Some(plan) => plan,
            None => return Err(e),
        },
    };
    let wallet_utxo_count: usize = bundles.iter().map(|b| b.coin_ids.len()).sum();

    let max_fee = baseline.fee.saturating_add(
        ((baseline.fee.saturating_mul(cfg.max_fee_overhead_bps)) / 10_000)
            .max(cfg.max_fee_overhead_abs),
    );

    // Collect candidate input index sets (indices into `bundles`).
    let mut input_sets: Vec<Vec<usize>> = Vec::with_capacity(20);
    let mut min_inputs = baseline.input_coin_ids.len();

    // 1. Minimal density-greedy prefix (without snowball cascade).
    if let Some(prefix_idx) = prefix {
        let prefix_in: usize = prefix_idx.iter().map(|&i| bundles[i].coin_ids.len()).sum();
        min_inputs = min_inputs.min(prefix_in);
        input_sets.push(prefix_idx.clone());

        // 2. Ambiguity-guided single-bundle additions to the prefix.
        add_guided_expansions(bundles, &prefix_idx, payment_denoms, &mut input_sets, 8);
    }

    // 3. Baseline indices + ambiguity-guided single-bundle additions to it.
    let base_indices = indices_for_plan(bundles, &baseline);
    if !base_indices.is_empty() {
        add_guided_expansions(bundles, &base_indices, payment_denoms, &mut input_sets, 6);
    }

    // Evaluate all candidates (plus controlled change splits) and pick the highest score.
    let mut best_plan = baseline.clone();
    let mut best_score = score_plan(&baseline, &baseline, min_inputs, wallet_utxo_count, cfg);

    for idxs in input_sets {
        let Some(base_cand) = resolve_candidate(bundles, &idxs, payment, &fee_fn) else {
            continue;
        };

        // Test unsplit candidate + up to `cfg.max_change_splits` controlled splits.
        let variants = generate_split_variants(base_cand, &fee_fn, cfg.max_change_splits);
        for cand in variants {
            if cand.fee > max_fee {
                continue;
            }
            // Fragmentation guardrail: when wallet is above comfort ceiling,
            // never allow worse net UTXO delta than snowball.
            if wallet_utxo_count >= cfg.utxo_comfort_ceiling
                && cand.net_utxo_delta() > baseline.net_utxo_delta()
            {
                continue;
            }
            let s = score_plan(&cand, &baseline, min_inputs, wallet_utxo_count, cfg);
            if s > best_score {
                best_score = s;
                best_plan = cand;
            }
        }
    }

    Ok(best_plan)
}

/// The density-greedy + snowball selection of `Wallet::select_coins`,
/// converged against `fee_fn`. Change is consolidated as before, dust
/// included, but only by merges after which the selection still pays its own
/// fee: a merge the send cannot afford is skipped, never made fatal.
pub fn plan_snowball_baseline<F>(
    bundles: &[SpendBundle],
    amount: u64,
    fee_fn: F,
) -> Result<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    if amount == 0 {
        bail!("amount must be greater than zero");
    }
    snowball_baseline(bundles, &Payment::of(amount), &fee_fn)
}

fn snowball_baseline<F>(bundles: &[SpendBundle], payment: &Payment, fee_fn: &F) -> Result<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    let amount = payment.amount();
    let pay_outs = payment.outputs();
    let mut target_fee = fee_fn(1, pay_outs);

    for _ in 0..32 {
        let needed = amount.saturating_add(target_fee);
        let mut chosen_idx = Vec::new();
        let mut n_in = 0usize;
        let mut total = 0u64;

        for (i, b) in bundles.iter().enumerate() {
            if total >= needed {
                break;
            }
            if n_in + b.coin_ids.len() > MAX_TX_INPUTS {
                continue;
            }
            chosen_idx.push(i);
            n_in += b.coin_ids.len();
            total = total.saturating_add(b.total_value);
        }

        if total < needed {
            return Err(shortfall(bundles, needed));
        }

        // Greedy snowball cascade on unselected size-1 bundles.
        let chosen_set: HashSet<usize> = chosen_idx.iter().copied().collect();
        let mut unselected_by_denom: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, b) in bundles.iter().enumerate() {
            if b.coin_ids.len() == 1 && !chosen_set.contains(&i) {
                unselected_by_denom.entry(b.total_value).or_default().push(i);
            }
        }

        let mut added = true;
        while added && n_in < MAX_TX_INPUTS {
            added = false;
            let change = total - needed;
            for denom in decompose_value(change) {
                let Some(list) = unselected_by_denom.get_mut(&denom) else { continue };
                let Some(&idx) = list.last() else { continue };
                let value = bundles[idx].total_value;
                // Merging dust costs more fee than it brings; worth it to keep
                // the wallet tidy, but not at the price of the send itself.
                if settle_fee(n_in + 1, total.saturating_add(value), amount, pay_outs, fee_fn).is_err() {
                    continue;
                }
                list.pop();
                chosen_idx.push(idx);
                n_in += 1;
                total = total.saturating_add(value);
                added = true;
                break;
            }
        }

        match settle_fee(n_in, total, amount, pay_outs, fee_fn) {
            Ok((fee, change_denoms)) => {
                let input_coin_ids = chosen_idx
                    .iter()
                    .flat_map(|&i| bundles[i].coin_ids.iter().copied())
                    .collect();
                return Ok(TxPlan {
                    input_coin_ids,
                    payment_denoms: payment.denoms.clone(),
                    change_denoms,
                    in_sum: total,
                    fee,
                    amount,
                    other_outputs: payment.other_outputs,
                });
            }
            // This selection cannot pay its own fee: select again for at least that.
            Err(fee) => target_fee = fee.max(target_fee.saturating_add(1)),
        }
    }

    bail!("fee convergence failed in snowball baseline")
}

/// The cheapest fee the model allows `n_in` inputs worth `in_sum` paying
/// `amount` over `pay_outs` non-change outputs: `fee_fn(n_in, pay_outs + k)`
/// for the least `k` whose change fits in `k` outputs. With `fee_fn`
/// non-decreasing in outputs this is start-independent, so every candidate
/// is priced the same way. `Err(fee)`: `in_sum` cannot cover `amount + fee`.
fn settle_fee<F>(
    n_in: usize,
    in_sum: u64,
    amount: u64,
    pay_outs: usize,
    fee_fn: &F,
) -> std::result::Result<(u64, Vec<u64>), u64>
where
    F: Fn(usize, usize) -> u64,
{
    let max_k = MAX_TX_OUTPUTS.saturating_sub(pay_outs).min(64);
    for k in 0..=max_k {
        let fee = fee_fn(n_in, pay_outs + k);
        if in_sum < amount.saturating_add(fee) {
            return Err(fee);
        }
        let change = in_sum - amount - fee;
        if change.count_ones() as usize <= k {
            return Ok((fee, decompose_value(change)));
        }
    }
    // Change too fragmented for the output limit at any affordable fee.
    Err(fee_fn(n_in, pay_outs + max_k).saturating_add(1))
}

/// Why no selection covers `needed`: too little, or too many coins.
fn shortfall(bundles: &[SpendBundle], needed: u64) -> anyhow::Error {
    let spendable = bundles.iter().fold(0u64, |a, b| a.saturating_add(b.total_value));
    if spendable >= needed {
        anyhow::anyhow!(
            "insufficient consolidated funds: {} is spendable, but covering {} takes more \
             than {} inputs. Consolidate small coins first.",
            spendable,
            needed,
            MAX_TX_INPUTS
        )
    } else {
        anyhow::anyhow!("insufficient funds: have {} spendable, need {}", spendable, needed)
    }
}

impl Wallet {
    /// Plan a transaction using the unilateral CoinJoin (self-PayJoin) planner
    /// with default settings (`distinct_payment_addresses = false`).
    /// Pure (`&self`): does not allocate WOTS keys or mutate wallet state.
    pub fn plan_transaction<F>(
        &self,
        live_coins: &[[u8; 32]],
        amount: u64,
        fee_fn: F,
    ) -> Result<TxPlan>
    where
        F: Fn(usize, usize) -> u64,
    {
        self.plan_transaction_with_config(live_coins, amount, fee_fn, &PlanConfig::default())
    }

    /// Plan a transaction with an explicit [`PlanConfig`] (e.g. setting
    /// `distinct_payment_addresses: true` when paying a multi-address request).
    pub fn plan_transaction_with_config<F>(
        &self,
        live_coins: &[[u8; 32]],
        amount: u64,
        fee_fn: F,
        cfg: &PlanConfig,
    ) -> Result<TxPlan>
    where
        F: Fn(usize, usize) -> u64,
    {
        if amount == 0 {
            bail!("amount must be greater than zero");
        }
        self.plan_payment_with_config(live_coins, &Payment::of(amount), fee_fn, cfg)
    }

    /// Plan a send with several recipients, burns or state threads
    /// ([`plan_payment_from_bundles`]).
    pub fn plan_payment_with_config<F>(
        &self,
        live_coins: &[[u8; 32]],
        payment: &Payment,
        fee_fn: F,
        cfg: &PlanConfig,
    ) -> Result<TxPlan>
    where
        F: Fn(usize, usize) -> u64,
    {
        let live_set: HashSet<[u8; 32]> = live_coins.iter().copied().collect();
        let bundles = self.spendable_bundles(&live_set, true);
        plan_payment_from_bundles(&bundles, payment, fee_fn, cfg)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

fn select_prefix_indices<F>(bundles: &[SpendBundle], payment: &Payment, fee_fn: &F) -> Result<Vec<usize>>
where
    F: Fn(usize, usize) -> u64,
{
    let amount = payment.amount();
    let pay_len = payment.outputs();
    let mut target_fee = fee_fn(1, pay_len + 1);
    for _ in 0..16 {
        let needed = amount.saturating_add(target_fee);
        let mut chosen = Vec::new();
        let mut n_in = 0usize;
        let mut total = 0u64;
        for (i, b) in bundles.iter().enumerate() {
            if total >= needed {
                break;
            }
            if n_in + b.coin_ids.len() > MAX_TX_INPUTS {
                continue;
            }
            chosen.push(i);
            n_in += b.coin_ids.len();
            total = total.saturating_add(b.total_value);
        }
        if total < needed {
            return Err(shortfall(bundles, needed));
        }
        match settle_fee(n_in, total, amount, pay_len, fee_fn) {
            Ok(_) => return Ok(chosen),
            Err(fee) => target_fee = fee.max(target_fee.saturating_add(1)),
        }
    }
    bail!("prefix fee convergence failed")
}

fn indices_for_plan(bundles: &[SpendBundle], plan: &TxPlan) -> Vec<usize> {
    let used: HashSet<[u8; 32]> = plan.input_coin_ids.iter().copied().collect();
    bundles
        .iter()
        .enumerate()
        .filter(|(_, b)| b.coin_ids.iter().all(|id| used.contains(id)))
        .map(|(i, _)| i)
        .collect()
}

fn add_guided_expansions(
    bundles: &[SpendBundle],
    base: &[usize],
    payment_denoms: &[u64],
    out: &mut Vec<Vec<usize>>,
    limit: usize,
) {
    let base_set: HashSet<usize> = base.iter().copied().collect();
    let n_in: usize = base.iter().map(|&i| bundles[i].coin_ids.len()).sum();
    if n_in >= MAX_TX_INPUTS {
        return;
    }

    let pay_set: HashSet<u64> = payment_denoms.iter().copied().collect();
    let mut added = 0usize;

    let mut scored: Vec<(u8, u64, usize)> = Vec::new();
    for (i, b) in bundles.iter().enumerate() {
        if base_set.contains(&i) || b.coin_ids.len() != 1 {
            continue;
        }
        let prio = if pay_set.contains(&b.total_value) {
            0
        } else if pay_set.contains(&(b.total_value.saturating_mul(2))) {
            1
        } else {
            2
        };
        scored.push((prio, b.total_value, i));
    }
    scored.sort_by_key(|&(prio, val, _)| (prio, val));

    for (_, _, idx) in scored {
        if added >= limit {
            break;
        }
        let mut next = base.to_vec();
        next.push(idx);
        out.push(next);
        added += 1;
    }
}

fn resolve_candidate<F>(
    bundles: &[SpendBundle],
    idxs: &[usize],
    payment: &Payment,
    fee_fn: &F,
) -> Option<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    let n_in: usize = idxs.iter().map(|&i| bundles[i].coin_ids.len()).sum();
    if n_in == 0 || n_in > MAX_TX_INPUTS {
        return None;
    }
    let in_sum: u64 = idxs.iter().fold(0u64, |a, &i| a.saturating_add(bundles[i].total_value));
    let amount = payment.amount();
    let (fee, change_denoms) = settle_fee(n_in, in_sum, amount, payment.outputs(), fee_fn).ok()?;
    let input_coin_ids = idxs
        .iter()
        .flat_map(|&i| bundles[i].coin_ids.iter().copied())
        .collect();
    Some(TxPlan {
        input_coin_ids,
        payment_denoms: payment.denoms.clone(),
        change_denoms,
        in_sum,
        fee,
        amount,
        other_outputs: payment.other_outputs,
    })
}

/// Generate up to `max_splits` controlled power-of-two change splits (`2d -> d + d`)
/// targeting denominations present in `payment_denoms`.
fn generate_split_variants<F>(base: TxPlan, fee_fn: &F, max_splits: usize) -> Vec<TxPlan>
where
    F: Fn(usize, usize) -> u64,
{
    let mut variants = vec![base.clone()];
    if max_splits == 0 || base.change_denoms.is_empty() {
        return variants;
    }

    let pay_set: HashSet<u64> = base.payment_denoms.iter().copied().collect();
    let mut current = base;

    for _ in 0..max_splits {
        let split_pos = current
            .change_denoms
            .iter()
            .position(|&c| c >= 2 && (c % 2 == 0) && pay_set.contains(&(c / 2)));

        let Some(pos) = split_pos else { break };
        let next_n_out = current.output_count() + 1;
        if next_n_out > MAX_TX_OUTPUTS {
            break;
        }

        let new_fee = fee_fn(current.input_coin_ids.len(), next_n_out);
        if new_fee != current.fee {
            if current.in_sum < current.amount.saturating_add(new_fee) {
                break;
            }
            let rem = current.in_sum - current.amount - new_fee;
            let mut next_change = decompose_value(rem);
            if let Some(p2) = next_change
                .iter()
                .position(|&c| c >= 2 && (c % 2 == 0) && pay_set.contains(&(c / 2)))
            {
                let half = next_change.remove(p2) / 2;
                next_change.push(half);
                next_change.push(half);
                if current.payment_denoms.len() + current.other_outputs + next_change.len()
                    == next_n_out
                {
                    current.change_denoms = next_change;
                    current.fee = new_fee;
                    variants.push(current.clone());
                    continue;
                }
            }
            break;
        }

        let half = current.change_denoms.remove(pos) / 2;
        current.change_denoms.push(half);
        current.change_denoms.push(half);
        variants.push(current.clone());
    }

    variants
}

/// Exact count of distinct subsets of `groups` summing to `target`.
/// Capped at 18 groups to stay sub-microsecond.
pub fn count_payment_interpretations(groups: &[u64], target: u64) -> u64 {
    if groups.len() > 18 {
        return 1;
    }
    let mut count = 0u64;
    let total_masks = 1usize << groups.len();
    for mask in 1..total_masks {
        let mut sum = 0u64;
        for (i, &v) in groups.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                sum = sum.saturating_add(v);
                if sum > target {
                    break;
                }
            }
        }
        if sum == target {
            count += 1;
        }
    }
    count.max(1)
}

/// Address-group-aware interpretation count reflecting what an on-chain
/// analyst actually sees.
///
/// Every change output receives a fresh independent WOTS address, so each
/// change output is its own address group. When `distinct_payment_addresses`
/// is `false`, all payment outputs share one recipient address and collapse
/// into a single atomic group of value `amount`.
pub fn count_address_group_interpretations(
    payment_denoms: &[u64],
    change_denoms: &[u64],
    amount: u64,
    distinct_payment_addresses: bool,
) -> u64 {
    let mut groups = Vec::with_capacity(
        change_denoms.len() + if distinct_payment_addresses { payment_denoms.len() } else { 1 },
    );
    if distinct_payment_addresses {
        groups.extend_from_slice(payment_denoms);
    } else {
        groups.push(amount);
    }
    groups.extend_from_slice(change_denoms);
    count_payment_interpretations(&groups, amount)
}

fn score_plan(
    plan: &TxPlan,
    baseline: &TxPlan,
    min_inputs: usize,
    wallet_utxos: usize,
    cfg: &PlanConfig,
) -> i64 {
    // Burns are visible as burns: only payments to addresses can be confused with change.
    let paid: u64 = plan.payment_denoms.iter().fold(0u64, |a, &d| a.saturating_add(d));
    let interp = count_address_group_interpretations(
        &plan.payment_denoms,
        &plan.change_denoms,
        paid,
        cfg.distinct_payment_addresses,
    );

    let pay_set: HashSet<u64> = plan.payment_denoms.iter().copied().collect();
    let overlap = plan
        .change_denoms
        .iter()
        .filter(|d| pay_set.contains(d))
        .count() as i64;

    // Equal-denomination overlap only produces direct output-level
    // indistinguishability when payment outputs aren't clustered on one address.
    let unclustered_payment = cfg.distinct_payment_addresses || plan.payment_denoms.len() == 1;
    let overlap_score = if unclustered_payment {
        overlap * 120
    } else {
        overlap * 20
    };
    let single_denom_bonus = if unclustered_payment && overlap > 0 {
        350
    } else {
        0
    };

    // Address-aware ambiguity score: strong reward for moving from 1 (deterministic) to >=2.
    let amb_score = if interp >= 2 {
        550 + (interp.min(8) as i64) * 80
    } else {
        0
    } + overlap_score
        + single_denom_bonus;

    // Consolidation score: reward negative net UTXO delta (stronger when fragmented).
    let frag_weight = if wallet_utxos >= cfg.utxo_comfort_ceiling {
        220
    } else {
        90
    };
    let frag_score = -(plan.net_utxo_delta() as i64) * frag_weight;

    // Soft penalty for each extra WOTS input beyond the minimal prefix.
    let extra_inputs = plan.input_coin_ids.len().saturating_sub(min_inputs) as i64;
    let input_penalty = extra_inputs * cfg.extra_input_penalty;

    // Fee penalty relative to snowball baseline.
    let fee_diff = plan.fee as i64 - baseline.fee as i64;
    let fee_penalty = fee_diff * 12;

    amb_score + frag_score - input_penalty - fee_penalty
}

/// Parse one or more comma-separated 72-char checksummed hex addresses.
/// Single address -> `len() == 1` (`distinct_payment_addresses = false`).
/// Multi-address  -> `len() > 1`  (`distinct_payment_addresses = true`).
pub fn parse_destinations(s: &str) -> Result<Vec<[u8; 32]>> {
    let mut addrs = Vec::new();
    let mut seen = HashSet::new();

    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let bytes = hex::decode(part).map_err(|_| anyhow::anyhow!("address is not valid hex"))?;
        if bytes.len() != 36 {
            bail!("address must be 72 hex characters (got {})", part.len());
        }
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&bytes[..32]);
        let expect = crate::core::hash(&addr);
        if expect[..4] != bytes[32..36] {
            bail!("address checksum does not match — check for typos");
        }
        if !seen.insert(addr) {
            bail!("duplicate address in multi-address destination");
        }
        addrs.push(addr);
    }

    if addrs.is_empty() {
        bail!("no destination address provided");
    }
    Ok(addrs)
}

/// Format one or more addresses as a comma-separated checksummed string.
pub fn format_destinations(addrs: &[[u8; 32]]) -> String {
    addrs
        .iter()
        .map(crate::core::encode_address_with_checksum)
        .collect::<Vec<_>>()
        .join(",")
}

impl Wallet {
    /// Generate `count` fresh WOTS receive keys in one atomic save for a
    /// multi-address receive bundle (default recommendation: 4).
    pub fn generate_receive_bundle(
        &mut self,
        count: usize,
        label: Option<String>,
    ) -> Result<Vec<[u8; 32]>> {
        if count == 0 {
            bail!("bundle count must be at least 1");
        }
        self.generate_keys(count, label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bundle(id_byte: u8, value: u64) -> SpendBundle {
        let mut id = [0u8; 32];
        id[0] = id_byte;
        SpendBundle {
            address: id,
            coin_ids: vec![id],
            total_value: value,
        }
    }

    #[test]
    fn controlled_split_creates_two_way_ambiguity() {
        let bundles = vec![make_bundle(1, 4096)];
        let plan = plan_transaction_from_bundles(
            &bundles,
            1024,
            standard_fee,
            &PlanConfig::default(),
        )
        .unwrap();

        let interp = count_address_group_interpretations(
            &plan.payment_denoms,
            &plan.change_denoms,
            plan.amount,
            false,
        );
        assert!(interp >= 2, "expected >=2 address-group interpretations, got {interp}");
        assert_eq!(
            plan.in_sum,
            plan.amount + plan.change_denoms.iter().sum::<u64>() + plan.fee
        );
    }

    fn web_fee(n_in: usize, n_out: usize) -> u64 {
        ((100 + n_in as u64 * 1636 + n_out as u64 * 100) * 100) / 1024 + 50
    }

    fn singles(vals: &[u64]) -> Vec<SpendBundle> {
        let mut b: Vec<SpendBundle> = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut id = [0u8; 32];
                id[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
                SpendBundle { address: id, coin_ids: vec![id], total_value: v }
            })
            .collect();
        b.sort_by(|x, y| y.total_value.cmp(&x.total_value));
        b
    }

    /// Owning one more coin must never make an affordable send fail. The
    /// snowball used to merge dust worth less than its input's fee, fail,
    /// and take every other candidate down with it.
    #[test]
    fn a_dust_coin_never_blocks_an_affordable_send() {
        let cfg = PlanConfig::default();
        let alone = plan_transaction_from_bundles(&singles(&[1024]), 640, web_fee, &cfg).unwrap();
        let with_dust = plan_transaction_from_bundles(&singles(&[1024, 1]), 640, web_fee, &cfg).unwrap();
        assert_eq!(with_dust.input_coin_ids.len(), 1);
        assert_eq!(with_dust.fee, alone.fee);
        assert!(plan_transaction_from_bundles(&singles(&[1024, 64, 8, 2, 1]), 616, web_fee, &cfg).is_ok());
        assert!(plan_transaction_from_bundles(&singles(&[128, 128, 64, 32, 16, 4, 2]), 253, standard_fee, &cfg).is_ok());

        let mut rng: u64 = 0xD057_C01E;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng >> 33
        };
        for (fee_fn, hi) in [(standard_fee as fn(usize, usize) -> u64, 8u64), (web_fee, 11)] {
            for _ in 0..3_000 {
                let vals: Vec<u64> = (0..2 + next() % 6).map(|_| 1u64 << (next() % hi)).collect();
                let total: u64 = vals.iter().sum();
                let amount = 1 + next() % total;
                if plan_transaction_from_bundles(&singles(&vals), amount, fee_fn, &cfg).is_err() {
                    continue;
                }
                for dust in [1u64, 2, 4, 8, 16] {
                    let mut more = vals.clone();
                    more.push(dust);
                    assert!(
                        plan_transaction_from_bundles(&singles(&more), amount, fee_fn, &cfg).is_ok(),
                        "{vals:?} sends {amount}, plus a coin of {dust} does not"
                    );
                }
            }
        }
    }

    /// Value is conserved, inputs are whole bundles, and the fee covers the
    /// plan's own shape, for single payments and for several recipients with
    /// a burn beside them.
    #[test]
    fn plans_conserve_value_and_pay_for_their_shape() {
        let mut rng: u64 = 0x5EED_0F_F1E5;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng >> 33
        };
        for iter in 0..4_000u64 {
            let mut bundles = Vec::new();
            for i in 0..1 + next() % 40 {
                let k = if next() % 7 == 0 { 2 + next() % 3 } else { 1 };
                let ids: Vec<[u8; 32]> = (0..k)
                    .map(|j| {
                        let mut id = [0u8; 32];
                        id[..8].copy_from_slice(&(i * 8 + j + 1).to_le_bytes());
                        id
                    })
                    .collect();
                let total = (0..k).map(|_| 1u64 << (next() % 24)).sum();
                bundles.push(SpendBundle { address: ids[0], coin_ids: ids, total_value: total });
            }
            bundles.sort_by(|a, b| {
                let l = (a.total_value as u128) * (b.coin_ids.len() as u128);
                let r = (b.total_value as u128) * (a.coin_ids.len() as u128);
                r.cmp(&l)
            });
            let total: u64 = bundles.iter().map(|b| b.total_value).sum();
            let payment = if iter % 3 == 0 {
                let mut denoms = Vec::new();
                for _ in 0..1 + next() % 3 {
                    denoms.extend(decompose_value(1 + next() % (total / 4).max(1)));
                }
                let burn = next() % (total / 8).max(1);
                Payment { denoms, other_value: burn, other_outputs: decompose_value(burn).len().max(1) }
            } else {
                Payment::of(1 + next() % total)
            };
            let fee_fn: fn(usize, usize) -> u64 = if iter % 2 == 0 { standard_fee } else { web_fee };
            let cfg = PlanConfig { distinct_payment_addresses: iter % 4 == 0, ..Default::default() };
            let Ok(p) = plan_payment_from_bundles(&bundles, &payment, fee_fn, &cfg) else { continue };

            let ids: HashSet<[u8; 32]> = p.input_coin_ids.iter().copied().collect();
            assert_eq!(ids.len(), p.input_coin_ids.len(), "duplicate input");
            let mut in_sum = 0u64;
            for b in &bundles {
                let n = b.coin_ids.iter().filter(|c| ids.contains(*c)).count();
                assert!(n == 0 || n == b.coin_ids.len(), "bundle split");
                if n > 0 {
                    in_sum += b.total_value;
                }
            }
            assert_eq!(p.in_sum, in_sum);
            assert_eq!(p.payment_denoms, payment.denoms);
            assert_eq!(p.amount, payment.amount());
            assert_eq!(p.other_outputs, payment.other_outputs);
            assert!(p.change_denoms.iter().all(|d| d.is_power_of_two()));
            assert_eq!(p.in_sum, p.amount + p.change_denoms.iter().sum::<u64>() + p.fee);
            assert!(p.fee >= fee_fn(p.input_coin_ids.len(), p.output_count()));
            assert!(p.input_coin_ids.len() <= MAX_TX_INPUTS && p.output_count() <= MAX_TX_OUTPUTS);
            // Split change pairs aim at what is paid, never at a burn.
            let mut seen = HashSet::new();
            for &c in &p.change_denoms {
                if !seen.insert(c) {
                    assert!(payment.denoms.contains(&c), "change split to {c}, which nothing pays");
                }
            }
        }
    }

    /// The fee is the least the model allows for the inputs chosen: a step
    /// lower leaves change that does not fit.
    #[test]
    fn the_fee_is_the_cheapest_that_pays_for_itself() {
        for (vals, amount) in [(vec![1u64 << 31], 1_000_000_000u64), (vec![1 << 20, 1 << 12, 8], 900_000), (vec![4096], 1024)] {
            let b = singles(&vals);
            let p = plan_transaction_from_bundles(&b, amount, web_fee, &PlanConfig { max_change_splits: 0, ..Default::default() }).unwrap();
            let n_in = p.input_coin_ids.len();
            let pay = p.payment_denoms.len();
            let m = (pay..=pay + 64).find(|&m| web_fee(n_in, m) == p.fee).expect("fee is a model step");
            assert!(m >= p.output_count(), "pays for at least its own outputs");
            for lower in pay..m {
                let fee = web_fee(n_in, lower);
                if fee == p.fee {
                    continue;
                }
                let change = p.in_sum - p.amount - fee;
                assert!(pay + decompose_value(change).len() > lower, "{vals:?}: {fee} would have done");
            }
        }
    }

    #[test]
    fn address_group_count_collapses_single_dest_payment() {
        // Payment of 6 = [4, 2] to one address; change = [4, 1].
        // Raw outputs [4, 2, 4, 1] have 2 subsets summing to 6 (both 4+2),
        // but an analyst collapses the payment into [6], leaving groups [6, 4, 1] -> only 1 interpretation.
        assert_eq!(count_address_group_interpretations(&[4, 2], &[4, 1], 6, false), 1);
        // With distinct payment addresses, groups are [4, 2, 4, 1] -> 2 interpretations.
        assert_eq!(count_address_group_interpretations(&[4, 2], &[4, 1], 6, true), 2);
    }

    /// 1,000-transaction simulation comparing Snowball vs. Self-PayJoin
    /// (single-address and multi-address) under address-group-aware ambiguity.
    /// Run with: `cargo test plan::tests::bench_1000_tx -- --nocapture`
    #[test]
    fn bench_1000_tx_snowball_vs_self_payjoin() {
        #[derive(Default)]
        struct Stats {
            txs: usize,
            total_fees: u64,
            utxo_counts: Vec<usize>,
            tx_bytes: Vec<u64>,
            ambiguous_txs: usize,
            entropy_sum: f64,
            multi_input_txs: usize,
        }

        fn run_sim(mode: u8) -> Stats {
            let mut rng: u64 = 0x5EED_CAFE_1234_5678;
            let mut next_u64 = || {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                rng
            };

            let mut utxos: Vec<SpendBundle> = Vec::new();
            let mut next_id: u64 = 1;
            let mut push_coin = |utxos: &mut Vec<SpendBundle>, val: u64| {
                for d in decompose_value(val) {
                    let mut id = [0u8; 32];
                    id[..8].copy_from_slice(&next_id.to_le_bytes());
                    next_id += 1;
                    utxos.push(SpendBundle {
                        address: id,
                        coin_ids: vec![id],
                        total_value: d,
                    });
                }
            };

            for _ in 0..8 {
                push_coin(&mut utxos, 16_384);
            }

            let mut stats = Stats::default();
            let distinct_addrs = mode == 2;
            let cfg = PlanConfig {
                distinct_payment_addresses: distinct_addrs,
                ..PlanConfig::default()
            };

            for step in 0..1000 {
                if step % 5 == 0 {
                    let dep = 1u64 << (10 + (next_u64() % 6));
                    push_coin(&mut utxos, dep);
                }

                utxos.sort_by(|a, b| b.total_value.cmp(&a.total_value));

                let amount = if next_u64() % 2 == 0 {
                    1u64 << (8 + (next_u64() % 6))
                } else {
                    200 + (next_u64() % 6000)
                };

                let plan = if mode == 0 {
                    plan_snowball_baseline(&utxos, amount, standard_fee)
                } else {
                    plan_transaction_from_bundles(&utxos, amount, standard_fee, &cfg)
                };

                let Ok(plan) = plan else { continue };

                let spent: HashSet<[u8; 32]> = plan.input_coin_ids.iter().copied().collect();
                utxos.retain(|b| !spent.contains(&b.coin_ids[0]));
                for &d in &plan.change_denoms {
                    push_coin(&mut utxos, d);
                }

                stats.txs += 1;
                stats.total_fees += plan.fee;
                stats.utxo_counts.push(utxos.len());
                stats
                    .tx_bytes
                    .push(estimate_tx_bytes(plan.input_coin_ids.len(), plan.output_count()));
                if plan.input_coin_ids.len() > 1 {
                    stats.multi_input_txs += 1;
                }
                let interp = count_address_group_interpretations(
                    &plan.payment_denoms,
                    &plan.change_denoms,
                    plan.amount,
                    distinct_addrs,
                );
                if interp >= 2 {
                    stats.ambiguous_txs += 1;
                }
                stats.entropy_sum += (interp as f64).log2();
            }
            stats
        }

        let base = run_sim(0);
        let pj_single = run_sim(1);
        let pj_multi = run_sim(2);

        let summarize = |name: &str, mut s: Stats| {
            s.utxo_counts.sort();
            s.tx_bytes.sort();
            let mean_utxo = s.utxo_counts.iter().sum::<usize>() as f64 / s.txs as f64;
            let p95_utxo = s.utxo_counts[(s.txs * 95) / 100];
            let max_utxo = *s.utxo_counts.last().unwrap();
            let mean_bytes = s.tx_bytes.iter().sum::<u64>() as f64 / s.txs as f64;
            let amb_pct = (s.ambiguous_txs as f64 / s.txs as f64) * 100.0;
            let mean_ent = s.entropy_sum / s.txs as f64;
            let multi_pct = (s.multi_input_txs as f64 / s.txs as f64) * 100.0;
            println!(
                "{name:22} | txs={:4} | utxo mean={:5.1} p95={:3} max={:3} | bytes={:6.0} | fees={:6} | ambig={:5.1}% (H={:.2}b) | multi-in={:4.1}%",
                s.txs, mean_utxo, p95_utxo, max_utxo, mean_bytes, s.total_fees, amb_pct, mean_ent, multi_pct
            );
        };

        println!("\n=== 1,000-tx Simulation (Address-Group-Aware Ambiguity) ===");
        summarize("Snowball (1-addr)", base);
        summarize("Self-PayJoin (1-addr)", pj_single);
        summarize("Self-PayJoin (m-addr)", pj_multi);
    }
}
