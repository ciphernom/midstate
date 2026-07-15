// src/wallet/defrag.rs
//
// Cross-address WOTS coin defragmentation.
// Sweeps fragmented one-time WOTS coins into a fresh reusable MSS destination
// using a sequence of standard commit-reveal transactions.
//
// This module is Tier 2 under the Midstate Documentation Standard because it
// participates in one-time signature discipline and persistent wallet state.

use super::*;
use crate::core::{compute_address, MAX_TX_INPUTS, MAX_TX_OUTPUTS};
use anyhow::{bail, Result};

/// Linear transaction fee model used for large defragmentation batches.
///
/// # Reasoning
/// Defrag batches are among the largest standard transactions the wallet
/// will ever construct (up to 256 inputs with multi-kilobyte WOTS witnesses).
/// A flat fee both underprices relay and distorts batch economics.
/// A linear model in (inputs, outputs) is the minimal shape that can be
/// mapped onto a real per-byte consensus fee rule.
#[derive(Clone, Copy, Debug)]
pub struct FeePolicy {
    pub base: u64,
    pub per_input: u64,
    pub per_output: u64,
}

impl FeePolicy {
    pub fn fee(&self, n_inputs: usize, n_outputs: usize) -> u64 {
        self.base
            .saturating_add(self.per_input.saturating_mul(n_inputs as u64))
            .saturating_add(self.per_output.saturating_mul(n_outputs as u64))
    }

    pub fn flat(f: u64) -> Self {
        FeePolicy { base: f, per_input: 0, per_output: 0 }
    }
}

/// A single, fully-specified defragmentation batch ready to be executed
/// via the normal `prepare_commit` → `sign_reveal` → `complete_reveal` path.
pub struct DefragBatchPlan {
    pub input_coin_ids: Vec<[u8; 32]>,
    pub outputs: Vec<OutputData>,
    pub total_in: u64,
    pub fee: u64,
    pub remaining_fragmented_coins: usize,
}

/// Internal atomic unit of selection: all live spendable coins at one address.
pub struct SpendBundle {
    pub address: [u8; 32],
    pub coin_ids: Vec<[u8; 32]>,
    pub total_value: u64,
}

/// Resolves the shape-dependent fee via fixed-point iteration.
///
/// The fee depends on output count, which depends on `decompose_value(total - fee)`.
/// This function finds the smallest `n` such that the resulting decomposition
/// does not require more than `n` outputs.
fn resolve_fee(policy: &FeePolicy, n_inputs: usize, total: u64) -> Option<(u64, Vec<u64>)> {
    let mut n_out = 1usize;
    loop {
        let fee = policy.fee(n_inputs, n_out);
        if fee >= total {
            return None;
        }
        let denoms = decompose_value(total - fee);
        if denoms.len() <= n_out {
            return Some((fee, denoms));
        }
        n_out = denoms.len();
    }
}

impl Wallet {
    /// Collects all currently spendable coins into atomic bundles while
    /// enforcing every wallet-level spendability rule in one place.
    ///
    /// # Reasoning
    /// Multiple call sites (defrag planning, pre-flight summary,
    /// `max_single_tx_spendable`, and future `select_coins` refactor) need
    /// an identical definition of "spendable". Duplicating the filter logic
    /// is how balance discrepancies and double-reservation bugs occur.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre: true (read-only)
    ///
    /// Post (pure):
    /// Let MssAddrs = { compute_address(k.master_pk) | k ∈ self.mss_keys }
    /// Let Pending = ⋃ { p.input_coin_ids | p ∈ self.pending }
    ///
    /// result contains exactly the coins c where:
    ///   c.coin_id ∈ live_set
    ///   c.coin_id ∉ Pending
    ///   c.wots_signed = false
    ///   c.commitment = None
    ///   c.value > 0
    ///   (include_mss ∨ c.address ∉ MssAddrs)
    ///
    /// WOTS coins are grouped by address into indivisible bundles.
    /// MSS coins appear as singleton bundles when include_mss = true.
    /// Bundles larger than MAX_TX_INPUTS are excluded (with warning).
    /// Result is sorted by value density (descending), deterministic.
    /// ```
    ///
    /// ```zed
    ///     SpendableBundles
    ///     ----------------
    ///     ΞWallet
    ///     live_set? : ℙ 𝔹³²
    ///     include_mss? : 𝔹
    ///     bundles! : seq SpendBundle
    ///
    ///     let MssAddrs = { compute_address(k.master_pk) | k ∈ mss_keys }
    ///     let Pending = ⋃ { p.input_coin_ids | p ∈ pending }
    ///
    ///     pre true
    ///     post ∀ b ∈ bundles! •
    ///            (b is WOTS ⇒ ∀ c ∈ b.coin_ids • c.address = b.address)
    ///          ∧ (b is MSS  ⇒ #b.coin_ids = 1)
    ///          ∧ b.total_value = Σ { c.value | c ∈ b.coin_ids }
    ///     post bundles! contains exactly the coins satisfying the rules above
    ///     post #bundles! ≤ #coins
    /// ```
    ///
    /// # Safety / Invariants
    /// - WOTS co-spend rule is never violated (siblings stay together).
    /// - Already-signed WOTS keys are never offered again.
    /// - Coins inside pending commits are invisible (no double reservation).
    /// - Confidential / State-Thread outputs (licenses) are excluded.
    pub fn spendable_bundles(
        &self,
        live_set: &std::collections::HashSet<[u8; 32]>,
        include_mss: bool,
    ) -> Vec<SpendBundle> {
        let mut mss_addrs = std::collections::HashSet::new();
        for k in &self.data.mss_keys {
            mss_addrs.insert(compute_address(&k.master_pk));
        }

        let pending_ids: std::collections::HashSet<[u8; 32]> = self
            .data
            .pending
            .iter()
            .flat_map(|p| p.input_coin_ids.iter().copied())
            .collect();

        let mut wots_groups: std::collections::HashMap<[u8; 32], Vec<&WalletCoin>> =
            std::collections::HashMap::new();
        let mut bundles: Vec<SpendBundle> = Vec::new();

        for coin in &self.data.coins {
            if !live_set.contains(&coin.coin_id) {
                continue;
            }
            if pending_ids.contains(&coin.coin_id) {
                continue;
            }
            if coin.commitment.is_some() || coin.value == 0 {
                continue;
            }
            if coin.wots_signed {
                continue;
            }

            if mss_addrs.contains(&coin.address) {
                if include_mss {
                    bundles.push(SpendBundle {
                        address: coin.address,
                        coin_ids: vec![coin.coin_id],
                        total_value: coin.value,
                    });
                }
            } else {
                wots_groups.entry(coin.address).or_default().push(coin);
            }
        }

        for (addr, coins) in wots_groups {
            let total_value = coins.iter().fold(0u64, |a, c| a.saturating_add(c.value));
            let coin_ids: Vec<[u8; 32]> = coins.iter().map(|c| c.coin_id).collect();
            bundles.push(SpendBundle {
                address: addr,
                coin_ids,
                total_value,
            });
        }

        bundles.retain(|b| b.coin_ids.len() <= MAX_TX_INPUTS);

        bundles.sort_by(|a, b| {
            let lhs = (a.total_value as u128) * (b.coin_ids.len() as u128);
            let rhs = (b.total_value as u128) * (a.coin_ids.len() as u128);
            rhs.cmp(&lhs)
                .then_with(|| b.total_value.cmp(&a.total_value))
                .then_with(|| a.address.cmp(&b.address))
                .then_with(|| a.coin_ids.first().cmp(&b.coin_ids.first()))
        });

        bundles
    }

    /// Plans one defragmentation batch: moves up to `max_inputs` fragmented
    /// WOTS coins to a fresh MSS destination address (minus shape-dependent fee).
    ///
    /// # Reasoning
    /// Wallets that receive many small coinbase outputs accumulate thousands
    /// of one-coin WOTS addresses. A single standard transaction can only
    /// spend `MAX_TX_INPUTS` (256) inputs. `Transaction::Consolidate` cannot
    /// help because it only works on coins sharing one address. The only safe
    /// escape is a sequence of standard self-sends that migrate value onto
    /// reusable MSS addresses.
    ///
    /// This function is deliberately pure (`&self`). All state mutation happens
    /// through the existing audited `prepare_commit` / `sign_reveal` /
    /// `complete_reveal` path.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    /// - dest is an MSS address owned by this wallet
    /// - 2 <= max_inputs <= MAX_TX_INPUTS
    /// - live_coins is the current set of confirmed UTXOs
    ///
    /// Post (pure — wallet state unchanged):
    /// result = Ok(Some(plan)) ⇒
    ///   plan.input_coin_ids is a union of whole SpendBundles
    ///   2 <= #plan.input_coin_ids <= min(max_inputs, MAX_TX_INPUTS)
    ///   plan.fee = policy.fee(n_in, n_out) for some n_out >= #plan.outputs
    ///   plan.fee < plan.total_in
    ///   sum(plan.outputs) = plan.total_in - plan.fee
    ///   ∀ o ∈ plan.outputs: o.address = dest
    ///   #plan.outputs <= MAX_TX_OUTPUTS
    ///
    /// result = Ok(None) ⇒ no economically valid batch of ≥2 coins exists
    /// result = Err(_) ⇒ precondition violated
    /// ```
    ///
    /// ```zed
    ///     PlanDefragBatch
    ///     ---------------
    ///     ΞWallet
    ///     live_coins? : ℙ 𝔹³²
    ///     dest? : 𝔹³²
    ///     policy? : FeePolicy
    ///     max_inputs? : ℕ
    ///     plan! : DefragBatchPlan ∪ {none}
    ///
    ///     let Spendable = spendable_bundles(live_coins?, false)
    ///     let Selectable = { b ∈ Spendable | b.total_value > policy?.per_input × #b }
    ///     let cap = min(max_inputs?, MAX_TX_INPUTS)
    ///
    ///     pre dest? ∈ { compute_address(k.master_pk) | k ∈ mss_keys }
    ///     pre cap ≥ 2
    ///
    ///     post plan! ≠ none ⇒
    ///       ∃ S ⊆ Selectable •
    ///         plan!.input_coin_ids = ⋃ { b.coin_ids | b ∈ S }
    ///       ∧ 2 ≤ #plan!.input_coin_ids ≤ cap
    ///       ∧ plan!.fee = policy?.fee(#plan!.input_coin_ids, n) for some n
    ///       ∧ plan!.fee < plan!.total_in
    ///       ∧ (∑ o ∈ plan!.outputs • o.value) = plan!.total_in − plan!.fee
    ///       ∧ ∀ o ∈ plan!.outputs • o.address = dest?
    ///     post plan! = none ⇒ no valid batch exists
    /// ```
    ///
    /// # Safety / Invariants
    /// - Never creates new one-time WOTS keys (only consumes existing coins).
    /// - Never violates WOTS co-spend rule (whole bundles only).
    /// - Fee never underpays for the realized transaction shape.
    /// - Economic guard prevents sweeping dust whose value ≤ marginal fee.
    /// - Destination is always a fresh MSS key created for this run.
    pub fn plan_defrag_batch(
        &self,
        live_coins: &[[u8; 32]],
        dest: [u8; 32],
        policy: &FeePolicy,
        max_inputs: usize,
    ) -> Result<Option<DefragBatchPlan>> {
        let mss_owned = self
            .data
            .mss_keys
            .iter()
            .any(|k| compute_address(&k.master_pk) == dest);
        if !mss_owned {
            bail!(
                "defrag destination {} is not an MSS address owned by this wallet",
                hex::encode(dest)
            );
        }

        let cap = max_inputs.min(MAX_TX_INPUTS);
        if cap < 2 {
            bail!("max_inputs must be at least 2 (got {})", max_inputs);
        }

        let live_set: std::collections::HashSet<[u8; 32]> = live_coins.iter().copied().collect();
        let bundles = self.spendable_bundles(&live_set, false);

        let selectable: Vec<&SpendBundle> = bundles
            .iter()
            .filter(|b| {
                let marginal = policy.per_input.saturating_mul(b.coin_ids.len() as u64);
                if b.total_value <= marginal {
                    tracing::info!(
                        "Skipping uneconomical bundle at {} (value {} <= marginal fee {})",
                        hex::encode(b.address),
                        b.total_value,
                        marginal
                    );
                    false
                } else {
                    true
                }
            })
            .collect();

        let selectable_coins: usize = selectable.iter().map(|b| b.coin_ids.len()).sum();

        let mut chosen: Vec<&SpendBundle> = Vec::new();
        let mut n_in = 0usize;
        let mut total = 0u64;

        for b in &selectable {
            if n_in + b.coin_ids.len() > cap {
                continue;
            }
            chosen.push(b);
            n_in += b.coin_ids.len();
            total = total.saturating_add(b.total_value);
        }

        if n_in < 2 {
            if let Some(best) = selectable
                .iter()
                .filter(|b| b.coin_ids.len() >= 2 && b.coin_ids.len() <= cap)
                .max_by_key(|b| b.total_value)
            {
                chosen = vec![best];
                n_in = best.coin_ids.len();
                total = best.total_value;
            } else {
                return Ok(None);
            }
        }

        let (fee, denoms) = loop {
            match resolve_fee(policy, n_in, total) {
                Some(ok) => break ok,
                None => {
                    if chosen.len() <= 1 {
                        return Ok(None);
                    }
                    let (idx, _) = chosen
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, b)| b.total_value)
                        .unwrap();
                    let dropped = chosen.remove(idx);
                    n_in -= dropped.coin_ids.len();
                    total -= dropped.total_value;
                    if n_in < 2 {
                        return Ok(None);
                    }
                }
            }
        };

        if denoms.len() > MAX_TX_OUTPUTS {
            bail!(
                "defrag batch would need {} outputs (> {})",
                denoms.len(),
                MAX_TX_OUTPUTS
            );
        }

        let selected: Vec<[u8; 32]> =
            chosen.iter().flat_map(|b| b.coin_ids.iter().copied()).collect();

        let outputs: Vec<OutputData> = denoms
            .into_iter()
            .map(|d| OutputData::Standard {
                address: dest,
                value: d,
                salt: rand::random(),
            })
            .collect();

        let remaining_fragmented_coins = selectable_coins - selected.len();

        Ok(Some(DefragBatchPlan {
            input_coin_ids: selected,
            outputs,
            total_in: total,
            fee,
            remaining_fragmented_coins,
        }))
    }

    /// Returns (count, total_value) of fragmented, spendable, non-MSS coins.
    ///
    /// # Reasoning
    /// The CLI shows this summary to the user *before* creating the fresh MSS
    /// destination key for the run. It must be policy-independent and
    /// destination-independent.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre: true (read-only)
    /// Post: returns (Σ #b.coin_ids, Σ b.total_value)
    ///       over spendable_bundles(live, include_mss = false)
    /// ```
    pub fn fragmented_summary(&self, live_coins: &[[u8; 32]]) -> (usize, u64) {
        let live_set: std::collections::HashSet<[u8; 32]> = live_coins.iter().copied().collect();
        let bundles = self.spendable_bundles(&live_set, false);
        let count = bundles.iter().map(|b| b.coin_ids.len()).sum();
        let value = bundles.iter().fold(0u64, |a, b| a.saturating_add(b.total_value));
        (count, value)
    }

    /// The largest value this wallet can move in ONE standard transaction
    /// right now (includes MSS coins).
    ///
    /// # Reasoning
    /// Used by "Fund" mode and to improve the error message in `select_coins`
    /// when a send fails due to fragmentation.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre: true (read-only)
    /// Post: returns sum of values of the density-greedy prefix of
    ///       spendable_bundles(live, include_mss = true) that fits in
    ///       MAX_TX_INPUTS inputs.
    /// ```
    pub fn max_single_tx_spendable(&self, live_coins: &[[u8; 32]]) -> u64 {
        let live_set: std::collections::HashSet<[u8; 32]> = live_coins.iter().copied().collect();
        let bundles = self.spendable_bundles(&live_set, true);
        let mut slots = 0usize;
        let mut total = 0u64;
        for bundle in &bundles {
            if slots + bundle.coin_ids.len() > MAX_TX_INPUTS {
                continue;
            }
            slots += bundle.coin_ids.len();
            total = total.saturating_add(bundle.total_value);
        }
        total
    }
}

// ==================== TESTS ====================
#[cfg(test)]
mod defrag_tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn fresh() -> (Wallet, std::path::PathBuf) {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();
        (Wallet::create(&path, b"pass").unwrap(), path)
    }

    fn live(w: &Wallet) -> Vec<[u8; 32]> {
        w.coins().iter().map(|c| c.coin_id).collect()
    }

    #[test]
    fn defrag_requires_owned_mss_dest() {
        let (mut w, _p) = fresh();
        w.import_coin([1; 32], 8, [10; 32], None).unwrap();
        let err = w
            .plan_defrag_batch(&live(&w), [0xEE; 32], &FeePolicy::flat(1), 256)
            .unwrap_err();
        assert!(err.to_string().contains("MSS"));
    }

    #[test]
    fn defrag_flat_policy_conserves_value() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        for i in 1..=5u8 {
            w.import_coin([i; 32], 16, [i + 100; 32], None).unwrap();
        }
        let pol = FeePolicy::flat(4);
        let plan = w
            .plan_defrag_batch(&live(&w), dest, &pol, 256)
            .unwrap()
            .unwrap();
        assert_eq!(plan.input_coin_ids.len(), 5);
        assert_eq!(plan.total_in, 80);
        assert_eq!(plan.fee, 4);
        let out_sum: u64 = plan.outputs.iter().map(|o| o.value()).sum();
        assert_eq!(out_sum + plan.fee, plan.total_in);
        assert!(plan.outputs.iter().all(|o| o.address() == dest));
    }

    #[test]
    fn defrag_fee_scales_and_never_underpays() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        for i in 1..=5u8 {
            w.import_coin([i; 32], 16, [i + 100; 32], None).unwrap();
        }
        let pol = FeePolicy {
            base: 0,
            per_input: 2,
            per_output: 1,
        };
        let plan = w
            .plan_defrag_batch(&live(&w), dest, &pol, 256)
            .unwrap()
            .unwrap();
        assert_eq!(plan.fee, 13);
        assert_eq!(plan.outputs.len(), 3);
        let required = pol.fee(plan.input_coin_ids.len(), plan.outputs.len());
        assert!(plan.fee >= required);
        let out_sum: u64 = plan.outputs.iter().map(|o| o.value()).sum();
        assert_eq!(out_sum + plan.fee, plan.total_in);
    }

    #[test]
    fn defrag_economic_guard_skips_dust_bundles() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        let _d1 = w.import_coin([1; 32], 3, [10; 32], None).unwrap();
        let g1 = w.import_coin([2; 32], 100, [20; 32], None).unwrap();
        let g2 = w.import_coin([3; 32], 100, [30; 32], None).unwrap();
        let pol = FeePolicy {
            base: 0,
            per_input: 5,
            per_output: 0,
        };
        let plan = w
            .plan_defrag_batch(&live(&w), dest, &pol, 256)
            .unwrap()
            .unwrap();
        assert!(!plan.input_coin_ids.contains(&_d1));
        assert!(plan.input_coin_ids.contains(&g1) && plan.input_coin_ids.contains(&g2));
    }

    #[test]
    fn defrag_keeps_wots_siblings_atomic() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        let c1 = w.import_coin([1; 32], 4, [10; 32], None).unwrap();
        let addr = w.find_coin(&c1).unwrap().address;
        let c2 = w.import_scanned(addr, 1, [20; 32], None).unwrap().unwrap();
        w.import_coin([2; 32], 32, [30; 32], None).unwrap();
        let plan = w
            .plan_defrag_batch(&live(&w), dest, &FeePolicy::flat(1), 256)
            .unwrap()
            .unwrap();
        assert!(plan.input_coin_ids.contains(&c1));
        assert!(plan.input_coin_ids.contains(&c2));
    }

    #[test]
    fn defrag_excludes_pending_signed_and_mss_coins() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        let pend = w.import_coin([1; 32], 100, [10; 32], None).unwrap();
        let signed = w.import_coin([2; 32], 100, [20; 32], None).unwrap();
        let ok1 = w.import_coin([3; 32], 8, [30; 32], None).unwrap();
        let ok2 = w.import_coin([4; 32], 8, [40; 32], None).unwrap();

        w.data.pending.push(PendingCommit {
            commitment: [7; 32],
            salt: [8; 32],
            input_coin_ids: vec![pend],
            outputs: vec![],
            change_seeds: vec![],
            created_at: 0,
            reveal_not_before: 0,
            is_consolidate: false,
        });

        let pos = w.data.coins.iter().position(|c| c.coin_id == signed).unwrap();
        w.data.coins[pos].wots_signed = true;

        let plan = w
            .plan_defrag_batch(&live(&w), dest, &FeePolicy::flat(1), 256)
            .unwrap()
            .unwrap();

        assert!(!plan.input_coin_ids.contains(&pend));
        assert!(!plan.input_coin_ids.contains(&signed));
        assert!(plan.input_coin_ids.contains(&ok1) && plan.input_coin_ids.contains(&ok2));
    }

    #[test]
    fn defrag_none_when_nothing_worth_merging() {
        let (mut w, _p) = fresh();
        let dest = w.generate_mss(4, None).unwrap();
        w.import_coin([1; 32], 1000, [10; 32], None).unwrap();
        assert!(w
            .plan_defrag_batch(&live(&w), dest, &FeePolicy::flat(10), 256)
            .unwrap()
            .is_none());
    }

    #[test]
    fn preflight_helpers() {
        let (mut w, _p) = fresh();
        let _dest = w.generate_mss(4, None).unwrap();
        for i in 1..=5u8 {
            w.import_coin([i; 32], 10, [i + 100; 32], None).unwrap();
        }
        assert_eq!(w.fragmented_summary(&live(&w)), (5, 50));
        assert_eq!(w.max_single_tx_spendable(&live(&w)), 50);
    }
}
