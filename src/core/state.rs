use super::types::*;
use super::transaction::apply_transaction;
use super::extension::verify_extension;
use anyhow::{bail, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Per-block difficulty adjustment using a Linearly Weighted Moving Average.
///
/// Recent blocks matter more than old ones: block i in the window gets weight i.
/// If miners arrive en masse, difficulty ramps quickly. If they leave, the
/// asymmetric clamp (can drop 50% per block but only rise 10%) lets the chain
/// recover in minutes rather than days.
pub fn adjust_difficulty(state: &State, previous_timestamps: &[u64]) -> [u8; 32] {
    if state.height == 0 {
        return state.target;
    }

    let n = DIFFICULTY_LOOKBACK as usize;

    // Need at least N timestamps in the window to compute N solve-time intervals.
    // previous_timestamps holds timestamps of prior blocks; state.timestamp is
    // the block we just applied.  Together they give us N+1 time-points → N intervals.
    if previous_timestamps.len() < n {
        return state.target;
    }

    // Build the N+1 time-point sequence: last N entries of previous_timestamps
    // plus the current block's timestamp.
    let base = previous_timestamps.len() - n;
    let mut weighted_sum: u128 = 0;

    for i in 0..n {
        let t_prev = previous_timestamps[base + i];
        let t_next = if i + 1 < n {
            previous_timestamps[base + i + 1]
        } else {
            state.timestamp
        };

        // Clamp individual solve times to [1, 6*T] to limit timestamp gaming.
        let raw = t_next.saturating_sub(t_prev).max(1).min(6 * TARGET_BLOCK_TIME);
        let weight = (i as u128) + 1; // 1, 2, 3, ..., N
        weighted_sum += (raw as u128) * weight;
    }

    // Denominator: if every solve time equalled TARGET_BLOCK_TIME the ratio is 1.0
    let total_weight: u128 = (n as u128) * (n as u128 + 1) / 2;
    let denominator = (TARGET_BLOCK_TIME as u128) * total_weight;

    // new_target = old_target * weighted_sum / denominator
    let old = primitive_types::U256::from_big_endian(&state.target);
    let num = primitive_types::U256::from(weighted_sum);
    let den = primitive_types::U256::from(denominator);

    if den.is_zero() {
        return state.target;
    }

    let q = old / den;
    let r = old % den;
    let unclamped = q * num + (r * num) / den;

    // Asymmetric per-block clamp: can rise 10% but drop 50%.
    let max_target = old * primitive_types::U256::from(MAX_DIFFICULTY_RISE) / primitive_types::U256::from(100u64);
    let min_target = old * primitive_types::U256::from(MAX_DIFFICULTY_DROP) / primitive_types::U256::from(100u64);

    let clamped = unclamped.max(min_target).min(max_target);
    let result: [u8; 32] = clamped.to_big_endian();

    if result != state.target {
        tracing::info!(
            "Difficulty adjustment at height {}: weighted_avg={:.1}s target={}s old={} new={}",
            state.height,
            weighted_sum as f64 / total_weight as f64,
            TARGET_BLOCK_TIME,
            hex::encode(state.target),
            hex::encode(result)
        );
    }

    result
}

pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Validate a block's timestamp against the chain.
pub fn validate_timestamp(
    new_timestamp: u64,
    previous_timestamps: &[u64],
    current_time: u64,
) -> Result<()> {
    const MAX_FUTURE_BLOCK_TIME: u64 = 2 * 60 * 60;

    if new_timestamp > current_time + MAX_FUTURE_BLOCK_TIME {
        bail!(
            "Block timestamp too far in future: {} > {} (max future: {}s)",
            new_timestamp,
            current_time,
            MAX_FUTURE_BLOCK_TIME
        );
    }

    if previous_timestamps.len() >= MEDIAN_TIME_PAST_WINDOW {
        let mut recent_timestamps: Vec<u64> = previous_timestamps
            .iter()
            .rev()
            .take(MEDIAN_TIME_PAST_WINDOW)
            .copied()
            .collect();

        recent_timestamps.sort_unstable();
        let median = recent_timestamps[MEDIAN_TIME_PAST_WINDOW / 2];

        if new_timestamp <= median {
            bail!(
                "Block timestamp {} must be greater than median of last {} blocks ({})",
                new_timestamp,
                MEDIAN_TIME_PAST_WINDOW,
                median
            );
        }
    } else if let Some(&last_ts) = previous_timestamps.last() {
        if new_timestamp <= last_ts {
            bail!(
                "Block timestamp {} must be greater than previous block timestamp {}",
                new_timestamp,
                last_ts
            );
        }
    }

    Ok(())
}

/// Apply a batch to the state
pub fn apply_batch(state: &mut State, batch: &Batch, previous_timestamps: &[u64]) -> Result<()> {
    // 1. Check parent linkage
    if batch.prev_midstate != state.midstate {
        bail!("Block parent mismatch: expected {}, got {}",
              hex::encode(state.midstate),
              hex::encode(batch.prev_midstate));
    }

    if batch.target != state.target {
        bail!("Batch target mismatch: expected {}, got {}",
              hex::encode(state.target),
              hex::encode(batch.target));
    }

    //timewarp prevention: Validate timestamp
    if state.height > 0 {
        validate_timestamp(batch.timestamp, previous_timestamps, current_timestamp())?;
    }

    // 2. Apply transactions and tally fees
    let mut total_fees: u64 = 0;
    for tx in &batch.transactions {
        total_fees += tx.fee();
        apply_transaction(state, tx)?;
    }

    // 3. Validate coinbase outputs
    let reward = block_reward(state.height);
    let allowed_value = reward + total_fees;

    let mut coinbase_total: u64 = 0;
    for (i, cb) in batch.coinbase.iter().enumerate() {
        if cb.value == 0 {
            bail!("Zero-value coinbase output {}", i);
        }
        if !cb.value.is_power_of_two() {
            bail!("Coinbase output {} value {} is not a power of 2", i, cb.value);
        }
        coinbase_total = coinbase_total.checked_add(cb.value)
            .ok_or_else(|| anyhow::anyhow!("Coinbase value overflow"))?;
    }
    if coinbase_total != allowed_value {
        bail!("Coinbase total {} != expected {} (reward {} + fees {})",
              coinbase_total, allowed_value, reward, total_fees);
    }

// 4. Compute future midstate with coinbase coin IDs
    let mut future_midstate = state.midstate;
    let coinbase_ids: Vec<[u8; 32]> = batch.coinbase.iter().map(|cb| cb.coin_id()).collect();
    for coin_id in &coinbase_ids {
        future_midstate = hash_concat(&future_midstate, coin_id);
    }

    // --- LOGGING START ---
    if state.height == 0 { // Only log for genesis to reduce noise
        tracing::error!("APPLY_BATCH DEBUG:");
        tracing::error!("  State Midstate: {}", hex::encode(state.midstate));
        tracing::error!("  Future Midstate (calculated): {}", hex::encode(future_midstate));
    }
    // --- LOGGING END ---

    {
        let expired: Vec<[u8; 32]> = state.commitment_heights.iter()
            .filter(|(_, &h)| state.height.saturating_sub(h) > COMMITMENT_TTL)
            .map(|(c, _)| *c)
            .collect();
        for c in &expired {
            state.commitments.remove(c);
            state.commitment_heights.remove(c);
        }
    }

    // 5. Verify extension against future midstate
    verify_extension(future_midstate, &batch.extension, &batch.target)?;

    // 6. Add coinbase coins to state
    for coin_id in &coinbase_ids {
        if !state.coins.insert(*coin_id) {
            bail!("Duplicate coinbase coin");
        }
        state.midstate = hash_concat(&state.midstate, coin_id);
    }

    // 7. Finalize
    state.midstate = batch.extension.final_hash;
    state.depth += EXTENSION_ITERATIONS;
    state.height += 1;
    state.timestamp = batch.timestamp;

    Ok(())
}

/// Choose the better of two states (fork resolution)
pub fn choose_best_state<'a>(a: &'a State, b: &'a State) -> &'a State {
    match a.depth.cmp(&b.depth) {
        std::cmp::Ordering::Greater => a,
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => {
            if a.midstate < b.midstate { a } else { b }
        }
    }
}
// ============================================================
// ADD THIS ENTIRE BLOCK at the bottom of src/core/state.rs
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extension::create_extension;
    use crate::core::mmr::UtxoAccumulator;
    use crate::core::wots;

    fn easy_target() -> [u8; 32] {
        [0xff; 32]
    }

    fn genesis_state() -> State {
        State::genesis().0
    }

    /// Build a valid batch on top of the given state (no transactions).
    fn make_empty_batch(state: &State, reward: u64, timestamp: u64) -> Batch {
        let coinbase = make_coinbase(state, reward);
        let mut mining_midstate = state.midstate;
        for cb in &coinbase {
            mining_midstate = hash_concat(&mining_midstate, &cb.coin_id());
        }
        // Search for a nonce that meets the target
        let mut nonce = 0u64;
        let extension = loop {
            let ext = create_extension(mining_midstate, nonce);
            if ext.final_hash < state.target {
                break ext;
            }
            nonce += 1;
        };
        Batch {
            prev_midstate: state.midstate,
            transactions: vec![],
            extension,
            coinbase,
            timestamp,
            target: state.target,
        }
    }

    fn make_coinbase(state: &State, total_value: u64) -> Vec<CoinbaseOutput> {
        let denoms = decompose_value(total_value);
        denoms.iter().enumerate().map(|(i, &value)| {
            let seed = hash_concat(&state.midstate, &(i as u64).to_le_bytes());
            let pk = wots::keygen(&seed);
            let address = compute_address(&pk);
            let salt = hash_concat(&seed, &[0xCBu8; 32]);
            CoinbaseOutput { address, value, salt }
        }).collect()
    }

    // ── apply_batch ─────────────────────────────────────────────────────

    #[test]
    fn apply_batch_advances_height() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        let timestamps = vec![state.timestamp];
        apply_batch(&mut state, &batch, &timestamps).unwrap();
        assert_eq!(state.height, 1);
    }

    #[test]
    fn apply_batch_advances_depth() {
        let mut state = genesis_state();
        let initial_depth = state.depth;
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
    let timestamps = vec![state.timestamp];
        apply_batch(&mut state, &batch, &timestamps).unwrap();
        assert_eq!(state.depth, initial_depth + EXTENSION_ITERATIONS);
    }

    #[test]
    fn apply_batch_updates_midstate() {
        let mut state = genesis_state();
        let old_midstate = state.midstate;
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        let timestamps = vec![state.timestamp];
        apply_batch(&mut state, &batch, &timestamps).unwrap();
        assert_ne!(state.midstate, old_midstate);
        assert_eq!(state.midstate, batch.extension.final_hash);
    }

    #[test]
    fn apply_batch_adds_coinbase_coins() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        let coinbase_ids: Vec<[u8; 32]> = batch.coinbase.iter().map(|c| c.coin_id()).collect();
        let timestamps = vec![state.timestamp];
        apply_batch(&mut state, &batch, &timestamps).unwrap();
        for id in &coinbase_ids {
            assert!(state.coins.contains(id), "coinbase coin should be in state");
        }
    }

    #[test]
    fn apply_batch_rejects_wrong_prev_midstate() {
        let state = genesis_state();
        let reward = block_reward(state.height);
        let mut batch = make_empty_batch(&state, reward, state.timestamp + 1);
        batch.prev_midstate = [0xFFu8; 32]; // wrong parent
        let mut state2 = state.clone();
        let timestamps = vec![state2.timestamp];
        assert!(apply_batch(&mut state2, &batch, &timestamps).is_err());
    }

    #[test]
    fn apply_batch_rejects_wrong_target() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        let mut batch = make_empty_batch(&state, reward, state.timestamp + 1);
        batch.target = [0x00; 32]; // wrong target
        let timestamps = vec![state.timestamp];
        assert!(apply_batch(&mut state, &batch, &timestamps).is_err());
    }

    #[test]
    fn apply_batch_rejects_wrong_coinbase_total() {
        let mut state = genesis_state();
        // Coinbase with too much value
        let batch = make_empty_batch(&state, block_reward(state.height) + 100, state.timestamp + 1);
        let timestamps = vec![state.timestamp];
        assert!(apply_batch(&mut state, &batch, &timestamps).is_err());
    }

    #[test]
    fn apply_batch_rejects_non_power_of_two_coinbase() {
        let mut state = genesis_state();
        let mut batch = make_empty_batch(&state, block_reward(state.height), state.timestamp + 1);
        // Corrupt a coinbase value to be non-power-of-2
        if let Some(cb) = batch.coinbase.first_mut() {
            cb.value = 3; // not a power of 2
        }
        let timestamps = vec![state.timestamp];
        assert!(apply_batch(&mut state, &batch, &timestamps).is_err());
    }

    #[test]
    fn apply_batch_rejects_past_timestamp() {
        let mut state = genesis_state();
        // Apply genesis batch first to get height > 0
        let reward = block_reward(state.height);
        let batch0 = make_empty_batch(&state, reward, state.timestamp + 1);
        let timestamps_0 = vec![state.timestamp];
        apply_batch(&mut state, &batch0, &timestamps_0).unwrap();

        // Now try a batch with timestamp <= previous
        let reward = block_reward(state.height);
        let batch1 = make_empty_batch(&state, reward, state.timestamp); // same timestamp
        let timestamps_1 = vec![state.timestamp];
        assert!(apply_batch(&mut state, &batch1, &timestamps_1).is_err());
    }

    #[test]
    fn apply_batch_validates_timestamp() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        
        // 1. Apply a valid block so height > 0
        let valid_batch = make_empty_batch(&state, reward, state.timestamp + 1);
        let timestamps_0 = vec![state.timestamp];
        apply_batch(&mut state, &valid_batch, &timestamps_0).unwrap();

        // 2. Try to apply an invalid block (timestamp not strictly greater)
        let reward2 = block_reward(state.height);
        let invalid_batch = make_empty_batch(&state, reward2, state.timestamp);
        let timestamps_1 = vec![state.timestamp];

        
        let result = apply_batch(&mut state, &invalid_batch, &timestamps_1);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("greater than previous"));
    }

    // ── choose_best_state ───────────────────────────────────────────────

    #[test]
    fn choose_best_state_higher_depth_wins() {
        let mut a = genesis_state();
        let mut b = genesis_state();
        a.depth = 100;
        b.depth = 200;
        assert_eq!(choose_best_state(&a, &b).depth, 200);
        assert_eq!(choose_best_state(&b, &a).depth, 200);
    }

    #[test]
    fn choose_best_state_equal_depth_uses_midstate() {
        let mut a = genesis_state();
        let mut b = genesis_state();
        a.depth = 100;
        b.depth = 100;
        a.midstate = [0x01; 32];
        b.midstate = [0x02; 32];
        // Lower midstate wins
        assert_eq!(choose_best_state(&a, &b).midstate, [0x01; 32]);
    }

    // ── adjust_difficulty ───────────────────────────────────────────────

    #[test]
    fn adjust_difficulty_no_change_at_height_zero() {
        let state = genesis_state();
        let result = adjust_difficulty(&state, &[]);
        assert_eq!(result, state.target);
    }

    #[test]
    fn adjust_difficulty_not_enough_history() {
        let mut state = genesis_state();
        state.height = 5;
        state.timestamp = state.timestamp + 300;

        let few_timestamps = vec![state.timestamp - 60];

        let result = adjust_difficulty(&state, &few_timestamps);
        assert_eq!(result, state.target);
    }

    #[test]
    fn adjust_difficulty_stable_when_on_target() {
        let mut state = genesis_state();
        state.height = DIFFICULTY_LOOKBACK + 1;

        // Build timestamps exactly on target (60s apart)
        let base = 1_000_000u64;
        let timestamps: Vec<u64> = (0..DIFFICULTY_LOOKBACK)
            .map(|i| base + i * TARGET_BLOCK_TIME)
            .collect();
        state.timestamp = base + DIFFICULTY_LOOKBACK * TARGET_BLOCK_TIME;

        let result = adjust_difficulty(&state, &timestamps);
        assert_eq!(result, state.target, "Target should not change when blocks are exactly on schedule");
    }

    #[test]
    fn adjust_difficulty_drops_when_blocks_slow() {
        let mut state = genesis_state();
        state.height = DIFFICULTY_LOOKBACK + 1;

        // Blocks taking 3x longer than target
        let base = 1_000_000u64;
        let timestamps: Vec<u64> = (0..DIFFICULTY_LOOKBACK)
            .map(|i| base + i * TARGET_BLOCK_TIME * 3)
            .collect();
        state.timestamp = base + DIFFICULTY_LOOKBACK * TARGET_BLOCK_TIME * 3;

        let result = adjust_difficulty(&state, &timestamps);
        // Target should increase (easier difficulty) — higher target = easier
        assert!(result > state.target, "Target should increase (get easier) when blocks are slow");
    }

    #[test]
    fn adjust_difficulty_rises_when_blocks_fast() {
        let mut state = genesis_state();
        state.height = DIFFICULTY_LOOKBACK + 1;

        // Blocks taking 1/3 the target time
        let base = 1_000_000u64;
        let timestamps: Vec<u64> = (0..DIFFICULTY_LOOKBACK)
            .map(|i| base + i * TARGET_BLOCK_TIME / 3)
            .collect();
        state.timestamp = base + DIFFICULTY_LOOKBACK * TARGET_BLOCK_TIME / 3;

        let result = adjust_difficulty(&state, &timestamps);
        // Target should decrease (harder difficulty) — lower target = harder
        assert!(result < state.target, "Target should decrease (get harder) when blocks are fast");
    }

    // ── validate_timestamp ──────────────────────────────────────────────

    #[test]
    fn validate_timestamp_accepts_recent() {
        let current_time = 1_000_000;
        let result = validate_timestamp(current_time - 10, &[], current_time);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_timestamp_rejects_far_future() {
        let current_time = 1_000_000;
        let far_future = current_time + 3 * 60 * 60; // 3 hours ahead
        let result = validate_timestamp(far_future, &[], current_time);
        assert!(result.is_err());
    }

    #[test]
    fn validate_timestamp_rejects_before_previous() {
        let prev = State {
            midstate: [0; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: easy_target(),
            height: 1,
            timestamp: 1000,
            commitment_heights: std::collections::HashMap::new(),
        };
        
        let timestamps = vec![prev.timestamp];
        
        let result = validate_timestamp(999, &timestamps, 2000);
        assert!(result.is_err());
    }
}
