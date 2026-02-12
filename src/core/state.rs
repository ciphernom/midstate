use super::types::*;
use super::transaction::apply_transaction;
use super::extension::verify_extension;
use anyhow::{bail, Result};
use std::time::{SystemTime, UNIX_EPOCH};

/// Calculate new difficulty target based on recent block times
pub fn adjust_difficulty(state: &State, previous_states: &[State]) -> [u8; 32] {
    if state.height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0 || state.height == 0 {
        return state.target;
    }

    if previous_states.len() < DIFFICULTY_ADJUSTMENT_INTERVAL as usize {
        return state.target;
    }

    let interval_start_time = previous_states
        [previous_states.len() - DIFFICULTY_ADJUSTMENT_INTERVAL as usize]
        .timestamp;
    let interval_end_time = state.timestamp;
    let actual_time = interval_end_time.saturating_sub(interval_start_time);
    let expected_time = TARGET_BLOCK_TIME * DIFFICULTY_ADJUSTMENT_INTERVAL;

    if actual_time == 0 {
        return state.target;
    }

    // Clamp ratio to [1/4, 4] — same as Bitcoin
    let clamped_actual = actual_time
        .max(expected_time / MAX_ADJUSTMENT_FACTOR)
        .min(expected_time * MAX_ADJUSTMENT_FACTOR);

    // Integer math: new_target = old_target * clamped_actual / expected_time
    // Work in u128 to avoid overflow
    let old = target_to_u128(&state.target);
    let new_target = (old as u128)
        .saturating_mul(clamped_actual as u128)
        / (expected_time as u128);
    let new_target = new_target.min(u128::MAX);

    let result = u128_to_target(new_target as u128);

    tracing::info!(
        "Difficulty adjustment at height {}: actual={}s expected={}s old={} new={}",
        state.height, actual_time, expected_time,
        hex::encode(state.target), hex::encode(result)
    );

    result
}

fn target_to_u128(target: &[u8; 32]) -> u128 {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&target[0..16]);
    u128::from_be_bytes(bytes)
}

fn u128_to_target(value: u128) -> [u8; 32] {
    let mut result = [0xffu8; 32];
    result[0..16].copy_from_slice(&value.to_be_bytes());
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
    previous_states: &[State],
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

    if previous_states.len() >= 11 {
        let mut recent_timestamps: Vec<u64> = previous_states
            .iter()
            .rev()
            .take(11)
            .map(|s| s.timestamp)
            .collect();

        recent_timestamps.sort_unstable();
        let median = recent_timestamps[5];

        if new_timestamp <= median {
            bail!(
                "Block timestamp {} must be greater than median of last 11 blocks ({})",
                new_timestamp,
                median
            );
        }
    } else if let Some(last_state) = previous_states.last() {
        if new_timestamp <= last_state.timestamp {
            bail!(
                "Block timestamp {} must be greater than previous block timestamp {}",
                new_timestamp,
                last_state.timestamp
            );
        }
    }

    Ok(())
}

/// Apply a batch to the state
pub fn apply_batch(state: &mut State, batch: &Batch) -> Result<()> {
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
        if batch.timestamp <= state.timestamp {
            bail!("Block timestamp {} must be greater than previous {}", batch.timestamp, state.timestamp);
        }
        let current_time = current_timestamp();
        const MAX_FUTURE: u64 = 2 * 60 * 60;
        if batch.timestamp > current_time + MAX_FUTURE {
            bail!("Block timestamp {} too far in future (now: {})", batch.timestamp, current_time);
        }
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
        apply_batch(&mut state, &batch).unwrap();
        assert_eq!(state.height, 1);
    }

    #[test]
    fn apply_batch_advances_depth() {
        let mut state = genesis_state();
        let initial_depth = state.depth;
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        apply_batch(&mut state, &batch).unwrap();
        assert_eq!(state.depth, initial_depth + EXTENSION_ITERATIONS);
    }

    #[test]
    fn apply_batch_updates_midstate() {
        let mut state = genesis_state();
        let old_midstate = state.midstate;
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        apply_batch(&mut state, &batch).unwrap();
        assert_ne!(state.midstate, old_midstate);
        assert_eq!(state.midstate, batch.extension.final_hash);
    }

    #[test]
    fn apply_batch_adds_coinbase_coins() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        let batch = make_empty_batch(&state, reward, state.timestamp + 1);
        let coinbase_ids: Vec<[u8; 32]> = batch.coinbase.iter().map(|c| c.coin_id()).collect();
        apply_batch(&mut state, &batch).unwrap();
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
        let mut state2 = state;
        assert!(apply_batch(&mut state2, &batch).is_err());
    }

    #[test]
    fn apply_batch_rejects_wrong_target() {
        let mut state = genesis_state();
        let reward = block_reward(state.height);
        let mut batch = make_empty_batch(&state, reward, state.timestamp + 1);
        batch.target = [0x00; 32]; // wrong target
        assert!(apply_batch(&mut state, &batch).is_err());
    }

    #[test]
    fn apply_batch_rejects_wrong_coinbase_total() {
        let mut state = genesis_state();
        // Coinbase with too much value
        let batch = make_empty_batch(&state, block_reward(state.height) + 100, state.timestamp + 1);
        assert!(apply_batch(&mut state, &batch).is_err());
    }

    #[test]
    fn apply_batch_rejects_non_power_of_two_coinbase() {
        let mut state = genesis_state();
        let mut batch = make_empty_batch(&state, block_reward(state.height), state.timestamp + 1);
        // Corrupt a coinbase value to be non-power-of-2
        if let Some(cb) = batch.coinbase.first_mut() {
            cb.value = 3; // not a power of 2
        }
        assert!(apply_batch(&mut state, &batch).is_err());
    }

    #[test]
    fn apply_batch_rejects_past_timestamp() {
        let mut state = genesis_state();
        // Apply genesis batch first to get height > 0
        let reward = block_reward(state.height);
        let batch0 = make_empty_batch(&state, reward, state.timestamp + 1);
        apply_batch(&mut state, &batch0).unwrap();

        // Now try a batch with timestamp <= previous
        let reward = block_reward(state.height);
        let batch1 = make_empty_batch(&state, reward, state.timestamp); // same timestamp
        assert!(apply_batch(&mut state, &batch1).is_err());
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
    fn adjust_difficulty_no_change_before_interval() {
        let mut state = genesis_state();
        state.height = 1; // not at adjustment interval
        let result = adjust_difficulty(&state, &[]);
        assert_eq!(result, state.target);
    }

    #[test]
    fn adjust_difficulty_no_change_at_height_zero() {
        let state = genesis_state();
        let result = adjust_difficulty(&state, &[]);
        assert_eq!(result, state.target);
    }

    #[test]
    fn adjust_difficulty_not_enough_history() {
        let mut state = genesis_state();
        state.height = DIFFICULTY_ADJUSTMENT_INTERVAL;
        let few_states = vec![genesis_state()]; // not enough
        let result = adjust_difficulty(&state, &few_states);
        assert_eq!(result, state.target);
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
        let result = validate_timestamp(999, &[prev], 2000);
        assert!(result.is_err());
    }
}
