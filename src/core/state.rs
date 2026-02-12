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
