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

    let ratio = actual_time as f64 / expected_time as f64;
    let clamped_ratio = ratio.clamp(
        1.0 / MAX_ADJUSTMENT_FACTOR as f64,
        MAX_ADJUSTMENT_FACTOR as f64,
    );

    let current_target = target_to_u256(&state.target);
    let new_target_f64 = current_target as f64 * clamped_ratio;
    let new_target = new_target_f64 as u128;
    let new_target = new_target.min(u128::MAX);

    let result = u256_to_target(new_target);

    tracing::info!(
        "Difficulty adjustment at height {}: actual={}s expected={}s ratio={:.2} old_target={} new_target={}",
        state.height,
        actual_time,
        expected_time,
        clamped_ratio,
        hex::encode(state.target),
        hex::encode(result)
    );

    result
}

fn target_to_u256(target: &[u8; 32]) -> u128 {
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&target[0..16]);
    u128::from_be_bytes(bytes)
}

fn u256_to_target(value: u128) -> [u8; 32] {
    let mut result = [0xffu8; 32];
    let bytes = value.to_be_bytes();
    result[0..16].copy_from_slice(&bytes);
    result
}

pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Validate a block's timestamp against the chain.
/// Returns an error if the timestamp is invalid.
pub fn validate_timestamp(
    new_timestamp: u64,
    previous_states: &[State],
    current_time: u64,
) -> Result<()> {
    // Rule 1: Block timestamp must not be more than 2 hours in the future
    const MAX_FUTURE_BLOCK_TIME: u64 = 2 * 60 * 60; // 2 hours
    
    if new_timestamp > current_time + MAX_FUTURE_BLOCK_TIME {
        bail!(
            "Block timestamp too far in future: {} > {} (max future: {}s)",
            new_timestamp,
            current_time,
            MAX_FUTURE_BLOCK_TIME
        );
    }

    // Rule 2: Block timestamp must be greater than median of last 11 blocks
    if previous_states.len() >= 11 {
        let mut recent_timestamps: Vec<u64> = previous_states
            .iter()
            .rev()
            .take(11)
            .map(|s| s.timestamp)
            .collect();
        
        recent_timestamps.sort_unstable();
        let median = recent_timestamps[5]; // Middle of 11 elements
        
        if new_timestamp <= median {
            bail!(
                "Block timestamp {} must be greater than median of last 11 blocks ({})",
                new_timestamp,
                median
            );
        }
    } else if let Some(last_state) = previous_states.last() {
        // If we don't have 11 blocks yet, just check it's after the previous block
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
    // 1. Check parent linkage immediately
    if batch.prev_midstate != state.midstate {
        // This specific error string can be caught by the node to trigger orphan logic
        bail!("Block parent mismatch: expected {}, got {}", 
              hex::encode(state.midstate), 
              hex::encode(batch.prev_midstate));
    }
       
    
    // Apply transactions and tally fees
    let mut total_fees: u64 = 0;
    for tx in &batch.transactions {
        total_fees += tx.fee() as u64;
        apply_transaction(state, tx)?;
    }

    // Validate coinbase count
    let reward = block_reward(state.height);
    let expected_coinbase = reward + total_fees;
    if batch.coinbase.len() as u64 != expected_coinbase {
        bail!("Invalid coinbase count...");
    }

    // CALCULATE what the midstate WILL BE after adding coinbase
    let mut future_midstate = state.midstate;
    for coin in &batch.coinbase {
        future_midstate = hash_concat(&future_midstate, coin);
    }

    // Verify extension against the FUTURE midstate
    verify_extension(future_midstate, &batch.extension, &batch.target)?;
    

    // NOW add coinbase coins
    for coin in &batch.coinbase {
        if !state.coins.insert(*coin) {
            bail!("Duplicate coinbase coin");
        }
        state.midstate = hash_concat(&state.midstate, coin);
    }

    // Update rest of state
    state.midstate = batch.extension.final_hash;
    state.depth += EXTENSION_ITERATIONS;
    state.height += 1;
    
    // Use the timestamp from the batch (set by miner)
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
