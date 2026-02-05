use super::types::*;
use super::transaction::apply_transaction;
use super::extension::verify_extension;
use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};

/// Calculate new difficulty target based on recent block times
pub fn adjust_difficulty(state: &State, previous_states: &[State]) -> [u8; 32] {
    if state.height % DIFFICULTY_ADJUSTMENT_INTERVAL != 0 || state.height == 0 {
        return state.target;
    }
    
    if previous_states.len() < DIFFICULTY_ADJUSTMENT_INTERVAL as usize {
        return state.target;
    }
    
    let interval_start_time = previous_states[previous_states.len() - DIFFICULTY_ADJUSTMENT_INTERVAL as usize].timestamp;
    let interval_end_time = state.timestamp;
    let actual_time = interval_end_time.saturating_sub(interval_start_time);
    
    let expected_time = TARGET_BLOCK_TIME * DIFFICULTY_ADJUSTMENT_INTERVAL;
    
    if actual_time == 0 {
        return state.target;
    }
    
    let ratio = actual_time as f64 / expected_time as f64;
    let clamped_ratio = ratio.clamp(1.0 / MAX_ADJUSTMENT_FACTOR as f64, MAX_ADJUSTMENT_FACTOR as f64);
    
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

/// Apply a batch to the state
pub fn apply_batch(state: &mut State, batch: &Batch) -> Result<()> {
    for tx in &batch.transactions {
        apply_transaction(state, tx)?;
    }
    
    verify_extension(state.midstate, &batch.extension, &state.target)?;
    
    state.midstate = batch.extension.final_hash;
    state.depth += EXTENSION_ITERATIONS;
    state.height += 1;
    state.timestamp = current_timestamp();
    
    tracing::info!(
        "Applied batch: height={} depth={} coins={} commitments={} timestamp={}",
        state.height,
        state.depth,
        state.coins.len(),
        state.commitments.len(),
        state.timestamp
    );
    
    Ok(())
}

/// Choose the better of two states (fork resolution)
pub fn choose_best_state<'a>(a: &'a State, b: &'a State) -> &'a State {
    match a.depth.cmp(&b.depth) {
        std::cmp::Ordering::Greater => {
            tracing::debug!("Chose state A (depth {} > {})", a.depth, b.depth);
            a
        }
        std::cmp::Ordering::Less => {
            tracing::debug!("Chose state B (depth {} > {})", b.depth, a.depth);
            b
        }
        std::cmp::Ordering::Equal => {
            if a.midstate < b.midstate {
                tracing::debug!("Chose state A (tiebreaker)");
                a
            } else {
                tracing::debug!("Chose state B (tiebreaker)");
                b
            }
        }
    }
}
