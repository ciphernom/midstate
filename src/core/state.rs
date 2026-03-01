use super::types::*;
use super::transaction::{apply_transaction_no_sig_check, verify_transaction_sigs};

use super::extension::verify_extension;
use anyhow::{bail, Result};
use primitive_types::U256;
use std::time::{SystemTime, UNIX_EPOCH};
use rayon::prelude::*;

/// Adjusts the mining difficulty using the ASERT algorithm.
///
/// ASERT compares absolute elapsed time since genesis against the ideal
/// schedule (height × TARGET_BLOCK_TIME) and applies an exponential
/// correction with a configurable half-life. This eliminates the
/// sliding-window exploits (time warp, hash-and-flee, echo effects)
/// inherent to relative algorithms like LWMA.
///
/// All arithmetic is deterministic integer math (16.16 fixed-point Taylor
/// polynomial for 2^x) — no floating-point is used. 
/// difficulty adjustment logic: 
pub fn calculate_target(height: u64, timestamp: u64) -> [u8; 32] {
    let (genesis, _) = State::genesis();
    if height == 0 { return genesis.target; }

    // 1. Drift = how far actual time is from ideal time
    let ideal_time = (height - genesis.height) as i64 * (TARGET_BLOCK_TIME as i64);
    let actual_time = (timestamp as i64).saturating_sub(genesis.timestamp as i64);
    let drift = actual_time - ideal_time;

    // 2. Fixed-point exponent: drift / half_life in 16.16
    let exponent = drift.saturating_mul(65536) / ASERT_HALF_LIFE;
    let shifts = exponent >> 16;       // integer part (whole powers of 2)
    let frac = exponent & 0xFFFF;      // fractional part

    // 3. Taylor polynomial approximation of 2^frac (16.16 fixed-point)
    //    Coefficients match the BCH aserti3-2d reference implementation.
    let mut factor = 65536i64;
    factor += (frac * 45426) >> 16;
    factor += (frac * frac * 15746) >> 32;
    factor += (frac * frac * frac * 3643) >> 48;

    // 4. Apply factor to genesis target (divide-first to avoid U256 overflow,
    //    since genesis target can be ~2^253 and factor ~2^17)
    let mut target = U256::from_big_endian(&genesis.target);
    let f = U256::from(factor as u64);
    let base = U256::from(65536u64);
    target = target / base * f + (target % base) * f / base;

    let ceiling = U256::from_big_endian(&[0xff; 32]);

    if shifts > 0 {
        let s = (shifts as usize).min(255);
        let headroom = ceiling >> s;
        target = if target > headroom { ceiling } else { target << s };
    } else if shifts < 0 {
        let s = ((-shifts) as usize).min(255);
        target = target >> s;
    }

    // 5. Clamp: never zero, never above the absolute ceiling
    if target > ceiling || target.is_zero() {
        target = ceiling;
    }

    target.to_big_endian()
}

/// Convenience wrapper around `calculate_target` for a full `State` object.
pub fn adjust_difficulty(state: &State) -> [u8; 32] {
    calculate_target(state.height, state.timestamp)
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

fn calculate_work(target: &[u8; 32]) -> u64 {
    let t = U256::from_big_endian(target);
    if t.is_zero() { return 0; }
    (U256::MAX / t).low_u64()
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

    // Reject oversized batches at consensus level (not just local mining policy)
    if batch.transactions.len() > MAX_BATCH_SIZE {
        bail!("Batch exceeds max transaction count: {} > {}", batch.transactions.len(), MAX_BATCH_SIZE);
    }

    //timewarp prevention: Validate timestamp
    if state.height > 0 {
        validate_timestamp(batch.timestamp, previous_timestamps, current_timestamp())?;
    }

    // 2. Reject batches that would require excessive signature verification
    let total_inputs: usize = batch.transactions.iter().map(|tx| match tx {
        Transaction::Reveal { inputs, .. } => inputs.len(),
        _ => 0,
    }).sum();
    if total_inputs > MAX_BATCH_INPUTS {
        bail!("Batch exceeds max total inputs: {} > {}", total_inputs, MAX_BATCH_INPUTS);
    }

    // 3. Apply transactions and tally fees
    // Phase 1: verify all signatures in parallel (pure, no state mutation)
    batch.transactions.par_iter().try_for_each(|tx| {
        verify_transaction_sigs(tx, state.height)
    })?;

    // Phase 2: apply sequentially (state mutation, sigs already verified)
    let mut total_fees: u64 = 0;
    for tx in &batch.transactions {
        total_fees = total_fees.checked_add(tx.fee()).ok_or_else(|| anyhow::anyhow!("Fee overflow"))?;
        apply_transaction_no_sig_check(state, tx)?;
    }

    // 3. Validate coinbase outputs
    let reward = block_reward(state.height);
    let allowed_value = reward.checked_add(total_fees).ok_or_else(|| anyhow::anyhow!("Reward overflow"))?;

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

    // --- NEW: State Root Validation ---
    // Simulate adding coinbase coins to calculate the exact state root
    let mut temp_state_coins = state.coins.clone();
    for coin_id in &coinbase_ids {
        temp_state_coins.insert(*coin_id);
    }
    let smt_root = hash_concat(&temp_state_coins.root(), &state.commitments.root());
    let expected_state_root = hash_concat(&smt_root, &state.chain_mmr.root());
    
    if batch.state_root != expected_state_root {
        bail!("State root mismatch: expected {}, got {}", hex::encode(expected_state_root), hex::encode(batch.state_root));
    }
    
    // Hash the state root into the midstate BEFORE verifying the PoW!
    future_midstate = hash_concat(&future_midstate, &batch.state_root);
    // -----------------------------------

    // 5. Verify extension against future midstate
    verify_extension(future_midstate, &batch.extension, &batch.target)?;

    // 6. Add coinbase coins to state
    for coin_id in &coinbase_ids {
        if !state.coins.insert(*coin_id) {
            bail!("Duplicate coinbase coin");
        }
    }

    // 7. Finalize
    state.midstate = batch.extension.final_hash;
    state.chain_mmr.append(&batch.extension.final_hash);
    state.depth += calculate_work(&batch.target);
    state.height += 1;
    state.timestamp = batch.timestamp;

    // 8. Deterministic GC: remove expired commitments from state.
    // This MUST happen at a deterministic point (after height increment)
    // so all nodes expire the same commitments at the same height.
    //
    // NOTE ON FUND SAFETY: Commits do not lock or modify UTXOs. If a 
    // commitment expires here, the user's funds are NOT lost or burned. 
    // The user simply loses their PoW effort and must submit a new Commit.
    //
    // Uses strict '>' so a commitment at height H expires at H + TTL + 1,
    // giving the full TTL window for reveals.
    let expired: Vec<[u8; 32]> = state.commitment_heights.iter()
        .filter(|(_, &h)| state.height.saturating_sub(h) > COMMITMENT_TTL)
        .map(|(k, _)| *k)
        .collect();
    for k in &expired {
        state.commitments.remove(k);
        state.commitment_heights.remove(k);
    }

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
        
        let mut temp_coins = state.coins.clone();
        for cb in &coinbase {
            let coin_id = cb.coin_id();
            mining_midstate = hash_concat(&mining_midstate, &coin_id);
            temp_coins.insert(coin_id);
        }
        
        let state_root = hash_concat(&temp_coins.root(), &state.chain_mmr.root());
        mining_midstate = hash_concat(&mining_midstate, &state_root);

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
            state_root,
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

    // ── adjust_difficulty (ASERT) ─────────────────────────────────────

    #[test]
    fn adjust_difficulty_no_change_at_height_zero() {
        let state = genesis_state();
        let result = adjust_difficulty(&state);
        assert_eq!(result, state.target);
    }

    #[test]
    fn adjust_difficulty_stable_when_on_target() {
        let genesis = genesis_state();
        let mut state = genesis.clone();
        state.height = 100;
        state.timestamp = genesis.timestamp + (100 * TARGET_BLOCK_TIME);

        let result = adjust_difficulty(&state);
        assert_eq!(result, genesis.target, "Target should not change when blocks are exactly on schedule");
    }

    #[test]
    fn adjust_difficulty_drops_when_blocks_slow() {
        let genesis = genesis_state();
        let mut state = genesis.clone();
        state.height = 10;
        // 10 blocks should take 600s. They took 2000s (too slow).
        state.timestamp = genesis.timestamp + 2000;

        let result = adjust_difficulty(&state);
        let old_u256 = U256::from_big_endian(&genesis.target);
        let new_u256 = U256::from_big_endian(&result);
        assert!(new_u256 > old_u256, "Target should increase (get easier) when blocks are slow");
    }

    #[test]
    fn adjust_difficulty_rises_when_blocks_fast() {
        let genesis = genesis_state();
        let mut state = genesis.clone();
        state.height = 10;
        // 10 blocks should take 600s. They took 100s (too fast).
        state.timestamp = genesis.timestamp + 100;

        let result = adjust_difficulty(&state);
        let old_u256 = U256::from_big_endian(&genesis.target);
        let new_u256 = U256::from_big_endian(&result);
        assert!(new_u256 < old_u256, "Target should decrease (get harder) when blocks are fast");
    }

    #[test]
    fn adjust_difficulty_exact_halving() {
        let genesis = genesis_state();
        let mut state = genesis.clone();
        // Mine 240 blocks instantly → drift = -14400s = exactly -1 half-life.
        state.height = 240;
        state.timestamp = genesis.timestamp; // no time passed

        let result = adjust_difficulty(&state);
        let old_u256 = U256::from_big_endian(&genesis.target);
        let new_u256 = U256::from_big_endian(&result);
        let expected = old_u256 >> 1;
        assert_eq!(new_u256, expected, "Target must exactly halve after 1 half-life of negative drift");
    }

    #[test]
    fn adjust_difficulty_ceiling_clamp() {
        let genesis = genesis_state();
        let mut state = genesis.clone();
        // Extreme stall: huge positive drift.
        state.height = 1;
        state.timestamp = genesis.timestamp + 999_999_999;

        let result = adjust_difficulty(&state);
        assert_eq!(result, [0xff; 32], "Target must clamp to the 0xff ceiling");
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
            commitment_heights: im::HashMap::new(),
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
        };
        
        let timestamps = vec![prev.timestamp];
        
        let result = validate_timestamp(999, &timestamps, 2000);
        assert!(result.is_err());
    }
}
