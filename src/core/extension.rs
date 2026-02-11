use super::types::*;
use anyhow::{bail, Result};

/// Compute the sequential hash chain, collecting checkpoints along the way.
/// Used by both create_extension and mine_extension.
fn compute_chain(midstate: &[u8; 32], nonce: u64) -> ([u8; 32], Vec<[u8; 32]>) {
    let mut x = hash_concat(midstate, &nonce.to_le_bytes());
    let mut checkpoints = Vec::with_capacity((EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize + 1);
    checkpoints.push(x);

    for i in 1..=EXTENSION_ITERATIONS {
        x = hash(&x);
        if i % CHECKPOINT_INTERVAL == 0 {
            checkpoints.push(x);
        }
    }

    (x, checkpoints)
}

/// Derive which segments to spot-check from the final hash.
/// Deterministic: all nodes check the same segments for the same block.
/// Unpredictable: attacker must complete the full chain to learn which are checked.
fn spot_check_indices(final_hash: &[u8; 32], num_segments: usize, count: usize) -> Vec<usize> {
    let count = count.min(num_segments);
    let mut indices = Vec::with_capacity(count);
    let mut seed = *final_hash;

    while indices.len() < count {
        seed = hash(&seed);
        let raw = u64::from_le_bytes(seed[..8].try_into().unwrap());
        let idx = (raw as usize) % num_segments;
        if !indices.contains(&idx) {
            indices.push(idx);
        }
    }

    indices
}

/// Create an extension by doing sequential work
pub fn create_extension(midstate: [u8; 32], nonce: u64) -> Extension {
    let (final_hash, checkpoints) = compute_chain(&midstate, nonce);
    Extension { nonce, final_hash, checkpoints }
}

/// Verify an extension by spot-checking random checkpoint segments.
/// Cost: O(SPOT_CHECK_COUNT * CHECKPOINT_INTERVAL) instead of O(EXTENSION_ITERATIONS).
pub fn verify_extension(midstate: [u8; 32], ext: &Extension, target: &[u8; 32]) -> Result<()> {
    // 1. Difficulty check (instant)
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }

    let num_segments = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize;
    let expected_checkpoints = num_segments + 1;

    // 2. Structural check
    if ext.checkpoints.len() != expected_checkpoints {
        bail!(
            "Wrong checkpoint count: got {}, expected {}",
            ext.checkpoints.len(),
            expected_checkpoints
        );
    }

    // 3. First checkpoint must match midstate + nonce
    let expected_start = hash_concat(&midstate, &ext.nonce.to_le_bytes());
    
    // --- LOGGING START ---
    if ext.checkpoints[0] != expected_start {
        tracing::error!("VERIFY ERROR DEBUG:");
        tracing::error!("  Input Midstate: {}", hex::encode(midstate));
        tracing::error!("  Nonce: {}", ext.nonce);
        tracing::error!("  Expected Checkpoint[0] (hash(midstate+nonce)): {}", hex::encode(expected_start));
        tracing::error!("  Actual Extension Checkpoint[0]: {}", hex::encode(ext.checkpoints[0]));
    }
    // --- LOGGING END ---

    if ext.checkpoints[0] != expected_start {
        bail!("First checkpoint doesn't match midstate+nonce");
    }

    // 4. Last checkpoint must equal final_hash
    if ext.checkpoints[num_segments] != ext.final_hash {
        bail!("Last checkpoint doesn't match final_hash");
    }

    // 5. Spot-check segments
    let indices = spot_check_indices(&ext.final_hash, num_segments, SPOT_CHECK_COUNT);

    for seg in indices {
        let mut x = ext.checkpoints[seg];
        for _ in 0..CHECKPOINT_INTERVAL {
            x = hash(&x);
        }
        if x != ext.checkpoints[seg + 1] {
            bail!("Checkpoint verification failed at segment {}", seg);
        }
    }

    Ok(())
}

/// Mine: try nonces until one produces a final_hash below target.
/// Each attempt pays the full sequential work cost.
pub fn mine_extension(midstate: [u8; 32], target: [u8; 32]) -> Extension {
    let mut attempts = 0u64;

    loop {
        attempts += 1;
        let nonce: u64 = rand::random();

        let (final_hash, checkpoints) = compute_chain(&midstate, nonce);

        if final_hash < target {
            tracing::info!(
                "Found valid extension! nonce={} attempts={} hash={}",
                nonce,
                attempts,
                hex::encode(final_hash)
            );
            return Extension { nonce, final_hash, checkpoints };
        }
    }
}
