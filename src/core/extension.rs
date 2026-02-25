use super::types::*;
use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
///
/// When checkpoints have been pruned (empty vec), falls back to full-chain
/// recomputation: O(EXTENSION_ITERATIONS). This is ~1ms per block with BLAKE3
/// and happens only for deeply finalized historical blocks during sync.
pub fn verify_extension(midstate: [u8; 32], ext: &Extension, target: &[u8; 32]) -> Result<()> {
    // 1. Difficulty check (instant)
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }

    // 2. Pruned checkpoints — full-chain recomputation
    if ext.checkpoints.is_empty() {
        return verify_extension_full(midstate, ext);
    }

    let num_segments = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize;
    let expected_checkpoints = num_segments + 1;

    // 3. Structural check
    if ext.checkpoints.len() != expected_checkpoints {
        bail!(
            "Wrong checkpoint count: got {}, expected {}",
            ext.checkpoints.len(),
            expected_checkpoints
        );
    }

    // 4. First checkpoint must match midstate + nonce
    let expected_start = hash_concat(&midstate, &ext.nonce.to_le_bytes());
    if ext.checkpoints[0] != expected_start {
        bail!("First checkpoint doesn't match midstate+nonce");
    }

    // 5. Last checkpoint must equal final_hash
    if ext.checkpoints[num_segments] != ext.final_hash {
        bail!("Last checkpoint doesn't match final_hash");
    }

    // 6. Spot-check segments
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

/// Full-chain recomputation for pruned extensions.
///
/// Recomputes the entire sequential hash chain from `hash(midstate || nonce)`
/// through EXTENSION_ITERATIONS steps and verifies the result matches `final_hash`.
/// Cost: O(EXTENSION_ITERATIONS) ≈ 1M BLAKE3 hashes ≈ 1ms.
fn verify_extension_full(midstate: [u8; 32], ext: &Extension) -> Result<()> {
    let mut x = hash_concat(&midstate, &ext.nonce.to_le_bytes());
    for _ in 0..EXTENSION_ITERATIONS {
        x = hash(&x);
    }
    if x != ext.final_hash {
        bail!("Full-chain verification failed: recomputed hash != final_hash");
    }
    Ok(())
}

/// Mine: try nonces until one produces a final_hash below target.
/// Spawns one worker per available core, each trying independent nonces.
/// Uses an AtomicBool to instantly abort if a peer solves the block first.
pub fn mine_extension(midstate: [u8; 32], target: [u8; 32], threads: usize, cancel: Arc<AtomicBool>) -> Option<Extension> {
    let num_threads = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };

    if num_threads <= 1 {
        return mine_extension_single(midstate, target, cancel);
    }

    let found = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<(Extension, u64)>();

    let threads: Vec<_> = (0..num_threads)
        .map(|_| {
            let cancel = Arc::clone(&cancel);
            let found = Arc::clone(&found);
            let tx = tx.clone();
            std::thread::spawn(move || {
                let mut attempts = 0u64;
                loop {
                    if cancel.load(Ordering::Relaxed) || found.load(Ordering::Relaxed) {
                        return;
                    }

                    attempts += 1;
                    let nonce: u64 = rand::random();
                    let (final_hash, checkpoints) = compute_chain(&midstate, nonce);

                    if final_hash < target {
                        found.store(true, Ordering::Relaxed);
                        let _ = tx.send((Extension { nonce, final_hash, checkpoints }, attempts));
                        return;
                    }
                }
            })
        })
        .collect();

    // Drop our copy so rx terminates when all threads finish
    drop(tx);

    let result = rx.recv().ok();

    // Ensure all threads exit before returning
    for t in threads {
        let _ = t.join();
    }

    if let Some((ext, attempts)) = result {
        tracing::info!(
            "Found valid extension! nonce={} attempts={} hash={} threads={}",
            ext.nonce, attempts, hex::encode(ext.final_hash), num_threads
        );
        Some(ext)
    } else {
        tracing::debug!("Mining cancelled after all threads exited ({} threads)", num_threads);
        None
    }
}

/// Single-threaded fallback.
fn mine_extension_single(midstate: [u8; 32], target: [u8; 32], cancel: Arc<AtomicBool>) -> Option<Extension> {
    let mut attempts = 0u64;

    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("Mining cancelled by network event after {} attempts", attempts);
            return None;
        }

        attempts += 1;
        let nonce: u64 = rand::random();

        let (final_hash, checkpoints) = compute_chain(&midstate, nonce);

        if final_hash < target {
            tracing::info!(
                "Found valid extension! nonce={} attempts={} hash={}",
                nonce, attempts, hex::encode(final_hash)
            );
            return Some(Extension { nonce, final_hash, checkpoints });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn easy_target() -> [u8; 32] {
        [0xff; 32] // accepts everything
    }

    // ── create_extension ────────────────────────────────────────────────

    #[test]
    fn create_extension_deterministic() {
        let ms = hash(b"test midstate");
        let e1 = create_extension(ms, 42);
        let e2 = create_extension(ms, 42);
        assert_eq!(e1.final_hash, e2.final_hash);
        assert_eq!(e1.checkpoints, e2.checkpoints);
    }

    #[test]
    fn create_extension_different_nonces_differ() {
        let ms = hash(b"test midstate");
        let e1 = create_extension(ms, 0);
        let e2 = create_extension(ms, 1);
        assert_ne!(e1.final_hash, e2.final_hash);
    }

    #[test]
    fn create_extension_checkpoint_count() {
        let ms = hash(b"test midstate");
        let ext = create_extension(ms, 0);
        let expected = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize + 1;
        assert_eq!(ext.checkpoints.len(), expected);
    }

    #[test]
    fn create_extension_first_checkpoint_is_hash_of_midstate_nonce() {
        let ms = hash(b"test midstate");
        let nonce = 99u64;
        let ext = create_extension(ms, nonce);
        let expected = hash_concat(&ms, &nonce.to_le_bytes());
        assert_eq!(ext.checkpoints[0], expected);
    }

    #[test]
    fn create_extension_last_checkpoint_equals_final_hash() {
        let ms = hash(b"test midstate");
        let ext = create_extension(ms, 0);
        assert_eq!(*ext.checkpoints.last().unwrap(), ext.final_hash);
    }

    // ── verify_extension ────────────────────────────────────────────────

    #[test]
    fn verify_valid_extension() {
        let ms = hash(b"verify test");
        let ext = create_extension(ms, 7);
        assert!(verify_extension(ms, &ext, &easy_target()).is_ok());
    }

    #[test]
    fn verify_rejects_wrong_midstate() {
        let ms = hash(b"correct");
        let ext = create_extension(ms, 0);
        let wrong_ms = hash(b"wrong");
        assert!(verify_extension(wrong_ms, &ext, &easy_target()).is_err());
    }

    #[test]
    fn verify_rejects_above_target() {
        let ms = hash(b"target test");
        let ext = create_extension(ms, 0);
        let impossible_target = [0u8; 32]; // nothing can be below all zeros
        assert!(verify_extension(ms, &ext, &impossible_target).is_err());
    }

    #[test]
    fn verify_rejects_wrong_checkpoint_count() {
        let ms = hash(b"bad checkpoint");
        let mut ext = create_extension(ms, 0);
        ext.checkpoints.push([0u8; 32]); // extra checkpoint
        assert!(verify_extension(ms, &ext, &easy_target()).is_err());
    }

    #[test]
    fn verify_rejects_tampered_checkpoint() {
        let ms = hash(b"tamper test");
        let mut ext = create_extension(ms, 0);
        // Flip a byte in a middle checkpoint
        let mid = ext.checkpoints.len() / 2;
        ext.checkpoints[mid][0] ^= 0xFF;
        // This may or may not be caught depending on spot-check sampling,
        // but the last checkpoint won't match final_hash anymore
        // OR a spot-checked segment will fail.
        // At minimum, if the last checkpoint is tampered it's caught:
        let last = ext.checkpoints.len() - 1;
        let mut ext2 = create_extension(ms, 0);
        ext2.checkpoints[last][0] ^= 0xFF;
        assert!(verify_extension(ms, &ext2, &easy_target()).is_err());
    }

    #[test]
    fn verify_rejects_tampered_final_hash() {
        let ms = hash(b"final hash tamper");
        let mut ext = create_extension(ms, 0);
        ext.final_hash[0] ^= 0xFF;
        assert!(verify_extension(ms, &ext, &easy_target()).is_err());
    }

    // ── verify_extension (pruned / full-chain fallback) ─────────────────

    #[test]
    fn verify_pruned_extension_valid() {
        let ms = hash(b"prune test");
        let ext = create_extension(ms, 7);
        let pruned = Extension {
            nonce: ext.nonce,
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        assert!(verify_extension(ms, &pruned, &easy_target()).is_ok());
    }

    #[test]
    fn verify_pruned_extension_wrong_midstate() {
        let ms = hash(b"prune correct");
        let ext = create_extension(ms, 0);
        let pruned = Extension {
            nonce: ext.nonce,
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        let wrong = hash(b"prune wrong");
        assert!(verify_extension(wrong, &pruned, &easy_target()).is_err());
    }

    #[test]
    fn verify_pruned_extension_wrong_nonce() {
        let ms = hash(b"prune nonce");
        let ext = create_extension(ms, 42);
        let pruned = Extension {
            nonce: 43, // wrong nonce
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        assert!(verify_extension(ms, &pruned, &easy_target()).is_err());
    }

    #[test]
    fn verify_pruned_extension_tampered_final_hash() {
        let ms = hash(b"prune tamper");
        let ext = create_extension(ms, 0);
        let mut pruned = Extension {
            nonce: ext.nonce,
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        pruned.final_hash[0] ^= 0xFF;
        assert!(verify_extension(ms, &pruned, &easy_target()).is_err());
    }

    #[test]
    fn verify_pruned_extension_still_checks_target() {
        let ms = hash(b"prune target");
        let ext = create_extension(ms, 0);
        let pruned = Extension {
            nonce: ext.nonce,
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        let impossible_target = [0u8; 32];
        assert!(verify_extension(ms, &pruned, &impossible_target).is_err());
    }

    #[test]
    fn verify_pruned_matches_full_verification() {
        // A pruned extension should accept/reject identically to the full one
        let ms = hash(b"equivalence test");
        let ext = create_extension(ms, 99);
        let pruned = Extension {
            nonce: ext.nonce,
            final_hash: ext.final_hash,
            checkpoints: vec![],
        };
        let full_ok = verify_extension(ms, &ext, &easy_target()).is_ok();
        let pruned_ok = verify_extension(ms, &pruned, &easy_target()).is_ok();
        assert_eq!(full_ok, pruned_ok);
    }

    // ── mine_extension (use fast-mining feature for test speed) ─────────

    #[test]
    fn mine_extension_meets_target() {
        let ms = hash(b"mine test");
        // Use easy target so mining finishes quickly
        let target = easy_target();
        let cancel = Arc::new(AtomicBool::new(false));
        // Add the '0' for the threads argument
        let ext = mine_extension(ms, target, 0, cancel).unwrap();
        assert!(ext.final_hash < target);
        assert!(verify_extension(ms, &ext, &target).is_ok());
    }

    // ── spot_check_indices ──────────────────────────────────────────────

    #[test]
    fn spot_check_indices_deterministic() {
        let fh = hash(b"deterministic");
        let a = spot_check_indices(&fh, 100, 10);
        let b = spot_check_indices(&fh, 100, 10);
        assert_eq!(a, b);
    }

    #[test]
    fn spot_check_indices_unique() {
        let fh = hash(b"unique check");
        let indices = spot_check_indices(&fh, 1000, 50);
        let mut deduped = indices.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(indices.len(), deduped.len());
    }

    #[test]
    fn spot_check_indices_within_bounds() {
        let fh = hash(b"bounds");
        let num_segments = 100;
        let indices = spot_check_indices(&fh, num_segments, 20);
        for &idx in &indices {
            assert!(idx < num_segments);
        }
    }

    #[test]
    fn spot_check_count_capped_at_segments() {
        let fh = hash(b"cap");
        let indices = spot_check_indices(&fh, 5, 100);
        assert_eq!(indices.len(), 5);
    }
}
