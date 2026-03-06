use super::types::*;
use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;



/// Create an extension by doing sequential work
pub fn create_extension(midstate: [u8; 32], nonce: u64) -> Extension {
    let mut x = hash_concat(&midstate, &nonce.to_le_bytes());
    for _ in 0..EXTENSION_ITERATIONS {
        x = hash(&x);
    }
    Extension { nonce, final_hash: x }
}


/// Verify an extension by recomputing the full sequential hash chain.
///
/// Recomputes all EXTENSION_ITERATIONS hashes from `hash(midstate || nonce)`
/// and confirms the result matches `final_hash`. This is the only
/// cryptographically sound verification method for a linear hash chain —
/// probabilistic spot-checking is insecure because interior checkpoints
/// have no algebraic binding to their neighbours, enabling subset-grinding
/// attacks regardless of the commitment scheme used.
///
/// Cost: O(EXTENSION_ITERATIONS) — exactly 1,000,000 BLAKE3 hashes ≈ 1ms.
/// This is strictly faster than mining at any non-trivial difficulty, since
/// verification requires one deterministic pass while mining requires an
/// expected (1 / difficulty_fraction) passes to find a valid nonce.
pub fn verify_extension(midstate: [u8; 32], ext: &Extension, target: &[u8; 32]) -> Result<()> {
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }
    if create_extension(midstate, ext.nonce).final_hash != ext.final_hash {
        bail!("Sequential work verification failed");
    }
    Ok(())
}


/// Mine: try nonces until one produces a final_hash below target.
/// Spawns one worker per available core, each trying independent nonces.
/// Uses an AtomicBool to instantly abort if a peer solves the block first.
pub enum MiningResult {
    Block(Extension),
    Share(Extension),
}

/// Mine: try nonces until one produces a final_hash below target (or pool_target).
/// Spawns one worker per available core, each trying independent nonces.
/// Uses an AtomicBool to instantly abort if a peer solves the block first.
pub fn mine_extension(
    midstate: [u8; 32], 
    target: [u8; 32], 
    pool_target: Option<[u8; 32]>, 
    threads: usize, 
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<std::sync::atomic::AtomicU64>
) -> Option<MiningResult> {
    let num_threads = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };

    if num_threads <= 1 {
        return mine_extension_single(midstate, target, pool_target, cancel, hash_counter); 
    }

    let found = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<(MiningResult, u64)>();

    let threads: Vec<_> = (0..num_threads)
        .map(|_| {
            let cancel = Arc::clone(&cancel);
            let found = Arc::clone(&found);
            let tx = tx.clone();
            let hash_counter = Arc::clone(&hash_counter);
            std::thread::spawn(move || {
                let lanes = crate::core::simd_mining::detected_level().lanes();
                let mut attempts = 0u64;
                loop {
                    if cancel.load(Ordering::Relaxed) || found.load(Ordering::Relaxed) {
                        return;
                    }

                    // Generate N random nonces where N = SIMD width (4 NEON, 8 AVX2)                   
                    let mut nonces = [0u64; 8]; // Max width for AVX2. Stack allocated.
                    for i in 0..lanes {
                        nonces[i] = rand::random();
                    }
                    
                    attempts += lanes as u64;
                    hash_counter.fetch_add(lanes as u64, Ordering::Relaxed);

                    //let results = crate::core::simd_mining::mine_batch(midstate, &nonces);
                    let results = crate::core::simd_mining::mine_batch(midstate, &nonces[..lanes]);
                    for &(nonce, final_hash) in &results {
                        if final_hash < target {
                            found.store(true, Ordering::Relaxed);
                            let ext = Extension { nonce, final_hash };
                            let _ = tx.send((MiningResult::Block(ext), attempts));
                            return;
                        } else if let Some(pt) = pool_target {
                            if final_hash < pt {
                                found.store(true, Ordering::Relaxed);
                                let ext = Extension { nonce, final_hash };
                                let _ = tx.send((MiningResult::Share(ext), attempts));
                                return;
                            }
                        }
                    }
                    
                    if attempts % 10_000 == 0 {
                        std::thread::yield_now();
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

    if let Some((res, attempts)) = result {
        match &res {
            MiningResult::Block(ext) => {
                tracing::info!(
                    "Found valid block extension! nonce={} attempts={} hash={} threads={}",
                    ext.nonce, attempts, hex::encode(ext.final_hash), num_threads
                );
            }
            MiningResult::Share(ext) => {
                tracing::info!(
                    "Found valid pool share! nonce={} attempts={} hash={} threads={}",
                    ext.nonce, attempts, hex::encode(ext.final_hash), num_threads
                );
            }
        }
        Some(res)
    } else {
        tracing::debug!("Mining cancelled after all threads exited ({} threads)", num_threads);
        None
    }
}

/// Single-threaded fallback.
fn mine_extension_single(
    midstate: [u8; 32], 
    target: [u8; 32], 
    pool_target: Option<[u8; 32]>, 
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<std::sync::atomic::AtomicU64>
) -> Option<MiningResult> {
    let lanes = crate::core::simd_mining::detected_level().lanes();
    let mut attempts = 0u64;

    loop {
        if cancel.load(Ordering::Relaxed) {
            tracing::debug!("Mining cancelled by network event after {} attempts", attempts);
            return None;
        }

        let mut nonces = [0u64; 8];
        for i in 0..lanes {
            nonces[i] = rand::random();
        }

        attempts += lanes as u64;
        hash_counter.fetch_add(lanes as u64, Ordering::Relaxed);

        // Pass the slice just like you did in the multi-threaded version
        let results = crate::core::simd_mining::mine_batch(midstate, &nonces[..lanes]);

        for &(nonce, final_hash) in &results {
            if final_hash < target {
                tracing::info!(
                    "Found valid block extension! nonce={} attempts={} hash={}",
                    nonce, attempts, hex::encode(final_hash)
                );
                return Some(MiningResult::Block(Extension { nonce, final_hash }));
            } else if let Some(pt) = pool_target {
                if final_hash < pt {
                    tracing::info!(
                        "Found valid pool share! nonce={} attempts={} hash={}",
                        nonce, attempts, hex::encode(final_hash)
                    );
                    return Some(MiningResult::Share(Extension { nonce, final_hash }));
                }
            }
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
    }

    #[test]
    fn create_extension_different_nonces_differ() {
        let ms = hash(b"test midstate");
        let e1 = create_extension(ms, 0);
        let e2 = create_extension(ms, 1);
        assert_ne!(e1.final_hash, e2.final_hash);
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
        };
        let full_ok = verify_extension(ms, &ext, &easy_target()).is_ok();
        let pruned_ok = verify_extension(ms, &pruned, &easy_target()).is_ok();
        assert_eq!(full_ok, pruned_ok);
    }

    // ── mine_extension (use fast-mining feature for test speed) ─────────

    #[test]
    fn mine_extension_meets_target() {
        let ms = hash(b"mine test");
        let target = easy_target();
        let cancel = Arc::new(AtomicBool::new(false));
        let counter = Arc::new(std::sync::atomic::AtomicU64::new(0)); // <-- Dummy counter for test
        
        let res = mine_extension(ms, target, None, 0, cancel, counter).unwrap();
        let ext = match res {
            MiningResult::Block(e) => e,
            MiningResult::Share(_) => panic!("Should not return share"),
        };
        assert!(ext.final_hash < target);
        assert!(verify_extension(ms, &ext, &target).is_ok());
    }


}
