use super::types::*;
use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Create a Proof-of-Work extension by executing sequential delay iterations.
///
/// This is the BLAKE3-chain proof of work, in force below
/// `WALK_ACTIVATION_HEIGHT`. Consensus code must not call it directly: use
/// [`create_extension_at`] / [`verify_extension_at`], which pick the proof of
/// work by height.
///
/// # Reasoning
/// Uses a zero-allocation `[0u8; 40]` stack array to generate the initial seed, 
/// bypassing dynamic heap allocation. It then relies entirely on the official 
/// `blake3::hash` API, which leverages highly-tuned SSE4.1/NEON/AVX-512 assembly 
/// to compute the internal column and diagonal matrix mixes simultaneously.
///
/// # Formal Specification
/// ```zed
///     CreateExtension
///     ---------------
///     midstate? : 𝔹³²
///     nonce? : ℕ₆₄
///     ext! : Extension
///
///     pre  true
///     post ext!.nonce = nonce?
///        ∧ ext!.final_hash = ℋ^{ITERS}(ℋ(midstate? ⌢ le8(nonce?)))
/// ```
pub fn create_extension(midstate: [u8; 32], nonce: u64) -> Extension {
    let mut data = [0u8; 40];
    data[0..32].copy_from_slice(&midstate);
    data[32..40].copy_from_slice(&nonce.to_le_bytes());
    
    let mut x = *blake3::hash(&data).as_bytes();
    
    for _ in 0..EXTENSION_ITERATIONS {
        x = *blake3::hash(&x).as_bytes();
    }
    
    Extension { nonce, final_hash: x }
}

/// Verify an extension by recomputing the full sequential hash chain.
///
/// BLAKE3-chain blocks only; see [`verify_extension_at`].
///
/// # Reasoning
/// Recomputes all `EXTENSION_ITERATIONS` hashes by routing directly through 
/// `create_extension`. This DRY design completely eliminates logic drift between 
/// mining and validation. Any change to the core VDF mechanics will apply 
/// identically to both roles.
///
/// Probabilistic spot-checking is insecure because interior checkpoints
/// have no algebraic binding to their neighbours, enabling subset-grinding
/// attacks. Full sequential recomputation is the only cryptographically sound method.
///
/// # Formal Specification
/// ```zed
///     VerifyExtension
///     ---------------
///     midstate? : 𝔹³²
///     ext? : Extension
///     target? : 𝔹³²
///     result! : Result<()>
///
///     pre  true
///     post result! = Ok(()) ⇔ 
///            (ext?.final_hash < target?) ∧ 
///            (create_extension(midstate?, ext?.nonce).final_hash = ext?.final_hash)
/// ```
pub fn verify_extension(midstate: [u8; 32], ext: &Extension, target: &[u8; 32]) -> Result<()> {
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }
    if create_extension(midstate, ext.nonce).final_hash != ext.final_hash {
        bail!("Sequential work verification failed");
    }
    Ok(())
}

/// The proof of work of `nonce` over `midstate` for a block at `height`:
/// the BLAKE3 chain below `WALK_ACTIVATION_HEIGHT`, Midstate Walk from it.
///
/// # Formal Specification
/// ```text
///   post  ext!.nonce = nonce?
///       ∧ ext!.final_hash = create_extension(midstate?, nonce?).final_hash   if pow_algo_at(height?) = Blake3Chain
///       ∧ ext!.final_hash = walk_hash(midstate?, nonce?)                     if pow_algo_at(height?) = Walk
/// ```
pub fn create_extension_at(height: u64, midstate: [u8; 32], nonce: u64) -> Extension {
    create_extension_with(pow_algo_at(height), midstate, nonce)
}

/// The proof of work of `nonce` over `midstate` under `algo`. For miners
/// that are told the algorithm (by name) rather than the height.
pub fn create_extension_with(algo: PowAlgo, midstate: [u8; 32], nonce: u64) -> Extension {
    match algo {
        PowAlgo::Walk => Extension {
            nonce,
            final_hash: super::walk::walk_hash(&midstate, nonce),
        },
        PowAlgo::Blake3Chain => create_extension(midstate, nonce),
    }
}

/// Verify a block's extension under the proof of work in force at `height`.
///
/// # Formal Specification
/// ```text
///   post  result! = Ok(()) ⇔ ext?.final_hash < target?
///                          ∧ create_extension_at(height?, midstate?, ext?.nonce).final_hash = ext?.final_hash
/// ```
pub fn verify_extension_at(
    height: u64,
    midstate: [u8; 32],
    ext: &Extension,
    target: &[u8; 32],
) -> Result<()> {
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }
    if create_extension_at(height, midstate, ext.nonce).final_hash != ext.final_hash {
        bail!("Proof-of-work verification failed");
    }
    Ok(())
}

/// How far from our tip an orphan's height is assumed to be, either way, by
/// [`verify_extension_near`].
pub const ORPHAN_POW_SLACK: u64 = 64;

/// Verify the extension of a block whose height is unknown (an orphan: its
/// parent is not ours, so its height cannot be read off our chain). Its
/// height is assumed to be within [`ORPHAN_POW_SLACK`] of `tip_height`, so it
/// is checked against every proof of work in force in that window: away from
/// the Midstate Walk switch that is exactly one, and across it the orphan
/// passes if either holds.
///
/// This only gates what enters the orphan pool, against spam. A block is
/// applied, and its work counted, by `apply_batch` at its real height.
pub fn verify_extension_near(
    tip_height: u64,
    midstate: [u8; 32],
    ext: &Extension,
    target: &[u8; 32],
) -> Result<()> {
    let low = tip_height.saturating_sub(ORPHAN_POW_SLACK);
    let high = tip_height.saturating_add(ORPHAN_POW_SLACK);
    match verify_extension_at(low, midstate, ext, target) {
        Ok(()) => Ok(()),
        Err(e) if pow_algo_at(low) == pow_algo_at(high) => Err(e),
        Err(_) => verify_extension_at(high, midstate, ext, target),
    }
}

/// Mine: try nonces until one produces a final_hash below target.
/// Spawns one worker per available core, each trying independent nonces.
/// Uses an AtomicBool to instantly abort if a peer solves the block first.
pub enum MiningResult {
    Block(Extension),
    Share(Extension),
}

/// Nonces one worker tries per round: the SIMD width for the BLAKE3 chain,
/// one for Midstate Walk (each attempt already has a core to itself).
fn lanes_for(algo: PowAlgo) -> usize {
    match algo {
        PowAlgo::Walk => 1,
        PowAlgo::Blake3Chain => crate::core::simd_mining::detected_level().lanes(),
    }
}

/// `(nonce, final_hash)` for each of `nonces` under `algo`.
fn attempt_batch(algo: PowAlgo, midstate: [u8; 32], nonces: &[u64]) -> Vec<(u64, [u8; 32])> {
    match algo {
        PowAlgo::Walk => nonces
            .iter()
            .map(|&n| (n, super::walk::walk_hash(&midstate, n)))
            .collect(),
        PowAlgo::Blake3Chain => crate::core::simd_mining::mine_batch(midstate, nonces),
    }
}

/// Mine: try nonces until one produces a final_hash below target (or pool_target).
/// Spawns one worker per available core, each trying independent nonces.
/// Uses an AtomicBool to instantly abort if a peer solves the block first.
///
/// `height` is the height of the block being mined: it selects the proof of
/// work ([`pow_algo_at`]).
pub fn mine_extension(
    height: u64,
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

    let algo = pow_algo_at(height);
    if num_threads <= 1 {
        return mine_extension_single(algo, midstate, target, pool_target, cancel, hash_counter);
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
                let lanes = lanes_for(algo);
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

                    let results = attempt_batch(algo, midstate, &nonces[..lanes]);
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
    algo: PowAlgo,
    midstate: [u8; 32], 
    target: [u8; 32], 
    pool_target: Option<[u8; 32]>, 
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<std::sync::atomic::AtomicU64>
) -> Option<MiningResult> {
    let lanes = lanes_for(algo);
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

        let results = attempt_batch(algo, midstate, &nonces[..lanes]);

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
        
        let res = mine_extension(0, ms, target, None, 0, cancel, counter).unwrap();
        let ext = match res {
            MiningResult::Block(e) => e,
            MiningResult::Share(_) => panic!("Should not return share"),
        };
        assert!(ext.final_hash < target);
        assert!(verify_extension(ms, &ext, &target).is_ok());
    }

    // ── proof of work by height (Midstate Walk switch) ──────────────────

    const LAST_CHAIN: u64 = WALK_ACTIVATION_HEIGHT - 1;
    const FIRST_WALK: u64 = WALK_ACTIVATION_HEIGHT;

    #[test]
    fn pow_switches_exactly_at_activation() {
        assert_eq!(pow_algo_at(0), PowAlgo::Blake3Chain);
        assert_eq!(pow_algo_at(LAST_CHAIN), PowAlgo::Blake3Chain);
        assert_eq!(pow_algo_at(FIRST_WALK), PowAlgo::Walk);
        assert_eq!(pow_algo_at(u64::MAX), PowAlgo::Walk);

        let ms = hash(b"switch");
        assert_eq!(create_extension_at(LAST_CHAIN, ms, 5).final_hash, create_extension(ms, 5).final_hash);
        assert_eq!(
            create_extension_at(FIRST_WALK, ms, 5).final_hash,
            crate::core::walk::walk_hash(&ms, 5)
        );
    }

    #[test]
    fn a_proof_only_verifies_at_heights_of_its_own_algorithm() {
        let ms = hash(b"cross-algorithm");
        let chain = create_extension_at(LAST_CHAIN, ms, 3);
        let walk = create_extension_at(FIRST_WALK, ms, 3);
        assert_ne!(chain.final_hash, walk.final_hash);

        assert!(verify_extension_at(LAST_CHAIN, ms, &chain, &easy_target()).is_ok());
        assert!(verify_extension_at(FIRST_WALK, ms, &walk, &easy_target()).is_ok());
        assert!(verify_extension_at(FIRST_WALK, ms, &chain, &easy_target()).is_err());
        assert!(verify_extension_at(LAST_CHAIN, ms, &walk, &easy_target()).is_err());
        assert!(verify_extension_at(FIRST_WALK, ms, &walk, &[0u8; 32]).is_err());
    }

    #[test]
    fn orphans_near_the_switch_accept_either_algorithm() {
        let ms = hash(b"orphan");
        let chain = create_extension_at(LAST_CHAIN, ms, 1);
        let walk = create_extension_at(FIRST_WALK, ms, 1);
        let t = easy_target();

        // Far below the switch (test builds switch too early to have a "far"): chain only.
        if FIRST_WALK > 2 * ORPHAN_POW_SLACK {
            assert!(verify_extension_near(0, ms, &chain, &t).is_ok());
            assert!(verify_extension_near(0, ms, &walk, &t).is_err());
        }
        // Either side of it, close: an orphan may be on the other side.
        for tip in [FIRST_WALK.saturating_sub(ORPHAN_POW_SLACK / 2), FIRST_WALK + ORPHAN_POW_SLACK / 2] {
            assert!(verify_extension_near(tip, ms, &chain, &t).is_ok());
            assert!(verify_extension_near(tip, ms, &walk, &t).is_ok());
        }
        // Far past it: Walk only.
        let far = FIRST_WALK + 2 * ORPHAN_POW_SLACK;
        assert!(verify_extension_near(far, ms, &walk, &t).is_ok());
        assert!(verify_extension_near(far, ms, &chain, &t).is_err());
    }

    #[test]
    fn mining_past_the_switch_produces_walk_proofs() {
        let ms = hash(b"mine walk");
        let target = easy_target();
        for threads in [1, 2] {
            let cancel = Arc::new(AtomicBool::new(false));
            let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let ext = match mine_extension(FIRST_WALK, ms, target, None, threads, cancel, counter) {
                Some(MiningResult::Block(e)) => e,
                _ => panic!("expected a block"),
            };
            assert!(verify_extension_at(FIRST_WALK, ms, &ext, &target).is_ok());
            assert!(verify_extension(ms, &ext, &target).is_err());
        }
    }
}
