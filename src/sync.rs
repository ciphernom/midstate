use crate::core::{State, BatchHeader, DIFFICULTY_LOOKBACK};
use crate::core::state::{apply_batch, adjust_difficulty};
use crate::storage::Storage;
use anyhow::{bail, Result};
use rayon::prelude::*;

pub struct Syncer {
    storage: Storage,
}

impl Syncer {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    /// Verify PoW and internal header-to-header linkage on a contiguous
    /// slice of headers. The first header's prev_midstate is NOT checked
    /// here — that is handled by the fork-point logic.
    ///
    /// `prior_timestamps` must contain the timestamps of the blocks immediately
    /// preceding `headers` in chain order (oldest first). Pass `&[]` when
    /// `headers` starts at genesis. This is required so that timestamp
    /// validation for the first headers in the slice uses the same
    /// `previous_timestamps` window that `apply_batch` / `validate_timestamp`
    /// would see, keeping the two code paths in exact consensus.
    pub fn verify_header_chain(headers: &[BatchHeader], prior_timestamps: &[u64]) -> Result<()> {
        Self::verify_header_chain_internal(headers, prior_timestamps, true)
    }

    pub fn verify_header_chain_no_pow(headers: &[BatchHeader], prior_timestamps: &[u64]) -> Result<()> {
        Self::verify_header_chain_internal(headers, prior_timestamps, false)
    }

    /// Verify PoW and internal header-to-header linkage on a contiguous slice of headers.
    ///
    /// # Reasoning
    /// Step 1 performs a fast, sequential pass to validate timestamps, targets, and 
    /// cryptographic linkage (`prev_hash` and `prev_midstate`). 
    /// Step 2 performs the heavy Proof-of-Work (VDF) validation. To maximize throughput, 
    /// it chunks the headers according to the CPU's native SIMD width (e.g., 8 for AVX2, 
    /// 4 for NEON) and dispatches them across all available CPU cores using Rayon.
    /// This allows a standard 8-core AVX2 machine to verify 64 blocks simultaneously 
    /// in the exact same time it takes to verify a single block.
    ///
    /// # Formal Specification
    /// ```text
    /// Let 𝕎 = { x ∈ ℤ | 0 ≤ x < 2³² }
    /// Let 𝔹 = { x ∈ ℤ | 0 ≤ x < 256 }
    /// Let ℋ: 𝔹³² → 𝔹³²
    /// ```
    ///
    /// ```zed
    ///     VerifyHeaderChainInternal
    ///     -------------------------
    ///     headers? : seq BatchHeader
    ///     prior_timestamps? : seq ℕ₆₄
    ///     check_pow? : 𝔹
    ///     result! : Result<()>
    ///
    ///     pre  true
    ///     post result! = Ok(()) ⇔ 
    ///            (∀ i ∈ 1..#headers?-1 • 
    ///               headers?[i].prev_header_hash = headers?[i-1].extension.final_hash ∧
    ///               headers?[i].prev_midstate = headers?[i-1].post_tx_midstate ∧
    ///               validate_timestamp(headers?[i].timestamp) ∧
    ///               headers?[i].target = expected_target) ∧
    ///            (check_pow? ⇒ ∀ h ∈ headers? • 
    ///               h.extension.final_hash < h.target ∧ 
    ///               h.extension.final_hash = ℋ^{ITERS}(ℋ(mining_hash ⌢ le8(h.extension.nonce))))
    /// ```
    fn verify_header_chain_internal(headers: &[BatchHeader], prior_timestamps: &[u64], check_pow: bool) -> Result<()> {
        if headers.is_empty() {
            return Ok(());
        }

        let current_time = crate::core::state::current_timestamp();
        let window_size = crate::core::MEDIAN_TIME_PAST_WINDOW;

        crate::core::state::validate_timestamp(headers[0].timestamp, prior_timestamps, current_time)
            .map_err(|e| anyhow::anyhow!("Header timestamp invalid at index 0: {}", e))?;

        // Pre-allocate a sliding window to prevent O(N^2) memory exhaustion
        let mut recent_ts: std::collections::VecDeque<u64> = prior_timestamps.iter().copied().collect();
        recent_ts.push_back(headers[0].timestamp);
        if recent_ts.len() > window_size {
            let overflow = recent_ts.len() - window_size;
            drop(recent_ts.drain(0..overflow));
        }

        // 1. Fast sequential check: Ensure chain linkage is intact AND validate targets
        for i in 1..headers.len() {
            let header = &headers[i];
            let prev = &headers[i - 1];

            if header.prev_header_hash != prev.extension.final_hash {
                bail!("Header chain linkage broken at index {}: prev_header_hash mismatch", i);
            }
            if header.prev_midstate != prev.post_tx_midstate {
                bail!("Header chain linkage broken at index {}: prev_midstate mismatch", i);
            }

            // O(1) Sliding Window MTP Check
            crate::core::state::validate_timestamp(header.timestamp, recent_ts.make_contiguous(), current_time)
                .map_err(|e| anyhow::anyhow!("Header timestamp invalid at index {}: {}", i, e))?;
            
            let expected_target = crate::core::state::calculate_target(prev.height + 1, prev.timestamp);
            if header.target != expected_target {
                bail!("Invalid difficulty target at height {} (expected {}, got {})", 
                    header.height, hex::encode(expected_target), hex::encode(header.target));
            }

            recent_ts.push_back(header.timestamp);
            if recent_ts.len() > window_size {
                recent_ts.pop_front();
            }
        }

        // 2. Heavy parallel check: ONLY run if check_pow is true
        if check_pow {
            // Dynamically detect the optimal SIMD width (e.g., 8 for AVX2, 4 for NEON)
            let lane_width = crate::core::wots_simd::detected_level().lanes();

            // Chunk the headers into SIMD-sized batches, then process the chunks in parallel with Rayon
            let results: Vec<Result<(), String>> = headers
                .par_chunks(lane_width)
                .flat_map(|chunk| {
                    // 1. Calculate the initial H(mining_hash || nonce) seed for each block in the SIMD chunk
                    let mut seeds = Vec::with_capacity(chunk.len());
                    for header in chunk {
                        let mining_target = crate::core::types::compute_header_hash(header);
                        let mut data = [0u8; 40];
                        data[0..32].copy_from_slice(&mining_target);
                        data[32..40].copy_from_slice(&header.extension.nonce.to_le_bytes());
                        seeds.push(crate::core::types::hash(&data));
                    }

                    // 2. Execute 1,000,000 iterations for ALL blocks in this chunk SIMULTANEOUSLY
                    // Uses the dedicated, branchless PoW verifier to preserve full SIMD speed.
                    let final_hashes = crate::core::wots_simd::verify_pow_batch(&seeds);

                    // 3. Verify the final outputs
                    let mut chunk_results = Vec::with_capacity(chunk.len());
                    for (i, final_hash) in final_hashes.into_iter().enumerate() {
                        if final_hash >= chunk[i].target {
                            chunk_results.push(Err(format!("Extension doesn't meet difficulty target at height {}", chunk[i].height)));
                        } else if final_hash != chunk[i].extension.final_hash {
                            chunk_results.push(Err(format!("Sequential work verification failed at height {}", chunk[i].height)));
                        } else {
                            chunk_results.push(Ok(()));
                        }
                    }
                    
                    chunk_results // Returns a Vec, which flat_map automatically flattens
                })
                .collect();

            // If any chunk returned an error, bail out immediately
            for res in results {
                if let Err(e) = res {
                    bail!("{}", e);
                }
            }
        }

        Ok(())
    }

    /// Find the first height where our locally stored chain and the peer's
    /// header chain diverge.  Everything below this height is shared history.
    ///
    /// `peer_headers` covers [0, peer_height).  We compare against our local
    /// batches stored on disk.
    pub fn find_fork_point(
        &self,
        peer_headers: &[BatchHeader],
        headers_start_height: u64, 
        our_height: u64,
    ) -> Result<u64> {
        if our_height <= headers_start_height {
            return Ok(headers_start_height);
        }
        let compare_end = our_height.min(headers_start_height + peer_headers.len() as u64);

        // Track expected parent to detect internal DB corruption
        let mut expected_parent = None;
        if headers_start_height > 0 {
            if let Ok(Some(prev)) = self.storage.load_batch(headers_start_height - 1) {
                expected_parent = Some(prev.extension.final_hash);
            }
        }

        for h in headers_start_height..compare_end {
            let idx = (h - headers_start_height) as usize;
            match self.storage.load_batch(h)? {
                Some(our_batch) => {
                    // FIX: Detect Frankenstein local database
                    if let Some(parent) = expected_parent {
                        if our_batch.prev_header_hash != parent {
                            tracing::warn!("Local database corruption (Frankenstein chain) detected at height {}. Forcing fork point here.", h);
                            return Ok(h);
                        }
                    }
                    expected_parent = Some(our_batch.extension.final_hash);

                    let peer_hdr = &peer_headers[idx];
                    if our_batch.extension.final_hash != peer_hdr.extension.final_hash {
                        tracing::info!("Fork detected at height {}", h);
                        return Ok(h);
                    }
                }
                None => {
                    return Ok(h);
                }
            }
        }

        Ok(compare_end)
    }

    /// Rebuild local state from genesis up to (but not including) `target`,
    /// using batches already on disk.
    pub fn rebuild_state_to(&self, target: u64) -> Result<State> {
        let mut state = State::genesis().0;
        let mut recent_headers: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
        let window_size = DIFFICULTY_LOOKBACK as usize;

        for h in 0..target {
            let batch = self
                .storage
                .load_batch(h)?
                .ok_or_else(|| anyhow::anyhow!("Missing batch at height {} during rebuild", h))?;
            
            apply_batch(&mut state, &batch, recent_headers.make_contiguous(), &mut std::collections::HashMap::new())?;

            
            recent_headers.push_back(batch.timestamp);
            if recent_headers.len() > window_size { recent_headers.pop_front(); }
            
            state.target = adjust_difficulty(&state);
        }
        Ok(state)
    }
}
