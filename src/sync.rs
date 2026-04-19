use crate::core::{State, BatchHeader, DIFFICULTY_LOOKBACK};
use crate::core::state::{apply_batch, adjust_difficulty};
use crate::core::extension::verify_extension;
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

    fn verify_header_chain_internal(headers: &[BatchHeader], prior_timestamps: &[u64], check_pow: bool) -> Result<()> {
        if headers.is_empty() {
            return Ok(());
        }

        let current_time = crate::core::state::current_timestamp();
        let window_size = crate::core::MEDIAN_TIME_PAST_WINDOW;

        crate::core::state::validate_timestamp(headers[0].timestamp, prior_timestamps, current_time)
            .map_err(|e| anyhow::anyhow!("Header timestamp invalid at index 0: {}", e))?;

        // 1. Fast sequential check: Ensure chain linkage is intact AND validate targets
        for i in 1..headers.len() {
            let header = &headers[i];
            let prev = &headers[i - 1];

            // CHECK LINKAGE FIRST!
            if header.prev_header_hash != prev.extension.final_hash {
                bail!("Header chain linkage broken at index {}: prev_header_hash mismatch", i);
            }
            if header.prev_midstate != prev.post_tx_midstate {
                bail!("Header chain linkage broken at index {}: prev_midstate mismatch", i);
            }

            let combined: Vec<u64> = prior_timestamps.iter()
                .chain(headers[..i].iter().map(|h| &h.timestamp))
                .copied()
                .collect();
            let window_start = combined.len().saturating_sub(window_size);
            let previous_timestamps = &combined[window_start..];

            // THEN CHECK MTP
            crate::core::state::validate_timestamp(header.timestamp, previous_timestamps, current_time)
                .map_err(|e| anyhow::anyhow!("Header timestamp invalid at index {}: {}", i, e))?;
            
            let expected_target = crate::core::state::calculate_target(prev.height + 1, prev.timestamp);
            if header.target != expected_target {
                bail!("Invalid difficulty target at height {} (expected {}, got {})", 
                    header.height, hex::encode(expected_target), hex::encode(header.target));
            }
        }

        // 2. Heavy parallel check: ONLY run if check_pow is true
        if check_pow {
            let results: Vec<Result<(), String>> = headers
                    .par_iter()
                    .enumerate()
                    .map(|(i, header)| {
                        let mining_target = crate::core::types::compute_header_hash(header);
                        verify_extension(
                            mining_target,
                            &header.extension,
                            &header.target,
                        ).map_err(|e| format!("Invalid PoW at header index {}: {}", i, e))
                    })
                    .collect();

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
        // If the peer's headers start way ahead of our local tip, 
        // force the fork point to the start of their headers to trigger the deep fork guard.
        if our_height <= headers_start_height {
            return Ok(headers_start_height);
        }
        let compare_end = our_height.min(headers_start_height + peer_headers.len() as u64);

        for h in headers_start_height..compare_end {
            let idx = (h - headers_start_height) as usize;
            match self.storage.load_batch(h)? {
                Some(our_batch) => {
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
