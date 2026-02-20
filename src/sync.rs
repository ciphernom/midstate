use crate::core::{State, BatchHeader, DIFFICULTY_LOOKBACK};
use crate::core::state::{apply_batch, adjust_difficulty};
use crate::core::extension::verify_extension;
use crate::storage::Storage;
use anyhow::{bail, Result};

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
    pub fn verify_header_chain(headers: &[BatchHeader]) -> Result<()> {
        for (i, header) in headers.iter().enumerate() {
            verify_extension(
                header.post_tx_midstate,
                &header.extension,
                &header.target,
            )
            .map_err(|e| anyhow::anyhow!("Invalid PoW at header index {}: {}", i, e))?;

            if i > 0 {
                let prev = &headers[i - 1];
                if header.prev_midstate != prev.extension.final_hash {
                    bail!(
                        "Header linkage broken at index {}: prev_midstate mismatch",
                        i
                    );
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
        our_height: u64,
    ) -> Result<u64> {
        let compare_end = our_height.min(peer_headers.len() as u64);

        for h in 0..compare_end {
            match self.storage.load_batch(h)? {
                Some(our_batch) => {
                    let peer_hdr = &peer_headers[h as usize];
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
        let mut recent_headers: Vec<u64> = Vec::new();
        let window_size = DIFFICULTY_LOOKBACK as usize;

        for h in 0..target {
            let batch = self
                .storage
                .load_batch(h)?
                .ok_or_else(|| anyhow::anyhow!("Missing batch at height {} during rebuild", h))?;
            
            recent_headers.push(state.timestamp);
            if recent_headers.len() > window_size { recent_headers.remove(0); }
            apply_batch(&mut state, &batch, &recent_headers)?;
            state.target = adjust_difficulty(&state, &recent_headers);
        }
        Ok(state)
    }
}
