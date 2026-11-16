//! Sync session state machine and helpers.
//!
//! Extracted from the monolithic `node` module during the god-module refactor (Phase 3).
//! The `SyncManager` owns the non-blocking sync session (`SyncSession` / `SyncPhase`),
//! prefetch/in-flight tracking, timeout/stall monitoring, backup (resume) logic, and
//! phase transitions. Node orchestrates by delegating pure session work here while
//! retaining reorg/apply/network side-effects (per plan guidance to mitigate risk).
//!
//! The lightweight `Syncer` (with `verify_header_chain*`, `find_fork_point`, etc.)
//! remains here for pure verification/rebuild used by both sync paths and tests.

use crate::core::{State, BatchHeader, Batch, DIFFICULTY_LOOKBACK};
use crate::core::state::{apply_batch, adjust_difficulty};
use crate::storage::Storage;
use anyhow::{bail, Result};
use rayon::prelude::*;
use libp2p::PeerId;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

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
    ///            (∀ i ∈ 1..#headers?-1 • headers?[i].height = headers?[i-1].height + 1) ∧
    ///            (check_pow? ⇒ ∀ h ∈ headers? • 
    ///               h.extension.final_hash < h.target ∧ 
    ///               h.extension.final_hash = create_extension_at(h.height, mining_hash, h.extension.nonce).final_hash)
    /// ```
    fn verify_header_chain_internal(headers: &[BatchHeader], prior_timestamps: &[u64], check_pow: bool) -> Result<()> {
        if headers.is_empty() {
            return Ok(());
        }

        let current_time = crate::core::state::current_timestamp();
        let window_size = crate::core::MEDIAN_TIME_PAST_WINDOW;

        crate::core::state::validate_timestamp(headers[0].timestamp, prior_timestamps, current_time, headers[0].height)
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
            // Heights choose the proof of work (`pow_algo_at`), so they must be
            // the chain's, not labels a peer picked: contiguous from the first.
            if header.height != prev.height + 1 {
                bail!("Header chain heights not contiguous at index {}: {} after {}", i, header.height, prev.height);
            }

            // O(1) Sliding Window MTP Check
            crate::core::state::validate_timestamp(header.timestamp, recent_ts.make_contiguous(), current_time, header.height)
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
            // Headers from the Midstate Walk switch on carry the Walk: one walk
            // per header, spread across cores. Below it, the BLAKE3 chain in
            // SIMD batches as before. Heights were checked contiguous above.
            let first_walk = headers
                .iter()
                .position(|h| crate::core::types::pow_algo_at(h.height) == crate::core::types::PowAlgo::Walk)
                .unwrap_or(headers.len());
            let (chain_headers, walk_headers) = headers.split_at(first_walk);

            let walk_results: Vec<Result<(), String>> = walk_headers
                .par_iter()
                .map(|header| {
                    let mining_hash = crate::core::types::compute_header_hash(header);
                    crate::core::extension::verify_extension_at(header.height, mining_hash, &header.extension, &header.target)
                        .map_err(|e| format!("{} at height {}", e, header.height))
                })
                .collect();
            for res in walk_results {
                if let Err(e) = res {
                    bail!("{}", e);
                }
            }

            // Dynamically detect the optimal SIMD width (e.g., 8 for AVX2, 4 for NEON)
            let lane_width = crate::core::wots_simd::detected_level().lanes();

            // Chunk the headers into SIMD-sized batches, then process the chunks in parallel with Rayon
            let results: Vec<Result<(), String>> = chain_headers
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

/// --- Non-blocking sync session (moved from node.rs during god-module refactor) ---

/// How many batch chunks to have in-flight simultaneously across peers.
/// Each chunk is up to 8 MB, so 3 = up to ~24 MB of in-flight data.
pub(crate) const BATCH_LOOKAHEAD: usize = 3;

pub(crate) const MAX_PREFETCH_BUFFER: usize = 32;
/// Hard RAM cap on the total serialized size of all prefetched batches.
/// 64 MiB is a safe limit for 512 MiB RAM devices (Raspberry Pi Zero / old routers)
/// while still allowing useful pipelining during sync.
pub(crate) const MAX_PREFETCH_RAM_BYTES: usize = 64 * 1024 * 1024;

pub(crate) const MAX_PREFETCH_DISTANCE: u64 = 10_000;

pub(crate) const SYNC_TIMEOUT_SECS: u64 = 30;
/// Extra grace period for the first chunk (relay handshake, NAT traversal, etc.)
pub(crate) const SYNC_INITIAL_TIMEOUT_SECS: u64 = 45;

/// Non-blocking sync session driven by the main event loop.
/// Replaces the old blocking `Syncer::sync_via_network` which hijacked the
/// network and dropped unrelated messages.
pub(crate) struct SyncSession {
    pub peer: PeerId,
    pub peer_height: u64,
    pub peer_depth: u128,
    pub phase: SyncPhase,
    pub started_at: Instant,
    /// Tracks when we last received useful data. Reset on every header/batch chunk.
    /// The timeout fires based on this, not `started_at`.
    pub last_progress_at: Instant,
}

pub(crate) enum SyncPhase {
    /// Downloading headers. If fast-forwarding, it holds the snapshot to verify against.
    /// `verifying` is true while a per-chunk PoW verification task is in flight;
    /// the stall monitor ignores the session in that case so queueing delays in
    /// the rayon pool don't trip a false timeout.
    Headers {
        accumulated: Vec<BatchHeader>,
        cursor: u64,
        snapshot: Option<Box<State>>,
        verifying: bool,
    },
    VerifyingHeaders,
    /// Headers verified, now downloading batches from fork_height forward.
    Batches {
        headers: Vec<BatchHeader>,
        fork_height: u64,
        candidate_state: State,
        cursor: u64,
        new_history: Vec<(u64, [u8; 32], Batch)>,
        is_fast_forward: bool,
        /// Chunks requested from secondary peers that haven't been applied yet.
        /// (start_height, peer_id)
        in_flight: BTreeMap<u64, PeerId>,
        /// Out-of-order chunks that arrived before we were ready for them.
        /// Keyed by start_height.
        prefetch_buffer: BTreeMap<u64, Vec<Batch>>,
    },
    VerifyingBatches {
        in_flight: BTreeMap<u64, PeerId>,
        prefetch_buffer: BTreeMap<u64, Vec<Batch>>,
    },
    /// Fork point found, GetBatches already sent, but state rebuild is still
    /// running in the background. Batches arriving from the peer are buffered here.
    PipelinedRebuild {
        headers: Vec<BatchHeader>,
        fork_height: u64,
        is_fast_forward: bool,
        buffered_batches: Vec<Batch>,
        in_flight: BTreeMap<u64, PeerId>,
    },
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct SyncStateBackup {
    pub cursor: u64,
    pub peer_height: u64,
    pub accumulated_headers: Vec<BatchHeader>,
}

/// How long a GetBatches sent by an ended session is remembered. Requests
/// time out after 60s (`network::MidstateNetwork::new`), so no reply can come
/// later than this.
const ABANDONED_REPLY_WINDOW: Duration = Duration::from_secs(120);

/// How long a peer whose headers broke our rules is left alone, unless it
/// claims more work first.
const INCOMPATIBLE_FOR: Duration = Duration::from_secs(3600);

#[derive(Default)]
struct SavedMeta {
    /// `cursor` reflects the file (false until the file is first looked at).
    known: bool,
    /// Height where the saved headers end. `None`: no file.
    cursor: Option<u64>,
    /// Bumped by every clear, so a write queued before the clear is dropped.
    generation: u64,
}

/// `sync_state.bin`: the PoW-checked headers sync sessions have gathered, so
/// a later session resumes where they left off instead of starting over.
///
/// # Reasoning
/// Any peer used to be able to throw this file away. A session with a peer
/// no further along than the saved headers deleted it, and that session's
/// own saves then replaced the headers with a shorter run: a peer on a dead
/// fork, or one merely at the same height, cost the node every header it
/// had and the slow re-check of the pre-Walk ones. Now nothing but a clear
/// (a finished sync, a peer caught feeding us bad blocks) removes it, and a
/// write never replaces saved headers with ones that end lower.
///
/// Writes run on the blocking pool, where they can land in any order; every
/// file operation takes the lock, and the cursor and generation checks run
/// under it.
#[derive(Default, Clone)]
pub(crate) struct SavedHeaders(Arc<Mutex<SavedMeta>>);

impl SavedHeaders {
    fn file(data_dir: &Path) -> PathBuf {
        data_dir.join("sync_state.bin")
    }

    fn lock(&self) -> MutexGuard<'_, SavedMeta> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn read(path: &Path) -> Option<SyncStateBackup> {
        let bytes = std::fs::read(path).ok()?;
        bincode::deserialize(&bytes).ok()
    }

    /// Write through a temporary file, so a crash mid-write can't leave a
    /// truncated file behind.
    fn write(path: &Path, backup: &SyncStateBackup) -> bool {
        let Ok(bytes) = bincode::serialize(backup) else { return false };
        let tmp = path.with_extension("bin.tmp");
        std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_ok()
    }

    /// The saved headers, if there are any.
    pub fn load(&self, data_dir: &Path) -> Option<SyncStateBackup> {
        let mut meta = self.lock();
        let backup = Self::read(&Self::file(data_dir));
        meta.cursor = backup.as_ref().map(|b| b.cursor);
        meta.known = true;
        backup
    }

    /// Where the saved headers end, if there are any.
    #[cfg(test)]
    pub fn cursor(&self, data_dir: &Path) -> Option<u64> {
        let mut meta = self.lock();
        if !meta.known {
            meta.cursor = Self::read(&Self::file(data_dir)).map(|b| b.cursor);
            meta.known = true;
        }
        meta.cursor
    }

    pub fn clear(&self, data_dir: &Path) {
        let mut meta = self.lock();
        let _ = std::fs::remove_file(Self::file(data_dir));
        meta.cursor = None;
        meta.known = true;
        meta.generation += 1;
    }

    /// Save `backup` on the blocking pool, unless the saved headers already
    /// reach further.
    pub fn store(&self, data_dir: &Path, backup: SyncStateBackup) {
        let this = self.clone();
        let path = Self::file(data_dir);
        let generation = self.lock().generation;
        tokio::task::spawn_blocking(move || {
            this.store_now(&path, &backup, generation);
        });
    }

    pub(crate) fn store_now(&self, path: &Path, backup: &SyncStateBackup, generation: u64) -> bool {
        let mut meta = self.lock();
        if meta.generation != generation {
            return false;
        }
        if !meta.known {
            meta.cursor = Self::read(path).map(|b| b.cursor);
            meta.known = true;
        }
        if meta.cursor.map_or(false, |saved| backup.cursor < saved) {
            return false;
        }
        let written = Self::write(path, backup);
        if written {
            meta.cursor = Some(backup.cursor);
        }
        written
    }

    /// Keep a header chunk whose PoW checked out after the session that asked
    /// for it had ended, most often because its peer dropped mid-check. It is
    /// added only where it continues the saved headers exactly, or, with
    /// nothing saved, where it continues our own chain (`links_to_local`).
    pub fn keep_chunk(&self, data_dir: &Path, chunk: Vec<BatchHeader>, links_to_local: bool) {
        let this = self.clone();
        let path = Self::file(data_dir);
        let generation = self.lock().generation;
        tokio::task::spawn_blocking(move || {
            let span = chunk.first().zip(chunk.last()).map(|(a, b)| (a.height, b.height));
            match (this.keep_chunk_now(&path, chunk, links_to_local, generation), span) {
                (Some(cursor), Some((from, to))) => tracing::info!(
                    "Kept headers {}..{}, verified after their sync session ended. The next sync resumes at {}.",
                    from, to, cursor
                ),
                (None, Some((from, to))) => tracing::debug!(
                    "Verified headers {}..{} outlived their sync session and don't continue the saved ones; dropped",
                    from, to
                ),
                _ => {}
            }
        });
    }

    pub(crate) fn keep_chunk_now(&self, path: &Path, chunk: Vec<BatchHeader>, links_to_local: bool, generation: u64) -> Option<u64> {
        let (first_height, first_prev_hash, first_prev_midstate) = {
            let first = chunk.first()?;
            (first.height, first.prev_header_hash, first.prev_midstate)
        };
        let mut meta = self.lock();
        if meta.generation != generation {
            return None;
        }
        let backup = match Self::read(path) {
            Some(mut saved) => {
                let last = saved.accumulated_headers.last()?;
                if first_height != saved.cursor
                    || last.height + 1 != first_height
                    || first_prev_hash != last.extension.final_hash
                    || first_prev_midstate != last.post_tx_midstate
                {
                    return None;
                }
                saved.cursor += chunk.len() as u64;
                saved.peer_height = saved.peer_height.max(saved.cursor);
                saved.accumulated_headers.extend(chunk);
                saved
            }
            None if links_to_local => {
                let cursor = first_height + chunk.len() as u64;
                SyncStateBackup { cursor, peer_height: cursor, accumulated_headers: chunk }
            }
            None => return None,
        };
        if !Self::write(path, &backup) {
            return None;
        }
        meta.cursor = Some(backup.cursor);
        meta.known = true;
        Some(backup.cursor)
    }

    #[cfg(test)]
    pub(crate) fn generation(&self) -> u64 {
        self.lock().generation
    }
}

/// Counts block checks running on the blocking pool. The guard is moved into
/// the check, so the count drops when it ends, even if it panics.
#[derive(Default, Clone)]
pub(crate) struct ChecksRunning(Arc<AtomicUsize>);

pub(crate) struct CheckGuard(Arc<AtomicUsize>);

impl ChecksRunning {
    pub fn start(&self) -> CheckGuard {
        self.0.fetch_add(1, Ordering::SeqCst);
        CheckGuard(self.0.clone())
    }

    pub fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl Drop for CheckGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Manager for the non-blocking sync session state machine.
/// Extracted to shrink the Node god module. Node holds one and delegates
/// session work here. For simplicity (per plan), heavy callbacks (apply/reorg/
/// storage/network) remain driven from Node methods that call into this.
#[derive(Default)]
pub struct SyncManager {
    pub(crate) session: Option<SyncSession>,
    // Additional per-session or cross-session state (in_flight maps etc) live
    // inside the SyncPhase variants for now; can be lifted if needed.
    pub last_sync_cursor: Option<u64>,
    pub(crate) retry_count: u32,
    pub(crate) backoff_until: Option<Instant>,
    pub in_progress: bool,
    // Rate limiting state for header/batch requests can be co-located here later.
    /// The headers saved for resuming (`sync_state.bin`).
    pub(crate) saved: SavedHeaders,
    /// Block checks (`handle_sync_batches`) still running.
    pub(crate) batch_checks: ChecksRunning,
    /// GetBatches sent by sessions that have ended: per peer, the requested
    /// start height and when its session ended. A late reply to one is not a
    /// competing chain, just blocks nobody is waiting for.
    abandoned: HashMap<PeerId, Vec<(u64, Instant)>>,
    /// Peers whose headers broke our consensus rules, with the work they
    /// claimed then.
    incompatible: HashMap<PeerId, (u128, Instant)>,
    /// The peer whose session already restarted once without the saved
    /// headers, after they didn't fit its chain. Reset by every new session.
    restarted_fresh: Option<PeerId>,
}

impl SyncManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_session(&self) -> bool {
        self.session.is_some()
    }

    pub fn abort(&mut self, reason: &str) -> (u32, Option<Instant>) {
        self.in_progress = false;
        if self.session.is_some() {
            self.retry_count += 1;
            // Exponential backoff: 2^n seconds, capped at 5 minutes.
            // Sequence: 2s, 4s, 8s, 16s, 32s, 64s, 128s, 256s, 300s, 300s...
            let backoff_secs = 2u64.saturating_pow(self.retry_count).min(300);
            let backoff = std::time::Instant::now() + std::time::Duration::from_secs(backoff_secs);
            self.backoff_until = Some(backoff);
            tracing::warn!(
                "Aborting sync session (attempt {}): {}. Retrying in {}s.",
                self.retry_count, reason, backoff_secs
            );
            if let Some(s) = self.session.take() {
                tracing::info!("Aborted sync session with {} ({}): {}", s.peer, s.phase_name(), reason);
                self.abandon_requests(&s);
            }
            (self.retry_count, Some(backoff))
        } else {
            (self.retry_count, None)
        }
    }

    // Simple delegation helpers (to be expanded). For phase 3 we start with
    // thin wrappers; full state machine methods can move body here over time.
    #[allow(dead_code)]
    pub(crate) fn session_mut(&mut self) -> Option<&mut SyncSession> {
        self.session.as_mut()
    }

    pub fn start(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, start_height: u64, recovered_headers: Vec<BatchHeader>) {
        // Now owns the session creation logic (moved from node during Phase 3).
        if let Some(old) = self.session.take() {
            self.abandon_requests(&old);
        }
        self.restarted_fresh = None;
        let now = Instant::now();
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::Headers {
                accumulated: recovered_headers,
                cursor: start_height,
                snapshot: None,
                verifying: false,
            },
            started_at: now,
            last_progress_at: now,
        });
        self.in_progress = true;
        self.last_sync_cursor = Some(start_height);
    }

    pub fn fire_batch_lookahead(&mut self, network: &mut crate::network::MidstateNetwork) {
        let (cursor, peer_height, in_flight_len, primary_peer) = match &self.session {
            Some(s) => match &s.phase {
                SyncPhase::Batches { cursor, in_flight, .. } => {
                    (*cursor, s.peer_height, in_flight.len(), s.peer)
                }
                _ => return,
            },
            None => return,
        };

        let slots_available = BATCH_LOOKAHEAD.saturating_sub(in_flight_len);
        if slots_available == 0 { return; }

        let mut next_start = cursor;
        let mut reqs_sent = 0;

        while reqs_sent < slots_available && next_start < peer_height {
            let is_in_flight = match &self.session {
                Some(s) => match &s.phase {
                    SyncPhase::Batches { in_flight, .. } => in_flight.contains_key(&next_start),
                    _ => false,
                },
                None => false,
            };

            let is_prefetched = match &self.session {
                Some(s) => match &s.phase {
                    SyncPhase::Batches { prefetch_buffer, .. } => prefetch_buffer.contains_key(&next_start),
                    _ => false,
                },
                None => false,
            };

            if !is_in_flight && !is_prefetched {
                let target_peer = primary_peer; // ALWAYS use primary peer to avoid cross-fork corruption

                let count = (peer_height - next_start).min(crate::network::MAX_GETBATCHES_COUNT);
                tracing::debug!("Lookahead: pipelining batches {}..{} from primary peer {}", next_start, next_start + count, target_peer);
                network.send(target_peer, crate::network::Message::GetBatches { start_height: next_start, count });

                if let Some(s) = &mut self.session {
                    if let SyncPhase::Batches { in_flight, .. } = &mut s.phase {
                        in_flight.insert(next_start, target_peer);
                    }
                }
                reqs_sent += 1;
            }

            next_start += crate::network::MAX_GETBATCHES_COUNT;
        }
    }

    pub fn take_prefetch_for_cursor(&mut self, cursor: u64) -> Option<Vec<Batch>> {
        match &mut self.session {
            Some(s) => match &mut s.phase {
                SyncPhase::Batches { prefetch_buffer, .. } => prefetch_buffer.remove(&cursor),
                _ => None,
            },
            None => None,
        }
    }

    pub fn transition_to_verifying_headers(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, started_at: Instant) {
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::VerifyingHeaders,
            started_at,
            last_progress_at: Instant::now(),
        });
    }

    pub fn restart_headers_with_step_back(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, headers: Vec<BatchHeader>, new_start: u64, started_at: Instant) {
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::Headers {
                accumulated: headers,
                cursor: new_start,
                snapshot: None,
                verifying: false,
            },
            started_at,
            last_progress_at: Instant::now(),
        });
    }

    pub fn set_pipelined_rebuild(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, headers: Vec<BatchHeader>, fork_height: u64, is_fast_forward: bool, buffered_batches: Vec<Batch>, in_flight: std::collections::BTreeMap<u64, PeerId>, started_at: Instant) {
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::PipelinedRebuild {
                headers,
                fork_height,
                is_fast_forward,
                buffered_batches,
                in_flight,
            },
            started_at,
            last_progress_at: Instant::now(),
        });
    }

    pub fn set_batches_phase(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, headers: Vec<BatchHeader>, fork_height: u64, candidate_state: State, cursor: u64, new_history: Vec<(u64, [u8; 32], Batch)>, is_fast_forward: bool, in_flight: std::collections::BTreeMap<u64, PeerId>, prefetch_buffer: std::collections::BTreeMap<u64, Vec<Batch>>, started_at: Instant) {
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::Batches {
                headers,
                fork_height,
                candidate_state,
                cursor,
                new_history,
                is_fast_forward,
                in_flight,
                prefetch_buffer,
            },
            started_at,
            last_progress_at: Instant::now(),
        });
    }

    pub fn transition_to_verifying_batches(&mut self, peer: PeerId, peer_height: u64, peer_depth: u128, in_flight: std::collections::BTreeMap<u64, PeerId>, prefetch_buffer: std::collections::BTreeMap<u64, Vec<Batch>>, started_at: Instant) {
        self.session = Some(SyncSession {
            peer,
            peer_height,
            peer_depth,
            phase: SyncPhase::VerifyingBatches {
                in_flight,
                prefetch_buffer,
            },
            started_at,
            last_progress_at: Instant::now(),
        });
    }

    pub fn save_backup(&self, data_dir: &std::path::PathBuf, cursor: u64, peer_height: u64) {
        if let Some(s) = &self.session {
            if let SyncPhase::Headers { accumulated, .. } = &s.phase {
                self.saved.store(data_dir, SyncStateBackup {
                    cursor,
                    peer_height,
                    accumulated_headers: accumulated.clone(),
                });
            }
        }
    }

    pub fn update_progress(&mut self, new_cursor: u64) {
        if let Some(s) = &mut self.session {
            s.last_progress_at = std::time::Instant::now();
        }
        self.last_sync_cursor = Some(new_cursor);
    }

    pub fn set_last_sync_cursor(&mut self, cursor: Option<u64>) {
        self.last_sync_cursor = cursor;
    }

    pub fn set_last_progress_now(&mut self) {
        if let Some(s) = &mut self.session {
            s.last_progress_at = std::time::Instant::now();
        }
    }

    pub(crate) fn take_session(&mut self) -> Option<SyncSession> {
        self.session.take()
    }

    pub(crate) fn set_session(&mut self, s: SyncSession) {
        self.session = Some(s);
    }

    pub fn has_active_session(&self) -> bool {
        self.session.is_some()
    }

    pub fn is_in_progress(&self) -> bool {
        self.in_progress
    }

    pub fn get_session_peer(&self) -> Option<PeerId> {
        self.session.as_ref().map(|s| s.peer)
    }

    pub fn is_sync_peer(&self, p: PeerId) -> bool {
        self.session.as_ref().map_or(false, |s| s.peer == p)
    }

    pub fn finish_sync(&mut self) {
        if let Some(s) = self.session.take() {
            self.abandon_requests(&s);
        }
        self.in_progress = false;
    }

    /// Whether `peer` and `started_at` name the session running now.
    ///
    /// # Reasoning
    /// Header, block and state-rebuild checks run on the blocking pool and
    /// report back through the command channel with the peer and start time
    /// of the session that asked for them. A session's peer can drop while
    /// its check runs; the session is aborted and another starts, sometimes
    /// with the same peer. Matching on the peer alone let such a leftover
    /// result take over the new session: it re-armed it for the old peer,
    /// which was usually gone, and the request failed and aborted it.
    pub fn is_session(&self, peer: PeerId, started_at: Instant) -> bool {
        self.session.as_ref().map_or(false, |s| s.peer == peer && s.started_at == started_at)
    }

    pub fn session_started_at(&self) -> Option<Instant> {
        self.session.as_ref().map(|s| s.started_at)
    }

    /// The cursor the Headers phase is waiting on, if the session is with
    /// `peer`, in that phase, and has a request out (no chunk being checked).
    pub fn awaited_header_start(&self, peer: PeerId) -> Option<u64> {
        match &self.session {
            Some(s) if s.peer == peer => match &s.phase {
                SyncPhase::Headers { cursor, verifying: false, .. } => Some(*cursor),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether the Headers phase with `peer` is mid-check.
    pub fn is_checking_headers(&self, peer: PeerId) -> bool {
        matches!(&self.session, Some(s) if s.peer == peer
            && matches!(s.phase, SyncPhase::Headers { verifying: true, .. }))
    }

    fn abandon_requests(&mut self, s: &SyncSession) {
        match &s.phase {
            SyncPhase::Batches { in_flight, .. }
            | SyncPhase::VerifyingBatches { in_flight, .. }
            | SyncPhase::PipelinedRebuild { in_flight, .. } => {
                let in_flight = in_flight.clone();
                self.abandon_in_flight(&in_flight);
            }
            _ => {}
        }
    }

    /// Remember the GetBatches in `in_flight` as sent by a session that has
    /// ended, so their replies are recognised when they come.
    pub(crate) fn abandon_in_flight(&mut self, in_flight: &BTreeMap<u64, PeerId>) {
        let now = Instant::now();
        self.prune_abandoned(now);
        for (&start, &peer) in in_flight {
            self.abandoned.entry(peer).or_default().push((start, now));
        }
    }

    fn prune_abandoned(&mut self, now: Instant) {
        self.abandoned.retain(|_, requests| {
            requests.retain(|&(_, at)| now.duration_since(at) < ABANDONED_REPLY_WINDOW);
            !requests.is_empty()
        });
    }

    /// Whether batches from `peer` starting at `start` answer a GetBatches
    /// whose session has ended. A match is used up.
    pub fn take_abandoned_reply(&mut self, peer: PeerId, start: u64) -> bool {
        self.prune_abandoned(Instant::now());
        let Some(requests) = self.abandoned.get_mut(&peer) else { return false };
        let count = crate::network::MAX_GETBATCHES_COUNT;
        let Some(i) = requests.iter().position(|&(req, _)| start >= req && start < req.saturating_add(count)) else {
            return false;
        };
        requests.swap_remove(i);
        if requests.is_empty() {
            self.abandoned.remove(&peer);
        }
        true
    }

    /// Forget any ended session's request to `peer` that a reply starting at
    /// `start` would match: a new request outside any session wants that reply.
    pub fn forget_abandoned(&mut self, peer: PeerId, start: u64) {
        while self.take_abandoned_reply(peer, start) {}
    }

    /// Remember that `peer`'s headers broke our consensus rules while it
    /// claimed `depth`.
    pub fn mark_incompatible(&mut self, peer: PeerId, depth: u128) {
        self.incompatible.insert(peer, (depth, Instant::now()));
    }

    /// Whether to leave `peer`, now claiming `depth`, alone: its headers broke
    /// our rules and it has claimed no more work since. A peer stuck on a
    /// chain from before a fork would otherwise be synced from, and rejected,
    /// every time it reconnected.
    pub fn is_incompatible(&mut self, peer: PeerId, depth: u128) -> bool {
        match self.incompatible.get(&peer) {
            Some(&(claimed, at)) if depth <= claimed && at.elapsed() < INCOMPATIBLE_FOR => true,
            Some(_) => {
                self.incompatible.remove(&peer);
                false
            }
            None => false,
        }
    }

    /// Mark the session as restarted without the saved headers. False if it
    /// already was, so a session can't loop on it.
    pub fn note_fresh_restart(&mut self) -> bool {
        let Some(peer) = self.session.as_ref().map(|s| s.peer) else { return false };
        if self.restarted_fresh == Some(peer) {
            return false;
        }
        self.restarted_fresh = Some(peer);
        true
    }

    /// Blocks were saved: whatever went wrong before, sync works. Start the
    /// next retry without the backoff the earlier failures built up.
    pub fn note_progress(&mut self) {
        self.retry_count = 0;
        self.backoff_until = None;
    }

    pub fn get_sync_role_for_peer(&self, from: PeerId) -> Option<&'static str> {
        self.session.as_ref().and_then(|s| {
            match &s.phase {
                SyncPhase::Batches { in_flight, .. } => {
                    if s.peer == from || in_flight.iter().any(|(_, p)| *p == from) {
                        Some("batches")
                    } else { None }
                }
                SyncPhase::VerifyingBatches { in_flight, .. } => {
                    if s.peer == from || in_flight.iter().any(|(_, p)| *p == from) {
                        Some("verifying")
                    } else { None }
                }
                SyncPhase::PipelinedRebuild { .. } if s.peer == from => Some("pipeline"),
                _ => None,
            }
        })
    }

    pub fn get_effective_cursor(&self) -> u64 {
        self.session.as_ref().map_or(0, |s| {
            match &s.phase {
                SyncPhase::Batches { cursor, .. } => *cursor,
                SyncPhase::VerifyingBatches { .. } => u64::MAX, // Force buffer
                _ => 0,
            }
        })
    }

    pub fn get_current_cursor(&self) -> Option<u64> {
        self.session.as_ref().and_then(|s| match &s.phase {
            SyncPhase::Batches { cursor, .. } => Some(*cursor),
            _ => None,
        })
    }

    /// For handle_sync_headers: extract phase info for the peer, take snapshot.
    pub fn prepare_header_chunk(&mut self, from: PeerId) -> Option<(u64, u128, u64, Option<Box<State>>)> {
        if let Some(s) = &mut self.session {
            if s.peer == from {
                if let SyncPhase::Headers { cursor, snapshot, .. } = &mut s.phase {
                    let ph = s.peer_height;
                    let pd = s.peer_depth;
                    let c = *cursor;
                    let snap = snapshot.take();
                    return Some((ph, pd, c, snap));
                }
            }
        }
        None
    }

    /// Put snapshot back after use.
    pub fn restore_header_snapshot(&mut self, snapshot: Option<Box<State>>) {
        if let Some(s) = &mut self.session {
            if let SyncPhase::Headers { snapshot: snap_ref, .. } = &mut s.phase {
                *snap_ref = snapshot;
            }
        }
    }

    /// Mark verifying flag for stall monitor.
    pub fn set_header_verifying(&mut self, from: PeerId, verifying: bool) {
        if let Some(s) = &mut self.session {
            if s.peer == from {
                if let SyncPhase::Headers { verifying: v, .. } = &mut s.phase {
                    *v = verifying;
                }
            }
        }
    }

    /// Get info for verified headers chunk processing.
    pub fn get_verified_header_info(&mut self, from: PeerId) -> Option<(u64, u64)> {
        if let Some(s) = &mut self.session {
            if s.peer == from {
                if let SyncPhase::Headers { cursor, verifying, .. } = &mut s.phase {
                    *verifying = false;
                    return Some((s.peer_height, *cursor));
                }
            }
        }
        None
    }

    pub fn clear_backup(&self, data_dir: &std::path::PathBuf) {
        self.saved.clear(data_dir);
    }

    pub fn take_headers_for_verification(&mut self) -> Option<(Vec<BatchHeader>, Option<Box<State>>, PeerId, u64, u128, Instant)> {
        if let Some(session) = self.session.take() {
            if let SyncPhase::Headers { accumulated, snapshot, .. } = session.phase {
                return Some((accumulated, snapshot, session.peer, session.peer_height, session.peer_depth, session.started_at));
            }
            // put back if not headers phase
            self.session = Some(session);
        }
        None
    }

    /// Check if the current sync session has stalled (no progress beyond timeout).
    /// Returns Some(msg) if should abort, None otherwise.
    /// Also logs stall warnings.
    /// Does not abort itself (delegation).
    pub fn check_for_stall(&self) -> Option<String> {
        if let Some(session) = &self.session {
            if matches!(
                session.phase,
                SyncPhase::VerifyingHeaders
                | SyncPhase::VerifyingBatches { .. }
                | SyncPhase::PipelinedRebuild { .. }
                | SyncPhase::Headers { verifying: true, .. }
            ) {
                return None;
            }
            let idle_secs = session.last_progress_at.elapsed().as_secs();
            let has_made_progress = session.last_progress_at != session.started_at;
            let timeout = if has_made_progress { SYNC_TIMEOUT_SECS } else { SYNC_INITIAL_TIMEOUT_SECS };
            let peer = session.peer;
            let phase_name = match &session.phase {
                SyncPhase::Headers { cursor, .. } => format!("Headers(cursor={})", cursor),
                SyncPhase::Batches { cursor, .. } => format!("Batches(cursor={})", cursor),
                _ => "Verifying".into(),
            };
            if idle_secs > timeout {
                let msg = format!("timed out after {}s in phase {}", idle_secs, phase_name);
                return Some(msg);
            } else if idle_secs > timeout / 2 && idle_secs % 30 < 5 {
                tracing::warn!(
                    "Sync stall warning: {}s idle in {} (timeout={}s, peer={})",
                    idle_secs, phase_name, timeout, peer
                );
            }
        }
        None
    }

    /// Accumulate verified headers into the current Headers phase session.
    /// Handles OOM DoS check, deep fork prepend, link validation.
    /// Returns the new_cursor on success.
    /// On fatal link issues, clears backup and aborts, returns Err so caller can short-circuit.
    pub fn accumulate_verified_headers(&mut self, from: PeerId, headers: Vec<BatchHeader>, peer_height: u64, data_dir: &std::path::PathBuf) -> Result<u64> {
        let (cursor, _ph) = match &mut self.session {
            Some(s) if s.peer == from => {
                match &mut s.phase {
                    SyncPhase::Headers { cursor, .. } => (*cursor, s.peer_height),
                    _ => return Ok(0),
                }
            }
            _ => return Ok(0),
        };

        if headers.is_empty() {
            return Ok(cursor);
        }

        let mut new_cursor = cursor + headers.len() as u64;

        match &mut self.session {
            Some(s) => {
                if let SyncPhase::Headers { accumulated, cursor: c, .. } = &mut s.phase {
                    if accumulated.len() + headers.len() > 10_000_000 {
                        self.abort("Peer attempted OOM DoS with too many headers");
                        return Err(anyhow::anyhow!("aborted"));
                    }

                    // Prepend backward chunks for deep forks
                    if !accumulated.is_empty() && headers.last().map(|h| h.height) < accumulated.first().map(|h| h.height) {
                        if let (Some(last_new), Some(first_acc)) = (headers.last(), accumulated.first()) {
                            if first_acc.prev_header_hash != last_new.extension.final_hash || first_acc.prev_midstate != last_new.post_tx_midstate {
                                self.clear_backup(data_dir);
                                self.set_last_sync_cursor(None);
                                self.abort("sync_state.bin deep-fork mismatch");
                                return Err(anyhow::anyhow!("aborted"));
                            }
                        }

                        let mut new_acc = headers.clone();
                        new_acc.append(accumulated);
                        *accumulated = new_acc;
                        new_cursor = peer_height;
                        *c = new_cursor;
                    } else {
                        // Normal forward
                        if let (Some(last_acc), Some(first_new)) = (accumulated.last(), headers.first()) {
                            if first_new.prev_header_hash != last_acc.extension.final_hash || first_new.prev_midstate != last_acc.post_tx_midstate {
                                self.clear_backup(data_dir);
                                self.set_last_sync_cursor(None);
                                self.abort("sync_state.bin fork mismatch");
                                return Err(anyhow::anyhow!("aborted"));
                            }
                        }

                        accumulated.extend(headers);
                        *c = new_cursor;
                    }

                    self.save_backup(data_dir, new_cursor, peer_height);
                }
            }
            _ => unreachable!(),
        }

        Ok(new_cursor)
    }

    /// The saved headers to resume a session with `peer_height` from, and
    /// the height to resume at.
    ///
    /// A peer ahead of the saved headers continues them. A peer at or behind
    /// their end (another peer's headers got us there) resumes one header
    /// short of its own tip: it sends that one header, which has to link to
    /// ours, and the session goes on to its blocks. Either way the file is
    /// left alone; it used to be deleted here, which threw away every saved
    /// header whenever a peer at the same height, or one stuck on a dead fork
    /// further down, came along.
    pub fn load_backup(&mut self, data_dir: &std::path::PathBuf, peer_height: u64) -> (Vec<BatchHeader>, Option<u64>) {
        let Some(backup) = self.saved.load(data_dir) else {
            return (Vec::new(), None);
        };
        if peer_height > backup.cursor {
            return (backup.accumulated_headers, Some(backup.cursor));
        }
        let resume = peer_height.saturating_sub(1);
        let mut headers = backup.accumulated_headers;
        headers.retain(|h| h.height < resume);
        match headers.last() {
            Some(last) if last.height + 1 == resume => (headers, Some(resume)),
            _ => (Vec::new(), None),
        }
    }

    /// Helper to know the current phase name for logging (avoids exposing full enum everywhere).
    pub fn phase_name(&self) -> &'static str {
        match &self.session {
            Some(s) => s.phase_name(),
            None => "None",
        }
    }
}

impl SyncSession {
    pub fn phase_name(&self) -> &'static str {
        match &self.phase {
            SyncPhase::Headers { .. } => "Headers",
            SyncPhase::VerifyingHeaders => "VerifyingHeaders",
            SyncPhase::Batches { .. } => "Batches",
            SyncPhase::VerifyingBatches { .. } => "VerifyingBatches",
            SyncPhase::PipelinedRebuild { .. } => "PipelinedRebuild",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Extension;
    use tempfile::tempdir;

    /// `n` linked headers from `start`. Different seeds give different chains.
    fn chain(start: u64, n: u64, seed: u8) -> Vec<BatchHeader> {
        let mut prev_header_hash = [seed; 32];
        let mut prev_midstate = [seed.wrapping_add(1); 32];
        (start..start + n)
            .map(|height| {
                let tag = [&height.to_le_bytes()[..], &[seed][..]].concat();
                let final_hash = *blake3::hash(&[&tag[..], b"hash"].concat()).as_bytes();
                let post_tx_midstate = *blake3::hash(&[&tag[..], b"mid"].concat()).as_bytes();
                let header = BatchHeader {
                    height,
                    prev_midstate,
                    post_tx_midstate,
                    extension: Extension { nonce: 0, final_hash },
                    timestamp: height,
                    target: [0xff; 32],
                    state_root: [0; 32],
                    prev_header_hash,
                };
                prev_header_hash = final_hash;
                prev_midstate = post_tx_midstate;
                header
            })
            .collect()
    }

    fn backup_of(headers: &[BatchHeader]) -> SyncStateBackup {
        let cursor = headers.last().map_or(0, |h| h.height + 1);
        SyncStateBackup { cursor, peer_height: cursor, accumulated_headers: headers.to_vec() }
    }

    fn save(saved: &SavedHeaders, dir: &Path, headers: &[BatchHeader]) -> bool {
        saved.store_now(&SavedHeaders::file(dir), &backup_of(headers), saved.generation())
    }

    #[test]
    fn saved_headers_are_never_replaced_by_ones_that_end_lower() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let all = chain(100, 50, 1);

        assert!(save(&saved, dir.path(), &all[..40]));
        // A session with a peer that's behind, or on a fork of its own, has less.
        assert!(!save(&saved, dir.path(), &all[..30]));
        assert!(!save(&saved, dir.path(), &chain(100, 20, 2)));
        assert_eq!(saved.cursor(dir.path()), Some(140));
        // The same height (a different tip) and further are fine.
        assert!(save(&saved, dir.path(), &chain(100, 40, 3)));
        assert!(save(&saved, dir.path(), &all));
        assert_eq!(SavedHeaders::default().load(dir.path()).unwrap().cursor, 150);
    }

    #[test]
    fn the_rule_holds_for_a_file_an_earlier_run_left() {
        let dir = tempdir().unwrap();
        let all = chain(100, 50, 1);
        assert!(save(&SavedHeaders::default(), dir.path(), &all));

        // A new process that hasn't looked at the file yet.
        let saved = SavedHeaders::default();
        assert!(!save(&saved, dir.path(), &all[..10]));
        assert_eq!(saved.load(dir.path()).unwrap().cursor, 150);
    }

    #[test]
    fn a_write_queued_before_a_clear_is_dropped() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let queued_at = saved.generation();
        saved.clear(dir.path());
        assert!(!saved.store_now(&SavedHeaders::file(dir.path()), &backup_of(&chain(0, 10, 1)), queued_at));
        assert!(saved.load(dir.path()).is_none());
    }

    #[test]
    fn a_checked_chunk_joins_the_saved_headers_only_where_it_continues_them() {
        let dir = tempdir().unwrap();
        let path = SavedHeaders::file(dir.path());
        let saved = SavedHeaders::default();
        let all = chain(100, 60, 1);
        assert!(save(&saved, dir.path(), &all[..20]));
        let generation = saved.generation();

        // Not where the saved headers end, or from another chain: dropped.
        assert_eq!(saved.keep_chunk_now(&path, all[25..40].to_vec(), false, generation), None);
        assert_eq!(saved.keep_chunk_now(&path, chain(120, 10, 2), false, generation), None);
        // The very next headers: kept.
        assert_eq!(saved.keep_chunk_now(&path, all[20..40].to_vec(), false, generation), Some(140));

        let kept = saved.load(dir.path()).unwrap();
        assert_eq!(kept.cursor, 140);
        assert_eq!(kept.accumulated_headers.len(), 40);
        assert_eq!(kept.accumulated_headers.last().unwrap().height, 139);
    }

    #[test]
    fn with_nothing_saved_a_checked_chunk_is_kept_only_if_it_continues_our_chain() {
        let dir = tempdir().unwrap();
        let path = SavedHeaders::file(dir.path());
        let saved = SavedHeaders::default();
        let headers = chain(500, 10, 1);

        assert_eq!(saved.keep_chunk_now(&path, headers.clone(), false, saved.generation()), None);
        assert!(saved.load(dir.path()).is_none());
        assert_eq!(saved.keep_chunk_now(&path, headers, true, saved.generation()), Some(510));
    }

    #[test]
    fn a_peer_ahead_of_the_saved_headers_continues_them() {
        let dir = tempdir().unwrap();
        let mut sync = SyncManager::default();
        assert!(save(&sync.saved, dir.path(), &chain(100, 50, 1)));

        let (headers, cursor) = sync.load_backup(&dir.path().to_path_buf(), 200);
        assert_eq!(cursor, Some(150));
        assert_eq!(headers.len(), 50);
    }

    #[test]
    fn a_peer_at_or_behind_the_saved_headers_resumes_one_header_short_of_its_tip() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().to_path_buf();
        let mut sync = SyncManager::default();
        assert!(save(&sync.saved, dir.path(), &chain(100, 50, 1)));

        // At the same height: it re-sends its last header, which must link to ours.
        let (headers, cursor) = sync.load_backup(&data_dir, 150);
        assert_eq!(cursor, Some(149));
        assert_eq!(headers.last().unwrap().height, 148);

        // Behind: a peer lagging, or one stuck on a fork further down.
        let (headers, cursor) = sync.load_backup(&data_dir, 130);
        assert_eq!(cursor, Some(129));
        assert_eq!(headers.last().unwrap().height, 128);

        // Below where the saved headers start: nothing to resume from.
        let (headers, cursor) = sync.load_backup(&data_dir, 100);
        assert!(headers.is_empty());
        assert_eq!(cursor, None);

        // None of that touched the file.
        assert_eq!(SavedHeaders::default().load(dir.path()).unwrap().cursor, 150);
    }

    #[test]
    fn replies_to_an_ended_sessions_requests_are_recognised_once() {
        let mut sync = SyncManager::default();
        let (a, b) = (PeerId::random(), PeerId::random());
        let count = crate::network::MAX_GETBATCHES_COUNT;
        sync.abandon_in_flight(&BTreeMap::from([(1000, a), (1000 + count, a), (5000, b)]));

        assert!(sync.take_abandoned_reply(a, 1000));
        assert!(!sync.take_abandoned_reply(a, 1000), "a match is used up");
        // A reply can start part-way into the range asked for.
        assert!(sync.take_abandoned_reply(a, 1000 + count + 7));
        assert!(!sync.take_abandoned_reply(a, 5000), "asked of another peer");
        assert!(!sync.take_abandoned_reply(PeerId::random(), 5000));
        assert!(sync.take_abandoned_reply(b, 5000));
    }

    #[test]
    fn a_new_request_outside_any_session_wants_its_reply() {
        let mut sync = SyncManager::default();
        let peer = PeerId::random();
        sync.abandon_in_flight(&BTreeMap::from([(1000, peer)]));
        sync.forget_abandoned(peer, 1200);
        assert!(!sync.take_abandoned_reply(peer, 1200));
    }

    #[test]
    fn a_peer_whose_headers_broke_the_rules_is_left_alone_until_it_claims_more_work() {
        let mut sync = SyncManager::default();
        let stuck = PeerId::random();
        sync.mark_incompatible(stuck, 49_337_247_483_311);

        assert!(sync.is_incompatible(stuck, 49_337_247_483_311));
        assert!(sync.is_incompatible(stuck, 49_000_000_000_000));
        assert!(!sync.is_incompatible(PeerId::random(), 1));
        assert!(!sync.is_incompatible(stuck, 49_337_247_483_312));
        // Claiming more work lifted the mark for good.
        assert!(!sync.is_incompatible(stuck, 1));
    }

    #[test]
    fn a_result_belongs_to_the_session_that_asked_for_it() {
        let mut sync = SyncManager::default();
        let peer = PeerId::random();
        sync.start(peer, 200, 1, 100, Vec::new());
        let first = sync.session_started_at().unwrap();
        assert!(sync.is_session(peer, first));
        assert!(!sync.is_session(PeerId::random(), first));

        // The peer drops and is straight back: a new session with the same peer.
        sync.abort("sync peer disconnected");
        std::thread::sleep(Duration::from_millis(2));
        sync.start(peer, 200, 1, 100, Vec::new());
        assert!(!sync.is_session(peer, first));
        assert!(sync.is_session(peer, sync.session_started_at().unwrap()));
    }

    #[test]
    fn a_session_drops_the_saved_headers_only_once() {
        let mut sync = SyncManager::default();
        let peer = PeerId::random();
        sync.start(peer, 200, 1, 100, Vec::new());
        assert!(sync.note_fresh_restart());
        assert!(!sync.note_fresh_restart());

        sync.start(peer, 201, 1, 100, Vec::new());
        assert!(sync.note_fresh_restart(), "a new session may again");
    }

    #[test]
    fn the_headers_phase_only_takes_the_reply_it_is_waiting_for() {
        let mut sync = SyncManager::default();
        let peer = PeerId::random();
        sync.start(peer, 200, 1, 100, Vec::new());
        assert_eq!(sync.awaited_header_start(peer), Some(100));
        assert_eq!(sync.awaited_header_start(PeerId::random()), None);
        assert!(!sync.is_checking_headers(peer));

        sync.set_header_verifying(peer, true);
        assert!(sync.is_checking_headers(peer));
        assert_eq!(sync.awaited_header_start(peer), None, "nothing is awaited mid-check");
    }
}
