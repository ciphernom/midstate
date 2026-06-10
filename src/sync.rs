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
use std::collections::BTreeMap;
use std::time::Instant;

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
                let backup = SyncStateBackup {
                    cursor,
                    peer_height,
                    accumulated_headers: accumulated.clone(),
                };
                let path = data_dir.join("sync_state.bin");
                tokio::task::spawn_blocking(move || {
                    if let Ok(bytes) = bincode::serialize(&backup) {
                        let _ = std::fs::write(&path, bytes);
                    }
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
        self.session = None;
        self.in_progress = false;
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
        let _ = std::fs::remove_file(data_dir.join("sync_state.bin"));
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

    pub fn load_backup(&mut self, data_dir: &std::path::PathBuf, peer_height: u64) -> (Vec<BatchHeader>, Option<u64>) {
        let sync_file_path = data_dir.join("sync_state.bin");
        let mut recovered_headers = Vec::new();
        let mut recovered_cursor = None;

        if let Ok(bytes) = std::fs::read(&sync_file_path) {
            if let Ok(backup) = bincode::deserialize::<SyncStateBackup>(&bytes) {
                if peer_height > backup.cursor {
                    recovered_headers = backup.accumulated_headers;
                    recovered_cursor = Some(backup.cursor);
                } else {
                    let _ = std::fs::remove_file(&sync_file_path);
                }
            }
        }
        (recovered_headers, recovered_cursor)
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
