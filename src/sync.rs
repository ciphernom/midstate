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

/// `sync_state.bin`: the PoW-checked headers sync sessions have gathered, so
/// a later session resumes where they left off instead of starting over.
///
/// Layout (integers little-endian): an 8-byte magic, the height of the first
/// header as a u64, then one `BatchHeader::to_record` per height. Fixed-size
/// records put any height at a known offset, so where the saved headers end,
/// and what the last of them is, take one small read however many there are.
///
/// # Reasoning
/// The file used to be rewritten whole, every header so far, after each
/// 5,000-header chunk: quadratic in the length of the sync. A new node
/// syncing ~380,000 headers wrote about 3 GB, and cloned its header list in
/// memory each time, to save 82 MB. Now a chunk that continues the file is
/// appended, and a different tip replaces just the records after the last
/// header both agree on. Whether they agree is one comparison: each header
/// names its parent's PoW hash, so headers that agree at one height agree
/// at every height below it. A whole-file rewrite is left for a chain that
/// doesn't share the file's, at most once a session.
///
/// Any peer used to be able to delete the file, and a session with a peer
/// behind it then replaced it with fewer headers. Only a clear (a finished
/// sync, a peer caught sending bad blocks) removes it now, and nothing
/// replaces it with headers that end lower.
///
/// Writes go, in order, to one writer thread: the main loop decides what to
/// write from what it has queued so far and never waits for the disk. The
/// writer re-checks each write against the file, and a load first waits for
/// it to finish what's queued. Appends aren't fsynced: a crash can at worst
/// cost the newest headers, and `load` keeps the longest run of records that
/// link up and drops the rest.
#[derive(Default, Clone)]
pub(crate) struct SavedHeaders(Arc<Mutex<SavedInner>>);

const SAVED_MAGIC: &[u8; 8] = b"MSHDRS\x00\x02";
const SAVED_PREFIX: u64 = 16;
const RECORD: u64 = crate::core::types::HEADER_RECORD_LEN as u64;

/// What the file holds, once every queued write has landed: heights
/// `base..cursor`, the last of them `last`.
#[derive(Clone)]
struct Extent {
    base: u64,
    cursor: u64,
    last: BatchHeader,
}

#[derive(Default)]
struct SavedInner {
    path: Option<PathBuf>,
    /// `None`: not looked at yet. `Some(None)`: no usable file.
    extent: Option<Option<Extent>>,
    writer: Option<std::sync::mpsc::Sender<WriteJob>>,
    /// Set by the writer when a write didn't match the file; the next
    /// decision then looks at the file again.
    stale: Arc<std::sync::atomic::AtomicBool>,
}

enum WriteJob {
    /// Cut the file to end at `start`, then append `headers` from there.
    Splice { path: PathBuf, start: u64, headers: Vec<BatchHeader> },
    /// Replace the file with `headers`.
    Rewrite { path: PathBuf, headers: Vec<BatchHeader> },
    Remove { path: PathBuf },
    /// Answer once every job queued before it has run.
    Barrier(std::sync::mpsc::SyncSender<()>),
}

impl SavedHeaders {
    fn file(data_dir: &Path) -> PathBuf {
        data_dir.join("sync_state.bin")
    }

    fn lock(&self) -> MutexGuard<'_, SavedInner> {
        self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The saved headers, if there are any.
    pub fn load(&self, data_dir: &Path) -> Option<SyncStateBackup> {
        self.flush();
        let path = Self::file(data_dir);
        let (headers, rewrite) = Self::read_all(&path);
        let mut inner = self.lock();
        inner.path = Some(path.clone());
        inner.extent = Some(Self::extent_of(&headers));
        if rewrite && !headers.is_empty() {
            // An old-format file, or one with a damaged tail: store it in
            // full in the current format, once.
            Self::enqueue(&mut inner, WriteJob::Rewrite { path, headers: headers.clone() });
        }
        drop(inner);
        let cursor = headers.last()?.height + 1;
        Some(SyncStateBackup { cursor, peer_height: cursor, accumulated_headers: headers })
    }

    /// Where the saved headers end, if there are any.
    #[cfg(test)]
    pub fn cursor(&self, data_dir: &Path) -> Option<u64> {
        let inner = self.lock_with_extent(&Self::file(data_dir));
        inner.extent.clone().flatten().map(|e| e.cursor)
    }

    pub fn clear(&self, data_dir: &Path) {
        let path = Self::file(data_dir);
        let mut inner = self.lock();
        inner.path = Some(path.clone());
        inner.extent = Some(None);
        Self::enqueue(&mut inner, WriteJob::Remove { path });
    }

    /// Save `accumulated`, a session's headers, the last `added` of them new.
    pub fn save(&self, data_dir: &Path, accumulated: &[BatchHeader], added: usize) {
        if added == 0 || accumulated.is_empty() {
            return;
        }
        let chunk_at = accumulated.len().saturating_sub(added);
        let (first, last) = (&accumulated[chunk_at], &accumulated[accumulated.len() - 1]);
        let (start, end) = (first.height, last.height + 1);
        let path = Self::file(data_dir);

        let mut inner = self.lock_with_extent(&path);
        let extent = inner.extent.clone().flatten();
        if extent.as_ref().map_or(false, |e| end < e.cursor) {
            return; // the file reaches further
        }
        let agrees = match &extent {
            Some(e) if start == e.cursor => e.last.is_followed_by(first),
            Some(e) if start > e.base && start < e.cursor => {
                // A different tip: the record below `start` is on disk once
                // the writer catches up.
                drop(inner);
                self.flush();
                let below = Self::read_record(&path, e.base, start - 1);
                inner = self.lock();
                below.map_or(false, |b| b.is_followed_by(first))
            }
            _ => false,
        };
        let base = match (&extent, agrees) {
            (Some(e), true) => {
                Self::enqueue(&mut inner, WriteJob::Splice { path, start, headers: accumulated[chunk_at..].to_vec() });
                e.base
            }
            _ => {
                Self::enqueue(&mut inner, WriteJob::Rewrite { path, headers: accumulated.to_vec() });
                accumulated[0].height
            }
        };
        inner.extent = Some(Some(Extent { base, cursor: end, last: last.clone() }));
    }

    /// Keep a header chunk whose PoW checked out after the session that asked
    /// for it had ended, most often because its peer dropped mid-check. It is
    /// added only where it continues the saved headers exactly, or, with
    /// nothing saved, where it continues our own chain (`links_to_local`).
    pub fn keep_chunk(&self, data_dir: &Path, chunk: Vec<BatchHeader>, links_to_local: bool) {
        let (Some(first), Some(last)) = (chunk.first(), chunk.last()) else { return };
        let (start, end, last) = (first.height, last.height + 1, last.clone());
        let path = Self::file(data_dir);
        let mut inner = self.lock_with_extent(&path);
        let base = match inner.extent.clone().flatten() {
            Some(e) if start == e.cursor && e.last.is_followed_by(first) => {
                Self::enqueue(&mut inner, WriteJob::Splice { path, start, headers: chunk });
                e.base
            }
            None if links_to_local => {
                Self::enqueue(&mut inner, WriteJob::Rewrite { path, headers: chunk });
                start
            }
            _ => {
                tracing::debug!("Headers {}..{} outlived their sync session and don't continue the saved ones; dropped", start, end - 1);
                return;
            }
        };
        inner.extent = Some(Some(Extent { base, cursor: end, last }));
        tracing::info!(
            "Kept headers {}..{}, verified after their sync session ended. The next sync resumes at {}.",
            start, end - 1, end
        );
    }

    /// Wait until every write queued so far has reached the file.
    pub fn flush(&self) {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let queued = {
            let mut inner = self.lock();
            inner.writer.is_some() && Self::enqueue(&mut inner, WriteJob::Barrier(tx))
        };
        if queued {
            let _ = rx.recv();
        }
    }

    /// The lock, with `extent` describing the file at `path`. When it has to
    /// be learnt (first use, or after a write went wrong) the file is read
    /// once the writer has finished what's queued, never mid-write: its first
    /// 16 bytes, its length and its last record.
    fn lock_with_extent(&self, path: &Path) -> MutexGuard<'_, SavedInner> {
        let needs_read = {
            let inner = self.lock();
            let stale = inner.stale.swap(false, Ordering::SeqCst);
            stale || inner.path.as_deref() != Some(path) || inner.extent.is_none()
        };
        if !needs_read {
            return self.lock();
        }
        self.flush();
        let extent = Self::read_extent(path);
        let mut inner = self.lock();
        inner.path = Some(path.to_path_buf());
        inner.extent = Some(extent);
        inner
    }

    fn read_extent(path: &Path) -> Option<Extent> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).ok()?;
        let len = f.metadata().ok()?.len();
        let mut prefix = [0u8; SAVED_PREFIX as usize];
        f.read_exact(&mut prefix).ok()?;
        if &prefix[..8] != SAVED_MAGIC {
            // The old format, which `load` converts.
            return Self::extent_of(&Self::read_all(path).0);
        }
        let base = u64::from_le_bytes(prefix[8..16].try_into().expect("8 bytes"));
        let count = len.saturating_sub(SAVED_PREFIX) / RECORD;
        let last = Self::read_record(path, base, base + count.checked_sub(1)?)?;
        Some(Extent { base, cursor: base + count, last })
    }

    fn read_record(path: &Path, base: u64, height: u64) -> Option<BatchHeader> {
        use std::io::{Read, Seek, SeekFrom};
        let index = height.checked_sub(base)?;
        let mut f = std::fs::File::open(path).ok()?;
        f.seek(SeekFrom::Start(SAVED_PREFIX + index * RECORD)).ok()?;
        let mut record = [0u8; crate::core::types::HEADER_RECORD_LEN];
        f.read_exact(&mut record).ok()?;
        let header = BatchHeader::from_record(&record);
        (header.height == height).then_some(header)
    }

    /// Every saved header, and whether the file should be stored afresh: it
    /// is in the old format, or its tail doesn't link up (only the run of
    /// records from the start that does is returned).
    fn read_all(path: &Path) -> (Vec<BatchHeader>, bool) {
        let Ok(bytes) = std::fs::read(path) else { return (Vec::new(), false) };
        if bytes.len() < SAVED_PREFIX as usize || &bytes[..8] != SAVED_MAGIC {
            return match bincode::deserialize::<SyncStateBackup>(&bytes) {
                Ok(old) => (Self::linked_prefix(old.accumulated_headers).0, true),
                Err(_) => (Vec::new(), false),
            };
        }
        let base = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        // `chunks_exact` drops a torn last record; what's before it is intact.
        let headers = bytes[SAVED_PREFIX as usize..]
            .chunks_exact(RECORD as usize)
            .map(|r| BatchHeader::from_record(r.try_into().expect("record length")))
            .collect::<Vec<_>>();
        if headers.first().map_or(false, |h| h.height != base) {
            return (Vec::new(), false);
        }
        Self::linked_prefix(headers)
    }

    /// The longest run from the start in which heights count up and each
    /// header follows the one before; and whether anything was cut.
    fn linked_prefix(mut headers: Vec<BatchHeader>) -> (Vec<BatchHeader>, bool) {
        let keep = headers
            .windows(2)
            .position(|w| w[1].height != w[0].height + 1 || !w[0].is_followed_by(&w[1]))
            .map_or(headers.len(), |i| i + 1);
        let cut = keep < headers.len();
        headers.truncate(keep);
        (headers, cut)
    }

    fn extent_of(headers: &[BatchHeader]) -> Option<Extent> {
        let (first, last) = (headers.first()?, headers.last()?);
        Some(Extent { base: first.height, cursor: last.height + 1, last: last.clone() })
    }

    /// Queue `job` for the writer thread, starting it if need be.
    fn enqueue(inner: &mut SavedInner, job: WriteJob) -> bool {
        if inner.writer.is_none() {
            let (tx, rx) = std::sync::mpsc::channel::<WriteJob>();
            let stale = inner.stale.clone();
            let spawned = std::thread::Builder::new()
                .name("saved-headers".into())
                .spawn(move || {
                    for job in rx {
                        if !Self::run(job) {
                            stale.store(true, Ordering::SeqCst);
                        }
                    }
                });
            if spawned.is_err() {
                return false;
            }
            inner.writer = Some(tx);
        }
        inner.writer.as_ref().map_or(false, |tx| tx.send(job).is_ok())
    }

    /// Apply one write. False when the file wasn't what the write expected.
    fn run(job: WriteJob) -> bool {
        match job {
            WriteJob::Barrier(done) => {
                let _ = done.send(());
                true
            }
            WriteJob::Remove { path } => {
                let _ = std::fs::remove_file(&path);
                true
            }
            WriteJob::Rewrite { path, headers } => {
                let Some(first) = headers.first() else { return true };
                let mut bytes = Vec::with_capacity(SAVED_PREFIX as usize + headers.len() * RECORD as usize);
                bytes.extend_from_slice(SAVED_MAGIC);
                bytes.extend_from_slice(&first.height.to_le_bytes());
                for h in &headers {
                    bytes.extend_from_slice(&h.to_record());
                }
                // Through a temporary file: a crash leaves the old file or the new one.
                let tmp = path.with_extension("bin.tmp");
                std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_ok()
            }
            WriteJob::Splice { path, start, headers } => {
                use std::io::{Seek, SeekFrom, Write};
                let Some(first) = headers.first() else { return true };
                let Some(extent) = Self::read_extent(&path) else { return false };
                if start <= extent.base || start > extent.cursor {
                    return false;
                }
                let below = if start == extent.cursor {
                    Some(extent.last)
                } else {
                    Self::read_record(&path, extent.base, start - 1)
                };
                if !below.map_or(false, |b| b.is_followed_by(first)) {
                    return false;
                }
                let mut bytes = Vec::with_capacity(headers.len() * RECORD as usize);
                for h in &headers {
                    bytes.extend_from_slice(&h.to_record());
                }
                let at = SAVED_PREFIX + (start - extent.base) * RECORD;
                let Ok(mut f) = std::fs::OpenOptions::new().write(true).open(&path) else { return false };
                f.set_len(at).is_ok()
                    && f.seek(SeekFrom::Start(at)).is_ok()
                    && f.write_all(&bytes).is_ok()
            }
        }
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
    /// Header checkpoints: segments of the chain whose PoW needn't be
    /// re-checked (see `crate::checkpoints`).
    pub(crate) checkpoints: crate::checkpoints::Checkpoints,
    /// How fast blocks are being verified, for the progress line.
    block_pace: Option<BlockPace>,
}

#[derive(Clone, Copy)]
struct BlockPace {
    at: Instant,
    height: u64,
    /// Blocks a second, smoothed; 0 until there are two readings.
    per_sec: f64,
}

/// ", ~24 min at 455/min": how long `to_go` blocks take at `per_sec`.
pub fn pace_text(to_go: u64, per_sec: f64) -> String {
    if per_sec <= 0.0 || to_go == 0 {
        return String::new();
    }
    let minutes = (to_go as f64 / per_sec / 60.0).ceil() as u64;
    let left = match minutes {
        0..=1 => "~1 min".to_string(),
        2..=89 => format!("~{} min", minutes),
        _ => format!("~{} h {} min", minutes / 60, minutes % 60),
    };
    format!(", {} at {:.0}/min", left, per_sec * 60.0)
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

    /// Save the session's headers, the last `added` of them new (all of them
    /// for `usize::MAX`). See `SavedHeaders::save`.
    pub fn save_backup(&self, data_dir: &std::path::PathBuf, added: usize) {
        if let Some(s) = &self.session {
            if let SyncPhase::Headers { accumulated, .. } = &s.phase {
                self.saved.save(data_dir, accumulated, added);
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

    /// The session's headers with heights in `from..to`, when it's fetching
    /// headers and has them.
    pub fn accumulated_between(&self, from: u64, to: u64) -> Vec<BatchHeader> {
        let Some(SyncSession { phase: SyncPhase::Headers { accumulated, .. }, .. }) = &self.session else {
            return Vec::new();
        };
        let Some(first) = accumulated.first().map(|h| h.height) else { return Vec::new() };
        let lo = from.saturating_sub(first).min(accumulated.len() as u64) as usize;
        let hi = to.saturating_sub(first).min(accumulated.len() as u64) as usize;
        accumulated[lo..hi.max(lo)].to_vec()
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

    /// Note that blocks are verified up to `height`, and return how fast
    /// they're coming, in blocks a second: an exponential moving average over
    /// the readings, O(1) to keep. A reading after a long pause, or a lower
    /// height (a new session from further back), starts the average afresh.
    pub fn block_rate(&mut self, height: u64) -> Option<f64> {
        const SMOOTHING: f64 = 0.3;
        const STALE: Duration = Duration::from_secs(600);
        let now = Instant::now();
        let per_sec = match self.block_pace {
            Some(p) if height == p.height => return (p.per_sec > 0.0).then_some(p.per_sec),
            Some(p) if height > p.height && now.duration_since(p.at) < STALE => {
                let latest = (height - p.height) as f64 / now.duration_since(p.at).as_secs_f64().max(1e-3);
                if p.per_sec > 0.0 { SMOOTHING * latest + (1.0 - SMOOTHING) * p.per_sec } else { latest }
            }
            _ => 0.0,
        };
        self.block_pace = Some(BlockPace { at: now, height, per_sec });
        (per_sec > 0.0).then_some(per_sec)
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
        let mut added = headers.len();

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
                        added = usize::MAX; // new headers at the front: save them all
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

                    self.save_backup(data_dir, added);
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

    /// Save all of `headers` as a session's, then wait for the writer.
    fn save_all(saved: &SavedHeaders, dir: &Path, headers: &[BatchHeader]) {
        saved.save(dir, headers, headers.len());
        saved.flush();
    }

    fn file_len(dir: &Path) -> u64 {
        std::fs::metadata(SavedHeaders::file(dir)).map(|m| m.len()).unwrap_or(0)
    }

    fn records(n: u64) -> u64 {
        SAVED_PREFIX + n * RECORD
    }

    #[test]
    fn a_header_record_round_trips_every_field() {
        let header = chain(371_499, 1, 9).remove(0);
        let back = BatchHeader::from_record(&header.to_record());
        assert_eq!(back.to_record(), header.to_record());
        assert_eq!(back.height, 371_499);
        assert_eq!(back.prev_midstate, header.prev_midstate);
        assert_eq!(back.extension.final_hash, header.extension.final_hash);
    }

    #[test]
    fn a_session_continuing_the_file_appends_only_its_new_headers() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let all = chain(100, 60, 1);

        save_all(&saved, dir.path(), &all[..20]);
        assert_eq!(file_len(dir.path()), records(20));

        // Next chunk: the session has 40 headers, 20 of them new.
        saved.save(dir.path(), &all[..40], 20);
        saved.save(dir.path(), &all[..60], 20);
        saved.flush();
        assert_eq!(file_len(dir.path()), records(60), "appended, not rewritten");
        let back = SavedHeaders::default().load(dir.path()).unwrap();
        assert_eq!(back.cursor, 160);
        assert_eq!(back.accumulated_headers.len(), 60);
        assert_eq!(back.accumulated_headers[59].to_record(), all[59].to_record());
    }

    #[test]
    fn saved_headers_are_never_replaced_by_ones_that_end_lower() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let all = chain(100, 50, 1);

        save_all(&saved, dir.path(), &all[..40]);
        // A session with a peer that's behind, or on a fork of its own, has less.
        save_all(&saved, dir.path(), &all[..30]);
        save_all(&saved, dir.path(), &chain(100, 20, 2));
        assert_eq!(saved.cursor(dir.path()), Some(140));
        // The same height on another chain, and further, are fine.
        save_all(&saved, dir.path(), &chain(100, 40, 3));
        assert_eq!(SavedHeaders::default().load(dir.path()).unwrap().accumulated_headers[0].to_record(), chain(100, 1, 3)[0].to_record());
        save_all(&saved, dir.path(), &all);
        assert_eq!(SavedHeaders::default().load(dir.path()).unwrap().cursor, 150);
    }

    #[test]
    fn a_different_tip_replaces_only_the_headers_after_the_fork() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let all = chain(100, 40, 1);
        save_all(&saved, dir.path(), &all);

        // A peer at the same height whose last header differs (a tip race):
        // the session resumed one header short of it and got the peer's.
        let mut theirs = all[..39].to_vec();
        let mut tip = chain(139, 1, 5).remove(0);
        tip.prev_header_hash = all[38].extension.final_hash;
        tip.prev_midstate = all[38].post_tx_midstate;
        theirs.push(tip.clone());
        saved.save(dir.path(), &theirs, 1);
        saved.flush();

        assert_eq!(file_len(dir.path()), records(40), "one record replaced in place");
        let back = SavedHeaders::default().load(dir.path()).unwrap();
        assert_eq!(back.accumulated_headers[39].to_record(), tip.to_record());
        assert_eq!(back.accumulated_headers[38].to_record(), all[38].to_record());
    }

    #[test]
    fn the_rule_holds_for_a_file_an_earlier_run_left() {
        let dir = tempdir().unwrap();
        let all = chain(100, 50, 1);
        save_all(&SavedHeaders::default(), dir.path(), &all);

        // A new process that hasn't looked at the file yet.
        let saved = SavedHeaders::default();
        save_all(&saved, dir.path(), &all[..10]);
        assert_eq!(saved.load(dir.path()).unwrap().cursor, 150);
    }

    #[test]
    fn a_clear_removes_what_was_queued_before_it() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        saved.save(dir.path(), &chain(0, 10, 1), 10);
        saved.clear(dir.path());
        saved.flush();
        assert!(saved.load(dir.path()).is_none());
        assert!(!SavedHeaders::file(dir.path()).exists());
    }

    #[test]
    fn a_file_in_the_old_format_is_read_and_converted() {
        let dir = tempdir().unwrap();
        let headers = chain(100, 30, 1);
        let old = SyncStateBackup { cursor: 130, peer_height: 140, accumulated_headers: headers.clone() };
        std::fs::write(SavedHeaders::file(dir.path()), bincode::serialize(&old).unwrap()).unwrap();

        let saved = SavedHeaders::default();
        let back = saved.load(dir.path()).unwrap();
        assert_eq!(back.cursor, 130);
        saved.flush();
        let bytes = std::fs::read(SavedHeaders::file(dir.path())).unwrap();
        assert_eq!(&bytes[..8], SAVED_MAGIC);
        assert_eq!(bytes.len() as u64, records(30));
    }

    #[test]
    fn a_damaged_tail_is_dropped_and_the_rest_kept() {
        let dir = tempdir().unwrap();
        let all = chain(100, 20, 1);
        save_all(&SavedHeaders::default(), dir.path(), &all);

        // A torn last record, then a crash-garbled one before it.
        let path = SavedHeaders::file(dir.path());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[7u8; 100]);
        let garbled = (SAVED_PREFIX + 18 * RECORD) as usize;
        bytes[garbled + 200] ^= 0xff; // record 18's prev_header_hash
        std::fs::write(&path, &bytes).unwrap();

        let back = SavedHeaders::default().load(dir.path()).unwrap();
        assert_eq!(back.cursor, 118, "records 0..18 link up; 18 and after are dropped");
    }

    #[test]
    fn a_checked_chunk_joins_the_saved_headers_only_where_it_continues_them() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let all = chain(100, 60, 1);
        save_all(&saved, dir.path(), &all[..20]);

        // Not where the saved headers end, or from another chain: dropped.
        saved.keep_chunk(dir.path(), all[25..40].to_vec(), false);
        saved.keep_chunk(dir.path(), chain(120, 10, 2), false);
        // The very next headers: kept.
        saved.keep_chunk(dir.path(), all[20..40].to_vec(), false);
        saved.flush();

        let kept = saved.load(dir.path()).unwrap();
        assert_eq!(kept.cursor, 140);
        assert_eq!(kept.accumulated_headers.len(), 40);
        assert_eq!(file_len(dir.path()), records(40));
    }

    #[test]
    fn with_nothing_saved_a_checked_chunk_is_kept_only_if_it_continues_our_chain() {
        let dir = tempdir().unwrap();
        let saved = SavedHeaders::default();
        let headers = chain(500, 10, 1);

        saved.keep_chunk(dir.path(), headers.clone(), false);
        saved.flush();
        assert!(saved.load(dir.path()).is_none());
        saved.keep_chunk(dir.path(), headers, true);
        saved.flush();
        assert_eq!(saved.load(dir.path()).unwrap().cursor, 510);
    }

    #[test]
    fn a_peer_ahead_of_the_saved_headers_continues_them() {
        let dir = tempdir().unwrap();
        let mut sync = SyncManager::default();
        save_all(&sync.saved, dir.path(), &chain(100, 50, 1));

        let (headers, cursor) = sync.load_backup(&dir.path().to_path_buf(), 200);
        assert_eq!(cursor, Some(150));
        assert_eq!(headers.len(), 50);
    }

    #[test]
    fn a_peer_at_or_behind_the_saved_headers_resumes_one_header_short_of_its_tip() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().to_path_buf();
        let mut sync = SyncManager::default();
        save_all(&sync.saved, dir.path(), &chain(100, 50, 1));

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
    fn the_block_pace_is_smoothed_and_restarts_after_a_jump_back() {
        let mut sync = SyncManager::default();
        assert_eq!(sync.block_rate(1_000), None, "one reading is no rate yet");
        sync.block_pace.as_mut().unwrap().at -= Duration::from_secs(100);
        let first = sync.block_rate(2_000).unwrap();
        assert!((first - 10.0).abs() < 0.5, "1,000 blocks in 100s: {first}");
        sync.block_pace.as_mut().unwrap().at -= Duration::from_secs(100);
        let second = sync.block_rate(4_000).unwrap();
        assert!((second - (0.3 * 20.0 + 0.7 * first)).abs() < 0.5, "smoothed: {second}");
        assert_eq!(sync.block_rate(4_000), Some(second), "no news, same rate");
        assert_eq!(sync.block_rate(3_000), None, "a session from further back starts over");
    }

    #[test]
    fn the_pace_reads_plainly() {
        assert_eq!(pace_text(11_061, 455.0 / 60.0), ", ~25 min at 455/min");
        assert_eq!(pace_text(380_000, 455.0 / 60.0), ", ~13 h 56 min at 455/min");
        assert_eq!(pace_text(30, 10.0), ", ~1 min at 600/min");
        assert_eq!(pace_text(0, 10.0), "");
        assert_eq!(pace_text(100, 0.0), "");
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
