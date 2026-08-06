mod batch_store;
pub mod search_index;
pub use batch_store::BatchStore;

use crate::core::State;
use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("state");
const MINING_SEED_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("mining_seed");
/// Maps spent WOTS address -> the commitment hash that legitimately spent it.
/// Allows safe replay of the exact same transaction during chain reorgs while
/// permanently blocking any *different* transaction from reusing the key.
const SPENT_ADDRESSES_TABLE: TableDefinition<&[u8; 32], &[u8; 32]> =
    TableDefinition::new("spent_addresses");

/// Archives the 576-byte WOTS signature from the first time a key was used.
/// Used by the MEV Bounty Hunter to mathematically penalize historical key reuses.
const SIGNATURE_ARCHIVE_TABLE: TableDefinition<&[u8; 32], &[u8]> = 
    TableDefinition::new("signature_archive");

/// Maps MSS master_pk -> highest (leaf_index + 1) seen on-chain.
/// Gives O(1) lookup for the /mss_state endpoint instead of scanning
/// every block from genesis.
const MSS_LEAF_INDEX_TABLE: TableDefinition<&[u8; 32], u64> =
    TableDefinition::new("mss_leaf_index");

/// Persistent address book: single key "book" -> bincode Vec<(multiaddr, alpha, beta)>.
/// The on-disk equivalent of Bitcoin's peers.dat.
const PEERS_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("peers");

/// Block storage tables
pub const BATCHES_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("batches");
pub const HEADERS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("headers");
pub const FILTERS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("filters");

/// Deserialize a `State` from bincode bytes and rebuild every cache that was
/// `#[serde(skip)]`'d on the wire.
///
/// # Formal specification
///
/// ```text
///   pre:   bytes encodes a valid State under either the CURRENT wire format
///          OR the LEGACY (v2.1.x pre-domain-separation) wire format
///   post:  result.coins, result.commitments, result.expirations are all
///          internally consistent caches over the deserialized canonical
///          fields, under is_v2 = is_v2_at(state.height)
/// ```
///
/// Two derived structures are reconstructed here:
///
/// 1. The `UtxoAccumulator` SMT caches (`nodes`, `buckets`) inside both
///    `state.coins` and `state.commitments` — empty after deserialisation
///    because they're `#[serde(skip)]`. Without this rebuild, calls to
///    `coins.root(...)` would silently return the empty-tree hash even
///    though the canonical coin set is populated.
///
/// 2. The `expirations` index, also `#[serde(skip)]`. Reconstructed from
///    `commitment_heights`. The per-height `Vec` is sorted lexicographically
///    so two processes loading the same on-disk state arrive at the same
///    in-memory layout regardless of `HashMap` iteration order.
///
/// `expirations` is not consensus-critical — only `commitment_heights`,
/// `coins`, `commitments`, and `chain_mmr` feed any hash — so the sort
/// choice is purely an in-memory hygiene call.
///
/// # Legacy migration
///
/// Pre-domain-separation builds (≤ v2.1.4) saved `State` with three
/// extra interspersed `is_v2: bool` bytes (one per accumulator and one
/// for the MMR) plus a fully-serialised `expirations` map between
/// `commitment_heights` and `chain_mmr`. The current code's `State`
/// has none of those: `is_v2` was removed entirely from the structs
/// (now passed as a parameter), and `expirations` is `#[serde(default,
/// skip)]`. So a strict bincode read of legacy bytes fails — usually
/// with `"Slice had bytes remaining"`, sometimes with a mid-stream
/// varint error if the misaligned bytes happen to hit an invalid
/// extension point.
///
/// On *any* strict-parse failure we fall back to a private
/// [`legacy::LegacyState`] shape that mirrors the pre-V2 wire layout
/// exactly. If THAT parse also fails, the file is genuinely corrupt or
/// follows an unknown third format, and we bubble up the original
/// (strict-new) error so the user sees the most relevant diagnostic.
///
/// The discarded `is_v2` bytes are *intentionally* dropped — the new
/// code derives the hashing mode from `state.height` via
/// [`is_v2_at`](crate::core::types::is_v2_at), which is the single
/// source of truth.
///
/// The first `save_state` after a successful legacy load writes the
/// state back in the current format, so the migration self-disables on
/// every running node.
fn deserialize_state(bytes: &[u8]) -> Result<State> {
    use bincode::Options;

    // FIX: Match bincode::serialize()'s implicit Fixint encoding!
    let strict = bincode::DefaultOptions::new()
        .with_limit(100_000_000)
        .with_fixint_encoding(); 

    let mut state = match strict.deserialize::<State>(bytes) {
        Ok(s) => s,
        Err(strict_err) => {
            // V3 -> V4 Safe Migration Fallback
            // This struct MUST exactly match the V3 disk format (NO burned_wots field!)
            #[derive(serde::Deserialize)]
            struct StateV3 {
                midstate: [u8; 32],
                coins: crate::core::mmr::UtxoAccumulator,
                commitments: crate::core::mmr::UtxoAccumulator,
                depth: u128,
                target: [u8; 32],
                height: u64,
                timestamp: u64,
                #[serde(default)]
                commitment_heights: im::HashMap<[u8; 32], u64>,
                #[serde(default)]
                chain_mmr: crate::core::mmr::MerkleMountainRange,
                header_hash: [u8; 32],
            }

            match strict.deserialize::<StateV3>(bytes) {
                Ok(v3) => State {
                    midstate: v3.midstate,
                    coins: v3.coins,
                    commitments: v3.commitments,
                    depth: v3.depth,
                    target: v3.target,
                    height: v3.height,
                    timestamp: v3.timestamp,
                    commitment_heights: v3.commitment_heights,
                    expirations: im::OrdMap::new(),
                    chain_mmr: v3.chain_mmr,
                    header_hash: v3.header_hash,
                    burned_wots: crate::core::mmr::UtxoAccumulator::new(), // Add the empty V4 SMT
                },
                Err(_) => match legacy::deserialize_legacy_state(bytes) {
                    Ok(migrated) => {
                        tracing::warn!(
                            "State on disk is in the pre-domain-separation wire format \
                             (strict parse failed: {}). Migrated successfully; the next \
                             save_state will write canonical form.",
                            strict_err
                        );
                        migrated
                    }
                    Err(legacy_err) => {
                        tracing::error!("Legacy migration also failed: {}", legacy_err);
                        return Err(anyhow::anyhow!("State deserialization failed: {}", strict_err));
                    }
                },
            }
        }
    };

    // Rebuild the SMT caches under the chain-height-implied hashing mode.
    let v2 = crate::core::types::is_v2_at(state.height);
    state.coins.rebuild_tree(v2);
    state.commitments.rebuild_tree(v2);
    state.burned_wots.rebuild_tree(v2);

    // Rebuild the expirations B-tree from commitment_heights.
    use std::collections::BTreeMap;
    let mut staging: BTreeMap<u64, Vec<[u8; 32]>> = BTreeMap::new();
    for (commitment, height) in &state.commitment_heights {
        staging.entry(*height).or_default().push(*commitment);
    }
    for list in staging.values_mut() {
        list.sort_unstable();
    }
    state.expirations = staging.into_iter().collect();

    Ok(state)
}

/// Pre-domain-separation wire format support. Module-private — no part of
/// this is intended to outlive the migration window.
mod legacy {
    use super::State;
    use anyhow::Result;
    use bincode::Options;
    use serde::Deserialize;

    /// Old `UtxoAccumulator` wire shape: canonical coin set followed by an
    /// `is_v2` byte. The `nodes` and `buckets` caches were `#[serde(skip)]`
    /// in the old code too, so they don't appear here.
    #[derive(Deserialize)]
    pub(super) struct LegacyUtxoAccumulator {
        pub coins: im::OrdSet<[u8; 32]>,
        #[serde(default)]
        pub _is_v2: bool,
    }

    /// Old `MerkleMountainRange` wire shape: post-order node array, leaf
    /// count, then an `is_v2` byte.
    #[derive(Deserialize, Default)]
    pub(super) struct LegacyMmr {
        pub nodes: im::Vector<[u8; 32]>,
        pub leaf_count: u64,
        #[serde(default)]
        pub _is_v2: bool,
    }

    /// Old `State` wire shape. Field order MUST match the pre-domain-
    /// separation `State` declaration order exactly — bincode is positional
    /// and does not record field names.
    #[derive(Deserialize)]
    pub(super) struct LegacyState {
        pub midstate: [u8; 32],
        pub coins: LegacyUtxoAccumulator,
        pub commitments: LegacyUtxoAccumulator,
        pub depth: u128,
        pub target: [u8; 32],
        pub height: u64,
        pub timestamp: u64,
        #[serde(default)]
        pub commitment_heights: im::HashMap<[u8; 32], u64>,
        /// Was a normal serialized field in the old code; in the new code
        /// it is `#[serde(default, skip)]` and rebuilt on load.
        #[allow(dead_code)]
        #[serde(default)]
        pub expirations: im::OrdMap<u64, Vec<[u8; 32]>>,
        #[serde(default)]
        pub chain_mmr: LegacyMmr,
        pub header_hash: [u8; 32],
    }

    /// Parse the legacy wire format and synthesise a current-shape `State`.
    ///
    /// # Formal specification
    /// ```text
    ///   pre:   bytes encodes a valid State under the LEGACY wire format
    ///   post:  result.* canonical fields = legacy.* canonical fields
    ///          result.coins, result.commitments, result.chain_mmr are
    ///                  reconstructed under is_v2 = is_v2_at(legacy.height)
    ///          legacy is_v2 bytes are discarded (the new code derives mode
    ///                  from height via is_v2_at, not from a stored field)
    /// ```
    ///
    /// Strict parse: legacy bytes must be consumed in full. If trailing
    /// bytes remain even under the legacy schema, we fail loudly rather
    /// than swallow them — at that point the file is genuinely corrupt
    /// or follows a third unknown format and forging ahead would silently
    /// produce a malformed in-memory state.
    pub(super) fn deserialize_legacy_state(bytes: &[u8]) -> Result<State> {
        let strict = bincode::DefaultOptions::new()
            .with_limit(100_000_000)
            .with_fixint_encoding();
            
        let legacy: LegacyState = strict
            .deserialize(bytes)
            .map_err(|e| anyhow::anyhow!("Legacy state parse failed: {}", e))?;

        let v2 = crate::core::types::is_v2_at(legacy.height);

        let coins = crate::core::mmr::UtxoAccumulator::from_canonical_coins(
            legacy.coins.coins,
            v2,
        );
        let commitments = crate::core::mmr::UtxoAccumulator::from_canonical_coins(
            legacy.commitments.coins,
            v2,
        );
        let chain_mmr = crate::core::mmr::MerkleMountainRange::from_raw_parts(
            legacy.chain_mmr.nodes,
            legacy.chain_mmr.leaf_count,
        );

        // expirations is rebuilt by the caller; we drop legacy.expirations on
        // purpose, since the canonical source-of-truth in the new code is
        // commitment_heights.

        Ok(State {
            midstate: legacy.midstate,
            coins,
            commitments,
            burned_wots: crate::core::mmr::UtxoAccumulator::new(),
            depth: legacy.depth,
            target: legacy.target,
            height: legacy.height,
            timestamp: legacy.timestamp,
            commitment_heights: legacy.commitment_heights,
            expirations: im::OrdMap::new(), 
            chain_mmr,
            header_hash: legacy.header_hash,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Storage {
    db: Arc<Database>,
    pub batches: BatchStore,
}

impl Storage {
    /// The raw database handle.
    ///
    /// Exposed for `search_index`, which needs its own read transactions.
    /// An accessor rather than a `pub` field so the ownership stays one-way:
    /// callers can read, they cannot swap the handle out from under BatchStore.
    pub fn db(&self) -> &Arc<Database> {
        &self.db
    }

    /// Build the search index up to `tip`, resuming if a previous run was
    /// interrupted.
    ///
    /// An existing node has none of this on disk, so the first run walks from
    /// genesis: ~200k batch deserialisations, minutes rather than hours. It is
    /// chunked and the watermark is committed with the data, so a kill -9 costs
    /// at most the last 1000 blocks.
    ///
    /// Safe to call on every startup — it is a no-op once current.
    pub fn build_search_index(&self) -> Result<u64> {
        let tip = self.batches.highest()?;
        let start = search_index::progress(&self.db)?;
        if start > tip {
            return Ok(0);
        }
        if start == 0 && tip > 1000 {
            tracing::info!(
                "Search index: building from genesis to {} (first run — this is a one-time pass)",
                tip
            );
        }
        let batches = self.batches.clone();
        let indexed = search_index::backfill(
            &self.db,
            |h| batches.load(h),
            tip,
            |done, total| {
                // Only log at 5% steps: a per-chunk line across 200k blocks is
                // 200 lines of noise, and silence looks like a hang.
                let step = (total / 20).max(1);
                if done % step < 1000 {
                    tracing::info!("Search index: {}/{} ({}%)", done, total, done * 100 / total.max(1));
                }
            },
        )?;
        if indexed > 0 {
            tracing::info!("Search index: indexed {} blocks, now current at {}", indexed, tip);
        }
        Ok(indexed)
    }

pub fn delete_spent_address(&self, address: &[u8; 32]) -> Result<bool> {
    let write_txn = self.db.begin_write()?;
    let existed = {
        let mut table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
        let removed = table.remove(address)?.is_some();
        removed
    };
    write_txn.commit()?;
    Ok(existed)
}

pub fn get_archived_signature(&self, wots_pk: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SIGNATURE_ARCHIVE_TABLE)?;
        Ok(table.get(wots_pk)?.map(|v| v.value().to_vec()))
    }

    pub fn get_spent_commitment(&self, wots_pk: &[u8; 32]) -> Result<Option<[u8; 32]>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SPENT_ADDRESSES_TABLE)?;
        Ok(table.get(wots_pk)?.map(|v| *v.value()))
    }
    
    /// Deletes any state snapshots at or above the given fork height.
    /// Called during a reorg to prevent stale snapshots from a dead chain
    /// from corrupting future state rebuilds.
    pub fn delete_snapshots_above(&self, fork_height: u64) -> Result<()> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        if !snapshot_dir.exists() { return Ok(()); }

        for entry in std::fs::read_dir(&snapshot_dir)? {
            if let Ok(entry) = entry {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("state_") && name.ends_with(".bin") {
                    if let Some(h_str) = name.strip_prefix("state_").and_then(|s| s.strip_suffix(".bin")) {
                        if let Ok(h) = h_str.parse::<u64>() {
                            if h >= fork_height {
                                let _ = std::fs::remove_file(entry.path());
                                tracing::debug!("Deleted stale snapshot at height {}", h);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn truncate_chain(&self, new_tip_height: u64) -> Result<()> {
        self.batches.truncate(new_tip_height)?;
        self.delete_snapshots_above(new_tip_height)?;
        Ok(())
    }

    /// Prunes old historical batches, headers, and filters according to PRUNE_DEPTH.
    ///
    /// This deletes data older than `current_height - PRUNE_DEPTH`.
    /// It is safe because blocks this deep are finalized and their UTXOs
    /// are represented in the rolling UtxoAccumulator state.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  current_height >= PRUNE_DEPTH
    /// Post: All data for height < (current_height - PRUNE_DEPTH) has been deleted
    ///       from the historical tables.
    /// ```
    ///
    /// ```zed
    ///     PruneOldData
    ///     ------------
    ///     ΞStorage
    ///     current_height? : ℕ
    ///
    ///     pre  current_height? ≥ PRUNE_DEPTH
    ///     post ∀ h • h < current_height? - PRUNE_DEPTH ⇒
    ///            h ∉ batches' ∧ h ∉ headers' ∧ h ∉ filters'
    /// ```
    pub fn prune_old_data(&self, current_height: u64) -> Result<()> {
        if current_height < crate::core::PRUNE_DEPTH {
            return Ok(());
        }

        let prune_up_to = current_height - crate::core::PRUNE_DEPTH;
        self.batches.prune_tail(prune_up_to)?;
        Ok(())
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let db_path = path.join("state.redb");
        let mut last_err = None;
        for attempt in 0..10 {
            match Database::create(&db_path) {
                Ok(mut db) => {
                    if attempt > 0 {
                        tracing::info!("Database lock acquired after {} retries", attempt);
                    }

                    tracing::info!("Compacting database to free dead pages...");
                    if let Err(e) = db.compact() {
                        tracing::warn!("Database compaction failed (non-fatal): {}", e);
                    } else {
                        tracing::info!("Database compaction complete.");
                    }

                    // Initialize tables
                    let write_txn = db.begin_write()?;
                    {
                        let _ = write_txn.open_table(STATE_TABLE)?;
                        let _ = write_txn.open_table(MINING_SEED_TABLE)?;
                        let _ = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
                        let _ = write_txn.open_table(SIGNATURE_ARCHIVE_TABLE)?;
                        let _ = write_txn.open_table(MSS_LEAF_INDEX_TABLE)?;
                        let _ = write_txn.open_table(BATCHES_TABLE)?;
                        let _ = write_txn.open_table(HEADERS_TABLE)?;
                        let _ = write_txn.open_table(FILTERS_TABLE)?;
                        // Search acceleration. Created here so a fresh database
                        // has them from the start; on an existing one they are
                        // created empty and populated by build_search_index().
                        let _ = write_txn.open_table(search_index::FILTER_META_TABLE)?;
                        let _ = write_txn.open_table(search_index::SEARCH_INDEX_TABLE)?;
                        let _ = write_txn.open_table(search_index::INDEX_META_TABLE)?;
                    }
                    write_txn.commit()?;

                    let db_arc = Arc::new(db);
                    let batches = BatchStore::new(db_arc.clone(), path.join("batches"))?;

                    return Ok(Self {
                        db: db_arc,
                        batches,
                    });
                }
                Err(e) => {
                    last_err = Some(e);
                    let delay = std::time::Duration::from_millis(100 * (1 << attempt.min(5)));
                    tracing::warn!(
                        "Database lock attempt {} failed, retrying in {:?}...",
                        attempt + 1, delay
                    );
                    std::thread::sleep(delay);
                }
            }
        }
        Err(last_err.unwrap().into())
    }

    /// Saves a historical snapshot of the state so it can be served to fast-syncing peers.
    /// Implements a rolling window: keeps only the 10 most recent snapshots.
    pub fn save_state_snapshot(&self, height: u64, state: &State) -> Result<()> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        std::fs::create_dir_all(&snapshot_dir)?;

        let path = snapshot_dir.join(format!("state_{}.bin", height));
        let bytes = bincode::serialize(state)?;
        std::fs::write(path, bytes)?;

        // Garbage-collect old snapshots: keep only the 10 most recent.
        let mut snapshots: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(&snapshot_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("state_") && name.ends_with(".bin") {
                    let h: u64 = name.strip_prefix("state_")?.strip_suffix(".bin")?.parse().ok()?;
                    Some((h, e.path()))
                } else {
                    None
                }
            })
            .collect();
        snapshots.sort_by_key(|(h, _)| std::cmp::Reverse(*h));
        for (_, old_path) in snapshots.into_iter().skip(10) {
            let _ = std::fs::remove_file(&old_path);
            tracing::debug!("Pruned old snapshot: {}", old_path.display());
        }

        Ok(())
    }

    /// Loads a historical snapshot to serve to a peer.
    ///
    /// Routes through the canonical [`deserialize_state`], which rebuilds
    /// the SMT caches and the `expirations` index automatically.
    pub fn load_state_snapshot(&self, height: u64) -> Result<Option<State>> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        let path = snapshot_dir.join(format!("state_{}.bin", height));

        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let state = deserialize_state(&bytes)?;
            Ok(Some(state))
        } else {
            Ok(None)
        }
    }

    pub fn save_state(&self, state: &State) -> Result<()> {
        let bytes = bincode::serialize(state)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(STATE_TABLE)?;
            table.insert("current", bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Loads the persisted current state from the redb-backed STATE_TABLE.
    ///
    /// Routes through the canonical [`deserialize_state`], which rebuilds
    /// the SMT caches and the `expirations` index automatically.
    pub fn load_state(&self) -> Result<Option<State>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(STATE_TABLE)?;
        match table.get("current")? {
            Some(bytes) => {
                let state = deserialize_state(bytes.value())?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    pub fn save_mining_seed(&self, seed: &[u8; 32]) -> Result<()> {
        // Save to flat file for concurrent CLI access
        let seed_path = self.batches.base_path().parent().unwrap().join("mining_seed.key");
        std::fs::write(&seed_path, seed)?;

        // Restrict permissions to owner-only on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(mut perms) = std::fs::metadata(&seed_path).map(|m| m.permissions()) {
                perms.set_mode(0o600);
                let _ = std::fs::set_permissions(&seed_path, perms);
            }
        }
        // Also save to redb for backwards compatibility
        if let Ok(write_txn) = self.db.begin_write() {
            if let Ok(mut table) = write_txn.open_table(MINING_SEED_TABLE) {
                let _ = table.insert("seed", seed.as_slice());
            }
            let _ = write_txn.commit();
        }
        Ok(())
    }

    pub fn load_mining_seed(&self) -> Result<Option<[u8; 32]>> {
        // 1. Try reading from the concurrent-safe flat file first
        let seed_path = self.batches.base_path().parent().unwrap().join("mining_seed.key");
        if seed_path.exists() {
            let bytes = std::fs::read(&seed_path)?;
            if bytes.len() == 32 {
                return Ok(Some(<[u8; 32]>::try_from(bytes.as_slice()).unwrap()));
            }
        }

        // 2. Fallback: load from redb (for existing nodes) and migrate to flat file
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MINING_SEED_TABLE)?;
        match table.get("seed")? {
            Some(bytes) => {
                let val = bytes.value();
                if val.len() != 32 {
                    anyhow::bail!("corrupt mining seed");
                }
                let seed = <[u8; 32]>::try_from(val).unwrap();
                // Auto-migrate to flat file so CLI doesn't need redb lock next time
                let _ = self.save_mining_seed(&seed);
                Ok(Some(<[u8; 32]>::try_from(val).unwrap()))
            }
            None => Ok(None),
        }
    }

    /// Persists the node's address book (the `peers.dat` analogue).
    ///
    /// # Reasoning
    /// Without an on-disk address book, a node re-derives its entire view of the
    /// network from the seed registry on every restart. That makes the registry
    /// load-bearing on every single boot rather than a cold-start fallback, and
    /// it means the network cannot survive the registry going away. Persisting
    /// peers is what lets the seed registry become optional.
    ///
    /// Stored as one bincode blob under a single key rather than one row per
    /// peer, so a save is a single write transaction regardless of book size.
    ///
    /// # Safety / Invariants
    /// - **Bounded Growth:** The caller passes an already-truncated top-N slice.
    /// - **Full Replacement:** The blob is overwritten wholesale, so entries the
    ///   caller dropped do not survive as orphans.
    pub fn save_peers(&self, peers: &[(String, u32, u32)]) -> Result<()> {
        let bytes = bincode::serialize(peers)?;
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(PEERS_TABLE)?;
            table.insert("book", bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Loads the persisted address book. Returns an empty vec on first run or if
    /// the blob is corrupt — a bad address book must never prevent startup.
    pub fn load_peers(&self) -> Result<Vec<(String, u32, u32)>> {
        let read_txn = self.db.begin_read()?;
        let table = match read_txn.open_table(PEERS_TABLE) {
            Ok(t) => t,
            // Table absent: first run on an existing database.
            Err(_) => return Ok(Vec::new()),
        };
        match table.get("book")? {
            Some(bytes) => Ok(bincode::deserialize(bytes.value()).unwrap_or_default()),
            None => Ok(Vec::new()),
        }
    }

    pub fn save_batch(&self, height: u64, batch: &crate::core::Batch) -> Result<()> {
        self.batches.save(height, batch)
    }

    pub fn load_batch(&self, height: u64) -> Result<Option<crate::core::Batch>> {
        self.batches.load(height)
    }

    pub fn load_batches(&self, start: u64, end: u64) -> Result<Vec<(u64, crate::core::Batch)>> {
        self.batches.load_range(start, end)
    }

    pub fn highest_batch(&self) -> Result<u64> {
        self.batches.highest()
    }

    /// Reverts the burning of one-time keys from an abandoned chain segment.
    ///
    /// # Reasoning
    ///
    /// A reorg un-mines the spends in the orphaned segment. If their keys stayed
    /// burned, the identical transaction could not be re-mined on the new chain:
    /// `Mempool::add` would reject it as reuse and the bounty hunter would treat
    /// an honest re-broadcast as a punishable double-signature.
    ///
    /// The counterpart of `burn_batch_addresses` and MUST cover exactly the same
    /// keys. It previously read only `wit_inputs.first()`, matching the burn side
    /// only because the burn side had the same defect; widening one without the
    /// other is what leaves ghosts.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  batch was previously burned by burn_batch_addresses
    ///
    /// Post: ∀ s ∈ mss_sigs(batch) •
    ///         s.wots_pk ∉ dom(spent')  ∧  s.wots_pk ∉ dom(archive')
    ///       ∀ w ∈ wots_first_sigs(batch) •
    ///         address(w) ∉ dom(spent') ∧ address(w) ∉ dom(archive')
    ///       mss_idx' = mss_idx        (deliberately untouched)
    ///       result = Err(_) ⇒ state unchanged (single write transaction)
    /// ```
    ///
    /// ```zed
    ///     UnburnBatchAddresses
    ///     --------------------
    ///     ΔSpentAddresses
    ///     ΔSignatureArchive
    ///     ΞMssLeafIndex
    ///     batch? : Batch
    ///
    ///     post dom(spent') = dom(spent) \ nullifiers(batch?)
    ///     post dom(archive') = dom(archive) \ nullifiers(batch?)
    ///     post mss_idx' = mss_idx
    /// ```
    ///
    /// # Safety / Invariants
    ///
    /// - **Must remain symmetric with `burn_batch_addresses`.** Same stack walk,
    ///   same nullifiers. Asymmetry in either direction is a live bug: too narrow
    ///   leaves ghosts, too wide un-burns a key another batch legitimately spent.
    /// - **The leaf counter is NOT rolled back**, by design. Leaving it high skips
    ///   one leaf out of `2^H`; lowering it hands back a leaf a wallet may
    ///   re-sign, which is the failure this whole subsystem exists to prevent.
    pub fn unburn_batch_addresses(&self, batch: &crate::core::Batch) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut spent_table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
            let mut sig_archive_table = write_txn.open_table(SIGNATURE_ARCHIVE_TABLE)?;
            // Note: We deliberately do not roll back the MSS_LEAF_INDEX_TABLE here.
            // That table only tracks the highest seen index for the `/mss_state` RPC endpoint.
            // Leaving it slightly high just skips a reusable index, which is perfectly safe.

            for tx in &batch.transactions {
                match tx {
                    crate::core::Transaction::Reveal { inputs, witnesses, .. } => {
                        for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                            let crate::core::types::Witness::ScriptInputs(wit_inputs) = witness;
                            if let Some(sig) = wit_inputs.first() {
                                if sig.len() == crate::core::wots::SIG_SIZE {
                                    let addr = input.predicate.address();
                                    spent_table.remove(&addr)?;
                                    sig_archive_table.remove(&addr)?;
                                }
                            }
                            // Must mirror burn_batch_addresses exactly: every MSS
                            // signature in the stack, not just the first. An unburn
                            // narrower than the burn leaves ghost entries after a
                            // reorg, which read as reuse and get the key swept.
                            for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                                spent_table.remove(&mss_sig.wots_pk)?;
                                sig_archive_table.remove(&mss_sig.wots_pk)?;
                            }
                        }
                    }
                    crate::core::Transaction::Consolidate { inputs, witness, .. } => {
                        if inputs.is_empty() { continue; }
                        let crate::core::types::Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            if sig.len() == crate::core::wots::SIG_SIZE {
                                let addr = inputs[0].predicate.address();
                                spent_table.remove(&addr)?;
                                sig_archive_table.remove(&addr)?;
                            }
                        }
                        for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                            spent_table.remove(&mss_sig.wots_pk)?;
                            sig_archive_table.remove(&mss_sig.wots_pk)?;
                        }
                    }
                    _ => {}
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Burn every one-time key a committed batch spent, map each to the
    /// commitment that authorised it, and advance the MSS leaf high-water mark
    /// for every master key involved.
    ///
    /// # Reasoning
    ///
    /// Three records come out of one batch and they must agree.
    /// `SPENT_ADDRESSES_TABLE` is what `Mempool::add` rejects reuse against and
    /// what the bounty hunter sweeps on. `SIGNATURE_ARCHIVE_TABLE` holds the
    /// evidence for that sweep. `MSS_LEAF_INDEX_TABLE` is advisory: the only
    /// answer `/mss_state` can give a wallet asking which leaf to sign next.
    ///
    /// Two defects made the third disagree with the first:
    ///
    ///   1. The leaf counter was gated on `input.predicate.owner_pk()`, which is
    ///      `Some` only for the 40-byte P2PK bytecode — while
    ///      `script::verify_signature` accepts an MSS signature at any
    ///      `OP_CHECKSIG`. Every leaf burned through `compile_htlc`,
    ///      `compile_covenant_htlc`, `compile_limit_order_covenant` or either
    ///      multisig script was recorded as spent but advanced no counter.
    ///   2. Only `wit_inputs.first()` was inspected, so in a 2-of-N spend the
    ///      second signer's one-time leaf was never burned at all.
    ///
    /// Consequence of (1): a wallet's local counter advanced while the node's did
    /// not, silently and permanently. Once the wallet lost its counter — rescan,
    /// import, second device — `max(local, node)` could not recover the truth,
    /// so it re-signed a burned leaf. The mempool accepted the reveal and
    /// `prune_on_new_block` evicted it against the chain on every block, forever,
    /// with no error returned at any layer.
    ///
    /// Attribution is therefore by `mss::recover_master_pk` — the key that
    /// actually signed — never by the shape of the predicate.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  batch has passed consensus validation (apply_batch returned Ok)
    ///
    /// Post: ∀ s ∈ mss_sigs(batch) •
    ///         spent'[s.wots_pk]   = commitment(tx of s)
    ///         archive'[s.wots_pk] = wots_sig_bytes(s)
    ///         mss_idx'[recover_master_pk(s)]
    ///                             = max(mss_idx[recover_master_pk(s)], s.leaf_index + 1)
    ///       ∀ w ∈ wots_first_sigs(batch) •
    ///         spent'[address(w)]  = commitment(tx of w)
    ///         archive'[address(w)]= w
    ///       ∀ pk • mss_idx'(pk) ≥ mss_idx(pk)
    ///       result = Err(_)       ⇒ state unchanged (single write transaction)
    /// ```
    ///
    /// ```zed
    ///     BurnBatchAddresses
    ///     ------------------
    ///     ΔSpentAddresses
    ///     ΔSignatureArchive
    ///     ΔMssLeafIndex
    ///     batch? : Batch
    ///
    ///     pre  valid(batch?)
    ///
    ///     post ∀ s ∈ mss_sigs(batch?) •
    ///            spent'(s.wots_pk) = commitment(s) ∧
    ///            mss_idx'(recover_master_pk(s))
    ///              = max(mss_idx(recover_master_pk(s)), s.leaf_index + 1)
    ///     post ∀ pk ∉ { recover_master_pk(s) | s ∈ mss_sigs(batch?) } •
    ///            mss_idx'(pk) = mss_idx(pk)
    ///     post ∀ pk • mss_idx'(pk) ≥ mss_idx(pk)
    /// ```
    ///
    /// # Safety / Invariants
    ///
    /// - **Monotonic.** `if next > current` is the entire safety argument for the
    ///   counter. A value left too high skips one leaf out of `2^H`; a value
    ///   allowed to fall hands back a leaf a wallet may re-sign.
    /// - **Not rolled back on reorg.** `unburn_batch_addresses` reverts the spent
    ///   set so an orphaned spend can be re-mined; the counter deliberately does
    ///   not move. See the note there.
    /// - **`unburn_batch_addresses` must stay symmetric with this function.** Both
    ///   walk the stack through `mss::mss_sigs_in_stack`; an unburn narrower than
    ///   the burn leaves ghost entries that read as key reuse.
    /// - **Idempotent.** Reorg replay of the same batch writes identical values in
    ///   all three tables.
    /// - **Not consensus.** None of these tables is read by `apply_batch`, which
    ///   only warns on reuse (`core/state.rs`). Widening what is burned changes
    ///   mempool policy and local evidence, never block acceptance.
    /// - **WOTS remains first-item-only, deliberately.** A raw WOTS nullifier is
    ///   `input.predicate.address()`, which for a multisig script is the script
    ///   hash rather than the signer's key. Burning per stack item would re-insert
    ///   that same script address, not the second signer's key. Fixing WOTS
    ///   multisig properly needs public-key recovery from the signature AND a
    ///   matching change in `core/state.rs`'s oracle key space — a consensus-path
    ///   edit, out of scope here. See the module note in `mss::mss_sigs_in_stack`.
    pub fn burn_batch_addresses(&self, batch: &crate::core::Batch, _block_height: u64) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut spent_table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
            let mut mss_idx_table = write_txn.open_table(MSS_LEAF_INDEX_TABLE)?;
            let mut sig_archive_table = write_txn.open_table(SIGNATURE_ARCHIVE_TABLE)?;

            for tx in &batch.transactions {
                match tx {
                    crate::core::Transaction::Reveal { inputs, witnesses, outputs, salt } => {
                        let input_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                        let output_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
                        let commitment = crate::core::compute_commitment(&input_ids, &output_hashes, salt);

                        for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                            let crate::core::types::Witness::ScriptInputs(wit_inputs) = witness;
                            if let Some(sig) = wit_inputs.first() {
                                if sig.len() == crate::core::wots::SIG_SIZE {
                                    let addr = input.predicate.address();
                                    spent_table.insert(&addr, &commitment)?;
                                    sig_archive_table.insert(&addr, sig.as_slice())?;
                                }
                            }
                            // Every MSS signature in the stack, attributed by the
                            // key that SIGNED rather than by the predicate's shape.
                            // The old code took wit_inputs.first() only (missing the
                            // second signer of a 2-of-N) and gated the leaf counter
                            // on input.predicate.owner_pk(), which is Some only for
                            // the 40-byte P2PK script — so every leaf burned through
                            // an HTLC, covenant or multisig predicate advanced no
                            // counter and /mss_state under-reported it forever.
                            for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                                spent_table.insert(&mss_sig.wots_pk, &commitment)?;
                                let wots_sig_bytes = crate::core::wots::sig_to_bytes(&mss_sig.wots_sig);
                                sig_archive_table.insert(&mss_sig.wots_pk, wots_sig_bytes.as_slice())?;

                                let master_pk = crate::core::mss::recover_master_pk(&mss_sig);
                                let next = mss_sig.leaf_index.saturating_add(1);
                                let current = mss_idx_table.get(&master_pk)?.map(|v: redb::AccessGuard<'_, u64>| v.value()).unwrap_or(0);
                                if next > current { mss_idx_table.insert(&master_pk, next)?; }
                            }
                        }
                    }
                    crate::core::Transaction::Consolidate { inputs, witness, outputs, salt } => {
                        if inputs.is_empty() { continue; }
                        let input_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                        let output_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
                        let commitment = crate::core::compute_commitment(&input_ids, &output_hashes, salt);

                        let crate::core::types::Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            if sig.len() == crate::core::wots::SIG_SIZE {
                                let addr = inputs[0].predicate.address();
                                spent_table.insert(&addr, &commitment)?;
                                sig_archive_table.insert(&addr, sig.as_slice())?;
                            }
                        }
                        for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                            spent_table.insert(&mss_sig.wots_pk, &commitment)?;
                            let wots_sig_bytes = crate::core::wots::sig_to_bytes(&mss_sig.wots_sig);
                            sig_archive_table.insert(&mss_sig.wots_pk, wots_sig_bytes.as_slice())?;

                            let master_pk = crate::core::mss::recover_master_pk(&mss_sig);
                            let next = mss_sig.leaf_index.saturating_add(1);
                            let current = mss_idx_table.get(&master_pk)?.map(|v: redb::AccessGuard<'_, u64>| v.value()).unwrap_or(0);
                            if next > current { mss_idx_table.insert(&master_pk, next)?; }
                        }
                    }
                    _ => {}
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }


    /// One-time replay of every historical batch still on disk through
    /// `burn_batch_addresses`, repairing the spent-address, signature-archive and
    /// MSS leaf-index tables in a single pass.
    ///
    /// # Reasoning
    ///
    /// Fixing `burn_batch_addresses` only corrects blocks mined from the upgrade
    /// onward. Every historical covenant spend is still missing from
    /// `MSS_LEAF_INDEX_TABLE`, and every historical second-signer leaf is still
    /// missing from `SPENT_ADDRESSES_TABLE`. Shipping the code fix alone leaves
    /// those holes on disk permanently — and a hole in the leaf index is exactly
    /// what makes a wallet re-sign a burned leaf.
    ///
    /// This calls `burn_batch_addresses` itself rather than reimplementing the
    /// walk, so the repair cannot diverge from the writer.
    ///
    /// `PRUNE_DEPTH` bounds what is recoverable: batches below the pruned floor
    /// are gone and their burns cannot be re-derived from any local source. For
    /// keys last used down there the tables remain a LOWER BOUND, which is why
    /// wallets must keep taking `max(local, node)` rather than assigning.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  true (safe to re-run; gated by a marker file in Node::new)
    ///
    /// Post: ∀ h ∈ available_heights • burn_batch_addresses(batch(h)) applied
    ///       ∀ pk • mss_idx'(pk) ≥ mss_idx(pk)
    ///       spent' ⊇ spent  ∧  archive' ⊇ archive
    ///       a failure at height h ⇒ that height skipped, later heights still
    ///                               processed (never aborts the whole pass)
    /// ```
    ///
    /// ```zed
    ///     BackfillSignatureArchive
    ///     ------------------------
    ///     ΔSpentAddresses
    ///     ΔSignatureArchive
    ///     ΔMssLeafIndex
    ///
    ///     post ∀ pk • mss_idx'(pk) ≥ mss_idx(pk)
    ///     post dom(spent') ⊇ dom(spent)
    ///     post dom(archive') ⊇ dom(archive)
    /// ```
    ///
    /// # Safety / Invariants
    ///
    /// - **Purely additive.** It never clears a table first, and every write it
    ///   makes is an insert or a forward max-merge. An interrupted run therefore
    ///   leaves the database strictly no worse than it found it, and re-running is
    ///   a no-op.
    /// - **Not atomic across heights.** Each `burn_batch_addresses` call is its
    ///   own write transaction. Acceptable precisely because of the point above;
    ///   the marker file is only written on a clean full pass, so an interrupted
    ///   run repeats from genesis on the next boot.
    /// - **Blocking at startup, by choice.** Serving `/mss_state` from a partially
    ///   repaired index means answering too LOW, which is the dangerous direction.
    /// - Returns the number of batches processed.
    pub fn backfill_signature_archive(&self) -> Result<usize> {
        let highest = self.highest_batch()?;
        if highest == 0 {
            return Ok(0);
        }

        let mut processed = 0usize;
        let start = 0;

        for h in start..=highest {
            if let Ok(Some(batch)) = self.load_batch(h) {
                // Re-use the exact same logic that runs on every new block
                if let Err(e) = self.burn_batch_addresses(&batch, h) {
                    tracing::warn!("backfill: burn_batch_addresses failed at height {}: {}", h, e);
                    continue;
                }
                processed += 1;

                if processed % 500 == 0 {
                    tracing::info!("backfill_signature_archive: processed {} batches (height {})", processed, h);
                }
            }
        }

        tracing::info!(
            "backfill_signature_archive complete: {} batches processed (heights {}..{})",
            processed, start, highest
        );
        Ok(processed)
    }

    /// Build a pre-flight oracle for a batch: returns a map of
    /// `nullifier -> prior_commitment` for every WOTS address or MSS leaf in the batch
    /// that already exists in the spent-address table.
    pub fn query_spent_addresses(
        &self,
        batch: &crate::core::Batch,
    ) -> Result<std::collections::HashMap<[u8; 32], [u8; 32]>> {
        use crate::core::types::Witness;
        use crate::core::wots::SIG_SIZE;

        let mut result = std::collections::HashMap::new();
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SPENT_ADDRESSES_TABLE)?;

        for tx in &batch.transactions {
            match tx {
                crate::core::Transaction::Reveal { inputs, witnesses, .. } => {
                    for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                        let Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            if sig.len() == SIG_SIZE {
                                let addr = input.predicate.address();
                                if let Some(existing) = table.get(&addr)? { result.insert(addr, *existing.value()); }
                            }
                        }
                        // Oracle must cover exactly what burn_batch_addresses writes,
                        // or a reuse that IS recorded goes undetected at admission.
                        for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                            if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                        }
                    }
                }
                crate::core::Transaction::Consolidate { inputs, witness, .. } => {
                    if inputs.is_empty() { continue; }
                    let Witness::ScriptInputs(wit_inputs) = witness;
                    if let Some(sig) = wit_inputs.first() {
                        if sig.len() == SIG_SIZE {
                            let addr = inputs[0].predicate.address();
                            if let Some(existing) = table.get(&addr)? { result.insert(addr, *existing.value()); }
                        }
                    }
                    for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                        if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                    }
                }
                _ => {}
            }
        }
        Ok(result)
    }

    /// Single-transaction variant of `query_spent_addresses`.
    /// Used at mempool admission time when only one tx is being checked.
    pub fn query_spent_addresses_for_tx(
        &self,
        tx: &crate::core::Transaction,
    ) -> Result<std::collections::HashMap<[u8; 32], [u8; 32]>> {
        use crate::core::types::Witness;
        use crate::core::wots::SIG_SIZE;

        let mut result = std::collections::HashMap::new();
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(SPENT_ADDRESSES_TABLE)?;

        match tx {
            crate::core::Transaction::Reveal { inputs, witnesses, .. } => {
                for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                    let Witness::ScriptInputs(wit_inputs) = witness;
                    if let Some(sig) = wit_inputs.first() {
                        if sig.len() == SIG_SIZE {
                            let addr = input.predicate.address();
                            if let Some(existing) = table.get(&addr)? { result.insert(addr, *existing.value()); }
                        }
                    }
                    for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                        if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                    }
                }
            }
            crate::core::Transaction::Consolidate { inputs, witness, .. } => {
                if inputs.is_empty() { return Ok(result); }
                let Witness::ScriptInputs(wit_inputs) = witness;
                if let Some(sig) = wit_inputs.first() {
                    if sig.len() == SIG_SIZE {
                        let addr = inputs[0].predicate.address();
                        if let Some(existing) = table.get(&addr)? { result.insert(addr, *existing.value()); }
                    }
                }
                for mss_sig in crate::core::mss::mss_sigs_in_stack(wit_inputs) {
                    if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                }
            }
            _ => {}
        }

        Ok(result)
    }

    /// O(1) lookup of the highest MSS leaf index used on-chain for a given master_pk.
    /// Returns 0 if the master_pk has never been seen (or pre-activation blocks only).
    pub fn query_mss_leaf_index(&self, master_pk: &[u8; 32]) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MSS_LEAF_INDEX_TABLE)?;
        Ok(table.get(master_pk)?.map(|v| v.value()).unwrap_or(0))
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Burn-path tests
//
// `storage.rs` previously had no test module, which is how two defects in
// `burn_batch_addresses` survived: MSS leaves were attributed via
// `Predicate::owner_pk()` (Some only for the 40-byte P2PK script) and only
// `wit_inputs.first()` was inspected. Together they meant a leaf spent through
// a covenant, HTLC or multisig predicate was recorded as spent but advanced no
// counter, so `/mss_state` under-reported it and wallets re-signed burned
// leaves. Everything below pins the properties that make that impossible.
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod burn_tests {
    use super::*;
    use crate::core::mss;
    use crate::core::script;
    use crate::core::types::{
        Batch, Extension, InputReveal, OutputData, Predicate, Transaction, Witness,
    };
    use tempfile::tempdir;

    fn store(dir: &tempfile::TempDir) -> Storage {
        Storage::open(dir.path()).expect("open storage")
    }

    /// A batch carrying exactly the transactions given. Only the fields the burn
    /// path reads matter; the rest is inert filler.
    fn batch_of(txs: Vec<Transaction>) -> Batch {
        Batch {
            prev_midstate: [0u8; 32],
            transactions: txs,
            extension: Extension { nonce: 0, final_hash: [0u8; 32] },
            coinbase: vec![],
            timestamp: 1_700_000_000,
            target: [0xFFu8; 32],
            state_root: [0u8; 32],
            prev_header_hash: [0u8; 32],
        }
    }

    fn reveal(predicate: Predicate, witness: Witness, salt: u8) -> Transaction {
        Transaction::Reveal {
            inputs: vec![InputReveal { predicate, value: 1_000, salt: [salt; 32], commitment: None }],
            witnesses: vec![witness],
            outputs: vec![OutputData::Standard { address: [0xAAu8; 32], value: 900, salt: [salt; 32] }],
            salt: [salt; 32],
        }
    }

    /// A predicate `owner_pk()` cannot see through — the shape that used to make
    /// the leaf counter silently skip a burn.
    fn htlc_predicate(receiver_pk: &[u8; 32]) -> Predicate {
        Predicate::Script {
            bytecode: script::compile_htlc(&[0x11u8; 32], receiver_pk, 500_000, &[0x22u8; 32]),
        }
    }

    /// THE regression test. An MSS leaf spent through an HTLC predicate must
    /// advance `MSS_LEAF_INDEX_TABLE`, even though `owner_pk()` returns None.
    #[test]
    fn covenant_spend_advances_the_leaf_index() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut kp = mss::keygen(&[3u8; 32], 4).unwrap();
        let master_pk = kp.public_key();
        let sig = kp.sign(&[0x42u8; 32]).unwrap();

        let predicate = htlc_predicate(&master_pk);
        assert!(
            predicate.owner_pk().is_none(),
            "test is meaningless if the predicate is P2PK-shaped"
        );

        // Realistic HTLC claim stack: [signature, preimage, 1]. Only the first
        // item is a signature; the others must not be mistaken for one.
        let witness = Witness::ScriptInputs(vec![sig.to_bytes(), vec![0x99u8; 32], vec![0x01]]);
        let batch = batch_of(vec![reveal(predicate, witness, 1)]);

        st.burn_batch_addresses(&batch, 1).unwrap();

        assert_eq!(
            st.query_mss_leaf_index(&master_pk).unwrap(),
            sig.leaf_index + 1,
            "leaf burned through a covenant must advance the index"
        );
    }

    /// Every signature in the stack burns, not just the first. This is the
    /// 2-of-N case where the second signer's one-time leaf was never recorded.
    #[test]
    fn multisig_burns_every_signature_in_the_stack() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut a = mss::keygen(&[4u8; 32], 4).unwrap();
        let mut b = mss::keygen(&[5u8; 32], 4).unwrap();
        let (pk_a, pk_b) = (a.public_key(), b.public_key());
        let sig_a = a.sign(&[0x43u8; 32]).unwrap();
        let sig_b = b.sign(&[0x43u8; 32]).unwrap();

        assert_ne!(pk_a, pk_b, "distinct trees");
        assert_ne!(sig_a.wots_pk, sig_b.wots_pk, "distinct leaves");

        let stack = vec![sig_a.to_bytes(), sig_b.to_bytes()];

        // ── Stage 1: does the stack walk see both signatures? ──
        let parsed = mss::mss_sigs_in_stack(&stack);
        assert_eq!(parsed.len(), 2, "STAGE 1: stack walk lost a signature");

        // ── Stage 2: does recovery attribute each to the right tree? ──
        assert_eq!(mss::recover_master_pk(&parsed[0]), pk_a, "STAGE 2: wrong root for signer A");
        assert_eq!(mss::recover_master_pk(&parsed[1]), pk_b, "STAGE 2: wrong root for signer B");

        let predicate = Predicate::Script { bytecode: script::compile_multisig_2of2(&pk_a, &pk_b) };
        let batch = batch_of(vec![reveal(predicate, Witness::ScriptInputs(stack), 2)]);
        st.burn_batch_addresses(&batch, 1).unwrap();

        // ── Stage 3: did both nullifiers burn? ──
        let oracle = st.query_spent_addresses(&batch).unwrap();
        assert!(oracle.contains_key(&sig_a.wots_pk), "STAGE 3: signer A missing from spent table");
        assert!(oracle.contains_key(&sig_b.wots_pk), "STAGE 3: signer B missing from spent table");

        // ── Stage 4: did both counters advance? ──
        assert_eq!(
            st.query_mss_leaf_index(&pk_a).unwrap(),
            sig_a.leaf_index + 1,
            "STAGE 4: signer A counter"
        );
        assert_eq!(
            st.query_mss_leaf_index(&pk_b).unwrap(),
            sig_b.leaf_index + 1,
            "STAGE 4: signer B counter — burned but not counted"
        );
    }

    /// `unburn_batch_addresses` must cover exactly what the burn covered — an
    /// unburn narrower than the burn leaves ghost entries that read as reuse
    /// after a reorg. The leaf counter must NOT be rolled back.
    #[test]
    fn unburn_is_symmetric_and_leaves_the_counter_high() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut a = mss::keygen(&[6u8; 32], 4).unwrap();
        let mut b = mss::keygen(&[7u8; 32], 4).unwrap();
        let (pk_a, pk_b) = (a.public_key(), b.public_key());
        let sig_a = a.sign(&[0x44u8; 32]).unwrap();
        let sig_b = b.sign(&[0x44u8; 32]).unwrap();

        let predicate = Predicate::Script { bytecode: script::compile_multisig_2of2(&pk_a, &pk_b) };
        let witness = Witness::ScriptInputs(vec![sig_a.to_bytes(), sig_b.to_bytes()]);
        let batch = batch_of(vec![reveal(predicate, witness, 3)]);

        st.burn_batch_addresses(&batch, 1).unwrap();
        assert_eq!(st.query_spent_addresses(&batch).unwrap().len(), 2);

        st.unburn_batch_addresses(&batch).unwrap();
        assert!(
            st.query_spent_addresses(&batch).unwrap().is_empty(),
            "every nullifier the burn wrote must be removed, or a reorg leaves ghosts"
        );

        assert_eq!(
            st.query_mss_leaf_index(&pk_a).unwrap(),
            sig_a.leaf_index + 1,
            "the leaf counter is deliberately NOT rolled back: skipping a leaf is \
             free, handing one back invites reuse"
        );
    }

    /// The counter only ever moves forward. An out-of-order or replayed batch
    /// carrying a lower leaf must not lower it.
    #[test]
    fn leaf_index_is_monotonic() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut kp = mss::keygen(&[8u8; 32], 4).unwrap();
        let master_pk = kp.public_key();

        let mut sigs = Vec::new();
        for i in 0..6u8 {
            sigs.push(kp.sign(&[i; 32]).unwrap());
        }
        let high = sigs.pop().unwrap(); // leaf 5
        let low = sigs[2].clone(); // leaf 2

        let p = || htlc_predicate(&master_pk);
        st.burn_batch_addresses(&batch_of(vec![reveal(p(), Witness::sig(high.to_bytes()), 4)]), 1)
            .unwrap();
        assert_eq!(st.query_mss_leaf_index(&master_pk).unwrap(), high.leaf_index + 1);

        st.burn_batch_addresses(&batch_of(vec![reveal(p(), Witness::sig(low.to_bytes()), 5)]), 2)
            .unwrap();
        assert_eq!(
            st.query_mss_leaf_index(&master_pk).unwrap(),
            high.leaf_index + 1,
            "a lower leaf must never pull the high-water mark down"
        );
    }

    /// The repair pass is re-runnable. It is gated by a marker file, but a crash
    /// between the last write and the marker means it runs again — that must be
    /// a no-op, not a corruption.
    #[test]
    fn backfill_is_idempotent() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut kp = mss::keygen(&[9u8; 32], 4).unwrap();
        let master_pk = kp.public_key();
        let sig = kp.sign(&[0x45u8; 32]).unwrap();

        let batch = batch_of(vec![reveal(
            htlc_predicate(&master_pk),
            Witness::ScriptInputs(vec![sig.to_bytes(), vec![0x01]]),
            6,
        )]);
        st.save_batch(1, &batch).unwrap();

        st.backfill_signature_archive().unwrap();
        let after_first = st.query_mss_leaf_index(&master_pk).unwrap();
        assert_eq!(
            after_first,
            sig.leaf_index + 1,
            "backfill must repair history the old writer skipped"
        );
        assert_eq!(st.query_spent_addresses(&batch).unwrap().len(), 1);

        st.backfill_signature_archive().unwrap();
        assert_eq!(
            st.query_mss_leaf_index(&master_pk).unwrap(),
            after_first,
            "re-running the backfill must change nothing"
        );
        assert_eq!(st.query_spent_addresses(&batch).unwrap().len(), 1);
    }

    /// Non-regression: the P2PK path must behave exactly as it did before.
    #[test]
    fn p2pk_spend_still_advances_the_leaf_index() {
        let dir = tempdir().unwrap();
        let st = store(&dir);

        let mut kp = mss::keygen(&[10u8; 32], 4).unwrap();
        let master_pk = kp.public_key();
        let sig = kp.sign(&[0x46u8; 32]).unwrap();

        let batch = batch_of(vec![reveal(
            Predicate::p2pk(&master_pk),
            Witness::sig(sig.to_bytes()),
            7,
        )]);
        st.burn_batch_addresses(&batch, 1).unwrap();

        assert_eq!(st.query_mss_leaf_index(&master_pk).unwrap(), sig.leaf_index + 1);
        assert!(st.query_spent_addresses(&batch).unwrap().contains_key(&sig.wots_pk));
    }
}
