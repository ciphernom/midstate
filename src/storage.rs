mod batch_store;
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

/// Maps MSS master_pk -> highest (leaf_index + 1) seen on-chain.
/// Gives O(1) lookup for the /mss_state endpoint instead of scanning
/// every block from genesis.
const MSS_LEAF_INDEX_TABLE: TableDefinition<&[u8; 32], u64> =
    TableDefinition::new("mss_leaf_index");

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
        .with_fixint_encoding(); // <--- ADD THIS LINE

    let mut state = match strict.deserialize::<State>(bytes) {
        Ok(s) => s,
        Err(strict_err) => match legacy::deserialize_legacy_state(bytes) {
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
                // Both parses failed. The strict-new error is usually the
                // more informative one (e.g. "Slice had bytes remaining"
                // immediately tells the operator what's going on); the
                // legacy error is logged for diagnostics in case the file
                // turns out to be a third unknown format.
                tracing::error!(
                    "Legacy migration also failed: {}",
                    legacy_err
                );
                return Err(anyhow::anyhow!(
                    "State deserialization failed: {}",
                    strict_err
                ));
            }
        },
    };

    // Rebuild the SMT caches under the chain-height-implied hashing mode.

    let v2 = crate::core::types::is_v2_at(state.height);
    state.coins.rebuild_tree(v2);
    state.commitments.rebuild_tree(v2);

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
                        let _ = write_txn.open_table(MSS_LEAF_INDEX_TABLE)?;
                        let _ = write_txn.open_table(BATCHES_TABLE)?;
                        let _ = write_txn.open_table(HEADERS_TABLE)?;
                        let _ = write_txn.open_table(FILTERS_TABLE)?;
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

    /// Reverts the burning of WOTS addresses from an abandoned chain segment.
    /// Called during a reorg to prevent "ghost" database entries.
    pub fn unburn_batch_addresses(&self, batch: &crate::core::Batch) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut spent_table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
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
                                } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                    spent_table.remove(&mss_sig.wots_pk)?;
                                }
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
                            } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                spent_table.remove(&mss_sig.wots_pk)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Burn every WOTS input address from a committed batch, mapping each to the
    /// commitment hash that authorised the spend.
    ///
    /// - For standard WOTS: burns the address hash.
    /// - For MSS (post-activation): burns the specific leaf's WOTS public key.
    /// - Idempotent: reorg replays of the same batch write the same value.
    pub fn burn_batch_addresses(&self, batch: &crate::core::Batch, _block_height: u64) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut spent_table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
            let mut mss_idx_table = write_txn.open_table(MSS_LEAF_INDEX_TABLE)?;

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
                                } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                    spent_table.insert(&mss_sig.wots_pk, &commitment)?;
                                    if let Some(master_pk) = input.predicate.owner_pk() {
                                        let next = mss_sig.leaf_index + 1;
                                        let current = mss_idx_table.get(&master_pk)?.map(|v: redb::AccessGuard<'_, u64>| v.value()).unwrap_or(0);
                                        if next > current { mss_idx_table.insert(&master_pk, next)?; }
                                    }
                                }
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
                            } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                spent_table.insert(&mss_sig.wots_pk, &commitment)?;
                                if let Some(master_pk) = inputs[0].predicate.owner_pk() {
                                    let next = mss_sig.leaf_index + 1;
                                    let current = mss_idx_table.get(&master_pk)?.map(|v: redb::AccessGuard<'_, u64>| v.value()).unwrap_or(0);
                                    if next > current { mss_idx_table.insert(&master_pk, next)?; }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        write_txn.commit()?;
        Ok(())
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
                            } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                            }
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
                        } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                            if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                        }
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
                        } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                            if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                        }
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
                    } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                        if let Some(existing) = table.get(&mss_sig.wots_pk)? { result.insert(mss_sig.wots_pk, *existing.value()); }
                    }
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
