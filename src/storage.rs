mod batch_store;
pub use batch_store::BatchStore;

use crate::core::State;
use crate::core::mmr::{MerkleMountainRange, UtxoAccumulator};
use anyhow::Result;
use redb::{Database, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("state");
const MINING_SEED_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("mining_seed");
/// Maps spent WOTS address -> the commitment hash that legitimately spent it.
/// Allows safe replay of the exact same transaction during chain reorgs while
/// permanently blocking any *different* transaction from reusing the key.
const SPENT_ADDRESSES_TABLE: TableDefinition<&[u8; 32], &[u8; 32]> =
    TableDefinition::new("spent_addresses");

/// V1 state layout (depth: u64). Used only for one-time migration from
/// pre-u128 databases. Identical field order to the old `State` so that
/// bincode positional decoding works.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct LegacyState {
    pub midstate: [u8; 32],
    pub coins: UtxoAccumulator,
    pub commitments: UtxoAccumulator,
    pub depth: u64,
    pub target: [u8; 32],
    pub height: u64,
    pub timestamp: u64,
    #[serde(default)]
    pub commitment_heights: im::HashMap<[u8; 32], u64>,
    #[serde(default)]
    pub chain_mmr: MerkleMountainRange,
}

impl LegacyState {
    fn into_current(self) -> State {
        State {
            midstate: self.midstate,
            coins: self.coins,
            commitments: self.commitments,
            depth: self.depth as u128,
            target: self.target,
            height: self.height,
            timestamp: self.timestamp,
            commitment_heights: self.commitment_heights,
            chain_mmr: self.chain_mmr,
        }
    }
}

/// Try deserializing as current State, fall back to LegacyState (u64 depth)
/// and convert. Returns the deserialized state and whether migration occurred.
fn deserialize_state_with_migration(bytes: &[u8]) -> Result<(State, bool)> {
    // Try current format first
    match bincode::deserialize::<State>(bytes) {
        Ok(state) => Ok((state, false)),
        Err(_) => {
            // Fall back to legacy format
            let legacy: LegacyState = bincode::deserialize(bytes)
                .map_err(|e| anyhow::anyhow!(
                    "State deserialization failed for both current and legacy formats: {}", e
                ))?;
            tracing::info!(
                "Migrating state from v1 (depth u64={}) to v2 (depth u128={})",
                legacy.depth, legacy.depth as u128
            );
            Ok((legacy.into_current(), true))
        }
    }
}

/// Same as above but with size-limited deserialization (for untrusted snapshots)
fn deserialize_state_with_migration_limited(bytes: &[u8]) -> Result<(State, bool)> {
    use bincode::Options;
    let opts = bincode::DefaultOptions::new().with_limit(500_000_000);
    // Try current format: first with DefaultOptions (matching old load path),
    // then standard bincode (matching save path)
    if let Ok(state) = opts.deserialize::<State>(bytes) {
        return Ok((state, false));
    }
    if let Ok(state) = bincode::deserialize::<State>(bytes) {
        return Ok((state, false));
    }
    // Fall back to legacy formats
    if let Ok(legacy) = opts.deserialize::<LegacyState>(bytes) {
        tracing::info!("Migrating snapshot from v1 to v2 (depth {})", legacy.depth);
        return Ok((legacy.into_current(), true));
    }
    let legacy: LegacyState = bincode::deserialize(bytes)
        .map_err(|e| anyhow::anyhow!(
            "Snapshot deserialization failed for all format combinations: {}", e
        ))?;
    tracing::info!("Migrating snapshot from v1 to v2 (depth {})", legacy.depth);
    Ok((legacy.into_current(), true))
}

#[derive(Debug, Clone)]
pub struct Storage {
    db: Arc<Database>,
    pub batches: BatchStore,
}

impl Storage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        // redb acquires an exclusive file lock.  If a previous node process
        // is still shutting down (race between kill and restart), the lock
        // may not yet be released.  Retry with back-off before giving up.
        let db_path = path.join("state.redb");
        let mut last_err = None;
        for attempt in 0..10 {
            match Database::create(&db_path) {
                Ok(db) => {
                    if attempt > 0 {
                        tracing::info!("Database lock acquired after {} retries", attempt);
                    }
                    // Initialize tables
                    let write_txn = db.begin_write()?;
                    {
                        let _ = write_txn.open_table(STATE_TABLE)?;
                        let _ = write_txn.open_table(MINING_SEED_TABLE)?;
                        let _ = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
                    }
                    write_txn.commit()?;

                    let batches = BatchStore::new(path.join("batches"))?;

                    let storage = Self {
                        db: Arc::new(db),
                        batches,
                    };

                    // State-aware WAL recovery: load the committed height from
                    // redb BEFORE recovering .tmp files so we know which belong
                    // to a committed reorg (promote) vs an aborted one (delete).
                    let committed_height = storage.load_committed_height()?;
                    storage.batches.recover_wal(committed_height)?;

                    return Ok(storage);
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
    pub fn load_state_snapshot(&self, height: u64) -> Result<Option<State>> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        let path = snapshot_dir.join(format!("state_{}.bin", height));
        
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let (mut state, migrated) = deserialize_state_with_migration_limited(&bytes)?;
            state.coins.rebuild_tree();
            state.commitments.rebuild_tree();
            // Re-save migrated snapshots so they load faster next time
            if migrated {
                let new_bytes = bincode::serialize(&state)?;
                std::fs::write(&path, new_bytes)?;
                tracing::info!("Re-saved snapshot at height {} in v2 format", height);
            }
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

    pub fn load_state(&self) -> Result<Option<State>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(STATE_TABLE)?;
        match table.get("current")? {
            Some(bytes) => {
                let (mut state, migrated) = deserialize_state_with_migration(bytes.value())?;
                state.coins.rebuild_tree();
                state.commitments.rebuild_tree();
                // If we migrated from legacy format, write back in new format
                // so subsequent loads don't need migration.
                drop(table);
                drop(read_txn);
                if migrated {
                    tracing::info!("Re-saving state in v2 format after migration");
                    self.save_state(&state)?;
                }
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    /// Load only the committed state height from redb (no tree rebuilds).
    /// Used by WAL recovery to determine which .tmp files to promote vs delete.
    fn load_committed_height(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(STATE_TABLE)?;
        match table.get("current")? {
            Some(bytes) => {
                let (state, _) = deserialize_state_with_migration(bytes.value())?;
                Ok(state.height)
            }
            None => Ok(0),
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

    /// Burn every WOTS input address from a committed batch, mapping each to the
    /// commitment hash that authorised the spend.
    ///
    /// - Ignores non-WOTS inputs (MSS etc.) by checking witness size == SIG_SIZE.
    /// - No-op for blocks below WOTS_REUSE_ACTIVATION_HEIGHT.
    /// - Idempotent: reorg replays of the same batch write the same value.
    pub fn burn_batch_addresses(&self, batch: &crate::core::Batch, block_height: u64) -> Result<()> {
        use crate::core::types::{WOTS_REUSE_ACTIVATION_HEIGHT, Witness, compute_commitment};
        use crate::core::wots::SIG_SIZE;

        if block_height < WOTS_REUSE_ACTIVATION_HEIGHT {
            return Ok(());
        }

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(SPENT_ADDRESSES_TABLE)?;
            for tx in &batch.transactions {
                if let crate::core::Transaction::Reveal { inputs, witnesses, outputs, salt } = tx {
                    let input_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                    let output_hashes: Vec<[u8; 32]> = outputs.iter()
                        .map(|o| o.hash_for_commitment())
                        .collect();
                    let commitment = compute_commitment(&input_ids, &output_hashes, salt);

                for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                        let Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            // Only burn WOTS keys. MSS and other script types
                            // have different witness sizes and must not be burned.
                            if sig.len() == SIG_SIZE {
                                let addr = input.predicate.address();
                                table.insert(&addr, &commitment)?;
                            }
                        }
                    }
                }
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Build a pre-flight oracle for a batch: returns a map of
    /// `address -> prior_commitment` for every WOTS address in the batch
    /// that already exists in the spent-address table.
    ///
    /// The caller uses this to detect reuse: an address is being reused if it
    /// is present in the map AND the prior commitment differs from the current one.
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
            if let crate::core::Transaction::Reveal { inputs, witnesses, .. } = tx {
                for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                    let Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            if sig.len() == SIG_SIZE {
                                let addr = input.predicate.address();
                                if let Some(existing) = table.get(&addr)? {
                                    result.insert(addr, *existing.value());
                                }
                            }
                        }
                }
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

        if let crate::core::Transaction::Reveal { inputs, witnesses, .. } = tx {
            for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                let Witness::ScriptInputs(wit_inputs) = witness;
                    if let Some(sig) = wit_inputs.first() {
                        if sig.len() == SIG_SIZE {
                            let addr = input.predicate.address();
                            if let Some(existing) = table.get(&addr)? {
                                result.insert(addr, *existing.value());
                            }
                        }
                    }
                }
            }
        
        Ok(result)
    }

}
