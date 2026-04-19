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


fn deserialize_state(bytes: &[u8]) -> Result<State> {
    bincode::deserialize::<State>(bytes).map_err(|e| anyhow::anyhow!("State deserialization failed: {}", e))
}

fn deserialize_state_limited(bytes: &[u8]) -> Result<State> {
    use bincode::Options;
    let opts = bincode::DefaultOptions::new().with_limit(50_000_000); 
    opts.deserialize::<State>(bytes).map_err(|e| anyhow::anyhow!("Snapshot deserialization failed: {}", e))
}

#[derive(Debug, Clone)]
pub struct Storage {
    db: Arc<Database>,
    pub batches: BatchStore,
}

impl Storage {
    
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
    pub fn load_state_snapshot(&self, height: u64) -> Result<Option<State>> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        let path = snapshot_dir.join(format!("state_{}.bin", height));
        
        if path.exists() {
            let bytes = std::fs::read(&path)?;
            let mut state = deserialize_state_limited(&bytes)?;
            state.coins.rebuild_tree();
            state.commitments.rebuild_tree();
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
                let mut state = deserialize_state(bytes.value())?;
                state.coins.rebuild_tree();
                state.commitments.rebuild_tree();
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
                if let crate::core::Transaction::Reveal { inputs, witnesses, outputs, salt } = tx {
                    let input_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                    let output_hashes: Vec<[u8; 32]> = outputs.iter()
                        .map(|o| o.hash_for_commitment())
                        .collect();
                    let commitment = crate::core::compute_commitment(&input_ids, &output_hashes, salt);

                    for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                        let crate::core::types::Witness::ScriptInputs(wit_inputs) = witness;
                        if let Some(sig) = wit_inputs.first() {
                            if sig.len() == crate::core::wots::SIG_SIZE {
                                let addr = input.predicate.address();
                                spent_table.insert(&addr, &commitment)?;
                            } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                                spent_table.insert(&mss_sig.wots_pk, &commitment)?;
                                
                                // O(1) MSS leaf index tracker
                                if let Some(master_pk) = input.predicate.owner_pk() {
                                    let next = mss_sig.leaf_index + 1;
                                    let current = mss_idx_table.get(&master_pk)?.map(|v: redb::AccessGuard<'_, u64>| v.value()).unwrap_or(0);
                                    if next > current {
                                        mss_idx_table.insert(&master_pk, next)?;
                                    }
                                }
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
            if let crate::core::Transaction::Reveal { inputs, witnesses, .. } = tx {
                for (input, witness) in inputs.iter().zip(witnesses.iter()) {
                    let Witness::ScriptInputs(wit_inputs) = witness;
                    if let Some(sig) = wit_inputs.first() {
                        if sig.len() == SIG_SIZE {
                            // Standard WOTS query
                            let addr = input.predicate.address();
                            if let Some(existing) = table.get(&addr)? {
                                result.insert(addr, *existing.value());
                            }
                        } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                            // NEW: MSS leaf query
                            if let Some(existing) = table.get(&mss_sig.wots_pk)? {
                                result.insert(mss_sig.wots_pk, *existing.value());
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
                        // Standard WOTS query
                        let addr = input.predicate.address();
                        if let Some(existing) = table.get(&addr)? {
                            result.insert(addr, *existing.value());
                        }
                    } else if let Ok(mss_sig) = crate::core::mss::MssSignature::from_bytes(sig) {
                        // NEW: MSS leaf query
                        if let Some(existing) = table.get(&mss_sig.wots_pk)? {
                            result.insert(mss_sig.wots_pk, *existing.value());
                        }
                    }
                }
            }
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
