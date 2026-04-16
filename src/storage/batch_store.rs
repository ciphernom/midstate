//! Database-backed storage for Blocks (Batches), Headers, and Compact Filters.
//!
//! # Architecture
//! Historically, this module stored one file per block, header, and filter. At 1 block 
//! per minute, this resulted in ~1.5 million files per year, which guarantees **Inode Exhaustion** 
//! on standard `ext4` filesystems (like those used on Raspberry Pis) in roughly 1.3 years, 
//! fatally bricking the OS.
//!
//! This module now uses **Redb** (a pure-Rust, Copy-On-Write B-Tree database) to store 
//! the entire chain in a single file. This provides:
//! 1. **1 Inode usage** regardless of chain length.
//! 2. **ACID compliance**, completely eliminating the need for complex `.tmp` file renaming during reorgs.
//! 3. **O(1) Tip resolution**, bypassing slow directory scans on startup.

use crate::core::{Batch, BatchHeader};
use crate::core::filter::CompactFilter;
use anyhow::Result;
use redb::{Database, ReadableTable};
use std::path::PathBuf;
use std::sync::Arc;
use crate::core::types::{Predicate, Witness, OutputData}; 

// ═══════════════════════════════════════════════════════════════════════════
//  LEGACY BINCODE MIGRATIONS
// ═══════════════════════════════════════════════════════════════════════════
// Bincode relies on strict structural positioning. When the protocol is upgraded 
// (e.g., adding `state_root` to blocks, or `commitment` to InputReveals for State Threads),
// older blocks saved on disk will fail to deserialize.
// We maintain these legacy structs to catch deserialization failures, read the old bytes 
// perfectly, and dynamically map them into the current active protocol structs.

#[derive(Clone, Debug, serde::Deserialize)]
struct LegacyInputReveal {
    pub predicate: Predicate,
    pub value: u64,
    pub salt: [u8; 32],
}

impl LegacyInputReveal {
    fn into_current(self) -> crate::core::InputReveal {
        crate::core::InputReveal {
            predicate: self.predicate,
            value: self.value,
            salt: self.salt,
            commitment: None, // Sentinel value for pre-State Thread inputs
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
enum LegacyTransaction {
    Commit { commitment: [u8; 32], spam_nonce: u64 },
    Reveal { inputs: Vec<LegacyInputReveal>, witnesses: Vec<Witness>, outputs: Vec<OutputData>, salt: [u8; 32] },
}

impl LegacyTransaction {
    fn into_current(self) -> crate::core::Transaction {
        match self {
            Self::Commit { commitment, spam_nonce } => crate::core::Transaction::Commit { commitment, spam_nonce },
            Self::Reveal { inputs, witnesses, outputs, salt } => crate::core::Transaction::Reveal {
                inputs: inputs.into_iter().map(|i| i.into_current()).collect(),
                witnesses, outputs, salt,
            },
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct LegacyBatch {
    pub prev_midstate: [u8; 32],
    pub transactions: Vec<LegacyTransaction>,
    pub extension: crate::core::Extension,
    #[serde(default)]
    pub coinbase: Vec<crate::core::CoinbaseOutput>,
    pub timestamp: u64,
    pub target: [u8; 32],
}

impl LegacyBatch {
    fn into_current(self) -> crate::core::Batch {
        crate::core::Batch {
            prev_midstate: self.prev_midstate,
            transactions: self.transactions.into_iter().map(|t| t.into_current()).collect(),
            extension: self.extension,
            coinbase: self.coinbase,
            timestamp: self.timestamp,
            target: self.target,
            state_root: [0u8; 32], // Sentinel for blocks mined before state_root activation
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
struct LegacyBatchHeader {
    pub height: u64,
    pub prev_midstate: [u8; 32],
    pub post_tx_midstate: [u8; 32],
    pub extension: crate::core::Extension,
    pub timestamp: u64,
    pub target: [u8; 32],
}

impl LegacyBatchHeader {
    fn into_current(self) -> crate::core::BatchHeader {
        crate::core::BatchHeader {
            height: self.height,
            prev_midstate: self.prev_midstate,
            post_tx_midstate: self.post_tx_midstate,
            extension: self.extension,
            timestamp: self.timestamp,
            target: self.target,
            state_root: [0u8; 32], 
        }
    }
}

/// Attempts to deserialize a Batch. If the current schema fails, falls back to the legacy schema.
fn deserialize_batch_with_migration(bytes: &[u8], height: u64) -> Result<crate::core::Batch> {
    match bincode::deserialize::<crate::core::Batch>(bytes) {
        Ok(batch) => Ok(batch),
        Err(_) => {
            let legacy: LegacyBatch = bincode::deserialize(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize batch {}: {}", height, e))?;
            Ok(legacy.into_current())
        }
    }
}

/// Attempts to deserialize a Header. If the current schema fails, falls back to the legacy schema.
fn deserialize_header_with_migration(bytes: &[u8], height: u64) -> Result<crate::core::BatchHeader> {
    match bincode::deserialize::<crate::core::BatchHeader>(bytes) {
        Ok(hdr) => Ok(hdr),
        Err(_) => {
            let legacy: LegacyBatchHeader = bincode::deserialize(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize header {}: {}", height, e))?;
            Ok(legacy.into_current())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  BATCH STORE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct BatchStore {
    db: Arc<Database>,
    /// The old flat-file path. Preserved so we can clean it up after migration, 
    /// and so upstream logic can locate the adjacent `snapshots/` directory.
    legacy_fs_path: PathBuf, 
}

impl BatchStore {
    /// Initializes the store and automatically triggers a seamless migration 
    /// from the old filesystem structure to the single-file database.
    pub fn new(db: Arc<Database>, legacy_fs_path: PathBuf) -> Result<Self> {
        let store = Self { db, legacy_fs_path };
        store.migrate_from_fs()?;
        Ok(store)
    }

    /// Safely scans for the highest legacy block if the `highest_height` marker file is missing.
    /// Guarantees no blocks are lost during migration from very old nodes.
    fn legacy_highest(&self) -> u64 {
        let marker = self.legacy_fs_path.join("highest_height");
        if marker.exists() {
            if let Ok(s) = std::fs::read_to_string(&marker) {
                if let Ok(v) = s.trim().parse::<u64>() {
                    return v;
                }
            }
        }

        // Fallback: Manually scan directories (O(N) operation, only runs once ever)
        let mut max = 0u64;
        if let Ok(entries) = std::fs::read_dir(&self.legacy_fs_path) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Ok(files) = std::fs::read_dir(entry.path()) {
                        for file in files.flatten() {
                            let name = file.file_name();
                            let name_str = name.to_string_lossy();
                            if let Some(h_str) = name_str.strip_prefix("batch_").and_then(|s| s.strip_suffix(".bin")) {
                                if let Ok(h) = h_str.parse::<u64>() {
                                    max = max.max(h);
                                }
                            }
                        }
                    }
                }
            }
        }
        max
    }

    /// The one-time migration engine. Reads all flat files and imports them into Redb.
    ///
    /// Memory Safety: We process the migration in 500-block transactions.
    /// Redb holds dirty pages in memory before `commit()` is called. At 100KB per block, 
    /// 500 blocks consumes ~50 MB of RAM, which is completely safe for a 512 MB Raspberry Pi.
    ///
    /// Atomic Deletion: The old flat files are ONLY deleted if the entire loop finishes 
    /// flawlessly. If power is lost mid-migration, the node simply resumes writing over 
    /// the DB from the flat files on next boot.
    fn migrate_from_fs(&self) -> Result<()> {
        if !self.legacy_fs_path.exists() { return Ok(()); }

        let highest = self.legacy_highest();

        // Edge case: If the directory exists but is completely empty, just clean it up.
        let folder0 = self.legacy_fs_path.join("000000");
        if highest == 0 && !folder0.join("batch_0.bin").exists() {
            let _ = std::fs::remove_dir_all(&self.legacy_fs_path);
            return Ok(());
        }

        tracing::info!("Migrating {} blocks to Database. Please wait...", highest + 1);

        let mut write_txn = self.db.begin_write()?;
        let mut batches_table = write_txn.open_table(super::BATCHES_TABLE)?;
        let mut headers_table = write_txn.open_table(super::HEADERS_TABLE)?;
        let mut filters_table = write_txn.open_table(super::FILTERS_TABLE)?;

        for h in 0..=highest {
            let folder = h / 1000;
            let folder_path = self.legacy_fs_path.join(format!("{:06}", folder));
            
            let b_path = folder_path.join(format!("batch_{}.bin", h));
            if b_path.exists() { batches_table.insert(h, std::fs::read(&b_path)?.as_slice())?; }
            
            let h_path = folder_path.join(format!("header_{}.bin", h));
            if h_path.exists() { headers_table.insert(h, std::fs::read(&h_path)?.as_slice())?; }
            
            let f_path = folder_path.join(format!("filter_{}.bin", h));
            if f_path.exists() { filters_table.insert(h, std::fs::read(&f_path)?.as_slice())?; }

            // 500-block boundary commit to prevent OOM kills on constrained hardware
            if h > 0 && h % 500 == 0 {
                drop(batches_table); drop(headers_table); drop(filters_table);
                write_txn.commit()?;
                
                write_txn = self.db.begin_write()?;
                batches_table = write_txn.open_table(super::BATCHES_TABLE)?;
                headers_table = write_txn.open_table(super::HEADERS_TABLE)?;
                filters_table = write_txn.open_table(super::FILTERS_TABLE)?;
                tracing::info!("Migrated {} / {} blocks...", h, highest);
            }
        }
        
        drop(batches_table); drop(headers_table); drop(filters_table);
        write_txn.commit()?;

        // ONLY delete the old file mess if the entire database migration succeeded perfectly.
        let _ = std::fs::remove_dir_all(&self.legacy_fs_path);
        tracing::info!("Migration to Database complete! Node is fully optimized.");
        Ok(())
    }

    /// Returns the legacy root path. Preserved so `node.rs` can resolve sibling directories 
    /// like `snapshots/`.
    pub fn base_path(&self) -> &PathBuf {
        &self.legacy_fs_path 
    }

    /// Deletes all blocks, headers, and filters at or above the given height.
    /// Used during a deep reorg to cleanly prune the abandoned fork.
    pub fn truncate(&self, new_tip_height: u64) -> Result<()> {
        let highest = self.highest()?;
        if new_tip_height >= highest { return Ok(()); }

        let write_txn = self.db.begin_write()?;
        {
            let mut b = write_txn.open_table(super::BATCHES_TABLE)?;
            let mut h = write_txn.open_table(super::HEADERS_TABLE)?;
            let mut f = write_txn.open_table(super::FILTERS_TABLE)?;

            for height in new_tip_height..=highest {
                b.remove(height)?;
                h.remove(height)?;
                f.remove(height)?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Persists a new block to the database. Atomically writes the full Batch, 
    /// its extracted Header, and its generated Neutrino Filter.
    pub fn save(&self, height: u64, batch: &Batch) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut b = write_txn.open_table(super::BATCHES_TABLE)?;
            let mut h = write_txn.open_table(super::HEADERS_TABLE)?;
            let mut f = write_txn.open_table(super::FILTERS_TABLE)?;

            b.insert(height, bincode::serialize(batch)?.as_slice())?;

            let mut header = batch.header();
            header.height = height;
            h.insert(height, bincode::serialize(&header)?.as_slice())?;

            let filter = CompactFilter::build(batch);
            f.insert(height, filter.data.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Loads a Compact Filter for the given height. Used by light clients to scan the chain.
    pub fn load_filter(&self, height: u64) -> Result<Option<Vec<u8>>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::FILTERS_TABLE)?;
        if let Some(guard) = table.get(height)? {
            Ok(Some(guard.value().to_vec()))
        } else {
            Ok(None)
        }
    }

    /// Loads a full Batch (Block) from the database. Handles legacy schema migration automatically.
    pub fn load(&self, height: u64) -> Result<Option<Batch>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::BATCHES_TABLE)?;
        if let Some(guard) = table.get(height)? {
            let batch = deserialize_batch_with_migration(guard.value(), height)?;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Loads an isolated Header. If the block was mined before isolated headers were supported,
    /// it falls back to loading the full batch and extracting the header in memory.
    fn load_header(&self, height: u64) -> Result<Option<BatchHeader>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::HEADERS_TABLE)?;
        if let Some(guard) = table.get(height)? {
            let header = deserialize_header_with_migration(guard.value(), height)?;
            Ok(Some(header))
        } else {
            // Fallback for pre-header migration blocks
            if let Some(batch) = self.load(height)? {
                let mut header = batch.header();
                header.height = height;
                Ok(Some(header))
            } else {
                Ok(None)
            }
        }
    }

    /// Loads a sequential array of headers. Stops early if a gap is encountered.
    pub fn load_headers(&self, start: u64, end: u64) -> Result<Vec<BatchHeader>> {
        if end <= start { return Ok(Vec::new()); }
        let mut headers = Vec::with_capacity((end - start) as usize);
        for h in start..end {
            if let Some(header) = self.load_header(h)? {
                headers.push(header);
            } else {
                break;
            }
        }
        Ok(headers)
    }

    /// Loads a sequential array of full batches. 
    pub fn load_range(&self, start: u64, end: u64) -> Result<Vec<(u64, Batch)>> {
        let mut batches = Vec::new();
        for height in start..end {
            match self.load(height) {
                Ok(Some(batch)) => batches.push((height, batch)),
                Ok(None) => break,
                Err(_) => break,
            }
        }
        Ok(batches)
    }

    /// O(1) B-Tree right-most edge traversal
    pub fn highest(&self) -> Result<u64> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::BATCHES_TABLE)?;
        let max_val = if let Some(last) = table.last()? {
            last.0.value()
        } else {
            0
        };
        Ok(max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use redb::TableDefinition;

    // We redefine these here just for the tests so they don't break if `super::` is messy in the test scope
    const BATCHES_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("batches");
    const HEADERS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("headers");
    const FILTERS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("filters");

    fn dummy_batch(nonce: u64) -> Batch {
        let ms = crate::core::types::hash(&nonce.to_le_bytes());
        let state_root = [0u8; 32];
        let mining_ms = crate::core::types::hash_concat(&ms, &state_root);
        let ext = crate::core::extension::create_extension(mining_ms, nonce);
        Batch {
            prev_midstate: ms,
            transactions: vec![],
            extension: ext,
            coinbase: vec![],
            timestamp: 1000 + nonce,
            target: [0xff; 32],
            state_root,
        }
    }

    fn setup_store(dir: &std::path::Path) -> BatchStore {
        let db_path = dir.join("test.redb");
        let db = Database::create(&db_path).unwrap();
        let write_txn = db.begin_write().unwrap();
        write_txn.open_table(BATCHES_TABLE).unwrap();
        write_txn.open_table(HEADERS_TABLE).unwrap();
        write_txn.open_table(FILTERS_TABLE).unwrap();
        write_txn.commit().unwrap();
        BatchStore::new(Arc::new(db), dir.join("batches")).unwrap()
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let store = setup_store(dir.path());

        let batch = dummy_batch(42);
        store.save(0, &batch).unwrap();
        let loaded = store.load(0).unwrap().unwrap();
        assert_eq!(loaded.prev_midstate, batch.prev_midstate);
        assert_eq!(loaded.extension.nonce, batch.extension.nonce);
    }

    #[test]
    fn highest_batch_fetches_btree_edge() {
        let dir = tempdir().unwrap();
        let store = setup_store(dir.path());

        assert_eq!(store.highest().unwrap(), 0);
        store.save(5, &dummy_batch(5)).unwrap();
        store.save(100, &dummy_batch(100)).unwrap();
        assert_eq!(store.highest().unwrap(), 100); // 100 is higher than 5
    }

    #[test]
    fn truncate_removes_higher_blocks() {
        let dir = tempdir().unwrap();
        let store = setup_store(dir.path());

        store.save(0, &dummy_batch(0)).unwrap();
        store.save(1, &dummy_batch(1)).unwrap();
        store.save(2, &dummy_batch(2)).unwrap();

        assert_eq!(store.highest().unwrap(), 2);
        store.truncate(1).unwrap(); // Should delete 1 and 2
        
        assert_eq!(store.highest().unwrap(), 0);
        assert!(store.load(1).unwrap().is_none());
        assert!(store.load(2).unwrap().is_none());
    }
}
