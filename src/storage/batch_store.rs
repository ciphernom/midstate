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
    pub fn new(db: Arc<Database>, legacy_fs_path: PathBuf) -> Result<Self> {
        let store = Self { db, legacy_fs_path };
        Ok(store)
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
            let batch: Batch = bincode::deserialize(guard.value())?;
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
            let header: BatchHeader = bincode::deserialize(guard.value())?;
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
        
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::HEADERS_TABLE)?;
        
        let mut headers = Vec::with_capacity((end - start) as usize);
        for h in start..end {
            if let Some(guard) = table.get(h)? {
                let header: BatchHeader = bincode::deserialize(guard.value())?;
                headers.push(header);
            } else {
                // Fallback for extremely old pre-migration blocks
                drop(table);
                drop(read_txn);
                for fh in h..end {
                    if let Some(header) = self.load_header(fh)? {
                        headers.push(header);
                    } else {
                        break;
                    }
                }
                break;
            }
        }
        Ok(headers)
    }

    pub fn load_range(&self, start: u64, end: u64) -> Result<Vec<(u64, Batch)>> {
        if end <= start { return Ok(Vec::new()); }
        
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(super::BATCHES_TABLE)?;
        
        let mut batches = Vec::new();
        for height in start..end {
            if let Some(guard) = table.get(height)? {
                let batch: Batch = bincode::deserialize(guard.value())?;
                batches.push((height, batch));
            } else {
                break;
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
        
        let candidate_header = BatchHeader {
            height: 0,
            prev_header_hash: [0u8; 32],
            prev_midstate: ms,
            post_tx_midstate: ms, // dummy
            extension: crate::core::Extension { nonce: 0, final_hash: [0u8; 32] },
            timestamp: 1000 + nonce,
            target: [0xff; 32],
            state_root,
        };
        let mining_hash = crate::core::types::compute_header_hash(&candidate_header);
        let ext = crate::core::extension::create_extension(mining_hash, nonce);
        
        Batch {
            prev_midstate: ms,
            prev_header_hash: [0u8; 32],
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
