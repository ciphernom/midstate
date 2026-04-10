use crate::core::{Batch, BatchHeader};
use crate::core::filter::CompactFilter;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug, Clone)]
pub struct BatchStore {
    base_path: PathBuf,
}


use crate::core::types::{Predicate, Witness, OutputData}; // Added for legacy mapping

#[derive(Clone, Debug, serde::Deserialize)]
struct LegacyInputReveal {
    pub predicate: Predicate,
    pub value: u64,
    pub salt: [u8; 32],
    // Missing: pub commitment: Option<[u8; 32]>
}

impl LegacyInputReveal {
    fn into_current(self) -> crate::core::InputReveal {
        crate::core::InputReveal {
            predicate: self.predicate,
            value: self.value,
            salt: self.salt,
            commitment: None, // Sentinel for old inputs
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
enum LegacyTransaction {
    Commit {
        commitment: [u8; 32],
        spam_nonce: u64,
    },
    Reveal {
        inputs: Vec<LegacyInputReveal>,
        witnesses: Vec<Witness>,
        outputs: Vec<OutputData>,
        salt: [u8; 32],
    },
}

impl LegacyTransaction {
    fn into_current(self) -> crate::core::Transaction {
        match self {
            Self::Commit { commitment, spam_nonce } => crate::core::Transaction::Commit { commitment, spam_nonce },
            Self::Reveal { inputs, witnesses, outputs, salt } => crate::core::Transaction::Reveal {
                inputs: inputs.into_iter().map(|i| i.into_current()).collect(),
                witnesses,
                outputs,
                salt,
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
    // Missing: pub state_root: [u8; 32]
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
            state_root: [0u8; 32], // Sentinel for old blocks
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
    // Missing: pub state_root: [u8; 32]
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
            state_root: [0u8; 32], // Sentinel for old headers
        }
    }
}


fn deserialize_batch_with_migration(bytes: &[u8], height: u64) -> Result<crate::core::Batch> {
    match bincode::deserialize::<crate::core::Batch>(bytes) {
        Ok(batch) => Ok(batch),
        Err(_) => {
            let legacy: LegacyBatch = bincode::deserialize(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize batch at height {} in all formats: {}", height, e))?;
            tracing::info!("Migrated legacy block at height {}", height);
            Ok(legacy.into_current())
        }
    }
}

fn deserialize_header_with_migration(bytes: &[u8], height: u64) -> Result<crate::core::BatchHeader> {
    match bincode::deserialize::<crate::core::BatchHeader>(bytes) {
        Ok(hdr) => Ok(hdr),
        Err(_) => {
            let legacy: LegacyBatchHeader = bincode::deserialize(bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize header at height {} in all formats: {}", height, e))?;
            Ok(legacy.into_current())
        }
    }
}

impl BatchStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        
        // NOTE: WAL recovery is deferred to recover_wal() which must be called
        // AFTER loading the committed state height from redb. Calling it here
        // would blindly promote .tmp files from an aborted reorg, corrupting
        // the block data on disk.
        
        Ok(Self { base_path })
    }

    /// Deletes all batch, header, and filter files at or above the given height,
    /// and updates the highest_height marker.
    pub fn truncate(&self, new_tip_height: u64) -> Result<()> {
        let highest = self.highest().unwrap_or(0);
        if new_tip_height >= highest { return Ok(()); }
        
        for h in new_tip_height..=highest {
            let folder = h / 1000;
            let folder_path = self.base_path.join(format!("{:06}", folder));
            
            let _ = fs::remove_file(folder_path.join(format!("batch_{}.bin", h)));
            let _ = fs::remove_file(folder_path.join(format!("header_{}.bin", h)));
            let _ = fs::remove_file(folder_path.join(format!("filter_{}.bin", h)));
            let _ = fs::remove_file(folder_path.join(format!("batch_{}.tmp", h)));
            let _ = fs::remove_file(folder_path.join(format!("header_{}.tmp", h)));
        }
        
        // Update the marker so highest() is O(1) again
        let marker = self.base_path.join("highest_height");
        let _ = fs::write(marker, new_tip_height.saturating_sub(1).to_string());
        Ok(())
    }

    /// State-aware WAL recovery. Must be called after loading state from redb.
    ///
    /// - If a `.tmp` file's height <= committed_height, the DB committed but
    ///   the rename was interrupted (crash at Step 3) → promote to `.bin`.
    /// - If a `.tmp` file's height > committed_height, the DB never committed
    ///   (crash at Step 1) → delete the orphaned `.tmp` file.
    pub fn recover_wal(&self, committed_height: u64) -> Result<()> {
        if !self.base_path.exists() { return Ok(()); }
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            if entry.path().is_dir() {
                for file in fs::read_dir(entry.path())? {
                    let file = file?;
                    let path = file.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("tmp") {
                        // Extract height from filename (e.g. "batch_105.tmp" -> 105)
                        let height = path.file_stem()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.split('_').last())
                            .and_then(|s| s.parse::<u64>().ok());

                        match height {
                            Some(h) if h <= committed_height => {
                                // DB committed this height — finish the rename
                                let mut bin_path = path.clone();
                                bin_path.set_extension("bin");
                                fs::rename(&path, &bin_path)?;
                                tracing::info!("WAL recovery: promoted {:?} (height {} <= committed {})",
                                    path.file_name().unwrap(), h, committed_height);
                            }
                            Some(h) => {
                                // DB never committed this height — discard
                                fs::remove_file(&path)?;
                                tracing::info!("WAL recovery: deleted orphan {:?} (height {} > committed {})",
                                    path.file_name().unwrap(), h, committed_height);
                            }
                            None => {
                                // Can't parse height — leave it alone
                                tracing::warn!("WAL recovery: skipping {:?} (could not parse height)", path);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // PHASE 1: Write to disk safely without overriding live data
    pub fn save_tmp(&self, height: u64, batch: &Batch) -> Result<()> {
        let folder = height / 1000;
        let folder_path = self.base_path.join(format!("{:06}", folder));
        std::fs::create_dir_all(&folder_path)?;
        
        let batch_tmp = folder_path.join(format!("batch_{}.tmp", height));
        let hdr_tmp = folder_path.join(format!("header_{}.tmp", height));
        
        std::fs::write(&batch_tmp, bincode::serialize(batch)?)?;
        let mut header = batch.header();
        header.height = height;
        std::fs::write(&hdr_tmp, bincode::serialize(&header)?)?;
        
        Ok(())
    }

    // PHASE 2: Promote temporary files to live files
    pub fn commit_tmp(&self, height: u64) -> Result<()> {
        let folder = height / 1000;
        let folder_path = self.base_path.join(format!("{:06}", folder));
        
        let batch_tmp = folder_path.join(format!("batch_{}.tmp", height));
        let batch_bin = folder_path.join(format!("batch_{}.bin", height));
        let hdr_tmp = folder_path.join(format!("header_{}.tmp", height));
        let hdr_bin = folder_path.join(format!("header_{}.bin", height));
        
        if batch_tmp.exists() { fs::rename(batch_tmp, batch_bin)?; }
        if hdr_tmp.exists() { fs::rename(hdr_tmp, hdr_bin)?; }
        Ok(())
    }
    
    pub fn base_path(&self) -> &PathBuf { &self.base_path }
    

    
    /// Save a batch (and its lightweight header for fast sync)
    pub fn save(&self, height: u64, batch: &Batch) -> Result<()> {
        let folder = height / 1000; // 1000 batches per folder
        let folder_path = self.base_path.join(format!("{:06}", folder));
        fs::create_dir_all(&folder_path)?;
        
        let file_path = folder_path.join(format!("batch_{}.bin", height));
        let bytes = bincode::serialize(batch)?;
        fs::write(&file_path, bytes)?;

        // Write header separately — avoids deserializing full batch (with
        // transactions + WOTS sigs) when peers request header chains.
        let mut header = batch.header();
        header.height = height;
        let hdr_path = folder_path.join(format!("header_{}.bin", height));
        let hdr_bytes = bincode::serialize(&header)?;
        fs::write(hdr_path, hdr_bytes)?;

        // --- Save Compact Filter ---
        let filter = CompactFilter::build(batch);
        let filter_path = folder_path.join(format!("filter_{}.bin", height));
        fs::write(filter_path, filter.data)?;

        // Update the highest-height marker so `highest()` is O(1) instead of
        // scanning every file in every subdirectory.  Written last so that a
        // crash mid-save cannot advance the marker past a fully-written block.
        let marker = self.base_path.join("highest_height");
        let current_highest = fs::read_to_string(&marker)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0);
        if height >= current_highest {
            fs::write(&marker, height.to_string())?;
        }

        Ok(())
    }
 
    /// Load a filter
    pub fn load_filter(&self, height: u64) -> Result<Option<Vec<u8>>> {
        let folder = height / 1000;
        let filter_path = self.base_path
            .join(format!("{:06}", folder))
            .join(format!("filter_{}.bin", height));

        if filter_path.exists() {
            Ok(Some(fs::read(filter_path)?))
        } else {
            Ok(None)
        }
    }
    
    /// Load a batch
    pub fn load(&self, height: u64) -> Result<Option<Batch>> {
        let folder = height / 1000;
        let file_path = self.base_path
            .join(format!("{:06}", folder))
            .join(format!("batch_{}.bin", height));
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        let bytes = fs::read(file_path)?;
        let batch = deserialize_batch_with_migration(&bytes, height)?;
        Ok(Some(batch))
    }

    /// Load a pre-computed header (falls back to full batch if header file missing)
    fn load_header(&self, height: u64) -> Result<Option<BatchHeader>> {
        let folder = height / 1000;
        let folder_path = self.base_path.join(format!("{:06}", folder));
        let hdr_path = folder_path.join(format!("header_{}.bin", height));

        if hdr_path.exists() {
            let bytes = fs::read(hdr_path)?;
            let header = deserialize_header_with_migration(&bytes, height)?;
            return Ok(Some(header));
        }

        // Fallback: load full batch (for batches saved before this change)
        if let Some(batch) = self.load(height)? {
            let mut header = batch.header();
            header.height = height;
            Ok(Some(header))
        } else {
            Ok(None)
        }
    }

    /// Load headers for a range — uses lightweight header files when available
    pub fn load_headers(&self, start: u64, end: u64) -> Result<Vec<BatchHeader>> {
        if end <= start { return Ok(Vec::new()); }
        let mut headers = Vec::with_capacity((end - start) as usize);
        for h in start..end {
            if let Some(header) = self.load_header(h)? {
                headers.push(header);
            }
        }
        Ok(headers)
    }

    /// Get all batches from height range
    pub fn load_range(&self, start: u64, end: u64) -> Result<Vec<(u64, Batch)>> {

        let mut batches = Vec::new();
        
        for height in start..end {
            match self.load(height) {
                Ok(Some(batch)) => batches.push((height, batch)),

                Ok(None) => {
                    tracing::warn!("Gap in batch store at height {}, returning {} contiguous batches", height, batches.len());
                    break;
                }
                Err(e) => {
                    eprintln!("[WARN] Error loading batch at height {}: {}, continuing", height, e);
                    break;
                }
            }
        }
        
        Ok(batches)
    }
    
    /// Get highest batch we have.
    ///
    /// Fast path: reads the `highest_height` marker file written by `save()`,
    /// giving O(1) startup cost regardless of chain length.
    ///
    /// Fallback: if the marker doesn't exist (fresh node or pre-marker data
    /// directory), performs the original O(N) directory scan so that existing
    /// nodes work correctly without any migration step.
    pub fn highest(&self) -> Result<u64> {
        let marker = self.base_path.join("highest_height");
        if marker.exists() {
            let val = fs::read_to_string(&marker)?
                .trim()
                .parse::<u64>()
                .unwrap_or(0);
            return Ok(val);
        }

        // Fallback: full directory scan for nodes that predate the marker.
        let mut max = 0u64;
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                for file in fs::read_dir(path)? {
                    let file = file?;
                    let name = file.file_name();
                    let name_str = name.to_string_lossy();
                    if let Some(height_str) = name_str
                        .strip_prefix("batch_")
                        .and_then(|s| s.strip_suffix(".bin"))
                    {
                        if let Ok(height) = height_str.parse::<u64>() {
                            max = max.max(height);
                        }
                    }
                }
            }
        }
        Ok(max)
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use tempfile::tempdir;

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

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        let batch = dummy_batch(42);
        store.save(0, &batch).unwrap();
        let loaded = store.load(0).unwrap().unwrap();
        assert_eq!(loaded.prev_midstate, batch.prev_midstate);
        assert_eq!(loaded.timestamp, batch.timestamp);
        assert_eq!(loaded.extension.nonce, batch.extension.nonce);
    }

    #[test]
    fn load_nonexistent_returns_none() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();
        assert!(store.load(999).unwrap().is_none());
    }

    #[test]
    fn save_overwrite() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(0, &dummy_batch(1)).unwrap();
        store.save(0, &dummy_batch(2)).unwrap();
        let loaded = store.load(0).unwrap().unwrap();
        assert_eq!(loaded.extension.nonce, 2);
    }

    #[test]
    fn load_range_contiguous() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        for h in 0..5 {
            store.save(h, &dummy_batch(h)).unwrap();
        }

        let range = store.load_range(1, 4).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].0, 1);
        assert_eq!(range[2].0, 3);
    }

    #[test]
    fn load_range_stops_at_gap() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(0, &dummy_batch(0)).unwrap();
        store.save(1, &dummy_batch(1)).unwrap();
        // Gap at height 2
        store.save(3, &dummy_batch(3)).unwrap();

        let range = store.load_range(0, 4).unwrap();
        assert_eq!(range.len(), 2); // stops at gap
    }

    #[test]
    fn highest_batch() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        assert_eq!(store.highest().unwrap(), 0);

        store.save(5, &dummy_batch(5)).unwrap();
        store.save(100, &dummy_batch(100)).unwrap();
        store.save(50, &dummy_batch(50)).unwrap();

        assert_eq!(store.highest().unwrap(), 100);
    }

    // ── highest_height marker ───────────────────────────────────────────

    #[test]
    fn highest_uses_marker_after_save() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(0, &dummy_batch(0)).unwrap();
        store.save(7, &dummy_batch(7)).unwrap();
        store.save(3, &dummy_batch(3)).unwrap();

        // Marker should reflect the highest height saved, not insertion order.
        let marker_path = dir.path().join("batches").join("highest_height");
        assert!(marker_path.exists(), "marker file must be written by save()");

        assert_eq!(store.highest().unwrap(), 7);
    }

    #[test]
    fn highest_marker_not_regressed_by_lower_save() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(10, &dummy_batch(10)).unwrap();
        assert_eq!(store.highest().unwrap(), 10);

        // Saving a lower height (e.g. reorg fill-in) must not move the marker back.
        store.save(5, &dummy_batch(5)).unwrap();
        assert_eq!(store.highest().unwrap(), 10);
    }

    #[test]
    fn highest_falls_back_to_scan_without_marker() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        // Manually write a batch file without going through save(),
        // simulating a pre-marker data directory.
        let folder = dir.path().join("batches/000000");
        std::fs::create_dir_all(&folder).unwrap();
        let batch = dummy_batch(42);
        let bytes = bincode::serialize(&batch).unwrap();
        std::fs::write(folder.join("batch_42.bin"), bytes).unwrap();

        // No marker file exists — must fall back to directory scan.
        let marker = dir.path().join("batches/highest_height");
        assert!(!marker.exists());
        assert_eq!(store.highest().unwrap(), 42);
    }

    #[test]
    fn highest_marker_persists_across_instances() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batches");
        let store = BatchStore::new(&path).unwrap();

        store.save(99, &dummy_batch(99)).unwrap();
        drop(store);

        // New instance should read marker, not scan.
        let store2 = BatchStore::new(&path).unwrap();
        assert_eq!(store2.highest().unwrap(), 99);
    }

    #[test]
    fn cross_folder_boundaries() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        // Heights 999 and 1000 go into different folders
        store.save(999, &dummy_batch(999)).unwrap();
        store.save(1000, &dummy_batch(1000)).unwrap();

        assert!(store.load(999).unwrap().is_some());
        assert!(store.load(1000).unwrap().is_some());
        assert_eq!(store.highest().unwrap(), 1000);
    }

   
}
