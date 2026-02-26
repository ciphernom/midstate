use crate::core::{Batch, BatchHeader};
use crate::core::filter::CompactFilter;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug, Clone)]
pub struct BatchStore {
    base_path: PathBuf,
}

impl BatchStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }
    pub fn base_path(&self) -> &PathBuf { &self.base_path }
    
    /// Height up to which checkpoints have already been pruned.
    /// Returns 0 if no pruning has occurred yet.
    pub fn pruned_up_to(&self) -> u64 {
        let marker = self.base_path.join("pruned_up_to");
        fs::read_to_string(marker)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Prune checkpoints from all batch and header files below `below_height`.
    ///
    /// Skips heights already pruned (tracked via a marker file). Each pruned
    /// batch drops ~32 KB of checkpoint data, reducing storage by ~98%.
    /// Pruned batches remain verifiable via full-chain recomputation.
    ///
    /// Returns the number of batches pruned in this call.
    pub fn prune_checkpoints(&self, below_height: u64) -> Result<u64> {
        let already_pruned = self.pruned_up_to();
        if below_height <= already_pruned {
            return Ok(0);
        }

        let mut count = 0u64;
        for height in already_pruned..below_height {
            let folder = height / 1000;
            let folder_path = self.base_path.join(format!("{:06}", folder));

            // Prune batch file
            let batch_path = folder_path.join(format!("batch_{}.bin", height));
            if batch_path.exists() {
                if let Ok(bytes) = fs::read(&batch_path) {
                    if let Ok(mut batch) = bincode::deserialize::<Batch>(&bytes) {
                        if !batch.extension.checkpoints.is_empty() {
                            batch.extension.checkpoints = vec![];
                            if let Ok(pruned_bytes) = bincode::serialize(&batch) {
                                fs::write(&batch_path, pruned_bytes)?;
                            }
                        }
                    }
                }
            }

            // Prune header file
            let hdr_path = folder_path.join(format!("header_{}.bin", height));
            if hdr_path.exists() {
                if let Ok(bytes) = fs::read(&hdr_path) {
                    if let Ok(mut header) = bincode::deserialize::<BatchHeader>(&bytes) {
                        if !header.extension.checkpoints.is_empty() {
                            header.extension.checkpoints = vec![];
                            if let Ok(pruned_bytes) = bincode::serialize(&header) {
                                fs::write(&hdr_path, pruned_bytes)?;
                            }
                        }
                    }
                }
            }

            count += 1;
        }

        // Update marker
        let marker = self.base_path.join("pruned_up_to");
        fs::write(marker, below_height.to_string())?;

        if count > 0 {
            tracing::info!(
                "Pruned checkpoints from {} batches (heights {}..{})",
                count, already_pruned, below_height
            );
        }

        Ok(count)
    }
    
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
        let batch = bincode::deserialize(&bytes)?;
        Ok(Some(batch))
    }

    /// Load a pre-computed header (falls back to full batch if header file missing)
    fn load_header(&self, height: u64) -> Result<Option<BatchHeader>> {
        let folder = height / 1000;
        let folder_path = self.base_path.join(format!("{:06}", folder));
        let hdr_path = folder_path.join(format!("header_{}.bin", height));

        if hdr_path.exists() {
            let bytes = fs::read(hdr_path)?;
            let header: BatchHeader = bincode::deserialize(&bytes)?;
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

    // ── Checkpoint pruning ──────────────────────────────────────────────

    #[test]
    fn prune_removes_checkpoints() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        for h in 0..5 {
            store.save(h, &dummy_batch(h)).unwrap();
        }

        // Before pruning: checkpoints present
        let batch = store.load(0).unwrap().unwrap();
        assert!(!batch.extension.checkpoints.is_empty());

        // Prune below height 3
        let count = store.prune_checkpoints(3).unwrap();
        assert_eq!(count, 3);

        // Heights 0-2: checkpoints gone
        for h in 0..3 {
            let batch = store.load(h).unwrap().unwrap();
            assert!(batch.extension.checkpoints.is_empty(),
                "height {} should be pruned", h);
        }

        // Heights 3-4: checkpoints intact
        for h in 3..5 {
            let batch = store.load(h).unwrap().unwrap();
            assert!(!batch.extension.checkpoints.is_empty(),
                "height {} should NOT be pruned", h);
        }
    }

    #[test]
    fn prune_also_prunes_headers() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(0, &dummy_batch(0)).unwrap();
        store.prune_checkpoints(1).unwrap();

        let headers = store.load_headers(0, 1).unwrap();
        assert_eq!(headers.len(), 1);
        assert!(headers[0].extension.checkpoints.is_empty());
    }

    #[test]
    fn prune_is_idempotent() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        for h in 0..3 {
            store.save(h, &dummy_batch(h)).unwrap();
        }

        store.prune_checkpoints(2).unwrap();
        let count = store.prune_checkpoints(2).unwrap();
        assert_eq!(count, 0, "second prune to same height should be no-op");
    }

    #[test]
    fn prune_incremental() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        for h in 0..10 {
            store.save(h, &dummy_batch(h)).unwrap();
        }

        store.prune_checkpoints(3).unwrap();
        assert_eq!(store.pruned_up_to(), 3);

        store.prune_checkpoints(7).unwrap();
        assert_eq!(store.pruned_up_to(), 7);

        // Heights 0-6 pruned, 7-9 intact
        for h in 0..7 {
            let batch = store.load(h).unwrap().unwrap();
            assert!(batch.extension.checkpoints.is_empty());
        }
        for h in 7..10 {
            let batch = store.load(h).unwrap().unwrap();
            assert!(!batch.extension.checkpoints.is_empty());
        }
    }

    #[test]
    fn prune_marker_persists() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("batches");
        let store = BatchStore::new(&path).unwrap();

        store.save(0, &dummy_batch(0)).unwrap();
        store.prune_checkpoints(1).unwrap();
        assert_eq!(store.pruned_up_to(), 1);

        // New store instance reads the marker
        let store2 = BatchStore::new(&path).unwrap();
        assert_eq!(store2.pruned_up_to(), 1);
    }

    #[test]
    fn pruned_batches_still_load_correctly() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        let batch = dummy_batch(42);
        store.save(0, &batch).unwrap();
        store.prune_checkpoints(1).unwrap();

        let loaded = store.load(0).unwrap().unwrap();
        assert_eq!(loaded.prev_midstate, batch.prev_midstate);
        assert_eq!(loaded.extension.nonce, batch.extension.nonce);
        assert_eq!(loaded.extension.final_hash, batch.extension.final_hash);
        assert!(loaded.extension.checkpoints.is_empty());
    }

    #[test]
    fn pruned_batch_passes_full_chain_verification() {
        use crate::core::extension::verify_extension;

        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        let nonce = 42u64;
        let batch = dummy_batch(nonce);
        let midstate = batch.header().post_tx_midstate;

        // Verify before pruning (spot-check path)
        assert!(verify_extension(
            midstate, &batch.extension, &[0xff; 32]
        ).is_ok());

        store.save(0, &batch).unwrap();
        store.prune_checkpoints(1).unwrap();
        let pruned = store.load(0).unwrap().unwrap();

        // Verify after pruning (full-chain path)
        assert!(verify_extension(
            midstate, &pruned.extension, &[0xff; 32]
        ).is_ok());
    }

    #[test]
    fn prune_skips_missing_heights() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        // Save only height 0 and 2 (gap at 1)
        store.save(0, &dummy_batch(0)).unwrap();
        store.save(2, &dummy_batch(2)).unwrap();

        // Should not panic on missing height 1
        let count = store.prune_checkpoints(3).unwrap();
        assert_eq!(count, 3); // iterates 0,1,2 — silently skips missing

        let b0 = store.load(0).unwrap().unwrap();
        assert!(b0.extension.checkpoints.is_empty());
        let b2 = store.load(2).unwrap().unwrap();
        assert!(b2.extension.checkpoints.is_empty());
    }

    #[test]
    fn prune_storage_savings() {
        let dir = tempdir().unwrap();
        let store = BatchStore::new(dir.path().join("batches")).unwrap();

        store.save(0, &dummy_batch(0)).unwrap();

        let folder = dir.path().join("batches/000000");
        let batch_path = folder.join("batch_0.bin");
        let header_path = folder.join("header_0.bin");

        let pre_batch = std::fs::metadata(&batch_path).unwrap().len();
        let pre_header = std::fs::metadata(&header_path).unwrap().len();

        store.prune_checkpoints(1).unwrap();

        let post_batch = std::fs::metadata(&batch_path).unwrap().len();
        let post_header = std::fs::metadata(&header_path).unwrap().len();

        // Pruned files should be dramatically smaller
        assert!(post_batch < pre_batch / 2,
            "batch should shrink: {} -> {}", pre_batch, post_batch);
        assert!(post_header < pre_header / 2,
            "header should shrink: {} -> {}", pre_header, post_header);
    }
}
