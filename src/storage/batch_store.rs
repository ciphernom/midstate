use crate::core::Batch;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::fs;

pub struct BatchStore {
    base_path: PathBuf,
}

impl BatchStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;
        Ok(Self { base_path })
    }
    
    /// Save a batch
    pub fn save(&self, height: u64, batch: &Batch) -> Result<()> {
        let folder = height / 1000; // 1000 batches per folder
        let folder_path = self.base_path.join(format!("{:06}", folder));
        fs::create_dir_all(&folder_path)?;
        
        let file_path = folder_path.join(format!("batch_{}.bin", height));
        let bytes = bincode::serialize(batch)?;
        fs::write(file_path, bytes)?;
        
        Ok(())
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
    
    /// Get highest batch we have
    pub fn highest(&self) -> Result<u64> {
        let mut max = 0u64;
        
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                for file in fs::read_dir(path)? {
                    let file = file?;
                    let name = file.file_name();
                    let name_str = name.to_string_lossy();
                    
                    if let Some(height_str) = name_str.strip_prefix("batch_").and_then(|s| s.strip_suffix(".bin")) {
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
// ============================================================
// ADD THIS ENTIRE BLOCK at the bottom of src/storage/batch_store.rs
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use crate::core::extension::create_extension;
    use tempfile::tempdir;

    fn dummy_batch(nonce: u64) -> Batch {
        let ms = hash(&nonce.to_le_bytes());
        let ext = create_extension(ms, nonce);
        Batch {
            prev_midstate: ms,
            transactions: vec![],
            extension: ext,
            coinbase: vec![],
            timestamp: 1000 + nonce,
            target: [0xff; 32],
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
