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
