mod batch_store;
pub use batch_store::BatchStore;

use crate::core::State;
use anyhow::Result;
use std::path::Path;

pub struct Storage {
    db: sled::Db,
    batches: BatchStore,
}

impl Storage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        
        let db = sled::open(path.join("state"))?;
        let batches = BatchStore::new(path.join("batches"))?;
        
        Ok(Self { db, batches })
    }
    
    pub fn save_state(&self, state: &State) -> Result<()> {
        let bytes = bincode::serialize(state)?;
        self.db.insert(b"state", bytes)?;
        self.db.flush()?;
        Ok(())
    }
    
    pub fn load_state(&self) -> Result<Option<State>> {
        match self.db.get(b"state")? {
            Some(bytes) => {
                let state = bincode::deserialize(&bytes)?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }
    
    // Batch storage methods
    pub fn save_batch(&self, height: u64, batch: &crate::core::Batch) -> Result<()> {
        self.batches.save(height, batch)
    }
    
    pub fn load_batch(&self, height: u64) -> Result<Option<crate::core::Batch>> {
        self.batches.load(height)
    }
    
    pub fn load_batches(&self, start: u64, end: u64) -> Result<Vec<crate::core::Batch>> {
        self.batches.load_range(start, end)
    }
    
    pub fn highest_batch(&self) -> Result<u64> {
        self.batches.highest()
    }
}
