use crate::core::State;
use anyhow::Result;
use std::path::Path;

pub struct Storage {
    db: sled::Db,
}

impl Storage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path)?;
        Ok(Self { db })
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
}
