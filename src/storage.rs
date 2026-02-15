mod batch_store;
pub use batch_store::BatchStore;

use crate::core::State;
use anyhow::Result;
use redb::{Database, TableDefinition};
use std::path::Path;
use std::sync::Arc;

const STATE_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("state");
const MINING_SEED_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("mining_seed");

#[derive(Debug, Clone)]
pub struct Storage {
    db: Arc<Database>,
    pub batches: BatchStore,
}

impl Storage {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;

        let db = Database::create(path.join("state.redb"))?;

        // Initialize tables
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(STATE_TABLE)?;
            let _ = write_txn.open_table(MINING_SEED_TABLE)?;
        }
        write_txn.commit()?;

        let batches = BatchStore::new(path.join("batches"))?;

        Ok(Self { 
            db: Arc::new(db), 
            batches 
        })
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
                let state = bincode::deserialize(bytes.value())?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    pub fn save_mining_seed(&self, seed: &[u8; 32]) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(MINING_SEED_TABLE)?;
            table.insert("seed", seed.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn load_mining_seed(&self) -> Result<Option<[u8; 32]>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(MINING_SEED_TABLE)?;
        match table.get("seed")? {
            Some(bytes) => {
                let val = bytes.value();
                if val.len() != 32 {
                    anyhow::bail!("corrupt mining seed");
                }
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
}
