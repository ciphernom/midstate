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

        // redb acquires an exclusive file lock.  If a previous node process
        // is still shutting down (race between kill and restart), the lock
        // may not yet be released.  Retry with back-off before giving up.
        let db_path = path.join("state.redb");
        let mut last_err = None;
        for attempt in 0..10 {
            match Database::create(&db_path) {
                Ok(db) => {
                    if attempt > 0 {
                        tracing::info!("Database lock acquired after {} retries", attempt);
                    }
                    // Initialize tables
                    let write_txn = db.begin_write()?;
                    {
                        let _ = write_txn.open_table(STATE_TABLE)?;
                        let _ = write_txn.open_table(MINING_SEED_TABLE)?;
                    }
                    write_txn.commit()?;

                    let batches = BatchStore::new(path.join("batches"))?;

                    return Ok(Self {
                        db: Arc::new(db),
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
    pub fn save_state_snapshot(&self, height: u64, state: &State) -> Result<()> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        std::fs::create_dir_all(&snapshot_dir)?;
        
        let path = snapshot_dir.join(format!("state_{}.bin", height));
        let bytes = bincode::serialize(state)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads a historical snapshot to serve to a peer.
    pub fn load_state_snapshot(&self, height: u64) -> Result<Option<State>> {
        let snapshot_dir = self.batches.base_path().parent().unwrap().join("snapshots");
        let path = snapshot_dir.join(format!("state_{}.bin", height));
        
        if path.exists() {
            let bytes = std::fs::read(path)?;
            let mut state: State = bincode::deserialize(&bytes)?;
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
                let mut state: State = bincode::deserialize(bytes.value())?;
                state.coins.rebuild_tree();
                state.commitments.rebuild_tree();
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

    /// Prune checkpoints from batches below the given height.
    /// See [`BatchStore::prune_checkpoints`] for details.
    pub fn prune_checkpoints(&self, below_height: u64) -> Result<u64> {
        self.batches.prune_checkpoints(below_height)
    }
}
