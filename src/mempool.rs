use crate::core::{hash, State, Transaction};
use crate::core::transaction::validate_transaction;
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

pub struct Mempool {
    transactions: Vec<Transaction>,
    seen_inputs: HashSet<[u8; 32]>,
    storage: sled::Db,
}

impl Mempool {
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let storage = sled::open(db_path)?;
        
        // Load persisted transactions
        let mut transactions = Vec::new();
        let mut seen_inputs = HashSet::new();
        
        for item in storage.iter() {
            let (_, value) = item?;
            let tx: Transaction = bincode::deserialize(&value)?;
            for input in tx.input_coins() {
                seen_inputs.insert(input);
            }
            transactions.push(tx);
        }
        
        tracing::info!("Loaded {} transactions from mempool storage", transactions.len());
        
        Ok(Self {
            transactions,
            seen_inputs,
            storage,
        })
    }
    
    pub fn add(&mut self, tx: Transaction, state: &State) -> Result<()> {
        // Validate against current state
        validate_transaction(state, &tx)?;
        
        // Check not already in mempool
        for input in tx.input_coins() {
            if self.seen_inputs.contains(&input) {
                anyhow::bail!("Transaction input already in mempool");
            }
        }
        
        // Persist
        let tx_bytes = bincode::serialize(&tx)?;
        let tx_hash = hash(&tx_bytes);
        self.storage.insert(&tx_hash[..], tx_bytes)?;
        
        for input in tx.input_coins() {
            self.seen_inputs.insert(input);
        }
        self.transactions.push(tx);
        
        tracing::debug!("Added transaction to mempool (size: {})", self.transactions.len());
        
        Ok(())
    }
    
    pub fn drain(&mut self, max: usize) -> Vec<Transaction> {
        let count = max.min(self.transactions.len());
        let drained: Vec<_> = self.transactions.drain(..count).collect();
        
        // Remove from storage
        for tx in &drained {
            for input in tx.input_coins() {
                self.seen_inputs.remove(&input);
            }
            let tx_bytes = bincode::serialize(tx).unwrap();
            let tx_hash = hash(&tx_bytes);
            let _ = self.storage.remove(&tx_hash[..]);
        }
        
        drained
    }
    
    pub fn len(&self) -> usize {
        self.transactions.len()
    }
    
    pub fn transactions(&self) -> &[Transaction] {
        &self.transactions
    }
    
    /// Remove transactions that conflict with current state
    pub fn prune_invalid(&mut self, state: &State) {
        let mut to_remove = Vec::new();
        
        for tx in &self.transactions {
            if validate_transaction(state, tx).is_err() {
                to_remove.extend(tx.input_coins());
            }
        }
        
        if !to_remove.is_empty() {
            tracing::info!("Pruning {} invalid transactions from mempool", to_remove.len());
            
            self.transactions.retain(|tx| {
                let inputs = tx.input_coins();
                let should_remove = inputs.iter().any(|input| to_remove.contains(input));
                
                if should_remove {
                    for input in inputs {
                        self.seen_inputs.remove(&input);
                    }
                    let tx_bytes = bincode::serialize(tx).unwrap();
                    let tx_hash = hash(&tx_bytes);
                    let _ = self.storage.remove(&tx_hash[..]);
                    false
                } else {
                    true
                }
            });
        }
    }
}
