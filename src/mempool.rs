use crate::core::{State, Transaction};
use crate::core::transaction::validate_transaction;
use anyhow::Result;
use std::collections::HashSet;

const MAX_MEMPOOL_SIZE: usize = 10_000;
const MAX_PENDING_COMMITS: usize = 1_000;
const MIN_REVEAL_FEE: u64 = 1;

pub struct Mempool {
    transactions: Vec<Transaction>,
    seen_inputs: HashSet<[u8; 32]>,
    seen_commitments: HashSet<[u8; 32]>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: Vec::new(),
            seen_inputs: HashSet::new(),
            seen_commitments: HashSet::new(),
        }
    }

    pub fn add(&mut self, tx: Transaction, state: &State) -> Result<()> {
        
        // DoS protection
        if self.transactions.len() >= MAX_MEMPOOL_SIZE {
            anyhow::bail!("Mempool full");
        }
        match &tx {
            Transaction::Commit { .. } => {
                if self.seen_commitments.len() >= MAX_PENDING_COMMITS {
                    anyhow::bail!("Too many pending commits");
                }
            }
            Transaction::Reveal { .. } => {
                if tx.fee() < MIN_REVEAL_FEE {
                    anyhow::bail!("Fee too low (minimum: {})", MIN_REVEAL_FEE);
                }
            }
        }
        
        validate_transaction(state, &tx)?;

        match &tx {
            Transaction::Commit { commitment, .. } => {
                if self.seen_commitments.contains(commitment) {
                    anyhow::bail!("Commitment already in mempool");
                }
            }
            Transaction::Reveal { .. } => {
                for input in tx.input_coin_ids() {
                    if self.seen_inputs.contains(&input) {
                        anyhow::bail!("Transaction input already in mempool");
                    }
                }
            }
        }

        match &tx {
            Transaction::Commit { commitment, .. } => {
                self.seen_commitments.insert(*commitment);
            }
            Transaction::Reveal { .. } => {
                for input in tx.input_coin_ids() {
                    self.seen_inputs.insert(input);
                }
            }
        }
        self.transactions.push(tx);

        tracing::debug!("Added transaction to mempool (size: {})", self.transactions.len());

        Ok(())
    }

    pub fn re_add(&mut self, txs: Vec<Transaction>, state: &State) {
        let mut restored = 0usize;
        for tx in txs {
            if validate_transaction(state, &tx).is_err() {
                continue;
            }

            let dominated = match &tx {
                Transaction::Commit { commitment, .. } => self.seen_commitments.contains(commitment),
                Transaction::Reveal { .. } => {
                    tx.input_coin_ids().iter().any(|i| self.seen_inputs.contains(i))
                }
            };
            if dominated {
                continue;
            }

            match &tx {
                Transaction::Commit { commitment, .. } => {
                    self.seen_commitments.insert(*commitment);
                }
                Transaction::Reveal { .. } => {
                    for input in tx.input_coin_ids() {
                        self.seen_inputs.insert(input);
                    }
                }
            }
            self.transactions.push(tx);
            restored += 1;
        }

        if restored > 0 {
            tracing::info!("Restored {} transactions to mempool", restored);
        }
    }

    pub fn drain(&mut self, max: usize) -> Vec<Transaction> {
        let count = max.min(self.transactions.len());
        let drained: Vec<_> = self.transactions.drain(..count).collect();

        for tx in &drained {
            match tx {
                Transaction::Commit { commitment, .. } => {
                    self.seen_commitments.remove(commitment);
                }
                Transaction::Reveal { .. } => {
                    for input in tx.input_coin_ids() {
                        self.seen_inputs.remove(&input);
                    }
                }
            }
        }

        drained
    }

    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    pub fn transactions(&self) -> &[Transaction] {
        &self.transactions
    }

    pub fn prune_invalid(&mut self, state: &State) {
        let mut inputs_to_remove = Vec::new();
        let mut commitments_to_remove = Vec::new();

        for tx in &self.transactions {
            if validate_transaction(state, tx).is_err() {
                match tx {
                    Transaction::Commit { commitment, .. } =>{
                        commitments_to_remove.push(*commitment);
                    }
                    Transaction::Reveal { .. } => {
                        inputs_to_remove.extend(tx.input_coin_ids());
                    }
                }
            }
        }

        if !inputs_to_remove.is_empty() || !commitments_to_remove.is_empty() {
            tracing::info!(
                "Pruning invalid transactions from mempool (inputs: {}, commitments: {})",
                inputs_to_remove.len(),
                commitments_to_remove.len()
            );

            self.transactions.retain(|tx| {
                let should_remove = match tx {
                    Transaction::Commit { commitment, .. } => {
                        commitments_to_remove.contains(commitment)
                    }
                    Transaction::Reveal { .. } => {
                        let inputs = tx.input_coin_ids();
                        inputs.iter().any(|input| inputs_to_remove.contains(input))
                    }
                };

                if should_remove {
                    match tx {
                        Transaction::Commit { commitment, .. } => {
                            self.seen_commitments.remove(commitment);
                        }
                        Transaction::Reveal { .. } => {
                            for input in tx.input_coin_ids() {
                                self.seen_inputs.remove(&input);
                            }
                        }
                    }
                    false
                } else {
                    true
                }
            });
        }
    }
}
