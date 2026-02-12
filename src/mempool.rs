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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use crate::core::mmr::UtxoAccumulator;

    fn empty_state() -> State {
        State {
            midstate: [0u8; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height: 1,
            timestamp: 1000,
            commitment_heights: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn mempool_rejects_bad_commit_pow() {
        let mut mp = Mempool::new();
        let state = empty_state();
        let commitment = hash(b"mempool test");
        // Find a bad nonce
        let mut bad = 0u64;
        loop {
            let h = hash_concat(&commitment, &bad.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) != 0x0000 { break; }
            bad += 1;
        }
        let tx = Transaction::Commit { commitment, spam_nonce: bad };
        assert!(mp.add(tx, &state).is_err());
        assert_eq!(mp.len(), 0);
    }

    #[test]
    fn mempool_accepts_good_commit_pow() {
        let mut mp = Mempool::new();
        let state = empty_state();
        let commitment = hash(b"mempool test good");
        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        let tx = Transaction::Commit { commitment, spam_nonce: n };
        assert!(mp.add(tx, &state).is_ok());
        assert_eq!(mp.len(), 1);
    }
    
#[test]
    fn mempool_full_rejects() {
        let state = empty_state();
        let mut mp = Mempool::new();
        // Fill to capacity — bypass add() so no PoW needed
        for i in 0..MAX_MEMPOOL_SIZE {
            let commitment = hash(&(i as u64).to_le_bytes());
            mp.transactions.push(Transaction::Commit { commitment, spam_nonce: 0 });
            mp.seen_commitments.insert(commitment);
        }
        assert_eq!(mp.len(), MAX_MEMPOOL_SIZE);

        let extra = hash(b"one more");
        let mut n = 0u64;
        loop {
            let h = hash_concat(&extra, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        let tx = Transaction::Commit { commitment: extra, spam_nonce: n };
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("Mempool full"));
    }

    #[test]
    fn max_pending_commits_enforced() {
        let state = empty_state();
        let mut mp = Mempool::new();
        for i in 0..MAX_PENDING_COMMITS {
            let commitment = hash(&(i as u64).to_le_bytes());
            mp.transactions.push(Transaction::Commit { commitment, spam_nonce: 0 });
            mp.seen_commitments.insert(commitment);
        }

        let extra = hash(b"commit overflow");
        let mut n = 0u64;
        loop {
            let h = hash_concat(&extra, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        let tx = Transaction::Commit { commitment: extra, spam_nonce: n };
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("Too many pending commits"));
    }

    #[test]
    fn duplicate_commitment_rejected() {
        let mut mp = Mempool::new();
        let state = empty_state();
        let commitment = hash(b"dup test");
        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        let tx = Transaction::Commit { commitment, spam_nonce: n };
        assert!(mp.add(tx.clone(), &state).is_ok());
        assert!(mp.add(tx, &state).is_err());
    }

}
