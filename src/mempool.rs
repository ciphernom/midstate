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

    // ── Reveal path ─────────────────────────────────────────────────────

    fn state_with_committed_coin() -> (State, [u8; 32], [u8; 32], [u8; 32], [u8; 32], OutputData) {
        // Returns (state, seed, coin_id, input_salt, commit_salt, output)
        use crate::core::{wots, types::*};

        let mut state = empty_state();
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        let input_salt = hash(b"test salt");
        let coin_id = compute_coin_id(&address, 16, &input_salt);
        state.coins.insert(coin_id);

        let output = OutputData { address: hash(b"recipient"), value: 8, salt: [0x11; 32] };
        let commit_salt: [u8; 32] = [0x22; 32];
        let commitment = compute_commitment(&[coin_id], &[output.coin_id()], &commit_salt);

        // Mine PoW and add commitment
        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        let commit_tx = Transaction::Commit { commitment, spam_nonce: n };
        crate::core::transaction::apply_transaction(&mut state, &commit_tx).unwrap();

        (state, seed, coin_id, input_salt, commit_salt, output)
    }

    fn make_reveal_tx(seed: &[u8; 32], value: u64, input_salt: [u8; 32], commit_salt: [u8; 32], output: OutputData) -> Transaction {
        use crate::core::wots;

        let owner_pk = wots::keygen(seed);
        let address = compute_address(&owner_pk);
        let coin_id = compute_coin_id(&address, value, &input_salt);

        let commitment = compute_commitment(&[coin_id], &[output.coin_id()], &commit_salt);
        let sig = wots::sign(seed, &commitment);

        Transaction::Reveal {
            inputs: vec![InputReveal { owner_pk, value, salt: input_salt }],
            signatures: vec![wots::sig_to_bytes(&sig)],
            outputs: vec![output],
            salt: commit_salt,
        }
    }

    #[test]
    fn mempool_accepts_valid_reveal() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) = state_with_committed_coin();
        let mut mp = Mempool::new();
        let tx = make_reveal_tx(&seed, 16, input_salt, commit_salt, output);
        assert!(mp.add(tx, &state).is_ok());
        assert_eq!(mp.len(), 1);
    }

    #[test]
    fn mempool_rejects_duplicate_input() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) = state_with_committed_coin();
        let mut mp = Mempool::new();
        let tx = make_reveal_tx(&seed, 16, input_salt, commit_salt, output);
        mp.add(tx.clone(), &state).unwrap();
        assert!(mp.add(tx, &state).is_err());
    }

    #[test]
    fn mempool_rejects_low_fee_reveal() {
        let mut state = empty_state();
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = crate::core::wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        let input_salt = [0u8; 32];
        let coin_id = compute_coin_id(&address, 1, &input_salt);
        state.coins.insert(coin_id);

        let output = OutputData { address: hash(b"r"), value: 1, salt: [0; 32] };
        let commit_salt = [1u8; 32];
        let commitment = compute_commitment(&[coin_id], &[output.coin_id()], &commit_salt);

        let mut n = 0u64;
        loop {
            let h = hash_concat(&commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 { break; }
            n += 1;
        }
        crate::core::transaction::apply_transaction(&mut state, &Transaction::Commit { commitment, spam_nonce: n }).unwrap();

        let sig = crate::core::wots::sign(&seed, &commitment);
        // in=1, out=1 → fee=0 < MIN_REVEAL_FEE
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { owner_pk, value: 1, salt: input_salt }],
            signatures: vec![crate::core::wots::sig_to_bytes(&sig)],
            outputs: vec![output],
            salt: commit_salt,
        };

        let mut mp = Mempool::new();
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("Fee too low"));
    }

    // ── drain ───────────────────────────────────────────────────────────

    #[test]
    fn drain_returns_and_removes() {
        let state = empty_state();
        let mut mp = Mempool::new();
        for i in 0..5 {
            let commitment = hash(&(i as u64).to_le_bytes());
            mp.transactions.push(Transaction::Commit { commitment, spam_nonce: 0 });
            mp.seen_commitments.insert(commitment);
        }
        assert_eq!(mp.len(), 5);

        let drained = mp.drain(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(mp.len(), 2);
        assert_eq!(mp.seen_commitments.len(), 2);
    }

    #[test]
    fn drain_more_than_available() {
        let mut mp = Mempool::new();
        mp.transactions.push(Transaction::Commit { commitment: [1; 32], spam_nonce: 0 });
        mp.seen_commitments.insert([1; 32]);
        let drained = mp.drain(100);
        assert_eq!(drained.len(), 1);
        assert_eq!(mp.len(), 0);
    }

    // ── prune_invalid ───────────────────────────────────────────────────

    #[test]
    fn prune_removes_invalid_commits() {
        let state = empty_state();
        let mut mp = Mempool::new();
        let commitment = hash(b"prune test");
        // Directly push a commit with a bad PoW nonce
        mp.transactions.push(Transaction::Commit { commitment, spam_nonce: u64::MAX });
        mp.seen_commitments.insert(commitment);
        assert_eq!(mp.len(), 1);
        mp.prune_invalid(&state);
        assert_eq!(mp.len(), 0);
    }

    // ── re_add ──────────────────────────────────────────────────────────

    #[test]
    fn re_add_skips_invalid() {
        let state = empty_state();
        let mut mp = Mempool::new();
        // Bad PoW commit won't re-add
        let bad_tx = Transaction::Commit { commitment: hash(b"bad"), spam_nonce: u64::MAX };
        mp.re_add(vec![bad_tx], &state);
        assert_eq!(mp.len(), 0);
    }

    #[test]
    fn re_add_skips_duplicates() {
        let state = empty_state();
        let mut mp = Mempool::new();
        let commitment = hash(b"dup re-add");
        mp.transactions.push(Transaction::Commit { commitment, spam_nonce: 0 });
        mp.seen_commitments.insert(commitment);

        // Try to re_add the same commitment
        mp.re_add(vec![Transaction::Commit { commitment, spam_nonce: 0 }], &state);
        assert_eq!(mp.len(), 1); // still just 1
    }
}
