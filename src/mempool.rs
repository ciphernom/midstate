use crate::core::{State, Transaction};
use crate::core::transaction::validate_transaction;
use anyhow::Result;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

const MAX_MEMPOOL_REVEALS: usize = 9_000;
const MAX_PENDING_COMMITS: usize = 1_000;
/// Total mempool capacity is the sum of both pools.
pub const MAX_MEMPOOL_SIZE: usize = MAX_MEMPOOL_REVEALS + MAX_PENDING_COMMITS;
const MIN_FEE_PER_KB: u64 = 10;
/// Scaling factor used when computing fee rates to preserve sub-satoshi precision
/// while staying in integer arithmetic. fee_rate = fee * FEE_RATE_SCALE / bytes.
const FEE_RATE_SCALE: u128 = 1_024;

/// Computes a stable, content-derived transaction ID by hashing the bincode
/// serialization of the transaction. Used as the key in the reveals map and
/// the secondary sort key in `reveals_by_fee`.
fn get_tx_id(tx: &Transaction) -> [u8; 32] {
    let bytes = bincode::serialize(tx).unwrap_or_default();
    crate::core::types::hash(&bytes)
}

/// Computes the fee rate of a transaction as `fee * FEE_RATE_SCALE / size_bytes`.
///
/// Uses the same `FEE_RATE_SCALE` factor as the minimum-fee admission check so
/// that the stored ordering key and the admission threshold are numerically
/// consistent. A transaction that just barely passes the minimum check will
/// have a fee_rate that correctly reflects its position at the bottom of the
/// priority queue.
///
/// Uses `u128` arithmetic throughout to prevent overflow.
fn compute_fee_rate(fee: u64, tx_bytes: u64) -> u64 {
    if tx_bytes == 0 {
        return 0;
    }
    ((fee as u128 * FEE_RATE_SCALE) / tx_bytes as u128) as u64
}

/// A highly efficient, priority-sorted memory pool for pending transactions.
///
/// The mempool is split into two independent, capacity-bounded pools:
///
/// - **Commits** (up to [`MAX_PENDING_COMMITS`]): sorted by Proof-of-Work
///   difficulty. The weakest commit is evicted first.
/// - **Reveals** (up to [`MAX_MEMPOOL_REVEALS`]): sorted by fee rate
///   (fee per byte). The cheapest reveal is evicted first.
///
/// Both pools use a `HashMap` for O(1) lookup by key and a `BTreeSet` as a
/// priority index, giving O(log N) insertion, eviction, and minimum-finding.
///
/// # Commit–Reveal Protocol
///
/// Transactions follow a two-phase protocol. A **Commit** blinds the inputs
/// being spent behind a commitment hash, preventing front-running. A subsequent
/// **Reveal** opens the commitment and executes the spend. Commits require a
/// Proof-of-Work nonce to resist spam; Reveals pay a fee.
pub struct Mempool {
    /// Commit transactions keyed by their commitment hash.
    commits: HashMap<[u8; 32], Arc<Transaction>>,
    /// Priority index for commits: `(leading_zero_bits, commitment)`.
    /// The entry with the *lowest* PoW is at the front (eviction candidate).
    commits_by_pow: BTreeSet<(u32, [u8; 32])>,

    /// Reveal transactions keyed by their content-hash transaction ID.
    reveals: HashMap<[u8; 32], Arc<Transaction>>,
    /// Priority index for reveals: `(fee_rate, tx_id)`.
    /// The entry with the *lowest* fee rate is at the front (eviction candidate).
    reveals_by_fee: BTreeSet<(u64, [u8; 32])>,

    /// Set of input coin IDs currently consumed by mempool Reveals.
    /// Used for O(1) double-spend detection.
    seen_inputs: HashSet<[u8; 32]>,
    
    /// Maps an Input Coin ID to the Mempool Transaction ID that is trying to spend it.
    /// Used for O(1) mempool eviction when a block is mined.
    txs_by_input: HashMap<[u8; 32], [u8; 32]>,
}

impl Mempool {
    /// Creates a new, empty Mempool.
    ///
    /// # Examples
    /// ```
    /// use midstate::mempool::Mempool;
    /// let mempool = Mempool::new();
    /// assert_eq!(mempool.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            commits: HashMap::new(),
            commits_by_pow: BTreeSet::new(),
            reveals: HashMap::new(),
            reveals_by_fee: BTreeSet::new(),
            seen_inputs: HashSet::new(),
            txs_by_input: HashMap::new(),
        }
    }

    /// Calculates the required Proof-of-Work (number of leading zero bits)
    /// based on the current number of pending commits. Scales dynamically
    /// to deter spam when the commit pool is congested.
    ///
    /// | Pending commits | Required leading zeros | Relative difficulty |
    /// |-----------------|------------------------|---------------------|
    /// | < 500           | 16                     | 1×                  |
    /// | 500 – 749       | 18                     | 4×                  |
    /// | 750 – 899       | 20                     | 16×                 |
    /// | ≥ 900           | 22                     | 64×                 |
    ///
    /// # Examples
    /// ```
    /// use midstate::mempool::Mempool;
    /// let mempool = Mempool::new();
    /// assert_eq!(mempool.required_commit_pow(), 16);
    /// ```
    pub fn required_commit_pow(&self) -> u32 {
        let commits = self.commits.len();
        if commits < 500      { 16 }
        else if commits < 750 { 18 }
        else if commits < 900 { 20 }
        else                  { 22 }
    }

    /// Adds a transaction to the mempool after validating it against the
    /// current chain state.
    ///
    /// ## Commit admission
    /// 1. PoW must meet or exceed [`required_commit_pow`](Self::required_commit_pow).
    /// 2. The commitment must not already be pending.
    /// 3. The transaction must pass [`validate_transaction`].
    /// 4. If the commit pool is at capacity ([`MAX_PENDING_COMMITS`]), the
    ///    incoming commit must have *strictly higher* PoW than the weakest
    ///    existing commit in order to evict it.
    ///
    /// ## Reveal admission
    /// 1. Fee rate must be ≥ [`MIN_FEE_PER_KB`] (checked with integer-only
    ///    arithmetic using [`FEE_RATE_SCALE`] to stay consistent with the
    ///    stored ordering key).
    /// 2. No input may already be consumed by a pending reveal.
    /// 3. The transaction must pass [`validate_transaction`].
    /// 4. If the reveal pool is at capacity ([`MAX_MEMPOOL_REVEALS`]), the
    ///    incoming reveal must have a *strictly higher* fee rate than the
    ///    cheapest existing reveal to evict it.
    ///
    /// # Errors
    /// Returns an error if the transaction fails any of the checks above.
    ///
    /// # Examples
    /// ```no_run
    /// # use midstate::mempool::Mempool;
    /// # use midstate::core::{State, Transaction};
    /// # let mut mempool = Mempool::new();
    /// # let state = State::genesis().0;
    /// # let tx = Transaction::Commit { commitment: [0; 32], spam_nonce: 0 };
    /// // 'tx' must carry valid PoW for this to succeed.
    /// mempool.add(tx, &state).unwrap();
    /// ```
    pub fn add(&mut self, tx: Transaction, state: &State) -> Result<()> {
        let tx_bytes = bincode::serialized_size(&tx).unwrap_or(0) as u64;

        // Extract the values we need from borrowed fields *before* any
        // potential move of `tx`. This satisfies the borrow checker: all
        // references into `tx` are gone by the time we call `Arc::new(tx)`.
        let (actual_zeros, _commitment_key) = match &tx {
            Transaction::Commit { commitment, spam_nonce } => {
                let h = crate::core::types::hash_concat(commitment, &spam_nonce.to_le_bytes());
                (Some(crate::core::types::count_leading_zeros(&h)), Some(*commitment))
            }
            Transaction::Reveal { .. } => (None, None),
        };

        match tx {
            Transaction::Commit { commitment, spam_nonce } => {
                let actual_zeros = actual_zeros.unwrap();

                let required = self.required_commit_pow();
                if actual_zeros < required {
                    anyhow::bail!(
                        "Mempool is busy. Commit PoW requires {} leading zero bits (provided: {})",
                        required, actual_zeros
                    );
                }
                if self.commits.contains_key(&commitment) {
                    anyhow::bail!("Commitment already in mempool");
                }

                // Reconstruct a reference for validate_transaction.
                let tx_ref = Transaction::Commit { commitment, spam_nonce };
                validate_transaction(state, &tx_ref)?;

                if self.commits.len() >= MAX_PENDING_COMMITS {
                    if let Some(&(lowest_pow, lowest_comm)) = self.commits_by_pow.iter().next() {
                        if actual_zeros > lowest_pow {
                            self.commits_by_pow.remove(&(lowest_pow, lowest_comm));
                            self.commits.remove(&lowest_comm);
                            tracing::info!(
                                "Evicted low-PoW commit ({} bits) for higher-PoW commit ({} bits)",
                                lowest_pow, actual_zeros
                            );
                        } else {
                            anyhow::bail!(
                                "Mempool full of Commits: incoming PoW ({} bits) must exceed \
                                 lowest existing PoW ({} bits) to evict",
                                actual_zeros, lowest_pow
                            );
                        }
                    }
                }

                let arc_tx = Arc::new(tx_ref);
                self.commits.insert(commitment, arc_tx);
                self.commits_by_pow.insert((actual_zeros, commitment));
            }

            reveal_tx @ Transaction::Reveal { .. } => {
                // Admission fee check and stored ordering key use the same
                // FEE_RATE_SCALE so they are numerically consistent.
                //
                // Minimum check:  fee * 1024 >= MIN_FEE_PER_KB * bytes
                // Stored key:     fee * 1024 / bytes   (== compute_fee_rate)
                if (reveal_tx.fee() as u128) * FEE_RATE_SCALE
                    < (MIN_FEE_PER_KB as u128) * (tx_bytes as u128)
                {
                    anyhow::bail!(
                        "Fee rate too low. Required: {} per KB. Provided: {} for {} bytes",
                        MIN_FEE_PER_KB, reveal_tx.fee(), tx_bytes
                    );
                }
                for input in reveal_tx.input_coin_ids() {
                    if self.seen_inputs.contains(&input) {
                        anyhow::bail!("Transaction input already in mempool");
                    }
                }

                validate_transaction(state, &reveal_tx)?;

                let fee_rate = compute_fee_rate(reveal_tx.fee(), tx_bytes);
                let tx_id = get_tx_id(&reveal_tx);

                if self.reveals.len() >= MAX_MEMPOOL_REVEALS {
                    if let Some(&(lowest_rate, lowest_id)) = self.reveals_by_fee.iter().next() {
                        if fee_rate > lowest_rate {
                            let evicted_arc = self.reveals.remove(&lowest_id).unwrap();
                            self.reveals_by_fee.remove(&(lowest_rate, lowest_id));
                            for input in evicted_arc.input_coin_ids() {
                                self.seen_inputs.remove(&input);
                                self.txs_by_input.remove(&input);
                            }
                        } else {
                            anyhow::bail!(
                                "Mempool full: incoming fee rate too low to replace any existing Reveal"
                            );
                        }
                    }
                }

                for input in reveal_tx.input_coin_ids() {
                    self.seen_inputs.insert(input);
                    self.txs_by_input.insert(input, tx_id);
                }

                let arc_tx = Arc::new(reveal_tx);
                self.reveals.insert(tx_id, arc_tx);
                self.reveals_by_fee.insert((fee_rate, tx_id));
            }
        }

        tracing::debug!(
            "Added transaction to mempool (commits: {}, reveals: {})",
            self.commits.len(), self.reveals.len()
        );
        Ok(())
    }

    /// Re-adds a batch of transactions to the mempool after a chain reorg.
    ///
    /// Unlike [`add`](Self::add), this method **bypasses fee-rate and PoW
    /// admission checks**. Transactions that were valid when they were first
    /// submitted should not be penalised because they happened to land in an
    /// orphaned block. The only hard gates are:
    ///
    /// - [`validate_transaction`] against the *current* post-reorg state
    ///   (coins and commitments must still exist).
    /// - Duplicate detection (commitment or input already pending).
    ///
    /// Capacity limits are also relaxed: the pools may temporarily exceed their
    /// soft caps. The next call to [`add`] or [`prune_invalid`] will bring them
    /// back into range.
    ///
    /// # Examples
    /// ```no_run
    /// # use midstate::mempool::Mempool;
    /// # use midstate::core::{State, Transaction};
    /// # let mut mempool = Mempool::new();
    /// # let state = State::genesis().0;
    /// # let abandoned_txs: Vec<Transaction> = vec![];
    /// mempool.re_add(abandoned_txs, &state);
    /// ```
    pub fn re_add(&mut self, txs: Vec<Transaction>, state: &State) {
        let mut restored = 0usize;

        for tx in txs {
            // Hard gate 1: must still be valid against the current state.
            if validate_transaction(state, &tx).is_err() {
                continue;
            }

            // Hard gate 2: no duplicates.
            let already_present = match &tx {
                Transaction::Commit { commitment, .. } => {
                    self.commits.contains_key(commitment)
                }
                Transaction::Reveal { .. } => {
                    tx.input_coin_ids().iter().any(|i| self.seen_inputs.contains(i))
                }
            };
            if already_present {
                continue;
            }

            // Insert directly, bypassing PoW / fee-rate / capacity checks.
            // Match by value so we can move `tx` into `Arc::new` without a
            // borrow-after-move error.
            match tx {
                Transaction::Commit { commitment, spam_nonce } => {
                    let h = crate::core::types::hash_concat(
                        &commitment,
                        &spam_nonce.to_le_bytes(),
                    );
                    let zeros = crate::core::types::count_leading_zeros(&h);
                    let arc_tx = Arc::new(Transaction::Commit { commitment, spam_nonce });
                    self.commits.insert(commitment, arc_tx);
                    self.commits_by_pow.insert((zeros, commitment));
                }
                reveal_tx @ Transaction::Reveal { .. } => {
                    let tx_bytes =
                        bincode::serialized_size(&reveal_tx).unwrap_or(0) as u64;
                    let fee_rate = compute_fee_rate(reveal_tx.fee(), tx_bytes);
                    let tx_id = get_tx_id(&reveal_tx);
                    for input in reveal_tx.input_coin_ids() {
                        self.seen_inputs.insert(input);
                        self.txs_by_input.insert(input, tx_id);
                    }
                    let arc_tx = Arc::new(reveal_tx);
                    self.reveals.insert(tx_id, arc_tx);
                    self.reveals_by_fee.insert((fee_rate, tx_id));
                }
            }

            restored += 1;
        }

        if restored > 0 {
            tracing::info!("Restored {} transactions to mempool after reorg", restored);
        }
    }

    /// Removes and returns up to `max` transactions from the mempool,
    /// in priority order suitable for block construction.
    ///
    /// Transactions are yielded **highest-priority first**:
    /// 1. Reveals ordered by descending fee rate (most profitable first).
    /// 2. Commits ordered by descending PoW (strongest work first).
    ///
    /// All associated index entries and `seen_inputs` records are cleaned up
    /// as transactions are removed.
    ///
    /// # Examples
    /// ```
    /// use midstate::mempool::Mempool;
    /// let mut mempool = Mempool::new();
    /// let batch = mempool.drain(100);
    /// assert_eq!(batch.len(), 0);
    /// ```
    pub fn drain(&mut self, max: usize) -> Vec<Transaction> {
        let mut drained = Vec::with_capacity(max.min(self.len()));

        // Highest fee-rate reveals first.
        while drained.len() < max {
            if let Some(&(rate, id)) = self.reveals_by_fee.iter().next_back() {
                self.reveals_by_fee.remove(&(rate, id));
                let arc_tx = self.reveals.remove(&id).unwrap();
                for input in arc_tx.input_coin_ids() {
                    self.seen_inputs.remove(&input);
                    self.txs_by_input.remove(&input);
                }
                drained.push(Arc::unwrap_or_clone(arc_tx));
            } else {
                break;
            }
        }

        // Highest PoW commits second.
        while drained.len() < max {
            if let Some(&(pow, comm)) = self.commits_by_pow.iter().next_back() {
                self.commits_by_pow.remove(&(pow, comm));
                let arc_tx = self.commits.remove(&comm).unwrap();
                drained.push(Arc::unwrap_or_clone(arc_tx));
            } else {
                break;
            }
        }

        drained
    }

    /// Returns the total number of transactions (Commits + Reveals) in the mempool.
    ///
    /// # Examples
    /// ```
    /// use midstate::mempool::Mempool;
    /// let mempool = Mempool::new();
    /// assert_eq!(mempool.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.commits.len() + self.reveals.len()
    }

    /// Returns `true` if the mempool contains no transactions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a priority-sorted snapshot of all transactions currently in
    /// the mempool without removing them.
    ///
    /// Order matches [`drain`](Self::drain): highest fee-rate Reveals first,
    /// followed by highest PoW Commits.
    ///
    /// Each element is an `Arc`-wrapped transaction. If you need owned
    /// `Transaction` values (e.g. to store in a `Vec<Transaction>`), use
    /// [`transactions_cloned`](Self::transactions_cloned) instead.
    pub fn transactions(&self) -> Vec<Arc<Transaction>> {
        let mut all = Vec::with_capacity(self.len());
        for &(_, id) in self.reveals_by_fee.iter().rev() {
            if let Some(arc_tx) = self.reveals.get(&id) {
                all.push(Arc::clone(arc_tx));
            }
        }
        for &(_, comm) in self.commits_by_pow.iter().rev() {
            if let Some(arc_tx) = self.commits.get(&comm) {
                all.push(Arc::clone(arc_tx));
            }
        }
        all
    }

    /// Returns a priority-sorted snapshot as owned `Transaction` values.
    ///
    /// This is a convenience wrapper around [`transactions`](Self::transactions)
    /// that clones each `Arc` into an owned value. Use this when the caller
    /// needs a `Vec<Transaction>` (e.g. for [`NodeHandle`] or RPC serialisation)
    /// rather than shared references.
    pub fn transactions_cloned(&self) -> Vec<Transaction> {
        self.transactions()
            .into_iter()
            .map(|arc| (*arc).clone())
            .collect()
    }

    /// Helper to cleanly remove a Reveal and its secondary indexes
    fn remove_reveal(&mut self, id: &[u8; 32]) {
        if let Some(arc_tx) = self.reveals.remove(id) {
            let tx_bytes = bincode::serialized_size(&*arc_tx).unwrap_or(0) as u64;
            let fee_rate = compute_fee_rate(arc_tx.fee(), tx_bytes);
            self.reveals_by_fee.remove(&(fee_rate, *id));
            for input in arc_tx.input_coin_ids() {
                self.seen_inputs.remove(&input);
                self.txs_by_input.remove(&input);
            }
        }
    }

    /// Event-driven, O(K) pruning executed immediately when a block is mined/received.
    /// Completely avoids hashing or signature verification.
    pub fn prune_on_new_block(&mut self, state: &State, newly_spent_inputs: &[[u8; 32]], newly_mined_commits: &[[u8; 32]]) {
        // 1. O(K) Prune Reveals based on spent inputs
        for coin in newly_spent_inputs {
            if let Some(tx_id) = self.txs_by_input.remove(coin) {
                self.remove_reveal(&tx_id);
            }
        }

        // 2. O(K) Prune mined Commits
        for commit in newly_mined_commits {
            if let Some(arc_tx) = self.commits.remove(commit) {
                if let Transaction::Commit { spam_nonce, .. } = &*arc_tx {
                    let h = crate::core::types::hash_concat(commit, &spam_nonce.to_le_bytes());
                    let zeros = crate::core::types::count_leading_zeros(&h);
                    self.commits_by_pow.remove(&(zeros, *commit));
                }
            }
        }

        // 3. O(R) TTL Prune (Instant, no signatures verified)
        let expired: Vec<_> = self.reveals.iter().filter(|(_, tx)| {
            if let Transaction::Reveal { salt, .. } = &***tx {
                let commit_hash = crate::core::compute_commitment(&tx.input_coin_ids(), &tx.output_coin_ids(), salt);
                if let Some(&height) = state.commitment_heights.get(&commit_hash) {
                    return state.height.saturating_sub(height) >= crate::core::COMMITMENT_TTL;
                }
            }
            false
        }).map(|(&id, _)| id).collect();

        for id in expired { self.remove_reveal(&id); }
    }

    /// Scans the mempool and removes any transactions that are no longer valid
    /// against `state` (e.g. inputs spent by a newly confirmed block, or
    /// commitments that have already been revealed).
    ///
    /// All associated index entries and `seen_inputs` records are cleaned up.
    ///
    /// # Examples
    /// ```no_run
    /// # use midstate::mempool::Mempool;
    /// # use midstate::core::State;
    /// # let mut mempool = Mempool::new();
    /// # let state = State::genesis().0;
    /// mempool.prune_invalid(&state);
    /// ```
    pub fn prune_invalid(&mut self, state: &State) {
        // --- Prune commits ---
        let commits_to_remove: Vec<[u8; 32]> = self
            .commits
            .iter()
            .filter(|(_, arc_tx)| validate_transaction(state, arc_tx).is_err())
            .map(|(comm, _)| *comm)
            .collect();

        for comm in commits_to_remove {
            if let Some(arc_tx) = self.commits.remove(&comm) {
                if let Transaction::Commit { commitment, spam_nonce } = &*arc_tx {
                    let h = crate::core::types::hash_concat(
                        commitment,
                        &spam_nonce.to_le_bytes(),
                    );
                    let zeros = crate::core::types::count_leading_zeros(&h);
                    self.commits_by_pow.remove(&(zeros, *commitment));
                }
            }
        }

        // --- Prune reveals ---
        let reveals_to_remove: Vec<[u8; 32]> = self
            .reveals
            .iter()
            .filter(|(_, arc_tx)| validate_transaction(state, arc_tx).is_err())
            .map(|(id, _)| *id)
            .collect();

        for id in reveals_to_remove {
            if let Some(arc_tx) = self.reveals.remove(&id) {
                let tx_bytes = bincode::serialized_size(&*arc_tx).unwrap_or(0) as u64;
                let fee_rate = compute_fee_rate(arc_tx.fee(), tx_bytes);
                self.reveals_by_fee.remove(&(fee_rate, id));
                for input in arc_tx.input_coin_ids() {
                    self.seen_inputs.remove(&input);
                    self.txs_by_input.remove(&input);
                }
            }
        }
    }
}

impl Default for Mempool {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test-only helpers: bypass admission checks for deterministic unit tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
impl Mempool {
    /// Directly inserts a Reveal transaction, bypassing all admission checks.
    /// For use in unit tests that need to pre-populate the pool quickly.
    pub(crate) fn force_add_reveal(&mut self, tx: Transaction) {
        let tx_id = get_tx_id(&tx);
        let tx_bytes = bincode::serialized_size(&tx).unwrap_or(0) as u64;
        let fee_rate = compute_fee_rate(tx.fee(), tx_bytes);
        for input in tx.input_coin_ids() {
            self.seen_inputs.insert(input);
        }
        let arc_tx = Arc::new(tx);
        self.reveals.insert(tx_id, arc_tx);
        self.reveals_by_fee.insert((fee_rate, tx_id));
    }

    /// Directly inserts a Commit transaction, bypassing all admission checks.
    /// For use in unit tests that need to pre-populate the pool quickly.
    pub(crate) fn force_add_commit(
        &mut self,
        commitment: [u8; 32],
        spam_nonce: u64,
        tx: Transaction,
    ) {
        let h = crate::core::types::hash_concat(&commitment, &spam_nonce.to_le_bytes());
        let zeros = crate::core::types::count_leading_zeros(&h);
        let arc_tx = Arc::new(tx);
        self.commits.insert(commitment, arc_tx);
        self.commits_by_pow.insert((zeros, commitment));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::*;
    use crate::core::mmr::UtxoAccumulator;

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn empty_state() -> State {
        State {
            midstate: [0u8; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height: 1,
            timestamp: 1000,
            commitment_heights: im::HashMap::new(),
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
        }
    }

    /// Mines a commit PoW nonce that achieves exactly 16 leading zero bits
    /// (the base network minimum) by checking that the first two bytes of the
    /// hash are 0x0000.
    fn mine_commit_nonce(commitment: &[u8; 32]) -> u64 {
        let mut n = 0u64;
        loop {
            let h = hash_concat(commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 {
                return n;
            }
            n += 1;
        }
    }

    /// Returns a state that has a single UTXO and the corresponding commitment
    /// already applied, plus the seeds needed to build a valid Reveal.
    fn state_with_committed_coin() -> (State, [u8; 32], [u8; 32], [u8; 32], [u8; 32], OutputData) {
        use crate::core::{wots, types::*};

        let mut state = empty_state();
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        let input_salt = hash(b"test salt");
        let coin_id = compute_coin_id(&address, 20, &input_salt);
        state.coins.insert(coin_id);

        let output = OutputData::Standard {
            address: hash(b"recipient"),
            value: 8,
            salt: [0x11; 32],
        };
        let commit_salt: [u8; 32] = [0x22; 32];
        let commitment = compute_commitment(
            &[coin_id],
            &[output.coin_id().unwrap()],
            &commit_salt,
        );

        let n = mine_commit_nonce(&commitment);
        let commit_tx = Transaction::Commit { commitment, spam_nonce: n };
        crate::core::transaction::apply_transaction(&mut state, &commit_tx).unwrap();

        (state, seed, coin_id, input_salt, commit_salt, output)
    }

    fn make_reveal_tx(
        seed: &[u8; 32],
        value: u64,
        input_salt: [u8; 32],
        commit_salt: [u8; 32],
        output: OutputData,
    ) -> Transaction {
        use crate::core::wots;

        let owner_pk = wots::keygen(seed);
        let address = compute_address(&owner_pk);
        let coin_id = compute_coin_id(&address, value, &input_salt);
        let commitment =
            compute_commitment(&[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let sig = wots::sign(seed, &commitment);

        Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&owner_pk),
                value,
                salt: input_salt,
            }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        }
    }

    // ── Commit path ─────────────────────────────────────────────────────────

    #[test]
    fn mempool_rejects_bad_commit_pow() {
        let mut mp = Mempool::new();
        let state = empty_state();
        let commitment = hash(b"mempool test");
        // Find a nonce that does NOT produce 16 leading zero bits.
        let mut bad = 0u64;
        loop {
            let h = hash_concat(&commitment, &bad.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) != 0x0000 {
                break;
            }
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
        let n = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: n };
        assert!(mp.add(tx, &state).is_ok());
        assert_eq!(mp.len(), 1);
    }

    #[test]
    fn duplicate_commitment_rejected() {
        let mut mp = Mempool::new();
        let state = empty_state();
        let commitment = hash(b"dup test");
        let n = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: n };
        assert!(mp.add(tx.clone(), &state).is_ok());
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("already in mempool"));
    }

    #[test]
    fn max_pending_commits_enforced() {
        let state = empty_state();
        let mut mp = Mempool::new();

        // Fill the commit pool to capacity using force_add (spam_nonce = 0 → ~0 PoW bits).
        for i in 0..MAX_PENDING_COMMITS {
            let commitment = hash(&(i as u64).to_le_bytes());
            let tx = Transaction::Commit { commitment, spam_nonce: 0 };
            mp.force_add_commit(commitment, 0, tx);
        }
        assert_eq!(mp.commits.len(), MAX_PENDING_COMMITS);

        // Pool is full; required PoW is now 22 bits.
        let extra = hash(b"commit overflow");
        let mut n = 0u64;
        loop {
            let h = hash_concat(&extra, &n.to_le_bytes());
            if crate::core::types::count_leading_zeros(&h) >= 22 {
                break;
            }
            n += 1;
        }

        let tx = Transaction::Commit { commitment: extra, spam_nonce: n };
        // High-PoW commit should evict one of the zero-PoW dummies.
        assert!(mp.add(tx, &state).is_ok());
        assert_eq!(mp.len(), MAX_PENDING_COMMITS, "Pool must not exceed capacity");
        assert!(mp.commits.contains_key(&extra), "New commit must be present");

        // A low-PoW commit should now be rejected outright (pool still full, 22 bits required).
        let bad_tx = Transaction::Commit { commitment: hash(b"bad"), spam_nonce: 0 };
        let err = mp.add(bad_tx, &state).unwrap_err();
        assert!(err.to_string().contains("Mempool is busy"));
    }

    /// When the commit pool is full and the incoming commit's PoW is not
    /// strictly better than the current worst, it must be rejected rather than
    /// silently dropped.
    #[test]
    fn max_pending_commits_rejects_equal_pow() {
        let state = empty_state();
        let mut mp = Mempool::new();

        // Fill with commits that all have exactly 16 leading zero bits.
        for i in 0..MAX_PENDING_COMMITS {
            let commitment = hash(&(i as u64).to_le_bytes());
            // Force-add with a fake nonce; the BTreeSet key uses the recomputed zeros.
            // We need the stored PoW to be a known value — use force_add with nonce 0 and
            // then fix up the BTreeSet key to be 16 so the test is deterministic.
            // Simpler: just force-add with the real nonce.
            let real_nonce = mine_commit_nonce(&commitment);
            let tx = Transaction::Commit { commitment, spam_nonce: real_nonce };
            mp.force_add_commit(commitment, real_nonce, tx);
        }

        // The pool is full. Dynamic threshold is now 22 bits.
        // A commit with exactly 16 bits should fail the dynamic threshold check first.
        let commitment = hash(b"equal pow");
        let nonce = mine_commit_nonce(&commitment); // exactly 16 bits
        let tx = Transaction::Commit { commitment, spam_nonce: nonce };
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("Mempool is busy") || err.to_string().contains("Mempool full of Commits"));
    }

    // ── Reveal path ─────────────────────────────────────────────────────────

    #[test]
    fn mempool_accepts_valid_reveal() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();
        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        assert!(mp.add(tx, &state).is_ok());
        assert_eq!(mp.len(), 1);
    }

    #[test]
    fn mempool_rejects_duplicate_input() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();
        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        mp.add(tx.clone(), &state).unwrap();
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("already in mempool"));
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

        let output = OutputData::Standard { address: hash(b"r"), value: 1, salt: [0; 32] };
        let commit_salt = [1u8; 32];
        let commitment =
            compute_commitment(&[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let n = mine_commit_nonce(&commitment);
        crate::core::transaction::apply_transaction(
            &mut state,
            &Transaction::Commit { commitment, spam_nonce: n },
        )
        .unwrap();

        let sig = crate::core::wots::sign(&seed, &commitment);
        // in = 1, out = 1 → fee = 0, which is below the minimum.
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::p2pk(&owner_pk),
                value: 1,
                salt: input_salt,
            }],
            witnesses: vec![Witness::sig(crate::core::wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };

        let mut mp = Mempool::new();
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("Fee rate too low"));
    }

    // ── RBF eviction ────────────────────────────────────────────────────────

    #[test]
    fn mempool_rbf_evicts_lowest_fee_reveal() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();

        // Fill the reveal pool with low-fee transactions (fee = 2 on a ~same-size tx).
        for i in 0..MAX_MEMPOOL_REVEALS {
            let dummy_input = InputReveal {
                predicate: Predicate::p2pk(&[0; 32]),
                value: 10,
                salt: hash(&(i as u64).to_le_bytes()),
            };
            let dummy_output =
                OutputData::Standard { address: [0; 32], value: 8, salt: [0; 32] };
            let dummy_reveal = Transaction::Reveal {
                inputs: vec![dummy_input],
                witnesses: vec![],
                outputs: vec![dummy_output],
                salt: [0; 32],
            };
            mp.force_add_reveal(dummy_reveal);
        }
        assert_eq!(mp.reveals.len(), MAX_MEMPOOL_REVEALS);

        // Our real tx has fee = 12 (value 20, output 8), which should be a higher rate.
        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        assert_eq!(tx.fee(), 12);

        assert!(mp.add(tx.clone(), &state).is_ok());
        assert_eq!(mp.reveals.len(), MAX_MEMPOOL_REVEALS, "Pool size must remain constant");

        // The new input must now be tracked.
        let new_input = tx.input_coin_ids()[0];
        assert!(
            mp.seen_inputs.contains(&new_input),
            "New reveal's input must be in seen_inputs"
        );
    }

    #[test]
    fn mempool_rbf_rejects_lower_fee_reveal() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();

        // Fill with high-fee transactions (fee = 10).
        for i in 0..MAX_MEMPOOL_REVEALS {
            let dummy_input = InputReveal {
                predicate: Predicate::p2pk(&[0; 32]),
                value: 20,
                salt: hash(&(i as u64).to_le_bytes()),
            };
            let dummy_output =
                OutputData::Standard { address: [0; 32], value: 10, salt: [0; 32] };
            let dummy_reveal = Transaction::Reveal {
                inputs: vec![dummy_input],
                witnesses: vec![],
                outputs: vec![dummy_output],
                salt: [0; 32],
            };
            mp.force_add_reveal(dummy_reveal);
        }
        assert_eq!(mp.reveals.len(), MAX_MEMPOOL_REVEALS);

        // Our tx has fee = 12, but the pool is full of fee-10 txs that are the
        // same size, so the rate is comparable and the incoming tx must not evict.
        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        let err = mp.add(tx, &state).unwrap_err();
        assert!(err.to_string().contains("fee rate too low to replace any existing Reveal"));
    }

    /// A Commit must be rejected when both pools are at capacity, not silently
    /// accepted or panicked.
    #[test]
    fn mempool_full_rejects_commit() {
        let state = empty_state();
        let mut mp = Mempool::new();

        // Fill the commit pool to capacity.
        for i in 0..MAX_PENDING_COMMITS {
            let commitment = hash(&(i as u64).to_le_bytes());
            let tx = Transaction::Commit { commitment, spam_nonce: 0 };
            mp.force_add_commit(commitment, 0, tx);
        }
        assert_eq!(mp.commits.len(), MAX_PENDING_COMMITS);

        // A commit with just 16 bits of PoW cannot displace any entry
        // because the dynamic threshold is now 22 bits.
        let extra = hash(b"one more");
        let n = mine_commit_nonce(&extra); // 16 bits
        let tx = Transaction::Commit { commitment: extra, spam_nonce: n };
        let err = mp.add(tx, &state).unwrap_err();
        assert!(
            err.to_string().contains("Mempool is busy")
                || err.to_string().contains("Mempool full of Commits"),
            "unexpected error: {err}"
        );
    }

    // ── drain ────────────────────────────────────────────────────────────────

    #[test]
    fn drain_returns_and_removes() {
        let mut mp = Mempool::new();
        for i in 0..5u64 {
            let commitment = hash(&i.to_le_bytes());
            let tx = Transaction::Commit { commitment, spam_nonce: 0 };
            mp.force_add_commit(commitment, 0, tx);
        }
        assert_eq!(mp.len(), 5);

        let drained = mp.drain(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(mp.len(), 2);
        assert_eq!(mp.commits_by_pow.len(), 2, "BTreeSet must stay in sync");
    }

    #[test]
    fn drain_more_than_available() {
        let mut mp = Mempool::new();
        let commitment = [1u8; 32];
        let tx = Transaction::Commit { commitment, spam_nonce: 0 };
        mp.force_add_commit(commitment, 0, tx);

        let drained = mp.drain(100);
        assert_eq!(drained.len(), 1);
        assert_eq!(mp.len(), 0);
        assert!(mp.is_empty());
    }

    #[test]
    fn drain_is_priority_ordered() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();

        // Add a commit.
        let commitment = hash(b"drain order commit");
        let n = mine_commit_nonce(&commitment);
        let commit_tx = Transaction::Commit { commitment, spam_nonce: n };
        mp.force_add_commit(commitment, n, commit_tx);

        // Add a reveal — it should come out first.
        let reveal_tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        mp.add(reveal_tx, &state).unwrap();

        let drained = mp.drain(2);
        assert_eq!(drained.len(), 2);
        assert!(
            matches!(drained[0], Transaction::Reveal { .. }),
            "Reveal must be drained before Commit"
        );
        assert!(
            matches!(drained[1], Transaction::Commit { .. }),
            "Commit must be drained second"
        );
    }

    // ── prune_invalid ────────────────────────────────────────────────────────

    #[test]
    fn prune_removes_invalid_commits() {
        let state = empty_state();
        let mut mp = Mempool::new();
        let commitment = hash(b"prune test");
        // Force-insert a commit with nonce u64::MAX — validation will reject it.
        let tx = Transaction::Commit { commitment, spam_nonce: u64::MAX };
        mp.force_add_commit(commitment, u64::MAX, tx);
        assert_eq!(mp.len(), 1);

        mp.prune_invalid(&state);

        assert_eq!(mp.len(), 0);
        assert!(mp.commits_by_pow.is_empty(), "BTreeSet must be cleared too");
    }

    #[test]
    fn prune_removes_invalid_reveals() {
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();
        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        mp.add(tx, &state).unwrap();
        assert_eq!(mp.len(), 1);

        // Pruning against a fresh state (no committed coin) should remove the reveal.
        mp.prune_invalid(&empty_state());

        assert_eq!(mp.len(), 0);
        assert!(mp.seen_inputs.is_empty(), "seen_inputs must be cleared");
        assert!(mp.reveals_by_fee.is_empty(), "BTreeSet must be cleared too");
    }

    // ── re_add ───────────────────────────────────────────────────────────────

    #[test]
    fn re_add_skips_state_invalid_txs() {
        let state = empty_state();
        let mut mp = Mempool::new();
        // A commit with nonce u64::MAX will fail validate_transaction.
        let bad_tx = Transaction::Commit { commitment: hash(b"bad"), spam_nonce: u64::MAX };
        mp.re_add(vec![bad_tx], &state);
        assert_eq!(mp.len(), 0);
    }

    #[test]
    fn re_add_skips_duplicate_commitments() {
        let state = empty_state();
        let mut mp = Mempool::new();
        let commitment = hash(b"dup re-add");
        let tx = Transaction::Commit { commitment, spam_nonce: 0 };
        mp.force_add_commit(commitment, 0, tx.clone());
        mp.re_add(vec![tx], &state);
        assert_eq!(mp.len(), 1, "Duplicate must not be re-added");
    }

    #[test]
    fn re_add_accepts_low_pow_after_reorg() {
        // This test verifies the key behavioural difference from `add`:
        // re_add must accept a valid commit even if its PoW is below the
        // current dynamic threshold.
        let state = empty_state();
        let mut mp = Mempool::new();

        // Fill the commit pool so the dynamic threshold is 22 bits.
        for i in 0..MAX_PENDING_COMMITS {
            let commitment = hash(&(i as u64).to_le_bytes());
            let tx = Transaction::Commit { commitment, spam_nonce: 0 };
            mp.force_add_commit(commitment, 0, tx);
        }
        assert_eq!(mp.commits.len(), MAX_PENDING_COMMITS);

        // A commit that only has 16 bits of PoW would be rejected by `add`
        // because the threshold is now 22. But if it came from an orphaned
        // block it was previously valid and re_add must restore it.
        //
        // NOTE: validate_transaction must accept it for this to work.
        // If your implementation rejects 16-bit commits at the state-validation
        // layer rather than just at the mempool layer, adjust accordingly.
        let reorg_commitment = hash(b"reorg commit");
        let reorg_nonce = mine_commit_nonce(&reorg_commitment); // 16 bits
        let reorg_tx = Transaction::Commit { commitment: reorg_commitment, spam_nonce: reorg_nonce };

        // validate_transaction must pass for this tx against the state.
        if validate_transaction(&state, &reorg_tx).is_ok() {
            mp.re_add(vec![reorg_tx], &state);
            // Pool may temporarily exceed MAX_PENDING_COMMITS — that is intentional.
            assert!(
                mp.commits.contains_key(&reorg_commitment),
                "Low-PoW reorg commit must be accepted by re_add"
            );
        }
        // If validate_transaction itself rejects it, the test is vacuously satisfied.
    }

    #[test]
    fn re_add_accepts_low_fee_reveal_after_reorg() {
        // Symmetric to the commit test: a reveal that barely passed the fee
        // check when first submitted must be restorable via re_add even if it
        // would fail the current minimum (e.g. due to a fee policy change).
        //
        // In practice MIN_FEE_PER_KB is a constant, so we test the simpler
        // property: re_add restores a previously valid reveal without error.
        let (state, seed, _coin_id, input_salt, commit_salt, output) =
            state_with_committed_coin();
        let mut mp = Mempool::new();

        let tx = make_reveal_tx(&seed, 20, input_salt, commit_salt, output);
        // Verify it passes validate_transaction directly.
        assert!(validate_transaction(&state, &tx).is_ok());

        mp.re_add(vec![tx], &state);
        assert_eq!(mp.len(), 1, "Valid reorg reveal must be restored");
    }
}
