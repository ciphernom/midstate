//! Address balances, the rich list, and a checked circulating supply.
//!
//! # Why this exists
//!
//! The chain state keeps coin IDs only — `BLAKE3(address ‖ value ‖ salt)` — so a
//! node can tell whether a coin exists but not what it is worth or who holds it.
//! The values are public all the same: every output in a block carries its
//! address and value, and every input reveals the address (as its predicate) and
//! the value of the coin it spends. This index keeps what the state discards,
//! which is what the explorer needs for its supply figures and rich list:
//!
//! * [`COINS`]: `coin_id → address ‖ value`, for every unspent coin with value.
//! * [`ADDRS`]: `address → balance ‖ coin count`, for every address holding any.
//! * [`RANK`]:  `balance (big-endian) ‖ address → ()`, so the largest holders
//!   are simply the last keys of the table.
//! * [`META`]:  how far the index has got, and running totals — value held in
//!   unspent coins, value destroyed by `DataBurn` outputs, number of holders.
//!
//! The totals are checkable: every block adds its reward and removes what it
//! burns (fees only move value to the miner), so the unspent value must equal
//! `cumulative_reward(blocks) − burned` exactly. The RPC reports that check
//! with every supply figure.
//!
//! # Consistency
//!
//! [`BatchStore`](super::BatchStore) calls [`BalanceIndex::on_save`] and
//! [`BalanceIndex::on_truncate`] inside the write transaction that stores or
//! deletes the batch, so a batch and its effect on the index commit together or
//! not at all. One invariant makes reorgs safe:
//!
//! > for every height `h < next`, the batch stored at `h` is exactly the batch
//! > whose effects the index holds.
//!
//! A reorg overwrites stored heights in place — `perform_reorg` saves the new
//! fork over the old one without truncating first. When `on_save` sees a height
//! below `next`, it unapplies every indexed height from the top down to that one,
//! reading each old batch before it is replaced, and then applies the new batch.
//! There is no undo log: an input reveals the address and value of the coin it
//! spends, so a block can be undone from the block alone.
//!
//! Heights at or above `next` belong to the backfill, which walks forward from
//! genesis in short write transactions, up to `chain_len` — the main chain's
//! length as last recorded by `on_save`/`on_truncate`. Not the store's highest
//! key: after a reorg to a fork with fewer blocks, the store still holds the
//! abandoned fork's higher blocks until the chain grows over them.
//!
//! # Opt-in
//!
//! The index costs disk (roughly 100 bytes per unspent coin and per holder) and
//! one pass over all history, so a node keeps it only when started with
//! `--rich-list`. While it is off, the hooks do nothing. Turning it back on runs
//! [`BalanceIndex::enable`], which checks that the last block indexed is still
//! where the index left it and rebuilds from genesis when it is not.

use anyhow::Result;
use redb::{Database, ReadableTable, Table, TableDefinition, WriteTransaction};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::core::types::{compute_coin_id, Batch, OutputData, Transaction};

/// `coin_id → address(32) ‖ value(8, LE)`.
pub const COINS: TableDefinition<&[u8; 32], &[u8; 40]> = TableDefinition::new("balance_coins");

/// `address → balance(8, LE) ‖ coins(8, LE)`. Present only while the address holds
/// at least one coin.
pub const ADDRS: TableDefinition<&[u8; 32], &[u8; 16]> = TableDefinition::new("balance_addresses");

/// `balance(8, BE) ‖ address(32) → ()`. Big-endian so byte order is numeric
/// order: the richest addresses are the last keys.
pub const RANK: TableDefinition<&[u8; 40], ()> = TableDefinition::new("balance_rank");

/// Watermark and totals. See [`Meta`].
pub const META: TableDefinition<&str, &[u8]> = TableDefinition::new("balance_meta");

/// Bump to force every node to rebuild on upgrade.
const VERSION: u64 = 1;

/// How long one backfill transaction may hold the database's write lock. The
/// node's own block writes wait behind it, so keep it short.
const STEP_BUDGET: Duration = Duration::from_millis(150);

/// Upper bound on blocks per backfill transaction, whatever the time.
const STEP_MAX_BLOCKS: u64 = 2_000;

/// Gap between backfill transactions, so a waiting block write gets the lock.
const STEP_PAUSE: Duration = Duration::from_millis(25);

/// How long the backfill worker sleeps when it has nothing to do. Hooks wake it
/// early; this only bounds the delay if a wake-up is ever missed.
const IDLE_WAIT: Duration = Duration::from_secs(60);

// ── Encoding ────────────────────────────────────────────────────────────────

fn pack_coin(address: &[u8; 32], value: u64) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[..32].copy_from_slice(address);
    out[32..].copy_from_slice(&value.to_le_bytes());
    out
}

fn unpack_coin(raw: &[u8; 40]) -> ([u8; 32], u64) {
    let mut address = [0u8; 32];
    address.copy_from_slice(&raw[..32]);
    (address, u64::from_le_bytes(raw[32..].try_into().unwrap()))
}

fn pack_holder(balance: u64, coins: u64) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&balance.to_le_bytes());
    out[8..].copy_from_slice(&coins.to_le_bytes());
    out
}

fn unpack_holder(raw: &[u8; 16]) -> (u64, u64) {
    (
        u64::from_le_bytes(raw[..8].try_into().unwrap()),
        u64::from_le_bytes(raw[8..].try_into().unwrap()),
    )
}

fn rank_key(balance: u64, address: &[u8; 32]) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[..8].copy_from_slice(&balance.to_be_bytes());
    out[8..].copy_from_slice(address);
    out
}

fn unpack_rank(raw: &[u8; 40]) -> (u64, [u8; 32]) {
    let mut address = [0u8; 32];
    address.copy_from_slice(&raw[8..]);
    (u64::from_be_bytes(raw[..8].try_into().unwrap()), address)
}

// ── Meta ────────────────────────────────────────────────────────────────────

/// The index's watermark and running totals, stored field by field in [`META`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    /// Format of the tables; a mismatch means rebuild.
    pub version: u64,
    /// Next height to apply — equivalently, how many blocks are indexed
    /// (genesis is height 0).
    pub next: u64,
    /// Main-chain length (tip height + 1) as last recorded by a save or
    /// truncate. The backfill never reads past it.
    pub chain_len: u64,
    /// `extension.final_hash` of the block at `next - 1`, so a restart can check
    /// that block is still the one on disk.
    pub tip_hash: [u8; 32],
    /// Total value of the unspent coins, the genesis outputs included.
    pub unspent_value: u64,
    /// Unspent coins with value.
    pub unspent_coins: u64,
    /// Addresses holding at least one coin.
    pub holders: u64,
    /// Value destroyed by `DataBurn` outputs.
    pub burned: u64,
    /// `DataBurn` outputs that destroyed value.
    pub burn_outputs: u64,
    /// Spends of coins the index did not hold, duplicate coins, and other
    /// things the chain's own rules make impossible. Stays 0 unless the index
    /// has a bug, and is reported so that a bug shows instead of hiding inside
    /// the totals.
    pub anomalies: u64,
    /// The index can no longer be trusted — a reorg needed a block that could
    /// not be read. The backfill wipes it and starts again from genesis.
    pub stale: bool,
    /// The backfill needed a block this node does not have (pruned history):
    /// `height + 1`, or 0 for none.
    pub missing: u64,
}

impl Meta {
    fn read<T: ReadableTable<&'static str, &'static [u8]>>(t: &T) -> Result<Self> {
        let u = |key: &str| -> Result<u64> {
            Ok(match t.get(key)? {
                Some(g) if g.value().len() == 8 => {
                    u64::from_le_bytes(g.value().try_into().unwrap())
                }
                _ => 0,
            })
        };
        let mut tip_hash = [0u8; 32];
        if let Some(g) = t.get("tip_hash")? {
            if g.value().len() == 32 {
                tip_hash.copy_from_slice(g.value());
            }
        }
        Ok(Self {
            version: u("version")?,
            next: u("next")?,
            chain_len: u("chain_len")?,
            tip_hash,
            unspent_value: u("unspent_value")?,
            unspent_coins: u("unspent_coins")?,
            holders: u("holders")?,
            burned: u("burned")?,
            burn_outputs: u("burn_outputs")?,
            anomalies: u("anomalies")?,
            stale: u("stale")? != 0,
            missing: u("missing")?,
        })
    }

    fn write(&self, t: &mut Table<&'static str, &'static [u8]>) -> Result<()> {
        let fields: [(&str, u64); 11] = [
            ("version", self.version),
            ("next", self.next),
            ("chain_len", self.chain_len),
            ("unspent_value", self.unspent_value),
            ("unspent_coins", self.unspent_coins),
            ("holders", self.holders),
            ("burned", self.burned),
            ("burn_outputs", self.burn_outputs),
            ("anomalies", self.anomalies),
            ("stale", self.stale as u64),
            ("missing", self.missing),
        ];
        for (key, value) in fields {
            t.insert(key, value.to_le_bytes().as_slice())?;
        }
        t.insert("tip_hash", self.tip_hash.as_slice())?;
        Ok(())
    }

    fn load(txn: &WriteTransaction) -> Result<Self> {
        let t = txn.open_table(META)?;
        Self::read(&t)
    }

    fn store(&self, txn: &WriteTransaction) -> Result<()> {
        let mut t = txn.open_table(META)?;
        self.write(&mut t)
    }

    /// Caught up with the chain and trustworthy.
    pub fn is_current(&self) -> bool {
        !self.stale
            && self.missing == 0
            && self.version == VERSION
            && self.next > 0
            && self.next >= self.chain_len
    }
}

// ── Applying and unapplying blocks ──────────────────────────────────────────

/// The three data tables, open for writing inside one transaction.
struct Writer<'t> {
    coins: Table<'t, &'static [u8; 32], &'static [u8; 40]>,
    addrs: Table<'t, &'static [u8; 32], &'static [u8; 16]>,
    rank: Table<'t, &'static [u8; 40], ()>,
}

impl<'t> Writer<'t> {
    fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(Self {
            coins: txn.open_table(COINS)?,
            addrs: txn.open_table(ADDRS)?,
            rank: txn.open_table(RANK)?,
        })
    }

    /// Move `address` from `before` to `after` = (balance, coins), keeping
    /// [`ADDRS`], [`RANK`] and the holder count in step.
    fn set_holder(
        &mut self,
        m: &mut Meta,
        address: &[u8; 32],
        before: (u64, u64),
        after: (u64, u64),
    ) -> Result<()> {
        if before.0 > 0 {
            self.rank.remove(&rank_key(before.0, address))?;
        }
        if after.1 == 0 {
            self.addrs.remove(address)?;
        } else {
            self.addrs.insert(address, &pack_holder(after.0, after.1))?;
        }
        if after.0 > 0 {
            self.rank.insert(&rank_key(after.0, address), ())?;
        }
        match (before.1 > 0, after.1 > 0) {
            (false, true) => m.holders += 1,
            (true, false) => m.holders = m.holders.saturating_sub(1),
            _ => {}
        }
        Ok(())
    }

    fn holder(&self, address: &[u8; 32]) -> Result<(u64, u64)> {
        Ok(self
            .addrs
            .get(address)?
            .map(|g| unpack_holder(g.value()))
            .unwrap_or((0, 0)))
    }

    /// Record a new unspent coin.
    fn add_coin(
        &mut self,
        m: &mut Meta,
        coin_id: &[u8; 32],
        address: &[u8; 32],
        value: u64,
    ) -> Result<()> {
        if value == 0 {
            return Ok(()); // State threads carry no value and are not tracked.
        }
        if self.coins.get(coin_id)?.is_some() {
            m.anomalies += 1; // consensus rejects duplicate coins
            return Ok(());
        }
        self.coins.insert(coin_id, &pack_coin(address, value))?;
        let (balance, coins) = self.holder(address)?;
        let new_balance = balance.checked_add(value).unwrap_or_else(|| {
            m.anomalies += 1;
            u64::MAX
        });
        self.set_holder(m, address, (balance, coins), (new_balance, coins + 1))?;
        m.unspent_value = m.unspent_value.saturating_add(value);
        m.unspent_coins += 1;
        Ok(())
    }

    /// Remove an unspent coin. False if the index did not hold it.
    fn remove_coin(&mut self, m: &mut Meta, coin_id: &[u8; 32]) -> Result<bool> {
        let Some((address, value)) = self.coins.remove(coin_id)?.map(|g| unpack_coin(g.value()))
        else {
            return Ok(false);
        };
        let (balance, coins) = self.holder(&address)?;
        if coins == 0 || balance < value {
            m.anomalies += 1;
        }
        let after = (balance.saturating_sub(value), coins.saturating_sub(1));
        self.set_holder(m, &address, (balance, coins), after)?;
        m.unspent_value = m.unspent_value.saturating_sub(value);
        m.unspent_coins = m.unspent_coins.saturating_sub(1);
        Ok(true)
    }

    /// Apply one block's effects, in the order the state applies them:
    /// transactions first (each one's inputs, then its outputs), coinbase last.
    fn apply(&mut self, m: &mut Meta, batch: &Batch) -> Result<()> {
        for tx in &batch.transactions {
            let (inputs, outputs) = match tx {
                Transaction::Commit { .. } => continue,
                Transaction::Reveal {
                    inputs, outputs, ..
                }
                | Transaction::Consolidate {
                    inputs, outputs, ..
                } => (inputs, outputs),
            };
            for input in inputs {
                // A zero-value input is a state thread (consensus requires 0 there).
                if input.value > 0 && !self.remove_coin(m, &input.coin_id())? {
                    m.anomalies += 1;
                }
            }
            for out in outputs {
                match out {
                    OutputData::Standard {
                        address,
                        value,
                        salt,
                    } => {
                        self.add_coin(m, &compute_coin_id(address, *value, salt), address, *value)?;
                    }
                    OutputData::Confidential { .. } => {}
                    OutputData::DataBurn { value_burned, .. } => {
                        m.burned = m.burned.saturating_add(*value_burned);
                        m.burn_outputs += (*value_burned > 0) as u64;
                    }
                }
            }
        }
        for cb in &batch.coinbase {
            self.add_coin(m, &cb.coin_id(), &cb.address, cb.value)?;
        }
        Ok(())
    }

    /// Exactly undo [`Writer::apply`] for the same block, in reverse order.
    fn unapply(&mut self, m: &mut Meta, batch: &Batch) -> Result<()> {
        for cb in batch.coinbase.iter().rev() {
            if cb.value > 0 && !self.remove_coin(m, &cb.coin_id())? {
                m.anomalies += 1;
            }
        }
        for tx in batch.transactions.iter().rev() {
            let (inputs, outputs) = match tx {
                Transaction::Commit { .. } => continue,
                Transaction::Reveal {
                    inputs, outputs, ..
                }
                | Transaction::Consolidate {
                    inputs, outputs, ..
                } => (inputs, outputs),
            };
            for out in outputs.iter().rev() {
                match out {
                    OutputData::Standard {
                        address,
                        value,
                        salt,
                    } => {
                        if *value > 0
                            && !self.remove_coin(m, &compute_coin_id(address, *value, salt))?
                        {
                            m.anomalies += 1;
                        }
                    }
                    OutputData::Confidential { .. } => {}
                    OutputData::DataBurn { value_burned, .. } => {
                        m.burned = m.burned.saturating_sub(*value_burned);
                        m.burn_outputs = m.burn_outputs.saturating_sub((*value_burned > 0) as u64);
                    }
                }
            }
            for input in inputs.iter().rev() {
                if input.value > 0 {
                    self.add_coin(m, &input.coin_id(), &input.predicate.address(), input.value)?;
                }
            }
        }
        Ok(())
    }
}

/// The batch stored at `height`, read inside `txn` (so a write in progress sees
/// what it is about to replace). `None` if absent or unreadable.
fn stored_batch(txn: &WriteTransaction, height: u64) -> Result<Option<Batch>> {
    let table = txn.open_table(super::BATCHES_TABLE)?;
    let batch = match table.get(height)? {
        Some(raw) => bincode::deserialize::<Batch>(raw.value()).ok(),
        None => None,
    };
    Ok(batch)
}

/// Unapply heights `next - 1` down to `to`, so that `next == to`. On a block it
/// cannot read, marks the index stale (the backfill rebuilds it) and stops.
fn rewind(txn: &WriteTransaction, m: &mut Meta, to: u64) -> Result<()> {
    let mut w = Writer::open(txn)?;
    while m.next > to {
        let h = m.next - 1;
        let Some(old) = stored_batch(txn, h)? else {
            tracing::warn!(
                "Rich-list index: block {} is unreadable during a reorg; rebuilding the index",
                h
            );
            m.stale = true;
            return Ok(());
        };
        w.unapply(m, &old)?;
        m.next = h;
        // A block's prev_header_hash is its parent's extension.final_hash.
        m.tip_hash = old.prev_header_hash;
    }
    Ok(())
}

/// Empty the data tables and reset the totals, keeping `chain_len`.
fn wipe(txn: &WriteTransaction, m: &mut Meta) -> Result<()> {
    txn.delete_table(COINS)?;
    txn.delete_table(ADDRS)?;
    txn.delete_table(RANK)?;
    Writer::open(txn)?; // recreate them empty
    *m = Meta {
        version: VERSION,
        chain_len: m.chain_len,
        ..Meta::default()
    };
    Ok(())
}

// ── The shared handle ───────────────────────────────────────────────────────

/// What one backfill step did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Applied blocks; `next` is the new watermark.
    Progress { next: u64, chain_len: u64 },
    /// Nothing to do: caught up (or disabled).
    Idle,
    /// The block at this height is not stored, so the index cannot go further.
    Missing(u64),
}

/// The index's switch, its hooks and its backfill worker, shared by every clone
/// of the [`BatchStore`](super::BatchStore).
#[derive(Clone, Debug, Default)]
pub struct BalanceIndex {
    inner: Arc<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    enabled: AtomicBool,
    worker_started: AtomicBool,
    wake: Mutex<bool>,
    cv: Condvar,
}

/// Where a just-enabled index stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Enabled {
    /// No index on disk; it is built from genesis.
    Fresh,
    /// The index on disk is still valid; it carries on from `next`.
    Resumed { next: u64 },
    /// The index on disk does not match the chain; it is rebuilt from genesis.
    Rebuild { reason: String },
}

impl BalanceIndex {
    pub fn is_enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Acquire)
    }

    fn wake(&self) {
        let mut flag = self.inner.wake.lock().unwrap_or_else(|e| e.into_inner());
        *flag = true;
        self.inner.cv.notify_all();
    }

    fn wait(&self, timeout: Duration) {
        let flag = self.inner.wake.lock().unwrap_or_else(|e| e.into_inner());
        let (mut flag, _) = self
            .inner
            .cv
            .wait_timeout_while(flag, timeout, |woken| !*woken)
            .unwrap_or_else(|e| e.into_inner());
        *flag = false;
    }

    /// Hook: `batch` is about to be stored at `height`, inside `txn`. Must run
    /// before the write replaces whatever is stored there now.
    ///
    /// ```text
    /// Pre:  enabled ∧ ¬stale ∧ stored[h] = applied[h] for all h < next
    /// Post: chain_len' = height + 1
    ///       height < next  ⇒ heights next-1 … height unapplied, then batch applied:
    ///                         next' = height + 1
    ///       height = next  ⇒ batch applied, next' = height + 1
    ///       height > next  ⇒ nothing applied (the backfill gets there); worker woken
    /// ```
    pub fn on_save(&self, txn: &WriteTransaction, height: u64, batch: &Batch) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let mut m = Meta::load(txn)?;
        m.chain_len = height + 1;
        if !m.stale && height < m.next {
            rewind(txn, &mut m, height)?;
        }
        if !m.stale && height == m.next {
            Writer::open(txn)?.apply(&mut m, batch)?;
            m.next = height + 1;
            m.tip_hash = batch.extension.final_hash;
            m.missing = 0;
        } else {
            self.wake();
        }
        m.store(txn)
    }

    /// Hook: every stored height `>= new_len` is about to be deleted, inside
    /// `txn`. Unapplies those that are indexed, while they can still be read.
    pub fn on_truncate(&self, txn: &WriteTransaction, new_len: u64) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let mut m = Meta::load(txn)?;
        m.chain_len = m.chain_len.min(new_len);
        if !m.stale && m.next > new_len {
            rewind(txn, &mut m, new_len)?;
        }
        m.store(txn)
    }

    /// Turn the index on for a chain of `chain_len` blocks: create the tables if
    /// needed, and check any index already on disk against the stored chain,
    /// marking it for a rebuild if it no longer matches. Call once, before the
    /// node starts storing blocks; [`BalanceIndex::spawn_worker`] then does the
    /// building.
    pub fn enable(&self, db: &Arc<Database>, chain_len: u64) -> Result<Enabled> {
        let txn = db.begin_write()?;
        Writer::open(&txn)?; // create the data tables on first use
        let mut m = Meta::load(&txn)?;
        let outcome = if m.version == 0 && m.next == 0 {
            m.version = VERSION;
            Enabled::Fresh
        } else if m.version != VERSION {
            m.stale = true;
            Enabled::Rebuild {
                reason: format!("index format {} replaced by {}", m.version, VERSION),
            }
        } else if m.stale {
            Enabled::Rebuild {
                reason: "the index was marked for rebuilding".into(),
            }
        } else if m.next > chain_len {
            // The index ran ahead of the chain this node now has: a reorg to a
            // shorter fork, or a block stored without its state. Either way the
            // blocks it holds may not be the ones on disk.
            m.stale = true;
            Enabled::Rebuild {
                reason: format!(
                    "it covers {} blocks but the chain has {}",
                    m.next, chain_len
                ),
            }
        } else if m.next > 0 {
            let on_disk = stored_batch(&txn, m.next - 1)?.map(|b| b.extension.final_hash);
            if on_disk == Some(m.tip_hash) {
                Enabled::Resumed { next: m.next }
            } else {
                m.stale = true;
                Enabled::Rebuild {
                    reason: format!("block {} changed while the index was off", m.next - 1),
                }
            }
        } else {
            Enabled::Resumed { next: 0 }
        };
        m.chain_len = chain_len;
        m.missing = 0; // a block that was missing may have been stored since
        m.store(&txn)?;
        txn.commit()?;
        self.inner.enabled.store(true, Ordering::Release);
        Ok(outcome)
    }

    /// Start the background worker that keeps the index caught up. Idempotent.
    pub fn spawn_worker(&self, db: Arc<Database>) {
        if self.inner.worker_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("rich-list-index".into())
            .spawn(move || me.worker(db));
        if let Err(e) = spawned {
            tracing::warn!(
                "Rich-list index: could not start the backfill thread: {}",
                e
            );
            self.inner.worker_started.store(false, Ordering::Release);
        }
    }

    fn worker(&self, db: Arc<Database>) {
        let mut last_log = Instant::now();
        let mut logged_missing = None;
        loop {
            match self.step(&db) {
                Ok(Step::Progress { next, chain_len }) => {
                    // redb's write lock is not fair: going straight into the next
                    // step could keep the node's own block writes waiting.
                    std::thread::sleep(STEP_PAUSE);
                    if next >= chain_len {
                        tracing::info!(
                            "Rich-list index: caught up at block {}",
                            next.saturating_sub(1)
                        );
                    } else if last_log.elapsed() > Duration::from_secs(15) {
                        tracing::info!(
                            "Rich-list index: {}/{} blocks ({}%)",
                            next,
                            chain_len,
                            next * 100 / chain_len.max(1)
                        );
                        last_log = Instant::now();
                    }
                }
                Ok(Step::Idle) => self.wait(IDLE_WAIT),
                Ok(Step::Missing(h)) => {
                    if logged_missing != Some(h) {
                        tracing::warn!(
                            "Rich-list index: block {} is not stored on this node (pruned?), so balances and \
                             supply cannot be indexed. It needs an archival node.",
                            h
                        );
                        logged_missing = Some(h);
                    }
                    self.wait(IDLE_WAIT);
                }
                Err(e) => {
                    tracing::warn!("Rich-list index: backfill step failed: {}", e);
                    self.wait(IDLE_WAIT);
                }
            }
        }
    }

    /// One backfill transaction: wipe first if the index is stale, then apply
    /// stored blocks from `next` towards `chain_len` for at most [`STEP_BUDGET`].
    pub fn step(&self, db: &Arc<Database>) -> Result<Step> {
        if !self.is_enabled() {
            return Ok(Step::Idle);
        }
        let txn = db.begin_write()?;
        let mut m = Meta::load(&txn)?;
        let mut changed = false;
        if m.stale || m.version != VERSION {
            wipe(&txn, &mut m)?;
            changed = true;
        }
        if m.next >= m.chain_len {
            if changed {
                m.store(&txn)?;
                txn.commit()?;
            }
            return Ok(Step::Idle);
        }
        let started = Instant::now();
        let first = m.next;
        let mut missing = None;
        {
            let batches = txn.open_table(super::BATCHES_TABLE)?;
            let mut w = Writer::open(&txn)?;
            while m.next < m.chain_len {
                let h = m.next;
                let batch = match batches.get(h)? {
                    Some(raw) => bincode::deserialize::<Batch>(raw.value()).ok(),
                    None => None,
                };
                let Some(batch) = batch else {
                    missing = Some(h);
                    break;
                };
                w.apply(&mut m, &batch)?;
                m.next = h + 1;
                m.tip_hash = batch.extension.final_hash;
                if started.elapsed() >= STEP_BUDGET || m.next - first >= STEP_MAX_BLOCKS {
                    break;
                }
            }
        }
        m.missing = missing.map(|h| h + 1).unwrap_or(0);
        m.store(&txn)?;
        txn.commit()?;
        Ok(match missing {
            Some(h) if m.next == first => Step::Missing(h),
            _ => Step::Progress {
                next: m.next,
                chain_len: m.chain_len,
            },
        })
    }

    /// Run backfill steps until there is nothing left to do. For tests and for
    /// callers that want the index current before they go on.
    pub fn catch_up(&self, db: &Arc<Database>) -> Result<Step> {
        loop {
            match self.step(db)? {
                Step::Progress { .. } => continue,
                other => return Ok(other),
            }
        }
    }
}

// ── Queries ─────────────────────────────────────────────────────────────────

/// The watermark and totals, or `None` if this database has never had the index.
pub fn meta(db: &Database) -> Result<Option<Meta>> {
    let txn = db.begin_read()?;
    let table = match txn.open_table(META) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let m = Meta::read(&table)?;
    Ok(if m.version == 0 { None } else { Some(m) })
}

/// One address's holdings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub address: [u8; 32],
    pub balance: u64,
    pub coins: u64,
}

/// A consistent view of the index: its totals and the largest holders, read in
/// one transaction so they describe the same block.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub meta: Meta,
    /// Largest first; ties broken by address, highest first.
    pub top: Vec<Holder>,
}

/// The `limit` largest holders, skipping any address `skip` matches, together
/// with the totals they were read against.
pub fn snapshot(
    db: &Database,
    limit: usize,
    skip: impl Fn(&[u8; 32]) -> bool,
) -> Result<Option<Snapshot>> {
    let txn = db.begin_read()?;
    let Ok(meta_t) = txn.open_table(META) else {
        return Ok(None);
    };
    let meta = Meta::read(&meta_t)?;
    if meta.version == 0 {
        return Ok(None);
    }
    let rank = txn.open_table(RANK)?;
    let addrs = txn.open_table(ADDRS)?;
    let mut top = Vec::with_capacity(limit);
    for entry in rank.iter()?.rev() {
        if top.len() >= limit {
            break;
        }
        let (key, _) = entry?;
        let (balance, address) = unpack_rank(key.value());
        if skip(&address) {
            continue;
        }
        let coins = addrs
            .get(&address)?
            .map(|g| unpack_holder(g.value()).1)
            .unwrap_or(0);
        top.push(Holder {
            address,
            balance,
            coins,
        });
    }
    Ok(Some(Snapshot { meta, top }))
}

/// An address's holdings, and its rank if within the first `rank_cap` holders
/// (counting only addresses `skip` does not match). Rank is a scan from the
/// top, so it is capped to bound the cost of looking up a small holder.
pub fn holder(
    db: &Database,
    address: &[u8; 32],
    rank_cap: usize,
    skip: impl Fn(&[u8; 32]) -> bool,
) -> Result<Option<(Holder, Option<u64>, Meta)>> {
    let txn = db.begin_read()?;
    let Ok(meta_t) = txn.open_table(META) else {
        return Ok(None);
    };
    let meta = Meta::read(&meta_t)?;
    if meta.version == 0 {
        return Ok(None);
    }
    let addrs = txn.open_table(ADDRS)?;
    let (balance, coins) = addrs
        .get(address)?
        .map(|g| unpack_holder(g.value()))
        .unwrap_or((0, 0));
    let mut rank = None;
    if balance > 0 && !skip(address) {
        let table = txn.open_table(RANK)?;
        let mine = rank_key(balance, address);
        let mut seen = 0u64;
        for entry in table.range::<&[u8; 40]>(&mine..)?.rev() {
            let (key, _) = entry?;
            let (_, other) = unpack_rank(key.value());
            if skip(&other) {
                continue;
            }
            seen += 1;
            if key.value() == &mine {
                rank = Some(seen);
                break;
            }
            if seen as usize >= rank_cap {
                break;
            }
        }
    }
    Ok(Some((
        Holder {
            address: *address,
            balance,
            coins,
        },
        rank,
        meta,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{hash, CoinbaseOutput, Extension, InputReveal, Predicate, Witness};
    use crate::storage::BatchStore;
    use std::collections::BTreeMap;

    // ── Fixtures ────────────────────────────────────────────────────────

    /// A block store over a fresh database. The index starts off, as in a node.
    fn store(dir: &std::path::Path) -> (Arc<Database>, BatchStore) {
        let db = Arc::new(Database::create(dir.join("t.redb")).unwrap());
        let txn = db.begin_write().unwrap();
        txn.open_table(crate::storage::BATCHES_TABLE).unwrap();
        txn.open_table(crate::storage::HEADERS_TABLE).unwrap();
        txn.open_table(crate::storage::FILTERS_TABLE).unwrap();
        txn.commit().unwrap();
        let bs = BatchStore::new(db.clone(), dir.join("batches")).unwrap();
        (db, bs)
    }

    fn script(n: u8) -> Predicate {
        Predicate::Script {
            bytecode: vec![0x51, n],
        }
    }

    fn addr(n: u8) -> [u8; 32] {
        script(n).address()
    }

    /// A coin paid to `owner`, identified by `tag` (its salt).
    #[derive(Clone, Copy)]
    struct Coin {
        owner: u8,
        value: u64,
        tag: u8,
    }

    fn salt(tag: u8) -> [u8; 32] {
        [tag; 32]
    }

    fn input(c: Coin) -> InputReveal {
        InputReveal {
            predicate: script(c.owner),
            value: c.value,
            salt: salt(c.tag),
            commitment: None,
        }
    }

    fn output(c: Coin) -> OutputData {
        OutputData::Standard {
            address: addr(c.owner),
            value: c.value,
            salt: salt(c.tag),
        }
    }

    fn reveal(inputs: &[Coin], outputs: Vec<OutputData>) -> Transaction {
        Transaction::Reveal {
            inputs: inputs.iter().map(|c| input(*c)).collect(),
            witnesses: inputs
                .iter()
                .map(|_| Witness::ScriptInputs(vec![]))
                .collect(),
            outputs,
            salt: [0xAB; 32],
        }
    }

    /// A block whose hash is derived from `prev` and `variant`, so two forks at
    /// the same height differ.
    fn block(prev: &[u8; 32], variant: u8, txs: Vec<Transaction>, coinbase: &[Coin]) -> Batch {
        let mut seed = prev.to_vec();
        seed.push(variant);
        Batch {
            prev_midstate: *prev,
            prev_header_hash: *prev,
            transactions: txs,
            extension: Extension {
                nonce: variant as u64,
                final_hash: hash(&seed),
            },
            coinbase: coinbase
                .iter()
                .map(|c| CoinbaseOutput {
                    address: addr(c.owner),
                    value: c.value,
                    salt: salt(c.tag),
                })
                .collect(),
            timestamp: 0,
            target: [0xff; 32],
            state_root: [0u8; 32],
        }
    }

    /// Store `chain` from height `from`, then let the backfill finish.
    fn save_all(db: &Arc<Database>, bs: &BatchStore, from: u64, chain: &[Batch]) {
        for (i, b) in chain.iter().enumerate() {
            bs.save(from + i as u64, b).unwrap();
        }
        bs.balances().catch_up(db).unwrap();
    }

    /// Everything the index holds, for comparing two indexes.
    #[derive(Debug, PartialEq, Eq)]
    struct Dump {
        coins: BTreeMap<[u8; 32], ([u8; 32], u64)>,
        addrs: BTreeMap<[u8; 32], (u64, u64)>,
        rank: Vec<(u64, [u8; 32])>,
        meta: Meta,
    }

    fn dump(db: &Database) -> Dump {
        let txn = db.begin_read().unwrap();
        let coins = txn.open_table(COINS).unwrap();
        let addrs = txn.open_table(ADDRS).unwrap();
        let rank = txn.open_table(RANK).unwrap();
        let mut meta = Meta::read(&txn.open_table(META).unwrap()).unwrap();
        meta.chain_len = 0; // a hook's view of the chain, not index content
        Dump {
            coins: coins
                .iter()
                .unwrap()
                .map(|e| {
                    let (k, v) = e.unwrap();
                    (*k.value(), unpack_coin(v.value()))
                })
                .collect(),
            addrs: addrs
                .iter()
                .unwrap()
                .map(|e| {
                    let (k, v) = e.unwrap();
                    (*k.value(), unpack_holder(v.value()))
                })
                .collect(),
            rank: rank
                .iter()
                .unwrap()
                .map(|e| unpack_rank(e.unwrap().0.value()))
                .collect(),
            meta,
        }
    }

    /// The index a node gets by storing `chain` from genesis in one go.
    fn fresh_dump(chain: &[Batch]) -> Dump {
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, chain);
        dump(&db)
    }

    /// Unspent value must be what the blocks issued minus what they burned.
    fn assert_balanced(db: &Database, issued: u64) {
        let m = meta(db).unwrap().unwrap();
        assert_eq!(m.anomalies, 0);
        assert_eq!(
            m.unspent_value,
            issued - m.burned,
            "unspent value vs issuance"
        );
        let d = dump(db);
        let by_coins: u64 = d.coins.values().map(|(_, v)| v).sum();
        let by_addrs: u64 = d.addrs.values().map(|(b, _)| b).sum();
        assert_eq!(by_coins, m.unspent_value);
        assert_eq!(by_addrs, m.unspent_value);
        assert_eq!(d.addrs.len() as u64, m.holders);
        assert_eq!(d.rank.len(), d.addrs.len());
        for (balance, a) in &d.rank {
            assert_eq!(d.addrs[a].0, *balance);
        }
    }

    /// Genesis, then a block paying 8 to address 1, then one where address 1
    /// pays 4 to address 2 and 2 back to itself, burning 1 and paying 1 in fee.
    /// Rewards here are pretend (16 a block); the totals only care that coinbase
    /// = reward + fees.
    fn base_chain() -> (Vec<Batch>, u64) {
        let g = block(
            &[0; 32],
            0,
            vec![],
            &[Coin {
                owner: 9,
                value: 16,
                tag: 1,
            }],
        );
        let b1 = block(
            &g.extension.final_hash,
            0,
            vec![],
            &[
                Coin {
                    owner: 1,
                    value: 8,
                    tag: 2,
                },
                Coin {
                    owner: 3,
                    value: 8,
                    tag: 3,
                },
            ],
        );
        let spend = reveal(
            &[Coin {
                owner: 1,
                value: 8,
                tag: 2,
            }],
            vec![
                output(Coin {
                    owner: 2,
                    value: 4,
                    tag: 4,
                }),
                output(Coin {
                    owner: 1,
                    value: 2,
                    tag: 5,
                }),
                OutputData::DataBurn {
                    payload: b"burn".to_vec(),
                    value_burned: 1,
                },
            ],
        );
        // Fee is 8 - 4 - 2 - 1 = 1, so the coinbase claims 16 + 1.
        let b2 = block(
            &b1.extension.final_hash,
            0,
            vec![spend],
            &[
                Coin {
                    owner: 3,
                    value: 16,
                    tag: 6,
                },
                Coin {
                    owner: 3,
                    value: 1,
                    tag: 7,
                },
            ],
        );
        (vec![g, b1, b2], 48)
    }

    // ── Tests ───────────────────────────────────────────────────────────

    #[test]
    fn applies_spends_burns_and_coinbase() {
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        let (chain, issued) = base_chain();
        save_all(&db, &bs, 0, &chain);
        assert_balanced(&db, issued);

        let m = meta(&db).unwrap().unwrap();
        assert!(m.is_current());
        assert_eq!(m.next, 3);
        assert_eq!(m.burned, 1);
        assert_eq!(m.burn_outputs, 1);
        assert_eq!(m.unspent_coins, 6);
        assert_eq!(m.tip_hash, chain[2].extension.final_hash);

        let snap = snapshot(&db, 10, |_| false).unwrap().unwrap();
        let top: Vec<(u64, u64)> = snap.top.iter().map(|h| (h.balance, h.coins)).collect();
        assert_eq!(top, vec![(25, 3), (16, 1), (4, 1), (2, 1)]);
        assert_eq!(snap.top[0].address, addr(3));
        assert_eq!(snap.top[3].address, addr(1));

        let (h, rank, _) = holder(&db, &addr(2), 100, |_| false).unwrap().unwrap();
        assert_eq!((h.balance, h.coins, rank), (4, 1, Some(3)));
        // Skipping an address takes it out of the ranking.
        let (_, rank, _) = holder(&db, &addr(2), 100, |a| *a == addr(9))
            .unwrap()
            .unwrap();
        assert_eq!(rank, Some(2));
        let skipped = snapshot(&db, 10, |a| *a == addr(9)).unwrap().unwrap();
        assert_eq!(skipped.top.len(), 3);
        // An address that holds nothing.
        let (h, rank, _) = holder(&db, &addr(77), 100, |_| false).unwrap().unwrap();
        assert_eq!((h.balance, h.coins, rank), (0, 0, None));
    }

    #[test]
    fn a_reorg_that_overwrites_blocks_matches_a_fresh_index() {
        let (mut chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        let b3 = block(
            &chain[2].extension.final_hash,
            0,
            vec![reveal(
                &[Coin {
                    owner: 2,
                    value: 4,
                    tag: 4,
                }],
                vec![output(Coin {
                    owner: 5,
                    value: 4,
                    tag: 8,
                })],
            )],
            &[Coin {
                owner: 3,
                value: 16,
                tag: 9,
            }],
        );
        chain.push(b3);
        save_all(&db, &bs, 0, &chain);

        // A competing fork from height 2: address 1's coin goes elsewhere.
        let alt2 = block(
            &chain[1].extension.final_hash,
            1,
            vec![reveal(
                &[Coin {
                    owner: 1,
                    value: 8,
                    tag: 2,
                }],
                vec![output(Coin {
                    owner: 6,
                    value: 8,
                    tag: 10,
                })],
            )],
            &[Coin {
                owner: 6,
                value: 16,
                tag: 11,
            }],
        );
        let alt3 = block(
            &alt2.extension.final_hash,
            1,
            vec![],
            &[Coin {
                owner: 6,
                value: 16,
                tag: 12,
            }],
        );
        let alt4 = block(
            &alt3.extension.final_hash,
            1,
            vec![],
            &[Coin {
                owner: 7,
                value: 16,
                tag: 13,
            }],
        );
        // perform_reorg's order: overwrite from the fork height up, no truncate.
        save_all(&db, &bs, 2, &[alt2.clone(), alt3.clone(), alt4.clone()]);

        let new_chain = vec![chain[0].clone(), chain[1].clone(), alt2, alt3, alt4];
        assert_eq!(dump(&db), fresh_dump(&new_chain));
        assert_balanced(&db, 16 * 5);
    }

    #[test]
    fn a_reorg_to_a_fork_with_fewer_blocks_ignores_the_leftovers() {
        let (chain, _) = base_chain();
        let mut long = chain.clone();
        for i in 0..3u8 {
            let prev = long.last().unwrap().extension.final_hash;
            long.push(block(
                &prev,
                0,
                vec![],
                &[Coin {
                    owner: 4,
                    value: 16,
                    tag: 20 + i,
                }],
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &long); // heights 0..=5

        // A heavier fork replacing 3..=5 with just 3..=4.
        let f3 = block(
            &long[2].extension.final_hash,
            2,
            vec![],
            &[Coin {
                owner: 8,
                value: 16,
                tag: 30,
            }],
        );
        let f4 = block(
            &f3.extension.final_hash,
            2,
            vec![],
            &[Coin {
                owner: 8,
                value: 16,
                tag: 31,
            }],
        );
        save_all(&db, &bs, 3, &[f3.clone(), f4.clone()]);
        // The old block 5 is still stored, but it is not on the chain.
        assert!(bs.load(5).unwrap().is_some());
        let mut fork = chain.clone();
        fork.extend([f3, f4]);
        assert_eq!(dump(&db), fresh_dump(&fork));
        let m = meta(&db).unwrap().unwrap();
        assert_eq!((m.next, m.chain_len), (5, 5));
        assert!(m.is_current());

        // The fork grows over the leftover.
        let f5 = block(
            &fork[4].extension.final_hash,
            2,
            vec![],
            &[Coin {
                owner: 8,
                value: 16,
                tag: 32,
            }],
        );
        save_all(&db, &bs, 5, &[f5.clone()]);
        fork.push(f5);
        assert_eq!(dump(&db), fresh_dump(&fork));
    }

    #[test]
    fn truncate_unapplies_the_deleted_blocks() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &chain);
        let extra = block(
            &chain[2].extension.final_hash,
            0,
            vec![],
            &[Coin {
                owner: 4,
                value: 16,
                tag: 40,
            }],
        );
        save_all(&db, &bs, 3, &[extra]);
        bs.truncate(2).unwrap(); // deletes heights 2 and 3
        assert_eq!(dump(&db), fresh_dump(&chain[..2]));
        let m = meta(&db).unwrap().unwrap();
        assert_eq!((m.next, m.chain_len), (2, 2));
    }

    #[test]
    fn backfill_builds_what_the_hooks_would_have() {
        let (chain, issued) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        // Blocks stored while the index is off...
        for (h, b) in chain.iter().enumerate() {
            bs.save(h as u64, b).unwrap();
        }
        assert_eq!(meta(&db).unwrap(), None);
        // ...are picked up once it is enabled.
        assert_eq!(bs.balances().enable(&db, 3).unwrap(), Enabled::Fresh);
        assert_eq!(bs.balances().catch_up(&db).unwrap(), Step::Idle);
        assert_eq!(dump(&db), fresh_dump(&chain));
        assert_balanced(&db, issued);
    }

    #[test]
    fn a_block_saved_ahead_of_the_backfill_waits_for_it() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.save(0, &chain[0]).unwrap();
        bs.save(1, &chain[1]).unwrap();
        bs.balances().enable(&db, 2).unwrap();
        // The node stores block 2 before the backfill has run at all.
        bs.save(2, &chain[2]).unwrap();
        let m = meta(&db).unwrap().unwrap();
        assert_eq!((m.next, m.chain_len), (0, 3));
        bs.balances().catch_up(&db).unwrap();
        assert_eq!(dump(&db), fresh_dump(&chain));
    }

    #[test]
    fn the_worker_thread_catches_up_on_its_own() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.save(0, &chain[0]).unwrap();
        bs.balances().enable(&db, 1).unwrap();
        bs.balances().spawn_worker(db.clone());
        // Blocks arriving ahead of the worker wake it.
        bs.save(1, &chain[1]).unwrap();
        bs.save(2, &chain[2]).unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while !meta(&db)
            .unwrap()
            .map_or(false, |m| m.is_current() && m.next == 3)
        {
            assert!(Instant::now() < deadline, "the worker did not catch up");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(dump(&db), fresh_dump(&chain));
    }

    #[test]
    fn restarting_on_the_same_chain_resumes() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &chain);
        drop(bs);

        // A new process: the switch starts off again.
        let bs = BatchStore::new(db.clone(), dir.path().join("batches")).unwrap();
        assert!(!bs.balances().is_enabled());
        assert_eq!(
            bs.balances().enable(&db, 3).unwrap(),
            Enabled::Resumed { next: 3 }
        );
        assert_eq!(dump(&db), fresh_dump(&chain));
    }

    #[test]
    fn a_reorg_while_the_index_was_off_forces_a_rebuild() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &chain);
        drop(bs);

        // Restarted without the index; a reorg replaces block 2.
        let bs = BatchStore::new(db.clone(), dir.path().join("batches")).unwrap();
        let alt2 = block(
            &chain[1].extension.final_hash,
            1,
            vec![],
            &[Coin {
                owner: 6,
                value: 16,
                tag: 50,
            }],
        );
        bs.save(2, &alt2).unwrap();

        // Back on: the index notices and rebuilds.
        match bs.balances().enable(&db, 3).unwrap() {
            Enabled::Rebuild { .. } => {}
            other => panic!("expected a rebuild, got {other:?}"),
        }
        bs.balances().catch_up(&db).unwrap();
        let new_chain = vec![chain[0].clone(), chain[1].clone(), alt2];
        assert_eq!(dump(&db), fresh_dump(&new_chain));
    }

    #[test]
    fn an_index_ahead_of_the_chain_is_rebuilt() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &chain);
        drop(bs);
        let bs = BatchStore::new(db.clone(), dir.path().join("batches")).unwrap();
        // The node's saved state says the chain is two blocks long.
        assert!(matches!(
            bs.balances().enable(&db, 2).unwrap(),
            Enabled::Rebuild { .. }
        ));
        bs.balances().catch_up(&db).unwrap();
        assert_eq!(dump(&db), fresh_dump(&chain[..2]));
    }

    #[test]
    fn pruned_history_stops_the_backfill() {
        let (chain, _) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        save_all(&db, &bs, 0, &chain);
        bs.prune_tail(1).unwrap(); // genesis is gone
        bs.balances().enable(&db, 3).unwrap();
        assert_eq!(bs.balances().catch_up(&db).unwrap(), Step::Missing(0));
        let m = meta(&db).unwrap().unwrap();
        assert_eq!((m.next, m.missing), (0, 1));
        assert!(!m.is_current());
    }

    #[test]
    fn pruning_after_the_index_is_built_changes_nothing() {
        let (chain, issued) = base_chain();
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &chain);
        let before = dump(&db);
        bs.prune_tail(2).unwrap();
        assert_eq!(dump(&db), before);
        assert_balanced(&db, issued);
    }

    #[test]
    fn spending_everything_removes_the_holder() {
        let g = block(
            &[0; 32],
            0,
            vec![],
            &[Coin {
                owner: 1,
                value: 16,
                tag: 1,
            }],
        );
        let b1 = block(
            &g.extension.final_hash,
            0,
            vec![reveal(
                &[Coin {
                    owner: 1,
                    value: 16,
                    tag: 1,
                }],
                vec![output(Coin {
                    owner: 2,
                    value: 8,
                    tag: 2,
                })],
            )],
            &[
                Coin {
                    owner: 2,
                    value: 16,
                    tag: 3,
                },
                Coin {
                    owner: 2,
                    value: 8,
                    tag: 4,
                },
            ],
        );
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &[g, b1]);
        let d = dump(&db);
        assert!(!d.addrs.contains_key(&addr(1)));
        assert_eq!(d.addrs[&addr(2)], (32, 3));
        assert_eq!(d.meta.holders, 1);
        assert_balanced(&db, 32);
    }

    #[test]
    fn state_threads_and_zero_value_burns_are_not_value() {
        let g = block(
            &[0; 32],
            0,
            vec![],
            &[Coin {
                owner: 1,
                value: 16,
                tag: 1,
            }],
        );
        let thread_in = InputReveal {
            predicate: script(4),
            value: 0,
            salt: salt(9),
            commitment: Some([7; 32]),
        };
        let tx = Transaction::Reveal {
            inputs: vec![
                input(Coin {
                    owner: 1,
                    value: 16,
                    tag: 1,
                }),
                thread_in,
            ],
            witnesses: vec![Witness::ScriptInputs(vec![]), Witness::ScriptInputs(vec![])],
            outputs: vec![
                output(Coin {
                    owner: 2,
                    value: 16,
                    tag: 2,
                }),
                OutputData::Confidential {
                    address: addr(4),
                    commitment: [8; 32],
                    salt: salt(10),
                },
                OutputData::DataBurn {
                    payload: b"ledger".to_vec(),
                    value_burned: 0,
                },
            ],
            salt: [1; 32],
        };
        let b1 = block(
            &g.extension.final_hash,
            0,
            vec![tx],
            &[Coin {
                owner: 3,
                value: 16,
                tag: 3,
            }],
        );
        let dir = tempfile::tempdir().unwrap();
        let (db, bs) = store(dir.path());
        bs.balances().enable(&db, 0).unwrap();
        save_all(&db, &bs, 0, &[g, b1]);
        let m = meta(&db).unwrap().unwrap();
        assert_eq!((m.burned, m.burn_outputs, m.anomalies), (0, 0, 0));
        assert_balanced(&db, 32);
    }
}
