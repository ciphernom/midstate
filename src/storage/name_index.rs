//! The MSN-1 name registry as the chain has it (see `core::names`).
//!
//! Follows the registry from its genesis coin through every spend of the
//! current thread, mirroring each into:
//!
//! * [`NODES`]: the Merkle tree, `level << 32 | index → hash`. Empty subtrees
//!   are not stored.
//! * [`LEAVES`]: `slot → key ‖ next ‖ expires ‖ owner ‖ data`.
//! * [`RECORDS`]: `slot → since ‖ registered ‖ records`: `since` the height
//!   its holder got it, `registered` the height its current registration
//!   began (see [`Record::registered`]).
//! * [`DEALS`]: `slot → terms`, for names offered or rented out.
//! * [`OWNERS`]: `owner ‖ slot`, every name by its control key.
//! * [`KEYS`]: `key → slot ‖ name`, in key order, so a new name's neighbour is
//!   the key just below it.
//! * [`LOG`]: each spend followed, with what undoes it.
//! * [`META`]: the watermark, the current thread, the lowest free slot.
//!
//! Consistency is the balance index's: the hooks run inside the write that
//! stores or deletes a block, and a reorg rewinds before it applies. Unlike
//! there, rewinding reads no block: [`LOG`] holds each spend's own undo, so a
//! reorg of any depth unwinds exactly, on a pruned node too.
//!
//! Every spend is checked against the root the chain gives it. A mismatch
//! would mean this mirror is wrong: the index stops there and says so rather
//! than serve a wrong owner.
//!
//! Always on. A block with no registry spend costs a metadata read and write.

use anyhow::{anyhow, bail, Result};
use redb::{Database, ReadableTable, Table, TableDefinition, WriteTransaction};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::core::names::{self, By, Deal, Leaf, Op};
use crate::core::types::{Batch, BatchHeader, InputReveal, OutputData, Transaction, Witness};

pub const NODES: TableDefinition<u64, &[u8; 32]> = TableDefinition::new("names_nodes");
pub const LEAVES: TableDefinition<u64, &[u8; 121]> = TableDefinition::new("names_leaves");
/// `slot → terms`, for slots with any (`Deal` other than none).
pub const DEALS: TableDefinition<u64, &[u8]> = TableDefinition::new("names_deals");
/// `owner ‖ slot → ()`, so a wallet finds every name its keys hold.
pub const OWNERS: TableDefinition<&[u8; 40], ()> = TableDefinition::new("names_owners");
pub const RECORDS: TableDefinition<u64, &[u8]> = TableDefinition::new("names_records");
pub const KEYS: TableDefinition<u64, &[u8]> = TableDefinition::new("names_keys");
pub const LOG: TableDefinition<u64, &[u8]> = TableDefinition::new("names_log");
pub const META: TableDefinition<&str, &[u8]> = TableDefinition::new("names_meta");

/// Bump to rebuild every node's index on upgrade.
const VERSION: u64 = 5;
const STEP_BUDGET: Duration = Duration::from_millis(250);
const STEP_PAUSE: Duration = Duration::from_millis(20);
const IDLE_WAIT: Duration = Duration::from_secs(30);

/// The registry's current coin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Thread {
    pub id: [u8; 32],
    pub root: [u8; 32],
    pub salt: [u8; 32],
}

impl Thread {
    fn of(o: &OutputData) -> Option<Self> {
        match o {
            OutputData::Confidential { address, commitment, salt } if *address == names::address() => {
                Some(Thread { id: o.coin_id()?, root: *commitment, salt: *salt })
            }
            _ => None,
        }
    }

    pub fn input(&self) -> InputReveal {
        names::tip_input(self.root, self.salt)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Meta {
    pub version: u64,
    /// Every height below this is applied.
    pub next: u64,
    pub chain_len: u64,
    /// Hash of block `next - 1`.
    pub tip_hash: [u8; 32],
    /// Rebuild from [`names::START_HEIGHT`].
    pub stale: bool,
    /// `h + 1` if block `h` is not stored here.
    pub missing: u64,
    /// `h + 1` if block `h` disagreed with this mirror; nothing past it is applied.
    pub broken: u64,
    /// `None` until the genesis coin appears.
    pub thread: Option<Thread>,
    pub names: u64,
    /// The lowest empty slot above the sentinel's.
    pub free: u64,
    pub log_len: u64,
}

impl Meta {
    fn read<T: ReadableTable<&'static str, &'static [u8]>>(t: &T) -> Result<Self> {
        Ok(match t.get("meta")? {
            Some(g) => bincode::deserialize(g.value()).unwrap_or_default(),
            None => Self::default(),
        })
    }
    fn load(txn: &WriteTransaction) -> Result<Self> {
        Self::read(&txn.open_table(META)?)
    }
    fn store(&self, txn: &WriteTransaction) -> Result<()> {
        txn.open_table(META)?.insert("meta", bincode::serialize(self)?.as_slice())?;
        Ok(())
    }
    pub fn ready(&self) -> bool {
        self.version == VERSION && !self.stale && self.broken == 0 && self.missing == 0 && self.next >= self.chain_len
    }
}

#[derive(Serialize, Deserialize)]
enum Logged {
    Genesis,
    /// With the slot's `since ‖ registered ‖ records` before a change.
    Spend(Op, Option<Vec<u8>>),
}

#[derive(Serialize, Deserialize)]
struct Entry {
    height: u64,
    before: Option<Thread>,
    what: Logged,
}

fn slot(level: usize, index: u64) -> u64 {
    (level as u64) << 32 | index
}

/// A stored tree, as `core::names` reads one.
struct Tree<'a, T>(&'a T);

impl<T: ReadableTable<u64, &'static [u8; 32]>> names::Nodes for Tree<'_, T> {
    fn stored(&self, level: usize, index: u64) -> Result<Option<[u8; 32]>> {
        Ok(self.0.get(slot(level, index))?.map(|g| *g.value()))
    }
}

// ── Applying and undoing ────────────────────────────────────────────────────

struct Writer<'t> {
    nodes: Table<'t, u64, &'static [u8; 32]>,
    leaves: Table<'t, u64, &'static [u8; 121]>,
    records: Table<'t, u64, &'static [u8]>,
    deals: Table<'t, u64, &'static [u8]>,
    owners: Table<'t, &'static [u8; 40], ()>,
    keys: Table<'t, u64, &'static [u8]>,
    log: Table<'t, u64, &'static [u8]>,
}

impl<'t> Writer<'t> {
    fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(Self {
            nodes: txn.open_table(NODES)?,
            leaves: txn.open_table(LEAVES)?,
            records: txn.open_table(RECORDS)?,
            deals: txn.open_table(DEALS)?,
            owners: txn.open_table(OWNERS)?,
            keys: txn.open_table(KEYS)?,
            log: txn.open_table(LOG)?,
        })
    }

    /// Set a slot's leaf; returns the new root.
    fn put(&mut self, pos: u64, leaf: Option<Leaf>) -> Result<[u8; 32]> {
        let owned = |owner: &[u8; 32]| -> [u8; 40] {
            let mut k = [0u8; 40];
            k[..32].copy_from_slice(owner);
            k[32..].copy_from_slice(&pos.to_be_bytes());
            k
        };
        let old = self.leaves.get(pos)?.map(|g| Leaf::from_bytes(g.value()));
        if let Some(o) = old.filter(|o| o.owner != [0; 32]) {
            self.owners.remove(&owned(&o.owner))?;
        }
        if let Some(n) = leaf.filter(|n| n.owner != [0; 32]) {
            self.owners.insert(&owned(&n.owner), ())?;
        }
        let path = names::path(&Tree(&self.nodes), pos, leaf.map(|l| l.hash()))?;
        for &(l, i, h) in &path {
            if h == names::empty(l) {
                self.nodes.remove(slot(l, i))?;
            } else {
                self.nodes.insert(slot(l, i), &h)?;
            }
        }
        match leaf {
            Some(l) => {
                self.leaves.insert(pos, &l.bytes())?;
            }
            None => {
                self.leaves.remove(pos)?;
            }
        }
        Ok(path.last().expect("a path ends at the root").2)
    }

    fn record(&mut self, m: &mut Meta, height: u64, before: Option<Thread>, what: Logged) -> Result<()> {
        self.log.insert(m.log_len, bincode::serialize(&Entry { height, before, what })?.as_slice())?;
        m.log_len += 1;
        Ok(())
    }

    fn set_terms(&mut self, pos: u64, terms: &[u8]) -> Result<()> {
        if terms == Deal::None.terms().as_slice() {
            self.deals.remove(pos)?;
        } else {
            self.deals.insert(pos, terms)?;
        }
        Ok(())
    }

    /// A new name in slot `pos`: held, and registered, from `height`.
    fn set_name(&mut self, pos: u64, name: &[u8], height: u64, records: &[u8]) -> Result<()> {
        self.keys.insert(names::ord(&names::key(name)), [&pos.to_le_bytes()[..], name].concat().as_slice())?;
        self.records.insert(pos, [&height.to_le_bytes()[..], &height.to_le_bytes()[..], records].concat().as_slice())?;
        Ok(())
    }

    fn apply(&mut self, m: &mut Meta, height: u64, batch: &Batch) -> Result<()> {
        if height < names::START_HEIGHT {
            return Ok(());
        }
        let genesis = names::genesis().coin_id();
        for tx in &batch.transactions {
            let (inputs, witnesses, outputs): (&[InputReveal], &[Witness], &[OutputData]) = match tx {
                Transaction::Reveal { inputs, witnesses, outputs, .. } => (inputs, witnesses, outputs),
                Transaction::Consolidate { outputs, .. } => (&[], &[], outputs),
                Transaction::Commit { .. } => continue,
            };
            let Some(thread) = m.thread else {
                if let Some(t) = outputs.iter().find(|o| o.coin_id() == genesis).and_then(Thread::of) {
                    let mut root = [0; 32];
                    for (pos, leaf) in names::genesis_leaves() {
                        root = self.put(pos, Some(leaf))?;
                    }
                    if root != t.root {
                        return self.broke(m, height, "a genesis this mirror does not compute");
                    }
                    self.keys.insert(0, 0u64.to_le_bytes().as_slice())?;
                    for (i, x) in names::seeded().iter().enumerate() {
                        self.set_name(i as u64 + 1, x.name.as_bytes(), height, &[])?;
                    }
                    m.free = names::seeded().len() as u64 + 1;
                    self.record(m, height, None, Logged::Genesis)?;
                    m.thread = Some(t);
                }
                continue;
            };
            let Some(i) = inputs.iter().position(|inp| inp.coin_id() == thread.id) else { continue };
            let Witness::ScriptInputs(items) = &witnesses[i];
            let (Some(op), Some(next)) = (names::decode(items), outputs.first().and_then(Thread::of)) else {
                return self.broke(m, height, "a spend this mirror cannot read");
            };
            let mut root = thread.root;
            for (pos, leaf) in op.writes() {
                root = self.put(pos, leaf)?;
            }
            if root != next.root {
                return self.broke(m, height, "a root this mirror does not compute");
            }
            let before = match &op {
                Op::Register { name, pos, records, terms, .. } => {
                    self.set_name(*pos, name, height, records)?;
                    self.set_terms(*pos, terms)?;
                    m.names += 1;
                    while self.leaves.get(m.free)?.is_some() {
                        m.free += 1;
                    }
                    None
                }
                Op::Change { by, pos, records, terms, leaf, owner, .. } => {
                    let Some(old) = self.records.get(*pos)?.map(|g| g.value().to_vec()) else {
                        return self.broke(m, height, "a change to a slot this mirror has no records for");
                    };
                    if old.len() < 16 {
                        return self.broke(m, height, "records this mirror cannot read");
                    }
                    // A reclaim or a sale starts a new holding; the holder's own
                    // change, or a rental, does not.
                    let since = match by {
                        By::Lapse | By::Buy => height.to_le_bytes().to_vec(),
                        By::Owner | By::Rent => old[..8].to_vec(),
                    };
                    let fresh = new_registration(*by, &leaf.owner, owner);
                    let registered = if fresh { height.to_le_bytes().to_vec() } else { old[8..16].to_vec() };
                    self.records.insert(*pos, [&since[..], &registered[..], records].concat().as_slice())?;
                    self.set_terms(*pos, terms)?;
                    Some(old)
                }
            };
            self.record(m, height, Some(thread), Logged::Spend(op, before))?;
            m.thread = Some(next);
        }
        Ok(())
    }

    fn broke(&mut self, m: &mut Meta, height: u64, what: &str) -> Result<()> {
        tracing::error!("Name index: block {height} has {what}; names are unavailable until this is fixed");
        m.broken = height + 1;
        Ok(())
    }

    /// Undo the newest logged spend.
    fn undo(&mut self, m: &mut Meta, e: &Entry) -> Result<()> {
        match &e.what {
            Logged::Genesis => {
                for (pos, _) in names::genesis_leaves().into_iter().rev() {
                    self.put(pos, None)?;
                    self.records.remove(pos)?;
                }
                for x in names::seeded() {
                    self.keys.remove(names::ord(&names::key(x.name.as_bytes())))?;
                }
                self.keys.remove(0)?;
                m.free = 0;
            }
            Logged::Spend(op, before) => {
                for (pos, leaf) in op.undoes() {
                    self.put(pos, leaf)?;
                }
                match (op, before) {
                    (Op::Register { name, pos, .. }, _) => {
                        self.keys.remove(names::ord(&names::key(name)))?;
                        self.records.remove(*pos)?;
                        self.deals.remove(*pos)?;
                        m.names -= 1;
                        m.free = m.free.min(*pos);
                    }
                    (Op::Change { pos, oterms, .. }, Some(old)) => {
                        self.records.insert(*pos, old.as_slice())?;
                        self.set_terms(*pos, oterms)?;
                    }
                    (Op::Change { .. }, None) => bail!("name log: a change without its old records"),
                }
            }
        }
        m.thread = e.before;
        Ok(())
    }
}

/// Whether a change starts a new registration of a name: a claim of a name
/// nobody held (a premium name, owner zero), or a reclaim after it lapsed by
/// a key that did not hold it. A sale, a transfer, a rental, a late renewal
/// (the owner's update after expiry) or the holder reclaiming its own name
/// goes on with the registration before ([`Record::registered`]).
fn new_registration(by: By, old_owner: &[u8; 32], new_owner: &[u8; 32]) -> bool {
    *old_owner == [0; 32] || (by == By::Lapse && new_owner != old_owner)
}

/// Undo every spend at or above height `to`, so that `next <= to`.
fn rewind(txn: &WriteTransaction, m: &mut Meta, to: u64) -> Result<()> {
    let mut w = Writer::open(txn)?;
    while m.log_len > 0 {
        let raw = w.log.get(m.log_len - 1)?.ok_or_else(|| anyhow!("name log entry missing"))?.value().to_vec();
        let e: Entry = bincode::deserialize(&raw)?;
        if e.height < to {
            break;
        }
        w.undo(m, &e)?;
        w.log.remove(m.log_len - 1)?;
        m.log_len -= 1;
    }
    if m.broken > to {
        m.broken = 0;
    }
    if m.next > to {
        m.next = to;
        m.tip_hash = header_hash(txn, to.wrapping_sub(1))?.unwrap_or_default();
    }
    Ok(())
}

fn header_hash(txn: &WriteTransaction, height: u64) -> Result<Option<[u8; 32]>> {
    let t = txn.open_table(super::HEADERS_TABLE)?;
    let h = t.get(height)?.and_then(|raw| bincode::deserialize::<BatchHeader>(raw.value()).ok());
    Ok(h.map(|h| h.extension.final_hash))
}

/// Empty the index; it starts again from [`names::START_HEIGHT`].
fn wipe(txn: &WriteTransaction, m: &mut Meta) -> Result<()> {
    txn.delete_table(NODES)?;
    txn.delete_table(LEAVES)?;
    txn.delete_table(RECORDS)?;
    txn.delete_table(DEALS)?;
    txn.delete_table(OWNERS)?;
    txn.delete_table(KEYS)?;
    txn.delete_table(LOG)?;
    Writer::open(txn)?;
    *m = Meta { version: VERSION, next: names::START_HEIGHT.min(m.chain_len), chain_len: m.chain_len, ..Meta::default() };
    Ok(())
}

// ── The shared handle ───────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct Inner {
    enabled: AtomicBool,
    worker_started: AtomicBool,
    wake: Mutex<bool>,
    cv: Condvar,
}

/// The index's hooks and backfill worker, shared by every clone of the
/// [`BatchStore`](super::BatchStore).
#[derive(Clone, Debug, Default)]
pub struct NameIndex {
    inner: Arc<Inner>,
}

impl NameIndex {
    fn enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Acquire)
    }

    fn wake(&self) {
        *self.inner.wake.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.inner.cv.notify_all();
    }

    fn wait(&self, timeout: Duration) {
        let flag = self.inner.wake.lock().unwrap_or_else(|e| e.into_inner());
        let (mut flag, _) = self.inner.cv.wait_timeout_while(flag, timeout, |w| !*w).unwrap_or_else(|e| e.into_inner());
        *flag = false;
    }

    /// Hook: `batch` is about to be stored at `height`, inside `txn`.
    pub fn on_save(&self, txn: &WriteTransaction, height: u64, batch: &Batch) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let mut m = Meta::load(txn)?;
        m.chain_len = height + 1;
        if height < m.next {
            rewind(txn, &mut m, height)?;
        }
        if !m.stale && m.broken == 0 && height == m.next {
            if height >= names::START_HEIGHT {
                Writer::open(txn)?.apply(&mut m, height, batch)?;
            }
            if m.broken == 0 {
                m.next = height + 1;
                m.tip_hash = batch.extension.final_hash;
                m.missing = 0;
            }
        } else {
            self.wake();
        }
        m.store(txn)
    }

    /// Hook: every stored height `>= new_len` is about to be deleted, inside `txn`.
    pub fn on_truncate(&self, txn: &WriteTransaction, new_len: u64) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let mut m = Meta::load(txn)?;
        m.chain_len = m.chain_len.min(new_len);
        rewind(txn, &mut m, new_len)?;
        m.store(txn)
    }

    /// Turn the index on for a chain of `chain_len` blocks and check what is
    /// on disk against it. Call before the node stores any block.
    pub fn enable(&self, db: &Arc<Database>, chain_len: u64) -> Result<()> {
        let txn = db.begin_write()?;
        Writer::open(&txn)?;
        let mut m = Meta::load(&txn)?;
        m.chain_len = chain_len;
        if m.version != VERSION {
            wipe(&txn, &mut m)?;
        } else if m.next > chain_len {
            rewind(&txn, &mut m, chain_len)?;
        } else if m.next > names::START_HEIGHT && header_hash(&txn, m.next - 1)? != Some(m.tip_hash) {
            tracing::warn!("Name index: block {} changed while the index was off; rebuilding", m.next - 1);
            m.stale = true;
        }
        m.missing = 0;
        m.store(&txn)?;
        txn.commit()?;
        self.inner.enabled.store(true, Ordering::Release);
        Ok(())
    }

    /// Start the thread that keeps the index caught up. Idempotent. It holds
    /// the database only while it works, so closing the store ends it.
    pub fn spawn_worker(&self, db: Arc<Database>) {
        if self.inner.worker_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = self.clone();
        let db = Arc::downgrade(&db);
        if let Err(e) = std::thread::Builder::new().name("name-index".into()).spawn(move || me.worker(db)) {
            tracing::warn!("Name index: could not start its thread: {e}");
            self.inner.worker_started.store(false, Ordering::Release);
        }
    }

    fn worker(&self, db: std::sync::Weak<Database>) {
        let mut said_missing = None;
        loop {
            let Some(strong) = db.upgrade() else { return };
            let stepped = self.step(&strong);
            drop(strong);
            match stepped {
                Ok(Some(next)) if next.missing == 0 => std::thread::sleep(STEP_PAUSE),
                Ok(Some(m)) => {
                    if said_missing != Some(m.missing) {
                        tracing::warn!(
                            "Name index: block {} is not stored here (pruned?), so names cannot be followed past it",
                            m.missing - 1
                        );
                        said_missing = Some(m.missing);
                    }
                    self.wait(IDLE_WAIT);
                }
                Ok(None) => self.wait(IDLE_WAIT),
                Err(e) => {
                    tracing::warn!("Name index: {e:#}");
                    self.wait(IDLE_WAIT);
                }
            }
        }
    }

    /// One backfill transaction. `None` when there is nothing to do.
    pub fn step(&self, db: &Arc<Database>) -> Result<Option<Meta>> {
        if !self.enabled() {
            return Ok(None);
        }
        let txn = db.begin_write()?;
        let mut m = Meta::load(&txn)?;
        let mut changed = false;
        if m.stale || m.version != VERSION {
            wipe(&txn, &mut m)?;
            changed = true;
        }
        if m.broken != 0 || m.next >= m.chain_len {
            if changed {
                m.store(&txn)?;
                txn.commit()?;
            }
            return Ok(None);
        }
        let started = Instant::now();
        let first = m.next;
        {
            let batches = txn.open_table(super::BATCHES_TABLE)?;
            let mut w = Writer::open(&txn)?;
            while m.next < m.chain_len && m.broken == 0 && started.elapsed() < STEP_BUDGET {
                let h = m.next;
                // Below the start nothing can touch the registry: skip unread.
                if h < names::START_HEIGHT {
                    m.next = names::START_HEIGHT.min(m.chain_len);
                    continue;
                }
                let Some(batch) = batches.get(h)?.and_then(|raw| bincode::deserialize::<Batch>(raw.value()).ok()) else {
                    m.missing = h + 1;
                    break;
                };
                w.apply(&mut m, h, &batch)?;
                if m.broken == 0 {
                    m.next = h + 1;
                    m.tip_hash = batch.extension.final_hash;
                    m.missing = 0;
                }
            }
        }
        m.store(&txn)?;
        txn.commit()?;
        Ok(if m.next == first && m.missing == 0 && !changed { None } else { Some(m) })
    }

    /// Backfill until caught up. For tests, and callers that need it now.
    pub fn catch_up(&self, db: &Arc<Database>) -> Result<()> {
        while let Some(m) = self.step(db)? {
            if m.missing != 0 {
                break;
            }
        }
        Ok(())
    }
}

// ── Queries ─────────────────────────────────────────────────────────────────

pub fn meta(db: &Database) -> Result<Meta> {
    let txn = db.begin_read()?;
    match txn.open_table(META) {
        Ok(t) => Meta::read(&t),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(Meta::default()),
        Err(e) => Err(e.into()),
    }
}

/// A name as the chain has it now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub name: String,
    pub slot: u64,
    /// The control key: signs changes, never payments. Zero for a premium
    /// name never claimed.
    pub owner: [u8; 32],
    /// The last height it is held at; from the next, anyone may reclaim it.
    pub expires: u64,
    pub tier: u8,
    /// When its holder got it: its registration, or the sale or reclaim
    /// that gave it to this holder.
    pub since: u64,
    /// When its current registration began: when it was registered, or
    /// claimed or reclaimed after lapsing by a key that did not hold it.
    /// Sales, transfers, rentals and late renewals do not start a new one.
    /// MSB-1 backing burned for the name counts from here: a name taken
    /// over after it lapsed starts with none.
    pub registered: u64,
    pub records: Vec<names::Rec>,
    pub terms: Vec<u8>,
    pub deal: Deal,
    /// The height a transaction sent now is checked at.
    pub height: u64,
}

impl Record {
    pub fn lapsed(&self) -> bool {
        self.height > self.expires
    }

    /// What it costs to hold it from now (or from its expiry, if later) for `years`.
    pub fn quote(&self, years: u64) -> (u64, u64) {
        (names::cost(self.tier, years), self.expires.max(self.height) + years * names::YEAR)
    }
}

fn not_ready(m: &Meta) -> anyhow::Error {
    if m.broken != 0 {
        anyhow!("the name index stopped at block {}; see the node log", m.broken - 1)
    } else if m.missing != 0 {
        anyhow!("the name index needs block {}, which this node does not store", m.missing - 1)
    } else {
        anyhow!("the name index is catching up (block {} of {})", m.next, m.chain_len)
    }
}

/// The slot and leaf holding the name's key, and whether that is this name
/// (not another sharing its key). `name` is canonical.
fn find(txn: &redb::ReadTransaction, name: &str) -> Result<Option<(u64, Leaf, bool)>> {
    let keys = txn.open_table(KEYS)?;
    let Some(v) = keys.get(names::ord(&names::key(name.as_bytes())))? else { return Ok(None) };
    let v = v.value();
    let slot = u64::from_le_bytes(v[..8].try_into()?);
    let raw = txn.open_table(LEAVES)?.get(slot)?.ok_or_else(|| anyhow!("name index: slot {slot} missing"))?.value().to_owned();
    Ok(Some((slot, Leaf::from_bytes(&raw), &v[8..] == name.as_bytes())))
}

fn record(txn: &redb::ReadTransaction, m: &Meta, name: String, slot: u64, leaf: Leaf) -> Result<Record> {
    let raw = txn.open_table(RECORDS)?.get(slot)?.ok_or_else(|| anyhow!("name index: records {slot} missing"))?.value().to_vec();
    let terms = txn.open_table(DEALS)?.get(slot)?.map_or_else(|| Deal::None.terms(), |g| g.value().to_vec());
    Ok(Record {
        name,
        slot,
        owner: leaf.owner,
        expires: leaf.expires,
        tier: leaf.tier,
        since: u64::from_le_bytes(raw[..8].try_into()?),
        registered: u64::from_le_bytes(raw[8..16].try_into()?),
        records: names::parse(&raw[16..]).unwrap_or_default(),
        deal: Deal::parse(&terms),
        terms,
        height: m.next,
    })
}

fn ready(db: &Database) -> Result<Meta> {
    let m = meta(db)?;
    if !m.ready() {
        return Err(not_ready(&m));
    }
    Ok(m)
}

/// A name as the chain has it — held, lapsed or a premium name never
/// claimed — or `None` if it was never registered.
pub fn lookup(db: &Database, name: &str) -> Result<Option<Record>> {
    let name = names::normalize(name)?;
    let m = ready(db)?;
    if m.thread.is_none() {
        return Ok(None);
    }
    let txn = db.begin_read()?;
    let Some((slot, leaf, true)) = find(&txn, &name)? else { return Ok(None) };
    Ok(Some(record(&txn, &m, name, slot, leaf)?))
}

/// Names offered for sale or rent now, and held (not lapsed).
pub fn listings(db: &Database) -> Result<Vec<Record>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let leaves = txn.open_table(LEAVES)?;
    let keys = txn.open_table(KEYS)?;
    let mut out = Vec::new();
    for entry in txn.open_table(DEALS)?.iter()? {
        let (slot, _) = entry?;
        let slot = slot.value();
        let Some(raw) = leaves.get(slot)? else { continue };
        let leaf = Leaf::from_bytes(raw.value());
        let Some(v) = keys.get(names::ord(&leaf.key))? else { continue };
        let name = String::from_utf8_lossy(&v.value()[8..]).into_owned();
        let r = record(&txn, &m, name, slot, leaf)?;
        if !r.lapsed() && matches!(r.deal, Deal::Sale { .. } | Deal::Rent { .. }) && names::normalize(&r.name).ok().as_deref() == Some(&r.name) {
            out.push(r);
        }
    }
    Ok(out)
}

/// Every name a control key holds, lapsed or not.
pub fn names_of(db: &Database, owner: &[u8; 32]) -> Result<Vec<Record>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let (mut lo, mut hi) = ([0u8; 40], [0xffu8; 40]);
    lo[..32].copy_from_slice(owner);
    hi[..32].copy_from_slice(owner);
    let leaves = txn.open_table(LEAVES)?;
    let keys = txn.open_table(KEYS)?;
    let mut out = Vec::new();
    for entry in txn.open_table(OWNERS)?.range::<&[u8; 40]>(&lo..=&hi)? {
        let slot = u64::from_be_bytes(entry?.0.value()[32..].try_into()?);
        let Some(raw) = leaves.get(slot)? else { continue };
        let leaf = Leaf::from_bytes(raw.value());
        let Some(v) = keys.get(names::ord(&leaf.key))? else { continue };
        let name = String::from_utf8_lossy(&v.value()[8..]).into_owned();
        out.push(record(&txn, &m, name, slot, leaf)?);
    }
    Ok(out)
}

/// What a client needs to check a name against the registry's root without
/// trusting the index that answered: the slot's leaf, the records and terms
/// it commits to, and its Merkle proof up to the root of the registry's
/// current thread. For a name never registered, the leaf is its predecessor
/// in key order, whose `next` skips over the name's key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameProof {
    /// The name's own leaf (`true`), or its predecessor's.
    pub registered: bool,
    pub slot: u64,
    pub leaf: Leaf,
    /// The records and terms the leaf's `data` and `deal` hash (the
    /// predecessor's when not registered).
    pub records: Vec<u8>,
    pub terms: Vec<u8>,
    /// As `OP_MERKLE_ROOT` reads it: per level, sibling ‖ side.
    pub proof: Vec<u8>,
    /// The current thread: its coin, root and salt.
    pub thread: Thread,
    /// The height the index stands at.
    pub height: u64,
}

impl NameProof {
    /// The root the proof reaches: `thread.root` for a good one.
    pub fn computed_root(&self) -> [u8; 32] {
        let mut acc = self.leaf.hash();
        for l in 0..self.proof.len() / 33 {
            let sib: [u8; 32] = self.proof[l * 33..l * 33 + 32].try_into().unwrap();
            acc = if self.proof[l * 33 + 32] & 1 == 1 { crate::core::types::hash_concat(&sib, &acc) } else { crate::core::types::hash_concat(&acc, &sib) };
        }
        acc
    }
}

/// A proof as `GET /name/:name/proof` serves it (midwimble's resolver reads
/// exactly this).
pub fn proof_json(name: &str, p: &NameProof) -> serde_json::Value {
    serde_json::json!({
        "name": format!("{name}{}", names::SUFFIX),
        "registered": p.registered,
        "slot": p.slot,
        "leaf": {
            "key": hex::encode(p.leaf.key),
            "next": hex::encode(p.leaf.next),
            "expires": p.leaf.expires,
            "tier": p.leaf.tier,
            "owner": hex::encode(p.leaf.owner),
            "data": hex::encode(p.leaf.data),
            "deal": hex::encode(p.leaf.deal),
        },
        "records": hex::encode(&p.records),
        "terms": hex::encode(&p.terms),
        "proof": hex::encode(&p.proof),
        "root": hex::encode(p.thread.root),
        "thread": { "coin": hex::encode(p.thread.id), "salt": hex::encode(p.thread.salt) },
        "registry": hex::encode(names::address()),
        "height": p.height,
    })
}

/// The proof that `name` holds what the index says, or that it was never
/// registered, against the registry's current root.
pub fn prove(db: &Database, name: &str) -> Result<NameProof> {
    let name = names::normalize(name)?;
    let m = ready(db)?;
    let Some(thread) = m.thread else { bail!("the name registry has not started on this chain yet") };
    let txn = db.begin_read()?;
    let nodes = txn.open_table(NODES)?;
    let tree = Tree(&nodes);
    let found = find(&txn, &name)?;
    let (slot, leaf, registered) = match found {
        Some((slot, leaf, true)) => (slot, leaf, true),
        Some((_, _, false)) => bail!("another name shares {name}'s key"),
        None => {
            let k = names::ord(&names::key(name.as_bytes()));
            let keys = txn.open_table(KEYS)?;
            let (_, v) = keys.range(..k)?.next_back().ok_or_else(|| anyhow!("name index: no sentinel"))??;
            let slot = u64::from_le_bytes(v.value()[..8].try_into()?);
            let raw = txn.open_table(LEAVES)?.get(slot)?.ok_or_else(|| anyhow!("name index: slot missing"))?.value().to_owned();
            (slot, Leaf::from_bytes(&raw), false)
        }
    };
    let raw = txn.open_table(RECORDS)?.get(slot)?.map(|g| g.value().to_vec()).unwrap_or_else(|| vec![0; 16]);
    let terms = txn.open_table(DEALS)?.get(slot)?.map_or_else(|| Deal::None.terms(), |g| g.value().to_vec());
    Ok(NameProof { registered, slot, leaf, records: raw[16..].to_vec(), terms, proof: names::proof(&tree, slot)?, thread, height: m.next })
}

/// A sale or rental that landed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sold {
    pub name: String,
    pub height: u64,
    /// The offer it was taken at.
    pub deal: Deal,
}

/// The latest sales and rentals, newest first.
pub fn recent_deals(db: &Database, limit: usize) -> Result<Vec<Sold>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let log = txn.open_table(LOG)?;
    let mut out = Vec::new();
    for entry in log.range(0..m.log_len)?.rev() {
        let (_, v) = entry?;
        let Ok(e) = bincode::deserialize::<Entry>(v.value()) else { continue };
        if let Logged::Spend(Op::Change { by: By::Buy | By::Rent, name, oterms, .. }, _) = e.what {
            out.push(Sold { name: String::from_utf8_lossy(&name).into_owned(), height: e.height, deal: Deal::parse(&oterms) });
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok(out)
}

/// A spend of the current thread: `input` opened by `items` (sign the
/// transaction into `items[1]` with `signer`, if any), `output` as its output
/// 0, and what else the transaction must carry: `burn` units burned, and
/// `pay` units to an address.
pub struct Plan {
    pub input: InputReveal,
    pub items: Vec<Vec<u8>>,
    pub signer: Option<[u8; 32]>,
    pub output: OutputData,
    pub thread: Thread,
    pub burn: u64,
    pub pay: Option<(u64, [u8; 32])>,
}

impl Plan {
    pub fn witness(mut self, sig: Option<Vec<u8>>) -> Witness {
        if let Some(sig) = sig {
            self.items[1] = sig;
        }
        Witness::ScriptInputs(self.items)
    }
}

/// How a spend comes by a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum How {
    /// Register it, reclaim it if lapsed, or change it signed by `signer` if that holds it.
    Hold(Option<[u8; 32]>),
    /// Take it as its terms offer.
    Buy,
    Rent,
}

/// Plan a spend that leaves `name` as `set` says. `set.expires` must be what
/// its `years` buy (see [`Record::quote`]).
pub fn plan(db: &Database, name: &str, how: How, set: &names::Set) -> Result<Plan> {
    let name = names::normalize(name)?;
    let shown = format!("{name}{}", names::SUFFIX);
    let m = ready(db)?;
    let Some(thread) = m.thread else { bail!("the name registry has not started on this chain yet") };
    let txn = db.begin_read()?;
    let nodes = txn.open_table(NODES)?;
    let tree = Tree(&nodes);
    let found = find(&txn, &name)?;
    if let Some((_, _, false)) = found {
        bail!("{shown} cannot be registered: another name has its key");
    }
    let mut pay = None;
    let (spend, signer, tier) = match (found, how) {
        (None, How::Hold(_)) => {
            let k = names::ord(&names::key(name.as_bytes()));
            let keys = txn.open_table(KEYS)?;
            let (_, v) = keys.range(..k)?.next_back().ok_or_else(|| anyhow!("name index: no sentinel"))??;
            let pred_pos = u64::from_le_bytes(v.value()[..8].try_into()?);
            let raw = txn.open_table(LEAVES)?.get(pred_pos)?.ok_or_else(|| anyhow!("name index: slot missing"))?.value().to_owned();
            let spend = names::register(&tree, &name, set, pred_pos, Leaf::from_bytes(&raw), m.free)?;
            (spend, None, names::tier_for_len(name.len()))
        }
        (None, _) => bail!("{shown} is not registered"),
        (Some((pos, leaf, _)), how) => {
            let r = record(&txn, &m, name.clone(), pos, leaf)?;
            let by = match how {
                How::Hold(_) if r.lapsed() => By::Lapse,
                How::Hold(s) if s == Some(leaf.owner) => By::Owner,
                How::Hold(_) => bail!("{shown} is taken"),
                How::Buy => By::Buy,
                How::Rent => By::Rent,
            };
            match (by, r.deal) {
                (By::Buy, Deal::Sale { price, payee }) | (By::Rent, Deal::Rent { price, payee, .. }) => pay = Some((price, payee)),
                (By::Buy | By::Rent, _) => bail!("{shown} is not offered that way"),
                _ => {}
            }
            let spend = names::change(&tree, by, &name, pos, leaf, &r.terms, set)?;
            (spend, (by == By::Owner).then_some(leaf.owner), leaf.tier)
        }
    };
    Ok(Plan {
        input: thread.input(),
        items: spend.items,
        signer,
        output: names::next_tip(spend.root, &thread.id),
        thread,
        burn: names::cost(tier, set.years),
        pay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::names::{Rec, Set, YEAR};
    use crate::core::types::{decompose_value, Extension, Predicate};
    use crate::core::wots;
    use crate::storage::BatchStore;

    const H: u64 = names::START_HEIGHT;

    fn block(variant: u8, txs: Vec<Transaction>) -> Batch {
        Batch {
            prev_midstate: [0; 32],
            transactions: txs,
            extension: Extension { nonce: variant as u64, final_hash: [variant; 32] },
            coinbase: vec![],
            timestamp: 0,
            target: [0xff; 32],
            state_root: [0; 32],
            prev_header_hash: [0; 32],
        }
    }

    fn reveal(inputs: Vec<InputReveal>, witnesses: Vec<Witness>, outputs: Vec<OutputData>) -> Transaction {
        Transaction::Reveal { inputs, witnesses, outputs, salt: [0; 32] }
    }

    /// Makes `out` from a plain coin.
    fn make(out: OutputData) -> Transaction {
        let coin = InputReveal { predicate: Predicate::p2pk(&[1; 32]), value: 2, salt: [1; 32], commitment: None };
        reveal(vec![coin], vec![Witness::ScriptInputs(vec![])], vec![out])
    }

    fn recs(text: &str) -> Vec<u8> {
        names::encode(&[Rec::MdsKey { key: [3; 32], height: 10 }, Rec::Text(text.into())]).unwrap()
    }

    /// A spend planned from the index — signed by `seed`'s one-time key if it
    /// needs it, carrying its burn and payment — checked by the covenant.
    fn spend_tx(db: &Database, name: &str, how: How, seed: Option<[u8; 32]>, set: Set) -> Transaction {
        let p = plan(db, name, how, &set).unwrap();
        let commitment = [4u8; 32];
        let sig = p.signer.map(|_| wots::sig_to_bytes(&wots::sign(&seed.unwrap(), &commitment)));
        let mut outputs = vec![p.output.clone()];
        outputs.extend(names::burn_outputs(name, p.burn));
        if let Some((price, payee)) = p.pay {
            outputs.extend(decompose_value(price).into_iter().map(|value| OutputData::Standard { address: payee, value, salt: [1; 32] }));
        }
        let input = p.input.clone();
        let Witness::ScriptInputs(items) = p.witness(sig);
        let ctx = crate::core::script::ExecContext {
            commitment: &commitment,
            height: meta(db).unwrap().next,
            outputs: &outputs,
            input_value: 0,
            input_state: input.commitment,
            this_address: input.predicate.address(),
            sum_input_value: 0,
        };
        crate::core::script::execute_script(names::covenant(), &items, &ctx).expect("the covenant accepts the plan");
        reveal(vec![input], vec![Witness::ScriptInputs(items)], outputs)
    }

    /// A year of `name` for `owner`, from the next block.
    fn hold(db: &Database, owner: [u8; 32], text: &str) -> Set {
        let next = meta(db).unwrap().next;
        Set { owner, records: recs(text), terms: Deal::None.terms(), years: 1, expires: next + YEAR }
    }

    fn register_tx(db: &Database, name: &str, owner: u8) -> Transaction {
        spend_tx(db, name, How::Hold(None), None, hold(db, [owner; 32], ""))
    }

    /// A store whose index is on (or not), with `blocks` saved from height H.
    fn chain(on: bool, blocks: &[Vec<Transaction>]) -> (tempfile::TempDir, Arc<Database>, BatchStore) {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("db")).unwrap());
        let bs = BatchStore::new(db.clone(), dir.path().join("legacy")).unwrap();
        if on {
            bs.names().enable(&db, H).unwrap();
        }
        for (i, txs) in blocks.iter().enumerate() {
            bs.save(H + i as u64, &block(i as u8 + 1, txs.clone())).unwrap();
        }
        (dir, db, bs)
    }

    fn owner(db: &Database, name: &str) -> Option<u8> {
        lookup(db, name).unwrap().filter(|r| !r.lapsed()).map(|r| r.owner[0])
    }

    #[test]
    fn follows_the_registry_through_reorgs() {
        let (_d, db, bs) = chain(true, &[]);
        assert!(lookup(&db, "nick").unwrap().is_none());
        assert!(plan(&db, "nick", How::Hold(None), &hold(&db, [1; 32], "")).is_err(), "no registry yet");
        bs.save(H, &block(1, vec![make(names::genesis())])).unwrap();
        let unclaimed = lookup(&db, "bitcoin").unwrap().unwrap();
        assert!(unclaimed.lapsed() && unclaimed.tier == 0, "premium names start unclaimed");
        let nick = register_tx(&db, "nick", 7);
        bs.save(H + 1, &block(2, vec![nick.clone()])).unwrap();
        let r = lookup(&db, "NICK.mds").unwrap().unwrap();
        assert_eq!((r.owner, r.since, r.expires, r.tier, r.records.len()), ([7; 32], H + 1, H + 1 + YEAR, 2, 2));
        assert!(plan(&db, "nick", How::Hold(None), &hold(&db, [8; 32], "")).is_err(), "taken");
        let base = vec![vec![make(names::genesis())], vec![nick]];

        // A copy of the thread, same root, spent with a valid witness: ignored.
        let t = meta(&db).unwrap().thread.unwrap();
        let copy = OutputData::Confidential { address: names::address(), commitment: t.root, salt: [9; 32] };
        let p = plan(&db, "eve", How::Hold(None), &hold(&db, [6; 32], "")).unwrap();
        let out = p.output.clone();
        let eve = reveal(vec![names::tip_input(t.root, [9; 32])], vec![p.witness(None)], vec![out]);
        bs.save(H + 2, &block(3, vec![make(copy), eve])).unwrap();
        assert_eq!((owner(&db, "eve"), meta(&db).unwrap().thread), (None, Some(t)));

        // Two in one block, the second spending the thread the first made.
        let alice = register_tx(&db, "alice", 3);
        let (_d2, db2, _) = chain(true, &[base.clone(), vec![vec![], vec![alice.clone()]]].concat());
        let bob = register_tx(&db2, "bob", 4);
        bs.save(H + 3, &block(4, vec![alice, bob])).unwrap();
        assert_eq!((owner(&db, "alice"), owner(&db, "bob")), (Some(3), Some(4)));
        assert_eq!(meta(&db).unwrap().names, 3);

        // Reorg: block H+2 replaced, H+3 gone with it.
        let (_d3, db3, _) = chain(true, &base);
        let carol = register_tx(&db3, "carol", 5);
        bs.save(H + 2, &block(5, vec![carol])).unwrap();
        assert_eq!((owner(&db, "alice"), owner(&db, "bob"), owner(&db, "carol")), (None, None, Some(5)));

        // Truncated back past the genesis coin: nothing left.
        bs.truncate(H).unwrap();
        let m = meta(&db).unwrap();
        assert_eq!((m.thread, m.names, m.log_len, m.next, m.free), (None, 0, 0, H, 0));
        let txn = db.begin_read().unwrap();
        assert_eq!(txn.open_table(NODES).unwrap().iter().unwrap().count(), 0);
        assert_eq!(txn.open_table(KEYS).unwrap().iter().unwrap().count(), 0);
        assert_eq!(txn.open_table(RECORDS).unwrap().iter().unwrap().count(), 0);
    }

    #[test]
    fn a_registration_ends_only_when_another_key_takes_the_name_over() {
        let (a, b, none) = ([1u8; 32], [2u8; 32], [0u8; 32]);
        // Claimed from nobody: a premium name's first holder.
        assert!(new_registration(By::Lapse, &none, &a));
        assert!(new_registration(By::Buy, &none, &a));
        // Reclaimed after it lapsed, by someone else: starts again.
        assert!(new_registration(By::Lapse, &a, &b));
        // The holder reclaiming its own, renewing late, rotating its key,
        // selling or renting: the registration goes on.
        assert!(!new_registration(By::Lapse, &a, &a));
        assert!(!new_registration(By::Owner, &a, &a));
        assert!(!new_registration(By::Owner, &a, &b));
        assert!(!new_registration(By::Buy, &a, &b));
        assert!(!new_registration(By::Rent, &a, &a));
    }

    #[test]
    fn a_names_backing_counts_from_its_registration() {
        use crate::core::burns::{Backing, Purpose};
        use crate::storage::burn_index::{self, backing, counted, counts_from, held, top_counted};
        let burn = |value: u64| {
            let coin = InputReveal { predicate: Predicate::p2pk(&[1; 32]), value, salt: [1; 32], commitment: None };
            reveal(vec![coin], vec![Witness::ScriptInputs(vec![])], Backing::name("bitcoin", "").unwrap().outputs(value))
        };
        let s = crate::core::burns::name_subject("bitcoin").unwrap();
        let (_d, db, bs) = chain(true, &[vec![make(names::genesis())]]);
        bs.burns().enable(&db, H + 1).unwrap();
        bs.burns().catch_up(&db).unwrap();
        // Burned for while unclaimed: nothing counts, though the burn stands.
        bs.save(H + 1, &block(2, vec![burn(1 << 20)])).unwrap();
        assert_eq!(counts_from(&db, Purpose::NAME, &s).unwrap(), None, "no one holds it");
        assert!(counted(&db, Purpose::NAME, &s).unwrap().is_none());
        assert_eq!(backing(&db, Purpose::NAME, &s, None).unwrap().unwrap().total, 1 << 20);
        assert!(top_counted(&db, Purpose::NAME, 5).unwrap().is_empty());
        assert!(held(&db).unwrap().iter().all(|(p, _, _)| *p != Purpose::NAME));
        // Claimed, and backed in the same block: that burn counts, the
        // earlier one never will.
        let claim = spend_tx(&db, "bitcoin", How::Hold(None), None, hold(&db, [7; 32], "one"));
        bs.save(H + 2, &block(3, vec![claim, burn(1 << 10)])).unwrap();
        assert_eq!(counts_from(&db, Purpose::NAME, &s).unwrap(), Some(H + 2));
        let c = counted(&db, Purpose::NAME, &s).unwrap().unwrap();
        assert_eq!((c.total, c.count, c.first), (1 << 10, 1, H + 2));
        assert_eq!(top_counted(&db, Purpose::NAME, 5).unwrap(), vec![c.clone()]);
        let names_held = held(&db).unwrap().into_iter().find(|(p, _, _)| *p == Purpose::NAME).unwrap();
        assert!((names_held.1 - c.backing).abs() < 1e-12 && names_held.2 == 1);
        // A burn for another subject entirely is untouched by the rule.
        assert_eq!(counts_from(&db, Purpose::POOL, &[9; 32]).unwrap(), Some(0));
        // The claim reorged out: no one holds it again.
        bs.save(H + 2, &block(4, vec![burn(1 << 10)])).unwrap();
        assert_eq!(counts_from(&db, Purpose::NAME, &s).unwrap(), None);
        assert!(counted(&db, Purpose::NAME, &s).unwrap().is_none());
        assert_eq!(burn_index::backing(&db, Purpose::NAME, &s, None).unwrap().unwrap().count, 2);
    }

    #[test]
    fn claims_sales_and_rentals_and_a_reorg_restores_them() {
        let seed = [5u8; 32];
        let me = wots::keygen(&seed);
        let (_d, db, bs) = chain(true, &[vec![make(names::genesis())]]);
        // A premium name, claimed at its tier's price.
        let claim = spend_tx(&db, "bitcoin", How::Hold(None), None, hold(&db, me, "one"));
        bs.save(H + 1, &block(2, vec![claim])).unwrap();
        let r = lookup(&db, "bitcoin").unwrap().unwrap();
        assert_eq!((r.owner, r.since, r.lapsed()), (me, H + 1, false));
        assert_eq!(r.registered, H + 1, "claiming a premium name registers it");
        assert!(plan(&db, "bitcoin", How::Hold(Some([9; 32])), &hold(&db, [9; 32], "")).is_err(), "not its owner");
        // Offered for sale, then bought.
        let payee = [6u8; 32];
        let offer = Set { terms: Deal::Sale { price: 4096, payee }.terms(), years: 0, expires: r.expires, ..hold(&db, me, "one") };
        bs.save(H + 2, &block(3, vec![spend_tx(&db, "bitcoin", How::Hold(Some(me)), Some(seed), offer)])).unwrap();
        assert_eq!(listings(&db).unwrap().iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["bitcoin"]);
        let buy = Set { years: 0, expires: r.expires, ..hold(&db, [8; 32], "two") };
        bs.save(H + 3, &block(4, vec![spend_tx(&db, "bitcoin", How::Buy, None, buy)])).unwrap();
        let r = lookup(&db, "bitcoin").unwrap().unwrap();
        assert_eq!((r.owner, r.since, r.deal, r.records[1].clone()), ([8; 32], H + 3, Deal::None, Rec::Text("two".into())));
        assert_eq!(r.registered, H + 1, "a sale passes the registration on");
        assert!(listings(&db).unwrap().is_empty());
        assert_eq!(names_of(&db, &[8; 32]).unwrap().iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["bitcoin"]);
        assert!(names_of(&db, &me).unwrap().is_empty(), "the seller holds it no more");
        assert_eq!(recent_deals(&db, 5).unwrap(), vec![Sold { name: "bitcoin".into(), height: H + 3, deal: Deal::Sale { price: 4096, payee } }]);
        // Reorged out: back to its seller, still offered.
        bs.save(H + 3, &block(9, vec![])).unwrap();
        let r = lookup(&db, "bitcoin").unwrap().unwrap();
        assert_eq!((r.owner, r.deal), (me, Deal::Sale { price: 4096, payee }));
        assert_eq!(names_of(&db, &me).unwrap().len(), 1);
        assert!(recent_deals(&db, 5).unwrap().is_empty(), "the sale was reorged out");
        // Offered for rent instead, and rented.
        let rent = Deal::Rent { price: 100, payee, span: 500 };
        let offer = Set { terms: rent.terms(), years: 0, expires: r.expires, ..hold(&db, me, "one") };
        bs.save(H + 3, &block(5, vec![spend_tx(&db, "bitcoin", How::Hold(Some(me)), Some(seed), offer)])).unwrap();
        let until = meta(&db).unwrap().next + 500;
        let take = Set { owner: me, terms: Deal::Rented { price: 100, payee, until }.terms(), years: 0, expires: r.expires, records: recs("renter") };
        bs.save(H + 4, &block(6, vec![spend_tx(&db, "bitcoin", How::Rent, None, take)])).unwrap();
        let r = lookup(&db, "bitcoin").unwrap().unwrap();
        assert_eq!((r.owner, r.deal, r.records[1].clone()), (me, Deal::Rented { price: 100, payee, until }, Rec::Text("renter".into())));
    }

    #[test]
    fn a_proof_reaches_the_threads_root_for_names_in_and_out() {
        let (_d, db, bs) = chain(true, &[vec![make(names::genesis())]]);
        bs.save(H + 1, &block(2, vec![register_tx(&db, "nick", 7)])).unwrap();
        let p = prove(&db, "nick").unwrap();
        assert!(p.registered);
        assert_eq!(p.computed_root(), p.thread.root);
        assert_eq!(p.leaf.key, names::key(b"nick"));
        assert_eq!(crate::core::types::hash(&p.records), p.leaf.data, "the records are the leaf's");
        assert_eq!(crate::core::types::hash(&p.terms), p.leaf.deal);
        assert_eq!(names::parse(&p.records).unwrap(), lookup(&db, "nick").unwrap().unwrap().records);
        // Not registered: the predecessor's leaf skips over its key.
        let q = prove(&db, "nobody-here").unwrap();
        let k = names::ord(&names::key(b"nobody-here"));
        assert!(!q.registered && names::ord(&q.leaf.key) < k && k < names::ord(&q.leaf.next));
        assert_eq!(q.computed_root(), q.thread.root);
        // A forged record set does not hash to the leaf.
        let forged = names::encode(&[names::Rec::MdsAddress([9; 32])]).unwrap();
        assert_ne!(crate::core::types::hash(&forged), p.leaf.data);
        // The coin the proof names is the registry thread's, at its address.
        assert_eq!(p.thread.id, OutputData::Confidential { address: names::address(), commitment: p.thread.root, salt: p.thread.salt }.coin_id().unwrap());
        if std::env::var("PRINT_NAME_PROOF").is_ok() {
            let set = Set { records: names::encode(&[Rec::Mdw("mw1fixture".into())]).unwrap(), ..hold(&db, [8; 32], "") };
            bs.save(H + 2, &block(3, vec![spend_tx(&db, "mdwname", How::Hold(None), None, set)])).unwrap();
            println!("{}", proof_json("mdwname", &prove(&db, "mdwname").unwrap()));
        }
    }

    #[test]
    fn backfill_builds_what_the_hooks_would_have() {
        let (_d, db, bs) = chain(true, &[vec![make(names::genesis())]]);
        let nick = register_tx(&db, "nick", 7);
        bs.save(H + 1, &block(2, vec![nick.clone()])).unwrap();
        let blocks = vec![vec![make(names::genesis())], vec![nick]];

        let (_d2, db2, bs2) = chain(false, &blocks);
        bs2.names().enable(&db2, H + 2).unwrap();
        bs2.names().catch_up(&db2).unwrap();
        let (want, got) = (meta(&db).unwrap(), meta(&db2).unwrap());
        assert!(got.ready());
        assert_eq!((got.thread, got.names, got.free, got.next), (want.thread, want.names, want.free, want.next));
        assert_eq!(lookup(&db2, "nick").unwrap(), lookup(&db, "nick").unwrap());
    }
}
