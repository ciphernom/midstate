//! Every burn on the chain, and MSB-1 backing by subject (`core::burns`).
//!
//! * [`BLOCKS`]: `height → time ‖ burns`, every burn of value a block made
//!   (and every MSB-1 burn, of any value): what undoes the block, too.
//! * [`DAYS`]: `day → totals` by kind (names, backing by purpose, dex,
//!   other), for the chain's burn against its issuance.
//! * [`SUBJECTS`]: `purpose ‖ subject ‖ height ‖ row → burn`, each subject's
//!   history.
//! * [`SCORES`]: `purpose ‖ subject → score`: its anchors' sum (see
//!   [`burns::anchor`]), total, count, first and last heights, latest label.
//! * [`RANKS`]: `purpose ‖ anchors ‖ subject`, subjects in order of backing.
//!   Decay is the same for every burn, so the order holds between burns.
//! * [`META`]: the watermark and totals.
//!
//! Kept the way the name index is: the hooks run inside the write that stores
//! or deletes a block, a reorg rewinds before it applies, and rewinding reads
//! only [`BLOCKS`], so it unwinds exactly on a pruned node too. A worker
//! backfills from the lowest block the node holds; a node that pruned its
//! early blocks says from where its totals run (`Meta::from`).
//!
//! Always on. A block with no burn costs a metadata read and write.

use anyhow::{anyhow, Result};
use redb::{Database, ReadableTable, Table, TableDefinition, WriteTransaction};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::core::burns::{self, Backing, Kind, Purpose};
use crate::core::types::{Batch, BatchHeader, Transaction, BLOCKS_PER_YEAR};

pub const BLOCKS: TableDefinition<u64, &[u8]> = TableDefinition::new("burns_blocks");
pub const DAYS: TableDefinition<u64, &[u8]> = TableDefinition::new("burns_days");
pub const SUBJECTS: TableDefinition<&[u8; 45], &[u8]> = TableDefinition::new("burns_subjects");
pub const SCORES: TableDefinition<&[u8; 33], &[u8]> = TableDefinition::new("burns_scores");
pub const RANKS: TableDefinition<&[u8; 41], ()> = TableDefinition::new("burns_ranks");
pub const META: TableDefinition<&str, &[u8]> = TableDefinition::new("burns_meta");

/// Bump to rebuild every node's index on upgrade.
const VERSION: u64 = 1;
/// Blocks a day: totals are kept by day.
pub const DAY: u64 = 1_440;
const STEP_BUDGET: Duration = Duration::from_millis(250);
const STEP_PAUSE: Duration = Duration::from_millis(20);
const IDLE_WAIT: Duration = Duration::from_secs(30);

/// Burns by kind: value and number.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    pub names: u64,
    /// Backing and notarizations, by purpose (index = purpose, 0 for
    /// purposes past the known ones).
    pub backing: [u64; 8],
    pub dex: u64,
    pub other: u64,
    /// Burns of value, any kind.
    pub count: u64,
    /// MSB-1 burns, any value.
    pub backing_count: u64,
}

impl Totals {
    pub fn burned(&self) -> u64 {
        self.names.saturating_add(self.backing_total()).saturating_add(self.dex).saturating_add(self.other)
    }

    pub fn backing_total(&self) -> u64 {
        self.backing.iter().fold(0u64, |a, v| a.saturating_add(*v))
    }

    fn slot(p: Purpose) -> usize {
        if (p.0 as usize) < 8 { p.0 as usize } else { 0 }
    }

    fn add(&mut self, kind: Kind, purpose: Option<Purpose>, value: u64, sign: i8) {
        let f = |x: &mut u64| *x = if sign > 0 { x.saturating_add(value) } else { x.saturating_sub(value) };
        match kind {
            Kind::Names => f(&mut self.names),
            Kind::Backing => f(&mut self.backing[Self::slot(purpose.unwrap_or(Purpose(0)))]),
            Kind::Dex => f(&mut self.dex),
            Kind::Other => f(&mut self.other),
        }
        let one = |x: &mut u64| *x = if sign > 0 { *x + 1 } else { x.saturating_sub(1) };
        if value > 0 {
            one(&mut self.count);
        }
        if kind == Kind::Backing {
            one(&mut self.backing_count);
        }
    }

    pub fn merge(&mut self, o: &Totals) {
        self.names = self.names.saturating_add(o.names);
        for (a, b) in self.backing.iter_mut().zip(o.backing) {
            *a = a.saturating_add(b);
        }
        self.dex = self.dex.saturating_add(o.dex);
        self.other = self.other.saturating_add(o.other);
        self.count += o.count;
        self.backing_count += o.backing_count;
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
    /// Rebuild from the lowest block held.
    pub stale: bool,
    /// `h + 1` if block `h` is not stored here.
    pub missing: u64,
    /// The first height the totals cover: 0 unless the node pruned its early
    /// blocks before the index was built.
    pub from: u64,
    pub totals: Totals,
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
        self.version == VERSION && !self.stale && self.missing == 0 && self.next >= self.chain_len
    }
}

/// One burn a block made: its transaction's index, value and payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub tx: u32,
    pub value: u64,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Block {
    time: u64,
    rows: Vec<Row>,
}

/// A subject's entry in [`SUBJECTS`].
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    value: u64,
    tx: u32,
    time: u64,
    label: String,
}

/// What a subject has, as [`SCORES`] keeps it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct Score {
    anchors: f64,
    total: u64,
    count: u64,
    first: u64,
    last: u64,
    label: String,
}

fn subject_key(p: Purpose, s: &[u8; 32]) -> [u8; 33] {
    let mut k = [0u8; 33];
    k[0] = p.0;
    k[1..].copy_from_slice(s);
    k
}

fn entry_key(p: Purpose, s: &[u8; 32], height: u64, row: u32) -> [u8; 45] {
    let mut k = [0u8; 45];
    k[..33].copy_from_slice(&subject_key(p, s));
    k[33..41].copy_from_slice(&height.to_be_bytes());
    k[41..].copy_from_slice(&row.to_be_bytes());
    k
}

fn rank_key(p: Purpose, anchors: f64, s: &[u8; 32]) -> [u8; 41] {
    let mut k = [0u8; 41];
    k[0] = p.0;
    // Positive finite floats order as their bits do.
    k[1..9].copy_from_slice(&anchors.max(0.0).to_bits().to_be_bytes());
    k[9..].copy_from_slice(s);
    k
}

/// The burns `batch` makes worth keeping: any of value, and MSB-1 ones of
/// none (a notarization may burn nothing but its fee).
pub fn rows_of(batch: &Batch) -> Vec<Row> {
    let mut rows = Vec::new();
    for (i, tx) in batch.transactions.iter().enumerate() {
        let outputs = match tx {
            Transaction::Reveal { outputs, .. } | Transaction::Consolidate { outputs, .. } => outputs,
            Transaction::Commit { .. } => continue,
        };
        for (payload, value) in burns::burns_of(outputs) {
            if value > 0 || Kind::of(&payload) == Kind::Backing {
                rows.push(Row { tx: i as u32, value, payload });
            }
        }
    }
    rows
}

// ── Applying and undoing ────────────────────────────────────────────────────

struct Writer<'t> {
    blocks: Table<'t, u64, &'static [u8]>,
    days: Table<'t, u64, &'static [u8]>,
    subjects: Table<'t, &'static [u8; 45], &'static [u8]>,
    scores: Table<'t, &'static [u8; 33], &'static [u8]>,
    ranks: Table<'t, &'static [u8; 41], ()>,
}

impl<'t> Writer<'t> {
    fn open(txn: &'t WriteTransaction) -> Result<Self> {
        Ok(Self {
            blocks: txn.open_table(BLOCKS)?,
            days: txn.open_table(DAYS)?,
            subjects: txn.open_table(SUBJECTS)?,
            scores: txn.open_table(SCORES)?,
            ranks: txn.open_table(RANKS)?,
        })
    }

    fn day(&self, day: u64) -> Result<Totals> {
        Ok(self.days.get(day)?.and_then(|g| bincode::deserialize(g.value()).ok()).unwrap_or_default())
    }

    fn set_day(&mut self, day: u64, t: &Totals) -> Result<()> {
        if *t == Totals::default() {
            self.days.remove(day)?;
        } else {
            self.days.insert(day, bincode::serialize(t)?.as_slice())?;
        }
        Ok(())
    }

    fn score(&self, k: &[u8; 33]) -> Result<Option<Score>> {
        Ok(self.scores.get(k)?.and_then(|g| bincode::deserialize(g.value()).ok()))
    }

    fn set_score(&mut self, p: Purpose, s: &[u8; 32], old: Option<&Score>, new: Option<&Score>) -> Result<()> {
        if let Some(o) = old {
            self.ranks.remove(&rank_key(p, o.anchors, s))?;
        }
        let k = subject_key(p, s);
        match new {
            Some(n) => {
                self.scores.insert(&k, bincode::serialize(n)?.as_slice())?;
                self.ranks.insert(&rank_key(p, n.anchors, s), ())?;
            }
            None => {
                self.scores.remove(&k)?;
            }
        }
        Ok(())
    }

    fn apply(&mut self, m: &mut Meta, height: u64, batch: &Batch) -> Result<()> {
        let rows = rows_of(batch);
        if rows.is_empty() {
            return Ok(());
        }
        let day = height / DAY;
        let mut totals = self.day(day)?;
        for (j, row) in rows.iter().enumerate() {
            let kind = Kind::of(&row.payload);
            let b = Backing::parse(&row.payload);
            totals.add(kind, b.as_ref().map(|b| b.purpose), row.value, 1);
            m.totals.add(kind, b.as_ref().map(|b| b.purpose), row.value, 1);
            let Some(b) = b else { continue };
            let e = Entry { value: row.value, tx: row.tx, time: batch.timestamp, label: b.label.clone() };
            self.subjects.insert(&entry_key(b.purpose, &b.subject, height, j as u32), bincode::serialize(&e)?.as_slice())?;
            let old = self.score(&subject_key(b.purpose, &b.subject))?;
            let mut s = old.clone().unwrap_or(Score { first: height, ..Score::default() });
            s.anchors += burns::anchor(row.value, height);
            s.total = s.total.saturating_add(row.value);
            s.count += 1;
            s.first = s.first.min(height);
            s.last = height;
            if !b.label.is_empty() {
                s.label = b.label.clone();
            }
            self.set_score(b.purpose, &b.subject, old.as_ref(), Some(&s))?;
        }
        self.set_day(day, &totals)?;
        self.blocks.insert(height, bincode::serialize(&Block { time: batch.timestamp, rows })?.as_slice())?;
        Ok(())
    }

    /// Undo block `height`; the subjects it touched, for [`Self::rescore`].
    fn undo(&mut self, m: &mut Meta, height: u64, touched: &mut BTreeSet<[u8; 33]>) -> Result<()> {
        let Some(raw) = self.blocks.get(height)?.map(|g| g.value().to_vec()) else { return Ok(()) };
        let block: Block = bincode::deserialize(&raw)?;
        let day = height / DAY;
        let mut totals = self.day(day)?;
        for (j, row) in block.rows.iter().enumerate() {
            let kind = Kind::of(&row.payload);
            let b = Backing::parse(&row.payload);
            totals.add(kind, b.as_ref().map(|b| b.purpose), row.value, -1);
            m.totals.add(kind, b.as_ref().map(|b| b.purpose), row.value, -1);
            if let Some(b) = b {
                self.subjects.remove(&entry_key(b.purpose, &b.subject, height, j as u32))?;
                touched.insert(subject_key(b.purpose, &b.subject));
            }
        }
        self.set_day(day, &totals)?;
        self.blocks.remove(height)?;
        Ok(())
    }

    /// A subject's score, summed again from the burns it has left: exact,
    /// whatever a float's running sum gathered.
    fn rescore(&mut self, k: &[u8; 33]) -> Result<()> {
        let p = Purpose(k[0]);
        let s: [u8; 32] = k[1..].try_into().expect("32 bytes");
        let (lo, hi) = (entry_key(p, &s, 0, 0), entry_key(p, &s, u64::MAX, u32::MAX));
        let mut n: Option<Score> = None;
        for item in self.subjects.range::<&[u8; 45]>(&lo..=&hi)? {
            let (key, v) = item?;
            let height = u64::from_be_bytes(key.value()[33..41].try_into()?);
            let e: Entry = bincode::deserialize(v.value())?;
            let x = n.get_or_insert(Score { first: height, ..Score::default() });
            x.anchors += burns::anchor(e.value, height);
            x.total = x.total.saturating_add(e.value);
            x.count += 1;
            x.last = height;
            if !e.label.is_empty() {
                x.label = e.label;
            }
        }
        let old = self.score(k)?;
        self.set_score(p, &s, old.as_ref(), n.as_ref())
    }
}

/// Undo every block at or above height `to`, so that `next <= to`.
fn rewind(txn: &WriteTransaction, m: &mut Meta, to: u64) -> Result<()> {
    if m.next <= to {
        return Ok(());
    }
    let mut w = Writer::open(txn)?;
    let heights: Vec<u64> = w.blocks.range(to..)?.map(|r| r.map(|(k, _)| k.value())).collect::<Result<_, _>>()?;
    let mut touched = BTreeSet::new();
    for h in heights.into_iter().rev() {
        w.undo(m, h, &mut touched)?;
    }
    for k in &touched {
        w.rescore(k)?;
    }
    m.next = to.max(m.from);
    if m.from > to {
        // Rewound past where this index began: start again.
        m.stale = true;
    }
    m.tip_hash = header_hash(txn, m.next.wrapping_sub(1))?.unwrap_or_default();
    Ok(())
}

fn header_hash(txn: &WriteTransaction, height: u64) -> Result<Option<[u8; 32]>> {
    let t = txn.open_table(super::HEADERS_TABLE)?;
    let h = t.get(height)?.and_then(|raw| bincode::deserialize::<BatchHeader>(raw.value()).ok());
    Ok(h.map(|h| h.extension.final_hash))
}

fn lowest_block(txn: &WriteTransaction) -> Result<u64> {
    let t = txn.open_table(super::BATCHES_TABLE)?;
    let lowest = t.first()?.map(|(k, _)| k.value()).unwrap_or(0);
    Ok(lowest)
}

/// Empty the index; it starts again from the lowest block held.
fn wipe(txn: &WriteTransaction, m: &mut Meta) -> Result<()> {
    txn.delete_table(BLOCKS)?;
    txn.delete_table(DAYS)?;
    txn.delete_table(SUBJECTS)?;
    txn.delete_table(SCORES)?;
    txn.delete_table(RANKS)?;
    Writer::open(txn)?;
    let from = lowest_block(txn)?.min(m.chain_len);
    *m = Meta { version: VERSION, next: from, from, chain_len: m.chain_len, ..Meta::default() };
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
pub struct BurnIndex {
    inner: Arc<Inner>,
}

impl BurnIndex {
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
        if !m.stale && height == m.next {
            Writer::open(txn)?.apply(&mut m, height, batch)?;
            m.next = height + 1;
            m.tip_hash = batch.extension.final_hash;
            m.missing = 0;
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
        } else if m.next > m.from && header_hash(&txn, m.next - 1)? != Some(m.tip_hash) {
            tracing::warn!("Burn index: block {} changed while the index was off; rebuilding", m.next - 1);
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
        if let Err(e) = std::thread::Builder::new().name("burn-index".into()).spawn(move || me.worker(db)) {
            tracing::warn!("Burn index: could not start its thread: {e}");
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
                        tracing::warn!("Burn index: block {} is not stored here, so burns cannot be counted past it", m.missing - 1);
                        said_missing = Some(m.missing);
                    }
                    self.wait(IDLE_WAIT);
                }
                Ok(None) => self.wait(IDLE_WAIT),
                Err(e) => {
                    tracing::warn!("Burn index: {e:#}");
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
        if m.next >= m.chain_len {
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
            while m.next < m.chain_len && started.elapsed() < STEP_BUDGET {
                let h = m.next;
                let Some(batch) = batches.get(h)?.and_then(|raw| bincode::deserialize::<Batch>(raw.value()).ok()) else {
                    m.missing = h + 1;
                    break;
                };
                w.apply(&mut m, h, &batch)?;
                m.next = h + 1;
                m.tip_hash = batch.extension.final_hash;
                m.missing = 0;
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

fn not_ready(m: &Meta) -> anyhow::Error {
    if m.missing != 0 {
        anyhow!("the burn index stops at block {}, which this node does not have", m.missing - 1)
    } else {
        anyhow!("the burn index is catching up: block {} of {}", m.next, m.chain_len)
    }
}

/// A subject's backing, as of a height.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Subject {
    pub purpose: Purpose,
    pub subject: [u8; 32],
    /// Blocks of backing at `at`.
    pub backing: f64,
    pub at: u64,
    /// Units burned for it, all told, and in how many burns.
    pub total: u64,
    pub count: u64,
    pub first: u64,
    pub last: u64,
    pub label: String,
}

impl Subject {
    fn of(p: Purpose, s: [u8; 32], sc: Score, at: u64) -> Self {
        Subject { purpose: p, subject: s, backing: burns::backing_at(sc.anchors, at), at, total: sc.total, count: sc.count, first: sc.first, last: sc.last, label: sc.label }
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "purpose": self.purpose.word(),
            "subject": burns::subject_text(self.purpose, &self.subject),
            "subject_hex": hex::encode(self.subject),
            "backing": self.backing,
            "backing_text": burns::blocks_text(self.backing),
            "tier": burns::tier(self.backing),
            "upkeep_per_year": burns::upkeep(self.backing),
            "at": self.at,
            "total": self.total,
            "count": self.count,
            "first": self.first,
            "last": self.last,
            "label": self.label,
        })
    }
}

/// One burn for a subject.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Burn {
    pub height: u64,
    pub tx: u32,
    pub time: u64,
    pub value: u64,
    /// Blocks it weighed when made, and weighs at the height asked.
    pub weight: f64,
    pub weight_now: f64,
    pub label: String,
}

fn ready(db: &Database) -> Result<Meta> {
    let m = meta(db)?;
    if m.version != VERSION || m.stale || (m.next == 0 && m.chain_len > 0) {
        return Err(not_ready(&m));
    }
    Ok(m)
}

/// What `subject` has, for `purpose`, at `at` (the index's height if
/// `None`). `None` if nothing was ever burned for it. Answers while the index
/// catches up, from what it has: callers that must be current check
/// [`meta`].
pub fn backing(db: &Database, purpose: Purpose, subject: &[u8; 32], at: Option<u64>) -> Result<Option<Subject>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let t = match txn.open_table(SCORES) {
        Ok(t) => t,
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let Some(raw) = t.get(&subject_key(purpose, subject))? else { return Ok(None) };
    let sc: Score = bincode::deserialize(raw.value())?;
    Ok(Some(Subject::of(purpose, *subject, sc, at.unwrap_or(m.next))))
}

/// A subject's burns, newest first.
pub fn history(db: &Database, purpose: Purpose, subject: &[u8; 32], limit: usize) -> Result<Vec<Burn>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let t = txn.open_table(SUBJECTS)?;
    let (lo, hi) = (entry_key(purpose, subject, 0, 0), entry_key(purpose, subject, u64::MAX, u32::MAX));
    let mut out = Vec::new();
    for item in t.range::<&[u8; 45]>(&lo..=&hi)?.rev().take(limit) {
        let (k, v) = item?;
        let height = u64::from_be_bytes(k.value()[33..41].try_into()?);
        let e: Entry = bincode::deserialize(v.value())?;
        out.push(Burn {
            height,
            tx: e.tx,
            time: e.time,
            value: e.value,
            weight: burns::weight(e.value, height),
            weight_now: burns::weight_at(e.value, height, m.next),
            label: e.label,
        });
    }
    Ok(out)
}

/// The most-backed subjects for `purpose`, at the index's height.
pub fn top(db: &Database, purpose: Purpose, limit: usize) -> Result<Vec<Subject>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let ranks = txn.open_table(RANKS)?;
    let scores = txn.open_table(SCORES)?;
    let mut lo = [0u8; 41];
    lo[0] = purpose.0;
    let mut hi = [0xffu8; 41];
    hi[0] = purpose.0;
    let mut out = Vec::new();
    for item in ranks.range::<&[u8; 41]>(&lo..=&hi)?.rev().take(limit) {
        let (k, _) = item?;
        let s: [u8; 32] = k.value()[9..].try_into()?;
        let Some(raw) = scores.get(&subject_key(purpose, &s))? else { continue };
        out.push(Subject::of(purpose, s, bincode::deserialize(raw.value())?, m.next));
    }
    Ok(out)
}

/// All the backing anyone holds, by purpose, at the index's height: blocks
/// and subjects.
pub fn held(db: &Database) -> Result<Vec<(Purpose, f64, u64)>> {
    let m = ready(db)?;
    let txn = db.begin_read()?;
    let t = txn.open_table(SCORES)?;
    let mut by: std::collections::BTreeMap<Purpose, (f64, u64)> = Default::default();
    for item in t.iter()? {
        let (k, v) = item?;
        let sc: Score = bincode::deserialize(v.value())?;
        let e = by.entry(Purpose(k.value()[0])).or_default();
        e.0 += burns::backing_at(sc.anchors, m.next);
        e.1 += 1;
    }
    Ok(by.into_iter().map(|(p, (b, n))| (p, b, n)).collect())
}

/// A recent MSB-1 burn.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Recent {
    pub height: u64,
    pub tx: u32,
    pub time: u64,
    pub value: u64,
    pub purpose: Purpose,
    pub subject: [u8; 32],
    pub label: String,
}

/// The newest MSB-1 burns, looking back at most `depth` blocks that burned.
pub fn recent(db: &Database, limit: usize, depth: usize) -> Result<Vec<Recent>> {
    ready(db)?;
    let txn = db.begin_read()?;
    let t = txn.open_table(BLOCKS)?;
    let mut out = Vec::new();
    for item in t.iter()?.rev().take(depth) {
        let (k, v) = item?;
        let b: Block = bincode::deserialize(v.value())?;
        for r in b.rows.iter().rev() {
            if let Some(x) = Backing::parse(&r.payload) {
                out.push(Recent { height: k.value(), tx: r.tx, time: b.time, value: r.value, purpose: x.purpose, subject: x.subject, label: x.label });
            }
        }
        if out.len() >= limit {
            break;
        }
    }
    out.truncate(limit);
    Ok(out)
}

/// Burns by day, for days `from..to`.
pub fn days(db: &Database, from: u64, to: u64) -> Result<Vec<(u64, Totals)>> {
    ready(db)?;
    let txn = db.begin_read()?;
    let t = txn.open_table(DAYS)?;
    let mut out = Vec::new();
    for item in t.range(from..to)? {
        let (k, v) = item?;
        out.push((k.value(), bincode::deserialize(v.value())?));
    }
    Ok(out)
}

/// One year of the chain: what it issued and what was burned.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Year {
    pub year: u64,
    pub from: u64,
    /// Exclusive; the index's height for the year under way.
    pub to: u64,
    pub issued: u64,
    pub burned: Totals,
}

/// Every year up to the index's height, and the 365 days before it.
pub fn years(db: &Database) -> Result<(Vec<Year>, Year)> {
    let m = ready(db)?;
    let tip = m.next;
    let all = days(db, 0, tip / DAY + 1)?;
    let days_per_year = BLOCKS_PER_YEAR / DAY;
    let mut out: Vec<Year> = Vec::new();
    for y in 0..=tip.saturating_sub(1) / BLOCKS_PER_YEAR {
        let (from, to) = (y * BLOCKS_PER_YEAR, ((y + 1) * BLOCKS_PER_YEAR).min(tip));
        let mut burned = Totals::default();
        for (_, t) in all.iter().filter(|(d, _)| d / days_per_year == y) {
            burned.merge(t);
        }
        out.push(Year { year: y, from, to, issued: burns::issued(from.max(m.from), to), burned });
    }
    let (from, to) = (tip.saturating_sub(BLOCKS_PER_YEAR), tip);
    let mut burned = Totals::default();
    for (_, t) in all.iter().filter(|(d, _)| *d >= from / DAY) {
        burned.merge(t);
    }
    let trailing = Year { year: u64::MAX, from, to, issued: burns::issued(from.max(m.from), to), burned };
    Ok((out, trailing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{block_reward, Extension, InputReveal, OutputData, Predicate, Witness};
    use crate::storage::BatchStore;

    fn block(variant: u8, time: u64, txs: Vec<Transaction>) -> Batch {
        Batch {
            prev_midstate: [0; 32],
            transactions: txs,
            extension: Extension { nonce: variant as u64, final_hash: [variant; 32] },
            coinbase: vec![],
            timestamp: time,
            target: [0xff; 32],
            state_root: [0; 32],
            prev_header_hash: [0; 32],
        }
    }

    fn burn_tx(outputs: Vec<OutputData>) -> Transaction {
        let coin = InputReveal { predicate: Predicate::p2pk(&[1; 32]), value: 1 << 40, salt: [1; 32], commitment: None };
        Transaction::Reveal { inputs: vec![coin], witnesses: vec![Witness::ScriptInputs(vec![])], outputs, salt: [0; 32] }
    }

    fn store(blocks: &[Vec<Transaction>]) -> (tempfile::TempDir, Arc<Database>, BatchStore) {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("db")).unwrap());
        let bs = BatchStore::new(db.clone(), dir.path().join("legacy")).unwrap();
        bs.burns().enable(&db, 0).unwrap();
        for (i, txs) in blocks.iter().enumerate() {
            bs.save(i as u64, &block(i as u8 + 1, 1_000 + i as u64, txs.clone())).unwrap();
        }
        (dir, db, bs)
    }

    fn nick() -> Backing {
        Backing::name("nick", "hello").unwrap()
    }

    #[test]
    fn follows_burns_through_reorgs_and_ranks_subjects() {
        let r = block_reward(0);
        let pool = Backing::new(Purpose::POOL, [9; 32], "pool one").unwrap();
        let (_d, db, bs) = store(&[
            vec![burn_tx(nick().outputs(3 * r))],
            vec![burn_tx([pool.outputs(10 * r), vec![OutputData::DataBurn { payload: b"MSN1nick".to_vec(), value_burned: 8 }]].concat())],
            vec![burn_tx(Backing::name("nick", "").unwrap().outputs(r)), burn_tx(vec![OutputData::DataBurn { payload: b"MDXA".to_vec(), value_burned: 0 }])],
        ]);
        let m = meta(&db).unwrap();
        assert!(m.ready());
        assert_eq!((m.totals.names, m.totals.backing[1], m.totals.backing[4], m.totals.count), (8, 4 * r, 10 * r, 4));
        let n = backing(&db, Purpose::NAME, &nick().subject, None).unwrap().unwrap();
        assert_eq!((n.total, n.count, n.first, n.last, n.label.as_str()), (4 * r, 2, 0, 2, "hello"));
        assert!((n.backing - 4.0).abs() < 1e-3, "{}", n.backing);
        let h = history(&db, Purpose::NAME, &nick().subject, 10).unwrap();
        assert_eq!(h.iter().map(|b| (b.height, b.value, b.time)).collect::<Vec<_>>(), vec![(2, r, 1_002), (0, 3 * r, 1_000)]);
        assert_eq!(top(&db, Purpose::POOL, 5).unwrap()[0].label, "pool one");
        let rec = recent(&db, 10, 100).unwrap();
        assert_eq!(rec.iter().map(|x| (x.height, x.purpose)).collect::<Vec<_>>(), vec![(2, Purpose::NAME), (1, Purpose::POOL), (0, Purpose::NAME)]);

        // A reorg replaces block 2 with one that backs the pool instead.
        bs.save(2, &block(9, 1_009, vec![burn_tx(pool.outputs(r))])).unwrap();
        let n = backing(&db, Purpose::NAME, &nick().subject, None).unwrap().unwrap();
        assert_eq!((n.total, n.count, n.last), (3 * r, 1, 0));
        assert_eq!(backing(&db, Purpose::POOL, &pool.subject, None).unwrap().unwrap().total, 11 * r);
        let m = meta(&db).unwrap();
        assert_eq!((m.totals.backing[1], m.totals.backing[4]), (3 * r, 11 * r));

        // Truncated to nothing: nothing left, ranks included.
        bs.truncate(0).unwrap();
        let m = meta(&db).unwrap();
        assert_eq!((m.totals.clone(), m.next), (Totals::default(), 0));
        assert!(backing(&db, Purpose::NAME, &nick().subject, None).unwrap().is_none());
        let txn = db.begin_read().unwrap();
        assert_eq!(txn.open_table(RANKS).unwrap().iter().unwrap().count(), 0);
        assert_eq!(txn.open_table(DAYS).unwrap().iter().unwrap().count(), 0);
    }

    #[test]
    fn the_backfill_catches_up_to_what_the_hooks_would_have_built() {
        let r = block_reward(0);
        let blocks: Vec<Vec<Transaction>> = (0..5u64).map(|i| vec![burn_tx(nick().outputs((i + 1) * r))]).collect();
        let (_d, live, _) = store(&blocks);
        // The same chain stored with the index off, then turned on.
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("db")).unwrap());
        let bs = BatchStore::new(db.clone(), dir.path().join("legacy")).unwrap();
        for (i, txs) in blocks.iter().enumerate() {
            bs.save(i as u64, &block(i as u8 + 1, 1_000 + i as u64, txs.clone())).unwrap();
        }
        bs.burns().enable(&db, 5).unwrap();
        assert!(backing(&db, Purpose::NAME, &nick().subject, None).is_err(), "not built yet");
        bs.burns().catch_up(&db).unwrap();
        assert_eq!(meta(&db).unwrap().totals, meta(&live).unwrap().totals);
        assert_eq!(backing(&db, Purpose::NAME, &nick().subject, None).unwrap(), backing(&live, Purpose::NAME, &nick().subject, None).unwrap());
        let (years, trailing) = years(&db).unwrap();
        assert_eq!(years.len(), 1);
        assert_eq!(years[0].issued, 5 * r);
        assert_eq!(years[0].burned.backing[1], 15 * r);
        assert_eq!(trailing.burned.backing_total(), 15 * r);
    }
}
