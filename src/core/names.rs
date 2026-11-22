//! MSN-1 names: `nick.mds` resolves to typed records — where to pay it in MDS
//! and MDW, its Base account, its identity key, a line of text — and can be
//! bought, sold and rented, built from script features the chain already has.
//! No consensus change.
//!
//! The registry is one state thread at [`address`]: a zero-value coin whose
//! commitment is the root of a depth-[`DEPTH`] Merkle tree. The leaves form a
//! sorted linked list,
//!
//! ```text
//! leaf = H(key ‖ next ‖ expires ‖ tier ‖ owner ‖ data ‖ deal)   8+8+8+1+32+32+32 bytes
//! key  = H("MSN1" ‖ name)[..8], a little-endian u64
//! data = H(records),  deal = H(terms)
//! ```
//!
//! starting at a sentinel in slot 0, then every [premium](premium) name,
//! unclaimed. Empty slots are `0^32`.
//!
//! * `owner` is the control key: it signs changes and nothing else.
//! * `records` say where payments go ([`Rec`]).
//! * A name is held until `expires`. Every year of holding burns its tier's
//!   [rate](RATES): a premium name's listed tier, or its length's.
//! * `terms` ([`Deal`]) offer it for sale or for rent, or record a rental.
//!
//! The [`covenant`] allows five spends, each only into a new thread at the
//! same address (output 0) whose root it computes:
//!
//! * **register** a name strictly between two neighbours (so a name exists
//!   once), burning its years;
//! * **update** it, signed by its owner, burning any years added;
//! * **reclaim** it once lapsed, burning its years — premium names start so;
//! * **buy** it as its terms offer, paying their price to their payee;
//! * **rent** it as its terms offer: the renter's records, until a height,
//!   with its owner locked out until then.
//!
//! Anyone can make look-alike threads at the same address. The registry is the
//! lineage that starts at [`genesis`] and follows output 0 of each spend, which
//! only the covenant can extend; `storage::name_index` follows it.
//!
//! Names are canonical ASCII ([`normalize`]). The covenant hashes whatever name
//! a witness carries, so a spend can register a name that is not canonical;
//! no client ever looks one up, so it reaches nobody.

use super::script::{
    push_data, OP_ADD, OP_CHECKSIGVERIFY, OP_CHECKTIMEVERIFY, OP_CONCAT, OP_DROP, OP_DUP, OP_ELSE, OP_ENDIF,
    OP_EQUAL, OP_EQUALVERIFY, OP_GREATER_OR_EQUAL, OP_HASH, OP_IF, OP_LESS_THAN, OP_MERKLE_ROOT, OP_MUL, OP_NIP,
    OP_OUTPUT_ADDRESS, OP_PICK, OP_READ_INPUT_STATE, OP_READ_OUTPUT_STATE, OP_ROT, OP_SIZE, OP_SLICE, OP_SUB,
    OP_SUM_TO_ADDR, OP_SWAP, OP_THIS_ADDRESS, OP_VERIFY,
};
use super::types::{compute_address, encode_address_with_checksum, hash, hash_concat, InputReveal, OutputData, Predicate};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const TAG: &[u8; 4] = b"MSN1";
pub const SUFFIX: &str = ".mds";
pub const MAX_NAME_LEN: usize = 32;
/// Tree depth: room for 2^24 names, decades of one thread's registrations.
pub const DEPTH: usize = 24;
pub const CAPACITY: u64 = 1 << DEPTH;
/// One 33-byte record (sibling ‖ side) per level, as `OP_MERKLE_ROOT` reads it.
pub const PROOF_LEN: usize = DEPTH * 33;
/// A year of holding: 525,600 sixty-second blocks.
pub const YEAR: u64 = 525_600;
/// The most years one spend may buy.
pub const MAX_YEARS: u64 = 10;
/// A rental ends at least 30 days before the name lapses, so its holder can renew.
pub const GRACE: u64 = 43_200;
/// Where trackers start looking for [`genesis`]: the covenant's opcodes
/// activate here, so no registry spend can be older.
pub const START_HEIGHT: u64 = super::types::V6_ACTIVATION_HEIGHT;
/// Burns are outputs that pay this address: `DataBurn`s report it, and no
/// script hashes to it.
pub const BURN: [u8; 32] = [0; 32];

// ── Names ───────────────────────────────────────────────────────────────────

/// The canonical form of a name: ASCII `a-z`, `0-9` and `-`, 1 to 32
/// characters, no `-` at either end. Case and a trailing `.mds` are ignored.
pub fn normalize(input: &str) -> Result<String> {
    let lower = input.trim().to_ascii_lowercase();
    let s = lower.strip_suffix(SUFFIX).unwrap_or(&lower);
    if s.is_empty() || s.len() > MAX_NAME_LEN {
        bail!("a name has 1 to {MAX_NAME_LEN} characters");
    }
    if !s.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
        bail!("a name uses only a-z, 0-9 and -");
    }
    if s.starts_with('-') || s.ends_with('-') {
        bail!("a name cannot start or end with -");
    }
    Ok(s.to_string())
}

/// Whether a destination someone typed is a name rather than an address.
pub fn is_name(s: &str) -> bool {
    s.trim().to_ascii_lowercase().ends_with(SUFFIX)
}

/// `H("MSN1" ‖ name)[..8]`, as the covenant computes it.
pub fn key(name: &[u8]) -> [u8; 8] {
    let mut pre = TAG.to_vec();
    pre.extend_from_slice(name);
    hash(&pre)[..8].try_into().unwrap()
}

/// A key's place in the list.
pub fn ord(key: &[u8; 8]) -> u64 {
    u64::from_le_bytes(*key)
}


// ── Tiers ───────────────────────────────────────────────────────────────────

/// Price tiers, dearest first.
pub const TIERS: [&str; 4] = ["platinum", "gold", "silver", "bronze"];
/// What a year of each tier burns, in units: 64, 8, 1 and ⅛ gMDS.
pub const RATES: [u64; 4] = [1 << 36, 1 << 33, 1 << 30, 1 << 27];

/// A name's tier by its length alone: 1–2 characters platinum, 3 gold, 4
/// silver, longer bronze. The covenant computes the same.
pub fn tier_for_len(len: usize) -> u8 {
    (len.max(2) - 2).min(3) as u8
}

/// Names in demand beyond their length, by tier. Each is priced above what
/// its length alone would make it.
const PREMIUM: [&str; 3] = [
    // platinum
    "btc eth usd nft dao dex pay mds mdw bitcoin ethereum crypto wallet exchange money bank cash \
     coin coins gold midstate midwimble trade market shop store mail email news music video games \
     google apple amazon microsoft facebook twitter tesla nvidia openai binance coinbase paypal visa \
     mastercard payments finance invest",
    // gold
    "love home life travel food health sport sports casino poker bets defi swap mine miner mining \
     pool node chain block blocks token tokens stake staking yield loan loans credit insurance hotel \
     cars auto phone mobile cloud data apps code tech social chat dating netflix spotify youtube \
     tiktok instagram whatsapp telegram discord reddit github samsung sony nike adidas uber airbnb \
     alibaba tencent ebay walmart oracle intel cisco wiki blog film radio media studio design \
     agency energy solar jobs work school learn academy doctor legal",
    // silver
    "james david michael sarah maria peter daniel olivia sophia emily jessica thomas robert william \
     joseph charles george oliver harry jacob ethan lucas mason logan alexander benjamin samuel \
     henry nicholas matthew andrew joshua kevin brian jason steven edward laura linda nancy karen \
     susan betty helen sandra donna carol world earth space future global smart happy lucky magic \
     power super ultra prime royal crown diamond silver bronze platinum dragon tiger eagle phoenix \
     rocket galaxy planet ocean river forest mountain garden coffee pizza burger fashion beauty \
     fitness photo camera books library museum gallery theater cinema concert festival wedding \
     family friends",
];

/// Names held from genesis, by their control key (hex), free until
/// [`RESERVED_UNTIL`]: the project's own, so no one else can pose as it. An
/// entry whose key is not filled in reserves nothing. After that height a
/// reserved name renews, lapses and is reclaimed like any other.
pub const RESERVED: [(&str, &str); 1] = [("ciphernom", "32ec53a747cfbccf3dbdab7ecdfda7e51c00c37ccf99e98fcbf76902681e2448")];
/// About 14 months after this was written.
pub const RESERVED_UNTIL: u64 = 1_000_000_000_000;

/// The reserved names whose keys are set.
pub fn reserved() -> Vec<(String, [u8; 32])> {
    RESERVED
        .iter()
        .filter_map(|(n, k)| Some((n.to_string(), hex::decode(k).ok()?.try_into().ok()?)))
        .collect()
}

/// A name in the registry from genesis: premium (unclaimed) or reserved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seeded {
    pub name: String,
    pub tier: u8,
    pub owner: [u8; 32],
    pub expires: u64,
}

/// Every name the genesis holds, sorted by key: slot 1 onwards.
pub fn seeded() -> &'static [Seeded] {
    static S: OnceLock<Vec<Seeded>> = OnceLock::new();
    S.get_or_init(|| seed(&reserved()))
}

fn seed(reserved: &[(String, [u8; 32])]) -> Vec<Seeded> {
    let mut v: Vec<Seeded> = premium().iter().map(|(n, t)| Seeded { name: n.clone(), tier: *t, owner: [0; 32], expires: 0 }).collect();
    for (n, owner) in reserved {
        v.retain(|s| &s.name != n);
        v.push(Seeded { tier: tier_of(n), name: n.clone(), owner: *owner, expires: RESERVED_UNTIL });
    }
    v.sort_by_key(|s| ord(&key(s.name.as_bytes())));
    v
}

/// The premium names and their tiers, sorted by key.
pub fn premium() -> &'static [(String, u8)] {
    static P: OnceLock<Vec<(String, u8)>> = OnceLock::new();
    P.get_or_init(|| {
        let mut v: Vec<(String, u8)> =
            PREMIUM.iter().enumerate().flat_map(|(t, s)| s.split_whitespace().map(move |n| (n.to_string(), t as u8))).collect();
        v.sort_by_key(|(n, _)| ord(&key(n.as_bytes())));
        v
    })
}

/// What a name costs a year: its premium tier if listed, else its length's.
pub fn tier_of(name: &str) -> u8 {
    premium().iter().find(|(n, _)| n == name).map_or(tier_for_len(name.len()), |(_, t)| *t)
}

/// The burn for `years` of a tier.
pub fn cost(tier: u8, years: u64) -> u64 {
    RATES[tier as usize].saturating_mul(years)
}

// ── Deals ───────────────────────────────────────────────────────────────────

/// What a name's holder offers, committed in its leaf as `H(terms)`. Terms are
/// `kind ‖ price (8) ‖ payee (32) ‖ time (8)`, or `[0]` for none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Deal {
    None,
    /// Anyone may take the name by paying `price` to `payee`.
    Sale { price: u64, payee: [u8; 32] },
    /// Anyone may take its records for `span` blocks by paying `price` to `payee`.
    Rent { price: u64, payee: [u8; 32], span: u64 },
    /// Taken: its records are the renter's, and its holder cannot change it, until `until`.
    Rented { price: u64, payee: [u8; 32], until: u64 },
}

impl Deal {
    pub fn terms(&self) -> Vec<u8> {
        let mut t = Vec::with_capacity(49);
        let (kind, price, payee, time) = match *self {
            Deal::None => return vec![0],
            Deal::Sale { price, payee } => (1, price, payee, None),
            Deal::Rent { price, payee, span } => (2, price, payee, Some(span)),
            Deal::Rented { price, payee, until } => (3, price, payee, Some(until)),
        };
        t.push(kind);
        t.extend_from_slice(&price.to_le_bytes());
        t.extend_from_slice(&payee);
        if let Some(time) = time {
            t.extend_from_slice(&time.to_le_bytes());
        }
        t
    }

    /// Terms as a client reads them; anything else is no offer.
    pub fn parse(t: &[u8]) -> Deal {
        let price = || u64::from_le_bytes(t[1..9].try_into().unwrap());
        let payee = || t[9..41].try_into().unwrap();
        let time = || u64::from_le_bytes(t[41..49].try_into().unwrap());
        match (t.first(), t.len()) {
            (Some(1), 41) => Deal::Sale { price: price(), payee: payee() },
            (Some(2), 49) => Deal::Rent { price: price(), payee: payee(), span: time() },
            (Some(3), 49) => Deal::Rented { price: price(), payee: payee(), until: time() },
            _ => Deal::None,
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        hash(&self.terms())
    }
}

// ── Leaves ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leaf {
    pub key: [u8; 8],
    pub next: [u8; 8],
    pub expires: u64,
    pub tier: u8,
    pub owner: [u8; 32],
    /// `H(records)`.
    pub data: [u8; 32],
    /// `H(terms)`.
    pub deal: [u8; 32],
}

impl Leaf {
    pub const LEN: usize = 121;

    /// Slot 0: before every key, never lapsing, owned by no key.
    pub fn sentinel(next: [u8; 8]) -> Leaf {
        Leaf { key: [0; 8], next, expires: u64::MAX, tier: 0, owner: [0; 32], data: hash(&[]), deal: Deal::None.hash() }
    }

    pub fn bytes(&self) -> [u8; Self::LEN] {
        let mut b = [0u8; Self::LEN];
        b[..8].copy_from_slice(&self.key);
        b[8..16].copy_from_slice(&self.next);
        b[16..24].copy_from_slice(&self.expires.to_le_bytes());
        b[24] = self.tier;
        b[25..57].copy_from_slice(&self.owner);
        b[57..89].copy_from_slice(&self.data);
        b[89..].copy_from_slice(&self.deal);
        b
    }

    pub fn from_bytes(b: &[u8; Self::LEN]) -> Self {
        Leaf {
            key: b[..8].try_into().unwrap(),
            next: b[8..16].try_into().unwrap(),
            expires: u64::from_le_bytes(b[16..24].try_into().unwrap()),
            tier: b[24],
            owner: b[25..57].try_into().unwrap(),
            data: b[57..89].try_into().unwrap(),
            deal: b[89..].try_into().unwrap(),
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        hash(&self.bytes())
    }

    /// Everything after `next`: what a neighbour keeps when it is re-pointed.
    fn tail(&self) -> Vec<u8> {
        self.bytes()[16..].to_vec()
    }
}

/// The registry at genesis: the sentinel, then each [seeded] name — premium
/// ones unclaimed, lapsed since block 0, so the first to pay its tier's price
/// reclaims it; reserved ones held by their keys.
pub fn genesis_leaves() -> Vec<(u64, Leaf)> {
    leaves_of(seeded())
}

fn leaves_of(s: &[Seeded]) -> Vec<(u64, Leaf)> {
    let k = |i: usize| s.get(i).map_or([0xff; 8], |x| key(x.name.as_bytes()));
    let mut out = vec![(0, Leaf::sentinel(k(0)))];
    for (i, x) in s.iter().enumerate() {
        let leaf = Leaf { key: key(x.name.as_bytes()), next: k(i + 1), expires: x.expires, tier: x.tier, owner: x.owner, data: hash(&[]), deal: Deal::None.hash() };
        out.push((i as u64 + 1, leaf));
    }
    out
}

// ── Records ─────────────────────────────────────────────────────────────────

/// Record tags. A client keeps the ones it does not know and ignores them.
pub mod tag {
    /// An MSS key, and its tree height: pay `compute_address(key)`.
    pub const MDS_KEY: u8 = 1;
    /// Any midstate address (multisig, vault, plan): pay it as it is.
    pub const MDS_ADDRESS: u8 = 2;
    /// A midwimble stealth address, as text (`mw1…`).
    pub const MDW: u8 = 3;
    /// A Base (Ethereum) account.
    pub const EVM: u8 = 4;
    /// The wallet's identity key: ask it for a fresh address, an invoice, a channel.
    pub const IDENTITY: u8 = 5;
    /// A line of UTF-8.
    pub const TEXT: u8 = 6;
    // 7 is held for a Nostr public key (32 bytes x-only), not built yet.
    /// `H(key)` of the name's ML-KEM-512 public key, for sealed messages. The
    /// key (800 bytes) is too big for a record; the identity handshake hands
    /// it over and the hash checks it.
    pub const KEM: u8 = 8;
    /// Part of the script behind an `MDS_ADDRESS`: the parts, in order, are
    /// the script, so anyone can see what kind of address the name pays.
    pub const SCRIPT: u8 = 9;
}

/// One record: encoded as `tag ‖ length (1 byte) ‖ value`, back to back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rec {
    MdsKey { key: [u8; 32], height: u8 },
    MdsAddress([u8; 32]),
    Mdw(String),
    Evm([u8; 20]),
    Identity([u8; 32]),
    Text(String),
    Kem([u8; 32]),
    Script(Vec<u8>),
    Other { tag: u8, value: Vec<u8> },
}

/// Payments skip a key with fewer signatures left than this: its owner is
/// keeping them to move what it already holds.
pub const SIGNATURES_KEPT: u64 = 16;

impl Rec {
    fn parts(&self) -> (u8, Vec<u8>) {
        match self {
            Rec::MdsKey { key, height } => (tag::MDS_KEY, [&key[..], &[*height]].concat()),
            Rec::MdsAddress(a) => (tag::MDS_ADDRESS, a.to_vec()),
            Rec::Mdw(s) => (tag::MDW, s.as_bytes().to_vec()),
            Rec::Evm(a) => (tag::EVM, a.to_vec()),
            Rec::Identity(k) => (tag::IDENTITY, k.to_vec()),
            Rec::Text(s) => (tag::TEXT, s.as_bytes().to_vec()),
            Rec::Kem(h) => (tag::KEM, h.to_vec()),
            Rec::Script(b) => (tag::SCRIPT, b.clone()),
            Rec::Other { tag, value } => (*tag, value.clone()),
        }
    }

    fn from_parts(tag: u8, v: &[u8]) -> Rec {
        let text = |v: &[u8]| String::from_utf8(v.to_vec()).ok();
        let known = match tag {
            tag::MDS_KEY if v.len() == 33 && (1..=20).contains(&v[32]) => {
                Some(Rec::MdsKey { key: v[..32].try_into().unwrap(), height: v[32] })
            }
            tag::MDS_ADDRESS => v.try_into().ok().map(Rec::MdsAddress),
            tag::MDW if v.is_ascii() => text(v).map(Rec::Mdw),
            tag::EVM => v.try_into().ok().map(Rec::Evm),
            tag::IDENTITY => v.try_into().ok().map(Rec::Identity),
            tag::TEXT => text(v).map(Rec::Text),
            tag::KEM => v.try_into().ok().map(Rec::Kem),
            tag::SCRIPT if !v.is_empty() => Some(Rec::Script(v.to_vec())),
            _ => None,
        };
        known.unwrap_or(Rec::Other { tag, value: v.to_vec() })
    }

    /// The record for people: its kind and what it says.
    pub fn describe(&self) -> (&'static str, String) {
        match self {
            Rec::MdsKey { key, .. } => ("mds-key", encode_address_with_checksum(&compute_address(key))),
            Rec::MdsAddress(a) => ("mds-address", encode_address_with_checksum(a)),
            Rec::Mdw(s) => ("mdw", s.clone()),
            Rec::Evm(a) => ("evm", format!("0x{}", hex::encode(a))),
            Rec::Identity(k) => ("identity", hex::encode(k)),
            Rec::Text(t) => ("text", t.clone()),
            Rec::Kem(h) => ("kem", hex::encode(h)),
            Rec::Script(b) => ("script", hex::encode(b)),
            Rec::Other { tag, value } => ("other", format!("{tag}:{}", hex::encode(value))),
        }
    }

    /// Where an MDS payment to this record goes, given how many signatures an
    /// MSS key has used on chain.
    pub fn pays(&self, used: impl Fn(&[u8; 32]) -> u64) -> Option<[u8; 32]> {
        match self {
            Rec::MdsAddress(a) => Some(*a),
            Rec::MdsKey { key, height } => {
                let left = (1u64 << *height).saturating_sub(used(key));
                (left >= SIGNATURES_KEPT).then(|| compute_address(key))
            }
            _ => None,
        }
    }
}

/// Records as a witness carries them. Fails on a value over 255 bytes or a
/// blob over the VM's item size.
pub fn encode(recs: &[Rec]) -> Result<Vec<u8>> {
    let mut b = Vec::new();
    for r in recs {
        let (t, v) = r.parts();
        if v.len() > 255 {
            bail!("a record holds at most 255 bytes");
        }
        b.push(t);
        b.push(v.len() as u8);
        b.extend_from_slice(&v);
    }
    if b.len() > super::script::MAX_ITEM_SIZE {
        bail!("records take at most {} bytes", super::script::MAX_ITEM_SIZE);
    }
    Ok(b)
}

/// The records in a blob; `None` if it is not a run of whole records.
pub fn parse(blob: &[u8]) -> Option<Vec<Rec>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < blob.len() {
        let len = *blob.get(i + 1)? as usize;
        let v = blob.get(i + 2..i + 2 + len)?;
        out.push(Rec::from_parts(blob[i], v));
        i += 2 + len;
    }
    Some(out)
}

/// Where an MDS payment to a name goes: its first record that takes one.
pub fn pays(recs: &[Rec], used: impl Fn(&[u8; 32]) -> u64) -> Option<[u8; 32]> {
    recs.iter().find_map(|r| r.pays(&used))
}

/// The script the records publish: their `script` parts, in order.
pub fn published_script(recs: &[Rec]) -> Option<Vec<u8>> {
    let s: Vec<u8> = recs.iter().filter_map(|r| if let Rec::Script(b) = r { Some(b.as_slice()) } else { None }).flatten().copied().collect();
    (!s.is_empty()).then_some(s)
}

/// `script` records publishing `script`, in parts of at most 255 bytes.
pub fn script_records(script: &[u8]) -> Vec<Rec> {
    script.chunks(255).map(|c| Rec::Script(c.to_vec())).collect()
}

/// What kind of address `address` is, as far as the records show: an MSS key
/// record's address is a key; an `mds-address` is whatever the published
/// script makes, if the records publish the script that hashes to it.
/// `None`: the records do not say.
pub fn address_kind(recs: &[Rec], address: &[u8; 32]) -> Option<super::address_kind::AddressKind> {
    if recs.iter().any(|r| matches!(r, Rec::MdsKey { key, .. } if &compute_address(key) == address)) {
        return Some(super::address_kind::AddressKind::Key);
    }
    let script = published_script(recs)?;
    (&hash(&script) == address).then(|| super::address_kind::classify(&script))
}

/// The hash of the name's ML-KEM public key, if it takes sealed messages.
pub fn kem_hash(recs: &[Rec]) -> Option<[u8; 32]> {
    recs.iter().find_map(|r| if let Rec::Kem(h) = r { Some(*h) } else { None })
}

/// The name's identity key, if it gives one.
pub fn identity(recs: &[Rec]) -> Option<[u8; 32]> {
    recs.iter().find_map(|r| if let Rec::Identity(k) = r { Some(*k) } else { None })
}

// ── The tree ────────────────────────────────────────────────────────────────

/// Stored tree nodes: level 0 holds leaves, level [`DEPTH`] the root. A node
/// that is not stored is the empty subtree's ([`empty`]).
pub trait Nodes {
    fn stored(&self, level: usize, index: u64) -> Result<Option<[u8; 32]>>;
}

/// An in-memory tree.
pub type Mem = HashMap<(usize, u64), [u8; 32]>;

impl Nodes for Mem {
    fn stored(&self, level: usize, index: u64) -> Result<Option<[u8; 32]>> {
        Ok(self.get(&(level, index)).copied())
    }
}

/// `top` over `base`: a tree with a few nodes changed.
pub struct Overlay<'a, N: ?Sized> {
    pub base: &'a N,
    pub top: Mem,
}

impl<N: Nodes + ?Sized> Nodes for Overlay<'_, N> {
    fn stored(&self, level: usize, index: u64) -> Result<Option<[u8; 32]>> {
        match self.top.get(&(level, index)) {
            Some(h) => Ok(Some(*h)),
            None => self.base.stored(level, index),
        }
    }
}

/// The root of an empty subtree of height `level`: `E0 = 0^32`,
/// `E(i+1) = H(Ei ‖ Ei)`.
pub fn empty(level: usize) -> [u8; 32] {
    static E: OnceLock<[[u8; 32]; DEPTH + 1]> = OnceLock::new();
    E.get_or_init(|| {
        let mut e = [[0u8; 32]; DEPTH + 1];
        for l in 1..=DEPTH {
            e[l] = hash_concat(&e[l - 1], &e[l - 1]);
        }
        e
    })[level]
}

fn node<N: Nodes + ?Sized>(n: &N, level: usize, index: u64) -> Result<[u8; 32]> {
    Ok(n.stored(level, index)?.unwrap_or_else(|| empty(level)))
}

pub fn root<N: Nodes + ?Sized>(n: &N) -> Result<[u8; 32]> {
    node(n, DEPTH, 0)
}

/// Slot `pos`'s proof, as `OP_MERKLE_ROOT` reads it: per level, the sibling
/// and a side byte (1: this side is the right one).
pub fn proof<N: Nodes + ?Sized>(n: &N, pos: u64) -> Result<Vec<u8>> {
    let mut p = Vec::with_capacity(PROOF_LEN);
    let mut i = pos;
    for level in 0..DEPTH {
        p.extend_from_slice(&node(n, level, i ^ 1)?);
        p.push((i & 1) as u8);
        i >>= 1;
    }
    Ok(p)
}

/// The slot a proof is for.
pub fn position(proof: &[u8]) -> u64 {
    (0..proof.len() / 33).map(|l| (proof[l * 33 + 32] as u64 & 1) << l).sum()
}

/// Every node on slot `pos`'s path once its leaf is `leaf` (`None`: empty),
/// from the leaf up; the root is last.
pub fn path<N: Nodes + ?Sized>(n: &N, pos: u64, leaf: Option<[u8; 32]>) -> Result<Vec<(usize, u64, [u8; 32])>> {
    let mut out = Vec::with_capacity(DEPTH + 1);
    let mut acc = leaf.unwrap_or([0; 32]);
    let mut i = pos;
    for level in 0..DEPTH {
        out.push((level, i, acc));
        let sib = node(n, level, i ^ 1)?;
        acc = if i & 1 == 0 { hash_concat(&acc, &sib) } else { hash_concat(&sib, &acc) };
        i >>= 1;
    }
    out.push((DEPTH, 0, acc));
    Ok(out)
}

/// Write a path into an in-memory tree, dropping nodes that are empty again.
pub fn write(m: &mut Mem, path: &[(usize, u64, [u8; 32])]) {
    for &(l, i, h) in path {
        if h == empty(l) {
            m.remove(&(l, i));
        } else {
            m.insert((l, i), h);
        }
    }
}

// ── The registry thread ─────────────────────────────────────────────────────

/// Bytecode with a model of the stack, so each `PICK` is by name and every
/// branch is checked to leave the stack as the other does.
struct Asm {
    b: Vec<u8>,
    /// Labels, top last; `~` for an unnamed value.
    s: Vec<&'static str>,
    frames: Vec<(Vec<&'static str>, Option<Vec<&'static str>>)>,
}

impl Asm {
    fn new(items: &[&'static str]) -> Self {
        Asm { b: Vec::new(), s: items.to_vec(), frames: Vec::new() }
    }
    fn raw_num(&mut self, v: u64) {
        let le = v.to_le_bytes();
        let len = (8 - le.iter().rev().take_while(|&&x| x == 0).count()).max(1);
        push_data(&mut self.b, &le[..len]);
    }
    fn num(&mut self, v: u64) {
        self.raw_num(v);
        self.s.push("~");
    }
    fn data(&mut self, d: &[u8]) {
        push_data(&mut self.b, d);
        self.s.push("~");
    }
    /// `op`, taking `pops` values and leaving `pushes`.
    fn op(&mut self, op: u8, pops: usize, pushes: usize) {
        self.b.push(op);
        assert!(self.s.len() >= pops, "stack underflow at op {op:#x}");
        self.s.truncate(self.s.len() - pops);
        self.s.extend(std::iter::repeat_n("~", pushes));
    }
    fn name(&mut self, label: &'static str) {
        *self.s.last_mut().expect("something to name") = label;
    }
    fn pick(&mut self, label: &'static str) {
        let i = self.s.iter().rposition(|x| *x == label).unwrap_or_else(|| panic!("{label} is not on the stack"));
        let depth = (self.s.len() - 1 - i) as u64;
        self.raw_num(depth);
        self.b.push(OP_PICK);
        self.s.push("~");
    }
    fn if_(&mut self) {
        self.op(OP_IF, 1, 0);
        self.frames.push((self.s.clone(), None));
    }
    fn else_(&mut self) {
        self.b.push(OP_ELSE);
        let f = self.frames.last_mut().expect("an IF");
        f.1 = Some(std::mem::replace(&mut self.s, f.0.clone()));
    }
    fn endif(&mut self) {
        self.b.push(OP_ENDIF);
        let (entry, then) = self.frames.pop().expect("an IF");
        assert_eq!(self.s, then.unwrap_or(entry), "both ways through an IF leave the same stack");
    }
    /// Fails unless `label` is `len` bytes; the stack is left as it was.
    fn sized(&mut self, label: &'static str, len: usize) {
        self.pick(label);
        self.op(OP_SIZE, 0, 1);
        self.num(len as u64);
        self.op(OP_EQUALVERIFY, 2, 0);
        self.op(OP_DROP, 1, 0);
    }
    /// `SLICE(label, at, len)` on top.
    fn field(&mut self, label: &'static str, at: u64, len: u64) {
        self.pick(label);
        self.num(at);
        self.num(len);
        self.op(OP_SLICE, 3, 1);
    }
    /// Fails unless the terms under `label` are of `kind`.
    fn kind_is(&mut self, label: &'static str, kind: u64) {
        self.field(label, 0, 1);
        self.num(kind);
        self.op(OP_EQUALVERIFY, 2, 0);
    }
    /// The hash of the parts in turn, each copied (and hashed first if marked).
    fn leaf(&mut self, parts: &[(&'static str, bool)]) {
        for (i, &(label, hashed)) in parts.iter().enumerate() {
            self.pick(label);
            if hashed {
                self.op(OP_HASH, 1, 1);
            }
            if i > 0 {
                self.op(OP_CONCAT, 2, 1);
            }
        }
        self.op(OP_HASH, 1, 1);
    }
    /// Fails unless what the transaction burns covers `years` at `tier`'s rate.
    fn burns(&mut self, tier: &'static str) {
        self.pick("z");
        self.op(OP_SUM_TO_ADDR, 1, 1);
        self.pick("rates");
        self.pick(tier);
        self.num(RATE_LEN);
        self.op(OP_MUL, 2, 1);
        self.num(RATE_LEN);
        self.op(OP_SLICE, 3, 1);
        self.pick("years");
        self.op(OP_MUL, 2, 1);
        self.op(OP_GREATER_OR_EQUAL, 2, 1);
        self.op(OP_VERIFY, 1, 0);
    }
    /// Fails unless the terms under `label` were paid for: their price to their payee.
    fn paid(&mut self, label: &'static str) {
        self.field(label, 9, 32);
        self.op(OP_SUM_TO_ADDR, 1, 1);
        self.field(label, 1, 8);
        self.op(OP_GREATER_OR_EQUAL, 2, 1);
        self.op(OP_VERIFY, 1, 0);
    }
    fn ok(&mut self, op: u8) {
        self.op(op, 2, 1);
        self.op(OP_VERIFY, 1, 0);
    }
    /// Remove everything under the top.
    fn keep_top(&mut self, below: usize) {
        while self.s.len() > below + 1 {
            self.op(OP_NIP, 2, 1);
        }
    }
}

/// Each rate as the covenant's table holds it: 5 bytes, little-endian.
const RATE_LEN: u64 = 5;

/// Witness layouts, bottom to top (the name's length and key are computed
/// from `name` and replace it).
const REGISTER: [&str; 12] = ["P2", "P1", "ltail", "lnext", "lkey", "years", "expires", "owner", "records", "terms", "sel", "name"];
const CHANGE: [&str; 15] = [
    "P", "auth", "oterms", "oowner", "odata", "otier", "oexpires", "next", "years", "expires", "owner", "records", "terms", "sel", "name",
];

/// The registry's covenant. The selector is exactly 0 (register), 1 (update,
/// signed into `auth` by the owner), 2 (reclaim), 3 (buy) or 4 (rent); see
/// [`REGISTER`] and [`CHANGE`] for the witnesses. `P1` proves the neighbour's
/// slot in the spent root, `P2` the new slot once the neighbour points at the
/// name; `ltail` is the neighbour's leaf after its `next`. Sizes are checked so
/// a leaf's bytes cannot be split differently from how they were written.
pub fn covenant() -> &'static [u8] {
    static C: OnceLock<Vec<u8>> = OnceLock::new();
    C.get_or_init(|| {
        // name → its length and key; the zero bytes and rates; the selector on top.
        fn prefix(a: &mut Asm) {
            a.op(OP_SIZE, 0, 1);
            a.name("len");
            a.op(OP_SWAP, 2, 2);
            a.s.truncate(a.s.len() - 2);
            a.s.extend(["len", "name"]);
            a.data(TAG);
            a.op(OP_SWAP, 2, 2);
            a.op(OP_CONCAT, 2, 1);
            a.op(OP_HASH, 1, 1);
            a.num(0);
            a.num(8);
            a.op(OP_SLICE, 3, 1);
            a.name("k");
            a.op(OP_ROT, 3, 3);
            a.s.truncate(a.s.len() - 3);
            a.s.extend(["len", "k", "sel"]);
            // 32 zero bytes (the burn address, and an empty slot), doubled up from one.
            a.num(0);
            for _ in 0..5 {
                a.op(OP_DUP, 1, 2);
                a.op(OP_CONCAT, 2, 1);
            }
            a.name("z");
            let table: Vec<u8> = RATES.iter().flat_map(|r| r.to_le_bytes()[..RATE_LEN as usize].to_vec()).collect();
            a.data(&table);
            a.name("rates");
            a.op(OP_ROT, 3, 3);
            a.s.truncate(a.s.len() - 3);
            a.s.extend(["z", "rates", "sel"]);
        }
        // Holding to `expires` buys at most `years` from `base` (the spent
        // leaf's expiry, or now if later): height ≥ expires − years·YEAR,
        // unless that is at most `base_label`.
        fn term(a: &mut Asm, base: Option<&'static str>) {
            a.pick("years");
            a.num(MAX_YEARS + 1);
            a.ok(OP_LESS_THAN);
            a.pick("expires");
            a.pick("years");
            a.num(YEAR);
            a.op(OP_MUL, 2, 1);
            a.op(OP_SUB, 2, 1);
            match base {
                None => a.op(OP_CHECKTIMEVERIFY, 1, 0),
                Some(b) => {
                    a.op(OP_DUP, 1, 2);
                    a.pick(b);
                    a.num(1);
                    a.op(OP_ADD, 2, 1);
                    a.op(OP_LESS_THAN, 2, 1);
                    a.if_();
                    a.op(OP_DROP, 1, 0);
                    a.else_();
                    a.op(OP_CHECKTIMEVERIFY, 1, 0);
                    a.endif();
                }
            }
        }
        fn at_least_a_year(a: &mut Asm) {
            a.pick("years");
            a.num(1);
            a.ok(OP_GREATER_OR_EQUAL);
        }

        let mut pre = Asm::new(&["name"]);
        pre.s = vec!["sel", "name"];
        prefix(&mut pre);
        let after: [&str; 5] = ["len", "k", "z", "rates", "sel"];
        let mut code = pre.b;
        let mut dispatch = Asm::new(&["sel"]);
        dispatch.op(OP_DUP, 1, 2);
        dispatch.num(0);
        dispatch.op(OP_EQUAL, 2, 1);
        dispatch.b.push(OP_IF);
        code.extend(dispatch.b);

        // Register: a new leaf between lkey and lnext.
        let mut r = Asm::new(&REGISTER[..REGISTER.len() - 2]);
        r.s.extend(after);
        r.op(OP_DROP, 1, 0);
        // Proofs need no size check: a leaf hashes 121 bytes and a node 64, so
        // only a proof of the tree's own depth can reach its root.
        for (l, n) in [("owner", 32), ("expires", 8), ("lkey", 8), ("lnext", 8)] {
            r.sized(l, n);
        }
        at_least_a_year(&mut r);
        term(&mut r, None);
        r.pick("len"); // its tier, by length: min(max(len, 2) − 2, 3)
        r.op(OP_DUP, 1, 2);
        r.num(2);
        r.op(OP_LESS_THAN, 2, 1);
        r.if_();
        r.op(OP_DROP, 1, 0);
        r.num(2);
        r.endif();
        r.num(2);
        r.op(OP_SUB, 2, 1);
        r.op(OP_DUP, 1, 2);
        r.num(3);
        r.op(OP_GREATER_OR_EQUAL, 2, 1);
        r.if_();
        r.op(OP_DROP, 1, 0);
        r.num(3);
        r.endif();
        r.name("tier");
        r.burns("tier");
        r.pick("lkey"); // lkey < k < lnext
        r.pick("k");
        r.ok(OP_LESS_THAN);
        r.pick("k");
        r.pick("lnext");
        r.ok(OP_LESS_THAN);
        r.leaf(&[("lkey", false), ("lnext", false), ("ltail", false)]); // the neighbour, in the spent root
        r.pick("P1");
        r.op(OP_MERKLE_ROOT, 2, 1);
        r.op(OP_READ_INPUT_STATE, 0, 1);
        r.op(OP_EQUALVERIFY, 2, 0);
        r.leaf(&[("lkey", false), ("k", false), ("ltail", false)]); // pointed at k: root M
        r.pick("P1");
        r.op(OP_MERKLE_ROOT, 2, 1);
        r.pick("z"); // the new slot is empty in M
        r.pick("P2");
        r.op(OP_MERKLE_ROOT, 2, 1);
        r.op(OP_EQUALVERIFY, 2, 0);
        r.leaf(&[("k", false), ("lnext", false), ("expires", false), ("tier", false), ("owner", false), ("records", true), ("terms", true)]);
        r.pick("P2");
        r.op(OP_MERKLE_ROOT, 2, 1);
        r.keep_top(0);
        code.extend(r.b);
        code.push(OP_ELSE);

        // Change an existing leaf: authorised four ways, then the same checks.
        let mut c = Asm::new(&CHANGE[..CHANGE.len() - 2]);
        c.s.extend(after);
        fn way(c: &mut Asm, sel: u64, last: bool) {
            if last {
                c.num(sel);
                c.op(OP_EQUALVERIFY, 2, 0);
            } else {
                c.op(OP_DUP, 1, 2);
                c.num(sel);
                c.op(OP_EQUAL, 2, 1);
                c.if_();
                c.op(OP_DROP, 1, 0);
            }
        }
        way(&mut c, 1, false); // update: the owner signs, and a rental locks it until its end
        c.pick("auth");
        c.pick("oowner");
        c.op(OP_CHECKSIGVERIFY, 2, 0);
        c.field("oterms", 0, 1);
        c.num(3);
        c.op(OP_EQUAL, 2, 1);
        c.if_();
        c.field("oterms", 41, 8);
        c.op(OP_CHECKTIMEVERIFY, 1, 0);
        c.endif();
        c.else_();
        way(&mut c, 2, false); // reclaim: it lapsed (height ≥ oexpires + 1)
        c.pick("oexpires");
        c.num(1);
        c.op(OP_ADD, 2, 1);
        c.op(OP_CHECKTIMEVERIFY, 1, 0);
        at_least_a_year(&mut c);
        c.else_();
        way(&mut c, 3, false); // buy: as offered, paid
        c.kind_is("oterms", 1);
        c.paid("oterms");
        c.else_();
        way(&mut c, 4, true); // rent: as offered, paid, owner and expiry kept
        c.kind_is("oterms", 2);
        c.paid("oterms");
        c.kind_is("terms", 3);
        c.field("terms", 1, 40); // same price and payee
        c.field("oterms", 1, 40);
        c.op(OP_EQUALVERIFY, 2, 0);
        c.field("terms", 41, 8); // until ≤ height + span
        c.field("oterms", 41, 8);
        c.op(OP_SUB, 2, 1);
        c.op(OP_CHECKTIMEVERIFY, 1, 0);
        c.field("terms", 41, 8); // until + GRACE ≤ oexpires
        c.num(GRACE);
        c.op(OP_ADD, 2, 1);
        c.pick("oexpires");
        c.num(1);
        c.op(OP_ADD, 2, 1);
        c.ok(OP_LESS_THAN);
        c.pick("owner");
        c.pick("oowner");
        c.op(OP_EQUALVERIFY, 2, 0);
        c.pick("expires");
        c.pick("oexpires");
        c.op(OP_EQUALVERIFY, 2, 0);
        c.endif();
        c.endif();
        c.endif();
        // next, oexpires and otier fix where the old leaf's fields fall; the
        // owner and data after them only need their 64 bytes together.
        for (l, n) in [("owner", 32), ("expires", 8), ("next", 8), ("oexpires", 8), ("otier", 1)] {
            c.sized(l, n);
        }
        term(&mut c, Some("oexpires"));
        c.burns("otier");
        c.leaf(&[("k", false), ("next", false), ("oexpires", false), ("otier", false), ("oowner", false), ("odata", false), ("oterms", true)]);
        c.pick("P");
        c.op(OP_MERKLE_ROOT, 2, 1);
        c.op(OP_READ_INPUT_STATE, 0, 1);
        c.op(OP_EQUALVERIFY, 2, 0);
        c.leaf(&[("k", false), ("next", false), ("expires", false), ("otier", false), ("owner", false), ("records", true), ("terms", true)]);
        c.pick("P");
        c.op(OP_MERKLE_ROOT, 2, 1);
        c.keep_top(0);
        code.extend(c.b);
        code.push(OP_ENDIF);

        // Every spend: output 0 carries the new root at this address.
        let mut t = Asm::new(&["root"]);
        t.num(0);
        t.op(OP_READ_OUTPUT_STATE, 1, 1);
        t.op(OP_EQUALVERIFY, 2, 0);
        t.num(0);
        t.op(OP_OUTPUT_ADDRESS, 1, 1);
        t.op(OP_THIS_ADDRESS, 0, 1);
        t.op(OP_EQUAL, 2, 1);
        assert_eq!(t.s.len(), 1);
        code.extend(t.b);
        code
    })
}

/// The registry's address.
pub fn address() -> [u8; 32] {
    hash(covenant())
}

/// The root of the registry at genesis.
pub fn genesis_root() -> [u8; 32] {
    static R: OnceLock<[u8; 32]> = OnceLock::new();
    *R.get_or_init(|| {
        let mut m = Mem::new();
        for (pos, leaf) in genesis_leaves() {
            let p = path(&m, pos, Some(leaf.hash())).expect("in memory");
            write(&mut m, &p);
        }
        root(&m).expect("in memory")
    })
}

/// The coin the registry starts from. Anyone may create it; the first one at
/// or after [`START_HEIGHT`] is the registry.
pub fn genesis() -> OutputData {
    OutputData::Confidential { address: address(), commitment: genesis_root(), salt: hash(b"MSN1 genesis") }
}

/// The thread a spend of `spent` creates: `root` at the registry's address.
/// The salt follows from the coin spent, so the coin is new.
pub fn next_tip(root: [u8; 32], spent: &[u8; 32]) -> OutputData {
    OutputData::Confidential { address: address(), commitment: root, salt: hash_concat(b"MSN1 tip", spent) }
}

/// The input that spends the thread `(root, salt)`.
pub fn tip_input(root: [u8; 32], salt: [u8; 32]) -> InputReveal {
    InputReveal { predicate: Predicate::Script { bytecode: covenant().to_vec() }, value: 0, salt, commitment: Some(root) }
}

/// The burn outputs for `amount`, each a power of two, naming the name.
pub fn burn_outputs(name: &str, amount: u64) -> Vec<OutputData> {
    let payload = [&TAG[..], name.as_bytes()].concat();
    super::types::decompose_value(amount).into_iter().map(|v| OutputData::DataBurn { payload: payload.clone(), value_burned: v }).collect()
}

// ── Spends ──────────────────────────────────────────────────────────────────

/// How a change to an existing name is allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum By {
    /// Signed by its owner.
    Owner,
    /// It had lapsed.
    Lapse,
    /// Bought as its terms offered.
    Buy,
    /// Rented as its terms offered.
    Rent,
}

impl By {
    fn selector(self) -> u8 {
        match self {
            By::Owner => 1,
            By::Lapse => 2,
            By::Buy => 3,
            By::Rent => 4,
        }
    }
}

/// A registry spend as its witness states it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    /// `name` (as written) into slot `pos`, after `pred` in slot `pred_pos`.
    Register { name: Vec<u8>, owner: [u8; 32], expires: u64, records: Vec<u8>, terms: Vec<u8>, pred: Leaf, pred_pos: u64, pos: u64 },
    /// The leaf in slot `pos`, as it was (with its terms), changed.
    Change {
        by: By,
        name: Vec<u8>,
        leaf: Leaf,
        oterms: Vec<u8>,
        owner: [u8; 32],
        expires: u64,
        records: Vec<u8>,
        terms: Vec<u8>,
        pos: u64,
    },
}

impl Op {
    /// What the spend writes, slot by slot (`None`: empty).
    pub fn writes(&self) -> Vec<(u64, Option<Leaf>)> {
        match self {
            Op::Register { name, owner, expires, records, terms, pred, pred_pos, pos } => {
                let k = key(name);
                let new = Leaf {
                    key: k,
                    next: pred.next,
                    expires: *expires,
                    tier: tier_for_len(name.len()),
                    owner: *owner,
                    data: hash(records),
                    deal: hash(terms),
                };
                vec![(*pred_pos, Some(Leaf { next: k, ..*pred })), (*pos, Some(new))]
            }
            Op::Change { leaf, owner, expires, records, terms, pos, .. } => {
                vec![(*pos, Some(Leaf { owner: *owner, expires: *expires, data: hash(records), deal: hash(terms), ..*leaf }))]
            }
        }
    }

    /// What undoing it writes.
    pub fn undoes(&self) -> Vec<(u64, Option<Leaf>)> {
        match self {
            Op::Register { pred, pred_pos, pos, .. } => vec![(*pos, None), (*pred_pos, Some(*pred))],
            Op::Change { leaf, pos, .. } => vec![(*pos, Some(*leaf))],
        }
    }

    pub fn name(&self) -> &[u8] {
        match self {
            Op::Register { name, .. } | Op::Change { name, .. } => name,
        }
    }
}

/// Reads a witness the covenant accepted. `None` for anything else.
pub fn decode(items: &[Vec<u8>]) -> Option<Op> {
    let b8 = |v: &Vec<u8>| -> Option<[u8; 8]> { v.as_slice().try_into().ok() };
    let b32 = |v: &Vec<u8>| -> Option<[u8; 32]> { v.as_slice().try_into().ok() };
    let height = |v: &Vec<u8>| b8(v).map(u64::from_le_bytes);
    let name = items.last()?.clone();
    let sel = items.get(items.len().checked_sub(2)?)?.as_slice();
    match (sel, items.len()) {
        ([0], 12) => {
            let mut l = [0u8; Leaf::LEN];
            if items[2].len() != Leaf::LEN - 16 {
                return None;
            }
            l[..8].copy_from_slice(&b8(&items[4])?);
            l[8..16].copy_from_slice(&b8(&items[3])?);
            l[16..].copy_from_slice(&items[2]);
            Some(Op::Register {
                owner: b32(&items[7])?,
                expires: height(&items[6])?,
                records: items[8].clone(),
                terms: items[9].clone(),
                pred: Leaf::from_bytes(&l),
                pred_pos: position(&items[1]),
                pos: position(&items[0]),
                name,
            })
        }
        ([s @ 1..=4], 15) => Some(Op::Change {
            by: [By::Owner, By::Lapse, By::Buy, By::Rent][*s as usize - 1],
            leaf: {
                // The covenant fixes the owner and data only as 64 bytes together.
                let od = [&items[3][..], &items[4][..]].concat();
                if od.len() != 64 {
                    return None;
                }
                Leaf {
                    key: key(&name),
                    next: b8(&items[7])?,
                    expires: height(&items[6])?,
                    tier: *items[5].first().filter(|_| items[5].len() == 1)?,
                    owner: od[..32].try_into().unwrap(),
                    data: od[32..].try_into().unwrap(),
                    deal: hash(&items[2]),
                }
            },
            oterms: items[2].clone(),
            owner: b32(&items[10])?,
            expires: height(&items[9])?,
            records: items[11].clone(),
            terms: items[12].clone(),
            pos: position(&items[0]),
            name,
        }),
        _ => None,
    }
}

/// A spend of the thread: its witness (for an update, `items[1]` waits for
/// the owner's signature over the transaction) and the root output 0 carries.
pub struct Spend {
    pub items: Vec<Vec<u8>>,
    pub root: [u8; 32],
}

/// What a spend sets: holder, records, terms, and how long, paid in `years`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Set {
    pub owner: [u8; 32],
    pub records: Vec<u8>,
    pub terms: Vec<u8>,
    pub years: u64,
    pub expires: u64,
}

/// Register `name` in the empty slot `pos`, after `pred` (in slot
/// `pred_pos`), the leaf with the largest key below the name's.
pub fn register<N: Nodes + ?Sized>(n: &N, name: &str, set: &Set, pred_pos: u64, pred: Leaf, pos: u64) -> Result<Spend> {
    let k = key(name.as_bytes());
    if !(ord(&pred.key) < ord(&k) && ord(&k) < ord(&pred.next)) {
        bail!("{name}{SUFFIX} is taken");
    }
    build_register(n, name, set, pred_pos, pred, pos)
}

/// [`register`] without the order check, which the covenant makes again.
fn build_register<N: Nodes + ?Sized>(n: &N, name: &str, set: &Set, pred_pos: u64, pred: Leaf, pos: u64) -> Result<Spend> {
    let k = key(name.as_bytes());
    if pos >= CAPACITY || pos == pred_pos {
        bail!("slot {pos} is not free");
    }
    let p1 = proof(n, pred_pos)?;
    let mut m = Overlay { base: n, top: Mem::new() };
    for (l, i, h) in path(n, pred_pos, Some(Leaf { next: k, ..pred }.hash()))? {
        m.top.insert((l, i), h);
    }
    if node(&m, 0, pos)? != [0; 32] {
        bail!("slot {pos} is not free");
    }
    let p2 = proof(&m, pos)?;
    let new = Leaf {
        key: k,
        next: pred.next,
        expires: set.expires,
        tier: tier_for_len(name.len()),
        owner: set.owner,
        data: hash(&set.records),
        deal: hash(&set.terms),
    };
    let root = path(&m, pos, Some(new.hash()))?.last().unwrap().2;
    let items = vec![
        p2,
        p1,
        pred.tail(),
        pred.next.to_vec(),
        pred.key.to_vec(),
        set.years.to_le_bytes().to_vec(),
        set.expires.to_le_bytes().to_vec(),
        set.owner.to_vec(),
        set.records.clone(),
        set.terms.clone(),
        vec![0],
        name.as_bytes().to_vec(),
    ];
    Ok(Spend { items, root })
}

/// Change the name whose leaf `leaf` (with terms `oterms`) sits in slot `pos`.
pub fn change<N: Nodes + ?Sized>(n: &N, by: By, name: &str, pos: u64, leaf: Leaf, oterms: &[u8], set: &Set) -> Result<Spend> {
    if hash(oterms) != leaf.deal {
        bail!("those are not the name's terms");
    }
    let new = Leaf { owner: set.owner, expires: set.expires, data: hash(&set.records), deal: hash(&set.terms), ..leaf };
    let root = path(n, pos, Some(new.hash()))?.last().unwrap().2;
    let items = vec![
        proof(n, pos)?,
        Vec::new(),
        oterms.to_vec(),
        leaf.owner.to_vec(),
        leaf.data.to_vec(),
        vec![leaf.tier],
        leaf.expires.to_le_bytes().to_vec(),
        leaf.next.to_vec(),
        set.years.to_le_bytes().to_vec(),
        set.expires.to_le_bytes().to_vec(),
        set.owner.to_vec(),
        set.records.clone(),
        set.terms.clone(),
        vec![by.selector()],
        name.as_bytes().to_vec(),
    ];
    Ok(Spend { items, root })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::script::{execute_script, validate_structure, ExecContext};
    use crate::core::types::V6_ACTIVATION_HEIGHT;
    use crate::core::wots;

    const H: u64 = V6_ACTIVATION_HEIGHT + 1000;

    /// A registry in memory: the tree and its leaves (with their terms) by slot.
    #[derive(Clone)]
    struct Reg {
        tree: Mem,
        leaves: HashMap<u64, (Leaf, Vec<u8>)>,
    }

    impl Reg {
        fn genesis() -> Self {
            let mut r = Reg { tree: Mem::new(), leaves: HashMap::new() };
            for (pos, leaf) in genesis_leaves() {
                r.put(pos, Some((leaf, Deal::None.terms())));
            }
            r
        }
        fn put(&mut self, pos: u64, leaf: Option<(Leaf, Vec<u8>)>) {
            let p = path(&self.tree, pos, leaf.as_ref().map(|(l, _)| l.hash())).unwrap();
            write(&mut self.tree, &p);
            match leaf {
                Some(l) => self.leaves.insert(pos, l),
                None => self.leaves.remove(&pos),
            };
        }
        fn root(&self) -> [u8; 32] {
            root(&self.tree).unwrap()
        }
        fn pred(&self, name: &str) -> (u64, Leaf) {
            let k = ord(&key(name.as_bytes()));
            let (p, (l, _)) = self.leaves.iter().filter(|(_, (l, _))| ord(&l.key) < k).max_by_key(|(_, (l, _))| ord(&l.key)).unwrap();
            (*p, *l)
        }
        fn free(&self) -> u64 {
            (1..).find(|p| !self.leaves.contains_key(p)).unwrap()
        }
        fn slot(&self, name: &str) -> (u64, Leaf, Vec<u8>) {
            let k = key(name.as_bytes());
            let (p, (l, t)) = self.leaves.iter().find(|(_, (l, _))| l.key == k).unwrap();
            (*p, *l, t.clone())
        }
        fn register(&self, name: &str, set: &Set) -> Result<Spend> {
            let (pp, pl) = self.pred(name);
            register(&self.tree, name, set, pp, pl, self.free())
        }
        fn change(&self, by: By, name: &str, set: &Set) -> Spend {
            let (pos, leaf, oterms) = self.slot(name);
            change(&self.tree, by, name, pos, leaf, &oterms, set).unwrap()
        }
        fn apply(&mut self, items: &[Vec<u8>]) {
            let op = decode(items).expect("decodes");
            let terms = match &op {
                Op::Register { terms, .. } | Op::Change { terms, .. } => terms.clone(),
            };
            let undo = op.undoes();
            for (pos, leaf) in op.writes() {
                let keep = undo.iter().any(|(p, l)| *p == pos && l.is_some() && matches!(op, Op::Register { .. }));
                let t = if keep { self.leaves[&pos].1.clone() } else { terms.clone() };
                self.put(pos, leaf.map(|l| (l, t)));
            }
        }
    }

    /// Runs the covenant at `height` on `s`, with `more` outputs after the new thread.
    fn run_with(height: u64, input_root: [u8; 32], s: &Spend, more: &[OutputData], commitment: &[u8; 32]) -> bool {
        let mut outputs = vec![next_tip(s.root, &[9; 32])];
        outputs.extend_from_slice(more);
        let ctx = ExecContext {
            commitment,
            height,
            outputs: &outputs,
            input_value: 0,
            input_state: Some(input_root),
            this_address: address(),
            sum_input_value: 0,
        };
        execute_script(covenant(), &s.items, &ctx).is_ok()
    }

    fn burn(amount: u64) -> Vec<OutputData> {
        burn_outputs("x", amount)
    }

    fn pay(to: [u8; 32], amount: u64) -> Vec<OutputData> {
        decompose_value(amount).into_iter().map(|value| OutputData::Standard { address: to, value, salt: [value as u8; 32] }).collect()
    }

    use crate::core::types::decompose_value;

    fn set(owner: [u8; 32], years: u64, expires: u64) -> Set {
        Set { owner, records: encode(&[Rec::MdsKey { key: [3; 32], height: 10 }]).unwrap(), terms: Deal::None.terms(), years, expires }
    }

    /// A registry holding `name`, owned by a one-time key from `seed`, bought for a year at H.
    fn holding(name: &str, seed: [u8; 32]) -> Reg {
        let mut reg = Reg::genesis();
        let s = reg.register(name, &set(wots::keygen(&seed), 1, H + YEAR)).unwrap();
        assert!(run_with(H, reg.root(), &s, &burn(cost(tier_of(name), 1)), &[0; 32]));
        reg.apply(&s.items);
        reg
    }

    fn sign(seed: &[u8; 32], c: &[u8; 32]) -> Vec<u8> {
        wots::sig_to_bytes(&wots::sign(seed, c))
    }

    #[test]
    fn the_covenant_is_valid_and_fits() {
        validate_structure(covenant(), H).unwrap();
        assert!(validate_structure(covenant(), V6_ACTIVATION_HEIGHT - 1).is_err(), "needs v6 opcodes");
        assert!(covenant().len() <= crate::core::script::MAX_SCRIPT_SIZE, "{} bytes", covenant().len());
        assert_eq!(covenant().len(), 987);
        assert_eq!(address(), Predicate::Script { bytecode: covenant().to_vec() }.address());
        assert_eq!(genesis_root(), Reg::genesis().root());
    }

    /// Clients outside this crate (midwimble's resolver) pin the registry's
    /// address and genesis root to check proofs against: they must not move.
    #[test]
    fn the_registry_address_and_genesis_root_are_pinned() {
        assert_eq!(hex::encode(address()), "c71ab5f9589e9ef822e950cce14fa0e40952fc7103d6358b13427ed96b0b742e");
        assert_eq!(hex::encode(genesis_root()), "4204a34119561ff0f7ef7c54b3e1879f3d689545921ee6f5a6827431ea40a0cd");
    }

    #[test]
    fn premium_names_are_canonical_unique_and_dearer_than_their_length() {
        let p = premium();
        assert!(p.len() > 200);
        let mut seen = std::collections::HashSet::new();
        for (n, t) in p {
            assert_eq!(normalize(n).unwrap(), *n);
            assert!(seen.insert(n.clone()), "{n} twice");
            assert!(*t < tier_for_len(n.len()), "{n} is no dearer listed");
        }
        assert_eq!((tier_of("bitcoin"), tier_of("nick"), tier_of("ab"), tier_of("abc"), tier_of("nicholas-x")), (0, 2, 0, 1, 3));
        for (n, _) in RESERVED {
            assert_eq!(normalize(n).unwrap(), n);
            assert!(reserved().iter().any(|(r, _)| r == n), "{n}'s key is not 64 hex characters");
            assert!(premium().iter().all(|(p, _)| p != n), "{n} is reserved and listed both");
        }
        let c = seeded().iter().find(|s| s.name == "ciphernom").unwrap();
        assert_eq!((c.owner[0], c.expires, c.tier), (0x32, RESERVED_UNTIL, 3));
    }

    #[test]
    fn names_normalize_to_one_form() {
        assert_eq!(normalize(" Nick.MDS ").unwrap(), "nick");
        for bad in ["", ".mds", "-a", "a-", "a b", "ni\u{0441}k", "a.b", &"x".repeat(33)] {
            assert!(normalize(bad).is_err(), "{bad:?}");
        }
        assert!(is_name("nick.mds") && !is_name("nick"));
    }

    #[test]
    fn records_and_terms_round_trip() {
        let all = vec![
            Rec::MdsKey { key: [1; 32], height: 10 },
            Rec::MdsAddress([2; 32]),
            Rec::Mdw("mw1abc".into()),
            Rec::Evm([4; 20]),
            Rec::Identity([5; 32]),
            Rec::Text("héllo".into()),
            Rec::Other { tag: 7, value: vec![6; 32] },
            Rec::Kem([8; 32]),
            Rec::Script(vec![1, 2, 3]),
            Rec::Other { tag: 99, value: vec![7, 7] },
        ];
        assert_eq!(parse(&encode(&all).unwrap()).unwrap(), all);
        // Wrong lengths stay as they are, kept but not read.
        assert_eq!(parse(&[tag::KEM, 2, 1, 2]).unwrap(), vec![Rec::Other { tag: tag::KEM, value: vec![1, 2] }]);
        assert_eq!(parse(&[tag::SCRIPT, 0]).unwrap(), vec![Rec::Other { tag: tag::SCRIPT, value: vec![] }]);
        assert_eq!(parse(&[tag::EVM, 3, 1, 2]), None);
        assert_eq!(pays(&all, |_| 0), Some(compute_address(&[1; 32])));
        assert_eq!(pays(&all, |_| 1024 - SIGNATURES_KEPT + 1), Some([2; 32]));
        for d in [
            Deal::None,
            Deal::Sale { price: 5, payee: [1; 32] },
            Deal::Rent { price: 6, payee: [2; 32], span: 7 },
            Deal::Rented { price: 6, payee: [2; 32], until: 9 },
        ] {
            assert_eq!(Deal::parse(&d.terms()), d);
        }
        assert_eq!(Deal::parse(&[]), Deal::None);
    }

    #[test]
    fn a_registration_burns_its_tier_for_its_years() {
        let reg = Reg::genesis();
        let r = reg.root();
        for (name, years) in [("nicholas-x", 2), ("ab", 1), ("abc", 3), ("nick", 1)] {
            let price = cost(tier_of(name), years);
            let s = reg.register(name, &set([1; 32], years, H + years * YEAR)).unwrap();
            assert!(run_with(H, r, &s, &burn(price), &[0; 32]), "{name}");
            assert!(!run_with(H, r, &s, &burn(price - 1), &[0; 32]), "{name} underpaid");
            assert!(crate::core::mss::mss_sigs_in_stack(&s.items).is_empty());
        }
        let over = reg.register("nicholas-x", &set([1; 32], 1, H + YEAR + 1)).unwrap();
        assert!(!run_with(H, r, &over, &burn(RATES[0]), &[0; 32]), "held past what it paid for");
        let none = reg.register("nicholas-x", &set([1; 32], 0, H)).unwrap();
        assert!(!run_with(H, r, &none, &[], &[0; 32]), "a year at least");
        let long = reg.register("nicholas-x", &set([1; 32], 11, H + 11 * YEAR)).unwrap();
        assert!(!run_with(H, r, &long, &burn(cost(3, 11)), &[0; 32]), "ten years at most");
    }

    #[test]
    fn registrations_chain_and_the_mirror_follows() {
        let mut reg = Reg::genesis();
        for (i, name) in ["nick", "alice", "zed", "a", "nicholas-x"].iter().enumerate() {
            let before = reg.root();
            let s = reg.register(name, &set([i as u8 + 1; 32], 1, H + YEAR)).unwrap();
            assert!(run_with(H, before, &s, &burn(cost(tier_of(name), 1)), &[0; 32]), "{name}");
            reg.apply(&s.items);
            assert_eq!(reg.root(), s.root, "the mirror's tree is the covenant's");
            let mut undo = reg.clone();
            for (pos, leaf) in decode(&s.items).unwrap().undoes() {
                undo.put(pos, leaf.map(|l| (l, Deal::None.terms())));
            }
            assert_eq!(undo.root(), before);
        }
    }

    #[test]
    fn a_name_exists_once_and_premium_names_only_by_reclaim() {
        let reg = holding("nick", [1; 32]);
        assert!(reg.register("nick", &set([2; 32], 1, H + YEAR)).is_err());
        assert!(reg.register("bitcoin", &set([2; 32], 1, H + YEAR)).is_err(), "listed at genesis");
        for (pp, (pl, _)) in reg.leaves.clone() {
            for name in ["nick", "bitcoin"] {
                let f = build_register(&reg.tree, name, &set([2; 32], 1, H + YEAR), pp, pl, reg.free()).unwrap();
                assert!(!run_with(H, reg.root(), &f, &burn(RATES[0]), &[0; 32]), "{name} after slot {pp}");
            }
        }
        let g = Reg::genesis();
        let claim = g.change(By::Lapse, "bitcoin", &set([2; 32], 1, H + YEAR));
        assert!(run_with(H, g.root(), &claim, &burn(RATES[0]), &[0; 32]));
        assert!(!run_with(H, g.root(), &claim, &burn(RATES[1]), &[0; 32]), "at its tier's price");
        let mut after = g.clone();
        after.apply(&claim.items);
        assert_eq!(after.root(), claim.root);
    }

    #[test]
    fn only_the_owner_updates_and_years_added_burn() {
        let seed = [42u8; 32];
        let reg = holding("nicholas-x", seed);
        let c = [6u8; 32];
        let (_, leaf, _) = reg.slot("nicholas-x");
        let mut free = reg.change(By::Owner, "nicholas-x", &Set { years: 0, ..set([8; 32], 0, leaf.expires) });
        free.items[1] = sign(&seed, &c);
        assert!(run_with(H + 5, reg.root(), &free, &[], &c), "records and owner change free");
        let mut other = free.items.clone();
        other[1] = sign(&[1; 32], &c);
        assert!(!run_with(H + 5, reg.root(), &Spend { items: other, root: free.root }, &[], &c), "not the owner");
        assert!(!run_with(H + 5, reg.root(), &free, &[], &[5; 32]), "signed for another transaction");
        let mut renew = reg.change(By::Owner, "nicholas-x", &set([8; 32], 2, leaf.expires + 2 * YEAR));
        renew.items[1] = sign(&seed, &c);
        assert!(run_with(H + 5, reg.root(), &renew, &burn(cost(3, 2)), &[0; 32].map(|_| 6)));
        assert!(!run_with(H + 5, reg.root(), &renew, &burn(cost(3, 2) - 1), &c), "years added burn");
        let mut stretch = reg.change(By::Owner, "nicholas-x", &set([8; 32], 0, leaf.expires + 1));
        stretch.items[1] = sign(&seed, &c);
        assert!(!run_with(H + 5, reg.root(), &stretch, &[], &c), "no time for free");
    }

    #[test]
    fn a_lapsed_name_is_anyones_at_its_price() {
        let reg = holding("nicholas-x", [42; 32]);
        let (_, leaf, _) = reg.slot("nicholas-x");
        let later = leaf.expires + 1;
        let s = reg.change(By::Lapse, "nicholas-x", &set([9; 32], 1, later + YEAR));
        assert!(!run_with(leaf.expires, reg.root(), &s, &burn(RATES[3]), &[0; 32]), "still held at its last block");
        assert!(run_with(later, reg.root(), &s, &burn(RATES[3]), &[0; 32]));
        let mut as_owner = s.items.clone();
        as_owner[13] = vec![1];
        assert!(!run_with(later, reg.root(), &Spend { items: as_owner, root: s.root }, &burn(RATES[3]), &[0; 32]));
    }

    /// `name` held by `seed`'s key and offered as `deal`.
    fn offering(name: &str, seed: [u8; 32], deal: Deal) -> Reg {
        let mut reg = holding(name, seed);
        let (_, leaf, _) = reg.slot(name);
        let c = [7u8; 32];
        let mut s = reg.change(By::Owner, name, &Set { terms: deal.terms(), ..set(leaf.owner, 0, leaf.expires) });
        s.items[1] = sign(&seed, &c);
        assert!(run_with(H + 1, reg.root(), &s, &[], &c));
        reg.apply(&s.items);
        reg
    }

    #[test]
    fn a_name_for_sale_goes_to_whoever_pays() {
        let payee = [5u8; 32];
        let reg = offering("nicholas-x", [42; 32], Deal::Sale { price: 1000, payee });
        let (_, leaf, _) = reg.slot("nicholas-x");
        let buy = reg.change(By::Buy, "nicholas-x", &set([9; 32], 0, leaf.expires));
        assert!(run_with(H + 2, reg.root(), &buy, &pay(payee, 1000), &[0; 32]));
        assert!(!run_with(H + 2, reg.root(), &buy, &pay(payee, 999), &[0; 32]), "underpaid");
        assert!(!run_with(H + 2, reg.root(), &buy, &pay([6; 32], 1000), &[0; 32]), "paid someone else");
        let mut after = reg.clone();
        after.apply(&buy.items);
        let (_, bought, _) = after.slot("nicholas-x");
        assert_eq!((bought.owner, bought.deal), ([9; 32], Deal::None.hash()));
        // Not for sale: no one buys it.
        let held = holding("nicholas-x", [42; 32]);
        let nope = held.change(By::Buy, "nicholas-x", &set([9; 32], 0, leaf.expires));
        assert!(!run_with(H + 2, held.root(), &nope, &pay(payee, 1000), &[0; 32]));
    }

    #[test]
    fn a_rental_gives_the_records_and_locks_the_owner_out_until_it_ends() {
        let seed = [42u8; 32];
        let payee = [5u8; 32];
        let span = 1000;
        let reg = offering("nicholas-x", seed, Deal::Rent { price: 300, payee, span });
        let (_, leaf, _) = reg.slot("nicholas-x");
        let until = H + 2 + span;
        let rented = Deal::Rented { price: 300, payee, until };
        let renter = Set { records: encode(&[Rec::MdsAddress([7; 32])]).unwrap(), terms: rented.terms(), ..set(leaf.owner, 0, leaf.expires) };
        let take = reg.change(By::Rent, "nicholas-x", &renter);
        assert!(run_with(H + 2, reg.root(), &take, &pay(payee, 300), &[0; 32]));
        assert!(!run_with(H + 2, reg.root(), &take, &pay(payee, 299), &[0; 32]), "underpaid");
        assert!(!run_with(H + 1, reg.root(), &take, &pay(payee, 300), &[0; 32]), "longer than offered");
        let thief = reg.change(By::Rent, "nicholas-x", &Set { owner: [9; 32], ..renter.clone() });
        assert!(!run_with(H + 2, reg.root(), &thief, &pay(payee, 300), &[0; 32]), "the owner stays");
        let cheap = Deal::Rented { price: 1, payee, until };
        let cheap = reg.change(By::Rent, "nicholas-x", &Set { terms: cheap.terms(), ..renter.clone() });
        assert!(!run_with(H + 2, reg.root(), &cheap, &pay(payee, 300), &[0; 32]), "the offer's price");
        let late = Deal::Rented { price: 300, payee, until: leaf.expires - GRACE + 1 };
        let late = reg.change(By::Rent, "nicholas-x", &Set { terms: late.terms(), ..renter.clone() });
        assert!(!run_with(leaf.expires, reg.root(), &late, &pay(payee, 300), &[0; 32]), "ends a grace before it lapses");

        // The old leaf's bytes split differently, shifting its owner: refused.
        let b = leaf.bytes();
        let mut shifted = take.items.clone();
        shifted[6] = b[16..23].to_vec();
        shifted[5] = b[23..24].to_vec();
        shifted[3] = b[24..56].to_vec();
        shifted[4] = b[56..89].to_vec();
        shifted[10] = b[24..56].to_vec();
        assert_eq!(hash(&[&b[..16], &shifted[6][..], &shifted[5][..], &shifted[3][..], &shifted[4][..], &b[89..]].concat()), leaf.hash());
        assert!(!run_with(H + 2, reg.root(), &Spend { items: shifted, root: take.root }, &pay(payee, 300), &[0; 32]));

        let mut during = reg.clone();
        during.apply(&take.items);
        let c = [8u8; 32];
        let (_, now, _) = during.slot("nicholas-x");
        let mut back = during.change(By::Owner, "nicholas-x", &set(leaf.owner, 0, now.expires));
        back.items[1] = sign(&seed, &c);
        assert!(!run_with(until - 1, during.root(), &back, &[], &c), "locked out until it ends");
        assert!(run_with(until, during.root(), &back, &[], &c));
    }

    #[test]
    fn a_reserved_name_is_its_keys_until_its_end() {
        let seed_key = [77u8; 32];
        let pk = wots::keygen(&seed_key);
        let mut reg = Reg { tree: Mem::new(), leaves: HashMap::new() };
        for (pos, leaf) in leaves_of(&seed(&[("ciphernom".into(), pk)])) {
            reg.put(pos, Some((leaf, Deal::None.terms())));
        }
        let (_, leaf, _) = reg.slot("ciphernom");
        assert_eq!((leaf.owner, leaf.expires, leaf.tier), (pk, RESERVED_UNTIL, 3));
        assert!(reg.register("ciphernom", &set([2; 32], 1, H + YEAR)).is_err(), "not free to register");
        let take = reg.change(By::Lapse, "ciphernom", &set([2; 32], 1, H + YEAR));
        assert!(!run_with(H, reg.root(), &take, &burn(RATES[0]), &[0; 32]), "not anyone's before its end");
        let c = [3u8; 32];
        let mut own = reg.change(By::Owner, "ciphernom", &set([8; 32], 0, RESERVED_UNTIL));
        own.items[1] = sign(&seed_key, &c);
        assert!(run_with(H, reg.root(), &own, &[], &c), "its key takes it, free");
    }

    #[test]
    fn records_say_what_kind_of_address_a_name_pays() {
        use crate::core::address_kind::AddressKind;
        use crate::core::script::compile_multisig_2of3;
        let ms = compile_multisig_2of3(&[1; 32], &[2; 32], &[3; 32]);
        let ms_addr = hash(&ms);
        let mut recs = vec![Rec::MdsAddress(ms_addr), Rec::MdsKey { key: [4; 32], height: 10 }];
        assert_eq!(address_kind(&recs, &ms_addr), None, "nothing published: the records do not say");
        assert_eq!(address_kind(&recs, &compute_address(&[4; 32])), Some(AddressKind::Key));
        recs.extend(script_records(&ms));
        assert_eq!(address_kind(&recs, &ms_addr), Some(AddressKind::Multisig { m: 2, n: 3 }));
        assert_eq!(address_kind(&recs, &[9; 32]), None, "a script that is not this address's");
        // A plan's script runs past one record; its parts join back up.
        let heirs = (0..6u8).map(|i| crate::core::plans::Heir { pk: [10 + i; 32], share: if i == 0 { 65_535 - 5 * 10_000 } else { 10_000 } }).collect();
        let plan = crate::core::plans::InheritancePlan { owner_pk: [9; 32], heirs, start: 0, period: 52_560, secret: [1; 32] };
        let script = plan.script(1);
        assert!(script.len() > 255);
        let recs: Vec<Rec> = [vec![Rec::MdsAddress(plan.address(1))], script_records(&script)].concat();
        let blob = encode(&recs).unwrap();
        assert_eq!(address_kind(&parse(&blob).unwrap(), &plan.address(1)), Some(AddressKind::Plan { heirs: 6 }));
        assert_eq!(pays(&recs, |_| 0), Some(plan.address(1)), "paid as it is");
    }

    #[test]
    fn decode_refuses_other_shapes() {
        let reg = Reg::genesis();
        let s = reg.register("nick", &set([1; 32], 1, H + YEAR)).unwrap();
        assert!(decode(&s.items).is_some());
        assert!(decode(&s.items[1..]).is_none());
        let mut sel = s.items.clone();
        sel[10] = vec![1];
        assert!(decode(&sel).is_none());
        let c = reg.change(By::Lapse, "bitcoin", &set([2; 32], 1, H + YEAR));
        let mut five = c.items.clone();
        five[13] = vec![5];
        assert!(decode(&five).is_none());
    }
}
