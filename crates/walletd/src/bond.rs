//! Midwimble mining bonds, from the desktop wallet.
//!
//! A bond is one midstate coin locked by `script::compile_mining_bond`:
//!
//! ```text
//! PUSH_DATA <mining key>    DROP              binds the key into the address
//! PUSH_INT  <bonded_until>  CHECKTIMEVERIFY   unspendable below that height
//! PUSH_DATA <owner key>     CHECKSIGVERIFY    only the owner can spend it
//! PUSH_INT  1
//! ```
//!
//! To midstate it is an ordinary pay-to-script-hash coin. Midwimble registers
//! it (`midwimble bond register`) and lets the mining key sign its blocks while
//! enough of the lock remains (midwimble `docs/BONDED_POW.md`). The CLI does
//! this with `wallet generate-mss`, `wallet send` and `wallet spend-script`;
//! this module is the same for the desktop wallet.
//!
//! # The owner key
//!
//! Every bond gets its own MSS key, derived from the recovery phrase at the HD
//! index [`owner_index`]: a hash of the bond's mining key and lock height, in a
//! range the wallet's own trees never reach. Nothing about it has to be saved.
//! Given those two values (midwimble's bond file holds both) the phrase gives
//! the key back, so a bond can be unbonded after a restore. The tree is small
//! ([`OWNER_HEIGHT`]): it signs once to unbond, plus any retries. The web
//! wallet derives the same key, so either wallet can unbond either's bonds.
//!
//! # Records
//!
//! A bond's salt exists nowhere but the funding transaction, so the wallet
//! keeps one [`BondRecord`] per bond in `<wallet>.bonds`, written before the
//! funding commit is broadcast.

use anyhow::{anyhow, bail, Context, Result};
use midstate::core::mss::{self, MssKeypair};
use midstate::core::script;
use midstate::core::types::{
    compute_coin_id, hash, InputReveal, OutputData, Predicate, Transaction, Witness,
};
use midstate::core::wots;
use midstate::wallet::hd;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Height of a bond's owner tree: 16 signatures, for one unbond and its retries.
pub const OWNER_HEIGHT: u32 = 4;
/// Owner keys live at HD indices `OWNER_INDEX_BASE | (hash & OWNER_INDEX_MASK)`:
/// above every index the wallet's own trees use, and still a `u32`, which is
/// what the web wallet's `get_mss_address` takes.
pub const OWNER_INDEX_BASE: u32 = 0x8000_0000;
const OWNER_INDEX_MASK: u32 = 0x3FFF_FFFF;
const OWNER_DOMAIN: &[u8] = b"midstate/bond-owner/v1";

/// The mempool's admission floor (`mempool.rs`, not public): a reveal needs
/// `fee * 1024 >= MIN_FEE_PER_KB * size`.
const MIN_FEE_PER_KB: u64 = 10;
/// Slack above that floor, as for swap sweeps.
const FEE_MARGIN: u64 = 20;

/// Midwimble's launch parameters (midwimble `core/types.rs` and
/// `core/bond.rs`). Midstate consensus knows nothing of them, so the wallet
/// uses them for guidance, never to refuse a bond.
pub mod midwimble {
    /// Smallest bond midwimble counts: 2^24 units (16 mMDS).
    pub const MIN_MINING_BOND: u64 = 1 << 24;
    /// A bond counts only while at least this many blocks of its lock remain
    /// by midwimble's clock: two days.
    pub const MIN_REMAINING_BOND_LOCK: u64 = 2 * 24 * 60;
    /// Midwimble's clock deliberately runs a week ahead of midstate's height.
    pub const CLOCK_MARGIN: u64 = 7 * 24 * 60;
    /// Midwimble refuses to register a bond locked further ahead than a year.
    pub const MAX_BOND_HORIZON: u64 = 365 * 24 * 60;
    /// A registration needs a day of midstate work buried above its proof.
    pub const REGISTRATION_DELAY_BLOCKS: u64 = 24 * 60;

    /// Midwimble's estimate of midstate's height at `timestamp`
    /// (`bond::est_midstate_height`): midstate's schedule from its genesis,
    /// plus the clock margin.
    pub fn est_midstate_height(timestamp: u64) -> u64 {
        timestamp.saturating_sub(midstate::core::types::BITCOIN_BLOCK_TIME) / 60 + CLOCK_MARGIN
    }

    /// The last unix time at which midwimble still counts a bond locked until
    /// `until`, or `None` if it never would.
    pub fn counts_until(until: u64) -> Option<u64> {
        let h = until.checked_sub(MIN_REMAINING_BOND_LOCK + CLOCK_MARGIN)?;
        Some(midstate::core::types::BITCOIN_BLOCK_TIME + h * 60 + 59)
    }

    /// The highest lock height midwimble would register at `now`.
    pub fn horizon(now: u64) -> u64 {
        est_midstate_height(now) + MAX_BOND_HORIZON
    }

    /// The lock height that keeps a bond counted for `days` after it
    /// registers (a day from now), by midwimble's clock.
    pub fn until_for_days(now: u64, days: u64) -> u64 {
        est_midstate_height(now) + MIN_REMAINING_BOND_LOCK + REGISTRATION_DELAY_BLOCKS + days * 24 * 60
    }
}

// ── Owner key ────────────────────────────────────────────────────────────

/// HD index of the owner key for a bond on `mining_key` locked until `until`.
pub fn owner_index(mining_key: &[u8; 32], until: u64) -> u32 {
    let mut buf = Vec::with_capacity(OWNER_DOMAIN.len() + 40);
    buf.extend_from_slice(OWNER_DOMAIN);
    buf.extend_from_slice(mining_key);
    buf.extend_from_slice(&until.to_le_bytes());
    let h = hash(&buf);
    OWNER_INDEX_BASE | (u32::from_le_bytes([h[0], h[1], h[2], h[3]]) & OWNER_INDEX_MASK)
}

/// The owner tree for that bond: a full keygen, so call it off the actor.
pub fn derive_owner(master_seed: &[u8; 32], mining_key: &[u8; 32], until: u64) -> Result<MssKeypair> {
    let seed = hd::derive_mss_seed(master_seed, owner_index(mining_key, until) as u64);
    mss::keygen(&seed, OWNER_HEIGHT)
}

// ── The script ───────────────────────────────────────────────────────────

/// What a bond script commits to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BondTerms {
    pub mining_key: [u8; 32],
    pub bonded_until: u64,
    pub owner_pk: [u8; 32],
}

impl BondTerms {
    pub fn script(&self) -> Vec<u8> {
        script::compile_mining_bond(&self.mining_key, self.bonded_until, &self.owner_pk)
    }
    pub fn address(&self) -> [u8; 32] {
        hash(&self.script())
    }
}

/// Read a bond script back. Only the exact encoding `compile_mining_bond`
/// produces is a bond: midwimble refuses anything else, a padded integer
/// included, so this does too.
pub fn parse_script(bytecode: &[u8]) -> Option<BondTerms> {
    let push32 = |at: usize| -> Option<[u8; 32]> {
        if bytecode.get(at) != Some(&script::OP_PUSH_DATA) || bytecode.get(at + 1..at + 3)? != [32, 0] {
            return None;
        }
        bytecode.get(at + 3..at + 35)?.try_into().ok()
    };
    let mining_key = push32(0)?;
    // PUSH_DATA <len:u16 le> <minimal little-endian height>
    if bytecode.get(35) != Some(&script::OP_DROP) || bytecode.get(36) != Some(&script::OP_PUSH_DATA) {
        return None;
    }
    let len = *bytecode.get(37)? as usize;
    if bytecode.get(38) != Some(&0) || !(1..=8).contains(&len) {
        return None;
    }
    let mut le = [0u8; 8];
    le[..len].copy_from_slice(bytecode.get(39..39 + len)?);
    let at = 39 + len;
    if bytecode.get(at) != Some(&script::OP_CHECKTIMEVERIFY) {
        return None;
    }
    let owner_pk = push32(at + 1)?;
    let terms = BondTerms { mining_key, bonded_until: u64::from_le_bytes(le), owner_pk };
    (terms.script() == bytecode).then_some(terms)
}

// ── Fees ─────────────────────────────────────────────────────────────────

/// Serialized length of an MSS signature from a tree of `height`
/// (`MssSignature::to_bytes`).
pub fn mss_sig_len(height: u32) -> usize {
    8 + 32 + wots::SIG_SIZE + 4 + 32 * height as usize
}

/// Exact serialized size of an unbond reveal: one bond input signed by an
/// owner signature of `sig_len` bytes, and `n_outputs` standard outputs. The
/// mempool prices reveals by `bincode::serialized_size`, so measure the same
/// way rather than estimate.
pub fn unbond_reveal_size(script_len: usize, sig_len: usize, n_outputs: usize) -> u64 {
    let tx = Transaction::Reveal {
        inputs: vec![InputReveal {
            predicate: Predicate::Script { bytecode: vec![0u8; script_len] },
            value: 0,
            salt: [0u8; 32],
            commitment: None,
        }],
        witnesses: vec![Witness::sig(vec![0u8; sig_len])],
        outputs: (0..n_outputs)
            .map(|_| OutputData::Standard { address: [0u8; 32], value: 0, salt: [0u8; 32] })
            .collect(),
        salt: [0u8; 32],
    };
    bincode::serialized_size(&tx).unwrap_or(u64::MAX)
}

/// The fee for an unbond that pays the bond's value back, less the fee, and
/// the power-of-two outputs that leaves. The fee depends on the output count,
/// which depends on the fee: iterate to the fixed point, as
/// `swap::resolve_sweep_fee` does. `None` if the bond cannot cover its fee.
pub fn resolve_unbond_fee(script_len: usize, sig_len: usize, value: u64) -> Option<(u64, Vec<u64>)> {
    let mut n_out = 1usize;
    loop {
        let size = unbond_reveal_size(script_len, sig_len, n_out);
        let fee = size.saturating_mul(MIN_FEE_PER_KB).div_ceil(1024) + FEE_MARGIN;
        if fee >= value {
            return None;
        }
        let denoms = midstate::core::decompose_value(value - fee);
        if denoms.len() <= n_out {
            return Some((fee, denoms));
        }
        n_out = denoms.len();
    }
}

/// The unbond reveal: the bond coin, opened by the owner's signature over
/// the commitment of exactly these outputs and salt.
pub fn unbond_reveal(rec: &BondRecord, owner_sig: Vec<u8>, outputs: Vec<OutputData>, salt: [u8; 32]) -> Transaction {
    Transaction::Reveal {
        inputs: vec![InputReveal {
            predicate: Predicate::Script { bytecode: rec.script() },
            value: rec.value,
            salt: rec.salt,
            commitment: None,
        }],
        witnesses: vec![Witness::sig(owner_sig)],
        outputs,
        salt,
    }
}

// ── Records ──────────────────────────────────────────────────────────────

/// Which key signs the unbond.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OwnerKind {
    /// The bond's own key, from [`derive_owner`].
    #[default]
    Derived,
    /// One of the wallet's MSS trees (a bond made with `wallet generate-mss`
    /// and imported here).
    WalletTree,
}

/// One bond this wallet made or imported.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BondRecord {
    #[serde(with = "hex32")]
    pub mining_key: [u8; 32],
    pub bonded_until: u64,
    #[serde(with = "hex32")]
    pub owner_pk: [u8; 32],
    #[serde(default)]
    pub owner: OwnerKind,
    pub value: u64,
    #[serde(with = "hex32")]
    pub salt: [u8; 32],
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub created_at: u64,
    /// The send that locked the coin, while it may still be pending.
    #[serde(default, with = "opt_hex32")]
    pub fund_commitment: Option<[u8; 32]>,
    /// The spend that unbonds it, once started.
    #[serde(default, with = "opt_hex32")]
    pub unbond_commitment: Option<[u8; 32]>,
    /// The owner's signature over `unbond_commitment`, kept so a retried
    /// reveal sends the same signature instead of spending another leaf.
    #[serde(default)]
    pub unbond_sig: Option<String>,
    /// First owner-tree leaf never used, by this wallet's own record. The
    /// chain's record (`MSS_LEAF_INDEX_TABLE`) is consulted too; the higher wins.
    #[serde(default)]
    pub owner_next_leaf: u64,
    /// The coin has been seen unspent on chain at least once.
    #[serde(default)]
    pub seen_live: bool,
}

impl BondRecord {
    pub fn terms(&self) -> BondTerms {
        BondTerms { mining_key: self.mining_key, bonded_until: self.bonded_until, owner_pk: self.owner_pk }
    }
    pub fn script(&self) -> Vec<u8> {
        self.terms().script()
    }
    pub fn address(&self) -> [u8; 32] {
        self.terms().address()
    }
    pub fn coin_id(&self) -> [u8; 32] {
        compute_coin_id(&self.address(), self.value, &self.salt)
    }

    /// `midwimble bond register` with every value this bond needs. It is run
    /// twice: once to take the proof, again a day later to finish it.
    pub fn register_command(&self, midstate_rpc: &str) -> String {
        format!(
            "midwimble bond register --file bond.json --owner-pk {} --until {} --value {} --salt {} --midstate-rpc {}",
            hex::encode(self.owner_pk),
            self.bonded_until,
            self.value,
            hex::encode(self.salt),
            midstate_rpc
        )
    }

    /// `<script hex> <value> <salt hex>`: what [`parse_import`] reads, so a
    /// bond can be carried to another wallet holding the same phrase.
    pub fn export_line(&self) -> String {
        format!("{} {} {}", hex::encode(self.script()), self.value, hex::encode(self.salt))
    }
}

/// Every bond this wallet knows, persisted beside the wallet file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BondBook {
    pub bonds: Vec<BondRecord>,
}

impl BondBook {
    pub fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("bonds")
    }

    /// Missing file: no bonds. An unreadable one is set aside, never
    /// overwritten: it may hold the only copy of a bond's salt.
    pub fn load(wallet_path: &Path) -> (Self, Option<String>) {
        let path = Self::path(wallet_path);
        let Ok(raw) = std::fs::read(&path) else { return (Self::default(), None) };
        match serde_json::from_slice(&raw) {
            Ok(book) => (book, None),
            Err(e) => {
                let aside = path.with_extension(format!("bonds.unreadable-{}", crate::bond::now_secs()));
                let moved = std::fs::rename(&path, &aside).is_ok();
                let text = if moved {
                    format!(
                        "The bond records could not be read ({e}); the file was kept as {}. \
                         Import your bonds again from midwimble's bond files.",
                        aside.display()
                    )
                } else {
                    format!("The bond records at {} could not be read: {e}", path.display())
                };
                (Self::default(), Some(text))
            }
        }
    }

    /// Write through a temporary file, so a crash mid-write cannot lose the
    /// records already there.
    pub fn save(&self, wallet_path: &Path) -> Result<()> {
        let path = Self::path(wallet_path);
        let tmp = path.with_extension("bonds.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?).context("writing bond records")?;
        std::fs::rename(&tmp, &path).context("saving bond records")?;
        Ok(())
    }

    pub fn find(&self, coin_id: &[u8; 32]) -> Option<&BondRecord> {
        self.bonds.iter().find(|b| &b.coin_id() == coin_id)
    }
    pub fn find_mut(&mut self, coin_id: &[u8; 32]) -> Option<&mut BondRecord> {
        self.bonds.iter_mut().find(|b| &b.coin_id() == coin_id)
    }
    pub fn by_unbond(&self, commitment: &[u8; 32]) -> Option<&BondRecord> {
        self.bonds.iter().find(|b| b.unbond_commitment.as_ref() == Some(commitment))
    }
    pub fn by_unbond_mut(&mut self, commitment: &[u8; 32]) -> Option<&mut BondRecord> {
        self.bonds.iter_mut().find(|b| b.unbond_commitment.as_ref() == Some(commitment))
    }
}

/// A block count as time at the 60-second target: "40 minutes", "5 hours",
/// "12 days".
pub fn blocks_human(blocks: u64) -> String {
    secs_human(blocks.saturating_mul(60))
}

/// A duration in the largest unit that reads naturally.
pub fn secs_human(secs: u64) -> String {
    let hours = secs / 3600;
    if hours < 1 {
        format!("{} minutes", (secs / 60).max(1))
    } else if hours < 48 {
        format!("{hours} hours")
    } else {
        format!("{} days", hours / 24)
    }
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Import ───────────────────────────────────────────────────────────────

/// A bond read from text: its terms, value and salt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBond {
    pub terms: BondTerms,
    pub value: u64,
    pub salt: [u8; 32],
}

/// Read a bond from any of:
///
/// - a midwimble bond file (`bond.json`, once `bond register` has filled in
///   its `coin`), or just its `coin` object;
/// - `<script hex> <value> <salt hex>`, as [`BondRecord::export_line`] writes
///   (`midwimble bond address` prints the script; the send prints the salt).
pub fn parse_import(text: &str) -> Result<ImportedBond> {
    let t = text.trim();
    if t.starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(t).context("that is not valid JSON")?;
        let coin = v.get("coin").unwrap_or(&v);
        if coin.is_null() {
            bail!("this bond file has no coin yet: run `midwimble bond register` on it first");
        }
        let script = coin.get("script").ok_or_else(|| anyhow!("no \"script\" in the bond"))?;
        let terms = BondTerms {
            mining_key: json32(script.get("mining_key"), "mining_key")?,
            bonded_until: script
                .get("bonded_until")
                .and_then(|x| x.as_u64())
                .ok_or_else(|| anyhow!("no \"bonded_until\" in the bond"))?,
            owner_pk: json32(script.get("owner_pk"), "owner_pk")?,
        };
        let value = coin.get("value").and_then(|x| x.as_u64()).ok_or_else(|| anyhow!("no \"value\" in the bond"))?;
        let salt = json32(coin.get("salt"), "salt")?;
        return Ok(ImportedBond { terms, value, salt });
    }
    let parts: Vec<&str> = t.split(|c: char| c.is_whitespace() || c == ',' || c == ':').filter(|s| !s.is_empty()).collect();
    let [script_hex, value, salt_hex] = parts.as_slice() else {
        bail!("paste midwimble's bond.json, or `<script hex> <value> <salt hex>`");
    };
    let bytecode = hex::decode(script_hex).context("the script is not hex")?;
    let terms = parse_script(&bytecode).ok_or_else(|| anyhow!("that script is not a mining bond"))?;
    let value: u64 = value.parse().context("the value is not a whole number of units")?;
    let salt: [u8; 32] = hex::decode(salt_hex)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| anyhow!("the salt must be 32 bytes of hex"))?;
    Ok(ImportedBond { terms, value, salt })
}

/// A 32-byte field as midwimble writes it (an array of numbers) or as hex.
fn json32(v: Option<&serde_json::Value>, what: &str) -> Result<[u8; 32]> {
    let v = v.ok_or_else(|| anyhow!("no \"{what}\" in the bond"))?;
    let bytes: Vec<u8> = if let Some(s) = v.as_str() {
        hex::decode(s).with_context(|| format!("\"{what}\" is not hex"))?
    } else if let Some(a) = v.as_array() {
        a.iter()
            .map(|x| x.as_u64().filter(|n| *n <= 255).map(|n| n as u8))
            .collect::<Option<Vec<u8>>>()
            .ok_or_else(|| anyhow!("\"{what}\" is not a list of bytes"))?
    } else {
        bail!("\"{what}\" is neither hex nor a list of bytes");
    };
    bytes.try_into().map_err(|_| anyhow!("\"{what}\" must be 32 bytes"))
}

/// A mining key typed by hand: 32 bytes of hex that decode to a Ristretto
/// point, which every midwimble mining key is. Many typos fail that decode,
/// which matters here: a bond on the wrong key is locked money that cannot
/// mine. (Not every typo does, so the UI also shows the key back.)
pub fn parse_mining_key(text: &str) -> Result<[u8; 32]> {
    let t = text.trim();
    let t = t.strip_prefix("mining key").map(str::trim).unwrap_or(t);
    let bytes = hex::decode(t).map_err(|_| anyhow!("the mining key must be hex"))?;
    let key: [u8; 32] = match bytes.len() {
        32 => bytes.try_into().expect("32 bytes"),
        36 => bail!(
            "that looks like a midstate address. Paste the mining key `midwimble bond new` \
             printed (or your pool's mining key), not an address"
        ),
        n => bail!("the mining key must be 32 bytes of hex; this is {n}"),
    };
    if curve25519_dalek::ristretto::CompressedRistretto(key).decompress().is_none() {
        bail!("that is not a valid mining key: check it was copied exactly");
    }
    Ok(key)
}

// ── Hex serde for byte fields ────────────────────────────────────────────

mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        hex::decode(&s)
            .ok()
            .and_then(|b| b.try_into().ok())
            .ok_or_else(|| serde::de::Error::custom("expected 32 bytes of hex"))
    }
}

mod opt_hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(b) => s.serialize_some(&hex::encode(b)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        let s = Option::<String>::deserialize(d)?;
        s.map(|s| {
            hex::decode(&s)
                .ok()
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| serde::de::Error::custom("expected 32 bytes of hex"))
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// midwimble `docs/BONDED_POW.md`: the vector both codebases carry.
    fn golden_script() -> String {
        format!("012000{}10010300801a0633012000{}3201010001", "11".repeat(32), "22".repeat(32))
    }
    const GOLDEN_ADDRESS: &str = "22fd4a4255a093856e1b4a714deb228676742accbe4e3145a6fbcb50f757b67f";

    fn golden() -> BondTerms {
        BondTerms { mining_key: [0x11; 32], bonded_until: 400_000, owner_pk: [0x22; 32] }
    }

    #[test]
    fn the_script_matches_midwimbles_vector() {
        let t = golden();
        assert_eq!(hex::encode(t.script()), golden_script());
        assert_eq!(t.script().len(), 83);
        assert_eq!(hex::encode(t.address()), GOLDEN_ADDRESS);
    }

    #[test]
    fn parse_reads_back_exactly_what_compile_writes() {
        for until in [1u64, 255, 256, 400_000, 1 << 40, u64::MAX] {
            let t = BondTerms { bonded_until: until, ..golden() };
            assert_eq!(parse_script(&t.script()), Some(t), "until {until}");
        }
        // A padded integer is the same script to the VM, but not a bond.
        let padded = hex::decode(format!(
            "012000{}10010800801a06000000000033012000{}3201010001",
            "11".repeat(32),
            "22".repeat(32)
        ))
        .unwrap();
        assert_eq!(padded.len(), 88);
        assert_eq!(parse_script(&padded), None);
        // Anything else is not a bond either.
        let bc = golden().script();
        assert_eq!(parse_script(&bc[..82]), None);
        assert_eq!(parse_script(&[bc.as_slice(), &[0]].concat()), None);
        assert_eq!(parse_script(&script::compile_p2pk(&[0x22; 32])), None);
    }

    #[test]
    fn owner_indices_are_per_bond_and_out_of_the_wallets_way() {
        let a = owner_index(&[1; 32], 500_000);
        assert_eq!(a, owner_index(&[1; 32], 500_000));
        assert_ne!(a, owner_index(&[1; 32], 500_001));
        assert_ne!(a, owner_index(&[2; 32], 500_000));
        // Pinned: the web wallet's worker.js carries the same vectors, so the
        // two wallets derive the same owner keys.
        assert_eq!(owner_index(&[0x33; 32], 360_000), 0x8ee2_a687);
        assert_eq!(owner_index(&[0x11; 32], 400_000), 0x99a1_10a8);
        let ramp: [u8; 32] = std::array::from_fn(|i| i as u8);
        assert_eq!(owner_index(&ramp, 123_456_789), 0x9704_d7d0);
        for (k, h) in [([0u8; 32], 0u64), ([0xff; 32], u64::MAX), ([7; 32], 340_000)] {
            let i = owner_index(&k, h);
            assert!(i >= OWNER_INDEX_BASE && i < OWNER_INDEX_BASE + (1 << 30), "{i:#x}");
        }
    }

    const PHRASE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon art";

    #[test]
    fn the_owner_key_comes_back_from_the_phrase() {
        let master = hd::master_seed_from_mnemonic(PHRASE).unwrap();
        let mk = [0x33; 32];
        let kp = derive_owner(&master, &mk, 360_000).unwrap();
        assert_eq!(kp.height, OWNER_HEIGHT);
        let seed = hd::derive_mss_seed(&master, owner_index(&mk, 360_000) as u64);
        assert_eq!(kp.master_seed, seed);
        // And it signs what the bond script checks.
        let mut kp = kp;
        let msg = [9u8; 32];
        let sig = kp.sign(&msg).unwrap().to_bytes();
        assert_eq!(sig.len(), mss_sig_len(OWNER_HEIGHT));
        let terms = BondTerms { mining_key: mk, bonded_until: 360_000, owner_pk: kp.master_pk };
        let ctx = |height| script::ExecContext {
            commitment: &msg,
            height,
            outputs: &[],
            input_value: 0,
            input_state: None,
            this_address: terms.address(),
            sum_input_value: 0,
        };
        assert!(script::execute_script(&terms.script(), &[sig.clone()], &ctx(360_000)).is_ok());
        assert!(script::execute_script(&terms.script(), &[sig], &ctx(359_999)).is_err());
    }

    /// The whole unbond, through midstate's own transaction rules: refused a
    /// block before the unlock height, accepted from it, and priced above the
    /// mempool's floor.
    #[test]
    fn an_unbond_passes_consensus_from_its_unlock_height() {
        use midstate::core::transaction::validate_transaction;
        let master = hd::master_seed_from_mnemonic(PHRASE).unwrap();
        let (mk, until) = ([0x33; 32], 500_000u64);
        let mut owner = derive_owner(&master, &mk, until).unwrap();
        let rec = BondRecord {
            mining_key: mk,
            bonded_until: until,
            owner_pk: owner.master_pk,
            owner: OwnerKind::Derived,
            value: 1 << 24,
            salt: [7; 32],
            label: String::new(),
            created_at: 0,
            fund_commitment: None,
            unbond_commitment: None,
            unbond_sig: None,
            owner_next_leaf: 0,
            seen_live: true,
        };
        let (mut state, _) = midstate::core::types::State::genesis();
        state.coins.insert(rec.coin_id(), false);

        let (fee, denoms) = resolve_unbond_fee(rec.script().len(), mss_sig_len(OWNER_HEIGHT), rec.value).unwrap();
        let outputs: Vec<OutputData> = denoms
            .iter()
            .enumerate()
            .map(|(i, d)| OutputData::Standard { address: hash(&[i as u8]), value: *d, salt: [i as u8; 32] })
            .collect();
        let salt = [9u8; 32];
        let hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = midstate::core::compute_commitment(&[rec.coin_id()], &hashes, &salt);
        state.commitments.insert(commitment, false);

        let sig = owner.sign(&commitment).unwrap().to_bytes();
        let reveal = unbond_reveal(&rec, sig, outputs, salt);
        state.height = until - 1;
        assert!(validate_transaction(&state, &reveal).is_err(), "a block early");
        state.height = until;
        validate_transaction(&state, &reveal).unwrap();

        assert_eq!(reveal.fee(), fee);
        let size = bincode::serialized_size(&reveal).unwrap();
        assert!(fee * 1024 >= MIN_FEE_PER_KB * size);

        // Signed by any other key, it is refused.
        let mut stranger = derive_owner(&master, &mk, until + 1).unwrap();
        let forged = unbond_reveal(
            &rec,
            stranger.sign(&commitment).unwrap().to_bytes(),
            match &reveal {
                Transaction::Reveal { outputs, .. } => outputs.clone(),
                _ => unreachable!(),
            },
            salt,
        );
        assert!(validate_transaction(&state, &forged).is_err());
    }

    #[test]
    fn unbond_fees_clear_the_mempool_floor() {
        let script_len = golden().script().len();
        for sig_len in [mss_sig_len(OWNER_HEIGHT), mss_sig_len(10)] {
            for value in [1u64 << 10, (1 << 24) + 1, 1 << 24, 1 << 37, (1 << 40) - 1] {
                let (fee, denoms) = resolve_unbond_fee(script_len, sig_len, value).unwrap();
                assert_eq!(denoms.iter().sum::<u64>() + fee, value);
                let size = unbond_reveal_size(script_len, sig_len, denoms.len());
                assert!(fee * 1024 >= MIN_FEE_PER_KB * size, "fee {fee} for {size} bytes");
            }
        }
        assert_eq!(resolve_unbond_fee(script_len, mss_sig_len(4), 16), None);
    }

    #[test]
    fn the_measured_size_is_the_real_reveals_size() {
        let mut kp = mss::keygen(&[5u8; 32], 2).unwrap();
        let sig = kp.sign(&[1u8; 32]).unwrap().to_bytes();
        let t = golden();
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal {
                predicate: Predicate::Script { bytecode: t.script() },
                value: 1 << 30,
                salt: [3u8; 32],
                commitment: None,
            }],
            witnesses: vec![Witness::sig(sig.clone())],
            outputs: vec![
                OutputData::Standard { address: [8u8; 32], value: 1 << 29, salt: [4u8; 32] },
                OutputData::Standard { address: [9u8; 32], value: 1 << 28, salt: [5u8; 32] },
            ],
            salt: [6u8; 32],
        };
        assert_eq!(bincode::serialized_size(&tx).unwrap(), unbond_reveal_size(83, sig.len(), 2));
    }

    #[test]
    fn imports_read_midwimbles_bond_file_and_the_export_line() {
        let t = BondTerms { mining_key: [0x11; 32], bonded_until: 400_000, owner_pk: [0x22; 32] };
        let arr = |b: u8| format!("[{}]", vec![b.to_string(); 32].join(","));
        let file = format!(
            r#"{{"secret":"","coin":{{"script":{{"mining_key":{},"bonded_until":400000,"owner_pk":{}}},"value":16777216,"salt":{}}}}}"#,
            arr(0x11),
            arr(0x22),
            arr(0x44)
        );
        let want = ImportedBond { terms: t, value: 1 << 24, salt: [0x44; 32] };
        assert_eq!(parse_import(&file).unwrap(), want);
        let line = format!("{} {} {}", hex::encode(t.script()), 1u64 << 24, hex::encode([0x44; 32]));
        assert_eq!(parse_import(&line).unwrap(), want);
        assert!(parse_import(r#"{"secret":"ab"}"#).is_err());
        assert!(parse_import("deadbeef 5 00").is_err());
    }

    #[test]
    fn mining_keys_must_be_points() {
        // The Ristretto basepoint, compressed.
        let base = "e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76";
        assert!(parse_mining_key(base).is_ok());
        assert!(parse_mining_key(&format!("mining key   {base}")).is_ok());
        // A typo that makes the encoding negative (low bit of the first byte)
        // never decodes; other typos fail about half the time.
        let typo = format!("e3{}", &base[2..]);
        assert!(parse_mining_key(&typo).is_err());
        assert!(parse_mining_key(&"ab".repeat(36)).unwrap_err().to_string().contains("address"));
        assert!(parse_mining_key("xyz").is_err());
    }

    #[test]
    fn midwimble_counts_a_bond_until_two_days_plus_its_margin_before_unlock() {
        use super::midwimble::*;
        let g = midstate::core::types::BITCOIN_BLOCK_TIME;
        assert_eq!(est_midstate_height(g), CLOCK_MARGIN);
        assert_eq!(est_midstate_height(g + 600), CLOCK_MARGIN + 10);
        let until = 400_000;
        let last = counts_until(until).unwrap();
        assert!(est_midstate_height(last) + MIN_REMAINING_BOND_LOCK <= until);
        assert!(est_midstate_height(last + 1) + MIN_REMAINING_BOND_LOCK > until);
        assert_eq!(counts_until(MIN_REMAINING_BOND_LOCK + CLOCK_MARGIN - 1), None);
        let now = g + 340_000 * 60;
        let u = until_for_days(now, 30);
        let counted_days = (counts_until(u).unwrap() - now) / 86_400;
        assert_eq!(counted_days, 31, "30 days after a day's registration");
    }

    #[test]
    fn records_survive_a_restart_and_a_bad_file_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let wp = dir.path().join("w.dat");
        let mut book = BondBook::default();
        book.bonds.push(BondRecord {
            mining_key: [1; 32],
            bonded_until: 9,
            owner_pk: [2; 32],
            owner: OwnerKind::Derived,
            value: 1 << 24,
            salt: [3; 32],
            label: "x".into(),
            created_at: 1,
            fund_commitment: Some([4; 32]),
            unbond_commitment: None,
            unbond_sig: None,
            owner_next_leaf: 0,
            seen_live: false,
        });
        book.save(&wp).unwrap();
        let (back, warn) = BondBook::load(&wp);
        assert!(warn.is_none());
        assert_eq!(back.bonds.len(), 1);
        assert_eq!(back.bonds[0].fund_commitment, Some([4; 32]));
        assert_eq!(back.find(&book.bonds[0].coin_id()).unwrap().salt, [3; 32]);

        std::fs::write(BondBook::path(&wp), "{ not json").unwrap();
        let (empty, warn) = BondBook::load(&wp);
        assert!(empty.bonds.is_empty() && warn.is_some());
        assert!(!BondBook::path(&wp).exists(), "moved aside, not left to be overwritten");
        let kept = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .any(|e| e.file_name().to_string_lossy().contains("unreadable"));
        assert!(kept);
    }
}
