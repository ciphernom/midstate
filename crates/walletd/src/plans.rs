//! Inheritance plans and payment streams from the desktop wallet
//! (`midstate::core::plans`, `docs/design/inheritance-and-streams.md`).
//!
//! Nothing here holds anyone's coins. A plan's or a stream's coins sit at
//! covenant addresses on chain; this wallet keeps the parameters, the keys it
//! signs with, and the sends it has in flight. Parameters travel between
//! wallets as codes (an heir's or recipient's *payee code*, an owner's *plan
//! code*, a payer's *stream code*) that the people involved hand each other.
//! A code names coins and lets one find them; it never lets one spend them.
//!
//! Coin salts are fixed by each plan's secret, so a wallet finds a plan's or
//! a stream's coins straight from the UTXO set (`live_plan_coins`,
//! `live_tranches`), after a restore or years later, without scanning blocks.
//! Only payouts made by a third party's claim or collection are read from the
//! block that made them (`payouts_in`).

use crate::bond::{self, now_secs};
use anyhow::{anyhow, bail, Context, Result};
use midstate::core::plans::{
    self as cp, payout_address, payout_input, payout_witness, InheritancePlan, PaymentStream, MAX_PLAN_COINS,
    MIN_COIN_EXP,
};
use midstate::core::{hash, InputReveal, OutputData, Transaction, Witness};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Signatures each key of this wallet's for plans and streams can make. An
/// owner signs once per check-in or close, a payer once per cancel, a payee
/// once per sweep (one signature covers every coin a sweep takes).
pub const KEY_HEIGHT: u32 = 10;
pub const PAYEE_CODE: &str = "mdpayee1";
pub const PLAN_CODE: &str = "mdplan1";
pub const STREAM_CODE: &str = "mdstream1";
/// The most coins one claim, collection, cancel or sweep spends.
pub const MAX_SPEND_INPUTS: usize = 64;
/// The smallest coin a plan or stream holds.
pub const MIN_COIN: u64 = cp::MIN_COIN;
/// Blocks searched per tick for the transaction that moved a watched coin.
pub const SCAN_BUDGET: u64 = 2_000;

/// Whether the covenants can run yet: they need `SUM_INPUT_VALUE`.
pub fn available(height: u64) -> Result<(), String> {
    if height >= cp::ACTIVATION_HEIGHT {
        Ok(())
    } else {
        Err(format!("Plans and streams start at midstate block {} ({} to go).", cp::ACTIVATION_HEIGHT, cp::ACTIVATION_HEIGHT - height))
    }
}

// ── Codes ────────────────────────────────────────────────────────────────

fn encode(prefix: &str, bytes: &[u8]) -> String {
    let check = &hash(bytes)[..4];
    format!("{prefix}{}{}", hex::encode(bytes), hex::encode(check))
}

fn decode(prefix: &str, code: &str) -> Result<Vec<u8>> {
    let body: String = code.split_whitespace().collect();
    let hex_part = body.strip_prefix(prefix).ok_or_else(|| anyhow!("that is not a {prefix}… code"))?;
    let raw = hex::decode(hex_part).map_err(|_| anyhow!("the code is damaged"))?;
    if raw.len() < 4 {
        bail!("the code is damaged");
    }
    let (bytes, check) = raw.split_at(raw.len() - 4);
    if &hash(bytes)[..4] != check {
        bail!("the code is damaged: its checksum does not match");
    }
    Ok(bytes.to_vec())
}

/// What an heir or a recipient hands out: the key it is paid to, and a name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Payee {
    pub pk: [u8; 32],
    pub name: String,
}

pub fn encode_payee(p: &Payee) -> String {
    let mut b = p.pk.to_vec();
    b.extend_from_slice(p.name.as_bytes());
    encode(PAYEE_CODE, &b)
}

pub fn decode_payee(code: &str) -> Result<Payee> {
    let b = decode(PAYEE_CODE, code)?;
    if b.len() < 32 || b.len() > 32 + 64 {
        bail!("the payee code is damaged");
    }
    let pk: [u8; 32] = b[..32].try_into().unwrap();
    let name = String::from_utf8(b[32..].to_vec()).map_err(|_| anyhow!("the payee code is damaged"))?;
    Ok(Payee { pk, name })
}

/// A plan's code: its parameters and its heirs' names.
pub fn encode_plan(p: &InheritancePlan, names: &[String]) -> String {
    let mut b = p.to_bytes();
    for n in names {
        let n = n.as_bytes();
        b.push(n.len().min(64) as u8);
        b.extend_from_slice(&n[..n.len().min(64)]);
    }
    encode(PLAN_CODE, &b)
}

pub fn decode_plan(code: &str) -> Result<(InheritancePlan, Vec<String>)> {
    let b = decode(PLAN_CODE, code)?;
    // The plan's own bytes end where its heirs do.
    let heirs = *b.get(81).ok_or_else(|| anyhow!("the plan code is damaged"))? as usize;
    let end = 82 + 36 * heirs;
    let plan = b.get(..end).and_then(InheritancePlan::from_bytes).ok_or_else(|| anyhow!("the plan code holds no valid plan"))?;
    let mut names = Vec::new();
    let mut rest = &b[end..];
    while let Some((&n, tail)) = rest.split_first() {
        let n = n as usize;
        if tail.len() < n {
            bail!("the plan code is damaged");
        }
        names.push(String::from_utf8_lossy(&tail[..n]).into_owned());
        rest = &tail[n..];
    }
    names.resize(plan.heirs.len(), String::new());
    Ok((plan, names))
}

pub fn encode_stream(s: &PaymentStream, label: &str) -> String {
    let mut b = s.to_bytes();
    let l = label.as_bytes();
    b.extend_from_slice(&l[..l.len().min(64)]);
    encode(STREAM_CODE, &b)
}

pub fn decode_stream(code: &str) -> Result<(PaymentStream, String)> {
    let b = decode(STREAM_CODE, code)?;
    const LEN: usize = 126;
    let s = b.get(..LEN).and_then(PaymentStream::from_bytes).ok_or_else(|| anyhow!("the stream code holds no valid stream"))?;
    Ok((s, String::from_utf8_lossy(&b[LEN..]).into_owned()))
}

/// Shares from percentages in hundredths (`10_000` is all of it), adding up
/// to exactly what a plan needs: the rounding goes to the first heir.
pub fn shares_from_bps(bps: &[u32]) -> Result<Vec<u32>> {
    if bps.iter().sum::<u32>() != 10_000 {
        bail!("the shares must add up to 100%");
    }
    if bps.contains(&0) {
        bail!("every heir needs a share");
    }
    let parts = (cp::SHARE_DENOM - 1) as u64;
    let mut shares: Vec<u32> = bps.iter().map(|b| (parts * *b as u64 / 10_000) as u32).collect();
    let short = parts as u32 - shares.iter().sum::<u32>();
    shares[0] += short;
    Ok(shares)
}

// ── Finding coins ────────────────────────────────────────────────────────

/// A coin of a plan: deadline, index, value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanCoin {
    pub k: u64,
    pub n: u32,
    pub value: u64,
}

impl PlanCoin {
    pub fn id(&self, p: &InheritancePlan) -> [u8; 32] {
        p.coin_id(self.k, self.n, self.value)
    }
}

/// Every coin of `p` on chain (`live` answers for a coin id) as of `height`.
pub fn live_plan_coins(p: &InheritancePlan, height: u64, live: impl Fn(&[u8; 32]) -> bool) -> Vec<PlanCoin> {
    let mut found = Vec::new();
    for k in p.epochs_to(height) {
        let address = p.address(k);
        for n in 0..MAX_PLAN_COINS {
            let salt = p.coin_salt(k, n);
            for e in MIN_COIN_EXP..64 {
                let value = 1u64 << e;
                if live(&midstate::core::compute_coin_id(&address, value, &salt)) {
                    found.push(PlanCoin { k, n, value });
                }
            }
        }
    }
    found
}

/// A stream's tranche coins on chain: `(tranche, index, value)`.
pub fn live_tranches(s: &PaymentStream, live: impl Fn(&[u8; 32]) -> bool) -> Vec<(u32, u32, u64)> {
    (0..s.count)
        .flat_map(|j| s.coins(j).into_iter().map(move |(n, v, id)| (j, n, v, id)))
        .filter(|(_, _, _, id)| live(id))
        .map(|(j, n, v, _)| (j, n, v))
        .collect()
}

/// The coins to fund a plan with from `amount`: at most [`MAX_PLAN_COINS`]
/// powers of two of [`MIN_COIN`] or more, the largest first. The rest stays
/// in the wallet. Empty if `amount` cannot make one.
pub fn plan_denominations(amount: u64) -> Vec<u64> {
    (MIN_COIN_EXP..64)
        .rev()
        .map(|e| 1u64 << e)
        .filter(|c| amount & c != 0)
        .take(MAX_PLAN_COINS as usize)
        .collect()
}

/// Indices free at deadline `k` of `p`, given the coins there.
pub fn free_indices(have: &[PlanCoin], k: u64, n: usize) -> Vec<u32> {
    (0..MAX_PLAN_COINS).filter(|i| !have.iter().any(|c| c.k == k && c.n == *i)).take(n).collect()
}

/// A payout made to one of this wallet's payee keys, waiting to be swept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayoutCoin {
    #[serde(with = "hex32")]
    pub tag: [u8; 32],
    #[serde(with = "hex32")]
    pub pk: [u8; 32],
    pub value: u64,
    #[serde(with = "hex32")]
    pub salt: [u8; 32],
    /// What paid it, in words.
    pub from: String,
    /// Swept by a send in flight.
    #[serde(default, with = "opt_hex32")]
    pub sweeping: Option<[u8; 32]>,
    /// When this wallet learnt of it (Unix seconds).
    #[serde(default)]
    pub found_at: u64,
}

impl PayoutCoin {
    pub fn input(&self) -> InputReveal {
        payout_input(&self.tag, &self.pk, self.value, self.salt)
    }
    pub fn id(&self) -> [u8; 32] {
        self.input().coin_id()
    }
}

/// The outputs of `tx` paying any of `payouts` (`(tag, pk)`), as payout coins.
pub fn payouts_in(tx: &Transaction, payouts: &[([u8; 32], [u8; 32])], from: &str) -> Vec<PayoutCoin> {
    let Transaction::Reveal { outputs, .. } = tx else { return Vec::new() };
    let addrs: Vec<([u8; 32], [u8; 32], [u8; 32])> =
        payouts.iter().map(|(tag, pk)| (payout_address(tag, pk), *tag, *pk)).collect();
    outputs
        .iter()
        .filter_map(|o| match o {
            OutputData::Standard { address, value, salt } => addrs
                .iter()
                .find(|(a, _, _)| a == address)
                .map(|(_, tag, pk)| PayoutCoin {
                    tag: *tag,
                    pk: *pk,
                    value: *value,
                    salt: *salt,
                    from: from.into(),
                    sweeping: None,
                    found_at: 0,
                }),
            _ => None,
        })
        .collect()
}

// ── Sends with covenant inputs ───────────────────────────────────────────

/// How one covenant input is opened at the reveal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Opening {
    /// A witness that needs no key (a claim, a collection).
    Fixed(Vec<String>),
    /// This wallet's signature over the commitment, then these items.
    Signed(Vec<String>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovInput {
    pub reveal: InputReveal,
    pub opening: Opening,
}

/// A send of covenant coins in flight. Everything its reveal needs is here,
/// so it reveals the same way after a restart.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CovSpend {
    #[serde(with = "hex32")]
    pub commitment: [u8; 32],
    /// "check-in" | "close" | "claim" | "fund-plan" | "cancel" | "collect" | "sweep"
    pub kind: String,
    /// Which plan or stream (its id), if any.
    pub of: String,
    pub inputs: Vec<CovInput>,
    /// The key that signs `Signed` inputs, and its signature once made (kept
    /// so a retried reveal sends the same one).
    #[serde(default, with = "opt_hex32")]
    pub signer: Option<[u8; 32]>,
    #[serde(default)]
    pub sig: Option<String>,
    /// Payouts to this wallet's keys the send makes, swept once it confirms.
    #[serde(default)]
    pub pays_us: Vec<PayoutCoin>,
    /// What it brings back into this wallet as plain coins (history shows it
    /// as received).
    pub returned: u64,
    pub created_at: u64,
}

impl CovSpend {
    pub fn contested(&self) -> [u8; 32] {
        self.inputs[0].reveal.coin_id()
    }
}

/// The exact reveal size of `inputs` (signed ones with a `sig_len` signature)
/// and `n_outputs` outputs, as the mempool measures it.
pub fn reveal_size(inputs: &[CovInput], sig_len: usize, n_outputs: usize) -> u64 {
    let witnesses = inputs
        .iter()
        .map(|i| match &i.opening {
            Opening::Fixed(items) => Witness::ScriptInputs(items.iter().map(|h| hex::decode(h).unwrap_or_default()).collect()),
            Opening::Signed(rest) => {
                let mut items = vec![vec![0u8; sig_len]];
                items.extend(rest.iter().map(|h| hex::decode(h).unwrap_or_default()));
                Witness::ScriptInputs(items)
            }
        })
        .collect();
    let tx = Transaction::Reveal {
        inputs: inputs.iter().map(|i| i.reveal.clone()).collect(),
        witnesses,
        outputs: (0..n_outputs).map(|_| OutputData::Standard { address: [0; 32], value: 0, salt: [0; 32] }).collect(),
        salt: [0; 32],
    };
    bincode::serialized_size(&tx).unwrap_or(u64::MAX)
}

/// The relay fee for `bytes`, with room to spare.
pub fn fee_for(bytes: u64) -> u64 {
    (bytes * 2 * 10).div_ceil(1024) + 20
}

/// The fee for sending `inputs` back to this wallet, and the coins that
/// leaves: the fee depends on how many outputs, which depends on the fee.
pub fn resolve_fee(inputs: &[CovInput], sig_len: usize, extra_outputs: usize, value: u64) -> Option<(u64, Vec<u64>)> {
    let mut fee = fee_for(reveal_size(inputs, sig_len, extra_outputs + 1));
    for _ in 0..8 {
        let denoms = midstate::core::decompose_value(value.checked_sub(fee)?);
        let need = fee_for(reveal_size(inputs, sig_len, extra_outputs + denoms.len()));
        if need <= fee {
            return Some((fee, denoms));
        }
        fee = need;
    }
    None
}

/// A plan's coins as inputs opened by its owner.
pub fn owner_inputs(p: &InheritancePlan, coins: &[PlanCoin]) -> Vec<CovInput> {
    coins
        .iter()
        .map(|c| CovInput { reveal: p.input(c.k, c.n, c.value), opening: Opening::Signed(vec!["01".into()]) })
        .collect()
}

/// A plan's coins at deadline `k`, opened for a claim.
pub fn claim_inputs(p: &InheritancePlan, coins: &[PlanCoin]) -> Vec<CovInput> {
    coins
        .iter()
        .map(|c| CovInput { reveal: p.input(c.k, c.n, c.value), opening: Opening::Fixed(vec!["00".into()]) })
        .collect()
}

pub fn tranche_inputs(s: &PaymentStream, coins: &[(u32, u32, u64)], cancel: bool) -> Vec<CovInput> {
    coins
        .iter()
        .map(|(j, n, _)| CovInput {
            reveal: s.input(*j, *n),
            opening: if cancel {
                Opening::Signed(vec!["01".into()])
            } else {
                let Witness::ScriptInputs(items) = s.collect_witness();
                Opening::Fixed(items.iter().map(hex::encode).collect())
            },
        })
        .collect()
}

pub fn payout_inputs(coins: &[PayoutCoin]) -> Vec<CovInput> {
    coins.iter().map(|c| CovInput { reveal: c.input(), opening: Opening::Signed(vec![]) }).collect()
}

/// The witnesses for `inputs`, with `sig` where this wallet signs.
pub fn witnesses(inputs: &[CovInput], sig: Option<&[u8]>) -> Result<Vec<Witness>> {
    inputs
        .iter()
        .map(|i| {
            let hexes = |v: &[String]| -> Result<Vec<Vec<u8>>> {
                v.iter().map(|h| hex::decode(h).context("a witness item is not hex")).collect()
            };
            Ok(match &i.opening {
                Opening::Fixed(items) => Witness::ScriptInputs(hexes(items)?),
                Opening::Signed(rest) => {
                    let sig = sig.ok_or_else(|| anyhow!("a signed input with no signature"))?;
                    let mut items = vec![sig.to_vec()];
                    items.extend(hexes(rest)?);
                    Witness::ScriptInputs(items)
                }
            })
        })
        .collect()
}

/// The witness a claim or collection's inputs need, for tests and checks.
pub fn payout_witness_for(sig: Vec<u8>) -> Witness {
    payout_witness(sig)
}

// ── Records ──────────────────────────────────────────────────────────────

/// A plan this wallet made: it holds the owner key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OwnedPlan {
    /// The plan's bytes (hex): `InheritancePlan::to_bytes`.
    pub plan: String,
    pub label: String,
    pub heir_names: Vec<String>,
    pub created_at: u64,
    /// "active" | "closed"
    pub status: String,
    #[serde(default)]
    pub note: String,
}

/// A plan that names one of this wallet's payee keys as an heir.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeldPlan {
    pub plan: String,
    pub label: String,
    pub heir_names: Vec<String>,
    pub imported_at: u64,
    #[serde(default)]
    pub note: String,
}

/// A stream this wallet pays.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SentStream {
    pub stream: String,
    pub label: String,
    pub recipient_name: String,
    pub created_at: u64,
    /// "funding" | "active" | "ended"
    pub status: String,
    #[serde(default, with = "opt_hex32")]
    pub fund_commitment: Option<[u8; 32]>,
    #[serde(default)]
    pub note: String,
}

/// A stream paying one of this wallet's payee keys.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeldStream {
    pub stream: String,
    pub label: String,
    pub imported_at: u64,
    /// Collect tranches as they come due (the default): a cancellable
    /// stream's payer can take back what has not been collected.
    #[serde(default = "yes")]
    pub auto_collect: bool,
    #[serde(default)]
    pub note: String,
}

fn yes() -> bool {
    true
}

/// Coins of a plan or stream of ours that left the chain without our doing,
/// and where the search for what moved them stands.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vanished {
    pub of: String,
    #[serde(with = "hex32_vec")]
    pub coins: Vec<[u8; 32]>,
    /// Payout `(tag, pk)` pairs of ours the move may have paid.
    pub payouts: Vec<(String, String)>,
    pub from: u64,
    pub what: String,
}

/// Plans, streams and their sends, persisted beside the wallet.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanBook {
    pub owned: Vec<OwnedPlan>,
    pub held: Vec<HeldPlan>,
    pub sent: Vec<SentStream>,
    pub received: Vec<HeldStream>,
    pub spends: Vec<CovSpend>,
    pub payouts: Vec<PayoutCoin>,
    pub vanished: Vec<Vanished>,
    /// This wallet's payee keys, newest last.
    #[serde(with = "hex32_vec")]
    pub payee_keys: Vec<[u8; 32]>,
    /// What was on chain at the last look, and when, by plan or stream id:
    /// kept, so coins that go while the wallet is closed are noticed.
    #[serde(default)]
    pub live: std::collections::HashMap<String, (u64, Vec<PlanCoin>)>,
    /// The tip the chain was last looked at for (not saved).
    #[serde(skip)]
    pub ticked_at: u64,
}

impl PlanBook {
    pub fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("plans")
    }

    /// Missing file: nothing yet. An unreadable one is set aside, never
    /// overwritten: it may hold the only copy of a plan's parameters.
    pub fn load(wallet_path: &Path) -> (Self, Option<String>) {
        let path = Self::path(wallet_path);
        let Ok(raw) = std::fs::read(&path) else { return (Self::default(), None) };
        match serde_json::from_slice(&raw) {
            Ok(book) => (book, None),
            Err(e) => {
                let aside = path.with_extension(format!("plans.unreadable-{}", now_secs()));
                let moved = std::fs::rename(&path, &aside).is_ok();
                let text = if moved {
                    format!("The plan and stream records could not be read ({e}); the file was kept as {}.", aside.display())
                } else {
                    format!("The plan and stream records at {} could not be read: {e}", path.display())
                };
                (Self::default(), Some(text))
            }
        }
    }

    pub fn save(&self, wallet_path: &Path) -> Result<()> {
        let path = Self::path(wallet_path);
        let tmp = path.with_extension("plans.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?).context("writing plan records")?;
        std::fs::rename(&tmp, &path).context("saving plan records")?;
        Ok(())
    }

    pub fn spend(&self, commitment: &[u8; 32]) -> Option<&CovSpend> {
        self.spends.iter().find(|s| &s.commitment == commitment)
    }

    /// Whether a send in flight already spends `coin`.
    pub fn spending(&self, coin: &[u8; 32]) -> bool {
        self.spends.iter().any(|s| s.inputs.iter().any(|i| &i.reveal.coin_id() == coin))
    }
}

/// A plan's or stream's id: the hash of its bytes, hex.
pub fn id_of(bytes: &[u8]) -> String {
    hex::encode(&hash(bytes)[..16])
}

pub fn plan_of(hex_bytes: &str) -> Option<InheritancePlan> {
    InheritancePlan::from_bytes(&hex::decode(hex_bytes).ok()?)
}

pub fn stream_of(hex_bytes: &str) -> Option<PaymentStream> {
    PaymentStream::from_bytes(&hex::decode(hex_bytes).ok()?)
}

/// "in about 3 months (block 400,000)".
pub fn when(height: u64, tip: u64) -> String {
    if height <= tip {
        format!("since block {height}")
    } else {
        format!("in about {} (block {height})", bond::blocks_human(height - tip))
    }
}

pub(crate) mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let s = String::deserialize(d)?;
        let v = hex::decode(&s).map_err(serde::de::Error::custom)?;
        v.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
    }
}

pub(crate) mod opt_hex32 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(v) => s.serialize_some(&hex::encode(v)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        let s: Option<String> = Option::deserialize(d)?;
        s.map(|s| {
            let v = hex::decode(&s).map_err(serde::de::Error::custom)?;
            v.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
        })
        .transpose()
    }
}

pub(crate) mod hex32_vec {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[[u8; 32]], s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(v.iter().map(hex::encode))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[u8; 32]>, D::Error> {
        let v: Vec<String> = Vec::deserialize(d)?;
        v.iter()
            .map(|s| {
                let b = hex::decode(s).map_err(serde::de::Error::custom)?;
                b.try_into().map_err(|_| serde::de::Error::custom("expected 32 bytes"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip_and_damage_is_caught() {
        let p = Payee { pk: [7; 32], name: "Alice".into() };
        assert_eq!(decode_payee(&encode_payee(&p)).unwrap(), p);
        let mut c = encode_payee(&p);
        c.replace_range(20..21, if &c[20..21] == "0" { "1" } else { "0" });
        assert!(decode_payee(&c).is_err());
        assert!(decode_payee("mdplan1abcd").is_err());

        let plan = InheritancePlan {
            owner_pk: [1; 32],
            heirs: vec![cp::Heir { pk: [2; 32], share: 40_000 }, cp::Heir { pk: [3; 32], share: 25_535 }],
            start: 400_000,
            period: 365 * cp::DAY,
            secret: [9; 32],
        };
        let names = vec!["Bea".to_string(), "Cal".to_string()];
        let (back, n) = decode_plan(&encode_plan(&plan, &names)).unwrap();
        assert_eq!(back, plan);
        assert_eq!(n, names);

        let s = PaymentStream {
            payer_pk: Some([4; 32]),
            recipient_pk: [5; 32],
            first: 400_000,
            period: 7 * cp::DAY,
            amount: MIN_COIN * 3,
            count: 10,
            secret: [6; 32],
        };
        let (back, l) = decode_stream(&encode_stream(&s, "rent")).unwrap();
        assert_eq!(back, s);
        assert_eq!(l, "rent");
    }

    #[test]
    fn shares_always_add_up_to_what_a_plan_needs() {
        for bps in [vec![10_000], vec![5_000, 5_000], vec![3_333, 3_333, 3_334], vec![1; 0], vec![9_999, 1]] {
            match shares_from_bps(&bps) {
                Ok(s) => assert_eq!(s.iter().map(|x| *x as u64).sum::<u64>(), cp::SHARE_DENOM - 1),
                Err(_) => assert!(bps.iter().sum::<u32>() != 10_000),
            }
        }
        assert!(shares_from_bps(&[5_000, 4_000]).is_err());
        assert!(shares_from_bps(&[10_000, 0]).is_err());
    }

    #[test]
    fn denominations_keep_to_eight_large_coins() {
        let amount = (1u64 << 40) - 1;
        let d = plan_denominations(amount);
        assert_eq!(d.len(), MAX_PLAN_COINS as usize);
        assert!(d.iter().all(|c| *c >= MIN_COIN && c.is_power_of_two()));
        assert!(plan_denominations(MIN_COIN - 1).is_empty());
        assert_eq!(plan_denominations(3 * MIN_COIN), vec![2 * MIN_COIN, MIN_COIN]);
    }

    #[test]
    fn a_plan_is_found_on_chain_from_its_code_alone() {
        let plan = InheritancePlan {
            owner_pk: [1; 32],
            heirs: vec![cp::Heir { pk: [2; 32], share: 65_535 }],
            start: 400_000,
            period: 30 * cp::DAY,
            secret: [9; 32],
        };
        let tip = plan.start + 95 * cp::DAY;
        let k = plan.epoch_after(plan.start + 60 * cp::DAY);
        let on_chain: std::collections::HashSet<[u8; 32]> =
            [plan.coin_id(k, 0, 1 << 33), plan.coin_id(k, 1, 1 << 27), plan.coin_id(1, 0, 1 << 30)].into();
        let mut found = live_plan_coins(&plan, tip, |id| on_chain.contains(id));
        found.sort_by_key(|c| (c.k, c.n));
        assert_eq!(
            found,
            vec![
                PlanCoin { k: 1, n: 0, value: 1 << 30 },
                PlanCoin { k, n: 0, value: 1 << 33 },
                PlanCoin { k, n: 1, value: 1 << 27 }
            ]
        );
        assert_eq!(free_indices(&found, k, 2), vec![2, 3]);
    }

    #[test]
    fn fees_cover_the_reveal_and_its_outputs() {
        let plan = InheritancePlan {
            owner_pk: [1; 32],
            heirs: vec![cp::Heir { pk: [2; 32], share: 65_535 }],
            start: 400_000,
            period: 30 * cp::DAY,
            secret: [9; 32],
        };
        let coins = [PlanCoin { k: 1, n: 0, value: 1 << 33 }, PlanCoin { k: 1, n: 1, value: 1 << 30 }];
        let inputs = owner_inputs(&plan, &coins);
        let sig_len = bond::mss_sig_len(KEY_HEIGHT);
        let (fee, denoms) = resolve_fee(&inputs, sig_len, 0, (1 << 33) + (1 << 30)).unwrap();
        assert_eq!(denoms.iter().sum::<u64>() + fee, (1 << 33) + (1 << 30));
        let bytes = reveal_size(&inputs, sig_len, denoms.len());
        assert!(fee * 1024 >= bytes * 10, "{fee} for {bytes} bytes");
    }
}
