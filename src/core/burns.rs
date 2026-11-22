//! MSB-1: burns that back something — a name, a key, a Base account, a pool,
//! a hub — or notarize a document, weighed in block rewards.
//!
//! Nothing here is consensus. An MSB-1 burn is an ordinary `DataBurn` output
//! whose payload starts with `MSB1`; the chain checks it as it checks any burn
//! (a payload of at most 80 bytes, a value that is zero or a power of two).
//! Nodes index MSB-1 burns (`storage::burn_index`), and wallets and the
//! explorer read the index.
//!
//! ```text
//! payload = "MSB1" ‖ purpose (1) ‖ subject (32) ‖ label (0–43 bytes, UTF-8)
//! ```
//!
//! | purpose | backs | subject |
//! |---|---|---|
//! | 1 name | a `.mds` name | the name, zero-padded |
//! | 2 key | any key or address | its 32 bytes |
//! | 3 evm | a Base (Ethereum) account | its 20 bytes, zero-padded |
//! | 4 pool | a mining pool | the pool's fee address |
//! | 5 hub | a payment-channel hub | the hub's identity key |
//! | 6 notary | a document, timestamped | its BLAKE3 hash |
//! | 7 notary-sha256 | the same | its SHA-256 hash |
//!
//! A burn of more than one power of two is several outputs with one payload;
//! they count as one burn.
//!
//! **Weight.** A burn of `v` units at height `h` weighs `v / block_reward(h)`
//! *blocks* — what it cost in issuance — and its weight halves every year
//! after ([`HALF_LIFE`]). A subject's *backing* at a height is the sum of its
//! burns' weights then.
//!
//! Why blocks: the reward halves every year, and so would the meaning of any
//! price fixed in units, as MSN-1's fixed name prices show (they burn twice as
//! large a share of issuance each year). Measured in blocks, backing costs a
//! steady share of what the chain issues: holding `B` blocks of backing costs
//! `B·ln 2` blocks a year, so all the backing anyone keeps, `ΣB`, burns
//! `ΣB·ln 2 / 525,600` of each year's issuance, in any year. About 680,000
//! blocks of backing, kept up, burns 90% of issuance.
//!
//! Decay applies to every burn alike, so subjects keep their order as time
//! passes: rank by [`anchor`] sums, which change only when burns do.

use super::types::{block_reward, cumulative_reward, decompose_value, OutputData, BLOCKS_PER_YEAR, MAX_BURN_DATA_SIZE};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const TAG: &[u8; 4] = b"MSB1";
const HEAD: usize = 4 + 1 + 32;
/// The longest label: what is left of a burn's 80 bytes.
pub const LABEL_MAX: usize = MAX_BURN_DATA_SIZE - HEAD;
/// A burn's weight halves every year: one halving of the reward.
pub const HALF_LIFE: u64 = BLOCKS_PER_YEAR;

/// What a burn backs. Unknown purposes are kept and shown as numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Purpose(pub u8);

impl Purpose {
    pub const NAME: Purpose = Purpose(1);
    pub const KEY: Purpose = Purpose(2);
    pub const EVM: Purpose = Purpose(3);
    pub const POOL: Purpose = Purpose(4);
    pub const HUB: Purpose = Purpose(5);
    pub const NOTARY: Purpose = Purpose(6);
    pub const NOTARY_SHA256: Purpose = Purpose(7);
    pub const ALL: [Purpose; 7] =
        [Self::NAME, Self::KEY, Self::EVM, Self::POOL, Self::HUB, Self::NOTARY, Self::NOTARY_SHA256];

    /// Its name in the API: `name`, `key`, `evm`, `pool`, `hub`, `notary`,
    /// `notary-sha256`, or `purpose-N`.
    pub fn word(self) -> String {
        match self.0 {
            1 => "name".into(),
            2 => "key".into(),
            3 => "evm".into(),
            4 => "pool".into(),
            5 => "hub".into(),
            6 => "notary".into(),
            7 => "notary-sha256".into(),
            n => format!("purpose-{n}"),
        }
    }

    pub fn parse(s: &str) -> Option<Purpose> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|p| p.word() == s).or_else(|| s.strip_prefix("purpose-")?.parse().ok().map(Purpose))
    }

    /// A timestamp rather than backing: weight means nothing to a document.
    pub fn is_notary(self) -> bool {
        self == Self::NOTARY || self == Self::NOTARY_SHA256
    }
}

/// One MSB-1 burn's payload, read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Backing {
    pub purpose: Purpose,
    pub subject: [u8; 32],
    /// Free text, UTF-8, at most [`LABEL_MAX`] bytes: a note, a pool's or a
    /// document's name.
    pub label: String,
}

impl Backing {
    pub fn new(purpose: Purpose, subject: [u8; 32], label: &str) -> Result<Self> {
        let label = label.trim();
        if label.len() > LABEL_MAX {
            bail!("a label is at most {LABEL_MAX} bytes");
        }
        let b = Backing { purpose, subject, label: label.to_string() };
        if Backing::parse(&b.payload()).as_ref() != Some(&b) {
            bail!("that is not a {} subject", purpose.word());
        }
        Ok(b)
    }

    /// Backing for the name `name` (`nick` or `nick.mds`).
    pub fn name(name: &str, label: &str) -> Result<Self> {
        Self::new(Purpose::NAME, name_subject(name)?, label)
    }

    pub fn evm(account: &[u8; 20], label: &str) -> Result<Self> {
        Self::new(Purpose::EVM, evm_subject(account), label)
    }

    pub fn payload(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(HEAD + self.label.len());
        p.extend_from_slice(TAG);
        p.push(self.purpose.0);
        p.extend_from_slice(&self.subject);
        p.extend_from_slice(self.label.as_bytes());
        p
    }

    /// An MSB-1 payload, if `payload` is one. A name or account subject must
    /// be one (the name canonical, the padding zero), and a label UTF-8.
    pub fn parse(payload: &[u8]) -> Option<Self> {
        if payload.len() < HEAD || payload.len() > MAX_BURN_DATA_SIZE || &payload[..4] != TAG {
            return None;
        }
        let purpose = Purpose(payload[4]);
        let subject: [u8; 32] = payload[5..HEAD].try_into().ok()?;
        let label = String::from_utf8(payload[HEAD..].to_vec()).ok()?;
        match purpose {
            Purpose::NAME => {
                subject_name(&subject)?;
            }
            Purpose::EVM if subject[20..] != [0; 12] => return None,
            Purpose(0) => return None,
            _ => {}
        }
        Some(Backing { purpose, subject, label })
    }

    /// The burn's outputs for `amount` units: one per power of two.
    pub fn outputs(&self, amount: u64) -> Vec<OutputData> {
        let payload = self.payload();
        decompose_value(amount).into_iter().map(|v| OutputData::DataBurn { payload: payload.clone(), value_burned: v }).collect()
    }

    /// The subject for people: the name, the `0x` account, or hex.
    pub fn subject_text(&self) -> String {
        subject_text(self.purpose, &self.subject)
    }
}

/// A name's subject: its canonical form, zero-padded.
pub fn name_subject(name: &str) -> Result<[u8; 32]> {
    let n = super::names::normalize(name)?;
    let mut s = [0u8; 32];
    s[..n.len()].copy_from_slice(n.as_bytes());
    Ok(s)
}

/// The name a name subject holds, if it holds one.
pub fn subject_name(subject: &[u8; 32]) -> Option<String> {
    let end = subject.iter().position(|b| *b == 0).unwrap_or(32);
    if subject[end..].iter().any(|b| *b != 0) {
        return None;
    }
    let n = std::str::from_utf8(&subject[..end]).ok()?;
    (super::names::normalize(n).ok()? == n).then(|| n.to_string())
}

pub fn evm_subject(account: &[u8; 20]) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[..20].copy_from_slice(account);
    s
}

pub fn subject_text(purpose: Purpose, subject: &[u8; 32]) -> String {
    match purpose {
        Purpose::NAME => subject_name(subject).map(|n| format!("{n}{}", super::names::SUFFIX)).unwrap_or_else(|| hex::encode(subject)),
        Purpose::EVM => format!("0x{}", hex::encode(&subject[..20])),
        _ => hex::encode(subject),
    }
}

/// A subject from text, as the API and the CLI take it: a name, a `0x`
/// account, a checksummed address or 64 hex characters.
pub fn parse_subject(purpose: Purpose, text: &str) -> Result<[u8; 32]> {
    let t = text.trim();
    match purpose {
        Purpose::NAME => name_subject(t),
        Purpose::EVM => {
            let h = t.strip_prefix("0x").unwrap_or(t);
            let b = hex::decode(h).ok().filter(|b| b.len() == 20);
            match b {
                Some(b) => Ok(evm_subject(&b.try_into().expect("20 bytes"))),
                None => bail!("a Base account is 0x and 40 hex characters"),
            }
        }
        _ => {
            let h = t.strip_prefix("0x").unwrap_or(t).to_ascii_lowercase();
            if let Some(b) = hex::decode(&h).ok().filter(|b| b.len() == 32) {
                return Ok(b.try_into().expect("32 bytes"));
            }
            match super::types::parse_address_flexible(&h) {
                Ok(a) => Ok(a),
                Err(_) => bail!("a {} subject is 64 hex characters (or an address)", purpose.word()),
            }
        }
    }
}

// ── Weight ──────────────────────────────────────────────────────────────────

/// What a burn of `value` units at `height` weighs when it is made, in blocks.
pub fn weight(value: u64, height: u64) -> f64 {
    value as f64 / block_reward(height) as f64
}

/// `weight(value, height)` scaled by `2^(height/HALF_LIFE)`: a sum of these
/// gives a subject's backing at any height `t` as `sum · 2^(-t/HALF_LIFE)`,
/// and sums compare as backing does.
pub fn anchor(value: u64, height: u64) -> f64 {
    weight(value, height) * (height as f64 / HALF_LIFE as f64).exp2()
}

/// The backing at `at` of burns whose anchors sum to `anchors`.
pub fn backing_at(anchors: f64, at: u64) -> f64 {
    anchors * (-(at as f64) / HALF_LIFE as f64).exp2()
}

/// What one burn of `value` at `height` weighs at `at` (≥ `height`).
pub fn weight_at(value: u64, height: u64, at: u64) -> f64 {
    weight(value, height) * (-(at.saturating_sub(height) as f64) / HALF_LIFE as f64).exp2()
}

/// Units to burn at `height` for `blocks` of weight.
pub fn cost(blocks: u64, height: u64) -> u64 {
    blocks.saturating_mul(block_reward(height))
}

/// Blocks a year it takes to keep `backing` blocks of it: what decay takes.
pub fn upkeep(backing: f64) -> f64 {
    backing * std::f64::consts::LN_2
}

/// What heights `from..to` issued, in units.
pub fn issued(from: u64, to: u64) -> u64 {
    cumulative_reward(to).saturating_sub(cumulative_reward(from))
}

/// How much backing counts as backed: an hour, a day and a week of issuance.
pub const TIERS: [(f64, &str); 3] = [(60.0, "backed"), (1_440.0, "well backed"), (10_080.0, "strongly backed")];

/// The tier `backing` blocks reach, if any.
pub fn tier(backing: f64) -> Option<&'static str> {
    TIERS.iter().rev().find(|(t, _)| backing >= *t).map(|(_, w)| *w)
}

/// Blocks for people: "45 blocks", "1.5 days of issuance".
pub fn blocks_text(blocks: f64) -> String {
    if blocks >= 20_160.0 {
        format!("{:.1} weeks of issuance", blocks / 10_080.0)
    } else if blocks >= 1_440.0 {
        format!("{:.1} days of issuance", blocks / 1_440.0)
    } else if blocks >= 60.0 {
        format!("{:.1} hours of issuance", blocks / 60.0)
    } else {
        format!("{blocks:.1} blocks")
    }
}

// ── Every burn, by kind ─────────────────────────────────────────────────────

/// What a burn is, for the chain's totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// MSN-1: registering, renewing or claiming a name.
    Names,
    /// MSB-1: backing or a notarization.
    Backing,
    /// Cross-chain order books (`MDX…`): usually no value.
    Dex,
    Other,
}

impl Kind {
    pub fn of(payload: &[u8]) -> Kind {
        if payload.starts_with(super::names::TAG) {
            Kind::Names
        } else if Backing::parse(payload).is_some() {
            Kind::Backing
        } else if payload.starts_with(b"MDX") {
            Kind::Dex
        } else {
            Kind::Other
        }
    }

    pub fn word(self) -> &'static str {
        match self {
            Kind::Names => "names",
            Kind::Backing => "backing",
            Kind::Dex => "dex",
            Kind::Other => "other",
        }
    }
}

/// A transaction's burns: each distinct payload once, its outputs' values
/// summed. Standard outputs to the burn address count, under an empty
/// payload: no script hashes to it.
pub fn burns_of(outputs: &[OutputData]) -> Vec<(Vec<u8>, u64)> {
    let mut out: Vec<(Vec<u8>, u64)> = Vec::new();
    for o in outputs {
        let (payload, value) = match o {
            OutputData::DataBurn { payload, value_burned } => (payload.as_slice(), *value_burned),
            OutputData::Standard { address, value, .. } if *address == super::names::BURN => (&[][..], *value),
            _ => continue,
        };
        match out.iter_mut().find(|(p, _)| p == payload) {
            Some((_, v)) => *v = v.saturating_add(value),
            None => out.push((payload.to_vec(), value)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payloads_round_trip_and_wrong_subjects_are_not_backing() {
        let b = Backing::name("Nick.mds", "for the hub").unwrap();
        assert_eq!(b.subject_text(), "nick.mds");
        assert_eq!(Backing::parse(&b.payload()), Some(b.clone()));
        assert_eq!(b.payload().len(), HEAD + 11);
        let long = "x".repeat(LABEL_MAX);
        let full = Backing::new(Purpose::NOTARY, [7; 32], &long).unwrap();
        assert_eq!(full.payload().len(), MAX_BURN_DATA_SIZE);
        assert!(Backing::new(Purpose::NOTARY, [7; 32], &format!("{long}x")).is_err());
        // A name subject must be a canonical name; an account's padding, zero.
        let mut bad = b.payload();
        bad[5] = b'A';
        assert_eq!(Backing::parse(&bad), None);
        assert!(Backing::new(Purpose::NAME, [b'a'; 32], "").is_ok());
        assert!(Backing::new(Purpose::NAME, { let mut s = [0; 32]; s[3] = b'a'; s }, "").is_err());
        assert!(Backing::new(Purpose::EVM, [1; 32], "").is_err());
        let e = Backing::evm(&[0xab; 20], "").unwrap();
        assert_eq!(e.subject_text(), format!("0x{}", "ab".repeat(20)));
        assert_eq!(parse_subject(Purpose::EVM, &e.subject_text()).unwrap(), e.subject);
        assert_eq!(parse_subject(Purpose::NAME, "nick").unwrap(), b.subject);
        assert_eq!(parse_subject(Purpose::KEY, &"cd".repeat(32)).unwrap(), [0xcd; 32]);
        // Unknown purposes are kept; zero is not one.
        assert!(Backing::parse(&Backing { purpose: Purpose(42), subject: [1; 32], label: String::new() }.payload()).is_some());
        assert!(Backing::parse(&Backing { purpose: Purpose(0), subject: [1; 32], label: String::new() }.payload()).is_none());
        assert_eq!(Purpose::parse("notary-sha256"), Some(Purpose::NOTARY_SHA256));
        assert_eq!(Purpose::parse("purpose-42"), Some(Purpose(42)));
        assert!(Purpose::ALL.iter().all(|p| Purpose::parse(&p.word()) == Some(*p)));
    }

    #[test]
    fn burns_are_weighed_in_blocks_and_halve_each_year() {
        let y = HALF_LIFE;
        // A block's reward weighs one block in any year.
        assert_eq!(weight(block_reward(10), 10), 1.0);
        assert_eq!(weight(block_reward(3 * y + 5), 3 * y + 5), 1.0);
        assert_eq!(cost(1_440, 2 * y), 1_440 * block_reward(2 * y));
        // A year on, half.
        assert!((weight_at(cost(100, 7), 7, 7 + y) - 50.0).abs() < 1e-9);
        // Anchors give the same backing, and keep the order as time passes.
        let a = anchor(cost(100, 7), 7) + anchor(cost(10, y + 9), y + 9);
        let direct = weight_at(cost(100, 7), 7, 2 * y) + weight_at(cost(10, y + 9), y + 9, 2 * y);
        assert!((backing_at(a, 2 * y) - direct).abs() < 1e-9 * direct);
        assert!(anchor(cost(10, y), y) > anchor(cost(10, 0), 0));
        assert_eq!(tier(59.0), None);
        assert_eq!(tier(1_440.0), Some("well backed"));
        assert!(upkeep(100.0) > 69.0 && upkeep(100.0) < 70.0);
        assert_eq!(issued(0, 10), 10 * block_reward(0));
    }

    #[test]
    fn a_transactions_burns_are_summed_by_payload_and_sorted_by_kind() {
        let b = Backing::name("nick", "").unwrap();
        let mut outs = b.outputs(5 << 30);
        assert_eq!(outs.len(), 2);
        assert!(outs.iter().all(|o| o.value().is_power_of_two()));
        outs.push(OutputData::DataBurn { payload: b"MSN1nick".to_vec(), value_burned: 8 });
        outs.push(OutputData::Standard { address: super::super::names::BURN, value: 4, salt: [0; 32] });
        outs.push(OutputData::Standard { address: [1; 32], value: 2, salt: [0; 32] });
        let burns = burns_of(&outs);
        assert_eq!(burns, vec![(b.payload(), 5 << 30), (b"MSN1nick".to_vec(), 8), (vec![], 4)]);
        assert_eq!(Kind::of(&burns[0].0), Kind::Backing);
        assert_eq!(Kind::of(&burns[1].0), Kind::Names);
        assert_eq!(Kind::of(b"MDXA...."), Kind::Dex);
        assert_eq!(Kind::of(&[]), Kind::Other);
    }
}
