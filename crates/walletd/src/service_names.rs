//! The walletd actor's side of MSN-1 names (`midstate::core::names`): paying
//! `name.mds`; registering, renewing, selling and renting out names this
//! wallet holds; buying and renting others'.
//!
//! Each name this wallet holds has a control key, which signs changes and
//! nothing else, and payment keys in its records. The wallet keeps a name
//! working without being asked: a payment key running out of signatures is
//! swapped for a fresh one (and its coins swept as they arrive), the control
//! key hands over to a fresh one before it runs out, its own records come back
//! when a rental ends, and — unless turned off — it renews a year at a time
//! when 60 days are left, burning the year's price.
//!
//! Every change spends the registry's one thread, so a change can lose to
//! another landing first; the wallet then tries again the next block, until
//! the chain shows what it wants or someone else holds the name.
//!
//! A child of `service`, sharing the actor's state, as `service_rental` is.

use super::{now_secs, Internal, SendMeta, Service, DEFAULT_MSS_HEIGHT};
use crate::api::{MarketView, NameAction, NameCard, NameEdit, NameRecordView, NameView, SendStage, SoldView, WalletEvent};
use crate::{rental, sendplan};
use anyhow::{anyhow, bail, Context, Result};
use midstate::core::names::{self, Deal, Rec};
use midstate::core::plans::InheritancePlan;
use midstate::core::{compute_address, compute_commitment, decompose_value, wots, OutputData, Transaction};
use midstate::storage::name_index::{self, How, Record};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Changes a name may try before the wallet stops retrying.
const MAX_TRIES: u32 = 8;
/// A payment key with this few signatures left is swapped out; what it holds
/// and still receives is swept with the rest.
const PAY_RESERVE: u64 = 64;
/// A control key with this few left hands the name to a fresh one.
const CONTROL_RESERVE: u64 = 16;
/// Senders refuse a name that lapses this soon (a day): its holder is late.
const LAPSING: u64 = 1_440;
/// Auto-renewal buys a year when this little (60 days) is left.
const RENEW_AHEAD: u64 = 86_400;
const DAY: u64 = 1_440;

/// The state a name should reach on chain, and how: hold (register,
/// reclaim, or change it as its owner), buy or rent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Want {
    #[serde(with = "crate::plans::hex32")]
    pub owner: [u8; 32],
    #[serde(with = "hex_bytes")]
    pub records: Vec<u8>,
    #[serde(with = "hex_bytes")]
    pub terms: Vec<u8>,
    /// Years it pays for, burning its tier's rate for each.
    pub years: u64,
    /// Reached once the name is held at least this long.
    pub expires: u64,
    pub how: String,
}

impl Want {
    fn set(&self) -> names::Set {
        names::Set { owner: self.owner, records: self.records.clone(), terms: self.terms.clone(), years: self.years, expires: self.expires }
    }
}

fn yes() -> bool {
    true
}

/// A name this wallet holds or is getting. Persisted.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct NameJob {
    pub name: String,
    /// The control key: this wallet's, once the name is held.
    #[serde(with = "crate::plans::hex32")]
    pub owner: [u8; 32],
    pub want: Option<Want>,
    /// The commit in flight, if any.
    #[serde(default, with = "opt_hex")]
    pub commitment: Option<[u8; 32]>,
    /// registering, registered, updating, taken, lost or failed.
    pub status: String,
    pub note: String,
    /// Tries at the current change, and the height of the last: one per block.
    pub tries: u32,
    pub tried_at: u64,
    /// Payment keys swapped out, swept as coins reach them.
    #[serde(default)]
    pub retired: Vec<String>,
    #[serde(default = "yes")]
    pub auto_renew: bool,
    /// This wallet rents the name; `owner` is its holder's key.
    #[serde(default)]
    pub renter: bool,
    /// The records to put back when a rental of it ends.
    #[serde(default, with = "opt_bytes")]
    pub home: Option<Vec<u8>>,
    /// Watching for it: claimed for `years` the block it is free, or bought if
    /// offered for at most `max_price`.
    #[serde(default)]
    pub watch: Option<Watch>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct Watch {
    pub years: u64,
    pub max_price: Option<u64>,
}

impl NameJob {
    pub(super) fn done(&self) -> bool {
        matches!(self.status.as_str(), "taken" | "lost" | "failed" | "ended" | "given")
    }

    fn how(&self) -> How {
        match self.want.as_ref().map(|w| w.how.as_str()) {
            Some("buy") => How::Buy,
            Some("rent") => How::Rent,
            _ => How::Hold(Some(self.owner)),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct NameBook {
    pub jobs: Vec<NameJob>,
    /// The commit making the registry's genesis coin, while this wallet starts it.
    #[serde(default, with = "opt_hex")]
    pub genesis: Option<[u8; 32]>,
    /// The height this wallet last looked for names its keys hold.
    #[serde(skip)]
    pub scanned_at: u64,
}

impl NameBook {
    fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("names")
    }

    fn load(wallet_path: &Path) -> Self {
        std::fs::read(Self::path(wallet_path)).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default()
    }

    fn save(&self, wallet_path: &Path) -> Result<()> {
        let path = Self::path(wallet_path);
        let tmp = path.with_extension("names.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?).context("writing name records")?;
        std::fs::rename(&tmp, &path).context("saving name records")
    }

    pub(super) fn by_commit(&self, c: &[u8; 32]) -> Option<&NameJob> {
        self.jobs.iter().find(|j| j.commitment.as_ref() == Some(c))
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut NameJob> {
        self.jobs.iter_mut().find(|j| j.name == name)
    }
}

mod opt_hex {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<[u8; 32]>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(b) => s.serialize_some(&hex::encode(b)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[u8; 32]>, D::Error> {
        let s: Option<String> = Option::deserialize(d)?;
        s.map(|s| hex::decode(s).ok().and_then(|b| b.try_into().ok()).ok_or_else(|| serde::de::Error::custom("bad hex")))
            .transpose()
    }
}

mod opt_bytes {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(b) => s.serialize_some(&hex::encode(b)),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
        let s: Option<String> = Option::deserialize(d)?;
        s.map(|s| hex::decode(s).map_err(serde::de::Error::custom)).transpose()
    }
}

mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(v))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        hex::decode(String::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

/// An amount for people.
fn amount(units: u64) -> String {
    crate::bond::gmds_text(units)
}

fn deal_text(d: &Deal) -> String {
    match d {
        Deal::None => String::new(),
        Deal::Sale { price, .. } => format!("for sale at {}", amount(*price)),
        Deal::Rent { price, span, .. } => format!("for rent at {} per {} days", amount(*price), span / DAY),
        Deal::Rented { until, .. } => format!("rented out until block {until}"),
    }
}

/// Whether the chain shows what `want` asks.
/// A name's records, with an `mds-address` that is one of `plans`' moved to
/// the address a check-in would move the plan's coins to now, whenever the
/// deadline it points at is less than a full period away: a payment to a
/// plan's address whose deadline has passed is its heirs' to claim at once,
/// so a name kept on a live plan never sends money there. The moved
/// address's script is published in place of the old one's. A record that
/// points at one of `closed` (a plan this wallet has ended) goes back to
/// `own`, this wallet's own key. `None`: nothing to move.
pub(super) fn follow_plans(recs: &[Rec], active: &[InheritancePlan], closed: &[InheritancePlan], tip: u64, own: impl FnOnce() -> Result<Rec>) -> Result<Option<Vec<Rec>>> {
    let at = |plans: &[InheritancePlan], a: &[u8; 32]| {
        plans.iter().find_map(|p| p.epochs_to(tip).find(|k| &p.address(*k) == a).map(|k| (p.clone(), k)))
    };
    let Some(i) = recs.iter().position(|r| matches!(r, Rec::MdsAddress(_))) else { return Ok(None) };
    let Rec::MdsAddress(a) = recs[i] else { return Ok(None) };
    let (head, script) = if let Some((p, k)) = at(active, &a) {
        let next = p.epoch_after(tip);
        if k >= next {
            return Ok(None);
        }
        (Rec::MdsAddress(p.address(next)), names::script_records(&p.script(next)))
    } else if at(closed, &a).is_some() {
        (own()?, Vec::new())
    } else {
        return Ok(None);
    };
    let mut out: Vec<Rec> = recs[..i].iter().filter(|r| !matches!(r, Rec::Script(_))).cloned().collect();
    out.push(head);
    out.extend(script);
    out.extend(recs[i + 1..].iter().filter(|r| !matches!(r, Rec::Script(_))).cloned());
    Ok(Some(out))
}

fn reached(r: &Record, want: &Want) -> bool {
    r.owner == want.owner
        && r.expires >= want.expires
        && r.terms == want.terms
        && names::encode(&r.records).ok().as_ref() == Some(&want.records)
}

impl Service {
    pub(super) fn names_load(&mut self) {
        self.names = NameBook::load(&self.wallet_path);
    }

    fn names_save(&self) {
        if let Err(e) = self.names.save(&self.wallet_path) {
            tracing::warn!("{e:#}");
        }
    }

    fn signatures_left(&self, key: &[u8; 32]) -> Option<u64> {
        self.wallet.as_ref()?.mss_keys().iter().find(|k| &k.master_pk == key).map(|k| k.remaining())
    }

    /// A fresh MSS key of this wallet.
    fn fresh_key(&mut self, label: String) -> Result<[u8; 32]> {
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let address = w.generate_mss(DEFAULT_MSS_HEIGHT, Some(label))?;
        w.mss_keys().iter().find(|k| compute_address(&k.master_pk) == address).map(|k| k.master_pk).ok_or_else(|| anyhow!("the new key is missing"))
    }

    /// A name as the chain has it now, refused if it cannot take a payment.
    pub(super) fn live_record(&self, to: &str) -> Result<Record> {
        let shown = to.trim().to_ascii_lowercase();
        let r = name_index::lookup(self.node.storage.db(), to)?.ok_or_else(|| anyhow!("{shown} is not registered"))?;
        if r.lapsed() {
            bail!("{shown} has lapsed: its holder did not renew it");
        }
        if r.expires < r.height + LAPSING {
            bail!("{shown} lapses within a day; ask its holder for an address instead");
        }
        Ok(r)
    }

    /// An address, or the address a name's records pay now.
    pub(super) fn resolve_destination(&self, to: &str) -> Result<[u8; 32]> {
        if !names::is_name(to) {
            return sendplan::decode_address(to);
        }
        let r = self.live_record(to)?;
        let used = |k: &[u8; 32]| self.node.storage.query_mss_leaf_index(k).unwrap_or(0);
        names::pays(&r.records, used).ok_or_else(|| anyhow!("{} has no MDS address that can take a payment", to.trim()))
    }

    /// A midwimble address, or the one a name's records give.
    pub(super) fn resolve_mdw(&self, to: &str) -> Result<String> {
        if !names::is_name(to) {
            return Ok(to.trim().to_string());
        }
        let r = self.live_record(to)?;
        r.records
            .iter()
            .find_map(|x| if let Rec::Mdw(a) = x { Some(a.clone()) } else { None })
            .ok_or_else(|| anyhow!("{} takes no MDW", to.trim()))
    }

    pub(super) fn name_views(&mut self) -> Vec<NameView> {
        let db = self.node.storage.db().clone();
        let jobs = self.names.jobs.clone();
        let mut out = Vec::with_capacity(jobs.len());
        for j in jobs {
            let chain = name_index::lookup(&db, &j.name).ok().flatten();
            let ours = |r: &&Record| r.owner == j.owner || j.want.as_ref().is_some_and(|w| w.owner == r.owner);
            let r = chain.as_ref().filter(ours);
            let records = r.map(|r| r.records.clone()).unwrap_or_default();
            let has = |f: fn(&Rec) -> bool| records.iter().any(f);
            let pays = r.and_then(|r| names::pays(&r.records, |k| self.node.storage.query_mss_leaf_index(k).unwrap_or(0)));
            out.push(NameView {
                name: format!("{}{}", j.name, names::SUFFIX),
                status: j.status.clone(),
                note: j.note.clone(),
                pays: pays.map(|a| midstate::core::encode_address_with_checksum(&a)),
                pays_kind: pays.map(|a| self.kind_label(&records, &a)).unwrap_or_default(),
                sealed: has(|x| matches!(x, Rec::Kem(_))),
                backing: self.name_backing(&j.name),
                expires: r.map_or(0, |r| r.expires),
                tier: names::TIERS[chain.as_ref().map_or(names::tier_of(&j.name), |r| r.tier) as usize].into(),
                deal: r.map(|r| deal_text(&r.deal)).unwrap_or_default(),
                renter: j.renter,
                auto_renew: j.auto_renew,
                records: records.iter().map(|x| x.describe()).map(|(k, v)| NameRecordView { kind: k.into(), value: v }).collect(),
                mdw: has(|x| matches!(x, Rec::Mdw(_))),
                identity: has(|x| matches!(x, Rec::Identity(_))),
                evm: has(|x| matches!(x, Rec::Evm(_))),
                text: records.iter().find_map(|x| if let Rec::Text(t) = x { Some(t.clone()) } else { None }).unwrap_or_default(),
                pay_to: records
                    .iter()
                    .find_map(|x| if let Rec::MdsAddress(a) = x { Some(midstate::core::encode_address_with_checksum(a)) } else { None })
                    .unwrap_or_default(),
            });
        }
        out
    }

    /// What kind of address `address` is, for people: what the records
    /// publish, or what this wallet knows of its own plans.
    pub(super) fn kind_label(&self, recs: &[Rec], address: &[u8; 32]) -> String {
        if let Some(k) = names::address_kind(recs, address) {
            return k.label();
        }
        if let Some(script) = self.own_script(address) {
            return format!("{} (yours)", midstate::core::address_kind::classify(&script).label());
        }
        "script address (kind not published)".into()
    }

    /// This wallet's inheritance plans: those it keeps up, and those it has
    /// closed.
    fn own_plans(&self) -> (Vec<InheritancePlan>, Vec<InheritancePlan>) {
        let (mut active, mut closed) = (Vec::new(), Vec::new());
        for o in &self.plans.owned {
            let Some(p) = hex::decode(&o.plan).ok().and_then(|b| InheritancePlan::from_bytes(&b)) else { continue };
            if o.status == "closed" { closed.push(p) } else { active.push(p) }
        }
        (active, closed)
    }

    /// The script of an address of one of this wallet's inheritance plans,
    /// at any deadline it has reached or will reach next.
    fn own_script(&self, address: &[u8; 32]) -> Option<Vec<u8>> {
        let tip = name_index::meta(self.node.storage.db()).map_or(0, |m| m.next);
        self.plans.owned.iter().filter(|p| p.status == "active").find_map(|p| {
            let plan = InheritancePlan::from_bytes(&hex::decode(&p.plan).ok()?)?;
            plan.epochs_to(tip).find(|k| &plan.address(*k) == address).map(|k| plan.script(k))
        })
    }

    /// This wallet's active plans, as `(label, address, script)` at the
    /// deadline a check-in now would move them to: what a name can point at.
    pub(super) fn plan_targets(&self) -> Vec<(String, [u8; 32], Vec<u8>)> {
        let tip = name_index::meta(self.node.storage.db()).map_or(0, |m| m.next);
        self.plans
            .owned
            .iter()
            .filter(|p| p.status == "active")
            .filter_map(|p| {
                let plan = InheritancePlan::from_bytes(&hex::decode(&p.plan).ok()?)?;
                let k = plan.epoch_after(tip);
                Some((p.label.clone(), plan.address(k), plan.script(k)))
            })
            .collect()
    }

    /// What a name is now, what it costs and what this wallet could do with it.
    pub(super) fn find_name(&mut self, name: &str, years: u64) -> Result<NameCard> {
        let name = names::normalize(name)?;
        let db = self.node.storage.db();
        let tip = name_index::meta(db)?.next;
        let r = name_index::lookup(db, &name)?;
        let years = years.clamp(1, names::MAX_YEARS);
        let mine: HashSet<[u8; 32]> = self.wallet.as_ref().map(|w| w.mss_keys().iter().map(|k| k.master_pk).collect()).unwrap_or_default();
        let mut card = NameCard {
            name: format!("{name}{}", names::SUFFIX),
            years,
            tip,
            fee: sendplan::required_fee(3, 4),
            ..NameCard::default()
        };
        let tier = match &r {
            None => {
                card.state = "available".into();
                names::tier_of(&name)
            }
            Some(r) => {
                card.since = r.since;
                card.records = r.records.iter().map(|x| x.describe()).map(|(k, v)| NameRecordView { kind: k.into(), value: v }).collect();
                if !r.lapsed() {
                    let pays = names::pays(&r.records, |k| self.node.storage.query_mss_leaf_index(k).unwrap_or(0));
                    card.pays = pays.map(|a| midstate::core::encode_address_with_checksum(&a));
                    card.pays_kind = pays.map(|a| self.kind_label(&r.records, &a)).unwrap_or_default();
                    card.identity = names::identity(&r.records).is_some();
                    card.sealed = names::kem_hash(&r.records).is_some();
                    card.postage = names::postage(&r.records).map_or(midstate::sealed::STAMP_BITS, u32::from);
                }
                card.state = match r.deal {
                    _ if r.lapsed() && r.owner == [0; 32] => "unclaimed",
                    _ if r.lapsed() => "lapsed",
                    _ if mine.contains(&r.owner) => "yours",
                    Deal::Sale { price, .. } => {
                        card.price = price;
                        "for sale"
                    }
                    Deal::Rent { price, span, .. } => {
                        card.price = price;
                        card.rent_days = span / DAY;
                        card.rent_until = (tip + span).min(r.expires.saturating_sub(names::GRACE));
                        "for rent"
                    }
                    Deal::Rented { .. } => "rented",
                    Deal::None => "held",
                }
                .into();
                card.expires = r.expires;
                r.tier
            }
        };
        card.tier = names::TIERS[tier as usize].into();
        card.backing = self.name_backing(&name);
        card.rate = names::RATES[tier as usize];
        if matches!(card.state.as_str(), "available" | "unclaimed" | "lapsed") {
            card.burn = names::cost(tier, years);
            card.expires = tip + years * names::YEAR;
        }
        Ok(card)
    }

    /// The latest sales and rentals.
    pub(super) fn name_deals(&self) -> Result<Vec<SoldView>> {
        Ok(name_index::recent_deals(self.node.storage.db(), 20)?
            .into_iter()
            .map(|d| {
                let (kind, price) = match d.deal {
                    Deal::Sale { price, .. } => ("sale", price),
                    Deal::Rent { price, .. } => ("rent", price),
                    _ => ("", 0),
                };
                SoldView { name: format!("{}{}", d.name, names::SUFFIX), kind: kind.into(), price, height: d.height }
            })
            .collect())
    }

    /// Watch a name: claim it for `years` the block it is free, or buy it if
    /// it is offered for at most `max_price`.
    pub(super) fn watch_name(&mut self, name: &str, years: u64, max_price: Option<u64>) -> Result<()> {
        self.ready_to_spend()?;
        let name = names::normalize(name)?;
        if !(1..=names::MAX_YEARS).contains(&years) {
            bail!("claim it for 1 to {} years", names::MAX_YEARS);
        }
        if self.names.jobs.iter().any(|j| j.name == name && !j.done()) {
            bail!("{name}{} is already this wallet's, or on its way", names::SUFFIX);
        }
        self.names.jobs.retain(|j| j.name != name);
        self.names.jobs.push(NameJob {
            name,
            owner: [0; 32],
            want: None,
            commitment: None,
            status: "watching".into(),
            note: match max_price {
                Some(p) => format!("claims it the block it is free, or buys it if offered for {} or less", amount(p)),
                None => "claims it the block it is free".into(),
            },
            tries: 0,
            tried_at: 0,
            retired: Vec::new(),
            auto_renew: true,
            renter: false,
            home: None,
            watch: Some(Watch { years, max_price }),
        });
        self.names_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    pub(super) fn unwatch_name(&mut self, name: &str) -> Result<()> {
        let name = names::normalize(name)?;
        let before = self.names.jobs.len();
        self.names.jobs.retain(|j| !(j.name == name && j.status == "watching"));
        if self.names.jobs.len() == before {
            bail!("not watching {name}{}", names::SUFFIX);
        }
        self.names_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Names offered for sale or rent now.
    pub(super) fn name_market(&self) -> Result<Vec<MarketView>> {
        Ok(name_index::listings(self.node.storage.db())?
            .into_iter()
            .map(|r| {
                let (kind, price, span) = match r.deal {
                    Deal::Sale { price, .. } => ("sale", price, 0),
                    Deal::Rent { price, span, .. } => ("rent", price, span),
                    _ => ("", 0, 0),
                };
                MarketView {
                    name: format!("{}{}", r.name, names::SUFFIX),
                    tier: names::TIERS[r.tier as usize].into(),
                    kind: kind.into(),
                    price,
                    span,
                    expires: r.expires,
                    mine: self.names.jobs.iter().any(|j| j.name == r.name && j.owner == r.owner && !j.renter),
                    backing: self.name_backing(&r.name).map_or(0.0, |b| b.blocks),
                }
            })
            .collect())
    }

    /// The records an edit asks for. Payments go to this wallet's own keys
    /// (kept as they are), or to `pay_to` — a vault, multisig, plan or shared
    /// wallet — with its script published when it is known, so anyone can see
    /// what kind of address the name pays. Unknown tags are kept.
    fn edited(&mut self, base: &[Rec], e: &NameEdit, shown: &str) -> Result<Vec<u8>> {
        let mut recs: Vec<Rec> = Vec::new();
        let pay_to = e.pay_to.trim();
        if pay_to.is_empty() {
            recs.extend(base.iter().filter(|r| matches!(r, Rec::MdsKey { .. })).cloned());
            if recs.is_empty() {
                recs.push(Rec::MdsKey { key: self.fresh_key(shown.to_string())?, height: DEFAULT_MSS_HEIGHT as u8 });
            }
        } else {
            let addr = sendplan::decode_address(pay_to)?;
            recs.push(Rec::MdsAddress(addr));
            let typed = e.script.trim();
            let script = if !typed.is_empty() {
                let b = hex::decode(typed).map_err(|_| anyhow!("the script is not hex"))?;
                if midstate::core::hash(&b) != addr {
                    bail!("that script is not the script of {pay_to}: it hashes to another address");
                }
                if b.len() > midstate::core::script::MAX_SCRIPT_SIZE {
                    bail!("a script is at most {} bytes", midstate::core::script::MAX_SCRIPT_SIZE);
                }
                Some(b)
            } else {
                self.own_script(&addr).or_else(|| names::published_script(base).filter(|b| midstate::core::hash(b) == addr))
            };
            if let Some(b) = script {
                recs.extend(names::script_records(&b));
            }
        }
        // Kept as they are: tags this wallet does not know, and (below) the
        // price strangers pay, which is set from Messages.
        recs.extend(base.iter().filter(|r| matches!(r, Rec::Other { .. })).cloned());
        if e.sealed {
            recs.push(Rec::Kem(self.kem_record()?));
        }
        if e.mdw {
            let a = self.mwb.keys.as_ref().map(|k| k.address().encode()).ok_or_else(|| anyhow!("this wallet has no MDW account yet"))?;
            recs.push(Rec::Mdw(a));
        }
        if e.identity {
            recs.push(Rec::Identity(self.ensure_identity()?));
        }
        if e.evm {
            recs.push(Rec::Evm(self.evm_key()?.address));
        }
        if !e.text.trim().is_empty() {
            recs.push(Rec::Text(e.text.trim().to_string()));
        }
        recs.extend(base.iter().filter(|r| matches!(r, Rec::Postage(_))).cloned());
        names::encode(&recs)
    }

    /// Records for a name this wallet takes: a payment key of its own, this
    /// wallet's MDW address if it has one, its identity (so payers can ask it
    /// for a fresh address) and its KEM key's hash (so anyone can send it a
    /// sealed message). Edit records takes any of them off.
    fn fresh_records(&mut self, shown: &str) -> Result<Vec<u8>> {
        // The identity first: in a wallet with no MSS key yet it gets one of
        // its own, rather than becoming this name's payment key.
        let identity = self.ensure_identity().ok();
        let pay = self.fresh_key(shown.to_string())?;
        let mut recs = vec![Rec::MdsKey { key: pay, height: DEFAULT_MSS_HEIGHT as u8 }];
        if let Some(a) = self.mwb.keys.as_ref().map(|k| k.address().encode()) {
            recs.push(Rec::Mdw(a));
        }
        if let Some(id) = identity {
            recs.push(Rec::Identity(id));
        }
        if let Ok(h) = self.kem_record() {
            recs.push(Rec::Kem(h));
        }
        names::encode(&recs)
    }

    pub(super) fn ready_to_spend(&self) -> Result<()> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — try again once it has caught up");
        }
        self.ensure_restore_done()
    }

    /// Start a job for `name` and make its first try now; a busy registry is
    /// not a failure: the tick tries again next block.
    async fn start_job(&mut self, job: NameJob) -> Result<String> {
        let name = job.name.clone();
        self.names.jobs.retain(|j| j.name != name);
        self.names.jobs.push(job);
        self.names_save();
        if let Err(e) = self.commit_name(&name).await {
            self.name_note(&name, e);
        }
        Ok(format!("{name}{}", names::SUFFIX))
    }

    /// Get `name` for this wallet for `years`: a control key and a payment
    /// key of its own, and this wallet's MDW address if it has one.
    pub(super) async fn register_name(&mut self, name: &str, years: u64) -> Result<String> {
        self.ready_to_spend()?;
        let name = names::normalize(name)?;
        let shown = format!("{name}{}", names::SUFFIX);
        if !(1..=names::MAX_YEARS).contains(&years) {
            bail!("a name is registered for 1 to {} years", names::MAX_YEARS);
        }
        if self.names.jobs.iter().any(|j| j.name == name && !j.done()) {
            bail!("{shown} is already this wallet's, or on its way");
        }
        let db = self.node.storage.db().clone();
        let m = name_index::meta(&db)?;
        if let Some(r) = name_index::lookup(&db, &name)? {
            if !r.lapsed() {
                bail!("{shown} is taken");
            }
        }
        let owner = self.fresh_key(format!("{shown} control"))?;
        let records = self.fresh_records(&shown)?;
        let want = Want { owner, records, terms: Deal::None.terms(), years, expires: m.next + years * names::YEAR, how: "hold".into() };
        self.start_job(NameJob {
            name,
            owner,
            want: Some(want),
            commitment: None,
            status: "registering".into(),
            note: String::new(),
            tries: 0,
            tried_at: 0,
            retired: Vec::new(),
            auto_renew: true,
            renter: false,
            home: None,
            watch: None,
        })
        .await
    }

    /// Buy a name offered for sale, or rent one offered for rent.
    pub(super) async fn take_name(&mut self, name: &str, rent: bool) -> Result<String> {
        self.ready_to_spend()?;
        let name = names::normalize(name)?;
        let shown = format!("{name}{}", names::SUFFIX);
        if self.names.jobs.iter().any(|j| j.name == name && !j.done()) {
            bail!("{shown} is already this wallet's, or on its way");
        }
        let r = name_index::lookup(self.node.storage.db(), &name)?.ok_or_else(|| anyhow!("{shown} is not registered"))?;
        let records = self.fresh_records(&shown)?;
        let (job_owner, want, renter) = match (rent, r.deal) {
            (false, Deal::Sale { .. }) => {
                let owner = self.fresh_key(format!("{shown} control"))?;
                (owner, Want { owner, records: records.clone(), terms: Deal::None.terms(), years: 0, expires: r.expires, how: "buy".into() }, false)
            }
            (true, Deal::Rent { price, payee, span }) => {
                let until = (r.height + span).min(r.expires.saturating_sub(names::GRACE));
                if until <= r.height {
                    bail!("{shown} lapses too soon to rent");
                }
                let terms = Deal::Rented { price, payee, until }.terms();
                (r.owner, Want { owner: r.owner, records: records.clone(), terms, years: 0, expires: r.expires, how: "rent".into() }, true)
            }
            _ => bail!("{shown} is not offered that way"),
        };
        self.start_job(NameJob {
            name,
            owner: job_owner,
            want: Some(want),
            commitment: None,
            status: if renter { "renting" } else { "buying" }.into(),
            note: String::new(),
            tries: 0,
            tried_at: 0,
            retired: Vec::new(),
            auto_renew: true,
            renter,
            // A renter's own records, to tell while the rental is still its.
            home: renter.then_some(records),
            watch: None,
        })
        .await
    }

    /// Change a name this wallet holds: its records, an offer, a renewal.
    pub(super) async fn name_action(&mut self, name: &str, act: NameAction) -> Result<()> {
        let name = names::normalize(name)?;
        let shown = format!("{name}{}", names::SUFFIX);
        let job = self.names.jobs.iter().find(|j| j.name == name && !j.renter && !j.done()).cloned();
        let job = job.ok_or_else(|| anyhow!("{shown} is not this wallet's"))?;
        if let NameAction::AutoRenew(on) = act {
            if let Some(j) = self.names.get_mut(&name) {
                j.auto_renew = on;
            }
            self.names_save();
            return Ok(());
        }
        if job.status != "registered" {
            bail!("{shown} is changing already; wait for that to land");
        }
        let db = self.node.storage.db().clone();
        let r = name_index::lookup(&db, &name)?.filter(|r| r.owner == job.owner).ok_or_else(|| anyhow!("the chain does not show it as this wallet's"))?;
        if let Deal::Rented { until, .. } = r.deal {
            if r.height < until {
                bail!("{shown} is rented out until block {until}");
            }
        }
        let records = names::encode(&r.records)?;
        let mut want = Want { owner: job.owner, records, terms: r.terms.clone(), years: 0, expires: r.expires, how: "hold".into() };
        let note = match act {
            NameAction::Edit(e) => {
                want.records = self.edited(&r.records, &e, &shown)?;
                "saving its records"
            }
            NameAction::Sell { price } => {
                let payee = compute_address(&self.fresh_key(format!("sale of {shown}"))?);
                want.terms = Deal::Sale { price, payee }.terms();
                "offering it for sale"
            }
            NameAction::RentOut { price, days } => {
                let payee = compute_address(&self.fresh_key(format!("rent of {shown}"))?);
                want.terms = Deal::Rent { price, payee, span: days.max(1) * DAY }.terms();
                if let Some(j) = self.names.get_mut(&name) {
                    j.home = Some(want.records.clone());
                }
                "offering it for rent"
            }
            NameAction::Unlist => {
                want.terms = Deal::None.terms();
                "taking it off the market"
            }
            NameAction::Postage { bits } => {
                want.records = names::encode(&super::ident_glue::with_postage(&r.records, bits)?)?;
                "changing what strangers pay to write to it"
            }
            NameAction::Renew { years } => {
                if !(1..=names::MAX_YEARS).contains(&years) {
                    bail!("renew for 1 to {} years", names::MAX_YEARS);
                }
                want.years = years;
                want.expires = r.expires.max(r.height) + years * names::YEAR;
                "renewing it"
            }
            NameAction::Give { to } => {
                let to = to.trim();
                let key: [u8; 32] = if names::is_name(to) {
                    let r = name_index::lookup(&db, to)?.filter(|r| !r.lapsed()).ok_or_else(|| anyhow!("{to} is not held by anyone"))?;
                    r.owner
                } else {
                    hex::decode(to).ok().and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("give it to a name.mds, or a 64-character control key"))?
                };
                if key == job.owner || key == [0; 32] {
                    bail!("that is not another wallet's key");
                }
                // The new holder sets its own records.
                want.owner = key;
                want.records = names::encode(&[])?;
                want.terms = Deal::None.terms();
                want.how = "give".into();
                "giving it away"
            }
            NameAction::AutoRenew(_) => unreachable!(),
        };
        self.want_change(&name, want, note);
        if let Err(e) = self.commit_name(&name).await {
            self.name_note(&name, e);
        }
        Ok(())
    }

    fn want_change(&mut self, name: &str, want: Want, note: &str) {
        if let Some(j) = self.names.get_mut(name) {
            j.want = Some(want);
            j.status = "updating".into();
            j.note = note.into();
            j.tries = 0;
            j.tried_at = 0;
        }
        self.names_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    /// Commit the change a name wants against the registry's current thread.
    async fn commit_name(&mut self, name: &str) -> Result<()> {
        let job = self.names.jobs.iter().find(|j| j.name == name).cloned().ok_or_else(|| anyhow!("no such name"))?;
        let want = job.want.clone().ok_or_else(|| anyhow!("nothing to change"))?;
        // Getting a name gives up; a held name's upkeep never does.
        if job.tries >= MAX_TRIES && job.status != "updating" {
            self.name_status(name, "failed", &format!("gave up after {MAX_TRIES} tries ({}); try again", job.note));
            bail!("gave up");
        }
        let state = self.node.get_state().await;
        if let Some(j) = self.names.get_mut(name) {
            j.tried_at = state.height;
        }
        let db = self.node.storage.db().clone();
        let m = name_index::meta(&db)?;
        if m.ready() && m.thread.is_none() {
            self.start_registry(&state).await?;
            bail!("starting the name registry: this is the chain's first name");
        }
        if let Some(j) = self.names.get_mut(name) {
            j.tries += 1;
        }
        self.names_save();
        let plan = name_index::plan(&db, name, job.how(), &want.set())?;
        if !state.coins.contains(&plan.thread.id) || self.in_flight_inputs().contains(&plan.thread.id) {
            bail!("another change to the registry is landing; the wallet tries again next block");
        }
        let to = format!("{name}{}", names::SUFFIX);
        let mut fixed = vec![plan.output.clone()];
        fixed.extend(names::burn_outputs(name, plan.burn));
        if let Some((price, payee)) = plan.pay {
            fixed.extend(decompose_value(price).into_iter().map(|value| OutputData::Standard { address: payee, value, salt: rand::random() }));
        }
        let commitment = self.commit_outputs(&state, fixed, Some(plan.thread.id), &to, format!("name: {to}")).await?;
        if let Some(j) = self.names.get_mut(name) {
            j.commitment = Some(commitment);
            j.note = "solving the commit's proof-of-work".into();
        }
        self.names_save();
        self.broadcast_commit(commitment).await?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Commit a send of `fixed` (then change), paid from this wallet, also
    /// spending `thread` if given. Returns the commitment, recorded as pending;
    /// the Send tab shows it as `what`.
    pub(super) async fn commit_outputs(
        &mut self,
        state: &midstate::core::State,
        fixed: Vec<OutputData>,
        thread: Option<[u8; 32]>,
        to: &str,
        what: String,
    ) -> Result<[u8; 32]> {
        self.verify_mss_indices().await?;
        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let live = w.spendable_now(state, &in_flight);
        // The thread is one more input to price, its witness about as big as two.
        let extra = if thread.is_some() { 2 } else { 0 };
        // What it burns and pays, then the fee its own size implies.
        let spend: u64 = fixed.iter().map(|o| o.value()).sum();
        let mut fee = 100u64;
        let (selected, change) = loop {
            let needed = spend.checked_add(fee).ok_or_else(|| anyhow!("too large"))?;
            let selected = w.select_coins(needed, &live)?;
            let in_sum: u64 = selected.iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum();
            if in_sum <= needed {
                bail!("not enough coins: this takes {needed} with the fee, and {in_sum} is spendable");
            }
            let need = sendplan::required_fee(selected.len() + extra, fixed.len() + decompose_value(in_sum - needed).len());
            if need <= fee {
                break (selected, in_sum - needed);
            }
            fee = need;
        };
        let mut outputs = fixed;
        let mut change_seeds = Vec::new();
        for value in decompose_value(change) {
            let seed = w.allocate_next_wots_seed()?;
            change_seeds.push((outputs.len(), seed));
            outputs.push(OutputData::Standard { address: compute_address(&wots::keygen(&seed)), value, salt: rand::random() });
        }
        let mut input_coin_ids = selected;
        input_coin_ids.extend(thread);
        let salt: [u8; 32] = rand::random();
        let hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let commitment = compute_commitment(&input_coin_ids, &hashes, &salt);
        let now = now_secs();
        w.data.pending.push(midstate::wallet::PendingCommit {
            commitment,
            salt,
            input_coin_ids,
            outputs,
            change_seeds,
            created_at: now,
            reveal_not_before: now + 5 + rand::random::<u64>() % 11,
            is_consolidate: false,
        });
        w.save()?;
        let detail = what;
        self.sends.insert(
            commitment,
            SendMeta { stage: SendStage::Committing, detail: detail.clone(), amount: spend, fee, to: to.into(), updated_at: now },
        );
        self.set_stage(commitment, SendStage::Committing, detail);
        Ok(commitment)
    }

    /// The chain's first name starts the registry: a plain send that makes
    /// its genesis coin. Once that is indexed, registrations go ahead.
    async fn start_registry(&mut self, state: &midstate::core::State) -> Result<()> {
        let pending = |c: &[u8; 32]| self.wallet.as_ref().is_some_and(|w| w.find_pending(c).is_some());
        if self.names.genesis.as_ref().is_some_and(pending) {
            return Ok(());
        }
        let c = self.commit_outputs(state, vec![names::genesis()], None, "the name registry", "name: the name registry".into()).await?;
        self.names.genesis = Some(c);
        self.names_save();
        self.broadcast_commit(c).await
    }

    fn name_status(&mut self, name: &str, status: &str, note: &str) {
        if let Some(j) = self.names.get_mut(name) {
            j.status = status.into();
            j.note = note.into();
            if j.done() {
                j.commitment = None;
                j.want = None;
            }
        }
        self.names_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    fn name_note(&mut self, name: &str, e: anyhow::Error) {
        if let Some(j) = self.names.get_mut(name).filter(|j| !j.done()) {
            j.note = format!("{e:#}");
        }
        self.names_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    /// The wallet's bookkeeping for a landed spend, through the actor's queue
    /// (the reveal monitor's way: calling it from here would recurse).
    fn confirm_later(&self, commitment: [u8; 32]) {
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            let _ = tx.send(super::Cmd::Internal(Internal::RevealConfirmed(commitment))).await;
        });
    }

    /// Drop a change's commit, freeing its coins.
    fn drop_name_commit(&mut self, commitment: [u8; 32], why: &str) {
        if let Some(w) = self.wallet.as_mut() {
            w.data.pending.retain(|p| p.commitment != commitment);
            let _ = w.save();
        }
        self.set_stage(commitment, SendStage::Failed, why.to_string());
    }

    /// Compare a name with what its job wants, and settle the job: reached,
    /// someone else's, or still to do (the next block tries again).
    async fn settle_name(&mut self, name: &str) {
        let Some(job) = self.names.jobs.iter().find(|j| j.name == name).cloned() else { return };
        let Ok(chain) = name_index::lookup(self.node.storage.db(), name) else { return };
        let first = job.commitment.and_then(|c| self.wallet.as_ref()?.find_pending(&c)).map(|p| p.input_coin_ids[0]);
        let landed = match first {
            Some(c) => !self.node.check_coin(c).await,
            None => false,
        };
        let commit = job.commitment;
        let finish = |me: &mut Self, why: &str| {
            if let Some(c) = commit {
                if landed {
                    me.confirm_later(c);
                } else {
                    me.drop_name_commit(c, why);
                }
            }
            if let Some(j) = me.names.get_mut(name) {
                j.commitment = None;
            }
        };
        match (chain, job.want.clone()) {
            (Some(r), Some(w)) if reached(&r, &w) => {
                finish(self, "the name already shows this change");
                if let Some(j) = self.names.get_mut(name) {
                    j.owner = w.owner;
                    j.want = None;
                }
                let status = if w.how == "give" { "given" } else if job.renter { "rented" } else { "registered" };
                self.name_status(name, status, if w.how == "give" { "given to another wallet" } else { "" });
            }
            // A take whose offer is gone: someone else took it, or it was withdrawn.
            (Some(r), Some(w)) if (w.how == "buy" && !matches!(r.deal, Deal::Sale { .. })) || (w.how == "rent" && !matches!(r.deal, Deal::Rent { .. })) => {
                finish(self, "the offer is gone; nothing was spent");
                self.name_status(name, "taken", "someone else took it first, or it is no longer offered");
            }
            (Some(r), _) if !job.renter && !r.lapsed() && r.owner != job.owner && job.want.as_ref().is_none_or(|w| w.owner != r.owner) => {
                finish(self, "someone else holds this name; nothing was spent");
                let lost = !matches!(job.status.as_str(), "registering" | "buying");
                self.name_status(name, if lost { "lost" } else { "taken" }, if lost { "it lapsed or was sold, and someone else holds it" } else { "someone else got it first" });
            }
            _ if commit.is_some() => {
                finish(self, "another change landed first; trying again");
                self.names_save();
            }
            _ => {}
        }
    }

    /// `try_reveal` for a name change: this wallet's coins signed here, the
    /// thread opened by a witness rebuilt from the index, which must still be
    /// for the thread and output the commit names.
    pub(super) async fn reveal_name(&mut self, commitment: [u8; 32], pending: midstate::wallet::PendingCommit) {
        let Some(job) = self.names.by_commit(&commitment).cloned() else { return };
        let Some(want) = job.want.clone() else { return self.settle_name(&job.name).await };
        let thread = *pending.input_coin_ids.last().expect("a change has inputs");
        let db = self.node.storage.db().clone();
        let plan = match name_index::plan(&db, &job.name, job.how(), &want.set()) {
            Ok(p) => p,
            Err(e) if name_index::meta(&db).is_ok_and(|m| !m.ready()) => {
                tracing::info!("name {}: waiting for the name index ({e:#})", job.name);
                return;
            }
            Err(_) => return self.settle_name(&job.name).await,
        };
        if plan.thread.id != thread || pending.outputs.first() != Some(&plan.output) || !self.node.check_coin(thread).await {
            return self.settle_name(&job.name).await;
        }
        let ours = pending.input_coin_ids[..pending.input_coin_ids.len() - 1].to_vec();
        // As `try_reveal`: never sign for a one-time key the chain has seen
        // sign something else.
        let reused: Vec<[u8; 32]> = {
            let w = self.wallet.as_ref().unwrap();
            let trees: HashSet<[u8; 32]> = w.mss_keys().iter().map(|m| compute_address(&m.master_pk)).collect();
            ours.iter()
                .filter_map(|id| w.find_coin(id))
                .filter(|c| !trees.contains(&c.address))
                .map(|c| c.address)
                .filter(|a| matches!(self.node.storage.get_spent_commitment(a), Ok(Some(prev)) if prev != commitment))
                .collect()
        };
        if !reused.is_empty() {
            let w = self.wallet.as_mut().unwrap();
            for c in w.data.coins.iter_mut().filter(|c| reused.contains(&c.address)) {
                c.wots_signed = true;
            }
            self.drop_name_commit(commitment, "refused to sign: a one-time key among its coins already signed something else");
            if let Some(j) = self.names.get_mut(&job.name) {
                j.commitment = None;
            }
            return self.names_save();
        }
        if let Err(e) = self.verify_mss_indices().await {
            return self.set_stage(commitment, SendStage::Failed, format!("could not sign: {e:#}. Nothing was spent."));
        }
        let w = self.wallet.as_mut().unwrap();
        // The control key signs a change of a name it holds; fee coins as usual.
        let control = match plan.signer {
            None => Ok(None),
            Some(pk) => match w.data.mss_keys.iter_mut().find(|k| k.master_pk == pk) {
                Some(k) if k.remaining() > 0 => k.sign(&commitment).map(|s| Some(s.to_bytes())),
                _ => Err(anyhow!("its control key has no signatures left")),
            },
        };
        let signed = control.and_then(|c| Ok((c, rental::sign_wallet_inputs(w, &ours, &commitment)?))).and_then(|s| w.save().map(|_| s));
        let (sig, signed) = match signed {
            Ok(s) => s,
            Err(e) => return self.set_stage(commitment, SendStage::Failed, format!("could not sign: {e:#}")),
        };
        let (mut inputs, mut witnesses): (Vec<_>, Vec<_>) = signed.into_iter().unzip();
        inputs.push(plan.input.clone());
        witnesses.push(plan.witness(sig));
        let tx = Transaction::Reveal { inputs, witnesses, outputs: pending.outputs.clone(), salt: pending.salt };
        match self.node.send_transaction(tx).await {
            Ok(()) => {
                self.set_stage(commitment, SendStage::RevealPending, "name change broadcast — waiting for a block");
                if let Some(j) = self.names.get_mut(&job.name) {
                    j.note = "waiting for a block".into();
                }
                self.names_save();
            }
            Err(_) if !self.node.check_coin(thread).await => self.settle_name(&job.name).await,
            Err(e) => self.set_stage(commitment, SendStage::Failed, format!("broadcast failed: {e}. Retry it from the Send tab.")),
        }
    }

    /// What a held name needs: its own records back when a rental ends, fresh
    /// payment keys for ones running low, a fresh control key, or a year more.
    /// `None` if nothing (or while it is rented out).
    fn upkeep(&mut self, job: &NameJob, r: &Record) -> Result<Option<(Want, &'static str)>> {
        let mut recs = r.records.clone();
        let mut terms = r.terms.clone();
        let mut note = "swapping in fresh keys";
        if let Deal::Rented { until, .. } = r.deal {
            if r.height < until {
                return Ok(None);
            }
            recs = match &job.home {
                Some(h) => names::parse(h).unwrap_or_default(),
                None => names::parse(&self.fresh_records(&format!("{}{}", job.name, names::SUFFIX))?).unwrap_or_default(),
            };
            terms = Deal::None.terms();
            note = "taking it back: the rental has ended";
        }
        let mut retired = Vec::new();
        for rec in recs.iter_mut() {
            if let Rec::MdsKey { key, .. } = rec {
                if self.signatures_left(key).is_some_and(|n| n <= PAY_RESERVE) {
                    retired.push(*key);
                    let fresh = self.fresh_key(format!("{}{}", job.name, names::SUFFIX))?;
                    *rec = Rec::MdsKey { key: fresh, height: DEFAULT_MSS_HEIGHT as u8 };
                }
            }
        }
        // The identity record follows this wallet's identity when it rotates:
        // requests to a retired key would go unanswered.
        let current = self.book.identity_pk;
        let mut identity_moved = false;
        for rec in recs.iter_mut() {
            if let (Rec::Identity(k), Some(now)) = (&*rec, current) {
                if *k != now && self.signatures_left(k).is_some() {
                    *rec = Rec::Identity(now);
                    identity_moved = true;
                    note = "pointing it at this wallet's current identity";
                }
            }
        }
        // A name pointed at one of this wallet's plans keeps to the plan's
        // next deadline (`follow_plans`).
        let (active, closed) = self.own_plans();
        let shown = format!("{}{}", job.name, names::SUFFIX);
        let plan_moved = match follow_plans(&recs, &active, &closed, r.height, || {
            Ok(Rec::MdsKey { key: self.fresh_key(shown.clone())?, height: DEFAULT_MSS_HEIGHT as u8 })
        })? {
            Some(next) => {
                recs = next;
                note = "moving it to its plan's next deadline";
                true
            }
            None => false,
        };
        let mut owner = job.owner;
        if self.signatures_left(&job.owner).is_some_and(|n| n <= CONTROL_RESERVE) {
            owner = self.fresh_key(format!("{}{} control", job.name, names::SUFFIX))?;
        }
        let renew = job.auto_renew && r.expires < r.height + RENEW_AHEAD;
        if renew {
            note = "renewing it for a year";
        }
        if retired.is_empty() && owner == job.owner && !renew && terms == r.terms && !identity_moved && !plan_moved {
            return Ok(None);
        }
        if let Some(j) = self.names.get_mut(&job.name) {
            j.retired.extend(retired.iter().map(hex::encode));
        }
        let (years, expires) = if renew { (1, r.expires.max(r.height) + names::YEAR) } else { (0, r.expires) };
        Ok(Some((Want { owner, records: names::encode(&recs)?, terms, years, expires, how: "hold".into() }, note)))
    }

    /// Sweep coins that reach a swapped-out payment key, two or more at a
    /// time, while it still has signatures to do it with.
    async fn sweep_retired(&mut self, job: &NameJob) {
        let in_flight = self.in_flight_inputs();
        for key in job.retired.iter().filter_map(|k| hex::decode(k).ok()?.try_into().ok()) {
            let address: [u8; 32] = compute_address(&key);
            let Some(w) = self.wallet.as_ref() else { return };
            let at: Vec<_> = w.coins().iter().filter(|c| c.address == address).map(|c| c.coin_id).collect();
            if at.len() < 2 || at.iter().any(|c| in_flight.contains(c)) {
                continue;
            }
            let shown = midstate::core::encode_address_with_checksum(&address);
            if let Err(e) = self.consolidate(&shown).await {
                tracing::debug!("name {}: sweeping a swapped-out key: {e:#}", job.name);
            }
        }
    }

    /// Names this wallet's keys hold that it has no job for: reserved to it
    /// at genesis, given to it, or held before a restore from its phrase. One
    /// with no records of its own, or controlled by the identity key, is set
    /// up with fresh keys (a free change).
    fn adopt_names(&mut self) {
        let Some(w) = self.wallet.as_ref() else { return };
        let keys: Vec<[u8; 32]> = w.mss_keys().iter().map(|k| k.master_pk).collect();
        let identity = self.book.identity_pk;
        let db = self.node.storage.db().clone();
        let mut found = Vec::new();
        for k in keys {
            for r in name_index::names_of(&db, &k).unwrap_or_default() {
                if !r.lapsed() && !self.names.jobs.iter().any(|j| j.name == r.name && !j.done()) {
                    found.push((k, r));
                }
            }
        }
        for (k, r) in found {
            let shown = format!("{}{}", r.name, names::SUFFIX);
            let mut job = NameJob {
                name: r.name.clone(),
                owner: k,
                want: None,
                commitment: None,
                status: "registered".into(),
                note: "found in this wallet".into(),
                tries: 0,
                tried_at: 0,
                retired: Vec::new(),
                auto_renew: true,
                renter: false,
                home: None,
                watch: None,
            };
            if r.records.is_empty() || Some(k) == identity {
                let set_up = (|| -> Result<Want> {
                    let owner = if Some(k) == identity { self.fresh_key(format!("{shown} control"))? } else { k };
                    Ok(Want { owner, records: self.fresh_records(&shown)?, terms: r.terms.clone(), years: 0, expires: r.expires, how: "hold".into() })
                })();
                if let Ok(want) = set_up {
                    job.want = Some(want);
                    job.status = "updating".into();
                    job.note = "setting it up with keys of its own".into();
                }
            }
            self.names.jobs.retain(|j| j.name != r.name);
            self.names.jobs.push(job);
        }
        self.names_save();
    }

    /// A watched name: claim it the block it is free, or buy it if offered
    /// within the watch's price.
    async fn watch_tick(&mut self, job: &NameJob) {
        let Some(w) = job.watch.clone() else { return };
        let db = self.node.storage.db().clone();
        let Ok(r) = name_index::lookup(&db, &job.name) else { return };
        let shown = format!("{}{}", job.name, names::SUFFIX);
        let tip = name_index::meta(&db).map_or(0, |m| m.next);
        let (how, status, years, expires) = match &r {
            None => ("hold", "registering", w.years, tip + w.years * names::YEAR),
            Some(r) if r.lapsed() => ("hold", "registering", w.years, tip + w.years * names::YEAR),
            Some(r) if matches!(r.deal, Deal::Sale { price, .. } if w.max_price.is_some_and(|m| price <= m)) => ("buy", "buying", 0, r.expires),
            _ => return,
        };
        let keys = (|| -> Result<([u8; 32], Vec<u8>)> { Ok((self.fresh_key(format!("{shown} control"))?, self.fresh_records(&shown)?)) })();
        let Ok((owner, records)) = keys else { return };
        if let Some(j) = self.names.get_mut(&job.name) {
            j.owner = owner;
            j.want = Some(Want { owner, records, terms: Deal::None.terms(), years, expires, how: how.into() });
            j.status = status.into();
            j.note = "it came free: getting it".into();
            j.watch = None;
        }
        self.names_save();
        if let Err(e) = self.commit_name(&job.name).await {
            self.name_note(&job.name, e);
        }
    }

    /// Each tick: settle changes that landed or lost, keep held names
    /// working, and retry what is waiting, once a block.
    pub(super) async fn tick_names(&mut self, tip: u64) {
        if tip > self.names.scanned_at {
            self.names.scanned_at = tip;
            self.adopt_names();
        }
        let jobs: Vec<NameJob> = self.names.jobs.iter().filter(|j| !j.done()).cloned().collect();
        for job in jobs {
            let pending = job.commitment.and_then(|c| self.wallet.as_ref()?.find_pending(&c).cloned());
            match (job.commitment, pending) {
                // Landed (its first coin is spent), lost (the thread moved
                // without it), or abandoned: settle against the chain.
                (Some(_), Some(p)) => {
                    let landed = !self.node.check_coin(p.input_coin_ids[0]).await;
                    if landed || !self.node.check_coin(*p.input_coin_ids.last().unwrap()).await {
                        self.settle_name(&job.name).await;
                    }
                }
                (Some(_), None) => self.settle_name(&job.name).await,
                (None, _) if tip <= job.tried_at => {}
                (None, _) if job.status == "watching" => {
                    if let Some(j) = self.names.get_mut(&job.name) {
                        j.tried_at = tip;
                    }
                    self.watch_tick(&job).await;
                }
                (None, _) if job.want.is_some() => {
                    self.settle_name(&job.name).await;
                    if self.names.jobs.iter().any(|j| j.name == job.name && j.want.is_some() && !j.done()) {
                        if let Err(e) = self.commit_name(&job.name).await {
                            self.name_note(&job.name, e);
                        }
                    }
                }
                (None, _) => {
                    self.sweep_retired(&job).await;
                    let r = match name_index::lookup(self.node.storage.db(), &job.name) {
                        Ok(Some(r)) => r,
                        Ok(None) => continue,
                        Err(_) => continue,
                    };
                    if job.renter {
                        // A rental this wallet holds ends at its height, or when its holder takes it back.
                        let ours = matches!(r.deal, Deal::Rented { until, .. } if r.height < until)
                            && job.home.as_ref() == names::encode(&r.records).ok().as_ref();
                        if !ours {
                            self.name_status(&job.name, "ended", "the rental has ended");
                        }
                        continue;
                    }
                    if r.owner != job.owner {
                        self.settle_name(&job.name).await;
                        continue;
                    }
                    if let Some(j) = self.names.get_mut(&job.name) {
                        j.tried_at = tip;
                    }
                    match self.upkeep(&job, &r) {
                        Ok(Some((want, note))) => self.want_change(&job.name, want, note),
                        Ok(None) => {}
                        Err(e) => self.name_note(&job.name, e),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::address_kind::AddressKind;
    use midstate::core::plans::{Heir, DAY as PLAN_DAY};

    fn plan(secret: u8) -> InheritancePlan {
        InheritancePlan {
            owner_pk: [1; 32],
            heirs: vec![Heir { pk: [2; 32], share: 65_535 }],
            start: 400_000,
            period: 30 * PLAN_DAY,
            secret: [secret; 32],
        }
    }

    fn pointed_at(p: &InheritancePlan, k: u64) -> Vec<Rec> {
        let mut recs = vec![Rec::MdsAddress(p.address(k))];
        recs.extend(names::script_records(&p.script(k)));
        recs.push(Rec::Identity([5; 32]));
        recs.push(Rec::Text("hi".into()));
        recs
    }

    fn no_key() -> Result<Rec> {
        bail!("not asked for")
    }

    #[test]
    fn a_name_on_a_plan_keeps_to_its_next_deadline() {
        let p = plan(9);
        let k = p.epoch_after(p.start);
        let recs = pointed_at(&p, k);
        // A full period or more before the deadline: it stays.
        assert_eq!(follow_plans(&recs, &[p.clone()], &[], p.deadline(k) - p.period, no_key).unwrap(), None);
        // Less than that left: it moves to where a check-in now puts the coins.
        let tip = p.deadline(k) - p.period + 1;
        let next = p.epoch_after(tip);
        assert_eq!(next, k + 1);
        let moved = follow_plans(&recs, &[p.clone()], &[], tip, no_key).unwrap().unwrap();
        assert_eq!(moved[0], Rec::MdsAddress(p.address(next)));
        assert_eq!(names::published_script(&moved), Some(p.script(next)));
        assert_eq!(names::address_kind(&moved, &p.address(next)), Some(AddressKind::Plan { heirs: 1 }));
        assert_eq!(names::pays(&moved, |_| 0), Some(p.address(next)));
        assert_eq!(&moved[moved.len() - 2..], &[Rec::Identity([5; 32]), Rec::Text("hi".into())]);
        assert!(names::encode(&moved).is_ok());
        // Long past, after the wallet was away: straight to the current deadline.
        let late = p.deadline(k) + 5 * p.period;
        let moved = follow_plans(&recs, &[p.clone()], &[], late, no_key).unwrap().unwrap();
        assert_eq!(moved[0], Rec::MdsAddress(p.address(p.epoch_after(late))));
        assert!(p.deadline(p.epoch_after(late)) >= late + p.period);
    }

    #[test]
    fn a_name_leaves_a_closed_plan_and_ignores_other_addresses() {
        let p = plan(9);
        let k = p.epoch_after(p.start);
        let recs = pointed_at(&p, k);
        let key = Rec::MdsKey { key: [3; 32], height: 10 };
        let back = follow_plans(&recs, &[], &[p.clone()], p.start + 10, || Ok(key.clone())).unwrap().unwrap();
        assert_eq!(back, vec![key, Rec::Identity([5; 32]), Rec::Text("hi".into())]);
        // Someone else's plan, a multisig, or no address at all: left alone.
        assert_eq!(follow_plans(&recs, &[plan(8)], &[plan(7)], p.deadline(k), no_key).unwrap(), None);
        let plain = vec![Rec::MdsKey { key: [3; 32], height: 10 }, Rec::Text("x".into())];
        assert_eq!(follow_plans(&plain, &[p.clone()], &[p], 500_000, no_key).unwrap(), None);
    }
}
