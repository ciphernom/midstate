//! Co-bonding with a pool by its address.
//!
//! A co-bond is a mining bond locked on a pool's mining key: the coin stays
//! this wallet's (only its owner key, from the recovery phrase, spends it),
//! and the pool's midwimble node signs blocks with it. A pool running
//! `midstate pool` publishes what co-bonding with it takes and gives at
//! `<pool>/api/co-bonds`: its mining key, its midwimble fee, the cut each
//! co-bond takes off its owner's fee, and the share of the pool's fee that
//! co-bonders get. Once the lock confirms, the wallet hands the bond in there
//! (its script, value and salt, and this wallet's MDW address for the
//! rewards), and the pool's node registers it, so nobody passes files about.
//!
//! The hand-in reveals the bond's script to the pool, which is what lets its
//! node prove the bond to midwimble. It grants nothing that spends.

use crate::bond::BondRecord;
use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// The pool desktop users co-bond with unless they name another: the one
/// midwimble's mining guide names, which merge-mines it.
pub const DEFAULT_POOL: &str = "https://rpc.cypherpunk.gold/pool";

/// A pool's address as people write it, made into the base its API hangs
/// off: `rpc.cypherpunk.gold/pool` → `https://rpc.cypherpunk.gold/pool`.
/// A pasted API or dashboard URL comes back to the same base.
pub fn pool_base(text: &str) -> Result<String> {
    let t = text.trim();
    if t.is_empty() {
        bail!("give the pool's address");
    }
    if t.contains(char::is_whitespace) {
        bail!("a pool's address has no spaces");
    }
    let with_scheme = if t.starts_with("http://") || t.starts_with("https://") {
        t.to_string()
    } else {
        format!("https://{t}")
    };
    let mut base = with_scheme.trim_end_matches('/').to_string();
    for tail in ["/api/co-bonds", "/stats", "/#"] {
        if let Some(stripped) = base.strip_suffix(tail) {
            base = stripped.trim_end_matches('/').to_string();
        }
    }
    if let Some(i) = base.find('#') {
        base.truncate(i);
        base = base.trim_end_matches('/').to_string();
    }
    let host = base.split("://").nth(1).unwrap_or("");
    if host.is_empty() || host.starts_with('/') {
        bail!("that is not a pool's address");
    }
    Ok(base)
}

/// The pool's host, for people: `rpc.cypherpunk.gold`.
pub fn pool_host(base: &str) -> String {
    let rest = base.split("://").nth(1).unwrap_or(base);
    rest.split('/').next().unwrap_or(rest).to_string()
}

fn endpoint(base: &str) -> String {
    format!("{base}/api/co-bonds")
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("starting an HTTP client")
}

/// What co-bonding with a pool takes and gives, as it says.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PoolTerms {
    /// The pool's base address, as [`pool_base`] made it.
    #[serde(default)]
    pub url: String,
    /// It merge-mines midwimble and its node takes co-bonds now.
    #[serde(default)]
    pub accepting: bool,
    /// Why it does not, when it does not.
    #[serde(default)]
    pub note: Option<String>,
    /// The key a co-bond with it must name (hex).
    #[serde(default)]
    pub mining_key: Option<String>,
    /// Its midwimble fee, in percent.
    #[serde(default)]
    pub fee_percent: f64,
    /// Percentage points off a miner's midwimble fee per co-bond.
    #[serde(default)]
    pub fee_cut: f64,
    /// Percent of the pool's midwimble fee shared among co-bonds.
    #[serde(default)]
    pub reward_percent: f64,
    /// Co-bonds signing for it now.
    #[serde(default)]
    pub registered: u64,
}

/// Asks the pool at `base` what co-bonding with it takes and gives.
pub async fn terms(base: &str) -> Result<PoolTerms> {
    let r = client()?
        .get(endpoint(base))
        .send()
        .await
        .with_context(|| format!("{} did not answer", pool_host(base)))?;
    if r.status() == reqwest::StatusCode::NOT_FOUND {
        bail!("{} takes no co-bonds by address: it may be an older pool", pool_host(base));
    }
    if !r.status().is_success() {
        bail!("{} answered {}", pool_host(base), r.status());
    }
    let mut t: PoolTerms = r.json().await.context("reading the pool's terms")?;
    t.url = base.to_string();
    Ok(t)
}

/// What the pool said to a hand-in.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct HandedIn {
    /// Where the pool's node has the bond: "waiting", "burying", "ready",
    /// "registered", …
    #[serde(default)]
    pub phase: String,
    /// The address its rewards go to, if the pool took one.
    #[serde(default)]
    pub address: Option<String>,
    #[serde(default)]
    pub note: String,
}

/// Hands `rec` in to the pool at `base`, with `address` for its rewards.
pub async fn hand_in(base: &str, rec: &BondRecord, address: Option<&str>) -> Result<HandedIn> {
    let body = serde_json::json!({
        "script": hex::encode(rec.script()),
        "value": rec.value,
        "salt": hex::encode(rec.salt),
        "address": address,
    });
    let r = client()?
        .post(endpoint(base))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("{} did not answer", pool_host(base)))?;
    let ok = r.status().is_success();
    let v: serde_json::Value = r.json().await.unwrap_or_default();
    if !ok {
        let why = v["error"].as_str().unwrap_or("it refused the bond");
        return Err(anyhow!("{}: {why}", pool_host(base)));
    }
    Ok(HandedIn {
        phase: v["co_bond"]["phase"].as_str().unwrap_or_default().to_string(),
        address: v["address"].as_str().map(str::to_string),
        note: v["note"].as_str().unwrap_or_default().to_string(),
    })
}

/// Where the pool at `base` has bond `id` (hex): its phase, progress and
/// note, or `None` if the pool does not list it.
pub async fn status(base: &str, id: &str) -> Result<Option<(String, u32, String)>> {
    let r = client()?.get(endpoint(base)).send().await?;
    if !r.status().is_success() {
        bail!("{} answered {}", pool_host(base), r.status());
    }
    let v: serde_json::Value = r.json().await?;
    Ok(v["co_bonds"].as_array().into_iter().flatten().find(|b| b["bond_id"] == id).map(|b| {
        (
            b["phase"].as_str().unwrap_or_default().to_string(),
            b["progress"].as_u64().unwrap_or(0) as u32,
            b["note"].as_str().unwrap_or_default().to_string(),
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pool_is_found_however_its_address_is_written() {
        for text in [
            "rpc.cypherpunk.gold/pool",
            "https://rpc.cypherpunk.gold/pool",
            "https://rpc.cypherpunk.gold/pool/",
            "https://rpc.cypherpunk.gold/pool/api/co-bonds",
            "https://rpc.cypherpunk.gold/pool#/mw/mw1q",
            " rpc.cypherpunk.gold/pool ",
        ] {
            assert_eq!(pool_base(text).unwrap(), DEFAULT_POOL, "{text}");
        }
        assert_eq!(pool_base("http://10.0.0.5:8081").unwrap(), "http://10.0.0.5:8081");
        assert_eq!(endpoint("http://10.0.0.5:8081"), "http://10.0.0.5:8081/api/co-bonds");
        assert_eq!(pool_host(DEFAULT_POOL), "rpc.cypherpunk.gold");
        assert!(pool_base("").is_err());
        assert!(pool_base("https://").is_err());
        assert!(pool_base("two words").is_err());
    }

    #[test]
    fn terms_read_whatever_a_pool_leaves_out() {
        let t: PoolTerms = serde_json::from_str(r#"{"accepting":true,"mining_key":"ab","fee_percent":1.0}"#).unwrap();
        assert!(t.accepting);
        assert_eq!(t.fee_cut, 0.0);
        assert_eq!(t.reward_percent, 0.0);
        assert_eq!(t.mining_key.as_deref(), Some("ab"));
    }
}
