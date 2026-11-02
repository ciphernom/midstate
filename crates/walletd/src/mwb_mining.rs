//! Mining MDW from the desktop wallet, the two ways midwimble's mining guide
//! (`mining-guide-simple.md`) sets out.
//!
//! # On a pool that merge-mines midwimble
//!
//! The hasher names this wallet's MDW address (`midstate miner
//! --midwimble-address`). Every share then earns midwimble score too, and the
//! pool's midwimble blocks pay it straight from their coinbase. No bond, and
//! nothing for this module to do beyond handing the hasher the address.
//!
//! # Solo, with this wallet's own bond
//!
//! The local pool commits every job to a midwimble block as well (`midstate
//! pool --midwimble-rpc`), so one search mines both chains. What to commit to
//! comes from a midwimble node holding a registered bond, through
//! `/merge/job` and `/merge/found` (midwimble `docs/POOL_MERGED_MINING.md`).
//! Here walletd answers those in that node's place ([`MergeServer`]): it
//! builds the template from the embedded node's chain and mempool, signs it
//! with the wallet's bond, and hands a found block back to the node. So the
//! bond's secret never leaves the unlocked wallet, and the embedded node
//! needs no restart to take it.
//!
//! # The mining key
//!
//! One per wallet, derived from the recovery phrase ([`mining_secret`]), so a
//! restored wallet signs with its bonds again. Every bond locked on that key
//! mines: the first ready one is the node's bond and the rest its co-bonds,
//! and each block is signed by whichever has the most fork-choice quota left.
//! Sharing a key links those bonds on chain; keeping bonds apart takes
//! another wallet.
//!
//! # Registration
//!
//! `midwimble bond register`, from the wallet ([`register_once`]): a proof
//! that the bond coin is unspent at midstate's tip, then midstate's headers
//! above it until a day of midwimble's work is buried there. The wallet runs
//! it again until the bond is ready, takes a fresh proof whenever the old one
//! is about to leave the relay's two-day window, and stops once midwimble
//! holds the bond. Midstate's side is read through its node's RPC, exactly as
//! the CLI reads it.

use crate::bond::{BondRecord, RegStatus};
use anyhow::{anyhow, bail, ensure, Context, Result};
use axum::extract::State as AxState;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use midwimble::core::auxpow::{bytes32, merge_commitment, ParentTemplate};
use midwimble::core::bond::{
    bond_proof_from_json, burial_headers, est_midstate_height, header_link_from_json,
    min_mining_bond, min_remaining_bond_lock, registration_cap, registration_min_bond,
    registration_work, BondCoin, BondFile, BondRegistration, BondScript, CoBond, HeldProof,
    MinerBond,
};
use midwimble::core::mw::stealth::StealthAddress;
use midwimble::core::relay::{window_left, MidstateHeaders, MAX_RELAY_HEADERS_PER_BLOCK, RELAY_WINDOW};
use midwimble::core::state::calculate_work;
use midwimble::core::template::BlockTemplate;
use midwimble::node::NodeHandle;
use midwimble::rpc::RpcClient;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

/// A pool known to merge-mine midwimble, as midwimble's mining guide names
/// it. Offered, never chosen for anyone.
pub const MERGE_POOL: &str = "stratum+tcp://rpc.cypherpunk.gold:3333";

// ── The mining key ─────────────────────────────────────────────────────────

/// This wallet's midwimble mining secret: BLAKE3's key derivation over the
/// recovery phrase's master seed, reduced to a scalar.
pub fn mining_secret(master_seed: &[u8; 32]) -> Scalar {
    let mut h = blake3::Hasher::new_derive_key("midstate.mwb-mining-key.v1");
    h.update(master_seed);
    let mut wide = [0u8; 64];
    h.finalize_xof().fill(&mut wide);
    Scalar::from_bytes_mod_order_wide(&wide)
}

/// The mining key (public) for `secret`, as a bond script names it.
pub fn mining_key_of(secret: &Scalar) -> [u8; 32] {
    RistrettoPoint::mul_base(secret).compress().to_bytes()
}

/// A bond record as midwimble knows the coin.
pub fn bond_coin(b: &BondRecord) -> BondCoin {
    BondCoin {
        script: BondScript { mining_key: b.mining_key, bonded_until: b.bonded_until, owner_pk: b.owner_pk },
        value: b.value,
        salt: b.salt,
    }
}

/// Whether a bond of `value` locked until `until` can sign a block made at
/// `now` (midwimble's `BondRules::eligible`).
pub fn eligible(value: u64, until: u64, now: u64) -> bool {
    value >= min_mining_bond(now) && until >= est_midstate_height(now).saturating_add(min_remaining_bond_lock(now))
}

/// The bond the wallet mines with: its bonds on `key` that are ready or on
/// chain and can still sign, the first as the node's bond and the rest as
/// its co-bonds. `None` without one.
pub fn miner_bond(secret: &Scalar, key: &[u8; 32], records: &[BondRecord], now: u64) -> Option<MinerBond> {
    let mut usable = records
        .iter()
        .filter(|b| &b.mining_key == key && matches!(b.reg.phase.as_str(), "ready" | "on_chain"))
        .filter(|b| eligible(b.value, b.bonded_until, now))
        .filter_map(|b| {
            let f = b.bond_file.as_ref()?;
            Some((f.registration.clone()?, f.midstate_headers.clone()))
        });
    let (registration, midstate) = usable.next()?;
    Some(MinerBond {
        secret: *secret,
        bond_id: registration.bond_id(),
        registration: Some(registration),
        midstate,
        co_bonds: usable
            .map(|(r, h)| CoBond { bond_id: r.bond_id(), registration: Some(r), midstate: h })
            .collect(),
    })
}

// ── Registration ───────────────────────────────────────────────────────────

/// Midwimble's chain, as a registration needs it: its target, and how far its
/// midstate relay has got.
#[derive(Clone, Copy, Debug)]
pub struct ChainNow {
    pub target: [u8; 32],
    pub relay_tip: u64,
    /// Unix seconds.
    pub now: u64,
}

impl ChainNow {
    pub fn of(state: &midwimble::core::types::State, now: u64) -> Self {
        Self { target: state.target, relay_tip: state.relay.tip().height, now }
    }
}

/// Midstate's header at `height`, or `None` if its node is not that far yet.
fn midstate_header(midstate: &RpcClient, height: u64) -> Result<Option<midwimble::core::anchor::HeaderLink>> {
    let v = midstate.get(&format!("/headers/{height}/1"))?;
    v["headers"]
        .as_array()
        .and_then(|headers| headers.first())
        .map(|h| header_link_from_json(h).map(|(link, _)| link))
        .transpose()
}

/// One run of `midwimble bond register` for `coin`, carrying on from `held`
/// (the bond file as the last run left it). Blocking: it reads midstate's
/// node at `midstate_rpc` (`host:port`).
///
/// # Formal Specification
///
/// ```text
/// Pre:  coin is a bond this wallet locked, confirmed on midstate
/// Post: Ok((file, status)) ⇒ file.proof is a proof of coin no older than
///                              the relay's window allows
///                            ∧ file.registration, file.midstate_headers
///                              carry it and the headers above it that
///                              midstate has
///                            ∧ status.phase = "ready"   ⇔ a day of
///                              midwimble's work is buried above the proof
///                            ∧ status.phase = "burying" otherwise
///       Err(e)           ⇒ nothing usable was learned; e says why
/// ```
pub fn register_once(
    coin: BondCoin,
    held: Option<BondFile>,
    midstate_rpc: &str,
    chain: &ChainNow,
) -> Result<(BondFile, RegStatus)> {
    let now = chain.now;
    let (value, until) = (coin.value, coin.script.bonded_until);
    ensure!(
        value >= min_mining_bond(now),
        "this bond holds {}; midwimble counts bonds of {} or more",
        crate::bond::gmds(value),
        crate::bond::gmds(min_mining_bond(now))
    );
    let shortest = est_midstate_height(now) + min_remaining_bond_lock(now);
    ensure!(
        until >= shortest,
        "its lock is too short: midwimble counts a bond only while more than {} days of its lock \
         remain, which takes a lock until block {shortest} or later",
        min_remaining_bond_lock(now) / 1440
    );
    let asked = registration_min_bond(&chain.target, now);
    ensure!(
        value >= asked,
        "at midwimble's difficulty a new bond needs {}; this one holds {}",
        crate::bond::gmds(asked),
        crate::bond::gmds(value)
    );

    let mut f = held.unwrap_or_else(|| BondFile::co_bond(coin.script.mining_key));
    let midstate = RpcClient::new(midstate_rpc);
    let anchor = midwimble::core::types::MIDSTATE_BLOCK_HEIGHT;
    // Midstate's `/state` height counts blocks, so its tip is one below.
    let midstate_tip = midstate.height().context("reading midstate's height")?.saturating_sub(1);
    let tip = midstate_tip.max(chain.relay_tip);

    // 1. The proof, against midstate's tip; one already taken is kept while
    //    it can still register.
    let keep = f.proof.clone().filter(|p| p.coin == coin).filter(|p| {
        p.midstate_height > anchor
            && !HeldProof::at(p.midstate_height, tip).needs_refresh()
            && (p.midstate_height > midstate_tip
                || matches!(
                    midstate_header(&midstate, p.midstate_height),
                    Ok(Some(h)) if p.verify(&h.state_root).is_ok()
                ))
    });
    let proof = match keep {
        Some(p) => p,
        None => {
            let v = midstate
                .get(&format!("/utxo_proof/{}", hex::encode(coin.coin_id())))
                .context("asking midstate for a proof of the bond coin")?;
            let p = bond_proof_from_json(coin, &v)?;
            p.verify(&bytes32(&v["state_root"])?)?;
            f.coin = Some(coin);
            f.proof = Some(p.clone());
            f.registration = None;
            f.midstate_headers = None;
            p
        }
    };
    ensure!(
        proof.midstate_height > anchor,
        "midstate's node is behind midwimble's anchor (block {anchor}): let it sync"
    );

    // 2. Midstate's headers from the relay's tip (or the proof's header, if
    //    the relay is past it) up through a day of work above the proof.
    let start = (chain.relay_tip + 1).min(proof.midstate_height);
    let required = registration_work(&chain.target);
    let goal = required + required / 4;
    let cap = registration_cap(&chain.target);
    let mut run: Option<MidstateHeaders> = None;
    let mut credited = 0u128;
    'fetch: loop {
        let next = run.as_ref().map_or(start, |r| r.start + r.headers.len() as u64);
        // Midstate's node answers up to 2,000 at a time, and none past its tip.
        let v = midstate.get(&format!("/headers/{next}/2000"))?;
        let headers = v["headers"].as_array().cloned().unwrap_or_default();
        if headers.is_empty() {
            break;
        }
        for h in &headers {
            let (link, prev) = header_link_from_json(h)?;
            let r = run.get_or_insert_with(|| MidstateHeaders { start, prev, headers: Vec::new() });
            ensure!(
                r.headers.last().map_or(prev == r.prev, |l| l.final_hash == prev),
                "midstate's headers do not link at height {}",
                r.start + r.headers.len() as u64
            );
            let height = r.start + r.headers.len() as u64;
            if height > proof.midstate_height {
                credited = credited.saturating_add(calculate_work(&link.target).min(cap));
            }
            r.headers.push(link);
            if credited >= goal || r.headers.len() >= MAX_RELAY_HEADERS_PER_BLOCK {
                break 'fetch;
            }
        }
    }
    let run = run.ok_or_else(|| anyhow!("midstate returned no headers from block {start}"))?;
    let registration = BondRegistration { proof };
    registration.check_carried(&run, &chain.target)?;
    let height = registration.proof.midstate_height;
    let burial = burial_headers(&chain.target, calculate_work(&run.headers[(height - run.start) as usize].target));
    f.registration = Some(registration);
    f.midstate_headers = Some(run);

    let hours = window_left(height, tip).unwrap_or(0) / 60;
    let status = if credited >= required {
        RegStatus {
            phase: "ready".into(),
            progress: 100,
            note: format!(
                "Ready: the first midwimble block mined with it registers it, if one is found within \
                 about {hours} hours (before the relay passes midstate block {}). After that the wallet \
                 takes a fresh proof and keeps going.",
                height + RELAY_WINDOW
            ),
            checked: now,
        }
    } else {
        let progress = (credited * 100 / required.max(1)).min(99) as u32;
        let note = if burial >= RELAY_WINDOW {
            format!(
                "At midwimble's difficulty a proof needs about {burial} midstate blocks above it and the \
                 relay keeps only {RELAY_WINDOW}: no bond can register until midwimble's difficulty falls."
            )
        } else {
            format!(
                "Waiting for a day of midwimble's work above the proof (midstate block {height}): \
                 {progress}% there, about {burial} midstate blocks in all. The wallet checks again by itself."
            )
        };
        RegStatus { phase: "burying".into(), progress, note, checked: now }
    };
    Ok((f, status))
}

// ── Merged mining, answered in a bonded node's place ───────────────────────

/// Serves what a midstate pool asks a bonded midwimble node when it
/// merge-mines (`docs/POOL_MERGED_MINING.md`): `GET /state` for the tip it
/// refreshes jobs on, `POST /merge/job` for what to commit to, and `POST
/// /merge/found` for a share that made a block. Loopback only, and only to
/// requests naming a loopback host, so no web page can reach it through a
/// name of its own.
pub struct MergeServer {
    pub addr: std::net::SocketAddr,
    bond: Arc<RwLock<Option<MinerBond>>>,
    node: NodeHandle,
    task: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct MergeCtx {
    node: NodeHandle,
    bond: Arc<RwLock<Option<MinerBond>>>,
    /// Templates handed out, by mining hash, newest last (as the node keeps
    /// them: a pool refreshes on either tip, so a handful is plenty).
    jobs: Arc<Mutex<VecDeque<([u8; 32], BlockTemplate)>>>,
}

type ApiResult = std::result::Result<Json<Value>, (StatusCode, Json<Value>)>;

fn bad(e: impl std::fmt::Display) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() })))
}

impl MergeServer {
    /// Binds a free loopback port and serves there until dropped.
    pub async fn start(node: NodeHandle, bond: Option<MinerBond>) -> Result<Self> {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.context("binding the merge endpoint")?;
        let addr = listener.local_addr()?;
        if let Some(b) = &bond {
            feed_relay(&node, b);
        }
        let bond = Arc::new(RwLock::new(bond));
        let ctx = MergeCtx { node: node.clone(), bond: bond.clone(), jobs: Arc::default() };
        let app = Router::new()
            .route("/state", get(merge_state))
            .route("/merge/job", post(merge_job))
            .route("/merge/found", post(merge_found))
            .layer(axum::middleware::from_fn(loopback_only))
            .with_state(ctx);
        let task = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::warn!("merge endpoint stopped: {e:#}");
            }
        });
        tracing::info!("merged-mining endpoint for the local pool on {addr}");
        Ok(Self { addr, bond, node, task })
    }

    /// `host:port`, as `midstate pool --midwimble-rpc` takes it.
    pub fn rpc(&self) -> String {
        self.addr.to_string()
    }

    /// Signs jobs with `bond` from now on (`None`: refuses them, and the
    /// pool mines midstate alone).
    pub fn set_bond(&self, bond: Option<MinerBond>) {
        if let Some(b) = &bond {
            feed_relay(&self.node, b);
        }
        if let Ok(mut held) = self.bond.write() {
            *held = bond;
        }
    }

    pub fn bond_id(&self) -> Option<[u8; 32]> {
        self.bond.read().ok()?.as_ref().map(|b| b.bond_id)
    }
}

impl Drop for MergeServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The midstate headers a bond's registration carries, into the node's relay
/// feed, so its templates relay what the chain lacks of them (as a node
/// started with the bond file does).
fn feed_relay(node: &NodeHandle, bond: &MinerBond) {
    let feed = node.relay_feed();
    let Ok(mut feed) = feed.write() else { return };
    let carried = std::iter::once(&bond.midstate).chain(bond.co_bonds.iter().map(|c| &c.midstate)).flatten();
    for run in carried {
        feed.add(run, false);
    }
}

/// Refuses a request whose `Host` is not a loopback name or address.
async fn loopback_only(request: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    use axum::response::IntoResponse;
    let host = request
        .headers()
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let name = match host.rsplit_once(':') {
        Some((n, port)) if port.bytes().all(|b| b.is_ascii_digit()) => n.to_string(),
        _ => host,
    };
    if matches!(name.as_str(), "127.0.0.1" | "localhost" | "[::1]") {
        next.run(request).await
    } else {
        (StatusCode::FORBIDDEN, Json(json!({ "error": "loopback only" }))).into_response()
    }
}

async fn merge_state(AxState(c): AxState<MergeCtx>) -> Json<Value> {
    let s = c.node.state();
    Json(json!({
        "height": s.height,
        "tip": hex::encode(s.header_hash),
        "target": hex::encode(s.target),
    }))
}

#[derive(Deserialize)]
struct PayoutWeight {
    address: String,
    weight: u64,
}

#[derive(Deserialize)]
struct JobBody {
    payouts: Vec<PayoutWeight>,
    #[serde(default)]
    extra: Option<String>,
}

#[derive(Deserialize)]
struct FoundBody {
    mining_hash: String,
    batch_template: Value,
    commit_index: usize,
    nonce: u64,
    final_hash: String,
}

fn hex32(s: &str) -> std::result::Result<[u8; 32], (StatusCode, Json<Value>)> {
    hex::decode(s).map_err(bad)?.try_into().map_err(|_| bad("expected 32 bytes of hex"))
}

/// A template paying `body`'s payouts, signed with the wallet's bond, built
/// as midwimble's node builds one for a merged-mining pool.
async fn build_template(c: &MergeCtx, body: JobBody) -> std::result::Result<BlockTemplate, (StatusCode, Json<Value>)> {
    let bond = c
        .bond
        .read()
        .map_err(bad)?
        .clone()
        .ok_or_else(|| bad("this wallet has no bond ready to mine with"))?;
    let payouts: Vec<(StealthAddress, u64)> = body
        .payouts
        .iter()
        .map(|p| StealthAddress::decode(&p.address).map(|a| (a, p.weight)))
        .collect::<Result<_>>()
        .map_err(bad)?;
    let extra = match body.extra {
        Some(h) => bytes32(&Value::String(h)).map_err(bad)?,
        None => [0u8; 32],
    };
    let info = c.node.info();
    if info.syncing {
        return Err(bad("node is syncing"));
    }
    let state = info.state;
    let timestamps = c
        .node
        .storage()
        .load_timestamps(state.height, midwimble::core::types::DIFFICULTY_LOOKBACK)
        .map_err(bad)?;
    let pool = c.node.mempool().await.map_err(bad)?;
    let feed = c.node.relay_feed();
    let (template, _) = tokio::task::spawn_blocking(move || {
        use midwimble::core::types::{KERNEL_WEIGHT, MAX_BLOCK_WEIGHT, OUTPUT_WEIGHT};
        let feed = feed.read().map_err(|e| anyhow!("relay feed: {e}"))?;
        let registration_weight = std::iter::once((bond.bond_id, bond.registration.as_ref()))
            .chain(bond.co_bonds.iter().map(|c| (c.bond_id, c.registration.as_ref())))
            .filter(|(id, _)| !state.bonds.contains_key(id))
            .find_map(|(_, r)| r)
            .map_or(0, |r| r.weight());
        let relay_weight = feed.next_for(&state.relay).map_or(0, |r| r.weight());
        let budget = (MAX_BLOCK_WEIGHT - (payouts.len() as u64) * OUTPUT_WEIGHT - KERNEL_WEIGHT)
            .saturating_sub(registration_weight)
            .saturating_sub(relay_weight);
        let txs = midwimble::core::template::select_transactions(&pool, state.height, budget);
        let build = |txs: &[midwimble::core::mw::Transaction]| {
            midwimble::core::template::build_template_relayed(
                &state,
                &timestamps,
                txs,
                &payouts,
                extra,
                None,
                Some(&bond),
                Some(&feed),
            )
        };
        build(&txs).or_else(|_| build(&[]))
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    Ok(template)
}

async fn merge_job(AxState(c): AxState<MergeCtx>, Json(body): Json<JobBody>) -> ApiResult {
    let template = build_template(&c, body).await?;
    let commitment = merge_commitment(&template.mining_hash);
    let answer = json!({
        "commitment": hex::encode(commitment),
        "target": hex::encode(template.batch.target),
        "mining_hash": hex::encode(template.mining_hash),
        "height": template.height,
        "reward": template.subsidy.saturating_add(template.fees),
        "pow": midwimble::core::types::pow_algo_at(template.height).name(),
    });
    let mut jobs = c.jobs.lock().map_err(bad)?;
    jobs.push_back((template.mining_hash, template));
    while jobs.len() > 16 {
        jobs.pop_front();
    }
    Ok(Json(answer))
}

async fn merge_found(AxState(c): AxState<MergeCtx>, Json(body): Json<FoundBody>) -> ApiResult {
    let mining_hash = hex32(&body.mining_hash)?;
    let final_hash = hex32(&body.final_hash)?;
    let template = c
        .jobs
        .lock()
        .map_err(bad)?
        .iter()
        .find(|(hash, _)| *hash == mining_hash)
        .map(|(_, t)| t.clone())
        .ok_or_else(|| bad("no merged-mining job with that mining hash; ask for a fresh one"))?;
    let parent = ParentTemplate::from_midstate_json(&body.batch_template, body.commit_index).map_err(bad)?;
    let sealed = template.seal_aux(parent.proof(body.nonce, final_hash));
    let merged = sealed.aux_pow.is_some();
    let height = template.height;
    c.node.submit_block(sealed).await.map_err(bad)?;
    tracing::info!("merged-mined midwimble block {height} for this wallet's bond");
    Ok(Json(json!({ "accepted": true, "height": height, "merged_mined": merged })))
}

/// Why solo mining MDW cannot start, if it cannot: what the Mining panel
/// says beside the local pool.
pub fn solo_blocker(has_account: bool, has_node: bool, bond: Option<&MinerBond>) -> Option<&'static str> {
    if !has_account {
        Some("the wallet has no MDW account")
    } else if !has_node {
        Some("the midwimble node is not running")
    } else if bond.is_none() {
        Some("no bond of this wallet's is ready to mine with")
    } else {
        None
    }
}

/// Fails unless `s` reads as a bond id this wallet's records hold.
pub fn parse_bond_id(s: &str) -> Result<[u8; 32]> {
    let raw = hex::decode(s.trim()).context("a bond id is hex")?;
    raw.try_into().map_err(|_| anyhow!("a bond id is 32 bytes"))
}

/// A bond file for `b` that holds no secret: what a pool operator adds with
/// `--co-bond`, or what `midwimble node --mining-bond` needs once its
/// secret is filled in.
pub fn co_bond_file(b: &BondRecord) -> Result<String> {
    let mut f = b.bond_file.clone().unwrap_or_else(|| BondFile::co_bond(b.mining_key));
    f.secret.clear();
    f.mining_pubkey = Some(hex::encode(b.mining_key));
    if f.coin.is_none() {
        f.coin = Some(bond_coin(b));
    }
    if f.registration.is_none() {
        bail!("this bond is not registered yet: register it first");
    }
    Ok(serde_json::to_string_pretty(&f)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mining_key_is_the_phrases_and_only_its() {
        let a = mining_secret(&[1; 32]);
        assert_eq!(a, mining_secret(&[1; 32]), "a restored wallet gets it back");
        assert_ne!(a, mining_secret(&[2; 32]));
        assert_ne!(a, Scalar::ZERO);
        let key = mining_key_of(&a);
        let file = BondFile { secret: hex::encode(a.to_bytes()), ..BondFile::default() };
        assert_eq!(file.mining_key().unwrap(), key, "midwimble reads the same key from it");
    }

    #[test]
    fn eligibility_follows_midwimbles_rules() {
        let now = 1_793_500_000;
        let edge = est_midstate_height(now) + min_remaining_bond_lock(now);
        let min = min_mining_bond(now);
        assert!(eligible(min, edge, now));
        assert!(!eligible(min, edge - 1, now), "a lock that ends too soon");
        assert!(!eligible(min / 2, edge, now), "a bond below the minimum");
    }

    #[test]
    fn only_ready_bonds_on_this_key_mine() {
        let secret = mining_secret(&[7; 32]);
        let key = mining_key_of(&secret);
        let rec = |phase: &str, mining_key| BondRecord {
            mining_key,
            bonded_until: u64::MAX / 2,
            owner_pk: [3; 32],
            owner: Default::default(),
            value: 1 << 40,
            salt: [4; 32],
            label: String::new(),
            created_at: 0,
            fund_commitment: None,
            unbond_commitment: None,
            unbond_sig: None,
            owner_next_leaf: 0,
            seen_live: true,
            bond_file: None,
            reg: RegStatus { phase: phase.into(), ..RegStatus::default() },
        };
        // Without a registration on file, nothing is usable, whatever the
        // phase says.
        let records = [rec("ready", key), rec("burying", key), rec("ready", [9; 32])];
        assert!(miner_bond(&secret, &key, &records, 1_793_500_000).is_none());
        assert_eq!(solo_blocker(true, true, None), Some("no bond of this wallet's is ready to mine with"));
        assert_eq!(solo_blocker(false, true, None), Some("the wallet has no MDW account"));
        assert!(co_bond_file(&records[0]).is_err(), "unregistered: nothing a pool can use");
    }
}
