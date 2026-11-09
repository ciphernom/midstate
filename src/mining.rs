//! Mining coordination for the full node.
//!
//! This module was extracted from the `node` god module during the refactor.
//! It owns the types for miner config, the result of a mining attempt (block or pool share),
//! and helpers for coinbase generation and logging.
//!
//! The CPU-intensive extension mining itself lives in `crate::core::extension`.
//! Template building for *external* miners (RPC/light) remains in `node.rs` as
//! `build_block_template_inner` (due to its dependency on `rpc::types`).

use crate::core::types::*;
use crate::core::{block_reward, decompose_value, CoinbaseOutput};
use crate::wallet::{coinbase_seed, coinbase_salt};
use crate::core::wots;
use std::path::PathBuf;
use rayon::prelude::*;
use tokio::net::TcpStream;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Default, Clone)]
pub struct StratumStats {
    pub network_target: [u8; 32],
    pub accepted_shares: u64,
    pub rejected_shares: u64,
    /// Connected to the pool now.
    pub connected: bool,
    /// The current job's block height (0 from a pool that does not send it).
    pub job_height: u64,
    /// Jobs received since the miner started.
    pub jobs: u64,
    /// Unix time of the last accepted share (0: none yet).
    pub last_share_unix: u64,
    /// Unix time this connection to the pool opened (0: not connected).
    pub connected_since: u64,
    /// Times the connection was lost and opened again.
    pub reconnects: u64,
    /// Unix time of the last job from the pool (0: none yet).
    pub last_job_unix: u64,
    /// The pool's share target for the current job.
    pub share_target: [u8; 32],
    /// Shares this miner found and sent, accepted or not.
    pub shares_found: u64,
    /// The lowest hash among them: the best share, by difficulty.
    pub best_share: Option<[u8; 32]>,
    /// How long the pool took to answer the last share (milliseconds).
    pub latency_ms: Option<u64>,
    /// The pool mined on now, or last tried (its stratum address).
    pub pool_url: String,
    /// Whether that is the fallback pool.
    pub on_fallback: bool,
}

/// Where and as whom `midstate miner` mines. The dashboard can change it
/// while the miner runs: the stratum client reconnects with it at once.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PoolSettings {
    pub pool_url: String,
    /// Mined on while `pool_url` cannot be reached.
    pub fallback_url: Option<String>,
    pub payout_address: String,
    pub worker: String,
    pub midwimble_address: Option<String>,
}

/// How long the stratum client waits, and when it gives up on a pool.
#[derive(Clone, Copy, Debug)]
pub struct StratumTiming {
    /// For the pool to accept the connection.
    pub connect: std::time::Duration,
    /// For any message from the pool. A pool that stops talking without
    /// closing the connection is left rather than waited on: the Pi once
    /// waited 24 minutes on one.
    pub idle: std::time::Duration,
    /// Between attempts to connect.
    pub retry: std::time::Duration,
    /// Failed attempts in a row before trying the other pool.
    pub failover_after: u32,
    /// How often, while on the fallback, the main pool is knocked on to see
    /// whether it is back. A knock is a TCP connect, closed at once.
    pub probe_every: std::time::Duration,
}

impl Default for StratumTiming {
    fn default() -> Self {
        use std::time::Duration;
        Self {
            connect: Duration::from_secs(10),
            idle: Duration::from_secs(300),
            retry: Duration::from_secs(5),
            failover_after: 3,
            probe_every: Duration::from_secs(120),
        }
    }
}

/// The hashing threads: how many are asked for, and how many may run now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Threads {
    /// Asked for; 0 is every core.
    pub want: usize,
    /// The most the heat limit lets run now (0: none), or `None` when it
    /// holds nothing back.
    pub heat_cap: Option<usize>,
    /// Stopped for a moment, while the unit test measures the board alone.
    pub paused: bool,
}

/// Cores this machine has.
pub fn cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

impl Threads {
    /// Asked for, with "every core" counted out.
    pub fn wanted(&self) -> usize {
        if self.want == 0 { cores() } else { self.want }
    }

    /// Hashing now: what is asked for, fewer while the heat limit holds them
    /// back, none while paused.
    pub fn running(&self) -> usize {
        if self.paused {
            return 0;
        }
        let w = self.wanted();
        self.heat_cap.map_or(w, |c| c.min(w))
    }
}

/// The live controls of `midstate miner`, shared by the stratum client, the
/// heat guard and the dashboard. A change reaches the client at once: new
/// pool settings reconnect it, a new thread count restarts the job's threads.
pub struct MinerControl {
    pool: tokio::sync::watch::Sender<PoolSettings>,
    threads: tokio::sync::watch::Sender<Threads>,
    /// Degrees Celsius above which threads are shed (`None`: no limit).
    heat_limit: std::sync::RwLock<Option<f32>>,
    pub timing: StratumTiming,
}

impl MinerControl {
    pub fn new(pool: PoolSettings, threads: usize, heat_limit: Option<f32>) -> Self {
        Self {
            pool: tokio::sync::watch::Sender::new(pool),
            threads: tokio::sync::watch::Sender::new(Threads { want: threads, heat_cap: None, paused: false }),
            heat_limit: std::sync::RwLock::new(heat_limit),
            timing: StratumTiming::default(),
        }
    }

    pub fn with_timing(mut self, timing: StratumTiming) -> Self {
        self.timing = timing;
        self
    }

    pub fn pool(&self) -> PoolSettings {
        self.pool.borrow().clone()
    }

    /// Mines with `pool` from now on. Whether anything changed.
    pub fn set_pool(&self, pool: PoolSettings) -> bool {
        self.pool.send_if_modified(|cur| {
            let changed = *cur != pool;
            *cur = pool;
            changed
        })
    }

    pub fn threads(&self) -> Threads {
        *self.threads.borrow()
    }

    fn update_threads(&self, f: impl FnOnce(&mut Threads)) {
        self.threads.send_if_modified(|t| {
            let before = *t;
            f(t);
            *t != before
        });
    }

    /// Threads asked for (0: every core).
    pub fn set_threads(&self, want: usize) {
        self.update_threads(|t| t.want = want);
    }

    pub fn set_heat_cap(&self, cap: Option<usize>) {
        self.update_threads(|t| t.heat_cap = cap);
    }

    pub fn set_paused(&self, paused: bool) {
        self.update_threads(|t| t.paused = paused);
    }

    pub fn heat_limit(&self) -> Option<f32> {
        *self.heat_limit.read().unwrap_or_else(|e| e.into_inner())
    }

    /// A new heat limit. With none, every thread asked for runs again.
    pub fn set_heat_limit(&self, limit: Option<f32>) {
        *self.heat_limit.write().unwrap_or_else(|e| e.into_inner()) = limit;
        if limit.is_none() {
            self.set_heat_cap(None);
        }
    }
}

/// Degrees under the heat limit before a shed thread comes back.
pub const HEAT_HYSTERESIS_C: f32 = 5.0;
/// The least time between two of the heat guard's changes.
pub const HEAT_STEP_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// Threads to run next, by the heat limit: one fewer at or over it (down to
/// none: hashing pauses), one more once it is [`HEAT_HYSTERESIS_C`] under it,
/// up to what is asked for.
pub fn heat_step(temp_c: f32, limit_c: f32, running: usize, wanted: usize) -> usize {
    if temp_c >= limit_c {
        running.saturating_sub(1)
    } else if temp_c <= limit_c - HEAT_HYSTERESIS_C && running < wanted {
        running + 1
    } else {
        running.min(wanted)
    }
}

/// Keeps the board under the heat limit by shedding hashing threads, one
/// every [`HEAT_STEP_EVERY`] at most, and adding them back as it cools.
/// `read_temp` reads the CPU's temperature.
pub async fn heat_guard(control: Arc<MinerControl>, read_temp: fn() -> Option<f32>) {
    let mut tick = tokio::time::interval(std::time::Duration::from_secs(5));
    let mut last_change: Option<std::time::Instant> = None;
    loop {
        tick.tick().await;
        let Some(limit) = control.heat_limit() else { continue };
        let Some(temp) = read_temp() else { continue };
        if last_change.is_some_and(|at| at.elapsed() < HEAT_STEP_EVERY) {
            continue;
        }
        let t = control.threads();
        if t.paused {
            continue;
        }
        let (wanted, running) = (t.wanted(), t.running());
        let next = heat_step(temp, limit, running, wanted);
        let cap = (next < wanted).then_some(next);
        if cap != t.heat_cap {
            control.set_heat_cap(cap);
            last_change = Some(std::time::Instant::now());
            match cap {
                Some(0) => tracing::warn!("heat limit: {temp:.1} °C at {limit:.0} °C, hashing paused until it cools"),
                Some(n) if n < running => tracing::warn!("heat limit: {temp:.1} °C at {limit:.0} °C, hashing on {n} of {wanted} threads"),
                Some(n) => tracing::info!("heat limit: cooled to {temp:.1} °C, hashing on {n} of {wanted} threads"),
                None => tracing::info!("heat limit: cooled to {temp:.1} °C, hashing on every thread again ({wanted})"),
            }
        }
    }
}

/// The pool's HTTP API for a stratum address: port 8081 for 3333, 8082 for
/// 3334, and so on.
pub fn pool_api(pool_url: &str) -> String {
    let host = pool_url.replace("stratum+tcp://", "");
    if let Some((ip, port)) = host.rsplit_once(':') {
        if let Ok(port) = port.parse::<u16>() {
            return format!("{}:{}", ip, 8081 + port.saturating_sub(3333));
        }
    }
    host
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MinerToml {
    pub mining: MiningConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiningConfig {
    pub mode: String,
    pub pool_url: Option<String>,
    pub payout_address: Option<String>,
    pub pool_address: Option<String>,
    /// Optional rig name reported to the pool in mining.authorize for the per-worker
    /// breakdown. Absent in older miner.toml files -> deserializes to None -> "default".
    pub worker: Option<String>,
    /// Optional midwimble address, sent in mining.authorize, so a pool that
    /// merge-mines midwimble pays this miner's share of midwimble there.
    #[serde(default)]
    pub midwimble_address: Option<String>,
}

pub enum MinedResult {
    Block(crate::core::Batch),
    Share {
        batch: crate::core::Batch,
        pool_url: String,
        payout_address: String,
    },
}

/// Generate coinbase outputs for a new block being mined.
///
/// In solo mode: uses the node's mining seed to derive WOTS keys/addresses/salts.
/// In pool mode: pays the pool's MSS address and watermarks the miner's payout
/// address into the salt (so the pool can later prove who earned the share).
pub fn generate_coinbase(
    mining_seed: &[u8; 32],
    height: u64,
    total_fees: u64,
    pool_target: Option<([u8; 32], [u8; 32])>, // (Pool MSS Address, Miner Payout Address)
) -> Vec<CoinbaseOutput> {
    let reward = block_reward(height);
    let total_value = reward.saturating_add(total_fees);
    let denominations = decompose_value(total_value);

    denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            match pool_target {
                Some((pool_addr, miner_addr)) => {
                    // POOL MINING MODE
                    // Pay the pool's address, but embed the miner's address in the salt
                    // so the pool can cryptographically verify who did the work.
                    let mut salt = [0u8; 32];
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"pool_share");
                    hasher.update(&miner_addr);
                    hasher.update(&height.to_le_bytes());
                    hasher.update(&(i as u64).to_le_bytes());
                    salt.copy_from_slice(hasher.finalize().as_bytes());

                    CoinbaseOutput { address: pool_addr, value, salt }
                }
                None => {
                    // SOLO MINING MODE (Original Logic)
                    let seed = coinbase_seed(mining_seed, height, i as u64);
                    let owner_pk = wots::keygen(&seed);
                    let address = compute_address(&owner_pk);
                    let salt = coinbase_salt(mining_seed, height, i as u64);
                    
                    CoinbaseOutput { address, value, salt }
                }
            }
        })
        .collect()
}

/// Append a JSONL entry for every coinbase output created at this height.
/// The seed itself is deliberately NOT logged (it is derivable from the
/// node's persistent mining_seed + height + index).
pub fn log_coinbase(
    mining_seed: &[u8; 32],
    data_dir: &PathBuf,
    height: u64,
    total_fees: u64,
) {
    let reward = block_reward(height);
    let total_value = reward + total_fees;
    let denominations = decompose_value(total_value);
    let log_path = data_dir.join("coinbase_seeds.jsonl");

    let entries: Vec<String> = denominations.into_par_iter()
        .enumerate()
        .map(move |(i, value)| {
            let seed = coinbase_seed(mining_seed, height, i as u64);
            let owner_pk = wots::keygen(&seed);
            let address = compute_address(&owner_pk);
            let salt = coinbase_salt(mining_seed, height, i as u64);
            let coin_id = compute_coin_id(&address, value, &salt);
            // NOTE: We intentionally do NOT log the seed (private key).
            // It is derivable from (mining_seed, height, index) when
            // the wallet needs to spend. Logging it in cleartext would
            // allow anyone with filesystem or RPC access to steal funds.
            format!(
                r#"{{"height":{},"index":{},"address":"{}","coin":"{}","value":{},"salt":"{}"}}"#,
                height, i,
                hex::encode(address),
                hex::encode(coin_id),
                value,
                hex::encode(salt)
            )
        })
        .collect();

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true).append(true).open(&log_path)
    {
        use std::io::Write;
        for entry in entries {
            let _ = writeln!(file, "{}", entry);
        }
    }
}

/// Lightweight coordinator for the node's autonomous mining.
/// Holds the configuration that is stable across mining attempts.
/// The heavy per-block preparation + delegation to core::extension::mine_extension
/// is still driven from Node (to keep access to live mempool/state), but
/// this gives the mining logic a named home outside the god module.
#[derive(Clone)]
pub struct MiningCoordinator {
    pub threads: Option<usize>,
    seed: [u8; 32],
    data_dir: PathBuf,
}

impl MiningCoordinator {
    pub fn new(threads: Option<usize>, seed: [u8; 32], data_dir: PathBuf) -> Self {
        Self {
            threads,
            seed,
            data_dir,
        }
    }

    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    /// Convenience wrapper around the free function, using the coordinator's seed.
    pub fn generate_coinbase(
        &self,
        height: u64,
        total_fees: u64,
        pool_target: Option<([u8; 32], [u8; 32])>,
    ) -> Vec<CoinbaseOutput> {
        generate_coinbase(&self.seed, height, total_fees, pool_target)
    }

    /// Convenience wrapper for logging using the coordinator's seed + data_dir.
    pub fn log_coinbase(&self, height: u64, total_fees: u64) {
        log_coinbase(&self.seed, &self.data_dir, height, total_fees);
    }
}

/// Unix seconds now.
fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_secs())
}

/// The job a pool sent: what the hashing threads work on.
#[derive(Clone, Copy)]
struct Job {
    id: u64,
    height: u64,
    m_hash: [u8; 32],
    n_target: [u8; 32],
    s_target: [u8; 32],
}

/// Starts `threads` hashing threads on `job` (none at 0: paused), sending
/// each share found to `shares`. Setting the returned flag stops them.
fn start_job(
    job: Job,
    threads: usize,
    shares: &tokio::sync::mpsc::Sender<(u64, u64, [u8; 32])>,
    hash_counter: &Arc<AtomicU64>,
) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    if threads == 0 {
        return cancel;
    }
    let (nc, hc, tx) = (cancel.clone(), hash_counter.clone(), shares.clone());
    std::thread::spawn(move || loop {
        if nc.load(Ordering::Relaxed) {
            break;
        }
        let Some(res) = crate::core::gpu_mining::mine(job.height, job.m_hash, job.n_target, Some(job.s_target), threads, nc.clone(), hc.clone())
        else {
            break;
        };
        let ext = match res {
            crate::core::extension::MiningResult::Block(ext) | crate::core::extension::MiningResult::Share(ext) => ext,
        };
        let _ = tx.blocking_send((job.id, ext.nonce, ext.final_hash));
    });
    cancel
}

/// How a connection to a pool ended.
#[derive(Debug, PartialEq)]
enum End {
    /// The settings changed: connect again with the new ones, now.
    Settings,
    /// Mining on the fallback, the main pool answered a knock.
    PrimaryBack,
    /// The pool refused this miner's login: wait for new settings.
    Refused,
    /// Lost, or never made. `worked`: a job came through it.
    Lost { worked: bool },
}

/// Mines for the pool in `control`'s settings, reconnecting when they change
/// or the connection is lost, and moving to the fallback pool (if there is
/// one) when the pool can't be reached.
pub async fn run_stratum_client(
    control: Arc<MinerControl>,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<std::sync::RwLock<StratumStats>>,
) {
    // A pool's API that doesn't answer must not hold up its job for ever.
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let mut pool_rx = control.pool.subscribe();
    let mut threads_rx = control.threads.subscribe();
    let timing = control.timing;
    let mut on_fallback = false;
    let mut failures = 0u32;

    loop {
        let settings = pool_rx.borrow_and_update().clone();
        on_fallback &= settings.fallback_url.is_some();
        let url = match (&settings.fallback_url, on_fallback) {
            (Some(fallback), true) => fallback.clone(),
            _ => settings.pool_url.clone(),
        };
        {
            let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
            s.pool_url = url.clone();
            s.on_fallback = on_fallback;
        }
        let probe = on_fallback.then(|| settings.pool_url.clone());
        let end = session(&url, &settings, probe, &control, &mut pool_rx, &mut threads_rx, &hash_counter, &stats, &http).await;
        {
            let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
            s.connected = false;
            s.connected_since = 0;
        }
        match end {
            End::Settings | End::PrimaryBack => {
                on_fallback = false;
                failures = 0;
                continue;
            }
            End::Refused => {
                // Retrying can't fix a refused address; new settings can.
                if pool_rx.changed().await.is_err() {
                    return;
                }
                on_fallback = false;
                failures = 0;
                continue;
            }
            End::Lost { worked } => failures = if worked { 1 } else { failures + 1 },
        }
        if settings.fallback_url.is_some() && failures >= timing.failover_after {
            on_fallback = !on_fallback;
            failures = 0;
            if on_fallback {
                tracing::warn!("{} failed {} times in a row: mining on the fallback pool", url, timing.failover_after);
            } else {
                tracing::warn!("the fallback pool failed {} times in a row: trying the main pool again", timing.failover_after);
            }
        }
        tracing::warn!("disconnected from stratum pool. reconnecting in {}s...", timing.retry.as_secs());
        tokio::select! {
            _ = tokio::time::sleep(timing.retry) => {}
            changed = pool_rx.changed() => {
                if changed.is_err() {
                    return;
                }
                on_fallback = false;
                failures = 0;
            }
        }
    }
}

/// One connection to the pool at `url`: logs in, audits and mines each job,
/// and sends the shares, until the connection ends one of the [`End`] ways.
/// `probe`: the main pool, knocked on now and then while this is the
/// fallback.
#[allow(clippy::too_many_arguments)]
async fn session(
    url: &str,
    settings: &PoolSettings,
    probe: Option<String>,
    control: &MinerControl,
    pool_rx: &mut tokio::sync::watch::Receiver<PoolSettings>,
    threads_rx: &mut tokio::sync::watch::Receiver<Threads>,
    hash_counter: &Arc<AtomicU64>,
    stats: &Arc<std::sync::RwLock<StratumStats>>,
    http_client: &reqwest::Client,
) -> End {
    let timing = control.timing;
    let host = url.replace("stratum+tcp://", "");
    let api_host = pool_api(url);
    let payout_address = &settings.payout_address;

    tracing::info!("stratum client connecting to {} (api: {})", host, api_host);
    let mut stream = match tokio::time::timeout(timing.connect, TcpStream::connect(&host)).await {
        Ok(Ok(stream)) => stream,
        Ok(Err(e)) => {
            tracing::warn!("could not reach the pool at {}: {}", host, e);
            return End::Lost { worked: false };
        }
        Err(_) => {
            tracing::warn!("the pool at {} did not answer in {}s", host, timing.connect.as_secs());
            return End::Lost { worked: false };
        }
    };
    {
        let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
        if s.connected_since != 0 || s.jobs > 0 {
            s.reconnects += 1;
        }
        s.connected = true;
        s.connected_since = unix_now();
    }
    let (read_half, mut write_half) = stream.split();
    // A line at a time, through `Lines`: its `next_line` keeps a
    // half-read line between calls, so it can wait in the select!
    // below beside the share channel. `read_line` could not: when a
    // share was ready while a notify was half-received (a notify
    // carries the whole block template, so it arrives in pieces), the
    // read was cancelled and its first half thrown away (tokio: "not
    // cancel safe"). The rest failed to parse and was skipped without
    // a word, the miner kept grinding the job it had, and every share
    // after the first few was a "Stale job".
    let mut lines = BufReader::new(read_half).lines();

    // A pool that merge-mines midwimble reads an optional third
    // parameter: the midwimble address to pay this miner's share to.
    let mut auth_params = vec![serde_json::json!(payout_address), serde_json::json!(settings.worker)];
    if let Some(mw) = &settings.midwimble_address {
        auth_params.push(serde_json::json!(mw));
    }
    let auth_req = serde_json::json!({
        "id": 1, "method": "mining.authorize", "params": auth_params
    });
    let _ = write_half.write_all(format!("{}\n", auth_req).as_bytes()).await;

    // A fresh channel for shares on each connection: the job, the nonce,
    // and the share's hash (for the best share).
    let (share_tx, mut share_rx) = tokio::sync::mpsc::channel::<(u64, u64, [u8; 32])>(100);
    // When each share in flight was sent, oldest first: the pool
    // answers them in order.
    let mut in_flight: std::collections::VecDeque<std::time::Instant> = std::collections::VecDeque::new();

    let mut s_target = [0xff; 32];
    s_target[0] = 0x00;
    s_target[1] = 0x0f;

    let mut job: Option<Job> = None;
    let mut cancel = Arc::new(AtomicBool::new(false));
    let mut worked = false;
    threads_rx.borrow_and_update();

    // While on the fallback: a knock on the main pool now and then.
    let (back_tx, mut back_rx) = tokio::sync::oneshot::channel::<()>();
    let probing = probe.is_some();
    let prober = probe.map(|primary| {
        let every = timing.probe_every;
        let host = primary.replace("stratum+tcp://", "");
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let knock = tokio::time::timeout(std::time::Duration::from_secs(5), TcpStream::connect(&host)).await;
                if matches!(knock, Ok(Ok(_))) {
                    let _ = back_tx.send(());
                    return;
                }
            }
        })
    });

    let idle = tokio::time::sleep(timing.idle);
    tokio::pin!(idle);

    let end = loop {
        tokio::select! {
            message = next_pool_message(&mut lines) => {
                idle.as_mut().reset(tokio::time::Instant::now() + timing.idle);
                let msg = match message {
                    PoolMessage::Message(msg) => msg,
                    PoolMessage::Unreadable(why) => {
                        tracing::warn!("unreadable message from the pool, skipped: {}", why);
                        continue;
                    }
                    PoolMessage::Closed(why) => {
                        tracing::warn!("stratum connection closed: {}", why);
                        break End::Lost { worked };
                    }
                };

                if msg["method"] == "mining.notify" {
                    let Some((job_id, m_hash, template_val)) = parse_notify(&msg) else {
                        tracing::warn!("malformed mining.notify from the pool, skipped: {:.200}", msg.to_string());
                        continue;
                    };
                    // The block's height picks the proof of work. A pool that
                    // predates Midstate Walk does not send it, and its node
                    // (not having switched either) wants the BLAKE3 chain.
                    let job_height = notify_height(&msg);
                    let template_val = &template_val;
                    if let Some(t) = msg["params"].get(3).and_then(|v| v.as_str()) {
                        let mut bytes = [0u8; 32];
                        if hex::decode_to_slice(t, &mut bytes).is_ok() { s_target = bytes; }
                    }

                    let batch: crate::core::Batch = match serde_json::from_value(template_val.clone()) {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::error!("failed to parse batch template: {}", e);
                            break End::Lost { worked };
                        }
                    };
                    let header = batch.header();
                    let n_target = batch.target;
                    {
                        let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
                        s.network_target = n_target;
                        s.share_target = s_target;
                        s.job_height = job_height;
                        s.jobs += 1;
                        s.last_job_unix = unix_now();
                    }
                    let calculated_hash = crate::core::types::compute_header_hash(&header);

                    if calculated_hash != m_hash {
                        tracing::error!("audit failed: template header hash mismatch. disconnecting.");
                        break End::Lost { worked };
                    }

                    if let Some(pool_cb) = batch.coinbase.first() {
                        let claimed_root = hex::encode(pool_cb.salt);

                        // Ask for the proof behind this job's own root: a pool
                        // that keeps its last few trees serves it even if a
                        // newer job came out meanwhile (older pools ignore it).
                        if let Ok(res) = http_client.get(format!("http://{}/api/proof?address={}&root={}", api_host, payout_address, claimed_root)).send().await {
                            if let Ok(proof_data) = res.json::<serde_json::Value>().await {
                                if proof_data.get("error").is_none() {
                                    let Ok(payout_bytes) = crate::core::types::parse_address_flexible(payout_address) else {
                                        tracing::error!("the payout address {} is not an address. fix it and the miner reconnects.", payout_address);
                                        break End::Refused;
                                    };
                                    let score = proof_data["score"].as_u64().unwrap_or(0);
                                    let mut data = [0u8; 40];
                                    data[0..32].copy_from_slice(&payout_bytes);
                                    data[32..40].copy_from_slice(&score.to_le_bytes());
                                    let mut current_hash = crate::core::types::hash(&data);

                                    if let (Some(proof_array), Some(mut current_idx)) = (proof_data["proof"].as_array(), proof_data["index"].as_u64()) {
                                        for sibling_hex in proof_array {
                                            let mut sibling = [0u8; 32];
                                            if sibling_hex.as_str().map(|h| hex::decode_to_slice(h, &mut sibling)).is_none_or(|r| r.is_err()) {
                                                tracing::error!("audit failed: the pool's proof holds a malformed hash. disconnecting.");
                                                break;
                                            }
                                            if current_idx % 2 == 1 {
                                                current_hash = crate::core::types::hash_concat(&sibling, &current_hash);
                                            } else {
                                                current_hash = crate::core::types::hash_concat(&current_hash, &sibling);
                                            }
                                            current_idx /= 2;
                                        }
                                    }

                                    if hex::encode(current_hash) != claimed_root {
                                        tracing::error!("audit failed: merkle root mismatch (computed {}, claimed {}). disconnecting.", hex::encode(current_hash), claimed_root);
                                        break End::Lost { worked };
                                    }

                                    if score > 0 && !batch.coinbase.iter().any(|cb| cb.address == payout_bytes) {
                                        tracing::error!("audit failed: omitted from payout array despite score of {}. disconnecting.", score);
                                        break End::Lost { worked };
                                    }
                                }
                            }
                        }
                    }
                    tracing::info!("audit passed. starting job {}", job_id);
                    worked = true;

                    // Stop the previous job's threads, start this one's.
                    cancel.store(true, Ordering::Relaxed);
                    let next = Job { id: job_id, height: job_height, m_hash, n_target, s_target };
                    cancel = start_job(next, control.threads().running(), &share_tx, hash_counter);
                    job = Some(next);
                } else if msg["id"].as_u64() == Some(1) && msg["result"].as_bool() == Some(false) {
                    // The pool refused the login (a bad payout or
                    // midwimble address). Retrying can't fix that.
                    tracing::error!(
                        "pool refused this miner: {}. fix the address on the dashboard or the command line; the miner waits for new settings.",
                        msg["error"].as_str().unwrap_or("no reason given")
                    );
                    break End::Refused;
                } else if msg["result"].as_bool() == Some(true) && msg["id"].as_u64() == Some(2) {
                    tracing::info!("share accepted");
                    let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
                    if let Some(sent) = in_flight.pop_front() {
                        s.latency_ms = Some(sent.elapsed().as_millis() as u64);
                    }
                    s.accepted_shares += 1;
                    s.last_share_unix = unix_now();
                } else if let Some(err) = msg["error"].as_str() {
                    tracing::warn!("share rejected: {}", err);
                    let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
                    s.rejected_shares += 1;
                    if let Some(sent) = in_flight.pop_front() {
                        s.latency_ms = Some(sent.elapsed().as_millis() as u64);
                    }
                }
            }
            Some((job_id, nonce, share_hash)) = share_rx.recv() => {
                {
                    let mut s = stats.write().unwrap_or_else(|e| e.into_inner());
                    s.shares_found += 1;
                    if s.best_share.is_none_or(|best| share_hash < best) {
                        s.best_share = Some(share_hash);
                    }
                }
                in_flight.push_back(std::time::Instant::now());
                if in_flight.len() > 100 {
                    in_flight.pop_front();
                }
                let submit_req = serde_json::json!({
                    "id": 2, "method": "mining.submit", "params": [payout_address, job_id, nonce]
                });
                let _ = write_half.write_all(format!("{}\n", submit_req).as_bytes()).await;
            }
            changed = pool_rx.changed() => {
                if changed.is_ok() {
                    tracing::info!("new pool settings: reconnecting");
                    break End::Settings;
                }
                break End::Lost { worked };
            }
            changed = threads_rx.changed() => {
                if changed.is_err() {
                    break End::Lost { worked };
                }
                let running = threads_rx.borrow_and_update().running();
                if let Some(j) = job {
                    cancel.store(true, Ordering::Relaxed);
                    cancel = start_job(j, running, &share_tx, hash_counter);
                }
                tracing::info!("hashing on {} thread{}", running, if running == 1 { "" } else { "s" });
            }
            _ = &mut back_rx, if probing => {
                tracing::info!("the main pool answers again: moving back to it");
                break End::PrimaryBack;
            }
            _ = &mut idle => {
                tracing::warn!("no word from the pool for {}s: reconnecting", timing.idle.as_secs());
                break End::Lost { worked };
            }
        }
    };
    cancel.store(true, Ordering::Relaxed);
    if let Some(p) = prober {
        p.abort();
    }
    end
}

/// What the pool sent: a message, a line that is not one (logged and skipped,
/// where it used to vanish silently), or the end of the connection.
enum PoolMessage {
    Message(serde_json::Value),
    Unreadable(String),
    Closed(String),
}

/// The pool's next message. Cancel-safe, so it can wait in a `select!`: the
/// only await is `Lines::next_line`, which keeps a half-read line for the
/// next call.
async fn next_pool_message<R: AsyncBufRead + Unpin>(lines: &mut tokio::io::Lines<R>) -> PoolMessage {
    match lines.next_line().await {
        Ok(Some(line)) => match serde_json::from_str(&line) {
            Ok(msg) => PoolMessage::Message(msg),
            Err(e) => PoolMessage::Unreadable(format!("{} ({} bytes: {:.80})", e, line.len(), line)),
        },
        Ok(None) => PoolMessage::Closed("the pool closed it".into()),
        Err(e) => PoolMessage::Closed(e.to_string()),
    }
}

/// The height of the block a notify's job mines (its fifth parameter), which
/// selects the proof of work. Pools that predate Midstate Walk send four
/// parameters; their jobs are BLAKE3-chain jobs, which height 0 selects.
fn notify_height(msg: &serde_json::Value) -> u64 {
    msg["params"].get(4).and_then(|v| v.as_u64()).unwrap_or(0)
}

/// A notify's job id, mining hash and block template, or `None` if any is
/// missing or malformed (it used to panic the miner).
fn parse_notify(msg: &serde_json::Value) -> Option<(u64, [u8; 32], serde_json::Value)> {
    let params = msg["params"].as_array()?;
    let job_id = params.first()?.as_u64()?;
    let mut m_hash = [0u8; 32];
    hex::decode_to_slice(params.get(1)?.as_str()?, &mut m_hash).ok()?;
    Some((job_id, m_hash, params.get(2)?.clone()))
}

#[cfg(test)]
mod stratum_tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;

    /// A notify that arrives in pieces while shares keep the select! busy is
    /// still read whole. With `read_line`, each share ready mid-notify threw
    /// the first part of the line away, and the rest was skipped as unreadable.
    #[tokio::test]
    async fn a_notify_arriving_in_pieces_survives_a_busy_select() {
        let (pool_side, miner_side) = tokio::io::duplex(64);
        let notify = serde_json::json!({
            "id": null,
            "method": "mining.notify",
            "params": [7, hex::encode([0xab_u8; 32]), {"pad": "x".repeat(20_000)}, hex::encode([0x0f_u8; 32])],
        })
        .to_string();
        let sent = notify.clone();
        tokio::spawn(async move {
            let mut pool_side = pool_side;
            for piece in sent.as_bytes().chunks(4_000) {
                pool_side.write_all(piece).await.unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            pool_side.write_all(b"\nnot json at all\n").await.unwrap();
        });

        let mut lines = BufReader::new(miner_side).lines();
        // Shares, ready every millisecond: they win the select! many times
        // while the notify is in flight.
        let mut shares = tokio::time::interval(Duration::from_millis(1));
        let (mut shares_sent, mut received) = (0, Vec::new());
        while received.len() < 2 {
            tokio::select! {
                message = next_pool_message(&mut lines) => received.push(message),
                _ = shares.tick() => shares_sent += 1,
            }
        }
        assert!(shares_sent > 5, "the select! was busy ({shares_sent} shares)");
        match &received[0] {
            PoolMessage::Message(msg) => {
                assert_eq!(msg.to_string(), notify);
                let (job_id, m_hash, template) = parse_notify(msg).expect("a whole notify");
                assert_eq!((job_id, m_hash), (7, [0xab; 32]));
                assert_eq!(template["pad"].as_str().map(str::len), Some(20_000));
            }
            _ => panic!("the notify was lost"),
        }
        assert!(matches!(&received[1], PoolMessage::Unreadable(why) if why.contains("not json at all")));
    }

    #[test]
    fn a_notify_carries_the_height_that_picks_the_proof_of_work() {
        use crate::core::types::{pow_algo_at, PowAlgo, WALK_ACTIVATION_HEIGHT};
        let m = hex::encode([1u8; 32]);
        let old_pool = serde_json::json!({"method": "mining.notify", "params": [1, m, {}, "00"]});
        assert_eq!(pow_algo_at(notify_height(&old_pool)), PowAlgo::Blake3Chain);
        let walk = serde_json::json!({"method": "mining.notify", "params": [1, m, {}, "00", WALK_ACTIVATION_HEIGHT]});
        assert_eq!(notify_height(&walk), WALK_ACTIVATION_HEIGHT);
        assert_eq!(pow_algo_at(notify_height(&walk)), PowAlgo::Walk);
    }

    /// A pool that accepts connections and hands each one's first line (the
    /// login) over, then keeps the connection open and says nothing.
    async fn silent_pool() -> (String, tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("stratum+tcp://{}", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            loop {
                let Ok((conn, _)) = listener.accept().await else { return };
                let tx = tx.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(conn).lines();
                    if let Ok(Some(line)) = lines.next_line().await {
                        let _ = tx.send(serde_json::from_str(&line).unwrap_or_default());
                    }
                    while let Ok(Some(_)) = lines.next_line().await {}
                });
            }
        });
        (url, rx, task)
    }

    /// A port nothing listens on.
    async fn closed_port() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        format!("stratum+tcp://{}", listener.local_addr().unwrap())
    }

    fn pool(url: &str, payout: &str) -> PoolSettings {
        PoolSettings {
            pool_url: url.into(),
            fallback_url: None,
            payout_address: payout.into(),
            worker: "rig".into(),
            midwimble_address: None,
        }
    }

    fn quick() -> StratumTiming {
        StratumTiming {
            connect: Duration::from_secs(2),
            idle: Duration::from_secs(60),
            retry: Duration::from_millis(20),
            failover_after: 3,
            probe_every: Duration::from_millis(100),
        }
    }

    async fn login(rx: &mut tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>) -> serde_json::Value {
        tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.expect("a login in time").expect("a login")
    }

    fn client(control: &Arc<MinerControl>) -> (Arc<std::sync::RwLock<StratumStats>>, tokio::task::JoinHandle<()>) {
        let stats = Arc::new(std::sync::RwLock::new(StratumStats::default()));
        let task = tokio::spawn(run_stratum_client(control.clone(), Arc::new(AtomicU64::new(0)), stats.clone()));
        (stats, task)
    }

    /// New settings from the dashboard reach a running miner: it leaves the
    /// pool it is on and logs in to the new one as the new payee.
    #[tokio::test]
    async fn new_pool_settings_reconnect_at_once() {
        let (a, mut a_logins, _a) = silent_pool().await;
        let (b, mut b_logins, _b) = silent_pool().await;
        let control = Arc::new(MinerControl::new(pool(&a, "alice"), 1, None).with_timing(quick()));
        let (stats, task) = client(&control);

        let first = login(&mut a_logins).await;
        assert_eq!(first["method"], "mining.authorize");
        assert_eq!(first["params"], serde_json::json!(["alice", "rig"]));

        let mut next = pool(&b, "bob");
        next.midwimble_address = Some("mw1xyz".into());
        assert!(control.set_pool(next.clone()));
        assert!(!control.set_pool(next), "the same settings again change nothing");
        let second = login(&mut b_logins).await;
        assert_eq!(second["params"], serde_json::json!(["bob", "rig", "mw1xyz"]));
        assert_eq!(stats.read().unwrap().pool_url, b);
        task.abort();
    }

    /// A pool that can't be reached three times running is left for the
    /// fallback, and the miner goes back once the main pool answers a knock.
    #[tokio::test]
    async fn an_unreachable_pool_fails_over_and_comes_back() {
        let main = closed_port().await;
        let (fallback, mut fallback_logins, _f) = silent_pool().await;
        let mut settings = pool(&main, "alice");
        settings.fallback_url = Some(fallback.clone());
        let control = Arc::new(MinerControl::new(settings, 1, None).with_timing(quick()));
        let (stats, task) = client(&control);

        assert_eq!(login(&mut fallback_logins).await["params"][0], "alice");
        {
            let s = stats.read().unwrap();
            assert!(s.on_fallback && s.pool_url == fallback, "on the fallback: {}", s.pool_url);
        }

        // The main pool comes up where it was.
        let listener = tokio::net::TcpListener::bind(main.trim_start_matches("stratum+tcp://")).await.unwrap();
        let (conn, _) = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                // The knock connects and closes; the miner's own login follows.
                let (conn, addr) = listener.accept().await.unwrap();
                let mut lines = BufReader::new(conn).lines();
                if let Ok(Ok(Some(line))) = tokio::time::timeout(Duration::from_millis(500), lines.next_line()).await {
                    break (line, addr);
                }
            }
        })
        .await
        .expect("back on the main pool");
        let back: serde_json::Value = serde_json::from_str(&conn).unwrap();
        assert_eq!(back["method"], "mining.authorize");
        assert!(!stats.read().unwrap().on_fallback);
        task.abort();
    }

    /// A pool that keeps the connection but says nothing is left, not waited
    /// on for ever.
    #[tokio::test]
    async fn a_silent_pool_is_left_after_the_idle_time() {
        let (a, mut logins, _a) = silent_pool().await;
        let mut timing = quick();
        timing.idle = Duration::from_millis(200);
        let control = Arc::new(MinerControl::new(pool(&a, "alice"), 1, None).with_timing(timing));
        let (stats, task) = client(&control);
        login(&mut logins).await;
        login(&mut logins).await;
        assert!(stats.read().unwrap().reconnects >= 1 || stats.read().unwrap().connected_since > 0);
        task.abort();
    }

    /// A refused login waits for new settings rather than retrying an
    /// address that will be refused again.
    #[tokio::test]
    async fn a_refused_login_waits_for_new_settings() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("stratum+tcp://{}", listener.local_addr().unwrap());
        let (tx, mut logins) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
        tokio::spawn(async move {
            loop {
                let (mut conn, _) = listener.accept().await.unwrap();
                let tx = tx.clone();
                tokio::spawn(async move {
                    let (r, mut w) = conn.split();
                    let mut lines = BufReader::new(r).lines();
                    if let Ok(Some(line)) = lines.next_line().await {
                        let login: serde_json::Value = serde_json::from_str(&line).unwrap();
                        let refused = login["params"][0] == "bad";
                        let _ = tx.send(login);
                        if refused {
                            let _ = w.write_all(b"{\"id\":1,\"result\":false,\"error\":\"invalid address\"}\n").await;
                        }
                    }
                    while let Ok(Some(_)) = lines.next_line().await {}
                });
            }
        });
        let control = Arc::new(MinerControl::new(pool(&url, "bad"), 1, None).with_timing(quick()));
        let (_stats, task) = client(&control);
        assert_eq!(login(&mut logins).await["params"][0], "bad");
        assert!(
            tokio::time::timeout(Duration::from_millis(300), logins.recv()).await.is_err(),
            "no retry with the refused address"
        );
        control.set_pool(pool(&url, "good"));
        assert_eq!(login(&mut logins).await["params"][0], "good");
        task.abort();
    }

    #[test]
    fn the_heat_limit_sheds_a_thread_at_a_time_and_gives_them_back() {
        // At or over the limit: one fewer, down to none.
        assert_eq!(heat_step(80.0, 80.0, 4, 4), 3);
        assert_eq!(heat_step(83.0, 80.0, 1, 4), 0);
        assert_eq!(heat_step(90.0, 80.0, 0, 4), 0);
        // In the band just under it: as it is.
        assert_eq!(heat_step(77.0, 80.0, 2, 4), 2);
        // Five degrees under: one more, up to what is asked for.
        assert_eq!(heat_step(75.0, 80.0, 2, 4), 3);
        assert_eq!(heat_step(60.0, 80.0, 4, 4), 4);
        assert_eq!(heat_step(60.0, 80.0, 6, 4), 4, "never above what is asked for");

        let t = Threads { want: 4, heat_cap: Some(2), paused: false };
        assert_eq!((t.wanted(), t.running()), (4, 2));
        assert_eq!(Threads { paused: true, ..t }.running(), 0);
        assert_eq!(Threads { want: 0, heat_cap: None, paused: false }.running(), cores());

        let control = MinerControl::new(pool("x:1", "a"), 4, Some(80.0));
        control.set_heat_cap(Some(1));
        control.set_heat_limit(None);
        assert_eq!(control.threads().heat_cap, None, "no limit, nothing held back");
    }

    #[test]
    fn malformed_notifies_are_refused_not_panicked_on() {
        let ok = serde_json::json!({"method": "mining.notify", "params": [1, hex::encode([1u8; 32]), {}]});
        assert!(parse_notify(&ok).is_some());
        for bad in [
            serde_json::json!({"method": "mining.notify"}),
            serde_json::json!({"method": "mining.notify", "params": []}),
            serde_json::json!({"method": "mining.notify", "params": ["one", hex::encode([1u8; 32]), {}]}),
            serde_json::json!({"method": "mining.notify", "params": [1, "zz", {}]}),
            serde_json::json!({"method": "mining.notify", "params": [1, hex::encode([1u8; 32])]}),
        ] {
            assert!(parse_notify(&bad).is_none(), "{bad}");
        }
    }
}
