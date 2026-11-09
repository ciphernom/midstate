//! The Axe dashboard for `midstate miner` (`hasher_dashboard.html`), served by
//! the hasher itself, with no node and no chain behind it. It is laid out the
//! way a Bitaxe's AxeOS is: the numbers a miner watches on top, a hashrate
//! chart, then power, heat, performance, the pool and the system, and a tab of
//! the miner's own log.
//!
//! # What it shows
//!
//! - Hashrate now (the last minute), over ten minutes and an hour, and each
//!   backend's (CPU threads and CUDA GPUs), from a history the miner keeps for
//!   a day, so the chart survives a reload.
//! - Efficiency: attempts per joule, from a Raspberry Pi 5's power monitor
//!   (`vcgencmd pmic_read_adc`: the SoC's rails, not what the wall supplies).
//! - Shares found, accepted and rejected, the pool's answer time, and the best
//!   share by difficulty (`2^256 / hash`: the attempts it takes on average),
//!   this run and ever (kept in `miner-best.json` beside the miner).
//! - The job's share and network difficulty, so how far a share is from a
//!   block, and the pool's view: this miner's part of the round, what a pool
//!   block pays it, and whether merged mining runs.
//! - The board: temperature, clock, fan, throttling and under-voltage, and the
//!   system's load, memory and uptime.
//!
//! # Cost
//!
//! The hashing loop never sees the dashboard: it bumps counters it already
//! had. A sampler reads the counters and sensors every five seconds, and
//! `vcgencmd` at most every ten. The pool's stats are asked for at most every
//! thirty seconds, and only while someone is looking.
//!
//! # Access
//!
//! It listens on 127.0.0.1 unless told otherwise (`--dashboard`), and then
//! answers only requests addressed to localhost, so a web page cannot reach it
//! by pointing a name at 127.0.0.1. It changes nothing: every route is a GET,
//! and the node's settings routes (Wi-Fi, clocks, mining mode, the bench) are
//! not served at all, so there is nothing on it to protect.

use crate::mining::StratumStats;
use axum::extract::{Query, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Where `midstate miner` serves its dashboard unless told otherwise.
pub const DEFAULT_BIND: &str = "127.0.0.1:8546";
/// How often the pool's stats are asked for, at most.
const POOL_STATS_EVERY: Duration = Duration::from_secs(30);
/// How often the sampler reads the counters and sensors.
const SAMPLE_EVERY: Duration = Duration::from_secs(5);
/// Samples kept: a day of them.
const KEEP_SAMPLES: usize = 24 * 3600 / 5;
/// The most points a chart is sent.
const CHART_POINTS: usize = 360;
/// Where the best share ever is kept, beside the miner.
const BEST_FILE: &str = "miner-best.json";

/// What the miner was started with.
#[derive(Clone, Debug, Serialize)]
pub struct MinerSettings {
    pub pool_url: String,
    pub payout_address: String,
    pub worker: String,
    pub midwimble_address: Option<String>,
    /// CPU threads asked for (0: every core).
    pub threads: usize,
    /// "auto", "gpu" or "cpu".
    pub backend: String,
}

/// One reading of the miner and its board.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
struct Sample {
    /// Unix seconds.
    t: u64,
    /// The attempt counter then.
    hashes: u64,
    temp_c: Option<f32>,
    watts: Option<f32>,
}

/// The best share ever, as kept in [`BEST_FILE`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Best {
    /// The share's hash (hex).
    hash: String,
    /// Unix seconds.
    unix: u64,
}

#[derive(Clone)]
struct Dash {
    settings: Arc<MinerSettings>,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<StratumStats>>,
    started: Instant,
    /// The pool's HTTP API (`host:8081` for a stratum port of 3333).
    pool_api: String,
    http: reqwest::Client,
    /// The pool's view of this miner, and when it was asked.
    pool: Arc<tokio::sync::Mutex<(Option<Instant>, Option<serde_json::Value>)>>,
    history: Arc<RwLock<VecDeque<Sample>>>,
    best_ever: Arc<RwLock<Option<Best>>>,
    loopback_only: bool,
}

/// The pool's API address, as the stratum client works it out.
pub fn pool_api(pool_url: &str) -> String {
    let host = pool_url.replace("stratum+tcp://", "");
    if let Some((ip, port)) = host.rsplit_once(':') {
        if let Ok(port) = port.parse::<u16>() {
            return format!("{}:{}", ip, 8081 + port.saturating_sub(3333));
        }
    }
    host
}

/// Serves the dashboard on `bind` until the process ends.
pub async fn serve(
    bind: SocketAddr,
    settings: MinerSettings,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<StratumStats>>,
) -> anyhow::Result<()> {
    let best = std::fs::read(BEST_FILE).ok().and_then(|b| serde_json::from_slice::<Best>(&b).ok());
    let dash = Dash {
        pool_api: pool_api(&settings.pool_url),
        settings: Arc::new(settings),
        hash_counter,
        stats,
        started: Instant::now(),
        http: reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?,
        pool: Arc::new(tokio::sync::Mutex::new((None, None))),
        history: Arc::new(RwLock::new(VecDeque::new())),
        best_ever: Arc::new(RwLock::new(best)),
        loopback_only: bind.ip().is_loopback(),
    };
    tokio::spawn(sampler(dash.clone()));
    let app = router(dash);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    if bind.ip().is_loopback() {
        tracing::info!("miner dashboard: http://{bind}/axe (this machine only; --dashboard <ip:port> to change)");
    } else {
        tracing::warn!("miner dashboard: http://{bind}/axe, reachable from your network. It is read-only.");
    }
    axum::serve(listener, app).await?;
    Ok(())
}

fn router(dash: Dash) -> Router {
    let loopback_only = dash.loopback_only;
    Router::new()
        .route("/", get(page))
        .route("/axe", get(page))
        .route("/axe/stats", get(stats))
        .route("/axe/history", get(history))
        .route("/axe/logs", get(logs))
        .route("/axe/config", get(config))
        .route("/midstate.css", get(super::handlers::midstate_css))
        .with_state(dash)
        .layer(middleware::from_fn(move |req: Request, next: Next| async move {
            if loopback_only && !addressed_to_localhost(&req) {
                return StatusCode::FORBIDDEN.into_response();
            }
            next.run(req).await
        }))
}

/// Whether a request names this machine in its Host header: what a browser
/// on it sends, and what a page that points another name at 127.0.0.1
/// (DNS rebinding) does not.
fn addressed_to_localhost(req: &Request) -> bool {
    let Some(host) = req.headers().get(header::HOST).and_then(|h| h.to_str().ok()) else {
        return false;
    };
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map_or(host, |(n, _)| n)
    };
    matches!(name.to_ascii_lowercase().as_str(), "localhost" | "127.0.0.1" | "::1")
}

async fn page() -> Html<&'static str> {
    Html(include_str!("hasher_dashboard.html"))
}

/// The mining settings, read-only.
async fn config(State(d): State<Dash>) -> Response {
    let s = &d.settings;
    Json(serde_json::json!({
        "mode": "pool",
        "pool_url": s.pool_url,
        "payout_address": s.payout_address,
        "worker": s.worker,
        "midwimble_address": s.midwimble_address,
        "threads": s.threads,
        "backend": s.backend,
        "read_only": true,
    }))
    .into_response()
}

// ── Sampling ────────────────────────────────────────────────────────────────

/// Reads the counters and sensors every [`SAMPLE_EVERY`], keeps a day of
/// them, and keeps the best share ever.
async fn sampler(d: Dash) {
    let mut tick = tokio::time::interval(SAMPLE_EVERY);
    loop {
        tick.tick().await;
        let sample = Sample {
            t: unix_now(),
            hashes: d.hash_counter.load(Ordering::Relaxed),
            temp_c: temp_c(),
            watts: pmic().await.map(|p| p.watts),
        };
        {
            let mut h = d.history.write().unwrap_or_else(|e| e.into_inner());
            h.push_back(sample);
            while h.len() > KEEP_SAMPLES {
                h.pop_front();
            }
        }
        let session = d.stats.read().unwrap_or_else(|e| e.into_inner()).best_share;
        if let Some(hash) = session {
            let better = {
                let best = d.best_ever.read().unwrap_or_else(|e| e.into_inner());
                best.as_ref().and_then(|b| hex32(&b.hash)).is_none_or(|old| hash < old)
            };
            if better {
                let best = Best { hash: hex::encode(hash), unix: unix_now() };
                if let Ok(bytes) = serde_json::to_vec(&best) {
                    let _ = std::fs::write(BEST_FILE, bytes);
                }
                *d.best_ever.write().unwrap_or_else(|e| e.into_inner()) = Some(best);
            }
        }
    }
}

/// Attempts a second over the last `secs` of `history`, if it covers at
/// least half of them.
fn rate_over(history: &VecDeque<Sample>, now: Sample, secs: u64) -> Option<f64> {
    let from = now.t.saturating_sub(secs);
    let first = history.iter().find(|s| s.t >= from)?;
    // At least half the window, so an hour's rate waits for half an hour.
    let span = now.t.checked_sub(first.t).filter(|s| *s * 2 >= secs)?;
    (span > 0).then(|| now.hashes.saturating_sub(first.hashes) as f64 / span as f64)
}

#[derive(Deserialize)]
struct HistoryQuery {
    /// How far back, in minutes (15 by default, a day at most).
    minutes: Option<u64>,
}

/// The chart: attempts a second, temperature and power over the last
/// `minutes`, at most [`CHART_POINTS`] points.
async fn history(State(d): State<Dash>, Query(q): Query<HistoryQuery>) -> Json<serde_json::Value> {
    let minutes = q.minutes.unwrap_or(15).clamp(1, 24 * 60);
    let samples: Vec<Sample> = {
        let h = d.history.read().unwrap_or_else(|e| e.into_inner());
        let from = unix_now().saturating_sub(minutes * 60);
        h.iter().copied().filter(|s| s.t >= from).collect()
    };
    Json(serde_json::json!({ "minutes": minutes, "points": chart_points(&samples, CHART_POINTS) }))
}

/// Samples as chart points `[t, attempts/s, temp, watts]`, each averaging the
/// samples it stands for so a long range is not one sample in many.
fn chart_points(samples: &[Sample], max: usize) -> Vec<serde_json::Value> {
    if samples.len() < 2 || max == 0 {
        return Vec::new();
    }
    let per = samples.len().div_ceil(max).max(1);
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < samples.len() {
        let j = (i + per).min(samples.len() - 1);
        let (a, b) = (samples[i], samples[j]);
        let span = b.t.saturating_sub(a.t);
        let rate = if span > 0 { b.hashes.saturating_sub(a.hashes) as f64 / span as f64 } else { 0.0 };
        let mean = |f: fn(&Sample) -> Option<f32>| {
            let v: Vec<f32> = samples[i + 1..=j].iter().filter_map(f).collect();
            (!v.is_empty()).then(|| v.iter().sum::<f32>() / v.len() as f32)
        };
        out.push(serde_json::json!([b.t, rate, mean(|s| s.temp_c), mean(|s| s.watts)]));
        i = j;
    }
    out
}

// ── The board ───────────────────────────────────────────────────────────────

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_secs())
}

fn read_trim(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().trim_end_matches('\0').to_string())
}

fn temp_c() -> Option<f32> {
    read_trim("/sys/class/thermal/thermal_zone0/temp")?.parse::<f32>().ok().map(|m| m / 1000.0).filter(|t| *t > 0.0)
}

/// What a Raspberry Pi 5's power monitor says.
#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq)]
pub struct Pmic {
    /// The rails it measures, added up: the SoC and what hangs off the PMIC,
    /// not the losses before it, so a little under what the wall supplies.
    pub watts: f32,
    /// The CPU cores' supply.
    pub core_v: Option<f32>,
    /// The 5 V input.
    pub input_v: Option<f32>,
}

/// `vcgencmd pmic_read_adc`'s readings: every rail with both a current
/// (`NAME_A current(n)=1.2A`) and a voltage (`NAME_V volt(n)=0.9V`) adds
/// their product.
pub fn parse_pmic(text: &str) -> Option<Pmic> {
    let mut amps: HashMap<&str, f32> = HashMap::new();
    let mut volts: HashMap<&str, f32> = HashMap::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let (Some(name), Some(reading)) = (words.next(), words.next()) else { continue };
        let Some(value) = reading.split('=').nth(1) else { continue };
        let Ok(value) = value.trim_end_matches(['A', 'V']).parse::<f32>() else { continue };
        if let Some(rail) = name.strip_suffix("_A") {
            amps.insert(rail, value);
        } else if let Some(rail) = name.strip_suffix("_V") {
            volts.insert(rail, value);
        }
    }
    let pairs: Vec<f32> = amps.iter().filter_map(|(rail, a)| volts.get(rail).map(|v| a * v)).collect();
    if pairs.is_empty() {
        return None;
    }
    Some(Pmic {
        watts: pairs.iter().sum(),
        core_v: volts.get("VDD_CORE").copied(),
        input_v: volts.get("EXT5V").copied(),
    })
}

/// The power monitor, read at most every ten seconds. `None` off a Pi 5, or
/// where `vcgencmd` cannot reach the firmware.
async fn pmic() -> Option<Pmic> {
    static LAST: std::sync::Mutex<Option<(Instant, Option<Pmic>)>> = std::sync::Mutex::new(None);
    if let Ok(last) = LAST.lock() {
        if let Some((at, p)) = *last {
            if at.elapsed() < Duration::from_secs(10) {
                return p;
            }
        }
    }
    let p = match tokio::time::timeout(
        Duration::from_secs(2),
        tokio::process::Command::new("vcgencmd").arg("pmic_read_adc").output(),
    )
    .await
    {
        Ok(Ok(out)) if out.status.success() => parse_pmic(&String::from_utf8_lossy(&out.stdout)),
        _ => None,
    };
    if let Ok(mut last) = LAST.lock() {
        *last = Some((Instant::now(), p));
    }
    p
}

/// The fan's speed and duty (percent), on a Pi 5 with its active cooler.
fn fan() -> Option<(Option<u32>, Option<u32>)> {
    let dir = std::fs::read_dir("/sys/devices/platform/cooling_fan/hwmon").ok()?.flatten().next()?.path();
    let rpm = std::fs::read_to_string(dir.join("fan1_input")).ok().and_then(|s| s.trim().parse().ok());
    let pct = std::fs::read_to_string(dir.join("pwm1"))
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .map(|pwm| (pwm * 100 + 127) / 255);
    (rpm.is_some() || pct.is_some()).then_some((rpm, pct))
}

/// Load, memory and uptime, from `/proc`.
fn system() -> serde_json::Value {
    let load: Vec<f32> = read_trim("/proc/loadavg")
        .map(|l| l.split_whitespace().take(3).filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_default();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kb = |key: &str| {
        meminfo
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    serde_json::json!({
        "hostname": read_trim("/proc/sys/kernel/hostname"),
        "model": read_trim("/sys/firmware/devicetree/base/model"),
        "os_uptime_s": read_trim("/proc/uptime").and_then(|u| u.split_whitespace().next()?.parse::<f64>().ok()).map(|s| s as u64),
        "load": load,
        "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        "mem_total_kb": kb("MemTotal:"),
        "mem_available_kb": kb("MemAvailable:"),
    })
}

// ── Difficulty ──────────────────────────────────────────────────────────────

/// A hash or target as attempts: `2^256 / value`, how many it takes on
/// average to find one this low.
fn difficulty(value: &[u8; 32]) -> f64 {
    let v = value.iter().fold(0f64, |acc, b| acc * 256.0 + *b as f64);
    2f64.powi(256) / v.max(1.0)
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    hex::decode_to_slice(s, &mut out).ok()?;
    Some(out)
}

// ── Stats ───────────────────────────────────────────────────────────────────

async fn stats(State(d): State<Dash>) -> Json<serde_json::Value> {
    let total = d.hash_counter.load(Ordering::Relaxed);
    let s = d.stats.read().unwrap_or_else(|e| e.into_inner()).clone();
    let now = unix_now();
    let current = Sample { t: now, hashes: total, temp_c: None, watts: None };
    let (m1, m10, h1) = {
        let h = d.history.read().unwrap_or_else(|e| e.into_inner());
        (rate_over(&h, current, 60), rate_over(&h, current, 600), rate_over(&h, current, 3600))
    };
    let throttled = super::handlers::throttled_flags().await;
    let power = pmic().await;
    let fan = fan();
    let cpu_threads = match d.settings.threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        n => n,
    };
    let ago = |unix: u64| (unix > 0).then(|| now.saturating_sub(unix));
    let best_ever = d.best_ever.read().unwrap_or_else(|e| e.into_inner()).clone();
    let known = |t: &[u8; 32]| (*t != [0u8; 32]).then(|| difficulty(t));
    let mut miner = serde_json::to_value(&*d.settings).unwrap_or_default();
    let extra = serde_json::json!({
        "connected": s.connected,
        "accepted": s.accepted_shares,
        "rejected": s.rejected_shares,
        "found": s.shares_found,
        "job_height": s.job_height,
        "jobs": s.jobs,
        "last_share_s": ago(s.last_share_unix),
        "last_job_s": ago(s.last_job_unix),
        "connected_s": ago(s.connected_since),
        "reconnects": s.reconnects,
        "latency_ms": s.latency_ms,
        "cpu_threads": cpu_threads,
        "share_difficulty": known(&s.share_target),
        "network_difficulty": known(&s.network_target),
        "best_difficulty": s.best_share.as_ref().map(difficulty),
        "best_ever_difficulty": best_ever.as_ref().and_then(|b| hex32(&b.hash)).map(|h| difficulty(&h)),
        "best_ever_unix": best_ever.as_ref().map(|b| b.unix),
        "pool": pool_view(&d).await,
    });
    if let (Some(m), Some(e)) = (miner.as_object_mut(), extra.as_object()) {
        m.extend(e.clone());
    }
    Json(serde_json::json!({
        // Tells a page it is a hasher's, not a node's.
        "mode": "miner",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_s": d.started.elapsed().as_secs(),
        "total_nonces": total,
        "hashrate": { "m1": m1, "m10": m10, "h1": h1 },
        "pow": crate::core::types::pow_algo_at(s.job_height).name(),
        "backends": crate::core::cuda_walk::backend_status(total),
        "is_axe_hardware": super::handlers::is_axe_device(),
        "temp_c": temp_c().unwrap_or(0.0),
        "freq_mhz": read_trim("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .and_then(|k| k.parse::<u64>().ok())
            .map(|khz| khz / 1000),
        "max_freq_mhz": read_trim("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            .and_then(|k| k.parse::<u64>().ok())
            .map(|khz| khz / 1000),
        "throttled": throttled,
        "throttle": throttled.map(super::axe::throttle_words).unwrap_or_default(),
        "power": power,
        "fan_rpm": fan.and_then(|f| f.0),
        "fan_pct": fan.and_then(|f| f.1),
        "system": system(),
        "miner": miner,
    }))
}

/// What the pool says about this miner, asked for at most every thirty
/// seconds: its part of the round, the pool's fee, what a block pays, and
/// merged mining.
async fn pool_view(d: &Dash) -> Option<serde_json::Value> {
    let mut cache = d.pool.lock().await;
    if cache.0.is_some_and(|at| at.elapsed() < POOL_STATS_EVERY) {
        return cache.1.clone();
    }
    let url = format!("http://{}/pool/stats", d.pool_api);
    let me = crate::core::types::parse_address_flexible(&d.settings.payout_address).ok();
    let fresh = async {
        let v: serde_json::Value = d.http.get(&url).send().await.ok()?.json().await.ok()?;
        let mine = v["miners"].as_array().into_iter().flatten().find(|m| {
            m["address"].as_str().and_then(|a| crate::core::types::parse_address_flexible(a).ok()) == me
        });
        let mw = &v["midwimble"];
        let merged = if mw.is_null() {
            serde_json::json!({ "active": false, "note": "this pool does not merge-mine midwimble" })
        } else {
            serde_json::json!({ "active": mw["active"], "note": mw["note"], "height": mw["height"] })
        };
        Some(serde_json::json!({
            "fee_percent": v["pool_fee_percent"],
            "miners": v["active_miners"],
            "total_score": v["total_score"],
            "your_score": mine.map_or(serde_json::json!(0), |m| m["score"].clone()),
            "block_reward": v["block_reward"],
            "recent_blocks": v["recent_blocks"].as_array().map_or(0, Vec::len),
            "network_hashrate": v["network_hashrate"],
            "merged": merged,
        }))
    }
    .await;
    *cache = (Some(Instant::now()), fresh);
    cache.1.clone()
}

#[derive(Deserialize)]
struct LogQuery {
    after: Option<u64>,
}

/// The miner's log lines after `after` (`super::log_ring`).
async fn logs(Query(q): Query<LogQuery>) -> Json<Vec<super::log_ring::LogLine>> {
    Json(super::log_ring::after(q.after.unwrap_or(0), super::log_ring::KEEP))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn dash(loopback_only: bool) -> Dash {
        Dash {
            settings: Arc::new(MinerSettings {
                pool_url: "127.0.0.1:1".into(),
                payout_address: "payout".into(),
                worker: "rig".into(),
                midwimble_address: None,
                threads: 2,
                backend: "auto".into(),
            }),
            hash_counter: Arc::new(AtomicU64::new(1234)),
            stats: Arc::new(RwLock::new(StratumStats {
                accepted_shares: 5,
                rejected_shares: 1,
                connected: true,
                job_height: 371_600,
                best_share: Some([0x00, 0x00, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                ..Default::default()
            })),
            started: Instant::now(),
            pool_api: "127.0.0.1:1".into(),
            http: reqwest::Client::new(),
            pool: Arc::new(tokio::sync::Mutex::new((None, None))),
            history: Arc::new(RwLock::new(VecDeque::new())),
            best_ever: Arc::new(RwLock::new(None)),
            loopback_only,
        }
    }

    /// One request over a real socket, with the Host header given.
    async fn call(d: Dash, method: &str, path: &str, host: &str) -> (StatusCode, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(d)).await.unwrap() });
        let mut conn = tokio::net::TcpStream::connect(addr).await.unwrap();
        let req = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        conn.write_all(req.as_bytes()).await.unwrap();
        let mut raw = Vec::new();
        conn.read_to_end(&mut raw).await.unwrap();
        let text = String::from_utf8_lossy(&raw).into_owned();
        let code: u16 = text.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
        (StatusCode::from_u16(code).unwrap(), body)
    }

    fn json_of(body: &str) -> serde_json::Value {
        let (open, close) = if body.trim_start().starts_with('[') || body.contains("\r\n[") { ('[', ']') } else { ('{', '}') };
        serde_json::from_str(&body[body.find(open).unwrap()..=body.rfind(close).unwrap()]).unwrap()
    }

    #[tokio::test]
    async fn it_serves_the_page_and_the_hashers_stats() {
        let (s, body) = call(dash(true), "GET", "/axe", "127.0.0.1:8546").await;
        assert_eq!(s, StatusCode::OK);
        assert!(body.contains("pollStats"));
        let (s, css) = call(dash(true), "GET", "/midstate.css", "localhost").await;
        assert_eq!(s, StatusCode::OK);
        assert!(!css.is_empty(), "the page's stylesheet");
        let (s, body) = call(dash(true), "GET", "/axe/stats", "localhost:8546").await;
        assert_eq!(s, StatusCode::OK);
        let v = json_of(&body);
        assert_eq!(v["mode"], "miner");
        assert_eq!(v["total_nonces"], 1234);
        assert_eq!(v["pow"], "walk");
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["miner"]["accepted"], 5);
        assert_eq!(v["miner"]["connected"], true);
        assert_eq!(v["miner"]["worker"], "rig");
        assert_eq!(v["miner"]["cpu_threads"], 2);
        assert_eq!(v["miner"]["best_difficulty"], 2f64.powi(20), "0x000010… is 2^236, so 2^20 attempts");
        assert!(v["miner"]["network_difficulty"].is_null(), "no job yet");
        assert!(v["miner"]["pool"].is_null(), "no pool API to ask");
        let (s, body) = call(dash(true), "GET", "/axe/history?minutes=60", "localhost").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(json_of(&body)["minutes"], 60);
        let (s, _) = call(dash(true), "GET", "/axe/logs?after=0", "localhost").await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn it_changes_nothing_and_answers_only_this_machine() {
        for (method, path) in [("POST", "/axe/config"), ("POST", "/axe/overclock"), ("POST", "/axe/wifi"), ("POST", "/axe/bench")] {
            let (s, _) = call(dash(true), method, path, "127.0.0.1").await;
            assert!(s == StatusCode::METHOD_NOT_ALLOWED || s == StatusCode::NOT_FOUND, "{method} {path}: {s}");
        }
        for path in ["/state", "/peers", "/mempool", "/metrics"] {
            let (s, _) = call(dash(true), "GET", path, "127.0.0.1").await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{path} is the node's");
        }
        let (s, _) = call(dash(true), "GET", "/axe/stats", "evil.example:8546").await;
        assert_eq!(s, StatusCode::FORBIDDEN, "a rebinding name is refused");
        let (s, _) = call(dash(true), "GET", "/axe/stats", "[::1]:8546").await;
        assert_eq!(s, StatusCode::OK);
        let (s, _) = call(dash(false), "GET", "/axe/stats", "axe.local:8546").await;
        assert_eq!(s, StatusCode::OK, "on the network, when asked for");
    }

    #[test]
    fn the_pool_api_is_where_the_stratum_client_looks() {
        assert_eq!(pool_api("stratum+tcp://pool.example:3333"), "pool.example:8081");
        assert_eq!(pool_api("1.2.3.4:3334"), "1.2.3.4:8082");
    }

    /// `vcgencmd pmic_read_adc` on a Raspberry Pi 5, trimmed.
    const PMIC: &str = "\
     3V7_WL_SW_A current(0)=0.00390372A
       3V3_SYS_A current(1)=0.05660400A
      VDD_CORE_A current(7)=2.00000000A
         EXT5V_V volt(24)=5.13534000V
      3V7_WL_SW_V volt(8)=3.71926000V
       3V3_SYS_V volt(9)=3.31054000V
      VDD_CORE_V volt(15)=0.90000000V
";

    #[test]
    fn a_pi_5s_power_is_its_rails_added_up() {
        let p = parse_pmic(PMIC).unwrap();
        let expected = 0.00390372 * 3.71926 + 0.056604 * 3.31054 + 2.0 * 0.9;
        assert!((p.watts - expected as f32).abs() < 1e-4, "{p:?}");
        assert_eq!(p.core_v, Some(0.9));
        assert_eq!(p.input_v, Some(5.13534));
        assert_eq!(parse_pmic("error=1 error_msg=\"Command not registered\""), None);
        assert_eq!(parse_pmic(""), None);
    }

    #[test]
    fn difficulty_is_the_attempts_a_hash_that_low_takes() {
        let mut h = [0u8; 32];
        h[0] = 0x80;
        assert_eq!(difficulty(&h), 2.0);
        h[0] = 0;
        h[2] = 0x01;
        assert_eq!(difficulty(&h), 2f64.powi(24), "0x000001 then 29 zero bytes is 2^232");
        assert!(difficulty(&[0u8; 32]).is_finite());
    }

    fn sample(t: u64, hashes: u64) -> Sample {
        Sample { t, hashes, temp_c: Some(60.0), watts: Some(8.0) }
    }

    #[test]
    fn rates_and_charts_come_from_the_history() {
        let h: VecDeque<Sample> = (0..=120).map(|i| sample(1000 + i * 5, i * 500)).collect();
        let now = sample(1600, 60_000);
        assert_eq!(rate_over(&h, now, 60), Some(100.0), "500 attempts every 5 seconds");
        assert_eq!(rate_over(&h, now, 600), Some(100.0));
        assert_eq!(rate_over(&h, now, 3600), None, "ten minutes of history is not an hour");
        let samples: Vec<Sample> = h.iter().copied().collect();
        let points = chart_points(&samples, 10);
        assert!(points.len() <= 10 && points.len() >= 9, "{}", points.len());
        assert_eq!(points[0][1], 100.0);
        assert_eq!(points[0][2], 60.0);
        assert!(chart_points(&samples[..1], 10).is_empty());
    }
}
