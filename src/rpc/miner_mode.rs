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
//! # Settings
//!
//! The Settings tab changes what the miner mines with, while it runs: the
//! pool and a fallback pool, who is paid, the threads and a heat limit
//! (`rpc::miner_settings`, kept in `miner-settings.toml`). New pool settings
//! reconnect the stratum client at once; a thread count restarts the job's
//! threads. The Pi's clock and fan curve, and a reboot, go through a small
//! root-owned helper set up once with `sudo midstate board-setup`
//! (`rpc::axe_board`), and take effect at the next boot. "Test this unit"
//! pauses mining for five seconds to measure the board alone.
//!
//! # Access
//!
//! It listens on 127.0.0.1 unless told otherwise (`--dashboard`), and then
//! answers only requests addressed to localhost, so a web page cannot reach it
//! by pointing a name at 127.0.0.1. On the network, anyone may look; a change
//! is taken only ([`may_change`]) from this machine or the local network,
//! addressed by an IP address or a local name (never a public name, which a
//! page could point at the Pi: DNS rebinding), sent as JSON (which a page on
//! another site cannot send without asking first) and not by another site's
//! page (its `Origin`). A node's routes (Wi-Fi among them) are not served.
use super::axe_board::{self, Change, Fan};
use super::miner_settings::MinerSettings;
use crate::mining::{MinerControl, StratumStats};
use axum::extract::{ConnectInfo, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

pub use crate::mining::pool_api;

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
    /// What the command line said, for "Use the command line's".
    command_line: Arc<MinerSettings>,
    /// "auto", "gpu" or "cpu", from the command line.
    backend: String,
    control: Arc<MinerControl>,
    /// Where saved settings are kept.
    settings_path: PathBuf,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<StratumStats>>,
    started: Instant,
    http: reqwest::Client,
    /// The pool's view of this miner: when it was asked, of which API, and
    /// what it said.
    pool: Arc<tokio::sync::Mutex<(Option<Instant>, String, Option<serde_json::Value>)>>,
    history: Arc<RwLock<VecDeque<Sample>>>,
    best_ever: Arc<RwLock<Option<Best>>>,
    loopback_only: bool,
    /// A unit test is running.
    testing: Arc<AtomicBool>,
}
/// Serves the dashboard on `bind` until the process ends. `command_line`:
/// the settings the command line gave; `control`: the miner's live ones.
pub async fn serve(
    bind: SocketAddr,
    command_line: MinerSettings,
    backend: String,
    control: Arc<MinerControl>,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<StratumStats>>,
) -> anyhow::Result<()> {
    let best = std::fs::read(BEST_FILE).ok().and_then(|b| serde_json::from_slice::<Best>(&b).ok());
    let dash = Dash {
        command_line: Arc::new(command_line),
        backend,
        control,
        settings_path: PathBuf::from(super::miner_settings::SETTINGS_FILE),
        hash_counter,
        stats,
        started: Instant::now(),
        http: reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?,
        pool: Arc::new(tokio::sync::Mutex::new((None, String::new(), None))),
        history: Arc::new(RwLock::new(VecDeque::new())),
        best_ever: Arc::new(RwLock::new(best)),
        loopback_only: bind.ip().is_loopback(),
        testing: Arc::new(AtomicBool::new(false)),
    };
    tokio::spawn(sampler(dash.clone()));
    let app = router(dash);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    if bind.ip().is_loopback() {
        tracing::info!("miner dashboard: http://{bind}/axe (this machine only; --dashboard <ip:port> to change)");
    } else {
        tracing::warn!("miner dashboard: http://{bind}/axe, reachable from your network, which can change its settings.");
    }
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

fn router(dash: Dash) -> Router {
    let loopback_only = dash.loopback_only;
    let changes = Router::new()
        .route("/axe/settings", post(save_settings))
        .route("/axe/settings/reset", post(reset_settings))
        .route("/axe/board", post(board))
        .route("/axe/reboot", post(reboot))
        .route("/axe/bench", post(bench))
        .route_layer(middleware::from_fn(|req: Request, next: Next| async move {
            let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0.ip());
            match peer.map(|p| may_change(p, req.headers())) {
                Some(Ok(())) => next.run(req).await,
                Some(Err(why)) => refuse(StatusCode::FORBIDDEN, why),
                None => refuse(StatusCode::FORBIDDEN, "Who sent this is not known.".into()),
            }
        }));
    Router::new()
        .route("/", get(page))
        .route("/axe", get(page))
        .route("/axe/stats", get(stats))
        .route("/axe/history", get(history))
        .route("/axe/logs", get(logs))
        .route("/axe/config", get(config))
        .route("/midstate.css", get(super::handlers::midstate_css))
        .merge(changes)
        .with_state(dash)
        .layer(middleware::from_fn(move |req: Request, next: Next| async move {
            if loopback_only && !addressed_to_localhost(&req) {
                return StatusCode::FORBIDDEN.into_response();
            }
            next.run(req).await
        }))
}

fn refuse(status: StatusCode, error: String) -> Response {
    (status, Json(serde_json::json!({ "ok": false, "error": error }))).into_response()
}

/// Whether `ip` is this machine or on the local network.
fn peer_is_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => peer_is_local(IpAddr::V4(v4)),
            // Loopback, unique local (fc00::/7) and link-local (fe80::/10).
            None => v6.is_loopback() || (v6.segments()[0] & 0xfe00) == 0xfc00 || (v6.segments()[0] & 0xffc0) == 0xfe80,
        },
    }
}

/// The name in a Host header, without its port.
fn host_name(host: &str) -> &str {
    match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.rsplit_once(':').map_or(host, |(n, _)| n),
    }
}

/// Whether `host` (a Host header) names this machine the way only the local
/// network can: an IP address, `localhost`, a bare host name, or a name under
/// a suffix no public DNS serves. A page on the web can point a name of its
/// own at the Pi's address (DNS rebinding), but not one of these.
fn host_is_local_name(host: &str) -> bool {
    let name = host_name(host).trim_end_matches('.').to_ascii_lowercase();
    if name.parse::<IpAddr>().is_ok() || name == "localhost" {
        return true;
    }
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.') {
        return false;
    }
    !name.contains('.')
        || [".local", ".lan", ".home", ".home.arpa", ".internal", ".localdomain"].iter().any(|s| name.ends_with(s))
}

/// Whether a request from `peer` with `headers` may change the miner's
/// settings, and if not, why (the text to show).
pub(crate) fn may_change(peer: IpAddr, headers: &HeaderMap) -> Result<(), String> {
    if !peer_is_local(peer) {
        return Err("Settings can be changed only from this machine or your local network.".into());
    }
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    if !host_is_local_name(host) {
        return Err(format!(
            "Settings can't be changed through the name {}. Open the dashboard by the Pi's IP address or its .local name.",
            host_name(host)
        ));
    }
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|o| o.to_str().ok()) {
        if origin.split("://").nth(1).unwrap_or(origin) != host {
            return Err("Refused: this came from another site's page.".into());
        }
    }
    if headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) == Some("cross-site") {
        return Err("Refused: this came from another site's page.".into());
    }
    Ok(())
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

/// The settings, the command line's beside them, and the board's: what the
/// Settings tab fills its forms with. `can_change` says whether this browser
/// may change them, and `why_not` why when it may not.
async fn config(State(d): State<Dash>, ConnectInfo(peer): ConnectInfo<SocketAddr>, headers: HeaderMap) -> Json<serde_json::Value> {
    let s = MinerSettings::current(&d.control);
    let model = axe_board::board_model();
    let profile = model.as_deref().and_then(super::axe::profile);
    let boot = axe_board::boot_config_path().and_then(|p| std::fs::read_to_string(p).ok());
    let (mhz, voltage) = boot
        .as_deref()
        .map(|c| super::axe::configured(c, profile.as_ref()))
        .unwrap_or((None, None));
    let pi5 = profile.as_ref().is_some_and(|p| p.family == super::axe::Family::Pi5);
    let fan = boot.as_deref().map(axe_board::configured_fan).filter(|_| pi5);
    let helper = match profile {
        Some(_) => Some(axe_board::helper_status().await),
        None => None,
    };
    let change = may_change(peer.ip(), &headers);
    Json(serde_json::json!({
        "mode": "pool",
        "pool_url": s.pool_url,
        "payout_address": s.payout_address,
        "worker": s.worker,
        "midwimble_address": s.midwimble_address,
        "threads": s.threads,
        "backend": d.backend,
        "settings": s,
        "command_line": *d.command_line,
        "saved": d.settings_path.exists(),
        "cores": crate::mining::cores(),
        "heat_limit_range": [super::miner_settings::HEAT_LIMIT_RANGE.start(), super::miner_settings::HEAT_LIMIT_RANGE.end()],
        "can_change": change.is_ok(),
        "why_not": change.err(),
        "board": {
            "model": model,
            "clock": profile,
            "configured_mhz": mhz,
            "configured_voltage": voltage,
            "fan": fan,
            "fan_text": fan.map(|f| f.describe()),
            "fan_min_percent": axe_board::FAN_MIN_PERCENT,
            "helper": helper,
            "setup_command": axe_board::setup_command(),
        },
    }))
}

/// POST /axe/settings: checks, keeps and applies new settings.
async fn save_settings(State(d): State<Dash>, Json(next): Json<MinerSettings>) -> Response {
    let next = match next.check() {
        Ok(s) => s,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = next.save(&d.settings_path) {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not keep the settings: {e}"));
    }
    let before = MinerSettings::current(&d.control);
    next.apply(&d.control);
    tracing::info!("settings saved from the dashboard ({})", d.settings_path.display());
    let message = if before.pool() != next.pool() {
        "Saved. Reconnecting to the pool with them."
    } else if before == next {
        "Saved. Nothing changed."
    } else {
        "Saved and in use."
    };
    Json(serde_json::json!({ "ok": true, "message": message, "settings": next })).into_response()
}

/// POST /axe/settings/reset: back to the command line's settings.
async fn reset_settings(State(d): State<Dash>, Json(_): Json<serde_json::Value>) -> Response {
    if let Err(e) = MinerSettings::forget(&d.settings_path) {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    d.command_line.apply(&d.control);
    tracing::info!("back to the command line's settings");
    Json(serde_json::json!({ "ok": true, "message": "Using the command line's settings again.", "settings": *d.command_line })).into_response()
}

#[derive(Deserialize)]
struct ClockBody {
    mhz: u32,
    /// In the board's unit (microvolts on a Pi 5).
    voltage: u32,
}

#[derive(Deserialize)]
struct BoardBody {
    clock: Option<ClockBody>,
    fan: Option<Fan>,
    #[serde(default)]
    reboot: bool,
}

/// POST /axe/board: the clock and fan curve, written to the boot config by
/// the helper, and a reboot to bring them in if asked.
async fn board(Json(body): Json<BoardBody>) -> Response {
    let change = Change { clock: body.clock.map(|c| (c.mhz, c.voltage)), fan: body.fan };
    if change == Change::default() {
        return refuse(StatusCode::BAD_REQUEST, "Nothing to set: give a clock, a fan curve or both.".into());
    }
    // Checked here for a quick answer; the helper checks again, as root.
    if let (Some(model), Some(boot)) = (axe_board::board_model(), axe_board::boot_config_path().and_then(|p| std::fs::read_to_string(p).ok())) {
        if let Err(e) = change.rewrite(&model, &boot) {
            return refuse(StatusCode::BAD_REQUEST, e);
        }
    }
    let message = match axe_board::helper_apply(change).await {
        Ok(m) => m,
        Err(e) => return refuse(StatusCode::CONFLICT, e),
    };
    tracing::info!(
        "board settings written{}{}",
        change.clock.map(|(m, v)| format!(": {m} MHz, +{v}")).unwrap_or_default(),
        change.fan.map(|f| format!(", fan {}", f.describe())).unwrap_or_default()
    );
    if body.reboot {
        schedule_reboot();
        return Json(serde_json::json!({ "ok": true, "rebooting": true, "message": format!("{message} Restarting now: the dashboard is back in about a minute.") })).into_response();
    }
    Json(serde_json::json!({ "ok": true, "rebooting": false, "message": message })).into_response()
}

/// Reboots through the helper in two seconds, once the answer has gone.
fn schedule_reboot() {
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_secs(2)).await;
        tracing::warn!("restarting the Pi, as asked from the dashboard");
        if let Err(e) = axe_board::helper_reboot().await {
            tracing::error!("could not restart the Pi: {e}");
        }
    });
}

/// POST /axe/reboot.
async fn reboot(Json(_): Json<serde_json::Value>) -> Response {
    let status = axe_board::helper_status().await;
    if !status.ready {
        return refuse(StatusCode::CONFLICT, status.problem.unwrap_or_else(|| "The board helper is not set up.".into()));
    }
    schedule_reboot();
    Json(serde_json::json!({ "ok": true, "message": "Restarting now: the dashboard is back in about a minute." })).into_response()
}

/// POST /axe/bench, "Test this unit": mining pauses, then every core checks
/// it computes Midstate Walk's known answers, counts attempts for five
/// seconds, and checks them again while hot. After an overclock, the way to
/// find out whether it holds: a core that gets one wrong loses shares.
async fn bench(State(d): State<Dash>, Json(_): Json<serde_json::Value>) -> Response {
    if d.testing.swap(true, Ordering::SeqCst) {
        return refuse(StatusCode::CONFLICT, "A test is already running.".into());
    }
    d.control.set_paused(true);
    // The hashing threads stop at their next check.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let threads = crate::mining::cores();
    let result = tokio::task::spawn_blocking(move || crate::core::walk::bench(threads, Duration::from_secs(5))).await;
    d.control.set_paused(false);
    d.testing.store(false, Ordering::SeqCst);
    let Ok((rate, correct)) = result else {
        return refuse(StatusCode::INTERNAL_SERVER_ERROR, "The test could not run.".into());
    };
    let message = if correct {
        format!("Every core computed the known answers: {rate:.0} attempts a second on {threads} threads.")
    } else {
        "A core computed a wrong answer: this clock is unstable. Lower it or add voltage, then test again.".to_string()
    };
    tracing::info!("unit test: {message}");
    Json(serde_json::json!({ "ok": true, "threads": threads, "attempts_per_second": rate, "correct": correct, "message": message })).into_response()
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

/// The CPU's temperature, °C.
pub fn temp_c() -> Option<f32> {
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
    let threads = d.control.threads();
    let settings = MinerSettings::current(&d.control);
    let ago = |unix: u64| (unix > 0).then(|| now.saturating_sub(unix));
    let best_ever = d.best_ever.read().unwrap_or_else(|e| e.into_inner()).clone();
    let known = |t: &[u8; 32]| (*t != [0u8; 32]).then(|| difficulty(t));
    let pool_in_use = if s.pool_url.is_empty() { settings.pool_url.clone() } else { s.pool_url.clone() };
    let mut miner = serde_json::to_value(&settings).unwrap_or_default();
    let extra = serde_json::json!({
        "backend": d.backend,
        "pool_in_use": pool_in_use,
        "on_fallback": s.on_fallback,
        "saved": d.settings_path.exists(),
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
        // Hashing now; fewer than asked while the heat limit holds some
        // back, none while paused (the unit test).
        "cpu_threads": threads.running(),
        "threads_wanted": threads.wanted(),
        "heat_holding_back": threads.heat_cap.is_some_and(|c| c < threads.wanted()),
        "paused": threads.paused,
        "share_difficulty": known(&s.share_target),
        "network_difficulty": known(&s.network_target),
        "best_difficulty": s.best_share.as_ref().map(difficulty),
        "best_ever_difficulty": best_ever.as_ref().and_then(|b| hex32(&b.hash)).map(|h| difficulty(&h)),
        "best_ever_unix": best_ever.as_ref().map(|b| b.unix),
        "pool": pool_view(&d, &pool_in_use, &settings.payout_address).await,
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

/// What the pool at `pool_url` says about `payout`, asked for at most every
/// thirty seconds: its part of the round, the pool's fee, what a block pays,
/// and merged mining.
async fn pool_view(d: &Dash, pool_url: &str, payout: &str) -> Option<serde_json::Value> {
    let api = pool_api(pool_url);
    let mut cache = d.pool.lock().await;
    if cache.1 == api && cache.0.is_some_and(|at| at.elapsed() < POOL_STATS_EVERY) {
        return cache.2.clone();
    }
    let url = format!("http://{api}/pool/stats");
    let me = crate::core::types::parse_address_flexible(payout).ok();
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
    *cache = (Some(Instant::now()), api, fresh);
    cache.2.clone()
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

    fn settings() -> MinerSettings {
        MinerSettings {
            pool_url: "127.0.0.1:1".into(),
            fallback_url: None,
            payout_address: "07".repeat(32),
            worker: "rig".into(),
            midwimble_address: None,
            threads: 2,
            heat_limit_c: None,
        }
    }

    fn dash(loopback_only: bool) -> Dash {
        let dir = std::env::temp_dir().join(format!("midstate-dash-{}-{}", std::process::id(), rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        Dash {
            command_line: Arc::new(settings()),
            backend: "auto".into(),
            control: Arc::new(MinerControl::new(settings().pool(), 2, None)),
            settings_path: dir.join(super::super::miner_settings::SETTINGS_FILE),
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
            http: reqwest::Client::new(),
            pool: Arc::new(tokio::sync::Mutex::new((None, String::new(), None))),
            history: Arc::new(RwLock::new(VecDeque::new())),
            best_ever: Arc::new(RwLock::new(None)),
            loopback_only,
            testing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// One request over a real socket, with the Host header and any other
    /// headers and body given.
    async fn send(d: Dash, method: &str, path: &str, host: &str, headers: &[(&str, &str)], body: &str) -> (StatusCode, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router(d).into_make_service_with_connect_info::<SocketAddr>()).await.unwrap()
        });
        let mut conn = tokio::net::TcpStream::connect(addr).await.unwrap();
        let extra: String = headers.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect();
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        conn.write_all(req.as_bytes()).await.unwrap();
        let mut raw = Vec::new();
        conn.read_to_end(&mut raw).await.unwrap();
        let text = String::from_utf8_lossy(&raw).into_owned();
        let code: u16 = text.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
        (StatusCode::from_u16(code).unwrap(), body)
    }

    async fn call(d: Dash, method: &str, path: &str, host: &str) -> (StatusCode, String) {
        send(d, method, path, host, &[], "").await
    }

    async fn post_json(d: Dash, path: &str, host: &str, body: &serde_json::Value) -> (StatusCode, serde_json::Value) {
        let (s, b) = send(d, "POST", path, host, &[("Content-Type", "application/json")], &body.to_string()).await;
        let v = if b.contains('{') { json_of(&b) } else { serde_json::Value::Null };
        (s, v)
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
        assert_eq!(v["miner"]["threads_wanted"], 2);
        assert_eq!(v["miner"]["pool_in_use"], "127.0.0.1:1");
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
    async fn it_answers_only_this_machine_unless_told_and_serves_no_node_routes() {
        for path in ["/state", "/peers", "/mempool", "/metrics", "/axe/wifi", "/axe/overclock"] {
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
    fn changes_come_only_from_the_local_network_by_a_local_name() {
        let headers = |pairs: &[(&'static str, &str)]| {
            let mut h = HeaderMap::new();
            for (k, v) in pairs {
                h.insert(*k, v.parse().unwrap());
            }
            h
        };
        let lan: IpAddr = "192.168.50.20".parse().unwrap();
        for host in ["192.168.50.17:8546", "midstate5axe:8546", "midstate5axe.local:8546", "localhost:8546", "[fe80::1]:8546", "axe.home.arpa"] {
            assert!(may_change(lan, &headers(&[("host", host)])).is_ok(), "{host}");
        }
        // A page on the web pointing its own name at the Pi.
        let rebound = may_change(lan, &headers(&[("host", "pi.attacker.example:8546")])).unwrap_err();
        assert!(rebound.contains("pi.attacker.example") && rebound.contains("IP address"), "{rebound}");
        assert!(may_change(lan, &headers(&[])).is_err(), "no Host");
        // Another site's page, by its Origin or the browser's own word.
        let h = headers(&[("host", "192.168.50.17:8546"), ("origin", "http://evil.example")]);
        assert!(may_change(lan, &h).unwrap_err().contains("another site"));
        let h = headers(&[("host", "192.168.50.17:8546"), ("origin", "http://192.168.50.17:8546")]);
        assert!(may_change(lan, &h).is_ok(), "the dashboard's own page");
        let h = headers(&[("host", "192.168.50.17:8546"), ("sec-fetch-site", "cross-site")]);
        assert!(may_change(lan, &h).is_err());
        // From outside the network (a forwarded port).
        let wan: IpAddr = "203.0.113.9".parse().unwrap();
        assert!(may_change(wan, &headers(&[("host", "192.168.50.17:8546")])).unwrap_err().contains("local network"));
        assert!(peer_is_local("::ffff:10.0.0.2".parse().unwrap()) && peer_is_local("fd12::1".parse().unwrap()));
        assert!(!peer_is_local("2001:db8::1".parse().unwrap()));
    }

    #[tokio::test]
    async fn settings_are_checked_kept_and_applied_while_it_runs() {
        let d = dash(false);
        let (control, path) = (d.control.clone(), d.settings_path.clone());

        // Only JSON: a form posted by another site's page is not taken.
        let (s, _) = send(d.clone(), "POST", "/axe/settings", "192.168.1.5", &[("Content-Type", "application/x-www-form-urlencoded")], "pool_url=x").await;
        assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);

        let mut next = settings();
        next.payout_address = "nope".into();
        let (s, v) = post_json(d.clone(), "/axe/settings", "192.168.1.5", &serde_json::to_value(&next).unwrap()).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["error"].as_str().unwrap().starts_with("Payout address"), "{v}");
        assert!(!path.exists(), "nothing kept");

        let mut next = settings();
        next.pool_url = "stratum+tcp://other.example:3334".into();
        next.fallback_url = Some("127.0.0.1:1".into());
        next.threads = 1;
        next.heat_limit_c = Some(75.0);
        let (s, v) = post_json(d.clone(), "/axe/settings", "192.168.1.5", &serde_json::to_value(&next).unwrap()).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["message"].as_str().unwrap().contains("Reconnecting"), "{v}");
        assert_eq!(MinerSettings::current(&control), next, "in use at once");
        assert_eq!(MinerSettings::load(&path).unwrap(), Some(next.clone()), "and kept");

        let (s, body) = call(d.clone(), "GET", "/axe/config", "192.168.1.5").await;
        assert_eq!(s, StatusCode::OK);
        let v = json_of(&body);
        assert_eq!(v["saved"], true);
        assert_eq!(v["settings"]["pool_url"], "stratum+tcp://other.example:3334");
        assert_eq!(v["command_line"]["pool_url"], "127.0.0.1:1");
        assert_eq!(v["can_change"], true);
        let (_, body) = call(d.clone(), "GET", "/axe/config", "pi.attacker.example").await;
        assert_eq!(json_of(&body)["can_change"], false);

        // Refused from a rebinding name.
        let (s, v) = post_json(d.clone(), "/axe/settings/reset", "pi.attacker.example", &serde_json::json!({})).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
        assert!(path.exists());

        let (s, v) = post_json(d.clone(), "/axe/settings/reset", "192.168.1.5", &serde_json::json!({})).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(!path.exists());
        assert_eq!(MinerSettings::current(&control), settings(), "the command line's again");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn board_changes_are_checked_and_need_the_helper() {
        let d = dash(true);
        let (s, v) = post_json(d.clone(), "/axe/board", "localhost", &serde_json::json!({})).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        let (s, v) = post_json(d.clone(), "/axe/board", "localhost", &serde_json::json!({ "fan": { "mode": "loud" } })).await;
        assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{v}");
        // Off a Pi with the helper set up, it says what is missing.
        if !std::path::Path::new(axe_board::HELPER_PATH).exists() {
            let (s, v) = post_json(d.clone(), "/axe/board", "localhost", &serde_json::json!({ "fan": { "mode": "cool" } })).await;
            assert!(s == StatusCode::CONFLICT || s == StatusCode::BAD_REQUEST, "{s} {v}");
            let (s, v) = post_json(d, "/axe/reboot", "localhost", &serde_json::json!({})).await;
            assert_eq!(s, StatusCode::CONFLICT, "{v}");
            assert!(v["error"].as_str().unwrap().contains("setup"), "{v}");
        }
    }

    #[tokio::test]
    async fn the_unit_test_pauses_mining_and_gives_it_back() {
        let d = dash(true);
        let control = d.control.clone();
        let (s, v) = post_json(d, "/axe/bench", "localhost", &serde_json::json!({})).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["correct"], true, "{v}");
        assert!(v["attempts_per_second"].as_f64().unwrap() > 0.0);
        assert!(!control.threads().paused, "mining resumes");
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
