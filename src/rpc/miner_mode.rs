//! The Axe dashboard for `midstate miner`: the node's own page
//! (`miner_dashboard.html`), served by the hasher itself, with no node and no
//! chain behind it.
//!
//! # What it shows
//!
//! The hasher's own numbers: its hashrate (all of it, and each backend's: CPU
//! threads and CUDA GPUs), the pool connection and its shares, the job it is
//! on, its uptime, and the board's temperature, clock and throttling. What
//! only a node knows (peers, the chain's depth, the mempool, blocks this
//! machine won) shows as unavailable: no node is started to fill it in. Where
//! the pool merge-mines midwimble, its `/pool/stats` says whether merged
//! mining is running or paused, and why.
//!
//! # Cost
//!
//! The hashing loop never sees the dashboard: it bumps atomic counters it
//! already had. A request reads those, the stratum client's counters (a
//! read lock the client takes only on pool messages), the board's sensors
//! (files, and `vcgencmd` at most every ten seconds), and the pool's stats
//! (at most every thirty seconds, and only while someone is looking).
//!
//! # Access
//!
//! It listens on 127.0.0.1 unless told otherwise (`--dashboard`), and then
//! answers only requests addressed to localhost, so a web page cannot reach it
//! by pointing a name at 127.0.0.1. It changes nothing: every route is a GET,
//! and the node's settings routes (Wi-Fi, clocks, mining mode, the bench) are
//! not served at all, so there is nothing on it to protect.

use crate::mining::StratumStats;
use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Where `midstate miner` serves its dashboard unless told otherwise.
pub const DEFAULT_BIND: &str = "127.0.0.1:8546";
/// How often the pool's stats are asked for, at most.
const POOL_STATS_EVERY: Duration = Duration::from_secs(30);

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

#[derive(Clone)]
struct Dash {
    settings: Arc<MinerSettings>,
    hash_counter: Arc<AtomicU64>,
    stats: Arc<RwLock<StratumStats>>,
    started: Instant,
    /// The pool's HTTP API (`host:8081` for a stratum port of 3333).
    pool_api: String,
    http: reqwest::Client,
    /// The pool's merged-mining status, and when it was asked.
    merged: Arc<tokio::sync::Mutex<(Option<Instant>, Option<serde_json::Value>)>>,
    loopback_only: bool,
}

#[derive(Serialize)]
struct MinerStats {
    /// Tells the page it is a hasher's, not a node's.
    mode: &'static str,
    temp_c: f32,
    /// Since the miner started.
    uptime_s: u64,
    total_nonces: u64,
    is_axe_hardware: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    freq_mhz: Option<u32>,
    pow: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    throttled: Option<u32>,
    throttle: Vec<String>,
    backends: Vec<crate::core::cuda_walk::BackendStatus>,
    miner: MinerSection,
}

#[derive(Serialize)]
struct MinerSection {
    #[serde(flatten)]
    settings: MinerSettings,
    connected: bool,
    accepted: u64,
    rejected: u64,
    job_height: u64,
    jobs: u64,
    /// Seconds since the last accepted share, if any.
    last_share_s: Option<u64>,
    /// What this machine's threads come to: the ones asked for, or its cores.
    cpu_threads: usize,
    /// The pool's merged-mining status (`active`, `note`, `height`), when it
    /// merge-mines midwimble and its API answers.
    merged: Option<serde_json::Value>,
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
    let dash = Dash {
        pool_api: pool_api(&settings.pool_url),
        settings: Arc::new(settings),
        hash_counter,
        stats,
        started: Instant::now(),
        http: reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?,
        merged: Arc::new(tokio::sync::Mutex::new((None, None))),
        loopback_only: bind.ip().is_loopback(),
    };
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
    Html(include_str!("miner_dashboard.html"))
}

/// The mining settings, read-only, in the shape the page's settings panel
/// reads. No board: clocks are the node's to set.
async fn config(State(d): State<Dash>) -> Response {
    let s = &d.settings;
    Json(serde_json::json!({
        "mode": "pool",
        "pool_url": s.pool_url,
        "payout_address": s.payout_address,
        "worker": s.worker,
        "midwimble_address": s.midwimble_address,
        "read_only": true,
    }))
    .into_response()
}

async fn stats(State(d): State<Dash>) -> Json<MinerStats> {
    let total = d.hash_counter.load(Ordering::Relaxed);
    let (connected, accepted, rejected, job_height, jobs, last_share) = {
        let s = d.stats.read().unwrap_or_else(|e| e.into_inner());
        (s.connected, s.accepted_shares, s.rejected_shares, s.job_height, s.jobs, s.last_share_unix)
    };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |t| t.as_secs());
    let throttled = super::handlers::throttled_flags().await;
    let cpu_threads = match d.settings.threads {
        0 => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        n => n,
    };
    Json(MinerStats {
        mode: "miner",
        temp_c: std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map_or(0.0, |m| m / 1000.0),
        uptime_s: d.started.elapsed().as_secs(),
        total_nonces: total,
        is_axe_hardware: super::handlers::is_axe_device(),
        freq_mhz: std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|khz| (khz / 1000) as u32),
        pow: crate::core::types::pow_algo_at(job_height).name().to_string(),
        throttled,
        throttle: throttled
            .map(super::axe::throttle_words)
            .unwrap_or_default()
            .into_iter()
            .map(str::to_string)
            .collect(),
        backends: crate::core::cuda_walk::backend_status(total),
        miner: MinerSection {
            settings: (*d.settings).clone(),
            connected,
            accepted,
            rejected,
            job_height,
            jobs,
            last_share_s: (last_share > 0).then(|| now.saturating_sub(last_share)),
            cpu_threads,
            merged: merged_status(&d).await,
        },
    })
}

/// The pool's merged-mining status, asked for at most every thirty seconds.
async fn merged_status(d: &Dash) -> Option<serde_json::Value> {
    let mut cache = d.merged.lock().await;
    if cache.0.is_some_and(|at| at.elapsed() < POOL_STATS_EVERY) {
        return cache.1.clone();
    }
    let url = format!("http://{}/pool/stats", d.pool_api);
    let fresh = async {
        let v: serde_json::Value = d.http.get(&url).send().await.ok()?.json().await.ok()?;
        let mw = v.get("midwimble")?;
        if mw.is_null() {
            return Some(serde_json::json!({ "active": false, "note": "this pool does not merge-mine midwimble" }));
        }
        Some(serde_json::json!({
            "active": mw.get("active").cloned().unwrap_or_default(),
            "note": mw.get("note").cloned().unwrap_or_default(),
            "height": mw.get("height").cloned().unwrap_or_default(),
        }))
    }
    .await;
    *cache = (Some(Instant::now()), fresh);
    cache.1.clone()
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
            stats: Arc::new(RwLock::new(StratumStats { accepted_shares: 5, rejected_shares: 1, connected: true, job_height: 371_600, ..Default::default() })),
            started: Instant::now(),
            pool_api: "127.0.0.1:1".into(),
            http: reqwest::Client::new(),
            merged: Arc::new(tokio::sync::Mutex::new((None, None))),
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
        let json = &body[body.find('{').unwrap()..=body.rfind('}').unwrap()];
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["mode"], "miner");
        assert_eq!(v["total_nonces"], 1234);
        assert_eq!(v["pow"], "walk");
        assert_eq!(v["miner"]["accepted"], 5);
        assert_eq!(v["miner"]["connected"], true);
        assert_eq!(v["miner"]["worker"], "rig");
        assert_eq!(v["miner"]["cpu_threads"], 2);
        assert!(v["miner"]["merged"].is_null(), "no pool API to ask");
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
}
