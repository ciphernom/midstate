//! Local mining supervisor.
//!
//! # Architecture
//!
//! The desktop app does not mine in-process. It supervises child processes, each
//! the `midstate` binary invoked with a subcommand:
//!
//! ```text
//!   midstate pool   --bind-addr 127.0.0.1:PORT --rpc-port 8545 --fee 0 ...
//!   midstate miner  --pool-url stratum+tcp://HOST:PORT --payout-address ...
//! ```
//!
//! Pointed at a remote pool, only the hasher is spawned.
//!
//! # Reasoning
//!
//! Child processes rather than in-process tasks, for four reasons:
//!
//! 1. **Payout correctness.** `Node::new(.., mining_threads, ..)` pays coinbase
//!    outputs to addresses derived from a random `mining_seed` generated in the
//!    chain data directory — key material the wallet has never seen. Mining that
//!    way produces rewards the user cannot spend. Stratum authorisation carries
//!    the payout address (`mining.authorize` params[0]), so routing through a
//!    pool pays the wallet directly.
//! 2. **Working directory.** `run_stratum_pool` creates `data/pool_stratum.redb`
//!    relative to the process CWD. A GUI launched from Finder or a `.desktop`
//!    entry has an arbitrary CWD — often `/`. Spawning with an explicit
//!    `current_dir` fixes this without modifying the pool.
//! 3. **Failure isolation.** The pool's boot path panics on port conflicts and
//!    disk errors. In-process that takes the whole wallet down; as a child it
//!    is an exit status the UI can explain.
//! 4. **Clean shutdown.** Hashing threads are killable by PID. Cancelling
//!    in-process rayon work mid-flight is considerably harder.
//!
//! # Startup is asynchronous, deliberately
//!
//! Bringing the local pool up involves waiting for it to bind and answer, which
//! can take seconds. An earlier version awaited that inside walletd's command
//! loop, which froze the entire actor — no balance, no node info, no mining
//! status — for the whole startup window, so the UI could show nothing at all
//! while the thing it had just been asked to do was happening.
//!
//! `start` and `stop` therefore return immediately and drive the work on a
//! background task, publishing progress into a shared [`MiningState`] that
//! status reads can sample cheaply at any time.
//!
//! # Safety / Invariants
//!
//! - **No orphans.** Children are killed on `stop()`; the pool is always started
//!   before the miner and killed after it.
//! - **Never load-bearing.** Every failure path leaves mining off, records the
//!   reason in `phase`, and leaves the wallet fully usable.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock};

/// First stratum port tried. `run_stratum_pool` bumps this by one (together
/// with the audit port) when a pair is already bound, up to ten times.
const BASE_STRATUM_PORT: u16 = 3333;
/// First audit-API port. Kept in lockstep with the stratum port by the pool's
/// tandem-binding loop, which is what makes discovery-by-probe reliable.
const BASE_AUDIT_PORT: u16 = 8081;
/// Matches the pool's own retry ceiling.
const MAX_PORT_OFFSET: u16 = 10;

/// How long to wait for the local pool's audit API to answer before giving up.
const POOL_READY_TIMEOUT: Duration = Duration::from_secs(25);

/// Where the mining stack currently is.
///
/// # Reasoning
/// The UI needs to distinguish "nothing is happening" from "we asked and it is
/// working on it" from "it failed and here is why". A single `running: bool`
/// collapses all three into one, which is what made the first version feel
/// broken even when it was succeeding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiningPhase {
    Idle,
    Starting,
    Running,
    Stopping,
    Error(String),
}

impl MiningPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Stopping => "stopping",
            Self::Error(_) => "error",
        }
    }
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Starting | Self::Stopping)
    }
}

/// Cheaply-readable snapshot of the mining stack.
#[derive(Clone, Debug)]
pub struct MiningState {
    pub phase: MiningPhase,
    /// Human-readable progress detail, e.g. "waiting for the local pool to bind".
    pub message: Option<String>,
    pub payout_address: String,
    /// Effective stratum URL the hasher is pointed at.
    pub pool_url: String,
    /// True when this process is also running the pool.
    pub local_pool: bool,
    /// Hasher threads. 0 means "all cores", matching the CLI default.
    pub threads: usize,
    pub stratum_port: u16,
    pub audit_port: u16,
    pub started_at: Option<Instant>,
}

impl Default for MiningState {
    fn default() -> Self {
        Self {
            phase: MiningPhase::Idle,
            message: None,
            payout_address: String::new(),
            pool_url: String::new(),
            local_pool: true,
            threads: 0,
            stratum_port: 0,
            audit_port: 0,
            started_at: None,
        }
    }
}

/// Everything the local pool's audit API tells us about this wallet's mining.
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    /// Estimated from accepted shares x hashes_per_share / elapsed.
    pub hashrate: f64,
    /// Whole-network rate, published by the pool from the live block target.
    pub network_hashrate: f64,
    /// This wallet's estimated fraction of network hashrate, 0.0..=1.0.
    pub network_share: f64,
    pub hashes_per_share: f64,
    pub accepted: u64,
    pub rejected: u64,
    pub score: u64,
    /// Blocks in the pool's recent list credited to this wallet.
    pub blocks_found: u64,
    /// Blocks in that list overall, for context.
    pub pool_blocks: u64,
    pub active_miners: u64,
    pub total_score: u64,
    pub network_height: u64,
    pub block_reward: u64,
    /// Per-rig accepted-share tallies for this wallet's address.
    pub workers: Vec<(String, u64)>,
}

/// The child processes, held separately from the observable state so a long
/// startup never blocks a status read.
#[derive(Default)]
struct Children {
    pool: Option<Child>,
    miner: Option<Child>,
}

/// Supervises the pool and hasher child processes.
///
/// Cheap to clone: all interior state is shared.
#[derive(Clone)]
pub struct MiningSupervisor {
    children: Arc<Mutex<Children>>,
    state: Arc<RwLock<MiningState>>,
    work_dir: PathBuf,
    node_rpc_port: u16,
}

impl MiningSupervisor {
    pub fn new(work_dir: PathBuf, node_rpc_port: u16) -> Self {
        Self {
            children: Arc::new(Mutex::new(Children::default())),
            state: Arc::new(RwLock::new(MiningState::default())),
            work_dir,
            node_rpc_port,
        }
    }

    /// Samples the current state. Never blocks on startup work.
    pub async fn state(&self) -> MiningState {
        self.state.read().await.clone()
    }

    /// The default pool URL: this machine's own pool.
    pub fn default_pool_url() -> String {
        format!("stratum+tcp://127.0.0.1:{}", BASE_STRATUM_PORT)
    }

    /// True when `url` refers to a pool this process should launch itself.
    ///
    /// # Reasoning
    /// A blank field means "use mine". A loopback host means the user typed the
    /// local address explicitly, which should behave identically — otherwise the
    /// hasher would be pointed at a pool nobody started.
    pub fn is_local(url: &str) -> bool {
        let u = url.trim().to_ascii_lowercase();
        if u.is_empty() {
            return true;
        }
        let hostless = u
            .trim_start_matches("stratum+tcp://")
            .trim_start_matches("stratum://")
            .trim_start_matches("tcp://");
        hostless.starts_with("127.0.0.1")
            || hostless.starts_with("localhost")
            || hostless.starts_with("[::1]")
    }

    /// Locates the `midstate` binary.
    ///
    /// # Reasoning
    /// Looks next to the running executable first, which is where both a cargo
    /// build (`target/release/`) and a packaged install put them. Falls back to
    /// `PATH`. Deliberately does not search the CWD: for a GUI that is attacker-
    /// influenceable in a way the executable's own directory is not.
    pub fn find_binary() -> Result<PathBuf> {
        let exe_name = if cfg!(windows) { "midstate.exe" } else { "midstate" };

        if let Ok(cur) = std::env::current_exe() {
            if let Some(dir) = cur.parent() {
                let candidate = dir.join(exe_name);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        if let Some(paths) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&paths) {
                let candidate = dir.join(exe_name);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        bail!(
            "could not find the `{}` binary next to this application or on PATH — \
             mining needs it to run the hasher",
            exe_name
        )
    }

    /// Begins startup and returns immediately.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  payout parses as a Midstate address ∧ phase ∈ {Idle, Error}
    /// Post: phase' = Starting, and a background task drives:
    ///         Starting → Running          (success)
    ///         Starting → Error(reason)    (failure, children reaped)
    ///       returns without waiting for either
    /// ```
    ///
    /// # Safety / Invariants
    /// - **No partial state.** If the hasher fails to launch, any pool this call
    ///   started is killed before `Error` is published.
    pub async fn start(&self, payout_address: String, threads: usize, pool_url: String) {
        {
            let st = self.state.read().await;
            if st.phase == MiningPhase::Running || st.phase.is_busy() {
                return;
            }
        }

        let local = Self::is_local(&pool_url);
        let effective = if local {
            Self::default_pool_url()
        } else {
            pool_url.trim().to_string()
        };

        {
            let mut st = self.state.write().await;
            *st = MiningState {
                phase: MiningPhase::Starting,
                message: Some(if local {
                    "starting local pool…".into()
                } else {
                    format!("connecting to {}…", effective)
                }),
                payout_address: payout_address.clone(),
                pool_url: effective.clone(),
                local_pool: local,
                threads,
                started_at: Some(Instant::now()),
                ..Default::default()
            };
        }

        let this = self.clone();
        tokio::spawn(async move {
            if let Err(e) = this.start_inner(payout_address, threads, effective, local).await {
                tracing::warn!("mining failed to start: {e:#}");
                this.kill_children().await;
                let mut st = this.state.write().await;
                st.phase = MiningPhase::Error(format!("{e:#}"));
                st.message = None;
                st.started_at = None;
            }
        });
    }

    async fn start_inner(
        &self,
        payout_address: String,
        threads: usize,
        pool_url: String,
        local: bool,
    ) -> Result<()> {
        if payout_address.trim().is_empty() {
            bail!("a payout address is required before mining can start");
        }
        let binary = Self::find_binary()?;
        std::fs::create_dir_all(&self.work_dir)
            .with_context(|| format!("could not create {}", self.work_dir.display()))?;

        let mut stratum_port = BASE_STRATUM_PORT;
        let mut audit_port = BASE_AUDIT_PORT;

        if local {
            self.set_message("starting local pool…".into()).await;

            // Always the canonical base. The pool bumps stratum and audit by the
            // same offset from their own bases (3333 and 8081), so handing it a
            // pre-probed port would desynchronise the pair — see
            // discover_pool_ports.
            //
            // fee 0: a solo pool of one takes no cut, so the user's own address
            // serves as both pool address and payout target.
            let pool = Command::new(&binary)
                .current_dir(&self.work_dir)
                .arg("pool")
                .arg("--pool-address").arg(&payout_address)
                .arg("--bind-addr").arg(format!("127.0.0.1:{}", BASE_STRATUM_PORT))
                .arg("--rpc-host").arg("127.0.0.1")
                .arg("--rpc-port").arg(self.node_rpc_port.to_string())
                .arg("--fee").arg("0")
                .kill_on_drop(true)
                .spawn()
                .with_context(|| format!("could not launch {} pool", binary.display()))?;
            self.children.lock().await.pool = Some(pool);

            self.set_message("waiting for the pool to accept connections…".into()).await;
            let (s_port, a_port) = self.discover_pool_ports().await?;
            stratum_port = s_port;
            audit_port = a_port;
            self.set_message(format!("pool up on port {}", s_port)).await;
        }

        let url = if local {
            format!("stratum+tcp://127.0.0.1:{}", stratum_port)
        } else {
            pool_url.clone()
        };

        self.set_message("starting hasher…".into()).await;
        let miner = Command::new(&binary)
            .current_dir(&self.work_dir)
            .arg("miner")
            .arg("--pool-url").arg(&url)
            .arg("--payout-address").arg(&payout_address)
            .arg("--worker").arg("desktop")
            .arg("--threads").arg(threads.to_string())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow!("could not launch hasher: {e}"))?;
        self.children.lock().await.miner = Some(miner);

        let mut st = self.state.write().await;
        st.phase = MiningPhase::Running;
        st.message = None;
        st.pool_url = url;
        st.stratum_port = stratum_port;
        st.audit_port = audit_port;
        st.started_at = Some(Instant::now());
        tracing::info!("mining running: {} threads via {}", threads, st.pool_url);
        Ok(())
    }

    /// Stops everything and returns immediately.
    ///
    /// # Safety / Invariants
    /// - **Order.** The hasher is killed first so it does not spend its final
    ///   moments submitting shares to a pool that is shutting down.
    /// - **Idempotent.** Safe to call when already stopped.
    pub async fn stop(&self) {
        {
            let mut st = self.state.write().await;
            if st.phase == MiningPhase::Idle {
                return;
            }
            st.phase = MiningPhase::Stopping;
            st.message = Some("stopping…".into());
        }
        let this = self.clone();
        tokio::spawn(async move {
            this.kill_children().await;
            let mut st = this.state.write().await;
            *st = MiningState::default();
            tracing::info!("mining stopped");
        });
    }

    async fn kill_children(&self) {
        let mut c = self.children.lock().await;
        if let Some(mut m) = c.miner.take() {
            let _ = m.kill().await;
        }
        if let Some(mut p) = c.pool.take() {
            let _ = p.kill().await;
        }
    }

    async fn set_message(&self, msg: String) {
        let mut st = self.state.write().await;
        st.message = Some(msg);
    }

    /// Reaps exited children so a crash surfaces instead of showing "running"
    /// forever. Call from whatever tick refreshes node info.
    pub async fn poll(&self) {
        if self.state.read().await.phase != MiningPhase::Running {
            return;
        }

        let mut died: Option<String> = None;
        {
            let mut c = self.children.lock().await;
            if let Some(m) = c.miner.as_mut() {
                if let Ok(Some(status)) = m.try_wait() {
                    died = Some(format!("the hasher exited ({status})"));
                }
            }
            if died.is_none() {
                if let Some(p) = c.pool.as_mut() {
                    if let Ok(Some(status)) = p.try_wait() {
                        died = Some(format!("the local pool exited ({status})"));
                    }
                }
            }
        }

        if let Some(reason) = died {
            tracing::warn!("mining stopped unexpectedly: {reason}");
            self.kill_children().await;
            let mut st = self.state.write().await;
            *st = MiningState {
                phase: MiningPhase::Error(reason),
                ..Default::default()
            };
        }
    }

    /// Discovers which port pair the pool actually bound.
    ///
    /// # Reasoning
    /// `run_stratum_pool` runs its own tandem-binding loop, starting from the
    /// port given in `--bind-addr` for stratum and **always from 8081** for the
    /// audit API, incrementing both by the same offset until a free pair is
    /// found. The two bases are independent.
    ///
    /// An earlier version pre-probed for a free pair and passed the result as
    /// `--bind-addr`. That silently broke the correspondence: probing found
    /// 3335/8083 free, the pool was told base 3335, computed `3335 + 0` and
    /// `8081 + 0`, and bound 3335/**8081** — while the supervisor polled 8083
    /// and timed out on a pool that was running perfectly.
    ///
    /// So: always pass the canonical base 3333, then find the pool by polling
    /// 8081 upward. Because the pool moves both ports by the same offset from
    /// the bases it was given, the stratum port follows from the audit port.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  a pool child has been spawned with --bind-addr host:3333
    /// Post: Ok((stratum, audit)) ⇒ audit ∈ [8081, 8081+10]
    ///                            ∧ stratum = 3333 + (audit - 8081)
    ///                            ∧ the audit API answered on that port
    ///       Err(_) ⇒ no pool answered anywhere in the range
    /// ```
    async fn discover_pool_ports(&self) -> Result<(u16, u16)> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(1500))
            .build()?;

        let deadline = Instant::now() + POOL_READY_TIMEOUT;
        while Instant::now() < deadline {
            for offset in 0..=MAX_PORT_OFFSET {
                let audit = BASE_AUDIT_PORT + offset;
                let url = format!("http://127.0.0.1:{}/pool/stats", audit);
                if let Ok(r) = client.get(&url).send().await {
                    if r.status().is_success() {
                        return Ok((BASE_STRATUM_PORT + offset, audit));
                    }
                }
            }

            // Bail early if the pool process already exited — no point waiting
            // out the full timeout on something that is gone.
            {
                let mut c = self.children.lock().await;
                if let Some(p) = c.pool.as_mut() {
                    if let Ok(Some(status)) = p.try_wait() {
                        bail!("the local pool exited during startup ({status})");
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        bail!(
            "no local pool answered on ports {}–{} within {}s",
            BASE_AUDIT_PORT,
            BASE_AUDIT_PORT + MAX_PORT_OFFSET,
            POOL_READY_TIMEOUT.as_secs()
        )
    }

    /// Full snapshot of the local pool's audit API.
    ///
    /// # Reasoning
    /// `/pool/stats` publishes far more than a share count, and using it
    /// properly removes two guesses an earlier version made:
    ///
    /// - Blocks live under `recent_blocks`, not `blocks`. Reading the wrong key
    ///   meant "blocks found" was permanently zero.
    /// - `hashes_per_share` is published by the pool, derived from the actual
    ///   share target in force. Hardcoding `2^12` was wrong for any pool started
    ///   with a non-default `--share-bits`.
    ///
    /// It also exposes `network_hashrate`, `network_height`, `block_reward` and
    /// a per-worker breakdown, which is what makes a real dashboard possible
    /// rather than three lonely numbers.
    ///
    /// Returns `None` for a remote pool: those stats belong on the operator's
    /// own dashboard, and we have no authority to present them as this wallet's.
    pub async fn audit_stats(&self) -> Option<PoolStats> {
        let st = self.state.read().await.clone();
        if st.phase != MiningPhase::Running || !st.local_pool {
            return None;
        }
        let url = format!("http://127.0.0.1:{}/pool/stats", st.audit_port);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .ok()?;
        let v: serde_json::Value = client.get(&url).send().await.ok()?.json().await.ok()?;

        let mine = st.payout_address.to_lowercase();
        let is_mine = |val: &serde_json::Value| -> bool {
            val.get("address")
                .and_then(|a| a.as_str())
                .map(|a| a.to_lowercase() == mine)
                .unwrap_or(false)
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
        let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).unwrap_or(0);

        let mut out = PoolStats {
            network_hashrate: f("network_hashrate"),
            hashes_per_share: f("hashes_per_share"),
            network_height: u("network_height"),
            block_reward: u("block_reward"),
            active_miners: u("active_miners"),
            total_score: u("total_score"),
            ..Default::default()
        };

        if let Some(arr) = v.get("miners").and_then(|m| m.as_array()) {
            for m in arr.iter().filter(|m| is_mine(m)) {
                out.accepted += m.get("accepted").and_then(|x| x.as_u64()).unwrap_or(0);
                out.rejected += m.get("rejected").and_then(|x| x.as_u64()).unwrap_or(0);
                out.score += m.get("score").and_then(|x| x.as_u64()).unwrap_or(0);
            }
        }
        if let Some(arr) = v.get("recent_blocks").and_then(|b| b.as_array()) {
            out.blocks_found = arr.iter().filter(|b| is_mine(b)).count() as u64;
            out.pool_blocks = arr.len() as u64;
        }
        if let Some(arr) = v.get("workers").and_then(|w| w.as_array()) {
            out.workers = arr
                .iter()
                .filter(|w| is_mine(w))
                .filter_map(|w| {
                    Some((
                        w.get("worker")?.as_str()?.to_string(),
                        w.get("score")?.as_u64()?,
                    ))
                })
                .collect();
        }

        // Hashrate from accepted shares against the pool's published
        // hashes-per-share. High variance early on; the UI should say so.
        let elapsed = st.started_at.map(|t| t.elapsed().as_secs()).unwrap_or(0).max(1);
        if out.hashes_per_share > 0.0 {
            out.hashrate = (out.accepted as f64 * out.hashes_per_share) / elapsed as f64;
        }
        if out.network_hashrate > 0.0 {
            out.network_share = (out.hashrate / out.network_hashrate).min(1.0);
        }
        Some(out)
    }

}

/// Working directory for mining children — keeps `pool_stratum.redb` next to
/// the rest of the app's data rather than wherever the GUI happened to launch.
pub fn work_dir_for(data_dir: &Path) -> PathBuf {
    data_dir.join("mining")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_loopback_urls_are_local() {
        assert!(MiningSupervisor::is_local(""));
        assert!(MiningSupervisor::is_local("   "));
        assert!(MiningSupervisor::is_local("stratum+tcp://127.0.0.1:3333"));
        assert!(MiningSupervisor::is_local("localhost:3333"));
        assert!(MiningSupervisor::is_local("STRATUM+TCP://LOCALHOST:3333"));
    }

    #[test]
    fn remote_urls_are_not_local() {
        assert!(!MiningSupervisor::is_local(
            "stratum+tcp://rpc.cypherpunk.gold:3333"
        ));
        assert!(!MiningSupervisor::is_local("stratum+tcp://10.0.0.5:3333"));
    }

    #[test]
    fn phase_busy_only_during_transitions() {
        assert!(MiningPhase::Starting.is_busy());
        assert!(MiningPhase::Stopping.is_busy());
        assert!(!MiningPhase::Idle.is_busy());
        assert!(!MiningPhase::Running.is_busy());
        assert!(!MiningPhase::Error("x".into()).is_busy());
    }
}
