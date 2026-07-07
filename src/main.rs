use anyhow::{Result, Context, bail};
use clap::{Parser, Subcommand};
use midstate::*;
use midstate::compute_address;
use midstate::wallet::{self, Wallet};
use midstate::core::wots;
use midstate::core::state::apply_batch;
use midstate::network::{MidstateNetwork, NetworkEvent, Message};
use midstate::core::types; // for hash + count_leading_zeros in PoAW generator
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use rayon::prelude::*;

// Use jemalloc on Linux/macOS (excellent for anti-fragmentation on the Pi)
#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

// Use mimalloc on Windows (since jemalloc does not support MSVC)
#[cfg(target_env = "msvc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// ── Config file ─────────────────────────────────────────────────────────────

#[derive(Debug, Default, serde::Deserialize, serde::Serialize, Clone)]
struct Config {
    /// Bootstrap peer multiaddrs
    #[serde(default)]
    bootstrap_peers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    peer_id: Option<String>,
    /// List of Peer IDs to block from connecting
    #[serde(default)]
    banned_peers: Vec<String>,
    /// Enable automatic pruning of old historical blocks.
    /// When true, the node will only retain the most recent PRUNE_DEPTH blocks.
    #[serde(default)]
    prune: bool,
    /// Optional path to a wallet whose Pruning Licenses should be auto-registered
    /// at node startup. Both held licenses (pruning exemption rights) and issued licenses
    /// (storage audit obligations as the original Issuer) are registered.
    /// Password must be supplied via the MIDSTATE_LICENSE_WALLET_PASSWORD environment variable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    license_wallet_path: Option<PathBuf>,
}

impl Config {
    fn load(path: &std::path::Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config: {}", path.display()))?;
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config: {}", path.display()))?;
        Ok(config)
    }

    /// Create a default config file if it doesn't exist.
    fn create_default(path: &std::path::Path) -> Result<()> {
        if path.exists() {
            return Ok(());
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let default_contents = r#"# Midstate node configuration
            #
            # Bootstrap peers — full multiaddr with peer ID.
            # Get peer ID from the bootstrap node's startup logs.
            #
            # Example:
            # bootstrap_peers = [
            #     "/ip4/203.0.113.10/tcp/9333/p2p/12D3KooWAbCdEf...",
            #     "/ip4/198.51.100.5/tcp/9333/p2p/12D3KooWGhIjKl...",
            # ]

            bootstrap_peers = [
                "/ip4/134.199.148.215/tcp/9333/p2p/12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv",
                "/ip4/74.208.253.44/tcp/9333/p2p/12D3KooWBqph3BWQxc3xsusvCijS88RaAEZagZZwwxAwP2Xs1CTE"
            ]

            # List of Peer IDs to block from connecting
            banned_peers = []

            # Enable pruning of old block data (keeps only the last PRUNE_DEPTH blocks).
            # Default is false (full archival mode). Set to true to save disk space.
            # prune = false

            # Optional: auto-register Pruning Licenses from a wallet at startup.
            # Registers held ranges (for exemption when pruning) + issued ranges (audit obligations as the original Issuer).
            # The password is never stored in the config; provide it via env var.
            "#;
        std::fs::write(path, default_contents)
            .with_context(|| format!("Failed to create default config: {}", path.display()))?;
        tracing::info!("Created default config at {}", path.display());
        Ok(())
    }
}

fn default_wallet_path() -> PathBuf {
    wallet::default_path()
}

#[derive(Parser)]
#[command(name = "midstate")]
#[command(about = "A minimal sequential-time cryptocurrency", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Node {
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        #[arg(long, default_value = "9333")]
        port: u16,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_bind: String,
        #[arg(long)]
        peer: Vec<String>,
        #[arg(long)]
        mine: bool,
        #[arg(long)] 
        threads: Option<usize>,
        
         /// Mining backend: "auto" (default, prefer GPU), "gpu", or "cpu".
        #[arg(long, default_value = "auto")]
        backend: String,
        
        #[arg(long)]
        verify_threads: Option<usize>,
        #[arg(long)]
        listen: Option<String>,
        #[arg(long)]
        config: Option<PathBuf>,

        /// Enable automatic pruning of old block data (keeps only the last PRUNE_DEPTH blocks).
        /// Default is false (archival mode - full history is retained).
        #[arg(long)]
        prune: bool,

        /// Optional wallet to auto-register Pruning Licenses from at startup.
        /// Registers both held licenses (pruning exemption) and issued licenses (audit obligations as Issuer).
        /// Requires MIDSTATE_LICENSE_WALLET_PASSWORD in the environment.
        #[arg(long)]
        license_wallet: Option<PathBuf>,
    },
    Wallet {
        #[command(subcommand)]
        action: WalletAction,
    },
    Commit {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: Vec<String>,
        #[arg(long)]
        dest: Vec<String>,
    },
    Balance {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: String,
    },
    State {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Mempool {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Peers {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Keygen {
        #[arg(long)]
        rpc_port: Option<u16>,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Sync {
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        #[arg(long)]
        peer: String,
        #[arg(long, default_value = "9333")]
        port: u16,
    },
 /// Run the Provably Fair Stratum Pool Server
    Pool {
        #[arg(long)]
        pool_address: String,
        #[arg(long, default_value = "0.0.0.0:3333")]
        bind_addr: String,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        /// The percentage fee the pool takes from block rewards (e.g. 1.0 for 1%)
        #[arg(long, default_value = "1.0")]
        fee: f64,
    },
    /// Pure Hasher: Connect to a Stratum pool without running a full node
    Miner {
        #[arg(long)]
        pool_url: String,
        #[arg(long)]
        payout_address: String,
        #[arg(long, default_value = "default")]
        worker: String,
        #[arg(long, default_value = "0")]
        threads: usize,
    },
}

#[derive(Subcommand)]
enum WalletAction {
    Create {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        legacy: bool,
    },
    Restore {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        phrase: Option<String>,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Receive {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        label: Option<String>,
    },
    Generate {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, short, default_value = "1")]
        count: usize,
        #[arg(long)]
        label: Option<String>,
    },
    GenerateMss {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "10")]
        height: u32,
        #[arg(long)]
        label: Option<String>,
    },
    List {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        full: bool,
    },
    Compile {
        #[arg(long)]
        file: PathBuf,
    },
    Balance {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    Abandon {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        address: String,
    },
    Send {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: Vec<String>,
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
        #[arg(long)]
        private: bool,
    },
    SpendScript {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: Vec<String>,             // <--- NOW AN ARRAY
        #[arg(long)]
        bytecode: String,
        #[arg(long)]
        inputs: Vec<String>,           // <--- NOW AN ARRAY
        #[arg(long)]
        input_state: Vec<String>,      // <--- NOW AN ARRAY
        #[arg(long)]
        burn_data: Option<String>,
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
    },
    Import {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        seed: String,
        #[arg(long)]
        value: u64,
        #[arg(long)]
        salt: String,
        #[arg(long)]
        label: Option<String>,
    },
    Export {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        coin: String,
    },
    Pending {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
    },
    Consolidate {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        address: String,
    },
    Reveal {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        commitment: Option<String>,
    },
    History {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, short, default_value = "20")]
        count: usize,
    },
    /// Import coinbase rewards from mining log
    ImportRewards {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        coinbase_file: PathBuf,
        /// Path to the node's data directory to read the mining seed
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
    },
    Scan {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        from_genesis: bool,
    },
    Mix {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        denomination: u64,
        #[arg(long)]
        coin: Option<String>,
        #[arg(long)]
        join: Option<String>,
        #[arg(long)]
        pay_fee: bool,
        #[arg(long, default_value = "300")]
        timeout: u64,
    },
    AutoMix {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: String,
        #[arg(long, default_value = "1200")]
        timeout: u64,
    },
    // ── Pruning Licenses (Phase 1 + 2 Wiring) ───────────────────────────────
    /// Issue a new Pruning License as a Confidential State Thread.
    /// Primary usage: provide --bundle from `poaw-generate`.
    IssueLicense {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// Path to the .poaw-bundle.json produced by `poaw-generate` (recommended)
        #[arg(long)]
        bundle: Option<PathBuf>,
        /// PoAW commitment (hex) - used if --bundle is not provided
        #[arg(long)]
        poaw_commitment: Option<String>,
        /// Fixed royalty fee (in Midstate) paid to the original issuer on every transfer.
        /// This is the recommended model (harder to evade than percentage royalties).
        #[arg(long, default_value = "100")]
        fixed_fee: u64,
        #[arg(long)]
        min_height: Option<u64>,
        #[arg(long)]
        max_height: Option<u64>,
        #[arg(long)]
        archival_weight: Option<u64>,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    /// Purchase an existing Pruning License from another holder.
    /// The transaction will include the required royalty payment + burn-to-boost output.
    BuyLicense {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// The coin_id (or address) of the license being purchased
        #[arg(long)]
        license: String,
        /// How much you are paying the current owner (in addition to royalty)
        #[arg(long)]
        price: u64,
        /// Public key of the current license holder (seller). Must be the raw 32-byte WOTS PK
        /// (not an address). This is required for the HTLC atomic swap.
        #[arg(long)]
        seller_pk: String,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long, default_value = "120")]
        timeout: u64,
    },
    /// After revealing a license on-chain, re-key its metadata from the old
    /// issuance commitment to the real coin_id of the Confidential output.
    RekeyLicense {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// The old key (usually the on-chain commitment used at issuance)
        #[arg(long)]
        old: String,
        /// The new real coin_id of the Confidential license output
        #[arg(long)]
        new_coin_id: String,
    },
    /// List all Pruning Licenses this wallet knows about (with their royalty terms).
    ListLicenses {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
    },
    /// Seller side of license sale via HTLC atomic swap.
    ///
    /// Without --secret: Generate a fresh secret + hash and print the hash for the buyer.
    /// With --secret: Claim mode — perform the license transfer Reveal and provide the preimage
    /// to claim the buyer's HTLC payment (full atomic multi-input claim is deeper integration).
    SellLicense {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// The coin_id of the license you are selling
        #[arg(long)]
        license: String,
        /// Raw secret (hex) to use in claim mode. If omitted, generates a new secret for the offer phase.
        #[arg(long)]
        secret: Option<String>,
        /// Buyer's raw WOTS public key (hex) — the new license Confidential output will be created for this key.
        /// Required in claim mode.
        #[arg(long)]
        buyer_pk: Option<String>,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    /// Recover LicenseMetadata from an on-chain DataBurn (bearer asset recovery after wallet.dat loss).
    /// Provide the tx hash of the issuance/transfer that contained the DataBurn and the block height.
    /// The tool scans the block outputs for a DataBurn payload that deserializes as LicenseMetadata
    /// and restores it into your encrypted wallet.dat so you can spend/transfer the license again.
    RecoverLicenseMetadata {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// Tx hash / commitment of the transaction containing the DataBurn
        #[arg(long)]
        tx: String,
        /// Height of the block containing the DataBurn transaction
        #[arg(long)]
        height: u64,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
    // ── Phase 2: Proof of Archival Work Generator ───────────────────────────
    /// Generate Proof-of-Archival-Work for a historical range.
    /// This is the expensive step required before you can issue a valuable
    /// Pruning License.
    PoawGenerate {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// Start block height (inclusive)
        #[arg(long)]
        start: u64,
        /// End block height (inclusive)
        #[arg(long)]
        end: u64,
        /// Sampling stride (e.g. 10 means sample every 10th block)
        #[arg(long, default_value = "10")]
        stride: u64,
        /// Difficulty target for the address-salted PoW (leading zero bits)
        #[arg(long, default_value = "20")]
        difficulty: u32,
        /// Your long-term issuer address (hex) used as the salt
        #[arg(long)]
        issuer: String,
        /// If set, automatically submit a Transaction::Commit with the PoAW root
        #[arg(long)]
        submit_commit: bool,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
    },
}

fn read_password(prompt: &str) -> Result<Vec<u8>> {
    if let Ok(val) = std::env::var("MIDSTATE_PASSWORD") {
        if val.is_empty() { anyhow::bail!("MIDSTATE_PASSWORD is set but empty"); }
        return Ok(val.into_bytes());
    }
    let input = rpassword::prompt_password(prompt)?;
    if input.is_empty() { anyhow::bail!("password cannot be empty"); }
    Ok(input.into_bytes())
}

fn read_password_confirm() -> Result<Vec<u8>> {
    if std::env::var("MIDSTATE_PASSWORD").is_ok() {
        return read_password(""); // env var skips confirmation
    }
    let p1 = read_password("Password: ")?;
    let p2 = read_password("Confirm:  ")?;
    if p1 != p2 { anyhow::bail!("passwords do not match"); }
    Ok(p1)
}

fn parse_hex32(s: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(s)?;
    if bytes.len() != 32 { anyhow::bail!("expected 32 bytes, got {}", bytes.len()); }
    Ok(<[u8; 32]>::try_from(bytes).unwrap())
}

enum ParsedOutput {
    Standard([u8; 32], u64),
    Stateful([u8; 32], [u8; 32]),
}

fn parse_output_spec(s: &str) -> Result<ParsedOutput> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        anyhow::bail!("expected format <address>:<value>[:<state_hex>]");
    }
    
    let addr = midstate::core::types::parse_address_flexible(parts[0])
        .map_err(|e| anyhow::anyhow!(e))?;
        
    let value: u64 = parts[1].parse()
        .map_err(|_| anyhow::anyhow!("invalid value: {}", parts[1]))?;
        
    if parts.len() == 3 {
        if value != 0 {
            anyhow::bail!("State threads (Confidential outputs) must have value exactly 0. To send value AND state, create two outputs.");
        }
        let state = parse_hex32(parts[2]).map_err(|_| anyhow::anyhow!("invalid state hex: {}", parts[2]))?;
        Ok(ParsedOutput::Stateful(addr, state))
    } else {
        if value == 0 {
            anyhow::bail!("Standard outputs must have value > 0");
        }
        Ok(ParsedOutput::Standard(addr, value))
    }
}

fn format_age(secs: u64) -> String {
    if secs < 60 { format!("{}s ago", secs) }
    else if secs < 3600 { format!("{}m ago", secs / 60) }
    else if secs < 86400 { format!("{}h ago", secs / 3600) }
    else { format!("{}d ago", secs / 86400) }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "midstate=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Node { data_dir, port, rpc_port, rpc_bind, peer, mine, threads,
                verify_threads, listen, config, prune, license_wallet, backend } => {
                    midstate::core::gpu_mining::set_backend(match backend.to_ascii_lowercase().as_str() {
                        "gpu"  => midstate::core::gpu_mining::Backend::Gpu,
                        "cpu"  => midstate::core::gpu_mining::Backend::Cpu,
                        "auto" => midstate::core::gpu_mining::Backend::Auto,
                        other  => {
                            tracing::warn!("unknown --backend '{other}', using auto");
                            midstate::core::gpu_mining::Backend::Auto
                        }
                    });
                    run_node(data_dir, port, rpc_port, rpc_bind, peer, mine, threads,
                             verify_threads, listen, config, prune, license_wallet).await
                }
        Command::Wallet { action } => handle_wallet(action).await,
        Command::Commit { rpc_port, rpc_host, coin, dest } => {
            commit_transaction(rpc_port, rpc_host, coin, dest).await
        }
        Command::Balance { rpc_port, rpc_host, coin } => check_balance(rpc_port, rpc_host, coin).await,
        Command::State { rpc_port, rpc_host } => get_state(rpc_port, rpc_host).await,
        Command::Mempool { rpc_port, rpc_host } => get_mempool(rpc_port, rpc_host).await,
        Command::Peers { rpc_port, rpc_host } => get_peers(rpc_port, rpc_host).await,
        Command::Keygen { rpc_port, rpc_host } => keygen(rpc_port, rpc_host).await,
        Command::Sync { data_dir, peer, port } => sync_from_genesis(data_dir, peer, port).await,
        Command::Pool { pool_address, bind_addr, rpc_port, rpc_host, fee } => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let node_rpc_url = format!("http://{}:{}", rpc_host, rpc_port);
                midstate::pool::run_stratum_pool(pool_address, bind_addr, node_rpc_url, fee).await;
                Ok(())
            }
            #[cfg(target_arch = "wasm32")]
            {
                anyhow::bail!("Pool server cannot run in WebAssembly");
            }
        }
        Command::Miner { pool_url, payout_address, worker, threads } => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let hash_counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
                let stats = std::sync::Arc::new(std::sync::RwLock::new(midstate::mining::StratumStats::default()));
                
                let hc = hash_counter.clone();
                let stats_clone = stats.clone();
                
                tokio::spawn(async move {
                    use tokio::io::AsyncBufReadExt;
                    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin());
                    let mut line = String::new();
                    
                    let mut last_hashes = 0;
                    let mut last_time = std::time::Instant::now();

                    // Hardcoded share target matching the pool (0x000f...)
                    let share_target = {
                        let mut t = [0xff; 32];
                        t[0] = 0x00; t[1] = 0x0f;
                        t
                    };

                    fn u256_to_f64(u: primitive_types::U256) -> f64 {
                        u.0[0] as f64 +
                        (u.0[1] as f64) * 2.0f64.powi(64) +
                        (u.0[2] as f64) * 2.0f64.powi(128) +
                        (u.0[3] as f64) * 2.0f64.powi(192)
                    }

                    fn format_time(secs: f64) -> String {
                        if secs < 60.0 { return format!("{:.0}s", secs); }
                        if secs < 3600.0 { return format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0); }
                        if secs < 86400.0 { return format!("{:.0}h {:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0); }
                        if secs < 31536000.0 { return format!("{:.0}d {:.0}h", secs / 86400.0, (secs % 86400.0) / 3600.0); }
                        format!("{:.1} years", secs / 31536000.0)
                    }

                    let share_target_f64 = u256_to_f64(primitive_types::U256::from_big_endian(&share_target));

                    loop {
                        line.clear();
                        if stdin.read_line(&mut line).await.unwrap_or(0) == 0 { break; } 
                        
                        let current = hc.load(std::sync::atomic::Ordering::Relaxed);
                        let now = std::time::Instant::now();
                        let elapsed = now.duration_since(last_time).as_secs_f64();
                        let rate = if elapsed > 0.0 { (current - last_hashes) as f64 / elapsed } else { 0.0 };
                        
                        println!("\n╔════════════ MINER STATUS ════════════╗");
                        println!("║ Hashrate:      {:.2} nonces/s", rate);
                        
                        let s = stats_clone.read().unwrap().clone();
                        if s.network_target != [0u8; 32] {
                            let target_f64 = u256_to_f64(primitive_types::U256::from_big_endian(&s.network_target));
                            
                            let expected_nonces = 2.0f64.powi(256) / target_f64.max(1.0);
                            let network_nps = expected_nonces / 60.0;
                            let share_pct = if network_nps > 0.0 { (rate / network_nps) * 100.0 } else { 0.0 };
                            
                            // Luck Math
                            let expected_shares_per_block = share_target_f64 / target_f64.max(1.0);
                            let session_effort_pct = if expected_shares_per_block > 0.0 {
                                (s.accepted_shares as f64 / expected_shares_per_block) * 100.0
                            } else { 0.0 };
                            
                            println!("║ Network:       {:.2} nonces/s", network_nps);
                            
                            if share_pct < 0.001 {
                                println!("║ Your Share:    < 0.001%");
                            } else {
                                println!("║ Your Share:    {:.4}%", share_pct);
                            }
                            
                            if rate > 0.0 {
                                println!("║ Solo ETA:      {}", format_time(expected_nonces / rate));
                            } else {
                                println!("║ Solo ETA:      ---");
                            }
                            println!("╠┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╣");
                            println!("║ Shares:        {} acc / {} rej", s.accepted_shares, s.rejected_shares);
                            
                            // Formatting the Luck / Effort string
                            println!("║ Expected:      1 block per {} shares", expected_shares_per_block.round() as u64);
                            println!("║ Session Luck:  {:.2}% {}", 
                                session_effort_pct, 
                                if session_effort_pct >= 100.0 { "🍀 (Due for a block!)" } else { "⏳" }
                            );
                        } else {
                            println!("║ Network:       Waiting for job...");
                        }
                        
                        println!("╚══════════════════════════════════════╝\n");
                        
                        last_hashes = current;
                        last_time = now;
                    }
                });

                tracing::info!("starting hasher (threads: {})", if threads == 0 { "max".to_string() } else { threads.to_string() });
                tracing::info!("press [ENTER] at any time to view dashboard");
                
                midstate::mining::run_stratum_client(pool_url, payout_address, worker, threads, hash_counter, stats).await;
                Ok(())
            }
            #[cfg(target_arch = "wasm32")]
            {
                anyhow::bail!("Pure miner cannot run in WebAssembly");
            }
        }
    }
}

// ── Wallet commands ─────────────────────────────────────────────────────────
async fn wallet_scan(path: &PathBuf, rpc_port: u16, rpc_host: String, from_genesis: bool) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let addresses = wallet.watched_addresses();
    if addresses.is_empty() {
        println!("No addresses to scan for. Generate keys first.");
        return Ok(());
    }

    let base_url = format!("http://{}:{}", rpc_host, rpc_port);
    let state: rpc::GetStateResponse = client.get(format!("{}/state", base_url))
        .send().await?.json().await?;
    let chain_height = state.height;

    // Use the from_genesis flag here:
    let mut start = if from_genesis { 0 } else { wallet.data.last_scan_height };

    // ── Reorg guard ──
    // If the network tip is now BELOW our last scanned height, the chain we scanned
    // was (partly) orphaned. Without clamping, the `start >= chain_height` early
    // return below fires on every run until the new fork outgrows the stale pointer
    // — and the replaced block range is then skipped forever, so coins on the
    // winning fork are never discovered. Rewind the pointer to the new tip and
    // PERSIST it, so scanning resumes from there as the winning fork grows.
    if !from_genesis && start > chain_height {
        println!(
            "Chain reorg detected: network height {} is below last scanned height {}. Rewinding scan pointer to {}.",
            chain_height, start, chain_height
        );
        println!("  (Coins received on the orphaned fork may no longer exist; run `wallet scan --from-genesis` if balances look wrong.)");
        start = chain_height;
        wallet.data.last_scan_height = chain_height;
        wallet.save()?;
    }

    if start >= chain_height && !from_genesis {
        println!("Already scanned to height {}. Chain is at {}.", start, chain_height);
        return Ok(());
    }

    println!("Scanning blocks {}..{} for {} address(es) using compact filters...", start, chain_height, addresses.len());

    // Phase 1: Filter scan — find which blocks might contain our addresses
    let matching_heights = filter_scan(&client, &base_url, &addresses, start, chain_height).await?;

    if matching_heights.is_empty() {
        println!("  No filter matches found.");
    } else {
        println!("  {} block(s) matched filters, fetching details...", matching_heights.len());
    }

    // Phase 2: Targeted scan — only fetch full data for matching blocks
    let imported = targeted_scan(&client, &base_url, &mut wallet, &addresses, &matching_heights).await?;

    wallet.data.last_scan_height = chain_height;
    
    // Sync MSS key indices (stateful recovery)
    if !wallet.data.mss_keys.is_empty() {
        println!("Syncing MSS key indices...");
        let mss_url = format!("{}/mss_state", base_url);
        for mss_key in &mut wallet.data.mss_keys {
            let req = rpc::GetMssStateRequest {
                master_pk: hex::encode(mss_key.master_pk),
            };
            match client.post(&mss_url).json(&req).send().await {
                Ok(resp) => {
                    if let Ok(mss_resp) = resp.json::<rpc::GetMssStateResponse>().await {
                        if mss_resp.next_index > mss_key.next_leaf {
                            const SAFETY_MARGIN: u64 = 20;
                            let new_leaf = mss_resp.next_index + SAFETY_MARGIN;
                            println!("  MSS {}: advancing leaf {} -> {}",
                                hex::encode(&mss_key.master_pk), mss_key.next_leaf, new_leaf);
                            mss_key.next_leaf = new_leaf;
                        }
                    }
                }
                Err(e) => {
                    println!("  MSS {} sync failed: {}", hex::encode(&mss_key.master_pk), e);
                }
            }
        }
    }
    
    wallet.save()?;

    println!("Scan complete. {} new coin(s) found. Scanned to height {}.", imported, chain_height);
    Ok(())
}

/// Download compact filters in batches and test which blocks might contain
/// any of the given addresses. Returns the list of block heights with
/// potential matches (may include false positives at rate ~1/1M per block).
async fn filter_scan(
    client: &reqwest::Client,
    base_url: &str,
    addresses: &[[u8; 32]],
    start: u64,
    end: u64,
) -> Result<Vec<u64>> {
    use midstate::core::filter::match_any;

    let mut matching = Vec::new();
    let filter_url = format!("{}/filters", base_url);
    const FILTER_BATCH: u64 = 500;

    let mut cursor = start;
    while cursor < end {
        let batch_end = (cursor + FILTER_BATCH).min(end);
        let req = rpc::GetFiltersRequest {
            start_height: cursor,
            end_height: batch_end,
        };
        let resp: rpc::GetFiltersResponse = client.post(&filter_url)
            .json(&req).send().await?.json().await?;

        for (i, filter_hex) in resp.filters.iter().enumerate() {
            let height = resp.start_height + i as u64;

            // Decode the metadata needed for client-side matching
            let block_hash = if i < resp.block_hashes.len() {
                parse_hex32(&resp.block_hashes[i])?
            } else {
                // Fallback: server didn't send hashes (old node), do full scan
                matching.push(height);
                continue;
            };
            let n = if i < resp.element_counts.len() {
                resp.element_counts[i]
            } else {
                matching.push(height);
                continue;
            };

            if n == 0 {
                continue;
            }

            let filter_data = hex::decode(filter_hex)?;
            if match_any(&filter_data, &block_hash, n, addresses) {
                matching.push(height);
            }
        }

        cursor = batch_end;
    }
    Ok(matching)
}

/// Fetch full blocks for specific heights and scan for our addresses client-side.
/// PRIVACY: This function never sends any addresses to the node. It downloads
/// raw blocks and matches outputs locally, preserving Neutrino-level privacy.
/// Returns count of newly imported coins.
async fn targeted_scan(
    client: &reqwest::Client,
    base_url: &str,
    wallet: &mut Wallet,
    addresses: &[[u8; 32]],
    heights: &[u64],
) -> Result<usize> {
    if heights.is_empty() {
        return Ok(0);
    }

    let addr_set: std::collections::HashSet<[u8; 32]> = addresses.iter().copied().collect();
    let mut imported = 0usize;

    for &height in heights {
        // Fetch the full block (witnesses stripped for bandwidth)
        let url = format!("{}/block/{}", base_url, height);
        let resp = client.get(&url).send().await?;
        if !resp.status().is_success() {
            continue;
        }
        let batch: midstate::core::Batch = resp.json().await?;
        let block_timestamp = batch.timestamp;
        let mut block_coins: Vec<[u8; 32]> = Vec::new();

        // Scan coinbase outputs
        for cb in &batch.coinbase {
            if addr_set.contains(&cb.address) {
                if let Some(coin_id) = wallet.import_scanned(cb.address, cb.value, cb.salt, None)? {
                    println!("  found: {} (value {}, height {})", hex::encode(&coin_id), cb.value, height);
                    block_coins.push(coin_id);
                    imported += 1;
                }
            }
        }

        // Scan transaction outputs
        for tx in &batch.transactions {
            if let midstate::core::Transaction::Reveal { outputs, .. } = tx {
                for out in outputs {
                    if addr_set.contains(&out.address()) {
                        if let Some(_cid) = out.coin_id() {
                            let commitment = if out.is_confidential() { out.commitment() } else { None };
                            if let Some(coin_id) = wallet.import_scanned(out.address(), out.value(), out.salt(), commitment)? {
                                println!("  found: {} (value {}, height {})", hex::encode(&coin_id), out.value(), height);
                                block_coins.push(coin_id);
                                imported += 1;
                            }
                        }
                    }
                }
            }
        }

        // Record all coins received in this block as a single history entry
        wallet.record_received(block_coins, block_timestamp);
    }

    Ok(imported)
}

async fn wallet_spend_script(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    coin_refs: Vec<String>,
    bytecode_hex: String,
    inputs_args: Vec<String>,
    input_states: Vec<String>,
    burn_data: Option<String>,
    to_args: Vec<String>,
    timeout_secs: u64,
) -> Result<()> {
    if coin_refs.is_empty() { anyhow::bail!("Must specify at least one --coin"); }
    
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let mut spending_coins = Vec::new();
    for cref in &coin_refs {
        // 1. Try local wallet first
        if let Ok(cid) = wallet.resolve_coin(cref) {
            if let Some(coin) = wallet.find_coin(&cid).cloned() {
                spending_coins.push(coin);
                continue;
            }
        }

        // 2. Fallback: Search the blockchain via RPC (for Smart Contract UTXOs)
        println!("Coin {} not in local wallet. Fetching from blockchain...", cref);
        let search_req = rpc::SearchRequest { query: cref.clone() };
        let search_url = format!("http://{}:{}/search", rpc_host, rpc_port);
        let search_resp = client.post(&search_url).json(&search_req).send().await?;
        
        if !search_resp.status().is_success() {
            anyhow::bail!("Failed to search network for coin {}", cref);
        }
        
        let search_data: rpc::SearchResponse = search_resp.json().await?;
        let mut found = false;
        
        for res in search_data.results {
            if res.result_type == "output_coin_id" || res.result_type == "coinbase_coin_id" {
                let block_url = format!("http://{}:{}/block/{}", rpc_host, rpc_port, res.height);
                let block_resp = client.get(&block_url).send().await?;
                if !block_resp.status().is_success() { continue; }
                
                let batch: midstate::core::Batch = block_resp.json().await?;
                
                if res.result_type == "coinbase_coin_id" {
                    for cb in &batch.coinbase {
                        if hex::encode(cb.coin_id()) == *cref {
                            spending_coins.push(midstate::wallet::WalletCoin {
                                seed: [0; 32], owner_pk: [0; 32],
                                address: cb.address, value: cb.value, salt: cb.salt,
                                coin_id: cb.coin_id(), label: Some("Coinbase UTXO".into()),
                                wots_signed: false, commitment: None,
                            });
                            found = true;
                            break;
                        }
                    }
                } else if let Some(tx_idx) = res.tx_index {
                    if let Some(tx) = batch.transactions.get(tx_idx) {
                        match tx {
                            midstate::core::Transaction::Reveal { outputs, .. } | 
                            midstate::core::Transaction::Consolidate { outputs, .. } => {
                                for out in outputs {
                                    if let Some(cid) = out.coin_id() {
                                        if hex::encode(cid) == *cref {
                                            spending_coins.push(midstate::wallet::WalletCoin {
                                                seed: [0; 32], owner_pk: [0; 32],
                                                address: out.address(), value: out.value(), salt: out.salt(),
                                                coin_id: cid, label: Some("Contract UTXO".into()),
                                                wots_signed: false, commitment: out.commitment(),
                                            });
                                            found = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if found { break; }
            }
        }
        
        if !found {
            anyhow::bail!("Coin {} not found in local wallet or on the blockchain.", cref);
        }
    }

    let bytecode = hex::decode(&bytecode_hex).context("Invalid bytecode hex")?;
    let script_address = midstate::core::types::hash(&bytecode);

    // Verify all spending coins match the bytecode address OR belong to the wallet (Co-spend)
    for coin in &spending_coins {
        if script_address != coin.address {
            if wallet.find_coin(&coin.coin_id).is_none() {
                anyhow::bail!("Coin {} does not belong to the contract AND is not in your wallet.", hex::encode(coin.coin_id));
            }
        }
    }

    // Because eUTXO allows manual input selection, we must ensure the user 
    // didn't accidentally leave a sibling UTXO behind, which would lead to WOTS key reuse.
    let input_set: std::collections::HashSet<[u8; 32]> = spending_coins.iter().map(|c| c.coin_id).collect();
    for coin in &spending_coins {
        // Only standard WOTS coins have siblings (MSS handles reuse safely)
        let siblings = wallet.wots_siblings(&coin.coin_id);
        for sib_id in siblings {
            if !input_set.contains(&sib_id) {
                anyhow::bail!(
                    "WOTS co-spend violation! You are trying to spend coin {}, but its sibling {} at the same address is missing from your inputs.\n\
                     To prevent catastrophic WOTS private key reuse, you must include ALL siblings in the transaction.\n\
                     Fix: Add `--coin {}` to your command.",
                    hex::encode(coin.coin_id), hex::encode(sib_id), hex::encode(sib_id)
                );
            }
        }
    }

    if to_args.is_empty() && burn_data.is_none() { 
        anyhow::bail!("Must specify at least one output via --to or --burn-data"); 
    }

    let in_sum: u64 = spending_coins.iter().map(|c| c.value).sum();
    let mut outputs = Vec::new();
    let mut out_sum = 0u64;
    
    for arg in &to_args {
        let salt: [u8; 32] = rand::random();
        match parse_output_spec(arg)? {
            ParsedOutput::Standard(addr, val) => {
                outputs.push(midstate::core::OutputData::Standard { address: addr, value: val, salt });
                out_sum += val;
            }
            ParsedOutput::Stateful(addr, state) => {
                outputs.push(midstate::core::OutputData::Confidential { address: addr, commitment: state, salt });
            }
        }
    }

    if let Some(burn_str) = burn_data {
        let parts: Vec<&str> = burn_str.splitn(2, ':').collect();
        if parts.len() != 2 { anyhow::bail!("Format: <hex_payload>:<value>"); }
        let payload = hex::decode(parts[0]).context("Invalid burn payload hex")?;
        let val: u64 = parts[1].parse().context("Invalid burn value")?;
        outputs.push(midstate::core::OutputData::DataBurn { payload, value_burned: val });
        out_sum += val;
    }

    if in_sum <= out_sum {
        anyhow::bail!("Total input value ({}) must exceed output value ({}) to pay fee", in_sum, out_sum);
    }

    // Auto-change generation
    let base_fee = 100;
    let mut change_seeds = Vec::new();
    if in_sum > out_sum + base_fee {
        let change_val = in_sum - out_sum - base_fee;
        for denom in midstate::core::decompose_value(change_val) {
            let seed = wallet.allocate_next_wots_seed()?;
            let pk = midstate::core::wots::keygen(&seed);
            let addr = midstate::core::compute_address(&pk);
            let salt: [u8; 32] = rand::random();
            let idx = outputs.len();
            outputs.push(midstate::core::OutputData::Standard { address: addr, value: denom, salt });
            change_seeds.push((idx, seed));
        }
        println!("Auto-generated {} change output(s)", change_seeds.len());
    }

    let input_coin_ids: Vec<[u8; 32]> = spending_coins.iter().map(|c| c.coin_id).collect();
    let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
    let tx_salt: [u8; 32] = rand::random();
    let commitment = midstate::core::compute_commitment(&input_coin_ids, &output_commit_hashes, &tx_salt);

    println!("Submitting Phase 1 Commit...");
    submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;

    if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), timeout_secs).await {
        anyhow::bail!("Timed out waiting for Commit to be mined.");
    }
    println!("✓ Commit mined!");

    let mut rpc_inputs = Vec::new();
    let mut rpc_signatures = Vec::new();

    // Map each selected coin to its corresponding witness and state stack.
    // NOTE: `input_coin_ids` contains the exact coins the wallet securely selected 
    // during prepare_commit, including any automatically grouped WOTS siblings!
    for coin_id in &input_coin_ids {
        // Fetch the full coin details (could be local or a remote contract coin)
        let coin = spending_coins.iter().find(|c| c.coin_id == *coin_id)
            .cloned()
            .unwrap_or_else(|| wallet.find_coin(coin_id).unwrap().clone());
            
        // Match the CLI inputs to the coins. 
        // Auto-selected siblings won't have a CLI argument, so they default to AUTO
        let cli_index = spending_coins.iter().position(|c| c.coin_id == *coin_id);
        let wit_arg = cli_index
            .and_then(|idx| inputs_args.get(idx))
            .map(|s| s.as_str())
            .unwrap_or("AUTO:none"); // Siblings are auto-signed
        
        let mut stack_items = Vec::new();
        for token in wit_arg.split(',').filter(|s| !s.is_empty()) {
            let token = token.trim();
            if token.starts_with("AUTO:") {
                // If it's a contract coin without an explicit key, skip auto-signing.
                // (Contracts handle their own logic). But if it's a local wallet coin, sign it!
                if let Some(pk) = token.strip_prefix("AUTO:").filter(|s| !s.is_empty() && *s != "none") {
                    let pk_bytes = parse_hex32(pk)?;
                    stack_items.push(wallet.auto_sign(&pk_bytes, &commitment)?);
                } else if wallet.find_coin(coin_id).is_some() {
                    // It's a local coin (like an auto-selected sibling). Sign it!
                    stack_items.push(wallet.auto_sign(&coin.owner_pk, &commitment)?);
                }
            } else {
                stack_items.push(hex::decode(token).context("Invalid hex in --inputs")?);
            }
        }
        rpc_signatures.push(stack_items.iter().map(hex::encode).collect::<Vec<_>>().join(","));

        // Match input states to the respective coins
        let state_hex = cli_index
            .and_then(|idx| input_states.get(idx))
            .filter(|s| !s.is_empty() && *s != "none")
            .cloned()
            .or_else(|| coin.commitment.map(hex::encode));

        // THE FIX: Differentiate between the Contract's Coin and Personal Wallet Coins
        let input_bytecode = if coin.address == script_address {
            bytecode_hex.clone()
        } else {
            hex::encode(midstate::core::script::compile_p2pk(&coin.owner_pk))
        };

        rpc_inputs.push(rpc::InputRevealJson {
            bytecode: input_bytecode,
            value: coin.value,
            salt: hex::encode(coin.salt),
            commitment: state_hex,
        });
    }

    let reveal_req = rpc::SendTransactionRequest {
        inputs: rpc_inputs,
        signatures: rpc_signatures,
        outputs: outputs.iter().map(|o| match o {
            midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
        }).collect(),
        salt: hex::encode(tx_salt),
        is_consolidate: false,
    };
    
    println!("Submitting Phase 2 Reveal...");
    let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
    let resp = client.post(&reveal_url).json(&reveal_req).send().await?;
    if !resp.status().is_success() {
        let err: rpc::ErrorResponse = resp.json().await?;
        anyhow::bail!("Reveal failed: {}", err.error);
    }

    wallet.data.coins.retain(|c| !input_coin_ids.contains(&c.coin_id));
    
    for (idx, seed) in change_seeds {
        if let midstate::core::OutputData::Standard { address, value, salt } = &outputs[idx] {
            let owner_pk = midstate::core::wots::keygen(&seed);
            wallet.data.coins.push(midstate::wallet::WalletCoin {
                seed, owner_pk, address: *address, value: *value, salt: *salt,
                coin_id: outputs[idx].coin_id().unwrap(),
                label: Some(format!("change ({})", value)),
                wots_signed: false, commitment: None,
            });
        }
    }
    wallet.save()?;

    println!("✓ Custom script spent successfully!");
    Ok(())
}

fn wallet_abandon(path: &PathBuf, address_hex: String) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    
    let addr = midstate::core::types::parse_address_flexible(&address_hex)
        .map_err(|e| anyhow::anyhow!(e))?;
        
    let removed = wallet.abandon_coins_at_address(&addr)?;
    if removed > 0 {
        println!("Successfully abandoned {} coin(s) at address {}", removed, address_hex);
    } else {
        println!("No coins found at address {}", address_hex);
    }
    Ok(())
}

async fn handle_wallet(action: WalletAction) -> Result<()> {
    match action {
        WalletAction::Create { path, legacy } => wallet_create(&path, legacy),
        WalletAction::Restore { path, phrase, rpc_port, rpc_host } => wallet_restore(&path, phrase, rpc_port, rpc_host).await,
        WalletAction::Receive { path, label } => wallet_receive(&path, label),
        WalletAction::Compile { file } => wallet_compile(&file),
        WalletAction::Generate { path, count, label } => wallet_generate(&path, count, label),
        WalletAction::List { path, rpc_port, rpc_host, full } => wallet_list(&path, rpc_port, rpc_host, full).await,
        WalletAction::Balance { path, rpc_port, rpc_host } => wallet_balance(&path, rpc_port, rpc_host).await,
        WalletAction::Scan { path, rpc_port, rpc_host, from_genesis } => {
            wallet_scan(&path, rpc_port, rpc_host, from_genesis).await
        }
        WalletAction::Send { path, rpc_port, rpc_host, coin, to, timeout, private } => {
            wallet_send(&path, rpc_port, rpc_host, coin, to, timeout, private).await
        }
        WalletAction::SpendScript { path, rpc_port, rpc_host, coin, bytecode, inputs, input_state, burn_data, to, timeout } => {
            wallet_spend_script(&path, rpc_port, rpc_host, coin, bytecode, inputs, input_state, burn_data, to, timeout).await
        }
        WalletAction::Import { path, seed, value, salt, label } => {
            wallet_import(&path, &seed, value, &salt, label)
        }
        WalletAction::Export { path, coin } => wallet_export(&path, &coin),
        WalletAction::Pending { path } => wallet_pending(&path),
        WalletAction::Consolidate { path, rpc_port, rpc_host, address } => {
            wallet_consolidate(&path, rpc_port, rpc_host, address).await
        }
        WalletAction::Reveal { path, rpc_port, rpc_host, commitment } => {
            wallet_reveal(&path, rpc_port, rpc_host, commitment).await
        }
        WalletAction::Abandon { path, address } => {
            wallet_abandon(&path, address)
        }
        WalletAction::History { path, count } => wallet_history(&path, count),
        WalletAction::ImportRewards { path, coinbase_file, data_dir } => {
            wallet_import_rewards(&path, &coinbase_file, &data_dir)
        }
        WalletAction::GenerateMss { path, height, label } => {
            wallet_generate_mss(&path, height, label)
        }
        WalletAction::Mix { path, rpc_port, rpc_host, denomination, coin, join, pay_fee, timeout } => {
            wallet_mix(&path, rpc_port, rpc_host, denomination, coin, join, pay_fee, timeout, None).await
        }
        WalletAction::AutoMix { path, rpc_port, rpc_host, coin, timeout } => {
            wallet_automix(&path, rpc_port, rpc_host, coin, timeout).await
        }
        // Phase 1+2 license issuance (wired to PoAW bundles)
        WalletAction::IssueLicense { path, bundle, poaw_commitment, fixed_fee, min_height, max_height, archival_weight, rpc_port, rpc_host } => {
            wallet_issue_license(&path, bundle.as_deref(), poaw_commitment.as_deref(), fixed_fee, min_height, max_height, archival_weight, rpc_port, rpc_host).await
        }
        WalletAction::BuyLicense { path, license, price, seller_pk, rpc_port, rpc_host, timeout } => {
            wallet_buy_license(&path, &license, price, &seller_pk, rpc_port, rpc_host, timeout).await
        }
        WalletAction::PoawGenerate { path, start, end, stride, difficulty, issuer, submit_commit, rpc_port, rpc_host } => {
            wallet_poaw_generate(&path, start, end, stride, difficulty, &issuer, submit_commit, rpc_port, &rpc_host).await
        }
        WalletAction::RekeyLicense { path, old, new_coin_id } => {
            wallet_rekey_license(&path, &old, &new_coin_id).await
        }
        WalletAction::ListLicenses { path } => {
            wallet_list_licenses(&path).await
        }
        WalletAction::SellLicense { path, license, secret, buyer_pk, rpc_port, rpc_host } => {
            wallet_sell_license(&path, &license, secret.as_deref(), buyer_pk.as_deref(), rpc_port, &rpc_host).await
        }
        WalletAction::RecoverLicenseMetadata { path, tx, height, rpc_port, rpc_host } => {
            wallet_recover_license_metadata(&path, &tx, height, rpc_port, &rpc_host).await
        }
    }
}

fn wallet_create(path: &PathBuf, legacy: bool) -> Result<()> {
    let password = read_password_confirm()?;

    if legacy {
        Wallet::create(path, &password)?;
        println!("Legacy wallet created: {}", path.display());
    } else {
        let (_wallet, phrase) = Wallet::create_hd(path, &password)?;
        println!();
        println!("=================================================================");
        println!("  WALLET CREATED SUCCESSFULLY");
        println!("  WRITE DOWN THESE 24 WORDS. THIS IS YOUR ONLY BACKUP.");
        println!("  If you lose this phrase, your funds are UNRECOVERABLE.");
        println!("-----------------------------------------------------------------");
        println!("  {}", phrase);
        println!("=================================================================");
        println!();
        println!("Wallet saved to: {}", path.display());
    }
    Ok(())
}

async fn wallet_restore(path: &PathBuf, phrase_arg: Option<String>, rpc_port: u16, rpc_host: String) -> Result<()> {
    let phrase = match phrase_arg {
        Some(p) => p,
        None => {
            print!("Enter your 24-word seed phrase: ");
            std::io::Write::flush(&mut std::io::stdout())?;
            let mut input = String::new();
            std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut input)?;
            input.trim().to_string()
        }
    };

    // Validate the phrase before asking for a password
    midstate::wallet::hd::master_seed_from_mnemonic(&phrase)?;

    let password = read_password_confirm()?;
    let mut wallet = Wallet::restore_from_mnemonic(path, &password, &phrase)?;

    println!("Wallet restored from seed phrase.");
    println!("Saved to: {}", path.display());
    println!("\nCRITICAL WARNING: Run `midstate wallet scan` before sending any transactions.");
    println!("Spending from an unscanned wallet may result in the permanent loss of sibling UTXOs.");
    println!();

    // Gap-limit scan: generate keys in batches, scan the chain, repeat until
    // a full window of GAP_LIMIT consecutive unused keys is found.
    println!("Starting chain scan to rediscover coins...");
    let client = reqwest::Client::new();
    let base_url = format!("http://{}:{}", rpc_host, rpc_port);
    
    let state: rpc::GetStateResponse = client.get(format!("{}/state", base_url))
        .send().await?.json().await?;
    let chain_height = state.height;

    if chain_height == 0 {
        println!("Chain is empty — nothing to scan. Generate keys as you go.");
        return Ok(());
    }

    const GAP_LIMIT: u64 = 20;
    const BATCH_SIZE: u64 = 50;
    let mut total_found = 0usize;
    let mut consecutive_empty = 0u64;
    // Force the wallet to derive your first MSS key so it gets added to watched_addresses()
    wallet.generate_mss(10, Some("Recovered MSS".to_string()))?; 

    loop {
        // Generate a batch of keys
        let idx_before = wallet.wots_index();
        wallet.restore_generate_keys(BATCH_SIZE)?;
        let addresses = wallet.watched_addresses();

        println!(
            "  Scanning with {} addresses (WOTS index {}..{})...",
            addresses.len(), idx_before, wallet.wots_index()
        );

        // Use compact filters to find which blocks might contain our addresses
        let matching_heights = filter_scan(&client, &base_url, &addresses, 0, chain_height).await?;

        if matching_heights.is_empty() {
            consecutive_empty += BATCH_SIZE;
        } else {
            // Do targeted full scans only on matching blocks
            let found = targeted_scan(&client, &base_url, &mut wallet, &addresses, &matching_heights).await?;
            if found > 0 {
                total_found += found;
                consecutive_empty = 0;
            } else {
                consecutive_empty += BATCH_SIZE;
            }
        }

        if consecutive_empty >= GAP_LIMIT {
            break;
        }
    }

    wallet.data.last_scan_height = chain_height;
    wallet.save()?;

    println!();
    println!("Restore complete. {} coin(s) recovered. Scanned to height {}.", total_found, chain_height);
    if wallet.total_value() > 0 {
        println!("Total balance: {}", wallet.total_value());
    }
    Ok(())
}
fn wallet_compile(path: &std::path::Path) -> Result<()> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read script file: {}", path.display()))?;
    
    match midstate::core::script::assemble(&source) {
        Ok(bytecode) => {
            let address = midstate::core::types::hash(&bytecode);
            println!("Compilation Successful!\n");
            println!("Bytecode (hex): {}", hex::encode(&bytecode));
            println!("Size:           {} bytes", bytecode.len());
            println!("Address:        {}", hex::encode(address));
        }
        Err(e) => {
            anyhow::bail!("Compilation failed: {}", e);
        }
    }
    Ok(())
}
fn wallet_receive(path: &PathBuf, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let label = label.unwrap_or_else(|| format!("receive #{}", wallet.keys().len() + 1));
    let address = wallet.generate_key(Some(label.clone()))?;
    println!("\n  Your receiving address ({}):\n", label);
    println!("  {}\n", midstate::core::types::encode_address_with_checksum(&address));
    println!("  Share this with the sender.");
    Ok(())
}

fn wallet_generate(path: &PathBuf, count: usize, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    for i in 0..count {
        let lbl = if count == 1 {
            label.clone()
        } else {
            label.as_ref().map(|l| format!("{} #{}", l, i + 1))
        };
        let pk = wallet.generate_key(lbl)?;
        println!("  [{}] {}", wallet.keys().len() - 1, midstate::core::types::encode_address_with_checksum(&pk));
        
    }
    println!("\nGenerated {} key(s). Total keys: {}, Total coins: {}",
        count, wallet.keys().len(), wallet.coin_count());
    Ok(())
}

async fn wallet_list(path: &PathBuf, rpc_port: u16, rpc_host: String, full: bool) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    if wallet.coin_count() > 0 || !wallet.list_licenses().is_empty() {
        println!("COINS:");
        if full {
            println!("{:<5} {:<66} {:<8} {:<10} {}", "#", "COIN_ID", "VALUE", "STATUS", "LABEL");
            println!("{}", "-".repeat(100));
        } else {
            println!("{:<5} {:<15} {:<8} {:<10} {}", "#", "COIN_ID", "VALUE", "STATUS", "LABEL");
            println!("{}", "-".repeat(55));
        }

        for (i, wc) in wallet.coins().iter().enumerate() {
            let coin_hex = hex::encode(wc.coin_id);
            let status = check_coin_rpc(&client, rpc_port, &rpc_host, &coin_hex).await;
            let label = wc.label.as_deref().unwrap_or("");
            let status_str = match status {
                Ok(true) => "✓ live",
                Ok(false) => "✗ unset",
                Err(_) => "? error",
            };
            let display = if full { coin_hex } else { hex::encode(&wc.coin_id) };
            println!("{:<5} {:<15} {:<8} {:<10} {}", i, display, wc.value, status_str, label);
        }
    }

    // License summary (polish for v1)
    let licenses = wallet.list_licenses();
    if !licenses.is_empty() {
        println!("\nPRUNING LICENSES: {} held", licenses.len());
        if full {
            for (i, (_key, meta)) in licenses.iter().enumerate() {
                println!("  [{}] fixed_fee={} weight={} range={}-{}",
                    i, meta.fixed_royalty_fee, meta.archival_weight, meta.min_height, meta.max_height);
            }
        }
    }

if !wallet.keys().is_empty() {
        println!("\nUNUSED RECEIVING KEYS:");
        for (i, k) in wallet.keys().iter().enumerate() {
            let display = if full { midstate::core::types::encode_address_with_checksum(&k.address) } else { hex::encode(&k.address) };
            let label = k.label.as_deref().unwrap_or("");
            println!("  [K{}] {} {}", i, display, label);
        }
    }

    if !full { println!("\nUse --full to show complete IDs."); }
    Ok(())
}

async fn wallet_balance(path: &PathBuf, rpc_port: u16, rpc_host: String) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let mut live_count = 0usize;
    let mut live_value = 0u64;
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_count += 1;
            live_value += wc.value;
        }
    }

    println!("Coins in wallet:  {}", wallet.coin_count());
    println!("Live on-chain:    {} (value: {})", live_count, live_value);
    println!("Unused keys:      {}", wallet.keys().len());
    println!("Pending commits:  {}", wallet.pending().len());
    Ok(())
}

async fn wallet_consolidate(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    address_str: String,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let target_addr = midstate::core::types::parse_address_flexible(&address_str)
        .map_err(|e| anyhow::anyhow!(e))?;

    // Find all live coins at this address
    let mut live_coins = Vec::new();
    let mut total_val = 0u64;
    for wc in wallet.coins() {
        if wc.address == target_addr {
            if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
                live_coins.push(wc.coin_id);
                total_val += wc.value;
            }
        }
    }

    if live_coins.len() < 2 {
        anyhow::bail!("Address only has {} live UTXOs. Consolidation is only for grouped sibling UTXOs.", live_coins.len());
    }
    
    if live_coins.len() > midstate::core::MAX_CONSOLIDATE_INPUTS {
        anyhow::bail!("Too many UTXOs to consolidate in one transaction ({} > {}). Wait for next update for multi-pass.", live_coins.len(), midstate::core::MAX_CONSOLIDATE_INPUTS);
    }

    println!("Found {} live UTXOs at this address totaling {} value.", live_coins.len(), total_val);

    // Create a fresh MSS tree to hold the consolidated funds safely
    let new_mss_pk = wallet.generate_mss(10, Some("Consolidated Sweeper".into()))?;
    
    // ~85 bytes per InputReveal (predicate + value + salt + overhead)
    let estimated_bytes = 100 + 1636 + 100 + (live_coins.len() as u64 * 85);
    let fee = (estimated_bytes * 10) / 1024 + 10;
    
    if total_val <= fee {
        anyhow::bail!("Total value {} is too low to pay the network fee of {}", total_val, fee);
    }

    let out_val = total_val - fee;
    let mut outputs = Vec::new();
    let change_seeds = Vec::new();
    
    // Break the output value into power-of-2 denominations
    for denom in midstate::core::decompose_value(out_val) {
        let salt: [u8; 32] = rand::random();
        outputs.push(midstate::core::OutputData::Standard { address: new_mss_pk, value: denom, salt });
    }

    // Pass `true` as the final parameter to prepare_commit to indicate this is a Consolidate tx
    let (commitment, _salt) = wallet.prepare_commit(&live_coins, &outputs, change_seeds, false, true)?;

    println!("Submitting Phase 1: Consolidate Commit...");
    submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;

    if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), 120).await {
        anyhow::bail!("Timed out waiting for Commit to be mined.");
    }
    println!("✓ Commit mined!");

    let pending = wallet.find_pending(&commitment).unwrap().clone();
    let (input_reveals, witness) = wallet.sign_consolidate(&pending)?;
    
    let mut witnesses = Vec::new();
    let midstate::core::types::Witness::ScriptInputs(inputs) = witness;
    witnesses.push(inputs.iter().map(hex::encode).collect::<Vec<_>>().join(","));

    let reveal_req = rpc::SendTransactionRequest {
        inputs: input_reveals.iter().map(|ir| rpc::InputRevealJson {
            bytecode: match &ir.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
            value: ir.value,
            salt: hex::encode(ir.salt),
            commitment: None,
        }).collect(),
        signatures: witnesses,
        outputs: outputs.iter().map(|o| match o {
            midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            _ => unreachable!(),
        }).collect(),
        salt: hex::encode(pending.salt),
        is_consolidate: true, // Tell the RPC server to construct a Consolidate tx
    };
    
    println!("Submitting Phase 2: Sweep Reveal...");
    let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
    let resp = client.post(&reveal_url).json(&reveal_req).send().await?;
    if !resp.status().is_success() {
        let err: rpc::ErrorResponse = resp.json().await?;
        anyhow::bail!("Reveal failed: {}", err.error);
    }
    
    wallet.complete_reveal(&commitment)?;
    println!("✓ Dust swept successfully into {} new UTXOs!", outputs.len());
    
    Ok(())
}

async fn wallet_send(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    coin_args: Vec<String>,
    to_args: Vec<String>,
    timeout_secs: u64,
    private: bool,
) -> Result<()> {
    if to_args.is_empty() {
        anyhow::bail!("must specify at least one --to <owner_pk>:<value>");
    }

    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    // MSS Index Verification
    if !wallet.data.mss_keys.is_empty() {
        println!("Connecting to node to verify MSS safety indices...");
        let mss_url = format!("http://{}:{}/mss_state", rpc_host, rpc_port);

        for i in 0..wallet.data.mss_keys.len() {
            let master_pk = wallet.data.mss_keys[i].master_pk;
            let current_leaf = wallet.data.mss_keys[i].next_leaf;

            let req = rpc::GetMssStateRequest {
                master_pk: hex::encode(master_pk),
            };

            // STRICT SAFETY: We use context() to ensure that if the node is offline,
            // the program crashes here rather than risking a reuse of the private key.
            let response = client.post(&mss_url).json(&req).send().await
                .context("CRITICAL: Could not connect to node. Aborting to prevent MSS key reuse.")?;

            if !response.status().is_success() {
                anyhow::bail!("Safety Check Failed: Node returned error checking MSS state.");
            }

            let mss_resp: rpc::GetMssStateResponse = response.json().await
                .context("Safety Check Failed: Invalid response from node.")?;

            // If the node has seen more signatures than we have locally, FAST FORWARD.
            if mss_resp.next_index > current_leaf {
                const SAFETY_MARGIN: u64 = 20;
                let new_leaf = mss_resp.next_index + SAFETY_MARGIN;
                
                println!("  ⚠️  MSS Key {}: Old state detected (Node: {}, Local: {})", 
                    hex::encode(&master_pk), mss_resp.next_index, current_leaf);
                println!("      Fast-forwarding index to {} to ensure safety.", new_leaf);
                
                wallet.data.mss_keys[i].set_next_leaf(new_leaf);
                
                // Save immediately. If save fails, we crash before signing.
                wallet.save().context("Failed to save updated wallet state")?;
            }
        }
        println!("  ✓ MSS indices verified safe.");
    }


    let recipient_specs: Vec<ParsedOutput> = to_args.iter()
        .map(|s| parse_output_spec(s))
        .collect::<Result<Vec<_>>>()?;

    let total_send: u64 = recipient_specs.iter().map(|o| match o {
        ParsedOutput::Standard(_, v) => *v,
        ParsedOutput::Stateful(_, _) => 0,
    }).sum();

    let mut live_coins = Vec::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_coins.push(wc.coin_id);
        }
    }

    if private {
        // Find standard outputs for private send logic
        let denoms: Vec<u64> = recipient_specs.iter().filter_map(|o| match o {
            ParsedOutput::Standard(_, v) => Some(*v),
            ParsedOutput::Stateful(_, _) => None,
        }).collect();
        
        let recipient_address = match &recipient_specs[0] {
            ParsedOutput::Standard(a, _) => *a,
            ParsedOutput::Stateful(a, _) => *a,
        };
        
        let pairs = wallet.plan_private_send(&live_coins, &recipient_address, &denoms)?;
        println!("Private send: {} independent transaction(s)\n", pairs.len());

        for (pair_idx, (inputs, outputs, change_seeds)) in pairs.iter().enumerate() {
            let in_val: u64 = inputs.iter()
                .filter_map(|id| wallet.find_coin(id))
                .map(|c| c.value)
                .sum();
            let out_val: u64 = outputs.iter().map(|o| o.value()).sum();
            println!("  Pair {}: {} in (value {}) → {} out (value {}, fee {})",
                pair_idx, inputs.len(), in_val, outputs.len(), out_val, in_val - out_val);

            let (commitment, _salt) = wallet.prepare_commit(
                inputs, outputs, change_seeds.clone(), true, false
            )?;

            if let Err(e) = submit_commit(&client, rpc_port, &rpc_host, &commitment).await {
                println!("  Pair {} commit failed: {}", pair_idx, e);
                continue;
            }

            println!("  ✓ Commit submitted ({})", hex::encode(&commitment));

            if !wait_for_commit_mined(&client, rpc_port,&rpc_host, &hex::encode(commitment), timeout_secs).await {
                println!("  ⏳ Not mined yet. Run `wallet reveal` later.");
                continue;
            }

            let pending = wallet.find_pending(&commitment).unwrap().clone();
            let delay = pending.reveal_not_before.saturating_sub(now_secs());
            if delay > 0 {
                println!("  Waiting {}s (privacy delay)...", delay);
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            do_reveal(&client, &mut wallet, rpc_port, &rpc_host, &commitment, timeout_secs).await?;
        }
    } else {
        let mut target_fee = 100u64; // Start with a conservative minimum guess
        let  input_coin_ids;
        let mut all_outputs;
        let mut change_seeds;
        let  in_sum;
        let final_fee;

        loop {
            // Dynamically calculate what we need based on the current fee guess
            let needed = total_send + target_fee;
            
            let selected = if !coin_args.is_empty() {
                coin_args.iter()
                    .map(|s| wallet.resolve_coin(s))
                    .collect::<Result<Vec<_>>>()?
            } else {
                wallet.select_coins(needed, &live_coins)?
            };

            let current_in_sum: u64 = selected.iter()
                .filter_map(|id| wallet.find_coin(id))
                .map(|c| c.value)
                .sum();

            if current_in_sum <= total_send {
                anyhow::bail!("input value ({}) must exceed output value ({}) to pay fee", current_in_sum, total_send);
            }

            // Estimate the number of outputs that will be created
            let change = current_in_sum.saturating_sub(total_send + target_fee);
            let mut num_outputs = 0;
            for spec in &recipient_specs {
                match spec {
                    ParsedOutput::Standard(_, value) => num_outputs += decompose_value(*value).len(),
                    ParsedOutput::Stateful(_, _) => num_outputs += 1,
                }
            }
            num_outputs += decompose_value(change).len();
            
            // Calculate a strict upper bound for serialized byte size
            // 1536 (WOTS sig) + 100 (input struct) = 1636 bytes per input
            // 100 bytes per output + 100 bytes transaction base overhead
            let estimated_bytes = 100 + (selected.len() as u64 * 1636) + (num_outputs as u64 * 100);
            
            // Calculate required fee based on Mempool limit (10 units per 1024 bytes)
            let required_fee = (estimated_bytes * 10) / 1024 + 10; // +10 unit safety padding
            
            if current_in_sum >= total_send + required_fee {
                // The selected coins cover the send AND the exact calculated fee! Lock it in.
                final_fee = required_fee;
                in_sum = current_in_sum;
                input_coin_ids = selected;
                
                let final_change = current_in_sum - total_send - final_fee;
                all_outputs = Vec::new();
                change_seeds = Vec::new();

                // 1. Build recipient outputs
                for spec in &recipient_specs {
                    match spec {
                        ParsedOutput::Standard(address, value) => {
                            for denom in midstate::core::decompose_value(*value) {
                                let salt: [u8; 32] = rand::random();
                                all_outputs.push(midstate::core::OutputData::Standard { address: *address, value: denom, salt });
                            }
                        }
                        ParsedOutput::Stateful(address, state) => {
                            let salt: [u8; 32] = rand::random();
                            all_outputs.push(midstate::core::OutputData::Confidential { address: *address, commitment: *state, salt });
                        }
                    }
                }

                // 2. Build exact change outputs
                if final_change > 0 {
                    for denom in midstate::core::decompose_value(final_change) {
                        let seed = wallet.allocate_next_wots_seed()?;
                        let pk = midstate::core::wots::keygen(&seed);
                        let addr = midstate::core::compute_address(&pk);
                        let salt: [u8; 32] = rand::random();
                        let idx = all_outputs.len();
                        all_outputs.push(midstate::core::OutputData::Standard { address: addr, value: denom, salt });
                        change_seeds.push((idx, seed));
                    }
                }

                // 3. Shuffle outputs to obfuscate which is the recipient vs change
                {
                    use rand::seq::SliceRandom;
                    let mut indices: Vec<usize> = (0..all_outputs.len()).collect();
                    indices.shuffle(&mut rand::thread_rng());
                    let shuffled: Vec<midstate::core::OutputData> = indices.iter().map(|&i| all_outputs[i].clone()).collect();
                    let mut rev = vec![0usize; indices.len()];
                    for (new_i, &old_i) in indices.iter().enumerate() { rev[old_i] = new_i; }
                    change_seeds = change_seeds.into_iter()
                        .map(|(old_idx, s)| (rev[old_idx], s)).collect();
                    all_outputs = shuffled;
                }
                
                break;
            } else {
                // We need more inputs to cover the real fee. Let's loop again with the new target!
                target_fee = required_fee;
                
                // If the user manually provided specific coins via CLI args, we can't auto-select more.
                if !coin_args.is_empty() {
                    anyhow::bail!(
                        "The manually selected coins do not cover the transaction amount plus the dynamically calculated fee of {}", 
                        required_fee
                    );
                }
            }
        }

        println!(
            "Spending {} coin(s) (value {}) → {} output(s) (value {}, fee: {})",
            input_coin_ids.len(), in_sum,
            all_outputs.len(), in_sum - final_fee,
            final_fee
        );

let (commitment, _salt) = wallet.prepare_commit(
            &input_coin_ids, &all_outputs, change_seeds.clone(), false, false
        )?;

        submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;


        println!("\n✓ Commit submitted ({})", hex::encode(&commitment));
        println!("  Waiting for commit to be mined...");

        if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), timeout_secs).await {
            println!("⏳ Not mined after {}s. Run `wallet reveal` later.", timeout_secs);
            return Ok(());
        }
        println!("✓ Commit mined!");

        do_reveal(&client, &mut wallet, rpc_port, &rpc_host, &commitment, timeout_secs).await?;
    }

    Ok(())
}



async fn fetch_state_info(client: &reqwest::Client, rpc_port: u16, rpc_host: &str) -> Result<(u32, u64, [u8; 32])> {
    let url = format!("http://{}:{}/state", rpc_host, rpc_port);
    let resp = client.get(&url).send().await?;
    let state: rpc::GetStateResponse = resp.json().await?;
    let mut hh = [0u8; 32];
    hex::decode_to_slice(&state.header_hash, &mut hh)?;
    Ok((state.required_pow, state.height, hh))
}

/// Best-effort fetch of the historical block hash used for PoAW at a given height.
/// In production this must return the *real* `batch.extension.final_hash` (or header hash)
/// from a trusted synced archival node so that the PoAW actually proves possession of that history.
/// Fetches the historical block and hashes the heavy transaction payload for PoAW.
/// This prevents "Header-Only" archivers from minting licenses without the real data.
async fn fetch_payload_hash_for_poaw(
    client: &reqwest::Client,
    rpc_port: u16,
    rpc_host: &str,
    height: u64,
) -> Result<[u8; 32]> {
    let block_url = format!("http://{}:{}/block/{}", rpc_host, rpc_port, height);
    let resp = client.get(&block_url).send().await?;
    if !resp.status().is_success() {
        bail!("Failed to fetch block {}", height);
    }
    let batch: midstate::core::Batch = resp.json().await?;
    
    let mut payload_hasher = blake3::Hasher::new();
    for tx in &batch.transactions {
        if let Ok(tx_bytes) = bincode::serialize(tx) {
            payload_hasher.update(&tx_bytes);
        }
    }
    Ok(*payload_hasher.finalize().as_bytes())
}

async fn submit_commit(
    client: &reqwest::Client, 
    rpc_port: u16, 
    rpc_host: &str, 
    commitment: &[u8; 32]) -> 
    Result<()> {
    let (required_pow, current_height, header_hash) = fetch_state_info(client, rpc_port, rpc_host).await?;
    println!("Mining PoW locally (difficulty: {} leading zeros, height: {})...", required_pow, current_height);
    let commitment_owned = *commitment;
    let spam_nonce = tokio::task::spawn_blocking(move || {
        midstate::core::transaction::mine_pow(&commitment_owned, required_pow, current_height, header_hash)
    }).await?;
    

    
    
    println!("✓ PoW found (nonce: {})", spam_nonce);

    let commit_req = rpc::CommitRequest {
        commitment: hex::encode(commitment),
        spam_nonce,
    };

    let url = format!("http://{}:{}/commit", rpc_host, rpc_port);
    let response = client.post(&url).json(&commit_req).send().await?;
    if !response.status().is_success() {
        let error: rpc::ErrorResponse = response.json().await?;
        anyhow::bail!("Commit failed: {}", error.error);
    }
    Ok(())
}

async fn wait_for_commit_mined(
    client: &reqwest::Client,
    rpc_port: u16,
    rpc_host: &str, 
    commitment_hex: &str,
    timeout_secs: u64,
) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let check_url = format!("http://{}:{}/check_commitment", rpc_host, rpc_port);
    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;
        // Confirm the commitment is in chain state, not just absent from mempool.
        // A commit can leave the mempool by being evicted or reorg'd out, not just mined.
        // Checking state directly ensures we only reveal when the commitment is spendable.
        let req = rpc::CheckCommitmentRequest { commitment: commitment_hex.to_string() };
        if let Ok(resp) = client.post(&check_url).json(&req).send().await {
            if let Ok(result) = resp.json::<rpc::CheckCommitmentResponse>().await {
                if result.exists {
                    return true;
                }
            }
        }
        eprint!(".");
    }
    eprintln!();
    false
}

async fn do_reveal(
    client: &reqwest::Client,
    wallet: &mut Wallet,
    rpc_port: u16,
    rpc_host: &str,
    commitment: &[u8; 32],
    timeout_secs: u64,
) -> Result<()> {
    let pending = wallet.find_pending(commitment)
        .ok_or_else(|| anyhow::anyhow!("pending commit not found"))?
        .clone();

    let (input_reveals, signatures) = match wallet.sign_reveal(&pending) {
            Ok(res) => res,
            Err(e) => {
                // If the coins are gone, delete this garbage commit to keep the wallet clean
                wallet.data.pending.retain(|p| p.commitment != *commitment);
                let _ = wallet.save();
                anyhow::bail!("Failed to prepare reveal: {}. Stale commit dropped.", e);
            }
        };

    let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
    let reveal_req = rpc::SendTransactionRequest {
        inputs: input_reveals.iter().map(|ir| rpc::InputRevealJson {
            bytecode: match &ir.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
            value: ir.value,
            salt: hex::encode(ir.salt),
            commitment: ir.commitment.map(hex::encode),
        }).collect(),
        signatures: signatures.iter().map(|s| match s {
            midstate::core::types::Witness::ScriptInputs(inputs) => {
                inputs.iter().map(hex::encode).collect::<Vec<_>>().join(",")
            }
        }).collect(),
        outputs: pending.outputs.iter().map(|o| match o {
            midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
        }).collect(),
        salt: hex::encode(pending.salt),
        is_consolidate: pending.is_consolidate,
    };

    let response = client.post(&reveal_url).json(&reveal_req).send().await?;
    if !response.status().is_success() {
        let error: rpc::ErrorResponse = response.json().await?;
        anyhow::bail!("reveal failed: {}", error.error);
    }
    let _result: rpc::SendTransactionResponse = response.json().await?;

    let check_coin_hex = hex::encode(pending.input_coin_ids[0]);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let mut revealed = false;
    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;
        if let Ok(resp) = client
            .post(&format!("http://{}:{}/check",rpc_host, rpc_port))
            .json(&rpc::CheckCoinRequest { coin: check_coin_hex.clone() })
            .send().await
        {
            if let Ok(check) = resp.json::<rpc::CheckCoinResponse>().await {
                if !check.exists { revealed = true; break; }
            }
        }
        eprint!(".");
    }
    eprintln!();

    if !revealed {
        println!("⏳ Reveal submitted but not yet mined.");
        return Ok(());
    }

    wallet.complete_reveal(commitment)?;
    println!("✓ Transfer complete!");
    for id in &pending.input_coin_ids {
        let val = input_reveals.iter().find(|ir| ir.coin_id() == *id).map(|ir| ir.value).unwrap_or(0);
        println!("  spent:   {} (value {})", hex::encode(id), val);
    }
    for out in &pending.outputs {
        if let Some(c_id) = out.coin_id() {
            println!("  created: {} (value {})", hex::encode(&c_id), out.value());
        } else {
            println!("  burned: (value {})", out.value());
        }
    }
    Ok(())
}

// ── CoinJoin Mix ────────────────────────────────────────────────────────────

async fn wallet_mix(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    denomination: u64,
    coin_arg: Option<String>,
    join_mix_id: Option<String>,
    pay_fee: bool,
    timeout_secs: u64,
    password_override: Option<Vec<u8>>,
) -> Result<()> {
    if !denomination.is_power_of_two() || denomination == 0 {
        anyhow::bail!("denomination must be a non-zero power of 2");
    }

    let password = match password_override {
        Some(pw) => pw,
        None => read_password("Password: ")?,
    };
    
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();
    let base_url = format!("http://{}:{}", rpc_host, rpc_port);

    // Find live coins
    let mut live_coins = Vec::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_coins.push(wc.coin_id);
        }
    }

    // Select the coin to mix
    let mix_coin_id: [u8; 32] = if let Some(ref coin_ref) = coin_arg {
        let resolved = wallet.resolve_coin(coin_ref)?;
        if !live_coins.contains(&resolved) {
            anyhow::bail!("coin {} is not live on-chain", hex::encode(&resolved));
        }
        let coin = wallet.find_coin(&resolved)
            .ok_or_else(|| anyhow::anyhow!("coin not in wallet"))?;
        if coin.value != denomination {
            anyhow::bail!(
                "coin {} has value {} but denomination is {}",
                hex::encode(&resolved), coin.value, denomination
            );
        }
        resolved
    } else {
        // Auto-select a coin matching the denomination
        let found = wallet.coins().iter()
            .find(|c| c.value == denomination && live_coins.contains(&c.coin_id))
            .ok_or_else(|| anyhow::anyhow!(
                "no live coin with denomination {} found in wallet", denomination
            ))?;
        found.coin_id
    };

    println!("CoinJoin Mix");
    println!("  Denomination: {}", denomination);
    println!("  Input coin:   {}", hex::encode(&mix_coin_id));

    // Step 1: Create or join a session
    let mix_id_hex: String = if let Some(ref join_hex) = join_mix_id {
        println!("  Joining session: {}", join_hex);
        join_hex.clone()
    } else {
        // Create new session
        let create_req = rpc::MixCreateRequest {
            denomination,
            min_participants: 2,
        };
        let resp = client.post(format!("{}/mix/create", base_url))
            .json(&create_req).send().await?;
        if !resp.status().is_success() {
            let error: rpc::ErrorResponse = resp.json().await?;
            anyhow::bail!("create failed: {}", error.error);
        }
        let create_resp: rpc::MixCreateResponse = resp.json().await?;
        println!("  Created session: {}", &create_resp.mix_id[..16]);
        println!("\n  Share this mix_id with other participants:");
        println!("  {}\n", create_resp.mix_id);
        create_resp.mix_id
    };

    // Step 2: Register our input/output
    let (input, output, output_seed) = wallet.prepare_mix_registration(&mix_coin_id)?;
    
    // --- Validate the mix_id formatting ---
    let _parsed_mix_id = parse_hex32(&mix_id_hex)?;
    
    // Do NOT sign the mix_id! Sending a WOTS signature here reuses the key
    // and instantly compromises the user's funds. The node ignores this field anyway.
    let join_sig = vec![]; 
    // -----------------------------------------------

    let register_req = rpc::MixRegisterRequest {
        mix_id: mix_id_hex.clone(),
        coin_id: hex::encode(mix_coin_id),
        input: rpc::InputRevealJson {
            bytecode: match &input.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
            value: input.value,
            salt: hex::encode(input.salt),
            commitment: input.commitment.map(hex::encode),
        },
        output: match &output {
            midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
        },
        signature: hex::encode(join_sig),
    };

    let resp = client.post(format!("{}/mix/register", base_url))
        .json(&register_req).send().await?;
    if !resp.status().is_success() {
        let error: rpc::ErrorResponse = resp.json().await?;
        anyhow::bail!("register failed: {}", error.error);
    }
    println!("  ✓ Registered in mix session");

    // Step 3: Optionally pay the fee
    let mut fee_coin_id: Option<[u8; 32]> = None;
    if pay_fee {
        match wallet.prepare_mix_fee(&live_coins) {
            Ok((fee_input, fee_cid)) => {
                let fee_req = rpc::MixFeeRequest {
                    mix_id: mix_id_hex.clone(),
                    input: rpc::InputRevealJson {
                        bytecode: match &fee_input.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
                        value: fee_input.value,
                        salt: hex::encode(fee_input.salt),
                        commitment: fee_input.commitment.map(hex::encode),
                    },
                };
                let resp = client.post(format!("{}/mix/fee", base_url))
                    .json(&fee_req).send().await?;
                if !resp.status().is_success() {
                    let error: rpc::ErrorResponse = resp.json().await?;
                    anyhow::bail!("fee failed: {}", error.error);
                }
                println!("  ✓ Fee coin registered ({})", hex::encode(&fee_cid));
                fee_coin_id = Some(fee_cid);
            }
            Err(e) => {
                println!("  ⚠ Cannot pay fee: {}", e);
                println!("    Another participant must contribute a denomination-1 coin.");
            }
        }
    }

    // Step 4: Wait for signing phase
    println!("\n  Waiting for all participants to register...");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let mut commitment_hex = String::new();

    loop {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for signing phase");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        let status_url = format!("{}/mix/status/{}", base_url, mix_id_hex);
        let resp = client.get(&status_url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("mix session not found");
        }
        let status: rpc::MixStatusResponse = resp.json().await?;

        match status.phase.as_str() {
            "collecting" => { eprint!("."); }
            "signing" => {
                eprintln!();
                commitment_hex = status.commitment.unwrap_or_default();
                println!("  ✓ All participants registered! Signing phase.");
                break;
            }
            "failed" => {
                anyhow::bail!("mix session failed");
            }
            other => {
                println!("  Unexpected phase: {}", other);
                break;
            }
        }
    }

    // Step 5: Sign our input(s)
    let commitment = parse_hex32(&commitment_hex)?;

    // Sign the mix input
    let sig = wallet.sign_mix_input(&mix_coin_id, &commitment)?;
    let sign_req = rpc::MixSignRequest {
        mix_id: mix_id_hex.clone(),
        coin_id: hex::encode(mix_coin_id),
        signature: hex::encode(&sig),
    };
    let resp = client.post(format!("{}/mix/sign", base_url))
        .json(&sign_req).send().await?;
    if !resp.status().is_success() {
        let error: rpc::ErrorResponse = resp.json().await?;
        anyhow::bail!("sign failed: {}", error.error);
    }
    println!("  ✓ Signed mix input");

    // Sign fee input if we're paying
    if let Some(fee_cid) = fee_coin_id {
        let fee_sig = wallet.sign_mix_input(&fee_cid, &commitment)?;
        let fee_sign_req = rpc::MixSignRequest {
            mix_id: mix_id_hex.clone(),
            coin_id: hex::encode(fee_cid),
            signature: hex::encode(&fee_sig),
        };
        let resp = client.post(format!("{}/mix/sign", base_url))
            .json(&fee_sign_req).send().await?;
        if !resp.status().is_success() {
            let error: rpc::ErrorResponse = resp.json().await?;
            anyhow::bail!("fee sign failed: {}", error.error);
        }
        println!("  ✓ Signed fee input");
    }

    // Step 6: Wait for completion
    println!("\n  Waiting for all signatures and on-chain confirmation...");
    loop {
        if tokio::time::Instant::now() >= deadline {
            println!("  ⏳ Timed out. The mix may still complete — check status later.");
            break;
        }
        tokio::time::sleep(Duration::from_secs(3)).await;

        let status_url = format!("{}/mix/status/{}", base_url, mix_id_hex);
        let resp = client.get(&status_url).send().await?;
        if !resp.status().is_success() { break; }
        let status: rpc::MixStatusResponse = resp.json().await?;

        match status.phase.as_str() {
            "signing" => { eprint!("."); }
            "commit_submitted" => { eprint!("c"); }
            "complete" => {
                eprintln!();
                // Update wallet: remove spent coin(s), import mixed output
                let mut spent = vec![mix_coin_id];
                if let Some(fee_cid) = fee_coin_id {
                    spent.push(fee_cid);
                }
                wallet.complete_mix(&spent, &output, output_seed)?;

                println!("\n  ✓ CoinJoin mix complete!");
                println!("  Spent:    {}", hex::encode(&mix_coin_id));
                println!("  Received: {} (value {})", hex::encode(&output.coin_id().unwrap()), output.value());
                if let Some(fee_cid) = fee_coin_id {
                    println!("  Fee paid: {} (value 1)", hex::encode(&fee_cid));
                }
                return Ok(());
            }
            "failed" => {
                eprintln!();
                anyhow::bail!("mix failed");
            }
            _ => { eprint!("?"); }
        }
    }

    Ok(())
}

// ── Auto-Shatter & Drip Mixing ──────────────────────────────────────────────

async fn wallet_automix(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    coin_ref: String,
    timeout_secs: u64,
) -> Result<()> {
    let password = read_password("Password: ")?;
    
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();
    let base_url = format!("http://{}:{}", rpc_host, rpc_port);

    // 1. Resolve initial coin
    let base_coin_id = wallet.resolve_coin(&coin_ref)?;
    
    // Auto-consolidate any WOTS siblings to prevent co-spend consensus errors
    let mut selected_coins = vec![base_coin_id];
    let siblings = wallet.wots_siblings(&base_coin_id);
    for sib in siblings {
        selected_coins.push(sib);
    }
    
    let mut coin_val = 0u64;
    for cid in &selected_coins {
        let val = wallet.find_coin(cid)
            .ok_or_else(|| anyhow::anyhow!("Coin not found in wallet"))?.value;
        coin_val += val;
    }

    println!("🔍 Sniffing network for active CoinJoin liquidity...");
    let list_resp: midstate::rpc::MixListResponse = client.get(format!("{}/mix/list", base_url))
        .send().await?.json().await?;

    let mut target_mixes = Vec::new();
    let mut required_value = 0u64;
    let mut outputs = Vec::new();
    let mut change_seeds = Vec::new();

    // 2. Discover Liquidity and Provision Outputs
    for session in list_resp.sessions {
        if session.phase == "collecting" {
            let cost = session.denomination + 1; // +1 for the fee provision coin
            
            // Can we afford to fill this pool? (+1 overall for the shatter transaction's own fee)
            if required_value + cost + 1 <= coin_val {
                target_mixes.push((session.mix_id, session.denomination));
                required_value += cost;
                
                // Provision the target mix denomination output
                let seed1 = wallet.allocate_next_wots_seed()?;
                let pk1 = midstate::core::wots::keygen(&seed1);
                let addr1 = midstate::core::compute_address(&pk1);
                let salt1: [u8; 32] = rand::random();
                let idx1 = outputs.len();
                outputs.push(midstate::core::OutputData::Standard { address: addr1, value: session.denomination, salt: salt1 });
                change_seeds.push((idx1, seed1));

                // Provision the exact fee output (denom 1) needed for this specific pool
                let seed2 = wallet.allocate_next_wots_seed()?;
                let pk2 = midstate::core::wots::keygen(&seed2);
                let addr2 = midstate::core::compute_address(&pk2);
                let salt2: [u8; 32] = rand::random();
                let idx2 = outputs.len();
                outputs.push(midstate::core::OutputData::Standard { address: addr2, value: 1, salt: salt2 });
                change_seeds.push((idx2, seed2));
            }
        }
    }

    if target_mixes.is_empty() {
        println!("No affordable active pools found. Try standard mix creation.");
        return Ok(());
    }

    println!("🎯 Found {} active pools. Shattering coin to match required liquidity...", target_mixes.len());

    // Calculate normal change
    let change_value = coin_val.saturating_sub(required_value).saturating_sub(1); // -1 for shatter tx fee
    if change_value > 0 {
        let change_denoms = midstate::core::decompose_value(change_value);
        for denom in change_denoms {
            let seed = wallet.allocate_next_wots_seed()?;
            let pk = midstate::core::wots::keygen(&seed);
            let addr = midstate::core::compute_address(&pk);
            let salt: [u8; 32] = rand::random();
            let idx = outputs.len();
            outputs.push(midstate::core::OutputData::Standard { address: addr, value: denom, salt });
            change_seeds.push((idx, seed));
        }
    }

    // Shuffle outputs to prevent sequential linkage on-chain
    {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..outputs.len()).collect();
        indices.shuffle(&mut rand::thread_rng());
        let shuffled_outputs: Vec<midstate::core::OutputData> = indices.iter().map(|&i| outputs[i].clone()).collect();
        let mut reverse_map = vec![0usize; indices.len()];
        for (new_i, &old_i) in indices.iter().enumerate() { reverse_map[old_i] = new_i; }
        change_seeds = change_seeds.into_iter().map(|(old_idx, s)| (reverse_map[old_idx], s)).collect();
        outputs = shuffled_outputs;
    }

    // 3. Execute Shatter Transaction
    let (commitment, _salt) = wallet.prepare_commit(&selected_coins, &outputs, change_seeds, false, false)?;
    submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;
    println!("  Shatter Commit submitted ({}). Waiting for inclusion...", hex::encode(&commitment));
    
    if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), timeout_secs).await {
        anyhow::bail!("Timed out waiting for shatter commit to be mined");
    }

    do_reveal(&client, &mut wallet, rpc_port, &rpc_host, &commitment, timeout_secs).await?;
    println!("✅ Shatter transaction complete. Coins are now perfectly sized.");

    // EXTREMELY IMPORTANT: Drop the wallet instance so `wallet_mix` can safely acquire 
    // the file lock on `wallet.dat` when it runs recursively.
    drop(wallet);

    // Track coins we've already assigned to a pool to prevent duplicate usage
    let mut used_coins = std::collections::HashSet::new();

    // 4. The Drip Phase
    for (i, (target_mix_id, denom)) in target_mixes.into_iter().enumerate() {
        
        // Stochastic delay (5 to 25 seconds) to de-correlate mix timing from the shatter block
        let delay = 5 + (rand::random::<u64>() % 20);
        println!("\n⏳ [Drip {}] Waiting {} seconds to de-correlate timing...", i+1, delay);
        tokio::time::sleep(std::time::Duration::from_secs(delay)).await;

        println!("💧 [Drip {}] Joining Mix Pool {} for denomination {}...", i+1, &target_mix_id[..16], denom);

        // Open wallet temporarily to find the exact unspent UTXO we just shattered
        let coin_to_mix = {
            let temp_wallet = Wallet::open(path, &password)?;
            let mut found = None;
            for wc in temp_wallet.coins() {
                if wc.value == denom && !wc.wots_signed && !used_coins.contains(&wc.coin_id) {
                    // Double check it's fully confirmed on-chain
                    if check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await.unwrap_or(false) {
                        found = Some(hex::encode(wc.coin_id));
                        used_coins.insert(wc.coin_id);
                        break;
                    }
                }
            }
            found
        };

        let coin_to_mix = match coin_to_mix {
            Some(c) => c,
            None => {
                println!("⚠️ Could not find the shattered coin of denom {} for mix. Skipping.", denom);
                continue;
            }
        };

        // Forward to the standard mix handler, passing the password securely in memory
        match wallet_mix(
            path,
            rpc_port,
            rpc_host.clone(),
            denom,
            Some(coin_to_mix),
            Some(target_mix_id.clone()),
            true, // We provisioned exactly one 'denom 1' coin for this pool, so auto-pay the fee!
            timeout_secs,
            Some(password.clone()), // SECURE MEMORY PASSING
        ).await {
            Ok(_) => println!("🎉 Successfully joined and mixed pool {}", &target_mix_id[..8]),
            Err(e) => println!("⚠️ Mix {} failed: {}", &target_mix_id[..8], e),
        }
    }
    
    println!("\n✅ Auto-Shatter & Drip complete!");
    Ok(())
}

fn wallet_import(path: &PathBuf, seed_hex: &str, value: u64, salt_hex: &str, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let seed = parse_hex32(seed_hex)?;
    let salt = parse_hex32(salt_hex)?;
    let coin_id = wallet.import_coin(seed, value, salt, label)?;
    println!("Imported: {} (value {})", hex::encode(&coin_id), value);
    Ok(())
}

fn wallet_export(path: &PathBuf, coin_ref: &str) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let coin_id = wallet.resolve_coin(coin_ref)?;
    let wc = wallet.find_coin(&coin_id)
        .ok_or_else(|| anyhow::anyhow!("coin not found in wallet"))?;
    println!("Seed:    {}", hex::encode(wc.seed));
    println!("Value:   {}", wc.value);
    println!("Salt:    {}", hex::encode(wc.salt));
    println!("CoinID:  {}", hex::encode(wc.coin_id));
    println!("Address: {}", hex::encode(wc.address));
    println!("\n⚠️  Anyone with the seed + value + salt can spend this coin.");
    Ok(())
}

fn wallet_pending(path: &PathBuf) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let pending = wallet.pending();
    if pending.is_empty() {
        println!("No pending commits.");
        return Ok(());
    }
    println!("{} pending commit(s):\n", pending.len());
for (i, p) in pending.iter().enumerate() {
            let age = now_secs().saturating_sub(p.created_at);
            let out_val: u64 = p.outputs.iter().map(|o| o.value()).sum();
            println!(
                "  [{}] {} — {} in, {} out (value {}), {}",
                i, hex::encode(&p.commitment), // FIX: Use full hex string
                p.input_coin_ids.len(),
                p.outputs.len(),
                out_val,
                format_age(age),
            );
        }
    Ok(())
}

fn wallet_history(path: &PathBuf, count: usize) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let history = wallet.history();
    if history.is_empty() {
        println!("No transaction history.");
        return Ok(());
    }
    let start = history.len().saturating_sub(count);
    let entries = &history[start..];
    println!("Transaction history ({} of {}):\n", entries.len(), history.len());
    for (i, entry) in entries.iter().enumerate() {
        let age = now_secs().saturating_sub(entry.timestamp);
        let label = match entry.kind.as_str() {
            "received" => "received",
            "mixed"    => "mixed",
            _          => "sent",
        };
        println!("  [{}] {} {} (fee: {})", start + i, label, format_age(age), entry.fee);
        for c in &entry.inputs  { println!("    spent:    {}", hex::encode(c)); }
        for c in &entry.outputs { println!("    created:  {}", hex::encode(c)); }
        println!();
    }
    Ok(())
}

async fn wallet_reveal(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String, 
    commitment_hex: Option<String>,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let targets: Vec<[u8; 32]> = if let Some(hex) = commitment_hex {
        vec![parse_hex32(&hex)?]
    } else {
        wallet.pending().iter().map(|p| p.commitment).collect()
    };

    if targets.is_empty() {
        println!("No pending commits to reveal.");
        return Ok(());
    }

    let client = reqwest::Client::new();

    for commitment in targets {
        let pending = match wallet.find_pending(&commitment) {
            Some(p) => p.clone(),
            None => {
                println!("  {} — not found, skipping", hex::encode(&commitment));
                continue;
            }
        };

        if pending.reveal_not_before > now_secs() {
            let wait = pending.reveal_not_before - now_secs();
            println!("  {} — waiting {}s (privacy delay)", hex::encode(&commitment), wait);
            tokio::time::sleep(Duration::from_secs(wait)).await;
        }

        // Consolidate commits are signed with ONE aggregated witness covering all
        // inputs (`sign_consolidate`), matching the `wallet consolidate` flow. The
        // node's /send handler enforces exactly 1 signature for Consolidate txs, so
        // the one-signature-per-input output of `sign_reveal` can never be accepted
        // when manually revealing a timed-out consolidate commit.
        let sign_result = if pending.is_consolidate {
            wallet.sign_consolidate(&pending)
                .map(|(input_reveals, witness)| (input_reveals, vec![witness]))
        } else {
            wallet.sign_reveal(&pending)
        };
        let (input_reveals, signatures) = match sign_result {
            Ok(res) => res,
            Err(e) => {
                println!("  {} — dropping stale commit ({})", hex::encode(&commitment), e);
                // Delete the garbage commit to unblock the queue
                wallet.data.pending.retain(|p| p.commitment != commitment);
                let _ = wallet.save();
                continue; // Skip to the next commit instead of crashing
            }
        };
        
        let url = format!("http://{}:{}/send", rpc_host, rpc_port);
        let req = rpc::SendTransactionRequest {
            inputs: input_reveals.iter().map(|ir| rpc::InputRevealJson {
                bytecode: match &ir.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
                value: ir.value,
                salt: hex::encode(ir.salt),
                // The reference `wallet consolidate` flow submits inputs WITHOUT
                // per-input commitments (the single aggregated witness covers
                // them); mirror it exactly so the node treats both paths the same.
                commitment: if pending.is_consolidate { None } else { ir.commitment.map(hex::encode) },
            }).collect(),
            signatures: signatures.iter().map(|s| match s {
                midstate::core::types::Witness::ScriptInputs(inputs) => {
                    inputs.iter().map(hex::encode).collect::<Vec<_>>().join(",")
                }
            }).collect(),
            outputs: pending.outputs.iter().map(|o| match o {
            midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
            midstate::core::OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
        }).collect(),
            salt: hex::encode(pending.salt),
            is_consolidate: pending.is_consolidate,
        };

        let response = client.post(&url).json(&req).send().await?;
        if response.status().is_success() {
            let _result: rpc::SendTransactionResponse = response.json().await?;
            wallet.complete_reveal(&commitment)?;
            println!("  {} — revealed ✓", hex::encode(&commitment));
        } else {
            let error: rpc::ErrorResponse = response.json().await?;
            
            if error.error.contains("No matching commitment found") {
                // The commitment is gone. Did it actually confirm while we were timed out?
                // Check if the first output of this pending commit exists on-chain.
                if let Some(first_out) = pending.outputs.first().and_then(|o| o.coin_id()) {
                    let check_url = format!("http://{}:{}/check", rpc_host, rpc_port);
                    let check_req = rpc::CheckCoinRequest { coin: hex::encode(first_out) };
                    if let Ok(check_resp) = client.post(&check_url).json(&check_req).send().await {
                        if let Ok(check_res) = check_resp.json::<rpc::CheckCoinResponse>().await {
                            if check_res.exists {
                                // It actually confirmed! Clean up the wallet state.
                                wallet.complete_reveal(&commitment)?;
                                println!("  {} — already confirmed on-chain, wallet state fixed ✓", hex::encode(&commitment));
                                continue;
                            }
                        }
                    }
                }
            }

            println!("  {} — failed: {}", hex::encode(&commitment), error.error);
        }
    }
    Ok(())
}

fn wallet_import_rewards(path: &PathBuf, coinbase_file: &PathBuf, data_dir: &PathBuf) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    // Infer the data dir from the coinbase file location if not explicitly overridden.
    // coinbase_seeds.jsonl always lives at <data-dir>/coinbase_seeds.jsonl, so
    // the natural data-dir is the file's parent. Only fall back to the CLI --data-dir
    // if the file isn't named coinbase_seeds.jsonl (i.e. user explicitly placed it elsewhere).
    let inferred_data_dir = if coinbase_file.file_name().and_then(|n| n.to_str()) == Some("coinbase_seeds.jsonl") {
        coinbase_file.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| data_dir.clone())
    } else {
        data_dir.clone()
    };

    println!("Opening node database to retrieve mining seed...");
    let storage = midstate::storage::Storage::open(inferred_data_dir.join("db"))
        .context("Failed to open node database. Is the node running? Stop the node first to safely import rewards.")?;
    
    let mining_seed = storage.load_mining_seed()?
        .context("No mining seed found in the database. Has the node started mining yet?")?;

    println!("Loading chain state to check for spent rewards...");
    let live_coins = storage.load_state()?
        .map(|s| s.coins)
        .context("No chain state found in the database. Has the node synced at all?")?;

    drop(storage); // Release the database lock immediately

    println!("Reading coinbase log...");
    let contents = std::fs::read_to_string(coinbase_file)?;

    #[derive(serde::Deserialize)]
    struct CoinbaseEntry {
        height: u64,
        index: u64,
        address: String,
        #[serde(rename = "coin")]
        _coin: String,
        value: u64,
        salt: String,
    }

    let entries: Vec<CoinbaseEntry> = contents
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    println!("Found {} rewards. Calculating private keys and importing...", entries.len());

    let new_coins: Vec<wallet::WalletCoin> = entries
        .into_par_iter()
        .map(|entry| {
            // Dynamically recreate the private seed using the master mining_seed!
            let seed = midstate::wallet::coinbase_seed(&mining_seed, entry.height, entry.index);
            let salt = parse_hex32(&entry.salt).unwrap();
            let address = parse_hex32(&entry.address).unwrap();
            let owner_pk = midstate::core::wots::keygen(&seed);
            let coin_id = midstate::core::compute_coin_id(&address, entry.value, &salt);

            midstate::wallet::WalletCoin {
                seed,
                owner_pk,
                address,
                value: entry.value,
                salt,
                coin_id,
                label: Some(format!("coinbase (value {})", entry.value)),
                wots_signed: false,
                commitment: None,
            }
        })
        .collect();

    let existing_coins: std::collections::HashSet<_> = wallet.data.coins
        .iter()
        .map(|c| c.coin_id)
        .collect();

    let mut imported = 0usize;
    let mut already_have = 0usize;
    let mut spent = 0usize;
    for wc in new_coins {
        if existing_coins.contains(&wc.coin_id) {
            already_have += 1;
        } else if !live_coins.contains(&wc.coin_id) {
            spent += 1;
        } else {
            wallet.data.coins.push(wc);
            imported += 1;
        }
    }

    println!("Saving wallet...");
    wallet.save()?;

    println!("Imported {} coinbase reward(s). Skipped {} already in wallet, {} spent on-chain. Total coins: {}, total value: {}",
        imported, already_have, spent, wallet.coin_count(), wallet.total_value());
    Ok(())
}

// ── Phase 1 License Skeletons (full implementations in follow-up) ───────────

/// Issue a new Pruning License.
///
/// This is currently a skeleton. When complete it will:
/// 1. Construct a Transaction::Reveal containing a Confidential output locked
///    to the royalty covenant (see Wallet::build_pruning_license_covenant).
/// 2. Include a DataBurn output for the burn-to-boost tax (see design §5.1).
/// 3. Store the LicenseMetadata in the wallet under the resulting coin_id.
///
/// See docs/design/pruning-licenses.md §3.1 and §7 for the exact covenant script
/// and the formal LicenseTransfer Z schema.
async fn wallet_issue_license(
    path: &PathBuf,
    bundle: Option<&std::path::Path>,
    poaw_commitment_hex: Option<&str>,
    fixed_fee: u64,
    min_height: Option<u64>,
    max_height: Option<u64>,
    archival_weight: Option<u64>,
    rpc_port: u16,
    rpc_host: String,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    // === Wiring: Load PoAW data (Phase 2 bundle preferred) ===
    // We no longer carry the (untrusted) poaw_commitment string from the bundle.
    // When a bundle is present the verified value is derived strictly from the
    // validated Merkle root inside the block below. Manual mode takes the hex directly.
    let (min_h, max_h, weight, issuer) = if let Some(bundle_path) = bundle {
        let data = std::fs::read(bundle_path)
            .with_context(|| format!("Failed to read PoAW bundle: {}", bundle_path.display()))?;
        let bundle: serde_json::Value = serde_json::from_slice(&data)?;

        let issuer = bundle["issuer"].as_str().unwrap_or("").to_string();

        let mh = min_height.or_else(|| bundle["start_height"].as_u64());
        let mxh = max_height.or_else(|| bundle["end_height"].as_u64());
        let w = archival_weight.or_else(|| {
            // Rough estimate if not provided
            bundle["end_height"].as_u64().zip(bundle["start_height"].as_u64())
                .map(|(e, s)| (e.saturating_sub(s)) / bundle["stride"].as_u64().unwrap_or(1))
        });

        (mh, mxh, w, issuer)
    } else if poaw_commitment_hex.is_some() {
        (min_height, max_height, archival_weight, String::new())
    } else {
        bail!("Either --bundle or --poaw-commitment must be provided");
    };

    let min_h = min_h.ok_or_else(|| anyhow::anyhow!("min_height required (provide via --min-height or in bundle)"))?;
    let max_h = max_h.ok_or_else(|| anyhow::anyhow!("max_height required"))?;
    let weight = weight.unwrap_or((max_h - min_h) / 10); // fallback heuristic

    // === PoAW commitment: authoritative bytes (verified when bundle present) ===
    // We compute/override the bytes here so the value that reaches LicenseMetadata
    // is *always* the one we are willing to stand behind.
    let poaw_commitment_bytes: [u8; 32] = if let Some(bundle_path) = bundle {
        let data = std::fs::read(bundle_path)
            .with_context(|| format!("Failed to re-read PoAW bundle for validation: {}", bundle_path.display()))?;
        let bundle_json: serde_json::Value = serde_json::from_slice(&data)?;

        let issuer_hex = bundle_json["issuer"].as_str().unwrap_or("").to_string();
        let issuer_bytes = if issuer_hex.is_empty() { [0u8; 32] } else { parse_hex32(&issuer_hex)? };

        let sampled: Vec<u64> = bundle_json["sampled_heights"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();

        let nonces_val = &bundle_json["nonces"];
        let nonces: Vec<u32> = if let Some(arr) = nonces_val.as_array() {
            arr.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect()
        } else {
            vec![]
        };

        let difficulty = bundle_json["difficulty"].as_u64().unwrap_or(20) as u32;
        let claimed_merkle_root = bundle_json["pow_merkle_root"]
            .as_str()
            .and_then(|s| hex::decode(s).ok())
            .and_then(|b| if b.len() == 32 { let mut a = [0u8;32]; a.copy_from_slice(&b); Some(a) } else { None })
            .unwrap_or([0u8; 32]);

        if sampled.len() != nonces.len() || sampled.is_empty() {
            bail!("Invalid PoAW bundle: sampled_heights and nonces length mismatch or empty");
        }

        let mut pow_leaves: Vec<[u8; 32]> = Vec::with_capacity(sampled.len());

        // Create RPC client early for fetching real historical block hashes (required for meaningful PoAW)
        let client = reqwest::Client::new();

        for (i, &h) in sampled.iter().enumerate() {
            // Fetch the REAL payload hash to verify against
            let payload_hash = fetch_payload_hash_for_poaw(&client, rpc_port, &rpc_host, h).await
                .map_err(|e| anyhow::anyhow!("PoAW validation FAILED: Could not fetch block data for height {}. Is your node fully synced? Error: {}", h, e))?;

            let nonce = nonces.get(i).copied().unwrap_or(0);

            let mut data = Vec::with_capacity(68);
            data.extend_from_slice(&issuer_bytes);
            data.extend_from_slice(&payload_hash);
            data.extend_from_slice(&nonce.to_le_bytes());

            let pow_hash = midstate::core::types::hash(&data);
            if midstate::core::types::count_leading_zeros(&pow_hash) < difficulty {
                bail!("PoAW validation FAILED at height {}: leading zeros < required difficulty {}", h, difficulty);
            }

            let mut leaf = [0u8; 32];
            leaf.copy_from_slice(&pow_hash);
            pow_leaves.push(leaf);
        }

        // Rebuild the Merkle tree exactly as the generator does and compare roots
        let computed_root = build_pow_merkle(&pow_leaves);
        if computed_root != claimed_merkle_root {
            bail!("PoAW validation FAILED: recomputed Merkle root does not match bundle pow_merkle_root");
        }

        // SECURITY: The *only* poaw_commitment value we accept when a bundle is supplied.
        // Derived strictly from the verified Merkle root of address-salted PoW samples.
        // The JSON field is ignored for the on-chain metadata (prevents spoofing).
        let verified_poaw_commitment = {
            let mut data = Vec::new();
            data.extend_from_slice(&computed_root);
            midstate::core::types::hash(&data)
        };

        tracing::info!("PoAW bundle cryptographically validated ({} samples, difficulty {})", sampled.len(), difficulty);
        println!("✅ Using VERIFIED PoAW commitment derived from bundle (anti-spoofing enforced)");

        verified_poaw_commitment
    } else {
        // Manual --poaw-commitment path (no bundle to verify)
        let bytes = parse_hex32(poaw_commitment_hex.expect("checked earlier"))?;
        if bytes == [0u8; 32] {
            bail!("PoAW commitment cannot be the zero hash");
        }
        bytes
    };

    // Build the on-chain metadata commitment (what goes into the Confidential output)
    let license_meta = midstate::wallet::LicenseMetadata {
        issuer: if issuer.is_empty() { [0u8; 32] } else { parse_hex32(&issuer).unwrap_or([0u8;32]) },
        fixed_royalty_fee: fixed_fee,
        min_height: min_h,
        max_height: max_h,
        archival_weight: weight,
        poaw_commitment: poaw_commitment_bytes,
        issuance_height: 0, // filled at reveal time if desired
        nonce: rand::random(),
    };

    // The commitment that will be stored on-chain in the Confidential output
    let meta_bytes = serde_json::to_vec(&license_meta)?;
    let meta_hash = midstate::core::types::hash(&meta_bytes);

    // SECURITY/UX FIX: Derive salt deterministically from next HD WOTS seed.
    // This lets us know the *final* coin_id of the license output *before* we broadcast the commit.
    // We store the metadata under the final coin_id immediately → no brittle re-keying later in complete_reveal.
    let salt_seed = wallet.allocate_next_wots_seed()?;
    let salt_input = [&b"license-salt-v1"[..], &salt_seed[..]].concat();
    let salt = midstate::core::types::hash(&salt_input);

    let covenant_script = midstate::wallet::Wallet::build_pruning_license_covenant(license_meta.issuer, fixed_fee);
    let covenant_address = midstate::compute_address(&midstate::core::types::hash(&covenant_script));

    let license_output = midstate::core::OutputData::Confidential {
        address: covenant_address,
        commitment: meta_hash,
        salt,
    };

    // Use the *correct* Confidential coin_id computation (not the Standard one)
    let final_coin_id = license_output.coin_id().expect("Confidential output must produce coin_id");

    // Store under the *final* coin_id right now.
    wallet.store_license_metadata(final_coin_id, license_meta.clone())?;

    let onchain_commitment = meta_hash; // still used for the Confidential commitment field (metadata hash part)

    println!("✅ License metadata stored in wallet (keyed by on-chain commitment).");
    println!("   Fixed Royalty Fee: {} Midstate (paid on every transfer)", fixed_fee);
    println!("   Range: {}..{}", min_h, max_h);
    println!("   Weight: {}", weight);

    // === Create a pending Reveal that will produce the license ===
    let covenant_script = midstate::wallet::Wallet::build_pruning_license_covenant(license_meta.issuer, fixed_fee);
    let covenant_address = midstate::compute_address(&midstate::core::types::hash(&covenant_script)); 

    let license_output = midstate::core::OutputData::Confidential {
        address: covenant_address,
        commitment: onchain_commitment,
        salt,
    };

    // Small burn-to-boost (issuance tax)
    let burn_amount: u64 = 100;
    let burn_output = midstate::core::OutputData::DataBurn {
        payload: b"pruning-license-issuance".to_vec(),
        value_burned: burn_amount,
    };

    // Bearer asset recoverability: Burn a copy of the license metadata on-chain.
    // Since MAX_BURN_DATA_SIZE is strictly 80 bytes, we chunk the metadata across
    // multiple DataBurn outputs. Each chunk has a 2-byte header: [index][total].
    // Recovery tool will reassemble them.
    let mut metadata_burns: Vec<midstate::core::OutputData> = Vec::new();
    const CHUNK_DATA_SIZE: usize = 78; // 80 - 2 byte header
    let total_chunks = (meta_bytes.len() + CHUNK_DATA_SIZE - 1) / CHUNK_DATA_SIZE;
    for (i, chunk) in meta_bytes.chunks(CHUNK_DATA_SIZE).enumerate() {
        let mut payload = Vec::with_capacity(2 + chunk.len());
        payload.push(i as u8);
        payload.push(total_chunks as u8);
        payload.extend_from_slice(chunk);
        metadata_burns.push(midstate::core::OutputData::DataBurn {
            payload,
            value_burned: 0,
        });
    }

    // Find inputs to fund the burn + fee
    let mut live_coins = Vec::new();
    let client = reqwest::Client::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_coins.push(wc.coin_id);
        }
    }

    let target_fee = 100;
    let needed = burn_amount + target_fee;
    let selected = wallet.select_coins(needed, &live_coins)?;

    let in_sum: u64 = selected.iter().filter_map(|id| wallet.find_coin(id)).map(|c| c.value).sum();
    let change = in_sum.saturating_sub(burn_amount + target_fee);
    
    let mut outputs = vec![license_output, burn_output];
    outputs.extend(metadata_burns);
    let mut change_seeds = Vec::new();

    if change > 0 {
        for denom in midstate::core::decompose_value(change) {
            let seed = wallet.allocate_next_wots_seed()?;
            let pk = midstate::core::wots::keygen(&seed);
            let addr = midstate::core::compute_address(&pk);
            let idx = outputs.len();
            outputs.push(midstate::core::OutputData::Standard { address: addr, value: denom, salt: rand::random() });
            change_seeds.push((idx, seed));
        }
    }

    // Prepare commit pushes the pending tx to the wallet
    let (tx_commitment, _) = wallet.prepare_commit(&selected, &outputs, change_seeds, false, false)?;

    // Submit the commit to the network
    submit_commit(&client, rpc_port, &rpc_host, &tx_commitment).await?;

    println!("\n✅ Pending license creation commit created.");
    println!("   On-chain metadata commitment: {}", hex::encode(onchain_commitment));
    println!("   Transaction commitment: {}", hex::encode(tx_commitment));
    println!("   PoAW commitment accepted (wallet-side basic validation passed; see design for full verification model)");
    println!("   Wait for the commit to be mined, then run:");
    println!("     midstate wallet reveal --path {} --rpc-port {} --rpc-host {} --commitment {}",
             path.display(), rpc_port, rpc_host, hex::encode(tx_commitment));

    println!("\nThe license metadata is now safely stored in your encrypted wallet (keyed by the above metadata commitment).");
    println!();
    println!("After the reveal confirms on-chain:");
    println!("  1. Find the real coin_id of the new Confidential output (use `wallet list --full` or a block explorer).");
    println!("  2. The wallet will automatically re-key it for you when you run the reveal command.");
    println!("  (Manual fallback: midstate wallet rekey-license --path {} --old {} --new-coin-id <real-coin-id>)",
             path.display(), hex::encode(onchain_commitment));
    println!();
    println!("The license will then be a fully tradable Pruning License with on-chain royalty enforcement.");
    println!();
    println!("⚠️  BEARER ASSET: The LicenseMetadata (royalty terms, ranges, PoAW) is stored in your");
    println!("   encrypted wallet.dat and was also DataBurned on-chain for recoverability. It is");
    println!("   NOT derivable from your seed phrase. Back up wallet.dat (or the burn txid) separately!");

    Ok(())
}

/// Purchase a license from another party.
///
/// The resulting transaction must pay both the seller *and* the original issuer
/// (royalty) plus the burn-to-boost component. The wallet will automatically
/// attach the stored LicenseMetadata when spending the Confidential input.
async fn wallet_buy_license(
    path: &PathBuf,
    license: &str,
    price: u64,
    seller_pk_hex: &str,
    rpc_port: u16,
    rpc_host: String,
    _timeout: u64,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let license_key = parse_hex32(license).ok();

    let meta = if let Some(key) = license_key {
        wallet.get_license_metadata(&key).cloned()
            .or_else(|| {
                wallet.list_licenses().into_iter()
                    .find(|(k, _)| **k == key)
                    .map(|(_, m)| m.clone())
            })
    } else {
        None
    };

    let meta = match meta {
        Some(m) => m,
        None => {
            println!("Could not find license metadata for '{}'.", license);
            println!("Make sure you have imported/stored the license metadata in this wallet.");
            return Ok(());
        }
    };

    // Ensure the metadata is stored under its canonical onchain_commitment key.
    // This allows complete_reveal (during the buy reveal) to automatically detect and
    // re-key it to the final coin_id of the new Confidential output.
    let onchain_commitment = midstate::core::types::hash(&serde_json::to_vec(&meta)?);
    wallet.store_license_metadata(onchain_commitment, meta.clone())?;

    let transfer_fee = meta.fixed_royalty_fee;
    let burn_amount = 100u64;

    println!("=== PURCHASE PRUNING LICENSE (functional) ===");
    println!("License range: {}..{}", meta.min_height, meta.max_height);
    println!("Fixed Transfer Fee to issuer ({}): {}", hex::encode(meta.issuer), transfer_fee);
    println!("Burn-to-boost on transfer: {}", burn_amount);
    println!("Payment to current seller: {}", price);
    println!();

    // Find coins to fund the purchase (price + fees + buffer)
    let mut live_coins = Vec::new();
    let client = reqwest::Client::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_coins.push(wc.coin_id);
        }
    }

    let needed = price + transfer_fee + burn_amount + 200; // buffer for fees
    let selected = wallet.select_coins(needed, &live_coins)?;

    let in_sum: u64 = selected.iter().filter_map(|id| wallet.find_coin(id)).map(|c| c.value).sum();
    let change = in_sum.saturating_sub(price + transfer_fee + burn_amount);

    // Build outputs:
    // - New Confidential license for the buyer (same covenant + state)
    // IMPORTANT: In the unidirectional + HTLC model, the *buyer* does NOT create a new license output.
    // Doing so would be forging a counterfeit license (the seller's original UTXO is never spent).
    // The buyer only creates the HTLC (price), royalty to issuer, and burn.
    // The seller is responsible for the actual license transfer when claiming the HTLC.
    //
    // We still store the metadata under a placeholder key so the buyer has it for when the seller eventually transfers it.
    let meta_commitment = midstate::core::types::hash(&serde_json::to_vec(&meta).unwrap_or_default());
    wallet.store_license_metadata(meta_commitment, meta.clone())?;

    let mut outputs = vec![];

    // === ATOMIC SWAP via HTLC (replaces unidirectional payment) ===
    let seller_pk = parse_hex32(seller_pk_hex)
        .context("Invalid --seller-pk (must be 64 hex chars, the raw WOTS public key of the current license holder)")?;

    // Buyer must obtain the secret_hash from the Seller off-chain.
    // The Seller generates the secret and gives only the hash to the Buyer.
    // This ensures the Seller can claim the HTLC by revealing the secret when they transfer the license.
    let _secret_hash_hex: String = /* In a real CLI this would come from --secret-hash arg */ String::new();
    // For now we require it via environment or placeholder — production version should add --secret-hash
    let secret_hash: [u8; 32] = if let Ok(h) = std::env::var("MIDSTATE_HTLC_SECRET_HASH") {
        let mut bytes = [0u8; 32];
        hex::decode_to_slice(h, &mut bytes).context("Invalid MIDSTATE_HTLC_SECRET_HASH")?;
        bytes
    } else {
        bail!("Buyer must provide seller-generated secret hash via MIDSTATE_HTLC_SECRET_HASH env var (or --secret-hash in full CLI)");
    };

    // Derive a refund keypair from the wallet (buyer gets price back if seller never claims)
    let refund_seed = wallet.allocate_next_wots_seed()?;
    let refund_pk = midstate::core::wots::keygen(&refund_seed);

    // Timeout: use a generous window (buyer can query current height via RPC in production)
    let timeout_height = 1_000_000u64; // placeholder; real impl should query /state + add buffer

    let htlc_script = midstate::core::script::compile_htlc(
        &secret_hash,
        &seller_pk,     // receiver PK (seller claims by revealing preimage)
        timeout_height,
        &refund_pk,     // refund PK (buyer can refund after timeout)
    );
    let htlc_address = midstate::compute_address(&midstate::core::types::hash(&htlc_script));

    // The price goes into the HTLC. The royalty goes directly to the issuer (as before).
    let htlc_output = midstate::core::OutputData::Standard {
        address: htlc_address,
        value: price,
        salt: rand::random(),
    };
    outputs.push(htlc_output);

    println!("\n[HTLC Atomic Swap] Using seller-provided secret hash.");
    println!("  Secret Hash:  {}", hex::encode(secret_hash));
    println!("  HTLC timeout height (approx): {}", timeout_height);
    println!("  After seller transfers the license and claims the HTLC by revealing the secret on-chain,");
    println!("  you (buyer) can use the revealed secret from the blockchain to claim any mirrored HTLC if needed.");
    println!("  If seller never claims, you can refund the HTLC after the timeout.");

    let royalty_output = midstate::core::OutputData::Standard {
        address: meta.issuer,
        value: transfer_fee,
        salt: rand::random(),
    };
    outputs.push(royalty_output);

    let burn_output = midstate::core::OutputData::DataBurn {
        payload: b"pruning-license-transfer".to_vec(),
        value_burned: burn_amount,
    };
    outputs.push(burn_output);

    let mut change_seeds = Vec::new();
    if change > 0 {
        for denom in midstate::core::decompose_value(change) {
            let seed = wallet.allocate_next_wots_seed()?;
            let pk = midstate::core::wots::keygen(&seed);
            let addr = midstate::core::compute_address(&pk);
            let idx = outputs.len();
            outputs.push(midstate::core::OutputData::Standard {
                address: addr,
                value: denom,
                salt: rand::random(),
            });
            change_seeds.push((idx, seed));
        }
    }

    let (tx_commitment, _) = wallet.prepare_commit(&selected, &outputs, change_seeds, false, false)?;

    submit_commit(&client, rpc_port, &rpc_host, &tx_commitment).await?;

    println!("\n✅ Purchase transaction prepared and commit submitted.");
    println!("   Transaction commitment: {}", hex::encode(tx_commitment));
    println!("   After mining, run `midstate wallet reveal --commitment {}` to complete the transfer.", hex::encode(tx_commitment));
    println!();

    // === Critical unidirectional payment warning (no HTLC atomic swap yet) ===
    tracing::warn!(
        "UNIDIRECTIONAL PAYMENT: You are sending the full `price` ({}) directly to the seller address in this transaction. \
         There is no on-chain atomic swap or HTLC enforcing that the seller must deliver the license UTXO to you. \
         Use a trusted escrow, a reputable marketplace with reputation, or only buy from parties you trust to manually transfer the license after seeing the payment on-chain. \
         This is a known limitation until full covenant-based atomic swaps or HTLC support are added.",
        price
    );
    println!();
    println!("⚠️  WARNING: This is a unidirectional payment. The seller must manually transfer the license UTXO to you after seeing your payment.");
    println!("   Use escrow or only transact with parties you trust. Future versions will support trustless HTLC-style atomic swaps for license purchases.");

    Ok(())
}

// ── Phase 2: PoAW Generator ─────────────────────────────────────────────────

/// Simple binary Merkle tree over 32-byte leaves (for PoW results).
/// This is intentionally small and self-contained for Phase 2.
/// Seller side of the HTLC atomic swap for a Pruning License.
///
/// Current implementation (Phase 1):
/// - Loads the license from the wallet
/// - Generates a fresh random secret + its hash
/// - Prints the hash for the buyer to use when calling `BuyLicense`
/// - Instructs the seller what to do next (watch for HTLC on-chain, then transfer + claim)
///
/// Full atomic claim (seller builds Reveal that both transfers the license covenant
/// output *and* claims the buyer's HTLC in one transaction by revealing the preimage)
/// is the deeper integration step still pending.
async fn wallet_sell_license(
    path: &PathBuf,
    license: &str,
    secret_hex: Option<&str>,
    buyer_pk_hex: Option<&str>,
    _rpc_port: u16,
    _rpc_host: &str,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let license_key = parse_hex32(license)?;

    let meta = wallet
        .get_license_metadata(&license_key)
        .cloned()
        .or_else(|| {
            wallet
                .list_licenses()
                .into_iter()
                .find(|(k, _)| **k == license_key)
                .map(|(_, m)| m.clone())
        });

    let meta = match meta {
        Some(m) => m,
        None => {
            println!("Could not find license metadata for '{}'.", license);
            println!("Make sure this wallet currently holds the license (or has the metadata stored).");
            return Ok(());
        }
    };

    println!("=== SELL PRUNING LICENSE (HTLC Atomic Swap - Seller) ===");
    println!("License range: {}..{}", meta.min_height, meta.max_height);
    println!("Fixed royalty on transfer: {}", meta.fixed_royalty_fee);
    println!();

    if let Some(secret_str) = secret_hex {
        // ==================== CLAIM MODE ====================
        let mut secret = [0u8; 32];
        hex::decode_to_slice(secret_str, &mut secret)
            .context("Invalid --secret hex")?;

        let secret_hash = midstate::core::types::hash(&secret);

        let _buyer_pk = if let Some(pk_hex) = buyer_pk_hex {
            parse_hex32(pk_hex).context("Invalid --buyer-pk")?
        } else {
            bail!("--buyer-pk is required in claim mode (the raw WOTS PK the buyer wants the new license sent to)");
        };

        println!("CLAIM MODE — preparing license transfer Reveal + HTLC claim data");
        println!("  Using secret hash: {}", hex::encode(secret_hash));
        println!();

        let transfer_fee = meta.fixed_royalty_fee;
        let burn_amount = 100u64;

        let mut outputs = vec![];

        // New Confidential license for the buyer (same royalty covenant)
        let covenant_script = midstate::wallet::Wallet::build_pruning_license_covenant(meta.issuer, transfer_fee);
        let covenant_address = midstate::compute_address(&midstate::core::types::hash(&covenant_script));

        let meta_bytes = serde_json::to_vec(&meta)?;
        let meta_hash = midstate::core::types::hash(&meta_bytes);

        let salt_seed = wallet.allocate_next_wots_seed()?;
        let salt_input = [&b"license-sale-claim-v1"[..], &salt_seed[..]].concat();
        let salt = midstate::core::types::hash(&salt_input);

        let new_license_output = midstate::core::OutputData::Confidential {
            address: covenant_address,
            commitment: meta_hash,
            salt,
        };
        outputs.push(new_license_output);

        // Royalty to issuer
        outputs.push(midstate::core::OutputData::Standard {
            address: meta.issuer,
            value: transfer_fee,
            salt: rand::random(),
        });

        // Burn
        outputs.push(midstate::core::OutputData::DataBurn {
            payload: b"pruning-license-transfer".to_vec(),
            value_burned: burn_amount,
        });

        // Select coins and prepare the Reveal for the license transfer
        let live_coins: Vec<_> = wallet.coins().iter().map(|c| c.coin_id).collect();
        let selected = wallet.select_coins(transfer_fee + burn_amount + 100, &live_coins)?;

        let (tx_commitment, _) = wallet.prepare_commit(&selected, &outputs, vec![], false, false)?;

        println!("✅ License transfer Reveal prepared.");
        println!("   Commitment: {}", hex::encode(tx_commitment));
        println!("   Submit with: midstate wallet reveal --commitment {}", hex::encode(tx_commitment));
        println!();

        // HTLC claim data
        println!("=== HTLC CLAIM DATA ===");
        println!("  Preimage (raw secret - reveal this on-chain to claim): {}", hex::encode(secret));
        println!("  Preimage hash: {}", hex::encode(secret_hash));
        println!();
        println!("To claim the HTLC the buyer locked for you:");
        println!("- Witness for HTLC input (IF branch): [your_sig, preimage_above, 0x01]");
        println!("- Full combined license-transfer + HTLC-claim in one tx is the remaining deeper integration.");
        println!();

        Ok(())
    } else {
        // ==================== OFFER / GENERATE HASH MODE ====================
        let mut secret = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut secret);
        let secret_hash = midstate::core::types::hash(&secret);

        println!("✅ Generated HTLC secret for this sale (offer phase).");
        println!("   Give ONLY the hash below to the buyer (via secure channel):");
        println!("   Secret Hash: {}", hex::encode(secret_hash));
        println!();
        println!("   Keep the raw secret safe. You will need it in claim mode later.");
        println!("   [DEV] Raw secret (hex): {}", hex::encode(secret));
        println!();
        println!("Buyer should run something like:");
        println!("  MIDSTATE_HTLC_SECRET_HASH={} midstate wallet buy-license ... --seller-pk YOUR_PK", hex::encode(secret_hash));
        println!();
        println!("When the buyer has locked the HTLC, re-run this command with:");
        println!("  --secret {} --buyer-pk THEIR_PK", hex::encode(secret));
        println!("to prepare your license transfer Reveal + obtain the preimage for claiming payment.");

        Ok(())
    }
}

fn build_pow_merkle(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current: Vec<[u8; 32]> = leaves.to_vec();

    while current.len() > 1 {
        let mut next = Vec::with_capacity((current.len() + 1) / 2);
        for chunk in current.chunks(2) {
            let mut data = Vec::with_capacity(64);
            data.extend_from_slice(&chunk[0]);
            if chunk.len() == 2 {
                data.extend_from_slice(&chunk[1]);
            } else {
                data.extend_from_slice(&chunk[0]); // duplicate last for odd length
            }
            let h = types::hash(&data);
            let mut out = [0u8; 32];
            out.copy_from_slice(&h);
            next.push(out);
        }
        current = next;
    }
    current[0]
}

/// Generate Proof of Archival Work for a range.
///
/// # Reasoning
/// Issuing a high-value Pruning License requires expensive, verifiable work
/// proving the issuer actually holds the historical data. We use:
/// - Address as salt (prevents grinding / theft of other people's work)
/// - Sampling + Merkle tree (compact on-chain commitment)
/// - Later: MMR proofs (from core/mmr.rs) for verifiability against chain history
///
/// This is the "make it hard" gate described in the original design discussions.
async fn wallet_poaw_generate(
    path: &PathBuf,
    start_height: u64,
    end_height: u64,
    stride: u64,
    difficulty: u32,
    issuer_hex: &str,
    submit_commit: bool,
    rpc_port: u16,
    rpc_host: &str,
) -> Result<()> {
    if start_height > end_height { bail!("start height must be <= end height"); }
    if stride == 0 { bail!("stride must be greater than 0"); }

    let issuer: [u8; 32] = parse_hex32(issuer_hex)?;

    println!("=== Proof of Archival Work Generation (Phase 2) ===");
    println!("Range: {} .. {}", start_height, end_height);
    println!("Stride: {}", stride);
    println!("Difficulty target: {} leading zero bits", difficulty);
    println!("Issuer (salt): {}", issuer_hex);
    println!();

    let mut sampled_heights = Vec::new();
    let mut pow_leaves = Vec::new();
    let mut nonces = Vec::new();
    let client = reqwest::Client::new();

    let mut h = start_height;
    while h <= end_height {
        sampled_heights.push(h);

        // Fetch the REAL transaction payload hash from the node
        let payload_hash = fetch_payload_hash_for_poaw(&client, rpc_port, rpc_host, h)
            .await
            .unwrap_or_else(|e| {
                println!("⚠️  Failed to fetch block {}: {}. Generating a fake hash for testing, this WILL fail validation on a real network.", h, e);
                let mut fake = [0u8; 32];
                fake[0..8].copy_from_slice(&h.to_le_bytes());
                fake
            });

        // Address-salted PoW forcing the archiver to grind the payload
        let mut nonce = 0u32;
        loop {
            let mut data = Vec::with_capacity(68);
            data.extend_from_slice(&issuer);
            data.extend_from_slice(&payload_hash);
            data.extend_from_slice(&nonce.to_le_bytes());

            let pow_hash = types::hash(&data);
            if types::count_leading_zeros(&pow_hash) >= difficulty {
                let mut leaf = [0u8; 32];
                leaf.copy_from_slice(&pow_hash);
                pow_leaves.push(leaf);
                nonces.push(nonce);
                println!("  Height {} -> nonce {} ({} bits)", h, nonce, difficulty);
                break;
            }
            nonce += 1;
            if nonce > 100_000_000 {
                bail!("Failed to find PoW solution for height {} within reasonable work", h);
            }
        }

        h = h.saturating_add(stride);
        if h == 0 { break; } 
    }

    let pow_merkle_root = build_pow_merkle(&pow_leaves);

    println!("\nSampled {} blocks", sampled_heights.len());
    println!("PoW Merkle root: {}", hex::encode(pow_merkle_root));

    let poaw_commitment = {
        let mut data = Vec::new();
        data.extend_from_slice(&pow_merkle_root);
        types::hash(&data)
    };

    println!("Combined PoAW commitment: {}", hex::encode(poaw_commitment));

    if submit_commit {
        println!("\nSubmitting Transaction::Commit...");
        println!("(In complete implementation this would call the RPC send endpoint with a Commit)");
    }

    let bundle_path = path.with_extension("poaw-bundle.json");
    let bundle = serde_json::json!({
        "issuer": issuer_hex,
        "start_height": start_height,
        "end_height": end_height,
        "stride": stride,
        "difficulty": difficulty,
        "sampled_heights": sampled_heights,
        "nonces": nonces,
        "pow_merkle_root": hex::encode(pow_merkle_root),
        "poaw_commitment": hex::encode(poaw_commitment),
    });

    std::fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle)?)?;
    println!("\nWrote PoAW bundle to: {}", bundle_path.display());
    Ok(())
}

async fn wallet_rekey_license(path: &PathBuf, old_hex: &str, new_coin_id_hex: &str) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let old_key = parse_hex32(old_hex)?;
    let new_coin_id = parse_hex32(new_coin_id_hex)?;

    wallet.rekey_license_metadata(old_key, new_coin_id)?;

    println!("✅ License re-keyed successfully.");
    println!("   Old key (commitment): {}", old_hex);
    println!("   New coin_id:          {}", new_coin_id_hex);
    println!();
    println!("You can now use the real coin_id when spending or listing the license.");

    Ok(())
}

async fn wallet_list_licenses(path: &PathBuf) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;

    let licenses = wallet.list_licenses();

    if licenses.is_empty() {
        println!("No Pruning Licenses stored in this wallet.");
        return Ok(());
    }

    println!("⚠️  CRITICAL BEARER-ASSET WARNING ⚠️");
    println!("   Pruning License metadata (issuer, royalty fee, ranges, PoAW commitment) lives ONLY");
    println!("   in this encrypted wallet.dat + any DataBurns you created at issuance time.");
    println!("   It is NOT protected by your seed phrase / mnemonic. Losing wallet.dat without a");
    println!("   separate backup means you may lose the ability to easily identify or re-key the");
    println!("   licenses even if the underlying covenant coins remain spendable on-chain.");
    println!("   Recovery: scan the chain for DataBurn payloads containing your serialized metadata.");
    println!();
    println!("Pruning Licenses in wallet ({}):", licenses.len());
    println!();

    for (i, (key, meta)) in licenses.iter().enumerate() {
        println!("  [{}] Commitment/Key: {}", i, hex::encode(key));
        println!("      Issuer:           {}", hex::encode(meta.issuer));
        println!("      Fixed Royalty Fee: {}", meta.fixed_royalty_fee);
        println!("      Range (serving health): {} .. {}", meta.min_height, meta.max_height);
        println!("      Archival Weight:  {}", meta.archival_weight);
        println!("      PoAW Commitment:  {}", hex::encode(meta.poaw_commitment));
        println!("      (Live P2P reputation for this license is maintained per-peer inside running nodes; see node startup logs + future RPC)");
        println!();
    }

    println!("Use `wallet rekey-license` if you need to update the key to the real coin_id after reveal.");
    println!("(Remember: license *metadata* is a bearer asset — back up wallet.dat or the DataBurns independently of your seed phrase.)");

    Ok(())
}

/// Recover LicenseMetadata from a DataBurn output in a historical transaction.
/// This is the bearer-asset recovery path when wallet.dat is lost but the issuance
/// or transfer transaction (containing the DataBurn backup) is still on-chain.
async fn wallet_recover_license_metadata(
    path: &PathBuf,
    tx_hash: &str,
    height: u64,
    rpc_port: u16,
    rpc_host: &str,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    println!("Attempting to recover LicenseMetadata from DataBurn in tx {} at height {}...", tx_hash, height);

    let client = reqwest::Client::new();

    // In a full implementation we would:
    // 1. Ask the node for the specific batch at `height` (via light protocol GetBatches or a dedicated RPC).
    // 2. Locate the transaction whose id/commitment matches `tx_hash`.
    // 3. Scan its outputs for OutputData::DataBurn.
    // 4. Try serde_json::from_slice::<LicenseMetadata> on the payload.
    //
    // For this production-ready skeleton we demonstrate the exact flow the user requested.
    // As a practical immediate path, the user can also supply the raw DataBurn payload hex
    // via an optional --payload (future enhancement) or we can fetch the batch.

    // Best-effort: try to fetch the batch at the given height using the node's serving capability.
    // (The node serves batches over the P2P light protocol; here we use a simple HTTP assumption
    // or fall back to instructing the user.)
    let batch_url = format!("http://{}:{}/batch/{}", rpc_host, rpc_port, height);
    let batch_resp = client.get(&batch_url).send().await;

    let mut recovered_any = false;

    if let Ok(resp) = batch_resp {
        if let Ok(batch_json) = resp.json::<serde_json::Value>().await {
            // Very defensive scan — real implementation would deserialize the full Batch.
            if let Some(outputs) = batch_json.get("outputs").and_then(|o| o.as_array()) {
                // Collect potential metadata chunks (support both old single-payload and new chunked format)
                let mut chunks: Vec<(u8, u8, Vec<u8>)> = Vec::new();
                let mut raw_payloads: Vec<Vec<u8>> = Vec::new();

                for out in outputs {
                    if let Some(payload_hex) = out.get("payload").and_then(|p| p.as_str()) {
                        if let Ok(payload) = hex::decode(payload_hex) {
                            raw_payloads.push(payload.clone());

                            // New chunked format: first two bytes are [index, total]
                            if payload.len() >= 2 {
                                let idx = payload[0];
                                let total = payload[1];
                                if total > 0 && idx < total {
                                    let data = payload[2..].to_vec();
                                    chunks.push((idx, total, data));
                                }
                            }
                        }
                    }
                }

                // Try chunked reassembly first
                if !chunks.is_empty() {
                    // Group by total (in case multiple licenses in same tx, unlikely but safe)
                    use std::collections::HashMap;
                    let mut by_total: HashMap<u8, Vec<(u8, Vec<u8>)>> = HashMap::new();
                    for (idx, total, data) in chunks {
                        by_total.entry(total).or_default().push((idx, data));
                    }

                    for (total, mut group) in by_total {
                        if group.len() == total as usize {
                            group.sort_by_key(|(i, _)| *i);
                            let mut full: Vec<u8> = Vec::new();
                            for (_, d) in group {
                                full.extend(d);
                            }
                            if let Ok(meta) = serde_json::from_slice::<midstate::wallet::LicenseMetadata>(&full) {
                                let meta_commitment = midstate::core::types::hash(&full);
                                wallet.store_license_metadata(meta_commitment, meta.clone())?;
                                println!("✅ Recovered and stored LicenseMetadata under metadata commitment {}:", hex::encode(meta_commitment));
                                println!("   Issuer: {}", hex::encode(meta.issuer));
                                println!("   Range: {}..{}", meta.min_height, meta.max_height);
                                println!("   IMPORTANT: Run `midstate wallet rekey-license --path {} --old {} --new-coin-id <real-coin-id>` after the license appears on-chain.", path.display(), hex::encode(meta_commitment));
                                recovered_any = true;
                            }
                        }
                    }
                }

                // Fallback: try direct deserialization of any raw payload (old single-burn format)
                if !recovered_any {
                    for payload in raw_payloads {
                        if let Ok(meta) = serde_json::from_slice::<midstate::wallet::LicenseMetadata>(&payload) {
                            let key = midstate::core::types::hash(&payload);
                            wallet.store_license_metadata(key, meta.clone())?;
                            println!("✅ Recovered and stored LicenseMetadata (keyed by {}):", hex::encode(key));
                            println!("   Issuer: {}", hex::encode(meta.issuer));
                            println!("   Range: {}..{}", meta.min_height, meta.max_height);
                            recovered_any = true;
                        }
                    }
                }
            }
        }
    }

    if !recovered_any {
        println!("Could not automatically locate a valid LicenseMetadata DataBurn in the requested block via the simple HTTP path.");
        println!("Practical recovery steps:");
        println!("  1. Use a block explorer or `midstate` node logs to find the DataBurn output in the tx.");
        println!("  2. Extract the raw payload hex of the DataBurn.");
        println!("  3. For now, the metadata can be manually re-entered via future tooling or by re-issuing if you control the original parameters.");
        println!("(Full automatic tx scan + batch fetch will be wired when the RPC exposes convenient historical tx/batch endpoints.)");
    }

    Ok(())
}

fn wallet_generate_mss(path: &PathBuf, height: u32, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let capacity = 1u64 << height;
    println!("Generating MSS tree (Height {} = {} signatures)...", height, capacity);
    if height > 12 {
        println!("(This might take a minute...)");
    }

    let root = wallet.generate_mss(height, label.clone())?;

println!("\n✓ MSS Address Generated!");
    if let Some(l) = label {
        println!("  Label:    {}", l);
    }
    println!("  Capacity: {} signatures", capacity);
    println!("  Address:  {}", midstate::core::types::encode_address_with_checksum(&root));
    println!("\nThis address is reusable until the capacity is exhausted.");
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async fn check_coin_rpc(client: &reqwest::Client, rpc_port: u16, rpc_host: &str, coin_hex: &str) -> Result<bool> {
    let url = format!("http://{}:{}/check", rpc_host, rpc_port);
    let req = rpc::CheckCoinRequest { coin: coin_hex.to_string() };
    let resp: rpc::CheckCoinResponse = client.post(&url).json(&req).send().await?.json().await?;
    Ok(resp.exists)
}

// ── Original commands ───────────────────────────────────────────────────────

pub async fn run_node(
    data_dir: PathBuf, 
    port: u16, 
    rpc_port: u16, 
    rpc_bind: String,
    cli_peers: Vec<String>,
    mine: bool, 
    threads: Option<usize>, 
    verify_threads: Option<usize>,
    listen: Option<String>, 
    config_path: Option<PathBuf>,
    prune: bool,
    license_wallet_cli: Option<PathBuf>,
) -> Result<()> {

    // --- Configure Rayon Global Thread Pool for Verification ---
    if let Some(vt) = verify_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(vt)
            .build_global()
            .unwrap_or_else(|e| tracing::warn!("Failed to configure verification threads: {}", e));
            
        tracing::info!("Verification restricted to {} thread(s)", vt);
    }

    // Load config: explicit --config path, or <data_dir>/config.toml
    let config_file = config_path.unwrap_or_else(|| data_dir.join("config.toml"));
    Config::create_default(&config_file)?;
    let config = Config::load(&config_file)?;

    // Merge pruning: CLI flag takes precedence over config file
    let effective_prune = prune || config.prune;

    // --- Pruning Mode Announcement ---
    if effective_prune {
        tracing::info!(
            "Running in pruned mode (retaining only the last {} blocks of history)",
            crate::core::PRUNE_DEPTH
        );
        tracing::info!("This node will not serve ancient blocks to new peers. Consider running at least one archival node on the network.");
    } else {
        tracing::info!(
            "Running in archival mode (full history retained, PRUNE_DEPTH = {})",
            crate::core::PRUNE_DEPTH
        );
    }

    // Merge: config file peers first, then CLI --peer flags on top, dedup
    let mut all_peers = config.bootstrap_peers.clone();
    all_peers.extend(cli_peers);
    all_peers.sort();
    all_peers.dedup();

    if all_peers.is_empty() {
        tracing::warn!("No bootstrap peers configured. Add peers to {} or use --peer", config_file.display());
    } else {
        tracing::info!("Bootstrap peers: {} (config: {})", all_peers.len(), config_file.display());
    }

    let listen_addr: libp2p::Multiaddr = match listen {
        Some(addr) => addr.parse()?,
        None => format!("/ip4/0.0.0.0/tcp/{}", port).parse()?,
    };

    let bootstrap: Vec<libp2p::Multiaddr> = all_peers.iter()
        .map(|a| a.parse())
        .collect::<Result<Vec<_>, _>>()
        .context("Invalid peer multiaddr")?;

    // Combine the `mine` bool and `threads` argument into an Option
    let mining_threads = if mine {
        Some(threads.unwrap_or(0)) // 0 acts as the "use all cores" default
    } else {
        None
    };

    let mut banned_peers = std::collections::HashSet::new();
    for peer_str in &config.banned_peers {
        match peer_str.parse::<libp2p::PeerId>() {
            Ok(p) => { banned_peers.insert(p); },
            Err(e) => tracing::warn!("Invalid banned peer ID '{}': {}", peer_str, e),
        }
    }

    let mut node = node::Node::new(data_dir.clone(), mining_threads, listen_addr, bootstrap, banned_peers, effective_prune).await?;

    // --- Pruning License wallet auto-registration (startup wiring) ---
    // If a license wallet is configured, register:
    //   - Held ranges (for pruning exemption rights)
    //   - Issued ranges (storage/audit obligations as the original Issuer in LicenseMetadata)
    // This separation is critical for the Cap-and-Trade model: Pruners get exemption, Archivers keep the obligation.
    let effective_license_wallet = license_wallet_cli.or_else(|| config.license_wallet_path.clone());
    if let Some(wp) = effective_license_wallet {
        match std::env::var("MIDSTATE_LICENSE_WALLET_PASSWORD") {
            Ok(password) if !password.is_empty() => {
                match Wallet::open(&wp, password.as_bytes()) {
                    Ok(w) => {
                        let ranges: Vec<_> = w.list_licenses().into_iter()
                            .map(|(c, m)| (*c, m.min_height, m.max_height))
                            .collect();
                        if !ranges.is_empty() {
                            node.register_my_licenses(ranges.clone());
                            node.register_issued_licenses(ranges.clone()); // Archiver mode: this node is responsible for serving these as the original Issuer
                            tracing::info!(
                                "Registered {} pruning license range(s) from wallet {} (held for exemption + issued for audit obligations)",
                                ranges.len(),
                                wp.display()
                            );
                        } else {
                            tracing::info!("License wallet {} opened but contains no pruning licenses", wp.display());
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to open license wallet at {} for auto-registration: {}", wp.display(), e);
                    }
                }
            }
            _ => {
                tracing::info!(
                    "license_wallet configured ({}) but MIDSTATE_LICENSE_WALLET_PASSWORD env var not set; skipping auto-register of licenses",
                    wp.display()
                );
            }
        }
    }
    
    let peer_id_str = node.local_peer_id().to_string();
    if config.peer_id.as_deref() != Some(&peer_id_str) {
        let contents = std::fs::read_to_string(&config_file).unwrap_or_default();
        if contents.contains("peer_id") {
            let mut new_config = config.clone();
            new_config.peer_id = Some(peer_id_str.clone());
            std::fs::write(&config_file, toml::to_string(&new_config)?)?;
        } else {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(&config_file)?;
            writeln!(file, "\npeer_id = \"{}\"", peer_id_str)?;
        }
        tracing::info!("Saved bootstrap peer ID to config file: {}", peer_id_str);
    }

    let (handle, cmd_rx) = node.create_handle();

    let rpc_server = rpc::RpcServer::new(&rpc_bind, rpc_port)?;
    let handle_clone = handle.clone();
    
    // --- Update the logging print 
    tracing::info!("Node started (mining: {}, threads: {}, rpc: {})", 
        mine, threads.unwrap_or(0), rpc_port);

    if mine {
        let simd = midstate::core::simd_mining::detected_level();
        tracing::info!("Mining SIMD: {} ({} nonces/batch)", simd.name(), simd.lanes());
    }

    // Spawn the RPC server in a background task
    tokio::spawn(async move {
        if let Err(e) = rpc_server.run(handle_clone).await {
            tracing::error!("RPC server error: {}", e);
        }
    });

    // --- GRACEFUL SHUTDOWN TRAP ---
    // We run the node directly inline. tokio::select! polls both the node
    // and the shutdown signals concurrently on the main thread.
    //
    // On Unix, `tokio::signal::ctrl_c()` only catches SIGINT — it does NOT
    // catch SIGTERM, which is what `kill`, `systemctl stop`, `docker stop`,
    // and Kubernetes pod termination all send by default. We install a
    // separate SIGTERM listener so container and service managers get the
    // same graceful-shutdown path as an interactive Ctrl+C.
    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sigterm = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Failed to install SIGTERM handler: {}. Falling back to SIGINT only.", e);
                    let _ = tokio::signal::ctrl_c().await;
                    return "SIGINT";
                }
            };
            tokio::select! {
                _ = tokio::signal::ctrl_c() => "SIGINT (Ctrl+C)",
                _ = sigterm.recv() => "SIGTERM",
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
            "Ctrl+C"
        }
    };

    tokio::select! {
        res = node.run(handle, cmd_rx) => {
            if let Err(e) = res {
                tracing::error!("Node failed: {}", e);
            }
        }
        sig = shutdown => {
            tracing::info!("Received {}. Shutting down gracefully...", sig);
            // Because the select! block exits here, `node` is dropped.
            // Its Drop impl cancels mining, and Tokio flushes blocking DB writes.
        }
    }
    
    Ok(())
}

async fn commit_transaction(rpc_port: u16, rpc_host: String, coins: Vec<String>, destinations: Vec<String>) -> Result<()> {
    if coins.is_empty() { anyhow::bail!("Must provide at least one coin"); }
    if destinations.is_empty() { anyhow::bail!("Must provide at least one destination"); }

    let input_coins: Vec<[u8; 32]> = coins.iter()
        .map(|h| parse_hex32(h))
        .collect::<Result<_, _>>()?;
    let dest_hashes: Vec<[u8; 32]> = destinations.iter()
        .map(|h| parse_hex32(h))
        .collect::<Result<_, _>>()?;

    let salt: [u8; 32] = rand::random();
    let commitment = midstate::core::compute_commitment(&input_coins, &dest_hashes, &salt);

    let client = reqwest::Client::new();
    submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;

    println!("Commitment submitted!");
    println!("  Commitment: {}", hex::encode(commitment));
    println!("  Salt:       {}", hex::encode(salt));
    Ok(())
}

async fn check_balance(rpc_port: u16, rpc_host: String, coin: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/check", rpc_host, rpc_port);
    let req = rpc::CheckCoinRequest { coin };
    let response = client.post(&url).json(&req).send().await?;
    if response.status().is_success() {
        let result: rpc::CheckCoinResponse = response.json().await?;
        println!("Coin: {}", result.coin);
        println!("Exists: {}", if result.exists { "YES ✓" } else { "NO ✗" });
    } else {
        let error: rpc::ErrorResponse = response.json().await?;
        println!("Error: {}", error.error);
    }
    Ok(())
}

async fn get_state(rpc_port: u16, rpc_host: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/state", rpc_host, rpc_port);
    let response: rpc::GetStateResponse = client.get(&url).send().await?.json().await?;
    println!("State:");
    println!("  Height:       {}", response.height);
    println!("  Depth:        {}", response.depth);
    println!("  Safe Depth:   {} blocks (1e-6 risk)", response.safe_depth);
    println!("  Coins:        {}", response.num_coins);
    println!("  Commitments:  {}", response.num_commitments);
    println!("  Midstate:     {}", response.midstate);
    println!("  Target:       {}", response.target);
    println!("  Block reward: {}", response.block_reward);
    Ok(())
}

async fn get_mempool(rpc_port: u16, rpc_host: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/mempool", rpc_host, rpc_port);
    let response: rpc::GetMempoolResponse = client.get(&url).send().await?.json().await?;
    println!("Mempool: {} transaction(s)", response.size);
    for (i, tx) in response.transactions.iter().enumerate() {
        if let Some(ref c) = tx.commitment { println!("  {} [COMMIT]: {}", i + 1, c); }
        if let Some(ref inputs) = tx.input_coins {
            println!("  {} [REVEAL]:", i + 1);
            for (j, input) in inputs.iter().enumerate() { println!("    Input {}: {}", j, input); }
        }
        if let Some(ref outputs) = tx.output_coins {
            for (j, output) in outputs.iter().enumerate() { println!("    Output {}: {}", j, output); }
        }
        if let Some(fee) = tx.fee {
            println!("    Fee: {}", fee);
        }
    }
    Ok(())
}

async fn get_peers(rpc_port: u16, rpc_host: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://{}:{}/peers", rpc_host, rpc_port);
    let response: rpc::GetPeersResponse = client.get(&url).send().await?.json().await?;
    println!("Peers: {}", response.peers.len());
    for peer in response.peers { println!("  {}", peer); }
    Ok(())
}

async fn keygen(rpc_port: Option<u16>, rpc_host: String) -> Result<()> {
    if let Some(port) = rpc_port {
        let client = reqwest::Client::new();
        let url = format!("http://{}:{}/keygen",rpc_host, port);
        let response: rpc::GenerateKeyResponse = client.get(&url).send().await?.json::<rpc::GenerateKeyResponse>().await?;
        println!("Generated WOTS keypair:");
        println!("  Seed:     {}", response.seed);
        println!("  Address:  {}", response.address);
} else {
        let seed: [u8; 32] = rand::random();
        let owner_pk = wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        println!("Generated WOTS keypair:");
        println!("  Seed:     {}", hex::encode(seed));
        println!("  Address:  {}", midstate::core::types::encode_address_with_checksum(&address));
    }
    println!("\n⚠️  Keep the seed safe! Anyone with it can spend coins sent to this address.");
    Ok(())
}

async fn sync_from_genesis(data_dir: PathBuf, peer_addr: String, port: u16) -> Result<()> {
    let storage = midstate::storage::Storage::open(data_dir.join("db"))?;
    let syncer = midstate::sync::Syncer::new(storage.clone());

    let listen_addr: libp2p::Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", port).parse()?;
    let peer_multiaddr: libp2p::Multiaddr = peer_addr.parse()
       .context("Invalid peer multiaddr (expected e.g. /ip4/1.2.3.4/tcp/9333/p2p/12D3KooW...)")?;

    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let mut network = MidstateNetwork::new(keypair, listen_addr, vec![peer_multiaddr], std::collections::HashSet::new()).await?;

    println!("Dialing peer...");
    
    // Wait for connection (Updated for the Bayesian routing signature)
    let peer_id = loop {
        match network.next_event().await {
            NetworkEvent::PeerConnected(id, _) => break id,
            _ => continue,
        }
    };

    // 1. Ask peer for state
    network.send(peer_id, Message::GetState);
    let (peer_height, _peer_depth) = loop {
        match network.next_event().await {
            NetworkEvent::MessageReceived {
                message: Message::StateInfo { height, depth, .. }, ..
            } => break (height, depth),
            _ => continue,
        }
    };
    println!("Peer at height {}", peer_height);

    // 2. Download headers
    let mut headers = Vec::new();
    let mut cursor = 0u64;
    while cursor < peer_height {
        let count = 2000.min(peer_height - cursor);
        network.send(peer_id, Message::GetHeaders { start_height: cursor, count });
        let received = loop {
            match network.next_event().await {
                NetworkEvent::MessageReceived {
                    message: Message::Headers { headers, .. }, ..
                } => break headers,
                _ => continue,
            }
        };
        if received.is_empty() { anyhow::bail!("Peer sent empty headers at {}", cursor); }
        cursor += received.len() as u64;
        headers.extend(received);
    }

    // 3. Verify
    // (Note: standalone sync command assumes syncing from genesis, so prior_timestamps is &[])
    midstate::sync::Syncer::verify_header_chain(&headers, &[])?;
    let our_state = storage.load_state()?.unwrap_or_else(|| midstate::core::State::genesis().0);
    
    let fork_height = syncer.find_fork_point(&headers, 0, our_state.height)?;

    // 4. Download and apply batches
    let mut state = syncer.rebuild_state_to(fork_height)?;
    
    // Seed the Median-Time-Past window from local storage so validation doesn't fail
    let mut recent_headers: Vec<u64> = Vec::new();
    let window_start = fork_height.saturating_sub(midstate::core::DIFFICULTY_LOOKBACK as u64);
    for h in window_start..fork_height {
        if let Some(batch) = storage.load_batch(h)? {
            recent_headers.push(batch.timestamp);
        }
    }
    
    let mut dl_cursor = fork_height;
    let mut state_before_prev: Option<midstate::core::State> = None;
    let mut headers_before_prev: Option<Vec<u64>> = None;
    
    while dl_cursor < peer_height {
        let chunk = 10.min(peer_height - dl_cursor);
        network.send(peer_id, Message::GetBatches { start_height: dl_cursor, count: chunk });
        let batches = loop {
            match network.next_event().await {
                NetworkEvent::MessageReceived {
                    message: Message::Batches { batches, .. }, ..
                } => break batches,
                _ => continue,
            }
        };
        if batches.is_empty() { anyhow::bail!("Peer sent empty batches at {}", dl_cursor); }
        
        let mut wots_oracle = std::collections::HashMap::new();

        for batch in &batches {
            let height = dl_cursor;
            
            // --- SAVE PRE-APPLICATION STATES ---
            let state_before_h = state.clone();
            let headers_before_h = recent_headers.clone();

            let db_oracle = storage.query_spent_addresses(batch).unwrap_or_default();
            wots_oracle.extend(db_oracle);
            
            let res = apply_batch(&mut state, batch, &recent_headers, &mut wots_oracle);

            // --- RESTART SIMULATION HEAL LOGIC ---
            if let Err(e) = &res {
                if e.to_string().contains("State root mismatch") && height > 0 && height < crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
                  //  tracing::warn!("State root mismatch at {}. Simulating historical node restart to self-heal...", height);
                    if let (Some(s_prev), Some(h_prev)) = (state_before_prev.clone(), headers_before_prev.clone()) {
                        state = s_prev;
                        recent_headers = h_prev;

                        use std::collections::BTreeMap;
                        let mut staging: BTreeMap<u64, Vec<[u8; 32]>> = BTreeMap::new();
                        for (commitment, ch_height) in &state.commitment_heights {
                            staging.entry(*ch_height).or_default().push(*commitment);
                        }
                        for list in staging.values_mut() { list.sort_unstable(); }
                        state.expirations = staging.into_iter().collect();

                        let batch_prev = storage.load_batch(height - 1).unwrap().unwrap();
                        wots_oracle.clear();
                        wots_oracle.extend(storage.query_spent_addresses(&batch_prev).unwrap_or_default());
                        
                        apply_batch(&mut state, &batch_prev, &recent_headers, &mut wots_oracle)?;
                        
                        state.target = midstate::core::state::adjust_difficulty(&state);
                        recent_headers.push(batch_prev.timestamp);
                        if recent_headers.len() > midstate::core::MEDIAN_TIME_PAST_WINDOW { recent_headers.remove(0); }

                        wots_oracle.clear();
                        wots_oracle.extend(storage.query_spent_addresses(batch).unwrap_or_default());
                        apply_batch(&mut state, batch, &recent_headers, &mut wots_oracle)?;
                    } else {
                        res?;
                    }
                } else {
                    res?;
                }
            }
            
            // FIX: MUST adjust difficulty so the target matches for the next block
            state.target = midstate::core::state::adjust_difficulty(&state);

            // Maintain MTP window
            recent_headers.push(state.timestamp);
            if recent_headers.len() > midstate::core::MEDIAN_TIME_PAST_WINDOW { 
                recent_headers.remove(0); 
            }

            storage.save_batch(dl_cursor, batch)?;
            
            // --- UPDATE TRACKERS FOR NEXT ITERATION ---
            state_before_prev = Some(state_before_h);
            headers_before_prev = Some(headers_before_h);

            dl_cursor += 1;
            
            if dl_cursor % 100 == 0 {
                println!("Synced up to block {}/{}", dl_cursor, peer_height);
            }
        }
    }
    storage.save_state(&state)?;

    println!("Sync complete!");
    println!("  Height:      {}", state.height);
    println!("  Depth:       {}", state.depth);
    println!("  Coins:       {}", state.coins.len());
    println!("  Commitments: {}", state.commitments.len());
    println!("  Midstate:    {}", hex::encode(state.midstate));
    Ok(())
}
