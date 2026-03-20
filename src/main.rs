use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use midstate::*;
use midstate::compute_address;
use midstate::wallet::{self, Wallet};
use midstate::core::wots;
use midstate::core::state::apply_batch;
use midstate::network::{MidstateNetwork, NetworkEvent, Message};
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

            bootstrap_peers = ["/ip4/134.199.148.215/tcp/9333/p2p/12D3KooWRACBS3RTvszkfPsq4nMw1sZUyNuGAD3SswqsybnoyDcA"]
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
        #[arg(long)]
        verify_threads: Option<usize>,
        #[arg(long)]
        listen: Option<String>,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        isbootstrap: bool,
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
        coin: String,
        #[arg(long)]
        bytecode: String,
        #[arg(long)]
        inputs: String,
        #[arg(long)]
        burn_data: Option<String>,
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
    },
    SpendStark {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: String,
        #[arg(long)]
        prove_value: u64, // The value we are proving is in the 64-bit range
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
    },
    SpendConfidential {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        rpc_host: String,
        #[arg(long)]
        coin: String,
        #[arg(long)]
        to: String,
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

fn parse_output_spec(s: &str) -> Result<([u8; 32], u64)> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        anyhow::bail!("expected format <address>:<value>, got: {}", s);
    }
    
    // Use the new flexible parser to validate the checksum
    let pk = midstate::core::types::parse_address_flexible(parts[0])
        .map_err(|e| anyhow::anyhow!(e))?;
        
    let value: u64 = parts[1].parse()
        .map_err(|_| anyhow::anyhow!("invalid value: {}", parts[1]))?;
    if value == 0 {
        anyhow::bail!("value must be > 0");
    }
    Ok((pk, value))
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
        Command::Node { data_dir, port, rpc_port, rpc_bind, peer, mine, threads, verify_threads, listen, config, isbootstrap } => {
            run_node(data_dir, port, rpc_port, rpc_bind, peer, mine, threads, verify_threads, listen, config, isbootstrap).await
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
    }
}

// ── Wallet commands ─────────────────────────────────────────────────────────
async fn wallet_scan(path: &PathBuf, rpc_port: u16, rpc_host: String) -> Result<()> {
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
    let start = wallet.data.last_scan_height;

    if start >= chain_height {
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
                if let Some(coin_id) = wallet.import_scanned(cb.address, cb.value, cb.salt)? {
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
                            if let Some(coin_id) = wallet.import_scanned(out.address(), out.value(), out.salt())? {
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
    coin_ref: String,
    bytecode_hex: String,
    inputs_arg: String,
    burn_data: Option<String>,
    to_args: Vec<String>,
    timeout_secs: u64,
) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let target_coin_id = wallet.resolve_coin(&coin_ref)?;
    let target_coin = wallet.find_coin(&target_coin_id).cloned()
        .ok_or_else(|| anyhow::anyhow!("Coin not found in local wallet"))?;

    let bytecode = hex::decode(&bytecode_hex).context("Invalid bytecode hex")?;
    let script_address = midstate::core::types::hash(&bytecode);
    if script_address != target_coin.address {
        anyhow::bail!("Bytecode hash does not match coin address");
    }

    if to_args.is_empty() && burn_data.is_none() { 
        anyhow::bail!("Must specify at least one output via --to or --burn-data"); 
    }

    // --- 1. SMART CO-SPEND: Gather all sibling coins ---
    let is_mss = wallet.data.mss_keys.iter().any(|k| midstate::core::compute_address(&k.master_pk) == target_coin.address);
    let mut spending_coins = vec![target_coin.clone()];
    
    if !is_mss {
        for c in wallet.data.coins.iter() {
            if c.address == target_coin.address && c.coin_id != target_coin_id {
                spending_coins.push(c.clone());
            }
        }
        if spending_coins.len() > 1 {
            println!("Auto-merging {} sibling coin(s) to protect WOTS key...", spending_coins.len() - 1);
        }
    }

    let in_sum: u64 = spending_coins.iter().map(|c| c.value).sum();
    // ----------------------------------------------------

    let mut outputs = Vec::new();
    let mut out_sum = 0u64;
    
    for arg in &to_args {
        let (addr, val) = parse_output_spec(arg)?;
        let salt: [u8; 32] = rand::random();
        outputs.push(OutputData::Standard { address: addr, value: val, salt });
        out_sum += val;
    }

    if let Some(burn_str) = burn_data {
        let parts: Vec<&str> = burn_str.splitn(2, ':').collect();
        if parts.len() != 2 { anyhow::bail!("Format: <hex_payload>:<value>"); }
        let payload = hex::decode(parts[0]).context("Invalid burn payload hex")?;
        let val: u64 = parts[1].parse().context("Invalid burn value")?;
        outputs.push(OutputData::DataBurn { payload, value_burned: val });
        out_sum += val;
    }

    if in_sum <= out_sum {
        anyhow::bail!("Total input value ({}) must exceed output value ({}) to pay fee", in_sum, out_sum);
    }

    // --- 2. AUTO-CHANGE GENERATION ---
    // If the inputs (including siblings) vastly exceed the requested outputs,
    // we generate change to prevent paying a massive fee. We assume a base 100-unit fee.
    let base_fee = 100;
    let mut change_seeds = Vec::new();
    
    if in_sum > out_sum + base_fee {
        let change_val = in_sum - out_sum - base_fee;
        let change_denoms = midstate::core::decompose_value(change_val);
        for denom in change_denoms {
            let seed = wallet.next_wots_seed();
            let pk = midstate::core::wots::keygen(&seed);
            let addr = midstate::core::compute_address(&pk);
            let salt: [u8; 32] = rand::random();
            let idx = outputs.len();
            outputs.push(OutputData::Standard { address: addr, value: denom, salt });
            change_seeds.push((idx, seed));
        }
        println!("Auto-generated {} change output(s) totaling {}", change_seeds.len(), change_val);
    }
    // ---------------------------------

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

    // Generate the witness stack (only needs to be solved once!)
    let mut stack_items = Vec::new();
    for token in inputs_arg.split(',').filter(|s| !s.is_empty()) {
        if token.starts_with("AUTO:") {
            let pk_hex = token.strip_prefix("AUTO:").unwrap();
            let pk = parse_hex32(pk_hex)?;
            println!("Auto-solving signature for {}...", hex::encode(&pk));
            stack_items.push(wallet.auto_sign(&pk, &commitment)?);
        } else {
            stack_items.push(hex::decode(token).context("Invalid hex in --inputs")?);
        }
    }
    let witness_string = stack_items.iter().map(hex::encode).collect::<Vec<_>>().join(",");

    // --- 3. DUPLICATE WITNESS FOR SIBLINGS ---
    let mut rpc_inputs = Vec::new();
    let mut rpc_signatures = Vec::new();

    for coin in &spending_coins {
        rpc_inputs.push(rpc::InputRevealJson {
            bytecode: bytecode_hex.clone(),
            value: coin.value,
            salt: hex::encode(coin.salt),
        });
        // The exact same witness string satisfies all sibling coins
        rpc_signatures.push(witness_string.clone());
    }
    // -----------------------------------------

    let reveal_req = rpc::SendTransactionRequest {
        inputs: rpc_inputs,
        signatures: rpc_signatures,
        outputs: outputs.iter().map(|o| match o {
            OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
            OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
        }).collect(),
        salt: hex::encode(tx_salt),
    };
    
    println!("Submitting Phase 2 Reveal...");
    let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
    let resp = client.post(&reveal_url).json(&reveal_req).send().await?;
    if !resp.status().is_success() {
        let err: rpc::ErrorResponse = resp.json().await?;
        anyhow::bail!("Reveal failed: {}", err.error);
    }

    // --- 4. CLEANUP WALLET STATE ---
    // Remove all spent coins
    wallet.data.coins.retain(|c| !input_coin_ids.contains(&c.coin_id));
    
    // Save new change coins
    for (idx, seed) in change_seeds {
        let out = &outputs[idx];
        if let OutputData::Standard { address, value, salt } = out {
            let owner_pk = midstate::core::wots::keygen(&seed);
            wallet.data.coins.push(midstate::wallet::WalletCoin {
                seed,
                owner_pk,
                address: *address,
                value: *value,
                salt: *salt,
                coin_id: out.coin_id().unwrap(),
                label: Some(format!("change ({})", value)),
                wots_signed: false,
                commitment: None,
            });
        }
    }
    wallet.save()?;
    // -------------------------------

    println!("✓ Custom script spent successfully!");
    Ok(())
}
#[allow(unused_variables)]
async fn wallet_spend_stark(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    coin_ref: String,
    prove_value: u64,
    to_args: Vec<String>,
    timeout_secs: u64,
) -> Result<()> {
    #[cfg(not(feature = "stark-prover"))]
    {
        anyhow::bail!("CLI must be compiled with --features stark-prover to generate STARK proofs.\nRun: cargo run --release --features stark-prover -- wallet spend-stark ...");
    }

    #[cfg(feature = "stark-prover")]
    {
        let password = read_password("Password: ")?;
        let mut wallet = Wallet::open(path, &password)?;
        let client = reqwest::Client::new();

        let target_coin_id = wallet.resolve_coin(&coin_ref)?;
        let target_coin = wallet.find_coin(&target_coin_id).cloned()
            .ok_or_else(|| anyhow::anyhow!("Coin not found in local wallet"))?;

        // Prevent the user from wasting CPU cycles generating a proof 
        // if the network hasn't reached the hard fork height yet.
        if wallet.data.last_scan_height < midstate::core::types::STARK_ACTIVATION_HEIGHT {
            anyhow::bail!(
                "Network has not reached STARK activation height ({}). Current scanned height is {}. Transaction would be rejected by consensus.", 
                midstate::core::types::STARK_ACTIVATION_HEIGHT, 
                wallet.data.last_scan_height
            );
        }

        if to_args.is_empty() { 
            anyhow::bail!("Must specify at least one output via --to"); 
        }

        println!("⚙️ Generating STARK proof for value {} (this takes a few seconds)...", prove_value);
        let prover = midstate::core::stark::prover::RangeProofProver::new(prove_value);
        let (pub_inputs, proof) = tokio::task::spawn_blocking(move || {
            prover.generate_proof()
        }).await.unwrap().map_err(|e| anyhow::anyhow!(e))?;
        
        let proof_bytes = proof.to_bytes();
        let pub_inputs_bytes = pub_inputs.value.as_int().to_le_bytes().to_vec();
        println!("✅ STARK Proof generated! Size: {} bytes", proof_bytes.len());

        // 1. Build the exact bytecode that locks this coin
        // Script: PUSH_HEX <program_id> VERIFY_STARK PUSH_HEX <owner_pk> CHECKSIGVERIFY PUSH_INT 1
        let mut bc = Vec::new();
        midstate::core::script::push_data(&mut bc, &*midstate::core::stark::RANGE_PROOF_64);
        bc.push(midstate::core::script::OP_VERIFY_STARK);
        midstate::core::script::push_data(&mut bc, &target_coin.owner_pk);
        bc.push(midstate::core::script::OP_CHECKSIGVERIFY);
        midstate::core::script::push_int(&mut bc, 1);

        let script_address = midstate::core::types::hash(&bc);
        if script_address != target_coin.address {
            anyhow::bail!("Coin is not locked by the expected STARK script. Address mismatch.");
        }

        let bytecode_hex = hex::encode(&bc);

        // 2. Parse outputs
        let mut outputs = Vec::new();
        let mut out_sum = 0u64;
        for arg in &to_args {
            let (addr, val) = parse_output_spec(arg)?;
            let salt: [u8; 32] = rand::random();
            outputs.push(midstate::core::OutputData::Standard { address: addr, value: val, salt });
            out_sum += val;
        }

        if target_coin.value <= out_sum {
            anyhow::bail!("Input value ({}) must exceed output value ({}) to pay fee", target_coin.value, out_sum);
        }

        let input_coin_ids = vec![target_coin_id];
        let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        
        let tx_salt: [u8; 32] = rand::random();
        let commitment = midstate::core::compute_commitment(&input_coin_ids, &output_commit_hashes, &tx_salt);

        // 3. Phase 1: Commit
        println!("Submitting Phase 1 Commit...");
        submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;

        if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), timeout_secs).await {
            anyhow::bail!("Timed out waiting for Commit to be mined.");
        }
        println!("✓ Commit mined!");

        // 4. Phase 2: Reveal (Constructing the Witness Stack)
        // Witness stack must match script execution order: [sig, pub_inputs, proof]
        let sig_bytes = wallet.auto_sign(&target_coin.owner_pk, &commitment)?;
        
        let witness_stack = vec![
            hex::encode(sig_bytes),
            hex::encode(pub_inputs_bytes),
            hex::encode(proof_bytes), // The 128KB payload!
        ].join(",");

        let reveal_req = rpc::SendTransactionRequest {
            inputs: vec![rpc::InputRevealJson {
                bytecode: bytecode_hex,
                value: target_coin.value,
                salt: hex::encode(target_coin.salt),
            }],
            signatures: vec![witness_stack],
            outputs: outputs.iter().map(|o| match o {
                midstate::core::OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                    address: hex::encode(address),
                    value: *value,
                    salt: hex::encode(salt),
                },
                _ => unreachable!(),
            }).collect(),
            salt: hex::encode(tx_salt),
        };
        
        println!("Submitting Phase 2 Reveal (Pushing massive STARK payload to mempool)...");
        let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
        let resp = client.post(&reveal_url).json(&reveal_req).send().await?;
        if !resp.status().is_success() {
            let err: rpc::ErrorResponse = resp.json().await?;
            anyhow::bail!("Reveal failed: {}", err.error);
        }

        // Cleanup
        wallet.data.coins.retain(|c| c.coin_id != target_coin_id);
        wallet.save()?;

        println!("✓ STARK Transaction sent successfully!");
        Ok(())
    }
}

async fn wallet_spend_confidential(
    path: &PathBuf,
    rpc_port: u16,
    rpc_host: String,
    coin_ref: String,
    to_arg: String,
    timeout_secs: u64,
) -> Result<()> {
    #[cfg(not(feature = "stark-prover"))]
    {
        anyhow::bail!("CLI must be compiled with --features stark-prover");
    }

    #[cfg(feature = "stark-prover")]
    {
        let password = read_password("Password: ")?;
        let mut wallet = Wallet::open(path, &password)?;
        let client = reqwest::Client::new();

        let target_coin_id = wallet.resolve_coin(&coin_ref)?;
        let target_coin = wallet.find_coin(&target_coin_id).cloned()
            .ok_or_else(|| anyhow::anyhow!("Coin not found in local wallet"))?;

        if wallet.data.last_scan_height < midstate::core::types::STARK_ACTIVATION_HEIGHT {
            anyhow::bail!("Network has not reached STARK activation height.");
        }

        let (recipient_addr, send_value) = parse_output_spec(&to_arg)?;

        // Fee Deduction Framework (Fix 4)
        let total_value = target_coin.value;
        // fees are dynamic, but we use a fixed minimum fee for simple sends.
        let network_fee = 1500;
        let change_value = total_value.saturating_sub(send_value).saturating_sub(network_fee);

        println!("⚙️ Generating CONFIDENTIAL TRANSFER proof...");

        let blind1: [u8; 32] = rand::random();
        let blind2: [u8; 32] = rand::random();

        // Include the explicit fee in the public inputs (Fix 4)
        let prover = midstate::core::confidential::ct_prover::ConfidentialTransferProver::new(
            send_value, blind1, change_value, blind2, network_fee
        );

        let (pub_inputs, proof) = tokio::task::spawn_blocking(move || {
            prover.generate_proof()
        }).await.unwrap().map_err(|e| anyhow::anyhow!(e))?;

        let proof_bytes = proof.to_bytes();
        println!("✅ STARK Proof generated! Size: {} bytes", proof_bytes.len());

        // Serialize public inputs exactly as verify_confidential_transfer expects (80 bytes)
        let mut pub_inputs_bytes = Vec::with_capacity(80);
        for w in &pub_inputs.commitment1 { pub_inputs_bytes.extend_from_slice(&w.to_le_bytes()); }
        for w in &pub_inputs.commitment2 { pub_inputs_bytes.extend_from_slice(&w.to_le_bytes()); }
        pub_inputs_bytes.extend_from_slice(&pub_inputs.input_sum.to_le_bytes());
        pub_inputs_bytes.extend_from_slice(&pub_inputs.fee.to_le_bytes());  

        // 1. Validate the coin is locked by the Confidential Transfer script
        let mut bc = Vec::new();
        midstate::core::script::push_data(&mut bc, &*midstate::core::stark::CONFIDENTIAL_TRANSFER);
        bc.push(midstate::core::script::OP_VERIFY_STARK);
        midstate::core::script::push_data(&mut bc, &target_coin.owner_pk);
        bc.push(midstate::core::script::OP_CHECKSIGVERIFY);
        midstate::core::script::push_int(&mut bc, 1);

        let script_address = midstate::core::types::hash(&bc);
        if script_address != target_coin.address {
            anyhow::bail!("Coin is not locked by the STARK Confidential Transfer script. Address mismatch.");
        }
        let bytecode_hex = hex::encode(&bc);

        // 2. Build Confidential Outputs using the generated commitments
        let mut c1_bytes = [0u8; 32];
        let mut c2_bytes = [0u8; 32];
        for (i, w) in pub_inputs.commitment1.iter().enumerate() {
            c1_bytes[i*4..(i+1)*4].copy_from_slice(&w.to_le_bytes());
        }
        for (i, w) in pub_inputs.commitment2.iter().enumerate() {
            c2_bytes[i*4..(i+1)*4].copy_from_slice(&w.to_le_bytes());
        }

        let salt1: [u8; 32] = rand::random();
        let salt2: [u8; 32] = rand::random();

        let change_seed = wallet.next_wots_seed();
        let change_pk = midstate::core::wots::keygen(&change_seed);
        let change_addr = midstate::core::types::compute_address(&change_pk);

        let outputs = vec![
            midstate::core::OutputData::Confidential {
                address: recipient_addr,
                commitment: c1_bytes,
                salt: salt1,
            },
            midstate::core::OutputData::Confidential {
                address: change_addr,
                commitment: c2_bytes,
                salt: salt2,
            },
        ];

        let input_coin_ids = vec![target_coin_id];
        let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let tx_salt: [u8; 32] = rand::random();
        let commitment = midstate::core::types::compute_commitment(&input_coin_ids, &output_commit_hashes, &tx_salt);

        // 3. Phase 1: Commit
        println!("Submitting Phase 1 Commit...");
        submit_commit(&client, rpc_port, &rpc_host, &commitment).await?;

        if !wait_for_commit_mined(&client, rpc_port, &rpc_host, &hex::encode(commitment), timeout_secs).await {
            anyhow::bail!("Timed out waiting for Commit to be mined.");
        }
        println!("✓ Commit mined!");

        // 4. Phase 2: Reveal
        let sig_bytes = wallet.auto_sign(&target_coin.owner_pk, &commitment)?;
        
        // Witness stack MUST exactly match the script execution popping order: [sig, pub_inputs, proof]
        let witness_stack = vec![
            hex::encode(sig_bytes),
            hex::encode(pub_inputs_bytes),
            hex::encode(proof_bytes),
        ].join(",");

        let reveal_req = midstate::rpc::SendTransactionRequest {
            inputs: vec![midstate::rpc::InputRevealJson {
                bytecode: bytecode_hex,
                value: target_coin.value,
                salt: hex::encode(target_coin.salt),
            }],
            signatures: vec![witness_stack],
            outputs: outputs.iter().map(|o| match o {
                midstate::core::OutputData::Confidential { address, commitment, salt } => midstate::rpc::OutputDataJson::Confidential {
                    address: hex::encode(address),
                    commitment: hex::encode(commitment),
                    salt: hex::encode(salt),
                },
                _ => unreachable!(),
            }).collect(),
            salt: hex::encode(tx_salt),
        };
        
        println!("Submitting Phase 2 Reveal (Pushing Confidential Transfer payload)...");
        let reveal_url = format!("http://{}:{}/send", rpc_host, rpc_port);
        let resp = client.post(&reveal_url).json(&reveal_req).send().await?;
        if !resp.status().is_success() {
            let err: midstate::rpc::ErrorResponse = resp.json().await?;
            anyhow::bail!("Reveal failed: {}", err.error);
        }

// Cleanup & Save Change locally so we know what we own
        wallet.data.coins.retain(|c| c.coin_id != target_coin_id);
        wallet.data.coins.push(midstate::wallet::WalletCoin {
            seed: change_seed,
            owner_pk: change_pk,
            address: change_addr,
            value: change_value,
            salt: salt2,
            coin_id: outputs[1].coin_id().unwrap(),
            label: Some(format!("confidential change (hidden)")),
            wots_signed: false,
            // FIX 2: Preserve the blinding hash for spending later
            commitment: Some(c2_bytes), 
        });
        wallet.save()?;

        println!("✓ Confidential STARK Transaction sent successfully!");
        Ok(())
    }
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
        WalletAction::Scan { path, rpc_port, rpc_host } => wallet_scan(&path, rpc_port, rpc_host).await,
        WalletAction::Send { path, rpc_port, rpc_host, coin, to, timeout, private } => {
            wallet_send(&path, rpc_port, rpc_host, coin, to, timeout, private).await
        }
        WalletAction::SpendConfidential { path, rpc_port, rpc_host, coin, to, timeout } => {
            wallet_spend_confidential(&path, rpc_port, rpc_host, coin, to, timeout).await
        }
        WalletAction::SpendScript { path, rpc_port, rpc_host, coin, bytecode, inputs, burn_data, to, timeout } => {
            wallet_spend_script(&path, rpc_port, rpc_host, coin, bytecode, inputs, burn_data, to, timeout).await
        }
        WalletAction::SpendStark { path, rpc_port, rpc_host, coin, prove_value, to, timeout } => {
            wallet_spend_stark(&path, rpc_port, rpc_host, coin, prove_value, to, timeout).await
        }
        WalletAction::Import { path, seed, value, salt, label } => {
            wallet_import(&path, &seed, value, &salt, label)
        }
        WalletAction::Export { path, coin } => wallet_export(&path, &coin),
        WalletAction::Pending { path } => wallet_pending(&path),
        WalletAction::Reveal { path, rpc_port, rpc_host, commitment } => {
            wallet_reveal(&path, rpc_port, rpc_host, commitment).await
        }
        WalletAction::History { path, count } => wallet_history(&path, count),
        WalletAction::ImportRewards { path, coinbase_file, data_dir } => {
            wallet_import_rewards(&path, &coinbase_file, &data_dir)
        }
        WalletAction::GenerateMss { path, height, label } => {
            wallet_generate_mss(&path, height, label)
        }
        WalletAction::Mix { path, rpc_port, rpc_host, denomination, coin, join, pay_fee, timeout } => {
            wallet_mix(&path, rpc_port, rpc_host, denomination, coin, join, pay_fee, timeout).await
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

    if wallet.coin_count() > 0 {
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


    let recipient_specs: Vec<([u8; 32], u64)> = to_args.iter()
        .map(|s| parse_output_spec(s))
        .collect::<Result<Vec<_>>>()?;

    let total_send: u64 = recipient_specs.iter().map(|(_, v)| v).sum();

    let mut live_coins = Vec::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &rpc_host, &hex::encode(wc.coin_id)).await {
            live_coins.push(wc.coin_id);
        }
    }

    if private {
        let denoms: Vec<u64> = recipient_specs.iter().map(|(_, v)| *v).collect();
        let recipient_address = recipient_specs[0].0;
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
                inputs, outputs, change_seeds.clone(), true
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
            for (_, value) in &recipient_specs {
                num_outputs += decompose_value(*value).len();
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
                for (address, value) in &recipient_specs {
                    for denom in decompose_value(*value) {
                        let salt: [u8; 32] = rand::random();
                        all_outputs.push(OutputData::Standard { address: *address, value: denom, salt });
                    }
                }

                // 2. Build exact change outputs
                if final_change > 0 {
                    for denom in decompose_value(final_change) {
                        let seed = wallet.next_wots_seed();
                        let pk = wots::keygen(&seed);
                        let addr = compute_address(&pk);
                        let salt: [u8; 32] = rand::random();
                        let idx = all_outputs.len();
                        all_outputs.push(OutputData::Standard { address: addr, value: denom, salt });
                        change_seeds.push((idx, seed));
                    }
                }

                // 3. Shuffle outputs to obfuscate which is the recipient vs change
                {
                    use rand::seq::SliceRandom;
                    let mut indices: Vec<usize> = (0..all_outputs.len()).collect();
                    indices.shuffle(&mut rand::thread_rng());
                    let shuffled: Vec<OutputData> = indices.iter().map(|&i| all_outputs[i].clone()).collect();
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
            &input_coin_ids, &all_outputs, change_seeds.clone(), false
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

fn mine_pow(commitment: &[u8; 32], required_pow: u32) -> u64 {
    let mut n = 0u64;
    loop {
        let h = midstate::core::hash_concat(commitment, &n.to_le_bytes());
        if midstate::core::types::count_leading_zeros(&h) >= required_pow {
            return n;
        }
        n += 1;
    }
}

async fn fetch_required_pow(client: &reqwest::Client, rpc_port: u16, rpc_host: &str) -> Result<u32> {
    let url = format!("http://{}:{}/state", rpc_host, rpc_port);
    let resp = client.get(&url).send().await?;
    let state: rpc::GetStateResponse = resp.json().await?;
    Ok(state.required_pow)
}

async fn submit_commit(
    client: &reqwest::Client,
    rpc_port: u16,
    rpc_host: &str,
    commitment: &[u8; 32],
) -> Result<()> {
    let required_pow = fetch_required_pow(client, rpc_port, rpc_host).await?;
    println!("Mining PoW locally (difficulty: {} leading zeros)...", required_pow);
    let spam_nonce = mine_pow(commitment, required_pow);
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
        }).collect(),
        signatures: signatures.iter().map(|s| match s {
            midstate::core::types::Witness::ScriptInputs(inputs) => {
                inputs.iter().map(hex::encode).collect::<Vec<_>>().join(",")
            }
        }).collect(),
        outputs: pending.outputs.iter().map(|o| match o {
            OutputData::Standard { address, value, salt } => rpc::OutputDataJson::Standard {
                address: hex::encode(address),
                value: *value,
                salt: hex::encode(salt),
            },
            OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
            OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
        }).collect(),
        salt: hex::encode(pending.salt),
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
) -> Result<()> {
    if !denomination.is_power_of_two() || denomination == 0 {
        anyhow::bail!("denomination must be a non-zero power of 2");
    }

    let password = read_password("Password: ")?;
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
    
    // --- Sign the mix_id to prove ownership ---
    let parsed_mix_id = parse_hex32(&mix_id_hex)?;
    let join_sig = wallet.sign_mix_input(&mix_coin_id, &parsed_mix_id)?;
    // -----------------------------------------------

    let register_req = rpc::MixRegisterRequest {
        mix_id: mix_id_hex.clone(),
        coin_id: hex::encode(mix_coin_id),
        input: rpc::InputRevealJson {
            bytecode: match &input.predicate { midstate::core::types::Predicate::Script { bytecode } => hex::encode(bytecode) },
            value: input.value,
            salt: hex::encode(input.salt),
        },
        output: rpc::OutputDataJson::Standard {
            address: hex::encode(output.address()),
            value: output.value(),
            salt: hex::encode(output.salt()),
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

let (input_reveals, signatures) = match wallet.sign_reveal(&pending) {
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
            midstate::core::OutputData::DataBurn { payload, value_burned } => rpc::OutputDataJson::DataBurn {
                payload: hex::encode(payload),
                value_burned: *value_burned,
            },
            midstate::core::OutputData::Confidential { address, commitment, salt } => rpc::OutputDataJson::Confidential {
                address: hex::encode(address),
                commitment: hex::encode(commitment),
                salt: hex::encode(salt),
            },
        }).collect(),
            salt: hex::encode(pending.salt),
        };

        let response = client.post(&url).json(&req).send().await?;
        if response.status().is_success() {
            let _result: rpc::SendTransactionResponse = response.json().await?;
            wallet.complete_reveal(&commitment)?;
            println!("  {} — revealed ✓", hex::encode(&commitment));
        } else {
            let error: rpc::ErrorResponse = response.json().await?;
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

async fn run_node(
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
    is_bootstrap: bool,
) -> Result<()> {

    // --- Configure Rayon Global Thread Pool for Verification ---
    if let Some(vt) = verify_threads {
        // If vt is 0, Rayon defaults to the number of logical cores.
        // If vt is 1, verification becomes strictly sequential.
        rayon::ThreadPoolBuilder::new()
            .num_threads(vt)
            .build_global()
            .unwrap_or_else(|e| tracing::warn!("Failed to configure verification threads: {}", e));
            
        tracing::info!("Verification restricted to {} thread(s)", vt);
    }
    // -----------------------------------------------------------

    // Load config: explicit --config path, or <data_dir>/config.toml
    let config_file = config_path.unwrap_or_else(|| data_dir.join("config.toml"));
    Config::create_default(&config_file)?;
    let config = Config::load(&config_file)?;

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

    let node = node::Node::new(data_dir.clone(), mining_threads, listen_addr, bootstrap, is_bootstrap).await?;
    
    if is_bootstrap {
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
    }

    let (handle, cmd_rx) = node.create_handle();

    let rpc_server = rpc::RpcServer::new(&rpc_bind, rpc_port)?;
    let handle_clone = handle.clone();
    tokio::spawn(async move {
        if let Err(e) = rpc_server.run(handle_clone).await {
            tracing::error!("RPC server error: {}", e);
        }
    });
    
    // Update the logging print
    tracing::info!("Node started (mining: {}, threads: {}, rpc: {})", 
        mine, threads.unwrap_or(0), rpc_port);

    if mine {
        let simd = midstate::core::simd_mining::detected_level();
        tracing::info!("Mining SIMD: {} ({} nonces/batch)", simd.name(), simd.lanes());
    }
        
    node.run(handle, cmd_rx).await
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
        let response: rpc::GenerateKeyResponse = client.get(&url).send().await?.json().await?;
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
    let storage = storage::Storage::open(data_dir.join("db"))?;
    let syncer = sync::Syncer::new(storage.clone());

    let listen_addr: libp2p::Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", port).parse()?;
    let peer_multiaddr: libp2p::Multiaddr = peer_addr.parse()
       .context("Invalid peer multiaddr (expected e.g. /ip4/1.2.3.4/tcp/9333/p2p/12D3KooW...)")?;

    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let mut network = MidstateNetwork::new(keypair, listen_addr, vec![peer_multiaddr]).await?;

    // Wait for connection
    let peer_id = loop {
        match network.next_event().await {
            NetworkEvent::PeerConnected(id) => break id,
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
        let count = 100.min(peer_height - cursor);
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
    sync::Syncer::verify_header_chain(&headers, &[])?;
    let our_state = storage.load_state()?.unwrap_or_else(|| State::genesis().0);
    // from genesis here...
    let fork_height = syncer.find_fork_point(&headers, 0, our_state.height)?;

    // 4. Download and apply batches
    let mut state = syncer.rebuild_state_to(fork_height)?;
    let mut recent_headers: Vec<u64> = Vec::new();
    let mut dl_cursor = fork_height;
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
        for batch in &batches {
            recent_headers.push(state.timestamp);
            if recent_headers.len() > MEDIAN_TIME_PAST_WINDOW { recent_headers.remove(0); }
            apply_batch(&mut state, batch, &recent_headers, &std::collections::HashMap::new())?;
            storage.save_batch(dl_cursor, batch)?;
            dl_cursor += 1;
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
