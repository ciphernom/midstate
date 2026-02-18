use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use midstate::*;
use midstate::compute_address;
use midstate::wallet::{self, Wallet, stealth_derive, build_stealth_output, short_hex};
use midstate::core::wots;
use midstate::core::state::apply_batch;
use midstate::network::{socket_to_multiaddr, MidstateNetwork, NetworkEvent, Message};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use rayon::prelude::*;

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
    /// Run a node
    Node {
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        #[arg(long, default_value = "9333")]
        port: u16,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long)]
        peer: Vec<SocketAddr>,
        #[arg(long)]
        mine: bool,
        #[arg(long)]
        listen: Option<String>,
    },

    /// Wallet operations
    Wallet {
        #[command(subcommand)]
        action: WalletAction,
    },

    /// Phase 1: Commit to a spend
    Commit {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long)]
        coin: Vec<String>,
        #[arg(long)]
        dest: Vec<String>,
    },

    /// Check if a coin exists
    Balance {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long)]
        coin: String,
    },

    /// Get current state
    State {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Get mempool info
    Mempool {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Get peer list
    Peers {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Generate a WOTS keypair
    Keygen {
        #[arg(long)]
        rpc_port: Option<u16>,
    },

    /// Sync from genesis
    Sync {
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        #[arg(long)]
        peer: SocketAddr,
        #[arg(long, default_value = "9333")]
        port: u16,
    },
}

#[derive(Subcommand)]
enum WalletAction {
    Create {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
    },
    /// Generate a receiving address (WOTS key)
    Receive {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        label: Option<String>,
    },
    /// Generate multiple receiving keys
    Generate {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, short, default_value = "1")]
        count: usize,
        #[arg(long)]
        label: Option<String>,
    },
    /// Generate a reusable MSS address (Merkle Tree)
    GenerateMss {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "10")]
        height: u32,
        #[arg(long)]
        label: Option<String>,
    },
    /// Generate a scan key for receiving stealth payments
    GenerateScanKey {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long)]
        label: Option<String>,
    },
    List {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long)]
        full: bool,
    },
    Balance {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },
    /// Send value. --to format: <address_hex>:<value>
    Send {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        /// Explicit input coin IDs (optional, auto-selects if omitted)
        #[arg(long)]
        coin: Vec<String>,
        /// Recipient outputs: <address_hex>:<value>
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
        #[arg(long)]
        private: bool,
        /// Recipient's stealth scan key (hex). When set, outputs go to an
        /// unlinkable stealth address derived from this key.
        #[arg(long)]
        stealth_scan_key: Option<String>,
    },
    /// Import a coin with known seed, value, and salt
    Import {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// WOTS seed (hex)
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
    },
    Scan {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
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
        anyhow::bail!("expected format <owner_pk_hex>:<value>, got: {}", s);
    }
    let pk = parse_hex32(parts[0])?;
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
        Command::Node { data_dir, port, rpc_port, peer, mine, listen } => {
            run_node(data_dir, port, rpc_port, peer, mine, listen).await
        }
        Command::Wallet { action } => handle_wallet(action).await,
        Command::Commit { rpc_port, coin, dest } => {
            commit_transaction(rpc_port, coin, dest).await
        }
        Command::Balance { rpc_port, coin } => check_balance(rpc_port, coin).await,
        Command::State { rpc_port } => get_state(rpc_port).await,
        Command::Mempool { rpc_port } => get_mempool(rpc_port).await,
        Command::Peers { rpc_port } => get_peers(rpc_port).await,
        Command::Keygen { rpc_port } => keygen(rpc_port).await,
        Command::Sync { data_dir, peer, port } => sync_from_genesis(data_dir, peer, port).await,
    }
}

// ── Wallet commands ─────────────────────────────────────────────────────────
async fn wallet_scan(path: &PathBuf, rpc_port: u16) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let addresses = wallet.watched_addresses();
    if addresses.is_empty() {
        println!("No addresses to scan for. Generate keys first.");
        return Ok(());
    }

    let state_url = format!("http://127.0.0.1:{}/state", rpc_port);
    let state: rpc::GetStateResponse = client.get(&state_url).send().await?.json().await?;
    let chain_height = state.height;
    let start = wallet.data.last_scan_height;

    if start >= chain_height {
        println!("Already scanned to height {}. Chain is at {}.", start, chain_height);
        return Ok(());
    }

    println!("Scanning blocks {}..{} for {} address(es)...", start, chain_height, addresses.len());

    let scan_url = format!("http://127.0.0.1:{}/scan", rpc_port);
    let req = rpc::ScanRequest {
        addresses: addresses.iter().map(|a| hex::encode(a)).collect(),
        start_height: start,
        end_height: chain_height,
    };
    let resp: rpc::ScanResponse = client.post(&scan_url).json(&req).send().await?.json().await?;

    let mut imported = 0usize;
    for sc in &resp.coins {
        let address = parse_hex32(&sc.address)?;
        let salt = parse_hex32(&sc.salt)?;
        if let Some(coin_id) = wallet.import_scanned(address, sc.value, salt, None)? {
            println!("  found: {} (value {}, height {})", short_hex(&coin_id), sc.value, sc.height);
            imported += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Stealth scan — fetch all nonces, test each against our scan keys.
    // -----------------------------------------------------------------------
    if !wallet.data.scan_keys.is_empty() {
        let stealth_url = format!("http://127.0.0.1:{}/scan_stealth", rpc_port);
        let stealth_req = rpc::ScanStealthRequest { start_height: start, end_height: chain_height };
        let stealth_resp: rpc::ScanStealthResponse = client
            .post(&stealth_url)
            .json(&stealth_req)
            .send().await?
            .json().await?;

        for entry in &stealth_resp.nonces {
            let nonce = parse_hex32(&entry.nonce)?;
            let salt  = parse_hex32(&entry.salt)?;

            // Try every scan key we hold against this nonce.
            for scan_key in &wallet.data.scan_keys.clone() {
                let (stealth_seed, _, stealth_addr) =
                    stealth_derive(&scan_key.public_key, &nonce);

                if let Some(coin_id) = wallet.import_scanned(
                    stealth_addr, entry.value, salt, Some(stealth_seed),
                )? {
                    println!(
                        "  found stealth: {} (value {}, height {})",
                        short_hex(&coin_id), entry.value, entry.height,
                    );
                    imported += 1;
                }
            }
        }
    }

    wallet.data.last_scan_height = chain_height;
    
    // Sync MSS key indices (stateful recovery)
    if !wallet.data.mss_keys.is_empty() {
        println!("Syncing MSS key indices...");
        let mss_url = format!("http://127.0.0.1:{}/mss_state", rpc_port);
        for mss_key in &mut wallet.data.mss_keys {
            let req = rpc::GetMssStateRequest {
                master_pk: hex::encode(mss_key.master_pk),
            };
            match client.post(&mss_url).json(&req).send().await {
                Ok(resp) => {
                    if let Ok(mss_resp) = resp.json::<rpc::GetMssStateResponse>().await {
                        if mss_resp.next_index >= mss_key.next_leaf {
                            const SAFETY_MARGIN: u64 = 20;
                            let new_leaf = mss_resp.next_index + SAFETY_MARGIN;
                            println!("  MSS {}: advancing leaf {} -> {}",
                                short_hex(&mss_key.master_pk), mss_key.next_leaf, new_leaf);
                            mss_key.next_leaf = new_leaf;
                        }
                    }
                }
                Err(e) => {
                    println!("  MSS {} sync failed: {}", short_hex(&mss_key.master_pk), e);
                }
            }
        }
    }
    
    
    wallet.save()?;

    println!("Scan complete. {} new coin(s) found. Scanned to height {}.", imported, chain_height);
    Ok(())
}


async fn handle_wallet(action: WalletAction) -> Result<()> {
    match action {
        WalletAction::Create { path } => wallet_create(&path),
        WalletAction::Receive { path, label } => wallet_receive(&path, label),
        WalletAction::Generate { path, count, label } => wallet_generate(&path, count, label),
        WalletAction::List { path, rpc_port, full } => wallet_list(&path, rpc_port, full).await,
        WalletAction::Balance { path, rpc_port } => wallet_balance(&path, rpc_port).await,
        WalletAction::Scan { path, rpc_port } => wallet_scan(&path, rpc_port).await,
        WalletAction::Send { path, rpc_port, coin, to, timeout, private, stealth_scan_key } => {
            wallet_send(&path, rpc_port, coin, to, timeout, private, stealth_scan_key.as_deref()).await
        }
        WalletAction::Import { path, seed, value, salt, label } => {
            wallet_import(&path, &seed, value, &salt, label)
        }
        WalletAction::Export { path, coin } => wallet_export(&path, &coin),
        WalletAction::Pending { path } => wallet_pending(&path),
        WalletAction::Reveal { path, rpc_port, commitment } => {
            wallet_reveal(&path, rpc_port, commitment).await
        }
        WalletAction::History { path, count } => wallet_history(&path, count),
        WalletAction::ImportRewards { path, coinbase_file } => {
            wallet_import_rewards(&path, &coinbase_file)
        }
        WalletAction::GenerateMss { path, height, label } => {
            wallet_generate_mss(&path, height, label)
        }
        WalletAction::GenerateScanKey { path, label } => {
            wallet_generate_scan_key(&path, label)
        }
        
    }
}

fn wallet_create(path: &PathBuf) -> Result<()> {
    let password = read_password_confirm()?;
    Wallet::create(path, &password)?;
    println!("Wallet created: {}", path.display());
    Ok(())
}

fn wallet_receive(path: &PathBuf, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let label = label.unwrap_or_else(|| format!("receive #{}", wallet.keys().len() + 1));
    let address = wallet.generate_key(Some(label.clone()))?;
    println!("\n  Your receiving address ({}):\n", label);
    println!("  {}\n", hex::encode(address));
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
        println!("  [{}] {}", wallet.keys().len() - 1, hex::encode(pk));
    }
    println!("\nGenerated {} key(s). Total keys: {}, Total coins: {}",
        count, wallet.keys().len(), wallet.coin_count());
    Ok(())
}

async fn wallet_list(path: &PathBuf, rpc_port: u16, full: bool) -> Result<()> {
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
            let status = check_coin_rpc(&client, rpc_port, &coin_hex).await;
            let label = wc.label.as_deref().unwrap_or("");
            let status_str = match status {
                Ok(true) => "✓ live",
                Ok(false) => "✗ unset",
                Err(_) => "? error",
            };
            let display = if full { coin_hex } else { short_hex(&wc.coin_id) };
            println!("{:<5} {:<15} {:<8} {:<10} {}", i, display, wc.value, status_str, label);
        }
    }

    if !wallet.keys().is_empty() {
        println!("\nUNUSED RECEIVING KEYS:");
        for (i, k) in wallet.keys().iter().enumerate() {
            let display = if full { hex::encode(k.address) } else { short_hex(&k.address) };
            let label = k.label.as_deref().unwrap_or("");
            println!("  [K{}] {} {}", i, display, label);
        }
    }

    if !full { println!("\nUse --full to show complete IDs."); }
    Ok(())
}

async fn wallet_balance(path: &PathBuf, rpc_port: u16) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let client = reqwest::Client::new();

    let mut live_count = 0usize;
    let mut live_value = 0u64;
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin_id)).await {
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
    coin_args: Vec<String>,
    to_args: Vec<String>,
    timeout_secs: u64,
    private: bool,
    stealth_scan_key: Option<&str>,
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
        let mss_url = format!("http://127.0.0.1:{}/mss_state", rpc_port);

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
                    short_hex(&master_pk), mss_resp.next_index, current_leaf);
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
    let needed = total_send + 1;

    let mut live_coins = Vec::new();
    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin_id)).await {
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
            let out_val: u64 = outputs.iter().map(|o| o.value).sum();
            println!("  Pair {}: {} in (value {}) → {} out (value {}, fee {})",
                pair_idx, inputs.len(), in_val, outputs.len(), out_val, in_val - out_val);

            let output_coin_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.coin_id()).collect();

            let (commitment, _salt) = wallet.prepare_commit(
                inputs, outputs, change_seeds.clone(), true
            )?;

            let commit_req = rpc::CommitRequest {
                coins: inputs.iter().map(|c| hex::encode(c)).collect(),
                destinations: output_coin_ids.iter().map(|d| hex::encode(d)).collect(),
            };

            let url = format!("http://127.0.0.1:{}/commit", rpc_port);
            let response = client.post(&url).json(&commit_req).send().await?;
            if !response.status().is_success() {
                let error: rpc::ErrorResponse = response.json().await?;
                println!("  Pair {} commit failed: {}", pair_idx, error.error);
                continue;
            }
            let commit_resp: rpc::CommitResponse = response.json().await?;
            let server_commitment = parse_hex32(&commit_resp.commitment)?;
            let server_salt = parse_hex32(&commit_resp.salt)?;

            wallet.data.pending.retain(|p| p.commitment != commitment);
            wallet.data.pending.push(wallet::PendingCommit {
                commitment: server_commitment,
                salt: server_salt,
                input_coin_ids: inputs.clone(),
                outputs: outputs.clone(),
                change_seeds: change_seeds.clone(),
                created_at: now_secs(),
                reveal_not_before: now_secs() + 10 + (rand::random::<u64>() % 41),
            });
            wallet.save()?;

            println!("  ✓ Commit submitted ({})", short_hex(&server_commitment));

            if !wait_for_commit_mined(&client, rpc_port, &commit_resp.commitment, timeout_secs).await {
                println!("  ⏳ Not mined yet. Run `wallet reveal` later.");
                continue;
            }

            let pending = wallet.find_pending(&server_commitment).unwrap().clone();
            let delay = pending.reveal_not_before.saturating_sub(now_secs());
            if delay > 0 {
                println!("  Waiting {}s (privacy delay)...", delay);
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            do_reveal(&client, &mut wallet, rpc_port, &server_commitment, timeout_secs).await?;
        }
    } else {
        let input_coin_ids: Vec<[u8; 32]> = if !coin_args.is_empty() {
            coin_args.iter()
                .map(|s| wallet.resolve_coin(s))
                .collect::<Result<Vec<_>>>()?
        } else {
            wallet.select_coins(needed, &live_coins)?
        };

        let in_sum: u64 = input_coin_ids.iter()
            .filter_map(|id| wallet.find_coin(id))
            .map(|c| c.value)
            .sum();

        if in_sum <= total_send {
            anyhow::bail!("input value ({}) must exceed output value ({}) to pay fee", in_sum, total_send);
        }

        let fee = 1u64;
        let change = in_sum - total_send - fee;

        let mut all_outputs = Vec::new();
        let mut change_seeds = Vec::new();
        let mut stealth_nonces: Vec<[u8; 32]> = Vec::new();

        // Parse stealth scan key once if provided.
        let parsed_scan_key: Option<[u8; 32]> = stealth_scan_key
            .map(|h| parse_hex32(h))
            .transpose()?;

        for (address, value) in &recipient_specs {
            if let Some(scan_key) = &parsed_scan_key {
                // Stealth path: build one output per denomination, each with its
                // own independently derived nonce and stealth address.
                for denom in decompose_value(*value) {
                    let (output, nonce) = build_stealth_output(scan_key, denom);
                    all_outputs.push(output);
                    stealth_nonces.push(nonce);
                }
            } else {
                // Normal path: decompose value, pad nonces with zeros.
                for denom in decompose_value(*value) {
                    let salt: [u8; 32] = rand::random();
                    all_outputs.push(OutputData { address: *address, value: denom, salt });
                    stealth_nonces.push([0u8; 32]);
                }
            }
        }

        if change > 0 {
            let change_denoms = decompose_value(change);
            for denom in change_denoms {
                let seed: [u8; 32] = rand::random();
                let pk = wots::keygen(&seed);
                let addr = compute_address(&pk);
                let salt: [u8; 32] = rand::random();
                let idx = all_outputs.len();
                all_outputs.push(OutputData { address: addr, value: denom, salt });
                change_seeds.push((idx, seed));
                stealth_nonces.push([0u8; 32]); // change is never stealth
            }
        }

        let output_coin_ids: Vec<[u8; 32]> = all_outputs.iter().map(|o| o.coin_id()).collect();

        println!(
            "Spending {} coin(s) (value {}) → {} output(s) (value {}, fee: {})",
            input_coin_ids.len(), in_sum,
            all_outputs.len(), total_send + change,
            fee
        );

        let (commitment, _salt) = wallet.prepare_commit(
            &input_coin_ids, &all_outputs, change_seeds.clone(), false
        )?;

        let commit_req = rpc::CommitRequest {
            coins: input_coin_ids.iter().map(|c| hex::encode(c)).collect(),
            destinations: output_coin_ids.iter().map(|d| hex::encode(d)).collect(),
        };

        let url = format!("http://127.0.0.1:{}/commit", rpc_port);
        let response = client.post(&url).json(&commit_req).send().await?;
        if !response.status().is_success() {
            let error: rpc::ErrorResponse = response.json().await?;
            anyhow::bail!("commit failed: {}", error.error);
        }
        let commit_resp: rpc::CommitResponse = response.json().await?;
        let server_commitment = parse_hex32(&commit_resp.commitment)?;
        let server_salt = parse_hex32(&commit_resp.salt)?;

        wallet.data.pending.retain(|p| p.commitment != commitment);
        wallet.data.pending.push(wallet::PendingCommit {
            commitment: server_commitment,
            salt: server_salt,
            input_coin_ids: input_coin_ids.clone(),
            outputs: all_outputs,
            change_seeds,
            created_at: now_secs(),
            reveal_not_before: 0,
        });
        wallet.save()?;

        println!("\n✓ Commit submitted ({})", short_hex(&server_commitment));
        println!("  Waiting for commit to be mined...");

        if !wait_for_commit_mined(&client, rpc_port, &commit_resp.commitment, timeout_secs).await {
            println!("⏳ Not mined after {}s. Run `wallet reveal` later.", timeout_secs);
            return Ok(());
        }
        println!("✓ Commit mined!");

        do_reveal(&client, &mut wallet, rpc_port, &server_commitment, timeout_secs).await?;
    }

    Ok(())
}

async fn wait_for_commit_mined(
    client: &reqwest::Client,
    rpc_port: u16,
    commitment_hex: &str,
    timeout_secs: u64,
) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let mp_url = format!("http://127.0.0.1:{}/mempool", rpc_port);
        if let Ok(resp) = client.get(&mp_url).send().await {
            if let Ok(mp) = resp.json::<rpc::GetMempoolResponse>().await {
                let still_pending = mp.transactions.iter().any(|tx| {
                    tx.commitment.as_deref() == Some(commitment_hex)
                });
                if !still_pending {
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
    commitment: &[u8; 32],
    timeout_secs: u64,
) -> Result<()> {
    let pending = wallet.find_pending(commitment)
        .ok_or_else(|| anyhow::anyhow!("pending commit not found"))?
        .clone();

    let (input_reveals, signatures) = wallet.sign_reveal(&pending)?;

    let reveal_url = format!("http://127.0.0.1:{}/send", rpc_port);
    let reveal_req = rpc::SendTransactionRequest {
        inputs: input_reveals.iter().map(|ir| rpc::InputRevealJson {
            owner_pk: hex::encode(ir.owner_pk),
            value: ir.value,
            salt: hex::encode(ir.salt),
        }).collect(),
        signatures: signatures.iter().map(|s| hex::encode(s)).collect(),
        outputs: pending.outputs.iter().map(|o| rpc::OutputDataJson {
            address: hex::encode(o.address),
            value: o.value,
            salt: hex::encode(o.salt),
        }).collect(),
        salt: hex::encode(pending.salt),
        stealth_nonces: vec![],
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
            .post(&format!("http://127.0.0.1:{}/check", rpc_port))
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
        println!("  spent:   {} (value {})", short_hex(id), val);
    }
    for out in &pending.outputs {
        println!("  created: {} (value {})", short_hex(&out.coin_id()), out.value);
    }
    Ok(())
}

fn wallet_import(path: &PathBuf, seed_hex: &str, value: u64, salt_hex: &str, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let seed = parse_hex32(seed_hex)?;
    let salt = parse_hex32(salt_hex)?;
    let coin_id = wallet.import_coin(seed, value, salt, label)?;
    println!("Imported: {} (value {})", short_hex(&coin_id), value);
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
        let out_val: u64 = p.outputs.iter().map(|o| o.value).sum();
        println!(
            "  [{}] {} — {} in, {} out (value {}), {}",
            i, short_hex(&p.commitment),
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
        println!("  [{}] {} (fee: {})", start + i, format_age(age), entry.fee);
        for c in &entry.inputs { println!("    spent:   {}", short_hex(c)); }
        for c in &entry.outputs { println!("    created: {}", short_hex(c)); }
        println!();
    }
    Ok(())
}

async fn wallet_reveal(
    path: &PathBuf,
    rpc_port: u16,
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
                println!("  {} — not found, skipping", short_hex(&commitment));
                continue;
            }
        };

        if pending.reveal_not_before > now_secs() {
            let wait = pending.reveal_not_before - now_secs();
            println!("  {} — waiting {}s (privacy delay)", short_hex(&commitment), wait);
            tokio::time::sleep(Duration::from_secs(wait)).await;
        }

        let (input_reveals, signatures) = wallet.sign_reveal(&pending)?;

        let url = format!("http://127.0.0.1:{}/send", rpc_port);
        let req = rpc::SendTransactionRequest {
            inputs: input_reveals.iter().map(|ir| rpc::InputRevealJson {
                owner_pk: hex::encode(ir.owner_pk),
                value: ir.value,
                salt: hex::encode(ir.salt),
            }).collect(),
            signatures: signatures.iter().map(|s| hex::encode(s)).collect(),
            outputs: pending.outputs.iter().map(|o| rpc::OutputDataJson {
                address: hex::encode(o.address),
                value: o.value,
                salt: hex::encode(o.salt),
            }).collect(),
            salt: hex::encode(pending.salt),
                    stealth_nonces: vec![],
        };

        let response = client.post(&url).json(&req).send().await?;
        if response.status().is_success() {
            let _result: rpc::SendTransactionResponse = response.json().await?;
            wallet.complete_reveal(&commitment)?;
            println!("  {} — revealed ✓", short_hex(&commitment));
        } else {
            let error: rpc::ErrorResponse = response.json().await?;
            println!("  {} — failed: {}", short_hex(&commitment), error.error);
        }
    }
    Ok(())
}

fn wallet_import_rewards(path: &PathBuf, coinbase_file: &PathBuf) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    println!("Reading coinbase log...");
    let contents = std::fs::read_to_string(coinbase_file)?;

    #[derive(serde::Deserialize)]
    struct CoinbaseEntry {
        #[allow(dead_code)]
        height: u64,
        #[allow(dead_code)]
        index: u64,
        seed: String,
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

    println!("Found {} rewards. Importing...", entries.len());

    let new_coins: Vec<wallet::WalletCoin> = entries
        .into_par_iter()
        .map(|entry| {
            let seed = parse_hex32(&entry.seed).unwrap();
            let salt = parse_hex32(&entry.salt).unwrap();
            let owner_pk = wots::keygen(&seed);
            let address = compute_address(&owner_pk);
            let coin_id = compute_coin_id(&address, entry.value, &salt);

            wallet::WalletCoin {
                seed,
                owner_pk,
                address,
                value: entry.value,
                salt,
                coin_id,
                label: Some(format!("coinbase (value {})", entry.value)),
            }
        })
        .collect();

    let existing_coins: std::collections::HashSet<_> = wallet.data.coins
        .iter()
        .map(|c| c.coin_id)
        .collect();

    let mut imported = 0usize;
    for wc in new_coins {
        if !existing_coins.contains(&wc.coin_id) {
            wallet.data.coins.push(wc);
            imported += 1;
        }
    }

    println!("Saving wallet...");
    wallet.save()?;

    println!("Imported {} coinbase reward(s). Total coins: {}, total value: {}",
        imported, wallet.coin_count(), wallet.total_value());
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
    println!("  Address:  {}", hex::encode(root));
    println!("\nThis address is reusable until the capacity is exhausted.");

    Ok(())
}

fn wallet_generate_scan_key(path: &PathBuf, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let public_key = wallet.generate_scan_key(label)?;
    println!("\n  Scan key (share this to receive stealth payments):\n");
    println!("  {}\n", hex::encode(public_key));
    println!("  Senders pass this to --stealth-scan-key when sending.");
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async fn check_coin_rpc(client: &reqwest::Client, rpc_port: u16, coin_hex: &str) -> Result<bool> {
    let url = format!("http://127.0.0.1:{}/check", rpc_port);
    let req = rpc::CheckCoinRequest { coin: coin_hex.to_string() };
    let resp: rpc::CheckCoinResponse = client.post(&url).json(&req).send().await?.json().await?;
    Ok(resp.exists)
}

// ── Original commands ───────────────────────────────────────────────────────

async fn run_node(
    data_dir: PathBuf, port: u16, rpc_port: u16, peers: Vec<SocketAddr>, mine: bool, listen: Option<String>,
) -> Result<()> {
    let listen_addr: libp2p::Multiaddr = match listen {
        Some(addr) => addr.parse()?,
        None => format!("/ip4/0.0.0.0/tcp/{}", port).parse()?,
    };

    let bootstrap: Vec<libp2p::Multiaddr> = peers.iter()
        .map(|a| socket_to_multiaddr(*a))
        .collect();

    let node = node::Node::new(data_dir, mine, listen_addr, bootstrap).await?;
    let (handle, cmd_rx) = node.create_handle();

    let rpc_server = rpc::RpcServer::new(rpc_port);
    let handle_clone = handle.clone();
    tokio::spawn(async move {
        if let Err(e) = rpc_server.run(handle_clone).await {
            tracing::error!("RPC server error: {}", e);
        }
    });
    tracing::info!("Node started (mining: {}, rpc: {})", mine, rpc_port);
    node.run(handle, cmd_rx).await
}

async fn commit_transaction(rpc_port: u16, coins: Vec<String>, destinations: Vec<String>) -> Result<()> {
    if coins.is_empty() { anyhow::bail!("Must provide at least one coin"); }
    if destinations.is_empty() { anyhow::bail!("Must provide at least one destination"); }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/commit", rpc_port);
    let req = rpc::CommitRequest { coins, destinations };
    let response = client.post(&url).json(&req).send().await?;

    if response.status().is_success() {
        let result: rpc::CommitResponse = response.json().await?;
        println!("Commitment submitted!");
        println!("  Commitment: {}", result.commitment);
        println!("  Salt:       {}", result.salt);
    } else {
        let error: rpc::ErrorResponse = response.json().await?;
        println!("Error: {}", error.error);
    }
    Ok(())
}

async fn check_balance(rpc_port: u16, coin: String) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/check", rpc_port);
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

async fn get_state(rpc_port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/state", rpc_port);
    let response: rpc::GetStateResponse = client.get(&url).send().await?.json().await?;
    println!("State:");
    println!("  Height:       {}", response.height);
    println!("  Depth:        {}", response.depth);
    println!("  Coins:        {}", response.num_coins);
    println!("  Commitments:  {}", response.num_commitments);
    println!("  Midstate:     {}", response.midstate);
    println!("  Target:       {}", response.target);
    println!("  Block reward: {}", response.block_reward);
    Ok(())
}

async fn get_mempool(rpc_port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/mempool", rpc_port);
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

async fn get_peers(rpc_port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/peers", rpc_port);
    let response: rpc::GetPeersResponse = client.get(&url).send().await?.json().await?;
    println!("Peers: {}", response.peers.len());
    for peer in response.peers { println!("  {}", peer); }
    Ok(())
}

async fn keygen(rpc_port: Option<u16>) -> Result<()> {
    if let Some(port) = rpc_port {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/keygen", port);
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
        println!("  Address:  {}", hex::encode(address));
    }
    println!("\n⚠️  Keep the seed safe! Anyone with it can spend coins sent to this address.");
    Ok(())
}

async fn sync_from_genesis(data_dir: PathBuf, peer_addr: SocketAddr, port: u16) -> Result<()> {
    let storage = storage::Storage::open(data_dir.join("db"))?;
    let syncer = sync::Syncer::new(storage.clone());

    let listen_addr: libp2p::Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", port).parse()?;
    let peer_multiaddr = socket_to_multiaddr(peer_addr);

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
    sync::Syncer::verify_header_chain(&headers)?;
    let our_state = storage.load_state()?.unwrap_or_else(|| State::genesis().0);
    let fork_height = syncer.find_fork_point(&headers, our_state.height)?;

    // 4. Download and apply batches
    let mut state = syncer.rebuild_state_to(fork_height)?;
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
            apply_batch(&mut state, batch)?;
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
