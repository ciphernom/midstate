use anyhow::Result;
use clap::{Parser, Subcommand};
use midstate::*;
use midstate::wallet::{self, Wallet, short_hex};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,

        /// Port to listen on for P2P
        #[arg(long, default_value = "9333")]
        port: u16,

        /// Port for RPC server
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Peer addresses to connect to
        #[arg(long)]
        peer: Vec<SocketAddr>,

        /// Enable mining
        #[arg(long)]
        mine: bool,
    },

    /// Wallet operations
    Wallet {
        #[command(subcommand)]
        action: WalletAction,
    },

    /// Phase 1: Commit to a spend (binds inputs to outputs)
    Commit {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Coin IDs being spent (hex)
        #[arg(long)]
        coin: Vec<String>,

        /// Destination coins (hex)
        #[arg(long)]
        dest: Vec<String>,
    },

    /// Phase 2: Reveal secrets and execute the spend
    Send {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Secrets (hex, can specify multiple to merge coins)
        #[arg(long)]
        secret: Vec<String>,

        /// Destination coins (hex, can specify multiple)
        #[arg(long)]
        dest: Vec<String>,

        /// Salt from the commit phase (hex)
        #[arg(long)]
        salt: String,
    },

    /// Check if a coin exists
    Balance {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Coin commitment (hex)
        #[arg(long)]
        coin: String,
    },

    /// Get current state
    State {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Get mempool info
    Mempool {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Get peer list
    Peers {
        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Generate a random secret and its commitment
    Keygen {
        /// RPC port (optional, will generate locally if not specified)
        #[arg(long)]
        rpc_port: Option<u16>,
    },

    /// Sync from genesis (trustless)
    Sync {
        /// Data directory
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,

        /// Peer to sync from
        #[arg(long)]
        peer: SocketAddr,
    },
}

#[derive(Subcommand)]
enum WalletAction {
    /// Create a new encrypted wallet
    Create {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
    },

    /// Generate a receiving address to share with a sender
    Receive {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// Label for this address
        #[arg(long)]
        label: Option<String>,
    },

    /// Generate one or more coin keypairs
    Generate {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// Number of coins to generate
        #[arg(long, short, default_value = "1")]
        count: usize,

        /// Label for the coin(s)
        #[arg(long)]
        label: Option<String>,
    },

    /// List all coins in the wallet
    List {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// RPC port to check on-chain status
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Show full 64-char hex IDs
        #[arg(long)]
        full: bool,
    },

    /// Show wallet balance summary
    Balance {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
    },

    /// Send coins (automated commit → mine → reveal)
    Send {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Coin to spend (index like "0", hex prefix, or full hex). Omit to auto-pick.
        #[arg(long)]
        coin: Vec<String>,

        /// Destination address (hex, from recipient's `wallet receive`)
        #[arg(long)]
        to: Vec<String>,

        /// Max seconds to wait for commit to be mined
        #[arg(long, default_value = "120")]
        timeout: u64,
    },

    /// Import a secret into the wallet
    Import {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// Secret (hex)
        #[arg(long)]
        secret: String,

        /// Label
        #[arg(long)]
        label: Option<String>,
    },

    /// Export a coin's secret (dangerous!)
    Export {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// Coin (index, hex prefix, or full hex)
        #[arg(long)]
        coin: String,
    },

    /// Show pending (unrevealed) commits
    Pending {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
    },

    /// Retry reveal for a pending commit
    Reveal {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// RPC port
        #[arg(long, default_value = "8545")]
        rpc_port: u16,

        /// Commitment to reveal (hex). If omitted, tries all pending.
        #[arg(long)]
        commitment: Option<String>,
    },

    /// Show transaction history
    History {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,

        /// Number of recent entries to show
        #[arg(long, short, default_value = "20")]
        count: usize,
    },
}

// ── Password helpers (masked input) ─────────────────────────────────────────

fn read_password(prompt: &str) -> Result<Vec<u8>> {
    let input = rpassword::prompt_password(prompt)?;
    if input.is_empty() {
        anyhow::bail!("password cannot be empty");
    }
    Ok(input.into_bytes())
}

fn read_password_confirm() -> Result<Vec<u8>> {
    let p1 = read_password("Password: ")?;
    let p2 = read_password("Confirm:  ")?;
    if p1 != p2 {
        anyhow::bail!("passwords do not match");
    }
    Ok(p1)
}

fn parse_hex32(s: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(s)?;
    if bytes.len() != 32 {
        anyhow::bail!("expected 32 bytes, got {}", bytes.len());
    }
    Ok(<[u8; 32]>::try_from(bytes).unwrap())
}

fn format_age(secs: u64) -> String {
    if secs < 60 {
        format!("{}s ago", secs)
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
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
        Command::Node { data_dir, port, rpc_port, peer, mine } => {
            run_node(data_dir, port, rpc_port, peer, mine).await
        }

        Command::Wallet { action } => handle_wallet(action).await,

        Command::Commit { rpc_port, coin, dest } => {
            commit_transaction(rpc_port, coin, dest).await
        }

        Command::Send { rpc_port, secret, dest, salt } => {
            send_transaction(rpc_port, secret, dest, salt).await
        }

        Command::Balance { rpc_port, coin } => {
            check_balance(rpc_port, coin).await
        }

        Command::State { rpc_port } => {
            get_state(rpc_port).await
        }

        Command::Mempool { rpc_port } => {
            get_mempool(rpc_port).await
        }

        Command::Peers { rpc_port } => {
            get_peers(rpc_port).await
        }

        Command::Keygen { rpc_port } => {
            keygen(rpc_port).await
        }

        Command::Sync { data_dir, peer } => {
            sync_from_genesis(data_dir, peer).await
        }
    }
}

// ── Wallet commands ─────────────────────────────────────────────────────────

async fn handle_wallet(action: WalletAction) -> Result<()> {
    match action {
        WalletAction::Create { path } => wallet_create(&path),

        WalletAction::Receive { path, label } => wallet_receive(&path, label),

        WalletAction::Generate { path, count, label } => wallet_generate(&path, count, label),

        WalletAction::List { path, rpc_port, full } => wallet_list(&path, rpc_port, full).await,

        WalletAction::Balance { path, rpc_port } => wallet_balance(&path, rpc_port).await,

        WalletAction::Send { path, rpc_port, coin, to, timeout } => {
            wallet_send(&path, rpc_port, coin, to, timeout).await
        }

        WalletAction::Import { path, secret, label } => wallet_import(&path, &secret, label),

        WalletAction::Export { path, coin } => wallet_export(&path, &coin),

        WalletAction::Pending { path } => wallet_pending(&path),

        WalletAction::Reveal { path, rpc_port, commitment } => {
            wallet_reveal(&path, rpc_port, commitment).await
        }

        WalletAction::History { path, count } => wallet_history(&path, count),
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

    let label = label.unwrap_or_else(|| format!("receive #{}", wallet.coin_count() + 1));
    let wc = wallet.generate(Some(label.clone()))?;

    println!();
    println!("  Your receiving address ({}):", label);
    println!();
    println!("  {}", hex::encode(wc.coin));
    println!();
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
        let wc = wallet.generate(lbl)?;
        let coin = wc.coin; // Copy the coin data to drop the borrow on wallet
        let idx = wallet.coin_count() - 1;
        println!("  [{}] {}", idx, hex::encode(coin));
    }

    println!("\nGenerated {} coin(s). Total: {}", count, wallet.coin_count());
    Ok(())
}

async fn wallet_list(path: &PathBuf, rpc_port: u16, full: bool) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;

    if wallet.coin_count() == 0 {
        println!("Wallet is empty. Use `wallet receive` to create an address.");
        return Ok(());
    }

    let client = reqwest::Client::new();

    if full {
        println!(
            "{:<5} {:<66} {:<10} {}",
            "#", "COIN", "STATUS", "LABEL"
        );
        println!("{}", "-".repeat(95));
    } else {
        println!(
            "{:<5} {:<15} {:<10} {}",
            "#", "COIN", "STATUS", "LABEL"
        );
        println!("{}", "-".repeat(50));
    }

    for (i, wc) in wallet.coins().iter().enumerate() {
        let coin_hex = hex::encode(wc.coin);
        let status = check_coin_rpc(&client, rpc_port, &coin_hex).await;
        let label = wc.label.as_deref().unwrap_or("");
        let status_str = match status {
            Ok(true) => "✓ live",
            Ok(false) => "✗ unset",
            Err(_) => "? error",
        };
        let display = if full {
            coin_hex
        } else {
            short_hex(&wc.coin)
        };
        println!("{:<5} {:<15} {:<10} {}", i, display, status_str, label);
    }

    if !full {
        println!("\nUse --full to show complete coin IDs.");
    }

    Ok(())
}

async fn wallet_balance(path: &PathBuf, rpc_port: u16) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;

    let client = reqwest::Client::new();
    let mut live = 0usize;

    for wc in wallet.coins() {
        if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin)).await {
            live += 1;
        }
    }

    println!("Coins in wallet: {}", wallet.coin_count());
    println!("Live on-chain:   {}", live);
    println!("Pending commits: {}", wallet.pending().len());
    Ok(())
}

async fn wallet_send(
    path: &PathBuf,
    rpc_port: u16,
    coin_args: Vec<String>,
    to_args: Vec<String>,
    timeout_secs: u64,
) -> Result<()> {
    if to_args.is_empty() {
        anyhow::bail!("must specify at least one --to destination");
    }

    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    // Parse destinations
    let destinations: Vec<[u8; 32]> = to_args
        .iter()
        .map(|s| parse_hex32(s))
        .collect::<Result<Vec<_>>>()?;

    // Select input coins (support index, prefix, or full hex)
    let input_coins: Vec<[u8; 32]> = if !coin_args.is_empty() {
        coin_args
            .iter()
            .map(|s| wallet.resolve_coin(s))
            .collect::<Result<Vec<_>>>()?
    } else {
        // Auto-pick: need as many inputs as destinations
        let client = reqwest::Client::new();
        let needed = destinations.len();
        let mut picked = Vec::new();

        for wc in wallet.coins() {
            if picked.len() >= needed {
                break;
            }
            if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin)).await {
                picked.push(wc.coin);
            }
        }

        if picked.len() < needed {
            anyhow::bail!(
                "not enough live coins: need {}, found {}",
                needed,
                picked.len()
            );
        }
        picked
    };

    println!(
        "Spending {} coin(s) → {} destination(s)",
        input_coins.len(),
        destinations.len()
    );
    for c in &input_coins {
        println!("  input:  {}", short_hex(c));
    }
    for d in &destinations {
        println!("  output: {}", short_hex(d));
    }

    // ── Phase 1: Commit ─────────────────────────────────────────────────────
    let (commitment, _salt) = wallet.prepare_commit(&input_coins, &destinations)?;

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/commit", rpc_port);

    let req = rpc::CommitRequest {
        coins: input_coins.iter().map(|c| hex::encode(c)).collect(),
        destinations: destinations.iter().map(|d| hex::encode(d)).collect(),
    };

    // The /commit RPC generates its own salt, so we submit via /commit
    // and then update our pending entry with the server's salt.
    let response = client.post(&url).json(&req).send().await?;

    if !response.status().is_success() {
        let error: rpc::ErrorResponse = response.json().await?;
        anyhow::bail!("commit failed: {}", error.error);
    }

    let commit_resp: rpc::CommitResponse = response.json().await?;

    // The server used its own salt+commitment. Update our pending entry.
    let server_commitment = parse_hex32(&commit_resp.commitment)?;
    let server_salt = parse_hex32(&commit_resp.salt)?;

    // Replace wallet's pending with server's actual commitment
    wallet.data.pending.retain(|p| p.commitment != commitment);

    let input_secrets: Vec<Vec<u8>> = input_coins
        .iter()
        .map(|c| {
            wallet
                .find_secret(c)
                .expect("we already verified ownership")
                .secret
                .clone()
        })
        .collect();

    wallet.data.pending.push(wallet::PendingCommit {
        commitment: server_commitment,
        salt: server_salt,
        input_secrets,
        destinations: destinations.clone(),
        created_at: now_secs(),
    });
    wallet.save()?;

    println!("\n✓ Commit submitted ({})", short_hex(&server_commitment));
    println!("  Waiting for commit to be mined...");

    // ── Wait for commit to be mined ─────────────────────────────────────────
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let mut mined = false;

    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;

        let state_url = format!("http://127.0.0.1:{}/state", rpc_port);
        if let Ok(resp) = client.get(&state_url).send().await {
            if let Ok(state) = resp.json::<rpc::GetStateResponse>().await {
                let mempool_url = format!("http://127.0.0.1:{}/mempool", rpc_port);
                if let Ok(mp_resp) = client.get(&mempool_url).send().await {
                    if let Ok(mp) = mp_resp.json::<rpc::GetMempoolResponse>().await {
                        let still_pending = mp.transactions.iter().any(|tx| {
                            tx.commitment.as_deref() == Some(&commit_resp.commitment)
                        });
                        if !still_pending && state.num_commitments > 0 {
                            mined = true;
                            break;
                        }
                    }
                }
            }
        }
        eprint!(".");
    }
    eprintln!();

    if !mined {
        println!("⏳ Commit not yet mined after {}s.", timeout_secs);
        println!("   Run `wallet reveal` later to complete the transfer.");
        return Ok(());
    }

    println!("✓ Commit mined!");

    // ── Phase 2: Reveal ─────────────────────────────────────────────────────
    let pending = wallet
        .find_pending(&server_commitment)
        .expect("we just saved it")
        .clone();

    let reveal_url = format!("http://127.0.0.1:{}/send", rpc_port);
    let reveal_req = rpc::SendTransactionRequest {
        secrets: pending
            .input_secrets
            .iter()
            .map(|s| hex::encode(s))
            .collect(),
        destinations: pending
            .destinations
            .iter()
            .map(|d| hex::encode(d))
            .collect(),
        salt: hex::encode(pending.salt),
    };

    let response = client.post(&reveal_url).json(&reveal_req).send().await?;

    if !response.status().is_success() {
        let error: rpc::ErrorResponse = response.json().await?;
        anyhow::bail!("reveal failed: {}", error.error);
    }

    let _result: rpc::SendTransactionResponse = response.json().await?;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_secs);
    let mut revealed = false;

    let input_coin_hex = hex::encode(input_coins[0]);

    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_secs(2)).await;

        if let Ok(resp) = client
            .post(&format!("http://127.0.0.1:{}/check", rpc_port))
            .json(&rpc::CheckCoinRequest { coin: input_coin_hex.clone() })
            .send()
            .await
        {
            if let Ok(check) = resp.json::<rpc::CheckCoinResponse>().await {
                if !check.exists {
                    revealed = true;
                    break;
                }
            }
        }
        eprint!(".");
    }
    eprintln!();

    if !revealed {
        println!("⏳ Reveal submitted but not yet mined.");
        println!("   Secrets are safe. Run `wallet reveal` to retry.");
        return Ok(());
    }


    wallet.complete_reveal(&server_commitment)?;

    println!("✓ Transfer complete!");
    for c in &input_coins {
        println!("  spent:   {}", short_hex(c));
    }
    for d in &destinations {
        println!("  created: {}", short_hex(d));
    }

    Ok(())
}

fn wallet_import(path: &PathBuf, secret_hex: &str, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let secret = hex::decode(secret_hex)?;
    let coin = wallet.import_secret(secret, label)?;
    println!("Imported: [{}] {}", wallet.coin_count() - 1, short_hex(&coin));
    Ok(())
}

fn wallet_export(path: &PathBuf, coin_ref: &str) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;

    let coin = wallet.resolve_coin(coin_ref)?;
    let wc = wallet
        .find_secret(&coin)
        .ok_or_else(|| anyhow::anyhow!("coin not found in wallet"))?;

    println!("Secret: {}", hex::encode(&wc.secret));
    println!("Coin:   {}", hex::encode(wc.coin));
    println!("\n⚠️  Anyone with the secret can spend this coin.");
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
        println!(
            "  [{}] {} — {} in, {} out, {}",
            i,
            short_hex(&p.commitment),
            p.input_secrets.len(),
            p.destinations.len(),
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

    println!(
        "Transaction history ({} of {}):\n",
        entries.len(),
        history.len()
    );
    for (i, entry) in entries.iter().enumerate() {
        let age = now_secs().saturating_sub(entry.timestamp);
        println!("  [{}] {}", start + i, format_age(age));
        for c in &entry.inputs {
            println!("    spent:   {}", short_hex(c));
        }
        for c in &entry.outputs {
            println!("    created: {}", short_hex(c));
        }
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

        let url = format!("http://127.0.0.1:{}/send", rpc_port);
        let req = rpc::SendTransactionRequest {
            secrets: pending
                .input_secrets
                .iter()
                .map(|s| hex::encode(s))
                .collect(),
            destinations: pending
                .destinations
                .iter()
                .map(|d| hex::encode(d))
                .collect(),
            salt: hex::encode(pending.salt),
        };

        let response = client.post(&url).json(&req).send().await?;

        if response.status().is_success() {
            let _result: rpc::SendTransactionResponse = response.json().await?;
            wallet.complete_reveal(&commitment)?;
            println!("  {} — revealed ✓", short_hex(&commitment));
            for c in &pending.destinations {
                println!("    created: {}", short_hex(c));
            }
        } else {
            let error: rpc::ErrorResponse = response.json().await?;
            println!(
                "  {} — failed: {}",
                short_hex(&commitment),
                error.error
            );
        }
    }

    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async fn check_coin_rpc(
    client: &reqwest::Client,
    rpc_port: u16,
    coin_hex: &str,
) -> Result<bool> {
    let url = format!("http://127.0.0.1:{}/check", rpc_port);
    let req = rpc::CheckCoinRequest {
        coin: coin_hex.to_string(),
    };
    let resp: rpc::CheckCoinResponse =
        client.post(&url).json(&req).send().await?.json().await?;
    Ok(resp.exists)
}

// ── Original commands (unchanged) ───────────────────────────────────────────

async fn run_node(
    data_dir: PathBuf,
    port: u16,
    rpc_port: u16,
    peers: Vec<SocketAddr>,
    mine: bool,
) -> Result<()> {
    let bind_addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let mut node = node::Node::new(data_dir, mine, bind_addr)?;

    node.listen(bind_addr).await?;

    for peer_addr in peers {
        if let Err(e) = node.connect_to_peer(peer_addr).await {
            tracing::warn!("Failed to connect to {}: {}", peer_addr, e);
        }
    }

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

async fn commit_transaction(
    rpc_port: u16,
    coins: Vec<String>,
    destinations: Vec<String>,
) -> Result<()> {
    if coins.is_empty() {
        anyhow::bail!("Must provide at least one coin");
    }
    if destinations.is_empty() {
        anyhow::bail!("Must provide at least one destination");
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/commit", rpc_port);

    let req = rpc::CommitRequest {
        coins,
        destinations,
    };

    let response = client.post(&url).json(&req).send().await?;

    if response.status().is_success() {
        let result: rpc::CommitResponse = response.json().await?;
        println!("Commitment submitted!");
        println!("  Commitment: {}", result.commitment);
        println!("  Salt:       {}", result.salt);
        println!();
        println!("⚠️  Save the salt! You need it for the reveal (send) phase.");
        println!("⏳ Wait for the commitment to be mined before sending.");
    } else {
        let error: rpc::ErrorResponse = response.json().await?;
        println!("Error: {}", error.error);
    }

    Ok(())
}

async fn send_transaction(
    rpc_port: u16,
    secrets: Vec<String>,
    destinations: Vec<String>,
    salt: String,
) -> Result<()> {
    if secrets.is_empty() {
        anyhow::bail!("Must provide at least one secret");
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/send", rpc_port);

    let req = rpc::SendTransactionRequest {
        secrets,
        destinations,
        salt,
    };

    let response = client.post(&url).json(&req).send().await?;

    if response.status().is_success() {
        let result: rpc::SendTransactionResponse = response.json().await?;
        println!("Transaction submitted!");
        for (i, input) in result.input_coins.iter().enumerate() {
            println!("  Input {}: {}", i, input);
        }
        for (i, output) in result.output_coins.iter().enumerate() {
            println!("  Output {}: {}", i, output);
        }
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
        println!(
            "Exists: {}",
            if result.exists { "YES ✓" } else { "NO ✗" }
        );
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
    println!("  Height:      {}", response.height);
    println!("  Depth:       {}", response.depth);
    println!("  Coins:       {}", response.num_coins);
    println!("  Commitments: {}", response.num_commitments);
    println!("  Midstate:    {}", response.midstate);
    println!("  Target:      {}", response.target);

    Ok(())
}

async fn get_mempool(rpc_port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/mempool", rpc_port);

    let response: rpc::GetMempoolResponse = client.get(&url).send().await?.json().await?;

    println!("Mempool:");
    println!("  Size: {}", response.size);

    if !response.transactions.is_empty() {
        println!("\nTransactions:");
        for (i, tx) in response.transactions.iter().enumerate() {
            if let Some(ref commitment) = tx.commitment {
                println!("  {} [COMMIT]: {}", i + 1, commitment);
            }
            if let Some(ref inputs) = tx.input_coins {
                println!("  {} [REVEAL]:", i + 1);
                for (j, input) in inputs.iter().enumerate() {
                    println!("    Input {}: {}", j, input);
                }
            }
            if let Some(ref outputs) = tx.output_coins {
                for (j, output) in outputs.iter().enumerate() {
                    println!("    Output {}: {}", j, output);
                }
            }
        }
    }

    Ok(())
}

async fn get_peers(rpc_port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/peers", rpc_port);

    let response: rpc::GetPeersResponse = client.get(&url).send().await?.json().await?;

    println!("Peers: {}", response.peers.len());
    for peer in response.peers {
        println!("  {}", peer);
    }

    Ok(())
}

async fn keygen(rpc_port: Option<u16>) -> Result<()> {
    if let Some(port) = rpc_port {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{}/keygen", port);

        let response: rpc::GenerateKeyResponse = client.get(&url).send().await?.json().await?;

        println!("Generated keypair:");
        println!("  Secret: {}", response.secret);
        println!("  Coin:   {}", response.coin);
    } else {
        let secret: [u8; 32] = rand::random();
        let coin = core::hash(&secret);

        println!("Generated keypair:");
        println!("  Secret: {}", hex::encode(secret));
        println!("  Coin:   {}", hex::encode(coin));
    }

    println!("\n⚠️  Keep the secret safe! Anyone with it can spend the coin.");

    Ok(())
}

async fn sync_from_genesis(data_dir: PathBuf, peer_addr: SocketAddr) -> Result<()> {
    let storage = storage::Storage::open(data_dir.join("db"))?;
    let syncer = sync::Syncer::new(storage);

    let mut peer =
        network::PeerConnection::connect(peer_addr, ([127, 0, 0, 1], 0).into()).await?;

    let state = syncer.sync_from_genesis(&mut peer).await?;

    println!("Sync complete!");
    println!("  Height:      {}", state.height);
    println!("  Depth:       {}", state.depth);
    println!("  Coins:       {}", state.coins.len());
    println!("  Commitments: {}", state.commitments.len());
    println!("  Midstate:    {}", hex::encode(state.midstate));

    Ok(())
}
