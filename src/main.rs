use anyhow::Result;
use clap::{Parser, Subcommand};
use midstate::*;
use midstate::wallet::{self, Wallet, short_hex};
use midstate::core::wots;
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

    /// Phase 2: Reveal with WOTS signatures
    Send {
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        /// Input coin IDs (hex)
        #[arg(long)]
        input_coin: Vec<String>,
        /// WOTS seeds for signing (hex, one per input)
        #[arg(long)]
        seed: Vec<String>,
        /// Destination coins (hex)
        #[arg(long)]
        dest: Vec<String>,
        /// Salt from the commit phase (hex)
        #[arg(long)]
        salt: String,
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
    },
}

#[derive(Subcommand)]
enum WalletAction {
    Create {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
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
    Send {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        #[arg(long, default_value = "8545")]
        rpc_port: u16,
        #[arg(long)]
        coin: Vec<String>,
        #[arg(long)]
        to: Vec<String>,
        #[arg(long, default_value = "120")]
        timeout: u64,
        /// Private mode: split into independent 2-in-1-out pairs with random reveal delays
        #[arg(long)]
        private: bool,
    },
    Import {
        #[arg(long, default_value_os_t = default_wallet_path())]
        path: PathBuf,
        /// WOTS seed (hex)
        #[arg(long)]
        seed: String,
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
        /// Path to coinbase_seeds.jsonl
        #[arg(long)]
        coinbase_file: PathBuf,
    },
}

fn read_password(prompt: &str) -> Result<Vec<u8>> {
    let input = rpassword::prompt_password(prompt)?;
    if input.is_empty() { anyhow::bail!("password cannot be empty"); }
    Ok(input.into_bytes())
}

fn read_password_confirm() -> Result<Vec<u8>> {
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
        Command::Node { data_dir, port, rpc_port, peer, mine } => {
            run_node(data_dir, port, rpc_port, peer, mine).await
        }
        Command::Wallet { action } => handle_wallet(action).await,
        Command::Commit { rpc_port, coin, dest } => {
            commit_transaction(rpc_port, coin, dest).await
        }
        Command::Send { rpc_port, input_coin, seed, dest, salt } => {
            send_transaction(rpc_port, input_coin, seed, dest, salt).await
        }
        Command::Balance { rpc_port, coin } => check_balance(rpc_port, coin).await,
        Command::State { rpc_port } => get_state(rpc_port).await,
        Command::Mempool { rpc_port } => get_mempool(rpc_port).await,
        Command::Peers { rpc_port } => get_peers(rpc_port).await,
        Command::Keygen { rpc_port } => keygen(rpc_port).await,
        Command::Sync { data_dir, peer } => sync_from_genesis(data_dir, peer).await,
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
        WalletAction::Send { path, rpc_port, coin, to, timeout, private } => {
            wallet_send(&path, rpc_port, coin, to, timeout, private).await
        }
        WalletAction::Import { path, seed, label } => wallet_import(&path, &seed, label),
        WalletAction::Export { path, coin } => wallet_export(&path, &coin),
        WalletAction::Pending { path } => wallet_pending(&path),
        WalletAction::Reveal { path, rpc_port, commitment } => {
            wallet_reveal(&path, rpc_port, commitment).await
        }
        WalletAction::History { path, count } => wallet_history(&path, count),
        WalletAction::ImportRewards { path, coinbase_file } => {
            wallet_import_rewards(&path, &coinbase_file)
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
    let label = label.unwrap_or_else(|| format!("receive #{}", wallet.coin_count() + 1));
    let wc = wallet.generate(Some(label.clone()))?;
    println!("\n  Your receiving address ({}):\n", label);
    println!("  {}\n", hex::encode(wc.coin));
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
        let coin = wc.coin;
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
        println!("{:<5} {:<66} {:<10} {}", "#", "COIN", "STATUS", "LABEL");
        println!("{}", "-".repeat(95));
    } else {
        println!("{:<5} {:<15} {:<10} {}", "#", "COIN", "STATUS", "LABEL");
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
        let display = if full { coin_hex } else { short_hex(&wc.coin) };
        println!("{:<5} {:<15} {:<10} {}", i, display, status_str, label);
    }
    if !full { println!("\nUse --full to show complete coin IDs."); }
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
    private: bool,
) -> Result<()> {
    if to_args.is_empty() {
        anyhow::bail!("must specify at least one --to destination");
    }

    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;

    let destinations: Vec<[u8; 32]> = to_args.iter()
        .map(|s| parse_hex32(s))
        .collect::<Result<Vec<_>>>()?;

    let client = reqwest::Client::new();

    if private {
        // Private mode: gather live coins, plan independent pairs
        let mut live_coins = Vec::new();
        for wc in wallet.coins() {
            if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin)).await {
                live_coins.push(wc.coin);
            }
        }

        let pairs = wallet.plan_private_send(&live_coins, &destinations)?;
        println!("Private send: {} independent transaction(s)\n", pairs.len());

        for (pair_idx, (inputs, outputs)) in pairs.iter().enumerate() {
            println!("  Pair {}: {} in → {} out", pair_idx, inputs.len(), outputs.len());

            // Commit
            let (commitment, _salt) = wallet.prepare_commit(inputs, outputs, true)?;

            let commit_req = rpc::CommitRequest {
                coins: inputs.iter().map(|c| hex::encode(c)).collect(),
                destinations: outputs.iter().map(|d| hex::encode(d)).collect(),
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

            // Replace pending with server's commitment
            wallet.data.pending.retain(|p| p.commitment != commitment);
            let input_seeds: Vec<[u8; 32]> = inputs.iter()
                .map(|c| wallet.find_coin(c).expect("own coin").seed)
                .collect();

            wallet.data.pending.push(wallet::PendingCommit {
                commitment: server_commitment,
                salt: server_salt,
                input_seeds,
                input_coin_ids: inputs.clone(),
                destinations: outputs.clone(),
                created_at: now_secs(),
                reveal_not_before: now_secs() + 10 + (rand::random::<u64>() % 41),
            });
            wallet.save()?;

            println!("  ✓ Commit submitted ({})", short_hex(&server_commitment));

            // Wait for commit to be mined
            if !wait_for_commit_mined(&client, rpc_port, &commit_resp.commitment, timeout_secs).await {
                println!("  ⏳ Not mined yet. Run `wallet reveal` later.");
                continue;
            }

            // Wait for privacy delay
            let pending = wallet.find_pending(&server_commitment).unwrap().clone();
            let delay = pending.reveal_not_before.saturating_sub(now_secs());
            if delay > 0 {
                println!("  Waiting {}s (privacy delay)...", delay);
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }

            // Reveal
            do_reveal(&client, &mut wallet, rpc_port, &server_commitment, timeout_secs).await?;
        }
    } else {
        // Normal mode: all inputs → all outputs in one tx
        // Need inputs > outputs for fee
        let needed_inputs = destinations.len() + 1; // +1 for fee

        let input_coins: Vec<[u8; 32]> = if !coin_args.is_empty() {
            coin_args.iter()
                .map(|s| wallet.resolve_coin(s))
                .collect::<Result<Vec<_>>>()?
        } else {
            let mut picked = Vec::new();
            for wc in wallet.coins() {
                if picked.len() >= needed_inputs { break; }
                if let Ok(true) = check_coin_rpc(&client, rpc_port, &hex::encode(wc.coin)).await {
                    picked.push(wc.coin);
                }
            }
            if picked.len() < needed_inputs {
                anyhow::bail!(
                    "not enough live coins: need {} (outputs + fee), found {}",
                    needed_inputs, picked.len()
                );
            }
            picked
        };

        if input_coins.len() <= destinations.len() {
            anyhow::bail!(
                "need more inputs ({}) than outputs ({}) to pay fee",
                input_coins.len(), destinations.len()
            );
        }

        println!(
            "Spending {} coin(s) → {} destination(s) (fee: {})",
            input_coins.len(), destinations.len(), input_coins.len() - destinations.len()
        );

        // Commit
        let (commitment, _salt) = wallet.prepare_commit(&input_coins, &destinations, false)?;

        let commit_req = rpc::CommitRequest {
            coins: input_coins.iter().map(|c| hex::encode(c)).collect(),
            destinations: destinations.iter().map(|d| hex::encode(d)).collect(),
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
        let input_seeds: Vec<[u8; 32]> = input_coins.iter()
            .map(|c| wallet.find_coin(c).expect("own coin").seed)
            .collect();

        wallet.data.pending.push(wallet::PendingCommit {
            commitment: server_commitment,
            salt: server_salt,
            input_seeds,
            input_coin_ids: input_coins.clone(),
            destinations: destinations.clone(),
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

/// Wait for a commitment to leave the mempool (i.e. be mined).
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

/// Submit a reveal transaction for a pending commit.
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

    // Build WOTS signatures
    let signatures = wallet.sign_reveal(&pending);

    let reveal_url = format!("http://127.0.0.1:{}/send", rpc_port);
    let reveal_req = rpc::SendTransactionRequest {
        input_coins: pending.input_coin_ids.iter().map(|c| hex::encode(c)).collect(),
        signatures: signatures.iter().map(|s| hex::encode(wots::sig_to_bytes(s))).collect(),
        destinations: pending.destinations.iter().map(|d| hex::encode(d)).collect(),
        salt: hex::encode(pending.salt),
    };

    let response = client.post(&reveal_url).json(&reveal_req).send().await?;
    if !response.status().is_success() {
        let error: rpc::ErrorResponse = response.json().await?;
        anyhow::bail!("reveal failed: {}", error.error);
    }
    let _result: rpc::SendTransactionResponse = response.json().await?;

    // Wait for reveal to be mined (input coin disappears)
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
    for c in &pending.input_coin_ids {
        println!("  spent:   {}", short_hex(c));
    }
    for d in &pending.destinations {
        println!("  created: {}", short_hex(d));
    }
    Ok(())
}

fn wallet_import(path: &PathBuf, seed_hex: &str, label: Option<String>) -> Result<()> {
    let password = read_password("Password: ")?;
    let mut wallet = Wallet::open(path, &password)?;
    let seed = parse_hex32(seed_hex)?;
    let coin = wallet.import_seed(seed, label)?;
    println!("Imported: [{}] {}", wallet.coin_count() - 1, short_hex(&coin));
    Ok(())
}

fn wallet_export(path: &PathBuf, coin_ref: &str) -> Result<()> {
    let password = read_password("Password: ")?;
    let wallet = Wallet::open(path, &password)?;
    let coin = wallet.resolve_coin(coin_ref)?;
    let wc = wallet.find_coin(&coin)
        .ok_or_else(|| anyhow::anyhow!("coin not found in wallet"))?;
    println!("Seed: {}", hex::encode(wc.seed));
    println!("Coin: {}", hex::encode(wc.coin));
    println!("\n⚠️  Anyone with the seed can spend this coin.");
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
            i, short_hex(&p.commitment),
            p.input_seeds.len(), p.destinations.len(), format_age(age),
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
        println!("  [{}] {}", start + i, format_age(age));
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

        // Check privacy delay
        if pending.reveal_not_before > now_secs() {
            let wait = pending.reveal_not_before - now_secs();
            println!("  {} — waiting {}s (privacy delay)", short_hex(&commitment), wait);
            tokio::time::sleep(Duration::from_secs(wait)).await;
        }

        let signatures = wallet.sign_reveal(&pending);

        let url = format!("http://127.0.0.1:{}/send", rpc_port);
        let req = rpc::SendTransactionRequest {
            input_coins: pending.input_coin_ids.iter().map(|c| hex::encode(c)).collect(),
            signatures: signatures.iter().map(|s| hex::encode(wots::sig_to_bytes(s))).collect(),
            destinations: pending.destinations.iter().map(|d| hex::encode(d)).collect(),
            salt: hex::encode(pending.salt),
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

    let contents = std::fs::read_to_string(coinbase_file)?;
    let mut imported = 0usize;

    for line in contents.lines() {
        if line.trim().is_empty() { continue; }

        #[derive(serde::Deserialize)]
        struct CoinbaseEntry {
            height: u64,
            index: u64,
            seed: String,
            #[serde(rename = "coin")]
            _coin: String,
        }

        let entry: CoinbaseEntry = serde_json::from_str(line)?;
        let seed = parse_hex32(&entry.seed)?;

        match wallet.import_seed(seed, Some(format!("coinbase h={} i={}", entry.height, entry.index))) {
            Ok(_) => imported += 1,
            Err(_) => {} // already imported
        }
    }

    println!("Imported {} coinbase reward(s). Total coins: {}", imported, wallet.coin_count());
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
    data_dir: PathBuf, port: u16, rpc_port: u16, peers: Vec<SocketAddr>, mine: bool,
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
        println!("\n⚠️  Save the salt! You need it for the reveal (send) phase.");
    } else {
        let error: rpc::ErrorResponse = response.json().await?;
        println!("Error: {}", error.error);
    }
    Ok(())
}

async fn send_transaction(
    rpc_port: u16,
    input_coins: Vec<String>,
    seeds: Vec<String>,
    destinations: Vec<String>,
    salt: String,
) -> Result<()> {
    if input_coins.is_empty() { anyhow::bail!("Must provide at least one input coin"); }
    if seeds.len() != input_coins.len() { anyhow::bail!("Must provide one seed per input coin"); }

    // Parse inputs and compute WOTS signatures
    let parsed_coins: Vec<[u8; 32]> = input_coins.iter()
        .map(|s| parse_hex32(s))
        .collect::<Result<_>>()?;
    let parsed_dests: Vec<[u8; 32]> = destinations.iter()
        .map(|s| parse_hex32(s))
        .collect::<Result<_>>()?;
    let parsed_salt = parse_hex32(&salt)?;

    let commitment = compute_commitment(&parsed_coins, &parsed_dests, &parsed_salt);

    let mut sigs_hex = Vec::new();
    for seed_hex in &seeds {
        let seed = parse_hex32(seed_hex)?;
        let sig = wots::sign(&seed, &commitment);
        sigs_hex.push(hex::encode(wots::sig_to_bytes(&sig)));
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/send", rpc_port);
    let req = rpc::SendTransactionRequest {
        input_coins,
        signatures: sigs_hex,
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
        println!("  Seed: {}", response.seed);
        println!("  Coin: {}", response.coin);
    } else {
        let seed: [u8; 32] = rand::random();
        let coin = wots::keygen(&seed);
        println!("Generated WOTS keypair:");
        println!("  Seed: {}", hex::encode(seed));
        println!("  Coin: {}", hex::encode(coin));
    }
    println!("\n⚠️  Keep the seed safe! Anyone with it can spend the coin.");
    Ok(())
}

async fn sync_from_genesis(data_dir: PathBuf, peer_addr: SocketAddr) -> Result<()> {
    let storage = storage::Storage::open(data_dir.join("db"))?;
    let syncer = sync::Syncer::new(storage);
    let mut peer = network::PeerConnection::connect(peer_addr, ([127, 0, 0, 1], 0).into()).await?;
    let state = syncer.sync_from_genesis(&mut peer).await?;
    println!("Sync complete!");
    println!("  Height:      {}", state.height);
    println!("  Depth:       {}", state.depth);
    println!("  Coins:       {}", state.coins.len());
    println!("  Commitments: {}", state.commitments.len());
    println!("  Midstate:    {}", hex::encode(state.midstate));
    Ok(())
}
