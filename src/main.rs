use anyhow::Result;
use clap::{Parser, Subcommand};
use midstate::*;
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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

async fn commit_transaction(rpc_port: u16, coins: Vec<String>, destinations: Vec<String>) -> Result<()> {
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

    let response = client
        .post(&url)
        .json(&req)
        .send()
        .await?;

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

async fn send_transaction(rpc_port: u16, secrets: Vec<String>, destinations: Vec<String>, salt: String) -> Result<()> {
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

    let response = client
        .post(&url)
        .json(&req)
        .send()
        .await?;

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

    let response = client
        .post(&url)
        .json(&req)
        .send()
        .await?;

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
