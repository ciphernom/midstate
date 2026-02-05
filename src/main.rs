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
    
    /// Send a transaction via RPC
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
        
        Command::Send { rpc_port, secret, dest } => {
            send_transaction(rpc_port, secret, dest).await
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

async fn send_transaction(rpc_port: u16, secrets: Vec<String>, destinations: Vec<String>) -> Result<()> {
    if secrets.is_empty() {
        anyhow::bail!("Must provide at least one secret");
    }
    
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}/send", rpc_port);
    
    let req = rpc::SendTransactionRequest {
        secrets,
        destinations,
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
    println!("  Height:   {}", response.height);
    println!("  Depth:    {}", response.depth);
    println!("  Coins:    {}", response.num_coins);
    println!("  Midstate: {}", response.midstate);
    println!("  Target:   {}", response.target);
    
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
            println!("  {}:", i + 1);
            for (j, input) in tx.input_coins.iter().enumerate() {
                println!("    Input {}: {}", j, input);
            }
            for (j, output) in tx.output_coins.iter().enumerate() {
                println!("    Output {}: {}", j, output);
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
