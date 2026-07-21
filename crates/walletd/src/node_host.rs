//! Embeds the Midstate full node in-process.
//!
//! Threading note (the reason this file looks the way it does): the libp2p
//! `Swarm` inside `MidstateNetwork` is `Send` but **not `Sync`**, and
//! `Node::run` holds `&self` across await points, so the `node.run(..)`
//! future is `!Send`. `tokio::spawn` requires `Send`, so the node loop can
//! never be spawned onto a shared runtime. The upstream binary avoids this
//! because `#[tokio::main]` drives `node.run` via `block_on`, which has no
//! `Send` bound. We reproduce that exactly: the node gets a dedicated OS
//! thread with its own tokio runtime, is *created* on that thread, and is
//! driven there with `block_on`. Only the `NodeHandle` (`Send + Sync`; it
//! already crosses threads as axum state upstream) leaves the thread.
//!
//! As before, the desktop app never mines (mining_threads = None) and never
//! opens a license wallet at the node layer (plan §12: single writer on
//! wallet.dat).

use anyhow::{anyhow, Context, Result};
use midstate::node::{Node, NodeHandle};
use midstate::rpc::RpcServer;
use std::collections::HashSet;
use std::path::PathBuf;

/// Mainnet bootstrap peers, from the node's own default config template.
pub const DEFAULT_BOOTSTRAP: &[&str] = &[
    "/ip4/134.199.148.215/tcp/9333/p2p/12D3KooWPbR63SQg1UBLpAMiNngqrRHGM4LaMP8ieAJUxhfw7dxv",
    "/ip4/74.208.253.44/tcp/9333/p2p/12D3KooWBqph3BWQxc3xsusvCijS88RaAEZagZZwwxAwP2Xs1CTE",
];

#[derive(Clone, Debug)]
pub struct NodeConfig {
    /// Chain database directory (platform data dir; see the app's boot()).
    pub data_dir: PathBuf,
    /// P2P listen port (chain default 9333).
    pub p2p_port: u16,
    /// Bootstrap peer multiaddrs. Empty ⇒ DEFAULT_BOOTSTRAP.
    pub bootstrap: Vec<String>,
    /// Bind the HTTP RPC (block explorer) on 127.0.0.1:port. None ⇒ off.
    pub rpc_port: Option<u16>,
    /// Retain only recent blocks. Default false (archival) — flipping this
    /// takes effect on restart, matching node behavior.
    pub prune: bool,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            p2p_port: 9333,
            bootstrap: Vec::new(),
            rpc_port: Some(8545),
            prune: false,
        }
    }
}

/// Start the embedded node on its own thread. Resolves once the node has
/// initialized (storage open, identity loaded); sync proceeds in the
/// background and is observable through the returned `NodeHandle`.
pub async fn start_node(cfg: NodeConfig) -> Result<NodeHandle> {
    // Parse addresses up front so configuration errors fail fast on the caller.
    let listen_addr: libp2p::Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", cfg.p2p_port)
        .parse()
        .context("invalid p2p listen multiaddr")?;

    let bootstrap_src: Vec<String> = if cfg.bootstrap.is_empty() {
        DEFAULT_BOOTSTRAP.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.bootstrap.clone()
    };
    let bootstrap: Vec<libp2p::Multiaddr> = bootstrap_src
        .iter()
        .map(|s| s.parse::<libp2p::Multiaddr>())
        .collect::<Result<Vec<_>, _>>()
        .context("invalid bootstrap peer multiaddr")?;

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<NodeHandle>>();
    let data_dir = cfg.data_dir.clone();
    let rpc_port = cfg.rpc_port;
    let prune = cfg.prune;

    std::thread::Builder::new()
        .name("midstate-node".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("midstate-node-io")
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow!("could not build node runtime: {e}")));
                    return;
                }
            };

            rt.block_on(async move {
                // Created on this thread so every internal resource belongs
                // to this runtime.
                let node = match Node::new(
                    data_dir,
                    None, // never mine from the desktop app
                    listen_addr,
                    bootstrap,
                    HashSet::new(),
                    prune,
                )
                .await
                {
                    Ok(n) => n,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.context(
                            "node startup failed (is another instance using this data dir?)",
                        )));
                        return;
                    }
                };

                let (handle, cmd_rx) = node.create_handle();

                if let Some(port) = rpc_port {
                    // Loopback only — the desktop build never exposes RPC
                    // beyond the machine.
                    match RpcServer::new("127.0.0.1", port) {
                        Ok(rpc) => {
                            let rpc_handle = handle.clone();
                            tokio::spawn(async move {
                                if let Err(e) = rpc.run(rpc_handle).await {
                                    tracing::error!("embedded RPC server exited: {e:#}");
                                }
                            });
                        }
                        Err(e) => tracing::warn!("explorer RPC disabled: {e:#}"),
                    }
                }

                let _ = ready_tx.send(Ok(handle.clone()));

                // The !Send future — driven here via block_on, never spawned.
                if let Err(e) = node.run(handle, cmd_rx).await {
                    tracing::error!("node event loop exited: {e:#}");
                }
            });
        })
        .context("failed to spawn the node thread")?;

    ready_rx
        .await
        .map_err(|_| anyhow!("node thread exited before startup finished"))?
}
