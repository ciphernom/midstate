//! Embeds the midwimble full node in-process, beside midstate's
//! ([`crate::node_host`]), for the MDW account and the MDW leg of swaps.
//!
//! The thread model is `node_host`'s, for the same reason: the libp2p swarm
//! inside the node is `Send` but not `Sync`, so the node's event loop is
//! created and driven on a dedicated OS thread with its own runtime, and only
//! the `NodeHandle` (`Clone + Send + Sync`) leaves it.
//!
//! The embedded node never mines by itself. It follows midstate's headers
//! through midstate's node RPC when it has one, so the blocks this wallet's
//! bond merge-mines can relay what a registration needs (`mwb_mining`). It
//! serves no RPC: walletd reads its state, storage and mempool through the
//! handle, and submits through it.
//! What it does serve is midwimble's block explorer, read-only and on
//! loopback, so every MDW payment the wallet shows is one click from the
//! chain this machine verified.

use anyhow::{anyhow, Context, Result};
use midwimble::node::{Node, NodeConfig, NodeHandle};
use std::path::PathBuf;

/// Mainnet seed, from midwimble's network manifest (`scripts/testnet.json`;
/// the network that opened as the testnet became mainnet in place on
/// 2026-10-31). It shares a host with midstate's first bootstrap node.
pub const DEFAULT_MWB_BOOTSTRAP: &[&str] =
    &["/ip4/134.199.148.215/tcp/9433/p2p/12D3KooWCvjd2SBokjWMH712NXNFWHKr7kJAeBsUQ447LHk7wrzu"];

/// The explorer's port: midwimble's own suggestion for `--explorer`, one
/// above its RPC (9434).
pub const DEFAULT_EXPLORER_PORT: u16 = 9435;

#[derive(Clone, Debug)]
pub struct MwbNodeConfig {
    /// Chain database directory, separate from midstate's.
    pub data_dir: PathBuf,
    /// P2P listen port (midwimble's default is 9433).
    pub p2p_port: u16,
    /// Bootstrap peer multiaddrs. Empty ⇒ [`DEFAULT_MWB_BOOTSTRAP`].
    pub bootstrap: Vec<String>,
    /// Loopback port for the block explorer. `None` serves none, `Some(0)`
    /// any free port; a port already taken falls back to a free one.
    pub explorer_port: Option<u16>,
    /// Midstate's node RPC (`host:port`), polled for the headers midwimble's
    /// blocks relay. A bond registers only while the relay holds its proof's
    /// header, so a node that may mine one in should follow midstate.
    pub midstate_rpc: Option<String>,
}

impl Default for MwbNodeConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./midwimble"),
            p2p_port: 9433,
            bootstrap: Vec::new(),
            explorer_port: Some(DEFAULT_EXPLORER_PORT),
            midstate_rpc: None,
        }
    }
}

/// The running node, and where to find it.
#[derive(Clone)]
pub struct MwbNode {
    pub handle: NodeHandle,
    /// `http://127.0.0.1:<port>/`, while the explorer is served.
    pub explorer: Option<String>,
    pub data_dir: PathBuf,
    pub p2p_port: u16,
}

/// Starts the embedded midwimble node on its own thread. Resolves once the
/// node has opened its database and identity; sync then runs in the
/// background, observable through the handle (`info().syncing`,
/// `state().height`).
pub async fn start_mwb_node(cfg: MwbNodeConfig) -> Result<MwbNode> {
    let listen: libp2p::Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", cfg.p2p_port)
        .parse()
        .context("invalid midwimble listen multiaddr")?;
    let sources: Vec<String> = if cfg.bootstrap.is_empty() {
        DEFAULT_MWB_BOOTSTRAP.iter().map(|s| s.to_string()).collect()
    } else {
        cfg.bootstrap.clone()
    };
    let bootstrap = sources
        .iter()
        .map(|s| s.parse::<libp2p::Multiaddr>())
        .collect::<Result<Vec<_>, _>>()
        .context("invalid midwimble bootstrap multiaddr")?;
    std::fs::create_dir_all(&cfg.data_dir)
        .with_context(|| format!("creating {}", cfg.data_dir.display()))?;

    let mut config = NodeConfig::new(cfg.data_dir.clone(), listen);
    config.bootstrap = bootstrap;
    config.midstate_rpc = cfg.midstate_rpc.clone();
    let explorer_port = cfg.explorer_port;

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(NodeHandle, Option<String>)>>();
    std::thread::Builder::new()
        .name("midwimble-node".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_name("midwimble-node-io")
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    let _ = ready_tx.send(Err(anyhow!("could not build the midwimble runtime: {e}")));
                    return;
                }
            };
            rt.block_on(async move {
                let (node, handle) = match Node::new(config).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.context(
                            "midwimble node startup failed (is another instance using its data dir?)",
                        )));
                        return;
                    }
                };
                let explorer = match explorer_port {
                    Some(port) => serve_explorer(&handle, port).await,
                    None => None,
                };
                let _ = ready_tx.send(Ok((handle, explorer)));
                if let Err(e) = node.run().await {
                    tracing::error!("midwimble node event loop exited: {e:#}");
                }
            });
        })
        .context("failed to spawn the midwimble node thread")?;

    let (handle, explorer) = ready_rx
        .await
        .map_err(|_| anyhow!("midwimble node thread exited before startup finished"))??;
    Ok(MwbNode { handle, explorer, data_dir: cfg.data_dir, p2p_port: cfg.p2p_port })
}

/// Binds the explorer on loopback (`port`, or any free one if that is
/// taken) and serves it on the node's runtime. The explorer only reads, and
/// builds its index the first time a page asks, so an unopened explorer
/// costs nothing. A failure leaves the wallet as it is, without the link.
async fn serve_explorer(handle: &NodeHandle, port: u16) -> Option<String> {
    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(l) => l,
        Err(e) if port != 0 => {
            tracing::info!("midwimble explorer: port {port} is taken ({e}); using a free one");
            match tokio::net::TcpListener::bind(("127.0.0.1", 0)).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("midwimble explorer not served: {e}");
                    return None;
                }
            }
        }
        Err(e) => {
            tracing::warn!("midwimble explorer not served: {e}");
            return None;
        }
    };
    let url = format!("http://{}/", listener.local_addr().ok()?);
    let node = handle.clone();
    tokio::spawn(async move {
        let index = midwimble::explorer::SharedIndex::default();
        if let Err(e) = midwimble::rpc::serve_explorer_on(node, listener, index).await {
            tracing::warn!("midwimble explorer stopped: {e:#}");
        }
    });
    tracing::info!("midwimble explorer at {url}");
    Some(url)
}
