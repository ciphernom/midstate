//! Embeds the midwimble full node in-process, beside midstate's
//! ([`crate::node_host`]), for the MDW account and the MDW leg of swaps.
//!
//! The thread model is `node_host`'s, for the same reason: the libp2p swarm
//! inside the node is `Send` but not `Sync`, so the node's event loop is
//! created and driven on a dedicated OS thread with its own runtime, and only
//! the `NodeHandle` (`Clone + Send + Sync`) leaves it.
//!
//! The embedded node never mines, relays no midstate headers (only block
//! producers need `midstate_rpc`) and serves no RPC: walletd reads its state,
//! storage and mempool through the handle, and submits through it.

use anyhow::{anyhow, Context, Result};
use midwimble::node::{Node, NodeConfig, NodeHandle};
use std::path::PathBuf;

/// Mainnet seed, from midwimble's network manifest (`scripts/testnet.json`;
/// the network that opened as the testnet became mainnet in place on
/// 2026-10-31). It shares a host with midstate's first bootstrap node.
pub const DEFAULT_MWB_BOOTSTRAP: &[&str] =
    &["/ip4/134.199.148.215/tcp/9433/p2p/12D3KooWCvjd2SBokjWMH712NXNFWHKr7kJAeBsUQ447LHk7wrzu"];

#[derive(Clone, Debug)]
pub struct MwbNodeConfig {
    /// Chain database directory, separate from midstate's.
    pub data_dir: PathBuf,
    /// P2P listen port (midwimble's default is 9433).
    pub p2p_port: u16,
    /// Bootstrap peer multiaddrs. Empty ⇒ [`DEFAULT_MWB_BOOTSTRAP`].
    pub bootstrap: Vec<String>,
}

impl Default for MwbNodeConfig {
    fn default() -> Self {
        Self { data_dir: PathBuf::from("./midwimble"), p2p_port: 9433, bootstrap: Vec::new() }
    }
}

/// Starts the embedded midwimble node on its own thread. Resolves once the
/// node has opened its database and identity; sync then runs in the
/// background, observable through the handle (`info().syncing`,
/// `state().height`).
pub async fn start_mwb_node(cfg: MwbNodeConfig) -> Result<NodeHandle> {
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

    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<NodeHandle>>();
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
                let _ = ready_tx.send(Ok(handle));
                if let Err(e) = node.run().await {
                    tracing::error!("midwimble node event loop exited: {e:#}");
                }
            });
        })
        .context("failed to spawn the midwimble node thread")?;

    ready_rx
        .await
        .map_err(|_| anyhow!("midwimble node thread exited before startup finished"))?
}
