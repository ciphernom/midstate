use super::handlers::*;
use crate::node::NodeHandle;
use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tower_http::trace::TraceLayer;

pub struct RpcServer {
    addr: SocketAddr,
}

impl RpcServer {
    pub fn new(port: u16) -> Self {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        Self { addr }
    }

    pub async fn run(self, node_handle: NodeHandle) -> Result<()> {
        let app = Router::new()
            .route("/health", get(health))
            .route("/state", get(get_state))
            .route("/commit", post(commit_transaction))
            .route("/send", post(send_transaction))
            .route("/check", post(check_coin))
            .route("/mempool", get(get_mempool))
            .route("/keygen", get(generate_key))
            .route("/peers", get(get_peers))
            .route("/scan", post(scan_addresses))
            .route("/scan_stealth", post(scan_stealth))
            .route("/mss_state", post(get_mss_state))
            .layer(TraceLayer::new_for_http())
            .with_state(node_handle);

        tracing::info!("RPC server listening on {}", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}
