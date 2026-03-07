use super::handlers::*;
use crate::node::NodeHandle;
use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};

use tower_http::trace::TraceLayer;
use tower_http::cors::CorsLayer;
use axum::http::{Method, HeaderValue, header};
use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

pub struct RpcServer {
    addr: SocketAddr,
}

impl RpcServer {
    pub fn new(bind_ip: &str, port: u16) -> anyhow::Result<Self> {
        let ip = IpAddr::from_str(bind_ip)
            .map_err(|e| anyhow::anyhow!("Invalid RPC bind IP: {}", e))?;
        let addr = SocketAddr::new(ip, port);
        Ok(Self { addr })
    }

    pub async fn run(self, node_handle: NodeHandle) -> Result<()> {
        
        // Create a strict list of allowed origins
        let allowed_origins = [
            "http://localhost:8080".parse::<HeaderValue>().unwrap(),
            "https://ciphernom.github.io".parse::<HeaderValue>().unwrap(), 
        ];
        
        let cors = CorsLayer::new()
            .allow_origin(allowed_origins)
            .allow_methods([Method::GET, Method::POST])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT]);
            
        let app = Router::new()
            .route("/", get(explorer_ui))
            .route("/batch/:height", get(get_batch))
            .route("/search", post(search))
            .route("/coin/:coin_id", get(check_coin_get))
            .route("/block/:height", get(get_block_raw))
            .route("/health", get(health))
            .route("/filters", post(get_filters))
            .route("/state", get(get_state))
            .route("/commit", post(commit_transaction))
            .route("/send", post(send_transaction))
            .route("/check", post(check_coin))
            .route("/check_commitment", post(check_commitment))
            .route("/mempool", get(get_mempool))
            .route("/keygen", get(generate_key))
            .route("/peers", get(get_peers))
            .route("/scan", post(scan_addresses))
            .route("/mss_state", post(get_mss_state))
            .route("/mix/create", post(mix_create))
            .route("/mix/register", post(mix_register))
            .route("/mix/fee", post(mix_fee))
            .route("/mix/sign", post(mix_sign))
            .route("/mix/status/:mix_id", get(mix_status))
            .route("/mix/list", get(mix_list))
            .route("/axe", get(axe_ui))
            .route("/axe/stats", get(axe_stats))
            .route("/axe/wifi", post(axe_wifi_setup)) // Captive portal endpoint
            .route("/axe/config", post(axe_save_config)) // Pool/Miner config endpoint
            .route("/axe/overclock", post(axe_apply_overclock))
            .route("/axe/rewards", get(axe_download_rewards))
            .layer(TraceLayer::new_for_http())
            .layer(cors)
            .with_state(node_handle);

        tracing::info!("RPC server listening on {}", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}
