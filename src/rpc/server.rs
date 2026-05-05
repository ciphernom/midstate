use super::handlers::*;
use crate::node::NodeHandle;
use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
    extract::{ConnectInfo, Request},
    middleware::{self, Next},
    response::Response,
};

use tower_http::trace::TraceLayer;
use tower_http::cors::{CorsLayer, Any};  
use axum::http::{Method, HeaderValue, StatusCode};
use axum::extract::DefaultBodyLimit;

use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;

/// Checks if an IP address belongs to localhost or a private LAN subnet.
fn is_lan_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ipv4) => {
            ipv4.is_loopback() || ipv4.is_private() || ipv4.is_link_local()
        }
        IpAddr::V6(ipv6) => {
            ipv6.is_loopback() || 
            (ipv6.segments()[0] & 0xfe00) == 0xfc00 || // Unique local (fc00::/7)
            (ipv6.segments()[0] & 0xffc0) == 0xfe80    // Link local (fe80::/10)
        }
    }
}

/// Axum middleware to drop requests originating from the public internet.
async fn lan_only_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let ip = addr.ip();
    if is_lan_ip(ip) {
        Ok(next.run(req).await)
    } else {
        tracing::warn!("Blocked WAN access to hardware endpoint from {}", ip);
        Err(StatusCode::FORBIDDEN)
    }
}

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
        
        // Strict whitelist for Origins, but permissive for Preflight (OPTIONS) and Headers
        let allowed_origins = [
            "http://localhost:8080".parse::<HeaderValue>().unwrap(),
            "https://ciphernom.github.io".parse::<HeaderValue>().unwrap(), 
            "https://cypherpunk.gold".parse::<HeaderValue>().unwrap(), 
            "https://www.cypherpunk.gold".parse::<HeaderValue>().unwrap(), 
        ];
        
        let cors = CorsLayer::new()
            .allow_origin(allowed_origins)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(Any);
            
        // --- Isolate Axe hardware routes and protect them ---
        let axe_routes = Router::new()
            .route("/", get(axe_ui))
            .route("/stats", get(axe_stats))
            .route("/wifi", post(axe_wifi_setup))
            .route("/config", post(axe_save_config))
            .route("/overclock", post(axe_apply_overclock))
            .route("/rewards", get(axe_download_rewards))
            .route_layer(middleware::from_fn(lan_only_middleware));

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
            .route("/check_output", post(check_output))
            .route("/check_commitment", post(check_commitment))
            .route("/mempool", get(get_mempool))
            .route("/keygen", get(generate_key))
            .route("/peers", get(get_peers))
            .route("/metrics", get(get_metrics))
            .route("/scan", post(scan_addresses))
            .route("/mss_state", post(get_mss_state))
            .route("/block_template", post(block_template))
            .route("/mix/create", post(mix_create))
            .route("/mix/register", post(mix_register))
            .route("/mix/fee", post(mix_fee))
            .route("/mix/sign", post(mix_sign))
            .route("/mix/status/:mix_id", get(mix_status))
            .route("/mix/list", get(mix_list))
            .nest("/axe", axe_routes) 
            .route("/api/internal/submit_batch", post(submit_batch))
            .route("/tx/by_input", post(get_tx_by_input))
            .layer(DefaultBodyLimit::max(2 * 1024 * 1024)) // 2 MB max request body
            .layer(TraceLayer::new_for_http())
            .layer(cors)
            .with_state(node_handle);

        tracing::info!("RPC server listening on {}", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        
        // --- Provide Connection Info to Axum so the middleware can read the IP ---
        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;

        Ok(())
    }
}
