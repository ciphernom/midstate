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
use tower_http::timeout::TimeoutLayer;
use std::time::Duration;
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

/// True if `host` (a Host header value, optionally with a port) names loopback, a
/// private/link-local address, or `localhost`.
fn host_is_local(host: &str) -> bool {
    let h = host.trim();
    let h = if h.starts_with('[') {
        // IPv6 literal: [::1] or [::1]:port
        h.trim_start_matches('[').split(']').next().unwrap_or(h)
    } else {
        h.rsplit_once(':').map(|(a, _)| a).unwrap_or(h)
    };
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    IpAddr::from_str(h).map(is_lan_ip).unwrap_or(false)
}

/// Audit MW1: under a remote-only web-wallet model, no first-party site needs to
/// reach a visitor's LOCAL node. Reject a cross-origin request (Origin present and
/// not matching Host) whenever the Host is loopback/LAN — exactly a public page
/// reaching into a local node, which the CORS allowlist otherwise permits (the
/// /axe payout-redirect among them). Same-origin requests (the node's own served
/// pages, which send no Origin or a matching one) and cross-origin requests to a
/// public node (non-local Host) are unaffected.
async fn block_cross_origin_to_local(req: Request, next: Next) -> Result<Response, StatusCode> {
    if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok()) {
        let host = req.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
        let origin_authority = origin.split("://").nth(1).unwrap_or(origin);
        if origin_authority != host && host_is_local(host) {
            tracing::warn!(
                "Blocked cross-origin request to a local node (Origin {:?}, Host {:?}) — audit MW1",
                origin, host
            );
            return Err(StatusCode::FORBIDDEN);
        }
    }
    Ok(next.run(req).await)
}

/// Axum middleware to drop requests originating from the public internet.
///
/// SECURITY: `X-Forwarded-For` is attacker-controlled. The previous version
/// read the FIRST entry of `X-Forwarded-For` (falling back to `X-Real-IP`) and
/// allowed the request whenever that parsed as loopback/LAN. Because the header
/// is supplied by the client, anyone who could reach this port could send
/// `X-Forwarded-For: 127.0.0.1` and reach the `/axe` hardware-management routes
/// (Wi-Fi setup, overclock, config, rewards) from the public internet.
///
/// The correct model (identical to `handlers::send_chat`) is:
///   1. The IMMEDIATE peer — the real TCP socket address, which the client
///      cannot forge — must itself be LAN/loopback. A WAN socket is rejected
///      outright no matter what headers it carries.
///   2. Only then, if a *trusted local* reverse proxy forwarded the request,
///      EVERY hop in the `X-Forwarded-For` chain must also be LAN. A single
///      public IP anywhere in the chain proves the request crossed the WAN at
///      some hop; a malformed entry fails closed.
///
/// This still supports the intended deployment (a loopback/LAN Nginx fronting
/// the node) while making header spoofing from a non-local socket ineffective.
async fn lan_only_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 1. The real socket must be local. This is the gate the client cannot spoof.
    if !is_lan_ip(addr.ip()) {
        tracing::warn!("Blocked direct WAN access to hardware endpoint from {}", addr.ip());
        return Err(StatusCode::FORBIDDEN);
    }

    // 2. The socket is local (e.g. a loopback/LAN reverse proxy, or localhost
    //    itself). If that proxy is forwarding on behalf of someone, require the
    //    entire forwarded chain to be LAN as well — one public hop means the
    //    request originated off-LAN and must be rejected.
    if let Some(forwarded) = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        for ip_str in forwarded.split(',') {
            match ip_str.trim().parse::<IpAddr>() {
                Ok(ip) if !is_lan_ip(ip) => {
                    tracing::warn!(
                        "Blocked proxied WAN access to hardware endpoint: XFF chain contains {}",
                        ip
                    );
                    return Err(StatusCode::FORBIDDEN);
                }
                Ok(_) => {} // LAN hop — keep checking the rest
                Err(_) => {
                    tracing::warn!(
                        "Blocked hardware endpoint: malformed XFF entry {:?}",
                        ip_str.trim()
                    );
                    return Err(StatusCode::FORBIDDEN);
                }
            }
        }
    }

    // Audit J27: on a state-changing request, reject a cross-origin `Origin`. A page
    // on another site that tries to POST to /axe/config (CSRF, or DNS-rebinding to
    // loopback) carries its own Origin; the loopback dashboard is same-origin and
    // sends none or a matching one. The LAN gate above admits the loopback socket, so
    // without this a malicious page in the operator's browser could redirect the
    // miner. Reads (GET/HEAD/OPTIONS) are unaffected.
    let is_write = !matches!(req.method().as_str(), "GET" | "HEAD" | "OPTIONS");
    if is_write {
        if let Some(origin) = req.headers().get("origin").and_then(|v| v.to_str().ok()) {
            let host = req.headers().get("host").and_then(|v| v.to_str().ok()).unwrap_or("");
            // Origin is scheme://authority; compare its authority to the Host header.
            let origin_authority = origin.split("://").nth(1).unwrap_or(origin);
            if origin_authority != host {
                tracing::warn!(
                    "Blocked cross-origin state-changing request to hardware endpoint (Origin {:?} vs Host {:?})",
                    origin, host
                );
                return Err(StatusCode::FORBIDDEN);
            }
        }
    }

    Ok(next.run(req).await)
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
            "https://midstate.cash".parse::<HeaderValue>().unwrap(), 
            "https://www.midstate.cash".parse::<HeaderValue>().unwrap(),
            "https://mds.cash".parse::<HeaderValue>().unwrap(), 
            "https://www.mds.cash".parse::<HeaderValue>().unwrap(),
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
            .route("/config", get(axe_get_config).post(axe_save_config))
            .route("/overclock", post(axe_apply_overclock))
            .route("/rewards", get(axe_download_rewards))
            .route_layer(middleware::from_fn(lan_only_middleware));

        let app = Router::new()
            .route("/", get(explorer_ui))
            .route("/chat", get(chat_ui))
            .route("/midstate.css", get(midstate_css))   
            .route("/api/chat", get(get_chat).post(send_chat))    
            .route("/api/chat/submit", post(submit_chat))     
            .route("/batch/:height", get(get_batch))
            .route("/search", post(search))
            .route("/coin/:coin_id", get(check_coin_get))
            .route("/block/:height", get(get_block_raw))
            .route("/health", get(health))
            .route("/filters", post(get_filters))
            .route("/state", get(get_state))
            .route("/utxo_proof/:coin_id", get(get_utxo_proof))
            .route("/headers/:start/:count", get(get_headers))
            .route("/stats/history", get(get_chain_stats))
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
            .route("/submit_batch", post(submit_batch))
            .route("/mix/create", post(mix_create))
            .route("/mix/register", post(mix_register))
            .route("/mix/fee", post(mix_fee))
            .route("/mix/sign", post(mix_sign))
            .route("/mix/status/:mix_id", get(mix_status))
            .route("/pow_params", get(pow_params))
            .route("/mix/list", get(mix_list))
            .nest("/axe", axe_routes) 
            .route("/tx/by_input", post(get_tx_by_input))
            // 16 MB max request body. The binding case is /submit_batch, which posts a
            // full block as JSON: blocks are legal up to MAX_BLOCK_BYTES (8 MB, node.rs),
            // and hex/JSON encoding inflates that further, so a 2 MB cap silently killed
            // large-but-legal submissions mid-stream (client saw a broken pipe, not a 413).
            // 16 MB clears the worst legal block with headroom and also covers a maximal
            // multi-KB-witness reveal near MAX_TX_INPUTS.
            .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
            .layer(TimeoutLayer::new(Duration::from_secs(30))) // Forcefully drop stalled requests
            .layer(TraceLayer::new_for_http())
            .layer(cors)
            .layer(middleware::from_fn(block_cross_origin_to_local))
            .with_state(node_handle);

        tracing::info!("RPC server listening on {}", self.addr);
        tracing::info!("💬 Midstate Chat available at: http://{}/chat", self.addr);

        let listener = tokio::net::TcpListener::bind(self.addr).await?;
        
        // --- Provide Connection Info to Axum so the middleware can read the IP ---
        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;

        Ok(())
    }
}
