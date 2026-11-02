//! The embedded midwimble node starts in-process on its own thread, serves
//! its chain state and blocks to walletd, and the MDW account scans what it
//! serves. Offline: the seed is unreachable here, so the chain is genesis.

use midstate_walletd::mwb_account::{BlockView, MwbAccount};
use midstate_walletd::mwb_node::{start_mwb_node, MwbNodeConfig};
use midwimble::core::mw::stealth::WalletKeys;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A bare HTTP/1.0 GET: status line and body.
async fn get(base: &str, path: &str) -> (String, String) {
    let host = base.trim_start_matches("http://").trim_end_matches('/');
    let mut s = tokio::net::TcpStream::connect(host).await.expect("the explorer listens");
    s.write_all(format!("GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n").as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).to_string();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    (head.lines().next().unwrap_or("").to_string(), body.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn the_midwimble_node_starts_serves_its_explorer_and_its_blocks_scan() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = MwbNodeConfig {
        data_dir: dir.path().join("midwimble"),
        p2p_port: 0,
        bootstrap: Vec::new(),
        explorer_port: Some(0),
    };
    let node = start_mwb_node(cfg.clone()).await.expect("the node starts");
    let h = node.handle.clone();

    let state = h.state();
    assert!(state.height >= 1, "genesis is applied");
    let genesis = h.storage().load_batch(0).unwrap().expect("block 0 is stored");

    let keys = WalletKeys::random();
    let mut acct = MwbAccount::starting_at(0);
    let d = acct.scan(&keys, &BlockView::of(0, &genesis)).unwrap();
    assert!(d.received.is_empty(), "nothing in genesis is ours");
    assert_eq!(acct.scanned, 1);
    assert!(h.mempool().await.unwrap().is_empty());

    // The explorer: on loopback, its page and the JSON it reads.
    let url = node.explorer.clone().expect("the explorer is served");
    assert!(url.starts_with("http://127.0.0.1:"), "{url}");
    let (status, page) = get(&url, "/").await;
    assert!(status.contains("200"), "{status}");
    assert!(page.contains("Midwimble"), "the explorer's page");
    let (status, block) = get(&url, "/explorer/block/0").await;
    assert!(status.contains("200"), "{status}: {block}");
    assert!(block.contains("\"height\":0"), "{block}");

    // A second node on the same data directory is refused, not corrupted.
    assert!(start_mwb_node(cfg).await.is_err());
    h.shutdown();
}
