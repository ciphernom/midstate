//! The embedded midwimble node starts in-process on its own thread, serves
//! its chain state and blocks to walletd, and the MDW account scans what it
//! serves. Offline: the seed is unreachable here, so the chain is genesis.

use midstate_walletd::mwb_account::{BlockView, MwbAccount};
use midstate_walletd::mwb_node::{start_mwb_node, MwbNodeConfig};
use midwimble::core::mw::stealth::WalletKeys;

#[tokio::test(flavor = "multi_thread")]
async fn the_midwimble_node_starts_and_its_blocks_scan() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = MwbNodeConfig { data_dir: dir.path().join("midwimble"), p2p_port: 0, bootstrap: Vec::new() };
    let h = start_mwb_node(cfg.clone()).await.expect("the node starts");

    let state = h.state();
    assert!(state.height >= 1, "genesis is applied");
    let genesis = h.storage().load_batch(0).unwrap().expect("block 0 is stored");

    let keys = WalletKeys::random();
    let mut acct = MwbAccount::starting_at(0);
    let d = acct.scan(&keys, &BlockView::of(0, &genesis)).unwrap();
    assert!(d.received.is_empty(), "nothing in genesis is ours");
    assert_eq!(acct.scanned, 1);
    assert!(h.mempool().await.unwrap().is_empty());

    // A second node on the same data directory is refused, not corrupted.
    assert!(start_mwb_node(cfg).await.is_err());
    h.shutdown();
}
