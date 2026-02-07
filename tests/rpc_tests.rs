use midstate::node::Node;
use midstate::rpc::RpcServer;
use midstate::core::types::block_reward;
use tempfile::TempDir;

/// Start a node + RPC server, return the port, client, and handles that must stay alive.
async fn start_rpc() -> (u16, reqwest::Client, TempDir, tokio::sync::mpsc::UnboundedReceiver<midstate::node::NodeCommand>) {
    let temp = TempDir::new().unwrap();
    let our_addr = "127.0.0.1:0".parse().unwrap();
    let node = Node::new(temp.path().into(), false, our_addr).unwrap();
    let (handle, cmd_rx) = node.create_handle();

    // Use a random high port to reduce collisions.
    let rpc_port = 18500 + (rand::random::<u16>() % 1000);
    let rpc_server = RpcServer::new(rpc_port);

    tokio::spawn(async move {
        rpc_server.run(handle).await.unwrap();
    });

    // Give the server time to bind
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    (rpc_port, reqwest::Client::new(), temp, cmd_rx)
}

#[tokio::test]
async fn test_rpc_health() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;
    let resp = client.get(format!("http://127.0.0.1:{}/health", port)).send().await.unwrap();
    assert!(resp.status().is_success());
    let body = resp.text().await.unwrap();
    assert_eq!(body, "OK");
}

#[tokio::test]
async fn test_rpc_get_state() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;
    let resp = client.get(format!("http://127.0.0.1:{}/state", port)).send().await.unwrap();
    assert!(resp.status().is_success());

    let json: serde_json::Value = resp.json().await.unwrap();

    // Genesis state assertions
    assert_eq!(json["height"], 0);
    assert_eq!(json["depth"], 0);
    assert_eq!(json["num_coins"], 3); // 3 genesis coins
    assert_eq!(json["num_commitments"], 0);
    assert_eq!(json["block_reward"], block_reward(0) as u64);

    // Midstate and target should be hex strings
    assert!(json["midstate"].as_str().unwrap().len() == 64);
    assert!(json["target"].as_str().unwrap().len() == 64);
}

#[tokio::test]
async fn test_rpc_check_coin_exists() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    // Check a genesis coin
    let seed = midstate::core::hash(b"genesis_coin_1");
    let coin = midstate::core::wots::keygen(&seed);
    let coin_hex = hex::encode(coin);

    let resp = client
        .post(format!("http://127.0.0.1:{}/check", port))
        .json(&serde_json::json!({ "coin": coin_hex }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["exists"], true);
    assert_eq!(json["coin"], coin_hex);
}

#[tokio::test]
async fn test_rpc_check_coin_not_exists() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let fake_coin = hex::encode([0xAA; 32]);
    let resp = client
        .post(format!("http://127.0.0.1:{}/check", port))
        .json(&serde_json::json!({ "coin": fake_coin }))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["exists"], false);
}

#[tokio::test]
async fn test_rpc_check_coin_bad_hex() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let resp = client
        .post(format!("http://127.0.0.1:{}/check", port))
        .json(&serde_json::json!({ "coin": "not_hex!" }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_rpc_keygen() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let resp = client
        .get(format!("http://127.0.0.1:{}/keygen", port))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();

    let seed_hex = json["seed"].as_str().unwrap();
    let coin_hex = json["coin"].as_str().unwrap();

    // Should be valid 32-byte hex strings
    assert_eq!(seed_hex.len(), 64);
    assert_eq!(coin_hex.len(), 64);

    // The coin should be derivable from the seed
    let seed_bytes: [u8; 32] = hex::decode(seed_hex).unwrap().try_into().unwrap();
    let derived = midstate::core::wots::keygen(&seed_bytes);
    assert_eq!(hex::encode(derived), coin_hex);
}

#[tokio::test]
async fn test_rpc_get_peers() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let resp = client
        .get(format!("http://127.0.0.1:{}/peers", port))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["peers"].is_array());
}

#[tokio::test]
async fn test_rpc_get_mempool_empty() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let resp = client
        .get(format!("http://127.0.0.1:{}/mempool", port))
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["size"], 0);
    assert!(json["transactions"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_rpc_commit_empty_coins_rejected() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let bad_commit = serde_json::json!({
        "coins": [],
        "destinations": []
    });

    let resp = client
        .post(format!("http://127.0.0.1:{}/commit", port))
        .json(&bad_commit)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["error"].as_str().unwrap().contains("coin"));
}

#[tokio::test]
async fn test_rpc_send_empty_inputs_rejected() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let bad_tx = serde_json::json!({
        "input_coins": [],
        "signatures": [],
        "destinations": [],
        "salt": ""
    });

    let resp = client
        .post(format!("http://127.0.0.1:{}/send", port))
        .json(&bad_tx)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_rpc_send_sig_count_mismatch_rejected() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    let bad_tx = serde_json::json!({
        "input_coins": [hex::encode([1u8; 32])],
        "signatures": [],  // 0 sigs for 1 input
        "destinations": [hex::encode([2u8; 32])],
        "salt": hex::encode([0u8; 32])
    });

    let resp = client
        .post(format!("http://127.0.0.1:{}/send", port))
        .json(&bad_tx)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json["error"].as_str().unwrap().contains("Signature count"));
}

#[tokio::test]
async fn test_rpc_commit_valid_flow() {
    let (port, client, _temp, _cmd_rx) = start_rpc().await;

    // Use a genesis coin
    let seed = midstate::core::hash(b"genesis_coin_1");
    let coin = midstate::core::wots::keygen(&seed);
    let dest: [u8; 32] = rand::random();

    let commit_req = serde_json::json!({
        "coins": [hex::encode(coin)],
        "destinations": [hex::encode(dest)]
    });

    let resp = client
        .post(format!("http://127.0.0.1:{}/commit", port))
        .json(&commit_req)
        .send()
        .await
        .unwrap();

    assert!(resp.status().is_success(), "Commit should succeed");
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["status"], "committed");
    assert_eq!(json["commitment"].as_str().unwrap().len(), 64);
    assert_eq!(json["salt"].as_str().unwrap().len(), 64);
}
