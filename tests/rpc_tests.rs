use midstate::node::Node;
use midstate::rpc::RpcServer;
use tempfile::TempDir;

#[tokio::test]
async fn test_rpc_server_integration() {
    // 1. Start a Node
    let temp = TempDir::new().unwrap();
    let node = Node::new(temp.path().into(), false, "127.0.0.1:0".parse().unwrap()).unwrap();
    let (handle, _) = node.create_handle();

    // 2. Start RPC Server
    let rpc_port = 8599;
    let rpc_server = RpcServer::new(rpc_port);

    tokio::spawn(async move {
        rpc_server.run(handle).await.unwrap();
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // 3. Test HTTP Client
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{}", rpc_port);

    // Test: Health
    let resp = client.get(format!("{}/health", url)).send().await.unwrap();
    assert!(resp.status().is_success());

    // Test: Get State (now includes num_commitments)
    let resp = client.get(format!("{}/state", url)).send().await.unwrap();
    assert!(resp.status().is_success());
    let json: serde_json::Value = resp.json().await.unwrap();
    assert!(json.get("height").is_some());
    assert!(json.get("num_commitments").is_some());

    // Test: Send Transaction (missing salt should fail)
    let bad_tx = serde_json::json!({
        "secrets": [],
        "destinations": [],
        "salt": ""
    });

    let resp = client.post(format!("{}/send", url))
        .json(&bad_tx)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);

    // Test: Commit endpoint (empty coins should fail)
    let bad_commit = serde_json::json!({
        "coins": [],
        "destinations": []
    });

    let resp = client.post(format!("{}/commit", url))
        .json(&bad_commit)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}
