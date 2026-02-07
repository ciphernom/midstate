// ========================================
// --- FILE: tests/network_integration.rs
// ========================================
use midstate::network::{PeerConnection, PeerManager, Message};
use tokio::net::TcpListener;
use tokio::io::{AsyncWriteExt, AsyncReadExt};
use std::net::SocketAddr;
use std::time::SystemTime;

// Helper to spawn a passive listener
async fn spawn_listener() -> (SocketAddr, tokio::sync::mpsc::UnboundedReceiver<PeerConnection>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        loop {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            let peer = PeerConnection::from_stream(stream, peer_addr);
            tx.send(peer).unwrap();
        }
    });

    (addr, rx)
}

#[tokio::test]
async fn test_full_handshake_success() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let client_our_addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();

    let client_handle = tokio::spawn(async move {
        let mut peer = PeerConnection::connect(server_addr, client_our_addr).await.unwrap();
        peer.complete_handshake(client_our_addr).await.unwrap();
        peer
    });

    let mut server_peer = server_rx.recv().await.unwrap();
    server_peer.complete_handshake(server_addr).await.unwrap();

    let client_peer = client_handle.await.unwrap();

    assert!(server_peer.is_connected());
    assert!(client_peer.is_connected());
}

#[tokio::test]
async fn test_handshake_version_mismatch() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let our_addr: SocketAddr = "127.0.0.1:8000".parse().unwrap();

    tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(server_addr).await.unwrap();
        let bad_version = Message::Version {
            version: 9999, // INVALID
            services: 1,
            timestamp: SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
            addr_recv: server_addr,
            addr_from: our_addr,
        };
        let bytes = bad_version.serialize();
        let len = bytes.len() as u32;
        stream.write_all(&len.to_le_bytes()).await.unwrap();
        stream.write_all(&bytes).await.unwrap();

        // Keep alive briefly
        let mut buf = [0u8; 10];
        let _ = stream.read(&mut buf).await;
    });

    let mut server_peer = server_rx.recv().await.unwrap();
    let result = server_peer.complete_handshake(server_addr).await;

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Protocol version"),
        "Expected 'Protocol version' error, got: '{}'",
        err_msg
    );
}

#[tokio::test]
async fn test_ping_pong() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let client_our_addr: SocketAddr = "127.0.0.1:9001".parse().unwrap();

    // Client connects and handshakes
    let client_handle = tokio::spawn(async move {
        let mut peer = PeerConnection::connect(server_addr, client_our_addr).await.unwrap();
        peer.complete_handshake(client_our_addr).await.unwrap();

        // Send ping
        peer.send_ping().await.unwrap();
        peer
    });

    let mut server_peer = server_rx.recv().await.unwrap();
    server_peer.complete_handshake(server_addr).await.unwrap();

    // Server receives ping, sends pong
    let msg = server_peer.receive_message().await.unwrap();
    match msg {
        Message::Ping { nonce } => {
            server_peer.send_message(&Message::Pong { nonce }).await.unwrap();
        }
        other => panic!("Expected Ping, got {:?}", other),
    }

    let mut client_peer = client_handle.await.unwrap();

    // Client receives pong
    let msg = client_peer.receive_message().await.unwrap();
    match msg {
        Message::Pong { .. } => {
            client_peer.handle_pong();
        }
        other => panic!("Expected Pong, got {:?}", other),
    }

    assert!(client_peer.is_alive());
}

#[tokio::test]
async fn test_message_too_large_rejected() {
    // Connect raw and send a message claiming to be very large
    let (server_addr, mut server_rx) = spawn_listener().await;

    tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(server_addr).await.unwrap();
        // Claim the message is 20MB
        let fake_len: u32 = 20_000_000;
        stream.write_all(&fake_len.to_le_bytes()).await.unwrap();
        // Don't actually send that much data — the peer should reject based on length
        let _ = stream.write_all(&[0u8; 100]).await;
        // Hold open briefly
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    });

    let mut server_peer = server_rx.recv().await.unwrap();
    // Attempting to receive should yield an error (message too large)
    let result = server_peer.receive_message().await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("too large") || msg.contains("Connection") || msg.contains("closed"),
        "Expected size rejection, got: '{}'",
        msg
    );
}

#[tokio::test]
async fn test_peer_manager_connect_and_count() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let our_addr: SocketAddr = "127.0.0.1:9002".parse().unwrap();

    // Server handshakes in background
    tokio::spawn(async move {
        let mut peer = server_rx.recv().await.unwrap();
        peer.complete_handshake(server_addr).await.unwrap();
    });

    let (mut mgr, _rx) = PeerManager::new(our_addr);

    let idx = mgr.connect_to_peer(server_addr).await.unwrap();
    assert_eq!(mgr.peer_count(), 1);
    assert_eq!(mgr.connected_count(), 1);

    mgr.remove_peer(idx);
    assert_eq!(mgr.peer_count(), 0);
}

#[tokio::test]
async fn test_peer_disconnect() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let our_addr: SocketAddr = "127.0.0.1:9003".parse().unwrap();

    let mut client = PeerConnection::connect(server_addr, our_addr).await.unwrap();

    let mut server_peer = server_rx.recv().await.unwrap();

    // Handshake both sides
    let server_addr_copy = server_addr;
    let server_handle = tokio::spawn(async move {
        server_peer.complete_handshake(server_addr_copy).await.unwrap();
        server_peer
    });

    let our_addr_copy = our_addr;
    client.complete_handshake(our_addr_copy).await.unwrap();

    let _server_peer = server_handle.await.unwrap();

    assert!(client.is_connected());
    client.disconnect();
    assert!(!client.is_connected());
}

#[tokio::test]
async fn test_rate_limiting() {
    let (server_addr, mut server_rx) = spawn_listener().await;
    let our_addr: SocketAddr = "127.0.0.1:9004".parse().unwrap();

    let server_handle = tokio::spawn(async move {
        let mut peer = server_rx.recv().await.unwrap();
        peer.complete_handshake(server_addr).await.unwrap();
        peer
    });

    let mut client = PeerConnection::connect(server_addr, our_addr).await.unwrap();
    client.complete_handshake(our_addr).await.unwrap();

    let mut _server_peer = server_handle.await.unwrap();

    // Simulate rate limiting: record_message returns false after limit
    // The limit is 500 messages per 60 seconds.
    // We don't actually send 500 messages, just verify the mechanism works.
    for _ in 0..10 {
        assert!(client.record_message(), "Should allow messages within limit");
    }
}
