// ========================================
// --- FILE: tests/network_integration.rs
// ========================================
use midstate::network::{PeerConnection, Message};
use tokio::net::TcpListener;
use tokio::io::{AsyncWriteExt, AsyncReadExt}; // Added AsyncReadExt
use std::net::SocketAddr;

// Helper to spawn a passive listener that acts like a Node
async fn spawn_listener() -> (SocketAddr, tokio::sync::mpsc::UnboundedReceiver<PeerConnection>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        loop {
            let (stream, addr) = listener.accept().await.unwrap();
            let peer = PeerConnection::from_stream(stream, addr);
            tx.send(peer).unwrap();
        }
    });

    (addr, rx)
}

#[tokio::test]
async fn test_full_handshake_success() {
    // 1. Start a "Server" peer
    let (server_addr, mut server_rx) = spawn_listener().await;

    // 2. Start a "Client" peer connecting to Server
    let client_addr: SocketAddr = "127.0.0.1:9000".parse().unwrap(); 
    let client_handle = tokio::spawn(async move {
        // FIX: Use TcpStream directly + from_stream.
        // PeerConnection::connect() sends a Version msg automatically, and 
        // complete_handshake sends ANOTHER one, causing a protocol error.
        // Using from_stream gives us a clean slate.
        let stream = tokio::net::TcpStream::connect(server_addr).await.unwrap();
        let mut peer = PeerConnection::from_stream(stream, client_addr);
        
        peer.complete_handshake(client_addr).await.unwrap();
        peer
    });

    // 3. Server accepts connection
    let mut server_peer = server_rx.recv().await.unwrap();
    
    // Server logic: Perform handshake
    server_peer.complete_handshake(server_addr).await.unwrap();

    // 4. Wait for client to finish
    let client_peer = client_handle.await.unwrap();

    // 5. Verify both are connected
    assert!(server_peer.is_connected());
    assert!(client_peer.is_connected());
}

#[tokio::test]
async fn test_handshake_version_mismatch() {
    let (server_addr, mut server_rx) = spawn_listener().await;

    // Client connects but sends WRONG protocol version manually
    tokio::spawn(async move {
        let mut stream = tokio::net::TcpStream::connect(server_addr).await.unwrap();
        let bad_version = Message::Version {
            version: 9999, // INVALID VERSION
            services: 1,
            timestamp: 0,
            addr_recv: server_addr,
            addr_from: "127.0.0.1:8000".parse().unwrap(),
        };
        // Manually send bytes
        let bytes = bad_version.serialize();
        let len = bytes.len() as u32;
        
        stream.write_all(&len.to_le_bytes()).await.unwrap();
        stream.write_all(&bytes).await.unwrap();
        
        // FIX: Keep the connection open! 
        // If we exit immediately, the server gets a "Broken Pipe" when it tries 
        // to send ITS version, and never gets to read our bad version.
        let mut buf = [0u8; 10];
        let _ = stream.read(&mut buf).await;
    });

    let mut server_peer = server_rx.recv().await.unwrap();
    
    // Server tries to complete handshake, should fail due to version mismatch
    let result = server_peer.complete_handshake(server_addr).await;
    
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Protocol version"), 
        "Expected 'Protocol version' error, got: '{}'", err_msg
    );
}
