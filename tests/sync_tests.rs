// ========================================
// --- FILE: tests/sync_tests.rs
// ========================================
use midstate::{storage::Storage, sync::Syncer, core::{State, Transaction, Batch, types::{hash, compute_commitment}}};
use midstate::network::{PeerConnection, Message};
use tokio::net::TcpListener;
use tempfile::TempDir;
use std::sync::Arc;

/// Generate a valid chain using commit-reveal flow.
/// Each "logical spend" takes 2 batches: one for commit, one for reveal.
async fn generate_chain(num_spends: u64) -> (State, Vec<Batch>) {
    let mut state = State::genesis();
    let mut batches = Vec::new();
    let mut current_secret = b"genesis_coin_1".to_vec();

    for i in 0..num_spends {
        let next_secret = format!("coin_{}", i).into_bytes();
        let next_coin = hash(&next_secret);
        let salt: [u8; 32] = rand::random();

        let input_coins = vec![hash(&current_secret)];
        let new_coins = vec![next_coin];
        let commitment = compute_commitment(&input_coins, &new_coins, &salt);

        // Batch 1: Commit
        let commit_tx = Transaction::Commit { commitment };
        let mut candidate = state.clone();
        midstate::core::transaction::apply_transaction(&mut candidate, &commit_tx).unwrap();

        let extension = midstate::core::extension::mine_extension(
            candidate.midstate,
            state.target,
        );
        let commit_batch = Batch {
            transactions: vec![commit_tx],
            extension,
        };
        midstate::core::state::apply_batch(&mut state, &commit_batch).unwrap();
        batches.push(commit_batch);

        // Batch 2: Reveal
        let reveal_tx = Transaction::Reveal {
            secrets: vec![current_secret.clone()],
            new_coins: new_coins.clone(),
            salt,
        };
        let mut candidate = state.clone();
        midstate::core::transaction::apply_transaction(&mut candidate, &reveal_tx).unwrap();

        let extension = midstate::core::extension::mine_extension(
            candidate.midstate,
            state.target,
        );
        let reveal_batch = Batch {
            transactions: vec![reveal_tx],
            extension,
        };
        midstate::core::state::apply_batch(&mut state, &reveal_batch).unwrap();
        batches.push(reveal_batch);

        current_secret = next_secret;
    }
    (state, batches)
}

#[tokio::test]
async fn test_sync_from_genesis() {
    // 1. Generate valid chain (5 spends = 10 batches)
    let (end_state, batches) = generate_chain(5).await;
    let batches_arc = Arc::new(batches);

    // 2. Setup mock peer
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();

    let batches_clone = batches_arc.clone();
    let end_height = end_state.height;

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut peer = PeerConnection::from_stream(stream, server_addr);

        loop {
            match peer.receive_message().await {
                Ok(Message::GetState) => {
                    peer.send_message(&Message::StateInfo {
                        height: end_height,
                        depth: end_state.depth,
                        midstate: end_state.midstate,
                    }).await.unwrap();
                },
                Ok(Message::GetBatches { start_height, count }) => {
                    let start = start_height as usize;
                    let end = (start + count as usize).min(batches_clone.len());
                    let slice = batches_clone[start..end].to_vec();
                    peer.send_message(&Message::Batches(slice)).await.unwrap();
                },
                _ => {}
            }
        }
    });

    // 3. Sync
    let temp_dir = TempDir::new().unwrap();
    let storage = Storage::open(temp_dir.path()).unwrap();
    let syncer = Syncer::new(storage);

    let mut peer = PeerConnection::connect(server_addr, "127.0.0.1:9000".parse().unwrap()).await.unwrap();

    let synced_state = syncer.sync_from_genesis(&mut peer).await.unwrap();

    assert_eq!(synced_state.height, 10); // 5 spends × 2 batches each
    assert_eq!(synced_state.midstate, end_state.midstate);
}
