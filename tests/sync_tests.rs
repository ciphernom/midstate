// ========================================
// --- FILE: tests/sync_tests.rs
// ========================================
use midstate::{storage::Storage, sync::Syncer, core::{State, Transaction, Batch, types::{hash, compute_commitment, block_reward, Extension, EXTENSION_ITERATIONS, CHECKPOINT_INTERVAL}}};
use midstate::network::{PeerConnection, Message};
use tokio::net::TcpListener;
use tempfile::TempDir;
use std::sync::Arc;

/// Mine a batch and return the batch plus the seeds for any coinbase coins created.
async fn mine_batch_internal(state: &State, transactions: Vec<Transaction>) -> (Batch, Vec<[u8; 32]>) {
    let mut candidate = state.clone();
    let mut total_fees = 0;
    for tx in &transactions {
        total_fees += tx.fee() as u64;
        midstate::core::transaction::apply_transaction(&mut candidate, tx).unwrap();
    }

    // Generate coinbase with known seeds so we can spend them later
    let reward = block_reward(state.height);
    let count = reward + total_fees;
    let mut coinbase_seeds = Vec::new();
    let mut coinbase_coins = Vec::new();

    for _ in 0..count {
        let seed: [u8; 32] = rand::random();
        let coin = midstate::core::wots::keygen(&seed);
        coinbase_seeds.push(seed);
        coinbase_coins.push(coin);
    }

    // Update midstate with coinbase
    let mut mining_midstate = candidate.midstate;
    for coin in &coinbase_coins {
        mining_midstate = midstate::core::hash_concat(&mining_midstate, coin);
    }

    let extension = midstate::core::extension::mine_extension(
        mining_midstate,
        state.target,
    );
    let batch = Batch {
        transactions,
        extension,
        coinbase: coinbase_coins,
    };

    (batch, coinbase_seeds)
}

/// Generate a valid chain using commit-reveal flow.
/// Maintains a pool of spendable seeds to fund transactions.
async fn generate_chain(num_spends: u64) -> (State, Vec<Batch>) {
    let mut state = State::genesis();
    let mut batches = Vec::new();

    // Initialize wallet with Genesis seeds
    let mut available_seeds: Vec<[u8; 32]> = vec![
        hash(b"genesis_coin_1"),
        hash(b"genesis_coin_2"),
        hash(b"genesis_coin_3"),
    ];

    for i in 0..num_spends {
        if available_seeds.len() < 2 {
            panic!("Not enough spendable coins to generate chain at step {}", i);
        }

        let seed1 = available_seeds.pop().unwrap();
        let seed2 = available_seeds.pop().unwrap();
        let coin1 = midstate::core::wots::keygen(&seed1);
        let coin2 = midstate::core::wots::keygen(&seed2);

        let next_seed = hash(&format!("step_{}_output", i).into_bytes());
        let next_coin = midstate::core::wots::keygen(&next_seed);
        let salt: [u8; 32] = rand::random();

        let input_coins = vec![coin1, coin2];
        let new_coins = vec![next_coin];

        available_seeds.push(next_seed);

        let commitment = compute_commitment(&input_coins, &new_coins, &salt);

        // Batch 1: Commit
        let commit_tx = Transaction::Commit { commitment };
        let (commit_batch, mut cb_seeds_1) = mine_batch_internal(&state, vec![commit_tx]).await;
        midstate::core::state::apply_batch(&mut state, &commit_batch).unwrap();
        batches.push(commit_batch);

        available_seeds.append(&mut cb_seeds_1);

        // Batch 2: Reveal
        let sig1 = midstate::core::wots::sign(&seed1, &commitment);
        let sig2 = midstate::core::wots::sign(&seed2, &commitment);

        let reveal_tx = Transaction::Reveal {
            input_coins,
            signatures: vec![sig1, sig2],
            new_coins,
            salt,
        };
        let (reveal_batch, mut cb_seeds_2) = mine_batch_internal(&state, vec![reveal_tx]).await;
        midstate::core::state::apply_batch(&mut state, &reveal_batch).unwrap();
        batches.push(reveal_batch);

        available_seeds.append(&mut cb_seeds_2);
    }
    (state, batches)
}

/// Start a mock peer that serves batches from a pre-built chain.
async fn start_mock_peer(end_state: State, batches: Arc<Vec<Batch>>) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (stream, addr) = listener.accept().await.unwrap();
        let mut peer = PeerConnection::from_stream(stream, addr);

        loop {
            match peer.receive_message().await {
                Ok(Message::GetState) => {
                    peer.send_message(&Message::StateInfo {
                        height: end_state.height,
                        depth: end_state.depth,
                        midstate: end_state.midstate,
                    }).await.unwrap();
                }
                Ok(Message::GetBatches { start_height, count }) => {
                    let start = start_height as usize;
                    let end = (start + count as usize).min(batches.len());
                    let slice = batches[start..end].to_vec();
                    peer.send_message(&Message::Batches(slice)).await.unwrap();
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    server_addr
}

#[tokio::test]
async fn test_sync_from_genesis() {
    // 1. Generate valid chain (5 spends = 10 batches)
    let (end_state, batches) = generate_chain(5).await;
    let batches_arc = Arc::new(batches);

    // 2. Setup mock peer
    let server_addr = start_mock_peer(end_state.clone(), batches_arc).await;

    // 3. Sync
    let temp_dir = TempDir::new().unwrap();
    let storage = Storage::open(temp_dir.path()).unwrap();
    let syncer = Syncer::new(storage);
    let our_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();

    let mut peer = PeerConnection::connect(server_addr, our_addr).await.unwrap();

    let synced_state = syncer.sync_from_genesis(&mut peer).await.unwrap();

    assert_eq!(synced_state.height, 10); // 5 spends × 2 batches each
    assert_eq!(synced_state.midstate, end_state.midstate);
}

#[tokio::test]
async fn test_sync_empty_chain() {
    // Peer at height 0 → nothing to sync
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (stream, addr) = listener.accept().await.unwrap();
        let mut peer = PeerConnection::from_stream(stream, addr);

        loop {
            match peer.receive_message().await {
                Ok(Message::GetState) => {
                    let genesis = State::genesis();
                    peer.send_message(&Message::StateInfo {
                        height: 0,
                        depth: 0,
                        midstate: genesis.midstate,
                    }).await.unwrap();
                }
                Ok(Message::GetBatches { .. }) => {
                    peer.send_message(&Message::Batches(vec![])).await.unwrap();
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let temp_dir = TempDir::new().unwrap();
    let storage = Storage::open(temp_dir.path()).unwrap();
    let syncer = Syncer::new(storage);
    let our_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();

    let mut peer = PeerConnection::connect(server_addr, our_addr).await.unwrap();
    let state = syncer.sync_from_genesis(&mut peer).await.unwrap();

    assert_eq!(state.height, 0);
}

#[tokio::test]
async fn test_sync_rejects_invalid_batch() {
    // Peer serves one valid batch followed by a batch with a bogus extension.
    // The syncer should fail when verifying the invalid batch.
    let mut state = State::genesis();

    // Mine one valid batch
    let commit_tx = Transaction::Commit { commitment: hash(b"dummy_commitment") };
    let (valid_batch, _) = mine_batch_internal(&state, vec![commit_tx]).await;
    midstate::core::state::apply_batch(&mut state, &valid_batch).unwrap();

    // Create an invalid batch with garbage extension
    let commit_tx2 = Transaction::Commit { commitment: hash(b"dummy_commitment_2") };
    let num_segments = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize;
    let invalid_batch = Batch {
        transactions: vec![commit_tx2],
        extension: Extension {
            nonce: 0,
            final_hash: [0u8; 32],
            checkpoints: vec![[0u8; 32]; num_segments + 1],
        },
        coinbase: vec![],
    };

    let batches = Arc::new(vec![valid_batch, invalid_batch]);
    let fake_end_state = State {
        height: 2,
        depth: state.depth + EXTENSION_ITERATIONS,
        ..state.clone()
    };

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();

    let batches_clone = batches.clone();
    tokio::spawn(async move {
        let (stream, addr) = listener.accept().await.unwrap();
        let mut peer = PeerConnection::from_stream(stream, addr);

        loop {
            match peer.receive_message().await {
                Ok(Message::GetState) => {
                    peer.send_message(&Message::StateInfo {
                        height: fake_end_state.height,
                        depth: fake_end_state.depth,
                        midstate: fake_end_state.midstate,
                    }).await.unwrap();
                }
                Ok(Message::GetBatches { start_height, count }) => {
                    let start = start_height as usize;
                    let end = (start + count as usize).min(batches_clone.len());
                    let slice = batches_clone[start..end].to_vec();
                    peer.send_message(&Message::Batches(slice)).await.unwrap();
                }
                Err(_) => break,
                _ => {}
            }
        }
    });

    let temp_dir = TempDir::new().unwrap();
    let storage = Storage::open(temp_dir.path()).unwrap();
    let syncer = Syncer::new(storage);
    let our_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();

    let mut peer = PeerConnection::connect(server_addr, our_addr).await.unwrap();
    let result = syncer.sync_from_genesis(&mut peer).await;

    // Should fail on the second (invalid) batch
    assert!(result.is_err(), "Sync should reject invalid batch");
}
