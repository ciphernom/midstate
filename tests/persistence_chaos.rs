use midstate::node::Node;
use midstate::storage::Storage;
use midstate::core::State;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_corrupt_database_recovery() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("db").join("state");

    // 1. Create a valid DB structure
    fs::create_dir_all(&db_path).unwrap();

    // 2. Write garbage into the Sled data files
    fs::write(db_path.join("conf"), b"GARBAGE_DATA_CORRUPTION").unwrap();
    fs::write(db_path.join("db"), b"GARBAGE_DATA_CORRUPTION").unwrap();

    // 3. Try to open Node
    let our_addr = "127.0.0.1:0".parse().unwrap();
    let result = Node::new(temp.path().into(), false, our_addr);

    match result {
        Ok(_) => {
            // Sled auto-recovered, acceptable
        }
        Err(e) => {
            // Failed gracefully — no panic
            println!("Node correctly failed to open corrupt DB: {}", e);
        }
    }
}

#[test]
fn test_corrupt_mempool_recovery() {
    let temp = TempDir::new().unwrap();
    let mempool_path = temp.path().join("mempool");

    // Create corrupt mempool sled DB
    fs::create_dir_all(&mempool_path).unwrap();
    fs::write(mempool_path.join("conf"), b"GARBAGE").unwrap();
    fs::write(mempool_path.join("db"), b"GARBAGE").unwrap();

    let result = midstate::mempool::Mempool::new(&mempool_path);

    match result {
        Ok(mp) => {
            // If Sled recovers, mempool should be empty
            assert_eq!(mp.len(), 0);
        }
        Err(e) => {
            println!("Mempool correctly failed to open corrupt DB: {}", e);
        }
    }
}

#[test]
fn test_storage_state_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = State::genesis();
    storage.save_state(&state).unwrap();

    let loaded = storage.load_state().unwrap().unwrap();
    assert_eq!(loaded.height, state.height);
    assert_eq!(loaded.midstate, state.midstate);
    assert_eq!(loaded.depth, state.depth);
    assert_eq!(loaded.coins.len(), state.coins.len());
    assert_eq!(loaded.target, state.target);
}

#[test]
fn test_storage_no_saved_state_returns_none() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    assert!(storage.load_state().unwrap().is_none());
}

#[test]
fn test_storage_mining_seed_persistence() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    assert!(storage.load_mining_seed().unwrap().is_none());

    let seed: [u8; 32] = rand::random();
    storage.save_mining_seed(&seed).unwrap();

    let loaded = storage.load_mining_seed().unwrap().unwrap();
    assert_eq!(loaded, seed);
}

#[test]
fn test_node_creates_data_dir() {
    let temp = TempDir::new().unwrap();
    let data_dir = temp.path().join("new_data_dir");

    assert!(!data_dir.exists());

    let our_addr = "127.0.0.1:0".parse().unwrap();
    let _node = Node::new(data_dir.clone(), false, our_addr).unwrap();

    assert!(data_dir.exists());
}

#[test]
fn test_node_loads_genesis_on_fresh_start() {
    let temp = TempDir::new().unwrap();
    let our_addr = "127.0.0.1:0".parse().unwrap();

    let node = Node::new(temp.path().into(), false, our_addr).unwrap();
    let (handle, _rx) = node.create_handle();

    // Use tokio runtime to check state
    let rt = tokio::runtime::Runtime::new().unwrap();
    let state = rt.block_on(handle.get_state());

    assert_eq!(state.height, 0);
    assert_eq!(state.coins.len(), 3);
}

#[test]
fn test_batch_store_highest() {
    let temp = TempDir::new().unwrap();
    let store = midstate::storage::BatchStore::new(temp.path()).unwrap();

    // Empty store
    assert_eq!(store.highest().unwrap(), 0);

    // Save some batches (use minimal structure for serialization)
    let dummy_batch = midstate::core::Batch {
        transactions: vec![],
        extension: midstate::core::Extension {
            nonce: 0,
            final_hash: [0u8; 32],
            checkpoints: vec![],
        },
        coinbase: vec![],
    };

    store.save(5, &dummy_batch).unwrap();
    store.save(10, &dummy_batch).unwrap();
    store.save(3, &dummy_batch).unwrap();

    assert_eq!(store.highest().unwrap(), 10);
}
