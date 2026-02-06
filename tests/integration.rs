use midstate::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_genesis_state() {
    let state = core::State::genesis();
    assert_eq!(state.height, 0);
    assert_eq!(state.depth, 0);
    assert_eq!(state.coins.len(), 3);
}

#[tokio::test]
async fn test_transaction_validation() {
    let state = core::State::genesis();
    
    let tx = Transaction {
        secrets: vec![b"genesis_coin_1".to_vec()],
        new_coins: vec![core::hash(b"new_coin")],
    };
    
    assert!(core::transaction::validate_transaction(&state, &tx).is_ok());
    
    let bad_tx = Transaction {
        secrets: vec![b"nonexistent".to_vec()],
        new_coins: vec![core::hash(b"new_coin")],
    };
    
    assert!(core::transaction::validate_transaction(&state, &bad_tx).is_err());
}

#[tokio::test]
async fn test_extension_creation() {
    let midstate = [0u8; 32];
    let nonce = 12345;
    
    let ext = core::extension::create_extension(midstate, nonce);
    
    let ext2 = core::extension::create_extension(midstate, nonce);
    assert_eq!(ext.final_hash, ext2.final_hash);
    
    let ext3 = core::extension::create_extension(midstate, nonce + 1);
    assert_ne!(ext.final_hash, ext3.final_hash);
}

#[tokio::test]
async fn test_batch_application() {
    let mut state = core::State::genesis();
    
    let tx = Transaction {
        secrets: vec![b"genesis_coin_1".to_vec()],
        new_coins: vec![core::hash(b"new_coin")],
    };
    
    let mut candidate_state = state.clone();
    core::transaction::apply_transaction(&mut candidate_state, &tx).unwrap();
    
    let extension = core::extension::create_extension(candidate_state.midstate, 0);
    
    let batch = Batch {
        transactions: vec![tx],
        extension,
    };
    
    let old_height = state.height;
    core::state::apply_batch(&mut state, &batch).unwrap();
    
    assert_eq!(state.height, old_height + 1);
    assert!(state.coins.contains(&core::hash(b"new_coin")));
    assert!(!state.coins.contains(&core::hash(b"genesis_coin_1")));
}

#[tokio::test]
async fn test_mempool() {
    let temp_dir = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp_dir.path()).unwrap();
    
    let state = core::State::genesis();
    
    let tx = Transaction {
        secrets: vec![b"genesis_coin_1".to_vec()],
        new_coins: vec![core::hash(b"new_coin")],
    };
    
    assert!(mempool.add(tx.clone(), &state).is_ok());
    assert_eq!(mempool.len(), 1);
    
    assert!(mempool.add(tx, &state).is_err());
}

#[tokio::test]
async fn test_coin_merge() {
    let mut state = core::State::genesis();
    
    // Transaction with multiple inputs (merge)
    let tx = Transaction {
        secrets: vec![
            b"genesis_coin_1".to_vec(),
            b"genesis_coin_2".to_vec(),
        ],
        new_coins: vec![core::hash(b"merged_coin")],
    };
    
    let mut candidate_state = state.clone();
    core::transaction::apply_transaction(&mut candidate_state, &tx).unwrap();
    
    let extension = core::extension::create_extension(candidate_state.midstate, 0);
    
    let batch = Batch {
        transactions: vec![tx],
        extension,
    };
    
    core::state::apply_batch(&mut state, &batch).unwrap();
    
    // Should have 2 coins now (genesis_coin_3 + merged_coin)
    assert_eq!(state.coins.len(), 2);
    assert!(state.coins.contains(&core::hash(b"merged_coin")));
    assert!(state.coins.contains(&core::hash(b"genesis_coin_3")));
}

#[tokio::test]
async fn test_storage() {
    let temp_dir = TempDir::new().unwrap();
    let storage = storage::Storage::open(temp_dir.path()).unwrap();
    
    let state = core::State::genesis();
    storage.save_state(&state).unwrap();
    
    let loaded = storage.load_state().unwrap().unwrap();
    assert_eq!(loaded.height, state.height);
    assert_eq!(loaded.midstate, state.midstate);
}

#[tokio::test]
async fn test_fork_choice() {
    let mut state_a = core::State::genesis();
    let mut state_b = core::State::genesis();
    
    state_a.depth = 1000;
    state_b.depth = 2000;
    
    let best = core::state::choose_best_state(&state_a, &state_b);
    assert_eq!(best.depth, 2000);
    
    state_b.depth = 1000;
    state_a.midstate = [0u8; 32];
    state_b.midstate = [1u8; 32];
    
    let best = core::state::choose_best_state(&state_a, &state_b);
    assert_eq!(best.midstate, [0u8; 32]);
}
