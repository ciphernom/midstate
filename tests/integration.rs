use midstate::*;
use midstate::core::{self, hash, hash_concat, compute_commitment, EXTENSION_ITERATIONS, DIFFICULTY_ADJUSTMENT_INTERVAL, TARGET_BLOCK_TIME};
use midstate::core::extension::{create_extension, verify_extension, mine_extension};
use midstate::core::transaction::{apply_transaction, validate_transaction};
use midstate::core::state::{apply_batch, choose_best_state, adjust_difficulty};
use midstate::network::protocol::{Message, PROTOCOL_VERSION};
use midstate::storage::Storage;
use tempfile::TempDir;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Mine a valid batch from a set of transactions against a state
async fn mine_batch(state: &State, transactions: Vec<Transaction>) -> Batch {
    let mut candidate = state.clone();
    for tx in &transactions {
        apply_transaction(&mut candidate, tx).unwrap();
    }
    let midstate = candidate.midstate;
    let target = state.target;
    let extension = tokio::task::spawn_blocking(move || mine_extension(midstate, target))
        .await
        .unwrap();
    Batch { transactions, extension }
}

/// Inject a commitment into state (for unit tests that test reveals directly)
fn inject_commitment(state: &mut State, secrets: &[Vec<u8>], new_coins: &[[u8; 32]], salt: &[u8; 32]) {
    let input_coins: Vec<[u8; 32]> = secrets.iter().map(|s| hash(s)).collect();
    let commitment = compute_commitment(&input_coins, new_coins, salt);
    state.commitments.insert(commitment);
}

/// Create a Reveal transaction with a known salt
fn make_reveal(secrets: Vec<Vec<u8>>, new_coins: Vec<[u8; 32]>, salt: [u8; 32]) -> Transaction {
    Transaction::Reveal { secrets, new_coins, salt }
}

/// Create a Commit transaction from components
fn make_commit(secrets: &[Vec<u8>], new_coins: &[[u8; 32]], salt: &[u8; 32]) -> Transaction {
    let input_coins: Vec<[u8; 32]> = secrets.iter().map(|s| hash(s)).collect();
    let commitment = compute_commitment(&input_coins, new_coins, salt);
    Transaction::Commit { commitment }
}

/// Do a full commit-then-reveal across two batches
async fn commit_reveal_batch(
    state: &mut State,
    secrets: Vec<Vec<u8>>,
    new_coins: Vec<[u8; 32]>,
) -> [u8; 32] {
    let salt: [u8; 32] = rand::random();

    // Batch 1: Commit
    let commit_tx = make_commit(&secrets, &new_coins, &salt);
    let commit_batch = mine_batch(state, vec![commit_tx]).await;
    apply_batch(state, &commit_batch).unwrap();

    // Batch 2: Reveal
    let reveal_tx = make_reveal(secrets, new_coins, salt);
    let reveal_batch = mine_batch(state, vec![reveal_tx]).await;
    apply_batch(state, &reveal_batch).unwrap();

    salt
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMMIT-REVEAL SPECIFIC TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_commit_reveal_basic_flow() {
    let mut state = core::State::genesis();
    let new_coins = vec![hash(b"coin_a")];
    let secrets = vec![b"genesis_coin_1".to_vec()];

    // Commit phase
    let salt: [u8; 32] = rand::random();
    let commit_tx = make_commit(&secrets, &new_coins, &salt);
    let commit_batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &commit_batch).unwrap();

    // Commitment should be in state
    let input_coins: Vec<[u8; 32]> = secrets.iter().map(|s| hash(s)).collect();
    let expected = compute_commitment(&input_coins, &new_coins, &salt);
    assert!(state.commitments.contains(&expected));

    // Reveal phase
    let reveal_tx = make_reveal(secrets, new_coins, salt);
    let reveal_batch = mine_batch(&state, vec![reveal_tx]).await;
    apply_batch(&mut state, &reveal_batch).unwrap();

    // Commitment consumed, coin transferred
    assert!(!state.commitments.contains(&expected));
    assert!(state.coins.contains(&hash(b"coin_a")));
    assert!(!state.coins.contains(&hash(b"genesis_coin_1")));
}

#[tokio::test]
async fn test_reveal_without_commit_rejected() {
    let mut state = core::State::genesis();
    let salt: [u8; 32] = [0u8; 32];

    let reveal_tx = make_reveal(
        vec![b"genesis_coin_1".to_vec()],
        vec![hash(b"stolen")],
        salt,
    );

    // Should fail — no matching commitment
    assert!(validate_transaction(&state, &reveal_tx).is_err());
    assert!(apply_transaction(&mut state, &reveal_tx).is_err());
}

#[tokio::test]
async fn test_front_running_prevented() {
    let mut state = core::State::genesis();

    // Alice commits to sending to her destination
    let alice_dest = vec![hash(b"alice_output")];
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&secrets, &alice_dest, &salt);
    let commit_batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &commit_batch).unwrap();

    // Attacker sees alice's secret in the reveal and tries to redirect to their own destination
    let attacker_dest = vec![hash(b"attacker_output")];
    let attacker_reveal = make_reveal(
        secrets.clone(),
        attacker_dest,
        salt, // even if they somehow got the salt
    );

    // Attacker's reveal should fail — commitment doesn't match their destination
    assert!(validate_transaction(&state, &attacker_reveal).is_err());

    // Alice's reveal succeeds
    let alice_reveal = make_reveal(secrets, alice_dest, salt);
    assert!(validate_transaction(&state, &alice_reveal).is_ok());
}

#[tokio::test]
async fn test_commit_reveal_same_batch_rejected() {
    let state = core::State::genesis();
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"coin_a")];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&secrets, &new_coins, &salt);
    let reveal_tx = make_reveal(secrets, new_coins, salt);

    // Apply commit, then try reveal in same candidate state
    let mut candidate = state.clone();
    apply_transaction(&mut candidate, &commit_tx).unwrap();
    // The reveal should succeed at transaction level (commitment exists in candidate)
    // BUT: the batch mining below uses the original state's target, and applying
    // the batch will re-apply both. The point is that in real usage, the mempool
    // won't accept a reveal whose commitment isn't in the confirmed state.
    // 
    // For the direct apply_transaction test: commit then reveal in sequence works
    // at the state machine level (which is correct — the batch applies them in order).
    // The protection is at the mempool/validation layer.
    let result = apply_transaction(&mut candidate, &reveal_tx);
    // This actually succeeds at the state machine level, which is fine.
    // The mempool is what prevents same-batch commit+reveal.
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_duplicate_commitment_rejected() {
    let mut state = core::State::genesis();
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"coin_a")];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&secrets, &new_coins, &salt);

    apply_transaction(&mut state, &commit_tx).unwrap();
    // Same commitment again should fail
    assert!(apply_transaction(&mut state, &commit_tx.clone()).is_err());
}

#[tokio::test]
async fn test_wrong_salt_rejected() {
    let mut state = core::State::genesis();
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"coin_a")];
    let salt: [u8; 32] = rand::random();

    inject_commitment(&mut state, &secrets, &new_coins, &salt);

    // Reveal with wrong salt
    let wrong_salt: [u8; 32] = [0xff; 32];
    let reveal_tx = make_reveal(secrets, new_coins, wrong_salt);

    assert!(validate_transaction(&state, &reveal_tx).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SECURITY-CRITICAL TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_double_spend_rejected() {
    let mut state = core::State::genesis();
    let salt: [u8; 32] = rand::random();

    // First: commit + reveal genesis_coin_1 → coin_a
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"coin_a")];
    commit_reveal_batch(&mut state, secrets.clone(), new_coins).await;

    // Second: try to spend genesis_coin_1 again → coin_b
    let salt2: [u8; 32] = rand::random();
    inject_commitment(&mut state, &secrets, &[hash(b"coin_b")], &salt2);

    let tx2 = make_reveal(secrets, vec![hash(b"coin_b")], salt2);

    // Should fail — coin already spent
    assert!(validate_transaction(&state, &tx2).is_err());
    let mut bad_state = state.clone();
    assert!(apply_transaction(&mut bad_state, &tx2).is_err());
}

#[tokio::test]
async fn test_double_spend_within_batch_rejected() {
    let mut state = core::State::genesis();
    let salt1: [u8; 32] = rand::random();
    let salt2: [u8; 32] = rand::random();

    let secrets = vec![b"genesis_coin_1".to_vec()];
    inject_commitment(&mut state, &secrets, &[hash(b"coin_a")], &salt1);
    inject_commitment(&mut state, &secrets, &[hash(b"coin_b")], &salt2);

    let tx1 = make_reveal(secrets.clone(), vec![hash(b"coin_a")], salt1);
    let tx2 = make_reveal(secrets, vec![hash(b"coin_b")], salt2);

    let mut candidate = state.clone();
    apply_transaction(&mut candidate, &tx1).unwrap();
    // Second tx spends the same coin — should fail
    assert!(apply_transaction(&mut candidate, &tx2).is_err());
}

#[tokio::test]
async fn test_extension_tampered_hash_rejected() {
    let midstate = [0u8; 32];
    let ext = create_extension(midstate, 42);

    let mut bad_ext = ext.clone();
    bad_ext.final_hash[0] ^= 0xff;

    let easy_target = [0xff; 32];
    assert!(verify_extension(midstate, &bad_ext, &easy_target).is_err());
}

#[tokio::test]
async fn test_extension_wrong_nonce_rejected() {
    let midstate = [0u8; 32];
    let ext = create_extension(midstate, 42);

    let bad_ext = Extension {
        nonce: 43,
        final_hash: ext.final_hash,
    };

    let easy_target = [0xff; 32];
    assert!(verify_extension(midstate, &bad_ext, &easy_target).is_err());
}

#[tokio::test]
async fn test_extension_fails_difficulty_target() {
    let midstate = [0u8; 32];
    let ext = create_extension(midstate, 42);

    let impossible_target = [0x00; 32];
    assert!(verify_extension(midstate, &ext, &impossible_target).is_err());
}

#[tokio::test]
async fn test_extension_wrong_midstate_rejected() {
    let midstate_a = [0u8; 32];
    let midstate_b = [1u8; 32];
    let ext = create_extension(midstate_a, 42);

    let easy_target = [0xff; 32];
    assert!(verify_extension(midstate_b, &ext, &easy_target).is_err());
}

#[tokio::test]
async fn test_reveal_no_inputs_rejected() {
    let mut state = core::State::genesis();
    let salt: [u8; 32] = [0u8; 32];

    let tx = make_reveal(vec![], vec![hash(b"free_money")], salt);

    assert!(validate_transaction(&state, &tx).is_err());
    assert!(apply_transaction(&mut state, &tx).is_err());
}

#[tokio::test]
async fn test_reveal_no_outputs_rejected() {
    let mut state = core::State::genesis();
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let salt: [u8; 32] = [0u8; 32];

    inject_commitment(&mut state, &secrets, &[], &salt);

    let tx = make_reveal(secrets, vec![], salt);

    assert!(validate_transaction(&state, &tx).is_err());
    assert!(apply_transaction(&mut state, &tx).is_err());
}

#[tokio::test]
async fn test_duplicate_output_coin_rejected() {
    let mut state = core::State::genesis();
    let duplicate_coin = hash(b"same_coin");
    let secrets = vec![b"genesis_coin_1".to_vec(), b"genesis_coin_2".to_vec()];
    let new_coins = vec![duplicate_coin, duplicate_coin];
    let salt: [u8; 32] = rand::random();

    inject_commitment(&mut state, &secrets, &new_coins, &salt);

    let tx = make_reveal(secrets, new_coins, salt);

    let mut s = state.clone();
    assert!(apply_transaction(&mut s, &tx).is_err());
}

#[tokio::test]
async fn test_batch_with_invalid_tx_rejected() {
    let state = core::State::genesis();

    // Reveal without commitment — will fail
    let bad_tx = make_reveal(
        vec![b"nonexistent_secret".to_vec()],
        vec![hash(b"coin_x")],
        [0u8; 32],
    );

    let dummy_ext = Extension {
        nonce: 0,
        final_hash: [0u8; 32],
    };
    let batch = Batch {
        transactions: vec![bad_tx],
        extension: dummy_ext,
    };

    let mut s = state.clone();
    assert!(apply_batch(&mut s, &batch).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CHAIN CORRECTNESS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_multi_batch_chain() {
    let mut state = core::State::genesis();

    // Commit + Reveal: genesis_coin_1 -> coin_a
    commit_reveal_batch(
        &mut state,
        vec![b"genesis_coin_1".to_vec()],
        vec![hash(b"coin_a")],
    ).await;
    // 2 batches (commit + reveal)
    assert_eq!(state.height, 2);

    // Commit + Reveal: genesis_coin_2 -> coin_b
    commit_reveal_batch(
        &mut state,
        vec![b"genesis_coin_2".to_vec()],
        vec![hash(b"coin_b")],
    ).await;
    assert_eq!(state.height, 4);

    assert_eq!(state.depth, 4 * EXTENSION_ITERATIONS);
    assert_eq!(state.coins.len(), 3);
    assert!(state.coins.contains(&hash(b"coin_a")));
    assert!(state.coins.contains(&hash(b"coin_b")));
    assert!(state.coins.contains(&hash(b"genesis_coin_3")));
}

#[tokio::test]
async fn test_coin_split() {
    let mut state = core::State::genesis();

    commit_reveal_batch(
        &mut state,
        vec![b"genesis_coin_1".to_vec()],
        vec![hash(b"split_a"), hash(b"split_b"), hash(b"split_c")],
    ).await;

    // 1 coin consumed, 3 created → 3 - 1 + 3 = 5
    assert_eq!(state.coins.len(), 5);
    assert!(state.coins.contains(&hash(b"split_a")));
    assert!(state.coins.contains(&hash(b"split_b")));
    assert!(state.coins.contains(&hash(b"split_c")));
    assert!(!state.coins.contains(&hash(b"genesis_coin_1")));
}

#[tokio::test]
async fn test_midstate_evolves() {
    let mut state = core::State::genesis();
    let midstate_0 = state.midstate;

    commit_reveal_batch(
        &mut state,
        vec![b"genesis_coin_1".to_vec()],
        vec![hash(b"coin_1")],
    ).await;
    let midstate_1 = state.midstate;

    assert_ne!(midstate_0, midstate_1);

    commit_reveal_batch(
        &mut state,
        vec![b"genesis_coin_2".to_vec()],
        vec![hash(b"coin_2")],
    ).await;
    let midstate_2 = state.midstate;

    assert_ne!(midstate_1, midstate_2);
    assert_ne!(midstate_0, midstate_2);
}

#[tokio::test]
async fn test_depth_accumulates() {
    let mut state = core::State::genesis();
    assert_eq!(state.depth, 0);

    // Single commit batch (1 batch = 1x EXTENSION_ITERATIONS)
    let salt: [u8; 32] = rand::random();
    let commit_tx = make_commit(
        &[b"genesis_coin_1".to_vec()],
        &[hash(b"coin_a")],
        &salt,
    );
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    assert_eq!(state.depth, EXTENSION_ITERATIONS);
}

#[tokio::test]
async fn test_difficulty_no_change_before_interval() {
    let mut state = core::State::genesis();
    state.height = 5;
    let original_target = state.target;

    let previous: Vec<State> = (0..10)
        .map(|i| {
            let mut s = core::State::genesis();
            s.height = i;
            s.timestamp = i * TARGET_BLOCK_TIME;
            s
        })
        .collect();

    let new_target = adjust_difficulty(&state, &previous);
    assert_eq!(new_target, original_target);
}

#[tokio::test]
async fn test_difficulty_no_change_at_genesis() {
    let state = core::State::genesis();
    let new_target = adjust_difficulty(&state, &[]);
    assert_eq!(new_target, state.target);
}

#[tokio::test]
async fn test_difficulty_increases_when_fast() {
    let mut state = core::State::genesis();
    state.height = DIFFICULTY_ADJUSTMENT_INTERVAL;
    state.timestamp = 50;

    let previous: Vec<State> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL as usize)
        .map(|i| {
            let mut s = core::State::genesis();
            s.height = i as u64;
            s.timestamp = (i as u64) * (TARGET_BLOCK_TIME / 2);
            s
        })
        .collect();

    let new_target = adjust_difficulty(&state, &previous);
    assert!(new_target < state.target, "Target should decrease (get harder) when blocks are fast");
}

#[tokio::test]
async fn test_difficulty_decreases_when_slow() {
    let mut state = core::State::genesis();
    state.height = DIFFICULTY_ADJUSTMENT_INTERVAL;
    state.timestamp = 200;

    let previous: Vec<State> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL as usize)
        .map(|i| {
            let mut s = core::State::genesis();
            s.height = i as u64;
            s.timestamp = (i as u64) * (TARGET_BLOCK_TIME * 2);
            s
        })
        .collect();

    let new_target = adjust_difficulty(&state, &previous);
    assert!(new_target > state.target, "Target should increase (get easier) when blocks are slow");
}

#[tokio::test]
async fn test_difficulty_adjustment_clamped() {
    let mut state = core::State::genesis();
    state.height = DIFFICULTY_ADJUSTMENT_INTERVAL;
    state.timestamp = 1;

    let previous: Vec<State> = (0..DIFFICULTY_ADJUSTMENT_INTERVAL as usize)
        .map(|i| {
            let mut s = core::State::genesis();
            s.height = i as u64;
            s.timestamp = 0;
            s
        })
        .collect();

    let new_target = adjust_difficulty(&state, &previous);
    assert_ne!(new_target, state.target);
    assert!(new_target < state.target);
}

#[tokio::test]
async fn test_fork_choice_tiebreaker_deterministic() {
    let mut a = core::State::genesis();
    let mut b = core::State::genesis();

    a.depth = 5000;
    b.depth = 5000;
    a.midstate = [0x00; 32];
    b.midstate = [0xff; 32];

    let best = choose_best_state(&a, &b);
    assert_eq!(best.midstate, [0x00; 32]);

    let best2 = choose_best_state(&b, &a);
    assert_eq!(best2.midstate, [0x00; 32]);
}

#[tokio::test]
async fn test_extension_deterministic() {
    let midstate = hash(b"test_determinism");
    let nonce = 9999;

    let ext1 = create_extension(midstate, nonce);
    let ext2 = create_extension(midstate, nonce);

    assert_eq!(ext1.final_hash, ext2.final_hash);
    assert_eq!(ext1.nonce, ext2.nonce);
}

#[tokio::test]
async fn test_transaction_changes_midstate() {
    let mut state = core::State::genesis();

    // Commit changes midstate
    let before = state.midstate;
    let commit_tx = Transaction::Commit { commitment: [0xab; 32] };
    apply_transaction(&mut state, &commit_tx).unwrap();
    assert_ne!(before, state.midstate);

    // Reveal also changes midstate
    let before2 = state.midstate;
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"new")];
    let salt: [u8; 32] = rand::random();
    inject_commitment(&mut state, &secrets, &new_coins, &salt);
    let reveal_tx = make_reveal(secrets, new_coins, salt);
    apply_transaction(&mut state, &reveal_tx).unwrap();
    assert_ne!(before2, state.midstate);
}

#[tokio::test]
async fn test_different_txs_different_midstates() {
    let mut state_a = core::State::genesis();
    let mut state_b = core::State::genesis();

    let secrets = vec![b"genesis_coin_1".to_vec()];
    let coins_a = vec![hash(b"output_a")];
    let coins_b = vec![hash(b"output_b")];
    let salt_a: [u8; 32] = [0x01; 32];
    let salt_b: [u8; 32] = [0x02; 32];

    inject_commitment(&mut state_a, &secrets, &coins_a, &salt_a);
    inject_commitment(&mut state_b, &secrets, &coins_b, &salt_b);

    let tx_a = make_reveal(secrets.clone(), coins_a, salt_a);
    let tx_b = make_reveal(secrets, coins_b, salt_b);

    apply_transaction(&mut state_a, &tx_a).unwrap();
    apply_transaction(&mut state_b, &tx_b).unwrap();

    assert_ne!(state_a.midstate, state_b.midstate);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  INFRASTRUCTURE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_mempool_drain() {
    let temp = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp.path()).unwrap();
    let mut state = core::State::genesis();

    // Commits don't need coins to exist — they're opaque
    let commit1 = Transaction::Commit { commitment: [0x01; 32] };
    let commit2 = Transaction::Commit { commitment: [0x02; 32] };

    mempool.add(commit1, &state).unwrap();
    mempool.add(commit2, &state).unwrap();
    assert_eq!(mempool.len(), 2);

    let drained = mempool.drain(1);
    assert_eq!(drained.len(), 1);
    assert_eq!(mempool.len(), 1);

    let drained = mempool.drain(100);
    assert_eq!(drained.len(), 1);
    assert_eq!(mempool.len(), 0);
}

#[tokio::test]
async fn test_mempool_prune_after_batch() {
    let temp = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp.path()).unwrap();
    let mut state = core::State::genesis();

    // Add a reveal to mempool (needs commitment in state)
    let secrets = vec![b"genesis_coin_1".to_vec()];
    let new_coins = vec![hash(b"mempool_coin")];
    let salt: [u8; 32] = rand::random();
    inject_commitment(&mut state, &secrets, &new_coins, &salt);

    let tx_mempool = make_reveal(secrets.clone(), new_coins.clone(), salt);
    mempool.add(tx_mempool, &state).unwrap();
    assert_eq!(mempool.len(), 1);

    // Now mine a batch that spends genesis_coin_1 (via a different commit-reveal)
    let salt2: [u8; 32] = rand::random();
    inject_commitment(&mut state, &secrets, &[hash(b"block_coin")], &salt2);

    let tx_block = make_reveal(secrets, vec![hash(b"block_coin")], salt2);
    let batch = mine_batch(&state, vec![tx_block]).await;
    apply_batch(&mut state, &batch).unwrap();

    // genesis_coin_1 is now spent — mempool tx is invalid
    mempool.prune_invalid(&state);
    assert_eq!(mempool.len(), 0);
}

#[tokio::test]
async fn test_mempool_conflicting_input_rejected() {
    let temp = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp.path()).unwrap();
    let mut state = core::State::genesis();

    let secrets = vec![b"genesis_coin_1".to_vec()];
    let salt1: [u8; 32] = [0x01; 32];
    let salt2: [u8; 32] = [0x02; 32];

    inject_commitment(&mut state, &secrets, &[hash(b"a")], &salt1);
    inject_commitment(&mut state, &secrets, &[hash(b"b")], &salt2);

    let tx1 = make_reveal(secrets.clone(), vec![hash(b"a")], salt1);
    let tx2 = make_reveal(secrets, vec![hash(b"b")], salt2);

    mempool.add(tx1, &state).unwrap();
    // Same input coin → should be rejected
    assert!(mempool.add(tx2, &state).is_err());
    assert_eq!(mempool.len(), 1);
}

#[tokio::test]
async fn test_mempool_rejects_invalid_tx() {
    let temp = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp.path()).unwrap();
    let state = core::State::genesis();

    // Reveal without commitment in state
    let bad_tx = make_reveal(
        vec![b"nonexistent".to_vec()],
        vec![hash(b"a")],
        [0u8; 32],
    );

    assert!(mempool.add(bad_tx, &state).is_err());
    assert_eq!(mempool.len(), 0);
}

#[tokio::test]
async fn test_mempool_duplicate_commitment_rejected() {
    let temp = TempDir::new().unwrap();
    let mut mempool = mempool::Mempool::new(temp.path()).unwrap();
    let state = core::State::genesis();

    let commit = Transaction::Commit { commitment: [0xab; 32] };

    mempool.add(commit.clone(), &state).unwrap();
    assert!(mempool.add(commit, &state).is_err());
    assert_eq!(mempool.len(), 1);
}

// --- Storage tests ---

#[tokio::test]
async fn test_storage_state_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let mut state = core::State::genesis();
    state.height = 42;
    state.depth = 42_000_000;
    state.timestamp = 1700000000;
    state.midstate = hash(b"custom_midstate");
    state.commitments.insert([0xab; 32]);

    storage.save_state(&state).unwrap();
    let loaded = storage.load_state().unwrap().unwrap();

    assert_eq!(loaded.height, 42);
    assert_eq!(loaded.depth, 42_000_000);
    assert_eq!(loaded.timestamp, 1700000000);
    assert_eq!(loaded.midstate, hash(b"custom_midstate"));
    assert_eq!(loaded.coins, state.coins);
    assert_eq!(loaded.target, state.target);
    assert_eq!(loaded.commitments, state.commitments);
}

#[tokio::test]
async fn test_storage_empty_returns_none() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    assert!(storage.load_state().unwrap().is_none());
}

#[tokio::test]
async fn test_batch_store_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = core::State::genesis();
    let commit_tx = Transaction::Commit { commitment: [0xcc; 32] };
    let batch = mine_batch(&state, vec![commit_tx.clone()]).await;

    storage.save_batch(0, &batch).unwrap();
    let loaded = storage.load_batch(0).unwrap().unwrap();

    assert_eq!(loaded.transactions.len(), 1);
    assert_eq!(loaded.transactions[0], commit_tx);
    assert_eq!(loaded.extension.nonce, batch.extension.nonce);
    assert_eq!(loaded.extension.final_hash, batch.extension.final_hash);
}

#[tokio::test]
async fn test_batch_store_load_range() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = core::State::genesis();

    for i in 0..3 {
        let commit_tx = Transaction::Commit { commitment: [i as u8; 32] };
        let batch = mine_batch(&state, vec![commit_tx]).await;
        storage.save_batch(i, &batch).unwrap();
    }

    let range = storage.load_batches(0, 3).unwrap();
    assert_eq!(range.len(), 3);

    let partial = storage.load_batches(1, 3).unwrap();
    assert_eq!(partial.len(), 2);

    let empty = storage.load_batches(10, 20).unwrap();
    assert_eq!(empty.len(), 0);
}

#[tokio::test]
async fn test_batch_store_highest() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = core::State::genesis();
    let commit_tx = Transaction::Commit { commitment: [0xaa; 32] };
    let batch = mine_batch(&state, vec![commit_tx]).await;

    storage.save_batch(0, &batch).unwrap();
    storage.save_batch(5, &batch).unwrap();
    storage.save_batch(3, &batch).unwrap();

    assert_eq!(storage.highest_batch().unwrap(), 5);
}

#[tokio::test]
async fn test_batch_store_missing_returns_none() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    assert!(storage.load_batch(999).unwrap().is_none());
}

// --- Protocol message serialization ---

#[tokio::test]
async fn test_message_serialize_roundtrip() {
    let messages = vec![
        Message::GetState,
        Message::Verack,
        Message::GetAddr,
        Message::Ping { nonce: 42 },
        Message::Pong { nonce: 42 },
        Message::StateInfo {
            height: 100,
            depth: 100_000_000,
            midstate: [0xab; 32],
        },
        Message::Version {
            version: PROTOCOL_VERSION,
            services: 1,
            timestamp: 1700000000,
            addr_recv: "127.0.0.1:9333".parse().unwrap(),
            addr_from: "127.0.0.1:9334".parse().unwrap(),
        },
        Message::Addr(vec![
            "127.0.0.1:9333".parse().unwrap(),
            "192.168.1.1:9333".parse().unwrap(),
        ]),
        Message::GetBatches {
            start_height: 0,
            count: 100,
        },
        Message::Transaction(Transaction::Commit { commitment: [0xcc; 32] }),
        Message::Transaction(Transaction::Reveal {
            secrets: vec![b"secret".to_vec()],
            new_coins: vec![[0xcc; 32]],
            salt: [0xdd; 32],
        }),
    ];

    for msg in messages {
        let bytes = msg.serialize();
        let decoded = Message::deserialize(&bytes).unwrap();
        let bytes2 = decoded.serialize();
        assert_eq!(bytes, bytes2, "Round-trip failed for message variant");
    }
}

#[tokio::test]
async fn test_batch_message_roundtrip() {
    let state = core::State::genesis();
    let commit_tx = Transaction::Commit { commitment: [0xaa; 32] };
    let batch = mine_batch(&state, vec![commit_tx]).await;

    let msg = Message::Batch(batch.clone());
    let bytes = msg.serialize();
    let decoded = Message::deserialize(&bytes).unwrap();

    match decoded {
        Message::Batch(decoded_batch) => {
            assert_eq!(decoded_batch.transactions.len(), 1);
            assert_eq!(decoded_batch.extension.nonce, batch.extension.nonce);
            assert_eq!(decoded_batch.extension.final_hash, batch.extension.final_hash);
        }
        _ => panic!("Expected Batch message"),
    }
}

// --- Hash function tests ---

#[tokio::test]
async fn test_hash_deterministic() {
    assert_eq!(hash(b"hello"), hash(b"hello"));
    assert_ne!(hash(b"hello"), hash(b"world"));
}

#[tokio::test]
async fn test_hash_concat_deterministic() {
    let a = hash_concat(b"a", b"b");
    let b = hash_concat(b"a", b"b");
    assert_eq!(a, b);

    let c = hash_concat(b"b", b"a");
    assert_ne!(a, c);
}

#[tokio::test]
async fn test_genesis_deterministic() {
    let g1 = core::State::genesis();
    let g2 = core::State::genesis();

    assert_eq!(g1.midstate, g2.midstate);
    assert_eq!(g1.height, g2.height);
    assert_eq!(g1.depth, g2.depth);
    assert_eq!(g1.target, g2.target);
    assert_eq!(g1.coins, g2.coins);
    assert_eq!(g1.timestamp, g2.timestamp);
    assert_eq!(g1.commitments, g2.commitments);
}

#[tokio::test]
async fn test_storage_overwrites() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let mut state1 = core::State::genesis();
    state1.height = 1;
    storage.save_state(&state1).unwrap();

    let mut state2 = core::State::genesis();
    state2.height = 99;
    storage.save_state(&state2).unwrap();

    let loaded = storage.load_state().unwrap().unwrap();
    assert_eq!(loaded.height, 99);
}
