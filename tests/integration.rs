use midstate::core::{self, hash, compute_commitment, EXTENSION_ITERATIONS, DIFFICULTY_ADJUSTMENT_INTERVAL};
use midstate::core::extension::{create_extension, verify_extension, mine_extension};
use midstate::core::transaction::{apply_transaction, validate_transaction};
use midstate::core::state::{apply_batch, choose_best_state, adjust_difficulty};
use midstate::core::types::{block_reward, Transaction, Batch, Extension, State, BLOCKS_PER_YEAR, INITIAL_REWARD, CHECKPOINT_INTERVAL};
use midstate::network::protocol::Message;
use midstate::storage::Storage;
use tempfile::TempDir;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Mine a valid batch from a set of transactions against a state
async fn mine_batch(state: &State, transactions: Vec<Transaction>) -> Batch {
    let mut candidate = state.clone();
    let mut total_fees = 0;

    for tx in &transactions {
        total_fees += tx.fee() as u64;
        apply_transaction(&mut candidate, tx).unwrap();
    }

    let midstate = candidate.midstate;
    let target = state.target;

    // Generate coinbase coins to satisfy protocol requirements
    let reward = block_reward(state.height);
    let coinbase_count = reward + total_fees;
    let coinbase: Vec<[u8; 32]> = (0..coinbase_count).map(|_| rand::random()).collect();

    // Fold coinbase into midstate (as done in apply_batch)
    let mut mining_midstate = midstate;
    for coin in &coinbase {
        mining_midstate = core::hash_concat(&mining_midstate, coin);
    }

    let extension = tokio::task::spawn_blocking(move || mine_extension(mining_midstate, target))
        .await
        .unwrap();

    Batch { transactions, extension, coinbase }
}

/// Create a Commit transaction
fn make_commit(input_coins: &[[u8; 32]], new_coins: &[[u8; 32]], salt: &[u8; 32]) -> Transaction {
    let commitment = compute_commitment(input_coins, new_coins, salt);
    Transaction::Commit { commitment }
}

/// Build a full valid reveal transaction using real WOTS seeds.
/// `seeds` are the private seeds for each input coin (coin = wots::keygen(seed)).
fn make_reveal(
    seeds: &[[u8; 32]],
    input_coins: &[[u8; 32]],
    new_coins: &[[u8; 32]],
    salt: &[u8; 32],
) -> Transaction {
    let commitment = compute_commitment(input_coins, new_coins, salt);
    // CHANGE: Type is now Vec<Vec<u8>>
    let signatures: Vec<Vec<u8>> = seeds
        .iter()
        .map(|seed| {
            let sig = core::wots::sign(seed, &commitment);
            core::wots::sig_to_bytes(&sig) // CHANGE: Convert chunks to bytes
        })
        .collect();

    Transaction::Reveal {
        input_coins: input_coins.to_vec(),
        signatures,
        new_coins: new_coins.to_vec(),
        salt: *salt,
    }
}

/// Helper: genesis seeds and coins.
fn genesis_seeds_and_coins() -> (Vec<[u8; 32]>, Vec<[u8; 32]>) {
    let seeds = vec![
        hash(b"genesis_coin_1"),
        hash(b"genesis_coin_2"),
        hash(b"genesis_coin_3"),
    ];
    let coins: Vec<[u8; 32]> = seeds.iter().map(|s| core::wots::keygen(s)).collect();
    (seeds, coins)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMMIT-REVEAL FLOW
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_commit_reveal_basic_flow() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    let input_coins = vec![coins[0], coins[1]];
    let new_coins = vec![hash(b"coin_output_a")];
    let salt: [u8; 32] = rand::random();

    // 1. Commit
    let commit_tx = make_commit(&input_coins, &new_coins, &salt);
    let commit_batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &commit_batch).unwrap();

    let expected_commitment = compute_commitment(&input_coins, &new_coins, &salt);
    assert!(state.commitments.contains(&expected_commitment));

    // 2. Reveal
    let reveal_tx = make_reveal(&[seeds[0], seeds[1]], &input_coins, &new_coins, &salt);
    let reveal_batch = mine_batch(&state, vec![reveal_tx]).await;
    apply_batch(&mut state, &reveal_batch).unwrap();

    // Commitment consumed, new coin created, inputs spent
    assert!(!state.commitments.contains(&expected_commitment));
    assert!(state.coins.contains(&new_coins[0]));
    assert!(!state.coins.contains(&coins[0]));
    assert!(!state.coins.contains(&coins[1]));
}

// ─── BUG FIX: test_reveal_without_commit_rejected ───────────────────────────
// ORIGINAL BUG: Used 1 input → 1 output, which hits the fee check
// (inputs <= outputs) BEFORE reaching the "no matching commitment" check.
// The test passed by accident — wrong error path exercised.
// FIX: Use 2 inputs → 1 output so fee check passes, and the missing-commitment
// error is what actually rejects the transaction.

#[tokio::test]
async fn test_reveal_without_commit_rejected() {
    let state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    let input_coins = vec![coins[0], coins[1]]; // 2 inputs
    let new_coins = vec![hash(b"stolen")];       // 1 output → fee satisfied
    let salt: [u8; 32] = [0u8; 32];

    // Build a reveal with real WOTS sigs but no commit on-chain
    let reveal_tx = make_reveal(&[seeds[0], seeds[1]], &input_coins, &new_coins, &salt);

    let err = validate_transaction(&state, &reveal_tx).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("No matching commitment"),
        "Expected 'No matching commitment' error, got: '{}'",
        msg
    );
}

// ─── BUG FIX: test_front_running_prevented ──────────────────────────────────
// ORIGINAL BUG: 1 input → 1 output → fee check fires before commitment check.
// Also used mock_sigs (all-zero) which never exercises real commitment binding.
// FIX: Use 2 inputs → 1 output. Commit with alice_dest, then try reveal with
// attacker_dest. The commitment mismatch (not the fee) should cause rejection.

#[tokio::test]
async fn test_front_running_prevented() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    // Alice commits: coin0 + coin1 → alice_dest
    let alice_dest = vec![hash(b"alice_output")];
    let input_coins = vec![coins[0], coins[1]];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&input_coins, &alice_dest, &salt);
    let commit_batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &commit_batch).unwrap();

    // Attacker tries to redirect to their own destination using same inputs + salt
    let attacker_dest = vec![hash(b"attacker_output")];

    // Attacker builds reveal with the REAL seeds (worst case: they somehow got them)
    // but changing the destination. The commitment won't match.
    let attacker_reveal = make_reveal(
        &[seeds[0], seeds[1]],
        &input_coins,
        &attacker_dest, // different destination
        &salt,
    );

    let err = validate_transaction(&state, &attacker_reveal).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("No matching commitment"),
        "Expected commitment mismatch error, got: '{}'",
        msg
    );
}

#[tokio::test]
async fn test_duplicate_commitment_rejected() {
    let mut state = State::genesis();
    let inputs = vec![[1u8; 32]];
    let outputs = vec![[2u8; 32]];
    let salt = [3u8; 32];

    let commit_tx = make_commit(&inputs, &outputs, &salt);

    apply_transaction(&mut state, &commit_tx).unwrap();
    // Same commitment again should fail
    let err = apply_transaction(&mut state, &commit_tx.clone()).unwrap_err();
    assert!(err.to_string().contains("Duplicate commitment"));
}

#[tokio::test]
async fn test_double_spend_rejected() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    // 1. Spend coin0 + coin1 → A
    let inputs = vec![coins[0], coins[1]];
    let dest_a = vec![hash(b"A")];
    let salt1: [u8; 32] = rand::random();

    let commit1 = make_commit(&inputs, &dest_a, &salt1);
    let batch1 = mine_batch(&state, vec![commit1]).await;
    apply_batch(&mut state, &batch1).unwrap();

    let reveal1 = make_reveal(&[seeds[0], seeds[1]], &inputs, &dest_a, &salt1);
    let batch2 = mine_batch(&state, vec![reveal1]).await;
    apply_batch(&mut state, &batch2).unwrap();

    assert!(!state.coins.contains(&coins[0]));

    // 2. Try to spend coin0 again (with fresh coin2 to satisfy fee)
    let inputs_bad = vec![coins[0], coins[2]];
    let dest_b = vec![hash(b"B")];
    let salt2: [u8; 32] = rand::random();

    // Commit succeeds (format is fine, doesn't check coin existence)
    let commit2 = make_commit(&inputs_bad, &dest_b, &salt2);
    apply_transaction(&mut state, &commit2).unwrap();

    // Reveal fails: coin0 is missing from state
    let reveal2 = make_reveal(&[seeds[0], seeds[2]], &inputs_bad, &dest_b, &salt2);
    let err = validate_transaction(&state, &reveal2).unwrap_err();
    assert!(
        err.to_string().contains("not found"),
        "Expected 'not found' for spent coin, got: '{}'",
        err
    );
}

#[tokio::test]
async fn test_reveal_input_output_count_mismatch() {
    let state = State::genesis();

    // Signatures length != Inputs length
    let tx = Transaction::Reveal {
        input_coins: vec![[1u8; 32], [2u8; 32]],
        // CHANGE: Use byte vectors of correct size
        signatures: vec![vec![0u8; core::wots::SIG_SIZE]], 
        new_coins: vec![[3u8; 32]],
        salt: [0u8; 32],
    };

    let err = validate_transaction(&state, &tx).unwrap_err();
    assert!(err.to_string().contains("Signature count"));
}

#[tokio::test]
async fn test_reveal_insufficient_fees() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    // Set up a commitment for 1 input → 1 output (no fee)
    let input_coins = vec![coins[0]];
    let new_coins = vec![hash(b"output")];
    let salt: [u8; 32] = rand::random();

    // Inject the commitment so the fee check is the rejection point, not missing commitment
    let commitment = compute_commitment(&input_coins, &new_coins, &salt);
    state.commitments.insert(commitment);

    let reveal_tx = make_reveal(&[seeds[0]], &input_coins, &new_coins, &salt);

    let err = validate_transaction(&state, &reveal_tx).unwrap_err();
    assert!(
        err.to_string().contains("Inputs") && err.to_string().contains("outputs"),
        "Expected fee-related error, got: '{}'",
        err
    );
}

#[tokio::test]
async fn test_reveal_empty_inputs_rejected() {
    let state = State::genesis();

    let tx = Transaction::Reveal {
        input_coins: vec![],
        signatures: vec![],
        new_coins: vec![hash(b"out")],
        salt: [0u8; 32],
    };

    let err = validate_transaction(&state, &tx).unwrap_err();
    assert!(err.to_string().contains("at least one coin"));
}

#[tokio::test]
async fn test_reveal_empty_outputs_rejected() {
    let state = State::genesis();

    let tx = Transaction::Reveal {
        input_coins: vec![[1u8; 32], [2u8; 32]],
        // CHANGE: Use byte vectors of correct size
        signatures: vec![vec![0u8; core::wots::SIG_SIZE]; 2],
        new_coins: vec![],
        salt: [0u8; 32],
    };

    let err = validate_transaction(&state, &tx).unwrap_err();
    assert!(err.to_string().contains("at least one"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WOTS SIGNATURE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_wots_forgery_rejected() {
    // Attacker has the right commitment on chain but uses the WRONG seed to sign.
    let mut state = State::genesis();
    let (_seeds, coins) = genesis_seeds_and_coins();

    let input_coins = vec![coins[0], coins[1]];
    let new_coins = vec![hash(b"output")];
    let salt: [u8; 32] = rand::random();

    // Commit
    let commit_tx = make_commit(&input_coins, &new_coins, &salt);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    // Reveal with WRONG seeds (attacker doesn't know the real seeds)
    let wrong_seed0: [u8; 32] = rand::random();
    let wrong_seed1: [u8; 32] = rand::random();
    let bad_reveal = make_reveal(&[wrong_seed0, wrong_seed1], &input_coins, &new_coins, &salt);

    let err = validate_transaction(&state, &bad_reveal).unwrap_err();
    assert!(
        err.to_string().contains("Invalid WOTS signature"),
        "Expected WOTS signature failure, got: '{}'",
        err
    );
}

#[tokio::test]
async fn test_wots_sig_wrong_message_rejected() {
    // Sign with the correct seed but against the WRONG commitment message.
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    let input_coins = vec![coins[0], coins[1]];
    let new_coins = vec![hash(b"output")];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&input_coins, &new_coins, &salt);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

// Sign against a completely different message
    let wrong_message: [u8; 32] = rand::random();
    let bad_sig0 = core::wots::sign(&seeds[0], &wrong_message);
    let bad_sig1 = core::wots::sign(&seeds[1], &wrong_message);

    let bad_reveal = Transaction::Reveal {
        input_coins: input_coins.to_vec(),
        // CHANGE: Convert raw sigs to bytes
        signatures: vec![
            core::wots::sig_to_bytes(&bad_sig0),
            core::wots::sig_to_bytes(&bad_sig1)
        ],
        new_coins: new_coins.to_vec(),
        salt,
    };

    let err = validate_transaction(&state, &bad_reveal).unwrap_err();
    assert!(err.to_string().contains("Invalid WOTS signature"));
}

#[tokio::test]
async fn test_wots_sig_serialization_roundtrip() {
    let seed: [u8; 32] = rand::random();
    let msg = hash(b"test message");
    let sig = core::wots::sign(&seed, &msg);

    let bytes = core::wots::sig_to_bytes(&sig);
    assert_eq!(bytes.len(), core::wots::SIG_SIZE);

    let sig2 = core::wots::sig_from_bytes(&bytes).unwrap();
    assert_eq!(sig, sig2);

    // Verify the deserialized sig still verifies
    let coin = core::wots::keygen(&seed);
    assert!(core::wots::verify(&sig2, &msg, &coin));
}

#[tokio::test]
async fn test_wots_sig_from_bytes_wrong_length() {
    assert!(core::wots::sig_from_bytes(&[0u8; 100]).is_none());
    assert!(core::wots::sig_from_bytes(&[]).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COINBASE & BLOCK REWARD TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_coinbase_wrong_count_rejected() {
    let state = State::genesis();
    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);

    let mut candidate = state.clone();
    apply_transaction(&mut candidate, &commit_tx).unwrap();

    let reward = block_reward(state.height);
    // Transaction has 0 fees, so coinbase should be exactly `reward` coins.
    // Give it reward + 1 (one too many).
    let bad_coinbase: Vec<[u8; 32]> = (0..reward + 1).map(|_| rand::random()).collect();

    let mut mining_midstate = candidate.midstate;
    for coin in &bad_coinbase {
        mining_midstate = core::hash_concat(&mining_midstate, coin);
    }

    let extension = tokio::task::spawn_blocking(move || mine_extension(mining_midstate, state.target))
        .await
        .unwrap();

    let bad_batch = Batch {
        transactions: vec![commit_tx],
        extension,
        coinbase: bad_coinbase,
    };

    let mut test_state = state.clone();
    let err = apply_batch(&mut test_state, &bad_batch).unwrap_err();
    assert!(
        err.to_string().contains("Invalid coinbase count"),
        "Expected coinbase count error, got: '{}'",
        err
    );
}

#[tokio::test]
async fn test_coinbase_includes_fees() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    // Set up a reveal that pays 1 coin fee (2 inputs → 1 output)
    let input_coins = vec![coins[0], coins[1]];
    let new_coins = vec![hash(b"out")];
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&input_coins, &new_coins, &salt);
    let batch1 = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch1).unwrap();

    let reveal_tx = make_reveal(&[seeds[0], seeds[1]], &input_coins, &new_coins, &salt);
    assert_eq!(reveal_tx.fee(), 1); // 2 inputs - 1 output = 1 fee

    // mine_batch should create reward + 1 coinbase coins
    let batch2 = mine_batch(&state, vec![reveal_tx]).await;
    let expected_coinbase = block_reward(state.height) + 1;
    assert_eq!(
        batch2.coinbase.len() as u64,
        expected_coinbase,
        "Coinbase should include reward ({}) + fees (1)",
        block_reward(state.height)
    );

    apply_batch(&mut state, &batch2).unwrap();
}

#[test]
fn test_block_reward_halving() {
    // Height 0: full reward
    assert_eq!(block_reward(0), INITIAL_REWARD);

    // After 1 halving
    assert_eq!(block_reward(BLOCKS_PER_YEAR), INITIAL_REWARD / 2);

    // After 2 halvings
    assert_eq!(block_reward(BLOCKS_PER_YEAR * 2), INITIAL_REWARD / 4);

    // After 8+ halvings: minimum 1
    assert_eq!(block_reward(BLOCKS_PER_YEAR * 8), 1);
    assert_eq!(block_reward(BLOCKS_PER_YEAR * 100), 1);
}

#[test]
fn test_block_reward_never_zero() {
    // Sweep a range of heights to ensure reward never hits 0
    for h in (0..BLOCKS_PER_YEAR * 20).step_by(BLOCKS_PER_YEAR as usize / 10) {
        assert!(block_reward(h) >= 1, "block_reward({}) was 0", h);
    }
}

#[tokio::test]
async fn test_duplicate_coinbase_coin_rejected() {
    let state = State::genesis();
    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);

    let mut candidate = state.clone();
    apply_transaction(&mut candidate, &commit_tx).unwrap();

    let reward = block_reward(state.height);
    // Create coinbase with a DUPLICATE coin
    let dup_coin: [u8; 32] = rand::random();
    let coinbase: Vec<[u8; 32]> = vec![dup_coin; reward as usize]; // all the same

    let mut mining_midstate = candidate.midstate;
    for coin in &coinbase {
        mining_midstate = core::hash_concat(&mining_midstate, coin);
    }

    let extension = tokio::task::spawn_blocking(move || mine_extension(mining_midstate, state.target))
        .await
        .unwrap();

    let bad_batch = Batch {
        transactions: vec![commit_tx],
        extension,
        coinbase,
    };

    let mut test_state = state.clone();
    let err = apply_batch(&mut test_state, &bad_batch).unwrap_err();
    assert!(
        err.to_string().contains("Duplicate coinbase"),
        "Expected duplicate coinbase error, got: '{}'",
        err
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  EXTENSION / PROOF OF WORK TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_extension_verify_rejects_bad_target() {
    let midstate = hash(b"test_midstate");
    // create_extension doesn't check target — it just builds the chain
    let ext = create_extension(midstate, 42);

    // Now verify against an impossibly hard target
    let hard_target = [0x00; 32];
    let err = verify_extension(midstate, &ext, &hard_target).unwrap_err();
    assert!(err.to_string().contains("difficulty target"));
}

#[tokio::test]
async fn test_extension_verify_rejects_wrong_midstate() {
    let midstate = hash(b"real_midstate");
    let target = [0xff; 32];
    let ext = create_extension(midstate, 42);

    let wrong_midstate = hash(b"wrong_midstate");
    let err = verify_extension(wrong_midstate, &ext, &target).unwrap_err();
    assert!(err.to_string().contains("First checkpoint"));
}

#[tokio::test]
async fn test_extension_verify_rejects_tampered_checkpoint() {
    let midstate = hash(b"test");
    let target = [0xff; 32];
    let mut ext = create_extension(midstate, 42);

    // Tamper ALL middle checkpoints so spot-checking deterministically catches it.
    // (With fast-mining there are few segments and few spot checks, so tampering
    // only one checkpoint can be missed by the random selector.)
    for i in 1..ext.checkpoints.len() - 1 {
        ext.checkpoints[i] = [0xAA; 32];
    }

    let result = verify_extension(midstate, &ext, &target);
    assert!(result.is_err(), "Tampered checkpoints should be detected");
}

#[tokio::test]
async fn test_extension_checkpoint_count() {
    let midstate = hash(b"test");
    let ext = create_extension(midstate, 0);

    let expected_segments = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize;
    let expected_checkpoints = expected_segments + 1; // start + one per segment
    assert_eq!(ext.checkpoints.len(), expected_checkpoints);
}

#[tokio::test]
async fn test_batch_with_invalid_extension_rejected() {
    let state = State::genesis();
    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);

    // Build a batch with completely bogus extension
    let reward = block_reward(state.height);
    let coinbase: Vec<[u8; 32]> = (0..reward).map(|_| rand::random()).collect();

    let num_segments = (EXTENSION_ITERATIONS / CHECKPOINT_INTERVAL) as usize;
    let fake_extension = Extension {
        nonce: 0,
        final_hash: [0u8; 32],
        checkpoints: vec![[0u8; 32]; num_segments + 1],
    };

    let bad_batch = Batch {
        transactions: vec![commit_tx],
        extension: fake_extension,
        coinbase,
    };

    let mut test_state = state.clone();
    assert!(apply_batch(&mut test_state, &bad_batch).is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CHAIN CORRECTNESS TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_midstate_evolves() {
    let mut state = State::genesis();
    let midstate_0 = state.midstate;

    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    assert_ne!(midstate_0, state.midstate);
}

#[tokio::test]
async fn test_height_increments() {
    let mut state = State::genesis();
    assert_eq!(state.height, 0);

    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    assert_eq!(state.height, 1);
}

#[tokio::test]
async fn test_depth_increments_by_extension_iterations() {
    let mut state = State::genesis();
    assert_eq!(state.depth, 0);

    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    assert_eq!(state.depth, EXTENSION_ITERATIONS);
}

#[tokio::test]
async fn test_empty_batch_with_only_coinbase() {
    // A batch with 0 user transactions is valid (coinbase only)
    let state = State::genesis();
    let batch = mine_batch(&state, vec![]).await;

    let mut test_state = state.clone();
    apply_batch(&mut test_state, &batch).unwrap();
    assert_eq!(test_state.height, 1);
}

#[tokio::test]
async fn test_difficulty_adjustment() {
    let mut state = State::genesis();
    state.height = DIFFICULTY_ADJUSTMENT_INTERVAL;
    state.timestamp = 1000;

    // Case 1: Blocks came too fast (1s apart vs 10s target)
    let mut previous_fast: Vec<State> = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        let mut s = State::genesis();
        s.height = i;
        s.timestamp = 1000 - (DIFFICULTY_ADJUSTMENT_INTERVAL - i) * 1;
        previous_fast.push(s);
    }

    let new_target_fast = adjust_difficulty(&state, &previous_fast);
    assert!(new_target_fast < state.target, "Target should decrease when blocks are fast");

    // Case 2: Blocks came too slow (100s apart vs 10s target)
    let mut previous_slow: Vec<State> = Vec::new();
    for i in 0..DIFFICULTY_ADJUSTMENT_INTERVAL {
        let mut s = State::genesis();
        s.height = i;
        s.timestamp = 1000 - (DIFFICULTY_ADJUSTMENT_INTERVAL - i) * 100;
        previous_slow.push(s);
    }

    let new_target_slow = adjust_difficulty(&state, &previous_slow);
    assert!(new_target_slow > state.target, "Target should increase when blocks are slow");
}

#[tokio::test]
async fn test_difficulty_no_adjustment_at_wrong_height() {
    let mut state = State::genesis();
    state.height = 5; // Not at adjustment interval

    let target_before = state.target;
    let new_target = adjust_difficulty(&state, &[]);
    assert_eq!(target_before, new_target, "No adjustment at non-interval heights");
}

#[tokio::test]
async fn test_choose_best_state_prefers_deeper() {
    let mut a = State::genesis();
    let mut b = State::genesis();
    a.depth = 100;
    b.depth = 200;

    let best = choose_best_state(&a, &b);
    assert_eq!(best.depth, 200);
}

#[tokio::test]
async fn test_choose_best_state_tiebreak_on_midstate() {
    let mut a = State::genesis();
    let mut b = State::genesis();
    a.depth = 100;
    b.depth = 100;
    a.midstate = [0x01; 32];
    b.midstate = [0x02; 32];

    let best = choose_best_state(&a, &b);
    assert_eq!(best.midstate, [0x01; 32], "Lower midstate wins tie");
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TRANSACTION FEE ACCOUNTING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_commit_fee_is_zero() {
    let tx = Transaction::Commit { commitment: [0u8; 32] };
    assert_eq!(tx.fee(), 0);
}

#[test]
fn test_reveal_fee_calculation() {
    let tx = Transaction::Reveal {
        input_coins: vec![[0u8; 32]; 3],
        signatures: vec![],
        new_coins: vec![[0u8; 32]; 1],
        salt: [0u8; 32],
    };
    assert_eq!(tx.fee(), 2); // 3 - 1

    let tx2 = Transaction::Reveal {
        input_coins: vec![[0u8; 32]; 5],
        signatures: vec![],
        new_coins: vec![[0u8; 32]; 2],
        salt: [0u8; 32],
    };
    assert_eq!(tx2.fee(), 3); // 5 - 2
}

#[test]
fn test_input_coins_method() {
    let commit = Transaction::Commit { commitment: [0u8; 32] };
    assert!(commit.input_coins().is_empty());

    let coins = vec![[1u8; 32], [2u8; 32]];
    let reveal = Transaction::Reveal {
        input_coins: coins.clone(),
        signatures: vec![],
        new_coins: vec![[3u8; 32]],
        salt: [0u8; 32],
    };
    assert_eq!(reveal.input_coins(), coins);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DUPLICATE COIN CREATION
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_reveal_duplicate_output_coin_rejected() {
    let mut state = State::genesis();
    let (seeds, coins) = genesis_seeds_and_coins();

    // 3 inputs → 2 identical outputs (duplicate new_coin)
    let dup_coin = hash(b"dup_output");
    let input_coins = vec![coins[0], coins[1], coins[2]];
    let new_coins = vec![dup_coin, dup_coin]; // DUPLICATE
    let salt: [u8; 32] = rand::random();

    let commit_tx = make_commit(&input_coins, &new_coins, &salt);
    let batch = mine_batch(&state, vec![commit_tx]).await;
    apply_batch(&mut state, &batch).unwrap();

    let reveal_tx = make_reveal(&[seeds[0], seeds[1], seeds[2]], &input_coins, &new_coins, &salt);

    let mut test_state = state.clone();
    let err = apply_transaction(&mut test_state, &reveal_tx).unwrap_err();
    assert!(
        err.to_string().contains("Duplicate coin"),
        "Expected duplicate coin error, got: '{}'",
        err
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SPENDING NON-EXISTENT COIN
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_spend_nonexistent_coin_rejected() {
    let mut state = State::genesis();

    let fake_seed: [u8; 32] = rand::random();
    let fake_coin = core::wots::keygen(&fake_seed);
    let (_, coins) = genesis_seeds_and_coins();

    // Use a coin that doesn't exist in state + a real coin for fee
    let input_coins = vec![fake_coin, coins[0]];
    let new_coins = vec![hash(b"out")];
    let salt: [u8; 32] = rand::random();

    // Inject commitment so we get past that check
    let commitment = compute_commitment(&input_coins, &new_coins, &salt);
    state.commitments.insert(commitment);

    let reveal = make_reveal(
        &[fake_seed, hash(b"genesis_coin_1")],
        &input_coins,
        &new_coins,
        &salt,
    );

    let err = validate_transaction(&state, &reveal).unwrap_err();
    assert!(err.to_string().contains("not found"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  INFRASTRUCTURE TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn test_mempool_persistence() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("mempool");

    let tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    let state = State::genesis();

    {
        let mut mempool = midstate::mempool::Mempool::new(&db_path).unwrap();
        mempool.add(tx.clone(), &state).unwrap();
        assert_eq!(mempool.len(), 1);
    }

    {
        let mempool = midstate::mempool::Mempool::new(&db_path).unwrap();
        assert_eq!(mempool.len(), 1);
        assert_eq!(mempool.transactions()[0], tx);
    }
}

#[tokio::test]
async fn test_mempool_duplicate_commit_rejected() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("mempool");
    let state = State::genesis();

    let mut mempool = midstate::mempool::Mempool::new(&db_path).unwrap();
    let tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);

    mempool.add(tx.clone(), &state).unwrap();
    let err = mempool.add(tx, &state).unwrap_err();
    assert!(err.to_string().contains("already in mempool"));
}

#[tokio::test]
async fn test_mempool_drain() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("mempool");
    let state = State::genesis();

    let mut mempool = midstate::mempool::Mempool::new(&db_path).unwrap();

    for i in 0u8..5 {
        let tx = make_commit(&[[i; 32]], &[[(i + 100); 32]], &[0u8; 32]);
        mempool.add(tx, &state).unwrap();
    }
    assert_eq!(mempool.len(), 5);

    let drained = mempool.drain(3);
    assert_eq!(drained.len(), 3);
    assert_eq!(mempool.len(), 2);
}

#[tokio::test]
async fn test_mempool_prune_invalid() {
    let temp = TempDir::new().unwrap();
    let db_path = temp.path().join("mempool");
    let state = State::genesis();

    let mut mempool = midstate::mempool::Mempool::new(&db_path).unwrap();
    let tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    mempool.add(tx.clone(), &state).unwrap();

    // Create a state where this commitment already exists (making the tx invalid)
    let mut new_state = state.clone();
    let commitment = compute_commitment(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    new_state.commitments.insert(commitment);

    mempool.prune_invalid(&new_state);
    assert_eq!(mempool.len(), 0);
}

// --- Storage tests ---

#[tokio::test]
async fn test_storage_batch_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = State::genesis();
    let commit_tx = make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32]);
    let batch = mine_batch(&state, vec![commit_tx.clone()]).await;

    storage.save_batch(1, &batch).unwrap();
    let loaded = storage.load_batch(1).unwrap().unwrap();

    assert_eq!(loaded.transactions.len(), 1);
    assert_eq!(loaded.transactions[0], commit_tx);
    assert!(!loaded.coinbase.is_empty());
}

#[tokio::test]
async fn test_storage_state_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = State::genesis();
    storage.save_state(&state).unwrap();

    let loaded = storage.load_state().unwrap().unwrap();
    assert_eq!(loaded.height, state.height);
    assert_eq!(loaded.midstate, state.midstate);
    assert_eq!(loaded.coins, state.coins);
}

#[tokio::test]
async fn test_storage_mining_seed_roundtrip() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let seed: [u8; 32] = rand::random();
    storage.save_mining_seed(&seed).unwrap();

    let loaded = storage.load_mining_seed().unwrap().unwrap();
    assert_eq!(loaded, seed);
}

#[tokio::test]
async fn test_storage_missing_batch_returns_none() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    assert!(storage.load_batch(999).unwrap().is_none());
}

#[tokio::test]
async fn test_storage_batch_range() {
    let temp = TempDir::new().unwrap();
    let storage = Storage::open(temp.path()).unwrap();

    let state = State::genesis();
    for h in 0..3 {
        let tx = make_commit(&[[(h as u8); 32]], &[[(h as u8 + 100); 32]], &[0u8; 32]);
        let batch = mine_batch(&state, vec![tx]).await;
        storage.save_batch(h, &batch).unwrap();
    }

    let range = storage.load_batches(0, 3).unwrap();
    assert_eq!(range.len(), 3);
}

// --- Protocol message serialization ---

#[tokio::test]
async fn test_message_serialize_roundtrip() {
    let messages = vec![
        Message::GetState,
        Message::Transaction(make_commit(&[[1u8; 32]], &[[2u8; 32]], &[0u8; 32])),
        Message::Ping { nonce: 123 },
        Message::Pong { nonce: 456 },
        Message::GetAddr,
        Message::GetBatches { start_height: 0, count: 10 },
    ];

    for msg in messages {
        let bytes = msg.serialize();
        let decoded = Message::deserialize(&bytes).unwrap();
        assert_eq!(bytes, decoded.serialize());
    }
}

// --- Hash function tests ---

#[test]
fn test_hash_concat_deterministic() {
    let a = hash(b"a");
    let b = hash(b"b");
    assert_eq!(core::hash_concat(&a, &b), core::hash_concat(&a, &b));
}

#[test]
fn test_hash_concat_order_matters() {
    let a = hash(b"a");
    let b = hash(b"b");
    assert_ne!(core::hash_concat(&a, &b), core::hash_concat(&b, &a));
}

#[test]
fn test_compute_commitment_deterministic() {
    let inputs = vec![[1u8; 32]];
    let outputs = vec![[2u8; 32]];
    let salt = [3u8; 32];
    assert_eq!(
        compute_commitment(&inputs, &outputs, &salt),
        compute_commitment(&inputs, &outputs, &salt)
    );
}

#[test]
fn test_compute_commitment_different_salt_different_result() {
    let inputs = vec![[1u8; 32]];
    let outputs = vec![[2u8; 32]];
    let salt1 = [3u8; 32];
    let salt2 = [4u8; 32];
    assert_ne!(
        compute_commitment(&inputs, &outputs, &salt1),
        compute_commitment(&inputs, &outputs, &salt2)
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
//  GENESIS STATE INVARIANTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_genesis_state() {
    let state = State::genesis();
    assert_eq!(state.height, 0);
    assert_eq!(state.depth, 0);
    assert_eq!(state.coins.len(), 3);
    assert!(state.commitments.is_empty());
    assert_eq!(state.timestamp, 0);

    // All genesis coins should be derivable from known seeds
    let (_, coins) = genesis_seeds_and_coins();
    for coin in &coins {
        assert!(state.coins.contains(coin));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COINBASE SEED DETERMINISM
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_coinbase_seed_deterministic() {
    let mining_seed = [42u8; 32];
    let s1 = midstate::wallet::coinbase_seed(&mining_seed, 10, 0);
    let s2 = midstate::wallet::coinbase_seed(&mining_seed, 10, 0);
    assert_eq!(s1, s2);

    // Different height → different seed
    let s3 = midstate::wallet::coinbase_seed(&mining_seed, 11, 0);
    assert_ne!(s1, s3);

    // Different index → different seed
    let s4 = midstate::wallet::coinbase_seed(&mining_seed, 10, 1);
    assert_ne!(s1, s4);
}
