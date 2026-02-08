use midstate::core::{wots, compute_commitment};
use midstate::wallet::{Wallet, WalletData, short_hex, coinbase_seed};
use tempfile::TempDir;
use std::path::PathBuf;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn wallet_path(dir: &TempDir) -> PathBuf {
    dir.path().join("test.wallet")
}

fn create_test_wallet(dir: &TempDir) -> Wallet {
    Wallet::create(&wallet_path(dir), b"testpass").unwrap()
}

fn open_test_wallet(dir: &TempDir) -> Wallet {
    Wallet::open(&wallet_path(dir), b"testpass").unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════════
//  WALLET LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_create_new_wallet() {
    let dir = TempDir::new().unwrap();
    let wallet = create_test_wallet(&dir);
    assert_eq!(wallet.coin_count(), 0);
    assert!(wallet.pending().is_empty());
    assert!(wallet.history().is_empty());
}

#[test]
fn test_create_wallet_already_exists() {
    let dir = TempDir::new().unwrap();
    let _w = create_test_wallet(&dir);
    // Creating again at the same path should fail
    let result = Wallet::create(&wallet_path(&dir), b"testpass");
    let err = result.err().expect("should fail on duplicate create");
    assert!(err.to_string().contains("already exists"));
}

#[test]
fn test_wallet_persistence_roundtrip() {
    let dir = TempDir::new().unwrap();

    let coin;
    {
        let mut w = create_test_wallet(&dir);
        let wc = w.generate(Some("persisted".into())).unwrap();
        coin = wc.coin;
        // wallet drops here, data should be on disk
    }

    {
        let w = open_test_wallet(&dir);
        assert_eq!(w.coin_count(), 1);
        assert_eq!(w.coins()[0].coin, coin);
        assert_eq!(w.coins()[0].label.as_deref(), Some("persisted"));
    }
}

#[test]
fn test_open_wrong_password() {
    let dir = TempDir::new().unwrap();
    let _w = Wallet::create(&wallet_path(&dir), b"correct").unwrap();

    let result = Wallet::open(&wallet_path(&dir), b"wrong");
    let err = result.err().expect("should fail with wrong password");
    assert!(err.to_string().contains("wrong password"));
}

#[test]
fn test_open_missing_file() {
    let dir = TempDir::new().unwrap();
    let result = Wallet::open(&dir.path().join("nonexistent.wallet"), b"pass");
    let err = result.err().expect("should fail on missing file");
    assert!(err.to_string().contains("not found"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COIN MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_generate_coin_with_label() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(Some("my savings".to_string())).unwrap();

    assert_eq!(wc.coin, wots::keygen(&wc.seed));
    assert_eq!(wc.label.as_deref(), Some("my savings"));
    assert_eq!(w.coin_count(), 1);
}

#[test]
fn test_generate_coin_without_label() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(None).unwrap();
    assert!(wc.label.is_none());
    // Coin should still be a valid WOTS derivation
    assert_eq!(wc.coin, wots::keygen(&wc.seed));
}

#[test]
fn test_import_seed() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let seed: [u8; 32] = rand::random();
    let expected_coin = wots::keygen(&seed);

    let coin = w.import_seed(seed, Some("imported".into())).unwrap();

    assert_eq!(coin, expected_coin);
    assert_eq!(w.coin_count(), 1);
    assert_eq!(w.coins()[0].label.as_deref(), Some("imported"));
}

#[test]
fn test_import_duplicate_seed_rejected() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let seed: [u8; 32] = rand::random();
    w.import_seed(seed, None).unwrap();

    let result = w.import_seed(seed, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already in wallet"));
}

#[test]
fn test_resolve_coin_by_index() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    let c0 = w.generate(None).unwrap().coin;
    let c1 = w.generate(None).unwrap().coin;

    assert_eq!(w.resolve_coin("0").unwrap(), c0);
    assert_eq!(w.resolve_coin("1").unwrap(), c1);
}

#[test]
fn test_resolve_coin_by_hex_prefix() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    let wc = w.generate(None).unwrap();
    let coin = wc.coin;

    let hex = hex::encode(coin);
    assert_eq!(w.resolve_coin(&hex[..8]).unwrap(), coin);
    // Full hex should also work
    assert_eq!(w.resolve_coin(&hex).unwrap(), coin);
}

#[test]
fn test_resolve_coin_not_found() {
    let dir = TempDir::new().unwrap();
    let w = create_test_wallet(&dir);

    let result = w.resolve_coin("deadbeef");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("no coin matching"));
}

#[test]
fn test_resolve_coin_out_of_range_index() {
    let dir = TempDir::new().unwrap();
    let w = create_test_wallet(&dir);

    let result = w.resolve_coin("999");
    assert!(result.is_err());
}

#[test]
fn test_remove_coin() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    let coin = w.generate(None).unwrap().coin;

    w.remove_coin(&coin).unwrap();
    assert_eq!(w.coin_count(), 0);
}

#[test]
fn test_remove_nonexistent_coin() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let result = w.remove_coin(&[0xAA; 32]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_find_coin() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    let coin = w.generate(Some("findme".into())).unwrap().coin;

    let found = w.find_coin(&coin).unwrap();
    assert_eq!(found.coin, coin);
    assert_eq!(found.label.as_deref(), Some("findme"));

    // find_secret is an alias
    assert!(w.find_secret(&coin).is_some());

    // Non-existent
    assert!(w.find_coin(&[0xBB; 32]).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMMIT / REVEAL
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_commit_no_privacy_delay() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;

    let inputs = vec![c1, c2];
    let dest: [u8; 32] = rand::random();
    let outputs = vec![dest];

    let (commitment, salt) = w.prepare_commit(&inputs, &outputs, false).unwrap();

    assert_eq!(w.pending().len(), 1);
    let pending = &w.pending()[0];
    assert_eq!(pending.commitment, commitment);
    assert_eq!(pending.salt, salt);
    assert_eq!(pending.input_coin_ids, inputs);
    assert_eq!(pending.destinations, outputs);
    assert_eq!(pending.reveal_not_before, 0);
}

#[test]
fn test_prepare_commit_with_privacy_delay() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;
    let dest: [u8; 32] = rand::random();

    let (_commitment, _salt) = w.prepare_commit(&[c1, c2], &[dest], true).unwrap();

    let pending = &w.pending()[0];
    // Privacy delay should be > 0 (somewhere between now+10 and now+50)
    assert!(
        pending.reveal_not_before > 0,
        "Privacy delay should set reveal_not_before > 0"
    );
    // Should be at least 10 seconds in the future relative to created_at
    assert!(
        pending.reveal_not_before >= pending.created_at + 10,
        "Privacy delay minimum is 10 seconds"
    );
    // Should be at most 50 seconds in the future
    assert!(
        pending.reveal_not_before <= pending.created_at + 50,
        "Privacy delay maximum is 50 seconds"
    );
}

#[test]
fn test_prepare_commit_for_unknown_coin_fails() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    let c1 = w.generate(None).unwrap().coin;

    let fake_coin: [u8; 32] = rand::random();
    let result = w.prepare_commit(&[c1, fake_coin], &[rand::random()], false);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not in wallet"));
}

#[test]
fn test_sign_reveal_produces_valid_wots_sigs() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;
    let dest: [u8; 32] = rand::random();

    let (commitment_hash, _) = w.prepare_commit(&[c1, c2], &[dest], false).unwrap();

    let pending = w.find_pending(&commitment_hash).unwrap();
    let signatures = w.sign_reveal(pending);

    assert_eq!(signatures.len(), 2);
    // CHANGE: Check against byte size, not chain count
    assert_eq!(signatures[0].len(), wots::SIG_SIZE);

    // Verify the signatures are actually valid WOTS sigs
    let recomputed_commitment = compute_commitment(
        &pending.input_coin_ids,
        &pending.destinations,
        &pending.salt,
    );
    
    // CHANGE: Deserialize bytes back to WOTS chunks for verification
    let sig0 = wots::sig_from_bytes(&signatures[0]).expect("valid sig bytes");
    let sig1 = wots::sig_from_bytes(&signatures[1]).expect("valid sig bytes");

    assert!(wots::verify(&sig0, &recomputed_commitment, &c1));
    assert!(wots::verify(&sig1, &recomputed_commitment, &c2));
}

#[test]
fn test_complete_reveal() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[c1, c2], &[dest], false).unwrap();

    w.complete_reveal(&commitment).unwrap();

    assert_eq!(w.pending().len(), 0);
    assert_eq!(w.coin_count(), 0, "Spent coins should be removed");
    assert_eq!(w.history().len(), 1);

    let entry = &w.history()[0];
    assert_eq!(entry.inputs, vec![c1, c2]);
    assert_eq!(entry.outputs, vec![dest]);
    assert!(entry.timestamp > 0);
}

#[test]
fn test_complete_reveal_nonexistent_commitment() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let result = w.complete_reveal(&[0xAA; 32]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_find_pending() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[c1, c2], &[dest], false).unwrap();

    assert!(w.find_pending(&commitment).is_some());
    assert!(w.find_pending(&[0xFF; 32]).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PRIVATE SEND PLANNING
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_plan_private_send() {
    let dir = TempDir::new().unwrap();
    let w = create_test_wallet(&dir);

    let live_coins: Vec<[u8; 32]> = (0..4).map(|_| rand::random()).collect();
    let destinations: Vec<[u8; 32]> = (0..2).map(|_| rand::random()).collect();

    let plan = w.plan_private_send(&live_coins, &destinations).unwrap();

    assert_eq!(plan.len(), 2);
    assert_eq!(plan[0].0.len(), 2); // Pair 1: 2 inputs
    assert_eq!(plan[0].1.len(), 1); // Pair 1: 1 output
    assert_eq!(plan[1].0.len(), 2); // Pair 2: 2 inputs
    assert_eq!(plan[1].1.len(), 1); // Pair 2: 1 output

    // Inputs should be distinct across pairs
    let all_inputs: Vec<[u8; 32]> = plan.iter().flat_map(|(ins, _)| ins.clone()).collect();
    let unique: std::collections::HashSet<[u8; 32]> = all_inputs.iter().copied().collect();
    assert_eq!(unique.len(), 4);
}

#[test]
fn test_plan_private_send_insufficient_coins() {
    let dir = TempDir::new().unwrap();
    let w = create_test_wallet(&dir);

    let live_coins: Vec<[u8; 32]> = (0..3).map(|_| rand::random()).collect(); // only 3
    let destinations: Vec<[u8; 32]> = (0..2).map(|_| rand::random()).collect(); // needs 4

    let result = w.plan_private_send(&live_coins, &destinations);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("needs"));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CRYPTO
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_crypto_roundtrip() {
    use midstate::wallet::crypto::{encrypt, decrypt};
    let pass = b"secure";
    let data = b"some data";

    let enc = encrypt(data, pass).unwrap();
    let dec = decrypt(&enc, pass).unwrap();
    assert_eq!(data.as_slice(), &dec);
}

#[test]
fn test_crypto_wrong_password() {
    use midstate::wallet::crypto::{encrypt, decrypt};
    let enc = encrypt(b"secret", b"correct").unwrap();
    assert!(decrypt(&enc, b"wrong").is_err());
}

#[test]
fn test_crypto_truncated_data() {
    use midstate::wallet::crypto::decrypt;
    // Too short to contain salt + nonce + tag
    assert!(decrypt(&[0u8; 10], b"pass").is_err());
}

#[test]
fn test_crypto_different_encryptions_differ() {
    use midstate::wallet::crypto::encrypt;
    let enc1 = encrypt(b"data", b"pass").unwrap();
    let enc2 = encrypt(b"data", b"pass").unwrap();
    // Random salt and nonce → different ciphertexts
    assert_ne!(enc1, enc2);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_short_hex_format() {
    let bytes = [0xab; 32];
    let s = short_hex(&bytes);
    assert_eq!(s, "abababab…abab");
}

#[test]
fn test_backward_compat_no_history() {
    let data_json = r#"{"coins":[],"pending":[]}"#;
    let data: WalletData = serde_json::from_str(data_json).unwrap();
    assert!(data.history.is_empty());
}

#[test]
fn test_coinbase_seed_deterministic() {
    let mining_seed = [42u8; 32];
    let s1 = coinbase_seed(&mining_seed, 10, 0);
    let s2 = coinbase_seed(&mining_seed, 10, 0);
    assert_eq!(s1, s2);

    assert_ne!(coinbase_seed(&mining_seed, 10, 0), coinbase_seed(&mining_seed, 11, 0));
    assert_ne!(coinbase_seed(&mining_seed, 10, 0), coinbase_seed(&mining_seed, 10, 1));
}
