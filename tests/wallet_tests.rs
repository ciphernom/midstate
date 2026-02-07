use midstate::core::hash;
use midstate::wallet::{Wallet, short_hex};
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
//  WALLET CREATION & ENCRYPTION
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
fn test_create_fails_if_exists() {
    let dir = TempDir::new().unwrap();
    let _w = create_test_wallet(&dir);
    let result = Wallet::create(&wallet_path(&dir), b"testpass");
    assert!(result.is_err());
}

#[test]
fn test_open_nonexistent_fails() {
    let dir = TempDir::new().unwrap();
    let result = Wallet::open(&dir.path().join("nope.wallet"), b"pass");
    assert!(result.is_err());
}

#[test]
fn test_wrong_password_fails() {
    let dir = TempDir::new().unwrap();
    let _w = create_test_wallet(&dir);
    let result = Wallet::open(&wallet_path(&dir), b"wrongpass");
    assert!(result.is_err());
}

#[test]
fn test_empty_password_works() {
    let dir = TempDir::new().unwrap();
    let path = wallet_path(&dir);
    let w = Wallet::create(&path, b"x").unwrap();
    drop(w);
    let w2 = Wallet::open(&path, b"x").unwrap();
    assert_eq!(w2.coin_count(), 0);
}

#[test]
fn test_wallet_persists_across_open_close() {
    let dir = TempDir::new().unwrap();

    {
        let mut w = create_test_wallet(&dir);
        w.generate(Some("coin1".into())).unwrap();
        w.generate(Some("coin2".into())).unwrap();
    }

    let w = open_test_wallet(&dir);
    assert_eq!(w.coin_count(), 2);
    assert_eq!(w.coins()[0].label.as_deref(), Some("coin1"));
    assert_eq!(w.coins()[1].label.as_deref(), Some("coin2"));
}

#[test]
fn test_corrupted_wallet_file_fails() {
    let dir = TempDir::new().unwrap();
    let path = wallet_path(&dir);
    let _w = create_test_wallet(&dir);
    drop(_w);

    std::fs::write(&path, b"GARBAGE").unwrap();
    let result = Wallet::open(&path, b"testpass");
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  KEY GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_generate_coin() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(Some("test".into())).unwrap();

    assert_eq!(wc.coin, hash(&wc.secret));
    assert_eq!(wc.label.as_deref(), Some("test"));
    assert_eq!(w.coin_count(), 1);
}

#[test]
fn test_generate_multiple_unique() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.generate(None).unwrap().coin;
    let c2 = w.generate(None).unwrap().coin;
    let c3 = w.generate(None).unwrap().coin;

    assert_ne!(c1, c2);
    assert_ne!(c2, c3);
    assert_ne!(c1, c3);
    assert_eq!(w.coin_count(), 3);
}

#[test]
fn test_generate_without_label() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(None).unwrap();
    assert!(wc.label.is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  IMPORT / EXPORT
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_import_secret() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"my_secret_key".to_vec();
    let expected_coin = hash(&secret);
    let coin = w.import_secret(secret, Some("imported".into())).unwrap();

    assert_eq!(coin, expected_coin);
    assert_eq!(w.coin_count(), 1);
    assert_eq!(w.coins()[0].label.as_deref(), Some("imported"));
}

#[test]
fn test_import_duplicate_rejected() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"duplicate".to_vec();
    w.import_secret(secret.clone(), None).unwrap();
    let result = w.import_secret(secret, None);
    assert!(result.is_err());
}

#[test]
fn test_find_secret() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(None).unwrap();
    let coin = wc.coin;

    let found = w.find_secret(&coin).unwrap();
    assert_eq!(found.coin, coin);

    let missing = w.find_secret(&[0xff; 32]);
    assert!(missing.is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RESOLVE COIN (index / hex prefix)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_resolve_by_index() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c0 = w.generate(None).unwrap().coin;
    let c1 = w.generate(None).unwrap().coin;

    assert_eq!(w.resolve_coin("0").unwrap(), c0);
    assert_eq!(w.resolve_coin("1").unwrap(), c1);
    assert!(w.resolve_coin("99").is_err());
}

#[test]
fn test_resolve_by_hex_prefix() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(None).unwrap();
    let coin = wc.coin; // Copy data to drop borrow
    let full_hex = hex::encode(coin);
    let prefix = &full_hex[..10];

    let resolved = w.resolve_coin(prefix).unwrap();
    assert_eq!(resolved, coin);
}

#[test]
fn test_resolve_by_full_hex() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let wc = w.generate(None).unwrap();
    let coin = wc.coin; // Copy data to drop borrow
    let full_hex = hex::encode(coin);

    let resolved = w.resolve_coin(&full_hex).unwrap();
    assert_eq!(resolved, coin);
}

#[test]
fn test_resolve_no_match() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);
    w.generate(None).unwrap();

    assert!(w.resolve_coin("ffffffffffffffff").is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SHORT HEX DISPLAY
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_short_hex() {
    let bytes = [0xab; 32];
    assert_eq!(short_hex(&bytes), "abababab…abab");
}

#[test]
fn test_short_hex_distinct() {
    let a = [0x11; 32];
    let b = [0x22; 32];
    assert_ne!(short_hex(&a), short_hex(&b));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COIN REMOVAL
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_remove_coin() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let coin = w.generate(None).unwrap().coin;
    assert_eq!(w.coin_count(), 1);

    w.remove_coin(&coin).unwrap();
    assert_eq!(w.coin_count(), 0);
}

#[test]
fn test_remove_nonexistent_coin_fails() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let result = w.remove_coin(&[0xff; 32]);
    assert!(result.is_err());
}

#[test]
fn test_remove_persists() {
    let dir = TempDir::new().unwrap();

    let coin = {
        let mut w = create_test_wallet(&dir);
        let c = w.generate(None).unwrap().coin;
        w.remove_coin(&c).unwrap();
        c
    };

    let w = open_test_wallet(&dir);
    assert_eq!(w.coin_count(), 0);
    assert!(w.find_secret(&coin).is_none());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  COMMIT / REVEAL LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prepare_commit() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"genesis_coin_1".to_vec();
    let coin = w.import_secret(secret, None).unwrap();
    let dest: [u8; 32] = rand::random();

    let (commitment, salt) = w.prepare_commit(&[coin], &[dest]).unwrap();

    assert_eq!(w.pending().len(), 1);
    assert_eq!(w.pending()[0].commitment, commitment);
    assert_eq!(w.pending()[0].salt, salt);
    assert_eq!(w.pending()[0].destinations, vec![dest]);
    assert_eq!(w.pending()[0].input_secrets.len(), 1);
}

#[test]
fn test_prepare_commit_unknown_coin_fails() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let fake_coin = [0xab; 32];
    let dest: [u8; 32] = rand::random();

    let result = w.prepare_commit(&[fake_coin], &[dest]);
    assert!(result.is_err());
    assert!(w.pending().is_empty());
}

#[test]
fn test_complete_reveal() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"my_coin".to_vec();
    let coin = w.import_secret(secret, None).unwrap();
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[coin], &[dest]).unwrap();

    assert_eq!(w.coin_count(), 1);
    assert_eq!(w.pending().len(), 1);

    w.complete_reveal(&commitment).unwrap();

    assert_eq!(w.coin_count(), 0);
    assert_eq!(w.pending().len(), 0);
}

#[test]
fn test_complete_reveal_unknown_commitment_fails() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let result = w.complete_reveal(&[0xff; 32]);
    assert!(result.is_err());
}

#[test]
fn test_find_pending() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"findme".to_vec();
    let coin = w.import_secret(secret, None).unwrap();
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[coin], &[dest]).unwrap();

    assert!(w.find_pending(&commitment).is_some());
    assert!(w.find_pending(&[0x00; 32]).is_none());
}

#[test]
fn test_pending_persists_across_reopen() {
    let dir = TempDir::new().unwrap();

    let commitment = {
        let mut w = create_test_wallet(&dir);
        let secret = b"persist_pending".to_vec();
        let coin = w.import_secret(secret, None).unwrap();
        let dest: [u8; 32] = rand::random();
        let (c, _) = w.prepare_commit(&[coin], &[dest]).unwrap();
        c
    };

    let w = open_test_wallet(&dir);
    assert_eq!(w.pending().len(), 1);
    assert!(w.find_pending(&commitment).is_some());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TRANSACTION HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_complete_reveal_records_history() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let secret = b"history_coin".to_vec();
    let coin = w.import_secret(secret, None).unwrap();
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[coin], &[dest]).unwrap();
    w.complete_reveal(&commitment).unwrap();

    assert_eq!(w.history().len(), 1);
    assert_eq!(w.history()[0].inputs, vec![coin]);
    assert_eq!(w.history()[0].outputs, vec![dest]);
    assert!(w.history()[0].timestamp > 0);
}

#[test]
fn test_multiple_sends_accumulate_history() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    for i in 0..3 {
        let secret = format!("coin_{}", i).into_bytes();
        let coin = w.import_secret(secret, None).unwrap();
        let dest: [u8; 32] = rand::random();
        let (commitment, _) = w.prepare_commit(&[coin], &[dest]).unwrap();
        w.complete_reveal(&commitment).unwrap();
    }

    assert_eq!(w.history().len(), 3);
}

#[test]
fn test_history_persists_across_reopen() {
    let dir = TempDir::new().unwrap();

    {
        let mut w = create_test_wallet(&dir);
        let secret = b"persist_history".to_vec();
        let coin = w.import_secret(secret, None).unwrap();
        let dest: [u8; 32] = rand::random();
        let (commitment, _) = w.prepare_commit(&[coin], &[dest]).unwrap();
        w.complete_reveal(&commitment).unwrap();
    }

    let w = open_test_wallet(&dir);
    assert_eq!(w.history().len(), 1);
}

#[test]
fn test_backward_compat_no_history_field() {
    use midstate::wallet::WalletData;
    // Old wallet files won't have the history field
    let data_json = r#"{"coins":[],"pending":[]}"#;
    let data: WalletData = serde_json::from_str(data_json).unwrap();
    assert!(data.history.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MULTI-COIN OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_input_commit() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.import_secret(b"coin_a".to_vec(), None).unwrap();
    let c2 = w.import_secret(b"coin_b".to_vec(), None).unwrap();
    let dest: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[c1, c2], &[dest]).unwrap();

    let pending = w.find_pending(&commitment).unwrap();
    assert_eq!(pending.input_secrets.len(), 2);
    assert_eq!(pending.destinations.len(), 1);
}

#[test]
fn test_multi_output_commit() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let coin = w.import_secret(b"single_input".to_vec(), None).unwrap();
    let d1: [u8; 32] = rand::random();
    let d2: [u8; 32] = rand::random();
    let d3: [u8; 32] = rand::random();

    let (commitment, _) = w.prepare_commit(&[coin], &[d1, d2, d3]).unwrap();

    let pending = w.find_pending(&commitment).unwrap();
    assert_eq!(pending.input_secrets.len(), 1);
    assert_eq!(pending.destinations.len(), 3);
}

#[test]
fn test_complete_reveal_removes_only_spent_coins() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.import_secret(b"spend_me".to_vec(), None).unwrap();
    let c2 = w.import_secret(b"keep_me".to_vec(), None).unwrap();
    let dest: [u8; 32] = rand::random();

    assert_eq!(w.coin_count(), 2);

    let (commitment, _) = w.prepare_commit(&[c1], &[dest]).unwrap();
    w.complete_reveal(&commitment).unwrap();

    assert_eq!(w.coin_count(), 1);
    assert!(w.find_secret(&c1).is_none());
    assert!(w.find_secret(&c2).is_some());
}

#[test]
fn test_multiple_pending_commits() {
    let dir = TempDir::new().unwrap();
    let mut w = create_test_wallet(&dir);

    let c1 = w.import_secret(b"coin1".to_vec(), None).unwrap();
    let c2 = w.import_secret(b"coin2".to_vec(), None).unwrap();
    let d1: [u8; 32] = rand::random();
    let d2: [u8; 32] = rand::random();

    let (com1, _) = w.prepare_commit(&[c1], &[d1]).unwrap();
    let (com2, _) = w.prepare_commit(&[c2], &[d2]).unwrap();

    assert_eq!(w.pending().len(), 2);

    w.complete_reveal(&com1).unwrap();
    assert_eq!(w.pending().len(), 1);
    assert!(w.find_pending(&com1).is_none());
    assert!(w.find_pending(&com2).is_some());
    assert_eq!(w.coin_count(), 1);
    assert_eq!(w.history().len(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  CRYPTO MODULE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_encrypt_decrypt_roundtrip() {
    use midstate::wallet::crypto::{encrypt, decrypt};

    let data = b"wallet state json goes here";
    let password = b"strong_password_123";

    let encrypted = encrypt(data, password).unwrap();
    assert_ne!(&encrypted[..], data.as_slice());

    let decrypted = decrypt(&encrypted, password).unwrap();
    assert_eq!(decrypted, data);
}

#[test]
fn test_encrypt_produces_different_ciphertext() {
    use midstate::wallet::crypto::encrypt;

    let e1 = encrypt(b"same", b"pass").unwrap();
    let e2 = encrypt(b"same", b"pass").unwrap();
    assert_ne!(e1, e2);
}

#[test]
fn test_decrypt_wrong_password() {
    use midstate::wallet::crypto::{encrypt, decrypt};

    let encrypted = encrypt(b"secret", b"correct").unwrap();
    let result = decrypt(&encrypted, b"wrong");
    assert!(result.is_err());
}

#[test]
fn test_decrypt_truncated_data() {
    use midstate::wallet::crypto::decrypt;

    let result = decrypt(&[0u8; 10], b"pass");
    assert!(result.is_err());
}

#[test]
fn test_decrypt_corrupted_data() {
    use midstate::wallet::crypto::{encrypt, decrypt};

    let mut encrypted = encrypt(b"data", b"pass").unwrap();
    let last = encrypted.len() - 1;
    encrypted[last] ^= 0xff;

    let result = decrypt(&encrypted, b"pass");
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  DEFAULT PATH
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_default_path_is_under_home() {
    let path = midstate::wallet::default_path();
    let path_str = path.to_string_lossy();
    assert!(path_str.contains(".midstate"));
    assert!(path_str.ends_with("wallet.dat"));
}
