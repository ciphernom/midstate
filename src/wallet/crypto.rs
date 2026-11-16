use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::{bail, Result};
use argon2::Argon2;

const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;

/// Derive a 32-byte key from password + salt via Argon2id.
///
/// ```
/// # use midstate::wallet::crypto::*;
/// // Internal doc test
/// ```
/// Derive a 32-byte key from password + salt via Argon2id.
///
/// # Formal Specification
/// ```text
/// Pre:  password.len() > 0
/// Post: result is 32 bytes suitable for AES-256-GCM
/// ```
fn derive_key(password: &[u8], salt: &[u8]) -> Result<[u8; 32]> {
    let mut key = [0u8; 32];
    Argon2::default()
        .hash_password_into(password, salt, &mut key)
        .map_err(|e| anyhow::anyhow!("Argon2 KDF failed: {}", e))?;
    Ok(key)
}

/// A password's file key: the Argon2id key, and the salt it was derived
/// with.
///
/// # Reasoning
/// Argon2id is what makes a wallet file slow to guess, and it made every
/// save slow too: it ran on each one, and a wallet saves after every change
/// (a swap's journal several times a minute). The saves of one session can
/// share one key, each under a fresh random nonce: the file stays the same
/// format, `salt || nonce || ciphertext`, and opens the same way. The key
/// lives no longer than the password it comes from, which the wallet holds
/// anyway.
#[derive(Clone)]
pub struct FileKey {
    salt: [u8; SALT_LEN],
    key: [u8; 32],
}

impl FileKey {
    /// A fresh salt, and the key `password` gives with it.
    pub fn new(password: &[u8]) -> Result<Self> {
        let salt: [u8; SALT_LEN] = rand::random();
        Ok(Self { salt, key: derive_key(password, &salt)? })
    }
}

/// Encrypt plaintext with a password.
/// Output: salt (16) || nonce (12) || ciphertext+tag
pub fn encrypt(plaintext: &[u8], password: &[u8]) -> Result<Vec<u8>> {
    encrypt_with(plaintext, &FileKey::new(password)?)
}

/// [`encrypt`] with a key already derived: no Argon2id, a fresh nonce.
pub fn encrypt_with(plaintext: &[u8], key: &FileKey) -> Result<Vec<u8>> {
    let nonce_bytes: [u8; NONCE_LEN] = rand::random();
    let cipher = Aes256Gcm::new_from_slice(&key.key)
        .map_err(|e| anyhow::anyhow!("invalid key length: {}", e))?;
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("encryption failed: {}", e))?;

    let mut out = Vec::with_capacity(SALT_LEN + NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&key.salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt ciphertext with a password.
pub fn decrypt(data: &[u8], password: &[u8]) -> Result<Vec<u8>> {
    Ok(decrypt_keyed(data, password)?.0)
}

/// [`decrypt`], also handing back the file's key, for saves to reuse.
pub fn decrypt_keyed(data: &[u8], password: &[u8]) -> Result<(Vec<u8>, FileKey)> {
    if data.len() < SALT_LEN + NONCE_LEN + 16 {
        bail!("wallet file too short or corrupted");
    }

    let mut salt = [0u8; SALT_LEN];
    salt.copy_from_slice(&data[..SALT_LEN]);
    let nonce_bytes = &data[SALT_LEN..SALT_LEN + NONCE_LEN];
    let ciphertext = &data[SALT_LEN + NONCE_LEN..];

    let key = derive_key(password, &salt)?;
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| anyhow::anyhow!("invalid key length: {}", e))?;
    let nonce = Nonce::from_slice(nonce_bytes);

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("wrong password or corrupted wallet"))?;
    Ok((plaintext, FileKey { salt, key }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argon2_derivation_deterministic() {
        let pwd = b"my_secure_password";
        let salt = b"random_salt_1234";
        let key1 = derive_key(pwd, salt).unwrap();
        let key2 = derive_key(pwd, salt).unwrap();
        assert_eq!(key1, key2);
        assert_ne!(key1, [0u8; 32]);
    }

    #[test]
    fn round_trip() {
        let data = b"test wallet data";
        let password = b"hunter2";
        let encrypted = encrypt(data, password).unwrap();
        let decrypted = decrypt(&encrypted, password).unwrap();
        assert_eq!(data.as_slice(), &decrypted);
    }

    #[test]
    fn wrong_password() {
        let encrypted = encrypt(b"secret", b"correct").unwrap();
        assert!(decrypt(&encrypted, b"wrong").is_err());
    }
    #[test]
    fn truncated_data_fails() {
        let encrypted = encrypt(b"hello", b"pass").unwrap();
        assert!(decrypt(&encrypted[..5], b"pass").is_err());
    }

    #[test]
    fn empty_data_fails() {
        assert!(decrypt(&[], b"pass").is_err());
    }

    /// Saves with one key share its salt, never a nonce, and open with
    /// the password alone.
    #[test]
    fn saves_with_one_key_open_with_the_password() {
        let key = FileKey::new(b"pw").unwrap();
        let a = encrypt_with(b"first", &key).unwrap();
        let b = encrypt_with(b"second", &key).unwrap();
        assert_eq!(a[..SALT_LEN], b[..SALT_LEN]);
        assert_ne!(a[SALT_LEN..SALT_LEN + NONCE_LEN], b[SALT_LEN..SALT_LEN + NONCE_LEN]);
        assert_eq!(decrypt(&a, b"pw").unwrap(), b"first");
        let (plain, reopened) = decrypt_keyed(&b, b"pw").unwrap();
        assert_eq!(plain, b"second");
        assert_eq!(decrypt(&encrypt_with(b"third", &reopened).unwrap(), b"pw").unwrap(), b"third");
        assert!(decrypt(&a, b"other").is_err());
    }

    #[test]
    fn large_payload_round_trip() {
        let data = vec![0xABu8; 100_000];
        let password = b"strong_password_123";
        let encrypted = encrypt(&data, password).unwrap();
        let decrypted = decrypt(&encrypted, password).unwrap();
        assert_eq!(data, decrypted);
    }
}
