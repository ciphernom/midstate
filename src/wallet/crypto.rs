use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

const KDF_ITERATIONS: u32 = 100_000;
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;

/// Derive a 32-byte key from password + salt via iterated SHA-256.
fn derive_key(password: &[u8], salt: &[u8]) -> [u8; 32] {
    let mut key = Sha256::digest([password, salt].concat()).into();
    for _ in 1..KDF_ITERATIONS {
        key = Sha256::digest(key).into();
    }
    key
}

/// Encrypt plaintext with a password.
/// Output: salt (16) || nonce (12) || ciphertext+tag
pub fn encrypt(plaintext: &[u8], password: &[u8]) -> Result<Vec<u8>> {
    let salt: [u8; SALT_LEN] = rand::random();
    let nonce_bytes: [u8; NONCE_LEN] = rand::random();

    let key = derive_key(password, &salt);
    let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("encryption failed: {}", e))?;

    let mut out = Vec::with_capacity(SALT_LEN + NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt ciphertext with a password.
pub fn decrypt(data: &[u8], password: &[u8]) -> Result<Vec<u8>> {
    if data.len() < SALT_LEN + NONCE_LEN + 16 {
        bail!("wallet file too short or corrupted");
    }

    let salt = &data[..SALT_LEN];
    let nonce_bytes = &data[SALT_LEN..SALT_LEN + NONCE_LEN];
    let ciphertext = &data[SALT_LEN + NONCE_LEN..];

    let key = derive_key(password, salt);
    let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("wrong password or corrupted wallet"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
