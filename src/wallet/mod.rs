pub mod crypto;

use crate::core::{hash, compute_commitment};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default wallet location: ~/.midstate/wallet.dat
pub fn default_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".midstate")
        .join("wallet.dat")
}

/// Short display: first 8 hex chars + "…" + last 4 hex chars
pub fn short_hex(bytes: &[u8; 32]) -> String {
    let h = hex::encode(bytes);
    format!("{}…{}", &h[..8], &h[60..])
}

/// A coin the wallet controls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletCoin {
    /// The secret preimage (knowing this = owning the coin)
    pub secret: Vec<u8>,
    /// hash(secret) — the on-chain coin ID
    pub coin: [u8; 32],
    /// Optional human label
    pub label: Option<String>,
}

/// A commit that has been submitted but not yet revealed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCommit {
    pub commitment: [u8; 32],
    pub salt: [u8; 32],
    /// Secrets for the coins being spent (needed for reveal)
    pub input_secrets: Vec<Vec<u8>>,
    /// Destination coins for the reveal
    pub destinations: Vec<[u8; 32]>,
    /// Unix timestamp when committed
    pub created_at: u64,
}

/// Record of a completed transaction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Coins that were spent
    pub inputs: Vec<[u8; 32]>,
    /// Coins that were created
    pub outputs: Vec<[u8; 32]>,
    /// Unix timestamp when completed
    pub timestamp: u64,
}

/// The wallet file contents (serialized to JSON, then encrypted).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletData {
    pub coins: Vec<WalletCoin>,
    pub pending: Vec<PendingCommit>,
    /// Transaction history (backward-compatible: missing = empty)
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

impl WalletData {
    fn empty() -> Self {
        Self {
            coins: Vec::new(),
            pending: Vec::new(),
            history: Vec::new(),
        }
    }
}

pub struct Wallet {
    path: PathBuf,
    password: Vec<u8>,
    pub data: WalletData,
}

impl Wallet {
    /// Create a new wallet file. Fails if the file already exists.
    pub fn create(path: &Path, password: &[u8]) -> Result<Self> {
        if path.exists() {
            bail!("wallet file already exists: {}", path.display());
        }

        let wallet = Self {
            path: path.to_path_buf(),
            password: password.to_vec(),
            data: WalletData::empty(),
        };
        wallet.save()?;
        Ok(wallet)
    }

    /// Open an existing wallet.
    pub fn open(path: &Path, password: &[u8]) -> Result<Self> {
        if !path.exists() {
            bail!("wallet file not found: {}", path.display());
        }

        let encrypted = std::fs::read(path)?;
        let plaintext = crypto::decrypt(&encrypted, password)?;
        let data: WalletData = serde_json::from_slice(&plaintext)?;

        Ok(Self {
            path: path.to_path_buf(),
            password: password.to_vec(),
            data,
        })
    }

    /// Write current state to disk (encrypted).
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let plaintext = serde_json::to_vec(&self.data)?;
        let encrypted = crypto::encrypt(&plaintext, &self.password)?;
        std::fs::write(&self.path, encrypted)?;
        Ok(())
    }

    /// Generate a new random coin and add it to the wallet.
    pub fn generate(&mut self, label: Option<String>) -> Result<&WalletCoin> {
        let secret: [u8; 32] = rand::random();
        let coin = hash(&secret);

        self.data.coins.push(WalletCoin {
            secret: secret.to_vec(),
            coin,
            label,
        });
        self.save()?;
        Ok(self.data.coins.last().unwrap())
    }

    /// Import an existing secret.
    pub fn import_secret(&mut self, secret: Vec<u8>, label: Option<String>) -> Result<[u8; 32]> {
        let coin = hash(&secret);

        if self.data.coins.iter().any(|c| c.coin == coin) {
            bail!("coin already in wallet");
        }

        self.data.coins.push(WalletCoin {
            secret,
            coin,
            label,
        });
        self.save()?;
        Ok(coin)
    }

    /// Look up a coin's secret by its coin ID.
    pub fn find_secret(&self, coin: &[u8; 32]) -> Option<&WalletCoin> {
        self.data.coins.iter().find(|c| &c.coin == coin)
    }

    /// Resolve a coin reference: numeric index ("0", "2") or hex prefix ("f8de45").
    pub fn resolve_coin(&self, reference: &str) -> Result<[u8; 32]> {
        // Try as index first
        if let Ok(idx) = reference.parse::<usize>() {
            if idx < self.data.coins.len() {
                return Ok(self.data.coins[idx].coin);
            }
        }

        // Try as hex prefix
        let reference_lower = reference.to_lowercase();
        let matches: Vec<_> = self
            .data
            .coins
            .iter()
            .filter(|c| hex::encode(c.coin).starts_with(&reference_lower))
            .collect();

        match matches.len() {
            0 => bail!("no coin matching '{}'", reference),
            1 => Ok(matches[0].coin),
            n => bail!(
                "'{}' is ambiguous ({} matches) — use more characters",
                reference,
                n
            ),
        }
    }

    /// Prepare a commit: picks coins from the wallet, computes the commitment,
    /// and stores the pending state so we can reveal later.
    pub fn prepare_commit(
        &mut self,
        input_coin_ids: &[[u8; 32]],
        destinations: &[[u8; 32]],
    ) -> Result<([u8; 32], [u8; 32])> {
        // Verify we own all input coins
        let mut input_secrets = Vec::new();
        for coin_id in input_coin_ids {
            let wc = self
                .find_secret(coin_id)
                .ok_or_else(|| anyhow::anyhow!("coin {} not in wallet", short_hex(coin_id)))?;
            input_secrets.push(wc.secret.clone());
        }

        let salt: [u8; 32] = rand::random();
        let commitment = compute_commitment(input_coin_ids, destinations, &salt);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.data.pending.push(PendingCommit {
            commitment,
            salt,
            input_secrets,
            destinations: destinations.to_vec(),
            created_at: now,
        });
        self.save()?;

        Ok((commitment, salt))
    }

    /// Find a pending commit by its commitment hash.
    pub fn find_pending(&self, commitment: &[u8; 32]) -> Option<&PendingCommit> {
        self.data.pending.iter().find(|p| &p.commitment == commitment)
    }

    /// Get all pending commits.
    pub fn pending(&self) -> &[PendingCommit] {
        &self.data.pending
    }

    /// Remove a pending commit after successful reveal.
    /// Also removes the spent coins and records history.
    pub fn complete_reveal(&mut self, commitment: &[u8; 32]) -> Result<()> {
        let pending = self
            .data
            .pending
            .iter()
            .find(|p| &p.commitment == commitment)
            .ok_or_else(|| anyhow::anyhow!("pending commit not found"))?
            .clone();

        // Compute spent coin IDs
        let spent_coins: Vec<[u8; 32]> = pending
            .input_secrets
            .iter()
            .map(|s| hash(s))
            .collect();

        // Remove spent coins from wallet
        self.data.coins.retain(|c| !spent_coins.contains(&c.coin));

        // Record in history
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.data.history.push(HistoryEntry {
            inputs: spent_coins,
            outputs: pending.destinations.clone(),
            timestamp: now,
        });

        // Remove the pending entry
        self.data.pending.retain(|p| &p.commitment != commitment);

        self.save()?;
        Ok(())
    }

    /// Transaction history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.data.history
    }

    /// Remove a coin from the wallet (e.g. if spent externally).
    pub fn remove_coin(&mut self, coin: &[u8; 32]) -> Result<()> {
        let before = self.data.coins.len();
        self.data.coins.retain(|c| &c.coin != coin);
        if self.data.coins.len() == before {
            bail!("coin not found in wallet");
        }
        self.save()?;
        Ok(())
    }

    /// Number of coins in the wallet.
    pub fn coin_count(&self) -> usize {
        self.data.coins.len()
    }

    /// All coins.
    pub fn coins(&self) -> &[WalletCoin] {
        &self.data.coins
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn create_and_reopen() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        w.generate(Some("test".into())).unwrap();
        assert_eq!(w.coin_count(), 1);

        let w2 = Wallet::open(&path, b"pass").unwrap();
        assert_eq!(w2.coin_count(), 1);
        assert_eq!(w2.coins()[0].label.as_deref(), Some("test"));
    }

    #[test]
    fn commit_reveal_records_history() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let coin_id = {
            let wc = w.generate(None).unwrap();
            wc.coin
        };

        let dest: [u8; 32] = rand::random();
        let (commitment, _salt) = w.prepare_commit(&[coin_id], &[dest]).unwrap();

        assert_eq!(w.pending().len(), 1);

        w.complete_reveal(&commitment).unwrap();

        assert_eq!(w.pending().len(), 0);
        assert_eq!(w.coin_count(), 0);
        assert_eq!(w.history().len(), 1);
        assert_eq!(w.history()[0].inputs, vec![coin_id]);
        assert_eq!(w.history()[0].outputs, vec![dest]);
    }

    #[test]
    fn resolve_by_index() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c0 = w.generate(None).unwrap().coin;
        let c1 = w.generate(None).unwrap().coin;

        assert_eq!(w.resolve_coin("0").unwrap(), c0);
        assert_eq!(w.resolve_coin("1").unwrap(), c1);
        assert!(w.resolve_coin("99").is_err());
    }

    #[test]
    fn resolve_by_hex_prefix() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let wc = w.generate(None).unwrap();
        let coin = wc.coin; // copy to drop borrow
        let full_hex = hex::encode(coin);
        let prefix = &full_hex[..10];

        let resolved = w.resolve_coin(prefix).unwrap();
        assert_eq!(resolved, coin);
    }

    #[test]
    fn short_hex_format() {
        let bytes = [0xab; 32];
        let s = short_hex(&bytes);
        assert_eq!(s, "abababab…abab");
    }

    #[test]
    fn backward_compat_no_history() {
        // Old wallet files won't have the history field
        let data_json = r#"{"coins":[],"pending":[]}"#;
        let data: WalletData = serde_json::from_str(data_json).unwrap();
        assert!(data.history.is_empty());
    }
}
