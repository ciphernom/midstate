pub mod crypto;

use crate::core::{hash_concat, compute_commitment, wots};
// [MSS-ENABLE] Import MSS modules
use crate::core::mss::{self, MssKeypair};
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

/// A coin the wallet controls (WOTS Legacy).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletCoin {
    pub seed: [u8; 32],
    pub coin: [u8; 32],
    pub label: Option<String>,
}

/// A commit that has been submitted but not yet revealed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCommit {
    pub commitment: [u8; 32],
    pub salt: [u8; 32],
    // [MSS-ENABLE] Removed 'input_seeds'. We look up keys at sign time.
    pub input_coin_ids: Vec<[u8; 32]>,
    pub destinations: Vec<[u8; 32]>,
    pub created_at: u64,
    #[serde(default)]
    pub reveal_not_before: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub inputs: Vec<[u8; 32]>,
    pub outputs: Vec<[u8; 32]>,
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletData {
    pub coins: Vec<WalletCoin>,
    // [MSS-ENABLE] Store MSS Trees
    #[serde(default)]
    pub mss_keys: Vec<MssKeypair>, 
    pub pending: Vec<PendingCommit>,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

impl WalletData {
    fn empty() -> Self {
        Self {
            coins: Vec::new(),
            mss_keys: Vec::new(),
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

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let plaintext = serde_json::to_vec(&self.data)?;
        let encrypted = crypto::encrypt(&plaintext, &self.password)?;
        std::fs::write(&self.path, encrypted)?;
        Ok(())
    }

    /// Generate a new random WOTS coin (Legacy).
    pub fn generate(&mut self, label: Option<String>) -> Result<&WalletCoin> {
        let seed: [u8; 32] = rand::random();
        let coin = wots::keygen(&seed);

        self.data.coins.push(WalletCoin { seed, coin, label });
        self.save()?;
        Ok(self.data.coins.last().unwrap())
    }

    /// [MSS-ENABLE] Generate a new MSS Tree (Reusable Address).
    pub fn generate_mss(&mut self, height: u32, _label: Option<String>) -> Result<[u8; 32]> {
        let seed: [u8; 32] = rand::random();
        // Generate the full tree (this may take seconds for height > 14)
        let keypair = mss::keygen(&seed, height)?;
        let root = keypair.master_pk;
        
        self.data.mss_keys.push(keypair);
        self.save()?;
        Ok(root)
    }

    pub fn import_seed(&mut self, seed: [u8; 32], label: Option<String>) -> Result<[u8; 32]> {
        let coin = wots::keygen(&seed);
        if self.data.coins.iter().any(|c| c.coin == coin) {
            bail!("coin already in wallet");
        }
        self.data.coins.push(WalletCoin { seed, coin, label });
        self.save()?;
        Ok(coin)
    }

    pub fn find_coin(&self, coin: &[u8; 32]) -> Option<&WalletCoin> {
        self.data.coins.iter().find(|c| &c.coin == coin)
    }

    // [MSS-ENABLE] Find an MSS key by its root (coin ID)
    pub fn find_mss(&self, coin: &[u8; 32]) -> Option<&MssKeypair> {
        self.data.mss_keys.iter().find(|k| &k.master_pk == coin)
    }

    pub fn resolve_coin(&self, reference: &str) -> Result<[u8; 32]> {
        if let Ok(idx) = reference.parse::<usize>() {
            if idx < self.data.coins.len() {
                return Ok(self.data.coins[idx].coin);
            }
        }
        let reference_lower = reference.to_lowercase();
        
        // Search WOTS coins
        for c in &self.data.coins {
            if hex::encode(c.coin).starts_with(&reference_lower) {
                return Ok(c.coin);
            }
        }
        // Search MSS keys
        for k in &self.data.mss_keys {
            if hex::encode(k.master_pk).starts_with(&reference_lower) {
                return Ok(k.master_pk);
            }
        }
        bail!("no matching coin found");
    }

    pub fn prepare_commit(
        &mut self,
        input_coin_ids: &[[u8; 32]],
        destinations: &[[u8; 32]],
        privacy_delay: bool,
    ) -> Result<([u8; 32], [u8; 32])> {
        // [MSS-ENABLE] Verify we own all inputs (WOTS or MSS)
        for coin_id in input_coin_ids {
            if self.find_coin(coin_id).is_none() && self.find_mss(coin_id).is_none() {
                bail!("coin {} not in wallet", short_hex(coin_id));
            }
        }

        let salt: [u8; 32] = rand::random();
        let commitment = compute_commitment(input_coin_ids, destinations, &salt);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let reveal_not_before = if privacy_delay {
            now + 10 + (rand::random::<u64>() % 41)
        } else {
            0
        };

        self.data.pending.push(PendingCommit {
            commitment,
            salt,
            input_coin_ids: input_coin_ids.to_vec(),
            destinations: destinations.to_vec(),
            created_at: now,
            reveal_not_before,
        });
        self.save()?;

        Ok((commitment, salt))
    }

    /// [MSS-ENABLE] Sign a reveal. Now requires &mut self to update MSS state.
    pub fn sign_reveal(&mut self, pending: &PendingCommit) -> Result<Vec<Vec<u8>>> {
        let commitment = compute_commitment(
            &pending.input_coin_ids,
            &pending.destinations,
            &pending.salt,
        );
        
        let mut signatures = Vec::new();

        for coin_id in &pending.input_coin_ids {
            // 1. Try WOTS
            if let Some(wc) = self.find_coin(coin_id) {
                let sig = wots::sign(&wc.seed, &commitment);
                signatures.push(wots::sig_to_bytes(&sig));
            } 
            // 2. Try MSS
            else if let Some(pos) = self.data.mss_keys.iter().position(|k| k.master_pk == *coin_id) {
                // We need mutable access to increment the leaf index
                let keypair = &mut self.data.mss_keys[pos];
                
                if keypair.remaining() == 0 {
                    bail!("MSS key {} exhausted", short_hex(coin_id));
                }
                
                // Sign and advance state
                let sig = keypair.sign(&commitment)?;
                signatures.push(sig.to_bytes());
            } else {
                bail!("key for {} not found during signing", short_hex(coin_id));
            }
        }

        // Save wallet because MSS indices might have incremented
        self.save()?;
        Ok(signatures)
    }

    pub fn find_pending(&self, commitment: &[u8; 32]) -> Option<&PendingCommit> {
        self.data.pending.iter().find(|p| &p.commitment == commitment)
    }

    pub fn pending(&self) -> &[PendingCommit] {
        &self.data.pending
    }

    pub fn complete_reveal(&mut self, commitment: &[u8; 32]) -> Result<()> {
        let pending = self
            .data
            .pending
            .iter()
            .find(|p| &p.commitment == commitment)
            .ok_or_else(|| anyhow::anyhow!("pending commit not found"))?
            .clone();

        let spent_coins = pending.input_coin_ids.clone();
        
        // Remove spent WOTS coins
        self.data.coins.retain(|c| !spent_coins.contains(&c.coin));
        
        // [MSS-ENABLE] Do NOT remove MSS keys. They are reusable (until exhausted).
        // However, if an MSS key is fully exhausted, you *could* remove it, 
        // but it's safer to keep it for history/verification.

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.data.history.push(HistoryEntry {
            inputs: spent_coins,
            outputs: pending.destinations.clone(),
            timestamp: now,
        });

        self.data.pending.retain(|p| &p.commitment != commitment);
        self.save()?;
        Ok(())
    }

    pub fn history(&self) -> &[HistoryEntry] {
        &self.data.history
    }

    pub fn coin_count(&self) -> usize {
        self.data.coins.len() + self.data.mss_keys.len()
    }

    pub fn coins(&self) -> &[WalletCoin] {
        &self.data.coins
    }
    
    // [MSS-ENABLE]
    pub fn mss_keys(&self) -> &[MssKeypair] {
        &self.data.mss_keys
    }

    pub fn plan_private_send(
        &self,
        live_coins: &[[u8; 32]],
        destinations: &[[u8; 32]],
    ) -> Result<Vec<(Vec<[u8; 32]>, Vec<[u8; 32]>)>> {
        let needed = destinations.len() * 2; 
        if live_coins.len() < needed {
            bail!("private send needs {} live coins", needed);
        }

        let mut pairs = Vec::with_capacity(destinations.len());
        for (i, dest) in destinations.iter().enumerate() {
            let inputs = vec![live_coins[i * 2], live_coins[i * 2 + 1]];
            let outputs = vec![*dest];
            pairs.push((inputs, outputs));
        }
        Ok(pairs)
    }
}

pub fn coinbase_seed(mining_seed: &[u8; 32], height: u64, index: u64) -> [u8; 32] {
    let height_key = hash_concat(mining_seed, &height.to_le_bytes());
    hash_concat(&height_key, &index.to_le_bytes())
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
        // Need 2 inputs for fee (inputs > outputs)
        let coin_id2 = {
            let wc = w.generate(None).unwrap();
            wc.coin
        };

        let dest: [u8; 32] = wots::keygen(&rand::random());
        let (commitment, _salt) = w.prepare_commit(&[coin_id, coin_id2], &[dest], false).unwrap();

        assert_eq!(w.pending().len(), 1);

        w.complete_reveal(&commitment).unwrap();

        assert_eq!(w.pending().len(), 0);
        assert_eq!(w.coin_count(), 0);
        assert_eq!(w.history().len(), 1);
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
    }

    #[test]
    fn short_hex_format() {
        let bytes = [0xab; 32];
        let s = short_hex(&bytes);
        assert_eq!(s, "abababab…abab");
    }

    #[test]
    fn backward_compat_no_history() {
        let data_json = r#"{"coins":[],"pending":[]}"#;
        let data: WalletData = serde_json::from_str(data_json).unwrap();
        assert!(data.history.is_empty());
    }
}
