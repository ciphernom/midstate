pub mod crypto;

use crate::core::{hash_concat, compute_commitment, compute_coin_id, decompose_value, wots, OutputData, InputReveal};
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

/// A receiving key (seed + public key). No value assigned yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletKey {
    pub seed: [u8; 32],
    pub owner_pk: [u8; 32],
    pub label: Option<String>,
}

/// A coin the wallet controls, with known value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletCoin {
    pub seed: [u8; 32],
    pub owner_pk: [u8; 32],
    pub value: u64,
    pub salt: [u8; 32],
    pub coin_id: [u8; 32],
    pub label: Option<String>,
}

/// A commit that has been submitted but not yet revealed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCommit {
    pub commitment: [u8; 32],
    pub salt: [u8; 32],
    pub input_coin_ids: Vec<[u8; 32]>,
    /// Full output data needed for the reveal transaction.
    pub outputs: Vec<OutputData>,
    /// (output_index, wots_seed) for change outputs we control.
    pub change_seeds: Vec<(usize, [u8; 32])>,
    pub created_at: u64,
    #[serde(default)]
    pub reveal_not_before: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub inputs: Vec<[u8; 32]>,
    pub outputs: Vec<[u8; 32]>,
    pub fee: u64,
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletData {
    /// Receiving keys (not yet associated with a value).
    #[serde(default)]
    pub keys: Vec<WalletKey>,
    /// Coins with known values that we can spend.
    pub coins: Vec<WalletCoin>,
    #[serde(default)]
    pub mss_keys: Vec<MssKeypair>,
    pub pending: Vec<PendingCommit>,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
}

impl WalletData {
    fn empty() -> Self {
        Self {
            keys: Vec::new(),
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

    // ── Key generation ──────────────────────────────────────────────────────

    /// Generate a new receiving key. Returns the owner_pk to share with the sender.
    pub fn generate_key(&mut self, label: Option<String>) -> Result<[u8; 32]> {
        let seed: [u8; 32] = rand::random();
        let owner_pk = wots::keygen(&seed);
        self.data.keys.push(WalletKey { seed, owner_pk, label });
        self.save()?;
        Ok(owner_pk)
    }

    /// Generate a new MSS tree (reusable address).
    pub fn generate_mss(&mut self, height: u32, _label: Option<String>) -> Result<[u8; 32]> {
        let seed: [u8; 32] = rand::random();
        let keypair = mss::keygen(&seed, height)?;
        let root = keypair.master_pk;
        self.data.mss_keys.push(keypair);
        self.save()?;
        Ok(root)
    }

    // ── Coin management ─────────────────────────────────────────────────────

    /// Import a coin with known seed, value, and salt.
    pub fn import_coin(
        &mut self,
        seed: [u8; 32],
        value: u64,
        salt: [u8; 32],
        label: Option<String>,
    ) -> Result<[u8; 32]> {
        let owner_pk = wots::keygen(&seed);
        let coin_id = compute_coin_id(&owner_pk, value, &salt);
        if self.data.coins.iter().any(|c| c.coin_id == coin_id) {
            bail!("coin already in wallet");
        }
        self.data.coins.push(WalletCoin {
            seed, owner_pk, value, salt, coin_id, label,
        });
        // Remove matching key from unused keys if present
        self.data.keys.retain(|k| k.owner_pk != owner_pk);
        self.save()?;
        Ok(coin_id)
    }

    /// Find a coin by coin_id.
    pub fn find_coin(&self, coin_id: &[u8; 32]) -> Option<&WalletCoin> {
        self.data.coins.iter().find(|c| &c.coin_id == coin_id)
    }

    /// Find an MSS key by master_pk.
    pub fn find_mss(&self, pk: &[u8; 32]) -> Option<&MssKeypair> {
        self.data.mss_keys.iter().find(|k| &k.master_pk == pk)
    }

    /// Resolve a coin reference (index, hex prefix, or full hex).
    pub fn resolve_coin(&self, reference: &str) -> Result<[u8; 32]> {
        if let Ok(idx) = reference.parse::<usize>() {
            if idx < self.data.coins.len() {
                return Ok(self.data.coins[idx].coin_id);
            }
        }
        let reference_lower = reference.to_lowercase();
        for c in &self.data.coins {
            if hex::encode(c.coin_id).starts_with(&reference_lower) {
                return Ok(c.coin_id);
            }
        }
        bail!("no matching coin found");
    }

    pub fn coins(&self) -> &[WalletCoin] {
        &self.data.coins
    }

    pub fn keys(&self) -> &[WalletKey] {
        &self.data.keys
    }

    pub fn mss_keys(&self) -> &[MssKeypair] {
        &self.data.mss_keys
    }

    pub fn coin_count(&self) -> usize {
        self.data.coins.len()
    }

    pub fn total_value(&self) -> u64 {
        self.data.coins.iter().map(|c| c.value).sum()
    }

    // ── Transaction building ────────────────────────────────────────────────

    /// Select coins whose total value >= needed. Returns selected coin_ids.
    pub fn select_coins(&self, needed: u64, live_coins: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        let mut selected = Vec::new();
        let mut total = 0u64;
        // Sort by value descending to minimize number of inputs
        let mut available: Vec<&WalletCoin> = self.data.coins.iter()
            .filter(|c| live_coins.contains(&c.coin_id))
            .collect();
        available.sort_by(|a, b| b.value.cmp(&a.value));

        for coin in available {
            if total >= needed { break; }
            selected.push(coin.coin_id);
            total += coin.value;
        }

        if total < needed {
            bail!("insufficient funds: have {}, need {}", total, needed);
        }
        Ok(selected)
    }

    /// Build outputs for a send: recipient outputs + change outputs.
    /// Returns (all_outputs, change_seeds).
    pub fn build_outputs(
        &mut self,
        recipient_pk: &[u8; 32],
        recipient_denominations: &[u64],
        change_value: u64,
    ) -> Result<(Vec<OutputData>, Vec<(usize, [u8; 32])>)> {
        let mut outputs = Vec::new();
        let mut change_seeds = Vec::new();

        // Recipient outputs
        for &denom in recipient_denominations {
            let salt: [u8; 32] = rand::random();
            outputs.push(OutputData {
                owner_pk: *recipient_pk,
                value: denom,
                salt,
            });
        }

        // Change outputs (decompose into power-of-2 denominations to self)
        if change_value > 0 {
            let change_denoms = decompose_value(change_value);
            for denom in change_denoms {
                let seed: [u8; 32] = rand::random();
                let owner_pk = wots::keygen(&seed);
                let salt: [u8; 32] = rand::random();
                let idx = outputs.len();
                outputs.push(OutputData { owner_pk, value: denom, salt });
                change_seeds.push((idx, seed));
            }
        }

        Ok((outputs, change_seeds))
    }

    /// Prepare a commit for given inputs and outputs.
    pub fn prepare_commit(
        &mut self,
        input_coin_ids: &[[u8; 32]],
        outputs: &[OutputData],
        change_seeds: Vec<(usize, [u8; 32])>,
        privacy_delay: bool,
    ) -> Result<([u8; 32], [u8; 32])> {
        // Verify we own all inputs
        for coin_id in input_coin_ids {
            if self.find_coin(coin_id).is_none() {
                bail!("coin {} not in wallet", short_hex(coin_id));
            }
        }

        let output_coin_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.coin_id()).collect();
        let salt: [u8; 32] = rand::random();
        let commitment = compute_commitment(input_coin_ids, &output_coin_ids, &salt);

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
            outputs: outputs.to_vec(),
            change_seeds,
            created_at: now,
            reveal_not_before,
        });
        self.save()?;

        Ok((commitment, salt))
    }

    /// Build InputReveals and signatures for a pending commit.
    pub fn sign_reveal(&mut self, pending: &PendingCommit) -> Result<(Vec<InputReveal>, Vec<Vec<u8>>)> {
        let output_coin_ids: Vec<[u8; 32]> = pending.outputs.iter().map(|o| o.coin_id()).collect();
        let commitment = compute_commitment(
            &pending.input_coin_ids,
            &output_coin_ids,
            &pending.salt,
        );

        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();

        for coin_id in &pending.input_coin_ids {
            // Try WOTS coin
            if let Some(wc) = self.find_coin(coin_id).cloned() {
                input_reveals.push(InputReveal {
                    owner_pk: wc.owner_pk,
                    value: wc.value,
                    salt: wc.salt,
                });
                let sig = wots::sign(&wc.seed, &commitment);
                signatures.push(wots::sig_to_bytes(&sig));
            }
            // Try MSS key (coin's owner_pk matches an MSS master_pk)
            else if let Some(wc) = self.data.coins.iter().find(|c| &c.coin_id == coin_id) {
                // Found the coin data, but the seed might be for an MSS key
                if let Some(pos) = self.data.mss_keys.iter().position(|k| k.master_pk == wc.owner_pk) {
                    input_reveals.push(InputReveal {
                        owner_pk: wc.owner_pk,
                        value: wc.value,
                        salt: wc.salt,
                    });
                    let keypair = &mut self.data.mss_keys[pos];
                    if keypair.remaining() == 0 {
                        bail!("MSS key {} exhausted", short_hex(&wc.owner_pk));
                    }
                    let sig = keypair.sign(&commitment)?;
                    signatures.push(sig.to_bytes());
                } else {
                    bail!("key for {} not found", short_hex(coin_id));
                }
            } else {
                bail!("coin {} not found in wallet", short_hex(coin_id));
            }
        }

        self.save()?;
        Ok((input_reveals, signatures))
    }

    pub fn find_pending(&self, commitment: &[u8; 32]) -> Option<&PendingCommit> {
        self.data.pending.iter().find(|p| &p.commitment == commitment)
    }

    pub fn pending(&self) -> &[PendingCommit] {
        &self.data.pending
    }

    /// Complete a reveal: remove spent coins, add change coins.
    pub fn complete_reveal(&mut self, commitment: &[u8; 32]) -> Result<()> {
        let pending = self.data.pending.iter()
            .find(|p| &p.commitment == commitment)
            .ok_or_else(|| anyhow::anyhow!("pending commit not found"))?
            .clone();

        let spent_coin_ids = pending.input_coin_ids.clone();
        let fee: u64 = {
            let in_sum: u64 = spent_coin_ids.iter()
                .filter_map(|id| self.find_coin(id))
                .map(|c| c.value)
                .sum();
            let out_sum: u64 = pending.outputs.iter().map(|o| o.value).sum();
            in_sum.saturating_sub(out_sum)
        };

        // Remove spent coins
        self.data.coins.retain(|c| !spent_coin_ids.contains(&c.coin_id));

        // Add change coins
        for (idx, seed) in &pending.change_seeds {
            let out = &pending.outputs[*idx];
            let coin_id = out.coin_id();
            if !self.data.coins.iter().any(|c| c.coin_id == coin_id) {
                self.data.coins.push(WalletCoin {
                    seed: *seed,
                    owner_pk: out.owner_pk,
                    value: out.value,
                    salt: out.salt,
                    coin_id,
                    label: Some(format!("change ({})", out.value)),
                });
            }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.data.history.push(HistoryEntry {
            inputs: spent_coin_ids,
            outputs: pending.outputs.iter().map(|o| o.coin_id()).collect(),
            fee,
            timestamp: now,
        });

        self.data.pending.retain(|p| &p.commitment != commitment);
        self.save()?;
        Ok(())
    }

    pub fn history(&self) -> &[HistoryEntry] {
        &self.data.history
    }

    /// Plan a private send: split into independent 2-in-1-out pairs.
    pub fn plan_private_send(
        &self,
        live_coins: &[[u8; 32]],
        recipient_pk: &[u8; 32],
        denominations: &[u64],
    ) -> Result<Vec<(Vec<[u8; 32]>, Vec<OutputData>, Vec<(usize, [u8; 32])>)>> {
        // Each denomination gets its own independent transaction.
        // Each tx: 1+ inputs → 1 recipient output + change outputs.
        // Inputs must sum to > denomination.
        let mut used = std::collections::HashSet::new();
        let mut pairs = Vec::new();

        for &denom in denominations {
            // Find inputs covering denom + 1 (minimum fee)
            let needed = denom + 1;
            let mut selected = Vec::new();
            let mut total = 0u64;

            let mut available: Vec<&WalletCoin> = self.data.coins.iter()
                .filter(|c| live_coins.contains(&c.coin_id) && !used.contains(&c.coin_id))
                .collect();
            available.sort_by(|a, b| b.value.cmp(&a.value));

            for coin in available {
                if total >= needed { break; }
                selected.push(coin.coin_id);
                used.insert(coin.coin_id);
                total += coin.value;
            }

            if total < needed {
                bail!("insufficient funds for private send denomination {}", denom);
            }

            let change = total - denom - 1; // fee = 1
            let salt: [u8; 32] = rand::random();
            let mut outputs = vec![OutputData {
                owner_pk: *recipient_pk,
                value: denom,
                salt,
            }];
            let mut change_seeds = Vec::new();

            if change > 0 {
                for cd in decompose_value(change) {
                    let seed: [u8; 32] = rand::random();
                    let pk = wots::keygen(&seed);
                    let cs: [u8; 32] = rand::random();
                    let idx = outputs.len();
                    outputs.push(OutputData { owner_pk: pk, value: cd, salt: cs });
                    change_seeds.push((idx, seed));
                }
            }

            pairs.push((selected, outputs, change_seeds));
        }

        Ok(pairs)
    }
}

/// Deterministic coinbase seed derivation.
pub fn coinbase_seed(mining_seed: &[u8; 32], height: u64, index: u64) -> [u8; 32] {
    let height_key = hash_concat(mining_seed, &height.to_le_bytes());
    hash_concat(&height_key, &index.to_le_bytes())
}

/// Deterministic coinbase salt derivation (different domain from seed).
pub fn coinbase_salt(mining_seed: &[u8; 32], height: u64, index: u64) -> [u8; 32] {
    let height_key = hash_concat(mining_seed, &height.to_le_bytes());
    hash_concat(&height_key, &(index | 0x8000000000000000).to_le_bytes())
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
        let pk = w.generate_key(Some("test".into())).unwrap();
        assert_eq!(w.keys().len(), 1);
        assert_eq!(w.keys()[0].owner_pk, pk);

        let w2 = Wallet::open(&path, b"pass").unwrap();
        assert_eq!(w2.keys().len(), 1);
    }

    #[test]
    fn import_coin_and_find() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let seed: [u8; 32] = [0x42; 32];
        let salt: [u8; 32] = [0x11; 32];
        let coin_id = w.import_coin(seed, 16, salt, Some("test coin".into())).unwrap();

        assert_eq!(w.coin_count(), 1);
        let found = w.find_coin(&coin_id).unwrap();
        assert_eq!(found.value, 16);
    }

    #[test]
    fn total_value() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        w.import_coin([1u8; 32], 8, [2u8; 32], None).unwrap();
        w.import_coin([3u8; 32], 4, [4u8; 32], None).unwrap();
        assert_eq!(w.total_value(), 12);
    }

    #[test]
    fn short_hex_format() {
        let bytes = [0xab; 32];
        let s = short_hex(&bytes);
        assert_eq!(s, "abababab…abab");
    }
}
