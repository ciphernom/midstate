pub mod crypto;
use crate::core::{hash, hash_concat, compute_commitment, compute_coin_id, compute_address, decompose_value, wots, OutputData, InputReveal};
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
    pub address: [u8; 32],
    pub label: Option<String>,
}

/// A stable identity key that the wallet owner publishes so others can send
/// stealth payments. It never appears on-chain and never signs anything.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanKey {
    /// Private half — never shared.
    pub seed: [u8; 32],
    /// Public half — share this with anyone who wants to send you stealth coins.
    pub public_key: [u8; 32],
    pub label: Option<String>,
}

/// A coin the wallet controls, with known value.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletCoin {
    pub seed: [u8; 32],
    pub owner_pk: [u8; 32],
    pub address: [u8; 32],
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
    #[serde(default)]
    pub scan_keys: Vec<ScanKey>,
    pub pending: Vec<PendingCommit>,
    #[serde(default)]
    pub history: Vec<HistoryEntry>,
    #[serde(default)]
    pub last_scan_height: u64,
}

impl WalletData {
    fn empty() -> Self {
        Self {
            keys: Vec::new(),
            coins: Vec::new(),
            mss_keys: Vec::new(),
            scan_keys: Vec::new(),
            pending: Vec::new(),
            history: Vec::new(),
            last_scan_height: 0,
        }
    }
}

pub struct Wallet {
    path: PathBuf,
    password: Vec<u8>,
    pub data: WalletData,
}

// ---------------------------------------------------------------------------
// Stealth address helpers
// ---------------------------------------------------------------------------

/// Derive a stealth WOTS keypair from a recipient's scan key and a fresh nonce.
///
/// Protocol:
///   shared_secret = BLAKE3(scan_public_key || nonce)
///   stealth_seed  = BLAKE3(shared_secret  || b"wots")   // domain separation
///   stealth_pk    = wots::keygen(&stealth_seed)
///   stealth_addr  = compute_address(&stealth_pk)         // = BLAKE3(stealth_pk)
///
/// Returns (stealth_seed, stealth_pk, stealth_addr).
/// The sender only needs stealth_addr to build the output.
/// The recipient needs stealth_seed to spend the coin later.
pub fn stealth_derive(
    scan_public_key: &[u8; 32],
    nonce: &[u8; 32],
) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let shared_secret = hash_concat(scan_public_key, nonce);
    let stealth_seed  = hash_concat(&shared_secret, b"wots");
    let stealth_pk    = wots::keygen(&stealth_seed);
    let stealth_addr  = compute_address(&stealth_pk);
    (stealth_seed, stealth_pk, stealth_addr)
}

/// Build a stealth OutputData for a recipient identified by their scan key.
/// Returns the output and the nonce; the caller must include the nonce in the
/// reveal transaction's `stealth_nonces` field at the matching index.
pub fn build_stealth_output(
    recipient_scan_key: &[u8; 32],
    value: u64,
) -> (OutputData, [u8; 32]) {
    let nonce: [u8; 32] = rand::random();
    let (_seed, _pk, stealth_addr) = stealth_derive(recipient_scan_key, &nonce);
    let salt: [u8; 32] = rand::random();
    let output = OutputData { address: stealth_addr, value, salt };
    (output, nonce)
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
/// All addresses the wallet watches for (keys + coin addresses).
pub fn watched_addresses(&self) -> Vec<[u8; 32]> {
    let mut addrs: Vec<[u8; 32]> = self.data.keys.iter().map(|k| k.address).collect();
    addrs.extend(self.data.mss_keys.iter().map(|k| k.master_pk));
    addrs.sort();
    addrs.dedup();
    addrs
}

/// Import a scanned coin, matching it to a wallet key. Returns true if new.
pub fn import_scanned(&mut self, address: [u8; 32], value: u64, salt: [u8; 32], stealth_seed: Option<[u8; 32]>) -> Result<Option<[u8; 32]>> {
    let coin_id = compute_coin_id(&address, value, &salt);

    if self.data.coins.iter().any(|c| c.coin_id == coin_id) {
        return Ok(None); // already have it
    }

    // Find matching key
    if let Some(pos) = self.data.keys.iter().position(|k| k.address == address) {
        let key = self.data.keys.remove(pos);
        self.data.coins.push(WalletCoin {
            seed: key.seed,
            owner_pk: key.owner_pk,
            address: key.address,
            value,
            salt,
            coin_id,
            label: key.label,
        });
        return Ok(Some(coin_id));
    }

    // MSS key match — keep key, just add coin
    if let Some(mss) = self.data.mss_keys.iter().find(|k| k.master_pk == address) {
        self.data.coins.push(WalletCoin {
            seed: mss.master_seed,
            owner_pk: mss.master_pk,
            address,
            value,
            salt,
            coin_id,
            label: Some(format!("received ({})", value)),
        });
        return Ok(Some(coin_id));
    }

    // Stealth coin — the caller derived the spending seed during scanning.
    if let Some(seed) = stealth_seed {
        let owner_pk = wots::keygen(&seed);
        self.data.coins.push(WalletCoin {
            seed,
            owner_pk,
            address,
            value,
            salt,
            coin_id,
            label: Some(format!("stealth ({})", value)),
        });
        self.save()?;
        return Ok(Some(coin_id));
    }

    Ok(None) // address not ours
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
        let address = compute_address(&owner_pk);
        self.data.keys.push(WalletKey { seed, owner_pk, address, label });
        self.save()?;
        Ok(address)
    }

    /// Generate a fresh scan key and persist it. Share the returned public key
    /// with anyone who wants to send you stealth payments.
    pub fn generate_scan_key(&mut self, label: Option<String>) -> anyhow::Result<[u8; 32]> {
        let seed: [u8; 32] = rand::random();
        // The scan key is just a random value — it never signs anything.
        let public_key = hash(&seed);
        self.data.scan_keys.push(ScanKey { seed, public_key, label });
        self.save()?;
        Ok(public_key)
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
        let address = compute_address(&owner_pk);
        let coin_id = compute_coin_id(&address, value, &salt);
        if self.data.coins.iter().any(|c| c.coin_id == coin_id) {
            bail!("coin already in wallet");
        }
        self.data.coins.push(WalletCoin {
            seed, owner_pk, address, value, salt, coin_id, label,
        });
        // Remove matching key from unused keys if present
        self.data.keys.retain(|k| k.address != address);
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
        recipient_address: &[u8; 32],
        recipient_denominations: &[u64],
        change_value: u64,
    ) -> Result<(Vec<OutputData>, Vec<(usize, [u8; 32])>)> {
        let mut outputs = Vec::new();
        let mut change_seeds = Vec::new();

        // Recipient outputs
        for &denom in recipient_denominations {
            let salt: [u8; 32] = rand::random();
            outputs.push(OutputData {
                address: *recipient_address,
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
                let address = compute_address(&owner_pk);
                let salt: [u8; 32] = rand::random();
                let idx = outputs.len();
                outputs.push(OutputData { address, value: denom, salt });
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
                let owner_pk = wots::keygen(seed);
                self.data.coins.push(WalletCoin {
                    seed: *seed,
                    owner_pk,
                    address: out.address,
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
        recipient_address: &[u8; 32],
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
                address: *recipient_address,
                value: denom,
                salt,
            }];
            let mut change_seeds = Vec::new();

            if change > 0 {
                for cd in decompose_value(change) {
                    let seed: [u8; 32] = rand::random();
                    let pk = wots::keygen(&seed);
                    let addr = compute_address(&pk);
                    let cs: [u8; 32] = rand::random();
                    let idx = outputs.len();
                    outputs.push(OutputData { address: addr, value: cd, salt: cs });
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
        let addr = w.generate_key(Some("test".into())).unwrap();
        assert_eq!(w.keys().len(), 1);
        assert_eq!(w.keys()[0].address, addr);

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

    // ── resolve_coin ────────────────────────────────────────────────────

    #[test]
    fn resolve_coin_by_index() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let seed = [0x42u8; 32];
        let salt = [0x11; 32];
        let coin_id = w.import_coin(seed, 16, salt, None).unwrap();
        assert_eq!(w.resolve_coin("0").unwrap(), coin_id);
    }

    #[test]
    fn resolve_coin_by_hex_prefix() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let coin_id = w.import_coin([0x42; 32], 8, [0x11; 32], None).unwrap();
        let prefix = &hex::encode(coin_id)[..8];
        assert_eq!(w.resolve_coin(prefix).unwrap(), coin_id);
    }

    #[test]
    fn resolve_coin_not_found() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let w = Wallet::create(&path, b"pass").unwrap();
        assert!(w.resolve_coin("99").is_err());
        assert!(w.resolve_coin("deadbeef").is_err());
    }

    // ── select_coins ────────────────────────────────────────────────────

    #[test]
    fn select_coins_minimal() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c1 = w.import_coin([1; 32], 4, [10; 32], None).unwrap();
        let c2 = w.import_coin([2; 32], 8, [20; 32], None).unwrap();
        let c3 = w.import_coin([3; 32], 16, [30; 32], None).unwrap();

        let live = vec![c1, c2, c3];
        // Need 9 → should select the 16-coin (largest first)
        let selected = w.select_coins(9, &live).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0], c3);
    }

    #[test]
    fn select_coins_insufficient() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c1 = w.import_coin([1; 32], 4, [10; 32], None).unwrap();
        assert!(w.select_coins(100, &[c1]).is_err());
    }

    #[test]
    fn select_coins_ignores_non_live() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c1 = w.import_coin([1; 32], 4, [10; 32], None).unwrap();
        let _c2 = w.import_coin([2; 32], 8, [20; 32], None).unwrap(); // not live

        let selected = w.select_coins(4, &[c1]).unwrap();
        assert_eq!(selected, vec![c1]);
    }

    // ── build_outputs ───────────────────────────────────────────────────

    #[test]
    fn build_outputs_with_change() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let dest = [0xAA; 32];
        let (outputs, change_seeds) = w.build_outputs(&dest, &[4, 2], 3).unwrap();

        // 2 recipient + decompose(3) = 1+2 = 2 change = 4 total
        let recipient_count = 2;
        let change_count = decompose_value(3).len(); // [1, 2]
        assert_eq!(outputs.len(), recipient_count + change_count);
        assert_eq!(change_seeds.len(), change_count);

        // First two are to recipient
        assert_eq!(outputs[0].address, dest);
        assert_eq!(outputs[0].value, 4);
        assert_eq!(outputs[1].address, dest);
        assert_eq!(outputs[1].value, 2);

        // Change values sum correctly
        let change_total: u64 = change_seeds.iter().map(|(idx, _)| outputs[*idx].value).sum();
        assert_eq!(change_total, 3);
    }

    #[test]
    fn build_outputs_no_change() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let (outputs, change_seeds) = w.build_outputs(&[0xBB; 32], &[8], 0).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(change_seeds.is_empty());
    }

    // ── coinbase_seed / coinbase_salt derivation ────────────────────────

    #[test]
    fn coinbase_seed_deterministic() {
        let ms = [0xAA; 32];
        assert_eq!(coinbase_seed(&ms, 100, 0), coinbase_seed(&ms, 100, 0));
    }

    #[test]
    fn coinbase_seed_varies_by_height() {
        let ms = [0xAA; 32];
        assert_ne!(coinbase_seed(&ms, 1, 0), coinbase_seed(&ms, 2, 0));
    }

    #[test]
    fn coinbase_seed_varies_by_index() {
        let ms = [0xAA; 32];
        assert_ne!(coinbase_seed(&ms, 1, 0), coinbase_seed(&ms, 1, 1));
    }

    #[test]
    fn coinbase_seed_differs_from_salt() {
        let ms = [0xAA; 32];
        assert_ne!(coinbase_seed(&ms, 1, 0), coinbase_salt(&ms, 1, 0));
    }

    // ── watched_addresses ───────────────────────────────────────────────

    #[test]
    fn watched_addresses_includes_keys_and_mss() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let addr1 = w.generate_key(None).unwrap();
        let mss_addr = w.generate_mss(4, None).unwrap();

        let watched = w.watched_addresses();
        assert!(watched.contains(&addr1));
        assert!(watched.contains(&mss_addr));
    }

    #[test]
    fn watched_addresses_deduped() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        w.generate_key(None).unwrap();
        w.generate_key(None).unwrap();
        let watched = w.watched_addresses();
        let mut sorted = watched.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(watched.len(), sorted.len());
    }

    // ── import_scanned ──────────────────────────────────────────────────

    #[test]
    fn import_scanned_matches_key() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let addr = w.generate_key(Some("scan test".into())).unwrap();

        let salt = [0x55; 32];
        let value = 8u64;
        let result = w.import_scanned(addr, value, salt, None).unwrap();
        assert!(result.is_some());
        assert_eq!(w.coin_count(), 1);
        assert_eq!(w.keys().len(), 0); // key consumed
    }

    #[test]
    fn import_scanned_ignores_unknown_address() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let result = w.import_scanned([0xFF; 32], 8, [0; 32], None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn import_scanned_dedup() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let addr = w.generate_key(None).unwrap();
        let salt = [0x55; 32];

        w.import_scanned(addr, 8, salt, None).unwrap();
        // Second import same coin → None
        let result = w.import_scanned(addr, 8, salt, None).unwrap();
        assert!(result.is_none());
        assert_eq!(w.coin_count(), 1);
    }

    // ── generate_mss ────────────────────────────────────────────────────

    #[test]
    fn generate_mss_creates_key() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let root = w.generate_mss(4, Some("test mss".into())).unwrap();
        assert_ne!(root, [0u8; 32]);
        assert_eq!(w.mss_keys().len(), 1);
        assert_eq!(w.mss_keys()[0].height, 4);
    }

    // ── plan_private_send ───────────────────────────────────────────────

    #[test]
    fn plan_private_send_independent_pairs() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c1 = w.import_coin([1; 32], 8, [10; 32], None).unwrap();
        let c2 = w.import_coin([2; 32], 4, [20; 32], None).unwrap();
        let c3 = w.import_coin([3; 32], 16, [30; 32], None).unwrap();

        let live = vec![c1, c2, c3];
        let dest = [0xAA; 32];
        let pairs = w.plan_private_send(&live, &dest, &[4, 2]).unwrap();

        assert_eq!(pairs.len(), 2);
        // Each pair should have non-overlapping inputs
        let all_inputs: Vec<[u8; 32]> = pairs.iter()
            .flat_map(|(ins, _, _)| ins.clone())
            .collect();
        let mut deduped = all_inputs.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(all_inputs.len(), deduped.len(), "inputs should not overlap between pairs");
    }

    #[test]
    fn plan_private_send_insufficient_funds() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        let mut w = Wallet::create(&path, b"pass").unwrap();
        let c1 = w.import_coin([1; 32], 2, [10; 32], None).unwrap();
        assert!(w.plan_private_send(&[c1], &[0xAA; 32], &[4, 4]).is_err());
    }

    // ── wrong password ──────────────────────────────────────────────────

    #[test]
    fn open_wrong_password() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        Wallet::create(&path, b"correct").unwrap();
        assert!(Wallet::open(&path, b"wrong").is_err());
    }

    #[test]
    fn create_duplicate_path_fails() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap();

        Wallet::create(&path, b"pass").unwrap();
        assert!(Wallet::create(&path, b"pass").is_err());
    }
    
    #[test]
    fn test_mss_safety_recovery_persistence() {
        // 1. Setup: Create a wallet with an MSS key
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        std::fs::remove_file(&path).unwrap(); // ensure it doesn't exist

        let mut w = Wallet::create(&path, b"password").unwrap();
        w.generate_mss(4, Some("test_mss".to_string())).unwrap();
        
        // Initial state: leaf index should be 0
        assert_eq!(w.mss_keys()[0].next_leaf, 0);

        // 2. Simulate the Fix:
        // The "network" tells us the index is actually 50.
        // We apply the fix logic: update internal state + safety margin.
        let remote_index = 50;
        let safety_margin = 20;
        let new_index = remote_index + safety_margin;

        // Apply fix directly to the data structure
        w.data.mss_keys[0].set_next_leaf(new_index);
        w.save().unwrap();

        // 3. Verify Persistence:
        // Close the wallet and reopen it from disk.
        let w_reloaded = Wallet::open(&path, b"password").unwrap();
        
        // The loaded wallet must have the updated index.
        assert_eq!(w_reloaded.mss_keys()[0].next_leaf, 70);
        
        // Ensure we didn't lose the key itself
        assert_eq!(w_reloaded.mss_keys().len(), 1);
        assert_eq!(w_reloaded.mss_keys()[0].height, 4);
    }
}
