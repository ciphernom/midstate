//! # wasm-wallet — WebAssembly wallet for the Midstate blockchain
//!
//! This crate provides a WASM-compatible wallet that runs entirely in the browser.
//! It handles key derivation, coin selection, transaction building, MSS signing,
//! and solo mining nonce search.
//!
//! ## Architecture
//!
//! The wallet runs inside a Web Worker. MSS Merkle trees (~64 KB each at height 10)
//! are stored in IndexedDB as compact binary blobs via [`WebWallet::export_mss_bytes`]
//! and [`WebWallet::import_mss_bytes`]. This gives the web wallet identical signing
//! performance to the native CLI — a simple tree lookup + one WOTS signature.
//!
//! ## MSS Binary Format
//!
//! The binary format for MSS tree export/import is:
//!
//! ```text
//! ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────────┐
//! │ height (4B)  │ master_seed  │ next_leaf    │ master_pk    │ tree_len     │ tree nodes     │
//! │ u32 LE       │ (32B)        │ u64 LE (8B)  │ (32B)        │ u32 LE (4B)  │ (N × 32B)      │
//! └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────────┘
//!   offset 0       offset 4       offset 36      offset 44      offset 76      offset 80
//! ```
//!
//! For height 10: N = 2048 nodes → 80 + 65,536 = 65,616 bytes (~64 KB).

use wasm_bindgen::prelude::*;
use midstate::core::wots;
use midstate::core::mss::MssKeypair;
use midstate::core::types::{compute_address, compute_commitment, compute_coin_id, decompose_value};
use midstate::wallet::hd::{generate_mnemonic, master_seed_from_mnemonic, derive_wots_seed, derive_mss_seed};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
use wasm_bindgen_test::wasm_bindgen_test_configure;

// noop //

// ─── Constants ──────────────────────────────────────────────────────────────

/// Size of the fixed header in the MSS binary export format.
///
/// Layout: height(4) + master_seed(32) + next_leaf(8) + master_pk(32) + tree_len(4) = 80
const MSS_BINARY_HEADER_SIZE: usize = 80;

/// Maximum number of inputs allowed in a single transaction.
/// Matches the consensus limit (MAX_TX_INPUTS = 256), with a safety margin.
const MAX_SELECTED_INPUTS: usize = 250;

const SALT_DOMAIN: &[u8] = b"midstate/salt/v1";

fn derive_deterministic_salt(master_seed: &[u8; 32], index: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SALT_DOMAIN);
    hasher.update(master_seed);
    hasher.update(&index.to_le_bytes());
    *hasher.finalize().as_bytes()
}

// ─── Global Wasm Helpers ────────────────────────────────────────────────────

/// Generate a new BIP39 24-word mnemonic phrase.
///
/// Returns the phrase as a space-separated string. The corresponding master
/// seed is derived when the phrase is passed to [`WebWallet::new`].
///
/// # Panics
///
/// Panics if the system CSPRNG fails (should never happen in a browser).
#[wasm_bindgen]
pub fn generate_phrase() -> String {
    let (_, phrase) = generate_mnemonic().unwrap();
    phrase
}

/// Decompose an amount into canonical power-of-2 denominations.
///
/// The Midstate UTXO model requires all coin values to be exact powers of 2.
/// This function splits any amount into the minimal set of such denominations.
///
/// # Example
///
/// An input of `13` yields `[1, 4, 8]` (three coins: 2^0 + 2^2 + 2^3 = 13).
#[wasm_bindgen]
pub fn decompose_amount(amount: u64) -> js_sys::BigUint64Array {
    let parts = decompose_value(amount);
    js_sys::BigUint64Array::from(&parts[..])
}

/// Compute the coin ID (UTXO identifier) from an address, value, and salt.
///
/// `coin_id = BLAKE3(address || value_le_bytes || salt)`
///
/// All inputs and output are hex-encoded strings.
///
/// # Returns
///
/// A 64-character hex string representing the 32-byte coin ID.
///
/// # Edge Cases
///
/// If `address_hex` or `salt_hex` are invalid hex (wrong length, bad chars),
/// the corresponding bytes default to all zeros. This matches the CLI behavior
/// where malformed inputs produce deterministic (but useless) outputs rather
/// than panicking.
#[wasm_bindgen]
pub fn compute_coin_id_hex(address_hex: &str, value: u64, salt_hex: &str) -> String {
    let mut addr = [0u8; 32];
    if let Ok(decoded) = hex::decode(address_hex) {
        if decoded.len() >= 32 {
            addr.copy_from_slice(&decoded[0..32]);
        }
    }
    let mut salt = [0u8; 32];
    if let Ok(decoded) = hex::decode(salt_hex) {
        if decoded.len() >= 32 {
            salt.copy_from_slice(&decoded[0..32]);
        }
    }
    let cid = compute_coin_id(&addr, value, &salt);
    hex::encode(cid)
}

/// Safely mines the Commitment PoW in the WebAssembly context.
///
/// # Reasoning
/// Intercepts invalid or missing hex strings (e.g., from an out-of-sync RPC cache)
/// and handles them gracefully. Replacing `.unwrap()` with silent fallbacks prevents
/// the Web Worker from panicking and permanently hanging the UI on "Mining PoW...".
///
/// # Formal Specification
/// ```text
/// Pre:  true
/// Post: result = mine_pow(commitment, required_pow, target_height, header_hash) if hex valid
///       result = 0 if commitment_hex invalid
/// ```
///
/// ```zed
///     MineCommitmentPowWasm
///     ---------------------
///     commitment_hex? : String
///     required_pow? : ℕ₃₂
///     target_height? : ℕ₆₄
///     header_hash_hex? : String
///     nonce! : ℕ₆₄
///
///     pre  true
///     post (isHex32(commitment_hex?) ⇒ nonce! = MinePow(...))
///        ∧ (¬isHex32(commitment_hex?) ⇒ nonce! = 0)
/// ```
#[wasm_bindgen]
pub fn mine_commitment_pow(commitment_hex: &str, required_pow: u32, target_height: u64, header_hash_hex: &str) -> u64 {
    let mut commitment = [0u8; 32];
    if hex::decode_to_slice(commitment_hex, &mut commitment).is_err() {
        return 0; // Return gracefully so we don't kill the worker
    }

    let mut header_hash = [0u8; 32];
    let _ = hex::decode_to_slice(header_hash_hex, &mut header_hash); // Ignore error, leaves as zeroes

    midstate::core::transaction::mine_pow(&commitment, required_pow, target_height, header_hash)
}

// ─── Solo Mining: Hot-Loop Nonce Search ─────────────────────────────────────

/// Search a range of nonces for a valid block extension hash.
///
/// This is the inner loop of the browser-based solo miner. Each call tests
/// `iterations × 4` nonces using SIMD 4-way parallelism (WASM SIMD128).
///
/// # Arguments
///
/// * `midstate_hex` — 64-char hex of the block header midstate.
/// * `target_hex` — 64-char hex of the difficulty target (big-endian).
/// * `start_nonce` — first nonce to test.
/// * `iterations` — number of SIMD batches (each batch = 4 nonces).
///
/// # Returns
///
/// `Some(winning_nonce)` if a hash below target was found, `None` otherwise.
///
/// # Performance
///
/// With WASM SIMD128 enabled, each call with `iterations=1` takes ~800ms
/// due to the expensive iterated hashing (EXTENSION_ITERATIONS = 1,000,000).
#[wasm_bindgen]
pub fn search_nonces(midstate_hex: &str, target_hex: &str, start_nonce: u64, iterations: u32) -> Option<u64> {
    let mut midstate = [0u8; 32];
    if hex::decode_to_slice(midstate_hex, &mut midstate).is_err() {
        return None;
    }

    let mut target = [0u8; 32];
    if hex::decode_to_slice(target_hex, &mut target).is_err() {
        return None;
    }

    for i in 0..iterations {
        let base = start_nonce + (i as u64 * 4);
        let nonces = [base, base + 1, base + 2, base + 3];

        let results = midstate::core::simd_mining::create_extensions_4way(midstate, nonces);

        for (nonce, final_hash) in results {
            if final_hash < target {
                return Some(nonce);
            }
        }
    }

    None
}

#[derive(serde::Deserialize)]
struct ScriptInputArg {
    /// 64-hex on-chain coin id of the contract UTXO being consumed.
    coin_id: String,
    /// Comma-separated hex witness stack, EXACTLY as the IDE emulator shows it
    /// (pushed left→right). May be empty for contracts that take no witness.
    #[serde(default)]
    witness: String,
    /// Value of this contract coin (0 for a pure state/confidential coin).
    #[serde(default)]
    value: u64,
    /// 64-hex salt this coin was created with. The wallet must know it — see the
    /// "contract-coin tracking" note in the integration plan.
    salt: String,
    /// 64-hex state, present iff this is a confidential (state) coin.
    #[serde(default)]
    state: Option<String>,
}

#[derive(serde::Deserialize)]
struct ScriptOutputArg {
    /// "standard" | "confidential"
    out_type: String,
    /// 64-hex destination address.
    address: String,
    /// Value in sats (standard outputs only; confidential = 0).
    #[serde(default)]
    value: u64,
    /// 64-hex state/commitment for confidential outputs.
    #[serde(default)]
    state: Option<String>,
    /// Optional explicit salt (64-hex). Supply this for outputs the wallet will
    /// later spend (e.g. a new contract-state coin) so the salt is recorded;
    /// otherwise a random salt is generated.
    #[serde(default)]
    salt: Option<String>,
}

// ── Context carried between phase 1 and phase 2 ─────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct ScriptWalletInput {
    coin_id: String,
    address: String,
    value: u64,
    salt: String,
    is_mss: bool,
    index: u32,
    mss_leaf: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScriptContractInput {
    coin_id: String,
    bytecode: String,
    witness: String,
    value: u64,
    salt: String,
    state: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ScriptSpendContext {
    contract_addr: String,
    contract_inputs: Vec<ScriptContractInput>,
    wallet_inputs: Vec<ScriptWalletInput>,
    /// Ordered output JSON values with salts already embedded.
    outputs: Vec<serde_json::Value>,
    /// Ordered input coin ids used to compute the commitment: contract…, wallet…
    input_coin_ids: Vec<String>,
    tx_salt: String,
    commitment: String,
    fee: u64,
    next_wots_index: u32,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Normalize a comma-separated witness string: trim each token, drop empties,
/// validate hex, re-join without spaces. "" stays "" (empty witness stack).
fn normalize_witness(w: &str) -> Result<String, JsValue> {
    let mut toks = Vec::new();
    for raw in w.split(',') {
        let t = raw.trim();
        if t.is_empty() {
            continue;
        }
        if hex::decode(t).is_err() {
            return Err(JsValue::from_str(&format!("Invalid hex in witness token: '{}'", t)));
        }
        toks.push(t.to_lowercase());
    }
    Ok(toks.join(","))
}

/// Coin id for a confidential (state) coin: blake3("CONFIDENTIAL" || addr || state || salt).
fn confidential_coin_hash(addr: &[u8; 32], state: &[u8; 32], salt: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"CONFIDENTIAL");
    h.update(addr);
    h.update(state);
    h.update(salt);
    *h.finalize().as_bytes()
}


// ─── JSON Interop Structs ───────────────────────────────────────────────────

/// A UTXO as represented in the JavaScript wallet state.
///
/// This struct is deserialized from the JSON array passed by `worker.js`
/// into [`WebWallet::prepare_spend`].
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
struct WasmUtxo {
    /// HD derivation index (WOTS) or MSS key index.
    index: u32,
    /// `true` if this UTXO is backed by an MSS reusable address.
    is_mss: bool,
    /// MSS tree height (only meaningful if `is_mss` is true).
    mss_height: u32,
    /// Current MSS leaf index for signing (only meaningful if `is_mss` is true).
    mss_leaf: u32,
    /// Hex-encoded 32-byte address (owner public key hash).
    address: String,
    /// Coin value (always a power of 2).
    value: u64,
    /// Hex-encoded 32-byte random salt.
    salt: String,
    /// Hex-encoded 32-byte coin ID (UTXO identifier).
    coin_id: String,
}

/// A transaction output as built by the wallet.
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
struct JsOutput {
    /// Hex-encoded 32-byte destination address.
    address: String,
    /// Output value (always a power of 2).
    value: u64,
    /// Hex-encoded 32-byte random salt. Defaults to empty if not provided.
    #[serde(default)]
    salt: String,
}

/// Complete context for a pending spend transaction.
///
/// This is serialized to JSON, passed back to JavaScript, and later
/// fed back into [`WebWallet::build_reveal`] to produce the reveal payload.
#[derive(Serialize, Deserialize, Debug)]
struct SpendContext {
    /// The UTXOs selected as inputs.
    selected_inputs: Vec<WasmUtxo>,
    /// The outputs (recipient + change) - also, generic JSON Value to support both Standard and DataBurn outputs
        outputs: Vec<serde_json::Value>, 
    /// The commitment payload sent to the network during the commit phase.
    commit_payload: serde_json::Value,
    /// Hex-encoded 32-byte transaction salt.
    tx_salt: String,
    /// Hex-encoded 32-byte commitment hash.
    commitment: String,
    /// Calculated transaction fee.
    fee: u64,
    /// Updated WOTS derivation index after change key derivation.
    next_wots_index: u32,
}

// ─── Main Wallet Object ─────────────────────────────────────────────────────

/// The core wallet struct holding the master seed and cached MSS trees.
///
/// All MSS Merkle trees are stored as complete `MssKeypair` structs — the
/// same type used by the native CLI wallet. This means signing is a simple
/// tree lookup (~10μs) rather than the expensive subtree recomputation that
/// the old `FractionalMss` design required.
///
/// ## Lifecycle
///
/// 1. Created via [`WebWallet::new`] (from mnemonic) or [`WebWallet::from_seed_hex`].
/// 2. MSS trees loaded from IndexedDB via [`WebWallet::import_mss_bytes`].
/// 3. Signing via [`WebWallet::build_reveal`] (uses cached trees).
/// 4. After generating new MSS keys, export via [`WebWallet::export_mss_bytes`]
///    for IndexedDB persistence.
#[wasm_bindgen]
pub struct WebWallet {
    master_seed: [u8; 32],
    mss_cache: HashMap<String, MssKeypair>,
    watchlist: Vec<[u8; 32]>,
}

/// Decrypt a native CLI wallet file (`.dat`) using its password.
///
/// Returns the decrypted JSON string containing the full wallet data
/// (master_seed, coins, mss_keys, history, etc.).
///
/// # Errors
///
/// Returns `Err` if decryption fails (wrong password) or the decrypted
/// data is not valid UTF-8.
#[wasm_bindgen]
pub fn decrypt_cli_wallet(data: &[u8], password: &str) -> Result<String, JsValue> {
    let decrypted = midstate::wallet::crypto::decrypt(data, password.as_bytes())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let json_str = String::from_utf8(decrypted)
        .map_err(|_| JsValue::from_str("Invalid UTF-8 in decrypted data"))?;

    Ok(json_str)
}

/// Safely parses a hex-encoded Midstate address into a 32-byte array.
///
/// This parser supports two address formats:
/// 1. **Legacy Format (64 characters):** A raw 32-byte hex string.
/// 2. **Checksummed Format (72 characters):** A 36-byte hex string where the 
///    first 32 bytes are the address, and the final 4 bytes are a BLAKE3 
///    checksum (`BLAKE3(address)[0..4]`).
///
/// This matches the native CLI wallet's `parse_address_flexible` behavior, 
/// ensuring that typos in the web UI are caught before a transaction is built.
///
/// # Arguments
///
/// * `s` - The hex-encoded address string provided by the user/UI.
///
/// # Returns
///
/// * `Ok([u8; 32])` - The extracted 32-byte address.
/// * `Err(JsValue)` - A string error suitable for throwing to JavaScript if the 
///   hex is invalid, the length is wrong, or the checksum fails.
fn parse_address_wasm(s: &str) -> Result<[u8; 32], JsValue> {
    let decoded = hex::decode(s).map_err(|_| JsValue::from_str("Invalid hex in address."))?;
    
    if decoded.len() == 32 {
        // Legacy 64-character format
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&decoded);
        Ok(addr)
    } else if decoded.len() == 36 {
        // New 72-character format with checksum
        let mut addr = [0u8; 32];
        addr.copy_from_slice(&decoded[0..32]);
        
        let expected_checksum = &decoded[32..36];
        let actual_checksum_hash = midstate::core::types::hash(&addr);
        
        if expected_checksum != &actual_checksum_hash[0..4] {
            return Err(JsValue::from_str("Checksum mismatch! The address contains a typo."));
        }
        Ok(addr)
    } else {
        Err(JsValue::from_str("Invalid address length. Expected 64 or 72 hex characters."))
    }
}

#[wasm_bindgen]
impl WebWallet {
    /// Create a new wallet from a BIP39 mnemonic phrase.
    ///
    /// The master seed is derived via BLAKE3 from the mnemonic. The wallet
    /// starts with an empty MSS cache — call [`import_mss_bytes`] to load
    /// persisted trees, or [`get_mss_address`] to generate new ones.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the mnemonic is invalid (wrong word count, unknown words,
    /// or bad checksum).
    #[wasm_bindgen(constructor)]
    pub fn new(phrase: &str) -> Result<WebWallet, JsValue> {
        let master_seed = master_seed_from_mnemonic(phrase)
            .map_err(|e| JsValue::from_str(&format!("Invalid mnemonic: {}", e)))?;
        Ok(WebWallet {
            master_seed,
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        })
    }


    /// Universal DeFi Transaction Builder
    /// Constructs a transaction that transitions a State Thread while securely attaching 
    /// physical UTXOs to satisfy covenants (like paying a Treasury).
    /// Uses dynamic fee calculation and greedy UTXO defragmentation.
    #[wasm_bindgen]
    pub fn build_state_thread_tx(
        &mut self,
        available_utxos_json: &str,
        contract_bytecode_hex: &str,
        current_state_hex: Option<String>,
        current_coin_id_hex: Option<String>,
        current_salt_hex: Option<String>,
        new_state_hex: &str,
        extra_outputs_json: &str,
        next_wots_index: u32,
    ) -> Result<String, JsValue> {
        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        #[derive(serde::Deserialize)]
        struct ExtraOut { 
            out_type: String, 
            address: String, 
            #[serde(default)] value: u64,
            #[serde(default)] commitment: Option<String>
        }
        let extra_outs: Vec<ExtraOut> = serde_json::from_str(extra_outputs_json).unwrap_or_default();
        
        let extra_value: u64 = extra_outs.iter().map(|o| o.value).sum();
        
        let mut avail_sorted = available.clone();
        avail_sorted.sort_by(|a, b| b.value.cmp(&a.value));

        // 1. Dynamic Fee Estimation & Coin Selection Loop
        let mut target_fee = 100u64;
        let mut selected = Vec::new();
        let mut total = 0u64;
        let final_fee;
        
        loop {
            let needed = extra_value + target_fee;
            selected.clear();
            let mut selected_set = HashSet::new();
            total = 0;

            // Greedy Selection
            for coin in &avail_sorted {
                if total >= needed { break; }
                selected_set.insert(coin.coin_id.clone());
                selected.push(coin.clone());
                total += coin.value;
            }

            if total < needed { return Err(JsValue::from_str("Insufficient funds for contract requirements and fees")); }

            // WOTS Co-Spend Enforcement
            let mut grouped_addresses = HashSet::new();
            for c in &selected {
                if !c.is_mss { grouped_addresses.insert(c.address.clone()); }
            }
            for coin in &available {
                if grouped_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected_set.insert(coin.coin_id.clone());
                    selected.push(coin.clone());
                    total += coin.value;
                }
            }

            // Snowball Merge (Defragmentation)
            let mut added_new = true;
            while added_new {
                added_new = false;
                let current_change = total.saturating_sub(extra_value).saturating_sub(target_fee);
                let change_denoms = decompose_value(current_change);
                for denom in change_denoms {
                    if let Some(pos) = available.iter().position(|c| c.value == denom && !selected_set.contains(&c.coin_id)) {
                        let coin_to_add = available[pos].clone();
                        selected_set.insert(coin_to_add.coin_id.clone());
                        selected.push(coin_to_add);
                        total += denom;
                        added_new = true;
                        break;
                    }
                }
                if selected.len() >= MAX_SELECTED_INPUTS { break; }
            }

            // Final Co-Spend Sweep
            let mut final_addresses = HashSet::new();
            for c in &selected {
                if !c.is_mss { final_addresses.insert(c.address.clone()); }
            }
            for coin in &available {
                if final_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected_set.insert(coin.coin_id.clone());
                    selected.push(coin.clone());
                    total += coin.value;
                }
            }

            // Calculate precise byte size of the transaction
            let mut num_outputs = extra_outs.len() + 1; // +1 for the State Thread output
            let final_change_val = total.saturating_sub(extra_value).saturating_sub(target_fee);
            num_outputs += decompose_value(final_change_val).len();

            // Bytes: Base Tx (100) + Wallet Inputs (~1636 each) + State Thread Input (~100) + Outputs (~100 each)
            let estimated_bytes = 100 
                + (selected.len() as u64 * 1636) 
                + (if current_state_hex.is_some() { 100 } else { 0 }) 
                + (num_outputs as u64 * 100);
                
            let required_fee = (estimated_bytes * 10) / 1024 + 10;

            if total >= extra_value + required_fee {
                final_fee = required_fee;
                break;
            } else {
                target_fee = required_fee;
            }
        }

        // 2. Build the Transaction
        let contract_bytes = hex::decode(contract_bytecode_hex).unwrap();
        let contract_addr = midstate::core::types::hash(&contract_bytes);

        let mut input_coin_ids = Vec::new();
        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();

        // -> Consume Contract Input (Old State)
        if let (Some(state), Some(cid), Some(salt)) = (&current_state_hex, &current_coin_id_hex, &current_salt_hex) {
            let mut cid_b = [0u8; 32]; hex::decode_to_slice(cid, &mut cid_b).unwrap();
            input_coin_ids.push(cid_b);
            input_reveals.push(serde_json::json!({
                "bytecode": contract_bytecode_hex,
                "value": 0,
                "salt": salt,
                "commitment": state
            }));
            // Dynamically inject the contract's own address into the witness stack 
            // before the routing integer (00), solving the circular hashing problem!
            signatures.push(format!("{},00", hex::encode(contract_addr)));

        }

        // -> Consume Wallet Inputs
        for inp in &selected {
            let mut cid_b = [0u8; 32]; hex::decode_to_slice(&inp.coin_id, &mut cid_b).unwrap();
            input_coin_ids.push(cid_b);
            
            let pk = if inp.is_mss {
                self.mss_cache.get(&inp.address).unwrap().master_pk
            } else {
                let seed = derive_wots_seed(&self.master_seed, inp.index as u64);
                wots::keygen(&seed)
            };
            let bytecode = midstate::core::script::compile_p2pk(&pk);
            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&bytecode),
                "value": inp.value,
                "salt": inp.salt
            }));
        }

        let mut outputs_json = Vec::new();
        let mut output_hashes = Vec::new();

        // -> Create Contract Output (New State)
        let mut new_state_b = [0u8; 32]; hex::decode_to_slice(new_state_hex, &mut new_state_b).unwrap();
        let mut contract_out_salt = [0u8; 32]; getrandom_02::getrandom(&mut contract_out_salt).unwrap();
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"CONFIDENTIAL");
        hasher.update(&contract_addr);
        hasher.update(&new_state_b);
        hasher.update(&contract_out_salt);
        output_hashes.push(*hasher.finalize().as_bytes());

        outputs_json.push(serde_json::json!({
            "type": "confidential",
            "address": hex::encode(contract_addr),
            "commitment": new_state_hex,
            "salt": hex::encode(contract_out_salt)
        }));

        // -> Create Extra Outputs (Treasury Payment & Token Minting)
        for ext in extra_outs {
            let mut addr_b = [0u8; 32]; hex::decode_to_slice(&ext.address, &mut addr_b).unwrap();
            let mut salt_b = [0u8; 32]; getrandom_02::getrandom(&mut salt_b).unwrap();
            
            if ext.out_type == "confidential" {
                let comm_hex = ext.commitment.unwrap_or_default();
                let mut comm_b = [0u8; 32]; hex::decode_to_slice(&comm_hex, &mut comm_b).unwrap();
                
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"CONFIDENTIAL");
                hasher.update(&addr_b);
                hasher.update(&comm_b);
                hasher.update(&salt_b);
                output_hashes.push(*hasher.finalize().as_bytes());

                outputs_json.push(serde_json::json!({
                    "type": "confidential",
                    "address": ext.address,
                    "commitment": comm_hex,
                    "salt": hex::encode(salt_b)
                }));
            } else {
                output_hashes.push(compute_coin_id(&addr_b, ext.value, &salt_b));
                outputs_json.push(serde_json::json!({
                    "type": "standard",
                    "address": ext.address,
                    "value": ext.value,
                    "salt": hex::encode(salt_b)
                }));
            }
        }

        // -> Create Change Output
        let change = total - extra_value - final_fee;
        let mut current_idx = next_wots_index;
        if change > 0 {
            for denom in decompose_value(change) {
                let change_seed = derive_wots_seed(&self.master_seed, current_idx as u64);
                let change_addr = compute_address(&wots::keygen(&change_seed));
                let change_salt = derive_deterministic_salt(&self.master_seed, current_idx as u64);
                
                output_hashes.push(compute_coin_id(&change_addr, denom, &change_salt));
                outputs_json.push(serde_json::json!({
                    "type": "standard",
                    "address": hex::encode(change_addr),
                    "value": denom,
                    "salt": hex::encode(change_salt)
                }));
                current_idx += 1;
            }
        }

        let mut tx_salt = [0u8; 32]; getrandom_02::getrandom(&mut tx_salt).unwrap();
        let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);

        // 3. Securely Sign Wallet Inputs
        for inp in &selected {
            if inp.is_mss {
                let kp = self.mss_cache.get_mut(&inp.address).unwrap();
                kp.next_leaf = inp.mss_leaf as u64;
                let sig = kp.sign(&commitment).unwrap();
                signatures.push(hex::encode(sig.to_bytes()));
            } else {
                let seed = derive_wots_seed(&self.master_seed, inp.index as u64);
                let sig = wots::sign(&seed, &commitment);
                signatures.push(hex::encode(wots::sig_to_bytes(&sig)));
            }
        }

        Ok(serde_json::json!({
            "commitment": hex::encode(commitment),
            "reveal": {
                "inputs": input_reveals,
                "signatures": signatures,
                "outputs": outputs_json,
                "salt": hex::encode(tx_salt)
            },
            "next_wots_index": current_idx,
            "fee": final_fee
        }).to_string())
    }

    /// Build coinbase outputs for web solo mining.
    ///
    /// Decomposes `total_value` into power-of-2 denominations and assigns
    /// them directly to the user's reusable MSS address.
    #[wasm_bindgen]
    pub fn build_coinbase_to_mss(
        &self,
        total_value: u64,
        address_hex: &str,
    ) -> Result<String, JsValue> {
        let mut address = [0u8; 32];
        hex::decode_to_slice(address_hex, &mut address)
            .map_err(|_| JsValue::from_str("Invalid address hex"))?;

        let denominations = decompose_value(total_value);
        let mut coinbase_json = Vec::with_capacity(denominations.len());

        for &denom in &denominations {
            let mut salt = [0u8; 32];
            getrandom_02::getrandom(&mut salt).unwrap();

            coinbase_json.push(serde_json::json!({
                "address": address_hex,
                "value": denom,
                "salt": hex::encode(salt)
            }));
        }

        // We no longer return or advance the next_wots_index!
        Ok(serde_json::json!({
            "coinbase": coinbase_json
        }).to_string())
    }

    /// Build coinbase outputs for solo mining.
    ///
    /// Decomposes `total_value` (block reward + fees) into power-of-2
    /// denominations, derives a fresh WOTS address for each, and returns
    /// a JSON string containing:
    ///
    /// - `coinbase`: array of `{address, value, salt}` for the block template.
    /// - `mining_addrs`: array of `{address, index}` for the wallet to track.
    /// - `next_wots_index`: the updated derivation counter.
    ///
    /// # Returns
    ///
    /// `None` if CSPRNG fails (should never happen in a browser).
    #[wasm_bindgen]
    pub fn build_coinbase(
        &self,
        total_value: u64,
        next_wots_index: u32,
    ) -> Option<String> {
        let denominations = decompose_value(total_value);

        let mut coinbase_json = Vec::with_capacity(denominations.len());
        let mut mining_addrs = Vec::with_capacity(denominations.len());
        let mut wots_idx = next_wots_index;

        for &denom in &denominations {
            let seed = derive_wots_seed(&self.master_seed, wots_idx as u64);
            let pk = wots::keygen(&seed);
            let address = compute_address(&pk);

            // Deterministic salt derived from the master seed and index!
            let salt = derive_deterministic_salt(&self.master_seed, wots_idx as u64);

            coinbase_json.push(serde_json::json!({
                "address": hex::encode(address),
                "value": denom,
                "salt": hex::encode(salt)
            }));

            mining_addrs.push(serde_json::json!({
                "address": hex::encode(address),
                "index": wots_idx
            }));

            wots_idx += 1;
        }

        Some(serde_json::json!({
            "coinbase": coinbase_json,
            "mining_addrs": mining_addrs,
            "next_wots_index": wots_idx
        }).to_string())
    }

    /// Recompute the block extension hash for a found nonce.
    ///
    /// Called after a mining worker finds a valid nonce. Produces the full
    /// `Extension { nonce, final_hash }` JSON needed for block submission.
    ///
    /// # Returns
    ///
    /// `None` if `midstate_hex` is not valid 64-character hex.
#[wasm_bindgen]
pub fn build_solo_extension(&self, midstate_hex: &str, nonce: u64) -> Option<String> {
    let mut midstate = [0u8; 32];
    hex::decode_to_slice(midstate_hex, &mut midstate).ok()?;

    let ext = midstate::core::extension::create_extension(midstate, nonce);

    Some(serde_json::json!({
        "nonce": ext.nonce,
        // FIX: Serialize as an array of numbers so it maps properly 
        // to `[u8; 32]` when deserialized by the Node.
        "final_hash": ext.final_hash.to_vec() 
    }).to_string())
}

    /// Create a wallet from a raw 32-byte master seed (hex-encoded).
    ///
    /// Used when importing a CLI wallet backup where the master seed is
    /// available directly rather than via a mnemonic phrase.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `seed_hex` is not exactly 64 valid hex characters.
    pub fn from_seed_hex(seed_hex: &str) -> Result<WebWallet, JsValue> {
        let mut master_seed = [0u8; 32];
        hex::decode_to_slice(seed_hex, &mut master_seed)
            .map_err(|_| JsValue::from_str("Invalid master seed hex"))?;

        Ok(WebWallet {
            master_seed,
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        })
    }

    /// Set the list of addresses the wallet watches during chain scanning.
    ///
    /// `addrs_json` is a JSON array of hex-encoded 32-byte addresses.
    /// Replaces the entire watchlist. Invalid hex entries are silently skipped.
    pub fn set_watchlist(&mut self, addrs_json: &str) {
        let addrs_str: Vec<String> = serde_json::from_str(addrs_json).unwrap_or_default();
        let mut byte_addrs = Vec::with_capacity(addrs_str.len());
        for a in addrs_str {
            let mut buf = [0u8; 32];
            if hex::decode_to_slice(&a, &mut buf).is_ok() { byte_addrs.push(buf); }
        }
        self.watchlist = byte_addrs;
    }

    /// Derive the WOTS address at a given HD index.
    ///
    /// Returns the hex-encoded 32-byte address. This is a pure computation
    /// with no side effects — the address is not cached.
    pub fn get_wots_address(&self, index: u32) -> String {
        let seed = derive_wots_seed(&self.master_seed, index as u64);
        let pk = wots::keygen(&seed);
        hex::encode(compute_address(&pk))
    }

    // ─── MSS: Full Tree Management ──────────────────────────────────────

    /// Generate a full MSS keypair and cache it in memory.
    ///
    /// This is computationally expensive: height 10 requires generating
    /// 1024 WOTS public keys (each: 18 chains × 65,535 BLAKE3 hashes).
    /// With WASM SIMD128, this takes ~20–30 seconds on desktop, 1–3 minutes
    /// on mobile.
    ///
    /// After generation, call [`export_mss_bytes`] to persist the tree to
    /// IndexedDB so that subsequent logins are instant.
    ///
    /// # Arguments
    ///
    /// * `index` — MSS HD derivation index.
    /// * `height` — Merkle tree height (10 = 1024 signatures).
    /// * `progress_cb` — Optional JS callback `(current: u32, total: u32) => void`
    ///   for UI progress reporting.
    ///
    /// # Returns
    ///
    /// The hex-encoded 32-byte MSS address (Merkle root hash).
    ///
    /// # Errors
    ///
    /// Returns `Err` if height is 0 or exceeds `MAX_HEIGHT` (20).
    pub fn get_mss_address(&mut self, index: u32, height: u32, progress_cb: Option<js_sys::Function>) -> Result<String, JsValue> {
        let seed = derive_mss_seed(&self.master_seed, index as u64);

        let kp = midstate::core::mss::keygen_with_progress(&seed, height, |current, total| {
            if let Some(cb) = &progress_cb {
                let _ = cb.call2(&JsValue::NULL, &JsValue::from(current), &JsValue::from(total));
            }
        }).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let addr = hex::encode(compute_address(&kp.master_pk));
        self.mss_cache.insert(addr.clone(), kp);
        Ok(addr)
    }

    /// Export an MSS tree as a compact binary blob for IndexedDB storage.
    ///
    /// The format is documented at the module level. For height 10, the
    /// output is ~64 KB — small enough for IndexedDB but too large for
    /// localStorage's 5 MB limit when hex-encoded into the wallet JSON.
    ///
    /// # Layout
    ///
    /// ```text
    /// [height:4][master_seed:32][next_leaf:8][master_pk:32][tree_len:4][tree:N×32]
    /// ```
    ///
    /// All multi-byte integers are little-endian.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the address is not in the WASM cache.
    pub fn export_mss_bytes(&self, address_hex: &str) -> Result<Vec<u8>, JsValue> {
        let kp = self.mss_cache.get(address_hex)
            .ok_or_else(|| JsValue::from_str("MSS tree not in cache"))?;

        let tree_len = kp.tree.len() as u32;
        let total_size = MSS_BINARY_HEADER_SIZE + kp.tree.len() * 32;
        let mut buf = Vec::with_capacity(total_size);

        buf.extend_from_slice(&kp.height.to_le_bytes());       // [0..4)
        buf.extend_from_slice(&kp.master_seed);                 // [4..36)
        buf.extend_from_slice(&kp.next_leaf.to_le_bytes());     // [36..44)
        buf.extend_from_slice(&kp.master_pk);                   // [44..76)
        buf.extend_from_slice(&tree_len.to_le_bytes());         // [76..80)
        for node in &kp.tree {
            buf.extend_from_slice(node);                        // [80..)
        }

        debug_assert_eq!(buf.len(), total_size);
        Ok(buf)
    }

    /// Import an MSS tree from a binary blob (previously exported via [`export_mss_bytes`]).
    ///
    /// After import, the tree is ready for signing — no recomputation needed.
    /// Loading a 64 KB blob takes ~1ms.
    ///
    /// # Validation
    ///
    /// - Header must be at least 80 bytes.
    /// - `tree_len` field must match the remaining data length.
    /// - The `master_pk` in the blob is trusted (it was computed during generation).
    ///
    /// # Security Note
    ///
    /// The blob contains the MSS master seed. The caller is responsible for
    /// encrypting the IndexedDB storage (the wallet uses AES-GCM via the
    /// Web Crypto API before persisting).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data is too short or truncated.
    pub fn import_mss_bytes(&mut self, address_hex: &str, data: &[u8]) -> Result<(), JsValue> {
        if data.len() < MSS_BINARY_HEADER_SIZE {
            return Err(JsValue::from_str(&format!(
                "MSS data too short: got {} bytes, need at least {}",
                data.len(), MSS_BINARY_HEADER_SIZE
            )));
        }

        let height = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let mut master_seed = [0u8; 32];
        master_seed.copy_from_slice(&data[4..36]);
        let next_leaf = u64::from_le_bytes(data[36..44].try_into().unwrap());
        let mut master_pk = [0u8; 32];
        master_pk.copy_from_slice(&data[44..76]);
        let tree_len = u32::from_le_bytes(data[76..80].try_into().unwrap()) as usize;

        let expected = MSS_BINARY_HEADER_SIZE + tree_len * 32;
        if data.len() < expected {
            return Err(JsValue::from_str(&format!(
                "MSS data truncated: got {} bytes, expected {} (tree_len={})",
                data.len(), expected, tree_len
            )));
        }

        let mut tree = vec![[0u8; 32]; tree_len];
        for i in 0..tree_len {
            let start = MSS_BINARY_HEADER_SIZE + i * 32;
            tree[i].copy_from_slice(&data[start..start + 32]);
        }

        let kp = MssKeypair { height, master_seed, tree, next_leaf, master_pk };
        self.mss_cache.insert(address_hex.to_string(), kp);
        Ok(())
    }

    /// Check whether an MSS tree is loaded in the WASM-side cache.
    ///
    /// Returns `true` if the tree is ready for signing, `false` if it
    /// needs to be loaded from IndexedDB or regenerated.
    pub fn has_mss_cache(&self, address_hex: &str) -> bool {
        self.mss_cache.contains_key(address_hex)
    }

    #[wasm_bindgen]
    pub fn get_mss_pubkey(&self, address_hex: &str) -> Option<String> {
        self.mss_cache.get(address_hex).map(|kp| hex::encode(kp.master_pk))
    }

    /// Update the next-leaf counter for an MSS tree.
    ///
    /// Called by the JS layer after loading wallet state to synchronize
    /// the WASM-side leaf counter with the persisted value.
    ///
    /// # No-op
    ///
    /// Silently does nothing if the address is not in the cache.
    pub fn set_mss_leaf_index(&mut self, address_hex: &str, leaf_index: u32) {
        if let Some(kp) = self.mss_cache.get_mut(address_hex) {
            kp.next_leaf = leaf_index as u64;
        }
    }

    /// Test a Golomb-coded compact block filter for wallet relevance.
    ///
    /// Returns `true` if any address in the watchlist matches the filter,
    /// indicating the block should be fetched and fully processed.
    ///
    /// # Returns
    ///
    /// `false` if:
    /// - The filter or block hash hex is invalid.
    /// - The watchlist is empty.
    /// - No watchlist address matches.
    pub fn check_filter(&self, filter_hex: &str, block_hash_hex: &str, n: u32) -> bool {
        let filter_data = match hex::decode(filter_hex) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut block_hash = [0u8; 32];
        if hex::decode_to_slice(block_hash_hex, &mut block_hash).is_err() { return false; }

        if self.watchlist.is_empty() { return false; }
        midstate::core::filter::match_any(&filter_data, &block_hash, n as u64, &self.watchlist)
    }

    #[wasm_bindgen]
    pub fn prepare_script_spend(
        &mut self,
        available_utxos_json: &str,
        contract_bytecode_hex: &str,
        contract_inputs_json: &str,
        outputs_json: &str,
        next_wots_index: u32,
    ) -> Result<String, JsValue> {
        // ── Parse & validate ───────────────────────────────────────────────
        let contract_bytes = hex::decode(contract_bytecode_hex)
            .map_err(|_| JsValue::from_str("Invalid contract bytecode hex"))?;
        let contract_addr = midstate::core::types::hash(&contract_bytes);

        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Bad utxos JSON: {}", e)))?;
        let in_args: Vec<ScriptInputArg> = serde_json::from_str(contract_inputs_json)
            .map_err(|e| JsValue::from_str(&format!("Bad contract inputs JSON: {}", e)))?;
        let out_args: Vec<ScriptOutputArg> = serde_json::from_str(outputs_json)
            .map_err(|e| JsValue::from_str(&format!("Bad outputs JSON: {}", e)))?;
        if in_args.is_empty() {
            return Err(JsValue::from_str("At least one contract input is required"));
        }

        // ── Contract inputs (canonical hashes + verbatim witnesses) ─────────
        let mut input_coin_ids: Vec<[u8; 32]> = Vec::new();
        let mut contract_inputs: Vec<ScriptContractInput> = Vec::new();
        let mut contract_in_value: u64 = 0;

        for a in &in_args {
            let mut salt_b = [0u8; 32];
            hex::decode_to_slice(&a.salt, &mut salt_b)
                .map_err(|_| JsValue::from_str("Invalid contract input salt hex"))?;

            // CONSENSUS: a state-thread (stateful) input must have value exactly 0.
            // apply_transaction bails on `commitment.is_some() && value != 0`. The
            // contract's spendable funds live in a SEPARATE standard coin (state None).
            if a.state.is_some() && a.value != 0 {
                return Err(JsValue::from_str(
                    "State-thread contract input must have value 0; put the contract's \
                     funds in a separate standard coin (a second input with no state).",
                ));
            }

            // Canonical coin id the node will derive from the reveal.
            let canonical = if let Some(state_hex) = &a.state {
                let mut st = [0u8; 32];
                hex::decode_to_slice(state_hex, &mut st)
                    .map_err(|_| JsValue::from_str("Invalid contract input state hex"))?;
                confidential_coin_hash(&contract_addr, &st, &salt_b)
            } else {
                compute_coin_id(&contract_addr, a.value, &salt_b)
            };

            // Sanity: the caller's coin_id must match what the node will compute.
            let mut given = [0u8; 32];
            hex::decode_to_slice(&a.coin_id, &mut given)
                .map_err(|_| JsValue::from_str("Invalid contract input coin_id hex"))?;
            if given != canonical {
                return Err(JsValue::from_str(
                    "Contract input coin_id does not match (address,value,salt,state). \
                     Wrong salt/state, or stale coin.",
                ));
            }

            input_coin_ids.push(canonical);
            contract_in_value = contract_in_value.saturating_add(a.value);
            contract_inputs.push(ScriptContractInput {
                coin_id: hex::encode(canonical),
                bytecode: contract_bytecode_hex.to_lowercase(),
                witness: normalize_witness(&a.witness)?,
                value: a.value,
                salt: a.salt.to_lowercase(),
                state: a.state.clone(),
            });
        }

        // ── Output value total (standard carry value; confidential = 0) ─────
        let total_out: u64 = out_args.iter().map(|o| o.value).sum();

        // ── Fee estimation + wallet coin selection (covers shortfall only) ──
        let mut avail_sorted = available.clone();
        avail_sorted.sort_by(|a, b| b.value.cmp(&a.value));

        let mut target_fee = 100u64;
        let mut selected: Vec<WasmUtxo> = Vec::new();
        let mut wallet_in: u64;
        let final_fee: u64;

        loop {
            let needed_total = total_out.saturating_add(target_fee);
            // Shortfall the wallet must cover after the contract's own value.
            let shortfall = needed_total.saturating_sub(contract_in_value);

            selected.clear();
            let mut selected_set = HashSet::new();
            wallet_in = 0;

            // Greedy selection up to the shortfall (may select nothing).
            for coin in &avail_sorted {
                if wallet_in >= shortfall {
                    break;
                }
                selected_set.insert(coin.coin_id.clone());
                selected.push(coin.clone());
                wallet_in += coin.value;
            }
            if wallet_in < shortfall {
                return Err(JsValue::from_str("Insufficient wallet funds to cover outputs + fee"));
            }

            // WOTS co-spend enforcement (SECURITY: never reuse a WOTS key across
            // txs — pull in every coin sharing a selected non-MSS address).
            let mut grouped = HashSet::new();
            for c in &selected {
                if !c.is_mss {
                    grouped.insert(c.address.clone());
                }
            }
            for coin in &available {
                if grouped.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected_set.insert(coin.coin_id.clone());
                    selected.push(coin.clone());
                    wallet_in += coin.value;
                }
            }
            if selected.len() > MAX_SELECTED_INPUTS {
                return Err(JsValue::from_str("Too many inputs selected for one transaction"));
            }

            // Precise size → fee. Bytes: base + wallet inputs (~1636 each) +
            // contract inputs (~120 each) + outputs (~100 each) + change outputs.
            let provisional_change =
                contract_in_value + wallet_in - total_out.min(contract_in_value + wallet_in);
            let change_for_size = (contract_in_value + wallet_in)
                .saturating_sub(total_out)
                .saturating_sub(target_fee);
            let _ = provisional_change;
            let num_change = decompose_value(change_for_size).len();
            let num_outputs = out_args.len() + num_change;
            // ── Fee sizing for contract inputs ──────────────────────────────
            // The node admits a reveal iff fee*1024 >= MIN_FEE_PER_KB * bincode_size(tx)
            // (mempool.rs). So estimated_bytes must track the REAL serialized size.
            // The dominant term per contract input is its witness, which for an HTLC
            // claim carries a full MSS signature. Its exact serialized length is the
            // same one mss.rs uses:
            //   leaf_index(8) + wots_pk(32) + wots_sig(SIG_SIZE) + auth_len(4)
            //   + auth_path(height * 32)
            let mss_sig_len: u64 = (8 + 32 + wots::SIG_SIZE + 4
                + (midstate::core::mss::DEFAULT_HEIGHT as usize) * 32) as u64;

            let contract_bytes_est: u64 = contract_inputs.iter().map(|ci| {
                let bytecode_bytes = ci.bytecode.len() as u64 / 2;
                // Witness is injected AFTER the commitment for HTLC claims, so it's
                // empty here — budget the worst case: an MSS signature plus the
                // preimage and routing-flag stack items, encoded as Vec<Vec<u8>>.
                let witness_bytes = if ci.witness.is_empty() {
                    mss_sig_len + 96
                } else {
                    let raw: u64 = ci.witness.split(',')
                        .filter(|t| !t.is_empty())
                        .map(|t| t.len() as u64 / 2)
                        .sum();
                    let items = ci.witness.split(',').filter(|t| !t.is_empty()).count() as u64;
                    raw + items * 8 + 16
                };
                bytecode_bytes + witness_bytes + 64 // input-reveal struct + bincode overhead
            }).sum();

            let estimated_bytes = 100
                + (selected.len() as u64 * 1636)
                + contract_bytes_est
                + (num_outputs as u64 * 100);
            let required_fee = (estimated_bytes * 10) / 1024 + 10;

            if contract_in_value + wallet_in >= total_out + required_fee {
                final_fee = required_fee;
                break;
            } else {
                target_fee = required_fee;
            }
        }

        // ── Build ordered outputs (user outputs first, then change) ─────────
        let mut outputs_out: Vec<serde_json::Value> = Vec::new();
        let mut output_hashes: Vec<[u8; 32]> = Vec::new();

        for o in &out_args {
            let mut addr_b = [0u8; 32];
            hex::decode_to_slice(&o.address, &mut addr_b)
                .map_err(|_| JsValue::from_str("Invalid output address hex"))?;

            // Salt: explicit if provided (recordable), else random.
            let salt_b = match &o.salt {
                Some(s) => {
                    let mut b = [0u8; 32];
                    hex::decode_to_slice(s, &mut b)
                        .map_err(|_| JsValue::from_str("Invalid output salt hex"))?;
                    b
                }
                None => {
                    let mut b = [0u8; 32];
                    getrandom_02::getrandom(&mut b).unwrap();
                    b
                }
            };

            if o.out_type == "confidential" {
                let state_hex = o
                    .state
                    .clone()
                    .ok_or_else(|| JsValue::from_str("confidential output requires a state"))?;
                let mut st = [0u8; 32];
                hex::decode_to_slice(&state_hex, &mut st)
                    .map_err(|_| JsValue::from_str("Invalid output state hex"))?;
                output_hashes.push(confidential_coin_hash(&addr_b, &st, &salt_b));
                outputs_out.push(serde_json::json!({
                    "type": "confidential",
                    "address": o.address.to_lowercase(),
                    "commitment": state_hex.to_lowercase(),
                    "salt": hex::encode(salt_b),
                }));
            } else {
                // CONSENSUS: standard outputs must be a NONZERO power of two
                // (apply_transaction bails otherwise). We deliberately do NOT
                // auto-decompose: that would shift output indices and break
                // covenants that use OP_READ_OUTPUT_STATE / OP_OUTPUT_ADDRESS.
                // The caller specifies valid denominations in contract order.
                if o.value == 0 || !o.value.is_power_of_two() {
                    return Err(JsValue::from_str(&format!(
                        "Standard output to {} has value {} — each standard output must be a \
                         nonzero power of two. Split it into power-of-two outputs yourself \
                         (order is preserved, so index-based covenants stay valid).",
                        o.address, o.value
                    )));
                }
                output_hashes.push(compute_coin_id(&addr_b, o.value, &salt_b));
                outputs_out.push(serde_json::json!({
                    "type": "standard",
                    "address": o.address.to_lowercase(),
                    "value": o.value,
                    "salt": hex::encode(salt_b),
                }));
            }
        }

        // Change → wallet (deterministic salts so we can re-find/spend it later).
        let change = (contract_in_value + wallet_in)
            .saturating_sub(total_out)
            .saturating_sub(final_fee);
        let mut current_idx = next_wots_index;
        if change > 0 {
            for denom in decompose_value(change) {
                let seed = derive_wots_seed(&self.master_seed, current_idx as u64);
                let addr = compute_address(&wots::keygen(&seed));
                let salt = derive_deterministic_salt(&self.master_seed, current_idx as u64);
                output_hashes.push(compute_coin_id(&addr, denom, &salt));
                outputs_out.push(serde_json::json!({
                    "type": "standard",
                    "address": hex::encode(addr),
                    "value": denom,
                    "salt": hex::encode(salt),
                }));
                current_idx += 1;
            }
        }

        // ── Wallet input coin ids (canonical) appended after contract ids ───
        let mut wallet_inputs: Vec<ScriptWalletInput> = Vec::new();
        for inp in &selected {
            let pk = if inp.is_mss {
                self.mss_cache
                    .get(&inp.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing from cache"))?
                    .master_pk
            } else {
                wots::keygen(&derive_wots_seed(&self.master_seed, inp.index as u64))
            };
            let p2pk_addr = midstate::core::types::hash(&midstate::core::script::compile_p2pk(&pk));
            let mut salt_b = [0u8; 32];
            hex::decode_to_slice(&inp.salt, &mut salt_b).unwrap();
            input_coin_ids.push(compute_coin_id(&p2pk_addr, inp.value, &salt_b));

            wallet_inputs.push(ScriptWalletInput {
                coin_id: inp.coin_id.clone(),
                address: inp.address.clone(),
                value: inp.value,
                salt: inp.salt.clone(),
                is_mss: inp.is_mss,
                index: inp.index,
                mss_leaf: inp.mss_leaf,
            });
        }

        // CONSENSUS: hard caps on transaction size (apply_transaction bails above these).
        if contract_inputs.len() + wallet_inputs.len() > midstate::core::types::MAX_TX_INPUTS {
            return Err(JsValue::from_str("Too many inputs for one transaction (max 256)"));
        }
        if outputs_out.len() > midstate::core::types::MAX_TX_OUTPUTS {
            return Err(JsValue::from_str(
                "Too many outputs for one transaction (max 256) — fewer/larger denominations needed",
            ));
        }

        // ── Commitment over canonical [contract…, wallet…] inputs + outputs ─
        let mut tx_salt = [0u8; 32];
        getrandom_02::getrandom(&mut tx_salt).unwrap();
        let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);

        let ctx = ScriptSpendContext {
            contract_addr: hex::encode(contract_addr),
            contract_inputs,
            wallet_inputs,
            outputs: outputs_out,
            input_coin_ids: input_coin_ids.iter().map(hex::encode).collect(),
            tx_salt: hex::encode(tx_salt),
            commitment: hex::encode(commitment),
            fee: final_fee,
            next_wots_index: current_idx,
        };
        serde_json::to_string(&ctx).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Phase 2: sign the wallet fee-inputs over the committed commitment and emit
    /// the wire `reveal` payload. Mirrors `build_reveal` but (a) splices contract
    /// witnesses verbatim, (b) hashes confidential outputs, (c) leaves contract
    /// inputs unsigned. `commitment_hex` / `salt_hex` are the ctx values returned
    /// by `prepare_script_spend` (pass ctx.commitment and ctx.tx_salt — there is
    /// no server-side salt contribution in this protocol).
    #[wasm_bindgen]
    pub fn build_script_reveal(
        &mut self,
        ctx_json: &str,
        commitment_hex: &str,
        salt_hex: &str,
    ) -> Result<String, JsValue> {
        let ctx: ScriptSpendContext =
            serde_json::from_str(ctx_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut commitment = [0u8; 32];
        hex::decode_to_slice(commitment_hex, &mut commitment)
            .map_err(|_| JsValue::from_str("Invalid commitment hex"))?;
        let mut salt_b = [0u8; 32];
        hex::decode_to_slice(salt_hex, &mut salt_b)
            .map_err(|_| JsValue::from_str("Invalid salt hex"))?;

        let mut contract_addr = [0u8; 32];
        hex::decode_to_slice(&ctx.contract_addr, &mut contract_addr).unwrap();

        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();
        let mut safety_in: Vec<[u8; 32]> = Vec::new();

        // 1) Contract inputs — reveal + verbatim witness, NOT signed by wallet.
        for ci in &ctx.contract_inputs {
            let mut salt = [0u8; 32];
            hex::decode_to_slice(&ci.salt, &mut salt).unwrap();
            if let Some(state_hex) = &ci.state {
                let mut st = [0u8; 32];
                hex::decode_to_slice(state_hex, &mut st).unwrap();
                safety_in.push(confidential_coin_hash(&contract_addr, &st, &salt));
                input_reveals.push(serde_json::json!({
                    "bytecode": ci.bytecode,
                    "value": ci.value,
                    "salt": ci.salt,
                    "commitment": state_hex,
                }));
            } else {
                safety_in.push(compute_coin_id(&contract_addr, ci.value, &salt));
                input_reveals.push(serde_json::json!({
                    "bytecode": ci.bytecode,
                    "value": ci.value,
                    "salt": ci.salt,
                }));
            }
            signatures.push(ci.witness.clone()); // comma-separated stack, verbatim
        }

        // 2) Wallet fee-inputs — P2PK reveal + real signature over the commitment.
        let mut mss_sig_cache: HashMap<String, Vec<u8>> = HashMap::new();
        for wi in &ctx.wallet_inputs {
            let (pk, sig_bytes) = if wi.is_mss {
                let kp = self
                    .mss_cache
                    .get_mut(&wi.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing from cache"))?;
                if let Some(cached) = mss_sig_cache.get(&wi.address) {
                    (kp.master_pk, cached.clone())
                } else {
                    kp.next_leaf = wi.mss_leaf as u64;
                    let sig = kp.sign(&commitment).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let b = sig.to_bytes();
                    mss_sig_cache.insert(wi.address.clone(), b.clone());
                    (kp.master_pk, b)
                }
            } else {
                let seed = derive_wots_seed(&self.master_seed, wi.index as u64);
                (wots::keygen(&seed), wots::sig_to_bytes(&wots::sign(&seed, &commitment)))
            };

            let bytecode = midstate::core::script::compile_p2pk(&pk);
            let address = midstate::core::types::hash(&bytecode);
            let mut salt = [0u8; 32];
            hex::decode_to_slice(&wi.salt, &mut salt).unwrap();
            safety_in.push(compute_coin_id(&address, wi.value, &salt));

            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&bytecode),
                "value": wi.value,
                "salt": wi.salt,
            }));
            signatures.push(hex::encode(&sig_bytes));
        }

        // 3) Outputs — pass through; recompute hashes for the safety check
        //    (standard, confidential AND data_burn — the burn branch matters for the
        //    DEX atomic announce, where prepare_fund_many embeds an MDXA burn: burns
        //    have no address/salt/value, so the old unconditional field unwraps here
        //    panicked and trapped the whole wasm module with `unreachable`).
        let mut safety_out: Vec<[u8; 32]> = Vec::new();
        for o in &ctx.outputs {
            let ty = o["type"].as_str().unwrap_or("standard");
            if ty == "data_burn" {
                // Hash exactly as prepare_spend / prepare_fund_many committed it:
                // blake3("DATABURN" ‖ value_burned_le ‖ payload).
                let payload_hex = o["payload"].as_str()
                    .ok_or_else(|| JsValue::from_str("data_burn output missing payload"))?;
                let payload = hex::decode(payload_hex)
                    .map_err(|_| JsValue::from_str("Invalid data_burn payload hex"))?;
                let value_burned = o["value_burned"].as_u64()
                    .ok_or_else(|| JsValue::from_str("data_burn output missing value_burned"))?;
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"DATABURN");
                hasher.update(&value_burned.to_le_bytes());
                hasher.update(&payload);
                safety_out.push(*hasher.finalize().as_bytes());
                continue;
            }
            let mut addr = [0u8; 32];
            hex::decode_to_slice(o["address"].as_str().unwrap_or_default(), &mut addr)
                .map_err(|_| JsValue::from_str("Invalid output address hex"))?;
            let mut salt = [0u8; 32];
            hex::decode_to_slice(o["salt"].as_str().unwrap_or_default(), &mut salt)
                .map_err(|_| JsValue::from_str("Invalid output salt hex"))?;
            if ty == "confidential" {
                let mut st = [0u8; 32];
                hex::decode_to_slice(o["commitment"].as_str().unwrap_or_default(), &mut st)
                    .map_err(|_| JsValue::from_str("Invalid output commitment hex"))?;
                safety_out.push(confidential_coin_hash(&addr, &st, &salt));
            } else {
                let value = o["value"].as_u64()
                    .ok_or_else(|| JsValue::from_str("Output missing value"))?;
                safety_out.push(compute_coin_id(&addr, value, &salt));
            }
        }

        // 4) Safety: the reveal must reconstruct the committed commitment exactly.
        if compute_commitment(&safety_in, &safety_out, &salt_b) != commitment {
            return Err(JsValue::from_str(
                "Fatal hash mismatch: reveal does not reconstruct the committed commitment.",
            ));
        }

        Ok(serde_json::json!({
            "inputs": input_reveals,
            "signatures": signatures,
            "outputs": ctx.outputs,
            "salt": salt_hex,
        })
        .to_string())
    }

/// Sign a raw commitment hash using a cached MSS key for Layer 2 Payment Channels.
    ///
    /// # Reasoning
    /// Payment channels require users to sign off-chain state updates (commitments) 
    /// without immediately broadcasting a `Reveal` transaction. This exposes the raw 
    /// MSS signature mechanism to JavaScript to facilitate trustless Hub-and-Spoke 
    /// L2 networks.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:  ∃ kp ∈ self.mss_cache.values() s.t. kp.master_pk == mss_pk_hex
    ///       commitment_hex is a valid 64-character hex string (32 bytes)
    ///       kp.remaining() > 0
    /// Post: kp.next_leaf' = kp.next_leaf + 1
    ///       result is Ok(signature_hex)
    /// ```
    ///
    /// ```zed
    ///     SignMssHex
    ///     ----------
    ///     ΔWebWallet
    ///     mss_pk_hex? : String
    ///     commitment_hex? : String
    ///     sig! : String
    ///
    ///     let kp == (μ k ∈ ran(mss_cache) | hex(k.master_pk) = mss_pk_hex?)
    ///
    ///     pre  kp exists
    ///     pre  kp.next_leaf < 2^{kp.height}
    ///     post kp'.next_leaf = kp.next_leaf + 1
    ///     post sig! = hex(sign(kp.master_seed, commitment))
    /// ```
    #[wasm_bindgen]
    pub fn sign_mss_hex(&mut self, mss_pk_hex: &str, commitment_hex: &str) -> Result<String, JsValue> {
        let mut commitment = [0u8; 32];
        hex::decode_to_slice(commitment_hex, &mut commitment)
            .map_err(|_| JsValue::from_str("Invalid commitment hex"))?;

        let mut pk = [0u8; 32];
        hex::decode_to_slice(mss_pk_hex, &mut pk)
            .map_err(|_| JsValue::from_str("Invalid PK hex"))?;

        let kp = self.mss_cache.values_mut()
            .find(|k| k.master_pk == pk)
            .ok_or_else(|| JsValue::from_str("MSS tree not found in cache. Run Network Sync."))?;

        if kp.remaining() == 0 {
            return Err(JsValue::from_str("MSS key capacity exhausted"));
        }

        // The Rust `sign` method automatically increments `kp.next_leaf` internally
        let sig = kp.sign(&commitment).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(hex::encode(sig.to_bytes()))
    }

// ── Funding a contract ──────────────────────────────────────────────────

    /// Phase 1 for FUNDING a contract. Pays `amount` to the contract address as
    /// power-of-two "value" coins, optionally seeds a confidential "state" coin,
    /// returns change to the wallet, and reuses `build_script_reveal` for phase 2
    /// (its `contract_inputs` list is simply empty here — the wallet pays).
    ///
    /// Mirrors the CLI fund instruction:  `--to addr:amount` (+ `--to addr:0:state`).
    /// `state_hex` = None for a plain value-only funding.
    #[wasm_bindgen]
    pub fn prepare_fund_tx(
        &mut self,
        available_utxos_json: &str,
        contract_addr_hex: &str,
        amount: u64,
        state_hex: Option<String>,
        next_wots_index: u32,
    ) -> Result<String, JsValue> {
        let mut contract_addr = [0u8; 32];
        hex::decode_to_slice(contract_addr_hex, &mut contract_addr)
            .map_err(|_| JsValue::from_str("Invalid contract address hex"))?;
        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Bad utxos JSON: {}", e)))?;
        if amount == 0 && state_hex.is_none() {
            return Err(JsValue::from_str("Nothing to fund: amount is 0 and no state given"));
        }

        // ── Outputs: value coins (power-of-two decomposition) + optional state ──
        let mut outputs_out: Vec<serde_json::Value> = Vec::new();
        let mut output_hashes: Vec<[u8; 32]> = Vec::new();

        for denom in decompose_value(amount) {
            let mut salt = [0u8; 32];
            getrandom_02::getrandom(&mut salt).unwrap();
            output_hashes.push(compute_coin_id(&contract_addr, denom, &salt));
            outputs_out.push(serde_json::json!({
                "type": "standard",
                "address": contract_addr_hex.to_lowercase(),
                "value": denom,
                "salt": hex::encode(salt),
            }));
        }
        if let Some(st_hex) = &state_hex {
            let mut st = [0u8; 32];
            hex::decode_to_slice(st_hex, &mut st)
                .map_err(|_| JsValue::from_str("Invalid state hex"))?;
            let mut salt = [0u8; 32];
            getrandom_02::getrandom(&mut salt).unwrap();
            output_hashes.push(confidential_coin_hash(&contract_addr, &st, &salt));
            outputs_out.push(serde_json::json!({
                "type": "confidential",
                "address": contract_addr_hex.to_lowercase(),
                "commitment": st_hex.to_lowercase(),
                "salt": hex::encode(salt),
            }));
        }

        // ── Wallet coin selection: cover amount + fee (with WOTS co-spend) ──────
        let mut avail_sorted = available.clone();
        avail_sorted.sort_by(|a, b| b.value.cmp(&a.value));
        let mut target_fee = 100u64;
        let mut selected: Vec<WasmUtxo> = Vec::new();
        let mut wallet_in: u64;
        let final_fee;
        loop {
            let needed = amount.saturating_add(target_fee);
            selected.clear();
            let mut set = HashSet::new();
            wallet_in = 0;
            for c in &avail_sorted {
                if wallet_in >= needed { break; }
                set.insert(c.coin_id.clone());
                selected.push(c.clone());
                wallet_in += c.value;
            }
            if wallet_in < needed {
                return Err(JsValue::from_str("Insufficient funds for amount + fee"));
            }
            let mut grouped = HashSet::new();
            for c in &selected { if !c.is_mss { grouped.insert(c.address.clone()); } }
            for c in &available {
                if grouped.contains(&c.address) && !set.contains(&c.coin_id) {
                    set.insert(c.coin_id.clone());
                    selected.push(c.clone());
                    wallet_in += c.value;
                }
            }
            if selected.len() > MAX_SELECTED_INPUTS {
                return Err(JsValue::from_str("Too many inputs selected"));
            }
            let change_for_size = wallet_in.saturating_sub(amount).saturating_sub(target_fee);
            let num_outputs = outputs_out.len() + decompose_value(change_for_size).len();
            let estimated = 100 + (selected.len() as u64 * 1636) + (num_outputs as u64 * 100);
            let req = (estimated * 10) / 1024 + 10;
            if wallet_in >= amount + req { final_fee = req; break; } else { target_fee = req; }
        }

        // ── Change → wallet (deterministic salts) ───────────────────────────────
        let change = wallet_in - amount - final_fee;
        let mut idx = next_wots_index;
        if change > 0 {
            for denom in decompose_value(change) {
                let seed = derive_wots_seed(&self.master_seed, idx as u64);
                let addr = compute_address(&wots::keygen(&seed));
                let salt = derive_deterministic_salt(&self.master_seed, idx as u64);
                output_hashes.push(compute_coin_id(&addr, denom, &salt));
                outputs_out.push(serde_json::json!({
                    "type": "standard",
                    "address": hex::encode(addr),
                    "value": denom,
                    "salt": hex::encode(salt),
                }));
                idx += 1;
            }
        }

        // ── Wallet input coin ids (canonical) ───────────────────────────────────
        let mut input_coin_ids: Vec<[u8; 32]> = Vec::new();
        let mut wallet_inputs: Vec<ScriptWalletInput> = Vec::new();
        for inp in &selected {
            let pk = if inp.is_mss {
                self.mss_cache.get(&inp.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing from cache"))?.master_pk
            } else {
                wots::keygen(&derive_wots_seed(&self.master_seed, inp.index as u64))
            };
            let p2pk = midstate::core::types::hash(&midstate::core::script::compile_p2pk(&pk));
            let mut salt = [0u8; 32];
            hex::decode_to_slice(&inp.salt, &mut salt).unwrap();
            input_coin_ids.push(compute_coin_id(&p2pk, inp.value, &salt));
            wallet_inputs.push(ScriptWalletInput {
                coin_id: inp.coin_id.clone(),
                address: inp.address.clone(),
                value: inp.value,
                salt: inp.salt.clone(),
                is_mss: inp.is_mss,
                index: inp.index,
                mss_leaf: inp.mss_leaf,
            });
        }

        if wallet_inputs.len() > midstate::core::types::MAX_TX_INPUTS {
            return Err(JsValue::from_str("Too many inputs (max 256)"));
        }
        if outputs_out.len() > midstate::core::types::MAX_TX_OUTPUTS {
            return Err(JsValue::from_str("Too many outputs (max 256)"));
        }

        let mut tx_salt = [0u8; 32];
        getrandom_02::getrandom(&mut tx_salt).unwrap();
        let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);

        let ctx = ScriptSpendContext {
            contract_addr: contract_addr_hex.to_lowercase(),
            contract_inputs: Vec::new(), // funding consumes no contract coin
            wallet_inputs,
            outputs: outputs_out,
            input_coin_ids: input_coin_ids.iter().map(hex::encode).collect(),
            tx_salt: hex::encode(tx_salt),
            commitment: hex::encode(commitment),
            fee: final_fee,
            next_wots_index: idx,
        };
        serde_json::to_string(&ctx).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Fund MANY contract addresses in ONE transaction.
    ///
    /// Identical to [`prepare_fund_tx`] but takes a list of `{address, amount}`
    /// fundings instead of a single address. Every funding's amount is split into
    /// power-of-two coins paid to its address; wallet inputs cover the SUM plus a
    /// size-scaled fee, with change returned to deterministic wallet addresses.
    /// Used to fund a bundle of independent limit-order covenants (one fresh
    /// secret/address each) in a single ~2-block commit/reveal rather than N of them.
    ///
    /// `fundings_json` — JSON array: `[{ "address": <64-hex>, "amount": <u64> }, ...]`.
    /// Returns the same `ScriptSpendContext` JSON as `prepare_fund_tx`; the caller
    /// recovers each covenant's coin by matching `outputs[].address`.
    #[wasm_bindgen]
    pub fn prepare_fund_many(
        &mut self,
        available_utxos_json: &str,
        fundings_json: &str,
        next_wots_index: u32,
        databurns_json: Option<String>,
    ) -> Result<String, JsValue> {
        #[derive(serde::Deserialize)]
        struct Funding {
            address: String,
            amount: u64,
            /// Optional caller-chosen coin salt (hex, 32 bytes). Lets the caller compute
            /// the resulting coin_id BEFORE this call — which is what allows the DEX to
            /// encode its MDXA announcement (salts included) and ship it via `databurns_json`
            /// so the announcement rides INSIDE the funding tx (atomic announce). Only
            /// valid when `amount` is a single power-of-two denomination; multi-denom
            /// fundings would need one salt per denom and are rejected instead of
            /// silently mis-binding. Old callers omit the field (serde default).
            #[serde(default)]
            salt: Option<String>,
        }
        let fundings: Vec<Funding> = serde_json::from_str(fundings_json)
            .map_err(|e| JsValue::from_str(&format!("Bad fundings JSON: {}", e)))?;
        if fundings.is_empty() {
            return Err(JsValue::from_str("No fundings provided"));
        }
        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Bad utxos JSON: {}", e)))?;

        // ── Outputs: per-funding power-of-two coins to each address ──────────────
        let mut outputs_out: Vec<serde_json::Value> = Vec::new();
        let mut output_hashes: Vec<[u8; 32]> = Vec::new();
        let mut total_amount: u64 = 0;

        for f in &fundings {
            let mut addr = [0u8; 32];
            hex::decode_to_slice(&f.address, &mut addr)
                .map_err(|_| JsValue::from_str("Invalid funding address hex"))?;
            if f.amount == 0 {
                return Err(JsValue::from_str("A funding amount is 0"));
            }
            total_amount = total_amount.checked_add(f.amount)
                .ok_or_else(|| JsValue::from_str("Funding total overflows u64"))?;
            let denoms = decompose_value(f.amount);
            if f.salt.is_some() && denoms.len() != 1 {
                return Err(JsValue::from_str(
                    "A per-funding salt override requires a single power-of-two amount",
                ));
            }
            for denom in denoms {
                let mut salt = [0u8; 32];
                match f.salt {
                    Some(ref s) => hex::decode_to_slice(s, &mut salt)
                        .map_err(|_| JsValue::from_str("Invalid funding salt hex (need 32 bytes)"))?,
                    None => getrandom_02::getrandom(&mut salt).unwrap(),
                }
                output_hashes.push(compute_coin_id(&addr, denom, &salt));
                outputs_out.push(serde_json::json!({
                    "type": "standard",
                    "address": f.address.to_lowercase(),
                    "value": denom,
                    "salt": hex::encode(salt),
                }));
            }
        }

        // ── Optional 0-value DataBurns riding IN the funding tx (atomic announce) ─
        // JSON array: [{"payload":"<hex>","value_burned":0}, ...]. Multiple burns let
        // large DEX bundles ship several SELF-CONTAINED MDXA announcements (each must
        // fit the node\'s MAX_BURN_DATA_SIZE) inside the same transaction. The hash of
        // each burn is consensus-identical to OutputData::hash_for_commitment:
        // blake3("DATABURN" || value_burned_le || payload). Ordering (fundings, burns,
        // change) is fixed here and mirrored by build_script_reveal iterating
        // ctx.outputs in order. burn_payload_bytes feeds the fee estimate below —
        // the mempool enforces MIN_FEE_PER_KB against ACTUAL serialized size, so
        // counting burns as flat 100-byte outputs would underpay and get rejected.
        let mut db_val: u64 = 0;
        let mut burn_payload_bytes: u64 = 0;
        if let Some(ref burns_str) = databurns_json {
            #[derive(serde::Deserialize)]
            struct BurnSpec { payload: String, #[serde(default)] value_burned: u64 }
            let burns: Vec<BurnSpec> = serde_json::from_str(burns_str)
                .map_err(|_| JsValue::from_str("Invalid databurns JSON"))?;
            for b in &burns {
                let payload = hex::decode(&b.payload)
                    .map_err(|_| JsValue::from_str("Invalid databurn payload hex"))?;
                if payload.len() > midstate::core::MAX_BURN_DATA_SIZE {
                    return Err(JsValue::from_str(&format!(
                        "DataBurn payload {}B exceeds node max {}B",
                        payload.len(), midstate::core::MAX_BURN_DATA_SIZE)));
                }
                db_val = db_val.checked_add(b.value_burned)
                    .ok_or_else(|| JsValue::from_str("Burn value overflow"))?;
                burn_payload_bytes += payload.len() as u64;
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"DATABURN");
                hasher.update(&b.value_burned.to_le_bytes());
                hasher.update(&payload);
                output_hashes.push(*hasher.finalize().as_bytes());
                outputs_out.push(serde_json::json!({
                    "type": "data_burn", "payload": b.payload, "value_burned": b.value_burned
                }));
            }
        }

        // ── Wallet coin selection: cover total_amount + fee (WOTS co-spend) ──────
        let mut avail_sorted = available.clone();
        avail_sorted.sort_by(|a, b| b.value.cmp(&a.value));
        let mut target_fee = 100u64;
        let mut selected: Vec<WasmUtxo> = Vec::new();
        let mut wallet_in: u64;
        let final_fee;
        loop {
            let needed = total_amount.saturating_add(db_val).saturating_add(target_fee);
            selected.clear();
            let mut set = HashSet::new();
            wallet_in = 0;
            for c in &avail_sorted {
                if wallet_in >= needed { break; }
                set.insert(c.coin_id.clone());
                selected.push(c.clone());
                wallet_in += c.value;
            }
            if wallet_in < needed {
                return Err(JsValue::from_str("Insufficient funds for fundings + fee"));
            }
            let mut grouped = HashSet::new();
            for c in &selected { if !c.is_mss { grouped.insert(c.address.clone()); } }
            for c in &available {
                if grouped.contains(&c.address) && !set.contains(&c.coin_id) {
                    set.insert(c.coin_id.clone());
                    selected.push(c.clone());
                    wallet_in += c.value;
                }
            }
            if selected.len() > MAX_SELECTED_INPUTS {
                return Err(JsValue::from_str("Too many inputs selected"));
            }
            // outputs_out already includes the optional data_burn, so it is counted here.
            let change_for_size = wallet_in.saturating_sub(total_amount).saturating_sub(db_val).saturating_sub(target_fee);
            let num_outputs = outputs_out.len() + decompose_value(change_for_size).len();
            let estimated = 100 + (selected.len() as u64 * 1636) + (num_outputs as u64 * 100) + burn_payload_bytes;
            let req = (estimated * 10) / 1024 + 10;
            if wallet_in >= total_amount + db_val + req { final_fee = req; break; } else { target_fee = req; }
        }

        // ── Change → wallet (deterministic salts) ───────────────────────────────
        let change = wallet_in - total_amount - db_val - final_fee;
        let mut idx = next_wots_index;
        if change > 0 {
            for denom in decompose_value(change) {
                let seed = derive_wots_seed(&self.master_seed, idx as u64);
                let addr = compute_address(&wots::keygen(&seed));
                let salt = derive_deterministic_salt(&self.master_seed, idx as u64);
                output_hashes.push(compute_coin_id(&addr, denom, &salt));
                outputs_out.push(serde_json::json!({
                    "type": "standard",
                    "address": hex::encode(addr),
                    "value": denom,
                    "salt": hex::encode(salt),
                }));
                idx += 1;
            }
        }

        // ── Wallet input coin ids (canonical) ───────────────────────────────────
        let mut input_coin_ids: Vec<[u8; 32]> = Vec::new();
        let mut wallet_inputs: Vec<ScriptWalletInput> = Vec::new();
        for inp in &selected {
            let pk = if inp.is_mss {
                self.mss_cache.get(&inp.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing from cache"))?.master_pk
            } else {
                wots::keygen(&derive_wots_seed(&self.master_seed, inp.index as u64))
            };
            let p2pk = midstate::core::types::hash(&midstate::core::script::compile_p2pk(&pk));
            let mut salt = [0u8; 32];
            hex::decode_to_slice(&inp.salt, &mut salt).unwrap();
            input_coin_ids.push(compute_coin_id(&p2pk, inp.value, &salt));
            wallet_inputs.push(ScriptWalletInput {
                coin_id: inp.coin_id.clone(),
                address: inp.address.clone(),
                value: inp.value,
                salt: inp.salt.clone(),
                is_mss: inp.is_mss,
                index: inp.index,
                mss_leaf: inp.mss_leaf,
            });
        }

        if wallet_inputs.len() > midstate::core::types::MAX_TX_INPUTS {
            return Err(JsValue::from_str("Too many inputs (max 256)"));
        }
        if outputs_out.len() > midstate::core::types::MAX_TX_OUTPUTS {
            return Err(JsValue::from_str("Too many outputs (max 256) — post fewer units per bundle"));
        }

        let mut tx_salt = [0u8; 32];
        getrandom_02::getrandom(&mut tx_salt).unwrap();
        let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);

        let ctx = ScriptSpendContext {
            // Multi-address funding consumes NO contract coin, so there is no single
            // contract address. build_script_reveal still hex-decodes this field, so
            // give it a valid (all-zero) placeholder; it is never otherwise used here.
            contract_addr: hex::encode([0u8; 32]),
            contract_inputs: Vec::new(),
            wallet_inputs,
            outputs: outputs_out,
            input_coin_ids: input_coin_ids.iter().map(hex::encode).collect(),
            tx_salt: hex::encode(tx_salt),
            commitment: hex::encode(commitment),
            fee: final_fee,
            next_wots_index: idx,
        };
        serde_json::to_string(&ctx).map_err(|e| JsValue::from_str(&e.to_string()))
    }


    /// Prepare a Consolidate transaction (dust sweeping) for the Web Wallet.
    ///
    /// # Reasoning
    /// Standard transactions (`prepare_spend`) budget for a 1.5 KB WOTS/MSS signature 
    /// *per input*. For dust sweeping (e.g., 100+ inputs), this overestimates the fee 
    /// massively, leading to false "Insufficient funds" errors. A `Consolidate` 
    /// transaction mathematically requires only *one* signature for the entire batch 
    /// of inputs (as long as they share the same address). This function applies 
    /// the heavily discounted single-signature fee calculation, enabling users to 
    /// sweep thousands of dust UTXOs affordably.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - available_utxos contains ≥ 2 UTXOs.
    ///   - All UTXOs in available_utxos share the exact same address.
    ///   - The sum of UTXO values > calculated_fee.
    ///
    /// Post:
    ///   result = Ok(ctx_json) ⇒
    ///     ctx_json.fee is calculated based on a 1-signature size budget.
    ///     ctx_json.outputs contains power-of-2 denominations of (total - fee) at dest_address.
    ///   result = Err(_) ⇒ state unchanged.
    /// ```
    ///
    /// ```zed
    ///     PrepareConsolidate
    ///     ------------------
    ///     ΔWebWallet
    ///     available? : seq WasmUtxo
    ///     dest_address? : String
    ///     next_wots_index? : ℕ₃₂
    ///     ctx! : String
    ///
    ///     pre  #available? ≥ 2
    ///     pre  ∀ u, v ∈ available? • u.address = v.address
    ///     let total = ∑ u ∈ available? • u.value
    ///     let fee = (((600 + 3000 + 100 + #available? * 125) * 10) / 1024) + 20
    ///     pre  total > fee
    ///     post ctx! = JSON(SpendContext)
    /// ```
    ///
    /// # Safety / Invariants
    /// - Output values strictly conform to consensus power-of-2 requirements via `decompose_value`.
    /// - Inputs are verified to share the same address to satisfy the `Transaction::Consolidate` rule.
    #[wasm_bindgen]
    pub fn prepare_consolidate(
        &mut self,
        available_utxos_json: &str,
        dest_address_hex: &str,
        next_wots_index: u32,
    ) -> Result<String, JsValue> {
        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Bad utxos JSON: {}", e)))?;

        if available.len() < 2 {
            return Err(JsValue::from_str("Need at least 2 UTXOs to consolidate"));
        }

        let first_addr = &available[0].address;
        let mut total = 0u64;
        let mut input_coin_ids: Vec<[u8; 32]> = Vec::new();

        for u in &available {
            if &u.address != first_addr {
                return Err(JsValue::from_str("All UTXOs must share the same address to consolidate"));
            }
            total += u.value;
            let mut cid = [0u8; 32];
            hex::decode_to_slice(&u.coin_id, &mut cid).unwrap();
            input_coin_ids.push(cid);
        }

        // Exact CLI math: 1 Signature + inputs
        let estimated_bytes = 600 + 3000 + 100 + (available.len() as u64 * 125);
        let final_fee = (estimated_bytes * 10) / 1024 + 20; // 20 units padding

        if total <= final_fee {
            return Err(JsValue::from_str(&format!("Total value {} is too low to pay the network fee of {}", total, final_fee)));
        }

        let out_val = total - final_fee;
        
        let mut dest_addr_b = [0u8; 32];
        hex::decode_to_slice(dest_address_hex, &mut dest_addr_b)
            .map_err(|_| JsValue::from_str("Invalid destination address"))?;

        let mut outputs_json = Vec::new();
        let mut output_hashes = Vec::new();

        for denom in decompose_value(out_val) {
            let mut salt = [0u8; 32];
            getrandom_02::getrandom(&mut salt).unwrap();
            output_hashes.push(compute_coin_id(&dest_addr_b, denom, &salt));
            outputs_json.push(serde_json::json!({
                "type": "standard",
                "address": dest_address_hex.to_lowercase(),
                "value": denom,
                "salt": hex::encode(salt),
            }));
        }

        // Shuffle outputs for privacy
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..outputs_json.len()).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut shuffled_json = Vec::with_capacity(indices.len());
        let mut shuffled_hashes = Vec::with_capacity(indices.len());
        for &idx in &indices {
            shuffled_json.push(outputs_json[idx].clone());
            shuffled_hashes.push(output_hashes[idx]);
        }

        let mut tx_salt = [0u8; 32];
        getrandom_02::getrandom(&mut tx_salt).unwrap();
        let commitment = compute_commitment(&input_coin_ids, &shuffled_hashes, &tx_salt);

        let ctx = SpendContext {
            selected_inputs: available,
            outputs: shuffled_json,
            commit_payload: serde_json::json!({
                "coins": input_coin_ids.iter().map(hex::encode).collect::<Vec<_>>(),
                "destinations": shuffled_hashes.iter().map(hex::encode).collect::<Vec<_>>()
            }),
            tx_salt: hex::encode(tx_salt),
            commitment: hex::encode(commitment),
            fee: final_fee,
            next_wots_index,
        };

        Ok(serde_json::to_string(&ctx).map_err(|e| JsValue::from_str(&e.to_string()))?)
    }


    /// Assemble the Reveal payload for a Consolidate transaction.
    ///
    /// # Reasoning
    /// Standard `build_reveal` generates a 1.5 KB signature for *every* input. For 5000+ dust 
    /// UTXOs, computing 5000 WOTS signatures requires billions of BLAKE3 hashes (freezing 
    /// the browser for 10+ seconds) and generates megabytes of useless signature data. 
    /// A Consolidate transaction strictly requires only ONE signature covering all inputs. 
    /// This function bypasses the redundant signing, keeping the browser lightning fast.
    ///
    /// # Formal Specification
    /// ```text
    /// Pre:
    ///   - ctx_json is a valid SpendContext.
    ///   - ctx_json.selected_inputs is not empty.
    ///
    /// Post:
    ///   result = Ok(reveal_json) ⇒
    ///     reveal_json.signatures contains EXACTLY ONE signature (the first input's signature).
    ///     reveal_json.inputs contains all inputs without signatures.
    /// ```
    #[wasm_bindgen]
    pub fn build_consolidate_reveal(
        &mut self,
        ctx_json: &str,
    ) -> Result<String, JsValue> {
        let ctx: SpendContext = serde_json::from_str(ctx_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if ctx.selected_inputs.is_empty() {
            return Err(JsValue::from_str("No inputs to consolidate"));
        }

        let mut commitment = [0u8; 32];
        hex::decode_to_slice(&ctx.commitment, &mut commitment)
            .map_err(|_| JsValue::from_str("Invalid commitment hex"))?;

        let mut input_reveals = Vec::new();
        for inp in &ctx.selected_inputs {
            let pk = if inp.is_mss {
                self.mss_cache.get(&inp.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing"))?.master_pk
            } else {
                wots::keygen(&derive_wots_seed(&self.master_seed, inp.index as u64))
            };
            let bytecode = midstate::core::script::compile_p2pk(&pk);
            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&bytecode),
                "value": inp.value,
                "salt": inp.salt,
            }));
        }

        // CRITICAL: SIGN ONLY THE VERY FIRST INPUT!
        let first_inp = &ctx.selected_inputs[0];
        let sig_bytes = if first_inp.is_mss {
            let kp = self.mss_cache.get_mut(&first_inp.address).unwrap();
            kp.next_leaf = first_inp.mss_leaf as u64;
            let sig = kp.sign(&commitment).map_err(|e| JsValue::from_str(&e.to_string()))?;
            sig.to_bytes()
        } else {
            let seed = derive_wots_seed(&self.master_seed, first_inp.index as u64);
            let wots_sig = wots::sign(&seed, &commitment);
            wots::sig_to_bytes(&wots_sig)
        };

        Ok(serde_json::json!({
            "inputs": input_reveals,
            "signatures": [hex::encode(sig_bytes)],
            "outputs": ctx.outputs,
            "salt": ctx.tx_salt,
        }).to_string())
    }


    /// This implements the full coin selection algorithm:
    ///
    /// 1. **Greedy selection**: picks largest coins first until the amount + fee is covered.
    /// 2. **WOTS co-spend grouping**: pulls in all coins at the same WOTS address
    ///    (required by the one-time signature security model).
    /// 3. **Snowball merge**: opportunistically pulls in coins matching change
    ///    denominations to consolidate the UTXO set.
    /// 4. **Fee estimation**: iterates until the fee estimate stabilizes.
    ///
    /// # Arguments
    ///
    /// * `available_utxos_json` — JSON array of [`WasmUtxo`] objects.
    /// * `to_address_hex` — recipient's 64-char or 72-char (checksummed) hex address.
    /// * `send_amount` — total value to send.
    /// * `next_wots_index` — current WOTS HD derivation counter.
    ///
    /// # Returns
    ///
    /// JSON string of [`SpendContext`] containing selected inputs, outputs,
    /// commitment, and fee — everything needed for commit and reveal.
    ///
    /// # Errors
    ///
    /// - `"Insufficient funds."` — UTXO values don't cover amount + fee.
    /// - `"MSS signing key not loaded."` — an MSS-backed UTXO's tree is missing.
    ///   The user should run a Network Sync to trigger cache loading.
    /// Selects coins and builds a transaction for a specified send amount.
    ///
    /// This is a complex state-machine loop that balances three competing goals:
    /// 1. Security: Strictly enforcing the "One-Time Signature" (WOTS) co-spend rule.
    /// 2. Efficiency: Consolidating fragmented UTXOs into larger denominations (Snowball Merge).
    /// 3. Restorability: Ensuring all change coins are discoverable from the seed phrase.
    pub fn prepare_spend(
        &mut self, 
        available_utxos_json: &str, 
        to_address_hex: &str, 
        send_amount: u64, 
        next_wots_index: u32,
        databurn_hex: Option<String>,
        databurn_value: Option<u64>
    ) -> Result<String, JsValue> {
        let available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse UTXOs: {}", e)))?;

        for utxo in &available {
            if utxo.is_mss && !self.mss_cache.contains_key(&utxo.address) {
                return Err(JsValue::from_str("MSS signing key not loaded. Please run a Network Sync first."));
            }
        }

        // Support empty address 
        let recipient_addr = if to_address_hex.is_empty() {
            [0u8; 32]
        } else {
            parse_address_wasm(to_address_hex)?
        };
        let recipient_hex = hex::encode(recipient_addr);

        let mut avail_sorted = available.clone();
        avail_sorted.sort_by(|a, b| b.value.cmp(&a.value));

        let mut target_fee = 100u64;
        let db_val = databurn_value.unwrap_or(0);

        loop {
            let needed = send_amount + db_val + target_fee;
            let mut selected = Vec::new();
            let mut selected_set = HashSet::new();
            let mut total = 0u64;

            for coin in &avail_sorted {
                if total >= needed { break; }
                selected_set.insert(coin.coin_id.clone());
                selected.push(coin.clone());
                total += coin.value;
            }

            if total < needed { return Err(JsValue::from_str("Insufficient funds.")); }

            let mut grouped_addresses = HashSet::new();
            for c in &selected { if !c.is_mss { grouped_addresses.insert(c.address.clone()); } }
            for coin in &avail_sorted {
                if grouped_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected_set.insert(coin.coin_id.clone());
                    selected.push(coin.clone());
                    total += coin.value;
                }
            }

            let mut added_new = true;
            while added_new {
                added_new = false;
                let current_change = total.saturating_sub(send_amount).saturating_sub(db_val).saturating_sub(target_fee);
                for denom in decompose_value(current_change) {
                    if let Some(pos) = avail_sorted.iter().position(|c| c.value == denom && !selected_set.contains(&c.coin_id)) {
                        let coin_to_add = avail_sorted[pos].clone();
                        selected_set.insert(coin_to_add.coin_id.clone());
                        selected.push(coin_to_add);
                        total += denom;
                        added_new = true;
                        break;
                    }
                }
                if selected.len() >= MAX_SELECTED_INPUTS { break; }
            }

            let mut final_addresses = HashSet::new();
            for c in &selected { if !c.is_mss { final_addresses.insert(c.address.clone()); } }
            for coin in &avail_sorted {
                if final_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected_set.insert(coin.coin_id.clone());
                    selected.push(coin.clone());
                    total += coin.value;
                }
            }

            let mut num_outputs = if send_amount > 0 { decompose_value(send_amount).len() } else { 0 };
            if databurn_hex.is_some() { num_outputs += 1; }
            let final_change_val = total.saturating_sub(send_amount).saturating_sub(db_val).saturating_sub(target_fee);
            num_outputs += decompose_value(final_change_val).len();

            let estimated_bytes = 100 + (selected.len() as u64 * 1636) + (num_outputs as u64 * 100)
                + databurn_hex.as_ref().map(|h| (h.len() / 2) as u64).unwrap_or(0);
            let required_fee = (estimated_bytes * 10) / 1024 + 10;

            if total >= send_amount + db_val + required_fee {
                let final_fee = required_fee;
                let actual_change = total - send_amount - db_val - final_fee;
                let mut final_outputs_json = Vec::new();
                let mut output_hashes = Vec::new();

                if send_amount > 0 {
                    for denom in decompose_value(send_amount) {
                        let mut salt = [0u8; 32]; getrandom_02::getrandom(&mut salt).unwrap();
                        output_hashes.push(compute_coin_id(&recipient_addr, denom, &salt));
                        final_outputs_json.push(serde_json::json!({
                            "type": "standard", "address": recipient_hex.clone(), "value": denom, "salt": hex::encode(salt)
                        }));
                    }
                }

                if let Some(ref hex_str) = databurn_hex {
                    let payload = hex::decode(hex_str).map_err(|_| JsValue::from_str("Invalid databurn hex"))?;
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"DATABURN");
                    hasher.update(&db_val.to_le_bytes());
                    hasher.update(&payload);
                    output_hashes.push(*hasher.finalize().as_bytes());
                    final_outputs_json.push(serde_json::json!({
                        "type": "data_burn", "payload": hex_str, "value_burned": db_val
                    }));
                }

                let mut current_wots_idx = next_wots_index;
                if actual_change > 0 {
                    for denom in decompose_value(actual_change) {
                        let change_seed = derive_wots_seed(&self.master_seed, current_wots_idx as u64);
                        let change_addr = compute_address(&wots::keygen(&change_seed));
                        let salt = derive_deterministic_salt(&self.master_seed, current_wots_idx as u64);
                        
                        output_hashes.push(compute_coin_id(&change_addr, denom, &salt));
                        final_outputs_json.push(serde_json::json!({
                            "type": "standard", "address": hex::encode(change_addr), "value": denom, "salt": hex::encode(salt)
                        }));
                        current_wots_idx += 1;
                    }
                }

                use rand::seq::SliceRandom;
                let mut indices: Vec<usize> = (0..final_outputs_json.len()).collect();
                indices.shuffle(&mut rand::thread_rng());

                let mut shuffled_json = Vec::with_capacity(indices.len());
                let mut shuffled_hashes = Vec::with_capacity(indices.len());
                for &idx in &indices {
                    shuffled_json.push(final_outputs_json[idx].clone());
                    shuffled_hashes.push(output_hashes[idx]);
                }

                let mut input_coin_ids = Vec::new();
                for inp in &selected {
                    let mut buf = [0u8; 32]; hex::decode_to_slice(&inp.coin_id, &mut buf).unwrap();
                    input_coin_ids.push(buf);
                }

                let mut tx_salt = [0u8; 32]; getrandom_02::getrandom(&mut tx_salt).unwrap();
                let commitment = compute_commitment(&input_coin_ids, &shuffled_hashes, &tx_salt);

                let ctx = SpendContext {
                    selected_inputs: selected,
                    outputs: shuffled_json,
                    commit_payload: serde_json::json!({
                        "coins": input_coin_ids.iter().map(hex::encode).collect::<Vec<_>>(),
                        "destinations": shuffled_hashes.iter().map(hex::encode).collect::<Vec<_>>()
                    }),
                    tx_salt: hex::encode(tx_salt),
                    commitment: hex::encode(commitment),
                    fee: final_fee,
                    next_wots_index: current_wots_idx,
                };

                return Ok(serde_json::to_string(&ctx).unwrap());
            } else {
                target_fee = required_fee;
            }
        }
    }

    /// Build the reveal payload (inputs + signatures + outputs) for a committed transaction.
    ///
    /// # Safety Check
    ///
    /// Before returning, the function recomputes the commitment hash from the
    /// generated reveals and verifies it matches the server commitment. This
    /// catches any internal payload tracking errors that would cause the
    /// transaction to be rejected on-chain.
    ///
    /// # MSS Signature Caching
    ///
    /// When multiple UTXOs share the same MSS address, only one MSS leaf is
    /// consumed. The signature is cached and reused for all UTXOs at that
    /// address within the same transaction (they all sign the same commitment,
    /// so the signatures are identical).
    ///
    /// # Errors
    ///
    /// - `"MSS tree missing from cache."` — a required MSS tree wasn't loaded.
    /// - `"Fatal Hash Mismatch!"` — internal consistency check failed.
    pub fn build_reveal(&mut self, spend_context_json: &str, server_commitment_hex: &str, server_salt_hex: &str) -> Result<String, JsValue> {
        let ctx: SpendContext = serde_json::from_str(spend_context_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut commitment = [0u8; 32];
        hex::decode_to_slice(server_commitment_hex, &mut commitment)
            .map_err(|_| JsValue::from_str("Invalid server commitment hex"))?;

        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();
        let mut safety_input_hashes = Vec::new();
        let mut mss_sig_cache: HashMap<String, Vec<u8>> = HashMap::new();

        for inp in ctx.selected_inputs {
            let (pk, sig_bytes) = if inp.is_mss {
                let kp = self.mss_cache.get_mut(&inp.address)
                    .ok_or_else(|| JsValue::from_str("MSS tree missing from cache."))?;
                if let Some(cached_sig) = mss_sig_cache.get(&inp.address) {
                    (kp.master_pk, cached_sig.clone())
                } else {
                    kp.next_leaf = inp.mss_leaf as u64;
                    let sig = kp.sign(&commitment).map_err(|e| JsValue::from_str(&e.to_string()))?;
                    let sig_bytes = sig.to_bytes();
                    mss_sig_cache.insert(inp.address.clone(), sig_bytes.clone());
                    (kp.master_pk, sig_bytes)
                }
            } else {
                let seed = derive_wots_seed(&self.master_seed, inp.index as u64);
                let wots_pk = wots::keygen(&seed);
                let wots_sig = wots::sign(&seed, &commitment);
                (wots_pk, wots::sig_to_bytes(&wots_sig))
            };

            let bytecode = midstate::core::script::compile_p2pk(&pk);
            let address = midstate::core::types::hash(&bytecode);
            let mut salt_bytes = [0u8; 32]; hex::decode_to_slice(&inp.salt, &mut salt_bytes).unwrap();
            safety_input_hashes.push(compute_coin_id(&address, inp.value, &salt_bytes));

            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&bytecode),
                "value": inp.value,
                "salt": inp.salt
            }));
            signatures.push(hex::encode(&sig_bytes));
        }

        let mut output_json = Vec::new();
        let mut safety_output_hashes = Vec::new();

        for o_val in ctx.outputs {
            if o_val["type"] == "data_burn" {
                let payload = hex::decode(o_val["payload"].as_str().unwrap()).unwrap();
                let value_burned = o_val["value_burned"].as_u64().unwrap();
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"DATABURN");
                hasher.update(&value_burned.to_le_bytes());
                hasher.update(&payload);
                safety_output_hashes.push(*hasher.finalize().as_bytes());
                output_json.push(o_val);
            } else if o_val["type"] == "standard" {
                let addr_bytes = parse_address_wasm(o_val["address"].as_str().unwrap())?;
                let mut salt_bytes = [0u8; 32]; hex::decode_to_slice(o_val["salt"].as_str().unwrap(), &mut salt_bytes).unwrap();
                let value = o_val["value"].as_u64().unwrap();
                safety_output_hashes.push(compute_coin_id(&addr_bytes, value, &salt_bytes));
                output_json.push(o_val);
            }
        }

        let mut server_salt = [0u8; 32]; hex::decode_to_slice(server_salt_hex, &mut server_salt).unwrap();
        let safety_check_commitment = compute_commitment(&safety_input_hashes, &safety_output_hashes, &server_salt);

        if safety_check_commitment != commitment {
            return Err(JsValue::from_str("Fatal Hash Mismatch! The commitment recomputed from reveals does not match."));
        }

        Ok(serde_json::json!({
            "inputs": input_reveals,
            "signatures": signatures,
            "outputs": output_json,
            "salt": server_salt_hex
        }).to_string())
    }
}

/// Hash a hex-encoded byte string with BLAKE3.
/// Returns the 32-byte hash as a 64-character hex string.
/// Used by the IDE to generate P2SH addresses.
#[wasm_bindgen]
pub fn blake3_hash_hex(hex_data: &str) -> String {
    let bytes = hex::decode(hex_data).unwrap_or_default();
    let h = midstate::core::types::hash(&bytes);
    hex::encode(h)
}


/// Compute a transaction commitment hash directly from WASM.
///
/// # Formal Specification
/// ```text
/// Pre:  input_ids_json and output_hashes_json are valid JSON arrays of 64-char hex strings.
///       salt_hex is a valid 64-character hex string.
/// Post: result = BLAKE3(MAGIC || len(inputs) || inputs || len(outputs) || outputs || salt)
/// ```
#[wasm_bindgen]
pub fn compute_commitment_hex(input_ids_json: &str, output_hashes_json: &str, salt_hex: &str) -> Result<String, JsValue> {
    let inputs: Vec<String> = serde_json::from_str(input_ids_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid inputs JSON: {}", e)))?;
    let outputs: Vec<String> = serde_json::from_str(output_hashes_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid outputs JSON: {}", e)))?;
    
    let in_bytes: Vec<[u8; 32]> = inputs.iter().map(|s| { 
        let mut b = [0u8; 32]; hex::decode_to_slice(s, &mut b).unwrap(); b 
    }).collect();
    
    let out_bytes: Vec<[u8; 32]> = outputs.iter().map(|s| { 
        let mut b = [0u8; 32]; hex::decode_to_slice(s, &mut b).unwrap(); b 
    }).collect();
    
    let mut salt = [0u8; 32];
    hex::decode_to_slice(salt_hex, &mut salt).map_err(|_| JsValue::from_str("Invalid salt hex"))?;
    
    let commitment = midstate::core::types::compute_commitment(&in_bytes, &out_bytes, &salt);
    Ok(hex::encode(commitment))
}

/// Mine the Anti-Spam Proof of Work for a P2P Chat Message directly in the browser.
///
/// # Reasoning
/// Pushing PoW to the client prevents node CPU exhaustion and enables true 
/// decentralized P2P dApps (like L2 Lightning Hubs) over WebRTC without relying
/// on central RPC servers to mine on the user's behalf.
///
/// # Formal Specification
/// ```text
/// Pre:  sender is a valid PeerId string
///       words_json is a JSON array of u8 (0-255)
///       attachments_json is a JSON array of valid ChatAttachment objects
/// Post: result = Ok(nonce) where verify_chat_pow_v2(..., nonce) == true
/// ```
#[wasm_bindgen]
pub fn mine_chat_pow_v2_wasm(
    sender: &str,
    timestamp: u64,
    reply_to_json: &str, // e.g. "null" or "42"
    words_json: &str,    // e.g. "[42, 81, 200]"
    attachments_json: &str, // e.g. '[{"kind":"signature", "value":"hex..."}]'
) -> Result<u64, JsValue> {
    let reply_to: Option<u64> = serde_json::from_str(reply_to_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid reply_to JSON: {}", e)))?;
        
    let words: Vec<u8> = serde_json::from_str(words_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid words JSON: {}", e)))?;
        
    let attachments: Vec<midstate::chat::ChatAttachment> = serde_json::from_str(attachments_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid attachments JSON: {}", e)))?;

    // The Rust miner function handles the heavy lifting
    let nonce = midstate::chat::mine_chat_pow_v2(
        sender.to_string(),
        timestamp,
        reply_to,
        words,
        attachments,
    );

    Ok(nonce)
}

#[derive(serde::Deserialize)]
pub struct HtlcDef {
    pub amount: u64,
    pub timeout: u64,
    pub receiver_is_alice: bool,
    pub secret_hash: String,
}

#[wasm_bindgen]
pub fn build_channel_state(
    channel_coin_id_hex: &str,
    alice_pk_hex: &str,
    bob_pk_hex: &str,
    alice_amount: u64,
    bob_amount: u64,
    nonce: u32,
    htlcs_json: &str,
) -> Result<String, JsValue> {
    let mut channel_coin_id = [0u8; 32];
    hex::decode_to_slice(channel_coin_id_hex, &mut channel_coin_id).unwrap();

    let mut alice_pk = [0u8; 32];
    hex::decode_to_slice(alice_pk_hex, &mut alice_pk).unwrap();

    let mut bob_pk = [0u8; 32];
    hex::decode_to_slice(bob_pk_hex, &mut bob_pk).unwrap();

    let mut output_hashes = Vec::new();
    let mut outputs_json = Vec::new();

    let alice_addr = compute_address(&alice_pk);
    for (i, denom) in decompose_value(alice_amount).into_iter().enumerate() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&channel_coin_id);
        hasher.update(&nonce.to_le_bytes());
        hasher.update(b"ALICE");
        hasher.update(&(i as u32).to_le_bytes());
        let salt = *hasher.finalize().as_bytes();

        output_hashes.push(compute_coin_id(&alice_addr, denom, &salt));
        outputs_json.push(serde_json::json!({
            "type": "standard",
            "address": hex::encode(alice_addr),
            "value": denom,
            "salt": hex::encode(salt)
        }));
    }

    let bob_addr = compute_address(&bob_pk);
    for (i, denom) in decompose_value(bob_amount).into_iter().enumerate() {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&channel_coin_id);
        hasher.update(&nonce.to_le_bytes());
        hasher.update(b"BOB");
        hasher.update(&(i as u32).to_le_bytes());
        let salt = *hasher.finalize().as_bytes();

        output_hashes.push(compute_coin_id(&bob_addr, denom, &salt));
        outputs_json.push(serde_json::json!({
            "type": "standard",
            "address": hex::encode(bob_addr),
            "value": denom,
            "salt": hex::encode(salt)
        }));
    }

    let htlcs: Vec<HtlcDef> = serde_json::from_str(htlcs_json).unwrap_or_default();
    for (i, h) in htlcs.into_iter().enumerate() {
        let mut secret_hash = [0u8; 32];
        hex::decode_to_slice(&h.secret_hash, &mut secret_hash).unwrap();
        
        let receiver_pk = if h.receiver_is_alice { &alice_pk } else { &bob_pk };
        let refund_pk = if h.receiver_is_alice { &bob_pk } else { &alice_pk };
        
        let script = midstate::core::script::compile_htlc(&secret_hash, receiver_pk, h.timeout, refund_pk);
        let htlc_addr = midstate::core::types::hash(&script);
        
        for (j, denom) in decompose_value(h.amount).into_iter().enumerate() {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&channel_coin_id);
            hasher.update(&nonce.to_le_bytes());
            hasher.update(b"HTLC");
            hasher.update(&(i as u32).to_le_bytes());
            hasher.update(&(j as u32).to_le_bytes());
            let salt = *hasher.finalize().as_bytes();

            output_hashes.push(compute_coin_id(&htlc_addr, denom, &salt));
            outputs_json.push(serde_json::json!({
                "type": "standard",
                "address": hex::encode(htlc_addr),
                "value": denom,
                "salt": hex::encode(salt)
            }));
        }
    }

    let mut salt_hasher = blake3::Hasher::new();
    salt_hasher.update(b"channel_state_salt");
    salt_hasher.update(&nonce.to_le_bytes());
    let tx_salt = *salt_hasher.finalize().as_bytes();

    let mut hasher = blake3::Hasher::new();
    hasher.update(midstate::core::types::NETWORK_MAGIC);
    hasher.update(&(1u32).to_le_bytes()); 
    hasher.update(&channel_coin_id);
    hasher.update(&(output_hashes.len() as u32).to_le_bytes());
    for h in &output_hashes {
        hasher.update(h);
    }
    hasher.update(&tx_salt);

    let commitment = *hasher.finalize().as_bytes();

    let result = serde_json::json!({
        "commitment": hex::encode(commitment),
        "outputs": outputs_json,
        "salt": hex::encode(tx_salt)
    });

    Ok(result.to_string())
}

#[wasm_bindgen]
pub fn build_multisig_2of2_address(pk1_hex: &str, pk2_hex: &str) -> String {
    let mut pk1 = [0u8; 32]; hex::decode_to_slice(pk1_hex, &mut pk1).unwrap();
    let mut pk2 = [0u8; 32]; hex::decode_to_slice(pk2_hex, &mut pk2).unwrap();
    // FIX: a real 2-of-2 (two CHECKSIG slots). The old compile_multisig_2of3(pk1,pk2,pk2)
    // needed a 3-item witness, but the channel close only ever supplies two signatures,
    // so the spend underflowed and channels could never cooperatively close.
    let script = midstate::core::script::compile_multisig_2of2(&pk1, &pk2);
    let addr = midstate::core::types::hash(&script);
    hex::encode(addr)
}

#[wasm_bindgen]
pub fn build_channel_reveal(
    channel_value: u64,
    channel_salt_hex: &str,
    alice_pk_hex: &str,
    bob_pk_hex: &str,
    state_json: &str,
    alice_sig_hex: &str,
    bob_sig_hex: &str,
) -> Result<String, JsValue> {
    let state: serde_json::Value = serde_json::from_str(state_json).unwrap();
    let mut pk1 = [0u8; 32]; hex::decode_to_slice(alice_pk_hex, &mut pk1).unwrap();
    let mut pk2 = [0u8; 32]; hex::decode_to_slice(bob_pk_hex, &mut pk2).unwrap();
    // FIX: must match build_multisig_2of2_address — the channel address is hash(script),
    // so both call sites have to build the SAME 2-of-2 script or the funded address and
    // the spend bytecode diverge.
    let script = midstate::core::script::compile_multisig_2of2(&pk1, &pk2);

    let input = serde_json::json!({
        "bytecode": hex::encode(script),
        "value": channel_value,
        "salt": channel_salt_hex,
    });

    let sigs = format!("{},{}", alice_sig_hex, bob_sig_hex);

    let reveal = serde_json::json!({
        "inputs": [input],
        "signatures": [sigs],
        "outputs": state["outputs"],
        "salt": state["salt"]
    });

    Ok(reveal.to_string())
}

#[wasm_bindgen]
pub fn verify_mss_sig_wasm(sig_hex: &str, msg_hex: &str, pk_hex: &str) -> bool {
    let sig_bytes = if let Ok(b) = hex::decode(sig_hex) { b } else { return false; };
    let mut msg = [0u8; 32]; if hex::decode_to_slice(msg_hex, &mut msg).is_err() { return false; }
    let mut pk = [0u8; 32]; if hex::decode_to_slice(pk_hex, &mut pk).is_err() { return false; }

    if let Ok(sig) = midstate::core::mss::MssSignature::from_bytes(&sig_bytes) {
        midstate::core::mss::verify(&sig, &msg, &pk)
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn compute_p2pk_address_hex(owner_pk_hex: &str) -> Result<String, JsValue> {
    let mut pk = [0u8; 32];
    hex::decode_to_slice(owner_pk_hex, &mut pk)
        .map_err(|_| JsValue::from_str("Invalid pubkey"))?;
    Ok(hex::encode(midstate::core::types::compute_address(&pk)))
}

#[wasm_bindgen]
pub fn build_covenant_htlc_bytecode_hex(
    secret_hash_hex:   &str,
    receiver_addr_hex: &str,
    min_payout:        u64,
    timeout_height:    u64,
    refund_pk_hex:     &str,
) -> Result<String, JsValue> {
    let mut sh = [0u8; 32];
    hex::decode_to_slice(secret_hash_hex, &mut sh)
        .map_err(|_| JsValue::from_str("Invalid secret_hash"))?;
    let mut ra = [0u8; 32];
    hex::decode_to_slice(receiver_addr_hex, &mut ra)
        .map_err(|_| JsValue::from_str("Invalid receiver_addr"))?;
    let mut refpk = [0u8; 32];
    hex::decode_to_slice(refund_pk_hex, &mut refpk)
        .map_err(|_| JsValue::from_str("Invalid refund_pk"))?;
    let bytecode = midstate::core::script::compile_covenant_htlc(
        &sh, &ra, min_payout, timeout_height, &refpk,
    );
    Ok(hex::encode(bytecode))
}

/// Builds the limit-order covenant bytecode (Feature 1). See
/// `midstate::core::script::compile_limit_order_covenant` for the security notes.
#[wasm_bindgen]
pub fn build_limit_order_covenant_bytecode_hex(
    secret_hash_hex: &str,
    max_claim:       u64,
    timeout_height:  u64,
    refund_pk_hex:   &str,
) -> Result<String, JsValue> {
    let mut sh = [0u8; 32];
    hex::decode_to_slice(secret_hash_hex, &mut sh)
        .map_err(|_| JsValue::from_str("Invalid secret_hash"))?;
    let mut refpk = [0u8; 32];
    hex::decode_to_slice(refund_pk_hex, &mut refpk)
        .map_err(|_| JsValue::from_str("Invalid refund_pk"))?;
    let bytecode = midstate::core::script::compile_limit_order_covenant(
        &sh, max_claim, timeout_height, &refpk,
    );
    Ok(hex::encode(bytecode))
}

/// Builds the Midstate HTLC bytecode for cross-chain atomic swaps.
#[wasm_bindgen]
pub fn build_htlc_bytecode_hex(
    secret_hash_hex: &str, 
    receiver_pk_hex: &str, 
    timeout_height: u64, 
    refund_pk_hex: &str
) -> Result<String, JsValue> {
    let mut sh = [0u8; 32];
    hex::decode_to_slice(secret_hash_hex, &mut sh)
        .map_err(|_| JsValue::from_str("Invalid secret_hash"))?;
        
    let mut rpk = [0u8; 32];
    hex::decode_to_slice(receiver_pk_hex, &mut rpk)
        .map_err(|_| JsValue::from_str("Invalid receiver_pk"))?;
        
    let mut refpk = [0u8; 32];
    hex::decode_to_slice(refund_pk_hex, &mut refpk)
        .map_err(|_| JsValue::from_str("Invalid refund_pk"))?;

    let bytecode = midstate::core::script::compile_htlc(&sh, &rpk, timeout_height, &refpk);
    Ok(hex::encode(bytecode))
}

    



// ═══════════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::types::hash;
    use midstate::core::mss;

    // 1. Import the test macro 
    use wasm_bindgen_test::wasm_bindgen_test;



    // ── Helpers ─────────────────────────────────────────────────────────

    fn test_seed() -> [u8; 32] { hash(b"wasm-wallet test seed") }

    /// Generate a small MSS keypair (height 4 = 16 leaves) for fast tests.
    fn make_test_keypair(height: u32) -> (String, MssKeypair) {
        let seed = test_seed();
        let mss_seed = derive_mss_seed(&seed, 0);
        let kp = mss::keygen(&mss_seed, height).unwrap();
        let addr = hex::encode(compute_address(&kp.master_pk));
        (addr, kp)
    }

    fn make_wallet_with_tree(height: u32) -> (WebWallet, String) {
        let (addr, kp) = make_test_keypair(height);
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        w.mss_cache.insert(addr.clone(), kp);
        (w, addr)
    }

    // ── MSS Binary Format: Layout ───────────────────────────────────────

    #[wasm_bindgen_test]
    fn binary_header_size_constant() {
        // height(4) + master_seed(32) + next_leaf(8) + master_pk(32) + tree_len(4) = 80
        assert_eq!(MSS_BINARY_HEADER_SIZE, 4 + 32 + 8 + 32 + 4);
    }

    #[wasm_bindgen_test]
    fn binary_format_correct_offsets_height4() {
        let (w, addr) = make_wallet_with_tree(4);
        let kp = w.mss_cache.get(&addr).unwrap();
        let bytes = w.export_mss_bytes(&addr).unwrap();

        // Verify each field at its documented offset
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 4, "height at offset 0");
        assert_eq!(&bytes[4..36], &kp.master_seed, "master_seed at offset 4");
        assert_eq!(u64::from_le_bytes(bytes[36..44].try_into().unwrap()), 0, "next_leaf at offset 36");
        assert_eq!(&bytes[44..76], &kp.master_pk, "master_pk at offset 44");

        let tree_len = u32::from_le_bytes(bytes[76..80].try_into().unwrap()) as usize;
        assert_eq!(tree_len, 32, "tree_len at offset 76 (2^4 * 2 = 32 nodes)");
        assert_eq!(bytes.len(), 80 + 32 * 32, "total size");
    }

    #[wasm_bindgen_test]
    fn binary_size_height10() {
        // Height 10: 2^10 * 2 = 2048 tree nodes
        // Total: 80 + 2048 * 32 = 65,616 bytes
        let expected_tree_nodes = 2048usize;
        let expected_size = MSS_BINARY_HEADER_SIZE + expected_tree_nodes * 32;
        assert_eq!(expected_size, 65_616);
    }

    // ── MSS Binary Format: Round-trip ───────────────────────────────────

    #[wasm_bindgen_test]
    fn export_import_roundtrip_height2() {
        let (mut w, addr) = make_wallet_with_tree(2);
        let original = w.mss_cache.get(&addr).unwrap().clone();

        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        w.import_mss_bytes(&addr, &bytes).unwrap();

        let imported = w.mss_cache.get(&addr).unwrap();
        assert_eq!(imported.height, original.height);
        assert_eq!(imported.master_seed, original.master_seed);
        assert_eq!(imported.next_leaf, original.next_leaf);
        assert_eq!(imported.master_pk, original.master_pk);
        assert_eq!(imported.tree, original.tree);
    }

    #[wasm_bindgen_test]
    fn export_import_roundtrip_height4() {
        let (mut w, addr) = make_wallet_with_tree(4);
        let original = w.mss_cache.get(&addr).unwrap().clone();

        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        assert!(!w.has_mss_cache(&addr));
        w.import_mss_bytes(&addr, &bytes).unwrap();
        assert!(w.has_mss_cache(&addr));

        let imported = w.mss_cache.get(&addr).unwrap();
        assert_eq!(imported.height, original.height);
        assert_eq!(imported.master_seed, original.master_seed);
        assert_eq!(imported.next_leaf, original.next_leaf);
        assert_eq!(imported.master_pk, original.master_pk);
        assert_eq!(imported.tree.len(), original.tree.len());
        for (i, (a, b)) in imported.tree.iter().zip(original.tree.iter()).enumerate() {
            assert_eq!(a, b, "tree node {} mismatch", i);
        }
    }

    // ── MSS Binary: Signing survives round-trip ─────────────────────────

    #[wasm_bindgen_test]
    fn signing_works_after_import() {
        let (mut w, addr) = make_wallet_with_tree(4);

        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        w.import_mss_bytes(&addr, &bytes).unwrap();

        let msg = hash(b"sign after import");
        let kp = w.mss_cache.get_mut(&addr).unwrap();
        let sig = kp.sign(&msg).unwrap();
        assert!(mss::verify(&sig, &msg, &kp.master_pk));
        assert_eq!(kp.next_leaf, 1);
    }

    #[wasm_bindgen_test]
    fn all_leaves_sign_after_import() {
        let (mut w, addr) = make_wallet_with_tree(2); // 4 leaves
        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        w.import_mss_bytes(&addr, &bytes).unwrap();

        let kp = w.mss_cache.get_mut(&addr).unwrap();
        let pk = kp.master_pk;
        for i in 0..4u8 {
            let msg = hash(&[i]);
            let sig = kp.sign(&msg).unwrap();
            assert!(mss::verify(&sig, &msg, &pk), "Leaf {} failed verification", i);
        }
        assert_eq!(kp.next_leaf, 4);
        assert!(kp.sign(&hash(b"extra")).is_err(), "Should be exhausted");
    }

    // ── MSS Binary: Leaf counter persistence ────────────────────────────

    #[wasm_bindgen_test]
    fn leaf_counter_survives_roundtrip() {
        let (addr, mut kp) = make_test_keypair(4);
        for i in 0..5u8 { kp.sign(&hash(&[i])).unwrap(); }
        assert_eq!(kp.next_leaf, 5);

        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        w.mss_cache.insert(addr.clone(), kp);

        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        w.import_mss_bytes(&addr, &bytes).unwrap();

        assert_eq!(w.mss_cache.get(&addr).unwrap().next_leaf, 5);
    }

    #[wasm_bindgen_test]
    fn set_leaf_index_updates_counter() {
        let (mut w, addr) = make_wallet_with_tree(4);
        assert_eq!(w.mss_cache.get(&addr).unwrap().next_leaf, 0);

        w.set_mss_leaf_index(&addr, 7);
        assert_eq!(w.mss_cache.get(&addr).unwrap().next_leaf, 7);
    }

    #[wasm_bindgen_test]
    fn set_leaf_index_noop_for_missing_address() {
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        w.set_mss_leaf_index("nonexistent", 42);
        assert!(!w.has_mss_cache("nonexistent"));
    }

    // ── MSS Binary: Error cases ─────────────────────────────────────────

    #[wasm_bindgen_test]
    fn import_empty_data_fails() {
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        assert!(w.import_mss_bytes("x", &[]).is_err());
    }

    #[wasm_bindgen_test]
    fn import_partial_header_fails() {
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        assert!(w.import_mss_bytes("x", &[0u8; 50]).is_err());
    }

    #[wasm_bindgen_test]
    fn import_header_only_with_nonzero_tree_len_fails() {
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        let mut header = [0u8; MSS_BINARY_HEADER_SIZE];
        header[76..80].copy_from_slice(&100u32.to_le_bytes()); // Claims 100 tree nodes
        assert!(w.import_mss_bytes("x", &header).is_err());
    }

    #[wasm_bindgen_test]
    fn import_truncated_tree_fails() {
        let (mut w, addr) = make_wallet_with_tree(4);
        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();

        let truncated = &bytes[..bytes.len() - 100];
        assert!(w.import_mss_bytes(&addr, truncated).is_err());
    }

    #[wasm_bindgen_test]
    fn import_with_trailing_junk_succeeds() {
        let (mut w, addr) = make_wallet_with_tree(4);
        let original_pk = w.mss_cache.get(&addr).unwrap().master_pk;

        let mut bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();

        bytes.extend_from_slice(&[0xFF; 256]);
        w.import_mss_bytes(&addr, &bytes).unwrap();

        assert_eq!(w.mss_cache.get(&addr).unwrap().master_pk, original_pk);
    }

    #[wasm_bindgen_test]
    fn export_missing_address_fails() {
        let w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        assert!(w.export_mss_bytes("nonexistent").is_err());
    }

    // ── MSS Binary: Determinism ─────────────────────────────────────────

    #[wasm_bindgen_test]
    fn export_is_deterministic() {
        let (addr, kp) = make_test_keypair(4);
        let mut w = WebWallet {
            master_seed: test_seed(),
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        w.mss_cache.insert(addr.clone(), kp.clone());
        let bytes1 = w.export_mss_bytes(&addr).unwrap();

        w.mss_cache.clear();
        w.mss_cache.insert(addr.clone(), kp);
        let bytes2 = w.export_mss_bytes(&addr).unwrap();

        assert_eq!(bytes1, bytes2);
    }

    #[wasm_bindgen_test]
    fn same_seed_same_tree() {
        let seed = test_seed();
        let mss_seed = derive_mss_seed(&seed, 0);
        let kp1 = mss::keygen(&mss_seed, 4).unwrap();
        let kp2 = mss::keygen(&mss_seed, 4).unwrap();
        assert_eq!(kp1.master_pk, kp2.master_pk);
        assert_eq!(kp1.tree, kp2.tree);
    }

    #[wasm_bindgen_test]
    fn different_indices_different_trees() {
        let seed = test_seed();
        let kp0 = mss::keygen(&derive_mss_seed(&seed, 0), 4).unwrap();
        let kp1 = mss::keygen(&derive_mss_seed(&seed, 1), 4).unwrap();
        assert_ne!(kp0.master_pk, kp1.master_pk);
    }

    // ── Multiple Trees ──────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn multiple_trees_coexist() {
        let seed = test_seed();
        let mut w = WebWallet {
            master_seed: seed,
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };

        let kp0 = mss::keygen(&derive_mss_seed(&seed, 0), 4).unwrap();
        let kp1 = mss::keygen(&derive_mss_seed(&seed, 1), 4).unwrap();
        let addr0 = hex::encode(compute_address(&kp0.master_pk));
        let addr1 = hex::encode(compute_address(&kp1.master_pk));

        w.mss_cache.insert(addr0.clone(), kp0);
        w.mss_cache.insert(addr1.clone(), kp1);

        let bytes0 = w.export_mss_bytes(&addr0).unwrap();
        let bytes1 = w.export_mss_bytes(&addr1).unwrap();
        assert_ne!(bytes0, bytes1);

        w.mss_cache.clear();
        w.import_mss_bytes(&addr0, &bytes0).unwrap();
        w.import_mss_bytes(&addr1, &bytes1).unwrap();

        assert!(w.has_mss_cache(&addr0));
        assert!(w.has_mss_cache(&addr1));
        assert_ne!(
            w.mss_cache.get(&addr0).unwrap().master_pk,
            w.mss_cache.get(&addr1).unwrap().master_pk
        );
    }

    // ── Cross-validation with CLI ───────────────────────────────────────

    #[wasm_bindgen_test]
    fn cli_generated_tree_verifies_after_import() {
        let seed = test_seed();
        let mss_seed = derive_mss_seed(&seed, 0);
        let cli_kp = mss::keygen(&mss_seed, 4).unwrap();

        let mut w = WebWallet {
            master_seed: seed,
            mss_cache: HashMap::new(),
            watchlist: Vec::new(),
        };
        let addr = hex::encode(compute_address(&cli_kp.master_pk));
        w.mss_cache.insert(addr.clone(), cli_kp.clone());

        // Round-trip through binary format
        let bytes = w.export_mss_bytes(&addr).unwrap();
        w.mss_cache.clear();
        w.import_mss_bytes(&addr, &bytes).unwrap();

        // Sign with imported, verify with original master_pk
        let msg = hash(b"cross-validation");
        let imported = w.mss_cache.get_mut(&addr).unwrap();
        let sig = imported.sign(&msg).unwrap();
        assert!(mss::verify(&sig, &msg, &cli_kp.master_pk));
    }

    // ── JSON Interop Structs ────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wasm_utxo_serde_roundtrip() {
        let utxo = WasmUtxo {
            index: 42, is_mss: true, mss_height: 10, mss_leaf: 5,
            address: "aa".repeat(32), value: 1024,
            salt: "bb".repeat(32), coin_id: "cc".repeat(32),
        };
        let json = serde_json::to_string(&utxo).unwrap();
        let decoded: WasmUtxo = serde_json::from_str(&json).unwrap();
        assert_eq!(utxo, decoded);
    }

    #[wasm_bindgen_test]
    fn js_output_default_salt() {
        let out: JsOutput = serde_json::from_str(r#"{"address":"aa","value":8}"#).unwrap();
        assert_eq!(out.salt, "");
    }

    #[wasm_bindgen_test]
    fn spend_context_roundtrip() {
        let ctx = SpendContext {
            selected_inputs: vec![],
            outputs: vec![JsOutput { address: "ab".repeat(32), value: 4, salt: "cd".repeat(32) }],
            commit_payload: serde_json::json!({"test": true}),
            tx_salt: "ee".repeat(32), commitment: "ff".repeat(32),
            fee: 100, next_wots_index: 7,
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let decoded: SpendContext = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.fee, 100);
        assert_eq!(decoded.next_wots_index, 7);
    }

    // ── Coin ID ─────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn coin_id_deterministic() {
        let a = "aa".repeat(32);
        let s = "bb".repeat(32);
        assert_eq!(compute_coin_id_hex(&a, 8, &s), compute_coin_id_hex(&a, 8, &s));
        assert_eq!(compute_coin_id_hex(&a, 8, &s).len(), 64);
    }

    #[wasm_bindgen_test]
    fn coin_id_varies_with_inputs() {
        let a = "aa".repeat(32);
        let s1 = "bb".repeat(32);
        let s2 = "cc".repeat(32);
        assert_ne!(compute_coin_id_hex(&a, 8, &s1), compute_coin_id_hex(&a, 16, &s1));
        assert_ne!(compute_coin_id_hex(&a, 8, &s1), compute_coin_id_hex(&a, 8, &s2));
    }

    #[wasm_bindgen_test]
    fn coin_id_invalid_hex_defaults_to_zeros() {
        let id_bad = compute_coin_id_hex("not_hex", 8, "also_bad");
        let id_zeros = compute_coin_id_hex(&"00".repeat(32), 8, &"00".repeat(32));
        assert_eq!(id_bad, id_zeros);
    }

    // ── Decompose Value ─────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn decompose_zero() { assert!(decompose_value(0).is_empty()); }

    #[wasm_bindgen_test]
    fn decompose_powers_of_two() {
        assert_eq!(decompose_value(1), vec![1]);
        assert_eq!(decompose_value(8), vec![8]);
        assert_eq!(decompose_value(1024), vec![1024]);
    }

    #[wasm_bindgen_test]
    fn decompose_mixed() {
        let parts = decompose_value(13);
        assert_eq!(parts.iter().sum::<u64>(), 13);
        for &p in &parts { assert!(p.is_power_of_two()); }
    }

    #[wasm_bindgen_test]
    fn decompose_u64_max() {
        let parts = decompose_value(u64::MAX);
        assert_eq!(parts.iter().sum::<u64>(), u64::MAX);
        assert_eq!(parts.len(), 64);
    }

    // ── WOTS Address ────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn wots_address_deterministic() {
        let s = test_seed();
        let a1 = hex::encode(compute_address(&wots::keygen(&derive_wots_seed(&s, 0))));
        let a2 = hex::encode(compute_address(&wots::keygen(&derive_wots_seed(&s, 0))));
        assert_eq!(a1, a2);
    }

    #[wasm_bindgen_test]
    fn wots_different_indices() {
        let s = test_seed();
        let a0 = hex::encode(compute_address(&wots::keygen(&derive_wots_seed(&s, 0))));
        let a1 = hex::encode(compute_address(&wots::keygen(&derive_wots_seed(&s, 1))));
        assert_ne!(a0, a1);
    }

    // ── Watchlist ────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn watchlist_valid() {
        let mut w = WebWallet { master_seed: test_seed(), mss_cache: HashMap::new(), watchlist: Vec::new() };
        let addrs = vec!["aa".repeat(32), "bb".repeat(32)];
        w.set_watchlist(&serde_json::to_string(&addrs).unwrap());
        assert_eq!(w.watchlist.len(), 2);
    }

    #[wasm_bindgen_test]
    fn watchlist_skips_invalid() {
        let mut w = WebWallet { master_seed: test_seed(), mss_cache: HashMap::new(), watchlist: Vec::new() };
        let addrs = vec!["aa".repeat(32), "nope".into(), "bb".repeat(32)];
        w.set_watchlist(&serde_json::to_string(&addrs).unwrap());
        assert_eq!(w.watchlist.len(), 2);
    }

    #[wasm_bindgen_test]
    fn watchlist_replaces() {
        let mut w = WebWallet { master_seed: test_seed(), mss_cache: HashMap::new(), watchlist: Vec::new() };
        w.set_watchlist(&serde_json::to_string(&vec!["aa".repeat(32)]).unwrap());
        assert_eq!(w.watchlist.len(), 1);
        w.set_watchlist(&serde_json::to_string(&vec!["bb".repeat(32), "cc".repeat(32)]).unwrap());
        assert_eq!(w.watchlist.len(), 2);
    }

    #[wasm_bindgen_test]
    fn watchlist_empty_and_invalid_json() {
        let mut w = WebWallet { master_seed: test_seed(), mss_cache: HashMap::new(), watchlist: vec![[0; 32]] };
        w.set_watchlist("[]");
        assert!(w.watchlist.is_empty());
        w.watchlist.push([0; 32]);
        w.set_watchlist("garbage");
        assert!(w.watchlist.is_empty());
    }

    // ── has_mss_cache ───────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn has_mss_cache_lifecycle() {
        let (addr, kp) = make_test_keypair(4);
        let mut w = WebWallet { master_seed: test_seed(), mss_cache: HashMap::new(), watchlist: Vec::new() };
        assert!(!w.has_mss_cache(&addr));
        w.mss_cache.insert(addr.clone(), kp);
        assert!(w.has_mss_cache(&addr));
        w.mss_cache.clear();
        assert!(!w.has_mss_cache(&addr));
    }
    
    // ── Address Parsing (WASM) ──────────────────────────────────────────
    #[wasm_bindgen_test]
    fn parse_address_wasm_legacy_64_valid() {
        let addr_hex = "aa".repeat(32);
        let result = parse_address_wasm(&addr_hex);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), [0xaa; 32]);
    }

    #[wasm_bindgen_test]
    fn parse_address_wasm_checksum_72_valid() {
        let addr_bytes = [0xbb; 32];
        let checksum = midstate::core::types::hash(&addr_bytes);
        
        // Build the 36-byte payload (32-byte address + 4-byte checksum)
        let mut payload = addr_bytes.to_vec();
        payload.extend_from_slice(&checksum[0..4]);
        
        let addr_hex = hex::encode(payload);
        
        let result = parse_address_wasm(&addr_hex);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), addr_bytes);
    }

    #[wasm_bindgen_test]
    fn parse_address_wasm_invalid_checksum_rejected() {
        let addr_bytes = [0xcc; 32];
        
        // Build a 36-byte payload with a deliberately WRONG checksum
        let mut payload = addr_bytes.to_vec();
        payload.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); 
        
        let addr_hex = hex::encode(payload);
        
        let result = parse_address_wasm(&addr_hex);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().as_string().unwrap(), "Checksum mismatch! The address contains a typo.");
    }

    #[wasm_bindgen_test]
    fn parse_address_wasm_invalid_length_rejected() {
        // 62 characters (31 bytes) - too short
        let too_short = "aa".repeat(31);
        assert!(parse_address_wasm(&too_short).is_err());

        // 66 characters (33 bytes) - invalid length
        let weird_length = "aa".repeat(33);
        assert!(parse_address_wasm(&weird_length).is_err());
    }
}
