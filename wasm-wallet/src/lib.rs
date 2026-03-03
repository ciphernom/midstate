use wasm_bindgen::prelude::*;
use midstate::core::{wots, mss};
use midstate::core::types::{compute_address, compute_commitment, compute_coin_id, decompose_value};
use midstate::wallet::hd::{generate_mnemonic, master_seed_from_mnemonic, derive_wots_seed, derive_mss_seed};
use serde::{Deserialize, Serialize};

// ─── Global Wasm Helpers ────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn generate_phrase() -> String {
    let (_, phrase) = generate_mnemonic().unwrap();
    phrase
}

#[wasm_bindgen]
pub fn decompose_amount(amount: u64) -> js_sys::BigUint64Array {
    let parts = decompose_value(amount);
    js_sys::BigUint64Array::from(&parts[..])
}

#[wasm_bindgen]
pub fn compute_coin_id_hex(address_hex: &str, value: u64, salt_hex: &str) -> String {
    let mut addr = [0u8; 32];
    hex::decode_to_slice(address_hex, &mut addr).unwrap_or_default();
    let mut salt = [0u8; 32];
    hex::decode_to_slice(salt_hex, &mut salt).unwrap_or_default();
    let cid = compute_coin_id(&addr, value, &salt);
    hex::encode(cid)
}

// ─── JSON Interop Structs ───────────────────────────────────────────────────

#[derive(Deserialize, Serialize, Clone, Debug)]
struct WasmUtxo {
    index: u32,
    is_mss: bool,
    mss_height: u32,
    mss_leaf: u32,
    address: String,
    value: u64,
    salt: String,
    coin_id: String,
}

#[derive(Deserialize, Serialize, Clone)]
struct JsOutput {
    address: String,
    value: u64,
    #[serde(default)]
    salt: String, 
}

#[derive(Serialize, Deserialize)]
struct SpendContext {
    selected_inputs: Vec<WasmUtxo>,
    outputs: Vec<JsOutput>,
    commit_payload: serde_json::Value,
    tx_salt: String,
    commitment: String,
    fee: u64,
    next_wots_index: u32,
}

// ─── Main Wallet Object ─────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct WebWallet {
    master_seed: [u8; 32],
}

#[wasm_bindgen]
impl WebWallet {
    #[wasm_bindgen(constructor)]
    pub fn new(phrase: &str) -> Result<WebWallet, JsValue> {
        let master_seed = master_seed_from_mnemonic(phrase)
            .map_err(|e| JsValue::from_str(&format!("Invalid mnemonic: {}", e)))?;
        Ok(WebWallet { master_seed })
    }

    /// Derives a single-use WOTS address (used internally for change outputs)
    pub fn get_wots_address(&self, index: u32) -> String {
        let seed = derive_wots_seed(&self.master_seed, index as u64);
        let pk = wots::keygen(&seed);
        hex::encode(compute_address(&pk))
    }

    /// Derives a reusable MSS address for receiving funds (Height 5 recommended)
    pub fn get_mss_address(&self, index: u32, height: u32) -> Result<String, JsValue> {
        let seed = derive_mss_seed(&self.master_seed, index as u64);
        let kp = mss::keygen(&seed, height).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(hex::encode(kp.master_pk))
    }

    pub fn check_filter(&self, filter_hex: &str, block_hash_hex: &str, n: u32, addrs_json: &str) -> bool {
        let filter_data = match hex::decode(filter_hex) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let mut block_hash = [0u8; 32];
        if hex::decode_to_slice(block_hash_hex, &mut block_hash).is_err() { return false; }

        let addrs_str: Vec<String> = serde_json::from_str(addrs_json).unwrap_or_default();
        let mut byte_addrs = Vec::with_capacity(addrs_str.len());
        for a in addrs_str {
            let mut buf = [0u8; 32];
            if hex::decode_to_slice(&a, &mut buf).is_ok() { byte_addrs.push(buf); }
        }

        if byte_addrs.is_empty() { return false; }
        midstate::core::filter::match_any(&filter_data, &block_hash, n as u64, &byte_addrs)
    }

    pub fn prepare_spend(&self, available_utxos_json: &str, to_address_hex: &str, send_amount: u64, mut next_wots_index: u32) -> Result<String, JsValue> {
        let mut available: Vec<WasmUtxo> = serde_json::from_str(available_utxos_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse UTXOs: {}", e)))?;

        available.sort_by(|a, b| b.value.cmp(&a.value));

        let mut target_fee = 100u64; 
        let final_selected;
        let mut final_outputs = Vec::new();
        let final_fee;

        loop {
            let needed = send_amount + target_fee;
            let mut selected = Vec::new();
            let mut selected_set = std::collections::HashSet::new();
            let mut total = 0u64;

            for coin in &available {
                if total >= needed { break; }
                selected.push(coin.clone());
                selected_set.insert(coin.coin_id.clone());
                total += coin.value;
            }

            if total < needed { return Err(JsValue::from_str("Insufficient funds.")); }

            let mut grouped_addresses = std::collections::HashSet::new();
            for c in &selected { 
                if !c.is_mss { grouped_addresses.insert(c.address.clone()); }
            }
            
            for coin in &available {
                if grouped_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected.push(coin.clone());
                    selected_set.insert(coin.coin_id.clone());
                    total += coin.value;
                }
            }

            let mut added_new = true;
            while added_new {
                added_new = false;
                let change = total.saturating_sub(send_amount).saturating_sub(target_fee);
                let change_denoms = decompose_value(change);
                
                for denom in change_denoms {
                    if let Some(pos) = available.iter().position(|c| c.value == denom && !selected_set.contains(&c.coin_id)) {
                        let coin = available[pos].clone();
                        selected.push(coin.clone());
                        selected_set.insert(coin.coin_id.clone());
                        total += denom;
                        added_new = true;
                        break; 
                    }
                }
                if selected.len() >= 250 { break; }
            }

            let mut final_addresses = std::collections::HashSet::new();
            for c in &selected { 
                if !c.is_mss { final_addresses.insert(c.address.clone()); }
            }
            for coin in &available {
                if final_addresses.contains(&coin.address) && !selected_set.contains(&coin.coin_id) {
                    selected.push(coin.clone());
                    selected_set.insert(coin.coin_id.clone());
                    total += coin.value;
                }
            }

            let mut num_outputs = decompose_value(send_amount).len();
            let final_change = total.saturating_sub(send_amount).saturating_sub(target_fee);
            num_outputs += decompose_value(final_change).len();

            let estimated_bytes = 100 + (selected.len() as u64 * 1636) + (num_outputs as u64 * 100);
            let required_fee = (estimated_bytes * 10) / 1024 + 10;

            if total >= send_amount + required_fee {
                final_fee = required_fee;
                let actual_change = total - send_amount - final_fee;
                
                for denom in decompose_value(send_amount) {
                    let mut salt = [0u8; 32];
                    getrandom_02::getrandom(&mut salt).unwrap();
                    final_outputs.push(JsOutput { address: to_address_hex.to_string(), value: denom, salt: hex::encode(salt) });
                }

                // Change outputs always use WOTS to save space/fees
                for denom in decompose_value(actual_change) {
                    let seed = derive_wots_seed(&self.master_seed, next_wots_index as u64);
                    let pk = wots::keygen(&seed);
                    let addr = hex::encode(compute_address(&pk));
                    next_wots_index += 1;

                    let mut salt = [0u8; 32];
                    getrandom_02::getrandom(&mut salt).unwrap();
                    final_outputs.push(JsOutput { address: addr, value: denom, salt: hex::encode(salt) });
                }
                final_selected = selected;
                break;
            } else {
                target_fee = required_fee;
            }
        }

        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        final_outputs.shuffle(&mut rng);

        let mut input_coin_ids = Vec::new();
        for inp in &final_selected {
            let mut buf = [0u8; 32];
            hex::decode_to_slice(&inp.coin_id, &mut buf).unwrap();
            input_coin_ids.push(buf);
        }

        let mut output_hashes = Vec::new();
        for out in &final_outputs {
            let mut addr_bytes = [0u8; 32];
            let mut salt_bytes = [0u8; 32];
            hex::decode_to_slice(&out.address, &mut addr_bytes).unwrap();
            hex::decode_to_slice(&out.salt, &mut salt_bytes).unwrap();
            output_hashes.push(compute_coin_id(&addr_bytes, out.value, &salt_bytes));
        }

        let mut tx_salt = [0u8; 32];
        getrandom_02::getrandom(&mut tx_salt).unwrap();

        let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);
        let dest_hashes: Vec<String> = output_hashes.iter().map(hex::encode).collect();

        let commit_payload = serde_json::json!({
            "coins": final_selected.iter().map(|i| i.coin_id.clone()).collect::<Vec<_>>(),
            "destinations": dest_hashes
        });

        let ctx = SpendContext {
            selected_inputs: final_selected,
            outputs: final_outputs,
            commit_payload,
            tx_salt: hex::encode(tx_salt),
            commitment: hex::encode(commitment),
            fee: final_fee,
            next_wots_index,
        };

        Ok(serde_json::to_string(&ctx).unwrap())
    }

    pub fn build_reveal(&self, spend_context_json: &str, server_commitment_hex: &str, server_salt_hex: &str) -> Result<String, JsValue> {
        let ctx: SpendContext = serde_json::from_str(spend_context_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut commitment = [0u8; 32];
        hex::decode_to_slice(server_commitment_hex, &mut commitment)
            .map_err(|_| JsValue::from_str("Invalid server commitment hex"))?;

        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();
        let mut safety_input_hashes = Vec::new();

        for inp in ctx.selected_inputs {
            let (pk, sig_bytes) = if inp.is_mss {
                let seed = derive_mss_seed(&self.master_seed, inp.index as u64);
                let mut kp = mss::keygen(&seed, inp.mss_height).unwrap();
                kp.set_next_leaf(inp.mss_leaf as u64);
                let sig = kp.sign(&commitment).map_err(|e| JsValue::from_str(&e.to_string()))?;
                (kp.master_pk, sig.to_bytes())
            } else {
                let seed = derive_wots_seed(&self.master_seed, inp.index as u64);
                let wots_pk = wots::keygen(&seed);
                let wots_sig = wots::sign(&seed, &commitment);
                (wots_pk, wots::sig_to_bytes(&wots_sig))
            };

            let bytecode = midstate::core::script::compile_p2pk(&pk);
            let address = midstate::core::types::hash(&bytecode);
            let mut salt_bytes = [0u8; 32];
            hex::decode_to_slice(&inp.salt, &mut salt_bytes).unwrap();
            safety_input_hashes.push(compute_coin_id(&address, inp.value, &salt_bytes));

            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&bytecode),
                "value": inp.value,
                "salt": inp.salt
            }));

            // The node expects a single continuous hex string for the 576-byte signature
            signatures.push(hex::encode(&sig_bytes));
        }

        let mut output_json = Vec::new();
        let mut safety_output_hashes = Vec::new();

        for o in ctx.outputs {
            let mut addr_bytes = [0u8; 32];
            let mut salt_bytes = [0u8; 32];
            hex::decode_to_slice(&o.address, &mut addr_bytes).unwrap();
            hex::decode_to_slice(&o.salt, &mut salt_bytes).unwrap();
            safety_output_hashes.push(compute_coin_id(&addr_bytes, o.value, &salt_bytes));

            output_json.push(serde_json::json!({
                "type": "standard", 
                "address": o.address, 
                "value": o.value, 
                "salt": o.salt 
            }));
        }

        let mut server_salt = [0u8; 32];
        hex::decode_to_slice(server_salt_hex, &mut server_salt).unwrap();
        let safety_check_commitment = compute_commitment(&safety_input_hashes, &safety_output_hashes, &server_salt);
        
        if safety_check_commitment != commitment {
            return Err(JsValue::from_str("Fatal Hash Mismatch! Internal payload tracking error."));
        }

        let payload = serde_json::json!({
            "inputs": input_reveals,
            "signatures": signatures,
            "outputs": output_json,
            "salt": server_salt_hex
        });

        Ok(payload.to_string())
    }
}
