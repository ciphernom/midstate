use wasm_bindgen::prelude::*;
use midstate::core::wots;
use midstate::core::types::compute_address;
use midstate::wallet::hd::master_seed_from_mnemonic;
use hex;

use serde::{Deserialize, Serialize};
use midstate::core::OutputData;

#[wasm_bindgen]
pub struct WebWallet {
    seed: [u8; 32],
    // Store state between the Commit and Reveal phases
    pending_inputs_json: String,
    pending_outputs_json: String,
}

#[derive(Deserialize)]
struct JsInput {
    coin_id: String,
    value: u64,
    salt: String,
}

#[wasm_bindgen]
impl WebWallet {
    #[wasm_bindgen(constructor)]
    pub fn new(phrase: &str) -> Result<WebWallet, JsValue> {
        let seed = master_seed_from_mnemonic(phrase)
            .map_err(|e| JsValue::from_str(&format!("Invalid mnemonic: {}", e)))?;
        Ok(WebWallet { 
            seed,
            pending_inputs_json: String::new(),
            pending_outputs_json: String::new(),
        })
    }

    pub fn get_primary_address(&self) -> String {
        let mut key_seed = [0u8; 32];
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(&0u32.to_le_bytes()); 
        key_seed.copy_from_slice(hasher.finalize().as_bytes());
        let pk = wots::keygen(&key_seed);
        hex::encode(compute_address(&pk))
    }
/// Checks a block's compact filter to see if it contains our address
    pub fn check_filter(&self, filter_hex: &str, block_hash_hex: &str, n: u32) -> bool {
        let filter_data = match hex::decode(filter_hex) {
            Ok(d) => d,
            Err(_) => return false,
        };
        
        let mut block_hash = [0u8; 32];
        if hex::decode_to_slice(block_hash_hex, &mut block_hash).is_err() {
            return false;
        }

        let addr_hex = self.get_primary_address();
        let mut addr_bytes = [0u8; 32];
        hex::decode_to_slice(&addr_hex, &mut addr_bytes).unwrap();

        // Notice we kept the `n as u64` fix here!
        midstate::core::filter::match_any(&filter_data, &block_hash, n as u64, &[addr_bytes])
    }
    /// Phase 1: Build the Commit payload
    pub fn build_commit(&mut self, inputs_json: &str, to_address: &str, send_amount: u64, fee: u64) -> Result<String, JsValue> {
        let inputs: Vec<JsInput> = serde_json::from_str(inputs_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid inputs JSON: {}", e)))?;

        let mut in_sum = 0u64;
        let mut coin_ids = Vec::new();
        for input in &inputs {
            in_sum += input.value;
            coin_ids.push(input.coin_id.clone());
        }

        if in_sum < send_amount + fee {
            return Err(JsValue::from_str("Insufficient funds for amount + fee"));
        }

        let change = in_sum - send_amount - fee;
        let mut outputs = Vec::new();

        // 1. Recipient Output
        let mut addr_bytes = [0u8; 32];
        hex::decode_to_slice(to_address, &mut addr_bytes).map_err(|_| JsValue::from_str("Invalid to_address"))?;
        
        let mut salt1 = [0u8; 32];
        getrandom_02::getrandom(&mut salt1).unwrap();
        outputs.push(OutputData::Standard { address: addr_bytes, value: send_amount, salt: salt1 });

        // 2. Change Output (sending back to ourselves)
        if change > 0 {
            let my_addr_hex = self.get_primary_address();
            let mut my_addr_bytes = [0u8; 32];
            hex::decode_to_slice(&my_addr_hex, &mut my_addr_bytes).unwrap();
            let mut salt2 = [0u8; 32];
            getrandom_02::getrandom(&mut salt2).unwrap();
            outputs.push(OutputData::Standard { address: my_addr_bytes, value: change, salt: salt2 });
        }

        // Save state for Reveal phase
        self.pending_inputs_json = inputs_json.to_string();
        self.pending_outputs_json = serde_json::to_string(&outputs).unwrap();

        // Build the Commit JSON for the node
        let dest_hashes: Vec<String> = outputs.iter().map(|o| hex::encode(o.hash_for_commitment())).collect();
        let payload = serde_json::json!({
            "coins": coin_ids,
            "destinations": dest_hashes
        });

        Ok(payload.to_string())
    }

    /// Phase 2: Sign the transaction and build the Reveal payload
    pub fn build_reveal(&self, commitment_hex: &str, server_salt_hex: &str) -> Result<String, JsValue> {
        let inputs: Vec<JsInput> = serde_json::from_str(&self.pending_inputs_json).unwrap();
        let outputs: Vec<OutputData> = serde_json::from_str(&self.pending_outputs_json).unwrap();

        let mut commitment = [0u8; 32];
        hex::decode_to_slice(commitment_hex, &mut commitment).map_err(|_| JsValue::from_str("Invalid commitment hex"))?;

        // Reconstruct our primary key (since all inputs belong to it)
        let mut key_seed = [0u8; 32];
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(&0u32.to_le_bytes());
        key_seed.copy_from_slice(hasher.finalize().as_bytes());
        let pk = wots::keygen(&key_seed);

        // Build the JSON payload for the node
        let mut input_reveals = Vec::new();
        let mut signatures = Vec::new();

        for input in inputs {
            // Add the reveal data
            input_reveals.push(serde_json::json!({
                "bytecode": hex::encode(&pk), // For WOTS, the bytecode is just the public key
                "value": input.value,
                "salt": input.salt
            }));

            // Sign the commitment
            let sig = wots::sign(&key_seed, &commitment);
            // Encode each 32-byte chunk and join them with a comma
            let sig_str = sig.iter().map(hex::encode).collect::<Vec<_>>().join(",");
            signatures.push(sig_str);
        }

        let mut output_json = Vec::new();
        for o in outputs {
            match o {
                OutputData::Standard { address, value, salt } => {
                    output_json.push(serde_json::json!({
                        "Standard": {
                            "address": hex::encode(address),
                            "value": value,
                            "salt": hex::encode(salt)
                        }
                    }));
                },
                _ => {}
            }
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

/// Helper to generate a brand new seed phrase in the browser
#[wasm_bindgen]
pub fn generate_new_phrase() -> String {
    // Generate 32 bytes of secure entropy using the browser's crypto API
    let mut entropy = [0u8; 32];
    getrandom_02::getrandom(&mut entropy).unwrap();
    
    // In a full implementation, you'd use the `bip39` crate here to turn 
    // this entropy into 24 words. For the MVP test, we'll just return hex!
    hex::encode(entropy) 
}
