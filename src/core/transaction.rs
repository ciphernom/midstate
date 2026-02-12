use super::types::*;
use super::wots;
use super::mss;
use anyhow::{bail, Result};

const COMMIT_POW_TARGET: u16 = 0x0000;

fn validate_commit_pow(commitment: &[u8; 32], nonce: u64) -> Result<()> {
    let h = super::types::hash_concat(commitment, &nonce.to_le_bytes());
    if u16::from_be_bytes([h[0], h[1]]) != COMMIT_POW_TARGET {
        bail!("Insufficient Commit PoW");
    }
    Ok(())
}

/// Verify a signature that may be either raw WOTS (576 bytes) or MSS (longer).
fn verify_signature(sig_bytes: &[u8], message: &[u8; 32], owner_pk: &[u8; 32]) -> bool {
    if sig_bytes.len() == wots::SIG_SIZE {
        match wots::sig_from_bytes(sig_bytes) {
            Some(sig) => wots::verify(&sig, message, owner_pk),
            None => false,
        }
    } else {
        match mss::MssSignature::from_bytes(sig_bytes) {
            Ok(mss_sig) => mss::verify(&mss_sig, message, owner_pk),
            Err(_) => false,
        }
    }
}

/// Apply a transaction to the state
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce)?;
            if !state.commitments.insert(*commitment) {
                bail!("Duplicate commitment");
            }
            state.commitment_heights.insert(*commitment, state.height);
            state.midstate = hash_concat(&state.midstate, commitment);
            Ok(())
        }

        Transaction::Reveal { inputs, signatures, outputs, salt } => {
            if inputs.is_empty() {
                bail!("Transaction must spend at least one coin");
            }
            if outputs.is_empty() {
                bail!("Transaction must create at least one new coin");
            }
            if inputs.len() > MAX_TX_INPUTS { 
                bail!("Too many inputs (max {})", MAX_TX_INPUTS); 
                }
            if outputs.len() > MAX_TX_OUTPUTS { 
                bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); 
            }
            
            if signatures.len() != inputs.len() {
                bail!("Signature count must match input count");
            }
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                }
            }
            // 1. Validate all output values are power of 2 and nonzero
            for (i, out) in outputs.iter().enumerate() {
                if out.value == 0 {
                    bail!("Zero-value output {}", i);
                }
                if !out.value.is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value);
                }
            }

            // 2. Value conservation: sum(inputs) > sum(outputs)
            let in_sum: u64 = inputs.iter().map(|i| i.value).sum();
            let out_sum: u64 = outputs.iter().map(|o| o.value).sum();
            if in_sum <= out_sum {
                bail!(
                    "Input value ({}) must exceed output value ({}) to pay fee",
                    in_sum, out_sum
                );
            }

            // 3. Compute coin IDs from preimages
            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_coin_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.coin_id()).collect();

            // 4. Verify commitment exists and matches
            let expected = compute_commitment(&input_coin_ids, &output_coin_ids, salt);
            if !state.commitments.remove(&expected) {
                bail!(
                    "No matching commitment found (expected {})",
                    hex::encode(expected)
                );
            }

            // 5. Verify each input coin exists and signature is valid against owner_pk
            for (i, (input, sig_bytes)) in inputs.iter().zip(signatures.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }
                if !verify_signature(sig_bytes, &expected, &input.owner_pk) {
                    bail!("Invalid signature for input {}", i);
                }
            }

            // 6. Remove spent coins
            for coin_id in &input_coin_ids {
                state.coins.remove(coin_id);
            }

            // 7. Add new coins (store only the coin_id hash)
            for coin_id in &output_coin_ids {
                if !state.coins.insert(*coin_id) {
                    bail!("Duplicate coin created");
                }
            }

            // 8. Update midstate
            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids {
                    hasher.update(coin_id);
                }
                for coin_id in &output_coin_ids {
                    hasher.update(coin_id);
                }
                hasher.update(salt);
                let tx_hash = *hasher.finalize().as_bytes();
                state.midstate = hash_concat(&state.midstate, &tx_hash);
            }

            Ok(())
        }
    }
}

/// Validate a transaction without applying it
pub fn validate_transaction(state: &State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce)?;
            if state.commitments.contains(commitment) {
                bail!("Duplicate commitment");
            }
            Ok(())
        }

        Transaction::Reveal { inputs, signatures, outputs, salt } => {
            if inputs.is_empty() {
                bail!("Must spend at least one coin");
            }
            if outputs.is_empty() {
                bail!("Must create at least one coin");
            }
            if inputs.len() > MAX_TX_INPUTS { 
                bail!("Too many inputs (max {})", MAX_TX_INPUTS); 
                }
            if outputs.len() > MAX_TX_OUTPUTS { 
                bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); 
            }
            
            if signatures.len() != inputs.len() {
                bail!("Signature count must match input count");
            }
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                }
            }
            for (i, out) in outputs.iter().enumerate() {
                if out.value == 0 {
                    bail!("Zero-value output {}", i);
                }
                if !out.value.is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value);
                }
            }

            let in_sum: u64 = inputs.iter().map(|i| i.value).sum();
            let out_sum: u64 = outputs.iter().map(|o| o.value).sum();
            if in_sum <= out_sum {
                bail!("Input value must exceed output value");
            }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_coin_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.coin_id()).collect();

            let expected = compute_commitment(&input_coin_ids, &output_coin_ids, salt);
            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found");
            }

            for (i, (input, sig_bytes)) in inputs.iter().zip(signatures.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found", hex::encode(coin_id));
                }
                if !verify_signature(sig_bytes, &expected, &input.owner_pk) {
                    bail!("Invalid signature for input {}", i);
                }
            }

            Ok(())
        }
    }
}
