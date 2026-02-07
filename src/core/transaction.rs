use super::types::*;
use super::wots;
use super::mss;
use anyhow::{bail, Result};

 /// Verify a signature that may be either raw WOTS (576 bytes) or MSS (longer).
 fn verify_signature(sig_bytes: &[u8], message: &[u8; 32], coin_id: &[u8; 32]) -> bool {
     if sig_bytes.len() == wots::SIG_SIZE {
         match wots::sig_from_bytes(sig_bytes) {
             Some(sig) => wots::verify(&sig, message, coin_id),
             None => false,
         }
     } else {
         match mss::MssSignature::from_bytes(sig_bytes) {
             Ok(mss_sig) => mss::verify(&mss_sig, message, coin_id),
             Err(_) => false,
         }
     }
 }

/// Apply a transaction to the state
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment } => {
            if !state.commitments.insert(*commitment) {
                bail!("Duplicate commitment");
            }
            state.midstate = hash_concat(&state.midstate, commitment);
            Ok(())
        }

        Transaction::Reveal { input_coins, signatures, new_coins, salt } => {
            if input_coins.is_empty() {
                bail!("Transaction must spend at least one coin");
            }
            if new_coins.is_empty() {
                bail!("Transaction must create at least one new coin");
            }
            if signatures.len() != input_coins.len() {
                bail!("Signature count must match input count");
            }
            // Fee enforcement: inputs > outputs
            if input_coins.len() <= new_coins.len() {
                bail!(
                    "Inputs ({}) must exceed outputs ({}) to pay fee",
                    input_coins.len(),
                    new_coins.len()
                );
            }

            // Verify commitment exists and matches
            let expected = compute_commitment(input_coins, new_coins, salt);
            if !state.commitments.remove(&expected) {
                bail!(
                    "No matching commitment found (expected {})",
                    hex::encode(expected)
                );
            }

            // Verify WOTS signatures and remove coins
            for (i, (coin_id, sig_bytes)) in input_coins.iter().zip(signatures.iter()).enumerate() {
                if !state.coins.contains(coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }
                if !verify_signature(sig_bytes, &expected, coin_id) {
                    bail!("Invalid signature for input {}", i);
                }
            }

            // Remove spent coins
            for coin_id in input_coins {
                state.coins.remove(coin_id);
            }

            // Add new coins
            for new_coin in new_coins {
                if !state.coins.insert(*new_coin) {
                    bail!("Duplicate coin created");
                }
            }

            // Update midstate
            let tx_bytes = bincode::serialize(tx)?;
            state.midstate = hash_concat(&state.midstate, &tx_bytes);

            Ok(())
        }
    }
}

/// Validate a transaction without applying it
pub fn validate_transaction(state: &State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment } => {
            if state.commitments.contains(commitment) {
                bail!("Duplicate commitment");
            }
            Ok(())
        }

        Transaction::Reveal { input_coins, signatures, new_coins, salt } => {
            if input_coins.is_empty() {
                bail!("Must spend at least one coin");
            }
            if new_coins.is_empty() {
                bail!("Must create at least one coin");
            }
            if signatures.len() != input_coins.len() {
                bail!("Signature count must match input count");
            }
            if input_coins.len() <= new_coins.len() {
                bail!("Inputs must exceed outputs to pay fee");
            }

            let expected = compute_commitment(input_coins, new_coins, salt);
            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found");
            }

            for (i, (coin_id, sig_bytes)) in input_coins.iter().zip(signatures.iter()).enumerate() {
                if !state.coins.contains(coin_id) {
                    bail!("Coin {} not found", hex::encode(coin_id));
                }
                if !verify_signature(sig_bytes, &expected, coin_id) {
                    bail!("Invalid signature for input {}", i);
                }
            }

            Ok(())
        }
    }
}
