use super::types::*;
use anyhow::{bail, Result};

/// Apply a transaction to the state
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment } => {
            // Register the commitment
            if !state.commitments.insert(*commitment) {
                bail!("Duplicate commitment");
            }

            // Update midstate
            state.midstate = hash_concat(&state.midstate, commitment);

            Ok(())
        }

        Transaction::Reveal { secrets, new_coins, salt } => {
            if secrets.is_empty() {
                bail!("Transaction must spend at least one coin");
            }
            if new_coins.is_empty() {
                bail!("Transaction must create at least one new coin");
            }

            // Compute the coins being spent
            let old_coins: Vec<[u8; 32]> = secrets.iter().map(|s| hash(s)).collect();

            // Verify commitment exists and matches
            let expected = compute_commitment(&old_coins, new_coins, salt);
            if !state.commitments.remove(&expected) {
                bail!(
                    "No matching commitment found (expected {})",
                    hex::encode(expected)
                );
            }

            // Check all coins exist and remove them
            for old_coin in &old_coins {
                if !state.coins.remove(old_coin) {
                    bail!("Coin {:?} not found or already spent", hex::encode(old_coin));
                }
            }

            // Add new coins
            for new_coin in new_coins {
                if !state.coins.insert(*new_coin) {
                    bail!("Duplicate coin created");
                }
            }

            // Update midstate with transaction data
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

        Transaction::Reveal { secrets, new_coins, salt } => {
            if secrets.is_empty() {
                bail!("Must spend at least one coin");
            }
            if new_coins.is_empty() {
                bail!("Must create at least one coin");
            }

            let old_coins: Vec<[u8; 32]> = secrets.iter().map(|s| hash(s)).collect();

            // Verify commitment exists
            let expected = compute_commitment(&old_coins, new_coins, salt);
            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found");
            }

            // Verify coins exist
            for old_coin in &old_coins {
                if !state.coins.contains(old_coin) {
                    bail!("Coin {:?} not found", hex::encode(old_coin));
                }
            }

            Ok(())
        }
    }
}
