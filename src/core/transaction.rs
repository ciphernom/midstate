use super::types::*;
use anyhow::{bail, Result};

/// Apply a transaction to the state
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    // Compute the coins being spent
    let old_coins = tx.input_coins();
    
    if old_coins.is_empty() {
        bail!("Transaction must spend at least one coin");
    }
    
    // Check all coins exist and remove them
    for old_coin in &old_coins {
        if !state.coins.remove(old_coin) {
            bail!("Coin {:?} not found or already spent", hex::encode(old_coin));
        }
    }
    
    // Validate new coins
    if tx.new_coins.is_empty() {
        bail!("Transaction must create at least one new coin");
    }
    
    // Add new coins
    for new_coin in &tx.new_coins {
        if !state.coins.insert(*new_coin) {
            bail!("Duplicate coin created");
        }
    }
    
    // Update midstate with transaction data
    let tx_bytes = bincode::serialize(tx)?;
    state.midstate = hash_concat(&state.midstate, &tx_bytes);
    
    Ok(())
}

/// Validate a transaction without applying it
pub fn validate_transaction(state: &State, tx: &Transaction) -> Result<()> {
    let old_coins = tx.input_coins();
    
    if old_coins.is_empty() {
        bail!("Must spend at least one coin");
    }
    
    for old_coin in &old_coins {
        if !state.coins.contains(old_coin) {
            bail!("Coin {:?} not found", hex::encode(old_coin));
        }
    }
    
    if tx.new_coins.is_empty() {
        bail!("Must create at least one coin");
    }
    
    Ok(())
}
