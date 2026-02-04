use super::types::*;
use super::transaction::apply_transaction;
use super::extension::verify_extension;
use anyhow::Result;

/// Apply a batch to the state
pub fn apply_batch(state: &mut State, batch: &Batch) -> Result<()> {
    // Apply all transactions
    for tx in &batch.transactions {
        apply_transaction(state, tx)?;
    }
    
    // Verify extension
    verify_extension(state.midstate, &batch.extension, &state.target)?;
    
    // Update state with extension
    state.midstate = batch.extension.final_hash;
    state.depth += EXTENSION_ITERATIONS;
    state.height += 1;
    
    tracing::info!(
        "Applied batch: height={} depth={} coins={}",
        state.height,
        state.depth,
        state.coins.len()
    );
    
    Ok(())
}

/// Choose the better of two states (fork resolution)
pub fn choose_best_state<'a>(a: &'a State, b: &'a State) -> &'a State {
    match a.depth.cmp(&b.depth) {
        std::cmp::Ordering::Greater => {
            tracing::debug!("Chose state A (depth {} > {})", a.depth, b.depth);
            a
        }
        std::cmp::Ordering::Less => {
            tracing::debug!("Chose state B (depth {} > {})", b.depth, a.depth);
            b
        }
        std::cmp::Ordering::Equal => {
            if a.midstate < b.midstate {
                tracing::debug!("Chose state A (tiebreaker)");
                a
            } else {
                tracing::debug!("Chose state B (tiebreaker)");
                b
            }
        }
    }
}
