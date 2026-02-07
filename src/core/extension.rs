use super::types::*;
use anyhow::{bail, Result};

/// Create an extension by doing sequential work
pub fn create_extension(midstate: [u8; 32], nonce: u64) -> Extension {
    let mut x = hash_concat(&midstate, &nonce.to_le_bytes());
    
    for _ in 0..EXTENSION_ITERATIONS {
        x = hash(&x);
    }
    
    Extension {
        nonce,
        final_hash: x,
    }
}

/// Verify an extension (requires redoing the work)
pub fn verify_extension(midstate: [u8; 32], ext: &Extension, target: &[u8; 32]) -> Result<()> {
    let mut x = hash_concat(&midstate, &ext.nonce.to_le_bytes());
    
    for _ in 0..EXTENSION_ITERATIONS {
        x = hash(&x);
    }
    
    if x != ext.final_hash {
        bail!("Extension verification failed: hash mismatch");
    }
    
    if ext.final_hash >= *target {
        bail!("Extension doesn't meet difficulty target");
    }
    
    Ok(())
}

/// Mine with WORK-FIRST approach
/// Each attempt costs EXTENSION_ITERATIONS of sequential work
pub fn mine_extension(midstate: [u8; 32], target: [u8; 32]) -> Extension {
    let mut attempts = 0u64;
    
    loop {
        attempts += 1;
        let nonce: u64 = rand::random();
        
        // STEP 1: Pay the sequential work cost (EXPENSIVE - can't parallelize)
        let mut x = hash_concat(&midstate, &nonce.to_le_bytes());
        for _ in 0..EXTENSION_ITERATIONS {
            x = hash(&x);
        }
        
        // STEP 2: Check if this ticket won the lottery
        if x < target {
            tracing::info!(
                "Found valid extension! nonce={} attempts={} hash={}",
                nonce,
                attempts,
                hex::encode(x)
            );
            return Extension {
                nonce,
                final_hash: x,
            };
        }
        
        // No win - must pay full sequential cost again for next ticket
    }
}
