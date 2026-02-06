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

/// Mine a batch by finding a valid extension
pub fn mine_extension(midstate: [u8; 32], target: [u8; 32]) -> Extension {
    loop {
        let nonce: u64 = rand::random();
        let ext = create_extension(midstate, nonce);
        
        if ext.final_hash < target {
            tracing::info!(
                "Found valid extension! nonce={} hash={}",
                nonce,
                hex::encode(ext.final_hash)
            );
            return ext;
        }
    }
}
