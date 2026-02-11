use crate::core::State;
use crate::core::state::apply_batch;
use crate::storage::Storage;
use crate::network::{Message, PeerConnection};
use anyhow::Result;

pub struct Syncer {
    storage: Storage,
}

impl Syncer {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }
    
    /// Sync from genesis (trustless)
    pub async fn sync_from_genesis(&self, peer: &mut PeerConnection) -> Result<State> {
        tracing::info!("Starting trustless sync from genesis...");
        
        // 1. Ask peer for their height
        peer.send_message(&Message::GetState).await?;
        
        let peer_height = match peer.receive_message().await? {
            Message::StateInfo { height, .. } => height,
            _ => anyhow::bail!("Expected state info"),
        };
        
        tracing::info!("Peer is at height {}, downloading batches...", peer_height);
        
        // 2. Download all batches
        let batch_size = 100u64;
        let mut current = 1u64;
        
        while current <= peer_height {
            let count = batch_size.min(peer_height - current+1);
            
            tracing::info!("Requesting batches {}-{}", current, current + count - 1);
            
            peer.send_message(&Message::GetBatches {
                start_height: current,
                count,
            }).await?;
            
            match peer.receive_message().await? {
                Message::Batches { batches, .. } => {
                    // Save batches
                    for (i, batch) in batches.iter().enumerate() {
                        let height = current + i as u64;
                        self.storage.save_batch(height, batch)?;
                    }
                    
                    current += batches.len() as u64;
                }
                _ => anyhow::bail!("Expected batches"),
            }
        }
        
        tracing::info!("Downloaded all batches, verifying...");
        
        // 3. Verify and apply all batches
        let state = self.verify_chain(peer_height)?;
        
        tracing::info!("Sync complete! Height: {}", state.height);
        
        Ok(state)
    }
    
    /// Verify entire chain from genesis
    fn verify_chain(&self, target_height: u64) -> Result<State> {
        let mut state = State::genesis().0;
        
        for height in 1..target_height {
            if height % 100 == 0 {
                tracing::info!("Verified {}/{} batches", height, target_height);
            }
            
            let batch = self.storage.load_batch(height)?
                .ok_or_else(|| anyhow::anyhow!("Missing batch at height {}", height))?;
            
            // This verifies the extension (redoes sequential work)
            apply_batch(&mut state, &batch)?;
        }
        
        Ok(state)
    }
    
    /// Quick catchup for returning nodes
    pub async fn catchup(&self, current_height: u64, peer: &mut PeerConnection) -> Result<State> {
        tracing::info!("Catching up from height {}", current_height);
        
        // Load current state
        let mut state = self.storage.load_state()?
            .ok_or_else(|| anyhow::anyhow!("No saved state"))?;
        
        // Request missing batches
        peer.send_message(&Message::GetState).await?;
        
        let peer_height = match peer.receive_message().await? {
            Message::StateInfo { height, .. } => height,
            _ => anyhow::bail!("Expected state info"),
        };
        
        if peer_height <= current_height {
            tracing::info!("Already up to date");
            return Ok(state);
        }
        
        let needed = peer_height - current_height;
        tracing::info!("Need {} batches to catch up", needed);
        
        // Download and apply
        peer.send_message(&Message::GetBatches {
            start_height: current_height+1,
            count: needed,
        }).await?;
        
        match peer.receive_message().await? {
            Message::Batches { batches, .. } => {
                for batch in batches {
                    apply_batch(&mut state, &batch)?;
                    self.storage.save_batch(state.height, &batch)?;
                }
            }
            _ => anyhow::bail!("Expected batches"),
        }
        
        tracing::info!("Caught up to height {}", state.height);
        
        Ok(state)
    }
}
