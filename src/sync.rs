use crate::core::{State, BatchHeader};
use crate::core::state::apply_batch;
use crate::core::extension::verify_extension;
use crate::storage::Storage;
use crate::network::{Message, MidstateNetwork, NetworkEvent};
use anyhow::{bail, Result};
use libp2p::PeerId;

pub struct Syncer {
    storage: Storage,
}

impl Syncer {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    pub async fn sync_via_network(&self, network: &mut MidstateNetwork, peer: PeerId) -> Result<State> {
        tracing::info!("Starting headers-first sync...");

        // Load current state to know where we are
        let mut state = self.storage.load_state()?.unwrap_or_else(|| State::genesis().0);
        let start_height = state.height;

        // Ask peer for their height
        network.send(peer, Message::GetState);

        let (peer_height, _) = loop {
            match network.next_event().await {
                NetworkEvent::MessageReceived { message: Message::StateInfo { height, depth, .. }, .. } => {
                    break (height, depth);
                }
                _ => continue,
            }
        };

        // Already caught up?
        if peer_height <= start_height {
            tracing::info!("Already at peer height {}, no sync needed", start_height);
            return Ok(state);
        }

        tracing::info!(
            "Peer at height {}. Syncing headers from {} to {}...",
            peer_height,
            start_height,
            peer_height
        );

        // 2. Download and verify headers
        let mut headers_buffer: Vec<BatchHeader> = Vec::new();
        let mut current_h = start_height;  // ← Start where we left off, not 0!

        while current_h < peer_height {
            let count = 100.min(peer_height - current_h);
            network.send(peer, Message::GetHeaders { start_height: current_h, count });

            let received_headers = loop {
                match network.next_event().await {
                    NetworkEvent::MessageReceived { message: Message::Headers { headers, .. }, .. } => break headers,
                    _ => continue,
                }
            };

            if received_headers.is_empty() {
                bail!("Peer sent empty headers");
            }
            
            // VERIFY HEADERS
            for (i, header) in received_headers.iter().enumerate() {
                let height = current_h + i as u64;
                
                // A. Check Linkage
                if height > start_height {
                    // For first batch, check against our current state
                    if headers_buffer.is_empty() && height == start_height {
                        if header.prev_midstate != state.midstate {
                            bail!("Header linkage broken at height {}: expected prev={}, got prev={}",
                                height, hex::encode(state.midstate), hex::encode(header.prev_midstate));
                        }
                    } else if !headers_buffer.is_empty() {
                        // Check against previous header in buffer
                        let prev_idx = headers_buffer.len() - 1;
                        let prev = &headers_buffer[prev_idx];
                        
                        if header.prev_midstate != prev.extension.final_hash {
                            bail!("Header linkage broken at height {}", height);
                        }
                    }
                }

                // B. Verify PoW
                verify_extension(header.post_tx_midstate, &header.extension, &header.target)
                    .map_err(|e| anyhow::anyhow!("Invalid PoW at height {}: {}", height, e))?;

                headers_buffer.push(header.clone());
            }

            current_h += received_headers.len() as u64;
            tracing::info!("Verified {} headers (PoW valid)", current_h - start_height);
        }

        tracing::info!("All {} headers verified. Downloading blocks...", headers_buffer.len());

        // 3. Download Blocks and Full Verify
        let mut current_dl = 0;
        while current_dl < headers_buffer.len() as u64 {
            let count = 10.min(headers_buffer.len() as u64 - current_dl);
            
            network.send(peer, Message::GetBatches { 
                start_height: start_height + current_dl,  // ← Offset by start_height
                count 
            });
            
            let batches = loop {
                match network.next_event().await {
                    NetworkEvent::MessageReceived { message: Message::Batches { batches, .. }, .. } => break batches,
                    _ => continue,
                }
            };

            if batches.is_empty() {
                bail!("Peer sent empty batches at height {}", start_height + current_dl);
            }

            for (i, batch) in batches.iter().enumerate() {
                let height = start_height + current_dl + i as u64;
                let header_idx = (current_dl + i as u64) as usize;
                
                if header_idx >= headers_buffer.len() {
                    bail!("Batch index {} exceeds header buffer length {}", header_idx, headers_buffer.len());
                }
                
                let header = &headers_buffer[header_idx];

                // A. Integrity Check
                let batch_header_calc = batch.header();
                
                if batch.extension.final_hash != header.extension.final_hash {
                    bail!("Batch at {} does not match verified header PoW", height);
                }
                if batch_header_calc.post_tx_midstate != header.post_tx_midstate {
                    bail!("Batch at {} transactions do not match header commitment", height);
                }

                // B. Full Application
                apply_batch(&mut state, batch)?;
                self.storage.save_batch(height, batch)?;
            }
            
            current_dl += batches.len() as u64;
            tracing::info!("Applied {}/{} blocks", current_dl, headers_buffer.len());
        }
 
        tracing::info!("Sync complete! Height: {}, Depth: {}", state.height, state.depth);
        Ok(state)
    }
}
