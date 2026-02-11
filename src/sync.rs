use crate::core::State;
use crate::core::state::apply_batch;
use crate::storage::Storage;
use crate::network::{Message, MidstateNetwork, NetworkEvent};
use anyhow::Result;
use libp2p::PeerId;

pub struct Syncer {
    storage: Storage,
}

impl Syncer {
    pub fn new(storage: Storage) -> Self {
        Self { storage }
    }

    pub async fn sync_via_network(&self, network: &mut MidstateNetwork, peer: PeerId) -> Result<State> {
        tracing::info!("Starting trustless sync from genesis...");

        // Ask peer for their height
        network.send(peer, Message::GetState);

        let peer_height = loop {
            match network.next_event().await {
                NetworkEvent::MessageReceived { message: Message::StateInfo { height, .. }, .. } => {
                    break height;
                }
                _ => continue,
            }
        };

        tracing::info!("Peer is at height {}, downloading batches...", peer_height);

        let batch_size = 100u64;
        let mut current = 0u64;

        while current < peer_height {
            let count = batch_size.min(peer_height - current);

            tracing::info!("Requesting batches {}-{}", current, current + count - 1);

            network.send(peer, Message::GetBatches {
                start_height: current,
                count,
            });

            let batches = loop {
                match network.next_event().await {
                    NetworkEvent::MessageReceived { message: Message::Batches { batches, .. }, .. } => {
                        break batches;
                    }
                    _ => continue,
                }
            };

            if batches.is_empty() {
                anyhow::bail!("Peer returned empty batches at height {}", current);
            }

            for (i, batch) in batches.iter().enumerate() {
                let height = current + i as u64;
                self.storage.save_batch(height, batch)?;
            }

            current += batches.len() as u64;
        }

        tracing::info!("Downloaded all batches, verifying...");

        let state = self.verify_chain(peer_height)?;

        tracing::info!("Sync complete! Height: {}", state.height);

        Ok(state)
    }

    fn verify_chain(&self, target_height: u64) -> Result<State> {
        let mut state = State::genesis().0;

        for height in 0..target_height {
            if height % 100 == 0 {
                tracing::info!("Verified {}/{} batches", height, target_height);
            }

            let batch = self.storage.load_batch(height)?
                .ok_or_else(|| anyhow::anyhow!("Missing batch at height {}", height))?;

            apply_batch(&mut state, &batch)?;
        }

        Ok(state)
    }
}
