//! Pruning License reputation and challenge manager.
//!
//! Extracted from the node god module (Phase 2 of refactor).
//! Handles:
//! - Registration of held licenses (pruning exemption) and issued licenses (archiver obligations).
//! - Bayesian reputation tracking per (peer, license_commitment).
//! - Pending MMR Gossip Challenges + DataHash verification gates.
//! - Advertising via chat attachments (coordinated with chat module).
//!
//! The actual sending of challenges and GetBatches exemption logic remains
//! wired in Node for now (tight coupling to network handlers and state), but
//! the pure state + scoring lives here.

use std::collections::HashMap;
use std::time::Instant;
use libp2p::PeerId;

#[derive(Clone, Default)]
pub struct LicenseManager {
    /// Peers that have advertised they *hold* certain Pruning Licenses (via Chat Commitment attachments).
    pub advertised_licenses: HashMap<PeerId, Vec<([u8; 32], u64)>>, // (commitment, weight)

    /// Bayesian reputation (alpha = successful responses, beta = failures/timeouts)
    /// for peers holding licenses, per license commitment.
    pub license_reputations: HashMap<PeerId, HashMap<[u8; 32], (u32, u32)>>,

    /// Licenses this node itself claims to *hold* (for exemption from serving historical data when pruning).
    pub my_license_ranges: Vec<([u8; 32], u64, u64)>, // (commitment, min_height, max_height)

    /// Licenses this node has *issued* (as the original Archiver / Issuer).
    pub my_issued_license_ranges: Vec<([u8; 32], u64, u64)>, // (commitment, min_height, max_height)

    /// Pending MMR Gossip Challenges we have sent to peers.
    /// Key: (peer, license_commitment, height)
    /// Value: (salt we sent, time sent)
    pub pending_license_challenges: HashMap<(PeerId, [u8; 32], u64), ([u8; 32], Instant)>,

    /// Claims received via DataHash replies to our LicenseChallenges, awaiting
    /// actual batch data arrival for cryptographic verification.
    pub pending_data_verifications: HashMap<u64, Vec<(PeerId, [u8; 32], [u8; 32])>>,
}

impl LicenseManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the license ranges this node *holds* (from its operator wallet).
    /// These give pruning exemption rights.
    pub fn register_my_licenses(&mut self, ranges: Vec<([u8; 32], u64, u64)>) {
        self.my_license_ranges = ranges;
    }

    /// Register licenses this node has *issued* as an Archiver (the 'issuer' field in LicenseMetadata).
    /// These define permanent storage/audit obligations under the Cap-and-Trade model.
    pub fn register_issued_licenses(&mut self, ranges: Vec<([u8; 32], u64, u64)>) {
        self.my_issued_license_ranges = ranges;
    }

    /// Record a successful data verification for a license-bearing peer (bumps alpha).
    pub fn credit_license_reputation_on_data_verified(&mut self, peer: PeerId, commitment: [u8; 32]) {
        let entry = self.license_reputations
            .entry(peer)
            .or_default()
            .entry(commitment)
            .or_insert((1, 1));
        entry.0 = entry.0.saturating_add(1).min(10_000);
    }

    /// Return a reliability score [0.0, 1.0] for a peer's license.
    /// Used by GetBatches exemption / rate limit scaling.
    pub fn get_license_reliability(&self, peer: PeerId, commitment: [u8; 32]) -> f32 {
        if let Some(per_license) = self.license_reputations.get(&peer) {
            if let Some(&(alpha, beta)) = per_license.get(&commitment) {
                let total = (alpha + beta) as f32;
                if total > 0.0 {
                    return (alpha as f32 / total).clamp(0.0, 1.0);
                }
            }
        }
        0.5 // neutral prior
    }

    // Additional helpers (challenge tick, timeout penalties, advertised lookup)
    // can be moved here in follow-up iterations as the coupling to chat + handle_message
    // is untangled.
}

