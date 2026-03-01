//! Node-side CoinJoin mix coordination.
//!
//! Wraps [`wallet::coinjoin::MixSession`] with peer tracking, signature
//! collection, and phase management. The [`MixManager`] is shared between
//! the node event loop (which drives p2p messages) and the RPC layer
//! (which drives local wallet interactions).

use crate::core::types::*;
use crate::wallet::coinjoin::{MixSession, MixProposal};
use anyhow::{bail, Result};
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Session timeout in seconds. Stale sessions are garbage collected.
const MIX_SESSION_TIMEOUT: u64 = 300;

/// Maximum concurrent mix sessions per node.
const MAX_MIX_SESSIONS: usize = 16;

/// Proof-of-Work bits required for MixJoin/MixFee messages.
/// 20 bits ≈ 1M BLAKE3 hashes ≈ ~100ms on modern hardware.
/// Prevents costless Sybil flooding of CoinJoin sessions with rotating PeerIds.
pub const MIX_JOIN_POW_BITS: u32 = 20;

/// Verify the PoW accompanying a MixJoin or MixFee message.
/// The challenge is: hash(mix_id || coin_id || nonce_le) must have
/// at least MIX_JOIN_POW_BITS leading zero bits.
pub fn verify_mix_join_pow(mix_id: &[u8; 32], coin_id: &[u8; 32], nonce: u64) -> bool {
    let mut data = Vec::with_capacity(72);
    data.extend_from_slice(mix_id);
    data.extend_from_slice(coin_id);
    data.extend_from_slice(&nonce.to_le_bytes());
    let h = hash(&data);
    count_leading_zeros(&h) >= MIX_JOIN_POW_BITS
}

/// Mine a MixJoin PoW nonce for the given mix_id and coin_id.
pub fn mine_mix_join_pow(mix_id: &[u8; 32], coin_id: &[u8; 32]) -> u64 {
    for nonce in 0u64.. {
        if verify_mix_join_pow(mix_id, coin_id, nonce) {
            return nonce;
        }
    }
    unreachable!()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MixPhase {
    /// Collecting registrations from participants.
    Collecting,
    /// Proposal built, awaiting signatures.
    Signing,
    /// All signatures collected, Commit transaction submitted.
    CommitSubmitted,
    /// Reveal transaction submitted, mix complete.
    Complete,
    /// Mix failed or timed out.
    Failed(String),
}

/// A participant in the mix — either local (via RPC) or remote (via p2p).
#[derive(Clone, Debug)]
struct Participant {
    /// None if this participant registered via local RPC.
    peer: Option<PeerId>,
    /// The coin_id this participant registered, for griefer identification.
    coin_id: [u8; 32],
}

/// Node-side state for a single CoinJoin session.
pub struct NodeMixSession {
    session: MixSession,
    participants: Vec<Participant>,
    fee_participant: Option<Participant>,
    proposal: Option<MixProposal>,
    signatures: HashMap<usize, Vec<u8>>,
    pub phase: MixPhase,
    pub phase_started_at: u64,
    created_at: u64,
    /// True if this node initiated the session (is the coordinator).
    pub is_coordinator: bool,
    /// The coordinator peer (if we're a joiner).
    pub coordinator_peer: Option<PeerId>,
    /// Outputs registered by the LOCAL wallet (peer = None).
    /// Used to verify that a coordinator's proposal actually includes our outputs.
    local_outputs: Vec<[u8; 32]>,
}

/// Snapshot of a mix session exposed to the RPC layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixStatusSnapshot {
    pub mix_id: String,
    pub denomination: u64,
    pub participants: usize,
    pub phase: MixPhase,
    /// Set when phase == Signing; the commitment the wallet needs to sign.
    pub commitment: Option<String>,
    /// Set when phase == Signing; serialized proposal inputs for the wallet
    /// to find its own input index.
    pub input_coin_ids: Vec<String>,
}

/// Manages all active mix sessions for a node.
pub struct MixManager {
    sessions: HashMap<[u8; 32], NodeMixSession>,
    /// Peers that failed to sign during a mix session, suggesting griefing.
    /// We ban the PEER (not the coin) because coin salts are public on-chain,
    /// meaning anyone can construct valid InputReveals for others' coins.
    /// Banning coins would let an attacker lock victims out of CoinJoin.
    banned_peers: HashSet<PeerId>,
}

impl MixManager {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            banned_peers: HashSet::new(), 
        }
    }

    /// Fails any active mix sessions reliant on a disconnected peer
    pub fn handle_peer_disconnect(&mut self, peer: PeerId) {
        for (_, ns) in self.sessions.iter_mut() {
            if matches!(ns.phase, MixPhase::Complete | MixPhase::Failed(_)) {
                continue;
            }
            let is_involved = ns.coordinator_peer == Some(peer)
                || ns.participants.iter().any(|p| p.peer == Some(peer))
                || ns.fee_participant.as_ref().map_or(false, |p| p.peer == Some(peer));

            if is_involved {
                ns.phase = MixPhase::Failed(format!("peer {} disconnected", peer));
            }
        }
    }

    /// Create a new mix session as coordinator. Returns the mix_id.
    pub fn create_session(&mut self, denomination: u64, min_participants: usize) -> Result<[u8; 32]> {
        if self.sessions.len() >= MAX_MIX_SESSIONS {
            bail!("too many active mix sessions");
        }
        let session = MixSession::new(denomination, min_participants)?;
        let mix_id: [u8; 32] = rand::random();
        self.sessions.insert(mix_id, NodeMixSession {
            session,
            participants: Vec::new(),
            fee_participant: None,
            proposal: None,
            signatures: HashMap::new(),
            phase: MixPhase::Collecting,
            phase_started_at: now(),
            created_at: now(),
            is_coordinator: true,
            coordinator_peer: None,
            local_outputs: Vec::new(),
        });
        Ok(mix_id)
    }

    /// Create a session as a joiner (responding to a peer's MixAnnounce).
    pub fn create_joining_session(
        &mut self,
        mix_id: [u8; 32],
        denomination: u64,
        coordinator: PeerId,
    ) -> Result<()> {
        if self.sessions.contains_key(&mix_id) {
            bail!("session already exists");
        }
        if self.sessions.len() >= MAX_MIX_SESSIONS {
            bail!("too many active mix sessions");
        }
        let session = MixSession::new(denomination, 2)?;
        self.sessions.insert(mix_id, NodeMixSession {
            session,
            participants: Vec::new(),
            fee_participant: None,
            proposal: None,
            signatures: HashMap::new(),
            phase: MixPhase::Collecting,
            phase_started_at: now(),
            created_at: now(),
            is_coordinator: false,
            coordinator_peer: Some(coordinator),
            local_outputs: Vec::new(),
        });
        Ok(())
    }

    /// Register a participant (local or remote).
pub fn register(
        &mut self,
        mix_id: &[u8; 32],
        input: InputReveal,
        output: OutputData,
        _signature: &[u8],  // DEPRECATED: kept for wire compat, not verified
        peer: Option<PeerId>,
    ) -> Result<()> {
        let coin_id = input.coin_id();
        
        // SECURITY NOTE: We intentionally do NOT verify a WOTS signature here.
        // WOTS keys are strictly one-time-use. Signing the mix_id during
        // registration AND signing the proposal commitment during the Signing
        // phase would constitute two uses of the same key, enabling forgery.
        //
        // NOTE: The InputReveal (predicate + value + salt) is NOT a secret —
        // all three fields are broadcast in cleartext when a coin is spent.
        // This means a third party can register someone else's coins in a mix.
        // To mitigate griefing, we ban the PEER on signing timeout, not the coin.
        // The actual spending signature happens exactly once in the Signing phase.

        if let Some(p) = peer {
            if self.banned_peers.contains(&p) {
                bail!("Peer is banned from mixing due to a previous signature timeout");
            }
        }
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        if ns.phase != MixPhase::Collecting {
            bail!("session not accepting registrations (phase: {:?})", ns.phase);
        }

        ns.session.register(input.clone(), output.clone())?;
        // Track outputs from LOCAL wallet so we can verify the coordinator's
        // proposal includes them (prevents proposal-stealing attack).
        if peer.is_none() {
            ns.local_outputs.push(output.hash_for_commitment());
        }
        ns.participants.push(Participant { peer, coin_id });
        Ok(())
    }

    /// Set the fee input for a session.
    pub fn set_fee_input(
        &mut self,
        mix_id: &[u8; 32],
        input: InputReveal,
        peer: Option<PeerId>,
    ) -> Result<()> {
        if let Some(p) = peer {
            if self.banned_peers.contains(&p) {
                bail!("Peer is banned from mixing due to a previous signature timeout");
            }
        }
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        if ns.phase != MixPhase::Collecting {
            bail!("session not accepting registrations");
        }

        ns.session.set_fee_input(input.clone())?;
        let coin_id = input.coin_id();
        ns.fee_participant = Some(Participant { peer, coin_id });
        Ok(())
    }

    /// Try to advance a session to the Signing phase.
    /// Returns the proposal if the session just became ready.
    pub fn try_finalize(&mut self, mix_id: &[u8; 32]) -> Result<Option<MixProposal>> {
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        if ns.phase != MixPhase::Collecting || !ns.session.is_ready() {
            return Ok(None);
        }

    let proposal = ns.session.proposal()?;
        ns.proposal = Some(proposal.clone());
        ns.phase = MixPhase::Signing;
        ns.phase_started_at = now(); 
        Ok(Some(proposal))
    }

    /// Record a signature for an input in the proposal.
    pub fn add_signature(
        &mut self,
        mix_id: &[u8; 32],
        input_index: usize,
        signature: Vec<u8>,
        current_height: u64,
        peer: Option<PeerId>,
    ) -> Result<()> {
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        if ns.phase != MixPhase::Signing {
            bail!("session not in signing phase");
        }

        let proposal = ns.proposal.as_ref()
            .ok_or_else(|| anyhow::anyhow!("no proposal"))?;

        if input_index >= proposal.inputs.len() {
            bail!("input_index {} out of range ({})", input_index, proposal.inputs.len());
        }

        // Cryptographic verification: execute the predicate script against the
        // provided signature BEFORE accepting it. Without this, an attacker can
        // submit garbage bytes, forcing the coordinator to waste CPU mining the
        // Commit PoW only for the Reveal to fail mempool validation.
        let input = &proposal.inputs[input_index];
        let is_valid = match &input.predicate {
            crate::core::types::Predicate::Script { bytecode } => {
                let ctx = crate::core::script::ExecContext {
                    commitment: &proposal.commitment,
                    height: current_height,
                    outputs: &proposal.outputs,
                };
                crate::core::script::execute_script(bytecode, &[signature.clone()], &ctx).is_ok()
            }
        };

        if !is_valid {
            if let Some(p) = peer {
                self.banned_peers.insert(p);
            }
            bail!("Invalid signature for input {}", input_index);
        }

        ns.signatures.insert(input_index, signature);
        Ok(())
    }

    /// Check if all signatures are collected and build the final transaction.
    pub fn try_build_transaction(&mut self, mix_id: &[u8; 32]) -> Result<Option<Transaction>> {
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        if ns.phase != MixPhase::Signing {
            return Ok(None);
        }

        let proposal = ns.proposal.as_ref()
            .ok_or_else(|| anyhow::anyhow!("no proposal"))?;

        if ns.signatures.len() != proposal.inputs.len() {
            return Ok(None); // still waiting
        }

        // Collect signatures in canonical order
        let sigs: Vec<Vec<u8>> = (0..proposal.inputs.len())
            .map(|i| ns.signatures.get(&i).cloned()
                .ok_or_else(|| anyhow::anyhow!("missing signature for input {}", i)))
            .collect::<Result<_>>()?;

        let tx = ns.session.build_reveal(sigs)?;
        Ok(Some(tx))
    }

    /// Mark a session phase.
    pub fn set_phase(&mut self, mix_id: &[u8; 32], phase: MixPhase) {
        if let Some(ns) = self.sessions.get_mut(mix_id) {
            ns.phase = phase;
        }
    }

    /// Apply a received proposal from a coordinator peer.
    pub fn apply_remote_proposal(
        &mut self,
        mix_id: &[u8; 32],
        inputs: Vec<InputReveal>,
        outputs: Vec<OutputData>,
        salt: [u8; 32],
        commitment: [u8; 32],
    ) -> Result<()> {
        let ns = self.sessions.get_mut(mix_id)
            .ok_or_else(|| anyhow::anyhow!("mix session not found"))?;

        // Verify the commitment matches
        let input_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let output_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
        let expected = compute_commitment(&input_ids, &output_ids, &salt);
        if expected != commitment {
            bail!("proposal commitment mismatch");
        }

        // SECURITY: Verify that ALL outputs we registered locally are present
        // in the proposal. Without this check, a malicious coordinator could
        // replace our destination address with their own.
        for local_hash in &ns.local_outputs {
            if !output_ids.contains(local_hash) {
                bail!(
                    "Proposal is missing our registered output {} — possible proposal-stealing attack",
                    hex::encode(local_hash)
                );
            }
        }

        ns.proposal = Some(MixProposal {
            inputs: inputs.clone(),
            outputs,
            salt,
            commitment,
        });
        ns.phase = MixPhase::Signing;
        Ok(())
    }

    /// Get status snapshot for RPC.
    pub fn status(&self, mix_id: &[u8; 32]) -> Option<MixStatusSnapshot> {
        let ns = self.sessions.get(mix_id)?;
        Some(MixStatusSnapshot {
            mix_id: hex::encode(mix_id),
            denomination: ns.session.denomination(),
            participants: ns.session.participant_count(),
            phase: ns.phase.clone(),
            commitment: ns.proposal.as_ref().map(|p| hex::encode(p.commitment)),
            input_coin_ids: ns.proposal.as_ref()
                .map(|p| p.inputs.iter().map(|i| hex::encode(i.coin_id())).collect())
                .unwrap_or_default(),
        })
    }

    /// List all active mix sessions.
    pub fn list_sessions(&self) -> Vec<MixStatusSnapshot> {
        self.sessions.iter()
            .map(|(id, _)| self.status(id).unwrap())
            .collect()
    }

    /// Get remote peers to notify for a session.
    pub fn remote_participants(&self, mix_id: &[u8; 32]) -> Vec<PeerId> {
        let Some(ns) = self.sessions.get(mix_id) else { return vec![]; };
        let mut peers: Vec<PeerId> = ns.participants.iter()
            .filter_map(|p| p.peer)
            .collect();
        if let Some(fp) = &ns.fee_participant {
            if let Some(peer) = fp.peer {
                if !peers.contains(&peer) {
                    peers.push(peer);
                }
            }
        }
        peers
    }

    /// Find the input index in the proposal for a given coin_id.
    pub fn find_input_index(&self, mix_id: &[u8; 32], coin_id: &[u8; 32]) -> Option<usize> {
        let ns = self.sessions.get(mix_id)?;
        let proposal = ns.proposal.as_ref()?;
        proposal.inputs.iter().position(|i| i.coin_id() == *coin_id)
    }

    /// Get session existence and coordinator status.
    pub fn get_session_info(&self, mix_id: &[u8; 32]) -> Option<(bool, Option<PeerId>)> {
        self.sessions.get(mix_id).map(|ns| (ns.is_coordinator, ns.coordinator_peer))
    }

    /// Remove timed-out and completed sessions.
    pub fn cleanup(&mut self) {
        let now_time = now();
        let mut peers_to_ban = Vec::new();

        self.sessions.retain(|id, ns| {
            // Griefer detection: 60 seconds in Signing phase without finishing
            if ns.phase == MixPhase::Signing && now_time.saturating_sub(ns.phase_started_at) > 60 {
                if let Some(proposal) = &ns.proposal {
                    for (i, input) in proposal.inputs.iter().enumerate() {
                        if !ns.signatures.contains_key(&i) {
                            let cid = input.coin_id();
                            // Find the PEER that registered this coin_id.
                            // Only ban remote peers — local wallet can't grief itself.
                            let peer = ns.participants.iter()
                                .find(|p| p.coin_id == cid)
                                .and_then(|p| p.peer)
                                .or_else(|| ns.fee_participant.as_ref()
                                    .filter(|fp| fp.coin_id == cid)
                                    .and_then(|fp| fp.peer));

                            if let Some(p) = peer {
                                peers_to_ban.push(p);
                                tracing::warn!(
                                    "Peer {} banned for stalling mix {} (unsigned input {})",
                                    p, hex::encode(id), hex::encode(cid)
                                );
                            }
                        }
                    }
                }
                ns.phase = MixPhase::Failed("signing timed out".to_string());
                ns.phase_started_at = now_time; // Reset so it gets swept cleanly
            }

            match &ns.phase {
                MixPhase::Complete | MixPhase::Failed(_) => {
                    now_time.saturating_sub(ns.phase_started_at) < 30
                }
                _ => {
                    now_time.saturating_sub(ns.created_at) < MIX_SESSION_TIMEOUT
                }
            }
        });

        for peer in peers_to_ban {
            self.banned_peers.insert(peer);
        }
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wots;
    use crate::core::types::Predicate;

    fn make_input(name: &[u8], value: u64) -> InputReveal {
        let seed = hash(name);
        let pk = wots::keygen(&seed);
        InputReveal { predicate: Predicate::p2pk(&pk), value, salt: hash_concat(name, b"salt") }
    }
    // Helper to dynamically generate a valid signature for the test inputs
    fn test_sig(name: &[u8], mix_id: &[u8; 32]) -> Vec<u8> {
        let seed = crate::core::types::hash(name);
        crate::core::wots::sig_to_bytes(&crate::core::wots::sign(&seed, mix_id))
    }
    fn make_output(name: &[u8], value: u64) -> OutputData {
        OutputData::Standard { address: hash_concat(name, b"dest"), value, salt: hash_concat(name, b"osalt") }
    }

    #[test]
    fn create_and_register() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();

        mgr.register(&mix_id, make_input(b"alice", 8), make_output(b"alice", 8), &test_sig(b"alice", &mix_id), None).unwrap();
        mgr.register(&mix_id, make_input(b"bob", 8), make_output(b"bob", 8), &test_sig(b"bob", &mix_id), None).unwrap();

        let status = mgr.status(&mix_id).unwrap();
        assert_eq!(status.participants, 2);
        assert_eq!(status.phase, MixPhase::Collecting);
    }

    #[test]
    fn rejects_wrong_denomination() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();
        assert!(mgr.register(&mix_id, make_input(b"bad", 4), make_output(b"bad", 8), &test_sig(b"bad", &mix_id), None).is_err());
    }

    #[test]
    fn finalize_produces_proposal() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();

        mgr.register(&mix_id, make_input(b"alice", 8), make_output(b"alice", 8), &test_sig(b"alice", &mix_id), None).unwrap();
        mgr.register(&mix_id, make_input(b"bob", 8), make_output(b"bob", 8), &test_sig(b"bob", &mix_id), None).unwrap();
        mgr.set_fee_input(&mix_id, make_input(b"fee", 1), None).unwrap();

        let proposal = mgr.try_finalize(&mix_id).unwrap();
        assert!(proposal.is_some());

        let status = mgr.status(&mix_id).unwrap();
        assert_eq!(status.phase, MixPhase::Signing);
        assert!(status.commitment.is_some());
    }

    #[test]
    fn finalize_returns_none_when_not_ready() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();
       mgr.register(&mix_id, make_input(b"alice", 8), make_output(b"alice", 8), &test_sig(b"alice", &mix_id), None).unwrap();
        // Only 1 of 2 participants
        assert!(mgr.try_finalize(&mix_id).unwrap().is_none());
    }

    #[test]
    fn full_signing_flow() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();

        let seed_a = hash(b"alice");
        let seed_b = hash(b"bob");
        let seed_f = hash(b"fee");

        mgr.register(&mix_id, make_input(b"alice", 8), make_output(b"alice", 8), &test_sig(b"alice", &mix_id), None).unwrap();
        mgr.register(&mix_id, make_input(b"bob", 8), make_output(b"bob", 8), &test_sig(b"bob", &mix_id), None).unwrap();
        mgr.set_fee_input(&mix_id, make_input(b"fee", 1), None).unwrap();

        let proposal = mgr.try_finalize(&mix_id).unwrap().unwrap();

        // Sign each input
        for (i, input) in proposal.inputs.iter().enumerate() {
            let pk = input.predicate.owner_pk().unwrap();
            let seed = if pk == wots::keygen(&seed_a) { seed_a }
                else if pk == wots::keygen(&seed_b) { seed_b }
                else { seed_f };
            let sig = wots::sig_to_bytes(&wots::sign(&seed, &proposal.commitment));
            mgr.add_signature(&mix_id, i, sig, 0, None).unwrap();
        }

        let tx = mgr.try_build_transaction(&mix_id).unwrap();
        assert!(tx.is_some());
        match tx.unwrap() {
            Transaction::Reveal { inputs, witnesses, outputs, .. } => {
                assert_eq!(inputs.len(), 3);
                assert_eq!(witnesses.len(), 3);
                assert_eq!(outputs.len(), 2);
            }
            _ => panic!("expected Reveal"),
        }
    }

    #[test]
    fn try_build_returns_none_until_all_sigs() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();

        let seed_a = hash(b"alice");
        let seed_b = hash(b"bob");
        let seed_f = hash(b"fee");

        mgr.register(&mix_id, make_input(b"alice", 8), make_output(b"alice", 8), &test_sig(b"alice", &mix_id), None).unwrap();
        mgr.register(&mix_id, make_input(b"bob", 8), make_output(b"bob", 8), &test_sig(b"bob", &mix_id), None).unwrap();
        mgr.set_fee_input(&mix_id, make_input(b"fee", 1), None).unwrap();

        let proposal = mgr.try_finalize(&mix_id).unwrap().unwrap();

        // Sign only input 0 with a real signature
        let pk = proposal.inputs[0].predicate.owner_pk().unwrap();
        let seed = if pk == wots::keygen(&seed_a) { seed_a }
            else if pk == wots::keygen(&seed_b) { seed_b }
            else { seed_f };
        let sig = wots::sig_to_bytes(&wots::sign(&seed, &proposal.commitment));
        mgr.add_signature(&mix_id, 0, sig, 0, None).unwrap();

        // Still missing sigs for inputs 1 and 2
        assert!(mgr.try_build_transaction(&mix_id).unwrap().is_none());
    }

    #[test]
    fn max_sessions_enforced() {
        let mut mgr = MixManager::new();
        for _ in 0..MAX_MIX_SESSIONS {
            mgr.create_session(8, 2).unwrap();
        }
        assert!(mgr.create_session(8, 2).is_err());
    }

    #[test]
    fn list_sessions() {
        let mut mgr = MixManager::new();
        mgr.create_session(8, 2).unwrap();
        mgr.create_session(16, 2).unwrap();
        let list = mgr.list_sessions();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn cleanup_removes_completed() {
        let mut mgr = MixManager::new();
        let mix_id = mgr.create_session(8, 2).unwrap();
        mgr.set_phase(&mix_id, MixPhase::Complete);
        // Won't be cleaned immediately (30s grace)
        mgr.cleanup();
        assert_eq!(mgr.session_count(), 1);
    }
}
