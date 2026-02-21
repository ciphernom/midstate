//! Denomination-uniform CoinJoin mixing.
//!
//! Constructs joint transactions where N participants each contribute one input
//! of the same power-of-2 denomination and each receive one output of that
//! denomination. An additional denomination-1 fee input covers the mandatory
//! tx fee. On-chain, the transaction is a perfect permutation — subset sum
//! analysis yields zero information about the input→output mapping.
//!
//! # Protocol overview
//!
//! 1. Initiator creates a [`MixSession`] for a denomination (e.g. 8).
//! 2. Participants register via [`MixSession::register`] with their input
//!    coin and desired output address.
//! 3. One participant sets a denomination-1 fee input via [`MixSession::set_fee_input`].
//! 4. Once ready, [`MixSession::proposal`] returns a deterministic [`MixProposal`]
//!    containing the canonical input/output ordering and commitment hash.
//! 5. Each participant signs the commitment for their own input.
//! 6. [`MixSession::build_reveal`] assembles the final `Reveal` transaction.
//!
//! No consensus changes required. The resulting `Commit`/`Reveal` pair is
//! indistinguishable from a normal multi-input transaction.
//!
//! ```
//! use midstate::core::types::*;
//! use midstate::core::wots;
//! use midstate::wallet::coinjoin::*;
//!
//! let mut session = MixSession::new(8, 2).unwrap();
//!
//! // Alice: input 8, output 8 to fresh address
//! let seed_a = hash(b"alice");
//! let pk_a = wots::keygen(&seed_a);
//! let input_a = InputReveal { owner_pk: pk_a, value: 8, salt: [0xAA; 32] };
//! let output_a = OutputData { address: hash(b"alice-dest"), value: 8, salt: [0xBB; 32] };
//! session.register(input_a, output_a).unwrap();
//!
//! // Bob: input 8, output 8 to fresh address
//! let seed_b = hash(b"bob");
//! let pk_b = wots::keygen(&seed_b);
//! let input_b = InputReveal { owner_pk: pk_b, value: 8, salt: [0xCC; 32] };
//! let output_b = OutputData { address: hash(b"bob-dest"), value: 8, salt: [0xDD; 32] };
//! session.register(input_b, output_b).unwrap();
//!
//! // Fee coin (denomination 1)
//! let seed_f = hash(b"fee");
//! let pk_f = wots::keygen(&seed_f);
//! let fee_input = InputReveal { owner_pk: pk_f, value: 1, salt: [0xEE; 32] };
//! session.set_fee_input(fee_input).unwrap();
//!
//! let proposal = session.proposal().unwrap();
//! assert_eq!(proposal.inputs.len(), 3);  // 2 mix + 1 fee
//! assert_eq!(proposal.outputs.len(), 2); // 2 mix outputs
//!
//! // Each participant signs the commitment for their input(s)
//! let mut sigs = vec![Vec::new(); proposal.inputs.len()];
//! for (i, input) in proposal.inputs.iter().enumerate() {
//!     let seed = if input.owner_pk == pk_a { &seed_a }
//!         else if input.owner_pk == pk_b { &seed_b }
//!         else { &seed_f };
//!     sigs[i] = wots::sig_to_bytes(&wots::sign(seed, &proposal.commitment));
//! }
//!
//! let reveal = session.build_reveal(sigs).unwrap();
//! match &reveal {
//!     midstate::core::types::Transaction::Reveal { inputs, outputs, .. } => {
//!         assert_eq!(inputs.len(), 3);
//!         assert_eq!(outputs.len(), 2);
//!     }
//!     _ => panic!("expected Reveal"),
//! }
//! ```

use crate::core::types::*;
use anyhow::{bail, Result};

/// Minimum participants in a mix (excluding the fee donor).
pub const MIN_MIX_PARTICIPANTS: usize = 2;

/// Maximum participants in a single mix session.
pub const MAX_MIX_PARTICIPANTS: usize = 16;

/// A single participant's contribution to a mix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixRegistration {
    pub input: InputReveal,
    pub output: OutputData,
}

/// A deterministic, canonical proposal that all participants can
/// independently verify and sign.
#[derive(Clone, Debug)]
pub struct MixProposal {
    /// Canonical inputs: mix inputs sorted by coin_id, fee input last.
    pub inputs: Vec<InputReveal>,
    /// Canonical outputs: sorted by coin_id.
    pub outputs: Vec<OutputData>,
    pub salt: [u8; 32],
    pub commitment: [u8; 32],
}

/// A denomination-uniform mix session.
///
/// Collects registrations, validates denomination uniformity, and produces
/// a canonical transaction proposal. Pure data — no IO or networking.
pub struct MixSession {
    denomination: u64,
    min_participants: usize,
    salt: [u8; 32],
    registrations: Vec<MixRegistration>,
    fee_input: Option<InputReveal>,
}

impl MixSession {
    /// Create a new mix session for the given power-of-2 denomination.
    ///
    /// `min_participants` is clamped to `[MIN_MIX_PARTICIPANTS, MAX_MIX_PARTICIPANTS]`.
    pub fn new(denomination: u64, min_participants: usize) -> Result<Self> {
        if denomination == 0 || !denomination.is_power_of_two() {
            bail!("denomination must be a nonzero power of 2, got {}", denomination);
        }
        let min = min_participants.clamp(MIN_MIX_PARTICIPANTS, MAX_MIX_PARTICIPANTS);
        let salt: [u8; 32] = rand::random();
        Ok(Self {
            denomination,
            min_participants: min,
            salt,
            registrations: Vec::new(),
            fee_input: None,
        })
    }

    /// Create a session with a specific salt (for deterministic testing).
    #[cfg(test)]
    pub fn with_salt(denomination: u64, min_participants: usize, salt: [u8; 32]) -> Result<Self> {
        if denomination == 0 || !denomination.is_power_of_two() {
            bail!("denomination must be a nonzero power of 2, got {}", denomination);
        }
        let min = min_participants.clamp(MIN_MIX_PARTICIPANTS, MAX_MIX_PARTICIPANTS);
        Ok(Self {
            denomination,
            min_participants: min,
            salt,
            registrations: Vec::new(),
            fee_input: None,
        })
    }

    pub fn denomination(&self) -> u64 {
        self.denomination
    }

    pub fn participant_count(&self) -> usize {
        self.registrations.len()
    }

    pub fn has_fee_input(&self) -> bool {
        self.fee_input.is_some()
    }

    /// Register a participant's input and output.
    ///
    /// Both must have value equal to the session denomination.
    /// Rejects duplicate inputs (same coin_id).
    pub fn register(&mut self, input: InputReveal, output: OutputData) -> Result<()> {
        if self.registrations.len() >= MAX_MIX_PARTICIPANTS {
            bail!("session full ({} participants)", MAX_MIX_PARTICIPANTS);
        }
        if input.value != self.denomination {
            bail!(
                "input value {} != session denomination {}",
                input.value, self.denomination
            );
        }
        if output.value != self.denomination {
            bail!(
                "output value {} != session denomination {}",
                output.value, self.denomination
            );
        }
        let coin_id = input.coin_id();
        if self.registrations.iter().any(|r| r.input.coin_id() == coin_id) {
            bail!("duplicate input coin");
        }
        self.registrations.push(MixRegistration { input, output });
        Ok(())
    }

    /// Set the denomination-1 fee input that covers the mandatory tx fee.
    ///
    /// Exactly one fee input per session. Must have value == 1.
    pub fn set_fee_input(&mut self, input: InputReveal) -> Result<()> {
        if input.value != 1 {
            bail!("fee input must be denomination 1, got {}", input.value);
        }
        if self.fee_input.is_some() {
            bail!("fee input already set");
        }
        // Must not collide with any mix input
        let coin_id = input.coin_id();
        if self.registrations.iter().any(|r| r.input.coin_id() == coin_id) {
            bail!("fee input collides with a mix input");
        }
        self.fee_input = Some(input);
        Ok(())
    }

    /// True when enough participants have registered and the fee input is set.
    pub fn is_ready(&self) -> bool {
        self.registrations.len() >= self.min_participants && self.fee_input.is_some()
    }

    /// Build the canonical proposal.
    ///
    /// Inputs are ordered: mix inputs sorted by coin_id, then the fee input.
    /// Outputs are sorted by coin_id. This ordering is deterministic — all
    /// participants independently compute the same commitment.
    pub fn proposal(&self) -> Result<MixProposal> {
        if self.registrations.len() < self.min_participants {
            bail!(
                "need {} participants, have {}",
                self.min_participants,
                self.registrations.len()
            );
        }
        let fee = self.fee_input.as_ref()
            .ok_or_else(|| anyhow::anyhow!("fee input not set"))?;

        // Canonical input order: mix inputs sorted by coin_id, fee last.
        let mut mix_inputs: Vec<InputReveal> = self.registrations
            .iter()
            .map(|r| r.input.clone())
            .collect();
        mix_inputs.sort_by_key(|i| i.coin_id());
        let mut inputs = mix_inputs;
        inputs.push(fee.clone());

        // Canonical output order: sorted by coin_id.
        let mut outputs: Vec<OutputData> = self.registrations
            .iter()
            .map(|r| r.output.clone())
            .collect();
        outputs.sort_by_key(|o| o.coin_id());

        let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let output_coin_ids: Vec<[u8; 32]> = outputs.iter().map(|o| o.coin_id()).collect();
        let commitment = compute_commitment(&input_coin_ids, &output_coin_ids, &self.salt);

        Ok(MixProposal {
            inputs,
            outputs,
            salt: self.salt,
            commitment,
        })
    }

    /// Assemble the final `Reveal` transaction from collected signatures.
    ///
    /// `signatures` must correspond 1:1 with `proposal().inputs` in the same order.
    /// Each participant signs the commitment with their own key; the caller
    /// collects all signatures and passes them here.
    pub fn build_reveal(&self, signatures: Vec<Vec<u8>>) -> Result<Transaction> {
        let proposal = self.proposal()?;
        if signatures.len() != proposal.inputs.len() {
            bail!(
                "expected {} signatures, got {}",
                proposal.inputs.len(),
                signatures.len()
            );
        }
        Ok(Transaction::Reveal {
            inputs: proposal.inputs,
            signatures,
            outputs: proposal.outputs,
            salt: proposal.salt,
        })
    }
}

/// Validate that a `Reveal` transaction has the structure of a denomination-uniform
/// CoinJoin: all mix inputs share one denomination, all outputs share that denomination,
/// and at most one input has denomination 1 (fee).
///
/// This is a heuristic check for observers/analysis; the consensus layer validates
/// the transaction normally regardless.
pub fn is_uniform_mix(tx: &Transaction) -> bool {
    let (inputs, outputs) = match tx {
        Transaction::Reveal { inputs, outputs, .. } => (inputs, outputs),
        _ => return false,
    };

    if inputs.len() < MIN_MIX_PARTICIPANTS + 1 || outputs.len() < MIN_MIX_PARTICIPANTS {
        return false;
    }

    // All outputs must share the same power-of-2 denomination.
    let denom = outputs[0].value;
    if denom == 0 || !denom.is_power_of_two() {
        return false;
    }
    if !outputs.iter().all(|o| o.value == denom) {
        return false;
    }

    // Inputs: exactly outputs.len() inputs at `denom`, plus at most one at 1 (fee).
    let mix_count = inputs.iter().filter(|i| i.value == denom).count();
    let fee_count = inputs.iter().filter(|i| i.value == 1).count();

    mix_count == outputs.len() && fee_count <= 1 && mix_count + fee_count == inputs.len()
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wots;

    fn make_participant(name: &[u8]) -> ([u8; 32], InputReveal, OutputData) {
        let seed = hash(name);
        let pk = wots::keygen(&seed);
        let input = InputReveal {
            owner_pk: pk,
            value: 8,
            salt: hash_concat(name, b"input-salt"),
        };
        let output = OutputData {
            address: hash_concat(name, b"dest"),
            value: 8,
            salt: hash_concat(name, b"output-salt"),
        };
        (seed, input, output)
    }

    fn make_fee_participant(name: &[u8]) -> ([u8; 32], InputReveal) {
        let seed = hash(name);
        let pk = wots::keygen(&seed);
        let input = InputReveal {
            owner_pk: pk,
            value: 1,
            salt: hash_concat(name, b"fee-salt"),
        };
        (seed, input)
    }

    fn ready_session() -> (MixSession, Vec<([u8; 32], InputReveal)>) {
        let mut session = MixSession::with_salt(8, 2, [0x42; 32]).unwrap();

        let (seed_a, in_a, out_a) = make_participant(b"alice");
        let (seed_b, in_b, out_b) = make_participant(b"bob");
        let (seed_f, fee) = make_fee_participant(b"fee-donor");

        session.register(in_a.clone(), out_a).unwrap();
        session.register(in_b.clone(), out_b).unwrap();
        session.set_fee_input(fee.clone()).unwrap();

        let seeds = vec![
            (seed_a, in_a),
            (seed_b, in_b),
            (seed_f, fee),
        ];
        (session, seeds)
    }

    // ── Construction ────────────────────────────────────────────────────

    #[test]
    fn new_rejects_zero_denomination() {
        assert!(MixSession::new(0, 2).is_err());
    }

    #[test]
    fn new_rejects_non_power_of_two() {
        assert!(MixSession::new(3, 2).is_err());
        assert!(MixSession::new(6, 2).is_err());
        assert!(MixSession::new(15, 2).is_err());
    }

    #[test]
    fn new_accepts_valid_denominations() {
        for d in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            assert!(MixSession::new(d, 2).is_ok());
        }
    }

    #[test]
    fn min_participants_clamped() {
        let s = MixSession::new(8, 0).unwrap();
        assert_eq!(s.min_participants, MIN_MIX_PARTICIPANTS);

        let s = MixSession::new(8, 100).unwrap();
        assert_eq!(s.min_participants, MAX_MIX_PARTICIPANTS);
    }

    // ── Registration ────────────────────────────────────────────────────

    #[test]
    fn register_accepts_matching_denomination() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, input, output) = make_participant(b"alice");
        assert!(s.register(input, output).is_ok());
        assert_eq!(s.participant_count(), 1);
    }

    #[test]
    fn register_rejects_wrong_input_denomination() {
        let mut s = MixSession::new(8, 2).unwrap();
        let seed = hash(b"bad");
        let pk = wots::keygen(&seed);
        let input = InputReveal { owner_pk: pk, value: 4, salt: [0; 32] };
        let output = OutputData { address: [0; 32], value: 8, salt: [0; 32] };
        assert!(s.register(input, output).is_err());
    }

    #[test]
    fn register_rejects_wrong_output_denomination() {
        let mut s = MixSession::new(8, 2).unwrap();
        let seed = hash(b"bad");
        let pk = wots::keygen(&seed);
        let input = InputReveal { owner_pk: pk, value: 8, salt: [0; 32] };
        let output = OutputData { address: [0; 32], value: 4, salt: [0; 32] };
        assert!(s.register(input, output).is_err());
    }

    #[test]
    fn register_rejects_duplicate_input() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, input, output) = make_participant(b"alice");
        s.register(input.clone(), output.clone()).unwrap();

        let output2 = OutputData { address: hash(b"other"), value: 8, salt: [0xFF; 32] };
        assert!(s.register(input, output2).is_err());
    }

    #[test]
    fn register_rejects_when_full() {
        let mut s = MixSession::new(8, 2).unwrap();
        for i in 0..MAX_MIX_PARTICIPANTS {
            let (_, input, output) = make_participant(&[i as u8; 4]);
            s.register(input, output).unwrap();
        }
        let (_, input, output) = make_participant(b"overflow");
        assert!(s.register(input, output).is_err());
    }

    // ── Fee input ───────────────────────────────────────────────────────

    #[test]
    fn fee_input_accepts_denomination_1() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, fee) = make_fee_participant(b"fee");
        assert!(s.set_fee_input(fee).is_ok());
        assert!(s.has_fee_input());
    }

    #[test]
    fn fee_input_rejects_wrong_denomination() {
        let mut s = MixSession::new(8, 2).unwrap();
        let seed = hash(b"bad-fee");
        let pk = wots::keygen(&seed);
        let input = InputReveal { owner_pk: pk, value: 2, salt: [0; 32] };
        assert!(s.set_fee_input(input).is_err());
    }

    #[test]
    fn fee_input_rejects_double_set() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, fee1) = make_fee_participant(b"fee1");
        let (_, fee2) = make_fee_participant(b"fee2");
        s.set_fee_input(fee1).unwrap();
        assert!(s.set_fee_input(fee2).is_err());
    }

    #[test]
    fn fee_input_rejects_collision_with_mix_input() {
        let mut s = MixSession::new(1, 2).unwrap();
        // Register a denomination-1 mix input
        let seed = hash(b"collider");
        let pk = wots::keygen(&seed);
        let input = InputReveal { owner_pk: pk, value: 1, salt: [0xAA; 32] };
        let output = OutputData { address: hash(b"dest"), value: 1, salt: [0xBB; 32] };
        s.register(input.clone(), output).unwrap();

        // Same coin as fee should fail
        assert!(s.set_fee_input(input).is_err());
    }

    // ── Readiness ───────────────────────────────────────────────────────

    #[test]
    fn not_ready_without_enough_participants() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, fee) = make_fee_participant(b"fee");
        s.set_fee_input(fee).unwrap();

        let (_, input, output) = make_participant(b"alice");
        s.register(input, output).unwrap();
        assert!(!s.is_ready()); // only 1 of 2
    }

    #[test]
    fn not_ready_without_fee() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, in_a, out_a) = make_participant(b"alice");
        let (_, in_b, out_b) = make_participant(b"bob");
        s.register(in_a, out_a).unwrap();
        s.register(in_b, out_b).unwrap();
        assert!(!s.is_ready());
    }

    #[test]
    fn ready_with_participants_and_fee() {
        let (session, _) = ready_session();
        assert!(session.is_ready());
    }

    // ── Proposal ────────────────────────────────────────────────────────

    #[test]
    fn proposal_fails_without_enough_participants() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, in_a, out_a) = make_participant(b"alice");
        let (_, fee) = make_fee_participant(b"fee");
        s.register(in_a, out_a).unwrap();
        s.set_fee_input(fee).unwrap();
        assert!(s.proposal().is_err());
    }

    #[test]
    fn proposal_fails_without_fee() {
        let mut s = MixSession::new(8, 2).unwrap();
        let (_, in_a, out_a) = make_participant(b"alice");
        let (_, in_b, out_b) = make_participant(b"bob");
        s.register(in_a, out_a).unwrap();
        s.register(in_b, out_b).unwrap();
        assert!(s.proposal().is_err());
    }

    #[test]
    fn proposal_has_correct_counts() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();
        assert_eq!(p.inputs.len(), 3);  // 2 mix + 1 fee
        assert_eq!(p.outputs.len(), 2);
    }

    #[test]
    fn proposal_fee_input_is_last() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();
        assert_eq!(p.inputs.last().unwrap().value, 1);
        // All preceding inputs are the mix denomination
        for input in &p.inputs[..p.inputs.len() - 1] {
            assert_eq!(input.value, 8);
        }
    }

    #[test]
    fn proposal_inputs_sorted_by_coin_id() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();
        // Mix inputs (all but last) should be sorted by coin_id
        let mix_ids: Vec<[u8; 32]> = p.inputs[..p.inputs.len() - 1]
            .iter()
            .map(|i| i.coin_id())
            .collect();
        let mut sorted = mix_ids.clone();
        sorted.sort();
        assert_eq!(mix_ids, sorted);
    }

    #[test]
    fn proposal_outputs_sorted_by_coin_id() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();
        let out_ids: Vec<[u8; 32]> = p.outputs.iter().map(|o| o.coin_id()).collect();
        let mut sorted = out_ids.clone();
        sorted.sort();
        assert_eq!(out_ids, sorted);
    }

    #[test]
    fn proposal_value_conservation() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();
        let in_sum: u64 = p.inputs.iter().map(|i| i.value).sum();
        let out_sum: u64 = p.outputs.iter().map(|o| o.value).sum();
        assert!(in_sum > out_sum);
        assert_eq!(in_sum - out_sum, 1); // fee = 1
    }

    #[test]
    fn proposal_commitment_is_deterministic() {
        let (session, _) = ready_session();
        let p1 = session.proposal().unwrap();
        let p2 = session.proposal().unwrap();
        assert_eq!(p1.commitment, p2.commitment);
        assert_eq!(p1.salt, p2.salt);
    }

    #[test]
    fn proposal_commitment_matches_compute_commitment() {
        let (session, _) = ready_session();
        let p = session.proposal().unwrap();

        let input_ids: Vec<[u8; 32]> = p.inputs.iter().map(|i| i.coin_id()).collect();
        let output_ids: Vec<[u8; 32]> = p.outputs.iter().map(|o| o.coin_id()).collect();
        let expected = compute_commitment(&input_ids, &output_ids, &p.salt);
        assert_eq!(p.commitment, expected);
    }

    // ── build_reveal ────────────────────────────────────────────────────

    #[test]
    fn build_reveal_wrong_sig_count() {
        let (session, _) = ready_session();
        assert!(session.build_reveal(vec![]).is_err());
        assert!(session.build_reveal(vec![vec![]; 2]).is_err()); // need 3
    }

    #[test]
    fn build_reveal_produces_valid_reveal() {
        let (session, seeds) = ready_session();
        let proposal = session.proposal().unwrap();

        let sigs: Vec<Vec<u8>> = proposal.inputs.iter().map(|input| {
            let seed = seeds.iter()
                .find(|(_, i)| i.owner_pk == input.owner_pk)
                .unwrap().0;
            wots::sig_to_bytes(&wots::sign(&seed, &proposal.commitment))
        }).collect();

        let tx = session.build_reveal(sigs).unwrap();
        match &tx {
            Transaction::Reveal { inputs, signatures, outputs, salt } => {
                assert_eq!(inputs.len(), 3);
                assert_eq!(signatures.len(), 3);
                assert_eq!(outputs.len(), 2);
                assert_eq!(*salt, proposal.salt);
            }
            _ => panic!("expected Reveal"),
        }
    }

    #[test]
    fn build_reveal_signatures_verify() {
        let (session, seeds) = ready_session();
        let proposal = session.proposal().unwrap();

        let sigs: Vec<Vec<u8>> = proposal.inputs.iter().map(|input| {
            let seed = seeds.iter()
                .find(|(_, i)| i.owner_pk == input.owner_pk)
                .unwrap().0;
            wots::sig_to_bytes(&wots::sign(&seed, &proposal.commitment))
        }).collect();

        let tx = session.build_reveal(sigs).unwrap();
        if let Transaction::Reveal { inputs, signatures, .. } = &tx {
            for (input, sig_bytes) in inputs.iter().zip(signatures.iter()) {
                let sig = wots::sig_from_bytes(sig_bytes).unwrap();
                assert!(wots::verify(&sig, &proposal.commitment, &input.owner_pk));
            }
        }
    }

    // ── is_uniform_mix ──────────────────────────────────────────────────

    #[test]
    fn is_uniform_mix_true_for_coinjoin() {
        let (session, seeds) = ready_session();
        let proposal = session.proposal().unwrap();
        let sigs: Vec<Vec<u8>> = proposal.inputs.iter().map(|input| {
            let seed = seeds.iter()
                .find(|(_, i)| i.owner_pk == input.owner_pk)
                .unwrap().0;
            wots::sig_to_bytes(&wots::sign(&seed, &proposal.commitment))
        }).collect();
        let tx = session.build_reveal(sigs).unwrap();
        assert!(is_uniform_mix(&tx));
    }

    #[test]
    fn is_uniform_mix_false_for_commit() {
        let tx = Transaction::Commit { commitment: [0; 32], spam_nonce: 0 };
        assert!(!is_uniform_mix(&tx));
    }

    #[test]
    fn is_uniform_mix_false_for_non_uniform_outputs() {
        let tx = Transaction::Reveal {
            inputs: vec![
                InputReveal { owner_pk: [1; 32], value: 8, salt: [0; 32] },
                InputReveal { owner_pk: [2; 32], value: 8, salt: [0; 32] },
                InputReveal { owner_pk: [3; 32], value: 1, salt: [0; 32] },
            ],
            signatures: vec![vec![]; 3],
            outputs: vec![
                OutputData { address: [0; 32], value: 8, salt: [0; 32] },
                OutputData { address: [0; 32], value: 4, salt: [0; 32] }, // mismatch
            ],
            salt: [0; 32],
        };
        assert!(!is_uniform_mix(&tx));
    }

    #[test]
    fn is_uniform_mix_false_for_too_few_participants() {
        // 1 mix input + 1 fee = 2 inputs, 1 output → below MIN_MIX_PARTICIPANTS
        let tx = Transaction::Reveal {
            inputs: vec![
                InputReveal { owner_pk: [1; 32], value: 8, salt: [0; 32] },
                InputReveal { owner_pk: [2; 32], value: 1, salt: [0; 32] },
            ],
            signatures: vec![vec![]; 2],
            outputs: vec![
                OutputData { address: [0; 32], value: 8, salt: [0; 32] },
            ],
            salt: [0; 32],
        };
        assert!(!is_uniform_mix(&tx));
    }

    // ── End-to-end with consensus ───────────────────────────────────────

    #[test]
    fn coinjoin_passes_consensus_validation() {
        use crate::core::transaction::apply_transaction;
        use crate::core::mmr::UtxoAccumulator;

        let mut state = State {
            midstate: [0u8; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height: 1,
            timestamp: 1000,
            commitment_heights: std::collections::HashMap::new(),
        };

        // Create 3 coins in the UTXO set
        let (seed_a, in_a, out_a) = make_participant(b"alice");
        let (seed_b, in_b, out_b) = make_participant(b"bob");
        let (seed_f, fee) = make_fee_participant(b"fee-donor");

        let pk_a = in_a.owner_pk;
        let pk_b = in_b.owner_pk;
        let pk_f = fee.owner_pk;
        let in_a_id = in_a.coin_id();
        let in_b_id = in_b.coin_id();
        let fee_id = fee.coin_id();

        state.coins.insert(in_a_id);
        state.coins.insert(in_b_id);
        state.coins.insert(fee_id);

        // Build the CoinJoin
        let mut session = MixSession::with_salt(8, 2, [0x42; 32]).unwrap();
        session.register(in_a, out_a).unwrap();
        session.register(in_b, out_b).unwrap();
        session.set_fee_input(fee).unwrap();

        let proposal = session.proposal().unwrap();

        // Mine commit PoW
        let mut nonce = 0u64;
        loop {
            let h = hash_concat(&proposal.commitment, &nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 {
                break;
            }
            nonce += 1;
        }

        // Apply Commit
        let commit_tx = Transaction::Commit {
            commitment: proposal.commitment,
            spam_nonce: nonce,
        };
        apply_transaction(&mut state, &commit_tx).unwrap();

        // Sign
        let seeds: Vec<([u8; 32], [u8; 32])> = vec![
            (seed_a, pk_a),
            (seed_b, pk_b),
            (seed_f, pk_f),
        ];
        let sigs: Vec<Vec<u8>> = proposal.inputs.iter().map(|input| {
            let (seed, _) = seeds.iter()
                .find(|(_, pk)| *pk == input.owner_pk)
                .unwrap();
            wots::sig_to_bytes(&wots::sign(seed, &proposal.commitment))
        }).collect();

        // Apply Reveal
        let reveal_tx = session.build_reveal(sigs).unwrap();
        apply_transaction(&mut state, &reveal_tx).unwrap();

        // Input coins spent
        assert!(!state.coins.contains(&in_a_id));
        assert!(!state.coins.contains(&in_b_id));
        assert!(!state.coins.contains(&fee_id));

        // Output coins created
        if let Transaction::Reveal { outputs, .. } = &reveal_tx {
            for o in outputs {
                assert!(state.coins.contains(&o.coin_id()));
            }
        }
    }

    // ── Three-participant session ───────────────────────────────────────

    #[test]
    fn three_participant_mix() {
        let mut session = MixSession::with_salt(16, 3, [0x99; 32]).unwrap();

        for i in 0..3u8 {
            let name = [i; 4];
            let seed = hash(&name);
            let pk = wots::keygen(&seed);
            let input = InputReveal { owner_pk: pk, value: 16, salt: hash(&[i + 100]) };
            let output = OutputData {
                address: hash(&[i + 200]),
                value: 16,
                salt: hash(&[i + 150]),
            };
            session.register(input, output).unwrap();
        }

        let (seed_f, fee) = make_fee_participant(b"fee3");
        session.set_fee_input(fee).unwrap();

        let p = session.proposal().unwrap();
        assert_eq!(p.inputs.len(), 4);  // 3 mix + 1 fee
        assert_eq!(p.outputs.len(), 3);

        let in_sum: u64 = p.inputs.iter().map(|i| i.value).sum();
        let out_sum: u64 = p.outputs.iter().map(|o| o.value).sum();
        assert_eq!(in_sum - out_sum, 1);
    }

    // ── Registration order doesn't affect proposal ──────────────────────

    #[test]
    fn proposal_independent_of_registration_order() {
        let (seed_a, in_a, out_a) = make_participant(b"alice");
        let (seed_b, in_b, out_b) = make_participant(b"bob");
        let (_, fee) = make_fee_participant(b"fee");

        let salt = [0x77; 32];

        // Order 1: alice then bob
        let mut s1 = MixSession::with_salt(8, 2, salt).unwrap();
        s1.register(in_a.clone(), out_a.clone()).unwrap();
        s1.register(in_b.clone(), out_b.clone()).unwrap();
        s1.set_fee_input(fee.clone()).unwrap();

        // Order 2: bob then alice
        let mut s2 = MixSession::with_salt(8, 2, salt).unwrap();
        s2.register(in_b, out_b).unwrap();
        s2.register(in_a, out_a).unwrap();
        s2.set_fee_input(fee).unwrap();

        let p1 = s1.proposal().unwrap();
        let p2 = s2.proposal().unwrap();

        assert_eq!(p1.commitment, p2.commitment);
        assert_eq!(
            p1.inputs.iter().map(|i| i.coin_id()).collect::<Vec<_>>(),
            p2.inputs.iter().map(|i| i.coin_id()).collect::<Vec<_>>(),
        );
        assert_eq!(
            p1.outputs.iter().map(|o| o.coin_id()).collect::<Vec<_>>(),
            p2.outputs.iter().map(|o| o.coin_id()).collect::<Vec<_>>(),
        );
    }

    // ── Denomination-1 mixing ───────────────────────────────────────────

    #[test]
    fn denomination_1_mix_needs_denomination_1_fee() {
        // Edge case: mixing denomination 1 coins. The fee input is *also*
        // denomination 1, so it looks like another mix participant on chain.
        // That's acceptable — the privacy set is N+1 instead of N.
        let mut session = MixSession::with_salt(1, 2, [0; 32]).unwrap();

        for i in 0..2u8 {
            let seed = hash(&[i]);
            let pk = wots::keygen(&seed);
            let input = InputReveal { owner_pk: pk, value: 1, salt: [i + 10; 32] };
            let output = OutputData { address: hash(&[i + 20]), value: 1, salt: [i + 30; 32] };
            session.register(input, output).unwrap();
        }

        let (_, fee) = make_fee_participant(b"fee-d1");
        session.set_fee_input(fee).unwrap();

        let p = session.proposal().unwrap();
        // 3 inputs of denom 1, 2 outputs of denom 1
        assert_eq!(p.inputs.iter().filter(|i| i.value == 1).count(), 3);
        assert_eq!(p.outputs.len(), 2);

        let in_sum: u64 = p.inputs.iter().map(|i| i.value).sum();
        let out_sum: u64 = p.outputs.iter().map(|o| o.value).sum();
        assert_eq!(in_sum - out_sum, 1);
    }

    // ── Accessor coverage ───────────────────────────────────────────────

    #[test]
    fn accessors() {
        let s = MixSession::new(16, 3).unwrap();
        assert_eq!(s.denomination(), 16);
        assert_eq!(s.participant_count(), 0);
        assert!(!s.has_fee_input());
        assert!(!s.is_ready());
    }
}
