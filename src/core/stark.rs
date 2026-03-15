//! STARK proof verification for MidstateScript.
//!
//! Provides the backend for `OP_VERIFY_STARK`: a single opcode that verifies
//! arbitrary STARK proofs using Blake3 for all hashing (Merkle commitments,
//! Fiat-Shamir challenges, FRI). Post-quantum secure by construction.
//!
//! # Architecture
//!
//! - **Blake3Hasher**: Implements winterfell's `ElementHasher` trait so the
//!   entire STARK proof infrastructure (Merkle trees, FRI, Fiat-Shamir) runs
//!   on Blake3 — no new cryptographic assumptions.
//!
//! - **Program Registry**: Each STARK program is identified by a 32-byte ID
//!   (the Blake3 hash of its canonical name). Only consensus-agreed programs
//!   are accepted. New programs are added via soft fork.
//!
//! - **Field**: Goldilocks (p = 2^64 - 2^32 + 1). 64-bit native arithmetic,
//!   fast on both x86-64 and aarch64/NEON.

use super::types::hash;
use winter_utils::{ByteReader, ByteWriter, Deserializable, DeserializationError, Serializable};
use winterfell::{
    crypto::{DefaultRandomCoin, Digest, ElementHasher, Hasher, MerkleTree},
    math::{fields::f64::BaseElement, FieldElement, ToElements},
    Air, AirContext, Assertion, EvaluationFrame, FieldExtension, Proof, ProofOptions,
    TraceInfo, TransitionConstraintDegree,
};

use crate::core::confidential;

// ── Consensus Limits ───────────────────────────────────────────────────────

/// Maximum STARK proof size in bytes. Bounds verifier memory and CPU.
/// 131,072 bytes (128 KB) is enough for proofs over traces up to ~2^20 rows.
///
/// ```rust
/// use midstate::core::stark::MAX_STARK_PROOF_SIZE;
/// assert_eq!(MAX_STARK_PROOF_SIZE, 131_072);
/// ```
pub const MAX_STARK_PROOF_SIZE: usize = 131_072;

/// Maximum public input size in bytes.
pub const MAX_STARK_PUBLIC_INPUT_SIZE: usize = 1_024;

// ── Program IDs ────────────────────────────────────────────────────────────

lazy_static::lazy_static! {
    /// Proves: value v ∈ [0, 2^64) and v == claimed_value.
    /// Public inputs: [value as 8 LE bytes]
    pub static ref RANGE_PROOF_64: [u8; 32] = hash(b"midstate.stark.range_proof_64");
    pub static ref CONFIDENTIAL_TRANSFER: [u8; 32] = hash(b"midstate.stark.confidential_transfer");
}

// ── Errors ─────────────────────────────────────────────────────────────────

/// Represents all possible failures during STARK proof verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StarkError {
    UnknownProgram,
    ProofTooLarge,
    PublicInputsTooLarge,
    InvalidPublicInputs(String),
    DeserializationFailed(String),
    VerificationFailed(String),
}

impl std::fmt::Display for StarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownProgram => write!(f, "unknown STARK program ID"),
            Self::ProofTooLarge => write!(f, "STARK proof exceeds {} bytes", MAX_STARK_PROOF_SIZE),
            Self::PublicInputsTooLarge => write!(f, "public inputs exceed {} bytes", MAX_STARK_PUBLIC_INPUT_SIZE),
            Self::InvalidPublicInputs(s) => write!(f, "invalid public inputs: {}", s),
            Self::DeserializationFailed(s) => write!(f, "proof deserialization failed: {}", s),
            Self::VerificationFailed(s) => write!(f, "STARK verification failed: {}", s),
        }
    }
}

impl std::error::Error for StarkError {}

// ═══════════════════════════════════════════════════════════════════════════
// Blake3 Hasher — winterfell integration
// ═══════════════════════════════════════════════════════════════════════════

/// 32-byte Blake3 digest used as the Merkle/Fiat-Shamir hash throughout
/// the STARK proof. Implements all traits winterfell requires of a Digest.
///
/// ```rust
/// use midstate::core::stark::Blake3Digest;
/// let digest = Blake3Digest::new([0xAA; 32]);
/// assert_eq!(digest.as_ref(), &[0xAA; 32]);
/// ```
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct Blake3Digest([u8; 32]);

impl Blake3Digest {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for Blake3Digest {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<[u8; 32]> for Blake3Digest {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl Digest for Blake3Digest {
    fn as_bytes(&self) -> [u8; 32] {
        self.0
    }
}

// Winterfell 0.10 requires explicit serialization logic for its Digests
impl Serializable for Blake3Digest {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        // FIX: The correct method for writing a byte slice in v0.10
        target.write_bytes(&self.0);
    }
}
impl Deserializable for Blake3Digest {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        // FIX: The compiler correctly suggested the generic read_array method
        let bytes = source.read_array()?;
        Ok(Self(bytes))
    }
}

/// Blake3-based hasher for the entire STARK proof infrastructure.
/// Every Merkle commitment, Fiat-Shamir challenge, and FRI fold uses Blake3.
///
/// ```rust
/// use midstate::core::stark::{Blake3Hasher, Blake3Digest};
/// use winterfell::crypto::Hasher;
/// 
/// let digest = Blake3Hasher::hash(b"midstate");
/// assert_ne!(digest.as_ref(), &[0u8; 32]);
/// ```
pub struct Blake3Hasher;

impl Hasher for Blake3Hasher {
    type Digest = Blake3Digest;

    const COLLISION_RESISTANCE: u32 = 128; // 256-bit hash → 128-bit collision resistance

    fn hash(bytes: &[u8]) -> Self::Digest {
        Blake3Digest(*blake3::hash(bytes).as_bytes())
    }

    fn merge(values: &[Self::Digest; 2]) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&values[0].0);
        hasher.update(&values[1].0);
        Blake3Digest(*hasher.finalize().as_bytes())
    }

    fn merge_with_int(seed: Self::Digest, value: u64) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.0);
        hasher.update(&value.to_le_bytes());
        Blake3Digest(*hasher.finalize().as_bytes())
    }

    fn merge_many(values: &[Self::Digest]) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        for v in values {
            hasher.update(&v.0);
        }
        Blake3Digest(*hasher.finalize().as_bytes())
    }
}

impl ElementHasher for Blake3Hasher {
    type BaseField = BaseElement;

    fn hash_elements<E: FieldElement>(elements: &[E]) -> Self::Digest {
        let mut hasher = blake3::Hasher::new();
        let mut bytes = Vec::new();
        // Since E implements FieldElement, the slice naturally implements Serializable
        elements.write_into(&mut bytes);
        hasher.update(&bytes);
        Blake3Digest(*hasher.finalize().as_bytes())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Range Proof AIR — proves v ∈ [0, 2^64)
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "stark-prover")]
const RANGE_PROOF_TRACE_WIDTH: usize = 3;
const RANGE_PROOF_TRACE_LENGTH: usize = 64;

/// Public inputs for the range proof: the value being proven in-range.
#[derive(Clone, Debug)]
pub struct RangeProofInputs {
    pub value: BaseElement,
}

impl ToElements<BaseElement> for RangeProofInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        vec![self.value]
    }
}

/// Algebraic Intermediate Representation (AIR) for a 64-bit range proof.
///
/// # Trace Layout (3 columns × 64 rows):
/// - **Column 0 (`bits`)**: `b_i \in {0, 1}` — the i-th bit of the value.
/// - **Column 1 (`acc`)**: Running sum. `acc_0 = b_0`, `acc_i = acc_{i-1} + b_i \times 2^i`.
/// - **Column 2 (`pow2`)**: Powers of two. `pow2_0 = 1`, `pow2_i = 2^i`.
///
/// # Transition Constraints:
/// 1. `bits_{next} * (1 - bits_{next}) = 0` (Degree 2)
/// 2. `pow2_{next} - pow2_{current} * 2 = 0` (Degree 1)
/// 3. `acc_{next} - acc_{current} - bits_{next} * pow2_{next} = 0` (Degree 2)
pub struct RangeProof64Air {
    context: AirContext<BaseElement>,
    value: BaseElement,
}

impl Air for RangeProof64Air {
    type BaseField = BaseElement;
    type PublicInputs = RangeProofInputs;
    type GkrProof = ();
    type GkrVerifier = ();

fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        let degrees = vec![
            TransitionConstraintDegree::new(2), // bit check: b * (1 - b)
            TransitionConstraintDegree::new(1), // pow2 doubling
            TransitionConstraintDegree::new(2), // accumulator: involves b * pow2
        ];
        
        // FIX: Change '3' to '2' to match the exactly 2 assertions we define in get_assertions()
        let context = AirContext::new(trace_info, degrees, 2, options);
        
        Self {
            context,
            value: pub_inputs.value,
        }
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self,
        frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        result: &mut [E],
    ) {
        let current = frame.current();
        let next = frame.next();

        let bits = 0;
        let acc = 1;
        let pow2 = 2;

        let one = E::ONE;
        let two = E::from(BaseElement::new(2));

        result[0] = next[bits] * (one - next[bits]);
        result[1] = next[pow2] - current[pow2] * two;
        result[2] = next[acc] - current[acc] - next[bits] * next[pow2];
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        vec![
            Assertion::single(2, 0, BaseElement::ONE), // pow2[0] = 1
            Assertion::single(1, RANGE_PROOF_TRACE_LENGTH - 1, self.value), // acc[63] = value
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Prover-side
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "stark-prover")]
pub mod prover {
    use super::*;
    use winterfell::{
        matrix::ColMatrix, AuxRandElements, ConstraintCompositionCoefficients,
        DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, Prover,
        StarkDomain, TracePolyTable, TraceTable,
    };

    /// Generates STARK proofs for the `RangeProof64Air` program.
    pub struct RangeProofProver {
        options: ProofOptions,
        pub_inputs: RangeProofInputs,
    }

    impl RangeProofProver {
        /// Initializes the prover for a specific 64-bit value.
        pub fn new(value: u64) -> Self {
            Self {
                options: ProofOptions::new(32, 8, 0, FieldExtension::None, 8, 31),
                pub_inputs: RangeProofInputs { value: BaseElement::new(value) },
            }
        }

        /// Builds the execution trace proving the value decomposes into 64 bits.
        pub fn build_trace(&self) -> TraceTable<BaseElement> {
            let value = self.pub_inputs.value.as_int();
            let mut trace = TraceTable::new(RANGE_PROOF_TRACE_WIDTH, RANGE_PROOF_TRACE_LENGTH);

            trace.fill(
                |state| {
                    let bit0 = value & 1;
                    state[0] = BaseElement::new(bit0);
                    state[1] = BaseElement::new(bit0);
                    state[2] = BaseElement::ONE;
                },
                |step, state| {
                    let i = step + 1;
                    let bit = (value >> i) & 1;
                    let pow2 = 1u64 << i;

                    state[0] = BaseElement::new(bit);
                    state[2] = BaseElement::new(pow2);
                    state[1] = state[1] + BaseElement::new(bit * pow2);
                },
            );
            trace
        }

        /// Generates the serialized STARK proof.
        pub fn generate_proof(&self) -> Result<(RangeProofInputs, Proof), String> {
            let trace = self.build_trace();
            let proof = Prover::prove(self, trace).map_err(|e| format!("Proving failed: {}", e))?;
            Ok((self.pub_inputs.clone(), proof))
        }
    }

    impl Prover for RangeProofProver {
        type BaseField = BaseElement;
        type Air = RangeProof64Air;
        type Trace = TraceTable<BaseElement>;
        type HashFn = Blake3Hasher;
        type RandomCoin = DefaultRandomCoin<Blake3Hasher>;
        type VC = MerkleTree<Blake3Hasher>;
        
        // FIX: Winterfell 0.10 requires explicit BaseField bounds and a 3rd VC generic
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = 
            DefaultTraceLde<E, Blake3Hasher, Self::VC>;
            
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> =
            DefaultConstraintEvaluator<'a, RangeProof64Air, E>;

        fn get_pub_inputs(&self, _trace: &Self::Trace) -> RangeProofInputs {
            self.pub_inputs.clone()
        }

        fn options(&self) -> &ProofOptions {
            &self.options
        }

        // FIX: Winterfell 0.10 requires explicit trace constructor
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            trace_info: &TraceInfo,
            main_trace: &ColMatrix<Self::BaseField>,
            domain: &StarkDomain<Self::BaseField>,
            partition_option: PartitionOptions,
        ) -> (Self::TraceLde<E>, TracePolyTable<E>) {
            DefaultTraceLde::new(trace_info, main_trace, domain, partition_option)
        }

        // FIX: Winterfell 0.10 requires explicit evaluator constructor
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(
            &self,
            air: &'a Self::Air,
            aux_rand_elements: Option<AuxRandElements<E>>,
            composition_coefficients: ConstraintCompositionCoefficients<E>,
        ) -> Self::ConstraintEvaluator<'a, E> {
            DefaultConstraintEvaluator::new(air, aux_rand_elements, composition_coefficients)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verifier
// ═══════════════════════════════════════════════════════════════════════════

/// Generates the exact `ProofOptions` expected by consensus.
fn acceptable_options() -> winterfell::AcceptableOptions {
    let options = ProofOptions::new(32, 8, 0, FieldExtension::None, 8, 31);
    winterfell::AcceptableOptions::OptionSet(vec![options])
}

/// Internal verifier for the 64-bit range proof.
fn verify_range_proof_64(public_inputs: &[u8], proof_bytes: &[u8]) -> Result<(), StarkError> {
    if public_inputs.len() != 8 {
        return Err(StarkError::InvalidPublicInputs(
            format!("expected 8 bytes, got {}", public_inputs.len())
        ));
    }
    let value = u64::from_le_bytes(public_inputs.try_into().unwrap());
    let pub_inputs = RangeProofInputs {
        value: BaseElement::new(value),
    };

    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| StarkError::DeserializationFailed(format!("{}", e)))?;

    winterfell::verify::<RangeProof64Air, Blake3Hasher, DefaultRandomCoin<Blake3Hasher>, MerkleTree<Blake3Hasher>>(
        proof,
        pub_inputs,
        &acceptable_options(),
    ).map_err(|e| StarkError::VerificationFailed(format!("{}", e)))
}

/// Verify a STARK proof against a registered consensus program.
///
/// Called dynamically by `OP_VERIFY_STARK` in the script VM.
///
/// # Arguments
/// - `program_id`: 32-byte Blake3 hash identifying the AIR program.
/// - `public_inputs`: Program-specific public inputs.
/// - `proof_bytes`: Serialized STARK proof.
///
/// # Errors
/// Returns `StarkError` if the program is unknown, limits are exceeded,
/// or the proof doesn't verify.
///
/// ```rust
/// use midstate::core::stark::{verify_stark_proof, RANGE_PROOF_64, StarkError};
/// 
/// // An empty proof byte array will correctly fail deserialization
/// let res = verify_stark_proof(&RANGE_PROOF_64, &[0u8; 8], &[]);
/// assert!(matches!(res, Err(StarkError::DeserializationFailed(_))));
/// ```
pub fn verify_stark_proof(
    program_id: &[u8; 32],
    public_inputs: &[u8],
    proof_bytes: &[u8],
) -> Result<(), StarkError> {
    if proof_bytes.len() > MAX_STARK_PROOF_SIZE {
        return Err(StarkError::ProofTooLarge);
    }
    if public_inputs.len() > MAX_STARK_PUBLIC_INPUT_SIZE {
        return Err(StarkError::PublicInputsTooLarge);
    }
 
    if program_id == RANGE_PROOF_64.as_ref() {
        verify_range_proof_64(public_inputs, proof_bytes)
    } else if program_id == CONFIDENTIAL_TRANSFER.as_ref() {
        confidential::verify_confidential_transfer(public_inputs, proof_bytes)
    } else {
        Err(StarkError::UnknownProgram)
    }
}
 

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_hasher_deterministic() {
        let a = Blake3Hasher::hash(b"hello");
        let b = Blake3Hasher::hash(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn blake3_hasher_different_inputs_differ() {
        let a = Blake3Hasher::hash(b"hello");
        let b = Blake3Hasher::hash(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn blake3_merge_deterministic() {
        let a = Blake3Hasher::hash(b"left");
        let b = Blake3Hasher::hash(b"right");
        let m1 = Blake3Hasher::merge(&[a, b]);
        let m2 = Blake3Hasher::merge(&[a, b]);
        assert_eq!(m1, m2);
    }

    #[test]
    fn blake3_merge_order_matters() {
        let a = Blake3Hasher::hash(b"left");
        let b = Blake3Hasher::hash(b"right");
        assert_ne!(Blake3Hasher::merge(&[a, b]), Blake3Hasher::merge(&[b, a]));
    }

    #[test]
    fn unknown_program_rejected() {
        let fake_id = [0xFFu8; 32];
        let result = verify_stark_proof(&fake_id, &[], &[]);
        assert_eq!(result, Err(StarkError::UnknownProgram));
    }

    #[test]
    fn oversized_proof_rejected() {
        let big_proof = vec![0u8; MAX_STARK_PROOF_SIZE + 1];
        let result = verify_stark_proof(&*RANGE_PROOF_64, &[0u8; 8], &big_proof);
        assert_eq!(result, Err(StarkError::ProofTooLarge));
    }

    #[test]
    fn range_proof_invalid_public_inputs() {
        let result = verify_range_proof_64(&[1, 2, 3], &[]); // wrong length
        assert!(matches!(result, Err(StarkError::InvalidPublicInputs(_))));
    }

    #[cfg(feature = "stark-prover")]
    #[test]
    fn range_proof_round_trip() {
        let prover = prover::RangeProofProver::new(42);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();

        let proof_bytes = proof.to_bytes();
        let value = pub_inputs.value.as_int();
        
        let result = verify_stark_proof(
            &*RANGE_PROOF_64,
            &value.to_le_bytes(),
            &proof_bytes,
        );
        assert!(result.is_ok());
    }

    #[cfg(feature = "stark-prover")]
    #[test]
    fn range_proof_wrong_value_fails() {
        let prover = prover::RangeProofProver::new(42);
        let (_, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();

        // Claim a different value
        let result = verify_stark_proof(
            &*RANGE_PROOF_64,
            &99u64.to_le_bytes(),
            &proof_bytes,
        );
        assert!(result.is_err());
    }

    #[cfg(feature = "stark-prover")]
    #[test]
    fn range_proof_max_value() {
        let prover = prover::RangeProofProver::new(u64::MAX);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();
        let value = pub_inputs.value.as_int();

        let result = verify_stark_proof(
            &*RANGE_PROOF_64,
            &value.to_le_bytes(),
            &proof_bytes,
        );
        assert!(result.is_ok());
    }
}
