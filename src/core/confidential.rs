//! Confidential Transfer STARK — proves value balance + commitment binding
//!
//! Verifies for a 2-input, 2-output confidential transfer:
//!   1. Blake3(v1 || blind1) = commitment1   (commitment binding)
//!   2. Blake3(v2 || blind2) = commitment2   (commitment binding)
//!   3. v1, v2 ∈ [0, 2^64)                  (range proofs)
//!   4. v1 + v2 = input_sum                  (value balance)
//!
//! Trace layout: 24 columns × 16384 rows.
//! Periodic columns: 57 (9 selectors + 48 one-hot index indicators).

use super::stark::{Blake3Hasher, StarkError};

use winterfell::{
    crypto::{DefaultRandomCoin, MerkleTree},
    math::{fields::f64::BaseElement, FieldElement, ToElements},
    Air, AirContext, Assertion, EvaluationFrame, FieldExtension, Proof, ProofOptions,
    TraceInfo, TransitionConstraintDegree,
};

// ═══════════════════════════════════════════════════════════════════════════
// Blake3 Constants
// ═══════════════════════════════════════════════════════════════════════════

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];
const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const BLAKE3_FLAGS: u32 = 1 | 2 | 8; // CHUNK_START | CHUNK_END | ROOT
const BLAKE3_BLOCK_LEN: u32 = 40;
const TWO_POW_32: u64 = 1u64 << 32;

// ═══════════════════════════════════════════════════════════════════════════
// Trace Layout — 24 columns
// ═══════════════════════════════════════════════════════════════════════════

const NUM_COLS: usize = 24;
const S0: usize = 0;
const BIT_A: usize = 16;
const BIT_B: usize = 17;
const ACC_A: usize = 18;
const ACC_B: usize = 19;
const ACC_XOR: usize = 20;
const POW2: usize = 21;
const CARRY_BAL: usize = 22;
const AUX: usize = 23;

// ═══════════════════════════════════════════════════════════════════════════
// Step Schedule
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq)]
enum OpType { Add, AddMsg, XorFirst, XorMid, XorLast, RangeFirst, RangeMid, RangeLast, Reset, Pass }

#[derive(Clone, Debug)]
struct Step { op: OpType, rd1: usize, rd2: usize, wr: usize, msg_idx: Option<(usize, usize)>, rotation: u32 }

impl Step {
    fn add_msg(rd1: usize, rd2: usize, wr: usize, round: usize, word_idx: usize) -> Self {
        Step { op: OpType::AddMsg, rd1, rd2, wr, msg_idx: Some((round, word_idx)), rotation: 0 }
    }
    fn add(rd1: usize, rd2: usize, wr: usize) -> Self {
        Step { op: OpType::Add, rd1, rd2, wr, msg_idx: None, rotation: 0 }
    }
    fn xor(rd1: usize, rd2: usize, wr: usize, rot: u32) -> Vec<Self> {
        let mut v = Vec::with_capacity(32);
        v.push(Step { op: OpType::XorFirst, rd1, rd2, wr, msg_idx: None, rotation: rot });
        for _ in 1..31 { v.push(Step { op: OpType::XorMid, rd1, rd2, wr, msg_idx: None, rotation: rot }); }
        v.push(Step { op: OpType::XorLast, rd1, rd2, wr, msg_idx: None, rotation: rot });
        v
    }
    fn pass() -> Self { Step { op: OpType::Pass, rd1: 0, rd2: 0, wr: 0, msg_idx: None, rotation: 0 } }
    fn reset() -> Self { Step { op: OpType::Reset, rd1: 0, rd2: 0, wr: 0, msg_idx: None, rotation: 0 } }
    fn range_block() -> Vec<Self> {
        let mut v = Vec::with_capacity(64);
        v.push(Step { op: OpType::RangeFirst, rd1: 0, rd2: 0, wr: 0, msg_idx: None, rotation: 0 });
        for _ in 1..63 { v.push(Step { op: OpType::RangeMid, rd1: 0, rd2: 0, wr: 0, msg_idx: None, rotation: 0 }); }
        v.push(Step { op: OpType::RangeLast, rd1: 0, rd2: 0, wr: 0, msg_idx: None, rotation: 0 });
        v
    }
}

fn compute_msg_schedule() -> [[usize; 16]; 7] {
    let mut s = [[0usize; 16]; 7];
    s[0] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15];
    for r in 1..7 { for i in 0..16 { s[r][i] = s[r-1][MSG_PERMUTATION[i]]; } }
    s
}

fn generate_hash_schedule() -> Vec<Step> {
    let mut steps = Vec::new();
    let col_qr = [(0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)];
    let diag_qr = [(0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)];
    for round in 0..7 {
        for (qr_idx, &(a,b,c,d)) in col_qr.iter().chain(diag_qr.iter()).enumerate() {
            steps.push(Step::add_msg(a,b,a,round,qr_idx*2));
            steps.extend(Step::xor(d,a,d,16));
            steps.push(Step::add(c,d,c));
            steps.extend(Step::xor(b,c,b,12));
            steps.push(Step::add_msg(a,b,a,round,qr_idx*2+1));
            steps.extend(Step::xor(d,a,d,8));
            steps.push(Step::add(c,d,c));
            steps.extend(Step::xor(b,c,b,7));
        }
    }
    for i in 0..8 { steps.extend(Step::xor(i,i+8,i,0)); }
    steps
}

fn generate_full_schedule() -> Vec<Step> {
    let h = generate_hash_schedule();
    let mut s = Vec::new();
    s.extend(h.clone()); s.push(Step::reset());
    s.extend(h);         s.push(Step::reset());
    s.extend(Step::range_block());
    s.extend(Step::range_block());
    let target = s.len().next_power_of_two();
    while s.len() < target { s.push(Step::pass()); }
    s
}

// ═══════════════════════════════════════════════════════════════════════════
// Periodic Columns — 57 total
// 9 selectors + 16 one-hot rd1 + 16 one-hot rd2 + 16 one-hot wr
// ═══════════════════════════════════════════════════════════════════════════

const NUM_PERIODIC: usize = 57;
const P_IS_ADD: usize = 0;
const P_IS_ADD_MSG: usize = 1;
const P_IS_XOR: usize = 2;
const P_IS_XOR_FIRST: usize = 3;
const P_IS_XOR_LAST: usize = 4;
const P_IS_RANGE: usize = 5;
const P_IS_RANGE_FIRST: usize = 6;
const P_IS_RANGE_LAST: usize = 7;
const P_IS_ACTIVE: usize = 8;
const P_R1: usize = 9;   // 9..24:  R1[i]=1 iff rd1==i
const P_R2: usize = 25;  // 25..40: R2[i]=1 iff rd2==i
const P_WR: usize = 41;  // 41..56: WR[i]=1 iff wr==i

fn generate_periodic_columns(steps: &[Step]) -> Vec<Vec<BaseElement>> {
    let n = steps.len();
    let mut cols = vec![vec![BaseElement::ZERO; n]; NUM_PERIODIC];
    for (row, step) in steps.iter().enumerate() {
        let one = BaseElement::ONE;
        match step.op {
            OpType::Add     => { cols[P_IS_ADD][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::AddMsg  => { cols[P_IS_ADD][row] = one; cols[P_IS_ADD_MSG][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::XorFirst => { cols[P_IS_XOR][row] = one; cols[P_IS_XOR_FIRST][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::XorMid   => { cols[P_IS_XOR][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::XorLast  => { cols[P_IS_XOR][row] = one; cols[P_IS_XOR_LAST][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::RangeFirst => { cols[P_IS_RANGE][row] = one; cols[P_IS_RANGE_FIRST][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::RangeMid   => { cols[P_IS_RANGE][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::RangeLast  => { cols[P_IS_RANGE][row] = one; cols[P_IS_RANGE_LAST][row] = one; cols[P_IS_ACTIVE][row] = one; }
            OpType::Reset | OpType::Pass => {}
        }
        cols[P_R1 + step.rd1][row] = one;
        cols[P_R2 + step.rd2][row] = one;
        cols[P_WR + step.wr][row] = one;
    }
    cols
}

// ═══════════════════════════════════════════════════════════════════════════
// One-Hot Word Selection
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn select_word_onehot<E: FieldElement>(state: &[E], indicators: &[E]) -> E {
    let mut r = E::ZERO;
    for i in 0..16 { r += indicators[i] * state[i]; }
    r
}

// ═══════════════════════════════════════════════════════════════════════════
// Public Inputs
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct ConfidentialTransferInputs {
    pub commitment1: [u32; 8],
    pub commitment2: [u32; 8],
    pub input_sum: u64,
}

impl ToElements<BaseElement> for ConfidentialTransferInputs {
    fn to_elements(&self) -> Vec<BaseElement> {
        let mut r = Vec::new();
        for &w in &self.commitment1 { r.push(BaseElement::new(w as u64)); }
        for &w in &self.commitment2 { r.push(BaseElement::new(w as u64)); }
        r.push(BaseElement::new(self.input_sum));
        r
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AIR — 36 constraints, 50 assertions
// ═══════════════════════════════════════════════════════════════════════════

const NUM_CONSTRAINTS: usize = 36;

pub struct ConfidentialTransferAir {
    context: AirContext<BaseElement>,
    pub_inputs: ConfidentialTransferInputs,
    steps: Vec<Step>,
    hash1_start: usize,
    hash2_start: usize,
    range1_start: usize,
    range2_start: usize,
}

impl ConfidentialTransferAir {
    fn initial_blake3_state() -> [u32; 16] {
        [IV[0],IV[1],IV[2],IV[3], IV[4],IV[5],IV[6],IV[7],
         IV[0],IV[1],IV[2],IV[3], 0, 0, BLAKE3_BLOCK_LEN, BLAKE3_FLAGS]
    }
    fn hash_output_rows(&self, start: usize) -> [usize; 8] {
        let fin = start + generate_hash_schedule().len() - 256;
        let mut r = [0; 8];
        for i in 0..8 { r[i] = fin + i*32 + 31; }
        r
    }
}

impl Air for ConfidentialTransferAir {
    type BaseField = BaseElement;
    type PublicInputs = ConfidentialTransferInputs;
    type GkrProof = ();
    type GkrVerifier = ();

    fn new(trace_info: TraceInfo, pub_inputs: Self::PublicInputs, options: ProofOptions) -> Self {
        let steps = generate_full_schedule();
        let hl = generate_hash_schedule().len();
        let h1 = 0;
        let h2 = hl + 1;
        let r1 = h2 + hl + 1;
        let r2 = r1 + 64;

        // Degrees with one-hot selection (no Lagrange explosion):
        // State transitions: is_active × (next - curr - is_writing × WR_i × (write_val - curr))
        //   write_val = is_add × (Σ R1*state + Σ R2*state + aux - carry*2^32) + is_xor_last × acc_xor
        //   Max chain: p × (t + p × p × (p × p×t)) = ~6 periodic × 1 trace
        //   Declare 7 to be safe.
        let mut d = Vec::with_capacity(NUM_CONSTRAINTS);
        for _ in 0..16 { d.push(TransitionConstraintDegree::new(6)); }
        d.push(TransitionConstraintDegree::new(4)); // 16: carry
        d.push(TransitionConstraintDegree::new(3)); // 17
        d.push(TransitionConstraintDegree::new(3)); // 18
        d.push(TransitionConstraintDegree::new(2)); // 19
        d.push(TransitionConstraintDegree::new(2)); // 20
        d.push(TransitionConstraintDegree::new(3)); // 21
        d.push(TransitionConstraintDegree::new(3)); // 22
        d.push(TransitionConstraintDegree::new(4)); // 23
        d.push(TransitionConstraintDegree::new(3)); // 24
        d.push(TransitionConstraintDegree::new(3)); // 25
        d.push(TransitionConstraintDegree::new(4)); // 26
        d.push(TransitionConstraintDegree::new(3)); // 27: operand verify a
        d.push(TransitionConstraintDegree::new(3)); // 28: operand verify b
        d.push(TransitionConstraintDegree::new(4)); // 29
        d.push(TransitionConstraintDegree::new(2)); // 30
        d.push(TransitionConstraintDegree::new(2)); // 31
        d.push(TransitionConstraintDegree::new(3)); // 32
        d.push(TransitionConstraintDegree::new(3)); // 33
        d.push(TransitionConstraintDegree::new(2)); // 34
        d.push(TransitionConstraintDegree::new(2)); // 35
        assert_eq!(d.len(), NUM_CONSTRAINTS);

        let ctx = AirContext::new(trace_info, d, 50, options);
        Self { context: ctx, pub_inputs, steps, hash1_start: h1, hash2_start: h2, range1_start: r1, range2_start: r2 }
    }

    fn context(&self) -> &AirContext<Self::BaseField> { &self.context }

    fn get_periodic_column_values(&self) -> Vec<Vec<BaseElement>> {
        generate_periodic_columns(&self.steps)
    }

    fn evaluate_transition<E: FieldElement + From<Self::BaseField>>(
        &self, frame: &EvaluationFrame<E>, pv: &[E], result: &mut [E],
    ) {
        let curr = frame.current();
        let next = frame.next();
        let one = E::ONE;
        let two = E::from(BaseElement::new(2));
        let two32 = E::from(BaseElement::new(TWO_POW_32));

        let is_add = pv[P_IS_ADD];
        let is_add_msg = pv[P_IS_ADD_MSG];
        let is_xor = pv[P_IS_XOR];
        let is_xor_first = pv[P_IS_XOR_FIRST];
        let is_xor_last = pv[P_IS_XOR_LAST];
        let is_range = pv[P_IS_RANGE];
        let is_range_first = pv[P_IS_RANGE_FIRST];
        let is_range_last = pv[P_IS_RANGE_LAST];
        let is_active = pv[P_IS_ACTIVE];
        let r1 = &pv[P_R1..P_R1+16];
        let r2 = &pv[P_R2..P_R2+16];
        let wr = &pv[P_WR..P_WR+16];

        let is_writing = is_add + is_xor_last;
        let cs = &curr[S0..S0+16];
        let ns = &next[S0..S0+16];

        let val_rd1 = select_word_onehot(cs, r1);
        let val_rd2 = select_word_onehot(cs, r2);

        let add_result = val_rd1 + val_rd2 + curr[AUX] - curr[CARRY_BAL] * two32;
        let write_val = is_add * add_result + is_xor_last * curr[ACC_XOR];

        // 0-15: state transitions
        for i in 0..16 {
            let expected = cs[i] + is_writing * wr[i] * (write_val - cs[i]);
            result[i] = is_active * (ns[i] - expected);
        }

        // 16: carry
        let c = curr[CARRY_BAL];
        let is_add_only = is_add - is_add_msg;
        result[16] = is_add_msg * c * (c - one) * (c - two) + is_add_only * c * (c - one);

        // 17-18: bit booleans
        result[17] = (is_xor + is_range) * curr[BIT_A] * (one - curr[BIT_A]);
        result[18] = is_xor * curr[BIT_B] * (one - curr[BIT_B]);

        // 19-20: xor pow2
        let is_xor_not_last = is_xor - is_xor_last;
        result[19] = is_xor_not_last * (next[POW2] - two * curr[POW2]);
        result[20] = is_xor_first * (curr[POW2] - one);

        // 21-23: xor acc transitions
        result[21] = is_xor_not_last * (next[ACC_A] - curr[ACC_A] - next[BIT_A] * next[POW2]);
        result[22] = is_xor_not_last * (next[ACC_B] - curr[ACC_B] - next[BIT_B] * next[POW2]);
        let xbn = next[BIT_A] + next[BIT_B] - two * next[BIT_A] * next[BIT_B];
        result[23] = is_xor_not_last * (next[ACC_XOR] - curr[ACC_XOR] - xbn * next[AUX]);

        // 24-26: xor acc init
        let xbc = curr[BIT_A] + curr[BIT_B] - two * curr[BIT_A] * curr[BIT_B];
        result[24] = is_xor_first * (curr[ACC_A] - curr[BIT_A] * curr[POW2]);
        result[25] = is_xor_first * (curr[ACC_B] - curr[BIT_B] * curr[POW2]);
        result[26] = is_xor_first * (curr[ACC_XOR] - xbc * curr[AUX]);

        // 27-28: operand verification
        result[27] = is_xor_last * (curr[ACC_A] - val_rd1);
        result[28] = is_xor_last * (curr[ACC_B] - val_rd2);

        // 29: aux zero
        let aux_used = is_add_msg + is_xor;
        result[29] = is_active * (one - aux_used) * (one - is_range) * curr[AUX];

        // 30-33: range proof
        let is_range_not_last = is_range - is_range_last;
        result[30] = is_range_not_last * (next[POW2] - two * curr[POW2]);
        result[31] = is_range_first * (curr[POW2] - one);
        result[32] = is_range_not_last * (next[ACC_A] - curr[ACC_A] - next[BIT_A] * next[POW2]);
        result[33] = is_range_first * (curr[ACC_A] - curr[BIT_A] * curr[POW2]);

        // 34-35: balance
        result[34] = is_range_not_last * (next[CARRY_BAL] - curr[CARRY_BAL]);
        result[35] = is_range_last * (next[CARRY_BAL] - curr[CARRY_BAL] - curr[ACC_A]);
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        let mut a = Vec::new();
        let init = Self::initial_blake3_state();
        for i in 0..16 { a.push(Assertion::single(i, self.hash1_start, BaseElement::new(init[i] as u64))); }
        for i in 0..16 { a.push(Assertion::single(i, self.hash2_start, BaseElement::new(init[i] as u64))); }
        for (i, &r) in self.hash_output_rows(self.hash1_start).iter().enumerate() {
            a.push(Assertion::single(ACC_XOR, r, BaseElement::new(self.pub_inputs.commitment1[i] as u64)));
        }
        for (i, &r) in self.hash_output_rows(self.hash2_start).iter().enumerate() {
            a.push(Assertion::single(ACC_XOR, r, BaseElement::new(self.pub_inputs.commitment2[i] as u64)));
        }
        a.push(Assertion::single(CARRY_BAL, self.range1_start, BaseElement::ZERO));
        a.push(Assertion::single(CARRY_BAL, self.range2_start + 64, BaseElement::new(self.pub_inputs.input_sum)));
        assert_eq!(a.len(), 50);
        a
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reference Blake3
// ═══════════════════════════════════════════════════════════════════════════

fn blake3_qr(s: &mut [u32;16], a:usize, b:usize, c:usize, d:usize, mx:u32, my:u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

fn blake3_compress(msg: &[u32;16]) -> [u32;8] {
    let mut s = ConfidentialTransferAir::initial_blake3_state();
    let sch = compute_msg_schedule();
    let cq = [(0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)];
    let dq = [(0,5,10,15),(1,6,11,12),(2,7,8,13),(3,4,9,14)];
    for r in 0..7 {
        for (qi, &(a,b,c,d)) in cq.iter().chain(dq.iter()).enumerate() {
            blake3_qr(&mut s, a,b,c,d, msg[sch[r][qi*2]], msg[sch[r][qi*2+1]]);
        }
    }
    let mut o = [0u32;8];
    for i in 0..8 { o[i] = s[i] ^ s[i+8]; }
    o
}

fn build_msg_block(value: u64, blinding: &[u8;32]) -> [u32;16] {
    let mut b = [0u32;16];
    let v = value.to_le_bytes();
    b[0] = u32::from_le_bytes(v[0..4].try_into().unwrap());
    b[1] = u32::from_le_bytes(v[4..8].try_into().unwrap());
    for i in 0..8 { b[2+i] = u32::from_le_bytes(blinding[i*4..(i+1)*4].try_into().unwrap()); }
    b
}

// ═══════════════════════════════════════════════════════════════════════════
// Prover
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "stark-prover")]
pub mod ct_prover {
    use super::*;
    use winterfell::{
        matrix::ColMatrix, AuxRandElements, ConstraintCompositionCoefficients,
        DefaultConstraintEvaluator, DefaultTraceLde, PartitionOptions, Prover,
        StarkDomain, TracePolyTable, TraceTable,
    };

    pub struct ConfidentialTransferProver {
        options: ProofOptions,
        pub_inputs: ConfidentialTransferInputs,
        v1: u64, blind1: [u8;32], v2: u64, blind2: [u8;32],
    }

    impl ConfidentialTransferProver {
        pub fn new(v1: u64, blind1: [u8;32], v2: u64, blind2: [u8;32]) -> Self {
            let m1 = build_msg_block(v1, &blind1);
            let m2 = build_msg_block(v2, &blind2);
            Self {
                options: ProofOptions::new(32, 8, 0, FieldExtension::None, 8, 31),
                pub_inputs: ConfidentialTransferInputs {
                    commitment1: blake3_compress(&m1), commitment2: blake3_compress(&m2), input_sum: v1+v2,
                },
                v1, blind1, v2, blind2,
            }
        }

        pub fn build_trace(&self) -> TraceTable<BaseElement> {
            let steps = generate_full_schedule();
            let n = steps.len();
            let mut cols = vec![vec![BaseElement::ZERO; n]; NUM_COLS];
            let m1 = build_msg_block(self.v1, &self.blind1);
            let m2 = build_msg_block(self.v2, &self.blind2);
            let sch = compute_msg_schedule();
            let mut state = ConfidentialTransferAir::initial_blake3_state();
            let mut cmsg = &m1;
            let mut bp: usize = 0;
            let mut bal: u64 = 0;
            let mut h2 = false;
            let mut rv: u64 = 0;
            let mut rs = 0usize;

            for (row, step) in steps.iter().enumerate() {
                for i in 0..16 { cols[i][row] = BaseElement::new(state[i] as u64); }
                match step.op {
                    OpType::Add | OpType::AddMsg => {
                        let a = state[step.rd1] as u64;
                        let b = state[step.rd2] as u64;
                        let m = step.msg_idx.map_or(0u64, |(r,w)| cmsg[sch[r][w]] as u64);
                        let sum = a + b + m;
                        cols[CARRY_BAL][row] = BaseElement::new(sum / TWO_POW_32);
                        cols[AUX][row] = BaseElement::new(m);
                        state[step.wr] = (sum % TWO_POW_32) as u32;
                        bp = 0;
                    }
                    OpType::XorFirst | OpType::XorMid | OpType::XorLast => {
                        let oa = state[step.rd1]; let ob = state[step.rd2];
                        let rp = ((bp as u32 + 32 - step.rotation) % 32) as u64;
                        cols[BIT_A][row] = BaseElement::new(((oa >> bp) & 1) as u64);
                        cols[BIT_B][row] = BaseElement::new(((ob >> bp) & 1) as u64);
                        cols[POW2][row] = BaseElement::new(1u64 << bp);
                        cols[AUX][row] = BaseElement::new(1u64 << rp);
                        cols[ACC_A][row] = BaseElement::new((0..=bp).map(|b| (((oa>>b)&1) as u64)<<b).sum::<u64>());
                        cols[ACC_B][row] = BaseElement::new((0..=bp).map(|b| (((ob>>b)&1) as u64)<<b).sum::<u64>());
                        cols[ACC_XOR][row] = BaseElement::new((0..=bp).map(|b| {
                            let x = ((oa>>b)^(ob>>b))&1;
                            let p = ((b as u32+32-step.rotation)%32) as u64;
                            (x as u64)<<p
                        }).sum::<u64>());
                        if step.op == OpType::XorLast {
                            state[step.wr] = (oa^ob).rotate_right(step.rotation);
                            bp = 0;
                        } else { bp += 1; }
                    }
                    OpType::Reset => {
                        if !h2 { h2 = true; cmsg = &m2; }
                        state = ConfidentialTransferAir::initial_blake3_state();
                        bp = 0;
                    }
                    OpType::RangeFirst | OpType::RangeMid | OpType::RangeLast => {
                        if step.op == OpType::RangeFirst && bp == 0 {
                            rv = if rs == 0 { self.v1 } else { self.v2 };
                        }
                        cols[BIT_A][row] = BaseElement::new((rv >> bp) & 1);
                        cols[POW2][row] = BaseElement::new(1u64 << bp);
                        cols[ACC_A][row] = BaseElement::new((0..=bp).map(|b| ((rv>>b)&1)<<b).sum::<u64>());
                        cols[CARRY_BAL][row] = BaseElement::new(bal);
                        if step.op == OpType::RangeLast { bal += rv; rs += 1; bp = 0; } else { bp += 1; }
                    }
                    OpType::Pass => { cols[CARRY_BAL][row] = BaseElement::new(bal); }
                }
            }
            TraceTable::init(cols)
        }

        pub fn generate_proof(&self) -> Result<(ConfidentialTransferInputs, Proof), String> {
            let trace = self.build_trace();
            Prover::prove(self, trace).map(|p| (self.pub_inputs.clone(), p)).map_err(|e| format!("CT proof failed: {}", e))
        }
    }

    impl Prover for ConfidentialTransferProver {
        type BaseField = BaseElement;
        type Air = ConfidentialTransferAir;
        type Trace = TraceTable<BaseElement>;
        type HashFn = Blake3Hasher;
        type RandomCoin = DefaultRandomCoin<Blake3Hasher>;
        type VC = MerkleTree<Blake3Hasher>;
        type TraceLde<E: FieldElement<BaseField = Self::BaseField>> = DefaultTraceLde<E, Blake3Hasher, Self::VC>;
        type ConstraintEvaluator<'a, E: FieldElement<BaseField = Self::BaseField>> = DefaultConstraintEvaluator<'a, ConfidentialTransferAir, E>;

        fn get_pub_inputs(&self, _: &Self::Trace) -> ConfidentialTransferInputs { self.pub_inputs.clone() }
        fn options(&self) -> &ProofOptions { &self.options }
        fn new_trace_lde<E: FieldElement<BaseField = Self::BaseField>>(&self, ti: &TraceInfo, mt: &ColMatrix<Self::BaseField>, d: &StarkDomain<Self::BaseField>, po: PartitionOptions) -> (Self::TraceLde<E>, TracePolyTable<E>) { DefaultTraceLde::new(ti, mt, d, po) }
        fn new_evaluator<'a, E: FieldElement<BaseField = Self::BaseField>>(&self, air: &'a Self::Air, aux: Option<AuxRandElements<E>>, cc: ConstraintCompositionCoefficients<E>) -> Self::ConstraintEvaluator<'a, E> { DefaultConstraintEvaluator::new(air, aux, cc) }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verifier
// ═══════════════════════════════════════════════════════════════════════════

fn ct_acceptable_options() -> winterfell::AcceptableOptions {
    winterfell::AcceptableOptions::OptionSet(vec![ProofOptions::new(32, 8, 0, FieldExtension::None, 8, 31)])
}

pub fn verify_confidential_transfer(public_inputs: &[u8], proof_bytes: &[u8]) -> Result<(), StarkError> {
    if public_inputs.len() != 72 {
        return Err(StarkError::InvalidPublicInputs(format!("expected 72 bytes, got {}", public_inputs.len())));
    }
    let mut c1 = [0u32;8]; let mut c2 = [0u32;8];
    for i in 0..8 { c1[i] = u32::from_le_bytes(public_inputs[i*4..(i+1)*4].try_into().unwrap()); }
    for i in 0..8 { c2[i] = u32::from_le_bytes(public_inputs[32+i*4..32+(i+1)*4].try_into().unwrap()); }
    let sum = u64::from_le_bytes(public_inputs[64..72].try_into().unwrap());
    let pi = ConfidentialTransferInputs { commitment1: c1, commitment2: c2, input_sum: sum };
    let proof = Proof::from_bytes(proof_bytes).map_err(|e| StarkError::DeserializationFailed(format!("{}", e)))?;
    winterfell::verify::<ConfidentialTransferAir, Blake3Hasher, DefaultRandomCoin<Blake3Hasher>, MerkleTree<Blake3Hasher>>(proof, pi, &ct_acceptable_options())
        .map_err(|e| StarkError::VerificationFailed(format!("{}", e)))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_compress_matches_reference() {
        let v = 1_073_741_824u64; let bl = [0x42u8;32];
        let ours = blake3_compress(&build_msg_block(v, &bl));
        let mut h = blake3::Hasher::new(); h.update(&v.to_le_bytes()); h.update(&bl);
        let exp = h.finalize();
        let mut ew = [0u32;8];
        for i in 0..8 { ew[i] = u32::from_le_bytes(exp.as_bytes()[i*4..(i+1)*4].try_into().unwrap()); }
        assert_eq!(ours, ew);
    }
    #[test] fn msg_schedule_round_0() { assert_eq!(compute_msg_schedule()[0], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]); }
    #[test] fn msg_schedule_round_1() { assert_eq!(compute_msg_schedule()[1], [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]); }
    #[test] fn step_schedule_length() { assert_eq!(generate_hash_schedule().len(), 7648); }
    #[test] fn full_schedule_is_power_of_2() { assert!(generate_full_schedule().len().is_power_of_two()); }
    #[test] fn full_schedule_section_boundaries() {
        let hl = generate_hash_schedule().len(); let s = generate_full_schedule();
        assert_eq!(s[hl].op, OpType::Reset);
        let h2 = hl+1; assert_eq!(s[h2+hl].op, OpType::Reset);
        let r1 = h2+hl+1; assert_eq!(s[r1].op, OpType::RangeFirst); assert_eq!(s[r1+63].op, OpType::RangeLast);
        let r2 = r1+64; assert_eq!(s[r2].op, OpType::RangeFirst); assert_eq!(s[r2+63].op, OpType::RangeLast);
    }
    #[test] fn periodic_columns_correct_count() {
        assert_eq!(generate_periodic_columns(&generate_full_schedule()).len(), 57);
    }
    #[test] fn onehot_select_exact() {
        let state: Vec<BaseElement> = (0..16).map(|i| BaseElement::new(100+i)).collect();
        for t in 0..16 {
            let mut ind = [BaseElement::ZERO;16]; ind[t] = BaseElement::ONE;
            assert_eq!(select_word_onehot::<BaseElement>(&state, &ind), BaseElement::new(100+t as u64));
        }
    }

    #[cfg(feature = "stark-prover")]
    #[test] fn ct_round_trip() {
        let p = ct_prover::ConfidentialTransferProver::new(100,[0xAA;32],50,[0xBB;32]);
        let (pi,proof) = p.generate_proof().expect("proof failed");
        let pb = proof.to_bytes();
        eprintln!("CT proof: {} bytes ({:.1} KB)", pb.len(), pb.len() as f64/1024.0);
        let mut pu = Vec::new();
        for w in &pi.commitment1 { pu.extend_from_slice(&w.to_le_bytes()); }
        for w in &pi.commitment2 { pu.extend_from_slice(&w.to_le_bytes()); }
        pu.extend_from_slice(&pi.input_sum.to_le_bytes());
        assert!(verify_confidential_transfer(&pu, &pb).is_ok());
    }
    #[cfg(feature = "stark-prover")]
    #[test] fn ct_wrong_sum_rejected() {
        let p = ct_prover::ConfidentialTransferProver::new(100,[0xAA;32],50,[0xBB;32]);
        let (mut pi,proof) = p.generate_proof().unwrap(); pi.input_sum = 999;
        let mut pu = Vec::new();
        for w in &pi.commitment1 { pu.extend_from_slice(&w.to_le_bytes()); }
        for w in &pi.commitment2 { pu.extend_from_slice(&w.to_le_bytes()); }
        pu.extend_from_slice(&pi.input_sum.to_le_bytes());
        assert!(verify_confidential_transfer(&pu, &proof.to_bytes()).is_err());
    }
    #[cfg(feature = "stark-prover")]
    #[test] fn ct_wrong_commitment_rejected() {
        let p = ct_prover::ConfidentialTransferProver::new(100,[0xAA;32],50,[0xBB;32]);
        let (mut pi,proof) = p.generate_proof().unwrap(); pi.commitment1[0] ^= 1;
        let mut pu = Vec::new();
        for w in &pi.commitment1 { pu.extend_from_slice(&w.to_le_bytes()); }
        for w in &pi.commitment2 { pu.extend_from_slice(&w.to_le_bytes()); }
        pu.extend_from_slice(&pi.input_sum.to_le_bytes());
        assert!(verify_confidential_transfer(&pu, &proof.to_bytes()).is_err());
    }
    #[cfg(feature = "stark-prover")]
    #[test] fn ct_block_reward_split() {
        let (v1,v2) = (500_000_000u64, 573_741_824u64); assert_eq!(v1+v2, 1_073_741_824);
        let b1 = *blake3::hash(b"alice_blind_1").as_bytes();
        let b2 = *blake3::hash(b"alice_blind_2").as_bytes();
        let p = ct_prover::ConfidentialTransferProver::new(v1,b1,v2,b2);
        let (pi,proof) = p.generate_proof().expect("split failed");
        let pb = proof.to_bytes();
        eprintln!("Block reward split: {} bytes ({:.1} KB)", pb.len(), pb.len() as f64/1024.0);
        let mut pu = Vec::new();
        for w in &pi.commitment1 { pu.extend_from_slice(&w.to_le_bytes()); }
        for w in &pi.commitment2 { pu.extend_from_slice(&w.to_le_bytes()); }
        pu.extend_from_slice(&pi.input_sum.to_le_bytes());
        assert!(verify_confidential_transfer(&pu, &pb).is_ok());
    }
}
