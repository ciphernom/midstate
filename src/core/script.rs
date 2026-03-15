//! MidstateScript — A Turing-incomplete stack machine for Midstate.
//!
//! Every UTXO is locked by a compiled bytecode script. The VM executes
//! witness inputs pushed onto the stack, then evaluates the script.
//! Execution succeeds iff the final top-of-stack item is exactly `[1]`.
//!
//! # STARK Integration (`OP_VERIFY_STARK`)
//! To prevent massive heap allocations when evaluating ZK-STARKs (~128 KB),
//! proofs are passed in the witness array but are **intentionally bypassed**
//! by the `SmallVec` stack. The VM reads the proof payload directly from the 
//! transaction slice when `OP_VERIFY_STARK` is called.

use super::types::{hash, OutputData};
use super::wots;
use super::mss;
use smallvec::SmallVec;

/// Stack element type alias. 32 bytes covers hashes and public keys inline;
/// larger items (e.g. WOTS signatures at 576 bytes) spill to the heap exactly
/// as a Vec would, so there is no correctness change — only fewer allocations
/// for the 99% case.
type StackItem = SmallVec<[u8; 32]>;

// ── Opcodes ────────────────────────────────────────────────────────────────

pub const OP_PUSH_DATA: u8        = 0x01;

pub const OP_DROP: u8             = 0x10;
pub const OP_DUP: u8              = 0x11;
pub const OP_SWAP: u8             = 0x12;

pub const OP_EQUAL: u8            = 0x20;
pub const OP_VERIFY: u8           = 0x21;
pub const OP_EQUALVERIFY: u8      = 0x22;
pub const OP_ADD: u8              = 0x23;
pub const OP_GREATER_OR_EQUAL: u8 = 0x24;

pub const OP_HASH: u8             = 0x30;
pub const OP_CHECKSIG: u8         = 0x31;
pub const OP_CHECKSIGVERIFY: u8   = 0x32;
pub const OP_CHECKTIMEVERIFY: u8  = 0x33;

pub const OP_IF: u8               = 0x40;
pub const OP_ELSE: u8             = 0x41;
pub const OP_ENDIF: u8            = 0x42;

pub const OP_SUM_TO_ADDR: u8      = 0x50;

/// STARK Verifier Opcode.
/// Stack behavior: pops `public_inputs` and `program_id`. Reads proof payload
/// directly from the end of the witness array without allocating it to the stack.
pub const OP_VERIFY_STARK: u8     = 0x60;

// ── Consensus limits ───────────────────────────────────────────────────────

pub const MAX_SCRIPT_SIZE: usize  = 1_024;
pub const MAX_STACK_DEPTH: usize  = 64;

/// Large enough for STARK proofs (~20-50 KB). Stack depth of 64 bounds
/// total memory: 64 × 131,072 = 8 MB worst case, which is acceptable.
pub const MAX_ITEM_SIZE: usize    = 131_072;
pub const MAX_SIGOPS_PER_SCRIPT: usize = 3;

/// STARK proofs are expensive to verify — at most 1 per script.
pub const MAX_STARK_OPS_PER_SCRIPT: usize = 1;

// ── Errors ─────────────────────────────────────────────────────────────────

/// Represents execution failures in the script VM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptError {
    ScriptTooLarge,
    StackOverflow,
    StackUnderflow,
    ItemTooLarge,
    InvalidOpcode(u8),
    PushDataOutOfBounds,
    UnbalancedConditional,
    VerifyFailed,
    MathOverflow,
    InvalidBooleanOnStack,
    ScriptMustFinishTrue,
    EmptyStack,
    CleanStackRuleFailed,
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScriptTooLarge => write!(f, "script exceeds {} bytes", MAX_SCRIPT_SIZE),
            Self::StackOverflow => write!(f, "stack depth exceeds {}", MAX_STACK_DEPTH),
            Self::StackUnderflow => write!(f, "stack underflow"),
            Self::ItemTooLarge => write!(f, "item exceeds {} bytes", MAX_ITEM_SIZE),
            Self::InvalidOpcode(op) => write!(f, "invalid opcode 0x{:02x}", op),
            Self::PushDataOutOfBounds => write!(f, "PUSH_DATA length exceeds bytecode"),
            Self::UnbalancedConditional => write!(f, "unbalanced IF/ELSE/ENDIF"),
            Self::VerifyFailed => write!(f, "VERIFY failed"),
            Self::MathOverflow => write!(f, "integer overflow in ADD"),
            Self::InvalidBooleanOnStack => write!(f, "expected boolean on stack"),
            Self::ScriptMustFinishTrue => write!(f, "script did not finish with [1] on top"),
            Self::EmptyStack => write!(f, "stack empty at end of execution"),
            Self::CleanStackRuleFailed => write!(f, "script execution finished with extra items on the stack (Clean Stack Rule)"),
        }
    }
}

impl std::error::Error for ScriptError {}

// ── Execution context ──────────────────────────────────────────────────────

/// Everything the VM needs beyond the script itself.
pub struct ExecContext<'a> {
    pub commitment: &'a [u8; 32],
    pub height: u64,
    pub outputs: &'a [OutputData],
}

// ── AOT validation ─────────────────────────────────────────────────────────

/// Ahead-of-time structural validation. O(N) single pass.
///
/// ```rust
/// use midstate::core::script::{validate_structure, OP_IF, OP_ENDIF, OP_ADD};
/// assert!(validate_structure(&[OP_IF, OP_ADD, OP_ENDIF], u64::MAX).is_ok());
/// assert!(validate_structure(&[OP_IF, OP_ADD], u64::MAX).is_err()); // Unbalanced
/// ```
pub fn validate_structure(bytecode: &[u8], height: u64) -> Result<bool, ScriptError> {
    if bytecode.len() > MAX_SCRIPT_SIZE {
        return Err(ScriptError::ScriptTooLarge);
    }

    let mut pc = 0usize;
    let mut if_depth: i32 = 0;
    let mut has_stark = false;

    while pc < bytecode.len() {
        let op = bytecode[pc];
        pc += 1;

        match op {
            OP_PUSH_DATA => {
                if pc + 2 > bytecode.len() {
                    return Err(ScriptError::PushDataOutOfBounds);
                }
                let len = u16::from_le_bytes([bytecode[pc], bytecode[pc + 1]]) as usize;
                pc += 2;
                if pc + len > bytecode.len() {
                    return Err(ScriptError::PushDataOutOfBounds);
                }
                if len > MAX_ITEM_SIZE {
                    return Err(ScriptError::ItemTooLarge);
                }
                pc += len;
            }
            OP_IF => { if_depth += 1; }
            OP_ELSE => {
                if if_depth <= 0 { return Err(ScriptError::UnbalancedConditional); }
            }
            OP_ENDIF => {
                if_depth -= 1;
                if if_depth < 0 { return Err(ScriptError::UnbalancedConditional); }
            }
            OP_DROP | OP_DUP | OP_SWAP |
            OP_EQUAL | OP_VERIFY | OP_EQUALVERIFY |
            OP_ADD | OP_GREATER_OR_EQUAL |
            OP_HASH | OP_CHECKSIG | OP_CHECKSIGVERIFY | OP_CHECKTIMEVERIFY |
            OP_SUM_TO_ADDR => {}
            
            OP_VERIFY_STARK => {
                if height < crate::core::types::STARK_ACTIVATION_HEIGHT {
                    return Err(ScriptError::InvalidOpcode(op));
                }
                has_stark = true; // <-- We safely found the actual opcode!
            }
            _ => return Err(ScriptError::InvalidOpcode(op)),
        }
    }

    if if_depth != 0 {
        return Err(ScriptError::UnbalancedConditional);
    }
    
    Ok(has_stark) // <-- Return the boolean
}

// ── Stack helpers ──────────────────────────────────────────────────────────

fn stack_push(stack: &mut Vec<StackItem>, item: StackItem) -> Result<(), ScriptError> {
    if item.len() > MAX_ITEM_SIZE {
        return Err(ScriptError::ItemTooLarge);
    }
    if stack.len() >= MAX_STACK_DEPTH {
        return Err(ScriptError::StackOverflow);
    }
    stack.push(item);
    Ok(())
}

fn stack_pop(stack: &mut Vec<StackItem>) -> Result<StackItem, ScriptError> {
    stack.pop().ok_or(ScriptError::StackUnderflow)
}

fn to_u64(item: &[u8]) -> Result<u64, ScriptError> {
    if item.len() > 8 {
        // Reject oversized integers to prevent malleability
        return Err(ScriptError::InvalidOpcode(0)); 
    }
    let mut buf = [0u8; 8];
    buf[..item.len()].copy_from_slice(item);
    Ok(u64::from_le_bytes(buf))
}

fn from_u64(v: u64) -> StackItem {
    if v == 0 {
        return SmallVec::from_slice(&[0]);
    }
    let mut bytes = v.to_le_bytes().to_vec();
    while bytes.len() > 1 && bytes.last() == Some(&0) {
        bytes.pop();
    }
    SmallVec::from_vec(bytes)
}

fn is_true(item: &[u8]) -> bool {
    item.iter().any(|&b| b != 0)
}

// ── Signature verification ─────────────────────────────────────────────────

fn verify_signature(sig_bytes: &[u8], message: &[u8; 32], pk_bytes: &[u8]) -> bool {
    if pk_bytes.len() != 32 {
        return false;
    }
    let pk: [u8; 32] = pk_bytes.try_into().unwrap();
    if sig_bytes.len() == wots::SIG_SIZE {
        match wots::sig_from_bytes(sig_bytes) {
            Some(sig) => wots::verify(&sig, message, &pk),
            None => false,
        }
    } else {
        match mss::MssSignature::from_bytes(sig_bytes) {
            Ok(mss_sig) => mss::verify(&mss_sig, message, &pk),
            Err(_) => false,
        }
    }
}

// ── Main execution engine ──────────────────────────────────────────────────

/// Executes the script VM over the provided bytecode.
///
/// **STARK Exception**: If `OP_VERIFY_STARK` is present, the final element in
/// the `witness` array is treated as the STARK proof payload. It is read
/// directly from the slice and *not* pushed to the `SmallVec` stack to prevent
/// huge heap allocation spikes during processing.
///
/// ```rust
/// use midstate::core::script::{execute_script, ExecContext, OP_ADD, OP_EQUAL};
/// 
/// // A simple script testing 2 + 2 == 4
/// let script = vec![OP_ADD, 0x01, 0x01, 0x00, 0x04, OP_EQUAL]; // PUSH 4, EQUAL
/// let witness = vec![vec![0x02], vec![0x02]];
/// let ctx = ExecContext { commitment: &[0; 32], height: 0, outputs: &[] };
/// 
/// assert!(execute_script(&script, &witness, &ctx).is_ok());
/// ```
pub fn execute_script(
    bytecode: &[u8],
    witness: &[Vec<u8>],
    ctx: &ExecContext,
) -> Result<(), ScriptError> {
    // FIX 2: Get the boolean safely from the structural parser!
    let has_stark = validate_structure(bytecode, ctx.height)?;

    // Bypass the SmallVec stack for STARK proofs to avoid massive memory allocations.
    let proof_bytes = if has_stark {
        witness.last().ok_or(ScriptError::StackUnderflow)?.as_slice()
    } else {
        &[]
    };

    // Only push standard stack items (signatures, public keys, inputs) to the VM stack
    let push_limit = if has_stark { witness.len().saturating_sub(1) } else { witness.len() };

    let mut stack: Vec<StackItem> = Vec::new();
    for item in &witness[..push_limit] {
        stack_push(&mut stack, SmallVec::from_slice(item))?;
    }
    
    let mut sigop_count = 0;
    let mut stark_op_count = 0;
    let mut pc = 0usize;
    let mut exec_stack: Vec<bool> = Vec::new();

    while pc < bytecode.len() {
        let executing = exec_stack.iter().all(|&e| e);
        let op = bytecode[pc];
        pc += 1;

        // Control flow opcodes are always processed
        match op {
            OP_IF => {
                if executing {
                    let cond = stack_pop(&mut stack)?;
                    exec_stack.push(is_true(&cond));
                } else {
                    exec_stack.push(false);
                }
                continue;
            }
            OP_ELSE => {
                if exec_stack.is_empty() {
                    return Err(ScriptError::UnbalancedConditional);
                }
                let parent_executing = exec_stack.len() <= 1
                    || exec_stack[..exec_stack.len() - 1].iter().all(|&e| e);
                if parent_executing {
                    let last = exec_stack.last_mut().unwrap();
                    *last = !*last;
                }
                continue;
            }
            OP_ENDIF => {
                if exec_stack.is_empty() {
                    return Err(ScriptError::UnbalancedConditional);
                }
                exec_stack.pop();
                continue;
            }
            OP_CHECKSIG | OP_CHECKSIGVERIFY => {
                if executing {
                    sigop_count += 1;
                    if sigop_count > MAX_SIGOPS_PER_SCRIPT {
                        return Err(ScriptError::VerifyFailed); 
                    }
                }
            }
            OP_VERIFY_STARK => {
                if executing {
                    stark_op_count += 1;
                    if stark_op_count > MAX_STARK_OPS_PER_SCRIPT {
                        return Err(ScriptError::VerifyFailed);
                    }
                }
            }
            _ => {}
        }

        if !executing {
            if op == OP_PUSH_DATA {
                if pc + 2 > bytecode.len() {
                    return Err(ScriptError::PushDataOutOfBounds);
                }
                let len = u16::from_le_bytes([bytecode[pc], bytecode[pc + 1]]) as usize;
                pc += 2 + len;
            }
            continue;
        }

        // ── Execute opcode ─────────────────────────────────────────────
        match op {
            OP_PUSH_DATA => {
                let len = u16::from_le_bytes([bytecode[pc], bytecode[pc + 1]]) as usize;
                pc += 2;
                let data = SmallVec::from_slice(&bytecode[pc..pc + len]);
                pc += len;
                stack_push(&mut stack, data)?;
            }

            OP_DROP => { stack_pop(&mut stack)?; }
            OP_DUP => {
                let top = stack.last().ok_or(ScriptError::StackUnderflow)?.clone();
                stack_push(&mut stack, top)?;
            }
            OP_SWAP => {
                let len = stack.len();
                if len < 2 { return Err(ScriptError::StackUnderflow); }
                stack.swap(len - 1, len - 2);
            }

            OP_EQUAL => {
                let b = stack_pop(&mut stack)?;
                let a = stack_pop(&mut stack)?;
                let result: StackItem = if a == b { SmallVec::from_slice(&[1u8]) } else { SmallVec::from_slice(&[0u8]) };
                stack_push(&mut stack, result)?;
            }
            OP_VERIFY => {
                let top = stack_pop(&mut stack)?;
                if !is_true(&top) { return Err(ScriptError::VerifyFailed); }
            }
            OP_EQUALVERIFY => {
                let b = stack_pop(&mut stack)?;
                let a = stack_pop(&mut stack)?;
                if a != b { return Err(ScriptError::VerifyFailed); }
            }
            OP_ADD => {
                let b = stack_pop(&mut stack)?;
                let a = stack_pop(&mut stack)?;
                let sum = to_u64(&a)?.checked_add(to_u64(&b)?).ok_or(ScriptError::MathOverflow)?;
                stack_push(&mut stack, from_u64(sum))?;
            }
            OP_GREATER_OR_EQUAL => {
                let b = stack_pop(&mut stack)?;
                let a = stack_pop(&mut stack)?;
                let result: StackItem = if to_u64(&a)? >= to_u64(&b)? { SmallVec::from_slice(&[1u8]) } else { SmallVec::from_slice(&[0u8]) };
                stack_push(&mut stack, result)?;
            }

            OP_HASH => {
                let data = stack_pop(&mut stack)?;
                let h = hash(&data);
                stack_push(&mut stack, SmallVec::from_slice(&h))?;
            }
            OP_CHECKSIG => {
                let pk = stack_pop(&mut stack)?;
                let sig = stack_pop(&mut stack)?;
                let valid = verify_signature(&sig, ctx.commitment, &pk);
                stack_push(&mut stack, if valid { SmallVec::from_slice(&[1u8]) } else { SmallVec::from_slice(&[0u8]) })?;
            }
            OP_CHECKSIGVERIFY => {
                let pk = stack_pop(&mut stack)?;
                let sig = stack_pop(&mut stack)?;
                if !verify_signature(&sig, ctx.commitment, &pk) {
                    return Err(ScriptError::VerifyFailed);
                }
            }
            OP_CHECKTIMEVERIFY => {
                let height_item = stack_pop(&mut stack)?;
                let required_height = to_u64(&height_item)?;
                if ctx.height < required_height {
                    return Err(ScriptError::VerifyFailed);
                }
            }

            OP_SUM_TO_ADDR => {
                let addr_item = stack_pop(&mut stack)?;
                if addr_item.len() != 32 {
                    return Err(ScriptError::VerifyFailed);
                }
                let addr: [u8; 32] = addr_item.as_slice().try_into().unwrap();
                let mut sum: u64 = 0;
                for out in ctx.outputs {
                    if out.address() == addr {
                        sum = sum.checked_add(out.value()).ok_or(ScriptError::MathOverflow)?;
                    }
                }
                stack_push(&mut stack, from_u64(sum))?;
            }

            OP_VERIFY_STARK => {
                // Stack: [... program_id, public_inputs] 
                // Proof bytes were intercepted directly from the witness array
                let public_inputs = stack_pop(&mut stack)?;
                let program_id_item = stack_pop(&mut stack)?;

                if program_id_item.len() != 32 {
                    return Err(ScriptError::VerifyFailed);
                }
                let program_id: [u8; 32] = program_id_item.as_slice().try_into().unwrap();

                super::stark::verify_stark_proof(
                    &program_id,
                    &public_inputs,
                    proof_bytes,
                ).map_err(|_| ScriptError::VerifyFailed)?;
                // Does NOT push anything — acts like VERIFY (fails or continues).
            }

            _ => return Err(ScriptError::InvalidOpcode(op)),
        }
    }

    if stack.is_empty() { return Err(ScriptError::EmptyStack); }
    // Clean Stack Rule: Exactly 1 item must remain
    if stack.len() > 1 { return Err(ScriptError::CleanStackRuleFailed); }
    
    let top = stack.last().unwrap();
    if top.as_slice() == &[1u8] { Ok(()) } else { Err(ScriptError::ScriptMustFinishTrue) }
}

// ── Script builders ────────────────────────────────────────────────────────

/// Standard Pay-to-Public-Key script.
///
/// ```rust
/// use midstate::core::script::{compile_p2pk, OP_PUSH_DATA, OP_CHECKSIGVERIFY};
/// let pk = [0xAA; 32];
/// let script = compile_p2pk(&pk);
/// assert_eq!(script[0], OP_PUSH_DATA);
/// assert_eq!(script[35], OP_CHECKSIGVERIFY);
/// ```
pub fn compile_p2pk(owner_pk: &[u8; 32]) -> Vec<u8> {
    let mut bc = Vec::new();
    push_data(&mut bc, owner_pk);
    bc.push(OP_CHECKSIGVERIFY);
    push_int(&mut bc, 1);
    bc
}

/// HTLC script. Claim: [Sig, Preimage, 1], Refund: [Sig, <dummy>, 0]
pub fn compile_htlc(
    secret_hash: &[u8; 32],
    receiver_pk: &[u8; 32],
    timeout_height: u64,
    refund_pk: &[u8; 32],
) -> Vec<u8> {
    let mut bc = Vec::new();
    bc.push(OP_IF);
    bc.push(OP_HASH);
    push_data(&mut bc, secret_hash);
    bc.push(OP_EQUALVERIFY);
    push_data(&mut bc, receiver_pk);
    bc.push(OP_CHECKSIGVERIFY);
    bc.push(OP_ELSE);
    bc.push(OP_DROP);
    push_int(&mut bc, timeout_height);
    bc.push(OP_CHECKTIMEVERIFY);
    push_data(&mut bc, refund_pk);
    bc.push(OP_CHECKSIGVERIFY);
    bc.push(OP_ENDIF);
    push_int(&mut bc, 1);
    bc
}

/// 2-of-3 multisig script. Witness: [Sig1, Sig2, Sig3] (0x00 for missing)
pub fn compile_multisig_2of3(
    pk1: &[u8; 32], pk2: &[u8; 32], pk3: &[u8; 32],
) -> Vec<u8> {
    let mut bc = Vec::new();
    push_data(&mut bc, pk3);
    bc.push(OP_CHECKSIG);
    bc.push(OP_SWAP);
    push_data(&mut bc, pk2);
    bc.push(OP_CHECKSIG);
    bc.push(OP_ADD);
    bc.push(OP_SWAP);
    push_data(&mut bc, pk1);
    bc.push(OP_CHECKSIG);
    bc.push(OP_ADD);
    push_int(&mut bc, 2);
    bc.push(OP_EQUAL);
    bc
}

// ── Bytecode assembly helpers ──────────────────────────────────────────────

pub fn push_data(bc: &mut Vec<u8>, data: &[u8]) {
    bc.push(OP_PUSH_DATA);
    let len = data.len() as u16;
    bc.extend_from_slice(&len.to_le_bytes());
    bc.extend_from_slice(data);
}

/// Append a PUSH_DATA instruction encoding a u64 as a minimal LE byte array.
pub fn push_int(bc: &mut Vec<u8>, value: u64) {
    push_data(bc, &from_u64(value));
}

// ── Assembler ──────────────────────────────────────────────────────────────

pub fn assemble(source: &str) -> Result<Vec<u8>, String> {
    let mut bc = Vec::new();
    let tokens: Vec<&str> = source.split_whitespace().collect();
    let mut i = 0;

    while i < tokens.len() {
        match tokens[i].to_uppercase().as_str() {
            "PUSH_HEX" => {
                i += 1;
                if i >= tokens.len() { return Err("PUSH_HEX requires a hex argument".into()); }
                let hex_str = tokens[i].trim_start_matches('<').trim_end_matches('>');
                let bytes = hex::decode(hex_str)
                    .map_err(|e| format!("invalid hex '{}': {}", hex_str, e))?;
                push_data(&mut bc, &bytes);
            }
            "PUSH_INT" => {
                i += 1;
                if i >= tokens.len() { return Err("PUSH_INT requires an integer argument".into()); }
                let val_str = tokens[i].trim_start_matches('<').trim_end_matches('>');
                let val: u64 = val_str.parse()
                    .map_err(|e| format!("invalid integer '{}': {}", val_str, e))?;
                push_int(&mut bc, val);
            }
            "DROP"              => bc.push(OP_DROP),
            "DUP"               => bc.push(OP_DUP),
            "SWAP"              => bc.push(OP_SWAP),
            "EQUAL"             => bc.push(OP_EQUAL),
            "VERIFY"            => bc.push(OP_VERIFY),
            "EQUALVERIFY"       => bc.push(OP_EQUALVERIFY),
            "ADD"               => bc.push(OP_ADD),
            "GREATER_OR_EQUAL"  => bc.push(OP_GREATER_OR_EQUAL),
            "HASH"              => bc.push(OP_HASH),
            "CHECKSIG"          => bc.push(OP_CHECKSIG),
            "CHECKSIGVERIFY"    => bc.push(OP_CHECKSIGVERIFY),
            "CHECKTIMEVERIFY"   => bc.push(OP_CHECKTIMEVERIFY),
            "IF"                => bc.push(OP_IF),
            "ELSE"              => bc.push(OP_ELSE),
            "ENDIF"             => bc.push(OP_ENDIF),
            "SUM_TO_ADDR"       => bc.push(OP_SUM_TO_ADDR),
            "VERIFY_STARK"      => bc.push(OP_VERIFY_STARK),
            other => return Err(format!("unknown mnemonic '{}'", other)),
        }
        i += 1;
    }

    let _ = validate_structure(&bc, u64::MAX).map_err(|e| e.to_string())?;
    Ok(bc)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wots;
    use crate::core::types::{hash, OutputData};

    fn empty_ctx() -> ExecContext<'static> {
        static ZERO: [u8; 32] = [0u8; 32];
        static OUTPUTS: [OutputData; 0] = [];
        ExecContext {
            commitment: &ZERO,
            height: 100,
            outputs: &OUTPUTS,
        }
    }

    #[test]
    fn aot_rejects_oversized_script() {
        let big = vec![OP_DUP; MAX_SCRIPT_SIZE + 1];
        assert_eq!(validate_structure(&big, u64::MAX), Err(ScriptError::ScriptTooLarge));
    }

    #[test]
    fn aot_rejects_push_data_oob() {
        let bc = vec![OP_PUSH_DATA, 0xFF, 0x00];
       assert_eq!(validate_structure(&bc, u64::MAX),  Err(ScriptError::PushDataOutOfBounds));
    }

    #[test]
    fn aot_rejects_unbalanced_if() {
        assert_eq!(validate_structure(&[OP_IF], u64::MAX), Err(ScriptError::UnbalancedConditional));
    }

    #[test]
    fn aot_rejects_extra_endif() {
        assert_eq!(validate_structure(&[OP_ENDIF], u64::MAX), Err(ScriptError::UnbalancedConditional));
    }

    #[test]
    fn aot_rejects_else_without_if() {
        assert_eq!(validate_structure(&[OP_ELSE], u64::MAX), Err(ScriptError::UnbalancedConditional));
    }

    #[test]
    fn aot_accepts_balanced_if_else_endif() {
        assert!(validate_structure(&[OP_IF, OP_ELSE, OP_ENDIF], u64::MAX).is_ok());
    }

    #[test]
    fn aot_rejects_invalid_opcode() {
        assert_eq!(validate_structure(&[0xFF], u64::MAX), Err(ScriptError::InvalidOpcode(0xFF)));
    }

    #[test]
    fn trivial_true_script() {
        let mut bc = Vec::new();
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &[], &empty_ctx()).is_ok());
    }

    #[test]
    fn trivial_false_script() {
        let mut bc = Vec::new();
        push_int(&mut bc, 0);
        assert_eq!(execute_script(&bc, &[], &empty_ctx()), Err(ScriptError::ScriptMustFinishTrue));
    }

    #[test]
    fn empty_script_fails() {
        assert_eq!(execute_script(&[], &[], &empty_ctx()), Err(ScriptError::EmptyStack));
    }

    #[test]
    fn dup_works() {
        let mut bc = Vec::new();
        bc.push(OP_DUP);
        bc.push(OP_EQUALVERIFY);
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &[vec![1u8]], &empty_ctx()).is_ok());
    }

    #[test]
    fn swap_works() {
        let mut bc = Vec::new();
        bc.push(OP_SWAP);
        bc.push(OP_DROP);
        assert!(execute_script(&bc, &[vec![42u8], vec![1u8]], &empty_ctx()).is_ok());
    }

    #[test]
    fn drop_underflow() {
        assert_eq!(execute_script(&[OP_DROP], &[], &empty_ctx()), Err(ScriptError::StackUnderflow));
    }

    #[test]
    fn equal_true() {
        let witness = vec![vec![0xAA; 4], vec![0xAA; 4]];
        assert!(execute_script(&[OP_EQUAL], &witness, &empty_ctx()).is_ok());
    }

    #[test]
    fn add_basic() {
        let mut bc = Vec::new();
        push_int(&mut bc, 3);
        push_int(&mut bc, 4);
        bc.push(OP_ADD);
        push_int(&mut bc, 7);
        bc.push(OP_EQUAL);
        assert!(execute_script(&bc, &[], &empty_ctx()).is_ok());
    }

    #[test]
    fn add_overflow_fails() {
        let witness = vec![u64::MAX.to_le_bytes().to_vec(), 1u64.to_le_bytes().to_vec()];
        assert_eq!(execute_script(&[OP_ADD], &witness, &empty_ctx()), Err(ScriptError::MathOverflow));
    }

    #[test]
    fn greater_or_equal_true() {
        let mut bc = Vec::new();
        push_int(&mut bc, 10);
        push_int(&mut bc, 5);
        bc.push(OP_GREATER_OR_EQUAL);
        assert!(execute_script(&bc, &[], &empty_ctx()).is_ok());
    }

    #[test]
    fn hash_opcode() {
        let preimage = b"secret";
        let expected = hash(preimage);
        let mut bc = Vec::new();
        bc.push(OP_HASH);
        push_data(&mut bc, &expected);
        bc.push(OP_EQUALVERIFY);
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &[preimage.to_vec()], &empty_ctx()).is_ok());
    }

    #[test]
    fn p2pk_valid_signature() {
        let seed = hash(b"test key seed");
        let pk = wots::keygen(&seed);
        let commitment = hash(b"test commitment");
        let sig = wots::sign(&seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);
        let bytecode = compile_p2pk(&pk);
        let ctx = ExecContext { commitment: &commitment, height: 100, outputs: &[] };
        assert!(execute_script(&bytecode, &[sig_bytes], &ctx).is_ok());
    }

    #[test]
    fn p2pk_invalid_signature() {
        let seed = hash(b"test key seed");
        let pk = wots::keygen(&seed);
        let commitment = hash(b"test commitment");
        let wrong_sig = vec![0u8; wots::SIG_SIZE];
        let bytecode = compile_p2pk(&pk);
        let ctx = ExecContext { commitment: &commitment, height: 100, outputs: &[] };
        assert!(execute_script(&bytecode, &[wrong_sig], &ctx).is_err());
    }

    #[test]
    fn checktimeverify_pass() {
        let mut bc = Vec::new();
        push_int(&mut bc, 50);
        bc.push(OP_CHECKTIMEVERIFY);
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &[], &empty_ctx()).is_ok());
    }

    #[test]
    fn checktimeverify_fail() {
        let mut bc = Vec::new();
        push_int(&mut bc, 200);
        bc.push(OP_CHECKTIMEVERIFY);
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &[], &empty_ctx()).is_err());
    }

    #[test]
    fn if_true_branch() {
        let mut bc = Vec::new();
        bc.push(OP_IF);
        push_int(&mut bc, 1);
        bc.push(OP_ELSE);
        push_int(&mut bc, 0);
        bc.push(OP_ENDIF);
        assert!(execute_script(&bc, &[vec![1u8]], &empty_ctx()).is_ok());
    }

    #[test]
    fn if_false_branch() {
        let mut bc = Vec::new();
        bc.push(OP_IF);
        push_int(&mut bc, 0);
        bc.push(OP_ELSE);
        push_int(&mut bc, 1);
        bc.push(OP_ENDIF);
        assert!(execute_script(&bc, &[vec![0u8]], &empty_ctx()).is_ok());
    }

    #[test]
    fn htlc_claim_path() {
        let secret = b"my secret preimage!!!!!!!!!!!!!!";
        let secret_hash = hash(secret);
        let receiver_seed = hash(b"receiver seed");
        let receiver_pk = wots::keygen(&receiver_seed);
        let refund_pk = [0xBB; 32];
        let commitment = hash(b"htlc commitment");
        let bytecode = compile_htlc(&secret_hash, &receiver_pk, 500, &refund_pk);
        let sig = wots::sign(&receiver_seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);
        let witness = vec![sig_bytes, secret.to_vec(), vec![1u8]];
        let ctx = ExecContext { commitment: &commitment, height: 100, outputs: &[] };
        assert!(execute_script(&bytecode, &witness, &ctx).is_ok());
    }

    #[test]
    fn htlc_refund_path() {
        let secret_hash = [0xAA; 32];
        let receiver_pk = [0xCC; 32];
        let refund_seed = hash(b"refund seed");
        let refund_pk = wots::keygen(&refund_seed);
        let commitment = hash(b"htlc refund commitment");
        let bytecode = compile_htlc(&secret_hash, &receiver_pk, 500, &refund_pk);
        let sig = wots::sign(&refund_seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);
        let witness = vec![sig_bytes, vec![0u8; 32], vec![0u8]];
        let ctx = ExecContext { commitment: &commitment, height: 600, outputs: &[] };
        assert!(execute_script(&bytecode, &witness, &ctx).is_ok());
    }

    #[test]
    fn htlc_refund_too_early_fails() {
        let secret_hash = [0xAA; 32];
        let receiver_pk = [0xCC; 32];
        let refund_seed = hash(b"refund seed");
        let refund_pk = wots::keygen(&refund_seed);
        let commitment = hash(b"htlc refund commitment");
        let bytecode = compile_htlc(&secret_hash, &receiver_pk, 500, &refund_pk);
        let sig = wots::sign(&refund_seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);
        let witness = vec![sig_bytes, vec![0u8; 32], vec![0u8]];
        let ctx = ExecContext { commitment: &commitment, height: 100, outputs: &[] };
        assert!(execute_script(&bytecode, &witness, &ctx).is_err());
    }

    #[test]
    fn sum_to_addr_covenant() {
        let alice_addr = [0xAA; 32];
        let outputs = vec![
            OutputData::Standard { address: alice_addr, value: 32, salt: [0; 32] },
            OutputData::Standard { address: alice_addr, value: 16, salt: [1; 32] },
            OutputData::Standard { address: [0xBB; 32], value: 4, salt: [2; 32] },
            OutputData::Standard { address: alice_addr, value: 2, salt: [3; 32] },
        ];
        let mut bc = Vec::new();
        push_data(&mut bc, &alice_addr);
        bc.push(OP_SUM_TO_ADDR);
        push_int(&mut bc, 50);
        bc.push(OP_GREATER_OR_EQUAL);
        bc.push(OP_VERIFY);
        push_int(&mut bc, 1);
        let ctx = ExecContext { commitment: &[0; 32], height: 0, outputs: &outputs };
        assert!(execute_script(&bc, &[], &ctx).is_ok());
    }

    #[test]
    fn sum_to_addr_insufficient_fails() {
        let alice_addr = [0xAA; 32];
        let outputs = vec![OutputData::Standard { address: alice_addr, value: 16, salt: [0; 32] }];
        let mut bc = Vec::new();
        push_data(&mut bc, &alice_addr);
        bc.push(OP_SUM_TO_ADDR);
        push_int(&mut bc, 50);
        bc.push(OP_GREATER_OR_EQUAL);
        bc.push(OP_VERIFY);
        push_int(&mut bc, 1);
        let ctx = ExecContext { commitment: &[0; 32], height: 0, outputs: &outputs };
        assert!(execute_script(&bc, &[], &ctx).is_err());
    }

    #[test]
    fn assemble_p2pk() {
        let pk_hex = "aa".repeat(32);
        let source = format!("PUSH_HEX {} CHECKSIGVERIFY PUSH_INT 1", pk_hex);
        let bc = assemble(&source).unwrap();
        assert_eq!(bc, compile_p2pk(&[0xAA; 32]));
    }

    #[test]
    fn assemble_invalid_mnemonic() {
        assert!(assemble("FOOBAR").is_err());
    }

    #[test]
    fn stack_overflow_rejected() {
        let witness: Vec<Vec<u8>> = (0..=MAX_STACK_DEPTH as u8).map(|_| vec![1u8]).collect();
        let mut bc = Vec::new();
        push_int(&mut bc, 1);
        assert!(execute_script(&bc, &witness, &empty_ctx()).is_err());
    }

    #[test]
    fn multisig_2of3_two_valid() {
        let seed1 = hash(b"key1");
        let seed2 = hash(b"key2");
        let seed3 = hash(b"key3");
        let pk1 = wots::keygen(&seed1);
        let pk2 = wots::keygen(&seed2);
        let pk3 = wots::keygen(&seed3);
        let commitment = hash(b"multisig commitment");
        let bytecode = compile_multisig_2of3(&pk1, &pk2, &pk3);
        let sig1 = wots::sig_to_bytes(&wots::sign(&seed1, &commitment));
        let sig2 = wots::sig_to_bytes(&wots::sign(&seed2, &commitment));
        let witness = vec![sig1, sig2, vec![0u8]];
        let ctx = ExecContext { commitment: &commitment, height: 0, outputs: &[] };
        assert!(execute_script(&bytecode, &witness, &ctx).is_ok());
    }

    #[test]
    fn multisig_2of3_one_valid_fails() {
        let seed1 = hash(b"key1");
        let seed2 = hash(b"key2");
        let seed3 = hash(b"key3");
        let pk1 = wots::keygen(&seed1);
        let pk2 = wots::keygen(&seed2);
        let pk3 = wots::keygen(&seed3);
        let commitment = hash(b"multisig commitment");
        let bytecode = compile_multisig_2of3(&pk1, &pk2, &pk3);
        let sig1 = wots::sig_to_bytes(&wots::sign(&seed1, &commitment));
        let witness = vec![sig1, vec![0u8], vec![0u8]];
        let ctx = ExecContext { commitment: &commitment, height: 0, outputs: &[] };
        assert!(execute_script(&bytecode, &witness, &ctx).is_err());
    }

    // Test 1: Basic STARK-gated script — prove a value is in range
    //
    // Witness: [program_id, public_inputs, proof_bytes]
    //   - program_id and public_inputs are pushed to stack
    //   - proof_bytes (last item) is intercepted directly, NOT pushed to stack
    //
    // Script: VERIFY_STARK PUSH_INT 1
    //   - pops public_inputs (top), then program_id (second)
    //   - verifies STARK proof
    //   - pushes 1 for clean stack exit
    #[cfg(feature = "stark-prover")]
    #[test]
    fn stark_range_proof_via_script() {
        use crate::core::stark::{self, RANGE_PROOF_64, prover::RangeProofProver};

        // 1. Generate proof for value = 1_000_000
        let value = 1_000_000u64;
        let prover = RangeProofProver::new(value);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();
        let value_bytes = pub_inputs.value.as_int().to_le_bytes().to_vec();

        println!("Proof size: {} bytes", proof_bytes.len());

        // 2. Build the script: VERIFY_STARK PUSH_INT 1
        let mut bytecode = Vec::new();
        bytecode.push(OP_VERIFY_STARK);
        push_int(&mut bytecode, 1);

        // 3. Witness: [program_id, public_inputs, proof]
        //    First two go on stack, last one is intercepted for STARK
        let witness = vec![
            RANGE_PROOF_64.to_vec(),  // → stack (program_id)
            value_bytes,               // → stack (public_inputs)
            proof_bytes,               // → intercepted (proof)
        ];

        // 4. Execute
        let ctx = ExecContext { commitment: &[0u8; 32], height: 100, outputs: &[] };
        let result = execute_script(&bytecode, &witness, &ctx);
        assert!(result.is_ok(), "STARK range proof script failed: {:?}", result);
    }

    // Test 2: STARK proof with wrong value should fail
    #[cfg(feature = "stark-prover")]
    #[test]
    fn stark_range_proof_wrong_value_fails() {
        use crate::core::stark::{RANGE_PROOF_64, prover::RangeProofProver};

        let prover = RangeProofProver::new(42);
        let (_, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();

        // Claim value is 99, but proof is for 42
        let wrong_value_bytes = 99u64.to_le_bytes().to_vec();

        let mut bytecode = Vec::new();
        bytecode.push(OP_VERIFY_STARK);
        push_int(&mut bytecode, 1);

        let witness = vec![
            RANGE_PROOF_64.to_vec(),
            wrong_value_bytes,
            proof_bytes,
        ];

        let ctx = ExecContext { commitment: &[0u8; 32], height: 100, outputs: &[] };
        let result = execute_script(&bytecode, &witness, &ctx);
        assert!(result.is_err(), "Should have failed with wrong value");
    }

    // Test 3: STARK + signature combo — prove value in range AND sign the tx
    //
    // This is the real use case: a UTXO locked by
    //   "prove your value is valid AND you own the key"
    //
    // Witness: [signature, program_id, public_inputs, proof_bytes]
    // Script:  VERIFY_STARK <owner_pk> CHECKSIGVERIFY PUSH_INT 1
    #[cfg(feature = "stark-prover")]
    #[test]
    fn stark_plus_signature_script() {
        use crate::core::stark::{RANGE_PROOF_64, prover::RangeProofProver};
        use crate::core::wots;
        use crate::core::types::hash;

        // 1. Generate STARK proof
        let value = 500u64;
        let prover = RangeProofProver::new(value);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();
        let value_bytes = pub_inputs.value.as_int().to_le_bytes().to_vec();

        // 2. Generate WOTS key and signature
        let seed = hash(b"stark test key");
        let pk = wots::keygen(&seed);
        let commitment = hash(b"stark tx commitment");
        let sig = wots::sign(&seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);

        // 3. Build script: VERIFY_STARK <pk> CHECKSIGVERIFY PUSH_INT 1
        let mut bytecode = Vec::new();
        bytecode.push(OP_VERIFY_STARK);
        push_data(&mut bytecode, &pk);
        bytecode.push(OP_CHECKSIGVERIFY);
        push_int(&mut bytecode, 1);

        // 4. Witness: [sig, program_id, public_inputs, proof]
        //    sig, program_id, public_inputs → stack
        //    proof → intercepted
        let witness = vec![
            sig_bytes,                 // → stack (signature)
            RANGE_PROOF_64.to_vec(),  // → stack (program_id)
            value_bytes,               // → stack (public_inputs)
            proof_bytes,               // → intercepted (proof)
        ];

        // Stack after witness push: [sig, program_id, public_inputs]
        // VERIFY_STARK: pops public_inputs, pops program_id → verifies proof
        // Stack now: [sig]
        // PUSH_DATA pk: stack = [sig, pk]
        // CHECKSIGVERIFY: pops pk, pops sig → verifies signature
        // Stack now: []
        // PUSH_INT 1: stack = [1]
        // Clean stack check passes ✓

        let ctx = ExecContext { commitment: &commitment, height: 100, outputs: &[] };
        let result = execute_script(&bytecode, &witness, &ctx);
        assert!(result.is_ok(), "STARK + sig script failed: {:?}", result);
    }

    // Test 4: STARK + covenant — prove value in range AND enforce output destination
    //
    // "These coins can only be spent if you prove the value is valid 
    //  AND at least 100 units go to address X"
    #[cfg(feature = "stark-prover")]
    #[test]
    fn stark_plus_covenant_script() {
        use crate::core::stark::{RANGE_PROOF_64, prover::RangeProofProver};
        use crate::core::types::{hash, OutputData};

        let value = 1000u64;
        let prover = RangeProofProver::new(value);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();
        let value_bytes = pub_inputs.value.as_int().to_le_bytes().to_vec();

        let required_addr = [0xAA; 32];
        let outputs = vec![
            OutputData::Standard { address: required_addr, value: 128, salt: [0; 32] },
            OutputData::Standard { address: [0xBB; 32], value: 64, salt: [1; 32] },
        ];

        // Script: VERIFY_STARK <required_addr> SUM_TO_ADDR <100> GREATER_OR_EQUAL VERIFY PUSH_INT 1
        let mut bytecode = Vec::new();
        bytecode.push(OP_VERIFY_STARK);
        push_data(&mut bytecode, &required_addr);
        bytecode.push(OP_SUM_TO_ADDR);
        push_int(&mut bytecode, 100);
        bytecode.push(OP_GREATER_OR_EQUAL);
        bytecode.push(OP_VERIFY);
        push_int(&mut bytecode, 1);

        let witness = vec![
            RANGE_PROOF_64.to_vec(),
            value_bytes,
            proof_bytes,
        ];

        let ctx = ExecContext { commitment: &[0; 32], height: 100, outputs: &outputs };
        let result = execute_script(&bytecode, &witness, &ctx);
        assert!(result.is_ok(), "STARK + covenant script failed: {:?}", result);
    }
    
    // ── End-to-end: Load private_spend.msc, compile, prove, execute ──

    #[cfg(feature = "stark-prover")]
    #[test]
    fn private_spend_msc_end_to_end() {
        use crate::core::stark::{RANGE_PROOF_64, prover::RangeProofProver};
        use crate::core::wots;
        use crate::core::types::hash;
        use std::time::Instant;

        // 1. Generate owner keypair
        let seed = hash(b"private_spend_msc_test_key");
        let pk = wots::keygen(&seed);
        let pk_hex = hex::encode(&pk);

        // 2. Load and compile the .msc file
        let msc_source = "VERIFY_STARK\nPUSH_HEX OWNER_PK\nCHECKSIGVERIFY\nPUSH_INT 1";

        let source: String = msc_source
            .lines()
            .map(|l| l.split('#').next().unwrap().trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
            .replace("OWNER_PK", &pk_hex);

        println!("Compiled source: {}", source);
        let bytecode = assemble(&source).expect("Failed to assemble private_spend.msc");
        println!("Bytecode: {} bytes → {}", bytecode.len(), hex::encode(&bytecode));

        // 3. Generate STARK range proof for 1 block reward
        let value = 1_073_741_824u64;
        let t0 = Instant::now();
        let prover = RangeProofProver::new(value);
        let (pub_inputs, proof) = prover.generate_proof().unwrap();
        let proof_bytes = proof.to_bytes();
        println!("Proof generated in {:?} — {} bytes", t0.elapsed(), proof_bytes.len());

        // 4. Sign
        let commitment = hash(b"private spend commitment");
        let sig = wots::sign(&seed, &commitment);
        let sig_bytes = wots::sig_to_bytes(&sig);

        // 5. Assemble witness and execute
        let witness = vec![
            sig_bytes,
            RANGE_PROOF_64.to_vec(),
            pub_inputs.value.as_int().to_le_bytes().to_vec(),
            proof_bytes,
        ];

        let ctx = ExecContext {
            commitment: &commitment,
            height: 30_000,
            outputs: &[],
        };

        let t0 = Instant::now();
        let result = execute_script(&bytecode, &witness, &ctx);
        println!("Script executed in {:?}", t0.elapsed());

        assert!(result.is_ok(), "private_spend.msc failed: {:?}", result);
        println!("✅ private_spend.msc — STARK verified, signature verified, script passed");
    }
    
}
