# Midstate Coding & Documentation Standards

This document defines the expected style and **mandatory formal documentation requirements** for security-critical and state-machine code in the Midstate cryptocurrency.

## 1. Philosophy

Midstate is financial software running on unreliable edge hardware with novel post-quantum cryptography and a custom state machine. Correctness is not optional.

All code that participates in:
- Consensus state transitions
- Cryptographic key management / one-time signature enforcement
- Persistent state (especially accumulators and HD counters)
- Commit-reveal lifecycle

**must** be accompanied by high-quality documentation that makes reasoning explicit and invariants checkable.

## 2. Formal Documentation Standard (Mandatory)

Every public or semi-public function in the above categories **must** contain a documentation block with the following sections (in this order):

### 2.1 Reasoning

Explain:
- Why this function exists.
- What danger the previous (or alternative) implementation carried.
- The key security, economic, or operational invariant it protects.

### 2.2 Formal Specification

Provide both a lightweight textual form **and** a Z-style schema where the operation has non-trivial state impact.

#### Textual Pre / Post Form (required)

```rust
/// # Formal Specification
///
/// ```text
/// Pre:
///   - commitment ∈ state.commitments
///   - ∀ input ∈ tx.inputs: input.coin_id ∈ state.coins
///   - value conservation holds
///
/// Post:
///   result = Ok(())  ⇒
///     state.coins'        = state.coins \ tx.inputs ∪ tx.outputs
///     state.commitments'  = state.commitments \ {commitment}
///     root(state.coins' ∪ state.commitments') = declared_state_root
///
///   result = Err(_)  ⇒ state unchanged
/// ```
```

#### Z Notation Schema (strongly preferred for core transitions)

Use the conventions already established in `src/core/mmr.rs`:

```text
Hash               ≜ BLAKE3 output ([u8; 32])
seq T              ≜ finite sequence of T
⟨a, b, c⟩          ≜ literal sequence
s ⌢ t              ≜ sequence concatenation
#s                 ≜ length / cardinality
ℙ T                ≜ power set of T
pre / post         ≜ pre- and postcondition
x'                 ≜ value of x AFTER the operation
x?                 ≜ input parameter named x
y!                 ≜ output named y
𝔹                  ≜ {true, false}
```

Example schema:

```zed
    ApplyReveal
    ----------------
    ΔUtxoState
    ΔCommitments
    tx : RevealTransaction
    v2 : 𝔹

    pre  tx.commitment ∈ commitments
    pre  ∀ cid ∈ tx.input_coin_ids • cid ∈ coins
    pre  value_sum(tx.inputs) = value_sum(tx.outputs) + fee

    post coins'        = (coins \ tx.input_coin_ids) ∪ tx.new_coins
    post commitments'  = commitments \ {tx.commitment}
    post root(coins' ∪ commitments', v2) = declared_root
```

### 2.3 Safety / Invariants

List any important global or module-level invariants this function participates in maintaining (or must not break).

## 3. Where Formal Documentation Is Required

**Tier 1 (Full Z + Pre/Post + Reasoning required)**:
- Everything in `core/mmr.rs` (UtxoAccumulator and MerkleMountainRange)
- `core/state.rs` (State transitions, `apply_batch*`)
- `core/transaction.rs` (apply_transaction*, commit/reveal logic)
- Wallet HD advancement and persistence logic (`wallet/mod.rs`, `wallet/hd.rs`)
- Storage snapshot / truncation / migration paths

**Tier 2 (At least clear Pre/Post + Reasoning required)**:
- Script VM execution
- Mining template construction
- Reorg handling paths
- MSS / WOTS key lifecycle functions

**Tier 3 (Good Reasoning + lightweight pre/post recommended)**:
- Most of the rest of the core and wallet crates

## 4. Practical Guidelines

- Put the formal block **after** the one-sentence summary and before examples / implementation notes.
- Keep Z schemas small and focused. One schema per major state transition is better than one giant unreadable schema.
- When a function has complex error cases, document both the success post-condition **and** the "state unchanged on error" post-condition.
- Update the formal documentation whenever you change the semantics of a Tier 1 or Tier 2 function — the docs are part of the specification.

## 5. Tooling & Future Work

- The long-term goal is to make as many of these Z schemas as possible executable as property tests.
- Consider adding a `cargo xtask spec` or similar in the future to extract schemas.
- External model checking or theorem proving (Z/EVES, Isabelle/HOL, Lean, etc.) is out of scope for normal development but welcomed for the highest-risk modules.

---

**This standard is non-negotiable for any code that touches money or consensus state.**

When in doubt, write the spec first, then the code.