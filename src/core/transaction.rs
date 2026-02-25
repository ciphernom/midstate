use super::types::*;
use super::script;
use anyhow::{bail, Result};

const COMMIT_POW_TARGET: u16 = 0x0000;

fn validate_commit_pow(commitment: &[u8; 32], nonce: u64) -> Result<()> {
    let h = super::types::hash_concat(commitment, &nonce.to_le_bytes());
    if u16::from_be_bytes([h[0], h[1]]) != COMMIT_POW_TARGET {
        bail!("Insufficient Commit PoW");
    }
    Ok(())
}

/// Pure signature verification only — no state reads or mutations.
/// Called in parallel across all transactions in a batch before sequential apply.
pub fn verify_transaction_sigs(tx: &Transaction, height: u64) -> Result<()> {
    if let Transaction::Reveal { inputs, witnesses, outputs, salt, .. } = tx {
        let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let output_commit_hashes: Vec<[u8; 32]> = outputs.iter()
            .map(|o| o.hash_for_commitment())
            .collect();
        let commitment = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

        for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
            if !verify_predicate(&input.predicate, witness, &commitment, height, outputs) {
                bail!("Predicate execution failed for input {}", i);
            }
        }
    }
    // Commits have no signature to verify
    Ok(())
}

/// Apply a transaction that has already passed signature verification.
/// Skips the verify_predicate call — all other validation still runs.
pub fn apply_transaction_no_sig_check(state: &mut State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { .. } => {
            // Commits are cheap — just delegate to the normal path
            apply_transaction(state, tx)
        }

        Transaction::Reveal { inputs, witnesses, outputs, salt, .. } => {
            if inputs.is_empty() { bail!("Transaction must spend at least one coin"); }
            if outputs.is_empty() { bail!("Transaction must create at least one new coin"); }
            if inputs.len() > MAX_TX_INPUTS { bail!("Too many inputs (max {})", MAX_TX_INPUTS); }
            if outputs.len() > MAX_TX_OUTPUTS { bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); }
            if witnesses.len() != inputs.len() { bail!("Witness count must match input count"); }

            let max_witness_size = MAX_SIGNATURE_SIZE * 16;
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }

            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                }
            }

            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 { bail!("Zero-value output {}", i); }
                if !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE {
                        bail!("DataBurn payload exceeds max size of {} bytes", crate::core::types::MAX_BURN_DATA_SIZE);
                    }
                }
            }

            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value))
                .ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value()))
                .ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum {
                bail!("Input value ({}) must exceed output value ({}) to pay fee", in_sum, out_sum);
            }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter()
                .map(|o| o.hash_for_commitment())
                .collect();
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !state.commitments.remove(&expected) {
                bail!("No matching commitment found (expected {})", hex::encode(expected));
            }

            // Coin existence check still needed — sig verification doesn't touch state
            for input in inputs.iter() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }
            }

            // NOTE: verify_predicate intentionally skipped here — already done in parallel

            for coin_id in &input_coin_ids {
                state.coins.remove(coin_id);
            }

            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id) {
                        bail!("Duplicate coin created");
                    }
                }
            }

            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids {
                    hasher.update(coin_id);
                }
                for hash in &output_commit_hashes {
                    hasher.update(hash);
                }
                hasher.update(salt);
                let tx_hash = *hasher.finalize().as_bytes();
                state.midstate = hash_concat(&state.midstate, &tx_hash);
            }

            Ok(())
        }
    }
}

/// Apply a transaction to the state
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce)?;
            if !state.commitments.insert(*commitment) {
                bail!("Duplicate commitment");
            }
            state.commitment_heights.insert(*commitment, state.height);
            state.midstate = hash_concat(&state.midstate, commitment);
            Ok(())
        }

        Transaction::Reveal { inputs, witnesses, outputs, salt, .. } => {
            if inputs.is_empty() {
                bail!("Transaction must spend at least one coin");
            }
            if outputs.is_empty() {
                bail!("Transaction must create at least one new coin");
            }
            if inputs.len() > MAX_TX_INPUTS { 
                bail!("Too many inputs (max {})", MAX_TX_INPUTS); 
                }
            if outputs.len() > MAX_TX_OUTPUTS { 
                bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); 
            }
            
            if witnesses.len() != inputs.len() {
                bail!("Witness count must match input count");
            }
            // Arbitrary payload size protection
            let max_witness_size = MAX_SIGNATURE_SIZE * 16; 
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                }
            }
            // 1. Validate all output values are power of 2 and nonzero
            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 {
                    bail!("Zero-value output {}", i);
                }
                if !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE {
                        bail!("DataBurn payload exceeds max size of {} bytes", crate::core::types::MAX_BURN_DATA_SIZE);
                    }
                }
            }

            // 2. Value conservation: sum(inputs) > sum(outputs)
            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value())).ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum {
                bail!(
                    "Input value ({}) must exceed output value ({}) to pay fee",
                    in_sum, out_sum
                );
            }

            // 3. Compute commitment hashes
            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();

            // 4. Verify commitment exists and matches
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);
            if !state.commitments.remove(&expected) {
                bail!(
                    "No matching commitment found (expected {})",
                    hex::encode(expected)
                );
            }

            // 5. Verify each input coin exists and executes cleanly against its Predicate
            for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }
                
                // Script Execution Engine
                if !verify_predicate(&input.predicate, witness, &expected, state.height, outputs) {
                    bail!("Predicate execution failed for input {}", i);
                }
            }

            // 6. Remove spent coins
            for coin_id in &input_coin_ids {
                state.coins.remove(coin_id);
            }

            // 7. Add new coins (Ignore DataBurns, protecting the SMT!)
            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id) {
                        bail!("Duplicate coin created");
                    }
                }
            }

            // 8. Update midstate
            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids {
                    hasher.update(coin_id);
                }
                for hash in &output_commit_hashes {
                    hasher.update(hash);
                }
                hasher.update(salt);
                let tx_hash = *hasher.finalize().as_bytes();
                state.midstate = hash_concat(&state.midstate, &tx_hash);
            }

            Ok(())
        }
    }
}

/// Execute a Witness against a Predicate via the MidstateScript VM.
fn verify_predicate(
    predicate: &Predicate,
    witness: &Witness,
    commitment: &[u8; 32],
    current_height: u64,
    outputs: &[OutputData],
) -> bool {
    match (predicate, witness) {
        (Predicate::Script { bytecode }, Witness::ScriptInputs(inputs)) => {
            let ctx = script::ExecContext {
                commitment,
                height: current_height,
                outputs,
            };
            script::execute_script(bytecode, inputs, &ctx).is_ok()
        }
    }
}

/// Validate a transaction without applying it
pub fn validate_transaction(state: &State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce)?;
            if state.commitments.contains(commitment) {
                bail!("Duplicate commitment");
            }
            Ok(())
        }

        Transaction::Reveal { inputs, witnesses, outputs, salt, .. } => {
            if inputs.is_empty() {
                bail!("Must spend at least one coin");
            }
            if outputs.is_empty() {
                bail!("Must create at least one coin");
            }
            if inputs.len() > MAX_TX_INPUTS { 
                bail!("Too many inputs (max {})", MAX_TX_INPUTS); 
                }
            if outputs.len() > MAX_TX_OUTPUTS { 
                bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); 
            }
            
            if witnesses.len() != inputs.len() {
                bail!("Witness count must match input count");
            }
            // Arbitrary payload size protection
            let max_witness_size = MAX_SIGNATURE_SIZE * 16; 
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                }
            }
                for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 {
                    bail!("Zero-value output {}", i);
                }
                if !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE {
                        bail!("DataBurn payload exceeds max size of {} bytes", crate::core::types::MAX_BURN_DATA_SIZE);
                    }
                }
            }

            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value())).ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum {
                bail!("Input value must exceed output value");
            }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();

            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);
            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found");
            }

            // Check commitment hasn't expired (or is about to expire this block)
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) >= COMMITMENT_TTL {
                    bail!("Commitment expired (committed at height {}, current {})", commit_height, state.height);
                }
            }
            // 5. Verify each Witness executes cleanly against its Predicate
            for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found", hex::encode(coin_id));
                }
                if !verify_predicate(&input.predicate, witness, &expected, state.height, outputs) {
                    bail!("Predicate execution failed for input {}", i);
                }
            }

            Ok(())
        }
    }
}

#[cfg(test)] 
mod tests {
    use super::*;
    use crate::core::mmr::UtxoAccumulator;
    use crate::core::wots;

    fn empty_state() -> State {
        State {
            midstate: [0u8; 32],
            coins: UtxoAccumulator::new(),
            commitments: UtxoAccumulator::new(),
            depth: 0,
            target: [0xff; 32],
            height: 1,
            timestamp: 1000,
            commitment_heights: im::HashMap::new(),
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
        }
    }

    fn mine_commit_nonce(commitment: &[u8; 32]) -> u64 {
        let mut n = 0u64;
        loop {
            let h = hash_concat(commitment, &n.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) == 0x0000 {
                return n;
            }
            n += 1;
        }
    }

    #[test]
    fn commit_pow_valid_nonce_passes() {
        let commitment = hash(b"test commitment");
        let nonce = mine_commit_nonce(&commitment);
        assert!(validate_commit_pow(&commitment, nonce).is_ok());
    }

    #[test]
    fn commit_pow_invalid_nonce_fails() {
        let commitment = hash(b"test commitment");
        // Nonce 0 is almost certainly invalid (1 in 65536 chance)
        // Try a few to find one that fails
        let mut bad_nonce = 0u64;
        loop {
            let h = hash_concat(&commitment, &bad_nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) != 0x0000 {
                break;
            }
            bad_nonce += 1;
        }
        assert!(validate_commit_pow(&commitment, bad_nonce).is_err());
    }

    #[test]
    fn commit_pow_benchmark() {
        // Measure time to mine a valid nonce — should be ~10-50ms
        let commitment = hash(b"benchmark commitment");
        let start = std::time::Instant::now();
        let nonce = mine_commit_nonce(&commitment);
        let elapsed = start.elapsed();
        // Verify it's actually valid
        assert!(validate_commit_pow(&commitment, nonce).is_ok());
        // Log timing (visible with `cargo test -- --nocapture`)
        eprintln!("Commit PoW mining took {:?} (nonce: {})", elapsed, nonce);
        // Soft assert: should complete within 5 seconds even on slow hardware
        assert!(elapsed.as_secs() < 5, "PoW took too long: {:?}", elapsed);
    }

    #[test]
    fn validate_transaction_rejects_bad_commit_pow() {
        let state = empty_state();
        let commitment = hash(b"reject test");
        // Find an invalid nonce
        let mut bad_nonce = 0u64;
        loop {
            let h = hash_concat(&commitment, &bad_nonce.to_le_bytes());
            if u16::from_be_bytes([h[0], h[1]]) != 0x0000 {
                break;
            }
            bad_nonce += 1;
        }
        let tx = Transaction::Commit { commitment, spam_nonce: bad_nonce };
        assert!(validate_transaction(&state, &tx).is_err());
    }

    #[test]
    fn validate_transaction_accepts_good_commit_pow() {
        let state = empty_state();
        let commitment = hash(b"accept test");
        let nonce = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: nonce };
        assert!(validate_transaction(&state, &tx).is_ok());
    }

    // ── Full Commit + Reveal flow ───────────────────────────────────────

    /// Helper: create a state with a spendable coin, returning (state, seed, coin_id, salt).
    fn state_with_coin(value: u64) -> (State, [u8; 32], [u8; 32], [u8; 32]) {
        let mut state = empty_state();
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        let salt = hash(b"test salt");
        let coin_id = compute_coin_id(&address, value, &salt);
        state.coins.insert(coin_id);
        (state, seed, coin_id, salt)
    }

    fn do_commit(state: &mut State, input_ids: &[[u8; 32]], output_ids: &[[u8; 32]], salt: &[u8; 32]) -> [u8; 32] {
        let commitment = compute_commitment(input_ids, output_ids, salt);
        let nonce = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: nonce };
        apply_transaction(state, &tx).unwrap();
        commitment
    }

    #[test]
    fn full_commit_reveal_flow() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(16);

        // Build output
        let out_addr = hash(b"recipient");
        let out_salt: [u8; 32] = [0x11; 32];
        let output = OutputData::Standard { address: out_addr, value: 8, salt: out_salt };
        let output_coin_id = output.coin_id();

        // Commit
        let commit_salt: [u8; 32] = [0x22; 32];
        let commitment = do_commit(
            &mut state,
            &[coin_id],
            &[output_coin_id.unwrap()],
            &commit_salt,
        );

        // Reveal
        let owner_pk = wots::keygen(&seed);
        let sig = wots::sign(&seed, &commitment);
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { 
                predicate: Predicate::p2pk(&owner_pk), 
                value: 16, 
                salt: input_salt 
            }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        apply_transaction(&mut state, &tx).unwrap();

        // Input coin spent
        assert!(!state.coins.contains(&coin_id));
        // Output coin created
        assert!(state.coins.contains(&output_coin_id.unwrap()));
    }

#[test]
    fn reveal_rejects_without_commit() {
        let (mut state, seed, _coin_id, input_salt) = state_with_coin(16);
        let owner_pk = wots::keygen(&seed);
        let output = OutputData::Standard { address: hash(b"r"), value: 8, salt: [0; 32] };
        let fake_commitment = hash(b"not committed");
        let sig = wots::sign(&seed, &fake_commitment);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: [0; 32],
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_wrong_signature() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(16);
        let owner_pk = wots::keygen(&seed);
        let output = OutputData::Standard { address: hash(b"r"), value: 8, salt: [0; 32] };
        let commit_salt: [u8; 32] = [0x33; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        
        // Sign with wrong key
        let wrong_seed: [u8; 32] = [0xFF; 32];
        let bad_sig = wots::sign(&wrong_seed, &commitment);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&bad_sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_nonexistent_coin() {
        let mut state = empty_state();
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = wots::keygen(&seed);
        let input_salt = [0u8; 32];
        let address = compute_address(&owner_pk);
        let coin_id = compute_coin_id(&address, 16, &input_salt);
        // Do NOT insert coin into state

        let output = OutputData::Standard { address: hash(b"r"), value: 8, salt: [0; 32] };
        let commit_salt = [1u8; 32];

        // Commit is allowed (doesn't check coins)
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let sig = wots::sign(&seed, &commitment);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    // ── Value conservation ──────────────────────────────────────────────

    #[test]
    fn reveal_rejects_output_exceeding_input() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(8);
        let owner_pk = wots::keygen(&seed);
        let output = OutputData::Standard { address: hash(b"r"), value: 8, salt: [0; 32] };
        let commit_salt = [0u8; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let sig = wots::sign(&seed, &commitment);

        // output value == input value, no fee → rejected
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn reveal_rejects_empty_inputs() {
        let mut state = empty_state();
        let tx = Transaction::Reveal {
            inputs: vec![],
            witnesses: vec![],
            outputs: vec![OutputData::Standard { address: [0; 32], value: 1, salt: [0; 32] }],
            salt: [0; 32],
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_empty_outputs() {
        let (mut state, seed, _coin_id, input_salt) = state_with_coin(8);
        let owner_pk = wots::keygen(&seed);
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt }],
            witnesses: vec![Witness::sig(vec![0; wots::SIG_SIZE])],
            outputs: vec![],
            salt: [0; 32],
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_mismatched_sig_count() {
        let (mut state, seed, _coin_id, input_salt) = state_with_coin(8);
        let owner_pk = wots::keygen(&seed);
        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt }],
            witnesses: vec![], // 0 sigs for 1 input
            outputs: vec![OutputData::Standard { address: [0; 32], value: 4, salt: [0; 32] }],
            salt: [0; 32],
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_zero_value_output() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(8);
        let owner_pk = wots::keygen(&seed);
        let output = OutputData::Standard { address: hash(b"r"), value: 0, salt: [0; 32] };
        let commit_salt = [0u8; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let sig = wots::sign(&seed, &commitment);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_non_power_of_two_output() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(16);
        let owner_pk = wots::keygen(&seed);
        let output = OutputData::Standard { address: hash(b"r"), value: 3, salt: [0; 32] };
        let commit_salt = [0u8; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);
        let sig = wots::sign(&seed, &commitment);

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig))],
            outputs: vec![output],
            salt: commit_salt,
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn reveal_rejects_duplicate_inputs() {
        let (mut state, seed, _coin_id, input_salt) = state_with_coin(16);
        let owner_pk = wots::keygen(&seed);
        let same_input = InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt };
        let tx = Transaction::Reveal {
            inputs: vec![same_input.clone(), same_input],
            witnesses: vec![Witness::sig(vec![0; wots::SIG_SIZE]), Witness::sig(vec![0; wots::SIG_SIZE])],
            outputs: vec![OutputData::Standard { address: [0; 32], value: 16, salt: [0; 32] }],
            salt: [0; 32],
        };
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    #[test]
    fn double_spend_prevented() {
        let (mut state, seed, coin_id, input_salt) = state_with_coin(16);
        let owner_pk = wots::keygen(&seed);

        // First spend
        let out1 = OutputData::Standard { address: hash(b"r1"), value: 8, salt: [1; 32] };
        let salt1 = [0xA0; 32];
        let commitment1 = do_commit(&mut state, &[coin_id], &[out1.coin_id().unwrap()], &salt1);
        let sig1 = wots::sign(&seed, &commitment1);
        let tx1 = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig1))],
            outputs: vec![out1],
            salt: salt1,
        };
        apply_transaction(&mut state, &tx1).unwrap();

        // Second spend of same coin should fail
        let out2 = OutputData::Standard { address: hash(b"r2"), value: 4, salt: [2; 32] };
        let salt2 = [0xB0; 32];
        let commitment2 = do_commit(&mut state, &[coin_id], &[out2.coin_id().unwrap()], &salt2);
        let sig2 = wots::sign(&seed, &commitment2);
        let tx2 = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt }],
            witnesses: vec![Witness::sig(wots::sig_to_bytes(&sig2))],
            outputs: vec![out2],
            salt: salt2,
        };
        assert!(apply_transaction(&mut state, &tx2).is_err());
    }

    // ── MSS signature path ──────────────────────────────────────────────

    #[test]
    fn reveal_with_mss_signature() {
        use crate::core::mss;

        let mut state = empty_state();
        let mss_seed = hash(b"mss test seed");
        let mut keypair = mss::keygen(&mss_seed, 4).unwrap();
        let master_pk = keypair.public_key(); // this is the owner_pk

        // The address = hash(master_pk)
        let address = compute_address(&master_pk);
        let input_salt = hash(b"mss coin salt");
        let value = 16u64;
        let coin_id = compute_coin_id(&address, value, &input_salt);
        state.coins.insert(coin_id);

        let output = OutputData::Standard { address: hash(b"dest"), value: 8, salt: [0; 32] };
        let commit_salt = [0xCC; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);

        // Sign with MSS
        let mss_sig = keypair.sign(&commitment).unwrap();
        let sig_bytes = mss_sig.to_bytes();

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&master_pk), value, salt: input_salt }],
            witnesses: vec![Witness::sig(sig_bytes)],
            outputs: vec![output],
            salt: commit_salt,
        };
        apply_transaction(&mut state, &tx).unwrap();
        assert!(!state.coins.contains(&coin_id));
    }

    // ── Duplicate commitment ────────────────────────────────────────────

    #[test]
    fn commit_duplicate_rejected() {
        let mut state = empty_state();
        let commitment = hash(b"dup commit");
        let nonce = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: nonce };

        apply_transaction(&mut state, &tx.clone()).unwrap();
        assert!(apply_transaction(&mut state, &tx).is_err());
    }

    // ── validate_transaction (read-only) ────────────────────────────────

    #[test]
    fn validate_commit_does_not_mutate() {
        let state = empty_state();
        let commitment = hash(b"validate only");
        let nonce = mine_commit_nonce(&commitment);
        let tx = Transaction::Commit { commitment, spam_nonce: nonce };
        validate_transaction(&state, &tx).unwrap();
        // State should not have the commitment
        assert!(!state.commitments.contains(&commitment));
    }
}
