use super::types::*;
use super::script;
use anyhow::{bail, Result};
use super::mmr::UtxoAccumulator;

/// Minimum number of leading zero bits required for a Commit transaction's PoW.
/// 24 bits ≈ 16M BLAKE3 hashes ≈ 15ms on modern hardware. High enough to
/// deter spam (~65 commits/second per core) while remaining instant for
/// legitimate users who submit one commit at a time.
#[cfg(not(feature = "fast-mining"))]
pub const MIN_COMMIT_POW_BITS: u32 = 24;
#[cfg(feature = "fast-mining")]
pub const MIN_COMMIT_POW_BITS: u32 = 16;

pub fn commit_pow_hash(commitment: &[u8; 32], actual_nonce: u32, target_height: u32) -> [u8; 32] {
    let mut data = Vec::with_capacity(40);
    data.extend_from_slice(&target_height.to_le_bytes());
    data.extend_from_slice(commitment);
    data.extend_from_slice(&actual_nonce.to_le_bytes());
    super::types::hash(&data)
}

pub fn pack_spam_nonce(actual_nonce: u32, target_height: u32) -> u64 {
    ((target_height as u64) << 32) | (actual_nonce as u64)
}

pub fn unpack_spam_nonce(spam_nonce: u64) -> (u32, u32) {
    let target_height = (spam_nonce >> 32) as u32;
    let actual_nonce = (spam_nonce & 0xFFFFFFFF) as u32;
    (target_height, actual_nonce)
}

pub fn mine_pow(commitment: &[u8; 32], required_pow: u32, current_height: u64, header_hash: [u8; 32]) -> u64 {
    let mut n = 0u32;
    
    if current_height >= crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
        let anchor_height = current_height.saturating_sub(1) as u32;
        loop {
            let mut data = Vec::with_capacity(68);
            data.extend_from_slice(&header_hash);
            data.extend_from_slice(commitment);
            data.extend_from_slice(&n.to_le_bytes());
            let h = crate::core::types::hash(&data);
            
            if crate::core::types::count_leading_zeros(&h) >= required_pow {
                return pack_spam_nonce(n, anchor_height);
            }
            n += 1;
        }
    } else {
        let target_height = current_height as u32;
        loop {
            let h = commit_pow_hash(commitment, n, target_height);
            if crate::core::types::count_leading_zeros(&h) >= required_pow {
                return pack_spam_nonce(n, target_height);
            }
            n += 1;
        }
    }
}

/// Evaluate the Commit-PoW associated with a commitment under either V1
/// (legacy) or V2 (single-hash with explicit target height) rules.
///
/// # Reasoning
/// Prior to the V2 activation height, the network must accept both V1 and V2
/// PoW formats to allow a seamless transition for upgraded wallets. The V1 format 
/// was a brute-forced 48-byte hash: 
/// `hash(height_u64 || commitment || spam_nonce_u64)`. The V2 format is a 
/// height-bound, O(1) validated 40-byte hash: 
/// `hash(target_height_u32 || commitment || actual_nonce_u32)`.
///
/// # Formal Specification
/// 
/// ```text
/// Let ℂ be the 32-byte commitment space.
/// Let ℕ₆₄ be the 64-bit integer space.
/// Let ℕ₃₂ be the 32-bit integer space.
/// Let 𝒵 : ℂ → ℕ be the leading zero counting function.
/// Let ℋ : seq 𝔹 → ℂ be the BLAKE3 hash function.
///
/// Inputs:
///   c?     ∈ ℂ           (commitment)
///   n?     ∈ ℕ₆₄         (spam_nonce)
///   h_cur? ∈ ℕ₆₄         (current_height)
///
/// Definitions:
///   (t, a) ≜ unpack(n?)  where t ∈ ℕ₃₂, a ∈ ℕ₃₂
///   H_v1(h) ≜ ℋ(h_u64 ⌢ c? ⌢ n?)
///   H_v2    ≜ ℋ(t ⌢ c? ⌢ a)
///
/// Preconditions:
///   h_cur? < V2_ACTIVATION_HEIGHT
///
/// Postconditions:
///   result! = Ok(z) ⇔ 
///       ( z = 𝒵(H_v2) ∧ z ≥ MIN_COMMIT_POW_BITS ∧ (h_cur? - t ≤ WINDOW) ∧ (t ≤ h_cur? + 1) )
///     ∨ ( ∃ h ∈ [h_cur? - WINDOW, h_cur?] : z = 𝒵(H_v1(h)) ∧ z ≥ MIN_COMMIT_POW_BITS )
///   
///   result! = Err ⇔ ¬(above)
/// ```
pub fn evaluate_commit_pow(commitment: &[u8; 32], spam_nonce: u64, state: &crate::core::State) -> Result<u32> {
    let (target_height, actual_nonce) = unpack_spam_nonce(spam_nonce);
    let current_height = state.height;

    // V3 HARD FORK: Fix Time-Travel Hashpower Exploit
    if current_height >= crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
        if target_height == 0 {
            bail!("Legacy V1/V2 Commits are deprecated. Upgrade your node/wallet.");
        }
        if current_height.saturating_sub(target_height as u64) > crate::core::types::COMMIT_POW_WINDOW {
            bail!("Commit PoW expired. Anchored to height {}, current is {}", target_height, current_height);
        }
        if (target_height as u64) >= current_height {
            bail!("Commit PoW anchor height must be a past block");
        }

        // O(1) Validation: Fetch the unpredictable block hash at `target_height`
        let mmr_pos = crate::core::mmr::mmr_size(target_height as u64);
        let anchor_hash = state.chain_mmr.get(mmr_pos).ok_or_else(|| {
            anyhow::anyhow!("Anchor height {} block hash not found in chain MMR", target_height)
        })?;

        // V3 Hash: anchor_hash || commitment || actual_nonce
        let mut data = Vec::with_capacity(68);
        data.extend_from_slice(anchor_hash);
        data.extend_from_slice(commitment);
        data.extend_from_slice(&actual_nonce.to_le_bytes());
        let hash = crate::core::types::hash(&data);

        let zeros = crate::core::types::count_leading_zeros(&hash);
        if zeros < MIN_COMMIT_POW_BITS {
            bail!("Insufficient Commit PoW");
        }
        return Ok(zeros);
    }

    // V2 Path
    if current_height >= crate::core::types::V2_ACTIVATION_HEIGHT {
        if target_height == 0 {
            bail!("Legacy V1 Commits are deprecated. Upgrade your node/wallet.");
        }
        if current_height.saturating_sub(target_height as u64) > crate::core::types::COMMIT_POW_WINDOW {
            bail!("Commit PoW expired. Mined for height {}, current is {}", target_height, current_height);
        }
        if (target_height as u64) > current_height + 1 {
            bail!("Commit PoW target height too far in the future");
        }

        let hash = commit_pow_hash(commitment, actual_nonce, target_height);
        let zeros = crate::core::types::count_leading_zeros(&hash);
        if zeros < MIN_COMMIT_POW_BITS {
            bail!("Insufficient Commit PoW");
        }
        return Ok(zeros);
    }

    // PRE-ACTIVATION: Accept both true V1 and V2 formats
    if current_height.saturating_sub(target_height as u64) <= crate::core::types::COMMIT_POW_WINDOW 
        && (target_height as u64) <= current_height + 1 
    {
        let hash_v2 = commit_pow_hash(commitment, actual_nonce, target_height);
        let zeros_v2 = crate::core::types::count_leading_zeros(&hash_v2);
        if zeros_v2 >= MIN_COMMIT_POW_BITS {
            return Ok(zeros_v2);
        }
    }

    let start = current_height.saturating_sub(crate::core::types::COMMIT_POW_WINDOW);
    let mut best_zeros = 0;
    for h in start..=current_height {
        let mut data = Vec::with_capacity(48);
        data.extend_from_slice(&h.to_le_bytes());
        data.extend_from_slice(commitment);
        data.extend_from_slice(&spam_nonce.to_le_bytes());
        let hash_v1 = super::types::hash(&data);

        let zeros_v1 = crate::core::types::count_leading_zeros(&hash_v1);
        if zeros_v1 > best_zeros {
            best_zeros = zeros_v1;
            if best_zeros >= MIN_COMMIT_POW_BITS { break; }
        }
    }

    if best_zeros < MIN_COMMIT_POW_BITS {
        bail!("Insufficient Commit PoW or expired");
    }
    Ok(best_zeros)
}

fn validate_commit_pow(commitment: &[u8; 32], nonce: u64, state: &crate::core::State) -> Result<()> {
    evaluate_commit_pow(commitment, nonce, state)?;
    Ok(())
}

/// Pure signature verification only — no state reads or mutations.
/// Called in parallel across all transactions in a batch before sequential apply.
/// Performs an O(1) read-only check against the commitment set to prevent
/// CPU exhaustion from stateless signature spam.
pub fn verify_transaction_sigs(
    tx: &Transaction,
    height: u64,
    commitments: &UtxoAccumulator,
    in_block_commits: &std::collections::HashSet<[u8; 32]>,
) -> Result<()> {
    match tx {
        Transaction::Commit { .. } => Ok(()),
        Transaction::Reveal { inputs, witnesses, outputs, salt, .. } => {
            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter()
                .map(|o| o.hash_for_commitment())
                .collect();
            let commitment = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !commitments.contains(&commitment) && !in_block_commits.contains(&commitment) {
                bail!("Phase 1 validation failed: Reveal transaction references an unknown commitment");
            }

            for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
                if !verify_predicate(&input.predicate, witness, &commitment, height, outputs, input.value, input.commitment) {
                    bail!("Predicate execution failed for input {}", i);
                }
            }
            Ok(())
        }
        Transaction::Consolidate { inputs, witness, outputs, salt, .. } => {
            if inputs.is_empty() { bail!("Phase 1 validation failed: Empty inputs"); }
            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter()
                .map(|o| o.hash_for_commitment())
                .collect();
            let commitment = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !commitments.contains(&commitment) && !in_block_commits.contains(&commitment) {
                bail!("Phase 1 validation failed: Consolidate transaction references an unknown commitment");
            }

            let first_addr = inputs[0].predicate.address();
            for input in inputs.iter().skip(1) {
                if input.predicate.address() != first_addr {
                    bail!("Phase 1 validation failed: All inputs in a Consolidate transaction must share the same predicate address");
                }
            }

            if !verify_predicate(&inputs[0].predicate, witness, &commitment, height, outputs, inputs[0].value, inputs[0].commitment) {
                bail!("Predicate execution failed for Consolidate witness");
            }
            Ok(())
        }
    }
}

/// Apply a transaction that has already passed signature verification.
/// Skips the verify_predicate call — all other validation still runs.
///
/// All accumulator mutations use `is_v2 = is_v2_at(state.height)` (captured
/// once at function entry) so that within a single block every UTXO/SMT
/// update lives in the same hashing universe.
pub fn apply_transaction_no_sig_check(state: &mut State, tx: &Transaction) -> Result<()> {
    let v2 = crate::core::types::is_v2_at(state.height);

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
            if witnesses.len() != inputs.len() {
                bail!("Witness count must match input count");
            }

            let max_witness_size = MAX_SIGNATURE_SIZE * MAX_TX_INPUTS;
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }

            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                    if input.commitment.is_some() {
                        if input.value != 0 {
                            bail!("State Thread inputs must have a value of exactly 0");
                        }
                    }
                }
            }

            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 {
                    if out.is_confidential() {
                        // Allowed: 0-value State Thread
                    } else {
                        bail!("Zero-value output {}", i);
                    }
                } else if !out.value().is_power_of_two() {
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

            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found (expected {})", hex::encode(expected));
            }
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > COMMITMENT_TTL {
                    bail!(
                        "Commitment expired at consensus (committed at height {}, current {}, TTL {})",
                        commit_height, state.height, COMMITMENT_TTL
                    );
                }
            }
            // PREVENT COMMIT REPLAY ATTACK: Do not delete commitments after activation height.
            // Let the deterministic GC sweep them when their PoW naturally expires.
            if state.height < crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
                state.commitments.remove(&expected, v2);
                state.commitment_heights.remove(&expected);
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
                state.coins.remove(coin_id, v2);
            }

            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id, v2) {
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
        Transaction::Consolidate { inputs, witness: _, outputs, salt, .. } => {
            if inputs.len() < 2 { bail!("Consolidate transactions must spend at least two coins"); }
            if outputs.is_empty() { bail!("Transaction must create at least one new coin"); }
            if inputs.len() > crate::core::types::MAX_CONSOLIDATE_INPUTS { bail!("Too many inputs (max {})", crate::core::types::MAX_CONSOLIDATE_INPUTS); }
            if outputs.len() > MAX_TX_OUTPUTS { bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); }

            let first_addr = inputs[0].predicate.address();
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if input.predicate.address() != first_addr {
                        bail!("Consolidate inputs must share the same address");
                    }
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                    if input.commitment.is_some() {
                        if input.value != 0 {
                            bail!("State Thread inputs must have a value of exactly 0");
                        }
                    }
                }
            }

            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 && !out.is_confidential() {
                    bail!("Zero-value output {}", i);
                } else if out.value() != 0 && !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE {
                        bail!("DataBurn payload exceeds max size");
                    }
                }
            }

            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value())).ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum {
                bail!("Input value ({}) must exceed output value ({}) to pay fee", in_sum, out_sum);
            }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found (expected {})", hex::encode(expected));
            }
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > crate::core::COMMITMENT_TTL {
                    bail!("Commitment expired at consensus");
                }
            }
            // PREVENT COMMIT REPLAY ATTACK: Do not delete commitments after activation height.
            // Let the deterministic GC sweep them when their PoW naturally expires.
            if state.height < crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
                state.commitments.remove(&expected, v2);
                state.commitment_heights.remove(&expected);
            }
            for input in inputs.iter() {
                if !state.coins.contains(&input.coin_id()) {
                    bail!("Coin {} not found or already spent", hex::encode(input.coin_id()));
                }
            }
            for coin_id in &input_coin_ids {
                state.coins.remove(coin_id, v2);
            }
            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id, v2) {
                        bail!("Duplicate coin created");
                    }
                }
            }
            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids { hasher.update(coin_id); }
                for hash in &output_commit_hashes { hasher.update(hash); }
                hasher.update(salt);
                state.midstate = hash_concat(&state.midstate, hasher.finalize().as_bytes());
            }

            Ok(())
        }
    }
}

/// Apply a transaction to the state, including signature verification.
///
/// All accumulator mutations use `is_v2 = is_v2_at(state.height)` (captured
/// once at function entry) so that within a single block every UTXO/SMT
/// update lives in the same hashing universe.
pub fn apply_transaction(state: &mut State, tx: &Transaction) -> Result<()> {
    let v2 = crate::core::types::is_v2_at(state.height);

    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce, state)?;
            if !state.commitments.insert(*commitment, v2) {
                bail!("Duplicate commitment");
            }
            state.commitment_heights.insert(*commitment, state.height);

            // O(log H) expiration tracking
            let mut list = state.expirations.get(&state.height).cloned().unwrap_or_default();
            list.push(*commitment);
            state.expirations.insert(state.height, list);

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

            let max_witness_size = MAX_SIGNATURE_SIZE * MAX_TX_INPUTS;
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }

            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                    if input.commitment.is_some() {
                        if input.value != 0 {
                            bail!("State Thread inputs must have a value of exactly 0");
                        }
                    }
                }
            }
            // 1. Validate all output values are power of 2 and nonzero
            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 {
                    if out.is_confidential()  {
                        // Allowed: 0-value State Thread
                    } else {
                        bail!("Zero-value output {}", i);
                    }
                } else if !out.value().is_power_of_two() {
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

            // 4. Verify commitment exists, is not expired, and matches
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);
            if !state.commitments.contains(&expected) {
                bail!(
                    "No matching commitment found (expected {})",
                    hex::encode(expected)
                );
            }
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > COMMITMENT_TTL {
                    bail!(
                        "Commitment expired at consensus (committed at height {}, current {}, TTL {})",
                        commit_height, state.height, COMMITMENT_TTL
                    );
                }
            }
            // PREVENT COMMIT REPLAY ATTACK: Do not delete commitments after activation height.
            // Let the deterministic GC sweep them when their PoW naturally expires.
            if state.height < crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
                state.commitments.remove(&expected, v2);
                state.commitment_heights.remove(&expected);
            }

            // 5. Verify each input coin exists and executes cleanly against its Predicate
            for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }

                // Script Execution Engine
                if !verify_predicate(&input.predicate, witness, &expected, state.height, outputs, input.value, input.commitment) {
                    bail!("Predicate execution failed for input {}", i);
                }
            }

            // 6. Remove spent coins
            for coin_id in &input_coin_ids {
                state.coins.remove(coin_id, v2);
            }

            // 7. Add new coins (Ignore DataBurns, protecting the SMT!)
            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id, v2) {
                        bail!("Duplicate coin created");
                    }
                }
            }

            // 8. Update midstate (SegWit: witnesses are explicitly excluded.
            //    The Commit phase already binds inputs, outputs, and salt via
            //    compute_commitment, so the transaction cannot be malleated.
            //    Excluding witnesses keeps apply_transaction in sync with
            //    apply_transaction_no_sig_check and Batch::header().)
            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids { hasher.update(coin_id); }
                for hash in &output_commit_hashes { hasher.update(hash); }
                hasher.update(salt);
                let tx_hash = *hasher.finalize().as_bytes();
                state.midstate = hash_concat(&state.midstate, &tx_hash);
            }

            Ok(())
        }
        Transaction::Consolidate { inputs, witness, outputs, salt, .. } => {
            if inputs.len() < 2 { bail!("Consolidate transactions must spend at least two coins"); }

            if outputs.is_empty() { bail!("Transaction must create at least one new coin"); }
            if inputs.len() > crate::core::types::MAX_CONSOLIDATE_INPUTS { bail!("Too many inputs (max {})", crate::core::types::MAX_CONSOLIDATE_INPUTS); }
            if outputs.len() > MAX_TX_OUTPUTS { bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); }

            let first_addr = inputs[0].predicate.address();
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if input.predicate.address() != first_addr {
                        bail!("Consolidate inputs must share the same address");
                    }
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                    if input.commitment.is_some() && input.value != 0 {
                        bail!("State Thread inputs must have a value of exactly 0");
                    }
                }
            }

            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 && !out.is_confidential() {
                    bail!("Zero-value output {}", i);
                } else if out.value() != 0 && !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE {
                        bail!("DataBurn payload exceeds max size");
                    }
                }
            }

            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value())).ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum {
                bail!("Input value ({}) must exceed output value ({}) to pay fee", in_sum, out_sum);
            }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !state.commitments.contains(&expected) {
                bail!("No matching commitment found (expected {})", hex::encode(expected));
            }
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > COMMITMENT_TTL {
                    bail!("Commitment expired at consensus");
                }
            }
            // PREVENT COMMIT REPLAY ATTACK: Do not delete commitments after activation height.
            // Let the deterministic GC sweep them when their PoW naturally expires.
            if state.height < crate::core::types::COMMIT_REPLAY_FIX_ACTIVATION_HEIGHT {
                state.commitments.remove(&expected, v2);
                state.commitment_heights.remove(&expected);
            }

            for input in inputs.iter() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found or already spent", hex::encode(coin_id));
                }
            }

            if !verify_predicate(&inputs[0].predicate, witness, &expected, state.height, outputs, inputs[0].value, inputs[0].commitment) {
                bail!("Predicate execution failed for Consolidate witness");
            }

            for coin_id in &input_coin_ids { state.coins.remove(coin_id, v2); }
            for out in outputs {
                if let Some(coin_id) = out.coin_id() {
                    if !state.coins.insert(coin_id, v2) { bail!("Duplicate coin created"); }
                }
            }
            {
                let mut hasher = blake3::Hasher::new();
                for coin_id in &input_coin_ids { hasher.update(coin_id); }
                for hash in &output_commit_hashes { hasher.update(hash); }
                hasher.update(salt);
                state.midstate = hash_concat(&state.midstate, hasher.finalize().as_bytes());
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
    input_value: u64,
    input_state: Option<[u8; 32]>,
) -> bool {
    match (predicate, witness) {
        (Predicate::Script { bytecode }, Witness::ScriptInputs(inputs)) => {
            let this_address = predicate.address();
            let ctx = script::ExecContext {
                commitment,
                height: current_height,
                outputs,
                input_value,
                input_state,
                this_address,
            };
            script::execute_script(bytecode, inputs, &ctx).is_ok()
        }
    }
}

/// Validate a transaction without applying it (read-only consensus check).
pub fn validate_transaction(state: &State, tx: &Transaction) -> Result<()> {
    match tx {
        Transaction::Commit { commitment, spam_nonce } => {
            validate_commit_pow(commitment, *spam_nonce, state)?;
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

            let max_witness_size = MAX_SIGNATURE_SIZE * MAX_TX_INPUTS;
            if bincode::serialized_size(&witnesses).unwrap_or(u64::MAX) > max_witness_size as u64 {
                bail!("Witnesses payload exceeds maximum allowed size");
            }

            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if !seen.insert(input.coin_id()) {
                        bail!("Duplicate input coin");
                    }
                    if input.commitment.is_some() {
                        if input.value != 0 {
                            bail!("State Thread inputs must have a value of exactly 0");
                        }
                    }
                }
            }
            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 {
                    if out.is_confidential() {
                        // Allowed: 0-value State Thread
                    } else {
                        bail!("Zero-value output {}", i);
                    }
                } else if !out.value().is_power_of_two() {
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

            // Check commitment hasn't expired
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > COMMITMENT_TTL {
                    bail!("Commitment expired (committed at height {}, current {})", commit_height, state.height);
                }
            }
            // 5. Verify each Witness executes cleanly against its Predicate
            for (i, (input, witness)) in inputs.iter().zip(witnesses.iter()).enumerate() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) {
                    bail!("Coin {} not found", hex::encode(coin_id));
                }
                if !verify_predicate(&input.predicate, witness, &expected, state.height, outputs, input.value, input.commitment) {
                    bail!("Predicate execution failed for input {}", i);
                }
            }

            Ok(())
        }
        Transaction::Consolidate { inputs, witness, outputs, salt, .. } => {
            if inputs.is_empty() { bail!("Must spend at least one coin"); }
            if outputs.is_empty() { bail!("Must create at least one coin"); }
            if inputs.len() > crate::core::types::MAX_CONSOLIDATE_INPUTS { bail!("Too many inputs (max {})", crate::core::types::MAX_CONSOLIDATE_INPUTS); }
            if outputs.len() > MAX_TX_OUTPUTS { bail!("Too many outputs (max {})", MAX_TX_OUTPUTS); }

            let first_addr = inputs[0].predicate.address();
            {
                let mut seen = std::collections::HashSet::new();
                for input in inputs {
                    if input.predicate.address() != first_addr { bail!("Consolidate inputs must share the same address"); }
                    if !seen.insert(input.coin_id()) { bail!("Duplicate input coin"); }
                    if input.commitment.is_some() && input.value != 0 { bail!("State Thread inputs must have a value of exactly 0"); }
                }
            }
            for (i, out) in outputs.iter().enumerate() {
                if out.value() == 0 && !out.is_confidential() {
                    bail!("Zero-value output {}", i);
                } else if out.value() != 0 && !out.value().is_power_of_two() {
                    bail!("Invalid denomination: output {} value {} is not a power of 2", i, out.value());
                }
                if let OutputData::DataBurn { payload, .. } = out {
                    if payload.len() > crate::core::types::MAX_BURN_DATA_SIZE { bail!("DataBurn payload exceeds max size"); }
                }
            }

            let in_sum = inputs.iter().try_fold(0u64, |acc, i| acc.checked_add(i.value)).ok_or_else(|| anyhow::anyhow!("Input value overflow"))?;
            let out_sum = outputs.iter().try_fold(0u64, |acc, o| acc.checked_add(o.value())).ok_or_else(|| anyhow::anyhow!("Output value overflow"))?;
            if in_sum <= out_sum { bail!("Input value must exceed output value"); }

            let input_coin_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
            let output_commit_hashes: Vec<[u8; 32]> = outputs.iter().map(|o| o.hash_for_commitment()).collect();
            let expected = compute_commitment(&input_coin_ids, &output_commit_hashes, salt);

            if !state.commitments.contains(&expected) { bail!("No matching commitment found"); }
            if let Some(&commit_height) = state.commitment_heights.get(&expected) {
                if state.height.saturating_sub(commit_height) > COMMITMENT_TTL {
                    bail!("Commitment expired (committed at height {}, current {})", commit_height, state.height);
                }
            }

            for input in inputs.iter() {
                let coin_id = input.coin_id();
                if !state.coins.contains(&coin_id) { bail!("Coin {} not found", hex::encode(coin_id)); }
            }
            if !verify_predicate(&inputs[0].predicate, witness, &expected, state.height, outputs, inputs[0].value, inputs[0].commitment) {
                bail!("Predicate execution failed for Consolidate witness");
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
            expirations: im::OrdMap::new(),
            chain_mmr: crate::core::mmr::MerkleMountainRange::new(),
            header_hash: [0u8; 32],
        }
    }

    fn mine_commit_nonce(commitment: &[u8; 32]) -> u64 {
        let mut n = 0u64;
        loop {
            let h = hash_concat(commitment, &n.to_le_bytes());
            if count_leading_zeros(&h) >= MIN_COMMIT_POW_BITS {
                return n;
            }
            n += 1;
        }
    }

    #[test]
    fn commit_pow_valid_nonce_passes() {
        let commitment = hash(b"test commitment");
        let nonce = mine_commit_nonce(&commitment);
        let state = empty_state();
        // Tests use V1 path (height 0), where target_height in nonce is unused.
        assert!(validate_commit_pow(&commitment, nonce, &state).is_ok());
    }

    #[test]
    fn commit_pow_invalid_nonce_fails() {
        let commitment = hash(b"test commitment");
        let state = empty_state();
        // Find a nonce that does NOT meet the PoW threshold
        let mut bad_nonce = 0u64;
        loop {
            let h = hash_concat(&commitment, &bad_nonce.to_le_bytes());
            if count_leading_zeros(&h) < MIN_COMMIT_POW_BITS {
                break;
            }
            bad_nonce += 1;
        }
        assert!(validate_commit_pow(&commitment, bad_nonce, &state).is_err());
    }

    #[test]
    fn commit_pow_benchmark() {
        // Measure time to mine a valid nonce — should be ~10-50ms
        let commitment = hash(b"benchmark commitment");
        let start = std::time::Instant::now();
        let nonce = mine_commit_nonce(&commitment);
        let elapsed = start.elapsed();
        let state = empty_state();
        assert!(validate_commit_pow(&commitment, nonce, &state).is_ok());
        eprintln!("Commit PoW mining took {:?} (nonce: {})", elapsed, nonce);
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
            if count_leading_zeros(&h) < MIN_COMMIT_POW_BITS {
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
        let v2 = crate::core::types::is_v2_at(state.height);
        let seed: [u8; 32] = [0x42; 32];
        let owner_pk = wots::keygen(&seed);
        let address = compute_address(&owner_pk);
        let salt = hash(b"test salt");
        let coin_id = compute_coin_id(&address, value, &salt);
        state.coins.insert(coin_id, v2);
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
                salt: input_salt,
                commitment: None,
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 8, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
        let same_input = InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None };
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&owner_pk), value: 16, salt: input_salt , commitment: None }],
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
        let v2 = crate::core::types::is_v2_at(state.height);
        let mss_seed = hash(b"mss test seed");
        let mut keypair = mss::keygen(&mss_seed, 4).unwrap();
        let master_pk = keypair.public_key(); // this is the owner_pk

        // The address = hash(master_pk)
        let address = compute_address(&master_pk);
        let input_salt = hash(b"mss coin salt");
        let value = 16u64;
        let coin_id = compute_coin_id(&address, value, &input_salt);
        state.coins.insert(coin_id, v2);

        let output = OutputData::Standard { address: hash(b"dest"), value: 8, salt: [0; 32] };
        let commit_salt = [0xCC; 32];
        let commitment = do_commit(&mut state, &[coin_id], &[output.coin_id().unwrap()], &commit_salt);

        // Sign with MSS
        let mss_sig = keypair.sign(&commitment).unwrap();
        let sig_bytes = mss_sig.to_bytes();

        let tx = Transaction::Reveal {
            inputs: vec![InputReveal { predicate: Predicate::p2pk(&master_pk), value, salt: input_salt , commitment: None }],
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
