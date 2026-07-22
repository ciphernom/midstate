//! Send planning: fee estimation, output construction, address codec.
//!
//! The fee loop is a faithful port of `wallet_send` in the upstream CLI
//! (main.rs): estimate serialized size from input/output counts, apply the
//! mempool floor of 10 units per 1024 bytes (+10 padding), and re-select
//! coins until the selection covers amount + the fee its own size implies.

use anyhow::{anyhow, bail, Result};
use midstate::core::{compute_address, decompose_value, wots, OutputData};
use midstate::wallet::Wallet;

/// Upstream size model (main.rs): 1536-byte WOTS sig + ~100 bytes per input,
/// ~100 bytes per output, ~100 bytes base.
const BYTES_PER_INPUT: u64 = 1636;
const BYTES_PER_OUTPUT: u64 = 100;
const BYTES_BASE: u64 = 100;
/// Mempool fee floor: 10 units per KiB, +10 units safety padding.
fn required_fee(inputs: usize, outputs: usize) -> u64 {
    let est = BYTES_BASE + inputs as u64 * BYTES_PER_INPUT + outputs as u64 * BYTES_PER_OUTPUT;
    (est * 10) / 1024 + 10
}

/// Decode a 72-char checksummed hex address (32-byte address ‖ 4-byte
/// BLAKE3 checksum) — inverse of `core::encode_address_with_checksum`.
pub fn decode_address(s: &str) -> Result<[u8; 32]> {
    let s = s.trim();
    let bytes = hex::decode(s).map_err(|_| anyhow!("address is not valid hex"))?;
    if bytes.len() != 36 {
        bail!("address must be 72 hex characters (got {})", s.len());
    }
    let mut addr = [0u8; 32];
    addr.copy_from_slice(&bytes[..32]);
    let expect = midstate::core::hash(&addr);
    if expect[..4] != bytes[32..36] {
        bail!("address checksum does not match — check for typos");
    }
    Ok(addr)
}

/// A fully planned (but unsigned, uncommitted) send.
pub struct SendPlan {
    pub input_coin_ids: Vec<[u8; 32]>,
    pub outputs: Vec<OutputData>,
    /// (output_index, wots_seed) for change outputs we control.
    pub change_seeds: Vec<(usize, [u8; 32])>,
    pub in_sum: u64,
    pub fee: u64,
    pub amount: u64,
}

/// Build a send plan against the wallet's live coins.
///
/// Mutates the wallet only through `allocate_next_wots_seed` (change keys),
/// which is monotone and persisted by the caller's save — an abandoned plan
/// wastes indices but never reuses them (upstream invariant).
pub fn plan_send(
    wallet: &mut Wallet,
    live_coins: &[[u8; 32]],
    dest: [u8; 32],
    amount: u64,
) -> Result<SendPlan> {
    if amount == 0 {
        bail!("amount must be greater than zero");
    }

    let mut target_fee = 100u64; // conservative starting guess, as upstream

    loop {
        let needed = amount + target_fee;
        let selected = wallet.select_coins(needed, live_coins)?;

        let in_sum: u64 = selected
            .iter()
            .filter_map(|id| wallet.find_coin(id))
            .map(|c| c.value)
            .sum();
        if in_sum <= amount {
            bail!(
                "input value ({in_sum}) must exceed send amount ({amount}) to pay the fee"
            );
        }

        // Output count at this fee guess: recipient denoms + change denoms.
        let change_guess = in_sum.saturating_sub(amount + target_fee);
        let num_outputs = decompose_value(amount).len() + decompose_value(change_guess).len();

        let fee = required_fee(selected.len(), num_outputs);

        if in_sum >= amount + fee {
            // Locked in. Build the real outputs.
            let change = in_sum - amount - fee;

            let mut outputs: Vec<OutputData> = Vec::new();
            let mut change_seeds: Vec<(usize, [u8; 32])> = Vec::new();

            for denom in decompose_value(amount) {
                let salt: [u8; 32] = rand::random();
                outputs.push(OutputData::Standard { address: dest, value: denom, salt });
            }
            if change > 0 {
                for denom in decompose_value(change) {
                    let seed = wallet.allocate_next_wots_seed()?;
                    let pk = wots::keygen(&seed);
                    let addr = compute_address(&pk);
                    let salt: [u8; 32] = rand::random();
                    change_seeds.push((outputs.len(), seed));
                    outputs.push(OutputData::Standard { address: addr, value: denom, salt });
                }
            }

            // Shuffle so recipient vs change position leaks nothing (upstream).
            {
                use rand::seq::SliceRandom;
                let mut idx: Vec<usize> = (0..outputs.len()).collect();
                idx.shuffle(&mut rand::thread_rng());
                let shuffled: Vec<OutputData> =
                    idx.iter().map(|&i| outputs[i].clone()).collect();
                let mut rev = vec![0usize; idx.len()];
                for (new_i, &old_i) in idx.iter().enumerate() {
                    rev[old_i] = new_i;
                }
                change_seeds = change_seeds
                    .into_iter()
                    .map(|(old, s)| (rev[old], s))
                    .collect();
                outputs = shuffled;
            }

            return Ok(SendPlan {
                input_coin_ids: selected,
                outputs,
                change_seeds,
                in_sum,
                fee,
                amount,
            });
        }

        // Selection can't cover its own fee — raise the target and re-select.
        target_fee = fee;
    }
}


/// Like [`plan_send`], but the recipient outputs are supplied verbatim (used
/// for channel funding, where the caller must know every output's salt).
/// Change handling, the fee loop, and the shuffle are identical.
pub fn plan_fixed_outputs(
    wallet: &mut Wallet,
    live_coins: &[[u8; 32]],
    recipient: Vec<OutputData>,
) -> Result<SendPlan> {
    let amount: u64 = recipient
        .iter()
        .map(|o| match o {
            OutputData::Standard { value, .. } => *value,
            _ => 0,
        })
        .sum();
    if amount == 0 {
        bail!("amount must be greater than zero");
    }

    let mut target_fee = 100u64;
    loop {
        let needed = amount + target_fee;
        let selected = wallet.select_coins(needed, live_coins)?;
        let in_sum: u64 = selected
            .iter()
            .filter_map(|id| wallet.find_coin(id))
            .map(|c| c.value)
            .sum();
        if in_sum <= amount {
            bail!("input value ({in_sum}) must exceed the amount ({amount}) to pay the fee");
        }
        let change_guess = in_sum.saturating_sub(amount + target_fee);
        let num_outputs = recipient.len() + decompose_value(change_guess).len();
        let fee = required_fee(selected.len(), num_outputs);

        if in_sum >= amount + fee {
            let change = in_sum - amount - fee;
            let mut outputs: Vec<OutputData> = recipient.clone();
            let mut change_seeds: Vec<(usize, [u8; 32])> = Vec::new();
            if change > 0 {
                for denom in decompose_value(change) {
                    let seed = wallet.allocate_next_wots_seed()?;
                    let pk = wots::keygen(&seed);
                    let addr = compute_address(&pk);
                    let salt: [u8; 32] = rand::random();
                    change_seeds.push((outputs.len(), seed));
                    outputs.push(OutputData::Standard { address: addr, value: denom, salt });
                }
            }
            {
                use rand::seq::SliceRandom;
                let mut idx: Vec<usize> = (0..outputs.len()).collect();
                idx.shuffle(&mut rand::thread_rng());
                let shuffled: Vec<OutputData> = idx.iter().map(|&i| outputs[i].clone()).collect();
                let mut rev = vec![0usize; idx.len()];
                for (new_i, &old_i) in idx.iter().enumerate() {
                    rev[old_i] = new_i;
                }
                change_seeds = change_seeds.into_iter().map(|(old, s)| (rev[old], s)).collect();
                outputs = shuffled;
            }
            return Ok(SendPlan { input_coin_ids: selected, outputs, change_seeds, in_sum, fee, amount });
        }
        target_fee = fee;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_roundtrip() {
        let addr = [7u8; 32];
        let enc = midstate::core::encode_address_with_checksum(&addr);
        assert_eq!(enc.len(), 72);
        assert_eq!(decode_address(&enc).unwrap(), addr);
    }

    #[test]
    fn checksum_rejects_typo() {
        let addr = [7u8; 32];
        let mut enc = midstate::core::encode_address_with_checksum(&addr);
        // flip one nibble in the body
        let flip = if &enc[10..11] == "0" { "1" } else { "0" };
        enc.replace_range(10..11, flip);
        assert!(decode_address(&enc).is_err());
    }

    #[test]
    fn fee_floor_matches_upstream_model() {
        // 1 input, 2 outputs: 100 + 1636 + 200 = 1936 bytes → 1936*10/1024+10 = 28
        assert_eq!(required_fee(1, 2), 28);
    }
}
