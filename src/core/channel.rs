//! Q-Bolt v2 payment channels — the pure, deterministic layer.
//!
//! Ported verbatim (byte-for-byte semantics) from `wasm-wallet/src/lib.rs`
//! and `worker.js` so native and web wallets interoperate on the same
//! channels: identical covenant bytecode, salt derivation, state
//! commitments, and wire codecs. Any change here is a protocol change.
//!
//! Design recap (see the wasm source header for the full model):
//! - Funding covenant: IF (receiver_sig + sender_sig ≥ 2) ELSE (expiry
//!   CHECKTIMEVERIFY + sender_sig) ENDIF — receiver closes with both sigs
//!   any time; sender refunds unilaterally after `expiry`.
//! - Only the SENDER signs balance states off-chain (MSS over the whole-tx
//!   commitment); the receiver co-signs exactly once, at close.
//! - Every close/refund reveal pays [`CLOSE_FEE`] out of channel value.
//! - Commitments are single-shot with a TTL: never commit a state you are
//!   not immediately revealing; the `attempt` counter exists so a re-signed
//!   retry yields a fresh commitment.

use super::script as sc;
use super::types::{
    compute_address, compute_coin_id, compute_commitment, decompose_value, hash, InputReveal,
    OutputData, Predicate, Witness,
};
use anyhow::{bail, Result};

/// Flat fee (base units) reserved out of channel capacity for the close /
/// refund reveal. Mirrored by `QB.CLOSE_FEE` in worker.js and
/// `QBOLT_CLOSE_FEE` in wasm-wallet.
pub const CLOSE_FEE: u64 = 2000;
/// Funding-coin ceiling per channel (wire + builder invariant).
pub const MAX_FUNDING_COINS: usize = 64;
/// HTLC ceiling per state (builder invariant; wire byte allows 12).
pub const MAX_HTLCS: usize = 12;
/// Flat fee a hub keeps per forward. In a unidirectional (Spilman) channel a
/// forward permanently CONSUMES the hub's outbound capacity toward that peer —
/// it is never replenished by return traffic — so this compensates consumed
/// capacity plus the two one-time signatures the hop burns, not "rent".
pub const HOP_FEE: u64 = 50;
/// An HTLC must not expire sooner than `now + HTLC_MIN_HEADROOM` (leaves room
/// to force-close and sweep on-chain).
pub const HTLC_MIN_HEADROOM: u64 = 60;
/// Each hop shortens the timeout by this much so an upstream hub can always
/// claim before its own downstream deadline passes.
pub const HTLC_HOP_DELTA: u64 = 30;
/// An HTLC may outlive its channel's expiry by at most this much.
pub const HTLC_MAX_PAST_EXPIRY: u64 = 2880;

/// Why an HTLC was refused. Wire-compatible with worker.js fail codes.
pub mod fail {
    pub const UNKNOWN_CHANNEL: u8 = 1;
    pub const UNDERPAID: u8 = 2;
    pub const FEE_EXCEEDS_AMOUNT: u8 = 3;
    pub const NO_ROUTE: u8 = 4;
    pub const FORWARD_FAILED: u8 = 5;
    pub const TIMEOUT_TOO_TIGHT: u8 = 6;
    pub const DOWNSTREAM_FAILED: u8 = 7;

    pub fn describe(code: u8) -> &'static str {
        match code {
            UNKNOWN_CHANNEL => "the peer does not know this channel",
            UNDERPAID => "the amount was below the invoice",
            FEE_EXCEEDS_AMOUNT => "the routing fee exceeded the amount",
            NO_ROUTE => "no onward channel could reach the destination",
            FORWARD_FAILED => "the onward hop could not be set up",
            TIMEOUT_TOO_TIGHT => "not enough time left to route safely",
            DOWNSTREAM_FAILED => "a later hop refused the payment",
            _ => "refused",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FundingCoin {
    pub value: u64,
    pub salt: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Htlc {
    pub amount: u64,
    pub timeout: u64,
    pub secret_hash: [u8; 32],
}

/// One HTLC output of a built state, with everything needed to sweep it.
#[derive(Clone, Debug)]
pub struct HtlcCoin {
    pub coin_id: [u8; 32],
    pub address: [u8; 32],
    pub bytecode: Vec<u8>,
    pub value: u64,
    pub salt: [u8; 32],
    pub secret_hash: [u8; 32],
    pub timeout: u64,
}

/// A fully-derived channel state: deterministic outputs, canonical tx salt,
/// and the consensus commitment over the sorted funding inputs.
#[derive(Clone, Debug)]
pub struct BuiltState {
    pub commitment: [u8; 32],
    pub outputs: Vec<OutputData>,
    pub salt: [u8; 32],
    pub fee: u64,
    pub capacity: u64,
    pub input_coin_ids: Vec<[u8; 32]>,
    pub htlc_coins: Vec<HtlcCoin>,
    pub nonce: u32,
    pub attempt: u32,
}

/// Assemble the Q-Bolt v2 funding covenant bytecode.
pub fn covenant_bytes(sender_pk: &[u8; 32], receiver_pk: &[u8; 32], expiry: u64) -> Vec<u8> {
    let mut bc = Vec::new();
    bc.push(sc::OP_IF);
    sc::push_data(&mut bc, receiver_pk);
    bc.push(sc::OP_CHECKSIG);
    bc.push(sc::OP_SWAP);
    sc::push_data(&mut bc, sender_pk);
    bc.push(sc::OP_CHECKSIG);
    bc.push(sc::OP_ADD);
    sc::push_int(&mut bc, 2);
    bc.push(sc::OP_GREATER_OR_EQUAL);
    bc.push(sc::OP_VERIFY);
    bc.push(sc::OP_ELSE);
    sc::push_int(&mut bc, expiry);
    bc.push(sc::OP_CHECKTIMEVERIFY);
    sc::push_data(&mut bc, sender_pk);
    bc.push(sc::OP_CHECKSIGVERIFY);
    bc.push(sc::OP_ENDIF);
    sc::push_int(&mut bc, 1);
    bc
}

/// Channel address = BLAKE3(covenant) — the universal pay-to-script rule.
pub fn channel_address(sender_pk: &[u8; 32], receiver_pk: &[u8; 32], expiry: u64) -> [u8; 32] {
    hash(&covenant_bytes(sender_pk, receiver_pk, expiry))
}

/// Derive canonical funding-coin ids at `channel_addr` and return them
/// sorted ascending by coin id. BOTH the state builder and the reveal
/// builders route through this so input ordering (and therefore the
/// commitment) is identical everywhere.
pub fn sorted_funding(
    funding: &[FundingCoin],
    channel_addr: &[u8; 32],
) -> Result<Vec<([u8; 32], u64, [u8; 32])>> {
    if funding.is_empty() {
        bail!("qbolt: funding coin list is empty");
    }
    if funding.len() > MAX_FUNDING_COINS {
        bail!("qbolt: too many funding coins (max {MAX_FUNDING_COINS})");
    }
    let mut out: Vec<([u8; 32], u64, [u8; 32])> = Vec::with_capacity(funding.len());
    for c in funding {
        if c.value == 0 || (c.value & (c.value - 1)) != 0 {
            bail!("qbolt: funding coin value must be a nonzero power of 2");
        }
        out.push((compute_coin_id(channel_addr, c.value, &c.salt), c.value, c.salt));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    for w in out.windows(2) {
        if w[0].0 == w[1].0 {
            bail!("qbolt: duplicate funding coin");
        }
    }
    Ok(out)
}

/// The channel's stable identifier: the lexicographically smallest funding
/// coin id.
pub fn channel_id(funding: &[FundingCoin], channel_addr: &[u8; 32]) -> Result<[u8; 32]> {
    Ok(sorted_funding(funding, channel_addr)?[0].0)
}

fn derived_salt(channel_id: &[u8; 32], nonce: u32, tag: &[u8], i: u32, j: u32) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"qbolt_v2_out_salt");
    h.update(channel_id);
    h.update(&nonce.to_le_bytes());
    h.update(tag);
    h.update(&i.to_le_bytes());
    h.update(&j.to_le_bytes());
    *h.finalize().as_bytes()
}

/// Shared core for close-state and refund-state construction. Enforces exact
/// conservation: sender + receiver + Σhtlc = capacity − CLOSE_FEE.
#[allow(clippy::too_many_arguments)]
fn build_state_core(
    channel_id: &[u8; 32],
    sender_pk: &[u8; 32],
    receiver_pk: &[u8; 32],
    channel_addr: &[u8; 32],
    funding: &[FundingCoin],
    sender_amt: u64,
    receiver_amt: u64,
    nonce: u32,
    htlcs: &[Htlc],
    attempt: u32,
    salt_domain: &[u8],
) -> Result<BuiltState> {
    let sorted = sorted_funding(funding, channel_addr)?;
    let capacity: u64 = sorted
        .iter()
        .try_fold(0u64, |a, f| a.checked_add(f.1))
        .ok_or_else(|| anyhow::anyhow!("qbolt: capacity overflow"))?;
    if capacity <= CLOSE_FEE {
        bail!("qbolt: capacity does not cover the close fee");
    }
    if htlcs.len() > MAX_HTLCS {
        bail!("qbolt: too many HTLCs (max {MAX_HTLCS})");
    }
    let htlc_sum: u64 = htlcs
        .iter()
        .try_fold(0u64, |a, h| a.checked_add(h.amount))
        .ok_or_else(|| anyhow::anyhow!("qbolt: HTLC sum overflow"))?;

    let distributable = capacity - CLOSE_FEE;
    let spoken_for = sender_amt
        .checked_add(receiver_amt)
        .and_then(|v| v.checked_add(htlc_sum))
        .ok_or_else(|| anyhow::anyhow!("qbolt: balance overflow"))?;
    if spoken_for != distributable {
        bail!(
            "qbolt: conservation violated — sender {} + receiver {} + htlcs {} must equal capacity {} − fee {}",
            sender_amt, receiver_amt, htlc_sum, capacity, CLOSE_FEE
        );
    }

    let mut output_hashes: Vec<[u8; 32]> = Vec::new();
    let mut outputs: Vec<OutputData> = Vec::new();
    let mut htlc_coins: Vec<HtlcCoin> = Vec::new();

    // Receiver coins first, then sender coins, then HTLC coins — fixed order.
    let recv_addr = compute_address(receiver_pk);
    for (i, denom) in decompose_value(receiver_amt).into_iter().enumerate() {
        let salt = derived_salt(channel_id, nonce, b"RECV", i as u32, 0);
        output_hashes.push(compute_coin_id(&recv_addr, denom, &salt));
        outputs.push(OutputData::Standard { address: recv_addr, value: denom, salt });
    }
    let send_addr = compute_address(sender_pk);
    for (i, denom) in decompose_value(sender_amt).into_iter().enumerate() {
        let salt = derived_salt(channel_id, nonce, b"SEND", i as u32, 0);
        output_hashes.push(compute_coin_id(&send_addr, denom, &salt));
        outputs.push(OutputData::Standard { address: send_addr, value: denom, salt });
    }
    for (i, h) in htlcs.iter().enumerate() {
        if h.amount == 0 {
            bail!("qbolt: zero-value HTLC");
        }
        // HTLCs always flow sender → receiver in a unidirectional channel:
        // claim path = receiver + preimage, refund path = sender after timeout.
        let script = sc::compile_htlc(&h.secret_hash, receiver_pk, h.timeout, sender_pk);
        let htlc_addr = hash(&script);
        for (j, denom) in decompose_value(h.amount).into_iter().enumerate() {
            let salt = derived_salt(channel_id, nonce, b"HTLC", i as u32, j as u32);
            let cid = compute_coin_id(&htlc_addr, denom, &salt);
            output_hashes.push(cid);
            outputs.push(OutputData::Standard { address: htlc_addr, value: denom, salt });
            htlc_coins.push(HtlcCoin {
                coin_id: cid,
                address: htlc_addr,
                bytecode: script.clone(),
                value: denom,
                salt,
                secret_hash: h.secret_hash,
                timeout: h.timeout,
            });
        }
    }

    if outputs.is_empty() {
        bail!("qbolt: state produces no outputs");
    }

    let mut sh = blake3::Hasher::new();
    sh.update(salt_domain);
    sh.update(channel_id);
    sh.update(&nonce.to_le_bytes());
    sh.update(&attempt.to_le_bytes());
    let tx_salt = *sh.finalize().as_bytes();

    let input_coin_ids: Vec<[u8; 32]> = sorted.iter().map(|f| f.0).collect();
    let commitment = compute_commitment(&input_coin_ids, &output_hashes, &tx_salt);

    Ok(BuiltState {
        commitment,
        outputs,
        salt: tx_salt,
        fee: CLOSE_FEE,
        capacity,
        input_coin_ids,
        htlc_coins,
        nonce,
        attempt,
    })
}

/// Build the canonical close state for a channel.
#[allow(clippy::too_many_arguments)]
pub fn build_state(
    channel_id: &[u8; 32],
    sender_pk: &[u8; 32],
    receiver_pk: &[u8; 32],
    expiry: u64,
    funding: &[FundingCoin],
    sender_amt: u64,
    receiver_amt: u64,
    nonce: u32,
    htlcs: &[Htlc],
    attempt: u32,
) -> Result<BuiltState> {
    let addr = channel_address(sender_pk, receiver_pk, expiry);
    build_state_core(
        channel_id, sender_pk, receiver_pk, &addr, funding, sender_amt, receiver_amt, nonce,
        htlcs, attempt, b"qbolt_close_v2",
    )
}

/// The sender's post-expiry refund state: everything (minus fee) back to the
/// sender. nonce = u32::MAX so its salts can never collide with a payment
/// state.
pub fn build_refund_state(
    channel_id: &[u8; 32],
    sender_pk: &[u8; 32],
    receiver_pk: &[u8; 32],
    expiry: u64,
    funding: &[FundingCoin],
    attempt: u32,
) -> Result<BuiltState> {
    let addr = channel_address(sender_pk, receiver_pk, expiry);
    let capacity: u64 = sorted_funding(funding, &addr)?.iter().map(|f| f.1).sum();
    if capacity <= CLOSE_FEE {
        bail!("qbolt: capacity does not cover the refund fee");
    }
    build_state_core(
        channel_id, sender_pk, receiver_pk, &addr, funding,
        capacity - CLOSE_FEE, 0, u32::MAX, &[], attempt, b"qbolt_refund_v2",
    )
}

fn reveal_core(
    covenant: &[u8],
    funding: &[FundingCoin],
    state: &BuiltState,
    witness_items: Vec<Vec<u8>>,
) -> Result<(Vec<InputReveal>, Vec<Witness>)> {
    let addr = hash(covenant);
    let sorted = sorted_funding(funding, &addr)?;
    // Consistency guard: the state must have been built over these exact inputs.
    if sorted.len() != state.input_coin_ids.len()
        || sorted.iter().zip(&state.input_coin_ids).any(|(f, id)| f.0 != *id)
    {
        bail!("qbolt: state input set does not match the funding list");
    }
    let inputs: Vec<InputReveal> = sorted
        .iter()
        .map(|f| InputReveal {
            predicate: Predicate::Script { bytecode: covenant.to_vec() },
            value: f.1,
            salt: f.2,
            commitment: None,
        })
        .collect();
    // One MSS signature binds the whole-tx commitment, so the SAME witness
    // satisfies every input's covenant.
    let witnesses: Vec<Witness> =
        (0..inputs.len()).map(|_| Witness::ScriptInputs(witness_items.clone())).collect();
    Ok((inputs, witnesses))
}

/// Cooperative / unilateral-receiver close reveal.
/// Witness per input: [sender_sig, receiver_sig, 0x01].
pub fn close_reveal(
    sender_pk: &[u8; 32],
    receiver_pk: &[u8; 32],
    expiry: u64,
    funding: &[FundingCoin],
    state: &BuiltState,
    sender_sig: &[u8],
    receiver_sig: &[u8],
) -> Result<(Vec<InputReveal>, Vec<Witness>)> {
    if sender_sig.is_empty() || receiver_sig.is_empty() {
        bail!("qbolt: close needs both signatures");
    }
    let covenant = covenant_bytes(sender_pk, receiver_pk, expiry);
    reveal_core(
        &covenant,
        funding,
        state,
        vec![sender_sig.to_vec(), receiver_sig.to_vec(), vec![0x01]],
    )
}

/// Sender's post-expiry refund reveal. Witness per input: [sender_sig, 0x00].
pub fn refund_reveal(
    sender_pk: &[u8; 32],
    receiver_pk: &[u8; 32],
    expiry: u64,
    funding: &[FundingCoin],
    state: &BuiltState,
    sender_sig: &[u8],
) -> Result<(Vec<InputReveal>, Vec<Witness>)> {
    if sender_sig.is_empty() {
        bail!("qbolt: refund needs the sender signature");
    }
    let covenant = covenant_bytes(sender_pk, receiver_pk, expiry);
    reveal_core(&covenant, funding, state, vec![sender_sig.to_vec(), vec![0x00]])
}


/// What a payee's signature over a bus-delivered invoice binds: their own pk
/// plus the hash, amount, expiry and route hints. The payer verifies this
/// against the pk it addressed, so nobody watching the public chat bus can
/// race a forged invoice (own hash, own hints) at an open request.
pub fn invoice_commit(
    payee_pk: &[u8; 32],
    hash: &[u8; 32],
    amount: u64,
    expiry: u64,
    hints: &[[u8; 32]],
) -> [u8; 32] {
    let mut head = Vec::with_capacity(87 + hints.len() * 32);
    head.extend_from_slice(b"qbinv1");
    head.extend_from_slice(payee_pk);
    head.extend_from_slice(hash);
    head.extend_from_slice(&amount.to_le_bytes());
    head.extend_from_slice(&expiry.to_le_bytes());
    head.push(hints.len() as u8);
    for h in hints {
        head.extend_from_slice(h);
    }
    hash_bytes(&head)
}

/// BLAKE3 of arbitrary bytes (the same hash the script VM's OP_HASH uses).
pub fn hash_bytes(b: &[u8]) -> [u8; 32] {
    *blake3::Hasher::new().update(b).finalize().as_bytes()
}

/// An HTLC coin sitting on-chain after a channel closed, plus the branch we
/// intend to take. Used to build the sweep reveal.
pub fn htlc_script(
    secret_hash: &[u8; 32],
    receiver_pk: &[u8; 32],
    timeout: u64,
    sender_pk: &[u8; 32],
) -> Vec<u8> {
    sc::compile_htlc(secret_hash, receiver_pk, timeout, sender_pk)
}

/// Sweep an on-chain HTLC coin via the CLAIM branch (receiver + preimage).
/// Witness (bottom → top): [receiver_sig, secret, 0x01].
pub fn htlc_claim_input(
    bytecode: &[u8],
    value: u64,
    salt: [u8; 32],
    receiver_sig: &[u8],
    secret: &[u8; 32],
) -> (InputReveal, Witness) {
    (
        InputReveal {
            predicate: Predicate::Script { bytecode: bytecode.to_vec() },
            value,
            salt,
            commitment: None,
        },
        Witness::ScriptInputs(vec![receiver_sig.to_vec(), secret.to_vec(), vec![0x01]]),
    )
}

/// Sweep an on-chain HTLC coin via the TIMEOUT branch (sender, after
/// `timeout`). Witness: [sender_sig, <32 filler bytes>, 0x00] — the ELSE
/// branch drops the filler before checking the timelock.
pub fn htlc_timeout_input(
    bytecode: &[u8],
    value: u64,
    salt: [u8; 32],
    sender_sig: &[u8],
) -> (InputReveal, Witness) {
    (
        InputReveal {
            predicate: Predicate::Script { bytecode: bytecode.to_vec() },
            value,
            salt,
            commitment: None,
        },
        Witness::ScriptInputs(vec![sender_sig.to_vec(), vec![0u8; 32], vec![0x00]]),
    )
}

/// Wire codecs — byte-identical to worker.js (`qbPackOpen`, `qbPackState`,
/// `qbPackU32` and their unpackers). A qbolt chat frame is
/// `words = [255, CMD]` with attachments `CoinId(channel_id)` +
/// `Signature(payload)` (+ `Address(pk)` for OPEN). All integers little-endian.
pub mod wire {
    use super::{FundingCoin, Htlc, MAX_HTLCS};

    pub const VERSION: u8 = 2;
    /// The chat-word protocol marker: `words[0]`.
    pub const MARKER: u8 = 255;

    pub const CMD_UPDATE: u8 = 50;
    pub const CMD_ACK: u8 = 51;
    pub const CMD_HTLC_ADD: u8 = 52;
    pub const CMD_HTLC_CLAIM: u8 = 53;
    pub const CMD_CLOSE_REQ: u8 = 54;
    pub const CMD_REJECT: u8 = 55;
    pub const CMD_RESIGN_REQ: u8 = 56;
    pub const CMD_RESIGN: u8 = 57;
    pub const CMD_LEGACY_CLOSE_REQ: u8 = 58;
    pub const CMD_LEGACY_CLOSE_SIG: u8 = 59;
    pub const CMD_CLOSED: u8 = 60;
    pub const CMD_HTLC_FAIL: u8 = 61;
    pub const CMD_INVOICE_REQ: u8 = 62;
    pub const CMD_INVOICE: u8 = 63;
    pub const CMD_OPEN: u8 = 110;

    /// OPEN payload: [ver][expiry u64][n u8][{value u64, salt 32}×n][sig0…]
    pub fn pack_open(expiry: u64, funding: &[FundingCoin], sig0: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(10 + funding.len() * 40 + sig0.len());
        out.push(VERSION);
        out.extend_from_slice(&expiry.to_le_bytes());
        out.push(funding.len() as u8);
        for c in funding {
            out.extend_from_slice(&c.value.to_le_bytes());
            out.extend_from_slice(&c.salt);
        }
        out.extend_from_slice(sig0);
        out
    }

    pub fn unpack_open(b: &[u8]) -> Option<(u64, Vec<FundingCoin>, Vec<u8>)> {
        if b.len() < 10 || b[0] != VERSION {
            return None;
        }
        let expiry = u64::from_le_bytes(b[1..9].try_into().ok()?);
        let n = b[9] as usize;
        if b.len() < 10 + n * 40 {
            return None;
        }
        let mut funding = Vec::with_capacity(n);
        let mut o = 10;
        for _ in 0..n {
            let value = u64::from_le_bytes(b[o..o + 8].try_into().ok()?);
            let salt: [u8; 32] = b[o + 8..o + 40].try_into().ok()?;
            funding.push(FundingCoin { value, salt });
            o += 40;
        }
        Some((expiry, funding, b[o..].to_vec()))
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct StateWire {
        pub nonce: u32,
        pub sender_amt: u64,
        pub receiver_amt: u64,
        pub htlcs: Vec<Htlc>,
        pub sig: Vec<u8>,
    }

    /// STATE payload: [ver][nonce u32][sender u64][receiver u64][h u8]
    /// [{amount u64, timeout u64, hash 32}×h][sender_sig…]
    pub fn pack_state(st: &StateWire) -> Vec<u8> {
        let mut out = Vec::with_capacity(22 + st.htlcs.len() * 48 + st.sig.len());
        out.push(VERSION);
        out.extend_from_slice(&st.nonce.to_le_bytes());
        out.extend_from_slice(&st.sender_amt.to_le_bytes());
        out.extend_from_slice(&st.receiver_amt.to_le_bytes());
        out.push(st.htlcs.len() as u8);
        for h in &st.htlcs {
            out.extend_from_slice(&h.amount.to_le_bytes());
            out.extend_from_slice(&h.timeout.to_le_bytes());
            out.extend_from_slice(&h.secret_hash);
        }
        out.extend_from_slice(&st.sig);
        out
    }

    pub fn unpack_state(b: &[u8]) -> Option<StateWire> {
        if b.len() < 22 || b[0] != VERSION {
            return None;
        }
        let nonce = u32::from_le_bytes(b[1..5].try_into().ok()?);
        let sender_amt = u64::from_le_bytes(b[5..13].try_into().ok()?);
        let receiver_amt = u64::from_le_bytes(b[13..21].try_into().ok()?);
        let n = b[21] as usize;
        if n > MAX_HTLCS || b.len() < 22 + n * 48 {
            return None;
        }
        let mut htlcs = Vec::with_capacity(n);
        let mut o = 22;
        for _ in 0..n {
            htlcs.push(Htlc {
                amount: u64::from_le_bytes(b[o..o + 8].try_into().ok()?),
                timeout: u64::from_le_bytes(b[o + 8..o + 16].try_into().ok()?),
                secret_hash: b[o + 16..o + 48].try_into().ok()?,
            });
            o += 48;
        }
        Some(StateWire { nonce, sender_amt, receiver_amt, htlcs, sig: b[o..].to_vec() })
    }

    /// INVOICE payload: [ver][amount u64][expiry u64][hash 32][n u8]
    /// [hint 32 ×n][payee_sig…]
    pub fn pack_invoice(
        hash: &[u8; 32],
        amount: u64,
        expiry: u64,
        hints: &[[u8; 32]],
        sig: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(50 + hints.len() * 32 + sig.len());
        out.push(VERSION);
        out.extend_from_slice(&amount.to_le_bytes());
        out.extend_from_slice(&expiry.to_le_bytes());
        out.extend_from_slice(hash);
        out.push(hints.len() as u8);
        for h in hints {
            out.extend_from_slice(h);
        }
        out.extend_from_slice(sig);
        out
    }

    pub struct InvoiceWire {
        pub amount: u64,
        pub expiry: u64,
        pub hash: [u8; 32],
        pub hints: Vec<[u8; 32]>,
        pub sig: Vec<u8>,
    }

    pub fn unpack_invoice(b: &[u8]) -> Option<InvoiceWire> {
        if b.len() < 50 || b[0] != VERSION {
            return None;
        }
        let amount = u64::from_le_bytes(b[1..9].try_into().ok()?);
        let expiry = u64::from_le_bytes(b[9..17].try_into().ok()?);
        let hash: [u8; 32] = b[17..49].try_into().ok()?;
        let n = b[49] as usize;
        if n > 2 || b.len() < 50 + n * 32 {
            return None;
        }
        let mut hints = Vec::with_capacity(n);
        let mut o = 50;
        for _ in 0..n {
            hints.push(b[o..o + 32].try_into().ok()?);
            o += 32;
        }
        Some(InvoiceWire { amount, expiry, hash, hints, sig: b[o..].to_vec() })
    }

    /// Small payload: [ver][u32][extra…] (ACK, CLOSE_REQ, CLOSED, REJECT…)
    pub fn pack_u32(n: u32, extra: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + extra.len());
        out.push(VERSION);
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(extra);
        out
    }

    pub fn unpack_u32(b: &[u8]) -> Option<(u32, Vec<u8>)> {
        if b.len() < 5 || b[0] != VERSION {
            return None;
        }
        Some((u32::from_le_bytes(b[1..5].try_into().ok()?), b[5..].to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(value: u64, byte: u8) -> FundingCoin {
        FundingCoin { value, salt: [byte; 32] }
    }

    #[test]
    fn open_codec_roundtrip() {
        let funding = vec![fc(4096, 1), fc(1024, 2)];
        let sig = vec![7u8; 90];
        let packed = wire::pack_open(123_456, &funding, &sig);
        let (e, f, s) = wire::unpack_open(&packed).unwrap();
        assert_eq!(e, 123_456);
        assert_eq!(f, funding);
        assert_eq!(s, sig);
    }

    #[test]
    fn state_codec_roundtrip_and_version_reject() {
        let st = wire::StateWire {
            nonce: 9,
            sender_amt: 5000,
            receiver_amt: 1192,
            htlcs: vec![Htlc { amount: 8, timeout: 42, secret_hash: [3; 32] }],
            sig: vec![9u8; 64],
        };
        let mut packed = wire::pack_state(&st);
        assert_eq!(wire::unpack_state(&packed).unwrap(), st);
        packed[0] = 1; // wrong version
        assert!(wire::unpack_state(&packed).is_none());
    }

    #[test]
    fn u32_codec_roundtrip() {
        let p = wire::pack_u32(77, &[1, 2, 3]);
        assert_eq!(wire::unpack_u32(&p).unwrap(), (77, vec![1, 2, 3]));
    }

    #[test]
    fn funding_rules() {
        let addr = [9u8; 32];
        assert!(sorted_funding(&[], &addr).is_err());
        assert!(sorted_funding(&[fc(3, 1)], &addr).is_err()); // not a power of 2
        assert!(sorted_funding(&[fc(8, 1), fc(8, 1)], &addr).is_err()); // duplicate
        let two = sorted_funding(&[fc(8, 1), fc(8, 2)], &addr).unwrap();
        assert!(two[0].0 <= two[1].0); // sorted
    }

    #[test]
    fn conservation_enforced_and_domains_differ() {
        let s = [1u8; 32];
        let r = [2u8; 32];
        let funding = vec![fc(4096, 5)];
        let addr = channel_address(&s, &r, 500);
        let id = channel_id(&funding, &addr).unwrap();

        // 4096 - 2000 = 2096 distributable.
        assert!(build_state(&id, &s, &r, 500, &funding, 2000, 100, 1, &[], 0).is_err());
        let close = build_state(&id, &s, &r, 500, &funding, 2000, 96, 1, &[], 0).unwrap();
        assert_eq!(close.capacity, 4096);

        let refund = build_refund_state(&id, &s, &r, 500, &funding, 0).unwrap();
        assert_eq!(refund.nonce, u32::MAX);
        // Different salt domains ⇒ different commitments even for equal splits.
        let close_like_refund =
            build_state(&id, &s, &r, 500, &funding, 2096, 0, u32::MAX, &[], 0).unwrap();
        assert_ne!(refund.commitment, close_like_refund.commitment);
    }

    #[test]
    fn attempt_freshens_commitment() {
        let s = [1u8; 32];
        let r = [2u8; 32];
        let funding = vec![fc(4096, 5)];
        let addr = channel_address(&s, &r, 500);
        let id = channel_id(&funding, &addr).unwrap();
        let a = build_state(&id, &s, &r, 500, &funding, 2000, 96, 1, &[], 0).unwrap();
        let b = build_state(&id, &s, &r, 500, &funding, 2000, 96, 1, &[], 1).unwrap();
        assert_ne!(a.commitment, b.commitment);
        assert_eq!(a.outputs, b.outputs); // same coins, fresh salt/commitment
    }
}
