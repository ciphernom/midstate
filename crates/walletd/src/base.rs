//! Base JSON-RPC client and a typed binding for `MidstateAtomicSwap`.
//!
//! The contract is deliberately never trusted to enforce cross-chain safety —
//! it cannot, because it has no idea what the Midstate side looks like.
//! `lock()` only checks `600 <= refundDelay <= 7 days`. The ordering rule that
//! actually keeps a swap atomic lives in [`check_ordering`] and is ours to get
//! right.

use crate::evm::*;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

/// Where the EVM leg points. Overridable so the whole flow can be rehearsed on
/// a testnet before real value is at stake.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChainConfig {
    pub rpc_url: String,
    pub chain_id: u64,
    /// `MidstateAtomicSwap` address, hex with or without `0x`.
    pub contract: String,
    /// Blocks to wait before believing a lock. Base reorgs are shallow but not
    /// impossible, and a swap acted on too early is a swap that can be undone.
    pub confirmations: u64,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            rpc_url: BASE_MAINNET_RPC.into(),
            chain_id: BASE_MAINNET_CHAIN_ID,
            contract: BASE_MAINNET_CONTRACT.into(),
            confirmations: 3,
        }
    }
}

pub struct BaseClient {
    http: reqwest::Client,
    pub cfg: ChainConfig,
    contract: [u8; 20],
}

fn hex_u64(v: &Value) -> Result<u64> {
    let s = v.as_str().ok_or_else(|| anyhow!("expected a hex quantity"))?;
    u64::from_str_radix(s.trim_start_matches("0x"), 16).context("bad hex quantity")
}
fn hex_u128(v: &Value) -> Result<u128> {
    let s = v.as_str().ok_or_else(|| anyhow!("expected a hex quantity"))?;
    u128::from_str_radix(s.trim_start_matches("0x"), 16).context("bad hex quantity")
}
fn hex_bytes(v: &Value) -> Result<Vec<u8>> {
    let s = v.as_str().ok_or_else(|| anyhow!("expected hex data"))?;
    hex::decode(s.trim_start_matches("0x")).context("bad hex data")
}

impl BaseClient {
    pub fn new(cfg: ChainConfig) -> Result<Self> {
        let contract = parse_address(&cfg.contract)?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .context("failed to build the HTTP client")?;
        Ok(Self { http, cfg, contract })
    }

    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let body = json!({"jsonrpc":"2.0","id":1,"method":method,"params":params});
        let resp: Value = self
            .http
            .post(&self.cfg.rpc_url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("{method}: could not reach {}", self.cfg.rpc_url))?
            .json()
            .await
            .with_context(|| format!("{method}: response was not JSON"))?;
        if let Some(e) = resp.get("error") {
            let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("unknown");
            bail!("{method} failed: {msg}");
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| anyhow!("{method}: response had no result"))
    }

    // ── Chain basics ────────────────────────────────────────────────────

    pub async fn chain_id(&self) -> Result<u64> {
        hex_u64(&self.call("eth_chainId", json!([])).await?)
    }

    pub async fn block_number(&self) -> Result<u64> {
        hex_u64(&self.call("eth_blockNumber", json!([])).await?)
    }

    pub async fn balance(&self, addr: &[u8; 20]) -> Result<u128> {
        hex_u128(
            &self
                .call("eth_getBalance", json!([to_checksum_address(addr), "latest"]))
                .await?,
        )
    }

    pub async fn nonce(&self, addr: &[u8; 20]) -> Result<u64> {
        // "pending" so several transactions can be queued in one session.
        hex_u64(
            &self
                .call("eth_getTransactionCount", json!([to_checksum_address(addr), "pending"]))
                .await?,
        )
    }

    /// Current base fee plus a tip, doubled for headroom — a transaction that
    /// underprices itself during a swap can miss its deadline.
    pub async fn fees(&self) -> Result<(u128, u128)> {
        let block = self
            .call("eth_getBlockByNumber", json!(["latest", false]))
            .await?;
        let base = block
            .get("baseFeePerGas")
            .map(hex_u128)
            .transpose()?
            .unwrap_or(1_000_000);
        let tip: u128 = self
            .call("eth_maxPriorityFeePerGas", json!([]))
            .await
            .ok()
            .and_then(|v| hex_u128(&v).ok())
            .unwrap_or(1_000_000);
        Ok((tip, base * 2 + tip))
    }

    pub async fn estimate_gas(&self, from: &[u8; 20], value: u128, data: &[u8]) -> Result<u64> {
        let v = self
            .call(
                "eth_estimateGas",
                json!([{
                    "from": to_checksum_address(from),
                    "to": to_checksum_address(&self.contract),
                    "value": format!("0x{value:x}"),
                    "data": format!("0x{}", hex::encode(data)),
                }]),
            )
            .await?;
        // 25% headroom: estimates are made against current state and the swap
        // contract's storage writes vary with whether a slot is already warm.
        Ok(hex_u64(&v)? * 5 / 4)
    }

    async fn eth_call(&self, data: &[u8]) -> Result<Vec<u8>> {
        let v = self
            .call(
                "eth_call",
                json!([{
                    "to": to_checksum_address(&self.contract),
                    "data": format!("0x{}", hex::encode(data)),
                }, "latest"]),
            )
            .await?;
        hex_bytes(&v)
    }

    /// Build, sign and broadcast a contract call. Returns the transaction hash.
    pub async fn send(
        &self,
        key: &EvmKey,
        value: u128,
        data: Vec<u8>,
    ) -> Result<[u8; 32]> {
        let onchain_id = self.chain_id().await?;
        if onchain_id != self.cfg.chain_id {
            bail!(
                "the RPC endpoint is chain {onchain_id}, but this wallet is configured for \
                 chain {} — refusing to sign",
                self.cfg.chain_id
            );
        }
        let nonce = self.nonce(&key.address).await?;
        let (tip, max_fee) = self.fees().await?;
        let gas_limit = self.estimate_gas(&key.address, value, &data).await?;

        let need = value + max_fee * gas_limit as u128;
        let have = self.balance(&key.address).await?;
        if have < need {
            bail!(
                "insufficient ETH on Base: need about {} wei including gas, have {}",
                need, have
            );
        }

        let tx = TxRequest {
            chain_id: self.cfg.chain_id,
            nonce,
            max_priority_fee: tip,
            max_fee,
            gas_limit,
            to: self.contract,
            value,
            data,
        };
        let raw = tx.sign(key)?;
        let v = self
            .call(
                "eth_sendRawTransaction",
                json!([format!("0x{}", hex::encode(&raw))]),
            )
            .await?;
        let h = hex_bytes(&v)?;
        h.try_into().map_err(|_| anyhow!("node returned a malformed transaction hash"))
    }

    /// Poll for a receipt. `None` while still pending.
    pub async fn receipt(&self, tx: &[u8; 32]) -> Result<Option<Receipt>> {
        let v = self
            .call(
                "eth_getTransactionReceipt",
                json!([format!("0x{}", hex::encode(tx))]),
            )
            .await?;
        if v.is_null() {
            return Ok(None);
        }
        let status = v.get("status").map(hex_u64).transpose()?.unwrap_or(0);
        let block = v.get("blockNumber").map(hex_u64).transpose()?.unwrap_or(0);
        let mut logs = Vec::new();
        if let Some(arr) = v.get("logs").and_then(|l| l.as_array()) {
            for l in arr {
                logs.push(parse_log(l)?);
            }
        }
        Ok(Some(Receipt { success: status == 1, block, logs }))
    }

    /// Fetch contract logs in a height range, optionally filtered by topic0.
    pub async fn logs(&self, from: u64, to: u64, topic0: Option<[u8; 32]>) -> Result<Vec<Log>> {
        let mut filter = json!({
            "fromBlock": format!("0x{from:x}"),
            "toBlock": format!("0x{to:x}"),
            "address": to_checksum_address(&self.contract),
        });
        if let Some(t) = topic0 {
            filter["topics"] = json!([format!("0x{}", hex::encode(t))]);
        }
        let v = self.call("eth_getLogs", json!([filter])).await?;
        let arr = v.as_array().ok_or_else(|| anyhow!("eth_getLogs did not return a list"))?;
        arr.iter().map(parse_log).collect()
    }

    // ── Contract calls ──────────────────────────────────────────────────

    pub async fn lock(
        &self,
        key: &EvmKey,
        hashlock: [u8; 32],
        beneficiary: [u8; 20],
        refund_delay: u64,
        wei: u128,
    ) -> Result<[u8; 32]> {
        if !(600..=604_800).contains(&refund_delay) {
            bail!("the contract requires a refund delay between 10 minutes and 7 days");
        }
        let data = encode_call(
            "lock(bytes32,address,uint256)",
            &[
                Word::Bytes32(hashlock),
                Word::Address(beneficiary),
                Word::U256(refund_delay as u128),
            ],
        );
        self.send(key, wei, data).await
    }

    pub async fn claim(&self, key: &EvmKey, swap_id: [u8; 32], preimage: [u8; 32]) -> Result<[u8; 32]> {
        let data = encode_call(
            "claim(bytes32,bytes32)",
            &[Word::Bytes32(swap_id), Word::Bytes32(preimage)],
        );
        self.send(key, 0, data).await
    }

    pub async fn refund(&self, key: &EvmKey, swap_id: [u8; 32]) -> Result<[u8; 32]> {
        let data = encode_call("refund(bytes32)", &[Word::Bytes32(swap_id)]);
        self.send(key, 0, data).await
    }

    // ── Resting bids ────────────────────────────────────────────────────

    pub async fn create_bid(
        &self,
        key: &EvmKey,
        hashlock: [u8; 32],
        mds_amount: u64,
        maker_mds_addr: [u8; 32],
        ttl_secs: u64,
        fill_bond: u128,
        wei: u128,
    ) -> Result<[u8; 32]> {
        if !(3_600..=7_776_000).contains(&ttl_secs) {
            bail!("the contract requires a bid TTL between 1 hour and 90 days");
        }
        let data = encode_call(
            "createBid(bytes32,uint64,bytes32,uint256,uint256)",
            &[
                Word::Bytes32(hashlock),
                Word::U64(mds_amount),
                Word::Bytes32(maker_mds_addr),
                Word::U256(ttl_secs as u128),
                Word::U256(fill_bond),
            ],
        );
        self.send(key, wei, data).await
    }

    /// Claim the exclusive right to fill a bid. The bond is at stake: it comes
    /// back on a successful fill and is forfeited to the maker otherwise.
    pub async fn reserve_bid(
        &self,
        key: &EvmKey,
        bid_id: [u8; 32],
        fill_window_secs: u64,
        bond: u128,
    ) -> Result<[u8; 32]> {
        let data = encode_call(
            "reserveBid(bytes32,uint256)",
            &[Word::Bytes32(bid_id), Word::U256(fill_window_secs as u128)],
        );
        self.send(key, bond, data).await
    }

    pub async fn claim_bid(&self, key: &EvmKey, bid_id: [u8; 32], preimage: [u8; 32]) -> Result<[u8; 32]> {
        let data = encode_call(
            "claimBid(bytes32,bytes32)",
            &[Word::Bytes32(bid_id), Word::Bytes32(preimage)],
        );
        self.send(key, 0, data).await
    }

    pub async fn cancel_bid(&self, key: &EvmKey, bid_id: [u8; 32]) -> Result<[u8; 32]> {
        let data = encode_call("cancelBid(bytes32)", &[Word::Bytes32(bid_id)]);
        self.send(key, 0, data).await
    }

    /// Scan contract history for the live order book. Bids are escrowed on
    /// creation, so a `BidCreated` that has not been claimed or cancelled is
    /// real, funded liquidity — no maker cooperation needed to verify it.
    pub async fn scan_events(&self, from: u64, to: u64) -> Result<Vec<(u64, Event)>> {
        Ok(self.scan_events_counted(from, to).await?.0)
    }

    /// Same scan, but also reports how many raw logs the contract emitted.
    /// A large gap between raw and decoded means an event signature is wrong,
    /// which otherwise looks exactly like "there is no activity".
    pub async fn scan_events_counted(&self, from: u64, to: u64) -> Result<(Vec<(u64, Event)>, usize)> {
        let mut out = Vec::new();
        let mut raw = 0usize;
        // Public endpoints cap the range, so walk it in chunks.
        let mut cursor = from;
        while cursor <= to {
            let end = (cursor + 9_999).min(to);
            for l in self.logs(cursor, end, None).await? {
                raw += 1;
                if let Some(e) = decode_event(&l) {
                    out.push((l.block, e));
                }
            }
            cursor = end + 1;
        }
        Ok((out, raw))
    }

    /// Read a swap back from contract storage.
    pub async fn get_swap(&self, swap_id: [u8; 32]) -> Result<Option<SwapState>> {
        let out = self
            .eth_call(&encode_call("swaps(bytes32)", &[Word::Bytes32(swap_id)]))
            .await?;
        let w = words(&out);
        if w.len() < 6 {
            return Ok(None);
        }
        let amount = word_u128(&w[2]);
        if amount == 0 {
            return Ok(None);
        }
        Ok(Some(SwapState {
            beneficiary: word_address(&w[0]),
            refund_to: word_address(&w[1]),
            amount,
            hashlock: w[3],
            timeout: word_u64(&w[4]),
            settled: word_bool(&w[5]),
        }))
    }
}

#[derive(Clone, Debug)]
pub struct SwapState {
    pub beneficiary: [u8; 20],
    pub refund_to: [u8; 20],
    pub amount: u128,
    pub hashlock: [u8; 32],
    pub timeout: u64,
    pub settled: bool,
}

#[derive(Clone, Debug)]
pub struct Receipt {
    pub success: bool,
    pub block: u64,
    pub logs: Vec<Log>,
}

#[derive(Clone, Debug)]
pub struct Log {
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
    pub block: u64,
}

fn parse_log(v: &Value) -> Result<Log> {
    let mut topics = Vec::new();
    if let Some(arr) = v.get("topics").and_then(|t| t.as_array()) {
        for t in arr {
            let b = hex_bytes(t)?;
            if b.len() == 32 {
                topics.push(b.try_into().unwrap());
            }
        }
    }
    Ok(Log {
        topics,
        data: v.get("data").map(hex_bytes).transpose()?.unwrap_or_default(),
        block: v.get("blockNumber").map(hex_u64).transpose()?.unwrap_or(0),
    })
}

// ── Events ──────────────────────────────────────────────────────────────
//
// Signatures are transcribed from MidstateAtomicSwap.sol exactly, because a
// topic0 that is one parameter off matches nothing at all — the log filter
// simply returns an empty list forever and the wallet looks like it is working.
// `Claimed` in particular takes THREE bytes32 (swapId, hashlock, preimage),
// not two.
//
// `swapId` and `bidId` both fold in `block.timestamp`, so neither can be
// computed before the transaction is mined — they are read back from these
// events, which is why every flow waits for a receipt.

pub fn topic_locked() -> [u8; 32] {
    keccak256(b"Locked(bytes32,address,address,uint256,uint64,bytes32)")
}
pub fn topic_claimed() -> [u8; 32] {
    keccak256(b"Claimed(bytes32,bytes32,bytes32)")
}
pub fn topic_refunded() -> [u8; 32] {
    keccak256(b"Refunded(bytes32)")
}
pub fn topic_bid_created() -> [u8; 32] {
    keccak256(b"BidCreated(bytes32,address,bytes32,uint256,uint256,uint64,bytes32,uint64)")
}
pub fn topic_bid_reserved() -> [u8; 32] {
    keccak256(b"BidReserved(bytes32,address,uint64)")
}
pub fn topic_bid_claimed() -> [u8; 32] {
    keccak256(b"BidClaimed(bytes32,bytes32,bytes32)")
}
pub fn topic_bid_cancelled() -> [u8; 32] {
    keccak256(b"BidCancelled(bytes32)")
}

/// Every contract event, decoded. Indexed parameters live in topics; the rest
/// are 32-byte words in `data`, in declaration order.
#[derive(Clone, Debug)]
pub enum Event {
    /// ETH escrowed against a hashlock.
    Locked {
        swap_id: [u8; 32],
        beneficiary: [u8; 20],
        refund_to: [u8; 20],
        amount: u128,
        timeout: u64,
        hashlock: [u8; 32],
    },
    /// The moment a secret becomes public. The far side of the swap races to
    /// use it before its own deadline.
    Claimed { swap_id: [u8; 32], hashlock: [u8; 32], preimage: [u8; 32] },
    Refunded { swap_id: [u8; 32] },
    /// A resting buy order: ETH escrowed up front, so bids are verifiable
    /// without trusting the maker.
    BidCreated {
        bid_id: [u8; 32],
        maker: [u8; 20],
        hashlock: [u8; 32],
        amount: u128,
        fill_bond: u128,
        mds_amount: u64,
        maker_mds_addr: [u8; 32],
        expiry: u64,
    },
    BidReserved { bid_id: [u8; 32], filler: [u8; 20], fill_deadline: u64 },
    BidClaimed { bid_id: [u8; 32], hashlock: [u8; 32], preimage: [u8; 32] },
    BidCancelled { bid_id: [u8; 32] },
}

pub fn decode_event(log: &Log) -> Option<Event> {
    let t0 = *log.topics.first()?;
    let d = words(&log.data);
    let topic = |i: usize| log.topics.get(i).copied();

    if t0 == topic_locked() {
        return Some(Event::Locked {
            swap_id: topic(1)?,
            beneficiary: word_address(&topic(2)?),
            refund_to: word_address(&topic(3)?),
            amount: word_u128(d.first()?),
            timeout: word_u64(d.get(1)?),
            hashlock: *d.get(2)?,
        });
    }
    if t0 == topic_claimed() {
        return Some(Event::Claimed {
            swap_id: topic(1)?,
            hashlock: *d.first()?,
            preimage: *d.get(1)?,
        });
    }
    if t0 == topic_refunded() {
        return Some(Event::Refunded { swap_id: topic(1)? });
    }
    if t0 == topic_bid_created() {
        return Some(Event::BidCreated {
            bid_id: topic(1)?,
            maker: word_address(&topic(2)?),
            hashlock: *d.first()?,
            amount: word_u128(d.get(1)?),
            fill_bond: word_u128(d.get(2)?),
            mds_amount: word_u64(d.get(3)?),
            maker_mds_addr: *d.get(4)?,
            expiry: word_u64(d.get(5)?),
        });
    }
    if t0 == topic_bid_reserved() {
        return Some(Event::BidReserved {
            bid_id: topic(1)?,
            filler: word_address(&topic(2)?),
            fill_deadline: word_u64(d.first()?),
        });
    }
    if t0 == topic_bid_claimed() {
        return Some(Event::BidClaimed {
            bid_id: topic(1)?,
            hashlock: *d.first()?,
            preimage: *d.get(1)?,
        });
    }
    if t0 == topic_bid_cancelled() {
        return Some(Event::BidCancelled { bid_id: topic(1)? });
    }
    None
}

/// Pull our own swap id out of the receipt for a `lock()` call.
pub fn locked_swap_id(logs: &[Log]) -> Option<[u8; 32]> {
    logs.iter().find_map(|l| match decode_event(l) {
        Some(Event::Locked { swap_id, .. }) => Some(swap_id),
        _ => None,
    })
}

/// Our own bid id from the receipt for a `createBid()` call.
pub fn created_bid_id(logs: &[Log]) -> Option<[u8; 32]> {
    logs.iter().find_map(|l| match decode_event(l) {
        Some(Event::BidCreated { bid_id, .. }) => Some(bid_id),
        _ => None,
    })
}

/// Any preimage revealed by this log, from either flow. This is how the far
/// side of a swap learns the secret it needs.
pub fn revealed_preimage(log: &Log) -> Option<([u8; 32], [u8; 32])> {
    match decode_event(log)? {
        Event::Claimed { swap_id, preimage, .. } => Some((swap_id, preimage)),
        Event::BidClaimed { bid_id, preimage, .. } => Some((bid_id, preimage)),
        _ => None,
    }
}

// ── The rule the contract cannot enforce ────────────────────────────────

/// Both swap directions reduce to one invariant: **the chain where the secret
/// is revealed must have the earlier deadline**, so the counterparty still has
/// room to act on the other chain afterwards.
///
/// * Maker sells MDS — reveals on Base via `claim()`, so the Base refund must
///   fall well before the Midstate HTLC timeout.
/// * Maker buys MDS (resting bid) — reveals on Midstate, so the covenant
///   timeout must fall well before `fillDeadline`.
///
/// `margin_secs` is the room the second actor gets. Midstate blocks are ~60s,
/// so this is expressed in seconds on both sides.
pub fn check_ordering(reveal_deadline: u64, act_deadline: u64, margin_secs: u64) -> Result<()> {
    if reveal_deadline + margin_secs > act_deadline {
        bail!(
            "unsafe swap timing: the revealing side's deadline ({reveal_deadline}) leaves less \
             than {margin_secs}s before the other side's deadline ({act_deadline}). Whoever \
             acts second could run out of time and lose the swap."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_rule_is_symmetric() {
        // Reveal-first with plenty of room: fine.
        assert!(check_ordering(1_000, 5_000, 1_800).is_ok());
        // Same order but not enough margin: refused.
        assert!(check_ordering(1_000, 2_000, 1_800).is_err());
        // Inverted: always refused.
        assert!(check_ordering(5_000, 1_000, 1).is_err());
    }

    #[test]
    fn claimed_has_three_parameters() {
        // Regression guard. `Claimed(bytes32,bytes32)` is a plausible-looking
        // signature that hashes to a topic no log will ever carry, so the
        // filter silently returns nothing and the wallet appears healthy while
        // never seeing a revealed secret.
        assert_eq!(topic_claimed(), keccak256(b"Claimed(bytes32,bytes32,bytes32)"));
        assert_ne!(topic_claimed(), keccak256(b"Claimed(bytes32,bytes32)"));
        for (a, b) in [
            (topic_locked(), topic_refunded()),
            (topic_bid_created(), topic_bid_claimed()),
            (topic_claimed(), topic_bid_claimed()),
        ] {
            assert_ne!(a, b);
        }
    }

    fn log_for(topic0: [u8; 32], topics: Vec<[u8; 32]>, data: Vec<Word>) -> Log {
        let mut t = vec![topic0];
        t.extend(topics);
        let mut d = Vec::new();
        for w in data {
            d.extend_from_slice(&w.encode());
        }
        Log { topics: t, data: d, block: 1 }
    }

    #[test]
    fn locked_decodes_indexed_and_data_fields() {
        // Three indexed params land in topics 1..3; the rest are data words.
        let log = log_for(
            topic_locked(),
            vec![[7u8; 32], Word::Address([1; 20]).encode(), Word::Address([2; 20]).encode()],
            vec![Word::U256(5_000), Word::U64(1_700_000_000), Word::Bytes32([9; 32])],
        );
        match decode_event(&log).unwrap() {
            Event::Locked { swap_id, beneficiary, refund_to, amount, timeout, hashlock } => {
                assert_eq!(swap_id, [7u8; 32]);
                assert_eq!(beneficiary, [1u8; 20]);
                assert_eq!(refund_to, [2u8; 20]);
                assert_eq!(amount, 5_000);
                assert_eq!(timeout, 1_700_000_000);
                assert_eq!(hashlock, [9u8; 32]);
            }
            other => panic!("expected Locked, got {other:?}"),
        }
        assert_eq!(locked_swap_id(&[log]), Some([7u8; 32]));
    }

    #[test]
    fn both_flows_surrender_their_preimage() {
        let claimed = log_for(
            topic_claimed(),
            vec![[1u8; 32]],
            vec![Word::Bytes32([2; 32]), Word::Bytes32([3; 32])],
        );
        assert_eq!(revealed_preimage(&claimed), Some(([1u8; 32], [3u8; 32])));

        let bid = log_for(
            topic_bid_claimed(),
            vec![[4u8; 32]],
            vec![Word::Bytes32([5; 32]), Word::Bytes32([6; 32])],
        );
        assert_eq!(revealed_preimage(&bid), Some(([4u8; 32], [6u8; 32])));

        assert!(revealed_preimage(&log_for(topic_refunded(), vec![[1u8; 32]], vec![])).is_none());
    }

    #[test]
    fn bid_created_decodes_all_six_data_words() {
        let log = log_for(
            topic_bid_created(),
            vec![[1u8; 32], Word::Address([2; 20]).encode()],
            vec![
                Word::Bytes32([3; 32]),
                Word::U256(9_000),
                Word::U256(100),
                Word::U64(4096),
                Word::Bytes32([4; 32]),
                Word::U64(1_800_000_000),
            ],
        );
        match decode_event(&log).unwrap() {
            Event::BidCreated { amount, fill_bond, mds_amount, expiry, .. } => {
                assert_eq!((amount, fill_bond, mds_amount, expiry), (9_000, 100, 4096, 1_800_000_000));
            }
            other => panic!("expected BidCreated, got {other:?}"),
        }
    }

    #[test]
    fn swap_state_decodes_from_words() {
        let mut data = Vec::new();
        for w in [
            Word::Address([1; 20]),
            Word::Address([2; 20]),
            Word::U256(999),
            Word::Bytes32([3; 32]),
            Word::U64(1234),
            Word::Bool(true),
        ] {
            data.extend_from_slice(&w.encode());
        }
        let w = words(&data);
        assert_eq!(word_u128(&w[2]), 999);
        assert_eq!(word_u64(&w[4]), 1234);
        assert!(word_bool(&w[5]));
        assert_eq!(word_address(&w[0]), [1u8; 20]);
    }
}
