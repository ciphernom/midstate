//! Base JSON-RPC client and a typed binding for `MidstateAtomicSwap` (ETH)
//! and `MidstateTokenSwap` (USDC). The two contracts share their events and
//! their `swaps`/`bids` getters, so one reader serves both; a client is made
//! for one of them ([`BaseClient::for_asset`]), and only its calls that move
//! value differ: the token's are paid in with an EIP-2612 permit, in the
//! same transaction.
//!
//! The contract is deliberately never trusted to enforce cross-chain safety —
//! it cannot, because it has no idea what the Midstate side looks like.
//! `lock()` only checks `600 <= refundDelay <= 7 days`. The ordering rule that
//! actually keeps a swap atomic lives in [`check_ordering`] and is ours to get
//! right.

use crate::eth_trade::Asset;
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
    /// `MidstateTokenSwap` for USDC; empty while there is none, and USDC
    /// trading is off.
    #[serde(default = "default_usdc_contract")]
    pub usdc_contract: String,
    /// The token that contract was deployed for: USDC.
    #[serde(default = "default_usdc_token")]
    pub usdc_token: String,
}

fn default_usdc_contract() -> String {
    BASE_MAINNET_TOKEN_CONTRACT.into()
}

fn default_usdc_token() -> String {
    BASE_USDC.into()
}

impl ChainConfig {
    /// Whether USDC can be traded: a `MidstateTokenSwap` is set.
    pub fn usdc_ready(&self) -> bool {
        !self.usdc_contract.trim().is_empty()
    }
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            rpc_url: BASE_MAINNET_RPC.into(),
            chain_id: BASE_MAINNET_CHAIN_ID,
            contract: BASE_MAINNET_CONTRACT.into(),
            confirmations: 3,
            usdc_contract: default_usdc_contract(),
            usdc_token: default_usdc_token(),
        }
    }
}

pub struct BaseClient {
    http: reqwest::Client,
    pub cfg: ChainConfig,
    contract: [u8; 20],
    /// What the contract trades: ETH, or the token at `token`.
    pub asset: Asset,
    token: Option<[u8; 20]>,
}

/// How long a permit stands, in seconds: past the time a payment may still
/// be mined, so it never runs out on a transaction that is on its way.
pub const PERMIT_SECS: u64 = 1_800;

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
    /// A client for the ETH contract.
    pub fn new(cfg: ChainConfig) -> Result<Self> {
        Self::for_asset(cfg, Asset::Eth)
    }

    /// A client for the contract that trades `asset`.
    pub fn for_asset(cfg: ChainConfig, asset: Asset) -> Result<Self> {
        let (contract, token) = match asset {
            Asset::Eth => (parse_address(&cfg.contract)?, None),
            Asset::Usdc => {
                if !cfg.usdc_ready() {
                    bail!("USDC trading is not set up: this wallet knows no MidstateTokenSwap contract yet (Trade settings)");
                }
                (parse_address(&cfg.usdc_contract).context("the USDC swap contract")?, Some(parse_address(&cfg.usdc_token).context("the USDC token")?))
            }
        };
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .context("failed to build the HTTP client")?;
        Ok(Self { http, cfg, contract, asset, token })
    }

    /// The contract this client calls.
    pub fn contract(&self) -> [u8; 20] {
        self.contract
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

    /// The latest block: its number and its timestamp, the clock the
    /// contract's deadlines are judged by.
    pub async fn head(&self) -> Result<(u64, u64)> {
        let b = self.call("eth_getBlockByNumber", json!(["latest", false])).await?;
        let n = hex_u64(b.get("number").ok_or_else(|| anyhow!("the latest block has no number"))?)?;
        let t = hex_u64(b.get("timestamp").ok_or_else(|| anyhow!("the latest block has no timestamp"))?)?;
        Ok((n, t))
    }

    pub async fn balance(&self, addr: &[u8; 20]) -> Result<u128> {
        hex_u128(
            &self
                .call("eth_getBalance", json!([to_checksum_address(addr), "latest"]))
                .await?,
        )
    }

    /// Transactions from `addr` already mined: a nonce below it is used.
    pub async fn nonce_mined(&self, addr: &[u8; 20]) -> Result<u64> {
        hex_u64(&self.call("eth_getTransactionCount", json!([to_checksum_address(addr), "latest"])).await?)
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
        self.eth_call_to(&self.contract, data).await
    }

    async fn eth_call_to(&self, to: &[u8; 20], data: &[u8]) -> Result<Vec<u8>> {
        let v = self
            .call(
                "eth_call",
                json!([{
                    "to": to_checksum_address(to),
                    "data": format!("0x{}", hex::encode(data)),
                }, "latest"]),
            )
            .await?;
        hex_bytes(&v)
    }

    // ── The token ───────────────────────────────────────────────────────

    fn token(&self) -> Result<[u8; 20]> {
        self.token.ok_or_else(|| anyhow!("the ETH contract has no token"))
    }

    /// What `owner` holds of the token, in its smallest unit.
    pub async fn token_balance(&self, owner: &[u8; 20]) -> Result<u128> {
        let out = self.eth_call_to(&self.token()?, &encode_call("balanceOf(address)", &[Word::Address(*owner)])).await?;
        Ok(words(&out).first().map(word_u128).unwrap_or(0))
    }

    /// A permit for the contract to take `value` of the token from `key`'s
    /// account, standing for [`PERMIT_SECS`] from Base's `now`: under the
    /// token's own domain, at the account's next permit nonce.
    pub async fn permit(&self, key: &EvmKey, value: u128, now: u64) -> Result<Permit> {
        let token = self.token()?;
        let sep = self.eth_call_to(&token, &encode_call("DOMAIN_SEPARATOR()", &[])).await?;
        let sep: [u8; 32] = sep.get(..32).and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("the token takes no permits (no DOMAIN_SEPARATOR)"))?;
        let nonce = self.eth_call_to(&token, &encode_call("nonces(address)", &[Word::Address(key.address)])).await?;
        let nonce = words(&nonce).first().map(word_u128).ok_or_else(|| anyhow!("the token takes no permits (no nonces)"))?;
        key.sign_permit(&sep, &self.contract, value, nonce, now + PERMIT_SECS)
    }

    /// Build, sign and broadcast a contract call. Returns the transaction hash.
    pub async fn send(&self, key: &EvmKey, value: u128, data: Vec<u8>) -> Result<[u8; 32]> {
        let t = self.sign_call(key, value, data).await?;
        self.broadcast(&t.raw).await?;
        Ok(t.hash)
    }

    /// Build and sign a contract call without sending it, so its hash can be
    /// written down first: a send whose answer is lost is then still found.
    pub async fn sign_call(&self, key: &EvmKey, value: u128, data: Vec<u8>) -> Result<Signed> {
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
        Ok(Signed { hash: keccak256(&raw), raw, nonce })
    }

    /// Sends a signed transaction. Sending one the node already holds, or
    /// has mined, is harmless.
    pub async fn broadcast(&self, raw: &[u8]) -> Result<()> {
        match self.call("eth_sendRawTransaction", json!([format!("0x{}", hex::encode(raw))])).await {
            Ok(_) => Ok(()),
            Err(e) if format!("{e:#}").to_lowercase().contains("already known") => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Replaces the unmined transaction `raw` with an empty one to ourselves
    /// at its nonce, priced above it, so the original can never be mined.
    pub async fn cancel(&self, key: &EvmKey, raw: &[u8]) -> Result<[u8; 32]> {
        let (nonce, tip0, max0) = tx_fees(raw).ok_or_else(|| anyhow!("not a transaction this wallet signed"))?;
        let (tip, max_fee) = self.fees().await?;
        // Nodes replace a pending transaction only for more on both fees.
        let tip = tip.max(tip0 + tip0 / 8 + 1);
        let max_fee = max_fee.max(max0 + max0 / 8 + 1).max(tip);
        let tx = TxRequest {
            chain_id: self.cfg.chain_id,
            nonce,
            max_priority_fee: tip,
            max_fee,
            gas_limit: 21_000,
            to: key.address,
            value: 0,
            data: Vec::new(),
        };
        let raw = tx.sign(key)?;
        self.broadcast(&raw).await?;
        Ok(keccak256(&raw))
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
        self.logs_where(from, to, &[topic0]).await
    }

    /// Contract logs in a height range whose topics match `topics`
    /// position by position (`None` matches anything).
    pub async fn logs_where(&self, from: u64, to: u64, topics: &[Option<[u8; 32]>]) -> Result<Vec<Log>> {
        let mut filter = json!({
            "fromBlock": format!("0x{from:x}"),
            "toBlock": format!("0x{to:x}"),
            "address": to_checksum_address(&self.contract),
        });
        if topics.iter().any(Option::is_some) {
            let t: Vec<Value> = topics
                .iter()
                .map(|t| t.map_or(Value::Null, |t| json!(format!("0x{}", hex::encode(t)))))
                .collect();
            filter["topics"] = json!(t);
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
        self.send(key, wei, lock_call(hashlock, beneficiary, refund_delay)?).await
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
        self.send(key, bond, reserve_call(bid_id, fill_window_secs)).await
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
    ///
    /// All or nothing: an error if any part of the range could not be read.
    /// Callers that keep a cursor should use [`BaseClient::scan`], which keeps
    /// what it got through.
    pub async fn scan_events_counted(&self, from: u64, to: u64) -> Result<(Vec<(u64, Event)>, usize)> {
        let part = self.scan(from, to).await;
        if let Some(e) = part.error {
            return Err(e);
        }
        Ok((part.events, part.raw))
    }

    /// Reads the contract's events from `from` to `to`, in order, in ranges
    /// of at most [`LOG_SPAN`] blocks. A range that fails is halved and
    /// retried, down to [`LOG_SPAN_MIN`]; one that fails even then ends the
    /// scan there, and `reached` says how far it got. Nothing past `reached`
    /// is in `events`, so a caller that resumes from `reached + 1` never
    /// skips a block it could not read.
    pub async fn scan(&self, from: u64, to: u64) -> ScanPart {
        let mut part = ScanPart { events: Vec::new(), raw: 0, reached: from.saturating_sub(1), error: None };
        let mut cursor = from;
        let mut span = LOG_SPAN;
        while cursor <= to {
            let end = cursor.saturating_add(span - 1).min(to);
            match self.logs(cursor, end, None).await {
                Ok(logs) => {
                    for l in logs {
                        part.raw += 1;
                        if let Some(e) = decode_event(&l) {
                            part.events.push((l.block, e));
                        }
                    }
                    part.reached = end;
                    cursor = end + 1;
                    span = (span * 2).min(LOG_SPAN);
                }
                Err(_) if span > LOG_SPAN_MIN => span /= 2,
                Err(e) => {
                    part.error = Some(e);
                    break;
                }
            }
        }
        part
    }

    /// The secret a claim of `swap_id` published, looked for from `from` to
    /// `to`. `None` proves nothing: see [`BaseClient::find_settlement`].
    pub async fn find_claim(&self, swap_id: [u8; 32], from: u64, to: u64) -> Result<Option<[u8; 32]>> {
        Ok(match self.find_settlement(swap_id, from, to).await? {
            Some(Settlement::Claimed(p)) => Some(p),
            _ => None,
        })
    }

    /// How `swap_id` ended, looked for from `from` to `to` by the swap's own
    /// topic: its claim, with the secret, or its refund. For a wallet whose
    /// call was refused as "settled" and that missed the event. `None` if
    /// neither is in the range, which says nothing about which happened.
    pub async fn find_settlement(&self, swap_id: [u8; 32], from: u64, to: u64) -> Result<Option<Settlement>> {
        let topics = [None, Some(swap_id)];
        let mut cursor = from;
        let mut span = LOG_SPAN;
        while cursor <= to {
            let end = cursor.saturating_add(span - 1).min(to);
            match self.logs_where(cursor, end, &topics).await {
                Ok(logs) => {
                    for l in logs {
                        match decode_event(&l) {
                            Some(Event::Claimed { swap_id: id, preimage, .. }) if id == swap_id => {
                                return Ok(Some(Settlement::Claimed(preimage)));
                            }
                            Some(Event::Refunded { swap_id: id }) if id == swap_id => return Ok(Some(Settlement::Refunded)),
                            _ => {}
                        }
                    }
                    cursor = end + 1;
                    span = (span * 2).min(LOG_SPAN);
                }
                Err(_) if span > LOG_SPAN_MIN => span /= 2,
                Err(e) => return Err(e),
            }
        }
        Ok(None)
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
        // The struct's order: beneficiary, refundTo, amount, timeout,
        // hashlock, settled.
        Ok(Some(SwapState {
            beneficiary: word_address(&w[0]),
            refund_to: word_address(&w[1]),
            amount,
            timeout: word_u64(&w[3]),
            hashlock: w[4],
            settled: word_bool(&w[5]),
        }))
    }

    /// Read a bid back from contract storage: who reserved it, and until when.
    pub async fn get_bid(&self, bid_id: [u8; 32]) -> Result<Option<BidState>> {
        let out = self
            .eth_call(&encode_call("bids(bytes32)", &[Word::Bytes32(bid_id)]))
            .await?;
        let w = words(&out);
        if w.len() < 10 {
            return Ok(None);
        }
        let amount = word_u128(&w[2]);
        if amount == 0 {
            return Ok(None);
        }
        Ok(Some(BidState {
            maker: word_address(&w[0]),
            filler: word_address(&w[1]),
            amount,
            fill_bond: word_u128(&w[3]),
            mds_amount: word_u64(&w[4]),
            expiry: word_u64(&w[5]),
            fill_deadline: word_u64(&w[6]),
            hashlock: w[7],
            maker_mds_addr: w[8],
            settled: word_bool(&w[9]),
        }))
    }
}

/// Widest block range asked of the RPC in one `eth_getLogs`. Base's
/// documentation asks for under 2,000 blocks for reliable results; public
/// endpoints refuse or time out on wider ones.
pub const LOG_SPAN: u64 = 2_000;
/// A failing range is halved down to this before a scan stops for the tick.
pub const LOG_SPAN_MIN: u64 = 125;

/// How far a [`BaseClient::scan`] got.
pub struct ScanPart {
    /// Events up to and including `reached`, in chain order.
    pub events: Vec<(u64, Event)>,
    pub raw: usize,
    /// The last block read. Below the requested end when a range kept
    /// failing: resume from `reached + 1`.
    pub reached: u64,
    pub error: Option<anyhow::Error>,
}

/// A bid as the contract holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BidState {
    pub maker: [u8; 20],
    /// Zero while unreserved.
    pub filler: [u8; 20],
    pub amount: u128,
    pub fill_bond: u128,
    pub mds_amount: u64,
    pub expiry: u64,
    pub fill_deadline: u64,
    pub hashlock: [u8; 32],
    pub maker_mds_addr: [u8; 32],
    pub settled: bool,
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
    /// The contract that emitted it.
    pub address: [u8; 20],
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
    let address = v.get("address").and_then(|a| a.as_str()).and_then(|a| parse_address(a).ok()).unwrap_or([0; 20]);
    Ok(Log {
        address,
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

/// A transaction signed and not yet sent.
#[derive(Clone, Debug)]
pub struct Signed {
    pub raw: Vec<u8>,
    pub hash: [u8; 32],
    pub nonce: u64,
}

/// How a swap on the contract ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settlement {
    Claimed([u8; 32]),
    Refunded,
}

/// The call data of `lock()`.
pub fn lock_call(hashlock: [u8; 32], beneficiary: [u8; 20], refund_delay: u64) -> Result<Vec<u8>> {
    if !(600..=604_800).contains(&refund_delay) {
        bail!("the contract requires a refund delay between 10 minutes and 7 days");
    }
    Ok(encode_call(
        "lock(bytes32,address,uint256)",
        &[Word::Bytes32(hashlock), Word::Address(beneficiary), Word::U256(refund_delay as u128)],
    ))
}

/// The call data of `reserveBid()`.
pub fn reserve_call(bid_id: [u8; 32], fill_window_secs: u64) -> Vec<u8> {
    encode_call("reserveBid(bytes32,uint256)", &[Word::Bytes32(bid_id), Word::U256(fill_window_secs as u128)])
}

fn permit_words(p: &Permit) -> [Word; 4] {
    [Word::U64(p.deadline), Word::U64(p.v.into()), Word::Bytes32(p.r), Word::Bytes32(p.s)]
}

fn with_permit(sig: &str, mut args: Vec<Word>, permit: Option<&Permit>) -> Vec<u8> {
    match permit {
        Some(p) => {
            args.extend(permit_words(p));
            let name = sig.split('(').next().unwrap_or(sig);
            let params = &sig[name.len()..sig.len() - 1];
            encode_call(&format!("{name}WithPermit{params},uint256,uint8,bytes32,bytes32)"), &args)
        }
        None => encode_call(sig, &args),
    }
}

/// The call data of `MidstateTokenSwap.lock()`, or of `lockWithPermit()`
/// with a permit for exactly `amount`.
pub fn token_lock_call(hashlock: [u8; 32], beneficiary: [u8; 20], amount: u128, refund_delay: u64, permit: Option<&Permit>) -> Result<Vec<u8>> {
    if !(600..=604_800).contains(&refund_delay) {
        bail!("the contract requires a refund delay between 10 minutes and 7 days");
    }
    let args = vec![Word::Bytes32(hashlock), Word::Address(beneficiary), Word::U256(amount), Word::U256(refund_delay as u128)];
    Ok(with_permit("lock(bytes32,address,uint256,uint256)", args, permit))
}

/// The call data of `MidstateTokenSwap.createBid()`, or of
/// `createBidWithPermit()` with a permit for exactly `amount`.
pub fn token_bid_call(
    hashlock: [u8; 32],
    amount: u128,
    mds_amount: u64,
    maker_mds_addr: [u8; 32],
    ttl_secs: u64,
    fill_bond: u128,
    permit: Option<&Permit>,
) -> Result<Vec<u8>> {
    if !(3_600..=7_776_000).contains(&ttl_secs) {
        bail!("the contract requires a bid lifetime between 1 hour and 90 days");
    }
    let args = vec![
        Word::Bytes32(hashlock),
        Word::U256(amount),
        Word::U64(mds_amount),
        Word::Bytes32(maker_mds_addr),
        Word::U256(ttl_secs as u128),
        Word::U256(fill_bond),
    ];
    Ok(with_permit("createBid(bytes32,uint256,uint64,bytes32,uint256,uint256)", args, permit))
}

/// The call data of `MidstateTokenSwap.reserveBid()`, or of
/// `reserveBidWithPermit()` with a permit for exactly the bid's bond.
pub fn token_reserve_call(bid_id: [u8; 32], fill_window_secs: u64, permit: Option<&Permit>) -> Vec<u8> {
    with_permit("reserveBid(bytes32,uint256)", vec![Word::Bytes32(bid_id), Word::U256(fill_window_secs as u128)], permit)
}

/// `(nonce, max_priority_fee, max_fee)` of a signed type-2 transaction.
pub fn tx_fees(raw: &[u8]) -> Option<(u64, u128, u128)> {
    // An RLP item's header: its length and its payload's.
    fn head(b: &[u8]) -> Option<(usize, usize, bool)> {
        let be = |n: usize| -> Option<usize> {
            let bytes = b.get(1..1 + n)?;
            (n <= 8).then(|| bytes.iter().fold(0usize, |a, &x| a << 8 | x as usize))
        };
        let p = *b.first()?;
        Some(match p {
            0x00..=0x7f => (0, 1, false),
            0x80..=0xb7 => (1, (p - 0x80) as usize, false),
            0xb8..=0xbf => ((p - 0xb6) as usize, be((p - 0xb7) as usize)?, false),
            0xc0..=0xf7 => (1, (p - 0xc0) as usize, true),
            0xf8..=0xff => ((p - 0xf6) as usize, be((p - 0xf7) as usize)?, true),
        })
    }
    let (&ty, rest) = raw.split_first()?;
    let (h, n, list) = head(rest)?;
    if ty != 0x02 || !list {
        return None;
    }
    let mut body = rest.get(h..h + n)?;
    let mut ints = [0u128; 4]; // chain id, nonce, tip, max fee
    for v in &mut ints {
        let (h, n, list) = head(body)?;
        if list || n > 16 {
            return None;
        }
        *v = body.get(h..h + n)?.iter().fold(0u128, |a, &x| a << 8 | x as u128);
        body = body.get(h + n..)?;
    }
    Some((u64::try_from(ints[1]).ok()?, ints[2], ints[3]))
}

/// Pull our own swap id out of the receipt for a `lock()` call to
/// `contract`. A token's own logs in the same receipt are passed over.
pub fn locked_swap_id(logs: &[Log], contract: &[u8; 20]) -> Option<[u8; 32]> {
    logs.iter().filter(|l| &l.address == contract).find_map(|l| match decode_event(l) {
        Some(Event::Locked { swap_id, .. }) => Some(swap_id),
        _ => None,
    })
}

/// Our own bid id from the receipt for a `createBid()` call to `contract`.
pub fn created_bid_id(logs: &[Log], contract: &[u8; 20]) -> Option<[u8; 32]> {
    logs.iter().filter(|l| &l.address == contract).find_map(|l| match decode_event(l) {
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
    fn token_swap_calls_are_the_ones_solidity_reads() {
        // From ethers v6's Interface over MidstateTokenSwap's compiled ABI.
        let p = Permit {
            deadline: 1_800_000_000,
            v: 27,
            r: hex::decode("8a6127509bbf9e7582b9d46d4090d8f78e232878a6eb7e67702843552b23d470").unwrap().try_into().unwrap(),
            s: hex::decode("44ada3f00850cc586510c0e7443e1df64f987166e33ac1412fb7bc21e606b5ae").unwrap().try_into().unwrap(),
        };
        let (h, bid, mds, who) = ([0xab; 32], [0xcd; 32], [0xef; 32], [0x11; 20]);
        let hexed = |b: Vec<u8>| hex::encode(b);
        let tail = "000000000000000000000000000000000000000000000000000000006b49d200000000000000000000000000000000000000000000000000000000000000001b8a6127509bbf9e7582b9d46d4090d8f78e232878a6eb7e67702843552b23d47044ada3f00850cc586510c0e7443e1df64f987166e33ac1412fb7bc21e606b5ae";
        let lock_args = "abababababababababababababababababababababababababababababababab000000000000000000000000111111111111111111111111111111111111111100000000000000000000000000000000000000000000000000000000004c4b400000000000000000000000000000000000000000000000000000000000000e10";
        assert_eq!(hexed(token_lock_call(h, who, 5_000_000, 3600, None).unwrap()), format!("9561080b{lock_args}"));
        assert_eq!(hexed(token_lock_call(h, who, 5_000_000, 3600, Some(&p)).unwrap()), format!("e1888dc6{lock_args}{tail}"));
        let bid_args = "abababababababababababababababababababababababababababababababab00000000000000000000000000000000000000000000000000000000006acfc00000000000000000000000000000000000000000000000000000000040000000efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef00000000000000000000000000000000000000000000000000000000000151800000000000000000000000000000000000000000000000000000000000011170";
        assert_eq!(hexed(token_bid_call(h, 7_000_000, 1 << 30, mds, 86_400, 70_000, None).unwrap()), format!("23c3d88c{bid_args}"));
        assert_eq!(hexed(token_bid_call(h, 7_000_000, 1 << 30, mds, 86_400, 70_000, Some(&p)).unwrap()), format!("c1d95de2{bid_args}{tail}"));
        let res_args = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd0000000000000000000000000000000000000000000000000000000000003840";
        assert_eq!(hexed(token_reserve_call(bid, 14_400, None)), format!("512bc49c{res_args}"));
        assert_eq!(hexed(token_reserve_call(bid, 14_400, Some(&p))), format!("6f25ab80{res_args}{tail}"));
        // The ETH contract's reservation is the same call, paid in ETH.
        assert_eq!(reserve_call(bid, 14_400), token_reserve_call(bid, 14_400, None));
        assert!(token_lock_call(h, who, 1, 599, None).is_err() && token_bid_call(h, 1, 1, mds, 3_599, 0, None).is_err());
    }

    #[test]
    fn a_signed_transaction_gives_back_its_nonce_and_fees() {
        let key = EvmKey::from_secret(&[7; 32]).unwrap();
        for (nonce, tip, max_fee, data) in [(0u64, 0u128, 1u128, vec![]), (1, 127, 128, vec![1; 3]), (300, 1_000_000, 70_000_000_000, vec![9; 600])] {
            let tx = TxRequest { chain_id: 8453, nonce, max_priority_fee: tip, max_fee, gas_limit: 90_000, to: [3; 20], value: 5, data };
            let raw = tx.sign(&key).unwrap();
            assert_eq!(tx_fees(&raw), Some((nonce, tip, max_fee)));
            assert_eq!(keccak256(&raw), tx.hash(&key).unwrap());
        }
        assert_eq!(tx_fees(&[0x01, 0xc0]), None);
    }

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
        Log { address: [0xaa; 20], topics: t, data: d, block: 1 }
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
        assert_eq!(locked_swap_id(&[log.clone()], &[0xaa; 20]), Some([7u8; 32]));
        // Another contract's event of the same shape (a token's, a decoy's)
        // is not the lock.
        assert_eq!(locked_swap_id(&[log], &[0xbb; 20]), None);
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
        // The getter returns the struct in declaration order: beneficiary,
        // refundTo, amount, timeout, hashlock, settled (both contracts).
        let mut data = Vec::new();
        for w in [
            Word::Address([1; 20]),
            Word::Address([2; 20]),
            Word::U256(999),
            Word::U64(1234),
            Word::Bytes32([3; 32]),
            Word::Bool(true),
        ] {
            data.extend_from_slice(&w.encode());
        }
        let w = words(&data);
        assert_eq!(word_u128(&w[2]), 999);
        assert_eq!(word_u64(&w[3]), 1234);
        assert_eq!(w[4], [3u8; 32]);
        assert!(word_bool(&w[5]));
        assert_eq!(word_address(&w[0]), [1u8; 20]);
    }
}
