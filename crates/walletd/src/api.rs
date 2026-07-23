//! Serializable types crossing the walletd ↔ UI boundary.
//!
//! Everything here is presentation-ready: addresses are checksummed hex,
//! ids are hex, amounts are u64 units. No key material ever crosses this
//! boundary except the one-time mnemonic reveal at wallet creation.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletStatus {
    /// A wallet file exists at the managed path.
    pub exists: bool,
    /// A wallet is currently open in memory.
    pub unlocked: bool,
    pub is_hd: bool,
    pub wallet_path: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Balance {
    /// Sum of coins currently live in the UTXO set.
    pub confirmed: u64,
    /// Sum of wallet coins the chain does not (yet) know about — freshly
    /// received and still unconfirmed, or stranded by a reorg.
    pub unconfirmed: u64,
    /// Value locked as inputs of live pending commits (in-flight sends).
    pub in_flight: u64,
    pub coin_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoinView {
    pub coin_id: String,
    /// 72-char checksummed hex address.
    pub address: String,
    pub value: u64,
    /// "wots" | "mss"
    pub kind: String,
    pub label: Option<String>,
    pub live: bool,
    /// True when this coin's one-time key has already produced a signature.
    pub wots_signed: bool,
    /// True when the coin is an input of a live pending commit.
    pub in_flight: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddressInfo {
    /// 72-char checksummed hex.
    pub address: String,
    /// "wots" (single-use) | "mss" (reusable, bounded).
    pub kind: String,
    pub label: Option<String>,
    /// For MSS: signatures remaining on this key. None for WOTS.
    pub remaining_sigs: Option<u64>,
    /// A coin has already arrived at this address (WOTS: stop sharing it).
    pub used: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryView {
    /// "sent" | "received" | "mixed" | "coinbase" | "consolidate"
    pub kind: String,
    pub fee: u64,
    pub timestamp: u64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Value that arrived in this wallet (receives, mining, sweeps). Priced
    /// from the ledger, so it does not change when those coins are later spent.
    pub amount: u64,
    /// What actually left the wallet on a send, recorded at the time. `None`
    /// for sends made before this was tracked — the chain stores no amounts,
    /// so it cannot be reconstructed afterwards.
    pub sent: Option<u64>,
    /// Destination of a recorded send.
    pub to: Option<String>,
    /// Change that came back to this wallet on a send.
    pub change: u64,
    pub n_in: usize,
    pub n_out: usize,
    /// How many of the outputs are ours. For a send, `n_out - ours_out` is
    /// the number that left the wallet.
    pub ours_out: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SyncStatus {
    pub height: u64,
    pub is_syncing: bool,
    pub peer_count: usize,
    pub mempool: usize,
    pub safe_depth: u64,
    pub num_coins: usize,
    pub num_commitments: usize,
    /// Chain tip midstate hash (hex) — ambient identity, shown on the Node screen.
    pub midstate: String,
    /// Expected current chain height, estimated from the tip timestamp and
    /// the 60-second block target. Denominator for sync progress; equals
    /// `height` once synced.
    pub est_target_height: u64,
    pub timestamp: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeInfo {
    pub peers: Vec<String>,
    pub data_dir: String,
    pub rpc_url: Option<String>,
    pub block_reward: u64,
    // ── Chain tip ──────────────────────────────────────────────────────
    pub height: u64,
    /// Unix time of the tip block.
    pub tip_timestamp: u64,
    pub header_hash: String,
    pub midstate: String,
    /// Cumulative chain work (u128, rendered as a decimal string).
    pub depth: String,
    /// Leading zero bits required by the current target.
    pub difficulty_bits: u32,
    // ── Accumulator sizes ──────────────────────────────────────────────
    /// Live coins in the whole chain's UTXO set.
    pub utxo_count: usize,
    /// Commitments awaiting their reveal.
    pub commitment_count: usize,
    /// One-time keys retired chain-wide.
    pub burned_count: usize,
    // ── Local ──────────────────────────────────────────────────────────
    pub mempool: usize,
    pub safe_depth: u64,
}

/// Stages of the two-phase send. Persisted implicitly via the wallet's
/// PendingCommit records; walletd re-derives the stage on resume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SendStage {
    /// Grinding commit PoW / broadcasting the commitment.
    Committing,
    /// Commitment broadcast; waiting for it to enter chain state.
    CommitPending,
    /// Commitment mined; waiting out the reveal delay (privacy or safety).
    WaitingReveal,
    /// Reveal signed and broadcast; waiting for inputs to leave the UTXO set.
    RevealPending,
    /// Inputs spent on-chain — the send is confirmed and recorded.
    Confirmed,
    /// Commit was not mined within the patience window; reveal later or abandon.
    Stalled,
    /// Signing or broadcast failed; coins remain unspent. Detail says why.
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SendProgress {
    /// Commitment hash (hex) — the send's identity end to end.
    pub id: String,
    pub stage: SendStage,
    pub detail: String,
    pub amount: u64,
    pub fee: u64,
    pub to: String,
    pub updated_at: u64,
}

/// Push events emitted over the broadcast channel and forwarded to the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WalletEvent {
    /// Once per poll tick (~2 s): current chain/network status.
    NodeTick { status: SyncStatus },
    /// Wallet contents changed (balance / coins / history are dirty).
    WalletChanged,
    /// A send advanced to a new stage.
    SendUpdate { progress: SendProgress },
    /// New coins detected for our addresses during scanning.
    Incoming { total_value: u64, count: usize, height: u64 },
    /// A payment-channel lifecycle event worth surfacing (open, payment
    /// received, close settled, refund, warnings).
    ChannelNotice { text: String },
    /// Something the person needs to know about that is not an error in any
    /// action they took — e.g. funds arriving at an unusable address.
    Warning { text: String },
    /// A peer answered an address request with a fresh, signature-verified
    /// destination.
    PeerAddress { peer: String, address: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatView {
    /// Sender peer id (base58) — the node identity that mined the message.
    pub sender: String,
    /// Decoded dictionary words joined with spaces.
    pub text: String,
    pub timestamp: u64,
    pub nonce: u64,
    pub reply_to: Option<u64>,
    /// Attachment count (payloads are protocol-level; qbolt channel messages
    /// will ride here later).
    pub attachments: usize,
}

/// Secret material for moving one coin between wallets. Anyone holding these
/// values controls the coin.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoinExport {
    pub coin_id: String,
    pub address: String,
    pub value: u64,
    pub seed: String,
    pub salt: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdentityView {
    /// This wallet's channel identity: an MSS master public key (hex).
    pub pk: String,
    /// One-time signatures left on the identity key (each off-chain state
    /// costs one; the wallet reserves 8 for closes).
    pub remaining_sigs: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelView {
    pub id: String,
    /// "sender" | "receiver"
    pub role: String,
    /// Counterparty MSS pk (hex).
    pub peer: String,
    pub capacity: u64,
    pub sender_amt: u64,
    pub receiver_amt: u64,
    /// What this wallet could spend/claim from the latest state.
    pub my_balance: u64,
    pub nonce: u32,
    /// Sender-side: latest state acknowledged by the peer.
    pub acked: bool,
    /// In-flight hash-locked payments riding this channel.
    pub htlcs: Vec<HtlcView>,
    pub expiry: u64,
    pub blocks_left: i64,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HtlcView {
    pub hash: String,
    pub amount: u64,
    /// Block height after which the sender can reclaim it.
    pub timeout: u64,
    /// We revealed the preimage and are waiting to be credited.
    pub claiming: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvoiceView {
    /// Shareable invoice string (`l2inv1:pk:hash:amount:expiry:hints`).
    pub text: String,
    pub hash: String,
    pub amount: u64,
    pub expiry: u64,
    pub hints: Vec<String>,
    /// Amount actually received, once an inbound HTLC was claimed.
    pub paid: Option<u64>,
}

/// Routing-hub settings. A unidirectional channel's capacity is CONSUMED by
/// forwarding — it never refills from return traffic — so a hub is a capacity
/// vendor that must re-fund lanes as they drain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubView {
    pub auto_accept: bool,
    pub forward: bool,
    pub jit_open: bool,
    pub jit_capacity: u64,
    pub min_leaves: u64,
    /// Fund a channel when a peer asks over the bus. Sellers need this for
    /// buyers to trade with them instantly.
    pub auto_open_on_request: bool,
    pub max_auto_capacity: u64,
    pub auto_capacity_budget: u64,
}

/// The wallet's Base account, derived from the same recovery phrase at the
/// standard BIP44 path so it also restores in MetaMask.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvmAccountView {
    /// EIP-55 checksummed address.
    pub address: String,
    /// Balance in wei; `None` if the RPC could not be reached.
    pub balance_wei: Option<String>,
    pub chain_id: u64,
    pub rpc_url: String,
    pub contract: String,
    /// Set when the wallet predates EVM key derivation.
    pub missing_key: bool,
}

/// A resting buy order: ETH escrowed in the contract, waiting for MDS.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BidView {
    pub bid_id: String,
    pub maker: String,
    pub wei: String,
    pub mds_amount: u64,
    /// Wei per MDS unit — the comparable figure across order sizes.
    pub price: f64,
    pub fill_bond: String,
    pub expiry: u64,
    pub reserved: bool,
    pub takeable: bool,
    pub mine: bool,
}

/// One independently-takeable unit of an ask.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AskUnitView {
    /// Index into the ask's live units, which is what `take_ask` expects.
    pub index: usize,
    pub mds: u64,
    pub wei: String,
}

/// A maker's sell order, announced on midstate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AskView {
    pub group_id: String,
    pub maker_evm: String,
    pub height: u64,
    pub timeout_height: u64,
    /// Units still unspent on-chain.
    pub live_units: usize,
    pub total_units: usize,
    pub mds_value: u64,
    pub wei: String,
    pub price: f64,
    pub mine: bool,
    /// The still-unsold units, each takeable on its own.
    pub units: Vec<AskUnitView>,
    /// Maker's channel identity, needed to ask them for a lane.
    pub maker_mds_pk: String,
    /// How MDS could reach you from this maker: "direct", "hub", or "none".
    ///
    /// A Spilman channel carries value one way only, so instant settlement
    /// needs a lane pointing FROM the maker TO you. You cannot open that
    /// yourself — hence "none" is answered with a request, not an action.
    pub route: String,
    /// Inbound capacity on the direct lane, when there is one.
    pub route_capacity: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderBookView {
    pub bids: Vec<BidView>,
    pub asks: Vec<AskView>,
    /// Highest Base block folded into the book.
    pub base_cursor: u64,
    pub mds_cursor: u64,
    pub last_error: Option<String>,
    /// What the last scan decoded. An empty book with zero events means the
    /// range held no activity; an empty book with events means they were all
    /// closed, and undecoded logs would mean a signature mismatch.
    pub bids_created: usize,
    pub bids_closed: usize,
    pub locks: usize,
    pub claims: usize,
    pub undecoded_logs: usize,
    pub announcements: usize,
}

/// Where the EVM leg points. Editable so the same build can follow a contract
/// redeploy or a different endpoint without a rebuild.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DexConfigView {
    pub rpc_url: String,
    pub chain_id: u64,
    pub contract: String,
    pub confirmations: u64,
    /// How many Base blocks back to scan on a cold start.
    pub scan_window: u64,
    /// Scan from this exact block instead of the window. 0 = use the window.
    pub start_block: u64,
}

/// One prerequisite for a swap, phrased so it can be acted on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckView {
    pub label: String,
    pub ok: bool,
    pub detail: String,
    /// What to do about it. Present only when the check fails.
    pub fix: Option<String>,
}

/// Deadlines for both legs of a swap, already reconciled against each other.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimingView {
    pub eth_refund_secs: u64,
    pub eth_deadline: u64,
    pub mds_timeout_height: u64,
    pub mds_deadline_est: u64,
    /// Slack between the two legs after allowing for block-time drift.
    pub margin_secs: u64,
}

/// Everything the guided flow needs to show before anything is signed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwapQuoteView {
    /// "buy" | "sell"
    pub side: String,
    /// "submarine" | "onchain"
    pub rail: String,
    pub mds_amount: u64,
    pub wei_amount: String,
    /// Estimated gas the Base leg needs, on top of any value sent.
    pub gas_estimate_wei: String,
    pub checks: Vec<CheckView>,
    pub ready: bool,
    /// `None` when the requested timing cannot be made safe.
    pub timings: Option<TimingView>,
    pub timing_error: Option<String>,
    /// Plain-language walkthrough of what will happen, in order. The desktop
    /// client renders its own copy so the explanation is visible before any
    /// terms are entered; this is here for other consumers of the API.
    pub steps: Vec<String>,
}

/// A sell order this wallet published.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyOrderView {
    pub group_id: String,
    pub mds_amount: u64,
    pub wei_amount: String,
    pub timeout_height: u64,
    pub created_height: u64,
    pub units: usize,
    /// Where the publishing transaction has got to. An order is only real once
    /// its reveal is mined — the covenant outputs and the announcement burns
    /// are both in that reveal, so before it lands there is nothing on-chain
    /// for anyone (including this wallet) to find.
    pub stage: String,
    pub detail: String,
    /// True once the announcement has actually been read back off the chain.
    pub on_chain: bool,
}

/// A live or finished cross-chain swap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwapView {
    pub id: String,
    /// "taker" | "maker"
    pub role: String,
    /// Plain-language stage.
    pub phase: String,
    pub detail: String,
    pub mds_value: u64,
    pub wei: String,
    pub counterparty: String,
    pub eth_deadline: u64,
    pub settled: bool,
    /// Base transaction hash for the current step, when there is one.
    pub tx: Option<String>,
}

/// A buy order this wallet has escrowed on Base.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyBidView {
    pub bid_id: String,
    pub tx: String,
    pub mds_amount: u64,
    pub wei: String,
    pub fill_bond: String,
    pub expiry: u64,
    /// "confirming" until the contract assigns an id, then "open".
    pub status: String,
    pub cancelled: bool,
}
