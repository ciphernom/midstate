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
    #[serde(default)]
    pub sync_phase: String,
    #[serde(default)]
    pub sync_cursor: u64,
    #[serde(default)]
    pub sync_target: u64,
    /// Non-zero while a restore scan is finding this recovery phrase's earlier
    /// coins: the pass number. Signing is paused until it returns to zero.
    #[serde(default)]
    pub restore_pass: u32,
    /// Height the restore pass has scanned up to (0 when not restoring).
    #[serde(default)]
    pub restore_scan_pos: u64,
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

/// Local mining status. The desktop supervises child processes — optionally a
/// loopback Stratum pool, always a hasher — rather than mining in-process; see
/// `walletd::mining` for why.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MiningView {
    /// "idle" | "starting" | "running" | "stopping" | "error".
    ///
    /// Deliberately not a bool: the UI must distinguish "not mining" from
    /// "starting, please wait" from "failed, here is why", and startup can take
    /// several seconds while the pool binds.
    pub phase: String,
    /// Progress detail while starting or stopping, e.g. "waiting for the pool
    /// to accept connections…".
    pub message: Option<String>,
    /// Whether the `midstate` binary needed to mine could be located.
    pub available: bool,
    /// Checksummed hex address coinbase rewards are paid to.
    pub payout_address: String,
    /// Stratum URL in effect. Empty when idle.
    pub pool_url: String,
    /// Default to prefill the pool field with (this machine's own pool).
    pub default_pool_url: String,
    /// True when this process is also running the pool.
    pub local_pool: bool,
    /// Hasher threads; 0 means every core.
    pub threads: usize,
    /// Cores available, for bounding the thread control.
    pub max_threads: usize,
    /// Estimated hashes per second. Local pool only.
    pub hashrate: f64,
    /// Whole-network hash rate, from the pool's live block target.
    pub network_hashrate: f64,
    /// This wallet's estimated share of network hashrate, 0.0..=1.0.
    pub network_share: f64,
    /// Shares accepted / rejected this round. Local pool only.
    pub shares: u64,
    pub rejected: u64,
    /// Round score, and the pool's total, for the payout split.
    pub score: u64,
    pub total_score: u64,
    /// Blocks this wallet has found, and the pool's recent total.
    pub blocks_found: u64,
    pub pool_blocks: u64,
    pub active_miners: u64,
    /// Chain height and reward as the pool sees them.
    pub network_height: u64,
    pub block_reward: u64,
    /// Per-rig accepted-share tallies.
    pub workers: Vec<(String, u64)>,
    pub uptime_secs: u64,
    /// Set when a child process exited unexpectedly or startup failed.
    pub last_error: Option<String>,
    /// The MDW address the hasher names to its pool (empty without an MDW
    /// account): a pool that merge-mines midwimble pays it.
    #[serde(default)]
    pub mdw_address: String,
    /// The local pool is merge-mining MDW, signed with this wallet's bond.
    #[serde(default)]
    pub mdw_merge: bool,
    /// That bond's id (hex), while it is.
    #[serde(default)]
    pub mdw_bond: String,
    /// What the running (or next) mining does for MDW, in a sentence.
    #[serde(default)]
    pub mdw_note: String,
    /// A pool known to merge-mine midwimble, to offer.
    #[serde(default)]
    pub merge_pool: String,
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

/// One midwimble mining bond (see walletd `bond`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BondView {
    /// The bond coin's id (hex). Midwimble uses it as the bond's id too.
    pub id: String,
    pub label: String,
    pub value: u64,
    pub mining_key: String,
    /// The bond's owner key (hex): what `midwimble bond register` calls
    /// `--owner-pk`. Only this wallet's phrase can sign for it.
    pub owner_pk: String,
    pub bonded_until: u64,
    /// The address the coin is locked at, and the script behind it (hex).
    pub address: String,
    pub script: String,
    pub salt: String,
    /// One of "locking", "locked", "unlocked", "unbonding", "unbonded",
    /// "not_funded".
    pub status: String,
    /// The status in a sentence, including what midwimble makes of it.
    pub status_text: String,
    /// Blocks until it can be unbonded (0 once it can).
    pub blocks_left: u64,
    /// Unix time until which midwimble counts the bond for mining, if it
    /// ever does (0 if it never does).
    pub counts_until: u64,
    /// Below midwimble's minimum bond: never counts (a rehearsal).
    pub below_min: bool,
    /// `midwimble bond register …` with this bond's values filled in.
    pub register_cmd: String,
    /// `<script> <value> <salt>`, for importing into another wallet with the
    /// same recovery phrase.
    pub export_line: String,
    pub can_unbond: bool,
    pub can_remove: bool,
    /// Locked on this wallet's own mining key, so it mines from here.
    #[serde(default)]
    pub wallet_key: bool,
    /// Its registration with midwimble: "" (not asked) | "waiting" |
    /// "burying" | "ready" | "on_chain" | "failed".
    #[serde(default)]
    pub reg_phase: String,
    /// Burial progress, percent.
    #[serde(default)]
    pub reg_progress: u32,
    /// Where the registration stands, in a sentence.
    #[serde(default)]
    pub reg_note: String,
    /// The wallet can register it (locked, large enough, lock long enough).
    #[serde(default)]
    pub can_register: bool,
}

/// Midwimble's bond rules as they stand, and this wallet's mining key.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BondRulesView {
    /// This wallet's mining key (hex): bonds on it mine from this wallet.
    pub mining_key: String,
    /// The smallest bond midwimble counts now.
    pub min_bond: u64,
    /// What a new bond must hold at midwimble's difficulty now (the minimum
    /// until the price starts following the chain's work).
    pub registration_min: u64,
    pub max_bond: u64,
    /// Midstate blocks of its lock a bond must keep to count.
    pub min_lock_blocks: u64,
    /// The highest lock height midwimble registers now.
    pub horizon: u64,
    /// Midwimble's estimate of midstate's height now.
    pub est_midstate_height: u64,
    /// Registration reads midstate through this app's node RPC.
    pub rpc_on: bool,
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
    /// Recently completed trades, newest first.
    pub trades: Vec<TradeView>,
}

/// A trade that settled, reconstructed from contract events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeView {
    pub block: u64,
    pub wei: String,
    /// `None` when the MDS side cannot be resolved from what has been scanned.
    pub mds: Option<u64>,
    pub price: Option<f64>,
    /// "sell" — an ask was taken; "buy" — a bid was filled.
    pub kind: String,
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

/// A hub heard advertising itself on the chat bus.
///
/// Everything except `connected` is the hub's own claim about itself. Treat it
/// as a lead worth trying, not a measurement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubAdView {
    pub pk: String,
    pub outbound: u64,
    pub min_capacity: u64,
    pub hop_fee: u64,
    /// Midstate height it was last heard at.
    pub heard: u64,
    /// Whether we already have a channel to it — this part we know.
    pub connected: bool,
}

// ── MDS <-> MDW ─────────────────────────────────────────────────────────────

/// The embedded midwimble node, in brief (the ribbon, the Dashboard).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbNodeView {
    pub height: u64,
    pub peers: usize,
    /// Downloading a heavier chain: its tip is about to change.
    pub syncing: bool,
    /// The node's block explorer on this machine (`http://127.0.0.1:…/`),
    /// when it is being served.
    #[serde(default)]
    pub explorer: Option<String>,
}

/// The embedded midwimble node in full, for the Node tab.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbNodeInfo {
    pub height: u64,
    /// Unix time of the tip block.
    pub tip_timestamp: u64,
    pub header_hash: String,
    /// Running hash over every applied block.
    pub midstate: String,
    /// Cumulative work (u128, as a decimal string).
    pub depth: String,
    /// Leading zero bits of the next block's target.
    pub difficulty_bits: u32,
    /// The next block's reward, in midwimble units.
    pub block_reward: u64,
    /// MDW minted so far, in midwimble units.
    pub supply: u64,
    pub utxo_count: usize,
    /// Mining bonds registered and still able to sign.
    pub bonds: usize,
    pub mempool: usize,
    pub syncing: bool,
    pub peer_id: String,
    pub peers: Vec<MwbPeerView>,
    pub data_dir: String,
    pub p2p_port: u16,
    pub explorer: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbPeerView {
    pub id: String,
    pub addr: String,
    /// This node dialled it (rather than it dialling us).
    pub outbound: bool,
}

/// The MDW account. Amounts are midwimble units (10^-8 MDW).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MwbAccountView {
    /// Whether this wallet has an MDW account (wallets made before it
    /// existed get one by re-entering their recovery phrase).
    pub enabled: bool,
    /// Stealth address, as midwimble's wallets print it.
    pub address: Option<String>,
    pub spendable: u64,
    pub immature: u64,
    /// Set aside for a payment waiting for a block, or locked in a swap.
    pub reserved: u64,
    /// Next midwimble height the account will scan.
    pub scanned: u64,
    /// `None` when no midwimble node is running.
    pub node: Option<MwbNodeView>,
    /// Unspent coins, whatever holds them.
    #[serde(default)]
    pub coins: usize,
    /// Values of the coins a payment may spend now, largest first: what
    /// `mwb_account::send_fee` and `max_send` price a send from.
    #[serde(default)]
    pub spendable_coins: Vec<u64>,
}

/// One entry of the MDW account's history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbHistoryView {
    /// Incoming: "received" | "mined" | "swap" (bought in a swap) |
    /// "refund" (a swap lock of ours, back). Outgoing: "sent" |
    /// "swap lock" (put into a swap).
    pub kind: String,
    /// What arrived, or what was paid (the fee apart).
    pub amount: u64,
    pub fee: u64,
    pub incoming: bool,
    /// Its block, once it has one (for a failed payment, the block that
    /// spent its coins some other way).
    pub height: Option<u64>,
    /// Unix seconds: the block's time for a coin, when it was made for a
    /// payment (0 when not known).
    pub time: u64,
    /// "pending" | "confirmed" | "failed" | "maturing"
    pub status: String,
    /// Who a send paid.
    pub to: Option<String>,
    /// An incoming coin's commitment: its page in the explorer.
    pub output: Option<String>,
    /// A payment's kernel excess: its page in the explorer.
    pub kernel: Option<String>,
    pub swap: Option<String>,
}

/// An unspent coin of the MDW account.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbCoinView {
    pub commitment: String,
    pub value: u64,
    pub height: u64,
    /// Unix seconds (0 when not known).
    pub time: u64,
    pub coinbase: bool,
    /// "spendable" | "maturing" | "payment" (a payment waiting for a block
    /// spends it) | "swap" (set aside for a swap's lock)
    pub status: String,
    /// Where a mined coin becomes spendable.
    pub matures_at: Option<u64>,
}

/// An order heard on the chat bus (or one of ours).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbBookEntryView {
    pub order_id: String,
    /// "sell_mds": the maker sells MDS for MDW; "sell_mdw": the reverse.
    pub side: String,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    /// Midstate height after which it no longer stands.
    pub expires: u64,
    /// Midstate height it was last heard at.
    pub heard: u64,
    pub mine: bool,
}

/// An order this wallet made.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbOrderView {
    pub order_id: String,
    pub side: String,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    pub expires: u64,
    /// "open" | "taken" | "filled" | "cancelled" | "expired"
    pub status: String,
    pub swap: Option<String>,
}

/// A swap this wallet is in, or was.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MwbSwapView {
    pub id: String,
    /// "maker" | "taker"
    pub role: String,
    pub side: String,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    /// "negotiating" | "waiting" | "locking" | "locked" | "claiming" |
    /// "refunding" | "done" | "refunded" | "failed"
    pub phase: String,
    /// Where it stands, in a sentence.
    pub note: String,
    pub settled: bool,
    /// "MDS" or "MDW": what this side gives, and gets.
    pub gives: String,
    pub gets: String,
    /// Refund heights: the reveal leg's and the follow leg's, each on its
    /// own chain ("midwimble" or "midstate").
    pub reveal_height: u64,
    pub reveal_chain: String,
    pub follow_height: u64,
    pub follow_chain: String,
    /// Unix seconds.
    pub created: u64,
    pub updated: u64,
}
