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
    /// Net value from the wallet's perspective where derivable.
    pub amount: u64,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    pub peers: Vec<String>,
    pub data_dir: String,
    pub rpc_url: Option<String>,
    pub block_reward: u64,
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
}
