//! Q-Bolt channel bookkeeping for walletd: durable records, inbound-open
//! staging, replay guarding, and the chat-frame parse helper. All protocol
//! *logic* (validation, signing, close/refund driving) lives in service.rs
//! where the wallet and node handles are; this module is the state it acts on.
//!
//! Persistence: a JSON sidecar next to wallet.dat (`<wallet>.channels.json`).
//! It holds channel metadata and counterparty signatures — nothing that
//! spends YOUR coins (your signing keys stay inside the encrypted wallet) —
//! but balances are visible to anyone who reads the file. Encrypting the
//! sidecar under the wallet password is a tracked follow-up.

use midstate::core::channel::FundingCoin;
use midstate::node::NodeHandle;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

pub const MIN_CAPACITY: u64 = 4096;
pub const DEFAULT_LIFETIME: u64 = 4320; // ~3 days
pub const MIN_LIFETIME: u64 = 360;
pub const MAX_LIFETIME: u64 = 43_200;
pub const CLOSE_MARGIN: u64 = 60; // receiver auto-closes at expiry − 60
pub const WARN_MARGIN: u64 = 240;
pub const PAY_CUTOFF: u64 = 90; // sender stops paying at expiry − 90
pub const MIN_LIFE_AT_ACCEPT: u64 = 180;
pub const OPEN_VERIFY_BLOCKS: u64 = 30;
pub const OPEN_REBROADCAST_EVERY: u64 = 10;
pub const UPDATE_REBROADCAST_EVERY: u64 = 5;
pub const REBROADCAST_MAX: u32 = 20;
pub const LEAF_RESERVE: u64 = 8; // always keep MSS leaves for closes/sweeps
pub const INVOICE_TTL: u64 = 720; // invoices expire ~12 h after minting
pub const JIT_MARGIN: u64 = 15; // extra headroom required to attempt a JIT open
pub const CLAIM_STALL_BLOCKS: u64 = 20; // preimage sent, no credit → force close
pub const MAX_OUTSTANDING_INVOICES: usize = 64;
/// How long a handed-out address stays claimable, in blocks (~1 day).
pub const ADDRESS_TTL: u64 = 1440;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Sender,
    Receiver,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChanStatus {
    /// Funding broadcast (sender) — waiting for coins on-chain + peer ACK.
    Opening,
    Active,
    /// Close in flight (receiver): commitment committed, reveal pending/spent.
    Closing {
        commitment: [u8; 32],
        receiver_sig: Vec<u8>,
        revealed: bool,
        started: u64,
    },
    Closed,
    /// Refund in flight (sender, post-expiry).
    Refunding {
        commitment: [u8; 32],
        sender_sig: Vec<u8>,
        revealed: bool,
        started: u64,
    },
    Refunded,
    Rejected(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelRecord {
    pub id: [u8; 32],
    pub role: Role,
    pub sender_pk: [u8; 32],
    pub receiver_pk: [u8; 32],
    pub expiry: u64,
    pub funding: Vec<FundingCoin>,
    pub capacity: u64,
    /// Latest state (nonce 0 = the OPEN state: everything to the sender).
    pub nonce: u32,
    pub sender_amt: u64,
    pub receiver_amt: u64,
    /// HTLCs live in the current state (each becomes a script output at close).
    #[serde(default)]
    pub htlcs: Vec<midstate::core::channel::Htlc>,
    /// The sender's MSS signature over the latest state's commitment.
    pub sender_sig: Vec<u8>,
    /// Hashes we sent a preimage for, awaiting a crediting state: hash → height.
    #[serde(default)]
    pub pending_claims: std::collections::HashMap<String, u64>,
    /// Hashes we have consented to remove uncredited: hash → height.
    #[serde(default)]
    pub failed_htlcs: std::collections::HashMap<String, u64>,
    /// Sender-side: peer has acknowledged the latest nonce.
    pub acked: bool,
    pub last_broadcast: u64,
    pub rebroadcasts: u32,
    pub opened_height: u64,
    pub refund_attempt: u32,
    pub status: ChanStatus,
}

impl ChannelRecord {
    pub fn peer_pk(&self, my_pk: &[u8; 32]) -> [u8; 32] {
        if &self.sender_pk == my_pk { self.receiver_pk } else { self.sender_pk }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingOpen {
    pub id: [u8; 32],
    pub sender_pk: [u8; 32],
    pub expiry: u64,
    pub funding: Vec<FundingCoin>,
    pub sig0: Vec<u8>,
    pub first_seen: u64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct ChannelBook {
    pub identity_pk: Option<[u8; 32]>,
    pub channels: Vec<ChannelRecord>,
    pub pending_opens: Vec<PendingOpen>,
    /// Replay guard over processed chat frames: (timestamp, pow-nonce, sender).
    #[serde(default)]
    pub seen: VecDeque<(u64, u64, String)>,
    /// Preimages we know: hash hex → secret hex (ours, or harvested from a
    /// claim we relayed — a hub needs them to sweep upstream on-chain).
    #[serde(default)]
    pub secrets: std::collections::HashMap<String, String>,
    /// Invoices we minted and still expect payment for: hash → record.
    #[serde(default)]
    pub invoices: std::collections::HashMap<String, Invoice>,
    /// Payments we initiated, awaiting preimage: hash → record.
    #[serde(default)]
    pub pay_pending: std::collections::HashMap<String, PayPending>,
    /// HTLCs we forwarded as a hub: hash → upstream channel we owe.
    #[serde(default)]
    pub routes: std::collections::HashMap<String, Route>,
    /// Forwards parked awaiting a just-in-time channel open: hash → record.
    #[serde(default)]
    pub parked: std::collections::HashMap<String, Parked>,
    /// Invoice requests we sent, awaiting a reply: request id → (payee, amount).
    #[serde(default)]
    pub inv_reqs: std::collections::HashMap<String, ([u8; 32], u64)>,
    /// Invoice requests we already answered (bounded; each answer costs a leaf).
    #[serde(default)]
    pub answered_reqs: std::collections::HashMap<String, u64>,
    #[serde(default)]
    pub hub: HubConfig,
    /// Sell orders this wallet has published. The per-unit secrets are the
    /// only part of an order that cannot be rebuilt from the chain — without
    /// them the maker can neither be paid nor reclaim early.
    #[serde(default)]
    pub orders: Vec<MyOrder>,
    /// Buy orders this wallet has escrowed on Base.
    #[serde(default)]
    pub bids: Vec<MyBid>,
    /// Hubs heard advertising on the bus: identity hex → what they claim.
    #[serde(default)]
    pub hubs: std::collections::HashMap<String, HubAd>,
    /// Height of our last self-advertisement, so it repeats but does not spam.
    #[serde(default)]
    pub last_hub_ad: u64,
    /// Address requests we sent, awaiting a signed reply: req id → peer pk.
    #[serde(default)]
    pub addr_reqs: std::collections::HashMap<String, [u8; 32]>,
    /// Address requests we have answered. Each answer costs one one-time
    /// signature, so the guard has to survive a restart.
    #[serde(default)]
    pub answered_addr_reqs: std::collections::HashMap<String, u64>,
    /// Fresh addresses peers handed us: peer pk hex → (address hex, expiry).
    #[serde(default)]
    pub peer_addrs: std::collections::HashMap<String, (String, u64)>,
}

/// Operating a routing hub. In a unidirectional channel every forward
/// permanently consumes outbound capacity toward that peer, so a hub is
/// really a capacity vendor: it must re-fund lanes as they drain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubConfig {
    /// Accept inbound channel opens automatically (they cost us nothing).
    pub auto_accept: bool,
    /// Forward HTLCs toward the next hop, keeping HOP_FEE.
    pub forward: bool,
    /// Fund a channel on demand when the final hop has none.
    pub jit_open: bool,
    /// Capacity to use for a JIT open when the routed amount is smaller.
    pub jit_capacity: u64,
    /// Refuse to forward if it would leave fewer than this many MSS leaves.
    pub min_leaves: u64,
    /// Open a channel when a peer asks for one over the bus. This is what lets
    /// a buyer trade instantly with a seller they have never met: they ask, and
    /// a seller who wants the sale funds the lane.
    #[serde(default)]
    pub auto_open_on_request: bool,
    /// Most this wallet will lock into a single unsolicited channel.
    #[serde(default)]
    pub max_auto_capacity: u64,
    /// Total outstanding capacity opened this way, so a stream of requests
    /// cannot drain the wallet one acceptable channel at a time.
    #[serde(default)]
    pub auto_capacity_budget: u64,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            auto_accept: true,
            forward: false,
            jit_open: false,
            jit_capacity: 65_536,
            min_leaves: 64,
            auto_open_on_request: false,
            max_auto_capacity: 65_536,
            auto_capacity_budget: 1_048_576,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Invoice {
    pub amount: u64,
    pub expiry: u64,
    pub hints: Vec<[u8; 32]>,
    /// Set once an inbound HTLC for this hash was claimed.
    #[serde(default)]
    pub paid: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayPending {
    pub total: u64,
    pub amount: u64,
    pub dest: [u8; 32],
    pub timeout: u64,
    pub at: u64,
    pub channel: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Route {
    /// Upstream channel (we are the receiver) that we owe on claim.
    pub upstream: [u8; 32],
    /// Amount of the UPSTREAM htlc (what we can pull if the preimage arrives).
    pub in_amount: u64,
    pub created: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parked {
    pub next_pk: [u8; 32],
    pub amount: u64,
    pub timeout: u64,
    pub upstream: [u8; 32],
    pub in_amount: u64,
    pub created: u64,
    pub remaining: Vec<[u8; 32]>,
}

impl ChannelBook {
    pub fn path_for(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("channels.json")
    }
    pub fn load(wallet_path: &Path) -> Self {
        std::fs::read_to_string(Self::path_for(wallet_path))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    pub fn save(&self, wallet_path: &Path) {
        if let Ok(s) = serde_json::to_string(self) {
            if let Err(e) = std::fs::write(Self::path_for(wallet_path), s) {
                tracing::error!("channel book save failed: {e}");
            }
        }
    }
    pub fn find_mut(&mut self, id: &[u8; 32]) -> Option<&mut ChannelRecord> {
        self.channels.iter_mut().find(|c| &c.id == id)
    }
    pub fn find(&self, id: &[u8; 32]) -> Option<&ChannelRecord> {
        self.channels.iter().find(|c| &c.id == id)
    }
    pub fn mark_seen(&mut self, ts: u64, nonce: u64, sender: &str) -> bool {
        if self.seen.iter().any(|(t, n, s)| *t == ts && *n == nonce && s == sender) {
            return false; // already processed
        }
        self.seen.push_back((ts, nonce, sender.to_string()));
        while self.seen.len() > 5000 {
            self.seen.pop_front();
        }
        true
    }
}

/// A parsed inbound qbolt chat frame.
pub struct Frame {
    pub cmd: u8,
    pub channel_id: Option<[u8; 32]>,
    pub payload: Option<Vec<u8>>,
    /// First Address attachment (sender pk on OPEN, target on INVOICE_REQ).
    pub address: Option<[u8; 32]>,
    /// All Address attachments in order — the remaining route for an HTLC.
    pub addresses: Vec<[u8; 32]>,
    /// A Midstate attachment carries an HTLC preimage on CLAIM.
    pub secret: Option<[u8; 32]>,
    pub ts: u64,
    pub pow_nonce: u64,
    pub sender: String,
}

/// Extract qbolt frames (`words = [255, cmd]`) from a chat message.
pub fn parse_frame(m: &midstate::chat::ChatMessage) -> Option<Frame> {
    use midstate::chat::ChatAttachment as A;
    if m.words.len() < 2 || m.words[0] != midstate::core::channel::wire::MARKER {
        return None;
    }
    let mut f = Frame {
        cmd: m.words[1],
        channel_id: None,
        payload: None,
        address: None,
        addresses: Vec::new(),
        secret: None,
        ts: m.timestamp,
        pow_nonce: m.nonce,
        sender: m.sender.clone(),
    };
    for a in &m.attachments {
        match a {
            A::CoinId(b) if f.channel_id.is_none() => f.channel_id = Some(*b),
            A::Signature(p) if f.payload.is_none() => f.payload = Some(p.clone()),
            A::Address(b) => {
                if f.address.is_none() {
                    f.address = Some(*b);
                }
                f.addresses.push(*b);
            }
            A::Midstate(b) if f.secret.is_none() => f.secret = Some(*b),
            _ => {}
        }
    }
    Some(f)
}

/// Build the attachments for an outbound frame.
pub fn frame_attachments(
    channel_id: [u8; 32],
    payload: Vec<u8>,
    address: Option<[u8; 32]>,
) -> Vec<midstate::chat::ChatAttachment> {
    use midstate::chat::ChatAttachment as A;
    let mut atts = vec![A::CoinId(channel_id), A::Signature(payload)];
    if let Some(a) = address {
        atts.push(A::Address(a));
    }
    atts
}

/// Send one qbolt wire frame over the node's chat (node mines the PoW).
pub fn send_frame(
    node: &NodeHandle,
    cmd: u8,
    channel_id: [u8; 32],
    payload: Vec<u8>,
    address: Option<[u8; 32]>,
) -> anyhow::Result<()> {
    node.send_chat(
        vec![midstate::core::channel::wire::MARKER, cmd],
        None,
        frame_attachments(channel_id, payload, address),
    )
}

/// A sell order published by this wallet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyOrder {
    pub group_id: String,
    /// Commitment of the transaction that funded and announced it.
    pub commitment: String,
    pub mds_amount: u64,
    pub wei_amount: String,
    pub timeout_height: u64,
    pub created_height: u64,
    /// (secret_hash, secret) per unit, hex. Revealing a secret is what
    /// releases that unit to a buyer.
    pub secrets: Vec<(String, String)>,
    /// (mds value, wei price) per unit, parallel to `secrets`.
    ///
    /// Units are power-of-two sized and priced in proportion, so they differ
    /// widely — a 512-unit may be worth 64x an 8-unit. Judging an incoming
    /// escrow against the order's average price would reject every small unit
    /// as underpaid, so the exact per-unit price has to be recorded.
    #[serde(default)]
    pub unit_prices: Vec<(u64, String)>,
}

/// A resting buy order escrowed in the Base contract.
///
/// The direction is inverted from a sell order: here WE hold the secret and
/// reveal it on midstate by claiming the seller's covenant, after which the
/// seller uses that preimage to collect our ETH. So the secret is what makes
/// the bid fillable at all — lose it and the escrow can only ever be
/// cancelled back.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MyBid {
    /// Assigned by the contract at mining time, so unknown until the receipt.
    #[serde(default)]
    pub bid_id: String,
    /// Transaction that created it, while the id is still unknown.
    pub tx: String,
    pub secret_hash: String,
    pub secret: String,
    pub mds_amount: u64,
    pub wei: String,
    pub fill_bond: String,
    /// Midstate address the seller's covenant must pay — ours.
    pub mds_addr: String,
    pub expiry: u64,
    pub created: u64,
    #[serde(default)]
    pub cancelled: bool,
}

/// A hub's self-description, as heard on the bus.
///
/// Unverified by construction — anyone can claim anything. It is a lead, not a
/// guarantee: the numbers only become real when a channel is actually opened,
/// and the wallet says so rather than presenting them as fact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubAd {
    pub outbound: u64,
    pub min_capacity: u64,
    pub hop_fee: u64,
    /// Midstate height we last heard from them, for staleness.
    pub heard: u64,
}
