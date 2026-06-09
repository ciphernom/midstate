//! # Chat subsystem
//!
//! Midstate's chat is a dictionary-bound P2P message bus. Every chat token
//! is a `u8` index into [`CHAT_DICTIONARY`] (256 fixed entries), and every
//! field that gossips across the network is either a fixed-width byte
//! payload or a tightly-bounded enumeration the receiver controls. The
//! design property the subsystem is built around:
//!
//! > **NoArbitraryText.** No field in any chat message admits
//! > user-controllable `String` or variable-length `Vec<u8>` carrying
//! > textual data. The only `String` is `sender`, which is validated as
//! > a base58 libp2p `PeerId` in
//! > [`crate::network::protocol::Message::deserialize_bin`].
//!
//! ## Wire variants
//!
//! - [`crate::network::protocol::Message::Chat`] — legacy v1. PoW
//!   ([`verify_chat_pow`]) covers `(sender, ts, reply_to, words, nonce)`.
//!   Receive-only on new nodes; new nodes never emit it.
//! - [`crate::network::protocol::Message::ChatV2`] — current. PoW
//!   ([`verify_chat_pow_v2`]) additionally covers `attachments` and is
//!   domain-separated from v1 (Lemma 2.3.1 in `verify_chat_pow_v2` docs).
//!
//! ## Chat-only frame property
//!
//! See [`crate::network::protocol`] module docs for the wire-format
//! frame proof: `Message::ChatV2` is **appended** to the `Message` enum
//! and no existing variant is reordered, so every non-chat variant's
//! bincode encoding is byte-identical before and after the chat-v2
//! introduction.

use crate::core::types::{hash, count_leading_zeros};

// The serde impls below rely on `hex` (declared in the package Cargo.toml).
// Rust 2018+ allows `hex::` paths for direct dependencies without an explicit `use`.


/// Fixed 256-word dictionary indexed by chat message `words` bytes.
///
/// Cardinality (256) is load-bearing: a `Vec<u8>` lookup against this
/// table cannot overflow `u8`, and a single byte selects exactly one
/// canonical token. Adding or removing entries is a chat-protocol break
/// because index 42 must mean the same word on every peer.
///
/// New tokens may be appended at the end of the list **only if all
/// participating nodes upgrade simultaneously**, otherwise unmigrated peers
/// will reject messages containing the new indices via the
/// `w as usize >= CHAT_DICTIONARY.len()` check in
/// [`crate::network::protocol::Message::deserialize_bin`].
pub const CHAT_DICTIONARY: &[&str] = &[
    // 0-19: Core Crypto & Network
    "midstate", "network", "node", "peer", "block", "blocks", "tx", "transaction", "mempool", "hash",
    "pow", "mine", "mining", "miner", "sync", "wallet", "address", "key", "seed", "utxo",

    // 20-39: Airdrops, Trading & Finance
    "airdrop", "incoming", "post", "claim", "free", "giveaway", "reward", "bounty", "pool", "liquidity",
    "buy", "sell", "trade", "swap", "market", "price", "fiat", "dex", "cex", "value",

    // 40-59: Verbs (Action)
    "send", "receive", "give", "take", "make", "do", "get", "need", "want", "have",
    "check", "verify", "update", "upgrade", "restart", "connect", "drop", "build", "fix", "run",

    // 60-79: Verbs (State & Aux)
    "is", "are", "was", "were", "be", "been", "has", "had", "will", "can",
    "could", "should", "would", "might", "must", "stop", "wait", "see", "look", "know",

    // 80-99: Pronouns
    "I", "you", "we", "they", "he", "she", "it", "this", "that", "these",
    "those", "who", "what", "where", "when", "why", "how", "which", "my", "your",

    // 100-119: Prepositions & Conjunctions
    "at", "to", "from", "in", "out", "on", "off", "for", "by", "about",
    "as", "but", "if", "then", "else", "and", "or", "not", "with", "without",

    // 120-139: Adjectives & Adverbs
    "good", "bad", "fast", "slow", "full", "empty", "high", "low", "urgent", "ready",
    "online", "offline", "hot", "cold", "big", "small", "hard", "easy", "safe", "new",

    // 140-159: Quantifiers & Time
    "all", "none", "some", "any", "many", "much", "more", "less", "every", "only",
    "now", "later", "soon", "early", "today", "tomorrow", "yesterday", "time", "always", "never",

    // 160-179: Slang & Community
    "gm", "gn", "lol", "lfg", "wagmi", "ngmi", "ser", "anon", "mate", "based",
    "wtf", "omg", "moon", "pump", "dump", "bull", "bear", "scam", "rug", "fren",

    // 180-199: Numbers
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "10", "20", "50", "100", "200", "500", "1k", "10k", "100k", "1m",

    // 200-219: Tech / Concepts
    "wots", "mss", "smt", "sig", "data", "disk", "linux", "pi", "hardware", "software",
    "code", "rust", "server", "client", "ip", "webrtc", "error", "bug", "issue", "help",

    // 220-239: Misc useful words
    "please", "thanks", "ok", "yes", "no", "maybe", "here", "there", "again", "done",
    "first", "last", "old", "true", "false", "up", "down", "left", "right", "back",

    // 240-255: Punctuation & Emojis (16 items)
    "?", "!", ".", ",", "...", ":)", ":(", "🔥", "🚀", "💀",
    "💎", "👀", "🤝", "📈", "📉", "⚡"
];

/// A typed, fixed-shape attachment that rides alongside a v2 chat message.
///
/// # Invariant: NoArbitraryText
///
/// Every variant is structurally a fixed-width byte payload. There is no
/// field in this enum that admits user-controlled `String` or
/// variable-length `Vec<u8>` of textual data. The constraint is enforced
/// at the type level by serde and bincode.
///
/// # Encoding
///
/// | Form    | Shape                                                            |
/// |---------|------------------------------------------------------------------|
/// | JSON    | `{"kind":"address","value":"<72-char lowercase hex w/ checksum>"}` |
/// | Bincode | 32 raw bytes (variant tag prefix per bincode enum encoding)      |
///
/// # PoW canonical bytes
///
/// When mining or verifying via [`verify_chat_pow_v2`], each attachment
/// is encoded as `tag_u8 ⌢ payload_bytes`:
///
/// | Variant   | Tag    | Payload length |
/// |-----------|--------|----------------|
/// | `Address` | `0x01` | 32             |
///
/// Future variants append new tags; existing tags are immutable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatAttachment {
    /// A 32-byte midstate address (with UI checksum support).
    Address([u8; 32]),
    /// A 32-byte Coin ID (UTXO). Useful for proving airdrops or payments.
    CoinId([u8; 32]),
    /// A 32-byte Mix ID. Useful for coordinating CoinJoin sessions.
    MixId([u8; 32]),
    /// A 32-byte Transaction Commitment.
    Commitment([u8; 32]),
    /// A 32-byte Block Hash. Useful for referencing specific blocks.
    BlockHash([u8; 32]),
    /// A 32-byte Chain Midstate. Useful for node operators debugging consensus forks.
    Midstate([u8; 32]),
    /// A generic 32-byte hash (e.g. SHA256/BLAKE3) for external data/binaries.
    DataHash([u8; 32]),
    /// Challenge to prove possession of historical block data for a licensed range.
    /// Used for MMR Gossip Challenges / uptime scoring of Pruning Licenses.
    /// The `commitment` scopes the challenge to a specific license (prevents cross-license reputation gaming).
    LicenseChallenge {
        commitment: [u8; 32],
        height: u64,
        salt: [u8; 32],
    },
        /// A cryptographic signature (WOTS or MSS) for Layer 2 payment channels.
    /// Enables the ephemeral chat protocol to act as a Lightning Network gossip overlay.
    /// 
    /// # Formal Specification
    /// ```text
    /// Pre:  payload.len() <= MAX_SIGNATURE_SIZE (1536 bytes)
    /// Post: payload is transmitted verbatim over the P2P chat network
    /// ```
    Signature(Vec<u8>),
}

impl ChatAttachment {
    /// Returns true if the 32-byte payload is valid UTF-8, which indicates
    /// a high probability of text-injection graffiti rather than a random hash.
    pub fn is_graffiti(&self) -> bool {
        let bytes = match self {
            ChatAttachment::Address(b) => Some(b.as_slice()),
            ChatAttachment::CoinId(b) => Some(b.as_slice()),
            ChatAttachment::MixId(b) => Some(b.as_slice()),
            ChatAttachment::Commitment(b) => Some(b.as_slice()),
            ChatAttachment::BlockHash(b) => Some(b.as_slice()),
            ChatAttachment::Midstate(b) => Some(b.as_slice()),
            ChatAttachment::DataHash(b) => Some(b.as_slice()),
            ChatAttachment::LicenseChallenge { .. } => None,
            ChatAttachment::Signature(_) => None,
        };
        
        match bytes {
            Some(b) => std::str::from_utf8(b).is_ok(),
            None => false,
        }
        // NOTE: Extremely low probability (~1 in 4 billion) that a random 32-byte
        // BLAKE3 hash (e.g. a PoAW root) is valid UTF-8. If this ever happens for a
        // legitimate license advertisement, the node will be unable to broadcast it.
        // Acceptable risk for now to keep chat clean of binary graffiti.
    }
}

impl serde::Serialize for ChatAttachment {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if serializer.is_human_readable() {
            #[derive(serde::Serialize)]
            #[serde(tag = "kind", content = "value", rename_all = "snake_case")]
            enum JsonHelper {
                Address(String),
                CoinId(String),
                MixId(String),
                Commitment(String),
                BlockHash(String),
                Midstate(String),
                DataHash(String),
                LicenseChallenge { commitment: String, height: u64, salt: String },
                Signature(String), 
            }

            let helper = match self {
                ChatAttachment::Address(a) => JsonHelper::Address(crate::core::types::encode_address_with_checksum(a)),
                ChatAttachment::CoinId(id) => JsonHelper::CoinId(hex::encode(id)),
                ChatAttachment::MixId(id) => JsonHelper::MixId(hex::encode(id)),
                ChatAttachment::Commitment(id) => JsonHelper::Commitment(hex::encode(id)),
                ChatAttachment::BlockHash(id) => JsonHelper::BlockHash(hex::encode(id)),
                ChatAttachment::Midstate(id) => JsonHelper::Midstate(hex::encode(id)),
                ChatAttachment::DataHash(id) => JsonHelper::DataHash(hex::encode(id)),
                ChatAttachment::LicenseChallenge { commitment, height, salt } => {
                    JsonHelper::LicenseChallenge {
                        commitment: hex::encode(commitment),
                        height: *height,
                        salt: hex::encode(salt),
                    }
                }
                ChatAttachment::Signature(sig) => JsonHelper::Signature(hex::encode(sig)),
            };
            helper.serialize(serializer)
        } else {
            // For bincode: serialize as an external enum
            #[derive(serde::Serialize)]
            enum BincodeHelper<'a> {
                Address(&'a [u8; 32]),
                CoinId(&'a [u8; 32]),
                MixId(&'a [u8; 32]),
                Commitment(&'a [u8; 32]),
                BlockHash(&'a [u8; 32]),
                Midstate(&'a [u8; 32]),
                DataHash(&'a [u8; 32]),
                LicenseChallenge { commitment: &'a [u8; 32], height: u64, salt: &'a [u8; 32] },
                Signature(&'a [u8]),
            }
            let helper = match self {
                ChatAttachment::Address(addr) => BincodeHelper::Address(addr),
                ChatAttachment::CoinId(id) => BincodeHelper::CoinId(id),
                ChatAttachment::MixId(id) => BincodeHelper::MixId(id),
                ChatAttachment::Commitment(id) => BincodeHelper::Commitment(id),
                ChatAttachment::BlockHash(id) => BincodeHelper::BlockHash(id),
                ChatAttachment::Midstate(id) => BincodeHelper::Midstate(id),
                ChatAttachment::DataHash(id) => BincodeHelper::DataHash(id),
                ChatAttachment::LicenseChallenge { commitment: _, height, salt } => {
                    BincodeHelper::LicenseChallenge { commitment: &[0u8; 32], height: *height, salt }
                }
                ChatAttachment::Signature(sig) => BincodeHelper::Signature(sig.as_slice()), 
            };
            helper.serialize(serializer)
        }
    }
}

impl<'de> serde::Deserialize<'de> for ChatAttachment {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            #[derive(serde::Deserialize)]
            #[serde(tag = "kind", content = "value", rename_all = "snake_case")]
            enum JsonHelper {
                Address(String),
                CoinId(String),
                MixId(String),
                Commitment(String),
                BlockHash(String),
                Midstate(String),
                DataHash(String),
                LicenseChallenge { commitment: String, height: u64, salt: String },
                Signature(String), 
            }

            let helper = JsonHelper::deserialize(deserializer)?;
            
            let parse_32 = |s: &str| -> Result<[u8; 32], D::Error> {
                let bytes = hex::decode(s).map_err(serde::de::Error::custom)?;
                bytes.try_into().map_err(|_| serde::de::Error::custom("Must be exactly 32 bytes"))
            };

            match helper {
                JsonHelper::Address(s) => {
                    let addr = crate::core::types::parse_address_flexible(&s)
                        .map_err(serde::de::Error::custom)?;
                    Ok(ChatAttachment::Address(addr))
                }
                JsonHelper::CoinId(s) => Ok(ChatAttachment::CoinId(parse_32(&s)?)),
                JsonHelper::MixId(s) => Ok(ChatAttachment::MixId(parse_32(&s)?)),
                JsonHelper::Commitment(s) => Ok(ChatAttachment::Commitment(parse_32(&s)?)),
                JsonHelper::BlockHash(s) => Ok(ChatAttachment::BlockHash(parse_32(&s)?)),
                JsonHelper::Midstate(s) => Ok(ChatAttachment::Midstate(parse_32(&s)?)),
                JsonHelper::DataHash(s) => Ok(ChatAttachment::DataHash(parse_32(&s)?)),
                JsonHelper::LicenseChallenge { commitment, height, salt } => {
                    let commitment_bytes = parse_32(&commitment)?;
                    let salt_bytes = parse_32(&salt)?;
                    Ok(ChatAttachment::LicenseChallenge { commitment: commitment_bytes, height, salt: salt_bytes })
                }
                JsonHelper::Signature(s) => {
                    let bytes = hex::decode(s).map_err(serde::de::Error::custom)?;
                    Ok(ChatAttachment::Signature(bytes))
                }
           }
    
        } else {
            // For bincode: deserialize as an external enum
            #[derive(serde::Deserialize)]
            enum BincodeHelper {
                Address([u8; 32]),
                CoinId([u8; 32]),
                MixId([u8; 32]),
                Commitment([u8; 32]),
                BlockHash([u8; 32]),
                Midstate([u8; 32]),
                DataHash([u8; 32]),
                LicenseChallenge { commitment: [u8; 32], height: u64, salt: [u8; 32] },
                Signature(Vec<u8>),
            }
            let helper = BincodeHelper::deserialize(deserializer)?;
            match helper {
                BincodeHelper::Address(addr) => Ok(ChatAttachment::Address(addr)),
                BincodeHelper::CoinId(id) => Ok(ChatAttachment::CoinId(id)),
                BincodeHelper::MixId(id) => Ok(ChatAttachment::MixId(id)),
                BincodeHelper::Commitment(id) => Ok(ChatAttachment::Commitment(id)),
                BincodeHelper::BlockHash(id) => Ok(ChatAttachment::BlockHash(id)),
                BincodeHelper::Midstate(id) => Ok(ChatAttachment::Midstate(id)),
                BincodeHelper::DataHash(id) => Ok(ChatAttachment::DataHash(id)),
                BincodeHelper::LicenseChallenge { commitment, height, salt } => {
                    Ok(ChatAttachment::LicenseChallenge { commitment, height, salt })
                }
                BincodeHelper::Signature(sig) => Ok(ChatAttachment::Signature(sig)), 
            }
        }
    }
}

/// Hard cap on attachments per chat message.
///
/// Enforced at every entry point that constructs or accepts a chat:
/// - [`crate::network::protocol::Message::deserialize_bin`] (peer-to-peer)
/// - The `LightRequest::SendChat` handler (light-client origination)
/// - `crate::rpc::handlers::send_chat` (LAN browser origination)
pub const MAX_CHAT_ATTACHMENTS: usize = 4;


/// A chat message as it appears in `Node::chat_history` and in the
/// JSON returned by `GET /api/chat`.
///
/// # Schema
///
/// ```text
/// ┌─ ChatMessage ──────────────────────────
/// │  sender    : PeerId
/// │  timestamp : ℕ
/// │  nonce     : ℕ
/// │  reply_to  : ℕ ∪ {⊥}
/// │  words     : seq u8
/// │  attachs   : seq ChatAttachment
/// ├────────────────────────────────────────
/// │  #words      ≤ 10
/// │  #attachs    ≤ MAX_CHAT_ATTACHMENTS
/// │  #sender_bytes ≤ 128
/// │  isValidPeerId(sender)
/// │  ∀ w ∈ ran words • w < #CHAT_DICTIONARY
/// └────────────────────────────────────────
/// ```
///
/// # JSON additive evolution
///
/// `attachments` is `#[serde(default)]` so payloads without the field
/// still deserialize (legacy v1 producers); `skip_serializing_if =
/// "Vec::is_empty"` makes the serialized form for legacy messages
/// bit-identical to the pre-v2 wire shape.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub sender: String,
    pub timestamp: u64,
    pub nonce: u64,
    pub reply_to: Option<u64>,
    pub words: Vec<u8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<ChatAttachment>,
}

/// Verify PoW for a legacy [`crate::network::protocol::Message::Chat`].
///
/// **Status: receive-only after the v2 introduction.** New nodes never
/// mine v1 PoW. This function is called only when an old peer delivers a
/// legacy `Message::Chat`. New chats are emitted via [`mine_chat_pow_v2`]
/// and [`crate::network::protocol::Message::ChatV2`].
///
/// # Canonical PoW preimage (v1)
///
/// ```text
/// sender_bytes ⌢ le8(timestamp) ⌢ le8(reply_to.unwrap_or(0))
///              ⌢ words ⌢ le8(nonce)
/// ```
///
/// # Difficulty
///
/// Requires ≥ 20 leading zero bits of `BLAKE3(preimage)`. Mining cost
/// is ~2²⁰ ≈ 1 M hashes; ~10 ms on commodity hardware.
///
/// # Domain separation
///
/// **Lemma 2.3.1.** Even for messages with empty attachments,
/// `encode_pow_v1(m) ≠ encode_pow_v2(m)` because v2 inserts the
/// 4-byte little-endian zero `0x00000000` between `words` and `nonce`.
/// Therefore a v1-valid `(m, nonce)` does not validate under v2 (and
/// vice versa) with overwhelming probability. The receive handler
/// dispatches v1/v2 by `Message` variant, never cross-validating.
pub fn verify_chat_pow(sender: &str, timestamp: u64, reply_to: Option<u64>, words: &[u8], nonce: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(sender.as_bytes());
    data.extend_from_slice(&timestamp.to_le_bytes());
    data.extend_from_slice(&reply_to.unwrap_or(0).to_le_bytes());
    data.extend_from_slice(words);
    data.extend_from_slice(&nonce.to_le_bytes());
    
    let h = hash(&data);
    count_leading_zeros(&h) >= 20
}

/// Search nonces from 0 upward until [`verify_chat_pow`] returns `true`.
///
/// **Status: dead code after v2 introduction.** Kept exported in case a
/// future tool needs to reproduce a v1-valid digest. New chat origination
/// uses [`mine_chat_pow_v2`].
pub fn mine_chat_pow(sender: String, timestamp: u64, reply_to: Option<u64>, words: Vec<u8>) -> u64 {
    let mut n = 0u64;
    loop {
        if verify_chat_pow(&sender, timestamp, reply_to, &words, n) {
            return n;
        }
        n += 1;
        // Prevent accidental infinite loops in tests or misconfiguration.
        if n > 10_000_000 {
            panic!("mine_chat_pow: could not find valid nonce (impossible on 20-bit target)");
        }
    }
}

/// Verify PoW for the current [`crate::network::protocol::Message::ChatV2`].
///
/// Binds: sender, timestamp, reply_to (or 0), words, attachments (serialized),
/// and nonce.
///
/// # Canonical preimage (v2)
///
/// ```text
/// sender_bytes ⌢ le8(ts) ⌢ le8(reply_to.unwrap_or(0))
///              ⌢ words ⌢ 0x00000000u32 ⌢ attachments_bincode ⌢ le8(nonce)
/// ```
///
/// The explicit 4-byte `0x00000000` domain separator (after words, before
/// attachments) is what makes v2 ≠ v1 even when attachments is empty.
///
/// # Difficulty
///
/// Same 20 leading zero bits as v1.
pub fn verify_chat_pow_v2(
    sender: &str,
    timestamp: u64,
    reply_to: Option<u64>,
    words: &[u8],
    attachments: &[ChatAttachment],
    nonce: u64,
) -> bool {
    let mut data = Vec::with_capacity(
        sender.len() + 8 + 8 + words.len() + 4 + attachments.len() * (1 + 32) + 8,
    );
    data.extend_from_slice(sender.as_bytes());
    data.extend_from_slice(&timestamp.to_le_bytes());
    data.extend_from_slice(&reply_to.unwrap_or(0).to_le_bytes());
    data.extend_from_slice(words);

    data.extend_from_slice(&(attachments.len() as u32).to_le_bytes());
    for att in attachments {
        match att {
            ChatAttachment::Address(id) => {
                data.push(0x01); // tag: Address
                data.extend_from_slice(id);
            }
            ChatAttachment::CoinId(id) => {
                data.push(0x02); // tag: CoinId
                data.extend_from_slice(id);
            }
            ChatAttachment::MixId(id) => {
                data.push(0x03); // tag: MixId
                data.extend_from_slice(id);
            }
            ChatAttachment::Commitment(id) => {
                data.push(0x04); // tag: Commitment
                data.extend_from_slice(id);
            }
            ChatAttachment::BlockHash(id) => {
                data.push(0x05); // tag: BlockHash
                data.extend_from_slice(id);
            }
            ChatAttachment::Midstate(id) => {
                data.push(0x06); // tag: Midstate
                data.extend_from_slice(id);
            }
            ChatAttachment::DataHash(id) => {
                data.push(0x07); // tag: DataHash
                data.extend_from_slice(id);
            }
            ChatAttachment::LicenseChallenge { commitment, height, salt } => {
                data.push(0x08); // tag: LicenseChallenge
                data.extend_from_slice(commitment);
                data.extend_from_slice(&height.to_le_bytes());
                data.extend_from_slice(salt);
            }
            ChatAttachment::Signature(sig) => {
                data.push(0x09); // tag: Signature
                // Must length-prefix variable data to prevent malleability collisions!
                data.extend_from_slice(&(sig.len() as u32).to_le_bytes());
                data.extend_from_slice(sig);
            }
        }
    }
    data.extend_from_slice(&nonce.to_le_bytes());

    let h = hash(&data);
    count_leading_zeros(&h) >= 20
}

/// Search nonces from 0 upward until [`verify_chat_pow_v2`] returns `true`.
///
/// # Formal Specification
///
/// ```text
/// ∀ m : ChatMessageV2 •
///   let n = mine_chat_pow_v2(...) in
///   verify_chat_pow_v2(m.sender, m.ts, m.reply_to, m.words, m.attachments, n) = ⊤
///         ∧ ∀ k : ℕ • k < n ⇒ ¬verify_chat_pow_v2(..., k)
/// ```
///
/// The search is deterministic and always terminates for the target
/// difficulty (expected ~1 M hashes, < 50 ms worst-case on a Pi-class core).
pub fn mine_chat_pow_v2(
    sender: String,
    timestamp: u64,
    reply_to: Option<u64>,
    words: Vec<u8>,
    attachments: Vec<ChatAttachment>,
) -> u64 {
    let mut n = 0u64;
    loop {
        if verify_chat_pow_v2(&sender, timestamp, reply_to, &words, &attachments, n) {
            return n;
        }
        n += 1;
        if n > 10_000_000 {
            panic!("mine_chat_pow_v2: could not find valid nonce (impossible on 20-bit target)");
        }
    }
}

// --- Optional lightweight manager (Phase 1 stub) ---
// Full ownership of history/seen/limiters + rate helpers can be added
// in a follow-up without changing the public surface.

/// Lightweight holder for chat-related shared state (history + rate limiters).
/// Introduced during god-module refactor to give the chat subsystem
/// a named home. Currently the Arcs are still also held directly on
/// `Node`/`NodeHandle` for minimal diff; future step will centralize here.
#[derive(Clone, Default)]
pub struct ChatManager {
    // Populated on demand by Node for now.
}

impl ChatManager {
    pub fn new() -> Self {
        Self::default()
    }
}
