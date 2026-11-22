//! # Chat subsystem
//!
//! Midstate's chat is a dictionary-bound P2P message bus. Every chat token
//! is a `u8` index into [`CHAT_DICTIONARY`] (256 fixed entries), and every
//! other field that gossips across the network is a byte payload the
//! receiver bounds. The design property the subsystem is built around:
//!
//! > **No readable text in the clear.** Words are dictionary indices, never
//! > free text. The only `String` is `sender`, validated as a base58 libp2p
//! > `PeerId` (or a 64-hex MSS key) in
//! > [`crate::network::protocol::Message::deserialize_bin`]. Fixed-width
//! > attachments that parse as UTF-8 are refused as graffiti
//! > ([`ChatAttachment::is_graffiti`]).
//!
//! What that property does **not** say, plainly:
//!
//! * Two attachments carry arbitrary bytes of variable length and are not
//!   graffiti-checked: [`ChatAttachment::Signature`] (up to 1,536 bytes) and
//!   [`ChatAttachment::EncryptedMessage`] (up to 1,536 bytes). `Signature`
//!   already carries more than signatures: qbolt frames and the swap markets'
//!   adverts ride in it, in the clear. Nothing stops a sender putting text
//!   there, and no node can tell ciphertext from text: random-looking bytes
//!   are all a node ever sees of a sealed message.
//! * So the property is about what Midstate software *shows*, not about what
//!   a determined sender can smuggle: no client renders an attachment's bytes
//!   as text, and a sealed message is readable only by the wallet holding the
//!   key it was sealed to. Smuggling costs proof of work by the byte
//!   ([`chat_pow_units`]) for `EncryptedMessage`, and the 20-bit chat proof of
//!   work for `Signature` (unchanged, so existing peers keep agreeing).
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
//! ## Sealed messages and older nodes
//!
//! [`ChatAttachment::EncryptedMessage`] is appended after every existing
//! variant, so every other attachment (and every other `Message`) encodes
//! byte for byte as before. A node that predates it cannot decode a `ChatV2`
//! carrying one: bincode fails on the unknown variant index inside the
//! attachments, `deserialize_bin` errs, and libp2p's request-response reports
//! an inbound failure, which `network::Network` only logs. The old node
//! drops that message and relays nothing; it neither scores nor disconnects
//! the sender. The *sender* sees its request fail, and a failed request to
//! its sync peer aborts its sync session. So a node sends a chat carrying a
//! sealed message only to peers that advertise
//! [`crate::network::protocol::SEALED_CHAT_PROTOCOL`] (`Network::send`), and
//! older peers never see one: the chat network does not split, sealed
//! messages simply travel among upgraded nodes.
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
/// # Invariant: no readable text in the clear
///
/// No variant is a `String`. The fixed-width variants are refused when they
/// parse as UTF-8 ([`ChatAttachment::is_graffiti`]). Two variants carry
/// arbitrary bytes of variable length, bounded at 1,536: `Signature`
/// (protocol payloads in the clear: signatures, qbolt frames, swap adverts)
/// and `EncryptedMessage` (ciphertext). See the module docs for what that
/// does and does not rule out.
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
/// | Variant            | Tag    | Payload                         |
/// |--------------------|--------|---------------------------------|
/// | `Address`          | `0x01` | 32 bytes                        |
/// | `CoinId`           | `0x02` | 32 bytes                        |
/// | `MixId`            | `0x03` | 32 bytes                        |
/// | `Commitment`       | `0x04` | 32 bytes                        |
/// | `BlockHash`        | `0x05` | 32 bytes                        |
/// | `Midstate`         | `0x06` | 32 bytes                        |
/// | `DataHash`         | `0x07` | 32 bytes                        |
/// | `LicenseChallenge` | `0x08` | commitment ‖ le8(height) ‖ salt |
/// | `Signature`        | `0x09` | le4(len) ‖ bytes                |
/// | `EncryptedMessage` | `0x0A` | le4(len) ‖ bytes                |
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
    /// A sealed message: ciphertext only its recipient's wallet can open
    /// (`crate::sealed`). Who it is for is not in the clear; recipients
    /// trial-decrypt every one, as wallets scan for stealth payments.
    ///
    /// ```text
    /// Pre:  payload.len() <= MAX_ENCRYPTED_MESSAGE_SIZE (1536 bytes)
    /// PoW:  scales with payload size (chat_pow_units)
    /// Wire: appended: older nodes cannot decode it, so it is sent only to
    ///       peers that advertise SEALED_CHAT_PROTOCOL
    /// ```
    EncryptedMessage(Vec<u8>),
}

/// Largest [`ChatAttachment::EncryptedMessage`] payload.
pub const MAX_ENCRYPTED_MESSAGE_SIZE: usize = 1_536;

/// Leading zero bits a chat's proof of work needs, per unit of work.
pub const CHAT_POW_BITS: u32 = 20;

/// Bytes of sealed payload one unit of chat proof of work covers.
pub const ENCRYPTED_POW_UNIT: usize = 512;

/// Units of proof of work a v2 chat with `attachments` needs: one, plus one
/// per [`ENCRYPTED_POW_UNIT`] bytes of `EncryptedMessage` payload beyond the
/// first. A chat without one needs exactly one unit, the 20 bits it always
/// has, so every existing chat verifies as before.
///
/// ```text
/// units(m) = max(1, ⌈Σ |e| / 512⌉)   over e ∈ EncryptedMessage attachments of m
/// ```
pub fn chat_pow_units(attachments: &[ChatAttachment]) -> u64 {
    let sealed: usize = attachments
        .iter()
        .map(|a| if let ChatAttachment::EncryptedMessage(b) = a { b.len() } else { 0 })
        .sum();
    (sealed.div_ceil(ENCRYPTED_POW_UNIT) as u64).max(1)
}

/// Whether `h` meets `units` of chat proof of work: its first 64 bits, big
/// endian, under `2^(64 − 20) / units`. One unit is exactly "20 leading zero
/// bits"; `units` of them take `units` times the hashes on average.
pub fn meets_chat_pow(h: &[u8; 32], units: u64) -> bool {
    let top = u64::from_be_bytes(h[..8].try_into().expect("8 bytes"));
    top < (1u64 << (64 - CHAT_POW_BITS)) / units.max(1)
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
            ChatAttachment::EncryptedMessage(_) => None,
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
                EncryptedMessage(String),
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
                ChatAttachment::EncryptedMessage(b) => JsonHelper::EncryptedMessage(hex::encode(b)),
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
                // Appended: every variant above keeps its index and bytes.
                EncryptedMessage(&'a [u8]),
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
                ChatAttachment::EncryptedMessage(b) => BincodeHelper::EncryptedMessage(b.as_slice()),
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
                EncryptedMessage(String),
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
                JsonHelper::EncryptedMessage(s) => {
                    let bytes = hex::decode(s).map_err(serde::de::Error::custom)?;
                    if bytes.len() > MAX_ENCRYPTED_MESSAGE_SIZE {
                        return Err(serde::de::Error::custom("encrypted message exceeds 1536 bytes"));
                    }
                    Ok(ChatAttachment::EncryptedMessage(bytes))
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
                EncryptedMessage(Vec<u8>),
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
                BincodeHelper::EncryptedMessage(b) => Ok(ChatAttachment::EncryptedMessage(b)),
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
/// [`chat_pow_units`] units of 20 leading zero bits ([`meets_chat_pow`]):
/// exactly v1's 20 bits for any chat without an `EncryptedMessage`, more as
/// sealed bytes grow.
///
/// # Variable-length attachments
///
/// `Signature` and `EncryptedMessage` enter the preimage as
/// `tag ‖ le4(len) ‖ bytes`, length-prefixed so no two attachment lists
/// serialize alike. The buffer's capacity is only a size hint.
pub fn verify_chat_pow_v2(
    sender: &str,
    timestamp: u64,
    reply_to: Option<u64>,
    words: &[u8],
    attachments: &[ChatAttachment],
    nonce: u64,
) -> bool {
    let attachment_bytes: usize = attachments
        .iter()
        .map(|a| match a {
            ChatAttachment::LicenseChallenge { .. } => 1 + 32 + 8 + 32,
            ChatAttachment::Signature(b) | ChatAttachment::EncryptedMessage(b) => 1 + 4 + b.len(),
            _ => 1 + 32,
        })
        .sum();
    let mut data = Vec::with_capacity(sender.len() + 8 + 8 + words.len() + 4 + attachment_bytes + 8);
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
            ChatAttachment::EncryptedMessage(b) => {
                data.push(0x0A); // tag: EncryptedMessage
                data.extend_from_slice(&(b.len() as u32).to_le_bytes());
                data.extend_from_slice(b);
            }
        }
    }
    data.extend_from_slice(&nonce.to_le_bytes());

    let h = hash(&data);
    meets_chat_pow(&h, chat_pow_units(attachments))
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
    // Expected 2^20 · units tries; giving up past 16 times that has odds of
    // about e^-16 for a correct miner.
    let limit = (1u64 << CHAT_POW_BITS) * 16 * chat_pow_units(&attachments);
    let mut n = 0u64;
    loop {
        if verify_chat_pow_v2(&sender, timestamp, reply_to, &words, &attachments, n) {
            return n;
        }
        n += 1;
        if n > limit {
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

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::Options;

    fn codec() -> impl Options {
        bincode::DefaultOptions::new()
    }

    /// The attachment enum as a node before `EncryptedMessage` has it, for
    /// checking what such a node makes of what this one sends.
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    enum OldWire {
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

    fn every_old_kind() -> (Vec<ChatAttachment>, Vec<OldWire>) {
        (
            vec![
                ChatAttachment::Address([1; 32]),
                ChatAttachment::CoinId([2; 32]),
                ChatAttachment::MixId([3; 32]),
                ChatAttachment::Commitment([4; 32]),
                ChatAttachment::BlockHash([5; 32]),
                ChatAttachment::Midstate([6; 32]),
                ChatAttachment::DataHash([7; 32]),
                ChatAttachment::LicenseChallenge { commitment: [8; 32], height: 9, salt: [10; 32] },
                ChatAttachment::Signature(vec![11; 600]),
            ],
            vec![
                OldWire::Address([1; 32]),
                OldWire::CoinId([2; 32]),
                OldWire::MixId([3; 32]),
                OldWire::Commitment([4; 32]),
                OldWire::BlockHash([5; 32]),
                OldWire::Midstate([6; 32]),
                OldWire::DataHash([7; 32]),
                // The wire has always carried a zero commitment here.
                OldWire::LicenseChallenge { commitment: [0; 32], height: 9, salt: [10; 32] },
                OldWire::Signature(vec![11; 600]),
            ],
        )
    }

    #[test]
    fn existing_attachments_encode_byte_for_byte_as_before() {
        let (new, old) = every_old_kind();
        assert_eq!(codec().serialize(&new).unwrap(), codec().serialize(&old).unwrap());
        let msg = crate::network::protocol::Message::ChatV2 {
            sender: "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".into(),
            timestamp: 1,
            nonce: 2,
            reply_to: Some(3),
            words: vec![4, 5],
            // Fixed-width test patterns would read as graffiti; these two are exempt.
            attachments: new[7..].to_vec(),
        };
        let bytes = msg.serialize_bin();
        assert!(crate::network::protocol::Message::deserialize_bin(&bytes).is_ok());
    }

    #[test]
    fn an_older_node_cannot_decode_a_sealed_message_and_only_drops_it() {
        let sealed = vec![ChatAttachment::Address([1; 32]), ChatAttachment::EncryptedMessage(vec![0xAB; 900])];
        let bytes = codec().serialize(&sealed).unwrap();
        // The unknown variant index fails the whole decode: the old node's
        // `deserialize_bin` errs, the stream fails, and the message is gone.
        assert!(codec().deserialize::<Vec<OldWire>>(&bytes).is_err());
        // And this node reads it back whole.
        assert_eq!(codec().deserialize::<Vec<ChatAttachment>>(&bytes).unwrap(), sealed);
        let json = serde_json::to_value(&sealed[1]).unwrap();
        assert_eq!(json["kind"], "encrypted_message");
        assert_eq!(serde_json::from_value::<ChatAttachment>(json).unwrap(), sealed[1]);
    }

    #[test]
    fn ciphertext_is_never_graffiti_but_fixed_text_still_is() {
        // Ciphertext may well be valid UTF-8 (all ASCII here); it is exempt.
        assert!(!ChatAttachment::EncryptedMessage(b"looks like text".to_vec()).is_graffiti());
        assert!(ChatAttachment::DataHash(*b"this is 32 bytes of ascii text!!").is_graffiti());
    }

    #[test]
    fn one_unit_is_exactly_the_old_twenty_bits() {
        for i in 0..20_000u32 {
            let h = hash(&i.to_le_bytes());
            assert_eq!(meets_chat_pow(&h, 1), count_leading_zeros(&h) >= 20, "hash {i}");
        }
        let mut edge = [0u8; 32];
        edge[2] = 0x0F; // 20 zero bits, then ones
        edge[3] = 0xFF;
        assert!(meets_chat_pow(&edge, 1));
        edge[2] = 0x10; // 19 zero bits
        assert!(!meets_chat_pow(&edge, 1));
    }

    #[test]
    fn proof_of_work_scales_with_sealed_bytes_only() {
        let sig = ChatAttachment::Signature(vec![0; 1536]);
        assert_eq!(chat_pow_units(&[]), 1);
        assert_eq!(chat_pow_units(&[sig.clone(), ChatAttachment::Address([1; 32])]), 1, "signatures stay at one unit");
        assert_eq!(chat_pow_units(&[ChatAttachment::EncryptedMessage(vec![0; 300])]), 1);
        assert_eq!(chat_pow_units(&[ChatAttachment::EncryptedMessage(vec![0; 513])]), 2);
        let full = ChatAttachment::EncryptedMessage(vec![0; MAX_ENCRYPTED_MESSAGE_SIZE]);
        assert_eq!(chat_pow_units(&[full.clone()]), 3);
        assert_eq!(chat_pow_units(&[full.clone(), full.clone(), full.clone(), full.clone()]), 12);
        // A harder target is a smaller one: anything meeting 3 units meets 1.
        let h = {
            let mut h = [0u8; 32];
            h[2] = 0x05;
            h
        };
        assert!(meets_chat_pow(&h, 1) && meets_chat_pow(&h, 3) && !meets_chat_pow(&h, 4));
    }

    #[test]
    fn a_sealed_chat_mines_and_verifies_and_is_bound_to_its_bytes() {
        let atts = vec![ChatAttachment::EncryptedMessage(vec![7; 700])];
        let n = mine_chat_pow_v2("peer".into(), 5, None, vec![1], atts.clone());
        assert!(verify_chat_pow_v2("peer", 5, None, &[1], &atts, n));
        let mut other = atts.clone();
        if let ChatAttachment::EncryptedMessage(b) = &mut other[0] {
            b[0] ^= 1;
        }
        // With overwhelming probability the altered bytes miss the target.
        assert!(!verify_chat_pow_v2("peer", 5, None, &[1], &other, n));
        // The same bytes as a Signature are another preimage (tag 0x09).
        let as_sig = vec![ChatAttachment::Signature(vec![7; 700])];
        assert!(!verify_chat_pow_v2("peer", 5, None, &[1], &as_sig, n));
    }
}
