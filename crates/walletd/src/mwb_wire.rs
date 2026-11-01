//! MDS <-> MDW swap messages on midstate's chat bus.
//!
//! Orders are advertised in the clear: what is for sale, for how much, and
//! the maker's session key for the order. Everything after that is between
//! two wallets, so it is encrypted to the counterparty: the taker's first
//! message carries a fresh session key of its own, both sides derive one
//! secret from the two (Ristretto Diffie–Hellman), and every message is
//! sealed with XChaCha20-Poly1305 under it. Outsiders learn that a session
//! exists, not which order it is for nor anything about the locks it
//! negotiates, and nobody without the session key can speak in it.
//!
//! # Framing
//!
//! A chat message carries at most four attachments of at most 1,536 bytes;
//! `Signature` is the only kind that takes arbitrary bytes, so a frame is
//! the concatenation of up to four of them. A sealed message longer than one
//! frame is split across several, each with its index and the count, and
//! reassembled before it is opened.
//!
//! ```text
//! words      = [WORD_SWAP, WORD_MARK, kind]
//! advert     = VERSION | Advert (fixed 74 bytes)
//! take       = VERSION | order_id[16] | taker_key[32] | index | count | chunk
//! session    = VERSION | tag[16] | to_taker | seq | index | count | chunk
//! sealed     = nonce[24] | XChaCha20-Poly1305(key, nonce, bincode(Msg), aad)
//! ```
//!
//! The chat bus loses messages (each node keeps its last hundred, and dedups
//! by proof-of-work nonce alone, which drops a different chat now and then),
//! so the negotiation above this resends until it hears the next step.

use crate::mwb::{JointView, KernelPlan};
use crate::mwb_link::{LinkCommitment, LinkOpening, LinkProof};
use anyhow::{anyhow, bail, ensure, Result};
use bincode::Options;
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::IsIdentity;
use midstate::chat::{ChatAttachment, ChatMessage};
use midwimble::core::mw::crypto::{compress, decompress, random_scalar, scalar_from_bytes, Point32, SchnorrSig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// "swap"
pub const WORD_SWAP: u8 = 33;
/// "⚡", as qbolt's frames lead with.
pub const WORD_MARK: u8 = 255;
/// "market": an order advert.
pub const KIND_ADVERT: u8 = 34;
/// "take": a taker's first message.
pub const KIND_TAKE: u8 = 43;
/// "trade": every later message of a session.
pub const KIND_SESSION: u8 = 32;

pub const VERSION: u8 = 1;
const ATTACHMENT_MAX: usize = 1_536;
const ATTACHMENTS_MAX: usize = 4;
/// One chat message's worth of frame.
pub const FRAME_MAX: usize = ATTACHMENT_MAX * ATTACHMENTS_MAX;
const TAKE_HEADER: usize = 1 + 16 + 32 + 2;
const SESSION_HEADER: usize = 1 + 16 + 1 + 1 + 2;
/// Fragments a message may span.
pub const MAX_FRAGMENTS: usize = 8;
/// Largest plaintext a message may have.
const MAX_PLAINTEXT: u64 = 40 * 1_024;
/// Reassembly gives up on a message whose fragments do not all arrive.
const REASSEMBLY_SECS: u64 = 600;
const REASSEMBLY_SLOTS: usize = 64;

fn codec() -> impl Options {
    bincode::DefaultOptions::new().with_fixint_encoding().with_limit(MAX_PLAINTEXT)
}

// ── Adverts ─────────────────────────────────────────────────────────────────

/// What the maker sells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    /// MDS for MDW: the maker locks MDS behind an OR lock; the secret comes
    /// out on midwimble.
    SellMds,
    /// MDW for MDS: the maker locks MDW; the taker's MDS sits behind an AND
    /// lock; the secret comes out on midstate.
    SellMdw,
}

impl Side {
    fn byte(self) -> u8 {
        match self {
            Side::SellMds => 0,
            Side::SellMdw => 1,
        }
    }
    fn of(b: u8) -> Option<Self> {
        match b {
            0 => Some(Side::SellMds),
            1 => Some(Side::SellMdw),
            _ => None,
        }
    }
}

/// An order, as advertised. Unauthenticated, like hub adverts: anyone can
/// claim anything here. Nothing is at stake until both locks are verified
/// on chain, and a taker's first message repeats the terms it took, so an
/// altered copy of an advert only reaches a maker who refuses it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Advert {
    pub order_id: [u8; 16],
    pub side: Side,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    /// The maker's session key for this order.
    pub maker_key: Point32,
    /// Midstate height after which the order no longer stands.
    pub expires: u64,
}

impl Advert {
    pub const LEN: usize = 16 + 1 + 8 + 8 + 32 + 8;

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(1 + Self::LEN);
        out.push(VERSION);
        out.extend_from_slice(&self.order_id);
        out.push(self.side.byte());
        out.extend_from_slice(&self.mds_amount.to_le_bytes());
        out.extend_from_slice(&self.mdw_amount.to_le_bytes());
        out.extend_from_slice(&self.maker_key);
        out.extend_from_slice(&self.expires.to_le_bytes());
        out
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() != 1 + Self::LEN || b[0] != VERSION {
            return None;
        }
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        Some(Self {
            order_id: b[1..17].try_into().ok()?,
            side: Side::of(b[17])?,
            mds_amount: u64_at(18),
            mdw_amount: u64_at(26),
            maker_key: b[34..66].try_into().ok()?,
            expires: u64_at(66),
        })
    }

    pub fn chat(&self) -> (Vec<u8>, Vec<ChatAttachment>) {
        (vec![WORD_SWAP, WORD_MARK, KIND_ADVERT], vec![ChatAttachment::Signature(self.encode())])
    }
}

// ── Session keys ────────────────────────────────────────────────────────────

/// A Diffie–Hellman key pair: the maker's per order, the taker's per swap.
pub struct KeyPair {
    secret: Scalar,
    pub public: Point32,
}

impl KeyPair {
    pub fn random() -> Self {
        Self::from_secret(random_scalar())
    }
    fn from_secret(secret: Scalar) -> Self {
        Self { secret, public: compress(&RistrettoPoint::mul_base(&secret)) }
    }
    pub fn to_bytes(&self) -> [u8; 32] {
        self.secret.to_bytes()
    }
    pub fn from_bytes(b: &[u8; 32]) -> Option<Self> {
        scalar_from_bytes(b).map(Self::from_secret)
    }
}

/// Everything one session derives from its two keys.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    pub key: [u8; 32],
    /// Routes session frames to this session; public, and unlinkable to the
    /// order without the key.
    pub tag: [u8; 16],
    /// Names the swap in every proof of knowledge and in the link's
    /// Fiat–Shamir challenge.
    pub swap_id: [u8; 32],
}

impl Session {
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:  ours = our key pair, theirs = the other side's public key
    /// Post: Ok(s) ⇒ both sides derive the same s, since
    ///                ours.secret · theirs = their.secret · ours.public
    ///       Err(_) ⇔ theirs is not a valid non-identity point
    /// ```
    pub fn derive(ours: &KeyPair, theirs: &Point32, order_id: &[u8; 16], maker_key: &Point32, taker_key: &Point32) -> Result<Self> {
        let p = decompress(theirs).filter(|p| !p.is_identity()).ok_or_else(|| anyhow!("session key is not a valid point"))?;
        let shared = compress(&(ours.secret * p));
        let mut material = Vec::with_capacity(32 + 16 + 64);
        material.extend_from_slice(&shared);
        material.extend_from_slice(order_id);
        material.extend_from_slice(maker_key);
        material.extend_from_slice(taker_key);
        let key = blake3::derive_key("midstate.mwb-swap.session-key.v1", &material);
        let tag_full = blake3::derive_key("midstate.mwb-swap.session-tag.v1", &material);
        // Public: both keys are in the clear on the bus.
        let swap_id = blake3::derive_key("midstate.mwb-swap.swap-id.v1", &material[32..]);
        Ok(Self { key, tag: tag_full[..16].try_into().unwrap(), swap_id })
    }

    fn cipher(&self) -> XChaCha20Poly1305 {
        XChaCha20Poly1305::new(Key::from_slice(&self.key))
    }
}

/// The swap id both sides derive, from public values alone.
pub fn swap_id(order_id: &[u8; 16], maker_key: &Point32, taker_key: &Point32) -> [u8; 32] {
    let mut m = Vec::with_capacity(80);
    m.extend_from_slice(order_id);
    m.extend_from_slice(maker_key);
    m.extend_from_slice(taker_key);
    blake3::derive_key("midstate.mwb-swap.swap-id.v1", &m)
}

// ── Messages ────────────────────────────────────────────────────────────────

/// The taker's opening move: the terms it took, and its first contribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Take {
    pub side: Side,
    pub mds_amount: u64,
    pub mdw_amount: u64,
    /// The taker's owner-key share and its proof of knowledge.
    pub share: Point32,
    pub pok: SchnorrSig,
    /// Selling MDS: the key the OR lock pays the taker with. Selling MDW:
    /// the key the AND lock refunds the taker with.
    pub mds_pk: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OfferLink {
    /// Selling MDS: the whole non-interactive proof.
    Proof(LinkProof),
    /// Selling MDW: the commitment; the taker's challenge comes next.
    Commitment(LinkCommitment),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Offer {
    pub share: Point32,
    pub pok: SchnorrSig,
    /// Selling MDS: the OR lock's refund key. Selling MDW: the AND lock's
    /// claim key.
    pub mds_pk: [u8; 32],
    /// Refund height of the reveal leg, on its chain (midwimble when selling
    /// MDS, midstate when selling MDW).
    pub reveal_height: u64,
    /// Refund height of the follow leg, on the other chain.
    pub follow_height: u64,
    pub link: OfferLink,
    /// Selling MDS: the salts of the coins the maker's lock will hold, one
    /// per denomination of the amount, so the taker can find them.
    pub mds_salts: Vec<[u8; 32]>,
    /// Selling MDW: the maker funds the midwimble lock.
    pub funder: Option<FunderCommit>,
}

/// The funder's lock, and its commitments to its four nonces: the input
/// signature's, and the refund kernel's Pedersen, refund kernel's owner
/// (the funder builds the refund) and claim kernel's owner (it co-signs the
/// claim).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunderCommit {
    pub view: JointView,
    pub commits: [[u8; 32]; 4],
}

/// The receiver's four nonces, in the clear (the funder committed first):
/// the input signature's, the refund kernel's owner (it co-signs the
/// refund), and the claim kernel's Pedersen and owner (it builds the claim).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiverNonces {
    pub nonces: [Point32; 4],
    /// Selling MDW: the taker's challenge to the link commitment.
    pub challenge: Option<Vec<bool>>,
    /// Selling MDW: the salts of the coins the taker's lock will hold.
    pub mds_salts: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunderNonces {
    pub nonces: [Point32; 4],
    pub input_partial: [u8; 32],
    pub refund_plan: KernelPlan,
    /// Selling MDW: the maker opens the challenged halves.
    pub opening: Option<LinkOpening>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReceiverSigs {
    pub input_partial: [u8; 32],
    pub refund_partial: [u8; 32],
    pub claim_plan: KernelPlan,
    /// Selling MDS: the receiver's own claim partial, which the funder (the
    /// taker) needs to read the secret off the published kernel.
    pub claim_owner_partial: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Msg {
    Take(Take),
    Offer(Offer),
    FunderCommit(FunderCommit),
    ReceiverNonces(ReceiverNonces),
    FunderNonces(FunderNonces),
    ReceiverSigs(ReceiverSigs),
    FunderClaimSig { claim_partial: [u8; 32] },
    Abort { reason: String },
}

impl Msg {
    pub fn name(&self) -> &'static str {
        match self {
            Msg::Take(_) => "take",
            Msg::Offer(_) => "offer",
            Msg::FunderCommit(_) => "funder commitments",
            Msg::ReceiverNonces(_) => "receiver nonces",
            Msg::FunderNonces(_) => "funder nonces",
            Msg::ReceiverSigs(_) => "receiver signatures",
            Msg::FunderClaimSig { .. } => "funder claim signature",
            Msg::Abort { .. } => "abort",
        }
    }
}

/// Which way a session message travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum To {
    Maker,
    Taker,
}

impl To {
    fn byte(self) -> u8 {
        match self {
            To::Maker => 0,
            To::Taker => 1,
        }
    }
    fn of(b: u8) -> Option<Self> {
        match b {
            0 => Some(To::Maker),
            1 => Some(To::Taker),
            _ => None,
        }
    }
}

fn aad(kind: u8, route: &[u8], to: To, seq: u8) -> Vec<u8> {
    let mut a = Vec::with_capacity(4 + route.len());
    a.push(VERSION);
    a.push(kind);
    a.extend_from_slice(route);
    a.push(to.byte());
    a.push(seq);
    a
}

fn seal(session: &Session, aad: &[u8], msg: &Msg) -> Result<Vec<u8>> {
    let plain = codec().serialize(msg)?;
    let nonce: [u8; 24] = rand::random();
    let ct = session
        .cipher()
        .encrypt(XNonce::from_slice(&nonce), Payload { msg: &plain, aad })
        .map_err(|_| anyhow!("encryption failed"))?;
    let mut out = Vec::with_capacity(24 + ct.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

fn open(session: &Session, aad: &[u8], sealed: &[u8]) -> Result<Msg> {
    ensure!(sealed.len() > 24, "sealed message too short");
    let plain = session
        .cipher()
        .decrypt(XNonce::from_slice(&sealed[..24]), Payload { msg: &sealed[24..], aad })
        .map_err(|_| anyhow!("message does not open under this session"))?;
    Ok(codec().deserialize(&plain)?)
}

/// Splits `bytes` into frames of at most [`FRAME_MAX`] after `header`.
fn fragment(header: &[u8], bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    let room = FRAME_MAX - header.len() - 2;
    let count = bytes.len().div_ceil(room).max(1);
    ensure!(count <= MAX_FRAGMENTS, "message too long for the chat bus ({} bytes)", bytes.len());
    Ok(bytes
        .chunks(room)
        .enumerate()
        .map(|(i, chunk)| {
            let mut f = Vec::with_capacity(header.len() + 2 + chunk.len());
            f.extend_from_slice(header);
            f.push(i as u8);
            f.push(count as u8);
            f.extend_from_slice(chunk);
            f
        })
        .collect())
}

fn attachments(frame: Vec<u8>) -> Vec<ChatAttachment> {
    frame.chunks(ATTACHMENT_MAX).map(|c| ChatAttachment::Signature(c.to_vec())).collect()
}

/// One chat message to send: `(words, attachments)`.
pub type Outgoing = (Vec<u8>, Vec<ChatAttachment>);

/// Seals the taker's first message to the maker of `order_id`.
pub fn take_messages(session: &Session, order_id: &[u8; 16], taker_key: &Point32, msg: &Msg) -> Result<Vec<Outgoing>> {
    let mut route = order_id.to_vec();
    route.extend_from_slice(taker_key);
    let sealed = seal(session, &aad(KIND_TAKE, &route, To::Maker, 0), msg)?;
    let mut header = vec![VERSION];
    header.extend_from_slice(&route);
    Ok(fragment(&header, &sealed)?
        .into_iter()
        .map(|f| (vec![WORD_SWAP, WORD_MARK, KIND_TAKE], attachments(f)))
        .collect())
}

/// Seals message `seq` of a session.
pub fn session_messages(session: &Session, to: To, seq: u8, msg: &Msg) -> Result<Vec<Outgoing>> {
    let sealed = seal(session, &aad(KIND_SESSION, &session.tag, to, seq), msg)?;
    let mut header = vec![VERSION];
    header.extend_from_slice(&session.tag);
    header.push(to.byte());
    header.push(seq);
    Ok(fragment(&header, &sealed)?
        .into_iter()
        .map(|f| (vec![WORD_SWAP, WORD_MARK, KIND_SESSION], attachments(f)))
        .collect())
}

// ── Receiving ───────────────────────────────────────────────────────────────

/// A chat message of this protocol, before its session is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    Advert(Advert),
    Take { order_id: [u8; 16], taker_key: Point32, index: u8, count: u8, chunk: Vec<u8> },
    Session { tag: [u8; 16], to: To, seq: u8, index: u8, count: u8, chunk: Vec<u8> },
}

/// Parses a chat message, or `None` if it is not one of ours.
pub fn parse(m: &ChatMessage) -> Option<Frame> {
    if m.words.len() != 3 || m.words[0] != WORD_SWAP || m.words[1] != WORD_MARK {
        return None;
    }
    let mut bytes = Vec::new();
    for a in &m.attachments {
        match a {
            ChatAttachment::Signature(b) => bytes.extend_from_slice(b),
            _ => return None,
        }
    }
    if bytes.first() != Some(&VERSION) {
        return None;
    }
    let fragment = |o: usize| -> Option<(u8, u8, Vec<u8>)> {
        let (index, count) = (*bytes.get(o)?, *bytes.get(o + 1)?);
        (index < count && count as usize <= MAX_FRAGMENTS).then(|| (index, count, bytes[o + 2..].to_vec()))
    };
    match m.words[2] {
        KIND_ADVERT => Advert::decode(&bytes).map(Frame::Advert),
        KIND_TAKE if bytes.len() > TAKE_HEADER => {
            let (index, count, chunk) = fragment(1 + 16 + 32)?;
            Some(Frame::Take {
                order_id: bytes[1..17].try_into().ok()?,
                taker_key: bytes[17..49].try_into().ok()?,
                index,
                count,
                chunk,
            })
        }
        KIND_SESSION if bytes.len() > SESSION_HEADER => {
            let (index, count, chunk) = fragment(1 + 16 + 2)?;
            Some(Frame::Session { tag: bytes[1..17].try_into().ok()?, to: To::of(bytes[17])?, seq: bytes[18], index, count, chunk })
        }
        _ => None,
    }
}

/// Where a reassembled message came from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    Take { order_id: [u8; 16], taker_key: Point32 },
    Session { tag: [u8; 16], to: To, seq: u8 },
}

impl Route {
    fn aad(&self) -> Vec<u8> {
        match self {
            Route::Take { order_id, taker_key } => {
                let mut r = order_id.to_vec();
                r.extend_from_slice(taker_key);
                aad(KIND_TAKE, &r, To::Maker, 0)
            }
            Route::Session { tag, to, seq } => aad(KIND_SESSION, tag, *to, *seq),
        }
    }

    /// Opens a reassembled message under its session.
    pub fn open(&self, session: &Session, sealed: &[u8]) -> Result<Msg> {
        open(session, &self.aad(), sealed)
    }
}

struct Partial {
    parts: Vec<Option<Vec<u8>>>,
    first: u64,
}

/// Collects fragments until a message is whole.
#[derive(Default)]
pub struct Reassembler {
    partial: HashMap<(Route, u8), Partial>,
}

impl Reassembler {
    /// Adds a fragment; returns the sealed message once every fragment of
    /// it has arrived. A message is identified by its route and fragment
    /// count; a resend of it (sealed afresh) may complete it with its own
    /// fragments, so mixing two sealings is caught when it fails to open.
    pub fn add(&mut self, route: Route, index: u8, count: u8, chunk: Vec<u8>, now: u64) -> Option<Vec<u8>> {
        self.partial.retain(|_, p| now.saturating_sub(p.first) < REASSEMBLY_SECS);
        if count == 1 {
            return Some(chunk);
        }
        if self.partial.len() >= REASSEMBLY_SLOTS && !self.partial.contains_key(&(route.clone(), count)) {
            return None;
        }
        let key = (route, count);
        let p = self
            .partial
            .entry(key.clone())
            .or_insert_with(|| Partial { parts: vec![None; count as usize], first: now });
        p.parts[index as usize] = Some(chunk);
        if p.parts.iter().all(Option::is_some) {
            let whole = self.partial.remove(&key)?.parts.into_iter().flatten().flatten().collect();
            return Some(whole);
        }
        None
    }
}

/// Bounds a reason string a counterparty sent before it is shown.
pub fn reason(s: &str) -> String {
    let clean: String = s.chars().filter(|c| !c.is_control()).take(160).collect();
    if clean.is_empty() {
        "no reason given".into()
    } else {
        clean
    }
}

/// Picks the session a take belongs to: the maker derives each open order's
/// session from the taker's key until one opens it.
pub fn maker_session(order_key: &KeyPair, order_id: &[u8; 16], taker_key: &Point32) -> Result<Session> {
    Session::derive(order_key, taker_key, order_id, &order_key.public, taker_key)
}

pub fn taker_session(ours: &KeyPair, advert: &Advert) -> Result<Session> {
    Session::derive(ours, &advert.maker_key, &advert.order_id, &advert.maker_key, &ours.public)
}

/// Checks that a counterparty's 32 bytes are a canonical scalar.
pub fn partial(b: &[u8; 32]) -> Result<Scalar> {
    scalar_from_bytes(b).ok_or_else(|| anyhow!("partial signature is not a canonical scalar"))
}

/// Refuses a point that does not decompress (or is the identity).
pub fn point(p: &Point32, what: &str) -> Result<RistrettoPoint> {
    match decompress(p) {
        Some(x) if !x.is_identity() => Ok(x),
        _ => bail!("{what} is not a valid point"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mwb_link::LinkSecret;

    fn chat(words: Vec<u8>, attachments: Vec<ChatAttachment>) -> ChatMessage {
        ChatMessage { sender: "peer".into(), timestamp: 1, nonce: 2, reply_to: None, words, attachments }
    }

    fn deliver(r: &mut Reassembler, out: Vec<Outgoing>) -> Option<(Route, Vec<u8>)> {
        let mut done = None;
        for (w, a) in out {
            assert!(a.len() <= ATTACHMENTS_MAX && a.iter().all(|x| matches!(x, ChatAttachment::Signature(b) if b.len() <= ATTACHMENT_MAX)));
            match parse(&chat(w, a)).expect("one of ours") {
                Frame::Take { order_id, taker_key, index, count, chunk } => {
                    let route = Route::Take { order_id, taker_key };
                    if let Some(s) = r.add(route.clone(), index, count, chunk, 0) {
                        done = Some((route, s));
                    }
                }
                Frame::Session { tag, to, seq, index, count, chunk } => {
                    let route = Route::Session { tag, to, seq };
                    if let Some(s) = r.add(route.clone(), index, count, chunk, 0) {
                        done = Some((route, s));
                    }
                }
                Frame::Advert(_) => unreachable!(),
            }
        }
        done
    }

    #[test]
    fn adverts_round_trip_and_strangers_are_ignored() {
        let a = Advert {
            order_id: [3; 16],
            side: Side::SellMdw,
            mds_amount: 1 << 20,
            mdw_amount: 12_345_678,
            maker_key: KeyPair::random().public,
            expires: 400_000,
        };
        let (w, att) = a.chat();
        assert_eq!(parse(&chat(w, att)), Some(Frame::Advert(a)));
        assert_eq!(parse(&chat(vec![33, 255, 34], vec![ChatAttachment::Address([0; 32])])), None);
        assert_eq!(parse(&chat(vec![255, 50], vec![])), None, "qbolt's frames are not ours");
    }

    #[test]
    fn a_session_carries_a_large_message_across_frames_and_only_its_keys_open_it() {
        let (maker, taker) = (KeyPair::random(), KeyPair::random());
        let advert = Advert { order_id: [1; 16], side: Side::SellMds, mds_amount: 1, mdw_amount: 1, maker_key: maker.public, expires: 9 };
        let ts = taker_session(&taker, &advert).unwrap();
        let ms = maker_session(&maker, &advert.order_id, &taker.public).unwrap();
        assert_eq!(ts, ms, "both sides derive one session");
        assert_eq!(ts.swap_id, swap_id(&advert.order_id, &maker.public, &taker.public));

        // The selling-MDS offer: the largest message of the protocol.
        let proof = LinkSecret::generate(crate::mwb_link::K_OR).prove(&ts.swap_id).unwrap();
        let share = crate::mwb::OwnerShare::random();
        let offer = Msg::Offer(Offer {
            share: share.public,
            pok: share.prove(&ts.swap_id),
            mds_pk: [4; 32],
            reveal_height: 100,
            follow_height: 200,
            link: OfferLink::Proof(proof.clone()),
            mds_salts: vec![[5; 32]; 7],
            funder: None,
        });
        let out = session_messages(&ts, To::Taker, 1, &offer).unwrap();
        assert_eq!(out.len(), 2, "8 KB of proof takes two chat messages");
        let mut r = Reassembler::default();
        let (route, sealed) = deliver(&mut r, out.clone()).expect("whole");
        let Msg::Offer(back) = route.open(&ts, &sealed).unwrap() else { panic!("not an offer") };
        let OfferLink::Proof(p) = back.link else { panic!("not a proof") };
        assert_eq!(p, proof);

        // A stranger's session does not open it.
        let other = taker_session(&KeyPair::random(), &advert).unwrap();
        assert!(route.open(&other, &sealed).is_err());
        // Nor does it open as another step or direction.
        let wrong = Route::Session { tag: ts.tag, to: To::Maker, seq: 1 };
        assert!(wrong.open(&ts, &sealed).is_err());
        // A lost fragment holds the message back; its resend completes it.
        let mut r = Reassembler::default();
        assert!(deliver(&mut r, out[..1].to_vec()).is_none());
        assert!(deliver(&mut r, out[1..].to_vec()).is_some());
    }

    #[test]
    fn a_take_reaches_only_the_maker_of_its_order() {
        let (maker, taker) = (KeyPair::random(), KeyPair::random());
        let advert = Advert { order_id: [8; 16], side: Side::SellMdw, mds_amount: 1, mdw_amount: 1, maker_key: maker.public, expires: 9 };
        let s = taker_session(&taker, &advert).unwrap();
        let share = crate::mwb::OwnerShare::random();
        let take = Msg::Take(Take {
            side: advert.side,
            mds_amount: 1,
            mdw_amount: 1,
            share: share.public,
            pok: share.prove(&s.swap_id),
            mds_pk: [7; 32],
        });
        let out = take_messages(&s, &advert.order_id, &taker.public, &take).unwrap();
        let (route, sealed) = deliver(&mut Reassembler::default(), out).unwrap();
        let Route::Take { order_id, taker_key } = route.clone() else { panic!() };
        let opened = maker_session(&maker, &order_id, &taker_key).unwrap();
        assert!(matches!(route.open(&opened, &sealed).unwrap(), Msg::Take(_)));
        let stranger = maker_session(&KeyPair::random(), &order_id, &taker_key).unwrap();
        assert!(route.open(&stranger, &sealed).is_err());
    }

    #[test]
    fn identity_session_keys_are_refused() {
        let me = KeyPair::random();
        assert!(Session::derive(&me, &[0u8; 32], &[0; 16], &me.public, &[0; 32]).is_err());
    }
}
