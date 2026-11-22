//! Sealed chat messages between names: ML-KEM-512 (FIPS 203) and
//! XChaCha20-Poly1305, carried in [`ChatAttachment::EncryptedMessage`].
//!
//! # Keys
//!
//! A wallet's KEM key pair comes from its seed ([`KemKeys::from_secret`]), so
//! a restore brings it back. A name publishes `H(public key)` in a `kem`
//! record (`core::names::tag::KEM`); the 800-byte key itself is too big for a
//! record and comes from the holder over the identity handshake, checked
//! against the hash ([`check_public`]).
//!
//! # Opening a session
//!
//! The opener encapsulates to the recipient's key and seals its first message
//! under a key derived from the shared secret:
//!
//! ```text
//! opening  = ct (768) ‖ seal(k_open, 0, hello)
//! hello    = v1 ‖ 0 ‖ name_len ‖ name ‖ identity (32) ‖ sig_len (2) ‖ sig ‖ body_len (2) ‖ body ‖ pad
//! k_open   = derive("…open.v1", ss ‖ ct)
//! sig      = the opener's identity key (MSS) over transcript(recipient's key hash, ct, name, identity)
//! ```
//!
//! Neither the recipient nor the sender is in the clear: the recipient
//! decapsulates every sealed message it sees and keeps those whose seal
//! opens, as wallets scan for stealth payments. The sender's name and its
//! signature are inside, so the recipient checks them against the registry
//! (the name's `identity` record must be the key that signed). One MSS
//! signature per session, never per message: one-time signatures are spent
//! by the session, not by the conversation. An opener without a name may
//! open anonymously (no name, no signature); wallets show it as such.
//!
//! # Within a session
//!
//! Both sides derive a key per direction from the shared secret. A message
//! is `route (16) ‖ nonce (24) ‖ XChaCha20-Poly1305(body)`, where `route` is
//! derived from the direction's key and the message's number, so outsiders
//! cannot link messages to each other or to the opening, and the recipient
//! knows in advance which routes are its own ([`Session::expects`]).
//!
//! Bodies are padded to 32 bytes and capped at [`MAX_BODY`], so a session
//! message fits one 1,536-byte attachment and an opening two.
//!
//! # What this does not do
//!
//! No forward secrecy within a session: whoever later learns the recipient's
//! KEM secret key and kept an opening can read the session it opened. Start
//! a new session (a fresh encapsulation) to bound that.

use crate::chat::{ChatAttachment, MAX_ENCRYPTED_MESSAGE_SIZE};
use crate::core::types::hash;
use anyhow::{anyhow, bail, ensure, Result};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use ml_kem::{Decapsulate, DecapsulationKey, EncapsulationKey, KeyExport, MlKem512};
use serde::{Deserialize, Serialize};

/// An ML-KEM-512 public (encapsulation) key's length.
pub const KEM_PUBLIC_LEN: usize = 800;
/// An ML-KEM-512 ciphertext's length.
pub const KEM_CT_LEN: usize = 768;
/// A session message's route.
pub const ROUTE_LEN: usize = 16;
const NONCE_LEN: usize = 24;
const TAG_LEN: usize = 16;
const VERSION: u8 = 1;
const KIND_HELLO: u8 = 0;
const KIND_MESSAGE: u8 = 1;
/// The longest message body, in bytes.
pub const MAX_BODY: usize = 750;
/// Attachments one sealed chat may span.
pub const MAX_ATTACHMENTS: usize = crate::chat::MAX_CHAT_ATTACHMENTS;
/// Routes a recipient watches past the last message it opened, for messages
/// the bus delivers out of order or loses.
pub const WINDOW: u64 = 16;

fn derive(context: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key(context);
    for p in parts {
        h.update(p);
    }
    *h.finalize().as_bytes()
}

/// The hash a name's `kem` record holds.
pub fn kem_hash(public: &[u8]) -> [u8; 32] {
    hash(public)
}

/// `public` is an ML-KEM-512 key whose hash is `expected`.
pub fn check_public(public: &[u8], expected: &[u8; 32]) -> Result<()> {
    ensure!(public.len() == KEM_PUBLIC_LEN, "a KEM key is {KEM_PUBLIC_LEN} bytes, not {}", public.len());
    ensure!(&kem_hash(public) == expected, "that KEM key is not the one the name publishes");
    encapsulation_key(public).map(|_| ())
}

fn encapsulation_key(public: &[u8]) -> Result<EncapsulationKey<MlKem512>> {
    let key = ml_kem::Key::<EncapsulationKey<MlKem512>>::try_from(public).map_err(|_| anyhow!("a KEM key is {KEM_PUBLIC_LEN} bytes"))?;
    EncapsulationKey::<MlKem512>::new(&key).map_err(|_| anyhow!("not a valid ML-KEM-512 key"))
}

/// A wallet's KEM key pair.
pub struct KemKeys {
    dk: DecapsulationKey<MlKem512>,
}

impl KemKeys {
    /// The key pair a wallet derives from a secret of its own (a seed-derived
    /// value), so restoring the wallet restores it.
    pub fn from_secret(secret: &[u8; 32]) -> Self {
        let mut seed = [0u8; 64];
        seed[..32].copy_from_slice(&derive("midstate.sealed.kem-d.v1", &[secret]));
        seed[32..].copy_from_slice(&derive("midstate.sealed.kem-z.v1", &[secret]));
        Self { dk: DecapsulationKey::<MlKem512>::from_seed(ml_kem::Seed::from(seed)) }
    }

    /// The public key, as the handshake hands it out.
    pub fn public(&self) -> Vec<u8> {
        self.dk.encapsulation_key().to_bytes().to_vec()
    }

    /// What the name's `kem` record holds.
    pub fn public_hash(&self) -> [u8; 32] {
        kem_hash(&self.public())
    }
}

/// Who opened a session, as the opening claims.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Sender {
    /// No name: nothing about the opener is known.
    Anonymous,
    /// A name, the identity key the opener says holds it, and that key's
    /// signature over the opening's transcript.
    Named { name: String, identity: [u8; 32], signature: Vec<u8>, transcript: [u8; 32] },
}

impl Sender {
    /// Whether the claim holds, given the identity key the name's records
    /// give now: they must be the key that signed, and the signature good.
    pub fn verify(&self, identity_of_name: Option<[u8; 32]>) -> bool {
        match self {
            Sender::Anonymous => false,
            Sender::Named { identity, signature, transcript, .. } => {
                identity_of_name == Some(*identity)
                    && crate::core::mss::MssSignature::from_bytes(signature)
                        .is_ok_and(|s| crate::core::mss::verify(&s, transcript, identity))
            }
        }
    }

    pub fn name(&self) -> Option<&str> {
        match self {
            Sender::Named { name, .. } => Some(name),
            Sender::Anonymous => None,
        }
    }
}

/// What the opener's identity key signs: the recipient's key, the
/// encapsulation, and the claim. A signature cannot be moved to another
/// session or recipient.
pub fn transcript(recipient_kem_hash: &[u8; 32], ct: &[u8], name: &str, identity: &[u8; 32]) -> [u8; 32] {
    derive("midstate.sealed.transcript.v1", &[recipient_kem_hash, ct, &[name.len() as u8], name.as_bytes(), identity])
}

/// A session's state on one side. Persist it: it is what reads and writes
/// the conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// `H(ct)`, the same on both sides.
    pub id: [u8; 32],
    /// This side opened it.
    pub opener: bool,
    send_key: [u8; 32],
    recv_key: [u8; 32],
    /// The next number this side sends.
    pub sent: u64,
    /// The highest number opened from the other side (0: none yet).
    pub received: u64,
    /// Numbers opened lately, so a message replayed onto the bus (with a
    /// fresh proof of work) is not read twice.
    #[serde(default)]
    pub opened: Vec<u64>,
}

impl Session {
    fn new(ss: &[u8], ct: &[u8], opener: bool) -> Self {
        let to_recipient = derive("midstate.sealed.session.v1", &[ss, ct, &[0]]);
        let to_opener = derive("midstate.sealed.session.v1", &[ss, ct, &[1]]);
        let (send_key, recv_key) = if opener { (to_recipient, to_opener) } else { (to_opener, to_recipient) };
        Session { id: hash(ct), opener, send_key, recv_key, sent: 1, received: 0, opened: Vec::new() }
    }

    fn route(key: &[u8; 32], n: u64) -> [u8; ROUTE_LEN] {
        derive("midstate.sealed.route.v1", &[key, &n.to_le_bytes()])[..ROUTE_LEN].try_into().unwrap()
    }

    /// Seals `body` as this side's next message.
    pub fn seal(&mut self, body: &[u8]) -> Result<Vec<u8>> {
        let n = self.sent;
        let route = Self::route(&self.send_key, n);
        let nonce: [u8; NONCE_LEN] = rand::random();
        let mut aad = route.to_vec();
        aad.extend_from_slice(&n.to_le_bytes());
        let ct = XChaCha20Poly1305::new(Key::from_slice(&self.send_key))
            .encrypt(XNonce::from_slice(&nonce), Payload { msg: &plaintext(KIND_MESSAGE, &[], body)?, aad: &aad })
            .map_err(|_| anyhow!("sealing failed"))?;
        self.sent += 1;
        Ok([&route[..], &nonce, &ct].concat())
    }

    /// The routes of the other side's messages this side still expects.
    pub fn expects(&self) -> impl Iterator<Item = ([u8; ROUTE_LEN], u64)> + '_ {
        (self.received.saturating_sub(WINDOW).max(1)..=self.received + WINDOW).map(|n| (Self::route(&self.recv_key, n), n))
    }

    /// Opens the other side's message `n` (its route matched), or `None`.
    pub fn open(&mut self, n: u64, bytes: &[u8]) -> Option<Vec<u8>> {
        if bytes.len() < ROUTE_LEN + NONCE_LEN + TAG_LEN || bytes[..ROUTE_LEN] != Self::route(&self.recv_key, n) || self.opened.contains(&n) {
            return None;
        }
        let mut aad = bytes[..ROUTE_LEN].to_vec();
        aad.extend_from_slice(&n.to_le_bytes());
        let nonce = &bytes[ROUTE_LEN..ROUTE_LEN + NONCE_LEN];
        let plain = XChaCha20Poly1305::new(Key::from_slice(&self.recv_key))
            .decrypt(XNonce::from_slice(nonce), Payload { msg: &bytes[ROUTE_LEN + NONCE_LEN..], aad: &aad })
            .ok()?;
        let (kind, _, body) = parse_plaintext(&plain)?;
        (kind == KIND_MESSAGE).then(|| {
            self.received = self.received.max(n);
            self.opened.push(n);
            let keep = (2 * WINDOW + 1) as usize;
            if self.opened.len() > keep {
                self.opened.drain(..self.opened.len() - keep);
            }
            body
        })
    }
}

/// `v ‖ kind ‖ head ‖ body_len (2) ‖ body`, padded with zeros to 32 bytes.
fn plaintext(kind: u8, head: &[u8], body: &[u8]) -> Result<Vec<u8>> {
    ensure!(body.len() <= MAX_BODY, "a message holds at most {MAX_BODY} bytes");
    let mut p = vec![VERSION, kind];
    p.extend_from_slice(head);
    p.extend_from_slice(&(body.len() as u16).to_le_bytes());
    p.extend_from_slice(body);
    p.resize(p.len().div_ceil(32) * 32, 0);
    Ok(p)
}

/// The kind, the hello's head (empty for a message) and the body.
fn parse_plaintext(p: &[u8]) -> Option<(u8, Option<(String, [u8; 32], Vec<u8>)>, Vec<u8>)> {
    if p.len() < 4 || p[0] != VERSION {
        return None;
    }
    let mut i = 2;
    let head = if p[1] == KIND_HELLO {
        let n = *p.get(i)? as usize;
        let name = String::from_utf8(p.get(i + 1..i + 1 + n)?.to_vec()).ok()?;
        i += 1 + n;
        let identity: [u8; 32] = p.get(i..i + 32)?.try_into().ok()?;
        i += 32;
        let s = u16::from_le_bytes(p.get(i..i + 2)?.try_into().ok()?) as usize;
        let sig = p.get(i + 2..i + 2 + s)?.to_vec();
        i += 2 + s;
        Some((name, identity, sig))
    } else {
        None
    };
    let b = u16::from_le_bytes(p.get(i..i + 2)?.try_into().ok()?) as usize;
    if b > MAX_BODY {
        return None;
    }
    let body = p.get(i + 2..i + 2 + b)?.to_vec();
    // The padding is zeros: anything else is not ours.
    p[i + 2 + b..].iter().all(|x| *x == 0).then_some((p[1], head, body))
}

/// How the opener says who it is: its name, the identity key the name's
/// records give, and that key's signer.
pub struct OpenAs<'a> {
    pub name: &'a str,
    pub identity: [u8; 32],
    pub sign: Box<dyn FnOnce(&[u8; 32]) -> Result<Vec<u8>> + 'a>,
}

/// Opens a session with the holder of `recipient_public` (checked against
/// `recipient_hash`, the name's `kem` record), carrying `body` as its first
/// message. Returns the opening's bytes and this side's session.
pub fn open_session(recipient_public: &[u8], recipient_hash: &[u8; 32], me: Option<OpenAs>, body: &[u8]) -> Result<(Vec<u8>, Session)> {
    check_public(recipient_public, recipient_hash)?;
    let ek = encapsulation_key(recipient_public)?;
    let m: [u8; 32] = rand::random();
    let (ct, ss) = ek.encapsulate_deterministic(&m.into());
    let (ct, ss) = (ct.to_vec(), ss.to_vec());
    let head = match me {
        None => [&[0u8][..], &[0; 32], &[0, 0]].concat(),
        Some(me) => {
            ensure!(!me.name.is_empty() && me.name.len() <= crate::core::names::MAX_NAME_LEN, "not a name");
            let sig = (me.sign)(&transcript(recipient_hash, &ct, me.name, &me.identity))?;
            ensure!(sig.len() <= u16::MAX as usize, "signature too large");
            let mut h = vec![me.name.len() as u8];
            h.extend_from_slice(me.name.as_bytes());
            h.extend_from_slice(&me.identity);
            h.extend_from_slice(&(sig.len() as u16).to_le_bytes());
            h.extend_from_slice(&sig);
            h
        }
    };
    let k = derive("midstate.sealed.open.v1", &[&ss, &ct]);
    let sealed = XChaCha20Poly1305::new(Key::from_slice(&k))
        .encrypt(XNonce::from_slice(&[0; NONCE_LEN]), Payload { msg: &plaintext(KIND_HELLO, &head, body)?, aad: &ct })
        .map_err(|_| anyhow!("sealing failed"))?;
    let opening = [ct.as_slice(), &sealed].concat();
    ensure!(opening.len() <= MAX_ENCRYPTED_MESSAGE_SIZE * MAX_ATTACHMENTS, "opening too large");
    Ok((opening, Session::new(&ss, &ct, true)))
}

/// An opening this wallet's key opened.
#[derive(Clone, Debug)]
pub struct Opened {
    pub session: Session,
    pub sender: Sender,
    pub body: Vec<u8>,
}

/// Tries `bytes` as an opening to `keys`: `None` unless it is one, sealed to
/// this key. A non-recipient's decapsulation yields an unrelated key (ML-KEM
/// rejects implicitly), so the seal fails to open.
pub fn try_open(keys: &KemKeys, bytes: &[u8]) -> Option<Opened> {
    if bytes.len() < KEM_CT_LEN + TAG_LEN + 32 {
        return None;
    }
    let (ct, sealed) = bytes.split_at(KEM_CT_LEN);
    let ss = keys.dk.decapsulate_slice(ct).ok()?.to_vec();
    let k = derive("midstate.sealed.open.v1", &[&ss, ct]);
    let plain = XChaCha20Poly1305::new(Key::from_slice(&k)).decrypt(XNonce::from_slice(&[0; NONCE_LEN]), Payload { msg: sealed, aad: ct }).ok()?;
    let (kind, head, body) = parse_plaintext(&plain)?;
    let (name, identity, signature) = head.filter(|_| kind == KIND_HELLO)?;
    let sender = if name.is_empty() {
        Sender::Anonymous
    } else {
        let transcript = transcript(&keys.public_hash(), ct, &name, &identity);
        Sender::Named { name, identity, signature, transcript }
    };
    Some(Opened { session: Session::new(&ss, ct, false), sender, body })
}

/// The attachments that carry `envelope` (an opening or a message).
pub fn attachments(envelope: &[u8]) -> Result<Vec<ChatAttachment>> {
    let parts: Vec<ChatAttachment> = envelope.chunks(MAX_ENCRYPTED_MESSAGE_SIZE).map(|c| ChatAttachment::EncryptedMessage(c.to_vec())).collect();
    if parts.is_empty() || parts.len() > MAX_ATTACHMENTS {
        bail!("a sealed message takes one to {MAX_ATTACHMENTS} attachments");
    }
    Ok(parts)
}

/// A chat's sealed envelope: its `EncryptedMessage` attachments, joined in
/// order. `None` if it carries none.
pub fn envelope(atts: &[ChatAttachment]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for a in atts {
        if let ChatAttachment::EncryptedMessage(b) = a {
            out.extend_from_slice(b);
        }
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::chat_pow_units;
    use crate::core::mss;

    fn named<'a>(name: &'a str, key: &'a mut mss::MssKeypair) -> OpenAs<'a> {
        let identity = key.public_key();
        OpenAs { name, identity, sign: Box::new(move |t| Ok(key.sign(t)?.to_bytes())) }
    }

    #[test]
    fn a_session_opens_only_for_its_recipient_and_carries_both_ways() {
        let bob = KemKeys::from_secret(&[2; 32]);
        let eve = KemKeys::from_secret(&[3; 32]);
        assert_eq!(bob.public().len(), KEM_PUBLIC_LEN);
        assert_eq!(KemKeys::from_secret(&[2; 32]).public(), bob.public(), "from the seed, so a restore brings it back");
        assert_ne!(eve.public_hash(), bob.public_hash());

        let mut alice_id = mss::keygen(&[1; 32], 4).unwrap();
        let alice_identity = alice_id.public_key();
        let (opening, mut alice) = open_session(&bob.public(), &bob.public_hash(), Some(named("alice", &mut alice_id)), b"hello bob").unwrap();
        assert_eq!(alice_id.remaining(), 15, "one signature for the session");

        let atts = attachments(&opening).unwrap();
        assert_eq!(atts.len(), 2, "an opening with a signature spans two attachments");
        assert!(chat_pow_units(&atts) >= 3, "and costs proof of work by the byte");
        let got = envelope(&atts).unwrap();
        assert!(try_open(&eve, &got).is_none(), "only its recipient opens it");
        let opened = try_open(&bob, &got).expect("bob opens it");
        assert_eq!(opened.body, b"hello bob");
        assert_eq!(opened.sender.name(), Some("alice"));
        assert!(opened.sender.verify(Some(alice_identity)), "alice.mds's identity signed it");
        assert!(!opened.sender.verify(Some([9; 32])), "a name whose identity is another key");
        assert!(!opened.sender.verify(None));
        let mut bob_side = opened.session;
        assert_eq!(bob_side.id, alice.id);

        // Both ways, and only the other side's routes open.
        let m1 = alice.seal(b"how are you").unwrap();
        assert_eq!(attachments(&m1).unwrap().len(), 1);
        let (route, n) = bob_side.expects().find(|(r, _)| m1[..ROUTE_LEN] == r[..]).expect("bob expects it");
        assert_eq!(n, 1);
        assert_eq!(bob_side.open(n, &m1).unwrap(), b"how are you");
        assert!(bob_side.open(n, &m1).is_none(), "a replay is not read twice");
        assert!(alice.open(1, &m1).is_none(), "not alice's own direction");
        let r1 = bob_side.seal(b"fine").unwrap();
        let (_, n) = alice.expects().find(|(r, _)| r1[..ROUTE_LEN] == r[..]).unwrap();
        assert_eq!(alice.open(n, &r1).unwrap(), b"fine");
        let _ = route;

        // Tampering breaks the seal; routes do not repeat.
        let mut bad = alice.seal(b"x").unwrap();
        let last = bad.len() - 1;
        bad[last] ^= 1;
        assert!(bob_side.open(2, &bad).is_none());
        let m3 = alice.seal(b"y").unwrap();
        assert_ne!(m3[..ROUTE_LEN], m1[..ROUTE_LEN]);
        assert_eq!(bob_side.open(3, &m3).unwrap(), b"y", "a lost message does not stop the next");
    }

    #[test]
    fn anonymous_openings_and_limits() {
        let bob = KemKeys::from_secret(&[2; 32]);
        let (opening, _) = open_session(&bob.public(), &bob.public_hash(), None, &[7; MAX_BODY]).unwrap();
        assert_eq!(attachments(&opening).unwrap().len(), 2);
        let o = try_open(&bob, &opening).unwrap();
        assert_eq!((o.sender.clone(), o.body.len()), (Sender::Anonymous, MAX_BODY));
        assert!(!o.sender.verify(Some([0; 32])));
        assert!(open_session(&bob.public(), &bob.public_hash(), None, &[7; MAX_BODY + 1]).is_err());
        // The handshake's key must be the one the name publishes.
        let other = KemKeys::from_secret(&[4; 32]);
        assert!(open_session(&other.public(), &bob.public_hash(), None, b"hi").is_err());
        assert!(check_public(&bob.public()[1..], &bob.public_hash()).is_err());
        // A plain session message fits one attachment at the largest body.
        let (_, mut s) = open_session(&bob.public(), &bob.public_hash(), None, b"").unwrap();
        assert!(s.seal(&[1; MAX_BODY]).unwrap().len() <= MAX_ENCRYPTED_MESSAGE_SIZE);
        assert!(envelope(&[ChatAttachment::Address([1; 32])]).is_none());
    }

    #[test]
    fn a_signature_cannot_be_replayed_into_another_session() {
        let bob = KemKeys::from_secret(&[2; 32]);
        let mut id = mss::keygen(&[1; 32], 4).unwrap();
        let pk = id.public_key();
        let (opening, _) = open_session(&bob.public(), &bob.public_hash(), Some(named("alice", &mut id)), b"hi").unwrap();
        let Sender::Named { signature, .. } = try_open(&bob, &opening).unwrap().sender else { panic!() };
        // The same signature under another encapsulation fails: it signed the ct.
        let (other, _) = open_session(&bob.public(), &bob.public_hash(), None, b"hi").unwrap();
        let t = transcript(&bob.public_hash(), &other[..KEM_CT_LEN], "alice", &pk);
        let forged = Sender::Named { name: "alice".into(), identity: pk, signature, transcript: t };
        assert!(!forged.verify(Some(pk)));
    }
}
