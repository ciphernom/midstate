//! Sealed messages for the web wallet: `midstate::sealed` for the page.
//!
//! The page (`messages.js`, run by the worker) keeps the conversations; this
//! keeps nothing. Sessions and openings go back and forth as the JSON that
//! `sealed::Session` and `sealed::Prepared` serialize to, and bytes as hex.
//!
//! The KEM key comes from the wallet's master seed exactly as the desktop
//! wallet derives it (`walletd::service_identity::kem_keys`), so one recovery
//! phrase is one key wherever it is restored, and a name's `kem` record set
//! from the desktop app reads here too.
//!
//! Wasm has no threads: a stamp, or the chat bus's proof of work, is searched
//! a slice at a time ([`sealed_stamp_search`], [`chat_pow_search`]) and the
//! worker yields between slices, so a search of seconds holds nothing up.

use crate::WebWallet;
use midstate::chat::{self, ChatAttachment};
use midstate::core::mss::MssKeypair;
use midstate::sealed::{self, KemKeys, Prepared, Sender, Session};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// The context the desktop wallet derives its KEM secret from its master
/// seed under. The same here: one seed, one key.
const KEM_CONTEXT: &str = "midstate.walletd.sealed-kem.v1";

pub(crate) fn kem_keys(master_seed: &[u8; 32]) -> KemKeys {
    KemKeys::from_secret(&blake3::derive_key(KEM_CONTEXT, master_seed))
}

fn unhex(s: &str, what: &str) -> Result<Vec<u8>, String> {
    hex::decode(s.trim()).map_err(|_| format!("{what} is not hex"))
}

fn unhex32(s: &str, what: &str) -> Result<[u8; 32], String> {
    unhex(s, what)?.try_into().map_err(|_| format!("{what} is not 32 bytes"))
}

fn session_of(json: &str) -> Result<Session, String> {
    serde_json::from_str(json).map_err(|e| format!("not a session: {e}"))
}

fn to_json<T: Serialize>(v: &T) -> Result<String, String> {
    serde_json::to_string(v).map_err(|e| e.to_string())
}

// ── What goes back to the page ──────────────────────────────────────────────

/// An opening this wallet's key opened. `name` is empty for an anonymous
/// opener; for a named one, `identity`, `signature` and `transcript` are
/// what [`sealed_verify_sender`] checks once the page has the name's records.
#[derive(Serialize)]
struct OpenedView {
    session: Session,
    name: String,
    identity: String,
    signature: String,
    transcript: String,
    body: String,
    ttl: u32,
    stamp_bits: u32,
}

/// A message the other side sent in a session, and the session after it.
#[derive(Serialize)]
struct ReceivedView {
    session: Session,
    body: String,
    ttl: u32,
    stamp_bits: u32,
}

#[derive(Serialize)]
struct Finished {
    opening: String,
    session: Session,
}

#[derive(Serialize)]
struct Reserved {
    session: Session,
    n: u64,
    key: String,
}

// ── The work, testable off the browser ──────────────────────────────────────

pub(crate) fn try_open(keys: &KemKeys, envelope: &[u8]) -> Result<Option<String>, String> {
    let Some(o) = sealed::try_open(keys, envelope) else { return Ok(None) };
    let (name, identity, signature, transcript) = match &o.sender {
        Sender::Anonymous => (String::new(), String::new(), String::new(), String::new()),
        Sender::Named { name, identity, signature, transcript } => (name.clone(), hex::encode(identity), hex::encode(signature), hex::encode(transcript)),
    };
    let view = OpenedView { session: o.session, name, identity, signature, transcript, body: hex::encode(&o.body), ttl: o.ttl, stamp_bits: o.stamp_bits };
    to_json(&view).map(Some)
}

/// An opening to the holder of `public` (checked against `hash`, the name's
/// `kem` record), made up to its stamp. Signed as `me` (a name, and the MSS
/// key its `identity` record gives) if there is one; anonymous if not. The
/// signature takes one of the key's leaves: the page persists the key's
/// leaf counter before the opening can go anywhere.
pub(crate) fn prepare(public: &[u8], hash: &[u8; 32], me: Option<(&str, &mut MssKeypair)>) -> Result<String, String> {
    let prepared = match me {
        None => sealed::prepare_opening(public, hash, None),
        Some((name, kp)) => {
            if kp.remaining() == 0 {
                return Err("the identity key has no signatures left".into());
            }
            let identity = kp.master_pk;
            let sign = move |t: &[u8; 32]| kp.sign(t).map(|s| s.to_bytes());
            sealed::prepare_opening(public, hash, Some(sealed::OpenAs { name, identity, sign: Box::new(sign) }))
        }
    }
    .map_err(|e| format!("{e:#}"))?;
    to_json(&prepared)
}

fn prepared_of(json: &str) -> Result<Prepared, String> {
    serde_json::from_str(json).map_err(|e| format!("not an opening: {e}"))
}

pub(crate) fn finish(prepared_json: &str, body: &[u8], ttl: u32, stamp: u64) -> Result<String, String> {
    let (opening, session) = prepared_of(prepared_json)?.finish(body, ttl, stamp).map_err(|e| format!("{e:#}"))?;
    to_json(&Finished { opening: hex::encode(opening), session })
}

pub(crate) fn reserve(session_json: &str) -> Result<String, String> {
    let mut session = session_of(session_json)?;
    let (n, key) = session.reserve();
    to_json(&Reserved { session, n, key: hex::encode(key) })
}

pub(crate) fn seal(session_json: &str, n: u64, body: &[u8], ttl: u32, stamp: u64) -> Result<String, String> {
    let session = session_of(session_json)?;
    if n == 0 || n >= session.sent {
        return Err(format!("message {n} was not reserved"));
    }
    session.seal_reserved(n, body, ttl, stamp).map(hex::encode).map_err(|e| format!("{e:#}"))
}

pub(crate) fn routes(session_json: &str) -> Result<String, String> {
    let session = session_of(session_json)?;
    let v: Vec<(String, u64)> = session.expects().map(|(r, n)| (hex::encode(r), n)).collect();
    to_json(&v)
}

pub(crate) fn open(session_json: &str, n: u64, envelope: &[u8]) -> Result<Option<String>, String> {
    let mut session = session_of(session_json)?;
    let Some(got) = session.open(n, envelope) else { return Ok(None) };
    to_json(&ReceivedView { session, body: hex::encode(got.body), ttl: got.ttl, stamp_bits: got.stamp_bits }).map(Some)
}

/// The first nonce in `start .. start + count` whose stamp under `key` shows
/// `bits`, if one does.
pub(crate) fn stamp_search(key: &[u8; 32], bits: u32, start: u64, count: u32) -> Option<u64> {
    if bits == 0 {
        return Some(0);
    }
    (0..u64::from(count)).map(|i| start.wrapping_add(i)).find(|&n| sealed::stamp_bits(key, n) >= bits)
}

/// The bytes `chat::verify_chat_pow_v2` hashes, all but the nonce, absorbed:
/// the search then hashes only what the nonce changes. Checked against
/// `verify_chat_pow_v2` itself before a nonce is handed out.
fn chat_pow_prefix(sender: &str, timestamp: u64, reply_to: Option<u64>, words: &[u8], attachments: &[ChatAttachment]) -> blake3::Hasher {
    let mut h = blake3::Hasher::new();
    h.update(sender.as_bytes());
    h.update(&timestamp.to_le_bytes());
    h.update(&reply_to.unwrap_or(0).to_le_bytes());
    h.update(words);
    h.update(&(attachments.len() as u32).to_le_bytes());
    for a in attachments {
        match a {
            ChatAttachment::Address(b) => {
                h.update(&[0x01]);
                h.update(b);
            }
            ChatAttachment::CoinId(b) => {
                h.update(&[0x02]);
                h.update(b);
            }
            ChatAttachment::MixId(b) => {
                h.update(&[0x03]);
                h.update(b);
            }
            ChatAttachment::Commitment(b) => {
                h.update(&[0x04]);
                h.update(b);
            }
            ChatAttachment::BlockHash(b) => {
                h.update(&[0x05]);
                h.update(b);
            }
            ChatAttachment::Midstate(b) => {
                h.update(&[0x06]);
                h.update(b);
            }
            ChatAttachment::DataHash(b) => {
                h.update(&[0x07]);
                h.update(b);
            }
            ChatAttachment::LicenseChallenge { commitment, height, salt } => {
                h.update(&[0x08]);
                h.update(commitment);
                h.update(&height.to_le_bytes());
                h.update(salt);
            }
            ChatAttachment::Signature(b) => {
                h.update(&[0x09]);
                h.update(&(b.len() as u32).to_le_bytes());
                h.update(b);
            }
            ChatAttachment::EncryptedMessage(b) => {
                h.update(&[0x0A]);
                h.update(&(b.len() as u32).to_le_bytes());
                h.update(b);
            }
        }
    }
    h
}

/// The first nonce in `start .. start + count` that gives this chat its
/// proof of work (`chat::verify_chat_pow_v2`), if one does.
pub(crate) fn pow_search(
    sender: &str,
    timestamp: u64,
    reply_to: Option<u64>,
    words: &[u8],
    attachments: &[ChatAttachment],
    start: u64,
    count: u32,
) -> Result<Option<u64>, String> {
    let prefix = chat_pow_prefix(sender, timestamp, reply_to, words, attachments);
    let units = chat::chat_pow_units(attachments);
    let found = (0..u64::from(count)).map(|i| start.wrapping_add(i)).find(|n| {
        let mut h = prefix.clone();
        h.update(&n.to_le_bytes());
        chat::meets_chat_pow(h.finalize().as_bytes(), units)
    });
    match found {
        Some(n) if !chat::verify_chat_pow_v2(sender, timestamp, reply_to, words, attachments, n) => {
            Err("the chat's proof of work did not check out".into())
        }
        other => Ok(other),
    }
}

/// Whether a named opener's claim holds: the name's `identity` record now is
/// the key that signed, and the signature is good (`sealed::Sender::verify`).
pub(crate) fn verify_sender(name: &str, identity: &[u8; 32], signature: &[u8], transcript: &[u8; 32], identity_of_name: Option<[u8; 32]>) -> bool {
    let s = Sender::Named { name: name.to_string(), identity: *identity, signature: signature.to_vec(), transcript: *transcript };
    s.verify(identity_of_name)
}

// ── For the page ────────────────────────────────────────────────────────────

fn js(e: String) -> JsValue {
    JsValue::from_str(&e)
}

#[wasm_bindgen]
impl WebWallet {
    /// This wallet's ML-KEM-512 public key (800 bytes, hex): what the
    /// identity handshake hands to whoever asks (`CMD_KEM`).
    pub fn sealed_kem_public(&self) -> String {
        hex::encode(kem_keys(&self.master_seed).public())
    }

    /// What a name's `kem` record must hold for messages sealed to it to
    /// open here.
    pub fn sealed_kem_hash(&self) -> String {
        hex::encode(kem_keys(&self.master_seed).public_hash())
    }

    /// Tries an envelope (a sealed chat's `encrypted_message` attachments,
    /// joined) as an opening sealed to this wallet's key: its JSON, or
    /// undefined if it is not one.
    pub fn sealed_try_open(&self, envelope_hex: &str) -> Result<Option<String>, JsValue> {
        let envelope = unhex(envelope_hex, "the envelope").map_err(js)?;
        try_open(&kem_keys(&self.master_seed), &envelope).map_err(js)
    }

    /// An opening to the holder of `recipient_public_hex`, made up to its
    /// stamp (`sealed::prepare_opening`), as JSON. With `as_name`, signed
    /// as that name by the MSS key `identity_pk_hex` (in this wallet), which
    /// spends one of its leaves: the page must save the key's new leaf
    /// counter before the opening goes anywhere.
    pub fn sealed_prepare_opening(
        &mut self,
        recipient_public_hex: &str,
        recipient_hash_hex: &str,
        as_name: Option<String>,
        identity_pk_hex: Option<String>,
    ) -> Result<String, JsValue> {
        let public = unhex(recipient_public_hex, "the key").map_err(js)?;
        let hash = unhex32(recipient_hash_hex, "the key's hash").map_err(js)?;
        let name = as_name.filter(|n| !n.is_empty());
        let Some(name) = name else { return prepare(&public, &hash, None).map_err(js) };
        let pk = unhex32(identity_pk_hex.as_deref().unwrap_or(""), "the identity key").map_err(js)?;
        let kp = self
            .mss_cache
            .values_mut()
            .find(|k| k.master_pk == pk)
            .ok_or_else(|| JsValue::from_str("The identity key's tree is not loaded. Wait for the wallet to finish opening."))?;
        prepare(&public, &hash, Some((&name, kp))).map_err(js)
    }
}

/// `public_hex` is an ML-KEM-512 key whose hash is `hash_hex` (a name's
/// `kem` record): a key a name's wallet handed over is checked with this.
#[wasm_bindgen]
pub fn sealed_check_public(public_hex: &str, hash_hex: &str) -> Result<(), JsValue> {
    let public = unhex(public_hex, "the key").map_err(js)?;
    let hash = unhex32(hash_hex, "the hash").map_err(js)?;
    sealed::check_public(&public, &hash).map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

/// The key an opening's stamp is worked under (hex).
#[wasm_bindgen]
pub fn sealed_prepared_stamp_key(prepared_json: &str) -> Result<String, JsValue> {
    Ok(hex::encode(prepared_of(prepared_json).map_err(js)?.stamp_key()))
}

/// The `kem` hash of the key an opening was sealed to (hex).
#[wasm_bindgen]
pub fn sealed_prepared_recipient(prepared_json: &str) -> Result<String, JsValue> {
    Ok(hex::encode(prepared_of(prepared_json).map_err(js)?.recipient_hash()))
}

/// An opening made up to its stamp, finished: `{"opening": hex, "session": …}`,
/// carrying `body_hex` under the timer `ttl` with the stamp nonce `stamp`.
#[wasm_bindgen]
pub fn sealed_finish_opening(prepared_json: &str, body_hex: &str, ttl: u32, stamp: u64) -> Result<String, JsValue> {
    let body = unhex(body_hex, "the message").map_err(js)?;
    finish(prepared_json, &body, ttl, stamp).map_err(js)
}

/// This side's next number in a session, taken: `{"session": …, "n": n,
/// "key": hex}`, the key its stamp is worked under.
#[wasm_bindgen]
pub fn sealed_session_reserve(session_json: &str) -> Result<String, JsValue> {
    reserve(session_json).map_err(js)
}

/// The key the stamp of this side's message `n` is worked under (hex).
#[wasm_bindgen]
pub fn sealed_session_stamp_key(session_json: &str, n: u32) -> Result<String, JsValue> {
    Ok(hex::encode(session_of(session_json).map_err(js)?.stamp_key(u64::from(n))))
}

/// This side's message `n` (reserved), sealed: the envelope, hex.
#[wasm_bindgen]
pub fn sealed_session_seal(session_json: &str, n: u32, body_hex: &str, ttl: u32, stamp: u64) -> Result<String, JsValue> {
    let body = unhex(body_hex, "the message").map_err(js)?;
    seal(session_json, u64::from(n), &body, ttl, stamp).map_err(js)
}

/// The routes of the other side's messages this side still expects:
/// `[[route hex, n], …]`.
#[wasm_bindgen]
pub fn sealed_session_routes(session_json: &str) -> Result<String, JsValue> {
    routes(session_json).map_err(js)
}

/// The other side's message `n` (its route matched), opened: `{"session": …,
/// "body": hex, "ttl", "stamp_bits"}`, or undefined.
#[wasm_bindgen]
pub fn sealed_session_open(session_json: &str, n: u32, envelope_hex: &str) -> Result<Option<String>, JsValue> {
    let envelope = unhex(envelope_hex, "the envelope").map_err(js)?;
    open(session_json, u64::from(n), &envelope).map_err(js)
}

/// A slice of a stamp's search: the first nonce from `start` (of `count`)
/// whose work under `key_hex` shows `bits`, or undefined.
#[wasm_bindgen]
pub fn sealed_stamp_search(key_hex: &str, bits: u32, start: u64, count: u32) -> Result<Option<u64>, JsValue> {
    let key = unhex32(key_hex, "the stamp's key").map_err(js)?;
    Ok(stamp_search(&key, bits, start, count))
}

/// The leading zero bits a stamp nonce shows under `key_hex`.
#[wasm_bindgen]
pub fn sealed_stamp_bits(key_hex: &str, nonce: u64) -> Result<u32, JsValue> {
    let key = unhex32(key_hex, "the stamp's key").map_err(js)?;
    Ok(sealed::stamp_bits(&key, nonce))
}

/// Whether a named opener is who it says (`sealed::Sender::verify`), given
/// the identity key its name's records give now (empty: none).
#[wasm_bindgen]
pub fn sealed_verify_sender(name: &str, identity_hex: &str, signature_hex: &str, transcript_hex: &str, identity_of_name_hex: &str) -> bool {
    let (Ok(identity), Ok(signature), Ok(transcript)) = (unhex32(identity_hex, ""), unhex(signature_hex, ""), unhex32(transcript_hex, "")) else {
        return false;
    };
    let of_name = unhex32(identity_of_name_hex, "").ok();
    verify_sender(name, &identity, &signature, &transcript, of_name)
}

/// A slice of a chat's proof-of-work search, as `mine_chat_pow_v2_wasm`
/// takes the chat: the first good nonce from `start` (of `count`), or
/// undefined. The worker yields between slices.
#[wasm_bindgen]
pub fn chat_pow_search(
    sender: &str,
    timestamp: u64,
    reply_to_json: &str,
    words_json: &str,
    attachments_json: &str,
    start: u64,
    count: u32,
) -> Result<Option<u64>, JsValue> {
    let reply_to: Option<u64> = serde_json::from_str(reply_to_json).map_err(|e| JsValue::from_str(&format!("Invalid reply_to JSON: {e}")))?;
    let words: Vec<u8> = serde_json::from_str(words_json).map_err(|e| JsValue::from_str(&format!("Invalid words JSON: {e}")))?;
    let attachments: Vec<ChatAttachment> =
        serde_json::from_str(attachments_json).map_err(|e| JsValue::from_str(&format!("Invalid attachments JSON: {e}")))?;
    pow_search(sender, timestamp, reply_to, &words, &attachments, start, count).map_err(js)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::mss;
    use serde_json::Value;

    fn bob() -> ([u8; 32], KemKeys) {
        let seed = midstate::core::types::hash(b"bob's master seed");
        (seed, kem_keys(&seed))
    }

    fn first_nonce(key: &[u8; 32], bits: u32) -> u64 {
        let mut start = 0u64;
        loop {
            if let Some(n) = stamp_search(key, bits, start, 1 << 14) {
                return n;
            }
            start += 1 << 14;
        }
    }

    fn json(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn the_key_is_the_desktop_wallets_for_the_same_seed() {
        let (seed, keys) = bob();
        let desktop = KemKeys::from_secret(&blake3::derive_key("midstate.walletd.sealed-kem.v1", &seed));
        assert_eq!(keys.public(), desktop.public());
        assert_eq!(keys.public().len(), sealed::KEM_PUBLIC_LEN);
        let other = kem_keys(&midstate::core::types::hash(b"someone else"));
        assert_ne!(other.public_hash(), keys.public_hash());
    }

    /// What the page does, as JSON goes back and forth: an anonymous
    /// opening, stamped a slice at a time, then messages both ways.
    #[test]
    fn an_opening_and_a_conversation_as_the_page_drives_them() {
        const BITS: u32 = 10;
        let (_, keys) = bob();
        let public = keys.public();
        let hash = keys.public_hash();

        let prepared = prepare(&public, &hash, None).unwrap();
        assert!(prepare(&public[1..], &hash, None).is_err(), "a key that is not the name's");
        let key = prepared_of(&prepared).unwrap().stamp_key();
        let nonce = first_nonce(&key, BITS);
        assert!(sealed::stamp_bits(&key, nonce) >= BITS);
        let done = json(&finish(&prepared, "hello bob".as_bytes(), 3_600, nonce).unwrap());
        let opening = hex::decode(done["opening"].as_str().unwrap()).unwrap();
        let alice = done["session"].to_string();

        let opened = json(&try_open(&keys, &opening).unwrap().expect("bob opens it"));
        assert_eq!(opened["name"], "", "anonymous");
        assert_eq!(hex::decode(opened["body"].as_str().unwrap()).unwrap(), b"hello bob");
        assert_eq!(opened["ttl"], 3_600);
        assert!(opened["stamp_bits"].as_u64().unwrap() >= u64::from(BITS));
        let other = kem_keys(&[9; 32]);
        assert!(try_open(&other, &opening).unwrap().is_none(), "only its recipient opens it");
        let bob_side = opened["session"].to_string();

        // Alice's message 1, unstamped.
        let r = json(&reserve(&alice).unwrap());
        assert_eq!(r["n"], 1);
        let alice = r["session"].to_string();
        assert!(seal(&alice, 2, b"x", 0, 0).is_err(), "a number not taken");
        let m1 = hex::decode(seal(&alice, 1, b"how are you", 0, 0).unwrap()).unwrap();
        let want: Vec<(String, u64)> = serde_json::from_str(&routes(&bob_side).unwrap()).unwrap();
        let (_, n) = want.iter().find(|(r, _)| hex::decode(r).unwrap() == m1[..sealed::ROUTE_LEN]).expect("bob expects it");
        let got = json(&open(&bob_side, *n, &m1).unwrap().unwrap());
        assert_eq!(hex::decode(got["body"].as_str().unwrap()).unwrap(), b"how are you");
        let bob_side = got["session"].to_string();
        assert!(open(&bob_side, *n, &m1).unwrap().is_none(), "a replay is not read twice");

        // Bob's reply, stamped.
        let r = json(&reserve(&bob_side).unwrap());
        let n = r["n"].as_u64().unwrap();
        let key = unhex32(r["key"].as_str().unwrap(), "").unwrap();
        let bob_side = r["session"].to_string();
        let stamp = first_nonce(&key, BITS);
        let back = hex::decode(seal(&bob_side, n, b"fine", 0, stamp).unwrap()).unwrap();
        let got = json(&open(&alice, n, &back).unwrap().unwrap());
        assert!(got["stamp_bits"].as_u64().unwrap() >= u64::from(BITS));
        let _ = bob_side;
    }

    #[test]
    fn a_named_opening_spends_one_leaf_and_checks_out_against_the_identity() {
        let (_, keys) = bob();
        // A small tree: WOTS keys are slow to make unoptimised.
        let mut kp = mss::keygen(&[5; 32], 1).unwrap();
        let identity = kp.master_pk;
        let prepared = prepare(&keys.public(), &keys.public_hash(), Some(("alice", &mut kp))).unwrap();
        assert_eq!(kp.remaining(), 1, "one leaf for the session");
        let done = json(&finish(&prepared, b"hi", 0, 0).unwrap());
        let opening = hex::decode(done["opening"].as_str().unwrap()).unwrap();
        let o = json(&try_open(&keys, &opening).unwrap().unwrap());
        assert_eq!(o["name"], "alice");
        let id = unhex32(o["identity"].as_str().unwrap(), "").unwrap();
        let sig = hex::decode(o["signature"].as_str().unwrap()).unwrap();
        let t = unhex32(o["transcript"].as_str().unwrap(), "").unwrap();
        assert_eq!(id, identity);
        assert!(verify_sender("alice", &id, &sig, &t, Some(identity)));
        assert!(!verify_sender("alice", &id, &sig, &t, Some([1; 32])), "the name lists another key");
        assert!(!verify_sender("alice", &id, &sig, &t, None));
        assert!(!sealed_verify_sender("alice", &hex::encode(id), &hex::encode(&sig), &hex::encode(t), ""));
        assert!(sealed_verify_sender("alice", &hex::encode(id), &hex::encode(&sig), &hex::encode(t), &hex::encode(identity)));

        kp.sign(&[0; 32]).unwrap();
        assert!(prepare(&keys.public(), &keys.public_hash(), Some(("alice", &mut kp))).is_err(), "no leaves left");
    }

    /// The attachments as the page writes them (`messages.js`): a key
    /// request's (the request id, the version-only payload, the identity
    /// asked, checksummed) and a sealed chunk. The first good nonce from
    /// zero is the one the node's own miner finds, so the slices hash what
    /// `verify_chat_pow_v2` hashes, for every kind the page sends.
    #[test]
    fn the_chat_proof_of_work_is_the_nodes() {
        use midstate::core::channel::wire;
        let (_, keys) = bob();
        let (_, mut s) = sealed::open_session(&keys.public(), &keys.public_hash(), None, b"", 0, 0).unwrap();
        let sealed_chunk = s.seal(b"hi", 0, 0).unwrap();
        assert!(sealed_chunk.len() <= chat::ENCRYPTED_POW_UNIT, "one unit of work");
        let identity = [0x5a; 32];
        let page = serde_json::json!([
            { "kind": "coin_id", "value": hex::encode([0x11; 32]) },
            { "kind": "signature", "value": "0200000000" },
            { "kind": "address", "value": midstate::core::types::encode_address_with_checksum(&identity) },
            { "kind": "encrypted_message", "value": hex::encode(&sealed_chunk) },
        ]);
        let atts: Vec<ChatAttachment> = serde_json::from_value(page).unwrap();
        assert_eq!(atts[1], ChatAttachment::Signature(wire::pack_u32(0, &[])));
        assert_eq!(atts[2], ChatAttachment::Address(identity));
        assert_eq!(sealed::envelope(&atts).unwrap(), sealed_chunk);
        let sender = hex::encode([7u8; 32]);
        let words = [wire::MARKER, wire::CMD_KEM_REQ];
        let first = (0u64..)
            .step_by(1 << 16)
            .find_map(|s| pow_search(&sender, 1_700_000_000, None, &words, &atts, s, 1 << 16).unwrap())
            .unwrap();
        assert_eq!(first, chat::mine_chat_pow_v2(sender.clone(), 1_700_000_000, None, words.to_vec(), atts.clone()));
        assert_eq!(pow_search(&sender, 1_700_000_000, None, &words, &atts, first, 1).unwrap(), Some(first), "from any start");
        assert_eq!(pow_search(&sender, 1_700_000_001, None, &words, &atts, first, 1).unwrap(), None, "bound to the chat");
        assert!(serde_json::from_value::<Vec<ChatAttachment>>(serde_json::json!([{ "kind": "encrypted_message", "value": "zz" }])).is_err());
    }

    #[test]
    fn a_stamp_search_finds_only_enough_work() {
        let key = [3u8; 32];
        assert_eq!(stamp_search(&key, 0, 99, 1), Some(0), "no work asked, none done");
        let n = first_nonce(&key, 12);
        assert!(sealed::stamp_bits(&key, n) >= 12);
        assert_eq!(stamp_search(&key, 12, n, 1), Some(n));
        assert_eq!(stamp_search(&key, 12, 0, n as u32), None, "nothing before the first");
        assert_eq!(stamp_search(&key, 64, 0, 1 << 10), None);
    }
}
