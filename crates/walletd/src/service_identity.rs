//! The identity handshake for names, and sealed messages between names.
//!
//! A name's `identity` record is this wallet's channel identity: an MSS key
//! that people reach over the chat bus (qbolt frames, `core::channel::wire`).
//! Asked by name, it answers with
//!
//! * a fresh one-time address, signed by the identity (`CMD_ADDR_REQ`), which
//!   a payment to the name now prefers over the name's fixed records — if
//!   the wallet does not answer within [`HANDSHAKE_WAIT`] the payment goes to
//!   the record, as before;
//! * an invoice, or a channel, as it already did for a typed key;
//! * its full ML-KEM public key (`CMD_KEM_REQ`), checked against the name's
//!   `kem` record, so anyone can open a sealed session with it.
//!
//! Sealed messages (`midstate::sealed`) ride the bus as `EncryptedMessage`
//! attachments: every wallet trial-decrypts every one, and keeps the
//! sessions it opened or was opened into in `<wallet>.messages.json`.
//!
//! A child of `service`, sharing the actor's state, as `service_names` is.

use super::{now_secs, Service};
use crate::api::{ConversationView, MessageLineView, SendStage, WalletEvent};
use anyhow::{anyhow, bail, Context, Result};
use midstate::chat::ChatAttachment;
use midstate::core::channel::wire as w;
use midstate::core::names::{self, Rec};
use midstate::sealed::{self, KemKeys, Sender, Session};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// How long a payment to a name waits for its wallet's fresh address before
/// paying the name's record instead.
pub const HANDSHAKE_WAIT: Duration = Duration::from_secs(20);
/// A handed-out address must outlive the payment's commit and reveal.
const ADDRESS_MARGIN: u64 = 30;
/// KEM keys handed out per block, at most: each answer costs chat proof of work.
const KEM_ANSWERS_PER_BLOCK: usize = 8;
/// An unanswered KEM request is asked again after this many blocks.
const KEM_RETRY: u64 = 3;
/// Messages waiting for a recipient's key give up after this many seconds.
const OUTBOX_PATIENCE: u64 = 15 * 60;

/// A payment to a name, waiting for the name's wallet to hand over a fresh
/// address. Not persisted: whoever asked is waiting on `resp`.
pub(super) struct NameSend {
    to: String,
    amount: u64,
    private: bool,
    identity: [u8; 32],
    deadline: Instant,
    resp: oneshot::Sender<Result<String>>,
}

/// A conversation: one sealed session and what was said in it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Conversation {
    pub session: Session,
    /// The other side's name: the one this wallet wrote to, or the one the
    /// opener claimed. Empty for an anonymous opener.
    pub peer: String,
    /// The other side is who `peer` says: this wallet sealed to the key that
    /// name publishes, or the opener's claim checked out against the registry.
    pub verified: bool,
    pub lines: Vec<Line>,
    pub started: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Line {
    pub mine: bool,
    pub text: String,
    pub at: u64,
}

/// A message waiting for its recipient's KEM key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Queued {
    pub to: String,
    pub text: String,
    pub queued: u64,
}

/// What the wallet keeps of sealed messaging. Persisted next to the wallet.
#[derive(Serialize, Deserialize)]
pub(super) struct SealedBook {
    #[serde(default)]
    pub chats: Vec<Conversation>,
    #[serde(default)]
    pub outbox: Vec<Queued>,
    /// KEM key requests sent: request id hex → (name, the `kem` hash it lists, asked at).
    #[serde(default)]
    pub kem_reqs: HashMap<String, (String, String, u64)>,
    /// Keys names' wallets handed over: name → public key hex.
    #[serde(default)]
    pub peer_kems: HashMap<String, String>,
    /// Sealed chats already read: (timestamp, pow nonce, sender).
    #[serde(default)]
    pub seen: VecDeque<(u64, u64, String)>,
    /// A wallet with no recovery seed keeps its KEM secret here (hex).
    #[serde(default)]
    pub legacy_secret: Option<String>,
    /// The MSB-1 backing (blocks) a sender's name needs for its messages to
    /// reach the inbox; below it they are message requests. `None`: every
    /// message reaches the inbox, saved as `null`. A new book, and one saved
    /// before there was a bar, take verified names only (`Some(0.0)`).
    #[serde(default = "verified_names")]
    pub inbox_bar: Option<f64>,
}

/// The inbox's bar until its user picks another: verified names, so a
/// stranger's first message is a request, not an interruption.
fn verified_names() -> Option<f64> {
    Some(0.0)
}

impl Default for SealedBook {
    fn default() -> Self {
        SealedBook {
            chats: Vec::new(),
            outbox: Vec::new(),
            kem_reqs: HashMap::new(),
            peer_kems: HashMap::new(),
            seen: VecDeque::new(),
            legacy_secret: None,
            inbox_bar: verified_names(),
        }
    }
}

impl SealedBook {
    fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("messages.json")
    }
    pub fn load(wallet_path: &Path) -> Self {
        std::fs::read(Self::path(wallet_path)).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default()
    }
    pub fn save(&self, wallet_path: &Path) {
        let path = Self::path(wallet_path);
        let tmp = path.with_extension("tmp");
        let ok = serde_json::to_vec(self).ok().and_then(|b| std::fs::write(&tmp, b).ok()).and_then(|_| std::fs::rename(&tmp, &path).ok());
        if ok.is_none() {
            tracing::warn!("could not save sealed messages to {}", path.display());
        }
    }
    fn mark_seen(&mut self, ts: u64, nonce: u64, sender: &str) -> bool {
        if self.seen.iter().any(|(t, n, s)| *t == ts && *n == nonce && s == sender) {
            return false;
        }
        self.seen.push_back((ts, nonce, sender.to_string()));
        while self.seen.len() > 2000 {
            self.seen.pop_front();
        }
        true
    }
}

/// The actor's live state for this module.
#[derive(Default)]
pub(super) struct IdentityLive {
    pub sends: Vec<NameSend>,
    pub book: SealedBook,
    pub kem: Option<KemKeys>,
    /// KEM keys handed out at this height, and how many.
    pub answered: (u64, usize),
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    hex::decode(s).ok()?.try_into().ok()
}

/// The identity a payment to a name with `recs` should ask for a fresh
/// address: its `identity` record, unless that is `mine`, or the name points
/// at a shared address whose published script says so (a fresh address of
/// one wallet would go around a vault, a multisig or a plan).
pub(super) fn handshake_with(recs: &[Rec], mine: Option<[u8; 32]>, used: impl Fn(&[u8; 32]) -> u64) -> Option<[u8; 32]> {
    let id = names::identity(recs)?;
    if Some(id) == mine {
        return None;
    }
    if let Some(pays) = names::pays(recs, used) {
        if names::address_kind(recs, &pays).is_some_and(|k| k.is_shared()) {
            return None;
        }
    }
    Some(id)
}

impl Service {
    pub(super) fn identity_load(&mut self) {
        self.ident.book = SealedBook::load(&self.wallet_path);
        self.ident.kem = None;
    }

    /// On lock: the KEM secret leaves memory, and payments still waiting on
    /// a name's wallet are refused rather than left hanging.
    pub(super) fn identity_unload(&mut self) {
        self.ident.kem = None;
        for s in std::mem::take(&mut self.ident.sends) {
            let _ = s.resp.send(Err(anyhow!("wallet is locked")));
        }
    }

    fn sealed_save(&self) {
        self.ident.book.save(&self.wallet_path);
    }

    /// This wallet's KEM keys: from its recovery seed, so a restore brings
    /// them back; a wallet without one keeps a random secret instead.
    pub(super) fn kem_keys(&mut self) -> Result<&KemKeys> {
        if self.ident.kem.is_none() {
            let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
            let secret = match w.data.master_seed {
                Some(seed) => blake3::derive_key("midstate.walletd.sealed-kem.v1", &seed),
                None => match self.ident.book.legacy_secret.as_deref().and_then(hex32) {
                    Some(s) => s,
                    None => {
                        let s: [u8; 32] = rand::random();
                        self.ident.book.legacy_secret = Some(hex::encode(s));
                        self.sealed_save();
                        s
                    }
                },
            };
            self.ident.kem = Some(KemKeys::from_secret(&secret));
        }
        Ok(self.ident.kem.as_ref().expect("just set"))
    }

    /// The hash this wallet's `kem` record carries.
    pub(super) fn kem_record(&mut self) -> Result<[u8; 32]> {
        Ok(self.kem_keys()?.public_hash())
    }

    /// A peer's identity key, typed as 64 hex characters or as a name whose
    /// `identity` record gives it.
    pub(super) fn peer_key(&self, s: &str) -> Result<[u8; 32]> {
        if names::is_name(s) {
            let r = self.live_record(s)?;
            return names::identity(&r.records).ok_or_else(|| anyhow!("{} lists no identity key to ask", s.trim()));
        }
        hex32(s.trim()).ok_or_else(|| anyhow!("give a name.mds or a 64-character key"))
    }

    // ── Paying a name: a fresh address first ────────────────────────────

    /// The identity to ask for a fresh address when paying `to`, if any: a
    /// name with an identity record that is not this wallet's, whose records
    /// do not point at a shared address (a vault, multisig or plan is paid as
    /// it is: a fresh address of one wallet would go around it).
    fn identity_to_ask(&self, to: &str) -> Option<[u8; 32]> {
        if !names::is_name(to) {
            return None;
        }
        let r = self.live_record(to).ok()?;
        handshake_with(&r.records, self.book.identity_pk, |k| self.node.storage.query_mss_leaf_index(k).unwrap_or(0))
    }

    /// A fresh address `identity` handed over and that is still good, taken
    /// so it is never paid twice.
    fn take_fresh_address(&mut self, identity: &[u8; 32], tip: u64) -> Option<[u8; 32]> {
        let key = hex::encode(identity);
        let (addr, expiry) = self.book.peer_addrs.get(&key).cloned()?;
        if expiry <= tip + ADDRESS_MARGIN {
            return None;
        }
        let a = midstate::core::parse_address_flexible(&addr).ok()?;
        self.book.peer_addrs.remove(&key);
        self.book.save(&self.wallet_path);
        Some(a)
    }

    /// A send asked for by the UI. To a name whose wallet can be asked, ask
    /// it for a fresh address and answer once it replies (or the wait runs
    /// out and the record is paid); otherwise send now.
    pub(super) async fn send_or_ask(&mut self, to: String, amount: u64, private: bool, resp: oneshot::Sender<Result<String>>) {
        if let Some(identity) = self.identity_to_ask(&to) {
            let tip = self.node.get_state().await.height;
            if let Some(dest) = self.take_fresh_address(&identity, tip) {
                let r = self.send_fresh(&to, dest, amount, private).await;
                let _ = resp.send(r);
                return;
            }
            if self.wallet.is_some() && !self.node.is_syncing() {
                match self.request_address(&hex::encode(identity)).await {
                    Ok(()) => {
                        self.ident.sends.push(NameSend { to, amount, private, identity, deadline: Instant::now() + HANDSHAKE_WAIT, resp });
                        return;
                    }
                    Err(e) => tracing::debug!("could not ask {to} for a fresh address: {e:#}"),
                }
            }
        }
        let r = self.start_send(&to, amount, private).await;
        let _ = resp.send(r);
    }

    async fn send_fresh(&mut self, to: &str, dest: [u8; 32], amount: u64, private: bool) -> Result<String> {
        let id = self.start_send_to(to, dest, amount, private).await?;
        if let Some(c) = hex32(&id) {
            self.set_stage(c, SendStage::Committing, format!("paying a fresh one-time address from {}'s wallet", to.trim()));
        }
        Ok(id)
    }

    /// Each tick: finish the payments whose fresh address came, or whose
    /// wait ran out.
    pub(super) async fn tick_name_sends(&mut self, tip: u64) {
        let waiting = std::mem::take(&mut self.ident.sends);
        for s in waiting {
            if let Some(dest) = self.take_fresh_address(&s.identity, tip) {
                let r = self.send_fresh(&s.to, dest, s.amount, s.private).await;
                let _ = s.resp.send(r);
            } else if Instant::now() >= s.deadline || s.resp.is_closed() {
                let r = self.start_send(&s.to, s.amount, s.private).await;
                if let Ok(c) = r.as_deref().map(hex32) {
                    if let Some(c) = c {
                        let note = format!("{}'s wallet did not answer, so this pays the address its name lists", s.to.trim());
                        self.set_stage(c, SendStage::Committing, note);
                    }
                }
                let _ = s.resp.send(r);
            } else {
                self.ident.sends.push(s);
            }
        }
    }

    // ── KEM keys over the handshake ─────────────────────────────────────

    /// Someone asked this wallet's identity for its KEM key.
    pub(super) fn answer_kem_request(&mut self, req_id: [u8; 32], tip: u64) -> Result<()> {
        if self.ident.answered.0 != tip {
            self.ident.answered = (tip, 0);
        }
        if self.ident.answered.1 >= KEM_ANSWERS_PER_BLOCK {
            bail!("answered enough KEM requests this block");
        }
        let public = self.kem_keys()?.public();
        crate::channels::send_frame(&self.node, w::CMD_KEM, req_id, public, None)?;
        self.ident.answered.1 += 1;
        Ok(())
    }

    /// A KEM key came back: keep it if it is the one the name lists, and send
    /// what was waiting for it.
    pub(super) async fn on_kem_reply(&mut self, req_id: [u8; 32], public: Vec<u8>) -> Result<()> {
        let Some((name, hash, _)) = self.ident.book.kem_reqs.get(&hex::encode(req_id)).cloned() else { return Ok(()) };
        let expected = hex32(&hash).ok_or_else(|| anyhow!("bad stored hash"))?;
        sealed::check_public(&public, &expected).context("a KEM key that is not the one the name lists")?;
        self.ident.book.kem_reqs.retain(|_, (n, ..)| n != &name);
        self.ident.book.peer_kems.insert(name.clone(), hex::encode(&public));
        self.sealed_save();
        self.flush_outbox(&name).await;
        Ok(())
    }

    fn ask_kem(&mut self, name: &str, identity: [u8; 32], hash: [u8; 32], tip: u64) -> Result<()> {
        let pending = self.ident.book.kem_reqs.values().any(|(n, _, at)| n == name && tip < at + KEM_RETRY);
        if pending {
            return Ok(());
        }
        let req_id: [u8; 32] = rand::random();
        let mut atts = crate::channels::frame_attachments(req_id, w::pack_u32(0, &[]), None);
        atts.push(ChatAttachment::Address(identity));
        self.node.send_chat(vec![w::MARKER, w::CMD_KEM_REQ], None, atts)?;
        self.ident.book.kem_reqs.retain(|_, (n, ..)| n != name);
        self.ident.book.kem_reqs.insert(hex::encode(req_id), (name.to_string(), hex::encode(hash), tip));
        self.sealed_save();
        Ok(())
    }

    // ── Sealed messages ─────────────────────────────────────────────────

    /// The name this wallet speaks as: one it holds whose identity record is
    /// this wallet's identity, so a recipient can check the claim.
    fn speaking_name(&self) -> Option<String> {
        let me = self.book.identity_pk?;
        self.names.jobs.iter().filter(|j| j.status == "registered" && !j.renter).find_map(|j| {
            let r = self.live_record(&format!("{}{}", j.name, names::SUFFIX)).ok()?;
            (names::identity(&r.records) == Some(me)).then(|| j.name.clone())
        })
    }

    /// Send `text` to `to` (a name), sealed. Within an open session it goes
    /// at once; otherwise it opens one with the name's KEM key, asking its
    /// wallet for the key first if need be.
    pub(super) async fn send_sealed(&mut self, to: &str, text: &str) -> Result<()> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let name = names::normalize(to)?;
        let text = text.trim();
        if text.is_empty() {
            bail!("nothing to send");
        }
        if text.len() > sealed::MAX_BODY {
            bail!("a message holds at most {} bytes", sealed::MAX_BODY);
        }
        if let Some(i) = self.ident.book.chats.iter().rposition(|c| c.peer == name && c.verified) {
            let bytes = self.ident.book.chats[i].session.seal(text.as_bytes())?;
            self.node.send_chat(vec![], None, sealed::attachments(&bytes)?)?;
            self.ident.book.chats[i].lines.push(Line { mine: true, text: text.to_string(), at: now_secs() });
            self.sealed_save();
            let _ = self.events.send(WalletEvent::WalletChanged);
            return Ok(());
        }
        self.ident.book.outbox.push(Queued { to: name.clone(), text: text.to_string(), queued: now_secs() });
        self.sealed_save();
        self.flush_outbox(&name).await;
        Ok(())
    }

    /// Open a session with `name` for what waits for it, if its key is known
    /// and still the one its record lists; ask for the key if not.
    async fn flush_outbox(&mut self, name: &str) {
        if let Err(e) = self.try_flush(name).await {
            tracing::debug!("sealed message to {name}: {e:#}");
            let _ = self.events.send(WalletEvent::ChannelNotice { text: format!("Message to {name}.mds is waiting: {e:#}") });
        }
    }

    async fn try_flush(&mut self, name: &str) -> Result<()> {
        if !self.ident.book.outbox.iter().any(|q| q.to == name) {
            return Ok(());
        }
        let shown = format!("{name}{}", names::SUFFIX);
        let r = self.live_record(&shown)?;
        let hash = names::kem_hash(&r.records).ok_or_else(|| anyhow!("{shown} takes no sealed messages (no kem record)"))?;
        let known = self.ident.book.peer_kems.get(name).and_then(|h| hex::decode(h).ok());
        let Some(public) = known.filter(|pk| sealed::kem_hash(pk) == hash) else {
            let identity = names::identity(&r.records).ok_or_else(|| anyhow!("{shown} lists no identity to ask for its key"))?;
            let tip = self.node.get_state().await.height;
            self.ask_kem(name, identity, hash, tip)?;
            bail!("asking {shown}'s wallet for its key");
        };
        let queued: Vec<Queued> = self.ident.book.outbox.iter().filter(|q| q.to == name).cloned().collect();
        let first = &queued[0];
        let as_name = self.speaking_name();
        let identity = self.book.identity_pk;
        let (opening, session) = match (as_name.clone(), identity) {
            (Some(me), Some(id)) => {
                let sign = |t: &[u8; 32]| self.sign_commitment(t);
                sealed::open_session(&public, &hash, Some(sealed::OpenAs { name: &me, identity: id, sign: Box::new(sign) }), first.text.as_bytes())?
            }
            _ => sealed::open_session(&public, &hash, None, first.text.as_bytes())?,
        };
        self.node.send_chat(vec![], None, sealed::attachments(&opening)?)?;
        let mut chat = Conversation { session, peer: name.to_string(), verified: true, lines: Vec::new(), started: now_secs() };
        chat.lines.push(Line { mine: true, text: first.text.clone(), at: now_secs() });
        for q in &queued[1..] {
            let bytes = chat.session.seal(q.text.as_bytes())?;
            self.node.send_chat(vec![], None, sealed::attachments(&bytes)?)?;
            chat.lines.push(Line { mine: true, text: q.text.clone(), at: now_secs() });
        }
        self.ident.book.outbox.retain(|q| q.to != name);
        self.ident.book.chats.push(chat);
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Each tick: read every sealed chat on the bus once, keep what opens for
    /// this wallet, and retry or give up on what waits for a key.
    pub(super) async fn tick_sealed(&mut self, tip: u64) {
        if self.wallet.is_none() || self.kem_keys().is_err() {
            return;
        }
        let hist = self.node.chat_history.read().await.clone();
        let mut changed = false;
        for m in hist.iter() {
            let Some(env) = sealed::envelope(&m.attachments) else { continue };
            if !self.ident.book.mark_seen(m.timestamp, m.nonce, &m.sender) {
                continue;
            }
            changed = true;
            // A message of a session this wallet is in: its route says which.
            let routed = self.ident.book.chats.iter().enumerate().find_map(|(i, c)| {
                c.session.expects().find(|(r, _)| env.len() >= r.len() && env[..r.len()] == r[..]).map(|(_, n)| (i, n))
            });
            if let Some((i, n)) = routed {
                if let Some(body) = self.ident.book.chats[i].session.open(n, &env) {
                    let text = String::from_utf8_lossy(&body).into_owned();
                    self.ident.book.chats[i].lines.push(Line { mine: false, text, at: m.timestamp });
                    self.notify_message(i);
                }
                continue;
            }
            let Some(opened) = sealed::try_open(self.kem_keys().expect("checked above"), &env) else { continue };
            if self.ident.book.chats.iter().any(|c| c.session.id == opened.session.id) {
                continue;
            }
            let (peer, verified) = match &opened.sender {
                Sender::Anonymous => (String::new(), false),
                Sender::Named { name, .. } => {
                    let identity = self.live_record(&format!("{name}{}", names::SUFFIX)).ok().and_then(|r| names::identity(&r.records));
                    (name.clone(), opened.sender.verify(identity))
                }
            };
            let text = String::from_utf8_lossy(&opened.body).into_owned();
            self.ident.book.chats.push(Conversation {
                session: opened.session,
                peer,
                verified,
                lines: vec![Line { mine: false, text, at: m.timestamp }],
                started: now_secs(),
            });
            self.notify_message(self.ident.book.chats.len() - 1);
        }
        // What waits for a key: ask again now and then, give up in the end.
        let now = now_secs();
        let stale: Vec<String> = self.ident.book.outbox.iter().filter(|q| now.saturating_sub(q.queued) > OUTBOX_PATIENCE).map(|q| q.to.clone()).collect();
        if !stale.is_empty() {
            self.ident.book.outbox.retain(|q| now.saturating_sub(q.queued) <= OUTBOX_PATIENCE);
            for n in stale {
                let _ = self.events.send(WalletEvent::ChannelNotice { text: format!("Gave up sending to {n}.mds: its wallet never handed over its key.") });
            }
            changed = true;
        }
        let waiting: Vec<String> = {
            let mut v: Vec<String> = self.ident.book.outbox.iter().map(|q| q.to.clone()).collect();
            v.dedup();
            v
        };
        for n in waiting {
            if self.ident.book.kem_reqs.values().all(|(r, _, at)| r != &n || tip >= at + KEM_RETRY) {
                self.flush_outbox(&n).await;
            }
        }
        if changed {
            self.sealed_save();
        }
    }

    fn notify_message(&self, i: usize) {
        let c = &self.ident.book.chats[i];
        let who = if c.peer.is_empty() { "someone anonymous".to_string() } else { format!("{}.mds{}", c.peer, if c.verified { "" } else { " (unverified)" }) };
        let _ = self.events.send(WalletEvent::ChannelNotice { text: format!("New sealed message from {who}.") });
        let _ = self.events.send(WalletEvent::WalletChanged);
    }

    pub(super) fn conversations(&self) -> Vec<ConversationView> {
        let mut v: Vec<ConversationView> = self
            .ident
            .book
            .chats
            .iter()
            .map(|c| {
                let backing = if c.verified && !c.peer.is_empty() { self.name_backing(&c.peer).map_or(0.0, |b| b.blocks) } else { 0.0 };
                ConversationView {
                    id: hex::encode(c.session.id),
                    peer: if c.peer.is_empty() { String::new() } else { format!("{}{}", c.peer, names::SUFFIX) },
                    verified: c.verified,
                    opened_by_me: c.session.opener,
                    lines: c.lines.iter().map(|l| MessageLineView { mine: l.mine, text: l.text.clone(), at: l.at }).collect(),
                    backing,
                    request: self.is_request(&c.peer, c.verified, c.session.opener, backing),
                }
            })
            .collect();
        v.extend(self.ident.book.outbox.iter().map(|q| ConversationView {
            id: String::new(),
            peer: format!("{}{}", q.to, names::SUFFIX),
            verified: false,
            opened_by_me: true,
            lines: vec![MessageLineView { mine: true, text: format!("{} (waiting for its wallet's key)", q.text), at: q.queued }],
            backing: self.name_backing(&q.to).map_or(0.0, |b| b.blocks),
            request: false,
        }));
        v.sort_by_key(|c| std::cmp::Reverse(c.lines.last().map_or(0, |l| l.at)));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::script::compile_multisig_2of3;

    #[test]
    fn the_inbox_takes_verified_names_unless_told_otherwise() {
        assert_eq!(SealedBook::default().inbox_bar, Some(0.0));
        let before: SealedBook = serde_json::from_str(r#"{"chats":[],"seen":[]}"#).unwrap();
        assert_eq!(before.inbox_bar, Some(0.0), "a book saved before there was a bar");
        for bar in [None, Some(0.0), Some(60.0), Some(1_440.0)] {
            let b = SealedBook { inbox_bar: bar, ..SealedBook::default() };
            let back: SealedBook = serde_json::from_slice(&serde_json::to_vec(&b).unwrap()).unwrap();
            assert_eq!(back.inbox_bar, bar, "a choice, \"everyone\" too, survives a save");
        }
    }

    #[test]
    fn a_payment_asks_the_identity_unless_the_name_points_at_a_shared_address() {
        let me = [7u8; 32];
        let key = Rec::MdsKey { key: [1; 32], height: 10 };
        // A wallet's own name with an identity: ask it.
        assert_eq!(handshake_with(&[key.clone(), Rec::Identity([5; 32])], Some(me), |_| 0), Some([5; 32]));
        // No identity, or this wallet's own: pay the record.
        assert_eq!(handshake_with(&[key.clone()], Some(me), |_| 0), None);
        assert_eq!(handshake_with(&[key, Rec::Identity(me)], Some(me), |_| 0), None);
        // A multisig whose script the name publishes: pay it as it is.
        let ms = compile_multisig_2of3(&[1; 32], &[2; 32], &[3; 32]);
        let mut recs = vec![Rec::MdsAddress(midstate::core::hash(&ms)), Rec::Identity([5; 32])];
        recs.extend(names::script_records(&ms));
        assert_eq!(handshake_with(&recs, Some(me), |_| 0), None);
        // An address whose kind is not published: the identity is asked.
        assert_eq!(handshake_with(&recs[..2], Some(me), |_| 0), Some([5; 32]));
    }

    #[test]
    fn the_sealed_book_keeps_what_it_read_once() {
        let mut b = SealedBook::default();
        assert!(b.mark_seen(1, 2, "p"));
        assert!(!b.mark_seen(1, 2, "p"), "read once");
        assert!(b.mark_seen(1, 3, "p"));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.dat");
        b.outbox.push(Queued { to: "nick".into(), text: "hi".into(), queued: 5 });
        b.save(&path);
        let back = SealedBook::load(&path);
        assert_eq!((back.seen.len(), back.outbox.len()), (2, 1));
        assert_eq!(back.outbox[0].to, "nick");
    }

    #[test]
    fn a_target_is_described_only_by_its_own_script() {
        let ms = compile_multisig_2of3(&[1; 32], &[2; 32], &[3; 32]);
        let a = midstate::core::encode_address_with_checksum(&midstate::core::hash(&ms));
        assert_eq!(crate::describe_target(&a, &hex::encode(&ms)).unwrap(), "2-of-3 multisig");
        assert_eq!(crate::describe_target(&a, "").unwrap(), "script address (kind not published)");
        assert!(crate::describe_target(&a, &hex::encode(&ms[1..])).is_err(), "another script");
        assert!(crate::describe_target("nope", "").is_err());
    }
}
