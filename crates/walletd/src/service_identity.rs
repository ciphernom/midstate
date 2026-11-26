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
//! Strangers pay to be heard, and the recipient sets the price. Until this
//! wallet answers or accepts a conversation, a message in it must carry a
//! stamp of the bits its names ask (their `postage` records, the lowest of
//! them; [`sealed::STAMP_BITS`] where they say none) or it is dropped; and
//! this wallet stamps what it sends, at the price the name it writes to
//! asks, until the other side answers. A wallet under attack raises its
//! price, and every message a stranger sends it costs that much more.
//! Stamps are worked off the actor ([`crate::stamper`]), so a dear one
//! holds nothing up: a message being stamped is kept, and shown, as such,
//! and goes once its stamp is done; the work is taken up again after a
//! restart.
//!
//! A message request keeps at most [`REQUEST_LINES`] messages and raises
//! no notice; requests are kept to [`REQUESTS_MAX`], and go after
//! [`REQUEST_AGE`] unanswered. Blocked names, and anonymous senders if the
//! user says so, are dropped unread. Conversations can have a timer: both
//! wallets delete a message once it has run.
//!
//! A child of `service`, sharing the actor's state, as `service_names` is.

use super::{now_secs, Cmd, Internal, Service};
use crate::api::{ConversationView, MessageLineView, NameAction, PostageView, SendStage, WalletEvent};
use crate::stamper::Stamper;
use anyhow::{anyhow, bail, Context, Result};
use midstate::chat::ChatAttachment;
use midstate::core::channel::wire as w;
use midstate::core::names::{self, Rec};
use midstate::sealed::{self, KemKeys, Sender, Session};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
/// Messages a request keeps from its sender until it is accepted.
pub const REQUEST_LINES: usize = 3;
/// Requests kept at most: past this, the one quiet longest goes.
pub const REQUESTS_MAX: usize = 100;
/// A request quiet this long goes when room is needed (seconds).
pub const REQUEST_AGE: u64 = 30 * 86_400;
// The timers a conversation can have.
use crate::SEALED_TIMERS as TIMERS;

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
    /// Lines its user has seen: `lines[..read]`. A book saved before there
    /// was a mark loads as all read.
    #[serde(default = "all_read")]
    pub read: usize,
    /// Its user took it out of Requests, by replying or accepting it: the
    /// other side's messages need no stamp from now on.
    #[serde(default)]
    pub accepted: bool,
    /// The other side has written: this wallet's messages need no stamp.
    #[serde(default)]
    pub heard: bool,
    /// The timer messages here carry, in seconds; 0 for none.
    #[serde(default)]
    pub ttl: u32,
}

fn all_read() -> usize {
    usize::MAX
}

impl Conversation {
    /// The other side's messages its user has not seen.
    pub fn unread(&self) -> usize {
        self.lines.iter().skip(self.read).filter(|l| !l.mine && l.timer.is_none()).count()
    }

    /// Lines whose timer has run are deleted; the read mark keeps its place.
    /// Whether any went.
    fn prune(&mut self, now: u64) -> bool {
        let before = self.lines.len();
        let mut read = self.read;
        let mut i = 0;
        self.lines.retain(|l| {
            let keep = !l.expired(now);
            if !keep && i < self.read {
                read -= 1;
            }
            i += 1;
            keep
        });
        self.read = read;
        self.lines.len() != before
    }

    /// When it last moved, for ordering conversations.
    fn last(&self) -> u64 {
        self.lines.last().map_or(self.started, |l| l.at)
    }

    fn id_hex(&self) -> String {
        hex::encode(self.session.id)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Line {
    pub mine: bool,
    pub text: String,
    pub at: u64,
    /// Its timer, in seconds: deleted `ttl` after `at`. 0 for never.
    #[serde(default)]
    pub ttl: u32,
    /// Not a message: a side set the conversation's timer to this.
    #[serde(default)]
    pub timer: Option<u32>,
    /// This wallet's, still being stamped: the number the session keeps for
    /// it. It goes, and this clears, once the stamp is done.
    #[serde(default)]
    pub stamping: Option<u64>,
}

impl Line {
    fn said(mine: bool, text: &str, at: u64, ttl: u32) -> Self {
        Line { mine, text: text.to_string(), at, ttl, timer: None, stamping: None }
    }

    fn timer_set(mine: bool, ttl: u32, at: u64) -> Self {
        Line { mine, text: String::new(), at, ttl: 0, timer: Some(ttl), stamping: None }
    }

    /// A message waiting on its stamp has not been sent: its timer has not
    /// started.
    fn expired(&self, now: u64) -> bool {
        self.stamping.is_none() && self.ttl > 0 && self.at.saturating_add(u64::from(self.ttl)) <= now
    }

    /// The timer it goes under: a timer's note carries the timer it sets.
    fn sealed_ttl(&self) -> u32 {
        self.timer.unwrap_or(self.ttl)
    }
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
    /// Names whose messages are dropped unread (canonical, no suffix).
    #[serde(default)]
    pub blocked: Vec<String>,
    /// Drop anonymous senders' messages unread, rather than show them as
    /// requests.
    #[serde(default)]
    pub drop_anonymous: bool,
    /// Openings being stamped, by the name they go to: made up to the stamp
    /// (encapsulated and signed), so the work goes on after a restart
    /// without signing again. Each holds its session's secret, as `chats` do.
    #[serde(default)]
    pub openings: HashMap<String, sealed::Prepared>,
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
            blocked: Vec::new(),
            drop_anonymous: false,
            openings: HashMap::new(),
        }
    }
}

impl SealedBook {
    fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("messages.json")
    }
    pub fn load(wallet_path: &Path) -> Self {
        let mut book: Self = std::fs::read(Self::path(wallet_path)).ok().and_then(|raw| serde_json::from_slice(&raw).ok()).unwrap_or_default();
        for c in &mut book.chats {
            c.read = c.read.min(c.lines.len());
            // The other side wrote first, or has written.
            c.heard |= !c.session.opener || c.lines.iter().any(|l| !l.mine);
        }
        book
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

    /// The conversation whose session id is `id` (hex).
    fn find(&self, id: &str) -> Option<usize> {
        self.chats.iter().position(|c| c.id_hex() == id)
    }

    /// Conversation `id` has been seen up to its last line. False if there
    /// is no such conversation or nothing was unseen.
    fn mark_read(&mut self, id: &str) -> bool {
        let Some(i) = self.find(id) else { return false };
        let c = &mut self.chats[i];
        let changed = c.read != c.lines.len();
        c.read = c.lines.len();
        changed
    }

    /// Takes conversation `id` out of Requests.
    fn accept(&mut self, id: &str) -> Result<()> {
        let i = self.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        self.chats[i].accepted = true;
        Ok(())
    }

    /// Forgets conversation `id`, its session too: what the other side sends
    /// in it later is never read. Messages from a new session still come.
    fn delete(&mut self, id: &str) -> Result<()> {
        let i = self.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        self.chats.remove(i);
        Ok(())
    }

    /// Whether messages from `peer` (a name, without its suffix) are dropped.
    fn is_blocked(&self, peer: &str) -> bool {
        !peer.is_empty() && self.blocked.iter().any(|b| b == peer)
    }

    /// Blocks `name`: its conversations go, and what waits for it, and
    /// nothing it sends is read until it is unblocked.
    fn block(&mut self, name: &str) {
        if !self.is_blocked(name) {
            self.blocked.push(name.to_string());
        }
        self.chats.retain(|c| c.peer != name);
        self.outbox.retain(|q| q.to != name);
        self.openings.remove(name);
    }

    fn unblock(&mut self, name: &str) -> bool {
        let before = self.blocked.len();
        self.blocked.retain(|b| b != name);
        self.blocked.len() != before
    }

    /// Makes room for one more request: requests (their session ids in
    /// `requests`) quiet for [`REQUEST_AGE`] go, then the quietest while
    /// `max` are held.
    fn make_room(&mut self, now: u64, requests: &HashSet<[u8; 32]>, max: usize) {
        let is_request = |c: &Conversation| requests.contains(&c.session.id);
        self.chats.retain(|c| !(is_request(c) && c.last().saturating_add(REQUEST_AGE) < now));
        loop {
            let held: Vec<usize> = (0..self.chats.len()).filter(|&i| is_request(&self.chats[i])).collect();
            if held.len() < max {
                return;
            }
            let quietest = *held.iter().min_by_key(|&&i| self.chats[i].last()).expect("not empty");
            self.chats.remove(quietest);
        }
    }

    /// Drops the messages waiting to go to `to` (for its key, or on the
    /// stamp of the opening); how many.
    fn cancel(&mut self, to: &str) -> usize {
        let before = self.outbox.len();
        self.outbox.retain(|q| q.to != to);
        self.openings.remove(to);
        before - self.outbox.len()
    }

    /// Drops this wallet's messages in conversation `id` still waiting on
    /// their stamps; how many. The numbers they held are given back when
    /// nothing after them was sent, so the other side's window of numbers
    /// it watches is not used up by messages that never came.
    fn unsend(&mut self, id: &str) -> Result<usize> {
        let i = self.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        let c = &mut self.chats[i];
        let dropped: HashSet<u64> = c.lines.iter().filter_map(|l| l.stamping).collect();
        let before = c.lines.len();
        let mut read = c.read;
        let mut j = 0;
        c.lines.retain(|l| {
            let keep = l.stamping.is_none();
            if !keep && j < c.read {
                read -= 1;
            }
            j += 1;
            keep
        });
        c.read = read;
        while c.session.sent > 1 && dropped.contains(&(c.session.sent - 1)) {
            c.session.sent -= 1;
        }
        Ok(before - c.lines.len())
    }

    /// The messages waiting for a key, by name, in the order first queued.
    fn waiting(&self) -> Vec<(String, Vec<&Queued>)> {
        let mut out: Vec<(String, Vec<&Queued>)> = Vec::new();
        for q in &self.outbox {
            match out.iter_mut().find(|(n, _)| *n == q.to) {
                Some((_, v)) => v.push(q),
                None => out.push((q.to.clone(), vec![q])),
            }
        }
        out
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
    /// The stamp worker, started when first needed.
    pub stamper: Option<Stamper>,
    /// What each stamp being worked is for, by its job.
    pub jobs: HashMap<u64, Stamping>,
    pub next_job: u64,
}

/// What a stamp is being worked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Stamping {
    /// The opening of a conversation with this name (no suffix).
    Opening(String),
    /// This wallet's message `n` in the conversation with this session id.
    Line([u8; 32], u64),
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
        // Work for another wallet's messages, if any, is not this one's.
        if let Some(s) = &self.ident.stamper {
            s.clear();
        }
        self.ident.jobs.clear();
    }

    /// On lock: the KEM secret leaves memory, and payments still waiting on
    /// a name's wallet are refused rather than left hanging.
    pub(super) fn identity_unload(&mut self) {
        self.ident.kem = None;
        for s in std::mem::take(&mut self.ident.sends) {
            let _ = s.resp.send(Err(anyhow!("wallet is locked")));
        }
        // Stamps stop with the wallet; what waits on them is kept, and the
        // work starts again on unlock.
        if let Some(s) = &self.ident.stamper {
            s.clear();
        }
        self.ident.jobs.clear();
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

    /// Unexpired fresh addresses cached from `identity`.
    fn peek_fresh_addresses(&self, identity: &[u8; 32], tip: u64) -> Vec<[u8; 32]> {
        let key = hex::encode(identity);
        let Some((raw, expiry)) = self.book.peer_addrs.get(&key) else {
            return Vec::new();
        };
        if *expiry <= tip + ADDRESS_MARGIN {
            return Vec::new();
        }
        raw.split(',')
            .filter_map(|s| midstate::core::parse_address_flexible(s.trim()).ok())
            .collect()
    }

    /// Drop the first `used` addresses from `identity`'s cache after a send succeeds.
    fn consume_fresh_addresses(&mut self, identity: &[u8; 32], used: usize) {
        let key = hex::encode(identity);
        let Some((raw, expiry)) = self.book.peer_addrs.get(&key).cloned() else {
            return;
        };
        let rest: Vec<&str> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .skip(used)
            .collect();
        if rest.is_empty() {
            self.book.peer_addrs.remove(&key);
        } else {
            self.book.peer_addrs.insert(key, (rest.join(","), expiry));
        }
        self.book.save(&self.wallet_path);
    }

    fn has_pending_addr_req(&self, identity: &[u8; 32]) -> bool {
        self.book.addr_reqs.values().any(|p| p == identity)
    }

    /// A send asked for by the UI. To a name whose wallet can be asked, use
    /// cached fresh addresses (or ask for a fresh batch) and answer once it
    /// replies (or the wait runs out and the record is paid); otherwise send now.
    pub(super) async fn send_or_ask(&mut self, to: String, amount: u64, private: bool, resp: oneshot::Sender<Result<String>>) {
        if let Some(identity) = self.identity_to_ask(&to) {
            let tip = self.node.get_state().await.height;
            let needed = midstate::core::decompose_value(amount).len().max(1);
            let cached = self.peek_fresh_addresses(&identity, tip);
            if cached.len() >= needed {
                let dests = cached[..needed].to_vec();
                let r = self.send_fresh(&to, &identity, &dests, amount, private).await;
                let _ = resp.send(r);
                return;
            }
            if self.wallet.is_some() && !self.node.is_syncing() {
                match self.request_address(&hex::encode(identity)).await {
                    Ok(()) => {
                        self.ident.sends.push(NameSend { to, amount, private, identity, deadline: Instant::now() + HANDSHAKE_WAIT, resp });
                        return;
                    }
                    Err(e) => tracing::debug!("could not ask {to} for fresh addresses: {e:#}"),
                }
            }
            if !cached.is_empty() {
                let r = self.send_fresh(&to, &identity, &cached, amount, private).await;
                let _ = resp.send(r);
                return;
            }
        }
        let r = self.start_send(&to, amount, private).await;
        let _ = resp.send(r);
    }

    async fn send_fresh(
        &mut self,
        to: &str,
        identity: &[u8; 32],
        dests: &[[u8; 32]],
        amount: u64,
        private: bool,
    ) -> Result<String> {
        let id = self.start_send_to_many(to, dests, amount, private).await?;
        self.consume_fresh_addresses(identity, dests.len());
        if let Some(c) = hex32(&id) {
            let msg = if dests.len() > 1 {
                format!("paying {} fresh one-time addresses from {}'s wallet", dests.len(), to.trim())
            } else {
                format!("paying a fresh one-time address from {}'s wallet", to.trim())
            };
            self.set_stage(c, SendStage::Committing, msg);
        }
        Ok(id)
    }

    /// Each tick: finish the payments whose fresh address batch came, or whose
    /// wait ran out.
    pub(super) async fn tick_name_sends(&mut self, tip: u64) {
        let waiting = std::mem::take(&mut self.ident.sends);
        for s in waiting {
            let needed = midstate::core::decompose_value(s.amount).len().max(1);
            let cached = self.peek_fresh_addresses(&s.identity, tip);
            let reply_arrived = !self.has_pending_addr_req(&s.identity) && !cached.is_empty();

            if reply_arrived || cached.len() >= needed {
                let take = needed.min(cached.len());
                let dests = cached[..take].to_vec();
                let r = self.send_fresh(&s.to, &s.identity, &dests, s.amount, s.private).await;
                let _ = s.resp.send(r);
            } else if Instant::now() >= s.deadline || s.resp.is_closed() {
                if !cached.is_empty() {
                    let r = self.send_fresh(&s.to, &s.identity, &cached, s.amount, s.private).await;
                    let _ = s.resp.send(r);
                } else {
                    let r = self.start_send(&s.to, s.amount, s.private).await;
                    if let Ok(c) = r.as_deref().map(hex32) {
                        if let Some(c) = c {
                            let note = format!("{}'s wallet did not answer, so this pays the address its name lists", s.to.trim());
                            self.set_stage(c, SendStage::Committing, note);
                        }
                    }
                    let _ = s.resp.send(r);
                }
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
        if self.ident.book.is_blocked(&name) {
            bail!("you blocked {name}{}: unblock it to write to it", names::SUFFIX);
        }
        let text = checked_text(text)?;
        if let Some(i) = self.ident.book.chats.iter().rposition(|c| c.peer == name && c.verified) {
            return self.seal_in(i, text);
        }
        self.ident.book.outbox.push(Queued { to: name.clone(), text: text.to_string(), queued: now_secs() });
        self.sealed_save();
        self.flush_outbox(&name).await;
        Ok(())
    }

    /// Send `text` in conversation `i`'s own session, under its timer, and
    /// stamped until the other side has written. Writing in a conversation
    /// is reading it, and takes it out of Requests.
    fn seal_in(&mut self, i: usize, text: &str) -> Result<()> {
        let line = Line::said(true, text, now_secs(), self.ident.book.chats[i].ttl);
        self.write_line(i, line)?;
        let c = &mut self.ident.book.chats[i];
        c.read = c.lines.len();
        c.accepted = true;
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Puts this wallet's `line` in conversation `i` and sends it: now, if
    /// it needs no stamp, or once its stamp is done.
    fn write_line(&mut self, i: usize, mut line: Line) -> Result<()> {
        let bits = self.stamp_for(&self.ident.book.chats[i]);
        let (n, key) = self.ident.book.chats[i].session.reserve();
        if bits == 0 {
            self.send_line(i, n, &line, 0)?;
        } else {
            line.stamping = Some(n);
            self.submit_stamp(Stamping::Line(self.ident.book.chats[i].session.id, n), key, bits);
        }
        self.ident.book.chats[i].lines.push(line);
        Ok(())
    }

    /// Seals `line` as this wallet's message `n` in conversation `i`, with
    /// the stamp nonce `nonce`, and puts it on the bus.
    fn send_line(&self, i: usize, n: u64, line: &Line, nonce: u64) -> Result<()> {
        let bytes = self.ident.book.chats[i].session.seal_reserved(n, line.text.as_bytes(), line.sealed_ttl(), nonce)?;
        self.node.send_chat(vec![], None, sealed::attachments(&bytes)?)?;
        Ok(())
    }

    /// Reply in conversation `id` (its session id, hex), to whoever is on
    /// the other side of it: a verified name, a name the opener claimed but
    /// that did not check out, or an anonymous opener. Only they can read it.
    pub(super) fn reply_sealed(&mut self, id: &str, text: &str) -> Result<()> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let text = checked_text(text)?;
        let i = self.ident.book.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        self.seal_in(i, text)
    }

    /// Set conversation `id`'s timer (one of [`TIMERS`]): what is sent from
    /// now on, by either side, is deleted from both wallets `ttl` seconds
    /// after it was sent. The other side's wallet learns it from a message
    /// that says only that.
    pub(super) fn set_timer(&mut self, id: &str, ttl: u32) -> Result<()> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if !TIMERS.contains(&ttl) {
            bail!("a timer is none, an hour, a day, a week or four weeks");
        }
        let i = self.ident.book.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        if self.ident.book.chats[i].ttl == ttl {
            return Ok(());
        }
        self.write_line(i, Line::timer_set(true, ttl, now_secs()))?;
        let c = &mut self.ident.book.chats[i];
        c.ttl = ttl;
        c.read = c.lines.len();
        // Setting the timer is writing in it: it leaves Requests, and the
        // other side's messages need no stamp.
        c.accepted = true;
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Its user has seen conversation `id`.
    pub(super) fn read_sealed(&mut self, id: &str) {
        if self.ident.book.mark_read(id) {
            self.sealed_save();
        }
    }

    /// Take conversation `id` out of Requests.
    pub(super) fn accept_sealed(&mut self, id: &str) -> Result<()> {
        self.ident.book.accept(id)?;
        self.sealed_save();
        Ok(())
    }

    /// Forget conversation `id`, and stop stamping what was to go in it.
    pub(super) fn delete_sealed(&mut self, id: &str) -> Result<()> {
        self.ident.book.delete(id)?;
        self.drop_stale_stamps();
        self.sealed_save();
        Ok(())
    }

    /// Stop sending what in conversation `id` still waits on its stamp.
    pub(super) fn unsend_sealed(&mut self, id: &str) -> Result<()> {
        if self.ident.book.unsend(id)? == 0 {
            bail!("nothing in that conversation is waiting to go");
        }
        self.drop_stale_stamps();
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Block whoever is on the other side of conversation `id`. A verified
    /// name is blocked by name: its conversations go, and nothing it sends is
    /// read until it is unblocked. Anyone else (anonymous, or a name whose
    /// claim did not check out) cannot be told from any other stranger, so
    /// only the conversation goes, and its session with it. Returns the name
    /// blocked, if one was.
    pub(super) fn block_sealed(&mut self, id: &str) -> Result<Option<String>> {
        let i = self.ident.book.find(id).ok_or_else(|| anyhow!("no such conversation"))?;
        let c = &self.ident.book.chats[i];
        let name = (c.verified && !c.peer.is_empty()).then(|| c.peer.clone());
        match &name {
            Some(n) => self.ident.book.block(n),
            None => {
                self.ident.book.chats.remove(i);
            }
        }
        self.drop_stale_stamps();
        self.sealed_save();
        Ok(name.map(|n| format!("{n}{}", names::SUFFIX)))
    }

    pub(super) fn unblock_sealed(&mut self, name: &str) -> Result<()> {
        let name = names::normalize(name)?;
        if !self.ident.book.unblock(&name) {
            bail!("{name}{} is not blocked", names::SUFFIX);
        }
        self.sealed_save();
        Ok(())
    }

    /// Blocked names, as shown (`nick.mds`).
    pub(super) fn blocked_names(&self) -> Vec<String> {
        self.ident.book.blocked.iter().map(|n| format!("{n}{}", names::SUFFIX)).collect()
    }

    pub(super) fn drop_anonymous(&self) -> bool {
        self.ident.book.drop_anonymous
    }

    pub(super) fn set_drop_anonymous(&mut self, on: bool) {
        self.ident.book.drop_anonymous = on;
        self.sealed_save();
    }

    /// Stop trying to send what waits to go to `to`: for its key, or on
    /// the stamp of the conversation's opening.
    pub(super) fn cancel_sealed(&mut self, to: &str) -> Result<()> {
        let name = names::normalize(to)?;
        if self.ident.book.cancel(&name) == 0 {
            bail!("nothing is waiting to go to {name}{}", names::SUFFIX);
        }
        self.drop_stale_stamps();
        self.sealed_save();
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
        if !self.ident.book.outbox.iter().any(|q| q.to == name) || self.ident.book.openings.contains_key(name) {
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
        // Made up to the stamp now, signed once; sealed when the stamp is done.
        let prepared = match (self.speaking_name(), self.book.identity_pk) {
            (Some(me), Some(id)) => {
                let sign = |t: &[u8; 32]| self.sign_commitment(t);
                sealed::prepare_opening(&public, &hash, Some(sealed::OpenAs { name: &me, identity: id, sign: Box::new(sign) }))?
            }
            _ => sealed::prepare_opening(&public, &hash, None)?,
        };
        let key = prepared.stamp_key();
        self.ident.book.openings.insert(name.to_string(), prepared);
        self.sealed_save();
        // A first message is a stranger's: stamped at what the name asks.
        self.submit_stamp(Stamping::Opening(name.to_string()), key, postage_in(&r.records));
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    // ── Stamps ──────────────────────────────────────────────────────────

    fn stamper(&mut self) -> &Stamper {
        if self.ident.stamper.is_none() {
            let tx = self.self_tx.clone();
            self.ident.stamper = Some(Stamper::new(Box::new(move |job, bits, nonce| {
                let _ = tx.blocking_send(Cmd::Internal(Internal::Stamped { job, bits, nonce }));
            })));
        }
        self.ident.stamper.as_ref().expect("just set")
    }

    fn submit_stamp(&mut self, what: Stamping, key: [u8; 32], bits: u32) {
        let job = self.ident.next_job;
        self.ident.next_job += 1;
        self.ident.jobs.insert(job, what);
        self.stamper().submit(job, key, bits);
    }

    fn job_for(&self, what: &Stamping) -> Option<u64> {
        self.ident.jobs.iter().find(|(_, w)| *w == what).map(|(job, _)| *job)
    }

    /// Whether what job `what` stamps still waits to go.
    fn stamp_wanted(&self, what: &Stamping) -> bool {
        let book = &self.ident.book;
        match what {
            Stamping::Opening(name) => book.openings.contains_key(name) && book.outbox.iter().any(|q| q.to == *name),
            Stamping::Line(id, n) => book.chats.iter().any(|c| c.session.id == *id && c.lines.iter().any(|l| l.stamping == Some(*n))),
        }
    }

    /// Stops the stamps whose messages went: sent, cancelled, deleted, or
    /// blocked.
    fn drop_stale_stamps(&mut self) {
        let stale: Vec<u64> = self.ident.jobs.iter().filter(|(_, w)| !self.stamp_wanted(w)).map(|(job, _)| *job).collect();
        for job in stale {
            self.ident.jobs.remove(&job);
            if let Some(s) = &self.ident.stamper {
                s.cancel(job);
            }
        }
    }

    /// A stamp was done (`Internal::Stamped`): send what it was for, unless
    /// that went meanwhile, or now asks more.
    pub(super) fn on_stamped(&mut self, job: u64, bits: u32, nonce: u64) {
        let Some(what) = self.ident.jobs.remove(&job) else { return };
        if self.wallet.is_none() {
            return;
        }
        let r = match &what {
            Stamping::Line(id, n) => self.stamped_line(*id, *n, bits, nonce),
            Stamping::Opening(name) => self.stamped_opening(name, bits, nonce),
        };
        if let Err(e) = r {
            // The message is stamped again, and tried again.
            tracing::warn!("a stamped sealed message did not go: {e:#}");
            if self.reconcile_stamps() {
                self.sealed_save();
            }
        }
    }

    fn stamped_line(&mut self, id: [u8; 32], n: u64, bits: u32, nonce: u64) -> Result<()> {
        let Some(i) = self.ident.book.chats.iter().position(|c| c.session.id == id) else { return Ok(()) };
        let Some(j) = self.ident.book.chats[i].lines.iter().position(|l| l.stamping == Some(n)) else { return Ok(()) };
        let need = self.stamp_for(&self.ident.book.chats[i]);
        if need > bits {
            // The price went up while it was worked: the work goes on.
            let key = self.ident.book.chats[i].session.stamp_key(n);
            self.submit_stamp(Stamping::Line(id, n), key, need);
            return Ok(());
        }
        let line = self.ident.book.chats[i].lines[j].clone();
        self.send_line(i, n, &line, nonce)?;
        let l = &mut self.ident.book.chats[i].lines[j];
        l.stamping = None;
        // Its timer runs from when it went, as the other side's does.
        l.at = now_secs();
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// The opening to `name` was stamped: it goes, with the first message
    /// that waited for it; the others follow in the session, each with a
    /// stamp of its own until `name` answers.
    fn stamped_opening(&mut self, name: &str, bits: u32, nonce: u64) -> Result<()> {
        let Some(prepared) = self.ident.book.openings.get(name).cloned() else { return Ok(()) };
        let need = self.postage_of(name);
        if need > bits {
            self.submit_stamp(Stamping::Opening(name.to_string()), prepared.stamp_key(), need);
            return Ok(());
        }
        let queued: Vec<Queued> = self.ident.book.outbox.iter().filter(|q| q.to == name).cloned().collect();
        let Some(first) = queued.first() else {
            self.ident.book.openings.remove(name);
            self.sealed_save();
            return Ok(());
        };
        let (opening, session) = prepared.finish(first.text.as_bytes(), 0, nonce)?;
        self.node.send_chat(vec![], None, sealed::attachments(&opening)?)?;
        let now = now_secs();
        let mut chat = Conversation { session, peer: name.to_string(), verified: true, lines: Vec::new(), started: now, read: 0, accepted: true, heard: false, ttl: 0 };
        chat.lines.push(Line::said(true, &first.text, now, 0));
        self.ident.book.openings.remove(name);
        self.ident.book.outbox.retain(|q| q.to != name);
        self.ident.book.chats.push(chat);
        let i = self.ident.book.chats.len() - 1;
        for q in &queued[1..] {
            if let Err(e) = self.write_line(i, Line::said(true, &q.text, now, 0)) {
                tracing::warn!("a sealed message to {name}.mds did not go: {e:#}");
            }
        }
        let c = &mut self.ident.book.chats[i];
        c.read = c.lines.len();
        self.sealed_save();
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// The other side of conversation `i` has written: what waits on its
    /// stamp there goes now, unstamped.
    fn send_stamped_now(&mut self, i: usize) {
        let id = self.ident.book.chats[i].session.id;
        for j in 0..self.ident.book.chats[i].lines.len() {
            let Some(n) = self.ident.book.chats[i].lines[j].stamping else { continue };
            let line = self.ident.book.chats[i].lines[j].clone();
            if let Err(e) = self.send_line(i, n, &line, 0) {
                tracing::warn!("a sealed message did not go: {e:#}");
                continue;
            }
            let l = &mut self.ident.book.chats[i].lines[j];
            l.stamping = None;
            l.at = now_secs();
            if let Some(job) = self.job_for(&Stamping::Line(id, n)) {
                self.ident.jobs.remove(&job);
                if let Some(s) = &self.ident.stamper {
                    s.cancel(job);
                }
            }
        }
    }

    /// Each tick: what waits on a stamp is being worked, at the price asked
    /// now (a raise or a cut lands on the work in hand); work for what went
    /// stops; and after a restart, the work starts again. Whether the book
    /// changed.
    fn reconcile_stamps(&mut self) -> bool {
        let mut changed = false;
        self.drop_stale_stamps();
        for i in 0..self.ident.book.chats.len() {
            let c = &self.ident.book.chats[i];
            let waiting: Vec<u64> = c.lines.iter().filter_map(|l| l.stamping).collect();
            if waiting.is_empty() {
                continue;
            }
            let need = self.stamp_for(c);
            if need == 0 {
                self.send_stamped_now(i);
                changed = true;
                continue;
            }
            let id = c.session.id;
            for n in waiting {
                match self.job_for(&Stamping::Line(id, n)) {
                    Some(job) => self.stamper().reprice(job, need),
                    None => {
                        let key = self.ident.book.chats[i].session.stamp_key(n);
                        self.submit_stamp(Stamping::Line(id, n), key, need);
                    }
                }
            }
        }
        let openings: Vec<(String, [u8; 32], [u8; 32])> =
            self.ident.book.openings.iter().map(|(n, p)| (n.clone(), p.stamp_key(), p.recipient_hash())).collect();
        for (name, key, sealed_to) in openings {
            let recs = self.held_records(&name);
            if recs.as_deref().and_then(names::kem_hash) != Some(sealed_to) {
                // The name's key changed, or it is no longer held: open
                // again, to the key it lists now (asked for afresh).
                self.ident.book.openings.remove(&name);
                self.drop_stale_stamps();
                changed = true;
                continue;
            }
            let need = recs.as_deref().map_or(sealed::STAMP_BITS, postage_in);
            match self.job_for(&Stamping::Opening(name.clone())) {
                Some(job) => self.stamper().reprice(job, need),
                None => self.submit_stamp(Stamping::Opening(name), key, need),
            }
        }
        changed
    }

    // ── Prices ──────────────────────────────────────────────────────────

    /// The records `name` (no suffix) carries, if someone holds it.
    fn held_records(&self, name: &str) -> Option<Vec<Rec>> {
        let r = midstate::storage::name_index::lookup(self.node.storage.db(), name).ok().flatten()?;
        (!r.lapsed()).then_some(r.records)
    }

    /// What `name` asks of a stranger's sealed message, in stamp bits.
    fn postage_of(&self, name: &str) -> u32 {
        self.held_records(name).as_deref().map_or(sealed::STAMP_BITS, postage_in)
    }

    /// The stamp this wallet's next message in `c` needs: what the other
    /// side's name asks, until they have written; nothing after.
    fn stamp_for(&self, c: &Conversation) -> u32 {
        match (c.heard, c.peer.is_empty()) {
            (true, _) => 0,
            (false, true) => sealed::STAMP_BITS,
            (false, false) => self.postage_of(&c.peer),
        }
    }

    /// The names this wallet holds or rents, with the records they carry
    /// now and (for a change on its way) the records they will.
    fn own_names(&self) -> Vec<(String, bool, Option<Vec<Rec>>, Option<Vec<Rec>>)> {
        self.names
            .jobs
            .iter()
            .filter(|j| !j.done())
            .map(|j| {
                let now = self.held_records(&j.name);
                let wanted = j.want.as_ref().filter(|_| j.status == "updating").and_then(|w| names::parse(&w.records));
                (j.name.clone(), j.status == "registered" && !j.renter, now, wanted)
            })
            .collect()
    }

    /// What this wallet asks of a stranger's sealed message, now and once
    /// the changes on their way land, and the names whose price it can set.
    pub(super) fn postage_view(&mut self) -> PostageView {
        let Ok(mine) = self.kem_record() else { return PostageView { bits: sealed::STAMP_BITS, ..PostageView::default() } };
        let own = self.own_names();
        let bits = required_postage(own.iter().filter_map(|(_, _, now, _)| now.as_deref()), &mine);
        let later = required_postage(own.iter().filter_map(|(_, _, now, wanted)| wanted.as_deref().or(now.as_deref())), &mine);
        let takes = |r: &Option<Vec<Rec>>| r.as_deref().and_then(names::kem_hash) == Some(mine);
        PostageView {
            bits,
            pending: (later != bits).then_some(later),
            names: own.iter().filter(|(_, _, now, wanted)| takes(now) || takes(wanted)).map(|(n, ..)| format!("{n}{}", names::SUFFIX)).collect(),
            settable: own.iter().filter(|(_, ok, now, _)| *ok && takes(now)).map(|(n, ..)| format!("{n}{}", names::SUFFIX)).collect(),
        }
    }

    /// What this wallet asks of a stranger's sealed message, in stamp bits.
    pub(super) fn postage_required(&mut self) -> u32 {
        let Ok(mine) = self.kem_record() else { return sealed::STAMP_BITS };
        let own = self.own_names();
        required_postage(own.iter().filter_map(|(_, _, now, _)| now.as_deref()), &mine)
    }

    /// Ask `bits` of strangers' sealed messages from now on: every name of
    /// this wallet's that takes them publishes the price (a record change
    /// each, with its network fee). The names changed.
    pub(super) async fn set_postage(&mut self, bits: u32) -> Result<Vec<String>> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let (lo, hi) = (u32::from(*names::POSTAGE_BITS.start()), u32::from(*names::POSTAGE_BITS.end()));
        if !(lo..=hi).contains(&bits) {
            bail!("a price is {lo} to {hi} bits");
        }
        let mine = self.kem_record()?;
        let mut changed = Vec::new();
        let mut busy = Vec::new();
        for (name, settable, now, _) in self.own_names() {
            let Some(recs) = now.filter(|r| names::kem_hash(r) == Some(mine)) else { continue };
            if postage_in(&recs) == bits {
                continue;
            }
            let shown = format!("{name}{}", names::SUFFIX);
            if !settable {
                busy.push(shown);
                continue;
            }
            self.name_action(&name, NameAction::Postage { bits }).await.with_context(|| format!("changing {shown}"))?;
            changed.push(shown);
        }
        if changed.is_empty() && !busy.is_empty() {
            bail!("{} is changing already, or rented: try again once that lands", busy.join(", "));
        }
        if changed.is_empty() && busy.is_empty() && self.postage_view().names.is_empty() {
            bail!("none of this wallet's names takes sealed messages");
        }
        Ok(changed)
    }

    /// Whether `c` is a message request now: not accepted, and under the
    /// inbox's bar.
    fn is_request_now(&self, c: &Conversation) -> bool {
        if c.accepted {
            return false;
        }
        let backing = if c.verified && !c.peer.is_empty() { self.name_backing(&c.peer).map_or(0.0, |b| b.blocks) } else { 0.0 };
        self.is_request(&c.peer, c.verified, c.session.opener, backing)
    }

    /// A message the other side sent in conversation `i`: kept if it paid
    /// what this wallet asks of it, `price` ([`admits`]). Once the other
    /// side has written, what waits here on a stamp goes without one.
    fn receive(&mut self, i: usize, got: sealed::Received, at: u64, price: u32) {
        let request = self.is_request_now(&self.ident.book.chats[i]);
        let c = &mut self.ident.book.chats[i];
        if !admits(c, &got, request, price) {
            return;
        }
        if !std::mem::replace(&mut c.heard, true) {
            self.send_stamped_now(i);
        }
        let c = &mut self.ident.book.chats[i];
        if got.ttl != c.ttl {
            c.ttl = got.ttl;
            c.lines.push(Line::timer_set(false, got.ttl, at));
        }
        if !got.body.is_empty() {
            c.lines.push(Line::said(false, &String::from_utf8_lossy(&got.body), at, got.ttl));
            // Requests wait quietly: a stranger cannot make this wallet speak up.
            if !request {
                self.notify_message(i);
            }
        }
    }

    /// An opening to this wallet's key: a new conversation, unless it did
    /// not pay `price`, or comes from a blocked name, or from someone
    /// anonymous when those are dropped. A request makes room for itself.
    fn welcome(&mut self, opened: sealed::Opened, at: u64, price: u32) {
        if opened.stamp_bits < price || opened.body.is_empty() {
            return;
        }
        let (peer, verified) = match &opened.sender {
            Sender::Anonymous => (String::new(), false),
            Sender::Named { name, .. } => {
                let identity = self.live_record(&format!("{name}{}", names::SUFFIX)).ok().and_then(|r| names::identity(&r.records));
                (name.clone(), opened.sender.verify(identity))
            }
        };
        if self.ident.book.is_blocked(&peer) || (peer.is_empty() && self.ident.book.drop_anonymous) {
            return;
        }
        let mut c = Conversation { session: opened.session, peer, verified, lines: Vec::new(), started: now_secs(), read: 0, accepted: false, heard: true, ttl: opened.ttl };
        if opened.ttl > 0 {
            c.lines.push(Line::timer_set(false, opened.ttl, at));
        }
        c.lines.push(Line::said(false, &String::from_utf8_lossy(&opened.body), at, opened.ttl));
        let request = self.is_request_now(&c);
        if request {
            let requests: HashSet<[u8; 32]> = self.ident.book.chats.iter().filter(|c| self.is_request_now(c)).map(|c| c.session.id).collect();
            self.ident.book.make_room(now_secs(), &requests, REQUESTS_MAX);
        }
        self.ident.book.chats.push(c);
        if !request {
            self.notify_message(self.ident.book.chats.len() - 1);
        }
    }

    /// Each tick: read every sealed chat on the bus once, keep what opens for
    /// this wallet and pays its way, let timers run, and retry or give up on
    /// what waits for a key.
    pub(super) async fn tick_sealed(&mut self, tip: u64) {
        if self.wallet.is_none() || self.kem_keys().is_err() {
            return;
        }
        // What strangers pay, as the chain shows this wallet's names now.
        let price = self.postage_required();
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
                if let Some(got) = self.ident.book.chats[i].session.open(n, &env) {
                    self.receive(i, got, m.timestamp, price);
                }
                continue;
            }
            let Some(opened) = sealed::try_open(self.kem_keys().expect("checked above"), &env) else { continue };
            if self.ident.book.chats.iter().any(|c| c.session.id == opened.session.id) {
                continue;
            }
            self.welcome(opened, m.timestamp, price);
        }
        // Timers that have run.
        let now = now_secs();
        for c in &mut self.ident.book.chats {
            changed |= c.prune(now);
        }
        // What waits on a stamp: worked, at today's price.
        changed |= self.reconcile_stamps();
        // What waits for a key: ask again now and then, give up in the end.
        // An opening being stamped has its key, and waits as long as it takes.
        let book = &self.ident.book;
        let gives_up = |q: &Queued| now.saturating_sub(q.queued) > OUTBOX_PATIENCE && !book.openings.contains_key(&q.to);
        let mut stale: Vec<String> = book.outbox.iter().filter(|q| gives_up(q)).map(|q| q.to.clone()).collect();
        stale.dedup();
        if !stale.is_empty() {
            let keep: Vec<Queued> = book.outbox.iter().filter(|q| !gives_up(q)).cloned().collect();
            self.ident.book.outbox = keep;
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
        let now = now_secs();
        let mut v: Vec<(u64, ConversationView)> = self
            .ident
            .book
            .chats
            .iter()
            .map(|c| {
                let backing = if c.verified && !c.peer.is_empty() { self.name_backing(&c.peer).map_or(0.0, |b| b.blocks) } else { 0.0 };
                let lines = c
                    .lines
                    .iter()
                    .filter(|l| !l.expired(now))
                    .map(|l| MessageLineView {
                        mine: l.mine,
                        text: l.text.clone(),
                        at: l.at,
                        pending: false,
                        expires: if l.ttl > 0 && l.stamping.is_none() { l.at.saturating_add(u64::from(l.ttl)) } else { 0 },
                        timer: l.timer,
                        stamping: l.stamping.is_some(),
                    })
                    .collect();
                let view = ConversationView {
                    id: c.id_hex(),
                    peer: if c.peer.is_empty() { String::new() } else { format!("{}{}", c.peer, names::SUFFIX) },
                    verified: c.verified,
                    opened_by_me: c.session.opener,
                    lines,
                    backing,
                    request: !c.accepted && self.is_request(&c.peer, c.verified, c.session.opener, backing),
                    unread: c.unread(),
                    ttl: c.ttl,
                    stamping: !c.heard,
                    stamp_bits: self.stamp_for(c),
                };
                (c.last(), view)
            })
            .collect();
        // What waits for a name's key: one conversation a name, not sent yet.
        // Or, its key in hand, for the opening's stamp.
        v.extend(self.ident.book.waiting().into_iter().map(|(to, queued)| {
            let stamped = self.ident.book.openings.contains_key(&to);
            let line = |q: &&Queued| MessageLineView { mine: true, text: q.text.clone(), at: q.queued, pending: !stamped, expires: 0, timer: None, stamping: stamped };
            let view = ConversationView {
                id: String::new(),
                peer: format!("{to}{}", names::SUFFIX),
                verified: false,
                opened_by_me: true,
                lines: queued.iter().map(line).collect(),
                backing: self.name_backing(&to).map_or(0.0, |b| b.blocks),
                request: false,
                unread: 0,
                ttl: 0,
                stamping: true,
                stamp_bits: self.postage_of(&to),
            };
            (queued.last().map_or(0, |q| q.queued), view)
        }));
        v.sort_by_key(|(last, _)| std::cmp::Reverse(*last));
        v.into_iter().map(|(_, c)| c).collect()
    }
}

/// A message's text, trimmed, if there is one and it fits.
fn checked_text(text: &str) -> Result<&str> {
    let text = text.trim();
    if text.is_empty() {
        bail!("nothing to send");
    }
    if text.len() > sealed::MAX_BODY {
        bail!("a message holds at most {} bytes", sealed::MAX_BODY);
    }
    Ok(text)
}

/// Whether a message the other side sent in `c` is kept. Until this wallet
/// has answered or accepted, it must carry a stamp of `price` bits; while
/// `c` is a request, it keeps [`REQUEST_LINES`] of the sender's messages at
/// most.
fn admits(c: &Conversation, got: &sealed::Received, request: bool, price: u32) -> bool {
    if !c.accepted && got.stamp_bits < price {
        return false;
    }
    !(request && c.lines.iter().filter(|l| !l.mine).count() >= REQUEST_LINES)
}

/// What a name with `recs` asks of a stranger's sealed message, in bits.
fn postage_in(recs: &[Rec]) -> u32 {
    names::postage(recs).map_or(sealed::STAMP_BITS, u32::from)
}

/// What a wallet whose KEM key hash is `mine`, holding names with these
/// records, asks of a stranger: the lowest price among the names that take
/// its sealed messages, so whoever paid what the name they wrote to asks is
/// heard; the standard if none does.
fn required_postage<'a>(held: impl Iterator<Item = &'a [Rec]>, mine: &[u8; 32]) -> u32 {
    held.filter(|r| names::kem_hash(r) == Some(*mine)).map(postage_in).min().unwrap_or(sealed::STAMP_BITS)
}

/// `recs` asking `bits` of strangers: the `postage` record replaced, or
/// gone at the standard price. Everything else is kept as it is.
pub(super) fn with_postage(recs: &[Rec], bits: u32) -> Result<Vec<Rec>> {
    let mut out: Vec<Rec> =
        recs.iter().filter(|r| !matches!(r, Rec::Postage(_) | Rec::Other { tag: names::tag::POSTAGE, .. })).cloned().collect();
    if bits != sealed::STAMP_BITS {
        let b = u8::try_from(bits).ok().filter(|b| names::POSTAGE_BITS.contains(b)).ok_or_else(|| anyhow!("a price is {} to {} bits", names::POSTAGE_BITS.start(), names::POSTAGE_BITS.end()))?;
        out.push(Rec::Postage(b));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use midstate::core::script::compile_multisig_2of3;

    /// A session opened to a fresh KEM key, from the opener's side.
    fn a_session() -> Session {
        let keys = KemKeys::from_secret(&rand::random());
        sealed::open_session(&keys.public(), &keys.public_hash(), None, b"hello", 0, 0).unwrap().1
    }

    fn conversation(peer: &str, lines: &[(bool, &str)]) -> Conversation {
        let lines = lines.iter().enumerate().map(|(i, (mine, t))| Line::said(*mine, t, i as u64, 0)).collect();
        Conversation { session: a_session(), peer: peer.into(), verified: true, lines, started: 0, read: 0, accepted: false, heard: true, ttl: 0 }
    }

    #[test]
    fn a_conversation_keeps_what_was_read_accepted_or_deleted() {
        let mut book = SealedBook::default();
        book.chats.push(conversation("alice", &[(false, "hi"), (false, "there"), (true, "hey")]));
        book.chats.push(conversation("bob", &[(false, "yo")]));
        let (a, b) = (book.chats[0].id_hex(), book.chats[1].id_hex());
        assert_eq!((book.chats[0].unread(), book.chats[1].unread()), (2, 1));
        assert!(book.mark_read(&a));
        assert!(!book.mark_read(&a), "nothing new to see");
        assert!(!book.mark_read("00"), "no such conversation");
        book.chats[0].lines.push(Line::said(false, "again", 9, 0));
        assert_eq!(book.chats[0].unread(), 1, "only what came after");
        book.accept(&b).unwrap();
        assert!(book.chats[1].accepted && !book.chats[0].accepted);
        book.delete(&a).unwrap();
        assert_eq!(book.chats.iter().map(|c| c.peer.as_str()).collect::<Vec<_>>(), vec!["bob"]);
        assert!(book.delete(&a).is_err() && book.accept(&a).is_err());
    }

    fn got(body: &str, stamp_bits: u32) -> sealed::Received {
        sealed::Received { body: body.as_bytes().to_vec(), ttl: 0, stamp_bits }
    }

    #[test]
    fn a_stranger_pays_to_be_heard_and_a_request_holds_three() {
        let paid = sealed::STAMP_BITS;
        let mut c = conversation("carol", &[]);
        // Not answered yet: a stamp, or it goes unread.
        assert!(!admits(&c, &got("hi", 0), false, paid));
        assert!(!admits(&c, &got("hi", paid - 1), false, paid));
        assert!(admits(&c, &got("hi", paid), false, paid));
        // A wallet under attack asks more: yesterday's price is not enough.
        assert!(!admits(&c, &got("hi", paid), false, 32));
        assert!(admits(&c, &got("hi", 33), false, 32));
        // A request keeps three of the sender's messages, however they pay.
        for i in 0..REQUEST_LINES {
            assert!(admits(&c, &got("more", paid), true, paid));
            c.lines.push(Line::said(false, "more", i as u64, 0));
        }
        assert!(!admits(&c, &got("and more", paid), true, paid));
        assert!(admits(&c, &got("and more", paid), false, paid), "in the inbox, no cap");
        // Answered or accepted: no stamp asked, no cap, whatever the price.
        c.accepted = true;
        assert!(admits(&c, &got("thanks", 0), false, 40));
    }

    fn held(postage: Option<u8>, kem: [u8; 32]) -> Vec<Rec> {
        let mut r = vec![Rec::MdsKey { key: [1; 32], height: 10 }, Rec::Kem(kem), Rec::Text("hi".into())];
        r.extend(postage.map(Rec::Postage));
        r
    }

    #[test]
    fn the_price_is_the_lowest_a_name_that_takes_the_wallets_messages_asks() {
        let mine = [7; 32];
        let price = |names: &[Vec<Rec>]| required_postage(names.iter().map(|r| r.as_slice()), &mine);
        assert_eq!(price(&[]), sealed::STAMP_BITS, "no names: the standard");
        assert_eq!(price(&[held(Some(36), mine)]), 36);
        // Someone who wrote to the cheaper name paid what it asked.
        assert_eq!(price(&[held(Some(36), mine), held(Some(28), mine)]), 28);
        assert_eq!(price(&[held(Some(36), mine), held(None, mine)]), sealed::STAMP_BITS);
        // A name that does not take this wallet's messages does not count.
        assert_eq!(price(&[held(Some(36), mine), held(Some(16), [8; 32]), vec![Rec::Text("x".into())]]), 36);
        assert_eq!(postage_in(&held(None, mine)), sealed::STAMP_BITS);
    }

    #[test]
    fn a_price_changes_only_the_postage_record() {
        let mine = [7; 32];
        let base = held(None, mine);
        let raised = with_postage(&base, 32).unwrap();
        assert_eq!(raised[..3], base[..], "the rest as it was");
        assert_eq!(names::postage(&raised), Some(32));
        let again = with_postage(&raised, 36).unwrap();
        assert_eq!((again.len(), names::postage(&again)), (4, Some(36)), "replaced, not added");
        assert_eq!(with_postage(&again, sealed::STAMP_BITS).unwrap(), base, "the standard publishes none");
        // One that did not read (out of range, kept as an unknown tag) goes too.
        let mut odd = base.clone();
        odd.push(Rec::Other { tag: names::tag::POSTAGE, value: vec![99] });
        assert_eq!(with_postage(&odd, 28).unwrap()[3], Rec::Postage(28));
        for bad in [15, 41, 0, 300] {
            assert!(with_postage(&base, bad).is_err(), "{bad}");
        }
        assert!(names::encode(&with_postage(&base, 40).unwrap()).is_ok());
    }

    #[test]
    fn a_message_waiting_on_its_stamp_keeps_its_place_and_can_be_taken_back() {
        let mut book = SealedBook::default();
        let mut c = conversation("erin", &[(true, "hello"), (false, "hi")]);
        c.heard = false;
        c.read = 2;
        // Two more, waiting on their stamps under the numbers they hold; a
        // timer that ran out long ago does not touch them until they go.
        for text in ["one", "two"] {
            let (n, _) = c.session.reserve();
            let mut l = Line::said(true, text, 0, 3_600);
            l.stamping = Some(n);
            c.lines.push(l);
        }
        c.read = 4;
        assert_eq!(c.session.sent, 3);
        assert!(!c.prune(1_000_000), "nothing sent has a timer to run");
        assert_eq!(c.lines.len(), 4);
        assert_eq!(c.lines.iter().filter(|l| l.stamping.is_some()).count(), 2);
        assert_eq!(c.unread(), 0, "this wallet's own are never unread");
        // Kept across a restart, numbers and all.
        book.chats.push(c);
        let kept: SealedBook = serde_json::from_slice(&serde_json::to_vec(&book).unwrap()).unwrap();
        assert_eq!(kept.chats[0].lines[3].stamping, Some(2));
        // Taken back: they go, and so do the numbers they held.
        let id = book.chats[0].id_hex();
        assert_eq!(book.unsend(&id).unwrap(), 2);
        let c = &book.chats[0];
        assert_eq!((c.lines.len(), c.read, c.session.sent), (2, 2, 1));
        assert_eq!(book.unsend(&id).unwrap(), 0);
        assert!(book.unsend("00").is_err());
    }

    #[test]
    fn a_number_after_a_sent_message_is_not_given_back() {
        let mut book = SealedBook::default();
        let mut c = conversation("frank", &[(true, "hello")]);
        let (n1, _) = c.session.reserve();
        let mut waiting = Line::said(true, "slow", 0, 0);
        waiting.stamping = Some(n1);
        c.lines.push(waiting);
        // A later message went (the other side wrote meanwhile).
        let (_n2, _) = c.session.reserve();
        c.lines.push(Line::said(true, "quick", 1, 0));
        book.chats.push(c);
        let id = book.chats[0].id_hex();
        assert_eq!(book.unsend(&id).unwrap(), 1);
        assert_eq!(book.chats[0].session.sent, 3, "number 2 went: number 1 stays a gap");
    }

    #[test]
    fn an_opening_being_stamped_is_kept_until_it_goes_or_is_called_off() {
        let keys = KemKeys::from_secret(&[9; 32]);
        let prepared = sealed::prepare_opening(&keys.public(), &keys.public_hash(), None).unwrap();
        let mut book = SealedBook::default();
        book.outbox.push(Queued { to: "grace".into(), text: "hi".into(), queued: 0 });
        book.openings.insert("grace".into(), prepared.clone());
        let kept: SealedBook = serde_json::from_slice(&serde_json::to_vec(&book).unwrap()).unwrap();
        assert_eq!(kept.openings["grace"].stamp_key(), prepared.stamp_key(), "the same work, after a restart");
        assert_eq!(book.cancel("grace"), 1);
        assert!(book.openings.is_empty());
        book.outbox.push(Queued { to: "grace".into(), text: "hi".into(), queued: 0 });
        book.openings.insert("grace".into(), prepared);
        book.block("grace");
        assert!(book.openings.is_empty() && book.outbox.is_empty());
    }

    #[test]
    fn blocking_a_name_drops_its_conversations_and_what_waits_for_it() {
        let mut book = SealedBook::default();
        book.chats.push(conversation("mallory", &[(false, "spam")]));
        book.chats.push(conversation("mallory", &[(false, "more spam")]));
        book.chats.push(conversation("alice", &[(false, "hi")]));
        book.outbox.push(Queued { to: "mallory".into(), text: "stop".into(), queued: 0 });
        book.block("mallory");
        book.block("mallory");
        assert_eq!(book.blocked, vec!["mallory".to_string()], "once");
        assert!(book.is_blocked("mallory") && !book.is_blocked("alice") && !book.is_blocked(""));
        assert_eq!(book.chats.iter().map(|c| c.peer.as_str()).collect::<Vec<_>>(), vec!["alice"]);
        assert!(book.outbox.is_empty());
        assert!(book.unblock("mallory") && !book.unblock("mallory"));
        assert!(!book.is_blocked("mallory"));
    }

    #[test]
    fn requests_make_room_for_new_ones() {
        let mut book = SealedBook::default();
        for (i, peer) in ["a", "b", "c", "friend"].iter().enumerate() {
            let mut c = conversation(peer, &[(false, "hi")]);
            c.lines[0].at = 1_000 + i as u64;
            book.chats.push(c);
        }
        let requests: HashSet<[u8; 32]> = book.chats.iter().filter(|c| c.peer != "friend").map(|c| c.session.id).collect();
        // Three requests held, room for three: the quietest goes.
        book.make_room(2_000, &requests, 3);
        assert_eq!(book.chats.iter().map(|c| c.peer.as_str()).collect::<Vec<_>>(), vec!["b", "c", "friend"]);
        // Requests quiet for a month go whatever the room; the inbox stays.
        book.make_room(1_003 + REQUEST_AGE + 1, &requests, 100);
        assert_eq!(book.chats.iter().map(|c| c.peer.as_str()).collect::<Vec<_>>(), vec!["friend"]);
    }

    #[test]
    fn timers_delete_messages_and_keep_the_read_mark_in_place() {
        let mut c = conversation("alice", &[]);
        c.lines = vec![
            Line::said(false, "kept", 0, 0),
            Line::said(false, "an hour", 100, 3_600),
            Line::timer_set(true, 3_600, 150),
            Line::said(true, "a day", 200, 86_400),
            Line::said(false, "unread, an hour", 300, 3_600),
        ];
        c.read = 4;
        assert!(!c.prune(3_000), "nothing has run yet");
        assert!(c.prune(100 + 3_600));
        assert_eq!(c.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>(), vec!["kept", "", "a day", "unread, an hour"]);
        assert_eq!((c.read, c.unread()), (3, 1), "the mark stays before the unread one");
        assert!(c.prune(300 + 3_600));
        assert_eq!(c.lines.len(), 3);
        assert_eq!((c.read, c.unread()), (3, 0));
        assert_eq!(c.lines[1].timer, Some(3_600), "a timer's note does not expire");
    }

    #[test]
    fn what_waits_for_a_key_is_one_conversation_a_name() {
        let mut book = SealedBook::default();
        for (to, text) in [("bob", "1"), ("carol", "x"), ("bob", "2")] {
            book.outbox.push(Queued { to: to.into(), text: text.into(), queued: 0 });
        }
        let w = book.waiting();
        let shape: Vec<(&str, Vec<&str>)> = w.iter().map(|(n, q)| (n.as_str(), q.iter().map(|q| q.text.as_str()).collect())).collect();
        assert_eq!(shape, vec![("bob", vec!["1", "2"]), ("carol", vec!["x"])]);
        assert_eq!(book.cancel("bob"), 2);
        assert_eq!(book.cancel("bob"), 0);
        assert_eq!(book.outbox.len(), 1);
    }

    #[test]
    fn a_book_saved_before_read_marks_loads_as_read() {
        let dir = tempfile::tempdir().unwrap();
        let wallet = dir.path().join("w.dat");
        let mut book = SealedBook::default();
        book.chats.push(conversation("alice", &[(false, "hi"), (false, "there")]));
        let mut raw = serde_json::to_value(&book).unwrap();
        let chat = raw["chats"][0].as_object_mut().unwrap();
        chat.remove("read");
        chat.remove("accepted");
        chat.remove("heard");
        chat.remove("ttl");
        std::fs::write(SealedBook::path(&wallet), serde_json::to_vec(&raw).unwrap()).unwrap();
        let back = SealedBook::load(&wallet);
        assert_eq!((back.chats[0].read, back.chats[0].unread(), back.chats[0].accepted), (2, 0, false));
        assert!(back.chats[0].heard, "the other side has written");
        // And a mark survives a save.
        let mut book = back;
        book.chats[0].lines.push(Line::said(false, "new", 5, 0));
        book.save(&wallet);
        assert_eq!(SealedBook::load(&wallet).chats[0].unread(), 1);
    }

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
