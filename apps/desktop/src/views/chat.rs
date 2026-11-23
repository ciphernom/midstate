//! Messages, laid out as a messenger: conversations down the left, the open
//! one beside them, the box to write in under it. The network's public room
//! is pinned at the top of the list.
//!
//! Sealed conversations (`midstate::sealed`) are read by two wallets alone:
//! sealed post-quantum (ML-KEM), addressed so nobody watching the network
//! sees who they are for. A stranger's messages carry a stamp of work, at
//! the price the recipient asks (Strangers pay, at the foot of the list),
//! until the recipient writes back. The public room is the chat bus itself:
//! up to ten words from a fixed dictionary, each message mined by your own
//! node.
//!
//! [`draw`] takes what it shows as [`Data`] and keeps its own state in a
//! [`Messenger`]; what it wants done comes back as [`Out`], so it can be laid
//! out and tested without a wallet behind it.

use crate::app::{App, Tab};
use crate::bridge::Action;
use crate::theme;
use eframe::egui::{
    self, text::LayoutJob, Align, Align2, Color32, FontId, Frame, Galley, Key, KeyboardShortcut, Layout, Margin, Modifiers, Rect, RichText, Sense,
    Stroke, StrokeKind, TextEdit, Ui,
};
use midstate_walletd::api::{ChatView, ConversationView, MessageLineView, NameCard, PostageView};
use midstate_walletd::{POSTAGE_STANDARD, SEALED_MAX, SEALED_TIMERS};
use std::collections::HashMap;
use std::sync::Arc;

/// The list's width beside an open conversation.
const LIST_W: f32 = 304.0;
/// Narrower than this, the list and a conversation take turns.
const NARROW: f32 = 760.0;
const ROW_H: f32 = 62.0;
/// Messages from one side closer together than this read as one run.
const RUN_SECS: u64 = 5 * 60;
/// Space between a conversation's edge and its messages.
const SIDE: f32 = 18.0;
/// Drafts' keys for the public room and a new message.
const PUBLIC: &str = "public";
const NEW: &str = "new";
/// What a wallet can ask of strangers, in bits of stamp work: each step is
/// sixteen times the work of the one before.
const PRICES: [(u32, &str); 5] = [(24, "Standard"), (28, "Raised"), (32, "High"), (36, "Very high"), (40, "Maximum")];

// ── State ───────────────────────────────────────────────────────────────────

/// What the conversation pane shows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Open {
    #[default]
    Nothing,
    Public,
    /// A message to a name typed in [`Messenger::to`].
    New,
    /// A sealed conversation, by [`key`].
    Thread(String),
}

/// A sealed message on its way, and where its answer belongs.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Sending {
    Thread(String),
    /// A new message to this name.
    To(String),
}

/// The messenger's own state, kept by the app between frames.
#[derive(Default)]
pub struct Messenger {
    pub open: Open,
    /// The list shows message requests rather than the inbox.
    pub requests: bool,
    pub filter: String,
    /// What is being written, by conversation key.
    drafts: HashMap<String, String>,
    /// A new message's recipient, as typed.
    pub to: String,
    /// The last recipient asked about, and what the wallet said of it.
    recipient: Option<(String, Result<NameCard, String>)>,
    asked: Option<String>,
    /// A name to write to, from elsewhere in the app: opened next frame.
    compose_to: Option<String>,
    sending: Option<Sending>,
    /// After a new message: open its conversation once it shows.
    follow: Option<String>,
    /// The conversation whose delete is being confirmed.
    confirm_delete: Option<String>,
    /// The conversation whose sender's block is being confirmed.
    confirm_block: Option<String>,
    /// A new price for strangers, being confirmed.
    confirm_postage: Option<u32>,
    /// The read mark last asked for, (key, lines): asked once.
    marked: Option<(String, usize)>,
    /// Where the open conversation's "new" line goes: (key, line).
    unread_from: Option<(String, Option<usize>)>,
    /// Put the cursor in the box to write in.
    focus: bool,
    dict_filter: String,
}

impl Messenger {
    /// Write to `name`: its conversation if there is one, a new message if
    /// not. Resolved when the messenger next draws.
    pub fn compose_to(&mut self, name: &str) {
        self.compose_to = Some(name.trim().to_ascii_lowercase());
    }

    /// The wallet took a sealed message: clear what was written, and after
    /// a new message, follow it to its conversation.
    pub fn sent(&mut self) {
        match self.sending.take() {
            Some(Sending::Thread(k)) => {
                self.drafts.remove(&k);
            }
            Some(Sending::To(name)) => {
                self.drafts.remove(NEW);
                self.follow = Some(name);
            }
            None => {}
        }
    }

    /// The wallet refused it: what was written stays.
    pub fn send_failed(&mut self) {
        self.sending = None;
    }

    /// The public room took a message.
    pub fn public_sent(&mut self) {
        self.drafts.remove(PUBLIC);
    }

    /// What the wallet found of a recipient asked about.
    pub fn recipient(&mut self, name: String, card: Result<NameCard, String>) {
        self.recipient = Some((name, card));
    }

    fn open(&mut self, o: Open) {
        if self.open != o {
            self.confirm_delete = None;
            self.confirm_block = None;
            self.focus = true;
        }
        self.open = o;
    }

    fn start_new(&mut self) {
        self.to.clear();
        self.open(Open::New);
    }

    /// Resolve what was asked from outside, and keep the open conversation
    /// pointing at one that exists.
    fn settle(&mut self, d: &Data) {
        if let Some(name) = self.compose_to.take() {
            match d.conversations.iter().find(|c| c.peer == name && c.verified) {
                Some(c) => {
                    self.requests = c.request;
                    self.open(Open::Thread(key(c)));
                }
                None => {
                    self.to = name;
                    self.open(Open::New);
                }
            }
        }
        if let Some(name) = self.follow.clone() {
            if let Some(c) = d.conversations.iter().find(|c| c.peer == name && (c.verified || c.id.is_empty())) {
                self.requests = false;
                self.open(Open::Thread(key(c)));
                self.follow = None;
            }
        }
        if let Open::Thread(k) = &self.open {
            if !d.conversations.iter().any(|c| key(c) == *k) {
                // Messages that waited for a key went: their conversation
                // has a session now.
                let opened = k.strip_prefix("to:").and_then(|peer| d.conversations.iter().find(|c| c.peer == peer && c.opened_by_me && !c.id.is_empty()));
                self.open = opened.map_or(Open::Nothing, |c| Open::Thread(key(c)));
            }
        }
    }
}

/// What the messenger shows.
pub struct Data<'a> {
    /// Newest first, as the wallet gives them.
    pub conversations: &'a [ConversationView],
    pub chat: &'a [ChatView],
    pub dict: &'a [String],
    /// The inbox's bar (`Status::inbox_bar`).
    pub inbox_bar: Option<f64>,
    /// Names whose messages are dropped (`nick.mds`), and whether anonymous
    /// senders' are.
    pub blocked: &'a [String],
    pub drop_anonymous: bool,
    /// What strangers pay to write to this wallet (`Status::postage`).
    pub postage: &'a PostageView,
    pub sealed_busy: bool,
    pub chat_busy: bool,
    pub error: &'a str,
    pub now: u64,
    /// The window has focus: what is on screen is being read.
    pub focused: bool,
}

/// What the messenger asks of the app.
pub enum Out {
    Go(Action),
    /// Conversation `id` was seen: mark it read, here at once.
    Read(String),
    /// Pay this name: the Send page, filled in.
    Pay(String),
    ClearError,
}

// ── Conversations ───────────────────────────────────────────────────────────

/// A conversation's key: its session id, or `to:` and the name its first
/// messages wait to go to.
pub fn key(c: &ConversationView) -> String {
    if c.id.is_empty() {
        format!("to:{}", c.peer)
    } else {
        c.id.clone()
    }
}

/// A price, by the name the picker gives it.
fn price_label(bits: u32) -> String {
    PRICES.iter().find(|(b, _)| *b == bits).map_or_else(|| format!("{bits} bits"), |(_, l)| l.to_string())
}

/// A stretch of time, roughly, as people say it: "about 5 minutes". Each
/// unit takes over at one and a half of it, so there is never "1 minute".
fn about(secs: f64) -> String {
    let say = |n: f64, what: &str| format!("about {} {what}s", n.round().max(2.0) as u64);
    match secs {
        s if s < 1.5 => "about a second".into(),
        s if s < 90.0 => say(s, "second"),
        s if s < 90.0 * 60.0 => say(s / 60.0, "minute"),
        s if s < 36.0 * 3_600.0 => say(s / 3_600.0, "hour"),
        s => say(s / 86_400.0, "day"),
    }
}

/// How long a stamp of `bits` takes this computer, in words.
fn work(bits: u32) -> String {
    about(midstate_walletd::stamp_seconds(bits))
}

/// What the stamp on a message to `who` costs, said under the box: the
/// work, here, at the price they ask.
fn stamp_note(who: &str, bits: u32) -> String {
    if bits <= POSTAGE_STANDARD {
        format!("Until {who} writes back, each message carries a stamp: {} of your computer's work, so nobody can flood an inbox for free.", work(bits))
    } else {
        format!(
            "{who} asks strangers more: until they write back, each message carries a stamp of {} of your computer's work ({}). It goes once the work is done; you can keep writing meanwhile.",
            work(bits),
            price_label(bits).to_lowercase()
        )
    }
}

/// This wallet's messages in `c` still being stamped.
fn stamping(c: &ConversationView) -> usize {
    c.lines.iter().filter(|l| l.stamping).count()
}

fn title(c: &ConversationView) -> &str {
    if c.peer.is_empty() {
        "Anonymous"
    } else {
        &c.peer
    }
}

/// The letter on a conversation's tile.
fn initial(title: &str) -> String {
    match title.chars().find(|c| c.is_alphanumeric()) {
        Some(c) if title != "Anonymous" => c.to_uppercase().collect(),
        _ => "?".into(),
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A conversation's last word, for the list.
fn preview(c: &ConversationView) -> String {
    match c.lines.last() {
        None if c.ttl > 0 => format!("Messages disappear after {}", timer_text(c.ttl)),
        None => String::new(),
        Some(l) if l.timer.is_some() => timer_note(c, l),
        Some(l) if l.pending => format!("Waiting for its key: {}", one_line(&l.text)),
        Some(l) if l.stamping => format!("Stamping: {}", one_line(&l.text)),
        Some(l) if l.mine => format!("You: {}", one_line(&l.text)),
        Some(l) => one_line(&l.text),
    }
}

/// Whether a search matches a conversation: its name, or anything said.
fn matches(c: &ConversationView, q: &str) -> bool {
    q.is_empty() || title(c).to_lowercase().contains(q) || c.lines.iter().any(|l| l.text.to_lowercase().contains(q))
}

/// Why a conversation waits under Requests.
fn request_reason(c: &ConversationView, bar: Option<f64>) -> String {
    if c.peer.is_empty() {
        "the sender did not say who they are".into()
    } else if !c.verified {
        format!("the sender's claim to be {} did not check out", c.peer)
    } else {
        match bar {
            Some(b) if b > 0.0 => format!(
                "{} has {} of backing, under your inbox's {}",
                c.peer,
                midstate_walletd::burns::blocks_text(c.backing),
                midstate_walletd::burns::blocks_text(b)
            ),
            _ => format!("{} is not in your inbox yet", c.peer),
        }
    }
}

/// The line under a conversation's name.
fn subtitle(c: &ConversationView) -> String {
    if c.id.is_empty() && stamping(c) > 0 {
        format!("Stamping your first message to {}: it goes, sealed, once the work is done.", c.peer)
    } else if c.id.is_empty() {
        format!("Waiting for {}'s wallet to hand over its key: your messages go as soon as it does, if it does within 15 minutes.", c.peer)
    } else if c.peer.is_empty() {
        "Whoever opened this did not say who they are. Only they read your replies.".into()
    } else if c.verified {
        format!("Sealed: only your wallet and {}'s can read this.", c.peer)
    } else {
        format!("Claims to be {}. Only whoever opened this reads your replies.", c.peer)
    }
}

/// Whether `b` follows `a` in one run of messages. A timer's note breaks
/// a run.
fn same_run(a: &MessageLineView, b: &MessageLineView) -> bool {
    a.mine == b.mine
        && a.pending == b.pending
        && a.stamping == b.stamping
        && a.timer.is_none()
        && b.timer.is_none()
        && b.at.saturating_sub(a.at) <= RUN_SECS
        && theme::local_day(a.at) == theme::local_day(b.at)
}

/// A timer, for people: "1 hour", "4 weeks", "off".
fn timer_text(ttl: u32) -> String {
    let unit = |n: u32, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
    match ttl {
        0 => "off".into(),
        t if t % (7 * 86_400) == 0 => unit(t / (7 * 86_400), "week"),
        t if t % 86_400 == 0 => unit(t / 86_400, "day"),
        t if t % 3_600 == 0 => unit(t / 3_600, "hour"),
        t => unit(t.div_ceil(60), "minute"),
    }
}

/// The note a timer's change leaves in a conversation.
fn timer_note(c: &ConversationView, l: &MessageLineView) -> String {
    let who = if l.mine {
        "You".to_string()
    } else if c.peer.is_empty() {
        "The sender".to_string()
    } else {
        c.peer.clone()
    };
    match l.timer {
        Some(0) => format!("{who} turned off disappearing messages"),
        Some(t) => format!("{who} set messages to disappear after {}", timer_text(t)),
        None => String::new(),
    }
}

/// Whether blocking `c` blocks a name, rather than one conversation.
fn blocks_name(c: &ConversationView) -> bool {
    c.verified && !c.peer.is_empty()
}

/// The words of `text` not in the public room's dictionary.
fn not_in_dictionary<'t>(text: &'t str, dict: &[String]) -> Vec<&'t str> {
    text.split_whitespace().filter(|w| !dict.iter().any(|d| d.eq_ignore_ascii_case(w))).collect()
}

/// What the public room would take: one to ten dictionary words.
fn public_ok(text: &str, dict: &[String]) -> bool {
    let n = text.split_whitespace().count();
    (1..=10).contains(&n) && (dict.is_empty() || not_in_dictionary(text, dict).is_empty())
}

/// A peer id's tail: peer ids all start alike.
fn peer_tag(sender: &str) -> String {
    let n = sender.chars().count();
    if n <= 8 {
        sender.to_string()
    } else {
        format!("…{}", sender.chars().skip(n - 6).collect::<String>())
    }
}

// ── Drawing ─────────────────────────────────────────────────────────────────

/// The messenger, filling `ui`.
pub fn draw(m: &mut Messenger, d: &Data, ui: &mut Ui) -> Vec<Out> {
    let mut out = Vec::new();
    let was = m.open.clone();
    m.settle(d);
    let rect = ui.available_rect_before_wrap();
    let narrow = rect.width() < NARROW;
    let list_shown = !narrow || m.open == Open::Nothing;
    let pane_shown = !narrow || m.open != Open::Nothing;
    if list_shown {
        egui::SidePanel::left("msgr_list")
            .resizable(false)
            .exact_width(if pane_shown { LIST_W } else { rect.width() })
            .show_separator_line(pane_shown)
            .frame(Frame::NONE.fill(theme::bg()))
            .show_inside(ui, |ui| list(m, d, ui, &mut out));
    }
    if pane_shown {
        egui::CentralPanel::default().frame(Frame::NONE.fill(theme::bg())).show_inside(ui, |ui| match m.open.clone() {
            Open::Nothing => empty(m, ui),
            Open::Public => public(m, d, ui, narrow, &mut out),
            Open::New => new_message(m, d, ui, narrow, &mut out),
            Open::Thread(k) => match d.conversations.iter().find(|c| key(c) == k) {
                Some(c) => thread(m, d, c, ui, narrow, &mut out),
                None => empty(m, ui),
            },
        });
    }
    ui.painter().rect_stroke(rect, 0.0, hair(theme::border()), StrokeKind::Inside);
    // An error belongs to what was open when it came.
    if m.open != was && !d.error.is_empty() {
        out.push(Out::ClearError);
    }
    out
}

/// A text box's padding, and the Send button beside it.
const BOX_MARGIN: Margin = Margin { left: 10, right: 10, top: 8, bottom: 8 };
const SEND_W: f32 = 74.0;

/// The width of the text in a box that leaves room for Send beside it
/// (a `TextEdit`'s width does not count its margin).
fn box_width(ui: &Ui) -> f32 {
    (ui.available_width() - SEND_W - ui.spacing().item_spacing.x - f32::from(BOX_MARGIN.left) - f32::from(BOX_MARGIN.right)).max(80.0)
}

/// Whether Enter was pressed without Shift this frame. The modifiers read
/// are the key's own: a quick Shift+Enter can let go of Shift before the
/// frame ends.
fn plain_enter(events: &[egui::Event]) -> bool {
    events.iter().any(|e| matches!(e, egui::Event::Key { key: Key::Enter, pressed: true, modifiers, .. } if !modifiers.shift))
}

/// Placeholder text: plainly not something written.
fn placeholder(text: &str) -> RichText {
    RichText::new(text).color(if theme::is_dark() { theme::muted() } else { theme::faint() })
}

/// A one-pixel line: the only stroke the design uses.
fn hair(color: Color32) -> Stroke {
    Stroke::new(1.0_f32, color)
}

/// One line of text, cut short with an ellipsis past `max_w`.
fn galley(ui: &Ui, text: &str, font: FontId, color: Color32, max_w: f32) -> Arc<Galley> {
    let mut job = LayoutJob::single_section(text.to_owned(), egui::TextFormat::simple(font, color));
    job.wrap = egui::text::TextWrapping { max_width: max_w.max(1.0), max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
    ui.fonts(|f| f.layout_job(job))
}

fn tile_at(p: &egui::Painter, r: Rect, text: &str, quiet: bool) {
    p.rect_stroke(r, 0.0, hair(if quiet { theme::border() } else { theme::border2() }), StrokeKind::Inside);
    p.text(r.center(), Align2::CENTER_CENTER, text, theme::font_mono_medium((r.height() * 0.42).round()), if quiet { theme::muted() } else { theme::ink() });
}

fn tile(ui: &mut Ui, text: &str, size: f32, quiet: bool) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(size, size), Sense::hover());
    tile_at(ui.painter(), r, text, quiet);
}

fn head_frame() -> Frame {
    Frame::NONE.fill(theme::bg()).inner_margin(Margin { left: 18, right: 14, top: 12, bottom: 12 })
}

fn compose_frame() -> Frame {
    Frame::NONE.fill(theme::bg()).inner_margin(Margin { left: 18, right: 14, top: 10, bottom: 12 })
}

fn back(m: &mut Messenger, ui: &mut Ui, narrow: bool) {
    if narrow && ui.button("Back").clicked() {
        m.open = Open::Nothing;
    }
}

// ── The list ────────────────────────────────────────────────────────────────

fn list(m: &mut Messenger, d: &Data, ui: &mut Ui, out: &mut Vec<Out>) {
    let requests = d.conversations.iter().filter(|c| c.request).count();
    let unread: usize = d.conversations.iter().filter(|c| !c.request).map(|c| c.unread).sum();
    if requests == 0 {
        m.requests = false;
    }
    egui::TopBottomPanel::top("msgr_list_head")
        .show_separator_line(false)
        .frame(Frame::NONE.fill(theme::bg()).inner_margin(Margin { left: 14, right: 14, top: 14, bottom: 10 }))
        .show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Messages").font(theme::font_medium(16.0)).color(theme::ink()));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("New").on_hover_text("Write to a name").clicked() {
                        m.start_new();
                    }
                });
            });
            ui.add_space(2.0);
            ui.add(TextEdit::singleline(&mut m.filter).hint_text(placeholder("Search names and messages")).desired_width(f32::INFINITY).margin(Margin::symmetric(8, 6)));
            if requests > 0 {
                ui.add_space(2.0);
                let labels = [if unread > 0 { format!("Inbox {unread}") } else { "Inbox".into() }, format!("Requests {requests}")];
                let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
                let mut i = usize::from(m.requests);
                if theme::segmented(ui, &labels, &mut i) {
                    m.requests = i == 1;
                }
            }
        });
    egui::TopBottomPanel::bottom("msgr_list_foot")
        .frame(Frame::NONE.fill(theme::bg()).inner_margin(Margin::symmetric(14, 10)))
        .show_inside(ui, |ui| list_foot(m, d, ui, out));
    egui::CentralPanel::default().frame(Frame::NONE.fill(theme::bg())).show_inside(ui, |ui| {
        egui::ScrollArea::vertical().id_salt("msgr_rows").auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            rows(m, d, ui);
        });
    });
}

fn rows(m: &mut Messenger, d: &Data, ui: &mut Ui) {
    let q = m.filter.trim().to_lowercase();
    let mut any = false;
    if !m.requests && (q.is_empty() || "public room".contains(&q)) {
        let last = d.chat.iter().rev().find(|c| !c.text.is_empty());
        let said = match last {
            Some(c) if c.mine => format!("You: {}", c.text),
            Some(c) => c.text.clone(),
            None => "Everyone on the network".into(),
        };
        let time = last.map_or(String::new(), |c| theme::list_time(c.timestamp, d.now));
        let r = Row { selected: m.open == Open::Public, tile: "#", title: "Public room", tag: None, time: &time, preview: &said, unread: 0, quiet: false };
        if row(ui, &r).clicked() {
            m.open(Open::Public);
        }
        any = true;
    }
    let requests = m.requests;
    for c in d.conversations.iter().filter(|c| c.request == requests && matches(c, &q)) {
        any = true;
        let k = key(c);
        let tag = if stamping(c) > 0 {
            Some("stamping")
        } else if c.id.is_empty() {
            Some("waiting")
        } else if !c.peer.is_empty() && !c.verified {
            Some("unverified")
        } else {
            None
        };
        let time = c.lines.last().map_or(String::new(), |l| theme::list_time(l.at, d.now));
        let (said, letter) = (preview(c), initial(title(c)));
        let r = Row { selected: m.open == Open::Thread(k.clone()), tile: &letter, title: title(c), tag, time: &time, preview: &said, unread: c.unread, quiet: c.peer.is_empty() };
        if row(ui, &r).clicked() {
            m.open(Open::Thread(k));
        }
    }
    if !any {
        ui.add_space(18.0);
        ui.horizontal(|ui| {
            ui.add_space(14.0);
            theme::hint(
                ui,
                if !q.is_empty() {
                    "Nothing matches."
                } else if m.requests {
                    "No message requests."
                } else {
                    "No conversations yet."
                },
            );
        });
    }
}

struct Row<'a> {
    selected: bool,
    tile: &'a str,
    title: &'a str,
    tag: Option<&'a str>,
    time: &'a str,
    preview: &'a str,
    unread: usize,
    /// Anonymous: drawn quieter.
    quiet: bool,
}

fn row(ui: &mut Ui, r: &Row) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), Sense::click());
    if !ui.is_rect_visible(rect) {
        return resp;
    }
    let p = ui.painter().clone();
    if r.selected {
        p.rect_filled(rect, 0.0, theme::highlight());
        p.rect_filled(Rect::from_min_size(rect.min, egui::vec2(2.0, rect.height())), 0.0, theme::ink());
    } else if resp.hovered() {
        p.rect_filled(rect, 0.0, theme::panel());
    }
    let t = Rect::from_min_size(rect.min + egui::vec2(14.0, (ROW_H - 36.0) / 2.0), egui::vec2(36.0, 36.0));
    tile_at(&p, t, r.tile, r.quiet);
    let (x, right) = (t.right() + 12.0, rect.right() - 14.0);
    let strong = r.unread > 0;
    // Name, tag and time.
    let top = rect.top() + 13.0;
    let time = galley(ui, r.time, FontId::proportional(11.5), if strong { theme::ink() } else { theme::faint() }, f32::INFINITY);
    let tag = r.tag.map(|t| galley(ui, &t.to_uppercase(), FontId::monospace(9.0), theme::muted(), f32::INFINITY));
    let tag_w = tag.as_ref().map_or(0.0, |g| g.size().x + 18.0);
    let font = if strong { theme::font_medium(13.5) } else { FontId::proportional(13.5) };
    let name = galley(ui, r.title, font, if r.quiet { theme::muted() } else { theme::ink() }, right - x - time.size().x - 10.0 - tag_w);
    let name_w = name.size().x;
    p.galley(egui::pos2(x, top), name, theme::ink());
    if let Some(g) = tag {
        let b = Rect::from_min_size(egui::pos2(x + name_w + 6.0, top + 2.0), g.size() + egui::vec2(12.0, 2.0));
        p.rect_stroke(b, 0.0, hair(theme::border2()), StrokeKind::Inside);
        p.galley(b.min + egui::vec2(6.0, 1.0), g, theme::muted());
    }
    p.galley(egui::pos2(right - time.size().x, top + 1.5), time, theme::faint());
    // What was said last, and how much is unread.
    let y = top + 22.0;
    let count = strong.then(|| galley(ui, &r.unread.min(99).to_string(), theme::font_mono_medium(10.5), theme::on_ink(), f32::INFINITY));
    let pill_w = count.as_ref().map_or(0.0, |g| g.size().x.max(7.0) + 12.0);
    let said = galley(ui, r.preview, FontId::proportional(12.5), if strong { theme::bright() } else { theme::muted() }, right - x - pill_w - 8.0);
    p.galley(egui::pos2(x, y), said, theme::muted());
    if let Some(g) = count {
        let b = Rect::from_min_size(egui::pos2(right - pill_w, y), egui::vec2(pill_w, 17.0));
        p.rect_filled(b, 0.0, theme::ink());
        p.galley(b.center() - g.size() / 2.0, g, theme::on_ink());
    }
    p.hline(x..=rect.right(), rect.bottom() - 0.5, hair(theme::border()));
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Who reaches the inbox (the rest wait under Requests), what strangers
/// pay to write at all, whether anonymous senders are heard, and who is
/// blocked.
fn list_foot(m: &mut Messenger, d: &Data, ui: &mut Ui, out: &mut Vec<Out>) {
    const BARS: [(Option<f64>, &str); 4] =
        [(Some(0.0), "verified names"), (None, "everyone"), (Some(60.0), "names backed an hour+"), (Some(1_440.0), "names backed a day+")];
    if let Some(bits) = m.confirm_postage {
        postage_confirm(m, d, bits, ui, out);
        ui.add_space(8.0);
    }
    let now = BARS.iter().find(|(b, _)| *b == d.inbox_bar).map_or("names backed more", |(_, l)| *l);
    ui.horizontal(|ui| {
        theme::hint(ui, "Inbox takes");
        egui::ComboBox::from_id_salt("msgr_inbox_bar")
            .selected_text(RichText::new(now).size(12.0))
            .show_ui(ui, |ui| {
                for (b, label) in BARS {
                    if ui.selectable_label(d.inbox_bar == b, label).clicked() && d.inbox_bar != b {
                        out.push(Out::Go(Action::SetInboxBar { bar: b }));
                        out.push(Out::Go(Action::LoadStatus));
                    }
                }
            })
            .response
            .on_hover_text("The rest wait under Requests, never dropped. Backing is MDS burned for a name (the Burns page), in blocks of issuance.");
    });
    postage_picker(m, d, ui);
    ui.horizontal(|ui| {
        let mut drop = d.drop_anonymous;
        if ui
            .checkbox(&mut drop, RichText::new("Drop anonymous senders").size(12.0).color(theme::muted()))
            .on_hover_text("Messages from senders who do not say who they are go unread, rather than waiting under Requests.")
            .changed()
        {
            out.push(Out::Go(Action::SetDropAnonymous { on: drop }));
        }
        if !d.blocked.is_empty() {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button(RichText::new(format!("Blocked {}", d.blocked.len())).size(12.0), |ui| {
                    for name in d.blocked {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(name).size(12.5));
                            if ui.small_button("Unblock").clicked() {
                                out.push(Out::Go(Action::SealedUnblock { name: name.clone() }));
                                ui.close_menu();
                            }
                        });
                    }
                });
            });
        }
    });
}

/// What strangers pay: the stamp a message from someone you have not
/// answered must carry. Picking another price asks first.
fn postage_picker(m: &mut Messenger, d: &Data, ui: &mut Ui) {
    let p = d.postage;
    ui.horizontal(|ui| {
        theme::hint(ui, "Strangers pay");
        ui.add_enabled_ui(!p.names.is_empty(), |ui| {
            egui::ComboBox::from_id_salt("msgr_postage")
                .selected_text(RichText::new(price_label(p.bits)).size(12.0))
                .show_ui(ui, |ui| {
                    for (bits, label) in PRICES {
                        let text = format!("{label} · {}", work(bits));
                        if ui.selectable_label(p.bits == bits, text).clicked() && p.bits != bits && p.pending != Some(bits) {
                            m.confirm_postage = Some(bits);
                        }
                    }
                })
                .response
                .on_hover_text(
                    "Each message from someone you have not answered carries a stamp of work at this price, or it is dropped unread. \
                     Raise it if strangers flood you: every message they send costs them that much more. People you have \
                     written back to pay nothing. Times are for a computer like this one.",
                )
                .on_disabled_hover_text("None of your names takes sealed messages yet: give one a kem record on the Names page.");
        });
    });
    if let Some(next) = p.pending {
        theme::hint(ui, &format!("Changing to {}: it applies once your names' new records confirm.", price_label(next)));
    }
}

/// Asks before a new price goes on chain: what it costs strangers, and what
/// it costs this wallet (a record change on each name, with its fee).
fn postage_confirm(m: &mut Messenger, d: &Data, bits: u32, ui: &mut Ui, out: &mut Vec<Out>) {
    let names = if d.postage.settable.is_empty() { "Your names".to_string() } else { and_list(&d.postage.settable) };
    let (ask, button) = match bits.cmp(&d.postage.bits) {
        std::cmp::Ordering::Greater => ("Raise", format!("Raise to {}", price_label(bits))),
        _ if bits == POSTAGE_STANDARD => ("Lower", "Back to standard".to_string()),
        _ => ("Lower", format!("Lower to {}", price_label(bits))),
    };
    let text = format!(
        "{ask} what strangers pay to {}? Each message from someone you have not answered then takes {} of work on a computer like this one, or it is dropped. \
         {names} change{} their records on chain, a network fee each; the price applies once that confirms.",
        price_label(bits),
        work(bits),
        if d.postage.settable.len() == 1 { "s" } else { "" },
    );
    Frame::NONE.fill(theme::panel()).stroke(hair(theme::border2())).inner_margin(Margin::same(10)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.add(egui::Label::new(RichText::new(text).size(12.0).color(theme::bright())).wrap());
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button(button).clicked() {
                out.push(Out::Go(Action::SetPostage { bits }));
                m.confirm_postage = None;
            }
            if ui.button("Cancel").clicked() {
                m.confirm_postage = None;
            }
        });
    });
}

/// Names as a sentence lists them: "a, b and c".
fn and_list(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

// ── The open conversation ───────────────────────────────────────────────────

/// Nothing open: what this is, and how to start.
fn empty(m: &mut Messenger, ui: &mut Ui) {
    let full = ui.available_rect_before_wrap();
    let w = full.width().min(400.0);
    let col = Rect::from_min_size(egui::pos2(full.center().x - w / 2.0, full.top() + (full.height() * 0.3).max(30.0)), egui::vec2(w, full.height()));
    ui.scope_builder(egui::UiBuilder::new().max_rect(col).layout(Layout::top_down(Align::Center)), |ui| {
        ui.label(RichText::new("Sealed messages").font(theme::font_medium(16.0)).color(theme::ink()));
        ui.add_space(4.0);
        ui.add(
            egui::Label::new(
                RichText::new(
                    "Write to any name that takes them. Only its wallet can read what you send: messages are sealed post-quantum \
                     (ML-KEM), and nobody watching the network can see who they are for.",
                )
                .size(12.5)
                .color(theme::muted()),
            )
            .wrap(),
        );
        ui.add_space(12.0);
        if ui.button("New message").clicked() {
            m.start_new();
        }
    });
}

/// A strip under a conversation's header: a request, a warning, a confirm.
fn banner(ui: &mut Ui, id: &str, text: &str, buttons: impl FnOnce(&mut Ui)) {
    egui::TopBottomPanel::top(egui::Id::new(("msgr_banner", id)))
        .frame(Frame::NONE.fill(theme::panel()).inner_margin(Margin { left: 18, right: 14, top: 10, bottom: 10 }))
        .show_inside(ui, |ui| {
            ui.add(egui::Label::new(RichText::new(text).size(12.5).color(theme::bright())).wrap());
            ui.add_space(2.0);
            ui.horizontal(buttons);
        });
}

fn error_line(d: &Data, ui: &mut Ui, out: &mut Vec<Out>) {
    if d.error.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        ui.add(egui::Label::new(RichText::new(d.error).size(12.0).color(theme::bright())).wrap());
        if ui.small_button("Dismiss").clicked() {
            out.push(Out::ClearError);
        }
    });
    ui.add_space(4.0);
}

fn thread(m: &mut Messenger, d: &Data, c: &ConversationView, ui: &mut Ui, narrow: bool, out: &mut Vec<Out>) {
    let k = key(c);
    let pending = c.id.is_empty();
    let named = !c.peer.is_empty();
    // What is on screen, in a window in front of someone, has been read.
    if c.unread > 0 && d.focused && !pending {
        let mark = (k.clone(), c.lines.len());
        if m.marked.as_ref() != Some(&mark) {
            m.marked = Some(mark);
            out.push(Out::Read(c.id.clone()));
        }
    }
    if m.unread_from.as_ref().map(|(x, _)| x) != Some(&k) {
        m.unread_from = Some((k.clone(), (c.unread > 0).then(|| c.lines.len().saturating_sub(c.unread))));
    }

    egui::TopBottomPanel::top("msgr_thread_head").frame(head_frame()).show_inside(ui, |ui| {
        ui.horizontal(|ui| {
            back(m, ui, narrow);
            tile(ui, &initial(title(c)), 38.0, !named);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                ui.horizontal(|ui| {
                    ui.label(RichText::new(title(c)).font(theme::font_medium(15.0)).color(theme::ink()));
                    if pending {
                        theme::badge(ui, if stamping(c) > 0 { "stamping" } else { "waiting" }, theme::muted());
                    } else if c.verified {
                        theme::badge(ui, "verified", theme::ink());
                    } else if named {
                        theme::badge(ui, "unverified", theme::muted());
                    }
                    if let Some(t) = midstate_walletd::burns::tier(c.backing) {
                        theme::badge(ui, t, theme::ink());
                    }
                    if c.ttl > 0 {
                        theme::badge(ui, &format!("disappears {}", timer_text(c.ttl)), theme::muted());
                    }
                });
                theme::hint(ui, &subtitle(c));
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button("More", |ui| {
                    if pending {
                        if ui.button("Stop sending").clicked() {
                            out.push(Out::Go(Action::SealedCancel { to: c.peer.clone() }));
                            ui.close_menu();
                        }
                    } else {
                        ui.menu_button("Disappearing messages", |ui| {
                            for t in SEALED_TIMERS {
                                let label = if t == 0 { "Off".to_string() } else { timer_text(t) };
                                if ui.radio(c.ttl == t, label).clicked() && c.ttl != t {
                                    out.push(Out::Go(Action::SealedTimer { id: c.id.clone(), ttl: t }));
                                    ui.close_menu();
                                }
                            }
                        });
                        let block = if blocks_name(c) { format!("Block {}…", c.peer) } else { "Block this sender…".to_string() };
                        if ui.button(block).clicked() {
                            m.confirm_block = Some(k.clone());
                            ui.close_menu();
                        }
                        if ui.button("Delete conversation…").clicked() {
                            m.confirm_delete = Some(k.clone());
                            ui.close_menu();
                        }
                    }
                    if named && ui.button("Copy name").clicked() {
                        ui.ctx().copy_text(c.peer.clone());
                        ui.close_menu();
                    }
                });
                if named && c.verified && ui.button("Pay").on_hover_text(format!("Send MDS to {}", c.peer)).clicked() {
                    out.push(Out::Pay(c.peer.clone()));
                }
            });
        });
    });

    if m.confirm_block.as_deref() == Some(k.as_str()) {
        let text = if blocks_name(c) {
            format!(
                "Block {0}? Its conversations are deleted, and nothing {0} sends reaches you until you unblock it, from Blocked at the foot of the list.",
                c.peer
            )
        } else {
            "Block this sender? This conversation is deleted and nothing more sent in it is read. A sender who does not prove a name cannot be told from other strangers: a new conversation from them would arrive as a new request.".to_string()
        };
        banner(ui, &k, &text, |ui| {
            if ui.button("Block").clicked() {
                out.push(Out::Go(Action::SealedBlock { id: c.id.clone() }));
                m.confirm_block = None;
            }
            if ui.button("Cancel").clicked() {
                m.confirm_block = None;
            }
        });
    } else if m.confirm_delete.as_deref() == Some(k.as_str()) {
        banner(ui, &k, "Delete this conversation? Nothing more sent in it will be read here, and what was said cannot be brought back.", |ui| {
            if ui.button("Delete").clicked() {
                out.push(Out::Go(Action::SealedDelete { id: c.id.clone() }));
                m.confirm_delete = None;
            }
            if ui.button("Keep it").clicked() {
                m.confirm_delete = None;
            }
        });
    } else if c.request {
        let text = format!("A message request: {}. Reply or accept to move it to your inbox.", request_reason(c, d.inbox_bar));
        banner(ui, &k, &text, |ui| {
            if ui.button("Accept").clicked() {
                out.push(Out::Go(Action::SealedAccept { id: c.id.clone() }));
                m.requests = false;
            }
            if ui.button("Block").clicked() {
                m.confirm_block = Some(k.clone());
            }
            if ui.button("Delete").clicked() {
                m.confirm_delete = Some(k.clone());
            }
            if named && !c.verified && ui.button(format!("Write to the real {}", c.peer)).clicked() {
                m.compose_to(&c.peer);
            }
        });
    } else if named && !c.verified && !pending {
        let text = format!(
            "This sender claims to be {0}, but the claim did not check out against the registry. Your replies go to whoever opened this conversation, not to {0}.",
            c.peer
        );
        banner(ui, &k, &text, |ui| {
            if ui.button(format!("Write to the real {}", c.peer)).clicked() {
                m.compose_to(&c.peer);
            }
        });
    }

    egui::TopBottomPanel::bottom("msgr_thread_compose").frame(compose_frame()).show_inside(ui, |ui| {
        error_line(d, ui, out);
        let waiting = stamping(c);
        if waiting > 0 {
            ui.horizontal(|ui| {
                let each = work(c.stamp_bits.max(1));
                let text = match waiting {
                    1 => format!("Stamping 1 message: {each} on this computer."),
                    n => format!("Stamping {n} messages: {each} each on this computer."),
                };
                theme::hint(ui, &text);
                if ui.small_button("Don't send").on_hover_text("Drop what is still being stamped. Nothing of it has gone.").clicked() {
                    out.push(Out::Go(if pending { Action::SealedCancel { to: c.peer.clone() } } else { Action::SealedUnsend { id: c.id.clone() } }));
                }
            });
            ui.add_space(4.0);
        }
        let hint = if pending {
            format!("Add to what waits for {}", c.peer)
        } else if c.verified {
            format!("Message {}", c.peer)
        } else {
            "Reply to whoever opened this".into()
        };
        let note = c.stamping.then(|| stamp_note(if named { c.peer.as_str() } else { "they" }, c.stamp_bits));
        if let Some(text) = composer(m, ui, &k, &hint, d.sealed_busy, note.as_deref()) {
            out.push(Out::ClearError);
            out.push(Out::Go(if pending { Action::SealedSend { to: c.peer.clone(), text } } else { Action::SealedReply { id: c.id.clone(), text } }));
            m.sending = Some(Sending::Thread(k.clone()));
            if c.request {
                m.requests = false;
            }
        }
    });

    egui::CentralPanel::default().frame(Frame::NONE.fill(theme::bg())).show_inside(ui, |ui| {
        let unread_at = m.unread_from.as_ref().and_then(|(_, i)| *i);
        egui::ScrollArea::vertical().id_salt(("msgr_lines", &k)).stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            lines(ui, &k, c, unread_at, d.now);
        });
    });
}

/// A line that is not a message (a timer's change), centred.
fn note(ui: &mut Ui, text: &str) {
    ui.add_space(10.0);
    let g = galley(ui, text, FontId::proportional(11.5), theme::muted(), ui.available_width() - 2.0 * SIDE);
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), g.size().y), Sense::hover());
    ui.painter().galley(egui::pos2(row.center().x - g.size().x / 2.0, row.top()), g, theme::muted());
    ui.add_space(4.0);
}

fn lines(ui: &mut Ui, k: &str, c: &ConversationView, unread_at: Option<usize>, now: u64) {
    ui.spacing_mut().item_spacing.y = 0.0;
    ui.add_space(6.0);
    let maxw = ((ui.available_width() - 2.0 * SIDE) * 0.72).clamp(160.0, 560.0);
    let ls = &c.lines;
    for (i, l) in ls.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| &ls[j]);
        let next = ls.get(i + 1);
        let new_day = prev.is_none_or(|p| theme::local_day(p.at) != theme::local_day(l.at));
        let day = theme::day_label(l.at, now).to_uppercase();
        match (new_day, unread_at == Some(i)) {
            (true, true) => divider(ui, &format!("{day} · NEW"), true),
            (true, false) => divider(ui, &day, false),
            (false, true) => divider(ui, "NEW", true),
            (false, false) => {}
        }
        if l.timer.is_some() {
            note(ui, &timer_note(c, l));
            continue;
        }
        let starts = prev.is_none_or(|p| !same_run(p, l)) || unread_at == Some(i);
        let ends = next.is_none_or(|n| !same_run(l, n)) || unread_at == Some(i + 1);
        ui.add_space(if starts { 8.0 } else { 3.0 });
        let mut when = theme::local_datetime(l.at);
        if l.expires > 0 {
            when.push_str(&format!(" · disappears in {}", theme::fmt_duration(l.expires.saturating_sub(now))));
        }
        let unsent = l.pending || l.stamping;
        if l.stamping {
            when.push_str(" · stamping, not sent yet");
        }
        bubble(ui, egui::Id::new(("msgr_line", k, i)), &l.text, l.mine, unsent, maxw, &when);
        if ends {
            let when = match (l.pending, l.stamping) {
                (true, _) => format!("{} · not sent yet", theme::clock(l.at)),
                (_, true) => format!("{} · stamping", theme::clock(l.at)),
                _ => theme::clock(l.at),
            };
            stamp(ui, &when, l.mine);
        }
    }
    ui.add_space(16.0);
}

/// A day's heading, or the line above what has not been read.
fn divider(ui: &mut Ui, text: &str, strong: bool) {
    ui.add_space(14.0);
    let g = galley(ui, text, FontId::monospace(9.5), if strong { theme::ink() } else { theme::faint() }, f32::INFINITY);
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), g.size().y), Sense::hover());
    let mid = row.center();
    let half = g.size().x / 2.0 + 10.0;
    let stroke = hair(if strong { theme::border2() } else { theme::border() });
    ui.painter().hline((row.left() + SIDE)..=(mid.x - half), mid.y, stroke);
    ui.painter().hline((mid.x + half)..=(row.right() - SIDE), mid.y, stroke);
    ui.painter().galley(egui::pos2(mid.x - g.size().x / 2.0, row.top()), g, theme::faint());
    ui.add_space(2.0);
}

/// One message: this wallet's in ink on the right, the other side's framed
/// on the left, one not sent yet in outline. Hover for its time; right-click
/// to copy it.
fn bubble(ui: &mut Ui, id: egui::Id, text: &str, mine: bool, pending: bool, maxw: f32, when: &str) {
    let (fill, color, stroke) = if pending {
        (Color32::TRANSPARENT, theme::muted(), hair(theme::border2()))
    } else if mine {
        (theme::ink(), theme::on_ink(), Stroke::NONE)
    } else {
        (theme::panel(), theme::ink(), hair(theme::border()))
    };
    let pad = egui::vec2(12.0, 8.0);
    let g = ui.fonts(|f| f.layout(text.to_owned(), FontId::proportional(13.5), color, maxw - 2.0 * pad.x));
    let size = g.size() + 2.0 * pad;
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), size.y), Sense::hover());
    let x = if mine { row.right() - SIDE - size.x } else { row.left() + SIDE };
    let r = Rect::from_min_size(egui::pos2(x, row.top()), size);
    let resp = ui.interact(r, id, Sense::click());
    ui.painter().rect(r, 0.0, fill, stroke, StrokeKind::Inside);
    ui.painter().galley(r.min + pad, g, color);
    resp.on_hover_text(when).context_menu(|ui| {
        if ui.button("Copy").clicked() {
            ui.ctx().copy_text(text.to_owned());
            ui.close_menu();
        }
    });
}

fn stamp(ui: &mut Ui, text: &str, mine: bool) {
    let g = galley(ui, text, FontId::proportional(10.5), theme::faint(), f32::INFINITY);
    let (row, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), g.size().y + 3.0), Sense::hover());
    let x = if mine { row.right() - SIDE - g.size().x } else { row.left() + SIDE };
    ui.painter().galley(egui::pos2(x, row.top() + 3.0), g, theme::faint());
}

/// The box a sealed message is written in, and its Send button. Enter sends;
/// Shift+Enter starts a new line. `stamping`, said under it, is what the
/// stamp on what is sent costs. Returns the text to send, once.
fn composer(m: &mut Messenger, ui: &mut Ui, k: &str, hint: &str, busy: bool, stamping: Option<&str>) -> Option<String> {
    let id = egui::Id::new(("msgr_composer", k));
    let enter = ui.memory(|mem| mem.has_focus(id)) && ui.input(|i| plain_enter(&i.events));
    let focus = std::mem::take(&mut m.focus);
    let draft = m.drafts.entry(k.to_string()).or_default();
    let bytes = draft.trim().len();
    let ok = !busy && bytes > 0 && bytes <= SEALED_MAX;
    let mut send = false;
    let text_w = box_width(ui);
    ui.horizontal(|ui| {
        egui::ScrollArea::vertical().id_salt(("msgr_composer_scroll", k)).max_height(132.0).show(ui, |ui| {
            let r = ui.add(
                TextEdit::multiline(draft)
                    .id(id)
                    .hint_text(placeholder(hint))
                    .desired_rows(1)
                    .desired_width(text_w)
                    .margin(BOX_MARGIN)
                    .return_key(KeyboardShortcut::new(Modifiers::SHIFT, Key::Enter)),
            );
            if focus {
                r.request_focus();
            }
        });
        let label = if busy { "Sealing…" } else { "Send" };
        if ui.add_enabled(ok, egui::Button::new(label).min_size(egui::vec2(SEND_W, 34.0))).clicked() {
            send = true;
        }
    });
    if let Some(why) = stamping {
        ui.add(egui::Label::new(RichText::new(why).size(11.0).color(theme::faint())).wrap());
    }
    ui.horizontal(|ui| {
        ui.label(RichText::new("Enter to send · Shift+Enter for a new line").size(11.0).color(theme::faint()));
        if bytes > SEALED_MAX * 4 / 5 {
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let over = bytes > SEALED_MAX;
                let text = if over { format!("{bytes} / {SEALED_MAX} bytes: too long") } else { format!("{bytes} / {SEALED_MAX} bytes") };
                ui.label(RichText::new(text).size(11.0).color(if over { theme::ink() } else { theme::faint() }));
            });
        }
    });
    (send || (enter && ok)).then(|| draft.trim().to_string())
}

// ── A new message ───────────────────────────────────────────────────────────

/// Whether a recipient takes a message, and if not, why.
fn recipient_says(card: &NameCard) -> Result<(), String> {
    match card.state.as_str() {
        _ if card.sealed && card.state != "yours" => Ok(()),
        "yours" => Err(format!("{} is your own name.", card.name)),
        "available" | "unclaimed" => Err(format!("No one holds {}.", card.name)),
        "lapsed" => Err(format!("{} has lapsed: no one holds it now.", card.name)),
        _ => Err(format!("{} takes no sealed messages: its records carry no kem key.", card.name)),
    }
}

fn new_message(m: &mut Messenger, d: &Data, ui: &mut Ui, narrow: bool, out: &mut Vec<Out>) {
    let want = midstate_walletd::names::normalize(&m.to).ok().map(|n| format!("{n}{}", midstate_walletd::names::SUFFIX));
    if let Some(w) = &want {
        if m.asked.as_ref() != Some(w) {
            m.asked = Some(w.clone());
            out.push(Out::Go(Action::CheckRecipient { name: w.clone() }));
        }
    }
    let answer = m.recipient.as_ref().filter(|(n, _)| Some(n) == want.as_ref()).map(|(_, a)| a.clone());
    let blocked = want.as_ref().is_some_and(|w| d.blocked.contains(w));
    let ready = !blocked && matches!(&answer, Some(Ok(card)) if recipient_says(card).is_ok());
    let existing = want.as_ref().and_then(|w| d.conversations.iter().find(|c| c.peer == *w && c.verified)).map(key);

    egui::TopBottomPanel::top("msgr_new_head").frame(head_frame()).show_inside(ui, |ui| {
        ui.horizontal(|ui| {
            back(m, ui, narrow);
            ui.label(RichText::new("New message").font(theme::font_medium(15.0)).color(theme::ink()));
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("To").color(theme::muted()));
            let r = ui.add(TextEdit::singleline(&mut m.to).id(egui::Id::new("msgr_to")).hint_text(placeholder("name.mds")).desired_width(260.0).margin(Margin::symmetric(8, 6)));
            // The cursor goes where writing starts: the name, unless one was
            // given; then the message, once the name checks out.
            if m.focus && m.to.trim().is_empty() {
                r.request_focus();
                m.focus = false;
            }
        });
        ui.add_space(2.0);
        let (status, tier) = match (&want, &answer) {
            _ if m.to.trim().is_empty() => ("Write to any name that takes sealed messages.".to_string(), None),
            (Some(w), _) if blocked => (format!("You blocked {w}."), None),
            (None, _) => ("Not a name: names are a–z, 0–9 and -.".to_string(), None),
            (Some(w), None) => (format!("Looking up {w}…"), None),
            (Some(_), Some(Err(e))) => (e.clone(), None),
            (Some(_), Some(Ok(card))) => match recipient_says(card) {
                Ok(()) if card.postage > POSTAGE_STANDARD => (
                    format!("{} takes sealed messages, and asks strangers more: {}.", card.name, price_label(card.postage).to_lowercase()),
                    card.backing.as_ref().map(|b| b.tier.clone()).filter(|t| !t.is_empty()),
                ),
                Ok(()) => (format!("{} takes sealed messages.", card.name), card.backing.as_ref().map(|b| b.tier.clone()).filter(|t| !t.is_empty())),
                Err(why) => (why, None),
            },
        };
        ui.horizontal(|ui| {
            theme::hint(ui, &status);
            if let Some(t) = tier {
                theme::badge(ui, &t, theme::ink());
            }
            if let (true, Some(w)) = (blocked, &want) {
                if ui.small_button("Unblock").clicked() {
                    out.push(Out::Go(Action::SealedUnblock { name: w.clone() }));
                }
            }
        });
        if let Some(k) = existing.clone() {
            ui.horizontal(|ui| {
                theme::hint(ui, "You already have a conversation with them.");
                if ui.small_button("Open it").clicked() {
                    m.requests = false;
                    m.open(Open::Thread(k));
                }
            });
        }
    });

    egui::TopBottomPanel::bottom("msgr_new_compose").frame(compose_frame()).show_inside(ui, |ui| {
        error_line(d, ui, out);
        let focus_later = !ready && std::mem::take(&mut m.focus);
        ui.add_enabled_ui(ready, |ui| {
            let hint = match &want {
                Some(w) if ready => format!("Message {w}"),
                Some(w) if blocked => format!("Unblock {w} to write to it"),
                _ => "Choose who to write to first".to_string(),
            };
            let price = match &answer {
                Some(Ok(card)) if ready && card.postage > 0 => card.postage,
                _ => POSTAGE_STANDARD,
            };
            let note = stamp_note(want.as_deref().filter(|_| ready).unwrap_or("they"), price);
            if let Some(text) = composer(m, ui, NEW, &hint, d.sealed_busy, Some(&note)) {
                if let Some(w) = want.clone() {
                    out.push(Out::ClearError);
                    out.push(Out::Go(Action::SealedSend { to: w.clone(), text }));
                    m.sending = Some(Sending::To(w));
                }
            }
        });
        m.focus |= focus_later;
    });
    egui::CentralPanel::default().frame(Frame::NONE.fill(theme::bg())).show_inside(ui, |ui| {
        if let (true, Some(w)) = (ready, &want) {
            let full = ui.available_rect_before_wrap();
            let width = full.width().min(420.0);
            let col = Rect::from_min_size(egui::pos2(full.center().x - width / 2.0, full.top() + (full.height() * 0.38).max(20.0)), egui::vec2(width, 80.0));
            ui.scope_builder(egui::UiBuilder::new().max_rect(col).layout(Layout::top_down(Align::Center)), |ui| {
                let text = format!(
                    "Only {w}'s wallet can read what you send. If you hold a name, your first message is signed as it, so they see who wrote; if not, it arrives anonymous, and may wait among their requests. Until they write back, they see at most three of your messages."
                );
                ui.add(egui::Label::new(RichText::new(text).size(12.5).color(theme::muted())).wrap());
            });
        }
    });
}

// ── The public room ─────────────────────────────────────────────────────────

fn public(m: &mut Messenger, d: &Data, ui: &mut Ui, narrow: bool, out: &mut Vec<Out>) {
    egui::TopBottomPanel::top("msgr_public_head").frame(head_frame()).show_inside(ui, |ui| {
        ui.horizontal(|ui| {
            back(m, ui, narrow);
            tile(ui, "#", 38.0, false);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                ui.label(RichText::new("Public room").font(theme::font_medium(15.0)).color(theme::ink()));
                theme::hint(ui, "Everyone on the network reads it, and nodes keep it. Each message is mined by your node; names here are peer ids.");
            });
        });
    });

    egui::TopBottomPanel::bottom("msgr_public_compose").frame(compose_frame()).show_inside(ui, |ui| {
        error_line(d, ui, out);
        if let Some(text) = public_composer(m, d, ui) {
            out.push(Out::ClearError);
            out.push(Out::Go(Action::ChatSend { text }));
        }
    });

    egui::CentralPanel::default().frame(Frame::NONE.fill(theme::bg())).show_inside(ui, |ui| {
        // Wordless traffic (other people's sealed messages, channel frames)
        // rides the room too; it is not conversation.
        let said: Vec<&ChatView> = d.chat.iter().filter(|c| !c.text.is_empty()).collect();
        egui::ScrollArea::vertical().id_salt("msgr_public_lines").stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.add_space(6.0);
            if said.is_empty() {
                ui.add_space(18.0);
                ui.horizontal(|ui| {
                    ui.add_space(SIDE);
                    theme::hint(ui, "Nothing yet: the room is quiet, or its history is still arriving from peers.");
                });
            }
            let maxw = ((ui.available_width() - 2.0 * SIDE) * 0.72).clamp(160.0, 560.0);
            let run = |a: &ChatView, b: &ChatView| a.sender == b.sender && b.timestamp.saturating_sub(a.timestamp) <= RUN_SECS && theme::local_day(a.timestamp) == theme::local_day(b.timestamp);
            for (i, c) in said.iter().enumerate() {
                let prev = i.checked_sub(1).map(|j| said[j]);
                let next = said.get(i + 1).copied();
                if prev.is_none_or(|p| theme::local_day(p.timestamp) != theme::local_day(c.timestamp)) {
                    divider(ui, &theme::day_label(c.timestamp, d.now).to_uppercase(), false);
                }
                let starts = prev.is_none_or(|p| !run(p, c));
                ui.add_space(if starts { 8.0 } else { 3.0 });
                if starts && !c.mine {
                    stamp(ui, &peer_tag(&c.sender), false);
                    ui.add_space(2.0);
                }
                let text = if c.attachments > 0 { format!("{}  (+{} attached)", c.text, c.attachments) } else { c.text.clone() };
                bubble(ui, egui::Id::new(("msgr_public_line", c.timestamp, c.nonce)), &text, c.mine, false, maxw, &theme::local_datetime(c.timestamp));
                if next.is_none_or(|n| !run(c, n)) {
                    stamp(ui, &theme::clock(c.timestamp), c.mine);
                }
            }
            ui.add_space(16.0);
        });
    });
}

/// The public room's box: dictionary words only, with what is not one
/// pointed out and the words that start the same suggested.
fn public_composer(m: &mut Messenger, d: &Data, ui: &mut Ui) -> Option<String> {
    let id = egui::Id::new("msgr_public_box");
    let focus = std::mem::take(&mut m.focus);
    let draft = m.drafts.entry(PUBLIC.to_string()).or_default();
    let ok = !d.chat_busy && public_ok(draft, d.dict);
    let mut send = false;
    let text_w = box_width(ui);
    ui.horizontal(|ui| {
        let r = ui.add(
            TextEdit::singleline(draft)
                .id(id)
                .hint_text(placeholder("Up to ten words from the dictionary"))
                .font(egui::TextStyle::Monospace)
                .desired_width(text_w)
                .margin(BOX_MARGIN),
        );
        if focus {
            r.request_focus();
        }
        if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
            send = true;
            // Enter takes the cursor out of a one-line box: put it back.
            m.focus = true;
        }
        if ui.add_enabled(ok, egui::Button::new(if d.chat_busy { "Mining…" } else { "Send" }).min_size(egui::vec2(SEND_W, 34.0))).clicked() {
            send = true;
        }
    });
    let words: Vec<String> = draft.split_whitespace().map(str::to_lowercase).collect();
    // The word being typed is not wrong yet: it gets suggestions, not a flag.
    let typing = !draft.is_empty() && !draft.ends_with(' ');
    let done = if typing { draft.rsplit_once(' ').map_or("", |(head, _)| head) } else { draft.as_str() };
    let bad = not_in_dictionary(done, d.dict);
    let last_bad = typing && words.last().is_some_and(|w| !d.dict.iter().any(|x| x.eq_ignore_ascii_case(w)));
    if words.len() > 10 {
        theme::hint(ui, &format!("Ten words at most ({} typed).", words.len()));
    }
    let mut replace: Option<String> = None;
    if (!bad.is_empty() || last_bad) && !d.dict.is_empty() {
        if !bad.is_empty() {
            theme::hint(ui, &format!("Not in the dictionary: {}", bad.join(", ")));
        }
        if let Some(last) = words.last().filter(|_| last_bad) {
            let near: Vec<&String> = d.dict.iter().filter(|w| w.starts_with(last.as_str()) && !w.eq_ignore_ascii_case(last)).take(6).collect();
            if !near.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    theme::hint(ui, "try");
                    for w in near {
                        if ui.small_button(RichText::new(w).font(FontId::monospace(11.0))).clicked() {
                            replace = Some(w.clone());
                        }
                    }
                });
            }
        }
    }
    let mut insert: Option<String> = None;
    egui::CollapsingHeader::new(RichText::new(format!("Dictionary, {} words", d.dict.len())).size(11.5).color(theme::muted())).id_salt("msgr_dict").show(ui, |ui| {
        ui.add(TextEdit::singleline(&mut m.dict_filter).hint_text(placeholder("filter")).desired_width(180.0));
        let f = m.dict_filter.trim().to_lowercase();
        egui::ScrollArea::vertical().max_height(120.0).id_salt("msgr_dict_words").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for w in d.dict.iter().filter(|w| f.is_empty() || w.to_lowercase().contains(&f)) {
                    if ui.small_button(RichText::new(w).font(FontId::monospace(11.0))).on_hover_text("add to your message").clicked() {
                        insert = Some(w.clone());
                    }
                }
            });
        });
    });
    let draft = m.drafts.entry(PUBLIC.to_string()).or_default();
    if let Some(w) = replace {
        let mut parts: Vec<String> = draft.split_whitespace().map(str::to_string).collect();
        if let Some(l) = parts.last_mut() {
            *l = w;
        }
        *draft = parts.join(" ") + " ";
    }
    if let Some(w) = insert {
        if !draft.is_empty() && !draft.ends_with(' ') {
            draft.push(' ');
        }
        draft.push_str(&w);
    }
    (send && public_ok(draft, d.dict) && !d.chat_busy).then(|| draft.trim().to_string())
}

// ── In the app ──────────────────────────────────────────────────────────────

/// The Messages tab: the messenger, with the app's state, and what it asks
/// done.
pub fn show(app: &mut App, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let full = ui.available_rect_before_wrap();
    let w = full.width().min(1240.0);
    let area = Rect::from_min_size(egui::pos2(full.center().x - w / 2.0, full.top()), egui::vec2(w, full.height()));
    let standard = PostageView { bits: POSTAGE_STANDARD, ..PostageView::default() };
    let outs = {
        let d = Data {
            conversations: &app.conversations,
            chat: &app.chat,
            dict: &app.chat_dict,
            inbox_bar: app.status.as_ref().and_then(|s| s.inbox_bar),
            blocked: app.status.as_ref().map_or(&[][..], |s| s.blocked.as_slice()),
            drop_anonymous: app.status.as_ref().is_some_and(|s| s.drop_anonymous),
            postage: app.status.as_ref().map_or(&standard, |s| &s.postage),
            sealed_busy: app.sealed_busy,
            chat_busy: app.chat_busy,
            error: &app.error,
            now: theme::now_secs(),
            focused: ctx.input(|i| i.viewport().focused.unwrap_or(true)),
        };
        let msgr = &mut app.msgr;
        ui.scope_builder(egui::UiBuilder::new().max_rect(area), |ui| draw(msgr, &d, ui)).inner
    };
    for o in outs {
        match o {
            Out::Go(a) => {
                match &a {
                    Action::SealedSend { .. } | Action::SealedReply { .. } => app.sealed_busy = true,
                    Action::ChatSend { .. } => app.chat_busy = true,
                    _ => {}
                }
                app.go(&ctx, a);
            }
            Out::Read(id) => {
                if let Some(c) = app.conversations.iter_mut().find(|c| c.id == id) {
                    c.unread = 0;
                }
                app.go(&ctx, Action::SealedRead { id });
            }
            Out::Pay(name) => {
                app.send_to = name.clone();
                app.addr_ok = None;
                app.addr_resolved = None;
                app.send_asset = 0;
                app.go(&ctx, Action::ValidateAddress { addr: name });
                app.tab = Tab::Send;
            }
            Out::ClearError => app.error.clear(),
        }
    }
}

/// Unread messages in the inbox, for the tab's label.
pub fn unread(conversations: &[ConversationView]) -> usize {
    conversations.iter().filter(|c| !c.request).map(|c| c.unread).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(mine: bool, text: &str, at: u64) -> MessageLineView {
        MessageLineView { mine, text: text.into(), at, pending: false, expires: 0, timer: None, stamping: false }
    }

    fn conv(id: &str, peer: &str, verified: bool, lines: Vec<MessageLineView>) -> ConversationView {
        ConversationView {
            id: id.into(),
            peer: peer.into(),
            verified,
            opened_by_me: false,
            lines,
            backing: 0.0,
            request: false,
            unread: 0,
            ttl: 0,
            stamping: false,
            stamp_bits: 0,
        }
    }

    static STANDARD: PostageView = PostageView { bits: POSTAGE_STANDARD, pending: None, names: Vec::new(), settable: Vec::new() };

    fn data(conversations: &[ConversationView]) -> Data<'_> {
        Data {
            conversations,
            chat: &[],
            dict: &[],
            inbox_bar: Some(0.0),
            blocked: &[],
            drop_anonymous: false,
            postage: &STANDARD,
            sealed_busy: false,
            chat_busy: false,
            error: "",
            now: 0,
            focused: true,
        }
    }

    #[test]
    fn timers_read_as_people_say_them() {
        assert_eq!(SEALED_TIMERS.map(timer_text), ["off", "1 hour", "1 day", "1 week", "4 weeks"].map(String::from));
        assert_eq!(timer_text(2 * 86_400), "2 days");
        assert_eq!(timer_text(90), "2 minutes");
        let mut c = conv("ab", "alice.mds", true, vec![]);
        let set = MessageLineView { timer: Some(86_400), ..line(false, "", 5) };
        assert_eq!(timer_note(&c, &set), "alice.mds set messages to disappear after 1 day");
        assert_eq!(timer_note(&c, &MessageLineView { mine: true, timer: Some(0), ..set.clone() }), "You turned off disappearing messages");
        c.lines.push(set.clone());
        assert_eq!(preview(&c), "alice.mds set messages to disappear after 1 day");
        assert!(!same_run(&line(false, "a", 1), &set) && !same_run(&set, &line(false, "b", 6)), "a note breaks a run");
        c.lines.clear();
        c.ttl = 3_600;
        assert_eq!(preview(&c), "Messages disappear after 1 hour", "all its messages have gone");
        assert!(blocks_name(&c) && !blocks_name(&conv("cd", "", false, vec![])) && !blocks_name(&conv("ef", "carol.mds", false, vec![])));
    }

    #[test]
    fn a_conversation_reads_as_a_messenger_lists_it() {
        let mut c = conv("ab", "alice.mds", true, vec![line(false, "hi\nthere", 10), line(true, "hey   you", 20)]);
        assert_eq!((key(&c), title(&c), initial(title(&c))), ("ab".to_string(), "alice.mds", "A".to_string()));
        assert_eq!(preview(&c), "You: hey you");
        c.lines.pop();
        assert_eq!(preview(&c), "hi there");
        assert!(matches(&c, "ali") && matches(&c, "there") && !matches(&c, "bob"));
        let anon = conv("cd", "", false, vec![line(false, "psst", 5)]);
        assert_eq!((title(&anon), initial(title(&anon))), ("Anonymous", "?".to_string()));
        let waiting = ConversationView { id: String::new(), opened_by_me: true, ..conv("", "dave.mds", false, vec![MessageLineView { pending: true, ..line(true, "yo", 1) }]) };
        assert_eq!((key(&waiting), preview(&waiting)), ("to:dave.mds".to_string(), "Waiting for its key: yo".to_string()));
        assert_eq!(peer_tag("12D3KooWQxyzAbCdEf"), "…AbCdEf");
    }

    #[test]
    fn enter_sends_and_shift_enter_does_not() {
        let key = |shift: bool, pressed: bool| egui::Event::Key {
            key: Key::Enter,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: if shift { Modifiers::SHIFT } else { Modifiers::NONE },
        };
        assert!(plain_enter(&[key(false, true)]));
        assert!(!plain_enter(&[key(true, true)]), "a new line");
        assert!(!plain_enter(&[key(false, false)]), "only the press");
        assert!(!plain_enter(&[egui::Event::Text("x".into())]));
    }

    #[test]
    fn messages_run_together_by_side_time_and_day() {
        let (a, b) = (line(false, "a", 1_000), line(false, "b", 1_000 + RUN_SECS));
        assert!(same_run(&a, &b));
        assert!(!same_run(&a, &line(false, "c", 1_001 + RUN_SECS)), "too far apart");
        assert!(!same_run(&a, &line(true, "d", 1_001)), "the other side");
        assert!(!same_run(&line(false, "e", 86_399), &line(false, "f", 86_401)), "another day");
    }

    #[test]
    fn the_public_room_takes_one_to_ten_dictionary_words() {
        let dict: Vec<String> = ["hello", "world", "midstate"].map(String::from).to_vec();
        assert!(public_ok("hello world", &dict));
        assert!(public_ok("HELLO", &dict), "case does not matter");
        assert!(!public_ok("hello there", &dict));
        assert_eq!(not_in_dictionary("hello there you", &dict), vec!["there", "you"]);
        assert!(!public_ok("", &dict) && !public_ok(&"hello ".repeat(11), &dict));
    }

    #[test]
    fn a_recipient_takes_a_message_only_if_its_name_can() {
        let card = |state: &str, sealed: bool| NameCard { name: "bob.mds".into(), state: state.into(), sealed, ..NameCard::default() };
        assert!(recipient_says(&card("held", true)).is_ok());
        assert!(recipient_says(&card("for sale", true)).is_ok());
        assert_eq!(recipient_says(&card("held", false)).unwrap_err(), "bob.mds takes no sealed messages: its records carry no kem key.");
        assert_eq!(recipient_says(&card("available", false)).unwrap_err(), "No one holds bob.mds.");
        assert_eq!(recipient_says(&card("yours", true)).unwrap_err(), "bob.mds is your own name.");
    }

    #[test]
    fn the_open_conversation_follows_what_the_wallet_does() {
        let alice = conv("aa", "alice.mds", true, vec![line(false, "hi", 1)]);
        let mut m = Messenger::default();
        // From the Names page: the conversation there is.
        m.compose_to("Alice.mds");
        m.settle(&data(std::slice::from_ref(&alice)));
        assert_eq!(m.open, Open::Thread("aa".into()));
        // A name with none: a new message to it.
        m.compose_to("bob.mds");
        m.settle(&data(std::slice::from_ref(&alice)));
        assert_eq!((m.open.clone(), m.to.as_str()), (Open::New, "bob.mds"));
        // Sent, and waiting for bob's key: follow it there.
        m.sending = Some(Sending::To("bob.mds".into()));
        m.drafts.insert(NEW.into(), "hello".into());
        m.sent();
        assert!(!m.drafts.contains_key(NEW));
        let waiting = ConversationView { id: String::new(), opened_by_me: true, ..conv("", "bob.mds", false, vec![MessageLineView { pending: true, ..line(true, "hello", 2) }]) };
        m.settle(&data(&[waiting, alice.clone()]));
        assert_eq!(m.open, Open::Thread("to:bob.mds".into()));
        // The key came, and the session opened: still bob.
        let opened = ConversationView { opened_by_me: true, ..conv("bb", "bob.mds", true, vec![line(true, "hello", 2)]) };
        m.settle(&data(&[opened, alice.clone()]));
        assert_eq!(m.open, Open::Thread("bb".into()));
        // Deleted: nothing open.
        m.settle(&data(std::slice::from_ref(&alice)));
        assert_eq!(m.open, Open::Nothing);
        // A failed send keeps what was written.
        m.drafts.insert("aa".into(), "draft".into());
        m.sending = Some(Sending::Thread("aa".into()));
        m.send_failed();
        assert_eq!(m.drafts.get("aa").map(String::as_str), Some("draft"));
    }

    #[test]
    fn prices_and_their_work_read_as_people_say_them() {
        assert_eq!(PRICES.map(|(b, _)| price_label(b)), ["Standard", "Raised", "High", "Very high", "Maximum"].map(String::from));
        assert_eq!(price_label(30), "30 bits");
        let names = ["a.mds", "b.mds", "c.mds"].map(String::from);
        assert_eq!([&names[..0], &names[..1], &names[..2], &names[..]].map(and_list), ["", "a.mds", "a.mds and b.mds", "a.mds, b.mds and c.mds"].map(String::from));
        assert!(PRICES.windows(2).all(|w| w[1].0 == w[0].0 + 4), "each step sixteen times the last");
        let said = [0.2, 1.4, 1.6, 44.0, 89.0, 91.0, 600.0, 3_700.0, 5_500.0, 40_000.0, 200_000.0].map(about);
        let want = [
            "about a second",
            "about a second",
            "about 2 seconds",
            "about 44 seconds",
            "about 89 seconds",
            "about 2 minutes",
            "about 10 minutes",
            "about 62 minutes",
            "about 2 hours",
            "about 11 hours",
            "about 2 days",
        ];
        assert_eq!(said, want.map(String::from));
        // The machine's own speed decides the words; the order never changes.
        assert!(midstate_walletd::stamp_seconds(36) > midstate_walletd::stamp_seconds(24));
        assert!(stamp_note("bob.mds", POSTAGE_STANDARD).starts_with("Until bob.mds writes back, each message carries a stamp: about"));
        assert!(stamp_note("bob.mds", 32).starts_with("bob.mds asks strangers more"));
        assert!(stamp_note("bob.mds", 32).contains("(high)"));
    }

    #[test]
    fn a_message_being_stamped_shows_as_one() {
        let mut slow = line(true, "hello there", 20);
        slow.stamping = true;
        let mut c = conv("ab", "bob.mds", true, vec![line(true, "hi", 10), slow.clone()]);
        c.stamping = true;
        c.stamp_bits = 32;
        assert_eq!(stamping(&c), 1);
        assert_eq!(preview(&c), "Stamping: hello there");
        assert!(!same_run(&c.lines[0], &slow), "what went and what waits are apart");
        // An opening being stamped: the key is in hand, it waits on the work.
        let opening = ConversationView { id: String::new(), opened_by_me: true, ..conv("", "bob.mds", false, vec![slow.clone()]) };
        assert!(subtitle(&opening).starts_with("Stamping your first message to bob.mds"));
        let waiting = ConversationView { id: String::new(), opened_by_me: true, ..conv("", "bob.mds", false, vec![MessageLineView { pending: true, ..line(true, "yo", 1) }]) };
        assert!(subtitle(&waiting).starts_with("Waiting for bob.mds's wallet"));
    }

    #[test]
    fn requests_say_why() {
        let mut c = conv("ab", "carol.mds", false, vec![]);
        assert!(request_reason(&c, Some(0.0)).contains("did not check out"));
        c.peer.clear();
        assert!(request_reason(&c, Some(0.0)).contains("did not say who"));
        let mut low = conv("cd", "dan.mds", true, vec![]);
        low.backing = 10.0;
        assert_eq!(request_reason(&low, Some(60.0)), "dan.mds has 10.0 blocks of backing, under your inbox's 1.0 hours of issuance");
        assert_eq!(unread(&[ConversationView { unread: 2, ..low.clone() }, ConversationView { unread: 5, request: true, ..low }]), 2);
    }
}
