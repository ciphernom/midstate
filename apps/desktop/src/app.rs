//! Application state and frame loop. All chain/wallet truth arrives as
//! [`Msg`]s from the bridge; the UI only renders state and emits [`Action`]s.

use crate::bridge::{dispatch, Action, Msg};
use crate::theme;
use crate::views;
use eframe::egui::{self, Align2, Color32, FontId, RichText};
use midstate_walletd::api::*;
use midstate_walletd::WalletdHandle;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Dashboard,
    Send,
    Receive,
    History,
    Coins,
    Chat,
    Channels,
    Trade,
    Node,
    Settings,
}

impl Tab {
    pub const ALL: [Tab; 10] = [
        Tab::Dashboard,
        Tab::Send,
        Tab::Receive,
        Tab::History,
        Tab::Coins,
        Tab::Chat,
        Tab::Channels,
        Tab::Trade,
        Tab::Node,
        Tab::Settings,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Send => "Send",
            Tab::Receive => "Receive",
            Tab::History => "History",
            Tab::Coins => "Coins",
            Tab::Chat => "Chat",
            Tab::Channels => "Channels",
            Tab::Trade => "Trade",
            Tab::Node => "Node",
            Tab::Settings => "Settings",
        }
    }
}

/// Sub-navigation within the Coins tab.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CoinsTab {
    Holdings,
    Housekeeping,
    Advanced,
}

/// Sub-navigation within the Channels tab.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChanTab {
    List,
    Pay,
    Hub,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Onboard {
    Menu,
    Create,
    Sheet,
    Confirm,
    Restore,
    Unlock,
}

pub struct App {
    rt: tokio::runtime::Runtime,
    pub msg_tx: Sender<Msg>,
    msg_rx: Receiver<Msg>,
    pub handle: Option<WalletdHandle>,
    pub boot_error: Option<String>,

    // Chain / wallet state
    pub status: Option<WalletStatus>,
    pub sync: Option<SyncStatus>,
    pub balance: Option<Balance>,
    pub coins: Vec<CoinView>,
    pub addresses: Vec<AddressInfo>,
    pub history: Vec<HistoryView>,
    pub sends: Vec<SendProgress>,
    pub node: Option<NodeInfo>,

    // UI state
    pub tab: Tab,
    pub onboard: Onboard,
    pub busy: bool,
    pub error: String,
    pub toasts: Vec<(f64, String)>,

    // Onboarding forms
    pub pw: String,
    pub pw2: String,
    pub phrase: String,
    pub mnemonic: Vec<String>,
    pub quiz: Vec<(usize, String)>,

    // Send form
    pub send_to: String,
    pub send_amount: String,
    pub send_private: bool,
    pub addr_ok: Option<bool>,
    pub addr_reason: Option<String>,

    // Receive
    pub recv_label: String,
    pub recv_mss: bool,
    pub current_addr: Option<AddressInfo>,
    pub qr: Option<(String, egui::TextureHandle)>,
    pub copied_at: Option<f64>,

    // Settings / node
    pub rescan_h: String,
    pub settings_msg: String,

    // Sync progress bookkeeping (client-side rate/ETA over recent ticks)
    pub t0: std::time::Instant,
    pub sync_samples: std::collections::VecDeque<(f64, u64)>,
    pub primer_page: usize,

    // Chat
    pub chat: Vec<ChatView>,
    pub chat_dict: Vec<String>,
    pub chat_input: String,
    pub chat_busy: bool,
    pub chat_last_poll: f64,

    // Q-Bolt channels
    pub channels: Vec<ChannelView>,
    pub chan_identity: Option<IdentityView>,
    pub chan_peer: String,
    pub chan_amount: String,
    pub chan_life: String,
    pub chan_pay: std::collections::HashMap<String, String>,
    pub chan_last_poll: f64,
    pub invoices: Vec<InvoiceView>,
    /// The hub policy as the daemon last reported it. Never written to by the
    /// settings UI — it is the baseline unsaved edits are compared against.
    pub hub: Option<HubView>,
    /// Local, unsaved edits to that policy.
    ///
    /// Kept separate from `hub` on purpose. Writing edits straight back into
    /// `hub` makes the two identical again on the very next frame, so the
    /// "changed" test goes false and the Save button disappears one frame
    /// after it appears — visible for about 16ms, and impossible to click.
    /// The edit then lives only in the GUI's memory and is never sent to the
    /// daemon, which is why toggling a policy appeared to work and changed
    /// nothing.
    pub hub_draft: Option<HubView>,
    pub inv_amount: String,
    pub pay_invoice_text: String,
    pub last_invoice: Option<InvoiceView>,
    pub node_last_poll: f64,
    pub theme_applied: Option<bool>,
    pub send_unit: usize,
    pub balance_unit: usize,
    pub dict_filter: String,
    pub ask_peer: String,
    pub ask_pending: bool,
    pub coins_tab: CoinsTab,
    pub hist_filter: usize,
    pub hist_search: String,
    pub hist_open: Option<usize>,
    pub evm: Option<EvmAccountView>,
    pub book: Option<OrderBookView>,
    pub dex_cfg: Option<DexConfigView>,
    /// What walletd currently holds, so edits have a baseline to differ from.
    pub dex_cfg_saved: Option<DexConfigView>,
    pub dex_tab: usize,
    pub dex_syncing: bool,
    pub dex_last_sync: f64,
    pub dex_last_poll: f64,
    // Guided swap
    pub swap_side: usize,
    pub swap_rail: usize,
    pub swap_mds: String,
    pub swap_eth: String,
    pub swap_peer: String,
    pub swap_hours: String,
    pub swap_mds_unit: usize,
    pub swap_eth_unit: usize,
    pub swap_unit: Option<usize>,
    /// Channel identity of the chosen order's maker, for the readiness check.
    pub swap_maker_pk: String,
    pub swap_quote: Option<SwapQuoteView>,
    pub my_orders: Vec<MyOrderView>,
    pub swaps: Vec<SwapView>,
    pub my_bids: Vec<MyBidView>,
    pub hubs: Vec<HubAdView>,
    pub bid_mds: String,
    pub bid_wei: String,
    pub bid_hours: String,
    pub bid_bond: String,
    pub ask_mds: String,
    pub ask_wei: String,
    pub ask_life: String,
    pub ask_notice: String,
    pub ask_mds_unit: usize,
    pub ask_eth_unit: usize,
    pub own_phrase: bool,
    pub chan_tab: ChanTab,
    pub verify_input: String,
    pub verify_result: Option<bool>,
    pub logo: Option<egui::TextureHandle>,
    pub req_payee: String,
    pub req_amount: String,

    // Coin-management forms (Coins tab)
    pub consolidate_addr: String,
    pub defrag_max: String,
    pub coins_notice: String,
    pub coin_export: Option<CoinExport>,
    pub import_seed: String,
    pub import_value: String,
    pub import_salt: String,
    pub import_label: String,
    pub abandon_addr: String,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::load_pref();
        theme::apply(&cc.egui_ctx);
        let logo = Some(theme::load_logo(&cc.egui_ctx));

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let (msg_tx, msg_rx) = std::sync::mpsc::channel::<Msg>();

        // Boot walletd + embedded node off-thread; UI shows progress meanwhile.
        {
            let tx = msg_tx.clone();
            let ctx = cc.egui_ctx.clone();
            rt.spawn(async move {
                match boot().await {
                    Ok(handle) => {
                        // Forward every walletd event into the message pump.
                        let mut rx = handle.subscribe();
                        let ev_tx = tx.clone();
                        let ev_ctx = ctx.clone();
                        tokio::spawn(async move {
                            loop {
                                match rx.recv().await {
                                    Ok(ev) => {
                                        let _ = ev_tx.send(Msg::Event(ev));
                                        ev_ctx.request_repaint();
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                        continue
                                    }
                                    Err(_) => break,
                                }
                            }
                        });
                        let _ = tx.send(Msg::Ready(handle));
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::BootFailed(format!("{e:#}")));
                    }
                }
                ctx.request_repaint();
            });
        }

        Self {
            rt,
            msg_tx,
            msg_rx,
            handle: None,
            boot_error: None,
            status: None,
            sync: None,
            balance: None,
            coins: Vec::new(),
            addresses: Vec::new(),
            history: Vec::new(),
            sends: Vec::new(),
            node: None,
            tab: Tab::Dashboard,
            onboard: Onboard::Menu,
            busy: false,
            error: String::new(),
            toasts: Vec::new(),
            pw: String::new(),
            pw2: String::new(),
            phrase: String::new(),
            mnemonic: Vec::new(),
            quiz: Vec::new(),
            send_to: String::new(),
            send_amount: String::new(),
            send_private: false,
            addr_ok: None,
            addr_reason: None,
            recv_label: String::new(),
            recv_mss: true,
            current_addr: None,
            qr: None,
            copied_at: None,
            rescan_h: String::new(),
            settings_msg: String::new(),
            t0: std::time::Instant::now(),
            sync_samples: std::collections::VecDeque::new(),
            primer_page: 0,
            chat: Vec::new(),
            chat_dict: Vec::new(),
            chat_input: String::new(),
            chat_busy: false,
            chat_last_poll: 0.0,
            channels: Vec::new(),
            chan_identity: None,
            chan_peer: String::new(),
            chan_amount: String::new(),
            chan_life: "4320".into(),
            chan_pay: std::collections::HashMap::new(),
            chan_last_poll: 0.0,
            invoices: Vec::new(),
            hub: None,
            hub_draft: None,
            inv_amount: String::new(),
            pay_invoice_text: String::new(),
            last_invoice: None,
            node_last_poll: 0.0,
            theme_applied: None,
            send_unit: 0,
            balance_unit: 3,
            dict_filter: String::new(),
            ask_peer: String::new(),
            ask_pending: false,
            coins_tab: CoinsTab::Holdings,
            hist_filter: 0,
            hist_search: String::new(),
            hist_open: None,
            evm: None,
            book: None,
            dex_cfg: None,
            dex_cfg_saved: None,
            dex_tab: 0,
            dex_syncing: false,
            dex_last_sync: -1.0,
            dex_last_poll: 0.0,
            swap_side: 0,
            swap_rail: 0,
            swap_mds: String::new(),
            swap_eth: String::new(),
            swap_peer: String::new(),
            swap_hours: "1".into(),
            swap_mds_unit: 0,
            swap_eth_unit: 0,
            swap_unit: None,
            swap_maker_pk: String::new(),
            swap_quote: None,
            my_orders: Vec::new(),
            swaps: Vec::new(),
            my_bids: Vec::new(),
            hubs: Vec::new(),
            bid_mds: String::new(),
            bid_wei: String::new(),
            bid_hours: "24".into(),
            bid_bond: "0".into(),
            ask_mds: String::new(),
            ask_wei: String::new(),
            ask_life: "4320".into(),
            ask_notice: String::new(),
            ask_mds_unit: 0,
            ask_eth_unit: 0,
            own_phrase: false,
            chan_tab: ChanTab::List,
            verify_input: String::new(),
            verify_result: None,
            logo,
            req_payee: String::new(),
            req_amount: String::new(),
            consolidate_addr: String::new(),
            defrag_max: "40".into(),
            coins_notice: String::new(),
            coin_export: None,
            import_seed: String::new(),
            import_value: String::new(),
            import_salt: String::new(),
            import_label: String::new(),
            abandon_addr: String::new(),
        }
    }

    /// Fire an action on the runtime (no-op until walletd is up).
    pub fn go(&self, ctx: &egui::Context, action: Action) {
        if let Some(h) = self.handle.clone() {
            dispatch(self.rt.handle(), h, self.msg_tx.clone(), ctx.clone(), action);
        }
    }

    pub fn reload_wallet(&self, ctx: &egui::Context) {
        for a in [
            Action::LoadBalance,
            Action::LoadCoins,
            Action::LoadAddresses,
            Action::LoadHistory,
            Action::LoadSends,
        ] {
            self.go(ctx, a);
        }
    }

    /// 0..1 sync fraction against the estimated target (None once synced).
    pub fn sync_fraction(&self) -> Option<f32> {
        let s = self.sync.as_ref()?;
        if !s.is_syncing {
            return None;
        }
        let target = s.est_target_height.max(s.height).max(1);
        Some((s.height as f32 / target as f32).clamp(0.0, 1.0))
    }

    /// Blocks applied per second over the recent sample window.
    pub fn sync_rate(&self) -> Option<f64> {
        let (t0, h0) = *self.sync_samples.front()?;
        let (t1, h1) = *self.sync_samples.back()?;
        if t1 - t0 < 8.0 || h1 <= h0 {
            return None;
        }
        Some((h1 - h0) as f64 / (t1 - t0))
    }

    pub fn sync_eta_secs(&self) -> Option<u64> {
        let s = self.sync.as_ref()?;
        let rate = self.sync_rate()?;
        if rate < 0.05 {
            return None;
        }
        let remaining = s.est_target_height.saturating_sub(s.height);
        Some((remaining as f64 / rate) as u64)
    }

    fn toast(&mut self, ctx: &egui::Context, text: String) {
        let t = ctx.input(|i| i.time);
        self.toasts.push((t, text));
    }

    fn on_msg(&mut self, ctx: &egui::Context, msg: Msg) {
        match msg {
            Msg::Ready(h) => {
                self.handle = Some(h);
                self.go(ctx, Action::LoadStatus);
                self.go(ctx, Action::LoadNodeInfo);
            }
            Msg::BootFailed(e) => self.boot_error = Some(e),
            Msg::Event(ev) => match ev {
                WalletEvent::NodeTick { status } => {
                    let t = self.t0.elapsed().as_secs_f64();
                    self.sync_samples.push_back((t, status.height));
                    while self.sync_samples.len() > 45 {
                        self.sync_samples.pop_front();
                    }
                    self.sync = Some(status);
                }
                WalletEvent::WalletChanged => self.reload_wallet(ctx),
                WalletEvent::SendUpdate { progress } => {
                    match self.sends.iter_mut().find(|s| s.id == progress.id) {
                        Some(s) => *s = progress,
                        None => self.sends.insert(0, progress),
                    }
                    self.sends.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
                }
                WalletEvent::Incoming { total_value, count, .. } => {
                    let text = format!(
                        "Received {} units ({} coin{})",
                        theme::units(total_value),
                        count,
                        if count == 1 { "" } else { "s" }
                    );
                    self.toast(ctx, text);
                }
                WalletEvent::Warning { text } => self.toast(ctx, text),
                WalletEvent::PeerAddress { peer, address } => {
                    // Drop it straight into the Send form — getting a fresh
                    // address is only useful if it lands where it is needed.
                    self.send_to = address;
                    self.addr_ok = Some(true);
                    self.addr_reason = None;
                    self.ask_pending = false;
                    self.ask_peer.clear();
                    self.toast(
                        ctx,
                        format!(
                            "{} sent a fresh address — filled in for you.",
                            crate::theme::short_hex(&peer, 8)
                        ),
                    );
                }
                WalletEvent::ChannelNotice { text } => {
                    self.toast(ctx, text);
                    if self.handle.is_some() {
                        self.go(ctx, Action::LoadChannels);
                    }
                }
            },
            Msg::Status(s) => {
                let was_unlocked = self.status.as_ref().map(|s| s.unlocked).unwrap_or(false);
                if s.unlocked && !was_unlocked {
                    self.reload_wallet(ctx);
                }
                self.onboard = if !s.exists {
                    if matches!(self.onboard, Onboard::Unlock) { Onboard::Menu } else { self.onboard }
                } else if !s.unlocked && self.mnemonic.is_empty() {
                    Onboard::Unlock
                } else {
                    self.onboard
                };
                self.status = Some(s);
            }
            Msg::Balance(b) => self.balance = Some(b),
            Msg::Coins(mut c) => {
                c.sort_by(|a, b| b.value.cmp(&a.value));
                self.coins = c;
            }
            Msg::Addresses(a) => self.addresses = a,
            Msg::History(h) => self.history = h,
            Msg::Sends(s) => self.sends = s,
            Msg::Node(n) => self.node = Some(n),
            Msg::PhraseVerified(ok) => {
                self.busy = false;
                self.verify_result = Some(ok);
                if ok {
                    // Never keep the words around after the check.
                    self.verify_input.clear();
                }
            }
            Msg::WalletCreated => {
                self.busy = false;
                self.mnemonic.clear();
                self.quiz.clear();
                self.pw.clear();
                self.pw2.clear();
                self.phrase.clear();
                self.own_phrase = false;
                self.error.clear();
                self.go(ctx, Action::LoadStatus);
                self.reload_wallet(ctx);
            }
            Msg::Mnemonic(p) => {
                self.busy = false;
                self.error.clear();
                self.mnemonic = p.split_whitespace().map(String::from).collect();
                self.own_phrase = false;
                self.phrase.clear();
                let mut idx = std::collections::BTreeSet::new();
                while idx.len() < 3.min(self.mnemonic.len()) {
                    idx.insert(rand::random::<usize>() % self.mnemonic.len());
                }
                self.quiz = idx.into_iter().map(|i| (i, String::new())).collect();
                self.onboard = Onboard::Sheet;
                self.go(ctx, Action::LoadStatus);
            }
            Msg::Restored | Msg::Unlocked => {
                self.busy = false;
                self.error.clear();
                self.pw.clear();
                self.pw2.clear();
                self.phrase.clear();
                self.go(ctx, Action::LoadStatus);
                self.reload_wallet(ctx);
            }
            Msg::Locked => {
                self.balance = None;
                self.coins.clear();
                self.addresses.clear();
                self.history.clear();
                self.sends.clear();
                self.current_addr = None;
                self.qr = None;
                self.chat.clear();
                self.coin_export = None;
                self.verify_input.clear();
                self.verify_result = None;
                self.channels.clear();
                self.chan_identity = None;
                self.chan_pay.clear();
                self.invoices.clear();
                self.last_invoice = None;
                self.hub = None;
                self.hub_draft = None;
                self.evm = None;
                self.book = None;
                self.tab = Tab::Dashboard;
                self.onboard = Onboard::Unlock;
                self.go(ctx, Action::LoadStatus);
            }
            Msg::AddressCreated(a) => {
                self.busy = false;
                self.recv_label.clear();
                self.current_addr = Some(a);
                self.go(ctx, Action::LoadAddresses);
            }
            Msg::SendStarted => {
                self.busy = false;
                self.send_to.clear();
                self.send_amount.clear();
                self.addr_ok = None;
                self.go(ctx, Action::LoadSends);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::RetryOk => self.go(ctx, Action::LoadSends),
            Msg::AddressValid { addr, ok, reason } => {
                if addr == self.send_to.trim() {
                    self.addr_ok = Some(ok);
                    self.addr_reason = reason;
                }
            }
            Msg::RescanOk => {
                self.settings_msg = "Rescanning — balance updates as the scan runs.".into();
            }
            Msg::ConsolidateStarted(c) => {
                self.busy = false;
                self.consolidate_addr.clear();
                self.coins_notice = format!(
                    "Consolidation started ({}). Watch it on the Send tab.",
                    crate::theme::short_hex(&c, 8)
                );
                self.go(ctx, Action::LoadSends);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::DefragDone(m) => {
                self.busy = false;
                self.coins_notice = m;
                self.go(ctx, Action::LoadSends);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::SendAbandoned => {
                self.go(ctx, Action::LoadSends);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::AddressAbandoned(n) => {
                self.busy = false;
                self.abandon_addr.clear();
                self.coins_notice = format!("Abandoned {n} coin record(s) from this wallet file.");
                self.reload_wallet(ctx);
            }
            Msg::CoinImported(id) => {
                self.busy = false;
                self.import_seed.clear();
                self.import_value.clear();
                self.import_salt.clear();
                self.import_label.clear();
                self.coins_notice =
                    format!("Imported coin {}.", crate::theme::short_hex(&id, 8));
                self.reload_wallet(ctx);
            }
            Msg::CoinExported(e) => {
                self.busy = false;
                self.coin_export = Some(e);
            }
            Msg::ChatSent => {
                self.chat_busy = false;
                self.chat_input.clear();
                self.go(ctx, Action::LoadChat);
            }
            Msg::Chat(v) => self.chat = v,
            Msg::ChatDict(d) => self.chat_dict = d,
            Msg::Channels(v) => self.channels = v,
            Msg::ChanIdentity(i) => self.chan_identity = Some(i),
            Msg::ChannelOpened(id) => {
                self.busy = false;
                self.chan_peer.clear();
                self.chan_amount.clear();
                self.toast(
                    ctx,
                    format!(
                        "Channel {} funding on-chain — it becomes payable once the peer acknowledges.",
                        crate::theme::short_hex(&id, 8)
                    ),
                );
                self.go(ctx, Action::LoadChannels);
                self.go(ctx, Action::LoadSends);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::InvoiceCreated(i) => {
                self.busy = false;
                self.inv_amount.clear();
                self.last_invoice = Some(i);
                self.go(ctx, Action::LoadInvoices);
            }
            Msg::Invoices(v) => self.invoices = v,
            Msg::Hub(h) => {
                // The daemon clamps some fields on save (jit_capacity and
                // max_auto_capacity are floored at MIN_CAPACITY), so what comes
                // back is not always what was sent. Dropping the draft here is
                // what makes the UI show the value that is actually in force.
                self.hub = Some(h);
                self.hub_draft = None;
            }
            Msg::HubSaved => {
                self.busy = false;
                self.go(ctx, Action::LoadHub);
            }
            Msg::AskPlaced(g) => {
                self.busy = false;
                self.ask_mds.clear();
                self.ask_wei.clear();
                self.ask_notice = format!(
                    "Order {} submitted. It is not offered to anyone until the reveal is \
                     mined — the coins and the announcement are in that same transaction. \
                     Follow it on the Send tab.",
                    crate::theme::short_hex(&g, 6)
                );
                self.go(ctx, Action::LoadMyOrders);
                self.go(ctx, Action::LoadSends);
                self.dex_last_sync = -1.0;
            }
            Msg::MyOrders(v) => self.my_orders = v,
            Msg::Swaps(v) => self.swaps = v,
            Msg::MyBids(v) => self.my_bids = v,
            Msg::Hubs(v) => self.hubs = v,
            Msg::BidPlaced(tx) => {
                self.busy = false;
                self.bid_mds.clear();
                self.bid_wei.clear();
                self.ask_notice = format!(
                    "Buy order escrowed ({}). It becomes fillable once the transaction is \
                     mined and the contract assigns it an id.",
                    crate::theme::short_hex(&tx, 8)
                );
                self.go(ctx, Action::LoadMyBids);
            }
            Msg::BidCancelled(m) => {
                self.busy = false;
                self.ask_notice = m;
                self.go(ctx, Action::LoadMyBids);
            }
            Msg::SwapStarted(tx) => {
                self.busy = false;
                self.ask_notice = format!(
                    "Escrow sent ({}). Your ETH is locked against the seller's hash — the wallet \
                     now watches for their claim and collects your MDS automatically.",
                    crate::theme::short_hex(&tx, 8)
                );
                self.go(ctx, Action::LoadSwaps);
                self.dex_tab = 3;
            }
            Msg::OrderReclaimed(m) => {
                self.busy = false;
                self.ask_notice = m;
                self.go(ctx, Action::LoadMyOrders);
                self.go(ctx, Action::LoadSends);
            }
            Msg::SwapQuote(q) => {
                self.busy = false;
                self.swap_quote = Some(q);
            }
            Msg::HistoryRepaired(m) => {
                self.busy = false;
                self.settings_msg = m;
                self.go(ctx, Action::LoadHistory);
            }
            Msg::EvmAccount(v) => self.evm = Some(v),
            Msg::OrderBook(v) => self.book = Some(v),
            Msg::OrderBookSynced => {
                self.dex_syncing = false;
                self.go(ctx, Action::LoadOrderBook);
                self.go(ctx, Action::LoadEvmAccount);
            }
            Msg::DexConfig(v) => {
                self.dex_cfg_saved = Some(v.clone());
                self.dex_cfg = Some(v);
            }
            Msg::DexConfigSaved => {
                self.busy = false;
                self.book = None;
                self.dex_last_sync = -1.0;
                self.go(ctx, Action::LoadDexConfig);
                self.go(ctx, Action::LoadEvmAccount);
            }
            Msg::AddressRequested => {
                self.busy = false;
                self.ask_pending = true;
            }
            Msg::IdentityRotated(pk) => {
                self.busy = false;
                self.chan_identity = None;
                self.toast(
                    ctx,
                    format!("New channel identity {}.", crate::theme::short_hex(&pk, 8)),
                );
                self.go(ctx, Action::ChannelIdentity);
            }
            Msg::ChannelDone => {
                self.pay_invoice_text.clear();
                self.busy = false;
                self.go(ctx, Action::LoadChannels);
                self.go(ctx, Action::LoadBalance);
            }
            Msg::Err { what, err } => {
                // Background loads fail benignly while locked; don't banner those.
                let background = matches!(
                    what,
                    "status" | "balance" | "coins" | "addresses" | "history" | "sends" | "node info" | "chat" | "channels" | "channel identity" | "invoices" | "hub" | "order book" | "base account" | "dex settings" | "swap quote" | "my orders" | "swaps" | "my bids" | "hubs"
                );
                if background {
                    tracing::debug!("{what}: {err}");
                } else {
                    self.busy = false;
                    self.error = format!("{what}: {err}");
                }
            }
        }
    }

    fn ribbon(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            match &self.sync {
                None => {
                    ui.label(RichText::new("starting node…").font(FontId::monospace(11.5)).color(theme::muted()));
                }
                Some(s) => {
                    theme::dot(ui, s.peer_count > 0);
                    kv(ui, "peers", &s.peer_count.to_string(), theme::ink());
                    let height_v = if s.is_syncing && s.est_target_height > s.height {
                        format!("{} / ~{}", theme::units(s.height), theme::units(s.est_target_height))
                    } else {
                        theme::units(s.height)
                    };
                    kv(ui, "height", &height_v, theme::ink());
                    if s.is_syncing {
                        ui.label(RichText::new("syncing…").font(FontId::monospace(11.5)).color(theme::muted()));
                    }
                    kv(ui, "mempool", &s.mempool.to_string(), theme::ink());
                    kv(ui, "coins", &theme::units(s.num_coins as u64), theme::ink());
                }
            }
        });

        fn kv(ui: &mut egui::Ui, k: &str, v: &str, vc: Color32) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                ui.label(RichText::new(k).font(FontId::monospace(11.5)).color(theme::faint()));
                ui.label(RichText::new(v).font(FontId::monospace(11.5)).color(vc));
            });
        }
    }

    fn draw_toasts(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        self.toasts.retain(|(t, _)| now - t < 7.0);
        if self.toasts.is_empty() {
            return;
        }
        egui::Area::new(egui::Id::new("toasts"))
            .anchor(Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                for (_, text) in &self.toasts {
                    egui::Frame::default()
                        .fill(theme::panel2())
                        .stroke(egui::Stroke::new(1.0, theme::border2()))
                        .corner_radius(egui::CornerRadius::same(6))
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .show(ui, |ui| ui.label(text.as_str()));
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.msg_rx.try_recv() {
            self.on_msg(ctx, msg);
        }
        // Theme: track the OS unless overridden, and re-tone when it flips.
        let os_dark = ctx.system_theme().map(|t| t == egui::Theme::Dark).unwrap_or(true);
        theme::set_os_dark(os_dark);
        if self.theme_applied != Some(theme::is_dark()) {
            theme::apply(ctx);
            self.theme_applied = Some(theme::is_dark());
        }

        // Keep relative times and the ribbon fresh even between events.
        ctx.request_repaint_after(std::time::Duration::from_secs(1));

        // The peer list lives in node info, which is otherwise only loaded
        // once — poll it while the Node tab is open so the list matches the
        // live count in the ribbon.
        if self.tab == Tab::Node && self.handle.is_some() {
            let t = self.t0.elapsed().as_secs_f64();
            if t - self.node_last_poll > 3.0 {
                self.node_last_poll = t;
                self.go(ctx, Action::LoadNodeInfo);
            }
        }

        // Poll chat while the tab is open (history lives in the node).
        if self.tab == Tab::Chat && self.handle.is_some() {
            let t = self.t0.elapsed().as_secs_f64();
            if t - self.chat_last_poll > 2.0 {
                self.chat_last_poll = t;
                self.go(ctx, Action::LoadChat);
                if self.chat_dict.is_empty() {
                    self.go(ctx, Action::LoadChatDict);
                }
            }
        }
        if self.tab == Tab::Trade && self.handle.is_some() {
            let t = self.t0.elapsed().as_secs_f64();
            // Scan both chains once when the tab is first opened. Polling alone
            // only re-reads what is already cached, so without this the book
            // stays empty until someone presses Refresh.
            // Rescan on open and then keep up: incremental scans are cheap
            // (about 30 new Base blocks a minute, and midstate is local), and
            // without this your own newly published order never shows up.
            if !self.dex_syncing && (self.dex_last_sync < 0.0 || t - self.dex_last_sync > 60.0) {
                self.dex_last_sync = t;
                self.dex_syncing = true;
                self.go(ctx, Action::SyncOrderBook);
            }
            if t - self.dex_last_poll > 5.0 {
                self.dex_last_poll = t;
                self.go(ctx, Action::LoadOrderBook);
                self.go(ctx, Action::LoadMyOrders);
                self.go(ctx, Action::LoadSwaps);
                self.go(ctx, Action::LoadMyBids);
                if self.evm.is_none() {
                    self.go(ctx, Action::LoadEvmAccount);
                }
                if self.dex_cfg.is_none() {
                    self.go(ctx, Action::LoadDexConfig);
                }
            }
        }
        if self.tab == Tab::Channels && self.handle.is_some() {
            let t = self.t0.elapsed().as_secs_f64();
            if t - self.chan_last_poll > 2.0 {
                self.chan_last_poll = t;
                self.go(ctx, Action::LoadChannels);
                if self.chan_identity.is_none() {
                    self.go(ctx, Action::ChannelIdentity);
                }
                if self.hub_draft.is_none() {
                    self.go(ctx, Action::LoadHub);
                }
                self.go(ctx, Action::LoadHubs);
                self.go(ctx, Action::LoadInvoices);
            }
        }

        egui::TopBottomPanel::top("ribbon")
            .frame(
                egui::Frame::default()
                    .fill(theme::bg())
                    .inner_margin(egui::Margin::symmetric(24, 8))
                    .stroke(egui::Stroke::new(1.0, theme::border())),
            )
            .show(ctx, |ui| self.ribbon(ui));

        if let Some(e) = self.boot_error.clone() {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add_space(80.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("The node could not start").size(16.0).strong());
                    ui.add_space(8.0);
                    ui.label(RichText::new(e).color(theme::red()));
                    theme::hint(ui, "Common cause: another midstate instance is using the same data directory.");
                });
            });
            return;
        }

        let ready = self.handle.is_some() && self.status.is_some();
        if !ready {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add_space(120.0);
                ui.vertical_centered(|ui| {
                    ui.spinner();
                    ui.add_space(10.0);
                    ui.label(RichText::new("Starting the embedded node…").color(theme::muted()));
                });
            });
            return;
        }

        let status = self.status.clone().unwrap();
        // A freshly created wallet is already unlocked, so `unlocked` alone
        // would hand control to the main UI and the recovery phrase would
        // never be drawn. Hold onboarding open until the phrase has been
        // shown AND confirmed — the quiz clears `mnemonic` on success.
        if !status.exists || !status.unlocked || !self.mnemonic.is_empty() {
            views::onboarding::show(self, ctx, &status);
            self.draw_toasts(ctx);
            return;
        }

        egui::SidePanel::left("rail")
            .exact_width(200.0)
            .resizable(false)
            .frame(
                egui::Frame::default()
                    .fill(theme::bg())
                    .inner_margin(egui::Margin::symmetric(12, 18))
                    .stroke(egui::Stroke::new(1.0, theme::border())),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(tex) = &self.logo {
                        theme::logo(ui, tex, 22.0, theme::ink());
                        ui.add_space(8.0);
                    }
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label(RichText::new("MID").font(FontId::monospace(13.0)).color(theme::muted()));
                    ui.label(RichText::new("STATE").font(theme::font_mono_medium(13.0)).color(theme::ink()));
                });
                ui.add_space(14.0);
                for t in Tab::ALL {
                    if ui.selectable_label(self.tab == t, t.name()).clicked() {
                        self.tab = t;
                        self.error.clear();
                    }
                }
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    if ui.button("Lock wallet").clicked() {
                        self.go(&ui.ctx().clone(), Action::Lock);
                    }
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(theme::bg()).inner_margin(egui::Margin::symmetric(28, 20)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.set_max_width(880.0);
                        ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                            match self.tab {
                                Tab::Dashboard => views::dashboard::show(self, ui),
                                Tab::Send => views::send::show(self, ui),
                                Tab::Receive => views::receive::show(self, ui),
                                Tab::History => views::history::show(self, ui),
                                Tab::Coins => views::coins::show(self, ui),
                                Tab::Chat => views::chat::show(self, ui),
                                Tab::Channels => views::channels::show(self, ui),
                                Tab::Trade => views::trade::show(self, ui),
                                Tab::Node => views::node::show(self, ui),
                                Tab::Settings => views::settings::show(self, ui, &status),
                            }
                        });
                    });
                });
            });

        self.draw_toasts(ctx);
    }
}

/// Where this instance keeps everything, and which ports its node binds.
///
/// Defaults match the single-instance case exactly. Overriding them is what
/// makes a second wallet possible on one machine — needed to trade with
/// yourself, and to run a hub beside a personal wallet. Ports must differ too:
/// each instance runs its own full node.
///
/// ```text
/// MIDSTATE_PROFILE=bob            # ~/.local/share/midstate-desktop-bob
/// MIDSTATE_DATA_DIR=/path/to/dir  # explicit, wins over PROFILE
/// MIDSTATE_P2P_PORT=9334
/// MIDSTATE_RPC_PORT=8546          # "none" disables the RPC listener
/// ```
struct Profile {
    base: std::path::PathBuf,
    p2p_port: u16,
    rpc_port: Option<u16>,
}

fn profile() -> Profile {
    let name = std::env::var("MIDSTATE_PROFILE").ok().filter(|s| !s.is_empty());
    let base = match std::env::var("MIDSTATE_DATA_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => {
            let dir = match &name {
                Some(n) => format!("midstate-desktop-{n}"),
                None => "midstate-desktop".to_string(),
            };
            dirs::data_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .join(dir)
        }
    };
    let p2p_port = std::env::var("MIDSTATE_P2P_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9333);
    let rpc_port = match std::env::var("MIDSTATE_RPC_PORT") {
        Ok(v) if v.eq_ignore_ascii_case("none") => None,
        Ok(v) => v.parse().ok(),
        Err(_) => Some(8545),
    };
    Profile { base, p2p_port, rpc_port }
}

async fn boot() -> anyhow::Result<WalletdHandle> {
    use midstate_walletd::{node_host, service};

    let p = profile();
    let chain_dir = p.base.join("chain");
    let wallet_path = p.base.join("wallets").join("wallet.dat");
    std::fs::create_dir_all(&chain_dir)?;
    tracing::info!(
        "profile: data {} · p2p {} · rpc {:?}",
        p.base.display(),
        p.p2p_port,
        p.rpc_port
    );

    let cfg = node_host::NodeConfig {
        data_dir: chain_dir.clone(),
        p2p_port: p.p2p_port,
        rpc_port: p.rpc_port,
        ..Default::default()
    };
    let rpc_url = cfg.rpc_port.map(|p| format!("http://127.0.0.1:{p}"));
    let node = node_host::start_node(cfg).await?;
    Ok(service::spawn(node, wallet_path, chain_dir, rpc_url))
}
