//! The egui ↔ tokio bridge. The UI thread never blocks: it pushes an
//! [`Action`], the dispatcher runs the async walletd call on the runtime,
//! and the result comes back as a [`Msg`] drained in `App::update` (with a
//! repaint request so results appear immediately).

use eframe::egui;
use midstate_walletd::api::*;
use midstate_walletd::WalletdHandle;
use std::sync::mpsc::Sender;

pub enum Action {
    LoadStatus,
    LoadBalance,
    LoadCoins,
    LoadAddresses,
    LoadHistory,
    LoadSends,
    LoadNodeInfo,
    Create { password: String },
    Restore { password: String, phrase: String },
    Unlock { password: String },
    Lock,
    NewAddress { mss: bool, label: Option<String> },
    Send { to: String, amount: u64, private: bool },
    RetrySend { id: String },
    ValidateAddress { addr: String },
    RescanFrom { height: u64 },
    Consolidate { address: String },
    Defrag { max_inputs: usize },
    AbandonSend { id: String },
    AbandonAddress { address: String },
    ImportCoin { seed: String, value: u64, salt: String, label: Option<String> },
    ExportCoin { id: String },
    ChatSend { text: String },
    LoadChat,
    LoadChatDict,
    LoadChannels,
    ChannelIdentity,
    ChannelOpen { peer: String, amount: u64, lifetime: u64 },
    ChannelPay { id: String, amount: u64 },
    ChannelClose { id: String },
    ChannelRefund { id: String },
    CreateInvoice { amount: u64 },
    PayInvoice { text: String },
    RequestInvoice { payee: String, amount: u64 },
    LoadInvoices,
    LoadHub,
    SetHub { cfg: HubView },
    RotateIdentity,
}

pub enum Msg {
    Ready(WalletdHandle),
    BootFailed(String),
    Event(WalletEvent),
    Status(WalletStatus),
    Balance(Balance),
    Coins(Vec<CoinView>),
    Addresses(Vec<AddressInfo>),
    History(Vec<HistoryView>),
    Sends(Vec<SendProgress>),
    Node(NodeInfo),
    Mnemonic(String),
    Restored,
    Unlocked,
    Locked,
    AddressCreated(AddressInfo),
    SendStarted,
    RetryOk,
    AddressValid { addr: String, ok: bool },
    RescanOk,
    ConsolidateStarted(String),
    DefragDone(String),
    SendAbandoned,
    AddressAbandoned(usize),
    CoinImported(String),
    CoinExported(CoinExport),
    ChatSent,
    Chat(Vec<ChatView>),
    ChatDict(Vec<String>),
    Channels(Vec<ChannelView>),
    ChanIdentity(IdentityView),
    ChannelOpened(String),
    ChannelDone,
    InvoiceCreated(InvoiceView),
    Invoices(Vec<InvoiceView>),
    Hub(HubView),
    HubSaved,
    IdentityRotated(String),
    /// `what` names the action for the error banner; wallet state is unchanged
    /// unless the message says otherwise.
    Err { what: &'static str, err: String },
}

pub fn dispatch(
    rt: &tokio::runtime::Handle,
    h: WalletdHandle,
    tx: Sender<Msg>,
    ctx: egui::Context,
    action: Action,
) {
    rt.spawn(async move {
        let out = |m: Msg| {
            let _ = tx.send(m);
            ctx.request_repaint();
        };
        macro_rules! run {
            ($what:literal, $fut:expr, $ok:expr) => {
                match $fut.await {
                    Ok(v) => out($ok(v)),
                    Err(e) => out(Msg::Err { what: $what, err: format!("{e:#}") }),
                }
            };
        }
        match action {
            Action::LoadStatus => run!("status", h.status(), Msg::Status),
            Action::LoadBalance => run!("balance", h.balance(), Msg::Balance),
            Action::LoadCoins => run!("coins", h.coins(), Msg::Coins),
            Action::LoadAddresses => run!("addresses", h.addresses(), Msg::Addresses),
            Action::LoadHistory => run!("history", h.history(), Msg::History),
            Action::LoadSends => run!("sends", h.sends(), Msg::Sends),
            Action::LoadNodeInfo => run!("node info", h.node_info(), Msg::Node),
            Action::Create { password } => run!("create wallet", h.create(password), Msg::Mnemonic),
            Action::Restore { password, phrase } => {
                run!("restore wallet", h.restore(password, phrase), |_| Msg::Restored)
            }
            Action::Unlock { password } => run!("unlock", h.unlock(password), |_| Msg::Unlocked),
            Action::Lock => run!("lock", h.lock(), |_| Msg::Locked),
            Action::NewAddress { mss, label } => {
                run!("new address", h.new_address(mss, label), Msg::AddressCreated)
            }
            Action::Send { to, amount, private } => {
                run!("send", h.send(to, amount, private), |_| Msg::SendStarted)
            }
            Action::RetrySend { id } => run!("retry send", h.retry_send(id), |_| Msg::RetryOk),
            Action::ValidateAddress { addr } => {
                let ok = h.validate_address(addr.clone()).await.is_ok();
                out(Msg::AddressValid { addr, ok });
            }
            Action::RescanFrom { height } => {
                run!("rescan", h.rescan_from(height), |_| Msg::RescanOk)
            }
            Action::Consolidate { address } => {
                run!("consolidate", h.consolidate(address), Msg::ConsolidateStarted)
            }
            Action::Defrag { max_inputs } => {
                run!("defrag", h.defrag(max_inputs), Msg::DefragDone)
            }
            Action::AbandonSend { id } => {
                run!("abandon send", h.abandon_send(id), |_| Msg::SendAbandoned)
            }
            Action::AbandonAddress { address } => {
                run!("abandon address", h.abandon_address(address), Msg::AddressAbandoned)
            }
            Action::ImportCoin { seed, value, salt, label } => {
                run!("import coin", h.import_coin(seed, value, salt, label), Msg::CoinImported)
            }
            Action::ExportCoin { id } => {
                run!("export coin", h.export_coin(id), Msg::CoinExported)
            }
            Action::ChatSend { text } => {
                run!("chat send", h.chat_send(text), |_| Msg::ChatSent)
            }
            Action::LoadChat => run!("chat", h.chat_history(), Msg::Chat),
            Action::LoadChatDict => run!("chat", h.chat_dictionary(), Msg::ChatDict),
            Action::LoadChannels => run!("channels", h.channels(), Msg::Channels),
            Action::ChannelIdentity => {
                run!("channel identity", h.channel_identity(), Msg::ChanIdentity)
            }
            Action::ChannelOpen { peer, amount, lifetime } => {
                run!("open channel", h.channel_open(peer, amount, lifetime), Msg::ChannelOpened)
            }
            Action::ChannelPay { id, amount } => {
                run!("channel pay", h.channel_pay(id, amount), |_| Msg::ChannelDone)
            }
            Action::ChannelClose { id } => {
                run!("close channel", h.channel_close(id), |_| Msg::ChannelDone)
            }
            Action::ChannelRefund { id } => {
                run!("refund channel", h.channel_refund(id), |_| Msg::ChannelDone)
            }
            Action::CreateInvoice { amount } => {
                run!("create invoice", h.create_invoice(amount), Msg::InvoiceCreated)
            }
            Action::PayInvoice { text } => {
                run!("pay invoice", h.pay_invoice(text), |_| Msg::ChannelDone)
            }
            Action::RequestInvoice { payee, amount } => {
                run!("request invoice", h.request_invoice(payee, amount), |_| Msg::ChannelDone)
            }
            Action::LoadInvoices => run!("invoices", h.invoices(), Msg::Invoices),
            Action::LoadHub => run!("hub", h.get_hub(), Msg::Hub),
            Action::SetHub { cfg } => run!("hub", h.set_hub(cfg), |_| Msg::HubSaved),
            Action::RotateIdentity => {
                run!("rotate identity", h.rotate_identity(), Msg::IdentityRotated)
            }
        }
    });
}
