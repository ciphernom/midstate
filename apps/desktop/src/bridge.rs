//! The egui ↔ tokio bridge. The UI thread never blocks: it pushes an
//! [`Action`], the dispatcher runs the async walletd call on the runtime,
//! and the result comes back as a [`Msg`] drained in `App::update` (with a
//! repaint request so results appear immediately).

use eframe::egui;
use midstate_walletd::api::*;
use midstate_walletd::WalletdHandle;
use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

pub enum Action {
    LoadStatus,
    LoadBalance,
    LoadCoins,
    LoadAddresses,
    LoadHistory,
    LoadSends,
    LoadNodeInfo,
    LoadMining,
    StartMining { threads: usize, backend: MinerBackend, pool_url: String },
    StopMining,
    NewPhrase,
    CheckPhrase { phrase: String },
    VerifyPhrase { phrase: String },
    Create { password: String, phrase: String },
    Restore { password: String, phrase: String },
    Unlock { password: String },
    Lock,
    NewAddress { mss: bool, label: Option<String> },
    Send { to: String, amount: u64, private: bool },
    RetrySend { id: String },
    ValidateAddress { addr: String },
    LoadNames,
    RegisterName { name: String, years: u64 },
    NameAct { name: String, act: NameAction },
    TakeName { name: String, rent: bool },
    FindName { name: String, years: u64 },
    LoadMarket,
    LoadNameDeals,
    WatchName { name: String, years: u64, max_price: Option<u64> },
    UnwatchName { name: String },
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
    RequestAddress { peer: String },
    LoadMyOrders,
    ReclaimOrder { group_id: String },
    LoadEthMarket,
    EthSell { mds: u64, wei_per_gmds: String, min: u64, life: u64 },
    EthCancelSell { id: String },
    EthBuy { order_id: String, mds: u64 },
    EthFillBid { bid_id: String },
    LoadSwaps,
    PlaceBid { mds_amount: u64, wei: String, ttl_secs: u64, fill_bond: String },
    CancelBid { bid_id: String },
    LoadMyBids,
    LoadHubs,
    RequestChannel { peer: String, capacity: u64 },
    RepairHistory,
    LoadEvmAccount,
    LoadOrderBook,
    SyncOrderBook,
    LoadDexConfig,
    SetDexConfig { cfg: DexConfigView },
    LoadBonds,
    /// `register`: ask midwimble to count it as soon as the lock confirms.
    /// `pool`: a pool's address, to hand the bond in to once it confirms.
    CreateBond { mining_key: String, value: u64, until: u64, label: String, register: bool, pool: Option<String> },
    /// What co-bonding with the pool at `url` takes and gives.
    PoolTerms { url: String },
    Unbond { id: String },
    ImportBond { text: String, label: String },
    RemoveBond { id: String },
    LoadMwbAccount,
    MwbEnable { phrase: String },
    MwbSend { to: String, amount: u64 },
    LoadMwbBook,
    MwbPlaceOrder { side: String, mds_amount: u64, mdw_amount: u64, lifetime_blocks: u64 },
    MwbCancelOrder { id: String },
    LoadMwbOrders,
    MwbTake { order_id: String },
    LoadMwbSwaps,
    LoadMwbNode,
    LoadMwbHistory,
    LoadMwbCoins,
    MwbRescan { from: u64 },
    RegisterBond { id: String },
    LoadBondRules,
    CopyCoBondFile { id: String },
    LoadRentals,
    PostOffer { days: u64, rent: u64 },
    CancelOffer { id: String },
    TakeOffer { id: String },
    LoadPlans,
    MakePayeeCode { name: String },
    CreatePlan { label: String, heirs: Vec<(String, u32)>, period_days: u64, amount: u64 },
    CheckIn { id: String },
    ClosePlan { id: String },
    ClaimPlan { id: String },
    ImportPlanCode { code: String, label: String },
    CreateStream { label: String, payee: String, amount: u64, period_blocks: u64, count: u32, delay_blocks: u64, cancellable: bool },
    CancelStream { id: String },
    CollectStream { id: String },
    AutoCollect { id: String, on: bool },
    SweepPayouts,
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
    Mining(MiningView),
    Mnemonic(String),
    WalletCreated,
    PhraseVerified(bool),
    Restored,
    Unlocked,
    Locked,
    AddressCreated(AddressInfo),
    SendStarted,
    RetryOk,
    /// `resolved`: the address a name pays.
    AddressValid { addr: String, ok: bool, reason: Option<String>, resolved: Option<String> },
    Names(Vec<NameView>),
    NameFound(NameCard),
    Market(Vec<MarketView>),
    NameDeals(Vec<SoldView>),
    NameChanged,
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
    AddressRequested,
    HistoryRepaired(String),
    EthMarket(EthMarketView),
    EthDone(String),
    MyOrders(Vec<MyOrderView>),
    OrderReclaimed(String),
    Swaps(Vec<SwapView>),
    BidPlaced(String),
    BidCancelled(String),
    MyBids(Vec<MyBidView>),
    Hubs(Vec<HubAdView>),
    EvmAccount(EvmAccountView),
    OrderBook(OrderBookView),
    OrderBookSynced,
    DexConfig(DexConfigView),
    DexConfigSaved,
    Bonds(Vec<BondView>),
    /// A bond action finished; the text says what happened.
    BondDone(String),
    /// `what` names the action for the error banner; wallet state is unchanged
    /// unless the message says otherwise.
    Err { what: &'static str, err: String },
    MwbAccount(MwbAccountView),
    MwbEnabled,
    MwbSent(String),
    MwbBook(Vec<MwbBookEntryView>),
    MwbOrderPlaced(String),
    MwbOrderCancelled,
    MwbOrders(Vec<MwbOrderView>),
    MwbTaken(String),
    MwbSwaps(Vec<MwbSwapView>),
    MwbNode(Option<MwbNodeInfo>),
    MwbHistory(Vec<MwbHistoryView>),
    MwbCoins(Vec<MwbCoinView>),
    MwbRescanOk,
    BondRules(BondRulesView),
    /// The bond rental market, this wallet's offers and rented bonds.
    Rentals(RentalsView),
    /// A rental action finished; the text says what happened.
    RentalDone(String),
    /// A bond file for a pool operator, to put on the clipboard.
    /// A bond's id and its co-bond file.
    CoBondFile(String, String),
    /// A pool's address, and its co-bond terms or why there are none.
    PoolTerms(String, Result<midstate_walletd::co_bond::PoolTerms, String>),
    Plans(midstate_walletd::api::PlansView),
    /// A plan or stream action finished; the text says what happened.
    PlansDone(String),
    PayeeCode(String),
    /// A stream started: the code for its recipient.
    StreamCode(String),
}

/// Reads (`Load*` actions) on their way to walletd, by kind.
///
/// # Reasoning
/// The interface polls: the Trade tab four reads every three seconds, the
/// ribbon and the dashboard more. While walletd is busy (a commit's proof of
/// work, a long scan) the polls kept coming, each a new task queued behind
/// the last, and once walletd was free it answered every stale copy in turn
/// before getting back to the swap it was meant to be watching, as the
/// interface drowned in their answers. A read already in flight is not sent
/// again; it is marked, and sent once more when the first lands, so the last
/// answer always postdates the last request. At most two of a kind exist at
/// once, however long walletd takes.
#[derive(Clone, Default)]
pub struct Reads(Arc<Mutex<HashMap<std::mem::Discriminant<Action>, bool>>>);

impl Reads {
    /// Whether to send `action` now: false when a read of its kind is
    /// already in flight, which is then marked to go once more.
    fn claim(&self, action: &Action) -> bool {
        let key = std::mem::discriminant(action);
        let mut reads = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match reads.get_mut(&key) {
            Some(again) => {
                *again = true;
                false
            }
            None => {
                reads.insert(key, false);
                true
            }
        }
    }

    /// A read of this kind has landed: whether to send it once more.
    fn finish(&self, key: std::mem::Discriminant<Action>) -> bool {
        let mut reads = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match reads.get_mut(&key) {
            Some(again) if *again => {
                *again = false;
                true
            }
            _ => {
                reads.remove(&key);
                false
            }
        }
    }
}

impl Action {
    /// A copy of this action if it is a read (`Load*`): those take nothing
    /// and change nothing, so copies of one in flight can be merged.
    fn read_copy(&self) -> Option<Action> {
        use Action::*;
        Some(match self {
            LoadStatus => LoadStatus,
            LoadBalance => LoadBalance,
            LoadCoins => LoadCoins,
            LoadAddresses => LoadAddresses,
            LoadNames => LoadNames,
            LoadMarket => LoadMarket,
            LoadNameDeals => LoadNameDeals,
            LoadHistory => LoadHistory,
            LoadSends => LoadSends,
            LoadNodeInfo => LoadNodeInfo,
            LoadMining => LoadMining,
            LoadChat => LoadChat,
            LoadChatDict => LoadChatDict,
            LoadChannels => LoadChannels,
            LoadInvoices => LoadInvoices,
            LoadHub => LoadHub,
            LoadMyOrders => LoadMyOrders,
            LoadSwaps => LoadSwaps,
            LoadMyBids => LoadMyBids,
            LoadHubs => LoadHubs,
            LoadEvmAccount => LoadEvmAccount,
            LoadOrderBook => LoadOrderBook,
            LoadDexConfig => LoadDexConfig,
            LoadBonds => LoadBonds,
            LoadMwbAccount => LoadMwbAccount,
            LoadMwbBook => LoadMwbBook,
            LoadMwbOrders => LoadMwbOrders,
            LoadMwbSwaps => LoadMwbSwaps,
            LoadMwbNode => LoadMwbNode,
            LoadMwbHistory => LoadMwbHistory,
            LoadMwbCoins => LoadMwbCoins,
            LoadBondRules => LoadBondRules,
            LoadRentals => LoadRentals,
            LoadPlans => LoadPlans,
            _ => return None,
        })
    }
}

pub fn dispatch(
    rt: &tokio::runtime::Handle,
    h: WalletdHandle,
    tx: Sender<Msg>,
    ctx: egui::Context,
    action: Action,
    reads: &Reads,
) {
    let Some(copy) = action.read_copy() else {
        rt.spawn(perform(h, tx, ctx, action));
        return;
    };
    if !reads.claim(&action) {
        return;
    }
    let slot = Slot { reads: reads.clone(), key: std::mem::discriminant(&action), held: true };
    rt.spawn(async move {
        let mut slot = slot;
        perform(h.clone(), tx.clone(), ctx.clone(), action).await;
        while slot.reads.finish(slot.key) {
            perform(h.clone(), tx.clone(), ctx.clone(), copy.read_copy().expect("a read")).await;
        }
        slot.held = false;
    });
}

/// A read's place in [`Reads`] while it is in flight. Freed if the task
/// ends without finishing (a panic, the runtime shutting down), so that kind
/// of read is never left blocked.
struct Slot {
    reads: Reads,
    key: std::mem::Discriminant<Action>,
    held: bool,
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.held {
            self.reads.0.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.key);
        }
    }
}

/// One action against walletd, its answer sent to the interface.
fn perform(
    h: WalletdHandle,
    tx: Sender<Msg>,
    ctx: egui::Context,
    action: Action,
) -> impl std::future::Future<Output = ()> + Send {
    async move {
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
            Action::LoadNames => run!("names", h.names(), Msg::Names),
            Action::RegisterName { name, years } => run!("register name", h.register_name(name, years), |_| Msg::NameChanged),
            Action::NameAct { name, act } => run!("name", h.name_action(name, act), |_| Msg::NameChanged),
            Action::TakeName { name, rent } => run!("take name", h.take_name(name, rent), |_| Msg::NameChanged),
            Action::FindName { name, years } => run!("find name", h.find_name(name, years), Msg::NameFound),
            Action::LoadMarket => run!("market", h.name_market(), Msg::Market),
            Action::LoadNameDeals => run!("name deals", h.name_deals(), Msg::NameDeals),
            Action::WatchName { name, years, max_price } => {
                run!("watch name", h.watch_name(name, years, max_price), |_| Msg::NameChanged)
            }
            Action::UnwatchName { name } => run!("unwatch name", h.unwatch_name(name), |_| Msg::NameChanged),
            Action::LoadHistory => run!("history", h.history(), Msg::History),
            Action::LoadSends => run!("sends", h.sends(), Msg::Sends),
            Action::LoadNodeInfo => run!("node info", h.node_info(), Msg::Node),
            Action::LoadMining => run!("mining status", h.mining_status(), Msg::Mining),
            Action::StartMining { threads, backend, pool_url } => {
                // Report the resulting status either way: a refused start needs
                // to clear the button's optimistic state, not just toast.
                match h.start_mining(threads, backend, pool_url).await {
                    Ok(()) => run!("mining status", h.mining_status(), Msg::Mining),
                    Err(e) => {
                        out(Msg::Err { what: "start mining", err: format!("{e:#}") });
                        run!("mining status", h.mining_status(), Msg::Mining)
                    }
                }
            }
            Action::StopMining => {
                let _ = h.stop_mining().await;
                run!("mining status", h.mining_status(), Msg::Mining)
            }
            Action::NewPhrase => run!("new phrase", h.new_phrase(), Msg::Mnemonic),
            Action::CheckPhrase { phrase } => {
                run!("check phrase", h.check_phrase(phrase), Msg::Mnemonic)
            }
            Action::VerifyPhrase { phrase } => {
                run!("verify phrase", h.verify_phrase(phrase), Msg::PhraseVerified)
            }
            Action::Create { password, phrase } => {
                run!("create wallet", h.create(password, phrase), |_| Msg::WalletCreated)
            }
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
                // Keep the reason: "checksum mismatch" and "this key is burned"
                // need very different responses from the person.
                let r = h.validate_address(addr.clone()).await;
                let (resolved, reason) = match r {
                    Ok(a) => (a, None),
                    Err(e) => (None, Some(e.to_string())),
                };
                out(Msg::AddressValid { addr, ok: reason.is_none(), reason, resolved });
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
            Action::RequestAddress { peer } => {
                run!("request address", h.request_address(peer), |_| Msg::AddressRequested)
            }
            Action::LoadMyOrders => run!("my orders", h.my_orders(), Msg::MyOrders),
            Action::LoadEthMarket => run!("ETH market", h.eth_market(), Msg::EthMarket),
            Action::EthSell { mds, wei_per_gmds, min, life } => {
                run!("post sell order", h.eth_sell(mds, wei_per_gmds, min, life), |_| Msg::EthDone(
                    "Sell order posted. Buyers see it while this app is open; each sale locks only what that buyer asked for.".into()
                ))
            }
            Action::EthCancelSell { id } => {
                run!("cancel sell order", h.eth_cancel_sell(id), |_| Msg::EthDone("Sell order cancelled. Sales already under way finish.".into()))
            }
            Action::EthBuy { order_id, mds } => {
                run!("buy", h.eth_buy(order_id, mds), |_| Msg::EthDone(
                    "Asked the seller. Their wallet locks the MDS for you; yours pays only after checking it on chain.".into()
                ))
            }
            Action::EthFillBid { bid_id } => {
                run!("sell into buy order", h.eth_fill_bid(bid_id), |_| Msg::EthDone(
                    "Reserving the buy order on Base; your MDS is locked for that buyer once the reservation is in.".into()
                ))
            }
            Action::LoadSwaps => run!("swaps", h.swaps(), Msg::Swaps),
            Action::PlaceBid { mds_amount, wei, ttl_secs, fill_bond } => {
                run!("place bid", h.place_bid(mds_amount, wei, ttl_secs, fill_bond), Msg::BidPlaced)
            }
            Action::CancelBid { bid_id } => {
                run!("cancel bid", h.cancel_bid(bid_id), Msg::BidCancelled)
            }
            Action::LoadMyBids => run!("my bids", h.my_bids(), Msg::MyBids),
            Action::LoadHubs => run!("hubs", h.hubs(), Msg::Hubs),
            Action::RequestChannel { peer, capacity } => {
                run!("request channel", h.request_channel(peer, capacity), |_| Msg::ChannelDone)
            }
            Action::ReclaimOrder { group_id } => {
                run!("reclaim order", h.reclaim_order(group_id), Msg::OrderReclaimed)
            }
            Action::RepairHistory => {
                run!("repair history", h.repair_history(), Msg::HistoryRepaired)
            }
            Action::LoadEvmAccount => run!("base account", h.evm_account(), Msg::EvmAccount),
            Action::LoadOrderBook => run!("order book", h.order_book(), Msg::OrderBook),
            Action::SyncOrderBook => {
                run!("order book sync", h.sync_order_book(), |_| Msg::OrderBookSynced)
            }
            Action::LoadDexConfig => run!("dex settings", h.dex_config(), Msg::DexConfig),
            Action::SetDexConfig { cfg } => {
                run!("dex settings", h.set_dex_config(cfg), |_| Msg::DexConfigSaved)
            }
            Action::LoadBonds => run!("bonds", h.bonds(), Msg::Bonds),
            Action::PoolTerms { url } => {
                let r = h.pool_terms(url.clone()).await.map_err(|e| format!("{e:#}"));
                out(Msg::PoolTerms(url, r))
            }
            Action::CreateBond { mining_key, value, until, label, register, pool } => {
                match h.create_bond(mining_key, value, until, label, pool).await {
                    Ok(id) => {
                        // Registration waits for the lock to confirm by
                        // itself. If asking fails, the bond's card offers
                        // Register instead.
                        if register {
                            if let Err(e) = h.register_bond(id).await {
                                tracing::warn!("registering a new bond: {e:#}");
                            }
                        }
                        out(Msg::BondDone("Locking: follow it on its card below.".into()))
                    }
                    Err(e) => out(Msg::Err { what: "lock bond", err: format!("{e:#}") }),
                }
            }
            Action::Unbond { id } => run!("unbond", h.unbond(id), |_| {
                Msg::BondDone("Unbonding: follow it on the Send tab.".into())
            }),
            Action::ImportBond { text, label } => {
                run!("import bond", h.import_bond(text, label), Msg::BondDone)
            }
            Action::RemoveBond { id } => {
                run!("forget bond", h.remove_bond(id), |_| Msg::BondDone("Bond forgotten.".into()))
            }
            Action::LoadMwbAccount => run!("mdw account", h.mwb_account(), Msg::MwbAccount),
            Action::MwbEnable { phrase } => run!("add MDW account", h.mwb_enable(phrase), |_| Msg::MwbEnabled),
            Action::MwbSend { to, amount } => run!("send MDW", h.mwb_send(to, amount), Msg::MwbSent),
            Action::LoadMwbBook => run!("mdw book", h.mwb_book(), Msg::MwbBook),
            Action::MwbPlaceOrder { side, mds_amount, mdw_amount, lifetime_blocks } => {
                run!("place MDW order", h.mwb_place_order(side, mds_amount, mdw_amount, lifetime_blocks), Msg::MwbOrderPlaced)
            }
            Action::MwbCancelOrder { id } => run!("cancel MDW order", h.mwb_cancel_order(id), |_| Msg::MwbOrderCancelled),
            Action::LoadMwbOrders => run!("mdw orders", h.mwb_orders(), Msg::MwbOrders),
            Action::MwbTake { order_id } => run!("take MDW order", h.mwb_take(order_id), Msg::MwbTaken),
            Action::LoadMwbSwaps => run!("mdw swaps", h.mwb_swaps(), Msg::MwbSwaps),
            Action::LoadMwbNode => run!("mdw node", h.mwb_node_info(), Msg::MwbNode),
            Action::LoadMwbHistory => run!("mdw history", h.mwb_history(), Msg::MwbHistory),
            Action::LoadMwbCoins => run!("mdw coins", h.mwb_coins(), Msg::MwbCoins),
            Action::MwbRescan { from } => run!("rescan MDW", h.mwb_rescan(from), |_| Msg::MwbRescanOk),
            Action::RegisterBond { id } => run!("register bond", h.register_bond(id), Msg::BondDone),
            Action::LoadBondRules => run!("bond rules", h.bond_rules(), Msg::BondRules),
            Action::LoadRentals => run!("rentals", h.rentals(), Msg::Rentals),
            Action::PostOffer { days, rent } => run!("offer a bond", h.post_offer(days, rent), |_| {
                Msg::RentalDone(
                    "Offered: the send that locks the coin is on the Send tab. Once it confirms, the offer is \
                     listed, and anyone can rent it while this wallet is closed."
                        .into(),
                )
            }),
            Action::CancelOffer { id } => run!("cancel offer", h.cancel_offer(id), |_| {
                Msg::RentalDone("Cancelling: the coin comes back once the send on the Send tab confirms.".into())
            }),
            Action::TakeOffer { id } => run!("rent bond", h.take_offer(id), |_| {
                Msg::RentalDone(
                    "Renting: the send is on the Send tab. Once it confirms, the wallet registers the bond, and \
                     about a day later your miner signs with it."
                        .into(),
                )
            }),
            Action::CopyCoBondFile { id } => {
                run!("co-bond file", h.co_bond_file(id.clone()), |text| Msg::CoBondFile(id, text))
            }
            Action::LoadPlans => run!("plans", h.plans(), Msg::Plans),
            Action::MakePayeeCode { name } => run!("make payee code", h.payee_code(name), Msg::PayeeCode),
            Action::CreatePlan { label, heirs, period_days, amount } => {
                run!("make plan", h.create_plan(label, heirs, period_days, amount), |_| {
                    Msg::PlansDone(
                        "Plan made: the send that funds it is on the Send tab. Once it confirms, give each heir the \
                         plan's code."
                            .into(),
                    )
                })
            }
            Action::CheckIn { id } => run!("check in", h.check_in(id), |_| {
                Msg::PlansDone("Checking in: follow it on the Send tab.".into())
            }),
            Action::ClosePlan { id } => run!("close plan", h.close_plan(id), |_| {
                Msg::PlansDone("Closing: the coins come back once the send on the Send tab confirms.".into())
            }),
            Action::ClaimPlan { id } => run!("claim plan", h.claim_plan(id), |_| {
                Msg::PlansDone("Paying the heirs: follow it on the Send tab. Your share comes in by itself.".into())
            }),
            Action::ImportPlanCode { code, label } => run!("follow code", h.import_plan_code(code, label), |_| {
                Msg::PlansDone("Following it: the wallet finds its coins on chain.".into())
            }),
            Action::CreateStream { label, payee, amount, period_blocks, count, delay_blocks, cancellable } => {
                run!(
                    "start stream",
                    h.create_stream(label, payee, amount, period_blocks, count, delay_blocks, cancellable),
                    Msg::StreamCode
                )
            }
            Action::CancelStream { id } => run!("cancel stream", h.cancel_stream(id), |_| {
                Msg::PlansDone("Taking back the payments not yet due: follow it on the Send tab.".into())
            }),
            Action::CollectStream { id } => run!("collect", h.collect_stream(id), |_| {
                Msg::PlansDone("Collecting: follow it on the Send tab.".into())
            }),
            Action::AutoCollect { id, on } => run!("auto collect", h.set_auto_collect(id, on), move |_| {
                Msg::PlansDone(if on { "Collecting payments as they fall due." } else { "Not collecting payments by itself." }.into())
            }),
            Action::SweepPayouts => run!("sweep", h.sweep_payouts(), |_| {
                Msg::PlansDone("Moving what you were paid into your balance: follow it on the Send tab.".into())
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read already in flight is not sent again: it is sent once more
    /// when it lands, however many times it was asked for meanwhile.
    #[test]
    fn a_read_in_flight_goes_once_more_not_once_per_poll() {
        let reads = Reads::default();
        assert!(reads.claim(&Action::LoadMwbSwaps), "the first goes");
        for _ in 0..5 {
            assert!(!reads.claim(&Action::LoadMwbSwaps), "copies wait");
        }
        assert!(reads.claim(&Action::LoadBalance), "other reads are not held up");
        let key = std::mem::discriminant(&Action::LoadMwbSwaps);
        assert!(reads.finish(key), "once more, for the copies asked for meanwhile");
        assert!(!reads.finish(key), "and then it is done");
        assert!(reads.claim(&Action::LoadMwbSwaps), "free again");
    }

    /// A read whose task dies frees its kind rather than blocking it.
    #[test]
    fn a_read_that_never_lands_does_not_block_its_kind() {
        let reads = Reads::default();
        assert!(reads.claim(&Action::LoadBalance));
        let key = std::mem::discriminant(&Action::LoadBalance);
        drop(Slot { reads: reads.clone(), key, held: true });
        assert!(reads.claim(&Action::LoadBalance), "freed");
        drop(Slot { reads: reads.clone(), key, held: false });
        assert!(!reads.claim(&Action::LoadBalance), "a finished slot leaves the next read alone");
    }

    #[test]
    fn only_reads_are_merged() {
        assert!(Action::LoadMwbSwaps.read_copy().is_some());
        assert!(Action::MwbTake { order_id: "o".into() }.read_copy().is_none());
        assert!(Action::Send { to: "a".into(), amount: 1, private: false }.read_copy().is_none());
    }
}
