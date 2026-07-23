//! The walletd actor. One task owns the open `Wallet`; everything else —
//! Tauri commands, monitors, the scan tick — talks to it via messages.

use crate::api::*;
use crate::base::{BaseClient, ChainConfig};
use crate::channels::{self, ChanStatus, ChannelBook, ChannelRecord, PendingOpen, Role};
use crate::dex::OrderBook;
use crate::ledger::{Ledger, SendRecord};
use crate::swap::{self, Rail, Side};
use crate::swapbook::{Phase, Swap, SwapBook};
use crate::sendplan::{self, SendPlan};
use midstate::core::channel as qb;
use anyhow::{anyhow, bail, Context, Result};
use midstate::core::encode_address_with_checksum;
use midstate::node::NodeHandle;
use midstate::wallet::Wallet;
use midstate::core::Transaction;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{broadcast, mpsc, oneshot};

const TICK: Duration = Duration::from_secs(2);
/// How long a commit monitor waits for the commitment to enter chain state
/// before declaring the send stalled (user can retry; pending is preserved).
const COMMIT_PATIENCE: Duration = Duration::from_secs(15 * 60);
/// Blocks scanned per tick during catch-up rescans (keeps the actor responsive).
const SCAN_CHUNK: u64 = 2_000;
/// MSS leaf fast-forward margin when the chain has seen more signatures than
/// the local wallet (mirrors the CLI's STRICT SAFETY check).
const MSS_SAFETY_MARGIN: u64 = 20;
/// MSS tree height for new receive addresses: 2^10 = 1024 signatures.
const DEFAULT_MSS_HEIGHT: u32 = 10;
/// HD indices derived up-front on restore before scanning (upstream floor
/// semantics; gap-limit extension beyond the floor is a v1.x follow-up).
const RESTORE_KEY_FLOOR: u64 = 1_000;

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ── Commands ────────────────────────────────────────────────────────────────

enum Cmd {
    Status(oneshot::Sender<WalletStatus>),
    Create { password: String, phrase: String, resp: oneshot::Sender<Result<()>> },
    NewPhrase(oneshot::Sender<Result<String>>),
    CheckPhrase { phrase: String, resp: oneshot::Sender<Result<String>> },
    VerifyPhrase { phrase: String, resp: oneshot::Sender<Result<bool>> },
    Restore { password: String, phrase: String, resp: oneshot::Sender<Result<()>> },
    Unlock { password: String, resp: oneshot::Sender<Result<()>> },
    Lock(oneshot::Sender<Result<()>>),
    NewAddress { mss: bool, label: Option<String>, resp: oneshot::Sender<Result<AddressInfo>> },
    Addresses(oneshot::Sender<Result<Vec<AddressInfo>>>),
    Balance(oneshot::Sender<Result<Balance>>),
    Coins(oneshot::Sender<Result<Vec<CoinView>>>),
    History(oneshot::Sender<Result<Vec<HistoryView>>>),
    Sends(oneshot::Sender<Vec<SendProgress>>),
    Send { to: String, amount: u64, private: bool, resp: oneshot::Sender<Result<String>> },
    RetrySend { id: String, resp: oneshot::Sender<Result<()>> },
    ValidateAddress { addr: String, resp: oneshot::Sender<Result<()>> },
    SyncStatus(oneshot::Sender<SyncStatus>),
    NodeInfo(oneshot::Sender<NodeInfo>),
    RescanFrom { height: u64, resp: oneshot::Sender<Result<()>> },
    Consolidate { address: String, resp: oneshot::Sender<Result<String>> },
    Defrag { max_inputs: usize, resp: oneshot::Sender<Result<String>> },
    AbandonSend { id: String, resp: oneshot::Sender<Result<()>> },
    AbandonAddress { address: String, resp: oneshot::Sender<Result<usize>> },
    ImportCoin { seed: String, value: u64, salt: String, label: Option<String>, resp: oneshot::Sender<Result<String>> },
    ExportCoin { id: String, resp: oneshot::Sender<Result<CoinExport>> },
    ChatSend { text: String, resp: oneshot::Sender<Result<()>> },
    ChatHistory(oneshot::Sender<Vec<ChatView>>),
    ChatDict(oneshot::Sender<Vec<String>>),
    ChannelIdentity(oneshot::Sender<Result<IdentityView>>),
    Channels(oneshot::Sender<Vec<ChannelView>>),
    ChannelOpen { peer: String, amount: u64, lifetime: u64, resp: oneshot::Sender<Result<String>> },
    ChannelPay { id: String, amount: u64, resp: oneshot::Sender<Result<()>> },
    ChannelClose { id: String, resp: oneshot::Sender<Result<()>> },
    ChannelRefund { id: String, resp: oneshot::Sender<Result<()>> },
    CreateInvoice { amount: u64, resp: oneshot::Sender<Result<InvoiceView>> },
    PayInvoice { text: String, resp: oneshot::Sender<Result<()>> },
    RequestInvoice { payee: String, amount: u64, resp: oneshot::Sender<Result<()>> },
    Invoices(oneshot::Sender<Vec<InvoiceView>>),
    GetHub(oneshot::Sender<HubView>),
    SetHub { cfg: HubView, resp: oneshot::Sender<Result<()>> },
    RotateIdentity(oneshot::Sender<Result<String>>),
    RequestAddress { peer: String, resp: oneshot::Sender<Result<()>> },
    RepairHistory(oneshot::Sender<Result<String>>),
    RequestChannel { peer: String, capacity: u64, resp: oneshot::Sender<Result<()>> },
    PlaceAsk {
        mds_amount: u64,
        wei_amount: String,
        lifetime_blocks: u64,
        resp: oneshot::Sender<Result<String>>,
    },
    MyOrders(oneshot::Sender<Vec<MyOrderView>>),
    ReclaimOrder { group_id: String, resp: oneshot::Sender<Result<String>> },
    TakeAsk { group_id: String, unit: usize, resp: oneshot::Sender<Result<String>> },
    PlaceBid {
        mds_amount: u64,
        wei: String,
        ttl_secs: u64,
        fill_bond: String,
        resp: oneshot::Sender<Result<String>>,
    },
    CancelBid { bid_id: String, resp: oneshot::Sender<Result<String>> },
    MyBids(oneshot::Sender<Vec<MyBidView>>),
    Swaps(oneshot::Sender<Vec<SwapView>>),
    SwapQuote {
        side: String,
        rail: String,
        mds_amount: u64,
        wei_amount: String,
        peer_mds_pk: String,
        eth_refund_secs: u64,
        resp: oneshot::Sender<Result<SwapQuoteView>>,
    },
    EvmAccount(oneshot::Sender<EvmAccountView>),
    OrderBook(oneshot::Sender<OrderBookView>),
    SyncOrderBook(oneshot::Sender<Result<()>>),
    GetDexConfig(oneshot::Sender<DexConfigView>),
    SetDexConfig { cfg: DexConfigView, resp: oneshot::Sender<Result<()>> },
    Internal(Internal),
}

/// Messages from background monitors back into the actor.
enum Internal {
    TryReveal([u8; 32]),
    CommitStalled([u8; 32]),
    RevealConfirmed([u8; 32]),
    Tick,
}

// ── Public handle ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct WalletdHandle {
    tx: mpsc::Sender<Cmd>,
    events: broadcast::Sender<WalletEvent>,
}

macro_rules! ask {
    ($self:ident, $variant:ident { $($f:ident : $v:expr),* }) => {{
        let (resp, rx) = oneshot::channel();
        $self.tx.send(Cmd::$variant { $($f: $v,)* resp }).await
            .map_err(|_| anyhow!("wallet service is not running"))?;
        rx.await.map_err(|_| anyhow!("wallet service dropped the request"))?
    }};
    ($self:ident, $variant:ident) => {{
        let (resp, rx) = oneshot::channel();
        $self.tx.send(Cmd::$variant(resp)).await
            .map_err(|_| anyhow!("wallet service is not running"))?;
        rx.await.map_err(|_| anyhow!("wallet service dropped the request"))?
    }};
}

impl WalletdHandle {
    pub fn subscribe(&self) -> broadcast::Receiver<WalletEvent> {
        self.events.subscribe()
    }
    pub async fn status(&self) -> Result<WalletStatus> {
        Ok(ask!(self, Status))
    }
    pub async fn create(&self, password: String, phrase: String) -> Result<()> {
        ask!(self, Create { password: password, phrase: phrase })
    }
    /// Mint a fresh BIP39 phrase. Nothing is written to disk — the wallet is
    /// only created once the user has confirmed the phrase and set a password.
    pub async fn new_phrase(&self) -> Result<String> {
        ask!(self, NewPhrase)
    }
    /// Validate a user-supplied phrase, echoing it back on success.
    pub async fn check_phrase(&self, phrase: String) -> Result<String> {
        ask!(self, CheckPhrase { phrase: phrase })
    }
    /// Does this phrase actually belong to the open wallet? Lets someone
    /// prove their written backup is correct without the wallet ever being
    /// able to reveal the words.
    pub async fn verify_phrase(&self, phrase: String) -> Result<bool> {
        ask!(self, VerifyPhrase { phrase: phrase })
    }
    pub async fn restore(&self, password: String, phrase: String) -> Result<()> {
        ask!(self, Restore { password: password, phrase: phrase })
    }
    pub async fn unlock(&self, password: String) -> Result<()> {
        ask!(self, Unlock { password: password })
    }
    pub async fn lock(&self) -> Result<()> {
        ask!(self, Lock)
    }
    pub async fn new_address(&self, mss: bool, label: Option<String>) -> Result<AddressInfo> {
        ask!(self, NewAddress { mss: mss, label: label })
    }
    pub async fn addresses(&self) -> Result<Vec<AddressInfo>> {
        ask!(self, Addresses)
    }
    pub async fn balance(&self) -> Result<Balance> {
        ask!(self, Balance)
    }
    pub async fn coins(&self) -> Result<Vec<CoinView>> {
        ask!(self, Coins)
    }
    pub async fn history(&self) -> Result<Vec<HistoryView>> {
        ask!(self, History)
    }
    pub async fn sends(&self) -> Result<Vec<SendProgress>> {
        Ok(ask!(self, Sends))
    }
    pub async fn send(&self, to: String, amount: u64, private: bool) -> Result<String> {
        ask!(self, Send { to: to, amount: amount, private: private })
    }
    pub async fn retry_send(&self, id: String) -> Result<()> {
        ask!(self, RetrySend { id: id })
    }
    pub async fn validate_address(&self, addr: String) -> Result<()> {
        ask!(self, ValidateAddress { addr: addr })
    }
    pub async fn sync_status(&self) -> Result<SyncStatus> {
        Ok(ask!(self, SyncStatus))
    }
    pub async fn node_info(&self) -> Result<NodeInfo> {
        Ok(ask!(self, NodeInfo))
    }
    pub async fn rescan_from(&self, height: u64) -> Result<()> {
        ask!(self, RescanFrom { height: height })
    }
    pub async fn consolidate(&self, address: String) -> Result<String> {
        ask!(self, Consolidate { address: address })
    }
    pub async fn defrag(&self, max_inputs: usize) -> Result<String> {
        ask!(self, Defrag { max_inputs: max_inputs })
    }
    pub async fn abandon_send(&self, id: String) -> Result<()> {
        ask!(self, AbandonSend { id: id })
    }
    pub async fn abandon_address(&self, address: String) -> Result<usize> {
        ask!(self, AbandonAddress { address: address })
    }
    pub async fn import_coin(&self, seed: String, value: u64, salt: String, label: Option<String>) -> Result<String> {
        ask!(self, ImportCoin { seed: seed, value: value, salt: salt, label: label })
    }
    pub async fn export_coin(&self, id: String) -> Result<CoinExport> {
        ask!(self, ExportCoin { id: id })
    }
    pub async fn chat_send(&self, text: String) -> Result<()> {
        ask!(self, ChatSend { text: text })
    }
    pub async fn chat_history(&self) -> Result<Vec<ChatView>> {
        Ok(ask!(self, ChatHistory))
    }
    pub async fn chat_dictionary(&self) -> Result<Vec<String>> {
        Ok(ask!(self, ChatDict))
    }
    pub async fn channel_identity(&self) -> Result<IdentityView> {
        ask!(self, ChannelIdentity)
    }
    pub async fn channels(&self) -> Result<Vec<ChannelView>> {
        Ok(ask!(self, Channels))
    }
    pub async fn channel_open(&self, peer: String, amount: u64, lifetime: u64) -> Result<String> {
        ask!(self, ChannelOpen { peer: peer, amount: amount, lifetime: lifetime })
    }
    pub async fn channel_pay(&self, id: String, amount: u64) -> Result<()> {
        ask!(self, ChannelPay { id: id, amount: amount })
    }
    pub async fn channel_close(&self, id: String) -> Result<()> {
        ask!(self, ChannelClose { id: id })
    }
    pub async fn channel_refund(&self, id: String) -> Result<()> {
        ask!(self, ChannelRefund { id: id })
    }
    pub async fn create_invoice(&self, amount: u64) -> Result<InvoiceView> {
        ask!(self, CreateInvoice { amount: amount })
    }
    pub async fn pay_invoice(&self, text: String) -> Result<()> {
        ask!(self, PayInvoice { text: text })
    }
    pub async fn request_invoice(&self, payee: String, amount: u64) -> Result<()> {
        ask!(self, RequestInvoice { payee: payee, amount: amount })
    }
    pub async fn invoices(&self) -> Result<Vec<InvoiceView>> {
        Ok(ask!(self, Invoices))
    }
    pub async fn get_hub(&self) -> Result<HubView> {
        Ok(ask!(self, GetHub))
    }
    pub async fn set_hub(&self, cfg: HubView) -> Result<()> {
        ask!(self, SetHub { cfg: cfg })
    }
    pub async fn rotate_identity(&self) -> Result<String> {
        ask!(self, RotateIdentity)
    }
    /// Ask a peer over the chat bus for a fresh receiving address.
    pub async fn request_address(&self, peer: String) -> Result<()> {
        ask!(self, RequestAddress { peer: peer })
    }
    /// Publish a sell order: lock MDS behind limit-order covenants and
    /// announce them on-chain, in a single transaction.
    pub async fn place_ask(
        &self,
        mds_amount: u64,
        wei_amount: String,
        lifetime_blocks: u64,
    ) -> Result<String> {
        ask!(self, PlaceAsk {
            mds_amount: mds_amount,
            wei_amount: wei_amount,
            lifetime_blocks: lifetime_blocks
        })
    }
    pub async fn my_orders(&self) -> Result<Vec<MyOrderView>> {
        Ok(ask!(self, MyOrders))
    }
    /// Take one unit of a published sell order: escrow ETH against its hash.
    pub async fn take_ask(&self, group_id: String, unit: usize) -> Result<String> {
        ask!(self, TakeAsk { group_id: group_id, unit: unit })
    }
    /// Escrow ETH as a resting buy order for MDS.
    pub async fn place_bid(
        &self,
        mds_amount: u64,
        wei: String,
        ttl_secs: u64,
        fill_bond: String,
    ) -> Result<String> {
        ask!(self, PlaceBid {
            mds_amount: mds_amount,
            wei: wei,
            ttl_secs: ttl_secs,
            fill_bond: fill_bond
        })
    }
    pub async fn cancel_bid(&self, bid_id: String) -> Result<String> {
        ask!(self, CancelBid { bid_id: bid_id })
    }
    pub async fn my_bids(&self) -> Result<Vec<MyBidView>> {
        Ok(ask!(self, MyBids))
    }
    pub async fn swaps(&self) -> Result<Vec<SwapView>> {
        Ok(ask!(self, Swaps))
    }
    /// Sweep an expired order's unsold coins back into the wallet.
    pub async fn reclaim_order(&self, group_id: String) -> Result<String> {
        ask!(self, ReclaimOrder { group_id: group_id })
    }
    /// Dry-run a swap: prerequisites, deadlines, and what will happen. Signs
    /// nothing and moves nothing.
    #[allow(clippy::too_many_arguments)]
    pub async fn swap_quote(
        &self,
        side: String,
        rail: String,
        mds_amount: u64,
        wei_amount: String,
        peer_mds_pk: String,
        eth_refund_secs: u64,
    ) -> Result<SwapQuoteView> {
        ask!(self, SwapQuote {
            side: side,
            rail: rail,
            mds_amount: mds_amount,
            wei_amount: wei_amount,
            peer_mds_pk: peer_mds_pk,
            eth_refund_secs: eth_refund_secs
        })
    }
    /// Rebuild missing history amounts by rereading the block store.
    pub async fn repair_history(&self) -> Result<String> {
        ask!(self, RepairHistory)
    }
    /// Ask a peer to open a payment channel toward us.
    pub async fn request_channel(&self, peer: String, capacity: u64) -> Result<()> {
        ask!(self, RequestChannel { peer: peer, capacity: capacity })
    }
    pub async fn evm_account(&self) -> Result<EvmAccountView> {
        Ok(ask!(self, EvmAccount))
    }
    pub async fn order_book(&self) -> Result<OrderBookView> {
        Ok(ask!(self, OrderBook))
    }
    pub async fn sync_order_book(&self) -> Result<()> {
        ask!(self, SyncOrderBook)
    }
    pub async fn dex_config(&self) -> Result<DexConfigView> {
        Ok(ask!(self, GetDexConfig))
    }
    pub async fn set_dex_config(&self, cfg: DexConfigView) -> Result<()> {
        ask!(self, SetDexConfig { cfg: cfg })
    }
}

/// Spawn the actor. `wallet_path` is the single managed wallet file
/// (multi-wallet is a v1.x item). `data_dir`/`rpc_url` are informational.
pub fn spawn(
    node: NodeHandle,
    wallet_path: PathBuf,
    data_dir: PathBuf,
    rpc_url: Option<String>,
) -> WalletdHandle {
    let (tx, rx) = mpsc::channel(64);
    let (events, _) = broadcast::channel(256);
    let handle = WalletdHandle { tx: tx.clone(), events: events.clone() };

    let svc = Service {
        node,
        wallet_path,
        data_dir,
        rpc_url,
        wallet: None,
        scan_pos: 0,
        sends: HashMap::new(),
        book: ChannelBook::default(),
        dex: OrderBook::new(),
        dex_cfg: ChainConfig::default(),
        // ~200k Base blocks at 2s each is a bit under five days of history,
        // enough to populate the book without hammering a public endpoint.
        dex_window: 200_000,
        dex_start_block: 0,
        dex_error: None,
        ledger: Ledger::default(),
        swaps: SwapBook::default(),
        events,
        self_tx: tx,
    };
    tokio::spawn(svc.run(rx));
    handle
}

// ── The actor ───────────────────────────────────────────────────────────────

struct SendMeta {
    stage: SendStage,
    detail: String,
    amount: u64,
    fee: u64,
    to: String,
    updated_at: u64,
}

struct Service {
    node: NodeHandle,
    wallet_path: PathBuf,
    data_dir: PathBuf,
    rpc_url: Option<String>,
    wallet: Option<Wallet>,
    /// Highest block height already scanned for incoming coins.
    scan_pos: u64,
    sends: HashMap<[u8; 32], SendMeta>,
    book: ChannelBook,
    /// Cross-chain order book. Read-only: it never signs or moves value.
    dex: OrderBook,
    dex_cfg: ChainConfig,
    dex_window: u64,
    dex_start_block: u64,
    dex_error: Option<String>,
    /// Remembers what coins were worth, so history stays truthful after they move.
    ledger: Ledger,
    /// Live cross-chain swaps. Persisted, because the gap between the two legs
    /// is exactly where losing state loses money.
    swaps: SwapBook,
    events: broadcast::Sender<WalletEvent>,
    self_tx: mpsc::Sender<Cmd>,
}

impl Service {
    async fn run(mut self, mut rx: mpsc::Receiver<Cmd>) {
        let mut ticker = tokio::time::interval(TICK);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                cmd = rx.recv() => match cmd {
                    Some(cmd) => self.handle(cmd).await,
                    None => break,
                },
                _ = ticker.tick() => self.handle(Cmd::Internal(Internal::Tick)).await,
            }
        }
    }

    async fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Status(resp) => {
                let _ = resp.send(WalletStatus {
                    exists: self.wallet_path.exists(),
                    unlocked: self.wallet.is_some(),
                    is_hd: self.wallet.as_ref().map(|w| w.is_hd()).unwrap_or(false),
                    wallet_path: self.wallet_path.display().to_string(),
                });
            }
            Cmd::Create { password, phrase, resp } => {
                let _ = resp.send(self.create(&password, &phrase).await);
            }
            Cmd::NewPhrase(resp) => {
                let _ = resp.send(
                    midstate::wallet::hd::generate_mnemonic().map(|(_, phrase)| phrase),
                );
            }
            Cmd::CheckPhrase { phrase, resp } => {
                let p = phrase.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
                let _ = resp.send(
                    midstate::wallet::hd::master_seed_from_mnemonic(&p).map(|_| p),
                );
            }
            Cmd::VerifyPhrase { phrase, resp } => {
                let _ = resp.send(self.verify_phrase(&phrase));
            }
            Cmd::Restore { password, phrase, resp } => {
                let _ = resp.send(self.restore(&password, &phrase).await);
            }
            Cmd::Unlock { password, resp } => {
                let _ = resp.send(self.unlock(&password).await);
            }
            Cmd::Lock(resp) => {
                let r = self.save_wallet();
                // Dropping the wallet clears keys from the actor. Zeroizing the
                // underlying buffers is the plan §8 upstream change (zeroize on
                // WalletKey/MssKeypair) — tracked, not yet in the vendored crate.
                self.wallet = None;
                let _ = resp.send(r);
            }
            Cmd::NewAddress { mss, label, resp } => {
                let _ = resp.send(self.new_address(mss, label));
            }
            Cmd::Addresses(resp) => {
                let _ = resp.send(self.addresses());
            }
            Cmd::Balance(resp) => {
                let _ = resp.send(self.balance().await);
            }
            Cmd::Coins(resp) => {
                let _ = resp.send(self.coins().await);
            }
            Cmd::History(resp) => {
                let _ = resp.send(self.history());
            }
            Cmd::Sends(resp) => {
                let _ = resp.send(self.send_progress_list());
            }
            Cmd::Send { to, amount, private, resp } => {
                let _ = resp.send(self.start_send(&to, amount, private).await);
            }
            Cmd::RetrySend { id, resp } => {
                let _ = resp.send(self.retry_send(&id).await);
            }
            Cmd::ValidateAddress { addr, resp } => {
                let _ = resp.send(self.validate_address(&addr).await);
            }
            Cmd::SyncStatus(resp) => {
                let _ = resp.send(self.sync_status().await);
            }
            Cmd::NodeInfo(resp) => {
                let peers = self.node.get_peers().await;
                let state = self.node.get_state().await;
                let (mempool, _) = self.node.get_mempool_info().await;
                let safe_depth = self.node.get_safe_depth().await;
                // Target is a 256-bit threshold; the count of leading zero bits
                // is the human-readable "difficulty" the miner logs report.
                let difficulty_bits = state
                    .target
                    .iter()
                    .map(|b| b.leading_zeros())
                    .take_while(|z| *z == 8)
                    .count() as u32
                    + state
                        .target
                        .iter()
                        .find(|b| **b != 0)
                        .map(|b| b.leading_zeros())
                        .unwrap_or(0);
                let _ = resp.send(NodeInfo {
                    peers,
                    data_dir: self.data_dir.display().to_string(),
                    rpc_url: self.rpc_url.clone(),
                    block_reward: midstate::core::block_reward(state.height),
                    height: state.height,
                    tip_timestamp: state.timestamp,
                    header_hash: hex::encode(state.header_hash),
                    midstate: hex::encode(state.midstate),
                    depth: state.depth.to_string(),
                    difficulty_bits,
                    utxo_count: state.coins.len(),
                    commitment_count: state.commitments.len(),
                    burned_count: state.burned_wots.len(),
                    mempool,
                    safe_depth,
                });
            }
            Cmd::RescanFrom { height, resp } => {
                let r = if self.wallet.is_some() {
                    self.scan_pos = height;
                    self.persist_scan_pos();
                    Ok(())
                } else {
                    Err(anyhow!("unlock the wallet first"))
                };
                let _ = resp.send(r);
            }
            Cmd::Consolidate { address, resp } => {
                let _ = resp.send(self.consolidate(&address).await);
            }
            Cmd::Defrag { max_inputs, resp } => {
                let _ = resp.send(self.defrag(max_inputs).await);
            }
            Cmd::AbandonSend { id, resp } => {
                let _ = resp.send(self.abandon_send(&id).await);
            }
            Cmd::AbandonAddress { address, resp } => {
                let _ = resp.send(self.abandon_address(&address));
            }
            Cmd::ImportCoin { seed, value, salt, label, resp } => {
                let _ = resp.send(self.import_coin_cmd(&seed, value, &salt, label));
            }
            Cmd::ExportCoin { id, resp } => {
                let _ = resp.send(self.export_coin(&id));
            }
            Cmd::ChatSend { text, resp } => {
                let _ = resp.send(self.chat_send(&text));
            }
            Cmd::ChatHistory(resp) => {
                let _ = resp.send(self.chat_history().await);
            }
            Cmd::ChatDict(resp) => {
                let _ = resp.send(chat_dictionary_vec());
            }
            Cmd::ChannelIdentity(resp) => {
                let _ = resp.send(self.identity_view());
            }
            Cmd::Channels(resp) => {
                let _ = resp.send(self.channels_list());
            }
            Cmd::ChannelOpen { peer, amount, lifetime, resp } => {
                let _ = resp.send(self.channel_open(&peer, amount, lifetime).await);
            }
            Cmd::ChannelPay { id, amount, resp } => {
                let _ = resp.send(self.channel_pay(&id, amount).await);
            }
            Cmd::ChannelClose { id, resp } => {
                let _ = resp.send(self.channel_close_cmd(&id).await);
            }
            Cmd::ChannelRefund { id, resp } => {
                let _ = resp.send(self.channel_refund_cmd(&id).await);
            }
            Cmd::CreateInvoice { amount, resp } => {
                let _ = resp.send(self.create_invoice(amount).await);
            }
            Cmd::PayInvoice { text, resp } => {
                let _ = resp.send(self.pay_invoice(&text).await);
            }
            Cmd::RequestInvoice { payee, amount, resp } => {
                let _ = resp.send(self.request_invoice(&payee, amount).await);
            }
            Cmd::Invoices(resp) => {
                let _ = resp.send(self.invoice_list());
            }
            Cmd::GetHub(resp) => {
                let h = &self.book.hub;
                let _ = resp.send(HubView {
                    auto_accept: h.auto_accept,
                    forward: h.forward,
                    jit_open: h.jit_open,
                    jit_capacity: h.jit_capacity,
                    min_leaves: h.min_leaves,
                    auto_open_on_request: h.auto_open_on_request,
                    max_auto_capacity: h.max_auto_capacity,
                    auto_capacity_budget: h.auto_capacity_budget,
                });
            }
            Cmd::SetHub { cfg, resp } => {
                self.book.hub = channels::HubConfig {
                    auto_accept: cfg.auto_accept,
                    forward: cfg.forward,
                    jit_open: cfg.jit_open,
                    jit_capacity: cfg.jit_capacity.max(channels::MIN_CAPACITY),
                    min_leaves: cfg.min_leaves,
                    auto_open_on_request: cfg.auto_open_on_request,
                    max_auto_capacity: cfg.max_auto_capacity.max(channels::MIN_CAPACITY),
                    auto_capacity_budget: cfg.auto_capacity_budget,
                };
                self.book.save(&self.wallet_path);
                let _ = resp.send(Ok(()));
            }
            Cmd::RotateIdentity(resp) => {
                let _ = resp.send(self.rotate_identity());
            }
            Cmd::RequestAddress { peer, resp } => {
                let _ = resp.send(self.request_address(&peer).await);
            }
            Cmd::PlaceAsk { mds_amount, wei_amount, lifetime_blocks, resp } => {
                let r = self.place_ask(mds_amount, &wei_amount, lifetime_blocks).await;
                let _ = resp.send(r);
            }
            Cmd::MyOrders(resp) => {
                let _ = resp.send(self.my_orders());
            }
            Cmd::TakeAsk { group_id, unit, resp } => {
                let _ = resp.send(self.take_ask(&group_id, unit).await);
            }
            Cmd::PlaceBid { mds_amount, wei, ttl_secs, fill_bond, resp } => {
                let r = self.place_bid(mds_amount, &wei, ttl_secs, &fill_bond).await;
                let _ = resp.send(r);
            }
            Cmd::CancelBid { bid_id, resp } => {
                let _ = resp.send(self.cancel_bid(&bid_id).await);
            }
            Cmd::MyBids(resp) => {
                let _ = resp.send(self.my_bids_view());
            }
            Cmd::Swaps(resp) => {
                let _ = resp.send(self.swaps_view());
            }
            Cmd::ReclaimOrder { group_id, resp } => {
                let _ = resp.send(self.reclaim_order(&group_id).await);
            }
            Cmd::SwapQuote { side, rail, mds_amount, wei_amount, peer_mds_pk, eth_refund_secs, resp } => {
                let r = self
                    .swap_quote(&side, &rail, mds_amount, &wei_amount, &peer_mds_pk, eth_refund_secs)
                    .await;
                let _ = resp.send(r);
            }
            Cmd::RepairHistory(resp) => {
                let _ = resp.send(self.repair_history().await);
            }
            Cmd::RequestChannel { peer, capacity, resp } => {
                let _ = resp.send(self.request_channel(&peer, capacity).await);
            }
            Cmd::EvmAccount(resp) => {
                let _ = resp.send(self.evm_account().await);
            }
            Cmd::OrderBook(resp) => {
                let _ = resp.send(self.order_book_view());
            }
            Cmd::SyncOrderBook(resp) => {
                let r = self.sync_order_book().await;
                self.dex_error = r.as_ref().err().map(|e| format!("{e:#}"));
                let _ = resp.send(r);
            }
            Cmd::GetDexConfig(resp) => {
                let c = &self.dex_cfg;
                let _ = resp.send(DexConfigView {
                    rpc_url: c.rpc_url.clone(),
                    chain_id: c.chain_id,
                    contract: c.contract.clone(),
                    confirmations: c.confirmations,
                    scan_window: self.dex_window,
                    start_block: self.dex_start_block,
                });
            }
            Cmd::SetDexConfig { cfg, resp } => {
                let r = (|| -> Result<()> {
                    crate::evm::parse_address(&cfg.contract)?;
                    if !cfg.rpc_url.starts_with("http") {
                        bail!("the RPC endpoint must be an http(s) URL");
                    }
                    Ok(())
                })();
                if r.is_ok() {
                    self.dex_cfg = ChainConfig {
                        rpc_url: cfg.rpc_url,
                        chain_id: cfg.chain_id,
                        contract: cfg.contract,
                        confirmations: cfg.confirmations,
                    };
                    self.dex_window = cfg.scan_window.clamp(1_000, 20_000_000);
                    self.dex_start_block = cfg.start_block;
                    // Pointing somewhere new invalidates everything gathered.
                    self.dex = OrderBook::new();
                    self.dex_error = None;
                }
                let _ = resp.send(r);
            }
            Cmd::Internal(i) => self.internal(i).await,
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    async fn create(&mut self, password: &str, phrase: &str) -> Result<()> {
        if self.wallet_path.exists() {
            bail!("a wallet already exists at {}", self.wallet_path.display());
        }
        if let Some(parent) = self.wallet_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // The phrase was minted and confirmed before we got here, so the
        // wallet is built FROM it rather than generating one internally —
        // that is what lets the UI show and verify it before anything is
        // written to disk.
        let mut wallet =
            Wallet::restore_from_mnemonic(&self.wallet_path, password.as_bytes(), phrase.trim())?;
        // Derive the Base account now, while the phrase is in hand. It is
        // never available again, and the standard BIP44 path is what makes
        // this account reachable from MetaMask with the same words.
        if let Ok(k) = crate::evm::EvmKey::from_mnemonic(phrase) {
            wallet.data.evm_secret = Some(k.secret_bytes());
            wallet.save()?;
        }
        self.wallet = Some(wallet);
        // Fresh wallet: nothing historical can belong to it — start scanning
        // from the current tip instead of genesis.
        self.scan_pos = self.node.get_state().await.height;
        self.persist_scan_pos();
        Ok(())
    }

    /// Compare the seed derived from `phrase` against the open wallet's own
    /// master seed. Nothing is stored and the words never leave this call —
    /// the comparison is one-way in both directions.
    fn verify_phrase(&self, phrase: &str) -> Result<bool> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let mine = w
            .data
            .master_seed
            .ok_or_else(|| anyhow!("this wallet predates recovery phrases and has no seed"))?;
        let p = phrase.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let theirs = midstate::wallet::hd::master_seed_from_mnemonic(&p)?;
        Ok(theirs == mine)
    }

    async fn restore(&mut self, password: &str, phrase: &str) -> Result<()> {
        // Same derivation as `create`, so restoring reproduces the very same
        // Base address rather than a fresh one.
        if self.wallet_path.exists() {
            bail!(
                "a wallet already exists at {} — move it aside before restoring",
                self.wallet_path.display()
            );
        }
        if let Some(parent) = self.wallet_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut wallet =
            Wallet::restore_from_mnemonic(&self.wallet_path, password.as_bytes(), phrase.trim())?;
        // Derive the unconditional key floor, then let the scan tick walk the
        // chain from genesis. (Gap-limit extension past the floor: v1.x.)
        wallet.restore_generate_keys(RESTORE_KEY_FLOOR)?;
        if let Ok(k) = crate::evm::EvmKey::from_mnemonic(phrase) {
            wallet.data.evm_secret = Some(k.secret_bytes());
        }
        wallet.save()?;
        self.wallet = Some(wallet);
        self.scan_pos = 0;
        self.persist_scan_pos();
        Ok(())
    }

    async fn unlock(&mut self, password: &str) -> Result<()> {
        if self.wallet.is_some() {
            return Ok(());
        }
        let wallet = Wallet::open(&self.wallet_path, password.as_bytes())?;
        self.wallet = Some(wallet);
        self.scan_pos = self.load_scan_pos();
        self.book = ChannelBook::load(&self.wallet_path);
        self.ledger = Ledger::load(&self.wallet_path);
        self.swaps = SwapBook::load(&self.wallet_path);
        self.resume_pendings().await;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    fn save_wallet(&mut self) -> Result<()> {
        if let Some(w) = self.wallet.as_ref() {
            w.save()?;
        }
        Ok(())
    }

    // ── Addresses ───────────────────────────────────────────────────────

    fn new_address(&mut self, mss: bool, label: Option<String>) -> Result<AddressInfo> {
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let (addr, kind, remaining) = if mss {
            let a = w.generate_mss(DEFAULT_MSS_HEIGHT, label.clone())?;
            (a, "mss", Some(1u64 << DEFAULT_MSS_HEIGHT))
        } else {
            let a = w.generate_key(label.clone())?;
            (a, "wots", None)
        };
        Ok(AddressInfo {
            address: encode_address_with_checksum(&addr),
            kind: kind.into(),
            label,
            remaining_sigs: remaining,
            used: false,
        })
    }

    fn addresses(&self) -> Result<Vec<AddressInfo>> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let coin_addrs: HashSet<[u8; 32]> = w.coins().iter().map(|c| c.address).collect();
        // A one-time address is unsafe to reuse only once its WOTS signature
        // has actually been consumed (a coin at it was spent). Receiving to it
        // without spending leaves the key intact.
        let signed_addrs: HashSet<[u8; 32]> =
            w.coins().iter().filter(|c| c.wots_signed).map(|c| c.address).collect();
        let mut out = Vec::new();
        for k in w.keys() {
            out.push(AddressInfo {
                address: encode_address_with_checksum(&k.address),
                kind: "wots".into(),
                label: k.label.clone(),
                remaining_sigs: None,
                used: signed_addrs.contains(&k.address),
            });
        }
        for m in w.mss_keys() {
            let addr = midstate::core::compute_address(&m.master_pk);
            out.push(AddressInfo {
                address: encode_address_with_checksum(&addr),
                kind: "mss".into(),
                label: None,
                remaining_sigs: Some(m.remaining()),
                used: coin_addrs.contains(&addr),
            });
        }
        Ok(out)
    }

    // ── Views ───────────────────────────────────────────────────────────

    fn in_flight_inputs(&self) -> HashSet<[u8; 32]> {
        self.wallet
            .as_ref()
            .map(|w| {
                w.pending()
                    .iter()
                    .flat_map(|p| p.input_coin_ids.iter().copied())
                    .collect()
            })
            .unwrap_or_default()
    }

    async fn balance(&self) -> Result<Balance> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let state = self.node.get_state().await;
        let in_flight_ids = self.in_flight_inputs();
        let mut b = Balance { coin_count: w.coins().len(), ..Default::default() };
        for c in w.coins() {
            let live = state.coins.contains(&c.coin_id);
            if in_flight_ids.contains(&c.coin_id) {
                b.in_flight += c.value;
            } else if live {
                b.confirmed += c.value;
            } else {
                b.unconfirmed += c.value;
            }
        }
        Ok(b)
    }

    async fn coins(&self) -> Result<Vec<CoinView>> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let state = self.node.get_state().await;
        let in_flight_ids = self.in_flight_inputs();
        let mss_addrs: HashSet<[u8; 32]> = w
            .mss_keys()
            .iter()
            .map(|m| midstate::core::compute_address(&m.master_pk))
            .collect();
        Ok(w.coins()
            .iter()
            .map(|c| CoinView {
                coin_id: hex::encode(c.coin_id),
                address: encode_address_with_checksum(&c.address),
                value: c.value,
                kind: if mss_addrs.contains(&c.address) { "mss" } else { "wots" }.into(),
                label: c.label.clone(),
                live: state.coins.contains(&c.coin_id),
                wots_signed: c.wots_signed,
                in_flight: in_flight_ids.contains(&c.coin_id),
            })
            .collect())
    }

    fn history(&mut self) -> Result<Vec<HistoryView>> {
        // Learn the value of everything currently held first: a coin seen even
        // once stays priced here after the wallet itself forgets it.
        let snapshot: Vec<([u8; 32], u64)> = {
            let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
            w.coins().iter().map(|c| (c.coin_id, c.value)).collect()
        };
        self.ledger.observe(&snapshot);
        self.ledger.save(&self.wallet_path);

        let w = self.wallet.as_ref().unwrap();
        Ok(w.history()
            .iter()
            .rev()
            .map(|h| {
                let (ours_value, ours_out) = self.ledger.value_of(&h.outputs);
                let rec = self.ledger.send_for(&h.inputs);
                let outgoing = h.kind == "sent";
                HistoryView {
                    kind: h.kind.clone(),
                    fee: h.fee,
                    timestamp: h.timestamp,
                    inputs: h.inputs.iter().map(hex::encode).collect(),
                    outputs: h.outputs.iter().map(hex::encode).collect(),
                    // Incoming value only. A send brings nothing in; its change
                    // is reported separately so the two are never conflated.
                    amount: if outgoing { 0 } else { ours_value },
                    sent: rec.map(|r| r.amount),
                    to: rec.map(|r| r.to.clone()),
                    change: if outgoing { ours_value } else { 0 },
                    n_in: h.inputs.len(),
                    n_out: h.outputs.len(),
                    ours_out,
                }
            })
            .collect())
    }

    async fn sync_status(&self) -> SyncStatus {
        let state = self.node.get_state().await;
        let peers = self.node.get_peers().await;
        let (mempool, _) = self.node.get_mempool_info().await;
        // 60-second block target ⇒ expected height ≈ tip height + elapsed/60.
        let now = now_secs();
        let est_target_height = if state.timestamp > 0 && now > state.timestamp {
            state.height + (now - state.timestamp) / 60
        } else {
            state.height
        };
        SyncStatus {
            height: state.height,
            is_syncing: self.node.is_syncing(),
            peer_count: peers.len(),
            mempool,
            safe_depth: self.node.get_safe_depth().await,
            num_coins: state.coins.len(),
            num_commitments: state.commitments.len(),
            midstate: hex::encode(state.midstate),
            est_target_height,
            timestamp: state.timestamp,
        }
    }

    fn send_progress_list(&self) -> Vec<SendProgress> {
        let mut v: Vec<SendProgress> = self
            .sends
            .iter()
            .map(|(id, m)| SendProgress {
                id: hex::encode(id),
                stage: m.stage,
                detail: m.detail.clone(),
                amount: m.amount,
                fee: m.fee,
                to: m.to.clone(),
                updated_at: m.updated_at,
            })
            .collect();
        v.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        v
    }

    fn set_stage(&mut self, id: [u8; 32], stage: SendStage, detail: impl Into<String>) {
        let m = self.sends.entry(id).or_insert_with(|| SendMeta {
            stage,
            detail: String::new(),
            amount: 0,
            fee: 0,
            to: String::new(),
            updated_at: now_secs(),
        });
        m.stage = stage;
        m.detail = detail.into();
        m.updated_at = now_secs();
        let progress = SendProgress {
            id: hex::encode(id),
            stage: m.stage,
            detail: m.detail.clone(),
            amount: m.amount,
            fee: m.fee,
            to: m.to.clone(),
            updated_at: m.updated_at,
        };
        let _ = self.events.send(WalletEvent::SendUpdate { progress });
    }

    // ── Send machine ────────────────────────────────────────────────────

    /// STRICT SAFETY (ported from the CLI): before any signing session,
    /// reconcile every MSS key's leaf counter with chain + mempool state and
    /// fast-forward with a margin if the network has seen more signatures.
    async fn verify_mss_indices(&mut self) -> Result<()> {
        let Some(w) = self.wallet.as_mut() else { return Ok(()) };
        if w.data.mss_keys.is_empty() {
            return Ok(());
        }
        let (_, mempool_txs) = self.node.get_mempool_info().await;
        let mut dirty = false;
        for i in 0..w.data.mss_keys.len() {
            let master_pk = w.data.mss_keys[i].master_pk;
            let local = w.data.mss_keys[i].next_leaf;
            let chain_max = self.node.storage.query_mss_leaf_index(&master_pk).unwrap_or(0);
            let mempool_max = midstate::node::scan_txs_for_mss_index(&mempool_txs, &master_pk);
            let seen = chain_max.max(mempool_max);
            if seen > local {
                let new_leaf = seen + MSS_SAFETY_MARGIN;
                tracing::warn!(
                    "MSS key {}: stale local index (network {seen}, local {local}) — fast-forwarding to {new_leaf}",
                    hex::encode(master_pk)
                );
                w.data.mss_keys[i].set_next_leaf(new_leaf);
                dirty = true;
            }
        }
        if dirty {
            // If this save fails we must not sign — surface the error.
            w.save().context("failed to persist MSS fast-forward; aborting before signing")?;
        }
        Ok(())
    }

    /// Decode a destination and reject it if its one-time key has already
    /// signed. Same consensus lookup as the send gate, run while typing so the
    /// problem surfaces before any value moves.
    async fn validate_address(&self, addr: &str) -> Result<()> {
        let dest = sendplan::decode_address(addr)?;
        if self.node.get_state().await.burned_wots.contains(&dest) {
            bail!(
                "this address has already spent its one-time key — anything sent to it \
                 cannot be recovered"
            );
        }
        Ok(())
    }

    async fn start_send(&mut self, to: &str, amount: u64, private: bool) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — sending against a stale coin set would fail");
        }
        let dest = sendplan::decode_address(to)?;
        self.verify_mss_indices().await?;

        let state = self.node.get_state().await;

        // A one-time key that has already signed can never sign again, so
        // coins sent to it are unspendable forever. Burning is CONSENSUS
        // state (`burned_wots` is folded into the state root), so this is a
        // local, deterministic lookup — no message from the recipient is
        // needed, and nothing about the payment is revealed to anyone.
        if state.burned_wots.contains(&dest) {
            bail!(
                "that address has already spent its one-time key — anything sent there \
                 cannot be recovered. Ask the recipient for a fresh address."
            );
        }

        // Live coins: on-chain AND not already promised to a pending commit.
        let in_flight_ids = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let live: Vec<[u8; 32]> = w
            .coins()
            .iter()
            .filter(|c| {
                !c.wots_signed
                    && !in_flight_ids.contains(&c.coin_id)
                    && state.coins.contains(&c.coin_id)
            })
            .map(|c| c.coin_id)
            .collect();

        let SendPlan { input_coin_ids, outputs, change_seeds, in_sum: _, fee, amount } =
            sendplan::plan_send(w, &live, dest, amount)?;

        let (commitment, _salt) =
            w.prepare_commit(&input_coin_ids, &outputs, change_seeds, private, false)?;
        w.save()?; // pending commit + allocated change indices are now durable

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "solving commit proof-of-work".into(),
                amount,
                fee,
                to: to.trim().to_string(),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "solving commit proof-of-work");

        self.broadcast_commit(commitment).await?;
        Ok(hex::encode(commitment))
    }

    /// Grind the commit anti-spam PoW against current chain parameters and
    /// broadcast `Transaction::Commit`, then start the mined-watch monitor.
    async fn broadcast_commit(&mut self, commitment: [u8; 32]) -> Result<()> {
        let state = self.node.get_state().await;
        let required =
            midstate::mempool::Mempool::calculate_required_pow(state.commitments.len());
        let height = state.height;
        let header_hash = state.header_hash;

        let spam_nonce = tokio::task::spawn_blocking(move || {
            midstate::core::transaction::mine_pow(&commitment, required, height, header_hash)
        })
        .await
        .context("commit PoW task panicked")?;

        self.node
            .send_transaction(Transaction::Commit { commitment, spam_nonce })
            .await
            .context("commit broadcast failed")?;

        let reveal_not_before = self
            .wallet
            .as_ref()
            .and_then(|w| w.find_pending(&commitment))
            .map(|p| p.reveal_not_before)
            .unwrap_or(0);

        self.set_stage(commitment, SendStage::CommitPending, "waiting for commitment to be mined");
        self.spawn_commit_monitor(commitment, reveal_not_before);
        Ok(())
    }

    fn spawn_commit_monitor(&self, commitment: [u8; 32], reveal_not_before: u64) {
        let node = self.node.clone();
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            let deadline = tokio::time::Instant::now() + COMMIT_PATIENCE;
            loop {
                if tokio::time::Instant::now() >= deadline {
                    let _ = tx.send(Cmd::Internal(Internal::CommitStalled(commitment))).await;
                    return;
                }
                if node.check_commitment(commitment).await {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
            // Commitment is in chain state (not merely in the mempool — a
            // commit can be evicted or reorged out; state is what makes the
            // reveal spendable). Honor the privacy delay, then hand back.
            let wait = reveal_not_before.saturating_sub(now_secs());
            if wait > 0 {
                tokio::time::sleep(Duration::from_secs(wait)).await;
            }
            let _ = tx.send(Cmd::Internal(Internal::TryReveal(commitment))).await;
        });
    }

    fn spawn_reveal_monitor(&self, commitment: [u8; 32], first_input: [u8; 32]) {
        let node = self.node.clone();
        let tx = self.self_tx.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                // The send is on-chain when its first input has left the UTXO
                // set (same signal the CLI uses).
                if !node.check_coin(first_input).await {
                    let _ = tx.send(Cmd::Internal(Internal::RevealConfirmed(commitment))).await;
                    return;
                }
            }
        });
    }

    async fn retry_send(&mut self, id: &str) -> Result<()> {
        let commitment = parse_hex32(id)?;
        let Some(w) = self.wallet.as_ref() else { bail!("wallet is locked") };
        if w.find_pending(&commitment).is_none() {
            bail!("no pending send with that id");
        }
        if self.node.check_commitment(commitment).await {
            // Already mined — go straight to reveal.
            self.internal(Internal::TryReveal(commitment)).await;
        } else {
            self.set_stage(commitment, SendStage::Committing, "re-solving commit proof-of-work");
            self.broadcast_commit(commitment).await?;
        }
        Ok(())
    }

    /// On unlock: every persisted pending commit resumes at the right stage.
    async fn resume_pendings(&mut self) {
        let pendings: Vec<_> = self
            .wallet
            .as_ref()
            .map(|w| {
                w.pending()
                    .iter()
                    .map(|p| (p.commitment, p.reveal_not_before, p.input_coin_ids.clone(), p.outputs.clone()))
                    .collect()
            })
            .unwrap_or_default();

        for (commitment, reveal_not_before, inputs, outputs) in pendings {
            // Reconstruct meta for the UI: recipient amount = outputs minus
            // change; fee = input values minus output values (best effort).
            let (amount, fee) = self
                .wallet
                .as_ref()
                .map(|w| reconstruct_meta(w, &inputs, &outputs, &commitment))
                .unwrap_or((0, 0));
            self.sends.insert(
                commitment,
                SendMeta {
                    stage: SendStage::CommitPending,
                    detail: "resumed after restart".into(),
                    amount,
                    fee,
                    to: String::new(),
                    updated_at: now_secs(),
                },
            );
            if self.node.check_commitment(commitment).await {
                let wait = reveal_not_before.saturating_sub(now_secs());
                if wait == 0 {
                    self.internal(Internal::TryReveal(commitment)).await;
                } else {
                    self.set_stage(commitment, SendStage::WaitingReveal, "resumed — waiting out reveal delay");
                    self.spawn_commit_monitor(commitment, reveal_not_before);
                }
            } else {
                self.set_stage(
                    commitment,
                    SendStage::Stalled,
                    "commitment not found on-chain after restart — retry to re-broadcast",
                );
            }
        }
    }

    async fn internal(&mut self, msg: Internal) {
        match msg {
            Internal::Tick => self.tick().await,
            Internal::CommitStalled(c) => {
                self.set_stage(
                    c,
                    SendStage::Stalled,
                    "commitment not mined yet — retry to re-broadcast, funds remain yours",
                );
            }
            Internal::TryReveal(c) => self.try_reveal(c).await,
            Internal::RevealConfirmed(c) => {
                if let Some(w) = self.wallet.as_mut() {
                    let spent: Vec<[u8; 32]> = w
                        .find_pending(&c)
                        .map(|p| p.input_coin_ids.clone())
                        .unwrap_or_default();
                    match w.complete_reveal(&c) {
                        Ok(()) => {
                            let _ = w.save();
                            // The recipient's outputs were never ours to price,
                            // so this is the only moment the real amount exists.
                            if let Some(m) = self.sends.get(&c) {
                                if !spent.is_empty() {
                                    self.ledger.record_send(
                                        &spent,
                                        SendRecord {
                                            amount: m.amount,
                                            fee: m.fee,
                                            to: m.to.clone(),
                                            at: now_secs(),
                                        },
                                    );
                                    self.ledger.save(&self.wallet_path);
                                }
                            }
                            self.set_stage(c, SendStage::Confirmed, "spend confirmed on-chain");
                            let _ = self.events.send(WalletEvent::WalletChanged);
                        }
                        Err(e) => {
                            self.set_stage(c, SendStage::Failed, format!("bookkeeping failed: {e}"));
                        }
                    }
                }
            }
        }
    }

    async fn try_reveal(&mut self, commitment: [u8; 32]) {
        let Some(w) = self.wallet.as_mut() else { return };
        let Some(pending) = w.find_pending(&commitment).cloned() else { return };

        self.set_stage(commitment, SendStage::WaitingReveal, "signing reveal");

        let w = self.wallet.as_mut().unwrap();
        let (input_reveals, witnesses) = match w.sign_reveal(&pending) {
            Ok(r) => r,
            Err(e) => {
                // Inputs are gone (stale commit) — drop it, as the CLI does.
                w.data.pending.retain(|p| p.commitment != commitment);
                let _ = w.save();
                self.set_stage(
                    commitment,
                    SendStage::Failed,
                    format!("could not build reveal: {e}. Stale commit removed; coins were not spent."),
                );
                return;
            }
        };

        // Persist signature side-effects (wots_signed flags, MSS leaf
        // advance) BEFORE broadcasting — a crash between broadcast and save
        // must never leave a signed key looking unsigned.
        if let Err(e) = w.save() {
            self.set_stage(commitment, SendStage::Failed, format!("could not persist wallet after signing: {e}"));
            return;
        }

        let first_input = pending.input_coin_ids[0];
        let tx = if pending.is_consolidate {
            let witness = witnesses.into_iter().next().expect("consolidate has one witness");
            Transaction::Consolidate {
                inputs: input_reveals,
                witness,
                outputs: pending.outputs.clone(),
                salt: pending.salt,
            }
        } else {
            Transaction::Reveal {
                inputs: input_reveals,
                witnesses,
                outputs: pending.outputs.clone(),
                salt: pending.salt,
            }
        };

        match self.node.send_transaction(tx).await {
            Ok(()) => {
                self.set_stage(commitment, SendStage::RevealPending, "reveal broadcast — waiting for confirmation");
                self.spawn_reveal_monitor(commitment, first_input);
            }
            Err(e) => {
                self.set_stage(
                    commitment,
                    SendStage::Failed,
                    format!("reveal broadcast failed: {e}. The signed reveal is preserved; retry when the node recovers."),
                );
            }
        }
    }


    // ── Consolidate / defrag / coin management ──────────────────────────

    /// Targeted full-range scan for specific addresses, importing anything
    /// the wallet was missing. This is the destructive-operation guard from
    /// the CLI: consolidation spends EVERY live coin at an address in one
    /// reveal (burning the one-time key), so unknown siblings must be found
    /// first or they are stranded forever.
    async fn scan_import_range(&mut self, addrs: Vec<[u8; 32]>, end: u64) -> Result<usize> {
        if addrs.is_empty() {
            return Ok(0);
        }
        let node = self.node.clone();
        let found = tokio::task::spawn_blocking(move || node.scan_addresses(&addrs, 0, end))
            .await
            .context("scan task panicked")??;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let mut imported = 0usize;
        for sc in &found {
            if let Ok(Some(_)) = w.import_scanned(sc.address, sc.value, sc.salt, None) {
                imported += 1;
            }
        }
        if imported > 0 {
            w.save()?;
            let _ = self.events.send(WalletEvent::WalletChanged);
        }
        Ok(imported)
    }

    /// Sweep every live coin at one address into a fresh reusable (MSS)
    /// address, as a consensus-level Consolidate transaction (one witness).
    /// Mirrors CLI `wallet consolidate` with the completeness guard always on.
    async fn consolidate(&mut self, address: &str) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — consolidating against a stale coin set could burn coins");
        }
        let addr = midstate::core::types::parse_address_flexible(address).map_err(|e| anyhow!(e))?;
        self.verify_mss_indices().await?;

        let state = self.node.get_state().await;
        let imported = self.scan_import_range(vec![addr], state.height).await?;
        if imported > 0 {
            tracing::info!("consolidate guard imported {imported} previously-unknown sibling coin(s)");
        }

        let w = self.wallet.as_mut().unwrap();
        let mut live: Vec<[u8; 32]> = Vec::new();
        let mut total = 0u64;
        for c in w.coins() {
            if c.address == addr && state.coins.contains(&c.coin_id) {
                live.push(c.coin_id);
                total += c.value;
            }
        }
        if live.len() < 2 {
            bail!(
                "that address has {} live coin(s) — consolidation is for grouped sibling coins (2 or more)",
                live.len()
            );
        }
        if live.len() > midstate::core::types::MAX_CONSOLIDATE_INPUTS {
            bail!(
                "too many coins to consolidate in one transaction ({} > {})",
                live.len(),
                midstate::core::types::MAX_CONSOLIDATE_INPUTS
            );
        }

        let dest = w.generate_mss(DEFAULT_MSS_HEIGHT, Some("Consolidated sweep".into()))?;
        // CLI fee model for consolidate: base 600 + ~3000-byte MSS witness +
        // 100 overhead + ~125 bytes per InputReveal, at 10 units/KiB, +20 pad.
        let estimated_bytes = 600 + 3000 + 100 + (live.len() as u64 * 125);
        let fee = (estimated_bytes * 10) / 1024 + 20;
        if total <= fee {
            bail!("total value {total} at that address cannot cover the network fee of {fee}");
        }
        let out_val = total - fee;
        let mut outputs = Vec::new();
        for denom in midstate::core::decompose_value(out_val) {
            let salt: [u8; 32] = rand::random();
            outputs.push(midstate::core::OutputData::Standard { address: dest, value: denom, salt });
        }

        let (commitment, _salt) = w.prepare_commit(&live, &outputs, vec![], false, true)?;
        w.save()?;

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "consolidating — solving commit proof-of-work".into(),
                amount: out_val,
                fee,
                to: midstate::core::encode_address_with_checksum(&dest),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "consolidating — solving commit proof-of-work");
        self.broadcast_commit(commitment).await?;
        Ok(hex::encode(commitment))
    }

    /// Sweep one batch of fragmented single-use coins (across addresses)
    /// into a fresh reusable address. Mirrors CLI `wallet defrag`: the
    /// wallet's own planner picks an economical batch; run again for more.
    async fn defrag(&mut self, max_inputs: usize) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — defragmenting against a stale coin set could burn coins");
        }
        self.verify_mss_indices().await?;
        let state = self.node.get_state().await;

        let live_of = |w: &Wallet| -> Vec<[u8; 32]> {
            w.coins()
                .iter()
                .filter(|c| state.coins.contains(&c.coin_id))
                .map(|c| c.coin_id)
                .collect()
        };

        let (targets, live) = {
            let w = self.wallet.as_ref().unwrap();
            let live = live_of(w);
            let live_set: HashSet<[u8; 32]> = live.iter().copied().collect();
            let bundles = w.spendable_bundles(&live_set, false);
            if bundles.len() < 2 {
                return Ok(format!(
                    "No defragmentation needed — found {} fragmented bundle(s).",
                    bundles.len()
                ));
            }
            let mut t: Vec<[u8; 32]> = bundles.iter().map(|b| b.address).collect();
            t.sort_unstable();
            t.dedup();
            (t, live)
        };

        // Completeness guard across every address in play.
        let imported = self.scan_import_range(targets, state.height).await?;
        let live = if imported > 0 {
            tracing::info!("defrag guard imported {imported} previously-unknown sibling coin(s)");
            live_of(self.wallet.as_ref().unwrap())
        } else {
            live
        };

        let w = self.wallet.as_mut().unwrap();
        let policy = midstate::wallet::FeePolicy { base: 20, per_input: 17, per_output: 2 };
        let dest = w.generate_mss(DEFAULT_MSS_HEIGHT, Some("Defrag sweep".into()))?;
        let plan = match w.plan_defrag_batch(&live, dest, &policy, max_inputs)? {
            Some(p) => p,
            None => {
                return Ok(
                    "No economical batch: the remaining fragments are too small to cover their own signature fees."
                        .into(),
                )
            }
        };

        let out_val = plan.total_in.saturating_sub(plan.fee);
        let n_inputs = plan.input_coin_ids.len();
        let (commitment, _salt) =
            w.prepare_commit(&plan.input_coin_ids, &plan.outputs, vec![], false, false)?;
        w.save()?;

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "defragmenting — solving commit proof-of-work".into(),
                amount: out_val,
                fee: plan.fee,
                to: midstate::core::encode_address_with_checksum(&dest),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "defragmenting — solving commit proof-of-work");
        let remaining = plan.remaining_fragmented_coins;
        self.broadcast_commit(commitment).await?;

        Ok(if remaining > 1 {
            format!(
                "Defrag batch started: {} coins sweeping into a fresh reusable address (fee {}). ~{} fragmented coin(s) will remain — run defrag again once this confirms.",
                n_inputs, plan.fee, remaining
            )
        } else {
            format!(
                "Defrag batch started: sweeping into a fresh reusable address (fee {}). This should be the last batch.",
                plan.fee
            )
        })
    }

    /// Drop a stalled pending commit. Only allowed before anything is
    /// signed (stage Stalled = the commitment never entered chain state),
    /// so the coins are untouched and the one-time keys unsigned.
    async fn abandon_send(&mut self, id: &str) -> Result<()> {
        let commitment = parse_hex32(id)?;
        let stage = self.sends.get(&commitment).map(|m| m.stage);
        if stage != Some(SendStage::Stalled) {
            bail!("only stalled sends (commit never mined) can be abandoned safely");
        }
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        w.data.pending.retain(|p| p.commitment != commitment);
        w.save()?;
        self.set_stage(commitment, SendStage::Failed, "abandoned — coins remain unspent and selectable");
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(())
    }

    /// Remove all wallet records for coins at an address (wallet-local only;
    /// the chain is unaffected). Mirrors CLI `wallet abandon`.
    fn abandon_address(&mut self, address: &str) -> Result<usize> {
        let addr = midstate::core::types::parse_address_flexible(address).map_err(|e| anyhow!(e))?;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let removed = w.abandon_coins_at_address(&addr)?;
        if removed > 0 {
            let _ = self.events.send(WalletEvent::WalletChanged);
        }
        Ok(removed)
    }

    fn import_coin_cmd(
        &mut self,
        seed_hex: &str,
        value: u64,
        salt_hex: &str,
        label: Option<String>,
    ) -> Result<String> {
        let seed = parse_hex32(seed_hex).context("seed must be 64 hex characters")?;
        let salt = parse_hex32(salt_hex).context("salt must be 64 hex characters")?;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let coin_id = w.import_coin(seed, value, salt, label)?;
        w.save()?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        Ok(hex::encode(coin_id))
    }

    fn export_coin(&self, id: &str) -> Result<CoinExport> {
        let coin_id = parse_hex32(id)?;
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        let c = w.find_coin(&coin_id).ok_or_else(|| anyhow!("no such coin in this wallet"))?;
        Ok(CoinExport {
            coin_id: hex::encode(c.coin_id),
            address: midstate::core::encode_address_with_checksum(&c.address),
            value: c.value,
            seed: hex::encode(c.seed),
            salt: hex::encode(c.salt),
        })
    }

    // ── Chat ────────────────────────────────────────────────────────────
    // Chat is dictionary-coded: up to ten words, each an index into the
    // node's fixed CHAT_DICTIONARY, with per-message PoW mined BY THE NODE
    // (NodeHandle::send_chat triggers node-side mining). Attachments are the
    // future qbolt channel-message transport; walletd sends none yet.

    fn chat_send(&self, text: &str) -> Result<()> {
        let mut words: Vec<u8> = Vec::new();
        for raw in text.split_whitespace() {
            let idx = midstate::chat::CHAT_DICTIONARY
                .iter()
                .position(|w| {
                    let w: &str = w.as_ref();
                    w.eq_ignore_ascii_case(raw)
                })
                .ok_or_else(|| {
                    anyhow!("\u{201c}{raw}\u{201d} is not in the chat dictionary")
                })?;
            words.push(idx as u8);
        }
        if words.is_empty() {
            bail!("message is empty");
        }
        if words.len() > 10 {
            bail!("messages are at most ten words ({} given)", words.len());
        }
        self.node.send_chat(words, None, vec![])
    }

    async fn chat_history(&self) -> Vec<ChatView> {
        let hist = self.node.chat_history.read().await;
        hist.iter()
            .map(|m| ChatView {
                sender: m.sender.clone(),
                text: m
                    .words
                    .iter()
                    .map(|&i| {
                        midstate::chat::CHAT_DICTIONARY
                            .get(i as usize)
                            .map(|w| -> &str { w.as_ref() })
                            .unwrap_or("?")
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                timestamp: m.timestamp,
                nonce: m.nonce,
                reply_to: m.reply_to,
                attachments: m.attachments.len(),
            })
            .collect()
    }


    // ── Q-Bolt channels ─────────────────────────────────────────────────

    fn ch_notice(&self, text: String) {
        let _ = self.events.send(WalletEvent::ChannelNotice { text });
    }

    /// This wallet's channel identity: the first MSS key (created on demand,
    /// mirroring the web wallet's "primary MSS pk").
    fn ensure_identity(&mut self) -> Result<[u8; 32]> {
        if let Some(pk) = self.book.identity_pk {
            let still_here = self
                .wallet
                .as_ref()
                .map(|w| w.mss_keys().iter().any(|m| m.master_pk == pk))
                .unwrap_or(false);
            if still_here {
                return Ok(pk);
            }
        }
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let pk = if let Some(m) = w.mss_keys().first() {
            m.master_pk
        } else {
            w.generate_mss(DEFAULT_MSS_HEIGHT, Some("qbolt identity".into()))?;
            w.mss_keys().last().expect("just generated").master_pk
        };
        self.book.identity_pk = Some(pk);
        self.book.save(&self.wallet_path);
        Ok(pk)
    }

    fn identity_view(&mut self) -> Result<IdentityView> {
        let pk = self.ensure_identity()?;
        let remaining = self
            .wallet
            .as_ref()
            .and_then(|w| w.mss_keys().iter().find(|m| m.master_pk == pk))
            .map(|m| m.remaining())
            .unwrap_or(0);
        Ok(IdentityView { pk: hex::encode(pk), remaining_sigs: remaining })
    }

    /// Sign a 32-byte commitment with the identity MSS key. The leaf advance
    /// is persisted BEFORE the signature leaves this function — releasing a
    /// signature whose leaf could be reused after a crash breaks the scheme.
    fn sign_commitment(&mut self, msg: &[u8; 32]) -> Result<Vec<u8>> {
        let pk = self.ensure_identity()?;
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        let i = w
            .data
            .mss_keys
            .iter()
            .position(|m| m.master_pk == pk)
            .ok_or_else(|| anyhow!("channel identity key missing from wallet"))?;
        if w.data.mss_keys[i].remaining() <= channels::LEAF_RESERVE {
            bail!(
                "channel identity key is nearly exhausted ({} signatures left, {} reserved for closes) — settle channels before opening new activity",
                w.data.mss_keys[i].remaining(),
                channels::LEAF_RESERVE
            );
        }
        let sig = w.data.mss_keys[i].sign(msg)?;
        w.save().context("failed to persist MSS leaf advance; refusing to release signature")?;
        Ok(sig.to_bytes())
    }

    async fn channel_open(&mut self, peer_hex: &str, amount: u64, lifetime: u64) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing");
        }
        let peer = parse_hex32(peer_hex).context("peer pk must be 64 hex characters")?;
        let me = self.ensure_identity()?;
        if peer == me {
            bail!("cannot open a channel to your own identity");
        }
        if amount < channels::MIN_CAPACITY {
            bail!("channel capacity must be at least {} units", channels::MIN_CAPACITY);
        }
        let lifetime = lifetime.clamp(channels::MIN_LIFETIME, channels::MAX_LIFETIME);
        self.verify_mss_indices().await?;
        let state = self.node.get_state().await;
        let expiry = state.height + lifetime;
        let chan_addr = qb::channel_address(&me, &peer, expiry);

        // Funding outputs with recorded salts — the whole point of the
        // fixed-output planner.
        let mut funding: Vec<qb::FundingCoin> = Vec::new();
        let mut recipient: Vec<midstate::core::OutputData> = Vec::new();
        for denom in midstate::core::decompose_value(amount) {
            let salt: [u8; 32] = rand::random();
            funding.push(qb::FundingCoin { value: denom, salt });
            recipient.push(midstate::core::OutputData::Standard {
                address: chan_addr,
                value: denom,
                salt,
            });
        }
        let id = qb::channel_id(&funding, &chan_addr)?;

        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let live: Vec<[u8; 32]> = w
            .coins()
            .iter()
            .filter(|c| {
                !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id)
            })
            .map(|c| c.coin_id)
            .collect();
        let plan = sendplan::plan_fixed_outputs(w, &live, recipient)?;
        let (commitment, _salt) =
            w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false)?;
        w.save()?;

        // Sign state 0 (everything-minus-fee to the sender) up front — it is
        // what the receiver holds so they can always settle honestly.
        let st0 = qb::build_state(&id, &me, &peer, expiry, &funding, amount - qb::CLOSE_FEE, 0, 0, &[], 0)?;
        let sig0 = self.sign_commitment(&st0.commitment)?;

        self.book.channels.push(ChannelRecord {
            id,
            role: Role::Sender,
            sender_pk: me,
            receiver_pk: peer,
            expiry,
            funding,
            capacity: amount,
            nonce: 0,
            sender_amt: amount - qb::CLOSE_FEE,
            receiver_amt: 0,
            htlcs: Vec::new(),
            sender_sig: sig0,
            pending_claims: Default::default(),
            failed_htlcs: Default::default(),
            acked: false,
            last_broadcast: 0,
            rebroadcasts: 0,
            opened_height: state.height,
            refund_attempt: 0,
            status: ChanStatus::Opening,
        });
        self.book.save(&self.wallet_path);

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "funding payment channel — solving commit proof-of-work".into(),
                amount,
                fee: plan.fee,
                to: hex::encode(chan_addr),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "funding payment channel — solving commit proof-of-work");
        self.broadcast_commit(commitment).await?;
        self.ch_notice(format!(
            "Opening channel {} — funding {} units on-chain, then announcing to the peer.",
            &hex::encode(id)[..12],
            amount
        ));
        Ok(hex::encode(id))
    }

    async fn channel_pay(&mut self, id_hex: &str, amount: u64) -> Result<()> {
        let id = parse_hex32(id_hex)?;
        let tip = self.node.get_state().await.height;
        let (sender_pk, receiver_pk, expiry, funding, nonce, sender_amt, receiver_amt) = {
            let rec = self.book.find(&id).ok_or_else(|| anyhow!("no such channel"))?;
            if rec.role != Role::Sender {
                bail!("only the channel sender can pay");
            }
            if rec.status != ChanStatus::Active {
                bail!("channel is not active");
            }
            if tip + channels::PAY_CUTOFF >= rec.expiry {
                bail!("too close to expiry to pay safely ({} blocks left)", rec.expiry.saturating_sub(tip));
            }
            if amount == 0 || amount > rec.sender_amt {
                bail!("amount must be between 1 and your channel balance ({})", rec.sender_amt);
            }
            (rec.sender_pk, rec.receiver_pk, rec.expiry, rec.funding.clone(), rec.nonce, rec.sender_amt, rec.receiver_amt)
        };
        let _ = (sender_pk, receiver_pk, expiry, funding, nonce);
        let draft = Draft {
            sender_amt: sender_amt - amount,
            receiver_amt: receiver_amt + amount,
            htlcs: self.book.find(&id).map(|r| r.htlcs.clone()).unwrap_or_default(),
        };
        self.sender_advance(id, draft, qb::wire::CMD_UPDATE, &[], tip).await?;
        self.ch_notice(format!("Paid {} units on channel {}.", amount, &id_hex[..12.min(id_hex.len())]));
        Ok(())
    }

    /// Build, sign, persist and transmit the next state on a channel we send
    /// on. Every off-chain balance change funnels through here so the nonce,
    /// signature, saved record and wire frame can never diverge.
    async fn sender_advance(
        &mut self,
        id: [u8; 32],
        draft: Draft,
        cmd: u8,
        route: &[[u8; 32]],
        tip: u64,
    ) -> Result<()> {
        let (sp, rp, expiry, funding, nonce) = {
            let r = self.book.find(&id).ok_or_else(|| anyhow!("no such channel"))?;
            if r.role != Role::Sender {
                bail!("not the sender on this channel");
            }
            (r.sender_pk, r.receiver_pk, r.expiry, r.funding.clone(), r.nonce)
        };
        if draft.htlcs.len() > qb::MAX_HTLCS {
            bail!("too many concurrent HTLCs on this channel");
        }
        let new_nonce = nonce + 1;
        let st = qb::build_state(
            &id, &sp, &rp, expiry, &funding,
            draft.sender_amt, draft.receiver_amt, new_nonce, &draft.htlcs, 0,
        )?;
        let sig = self.sign_commitment(&st.commitment)?;
        if let Some(r) = self.book.find_mut(&id) {
            r.nonce = new_nonce;
            r.sender_amt = draft.sender_amt;
            r.receiver_amt = draft.receiver_amt;
            r.htlcs = draft.htlcs.clone();
            r.sender_sig = sig.clone();
            r.acked = false;
            r.last_broadcast = tip;
            r.rebroadcasts = 0;
        }
        self.book.save(&self.wallet_path);

        // Route hints ride as extra Address attachments; the bus caps a frame
        // at 4 attachments and we already use two.
        let payload = qb::wire::pack_state(&qb::wire::StateWire {
            nonce: new_nonce,
            sender_amt: draft.sender_amt,
            receiver_amt: draft.receiver_amt,
            htlcs: draft.htlcs,
            sig,
        });
        let mut atts = channels::frame_attachments(id, payload, None);
        for pk in route.iter().take(2) {
            atts.push(midstate::chat::ChatAttachment::Address(*pk));
        }
        self.node.send_chat(vec![qb::wire::MARKER, cmd], None, atts)
    }

    async fn channel_close_cmd(&mut self, id_hex: &str) -> Result<()> {
        let id = parse_hex32(id_hex)?;
        let ok = matches!(
            self.book.find(&id).map(|r| (r.role, r.status.clone())),
            Some((Role::Receiver, ChanStatus::Active))
        );
        if !ok {
            bail!("only the receiver of an active channel can close it");
        }
        self.start_close(id).await
    }

    async fn channel_refund_cmd(&mut self, id_hex: &str) -> Result<()> {
        let id = parse_hex32(id_hex)?;
        let tip = self.node.get_state().await.height;
        {
            let rec = self.book.find(&id).ok_or_else(|| anyhow!("no such channel"))?;
            if rec.role != Role::Sender {
                bail!("only the channel sender can refund");
            }
            if tip < rec.expiry {
                bail!("refund unlocks at expiry — {} blocks to go", rec.expiry - tip);
            }
        }
        self.start_refund(id).await
    }

    /// Grind PoW for an externally-built commitment and broadcast the Commit.
    async fn commit_external(&self, commitment: [u8; 32]) -> Result<()> {
        let state = self.node.get_state().await;
        let required = midstate::mempool::Mempool::calculate_required_pow(state.commitments.len());
        let (h, hh) = (state.height, state.header_hash);
        let spam_nonce = tokio::task::spawn_blocking(move || {
            midstate::core::transaction::mine_pow(&commitment, required, h, hh)
        })
        .await
        .context("commit PoW task panicked")?;
        self.node
            .send_transaction(Transaction::Commit { commitment, spam_nonce })
            .await
            .context("commit broadcast failed")
    }

    async fn start_close(&mut self, id: [u8; 32]) -> Result<()> {
        let tip = self.node.get_state().await.height;
        let (sender_pk, receiver_pk, expiry, funding, nonce, sa, ra, ssig, htlcs) = {
            let rec = self.book.find(&id).ok_or_else(|| anyhow!("no such channel"))?;
            (rec.sender_pk, rec.receiver_pk, rec.expiry, rec.funding.clone(), rec.nonce, rec.sender_amt, rec.receiver_amt, rec.sender_sig.clone(), rec.htlcs.clone())
        };
        if ssig.is_empty() {
            bail!("no sender-signed state to close with");
        }
        let st = qb::build_state(&id, &sender_pk, &receiver_pk, expiry, &funding, sa, ra, nonce, &htlcs, 0)?;
        let receiver_sig = self.sign_commitment(&st.commitment)?;
        self.commit_external(st.commitment).await?;
        if let Some(rec) = self.book.find_mut(&id) {
            rec.status = ChanStatus::Closing {
                commitment: st.commitment,
                receiver_sig,
                revealed: false,
                started: tip,
            };
        }
        self.book.save(&self.wallet_path);
        self.ch_notice(format!(
            "Closing channel {} at state {} — commit broadcast, reveal follows once mined.",
            &hex::encode(id)[..12],
            nonce
        ));
        Ok(())
    }

    async fn start_refund(&mut self, id: [u8; 32]) -> Result<()> {
        let tip = self.node.get_state().await.height;
        let (sender_pk, receiver_pk, expiry, funding, attempt) = {
            let rec = self.book.find_mut(&id).ok_or_else(|| anyhow!("no such channel"))?;
            let a = rec.refund_attempt;
            rec.refund_attempt += 1; // persisted before signing: retries must be fresh
            (rec.sender_pk, rec.receiver_pk, rec.expiry, rec.funding.clone(), a)
        };
        self.book.save(&self.wallet_path);
        let st = qb::build_refund_state(&id, &sender_pk, &receiver_pk, expiry, &funding, attempt)?;
        let sender_sig = self.sign_commitment(&st.commitment)?;
        self.commit_external(st.commitment).await?;
        if let Some(rec) = self.book.find_mut(&id) {
            rec.status = ChanStatus::Refunding {
                commitment: st.commitment,
                sender_sig,
                revealed: false,
                started: tip,
            };
        }
        self.book.save(&self.wallet_path);
        self.ch_notice(format!(
            "Refunding expired channel {} — commit broadcast.",
            &hex::encode(id)[..12]
        ));
        Ok(())
    }

    fn channels_list(&self) -> Vec<ChannelView> {
        let tip = self.book.channels.iter().map(|c| c.opened_height).max().unwrap_or(0);
        let _ = tip; // tip comes from sync status on the UI side; blocks_left uses expiry only when sync known
        self.book
            .channels
            .iter()
            .map(|c| {
                let status = match &c.status {
                    ChanStatus::Opening => "opening — funding / awaiting peer ACK".to_string(),
                    ChanStatus::Active => "active".to_string(),
                    ChanStatus::Closing { revealed: false, .. } => "closing — commit pending".into(),
                    ChanStatus::Closing { revealed: true, .. } => "closing — reveal broadcast".into(),
                    ChanStatus::Closed => "closed".into(),
                    ChanStatus::Refunding { revealed: false, .. } => "refunding — commit pending".into(),
                    ChanStatus::Refunding { revealed: true, .. } => "refunding — reveal broadcast".into(),
                    ChanStatus::Refunded => "refunded".into(),
                    ChanStatus::Rejected(r) => format!("rejected: {r}"),
                };
                ChannelView {
                    id: hex::encode(c.id),
                    role: match c.role { Role::Sender => "sender", Role::Receiver => "receiver" }.into(),
                    peer: hex::encode(c.peer_pk(&self.book.identity_pk.unwrap_or([0; 32]))),
                    capacity: c.capacity,
                    sender_amt: c.sender_amt,
                    receiver_amt: c.receiver_amt,
                    my_balance: match c.role { Role::Sender => c.sender_amt, Role::Receiver => c.receiver_amt },
                    nonce: c.nonce,
                    acked: c.acked,
                    htlcs: c
                        .htlcs
                        .iter()
                        .map(|h| HtlcView {
                            hash: hex::encode(h.secret_hash),
                            amount: h.amount,
                            timeout: h.timeout,
                            claiming: c.pending_claims.contains_key(&hex::encode(h.secret_hash)),
                        })
                        .collect(),
                    expiry: c.expiry,
                    blocks_left: c.expiry as i64, // UI subtracts current height
                    status,
                }
            })
            .collect()
    }


    /// Per-tick channel work: process inbound wire frames, verify pending
    /// opens against the chain, drive rebroadcasts and the close/refund
    /// commit→reveal machines, and enforce the expiry autopilot.
    async fn tick_channels(&mut self, tip: u64) -> Result<()> {
        if self.wallet.is_none() {
            return Ok(());
        }
        let hist = self.node.chat_history.read().await.clone();
        let mut dirty = false;

        // ── Inbound frames ──────────────────────────────────────────────
        let frames: Vec<channels::Frame> = hist.iter().filter_map(channels::parse_frame).collect();
        for f in frames {
            if !self.book.mark_seen(f.ts, f.pow_nonce, &f.sender) {
                continue;
            }
            dirty = true;
            if let Err(e) = self.handle_frame(f, tip).await {
                tracing::debug!("channel frame ignored: {e:#}");
            }
        }

        let state = self.node.get_state().await;
        let me = self.book.identity_pk;

        // ── Pending inbound opens: promote once funding is on-chain ────
        let mut promote: Vec<usize> = Vec::new();
        let mut drop_idx: Vec<usize> = Vec::new();
        for (i, p) in self.book.pending_opens.iter().enumerate() {
            let addr = match me {
                Some(m) => qb::channel_address(&p.sender_pk, &m, p.expiry),
                None => continue,
            };
            let all_live = p
                .funding
                .iter()
                .all(|f| state.coins.contains(&midstate::core::compute_coin_id(&addr, f.value, &f.salt)));
            if all_live {
                promote.push(i);
            } else if tip.saturating_sub(p.first_seen) > channels::OPEN_VERIFY_BLOCKS {
                drop_idx.push(i);
            }
        }
        for i in promote.into_iter().rev() {
            let p = self.book.pending_opens.remove(i);
            dirty = true;
            if let Err(e) = self.accept_open(p, tip) {
                tracing::warn!("rejecting inbound channel open: {e:#}");
            }
        }
        for i in drop_idx.into_iter().rev() {
            let p = self.book.pending_opens.remove(i);
            dirty = true;
            self.ch_notice(format!(
                "Ignored a channel open from {} — its funding never appeared on-chain.",
                &hex::encode(p.sender_pk)[..12]
            ));
        }

        // ── Lifecycle decisions (immutable pass), then actions ─────────
        enum Act {
            SendOpen([u8; 32]),
            SendUpdate([u8; 32]),
            AutoClose([u8; 32]),
            AutoRefund([u8; 32]),
            Warn([u8; 32], u64),
            Reveal([u8; 32]),
            Settled([u8; 32]),
            RefundReveal([u8; 32]),
            RefundSettled([u8; 32]),
        }
        let mut acts: Vec<Act> = Vec::new();
        for c in &self.book.channels {
            let first_id = qb::channel_id(&c.funding, &qb::channel_address(&c.sender_pk, &c.receiver_pk, c.expiry));
            match &c.status {
                ChanStatus::Opening if c.role == Role::Sender => {
                    let addr = qb::channel_address(&c.sender_pk, &c.receiver_pk, c.expiry);
                    let live = c.funding.iter().all(|f| {
                        state.coins.contains(&midstate::core::compute_coin_id(&addr, f.value, &f.salt))
                    });
                    if tip >= c.expiry && live {
                        acts.push(Act::AutoRefund(c.id));
                    } else if live
                        && tip.saturating_sub(c.last_broadcast) >= channels::OPEN_REBROADCAST_EVERY
                        && c.rebroadcasts < channels::REBROADCAST_MAX
                    {
                        acts.push(Act::SendOpen(c.id));
                    }
                }
                ChanStatus::Active => {
                    if c.role == Role::Sender {
                        if tip >= c.expiry {
                            acts.push(Act::AutoRefund(c.id));
                        } else if !c.acked
                            && tip.saturating_sub(c.last_broadcast) >= channels::UPDATE_REBROADCAST_EVERY
                            && c.rebroadcasts < channels::REBROADCAST_MAX
                        {
                            acts.push(Act::SendUpdate(c.id));
                        }
                    } else {
                        if tip + channels::CLOSE_MARGIN >= c.expiry {
                            acts.push(Act::AutoClose(c.id));
                        } else if tip + channels::WARN_MARGIN >= c.expiry && !c.acked {
                            // `acked` is repurposed receiver-side as "warned".
                            acts.push(Act::Warn(c.id, c.expiry - tip));
                        }
                    }
                }
                ChanStatus::Closing { commitment, revealed, .. } => {
                    if !*revealed {
                        if self.node.check_commitment(*commitment).await {
                            acts.push(Act::Reveal(c.id));
                        }
                    } else if let Ok(fid) = first_id {
                        if !state.coins.contains(&fid) {
                            acts.push(Act::Settled(c.id));
                        }
                    }
                }
                ChanStatus::Refunding { commitment, revealed, .. } => {
                    if !*revealed {
                        if self.node.check_commitment(*commitment).await {
                            acts.push(Act::RefundReveal(c.id));
                        }
                    } else if let Ok(fid) = first_id {
                        if !state.coins.contains(&fid) {
                            acts.push(Act::RefundSettled(c.id));
                        }
                    }
                }
                _ => {}
            }
        }

        for act in acts {
            dirty = true;
            match act {
                Act::SendOpen(id) => {
                    let (expiry, funding, sig0, me_pk) = {
                        let r = self.book.find(&id).unwrap();
                        (r.expiry, r.funding.clone(), r.sender_sig.clone(), r.sender_pk)
                    };
                    let _ = channels::send_frame(
                        &self.node,
                        qb::wire::CMD_OPEN,
                        id,
                        qb::wire::pack_open(expiry, &funding, &sig0),
                        Some(me_pk),
                    );
                    if let Some(r) = self.book.find_mut(&id) {
                        r.last_broadcast = tip;
                        r.rebroadcasts += 1;
                    }
                }
                Act::SendUpdate(id) => {
                    let (nonce, sa, ra, sig) = {
                        let r = self.book.find(&id).unwrap();
                        (r.nonce, r.sender_amt, r.receiver_amt, r.sender_sig.clone())
                    };
                    let _ = channels::send_frame(
                        &self.node,
                        qb::wire::CMD_UPDATE,
                        id,
                        qb::wire::pack_state(&qb::wire::StateWire {
                            nonce, sender_amt: sa, receiver_amt: ra, htlcs: vec![], sig,
                        }),
                        None,
                    );
                    if let Some(r) = self.book.find_mut(&id) {
                        r.last_broadcast = tip;
                        r.rebroadcasts += 1;
                    }
                }
                Act::AutoClose(id) => {
                    if let Err(e) = self.start_close(id).await {
                        tracing::warn!("auto-close failed: {e:#}");
                    }
                }
                Act::AutoRefund(id) => {
                    if let Err(e) = self.start_refund(id).await {
                        tracing::warn!("auto-refund failed: {e:#}");
                    }
                }
                Act::Warn(id, left) => {
                    self.ch_notice(format!(
                        "Channel {} expires in {} blocks — it will auto-close {} blocks before expiry.",
                        &hex::encode(id)[..12], left, channels::CLOSE_MARGIN
                    ));
                    if let Some(r) = self.book.find_mut(&id) {
                        r.acked = true; // receiver-side: warned once
                    }
                }
                Act::Reveal(id) => {
                    let (sp, rp, expiry, funding, nonce, sa, ra, ssig, rsig, hl) = {
                        let r = self.book.find(&id).unwrap();
                        let rsig = match &r.status {
                            ChanStatus::Closing { receiver_sig, .. } => receiver_sig.clone(),
                            _ => continue,
                        };
                        (r.sender_pk, r.receiver_pk, r.expiry, r.funding.clone(), r.nonce, r.sender_amt, r.receiver_amt, r.sender_sig.clone(), rsig, r.htlcs.clone())
                    };
                    match (|| -> Result<Transaction> {
                        let st = qb::build_state(&id, &sp, &rp, expiry, &funding, sa, ra, nonce, &hl, 0)?;
                        let (inputs, witnesses) = qb::close_reveal(&sp, &rp, expiry, &funding, &st, &ssig, &rsig)?;
                        Ok(Transaction::Reveal { inputs, witnesses, outputs: st.outputs, salt: st.salt })
                    })() {
                        Ok(tx) => {
                            if self.node.send_transaction(tx).await.is_ok() {
                                if let Some(r) = self.book.find_mut(&id) {
                                    if let ChanStatus::Closing { revealed, .. } = &mut r.status {
                                        *revealed = true;
                                    }
                                }
                                self.ch_notice(format!("Channel {} close reveal broadcast.", &hex::encode(id)[..12]));
                            }
                        }
                        Err(e) => tracing::warn!("close reveal build failed: {e:#}"),
                    }
                }
                Act::RefundReveal(id) => {
                    let (sp, rp, expiry, funding, attempt, sig) = {
                        let r = self.book.find(&id).unwrap();
                        let sig = match &r.status {
                            ChanStatus::Refunding { sender_sig, .. } => sender_sig.clone(),
                            _ => continue,
                        };
                        (r.sender_pk, r.receiver_pk, r.expiry, r.funding.clone(), r.refund_attempt.saturating_sub(1), sig)
                    };
                    match (|| -> Result<Transaction> {
                        let st = qb::build_refund_state(&id, &sp, &rp, expiry, &funding, attempt)?;
                        let (inputs, witnesses) = qb::refund_reveal(&sp, &rp, expiry, &funding, &st, &sig)?;
                        Ok(Transaction::Reveal { inputs, witnesses, outputs: st.outputs, salt: st.salt })
                    })() {
                        Ok(tx) => {
                            if self.node.send_transaction(tx).await.is_ok() {
                                if let Some(r) = self.book.find_mut(&id) {
                                    if let ChanStatus::Refunding { revealed, .. } = &mut r.status {
                                        *revealed = true;
                                    }
                                }
                                self.ch_notice(format!("Channel {} refund reveal broadcast.", &hex::encode(id)[..12]));
                            }
                        }
                        Err(e) => tracing::warn!("refund reveal build failed: {e:#}"),
                    }
                }
                Act::Settled(id) => {
                    let nonce = self.book.find(&id).map(|r| r.nonce).unwrap_or(0);
                    if let Some(r) = self.book.find_mut(&id) {
                        r.status = ChanStatus::Closed;
                    }
                    let _ = channels::send_frame(&self.node, qb::wire::CMD_CLOSED, id, qb::wire::pack_u32(nonce, &[]), None);
                    self.ch_notice(format!(
                        "Channel {} settled on-chain — your share arrives via the normal wallet scan.",
                        &hex::encode(id)[..12]
                    ));
                    let _ = self.events.send(WalletEvent::WalletChanged);
                }
                Act::RefundSettled(id) => {
                    if let Some(r) = self.book.find_mut(&id) {
                        r.status = ChanStatus::Refunded;
                    }
                    self.ch_notice(format!(
                        "Channel {} refunded — the full balance (minus fee) returns to your wallet.",
                        &hex::encode(id)[..12]
                    ));
                    let _ = self.events.send(WalletEvent::WalletChanged);
                }
            }
        }

        if dirty {
            self.book.save(&self.wallet_path);
        }
        Ok(())
    }

    async fn handle_frame(&mut self, f: channels::Frame, tip: u64) -> Result<()> {
        use qb::wire as w;
        let me = self.ensure_identity()?;
        let Some(id) = f.channel_id else { return Ok(()) };
        match f.cmd {
            w::CMD_OPEN => {
                let sender_pk = f.address.ok_or_else(|| anyhow!("OPEN without sender pk"))?;
                if sender_pk == me {
                    return Ok(()); // our own broadcast echo
                }
                if let Some(rec) = self.book.find(&id) {
                    if rec.role == Role::Receiver {
                        let _ = channels::send_frame(&self.node, w::CMD_ACK, id, w::pack_u32(rec.nonce, &[]), None);
                    }
                    return Ok(());
                }
                if self.book.pending_opens.iter().any(|p| p.id == id) {
                    return Ok(());
                }
                let payload = f.payload.ok_or_else(|| anyhow!("OPEN without payload"))?;
                let (expiry, funding, sig0) =
                    qb::wire::unpack_open(&payload).ok_or_else(|| anyhow!("unreadable OPEN (version mismatch?)"))?;
                if expiry <= tip + channels::MIN_LIFE_AT_ACCEPT {
                    bail!("open rejected: expires too soon");
                }
                if expiry > tip + channels::MAX_LIFETIME + 1440 {
                    bail!("open rejected: expiry too far out");
                }
                let addr = qb::channel_address(&sender_pk, &me, expiry);
                if qb::channel_id(&funding, &addr)? != id {
                    bail!("open rejected: channel id does not match funding");
                }
                if !self.book.hub.auto_accept {
                    bail!("inbound channel opens are turned off in settings");
                }
                self.book.pending_opens.push(PendingOpen {
                    id, sender_pk, expiry, funding, sig0, first_seen: tip,
                });
                self.ch_notice(format!(
                    "Incoming channel open from {} — verifying its funding on-chain…",
                    &hex::encode(sender_pk)[..12]
                ));
            }
            w::CMD_UPDATE | w::CMD_HTLC_ADD => {
                let payload = f.payload.clone().ok_or_else(|| anyhow!("UPDATE without payload"))?;
                let st = qb::wire::unpack_state(&payload).ok_or_else(|| anyhow!("unreadable state"))?;
                let Some(rec) = self.book.find(&id) else {
                    let _ = channels::send_frame(
                        &self.node, w::CMD_REJECT, id,
                        w::pack_u32(0, &[qb::fail::UNKNOWN_CHANNEL]), None,
                    );
                    return Ok(());
                };
                if rec.role != Role::Receiver || rec.status != ChanStatus::Active {
                    return Ok(());
                }
                if st.nonce <= rec.nonce {
                    // Stale or replayed — re-ACK so a sender waiting on us stops resending.
                    let _ = channels::send_frame(&self.node, w::CMD_ACK, id, w::pack_u32(rec.nonce, &[]), None);
                    return Ok(());
                }
                // Spilman monotonicity: our claimable balance may never shrink.
                // (HTLC adds leave receiver_amt untouched and only reduce the
                // sender's side, so this holds across every legitimate change.)
                if st.receiver_amt < rec.receiver_amt {
                    bail!("state pays the receiver LESS — refusing");
                }
                if st.htlcs.len() > qb::MAX_HTLCS {
                    let _ = channels::send_frame(
                        &self.node, w::CMD_REJECT, id,
                        w::pack_u32(st.nonce, &[qb::fail::NO_ROUTE]), None,
                    );
                    return Ok(());
                }
                let (sp, expiry, funding, prev_recv, prev_htlcs) = (
                    rec.sender_pk, rec.expiry, rec.funding.clone(), rec.receiver_amt, rec.htlcs.clone(),
                );
                let rebuilt = qb::build_state(
                    &id, &sp, &me, expiry, &funding,
                    st.sender_amt, st.receiver_amt, st.nonce, &st.htlcs, 0,
                )?;
                let sig = midstate::core::mss::MssSignature::from_bytes(&st.sig)
                    .map_err(|_| anyhow!("undecodable sender signature"))?;
                if !midstate::core::mss::verify(&sig, &rebuilt.commitment, &sp) {
                    bail!("sender signature does not verify");
                }

                let delta = st.receiver_amt - prev_recv;
                let added: Vec<qb::Htlc> = st
                    .htlcs
                    .iter()
                    .filter(|h| !prev_htlcs.iter().any(|p| p.secret_hash == h.secret_hash))
                    .cloned()
                    .collect();
                let live: Vec<String> =
                    st.htlcs.iter().map(|h| hex::encode(h.secret_hash)).collect();
                if let Some(r) = self.book.find_mut(&id) {
                    r.nonce = st.nonce;
                    r.sender_amt = st.sender_amt;
                    r.receiver_amt = st.receiver_amt;
                    r.htlcs = st.htlcs.clone();
                    r.sender_sig = st.sig;
                    // Anything that left the state is settled: stop tracking it.
                    r.pending_claims.retain(|h, _| live.contains(h));
                    r.failed_htlcs.retain(|h, _| live.contains(h));
                }
                self.book.save(&self.wallet_path); // persist BEFORE the ACK leaves
                let _ = channels::send_frame(&self.node, w::CMD_ACK, id, w::pack_u32(st.nonce, &[]), None);

                if delta > 0 {
                    self.ch_notice(format!(
                        "Received {} units over channel {}.",
                        delta, &hex::encode(id)[..12]
                    ));
                }
                let route: Vec<[u8; 32]> = f.addresses.clone();
                for h in added {
                    if let Err(e) = self.on_htlc_added(id, h, &route, me, tip).await {
                        tracing::warn!("htlc handling failed: {e:#}");
                    }
                }
            }
            w::CMD_HTLC_CLAIM => {
                let payload = f.payload.ok_or_else(|| anyhow!("CLAIM without payload"))?;
                let (_, extra) = qb::wire::unpack_u32(&payload).ok_or_else(|| anyhow!("unreadable CLAIM"))?;
                if extra.len() < 32 {
                    bail!("CLAIM without a hash");
                }
                let hash: [u8; 32] = extra[..32].try_into().unwrap();
                let secret = f.secret.ok_or_else(|| anyhow!("CLAIM without a preimage"))?;
                if qb::hash_bytes(&secret) != hash {
                    bail!("preimage does not match the hash");
                }
                self.on_claim(id, hash, secret, tip).await?;
            }
            w::CMD_HTLC_FAIL => {
                let payload = f.payload.ok_or_else(|| anyhow!("FAIL without payload"))?;
                let (_, extra) = qb::wire::unpack_u32(&payload).ok_or_else(|| anyhow!("unreadable FAIL"))?;
                if extra.len() < 33 {
                    bail!("malformed FAIL");
                }
                let hash: [u8; 32] = extra[..32].try_into().unwrap();
                self.on_fail(id, hash, extra[32], tip).await?;
            }
            w::CMD_INVOICE_REQ => {
                // `id` is an opaque request id minted by the requester, not a channel.
                let target = f.address.ok_or_else(|| anyhow!("invoice request without a target"))?;
                if target != me {
                    return Ok(());
                }
                let payload = f.payload.ok_or_else(|| anyhow!("invoice request without payload"))?;
                let (_, extra) = qb::wire::unpack_u32(&payload).ok_or_else(|| anyhow!("unreadable request"))?;
                if extra.len() < 8 {
                    bail!("invoice request without an amount");
                }
                let amount = u64::from_le_bytes(extra[..8].try_into().unwrap());
                self.answer_invoice_request(id, amount, tip).await?;
            }
            w::CMD_INVOICE => {
                let Some((payee, want)) = self.book.inv_reqs.get(&hex::encode(id)).copied() else {
                    return Ok(());
                };
                let payload = f.payload.ok_or_else(|| anyhow!("invoice without payload"))?;
                let inv = qb::wire::unpack_invoice(&payload).ok_or_else(|| anyhow!("unreadable invoice"))?;
                if inv.amount != want {
                    bail!("invoice amount does not match the request");
                }
                // The bus is public: without this check anyone could race a
                // forged invoice (their hash, their hints) at our request.
                let commit = qb::invoice_commit(&payee, &inv.hash, inv.amount, inv.expiry, &inv.hints);
                let sig = midstate::core::mss::MssSignature::from_bytes(&inv.sig)
                    .map_err(|_| anyhow!("undecodable invoice signature"))?;
                if !midstate::core::mss::verify(&sig, &commit, &payee) {
                    bail!("invoice signature does not verify — refusing to pay");
                }
                self.book.inv_reqs.remove(&hex::encode(id));
                self.book.save(&self.wallet_path);
                if let Err(e) = self
                    .pay_resolved(payee, inv.hash, inv.amount, inv.expiry, inv.hints, tip)
                    .await
                {
                    self.ch_notice(format!("Payment failed: {e}"));
                }
            }
            w::CMD_ACK => {
                let payload = f.payload.ok_or_else(|| anyhow!("ACK without payload"))?;
                let (n, _) = qb::wire::unpack_u32(&payload).ok_or_else(|| anyhow!("unreadable ACK"))?;
                let mut confirmed = false;
                if let Some(r) = self.book.find_mut(&id) {
                    if r.role == Role::Sender && n >= r.nonce {
                        r.acked = true;
                        if r.status == ChanStatus::Opening {
                            r.status = ChanStatus::Active;
                            confirmed = true;
                        }
                    }
                }
                if confirmed {
                    self.ch_notice(format!(
                        "Channel {} confirmed by the peer — ready to pay.",
                        &hex::encode(id)[..12]
                    ));
                    let peer = self.book.find(&id).map(|r| r.receiver_pk);
                    if let Some(pk) = peer {
                        if let Err(e) = self.deliver_parked(pk, tip).await {
                            tracing::warn!("parked delivery: {e:#}");
                        }
                    }
                }
            }
            w::CMD_CLOSE_REQ => {
                let is_recv_active = matches!(
                    self.book.find(&id).map(|r| (r.role, r.status.clone())),
                    Some((Role::Receiver, ChanStatus::Active))
                );
                if is_recv_active {
                    self.ch_notice(format!("Peer asked to close channel {} — closing.", &hex::encode(id)[..12]));
                    let _ = self.start_close(id).await;
                }
            }
            w::CMD_CLOSED => {
                let mut changed = false;
                if let Some(r) = self.book.find_mut(&id) {
                    if !matches!(r.status, ChanStatus::Closed | ChanStatus::Refunded) {
                        r.status = ChanStatus::Closed;
                        changed = true;
                    }
                }
                if changed {
                    self.ch_notice(format!(
                        "Peer settled channel {} on-chain — your balance returns via the wallet scan.",
                        &hex::encode(id)[..12]
                    ));
                    let _ = self.events.send(WalletEvent::WalletChanged);
                }
            }
            w::CMD_REJECT => {
                let mut note = None;
                if let Some(r) = self.book.find_mut(&id) {
                    if r.status == ChanStatus::Opening {
                        r.status = ChanStatus::Rejected("peer rejected the open".into());
                        note = Some(format!("Channel {} was rejected by the peer.", &hex::encode(id)[..12]));
                    }
                }
                if let Some(n) = note {
                    self.ch_notice(n);
                }
            }
            w::CMD_ADDR_REQ => {
                let target = f.address.ok_or_else(|| anyhow!("address request without a target"))?;
                if target != me {
                    return Ok(());
                }
                self.answer_address_request(id, tip).await?;
            }
            w::CMD_CHAN_REQ => {
                // A peer wants a lane from us. Only the sender can pay in a
                // unidirectional channel, so this is the only way a buyer can
                // get instant settlement from a seller they have not met.
                let requester = f.address.ok_or_else(|| anyhow!("channel request without a key"))?;
                if requester == me {
                    return Ok(());
                }
                let payload = f.payload.ok_or_else(|| anyhow!("channel request without payload"))?;
                let (want, _) = qb::wire::unpack_u32(&payload)
                    .ok_or_else(|| anyhow!("unreadable channel request"))?;
                let want = (want as u64).max(channels::MIN_CAPACITY);

                let decline = |code: u8| qb::wire::pack_u32(code as u32, &[]);
                if !self.book.hub.auto_open_on_request {
                    let _ = channels::send_frame(&self.node, w::CMD_CHAN_DECLINE, id, decline(1), None);
                    self.ch_notice(format!(
                        "{} asked for a payment channel. Turn on \"open channels on request\" \
                         under Channels to accept automatically.",
                        &hex::encode(requester)[..12]
                    ));
                    return Ok(());
                }
                if want > self.book.hub.max_auto_capacity {
                    let _ = channels::send_frame(&self.node, w::CMD_CHAN_DECLINE, id, decline(2), None);
                    return Ok(());
                }
                // Budget guard: a stream of individually-acceptable requests
                // must not be able to drain the wallet.
                let committed: u64 = self
                    .book
                    .channels
                    .iter()
                    .filter(|c| {
                        c.role == Role::Sender
                            && matches!(c.status, ChanStatus::Opening | ChanStatus::Active)
                    })
                    .map(|c| c.capacity)
                    .sum();
                if committed + want > self.book.hub.auto_capacity_budget {
                    let _ = channels::send_frame(&self.node, w::CMD_CHAN_DECLINE, id, decline(3), None);
                    self.ch_notice(
                        "Declined a channel request — it would exceed your automatic capacity \
                         budget."
                            .into(),
                    );
                    return Ok(());
                }
                if self.book.channels.iter().any(|c| {
                    c.role == Role::Sender
                        && c.receiver_pk == requester
                        && matches!(c.status, ChanStatus::Opening | ChanStatus::Active)
                }) {
                    return Ok(()); // already have a lane to them
                }
                match self
                    .channel_open(&hex::encode(requester), want, channels::DEFAULT_LIFETIME)
                    .await
                {
                    Ok(_) => self.ch_notice(format!(
                        "Opening a {} unit channel to {} on request — they can be paid instantly \
                         once it confirms.",
                        want,
                        &hex::encode(requester)[..12]
                    )),
                    Err(e) => {
                        tracing::warn!("auto channel open failed: {e:#}");
                        let _ = channels::send_frame(&self.node, w::CMD_CHAN_DECLINE, id, decline(4), None);
                    }
                }
            }
            w::CMD_CHAN_DECLINE => {
                let payload = f.payload.unwrap_or_default();
                let code = qb::wire::unpack_u32(&payload).map(|(c, _)| c).unwrap_or(0);
                self.ch_notice(format!(
                    "Your channel request was declined: {}.",
                    match code {
                        1 => "the peer does not open channels on request",
                        2 => "the amount was above what they will open automatically",
                        3 => "they have reached their automatic capacity limit",
                        _ => "they could not open one right now",
                    }
                ));
            }
            w::CMD_ADDR => {
                // `id` is the request id we minted, so an unsolicited reply is
                // ignored outright.
                let Some(peer) = self.book.addr_reqs.get(&hex::encode(id)).copied() else {
                    return Ok(());
                };
                let payload = f.payload.ok_or_else(|| anyhow!("address reply without payload"))?;
                let (addr, expiry, sig_bytes) = qb::wire::unpack_address(&payload)
                    .ok_or_else(|| anyhow!("unreadable address reply"))?;
                if expiry <= tip {
                    bail!("address reply has already expired");
                }
                let commit = qb::address_commit(&peer, &id, &addr, expiry);
                let sig = midstate::core::mss::MssSignature::from_bytes(&sig_bytes)
                    .map_err(|_| anyhow!("undecodable signature on address reply"))?;
                // The bus is public: without this, anyone could answer our
                // request with an address of their own and collect the payment.
                if !midstate::core::mss::verify(&sig, &commit, &peer) {
                    bail!("address reply signature does not verify — ignoring");
                }
                let encoded = midstate::core::encode_address_with_checksum(&addr);
                self.book.addr_reqs.remove(&hex::encode(id));
                self.book
                    .peer_addrs
                    .insert(hex::encode(peer), (encoded.clone(), expiry));
                self.book.save(&self.wallet_path);
                let _ = self.events.send(WalletEvent::PeerAddress {
                    peer: hex::encode(peer),
                    address: encoded,
                });
            }
            _ => {} // HTLC / resign / legacy traffic: not handled in this build
        }
        Ok(())
    }

    /// Verify a pending inbound open's sig0 and promote it to an active
    /// receiver-side channel record.
    fn accept_open(&mut self, p: PendingOpen, tip: u64) -> Result<()> {
        let me = self.book.identity_pk.ok_or_else(|| anyhow!("no channel identity"))?;
        let capacity: u64 = p.funding.iter().map(|f| f.value).sum();
        let st0 = qb::build_state(
            &p.id, &p.sender_pk, &me, p.expiry, &p.funding,
            capacity.saturating_sub(qb::CLOSE_FEE), 0, 0, &[], 0,
        )?;
        let sig = midstate::core::mss::MssSignature::from_bytes(&p.sig0)
            .map_err(|_| anyhow!("undecodable open signature"))?;
        if !midstate::core::mss::verify(&sig, &st0.commitment, &p.sender_pk) {
            bail!("open signature does not verify");
        }
        self.book.channels.push(ChannelRecord {
            id: p.id,
            role: Role::Receiver,
            sender_pk: p.sender_pk,
            receiver_pk: me,
            expiry: p.expiry,
            funding: p.funding,
            capacity,
            nonce: 0,
            sender_amt: capacity.saturating_sub(qb::CLOSE_FEE),
            receiver_amt: 0,
            htlcs: Vec::new(),
            sender_sig: p.sig0,
            pending_claims: Default::default(),
            failed_htlcs: Default::default(),
            acked: false, // receiver-side: reused as the expiry-warning latch
            last_broadcast: tip,
            rebroadcasts: 0,
            opened_height: tip,
            refund_attempt: 0,
            status: ChanStatus::Active,
        });
        let _ = channels::send_frame(&self.node, qb::wire::CMD_ACK, p.id, qb::wire::pack_u32(0, &[]), None);
        self.ch_notice(format!(
            "Channel opened to you by {}: {} units of inbound capacity.",
            &hex::encode(self.book.channels.last().unwrap().sender_pk)[..12],
            capacity.saturating_sub(qb::CLOSE_FEE)
        ));
        Ok(())
    }


    // ── HTLC routing ────────────────────────────────────────────────────

    /// Consent to removing an HTLC uncredited, and tell the sender why.
    async fn fail_htlc(&mut self, id: [u8; 32], hash: [u8; 32], code: u8) -> Result<()> {
        let hh = hex::encode(hash);
        if let Some(r) = self.book.find_mut(&id) {
            r.failed_htlcs.insert(hh.clone(), 0);
        }
        self.book.save(&self.wallet_path);
        let mut extra = hash.to_vec();
        extra.push(code);
        channels::send_frame(&self.node, qb::wire::CMD_HTLC_FAIL, id, qb::wire::pack_u32(0, &extra), None)?;
        tracing::info!("HTLC {} failed: {}", &hh[..12], qb::fail::describe(code));
        Ok(())
    }

    /// A new HTLC landed on a channel where we are the receiver. Either it is
    /// for us (reveal the preimage to claim it) or we are a hub (forward the
    /// remainder onward, keeping HOP_FEE).
    async fn on_htlc_added(
        &mut self,
        id: [u8; 32],
        htlc: qb::Htlc,
        route: &[[u8; 32]],
        me: [u8; 32],
        tip: u64,
    ) -> Result<()> {
        let hh = hex::encode(htlc.secret_hash);
        let i_am_dest = route.is_empty() || (route.len() == 1 && route[0] == me);

        if i_am_dest {
            let Some(secret_hex) = self.book.secrets.get(&hh).cloned() else {
                // Not ours — never FAIL here: another device may hold the
                // preimage, and a FAIL would cancel a payment we could claim.
                return Ok(());
            };
            let secret = parse_hex32(&secret_hex)?;
            if let Some(inv) = self.book.invoices.get(&hh) {
                if htlc.amount < inv.amount {
                    // NEVER reveal a preimage below the invoiced amount: a hub
                    // could underpay us and use it to collect the full HTLC
                    // it is holding upstream.
                    let want = inv.amount;
                    self.fail_htlc(id, htlc.secret_hash, qb::fail::UNDERPAID).await?;
                    self.ch_notice(format!(
                        "Refused an underpaid invoice: {} units offered, {} invoiced.",
                        htlc.amount, want
                    ));
                    return Ok(());
                }
            }
            if let Some(r) = self.book.find_mut(&id) {
                r.pending_claims.insert(hh.clone(), tip);
            }
            if let Some(inv) = self.book.invoices.get_mut(&hh) {
                inv.paid = Some(htlc.amount);
            }
            self.book.save(&self.wallet_path);
            let nonce = self.book.find(&id).map(|r| r.nonce).unwrap_or(0);
            let extra = htlc.secret_hash.to_vec();
            let atts = {
                let mut a = channels::frame_attachments(id, qb::wire::pack_u32(nonce, &extra), None);
                a.push(midstate::chat::ChatAttachment::Midstate(secret));
                a
            };
            self.node.send_chat(vec![qb::wire::MARKER, qb::wire::CMD_HTLC_CLAIM], None, atts)?;
            self.ch_notice(format!("Payment of {} units received — claiming.", htlc.amount));
            return Ok(());
        }

        // ── Hub: forward toward the next hop ────────────────────────────
        if !self.book.hub.forward {
            return self.fail_htlc(id, htlc.secret_hash, qb::fail::NO_ROUTE).await;
        }
        let skip_self = route[0] == me && route.len() > 1;
        let next_pk = if skip_self { route[1] } else { route[0] };
        let remaining: Vec<[u8; 32]> =
            route.iter().skip(if skip_self { 2 } else { 1 }).copied().collect();

        let out_amt = match htlc.amount.checked_sub(qb::HOP_FEE) {
            Some(v) if v > 0 => v,
            _ => return self.fail_htlc(id, htlc.secret_hash, qb::fail::FEE_EXCEEDS_AMOUNT).await,
        };
        let down_timeout = match htlc.timeout.checked_sub(qb::HTLC_HOP_DELTA) {
            Some(t) if t >= tip + qb::HTLC_MIN_HEADROOM => t,
            _ => return self.fail_htlc(id, htlc.secret_hash, qb::fail::TIMEOUT_TOO_TIGHT).await,
        };
        // Forwarding burns two one-time signatures and permanently consumes
        // outbound capacity; protect the identity key's remaining budget.
        if self.identity_remaining() <= self.book.hub.min_leaves {
            self.ch_notice(
                "Declined to forward a payment — the channel identity key is running low on signatures.".into(),
            );
            return self.fail_htlc(id, htlc.secret_hash, qb::fail::NO_ROUTE).await;
        }

        let fwd = self.book.channels.iter().find(|c| {
            c.role == Role::Sender
                && c.status == ChanStatus::Active
                && c.acked
                && c.receiver_pk == next_pk
                && c.sender_amt >= out_amt
                && tip + channels::PAY_CUTOFF < c.expiry
                && down_timeout <= c.expiry + qb::HTLC_MAX_PAST_EXPIRY
        }).map(|c| (c.id, c.sender_amt, c.receiver_amt, c.htlcs.clone()));

        if let Some((fid, sa, ra, mut hl)) = fwd {
            hl.push(qb::Htlc { amount: out_amt, timeout: down_timeout, secret_hash: htlc.secret_hash });
            let draft = Draft { sender_amt: sa - out_amt, receiver_amt: ra, htlcs: hl };
            match self.sender_advance(fid, draft, qb::wire::CMD_HTLC_ADD, &remaining, tip).await {
                Ok(()) => {
                    self.book.routes.insert(
                        hh,
                        channels::Route { upstream: id, in_amount: htlc.amount, created: tip },
                    );
                    self.book.save(&self.wallet_path);
                    self.ch_notice(format!(
                        "Forwarded {} units toward {} (fee {}).",
                        out_amt, &hex::encode(next_pk)[..12], qb::HOP_FEE
                    ));
                }
                Err(e) => {
                    tracing::warn!("forward failed: {e:#}");
                    self.fail_htlc(id, htlc.secret_hash, qb::fail::FORWARD_FAILED).await?;
                }
            }
            return Ok(());
        }

        // ── Just-in-time open: the last mile has no channel ─────────────
        let already = self.book.channels.iter().any(|c| {
            c.role == Role::Sender && c.receiver_pk == next_pk
                && matches!(c.status, ChanStatus::Opening | ChanStatus::Active)
        });
        let can_jit = self.book.hub.jit_open
            && !already
            && remaining.is_empty()
            && !self.book.parked.contains_key(&hh)
            && htlc.timeout.saturating_sub(tip)
                >= qb::HTLC_MIN_HEADROOM + qb::HTLC_HOP_DELTA + channels::JIT_MARGIN;
        if !can_jit {
            let code = if already { qb::fail::FORWARD_FAILED } else { qb::fail::NO_ROUTE };
            return self.fail_htlc(id, htlc.secret_hash, code).await;
        }

        self.book.parked.insert(
            hh.clone(),
            channels::Parked {
                next_pk,
                amount: out_amt,
                timeout: down_timeout,
                upstream: id,
                in_amount: htlc.amount,
                created: tip,
                remaining: remaining.clone(),
            },
        );
        self.book.save(&self.wallet_path);
        let capacity = (out_amt + qb::CLOSE_FEE)
            .max(channels::MIN_CAPACITY)
            .max(self.book.hub.jit_capacity);
        self.ch_notice(format!(
            "No channel to {} — funding one just-in-time to route {} units.",
            &hex::encode(next_pk)[..12], out_amt
        ));
        if let Err(e) = self
            .channel_open(&hex::encode(next_pk), capacity, channels::DEFAULT_LIFETIME)
            .await
        {
            tracing::warn!("JIT open failed: {e:#}");
            self.book.parked.remove(&hh);
            self.book.save(&self.wallet_path);
            self.fail_htlc(id, htlc.secret_hash, qb::fail::FORWARD_FAILED).await?;
        }
        Ok(())
    }

    /// A preimage arrived. Credit it downstream if we sent that HTLC, and
    /// pull it upstream if we are a hub that forwarded it.
    async fn on_claim(
        &mut self,
        id: [u8; 32],
        hash: [u8; 32],
        secret: [u8; 32],
        tip: u64,
    ) -> Result<()> {
        let hh = hex::encode(hash);
        // Persist the preimage: a hub that gets force-closed still needs it to
        // sweep its upstream HTLC coin on-chain.
        self.book.secrets.entry(hh.clone()).or_insert_with(|| hex::encode(secret));

        // Downstream credit: we are the sender holding this HTLC.
        let mine = self.book.find(&id).and_then(|r| {
            if r.role != Role::Sender {
                return None;
            }
            r.htlcs.iter().find(|h| h.secret_hash == hash).map(|h| {
                (h.amount, r.sender_amt, r.receiver_amt, r.htlcs.clone())
            })
        });
        if let Some((amt, sa, ra, hl)) = mine {
            let htlcs: Vec<qb::Htlc> = hl.into_iter().filter(|h| h.secret_hash != hash).collect();
            let draft = Draft { sender_amt: sa, receiver_amt: ra + amt, htlcs };
            if let Err(e) = self.sender_advance(id, draft, qb::wire::CMD_UPDATE, &[], tip).await {
                tracing::warn!("claim credit failed: {e:#}");
            }
            if let Some(p) = self.book.pay_pending.remove(&hh) {
                self.ch_notice(format!(
                    "Payment of {} units to {} completed — preimage {} is your receipt.",
                    p.amount, &hex::encode(p.dest)[..12], &hex::encode(secret)[..12]
                ));
            }
        }

        // Upstream pull: we forwarded this, so collect from whoever sent it.
        if let Some(route) = self.book.routes.remove(&hh) {
            if route.upstream != id {
                let up = self.book.find(&route.upstream).map(|r| (r.role, r.status.clone(), r.nonce));
                if let Some((Role::Receiver, ChanStatus::Active, nonce)) = up {
                    if let Some(r) = self.book.find_mut(&route.upstream) {
                        r.pending_claims.insert(hh.clone(), tip);
                    }
                    let mut atts = channels::frame_attachments(
                        route.upstream,
                        qb::wire::pack_u32(nonce, &hash),
                        None,
                    );
                    atts.push(midstate::chat::ChatAttachment::Midstate(secret));
                    let _ = self
                        .node
                        .send_chat(vec![qb::wire::MARKER, qb::wire::CMD_HTLC_CLAIM], None, atts);
                }
            }
        }
        self.book.save(&self.wallet_path);
        Ok(())
    }

    /// An HTLC we sent was refused: cancel it (the peer consented to the
    /// uncredited removal), then propagate the failure to whoever sent it to us.
    async fn on_fail(&mut self, id: [u8; 32], hash: [u8; 32], code: u8, tip: u64) -> Result<()> {
        let hh = hex::encode(hash);
        let mine = self.book.find(&id).and_then(|r| {
            if r.role != Role::Sender {
                return None;
            }
            r.htlcs.iter().find(|h| h.secret_hash == hash).map(|h| {
                (h.amount, r.sender_amt, r.receiver_amt, r.htlcs.clone())
            })
        });
        if let Some((amt, sa, ra, hl)) = mine {
            let htlcs: Vec<qb::Htlc> = hl.into_iter().filter(|h| h.secret_hash != hash).collect();
            let draft = Draft { sender_amt: sa + amt, receiver_amt: ra, htlcs };
            if let Err(e) = self.sender_advance(id, draft, qb::wire::CMD_UPDATE, &[], tip).await {
                tracing::warn!("fail cancel deferred: {e:#}");
            }
        }
        self.book.parked.remove(&hh);
        if let Some(route) = self.book.routes.remove(&hh) {
            if route.upstream != id {
                let active = matches!(
                    self.book.find(&route.upstream).map(|r| (r.role, r.status.clone())),
                    Some((Role::Receiver, ChanStatus::Active))
                );
                if active {
                    self.fail_htlc(route.upstream, hash, qb::fail::DOWNSTREAM_FAILED).await?;
                }
            }
        }
        if let Some(p) = self.book.pay_pending.remove(&hh) {
            self.ch_notice(format!(
                "Payment of {} units failed ({}) — the balance was returned.",
                p.amount,
                qb::fail::describe(code)
            ));
        }
        self.book.save(&self.wallet_path);
        Ok(())
    }


    // ── Invoices & paying ───────────────────────────────────────────────

    fn identity_remaining(&self) -> u64 {
        self.book
            .identity_pk
            .and_then(|pk| {
                self.wallet
                    .as_ref()
                    .and_then(|w| w.mss_keys().iter().find(|m| m.master_pk == pk))
                    .map(|m| m.remaining())
            })
            .unwrap_or(0)
    }

    /// Retire the current identity key and start a fresh one. Existing
    /// channels keep working — they are bound to the old key and settle with
    /// it (which is why the reserve exists) — but new channels and invoices
    /// use the new identity.
    fn rotate_identity(&mut self) -> Result<String> {
        let open = self
            .book
            .channels
            .iter()
            .filter(|c| matches!(c.status, ChanStatus::Opening | ChanStatus::Active))
            .count();
        let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
        w.generate_mss(DEFAULT_MSS_HEIGHT, Some("qbolt identity".into()))?;
        let pk = w.mss_keys().last().expect("just generated").master_pk;
        w.save()?;
        self.book.identity_pk = Some(pk);
        self.book.save(&self.wallet_path);
        self.ch_notice(format!(
            "New channel identity in use. {} existing channel(s) keep settling with the old key.",
            open
        ));
        Ok(hex::encode(pk))
    }

    /// Mint an invoice: fresh preimage, recorded expected amount (the
    /// underpay guard), and route hints naming hubs that hold outbound
    /// capacity toward us, best-funded first.
    fn mint_invoice(&mut self, amount: u64, tip: u64) -> Result<(([u8; 32], u64), Vec<[u8; 32]>)> {
        if amount == 0 {
            bail!("invoice amount must be positive");
        }
        self.book.invoices.retain(|_, i| i.expiry == 0 || i.expiry > tip);
        if self.book.invoices.len() > channels::MAX_OUTSTANDING_INVOICES {
            bail!("too many outstanding invoices");
        }
        let secret: [u8; 32] = rand::random();
        let hash = qb::hash_bytes(&secret);
        let expiry = tip + channels::INVOICE_TTL;

        let mut hints: Vec<(u64, [u8; 32])> = self
            .book
            .channels
            .iter()
            .filter(|c| {
                c.role == Role::Receiver
                    && c.status == ChanStatus::Active
                    && c.sender_amt >= amount
                    && tip + channels::PAY_CUTOFF + qb::HTLC_MIN_HEADROOM < c.expiry
            })
            .map(|c| (c.sender_amt, c.sender_pk))
            .collect();
        hints.sort_by(|a, b| b.0.cmp(&a.0));
        let hints: Vec<[u8; 32]> = hints.into_iter().take(2).map(|(_, pk)| pk).collect();

        self.book.secrets.insert(hex::encode(hash), hex::encode(secret));
        self.book.invoices.insert(
            hex::encode(hash),
            channels::Invoice { amount, expiry, hints: hints.clone(), paid: None },
        );
        self.book.save(&self.wallet_path);
        Ok(((hash, expiry), hints))
    }

    async fn create_invoice(&mut self, amount: u64) -> Result<InvoiceView> {
        let tip = self.node.get_state().await.height;
        let me = self.ensure_identity()?;
        let ((hash, expiry), hints) = self.mint_invoice(amount, tip)?;
        Ok(InvoiceView {
            text: format!(
                "l2inv1:{}:{}:{}:{}:{}",
                hex::encode(me),
                hex::encode(hash),
                amount,
                expiry,
                hints.iter().map(hex::encode).collect::<Vec<_>>().join(",")
            ),
            hash: hex::encode(hash),
            amount,
            expiry,
            hints: hints.iter().map(hex::encode).collect(),
            paid: None,
        })
    }

    /// Someone asked us for an invoice over the bus. Answering costs one
    /// one-time signature, so the replay guard has to be durable.
    async fn answer_invoice_request(&mut self, req_id: [u8; 32], amount: u64, tip: u64) -> Result<()> {
        let key = hex::encode(req_id);
        if self.book.answered_reqs.contains_key(&key) {
            return Ok(());
        }
        if self.identity_remaining() <= channels::LEAF_RESERVE + 1 {
            bail!("identity key nearly exhausted — not answering invoice requests");
        }
        if self.book.answered_reqs.len() > 200 {
            self.book.answered_reqs.clear();
        }
        self.book.answered_reqs.insert(key, tip);
        let me = self.ensure_identity()?;
        let ((hash, expiry), hints) = self.mint_invoice(amount, tip)?;
        let commit = qb::invoice_commit(&me, &hash, amount, expiry, &hints);
        let sig = self.sign_commitment(&commit)?;
        channels::send_frame(
            &self.node,
            qb::wire::CMD_INVOICE,
            req_id,
            qb::wire::pack_invoice(&hash, amount, expiry, &hints, &sig),
            None,
        )?;
        self.ch_notice(format!("Issued an invoice for {amount} units on request."));
        Ok(())
    }

    /// Ask a peer for an invoice, to be paid automatically when it arrives.
    async fn request_invoice(&mut self, payee_hex: &str, amount: u64) -> Result<()> {
        let payee = parse_hex32(payee_hex).context("payee must be 64 hex characters")?;
        let _ = self.ensure_identity()?;
        let req_id: [u8; 32] = rand::random();
        self.book.inv_reqs.insert(hex::encode(req_id), (payee, amount));
        self.book.save(&self.wallet_path);
        let atts = {
            let mut a = channels::frame_attachments(
                req_id,
                qb::wire::pack_u32(0, &amount.to_le_bytes()),
                None,
            );
            a.push(midstate::chat::ChatAttachment::Address(payee));
            a
        };
        self.node.send_chat(vec![qb::wire::MARKER, qb::wire::CMD_INVOICE_REQ], None, atts)?;
        self.ch_notice(format!(
            "Asked {} for an invoice for {} units.",
            &payee_hex[..12.min(payee_hex.len())],
            amount
        ));
        Ok(())
    }

    /// Pay an invoice string of the form `l2inv1:<pk>:<hash>:<amt>:<exp>:<hints>`.
    async fn pay_invoice(&mut self, text: &str) -> Result<()> {
        let p: Vec<&str> = text.trim().split(':').collect();
        let (dest, hash, amount, expiry, hints) = match p.as_slice() {
            ["l2inv", d, h, a] => (*d, *h, a.parse::<u64>().unwrap_or(0), 0u64, Vec::new()),
            ["l2inv1", d, h, a, e, rest @ ..] => {
                let hints = rest
                    .first()
                    .map(|s| s.split(',').filter(|x| x.len() == 64).map(|x| x.to_string()).collect())
                    .unwrap_or_default();
                (*d, *h, a.parse::<u64>().unwrap_or(0), e.parse::<u64>().unwrap_or(0), hints)
            }
            _ => bail!("that does not look like an invoice"),
        };
        let dest = parse_hex32(dest).context("invoice destination is malformed")?;
        let hash = parse_hex32(hash).context("invoice hash is malformed")?;
        if amount == 0 {
            bail!("invoice amount is missing");
        }
        let mut hint_pks = Vec::new();
        for h in hints {
            hint_pks.push(parse_hex32(&h)?);
        }
        let tip = self.node.get_state().await.height;
        if expiry > 0 && tip >= expiry {
            bail!("that invoice has expired — ask for a fresh one");
        }
        if self.book.invoices.contains_key(&hex::encode(hash)) {
            bail!("that is this wallet's own invoice");
        }
        self.pay_resolved(dest, hash, amount, expiry, hint_pks, tip).await
    }

    /// Choose the cheapest viable path and launch the HTLC. Direct channel
    /// first, then via a hinted hub, then via our best hub into a hinted hub.
    async fn pay_resolved(
        &mut self,
        dest: [u8; 32],
        hash: [u8; 32],
        amount: u64,
        _expiry: u64,
        hints: Vec<[u8; 32]>,
        tip: u64,
    ) -> Result<()> {
        let usable: Vec<(([u8; 32], [u8; 32]), u64, u64)> = self
            .book
            .channels
            .iter()
            .filter(|c| {
                c.role == Role::Sender
                    && c.status == ChanStatus::Active
                    && tip + channels::PAY_CUTOFF < c.expiry
            })
            .map(|c| ((c.id, c.receiver_pk), c.sender_amt, c.expiry))
            .collect();

        let mut chosen: Option<([u8; 32], Vec<[u8; 32]>, u64)> = None; // (channel, route, hops)
        if let Some((idpk, _, _)) =
            usable.iter().find(|((_, pk), bal, _)| *pk == dest && *bal >= amount)
        {
            chosen = Some((idpk.0, Vec::new(), 0));
        }
        if chosen.is_none() {
            for h in &hints {
                if let Some((idpk, _, _)) = usable
                    .iter()
                    .find(|((_, pk), bal, _)| pk == h && *bal >= amount + qb::HOP_FEE)
                {
                    chosen = Some((idpk.0, vec![dest], 1));
                    break;
                }
            }
        }
        if chosen.is_none() && !hints.is_empty() {
            let mut cands: Vec<_> = usable
                .iter()
                .filter(|((_, pk), bal, _)| {
                    *pk != dest && !hints.contains(pk) && *bal >= amount + 2 * qb::HOP_FEE
                })
                .collect();
            cands.sort_by(|a, b| b.1.cmp(&a.1));
            if let Some((idpk, _, _)) = cands.first() {
                chosen = Some((idpk.0, vec![hints[0], dest], 2));
            }
        }
        if chosen.is_none() && hints.is_empty() {
            let mut cands: Vec<_> = usable
                .iter()
                .filter(|((_, pk), bal, _)| *pk != dest && *bal >= amount + qb::HOP_FEE)
                .collect();
            cands.sort_by(|a, b| b.1.cmp(&a.1));
            if let Some((idpk, _, _)) = cands.first() {
                chosen = Some((idpk.0, vec![dest], 1));
            }
        }

        let (cid, route, hops) = chosen.ok_or_else(|| {
            anyhow!(
                "no outbound channel can reach that payee — open one to them directly{}",
                if hints.is_empty() {
                    String::new()
                } else {
                    format!(" or to one of their hubs ({})", &hex::encode(hints[0])[..12])
                }
            )
        })?;

        let total = amount + hops * qb::HOP_FEE;
        let timeout = tip + qb::HTLC_MIN_HEADROOM + (hops + 1) * qb::HTLC_HOP_DELTA;
        let (sa, ra, mut hl, cexp) = {
            let c = self.book.find(&cid).ok_or_else(|| anyhow!("channel vanished"))?;
            (c.sender_amt, c.receiver_amt, c.htlcs.clone(), c.expiry)
        };
        if timeout > cexp + qb::HTLC_MAX_PAST_EXPIRY {
            bail!("that channel is too close to expiry to route this payment — open a fresh one");
        }
        if sa < total {
            bail!("insufficient channel balance ({sa} spendable, need {total} including routing fees)");
        }
        hl.push(qb::Htlc { amount: total, timeout, secret_hash: hash });
        let draft = Draft { sender_amt: sa - total, receiver_amt: ra, htlcs: hl };

        self.book.pay_pending.insert(
            hex::encode(hash),
            channels::PayPending { total, amount, dest, timeout, at: tip, channel: cid },
        );
        self.book.save(&self.wallet_path);
        if let Err(e) = self.sender_advance(cid, draft, qb::wire::CMD_HTLC_ADD, &route, tip).await {
            self.book.pay_pending.remove(&hex::encode(hash));
            self.book.save(&self.wallet_path);
            return Err(e);
        }
        self.ch_notice(format!(
            "Paying {} units to {} ({} hop{}, fee {}).",
            amount,
            &hex::encode(dest)[..12],
            hops,
            if hops == 1 { "" } else { "s" },
            hops * qb::HOP_FEE
        ));
        Ok(())
    }

    /// A just-in-time channel came up: deliver any forward parked on it.
    async fn deliver_parked(&mut self, peer: [u8; 32], tip: u64) -> Result<()> {
        let ready: Vec<(String, channels::Parked)> = self
            .book
            .parked
            .iter()
            .filter(|(_, p)| p.next_pk == peer)
            .map(|(h, p)| (h.clone(), p.clone()))
            .collect();
        for (hh, p) in ready {
            let hash = parse_hex32(&hh)?;
            let target = self.book.channels.iter().find(|c| {
                c.role == Role::Sender
                    && c.status == ChanStatus::Active
                    && c.acked
                    && c.receiver_pk == peer
                    && c.sender_amt >= p.amount
            }).map(|c| (c.id, c.sender_amt, c.receiver_amt, c.htlcs.clone()));
            let Some((cid, sa, ra, mut hl)) = target else { continue };
            if p.timeout < tip + qb::HTLC_MIN_HEADROOM {
                self.book.parked.remove(&hh);
                self.fail_htlc(p.upstream, hash, qb::fail::TIMEOUT_TOO_TIGHT).await?;
                continue;
            }
            hl.push(qb::Htlc { amount: p.amount, timeout: p.timeout, secret_hash: hash });
            let draft = Draft { sender_amt: sa - p.amount, receiver_amt: ra, htlcs: hl };
            match self.sender_advance(cid, draft, qb::wire::CMD_HTLC_ADD, &p.remaining, tip).await {
                Ok(()) => {
                    self.book.parked.remove(&hh);
                    self.book.routes.insert(
                        hh,
                        channels::Route { upstream: p.upstream, in_amount: p.in_amount, created: tip },
                    );
                    self.book.save(&self.wallet_path);
                    self.ch_notice(format!(
                        "Just-in-time channel is live — forwarded {} units.",
                        p.amount
                    ));
                }
                Err(e) => {
                    tracing::warn!("parked forward failed: {e:#}");
                    self.book.parked.remove(&hh);
                    self.fail_htlc(p.upstream, hash, qb::fail::FORWARD_FAILED).await?;
                }
            }
        }
        Ok(())
    }

    fn invoice_list(&self) -> Vec<InvoiceView> {
        let me = self.book.identity_pk.unwrap_or([0; 32]);
        let mut v: Vec<InvoiceView> = self
            .book
            .invoices
            .iter()
            .map(|(h, i)| InvoiceView {
                text: format!(
                    "l2inv1:{}:{}:{}:{}:{}",
                    hex::encode(me),
                    h,
                    i.amount,
                    i.expiry,
                    i.hints.iter().map(hex::encode).collect::<Vec<_>>().join(",")
                ),
                hash: h.clone(),
                amount: i.amount,
                expiry: i.expiry,
                hints: i.hints.iter().map(hex::encode).collect(),
                paid: i.paid,
            })
            .collect();
        v.sort_by(|a, b| b.expiry.cmp(&a.expiry));
        v
    }


    /// Ask a peer to fund a channel toward us, so they can pay us instantly.
    async fn request_channel(&mut self, peer_hex: &str, capacity: u64) -> Result<()> {
        let peer = parse_hex32(peer_hex).context("peer key must be 64 hex characters")?;
        let me = self.ensure_identity()?;
        if peer == me {
            bail!("that is this wallet's own identity key");
        }
        if self.book.channels.iter().any(|c| {
            c.role == Role::Receiver
                && c.sender_pk == peer
                && matches!(c.status, ChanStatus::Opening | ChanStatus::Active)
        }) {
            bail!("that peer already has a channel open toward you");
        }
        let req_id: [u8; 32] = rand::random();
        let mut atts = channels::frame_attachments(
            req_id,
            qb::wire::pack_u32(capacity.min(u32::MAX as u64) as u32, &[]),
            None,
        );
        atts.push(midstate::chat::ChatAttachment::Address(me));
        self.node
            .send_chat(vec![qb::wire::MARKER, qb::wire::CMD_CHAN_REQ], None, atts)?;
        self.ch_notice(format!(
            "Asked {} to open a {} unit channel. They fund it, since only the sender can pay.",
            &peer_hex[..12.min(peer_hex.len())],
            capacity
        ));
        Ok(())
    }

    // ── Address rotation ────────────────────────────────────────────────
    // A one-time address dies when it signs, so payers need a way to get a
    // fresh one without an out-of-band round trip. The chat bus carries the
    // request; an MSS signature over `address_commit` is what makes a reply
    // trustworthy on a public medium.

    async fn request_address(&mut self, peer_hex: &str) -> Result<()> {
        let peer = parse_hex32(peer_hex).context("peer key must be 64 hex characters")?;
        let me = self.ensure_identity()?;
        if peer == me {
            bail!("that is this wallet's own identity key");
        }
        let req_id: [u8; 32] = rand::random();
        self.book.addr_reqs.insert(hex::encode(req_id), peer);
        self.book.save(&self.wallet_path);
        let mut atts = channels::frame_attachments(req_id, qb::wire::pack_u32(0, &[]), None);
        atts.push(midstate::chat::ChatAttachment::Address(peer));
        self.node
            .send_chat(vec![qb::wire::MARKER, qb::wire::CMD_ADDR_REQ], None, atts)?;
        self.ch_notice(format!(
            "Asked {} for a fresh receiving address.",
            &peer_hex[..12.min(peer_hex.len())]
        ));
        Ok(())
    }

    /// Answer a peer's request with a brand-new one-time address, signed so
    /// they can prove it came from us.
    async fn answer_address_request(&mut self, req_id: [u8; 32], tip: u64) -> Result<()> {
        let key = hex::encode(req_id);
        if self.book.answered_addr_reqs.contains_key(&key) {
            return Ok(()); // already answered; replies are idempotent
        }
        if self.identity_remaining() <= channels::LEAF_RESERVE + 1 {
            bail!("identity key nearly exhausted — not answering address requests");
        }
        if self.book.answered_addr_reqs.len() > 200 {
            self.book.answered_addr_reqs.clear();
        }
        let me = self.ensure_identity()?;
        let expiry = tip + channels::ADDRESS_TTL;
        let addr = {
            let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
            w.generate_key(Some("given out on request".into()))?
        };
        let sig = self.sign_commitment(&qb::address_commit(&me, &req_id, &addr, expiry))?;
        self.book.answered_addr_reqs.insert(key, tip);
        self.book.save(&self.wallet_path);
        channels::send_frame(
            &self.node,
            qb::wire::CMD_ADDR,
            req_id,
            qb::wire::pack_address(&addr, expiry, &sig),
            None,
        )?;
        let _ = self.events.send(WalletEvent::WalletChanged);
        self.ch_notice("Gave a peer a fresh one-time address on request.".into());
        Ok(())
    }


    // ── Cross-chain DEX (read-only) ─────────────────────────────────────

    /// The Base account derived from this wallet's recovery phrase.
    async fn evm_account(&self) -> EvmAccountView {
        let secret = self.wallet.as_ref().and_then(|w| w.data.evm_secret);
        let (address, missing_key) = match secret.and_then(|s| crate::evm::EvmKey::from_secret(&s).ok()) {
            Some(k) => (k.checksum_address(), false),
            // Wallets created before the field existed have no EVM key. It
            // cannot be recovered without the phrase, which is deliberate.
            None => ("—".to_string(), true),
        };
        let balance_wei = match (secret, BaseClient::new(self.dex_cfg.clone())) {
            (Some(s), Ok(c)) => {
                if let Ok(k) = crate::evm::EvmKey::from_secret(&s) {
                    c.balance(&k.address).await.ok().map(|b| b.to_string())
                } else {
                    None
                }
            }
            _ => None,
        };
        EvmAccountView {
            address,
            balance_wei,
            chain_id: self.dex_cfg.chain_id,
            rpc_url: self.dex_cfg.rpc_url.clone(),
            contract: self.dex_cfg.contract.clone(),
            missing_key,
        }
    }

    /// Fold both chains into the book. Nothing here signs or spends.
    async fn sync_order_book(&mut self) -> Result<()> {
        let client = BaseClient::new(self.dex_cfg.clone())?;
        let tip = client.block_number().await?;
        // Stay `confirmations` back: a bid read out of a block that later
        // reorgs away would show as liquidity that does not exist.
        let safe = tip.saturating_sub(self.dex_cfg.confirmations);
        if self.dex.base_cursor == 0 {
            self.dex.base_cursor = if self.dex_start_block > 0 {
                self.dex_start_block.saturating_sub(1)
            } else {
                safe.saturating_sub(self.dex_window)
            };
        }
        self.dex.sync_base(&client, safe).await?;

        let height = self.node.get_state().await.height;
        if self.dex.mds_cursor == 0 {
            // Announcements are only useful while their orders can still be
            // filled, so there is no value in walking the whole chain.
            self.dex.mds_cursor = height.saturating_sub(50_000);
        }
        self.dex.sync_midstate(&self.node, height).await?;
        self.dex.refresh_ask_liveness(&self.node).await?;
        Ok(())
    }

    fn order_book_view(&self) -> OrderBookView {
        let now = now_secs();
        let my_evm = self
            .wallet
            .as_ref()
            .and_then(|w| w.data.evm_secret)
            .and_then(|s| crate::evm::EvmKey::from_secret(&s).ok())
            .map(|k| k.address);

        let bids = self
            .dex
            .sorted_bids(now)
            .into_iter()
            .map(|b| BidView {
                bid_id: hex::encode(b.bid_id),
                maker: crate::evm::to_checksum_address(&b.maker),
                wei: b.amount.to_string(),
                mds_amount: b.mds_amount,
                price: b.wei_per_unit(),
                fill_bond: b.fill_bond.to_string(),
                expiry: b.expiry,
                reserved: b.reserved_by.is_some(),
                takeable: b.is_takeable(now),
                mine: my_evm == Some(b.maker),
            })
            .collect();

        // Inbound lanes: channels where someone else is the sender, so they
        // can push value to us. Anything else is irrelevant to receiving MDS.
        let inbound: Vec<(&[u8; 32], u64)> = self
            .book
            .channels
            .iter()
            .filter(|c| c.role == Role::Receiver && c.status == ChanStatus::Active)
            .map(|c| (&c.sender_pk, c.sender_amt))
            .collect();
        let any_inbound = inbound.iter().any(|(_, cap)| *cap > 0);

        let asks = self
            .dex
            .sorted_asks()
            .into_iter()
            .map(|a| {
                let direct_cap = inbound
                    .iter()
                    .find(|(pk, _)| **pk == a.announcement.maker_mds_pk)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);
                let route = if direct_cap > 0 {
                    "direct"
                } else if any_inbound {
                    // A hub with a lane to us might also have one to the maker.
                    // We cannot see its far side, so this is a possibility, not
                    // a promise.
                    "hub"
                } else {
                    "none"
                }
                .to_string();
                AskView {
                group_id: hex::encode(a.announcement.group_id),
                maker_evm: crate::evm::to_checksum_address(&a.announcement.maker_evm_addr),
                height: a.height,
                timeout_height: a.announcement.timeout_height,
                live_units: a.live_units.len(),
                total_units: a.announcement.units.len(),
                mds_value: a.live_value(),
                wei: a.live_wei().to_string(),
                price: a.wei_per_unit(),
                mine: my_evm == Some(a.announcement.maker_evm_addr),
                units: a
                    .live_units
                    .iter()
                    .enumerate()
                    .filter_map(|(i, u)| {
                        a.announcement.units.get(*u).map(|unit| AskUnitView {
                            index: i,
                            mds: unit.value,
                            wei: unit.wei_amount.to_string(),
                        })
                    })
                    .collect(),
                maker_mds_pk: hex::encode(a.announcement.maker_mds_pk),
                route: route.clone(),
                route_capacity: direct_cap,
            }})
            .collect();

        let st = self.dex.stats;
        OrderBookView {
            bids,
            asks,
            base_cursor: self.dex.base_cursor,
            mds_cursor: self.dex.mds_cursor,
            last_error: self.dex_error.clone(),
            bids_created: st.bids_created,
            bids_closed: st.bids_closed,
            locks: st.locks,
            claims: st.claims,
            undecoded_logs: st.undecoded,
            announcements: st.announcements,
        }
    }


    /// Rebuild history amounts from the chain.
    ///
    /// A `Reveal` publishes `value` on every input and output — consensus
    /// needs it to check conservation — so nothing about a past transaction is
    /// actually lost. It simply is not in `HistoryEntry`, which records only
    /// coin ids. This walks the local block store, matches transactions to
    /// history entries by their spent-input set, and writes the real figures
    /// into the ledger.
    async fn repair_history(&mut self) -> Result<String> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;

        // Every address this wallet controls, so an output can be classified
        // as change (ours) or payment (theirs).
        let mut mine: HashSet<[u8; 32]> = w.coins().iter().map(|c| c.address).collect();
        mine.extend(w.keys().iter().map(|k| k.address));
        mine.extend(
            w.mss_keys().iter().map(|m| midstate::core::compute_address(&m.master_pk)),
        );
        mine.extend(w.watched_addresses());

        // Sends still missing an amount, keyed by their input set.
        let mut wanted: HashMap<String, (Vec<[u8; 32]>, u64)> = HashMap::new();
        let mut earliest = u64::MAX;
        for h in w.history() {
            if h.kind == "sent" && !self.ledger.has_send(&h.inputs) {
                wanted.insert(
                    crate::ledger::input_key(&h.inputs),
                    (h.inputs.clone(), h.fee),
                );
                earliest = earliest.min(h.timestamp);
            } else if h.timestamp > 0 {
                // Received entries may also be unpriced if their coins were
                // spent before the ledger existed.
                let (_, priced) = self.ledger.value_of(&h.outputs);
                if priced < h.outputs.len() {
                    earliest = earliest.min(h.timestamp);
                }
            }
        }
        if earliest == u64::MAX {
            return Ok("Nothing to repair — every transaction already has its amounts.".into());
        }

        let state = self.node.get_state().await;
        let tip = state.height;
        // Blocks target 60s, so convert the oldest gap into a height and add a
        // wide margin rather than walking the whole chain.
        let now = now_secs();
        let age_blocks = now.saturating_sub(earliest) / 60;
        let start = tip.saturating_sub(age_blocks + 2_000);

        let mut fixed_sends = 0usize;
        let mut priced_coins = 0usize;
        let mut mismatched = 0usize;

        for height in start..=tip {
            let Some(batch) = self.node.storage.batches.load(height)? else {
                continue;
            };
            for tx in &batch.transactions {
                let (inputs, outputs) = match tx {
                    Transaction::Reveal { inputs, outputs, .. } => (inputs, outputs),
                    Transaction::Consolidate { inputs, outputs, .. } => (inputs, outputs),
                    _ => continue,
                };

                // Price every output that belongs to us, which repairs
                // receives as well as sends.
                for o in outputs {
                    if let (Some(id), midstate::core::OutputData::Standard { address, value, .. }) =
                        (o.coin_id(), o)
                    {
                        if mine.contains(address) {
                            self.ledger.learn(&id, *value);
                            priced_coins += 1;
                        }
                    }
                }

                let in_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
                let key = crate::ledger::input_key(&in_ids);
                let Some((_, recorded_fee)) = wanted.get(&key).cloned() else {
                    continue;
                };

                let in_sum: u64 = inputs.iter().map(|i| i.value).sum();
                let mut out_sum = 0u64;
                let mut change = 0u64;
                // Outputs that are not ours are the payment. Their `address`
                // IS the address — the same 32 bytes the typed string encodes —
                // so the destination is recoverable, not merely inferable.
                let mut payees: Vec<[u8; 32]> = Vec::new();
                for o in outputs {
                    if let midstate::core::OutputData::Standard { address, value, .. } = o {
                        out_sum += *value;
                        if mine.contains(address) {
                            change += *value;
                        } else if !payees.contains(address) {
                            payees.push(*address);
                        }
                    }
                }

                // Conservation is the proof we matched the right transaction.
                // If the arithmetic disagrees with the fee the wallet recorded,
                // this is not that spend and guessing would be worse than
                // leaving the row blank.
                if in_sum.saturating_sub(out_sum) != recorded_fee {
                    mismatched += 1;
                    continue;
                }

                // A normal send splits into power-of-two denominations all at
                // one address, so this is usually exactly one payee.
                let to = payees
                    .iter()
                    .map(midstate::core::encode_address_with_checksum)
                    .collect::<Vec<_>>()
                    .join(", ");
                self.ledger.record_send(
                    &in_ids,
                    crate::ledger::SendRecord {
                        amount: out_sum.saturating_sub(change),
                        fee: recorded_fee,
                        to,
                        at: batch.timestamp,
                    },
                );
                wanted.remove(&key);
                fixed_sends += 1;
            }

            if height % 5_000 == 0 && height > start {
                self.ledger.save(&self.wallet_path);
            }
        }

        self.ledger.save(&self.wallet_path);
        let _ = self.events.send(WalletEvent::WalletChanged);

        let mut msg = format!(
            "Scanned blocks {}–{}. Recovered {} send amount(s) and priced {} coin(s).",
            start, tip, fixed_sends, priced_coins
        );
        if !wanted.is_empty() {
            msg.push_str(&format!(
                " {} send(s) were not found in that range — they may be older than the scan window.",
                wanted.len()
            ));
        }
        if mismatched > 0 {
            msg.push_str(&format!(
                " {mismatched} candidate(s) failed the conservation check and were skipped."
            ));
        }
        Ok(msg)
    }


    // ── Guided swaps ────────────────────────────────────────────────────

    /// Everything the wallet can tell someone *before* value moves: whether
    /// they are able to do this at all, what the deadlines would be, and what
    /// is going to happen in what order.
    async fn swap_quote(
        &self,
        side_s: &str,
        rail_s: &str,
        mds_amount: u64,
        wei_s: &str,
        peer_mds_pk: &str,
        eth_refund_secs: u64,
    ) -> Result<SwapQuoteView> {
        let side = match side_s {
            "sell" => Side::SellMds,
            _ => Side::BuyMds,
        };
        let rail = match rail_s {
            "onchain" => Rail::OnChain,
            _ => Rail::Submarine,
        };
        let wei: u128 = wei_s.parse().unwrap_or(0);

        let status = self.sync_status().await;
        let tip = status.height;
        let now = now_secs();

        // Deadlines first: if they cannot be made safe there is nothing to
        // quote, and the reason should be the headline.
        let timings = swap::plan_timings(now, tip, eth_refund_secs);
        let (timing_view, timing_error, mds_timeout_height) = match &timings {
            Ok(t) => (
                Some(TimingView {
                    eth_refund_secs: t.eth_refund_secs,
                    eth_deadline: t.eth_deadline,
                    mds_timeout_height: t.mds_timeout_height,
                    mds_deadline_est: t.mds_deadline_est,
                    margin_secs: t.margin_secs,
                }),
                None,
                t.mds_timeout_height,
            ),
            Err(e) => (None, Some(format!("{e:#}")), tip),
        };

        // Gas: buying pays the escrow plus gas, selling only pays gas to claim.
        let gas_estimate: u128 = 300_000 * 50_000_000; // ~0.000015 ETH at 50 gwei
        let wei_needed = match side {
            Side::BuyMds => wei + gas_estimate,
            Side::SellMds => gas_estimate,
        };

        let evm = self.evm_account().await;
        let eth_balance_wei = evm.balance_wei.as_ref().and_then(|b| b.parse::<u128>().ok());

        let mds_spendable = self
            .wallet
            .as_ref()
            .map(|w| {
                w.coins()
                    .iter()
                    .filter(|c| !c.wots_signed)
                    .map(|c| c.value)
                    .sum::<u64>()
            })
            .unwrap_or(0);

        // Channel toward this counterparty, if we have one.
        // Direction matters, and getting it backwards makes every check fail.
        // A channel carries value one way: buying MDS needs an INBOUND lane —
        // the seller pushing to us, so we look at channels where they are the
        // sender and read THEIR spendable balance. Selling needs the opposite.
        let peer = parse_hex32(peer_mds_pk).ok();
        let chan = peer.and_then(|p| {
            self.book
                .channels
                .iter()
                .find(|c| {
                    c.status == ChanStatus::Active
                        && match side {
                            Side::BuyMds => c.role == Role::Receiver && c.sender_pk == p,
                            Side::SellMds => c.role == Role::Sender && c.receiver_pk == p,
                        }
                })
                .map(|c| (c.sender_amt, c.expiry))
        });

        let prereqs = swap::Prereqs {
            side,
            rail,
            synced: !status.is_syncing,
            has_evm_key: !evm.missing_key,
            eth_balance_wei,
            wei_needed,
            mds_spendable,
            mds_needed: mds_amount,
            channel_capacity: chan.map(|c| c.0),
            channel_expiry: chan.map(|c| c.1),
            tip_height: tip,
            mds_timeout_height,
        };
        let checks: Vec<CheckView> = prereqs
            .evaluate()
            .into_iter()
            .map(|c| CheckView { label: c.label, ok: c.ok, detail: c.detail, fix: c.fix })
            .collect();
        let ready = checks.iter().all(|c| c.ok) && timings.is_ok();

        Ok(SwapQuoteView {
            side: side_s.to_string(),
            rail: rail_s.to_string(),
            mds_amount,
            wei_amount: wei.to_string(),
            gas_estimate_wei: gas_estimate.to_string(),
            checks,
            ready,
            timings: timing_view,
            timing_error,
            steps: swap_steps(side, rail),
        })
    }


    // ── Placing orders ──────────────────────────────────────────────────

    /// Publish a sell order.
    ///
    /// Everything happens in ONE transaction, and that is the point: the coins
    /// funding the order and the MDXA announcement describing it are committed
    /// together. A buyer who sees the announcement can therefore verify the
    /// backing coins exist in the same block, and the maker cannot advertise
    /// liquidity that was never funded.
    ///
    /// The order is split into power-of-two units, each with its own secret and
    /// its own price share, so a buyer can take part of it without the maker
    /// having to re-post the remainder.
    async fn place_ask(
        &mut self,
        mds_amount: u64,
        wei_s: &str,
        lifetime_blocks: u64,
    ) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — an order placed against a stale coin set would fail");
        }
        let wei_total: u128 = wei_s.parse().unwrap_or(0);
        if mds_amount == 0 || wei_total == 0 {
            bail!("both the MDS amount and the asking price must be greater than zero");
        }
        let lifetime = lifetime_blocks.clamp(240, 100_000);

        let evm_addr = self
            .wallet
            .as_ref()
            .and_then(|w| w.data.evm_secret)
            .and_then(|s| crate::evm::EvmKey::from_secret(&s).ok())
            .map(|k| k.address)
            .ok_or_else(|| {
                anyhow!("this wallet has no Base account, so buyers would have nowhere to pay")
            })?;

        self.verify_mss_indices().await?;
        let state = self.node.get_state().await;
        let timeout_height = state.height + lifetime;
        let refund_pk = self.ensure_identity()?;

        // Split into power-of-two units, pricing each in proportion to size so
        // a partial fill costs the same per unit as the whole order.
        let denoms = midstate::core::decompose_value(mds_amount);
        if denoms.len() > 32 {
            bail!("that amount splits into {} units — too many for one order", denoms.len());
        }
        let mut units = Vec::with_capacity(denoms.len());
        let mut outputs: Vec<midstate::core::OutputData> = Vec::new();
        let mut secrets: Vec<(String, String)> = Vec::new();
        let mut unit_prices: Vec<(u64, String)> = Vec::new();
        let mut allocated: u128 = 0;

        for (i, value) in denoms.iter().enumerate() {
            let secret: [u8; 32] = rand::random();
            let secret_hash = qb::hash_bytes(&secret);
            let salt: [u8; 32] = rand::random();

            // Last unit absorbs the rounding remainder so the parts sum exactly.
            let wei_amount = if i + 1 == denoms.len() {
                wei_total - allocated
            } else {
                let share = wei_total * (*value as u128) / (mds_amount as u128);
                allocated += share;
                share
            };

            let script = midstate::core::script::compile_limit_order_covenant(
                &secret_hash,
                *value,
                timeout_height,
                &refund_pk,
            );
            let address = midstate::core::types::hash(&script);
            outputs.push(midstate::core::OutputData::Standard { address, value: *value, salt });
            secrets.push((hex::encode(secret_hash), hex::encode(secret)));
            unit_prices.push((*value, wei_amount.to_string()));
            units.push(midstate::core::dex::AnnUnit {
                secret_hash,
                salt,
                value: *value,
                wei_amount,
            });
        }

        let group_id: [u8; 6] = rand::random();
        let ann = midstate::core::dex::MakerAnnouncement {
            maker_evm_addr: evm_addr,
            maker_mds_pk: refund_pk,
            timeout_height,
            group_id,
            units,
        };
        // A self-contained announcement cannot fit one burn, so it rides as
        // fragments — all in this same transaction, landing in one block.
        for frag in midstate::core::dex::fragment(&ann.encode()?, &group_id)? {
            outputs.push(midstate::core::OutputData::DataBurn { payload: frag, value_burned: 0 });
        }

        let in_flight = self.in_flight_inputs();
        let w = self.wallet.as_mut().unwrap();
        let live: Vec<[u8; 32]> = w
            .coins()
            .iter()
            .filter(|c| {
                !c.wots_signed && !in_flight.contains(&c.coin_id) && state.coins.contains(&c.coin_id)
            })
            .map(|c| c.coin_id)
            .collect();
        let plan = sendplan::plan_fixed_outputs(w, &live, outputs)?;
        let (commitment, _salt) =
            w.prepare_commit(&plan.input_coin_ids, &plan.outputs, plan.change_seeds, false, false)?;
        w.save()?;

        // Secrets are the only thing that cannot be rebuilt from the chain:
        // without them the maker can neither be paid nor reclaim early.
        self.book.orders.push(channels::MyOrder {
            group_id: hex::encode(group_id),
            commitment: hex::encode(commitment),
            mds_amount,
            wei_amount: wei_total.to_string(),
            timeout_height,
            created_height: state.height,
            secrets,
            unit_prices,
        });
        self.book.save(&self.wallet_path);

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "publishing sell order — solving commit proof-of-work".into(),
                amount: mds_amount,
                fee: plan.fee,
                to: "sell order".into(),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "publishing sell order");
        self.broadcast_commit(commitment).await?;
        Ok(hex::encode(group_id))
    }

    fn my_orders(&self) -> Vec<MyOrderView> {
        self.book
            .orders
            .iter()
            .map(|o| {
                // Two independent facts: how far the publishing transaction
                // got, and whether the announcement has actually been seen on
                // the chain. They can disagree — the reveal can be mined a
                // moment before the next scan picks it up.
                let commitment = parse_hex32(&o.commitment).ok();
                let meta = commitment.and_then(|c| self.sends.get(&c));
                let (stage, detail) = match meta {
                    Some(m) => (format!("{:?}", m.stage).to_lowercase(), m.detail.clone()),
                    None => ("unknown".into(), "No record of the publishing transaction.".into()),
                };
                let on_chain = self
                    .dex
                    .asks
                    .iter()
                    .any(|a| hex::encode(a.announcement.group_id) == o.group_id);
                MyOrderView {
                    group_id: o.group_id.clone(),
                    mds_amount: o.mds_amount,
                    wei_amount: o.wei_amount.clone(),
                    timeout_height: o.timeout_height,
                    created_height: o.created_height,
                    units: o.secrets.len(),
                    stage,
                    detail,
                    on_chain,
                }
            })
            .collect()
    }


    /// Reclaim the unsold units of an expired order.
    ///
    /// Without this, publishing an order is a one-way door: the coins sit
    /// behind a covenant whose only other exit is a buyer's preimage. The
    /// script's ELSE branch opens after `timeout_height` against the maker's
    /// signature, and this is the path that walks it.
    async fn reclaim_order(&mut self, group_id: &str) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing");
        }
        let order = self
            .book
            .orders
            .iter()
            .find(|o| o.group_id == group_id)
            .cloned()
            .ok_or_else(|| anyhow!("no such order in this wallet"))?;

        let state = self.node.get_state().await;
        if state.height < order.timeout_height {
            bail!(
                "this order does not expire until height {} — {} blocks away. Until then a \
                 buyer can still take it.",
                order.timeout_height,
                order.timeout_height - state.height
            );
        }

        let refund_pk = self.ensure_identity()?;
        let denoms = midstate::core::decompose_value(order.mds_amount);

        // Rebuild each unit's address and keep only those still unspent —
        // anything sold is gone, and asking the chain is the only way to know.
        let mut live: Vec<(usize, [u8; 32], u64, [u8; 32])> = Vec::new();
        let mut recovered = 0u64;
        for (i, (hash_hex, _)) in order.secrets.iter().enumerate() {
            let Some(value) = denoms.get(i).copied() else { continue };
            let secret_hash = parse_hex32(hash_hex)?;
            let addr = midstate::core::dex::limit_order_address(
                &secret_hash,
                value,
                order.timeout_height,
                &refund_pk,
            );
            // Salts were derived per unit at publish time and live in the
            // announcement; recover this unit's from the wallet's own coins.
            let salt = self
                .wallet
                .as_ref()
                .and_then(|w| w.coins().iter().find(|c| c.address == addr).map(|c| c.salt));
            let Some(salt) = salt else { continue };
            let coin = midstate::core::compute_coin_id(&addr, value, &salt);
            if state.coins.contains(&coin) {
                recovered += value;
                live.push((i, secret_hash, value, salt));
            }
        }
        if live.is_empty() {
            bail!("nothing left to reclaim — every unit of this order was sold or already swept");
        }

        // Pay it all back to a fresh reusable address.
        let dest = {
            let w = self.wallet.as_mut().unwrap();
            w.generate_mss(DEFAULT_MSS_HEIGHT, Some("reclaimed order".into()))?
        };
        let fee = (600 + 3000 + 100 + live.len() as u64 * 200) * 10 / 1024 + 20;
        if recovered <= fee {
            bail!("the remaining {recovered} units would not cover the {fee} unit fee to move them");
        }
        let mut outputs = Vec::new();
        for denom in midstate::core::decompose_value(recovered - fee) {
            let salt: [u8; 32] = rand::random();
            outputs.push(midstate::core::OutputData::Standard { address: dest, value: denom, salt });
        }

        let coin_ids: Vec<[u8; 32]> = live
            .iter()
            .map(|(_, h, v, s)| {
                let addr = midstate::core::dex::limit_order_address(h, *v, order.timeout_height, &refund_pk);
                midstate::core::compute_coin_id(&addr, *v, s)
            })
            .collect();

        let w = self.wallet.as_mut().unwrap();
        let (commitment, _salt) = w.prepare_commit(&coin_ids, &outputs, vec![], false, false)?;
        w.save()?;

        self.sends.insert(
            commitment,
            SendMeta {
                stage: SendStage::Committing,
                detail: "reclaiming expired order — solving commit proof-of-work".into(),
                amount: recovered - fee,
                fee,
                to: "reclaimed".into(),
                updated_at: now_secs(),
            },
        );
        self.set_stage(commitment, SendStage::Committing, "reclaiming expired order");
        self.broadcast_commit(commitment).await?;
        Ok(format!(
            "Reclaiming {} unit(s) worth {} from order {}.",
            live.len(),
            recovered - fee,
            &group_id[..8.min(group_id.len())]
        ))
    }


    // ── Taking an order ─────────────────────────────────────────────────

    /// Escrow ETH against one unit of a published sell order.
    ///
    /// This is the taker's only irreversible-ish act, and it is deliberately
    /// the *second* commitment in the swap: the maker's MDS is already locked
    /// and verifiable on-chain before any ETH moves. If the maker then goes
    /// quiet, the escrow refunds itself after `eth_deadline`.
    async fn take_ask(&mut self, group_id: &str, unit: usize) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        if self.node.is_syncing() {
            bail!("node is still syncing — the order's backing coins cannot be verified yet");
        }

        let ask = self
            .dex
            .asks
            .iter()
            .find(|a| hex::encode(a.announcement.group_id) == group_id)
            .cloned()
            .ok_or_else(|| anyhow!("that order is not in the current book — rescan and retry"))?;
        let idx = *ask
            .live_units
            .get(unit)
            .ok_or_else(|| anyhow!("that unit is no longer available"))?;
        let u = ask
            .announcement
            .units
            .get(idx)
            .ok_or_else(|| anyhow!("malformed order"))?
            .clone();

        let state = self.node.get_state().await;
        let now = now_secs();

        // The maker's coins must still be there. An order whose backing has
        // been spent is not an order.
        let addr = midstate::core::dex::limit_order_address(
            &u.secret_hash,
            u.value,
            ask.announcement.timeout_height,
            &ask.announcement.maker_mds_pk,
        );
        let coin = midstate::core::compute_coin_id(&addr, u.value, &u.salt);
        if !state.coins.contains(&coin) {
            bail!("that unit has already been taken");
        }

        // Deadlines: the ETH escrow must expire well before the MDS lock, so
        // that after the maker reveals we still have time to sweep.
        let mds_secs_left = swap::blocks_to_secs_pessimistic(
            ask.announcement.timeout_height.saturating_sub(state.height),
        );
        if mds_secs_left <= swap::SETTLE_MARGIN_SECS * 2 {
            bail!(
                "that order expires too soon to swap against safely — it would leave under \
                 {} minutes to settle",
                swap::SETTLE_MARGIN_SECS / 60
            );
        }
        let eth_refund = (mds_secs_left - swap::SETTLE_MARGIN_SECS).clamp(600, 604_800);
        swap::check_ordering(now + eth_refund, now + mds_secs_left)?;

        let key = self.evm_key()?;
        let client = BaseClient::new(self.dex_cfg.clone())?;

        // Record BEFORE broadcasting. A crash between the two must leave a
        // swap the watcher will investigate, not ETH nobody is tracking.
        let id = hex::encode(&rand::random::<[u8; 8]>());
        self.swaps.swaps.push(Swap {
            id: id.clone(),
            role: crate::swapbook::Role::Taker,
            secret_hash: hex::encode(u.secret_hash),
            preimage: None,
            mds_value: u.value,
            wei: u.wei_amount.to_string(),
            group_id: group_id.to_string(),
            max_claim: u.value,
            mds_timeout_height: ask.announcement.timeout_height,
            refund_pk: hex::encode(ask.announcement.maker_mds_pk),
            salt: hex::encode(u.salt),
            counterparty_evm: crate::evm::to_checksum_address(&ask.announcement.maker_evm_addr),
            eth_deadline: now + eth_refund,
            sweep_dest: None,
            covenant_hex: None,
            phase: Phase::LockingEth { tx: String::new() },
            created: now,
            updated: now,
        });
        self.swaps.save(&self.wallet_path);

        let tx = client
            .lock(
                &key,
                u.secret_hash,
                ask.announcement.maker_evm_addr,
                eth_refund,
                u.wei_amount,
            )
            .await;
        match tx {
            Ok(h) => {
                if let Some(s) = self.swaps.find_mut(&id) {
                    s.phase = Phase::LockingEth { tx: hex::encode(h) };
                    s.updated = now;
                }
                self.swaps.save(&self.wallet_path);
                self.ch_notice(format!(
                    "Escrowed {} wei for {} MDS. Waiting for the seller to claim it, which is \
                     what releases your coins.",
                    u.wei_amount, u.value
                ));
                Ok(hex::encode(h))
            }
            Err(e) => {
                if let Some(s) = self.swaps.find_mut(&id) {
                    s.phase = Phase::Failed { reason: format!("{e:#}") };
                }
                self.swaps.save(&self.wallet_path);
                Err(e)
            }
        }
    }

    fn evm_key(&self) -> Result<crate::evm::EvmKey> {
        let secret = self
            .wallet
            .as_ref()
            .and_then(|w| w.data.evm_secret)
            .ok_or_else(|| anyhow!("this wallet has no Base account"))?;
        crate::evm::EvmKey::from_secret(&secret)
    }

    fn swaps_view(&self) -> Vec<SwapView> {
        self.swaps
            .swaps
            .iter()
            .rev()
            .map(|s| SwapView {
                id: s.id.clone(),
                role: match s.role {
                    crate::swapbook::Role::Taker => "taker",
                    crate::swapbook::Role::Maker => "maker",
                }
                .into(),
                phase: s.phase.label().into(),
                detail: match &s.phase {
                    Phase::Failed { reason } => reason.clone(),
                    Phase::Done { note } => note.clone(),
                    Phase::EthLocked { .. } => {
                        "Your ETH is escrowed. If the seller never claims it, it refunds itself \
                         after the deadline."
                            .into()
                    }
                    Phase::SweepingMds { .. } => {
                        "The secret is out and your MDS is being collected on-chain.".into()
                    }
                    Phase::ConfirmingMds { .. } => {
                        "The collecting transaction is broadcast. Your MDS arrives once it is \
                         mined — usually a block or two."
                            .into()
                    }
                    _ => String::new(),
                },
                mds_value: s.mds_value,
                wei: s.wei.clone(),
                counterparty: s.counterparty_evm.clone(),
                eth_deadline: s.eth_deadline,
                settled: s.phase.settled(),
                tx: match &s.phase {
                    Phase::LockingEth { tx } | Phase::ClaimingEth { tx, .. } | Phase::RefundingEth { tx, .. } => {
                        Some(tx.clone())
                    }
                    _ => None,
                },
            })
            .collect()
    }


    // ── The swap watcher ────────────────────────────────────────────────

    /// Advance every live swap.
    ///
    /// This is the component that makes a swap safe to walk away from. Each
    /// leg has a deadline, and the loss cases are all "nobody acted in time"
    /// rather than "someone cheated" — so the job is simply never to be the
    /// side that failed to act.
    async fn tick_swaps(&mut self, tip: u64) -> Result<()> {
        if self.wallet.is_none() {
            return Ok(());
        }
        // A maker with live orders MUST scan even with no swaps yet — that is
        // precisely how an incoming lock is discovered. Gating on "already
        // watching something" made the maker side unreachable: it could never
        // acquire the first swap it was waiting for.
        let watching = self.swaps.swaps.iter().any(|s| s.active());
        let selling = !self.book.orders.is_empty();
        if !watching && !selling {
            return Ok(());
        }
        let now = now_secs();
        let client = match BaseClient::new(self.dex_cfg.clone()) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("swap watcher cannot reach Base: {e:#}");
                return Ok(());
            }
        };
        let key = self.evm_key()?;

        // 1. Turn pending Base transactions into confirmed state. Swap ids
        //    fold in the mining timestamp, so they only exist post-receipt.
        let pending: Vec<(String, String)> = self
            .swaps
            .swaps
            .iter()
            .filter_map(|s| match &s.phase {
                Phase::LockingEth { tx } if !tx.is_empty() => Some((s.id.clone(), tx.clone())),
                _ => None,
            })
            .collect();
        for (id, tx_hex) in pending {
            let Ok(tx) = parse_hex32(&tx_hex) else { continue };
            match client.receipt(&tx).await {
                Ok(Some(r)) if r.success => {
                    if let Some(swap_id) = crate::base::locked_swap_id(&r.logs) {
                        if let Some(s) = self.swaps.find_mut(&id) {
                            s.phase = Phase::EthLocked { swap_id: hex::encode(swap_id) };
                            s.updated = now;
                        }
                        self.swaps.touch();
                    }
                }
                Ok(Some(_)) => {
                    if let Some(s) = self.swaps.find_mut(&id) {
                        s.phase = Phase::Failed { reason: "the escrow transaction reverted".into() };
                    }
                    self.swaps.touch();
                }
                _ => {}
            }
        }

        // 1b. A freshly created bid has no id until its receipt lands.
        let pending_bids: Vec<(String, String)> = self
            .book
            .bids
            .iter()
            .filter(|b| b.bid_id.is_empty() && !b.tx.is_empty())
            .map(|b| (b.tx.clone(), b.secret_hash.clone()))
            .collect();
        for (tx_hex, hash_hex) in pending_bids {
            let Ok(tx) = parse_hex32(&tx_hex) else { continue };
            if let Ok(Some(r)) = client.receipt(&tx).await {
                if r.success {
                    if let Some(id) = crate::base::created_bid_id(&r.logs) {
                        if let Some(b) =
                            self.book.bids.iter_mut().find(|b| b.secret_hash == hash_hex)
                        {
                            b.bid_id = hex::encode(id);
                        }
                        self.book.save(&self.wallet_path);
                    }
                }
            }
        }

        // 1c. Fill our resting bids.
        //
        // A seller who takes a bid locks MDS in a covenant paying our address
        // and announces it on midstate (MDXT) — which exists precisely because
        // the covenant address cannot be derived from the hashlock alone: it
        // also commits to the seller's timeout and refund key, which only they
        // know. Claiming that covenant is what publishes our secret and lets
        // them collect the ETH, so this step is the whole fill.
        let open_bids: Vec<crate::channels::MyBid> = self
            .book
            .bids
            .iter()
            .filter(|b| !b.cancelled && !b.bid_id.is_empty())
            .cloned()
            .collect();
        if !open_bids.is_empty() {
            let height = self.node.get_state().await.height;
            let _ = self.dex.sync_midstate(&self.node, height).await;
            let state = self.node.get_state().await;
            let locks = self.dex.taker_locks.clone();
            for (_, t) in &locks {
                let hh = hex::encode(t.secret_hash);
                let Some(bid) = open_bids.iter().find(|b| b.secret_hash == hh) else { continue };
                if self.swaps.swaps.iter().any(|s| s.secret_hash == hh) {
                    continue; // already filling
                }
                // The covenant must pay US, and pay enough.
                if hex::encode(t.receiver_addr) != bid.mds_addr {
                    continue;
                }
                if t.value < bid.mds_amount {
                    self.ch_notice(format!(
                        "Ignored a partial fill of your buy order: {} MDS offered against {} asked.",
                        t.value, bid.mds_amount
                    ));
                    continue;
                }

                // `min_payout` is not carried in the announcement, so try the
                // sensible candidates and let the chain say which is real —
                // only the true address holds a live coin.
                let mut found = None;
                for min_payout in [t.value, bid.mds_amount] {
                    let script = midstate::core::script::compile_covenant_htlc(
                        &t.secret_hash,
                        &t.receiver_addr,
                        min_payout,
                        t.timeout_height,
                        &t.taker_mds_pk,
                    );
                    let addr = midstate::core::types::hash(&script);
                    let coin = midstate::core::compute_coin_id(&addr, t.value, &t.salt);
                    if state.coins.contains(&coin) {
                        found = Some(script);
                        break;
                    }
                }
                let Some(script) = found else { continue };

                // Enough time to claim before the seller can reclaim?
                if t.timeout_height <= height + 3 {
                    self.ch_notice(
                        "A fill for your buy order arrived too close to its timeout to claim \
                         safely."
                            .into(),
                    );
                    continue;
                }

                self.swaps.swaps.push(Swap {
                    id: hex::encode(&rand::random::<[u8; 8]>()),
                    role: crate::swapbook::Role::Maker,
                    secret_hash: hh.clone(),
                    preimage: Some(bid.secret.clone()),
                    mds_value: t.value,
                    wei: bid.wei.clone(),
                    group_id: bid.bid_id.clone(),
                    max_claim: t.value,
                    mds_timeout_height: t.timeout_height,
                    refund_pk: hex::encode(t.taker_mds_pk),
                    salt: hex::encode(t.salt),
                    counterparty_evm: String::new(),
                    eth_deadline: bid.expiry,
                    // The covenant only releases if we pay ourselves, so the
                    // destination is fixed by the script.
                    sweep_dest: Some(bid.mds_addr.clone()),
                    covenant_hex: Some(hex::encode(script)),
                    phase: Phase::SweepingMds {
                        preimage: bid.secret.clone(),
                        commitment: String::new(),
                    },
                    created: now,
                    updated: now,
                });
                self.swaps.save(&self.wallet_path);
                self.ch_notice(format!(
                    "Your buy order is being filled — collecting {} MDS, which releases the \
                     secret the seller needs.",
                    t.value
                ));
            }
        }

        // 2. Read new contract events once, and let both roles use them.
        let head = client.block_number().await.unwrap_or(0);
        let safe = head.saturating_sub(self.dex_cfg.confirmations);
        if self.swaps.base_cursor == 0 {
            self.swaps.base_cursor = safe.saturating_sub(5_000);
        }
        let events = if safe > self.swaps.base_cursor {
            let from = self.swaps.base_cursor + 1;
            let got = client.scan_events(from, safe).await.unwrap_or_default();
            self.swaps.base_cursor = safe;
            self.swaps.touch();
            got
        } else {
            Vec::new()
        };

        for (_, ev) in &events {
            match ev {
                // Maker: someone escrowed ETH against one of our order hashes.
                crate::base::Event::Locked { swap_id, beneficiary, amount, hashlock, timeout, .. } => {
                    let hh = hex::encode(hashlock);
                    let mine = self.book.orders.iter().find_map(|o| {
                        o.secrets
                            .iter()
                            .position(|(h, _)| *h == hh)
                            .map(|i| (o.clone(), o.secrets[i].1.clone(), i))
                    });
                    let Some((order, secret, unit_ix)) = mine else { continue };
                    if *beneficiary != key.address {
                        continue; // escrowed to someone else
                    }
                    if self.swaps.by_hash_mut(&hh).is_some() {
                        continue; // already tracking
                    }
                    // Only claim if the money and the clock are both right.
                    // Claiming publishes our secret, so an underpaid lock must
                    // never be taken — it would release that unit for less.
                    //
                    // The comparison is against THIS unit's price. Units are
                    // power-of-two sized and priced proportionally, so an
                    // order-wide average would reject every small unit.
                    let (unit_mds, expected) = match order.unit_prices.get(unit_ix) {
                        Some((m, w)) => (*m, w.parse::<u128>().unwrap_or(u128::MAX)),
                        None => {
                            // Orders published before per-unit prices were kept
                            // locally. The announcement on-chain is
                            // authoritative and carries the real per-unit price,
                            // so read it from there rather than falling back to
                            // an order-wide average — an average rejects every
                            // unit smaller than the mean, which for a
                            // power-of-two split is most of them.
                            let from_chain = self
                                .dex
                                .asks
                                .iter()
                                .find(|a| hex::encode(a.announcement.group_id) == order.group_id)
                                .and_then(|a| {
                                    a.announcement
                                        .units
                                        .iter()
                                        .find(|u| hex::encode(u.secret_hash) == hh)
                                        .map(|u| (u.value, u.wei_amount))
                                });
                            match from_chain {
                                Some((v, w)) => (v, w),
                                None => (
                                    0,
                                    order.wei_amount.parse::<u128>().unwrap_or(u128::MAX)
                                        / order.secrets.len().max(1) as u128,
                                ),
                            }
                        }
                    };
                    if *amount < expected {
                        self.ch_notice(format!(
                            "Ignored an underpaid escrow: {} wei offered for a unit priced at \
                             {} wei.",
                            amount, expected
                        ));
                        continue;
                    }
                    if now + swap::SETTLE_MARGIN_SECS / 2 > *timeout {
                        self.ch_notice(
                            "Ignored an escrow that expires too soon to claim safely.".into(),
                        );
                        continue;
                    }

                    self.swaps.swaps.push(Swap {
                        id: hex::encode(&rand::random::<[u8; 8]>()),
                        role: crate::swapbook::Role::Maker,
                        secret_hash: hh.clone(),
                        preimage: Some(secret.clone()),
                        mds_value: unit_mds,
                        wei: amount.to_string(),
                        group_id: order.group_id.clone(),
                        max_claim: 0,
                        mds_timeout_height: order.timeout_height,
                        refund_pk: String::new(),
                        salt: String::new(),
                        counterparty_evm: String::new(),
                        eth_deadline: *timeout,
                        sweep_dest: None,
                        covenant_hex: None,
                        phase: Phase::ClaimingEth { swap_id: hex::encode(swap_id), tx: String::new() },
                        created: now,
                        updated: now,
                    });
                    self.swaps.save(&self.wallet_path);
                    self.ch_notice(format!(
                        "Someone escrowed {} wei for {} MDS of your order — claiming it now.",
                        amount, unit_mds
                    ));
                }
                // Taker: the maker claimed, which published the preimage.
                crate::base::Event::Claimed { swap_id, preimage, .. } => {
                    let sid = hex::encode(swap_id);
                    let found = self.swaps.swaps.iter_mut().find(|s| {
                        matches!(&s.phase, Phase::EthLocked { swap_id } if *swap_id == sid)
                    });
                    if let Some(s) = found {
                        s.preimage = Some(hex::encode(preimage));
                        s.phase = Phase::SweepingMds {
                            preimage: hex::encode(preimage),
                            commitment: String::new(),
                        };
                        s.updated = now;
                        self.swaps.touch();
                    }
                }
                _ => {}
            }
        }

        // 3. Maker: send the claim that releases the preimage.
        let to_claim: Vec<(String, String, String)> = self
            .swaps
            .swaps
            .iter()
            .filter_map(|s| match (&s.phase, &s.preimage) {
                (Phase::ClaimingEth { swap_id, tx }, Some(p)) if tx.is_empty() => {
                    Some((s.id.clone(), swap_id.clone(), p.clone()))
                }
                _ => None,
            })
            .collect();
        for (id, swap_id, preimage) in to_claim {
            let (Ok(sid), Ok(pre)) = (parse_hex32(&swap_id), parse_hex32(&preimage)) else {
                continue;
            };
            match client.claim(&key, sid, pre).await {
                Ok(h) => {
                    if let Some(s) = self.swaps.find_mut(&id) {
                        s.phase = Phase::Done {
                            note: format!("ETH claimed in {}", hex::encode(h)),
                        };
                        s.updated = now;
                    }
                    self.swaps.touch();
                    self.ch_notice("Claimed the ETH — the trade is complete on your side.".into());
                }
                Err(e) => tracing::warn!("swap claim failed, will retry: {e:#}"),
            }
        }

        // 4. Taker: collect the MDS now that the preimage is public. Like any
        //    spend this is commit-then-reveal, so it takes two passes.
        let _ = tip;
        let sweeping: Vec<Swap> = self
            .swaps
            .swaps
            .iter()
            .filter(|s| matches!(&s.phase, Phase::SweepingMds { .. }))
            .cloned()
            .collect();
        for sw in sweeping {
            let Phase::SweepingMds { preimage, commitment } = sw.phase.clone() else { continue };
            if commitment.is_empty() {
                match self.begin_sweep(&sw).await {
                    Ok(c) => {
                        if let Some(s) = self.swaps.find_mut(&sw.id) {
                            s.phase = Phase::SweepingMds { preimage, commitment: c };
                            s.updated = now;
                        }
                        self.swaps.touch();
                    }
                    Err(e) => tracing::warn!("sweep commit deferred: {e:#}"),
                }
            } else if let Ok(c) = parse_hex32(&commitment) {
                // Only reveal once the commitment is actually in chain state,
                // exactly as the ordinary send machine does.
                if self.node.check_commitment(c).await {
                    let mut fixed = sw.clone();
                    fixed.phase = Phase::SweepingMds { preimage, commitment: commitment.clone() };
                    match self.finish_sweep(&fixed).await {
                        Ok(()) => {
                            // Broadcast, not banked. The coins are only really
                            // ours once the reveal is mined and they appear in
                            // the UTXO set, so say "confirming" until then.
                            if let Some(s) = self.swaps.find_mut(&sw.id) {
                                s.phase = Phase::ConfirmingMds {
                                    commitment: commitment.clone(),
                                    sent_height: tip,
                                };
                                s.updated = now;
                            }
                            self.swaps.touch();
                            self.ch_notice(format!(
                                "Collecting {} MDS — the reveal is broadcast and will confirm \
                                 in a block or two.",
                                sw.mds_value
                            ));
                        }
                        Err(e) => tracing::warn!("sweep reveal deferred: {e:#}"),
                    }
                }
            }
        }

        // 4b. The reveal is out; wait for the coins to actually exist. A
        //     transaction in a mempool is not a transaction in a block.
        let confirming: Vec<Swap> = self
            .swaps
            .swaps
            .iter()
            .filter(|s| matches!(&s.phase, Phase::ConfirmingMds { .. }))
            .cloned()
            .collect();
        if !confirming.is_empty() {
            let state = self.node.get_state().await;
            for sw in confirming {
                let Phase::ConfirmingMds { commitment, sent_height } = sw.phase.clone() else {
                    continue;
                };
                let Ok((_, _, outputs, _)) = self.build_sweep(&sw) else { continue };
                let landed = outputs
                    .iter()
                    .filter_map(|o| o.coin_id())
                    .all(|id| state.coins.contains(&id));
                if landed {
                    if let Some(s) = self.swaps.find_mut(&sw.id) {
                        s.phase = Phase::Done {
                            note: format!("{} MDS collected", sw.mds_value),
                        };
                        s.updated = now;
                    }
                    self.swaps.touch();
                    let _ = self.events.send(WalletEvent::WalletChanged);
                    self.ch_notice(format!("Swap complete — {} MDS collected.", sw.mds_value));
                } else if tip.saturating_sub(sent_height) >= 3 {
                    // Still nothing after a few blocks: rebroadcast rather than
                    // wait indefinitely on a transaction that may have been
                    // dropped. Re-sending an identical reveal is harmless.
                    if self.finish_sweep(&sw).await.is_ok() {
                        if let Some(s) = self.swaps.find_mut(&sw.id) {
                            s.phase = Phase::ConfirmingMds { commitment, sent_height: tip };
                        }
                        self.swaps.touch();
                    }
                }
            }
        }

        // 5. Taker: the maker never claimed, so take the ETH back.
        let to_refund: Vec<(String, String)> = self
            .swaps
            .swaps
            .iter()
            .filter(|s| s.should_refund(now))
            .filter_map(|s| match &s.phase {
                Phase::EthLocked { swap_id } => Some((s.id.clone(), swap_id.clone())),
                _ => None,
            })
            .collect();
        for (id, swap_id) in to_refund {
            let Ok(sid) = parse_hex32(&swap_id) else { continue };
            match client.refund(&key, sid).await {
                Ok(h) => {
                    if let Some(s) = self.swaps.find_mut(&id) {
                        s.phase = Phase::RefundingEth { swap_id, tx: hex::encode(h) };
                        s.updated = now;
                    }
                    self.swaps.touch();
                    self.ch_notice(
                        "The seller did not claim in time — your ETH has been refunded.".into(),
                    );
                }
                Err(e) => tracing::warn!("refund failed, will retry: {e:#}"),
            }
        }

        if self.swaps.is_dirty() {
            self.swaps.save(&self.wallet_path);
        }
        Ok(())
    }

    /// Rebuild the sweep transaction exactly.
    ///
    /// Every value here is derived from the swap record, never randomised: the
    /// commitment made in the first phase covers this precise transaction, so
    /// a reveal that regenerated its salts would hash differently and be
    /// rejected — after the commit has already been paid for.
    fn build_sweep(
        &self,
        sw: &Swap,
    ) -> Result<(
        Vec<midstate::core::types::InputReveal>,
        Vec<midstate::core::types::Witness>,
        Vec<midstate::core::OutputData>,
        [u8; 32],
    )> {
        let preimage = parse_hex32(sw.preimage.as_deref().unwrap_or(""))?;
        let secret_hash = parse_hex32(&sw.secret_hash)?;
        let refund_pk = parse_hex32(&sw.refund_pk)?;
        let salt = parse_hex32(&sw.salt)?;
        let dest = parse_hex32(
            sw.sweep_dest
                .as_deref()
                .ok_or_else(|| anyhow!("sweep destination not chosen yet"))?,
        )?;

        let (input, witness) = match &sw.covenant_hex {
            // A covenant HTLC from a seller filling our bid. Same claim
            // witness — preimage plus branch selector, no signature — but the
            // script is not a limit order, so use the bytecode as stored.
            Some(hexs) => {
                let bytecode = hex::decode(hexs).context("stored covenant is not hex")?;
                if midstate::core::types::hash(&preimage) != secret_hash {
                    bail!("stored preimage does not open this covenant");
                }
                (
                    midstate::core::types::InputReveal {
                        predicate: midstate::core::types::Predicate::Script { bytecode },
                        value: sw.mds_value,
                        salt,
                        commitment: None,
                    },
                    midstate::core::types::Witness::ScriptInputs(vec![
                        preimage.to_vec(),
                        vec![0x01],
                    ]),
                )
            }
            None => midstate::core::dex::limit_claim_input(
                &secret_hash,
                sw.max_claim,
                sw.mds_timeout_height,
                &refund_pk,
                sw.mds_value,
                salt,
                &preimage,
            )?,
        };

        // Fee covers a single-input reveal carrying the covenant bytecode.
        let fee = 60u64;
        if sw.mds_value <= fee {
            bail!("this unit is too small to cover the {fee}-unit sweep fee");
        }
        let mut outputs = Vec::new();
        for (i, denom) in midstate::core::decompose_value(sw.mds_value - fee).into_iter().enumerate() {
            let mut h = Vec::with_capacity(48);
            h.extend_from_slice(b"mds_swap_out_v1");
            h.extend_from_slice(&preimage);
            h.extend_from_slice(&(i as u32).to_le_bytes());
            let out_salt = midstate::core::types::hash(&h);
            outputs.push(midstate::core::OutputData::Standard {
                address: dest,
                value: denom,
                salt: out_salt,
            });
        }

        let mut h = Vec::with_capacity(48);
        h.extend_from_slice(b"mds_swap_sweep_v1");
        h.extend_from_slice(&preimage);
        let tx_salt = midstate::core::types::hash(&h);

        Ok((vec![input], vec![witness], outputs, tx_salt))
    }

    fn sweep_commitment(&self, sw: &Swap) -> Result<[u8; 32]> {
        let (inputs, _, outputs, tx_salt) = self.build_sweep(sw)?;
        let in_ids: Vec<[u8; 32]> = inputs.iter().map(|i| i.coin_id()).collect();
        let out_ids: Vec<[u8; 32]> = outputs.iter().filter_map(|o| o.coin_id()).collect();
        Ok(midstate::core::types::compute_commitment(&in_ids, &out_ids, &tx_salt))
    }

    /// Phase one of collecting the MDS: choose a destination, commit.
    async fn begin_sweep(&mut self, sw: &Swap) -> Result<String> {
        // A covenant HTLC only releases if the spend pays its receiver address,
        // so the destination is dictated by the script rather than chosen. A
        // limit order has no such rule and can go anywhere.
        let dest = match &sw.sweep_dest {
            Some(fixed) => parse_hex32(fixed)?,
            None => {
                let w = self.wallet.as_mut().ok_or_else(|| anyhow!("wallet is locked"))?;
                w.generate_mss(DEFAULT_MSS_HEIGHT, Some("swap proceeds".into()))?
            }
        };
        // Persist the destination before committing to it.
        if let Some(rec) = self.swaps.find_mut(&sw.id) {
            rec.sweep_dest = Some(hex::encode(dest));
        }
        self.swaps.save(&self.wallet_path);

        let mut fixed = sw.clone();
        fixed.sweep_dest = Some(hex::encode(dest));
        let commitment = self.sweep_commitment(&fixed)?;
        self.commit_external(commitment).await?;
        Ok(hex::encode(commitment))
    }

    /// Phase two: the commitment is on-chain, so publish the reveal that
    /// actually moves the coins.
    async fn finish_sweep(&mut self, sw: &Swap) -> Result<()> {
        let (inputs, witnesses, outputs, salt) = self.build_sweep(sw)?;
        self.node
            .send_transaction(Transaction::Reveal { inputs, witnesses, outputs, salt })
            .await
            .context("sweep reveal rejected")?;
        Ok(())
    }


    // ── Placing a buy order ─────────────────────────────────────────────

    /// Escrow ETH in the contract as a resting bid for MDS.
    ///
    /// The direction is inverted from a sell order, and it matters: here WE
    /// generate the secret. A seller who wants the bid locks MDS on midstate
    /// against our hash, we claim that covenant — which publishes the secret —
    /// and the seller then uses it to collect the escrow. So the ETH is only
    /// ever paid out against a secret we chose to release.
    async fn place_bid(
        &mut self,
        mds_amount: u64,
        wei_s: &str,
        ttl_secs: u64,
        bond_s: &str,
    ) -> Result<String> {
        if self.wallet.is_none() {
            bail!("wallet is locked");
        }
        let wei: u128 = wei_s.parse().unwrap_or(0);
        let bond: u128 = bond_s.parse().unwrap_or(0);
        if mds_amount == 0 || wei == 0 {
            bail!("both the MDS amount and the ETH you are offering must be greater than zero");
        }
        if !(3_600..=7_776_000).contains(&ttl_secs) {
            bail!("the contract requires a bid lifetime between 1 hour and 90 days");
        }

        let key = self.evm_key()?;
        let client = BaseClient::new(self.dex_cfg.clone())?;

        // Somewhere for the seller to pay. A fresh reusable address, because a
        // one-time key that gets used elsewhere would strand the fill.
        let mds_addr = {
            let w = self.wallet.as_mut().unwrap();
            w.generate_mss(DEFAULT_MSS_HEIGHT, Some("buy order".into()))?
        };

        // The contract rejects a reused hashlock outright, and reuse would let
        // a stranger collect with an already-public preimage.
        let secret: [u8; 32] = rand::random();
        let secret_hash = qb::hash_bytes(&secret);
        let now = now_secs();

        // Record before broadcasting: the secret is the only irreplaceable
        // part, and ETH must never be escrowed against a hash we cannot open.
        self.book.bids.push(channels::MyBid {
            bid_id: String::new(),
            tx: String::new(),
            secret_hash: hex::encode(secret_hash),
            secret: hex::encode(secret),
            mds_amount,
            wei: wei.to_string(),
            fill_bond: bond.to_string(),
            mds_addr: hex::encode(mds_addr),
            expiry: now + ttl_secs,
            created: now,
            cancelled: false,
        });
        self.book.save(&self.wallet_path);

        let tx = client
            .create_bid(&key, secret_hash, mds_amount, mds_addr, ttl_secs, bond, wei)
            .await?;
        if let Some(b) = self.book.bids.last_mut() {
            b.tx = hex::encode(tx);
        }
        self.book.save(&self.wallet_path);
        self.ch_notice(format!(
            "Buy order escrowed: {} wei for {} MDS. Sellers can now fill it.",
            wei, mds_amount
        ));
        Ok(hex::encode(tx))
    }

    async fn cancel_bid(&mut self, bid_id: &str) -> Result<String> {
        let bid = self
            .book
            .bids
            .iter()
            .find(|b| b.bid_id == bid_id)
            .cloned()
            .ok_or_else(|| anyhow!("no such buy order in this wallet"))?;
        let id = parse_hex32(&bid.bid_id)?;
        let key = self.evm_key()?;
        let client = BaseClient::new(self.dex_cfg.clone())?;
        let tx = client.cancel_bid(&key, id).await?;
        if let Some(b) = self.book.bids.iter_mut().find(|b| b.bid_id == bid_id) {
            b.cancelled = true;
        }
        self.book.save(&self.wallet_path);
        Ok(format!("Cancelling — your ETH returns in {}", hex::encode(tx)))
    }

    fn my_bids_view(&self) -> Vec<MyBidView> {
        self.book
            .bids
            .iter()
            .rev()
            .map(|b| MyBidView {
                bid_id: b.bid_id.clone(),
                tx: b.tx.clone(),
                mds_amount: b.mds_amount,
                wei: b.wei.clone(),
                fill_bond: b.fill_bond.clone(),
                expiry: b.expiry,
                status: if b.cancelled {
                    "cancelled".into()
                } else if b.bid_id.is_empty() {
                    // The contract folds the mining timestamp into the id, so
                    // it does not exist until the transaction is mined.
                    "confirming".into()
                } else {
                    "open".into()
                },
                cancelled: b.cancelled,
            })
            .collect()
    }

    // ── Incoming scan ───────────────────────────────────────────────────

    async fn tick(&mut self) {
        let status = self.sync_status().await;
        let _ = self.events.send(WalletEvent::NodeTick { status: status.clone() });

        if self.wallet.is_none() || status.is_syncing {
            return;
        }
        if let Err(e) = self.tick_channels(status.height).await {
            tracing::warn!("channel tick: {e:#}");
        }
        if let Err(e) = self.tick_swaps(status.height).await {
            tracing::warn!("swap tick: {e:#}");
        }
        let tip = status.height;
        if tip <= self.scan_pos {
            return;
        }
        // `scan_pos` is the first height NOT yet scanned (it is assigned `end`,
        // which is an exclusive bound). Start there — adding 1 would skip it.
        // Valid batch indices are 0..tip, so `end = tip` covers the tip block.
        let start = self.scan_pos;
        let end = tip.min(self.scan_pos + SCAN_CHUNK);

        // Watch targets: HD-derived watch list ∪ every address we already
        // hold coins or MSS keys on (covers sibling sends to used addresses).
        let addrs: Vec<[u8; 32]> = {
            let w = self.wallet.as_ref().unwrap();
            let mut set: HashSet<[u8; 32]> = w.watched_addresses().into_iter().collect();
            set.extend(w.coins().iter().map(|c| c.address));
            set.extend(w.mss_keys().iter().map(|m| midstate::core::compute_address(&m.master_pk)));
            set.into_iter().collect()
        };

        // Storage-backed scan is synchronous — keep it off the actor thread.
        let node = self.node.clone();
        let scan = tokio::task::spawn_blocking(move || node.scan_addresses(&addrs, start, end)).await;

        let found = match scan {
            Ok(Ok(coins)) => coins,
            Ok(Err(e)) => {
                tracing::warn!("scan [{start},{end}] failed: {e:#}");
                return;
            }
            Err(e) => {
                tracing::warn!("scan task panicked: {e}");
                return;
            }
        };

        // Coins arriving at a key we already spent from are quarantined by
        // `import_scanned` (a second signature would leak the key), so they
        // never reach the balance. Say so explicitly rather than dropping them
        // silently — the money is real and the recipient needs to know.
        let mut stranded: Vec<(u64, [u8; 32])> = Vec::new();
        {
            let w = self.wallet.as_ref().unwrap();
            let mss_addrs: HashSet<[u8; 32]> = w
                .mss_keys()
                .iter()
                .map(|m| midstate::core::compute_address(&m.master_pk))
                .collect();
            let burned: HashSet<[u8; 32]> = w
                .coins()
                .iter()
                .filter(|c| c.wots_signed)
                .map(|c| c.address)
                .collect();
            for sc in &found {
                if burned.contains(&sc.address) && !mss_addrs.contains(&sc.address) {
                    stranded.push((sc.value, sc.address));
                }
            }
        }
        for (value, addr) in &stranded {
            let _ = self.events.send(WalletEvent::Warning {
                text: format!(
                    "Payment of {} arrived at {}, a one-time address this wallet has already \
                     spent from. It cannot be recovered — that key can never sign again.",
                    value,
                    &midstate::core::encode_address_with_checksum(addr)[..12]
                ),
            });
        }

        let mut new_ids = Vec::new();
        let mut new_value = 0u64;
        {
            let w = self.wallet.as_mut().unwrap();
            for sc in &found {
                match w.import_scanned(sc.address, sc.value, sc.salt, None) {
                    Ok(Some(id)) => {
                        new_ids.push(id);
                        new_value += sc.value;
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("import of scanned coin failed: {e:#}"),
                }
            }
            if !new_ids.is_empty() {
                w.record_received(new_ids.clone(), status.timestamp);
                if let Err(e) = w.save() {
                    tracing::error!("wallet save after import failed: {e:#}");
                }
            }
        }

        self.scan_pos = end;
        self.persist_scan_pos();

        if !new_ids.is_empty() {
            let _ = self.events.send(WalletEvent::Incoming {
                total_value: new_value,
                count: new_ids.len(),
                height: end,
            });
            let _ = self.events.send(WalletEvent::WalletChanged);
        }
    }

    // ── Scan-position sidecar ───────────────────────────────────────────

    fn scan_pos_path(&self) -> PathBuf {
        self.wallet_path.with_extension("scanpos")
    }
    fn load_scan_pos(&self) -> u64 {
        std::fs::read_to_string(self.scan_pos_path())
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }
    fn persist_scan_pos(&self) {
        let _ = std::fs::write(self.scan_pos_path(), self.scan_pos.to_string());
    }
}

fn parse_hex32(s: &str) -> Result<[u8; 32]> {
    let b = hex::decode(s.trim())?;
    let arr: [u8; 32] = b.as_slice().try_into().map_err(|_| anyhow!("expected 32 bytes"))?;
    Ok(arr)
}

/// Reconstruct (recipient_amount, fee) for a resumed pending commit:
/// change outputs are identified by `change_seeds`; fee = inputs − outputs.
fn reconstruct_meta(
    w: &Wallet,
    inputs: &[[u8; 32]],
    outputs: &[midstate::core::OutputData],
    commitment: &[u8; 32],
) -> (u64, u64) {
    let out_value = |o: &midstate::core::OutputData| match o {
        midstate::core::OutputData::Standard { value, .. } => *value,
        _ => 0,
    };
    let out_sum: u64 = outputs.iter().map(out_value).sum();
    let in_sum: u64 = inputs.iter().filter_map(|id| w.find_coin(id)).map(|c| c.value).sum();
    let change_sum: u64 = w
        .find_pending(commitment)
        .map(|p| {
            p.change_seeds
                .iter()
                .filter_map(|(idx, _)| outputs.get(*idx))
                .map(out_value)
                .sum()
        })
        .unwrap_or(0);
    (out_sum.saturating_sub(change_sum), in_sum.saturating_sub(out_sum))
}

/// A proposed next state for a channel we send on.
struct Draft {
    sender_amt: u64,
    receiver_amt: u64,
    htlcs: Vec<qb::Htlc>,
}

fn chat_dictionary_vec() -> Vec<String> {
    midstate::chat::CHAT_DICTIONARY
        .iter()
        .map(|w| -> String {
            let w: &str = w.as_ref();
            w.to_string()
        })
        .collect()
}

/// The walkthrough shown before a swap starts. Written for someone who has
/// never done one: what happens, in order, and who is waiting on whom.
fn swap_steps(side: Side, rail: Rail) -> Vec<String> {
    let mds_leg = match rail {
        Rail::Submarine => "over your payment channel — instant, nothing touches the chain",
        Rail::OnChain => "as an on-chain lock — two confirmations on a 60-second chain",
    };
    match side {
        Side::SellMds => vec![
            "Your wallet picks a secret and publishes only its hash. Nobody can              reverse it, and both chains lock against that same hash.".into(),
            format!("You lock the MDS {mds_leg}. If the other side never pays, it comes                      straight back to you."),
            "They escrow the ETH on Base against the same hash. You can verify the              amount before doing anything further.".into(),
            "You claim the ETH, which publishes the secret. This is the only moment              either side is committed.".into(),
            "They read the secret and take the MDS. The swap is done — if they never              do, your lock expires and returns to you anyway.".into(),
        ],
        Side::BuyMds => vec![
            "The seller picks a secret and shows you its hash. Both legs lock to it.".into(),
            format!("They lock the MDS {mds_leg}. Your wallet checks the amount and the                      deadline before you commit anything."),
            "You escrow the ETH on Base. It is refundable to you if the swap stalls.".into(),
            "They claim the ETH, publishing the secret in the process.".into(),
            "Your wallet spots the secret and takes the MDS automatically. If the seller              never claims, your ETH refunds itself after the deadline.".into(),
        ],
    }
}
