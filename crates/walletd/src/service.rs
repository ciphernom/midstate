//! The walletd actor. One task owns the open `Wallet`; everything else —
//! Tauri commands, monitors, the scan tick — talks to it via messages.

use crate::api::*;
use crate::channels::{self, ChanStatus, ChannelBook, ChannelRecord, PendingOpen, Role};
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
    Create { password: String, resp: oneshot::Sender<Result<String>> },
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
    pub async fn create(&self, password: String) -> Result<String> {
        ask!(self, Create { password: password })
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
            Cmd::Create { password, resp } => {
                let _ = resp.send(self.create(&password).await);
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
                let _ = resp.send(sendplan::decode_address(&addr).map(|_| ()));
            }
            Cmd::SyncStatus(resp) => {
                let _ = resp.send(self.sync_status().await);
            }
            Cmd::NodeInfo(resp) => {
                let peers = self.node.get_peers().await;
                let state = self.node.get_state().await;
                let _ = resp.send(NodeInfo {
                    peers,
                    data_dir: self.data_dir.display().to_string(),
                    rpc_url: self.rpc_url.clone(),
                    block_reward: midstate::core::block_reward(state.height),
                });
            }
            Cmd::RescanFrom { height, resp } => {
                let r = if self.wallet.is_some() {
                    self.scan_pos = height.saturating_sub(1);
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
                });
            }
            Cmd::SetHub { cfg, resp } => {
                self.book.hub = channels::HubConfig {
                    auto_accept: cfg.auto_accept,
                    forward: cfg.forward,
                    jit_open: cfg.jit_open,
                    jit_capacity: cfg.jit_capacity.max(channels::MIN_CAPACITY),
                    min_leaves: cfg.min_leaves,
                };
                self.book.save(&self.wallet_path);
                let _ = resp.send(Ok(()));
            }
            Cmd::RotateIdentity(resp) => {
                let _ = resp.send(self.rotate_identity());
            }
            Cmd::Internal(i) => self.internal(i).await,
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    async fn create(&mut self, password: &str) -> Result<String> {
        if self.wallet_path.exists() {
            bail!("a wallet already exists at {}", self.wallet_path.display());
        }
        if let Some(parent) = self.wallet_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let (wallet, phrase) = Wallet::create_hd(&self.wallet_path, password.as_bytes())?;
        self.wallet = Some(wallet);
        // Fresh wallet: nothing historical can belong to it — start scanning
        // from the current tip instead of genesis.
        self.scan_pos = self.node.get_state().await.height;
        self.persist_scan_pos();
        Ok(phrase)
    }

    async fn restore(&mut self, password: &str, phrase: &str) -> Result<()> {
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
        let mut out = Vec::new();
        for k in w.keys() {
            out.push(AddressInfo {
                address: encode_address_with_checksum(&k.address),
                kind: "wots".into(),
                label: k.label.clone(),
                remaining_sigs: None,
                used: coin_addrs.contains(&k.address),
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

    fn history(&self) -> Result<Vec<HistoryView>> {
        let w = self.wallet.as_ref().ok_or_else(|| anyhow!("wallet is locked"))?;
        Ok(w.history()
            .iter()
            .rev()
            .map(|h| {
                // Best-effort amount: sum values of output coins we still hold.
                let amount: u64 = h
                    .outputs
                    .iter()
                    .filter_map(|id| w.find_coin(id))
                    .map(|c| c.value)
                    .sum();
                HistoryView {
                    kind: h.kind.clone(),
                    fee: h.fee,
                    timestamp: h.timestamp,
                    inputs: h.inputs.iter().map(hex::encode).collect(),
                    outputs: h.outputs.iter().map(hex::encode).collect(),
                    amount,
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
                    match w.complete_reveal(&c) {
                        Ok(()) => {
                            let _ = w.save();
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
            _ => {} // HTLC / invoice / resign / legacy traffic: not handled in this build
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
        let tip = status.height;
        if tip <= self.scan_pos {
            return;
        }
        let start = self.scan_pos + 1;
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
