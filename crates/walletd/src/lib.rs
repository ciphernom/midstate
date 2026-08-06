//! midstate-walletd — the headless wallet service behind the Midstate desktop app.
//!
//! Design (see docs/midstate-gui-wallet-plan.md §3–§4):
//! - One actor owns the open `midstate::wallet::Wallet` and is the ONLY code
//!   allowed to mutate it. All UI surfaces talk to it through [`WalletdHandle`].
//! - The full node runs in-process ([`node_host`]); walletd holds its
//!   `NodeHandle` and answers every chain question from local validated state.
//! - Sends are persisted state machines (commit → mined → reveal → confirmed),
//!   resumed on unlock from the wallet's own `PendingCommit` records.
//!
//! WOTS one-time-signature invariants enforced here, not in the UI:
//! - a coin whose key has signed (`wots_signed`) is never selected again;
//! - coins referenced by any live pending commit are never double-planned;
//! - MSS leaf indices are verified against chain + mempool and fast-forwarded
//!   (safety margin 20) before every signing session, mirroring the CLI.

pub mod api;
pub mod base;
pub mod channels;
pub mod dex;
pub mod evm;
pub mod ledger;
pub mod mining;
pub mod node_host;
pub mod sendplan;
pub mod swap;
pub mod swapbook;
pub mod service;

pub use api::{
    AddressInfo, Balance, CoinView, HistoryView, MiningView, NodeInfo, SendProgress, SendStage,
    SyncStatus, WalletEvent, WalletStatus,
};
pub use node_host::{start_node, NodeConfig};
pub use service::{spawn, WalletdHandle};

/// Display convention: the chain has no decimal subunit — amounts are raw
/// integer units everywhere (matching the bundled explorer). Grouping is a
/// presentation concern; walletd always speaks u64 units.
pub const UNIT_NAME: &str = "units";
