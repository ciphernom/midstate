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
pub mod bond;
pub mod channels;
pub mod co_bond;
pub mod dex;
pub mod eth_trade;
pub mod eth_wire;
pub mod evm;
pub mod ledger;
pub mod mining;
pub mod mwb;
pub mod mwb_account;
pub mod mwb_link;
pub mod mwb_mining;
pub mod mwb_negotiate;
pub mod mwb_node;
pub mod mwb_trade;
pub mod mwb_wire;
pub mod node_host;
pub mod plans;
pub mod rental;
pub mod restore;
pub mod sendplan;
pub mod swap;
pub mod swapbook;
pub mod service;
/// MSN-1 names, for clients that check one before sending it.
pub use midstate::core::names;
/// What kind of address a script makes: for name editors.
pub use midstate::core::address_kind;
/// MSB-1 burns: purposes, subjects and weights, for burn forms.
pub use midstate::core::burns;
/// The most a sealed message holds, in bytes.
pub const SEALED_MAX: usize = midstate::sealed::MAX_BODY;
/// The timers a sealed conversation can have, in seconds: none, an hour, a
/// day, a week, four weeks.
pub const SEALED_TIMERS: [u32; 5] = [0, 3_600, 86_400, 7 * 86_400, 28 * 86_400];

/// A document's hashes for notarizing it (MSB-1): BLAKE3 (`notary`) and
/// SHA-256 (`notary-sha256`), hex.
pub fn document_hashes(bytes: &[u8]) -> (String, String) {
    use sha2::Digest;
    (hex::encode(blake3::hash(bytes).as_bytes()), hex::encode(sha2::Sha256::digest(bytes)))
}

/// What a name pointed at `address` would show itself paying: the kind of
/// address `script` (hex) makes, checked to be `address`'s script. An empty
/// script says only that the kind is not published.
pub fn describe_target(address: &str, script_hex: &str) -> Result<String, String> {
    let a = midstate::core::parse_address_flexible(address.trim()).map_err(|e| format!("not an address: {e}"))?;
    let script = script_hex.trim();
    if script.is_empty() {
        return Ok("script address (kind not published)".into());
    }
    let b = hex::decode(script).map_err(|_| "the script is not hex".to_string())?;
    if midstate::core::hash(&b) != a {
        return Err("that script is not this address's: it hashes to another".into());
    }
    Ok(midstate::core::address_kind::classify(&b).label())
}

pub use api::{
    AddressInfo, Balance, CoinView, HistoryView, MinerBackend, MiningView, NodeInfo, SendProgress,
    SendStage, SyncStatus, WalletEvent, WalletStatus,
};
pub use node_host::{start_node, NodeConfig};
pub use service::{spawn, WalletdHandle};

/// Display convention: the chain has no decimal subunit — amounts are raw
/// integer units everywhere (matching the bundled explorer). Grouping is a
/// presentation concern; walletd always speaks u64 units.
pub const UNIT_NAME: &str = "units";
