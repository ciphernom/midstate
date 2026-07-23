//! A durable record of what things were worth.
//!
//! The chain stores only coin ids — no amounts — so a wallet can price a
//! transaction solely from coins it still holds. That makes naive history
//! arithmetic decay: spend the change from a send and the send's recorded
//! value shrinks; spend a received coin and the receipt shrinks too. The
//! displayed past changes because the present changed, which is worse than
//! showing nothing.
//!
//! This ledger fixes that by writing values down at the moment they are known
//! and never forgetting them:
//!
//! * **Coin values** are captured opportunistically from whatever the wallet
//!   currently holds. A coin observed even once keeps its value here after it
//!   is spent, which is exactly when the wallet itself forgets.
//! * **Sends** are recorded on completion, when walletd knows the amount and
//!   the destination as typed.
//!
//! Nothing here is truly lost to the chain. A `Reveal` publishes `value` on
//! every input and output (consensus has to check conservation), and each
//! output carries its `address` — which IS the address, the same 32 bytes the
//! 72-character string encodes. So amounts and destinations alike can be
//! rebuilt by rescanning the block store; see `repair_history`. This ledger
//! exists to avoid that rescan, not because the data is unrecoverable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SendRecord {
    /// What actually left the wallet, excluding change and fee.
    pub amount: u64,
    pub fee: u64,
    /// Destination address, as entered.
    pub to: String,
    pub at: u64,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Ledger {
    /// coin id (hex) → value. Grows only with coins this wallet has held.
    #[serde(default)]
    pub coin_values: HashMap<String, u64>,
    /// A stable hash of the spent-input set → what that send really was.
    /// Keyed this way because `HistoryEntry` carries no transaction id, but it
    /// does carry its inputs, so the join works from history alone.
    #[serde(default)]
    pub sends: HashMap<String, SendRecord>,
    #[serde(skip)]
    dirty: bool,
}

/// Order-independent identity for a set of spent coins.
pub fn input_key(inputs: &[[u8; 32]]) -> String {
    let mut ids: Vec<[u8; 32]> = inputs.to_vec();
    ids.sort_unstable();
    // midstate's own BLAKE3 helper, rather than taking a direct dependency
    // on the crate for one call.
    let mut buf = Vec::with_capacity(24 + ids.len() * 32);
    buf.extend_from_slice(b"midstate_send_inputs_v1");
    for id in &ids {
        buf.extend_from_slice(id);
    }
    hex::encode(&midstate::core::types::hash(&buf)[..16])
}

impl Ledger {
    pub fn path_for(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("ledger.json")
    }

    pub fn load(wallet_path: &Path) -> Self {
        std::fs::read_to_string(Self::path_for(wallet_path))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&mut self, wallet_path: &Path) {
        if !self.dirty {
            return;
        }
        if let Ok(s) = serde_json::to_string(self) {
            if std::fs::write(Self::path_for(wallet_path), s).is_ok() {
                self.dirty = false;
            }
        }
    }

    /// Learn the value of every coin currently held. Cheap, idempotent, and
    /// called often — a coin only needs to be seen once, ever.
    pub fn observe(&mut self, coins: &[([u8; 32], u64)]) {
        for (id, value) in coins {
            let k = hex::encode(id);
            if self.coin_values.insert(k, *value).is_none() {
                self.dirty = true;
            }
        }
    }

    /// Price a single coin learned from a chain rescan.
    pub fn learn(&mut self, coin_id: &[u8; 32], value: u64) {
        if self.coin_values.insert(hex::encode(coin_id), value).is_none() {
            self.dirty = true;
        }
    }

    pub fn has_send(&self, inputs: &[[u8; 32]]) -> bool {
        self.sends.contains_key(&input_key(inputs))
    }

    pub fn record_send(&mut self, inputs: &[[u8; 32]], rec: SendRecord) {
        self.sends.insert(input_key(inputs), rec);
        self.dirty = true;
    }

    pub fn send_for(&self, inputs: &[[u8; 32]]) -> Option<&SendRecord> {
        self.sends.get(&input_key(inputs))
    }

    /// Total of the outputs this wallet has ever been able to price — i.e.
    /// the ones that were ours. Unlike counting current holdings, this does
    /// not change when those coins are later spent.
    pub fn value_of(&self, ids: &[[u8; 32]]) -> (u64, usize) {
        let mut total = 0u64;
        let mut n = 0usize;
        for id in ids {
            if let Some(v) = self.coin_values.get(&hex::encode(id)) {
                total += *v;
                n += 1;
            }
        }
        (total, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_key_ignores_ordering() {
        let a = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let b = [[3u8; 32], [1u8; 32], [2u8; 32]];
        assert_eq!(input_key(&a), input_key(&b));
        assert_ne!(input_key(&a), input_key(&[[1u8; 32], [2u8; 32]]));
    }

    #[test]
    fn values_survive_the_coin_being_spent() {
        let mut l = Ledger::default();
        l.observe(&[([7u8; 32], 4096), ([8u8; 32], 1024)]);
        // The wallet later drops both coins; the ledger still prices them.
        assert_eq!(l.value_of(&[[7u8; 32], [8u8; 32]]), (5120, 2));
        // Outputs that were never ours stay unpriced rather than counting zero.
        assert_eq!(l.value_of(&[[9u8; 32]]), (0, 0));
    }

    #[test]
    fn send_records_round_trip_by_input_set() {
        let mut l = Ledger::default();
        let inputs = [[4u8; 32], [5u8; 32]];
        l.record_send(
            &inputs,
            SendRecord { amount: 300_000_000, fee: 74, to: "abc".into(), at: 1 },
        );
        // Order the history happens to report them in must not matter.
        let rec = l.send_for(&[[5u8; 32], [4u8; 32]]).unwrap();
        assert_eq!(rec.amount, 300_000_000);
        assert!(l.send_for(&[[6u8; 32]]).is_none());
    }
}
