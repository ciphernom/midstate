//! Restore discovery for the desktop wallet (audit W6).
//!
//! A wallet restored from its recovery phrase starts with nothing but the seed.
//! Before it signs anything it has to find every key the phrase ever used: WOTS
//! receive and change keys, and the MSS trees behind mining payouts, receive
//! addresses, consolidations and defrags. Otherwise it hands out change on a
//! one-time key the original device already signed with, and shows only part of
//! the balance.
//!
//! The service's scan tick walks the chain from genesis over a watch window this
//! module keeps ahead of the highest key that has received a coin: [`WOTS_GAP`]
//! unused WOTS keys and [`MSS_GAP`] unused MSS trees past it. Keys derived while a
//! pass is running were not watched in the blocks it had already scanned, so a
//! pass that grew the window is repeated from genesis ([`RestoreScan::end`]);
//! passes stop once the window holds still. Signing waits until then.
//!
//! The CLI restore (`wallet_restore` in `src/main.rs`) finds WOTS keys the same
//! way (a 1,024-key floor, a 256-key gap limit, a full rescan per extension) but
//! derives a fixed 4 MSS trees.

use anyhow::Result;
use midstate::core::compute_address;
use midstate::wallet::{hd, Wallet};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// WOTS keys derived up-front on restore, before the first pass.
pub const WOTS_FLOOR: u64 = 1_000;
/// MSS trees derived up-front on restore: one more than [`MSS_GAP`], so a
/// wallet that has only ever used its first tree is found in a single pass.
pub const MSS_FLOOR: u64 = 5;
/// Unused WOTS keys kept watched past the highest key that received a coin.
/// The desktop wallet allocates one key per power-of-two change output, so its
/// used keys are close together; 256 (the CLI's gap limit) leaves room for the
/// keys that failed or abandoned sends reserved and never used.
pub const WOTS_GAP: u64 = 256;
/// Unused MSS trees kept watched past the highest tree that received a coin.
/// Trees are cheap to skip but expensive to derive, and a user can make a few
/// receive addresses in a row without using them.
pub const MSS_GAP: u64 = 4;
/// From-genesis passes before the restore stops waiting for the window to hold
/// still. Each pass is a full chain scan.
pub const MAX_PASSES: u32 = 8;
/// A pass only ends at a tip whose block is at most this old (blocks target 60
/// seconds).
pub const TIP_MAX_AGE_SECS: u64 = 2 * 60 * 60;

/// May a restore pass end at a tip with this timestamp? The node does not
/// report "syncing" before it has peers, or between sync sessions, so a restore
/// on a fresh install could otherwise reach the local tip, hours or months
/// behind the network's, call the history complete, and stop looking for keys
/// before the blocks that use them have even arrived.
pub fn tip_is_current(tip_timestamp: u64, now: u64) -> bool {
    tip_timestamp > 0 && now.saturating_sub(tip_timestamp) <= TIP_MAX_AGE_SECS
}

/// Progress of a from-genesis restore scan, persisted beside the wallet so a
/// restart cannot unblock signing before the scan has converged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreScan {
    /// 1-based pass number.
    pub pass: u32,
    /// The watch window `(wots_index, mss_index)` when this pass started at
    /// height 0. `None` until it starts; a pass whose start was never recorded
    /// is restarted rather than trusted.
    pub start: Option<(u64, u64)>,
    /// Highest WOTS index that any output on chain was ever paid to, spent or
    /// not. Only live coins are imported, so this is the one record of how far
    /// the phrase's history reaches.
    #[serde(default)]
    pub highest_wots: Option<u64>,
    /// Highest MSS tree index any output was ever paid to.
    #[serde(default)]
    pub highest_mss: Option<u64>,
    /// The wallet's HD counters `(wots, mss)` when the scan began, if it began
    /// on a wallet already in use. Settling never moves a counter below them:
    /// whatever the wallet had handed out by then (a receive address given to
    /// a payer who has not paid yet, say) is never handed out again. `None`
    /// for a restore from the phrase, which starts from nothing.
    #[serde(default)]
    pub counters_before: Option<(u64, u64)>,
}

/// What [`RestoreScan::end`] decided when a pass reached the tip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassEnd {
    /// The window grew during the pass: scan again from genesis.
    Again,
    /// The window held still: every block was checked against every key.
    Done,
    /// Still growing after [`MAX_PASSES`]; stop and let the wallet be used.
    GaveUp,
}

impl RestoreScan {
    pub fn new() -> Self {
        Self { pass: 1, ..Default::default() }
    }

    pub fn path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("restore")
    }

    /// Marker written once a wallet's history is known complete: after a
    /// restore converges, or at creation. A wallet without it (saved before
    /// this module existed) is checked once on unlock.
    pub fn done_path(wallet_path: &Path) -> PathBuf {
        wallet_path.with_extension("restore-done")
    }

    pub fn is_done(wallet_path: &Path) -> bool {
        Self::done_path(wallet_path).exists()
    }

    pub fn mark_done(wallet_path: &Path) {
        let _ = std::fs::write(Self::done_path(wallet_path), "1");
    }

    /// Raise the recorded high-water marks. Returns whether either moved.
    pub fn note_used(&mut self, used: (Option<u64>, Option<u64>)) -> bool {
        let (w, m) = (self.highest_wots.max(used.0), self.highest_mss.max(used.1));
        let moved = (w, m) != (self.highest_wots, self.highest_mss);
        self.highest_wots = w;
        self.highest_mss = m;
        moved
    }

    pub fn load(wallet_path: &Path) -> Option<Self> {
        let raw = std::fs::read_to_string(Self::path(wallet_path)).ok()?;
        match serde_json::from_str(&raw) {
            Ok(s) => Some(s),
            // A sidecar we cannot read still means a restore was in progress:
            // keep signing blocked and start the passes over.
            Err(_) => Some(Self::new()),
        }
    }

    pub fn save(&self, wallet_path: &Path) -> Result<()> {
        std::fs::write(Self::path(wallet_path), serde_json::to_string(self)?)?;
        Ok(())
    }

    pub fn clear(wallet_path: &Path) {
        let _ = std::fs::remove_file(Self::path(wallet_path));
    }

    /// Record the window a pass starts with (called when scanning from height 0).
    pub fn begin(&mut self, window: (u64, u64)) {
        self.start = Some(window);
    }

    /// Decide what happens when a pass reaches the chain tip.
    pub fn end(&mut self, window: (u64, u64)) -> PassEnd {
        let Some(start) = self.start else {
            self.start = None;
            return PassEnd::Again;
        };
        let grew = window.0 > start.0 || window.1 > start.1;
        if !grew {
            return PassEnd::Done;
        }
        if self.pass >= MAX_PASSES {
            return PassEnd::GaveUp;
        }
        self.pass += 1;
        self.start = None;
        PassEnd::Again
    }
}

/// The watch window: how many WOTS keys and MSS trees the wallet has derived.
/// During restore passes the HD counters only grow, so they are the window.
pub fn window(w: &Wallet) -> (u64, u64) {
    (w.wots_index(), w.mss_index())
}

/// Map each seed in `needed` to its HD index. Seeds are cheap hashes of
/// (master seed, index), so this re-derives seeds from 0 rather than running
/// any keygen. It covers at least `0..at_least`, and keeps going (up to a cap)
/// while a needed seed is still unplaced, so keys derived beyond a rewound
/// counter are found too. Seeds that never turn up (coins imported with a
/// foreign seed) are simply absent from the map.
fn seed_indices(
    master: &[u8; 32],
    derive: fn(&[u8; 32], u64) -> [u8; 32],
    at_least: u64,
    needed: &HashSet<[u8; 32]>,
) -> HashMap<[u8; 32], u64> {
    const SEARCH_PAST: u64 = 100_000;
    let mut map = HashMap::new();
    let mut missing = needed.clone();
    let cap = at_least.saturating_add(SEARCH_PAST);
    let mut i = 0u64;
    while i < at_least || (!missing.is_empty() && i < cap) {
        let seed = derive(master, i);
        missing.remove(&seed);
        map.insert(seed, i);
        i += 1;
    }
    map
}

/// The wallet's WOTS and MSS seeds, keyed by address, with their HD indices.
struct Indexed {
    wots_seed_of: HashMap<[u8; 32], [u8; 32]>,
    mss_seed_of: HashMap<[u8; 32], [u8; 32]>,
    wots_idx: HashMap<[u8; 32], u64>,
    mss_idx: HashMap<[u8; 32], u64>,
}

impl Indexed {
    fn build(w: &Wallet, master: &[u8; 32]) -> Self {
        let mut mss_seed_of = HashMap::new();
        for m in w.mss_keys() {
            mss_seed_of.insert(compute_address(&m.master_pk), m.master_seed);
        }
        let mut wots_seed_of = HashMap::new();
        for k in w.keys() {
            wots_seed_of.insert(k.address, k.seed);
        }
        for c in w.coins() {
            if !mss_seed_of.contains_key(&c.address) {
                wots_seed_of.entry(c.address).or_insert(c.seed);
            }
        }
        let mut wots_needed: HashSet<[u8; 32]> = wots_seed_of.values().copied().collect();
        for p in w.pending() {
            wots_needed.extend(p.change_seeds.iter().map(|(_, s)| *s));
        }
        let mss_needed: HashSet<[u8; 32]> = mss_seed_of.values().copied().collect();
        Self {
            wots_idx: seed_indices(master, hd::derive_wots_seed, w.wots_index(), &wots_needed),
            mss_idx: seed_indices(master, hd::derive_mss_seed, w.mss_index(), &mss_needed),
            wots_seed_of,
            mss_seed_of,
        }
    }
}

/// Highest WOTS and MSS HD index among `addresses` that belong to this wallet.
/// Legacy (non-HD) wallets have no indices and return `(None, None)`.
pub fn highest_used(w: &Wallet, addresses: &[[u8; 32]]) -> (Option<u64>, Option<u64>) {
    let Some(master) = w.data.master_seed else {
        return (None, None);
    };
    let ix = Indexed::build(w, &master);
    let (mut top_wots, mut top_mss) = (None, None);
    for a in addresses {
        if let Some(i) = ix.mss_seed_of.get(a).and_then(|s| ix.mss_idx.get(s)) {
            top_mss = top_mss.max(Some(*i));
        } else if let Some(i) = ix.wots_seed_of.get(a).and_then(|s| ix.wots_idx.get(s)) {
            top_wots = top_wots.max(Some(*i));
        }
    }
    (top_wots, top_mss)
}

/// Derive keys and trees so that [`WOTS_GAP`] unused WOTS keys and [`MSS_GAP`]
/// unused MSS trees stay watched past the highest used ones. Returns whether
/// the window grew.
pub fn extend(
    w: &mut Wallet,
    used: (Option<u64>, Option<u64>),
    mss_height: u32,
) -> Result<bool> {
    extend_by(w, used, (WOTS_GAP, MSS_GAP), mss_height)
}

/// [`extend`] with explicit gaps (tests use small ones: every key is a full
/// WOTS keygen, which is slow in an unoptimised build).
pub fn extend_by(
    w: &mut Wallet,
    used: (Option<u64>, Option<u64>),
    gaps: (u64, u64),
    mss_height: u32,
) -> Result<bool> {
    let mut grew = false;
    if let Some(i) = used.0 {
        let want = i + 1 + gaps.0;
        if w.wots_index() < want {
            w.restore_generate_keys(want - w.wots_index())?;
            grew = true;
        }
    }
    if let Some(j) = used.1 {
        let want = j + 1 + gaps.1;
        while w.mss_index() < want {
            w.generate_mss(mss_height, Some("Recovered MSS".into()))?;
            grew = true;
        }
    }
    Ok(grew)
}

/// Derive every tree below the MSS counter that the wallet does not hold (see
/// `Wallet::restore_mss_tree`): an interrupted keygen leaves its index
/// allocated with no tree, and a restore must watch every index below the
/// counter. Returns how many trees were derived.
pub fn fill_mss_holes(w: &mut Wallet, mss_height: u32) -> Result<u64> {
    let Some(master) = w.data.master_seed else { return Ok(0) };
    let have: HashSet<[u8; 32]> = w.mss_keys().iter().map(|k| k.master_seed).collect();
    let mut filled = 0;
    for i in 0..w.mss_index() {
        if !have.contains(&hd::derive_mss_seed(&master, i)) && w.restore_mss_tree(i, mss_height)?.is_some() {
            filled += 1;
        }
    }
    Ok(filled)
}

/// Once a restore has converged, move the HD counters back to just past the
/// highest index the chain has seen (or that this wallet holds or has reserved),
/// so new keys continue the phrase's sequence with no gap.
///
/// Without this the next change key would sit [`WOTS_GAP`] past the last used
/// one, and a later restore — whose window ends exactly there — would miss it:
/// the same off-by-one the old web wallet had at 100. Everything between is a
/// derived watch key that never received an output, so it has never signed;
/// `Wallet::allocate_next_wots_seed` hands such keys out again and drops them
/// from the receive list as it does. Trees are reusable, and `generate_mss`
/// returns an already-derived tree rather than duplicating it.
pub fn settle_counters(w: &mut Wallet, rs: &RestoreScan) -> Result<()> {
    let Some(master) = w.data.master_seed else { return Ok(()) };
    let ix = Indexed::build(w, &master);

    let (wots_before, mss_before) = rs.counters_before.unwrap_or((0, 0));

    // WOTS: the chain's high-water mark, the counter the scan started from,
    // every coin held, every change key a pending commit has reserved.
    let mut wots_floor = rs.highest_wots.map(|i| i + 1).unwrap_or(0).max(wots_before);
    for c in w.coins() {
        if !ix.mss_seed_of.contains_key(&c.address) {
            if let Some(i) = ix.wots_idx.get(&c.seed) {
                wots_floor = wots_floor.max(i + 1);
            }
        }
    }
    for p in w.pending() {
        for (_, seed) in &p.change_seeds {
            if let Some(i) = ix.wots_idx.get(seed) {
                wots_floor = wots_floor.max(i + 1);
            }
        }
    }

    // MSS: the high-water mark, the starting counter, and every tree that
    // holds a coin or has signed.
    let mut mss_floor = rs.highest_mss.map(|i| i + 1).unwrap_or(0).max(mss_before);
    for m in w.mss_keys() {
        let addr = compute_address(&m.master_pk);
        let held = w.coins().iter().any(|c| c.address == addr);
        if held || m.next_leaf > 0 {
            if let Some(i) = ix.mss_idx.get(&m.master_seed) {
                mss_floor = mss_floor.max(i + 1);
            }
        }
    }

    w.rewind_hd_counters(wots_floor, mss_floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHRASE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon abandon abandon abandon \
                          abandon abandon abandon abandon abandon abandon abandon art";

    fn restored(dir: &tempfile::TempDir) -> Wallet {
        Wallet::restore_from_mnemonic(&dir.path().join("w.dat"), b"pw", PHRASE).unwrap()
    }

    #[test]
    fn pass_repeats_until_the_window_holds_still() {
        let mut rs = RestoreScan::new();
        // A pass whose start was never recorded is not trusted.
        assert_eq!(rs.end((10, 1)), PassEnd::Again);
        rs.begin((10, 1));
        assert_eq!(rs.end((300, 1)), PassEnd::Again, "WOTS window grew");
        assert_eq!(rs.pass, 2);
        assert_eq!(rs.start, None);
        rs.begin((300, 1));
        assert_eq!(rs.end((300, 2)), PassEnd::Again, "an MSS tree was added");
        rs.begin((300, 2));
        assert_eq!(rs.end((300, 2)), PassEnd::Done);
    }

    #[test]
    fn a_pass_only_ends_at_a_current_tip() {
        let now = 1_800_000_000;
        assert!(tip_is_current(now - 90, now));
        assert!(tip_is_current(now - TIP_MAX_AGE_SECS, now));
        assert!(!tip_is_current(now - TIP_MAX_AGE_SECS - 1, now), "a stale local tip");
        assert!(!tip_is_current(0, now), "no blocks yet");
        assert!(tip_is_current(now + 30, now), "a tip slightly ahead of the local clock");
    }

    #[test]
    fn pass_gives_up_after_the_cap() {
        let mut rs = RestoreScan { pass: MAX_PASSES, start: Some((0, 0)), ..Default::default() };
        assert_eq!(rs.end((1, 0)), PassEnd::GaveUp);
    }

    #[test]
    fn sidecar_round_trips_and_a_corrupt_one_keeps_signing_blocked() {
        let dir = tempfile::tempdir().unwrap();
        let wp = dir.path().join("w.dat");
        assert_eq!(RestoreScan::load(&wp), None);
        let rs = RestoreScan {
            pass: 3,
            start: Some((1256, 5)),
            highest_wots: Some(999),
            highest_mss: None,
            counters_before: Some((1000, 0)),
        };
        rs.save(&wp).unwrap();
        assert_eq!(RestoreScan::load(&wp), Some(rs));
        std::fs::write(RestoreScan::path(&wp), "not json").unwrap();
        assert_eq!(RestoreScan::load(&wp), Some(RestoreScan::new()));
        RestoreScan::clear(&wp);
        assert_eq!(RestoreScan::load(&wp), None);
    }

    #[test]
    fn extension_follows_the_highest_used_key() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.restore_generate_keys(10).unwrap();
        assert_eq!(w.wots_index(), 10);

        let addr_of = |w: &Wallet, i: u64| -> [u8; 32] {
            let seed = hd::derive_wots_seed(&w.data.master_seed.unwrap(), i);
            w.keys().iter().find(|k| k.seed == seed).unwrap().address
        };
        let a3 = addr_of(&w, 3);
        let a7 = addr_of(&w, 7);
        assert_eq!(highest_used(&w, &[a3, a7, [9u8; 32]]), (Some(7), None));

        assert!(extend_by(&mut w, (Some(7), None), (4, 2), 1).unwrap());
        assert_eq!(w.wots_index(), 7 + 1 + 4);
        // Already far enough ahead: nothing to derive.
        assert!(!extend_by(&mut w, (Some(7), None), (4, 2), 1).unwrap());
        // The keys are the phrase's own: index 11 maps back to 11.
        let a11 = addr_of(&w, 11);
        assert_eq!(highest_used(&w, &[a11]).0, Some(11));
    }

    #[test]
    fn extension_follows_the_highest_used_tree() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        let t0 = w.generate_mss(1, None).unwrap();
        let t1 = w.generate_mss(1, None).unwrap();
        assert_eq!(highest_used(&w, &[t0]), (None, Some(0)));
        assert_eq!(highest_used(&w, &[t1, t0]), (None, Some(1)));
        assert!(extend_by(&mut w, (None, Some(1)), (8, 2), 1).unwrap());
        assert_eq!(w.mss_index(), 1 + 1 + 2);
    }

    #[test]
    fn high_water_marks_only_rise() {
        let mut rs = RestoreScan::new();
        assert!(rs.note_used((Some(5), None)));
        assert!(!rs.note_used((Some(3), None)));
        assert!(rs.note_used((None, Some(1))));
        assert_eq!((rs.highest_wots, rs.highest_mss), (Some(5), Some(1)));
    }

    #[test]
    fn settling_rewinds_to_just_past_the_highest_used_key() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.restore_generate_keys(12).unwrap();
        w.generate_mss(1, None).unwrap();
        w.generate_mss(1, None).unwrap();
        w.generate_mss(1, None).unwrap();
        let rs = RestoreScan { highest_wots: Some(5), highest_mss: Some(0), ..RestoreScan::new() };
        settle_counters(&mut w, &rs).unwrap();
        assert_eq!(w.wots_index(), 6, "next change key continues right after the last used one");
        assert_eq!(w.mss_index(), 1);
        // The next key handed out is index 6, and it leaves the receive list.
        let master = w.data.master_seed.unwrap();
        let seed = w.allocate_next_wots_seed().unwrap();
        assert_eq!(seed, hd::derive_wots_seed(&master, 6));
        assert!(w.keys().iter().all(|k| k.seed != seed));
        // The next tree is the already-derived tree 1, not a duplicate.
        let before = w.mss_keys().len();
        let t1 = w.generate_mss(1, None).unwrap();
        assert_eq!(w.mss_keys().len(), before);
        assert_eq!(t1, compute_address(&w.mss_keys()[1].master_pk));
    }

    #[test]
    fn settling_never_rewinds_below_a_held_coin() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.restore_generate_keys(10).unwrap();
        let k = w.keys()[7].clone();
        w.import_scanned(k.address, 8, [1u8; 32], None).unwrap().unwrap();
        // The chain's record says 2, but the wallet holds a coin on key 7.
        let rs = RestoreScan { highest_wots: Some(2), ..RestoreScan::new() };
        settle_counters(&mut w, &rs).unwrap();
        assert_eq!(w.wots_index(), 8);
    }

    #[test]
    fn settling_never_rewinds_below_where_the_scan_started() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.restore_generate_keys(4).unwrap();
        // A wallet already in use when its scan began: key 2 may have been
        // given to a payer, though nothing has been paid to it yet.
        let rs = RestoreScan {
            highest_wots: Some(0),
            counters_before: Some((3, 0)),
            ..RestoreScan::new()
        };
        settle_counters(&mut w, &rs).unwrap();
        assert_eq!(w.wots_index(), 3);
    }

    /// The discovery loop exactly as the service tick drives it, over a
    /// synthetic chain, with small floors and gaps so it runs quickly in an
    /// unoptimised build. Every output is treated as spent (nothing imported),
    /// as for a wallet whose history is all in the past.
    #[test]
    fn discovery_finds_history_with_gaps_and_out_of_order_coins() {
        let dir = tempfile::tempdir().unwrap();
        let master = hd::master_seed_from_mnemonic(PHRASE).unwrap();
        let wots_addr = |i: u64| compute_address(&midstate::core::wots::keygen(&hd::derive_wots_seed(&master, i)));
        let mss_addr = |j: u64| {
            compute_address(&midstate::core::mss::keygen(&hd::derive_mss_seed(&master, j), 1).unwrap().master_pk)
        };
        // (height, address). WOTS gaps of up to 3 unused keys (the gap is 4);
        // key 16 is paid at height 2 but only becomes watched once key 12 is
        // seen at height 8, so only a second pass can find it. MSS tree 3 is
        // two unused trees past tree 0 (the MSS gap here is 3).
        let wots_used = [(1, 0u64), (2, 1), (3, 4), (5, 8), (8, 12), (2, 16)];
        let mss_used = [(4, 0u64), (7, 3)];
        let mut outputs: Vec<(u64, [u8; 32])> = Vec::new();
        outputs.extend(wots_used.iter().map(|&(h, i)| (h, wots_addr(i))));
        outputs.extend(mss_used.iter().map(|&(h, j)| (h, mss_addr(j))));
        let tip = 10u64;

        let mut w = restored(&dir);
        w.restore_generate_keys(4).unwrap(); // the WOTS floor
        w.generate_mss(1, None).unwrap(); // the MSS floor
        let mut rs = RestoreScan::new();
        let (mut scan_pos, mut passes) = (0u64, 0u32);
        loop {
            if rs.start.is_none() {
                scan_pos = 0;
                rs.begin(window(&w));
                passes += 1;
            }
            if scan_pos >= tip {
                match rs.end(window(&w)) {
                    PassEnd::Again => continue,
                    PassEnd::Done => break,
                    PassEnd::GaveUp => panic!("did not converge"),
                }
            }
            let end = (scan_pos + 3).min(tip);
            let watch: HashSet<[u8; 32]> = w.watched_addresses().into_iter().collect();
            let found: Vec<[u8; 32]> = outputs
                .iter()
                .filter(|(h, a)| *h >= scan_pos && *h < end && watch.contains(a))
                .map(|(_, a)| *a)
                .collect();
            // Like the tick: a chunk that grew the window is scanned again.
            if !found.is_empty()
                && rs.note_used(highest_used(&w, &found))
                && extend_by(&mut w, (rs.highest_wots, rs.highest_mss), (4, 3), 1).unwrap()
            {
                continue;
            }
            scan_pos = end;
        }

        let watch: HashSet<[u8; 32]> = w.watched_addresses().into_iter().collect();
        assert!(outputs.iter().all(|(_, a)| watch.contains(a)), "every output is watched");
        assert_eq!((rs.highest_wots, rs.highest_mss), (Some(16), Some(3)));
        assert!(passes >= 2, "the out-of-order coin needs a second pass");

        settle_counters(&mut w, &rs).unwrap();
        assert_eq!(w.wots_index(), 17, "change continues right after key 16");
        assert_eq!(w.mss_index(), 4);
        let seed = w.allocate_next_wots_seed().unwrap();
        assert_eq!(seed, hd::derive_wots_seed(&master, 17));
    }

    #[test]
    fn a_tree_whose_keygen_never_finished_is_derived_again() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.generate_mss(1, None).unwrap();
        // Index 1 allocated (the counter is saved first), keygen never ran.
        w.allocate_next_mss_seed().unwrap();
        w.generate_mss(1, None).unwrap();
        assert_eq!((w.mss_index(), w.mss_keys().len()), (3, 2));
        assert_eq!(fill_mss_holes(&mut w, 1).unwrap(), 1);
        assert_eq!((w.mss_index(), w.mss_keys().len()), (3, 3));
        let master = w.data.master_seed.unwrap();
        assert!(w.mss_keys().iter().any(|k| k.master_seed == hd::derive_mss_seed(&master, 1)));
        assert_eq!(fill_mss_holes(&mut w, 1).unwrap(), 0);
    }

    #[test]
    fn restore_generate_keys_matches_one_by_one_derivation() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = restored(&dir);
        w.restore_generate_keys(3).unwrap();
        w.restore_generate_keys(2).unwrap();
        assert_eq!(w.wots_index(), 5);
        let master = w.data.master_seed.unwrap();
        for (i, k) in w.keys().iter().enumerate() {
            assert_eq!(k.seed, hd::derive_wots_seed(&master, i as u64));
            assert_eq!(k.address, compute_address(&midstate::core::wots::keygen(&k.seed)));
        }
        // The counter persisted: reopening sees the same window.
        drop(w);
        let w2 = Wallet::open(&dir.path().join("w.dat"), b"pw").unwrap();
        assert_eq!(w2.wots_index(), 5);
        assert_eq!(w2.keys().len(), 5);
    }
}
