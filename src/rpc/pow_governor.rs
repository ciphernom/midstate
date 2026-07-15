//! Server-authoritative proof-of-work difficulty.
//!
//! # Why this exists
//!
//! The explorer used to pick its own difficulty from a `localStorage`
//! timestamp. The server accepted any proof with 4 leading zeros regardless, so
//! the escalation was decoration: an abuser edits the page (or just sends the
//! POST directly) and always pays the floor, while an honest user browsing the
//! explorer pays a self-imposed tax. A throttle the caller controls is not a
//! throttle.
//!
//! Difficulty therefore lives here. The client asks what it owes, mines that,
//! and the server verifies against the figure IT computes at request time.
//!
//! # Shape
//!
//! A fixed-window counter per source IP. Difficulty is a step function of how
//! many *expensive* requests that IP has completed in the current window.
//! Cheap rejections (bad hash, stale timestamp) are deliberately NOT counted —
//! verifying a proof is a single BLAKE3 of ~90 bytes, so making failures
//! escalate would let one attacker cheaply raise the price for a shared NAT
//! without ever touching the disk.
//!
//! Steps are in leading HEX zeros, so each one costs 16x the last. Measured in
//! the browser at ~124k H/s:
//!
//! | zeros | expected hashes | browser time |
//! |-------|-----------------|--------------|
//! |   4   |          65,536 |       ~0.5 s |
//! |   5   |       1,048,576 |       ~8.5 s |
//! |   6   |      16,777,216 |       ~135 s |
//!
//! 6 is the ceiling on purpose. 7 would be ~36 minutes, which is not a throttle
//! but a ban, and bans belong in a firewall where they can be seen and lifted.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::RwLock;

/// The floor. Every request pays at least this.
pub const MIN_ZEROS: u32 = 4;
/// The ceiling. See the table above for why it is not higher.
pub const MAX_ZEROS: u32 = 6;
/// Counter window.
pub const WINDOW_SECS: u64 = 60;
/// Requests per window before the first step up.
///
/// Sized against real client behaviour, not a guess: one explorer address view
/// costs TWO gated requests (/search + /scan). At the old value of 3, the second
/// address a user opened pushed them to 5 zeros — 8.5s per solve, twice per
/// view. That punishes browsing, which is the entire purpose of the explorer.
/// 12 leaves room for ~6 address views a minute at the 0.5s floor.
pub const FREE_REQUESTS: u32 = 12;
/// Requests per window before the second step up.
pub const TIER2_REQUESTS: u32 = 30;
/// Bucket-map cap. Beyond this we prune aggressively rather than grow.
const MAX_TRACKED_IPS: usize = 50_000;

#[derive(Debug, Clone, Copy)]
struct Bucket {
    window_start: u64,
    count: u32,
}

#[derive(Debug, Default)]
pub struct PowGovernor {
    buckets: RwLock<HashMap<IpAddr, Bucket>>,
}

impl PowGovernor {
    pub fn new() -> Self {
        Self { buckets: RwLock::new(HashMap::new()) }
    }

    /// Difficulty this IP must pay for its next expensive request.
    ///
    /// Read-only: asking is free and must stay free, or clients could not
    /// discover the price without paying it.
    pub fn required_zeros(&self, ip: IpAddr, now: u64) -> u32 {
        let count = self
            .buckets
            .read()
            .ok()
            .and_then(|b| b.get(&ip).copied())
            .filter(|b| now.saturating_sub(b.window_start) < WINDOW_SECS)
            .map(|b| b.count)
            .unwrap_or(0);
        Self::zeros_for_count(count)
    }

    /// Pure step function — separated so it can be tested without a clock or a map.
    pub fn zeros_for_count(count: u32) -> u32 {
        let z = if count < FREE_REQUESTS {
            MIN_ZEROS
        } else if count < TIER2_REQUESTS {
            MIN_ZEROS + 1
        } else {
            MIN_ZEROS + 2
        };
        z.min(MAX_ZEROS)
    }

    /// Charge this IP for one completed expensive request.
    ///
    /// Call AFTER the proof verifies and the work is about to be done. Counting
    /// failures instead would let an attacker inflate a shared NAT's difficulty
    /// for the price of one hash each.
    pub fn record(&self, ip: IpAddr, now: u64) {
        let Ok(mut map) = self.buckets.write() else { return };

        if map.len() >= MAX_TRACKED_IPS {
            map.retain(|_, b| now.saturating_sub(b.window_start) < WINDOW_SECS);
            // Still full of live entries: we are under a wide distributed load and
            // per-IP accounting has stopped being meaningful. Drop the map rather
            // than grow without bound; every IP falls back to the floor, which is
            // still a real cost per request.
            if map.len() >= MAX_TRACKED_IPS {
                map.clear();
            }
        }

        let entry = map.entry(ip).or_insert(Bucket { window_start: now, count: 0 });
        if now.saturating_sub(entry.window_start) >= WINDOW_SECS {
            entry.window_start = now;
            entry.count = 0;
        }
        entry.count = entry.count.saturating_add(1);
    }

    /// Drop expired buckets. Cheap to call periodically; not required for
    /// correctness (`required_zeros` ignores stale windows) — only for memory.
    pub fn prune(&self, now: u64) {
        if let Ok(mut map) = self.buckets.write() {
            map.retain(|_, b| now.saturating_sub(b.window_start) < WINDOW_SECS);
        }
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.buckets.read().map(|m| m.len()).unwrap_or(0)
    }
}


/// Mine a proof of work for a gated endpoint.
///
/// The exact counterpart of `handlers::verify_pow` and of `solvePow()` in
/// explorer.html. All three build the same preimage:
///
/// ```text
/// blake3(format!("{}:{}:{}:{}", subject, height, timestamp, nonce))
/// ```
///
/// It lives here so Rust callers (the CLI's contract-UTXO lookup) share one
/// implementation with the verifier instead of open-coding a third copy of the
/// format string. Three independent copies of one preimage is how you get a
/// client that mines a hash the server will never accept — which is exactly the
/// bug the explorer shipped with.
///
/// Cheap in native code: 4 zeros averages ~65k hashes, single-digit
/// milliseconds. Returns `(nonce, hex_digest)`.
pub fn solve_pow(subject: &str, height: u64, timestamp: u64, zeros: u32) -> (u64, String) {
    let prefix = "0".repeat(zeros as usize);
    let mut nonce: u64 = 0;
    loop {
        let input = format!("{}:{}:{}:{}", subject, height, timestamp, nonce);
        let hash = hex::encode(blake3::hash(input.as_bytes()).as_bytes());
        if hash.starts_with(&prefix) {
            return (nonce, hash);
        }
        nonce = nonce.wrapping_add(1);
    }
}

/// Count leading zero characters of a lowercase hex digest.
///
/// Deliberately counts HEX zeros, not bits, because that is what the wire
/// format and the client's `startsWith('0'.repeat(n))` both speak. Keeping one
/// unit end to end is what stops a client mining 4 bits and a server demanding
/// 4 nibbles.
pub fn leading_hex_zeros(hex: &str) -> u32 {
    hex.bytes().take_while(|b| *b == b'0').count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn ip(n: u8) -> IpAddr { IpAddr::V4(Ipv4Addr::new(10, 0, 0, n)) }

    #[test]
    fn floor_applies_to_a_fresh_ip() {
        let g = PowGovernor::new();
        assert_eq!(g.required_zeros(ip(1), 1000), MIN_ZEROS);
    }

    #[test]
    fn difficulty_escalates_with_use() {
        let g = PowGovernor::new();
        let a = ip(1);
        for _ in 0..FREE_REQUESTS { g.record(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MIN_ZEROS + 1, "should step up after the free allowance");
        for _ in FREE_REQUESTS..TIER2_REQUESTS { g.record(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MIN_ZEROS + 2, "should step up again");
    }

    #[test]
    fn difficulty_is_capped() {
        let g = PowGovernor::new();
        let a = ip(1);
        for _ in 0..10_000 { g.record(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MAX_ZEROS, "must never exceed the ceiling");
        assert!(MAX_ZEROS <= 6, "7 hex zeros is ~36 min in a browser — a ban, not a throttle");
    }

    #[test]
    fn window_expiry_resets_the_price() {
        let g = PowGovernor::new();
        let a = ip(1);
        for _ in 0..TIER2_REQUESTS { g.record(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MAX_ZEROS);
        assert_eq!(g.required_zeros(a, 1000 + WINDOW_SECS), MIN_ZEROS, "a stale window must not keep charging");
        assert_eq!(g.required_zeros(a, 1000 + WINDOW_SECS * 100), MIN_ZEROS);
    }

    #[test]
    fn recording_after_expiry_starts_a_new_window() {
        let g = PowGovernor::new();
        let a = ip(1);
        for _ in 0..TIER2_REQUESTS { g.record(a, 1000); }
        g.record(a, 1000 + WINDOW_SECS + 1);
        assert_eq!(g.required_zeros(a, 1000 + WINDOW_SECS + 1), MIN_ZEROS, "first request of a new window pays the floor");
    }

    #[test]
    fn ips_are_accounted_independently() {
        let g = PowGovernor::new();
        for _ in 0..TIER2_REQUESTS { g.record(ip(1), 1000); }
        assert_eq!(g.required_zeros(ip(1), 1000), MAX_ZEROS);
        assert_eq!(g.required_zeros(ip(2), 1000), MIN_ZEROS, "one abuser must not tax everyone else");
    }

    #[test]
    fn asking_the_price_is_free() {
        let g = PowGovernor::new();
        let a = ip(1);
        for _ in 0..100 { g.required_zeros(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MIN_ZEROS,
            "required_zeros must not charge — a client cannot discover the price without it");
    }

    #[test]
    fn prune_drops_only_stale_buckets() {
        let g = PowGovernor::new();
        g.record(ip(1), 1000);
        g.record(ip(2), 1000 + WINDOW_SECS);
        g.prune(1000 + WINDOW_SECS);
        assert_eq!(g.tracked(), 1, "the live bucket must survive");
        g.prune(1000 + WINDOW_SECS * 3);
        assert_eq!(g.tracked(), 0);
    }

    #[test]
    fn bucket_map_stays_bounded() {
        let g = PowGovernor::new();
        for i in 0..(MAX_TRACKED_IPS + 5_000) {
            let o = (i % 256) as u8;
            let a = IpAddr::V4(Ipv4Addr::new((i >> 24) as u8, (i >> 16) as u8, (i >> 8) as u8, o));
            g.record(a, 1000);
        }
        assert!(g.tracked() <= MAX_TRACKED_IPS, "must not grow without bound under a distributed flood");
    }


    #[test]
    fn solve_pow_output_satisfies_the_verifier() {
        for zeros in [1u32, 2, 3, 4] {
            let (nonce, hash) = solve_pow("abc", 42, 1_700_000_000, zeros);
            assert!(leading_hex_zeros(&hash) >= zeros, "miner produced work below its own target");
            // Recompute exactly as handlers::verify_pow does.
            let input = format!("{}:{}:{}:{}", "abc", 42, 1_700_000_000u64, nonce);
            let recomputed = hex::encode(blake3::hash(input.as_bytes()).as_bytes());
            assert_eq!(recomputed, hash, "verifier disagrees with the miner about the preimage");
        }
    }

    #[test]
    fn solve_pow_is_bound_to_every_field() {
        // Change any one input and the digest must change: this is what makes a
        // proof non-transferable between subjects/heights/timestamps.
        let base = solve_pow("abc", 42, 1_700_000_000, 1);
        let inp = |s: &str, h: u64, t: u64, n: u64| hex::encode(blake3::hash(format!("{}:{}:{}:{}", s, h, t, n).as_bytes()).as_bytes());
        assert_ne!(base.1, inp("abd", 42, 1_700_000_000, base.0), "subject must bind");
        assert_ne!(base.1, inp("abc", 43, 1_700_000_000, base.0), "height must bind");
        assert_ne!(base.1, inp("abc", 42, 1_700_000_001, base.0), "timestamp must bind");
        assert_ne!(base.1, inp("abc", 42, 1_700_000_000, base.0 + 1), "nonce must bind");
    }
    /// An explorer address view costs TWO gated requests (/search + /scan).
    /// The free allowance must cover several views or browsing gets taxed —
    /// which is what the old value of 3 did.
    #[test]
    fn free_allowance_covers_normal_browsing() {
        const REQUESTS_PER_ADDRESS_VIEW: u32 = 2;
        let g = PowGovernor::new();
        let a = ip(1);
        let views = 4;
        for _ in 0..(views * REQUESTS_PER_ADDRESS_VIEW) { g.record(a, 1000); }
        assert_eq!(g.required_zeros(a, 1000), MIN_ZEROS,
            "{} address views must still cost the floor; FREE_REQUESTS={} is too tight",
            views, FREE_REQUESTS);
    }

    #[test]
    fn hex_zero_counting() {
        assert_eq!(leading_hex_zeros("0000abcd"), 4);
        assert_eq!(leading_hex_zeros("00000abc"), 5);
        assert_eq!(leading_hex_zeros("abcd0000"), 0);
        assert_eq!(leading_hex_zeros(""), 0);
        assert_eq!(leading_hex_zeros("00000000000000000000000000000000000000000000000000000000000000000"), 65);
    }

    #[test]
    fn a_proof_that_meets_the_floor_is_accepted_at_the_floor() {
        // Guards the comparison direction: >= required, not == required.
        let g = PowGovernor::new();
        let a = ip(1);
        let required = g.required_zeros(a, 1000);
        assert!(leading_hex_zeros("00000abc") >= required, "over-mining must be accepted");
        assert!(leading_hex_zeros("000abcde") < required, "under-mining must be rejected");
    }
}
