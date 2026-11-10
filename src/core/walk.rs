//! Midstate Walk: the proof of work from `WALK_ACTIVATION_HEIGHT` on.
//!
//! # Why
//!
//! The original proof of work iterates BLAKE3 a million times per nonce. A
//! nonce needs 32 bytes of state and nothing but 32-bit add/rotate/xor, so a
//! GPU runs tens of thousands of nonces side by side. Making one nonce
//! sequential never helped: mining is a lottery, and a GPU simply holds more
//! tickets at once.
//!
//! Midstate Walk makes each attempt need what a CPU core has and a GPU thread
//! does not:
//!
//! * **A private 256 KiB scratchpad**, read and rewritten at random, one
//!   dependent step after another. It fits a core's L2 cache (a Raspberry
//!   Pi 5's Cortex-A76 has 512 KiB per core) but no GPU thread's fast on-chip
//!   memory, so a GPU waits on L2 or VRAM at every step, moving four 32-byte
//!   sectors in and four out.
//! * **A program per attempt.** Which operation each register runs is drawn
//!   from the data and redrawn every few thousand steps. A CPU's branch
//!   predictor learns it within a few steps; the 32 threads of a GPU warp each
//!   have their own and diverge.
//! * **Operations GPUs emulate in software:** 64-bit multiply, 64×64→128 high
//!   multiply, 64-bit division, and an AES round (AES-NI on x86, the ARMv8
//!   crypto extensions on a Pi 5).
//!
//! It is still hashing, so it is as post-quantum as before: Grover gives the
//! same at-most-quadratic edge it gives against any hash-based proof of work.
//!
//! # The name
//!
//! The eight registers are the walk's *midstate*. At every step they choose
//! which memory to read, which register to combine with and where the result
//! goes, and every few thousand steps they write the next program.
//!
//! # Definition (version 1)
//!
//! Words are little-endian u64 and arithmetic wraps.
//!
//! ```text
//! seed = BLAKE3-XOF(derive_key "midstate walk v1 init", mining_hash ‖ le64(nonce))[0..200]
//! r[0..8] = seed[0..64]                       registers
//! S[0..4] = seed[64..128], K[0..4] = seed[128..192]   four AES lanes and their keys
//! prog    = program(seed[192..200])
//! for n in 0..LINES:  for q in 0..4:  S[q] = AESENC(S[q], K[q]);  pad[n][2q..2q+2] = S[q]
//!
//! c = 0
//! for i in 0..STEPS:
//!     s  = i mod 8;  x = r[s] xor c                c: the previous step's output
//!     a  = x mod LINES                       bits  0..11  first line
//!     t  = (s + 1 + (x >> 12) mod 4) mod 8   bits 12..13  partner: one of the next four registers
//!     j  = (x >> 14) mod 8                   bits 14..16  word
//!     e  = (x >> 17) mod LINES               bits 17..28  second line
//!     b  = (x >> 32) mod LINES               bits 32..43  third line
//!     f  = (x >> 44) mod LINES               bits 44..55  fourth line
//!     m0 = pad[a][j];  m1 = pad[b][j xor 4];  m2 = pad[e][j xor 2];  m3 = pad[f][j xor 6]
//!     k  = r[t];  v = x xor m0 xor m2
//!     y  = by prog[s]:  0  v * (k | 1)
//!                       1  v xor mulhi(v, k)
//!                       2  v + k / (lo32(v) | 0x8000_0001)
//!                       3  lo64(AESENC(v ‖ k, 0)), with h = its hi64
//!                       4  v rotr (k mod 64)
//!                       5  v + k
//!                       6  v - k
//!                       7  v xor k
//!     out = y xor m1 xor m3
//!     r[s] = out;  r[t] = r[t] xor h;  c = out      (h = 0 unless the AES operation ran)
//!     pad[a][j] = m0 xor y;  pad[b][j xor 4] = m1 + x;  pad[e][j xor 2] = m2 + y;  pad[f][j xor 6] = m3 xor x
//!     if (i + 1) mod REMIX = 0:
//!         o = BLAKE3-XOF(derive_key "midstate walk v1 remix", r)[0..72]
//!         r = o[0..64];  prog = program(o[64..72])      (c carries on)
//!
//! final_hash = BLAKE3(derive_key "midstate walk v1 final", mining_hash ‖ le64(nonce) ‖ r)
//!
//! program(z): p = [0, 1, .., 7];  for i = 7 down to 1: swap p[i], p[z mod (i+1)];  z = z div (i+1)
//! ```
//!
//! `AESENC` is one AES encryption round exactly as x86's `AESENC` instruction
//! computes it: MixColumns(ShiftRows(SubBytes(state))) xor key. On ARMv8 it is
//! `AESMC(AESE(state, 0)) xor key`. A portable table implementation stands in
//! on hardware without either; all three are checked against each other and
//! against FIPS-197 below.
//!
//! A program is a permutation, so every program runs each operation exactly
//! once per eight steps and no attempt is cheaper than another. The partner
//! is never the step's own register (that would let `v xor k` cancel `x`).
//! The four words are `j`, `j xor 4`, `j xor 2` and `j xor 6`, distinct even
//! when lines coincide, so the four writes never collide and their order is
//! immaterial.
//!
//! `c` makes the walk one chain: a step's addresses depend on the step
//! before, so no step's loads can start before the previous step's loads and
//! operation have finished. A CPU core hits L2 in a few nanoseconds; a GPU
//! thread waits hundreds of cycles for its L2 or VRAM on every step and
//! cannot run ahead, so its only recourse is more attempts at once, each
//! needing its own 256 KiB, which pushes them out of L2 and into VRAM. There
//! each step costs four 32-byte sectors read and four written back, where a
//! CPU core keeps all four loads of a step in flight together.

use std::cell::RefCell;

/// Lines of 64 bytes in the scratchpad: 256 KiB.
#[cfg(not(feature = "fast-mining"))]
pub const WALK_LINES: usize = 4096;
/// Steps per attempt.
#[cfg(not(feature = "fast-mining"))]
pub const WALK_STEPS: usize = 1 << 18;
/// Steps between BLAKE3 re-mixes of the registers.
#[cfg(not(feature = "fast-mining"))]
pub const WALK_REMIX: usize = 1 << 12;

/// Test builds: a 16 KiB pad and a thousand steps, so tests mine instantly.
#[cfg(feature = "fast-mining")]
pub const WALK_LINES: usize = 256;
#[cfg(feature = "fast-mining")]
pub const WALK_STEPS: usize = 1 << 10;
#[cfg(feature = "fast-mining")]
pub const WALK_REMIX: usize = 1 << 8;

/// Line indices are 12-bit fields of `x` (bits 0, 17, 32 and 44 up).
const _: () = assert!(WALK_LINES.is_power_of_two() && WALK_LINES <= 1 << 12);
const _: () = assert!(WALK_REMIX.is_power_of_two() && WALK_STEPS % WALK_REMIX == 0);

const CTX_INIT: &str = "midstate walk v1 init";
const CTX_REMIX: &str = "midstate walk v1 remix";
const CTX_FINAL: &str = "midstate walk v1 final";

/// One 64-byte line of the scratchpad.
pub type Line = [u64; 8];
/// The scratchpad. Reused per thread; every attempt overwrites all of it.
pub type Pad = [Line; WALK_LINES];

/// Allocates a scratchpad on the heap (too big for most thread stacks).
pub fn new_pad() -> Box<Pad> {
    vec![[0u64; 8]; WALK_LINES]
        .into_boxed_slice()
        .try_into()
        .expect("vec has WALK_LINES lines")
}

thread_local! {
    static PAD: RefCell<Box<Pad>> = RefCell::new(new_pad());
    /// This thread's scratchpads for [`walk_hash_batch`], grown as needed.
    static PADS: RefCell<Vec<Box<Pad>>> = const { RefCell::new(Vec::new()) };
}

/// The proof-of-work hash of `nonce` over `mining_hash`.
///
/// Uses this thread's scratchpad and the fastest AES this CPU has. The
/// result is identical on every platform and every code path.
pub fn walk_hash(mining_hash: &[u8; 32], nonce: u64) -> [u8; 32] {
    PAD.with(|pad| match pad.try_borrow_mut() {
        Ok(mut pad) => walk_with_pad(mining_hash, nonce, &mut pad),
        // Only reachable if a caller re-enters from inside a walk; correct, just slower.
        Err(_) => walk_with_pad(mining_hash, nonce, &mut new_pad()),
    })
}

/// [`walk_hash`] with a caller-owned scratchpad.
pub fn walk_with_pad(mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> [u8; 32] {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => unsafe { x86::walk(mining_hash, nonce, pad) },
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => unsafe { arm::walk(mining_hash, nonce, pad) },
        _ => walk_generic::<SoftAes>(mining_hash, nonce, pad),
    }
}

/// The table-based reference path, whatever the CPU offers. For tests and
/// for anyone checking the hardware paths against it.
pub fn walk_hash_portable(mining_hash: &[u8; 32], nonce: u64) -> [u8; 32] {
    walk_generic::<SoftAes>(mining_hash, nonce, &mut new_pad())
}

/// Which AES implementation [`walk_hash`] uses on this CPU.
pub fn backend_name() -> &'static str {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => "aes-ni",
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => "armv8-crypto",
        _ => "portable",
    }
}

// ── Known answers ───────────────────────────────────────────────────────────

/// Pinned outputs, as (mining hash, nonce, proof-of-work hash) in hex.
/// Cross-checked against two independent implementations written from the
/// definition in this module (midstate's `docs/walk-reference`): a Python one
/// (S-box derived from GF(2^8) inversion) and a C one run on an emulated
/// Cortex-A76 through the ARMv8 AES instructions. Changing any of these is a
/// hard fork.
///
/// The tests check them, and so does [`check_golden`], which an overclocked
/// miner runs to find out whether its CPU still computes the walk correctly.
#[cfg(not(feature = "fast-mining"))]
pub const GOLDEN: [(&str, u64, &str); 3] = [
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        0,
        "a934fced2d9a53fb09d6abe089f3d2f5bba9a421b41ee5726233a525cc7438e7",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        u64::MAX,
        "65d237f595f07bee156c635ad45f721987ce38d79ea14c89e40c5555613d0768",
    ),
    (
        "374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50",
        42,
        "5f2da9d320de391aef28bcaab63e3d1a25402b4541cdff8680059a87c7d515b7",
    ),
];

#[cfg(feature = "fast-mining")]
pub const GOLDEN: [(&str, u64, &str); 3] = [
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        0,
        "592cd5f394f7dc705343585a7cb48c58ce8da5f646dadb81af440e0d7a470e40",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        u64::MAX,
        "f1488d17741b4eaaa909374c90ea70adfe89ceaa16d7ca8605c0da25f031e9f3",
    ),
    (
        "374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50",
        42,
        "8f9d7f1d42826a15f81dbee9fd7e25ca3edee11776dfb48e078feebfeab2ab13",
    ),
];

/// Whether this thread's CPU computes [`GOLDEN`] exactly, through
/// [`walk_hash`] (the fast path this CPU mines with).
///
/// A CPU pushed past what it can do stays up but computes wrong results now
/// and then. A miner like that loses shares (the pool re-checks each one)
/// rather than producing bad blocks, but it is wasting its power: an
/// overclock is only good if this holds on every core, while hot.
pub fn check_golden() -> bool {
    GOLDEN.iter().all(|(mining_hash, nonce, want)| {
        let (mut m, mut w) = ([0u8; 32], [0u8; 32]);
        hex::decode_to_slice(mining_hash, &mut m).is_ok()
            && hex::decode_to_slice(want, &mut w).is_ok()
            && walk_hash(&m, *nonce) == w
    })
}

/// Walk attempts a second over `threads` threads for `duration`, and whether
/// every thread computed [`GOLDEN`] correctly both before and after (so the
/// second check runs on a hot CPU). For the miner's "test this unit".
pub fn bench(threads: usize, duration: std::time::Duration) -> (f64, bool) {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    let mining_hash = *blake3::hash(b"midstate walk bench").as_bytes();
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));
    let start = std::time::Instant::now();
    let workers: Vec<_> = (0..threads.max(1) as u64)
        .map(|t| {
            let (stop, done) = (stop.clone(), done.clone());
            std::thread::spawn(move || {
                let mut good = check_golden();
                let mut nonce = t << 40;
                while !stop.load(Ordering::Relaxed) {
                    std::hint::black_box(walk_hash(&mining_hash, nonce));
                    nonce += 1;
                    done.fetch_add(1, Ordering::Relaxed);
                }
                good &= check_golden();
                good
            })
        })
        .collect();
    std::thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);
    let good = workers.into_iter().all(|w| w.join().unwrap_or(false));
    (done.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64(), good)
}

// ── The walk ────────────────────────────────────────────────────────────────

#[inline(always)]
fn le64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
}

#[inline(always)]
fn regs_bytes(r: &[u64; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (k, w) in r.iter().enumerate() {
        out[8 * k..8 * k + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// The program: which operation each of the eight registers runs, a
/// permutation of the eight operations decoded from `z` (Fisher–Yates with
/// mixed-radix digits of `z`). Every program runs every operation exactly
/// once per eight steps, so no program is cheaper than another.
#[inline(always)]
fn program(mut z: u64) -> [u8; 8] {
    let mut p = [0u8, 1, 2, 3, 4, 5, 6, 7];
    for i in (1..8usize).rev() {
        let n = i as u64 + 1;
        p.swap(i, (z % n) as usize);
        z /= n;
    }
    p
}

/// Re-keys the registers and draws the next program with BLAKE3, so no
/// register state can persist or degenerate for longer than `WALK_REMIX`
/// steps.
#[inline(never)]
fn remix(r: &mut [u64; 8], prog: &mut [u8; 8]) {
    let mut out = [0u8; 72];
    blake3::Hasher::new_derive_key(CTX_REMIX)
        .update(&regs_bytes(r))
        .finalize_xof()
        .fill(&mut out);
    for (k, w) in r.iter_mut().enumerate() {
        *w = le64(&out[8 * k..]);
    }
    *prog = program(le64(&out[64..]));
}

#[inline(always)]
fn mulhi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// One step of the walk on register `s` with operation `op`. `c` carries
/// the previous step's output into this step's addresses, so that every
/// step waits for the one before it.
///
/// This is the step as the walk defines it, and the reference the other ways
/// of writing it are tested against. The CPU miner runs the same step written
/// one of two faster ways, [`step_unindexed`] or [`step_ahead`]: see
/// [`MINER_STEP`].
#[inline(always)]
fn step<A: AesRound>(s: usize, op: u8, r: &mut [u64; 8], c: &mut u64, pad: &mut Pad) {
    const MASK: u64 = WALK_LINES as u64 - 1;
    let x = r[s] ^ *c;
    let a = (x & MASK) as usize;
    let t = (s + 1 + ((x >> 12) & 3) as usize) & 7;
    let j = ((x >> 14) & 7) as usize;
    let e = ((x >> 17) & MASK) as usize;
    let b = ((x >> 32) & MASK) as usize;
    let f = ((x >> 44) & MASK) as usize;

    // Four independent loads: a CPU core has them all in flight at once.
    let m0 = pad[a][j];
    let m1 = pad[b][j ^ 4];
    let m2 = pad[e][j ^ 2];
    let m3 = pad[f][j ^ 6];
    let k = r[t];
    let v = x ^ m0 ^ m2;

    let mut hi = 0u64;
    let y = match op {
        0 => v.wrapping_mul(k | 1),
        1 => v ^ mulhi(v, k),
        2 => v.wrapping_add(k / ((v as u32 as u64) | 0x8000_0001)),
        3 => {
            let (lo, h) = A::round_pair(v, k);
            hi = h;
            lo
        }
        4 => v.rotate_right((k & 63) as u32),
        5 => v.wrapping_add(k),
        6 => v.wrapping_sub(k),
        _ => v ^ k,
    };
    let out = y ^ m1 ^ m3;
    r[s] = out;
    r[t] ^= hi;
    *c = out;

    // Four distinct words even when lines coincide ({j, j^4, j^2, j^6}).
    pad[a][j] = m0 ^ y;
    pad[b][j ^ 4] = m1.wrapping_add(x);
    pad[e][j ^ 2] = m2.wrapping_add(y);
    pad[f][j ^ 6] = m3 ^ x;
}

/// Seeds a walk: its registers and first program from BLAKE3, and `pad`
/// grown from its four AES lanes.
#[inline(always)]
fn start_generic<A: AesRound>(mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> ([u64; 8], [u8; 8]) {
    let mut seed = [0u8; 200];
    blake3::Hasher::new_derive_key(CTX_INIT)
        .update(mining_hash)
        .update(&nonce.to_le_bytes())
        .finalize_xof()
        .fill(&mut seed);

    let mut r = [0u64; 8];
    for (k, w) in r.iter_mut().enumerate() {
        *w = le64(&seed[8 * k..]);
    }
    let prog = program(le64(&seed[192..]));

    // Grow the pad from four AES lanes, one round per 16 bytes.
    let mut lanes = [[0u8; 16]; 4];
    let mut keys = [[0u8; 16]; 4];
    for q in 0..4 {
        lanes[q].copy_from_slice(&seed[64 + 16 * q..80 + 16 * q]);
        keys[q].copy_from_slice(&seed[128 + 16 * q..144 + 16 * q]);
    }
    for line in pad.iter_mut() {
        for q in 0..4 {
            lanes[q] = A::round(lanes[q], keys[q]);
            line[2 * q] = le64(&lanes[q][..8]);
            line[2 * q + 1] = le64(&lanes[q][8..]);
        }
    }
    (r, prog)
}

/// The proof-of-work hash of a walk's last registers.
#[inline(always)]
fn finish(mining_hash: &[u8; 32], nonce: u64, r: &[u64; 8]) -> [u8; 32] {
    *blake3::Hasher::new_derive_key(CTX_FINAL)
        .update(mining_hash)
        .update(&nonce.to_le_bytes())
        .update(&regs_bytes(r))
        .finalize()
        .as_bytes()
}

#[inline(always)]
fn walk_generic<A: AesRound>(mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> [u8; 32] {
    walk_written::<A>(MINER_STEP, mining_hash, nonce, pad)
}

/// How the miner writes the walk's step on this CPU: the fastest of the ways
/// `cpu-probe` measures that has been measured here. Each computes the same
/// hash (tested), so this is a choice of speed only. On a Pi 5 (Cortex-A76)
/// [`step_unindexed`] takes 28.3 cycles to [`step`]'s 29.8 (5% less;
/// [`step_ahead`] not yet measured there). On a Xeon, whole walks on one
/// thread are 5% faster with `step_ahead` than with `step`, and 3% faster
/// with `step_unindexed`.
#[cfg(target_arch = "aarch64")]
pub const MINER_STEP: StepCode = StepCode::Unindexed;
/// How the miner writes the walk's step on this CPU: see the aarch64 one.
#[cfg(not(target_arch = "aarch64"))]
pub const MINER_STEP: StepCode = StepCode::Ahead;

/// The walk with its step written as `code` says (not [`StepCode::NoStores`],
/// which is not the walk). Eight steps a round, one per register, written out
/// so that each step's operation is a branch whose target is fixed for the
/// whole program: a CPU predicts all eight, while a GPU warp's threads (each
/// with its own program) diverge.
#[inline(always)]
fn walk_written<A: AesRound>(code: StepCode, mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> [u8; 32] {
    let (mut r, mut prog) = start_generic::<A>(mining_hash, nonce, pad);
    match code {
        StepCode::Ahead => {
            let mut x = r[0];
            for round in 0..WALK_STEPS / 8 {
                let (next, out) = round_ahead::<A>(&prog, x, &mut r, pad);
                x = next;
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    remix(&mut r, &mut prog);
                    x = r[0] ^ out;
                }
            }
        }
        StepCode::Unindexed => {
            let mut c = 0u64;
            for round in 0..WALK_STEPS / 8 {
                round_unindexed::<A, true>(&prog, &mut r, &mut c, pad);
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    remix(&mut r, &mut prog);
                }
            }
        }
        StepCode::Indexed => {
            let mut c = 0u64;
            for round in 0..WALK_STEPS / 8 {
                round_indexed::<A>(&prog, &mut r, &mut c, pad);
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    remix(&mut r, &mut prog);
                }
            }
        }
        StepCode::NoStores => panic!("a walk without its writes to the pad is not the walk"),
    }
    finish(mining_hash, nonce, &r)
}

/// Eight steps (a round) of [`step`].
#[inline(always)]
fn round_indexed<A: AesRound>(prog: &[u8; 8], r: &mut [u64; 8], c: &mut u64, pad: &mut Pad) {
    step::<A>(0, prog[0], r, c, pad);
    step::<A>(1, prog[1], r, c, pad);
    step::<A>(2, prog[2], r, c, pad);
    step::<A>(3, prog[3], r, c, pad);
    step::<A>(4, prog[4], r, c, pad);
    step::<A>(5, prog[5], r, c, pad);
    step::<A>(6, prog[6], r, c, pad);
    step::<A>(7, prog[7], r, c, pad);
}

// ── Several walks per thread ────────────────────────────────────────────────
//
// One walk keeps a core waiting: every step's loads depend on the step
// before, so an out-of-order core that could issue four instructions a cycle
// averages about one. Walks of other nonces are independent, so a thread
// can step several in turn (A0, B0, A1, B1, ...) and the core overlaps
// their waits, as SMT does on CPUs that have it. Their pads share the core's
// L2, which two of them fill on a Cortex-A76: whether it pays is measured
// (`midstate bench-pow --interleave`), not assumed. Each hash is its walk's.

/// The walks of `nonces`, `K` of them stepped in turn on this thread, with a
/// scratchpad each, their steps written as `code` says (not
/// [`StepCode::NoStores`]). Step by step (A0 B0 C0 A1 ...), so that the walks'
/// loads sit close together in the instruction stream, where an out-of-order
/// core issues them together.
#[inline(always)]
fn walk_many_written<A: AesRound, const K: usize>(
    code: StepCode,
    mining_hash: &[u8; 32],
    nonces: &[u64; K],
    pads: &mut [Box<Pad>; K],
) -> [[u8; 32]; K] {
    let mut r = [[0u64; 8]; K];
    let mut prog = [[0u8; 8]; K];
    for k in 0..K {
        (r[k], prog[k]) = start_generic::<A>(mining_hash, nonces[k], &mut pads[k]);
    }
    match code {
        StepCode::Ahead => {
            let mut x: [u64; K] = std::array::from_fn(|k| r[k][0]);
            let mut out = [0u64; K];
            for round in 0..WALK_STEPS / 8 {
                macro_rules! steps {
                    ($($s:literal)*) => {$(
                        for k in 0..K {
                            (x[k], out[k]) = step_ahead::<A, $s>(prog[k][$s], x[k], &mut r[k], &mut pads[k]);
                        }
                    )*};
                }
                steps!(0 1 2 3 4 5 6 7);
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    for k in 0..K {
                        remix(&mut r[k], &mut prog[k]);
                        x[k] = r[k][0] ^ out[k];
                    }
                }
            }
        }
        StepCode::Unindexed => {
            let mut c = [0u64; K];
            for round in 0..WALK_STEPS / 8 {
                macro_rules! steps {
                    ($($s:literal)*) => {$(
                        for k in 0..K {
                            step_unindexed::<A, $s, true>(prog[k][$s], &mut r[k], &mut c[k], &mut pads[k]);
                        }
                    )*};
                }
                steps!(0 1 2 3 4 5 6 7);
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    for k in 0..K {
                        remix(&mut r[k], &mut prog[k]);
                    }
                }
            }
        }
        StepCode::Indexed => {
            let mut c = [0u64; K];
            for round in 0..WALK_STEPS / 8 {
                macro_rules! steps {
                    ($($s:literal)*) => {$(
                        for k in 0..K {
                            step::<A>($s, prog[k][$s], &mut r[k], &mut c[k], &mut pads[k]);
                        }
                    )*};
                }
                steps!(0 1 2 3 4 5 6 7);
                if (round + 1) % (WALK_REMIX / 8) == 0 {
                    for k in 0..K {
                        remix(&mut r[k], &mut prog[k]);
                    }
                }
            }
        }
        StepCode::NoStores => panic!("a walk without its writes to the pad is not the walk"),
    }
    std::array::from_fn(|k| finish(mining_hash, nonces[k], &r[k]))
}

/// How the miner writes the step with `k` walks per thread: [`MINER_STEP`]
/// for one, and [`step`] (indexed) for more. The faster ways keep a walk's
/// eight registers in the CPU's; with several walks they cannot all fit. On
/// a Xeon, 3 or 4 walks per thread did 2-13% better indexed than unindexed
/// (2 did 3% worse, but 3 or 4 beat 2), and with `step_ahead`, 3 or 4 did
/// worse than one.
pub fn miner_step_for(k: usize) -> StepCode {
    if k <= 1 {
        MINER_STEP
    } else {
        StepCode::Indexed
    }
}

/// The proof-of-work hashes of `K` nonces, their walks stepped in turn on
/// this thread (`pads`: one scratchpad each), with the fastest AES this CPU
/// has and the miner's step for `K` walks ([`miner_step_for`]). Each is
/// exactly [`walk_hash`] of its nonce.
pub fn walk_hash_many<const K: usize>(mining_hash: &[u8; 32], nonces: &[u64; K], pads: &mut [Box<Pad>; K]) -> [[u8; 32]; K] {
    walk_hash_many_written::<K>(miner_step_for(K), mining_hash, nonces, pads)
}

/// [`walk_hash_many`] with the step written as `code` says: for measuring
/// the ways against each other.
pub fn walk_hash_many_written<const K: usize>(
    code: StepCode,
    mining_hash: &[u8; 32],
    nonces: &[u64; K],
    pads: &mut [Box<Pad>; K],
) -> [[u8; 32]; K] {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => unsafe { x86::walk_many(code, mining_hash, nonces, pads) },
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => unsafe { arm::walk_many(code, mining_hash, nonces, pads) },
        _ => walk_many_written::<SoftAes, K>(code, mining_hash, nonces, pads),
    }
}

/// Pins this thread to CPU `cpu`, where the system allows (Linux). Whether
/// it did.
pub fn pin_to_cpu(cpu: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
        }
        let mut set = [0u64; 16]; // a cpu_set_t: 1024 CPUs
        if cpu >= 1024 {
            return false;
        }
        set[cpu / 64] |= 1 << (cpu % 64);
        // SAFETY: pid 0 is this thread; the mask is a whole cpu_set_t.
        unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        false
    }
}

/// Attempts a second with `K` walks stepped in turn per thread, over
/// `threads` threads for `duration`, pinned to a core each if `pin`; and
/// whether every thread computed [`GOLDEN`] right that way before and after.
pub fn bench_many<const K: usize>(threads: usize, duration: std::time::Duration, pin: bool) -> (f64, bool) {
    bench_many_written::<K>(miner_step_for(K), threads, duration, pin)
}

/// [`bench_many`] with the step written as `code` says. Each thread checks the
/// known answers before and after, and only the hashing between is timed.
pub fn bench_many_written<const K: usize>(code: StepCode, threads: usize, duration: std::time::Duration, pin: bool) -> (f64, bool) {
    let (rate, good) = time_many::<K>(code, threads, duration, pin, true);
    (rate, good)
}

/// Attempts a second over `threads` threads stepping `K` walks each, for
/// `duration` of hashing; with `check`, whether every thread computed the
/// known answers that way before and after.
fn time_many<const K: usize>(code: StepCode, threads: usize, duration: std::time::Duration, pin: bool, check: bool) -> (f64, bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    let mining_hash = *blake3::hash(b"midstate walk bench").as_bytes();
    let threads = threads.max(1);
    let stop = Arc::new(AtomicBool::new(false));
    // Every thread starts hashing together, after its checks.
    let ready = Arc::new(Barrier::new(threads + 1));
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let workers: Vec<_> = (0..threads)
        .map(|t| {
            let (stop, ready) = (stop.clone(), ready.clone());
            std::thread::spawn(move || {
                if pin {
                    pin_to_cpu(t % cores);
                }
                let mut pads: [Box<Pad>; K] = std::array::from_fn(|_| new_pad());
                let mut good = !check || golden_many::<K>(code, &mut pads);
                let mut nonce = (t as u64) << 40;
                ready.wait();
                let begun = std::time::Instant::now();
                let mut done = 0u64;
                // At least one batch each, however short the run.
                loop {
                    let nonces: [u64; K] = std::array::from_fn(|k| nonce + k as u64);
                    std::hint::black_box(walk_hash_many_written::<K>(code, &mining_hash, &nonces, &mut pads));
                    nonce += K as u64;
                    done += K as u64;
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                }
                let rate = done as f64 / begun.elapsed().as_secs_f64();
                good &= !check || golden_many::<K>(code, &mut pads);
                (rate, good)
            })
        })
        .collect();
    ready.wait();
    std::thread::sleep(duration);
    stop.store(true, Ordering::Relaxed);
    workers.into_iter().fold((0.0, true), |(rate, good), w| match w.join() {
        Ok((r, g)) => (rate + r, good && g),
        Err(_) => (rate, false),
    })
}

/// Most walks [`walk_hash_batch`] steps in turn on one thread.
pub const MAX_WALKS_PER_THREAD: usize = 4;

/// The walk hashes of `nonces` (1 to [`MAX_WALKS_PER_THREAD`] of them), their
/// walks stepped in turn on this thread with its own scratchpads: what the
/// CPU miner runs, [`walks_per_thread`] nonces at a time. Each is exactly
/// [`walk_hash`] of its nonce.
pub fn walk_hash_batch(mining_hash: &[u8; 32], nonces: &[u64]) -> Vec<(u64, [u8; 32])> {
    fn run<const K: usize>(m: &[u8; 32], nonces: &[u64], pads: &mut Vec<Box<Pad>>) -> Vec<(u64, [u8; 32])> {
        while pads.len() < K {
            pads.push(new_pad());
        }
        let pads: &mut [Box<Pad>; K] = (&mut pads[..K]).try_into().expect("K pads");
        let nonces: [u64; K] = nonces.try_into().expect("K nonces");
        let hashes = walk_hash_many::<K>(m, &nonces, pads);
        nonces.into_iter().zip(hashes).collect()
    }
    PADS.with(|pads| {
        let Ok(mut pads) = pads.try_borrow_mut() else {
            // Only reachable if a caller re-enters from inside a walk.
            return nonces.iter().map(|&n| (n, walk_hash(mining_hash, n))).collect();
        };
        let mut out = Vec::with_capacity(nonces.len());
        for chunk in nonces.chunks(MAX_WALKS_PER_THREAD) {
            out.extend(match chunk.len() {
                1 => run::<1>(mining_hash, chunk, &mut pads),
                2 => run::<2>(mining_hash, chunk, &mut pads),
                3 => run::<3>(mining_hash, chunk, &mut pads),
                _ => run::<4>(mining_hash, chunk, &mut pads),
            });
        }
        out
    })
}

/// How many walks the CPU miner steps in turn per thread, chosen once per
/// process: `MINER_WALKS_PER_THREAD` (1 to 4) if it is set, otherwise
/// measured on `threads` threads (all of the miner's, so that their pads
/// share the caches as they will while mining).
///
/// A walk's core spends most of each step waiting for the step's load from
/// the pad, so several walks per thread fill the waits, until their pads no
/// longer fit the core's cache: on a 2-core Xeon with 2 MiB of L2 a core, 3
/// walks a thread did 87% more than one; on a Pi 5, whose 512 KiB of L2 holds
/// one pad well, 2 did 75% less. So it is measured, not assumed.
pub fn walks_per_thread(threads: usize) -> usize {
    static CHOSEN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(|| choose_walks_per_thread(threads.max(1)))
}

fn choose_walks_per_thread(threads: usize) -> usize {
    if let Ok(v) = std::env::var("MINER_WALKS_PER_THREAD") {
        match v.trim().parse::<usize>() {
            Ok(k) if (1..=MAX_WALKS_PER_THREAD).contains(&k) => {
                tracing::info!("Midstate Walk: {k} walk(s) per thread, from MINER_WALKS_PER_THREAD");
                return k;
            }
            _ => tracing::warn!("MINER_WALKS_PER_THREAD={v:?} is not 1 to {MAX_WALKS_PER_THREAD}: measuring instead"),
        }
    }
    // About 6 s once per process, before the first Walk job mines (tests: a
    // fraction of a second).
    let each = std::time::Duration::from_millis(if cfg!(test) { 30 } else { 700 });
    let rates = measure_walks_per_thread(threads, each);
    let best = (1..=MAX_WALKS_PER_THREAD).max_by(|a, b| rates[a - 1].total_cmp(&rates[b - 1])).unwrap_or(1);
    // The interleaved walk must give the known answers on this CPU before it
    // mines; if it does not, one walk per thread (walk_hash) does.
    let code = miner_step_for(best);
    let good = match best {
        1 => true,
        2 => golden_many::<2>(code, &mut std::array::from_fn(|_| new_pad())),
        3 => golden_many::<3>(code, &mut std::array::from_fn(|_| new_pad())),
        _ => golden_many::<4>(code, &mut std::array::from_fn(|_| new_pad())),
    };
    let summary: Vec<String> = rates.iter().enumerate().map(|(i, r)| format!("{}: {r:.0}/s", i + 1)).collect();
    if !good {
        tracing::error!("Midstate Walk: {best} walks per thread got a known answer wrong; mining one walk per thread");
        return 1;
    }
    tracing::info!(
        "Midstate Walk: attempts with 1 to {MAX_WALKS_PER_THREAD} walks per thread on {threads} threads: {}; mining with {best} (MINER_WALKS_PER_THREAD sets it)",
        summary.join(", ")
    );
    best
}

/// Attempts a second with 1 to [`MAX_WALKS_PER_THREAD`] walks per thread on
/// `threads` threads, each with the step the miner would use for it
/// ([`miner_step_for`]) and measured twice, up the counts and back down, so
/// that a CPU warming up does not favour the ones measured first.
pub fn measure_walks_per_thread(threads: usize, each: std::time::Duration) -> [f64; MAX_WALKS_PER_THREAD] {
    let once = |k: usize| match k {
        1 => time_many::<1>(miner_step_for(1), threads, each, false, false).0,
        2 => time_many::<2>(miner_step_for(2), threads, each, false, false).0,
        3 => time_many::<3>(miner_step_for(3), threads, each, false, false).0,
        _ => time_many::<4>(miner_step_for(4), threads, each, false, false).0,
    };
    let mut rates = [0.0; MAX_WALKS_PER_THREAD];
    for k in (1..=MAX_WALKS_PER_THREAD).chain((1..=MAX_WALKS_PER_THREAD).rev()) {
        rates[k - 1] += once(k) / 2.0;
    }
    rates
}

/// `bench-pow --interleave`: attempts a second with 1 to 4 walks stepped in
/// turn per thread, for each way of writing the step, each checking the known
/// answers, with the temperature (`temp`, °C) after each row; then the count
/// the miner would choose here. Prints as it goes.
pub fn print_interleave(seconds: u64, threads: usize, temp: fn() -> Option<f32>) {
    let threads = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { threads };
    let d = std::time::Duration::from_secs(seconds.max(1));
    println!(
        "Midstate Walk, walks stepped in turn per thread: {threads} threads, {}s each, AES: {}",
        d.as_secs(),
        backend_name()
    );
    let codes = [(StepCode::Indexed, "indexed"), (StepCode::Unindexed, "unindexed"), (StepCode::Ahead, "ahead")];
    print!("  {:<18}", "attempts/s");
    for (_, name) in &codes {
        print!("{name:>13}");
    }
    println!();
    let mut wrong = false;
    for k in 1..=MAX_WALKS_PER_THREAD {
        print!("  {k} walk{} per thread ", if k == 1 { " " } else { "s" });
        for (code, _) in &codes {
            let (rate, good) = match k {
                1 => bench_many_written::<1>(*code, threads, d, false),
                2 => bench_many_written::<2>(*code, threads, d, false),
                3 => bench_many_written::<3>(*code, threads, d, false),
                _ => bench_many_written::<4>(*code, threads, d, false),
            };
            wrong |= !good;
            let mark = if !good { "!" } else if *code == miner_step_for(k) { "*" } else { " " };
            print!("{rate:>12.1}{mark}");
        }
        let t = temp().map(|t| format!("  {t:.1} °C")).unwrap_or_default();
        println!("{t}");
    }
    if wrong {
        println!("  ! a known answer came out wrong that way");
    }
    println!("  * the step this CPU's miner uses with that many walks per thread");
    let rates = measure_walks_per_thread(threads, std::time::Duration::from_millis(700));
    let best = (1..=MAX_WALKS_PER_THREAD).max_by(|a, b| rates[a - 1].total_cmp(&rates[b - 1])).unwrap_or(1);
    println!("The miner measures this itself when it starts (MINER_WALKS_PER_THREAD sets it) and would mine here with {best} walk(s) per thread.");
    println!("Rows run one after another, so a board that heats up throttles the later ones: compare the temperatures.");
}

/// Whether [`walk_hash_many`] computes [`GOLDEN`], the known answers spread
/// over its `K` walks (each walk's own mining hash and nonce).
fn golden_many<const K: usize>(code: StepCode, pads: &mut [Box<Pad>; K]) -> bool {
    let decode = |h: &str| {
        let mut b = [0u8; 32];
        hex::decode_to_slice(h, &mut b).map(|_| b)
    };
    // Walks in one call share a mining hash, so each known answer is run
    // in every slot, with the other slots running other nonces beside it.
    (0..K).all(|slot| {
        GOLDEN.iter().all(|(m, nonce, want)| {
            let (Ok(m), Ok(want)) = (decode(m), decode(want)) else { return false };
            let nonces: [u64; K] = std::array::from_fn(|k| if k == slot { *nonce } else { nonce.wrapping_add(1 + k as u64) });
            walk_hash_many_written::<K>(code, &m, &nonces, pads)[slot] == want
        })
    })
}

// ── Ways of writing a step, and where its time goes ─────────────────────────

/// A step exactly as [`step`] computes it, written so that no register goes
/// through memory: what the miner runs on aarch64. The register written (`S`) is
/// known when it compiles, `k` is picked with two selects instead of an
/// index, and AES's high half is XORed in under masks on AES steps only.
/// `step` indexes the registers with `t`, so all eight live in memory, and
/// it XORs into `r[t]` on every step. On a Pi 5 this takes 28.3 cycles a
/// step to `step`'s 29.8.
///
/// `STORE` false leaves out the four writes to the pad: not the walk, only
/// for `cpu-probe` to measure what the writes cost.
#[inline(always)]
fn step_unindexed<A: AesRound, const S: usize, const STORE: bool>(op: u8, r: &mut [u64; 8], c: &mut u64, pad: &mut Pad) {
    const MASK: u64 = WALK_LINES as u64 - 1;
    let x = r[S] ^ *c;
    let a = (x & MASK) as usize;
    let off = (x >> 12) & 3;
    let j = ((x >> 14) & 7) as usize;
    let e = ((x >> 17) & MASK) as usize;
    let b = ((x >> 32) & MASK) as usize;
    let f = ((x >> 44) & MASK) as usize;
    let m0 = pad[a][j];
    let m1 = pad[b][j ^ 4];
    let m2 = pad[e][j ^ 2];
    let m3 = pad[f][j ^ 6];
    let (r1, r2, r3, r4) = (r[(S + 1) & 7], r[(S + 2) & 7], r[(S + 3) & 7], r[(S + 4) & 7]);
    let low_pair = if off & 1 == 0 { r1 } else { r2 };
    let high_pair = if off & 1 == 0 { r3 } else { r4 };
    let k = if off & 2 == 0 { low_pair } else { high_pair };
    let v = x ^ m0 ^ m2;
    let m13 = m1 ^ m3;
    let y = match op {
        0 => v.wrapping_mul(k | 1),
        1 => v ^ mulhi(v, k),
        2 => v.wrapping_add(k / ((v as u32 as u64) | 0x8000_0001)),
        3 => {
            let (lo, h) = A::round_pair(v, k);
            let pick = |d: u64| h & 0u64.wrapping_sub((off == d) as u64);
            r[(S + 1) & 7] ^= pick(0);
            r[(S + 2) & 7] ^= pick(1);
            r[(S + 3) & 7] ^= pick(2);
            r[(S + 4) & 7] ^= pick(3);
            lo
        }
        4 => v.rotate_right((k & 63) as u32),
        5 => v.wrapping_add(k),
        6 => v.wrapping_sub(k),
        _ => v ^ k,
    };
    let out = y ^ m13;
    r[S] = out;
    *c = out;
    if STORE {
        pad[a][j] = m0 ^ y;
        pad[b][j ^ 4] = m1.wrapping_add(x);
        pad[e][j ^ 2] = m2.wrapping_add(y);
        pad[f][j ^ 6] = m3 ^ x;
    }
}

/// Eight steps (a round) of [`step_unindexed`].
#[inline(always)]
fn round_unindexed<A: AesRound, const STORE: bool>(prog: &[u8; 8], r: &mut [u64; 8], c: &mut u64, pad: &mut Pad) {
    step_unindexed::<A, 0, STORE>(prog[0], r, c, pad);
    step_unindexed::<A, 1, STORE>(prog[1], r, c, pad);
    step_unindexed::<A, 2, STORE>(prog[2], r, c, pad);
    step_unindexed::<A, 3, STORE>(prog[3], r, c, pad);
    step_unindexed::<A, 4, STORE>(prog[4], r, c, pad);
    step_unindexed::<A, 5, STORE>(prog[5], r, c, pad);
    step_unindexed::<A, 6, STORE>(prog[6], r, c, pad);
    step_unindexed::<A, 7, STORE>(prog[7], r, c, pad);
}

/// [`step_unindexed`] handed its `x`, working out the next step's as
/// `(r[S+1] ^ m1 ^ m3) ^ y`: one XOR after the operation, where the step as
/// written has two (`out = y ^ m1 ^ m3`, then the next `x = r[S+1] ^ out`).
/// On AES steps the high half that goes into `r[S+1]` is XORed in too.
/// Returns the next step's `x` and this step's output. The same walk: what
/// the miner runs on x86.
#[inline(always)]
fn step_ahead<A: AesRound, const S: usize>(op: u8, x: u64, r: &mut [u64; 8], pad: &mut Pad) -> (u64, u64) {
    const MASK: u64 = WALK_LINES as u64 - 1;
    let a = (x & MASK) as usize;
    let off = (x >> 12) & 3;
    let j = ((x >> 14) & 7) as usize;
    let e = ((x >> 17) & MASK) as usize;
    let b = ((x >> 32) & MASK) as usize;
    let f = ((x >> 44) & MASK) as usize;
    let m0 = pad[a][j];
    let m1 = pad[b][j ^ 4];
    let m2 = pad[e][j ^ 2];
    let m3 = pad[f][j ^ 6];
    let (r1, r2, r3, r4) = (r[(S + 1) & 7], r[(S + 2) & 7], r[(S + 3) & 7], r[(S + 4) & 7]);
    let low_pair = if off & 1 == 0 { r1 } else { r2 };
    let high_pair = if off & 1 == 0 { r3 } else { r4 };
    let k = if off & 2 == 0 { low_pair } else { high_pair };
    let v = x ^ m0 ^ m2;
    let m13 = m1 ^ m3;
    let ahead = r1 ^ m13;
    let (y, next) = match op {
        3 => {
            let (lo, h) = A::round_pair(v, k);
            let pick = |d: u64| h & 0u64.wrapping_sub((off == d) as u64);
            r[(S + 1) & 7] ^= pick(0);
            r[(S + 2) & 7] ^= pick(1);
            r[(S + 3) & 7] ^= pick(2);
            r[(S + 4) & 7] ^= pick(3);
            (lo, (ahead ^ lo) ^ pick(0))
        }
        _ => {
            let y = match op {
                0 => v.wrapping_mul(k | 1),
                1 => v ^ mulhi(v, k),
                2 => v.wrapping_add(k / ((v as u32 as u64) | 0x8000_0001)),
                4 => v.rotate_right((k & 63) as u32),
                5 => v.wrapping_add(k),
                6 => v.wrapping_sub(k),
                _ => v ^ k,
            };
            (y, ahead ^ y)
        }
    };
    let out = y ^ m13;
    r[S] = out;
    pad[a][j] = m0 ^ y;
    pad[b][j ^ 4] = m1.wrapping_add(x);
    pad[e][j ^ 2] = m2.wrapping_add(y);
    pad[f][j ^ 6] = m3 ^ x;
    (next, out)
}

/// Eight steps (a round) of [`step_ahead`]: the next round's `x`, and the
/// last step's output.
#[inline(always)]
fn round_ahead<A: AesRound>(prog: &[u8; 8], x: u64, r: &mut [u64; 8], pad: &mut Pad) -> (u64, u64) {
    let (x, _) = step_ahead::<A, 0>(prog[0], x, r, pad);
    let (x, _) = step_ahead::<A, 1>(prog[1], x, r, pad);
    let (x, _) = step_ahead::<A, 2>(prog[2], x, r, pad);
    let (x, _) = step_ahead::<A, 3>(prog[3], x, r, pad);
    let (x, _) = step_ahead::<A, 4>(prog[4], x, r, pad);
    let (x, _) = step_ahead::<A, 5>(prog[5], x, r, pad);
    let (x, _) = step_ahead::<A, 6>(prog[6], x, r, pad);
    step_ahead::<A, 7>(prog[7], x, r, pad)
}

/// [`walk_hash`] with the step written as `code` says, with the fastest AES
/// this CPU has: for checking each way against the others.
pub fn walk_hash_written(code: StepCode, mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> [u8; 32] {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => unsafe { x86::walk_written(code, mining_hash, nonce, pad) },
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => unsafe { arm::walk_written(code, mining_hash, nonce, pad) },
        _ => walk_written::<SoftAes>(code, mining_hash, nonce, pad),
    }
}

/// Ways of writing the walk's step: all the same walk but the last.
/// [`MINER_STEP`] says which one mines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepCode {
    /// [`step`], the reference: what mined everywhere before these.
    Indexed,
    /// [`step_unindexed`].
    Unindexed,
    /// [`step_ahead`].
    Ahead,
    /// [`step_unindexed`] without its writes to the pad. Not the walk: what
    /// the writes cost.
    NoStores,
}

/// `n` steps (a multiple of 8) of a walk on this thread, with the walk's own
/// program or (`op`) every operation the same, unrolled as [`walk_generic`]
/// runs them; no re-mix. Nanoseconds a step.
#[inline(always)]
fn time_steps_generic<A: AesRound>(op: Option<u8>, n: usize, code: StepCode) -> f64 {
    let m = *blake3::hash(b"midstate walk step timing").as_bytes();
    let mut pad = new_pad();
    let (mut r, mut prog) = start_generic::<A>(&m, 1, &mut pad);
    if let Some(o) = op {
        prog = [o & 7; 8];
    }
    let mut c = 0u64;
    let begun = std::time::Instant::now();
    match code {
        StepCode::Unindexed => {
            for _ in 0..n / 8 {
                round_unindexed::<A, true>(&prog, &mut r, &mut c, &mut pad);
            }
        }
        StepCode::NoStores => {
            for _ in 0..n / 8 {
                round_unindexed::<A, false>(&prog, &mut r, &mut c, &mut pad);
            }
        }
        StepCode::Ahead => {
            let mut x = r[0];
            for _ in 0..n / 8 {
                (x, c) = round_ahead::<A>(&prog, x, &mut r, &mut pad);
            }
        }
        StepCode::Indexed => {
            for _ in 0..n / 8 {
                round_indexed::<A>(&prog, &mut r, &mut c, &mut pad);
            }
        }
    }
    let took = begun.elapsed();
    std::hint::black_box((&r, c, &pad[0]));
    took.as_nanos() as f64 / (n / 8 * 8).max(1) as f64
}

/// Nanoseconds a step of the walk takes on this thread, with the fastest AES
/// this CPU has, over `n` steps: with the walk's own program (`None`), or
/// with every operation `op`. The steps are written as `code` says; setting
/// one operation against another shows what each costs, and the XOR (the
/// cheapest) shows what the rest of a step costs.
pub fn time_steps(op: Option<u8>, n: usize, code: StepCode) -> f64 {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => unsafe { x86::time_steps(op, n, code) },
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => unsafe { arm::time_steps(op, n, code) },
        _ => time_steps_generic::<SoftAes>(op, n, code),
    }
}

/// Nanoseconds seeding a walk and growing its pad takes (once an attempt),
/// over `n` of them.
pub fn time_start(n: usize) -> f64 {
    let m = *blake3::hash(b"midstate walk step timing").as_bytes();
    let mut pad = new_pad();
    let begun = std::time::Instant::now();
    for i in 0..n {
        std::hint::black_box(walk_start(&m, i as u64, &mut pad));
    }
    begun.elapsed().as_nanos() as f64 / n.max(1) as f64
}

/// Nanoseconds a re-mix takes (64 an attempt), over `n` of them.
pub fn time_remix(n: usize) -> f64 {
    let mut r = [1u64, 2, 3, 4, 5, 6, 7, 8];
    let begun = std::time::Instant::now();
    for _ in 0..n {
        r = remix_of(std::hint::black_box(&r)).0;
    }
    std::hint::black_box(r);
    begun.elapsed().as_nanos() as f64 / n.max(1) as f64
}

// ── The walk in three parts, for GPUs ───────────────────────────────────────
//
// A GPU runs the steps (`core::wgpu_walk`); the CPU seeds each walk and grows
// its pad, which is BLAKE3 and AES and a few microseconds, and hashes what
// the steps leave. Together they are exactly `walk_hash`.

/// A walk's state after `steps` of its steps: its registers, its program,
/// and the previous step's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkState {
    pub r: [u64; 8],
    pub prog: [u8; 8],
    pub c: u64,
    pub steps: usize,
}

/// Seeds the walk of `nonce` and grows `pad`, with the fastest AES this CPU
/// has: the state before its first step.
pub fn walk_start(mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> WalkState {
    let (r, prog) = match backend() {
        #[cfg(target_arch = "x86_64")]
        Backend::AesNi => unsafe { x86::start(mining_hash, nonce, pad) },
        #[cfg(target_arch = "aarch64")]
        Backend::Armv8 => unsafe { arm::start(mining_hash, nonce, pad) },
        _ => start_generic::<SoftAes>(mining_hash, nonce, pad),
    };
    WalkState { r, prog, c: 0, steps: 0 }
}

/// Runs `n` more steps of a walk on the CPU (the portable path), re-mixing
/// where the walk does. The reference a GPU's partial runs are checked
/// against, a step at a time if need be.
pub fn walk_steps(st: &mut WalkState, pad: &mut Pad, n: usize) {
    assert!(st.steps + n <= WALK_STEPS, "within the walk");
    for _ in 0..n {
        let s = st.steps % 8;
        step::<SoftAes>(s, st.prog[s], &mut st.r, &mut st.c, pad);
        st.steps += 1;
        if st.steps % WALK_REMIX == 0 {
            remix(&mut st.r, &mut st.prog);
        }
    }
}

/// The operations by number, as a program names them.
pub const OP_NAMES: [&str; 8] = ["mul", "mulhi", "div", "aes", "rotr", "add", "sub", "xor"];

/// What a walk's next step works on: for finding where a GPU's walk goes
/// wrong. `words` are the four pad words' places (line * 8 + word), for
/// `m[0..4]` in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepInputs {
    /// The register the step writes.
    pub s: usize,
    pub op: u8,
    /// The register `k` comes from, which AES's high half is XORed into.
    pub t: usize,
    pub x: u64,
    pub k: u64,
    pub v: u64,
    pub m: [u64; 4],
    pub words: [usize; 4],
}

/// Operation `op` of the walk on `v` and `k`, the portable way: its result,
/// and the half AES XORs into register `t` (0 for the others).
pub fn apply_op(op: u8, v: u64, k: u64) -> (u64, u64) {
    match op {
        0 => (v.wrapping_mul(k | 1), 0),
        1 => (v ^ mulhi(v, k), 0),
        2 => (v.wrapping_add(k / ((v as u32 as u64) | 0x8000_0001)), 0),
        3 => SoftAes::round_pair(v, k),
        4 => (v.rotate_right((k & 63) as u32), 0),
        5 => (v.wrapping_add(k), 0),
        6 => (v.wrapping_sub(k), 0),
        _ => (v ^ k, 0),
    }
}

/// The inputs of `st`'s next step over `pad`, as [`step`] works them out.
pub fn step_inputs(st: &WalkState, pad: &Pad) -> StepInputs {
    const MASK: u64 = WALK_LINES as u64 - 1;
    let s = st.steps % 8;
    let x = st.r[s] ^ st.c;
    let a = (x & MASK) as usize;
    let t = (s + 1 + ((x >> 12) & 3) as usize) & 7;
    let j = ((x >> 14) & 7) as usize;
    let e = ((x >> 17) & MASK) as usize;
    let b = ((x >> 32) & MASK) as usize;
    let f = ((x >> 44) & MASK) as usize;
    let words = [a * 8 + j, b * 8 + (j ^ 4), e * 8 + (j ^ 2), f * 8 + (j ^ 6)];
    let m = words.map(|w| pad[w / 8][w % 8]);
    StepInputs { s, op: st.prog[s], t, x, k: st.r[t], v: x ^ m[0] ^ m[2], m, words }
}

/// The re-mix of registers `r`: the new registers and program, as the walk
/// makes them every [`WALK_REMIX`] steps.
pub fn remix_of(r: &[u64; 8]) -> ([u64; 8], [u8; 8]) {
    let (mut r, mut prog) = (*r, [0u8; 8]);
    remix(&mut r, &mut prog);
    (r, prog)
}

/// The proof-of-work hash of a walk whose steps have all run.
pub fn walk_finish(mining_hash: &[u8; 32], nonce: u64, r: &[u64; 8]) -> [u8; 32] {
    finish(mining_hash, nonce, r)
}

/// One AES round (`AESENC`) of `lo ‖ hi` under a zero key, as the walk's
/// AES operation computes it, through the portable table: for checking a
/// GPU's.
pub fn aes_round_pair(lo: u64, hi: u64) -> (u64, u64) {
    SoftAes::round_pair(lo, hi)
}

/// The program a 64-bit draw decodes to (which operation each register
/// runs): for checking a GPU's.
pub fn program_of(z: u64) -> [u8; 8] {
    program(z)
}

// ── One AES round ───────────────────────────────────────────────────────────

trait AesRound {
    /// `MixColumns(ShiftRows(SubBytes(state))) xor key`: x86 `AESENC`.
    fn round(state: [u8; 16], key: [u8; 16]) -> [u8; 16];

    /// [`AesRound::round`] of the 16 bytes `le64(lo) ‖ le64(hi)` under a zero
    /// key, as two words.
    #[inline(always)]
    fn round_pair(lo: u64, hi: u64) -> (u64, u64) {
        let mut s = [0u8; 16];
        s[..8].copy_from_slice(&lo.to_le_bytes());
        s[8..].copy_from_slice(&hi.to_le_bytes());
        let o = Self::round(s, [0u8; 16]);
        (le64(&o[..8]), le64(&o[8..]))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Backend {
    Portable,
    #[cfg(target_arch = "x86_64")]
    AesNi,
    #[cfg(target_arch = "aarch64")]
    Armv8,
}

fn backend() -> Backend {
    static CHOSEN: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
    *CHOSEN.get_or_init(detect)
}

#[allow(unreachable_code)]
fn detect() -> Backend {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("aes") && std::arch::is_x86_feature_detected!("sse2") {
        return Backend::AesNi;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("aes") {
        return Backend::Armv8;
    }
    Backend::Portable
}

/// The AES S-box (FIPS-197 Figure 7). The CUDA miner builds its table from
/// it too (`cuda_walk::te_words`).
pub(crate) const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// `TE[x]` is column `(2·S(x), S(x), S(x), 3·S(x))`: SubBytes and the
/// MixColumns contribution of a row-0 byte. Rows 1..3 are its byte rotations.
const TE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut x = 0;
    while x < 256 {
        let s = SBOX[x] as u32;
        let s2 = ((s << 1) ^ if s & 0x80 != 0 { 0x11b } else { 0 }) & 0xff;
        let s3 = s2 ^ s;
        t[x] = s2 | (s << 8) | (s << 16) | (s3 << 24);
        x += 1;
    }
    t
};

/// Portable AES round from the table above. Not constant time, which is
/// irrelevant here: nothing in a proof of work is secret.
struct SoftAes;

impl AesRound for SoftAes {
    #[inline(always)]
    fn round(st: [u8; 16], key: [u8; 16]) -> [u8; 16] {
        let mut out = [0u8; 16];
        for c in 0..4 {
            let w = TE[st[4 * c] as usize]
                ^ TE[st[4 * ((c + 1) & 3) + 1] as usize].rotate_left(8)
                ^ TE[st[4 * ((c + 2) & 3) + 2] as usize].rotate_left(16)
                ^ TE[st[4 * ((c + 3) & 3) + 3] as usize].rotate_left(24)
                ^ u32::from_le_bytes([key[4 * c], key[4 * c + 1], key[4 * c + 2], key[4 * c + 3]]);
            out[4 * c..4 * c + 4].copy_from_slice(&w.to_le_bytes());
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;

    pub(super) struct AesNi;

    impl super::AesRound for AesNi {
        #[inline(always)]
        fn round(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
            // SAFETY: only reached through `walk`, which requires AES-NI.
            unsafe {
                let s = _mm_loadu_si128(state.as_ptr() as *const __m128i);
                let k = _mm_loadu_si128(key.as_ptr() as *const __m128i);
                let mut out = [0u8; 16];
                _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, _mm_aesenc_si128(s, k));
                out
            }
        }

        #[inline(always)]
        fn round_pair(lo: u64, hi: u64) -> (u64, u64) {
            // SAFETY: as above.
            unsafe {
                let s = _mm_set_epi64x(hi as i64, lo as i64);
                let o = _mm_aesenc_si128(s, _mm_setzero_si128());
                (
                    _mm_cvtsi128_si64(o) as u64,
                    _mm_cvtsi128_si64(_mm_unpackhi_epi64(o, o)) as u64,
                )
            }
        }
    }

    /// # Safety
    /// The CPU must support AES-NI and SSE2.
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn walk(m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> [u8; 32] {
        super::walk_generic::<AesNi>(m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support AES-NI and SSE2.
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn start(m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> ([u64; 8], [u8; 8]) {
        super::start_generic::<AesNi>(m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support AES-NI and SSE2.
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn time_steps(op: Option<u8>, n: usize, code: super::StepCode) -> f64 {
        super::time_steps_generic::<AesNi>(op, n, code)
    }

    /// # Safety
    /// As [`time_steps`].
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn walk_written(code: super::StepCode, m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> [u8; 32] {
        super::walk_written::<AesNi>(code, m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support AES-NI and SSE2.
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn walk_many<const K: usize>(
        code: super::StepCode,
        m: &[u8; 32],
        nonces: &[u64; K],
        pads: &mut [Box<super::Pad>; K],
    ) -> [[u8; 32]; K] {
        super::walk_many_written::<AesNi, K>(code, m, nonces, pads)
    }

    /// One AESENC round, for cross-checks.
    ///
    /// # Safety
    /// The CPU must support AES-NI and SSE2.
    #[cfg(test)]
    #[target_feature(enable = "aes,sse2")]
    pub(super) unsafe fn round(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
        <AesNi as super::AesRound>::round(state, key)
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::aarch64::*;

    pub(super) struct Armv8;

    impl super::AesRound for Armv8 {
        #[inline(always)]
        fn round(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
            // SAFETY: only reached through `walk`, which requires the AES extension.
            unsafe {
                let s = vld1q_u8(state.as_ptr());
                let k = vld1q_u8(key.as_ptr());
                let o = veorq_u8(vaesmcq_u8(vaeseq_u8(s, vdupq_n_u8(0))), k);
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), o);
                out
            }
        }

        #[inline(always)]
        fn round_pair(lo: u64, hi: u64) -> (u64, u64) {
            // SAFETY: as above.
            unsafe {
                let s = vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)));
                let o = vreinterpretq_u64_u8(vaesmcq_u8(vaeseq_u8(s, vdupq_n_u8(0))));
                (vgetq_lane_u64::<0>(o), vgetq_lane_u64::<1>(o))
            }
        }
    }

    /// # Safety
    /// The CPU must support the ARMv8 AES extension.
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn walk(m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> [u8; 32] {
        super::walk_generic::<Armv8>(m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support the ARMv8 AES extension.
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn start(m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> ([u64; 8], [u8; 8]) {
        super::start_generic::<Armv8>(m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support the ARMv8 AES extension.
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn time_steps(op: Option<u8>, n: usize, code: super::StepCode) -> f64 {
        super::time_steps_generic::<Armv8>(op, n, code)
    }

    /// # Safety
    /// As [`time_steps`].
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn walk_written(code: super::StepCode, m: &[u8; 32], nonce: u64, pad: &mut super::Pad) -> [u8; 32] {
        super::walk_written::<Armv8>(code, m, nonce, pad)
    }

    /// # Safety
    /// The CPU must support the ARMv8 AES extension.
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn walk_many<const K: usize>(
        code: super::StepCode,
        m: &[u8; 32],
        nonces: &[u64; K],
        pads: &mut [Box<super::Pad>; K],
    ) -> [[u8; 32]; K] {
        super::walk_many_written::<Armv8, K>(code, m, nonces, pads)
    }

    /// One AESENC round, for cross-checks.
    ///
    /// # Safety
    /// The CPU must support the ARMv8 AES extension.
    #[cfg(test)]
    #[target_feature(enable = "aes,neon")]
    pub(super) unsafe fn round(state: [u8; 16], key: [u8; 16]) -> [u8; 16] {
        <Armv8 as super::AesRound>::round(state, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex16(s: &str) -> [u8; 16] {
        let mut out = [0u8; 16];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    /// FIPS-197 Appendix B, round 1: the state at the start of round 1 under
    /// round key 1 gives the state at the start of round 2.
    const FIPS_IN: &str = "193de3bea0f4e22b9ac68d2ae9f84808";
    const FIPS_KEY: &str = "a0fafe1788542cb123a339392a6c7605";
    const FIPS_OUT: &str = "a49c7ff2689f352b6b5bea43026a5049";

    #[test]
    fn portable_round_matches_fips_197() {
        assert_eq!(SoftAes::round(hex16(FIPS_IN), hex16(FIPS_KEY)), hex16(FIPS_OUT));
    }

    #[test]
    fn hardware_round_matches_fips_197_and_portable() {
        let mut checked = false;
        #[cfg(target_arch = "x86_64")]
        if detect() == Backend::AesNi {
            let hw = |s, k| unsafe { x86::round(s, k) };
            assert_eq!(hw(hex16(FIPS_IN), hex16(FIPS_KEY)), hex16(FIPS_OUT));
            let mut s = [0u8; 16];
            let mut k = [0u8; 16];
            for n in 0..2000u32 {
                let h = blake3::hash(&n.to_le_bytes());
                s.copy_from_slice(&h.as_bytes()[..16]);
                k.copy_from_slice(&h.as_bytes()[16..]);
                assert_eq!(hw(s, k), SoftAes::round(s, k));
                let (lo, hi) = (le64(&s[..8]), le64(&s[8..]));
                assert_eq!(x86::AesNi::round_pair(lo, hi), SoftAes::round_pair(lo, hi));
            }
            checked = true;
        }
        #[cfg(target_arch = "aarch64")]
        if detect() == Backend::Armv8 {
            let hw = |s, k| unsafe { arm::round(s, k) };
            assert_eq!(hw(hex16(FIPS_IN), hex16(FIPS_KEY)), hex16(FIPS_OUT));
            let mut s = [0u8; 16];
            let mut k = [0u8; 16];
            for n in 0..2000u32 {
                let h = blake3::hash(&n.to_le_bytes());
                s.copy_from_slice(&h.as_bytes()[..16]);
                k.copy_from_slice(&h.as_bytes()[16..]);
                assert_eq!(hw(s, k), SoftAes::round(s, k));
                let (lo, hi) = (le64(&s[..8]), le64(&s[8..]));
                assert_eq!(arm::Armv8::round_pair(lo, hi), SoftAes::round_pair(lo, hi));
            }
            checked = true;
        }
        if !checked {
            eprintln!("no AES hardware on this CPU: only the portable path was checked");
        }
    }

    #[test]
    fn every_path_gives_the_same_hash() {
        for n in 0..4u64 {
            let m = *blake3::hash(&n.to_le_bytes()).as_bytes();
            let nonce = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let portable = walk_hash_portable(&m, nonce);
            assert_eq!(walk_hash(&m, nonce), portable);
            assert_eq!(walk_with_pad(&m, nonce, &mut new_pad()), portable);
        }
    }

    #[test]
    fn deterministic_and_input_sensitive() {
        let m = [7u8; 32];
        let a = walk_hash(&m, 1);
        assert_eq!(a, walk_hash(&m, 1));
        assert_ne!(a, walk_hash(&m, 2));
        let mut m2 = m;
        m2[31] ^= 1;
        assert_ne!(a, walk_hash(&m2, 1));
    }

    fn hex32(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap();
        }
        out
    }

    #[test]
    fn golden_vectors() {
        for (m, nonce, want) in GOLDEN {
            let m = hex32(m);
            assert_eq!(walk_hash(&m, nonce), hex32(want), "fast path, nonce {nonce}");
            assert_eq!(walk_hash_portable(&m, nonce), hex32(want), "portable, nonce {nonce}");
        }
        assert!(check_golden());
    }

    /// Seeding, the steps in pieces, and the final hash are the walk.
    #[test]
    fn the_walk_in_three_parts_is_the_walk() {
        for (m, nonce, want) in GOLDEN {
            let m = hex32(m);
            let mut pad = new_pad();
            let mut st = walk_start(&m, nonce, &mut pad);
            // Pieces that cross re-mixes and end on them.
            for n in [8, WALK_REMIX - 8, WALK_REMIX, WALK_STEPS - 2 * WALK_REMIX] {
                walk_steps(&mut st, &mut pad, n);
            }
            assert_eq!(st.steps, WALK_STEPS);
            assert_eq!(walk_finish(&m, nonce, &st.r), hex32(want), "nonce {nonce}");
            // The start with hardware AES is the portable one's.
            let mut other = new_pad();
            let (r, prog) = start_generic::<SoftAes>(&m, nonce, &mut other);
            let mut again = new_pad();
            let fast = walk_start(&m, nonce, &mut again);
            assert_eq!((fast.r, fast.prog), (r, prog));
            assert!(other[..] == again[..], "the same pad");
        }
    }

    /// Steps one at a time are the walk, and the inputs worked out for a
    /// step are the ones it uses.
    #[test]
    fn single_steps_and_their_inputs() {
        let m = hex32(GOLDEN[1].0);
        let mut pad = new_pad();
        let mut st = walk_start(&m, GOLDEN[1].1, &mut pad);
        let (mut whole, mut whole_pad) = (st, pad.clone());
        walk_steps(&mut whole, &mut whole_pad, WALK_REMIX + 24);
        for _ in 0..WALK_REMIX + 24 {
            let i = step_inputs(&st, &pad);
            let (y, aes_hi) = apply_op(i.op, i.v, i.k);
            let crossing = (st.steps + 1) % WALK_REMIX == 0;
            walk_steps(&mut st, &mut pad, 1);
            if !crossing {
                assert_eq!(st.c, y ^ i.m[1] ^ i.m[3], "step {}", st.steps);
                if i.op == 3 && i.t != i.s {
                    assert_eq!(st.r[i.t], i.k ^ aes_hi);
                }
            }
            assert_eq!(pad[i.words[0] / 8][i.words[0] % 8], i.m[0] ^ y);
            assert_eq!(pad[i.words[3] / 8][i.words[3] % 8], i.m[3] ^ i.x);
        }
        assert_eq!((st, &pad[..]), (whole, &whole_pad[..]));
        let r = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let (r2, p2) = remix_of(&r);
        assert!(r2 != r && p2.iter().map(|o| 1u8 << o).fold(0, |a, b| a | b) == 0xff);
    }

    #[test]
    fn step_timings_are_positive_for_every_operation() {
        for op in [None, Some(0), Some(2), Some(3), Some(7)] {
            for code in [StepCode::Indexed, StepCode::Unindexed, StepCode::Ahead, StepCode::NoStores] {
                assert!(time_steps(op, 64, code) > 0.0, "{op:?} {code:?}");
            }
        }
        assert!(time_start(1) > 0.0 && time_remix(4) > 0.0);
    }

    /// Every way of writing a step agrees with the reference, round by
    /// round: from random registers, under random programs (every operation
    /// at every register, every register offset), over one pad each that the
    /// steps keep writing. The miner's step_unindexed, and step_ahead.
    #[test]
    fn every_way_of_writing_a_step_agrees() {
        let mut z = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            z
        };
        let mut reference = new_pad();
        for line in reference.iter_mut() {
            for w in line.iter_mut() {
                *w = next();
            }
        }
        let (mut unindexed, mut ahead) = (reference.clone(), reference.clone());
        let mut r: [u64; 8] = std::array::from_fn(|_| next());
        let (mut r_u, mut r_a) = (r, r);
        let mut c = next();
        let mut c_u = c;
        let mut x_a = r[0] ^ c;
        for round in 0..6000 {
            let prog = program_of(next());
            for s in 0..8 {
                step::<SoftAes>(s, prog[s], &mut r, &mut c, &mut reference);
            }
            round_unindexed::<SoftAes, true>(&prog, &mut r_u, &mut c_u, &mut unindexed);
            let (nx, out) = round_ahead::<SoftAes>(&prog, x_a, &mut r_a, &mut ahead);
            x_a = nx;
            assert_eq!((r_u, c_u), (r, c), "unindexed, round {round}, program {prog:?}");
            assert_eq!((r_a, out, x_a), (r, c, r[0] ^ c), "ahead, round {round}, program {prog:?}");
        }
        assert!(unindexed[..] == reference[..] && ahead[..] == reference[..], "the same pads");
    }

    /// The walks written those ways give the known answers, with the fastest
    /// AES here and the portable one.
    #[test]
    fn the_walk_written_every_way_gives_the_known_answers() {
        for (m, nonce, want) in GOLDEN {
            let m = hex32(m);
            assert_eq!(walk_hash(&m, nonce), hex32(want), "nonce {nonce}, the miner's");
            for code in [StepCode::Indexed, StepCode::Unindexed, StepCode::Ahead] {
                assert_eq!(walk_hash_written(code, &m, nonce, &mut new_pad()), hex32(want), "nonce {nonce}, {code:?}");
                assert_eq!(walk_written::<SoftAes>(code, &m, nonce, &mut new_pad()), hex32(want), "nonce {nonce}, {code:?}, portable");
            }
            let mut pad = new_pad();
            let mut st = walk_start(&m, nonce, &mut pad);
            walk_steps(&mut st, &mut pad, WALK_STEPS);
            assert_eq!(walk_finish(&m, nonce, &st.r), hex32(want), "nonce {nonce}, the reference step");
        }
    }

    /// Walks stepped in turn on one thread are each the walk of its nonce.
    #[test]
    fn walks_stepped_in_turn_are_each_their_own() {
        let m = hex32(GOLDEN[2].0);
        let mut pads2: [Box<Pad>; 2] = std::array::from_fn(|_| new_pad());
        let two = walk_hash_many::<2>(&m, &[42, 43], &mut pads2);
        assert_eq!(two, [walk_hash_portable(&m, 42), walk_hash_portable(&m, 43)]);
        assert_eq!(two[0], hex32(GOLDEN[2].2));
        let mut pads3: [Box<Pad>; 3] = std::array::from_fn(|_| new_pad());
        let three = walk_hash_many::<3>(&m, &[1, 42, 2], &mut pads3);
        assert_eq!(three[1], hex32(GOLDEN[2].2));
        assert_eq!(three[2], walk_hash(&m, 2));
        assert!(golden_many::<2>(MINER_STEP, &mut pads2) && golden_many::<1>(MINER_STEP, &mut [new_pad()]));
        let (rate, good) = bench_many::<2>(2, std::time::Duration::from_millis(300), true);
        assert!(rate > 0.0 && good);
    }

    /// Every way of writing the step, interleaved 1 to 4 walks a thread,
    /// gives the known answers in every slot, and the miner's batches are
    /// each nonce's walk however many it takes at once.
    #[test]
    fn interleaved_walks_written_every_way_are_each_their_own() {
        for code in [StepCode::Indexed, StepCode::Unindexed, StepCode::Ahead] {
            assert!(golden_many::<1>(code, &mut std::array::from_fn(|_| new_pad())), "{code:?} 1");
            assert!(golden_many::<2>(code, &mut std::array::from_fn(|_| new_pad())), "{code:?} 2");
            assert!(golden_many::<3>(code, &mut std::array::from_fn(|_| new_pad())), "{code:?} 3");
            assert!(golden_many::<4>(code, &mut std::array::from_fn(|_| new_pad())), "{code:?} 4");
        }
        let m = hex32(GOLDEN[2].0);
        let nonces: Vec<u64> = vec![42, 7, 1 << 62, 9, 10, 42];
        for n in 1..=nonces.len() {
            let got = walk_hash_batch(&m, &nonces[..n]);
            assert_eq!(got.len(), n);
            for (i, (nonce, h)) in got.iter().enumerate() {
                assert_eq!(*nonce, nonces[i]);
                assert_eq!(*h, walk_hash_portable(&m, *nonce), "batch of {n}, slot {i}");
            }
        }
        assert_eq!(walk_hash_batch(&m, &[]), vec![]);
    }

    /// The choice of walks per thread measures every count, and an override
    /// out of range is refused.
    #[test]
    fn walks_per_thread_are_measured() {
        let rates = measure_walks_per_thread(2, std::time::Duration::from_millis(60));
        assert!(rates.iter().all(|r| *r > 0.0), "{rates:?}");
        let k = walks_per_thread(2);
        assert!((1..=MAX_WALKS_PER_THREAD).contains(&k));
    }

    #[test]
    fn a_bench_counts_attempts_and_checks_every_thread() {
        let (rate, good) = bench(2, std::time::Duration::from_millis(300));
        assert!(rate > 0.0);
        assert!(good, "this CPU computes the known answers");
    }

    #[test]
    fn program_is_always_a_permutation() {
        for z in [0u64, 1, 7, 40_319, 40_320, u64::MAX, 0x0123_4567_89ab_cdef] {
            let mut p = program(z);
            p.sort_unstable();
            assert_eq!(p, [0, 1, 2, 3, 4, 5, 6, 7]);
        }
        assert_eq!(program(0), [1, 2, 3, 4, 5, 6, 7, 0]);
    }

    /// A dirty pad from a previous attempt must not leak into the next one.
    #[test]
    fn reused_pad_gives_the_same_hash() {
        let mut pad = new_pad();
        let m = [3u8; 32];
        let first = walk_with_pad(&m, 9, &mut pad);
        walk_with_pad(&[4u8; 32], 10, &mut pad);
        assert_eq!(walk_with_pad(&m, 9, &mut pad), first);
    }
}
