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

#[inline(always)]
fn walk_generic<A: AesRound>(mining_hash: &[u8; 32], nonce: u64, pad: &mut Pad) -> [u8; 32] {
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
    let mut prog = program(le64(&seed[192..]));

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

    let mut c = 0u64;
    for round in 0..WALK_STEPS / 8 {
        // Eight steps, one per register, written out so that each step's
        // operation is a branch whose target is fixed for the whole program:
        // a CPU predicts all eight, while a GPU warp's threads (each with its
        // own program) diverge.
        step::<A>(0, prog[0], &mut r, &mut c, pad);
        step::<A>(1, prog[1], &mut r, &mut c, pad);
        step::<A>(2, prog[2], &mut r, &mut c, pad);
        step::<A>(3, prog[3], &mut r, &mut c, pad);
        step::<A>(4, prog[4], &mut r, &mut c, pad);
        step::<A>(5, prog[5], &mut r, &mut c, pad);
        step::<A>(6, prog[6], &mut r, &mut c, pad);
        step::<A>(7, prog[7], &mut r, &mut c, pad);
        if (round + 1) % (WALK_REMIX / 8) == 0 {
            remix(&mut r, &mut prog);
        }
    }

    *blake3::Hasher::new_derive_key(CTX_FINAL)
        .update(mining_hash)
        .update(&nonce.to_le_bytes())
        .update(&regs_bytes(&r))
        .finalize()
        .as_bytes()
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

/// The AES S-box (FIPS-197 Figure 7).
const SBOX: [u8; 256] = [
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

    /// Pinned outputs. Cross-checked against two independent implementations
    /// written from the definition above (midstate's `docs/walk-reference`):
    /// a Python one (S-box derived from GF(2^8) inversion) and a C one run on
    /// an emulated Cortex-A76 through the ARMv8 AES instructions. Changing
    /// any of these is a hard fork.
    #[cfg(not(feature = "fast-mining"))]
    const GOLDEN: [(&str, u64, &str); 3] = [
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
    const GOLDEN: [(&str, u64, &str); 3] = [
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
