//! Multi-width parallel BLAKE3 mining with automatic CPU feature detection.
//!
//! At startup, [`detect()`] queries the CPU and returns the widest available
//! SIMD path. The mining loop calls [`mine_batch()`] which dispatches to:
//!
//! | Platform                     | Register width | Lanes | Nonces/batch |
//! |------------------------------|---------------|-------|--------------|
//! | x86_64 + AVX2                | 256-bit       | 8     | 8            |
//! | aarch64 (NEON, dual-issue)   | 2 × 128-bit   | 8     | 8            |
//! | aarch64 (NEON, single-issue) | 128-bit       | 4     | 4            |
//! | wasm32 + simd128             | 128-bit       | 4     | 4            |
//! | Scalar                       | 32-bit        | 1     | 4 (serial)   |
//!
//! **Consensus safety:** Only the nonce *search* uses SIMD. Verification
//! remains scalar via `create_extension` / `blake3` crate. The 8-way NEON
//! path is bit-identical to two consecutive 4-way calls — only the
//! instruction schedule differs.
//!
//! # Performance: fully-unrolled iterated compression (aarch64)
//!
//! The iterated hashing hot loop runs `EXTENSION_ITERATIONS` (= 1,000,000)
//! compressions of a 32-byte block whose layout is fixed:
//!
//! ```text
//!   msg[0..8]  = previous_hash  (the only varying part)
//!   msg[8..16] = 0              (constant zero)
//!   block_len  = 32             (constant)
//!   flags      = HASH_FLAGS     (constant)
//! ```
//!
//! On aarch64 we exploit this with **full manual unrolling of all 7 rounds**
//! using the precomputed `MSG_SCHEDULE`. Each G call has its message indices
//! hardcoded at the source level (not looked up at runtime), so the compiler
//! sees concretely which of `m0..m7` (or constant zero) each G call needs.
//!
//! Earlier attempts used closures that returned `m[i]` or `zero` based on a
//! runtime-supplied schedule index. Despite all inputs being statically known
//! the compiler did not constant-fold those lookups, retaining branches or
//! lookup tables in the hot loop. The fully-unrolled macros below force the
//! compiler to see the message source as a compile-time fact at every G call.
//!
//! # Formal specification (Z notation)
//!
//! Let `M : seq BYTE` with `#M = 32` denote the midstate, `N : NONCE` a nonce
//! (where `NONCE == 0 .. 2^64 - 1`), and `H : seq BYTE` with `#H = 32` a hash
//! result. Define the canonical scalar miner as:
//!
//! ```text
//! Scalar : (seq BYTE × NONCE) → seq BYTE
//! Scalar(M, N) = blake3^k(blake3(M ‖ N_le8))
//! ```
//!
//! where `k = EXTENSION_ITERATIONS`. Every SIMD path `Φ_w` (for lane width
//! `w ∈ {4, 8}`) satisfies the **consensus invariant**:
//!
//! ```text
//! ∀ M ∈ BYTE^32 . ∀ N ∈ NONCE^w . ∀ i ∈ 0..w-1 .
//!     Φ_w(M, N)(i) = (N(i), Scalar(M, N(i)))
//! ```

use super::types::EXTENSION_ITERATIONS;

// ═══════════════════════════════════════════════════════════════════════════
//  BLAKE3 Constants (shared by all backends)
// ═══════════════════════════════════════════════════════════════════════════

/// BLAKE3 initialisation vector (same as SHA-256 IV, truncated).
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Combined chunk flags: `CHUNK_START | CHUNK_END | ROOT`.
const HASH_FLAGS: u32 = 1 | 2 | 8;

/// BLAKE3 message word permutation schedule, one row per round (7 rounds total).
///
/// This is duplicated as compile-time literal indices in the unrolled NEON
/// path below — that path needs each index baked into source code, not
/// looked up from this array at runtime.
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Converts a 32-byte array into 8 little-endian `u32` words.
///
/// # Formal specification
///
/// ```text
/// bytes_to_words : BYTE^32 → WORD^8
/// ∀ b ∈ BYTE^32 . ∀ i ∈ 0..7 .
///     bytes_to_words(b)(i) =
///         b(4i) + 2^8 · b(4i+1) + 2^16 · b(4i+2) + 2^24 · b(4i+3)
/// ```
#[inline(always)]
fn bytes_to_words(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
    }
    w
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════

/// The SIMD capability level detected on this CPU/target.
///
/// # Formal specification
///
/// ```text
/// SimdLevel ::= Scalar | Wasm128_4 | Neon4 | Neon8 | Avx2_8
///
/// lanes : SimdLevel → ℕ
/// lanes(Scalar)    = 4
/// lanes(Wasm128_4) = 4
/// lanes(Neon4)     = 4
/// lanes(Neon8)     = 8
/// lanes(Avx2_8)    = 8
///
/// ∀ ℓ ∈ SimdLevel . lanes(ℓ) ∈ {4, 8}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No usable SIMD — batch of 4, processed serially.
    Scalar,
    /// WebAssembly 128-bit SIMD — 4 lanes.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm128_4,
    /// ARM NEON: 128-bit registers, 4 lanes × 32-bit, fully unrolled.
    ///
    /// Used on in-order / single-issue NEON cores (A53, A55) when explicitly
    /// selected via `MINER_NEON_FORCE=4`.
    #[cfg(target_arch = "aarch64")]
    Neon4,
    /// ARM NEON dual-issue: two interleaved 4-way streams, 8 lanes total,
    /// fully unrolled. Default on aarch64.
    #[cfg(target_arch = "aarch64")]
    Neon8,
    /// x86 AVX2: 256-bit registers, 8 lanes × 32-bit.
    #[cfg(target_arch = "x86_64")]
    Avx2_8,
}

impl SimdLevel {
    /// How many nonces are processed per batch call.
    pub fn lanes(self) -> usize {
        match self {
            SimdLevel::Scalar => 4,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            SimdLevel::Wasm128_4 => 4,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => 4,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon8 => 8,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2_8 => 8,
        }
    }

    /// Human-readable name for logging and diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            SimdLevel::Wasm128_4 => "WASM SIMD128 4-way",
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => "NEON 4-way (unrolled)",
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon8 => "NEON 8-way (unrolled, dual-issue)",
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2_8 => "AVX2 8-way",
        }
    }
}

impl std::fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Detects the best available SIMD level for the current CPU/target.
///
/// # Formal specification
///
/// ```text
/// detect : ⊥ → SimdLevel
///
///   CPU = x86_64_avx2                                  ⇒ result = Avx2_8
///   CPU = x86_64_noavx2                                ⇒ result = Scalar
///   CPU = aarch64 ∧ env("MINER_NEON_FORCE") = "4"      ⇒ result = Neon4
///   CPU = aarch64 ∧ env("MINER_NEON_FORCE") ≠ "4"      ⇒ result = Neon8
///   CPU = wasm32_simd128                               ⇒ result = Wasm128_4
///   CPU = other                                        ⇒ result = Scalar
/// ```
pub fn detect() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2_8;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Ok(forced) = std::env::var("MINER_NEON_FORCE") {
            if forced.trim() == "4" {
                return SimdLevel::Neon4;
            }
        }
        return SimdLevel::Neon8;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return SimdLevel::Wasm128_4;
    }
    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

/// Returns the cached SIMD level, detecting it on the first call.
pub fn detected_level() -> SimdLevel {
    static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(detect)
}

/// Mine a batch of nonces using the best available SIMD.
///
/// # Formal specification
///
/// Let `w = lanes(detected_level())`. Then:
///
/// ```text
/// mine_batch : BYTE^32 × seq NONCE → seq (NONCE × BYTE^32)
///
/// pre:   #nonces ≥ w
/// post:  #result = w  ∧
///        ∀ i ∈ 0..w-1 .
///            result(i).0 = nonces(i)  ∧
///            result(i).1 = Scalar(midstate, nonces(i))
/// ```
///
/// # Panics
///
/// Panics if `nonces.len()` is less than `detected_level().lanes()`.
pub fn mine_batch(midstate: [u8; 32], nonces: &[u64]) -> Vec<(u64, [u8; 32])> {
    match detected_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2_8 => {
            assert!(nonces.len() >= 8);
            let n: [u64; 8] = [
                nonces[0], nonces[1], nonces[2], nonces[3],
                nonces[4], nonces[5], nonces[6], nonces[7],
            ];
            unsafe { avx2::create_extensions_8way_avx2(midstate, n) }.to_vec()
        }
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon8 => {
            assert!(nonces.len() >= 8);
            let n: [u64; 8] = [
                nonces[0], nonces[1], nonces[2], nonces[3],
                nonces[4], nonces[5], nonces[6], nonces[7],
            ];
            unsafe { neon::create_extensions_8way_neon(midstate, n) }.to_vec()
        }
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon4 => {
            assert!(nonces.len() >= 4);
            let n: [u64; 4] = [nonces[0], nonces[1], nonces[2], nonces[3]];
            unsafe { neon::create_extensions_4way_neon(midstate, n) }.to_vec()
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdLevel::Wasm128_4 => {
            assert!(nonces.len() >= 4);
            let n: [u64; 4] = [nonces[0], nonces[1], nonces[2], nonces[3]];
            unsafe { wasm_simd::create_extensions_4way_wasm(midstate, n) }.to_vec()
        }
        SimdLevel::Scalar => {
            nonces.iter().take(4).map(|&nonce| {
                let ext = super::extension::create_extension(midstate, nonce);
                (ext.nonce, ext.final_hash)
            }).collect()
        }
    }
}

/// Convenience: 4-way entry point (backward compat + tests).
pub fn create_extensions_4way(
    midstate: [u8; 32],
    nonces: [u64; 4],
) -> [(u64, [u8; 32]); 4] {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::create_extensions_4way_neon(midstate, nonces) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm_simd::create_extensions_4way_wasm(midstate, nonces) }
    }
    #[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
    {
        let mut results = [(0u64, [0u8; 32]); 4];
        for i in 0..4 {
            let ext = super::extension::create_extension(midstate, nonces[i]);
            results[i] = (ext.nonce, ext.final_hash);
        }
        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  WASM 128-bit SIMD (wasm32 + simd128) — unchanged
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use super::*;
    use core::arch::wasm32::*;

    #[inline(always)] unsafe fn vrot16(x: v128) -> v128 { v128_or(u32x4_shr(x, 16), u32x4_shl(x, 16)) }
    #[inline(always)] unsafe fn vrot12(x: v128) -> v128 { v128_or(u32x4_shr(x, 12), u32x4_shl(x, 20)) }
    #[inline(always)] unsafe fn vrot8(x: v128)  -> v128 { v128_or(u32x4_shr(x, 8),  u32x4_shl(x, 24)) }
    #[inline(always)] unsafe fn vrot7(x: v128)  -> v128 { v128_or(u32x4_shr(x, 7),  u32x4_shl(x, 25)) }

    #[inline(always)]
    unsafe fn g(v: &mut [v128; 16], a: usize, b: usize, c: usize, d: usize, mx: v128, my: v128) {
        v[a] = u32x4_add(u32x4_add(v[a], v[b]), mx);
        v[d] = vrot16(v128_xor(v[d], v[a]));
        v[c] = u32x4_add(v[c], v[d]);
        v[b] = vrot12(v128_xor(v[b], v[c]));
        v[a] = u32x4_add(u32x4_add(v[a], v[b]), my);
        v[d] = vrot8(v128_xor(v[d], v[a]));
        v[c] = u32x4_add(v[c], v[d]);
        v[b] = vrot7(v128_xor(v[b], v[c]));
    }

    #[inline(always)]
    unsafe fn round(v: &mut [v128; 16], m: &[v128; 16], s: &[usize; 16]) {
        g(v, 0, 4,  8, 12, m[s[0]],  m[s[1]]);
        g(v, 1, 5,  9, 13, m[s[2]],  m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    #[inline(always)]
    unsafe fn compress_4way(cv: &[v128; 8], msg: &[v128; 16], block_len: u32) -> [v128; 8] {
        let zero = u32x4_splat(0);
        let mut v: [v128; 16] = [zero; 16];
        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];
        v[8]  = u32x4_splat(IV[0]); v[9]  = u32x4_splat(IV[1]);
        v[10] = u32x4_splat(IV[2]); v[11] = u32x4_splat(IV[3]);
        v[12] = zero; v[13] = zero;
        v[14] = u32x4_splat(block_len);
        v[15] = u32x4_splat(HASH_FLAGS);
        for r in 0..7 { round(&mut v, msg, &MSG_SCHEDULE[r]); }
        [
            v128_xor(v[0], v[8]),  v128_xor(v[1], v[9]),
            v128_xor(v[2], v[10]), v128_xor(v[3], v[11]),
            v128_xor(v[4], v[12]), v128_xor(v[5], v[13]),
            v128_xor(v[6], v[14]), v128_xor(v[7], v[15]),
        ]
    }

    unsafe fn extract_hash(out: &[v128; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        for i in 0..8 {
            let words: [u32; 4] = core::mem::transmute(out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&words[lane].to_le_bytes());
        }
        result
    }

    pub unsafe fn create_extensions_4way_wasm(midstate: [u8; 32], nonces: [u64; 4]) -> [(u64, [u8; 32]); 4] {
        let zero = u32x4_splat(0);
        let cv: [v128; 8] = [
            u32x4_splat(IV[0]), u32x4_splat(IV[1]),
            u32x4_splat(IV[2]), u32x4_splat(IV[3]),
            u32x4_splat(IV[4]), u32x4_splat(IV[5]),
            u32x4_splat(IV[6]), u32x4_splat(IV[7]),
        ];
        let ms_words = bytes_to_words(&midstate);
        let nonce_lo: [u32; 4] = [
            nonces[0] as u32, nonces[1] as u32, nonces[2] as u32, nonces[3] as u32,
        ];
        let nonce_hi: [u32; 4] = [
            (nonces[0] >> 32) as u32, (nonces[1] >> 32) as u32,
            (nonces[2] >> 32) as u32, (nonces[3] >> 32) as u32,
        ];
        let mut msg: [v128; 16] = [zero; 16];
        for i in 0..8 { msg[i] = u32x4_splat(ms_words[i]); }
        msg[8] = core::mem::transmute(nonce_lo);
        msg[9] = core::mem::transmute(nonce_hi);
        let mut hw = compress_4way(&cv, &msg, 40);
        for _ in 0..EXTENSION_ITERATIONS {
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];
            msg[8]  = zero; msg[9]  = zero; msg[10] = zero; msg[11] = zero;
            msg[12] = zero; msg[13] = zero; msg[14] = zero; msg[15] = zero;
            hw = compress_4way(&cv, &msg, 32);
        }
        [
            (nonces[0], extract_hash(&hw, 0)),
            (nonces[1], extract_hash(&hw, 1)),
            (nonces[2], extract_hash(&hw, 2)),
            (nonces[3], extract_hash(&hw, 3)),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NEON 4-way and 8-way (aarch64) — fully unrolled iterated hashing
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    // ─── Rotates ─────────────────────────────────────────────────────────

    /// Rotates each 32-bit lane right by 16 bits using NEON byte-reverse within 32-bit elements.
    #[inline(always)]
    unsafe fn vrot16(x: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
    }
    /// Rotates each 32-bit lane right by 12 bits.
    #[inline(always)]
    unsafe fn vrot12(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<12>(x), vshlq_n_u32::<20>(x))
    }
    /// Rotates each 32-bit lane right by 8 bits.
    #[inline(always)]
    unsafe fn vrot8(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<8>(x), vshlq_n_u32::<24>(x))
    }
    /// Rotates each 32-bit lane right by 7 bits.
    #[inline(always)]
    unsafe fn vrot7(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<7>(x), vshlq_n_u32::<25>(x))
    }

    // ─── Generic 4-way primitives (used for the initial 40-byte block) ───

    /// BLAKE3 `G` mixing function over 4 interleaved NEON lanes simultaneously.
    #[inline(always)]
    unsafe fn g(
        v: &mut [uint32x4_t; 16], a: usize, b: usize, c: usize, d: usize,
        mx: uint32x4_t, my: uint32x4_t,
    ) {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), mx);
        v[d] = vrot16(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = vrot12(veorq_u32(v[b], v[c]));
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), my);
        v[d] = vrot8(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = vrot7(veorq_u32(v[b], v[c]));
    }

    /// Applies one full BLAKE3 round (4 column + 4 diagonal `G` calls).
    #[inline(always)]
    unsafe fn round(v: &mut [uint32x4_t; 16], m: &[uint32x4_t; 16], s: &[usize; 16]) {
        g(v, 0, 4,  8, 12, m[s[0]],  m[s[1]]);
        g(v, 1, 5,  9, 13, m[s[2]],  m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    /// Performs one BLAKE3 compression over 4 independent chains in parallel using NEON.
    ///
    /// Used only for the initial 40-byte block. The iterated phase uses the
    /// fully-unrolled `compress_iter_4way` instead.
    #[inline(always)]
    unsafe fn compress_4way(
        cv: &[uint32x4_t; 8], msg: &[uint32x4_t; 16], block_len: u32,
    ) -> [uint32x4_t; 8] {
        let zero = vdupq_n_u32(0);
        let mut v: [uint32x4_t; 16] = [zero; 16];
        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];
        v[8]  = vdupq_n_u32(IV[0]); v[9]  = vdupq_n_u32(IV[1]);
        v[10] = vdupq_n_u32(IV[2]); v[11] = vdupq_n_u32(IV[3]);
        v[12] = zero; v[13] = zero;
        v[14] = vdupq_n_u32(block_len); v[15] = vdupq_n_u32(HASH_FLAGS);
        for r in 0..7 { round(&mut v, msg, &MSG_SCHEDULE[r]); }
        [
            veorq_u32(v[0], v[8]),  veorq_u32(v[1], v[9]),
            veorq_u32(v[2], v[10]), veorq_u32(v[3], v[11]),
            veorq_u32(v[4], v[12]), veorq_u32(v[5], v[13]),
            veorq_u32(v[6], v[14]), veorq_u32(v[7], v[15]),
        ]
    }

    /// Extracts the 32-byte hash for a single `lane` (0–3) from the transposed output.
    unsafe fn extract_hash(out: &[uint32x4_t; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0u32; 4];
        for i in 0..8 {
            vst1q_u32(buf.as_mut_ptr(), out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&buf[lane].to_le_bytes());
        }
        result
    }

    // ═══════════════════════════════════════════════════════════════════
    //  FULLY UNROLLED 4-WAY ITERATED COMPRESSION
    // ═══════════════════════════════════════════════════════════════════
    //
    // # Why this is structured this way
    //
    // Earlier attempts used:
    //   - a `round_fused` function that took the 8 message words as
    //     parameters and looked up `m[s[i]]` at runtime via a closure
    //   - a `pick(i)` match that returned m0..m7 or zero
    //
    // Despite all inputs being statically known, LLVM did NOT constant-fold
    // those lookups reliably — the generated code retained branches or
    // lookup tables in the hot loop. Measured result: 0.92× (slower).
    //
    // The fix: write each of the 7 rounds as a separate macro invocation
    // with the MSG_SCHEDULE indices BAKED INTO THE SOURCE. Every G call
    // explicitly knows whether each of its two message inputs is one of
    // m0..m7 (the variable chaining-value words) or is the constant ZERO.
    //
    // This gives the compiler concrete static facts:
    //   - "this G call's mx is m3, my is m7" — bind to registers directly
    //   - "this G call's mx is zero, my is m4" — the `add` with zero may
    //     simplify (or stay as cheap add-with-zero-register, which on A76
    //     is the same cycle as a register-register add minus the dependency)

    /// `mw!(N, m0..m7, zero)` expands at macro-expansion time to either
    /// `m{N}` (when N ∈ 0..7) or `zero` (when N ∈ 8..15).
    ///
    /// # Formal specification
    ///
    /// ```text
    /// mw : INDEX × WORD_4^8 × WORD_4 → WORD_4
    /// mw(i, ⟨m₀,..,m₇⟩, z) = if i < 8 then m_i else z
    /// ```
    ///
    /// The selection is performed by the macro expander, not by generated
    /// code — there is no runtime branch.
    macro_rules! mw {
        // m0..m7: use the concrete chaining word
        (0, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m0 };
        (1, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m1 };
        (2, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m2 };
        (3, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m3 };
        (4, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m4 };
        (5, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m5 };
        (6, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m6 };
        (7, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $m7 };
        // m8..m15: always zero in the iterated phase
        (8,  $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (9,  $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (10, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (11, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (12, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (13, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (14, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
        (15, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident, $z:ident) => { $z };
    }
    pub(crate) use mw;

    /// Emits one fully-unrolled round given the 16 schedule indices as
    /// literal macro args. Each `g` call receives its message inputs as
    /// direct identifiers (either `m0..m7` or `zero`) selected by `mw!`.
    ///
    /// # Formal specification
    ///
    /// For schedule `s : INDEX^16` and chaining words `m₀..m₇ : WORD_4`:
    ///
    /// ```text
    /// round_unrolled(v, m₀..m₇, z, s) ≡
    ///   G(v, 0, 4,  8, 12, mw(s₀, m, z),  mw(s₁,  m, z));
    ///   G(v, 1, 5,  9, 13, mw(s₂, m, z),  mw(s₃,  m, z));
    ///   G(v, 2, 6, 10, 14, mw(s₄, m, z),  mw(s₅,  m, z));
    ///   G(v, 3, 7, 11, 15, mw(s₆, m, z),  mw(s₇,  m, z));
    ///   G(v, 0, 5, 10, 15, mw(s₈, m, z),  mw(s₉,  m, z));
    ///   G(v, 1, 6, 11, 12, mw(s₁₀,m, z),  mw(s₁₁, m, z));
    ///   G(v, 2, 7,  8, 13, mw(s₁₂,m, z),  mw(s₁₃, m, z));
    ///   G(v, 3, 4,  9, 14, mw(s₁₄,m, z),  mw(s₁₅, m, z));
    /// ```
    macro_rules! round_unrolled {
        ($v:expr,
         $m0:ident, $m1:ident, $m2:ident, $m3:ident,
         $m4:ident, $m5:ident, $m6:ident, $m7:ident,
         $z:ident,
         $s0:tt, $s1:tt, $s2:tt, $s3:tt,
         $s4:tt, $s5:tt, $s6:tt, $s7:tt,
         $s8:tt, $s9:tt, $s10:tt, $s11:tt,
         $s12:tt, $s13:tt, $s14:tt, $s15:tt
        ) => {
            g($v, 0, 4,  8, 12,
                mw!($s0,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s1,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 1, 5,  9, 13,
                mw!($s2,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s3,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 2, 6, 10, 14,
                mw!($s4,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s5,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 3, 7, 11, 15,
                mw!($s6,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s7,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 0, 5, 10, 15,
                mw!($s8,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s9,  $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 1, 6, 11, 12,
                mw!($s10, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s11, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 2, 7,  8, 13,
                mw!($s12, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s13, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
            g($v, 3, 4,  9, 14,
                mw!($s14, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z),
                mw!($s15, $m0,$m1,$m2,$m3,$m4,$m5,$m6,$m7,$z));
        };
    }
    pub(crate) use round_unrolled;

    /// Fully unrolled 7-round compression specialised for the iterated phase.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// compress_iter_4way :
    ///     (cv: WORD_4^8, m₀..m₇: WORD_4) → WORD_4^8
    ///
    /// post:  result = BLAKE3_compress(
    ///            chaining_value = cv,
    ///            block          = ⟨m₀,m₁,...,m₇, 0,0,...,0⟩,  -- 16 words
    ///            counter        = 0,
    ///            block_len      = 32,
    ///            flags          = CHUNK_START | CHUNK_END | ROOT)
    /// ```
    ///
    /// **Pre:** all inputs are well-formed 4-lane SIMD vectors.
    /// **Post:** `result` is the lane-parallel BLAKE3 compression of a block
    /// whose first 8 words are `m₀..m₇` and whose last 8 words are zero.
    #[inline(always)]
    unsafe fn compress_iter_4way(
        cv: &[uint32x4_t; 8],
        m0: uint32x4_t, m1: uint32x4_t, m2: uint32x4_t, m3: uint32x4_t,
        m4: uint32x4_t, m5: uint32x4_t, m6: uint32x4_t, m7: uint32x4_t,
        block_len_const: uint32x4_t,
        flags_const: uint32x4_t,
        iv_lo: &[uint32x4_t; 4],
        zero: uint32x4_t,
    ) -> [uint32x4_t; 8] {
        let mut v: [uint32x4_t; 16] = [zero; 16];
        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];
        v[8]  = iv_lo[0]; v[9]  = iv_lo[1];
        v[10] = iv_lo[2]; v[11] = iv_lo[3];
        v[12] = zero; v[13] = zero;
        v[14] = block_len_const;
        v[15] = flags_const;

        // Round 0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        // Round 1: [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        // Round 2: [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        // Round 3: [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        // Round 4: [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        // Round 5: [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        // Round 6: [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13]
        round_unrolled!(&mut v, m0, m1, m2, m3, m4, m5, m6, m7, zero,
            11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        [
            veorq_u32(v[0], v[8]),  veorq_u32(v[1], v[9]),
            veorq_u32(v[2], v[10]), veorq_u32(v[3], v[11]),
            veorq_u32(v[4], v[12]), veorq_u32(v[5], v[13]),
            veorq_u32(v[6], v[14]), veorq_u32(v[7], v[15]),
        ]
    }

    /// Iterated hashing loop, 4-way, fully unrolled.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// iterated_4way :
    ///     (cv: WORD_4^8, h₀: WORD_4^8, k: ℕ) → WORD_4^8
    ///
    /// iterated_4way(cv, h₀, 0)     = h₀
    /// iterated_4way(cv, h₀, k + 1) =
    ///     compress_iter_4way(cv, iterated_4way(cv, h₀, k))
    /// ```
    ///
    /// **Pre:** `iterations ≥ 0`.
    /// **Post:** `result = compress_iter_4way^k(cv, h₀)` where `k = iterations`.
    #[inline(never)]
    unsafe fn iterated_4way(
        cv: &[uint32x4_t; 8],
        initial_hw: [uint32x4_t; 8],
        iterations: u64,
    ) -> [uint32x4_t; 8] {
        let zero = vdupq_n_u32(0);
        let block_len_const = vdupq_n_u32(32);
        let flags_const = vdupq_n_u32(HASH_FLAGS);
        let iv_lo: [uint32x4_t; 4] = [
            vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
        ];

        let mut h0 = initial_hw[0]; let mut h1 = initial_hw[1];
        let mut h2 = initial_hw[2]; let mut h3 = initial_hw[3];
        let mut h4 = initial_hw[4]; let mut h5 = initial_hw[5];
        let mut h6 = initial_hw[6]; let mut h7 = initial_hw[7];

        for _ in 0..iterations {
            let out = compress_iter_4way(
                cv, h0, h1, h2, h3, h4, h5, h6, h7,
                block_len_const, flags_const, &iv_lo, zero,
            );
            h0 = out[0]; h1 = out[1]; h2 = out[2]; h3 = out[3];
            h4 = out[4]; h5 = out[5]; h6 = out[6]; h7 = out[7];
        }
        [h0, h1, h2, h3, h4, h5, h6, h7]
    }

    /// Mines 4 independent nonces in parallel using ARM NEON (fully unrolled).
    ///
    /// # Formal specification
    ///
    /// ```text
    /// create_extensions_4way_neon :
    ///     BYTE^32 × NONCE^4 → (NONCE × BYTE^32)^4
    ///
    /// post:  ∀ i ∈ 0..3 .
    ///            result(i).0 = nonces(i)  ∧
    ///            result(i).1 = Scalar(midstate, nonces(i))
    /// ```
    ///
    /// **Pre:** running on `aarch64` (NEON is mandatory there).
    /// **Post:** each lane's hash equals the scalar reference.
    ///
    /// # Safety
    ///
    /// NEON is mandatory on `aarch64`. `unsafe` is required because it calls
    /// NEON intrinsics.
    pub unsafe fn create_extensions_4way_neon(
        midstate: [u8; 32], nonces: [u64; 4],
    ) -> [(u64, [u8; 32]); 4] {
        let zero = vdupq_n_u32(0);
        let cv: [uint32x4_t; 8] = [
            vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]), vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]), vdupq_n_u32(IV[7]),
        ];
        let ms_words = bytes_to_words(&midstate);
        let nonce_lo: [u32; 4] = [
            nonces[0] as u32, nonces[1] as u32,
            nonces[2] as u32, nonces[3] as u32,
        ];
        let nonce_hi: [u32; 4] = [
            (nonces[0] >> 32) as u32, (nonces[1] >> 32) as u32,
            (nonces[2] >> 32) as u32, (nonces[3] >> 32) as u32,
        ];

        // Initial 40-byte compression: uses the generic compress_4way path
        // (block_len = 40 here, plus nonce-bearing msg[8..10] are nonzero).
        let mut msg: [uint32x4_t; 16] = [zero; 16];
        for i in 0..8 { msg[i] = vdupq_n_u32(ms_words[i]); }
        msg[8] = vld1q_u32(nonce_lo.as_ptr());
        msg[9] = vld1q_u32(nonce_hi.as_ptr());
        let initial_hw = compress_4way(&cv, &msg, 40);

        // Iterated hashing: uses the unrolled path.
        let hw = iterated_4way(&cv, initial_hw, EXTENSION_ITERATIONS as u64);

        [
            (nonces[0], extract_hash(&hw, 0)), (nonces[1], extract_hash(&hw, 1)),
            (nonces[2], extract_hash(&hw, 2)), (nonces[3], extract_hash(&hw, 3)),
        ]
    }



    // ═══════════════════════════════════════════════════════════════════
    //  HAND-WRITTEN AARCH64 ASSEMBLY 8-WAY ITERATED COMPRESSION
    // ═══════════════════════════════════════════════════════════════════
    //
    // # Why hand-written assembly
    //
    // Three previous Rust-based attempts to make the 8-way path faster
    // than two sequential 4-way calls all failed because LLVM aggressively
    // re-merges the two streams' instructions even when source code
    // separates them:
    //
    //   - v4 (source-interleaved):   60 ns/hash/lane, ~169 inner-loop spills
    //   - v5 (alternating streams):  asm same as v4 (LLVM merged), spills slightly reduced
    //   - v6 (alternating + black_box barrier): spills got WORSE, not better
    //
    // The fundamental problem: 32 NEON state vectors + 16 chaining words
    // = 48 simultaneously-live vectors when LLVM merges the two streams'
    // bodies. A76 has 32 NEON registers. LLVM has no choice but to spill.
    //
    // Solution: hand-write the entire iteration loop in aarch64 assembly,
    // with explicit register allocation that fits in the register file:
    //
    //   v0 - v15  : state v[0..16] during a compression (reused per stream)
    //   v16 - v23 : stream A chaining words a0..a7 (persistent across iterations)
    //   v24 - v31 : stream B chaining words b0..b7 (persistent across iterations)
    //
    // At any moment during stream A's compression, stream B's chaining
    // words sit untouched in v24..v31 (no spill). During stream B's
    // compression, stream A's NEW chaining words (just written) sit
    // untouched in v16..v23 (no spill).
    //
    // Total live during a compression: 16 (state) + 16 (both streams'
    // chaining words) = 32 = exactly the register file. No spills.
    //
    // # Trade-off
    //
    // - Constants (block_len, flags, IV[0..3]) are NOT register-resident;
    //   they're rematerialised via `movi` (one cycle, single uop) or loaded
    //   from stack per compression. Cost: ~10 cycles per compression vs the
    //   ~2000 cycles of work. Amortised: < 0.5% overhead.
    // - `cv[0..7]` is loaded from caller-supplied pointer each compression
    //   via 4 `ldp` instructions. Cost: ~4 cycles. Same amortisation.
    // - One scratch vector per G call is used for rotate temporaries. We
    //   pick the lowest state slot NOT in {a, b, c, d} (12 free slots out
    //   of 16). This works because the chosen slot is dead during the G
    //   call body — we overwrite it later in the round when the schedule
    //   reaches it.
    //
    // # Safety contract for iterated_4way_x2_asm
    //
    // ABI (custom; not standard AAPCS for the parameter passing):
    //   x0 = pointer to cv[0..7] (read-only, 128 bytes; loop-invariant)
    //   x1 = pointer to initial_a[0..7] (read-only, 128 bytes; consumed once)
    //   x2 = pointer to initial_b[0..7] (read-only, 128 bytes; consumed once)
    //   x3 = iteration count (u64); must be ≥ 1
    //   x4 = pointer to out_a[0..7] (write-only, 128 bytes; written once at exit)
    //   x5 = pointer to out_b[0..7] (write-only, 128 bytes; written once at exit)
    //
    // The function preserves AAPCS callee-saved registers (low 64 bits of
    // v8-v15) by saving them to a 144-byte stack frame on entry.
    // x16, x17 are caller-saved (IP scratch) and freely used.
    //
    // # Formal specification (Z notation)
    //
    // ```text
    // iterated_4way_x2_asm :
    //     CV* × A* × B* × ℕ × OutA* × OutB* → ⊥
    //
    // where
    //     CV*   = address of WORD_4^8  (initial chaining values)
    //     A*    = address of WORD_4^8  (stream A initial words)
    //     B*    = address of WORD_4^8  (stream B initial words)
    //     OutA* = address of WORD_4^8  (writable, stream A output)
    //     OutB* = address of WORD_4^8  (writable, stream B output)
    //
    // post:  *out_a = iterated_4way(*cv, *initial_a, k)  ∧
    //        *out_b = iterated_4way(*cv, *initial_b, k)
    //        where k = iterations.
    // ```
    //
    // The post-condition references the Rust `iterated_4way` function as
    // the specification — i.e. the asm version is required to produce
    // bit-identical output to running the unrolled 4-way Rust function
    // twice. The test `eight_way_neon_matches_two_four_way` verifies.

    // The assembly body is generated by /home/claude/gen_asm.py (see Z spec
    // above for the input/output contract). It contains exactly:
    //   - 2 × 7 = 14 rounds of 8 G calls = 112 G calls total
    //   - 560 ADDs, 464 EORs, 336 each of SHL/USHR/ORR, 112 REV32s
    //   - 8 EORs per compression × 2 compressions = 16 finalise EORs
    //   - 1 inner-loop branch
    // Per iteration: ~1900 instructions, NO inner-loop stack spills.
    core::arch::global_asm!(r#"
.global iterated_4way_x2_asm
    .type iterated_4way_x2_asm, @function
    iterated_4way_x2_asm:
    // prologue
    sub sp, sp, #144
    stp d8, d9,   [sp, #80]
    stp d10, d11, [sp, #96]
    stp d12, d13, [sp, #112]
    stp d14, d15, [sp, #128]
    // Load IV[0..3] into scratch regs, broadcast, save to stack
    mov w16, #0xE667
    movk w16, #0x6A09, lsl #16
    dup v0.4s, w16
    mov w16, #0xAE85
    movk w16, #0xBB67, lsl #16
    dup v1.4s, w16
    stp q0, q1, [sp, #16]
    mov w16, #0xF372
    movk w16, #0x3C6E, lsl #16
    dup v0.4s, w16
    mov w16, #0xF53A
    movk w16, #0xA54F, lsl #16
    dup v1.4s, w16
    stp q0, q1, [sp, #48]
    // Load initial chaining words
    ldp q16, q17, [x1]
    ldp q18, q19, [x1, #32]
    ldp q20, q21, [x1, #64]
    ldp q22, q23, [x1, #96]
    ldp q24, q25, [x2]
    ldp q26, q27, [x2, #32]
    ldp q28, q29, [x2, #64]
    ldp q30, q31, [x2, #96]
    // Outer loop: iterate `x3` times
    .Liter_loop:
    // Save v24 (b0) for use as scratch during stream A
    str q24, [sp, #0]
    // === STREAM A COMPRESSION (scratch=v24) ===
    // state init: v[0..7] = cv[0..7]
    ldp q0, q1, [x0]
    ldp q2, q3, [x0, #32]
    ldp q4, q5, [x0, #64]
    ldp q6, q7, [x0, #96]
    // state init: v[8..11] = IV[0..3]  (loaded from stack)
    ldp q8, q9, [sp, #16]
    ldp q10, q11, [sp, #48]
    // state init: v[12..13] = 0
    movi v12.4s, #0
    movi v13.4s, #0
    // state init: v[14] = 32 (block_len), v[15] = 11 (HASH_FLAGS)
    movi v14.4s, #32
    movi v15.4s, #11
    // --- Round 0 ---
    // G(0,4,8,12, mx=v16/sched[0]=0, my=v17/sched[1]=1)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v16.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v17.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=v18/sched[2]=2, my=v19/sched[3]=3)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v18.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v19.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=v20/sched[4]=4, my=v21/sched[5]=5)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v20.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v21.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=v22/sched[6]=6, my=v23/sched[7]=7)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v22.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v23.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=8, my=ZERO/sched[9]=9)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=10, my=ZERO/sched[11]=11)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=12, my=ZERO/sched[13]=13)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=14, my=ZERO/sched[15]=15)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 1 ---
    // G(0,4,8,12, mx=v18/sched[0]=2, my=v22/sched[1]=6)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v18.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v22.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=v19/sched[2]=3, my=ZERO/sched[3]=10)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v19.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=v23/sched[4]=7, my=v16/sched[5]=0)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v23.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v16.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=v20/sched[6]=4, my=ZERO/sched[7]=13)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v20.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=v17/sched[8]=1, my=ZERO/sched[9]=11)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v17.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=12, my=v21/sched[11]=5)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v21.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=9, my=ZERO/sched[13]=14)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=15, my=ZERO/sched[15]=8)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 2 ---
    // G(0,4,8,12, mx=v19/sched[0]=3, my=v20/sched[1]=4)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v19.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v20.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=10, my=ZERO/sched[3]=12)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=13, my=v18/sched[5]=2)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v18.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=v23/sched[6]=7, my=ZERO/sched[7]=14)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v23.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=v22/sched[8]=6, my=v21/sched[9]=5)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v22.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v21.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=9, my=v16/sched[11]=0)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v16.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=11, my=ZERO/sched[13]=15)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=8, my=v17/sched[15]=1)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v17.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 3 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=10, my=v23/sched[1]=7)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v23.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=12, my=ZERO/sched[3]=9)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=14, my=v19/sched[5]=3)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v19.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=13, my=ZERO/sched[7]=15)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=v20/sched[8]=4, my=v16/sched[9]=0)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v20.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v16.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=11, my=v18/sched[11]=2)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v18.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=v21/sched[12]=5, my=ZERO/sched[13]=8)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v21.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=v17/sched[14]=1, my=v22/sched[15]=6)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v17.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v22.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 4 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=12, my=ZERO/sched[1]=13)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=9, my=ZERO/sched[3]=11)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=15, my=ZERO/sched[5]=10)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=14, my=ZERO/sched[7]=8)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=v23/sched[8]=7, my=v18/sched[9]=2)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v23.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v18.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=v21/sched[10]=5, my=v19/sched[11]=3)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v21.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v19.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=v16/sched[12]=0, my=v17/sched[13]=1)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v16.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v17.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=v22/sched[14]=6, my=v20/sched[15]=4)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v22.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v20.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 5 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=9, my=ZERO/sched[1]=14)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=11, my=v21/sched[3]=5)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v21.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=8, my=ZERO/sched[5]=12)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=15, my=v17/sched[7]=1)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v17.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=13, my=v19/sched[9]=3)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v19.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=v16/sched[10]=0, my=ZERO/sched[11]=10)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v16.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=v18/sched[12]=2, my=v22/sched[13]=6)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v18.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v22.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=v20/sched[14]=4, my=v23/sched[15]=7)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v20.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v23.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // --- Round 6 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=11, my=ZERO/sched[1]=15)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // G(1,5,9,13, mx=v21/sched[2]=5, my=v16/sched[3]=0)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v21.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v16.4s
    eor v13.16b, v13.16b, v1.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(2,6,10,14, mx=v17/sched[4]=1, my=ZERO/sched[5]=9)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v17.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=8, my=v22/sched[7]=6)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v22.4s
    eor v15.16b, v15.16b, v3.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=14, my=ZERO/sched[9]=10)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v24.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v24.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v24.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v24.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v24.16b
    // G(1,6,11,12, mx=v18/sched[10]=2, my=ZERO/sched[11]=12)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v18.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v24.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v24.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v24.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v24.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v24.16b
    // G(2,7,8,13, mx=v19/sched[12]=3, my=v20/sched[13]=4)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v19.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v24.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v20.4s
    eor v13.16b, v13.16b, v2.16b
    shl v24.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v24.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v24.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v24.16b
    // G(3,4,9,14, mx=v23/sched[14]=7, my=ZERO/sched[15]=13)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v23.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v24.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v24.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v24.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v24.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v24.16b
    // finalise: new chaining words = v[i] ^ v[i+8]
    eor v16.16b, v0.16b, v8.16b
    eor v17.16b, v1.16b, v9.16b
    eor v18.16b, v2.16b, v10.16b
    eor v19.16b, v3.16b, v11.16b
    eor v20.16b, v4.16b, v12.16b
    eor v21.16b, v5.16b, v13.16b
    eor v22.16b, v6.16b, v14.16b
    eor v23.16b, v7.16b, v15.16b
    // Restore v24 (b0) for stream B's compression
    ldr q24, [sp, #0]
    // Save v16 (new a0) for use as scratch during stream B
    str q16, [sp, #0]
    // === STREAM B COMPRESSION (scratch=v16) ===
    // state init: v[0..7] = cv[0..7]
    ldp q0, q1, [x0]
    ldp q2, q3, [x0, #32]
    ldp q4, q5, [x0, #64]
    ldp q6, q7, [x0, #96]
    // state init: v[8..11] = IV[0..3]  (loaded from stack)
    ldp q8, q9, [sp, #16]
    ldp q10, q11, [sp, #48]
    // state init: v[12..13] = 0
    movi v12.4s, #0
    movi v13.4s, #0
    // state init: v[14] = 32 (block_len), v[15] = 11 (HASH_FLAGS)
    movi v14.4s, #32
    movi v15.4s, #11
    // --- Round 0 ---
    // G(0,4,8,12, mx=v24/sched[0]=0, my=v25/sched[1]=1)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v24.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v25.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=v26/sched[2]=2, my=v27/sched[3]=3)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v26.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v27.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=v28/sched[4]=4, my=v29/sched[5]=5)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v28.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v29.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=v30/sched[6]=6, my=v31/sched[7]=7)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v30.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v31.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=8, my=ZERO/sched[9]=9)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=10, my=ZERO/sched[11]=11)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=12, my=ZERO/sched[13]=13)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=14, my=ZERO/sched[15]=15)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 1 ---
    // G(0,4,8,12, mx=v26/sched[0]=2, my=v30/sched[1]=6)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v26.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v30.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=v27/sched[2]=3, my=ZERO/sched[3]=10)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v27.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=v31/sched[4]=7, my=v24/sched[5]=0)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v31.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v24.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=v28/sched[6]=4, my=ZERO/sched[7]=13)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v28.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=v25/sched[8]=1, my=ZERO/sched[9]=11)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v25.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=12, my=v29/sched[11]=5)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v29.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=9, my=ZERO/sched[13]=14)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=15, my=ZERO/sched[15]=8)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 2 ---
    // G(0,4,8,12, mx=v27/sched[0]=3, my=v28/sched[1]=4)
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v27.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v28.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=10, my=ZERO/sched[3]=12)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=13, my=v26/sched[5]=2)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v26.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=v31/sched[6]=7, my=ZERO/sched[7]=14)
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v31.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=v30/sched[8]=6, my=v29/sched[9]=5)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v30.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v29.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=9, my=v24/sched[11]=0)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v24.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=ZERO/sched[12]=11, my=ZERO/sched[13]=15)
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=ZERO/sched[14]=8, my=v25/sched[15]=1)
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v25.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 3 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=10, my=v31/sched[1]=7)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    add v0.4s, v0.4s, v31.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=12, my=ZERO/sched[3]=9)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=14, my=v27/sched[5]=3)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v27.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=13, my=ZERO/sched[7]=15)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=v28/sched[8]=4, my=v24/sched[9]=0)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v28.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v24.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=ZERO/sched[10]=11, my=v26/sched[11]=2)
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v26.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=v29/sched[12]=5, my=ZERO/sched[13]=8)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v29.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=v25/sched[14]=1, my=v30/sched[15]=6)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v25.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v30.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 4 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=12, my=ZERO/sched[1]=13)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=9, my=ZERO/sched[3]=11)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=15, my=ZERO/sched[5]=10)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=14, my=ZERO/sched[7]=8)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=v31/sched[8]=7, my=v26/sched[9]=2)
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v31.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v26.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=v29/sched[10]=5, my=v27/sched[11]=3)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v29.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v27.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=v24/sched[12]=0, my=v25/sched[13]=1)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v24.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v25.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=v30/sched[14]=6, my=v28/sched[15]=4)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v30.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v28.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 5 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=9, my=ZERO/sched[1]=14)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=ZERO/sched[2]=11, my=v29/sched[3]=5)
    add v1.4s, v1.4s, v5.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v29.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=ZERO/sched[4]=8, my=ZERO/sched[5]=12)
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=15, my=v25/sched[7]=1)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v25.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=13, my=v27/sched[9]=3)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    add v0.4s, v0.4s, v27.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=v24/sched[10]=0, my=ZERO/sched[11]=10)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v24.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=v26/sched[12]=2, my=v30/sched[13]=6)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v26.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v30.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=v28/sched[14]=4, my=v31/sched[15]=7)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v28.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v31.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // --- Round 6 ---
    // G(0,4,8,12, mx=ZERO/sched[0]=11, my=ZERO/sched[1]=15)
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    rev32 v12.8h, v12.8h
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v0.4s, v0.4s, v4.4s
    eor v12.16b, v12.16b, v0.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v8.4s, v8.4s, v12.4s
    eor v4.16b, v4.16b, v8.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // G(1,5,9,13, mx=v29/sched[2]=5, my=v24/sched[3]=0)
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v29.4s
    eor v13.16b, v13.16b, v1.16b
    rev32 v13.8h, v13.8h
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v1.4s, v1.4s, v5.4s
    add v1.4s, v1.4s, v24.4s
    eor v13.16b, v13.16b, v1.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v9.4s, v9.4s, v13.4s
    eor v5.16b, v5.16b, v9.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(2,6,10,14, mx=v25/sched[4]=1, my=ZERO/sched[5]=9)
    add v2.4s, v2.4s, v6.4s
    add v2.4s, v2.4s, v25.4s
    eor v14.16b, v14.16b, v2.16b
    rev32 v14.8h, v14.8h
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v2.4s, v2.4s, v6.4s
    eor v14.16b, v14.16b, v2.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v10.4s, v10.4s, v14.4s
    eor v6.16b, v6.16b, v10.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(3,7,11,15, mx=ZERO/sched[6]=8, my=v30/sched[7]=6)
    add v3.4s, v3.4s, v7.4s
    eor v15.16b, v15.16b, v3.16b
    rev32 v15.8h, v15.8h
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v3.4s, v3.4s, v7.4s
    add v3.4s, v3.4s, v30.4s
    eor v15.16b, v15.16b, v3.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v11.4s, v11.4s, v15.4s
    eor v7.16b, v7.16b, v11.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(0,5,10,15, mx=ZERO/sched[8]=14, my=ZERO/sched[9]=10)
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    rev32 v15.8h, v15.8h
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #20
    ushr v5.4s, v5.4s, #12
    orr v5.16b, v5.16b, v16.16b
    add v0.4s, v0.4s, v5.4s
    eor v15.16b, v15.16b, v0.16b
    shl v16.4s, v15.4s, #24
    ushr v15.4s, v15.4s, #8
    orr v15.16b, v15.16b, v16.16b
    add v10.4s, v10.4s, v15.4s
    eor v5.16b, v5.16b, v10.16b
    shl v16.4s, v5.4s, #25
    ushr v5.4s, v5.4s, #7
    orr v5.16b, v5.16b, v16.16b
    // G(1,6,11,12, mx=v26/sched[10]=2, my=ZERO/sched[11]=12)
    add v1.4s, v1.4s, v6.4s
    add v1.4s, v1.4s, v26.4s
    eor v12.16b, v12.16b, v1.16b
    rev32 v12.8h, v12.8h
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #20
    ushr v6.4s, v6.4s, #12
    orr v6.16b, v6.16b, v16.16b
    add v1.4s, v1.4s, v6.4s
    eor v12.16b, v12.16b, v1.16b
    shl v16.4s, v12.4s, #24
    ushr v12.4s, v12.4s, #8
    orr v12.16b, v12.16b, v16.16b
    add v11.4s, v11.4s, v12.4s
    eor v6.16b, v6.16b, v11.16b
    shl v16.4s, v6.4s, #25
    ushr v6.4s, v6.4s, #7
    orr v6.16b, v6.16b, v16.16b
    // G(2,7,8,13, mx=v27/sched[12]=3, my=v28/sched[13]=4)
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v27.4s
    eor v13.16b, v13.16b, v2.16b
    rev32 v13.8h, v13.8h
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #20
    ushr v7.4s, v7.4s, #12
    orr v7.16b, v7.16b, v16.16b
    add v2.4s, v2.4s, v7.4s
    add v2.4s, v2.4s, v28.4s
    eor v13.16b, v13.16b, v2.16b
    shl v16.4s, v13.4s, #24
    ushr v13.4s, v13.4s, #8
    orr v13.16b, v13.16b, v16.16b
    add v8.4s, v8.4s, v13.4s
    eor v7.16b, v7.16b, v8.16b
    shl v16.4s, v7.4s, #25
    ushr v7.4s, v7.4s, #7
    orr v7.16b, v7.16b, v16.16b
    // G(3,4,9,14, mx=v31/sched[14]=7, my=ZERO/sched[15]=13)
    add v3.4s, v3.4s, v4.4s
    add v3.4s, v3.4s, v31.4s
    eor v14.16b, v14.16b, v3.16b
    rev32 v14.8h, v14.8h
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #20
    ushr v4.4s, v4.4s, #12
    orr v4.16b, v4.16b, v16.16b
    add v3.4s, v3.4s, v4.4s
    eor v14.16b, v14.16b, v3.16b
    shl v16.4s, v14.4s, #24
    ushr v14.4s, v14.4s, #8
    orr v14.16b, v14.16b, v16.16b
    add v9.4s, v9.4s, v14.4s
    eor v4.16b, v4.16b, v9.16b
    shl v16.4s, v4.4s, #25
    ushr v4.4s, v4.4s, #7
    orr v4.16b, v4.16b, v16.16b
    // finalise: new chaining words = v[i] ^ v[i+8]
    eor v24.16b, v0.16b, v8.16b
    eor v25.16b, v1.16b, v9.16b
    eor v26.16b, v2.16b, v10.16b
    eor v27.16b, v3.16b, v11.16b
    eor v28.16b, v4.16b, v12.16b
    eor v29.16b, v5.16b, v13.16b
    eor v30.16b, v6.16b, v14.16b
    eor v31.16b, v7.16b, v15.16b
    // Restore v16 (a0) for next iteration
    ldr q16, [sp, #0]
    // Loop close
    subs x3, x3, #1
    b.ne .Liter_loop
    // epilogue: write back final chaining words
    stp q16, q17, [x4]
    stp q18, q19, [x4, #32]
    stp q20, q21, [x4, #64]
    stp q22, q23, [x4, #96]
    stp q24, q25, [x5]
    stp q26, q27, [x5, #32]
    stp q28, q29, [x5, #64]
    stp q30, q31, [x5, #96]
    // Restore callee-saved
    ldp d8, d9,   [sp, #80]
    ldp d10, d11, [sp, #96]
    ldp d12, d13, [sp, #112]
    ldp d14, d15, [sp, #128]
    add sp, sp, #144
    ret
    .size iterated_4way_x2_asm, .-iterated_4way_x2_asm
"#);

    extern "C" {
        /// Generated assembly entry point. See the contract documented above.
        fn iterated_4way_x2_asm(
            cv: *const uint32x4_t,
            initial_a: *const uint32x4_t,
            initial_b: *const uint32x4_t,
            iterations: u64,
            out_a: *mut uint32x4_t,
            out_b: *mut uint32x4_t,
        );
    }

    /// Iterated hashing loop, 8-way, hand-written aarch64 assembly.
    ///
    /// Thin Rust wrapper around the global_asm! routine. Materialises the
    /// inputs to stack arrays, calls the asm, materialises outputs.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// iterated_4way_x2 :
    ///     (cv: WORD_4^8, ha₀, hb₀: WORD_4^8, k: ℕ)
    ///     → (WORD_4^8 × WORD_4^8)
    ///
    /// post: let (a, b) = iterated_4way_x2(cv, ha₀, hb₀, k) in
    ///         a = iterated_4way(cv, ha₀, k)  ∧
    ///         b = iterated_4way(cv, hb₀, k)
    /// ```
    ///
    /// # Safety
    ///
    /// The four input/output arrays are stack-allocated `[uint32x4_t; 8]`
    /// which is exactly 128 bytes each, satisfying the asm routine's
    /// pre-conditions on pointer validity and buffer size.
    #[inline(never)]
    unsafe fn iterated_4way_x2(
        cv: &[uint32x4_t; 8],
        initial_a: [uint32x4_t; 8],
        initial_b: [uint32x4_t; 8],
        iterations: u64,
    ) -> ([uint32x4_t; 8], [uint32x4_t; 8]) {
        // If iterations == 0, the asm would loop forever (it uses `subs;b.ne`
        // and assumes count ≥ 1). Return the initial values unchanged.
        if iterations == 0 {
            return (initial_a, initial_b);
        }

        let zero = vdupq_n_u32(0);
        let mut out_a: [uint32x4_t; 8] = [zero; 8];
        let mut out_b: [uint32x4_t; 8] = [zero; 8];

        iterated_4way_x2_asm(
            cv.as_ptr(),
            initial_a.as_ptr(),
            initial_b.as_ptr(),
            iterations,
            out_a.as_mut_ptr(),
            out_b.as_mut_ptr(),
        );

        (out_a, out_b)
    }

    /// Mines 8 independent nonces in parallel via two interleaved 4-way NEON
    /// streams, fully unrolled, exploiting dual-issue ASIMD on A75+ cores
    /// (Pi 5's A76).
    ///
    /// Output is **bit-identical** to running [`create_extensions_4way_neon`]
    /// twice with nonces `[0..4]` and `[4..8]` — only the instruction schedule
    /// differs. Consensus is preserved by construction.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// create_extensions_8way_neon :
    ///     BYTE^32 × NONCE^8 → (NONCE × BYTE^32)^8
    ///
    /// post:  ∀ i ∈ 0..7 .
    ///            result(i).0 = nonces(i)  ∧
    ///            result(i).1 = Scalar(midstate, nonces(i))
    /// ```
    ///
    /// **Consensus invariant (schedule equivalence):**
    ///
    /// ```text
    /// ∀ M ∈ BYTE^32 . ∀ N ∈ NONCE^8 .
    ///     let R_lo = create_extensions_4way_neon(M, ⟨N(0..3)⟩) in
    ///     let R_hi = create_extensions_4way_neon(M, ⟨N(4..7)⟩) in
    ///     let R_8  = create_extensions_8way_neon(M, N) in
    ///         ( ∀ i ∈ 0..3 . R_8(i)     = R_lo(i) )  ∧
    ///         ( ∀ i ∈ 0..3 . R_8(i + 4) = R_hi(i) )
    /// ```
    ///
    /// **Pre:** running on `aarch64`.
    /// **Post:** each lane's hash equals the scalar reference; outputs are
    /// bit-identical to two consecutive 4-way calls covering the same nonces.
    ///
    /// # Safety
    ///
    /// NEON is mandatory on `aarch64`.
    pub unsafe fn create_extensions_8way_neon(
        midstate: [u8; 32], nonces: [u64; 8],
    ) -> [(u64, [u8; 32]); 8] {
        let zero = vdupq_n_u32(0);
        let cv: [uint32x4_t; 8] = [
            vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]), vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]), vdupq_n_u32(IV[7]),
        ];
        let ms_words = bytes_to_words(&midstate);

        // Stream A holds lanes 0–3, stream B holds lanes 4–7.
        let nonce_lo_a: [u32; 4] = [
            nonces[0] as u32, nonces[1] as u32,
            nonces[2] as u32, nonces[3] as u32,
        ];
        let nonce_hi_a: [u32; 4] = [
            (nonces[0] >> 32) as u32, (nonces[1] >> 32) as u32,
            (nonces[2] >> 32) as u32, (nonces[3] >> 32) as u32,
        ];
        let nonce_lo_b: [u32; 4] = [
            nonces[4] as u32, nonces[5] as u32,
            nonces[6] as u32, nonces[7] as u32,
        ];
        let nonce_hi_b: [u32; 4] = [
            (nonces[4] >> 32) as u32, (nonces[5] >> 32) as u32,
            (nonces[6] >> 32) as u32, (nonces[7] >> 32) as u32,
        ];

        let mut msg_a: [uint32x4_t; 16] = [zero; 16];
        let mut msg_b: [uint32x4_t; 16] = [zero; 16];
        for i in 0..8 {
            msg_a[i] = vdupq_n_u32(ms_words[i]);
            msg_b[i] = vdupq_n_u32(ms_words[i]);
        }
        msg_a[8] = vld1q_u32(nonce_lo_a.as_ptr());
        msg_a[9] = vld1q_u32(nonce_hi_a.as_ptr());
        msg_b[8] = vld1q_u32(nonce_lo_b.as_ptr());
        msg_b[9] = vld1q_u32(nonce_hi_b.as_ptr());

        // Initial 40-byte compressions, generic path.
        let initial_a = compress_4way(&cv, &msg_a, 40);
        let initial_b = compress_4way(&cv, &msg_b, 40);

        // Unrolled interleaved iterated phase.
        let (hw_a, hw_b) = iterated_4way_x2(
            &cv, initial_a, initial_b, EXTENSION_ITERATIONS as u64,
        );

        [
            (nonces[0], extract_hash(&hw_a, 0)),
            (nonces[1], extract_hash(&hw_a, 1)),
            (nonces[2], extract_hash(&hw_a, 2)),
            (nonces[3], extract_hash(&hw_a, 3)),
            (nonces[4], extract_hash(&hw_b, 0)),
            (nonces[5], extract_hash(&hw_b, 1)),
            (nonces[6], extract_hash(&hw_b, 2)),
            (nonces[7], extract_hash(&hw_b, 3)),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2 8-way (x86_64) — unchanged
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    #[inline(always)]
    unsafe fn vrot16(x: __m256i) -> __m256i {
        let mask = _mm256_set_epi8(
            13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2,
            13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2,
        );
        _mm256_shuffle_epi8(x, mask)
    }
    #[inline(always)]
    unsafe fn vrot12(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<12>(x), _mm256_slli_epi32::<20>(x))
    }
    #[inline(always)]
    unsafe fn vrot8(x: __m256i) -> __m256i {
        let mask = _mm256_set_epi8(
            12, 15, 14, 13,  8, 11, 10,  9,  4,  7,  6,  5,  0,  3,  2,  1,
            12, 15, 14, 13,  8, 11, 10,  9,  4,  7,  6,  5,  0,  3,  2,  1,
        );
        _mm256_shuffle_epi8(x, mask)
    }
    #[inline(always)]
    unsafe fn vrot7(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<7>(x), _mm256_slli_epi32::<25>(x))
    }

    #[inline(always)]
    unsafe fn g(
        v: &mut [__m256i; 16], a: usize, b: usize, c: usize, d: usize,
        mx: __m256i, my: __m256i,
    ) {
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), mx);
        v[d] = vrot16(_mm256_xor_si256(v[d], v[a]));
        v[c] = _mm256_add_epi32(v[c], v[d]);
        v[b] = vrot12(_mm256_xor_si256(v[b], v[c]));
        v[a] = _mm256_add_epi32(_mm256_add_epi32(v[a], v[b]), my);
        v[d] = vrot8(_mm256_xor_si256(v[d], v[a]));
        v[c] = _mm256_add_epi32(v[c], v[d]);
        v[b] = vrot7(_mm256_xor_si256(v[b], v[c]));
    }

    #[inline(always)]
    unsafe fn round(v: &mut [__m256i; 16], m: &[__m256i; 16], s: &[usize; 16]) {
        g(v, 0, 4,  8, 12, m[s[0]],  m[s[1]]);
        g(v, 1, 5,  9, 13, m[s[2]],  m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    #[target_feature(enable = "avx2")]
    unsafe fn compress_8way(
        cv: &[__m256i; 8], msg: &[__m256i; 16], block_len: u32,
    ) -> [__m256i; 8] {
        let zero = _mm256_setzero_si256();
        let mut v: [__m256i; 16] = [zero; 16];
        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];
        v[8]  = _mm256_set1_epi32(IV[0] as i32); v[9]  = _mm256_set1_epi32(IV[1] as i32);
        v[10] = _mm256_set1_epi32(IV[2] as i32); v[11] = _mm256_set1_epi32(IV[3] as i32);
        v[12] = zero; v[13] = zero;
        v[14] = _mm256_set1_epi32(block_len as i32);
        v[15] = _mm256_set1_epi32(HASH_FLAGS as i32);
        for r in 0..7 { round(&mut v, msg, &MSG_SCHEDULE[r]); }
        [
            _mm256_xor_si256(v[0], v[8]),  _mm256_xor_si256(v[1], v[9]),
            _mm256_xor_si256(v[2], v[10]), _mm256_xor_si256(v[3], v[11]),
            _mm256_xor_si256(v[4], v[12]), _mm256_xor_si256(v[5], v[13]),
            _mm256_xor_si256(v[6], v[14]), _mm256_xor_si256(v[7], v[15]),
        ]
    }

    unsafe fn extract_hash(out: &[__m256i; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0i32; 8];
        for i in 0..8 {
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&(buf[lane] as u32).to_le_bytes());
        }
        result
    }

    /// Mines 8 independent nonces in parallel using AVX2 256-bit SIMD.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// create_extensions_8way_avx2 :
    ///     BYTE^32 × NONCE^8 → (NONCE × BYTE^32)^8
    ///
    /// post:  ∀ i ∈ 0..7 .
    ///            result(i).0 = nonces(i)  ∧
    ///            result(i).1 = Scalar(midstate, nonces(i))
    /// ```
    ///
    /// **Pre:** AVX2 available on the host CPU.
    /// **Post:** each lane's hash equals the scalar reference.
    #[target_feature(enable = "avx2")]
    pub unsafe fn create_extensions_8way_avx2(
        midstate: [u8; 32], nonces: [u64; 8],
    ) -> [(u64, [u8; 32]); 8] {
        let zero = _mm256_setzero_si256();
        let cv: [__m256i; 8] = [
            _mm256_set1_epi32(IV[0] as i32), _mm256_set1_epi32(IV[1] as i32),
            _mm256_set1_epi32(IV[2] as i32), _mm256_set1_epi32(IV[3] as i32),
            _mm256_set1_epi32(IV[4] as i32), _mm256_set1_epi32(IV[5] as i32),
            _mm256_set1_epi32(IV[6] as i32), _mm256_set1_epi32(IV[7] as i32),
        ];
        let ms_words = bytes_to_words(&midstate);
        let nonce_lo = _mm256_set_epi32(
            nonces[7] as i32, nonces[6] as i32, nonces[5] as i32, nonces[4] as i32,
            nonces[3] as i32, nonces[2] as i32, nonces[1] as i32, nonces[0] as i32,
        );
        let nonce_hi = _mm256_set_epi32(
            (nonces[7] >> 32) as i32, (nonces[6] >> 32) as i32,
            (nonces[5] >> 32) as i32, (nonces[4] >> 32) as i32,
            (nonces[3] >> 32) as i32, (nonces[2] >> 32) as i32,
            (nonces[1] >> 32) as i32, (nonces[0] >> 32) as i32,
        );
        let mut msg: [__m256i; 16] = [zero; 16];
        for i in 0..8 { msg[i] = _mm256_set1_epi32(ms_words[i] as i32); }
        msg[8] = nonce_lo; msg[9] = nonce_hi;
        let mut hw = compress_8way(&cv, &msg, 40);
        for _ in 0..EXTENSION_ITERATIONS {
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];
            msg[8]  = zero; msg[9]  = zero; msg[10] = zero; msg[11] = zero;
            msg[12] = zero; msg[13] = zero; msg[14] = zero; msg[15] = zero;
            hw = compress_8way(&cv, &msg, 32);
        }
        [
            (nonces[0], extract_hash(&hw, 0)), (nonces[1], extract_hash(&hw, 1)),
            (nonces[2], extract_hash(&hw, 2)), (nonces[3], extract_hash(&hw, 3)),
            (nonces[4], extract_hash(&hw, 4)), (nonces[5], extract_hash(&hw, 5)),
            (nonces[6], extract_hash(&hw, 6)), (nonces[7], extract_hash(&hw, 7)),
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{hash, hash_concat};

    /// Scalar reference implementation for cross-checking all SIMD paths.
    ///
    /// Implements the canonical `Scalar` function from the module-level Z spec:
    ///
    /// ```text
    /// Scalar(M, N) = blake3^k(blake3(M ‖ N_le8))
    /// ```
    fn scalar_reference(midstate: [u8; 32], nonce: u64) -> [u8; 32] {
        let mut x = hash_concat(&midstate, &nonce.to_le_bytes());
        for _ in 0..EXTENSION_ITERATIONS {
            x = hash(&x);
        }
        x
    }

    // ── Detection ────────────────────────────────────────────────────────

    #[test]
    fn detect_returns_valid_level() {
        let level = detect();
        assert!(level.lanes() >= 4);
        println!("Detected SIMD level: {} ({} lanes)", level.name(), level.lanes());
    }

    #[test]
    fn detected_level_is_stable() {
        let a = detected_level();
        let b = detected_level();
        assert_eq!(a, b);
    }

    // ── mine_batch ───────────────────────────────────────────────────────

    #[test]
    fn mine_batch_matches_scalar() {
        let midstate = hash(b"batch test");
        let nonces: Vec<u64> = (0..detected_level().lanes() as u64).collect();
        let results = mine_batch(midstate, &nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Lane {} nonce={}", i, nonce);
        }
    }

    #[test]
    fn mine_batch_all_lanes_differ() {
        let midstate = hash(b"lane uniqueness");
        let lanes = detected_level().lanes();
        let nonces: Vec<u64> = (100..100 + lanes as u64).collect();
        let results = mine_batch(midstate, &nonces);
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(results[i].1, results[j].1);
            }
        }
    }

    #[test]
    fn mine_batch_preserves_nonces() {
        let midstate = hash(b"nonce echo");
        let lanes = detected_level().lanes();
        let nonces: Vec<u64> = (500..500 + lanes as u64).collect();
        let results = mine_batch(midstate, &nonces);
        for (i, &(nonce, _)) in results.iter().enumerate() {
            assert_eq!(nonce, nonces[i], "Nonce mismatch at lane {}", i);
        }
    }

    #[test]
    fn mine_batch_large_nonces() {
        let midstate = hash(b"large nonce test");
        let base: u64 = (1u64 << 33) + 7;
        let lanes = detected_level().lanes();
        let nonces: Vec<u64> = (0..lanes as u64).map(|i| base + i).collect();
        let results = mine_batch(midstate, &nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Large nonce lane {} nonce={}", i, nonce);
        }
    }

    // ── create_extensions_4way ───────────────────────────────────────────

    #[test]
    fn four_way_matches_scalar() {
        let midstate = hash(b"test midstate for simd");
        let nonces: [u64; 4] = [0, 1, 42, u64::MAX];
        let results = create_extensions_4way(midstate, nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Lane {} nonce={}", i, nonce);
        }
    }

    #[test]
    fn four_way_matches_create_extension() {
        use crate::core::extension::create_extension;
        let midstate = hash(b"cross-check with create_extension");
        let nonces: [u64; 4] = [7, 13, 99, 1000];
        let results = create_extensions_4way(midstate, nonces);
        for &(nonce, ref fh) in &results {
            let ext = create_extension(midstate, nonce);
            assert_eq!(*fh, ext.final_hash, "Mismatch nonce={}", nonce);
        }
    }

    #[test]
    fn four_way_large_nonces() {
        let midstate = hash(b"4way large nonce");
        let nonces: [u64; 4] = [u64::MAX, u64::MAX - 1, 1u64 << 32, (1u64 << 32) + 1];
        let results = create_extensions_4way(midstate, nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Large nonce 4way lane {} nonce={}", i, nonce);
        }
    }

    #[test]
    fn four_way_identical_nonces() {
        let midstate = hash(b"identical nonces");
        let nonces: [u64; 4] = [42, 42, 42, 42];
        let results = create_extensions_4way(midstate, nonces);
        let expected = scalar_reference(midstate, 42);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            assert_eq!(nonce, 42);
            assert_eq!(*fh, expected, "Identical nonce lane {} mismatch", i);
        }
    }

    #[test]
    fn four_way_zero_nonces() {
        let midstate = hash(b"zero nonces");
        let nonces: [u64; 4] = [0, 0, 0, 0];
        let results = create_extensions_4way(midstate, nonces);
        let expected = scalar_reference(midstate, 0);
        for &(nonce, ref fh) in &results {
            assert_eq!(nonce, 0);
            assert_eq!(*fh, expected);
        }
    }

    #[test]
    fn different_midstates_differ() {
        let m1 = hash(b"midstate A");
        let m2 = hash(b"midstate B");
        let nonces: [u64; 4] = [0, 1, 2, 3];
        let r1 = create_extensions_4way(m1, nonces);
        let r2 = create_extensions_4way(m2, nonces);
        for i in 0..4 {
            assert_ne!(r1[i].1, r2[i].1, "Midstate collision at lane {}", i);
        }
    }

    #[test]
    fn simd_level_display_matches_name() {
        let level = detected_level();
        assert_eq!(format!("{}", level), level.name());
    }

    // ── 8-way NEON specific tests ────────────────────────────────────────

    /// 8-way NEON matches the scalar reference on every lane.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_matches_scalar() {
        let midstate = hash(b"8-way neon vs scalar");
        let nonces: [u64; 8] = [
            0, 1, 42, 1000, u64::MAX, 1u64 << 32, (1u64 << 33) + 7, 99,
        ];
        let results = unsafe { neon::create_extensions_8way_neon(midstate, nonces) };
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "8-way lane {} nonce={}", i, nonce);
        }
    }

    /// **Consensus invariant test** — 8-way output is bit-identical to two 4-way calls.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_matches_two_four_way() {
        let midstate = hash(b"8-way == 2 x 4-way");
        let n8: [u64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let n4a: [u64; 4] = [10, 20, 30, 40];
        let n4b: [u64; 4] = [50, 60, 70, 80];

        let r8  = unsafe { neon::create_extensions_8way_neon(midstate, n8) };
        let r4a = unsafe { neon::create_extensions_4way_neon(midstate, n4a) };
        let r4b = unsafe { neon::create_extensions_4way_neon(midstate, n4b) };

        for i in 0..4 {
            assert_eq!(r8[i],     r4a[i], "lower half lane {}", i);
            assert_eq!(r8[i + 4], r4b[i], "upper half lane {}", i);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_matches_create_extension() {
        use crate::core::extension::create_extension;
        let midstate = hash(b"8-way vs create_extension");
        let nonces: [u64; 8] = [7, 13, 99, 1000, 0, u64::MAX, 1u64 << 40, 12345];
        let results = unsafe { neon::create_extensions_8way_neon(midstate, nonces) };
        for &(nonce, ref fh) in &results {
            let ext = create_extension(midstate, nonce);
            assert_eq!(*fh, ext.final_hash, "Mismatch nonce={}", nonce);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_zero_nonces() {
        let midstate = hash(b"8-way zero nonces");
        let nonces: [u64; 8] = [0; 8];
        let results = unsafe { neon::create_extensions_8way_neon(midstate, nonces) };
        let expected = scalar_reference(midstate, 0);
        for &(nonce, ref fh) in &results {
            assert_eq!(nonce, 0);
            assert_eq!(*fh, expected);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_identical_nonces() {
        let midstate = hash(b"8-way identical");
        let nonces: [u64; 8] = [12345; 8];
        let results = unsafe { neon::create_extensions_8way_neon(midstate, nonces) };
        let expected = scalar_reference(midstate, 12345);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            assert_eq!(nonce, 12345);
            assert_eq!(*fh, expected, "Identical lane {} mismatch", i);
        }
    }

    /// Run with: `cargo test --release bench_neon_paths -- --ignored --nocapture`
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_neon_paths() {
        use std::time::Instant;

        println!();
        println!("BUILD MARKER: hand-written-asm v7");

        let midstate = hash(b"benchmark midstate");
        let n4: [u64; 4] = [1, 2, 3, 4];
        let n8: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        for _ in 0..2 {
            let _ = unsafe { neon::create_extensions_4way_neon(midstate, n4) };
            let _ = unsafe { neon::create_extensions_8way_neon(midstate, n8) };
        }

        const ITERS: u32 = 10;

        let t4 = {
            let start = Instant::now();
            for _ in 0..ITERS {
                let r = unsafe { neon::create_extensions_4way_neon(midstate, n4) };
                std::hint::black_box(r);
            }
            start.elapsed()
        };

        let t8 = {
            let start = Instant::now();
            for _ in 0..ITERS {
                let r = unsafe { neon::create_extensions_8way_neon(midstate, n8) };
                std::hint::black_box(r);
            }
            start.elapsed()
        };

        let ns_per_hash_4 = t4.as_nanos() / (ITERS as u128 * 4);
        let ns_per_hash_8 = t8.as_nanos() / (ITERS as u128 * 8);
        let speedup = ns_per_hash_4 as f64 / ns_per_hash_8 as f64;

        println!("4-way: {:>8} ms total, {:>6} ns/hash/lane",
                 t4.as_millis(), ns_per_hash_4);
        println!("8-way: {:>8} ms total, {:>6} ns/hash/lane",
                 t8.as_millis(), ns_per_hash_8);
        println!("Per-hash speedup: {:.2}x", speedup);
        println!();
    }
}
