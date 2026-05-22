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
//! where `k = EXTENSION_ITERATIONS`, `N_le8` is `N` encoded little-endian to
//! 8 bytes, and `blake3^k` denotes `k`-fold iteration of the 32-byte BLAKE3
//! compression. Every SIMD path `Φ_w` (for lane width `w ∈ {4, 8}`) satisfies
//! the **consensus invariant**:
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
///
/// Each mining hash step operates on exactly one chunk that is simultaneously
/// the first chunk, last chunk, and root of its tree, so all three flags are
/// set together on every compression call.
const HASH_FLAGS: u32 = 1 | 2 | 8;

/// BLAKE3 message word permutation schedule, one row per round (7 rounds total).
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
///
/// **Pre:** `b : BYTE^32` (enforced by type).
/// **Post:** result is the little-endian word decomposition of `b`.
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
/// Each variant corresponds to a hardware backend. The [`lanes()`](SimdLevel::lanes)
/// method returns how many nonces are processed simultaneously under that backend.
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
    ///
    /// Only available when compiled with `target_arch = "wasm32"` and the
    /// `simd128` target feature enabled (e.g. `-C target-feature=+simd128`).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm128_4,
    /// ARM NEON: 128-bit registers, 4 lanes × 32-bit.
    ///
    /// Used on in-order / single-issue NEON cores (A53, A55) when explicitly
    /// selected via `MINER_NEON_FORCE=4`. On dual-issue cores the [`Neon8`]
    /// path is faster and is the default.
    ///
    /// [`Neon8`]: SimdLevel::Neon8
    #[cfg(target_arch = "aarch64")]
    Neon4,
    /// ARM NEON dual-issue: two interleaved 4-way streams, 8 lanes total.
    ///
    /// Default on aarch64. Targets cores with two 128-bit ASIMD pipelines
    /// (A75+, X-series, Neoverse N1+, Apple M-series, Pi 5's A76). On
    /// single-pipe cores (A53/A55) it still works correctly but loses
    /// ~10–15% to [`Neon4`]; set `MINER_NEON_FORCE=4` to opt out.
    ///
    /// [`Neon4`]: SimdLevel::Neon4
    #[cfg(target_arch = "aarch64")]
    Neon8,
    /// x86 AVX2: 256-bit registers, 8 lanes × 32-bit.
    ///
    /// Detected at runtime via `std::is_x86_feature_detected!("avx2")`.
    #[cfg(target_arch = "x86_64")]
    Avx2_8,
}

impl SimdLevel {
    /// How many nonces are processed per batch call.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// lanes : SimdLevel → ℕ
    /// lanes(self) ∈ {4, 8}
    /// ```
    ///
    /// **Post:** `result ∈ {4, 8}`.
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
    ///
    /// # Formal specification
    ///
    /// ```text
    /// name : SimdLevel → STRING
    /// ∀ ℓ₁, ℓ₂ ∈ SimdLevel . ℓ₁ ≠ ℓ₂ ⇒ name(ℓ₁) ≠ name(ℓ₂)
    /// ```
    ///
    /// **Post:** the returned string is injective on `SimdLevel`.
    pub fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            SimdLevel::Wasm128_4 => "WASM SIMD128 4-way",
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => "NEON 4-way",
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon8 => "NEON 8-way (dual-issue)",
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
/// On x86_64, AVX2 support is checked at runtime. On aarch64, defaults to the
/// 8-way dual-issue path, which is faster on every out-of-order aarch64 core
/// (A75+, Apple M-series, Neoverse, Pi 5's A76). On single-issue cores like
/// A53/A55 (Pi 3, Pi Zero 2 W), set `MINER_NEON_FORCE=4` to opt into the
/// 4-way path.
///
/// On WASM SIMD128 targets, the level is determined at compile time since
/// those features are either mandatory or statically enabled.
///
/// The result is cached via [`detected_level`] so this detection cost is
/// paid at most once per process.
///
/// # Formal specification
///
/// Let `CPU ∈ {x86_64_avx2, x86_64_noavx2, aarch64, wasm32_simd128, other}`
/// denote the host's effective execution environment, and let `env` denote
/// the process environment map. Then:
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
///
/// **Pre:** none.
/// **Post:** `result ∈ SimdLevel ∧ lanes(result) ∈ {4, 8}`.
/// **Side effects:** reads `MINER_NEON_FORCE` from process environment on
/// aarch64; pure on other targets.
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
///
/// Subsequent calls return the cached result with no overhead.
///
/// # Formal specification
///
/// Let `cache : OnceLock⟨SimdLevel⟩` be a process-global cell. Then:
///
/// ```text
/// detected_level : ⊥ → SimdLevel
///
///   cache = ∅  ⇒  cache' = {detect()}  ∧  result = detect()
///   cache ≠ ∅  ⇒  cache' = cache       ∧  result = the cache
/// ```
///
/// **Pre:** none.
/// **Post:** `result = detect()` for the first ever call;
/// for all subsequent calls, `result = first_call_result`
/// (i.e. value is stable across the lifetime of the process).
/// **Invariant:** `∀ t₁, t₂ . detected_level()@t₁ = detected_level()@t₂`.
pub fn detected_level() -> SimdLevel {
    static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(detect)
}

/// Mine a batch of nonces using the best available SIMD.
///
/// Returns `Vec<(nonce, final_hash)>` with `detected_level().lanes()` entries.
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
/// where `Scalar` is the canonical scalar miner defined in the module docs.
///
/// **Pre:** `nonces.len() >= detected_level().lanes()`.
/// **Post:** result length equals `detected_level().lanes()`; each entry's
/// hash equals the scalar reference computation for that nonce.
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
///
/// Dispatches to the best available 4-lane backend (NEON, WASM SIMD128, or
/// scalar fallback). On x86_64, this function always uses the scalar path
/// since the native width is 8 lanes via AVX2.
///
/// This function is **independent** of [`detected_level`]: even when
/// `detected_level() = Neon8`, this entry point always uses the 4-way NEON
/// implementation. Callers depending on a strict 4-lane interface are
/// preserved.
///
/// # Formal specification
///
/// ```text
/// create_extensions_4way :
///     BYTE^32 × NONCE^4 → (NONCE × BYTE^32)^4
///
/// post:  ∀ i ∈ 0..3 .
///            result(i).0 = nonces(i)  ∧
///            result(i).1 = Scalar(midstate, nonces(i))
/// ```
///
/// **Pre:** none beyond the type signature.
/// **Post:** each lane's hash equals the scalar reference; nonces are echoed.
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
//  WASM 128-bit SIMD (wasm32 + simd128)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use super::*;
    use core::arch::wasm32::*;

    /// Rotates each 32-bit lane right by 16 bits.
    #[inline(always)]
    unsafe fn vrot16(x: v128) -> v128 {
        v128_or(u32x4_shr(x, 16), u32x4_shl(x, 16))
    }

    /// Rotates each 32-bit lane right by 12 bits.
    #[inline(always)]
    unsafe fn vrot12(x: v128) -> v128 {
        v128_or(u32x4_shr(x, 12), u32x4_shl(x, 20))
    }

    /// Rotates each 32-bit lane right by 8 bits.
    #[inline(always)]
    unsafe fn vrot8(x: v128) -> v128 {
        v128_or(u32x4_shr(x, 8), u32x4_shl(x, 24))
    }

    /// Rotates each 32-bit lane right by 7 bits.
    #[inline(always)]
    unsafe fn vrot7(x: v128) -> v128 {
        v128_or(u32x4_shr(x, 7), u32x4_shl(x, 25))
    }

    /// BLAKE3 `G` mixing function over 4 interleaved WASM SIMD lanes simultaneously.
    #[inline(always)]
    unsafe fn g(
        v: &mut [v128; 16],
        a: usize, b: usize, c: usize, d: usize,
        mx: v128, my: v128,
    ) {
        v[a] = u32x4_add(u32x4_add(v[a], v[b]), mx);
        v[d] = vrot16(v128_xor(v[d], v[a]));
        v[c] = u32x4_add(v[c], v[d]);
        v[b] = vrot12(v128_xor(v[b], v[c]));
        v[a] = u32x4_add(u32x4_add(v[a], v[b]), my);
        v[d] = vrot8(v128_xor(v[d], v[a]));
        v[c] = u32x4_add(v[c], v[d]);
        v[b] = vrot7(v128_xor(v[b], v[c]));
    }

    /// Applies one full BLAKE3 round (4 column + 4 diagonal `G` calls).
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

    /// Performs one BLAKE3 compression over 4 independent chains in parallel using WASM SIMD128.
    ///
    /// Each element of `cv` holds one chaining-value word from all four chains.
    /// `msg` is laid out the same way. `block_len` is broadcast across all lanes.
    #[inline(always)]
    unsafe fn compress_4way(
        cv: &[v128; 8],
        msg: &[v128; 16],
        block_len: u32,
    ) -> [v128; 8] {
        let zero = u32x4_splat(0);
        let mut v: [v128; 16] = [zero; 16];

        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];

        v[8]  = u32x4_splat(IV[0]); v[9]  = u32x4_splat(IV[1]);
        v[10] = u32x4_splat(IV[2]); v[11] = u32x4_splat(IV[3]);

        v[12] = zero; v[13] = zero;
        v[14] = u32x4_splat(block_len);
        v[15] = u32x4_splat(HASH_FLAGS);

        for r in 0..7 {
            round(&mut v, msg, &MSG_SCHEDULE[r]);
        }

        [
            v128_xor(v[0],  v[8]),
            v128_xor(v[1],  v[9]),
            v128_xor(v[2],  v[10]),
            v128_xor(v[3],  v[11]),
            v128_xor(v[4],  v[12]),
            v128_xor(v[5],  v[13]),
            v128_xor(v[6],  v[14]),
            v128_xor(v[7],  v[15]),
        ]
    }

    /// Extracts the 32-byte hash for a single `lane` (0–3) from the transposed output.
    ///
    /// # Safety
    ///
    /// `lane` must be in `0..4`. `out` must be a valid completed compression output.
    unsafe fn extract_hash(out: &[v128; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        for i in 0..8 {
            let words: [u32; 4] = core::mem::transmute(out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&words[lane].to_le_bytes());
        }
        result
    }

    /// Mines 4 independent nonces in parallel using WASM SIMD128.
    ///
    /// Performs the initial 40-byte compression (midstate ‖ nonce), then
    /// `EXTENSION_ITERATIONS` rounds of 32-byte iterated hashing, all in
    /// 4-wide lockstep.
    ///
    /// # Safety
    ///
    /// Must only be called when the `simd128` target feature is active, which
    /// is guaranteed by the enclosing `#[cfg(...)]` gate and the
    /// [`SimdLevel::Wasm128_4`] dispatch path.
    pub unsafe fn create_extensions_4way_wasm(
        midstate: [u8; 32],
        nonces: [u64; 4],
    ) -> [(u64, [u8; 32]); 4] {
        let zero = u32x4_splat(0);

        let cv: [v128; 8] = [
            u32x4_splat(IV[0]), u32x4_splat(IV[1]),
            u32x4_splat(IV[2]), u32x4_splat(IV[3]),
            u32x4_splat(IV[4]), u32x4_splat(IV[5]),
            u32x4_splat(IV[6]), u32x4_splat(IV[7]),
        ];

        let ms_words = bytes_to_words(&midstate);

        // Split each 64-bit nonce into low and high 32-bit halves, packed across lanes.
        let nonce_lo: [u32; 4] = [
            nonces[0] as u32, nonces[1] as u32,
            nonces[2] as u32, nonces[3] as u32,
        ];
        let nonce_hi: [u32; 4] = [
            (nonces[0] >> 32) as u32, (nonces[1] >> 32) as u32,
            (nonces[2] >> 32) as u32, (nonces[3] >> 32) as u32,
        ];

        // Build message: words 0–7 = midstate (broadcast), words 8–9 = nonce, rest zero.
        let mut msg: [v128; 16] = [zero; 16];
        for i in 0..8 {
            msg[i] = u32x4_splat(ms_words[i]);
        }
        msg[8] = core::mem::transmute(nonce_lo);
        msg[9] = core::mem::transmute(nonce_hi);

        // Initial compression: 40-byte block (32 midstate + 8 nonce).
        let mut hw = compress_4way(&cv, &msg, 40);

        // Iterated hashing: EXTENSION_ITERATIONS rounds of 32-byte blocks.
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
//  NEON 4-way and 8-way (aarch64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

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

    // ─── 4-way primitives ────────────────────────────────────────────────

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
        g(v, 0, 4,  8, 12, m[s[0]], m[s[1]]);
        g(v, 1, 5,  9, 13, m[s[2]], m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    /// Performs one BLAKE3 compression over 4 independent chains in parallel using NEON.
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

    /// Mines 4 independent nonces in parallel using ARM NEON.
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
    /// NEON is mandatory on `aarch64`, so this is always safe to call on that
    /// target. The `unsafe` marker is required because it calls NEON intrinsics.
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
        let mut msg: [uint32x4_t; 16] = [zero; 16];
        for i in 0..8 { msg[i] = vdupq_n_u32(ms_words[i]); }
        msg[8] = vld1q_u32(nonce_lo.as_ptr());
        msg[9] = vld1q_u32(nonce_hi.as_ptr());
        let mut hw = compress_4way(&cv, &msg, 40);
        for _ in 0..EXTENSION_ITERATIONS {
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];
            msg[8]  = zero; msg[9]  = zero; msg[10] = zero; msg[11] = zero;
            msg[12] = zero; msg[13] = zero; msg[14] = zero; msg[15] = zero;
            hw = compress_4way(&cv, &msg, 32);
        }
        [
            (nonces[0], extract_hash(&hw, 0)), (nonces[1], extract_hash(&hw, 1)),
            (nonces[2], extract_hash(&hw, 2)), (nonces[3], extract_hash(&hw, 3)),
        ]
    }

    // ─── 8-way dual-issue primitives ─────────────────────────────────────

    /// BLAKE3 `G` mixing function over **two interleaved 4-way streams**.
    ///
    /// Each pair of NEON ops (`va` line then `vb` line) is data-independent,
    /// allowing dual-issue cores (A75+, Neoverse, Apple M-series, Pi 5's A76)
    /// to dispatch them on separate ASIMD pipes in the same cycle.
    ///
    /// # Formal specification
    ///
    /// Let `G_4way(v, a, b, c, d, mx, my)` denote the single-stream `G`
    /// function defined above. Then `g_x2` satisfies, for all valid inputs:
    ///
    /// ```text
    /// g_x2(va, vb, a, b, c, d, mxa, mya, mxb, myb) ≡
    ///     ( G_4way(va, a, b, c, d, mxa, mya)
    ///     ‖ G_4way(vb, a, b, c, d, mxb, myb) )
    /// ```
    ///
    /// where `‖` denotes independent parallel composition (no inter-stream
    /// data flow). Streams `va` and `vb` therefore evolve identically to
    /// two independent calls to `g`.
    #[inline(always)]
    unsafe fn g_x2(
        va: &mut [uint32x4_t; 16], vb: &mut [uint32x4_t; 16],
        a: usize, b: usize, c: usize, d: usize,
        mxa: uint32x4_t, mya: uint32x4_t,
        mxb: uint32x4_t, myb: uint32x4_t,
    ) {
        va[a] = vaddq_u32(vaddq_u32(va[a], va[b]), mxa);
        vb[a] = vaddq_u32(vaddq_u32(vb[a], vb[b]), mxb);
        va[d] = vrot16(veorq_u32(va[d], va[a]));
        vb[d] = vrot16(veorq_u32(vb[d], vb[a]));
        va[c] = vaddq_u32(va[c], va[d]);
        vb[c] = vaddq_u32(vb[c], vb[d]);
        va[b] = vrot12(veorq_u32(va[b], va[c]));
        vb[b] = vrot12(veorq_u32(vb[b], vb[c]));
        va[a] = vaddq_u32(vaddq_u32(va[a], va[b]), mya);
        vb[a] = vaddq_u32(vaddq_u32(vb[a], vb[b]), myb);
        va[d] = vrot8(veorq_u32(va[d], va[a]));
        vb[d] = vrot8(veorq_u32(vb[d], vb[a]));
        va[c] = vaddq_u32(va[c], va[d]);
        vb[c] = vaddq_u32(vb[c], vb[d]);
        va[b] = vrot7(veorq_u32(va[b], va[c]));
        vb[b] = vrot7(veorq_u32(vb[b], vb[c]));
    }

    /// One full BLAKE3 round over two interleaved 4-way streams.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// round_x2(va, vb, ma, mb, s) ≡
    ///     ( round(va, ma, s) ‖ round(vb, mb, s) )
    /// ```
    #[inline(always)]
    unsafe fn round_x2(
        va: &mut [uint32x4_t; 16], vb: &mut [uint32x4_t; 16],
        ma: &[uint32x4_t; 16], mb: &[uint32x4_t; 16],
        s: &[usize; 16],
    ) {
        g_x2(va, vb, 0, 4,  8, 12, ma[s[0]],  ma[s[1]],  mb[s[0]],  mb[s[1]]);
        g_x2(va, vb, 1, 5,  9, 13, ma[s[2]],  ma[s[3]],  mb[s[2]],  mb[s[3]]);
        g_x2(va, vb, 2, 6, 10, 14, ma[s[4]],  ma[s[5]],  mb[s[4]],  mb[s[5]]);
        g_x2(va, vb, 3, 7, 11, 15, ma[s[6]],  ma[s[7]],  mb[s[6]],  mb[s[7]]);
        g_x2(va, vb, 0, 5, 10, 15, ma[s[8]],  ma[s[9]],  mb[s[8]],  mb[s[9]]);
        g_x2(va, vb, 1, 6, 11, 12, ma[s[10]], ma[s[11]], mb[s[10]], mb[s[11]]);
        g_x2(va, vb, 2, 7,  8, 13, ma[s[12]], ma[s[13]], mb[s[12]], mb[s[13]]);
        g_x2(va, vb, 3, 4,  9, 14, ma[s[14]], ma[s[15]], mb[s[14]], mb[s[15]]);
    }

    /// Two parallel 4-way BLAKE3 compressions, interleaved for dual-issue.
    ///
    /// Register pressure: 32 × `uint32x4_t` exceeds the 32 NEON registers, so
    /// the compiler will spill. That's expected — spills become L1 loads that
    /// the OoO core hides behind the dual-issued ALU work. Net throughput on
    /// A76 is ~1.8–2.0× a single `compress_4way` call.
    ///
    /// # Formal specification
    ///
    /// ```text
    /// compress_4way_x2(cv, msg_a, msg_b, block_len) =
    ///     ( compress_4way(cv, msg_a, block_len)
    ///     , compress_4way(cv, msg_b, block_len) )
    /// ```
    ///
    /// **Pre:** all arrays well-formed.
    /// **Post:** returned tuple `(out_a, out_b)` satisfies
    /// `out_a = compress_4way(cv, msg_a, block_len)` and
    /// `out_b = compress_4way(cv, msg_b, block_len)`.
    #[inline(always)]
    unsafe fn compress_4way_x2(
        cv: &[uint32x4_t; 8],
        msg_a: &[uint32x4_t; 16], msg_b: &[uint32x4_t; 16],
        block_len: u32,
    ) -> ([uint32x4_t; 8], [uint32x4_t; 8]) {
        let zero = vdupq_n_u32(0);
        let mut va: [uint32x4_t; 16] = [zero; 16];
        let mut vb: [uint32x4_t; 16] = [zero; 16];

        // Initialise both streams identically from cv + IV + counter + block_len + flags.
        va[0] = cv[0]; vb[0] = cv[0];
        va[1] = cv[1]; vb[1] = cv[1];
        va[2] = cv[2]; vb[2] = cv[2];
        va[3] = cv[3]; vb[3] = cv[3];
        va[4] = cv[4]; vb[4] = cv[4];
        va[5] = cv[5]; vb[5] = cv[5];
        va[6] = cv[6]; vb[6] = cv[6];
        va[7] = cv[7]; vb[7] = cv[7];

        va[8]  = vdupq_n_u32(IV[0]); vb[8]  = vdupq_n_u32(IV[0]);
        va[9]  = vdupq_n_u32(IV[1]); vb[9]  = vdupq_n_u32(IV[1]);
        va[10] = vdupq_n_u32(IV[2]); vb[10] = vdupq_n_u32(IV[2]);
        va[11] = vdupq_n_u32(IV[3]); vb[11] = vdupq_n_u32(IV[3]);
        // va[12], va[13], vb[12], vb[13] already zero from init.
        va[14] = vdupq_n_u32(block_len); vb[14] = vdupq_n_u32(block_len);
        va[15] = vdupq_n_u32(HASH_FLAGS); vb[15] = vdupq_n_u32(HASH_FLAGS);

        for r in 0..7 {
            round_x2(&mut va, &mut vb, msg_a, msg_b, &MSG_SCHEDULE[r]);
        }

        let out_a = [
            veorq_u32(va[0], va[8]),  veorq_u32(va[1], va[9]),
            veorq_u32(va[2], va[10]), veorq_u32(va[3], va[11]),
            veorq_u32(va[4], va[12]), veorq_u32(va[5], va[13]),
            veorq_u32(va[6], va[14]), veorq_u32(va[7], va[15]),
        ];
        let out_b = [
            veorq_u32(vb[0], vb[8]),  veorq_u32(vb[1], vb[9]),
            veorq_u32(vb[2], vb[10]), veorq_u32(vb[3], vb[11]),
            veorq_u32(vb[4], vb[12]), veorq_u32(vb[5], vb[13]),
            veorq_u32(vb[6], vb[14]), veorq_u32(vb[7], vb[15]),
        ];
        (out_a, out_b)
    }

    /// Mines 8 independent nonces in parallel via two interleaved 4-way NEON
    /// streams, exploiting dual-issue ASIMD on A75+ cores (Pi 5 is A76).
    ///
    /// Output is **bit-identical** to running [`create_extensions_4way_neon`]
    /// twice with nonces `[0..4]` and `[4..8]` — only the instruction schedule
    /// differs. Consensus is preserved by construction.
    ///
    /// # Formal specification
    ///
    /// Let `Φ_4 = create_extensions_4way_neon` and `Φ_8 = create_extensions_8way_neon`.
    /// Then:
    ///
    /// ```text
    /// Φ_8 : BYTE^32 × NONCE^8 → (NONCE × BYTE^32)^8
    ///
    /// pre:   true
    /// post:  ∀ i ∈ 0..7 .
    ///            result(i).0 = nonces(i)  ∧
    ///            result(i).1 = Scalar(midstate, nonces(i))
    /// ```
    ///
    /// **Consensus invariant (schedule equivalence):**
    ///
    /// ```text
    /// ∀ M ∈ BYTE^32 . ∀ N ∈ NONCE^8 .
    ///     let R_lo = Φ_4(M, ⟨N(0), N(1), N(2), N(3)⟩) in
    ///     let R_hi = Φ_4(M, ⟨N(4), N(5), N(6), N(7)⟩) in
    ///     let R_8  = Φ_8(M, N) in
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
    /// NEON is mandatory on `aarch64`, so this is always safe to call on that
    /// target regardless of which microarchitecture is hosting it. The
    /// `unsafe` marker is required because it calls NEON intrinsics.
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

        let (mut hw_a, mut hw_b) = compress_4way_x2(&cv, &msg_a, &msg_b, 40);

        for _ in 0..EXTENSION_ITERATIONS {
            msg_a[0] = hw_a[0]; msg_a[1] = hw_a[1]; msg_a[2] = hw_a[2]; msg_a[3] = hw_a[3];
            msg_a[4] = hw_a[4]; msg_a[5] = hw_a[5]; msg_a[6] = hw_a[6]; msg_a[7] = hw_a[7];
            msg_b[0] = hw_b[0]; msg_b[1] = hw_b[1]; msg_b[2] = hw_b[2]; msg_b[3] = hw_b[3];
            msg_b[4] = hw_b[4]; msg_b[5] = hw_b[5]; msg_b[6] = hw_b[6]; msg_b[7] = hw_b[7];
            msg_a[8] = zero; msg_a[9] = zero;
            msg_b[8] = zero; msg_b[9] = zero;
            // msg_a[10..16] and msg_b[10..16] remain zero from init.
            let (a, b) = compress_4way_x2(&cv, &msg_a, &msg_b, 32);
            hw_a = a;
            hw_b = b;
        }

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
//  AVX2 8-way (x86_64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// Rotates each 32-bit lane right by 16 bits using a byte-shuffle (faster than shift+or on AVX2).
    #[inline(always)]
    unsafe fn vrot16(x: __m256i) -> __m256i {
        let mask = _mm256_set_epi8(
            13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2,
            13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2,
        );
        _mm256_shuffle_epi8(x, mask)
    }
    /// Rotates each 32-bit lane right by 12 bits.
    #[inline(always)]
    unsafe fn vrot12(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<12>(x), _mm256_slli_epi32::<20>(x))
    }
    /// Rotates each 32-bit lane right by 8 bits using a byte-shuffle.
    #[inline(always)]
    unsafe fn vrot8(x: __m256i) -> __m256i {
        let mask = _mm256_set_epi8(
            12, 15, 14, 13,  8, 11, 10,  9,  4,  7,  6,  5,  0,  3,  2,  1,
            12, 15, 14, 13,  8, 11, 10,  9,  4,  7,  6,  5,  0,  3,  2,  1,
        );
        _mm256_shuffle_epi8(x, mask)
    }
    /// Rotates each 32-bit lane right by 7 bits.
    #[inline(always)]
    unsafe fn vrot7(x: __m256i) -> __m256i {
        _mm256_or_si256(_mm256_srli_epi32::<7>(x), _mm256_slli_epi32::<25>(x))
    }

    /// BLAKE3 `G` mixing function over 8 interleaved AVX2 lanes simultaneously.
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

    /// Applies one full BLAKE3 round (4 column + 4 diagonal `G` calls).
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

    /// Performs one BLAKE3 compression over 8 independent chains in parallel using AVX2.
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

    /// Extracts the 32-byte hash for a single `lane` (0–7) from the transposed output.
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
    ///
    /// # Safety
    ///
    /// Must only be called when AVX2 is available. The `#[target_feature(enable = "avx2")]`
    /// attribute enforces this at codegen, and [`mine_batch`] only reaches this
    /// function after a positive runtime `is_x86_feature_detected!("avx2")` check.
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

    /// Verifies that detection returns a valid level with at least 4 lanes.
    #[test]
    fn detect_returns_valid_level() {
        let level = detect();
        assert!(level.lanes() >= 4);
        println!("Detected SIMD level: {} ({} lanes)", level.name(), level.lanes());
    }

    /// Verifies that `detected_level` returns a stable, cached result.
    #[test]
    fn detected_level_is_stable() {
        let a = detected_level();
        let b = detected_level();
        assert_eq!(a, b);
    }

    // ── mine_batch ───────────────────────────────────────────────────────

    /// Core correctness: every lane from `mine_batch` must match the scalar reference.
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

    /// Verifies that all lanes produce distinct hashes (different nonces → different results).
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

    /// Verifies that `mine_batch` correctly returns the nonces it was given.
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

    /// Verifies mine_batch with large nonce values (exercises high 32-bit half of u64 split).
    #[test]
    fn mine_batch_large_nonces() {
        let midstate = hash(b"large nonce test");
        let base: u64 = (1u64 << 33) + 7; // Ensures nonce_hi != 0
        let lanes = detected_level().lanes();
        let nonces: Vec<u64> = (0..lanes as u64).map(|i| base + i).collect();
        let results = mine_batch(midstate, &nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Large nonce lane {} nonce={}", i, nonce);
        }
    }

    // ── create_extensions_4way ───────────────────────────────────────────

    /// Core 4-way correctness against the scalar reference.
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

    /// Cross-checks 4-way results against `create_extension` (the canonical scalar path).
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

    /// Verifies 4-way with nonces that exercise the upper 32-bit half of the u64 split.
    #[test]
    fn four_way_large_nonces() {
        let midstate = hash(b"4way large nonce");
        let nonces: [u64; 4] = [
            u64::MAX,
            u64::MAX - 1,
            1u64 << 32,
            (1u64 << 32) + 1,
        ];
        let results = create_extensions_4way(midstate, nonces);
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "Large nonce 4way lane {} nonce={}", i, nonce);
        }
    }

    /// Verifies 4-way when all nonces are identical (degenerate but valid input).
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

    /// Verifies 4-way with nonce = 0 in all lanes (zero-value edge case).
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

    /// Verifies that different midstates produce different results for the same nonce.
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

    /// Verifies the `Display` impl for `SimdLevel` matches `name()`.
    #[test]
    fn simd_level_display_matches_name() {
        let level = detected_level();
        assert_eq!(format!("{}", level), level.name());
    }

    // ── 8-way NEON specific tests ────────────────────────────────────────

    /// 8-way NEON matches the scalar reference on every lane.
    ///
    /// This is the primary correctness test: it directly checks the post
    /// condition of `create_extensions_8way_neon` against the canonical
    /// `Scalar` definition from the module spec.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn eight_way_neon_matches_scalar() {
        let midstate = hash(b"8-way neon vs scalar");
        let nonces: [u64; 8] = [
            0, 1, 42, 1000,
            u64::MAX, 1u64 << 32, (1u64 << 33) + 7, 99,
        ];
        let results = unsafe { neon::create_extensions_8way_neon(midstate, nonces) };
        for (i, &(nonce, ref fh)) in results.iter().enumerate() {
            let expected = scalar_reference(midstate, nonces[i]);
            assert_eq!(*fh, expected, "8-way lane {} nonce={}", i, nonce);
        }
    }

    /// 8-way NEON output is bit-identical to two consecutive 4-way calls.
    ///
    /// This is the **consensus-preservation invariant** in test form: it
    /// directly checks the schedule-equivalence post condition documented
    /// in the Z spec of `create_extensions_8way_neon`. If this passes, the
    /// new path cannot produce a different hash than the old 4-way path.
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

    /// Cross-check 8-way against `create_extension` (the canonical scalar path).
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

    /// 8-way NEON with all-zero nonces (degenerate but valid input).
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

    /// 8-way NEON with all-identical nonces (verifies determinism across lanes).
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
    
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn bench_neon_paths() {
        use std::time::Instant;

        let midstate = hash(b"benchmark midstate");
        let n4: [u64; 4] = [1, 2, 3, 4];
        let n8: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        // Warmup
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

        println!();
        println!("4-way: {:>8} ms total, {:>6} ns/hash/lane",
                 t4.as_millis(), ns_per_hash_4);
        println!("8-way: {:>8} ms total, {:>6} ns/hash/lane",
                 t8.as_millis(), ns_per_hash_8);
        println!("Per-hash speedup: {:.2}x", speedup);
        println!();
    }
    
}

