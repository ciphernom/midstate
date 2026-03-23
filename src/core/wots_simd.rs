//! Multi-width parallel BLAKE3 hashing for WOTS and MSS generation/verification.
//!
//! This module accelerates the creation and verification of WOTS (Winternitz
//! One-Time Signatures) and Merkle Signature Schemes (MSS). Because WOTS
//! requires dozens of independent, variable-length hash chains, we can map
//! those chains directly into physical SIMD registers.
//!
//! Hardware acceleration is provided for three targets:
//! - **AVX2** (x86_64): 8 lanes × 256-bit registers
//! - **NEON** (aarch64): 4 lanes × 128-bit registers
//! - **WASM SIMD128** (wasm32 + `simd128` target feature): 4 lanes × 128-bit registers
//! - **Scalar** fallback for all other targets or remainder chains
//!
//! ### The SIMD Masking Trick
//!
//! Because SIMD lanes execute in lockstep, Lane A might need 5 iterations
//! while Lane B needs 250 iterations. This implementation loops up to the
//! maximum iterations required by the batch, safely extracting each lane's
//! state exactly when its specific target iteration is reached, while allowing
//! the lane to "ghost hash" through the remainder of the loop without affecting
//! the already-captured result.
//!
//! ### Batch Sorting Optimisation
//!
//! Before chunking into SIMD batches, chains are sorted by their required
//! iteration count (descending) so that chains with similar depths land in the
//! same batch. This minimises the `max_iters` drag caused by one long chain
//! pulling shorter chains through thousands of ghost-hash iterations. Results
//! are unshuffled back to the original index order before returning.

use crate::core::types::hash;

// ═══════════════════════════════════════════════════════════════════════════
//  BLAKE3 Constants
// ═══════════════════════════════════════════════════════════════════════════

/// BLAKE3 initialisation vector (same as SHA-256 IV, truncated).
const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// Combined chunk flags: `CHUNK_START | CHUNK_END | ROOT`.
///
/// Each WOTS hash step operates on exactly one 32-byte chunk that is
/// simultaneously the first chunk, last chunk, and root of its tree, so all
/// three flags are set together on every compression call.
const HASH_FLAGS: u32 = 1 | 2 | 8;

/// BLAKE3 message word permutation schedule, one row per round (7 rounds total).
///
/// Each row is a permutation of `[0..16]` that determines which message words
/// feed into each `G` mixing function column/diagonal during that round.
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

// ═══════════════════════════════════════════════════════════════════════════
//  Public API & Hardware Detection
// ═══════════════════════════════════════════════════════════════════════════

/// The SIMD execution width selected at runtime (or compile time for WASM/NEON).
///
/// Each variant corresponds to a hardware backend. The `lanes()` method returns
/// how many independent WOTS chains are processed simultaneously under that
/// backend.
///
/// # Examples
///
/// ```
/// # use crate::wots_simd::{SimdLevel, detect};
/// let level = detect();
/// assert!(level.lanes() >= 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// Pure scalar fallback — one chain at a time.
    Scalar,
    /// WebAssembly 128-bit SIMD — 4 lanes.
    ///
    /// Only available when compiled with `target_arch = "wasm32"` and the
    /// `simd128` target feature enabled (e.g. `-C target-feature=+simd128`).
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Wasm128_4,
    /// ARM NEON 128-bit SIMD — 4 lanes.
    ///
    /// Always available on `aarch64`; NEON is mandatory on that target.
    #[cfg(target_arch = "aarch64")]
    Neon4,
    /// x86_64 AVX2 256-bit SIMD — 8 lanes.
    ///
    /// Detected at runtime via `std::is_x86_feature_detected!("avx2")`.
    #[cfg(target_arch = "x86_64")]
    Avx2_8,
}

impl SimdLevel {
    /// Returns the number of WOTS chains processed in parallel by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::wots_simd::SimdLevel;
    /// assert_eq!(SimdLevel::Scalar.lanes(), 1);
    /// ```
    pub fn lanes(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            SimdLevel::Wasm128_4 => 4,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => 4,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2_8 => 8,
        }
    }
}

/// Detects the best available SIMD level for the current CPU/target.
///
/// On x86_64, AVX2 support is checked at runtime. On aarch64 and WASM SIMD128
/// targets, the level is determined at compile time since those features are
/// either mandatory or statically enabled.
///
/// The result is cached via [`detected_level`] so this detection cost is
/// paid at most once per process.
///
/// # Examples
///
/// ```
/// # use crate::wots_simd::{detect, SimdLevel};
/// let level = detect();
/// // On any platform the result is always a valid, usable level.
/// assert!(level.lanes() >= 1);
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
        return SimdLevel::Neon4;
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
/// # Examples
///
/// ```
/// # use crate::wots_simd::detected_level;
/// let a = detected_level();
/// let b = detected_level();
/// assert_eq!(a, b); // Always the same once cached.
/// ```
pub fn detected_level() -> SimdLevel {
    static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(detect)
}

/// Applies `iters` sequential BLAKE3 compressions to `val` and returns the result.
///
/// This is the scalar fallback used for remainder chains that do not fill a full
/// SIMD batch, and for platforms with no SIMD support.
///
/// Passing `iters = 0` returns `val` unchanged — this is the correct behaviour
/// for a WOTS chain element whose checksum digit is zero.
///
/// # Examples
///
/// ```
/// # use crate::wots_simd::scalar_wots_chain;
/// let input = [0u8; 32];
///
/// // Zero iterations: identity.
/// assert_eq!(scalar_wots_chain(input, 0), input);
///
/// // Chaining is cumulative: 2 steps == 1 step applied twice.
/// let once  = scalar_wots_chain(input, 1);
/// let twice = scalar_wots_chain(input, 2);
/// assert_eq!(scalar_wots_chain(once, 1), twice);
/// ```
pub fn scalar_wots_chain(mut val: [u8; 32], iters: usize) -> [u8; 32] {
    for _ in 0..iters {
        val = hash(&val);
    }
    val
}

/// Processes a batch of WOTS hash chains in parallel using the best available SIMD backend.
///
/// Each element of `inputs` is an independent 32-byte chain starting value.
/// The corresponding element of `iters` specifies how many BLAKE3 hash steps
/// to apply to that chain. `inputs` and `iters` must have the same length.
///
/// Chains are sorted by iteration count (descending) before being packed into
/// SIMD batches. This reduces ghost-hash overhead by grouping chains with
/// similar depths together. Results are returned in the original caller order.
///
/// # Panics
///
/// Panics if `inputs.len() != iters.len()`.
///
/// # Examples
///
/// ```
/// # use crate::wots_simd::{process_wots_batch, scalar_wots_chain};
/// # use crate::core::types::hash;
/// let inputs = [hash(b"key0"), hash(b"key1"), hash(b"key2")];
/// let iters  = [3usize, 0, 7];
///
/// let results = process_wots_batch(&inputs, &iters);
///
/// // Each result must match the scalar reference.
/// for i in 0..inputs.len() {
///     assert_eq!(results[i], scalar_wots_chain(inputs[i], iters[i]));
/// }
/// ```
pub fn process_wots_batch(inputs: &[[u8; 32]], iters: &[usize]) -> Vec<[u8; 32]> {
    assert_eq!(inputs.len(), iters.len(), "inputs and iters length must match");
    let n = inputs.len();

    // Build a sorted index: highest iteration count first.
    // For the typical WOTS case (18 chains) this is a tiny sort.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| iters[b].cmp(&iters[a]));

    let sorted_inputs: Vec<[u8; 32]> = order.iter().map(|&i| inputs[i]).collect();
    let sorted_iters: Vec<usize>     = order.iter().map(|&i| iters[i]).collect();

    let mut sorted_results = Vec::with_capacity(n);
    let lanes = detected_level().lanes();
    let mut i = 0;

    while i < n {
        let remain = n - i;
        if remain >= lanes && lanes > 1 {
            match detected_level() {
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx2_8 => {
                    let mut starts = [[0u8; 32]; 8];
                    starts.copy_from_slice(&sorted_inputs[i..i + 8]);
                    let mut it_arr = [0usize; 8];
                    it_arr.copy_from_slice(&sorted_iters[i..i + 8]);
                    let res = unsafe { avx2::compute_8way_avx2(&starts, &it_arr) };
                    sorted_results.extend_from_slice(&res);
                    i += 8;
                }
                #[cfg(target_arch = "aarch64")]
                SimdLevel::Neon4 => {
                    let mut starts = [[0u8; 32]; 4];
                    starts.copy_from_slice(&sorted_inputs[i..i + 4]);
                    let mut it_arr = [0usize; 4];
                    it_arr.copy_from_slice(&sorted_iters[i..i + 4]);
                    let res = unsafe { neon::compute_4way_neon(&starts, &it_arr) };
                    sorted_results.extend_from_slice(&res);
                    i += 4;
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                SimdLevel::Wasm128_4 => {
                    let mut starts = [[0u8; 32]; 4];
                    starts.copy_from_slice(&sorted_inputs[i..i + 4]);
                    let mut it_arr = [0usize; 4];
                    it_arr.copy_from_slice(&sorted_iters[i..i + 4]);
                    let res = unsafe { wasm_simd::compute_4way_wasm(&starts, &it_arr) };
                    sorted_results.extend_from_slice(&res);
                    i += 4;
                }
                // `lanes > 1` is guaranteed by the outer `if`, so Scalar is
                // never reached inside this branch — but the arm is required
                // for an exhaustive match on all targets.
                SimdLevel::Scalar => unreachable!(),
            }
        } else {
            // Remainder chains, or scalar-only CPUs: one chain at a time.
            sorted_results.push(scalar_wots_chain(sorted_inputs[i], sorted_iters[i]));
            i += 1;
        }
    }

    // Unshuffle: place each result back at its original caller index.
    let mut results = vec![[0u8; 32]; n];
    for (sorted_pos, &original_idx) in order.iter().enumerate() {
        results[original_idx] = sorted_results[sorted_pos];
    }
    results
}

// ═══════════════════════════════════════════════════════════════════════════
//  WASM 128-bit SIMD (wasm32 + simd128)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use super::*;
    use core::arch::wasm32::*;

    /// Reads one little-endian `u32` word from a 32-byte array at word index `idx`.
    ///
    /// Word index `i` covers bytes `[i*4 .. i*4+4]`.
    #[inline(always)]
    fn read_word_u32(b: &[u8; 32], idx: usize) -> u32 {
        u32::from_le_bytes([b[idx * 4], b[idx * 4 + 1], b[idx * 4 + 2], b[idx * 4 + 3]])
    }

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

    /// BLAKE3 `G` mixing function operating on four interleaved WASM SIMD lanes simultaneously.
    ///
    /// Mutates state vector `v` at positions `a`, `b`, `c`, `d` using message words `mx` and `my`.
    /// Each `v128` register holds one state word from all four chains packed across its four u32 lanes.
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

    /// Applies one full BLAKE3 round (8 `G` calls: 4 column + 4 diagonal) to state `v`.
    ///
    /// `m` is the message schedule array and `s` is the permutation row for this round.
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
    /// Each element of `cv` is a `v128` holding one chaining-value word from all four chains.
    /// `msg` is laid out the same way: `msg[i]` holds word `i` from all four chains.
    /// `block_len` is broadcast to all lanes (always 32 for WOTS).
    ///
    /// Returns the 8-word output chaining value in the same transposed layout.
    #[inline(always)]
    unsafe fn compress_4way(
        cv: &[v128; 8],
        msg: &[v128; 16],
        block_len: u32,
    ) -> [v128; 8] {
        let zero = u32x4_splat(0);
        let mut v: [v128; 16] = [zero; 16];

        // Load chaining value into first 8 state words.
        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];

        // Load IV into words 8–11.
        v[8]  = u32x4_splat(IV[0]); v[9]  = u32x4_splat(IV[1]);
        v[10] = u32x4_splat(IV[2]); v[11] = u32x4_splat(IV[3]);

        // Words 12–13 are the 64-bit counter, always zero for WOTS single-block chunks.
        v[12] = zero; v[13] = zero;

        // Word 14: block length (32 bytes). Word 15: flags.
        v[14] = u32x4_splat(block_len);
        v[15] = u32x4_splat(HASH_FLAGS);

        for r in 0..7 {
            round(&mut v, msg, &MSG_SCHEDULE[r]);
        }

        // XOR upper half of state back into lower half to produce the output CV.
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

    /// Extracts the 32-byte hash for a single `lane` (0–3) from the transposed output `out`.
    ///
    /// # Safety
    ///
    /// `lane` must be in `0..4`. `out` must be a valid 8-element `v128` array representing
    /// a completed BLAKE3 compression output in transposed (SoA) layout.
    unsafe fn extract_hash(out: &[v128; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        for i in 0..8 {
            // Transmute v128 → [u32; 4] is sound: v128 is 128 bits, [u32;4] is 128 bits,
            // same endianness. This avoids a round-trip through memory.
            let words: [u32; 4] = core::mem::transmute(out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&words[lane].to_le_bytes());
        }
        result
    }

    /// Hashes 4 independent WOTS chains in parallel using WASM SIMD128 (`v128`).
    ///
    /// `starts[k]` is the 32-byte initial value for chain `k`.
    /// `iters[k]`  is the number of BLAKE3 compressions to apply to chain `k`.
    ///
    /// Chains with `iters[k] == 0` are returned unchanged (their starting value).
    /// All four chains are hashed in lockstep up to `max(iters)`, with each chain's
    /// result captured at the exact step it reaches its target (SIMD masking trick).
    ///
    /// # Safety
    ///
    /// Must only be called when the `simd128` target feature is active, which is
    /// guaranteed by the enclosing `#[cfg(...)]` gate and the [`SimdLevel::Wasm128_4`]
    /// dispatch path.
    pub unsafe fn compute_4way_wasm(
        starts: &[[u8; 32]; 4],
        iters: &[usize; 4],
    ) -> [[u8; 32]; 4] {
        // Pre-load results with the starting values so that chains with iters == 0
        // are correctly handled without entering the loop.
        let mut results = *starts;
        let max_iters = *iters.iter().max().unwrap_or(&0);
        if max_iters == 0 {
            return results;
        }

        let zero = u32x4_splat(0);

        // The BLAKE3 chaining value is always the IV for single-block, root-only
        // chunks (which is the entire WOTS use case).
        let cv: [v128; 8] = [
            u32x4_splat(IV[0]), u32x4_splat(IV[1]),
            u32x4_splat(IV[2]), u32x4_splat(IV[3]),
            u32x4_splat(IV[4]), u32x4_splat(IV[5]),
            u32x4_splat(IV[6]), u32x4_splat(IV[7]),
        ];

        // Transpose: hw[i] holds word `i` from all four input chains across its four u32 lanes.
        // Layout: hw[i] = [ starts[0][i], starts[1][i], starts[2][i], starts[3][i] ]
        let mut hw = [zero; 8];
        for i in 0..8 {
            let arr: [u32; 4] = [
                read_word_u32(&starts[0], i),
                read_word_u32(&starts[1], i),
                read_word_u32(&starts[2], i),
                read_word_u32(&starts[3], i),
            ];
            hw[i] = core::mem::transmute(arr);
        }

        // Message buffer: words 0–7 are filled from hw each step; words 8–15 stay zero
        // (WOTS input is exactly 32 bytes = 8 words, so the upper half is always zero-padded).
        let mut msg: [v128; 16] = [zero; 16];

        for step in 0..max_iters {
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];
            // msg[8..15] remain zero — correct zero-padding for a 32-byte block.

            hw = compress_4way(&cv, &msg, 32);

            // SIMD Masking Extraction: capture each lane's result the moment it
            // reaches its target iteration, then let it ghost-hash freely.
            for lane in 0..4 {
                if iters[lane] > 0 && step == iters[lane] - 1 {
                    results[lane] = extract_hash(&hw, lane);
                }
            }
        }

        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2 8-way (x86_64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    /// Reads one little-endian `i32` word from a 32-byte array at word index `idx`.
    ///
    /// Returned as `i32` to match the sign convention of `_mm256_set_epi32`.
    #[inline(always)]
    fn read_word(b: &[u8; 32], idx: usize) -> i32 {
        i32::from_le_bytes([b[idx * 4], b[idx * 4 + 1], b[idx * 4 + 2], b[idx * 4 + 3]])
    }

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
    ///
    /// Mutates state vector `v` at positions `a`, `b`, `c`, `d` using message words `mx` and `my`.
    #[inline(always)]
    unsafe fn g(
        v: &mut [__m256i; 16],
        a: usize, b: usize, c: usize, d: usize,
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

    /// Applies one full BLAKE3 round (8 `G` calls: 4 column + 4 diagonal) to state `v`.
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
    ///
    /// Each element of `cv` is a `__m256i` holding one chaining-value word from all eight chains.
    /// `msg` is laid out the same way. `block_len` is broadcast across all lanes (always 32 for WOTS).
    ///
    /// Returns the 8-word output chaining value in the same transposed layout.
    #[target_feature(enable = "avx2")]
    unsafe fn compress_8way(
        cv: &[__m256i; 8],
        msg: &[__m256i; 16],
        block_len: u32,
    ) -> [__m256i; 8] {
        let zero = _mm256_setzero_si256();
        let mut v: [__m256i; 16] = [zero; 16];

        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];

        v[8]  = _mm256_set1_epi32(IV[0] as i32); v[9]  = _mm256_set1_epi32(IV[1] as i32);
        v[10] = _mm256_set1_epi32(IV[2] as i32); v[11] = _mm256_set1_epi32(IV[3] as i32);

        // 64-bit counter, always zero for WOTS single-block chunks.
        v[12] = zero; v[13] = zero;

        v[14] = _mm256_set1_epi32(block_len as i32);
        v[15] = _mm256_set1_epi32(HASH_FLAGS as i32);

        for r in 0..7 {
            round(&mut v, msg, &MSG_SCHEDULE[r]);
        }

        [
            _mm256_xor_si256(v[0],  v[8]),
            _mm256_xor_si256(v[1],  v[9]),
            _mm256_xor_si256(v[2],  v[10]),
            _mm256_xor_si256(v[3],  v[11]),
            _mm256_xor_si256(v[4],  v[12]),
            _mm256_xor_si256(v[5],  v[13]),
            _mm256_xor_si256(v[6],  v[14]),
            _mm256_xor_si256(v[7],  v[15]),
        ]
    }

    /// Extracts the 32-byte hash for a single `lane` (0–7) from the transposed output `out`.
    ///
    /// Stores each `__m256i` to a temporary 8-word buffer and picks the word at `lane`.
    ///
    /// # Safety
    ///
    /// `lane` must be in `0..8`. `out` must be a valid completed AVX2 compression output.
    unsafe fn extract_hash(out: &[__m256i; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0i32; 8];
        for i in 0..8 {
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&(buf[lane] as u32).to_le_bytes());
        }
        result
    }

    /// Hashes 8 independent WOTS chains in parallel using AVX2 256-bit SIMD.
    ///
    /// `starts[k]` is the 32-byte initial value for chain `k`.
    /// `iters[k]`  is the number of BLAKE3 compressions to apply to chain `k`.
    ///
    /// Chains with `iters[k] == 0` are returned unchanged. All eight chains are
    /// hashed in lockstep up to `max(iters)`, with each chain's result captured at
    /// the exact step it reaches its target (SIMD masking trick).
    ///
    /// # Safety
    ///
    /// Must only be called when AVX2 is available. The `#[target_feature(enable = "avx2")]`
    /// attribute enforces this at the codegen level, and the dispatch in
    /// [`process_wots_batch`] only reaches this function after a positive runtime
    /// `is_x86_feature_detected!("avx2")` check.
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_8way_avx2(
        starts: &[[u8; 32]; 8],
        iters: &[usize; 8],
    ) -> [[u8; 32]; 8] {
        let mut results = *starts;
        let max_iters = *iters.iter().max().unwrap_or(&0);
        if max_iters == 0 {
            return results;
        }

        let zero = _mm256_setzero_si256();
        let cv: [__m256i; 8] = [
            _mm256_set1_epi32(IV[0] as i32), _mm256_set1_epi32(IV[1] as i32),
            _mm256_set1_epi32(IV[2] as i32), _mm256_set1_epi32(IV[3] as i32),
            _mm256_set1_epi32(IV[4] as i32), _mm256_set1_epi32(IV[5] as i32),
            _mm256_set1_epi32(IV[6] as i32), _mm256_set1_epi32(IV[7] as i32),
        ];

        // Transpose: hw[i] holds word `i` from all eight input chains across its eight i32 lanes.
        // _mm256_set_epi32 fills lanes high-to-low, so starts[7] goes into lane 7, starts[0] into lane 0.
        let mut hw = [zero; 8];
        for i in 0..8 {
            hw[i] = _mm256_set_epi32(
                read_word(&starts[7], i), read_word(&starts[6], i),
                read_word(&starts[5], i), read_word(&starts[4], i),
                read_word(&starts[3], i), read_word(&starts[2], i),
                read_word(&starts[1], i), read_word(&starts[0], i),
            );
        }

        let mut msg: [__m256i; 16] = [zero; 16];

        for step in 0..max_iters {
            // Fill lower 8 message words from current hash state.
            // Upper 8 words remain zero (32-byte WOTS block, zero-padded to 64 bytes).
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];

            hw = compress_8way(&cv, &msg, 32);

            // SIMD Masking Extraction
            for lane in 0..8 {
                if iters[lane] > 0 && step == iters[lane] - 1 {
                    results[lane] = extract_hash(&hw, lane);
                }
            }
        }
        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NEON 4-way (aarch64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    /// Reads one little-endian `u32` word from a 32-byte array at word index `idx`.
    #[inline(always)]
    fn read_word_u32(b: &[u8; 32], idx: usize) -> u32 {
        u32::from_le_bytes([b[idx * 4], b[idx * 4 + 1], b[idx * 4 + 2], b[idx * 4 + 3]])
    }

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

    /// BLAKE3 `G` mixing function over 4 interleaved NEON lanes simultaneously.
    ///
    /// Mutates state vector `v` at positions `a`, `b`, `c`, `d` using message words `mx` and `my`.
    #[inline(always)]
    unsafe fn g(
        v: &mut [uint32x4_t; 16],
        a: usize, b: usize, c: usize, d: usize,
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

    /// Applies one full BLAKE3 round (8 `G` calls: 4 column + 4 diagonal) to state `v`.
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
    /// Each element of `cv` is a `uint32x4_t` holding one chaining-value word from all four chains.
    /// `msg` is laid out the same way. `block_len` is broadcast across all lanes (always 32 for WOTS).
    ///
    /// Returns the 8-word output chaining value in the same transposed layout.
    unsafe fn compress_4way(
        cv: &[uint32x4_t; 8],
        msg: &[uint32x4_t; 16],
        block_len: u32,
    ) -> [uint32x4_t; 8] {
        let zero = vdupq_n_u32(0);
        let mut v: [uint32x4_t; 16] = [zero; 16];

        v[0] = cv[0]; v[1] = cv[1]; v[2] = cv[2]; v[3] = cv[3];
        v[4] = cv[4]; v[5] = cv[5]; v[6] = cv[6]; v[7] = cv[7];

        v[8]  = vdupq_n_u32(IV[0]); v[9]  = vdupq_n_u32(IV[1]);
        v[10] = vdupq_n_u32(IV[2]); v[11] = vdupq_n_u32(IV[3]);

        // 64-bit counter, always zero for WOTS single-block chunks.
        v[12] = zero; v[13] = zero;

        v[14] = vdupq_n_u32(block_len);
        v[15] = vdupq_n_u32(HASH_FLAGS);

        for r in 0..7 {
            round(&mut v, msg, &MSG_SCHEDULE[r]);
        }

        [
            veorq_u32(v[0],  v[8]),
            veorq_u32(v[1],  v[9]),
            veorq_u32(v[2],  v[10]),
            veorq_u32(v[3],  v[11]),
            veorq_u32(v[4],  v[12]),
            veorq_u32(v[5],  v[13]),
            veorq_u32(v[6],  v[14]),
            veorq_u32(v[7],  v[15]),
        ]
    }

    /// Extracts the 32-byte hash for a single `lane` (0–3) from the transposed output `out`.
    ///
    /// Stores each `uint32x4_t` to a temporary 4-word buffer and picks the word at `lane`.
    ///
    /// # Safety
    ///
    /// `lane` must be in `0..4`. `out` must be a valid completed NEON compression output.
    unsafe fn extract_hash(out: &[uint32x4_t; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0u32; 4];
        for i in 0..8 {
            vst1q_u32(buf.as_mut_ptr(), out[i]);
            result[i * 4..i * 4 + 4].copy_from_slice(&buf[lane].to_le_bytes());
        }
        result
    }

    /// Hashes 4 independent WOTS chains in parallel using ARM NEON 128-bit SIMD.
    ///
    /// `starts[k]` is the 32-byte initial value for chain `k`.
    /// `iters[k]`  is the number of BLAKE3 compressions to apply to chain `k`.
    ///
    /// Chains with `iters[k] == 0` are returned unchanged. All four chains are
    /// hashed in lockstep up to `max(iters)`, with each chain's result captured at
    /// the exact step it reaches its target (SIMD masking trick).
    ///
    /// # Safety
    ///
    /// NEON is mandatory on `aarch64`, so this function is always safe to call on
    /// that target. The `unsafe` marker is required because it calls NEON intrinsics.
    pub unsafe fn compute_4way_neon(
        starts: &[[u8; 32]; 4],
        iters: &[usize; 4],
    ) -> [[u8; 32]; 4] {
        let mut results = *starts;
        let max_iters = *iters.iter().max().unwrap_or(&0);
        if max_iters == 0 {
            return results;
        }

        let zero = vdupq_n_u32(0);
        let cv: [uint32x4_t; 8] = [
            vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]), vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]), vdupq_n_u32(IV[7]),
        ];

        // Transpose: hw[i] holds word `i` from all four input chains across its four u32 lanes.
        let mut hw = [zero; 8];
        for i in 0..8 {
            let arr = [
                read_word_u32(&starts[0], i),
                read_word_u32(&starts[1], i),
                read_word_u32(&starts[2], i),
                read_word_u32(&starts[3], i),
            ];
            hw[i] = vld1q_u32(arr.as_ptr());
        }

        let mut msg: [uint32x4_t; 16] = [zero; 16];

        for step in 0..max_iters {
            msg[0] = hw[0]; msg[1] = hw[1]; msg[2] = hw[2]; msg[3] = hw[3];
            msg[4] = hw[4]; msg[5] = hw[5]; msg[6] = hw[6]; msg[7] = hw[7];
            // msg[8..15] remain zero — correct zero-padding for a 32-byte block.

            hw = compress_4way(&cv, &msg, 32);

            // SIMD Masking Extraction
            for lane in 0..4 {
                if iters[lane] > 0 && step == iters[lane] - 1 {
                    results[lane] = extract_hash(&hw, lane);
                }
            }
        }
        results
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that `scalar_wots_chain` with 0 iterations is an identity function.
    #[test]
    fn test_scalar_chain_zero_iters_is_identity() {
        let input = hash(b"identity check");
        assert_eq!(scalar_wots_chain(input, 0), input);
    }

    /// Verifies that `scalar_wots_chain` is correctly cumulative:
    /// applying N steps in one call equals applying them one-by-one.
    #[test]
    fn test_scalar_chain_cumulative() {
        let input = hash(b"cumulative check");
        let two_steps = scalar_wots_chain(input, 2);
        let one_then_one = scalar_wots_chain(scalar_wots_chain(input, 1), 1);
        assert_eq!(two_steps, one_then_one);
    }

    /// Core correctness test: verifies all three SIMD paths (and the scalar remainder)
    /// produce results identical to the scalar reference for a batch with:
    /// - Highly varied iteration counts (exercises the masking trick at every boundary)
    /// - Zero-iteration chains (must return input unchanged)
    /// - More than 8 chains (forces a remainder to exercise the scalar fallback path)
    #[test]
    fn test_process_wots_batch_correctness() {
        let inputs = [
            hash(b"WOTS chain 0"),
            hash(b"WOTS chain 1"),
            hash(b"WOTS chain 2"),
            hash(b"WOTS chain 3"),
            hash(b"WOTS chain 4"),
            hash(b"WOTS chain 5"),
            hash(b"WOTS chain 6"),
            hash(b"WOTS chain 7"),
            hash(b"WOTS chain 8 (remainder)"),
            hash(b"WOTS chain 9 (remainder)"),
        ];
        let iterations = [0, 1, 5, 255, 12, 12, 128, 2, 55, 0];

        let simd_results = process_wots_batch(&inputs, &iterations);

        assert_eq!(simd_results.len(), inputs.len());
        for i in 0..inputs.len() {
            let expected = scalar_wots_chain(inputs[i], iterations[i]);
            assert_eq!(
                simd_results[i], expected,
                "Mismatch at index {i} with {} iterations", iterations[i]
            );
        }
    }

    /// Verifies that `process_wots_batch` handles an empty input slice without panicking.
    #[test]
    fn test_process_wots_batch_empty() {
        let results = process_wots_batch(&[], &[]);
        assert!(results.is_empty());
    }

    /// Verifies that a batch where every chain has 0 iterations returns all inputs unchanged.
    #[test]
    fn test_process_wots_batch_all_zero_iters() {
        let inputs = [hash(b"a"), hash(b"b"), hash(b"c")];
        let iters = [0usize, 0, 0];
        let results = process_wots_batch(&inputs, &iters);
        assert_eq!(results, inputs.to_vec());
    }

    /// Verifies that a batch of exactly one chain works correctly (no SIMD, pure scalar path).
    #[test]
    fn test_process_wots_batch_single_chain() {
        let input = hash(b"single");
        let iters = [17usize];
        let results = process_wots_batch(&[input], &iters);
        assert_eq!(results[0], scalar_wots_chain(input, 17));
    }

    /// Verifies that result ordering is correct after the internal sort-and-unshuffle:
    /// each result must correspond to its original input index, not the sorted order.
    #[test]
    fn test_process_wots_batch_index_ordering() {
        // Use deliberately reversed iteration counts to maximise sort disruption.
        let inputs: Vec<[u8; 32]> = (0..8u8).map(|i| hash(&[i; 1])).collect();
        let iters: Vec<usize> = (0..8).map(|i| (7 - i) * 10).collect(); // [70, 60, 50, 40, 30, 20, 10, 0]

        let results = process_wots_batch(&inputs, &iters);

        for i in 0..8 {
            let expected = scalar_wots_chain(inputs[i], iters[i]);
            assert_eq!(
                results[i], expected,
                "Index ordering failure at position {i}"
            );
        }
    }

    /// Verifies behaviour at the boundary between a full SIMD batch and a scalar remainder,
    /// using exactly `lanes + 1` chains so one chain always falls through to the scalar path.
    #[test]
    fn test_process_wots_batch_simd_remainder_boundary() {
        let lanes = detected_level().lanes();
        let n = lanes + 1;
        let inputs: Vec<[u8; 32]> = (0..n).map(|i| hash(&[i as u8; 1])).collect();
        let iters: Vec<usize> = (0..n).map(|i| i * 3).collect();

        let results = process_wots_batch(&inputs, &iters);

        for i in 0..n {
            let expected = scalar_wots_chain(inputs[i], iters[i]);
            assert_eq!(results[i], expected, "Boundary failure at index {i}");
        }
    }
}
