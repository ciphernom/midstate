//! Multi-width parallel BLAKE3 mining with automatic CPU feature detection.
//!
//! At startup, [`detect()`] queries the CPU and returns the widest available
//! SIMD path. The mining loop calls [`mine_batch()`] which dispatches to:
//!
//! | Platform       | Register width | Lanes | Nonces/batch |
//! |----------------|---------------|-------|--------------|
//! | x86_64 + AVX2  | 256-bit       | 8     | 8            |
//! | aarch64 (NEON) | 128-bit       | 4     | 4            |
//! | Scalar         | 32-bit        | 1     | 4 (serial)   |
//!
//! **Consensus safety:** Only the nonce *search* uses SIMD. Verification
//! remains scalar via `create_extension` / `blake3` crate.

use super::types::EXTENSION_ITERATIONS;

// ═══════════════════════════════════════════════════════════════════════════
//  BLAKE3 Constants (shared by all backends)
// ═══════════════════════════════════════════════════════════════════════════

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const HASH_FLAGS: u32 = 1 | 2 | 8; // CHUNK_START | CHUNK_END | ROOT

const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[inline(always)]
fn bytes_to_words(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]]);
    }
    w
}

// ═══════════════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════════════

/// The SIMD capability level detected on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No usable SIMD — batch of 4, processed serially.
    Scalar,
    /// ARM NEON: 128-bit registers, 4 lanes x 32-bit.
    #[cfg(target_arch = "aarch64")]
    Neon4,
    /// x86 AVX2: 256-bit registers, 8 lanes x 32-bit.
    #[cfg(target_arch = "x86_64")]
    Avx2_8,
}

impl SimdLevel {
    /// How many nonces are processed per batch call.
    pub fn lanes(self) -> usize {
        match self {
            SimdLevel::Scalar => 4,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => 4,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2_8 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => "NEON 4-way",
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

/// Detect the best SIMD level available on this CPU.
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
    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

/// Get the detected SIMD level (cached after first call).
pub fn detected_level() -> SimdLevel {
    static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(detect)
}

/// Mine a batch of nonces using the best available SIMD.
///
/// Returns `Vec<(nonce, final_hash)>` with `detected_level().lanes()` entries.
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
        SimdLevel::Neon4 => {
            assert!(nonces.len() >= 4);
            let n: [u64; 4] = [nonces[0], nonces[1], nonces[2], nonces[3]];
            unsafe { neon::create_extensions_4way_neon(midstate, n) }.to_vec()
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
    { unsafe { neon::create_extensions_4way_neon(midstate, nonces) } }
    #[cfg(not(target_arch = "aarch64"))]
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
//  NEON 4-way (aarch64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn vrot16(x: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
    }
    #[inline(always)]
    unsafe fn vrot12(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<12>(x), vshlq_n_u32::<20>(x))
    }
    #[inline(always)]
    unsafe fn vrot8(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<8>(x), vshlq_n_u32::<24>(x))
    }
    #[inline(always)]
    unsafe fn vrot7(x: uint32x4_t) -> uint32x4_t {
        vorrq_u32(vshrq_n_u32::<7>(x), vshlq_n_u32::<25>(x))
    }

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

    #[inline(always)]
    unsafe fn compress_4way(
        cv: &[uint32x4_t; 8], msg: &[uint32x4_t; 16], block_len: u32,
    ) -> [uint32x4_t; 8] {
        let zero = vdupq_n_u32(0);
        let mut v: [uint32x4_t; 16] = [zero; 16];
        v[0]=cv[0]; v[1]=cv[1]; v[2]=cv[2]; v[3]=cv[3];
        v[4]=cv[4]; v[5]=cv[5]; v[6]=cv[6]; v[7]=cv[7];
        v[8]=vdupq_n_u32(IV[0]); v[9]=vdupq_n_u32(IV[1]);
        v[10]=vdupq_n_u32(IV[2]); v[11]=vdupq_n_u32(IV[3]);
        v[12]=zero; v[13]=zero;
        v[14]=vdupq_n_u32(block_len); v[15]=vdupq_n_u32(HASH_FLAGS);
        for r in 0..7 { round(&mut v, msg, &MSG_SCHEDULE[r]); }
        [
            veorq_u32(v[0],v[8]),  veorq_u32(v[1],v[9]),
            veorq_u32(v[2],v[10]), veorq_u32(v[3],v[11]),
            veorq_u32(v[4],v[12]), veorq_u32(v[5],v[13]),
            veorq_u32(v[6],v[14]), veorq_u32(v[7],v[15]),
        ]
    }

    unsafe fn extract_hash(out: &[uint32x4_t; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0u32; 4];
        for i in 0..8 {
            vst1q_u32(buf.as_mut_ptr(), out[i]);
            result[i*4..i*4+4].copy_from_slice(&buf[lane].to_le_bytes());
        }
        result
    }

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
        let nonce_lo: [u32; 4] = [nonces[0] as u32, nonces[1] as u32, nonces[2] as u32, nonces[3] as u32];
        let nonce_hi: [u32; 4] = [(nonces[0]>>32) as u32, (nonces[1]>>32) as u32, (nonces[2]>>32) as u32, (nonces[3]>>32) as u32];
        let mut msg: [uint32x4_t; 16] = [zero; 16];
        for i in 0..8 { msg[i] = vdupq_n_u32(ms_words[i]); }
        msg[8] = vld1q_u32(nonce_lo.as_ptr());
        msg[9] = vld1q_u32(nonce_hi.as_ptr());
        let mut hw = compress_4way(&cv, &msg, 40);
        for _ in 0..EXTENSION_ITERATIONS {
            msg[0]=hw[0]; msg[1]=hw[1]; msg[2]=hw[2]; msg[3]=hw[3];
            msg[4]=hw[4]; msg[5]=hw[5]; msg[6]=hw[6]; msg[7]=hw[7];
            msg[8]=zero; msg[9]=zero; msg[10]=zero; msg[11]=zero;
            msg[12]=zero; msg[13]=zero; msg[14]=zero; msg[15]=zero;
            hw = compress_4way(&cv, &msg, 32);
        }
        [
            (nonces[0], extract_hash(&hw, 0)), (nonces[1], extract_hash(&hw, 1)),
            (nonces[2], extract_hash(&hw, 2)), (nonces[3], extract_hash(&hw, 3)),
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

    #[inline(always)]
    unsafe fn vrot16(x: __m256i) -> __m256i {
        let mask = _mm256_set_epi8(
            13,12,15,14, 9,8,11,10, 5,4,7,6, 1,0,3,2,
            13,12,15,14, 9,8,11,10, 5,4,7,6, 1,0,3,2,
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
            12,15,14,13, 8,11,10,9, 4,7,6,5, 0,3,2,1,
            12,15,14,13, 8,11,10,9, 4,7,6,5, 0,3,2,1,
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
        g(v, 0, 4,  8, 12, m[s[0]], m[s[1]]);
        g(v, 1, 5,  9, 13, m[s[2]], m[s[3]]);
        g(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
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
        v[0]=cv[0]; v[1]=cv[1]; v[2]=cv[2]; v[3]=cv[3];
        v[4]=cv[4]; v[5]=cv[5]; v[6]=cv[6]; v[7]=cv[7];
        v[8]=_mm256_set1_epi32(IV[0] as i32); v[9]=_mm256_set1_epi32(IV[1] as i32);
        v[10]=_mm256_set1_epi32(IV[2] as i32); v[11]=_mm256_set1_epi32(IV[3] as i32);
        v[12]=zero; v[13]=zero;
        v[14]=_mm256_set1_epi32(block_len as i32);
        v[15]=_mm256_set1_epi32(HASH_FLAGS as i32);
        for r in 0..7 { round(&mut v, msg, &MSG_SCHEDULE[r]); }
        [
            _mm256_xor_si256(v[0],v[8]),  _mm256_xor_si256(v[1],v[9]),
            _mm256_xor_si256(v[2],v[10]), _mm256_xor_si256(v[3],v[11]),
            _mm256_xor_si256(v[4],v[12]), _mm256_xor_si256(v[5],v[13]),
            _mm256_xor_si256(v[6],v[14]), _mm256_xor_si256(v[7],v[15]),
        ]
    }

    unsafe fn extract_hash(out: &[__m256i; 8], lane: usize) -> [u8; 32] {
        let mut result = [0u8; 32];
        let mut buf = [0i32; 8];
        for i in 0..8 {
            _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, out[i]);
            result[i*4..i*4+4].copy_from_slice(&(buf[lane] as u32).to_le_bytes());
        }
        result
    }

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
            (nonces[7]>>32) as i32, (nonces[6]>>32) as i32,
            (nonces[5]>>32) as i32, (nonces[4]>>32) as i32,
            (nonces[3]>>32) as i32, (nonces[2]>>32) as i32,
            (nonces[1]>>32) as i32, (nonces[0]>>32) as i32,
        );
        let mut msg: [__m256i; 16] = [zero; 16];
        for i in 0..8 { msg[i] = _mm256_set1_epi32(ms_words[i] as i32); }
        msg[8] = nonce_lo; msg[9] = nonce_hi;
        let mut hw = compress_8way(&cv, &msg, 40);
        for _ in 0..EXTENSION_ITERATIONS {
            msg[0]=hw[0]; msg[1]=hw[1]; msg[2]=hw[2]; msg[3]=hw[3];
            msg[4]=hw[4]; msg[5]=hw[5]; msg[6]=hw[6]; msg[7]=hw[7];
            msg[8]=zero; msg[9]=zero; msg[10]=zero; msg[11]=zero;
            msg[12]=zero; msg[13]=zero; msg[14]=zero; msg[15]=zero;
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

    fn scalar_reference(midstate: [u8; 32], nonce: u64) -> [u8; 32] {
        let mut x = hash_concat(&midstate, &nonce.to_le_bytes());
        for _ in 0..EXTENSION_ITERATIONS { x = hash(&x); }
        x
    }

    #[test]
    fn detect_returns_valid_level() {
        let level = detect();
        assert!(level.lanes() >= 4);
        println!("Detected SIMD level: {} ({} lanes)", level.name(), level.lanes());
    }

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
    fn mine_batch_all_lanes_differ() {
        let midstate = hash(b"lane uniqueness");
        let lanes = detected_level().lanes();
        let nonces: Vec<u64> = (100..100 + lanes as u64).collect();
        let results = mine_batch(midstate, &nonces);
        for i in 0..results.len() {
            for j in (i+1)..results.len() {
                assert_ne!(results[i].1, results[j].1);
            }
        }
    }
}
