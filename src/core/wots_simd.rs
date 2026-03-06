//! Multi-width parallel BLAKE3 hashing for WOTS and MSS generation/verification.
//!
//! This module accelerates the creation and verification of WOTS (Winternitz 
//! One-Time Signatures) and Merkle Signature Schemes (MSS). Because WOTS 
//! requires dozens of independent, variable-length hash chains, we can map 
//! those chains directly into physical SIMD registers.
//!
//! ### The SIMD Masking Trick
//! Because SIMD lanes execute in lockstep, Lane A might need 5 iterations 
//! while Lane B needs 250 iterations. This implementation loops up to the 
//! maximum iterations required by the batch, safely extracting each lane's 
//! state exactly when its specific target is reached, while allowing it to 
//! "ghost hash" for the remainder of the loop.

use crate::core::types::hash;

// ═══════════════════════════════════════════════════════════════════════════
//  BLAKE3 Constants
// ═══════════════════════════════════════════════════════════════════════════

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

// CHUNK_START | CHUNK_END | ROOT (since each WOTS step is a single 32-byte chunk)
const HASH_FLAGS: u32 = 1 | 2 | 8; 

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon4,
    #[cfg(target_arch = "x86_64")]
    Avx2_8,
}

impl SimdLevel {
    pub fn lanes(self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon4 => 4,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2_8 => 8,
        }
    }
}

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

pub fn detected_level() -> SimdLevel {
    static LEVEL: std::sync::OnceLock<SimdLevel> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(detect)
}

/// Fallback for remainders or scalar-only CPUs.
pub fn scalar_wots_chain(mut val: [u8; 32], iters: usize) -> [u8; 32] {
    for _ in 0..iters {
        val = hash(&val);
    }
    val
}

/// Process an arbitrary number of WOTS chains dynamically.
/// Automatically chunks the workload into optimal SIMD batches and uses 
/// the scalar fallback for any remainder.
pub fn process_wots_batch(inputs: &[[u8; 32]], iters: &[usize]) -> Vec<[u8; 32]> {
    assert_eq!(inputs.len(), iters.len(), "Inputs and iterations must match");
    let mut results = Vec::with_capacity(inputs.len());

    let lanes = detected_level().lanes();
    let mut i = 0;

    while i < inputs.len() {
        let remain = inputs.len() - i;
        if remain >= lanes && lanes > 1 {
            match detected_level() {
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx2_8 => {
                    let mut starts = [[0u8; 32]; 8];
                    starts.copy_from_slice(&inputs[i..i+8]);
                    let mut it_arr = [0usize; 8];
                    it_arr.copy_from_slice(&iters[i..i+8]);
                    let res = unsafe { avx2::compute_8way_avx2(&starts, &it_arr) };
                    results.extend_from_slice(&res);
                    i += 8;
                }
                #[cfg(target_arch = "aarch64")]
                SimdLevel::Neon4 => {
                    let mut starts = [[0u8; 32]; 4];
                    starts.copy_from_slice(&inputs[i..i+4]);
                    let mut it_arr = [0usize; 4];
                    it_arr.copy_from_slice(&iters[i..i+4]);
                    let res = unsafe { neon::compute_4way_neon(&starts, &it_arr) };
                    results.extend_from_slice(&res);
                    i += 4;
                }
                SimdLevel::Scalar => unreachable!(),
            }
        } else {
            // Remainder (or strict scalar CPU) processed one by one
            results.push(scalar_wots_chain(inputs[i], iters[i]));
            i += 1;
        }
    }
    results
}

// ═══════════════════════════════════════════════════════════════════════════
//  AVX2 8-way (x86_64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use core::arch::x86_64::*;

    #[inline(always)]
    fn read_word(b: &[u8; 32], idx: usize) -> i32 {
        i32::from_le_bytes([b[idx*4], b[idx*4+1], b[idx*4+2], b[idx*4+3]])
    }

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
    pub unsafe fn compute_8way_avx2(
        starts: &[[u8; 32]; 8],
        iters: &[usize; 8],
    ) -> [[u8; 32]; 8] {
        let mut results = *starts; // Pre-load 0-iteration results
        let max_iters = *iters.iter().max().unwrap_or(&0);
        if max_iters == 0 { return results; }

        let zero = _mm256_setzero_si256();
        let cv: [__m256i; 8] = [
            _mm256_set1_epi32(IV[0] as i32), _mm256_set1_epi32(IV[1] as i32),
            _mm256_set1_epi32(IV[2] as i32), _mm256_set1_epi32(IV[3] as i32),
            _mm256_set1_epi32(IV[4] as i32), _mm256_set1_epi32(IV[5] as i32),
            _mm256_set1_epi32(IV[6] as i32), _mm256_set1_epi32(IV[7] as i32),
        ];

        // Transpose the 8 input arrays into AVX2 registers
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
            // Load state into message (pad with 8 zeros since WOTS input is exactly 32 bytes)
            msg[0]=hw[0]; msg[1]=hw[1]; msg[2]=hw[2]; msg[3]=hw[3];
            msg[4]=hw[4]; msg[5]=hw[5]; msg[6]=hw[6]; msg[7]=hw[7];
            
            // Block length is strictly 32 for WOTS
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

    #[inline(always)]
    fn read_word_u32(b: &[u8; 32], idx: usize) -> u32 {
        u32::from_le_bytes([b[idx*4], b[idx*4+1], b[idx*4+2], b[idx*4+3]])
    }

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

    pub unsafe fn compute_4way_neon(
        starts: &[[u8; 32]; 4],
        iters: &[usize; 4],
    ) -> [[u8; 32]; 4] {
        let mut results = *starts; // Pre-load 0-iteration results
        let max_iters = *iters.iter().max().unwrap_or(&0);
        if max_iters == 0 { return results; }

        let zero = vdupq_n_u32(0);
        let cv: [uint32x4_t; 8] = [
            vdupq_n_u32(IV[0]), vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]), vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]), vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]), vdupq_n_u32(IV[7]),
        ];

        // Transpose the 4 input arrays into NEON registers
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
            msg[0]=hw[0]; msg[1]=hw[1]; msg[2]=hw[2]; msg[3]=hw[3];
            msg[4]=hw[4]; msg[5]=hw[5]; msg[6]=hw[6]; msg[7]=hw[7];
            
            // Block length is strictly 32 for WOTS
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

    #[test]
    fn test_process_wots_batch_correctness() {
        // Create 10 dummy hash chains with highly varied targets to test the masking trick
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

        // Verify mathematically against the standard scalar implementation
        assert_eq!(simd_results.len(), 10);
        for i in 0..10 {
            let expected = scalar_wots_chain(inputs[i], iterations[i]);
            assert_eq!(
                simd_results[i], expected, 
                "Mismatch at index {} with {} iterations", i, iterations[i]
            );
        }
    }
}
