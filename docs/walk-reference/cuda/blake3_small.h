// blake3_small.h: the parts of BLAKE3 that Midstate Walk uses.
//
// compress() runs on the GPU as well as the host. The rest (hashing a whole
// chunk, deriving context keys) is host-only. Walk's inputs are at most two
// 64-byte blocks, so no chunk tree is needed.
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define B3_HD __host__ __device__ __forceinline__
#define B3_UNROLL _Pragma("unroll")
#else
#define B3_HD inline
#define B3_UNROLL
#endif

namespace b3 {

enum : uint32_t {
  CHUNK_START = 1,
  CHUNK_END = 2,
  PARENT = 4,
  ROOT = 8,
  KEYED_HASH = 16,
  DERIVE_KEY_CONTEXT = 32,
  DERIVE_KEY_MATERIAL = 64,
};

B3_HD uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

B3_HD void g(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t mx, uint32_t my) {
  a = a + b + mx;
  d = rotr(d ^ a, 16);
  c = c + d;
  b = rotr(b ^ c, 12);
  a = a + b + my;
  d = rotr(d ^ a, 8);
  c = c + d;
  b = rotr(b ^ c, 7);
}

// The compression function. Every index is a constant once the loops are
// unrolled, so on a GPU the state and the message stay in registers. (Reading
// the message permutation from a table, as a straightforward port does, puts
// them in local memory instead.) `out` may be `cv`.
B3_HD void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter,
                    uint32_t block_len, uint32_t flags, uint32_t out[16]) {
  uint32_t s[16] = {cv[0],       cv[1],       cv[2],       cv[3],
                    cv[4],       cv[5],       cv[6],       cv[7],
                    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                    (uint32_t)counter, (uint32_t)(counter >> 32), block_len, flags};
  uint32_t m[16];
  B3_UNROLL for (int i = 0; i < 16; ++i) m[i] = block[i];
  B3_UNROLL for (int r = 0; r < 7; ++r) {
    g(s[0], s[4], s[8], s[12], m[0], m[1]);
    g(s[1], s[5], s[9], s[13], m[2], m[3]);
    g(s[2], s[6], s[10], s[14], m[4], m[5]);
    g(s[3], s[7], s[11], s[15], m[6], m[7]);
    g(s[0], s[5], s[10], s[15], m[8], m[9]);
    g(s[1], s[6], s[11], s[12], m[10], m[11]);
    g(s[2], s[7], s[8], s[13], m[12], m[13]);
    g(s[3], s[4], s[9], s[14], m[14], m[15]);
    // The message permutation: 2 6 3 10 7 0 4 13 1 11 12 5 9 14 15 8.
    const uint32_t p[16] = {m[2], m[6],  m[3],  m[10], m[7], m[0],  m[4],  m[13],
                            m[1], m[11], m[12], m[5],  m[9], m[14], m[15], m[8]};
    B3_UNROLL for (int i = 0; i < 16; ++i) m[i] = p[i];
  }
  B3_UNROLL for (int i = 0; i < 8; ++i) {
    const uint32_t c = cv[i];
    out[i] = s[i] ^ s[i + 8];
    out[i + 8] = s[i + 8] ^ c;
  }
}

// ── Host only ───────────────────────────────────────────────────────────────

static const uint32_t IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                               0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

inline uint32_t load32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline void store32(uint8_t* p, uint32_t x) {
  p[0] = (uint8_t)x;
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
}

// BLAKE3 of one chunk (at most 1,024 bytes) under `key` and `flags`, with
// `out_len` bytes of extended output.
inline void hash_chunk(const uint8_t* in, size_t len, const uint32_t key[8], uint32_t flags,
                       uint8_t* out, size_t out_len) {
  uint32_t cv[8], block[16] = {0};
  for (int i = 0; i < 8; ++i) cv[i] = key[i];
  const size_t blocks = len == 0 ? 1 : (len + 63) / 64;
  uint32_t block_len = 0, block_flags = 0;
  for (size_t b = 0; b < blocks; ++b) {
    uint8_t buf[64] = {0};
    const size_t off = b * 64, n = len - off < 64 ? len - off : 64;
    for (size_t i = 0; i < n; ++i) buf[i] = in[off + i];
    for (int i = 0; i < 16; ++i) block[i] = load32(buf + 4 * i);
    block_len = (uint32_t)n;
    block_flags = flags | (b == 0 ? (uint32_t)CHUNK_START : 0u) | (b + 1 == blocks ? (uint32_t)CHUNK_END : 0u);
    if (b + 1 < blocks) {
      uint32_t o[16];
      compress(cv, block, 0, block_len, block_flags, o);
      for (int i = 0; i < 8; ++i) cv[i] = o[i];
    }
  }
  for (uint64_t t = 0; out_len > 0; ++t) {
    uint32_t o[16];
    uint8_t bytes[64];
    compress(cv, block, t, block_len, block_flags | ROOT, o);
    for (int i = 0; i < 16; ++i) store32(bytes + 4 * i, o[i]);
    const size_t n = out_len < 64 ? out_len : 64;
    for (size_t i = 0; i < n; ++i) *out++ = bytes[i];
    out_len -= n;
  }
}

// BLAKE3's plain hash, 32 bytes.
inline void hash(const uint8_t* in, size_t len, uint8_t out[32]) { hash_chunk(in, len, IV, 0, out, 32); }

// The key that derive_key(context, ...) hashes its input under.
inline void context_key(const char* context, uint32_t key[8]) {
  size_t n = 0;
  while (context[n]) ++n;
  uint8_t k[32];
  hash_chunk((const uint8_t*)context, n, IV, DERIVE_KEY_CONTEXT, k, 32);
  for (int i = 0; i < 8; ++i) key[i] = load32(k + 4 * i);
}

// derive_key(context, input), with `out_len` bytes of output.
inline void derive_key(const char* context, const uint8_t* in, size_t len, uint8_t* out, size_t out_len) {
  uint32_t key[8];
  context_key(context, key);
  hash_chunk(in, len, key, DERIVE_KEY_MATERIAL, out, out_len);
}

}  // namespace b3
