// walk_core.h: Midstate Walk (version 1) for the GPU and the host.
//
// The CUDA kernels and the CPU reference run this same code. The one
// difference is where the AES table lives: shared memory on the GPU, an
// ordinary array on the host. The specification is midstate's
// docs/MIDSTATE_WALK.md; src/core/walk.rs pins the known answers below.
//
// A walk runs in pieces: init() grows the scratchpad, run() advances the walk
// any multiple of 8 steps, finish() hashes the result. The GPU does a few
// thousand steps per kernel launch, so no launch comes near a display
// watchdog's limit, and a walk's registers wait in device memory in between.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "blake3_small.h"

#if defined(__CUDACC__)
#define WALK_HD __host__ __device__ __forceinline__
#define WALK_UNROLL _Pragma("unroll")
#define WALK_NO_UNROLL _Pragma("unroll 1")
#else
#define WALK_HD inline
#define WALK_UNROLL
#define WALK_NO_UNROLL
#endif

namespace walk {

// ── Parameters ──────────────────────────────────────────────────────────────

struct Params {
  uint32_t lines;  // 64-byte lines in the scratchpad
  uint32_t steps;  // steps per attempt
  uint32_t remix;  // steps between BLAKE3 re-mixes of the registers
};
// Mainnet: a 256 KiB scratchpad and 2^18 steps.
static const Params MAINNET = {4096, 1u << 18, 1u << 12};
// midstate's fast-mining test build: a 16 KiB scratchpad and 1,024 steps.
static const Params FAST = {256, 1u << 10, 1u << 8};

// ── Helpers ─────────────────────────────────────────────────────────────────

WALK_HD uint64_t join(uint32_t lo, uint32_t hi) { return (uint64_t)lo | ((uint64_t)hi << 32); }

WALK_HD uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

// The high 64 bits of a 64×64-bit product, without 128-bit integers.
WALK_HD uint64_t mulhi64_portable(uint64_t a, uint64_t b) {
  const uint64_t al = (uint32_t)a, ah = a >> 32, bl = (uint32_t)b, bh = b >> 32;
  const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  const uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
  return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

WALK_HD uint64_t mulhi64(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
  return __umul64hi(a, b);
#elif defined(__SIZEOF_INT128__)
  return (uint64_t)(((unsigned __int128)a * b) >> 64);
#else
  return mulhi64_portable(a, b);
#endif
}

// ── AES ─────────────────────────────────────────────────────────────────────

static const uint8_t SBOX[256] = {
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
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

// One AES encryption round exactly as x86's AESENC computes it:
// MixColumns(ShiftRows(SubBytes(state))) xor key. The state is four
// little-endian 32-bit columns. te[x] is the column (2·S(x), S(x), S(x),
// 3·S(x)); rotating it gives the other three rows. `o` may be `s`.
WALK_HD void aesenc(uint32_t o[4], const uint32_t s[4], const uint32_t k[4], const uint32_t* te) {
  const uint32_t s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3];
  o[0] = te[s0 & 0xff] ^ rotl32(te[(s1 >> 8) & 0xff], 8) ^ rotl32(te[(s2 >> 16) & 0xff], 16) ^
         rotl32(te[s3 >> 24], 24) ^ k[0];
  o[1] = te[s1 & 0xff] ^ rotl32(te[(s2 >> 8) & 0xff], 8) ^ rotl32(te[(s3 >> 16) & 0xff], 16) ^
         rotl32(te[s0 >> 24], 24) ^ k[1];
  o[2] = te[s2 & 0xff] ^ rotl32(te[(s3 >> 8) & 0xff], 8) ^ rotl32(te[(s0 >> 16) & 0xff], 16) ^
         rotl32(te[s1 >> 24], 24) ^ k[2];
  o[3] = te[s3 & 0xff] ^ rotl32(te[(s0 >> 8) & 0xff], 8) ^ rotl32(te[(s1 >> 16) & 0xff], 16) ^
         rotl32(te[s2 >> 24], 24) ^ k[3];
}

// The table aesenc() reads, built from the S-box.
inline void build_te(uint32_t te[256]) {
  for (int x = 0; x < 256; ++x) {
    const uint8_t s = SBOX[x], s2 = (uint8_t)((s << 1) ^ ((s & 0x80) ? 0x1b : 0)), s3 = (uint8_t)(s2 ^ s);
    te[x] = (uint32_t)s2 | ((uint32_t)s << 8) | ((uint32_t)s << 16) | ((uint32_t)s3 << 24);
  }
}

inline const uint32_t* host_te() {
  struct Table {
    uint32_t t[256];
    Table() { build_te(t); }
  };
  static const Table table;
  return table.t;
}

// ── The walk ────────────────────────────────────────────────────────────────

// Which operation each register runs: a permutation of 0..7 drawn from z by
// Fisher–Yates with mixed-radix digits, packed 3 bits a register (register s
// at bits 3s..3s+2) so that on a GPU it fits one register.
WALK_HD uint32_t program(uint64_t z) {
  uint32_t p = 0xFAC688u;  // the identity: register i runs operation i
  WALK_UNROLL for (int i = 7; i >= 1; --i) {
    const uint64_t n = (uint64_t)i + 1;
    const uint32_t j = (uint32_t)(z % n);
    z /= n;
    const uint32_t d = ((p >> (3 * i)) ^ (p >> (3 * j))) & 7;  // swap fields i and j
    p ^= (d << (3 * i)) | (d << (3 * j));
  }
  return p;
}

// A walk in progress.
struct State {
  uint64_t r[8];  // the registers
  uint64_t c;     // the previous step's output
  uint32_t prog;  // program()
};

// One step on register S (the step number mod 8). S is a constant, so every
// register index below is fixed and, on a GPU, the registers stay registers.
template <int S, uint32_t LINES>
WALK_HD void step(State& st, uint64_t* __restrict__ pad, const uint32_t* __restrict__ te) {
  constexpr uint64_t MASK = LINES - 1;
  constexpr int T1 = (S + 1) & 7, T2 = (S + 2) & 7, T3 = (S + 3) & 7, T4 = (S + 4) & 7;
  const uint64_t x = st.r[S] ^ st.c;
  const uint32_t a = (uint32_t)(x & MASK);
  const uint32_t d = (uint32_t)(x >> 12) & 3;  // the partner is register S + 1 + d
  const uint32_t j = (uint32_t)(x >> 14) & 7;
  const uint32_t e = (uint32_t)((x >> 17) & MASK);
  const uint32_t b = (uint32_t)((x >> 32) & MASK);
  const uint32_t f = (uint32_t)((x >> 44) & MASK);
  uint64_t* const pa = pad + ((size_t)a << 3) + j;
  uint64_t* const pb = pad + ((size_t)b << 3) + (j ^ 4);
  uint64_t* const pe = pad + ((size_t)e << 3) + (j ^ 2);
  uint64_t* const pf = pad + ((size_t)f << 3) + (j ^ 6);
  const uint64_t m0 = *pa, m1 = *pb, m2 = *pe, m3 = *pf;
  const uint64_t k = d == 0 ? st.r[T1] : d == 1 ? st.r[T2] : d == 2 ? st.r[T3] : st.r[T4];
  const uint64_t v = x ^ m0 ^ m2;
  uint64_t y, hi = 0;
  switch ((st.prog >> (3 * S)) & 7) {
    case 0:
      y = v * (k | 1);
      break;
    case 1:
      y = v ^ mulhi64(v, k);
      break;
    case 2:
      y = v + k / ((uint64_t)(uint32_t)v | 0x80000001ull);
      break;
    case 3: {
      const uint32_t in[4] = {(uint32_t)v, (uint32_t)(v >> 32), (uint32_t)k, (uint32_t)(k >> 32)};
      const uint32_t zero[4] = {0, 0, 0, 0};
      uint32_t o[4];
      aesenc(o, in, zero, te);
      y = join(o[0], o[1]);
      hi = join(o[2], o[3]);
      break;
    }
    case 4: {
      const uint32_t n = (uint32_t)k & 63;
      y = (v >> n) | (v << ((64 - n) & 63));
      break;
    }
    case 5:
      y = v + k;
      break;
    case 6:
      y = v - k;
      break;
    default:
      y = v ^ k;
      break;
  }
  const uint64_t out = y ^ m1 ^ m3;
  st.r[S] = out;
  st.r[T1] ^= d == 0 ? hi : 0;  // the AES operation's high half goes to the partner
  st.r[T2] ^= d == 1 ? hi : 0;
  st.r[T3] ^= d == 2 ? hi : 0;
  st.r[T4] ^= d == 3 ? hi : 0;
  st.c = out;
  *pa = m0 ^ y;
  *pb = m1 + x;
  *pe = m2 + y;
  *pf = m3 ^ x;
}

// Re-keys the registers and draws the next program: derive_key("midstate walk
// v1 remix", registers), 72 bytes. `key` is that context's key.
WALK_HD void remix(State& st, const uint32_t key[8]) {
  const uint32_t flags = b3::DERIVE_KEY_MATERIAL | b3::CHUNK_START | b3::CHUNK_END | b3::ROOT;
  uint32_t m[16], o[16];
  WALK_UNROLL for (int k = 0; k < 8; ++k) {
    m[2 * k] = (uint32_t)st.r[k];
    m[2 * k + 1] = (uint32_t)(st.r[k] >> 32);
  }
  b3::compress(key, m, 0, 64, flags, o);
  WALK_UNROLL for (int k = 0; k < 8; ++k) st.r[k] = join(o[2 * k], o[2 * k + 1]);
  b3::compress(key, m, 1, 64, flags, o);
  st.prog = program(join(o[0], o[1]));
}

// Advances a walk `nsteps` steps from step `first` (both multiples of 8),
// re-mixing after every REMIX-th step.
template <uint32_t LINES, uint32_t REMIX>
WALK_HD void run(State& st, uint64_t* __restrict__ pad, const uint32_t* __restrict__ te,
                 const uint32_t remix_key[8], uint32_t first, uint32_t nsteps) {
  WALK_NO_UNROLL for (uint32_t i = 0; i < nsteps; i += 8) {
    step<0, LINES>(st, pad, te);
    step<1, LINES>(st, pad, te);
    step<2, LINES>(st, pad, te);
    step<3, LINES>(st, pad, te);
    step<4, LINES>(st, pad, te);
    step<5, LINES>(st, pad, te);
    step<6, LINES>(st, pad, te);
    step<7, LINES>(st, pad, te);
    if (((first + i + 8) & (REMIX - 1)) == 0) remix(st, remix_key);
  }
}

// The start of a walk: registers, program and the whole scratchpad, from
// derive_key("midstate walk v1 init", mining_hash ‖ le64(nonce)), 200 bytes.
template <uint32_t LINES>
WALK_HD void init(State& st, uint64_t* __restrict__ pad, const uint32_t* __restrict__ te,
                  const uint32_t key[8], const uint32_t hash[8], uint64_t nonce) {
  const uint32_t flags = b3::DERIVE_KEY_MATERIAL | b3::CHUNK_START | b3::CHUNK_END | b3::ROOT;
  uint32_t m[16], o[16], lanes[16], keys[16];
  WALK_UNROLL for (int i = 0; i < 8; ++i) m[i] = hash[i];
  m[8] = (uint32_t)nonce;
  m[9] = (uint32_t)(nonce >> 32);
  WALK_UNROLL for (int i = 10; i < 16; ++i) m[i] = 0;
  b3::compress(key, m, 0, 40, flags, o);      // seed bytes 0..64: the registers
  WALK_UNROLL for (int k = 0; k < 8; ++k) st.r[k] = join(o[2 * k], o[2 * k + 1]);
  b3::compress(key, m, 1, 40, flags, lanes);  // 64..128: four AES lanes
  b3::compress(key, m, 2, 40, flags, keys);   // 128..192: their keys
  b3::compress(key, m, 3, 40, flags, o);      // 192..200: the program
  st.prog = program(join(o[0], o[1]));
  st.c = 0;
  WALK_NO_UNROLL for (uint32_t n = 0; n < LINES; ++n) {
    uint64_t* const line = pad + (size_t)n * 8;
    WALK_UNROLL for (int q = 0; q < 4; ++q) {
      aesenc(&lanes[4 * q], &lanes[4 * q], &keys[4 * q], te);
      line[2 * q] = join(lanes[4 * q], lanes[4 * q + 1]);
      line[2 * q + 1] = join(lanes[4 * q + 2], lanes[4 * q + 3]);
    }
  }
}

// The proof-of-work hash: derive_key("midstate walk v1 final",
// mining_hash ‖ le64(nonce) ‖ registers), 104 bytes in two blocks.
WALK_HD void finish(const State& st, const uint32_t key[8], const uint32_t hash[8], uint64_t nonce,
                    uint32_t out[8]) {
  uint32_t b0[16], b1[16], o[16], cv[8];
  WALK_UNROLL for (int i = 0; i < 8; ++i) b0[i] = hash[i];
  b0[8] = (uint32_t)nonce;
  b0[9] = (uint32_t)(nonce >> 32);
  WALK_UNROLL for (int k = 0; k < 3; ++k) {
    b0[10 + 2 * k] = (uint32_t)st.r[k];
    b0[11 + 2 * k] = (uint32_t)(st.r[k] >> 32);
  }
  WALK_UNROLL for (int k = 0; k < 5; ++k) {
    b1[2 * k] = (uint32_t)st.r[3 + k];
    b1[2 * k + 1] = (uint32_t)(st.r[3 + k] >> 32);
  }
  WALK_UNROLL for (int i = 10; i < 16; ++i) b1[i] = 0;
  b3::compress(key, b0, 0, 64, b3::DERIVE_KEY_MATERIAL | b3::CHUNK_START, o);
  WALK_UNROLL for (int i = 0; i < 8; ++i) cv[i] = o[i];
  b3::compress(cv, b1, 0, 40, b3::DERIVE_KEY_MATERIAL | b3::CHUNK_END | b3::ROOT, o);
  WALK_UNROLL for (int i = 0; i < 8; ++i) out[i] = o[i];
}

// ── Many walks at once ──────────────────────────────────────────────────────

// derive_key's three context keys, worked out once on the host.
struct Keys {
  uint32_t init[8], remix[8], fin[8];
};

inline Keys context_keys() {
  Keys k;
  b3::context_key("midstate walk v1 init", k.init);
  b3::context_key("midstate walk v1 remix", k.remix);
  b3::context_key("midstate walk v1 final", k.fin);
  return k;
}

// Walks in progress, as a structure of arrays so that neighbouring GPU threads
// read neighbouring words. Walk w's register k is r[k * capacity + w]; its
// scratchpad is the `lines * 8` words from pads + w * lines * 8.
struct Slots {
  uint64_t* r;
  uint64_t* c;
  uint32_t* prog;
  uint64_t* pads;
  uint32_t capacity;
};

// What each walk hashes, and where its result goes. Walk w's mining hash is
// the 8 little-endian words at hashes + w * hash_stride (a stride of 0 shares
// one hash between all walks); its result is the 8 words at out + w * 8.
struct Jobs {
  const uint32_t* hashes;
  uint32_t hash_stride;
  const uint64_t* nonces;
  uint32_t* out;
};

WALK_HD void load_slot(const Slots& s, uint32_t w, State& st) {
  WALK_UNROLL for (int k = 0; k < 8; ++k) st.r[k] = s.r[(size_t)k * s.capacity + w];
  st.c = s.c[w];
  st.prog = s.prog[w];
}

WALK_HD void store_slot(const Slots& s, uint32_t w, const State& st) {
  WALK_UNROLL for (int k = 0; k < 8; ++k) s.r[(size_t)k * s.capacity + w] = st.r[k];
  s.c[w] = st.c;
  s.prog[w] = st.prog;
}

// What one GPU thread does in each kernel, for walk w. The host self-test runs
// these too, one walk at a time.
template <uint32_t LINES>
WALK_HD void init_slot(const Slots& s, const Keys& k, const Jobs& j, uint32_t w, const uint32_t* te) {
  uint32_t key[8], h[8];
  const uint32_t* hp = j.hashes + (size_t)w * j.hash_stride;
  WALK_UNROLL for (int i = 0; i < 8; ++i) {
    key[i] = k.init[i];
    h[i] = hp[i];
  }
  State st;
  init<LINES>(st, s.pads + (size_t)w * LINES * 8, te, key, h, j.nonces[w]);
  store_slot(s, w, st);
}

template <uint32_t LINES, uint32_t REMIX>
WALK_HD void run_slot(const Slots& s, const Keys& k, uint32_t w, uint32_t first, uint32_t nsteps,
                      const uint32_t* te) {
  uint32_t key[8];
  WALK_UNROLL for (int i = 0; i < 8; ++i) key[i] = k.remix[i];
  State st;
  load_slot(s, w, st);
  run<LINES, REMIX>(st, s.pads + (size_t)w * LINES * 8, te, key, first, nsteps);
  store_slot(s, w, st);
}

WALK_HD void finish_slot(const Slots& s, const Keys& k, const Jobs& j, uint32_t w) {
  uint32_t key[8], h[8];
  const uint32_t* hp = j.hashes + (size_t)w * j.hash_stride;
  WALK_UNROLL for (int i = 0; i < 8; ++i) {
    key[i] = k.fin[i];
    h[i] = hp[i];
  }
  State st;
  load_slot(s, w, st);
  finish(st, key, h, j.nonces[w], j.out + (size_t)w * 8);
}

// ── On the host ─────────────────────────────────────────────────────────────

template <uint32_t LINES, uint32_t REMIX>
inline void hash_words(const Keys& k, const uint32_t h[8], uint64_t nonce, uint32_t steps, uint64_t* pad,
                       uint32_t out[8]) {
  State st;
  init<LINES>(st, pad, host_te(), k.init, h, nonce);
  run<LINES, REMIX>(st, pad, host_te(), k.remix, 0, steps);
  finish(st, k.fin, h, nonce, out);
}

// The walk hash of `nonce` over `mining_hash`, on this CPU. `pad` holds
// p.lines * 8 words. False for parameters other than MAINNET and FAST.
inline bool hash_host(const Params& p, const Keys& k, const uint8_t mining_hash[32], uint64_t nonce,
                      uint64_t* pad, uint8_t out[32]) {
  uint32_t h[8], o[8];
  for (int i = 0; i < 8; ++i) h[i] = b3::load32(mining_hash + 4 * i);
  if (p.lines == MAINNET.lines && p.steps == MAINNET.steps && p.remix == MAINNET.remix)
    hash_words<4096, 4096>(k, h, nonce, p.steps, pad, o);
  else if (p.lines == FAST.lines && p.steps == FAST.steps && p.remix == FAST.remix)
    hash_words<256, 256>(k, h, nonce, p.steps, pad, o);
  else
    return false;
  for (int i = 0; i < 8; ++i) b3::store32(out + 4 * i, o[i]);
  return true;
}

// Known answers, from midstate's src/core/walk.rs (GOLDEN): mining hash,
// nonce, walk hash. Changing any of these there is a hard fork, and
// ci/check_golden.py fails until these match again.
struct Known {
  const char* hash;
  uint64_t nonce;
  const char* want;
};

static const Known GOLDEN_MAINNET[3] = {
    {"0000000000000000000000000000000000000000000000000000000000000000", 0,
     "a934fced2d9a53fb09d6abe089f3d2f5bba9a421b41ee5726233a525cc7438e7"},
    {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 18446744073709551615ull,
     "65d237f595f07bee156c635ad45f721987ce38d79ea14c89e40c5555613d0768"},
    {"374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50", 42,
     "5f2da9d320de391aef28bcaab63e3d1a25402b4541cdff8680059a87c7d515b7"},
};

static const Known GOLDEN_FAST[3] = {
    {"0000000000000000000000000000000000000000000000000000000000000000", 0,
     "592cd5f394f7dc705343585a7cb48c58ce8da5f646dadb81af440e0d7a470e40"},
    {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 18446744073709551615ull,
     "f1488d17741b4eaaa909374c90ea70adfe89ceaa16d7ca8605c0da25f031e9f3"},
    {"374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50", 42,
     "8f9d7f1d42826a15f81dbee9fd7e25ca3edee11776dfb48e078feebfeab2ab13"},
};

inline const Known* golden(const Params& p) { return p.lines == FAST.lines ? GOLDEN_FAST : GOLDEN_MAINNET; }

// Exactly 2n hex digits into n bytes.
inline bool from_hex(const char* s, uint8_t* out, size_t n) {
  for (size_t i = 0; i < 2 * n; ++i)
    if (!s[i]) return false;
  if (s[2 * n]) return false;
  for (size_t i = 0; i < n; ++i) {
    int v = 0;
    for (int h = 0; h < 2; ++h) {
      const char c = s[2 * i + h];
      const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (d < 0) return false;
      v = v * 16 + d;
    }
    out[i] = (uint8_t)v;
  }
  return true;
}

// n bytes as 2n hex digits and a terminating NUL.
inline void to_hex(const uint8_t* p, size_t n, char* out) {
  static const char* const digits = "0123456789abcdef";
  for (size_t i = 0; i < n; ++i) {
    out[2 * i] = digits[p[i] >> 4];
    out[2 * i + 1] = digits[p[i] & 15];
  }
  out[2 * n] = 0;
}

}  // namespace walk
