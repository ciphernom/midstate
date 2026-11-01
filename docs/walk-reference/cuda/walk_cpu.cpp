// walk_cpu.cpp: walk-cpu, the CPU reference and self-test.
//
//   walk-cpu --selftest                         check everything the GPU code relies on
//   walk-cpu [--fast] [--hash HEX] [--nonce N]  one walk hash, and how long it took
//
// This is the portable code with a table AES, the same code the GPU runs: the
// reference, not a fast CPU miner. `midstate bench-pow` measures a CPU.

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "walk_core.h"

namespace {

int failures = 0;

void report(const char* what, bool ok, const std::string& detail = "") {
  printf("  %-52s %s%s%s\n", what, ok ? "PASS" : "FAIL", detail.empty() ? "" : "  ", detail.c_str());
  if (!ok) ++failures;
}

std::string hex(const uint8_t* p, size_t n) {
  std::vector<char> s(2 * n + 1);
  walk::to_hex(p, n, s.data());
  return s.data();
}

std::vector<uint8_t> pattern(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (uint8_t)(i % 251);
  return v;
}

uint64_t xorshift(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

// The program as the specification writes it: an array and swaps.
uint32_t program_reference(uint64_t z) {
  uint8_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  for (int i = 7; i >= 1; --i) {
    const uint64_t n = (uint64_t)i + 1;
    const int j = (int)(z % n);
    const uint8_t t = p[i];
    p[i] = p[j];
    p[j] = t;
    z /= n;
  }
  uint32_t packed = 0;
  for (int i = 0; i < 8; ++i) packed |= (uint32_t)p[i] << (3 * i);
  return packed;
}

void test_aes() {
  // FIPS-197 appendix B, round 1: the state after the first AddRoundKey, and
  // the round-1 key.
  const uint8_t in[16] = {0x19, 0x3d, 0xe3, 0xbe, 0xa0, 0xf4, 0xe2, 0x2b,
                          0x9a, 0xc6, 0x8d, 0x2a, 0xe9, 0xf8, 0x48, 0x08};
  const uint8_t key[16] = {0xa0, 0xfa, 0xfe, 0x17, 0x88, 0x54, 0x2c, 0xb1,
                           0x23, 0xa3, 0x39, 0x39, 0x2a, 0x6c, 0x76, 0x05};
  const uint8_t want[16] = {0xa4, 0x9c, 0x7f, 0xf2, 0x68, 0x9f, 0x35, 0x2b,
                            0x6b, 0x5b, 0xea, 0x43, 0x02, 0x6a, 0x50, 0x49};
  uint32_t s[4], k[4], o[4];
  uint8_t got[16];
  for (int i = 0; i < 4; ++i) {
    s[i] = b3::load32(in + 4 * i);
    k[i] = b3::load32(key + 4 * i);
  }
  walk::aesenc(o, s, k, walk::host_te());
  for (int i = 0; i < 4; ++i) b3::store32(got + 4 * i, o[i]);
  report("AES round (FIPS-197 appendix B)", memcmp(got, want, 16) == 0, hex(got, 16));
}

void test_blake3() {
  // From the blake3 crate; inputs are the official test vectors' i % 251.
  struct {
    size_t len;
    const char* want;
  } const hashes[] = {
      {0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
      {64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
      {65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"},
      {1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
  };
  bool ok = true;
  for (const auto& h : hashes) {
    uint8_t out[32];
    const std::vector<uint8_t> in = pattern(h.len);
    b3::hash(in.data(), in.size(), out);
    ok &= hex(out, 32) == h.want;
  }
  report("BLAKE3 hash (0, 64, 65 and 1,024 bytes)", ok);

  const char* ctx = "BLAKE3 2019-12-27 16:29:52 test vectors context";
  uint8_t short_out[32], long_out[200];
  const std::vector<uint8_t> in = pattern(104);
  b3::derive_key(ctx, nullptr, 0, short_out, 32);
  b3::derive_key(ctx, in.data(), in.size(), long_out, 200);
  report("BLAKE3 derive_key (0 bytes, and 104 bytes to 200)",
         hex(short_out, 32) == "2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d" &&
             hex(long_out, 200) ==
                 "8dcd945721896dc64ef8fa3bdbb50ae3eb5664f75b9ccb17fcd041a1259a7811b2b5598e7ff68533a477c817809f28a8"
                 "67ebd2f624b5b9ec02ad924bf15d9c9f35a7bcc3b8d41372101d45f6330fcb89363af652e7ace35e792c9b2bef5fe3e0"
                 "fa526cbe0c085c338ab759b520c40f6e5b580b582e39c30ab6eacea3bb8f0bee9e422e1d9f9f69d1ca7258bc0a2e28b3"
                 "a00ca8ddcb3f99b581fa95bdc50291ad7a39991346321d104f9fc23fedcd4c89ccfb02a20fbb821e2ab118311efc3d5d"
                 "07ea3fa4dd47d655");
}

void test_arithmetic() {
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  bool program_ok = true, mulhi_ok = true;
  for (int i = 0; i < 200000; ++i) {
    const uint64_t z = xorshift(seed);
    program_ok &= walk::program(z) == program_reference(z);
    const uint64_t a = xorshift(seed), b = i % 7 == 0 ? ~0ull : xorshift(seed);
    mulhi_ok &= walk::mulhi64_portable(a, b) == walk::mulhi64(a, b);
  }
  program_ok &= walk::program(0) == program_reference(0) && walk::program(~0ull) == program_reference(~0ull);
  report("program() against the array version", program_ok);
  report("64x64 high multiply, portable against native", mulhi_ok);
}

void test_golden(const char* name, const walk::Params& p) {
  const walk::Keys keys = walk::context_keys();
  std::vector<uint64_t> pad((size_t)p.lines * 8);
  const walk::Known* g = walk::golden(p);
  int right = 0;
  for (int i = 0; i < 3; ++i) {
    uint8_t mh[32], want[32], got[32];
    walk::from_hex(g[i].hash, mh, 32);
    walk::from_hex(g[i].want, want, 32);
    walk::hash_host(p, keys, mh, g[i].nonce, pad.data(), got);
    right += memcmp(got, want, 32) == 0;
  }
  report(name, right == 3, std::to_string(right) + " of 3");
}

// The GPU's path, emulated: walks in structure-of-arrays slots, advanced a
// few steps at a time by init_slot / run_slot / finish_slot, as the kernels
// do, then compared with hash_host.
template <uint32_t LINES, uint32_t REMIX>
bool slots_match(const walk::Params& p, uint32_t walks, uint32_t chunk, bool shared_hash) {
  const walk::Keys keys = walk::context_keys();
  std::vector<uint64_t> r((size_t)8 * walks), c(walks), pads((size_t)walks * LINES * 8), nonces(walks);
  std::vector<uint32_t> prog(walks), hashes((size_t)8 * walks), out((size_t)8 * walks);
  std::vector<uint8_t> hash_bytes((size_t)32 * walks);
  uint64_t seed = 0x2545f4914f6cdd1dull + walks + chunk;
  for (uint32_t w = 0; w < walks; ++w) {
    nonces[w] = w % 3 == 0 ? ~0ull - w : xorshift(seed);
    for (int i = 0; i < 32; ++i) hash_bytes[(size_t)32 * w + i] = shared_hash ? (uint8_t)(i * 7) : (uint8_t)xorshift(seed);
    for (int i = 0; i < 8; ++i) hashes[(size_t)8 * w + i] = b3::load32(&hash_bytes[(size_t)32 * w + 4 * i]);
  }
  const walk::Slots s{r.data(), c.data(), prog.data(), pads.data(), walks};
  const walk::Jobs j{hashes.data(), shared_hash ? 0u : 8u, nonces.data(), out.data()};
  const uint32_t* te = walk::host_te();
  for (uint32_t w = 0; w < walks; ++w) walk::init_slot<LINES>(s, keys, j, w, te);
  for (uint32_t first = 0; first < p.steps; first += chunk)
    for (uint32_t w = 0; w < walks; ++w) walk::run_slot<LINES, REMIX>(s, keys, w, first, chunk, te);
  for (uint32_t w = 0; w < walks; ++w) walk::finish_slot(s, keys, j, w);
  std::vector<uint64_t> pad((size_t)LINES * 8);
  for (uint32_t w = 0; w < walks; ++w) {
    uint8_t want[32], got[32];
    walk::hash_host(p, keys, &hash_bytes[shared_hash ? 0 : (size_t)32 * w], nonces[w], pad.data(), want);
    for (int i = 0; i < 8; ++i) b3::store32(got + 4 * i, out[(size_t)8 * w + i]);
    if (memcmp(want, got, 32) != 0) return false;
  }
  return true;
}

int selftest() {
  printf("walk-cpu self-test\n");
  test_aes();
  test_blake3();
  test_arithmetic();
  test_golden("known answers, mainnet (midstate's GOLDEN)", walk::MAINNET);
  test_golden("known answers, fast-mining test build", walk::FAST);
  report("GPU slot path, fast: 37 walks, 8 to 256 steps a go",
         slots_match<256, 256>(walk::FAST, 37, 8, false) && slots_match<256, 256>(walk::FAST, 37, 64, true) &&
             slots_match<256, 256>(walk::FAST, 37, 256, false));
  report("GPU slot path, mainnet: 3 walks, 4,096 steps a go", slots_match<4096, 4096>(walk::MAINNET, 3, 4096, false));
  printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}

void usage(FILE* f) {
  fputs("usage: walk-cpu --selftest\n       walk-cpu [--fast] [--hash HEX] [--nonce N]\n", f);
}

}  // namespace

int main(int argc, char** argv) {
  bool fast = false;
  uint8_t hash[32] = {0};
  uint64_t nonce = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--selftest")) {
      return selftest();
    } else if (!strcmp(argv[i], "--fast")) {
      fast = true;
    } else if (!strcmp(argv[i], "--hash") && i + 1 < argc) {
      if (!walk::from_hex(argv[++i], hash, 32)) {
        fprintf(stderr, "walk-cpu: --hash wants 64 hex digits\n");
        return 2;
      }
    } else if (!strcmp(argv[i], "--nonce") && i + 1 < argc) {
      char* end = nullptr;
      nonce = strtoull(argv[++i], &end, 10);
      if (*end) {
        fprintf(stderr, "walk-cpu: --nonce wants a number\n");
        return 2;
      }
    } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      usage(stdout);
      return 0;
    } else {
      fprintf(stderr, "walk-cpu: unknown argument %s\n", argv[i]);
      usage(stderr);
      return 2;
    }
  }
  const walk::Params& p = fast ? walk::FAST : walk::MAINNET;
  const walk::Keys keys = walk::context_keys();
  std::vector<uint64_t> pad((size_t)p.lines * 8);
  uint8_t out[32];
  const auto t = std::chrono::steady_clock::now();
  walk::hash_host(p, keys, hash, nonce, pad.data(), out);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
  printf("%s\n", hex(out, 32).c_str());
  printf("%.1f ms on one core (portable reference code; midstate bench-pow measures a CPU properly)\n", ms);
  return 0;
}
