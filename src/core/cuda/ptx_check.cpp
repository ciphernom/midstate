// CI: runs walk_miner.ptx itself, as compiled, on the CPU against the walk's
// known answers. ptx_to_c.py translates the PTX into C with PTX's rules; this
// launches those kernels as core/cuda_walk.rs launches them on a GPU (the
// same buffers and sizes, the steps in launches of varying length that cross
// the re-mix points, the result as little-endian words).
//
// host_check.cpp tests walk_miner.cu's source; this tests what clang made of
// it. It would have caught the PTX that clang 18 built with a broken 64-bit
// rotate, which got every walk wrong on a Quadro M6000: run on it, that PTX
// gives the same wrong hash the card did. What it cannot check is the GPU's
// driver compiling the PTX for the card; the miner's self-test does that.
//
//   python3 ptx_to_c.py walk_miner.ptx > walk_miner_ptx.c
//   gcc -O1 -c walk_miner_ptx.c -o walk_miner_ptx.o
//   g++ -O2 -std=c++17 -Wall ptx_check.cpp walk_miner_ptx.o -o ptx_check
//   ./ptx_check
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../../docs/walk-reference/cuda/walk_core.h"

extern "C" {
extern uint8_t ms_te[1024], ms_keys[96];
uint32_t tid_x, ntid_x, ctaid_x;
void ms_init_main(const uint64_t*);
void ms_steps_main(const uint64_t*);
void ms_init_fast(const uint64_t*);
void ms_steps_fast(const uint64_t*);
void ms_final(const uint64_t*);
}

static uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

// As cuLaunchKernel: ceil(walks / block) blocks of `block` threads. Threads
// run one after another, so a block's shared AES table is whole only once a
// thread has filled all of it: blocks of one thread first (as a barrier
// would), then blocks of 256, which find each kernel's table filled and run
// the kernels' bounds on the threads past the last walk.
static uint32_t g_block = 1;
static void launch(void (*kernel)(const uint64_t*), uint32_t walks, const std::vector<uint64_t>& args) {
  ntid_x = g_block;
  for (ctaid_x = 0; ctaid_x < (walks + g_block - 1) / g_block; ++ctaid_x)
    for (tid_x = 0; tid_x < g_block; ++tid_x) kernel(args.data());
}

// Steps per launch: multiples of 8, as the miner's are, of many sizes, so
// launches start and end on and off the re-mix points.
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t next_chunk() {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  static const uint32_t sizes[] = {8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 24, 1000};
  return sizes[rng % 12];
}

// The walk hashes of `nonces` over `mh`, through the PTX's kernels.
template <uint32_t LINES, uint32_t STEPS, bool MAIN>
std::vector<std::string> run(const uint8_t mh[32], const std::vector<uint64_t>& nonces, uint32_t capacity) {
  const size_t n = capacity;
  // core/cuda_walk.rs: [n*64, n*8, n*4, n*pad_bytes, 32, n*8, n*32] bytes.
  std::vector<uint64_t> r(n * 8), c(n), pads(n * LINES * 8), nb(n);
  std::vector<uint32_t> prog(n), hash(8), out(n * 8);
  memcpy(hash.data(), mh, 32);
  memcpy(nb.data(), nonces.data(), nonces.size() * 8);
  const uint32_t walks = nonces.size();
  const uint64_t slots[5] = {ptr(r.data()), ptr(c.data()), ptr(prog.data()), ptr(pads.data()), capacity};
  launch(MAIN ? ms_init_main : ms_init_fast, walks,
         {slots[0], slots[1], slots[2], slots[3], slots[4], ptr(hash.data()), ptr(nb.data()), walks});
  for (uint32_t first = 0; first < STEPS;) {
    const uint32_t want = next_chunk();
    const uint32_t k = want < STEPS - first ? want : STEPS - first;
    launch(MAIN ? ms_steps_main : ms_steps_fast, walks,
           {slots[0], slots[1], slots[2], slots[3], slots[4], walks, first, k});
    first += k;
  }
  launch(ms_final, walks,
         {slots[0], slots[1], slots[2], slots[3], slots[4], ptr(hash.data()), ptr(nb.data()), ptr(out.data()), walks});
  std::vector<std::string> res;
  for (uint32_t w = 0; w < walks; ++w) {
    uint8_t h[32];
    for (int i = 0; i < 8; ++i)
      for (int b = 0; b < 4; ++b) h[i * 4 + b] = (out[w * 8 + i] >> (8 * b)) & 0xff;
    char s[65];
    walk::to_hex(h, 32, s);
    res.push_back(s);
  }
  return res;
}

// The CPU's walk hash, from walk_core.h (which CI checks against walk.rs).
static std::string reference(const walk::Params& p, const uint8_t mh[32], uint64_t nonce) {
  std::vector<uint64_t> pad(p.lines * 8);
  uint8_t h[32];
  walk::hash_host(p, walk::context_keys(), mh, nonce, pad.data(), h);
  char s[65];
  walk::to_hex(h, 32, s);
  return s;
}

template <uint32_t LINES, uint32_t STEPS, bool MAIN>
int checks(const walk::Params& p, const char* name, uint32_t batch) {
  int fails = 0;
  const walk::Known* known = walk::golden(p);
  for (int i = 0; i < 3; ++i) {
    uint8_t mh[32];
    walk::from_hex(known[i].hash, mh, 32);
    const std::string got = run<LINES, STEPS, MAIN>(mh, {known[i].nonce}, 256)[0];
    const bool ok = got == known[i].want;
    printf("%s  nonce %-20llu %s\n", name, (unsigned long long)known[i].nonce, ok ? "ok" : ("WRONG " + got).c_str());
    fails += !ok;
  }
  // A batch of more than one block, not a multiple of 256, with a known
  // answer in it: each walk gets its own hash, so none reads another's memory.
  uint8_t mh[32];
  walk::from_hex(known[2].hash, mh, 32);
  std::vector<uint64_t> ns;
  for (uint64_t i = 0; i < batch; ++i) ns.push_back((1ull << 63) | (3ull << 56) | (i * 7919));
  ns[5] = known[2].nonce;
  const auto got = run<LINES, STEPS, MAIN>(mh, ns, 512);
  int bad = got[5] != known[2].want;
  for (size_t i = 0; i < ns.size(); i += 37) bad += got[i] != reference(p, mh, ns[i]);
  printf("%s  batch of %u walks: %s\n", name, batch, bad ? "WRONG" : "ok");
  return fails + (bad != 0);
}

int main() {
  // The kernels' constants, as core/cuda_walk.rs uploads them (it pins its
  // copies to walk_core's).
  uint32_t te[256];
  walk::build_te(te);
  memcpy(ms_te, te, sizeof te);
  const walk::Keys k = walk::context_keys();
  memcpy(ms_keys, k.init, 32);
  memcpy(ms_keys + 32, k.remix, 32);
  memcpy(ms_keys + 64, k.fin, 32);
  int fails = 0;
  for (uint32_t b : {1u, 256u}) {
    g_block = b;
    printf("-- blocks of %u threads\n", b);
    fails += checks<256, 1u << 10, false>(walk::FAST, "fast", 300);
    fails += checks<4096, 1u << 18, true>(walk::MAINNET, "main", 260);
  }
  printf("%s\n", fails ? "FAILED: the PTX does not compute the walk" : "the PTX gives every known answer");
  return fails != 0;
}
