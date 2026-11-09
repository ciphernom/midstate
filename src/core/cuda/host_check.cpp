// CI: runs walk_miner.cu's five kernels on the CPU, launched as
// core/cuda_walk.rs launches them on a GPU (the same buffers and sizes, the
// steps in launches of varying length that cross the re-mix points, the
// result as little-endian words), and checks they reproduce the walk's known
// answers (walk.rs's GOLDEN). Built with the address and undefined-behaviour
// sanitizers, so a kernel reading or writing outside its walk's memory
// fails it too. It cannot check what clang makes of the source:
// ptx_check.cpp runs the PTX itself. Nor the GPU's driver: each GPU runs the
// same known answers when the miner opens it.
//
//   g++ -O2 -std=c++17 -fsanitize=address,undefined -fno-sanitize-recover=undefined host_check.cpp -o host_check
//   ./host_check
#define WALK_MINER_HOST_CHECK 1
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct Dim { uint32_t x, y, z; };
static Dim threadIdx{0, 0, 0}, blockIdx{0, 0, 0}, blockDim{1, 1, 1};
#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __constant__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __syncthreads() ((void)0)
#include "walk_miner.cu"

template <class F> void launch(uint32_t walks, uint32_t block, F f) {
  // As cuLaunchKernel: ceil(walks / block) blocks of `block` threads; every
  // thread runs, the kernel's own bound stops the extra ones.
  uint32_t grid = (walks + block - 1) / block;
  blockDim = {block, 1, 1};
  for (uint32_t b = 0; b < grid; ++b)
    for (uint32_t t = 0; t < block; ++t) {
      blockIdx = {b, 0, 0};
      threadIdx = {t, 0, 0};
      f();
    }
}

// Threads run one after another here, so a block's shared table is whole
// only once a thread has filled all of it: blocks of one thread first (as a
// barrier would), then blocks of 256, which find the table already filled
// and exercise the kernels' bounds on the threads past the last walk.
static uint32_t g_block = 1;
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t next_chunk() {
  rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
  static const uint32_t sizes[] = {8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 24, 1000};
  return sizes[rng % 12];
}

template <uint32_t LINES, uint32_t STEPS, bool MAIN>
std::vector<std::string> run(const uint8_t mh[32], const std::vector<uint64_t>& nonces, uint32_t capacity) {
  size_t n = capacity;
  // core/cuda_walk.rs: [n*64, n*8, n*4, n*pad_bytes, 32, n*8, n*32] bytes.
  std::vector<uint64_t> r(n * 8), c(n), pads(n * LINES * 8), nb(n);
  std::vector<uint32_t> prog(n), hash(8), out(n * 8);
  memcpy(hash.data(), mh, 32);
  memcpy(nb.data(), nonces.data(), nonces.size() * 8);
  uint32_t walks = nonces.size();
  const uint32_t BLOCK = g_block;
  launch(walks, BLOCK, [&] {
    if (MAIN) ms_init_main(r.data(), c.data(), prog.data(), pads.data(), capacity, hash.data(), nb.data(), walks);
    else ms_init_fast(r.data(), c.data(), prog.data(), pads.data(), capacity, hash.data(), nb.data(), walks);
  });
  uint32_t first = 0;
  while (first < STEPS) {
    uint32_t want = next_chunk();  // multiples of 8, as the host's are
    uint32_t k = want < STEPS - first ? want : STEPS - first;
    launch(walks, BLOCK, [&] {
      if (MAIN) ms_steps_main(r.data(), c.data(), prog.data(), pads.data(), capacity, walks, first, k);
      else ms_steps_fast(r.data(), c.data(), prog.data(), pads.data(), capacity, walks, first, k);
    });
    first += k;
  }
  launch(walks, BLOCK, [&] {
    ms_final(r.data(), c.data(), prog.data(), pads.data(), capacity, hash.data(), nb.data(), out.data(), walks);
  });
  std::vector<std::string> res;
  for (uint32_t w = 0; w < walks; ++w) {
    uint8_t h[32];
    for (int i = 0; i < 8; ++i) for (int b = 0; b < 4; ++b) h[i * 4 + b] = (out[w * 8 + i] >> (8 * b)) & 0xff;
    char s[65];
    for (int i = 0; i < 32; ++i) sprintf(s + 2 * i, "%02x", h[i]);
    res.push_back(s);
  }
  return res;
}

static void unhex(const char* s, uint8_t* o) { for (int i = 0; i < 32; ++i) sscanf(s + 2 * i, "%2hhx", &o[i]); }

struct G { const char* h; uint64_t n; const char* want; };

int checks() {
  // The host's constants: walk_core's own (cuda_walk.rs pins its copies to these).
  walk::build_te(ms_te);
  walk::Keys k = walk::context_keys();
  memcpy(ms_keys, k.init, 32); memcpy(ms_keys + 8, k.remix, 32); memcpy(ms_keys + 16, k.fin, 32);
  int fails = 0;
  G main_g[] = {
    {"0000000000000000000000000000000000000000000000000000000000000000", 0, "a934fced2d9a53fb09d6abe089f3d2f5bba9a421b41ee5726233a525cc7438e7"},
    {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", UINT64_MAX, "65d237f595f07bee156c635ad45f721987ce38d79ea14c89e40c5555613d0768"},
    {"374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50", 42, "5f2da9d320de391aef28bcaab63e3d1a25402b4541cdff8680059a87c7d515b7"},
  };
  G fast_g[] = {
    {"0000000000000000000000000000000000000000000000000000000000000000", 0, "592cd5f394f7dc705343585a7cb48c58ce8da5f646dadb81af440e0d7a470e40"},
    {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", UINT64_MAX, "f1488d17741b4eaaa909374c90ea70adfe89ceaa16d7ca8605c0da25f031e9f3"},
    {"374bf98b3b8f7f41a9e291154fc582a65bb3e14180cb9a07fef08e55f3006c50", 42, "8f9d7f1d42826a15f81dbee9fd7e25ca3edee11776dfb48e078feebfeab2ab13"},
  };
  for (auto& g : main_g) {
    uint8_t mh[32]; unhex(g.h, mh);
    auto got = run<4096, 1u << 18, true>(mh, {g.n}, 256)[0];
    printf("main  nonce %-20llu %s\n", (unsigned long long)g.n, got == g.want ? "ok" : ("WRONG " + got).c_str());
    fails += got != g.want;
  }
  for (auto& g : fast_g) {
    uint8_t mh[32]; unhex(g.h, mh);
    auto got = run<256, 1u << 10, false>(mh, {g.n}, 256)[0];
    printf("fast  nonce %-20llu %s\n", (unsigned long long)g.n, got == g.want ? "ok" : ("WRONG " + got).c_str());
    fails += got != g.want;
  }
  // A batch of many walks (more than one block, not a multiple of 256) gives
  // each walk the hash it gets alone: no walk reads another's memory.
  uint8_t mh[32]; unhex(fast_g[2].h, mh);
  std::vector<uint64_t> ns;
  for (uint64_t i = 0; i < 300; ++i) ns.push_back((1ull << 63) | (3ull << 56) | (i * 7919));
  ns[5] = 42;  // the known answer, in the middle of a batch
  auto batch = run<256, 1u << 10, false>(mh, ns, 512);
  int bad = 0;
  for (size_t i = 0; i < ns.size(); i += 37) {
    auto alone = run<256, 1u << 10, false>(mh, {ns[i]}, 256)[0];
    bad += alone != batch[i];
  }
  bad += batch[5] != fast_g[2].want;
  printf("batch of %zu walks: %s\n", ns.size(), bad ? "WRONG" : "ok");
  fails += bad;
  // Mainnet: a batch of 3 across 2 remix-aligned and unaligned chunkings.
  unhex(main_g[2].h, mh);
  auto b2 = run<4096, 1u << 18, true>(mh, {7, 42, 9}, 256);
  printf("main batch: %s\n", b2[1] == main_g[2].want ? "ok" : "WRONG");
  fails += b2[1] != main_g[2].want;
  return fails;
}

int main() {
  int fails = 0;
  for (uint32_t b : {1u, 256u}) {
    g_block = b;
    printf("-- blocks of %u threads\n", b);
    fails += checks();
  }
  printf("%s\n", fails ? "FAILED" : "all known answers reproduced");
  return fails != 0;
}
