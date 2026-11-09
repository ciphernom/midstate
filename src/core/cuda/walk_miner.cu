// The kernels midstate mines Midstate Walk with on NVIDIA GPUs
// (core/cuda_walk.rs). They are the benchmark's walk, unchanged: walk_core.h
// in docs/walk-reference/cuda, whose CPU self-test and known answers CI checks
// against src/core/walk.rs. This file only gives its slot functions extern "C"
// names and flat arguments, so the miner can load them from PTX through the
// CUDA driver API, with no CUDA toolkit at build time.
//
// walk_miner.ptx is this file compiled by gen_ptx.sh (clang, no toolkit). CI
// rebuilds it and fails if the committed copy differs, so the PTX the miner
// embeds is always this source. CI also runs these kernels on the CPU,
// launched as the miner launches them, against the known answers: from this
// source (host_check.cpp, which defines WALK_MINER_HOST_CHECK), and from the
// PTX itself (ptx_check.cpp), which is what caught clang miscompiling the
// walk's 64-bit rotate.
//
// One thread per walk, as in the benchmark. For walk w:
//   init:   seed registers and scratchpad from (mining hash, nonces[w])
//   steps:  steps [first, first + nsteps) of the walk (many launches, so no
//           launch runs long enough to trip a display watchdog, and a stale
//           job stops within one launch)
//   final:  the final hash, as 8 little-endian words at out + w * 8

#ifndef WALK_MINER_HOST_CHECK
#define __CUDACC__ 1  // nvcc defines it; clang only through the toolkit's headers
#include <__clang_cuda_builtin_vars.h>
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __syncthreads() __nvvm_bar_sync(0)
#define __umul64hi(a, b) __nvvm_mulhi_ull(a, b)
#endif

#include "../../../docs/walk-reference/cuda/walk_core.h"

// Set by the host after loading the module: the AES table (build_te) and
// derive_key's three context keys (context_keys), 8 words each.
extern "C" {
__constant__ uint32_t ms_te[256];
__constant__ uint32_t ms_keys[24];
}

namespace {

__device__ __forceinline__ void load_te(uint32_t* te) {
  for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) te[i] = ms_te[i];
  __syncthreads();
}

__device__ __forceinline__ walk::Keys keys() {
  walk::Keys k;
  for (int i = 0; i < 8; ++i) {
    k.init[i] = ms_keys[i];
    k.remix[i] = ms_keys[8 + i];
    k.fin[i] = ms_keys[16 + i];
  }
  return k;
}

__device__ __forceinline__ walk::Slots slots(uint64_t* r, uint64_t* c, uint32_t* prog, uint64_t* pads,
                                             uint32_t capacity) {
  walk::Slots s;
  s.r = r;
  s.c = c;
  s.prog = prog;
  s.pads = pads;
  s.capacity = capacity;
  return s;
}

__device__ __forceinline__ walk::Jobs jobs(const uint32_t* hash, const uint64_t* nonces, uint32_t* out) {
  walk::Jobs j;
  j.hashes = hash;
  j.hash_stride = 0;  // one mining hash for every walk
  j.nonces = nonces;
  j.out = out;
  return j;
}

template <uint32_t LINES>
__device__ __forceinline__ void init(uint64_t* r, uint64_t* c, uint32_t* prog, uint64_t* pads, uint32_t capacity,
                                     const uint32_t* hash, const uint64_t* nonces, uint32_t walks) {
  __shared__ uint32_t te[256];
  load_te(te);
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::init_slot<LINES>(slots(r, c, prog, pads, capacity), keys(), jobs(hash, nonces, nullptr), w, te);
}

template <uint32_t LINES, uint32_t REMIX>
__device__ __forceinline__ void steps(uint64_t* r, uint64_t* c, uint32_t* prog, uint64_t* pads, uint32_t capacity,
                                      uint32_t walks, uint32_t first, uint32_t nsteps) {
  __shared__ uint32_t te[256];
  load_te(te);
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::run_slot<LINES, REMIX>(slots(r, c, prog, pads, capacity), keys(), w, first, nsteps, te);
}

}  // namespace

#define MS_ARGS uint64_t *r, uint64_t *c, uint32_t *prog, uint64_t *pads, uint32_t capacity

extern "C" {

// Mainnet: a 256 KiB scratchpad, 2^18 steps, a re-mix every 2^12.
__global__ void __launch_bounds__(256) ms_init_main(MS_ARGS, const uint32_t* hash, const uint64_t* nonces,
                                                    uint32_t walks) {
  init<4096>(r, c, prog, pads, capacity, hash, nonces, walks);
}
__global__ void __launch_bounds__(256) ms_steps_main(MS_ARGS, uint32_t walks, uint32_t first, uint32_t nsteps) {
  steps<4096, 4096>(r, c, prog, pads, capacity, walks, first, nsteps);
}

// midstate's fast-mining test build: 16 KiB, 2^10 steps, a re-mix every 2^8.
__global__ void __launch_bounds__(256) ms_init_fast(MS_ARGS, const uint32_t* hash, const uint64_t* nonces,
                                                    uint32_t walks) {
  init<256>(r, c, prog, pads, capacity, hash, nonces, walks);
}
__global__ void __launch_bounds__(256) ms_steps_fast(MS_ARGS, uint32_t walks, uint32_t first, uint32_t nsteps) {
  steps<256, 256>(r, c, prog, pads, capacity, walks, first, nsteps);
}

__global__ void __launch_bounds__(256) ms_final(MS_ARGS, const uint32_t* hash, const uint64_t* nonces, uint32_t* out,
                                                uint32_t walks) {
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::finish_slot(slots(r, c, prog, pads, capacity), keys(), jobs(hash, nonces, out), w);
}

}  // extern "C"
