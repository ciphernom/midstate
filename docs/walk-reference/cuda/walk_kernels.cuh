// walk_kernels.cuh: the CUDA kernels. One thread per walk.
//
// The work itself is walk_core.h's init_slot, run_slot and finish_slot, the
// same code the CPU self-test runs; these kernels only add the thread index
// and the shared-memory AES table.
#pragma once
#include "walk_core.h"

namespace walkgpu {

// The AES table. Each block copies it to shared memory: a warp's lanes look
// up different entries, which constant memory would serve one at a time.
__constant__ uint32_t c_te[256];

__device__ __forceinline__ void load_te(uint32_t* te) {
  for (uint32_t i = threadIdx.x; i < 256; i += blockDim.x) te[i] = c_te[i];
  __syncthreads();
}

template <uint32_t LINES>
__global__ void __launch_bounds__(256) init_kernel(walk::Slots s, walk::Keys k, walk::Jobs j, uint32_t walks) {
  __shared__ uint32_t te[256];
  load_te(te);
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::init_slot<LINES>(s, k, j, w, te);
}

template <uint32_t LINES, uint32_t REMIX>
__global__ void __launch_bounds__(256)
    steps_kernel(walk::Slots s, walk::Keys k, uint32_t walks, uint32_t first, uint32_t nsteps) {
  __shared__ uint32_t te[256];
  load_te(te);
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::run_slot<LINES, REMIX>(s, k, w, first, nsteps, te);
}

__global__ void __launch_bounds__(256) final_kernel(walk::Slots s, walk::Keys k, walk::Jobs j, uint32_t walks) {
  const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < walks) walk::finish_slot(s, k, j, w);
}

}  // namespace walkgpu
