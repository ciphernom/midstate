// CI only: compiles the kernels with clang, which needs no CUDA toolkit. It
// defines the CUDA keywords that the toolkit's headers would, then
// instantiates every kernel. kernels.sh runs it.
#define __CUDACC__ 1  // nvcc always defines it; clang only through the toolkit's headers
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

#include "../walk_kernels.cuh"

template __global__ void walkgpu::init_kernel<4096u>(walk::Slots, walk::Keys, walk::Jobs, uint32_t);
template __global__ void walkgpu::init_kernel<256u>(walk::Slots, walk::Keys, walk::Jobs, uint32_t);
template __global__ void walkgpu::steps_kernel<4096u, 4096u>(walk::Slots, walk::Keys, uint32_t, uint32_t, uint32_t);
template __global__ void walkgpu::steps_kernel<256u, 256u>(walk::Slots, walk::Keys, uint32_t, uint32_t, uint32_t);
