// walk_cuda.cu: the CUDA side of walk-cuda. Device setup, memory and kernel
// launches, behind the plain interface in gpu.h.
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "gpu.h"
#include "walk_kernels.cuh"

namespace gpu {

struct Context {
  bool fast;
  uint32_t capacity, block;
  walk::Slots slots;
  walk::Keys keys;
  uint32_t* hashes;    // capacity × 8 words
  uint32_t hash_stride;
  uint64_t* nonces;    // capacity
  uint32_t* out;       // capacity × 8 words
  cudaStream_t stream;
  char err[512];
};

namespace {

bool fail(char* err, size_t len, const char* what, cudaError_t e) {
  snprintf(err, len, "%s: %s", what, cudaGetErrorString(e));
  return false;
}

bool check(Context* c, const char* what, cudaError_t e) {
  if (e == cudaSuccess) return true;
  return fail(c->err, sizeof c->err, what, e);
}

uint32_t grid(uint32_t walks, uint32_t block) { return (walks + block - 1) / block; }

walk::Jobs jobs(const Context* c) {
  walk::Jobs j;
  j.hashes = c->hashes;
  j.hash_stride = c->hash_stride;
  j.nonces = c->nonces;
  j.out = c->out;
  return j;
}

template <class Kernel>
bool describe(Kernel kernel, uint32_t block, int sms, KernelInfo* ki, char* err, size_t errlen) {
  cudaFuncAttributes a;
  cudaError_t e = cudaFuncGetAttributes(&a, kernel);
  if (e != cudaSuccess) return fail(err, errlen, "kernel attributes", e);
  int blocks = 0;
  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, (int)block, 0);
  if (e != cudaSuccess) return fail(err, errlen, "occupancy", e);
  ki->regs = a.numRegs;
  ki->local_bytes = (int)a.localSizeBytes;
  ki->blocks_per_sm = blocks;
  ki->resident_walks = (uint32_t)blocks * block * (uint32_t)sms;
  return true;
}

}  // namespace

bool open(int device, Info* info, char* err, size_t errlen) {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) return fail(err, errlen, "no usable CUDA device", e);
  if (device < 0 || device >= count) {
    snprintf(err, errlen, "there is no CUDA device %d (this machine has %d)", device, count);
    return false;
  }
  if ((e = cudaSetDevice(device)) != cudaSuccess) return fail(err, errlen, "selecting the device", e);
  cudaDeviceProp p;
  if ((e = cudaGetDeviceProperties(&p, device)) != cudaSuccess) return fail(err, errlen, "device properties", e);
  memset(info, 0, sizeof *info);
  snprintf(info->name, sizeof info->name, "%s", p.name);
  info->major = p.major;
  info->minor = p.minor;
  info->sms = p.multiProcessorCount;
  info->l2_bytes = p.l2CacheSize;
  info->total_mem = p.totalGlobalMem;
  int watchdog = 0;
  cudaDeviceGetAttribute(&info->mem_clock_khz, cudaDevAttrMemoryClockRate, device);
  cudaDeviceGetAttribute(&info->bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device);
  cudaDeviceGetAttribute(&watchdog, cudaDevAttrKernelExecTimeout, device);
  info->watchdog = watchdog != 0;
  cudaDeviceGetPCIBusId(info->pci_bus_id, (int)sizeof info->pci_bus_id, device);
  cudaDriverGetVersion(&info->driver_version);
  cudaRuntimeGetVersion(&info->runtime_version);
  size_t free_bytes = 0, total_bytes = 0;
  if ((e = cudaMemGetInfo(&free_bytes, &total_bytes)) != cudaSuccess) return fail(err, errlen, "memory info", e);
  info->free_mem = free_bytes;
  return true;
}

bool kernel_info(bool fast, uint32_t block, int sms, KernelInfo* out, char* err, size_t errlen) {
  return fast ? describe(walkgpu::steps_kernel<256, 256>, block, sms, out, err, errlen)
              : describe(walkgpu::steps_kernel<4096, 4096>, block, sms, out, err, errlen);
}

void destroy(Context* c) {
  if (!c) return;
  cudaFree(c->slots.r);
  cudaFree(c->slots.c);
  cudaFree(c->slots.prog);
  cudaFree(c->slots.pads);
  cudaFree(c->hashes);
  cudaFree(c->nonces);
  cudaFree(c->out);
  if (c->stream) cudaStreamDestroy(c->stream);
  delete c;
}

Context* create(bool fast, uint32_t capacity, uint32_t block, char* err, size_t errlen) {
  Context* c = new Context();
  c->fast = fast;
  c->capacity = capacity;
  c->block = block;
  c->keys = walk::context_keys();
  c->slots.capacity = capacity;
  const walk::Params& p = fast ? walk::FAST : walk::MAINNET;
  const size_t n = capacity;
  struct Alloc {
    void** ptr;
    size_t bytes;
    const char* what;
  } allocs[] = {
      {(void**)&c->slots.r, n * 8 * sizeof(uint64_t), "walk registers"},
      {(void**)&c->slots.c, n * sizeof(uint64_t), "walk state"},
      {(void**)&c->slots.prog, n * sizeof(uint32_t), "walk programs"},
      {(void**)&c->slots.pads, n * p.lines * 8 * sizeof(uint64_t), "scratchpads"},
      {(void**)&c->hashes, n * 8 * sizeof(uint32_t), "mining hashes"},
      {(void**)&c->nonces, n * sizeof(uint64_t), "nonces"},
      {(void**)&c->out, n * 8 * sizeof(uint32_t), "results"},
  };
  for (const Alloc& a : allocs) {
    const cudaError_t e = cudaMalloc(a.ptr, a.bytes);
    if (e != cudaSuccess) {
      snprintf(err, errlen, "allocating %.1f MiB of %s: %s", a.bytes / 1048576.0, a.what, cudaGetErrorString(e));
      destroy(c);
      return nullptr;
    }
  }
  uint32_t te[256];
  walk::build_te(te);
  cudaError_t e = cudaMemcpyToSymbol(walkgpu::c_te, te, sizeof te);
  if (e == cudaSuccess) e = cudaStreamCreate(&c->stream);
  if (e != cudaSuccess) {
    fail(err, errlen, "setting up", e);
    destroy(c);
    return nullptr;
  }
  return c;
}

const char* error(const Context* c) { return c->err; }

bool set_jobs(Context* c, const uint8_t* hashes, uint32_t hash_count, const uint64_t* nonces, uint32_t walks) {
  if (walks == 0 || walks > c->capacity || (hash_count != 1 && hash_count != walks)) {
    snprintf(c->err, sizeof c->err, "bad job: %u walks, %u hashes, room for %u", walks, hash_count, c->capacity);
    return false;
  }
  std::vector<uint32_t> words((size_t)hash_count * 8);
  for (size_t i = 0; i < words.size(); ++i) words[i] = b3::load32(hashes + 4 * i);
  c->hash_stride = hash_count == 1 ? 0 : 8;
  return check(c, "uploading jobs", cudaMemcpyAsync(c->hashes, words.data(), words.size() * 4, cudaMemcpyHostToDevice, c->stream)) &&
         check(c, "uploading jobs", cudaMemcpyAsync(c->nonces, nonces, (size_t)walks * 8, cudaMemcpyHostToDevice, c->stream)) &&
         check(c, "uploading jobs", cudaStreamSynchronize(c->stream));
}

bool launch_init(Context* c, uint32_t walks) {
  const walk::Jobs j = jobs(c);
  if (c->fast)
    walkgpu::init_kernel<256><<<grid(walks, c->block), c->block, 0, c->stream>>>(c->slots, c->keys, j, walks);
  else
    walkgpu::init_kernel<4096><<<grid(walks, c->block), c->block, 0, c->stream>>>(c->slots, c->keys, j, walks);
  return check(c, "launching init", cudaGetLastError());
}

bool launch_steps(Context* c, uint32_t walks, uint32_t first_step, uint32_t nsteps) {
  if (c->fast)
    walkgpu::steps_kernel<256, 256>
        <<<grid(walks, c->block), c->block, 0, c->stream>>>(c->slots, c->keys, walks, first_step, nsteps);
  else
    walkgpu::steps_kernel<4096, 4096>
        <<<grid(walks, c->block), c->block, 0, c->stream>>>(c->slots, c->keys, walks, first_step, nsteps);
  return check(c, "launching steps", cudaGetLastError());
}

bool launch_final(Context* c, uint32_t walks) {
  walkgpu::final_kernel<<<grid(walks, c->block), c->block, 0, c->stream>>>(c->slots, c->keys, jobs(c), walks);
  return check(c, "launching final", cudaGetLastError());
}

bool sync(Context* c) { return check(c, "running the walk", cudaStreamSynchronize(c->stream)); }

bool results(Context* c, uint8_t* out, uint32_t walks) {
  std::vector<uint32_t> words((size_t)walks * 8);
  if (!check(c, "running the walk", cudaMemcpyAsync(words.data(), c->out, words.size() * 4, cudaMemcpyDeviceToHost, c->stream)) ||
      !check(c, "running the walk", cudaStreamSynchronize(c->stream)))
    return false;
  for (size_t i = 0; i < words.size(); ++i) b3::store32(out + 4 * i, words[i]);
  return true;
}

}  // namespace gpu
