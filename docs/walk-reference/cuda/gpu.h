// gpu.h: what bench.cpp needs from the CUDA side (walk_cuda.cu). Plain C++,
// so the host compiler never sees a CUDA header.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace gpu {

struct Info {
  char name[256];
  char pci_bus_id[32];   // as CUDA reports it, e.g. "0000:02:00.0"
  int major, minor;      // compute capability
  int sms;               // streaming multiprocessors
  int l2_bytes;
  size_t total_mem, free_mem;
  int mem_clock_khz;     // memory clock; transfers a second are twice this
  int bus_width_bits;
  bool watchdog;         // the GPU drives a display, which limits how long a kernel may run
  int driver_version;    // CUDA versions as 1000 * major + 10 * minor
  int runtime_version;
};

struct KernelInfo {
  int regs;                 // registers per thread in the step kernel
  int local_bytes;          // local memory per thread (spills): ideally 0
  int blocks_per_sm;        // resident blocks of `block` threads
  uint32_t resident_walks;  // walks the whole GPU can have in flight at once
};

struct Context;

// Selects `device` and describes it.
bool open(int device, Info* info, char* err, size_t errlen);
bool kernel_info(bool fast, uint32_t block, int sms, KernelInfo* out, char* err, size_t errlen);

// Device memory for `capacity` walks, launched `block` threads to a block.
Context* create(bool fast, uint32_t capacity, uint32_t block, char* err, size_t errlen);
void destroy(Context* c);
const char* error(const Context* c);

// Walks 0..walks-1 hash `hashes` (one 32-byte hash shared by all when
// hash_count is 1, else one each) with `nonces`. Waits for queued work.
bool set_jobs(Context* c, const uint8_t* hashes, uint32_t hash_count, const uint64_t* nonces, uint32_t walks);

// Queued on the context's stream; a failure inside a kernel surfaces at the
// next sync() or results().
bool launch_init(Context* c, uint32_t walks);
bool launch_steps(Context* c, uint32_t walks, uint32_t first_step, uint32_t nsteps);
bool launch_final(Context* c, uint32_t walks);
bool sync(Context* c);

// The finished walk hashes, 32 bytes a walk. Waits for queued work.
bool results(Context* c, uint8_t* out, uint32_t walks);

}  // namespace gpu
