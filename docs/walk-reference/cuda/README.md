# Midstate Walk GPU bench

Measures Midstate Walk, midstate's proof of work from block 420,000, on an
NVIDIA GPU, so that GPUs can be compared with CPUs. It is a benchmark, not a
miner. The specification is [`docs/MIDSTATE_WALK.md`](../../MIDSTATE_WALK.md),
whose [GPUs](../../MIDSTATE_WALK.md#gpus) section has the results so far.

## Build

You need CMake 3.18 or newer, a CUDA toolkit that knows your GPU, and the
NVIDIA driver. The bench builds on its own, outside midstate's Cargo build:

```bash
cd docs/walk-reference/cuda
cmake -S . -B build
cmake --build build -j
```

CMake asks `nvidia-smi` which GPU is in the machine and builds for it. To
choose, add `-DCMAKE_CUDA_ARCHITECTURES=<n>` to the first command:

| GPU | n |
|---|---|
| Maxwell (GTX 900, Quadro M) | 52 |
| Pascal (GTX 10) | 61 |
| Turing (RTX 20) | 75 |
| Ampere (RTX 30) | 86 |
| Ada (RTX 40) | 89 |
| Hopper (H100) | 90 |

A toolkit too old for your card can still build for an older architecture, and
the result runs on newer cards: `86` from CUDA 11.5 runs on an RTX 40. CUDA 13
no longer builds for Maxwell, Pascal or Volta, so use a 12.x toolkit for those.

## Run

```bash
./build/walk-cpu --selftest     # the CPU reference: AES, BLAKE3, known answers, the GPU's code path
./build/walk-cuda --seconds 60  # about a minute and a half; post the last line
```

`walk-cuda` does five things:

1. **Known answers.** It runs midstate's three known answers on the GPU and
   stops if any is wrong. A wrong GPU's speed means nothing.
2. **Sweep.** It times more and more walks in flight, from one warp up, until
   the rate stops growing. The first rows show how long one walk takes on its
   own; the rate stops growing when the GPU's memory can't serve any more
   random accesses.
3. **Sustained run.** It runs the fewest walks that reach full speed for 30 s
   (`--seconds`), reading the board's power, clocks and temperature through
   NVML.
4. **Check.** It recomputes up to 4,096 walks of the first batch and of the
   last on the CPU and compares them, so an overclock that gets walks wrong
   when hot shows up.
5. **Report.** It prints a summary and a one-line result to post.

| Option | Meaning |
|---|---|
| `--seconds S` | Length of the sustained run (default 30; 0 skips it). Post a 60 s run. |
| `--batch N` | Walks in flight, instead of picking them from the sweep |
| `--sweep` | Time every batch size that fits, not just until the rate levels off |
| `--no-sweep` | Skip the sweep |
| `--threads N` | Threads a block: 32, 64, 128 (default) or 256 |
| `--verify N\|all` | Walks of the first and last batch to check on the CPU (default 4,096) |
| `--no-verify` | Skip that check. The known answers are always checked. |
| `--hash HEX`, `--start N` | Mining hash and first nonce (default zeros and 0) |
| `--fast` | midstate's fast-mining test parameters: a correctness check, not a benchmark |
| `--device N` | Which GPU (default 0) |

Exit status: 0 for success, 1 for a bad option, 2 for a CUDA error, 3 if the
known answers are wrong, 4 if the CPU check finds wrong walks.

## Reading the results

- **attempts/s** is the sustained rate, the number to compare with
  `midstate bench-pow` on a CPU.
- **a walk takes** is how long one attempt is in flight. On a GPU it is
  hundreds of times longer than on a CPU core: every step waits for the one
  before.
- **Memory (implied)** is the rate times 2^18 steps times 8 random 32-byte
  transfers: each step reads 4 random words and writes 4 back. It is worked
  out, not measured. At full speed a GPU's rate is set by how many random
  accesses its memory can serve. The percentage shows how much of the
  memory's peak (streaming) bandwidth that comes to. Random access only ever
  gets a fraction of it. A Quadro M6000 does 854 attempts/s: 57 GB/s, 18% of
  its 317 GB/s.
- **Power** is the board's average draw over the sustained run, from NVML. It
  is not measured at the wall. **attempts per joule** is the efficiency number
  to set against a CPU's.
- **held back by** lists anything that slowed the clocks: the power limit,
  temperature, or a hardware slowdown.

## How it works

- One CUDA thread per walk, each with its own 256 KiB scratchpad in device
  memory.
- **Walks run in launches of 4,096 steps.** A walk's registers wait in device
  memory between launches. A walk takes a second or two on a GPU. One launch
  per walk would run into display watchdogs (Windows allows 2 s), and none of
  these launches comes near that.
- **On-chip data stays on chip.** The AES round uses a table in shared memory.
  The walk's registers, its program and BLAKE3's state stay in registers: every
  array index is a constant once the code is unrolled. `walk-cuda` prints the
  register count and any local memory. In the first version, tables in
  constant memory and arrays indexed at run time put them in local memory:
  205 registers and 576 bytes of local memory a thread, against 64 and none
  now. On a Quadro M6000 that left a lone walk at 0.67 s and raised full speed
  by 3%, which is what to expect if memory sets the pace.
- **One copy of the walk.** `walk_core.h` runs on both sides: the GPU kernels,
  the CPU check and `walk-cpu`. `walk-cpu --selftest` also runs the kernels'
  per-thread code (`init_slot`, `run_slot`, `finish_slot`) on the CPU, a few
  steps at a time as the GPU does, and compares it with the one-pass walk.
- **Checks.** The known answers are midstate's `GOLDEN` vectors from
  `src/core/walk.rs`. The BLAKE3 test outputs come from the `blake3` crate.

## CI

`.github/workflows/walk-reference.yml` runs whenever this folder or
`src/core/walk.rs` changes. It needs no GPU and no CUDA toolkit:

- `ci/check_golden.py` checks that `walk_core.h`'s known answers are
  `walk.rs`'s. If the walk ever changes, this fails until the bench is
  brought up to date too.
- `walk-cpu --selftest`, built with `-Werror`.
- `ci/kernels.sh` compiles the kernels to PTX with clang for Maxwell and Ada
  GPUs and fails if any of them uses local memory.

## Limits

- The 64-bit division is the compiler's routine. A hand-written 64-by-32-bit
  division would shorten each step a little. At full speed a GPU waits on
  memory, so this mostly affects the first rows of the sweep.
- Built and tested on Linux. NVML is loaded at run time, so a machine without
  it builds and runs and just reports no power.
- Profilers: Nsight Compute handles Volta and newer GPUs; Maxwell and Pascal
  need `nvprof`.

## Files

| File | What |
|---|---|
| `walk_core.h` | The walk, for the GPU and the host: AES, program, steps, init and finish, slots, known answers |
| `blake3_small.h` | The parts of BLAKE3 the walk uses |
| `walk_kernels.cuh` | The three kernels |
| `walk_cuda.cu`, `gpu.h` | Device setup, memory and launches, behind a plain C++ interface |
| `bench.cpp` | `walk-cuda`: known answers, sweep, sustained run, checks, report |
| `nvml_power.cpp`, `nvml_power.h` | Power, clocks and temperature through NVML |
| `walk_cpu.cpp` | `walk-cpu`: the reference and self-test |
| `ci/` | The CI checks above |
