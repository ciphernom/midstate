# Midstate Walk

Midstate Walk is the proof of work from block `WALK_ACTIVATION_HEIGHT`
(371,500) on. It replaces the 1,000,000-step BLAKE3 chain, which GPUs
dominate. The code is `src/core/walk.rs`, and its module docs hold the exact
definition.

## Why the BLAKE3 chain favours GPUs

Mining is a lottery, so what counts is attempts per second, not the time one
attempt takes. A BLAKE3-chain attempt needs 32 bytes of state and nothing but
32-bit add, rotate and xor. A GPU runs tens of thousands of them side by side.
Making each attempt sequential never limited that.

## What Walk asks of each attempt

- **A 256 KiB scratchpad, walked as one dependent chain.** Each step's
  addresses depend on the previous step's output, so nothing can run ahead.
  The pad fits a CPU core's L2 cache, including the Raspberry Pi 5's
  Cortex-A76, which has 512 KiB per core. It does not fit a GPU thread's
  on-chip memory.
  - A GPU either waits on its shared L2 at every step, with only about
    72 MB / 256 KiB ≈ 288 attempts in flight on an RTX 4090.
  - Or it streams pads from VRAM. That moves four 32-byte sectors in and four
    out per step, which is 64 MiB of traffic per attempt.
- **Four independent loads per step.** A CPU core has all four in flight at
  once, so they cost it little. They double a GPU's VRAM traffic.
- **A program per attempt.** BLAKE3 picks which operation each of the eight
  registers runs, and re-picks it every 4,096 steps.
  - A CPU predicts the branches after a few steps.
  - A GPU warp's 32 threads each have their own program and diverge.
- **Operations GPUs emulate.** 64-bit multiply, 64×64→128 multiply-high,
  64-bit divide, and an AES round. CPUs have hardware for the AES round: x86
  AES-NI and the ARMv8 crypto extensions (Pi 5 yes, Pi 4 no). A portable
  table implementation produces the same results on anything else.

It is still hashing, so it is as post-quantum as before.

## Parameters

| | Value | Test builds (`fast-mining`) |
|---|---|---|
| Scratchpad | 4,096 lines × 64 B = 256 KiB | 256 lines (16 KiB) |
| Steps per attempt | 2^18 | 2^10 |
| BLAKE3 re-mix | every 2^12 steps | every 2^8 |
| Activation | height 371,500 | height 20 |
| Difficulty reset (`WALK_TARGET_SHIFT`) | 3 (target × 8 at activation) | 0 |

### Measured costs

These were measured on one core of a 2.8 GHz Xeon cloud VM with `midstate
bench-pow`.

| | Per attempt | Per core |
|---|---|---|
| BLAKE3 chain | 66.7 ms | 15 attempts/s |
| Walk | 3.1 ms | ~320 attempts/s |

Verifying a block or a share costs one attempt.

Variants measured on the same core, from which the shipped one was chosen:

| Variant | Per attempt | Why not chosen |
|---|---|---|
| Independent steps, 2 loads | 1.6 ms | Leaves a GPU free to overlap steps |
| Chained steps, 2 loads | 2.5 ms | Half the VRAM traffic per step |
| **Chained steps, 4 loads** | **3.1 ms** | Shipped |

### GPUs

[`docs/walk-reference/cuda/`](walk-reference/cuda/README.md) measures Walk on
an NVIDIA GPU with one command. It checks the known answers on the GPU, finds
how many walks in flight the GPU needs, runs them for a minute while reading
the board's power through NVML, and checks the GPU's results on the CPU.

Measured so far, the CPUs with `midstate bench-pow`:

| Machine | Attempts/s | |
|---|---|---|
| EPYC 9554, 64 threads | 17,359 | Server, Zen 4 |
| Zen 4, 16 cores, 32 threads | 13,004 | Desktop |
| H100 SXM GPU (Hopper, 2022) | 11,418 | 332 W board power: 34.4 attempts per joule |
| RTX 5090 GPU (Blackwell, 2025) | 5,378 | 256 W board power: 21.0 attempts per joule |
| EPYC 7713, 15 threads | 5,217 | A 15-CPU container on a 128-core server |
| A100 SXM GPU (Ampere, 2020) | 5,080 | 205 W board power: 24.8 attempts per joule |
| Ryzen 7 5800X3D, 16 threads | 4,164 | |
| Xeon E5-1650 v3, 12 threads | 1,761 | |
| Raspberry Pi 5, 4 cores at 3.1 GHz | 1,545 | Overclocked, with the active cooler |
| Quadro M6000 GPU (Maxwell, 2015) | 854 | 129 W board power: 6.6 attempts per joule |

What the GPU runs show:

- **Memory bandwidth sets the rate.** Each step reads four random 32-byte
  sectors and writes four back, so an attempt moves 64 MiB. Every card tops
  out at 17–23% of its peak bandwidth in random transfers, GDDR and HBM
  alike: 18% on the M6000, 20% on the 5090, 17% on the A100, 23% on the H100.
  All four ran 100% busy at about half their power limit, waiting on memory.
  The H100 is fastest because it has the most bandwidth, 3.35 TB/s.
- **Keeping scratchpads in L2 loses.** The 5090's 96 MiB L2 holds 384 of
  them. Run that way, it did 1,664 attempts/s: under a third of streaming, at
  10.8 attempts per joule.
- **A lone walk costs about 1,800 clock cycles a step** with its scratchpad in
  L2, on the A100, the 5090 and the H100 alike: 0.16 s a walk on the 5090,
  against 2.6 ms on a Pi 5 core. The same count on all three points to the
  walk's own work, its diverging programs and emulated operations, rather
  than to memory.
- **Per joule a Pi 5 wins.** Even drawing its power supply's full 27 W, a Pi 5
  at 1,545 attempts/s does 57 attempts per joule. The best GPU, the H100,
  does 34.

On the BLAKE3 chain, GPUs were orders of magnitude ahead.

The estimate made before any GPU was measured, against the measurements:

- It put an RTX 4090 level with a 16-core desktop. The 5090, a bigger card,
  does 41% of one, and the H100 88%.
- Streaming from memory, it assumed random traffic at 40% of peak; it is
  17–23%. A GPU's rate is about a fifth of its peak bandwidth divided by
  64 MiB: about 3K attempts/s for a 4090.
- With the scratchpads in L2, it estimated 5K attempts/s for a 4090, which
  needs a step every 220 ns. The 5090 takes 880 ns with its L2 full.

Because bandwidth sets the rate, the newest HBM cards should be faster still.
By the same rule an H200 (4.8 TB/s) would do about 14K attempts/s, and a B200
(8 TB/s) 20–27K, more than a 16-core desktop per card. Those are predictions,
not measurements.

The knobs, if that margin needs widening:
- Loads per step: a GPU pays for every load in memory traffic, while a CPU
  core overlaps them in its L2.
- `WALK_LINES`: a bigger pad pushes GPUs out of L2, but must still fit a
  Pi 5's 512 KiB L2.
- The operation mix.

### Any GPU through wgpu, the Pi 5's first

`src/core/wgpu_walk.rs` is a second GPU kernel, in WGSL, so the Walk runs on
whatever wgpu reaches: Vulkan, Metal or DX12. That includes the Raspberry
Pi 5's VideoCore VII (through Mesa's v3dv) and AMD, Intel and Apple GPUs.
For now it is measured, not mined with:

```bash
midstate bench-pow --gpu [--with-cpu] [--gpu-adapter NAME] [--gpu-walks N] [--seconds S]
```

The CPU seeds each walk and grows its pad, with its AES instructions. That
is microseconds per attempt. The GPU runs the 2^18 steps and the 64 BLAKE3
re-mixes, and the CPU hashes the final registers. A WGSL kernel has 32-bit
words only, and the Pi's GPU has a 24-bit multiplier. So the kernel works
like this:

- **64-bit words** are pairs of 32-bit words. Multiplies are built from
  16-bit limbs; the 128-bit high product takes sixteen.
- **The division** always has a divisor with its top bit set, so 64÷32 is two
  base-2^16 digits of Knuth's algorithm D. Each digit is exact after at most
  two corrections.
- **The program draw** only needs `z mod 8!`, because Fisher–Yates reads its
  digits in radices 8 to 2.
- **The AES round** uses four rotated 1 KiB tables.
- **Each launch** lasts about 60 ms. The Pi's kernel driver resets a GPU
  whose job runs for half a second.

Lanes in lockstep diverge on each walk's program, so there is a second
version that **groups the walks by their programs**. At every re-mix the CPU
sorts the walks by the step their AES round falls on, then the division, then
the high multiply. Lanes side by side then take the costly paths at the
same step: with a few hundred walks in flight, a group of 16 lanes has its
AES at one step of eight instead of nearly all of them. The hashes don't
change, because which lane runs which walk is free.

The benchmark:

1. Checks the GPU against the three known answers as whole walks, and stops
   if any comes out wrong.
2. Sweeps the workgroup size and the walks at once, then tries grouping at
   the best size.
3. Runs two probes. Every operation an XOR leaves only the memory traffic.
   A shared pad of eight lines, which stays cached, leaves only the
   arithmetic. Together they say which limits the GPU.
4. Measures the CPU, and with `--with-cpu` both at once, with the board's
   temperature and power before and after. The GPU's heat can slow the CPU.

The kernel reproduces every operation, the re-mix, partial walks and the
known answers on a software Vulkan driver (`the_gpu_walk_is_the_walk`, run
with `VK_ICD_FILENAMES` pointing at SwiftShader or lavapipe and
`MIDSTATE_TEST_GPU` naming it). It has not yet run on a Pi 5's GPU.

The expectation from the measured GPUs above is modest. Take random traffic
at 17–23% of the Pi's 17 GB/s, and 64 MiB an attempt (128 MiB if each access
moves a 64-byte line rather than a 32-byte sector). That caps it at roughly
25–50 attempts/s, 2–3% of the four cores, and the emulated arithmetic may hold
it under that. The benchmark is how to find out.

### Several walks per thread on a CPU

One walk keeps a core waiting. At 3.1 GHz a Pi 5 core takes about 31 cycles a
step, mostly the L2 load and the operation that the next step's addresses
depend on. The core could issue four instructions a cycle and averages about
one. Walks of other nonces are independent, so a thread can step two or three
in turn and the core overlaps their waits, as SMT does on the CPUs that have
it. The catch: two pads fill a Cortex-A76's 512 KiB L2, so some loads fall to
the shared L3. `walk::walk_hash_many` steps K walks in turn, and each hash is
exactly that nonce's walk.

```bash
midstate bench-pow --interleave [--seconds S] [--threads T]
```

This measured 1, 2 and 3 walks per thread, with the threads pinned to their
cores and not (pinning could matter because the L2 is per core, so a thread
that migrates warms 256 KiB again). It now measures 1 to 4 walks per thread
for each way of writing the step, unpinned, and then the count the miner
would choose. Each run checks the known answers, and only the hashing is
timed.

On a Raspberry Pi 5 (4 threads, 10 s each) it does not pay:

| Walks per thread | Attempts/s | Against one |
|---|---|---|
| 1 | 1,495.9 | |
| 1, pinned | 1,490.5 | −0.4% |
| 2 | 364.4 | −75.6% |
| 2, pinned | 370.3 | −75.2% |
| 3 | 194.6 | −87.0% |
| 3, pinned | 193.7 | −87.0% |

Two pads fill a core's 512 KiB L2, and four cores' eight pads (2 MiB) fill
the shared L3, so the loads go out to DRAM. That is the scratchpad doing what
it was sized for: one walk per core is the most a Pi 5 can run. Pinning
changes nothing.

CPUs with more cache per core gain a lot. On a 2-core Xeon with 2 MiB of L2
a core (2 threads, 3 s each):

| Walks per thread | Indexed | Unindexed | Ahead |
|---|---|---|---|
| 1 | 596.7 | 585.5 | **629.9** |
| 2 | **908.2** | 946.3 | 895.1 |
| 3 | **1,064.4** | 1,032.0 | 548.9 |
| 4 | **1,014.8** | 997.5 | 519.3 |

In bold: the step the miner uses for that count. Three walks per thread did
69% more than the best single walk, and 78% more than v6.4.0's loop (one
indexed walk). The faster steps keep one walk's eight registers in the CPU's,
which several walks cannot all do, so with 2 to 4 walks the miner uses the
indexed step (`walk::miner_step_for`). Zen 4 and Zen 5 cores have 1 MiB of
L2 and the X3D parts 96-128 MB of L3, so they should gain too; the miner
does not assume it.

**The miner measures it.** Before its first Walk job, the CPU miner times 1
to 4 walks per thread on all of its threads, each with the step it would
use, up the counts and back down (about 6 s, once per process). It checks
the known answers that way and mines with the fastest
(`walk::walks_per_thread`), and it logs the rates. On a Pi that is one walk
per thread; on the Xeon above, three, which `midstate bench-pow` measured at
1,132 attempts/s on 2 threads against about 600 before.
`MINER_WALKS_PER_THREAD=1` to `4` sets it instead. Each hash is exactly its
nonce's walk: a test runs every way of writing the step, 1 to 4 walks per
thread, against the known answers in every slot.

### The standalone bencher

`tools/walk-bench` measures the Walk without the rest of midstate: no node,
wallet or networking, and no fat LTO. It compiles in about a minute where
midstate takes many. It builds the miner's own `src/core/walk.rs` and
`src/core/wgpu_walk.rs`, not copies, so what it measures is what mines, and
its own build of those files runs their tests.

```bash
cd tools/walk-bench
cargo build --release --target aarch64-unknown-linux-gnu   # for a Pi 5, from another machine
midstate-walk-bench cpu          # 1 thread and all threads, with the known answers
midstate-walk-bench interleave   # as bench-pow --interleave
midstate-walk-bench gpu [--with-cpu] [--adapter NAME] [--walks N] [--wgsl FILE]
midstate-walk-bench dump-wgsl walk.wgsl
```

The GPU kernel is a single WGSL source. What tells one variant from another
(the workgroup size, the probes, grouping, stopping at a step) is a pipeline
constant, set when each pipeline is built. So `dump-wgsl` writes the whole
kernel, and `gpu --wgsl` runs an edited one: a change to the kernel can be
tried on a GPU without compiling anything.

### Checking a GPU

`bench-pow --gpu` checks the kernel before it measures:

1. Every operation on 2^20 operand pairs against the CPU, then the re-mix on
   2^16 register sets.
2. The three known answers, each as a whole walk.
3. If one comes out wrong, it finds where. It runs the walk in stretches of
   one re-mix, checking each against the CPU. It runs a wrong stretch again
   from the CPU's state, which tells a fault that comes and goes (a GPU
   reset, a race) from one that repeats. Then it narrows the fault to one
   step, with a variant of the kernel that stops at any step.
4. It reports that step: the register, the operation and its inputs, and
   when each of its four pad words was last written. It also reports whether
   the operation alone gives the right answer on the GPU, and whether the
   GPU's result is what the step gives with one of its words as it was
   before an earlier step wrote it (a stale read).

The search ran on a Pi 5 with Mesa 25.0.7's v3dv. Every operation and the
re-mix were right, but the walk went wrong at its first step, and after one
step every register and the program were those of an earlier run's end. That
looked like the kernel reading its state as it last wrote it rather than as
just uploaded, so uploads now go into a buffer the kernel only reads, and a
copy kernel (`copy_in`) puts them in place. Uploads also go in batches of
8 MiB: 1,024 pads in one go (256 MiB) was more than the Pi's driver could
stage.

The next run disproved that explanation. The bench checks uploads both ways
at the start, four times each, and the Pi's GPU saw every one, written
straight in or through the copy kernel. In that run, with the upload buffer
as the kernel's sixth binding, the AES tables and the re-mix's constants read
as zeros (AES of zero gave zero, not 0x63 in every byte), and the walk was
still wrong at its first step. The fault is in v3dv or in how the kernel
binds its buffers there, and it is not found.

It does not matter for mining. The same run measured what the kernel costs on
the Pi's GPU, right or wrong: 13.3 attempts/s at best (workgroup 64, 1,024
walks, grouped). With every operation an XOR it does 15.1, only 1.1 times as
many, so it is waiting on memory, not arithmetic: 28 million random 8-byte
reads a second. The four cores do 1,490.8, so the GPU would add 0.9%, less
than its heat costs the CPU. The Pi mines on its CPU, and work on its GPU
stops here. The kernel stays, checked on a software Vulkan driver, for GPUs
that run it right.

### Where a CPU's step goes

`midstate-walk-bench cpu-probe` measures, on one thread:

- the latency of a load whose address is the last load's result, for working
  sets from 16 KiB to 16 MiB: each cache level, and the pad's 256 KiB;
- a step of the walk with its own mix of operations, and with every
  operation each of the eight in turn, for each way of writing the step
  (below);
- whole walks with each way, and that each gives the reference's hashes;
- seeding and growing the pad, and the re-mixes, per attempt.

It gives cycles at the clock the CPU runs at. The attempts per second it
predicts from the steps match the ones it measures, so nothing is left out:
seeding, the pad and the re-mixes take under 1% of an attempt.

On a Pi 5 at 3.0 GHz a load from 256 KiB takes 13.1 cycles (4 from L1). A
step took 29.8 cycles: 13.1 for the load, 3.3 for the operation on average
(the division 12 over the XOR, AES 9, the multiplies 1.5 and 3) and 13.4 for
the rest: the addresses, the second load, the XORs and the step's writes.
The division and AES are already single hardware instructions.

#### Ways of writing the step

There are three ways of writing the same step. Each computes the same walk:
a test runs all three from random registers under random programs, over pads
they keep writing, and compares them after every round of eight. Each also
gives the known answers.

- **Indexed** (`step`) is the walk as it is defined and the reference. It
  indexes the registers with `t`, so all eight live in memory, and it XORs
  AES's high half into `r[t]` on every step.
- **Unindexed** (`step_unindexed`) keeps the registers in registers. It
  picks `k` with two selects and applies the high half under masks, on AES
  steps only.
- **Ahead** (`step_ahead`) also works out the next step's address as
  `(r[s+1] ^ m1 ^ m3) ^ y`. That leaves one XOR after the operation, where
  the step as written has two (`out = y ^ m1 ^ m3`, then `x = r[s+1] ^ out`).

The miner uses the fastest that has been measured on each kind of CPU
(`walk::MINER_STEP`):

| CPU | Indexed | Unindexed | Ahead | Mines with |
|---|---|---|---|---|
| Raspberry Pi 5, a step | 29.8 cycles | 28.3 cycles (−5.1%) | not yet measured | unindexed |
| Xeon (cloud), whole walks on one thread | 412–426 /s | 425–438 /s (+3%) | 433–448 /s (+5%) | ahead |

`cpu-probe` also times the unindexed step without its four writes to the
pad. That is not the walk; it only shows what the writes cost.

## What changed in consensus

- **Proof of work by height.**
  - `pow_algo_at(height)` picks the algorithm.
  - `create_extension_at` and `verify_extension_at` route by it.
  - `apply_batch` verifies each block at its real height, so history below
    371,500 still validates with the BLAKE3 chain.
- **Difficulty.**
  - From 371,500, ASERT's integer shift gets `WALK_TARGET_SHIFT` added. This
    is the same as moving the schedule by that many half-lives.
  - The rule stays a pure function of (height, timestamp), and ASERT corrects
    whatever error the estimate has.
  - Erring high gives an hour or two of fast blocks. Erring low gives several
    hours of slow ones.
- **Sync.**
  - Header heights select the algorithm, so they are now enforced:
    `verify_header_chain` requires consecutive heights.
  - The batch phase refuses a header whose height differs from the height it
    is applied at.
  - Without this, a peer could label Walk-era headers with older heights and
    pass GPU-mined BLAKE3-chain work through `apply_batch_skip_pow`.
- **Orphans.**
  - An orphan's height is unknown, so `verify_extension_near` accepts either
    algorithm within 64 blocks of the switch.
  - The orphan target screen widens by `2^WALK_TARGET_SHIFT` there.
  - The real check happens when the block is applied.

## What changed for miners and pools

- **`/block_template` responses** (HTTP and WebRTC) carry `height` and `pow`
  (`"blake3-chain"` or `"walk"`).
- **Stratum `mining.notify`** carries the job's height as a fifth parameter.
  - `midstate miner` picks the algorithm from it.
  - A pool that sends four parameters is treated as a BLAKE3-chain pool.
- **The pool's HTTP and WebRTC templates** carry `height` and `pow`.
- **GPU backend.** The miner's wgpu kernel is the BLAKE3 chain's; a Walk
  kernel for wgpu exists for measuring (`bench-pow --gpu`, see
  [Any GPU through wgpu](#any-gpu-through-wgpu-the-pi-5s-first)). Walk mines on
  NVIDIA GPUs through CUDA (`src/core/cuda_walk.rs`, the benchmark's kernels
  loaded through the driver): `Auto` mines on them and the CPU threads,
  `--backend gpu` on them alone, and with no usable GPU both mine on the CPU,
  saying why once. Each GPU must reproduce the known answers before it
  mines, and the CPU re-checks every hash it reports. MANUAL.md, section 6.
  The first PTX failed that check on every GPU (a Quadro M6000 was the first
  to run it): clang 18 had compiled the walk's 64-bit rotate to two 64-bit
  shifts without reducing the amount mod 64, and PTX clamps shifts of 64 or
  more to zero, so the rotate gave 0. The kernels now build the rotate from
  32-bit shifts (`walk_core.h`'s `rotr64`), and CI runs the PTX itself
  (below).
- **Merged mining with midwimble.**
  - The pool reads `pow` from midwimble's `/merge/job` (absent means
    BLAKE3 chain).
  - It merges only when the two match. Otherwise it mines midstate alone and
    says why on the dashboard.
- **Share rate.**
  - A core makes ~20× more Walk attempts per second than chain attempts, so
    at the same `--share-bits` shares arrive ~20× as often.
  - Each share is ~20× cheaper to verify, so verification CPU stays similar.
    Message rate does not.
  - Consider adding 4 to `--share-bits` at the switch.
- **`midstate bench-pow [--seconds N] [--threads T]`** measures Walk (one
  thread and all threads, with the walks per thread the miner chooses) and
  the BLAKE3 chain on this machine. With `--gpu` it measures the Walk on a
  GPU instead, and with `--interleave` 1 to 4 walks stepped in turn per
  thread, each way of writing the step.

## Browser wallet

- `wasm-wallet/src/lib.rs` exports `search_nonces_pow(pow, …)` and
  `WebWallet.build_solo_extension_pow(pow, …)`.
- The miner worker passes each template's `pow` through and sizes its
  batches by time.
- The browser GPU path runs only for BLAKE3-chain templates.
- `wasm-wallet/pkg` is rebuilt with both functions (v6.0.0). The Pages
  deploy rebuilds it from source anyway. A `pkg/` older than the Walk keeps
  mining the chain through `search_nonces` and refuses Walk templates with an
  error.
- WebAssembly has no AES instructions, so the WASM Walk computes AES with
  tables. It reproduces the golden vectors, and one thread runs about 270
  attempts/s under Node on a 2-core cloud VM.

## SDK

`midstate-sdk` 2.10.0 (`sdk/`) mines whatever a template's `pow` says, solo
and pooled, single- and multi-threaded:

- `Miner` reads `pow` from node templates and pool jobs. A template without
  one means the BLAKE3 chain, and an unknown name is refused.
- A custom `searchFn` gets `pow` as its fifth argument.
- `MinerPool` workers restart when `pow` changes.
- `sdk/pkg` and `sdk/dist` are rebuilt.

`sdk/CHANGES.md` has the details, including a fix for worker loops that
outlived their template.

## Before release

1. **Re-measure the network's chain attempt rate**, and estimate the CPU Walk
   rate expected at the switch.
   - Set `WALK_TARGET_SHIFT = round(log2(chain rate ÷ walk rate))`.
   - At ~9.2M chain attempts/s (height ~354,000), 3 assumes ~1.1M Walk
     attempts/s.
2. ~~**Benchmark a GPU port.**~~ Done with
   [`docs/walk-reference/cuda/`](walk-reference/cuda/README.md) on four GPUs,
   the fastest an H100 (see [GPUs](#gpus)). The newest HBM cards (H200,
   B200) are predicted, not measured.
3. ~~**Run `midstate bench-pow` on a Pi 5.**~~ Done: 1,545 attempts/s on four
   cores at 3.1 GHz (1,383 at 2.8 GHz).
4. **Confirm `WALK_ACTIVATION_HEIGHT`** leaves operators enough time.
   - 371,500 is ~6.6 days after height ~362,060 (moved forward from
     420,000 in v6.2.0; v6.0.0 to v6.1.1 still switch there).
   - Midwimble mirrors it as `MIDSTATE_WALK_ACTIVATION_HEIGHT` and
     `MIDSTATE_WALK_TARGET_SHIFT`. Both must change together.
5. ~~Rebuild `wasm-wallet/pkg`, update the SDK miner, rebuild `sdk/pkg` and
   `sdk/dist`.~~ Done in v6.0.0 (see [SDK](#sdk)).
6. **Third-party miners** (MidstateAxe firmware, GPU miners, scripts) must
   switch to Walk at the height. They can tell from the template's `pow`.

## Checking an implementation

`docs/walk-reference/` holds two implementations written from the
definition rather than from the Rust, and a GPU benchmark:

- **Python.** Its AES S-box is derived from GF(2^8) inversion.
- **C.** It uses the ARMv8 AES instructions and runs under
  `qemu-aarch64 -cpu cortex-a76`.
- **C++ and CUDA** (`cuda/`), ported from the C. CI
  (`.github/workflows/walk-reference.yml`) checks its known answers against
  `walk.rs`, runs its CPU self-test and compiles its kernels. The miner's
  kernels (`src/core/cuda/walk_miner.cu`) are these; CI checks the PTX the
  miner embeds is built from them, and runs them on the CPU as the miner
  launches them, against the known answers: compiled from their source
  (`host_check.cpp`), and as the PTX itself, translated into C instruction
  by instruction with PTX's rules (`ptx_to_c.py`, `ptx_check.cpp`). Only the
  second sees what clang made of the source. The broken rotate passed the
  first and fails the second, with the same wrong hash the M6000 gave.

All three reproduce the golden vectors in `walk.rs`'s tests. So does the Rust code:

- on x86 AES-NI and on its portable path;
- on ARMv8 AES instructions, with `walk.rs`'s tests cross-built for
  `aarch64-unknown-linux-gnu` and run under `qemu-aarch64 -cpu cortex-a76`
  (`backend_name()` reports `armv8-crypto`);
- compiled to WebAssembly (the SDK's `test-miner.mjs`).

Running `cargo test --lib core::walk` on a real Pi 5 is still worth doing
once.
