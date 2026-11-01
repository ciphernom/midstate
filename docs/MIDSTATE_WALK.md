# Midstate Walk

Midstate Walk is the proof of work from block `WALK_ACTIVATION_HEIGHT`
(420,000) on. It replaces the 1,000,000-step BLAKE3 chain, which GPUs
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
| Activation | height 420,000 | height 20 |
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

## What changed in consensus

- **Proof of work by height.**
  - `pow_algo_at(height)` picks the algorithm.
  - `create_extension_at` and `verify_extension_at` route by it.
  - `apply_batch` verifies each block at its real height, so history below
    420,000 still validates with the BLAKE3 chain.
- **Difficulty.**
  - From 420,000, ASERT's integer shift gets `WALK_TARGET_SHIFT` added. This
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
- **GPU backend.** `--backend gpu` and `Auto` mine Walk on the CPU, because
  the wgpu kernel is the BLAKE3 chain's. A warning says so once.
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
  thread and all threads) and the BLAKE3 chain on this machine.

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
   - 420,000 is ~46 days after height ~354,000.
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
  `walk.rs`, runs its CPU self-test and compiles its kernels.

All three reproduce the golden vectors in `walk.rs`'s tests. So does the Rust code:

- on x86 AES-NI and on its portable path;
- on ARMv8 AES instructions, with `walk.rs`'s tests cross-built for
  `aarch64-unknown-linux-gnu` and run under `qemu-aarch64 -cpu cortex-a76`
  (`backend_name()` reports `armv8-crypto`);
- compiled to WebAssembly (the SDK's `test-miner.mjs`).

Running `cargo test --lib core::walk` on a real Pi 5 is still worth doing
once.
