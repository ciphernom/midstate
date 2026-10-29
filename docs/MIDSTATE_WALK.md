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

### GPU estimate (not measured)

This is a model with rough assumptions, not a benchmark. Take an RTX 4090
with GPU L2 latency of about 250 cycles, a 64-bit divide at around 100
dependent instructions, and random 32-byte-sector VRAM traffic at about
40% of peak.

| Strategy | Estimated rate |
|---|---|
| Pads resident in L2 | ~5K attempts/s (latency-bound) |
| Pads streamed from VRAM | ~6K attempts/s (bandwidth-bound) |

That puts a 4090 around 1× a 16-core desktop per device and 2–3× worse per
watt. For comparison, the BLAKE3 chain puts it orders of magnitude ahead.

**This estimate must be confirmed with a real GPU port before activation.**
Write a CUDA or wgpu Walk kernel, measure it, and compare it against
`bench-pow` on a Pi 5 and a desktop.

The knobs, if it comes out worse than expected:
- `WALK_LINES`: a bigger pad pushes GPUs out of L2, but must still fit a
  Pi 5's 512 KiB L2.
- Loads per step: more loads add VRAM traffic.
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
- **`wasm-wallet/pkg` must be rebuilt (`wasm-pack build`) before
  deploying.** Until it is, the worker keeps mining the chain through the old
  `search_nonces` and refuses Walk templates with an error.

## Before release

1. **Re-measure the network's chain attempt rate**, and estimate the CPU Walk
   rate expected at the switch.
   - Set `WALK_TARGET_SHIFT = round(log2(chain rate ÷ walk rate))`.
   - At ~9.2M chain attempts/s (height ~354,000), 3 assumes ~1.1M Walk
     attempts/s.
2. **Benchmark a GPU port** (see the GPU estimate above).
3. **Run `midstate bench-pow` on a Pi 5** (`--release`, cross-compiled with
   `.cargo/config.toml`'s `cortex-a76` target).
4. **Confirm `WALK_ACTIVATION_HEIGHT`** leaves operators enough time.
   - 420,000 is ~46 days after height ~354,000.
   - Midwimble mirrors it as `MIDSTATE_WALK_ACTIVATION_HEIGHT` and
     `MIDSTATE_WALK_TARGET_SHIFT`. Both must change together.
5. **Rebuild `wasm-wallet/pkg`.** Update the SDK miner (`sdk/src/miner*.js`)
   to pass `template.pow` to `search_nonces_pow`, then rebuild `sdk/pkg` and
   `sdk/dist`.
6. **Third-party miners** (MidstateAxe firmware, GPU miners, scripts) must
   switch to Walk at the height. They can tell from the template's `pow`.

## Checking an implementation

`docs/walk-reference/` holds two implementations written from the
definition rather than from the Rust:

- **Python.** Its AES S-box is derived from GF(2^8) inversion.
- **C.** It uses the ARMv8 AES instructions and runs under
  `qemu-aarch64 -cpu cortex-a76`.

Both reproduce the golden vectors in `walk.rs`'s tests. So does the Rust code
on x86 AES-NI and on its portable path.
