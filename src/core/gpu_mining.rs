#![cfg(not(target_arch = "wasm32"))]
//! GPU mining backend (wgpu / WGSL) for the PoW extension chain.
//!
//! This mirrors the CPU search in `simd_mining` / `extension`, but spreads the
//! nonce search across thousands of GPU threads. The hash itself is unchanged
//! and bit-identical to `create_extension` (enforced by [`GpuMiner::self_test`]).
//!
//! ## Why the search is checkpointed across many dispatches
//! With `EXTENSION_ITERATIONS = 1_000_000`, one nonce is ~1e6 sequential BLAKE3
//! compressions. A single GPU thread therefore takes hundreds of milliseconds
//! to seconds to finish one chain. If a kernel ran a whole chain in one launch
//! it would exceed the OS GPU watchdog (TDR, ~2s on desktops with a display)
//! and be killed. Instead we keep each nonce's 32-byte chaining state in a GPU
//! buffer and advance it `ITERS_PER_DISPATCH` iterations per dispatch, polling
//! `cancel` and updating `hash_counter` between dispatches.
//!
//! ## Safety net
//! The kernel only *surfaces candidate nonces*. Every candidate is recomputed
//! on the CPU with `create_extension` and re-checked against the target before
//! it is returned, so a buggy/non-deterministic driver can never cause an
//! invalid block/share to be accepted (it could only cost throughput).

// Confirmed against the real crate: `Extension` + `EXTENSION_ITERATIONS` live in
// `core::types`; `MiningResult`, `create_extension`, and `mine_extension` (the CPU
// fallback) live in `core::extension`. If `EXTENSION_ITERATIONS` ever moves, this
// is the only line to touch.
use super::types::{Extension, EXTENSION_ITERATIONS};
use super::extension::{create_extension, mine_extension, MiningResult};
use anyhow::{anyhow, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use std::sync::{Arc, Mutex, OnceLock};

// ── Tunables ────────────────────────────────────────────────────────────────

/// Nonces ground per batch. **CRITICAL for finding blocks/shares:** a batch is
/// only tested against the target *after every nonce in it finishes the full
/// EXTENSION_ITERATIONS chain*. So the batch must **complete faster than the
/// mining job changes** — otherwise `cancel` fires mid-batch, the test phase
/// never runs, and nothing is ever surfaced (while the hash counter still
/// climbs, making it look like it's working). Batch time ≈ BATCH_NONCES ÷
/// (nonces/sec). At ~2k nonces/s, 8192 finishes in ~4s, which fits typical pool
/// job intervals. Keep `BATCH_NONCES ÷ your_nonces_per_sec` below your job/tip
/// interval: lower it if your pool pushes jobs very frequently; raise it for
/// solo or pure throughput on a big GPU. State buffer is BATCH_NONCES * 32 bytes
/// (256 KiB at 8192). Must stay ≥ a few thousand to keep the GPU saturated.
pub const BATCH_NONCES: u32 = 1 << 13; // 8,192

/// 32-byte iterations applied per `k_step` dispatch. Picked so each dispatch
/// stays well under the ~2s GPU watchdog even on slow GPUs. On a headless /
/// compute-only GPU (no display, no TDR) you can raise this to cut per-dispatch
/// overhead; lower it if you ever see device-lost / timeout errors.
pub const ITERS_PER_DISPATCH: u32 = 2_000;

/// When throttled (duty < 1.0), each dispatch is capped to this many iterations
/// so the GPU is held for only a few ms at a time, letting the desktop
/// compositor interleave frames. Full-speed mining keeps using ITERS_PER_DISPATCH.
const RESPONSIVE_ITERS: u32 = 384;

/// GPU duty cycle × 1000 (1000 = full speed). See [`set_gpu_duty`].
static GPU_DUTY_MILLI: AtomicU32 = AtomicU32::new(1000);

/// Set the GPU duty cycle, clamped to `0.02..=1.0` (1.0 = full speed). Values
/// below 1.0 insert idle gaps between dispatches so the GPU isn't pinned at 100%,
/// trading hashrate for desktop responsiveness / lower heat. Call once at startup
/// (e.g. from a `--gpu-duty` flag).
pub fn set_gpu_duty(duty: f32) {
    GPU_DUTY_MILLI.store((duty.clamp(0.02, 1.0) * 1000.0) as u32, Ordering::Relaxed);
}

/// Current duty cycle. The `GPU_MINE_DUTY` env var (if set and parseable) takes
/// precedence so it can be tuned live without a rebuild; otherwise the value from
/// [`set_gpu_duty`] (default 1.0) is used.
fn gpu_duty() -> f32 {
    if let Ok(s) = std::env::var("GPU_MINE_DUTY") {
        if let Ok(v) = s.parse::<f32>() {
            return v.clamp(0.02, 1.0);
        }
    }
    GPU_DUTY_MILLI.load(Ordering::Relaxed) as f32 / 1000.0
}

const MAX_WINNERS: u32 = 256;
const WINNERS_BYTES: u64 = 16 + (MAX_WINNERS as u64) * 4 * 3;
const SELFTEST_N: u32 = 8;

const SHADER: &str = r#"// BLAKE3 mining kernel for the PoW extension chain.
//
// CONSENSUS-CRITICAL: every value below mirrors `create_extension` /
// `simd_mining::compress_4way` exactly. The per-nonce result MUST be
// bit-identical to the scalar reference or the miner produces rejected
// blocks/shares. The compression body is machine-generated from the same
// MSG_SCHEDULE the CPU path uses.
//
//   per nonce:  h = blake3_40(midstate || nonce_le8)        // block_len = 40
//               repeat EXTENSION_ITERATIONS:  h = blake3_32(h)  // block_len = 32
//   the chaining value fed into every compression is the IV (each step is a
//   fresh BLAKE3 of a 32-byte input, NOT a running hash).

const IV0: u32 = 0x6A09E667u;
const IV1: u32 = 0xBB67AE85u;
const IV2: u32 = 0x3C6EF372u;
const IV3: u32 = 0xA54FF53Au;
const IV4: u32 = 0x510E527Fu;
const IV5: u32 = 0x9B05688Cu;
const IV6: u32 = 0x1F83D9ABu;
const IV7: u32 = 0x5BE0CD19u;
const FLAGS: u32 = 11u; // CHUNK_START | CHUNK_END | ROOT  (1 | 2 | 8)

struct Params {
    midstate: array<u32, 8>,  // u32::from_le_bytes of each 4-byte group of the 32-byte midstate
    tgt:      array<u32, 8>,  // from_be_bytes of each 4-byte group of the 32-byte target ('target' is a WGSL reserved word)
    pool:     array<u32, 8>,  // same encoding as target; used only when has_pool != 0
    base_lo:  u32,
    base_hi:  u32,
    n_nonces: u32,
    iters:    u32,            // k_step: how many 32-byte iterations to apply this dispatch
    has_pool: u32,
    pad0: u32, pad1: u32, pad2: u32,
};

struct Winners {
    count: atomic<u32>,
    cap: u32,
    pad0: u32, pad1: u32,
    nonce_lo: array<u32, 256>,
    nonce_hi: array<u32, 256>,
    kind:     array<u32, 256>,   // 0 = block, 1 = share
};

@group(0) @binding(0) var<storage, read>       P:     Params;
@group(0) @binding(1) var<storage, read_write> state: array<u32>;   // n_nonces * 8 chaining words
@group(0) @binding(2) var<storage, read_write> out:   Winners;

fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// Reverse the 4 bytes of a word: turns a little-endian hash word into the
// big-endian key whose numeric order matches the [u8;32] lexicographic order.
fn bswap(x: u32) -> u32 {
    return ((x & 0xFFu) << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | ((x >> 24u) & 0xFFu);
}

fn compress(m: array<u32,16>, block_len: u32) -> array<u32,8> {
    var v0 = IV0; var v1 = IV1; var v2 = IV2; var v3 = IV3;
    var v4 = IV4; var v5 = IV5; var v6 = IV6; var v7 = IV7;
    var v8 = IV0; var v9 = IV1; var v10 = IV2; var v11 = IV3;
    var v12 = 0u; var v13 = 0u; var v14 = block_len; var v15 = FLAGS;
  // round 0
  v0 = v0 + v4 + m[0]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[1]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[2]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[3]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[4]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[5]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[6]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[7]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[8]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[9]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[10]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[11]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[12]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[13]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[14]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[15]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 1
  v0 = v0 + v4 + m[2]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[6]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[3]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[10]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[7]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[0]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[4]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[13]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[1]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[11]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[12]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[5]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[9]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[14]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[15]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[8]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 2
  v0 = v0 + v4 + m[3]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[4]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[10]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[12]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[13]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[2]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[7]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[14]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[6]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[5]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[9]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[0]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[11]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[15]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[8]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[1]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 3
  v0 = v0 + v4 + m[10]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[7]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[12]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[9]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[14]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[3]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[13]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[15]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[4]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[0]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[11]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[2]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[5]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[8]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[1]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[6]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 4
  v0 = v0 + v4 + m[12]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[13]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[9]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[11]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[15]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[10]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[14]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[8]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[7]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[2]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[5]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[3]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[0]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[1]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[6]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[4]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 5
  v0 = v0 + v4 + m[9]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[14]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[11]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[5]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[8]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[12]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[15]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[1]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[13]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[3]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[0]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[10]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[2]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[6]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[4]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[7]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
  // round 6
  v0 = v0 + v4 + m[11]; v12 = rotr(v12 ^ v0, 16u); v8 = v8 + v12; v4 = rotr(v4 ^ v8, 12u);
  v0 = v0 + v4 + m[15]; v12 = rotr(v12 ^ v0, 8u);  v8 = v8 + v12; v4 = rotr(v4 ^ v8, 7u);
  v1 = v1 + v5 + m[5]; v13 = rotr(v13 ^ v1, 16u); v9 = v9 + v13; v5 = rotr(v5 ^ v9, 12u);
  v1 = v1 + v5 + m[0]; v13 = rotr(v13 ^ v1, 8u);  v9 = v9 + v13; v5 = rotr(v5 ^ v9, 7u);
  v2 = v2 + v6 + m[1]; v14 = rotr(v14 ^ v2, 16u); v10 = v10 + v14; v6 = rotr(v6 ^ v10, 12u);
  v2 = v2 + v6 + m[9]; v14 = rotr(v14 ^ v2, 8u);  v10 = v10 + v14; v6 = rotr(v6 ^ v10, 7u);
  v3 = v3 + v7 + m[8]; v15 = rotr(v15 ^ v3, 16u); v11 = v11 + v15; v7 = rotr(v7 ^ v11, 12u);
  v3 = v3 + v7 + m[6]; v15 = rotr(v15 ^ v3, 8u);  v11 = v11 + v15; v7 = rotr(v7 ^ v11, 7u);
  v0 = v0 + v5 + m[14]; v15 = rotr(v15 ^ v0, 16u); v10 = v10 + v15; v5 = rotr(v5 ^ v10, 12u);
  v0 = v0 + v5 + m[10]; v15 = rotr(v15 ^ v0, 8u);  v10 = v10 + v15; v5 = rotr(v5 ^ v10, 7u);
  v1 = v1 + v6 + m[2]; v12 = rotr(v12 ^ v1, 16u); v11 = v11 + v12; v6 = rotr(v6 ^ v11, 12u);
  v1 = v1 + v6 + m[12]; v12 = rotr(v12 ^ v1, 8u);  v11 = v11 + v12; v6 = rotr(v6 ^ v11, 7u);
  v2 = v2 + v7 + m[3]; v13 = rotr(v13 ^ v2, 16u); v8 = v8 + v13; v7 = rotr(v7 ^ v8, 12u);
  v2 = v2 + v7 + m[4]; v13 = rotr(v13 ^ v2, 8u);  v8 = v8 + v13; v7 = rotr(v7 ^ v8, 7u);
  v3 = v3 + v4 + m[7]; v14 = rotr(v14 ^ v3, 16u); v9 = v9 + v14; v4 = rotr(v4 ^ v9, 12u);
  v3 = v3 + v4 + m[13]; v14 = rotr(v14 ^ v3, 8u);  v9 = v9 + v14; v4 = rotr(v4 ^ v9, 7u);
    return array<u32,8>(v0 ^ v8, v1 ^ v9, v2 ^ v10, v3 ^ v11, v4 ^ v12, v5 ^ v13, v6 ^ v14, v7 ^ v15);
}

fn nonce_for(gid: u32) -> vec2<u32> {
    let lo = P.base_lo + gid;          // gid < n_nonces <= 2^18, so at most one carry
    var carry = 0u;
    if (lo < P.base_lo) { carry = 1u; }
    let hi = P.base_hi + carry;
    return vec2<u32>(lo, hi);
}

fn first_compress(gid: u32) -> array<u32,8> {
    var m: array<u32,16>;
    m[0] = P.midstate[0]; m[1] = P.midstate[1]; m[2] = P.midstate[2]; m[3] = P.midstate[3];
    m[4] = P.midstate[4]; m[5] = P.midstate[5]; m[6] = P.midstate[6]; m[7] = P.midstate[7];
    let n = nonce_for(gid);
    m[8] = n.x; m[9] = n.y;
    m[10] = 0u; m[11] = 0u; m[12] = 0u; m[13] = 0u; m[14] = 0u; m[15] = 0u;
    return compress(m, 40u);
}

fn iterate(h: array<u32,8>) -> array<u32,8> {
    var m: array<u32,16>;
    m[0] = h[0]; m[1] = h[1]; m[2] = h[2]; m[3] = h[3];
    m[4] = h[4]; m[5] = h[5]; m[6] = h[6]; m[7] = h[7];
    m[8] = 0u; m[9] = 0u; m[10] = 0u; m[11] = 0u; m[12] = 0u; m[13] = 0u; m[14] = 0u; m[15] = 0u;
    return compress(m, 32u);
}

// final_hash[u8;32] < ref ?  (byte 0 most significant), unrolled to avoid
// dynamic indexing into value arrays.
fn lt8(h: array<u32,8>, r: array<u32,8>) -> bool {
    var k: u32;
    k = bswap(h[0]); if (k < r[0]) { return true; } if (k > r[0]) { return false; }
    k = bswap(h[1]); if (k < r[1]) { return true; } if (k > r[1]) { return false; }
    k = bswap(h[2]); if (k < r[2]) { return true; } if (k > r[2]) { return false; }
    k = bswap(h[3]); if (k < r[3]) { return true; } if (k > r[3]) { return false; }
    k = bswap(h[4]); if (k < r[4]) { return true; } if (k > r[4]) { return false; }
    k = bswap(h[5]); if (k < r[5]) { return true; } if (k > r[5]) { return false; }
    k = bswap(h[6]); if (k < r[6]) { return true; } if (k > r[6]) { return false; }
    k = bswap(h[7]); if (k < r[7]) { return true; } if (k > r[7]) { return false; }
    return false;
}

fn load_state(gid: u32) -> array<u32,8> {
    let b = gid * 8u;
    var h: array<u32,8>;
    h[0] = state[b + 0u]; h[1] = state[b + 1u]; h[2] = state[b + 2u]; h[3] = state[b + 3u];
    h[4] = state[b + 4u]; h[5] = state[b + 5u]; h[6] = state[b + 6u]; h[7] = state[b + 7u];
    return h;
}

fn store_state(gid: u32, h: array<u32,8>) {
    let b = gid * 8u;
    state[b + 0u] = h[0]; state[b + 1u] = h[1]; state[b + 2u] = h[2]; state[b + 3u] = h[3];
    state[b + 4u] = h[4]; state[b + 5u] = h[5]; state[b + 6u] = h[6]; state[b + 7u] = h[7];
}

@compute @workgroup_size(64)
fn k_init(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    store_state(gid, first_compress(gid));
}

@compute @workgroup_size(64)
fn k_step(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    var h = load_state(gid);
    for (var i = 0u; i < P.iters; i = i + 1u) {
        h = iterate(h);
    }
    store_state(gid, h);
}

@compute @workgroup_size(64)
fn k_test(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let gid = gid3.x;
    if (gid >= P.n_nonces) { return; }
    let h = load_state(gid);
    var kind = 0xFFFFFFFFu;
    if (lt8(h, P.tgt)) {
        kind = 0u;
    } else if (P.has_pool != 0u && lt8(h, P.pool)) {
        kind = 1u;
    }
    if (kind != 0xFFFFFFFFu) {
        let idx = atomicAdd(&out.count, 1u);
        if (idx < out.cap) {
            let n = nonce_for(gid);
            out.nonce_lo[idx] = n.x;
            out.nonce_hi[idx] = n.y;
            out.kind[idx] = kind;
        }
    }
}
"#;

// ── Param block mirrored 1:1 by the WGSL `Params` struct (std430, 128 bytes) ──

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    midstate: [u32; 8],
    target:   [u32; 8],
    pool:     [u32; 8],
    base_lo:  u32,
    base_hi:  u32,
    n_nonces: u32,
    iters:    u32,
    has_pool: u32,
    pad0: u32, pad1: u32, pad2: u32,
}
const ITERS_FIELD_OFFSET: u64 = 96 + 3 * 4; // byte offset of `iters` within Params

fn words_le(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]]);
    }
    w
}

fn words_be(b: &[u8; 32]) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_be_bytes([b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]]);
    }
    w
}

// ── The reusable GPU context ──────────────────────────────────────────────────

/// Identifies a mining job. Surplus winners from one job must never be served to
/// a different one, so the stash is invalidated whenever this changes.
type JobKey = ([u8; 32], [u8; 32], Option<[u8; 32]>); // (midstate, target, pool_target)

/// Guarded mutable state. The single `Mutex` does double duty: it serializes all
/// GPU dispatches (the buffers are only touched while it's held) *and* protects
/// the surplus-winner queue. Concurrent callers (e.g. a stratum job handing off
/// to its successor) block here and run one at a time, which is correct — a
/// single GPU can't usefully run two independent searches at once anyway.
struct MinerState {
    job: Option<JobKey>,
    pending: VecDeque<MiningResult>,
}

pub struct GpuMiner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_init: wgpu::ComputePipeline,
    pipe_step: wgpu::ComputePipeline,
    pipe_test: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    winners_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    adapter_name: String,
    state: Mutex<MinerState>,
}

impl GpuMiner {
    /// Build the GPU context. Cheap-ish but not free (device init + shader
    /// compile); construct once and reuse across blocks.
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self> {
        // wgpu 29: InstanceDescriptor lost `Default`; this constructor reads
        // backend/power prefs from env vars (e.g. WGPU_BACKEND=vulkan) and needs
        // no window/display handle (we're headless compute).
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        // VERSION: wgpu >=24 returns Result here. On <=23 it returns Option ->
        // use `.ok_or_else(|| anyhow!("no GPU adapter"))?` instead of `map_err`.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| anyhow!("no suitable GPU adapter: {e:?}"))?;

        let adapter_name = adapter.get_info().name;

        // VERSION: wgpu >=24 takes a single `&DeviceDescriptor` and returns
        // Result. The `trace` field exists on >=~25; delete it on older
        // versions. On <=23 the signature is
        // `request_device(&desc, None /* trace path */)`.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("pow-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(), // wgpu 27+
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| anyhow!("request_device failed: {e:?}"))?;

        // Capture shader-validation errors instead of letting wgpu's default
        // handler abort the process; on failure we return Err and fall back to CPU.
        // wgpu 29: push_error_scope returns an RAII guard whose .pop() yields the error.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pow-blake3"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        if let Some(e) = scope.pop().await {
            return Err(anyhow!("shader validation failed: {e}"));
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pow-bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pow-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)], // wgpu 29: Option-wrapped
            immediate_size: 0,                               // wgpu 29: replaces push_constant_ranges
        });

        let make_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipe_init = make_pipe("k_init");
        let pipe_step = make_pipe("k_step");
        let pipe_test = make_pipe("k_test");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state"),
            size: (BATCH_NONCES as u64) * 8 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let winners_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("winners"),
            size: WINNERS_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: WINNERS_BYTES, // also big enough for the self-test state copy
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pow-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: winners_buf.as_entire_binding() },
            ],
        });

        Ok(Self {
            device, queue, pipe_init, pipe_step, pipe_test, bind_group,
            params_buf, state_buf, winners_buf, readback_buf, adapter_name,
            state: Mutex::new(MinerState { job: None, pending: VecDeque::new() }),
        })
    }

    pub fn adapter_name(&self) -> &str { &self.adapter_name }

    fn dispatch(&self, pipe: &wgpu::ComputePipeline, groups: u32) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &self.bind_group, &[]);
            cp.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit([enc.finish()]);
    }

    fn wait(&self) {
        // wgpu 29: PollType::Wait now carries { submission_index, timeout };
        // wait_indefinitely() is the old "block until all submitted work is done".
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Copy `len` bytes out of `winners_buf` (or state, see callers) into the
    /// readback buffer and return them. Assumes the source copy was issued by
    /// the caller via `copy_buffer_to_buffer` before calling.
    fn map_readback(&self, len: u64) -> Vec<u8> {
        let slice = self.readback_buf.slice(0..len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.wait();
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let out = data.to_vec();
        drop(data);
        self.readback_buf.unmap();
        out
    }

    fn groups(n: u32) -> u32 { (n + 63) / 64 }

    /// Run one batch of `BATCH_NONCES` nonces starting at `base`, applying the
    /// full `EXTENSION_ITERATIONS` chain, then test against target/pool. Returns
    /// the list of (nonce, kind) candidates the GPU surfaced. Returns `None` if
    /// `cancel` fired partway through.
    fn run_batch(
        &self,
        params: &mut Params,
        base: u64,
        n_nonces: u32,
        cancel: &AtomicBool,
        hash_counter: &AtomicU64,
        collect_winners: bool,
    ) -> Option<Vec<(u64, u32)>> {
        params.base_lo = base as u32;
        params.base_hi = (base >> 32) as u32;
        params.n_nonces = n_nonces;
        params.iters = 0;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&*params));
        self.queue.write_buffer(&self.winners_buf, 0, bytemuck::cast_slice(&[0u32, MAX_WINNERS]));

        let groups = Self::groups(n_nonces);
        self.dispatch(&self.pipe_init, groups);
        self.wait();

        let total = EXTENSION_ITERATIONS;
        // Throttle real mining only (never the self-test). duty < 1.0 -> shorter
        // dispatches with idle gaps so the GPU isn't pinned and the desktop stays
        // responsive. `collect_winners` is false only for the self-test.
        let duty = if collect_winners { gpu_duty() } else { 1.0 };
        let throttling = duty < 0.999;
        let chunk = if throttling { ITERS_PER_DISPATCH.min(RESPONSIVE_ITERS) } else { ITERS_PER_DISPATCH };

        let mut remaining = total;
        while remaining > 0 {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let k = remaining.min(chunk as u64) as u32;
            self.queue.write_buffer(&self.params_buf, ITERS_FIELD_OFFSET, &k.to_le_bytes());
            let t0 = Instant::now();
            self.dispatch(&self.pipe_step, groups);
            self.wait(); // bound watchdog exposure + keep cancel responsive
            if throttling {
                // active fraction ≈ duty  ->  idle = work * (1/duty - 1)
                let factor = (1.0 / duty as f64 - 1.0).min(32.0);
                std::thread::sleep(t0.elapsed().mul_f64(factor));
            }
            remaining -= k as u64;
            // Count nonces (matching the CPU counter semantics), smoothed across
            // the batch's dispatches.
            let add = (n_nonces as u64).saturating_mul(k as u64) / total;
            hash_counter.fetch_add(add, Ordering::Relaxed);
        }

        if !collect_winners {
            return Some(Vec::new());
        }

        self.dispatch(&self.pipe_test, groups);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.winners_buf, 0, &self.readback_buf, 0, WINNERS_BYTES);
        self.queue.submit([enc.finish()]);

        let bytes = self.map_readback(WINNERS_BYTES);
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).min(MAX_WINNERS);
        let lo_off = 16usize;
        let hi_off = lo_off + (MAX_WINNERS as usize) * 4;
        let kind_off = hi_off + (MAX_WINNERS as usize) * 4;
        let mut winners = Vec::with_capacity(count as usize);
        for j in 0..count as usize {
            let lo = u32::from_le_bytes(bytes[lo_off+j*4..lo_off+j*4+4].try_into().unwrap());
            let hi = u32::from_le_bytes(bytes[hi_off+j*4..hi_off+j*4+4].try_into().unwrap());
            let kind = u32::from_le_bytes(bytes[kind_off+j*4..kind_off+j*4+4].try_into().unwrap());
            winners.push(((lo as u64) | ((hi as u64) << 32), kind));
        }
        Some(winners)
    }

    /// Mine until a block (or pool share) is found or `cancel` fires. Returns
    /// `None` only when cancelled before producing a hit. Every returned hit is
    /// re-verified on the CPU with `create_extension`, so the kernel is trusted
    /// only to *surface* candidates, never to accept them.
    ///
    /// A single GPU batch can produce many winners against an easy share target.
    /// The first is returned; the rest are stashed (keyed to this exact job) and
    /// handed back on subsequent calls with no further GPU work — which is what
    /// the stratum loop's "call again for the next share" pattern wants. The
    /// whole call holds `self.state`, so concurrent callers run one at a time.
    pub fn mine_gpu(
        &self,
        midstate: [u8; 32],
        target: [u8; 32],
        pool_target: Option<[u8; 32]>,
        cancel: Arc<AtomicBool>,
        hash_counter: Arc<AtomicU64>,
    ) -> Option<MiningResult> {
        let job: JobKey = (midstate, target, pool_target);
        // Serialize GPU access (and guard the stash) for the whole call.
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());

        // New job? Drop any stale surplus from a previous one.
        if st.job.as_ref() != Some(&job) {
            st.job = Some(job);
            st.pending.clear();
        }
        // Serve a stashed winner immediately, no GPU work.
        if let Some(hit) = st.pending.pop_front() {
            return Some(hit);
        }

        let (pool_words, has_pool) = match pool_target {
            Some(p) => (words_be(&p), 1u32),
            None => ([0u32; 8], 0u32),
        };
        let mut params = Params {
            midstate: words_le(&midstate),
            target: words_be(&target),
            pool: pool_words,
            base_lo: 0, base_hi: 0, n_nonces: BATCH_NONCES, iters: 0, has_pool,
            pad0: 0, pad1: 0, pad2: 0,
        };

        loop {
            if cancel.load(Ordering::Relaxed) {
                tracing::debug!("GPU mining cancelled");
                return None;
            }
            let base: u64 = rand::random();
            let winners = self.run_batch(&mut params, base, BATCH_NONCES, &cancel, &hash_counter, true)?;

            // CPU-verify every candidate; the CPU result is authoritative. Blocks
            // sort ahead of shares so a block is always returned first.
            let mut hits: Vec<MiningResult> = Vec::new();
            for (nonce, _kind) in winners {
                let final_hash = create_extension(midstate, nonce).final_hash;
                if final_hash < target {
                    hits.push(MiningResult::Block(Extension { nonce, final_hash }));
                } else if let Some(pt) = pool_target {
                    if final_hash < pt {
                        hits.push(MiningResult::Share(Extension { nonce, final_hash }));
                    }
                }
            }
            if hits.is_empty() {
                continue; // no winner this batch -> next batch, fresh random base
            }
            hits.sort_by_key(|h| matches!(h, MiningResult::Share(_))); // blocks (false) first

            let mut it = hits.into_iter();
            let first = it.next().unwrap();
            match &first {
                MiningResult::Block(e) => tracing::info!(
                    "GPU found valid block! nonce={} hash={} gpu={}",
                    e.nonce, hex::encode(e.final_hash), self.adapter_name),
                MiningResult::Share(e) => tracing::info!(
                    "GPU found valid pool share! nonce={} hash={}",
                    e.nonce, hex::encode(e.final_hash)),
            }
            // Stash the surplus for the next call on this same job.
            st.pending.extend(it);
            return Some(first);
        }
    }

    /// Prove the GPU reproduces `create_extension` bit-for-bit on the full
    /// 1,000,000-iteration chain. Runs a tiny batch and reads back the raw
    /// chaining state (which equals the final hash). Returns an error on any
    /// mismatch, so a broken driver never mines. Costs ~one chain of latency.
    pub fn self_test(&self) -> Result<()> {
        let midstate = [0xA5u8; 32]; // any fixed input; we compare GPU vs CPU on it
        let never = AtomicBool::new(false);
        let sink = AtomicU64::new(0);
        let base: u64 = 0;

        let mut params = Params {
            midstate: words_le(&midstate),
            target: [0u32; 8], pool: [0u32; 8],
            base_lo: 0, base_hi: 0, n_nonces: SELFTEST_N, iters: 0, has_pool: 0,
            pad0: 0, pad1: 0, pad2: 0,
        };
        // collect_winners = false: we read state directly instead.
        self.run_batch(&mut params, base, SELFTEST_N, &never, &sink, false)
            .ok_or_else(|| anyhow!("self-test batch was unexpectedly cancelled"))?;

        let state_bytes_len = (SELFTEST_N as u64) * 8 * 4;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.state_buf, 0, &self.readback_buf, 0, state_bytes_len);
        self.queue.submit([enc.finish()]);
        let bytes = self.map_readback(state_bytes_len);

        for gid in 0..SELFTEST_N as u64 {
            let expected = super::extension::create_extension(midstate, base + gid).final_hash;
            let mut got = [0u8; 32];
            for i in 0..8usize {
                let off = (gid as usize) * 32 + i * 4;
                let w = u32::from_le_bytes(bytes[off..off+4].try_into().unwrap());
                got[i*4..i*4+4].copy_from_slice(&w.to_le_bytes());
            }
            if got != expected {
                return Err(anyhow!(
                    "GPU self-test FAILED at nonce {gid}: kernel is not consensus-identical \
                     (gpu={} expected={}). Refusing to GPU-mine.",
                    hex::encode(got), hex::encode(expected)
                ));
            }
        }
        tracing::info!("GPU self-test passed on {} ({} nonces)", self.adapter_name, SELFTEST_N);
        Ok(())
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ── Process-wide lazy handle ──────────────────────────────────────────────────

/// Lazily constructed, self-tested, process-wide GPU handle. The result
/// (including "no usable GPU") is cached, so init is never retried per block.
/// Returns `None` when there's no GPU, the self-test failed, or
/// `MINER_DISABLE_GPU` is set; in every such case the caller should fall back to
/// its existing CPU miner. See the integration note for the call-site pattern.
pub fn shared() -> Option<&'static GpuMiner> {
    static SHARED: OnceLock<Option<GpuMiner>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            if std::env::var("MINER_DISABLE_GPU").map(|v| v != "0").unwrap_or(false) {
                tracing::info!("GPU mining disabled via MINER_DISABLE_GPU");
                return None;
            }
            match GpuMiner::new() {
                Ok(g) => match g.self_test() {
                    Ok(()) => {
                        tracing::info!("GPU mining enabled on {}", g.adapter_name());
                        Some(g)
                    }
                    Err(e) => {
                        tracing::warn!("GPU mining disabled (self-test failed): {e}");
                        None
                    }
                },
                Err(e) => {
                    tracing::info!("GPU mining disabled (no usable device): {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// `true` if a self-tested GPU backend is available for mining.
pub fn gpu_available() -> bool {
    shared().is_some()
}

/// Which mining backend `mine()` should use.
///
/// - `Auto` (default): prefer the GPU, silently fall back to CPU if none is usable.
/// - `Gpu`: prefer the GPU; if it genuinely can't initialize, warn and use CPU
///   (mining on a broken GPU is never worth producing rejected blocks).
/// - `Cpu`: always use the multithreaded CPU miner.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Backend {
    #[default]
    Auto = 0,
    Gpu = 1,
    Cpu = 2,
}

static BACKEND: AtomicU8 = AtomicU8::new(Backend::Auto as u8);

/// Set the process-wide mining backend. Call once at startup from your CLI/config
/// (e.g. in the `node` command handler) before mining begins. Cheap and
/// thread-safe; the choice is read by every `mine()` call.
pub fn set_backend(b: Backend) {
    BACKEND.store(b as u8, Ordering::Relaxed);
}

/// The currently selected backend.
pub fn backend() -> Backend {
    match BACKEND.load(Ordering::Relaxed) {
        1 => Backend::Gpu,
        2 => Backend::Cpu,
        _ => Backend::Auto,
    }
}

/// Drop-in replacement for `core::extension::mine_extension` — identical
/// signature and semantics. Honors [`set_backend`]: routes to the GPU under
/// `Auto`/`Gpu` when one is available and passed self-test, otherwise (or under
/// `Cpu`) calls the existing multithreaded CPU miner. To switch a call site
/// over, change exactly one path:
///
/// ```ignore
/// // before:
/// crate::core::extension::mine_extension(mining_hash, target, pool_target, threads, cancel, hash_counter)
/// // after:
/// crate::core::gpu_mining::mine(mining_hash, target, pool_target, threads, cancel, hash_counter)
/// ```
///
/// `threads` is forwarded to the CPU path and ignored by the GPU path (the GPU
/// is the parallelism). `MINER_DISABLE_GPU=1` also forces CPU regardless.
pub fn mine(
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    threads: usize,
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<AtomicU64>,
) -> Option<MiningResult> {
    let want_gpu = backend() != Backend::Cpu;
    if want_gpu {
        if let Some(g) = shared() {
            return g.mine_gpu(midstate, target, pool_target, cancel, hash_counter);
        }
        if backend() == Backend::Gpu {
            tracing::warn!("GPU backend requested but no usable GPU; mining on CPU");
        }
    }
    mine_extension(midstate, target, pool_target, threads, cancel, hash_counter)
}
