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
use anyhow::{anyhow, bail, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use std::sync::{Arc, Mutex, OnceLock};

// ── Tunables ────────────────────────────────────────────────────────────────
//
// All of these can be overridden at runtime from a `GPU_OC_SETTINGS.toml` file
// (path overridable via the GPU_OC_SETTINGS env var) without recompiling. The
// file is optional; if it's absent or unparseable these defaults are used.
//
//     # GPU_OC_SETTINGS.toml
//     batch_nonces       = 24576   # nonces per batch (GPU saturation ↔ share latency)
//     iters_per_dispatch = 2000    # chain steps per dispatch (host round-trips ↔ watchdog)
//     responsive_iters   = 384     # dispatch size while throttled (duty < 1.0)
//     duty               = 1.0     # 0.02..=1.0 fraction of time the GPU works

const DEFAULT_BATCH_NONCES: u32 = 1 << 13; // 8,192
const DEFAULT_ITERS_PER_DISPATCH: u32 = 2_000;
const DEFAULT_RESPONSIVE_ITERS: u32 = 384;
const DEFAULT_DUTY: f32 = 1.0;

/// Runtime-tunable GPU knobs, loaded once from `GPU_OC_SETTINGS.toml`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct GpuSettings {
    /// Nonces ground per batch. **Critical for finding blocks/shares:** a batch
    /// is only tested after every nonce finishes the full EXTENSION_ITERATIONS
    /// chain, so it must *complete* faster than the mining job changes
    /// (≈ batch_nonces ÷ nonces_per_sec < job interval) or nothing is surfaced.
    /// Bigger = better GPU saturation but longer batches; keep it ≥ a few
    /// thousand to keep the GPU busy. State buffer is batch_nonces × 32 bytes.
    pub batch_nonces: u32,
    /// Chain steps per GPU dispatch. Higher cuts host↔GPU round-trips but each
    /// dispatch holds the GPU longer (watchdog / display-freeze risk).
    pub iters_per_dispatch: u32,
    /// Dispatch size used while throttled (duty < 1.0); smaller = smoother desktop.
    pub responsive_iters: u32,
    /// Fraction of time the GPU works, `0.02..=1.0`. Overridden by the
    /// `GPU_MINE_DUTY` env var and by `set_gpu_duty()` when those are set.
    pub duty: f32,
    /// Force a specific GPU by case-insensitive name substring, e.g. "NVIDIA",
    /// "Quadro", "RTX 4080". Overrides automatic selection (which prefers the
    /// discrete card). Also settable via the `WGPU_ADAPTER_NAME` env var. Unset =
    /// automatic.
    pub adapter: Option<String>,
    /// Maximum number of GPUs to mine on simultaneously. `0` or unset = all
    /// usable devices. Also settable via the `MINER_GPU_DEVICES` env var. Useful
    /// to leave one card free for the desktop on a multi-GPU workstation.
    pub max_devices: Option<usize>,
}

impl Default for GpuSettings {
    fn default() -> Self {
        Self {
            batch_nonces: DEFAULT_BATCH_NONCES,
            iters_per_dispatch: DEFAULT_ITERS_PER_DISPATCH,
            responsive_iters: DEFAULT_RESPONSIVE_ITERS,
            duty: DEFAULT_DUTY,
            adapter: None,
            max_devices: None,
        }
    }
}

impl GpuSettings {
    fn sanitized(mut self) -> Self {
        self.batch_nonces = self.batch_nonces.clamp(64, 1 << 24);
        self.iters_per_dispatch = self.iters_per_dispatch.max(1);
        self.responsive_iters = self.responsive_iters.max(1);
        self.duty = self.duty.clamp(0.02, 1.0);
        self
    }
}

/// Process-wide GPU settings, loaded once from `GPU_OC_SETTINGS.toml` (or the
/// path in the `GPU_OC_SETTINGS` env var). A missing/invalid file → defaults.
fn settings() -> &'static GpuSettings {
    static S: OnceLock<GpuSettings> = OnceLock::new();
    S.get_or_init(|| {
        let path =
            std::env::var("GPU_OC_SETTINGS").unwrap_or_else(|_| "GPU_OC_SETTINGS.toml".to_string());
        match std::fs::read_to_string(&path) {
            Ok(text) => match toml::from_str::<GpuSettings>(&text) {
                Ok(s) => {
                    let s = s.sanitized();
                    tracing::info!("loaded GPU OC settings from {path}: {s:?}");
                    s
                }
                Err(e) => {
                    tracing::warn!("failed to parse {path} ({e}); using GPU defaults");
                    GpuSettings::default()
                }
            },
            Err(_) => GpuSettings::default(), // no file -> silent defaults
        }
    })
}

/// GPU duty × 1000 set via [`set_gpu_duty`]; 0 = unset (fall back to TOML/default).
static GPU_DUTY_MILLI: AtomicU32 = AtomicU32::new(0);

/// Set the GPU duty cycle, clamped to `0.02..=1.0` (1.0 = full speed). Values
/// below 1.0 insert idle gaps between dispatches so the GPU isn't pinned at 100%,
/// trading hashrate for desktop responsiveness / lower heat. Call once at startup
/// (e.g. from a `--gpu-duty` flag); overrides the TOML `duty`.
pub fn set_gpu_duty(duty: f32) {
    GPU_DUTY_MILLI.store((duty.clamp(0.02, 1.0) * 1000.0) as u32, Ordering::Relaxed);
}

/// Current duty cycle. Precedence: `GPU_MINE_DUTY` env (live, no rebuild) >
/// `set_gpu_duty` (CLI) > TOML `duty` > 1.0.
fn gpu_duty() -> f32 {
    if let Ok(s) = std::env::var("GPU_MINE_DUTY") {
        if let Ok(v) = s.parse::<f32>() {
            return v.clamp(0.02, 1.0);
        }
    }
    let milli = GPU_DUTY_MILLI.load(Ordering::Relaxed);
    if milli != 0 {
        return milli as f32 / 1000.0;
    }
    settings().duty
}

const MAX_WINNERS: u32 = 256;
const WINNERS_BYTES: u64 = 16 + (MAX_WINNERS as u64) * 4 * 3;
const SELFTEST_N: u32 = 8;

// NOTE: `wasm-wallet/pow.wgsl` is a copy of this kernel, loaded by the browser
// miner (`gpu_miner.js`) via WebGPU. A fix here should be ported there.
//
// The copies are deliberately independent rather than a shared `include_str!`:
// keeping the node source untouched by the wallet build is worth more than
// deduplication, and the safety property does not depend on the two files
// matching each other. What each host actually guarantees is that ITS kernel
// reproduces ITS CPU reference — `create_extension` here, WASM
// `build_solo_extension` in the browser — enforced by a startup self-test over a
// full EXTENSION_ITERATIONS chain. Both references implement the same spec, so a
// drift between the two copies surfaces as a self-test failure on whichever side
// broke, not as silently rejected blocks.
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

/// Choose every GPU adapter to mine on, best first.
///
/// Enumerates all adapters across all backends, logs them, then:
///   1. if a name override is set (TOML `adapter` / `WGPU_ADAPTER_NAME`), keeps
///      only adapters whose name contains it — *all* of them, not just the first,
///      so "RTX 4090" matches both cards in a twin-4090 box;
///   2. drops pure-software adapters (the CPU SIMD miner beats llvmpipe);
///   3. collapses to a SINGLE backend, then returns every device on it.
///
/// # Reasoning
///
/// The single-device `pick_adapter` this replaces returned `adapters[0]`, so a
/// multi-GPU rig logged every card and then mined on one. Returning the full set
/// is the fix; the subtlety is step 3.
///
/// The same physical GPU is usually enumerated once per backend it supports — on
/// Linux an NVIDIA card shows up under both Vulkan and GL. Building one
/// `GpuMiner` per adapter would then open two contexts on one card and report
/// double the device count while delivering none of the throughput.
///
/// Deduplicating by `AdapterInfo` instead does not work: two *identical* cards
/// (same vendor, same device id, same name) are indistinguishable in wgpu's info
/// struct, and dropping one of them would silently halve a twin-GPU rig. Within
/// a single backend, however, every adapter is a distinct physical device — so
/// picking the best backend and taking all of its devices is correct in both
/// cases.
///
/// # Formal Specification
///
/// ```text
/// Let  A       = all adapters over all backends
///      soft(a) = a.device_type = Cpu
///      rank(a) = (type_rank a.device_type, backend_rank a.backend)
///
/// Pre:
///   - true
///
/// Post:
///   result! = Err(_)  ⇔  A = ∅ ∨ { a ∈ A | ¬soft(a) } = ∅
///
///   result! = Ok(s)   ⇒
///       ∀ a ∈ ran s • ¬soft(a) ∧ a.backend = b*
///       where b* = the backend of the highest-ranked non-software adapter
///
///       ran s = { a ∈ A | ¬soft(a) ∧ a.backend = b* }   (before the cap)
///       #s ≤ cap                                        (cap applied last)
///       ∀ i, j ∈ dom s • i < j ⇒ type_rank(s i) ≤ type_rank(s j)
///
///   One-backend property (why this is exactly one context per physical GPU):
///       ∀ a₁, a₂ ∈ ran s • a₁ ≠ a₂ ⇒ a₁ and a₂ are distinct physical devices
///   because within a single backend wgpu enumerates each device exactly once.
/// ```
///
/// # Safety / Invariants
///
/// - **One context per physical GPU.** Violating the one-backend property would
///   open two `Device`s on one card, doubling the reported device count while
///   halving each context's share of it — an operator would see "2 devices" and
///   no extra hashrate.
/// - **Ordering is load-bearing.** `max_devices` truncates from the tail, so the
///   sort by `type_rank` is what makes a cap of 1 keep the discrete card rather
///   than an integrated one.
/// - **Software adapters are excluded, not deprioritised.** llvmpipe is slower
///   than the real CPU SIMD miner, so mining on it is a net loss; the empty-set
///   error routes the caller to `mine_extension` instead.
/// - **A name filter that matches nothing must not mean "no GPU".** It falls back
///   to full automatic selection with a warning, because a typo in
///   `WGPU_ADAPTER_NAME` should cost the operator their preference, not all mining.
async fn pick_adapters(instance: &wgpu::Instance) -> Result<Vec<wgpu::Adapter>> {
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    if adapters.is_empty() {
        bail!(
            "no GPU adapters found. wgpu uses Vulkan/GL, not CUDA — an NVIDIA card \
             needs its Vulkan ICD installed (verify with `vulkaninfo --summary`)."
        );
    }
    for a in &adapters {
        let i = a.get_info();
        tracing::info!("GPU adapter found: {} [{:?} via {:?}]", i.name, i.device_type, i.backend);
    }

    // (1) explicit override by case-insensitive name substring — retain, not find.
    let name_pref = settings()
        .adapter
        .clone()
        .or_else(|| std::env::var("WGPU_ADAPTER_NAME").ok())
        .filter(|s| !s.trim().is_empty());
    if let Some(want) = name_pref {
        let want_lc = want.to_lowercase();
        let before = adapters.len();
        adapters.retain(|a| a.get_info().name.to_lowercase().contains(&want_lc));
        if adapters.is_empty() {
            tracing::warn!("no GPU adapter matched name '{want}'; using automatic selection");
            adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        } else if adapters.len() < before {
            tracing::info!("adapter filter '{want}' matched {} of {} adapters", adapters.len(), before);
        }
    }

    // (2) drop software adapters.
    adapters.retain(|a| a.get_info().device_type != wgpu::DeviceType::Cpu);
    if adapters.is_empty() {
        bail!("only software (CPU) GPU adapters available; using the CPU miner instead");
    }

    let backend_rank = |b: wgpu::Backend| match b {
        wgpu::Backend::Vulkan => 0u8,
        wgpu::Backend::Dx12 => 1,
        wgpu::Backend::Metal => 2,
        wgpu::Backend::Gl => 3,
        _ => 4,
    };
    let type_rank = |t: wgpu::DeviceType| match t {
        wgpu::DeviceType::DiscreteGpu => 0u8,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        _ => 3,
    };

    // (3) pick the backend whose best device ranks highest, then keep only it.
    //
    // `min_by_key` rather than mapping to a tuple that carries the backend:
    // `wgpu::Backend` implements neither `Ord` nor `PartialOrd`, so it cannot sit
    // inside a comparison key. Rank on `(device_type, backend)` — both plain u8 —
    // and read the backend off the winning adapter afterwards.
    let best_backend = adapters
        .iter()
        .min_by_key(|a| {
            let i = a.get_info();
            (type_rank(i.device_type), backend_rank(i.backend))
        })
        .map(|a| a.get_info().backend)
        .expect("adapters is non-empty");
    adapters.retain(|a| a.get_info().backend == best_backend);

    // Best device first, so a single-device cap keeps the fastest card.
    adapters.sort_by_key(|a| type_rank(a.get_info().device_type));

    let cap = settings()
        .max_devices
        .or_else(|| std::env::var("MINER_GPU_DEVICES").ok().and_then(|v| v.parse().ok()))
        .filter(|n| *n > 0)
        .unwrap_or(usize::MAX);
    if adapters.len() > cap {
        tracing::info!("limiting GPU mining to {} of {} devices", cap, adapters.len());
        adapters.truncate(cap);
    }

    Ok(adapters)
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
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = pick_adapters(&instance)
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no usable GPU adapter"))?;
            Self::new_async(adapter).await
        })
    }

    /// Build every usable device, best first. One `GpuMiner` per physical GPU.
    ///
    /// # Reasoning
    /// Partial failure must be survivable. A mixed rig — a current card plus an
    /// older one on a shakier driver — is the common multi-GPU case, and a single
    /// `request_device` failure taking down all mining would be a worse outcome
    /// than running on one card. Each device is therefore built independently and
    /// a failure is logged and skipped.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - true
    ///
    /// Post:
    ///   result! = Ok(s) ⇒ #s ≥ 1
    ///                   ∧ ∀ m ∈ ran s • m has its own Device, Queue and buffers
    ///                   ∧ ran s ⊆ { successful init over pick_adapters() }
    ///   result! = Err(_) ⇒ no adapter could be initialized
    ///                    (caller falls back to the CPU miner)
    ///
    ///   Order is inherited from pick_adapters: best device first.
    /// ```
    ///
    /// # Safety / Invariants
    /// - **No buffer sharing between devices.** Each `GpuMiner` owns its own
    ///   `state_buf` / `winners_buf` / `params_buf` and its own `Mutex`, so the
    ///   per-device serialisation that `mine_gpu` relies on still holds when N
    ///   devices dispatch concurrently.
    /// - **An empty result is an error, not `Ok(vec![])`.** Callers distinguish
    ///   "no GPU" from "GPU present" by that error; returning an empty vec would
    ///   make `gpu_available()` lie.
    pub fn new_all() -> Result<Vec<Self>> {
        pollster::block_on(async {
            // wgpu 29: InstanceDescriptor lost `Default`. This constructor reads
            // backend/power prefs from env (e.g. WGPU_BACKEND=vulkan to force
            // Vulkan) and needs no window/display handle (we're headless compute).
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapters = pick_adapters(&instance).await?;
            let mut out = Vec::new();
            for adapter in adapters {
                let name = adapter.get_info().name.clone();
                match Self::new_async(adapter).await {
                    Ok(m) => out.push(m),
                    Err(e) => tracing::warn!("skipping GPU '{name}': {e}"),
                }
            }
            if out.is_empty() {
                bail!("no GPU device could be initialized");
            }
            Ok(out)
        })
    }

    async fn new_async(adapter: wgpu::Adapter) -> Result<Self> {
        let info = adapter.get_info();
        tracing::info!(
            "GPU adapter selected: {} [{:?} via {:?}]",
            info.name, info.device_type, info.backend
        );
        let adapter_name = info.name.clone();

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
            size: (settings().batch_nonces as u64) * 8 * 4,
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
        let ipd = settings().iters_per_dispatch;
        let chunk = if throttling { ipd.min(settings().responsive_iters) } else { ipd };

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
            base_lo: 0, base_hi: 0, n_nonces: settings().batch_nonces, iters: 0, has_pool,
            pad0: 0, pad1: 0, pad2: 0,
        };

        loop {
            if cancel.load(Ordering::Relaxed) {
                tracing::debug!("GPU mining cancelled");
                return None;
            }
            let base: u64 = rand::random();
            let winners = self.run_batch(&mut params, base, settings().batch_nonces, &cancel, &hash_counter, true)?;

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

    /// Pop a surplus winner stashed by an earlier `mine_gpu` on this exact job,
    /// without touching the GPU.
    ///
    /// # Reasoning
    /// The multi-device coordinator must drain every card's stash before spawning
    /// a fresh search, or winners a card already found sit unserved while the rig
    /// burns power re-hashing for them. `mine_gpu` cannot be used for this probe:
    /// it *sets* `job` on mismatch and clears `pending`, so probing device B for
    /// job X would destroy B's stash for job Y. This is the read-only sibling.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - true  (total; a mismatched or absent job yields None)
    ///
    /// Post:
    ///   job = job? ∧ #pending > 0 ⇒
    ///       result! = Some(head pending) ∧ pending' = tail pending
    ///   job ≠ job? ∨ #pending = 0 ⇒
    ///       result! = None ∧ pending' = pending ∧ job' = job
    ///
    ///   In all cases:  job' = job          (NEVER rotates the job — this is the
    ///                                       difference from mine_gpu)
    /// ```
    ///
    /// ```zed
    ///     TakePending
    ///     ----------------
    ///     ΔPending
    ///     ΞJob
    ///     job?     : JobKey
    ///     result!  : MiningResult ∪ {⊥}
    ///
    ///     pre  true
    ///
    ///     post job = job? ∧ pending ≠ ⟨⟩ ⇒
    ///            result! = head pending ∧ pending' = tail pending
    ///     post job ≠ job? ∨ pending = ⟨⟩ ⇒
    ///            result! = ⊥ ∧ pending' = pending
    ///     post job' = job
    /// ```
    ///
    /// # Safety / Invariants
    /// - **Read-only with respect to `job`.** A probe must not be able to discard
    ///   another job's stash; that would silently throw away verified work.
    /// - **Only serves hits mined for the requested job.** A winner from a
    ///   superseded template would be submitted and rejected as stale, so the
    ///   job guard is a correctness requirement, not an optimisation.
    /// - **FIFO.** `mine_gpu` sorts blocks ahead of shares before stashing, so
    ///   draining from the front preserves "a block is always served first".
    pub fn take_pending(&self, job_midstate: [u8; 32], job_target: [u8; 32], job_pool: Option<[u8; 32]>) -> Option<MiningResult> {
        let job: JobKey = (job_midstate, job_target, job_pool);
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.job.as_ref() != Some(&job) {
            return None;
        }
        st.pending.pop_front()
    }

    /// Return an already-verified hit to this device's stash, to be served by a
    /// later `mine_gpu`/`take_pending` on the same job.
    ///
    /// # Reasoning
    /// Without this, work is lost. When several cards search concurrently, more
    /// than one can complete a batch before the winner's `stop` flag is observed.
    /// The coordinator returns the first hit and every *other* hit — already
    /// CPU-verified, already paid for in electricity — was silently dropped when
    /// its channel send found no reader. Stashing it against the originating
    /// device makes the next call serve it for free.
    ///
    /// # Formal Specification
    ///
    /// ```text
    /// Pre:
    ///   - hit? has been CPU-verified by mine_gpu (this method does NOT re-verify)
    ///
    /// Post:
    ///   job = job? ⇒ pending' = pending ⌢ ⟨hit?⟩
    ///   job ≠ job? ⇒ pending' = pending          (dropped: wrong job)
    /// ```
    ///
    /// # Safety / Invariants
    /// - **A hit is only stashed under the job it was mined for.** The job guard
    ///   is what stops a winner from one template being served against another,
    ///   which would produce a share the pool rejects as stale.
    /// - **Verification is the caller's responsibility.** Every path into this
    ///   method comes from `mine_gpu`, which has already re-run
    ///   `create_extension` on the CPU, so the kernel is never trusted here.
    pub fn push_pending(
        &self,
        job_midstate: [u8; 32],
        job_target: [u8; 32],
        job_pool: Option<[u8; 32]>,
        hit: MiningResult,
    ) {
        let job: JobKey = (job_midstate, job_target, job_pool);
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.job.as_ref() == Some(&job) {
            st.pending.push_back(hit);
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
    shared_all().first()
}

/// Every self-tested GPU available for mining, best device first. Empty when
/// there is no usable GPU or `MINER_DISABLE_GPU` is set.
///
/// # Reasoning
/// This replaces a `OnceLock<Option<GpuMiner>>` that structurally could not hold
/// more than one device, which is why a two-GPU rig detected both cards and mined
/// on one. Per-device self-testing is the other half: on a mixed rig a card whose
/// driver miscomputes BLAKE3 must be dropped *on its own* rather than disabling
/// all mining, which is what a single combined pass/fail would have done.
///
/// # Formal Specification
///
/// ```text
/// Pre:
///   - true
///
/// Post (memoised once per process; the result is never recomputed):
///   MINER_DISABLE_GPU set   ⇒ result! = ⟨⟩
///   otherwise               ⇒ ran result! = { d ∈ new_all() | self_test(d) = Ok }
///                             ∧ order inherited from pick_adapters (best first)
///
///   gpu_available() ⇔ #result! > 0
/// ```
///
/// # Safety / Invariants
/// - **Self-test is a hard gate.** A device that cannot reproduce
///   `create_extension` bit-for-bit over the full 1,000,000-iteration chain never
///   enters the set. This is the only defence against a driver that computes
///   plausible-but-wrong hashes, which would otherwise surface as universally
///   rejected shares.
/// - **Failures are per-device.** One card excluded must not remove the others,
///   and all-failed must degrade to the CPU miner rather than to nothing.
/// - **Memoised deliberately.** Init cost (device + shader compile + a full
///   self-test chain) is paid once, never per block. A transient failure at
///   startup therefore disables GPU mining for the life of the process — the
///   accepted trade for not re-testing on every template.
pub fn shared_all() -> &'static [GpuMiner] {
    static SHARED: OnceLock<Vec<GpuMiner>> = OnceLock::new();
    SHARED.get_or_init(|| {
        if std::env::var("MINER_DISABLE_GPU").map(|v| v != "0").unwrap_or(false) {
            tracing::info!("GPU mining disabled via MINER_DISABLE_GPU");
            return Vec::new();
        }
        let candidates = match GpuMiner::new_all() {
            Ok(v) => v,
            Err(e) => {
                tracing::info!("GPU mining disabled (no usable device): {e}");
                return Vec::new();
            }
        };
        let total = candidates.len();
        let mut ok = Vec::new();
        for g in candidates {
            match g.self_test() {
                Ok(()) => ok.push(g),
                Err(e) => tracing::warn!(
                    "GPU '{}' excluded from mining (self-test failed): {e}",
                    g.adapter_name()
                ),
            }
        }
        match ok.len() {
            0 => tracing::warn!("GPU mining disabled: all {total} device(s) failed self-test"),
            1 => tracing::info!("GPU mining enabled on {}", ok[0].adapter_name()),
            n => tracing::info!(
                "GPU mining enabled on {n} devices: {}",
                ok.iter().map(|g| g.adapter_name()).collect::<Vec<_>>().join(", ")
            ),
        }
        ok
    })
}

/// `true` if a self-tested GPU backend is available for mining.
pub fn gpu_available() -> bool {
    !shared_all().is_empty()
}

/// How many GPUs are mining.
pub fn gpu_device_count() -> usize {
    shared_all().len()
}

/// Run the nonce search across every available GPU, returning the first hit.
///
/// # Reasoning
///
/// `mine()` previously called `shared()`, which was a `OnceLock<Option<GpuMiner>>`
/// holding exactly one device. On a two-GPU rig both cards were enumerated and
/// logged — so the operator could see both — while only the best-ranked one ever
/// dispatched a single instruction. The other card sat idle for the life of the
/// process. This function is the fan-out that fixes that.
///
/// Threads rather than one queue: `mine_gpu` is synchronous and holds its
/// device's `Mutex` for the whole call (the buffers are only safe to touch while
/// it is held), so N cards require N threads. No nonce striding is needed —
/// each card draws `base: u64 = rand::random()` per batch, so devices explore
/// disjoint regions of a 2^64 space with no coordination and a collision
/// probability that rounds to zero.
///
/// The internal `stop` flag is what keeps the fan-out cheap: once one card has a
/// verified hit the others abandon in-flight batches at the next dispatch
/// boundary (~`iters_per_dispatch` steps) instead of running a full chain to
/// completion.
///
/// # Formal Specification
///
/// ```text
/// Let  D          = shared_all()                    (self-tested devices)
///      job        = (midstate?, target?, pool_target?)
///      pending(d) = device d's surplus-winner queue for job
///      verified(h)= create_extension(midstate?, h.nonce).final_hash < the
///                   target implied by h's variant                (from mine_gpu)
///
/// Pre:
///   - true  (total; #D = 0 yields None, which routes the caller to the CPU miner)
///
/// Post:
///   result! = Some(h) ⇒ verified(h)
///                     ∧ h was produced by some d ∈ D
///
///   result! = None    ⇒ cancel? was observed set
///                     ∨ every d ∈ D returned without a hit
///
///   ∀ d ∈ D • every verified hit d produced during this call is either
///             returned as result! or appended to pending'(d)     (no work lost)
///
///   hash_counter' ≥ hash_counter                    (summed across all of D)
///
///   #{ threads alive on return } = 0                (all joined, always)
/// ```
///
/// ```zed
///     MineAllGpus
///     ----------------
///     ΔPending
///     midstate?, target? : Hash
///     pool_target?       : Hash ∪ {⊥}
///     cancel?            : 𝔹
///     result!            : MiningResult ∪ {⊥}
///
///     pre  true
///
///     post result! ≠ ⊥ ⇒ verified result!
///     post result! = ⊥ ⇒ cancel? ∨ (∀ d : D • d exhausted its search)
///
///     post ∀ d : D •
///            pending' d = pending d ⌢ (hits d \ ⟨result!⟩)
///
///     post #D = 0 ⇒ result! = ⊥ ∧ pending' = pending
/// ```
///
/// # Safety / Invariants
///
/// - **Never trusts a kernel.** Every returned hit was re-verified against
///   `create_extension` inside `mine_gpu`. A driver that computes wrong hashes —
///   or a device whose `self_test` passed but which degrades under thermal load —
///   can cost throughput but can never surface an invalid share or block.
/// - **No mined work is discarded.** After the winner is chosen the channel is
///   drained and any *other* device's verified hit is returned to that device's
///   stash via `push_pending`. Before this, a second card finishing a batch in
///   the same instant had its hit dropped when the send found no reader.
/// - **Stashes are drained before any GPU work.** A card holding surplus winners
///   from an earlier batch on this job must serve them before the rig re-hashes
///   for results it already has.
/// - **The caller's `cancel` is polled here, not handed to the workers.** Workers
///   need `stop`, which fires both on cancellation *and* on a win; conflating the
///   two would leave the losers running after a block was already found.
/// - **Termination is unconditional.** Every spawned thread is joined on all
///   paths — win, cancel, and exhaustion — so a caller can never outlive its
///   workers and two mining rounds can never dispatch to one device at once.
/// - **`hash_counter` is shared across devices,** so the rate the node reports is
///   the rig total rather than one card's contribution.
fn mine_all_gpus(
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<AtomicU64>,
) -> Option<MiningResult> {
    let devices = shared_all();
    match devices.len() {
        0 => return None,
        // Single GPU: call straight through. Spawning a thread and a channel to
        // supervise one device would only add latency to the common case.
        1 => return devices[0].mine_gpu(midstate, target, pool_target, cancel, hash_counter),
        _ => {}
    }

    // Drain stashes first — free winners, no GPU work.
    for d in devices {
        if let Some(hit) = d.take_pending(midstate, target, pool_target) {
            return Some(hit);
        }
    }

    let stop = Arc::new(AtomicBool::new(false));
    // Carries the device index so a hit that arrives after the winner can be
    // returned to the stash of the card that actually mined it.
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Option<MiningResult>)>();
    let mut handles = Vec::with_capacity(devices.len());

    for (idx, d) in devices.iter().enumerate() {
        let stop = stop.clone();
        let hc = hash_counter.clone();
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || {
            let r = d.mine_gpu(midstate, target, pool_target, stop, hc);
            let _ = tx.send((idx, r));
        }));
    }
    drop(tx); // so the channel disconnects once every worker has finished

    let mut result = None;
    loop {
        if cancel.load(Ordering::Relaxed) {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok((_, Some(hit))) => {
                result = Some(hit);
                stop.store(true, Ordering::Relaxed);
                break;
            }
            // That device stopped without a hit; others may still be searching.
            Ok((_, None)) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    for h in handles {
        let _ = h.join();
    }

    // Recover hits that landed while we were breaking out of the loop above.
    // `rx` is still in scope, so those sends succeeded into the queue rather
    // than failing — draining it now is what makes "no work lost" true.
    while let Ok((idx, maybe_hit)) = rx.try_recv() {
        if let Some(hit) = maybe_hit {
            match result {
                None => result = Some(hit),
                Some(_) => devices[idx].push_pending(midstate, target, pool_target, hit),
            }
        }
    }

    result
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
        // Fans out across every self-tested device; falls through to CPU if none.
        if gpu_available() {
            return mine_all_gpus(midstate, target, pool_target, cancel, hash_counter);
        }
        if backend() == Backend::Gpu {
            tracing::warn!("GPU backend requested but no usable GPU; mining on CPU");
        }
    }
    mine_extension(midstate, target, pool_target, threads, cancel, hash_counter)
}
