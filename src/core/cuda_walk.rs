//! Midstate Walk on NVIDIA GPUs: the production side of the CUDA benchmark in
//! `docs/walk-reference/cuda`.
//!
//! # What runs on the GPU
//!
//! The kernels are `cuda/walk_miner.cu`: the benchmark's own walk
//! (`walk_core.h`, which CI checks against this crate's [`walk`]), compiled
//! to PTX by `cuda/gen_ptx.sh` and embedded here. The driver compiles the PTX
//! for whatever card it finds, so nothing here needs a CUDA toolkit, at build
//! time or at run time: only the NVIDIA driver, which is loaded when mining
//! starts (`libcuda.so.1`, `nvcuda.dll`). A machine without one mines on its
//! CPU, as before, and says why once.
//!
//! # Why a wrong GPU cannot cost a block
//!
//! - Every device runs the three known answers ([`walk::GOLDEN`]) when it
//!   opens, and is not used if any is wrong. (The first PTX failed this on
//!   every GPU, through a clang bug in the 64-bit rotate. CI now runs the PTX
//!   itself on the CPU against the known answers: `cuda/ptx_check.cpp`.)
//! - Every hash a GPU reports below a target is recomputed on the CPU with
//!   [`walk::walk_hash`] before it becomes a share or a block. A GPU that
//!   goes wrong later (an overclock running hot, say) only loses work, and
//!   the log says so.
//!
//! Nothing here changes the walk or any rule: the GPU searches nonces, and
//! the CPU's reference decides.
//!
//! # Working with the CPU
//!
//! The CPU threads draw random nonces below 2^63 ([`crate::core::extension`]).
//! GPU `d` searches `2^63 | d << 56 | counter`, the counter starting at a
//! random point for each job, so no two workers try the same nonce. A job
//! ends for every worker as soon as one finds something or the job is
//! cancelled: GPUs check between launches, which last tens of milliseconds.

#![cfg(not(target_arch = "wasm32"))]

use super::extension::{mine_extension, MiningResult};
use super::types::{Extension, PowAlgo};
use super::walk;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The kernels, as PTX (see `cuda/gen_ptx.sh`).
pub const PTX: &str = include_str!("cuda/walk_miner.ptx");

/// Threads per block, as the kernels' `__launch_bounds__`.
const BLOCK: u32 = 256;
/// How long one step launch should take: short enough that a stale job stops
/// promptly and a display stays responsive, long enough to keep the GPU fed.
const LAUNCH_TARGET: Duration = Duration::from_millis(40);
/// Bytes of device memory each walk needs besides its scratchpad.
const PER_WALK_EXTRA: u64 = 8 * 8 + 8 + 4 + 8 + 32;

#[cfg(not(feature = "fast-mining"))]
const KERNELS: (&str, &str) = ("ms_init_main", "ms_steps_main");
#[cfg(feature = "fast-mining")]
const KERNELS: (&str, &str) = ("ms_init_fast", "ms_steps_fast");

/// derive_key's context keys, as `walk.rs` uses them: init, remix, final.
const CONTEXTS: [&str; 3] = ["midstate walk v1 init", "midstate walk v1 remix", "midstate walk v1 final"];

/// The 24 words the kernels' `ms_keys` holds.
pub fn context_words() -> [u32; 24] {
    let mut out = [0u32; 24];
    for (i, ctx) in CONTEXTS.iter().enumerate() {
        let key = blake3::hazmat::hash_derive_key_context(ctx);
        for (j, w) in key.chunks_exact(4).enumerate() {
            out[i * 8 + j] = u32::from_le_bytes(w.try_into().unwrap());
        }
    }
    out
}

/// The AES table the kernels' `ms_te` holds (`walk_core.h`'s `build_te`).
pub fn te_words() -> [u32; 256] {
    let mut te = [0u32; 256];
    for (x, slot) in te.iter_mut().enumerate() {
        let s = walk::SBOX[x];
        let s2 = (s << 1) ^ if s & 0x80 != 0 { 0x1b } else { 0 };
        let s3 = s2 ^ s;
        *slot = s2 as u32 | (s as u32) << 8 | (s as u32) << 16 | (s3 as u32) << 24;
    }
    te
}

// ── The driver API ──────────────────────────────────────────────────────────

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevicePtr = u64;

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_CTX_SCHED_BLOCKING_SYNC: c_uint = 0x04;

/// The driver's entry points this miner uses, loaded once.
struct Api {
    _lib: libloading::Library,
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    primary_ctx_set_flags: unsafe extern "C" fn(CuDevice, c_uint) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    memcpy_htod: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    module_get_global: unsafe extern "C" fn(*mut CuDevicePtr, *mut usize, CuModule, *const c_char) -> CuResult,
    launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    get_error_string: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

// The function pointers are plain C entry points, safe to call from any
// thread; the library stays loaded for the life of the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

#[cfg(target_os = "windows")]
const LIBRARIES: &[&str] = &["nvcuda.dll"];
#[cfg(not(target_os = "windows"))]
const LIBRARIES: &[&str] = &["libcuda.so.1", "libcuda.so"];

impl Api {
    fn load() -> Result<Api, String> {
        let mut last = String::from("no CUDA driver library for this platform");
        for name in LIBRARIES {
            // SAFETY: loading the NVIDIA driver runs its initializers, as any
            // CUDA program does.
            match unsafe { libloading::Library::new(name) } {
                Ok(lib) => return Self::bind(lib),
                Err(e) => last = format!("{name}: {e}"),
            }
        }
        Err(format!("the NVIDIA driver is not installed ({last})"))
    }

    fn bind(lib: libloading::Library) -> Result<Api, String> {
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the type is the driver API's documented signature.
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| format!("the CUDA driver lacks {}: {e}", $name))?
            };
        }
        let api = Api {
            init: sym!("cuInit"),
            device_get_count: sym!("cuDeviceGetCount"),
            device_get: sym!("cuDeviceGet"),
            device_get_name: sym!("cuDeviceGetName"),
            device_get_attribute: sym!("cuDeviceGetAttribute"),
            primary_ctx_set_flags: sym!("cuDevicePrimaryCtxSetFlags_v2"),
            primary_ctx_retain: sym!("cuDevicePrimaryCtxRetain"),
            ctx_set_current: sym!("cuCtxSetCurrent"),
            ctx_synchronize: sym!("cuCtxSynchronize"),
            mem_get_info: sym!("cuMemGetInfo_v2"),
            mem_alloc: sym!("cuMemAlloc_v2"),
            mem_free: sym!("cuMemFree_v2"),
            memcpy_htod: sym!("cuMemcpyHtoD_v2"),
            memcpy_dtoh: sym!("cuMemcpyDtoH_v2"),
            module_load_data: sym!("cuModuleLoadData"),
            module_get_function: sym!("cuModuleGetFunction"),
            module_get_global: sym!("cuModuleGetGlobal_v2"),
            launch_kernel: sym!("cuLaunchKernel"),
            get_error_string: sym!("cuGetErrorString"),
            _lib: lib,
        };
        api.check(unsafe { (api.init)(0) }, "starting the CUDA driver")?;
        Ok(api)
    }

    fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorString writes a pointer to a static string.
        let text = if unsafe { (self.get_error_string)(r, &mut s) } == 0 && !s.is_null() {
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        } else {
            format!("CUDA error {r}")
        };
        Err(format!("{what}: {text}"))
    }
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

fn api() -> Result<&'static Api, String> {
    API.get_or_init(Api::load).as_ref().map_err(Clone::clone)
}

/// A CUDA device, as found.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub sms: u32,
}

/// Every CUDA device the driver reports; `Err` says why there are none.
pub fn devices() -> Result<Vec<DeviceInfo>, String> {
    let api = api()?;
    let mut n: c_int = 0;
    api.check(unsafe { (api.device_get_count)(&mut n) }, "counting CUDA devices")?;
    let mut out = Vec::new();
    for i in 0..n.max(0) {
        let mut dev: CuDevice = 0;
        api.check(unsafe { (api.device_get)(&mut dev, i) }, "opening a CUDA device")?;
        let mut name = [0 as c_char; 128];
        api.check(unsafe { (api.device_get_name)(name.as_mut_ptr(), 127, dev) }, "naming a CUDA device")?;
        let mut sms: c_int = 0;
        api.check(
            unsafe { (api.device_get_attribute)(&mut sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) },
            "reading a CUDA device",
        )?;
        out.push(DeviceInfo {
            index: i as usize,
            name: unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned(),
            sms: sms.max(1) as u32,
        });
    }
    Ok(out)
}

// ── One GPU ─────────────────────────────────────────────────────────────────

/// Runs batches of walks. The CUDA walker implements it; tests use a CPU
/// stand-in, so the search around it can be tested without a GPU.
pub trait WalkBatcher: Send {
    /// "CUDA 0: NVIDIA GeForce RTX 4090", for logs and the dashboard.
    fn label(&self) -> String;
    /// The most walks one batch can run.
    fn capacity(&self) -> usize;
    /// The final hashes of `nonces` (at most `capacity()`) for
    /// `mining_hash`, or `Ok(None)` if `cancel` was set first.
    fn hash_batch(
        &mut self,
        mining_hash: &[u8; 32],
        nonces: &[u64],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<[u8; 32]>>, String>;
}

struct Buffers {
    r: CuDevicePtr,
    c: CuDevicePtr,
    prog: CuDevicePtr,
    pads: CuDevicePtr,
    hash: CuDevicePtr,
    nonces: CuDevicePtr,
    out: CuDevicePtr,
}

/// One GPU, its module loaded and its walks' memory allocated.
pub struct CudaWalker {
    api: &'static Api,
    info: DeviceInfo,
    ctx: CuContext,
    init: CuFunction,
    steps: CuFunction,
    finish: CuFunction,
    buf: Buffers,
    capacity: u32,
    /// Steps per launch, adjusted to `LAUNCH_TARGET`; a multiple of 8.
    chunk: u32,
}

// The context is the device's primary one, made current on whichever thread
// uses the walker before every call; only one thread uses a walker at a time.
unsafe impl Send for CudaWalker {}

impl CudaWalker {
    /// Opens device `index` for `walks` walks at a time (`None`: as many as
    /// keep it busy, within its free memory), and checks it against the known
    /// answers.
    pub fn open(index: usize, walks: Option<u32>) -> Result<Self, String> {
        let api = api()?;
        let info = devices()?
            .into_iter()
            .find(|d| d.index == index)
            .ok_or_else(|| format!("there is no CUDA device {index}"))?;
        let mut dev: CuDevice = 0;
        api.check(unsafe { (api.device_get)(&mut dev, index as c_int) }, "opening the device")?;
        // Sleep in cuCtxSynchronize rather than spin, so the thread that feeds
        // the GPU leaves its core to the CPU miner. Fails harmlessly if
        // another part of the process made the context first.
        let _ = unsafe { (api.primary_ctx_set_flags)(dev, CU_CTX_SCHED_BLOCKING_SYNC) };
        let mut ctx: CuContext = std::ptr::null_mut();
        api.check(unsafe { (api.primary_ctx_retain)(&mut ctx, dev) }, "creating a context")?;
        api.check(unsafe { (api.ctx_set_current)(ctx) }, "using the context")?;

        let ptx = CString::new(PTX).expect("PTX has no NUL");
        let mut module: CuModule = std::ptr::null_mut();
        api.check(
            unsafe { (api.module_load_data)(&mut module, ptx.as_ptr() as *const c_void) },
            "loading the walk kernels (the driver may be too old for this GPU)",
        )?;
        let func = |name: &str| -> Result<CuFunction, String> {
            let n = CString::new(name).unwrap();
            let mut f: CuFunction = std::ptr::null_mut();
            api.check(unsafe { (api.module_get_function)(&mut f, module, n.as_ptr()) }, name)?;
            Ok(f)
        };
        let (init_name, steps_name) = KERNELS;
        let (init, steps, finish) = (func(init_name)?, func(steps_name)?, func("ms_final")?);
        let upload = |name: &str, words: &[u32]| -> Result<(), String> {
            let n = CString::new(name).unwrap();
            let mut ptr: CuDevicePtr = 0;
            let mut bytes = 0usize;
            api.check(unsafe { (api.module_get_global)(&mut ptr, &mut bytes, module, n.as_ptr()) }, name)?;
            if bytes != words.len() * 4 {
                return Err(format!("{name} is {bytes} bytes on the device, expected {}", words.len() * 4));
            }
            api.check(unsafe { (api.memcpy_htod)(ptr, words.as_ptr() as *const c_void, bytes) }, name)
        };
        upload("ms_te", &te_words())?;
        upload("ms_keys", &context_words())?;

        let pad_bytes = (walk::WALK_LINES * 64) as u64;
        let per_walk = pad_bytes + PER_WALK_EXTRA;
        let (mut free, mut total) = (0usize, 0usize);
        api.check(unsafe { (api.mem_get_info)(&mut free, &mut total) }, "reading free memory")?;
        // Leave a fifth of what is free to the display and the driver.
        let fits = (free as u64 * 4 / 5 / per_walk).min(u32::MAX as u64) as u32;
        // Enough walks in flight to hide the scratchpads' memory latency: the
        // benchmark's sweep stops gaining well before 128 per SM.
        let want = walks.unwrap_or(info.sms * 128).max(BLOCK);
        let capacity = (want.min(fits) / BLOCK) * BLOCK;
        if capacity == 0 {
            return Err(format!("{} has too little free memory for one block of walks", info.name));
        }

        let n = capacity as usize;
        let mut bufs = [0u64; 7];
        let sizes = [n * 64, n * 8, n * 4, n * pad_bytes as usize, 32, n * 8, n * 32];
        for (slot, bytes) in bufs.iter_mut().zip(sizes) {
            if let Err(e) = api.check(unsafe { (api.mem_alloc)(slot, bytes) }, "allocating walk memory") {
                for p in bufs.iter().filter(|p| **p != 0) {
                    unsafe { (api.mem_free)(*p) };
                }
                return Err(e);
            }
        }
        let buf = Buffers { r: bufs[0], c: bufs[1], prog: bufs[2], pads: bufs[3], hash: bufs[4], nonces: bufs[5], out: bufs[6] };
        let mut w = CudaWalker { api, info, ctx, init, steps, finish, buf, capacity, chunk: 512 };
        w.self_test()?;
        Ok(w)
    }

    /// The known answers, on this GPU.
    fn self_test(&mut self) -> Result<(), String> {
        let never = AtomicBool::new(false);
        for (hash_hex, nonce, want_hex) in walk::GOLDEN {
            let mut mh = [0u8; 32];
            hex::decode_to_slice(hash_hex, &mut mh).map_err(|e| e.to_string())?;
            let got = self.hash_batch(&mh, &[nonce], &never)?.expect("not cancelled");
            if hex::encode(got[0]) != want_hex {
                return Err(format!(
                    "{} got a known answer wrong (nonce {nonce}: {} instead of {want_hex}); not mining on it",
                    self.label(),
                    hex::encode(got[0])
                ));
            }
        }
        Ok(())
    }

    fn launch(&self, f: CuFunction, walks: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        let grid = walks.div_ceil(BLOCK);
        self.api.check(
            unsafe {
                (self.api.launch_kernel)(
                    f,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "launching a walk kernel",
        )
    }

    fn sync(&self) -> Result<(), String> {
        self.api.check(unsafe { (self.api.ctx_synchronize)() }, "running the walk kernels")
    }
}

/// A kernel argument: a pointer to the value, as `cuLaunchKernel` wants.
fn arg<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}

impl WalkBatcher for CudaWalker {
    fn label(&self) -> String {
        format!("CUDA {}: {}", self.info.index, self.info.name)
    }

    fn capacity(&self) -> usize {
        self.capacity as usize
    }

    fn hash_batch(
        &mut self,
        mining_hash: &[u8; 32],
        nonces: &[u64],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<[u8; 32]>>, String> {
        let walks = nonces.len().min(self.capacity as usize) as u32;
        if walks == 0 {
            return Ok(Some(Vec::new()));
        }
        let api = self.api;
        api.check(unsafe { (api.ctx_set_current)(self.ctx) }, "using the context")?;
        api.check(unsafe { (api.memcpy_htod)(self.buf.hash, mining_hash.as_ptr() as *const c_void, 32) }, "uploading the job")?;
        api.check(
            unsafe { (api.memcpy_htod)(self.buf.nonces, nonces.as_ptr() as *const c_void, walks as usize * 8) },
            "uploading nonces",
        )?;
        let b = &self.buf;
        let cap = self.capacity;
        self.launch(
            self.init,
            walks,
            &mut [arg(&b.r), arg(&b.c), arg(&b.prog), arg(&b.pads), arg(&cap), arg(&b.hash), arg(&b.nonces), arg(&walks)],
        )?;
        let total = walk::WALK_STEPS as u32;
        let mut first = 0u32;
        while first < total {
            if cancel.load(Ordering::Relaxed) {
                self.sync()?;
                return Ok(None);
            }
            let n = self.chunk.min(total - first);
            let t = Instant::now();
            self.launch(
                self.steps,
                walks,
                &mut [arg(&b.r), arg(&b.c), arg(&b.prog), arg(&b.pads), arg(&cap), arg(&walks), arg(&first), arg(&n)],
            )?;
            self.sync()?;
            let took = t.elapsed();
            // Keep launches near the target: halve or double, staying a
            // multiple of 8 (the kernel's unit) within the walk.
            if took > LAUNCH_TARGET * 2 && self.chunk > 64 {
                self.chunk /= 2;
            } else if took < LAUNCH_TARGET / 2 && self.chunk < total {
                self.chunk *= 2;
            }
            first += n;
        }
        self.launch(
            self.finish,
            walks,
            &mut [arg(&b.r), arg(&b.c), arg(&b.prog), arg(&b.pads), arg(&cap), arg(&b.hash), arg(&b.nonces), arg(&b.out), arg(&walks)],
        )?;
        self.sync()?;
        let mut words = vec![0u32; walks as usize * 8];
        api.check(
            unsafe { (api.memcpy_dtoh)(words.as_mut_ptr() as *mut c_void, b.out, words.len() * 4) },
            "reading results",
        )?;
        Ok(Some(
            words
                .chunks_exact(8)
                .map(|w| {
                    let mut h = [0u8; 32];
                    for (i, x) in w.iter().enumerate() {
                        h[i * 4..i * 4 + 4].copy_from_slice(&x.to_le_bytes());
                    }
                    h
                })
                .collect(),
        ))
    }
}

impl Drop for CudaWalker {
    fn drop(&mut self) {
        let api = self.api;
        unsafe {
            (api.ctx_set_current)(self.ctx);
            let b = &self.buf;
            for p in [b.r, b.c, b.prog, b.pads, b.hash, b.nonces, b.out] {
                (api.mem_free)(p);
            }
        }
    }
}

// ── Settings and status ─────────────────────────────────────────────────────

/// Which GPUs to mine on: `MINER_CUDA_DEVICES=0,2` (or `set_devices`), else
/// every one. `MINER_DISABLE_GPU=1` turns them all off.
static DEVICE_CHOICE: Mutex<Option<Vec<usize>>> = Mutex::new(None);
/// Walks per batch: `MINER_CUDA_WALKS`, else sized to each GPU.
static WALKS_CHOICE: AtomicU64 = AtomicU64::new(0);

pub fn set_devices(devices: Option<Vec<usize>>) {
    *DEVICE_CHOICE.lock().unwrap() = devices;
}

pub fn set_walks(walks: Option<u32>) {
    WALKS_CHOICE.store(walks.unwrap_or(0) as u64, Ordering::Relaxed);
}

fn gpu_disabled() -> bool {
    std::env::var("MINER_DISABLE_GPU").map(|v| v != "0").unwrap_or(false)
}

fn chosen_devices(found: &[DeviceInfo]) -> Vec<usize> {
    if let Some(d) = DEVICE_CHOICE.lock().unwrap().clone() {
        return d;
    }
    if let Ok(v) = std::env::var("MINER_CUDA_DEVICES") {
        return v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    }
    found.iter().map(|d| d.index).collect()
}

fn chosen_walks() -> Option<u32> {
    match WALKS_CHOICE.load(Ordering::Relaxed) {
        0 => std::env::var("MINER_CUDA_WALKS").ok().and_then(|v| v.parse().ok()),
        n => Some(n as u32),
    }
}

/// A backend at work, for the dashboard.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BackendStatus {
    /// "cpu" or "cuda".
    pub kind: String,
    /// "CPU (8 threads)", "CUDA 0: NVIDIA GeForce RTX 4090".
    pub label: String,
    /// Walk attempts since the process started.
    pub hashes: u64,
}

struct Registered {
    label: String,
    hashes: Arc<AtomicU64>,
}

/// Opened GPUs, kept between jobs (opening one allocates gigabytes), and
/// their counters. Locked when a job starts and ends, never while hashing.
struct Pool {
    tried: bool,
    walkers: Vec<Option<Box<dyn WalkBatcher>>>,
    counters: Vec<Registered>,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { tried: false, walkers: Vec::new(), counters: Vec::new() });
/// What the last Walk job ran on: CPU threads (0: none), and whether GPUs did.
static LAST_CPU_THREADS: AtomicU64 = AtomicU64::new(0);
static GPUS_ACTIVE: AtomicBool = AtomicBool::new(false);
static WALK_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Opens the chosen GPUs once; later calls reuse them. Logs once why there
/// are none.
fn open_pool(pool: &mut Pool) {
    if pool.tried {
        return;
    }
    pool.tried = true;
    if gpu_disabled() {
        tracing::info!("Midstate Walk: GPUs disabled by MINER_DISABLE_GPU; mining on the CPU");
        return;
    }
    let found = match devices() {
        Ok(d) if d.is_empty() => {
            tracing::info!("Midstate Walk: no CUDA GPU found; mining on the CPU");
            return;
        }
        Ok(d) => d,
        Err(e) => {
            tracing::info!("Midstate Walk: no CUDA GPU ({e}); mining on the CPU");
            return;
        }
    };
    for index in chosen_devices(&found) {
        match CudaWalker::open(index, chosen_walks()) {
            Ok(w) => {
                tracing::info!("Midstate Walk: mining on {} ({} walks at a time)", w.label(), w.capacity());
                register(pool, Box::new(w));
            }
            Err(e) => tracing::warn!("Midstate Walk: not mining on CUDA device {index}: {e}"),
        }
    }
    if pool.walkers.is_empty() {
        tracing::warn!("Midstate Walk: no CUDA GPU passed its checks; mining on the CPU");
    }
}

fn register(pool: &mut Pool, w: Box<dyn WalkBatcher>) {
    pool.counters.push(Registered { label: w.label(), hashes: Arc::new(AtomicU64::new(0)) });
    pool.walkers.push(Some(w));
}

/// Adds a walker as if found on a GPU: tests use it with a CPU stand-in.
#[doc(hidden)]
pub fn install_for_tests(walkers: Vec<Box<dyn WalkBatcher>>) {
    end_session();
    let mut pool = POOL.lock().unwrap();
    pool.tried = true;
    pool.walkers.clear();
    pool.counters.clear();
    for w in walkers {
        register(&mut pool, w);
    }
}

fn cpu_label(threads: u64) -> String {
    format!("CPU ({threads} thread{})", if threads == 1 { "" } else { "s" })
}

/// The backends of the Walk search now running (or last run), with their
/// attempts so far. `total` is every attempt the caller counted, CPU and GPU;
/// the CPU's are what the GPUs did not count. Empty before Walk has run.
/// Never waits: while GPUs are being opened (the driver compiling the
/// kernels can take seconds) it reports the CPU alone.
pub fn backend_status(total: u64) -> Vec<BackendStatus> {
    if !WALK_ACTIVE.load(Ordering::Relaxed) {
        return Vec::new();
    }
    let Ok(pool) = POOL.try_lock() else {
        let threads = LAST_CPU_THREADS.load(Ordering::Relaxed);
        return if threads > 0 {
            vec![BackendStatus { kind: "cpu".into(), label: cpu_label(threads), hashes: total }]
        } else {
            Vec::new()
        };
    };
    let gpus: Vec<BackendStatus> = if GPUS_ACTIVE.load(Ordering::Relaxed) {
        pool.counters
            .iter()
            .map(|r| BackendStatus { kind: "cuda".into(), label: r.label.clone(), hashes: r.hashes.load(Ordering::Relaxed) })
            .collect()
    } else {
        Vec::new()
    };
    let gpu_total: u64 = pool.counters.iter().map(|r| r.hashes.load(Ordering::Relaxed)).sum();
    let threads = LAST_CPU_THREADS.load(Ordering::Relaxed);
    let mut out = Vec::new();
    if threads > 0 {
        out.push(BackendStatus {
            kind: "cpu".into(),
            label: cpu_label(threads),
            hashes: total.saturating_sub(gpu_total),
        });
    }
    out.extend(gpus);
    out
}

// ── The search ──────────────────────────────────────────────────────────────

/// The highest bit marks a GPU's nonces; the CPU's never have it.
const GPU_NONCE: u64 = 1 << 63;

/// GPU `slot`'s nonce number `counter`.
fn gpu_nonce(slot: usize, counter: u64) -> u64 {
    GPU_NONCE | ((slot as u64 & 0x7f) << 56) | (counter & ((1 << 56) - 1))
}

/// How a Walk job is split between the CPU and the GPUs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// CPU threads only (`--backend cpu`, `MINER_DISABLE_GPU`, or no GPU).
    Cpu,
    /// GPUs only (`--backend gpu`), falling back to the CPU without one.
    Gpu,
    /// GPUs and CPU threads together (`--backend auto`, the default).
    Both,
}

/// How long the GPUs go on with a job nobody is waiting on. The pool miner
/// asks again within microseconds of each share; a node starts its next
/// template at once. Past this, the job is presumed over and the GPUs stop.
const IDLE_GRACE: Duration = Duration::from_secs(2);

/// The GPUs' search for one job. It outlives the [`mine_walk`] call that
/// starts it: a pool's share target is easy, so one batch of thousands of
/// walks can hold several shares, and a caller that asks again for the same
/// job (as the pool miner does after every share) gets the rest at once
/// instead of the GPUs discarding them and starting over. It ends when its
/// job is cancelled, another job starts, or nobody has asked for
/// [`IDLE_GRACE`].
struct Session {
    job: ([u8; 32], [u8; 32], Option<[u8; 32]>),
    /// The job's own cancel flag: a new block or pool job sets it.
    cancel: Arc<AtomicBool>,
    /// Set when the session ends; the walkers watch it between launches.
    stop: Arc<AtomicBool>,
    /// `mine_walk` calls now waiting on this session.
    callers: std::sync::atomic::AtomicUsize,
    /// When the last caller left, in milliseconds after `started`.
    left_at: AtomicU64,
    started: Instant,
    results: Mutex<mpsc::Receiver<MiningResult>>,
    handles: Mutex<Vec<std::thread::JoinHandle<(usize, Option<Box<dyn WalkBatcher>>)>>>,
}

impl Session {
    fn now_ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    /// Stops its GPUs, waits for them, and puts them back in the pool (all
    /// but any that failed).
    fn end(&self) {
        self.stop.store(true, Ordering::Relaxed);
        let handles = std::mem::take(&mut *self.handles.lock().unwrap());
        let back: Vec<_> = handles.into_iter().filter_map(|h| h.join().ok()).collect();
        let mut pool = POOL.lock().unwrap();
        for (slot, w) in back {
            if let (Some(w), Some(place)) = (w, pool.walkers.get_mut(slot)) {
                *place = Some(w);
            }
        }
    }
}

static SESSION: Mutex<Option<Arc<Session>>> = Mutex::new(None);

/// Ends the GPUs' current search, if any.
fn end_session() {
    let old = SESSION.lock().unwrap().take();
    if let Some(old) = old {
        old.end();
    }
}

/// Starts the GPUs on a job; `None` if there are none to start.
fn start_session(
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    cancel: &Arc<AtomicBool>,
    hash_counter: &Arc<AtomicU64>,
) -> Option<Arc<Session>> {
    let mut taken = Vec::new();
    {
        let mut pool = POOL.lock().unwrap();
        open_pool(&mut pool);
        let Pool { walkers, counters, .. } = &mut *pool;
        for (slot, w) in walkers.iter_mut().enumerate() {
            if let Some(w) = w.take() {
                taken.push((slot, w, counters[slot].hashes.clone()));
            }
        }
    }
    if taken.is_empty() {
        return None;
    }
    let (tx, rx) = mpsc::channel::<MiningResult>();
    let session = Arc::new(Session {
        job: (midstate, target, pool_target),
        cancel: cancel.clone(),
        stop: Arc::new(AtomicBool::new(false)),
        callers: Default::default(),
        left_at: AtomicU64::new(0),
        started: Instant::now(),
        results: Mutex::new(rx),
        handles: Mutex::new(Vec::new()),
    });

    // One watcher turns the job's cancel, or the session going idle, into
    // `stop`, which the walkers check between kernel launches.
    {
        let s = session.clone();
        std::thread::spawn(move || loop {
            if s.stop.load(Ordering::Relaxed) {
                return;
            }
            let idle = s.callers.load(Ordering::Acquire) == 0
                && s.now_ms().saturating_sub(s.left_at.load(Ordering::Acquire)) > IDLE_GRACE.as_millis() as u64;
            if s.cancel.load(Ordering::Relaxed) || idle {
                s.stop.store(true, Ordering::Relaxed);
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        });
    }

    let mut handles = Vec::new();
    for (slot, mut walker, counter) in taken {
        let (stop, tx, hash_counter) = (session.stop.clone(), tx.clone(), hash_counter.clone());
        handles.push(std::thread::spawn(move || {
            let mut next: u64 = rand::random::<u64>() & ((1 << 56) - 1);
            let mut nonces = vec![0u64; walker.capacity()];
            while !stop.load(Ordering::Relaxed) {
                for (i, n) in nonces.iter_mut().enumerate() {
                    *n = gpu_nonce(slot, next.wrapping_add(i as u64));
                }
                next = next.wrapping_add(nonces.len() as u64);
                let hashes = match walker.hash_batch(&midstate, &nonces, &stop) {
                    Ok(Some(h)) => h,
                    Ok(None) => break,
                    Err(e) => {
                        tracing::error!("{}: {e}; it stops mining until restart", walker.label());
                        // Kept out of the pool: a failed device is not retried.
                        return (slot, None);
                    }
                };
                hash_counter.fetch_add(hashes.len() as u64, Ordering::Relaxed);
                counter.fetch_add(hashes.len() as u64, Ordering::Relaxed);
                for (nonce, h) in nonces.iter().zip(&hashes) {
                    let block = *h < target;
                    if !block && !pool_target.is_some_and(|p| *h < p) {
                        continue;
                    }
                    // The CPU's reference decides, never the GPU.
                    let real = walk::walk_hash(&midstate, *nonce);
                    if real != *h {
                        tracing::error!(
                            "{} reported a wrong hash for nonce {nonce} (check its clocks and temperature); discarded",
                            walker.label()
                        );
                        continue;
                    }
                    let ext = Extension { nonce: *nonce, final_hash: real };
                    let _ = tx.send(if block { MiningResult::Block(ext) } else { MiningResult::Share(ext) });
                }
            }
            (slot, Some(walker))
        }));
    }
    *session.handles.lock().unwrap() = handles;
    Some(session)
}

/// Mines one Walk job: [`mine_extension`]'s contract (the first block or
/// share found, `None` once cancelled), on the CPU threads and GPUs that
/// `split` allows. Asked again for the same job, it returns first what the
/// GPUs found meanwhile.
pub fn mine_walk(
    split: Split,
    midstate: [u8; 32],
    target: [u8; 32],
    pool_target: Option<[u8; 32]>,
    threads: usize,
    cancel: Arc<AtomicBool>,
    hash_counter: Arc<AtomicU64>,
) -> Option<MiningResult> {
    WALK_ACTIVE.store(true, Ordering::Relaxed);
    let cpu_only = |hash_counter: Arc<AtomicU64>, cancel: Arc<AtomicBool>| {
        GPUS_ACTIVE.store(false, Ordering::Relaxed);
        let n = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { threads };
        LAST_CPU_THREADS.store(n as u64, Ordering::Relaxed);
        mine_extension(crate::core::types::WALK_ACTIVATION_HEIGHT, midstate, target, pool_target, threads, cancel, hash_counter)
    };

    // The GPUs' search for this job: the one running, or a new one.
    let session = if split == Split::Cpu {
        end_session();
        None
    } else {
        let mut current = SESSION.lock().unwrap();
        let same = current.as_ref().is_some_and(|s| {
            s.job == (midstate, target, pool_target) && Arc::ptr_eq(&s.cancel, &cancel) && !s.stop.load(Ordering::Relaxed)
        });
        if !same {
            if let Some(old) = current.take() {
                old.end();
            }
            *current = start_session(midstate, target, pool_target, &cancel, &hash_counter);
        }
        if let Some(s) = current.as_ref() {
            // Counted before the lock is let go, so the watcher never sees
            // this session idle while a caller is on its way in.
            s.callers.fetch_add(1, Ordering::AcqRel);
        }
        current.clone()
    };

    let Some(session) = session else {
        if split == Split::Gpu {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| tracing::warn!("Midstate Walk: GPU mining asked for, but no GPU is usable; mining on the CPU"));
        }
        return cpu_only(hash_counter, cancel);
    };
    GPUS_ACTIVE.store(true, Ordering::Relaxed);
    let cpu_threads = match (split, threads) {
        (Split::Both, 0) => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        (Split::Both, n) => n,
        _ => 0,
    };
    LAST_CPU_THREADS.store(cpu_threads as u64, Ordering::Relaxed);

    // The CPU threads work for this call only.
    let call_stop = Arc::new(AtomicBool::new(false));
    let (cpu_tx, cpu_rx) = mpsc::channel::<MiningResult>();
    let cpu = (cpu_threads > 0).then(|| {
        let (stop, hash_counter) = (call_stop.clone(), hash_counter.clone());
        std::thread::spawn(move || {
            if let Some(r) = mine_extension(
                crate::core::types::WALK_ACTIVATION_HEIGHT,
                midstate,
                target,
                pool_target,
                cpu_threads,
                stop,
                hash_counter,
            ) {
                let _ = cpu_tx.send(r);
            }
        })
    });

    let mut result = None;
    let mut gpus_gone = false;
    while !cancel.load(Ordering::Relaxed) {
        if let Ok(r) = cpu_rx.try_recv() {
            result = Some(r);
            break;
        }
        match session.results.lock().unwrap().recv_timeout(Duration::from_millis(20)) {
            Ok(r) => {
                result = Some(r);
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // Every GPU has stopped or failed.
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                gpus_gone = true;
                if cpu.is_none() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
    call_stop.store(true, Ordering::Relaxed);
    if let Some(c) = cpu {
        let _ = c.join();
    }
    if result.is_none() {
        result = cpu_rx.try_recv().ok();
    }
    session.left_at.store(session.now_ms(), Ordering::Release);
    session.callers.fetch_sub(1, Ordering::AcqRel);

    if cancel.load(Ordering::Relaxed) {
        // The job is over: free the GPUs for the next one now.
        let mut current = SESSION.lock().unwrap();
        if current.as_ref().is_some_and(|s| Arc::ptr_eq(s, &session)) {
            current.take();
            drop(current);
            session.end();
        }
        return None;
    }
    if result.is_none() && gpus_gone {
        // The GPUs failed mid-job: the CPU carries on with it.
        tracing::warn!("Midstate Walk: no GPU left mining; going on with the CPU");
        return cpu_only(hash_counter, cancel);
    }
    result
}

/// Whether `algo` is the one these GPUs mine.
pub fn mines(algo: PowAlgo) -> bool {
    algo == PowAlgo::Walk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernels_constants_are_walk_cores() {
        // walk_core.h's context_keys() and build_te(), printed by its own code.
        let want_keys: [u32; 24] = [
            0x783a53be, 0x7bbc0d85, 0x58516945, 0xde95c1bc, 0xd8624c0d, 0xc84972aa, 0x10f95d0d, 0x036cdd20,
            0x6b899b7d, 0x7c47fac9, 0xaeec5879, 0x40e56086, 0x86d35318, 0x460c91b7, 0x47af0ae8, 0x843e4714,
            0xac57d11d, 0xf2f25f89, 0xd77adddf, 0x49f1fa04, 0x9d170811, 0xa21c478d, 0x78da02df, 0x933bff8b,
        ];
        assert_eq!(context_words(), want_keys);
        let te = te_words();
        assert_eq!((te[0], te[1], te[255]), (0xa56363c6, 0x847c7cf8, 0x3a16162c));
    }

    #[test]
    fn the_ptx_has_the_kernels_the_miner_loads() {
        for name in ["ms_init_main", "ms_steps_main", "ms_init_fast", "ms_steps_fast", "ms_final"] {
            assert!(PTX.contains(&format!(".visible .entry {name}(")), "{name}");
        }
        assert!(PTX.contains(".visible .const .align 4 .b8 ms_te[1024]"));
        assert!(PTX.contains(".visible .const .align 4 .b8 ms_keys[96]"));
        assert!(!PTX.contains(".local"), "a kernel spills to local memory");
        assert!(PTX.starts_with("// sources sha256 "));
        // clang's software 64-bit rotate, which shifts by the amount unreduced
        // and gives 0 for most amounts (walk_core.h's rotr64 says why).
        assert!(!PTX.contains(".reg .b64 %lhs"), "a 64-bit rotate clang compiles wrongly");
    }

    #[test]
    fn gpu_nonces_never_meet_the_cpus_or_each_other() {
        assert!(gpu_nonce(0, 0) >= GPU_NONCE);
        assert_ne!(gpu_nonce(0, 5), gpu_nonce(1, 5));
        assert_eq!(gpu_nonce(3, (1 << 56) + 7), gpu_nonce(3, 7), "the counter wraps within its slot");
        for _ in 0..1000 {
            assert!(crate::core::extension::cpu_nonce() < GPU_NONCE);
        }
    }
}
