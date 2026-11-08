//! The search around the CUDA walker, with CPU stand-ins for GPUs: which
//! backend a job runs on at each height, how GPUs and CPU threads share a
//! job, and that only results the CPU's reference agrees with come out. The
//! kernels themselves are checked against the same known answers when a real
//! GPU opens (`cuda_walk::CudaWalker::open`), and by CI on the CPU.
//!
//! One test, because the walkers live in a process-wide pool.

#![cfg(not(target_arch = "wasm32"))]

use midstate::core::cuda_walk::{self, install_for_tests, mine_walk, Split, WalkBatcher};
use midstate::core::extension::MiningResult;
use midstate::core::gpu_mining::{self, Backend};
use midstate::core::types::WALK_ACTIVATION_HEIGHT;
use midstate::core::walk;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const GPU_BIT: u64 = 1 << 63;

/// A GPU that hashes on the CPU, correctly.
struct Honest(&'static str);
impl WalkBatcher for Honest {
    fn label(&self) -> String {
        self.0.to_string()
    }
    fn capacity(&self) -> usize {
        2
    }
    fn hash_batch(&mut self, h: &[u8; 32], nonces: &[u64], cancel: &AtomicBool) -> Result<Option<Vec<[u8; 32]>>, String> {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        Ok(Some(nonces.iter().map(|n| walk::walk_hash(h, *n)).collect()))
    }
}

/// A GPU gone wrong: every answer looks like a block, and none is right.
struct Broken;
impl WalkBatcher for Broken {
    fn label(&self) -> String {
        "broken".into()
    }
    fn capacity(&self) -> usize {
        4
    }
    fn hash_batch(&mut self, _: &[u8; 32], nonces: &[u64], cancel: &AtomicBool) -> Result<Option<Vec<[u8; 32]>>, String> {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(5));
        Ok(Some(vec![[0u8; 32]; nonces.len()]))
    }
}

fn job() -> ([u8; 32], Arc<AtomicBool>, Arc<AtomicU64>) {
    (rand::random(), Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)))
}

fn nonce_of(r: &MiningResult) -> (u64, [u8; 32]) {
    match r {
        MiningResult::Block(e) | MiningResult::Share(e) => (e.nonce, e.final_hash),
    }
}

#[test]
fn the_walk_search_with_stand_in_gpus() {
    let easy = [0xffu8; 32];
    let never = [0u8; 32];

    // GPUs only: the share comes from a GPU, and is the reference's hash.
    install_for_tests(vec![Box::new(Honest("gpu a")), Box::new(Honest("gpu b"))]);
    let (mh, cancel, counter) = job();
    let r = mine_walk(Split::Gpu, mh, never, Some(easy), 2, cancel.clone(), counter.clone()).expect("a share");
    let (nonce, hash) = nonce_of(&r);
    assert!(matches!(r, MiningResult::Share(_)));
    assert!(nonce & GPU_BIT != 0, "found by a GPU");
    assert_eq!(hash, walk::walk_hash(&mh, nonce));
    assert!(counter.load(Ordering::Relaxed) >= 2);
    let status = cuda_walk::backend_status(counter.load(Ordering::Relaxed));
    assert!(status.iter().all(|b| b.kind == "cuda"), "no CPU in a GPU-only job: {status:?}");
    assert_eq!(status.len(), 2);

    // Asked again for the same job, as the pool miner does after each share:
    // the GPUs carried on, and the next share is a different one.
    let r = mine_walk(Split::Gpu, mh, never, Some(easy), 2, cancel.clone(), counter.clone()).expect("another share");
    let (again, hash) = nonce_of(&r);
    assert_ne!(again, nonce);
    assert_eq!(hash, walk::walk_hash(&mh, again));

    // Nobody asks again: the GPUs stop on their own.
    let gpu_hashes = || -> u64 { cuda_walk::backend_status(0).iter().map(|b| b.hashes).sum() };
    std::thread::sleep(Duration::from_secs(4));
    let before = gpu_hashes();
    std::thread::sleep(Duration::from_millis(700));
    assert_eq!(gpu_hashes(), before, "an abandoned job stops");

    // CPU only: the GPUs are left alone.
    let (mh, cancel, counter) = job();
    let r = mine_walk(Split::Cpu, mh, never, Some(easy), 1, cancel, counter).expect("a share");
    assert!(nonce_of(&r).0 & GPU_BIT == 0, "found by the CPU");

    // Both: whoever finds first, it is right; the status lists both kinds.
    let (mh, cancel, counter) = job();
    let r = mine_walk(Split::Both, mh, never, Some(easy), 1, cancel, counter.clone()).expect("a share");
    let (nonce, hash) = nonce_of(&r);
    assert_eq!(hash, walk::walk_hash(&mh, nonce));
    let kinds: Vec<String> = cuda_walk::backend_status(counter.load(Ordering::Relaxed)).into_iter().map(|b| b.kind).collect();
    assert_eq!(kinds, ["cpu", "cuda", "cuda"]);

    // A GPU whose answers are wrong never produces a result, and a cancel
    // ends the job promptly.
    install_for_tests(vec![Box::new(Broken)]);
    let (mh, cancel, counter) = job();
    let c = cancel.clone();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        c.store(true, Ordering::Relaxed);
    });
    let t = Instant::now();
    assert!(mine_walk(Split::Gpu, mh, never, Some(easy), 1, cancel, counter).is_none(), "a wrong hash is discarded");
    assert!(t.elapsed() < Duration::from_secs(5), "cancel stops the job");
    stopper.join().unwrap();

    // No GPU at all: GPU mining asked for falls back to the CPU.
    install_for_tests(vec![]);
    let (mh, cancel, counter) = job();
    let r = mine_walk(Split::Gpu, mh, never, Some(easy), 1, cancel, counter).expect("a share from the CPU");
    assert!(nonce_of(&r).0 & GPU_BIT == 0);

    // Through gpu_mining::mine, as the node, the pool miner and `midstate
    // miner` call it: the walk from WALK_ACTIVATION_HEIGHT, on the GPUs when
    // asked; the BLAKE3 chain below it, untouched.
    install_for_tests(vec![Box::new(Honest("gpu a"))]);
    gpu_mining::set_backend(Backend::Gpu);
    let (mh, cancel, counter) = job();
    let r = gpu_mining::mine(WALK_ACTIVATION_HEIGHT, mh, never, Some(easy), 1, cancel, counter).expect("a share");
    let (nonce, hash) = nonce_of(&r);
    assert!(nonce & GPU_BIT != 0);
    assert_eq!(hash, walk::walk_hash(&mh, nonce));

    gpu_mining::set_backend(Backend::Cpu);
    let (mh, cancel, counter) = job();
    let r = gpu_mining::mine(WALK_ACTIVATION_HEIGHT - 1, mh, never, Some(easy), 1, cancel, counter).expect("a share");
    let (nonce, hash) = nonce_of(&r);
    assert!(nonce & GPU_BIT == 0);
    assert_eq!(
        hash,
        midstate::core::extension::create_extension_at(WALK_ACTIVATION_HEIGHT - 1, mh, nonce).final_hash,
        "below the switch it is still the BLAKE3 chain"
    );
}
