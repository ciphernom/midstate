//! `midstate-walk-bench`: Midstate Walk measured without the rest of
//! midstate (no node, wallet, networking or fat LTO), so it builds in a
//! fraction of the time. It compiles the miner's own `src/core/walk.rs` and
//! `src/core/wgpu_walk.rs`, not copies, so what it measures is what mines.
//!
//! ```text
//! midstate-walk-bench cpu [--seconds S] [--threads T]
//! midstate-walk-bench interleave [--seconds S] [--threads T]
//! midstate-walk-bench gpu [--with-cpu] [--adapter NAME] [--walks N] [--seconds S] [--wgsl FILE]
//! midstate-walk-bench dump-wgsl FILE
//! ```
//!
//! `dump-wgsl` writes the GPU kernel out, and `gpu --wgsl` runs an edited
//! one: a change to the kernel tried on a GPU with nothing compiled.

#[allow(dead_code)]
#[path = "../../../src/core/walk.rs"]
mod walk;
#[allow(dead_code)]
#[path = "../../../src/core/wgpu_walk.rs"]
mod wgpu_walk;

use std::time::Duration;

const USAGE: &str = "\
usage: midstate-walk-bench <command> [options]

  cpu          the Walk on 1 thread and on all of them, and the known answers
  interleave   1, 2 and 3 walks stepped in turn per thread, pinned and not
  gpu          the Walk on a GPU through wgpu (Vulkan, Metal, DX12), checked first
  dump-wgsl F  write the GPU kernel to file F, to edit and run with gpu --wgsl F

options:
  --seconds S   seconds each measurement runs (default 10)
  --threads T   CPU threads (default: every core)
  --adapter N   gpu: part of the GPU's name (default: the first hardware GPU)
  --walks N     gpu: the most walks at once to try (default 1024)
  --with-cpu    gpu: also run the CPU and the GPU together
  --wgsl F      gpu: run the kernel in file F instead of the built-in one
";

/// The CPU's temperature, °C, from the kernel's first thermal zone.
fn temp_c() -> Option<f32> {
    let milli: f32 = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?.trim().parse().ok()?;
    Some(milli / 1000.0)
}

/// The temperature and the CPU's clock now, in words.
fn board() -> String {
    let mhz = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        .ok()
        .and_then(|k| k.trim().parse::<u64>().ok())
        .map(|k| format!("{} MHz", k / 1000));
    match (temp_c().map(|t| format!("{t:.1} °C")), mhz) {
        (Some(t), Some(m)) => format!("{t}, {m}"),
        (Some(t), None) => t,
        (None, Some(m)) => m,
        (None, None) => "no sensors".into(),
    }
}

fn cpu(seconds: u64, threads: usize) {
    let threads = if threads == 0 { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) } else { threads };
    let d = Duration::from_secs(seconds.max(1));
    println!(
        "Midstate Walk: {} KiB scratchpad, {} steps, AES: {}",
        walk::WALK_LINES * 64 / 1024,
        walk::WALK_STEPS,
        walk::backend_name()
    );
    let (one, good1) = walk::bench(1, d);
    println!("  1 thread   {one:>10.1} attempts/s");
    let mut good = good1;
    if threads > 1 {
        let (all, good_all) = walk::bench(threads, d);
        good &= good_all;
        println!(
            "  {threads} threads {all:>10.1} attempts/s  ({:.1} per thread, {:.0}% of {threads} x 1 thread)",
            all / threads as f64,
            100.0 * all / (one * threads as f64)
        );
    }
    println!(
        "  known answers: {}  ({} after)",
        if good { "right on every thread, before and after" } else { "WRONG on a thread: this CPU computes the walk incorrectly; lower its overclock" },
        board()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().map(String::as_str) else {
        print!("{USAGE}");
        return;
    };
    let mut seconds = 10u64;
    let mut threads = 0usize;
    let mut adapter = None;
    let mut walks = 1024u32;
    let mut with_cpu = false;
    let mut wgsl = None;
    let mut rest = args[1..].iter();
    let mut positional = Vec::new();
    let fail = |msg: String| -> ! {
        eprintln!("{msg}\n\n{USAGE}");
        std::process::exit(2);
    };
    while let Some(a) = rest.next() {
        let mut value = |name: &str| rest.next().cloned().unwrap_or_else(|| fail(format!("{name} needs a value")));
        match a.as_str() {
            "--seconds" => seconds = value("--seconds").parse().unwrap_or_else(|_| fail("--seconds takes a number".into())),
            "--threads" => threads = value("--threads").parse().unwrap_or_else(|_| fail("--threads takes a number".into())),
            "--adapter" => adapter = Some(value("--adapter")),
            "--walks" => walks = value("--walks").parse().unwrap_or_else(|_| fail("--walks takes a number".into())),
            "--with-cpu" => with_cpu = true,
            "--wgsl" => {
                let path = value("--wgsl");
                wgsl = Some(std::fs::read_to_string(&path).unwrap_or_else(|e| fail(format!("{path}: {e}"))));
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return;
            }
            other if other.starts_with("--") => fail(format!("unknown option {other}")),
            other => positional.push(other.to_string()),
        }
    }
    match command {
        "cpu" => cpu(seconds, threads),
        "interleave" => walk::print_interleave(seconds, threads, temp_c),
        "gpu" => wgpu_walk::bench(&wgpu_walk::BenchOptions { adapter, max_walks: walks, seconds, threads, with_cpu, board, wgsl }),
        "dump-wgsl" => {
            let path = positional.first().cloned().unwrap_or_else(|| fail("dump-wgsl needs a file to write".into()));
            std::fs::write(&path, wgpu_walk::kernel_source()).unwrap_or_else(|e| fail(format!("{path}: {e}")));
            println!("Wrote the kernel to {path}. Run an edited copy with: midstate-walk-bench gpu --wgsl {path}");
        }
        "-h" | "--help" | "help" => print!("{USAGE}"),
        other => fail(format!("unknown command {other}")),
    }
}
