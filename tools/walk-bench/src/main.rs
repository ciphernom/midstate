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
//! midstate-walk-bench cpu-probe
//! ```
//!
//! `cpu-probe` shows where a step's time goes on this CPU: how long a load
//! takes from each level of cache, and what a step costs with each of the
//! walk's operations, in cycles at the clock it runs at, for each way the
//! step can be written (what mines, what mined before, a candidate, and the
//! step without its writes to the pad).
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
  interleave   1 to 4 walks stepped in turn per thread, each way of writing the step
  gpu          the Walk on a GPU through wgpu (Vulkan, Metal, DX12), checked first
  dump-wgsl F  write the GPU kernel to file F, to edit and run with gpu --wgsl F
  cpu-probe    where a step's time goes: cache latencies and each operation's cost

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

/// The CPU's clock now, MHz (cpu0's, as the kernel reports it).
fn mhz() -> Option<f64> {
    let khz: f64 = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").ok()?.trim().parse().ok()?;
    Some(khz / 1000.0)
}

/// Nanoseconds a load takes when its address is the last load's result, on a
/// random cycle through `bytes` of 64-byte lines: the cache's latency for a
/// working set that size, as a walk's step meets it.
fn chase_ns(bytes: usize) -> f64 {
    let lines = (bytes / 64).max(2);
    // One random cycle through every line (Sattolo's shuffle), a pointer a line.
    let mut order: Vec<usize> = (0..lines).collect();
    let mut x = 0x2545_f491_4f6c_dd1du64 ^ bytes as u64;
    for i in (1..lines).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        order.swap(i, (x % i as u64) as usize);
    }
    let mut next = vec![0usize; lines * 8];
    for i in 0..lines {
        next[order[i] * 8] = order[(i + 1) % lines] * 8;
    }
    let loads = 1usize << 22;
    let mut best = f64::MAX;
    let mut p = 0usize;
    for _ in 0..lines * 2 {
        p = next[p];
    }
    for _ in 0..3 {
        let begun = std::time::Instant::now();
        for _ in 0..loads {
            p = next[p];
        }
        best = best.min(begun.elapsed().as_nanos() as f64 / loads as f64);
    }
    std::hint::black_box(p);
    best
}

fn cpu_probe() {
    let page = std::process::Command::new("getconf").arg("PAGESIZE").output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let f0 = mhz();
    println!(
        "Where a step's time goes, on one thread. Clock {}, AES {}, pages of {} bytes.",
        f0.map_or("unknown".into(), |f| format!("{f:.0} MHz")),
        walk::backend_name(),
        page.as_deref().unwrap_or("?")
    );
    let cyc = |ns: f64, f: Option<f64>| f.map_or(String::new(), |f| format!("{:>6.1} cycles", ns * f / 1000.0));

    println!("\nA load whose address is the last load's result, by working set:");
    let mut l2 = None;
    for kib in [16usize, 32, 48, 64, 128, 256, 384, 512, 1024, 2048, 4096, 16384] {
        let ns = chase_ns(kib * 1024);
        if kib == 256 {
            l2 = Some(ns);
        }
        println!("  {:>6} KiB {:>7.2} ns {}{}", kib, ns, cyc(ns, f0), if kib == 256 { "   <- the pad's size" } else { "" });
    }

    use walk::StepCode;
    let codes = [
        (StepCode::Indexed, "indexed"),
        (StepCode::Unindexed, "unindexed"),
        (StepCode::Ahead, "ahead"),
        (StepCode::NoStores, "no writes"),
    ];
    let mine = codes.iter().position(|(c, _)| *c == walk::MINER_STEP).unwrap();
    walk::time_steps(None, 1 << 16, walk::MINER_STEP);
    let show = |ns: f64| match f0 {
        Some(f) => format!("{:>9.1}", ns * f / 1000.0),
        None => format!("{:>9.2}", ns),
    };
    let names = ["mul", "mulhi", "div", "aes", "rotr", "add", "sub", "xor"];
    // [code][0] the walk's own mix of operations, [code][1 + op] every step that operation.
    let mut t = [[0.0f64; 9]; 4];
    for (k, (code, _)) in codes.iter().enumerate() {
        t[k][0] = walk::time_steps(None, 1 << 20, *code);
        for op in 0..8u8 {
            t[k][1 + op as usize] = walk::time_steps(Some(op), 1 << 19, *code);
        }
    }
    println!("\nA step, in {}, by how it is written (* is what mines on this CPU):", if f0.is_some() { "cycles" } else { "ns" });
    println!("  indexed    the reference, registers indexed through memory (mined before)");
    println!("  unindexed  no register through memory");
    println!("  ahead      unindexed, and the next step's address one XOR after the operation");
    println!("  no writes  unindexed without its four writes to the pad: not the walk, a probe");
    print!("  {:<22}", "");
    for (k, (_, name)) in codes.iter().enumerate() {
        print!("{:>10}", if k == mine { format!("{name}*") } else { name.to_string() });
    }
    println!();
    for row in 0..9 {
        let label = if row == 0 { "its own mix".to_string() } else { format!("every step {}", names[row - 1]) };
        print!("  {label:<22}");
        for k in 0..codes.len() {
            print!(" {}", show(t[k][row]));
        }
        println!();
    }
    let pct = |a: f64, b: f64| 100.0 * (a / b - 1.0);
    for k in 0..3 {
        if k != mine {
            println!("  {} against what mines: {:+.1}%", codes[k].1, pct(t[k][0], t[mine][0]));
        }
    }
    println!("  the four writes to the pad are {:.1}% of the unindexed step", 100.0 * (1.0 - t[3][0] / t[1][0]));

    // Whole walks: the measure that counts, as steps alone leave out the
    // re-mixes and the loop around them.
    let mut pad = walk::new_pad();
    let m = *blake3::hash(b"cpu-probe").as_bytes();
    let mut whole = [0.0f64; 3];
    for _ in 0..2 {
        for (k, (code, _)) in codes[..3].iter().enumerate() {
            let begun = std::time::Instant::now();
            let mut n = 0u64;
            while begun.elapsed() < Duration::from_millis(1500) {
                std::hint::black_box(walk::walk_hash_written(*code, &m, n, &mut pad));
                n += 1;
            }
            whole[k] = whole[k].max(n as f64 / begun.elapsed().as_secs_f64());
        }
    }
    print!("\nWhole walks on one thread, attempts/s:");
    for (k, (_, name)) in codes[..3].iter().enumerate() {
        print!("  {}{} {:.1}", name, if k == mine { "*" } else { "" }, whole[k]);
    }
    println!();
    let best = (0..3).max_by(|&a, &b| whole[a].total_cmp(&whole[b])).unwrap();
    if best != mine && whole[best] > whole[mine] * 1.02 {
        println!("  {} is {:.1}% faster here than what mines: worth telling us", codes[best].1, pct(whole[best], whole[mine]));
    }
    let same = (0..3).all(|n| {
        let mut st = walk::walk_start(&m, n, &mut pad);
        walk::walk_steps(&mut st, &mut pad, walk::WALK_STEPS);
        let reference = walk::walk_finish(&m, n, &st.r);
        walk::walk_hash(&m, n) == reference
            && [StepCode::Indexed, StepCode::Unindexed, StepCode::Ahead].iter().all(|c| walk::walk_hash_written(*c, &m, n, &mut pad) == reference)
    });
    println!("  every way computes the same hashes as the reference: {}", if same { "yes" } else { "NO" });
    let all = t[mine][0];
    let xor = t[mine][8];

    let start = walk::time_start(32);
    let remix = walk::time_remix(20_000);
    let steps = walk::WALK_STEPS as f64;
    let attempt = steps * all + start + 64.0 * remix;
    let (measured, _) = walk::bench(1, Duration::from_secs(3));
    let f1 = mhz();
    println!("\nAn attempt on one thread:");
    println!("  {} steps          {:>8.3} ms", walk::WALK_STEPS, steps * all / 1e6);
    println!("  seed and pad          {:>8.3} ms", start / 1e6);
    println!("  64 re-mixes           {:>8.3} ms", 64.0 * remix / 1e6);
    println!("  so {:.1} attempts/s; measured {:.1}", 1e9 / attempt, measured);

    if let (Some(f), Some(l2)) = (f0, l2) {
        let c = |ns: f64| ns * f / 1000.0;
        println!("\nSo, of {:.1} cycles a step:", c(all));
        println!("  {:>5.1} the load from the pad (a load from 256 KiB)", c(l2));
        println!("  {:>5.1} the rest of the step besides its operation (an all-xor step less the load)", c(xor) - c(l2));
        println!("  {:>5.1} the operations, on average", c(all) - c(xor));
    }
    if let (Some(a), Some(b)) = (f0, f1) {
        if (a - b).abs() > 50.0 {
            println!("The clock moved from {a:.0} to {b:.0} MHz during the probe (throttling?): its cycle counts are rough.");
        }
    }
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
        "cpu-probe" => cpu_probe(),
        "-h" | "--help" | "help" => print!("{USAGE}"),
        other => fail(format!("unknown command {other}")),
    }
}
