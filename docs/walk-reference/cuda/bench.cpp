// bench.cpp: walk-cuda, a Midstate Walk benchmark for NVIDIA GPUs.
//
// In order, it:
//   1. checks the GPU against midstate's known answers, and stops if it's wrong;
//   2. finds how many walks in flight the GPU needs to reach full speed;
//   3. runs that many for a while, reading the board's power through NVML;
//   4. checks a sample of what the GPU computed against the CPU reference.
// The last line of output is the one to post.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu.h"
#include "nvml_power.h"
#include "walk_core.h"

namespace {

const char* const VERSION = "2";
const uint32_t ALL = UINT32_MAX;

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

struct Options {
  int device = 0;
  bool fast = false;
  uint32_t batch = 0;   // walks in flight; 0 picks the fewest that reach full speed
  uint32_t block = 128;
  double seconds = 30;
  int sweep = -1;       // 0 off, 1 until the rate levels off, 2 every size; -1 decides from --batch
  uint64_t start = 0;
  uint8_t hash[32] = {};
  uint32_t verify = 4096;  // walks checked from the first batch and from the last
};

void usage(FILE* f) {
  fputs(
      "usage: walk-cuda [options]\n"
      "\n"
      "Checks the GPU against midstate's known answers, finds how many walks in flight it\n"
      "needs, runs them for a while reading power through NVML, then checks a sample of the\n"
      "results on the CPU. Post the last line.\n"
      "\n"
      "  --seconds S     length of the sustained run (default 30; 0 skips it)\n"
      "  --batch N       walks in flight (default: the fewest that reach full speed)\n"
      "  --sweep         time every batch size that fits, not just until the rate levels off\n"
      "  --no-sweep      skip the sweep (runs --batch, or every walk the GPU can hold)\n"
      "  --threads N     threads a block: 32, 64, 128 (default) or 256\n"
      "  --verify N|all  walks from the first and last batch to check on the CPU (default 4096)\n"
      "  --no-verify     skip that check (the known answers are always checked)\n"
      "  --hash HEX      the mining hash, 64 hex digits (default all zeros)\n"
      "  --start N       the first nonce (default 0)\n"
      "  --fast          midstate's fast-mining test parameters: a correctness check, not a benchmark\n"
      "  --device N      which GPU (default 0)\n",
      f);
}

[[noreturn]] void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fputs("walk-cuda: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  exit(1);
}

bool parse_u64(const char* s, uint64_t* out) {
  if (!s || !*s || *s == '-' || *s == '+') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = strtoull(s, &end, 10);
  if (errno || *end) return false;
  *out = v;
  return true;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto value = [&]() -> const char* {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    uint64_t v = 0;
    if (a == "--help" || a == "-h") {
      usage(stdout);
      exit(0);
    } else if (a == "--seconds") {
      const char* s = value();
      char* end = nullptr;
      o.seconds = strtod(s, &end);
      if (!*s || *end || !(o.seconds >= 0) || o.seconds > 86400) die("--seconds wants a number of seconds");
    } else if (a == "--batch") {
      if (!parse_u64(value(), &v) || v == 0 || v > (1u << 24)) die("--batch wants a number of walks");
      o.batch = (uint32_t)v;
    } else if (a == "--sweep") {
      o.sweep = 2;
    } else if (a == "--no-sweep") {
      o.sweep = 0;
    } else if (a == "--threads") {
      if (!parse_u64(value(), &v) || (v != 32 && v != 64 && v != 128 && v != 256))
        die("--threads wants 32, 64, 128 or 256");
      o.block = (uint32_t)v;
    } else if (a == "--verify") {
      // A bare --verify (the old flag) checks everything.
      if (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0) {
        const char* s = value();
        if (!strcmp(s, "all"))
          o.verify = ALL;
        else if (parse_u64(s, &v) && v <= ALL)
          o.verify = (uint32_t)v;
        else
          die("--verify wants a number of walks, or all");
      } else {
        o.verify = ALL;
      }
    } else if (a == "--no-verify") {
      o.verify = 0;
    } else if (a == "--hash") {
      if (!walk::from_hex(value(), o.hash, 32)) die("--hash wants 64 hex digits");
    } else if (a == "--start") {
      if (!parse_u64(value(), &o.start)) die("--start wants a nonce");
    } else if (a == "--fast") {
      o.fast = true;
    } else if (a == "--device") {
      if (!parse_u64(value(), &v) || v > 64) die("--device wants a GPU number");
      o.device = (int)v;
    } else {
      fprintf(stderr, "walk-cuda: unknown option %s\n\n", a.c_str());
      usage(stderr);
      exit(1);
    }
  }
  if (o.sweep < 0) o.sweep = o.batch ? 0 : 1;
  return o;
}

// 1234567 → "1,234,567".
std::string group(uint64_t v) {
  std::string s = std::to_string(v);
  for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, ",");
  return s;
}

std::string fmt(const char* f, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

// Attempts a second: whole numbers from 100 up, one decimal below.
std::string rate_text(double r) { return r >= 100 ? group((uint64_t)std::llround(r)) : fmt("%.1f", r); }

std::string seconds_text(double s) {
  return s >= 10 ? fmt("%.0f s", s) : s >= 0.1 ? fmt("%.2f s", s) : s >= 1e-3 ? fmt("%.1f ms", s * 1e3) : fmt("%.1f µs", s * 1e6);
}

// Gigabytes a second moved by `rate` attempts a second, if every step reads
// four random 32-byte sectors and writes four back: the memory-bound model.
double implied_gbs(const walk::Params& p, double rate) { return rate * p.steps * 8 * 32 / 1e9; }

[[noreturn]] void gpu_failed(gpu::Context* c) {
  fprintf(stderr, "\nwalk-cuda: %s\n", gpu::error(c));
  if (strstr(gpu::error(c), "timeout") || strstr(gpu::error(c), "timed out"))
    fprintf(stderr, "A display watchdog stopped a kernel. Try a smaller --batch, or a GPU without a display.\n");
  exit(2);
}

// Runs walks 0..walks-1 (whose jobs are set) from start to finish.
void run_walks(gpu::Context* c, const walk::Params& p, uint32_t walks, uint32_t chunk) {
  if (!gpu::launch_init(c, walks)) gpu_failed(c);
  for (uint32_t s = 0; s < p.steps; s += chunk)
    if (!gpu::launch_steps(c, walks, s, chunk)) gpu_failed(c);
  if (!gpu::launch_final(c, walks)) gpu_failed(c);
}

struct Point {
  uint32_t walks;
  double rate;          // attempts a second at this pace
  double walk_seconds;  // how long one walk takes in flight
};

// Times `walks` walks in flight. Steps them for long enough to time (past the
// end of a walk if need be: the work per step doesn't change) and halves
// `chunk` if a launch takes over half a second.
Point measure(gpu::Context* c, const walk::Params& p, uint32_t walks, uint32_t& chunk, const Options& o) {
  std::vector<uint64_t> nonces(walks);
  for (uint32_t w = 0; w < walks; ++w) nonces[w] = o.start + w;
  if (!gpu::set_jobs(c, o.hash, 1, nonces.data(), walks) || !gpu::launch_init(c, walks) || !gpu::sync(c))
    gpu_failed(c);
  uint32_t first = 0;
  double one = 0;
  for (;;) {
    const Clock::time_point t = Clock::now();
    if (!gpu::launch_steps(c, walks, first, chunk) || !gpu::sync(c)) gpu_failed(c);
    one = since(t);
    first += chunk;
    if (one <= 0.5 || chunk <= 64) break;
    chunk /= 2;
  }
  const int n = (int)std::min(256.0, std::max(2.0, std::ceil(0.4 / std::max(one, 1e-6))));
  const Clock::time_point t = Clock::now();
  for (int i = 0; i < n; ++i, first += chunk)
    if (!gpu::launch_steps(c, walks, first, chunk)) gpu_failed(c);
  if (!gpu::sync(c)) gpu_failed(c);
  const double per_walk = (double)n * chunk / since(t);  // steps a second, each walk
  return Point{walks, walks * per_walk / p.steps, p.steps / per_walk};
}

struct Check {
  uint32_t checked = 0, wrong = 0;
  int64_t first_wrong = -1;
};

// Recomputes a sample of one batch's walks on the CPU (every core) and
// compares. The sample is evenly spread and includes the first and last walk.
Check verify(const walk::Params& p, const walk::Keys& keys, const uint8_t hash[32], uint64_t first_nonce,
             const std::vector<uint8_t>& got, uint32_t walks, uint32_t sample) {
  std::vector<uint32_t> idx;
  if (sample >= walks) {
    for (uint32_t w = 0; w < walks; ++w) idx.push_back(w);
  } else if (sample == 1) {
    idx.push_back(0);
  } else {
    for (uint32_t i = 0; i < sample; ++i) idx.push_back((uint32_t)((uint64_t)i * (walks - 1) / (sample - 1)));
  }
  std::atomic<size_t> next{0};
  std::mutex m;
  Check c;
  c.checked = (uint32_t)idx.size();
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&] {
      std::vector<uint64_t> pad((size_t)p.lines * 8);
      for (size_t i; (i = next++) < idx.size();) {
        const uint32_t w = idx[i];
        uint8_t want[32];
        walk::hash_host(p, keys, hash, first_nonce + w, pad.data(), want);
        if (memcmp(want, &got[(size_t)w * 32], 32) != 0) {
          std::lock_guard<std::mutex> lock(m);
          ++c.wrong;
          if (c.first_wrong < 0 || w < c.first_wrong) c.first_wrong = w;
        }
      }
    });
  for (std::thread& t : pool) t.join();
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  const walk::Params& p = o.fast ? walk::FAST : walk::MAINNET;
  const walk::Keys keys = walk::context_keys();
  char err[512];
  setvbuf(stdout, nullptr, _IOLBF, 0);

  gpu::Info info;
  if (!gpu::open(o.device, &info, err, sizeof err)) {
    fprintf(stderr, "walk-cuda: %s\n", err);
    return 2;
  }
  gpu::KernelInfo ki;
  if (!gpu::kernel_info(o.fast, o.block, info.sms, &ki, err, sizeof err)) {
    fprintf(stderr, "walk-cuda: %s\n", err);
    return 2;
  }
  if (ki.resident_walks == 0) die("the kernel can't run %u threads a block on this GPU; try --threads 64", o.block);
  PowerSampler power(info.pci_bus_id);
  // The driver's version through NVML (e.g. 580.173.02), else the CUDA version it supports.
  const std::string cuda = fmt("CUDA %d.%d", info.driver_version / 1000, info.driver_version % 1000 / 10);
  const std::string driver = power.driver().empty() ? cuda : "driver " + power.driver();

  // ── What we're running on ──
  const double gib = 1024.0 * 1024 * 1024;
  const double peak_gbs = 2.0 * info.mem_clock_khz * 1e3 * (info.bus_width_bits / 8) / 1e9;
  printf("Midstate Walk GPU bench %s, %s (%s scratchpad, %s steps an attempt)\n", VERSION,
         o.fast ? "fast-mining TEST parameters" : "mainnet parameters", o.fast ? "16 KiB" : "256 KiB",
         group(p.steps).c_str());
  printf("GPU %d       %s · compute %d.%d · %d SMs · %.1f MiB L2 · %.1f GiB (%.1f GiB free)\n", o.device,
         info.name, info.major, info.minor, info.sms, info.l2_bytes / 1048576.0, info.total_mem / gib,
         info.free_mem / gib);
  printf("            memory %d-bit at %.1f GT/s: %.0f GB/s peak · %s%s driver, %d.%d runtime\n", info.bus_width_bits,
         2.0 * info.mem_clock_khz / 1e6, peak_gbs, power.driver().empty() ? "" : (driver + " · ").c_str(),
         cuda.c_str(), info.runtime_version / 1000, info.runtime_version % 1000 / 10);
  if (info.watchdog) printf("            drives a display: every launch is kept short, well inside its watchdog\n");
  printf("Kernel      %d registers a thread, %s · %u threads a block · %s walks fit in flight at once\n", ki.regs,
         ki.local_bytes ? fmt("%d bytes of local memory a thread", ki.local_bytes).c_str() : "no local memory",
         o.block, group(ki.resident_walks).c_str());

  // ── Room ──
  const double walk_bytes = (double)p.lines * 64 + 8 * 8 + 8 + 4 + 8 + 32 + 32;
  const uint64_t fits = (uint64_t)(0.85 * (double)info.free_mem / walk_bytes);
  if (fits < 3) die("not enough free GPU memory");
  if (o.batch > fits)
    die("%s walks need %.1f GiB of GPU memory and %.1f GiB is free", group(o.batch).c_str(),
        o.batch * walk_bytes / gib, info.free_mem / gib);
  const uint32_t capacity =
      std::max<uint32_t>(3, o.batch ? o.batch : (uint32_t)std::min<uint64_t>(fits, ki.resident_walks));
  gpu::Context* ctx = gpu::create(o.fast, capacity, o.block, err, sizeof err);
  if (!ctx) {
    fprintf(stderr, "walk-cuda: %s\n", err);
    return 2;
  }
  uint32_t chunk = std::min<uint32_t>(p.remix, 4096);  // steps a launch

  // ── 1. Known answers ──
  printf("\nKnown answers (midstate's golden vectors) on the GPU: ");
  fflush(stdout);
  {
    const walk::Known* g = walk::golden(p);
    uint8_t hashes[3 * 32], got[3 * 32], want[32];
    uint64_t nonces[3];
    for (int i = 0; i < 3; ++i) {
      walk::from_hex(g[i].hash, hashes + 32 * i, 32);
      nonces[i] = g[i].nonce;
    }
    if (!gpu::set_jobs(ctx, hashes, 3, nonces, 3)) gpu_failed(ctx);
    run_walks(ctx, p, 3, chunk);
    if (!gpu::results(ctx, got, 3)) gpu_failed(ctx);
    bool ok[3];
    int right = 0;
    for (int i = 0; i < 3; ++i) {
      walk::from_hex(g[i].want, want, 32);
      ok[i] = memcmp(want, got + 32 * i, 32) == 0;
      right += ok[i];
    }
    printf("%d of 3 correct\n", right);
    if (right != 3) {
      for (int i = 0; i < 3; ++i) {
        if (ok[i]) continue;
        char hex[65];
        walk::to_hex(got + 32 * i, 32, hex);
        printf("  hash %s, nonce %" PRIu64 "\n    want %s\n    got  %s\n", g[i].hash, nonces[i], g[i].want, hex);
      }
      printf("This GPU computes the walk wrongly, so its speed means nothing. Please report it with this output.\n");
      return 3;
    }
  }

  // ── 2. How many walks in flight ──
  uint32_t walks = o.batch;
  if (o.sweep) {
    printf("\nWalks in flight   attempts/s   a walk takes   memory traffic (implied)\n");
    std::vector<uint32_t> sizes{std::min<uint32_t>(32, capacity)};
    for (uint64_t w = (uint64_t)info.sms * 32; w < capacity; w *= 2)
      if (w > sizes.back()) sizes.push_back((uint32_t)w);
    if (sizes.back() != capacity) sizes.push_back(capacity);
    std::vector<Point> points;
    for (const uint32_t w : sizes) {
      const Point pt = measure(ctx, p, w, chunk, o);
      printf("%15s   %10s   %12s   %5.1f GB/s\n", group(w).c_str(), rate_text(pt.rate).c_str(),
             seconds_text(pt.walk_seconds).c_str(), implied_gbs(p, pt.rate));
      // Stop once doubling the walks gains less than 5%.
      const bool level = points.size() >= 2 && pt.rate < 1.05 * points.back().rate;
      points.push_back(pt);
      if (o.sweep == 1 && level) break;
    }
    double top = 0;
    for (const Point& pt : points) top = std::max(top, pt.rate);
    if (!walks)
      for (const Point& pt : points)
        if (pt.rate >= 0.97 * top) {
          walks = pt.walks;
          break;
        }
    if (!o.batch) printf("Full speed from %s walks in flight\n", group(walks).c_str());
  }
  if (!walks) walks = capacity;
  if (chunk < std::min<uint32_t>(p.remix, 4096))
    printf("(launches cut to %u steps each to stay under half a second)\n", chunk);

  if (o.seconds <= 0) {
    gpu::destroy(ctx);
    return 0;
  }

  // ── 3. Sustained run ──
  printf("\nSustained: %s walks in flight for %.0f s, %s\n", group(walks).c_str(), o.seconds,
         power.available() ? "reading power through NVML" : "without power readings (NVML not found)");
  std::vector<uint64_t> nonces(walks);
  std::vector<uint8_t> out((size_t)walks * 32), first_out, last_out;
  uint64_t next = o.start, first_nonce = o.start, last_nonce = o.start;
  uint32_t batches = 0;
  double slowest = 0;
  power.start(100);
  const Clock::time_point t0 = Clock::now();
  Clock::time_point reported = t0;
  do {
    const Clock::time_point tb = Clock::now();
    for (uint32_t w = 0; w < walks; ++w) nonces[w] = next + w;
    if (!gpu::set_jobs(ctx, o.hash, 1, nonces.data(), walks)) gpu_failed(ctx);
    run_walks(ctx, p, walks, chunk);
    if (!gpu::results(ctx, out.data(), walks)) gpu_failed(ctx);
    slowest = std::max(slowest, since(tb));
    if (batches == 0) {
      first_out = out;
      first_nonce = next;
    }
    last_out = out;
    last_nonce = next;
    next += walks;
    ++batches;
    if (since(reported) >= 5 && since(t0) < o.seconds) {
      reported = Clock::now();
      printf("  %5.0f s   %s attempts/s%s\n", since(t0), rate_text((double)batches * walks / since(t0)).c_str(),
             power.available() ? fmt("   %.0f W", power.watts_now()).c_str() : "");
    }
  } while (since(t0) < o.seconds);
  const double secs = since(t0);
  const PowerSummary ps = power.stop();
  const uint64_t attempts = (uint64_t)batches * walks;
  const double rate = attempts / secs;

  // ── 4. Check the GPU's work on the CPU ──
  Check first_check, last_check;
  if (o.verify) {
    const uint32_t each = std::min(o.verify, walks);
    printf("\nChecking %s walks of the first batch%s on the CPU (%u threads)...\n", group(each).c_str(),
           batches > 1 ? " and as many of the last" : "", std::max(1u, std::thread::hardware_concurrency()));
    first_check = verify(p, keys, o.hash, first_nonce, first_out, walks, each);
    if (batches > 1) last_check = verify(p, keys, o.hash, last_nonce, last_out, walks, each);
  }
  const uint32_t wrong = first_check.wrong + last_check.wrong;

  // ── Report ──
  printf("\nResult      %s attempts/s, sustained over %.1f s (%s batches of %s walks)\n", rate_text(rate).c_str(), secs,
         group(batches).c_str(), group(walks).c_str());
  printf("            a walk takes %s in flight; the GPU finishes one every %s\n", seconds_text(slowest).c_str(),
         seconds_text(1 / rate).c_str());
  printf("Memory      implied %.2f billion random 32-byte transfers a second, %.0f GB/s: %.0f%% of the %.0f GB/s peak\n",
         rate * p.steps * 8 / 1e9, implied_gbs(p, rate), 100 * implied_gbs(p, rate) / peak_gbs, peak_gbs);
  printf("            (if each step reads 4 random sectors and writes 4 back, as the walk is designed to make it)\n");
  if (ps.available && ps.avg_watts > 0) {
    printf("Power       %.0f W average, %.0f W peak%s, from %s → %.1f attempts per joule\n", ps.avg_watts, ps.max_watts,
           ps.limit_watts ? fmt(", %u W limit", ps.limit_watts).c_str() : "",
           ps.energy_counter ? "the board's energy counter" : fmt("%d samples", ps.samples).c_str(),
           rate / ps.avg_watts);
    const std::string held = throttle_words(ps.throttle);
    printf("            SM %.0f MHz · memory %.0f MHz · up to %u °C · busy %.0f%%%s\n", ps.avg_sm_mhz, ps.avg_mem_mhz,
           ps.max_temp_c, ps.avg_util, held.empty() ? "" : (" · held back by " + held).c_str());
  } else {
    printf("Power       not measured (NVML not found)\n");
  }
  if (!o.verify) {
    printf("Checks      known answers 3 of 3 · sample check skipped (--no-verify)\n");
  } else if (wrong == 0) {
    printf("Checks      known answers 3 of 3 · %s of the first batch%s match the CPU\n",
           first_check.checked == walks ? fmt("all %s walks", group(walks).c_str()).c_str()
                                        : fmt("%s walks", group(first_check.checked).c_str()).c_str(),
           batches > 1 ? fmt(" and %s of the last", group(last_check.checked).c_str()).c_str() : "");
  } else {
    const Check& bad = first_check.wrong ? first_check : last_check;
    const uint64_t at = (first_check.wrong ? first_nonce : last_nonce) + (uint64_t)bad.first_wrong;
    printf("Checks      WRONG: %s of %s walks differ from the CPU (the first at nonce %" PRIu64 ")\n",
           group(wrong).c_str(), group(first_check.checked + last_check.checked).c_str(), at);
    printf("\nThe GPU computed some walks wrongly (overclocked or overheating?), so this rate doesn't count.\n");
    gpu::destroy(ctx);
    return 4;
  }

  printf("\nPost this:\n");
  printf("walk-cuda %s | %s (cc %d.%d) | %s attempts/s | %s | %s walks | %.0f s | %s | %s\n", VERSION, info.name,
         info.major, info.minor, rate_text(rate).c_str(),
         ps.available && ps.avg_watts > 0 ? fmt("%.0f W, %.1f attempts/J", ps.avg_watts, rate / ps.avg_watts).c_str()
                                          : "power n/a",
         group(walks).c_str(), secs, driver.c_str(),
         o.fast ? "FAST-MINING TEST PARAMETERS" : o.verify ? "checked" : "known answers only");
  gpu::destroy(ctx);
  return 0;
}
