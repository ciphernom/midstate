// nvml_power.cpp: see nvml_power.h.
#include "nvml_power.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

// The few NVML types used here, as nvml.h declares them.
typedef int nvmlReturn_t;  // 0 is NVML_SUCCESS
typedef struct nvmlDevice_st* nvmlDevice_t;
struct nvmlUtilization_t {
  unsigned int gpu, memory;
};
const int CLOCK_SM = 1, CLOCK_MEM = 2, TEMPERATURE_GPU = 0;

struct Nvml {
  void* lib = nullptr;
  bool up = false;
  nvmlReturn_t (*init)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  nvmlReturn_t (*driver_version)(char*, unsigned) = nullptr;
  nvmlReturn_t (*by_bus_id)(const char*, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*power)(nvmlDevice_t, unsigned*) = nullptr;
  nvmlReturn_t (*energy)(nvmlDevice_t, unsigned long long*) = nullptr;
  nvmlReturn_t (*clock)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*temperature)(nvmlDevice_t, int, unsigned*) = nullptr;
  nvmlReturn_t (*utilization)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
  nvmlReturn_t (*reasons)(nvmlDevice_t, unsigned long long*) = nullptr;
  nvmlReturn_t (*limit)(nvmlDevice_t, unsigned*) = nullptr;

  void* symbol(const char* name) const {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
  }

  template <class F>
  void bind(F& f, const char* name) {
    void* p = symbol(name);
    static_assert(sizeof f == sizeof p, "function and object pointers differ in size");
    std::memcpy(&f, &p, sizeof p);
  }

  bool load() {
#if defined(_WIN32)
    lib = (void*)LoadLibraryA("nvml.dll");
#else
    lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
#endif
    if (!lib) return false;
    bind(init, "nvmlInit_v2");
    bind(shutdown, "nvmlShutdown");
    bind(driver_version, "nvmlSystemGetDriverVersion");
    bind(by_bus_id, "nvmlDeviceGetHandleByPciBusId_v2");
    bind(power, "nvmlDeviceGetPowerUsage");
    bind(energy, "nvmlDeviceGetTotalEnergyConsumption");
    bind(clock, "nvmlDeviceGetClockInfo");
    bind(temperature, "nvmlDeviceGetTemperature");
    bind(utilization, "nvmlDeviceGetUtilizationRates");
    bind(reasons, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!reasons) bind(reasons, "nvmlDeviceGetCurrentClocksThrottleReasons");
    bind(limit, "nvmlDeviceGetEnforcedPowerLimit");
    up = init && shutdown && by_bus_id && power && init() == 0;
    return up;
  }
};

using Clock = std::chrono::steady_clock;

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

}  // namespace

struct PowerSampler::Impl {
  Nvml nvml;
  nvmlDevice_t dev = nullptr;
  bool ok = false;
  std::thread thread;
  std::atomic<bool> stopping{false};
  std::atomic<double> latest{0};
  // Written by the sampling thread, read after it is joined.
  PowerSummary sum;
  double watts_total = 0, sm_total = 0, mem_total = 0, util_total = 0;
  int clock_samples = 0, util_samples = 0;
  unsigned long long energy_start = 0;
  bool energy_ok = false;
  Clock::time_point t0;

  void loop(int interval_ms) {
    Clock::time_point next = Clock::now();
    while (!stopping.load()) {
      unsigned mw = 0, mhz = 0, mem = 0, temp = 0;
      if (nvml.power(dev, &mw) == 0) {
        const double w = mw / 1000.0;
        latest.store(w);
        watts_total += w;
        sum.max_watts = std::max(sum.max_watts, w);
        ++sum.samples;
      }
      if (nvml.clock && nvml.clock(dev, CLOCK_SM, &mhz) == 0 && nvml.clock(dev, CLOCK_MEM, &mem) == 0) {
        sm_total += mhz;
        mem_total += mem;
        ++clock_samples;
      }
      if (nvml.temperature && nvml.temperature(dev, TEMPERATURE_GPU, &temp) == 0)
        sum.max_temp_c = std::max(sum.max_temp_c, temp);
      nvmlUtilization_t u{0, 0};
      if (nvml.utilization && nvml.utilization(dev, &u) == 0) {
        util_total += u.gpu;
        ++util_samples;
      }
      unsigned long long r = 0;
      if (nvml.reasons && nvml.reasons(dev, &r) == 0) sum.throttle |= r;
      next += std::chrono::milliseconds(interval_ms);
      const Clock::time_point now = Clock::now();
      if (next < now) next = now;
      std::this_thread::sleep_until(next);
    }
  }
};

PowerSampler::PowerSampler(const char* pci_bus_id) : impl_(new Impl) {
  if (impl_->nvml.load() && impl_->nvml.by_bus_id(pci_bus_id, &impl_->dev) == 0) impl_->ok = true;
}

PowerSampler::~PowerSampler() {
  if (impl_->thread.joinable()) {
    impl_->stopping.store(true);
    impl_->thread.join();
  }
  if (impl_->nvml.up) impl_->nvml.shutdown();
  delete impl_;
}

bool PowerSampler::available() const { return impl_->ok; }

std::string PowerSampler::driver() const {
  char buf[96] = {0};
  if (!impl_->nvml.up || !impl_->nvml.driver_version || impl_->nvml.driver_version(buf, sizeof buf) != 0) return "";
  return buf;
}

double PowerSampler::watts_now() const { return impl_->latest.load(); }

void PowerSampler::start(int interval_ms) {
  Impl& m = *impl_;
  if (!m.ok || m.thread.joinable()) return;
  m.sum = PowerSummary();
  m.watts_total = m.sm_total = m.mem_total = m.util_total = 0;
  m.clock_samples = m.util_samples = 0;
  m.energy_ok = m.nvml.energy && m.nvml.energy(m.dev, &m.energy_start) == 0;
  m.stopping.store(false);
  m.t0 = Clock::now();
  m.thread = std::thread([&m, interval_ms] { m.loop(interval_ms); });
}

PowerSummary PowerSampler::stop() {
  Impl& m = *impl_;
  if (!m.thread.joinable()) return PowerSummary();
  m.stopping.store(true);
  m.thread.join();
  PowerSummary s = m.sum;
  s.seconds = seconds(Clock::now() - m.t0);
  // The board's energy counter (Volta and newer) beats averaging samples, as
  // long as it roughly agrees with them.
  const double sampled = s.samples > 0 ? m.watts_total / s.samples : 0;
  unsigned long long energy_end = 0;
  if (m.energy_ok && m.nvml.energy(m.dev, &energy_end) == 0 && energy_end > m.energy_start && s.seconds > 0) {
    const double counted = (energy_end - m.energy_start) / 1000.0 / s.seconds;
    if (sampled == 0 || (counted > 0.5 * sampled && counted < 2 * sampled)) {
      s.energy_counter = true;
      s.avg_watts = counted;
    }
  }
  if (!s.energy_counter) s.avg_watts = sampled;
  s.joules = s.avg_watts * s.seconds;
  s.available = s.samples > 0;
  if (m.clock_samples > 0) {
    s.avg_sm_mhz = m.sm_total / m.clock_samples;
    s.avg_mem_mhz = m.mem_total / m.clock_samples;
  }
  if (m.util_samples > 0) s.avg_util = m.util_total / m.util_samples;
  unsigned mw = 0;
  if (m.nvml.limit && m.nvml.limit(m.dev, &mw) == 0) s.limit_watts = (mw + 500) / 1000;
  return s;
}

std::string throttle_words(uint64_t r) {
  // NVML's clock-throttle (clock-event) reason bits that mean the GPU was held
  // back; idle, application-clock and display-clock reasons are left out.
  struct Reason {
    uint64_t bits;
    const char* words;
  } const reasons[] = {
      {0x4, "power limit"},
      {0x20 | 0x40, "temperature"},
      {0x8, "hardware slowdown"},
      {0x80, "power brake"},
  };
  std::string out;
  for (const Reason& reason : reasons)
    if (r & reason.bits) out += (out.empty() ? "" : ", ") + std::string(reason.words);
  return out;
}
