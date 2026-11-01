// nvml_power.h: reads a GPU's power, clocks and temperature through NVML while
// the benchmark runs. NVML comes with the NVIDIA driver and is loaded at run
// time, so the bench builds without it and simply reports no power if it is
// missing.
#pragma once
#include <stdint.h>

#include <string>

struct PowerSummary {
  bool available = false;
  double seconds = 0;           // how long it sampled
  int samples = 0;
  double avg_watts = 0, max_watts = 0, joules = 0;
  bool energy_counter = false;  // joules from the board's own energy counter (Volta and newer), else from samples
  double avg_sm_mhz = 0, avg_mem_mhz = 0, avg_util = 0;
  unsigned max_temp_c = 0;
  unsigned limit_watts = 0;     // the board's enforced power limit
  uint64_t throttle = 0;        // every clock-throttle reason seen (NVML's bits)
};

class PowerSampler {
 public:
  // `pci_bus_id` as CUDA reports it, e.g. "0000:02:00.0".
  explicit PowerSampler(const char* pci_bus_id);
  ~PowerSampler();
  PowerSampler(const PowerSampler&) = delete;
  PowerSampler& operator=(const PowerSampler&) = delete;

  bool available() const;
  std::string driver() const;   // e.g. "580.173.02", or "" without NVML
  double watts_now() const;     // the latest sample, while sampling
  void start(int interval_ms = 100);
  PowerSummary stop();

 private:
  struct Impl;
  Impl* impl_;
};

// The throttle reasons worth reporting, in words ("" for none).
std::string throttle_words(uint64_t reasons);
