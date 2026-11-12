#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

namespace simpart {

// Wall-clock stopwatch reporting milliseconds.
class Timer {
 public:
  Timer() : start_(Clock::now()) {}
  void restart() { start_ = Clock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

struct Summary {
  double mean = 0, stddev = 0, min = 0, max = 0;
  size_t count = 0;
};

inline Summary summarize(const std::vector<double>& samples) {
  Summary s;
  s.count = samples.size();
  if (samples.empty()) return s;
  s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  double var = 0;
  for (double x : samples) var += (x - s.mean) * (x - s.mean);
  s.stddev = samples.size() > 1 ? std::sqrt(var / (samples.size() - 1)) : 0.0;
  auto [lo, hi] = std::minmax_element(samples.begin(), samples.end());
  s.min = *lo;
  s.max = *hi;
  return s;
}

}  // namespace simpart
