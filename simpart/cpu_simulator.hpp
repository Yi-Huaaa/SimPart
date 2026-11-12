#pragma once

#include <cstdint>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/partition.hpp"
#include "simpart/patterns.hpp"

namespace simpart {

struct SimulationResult {
  std::vector<uint64_t> po_values;    // [round * num_pos + po]
  std::vector<uint64_t> node_values;  // all nodes after the last round
};

// Sequential reference simulation in topological order.
SimulationResult simulate_reference(const Circuit& circuit, const Patterns& patterns);

// CPU-parallel simulation on top of a partition, following RepCut's setting:
// DSP levels are parallelized node by node, and RAP groups are distributed over
// the OpenMP threads (one group runs sequentially on one thread).
class CpuSimulator {
 public:
  CpuSimulator(const Circuit& circuit, const Partition& partition, const Patterns& patterns,
               int num_threads);

  void run();
  SimulationResult result() const { return {po_values_, values_}; }

 private:
  const Circuit& circuit_;
  const Partition& partition_;
  const Patterns& patterns_;
  int num_threads_;
  std::vector<uint64_t> values_;
  std::vector<uint64_t> po_values_;
};

}  // namespace simpart
