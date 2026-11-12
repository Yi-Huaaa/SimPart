#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/partition.hpp"
#include "simpart/patterns.hpp"

namespace simpart {

enum class Executor {
  Stream,  // one kernel launch per DSP level plus one RAP launch, per round
  Graph,   // all rounds in a single conditional (while-loop) CUDA graph (Sec. 3.3)
};

const char* to_string(Executor executor);

// Simulates all pattern rounds of a circuit on the GPU on top of a partition.
//
// The rounds are simulated in passes of `batch` rounds (1, 2, 4, 8 or 16; the
// paper uses 1). Per pass, the DSP levels are evaluated one kernel launch per
// level (Algorithm 3) and the RAP region by a single launch with one thread
// block per group (Algorithm 4); every thread evaluates its node for all rounds
// of the pass. The value of every PO is recorded for every round.
class GpuSimulator {
 public:
  static constexpr int kBlockSize = 1024;
  static constexpr int kMaxBatch = 16;

  GpuSimulator(const Circuit& circuit, const Partition& partition, const Patterns& patterns,
               int32_t batch = 1);
  ~GpuSimulator();
  GpuSimulator(const GpuSimulator&) = delete;
  GpuSimulator& operator=(const GpuSimulator&) = delete;

  // Builds and instantiates the conditional CUDA graph (done once, lazily by
  // run(Executor::Graph)); returns the time it took in milliseconds.
  double prepare_graph();

  // Simulates all rounds and blocks until the GPU is done.
  void run(Executor executor);

  // PO values of every round: po_values()[r * num_pos + i] is PO i in round r.
  std::vector<uint64_t> po_values() const;
  // Values of all nodes after the last round.
  std::vector<uint64_t> node_values() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace simpart
