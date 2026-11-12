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
// The rounds are simulated in passes of `batch` rounds (1, 2, 4, 8 or 16).
// Per pass, the DSP levels are evaluated one kernel launch per level
// (Algorithm 3) and the RAP region by a single launch with one thread block per
// group (Algorithm 4); every thread evaluates its node for all rounds of the
// pass. The value of every PO is recorded for every round.
class GpuSimulator {
 public:
  // Threads per block of the RAP kernel; it matches the default replication
  // threshold gamma = 1024 of SimPart.
  static constexpr int kBlockSize = 1024;

  // Threads per block of the DSP kernels. Small blocks spread a level over more
  // SMs: a level of 2,000 nodes runs on 8 SMs instead of 2 with 1024-thread
  // blocks. On an RTX A4000 this made the circuits of the paper 1.2-1.4x faster
  // (up to 1.5x with 128 threads, which is slightly slower on the largest ones).
  static constexpr int kDspBlockSize = 256;

  static constexpr int kMaxBatch = 16;

  // Rounds per pass by default. The rounds are independent, so a pass of 4
  // rounds pays the per-level synchronization once for 4 rounds, and every
  // fanin read fetches 4 consecutive 64-bit words, i.e. one full 32-byte memory
  // sector instead of a quarter of one. On an RTX A4000 a batch of 4 was the
  // fastest or within 7% of the fastest (batch 8) on all circuits of the paper,
  // 2-3x faster than one round per pass.
  static constexpr int32_t kDefaultBatch = 4;

  GpuSimulator(const Circuit& circuit, const Partition& partition, const Patterns& patterns,
               int32_t batch = kDefaultBatch);
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
