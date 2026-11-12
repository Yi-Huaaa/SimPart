#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/partition.hpp"

namespace simpart {

struct SimPartOptions {
  static constexpr int64_t kInfinity = std::numeric_limits<int64_t>::max();
  static constexpr int32_t kMaxPartitions = 1024;

  int32_t num_partitions = 16;  // k
  int64_t gamma = 1024;         // replication threshold: 0 = pure DSP, kInfinity = pure RAP
};

struct SimPartResult {
  Partition partition;
  std::vector<int32_t> group;  // group id of every node (majority-based grouping)
  std::vector<int32_t> level;  // level of every node
  double grouping_ms = 0;      // Algorithm 1 on the GPU, including graph upload/download
  double hybrid_ms = 0;        // Algorithm 2 (DSP-RAP hybrid partition) on the CPU

  double total_ms() const { return grouping_ms + hybrid_ms; }
};

// Runs the SimPart partitioner: GPU-parallel majority-based grouping and
// levelization (Algorithm 1) followed by DSP-RAP hybrid partitioning
// (Algorithm 2).
SimPartResult simpart_partition(const Circuit& circuit, const SimPartOptions& options);

// Algorithm 2 alone: builds the hybrid partition from per-node levels and
// group ids. Exposed for testing.
Partition build_hybrid_partition(const Circuit& circuit, const std::vector<int32_t>& level,
                                 const std::vector<int32_t>& group, int32_t num_groups,
                                 int64_t gamma);

}  // namespace simpart
