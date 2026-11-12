#pragma once

#include <cstdint>
#include <vector>

#include "simpart/circuit.hpp"

namespace simpart {

// Nodes grouped by topological level (longest path from a primary input).
struct Levels {
  int32_t num_levels = 0;
  std::vector<int32_t> level;    // per node
  std::vector<int32_t> offsets;  // size num_levels + 1
  std::vector<int32_t> nodes;    // nodes sorted by (level, index)

  int32_t size(int32_t l) const { return offsets[l + 1] - offsets[l]; }
};

// Computes levels on the CPU (Kahn's algorithm). Throws if the circuit has a cycle.
Levels levelize(const Circuit& circuit);

// Buckets nodes by a given per-node level (stable counting sort by index).
Levels bucket_by_level(std::vector<int32_t> level);

}  // namespace simpart
