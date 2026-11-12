#pragma once

#include <cstdint>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/partition.hpp"

namespace simpart {

struct RepCutOptions {
  int32_t num_partitions = 16;  // k
  int32_t num_threads = 16;     // Mt-KaHyPar threads
  double imbalance = 0.015;
  uint64_t seed = 42;
};

struct RepCutResult {
  Partition partition;          // pure RAP: one group per hypergraph block
  std::vector<int32_t> level;   // level of every node
  int64_t num_vertices = 0;     // hypergraph vertices (one per sink)
  int64_t num_hyperedges = 0;   // hyperedges (clusters of shared nodes)
  int64_t num_pins = 0;
  double levelize_ms = 0;       // CPU levelization
  double build_ms = 0;          // fanin cones, clusters and hypergraph construction
  double partition_ms = 0;      // Mt-KaHyPar
  double assemble_ms = 0;       // expanding the hypergraph partition into cones

  double hypergraph_ms() const { return levelize_ms + build_ms; }
};

// The RepCut baseline [Wang et al., ASPLOS'23]: builds the proxy hypergraph in
// which every sink is a vertex and every set of nodes shared by the same sinks
// is a hyperedge, partitions it with Mt-KaHyPar (km1), and replicates every
// node into all blocks whose sinks it reaches.
RepCutResult repcut_partition(const Circuit& circuit, const RepCutOptions& options);

}  // namespace simpart
