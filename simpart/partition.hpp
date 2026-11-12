#pragma once

#include <cstdint>
#include <vector>

#include "simpart/circuit.hpp"

namespace simpart {

// A DSP-RAP hybrid partition of a levelized circuit (Sec. 3.2 of the paper).
//
// Levels [0, dsp_levels) form the DSP region: every node of these levels is
// listed exactly once and simulated level by level across the whole GPU.
// Levels [dsp_levels, num_levels) form the RAP region: it is split into
// num_groups independent fanin cones; a node shared by several cones is
// replicated in each of them. Inside every list nodes are sorted by index.
//
// Special cases: dsp_levels == num_levels is pure DSP (levelization) and
// dsp_levels == 0 is pure RAP (e.g. RepCut).
struct Partition {
  int32_t num_levels = 0;
  int32_t dsp_levels = 0;
  std::vector<int32_t> dsp_offsets{0};  // size dsp_levels + 1
  std::vector<int32_t> dsp_nodes;

  int32_t num_groups = 0;
  std::vector<int32_t> rap_offsets{0};  // size num_groups * rap_levels() + 1 (group-major)
  std::vector<int32_t> rap_nodes;

  int32_t rap_levels() const { return num_levels - dsp_levels; }
  int32_t dsp_level_size(int32_t l) const { return dsp_offsets[l + 1] - dsp_offsets[l]; }
  // Index into rap_offsets of RAP level `l` (absolute level) of group `g`.
  int64_t rap_index(int32_t g, int32_t l) const {
    return int64_t(g) * rap_levels() + (l - dsp_levels);
  }
};

struct ReplicationStats {
  int64_t node_instances = 0;  // nodes simulated, counting replicas
  int64_t edge_instances = 0;  // fanin edges evaluated, counting replicas
  double node_ratio = 0;       // R|V| = node_instances / |V|
  double edge_ratio = 0;       // R|E| = edge_instances / |E|
  int64_t rap_nodes = 0;       // node instances in the RAP region
  int64_t max_group_nodes = 0;
};

ReplicationStats replication_stats(const Circuit& circuit, const Partition& partition);

// Checks that `partition` is a valid simulation schedule for `circuit` given
// the node levels: every node is covered, nodes sit at their own level, and the
// fanins of every RAP node are either in the DSP region or replicated in the
// same group. Throws std::runtime_error describing the first violation.
void validate_partition(const Circuit& circuit, const std::vector<int32_t>& level,
                        const Partition& partition);

}  // namespace simpart
