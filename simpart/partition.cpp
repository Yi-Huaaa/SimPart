#include "simpart/partition.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace simpart {

ReplicationStats replication_stats(const Circuit& c, const Partition& p) {
  ReplicationStats s;
  auto count = [&](int32_t v) {
    ++s.node_instances;
    s.edge_instances += c.fanin_count(v);
  };
  for (int32_t v : p.dsp_nodes) count(v);
  for (int32_t v : p.rap_nodes) count(v);
  s.rap_nodes = static_cast<int64_t>(p.rap_nodes.size());
  const int32_t r = p.rap_levels();
  for (int32_t g = 0; g < p.num_groups; ++g) {
    const int64_t size = p.rap_offsets[int64_t(g + 1) * r] - p.rap_offsets[int64_t(g) * r];
    s.max_group_nodes = std::max(s.max_group_nodes, size);
  }
  s.node_ratio = c.num_nodes() ? double(s.node_instances) / c.num_nodes() : 0.0;
  s.edge_ratio = c.num_edges() ? double(s.edge_instances) / double(c.num_edges()) : 0.0;
  return s;
}

namespace {
[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error("invalid partition: " + msg);
}

void check_sorted(const std::vector<int32_t>& nodes, int64_t begin, int64_t end,
                  const char* what) {
  if (!std::is_sorted(nodes.begin() + begin, nodes.begin() + end)) {
    fail(std::string(what) + " list is not sorted");
  }
  if (std::adjacent_find(nodes.begin() + begin, nodes.begin() + end) != nodes.begin() + end) {
    fail(std::string(what) + " list has duplicates");
  }
}
}  // namespace

void validate_partition(const Circuit& c, const std::vector<int32_t>& level, const Partition& p) {
  const int32_t n = c.num_nodes();
  const int32_t r = p.rap_levels();
  if (p.dsp_levels < 0 || p.dsp_levels > p.num_levels) fail("bad dsp_levels");
  if (static_cast<int32_t>(p.dsp_offsets.size()) != p.dsp_levels + 1) fail("bad dsp_offsets size");
  if (static_cast<int64_t>(p.rap_offsets.size()) != int64_t(p.num_groups) * r + 1) {
    fail("bad rap_offsets size");
  }
  if (p.dsp_offsets.back() != static_cast<int32_t>(p.dsp_nodes.size()) ||
      p.rap_offsets.back() != static_cast<int32_t>(p.rap_nodes.size())) {
    fail("offsets do not match node arrays");
  }

  std::vector<uint8_t> covered(n, 0);
  for (int32_t l = 0; l < p.dsp_levels; ++l) {
    check_sorted(p.dsp_nodes, p.dsp_offsets[l], p.dsp_offsets[l + 1], "DSP");
    for (int32_t i = p.dsp_offsets[l]; i < p.dsp_offsets[l + 1]; ++i) {
      const int32_t v = p.dsp_nodes[i];
      if (level[v] != l) fail("DSP node " + std::to_string(v) + " placed at a wrong level");
      if (covered[v]) fail("DSP node " + std::to_string(v) + " listed twice");
      covered[v] = 1;
    }
  }
  for (int32_t v = 0; v < n; ++v) {
    if (level[v] < p.dsp_levels && !covered[v]) {
      fail("node " + std::to_string(v) + " missing from the DSP region");
    }
  }

  // RAP region: per group, every node at its level and fanins available.
  std::vector<int32_t> in_group(n, -1);
  for (int32_t g = 0; g < p.num_groups; ++g) {
    for (int32_t l = p.dsp_levels; l < p.num_levels; ++l) {
      const int64_t idx = p.rap_index(g, l);
      check_sorted(p.rap_nodes, p.rap_offsets[idx], p.rap_offsets[idx + 1], "RAP");
      for (int32_t i = p.rap_offsets[idx]; i < p.rap_offsets[idx + 1]; ++i) {
        const int32_t v = p.rap_nodes[i];
        if (level[v] != l) fail("RAP node " + std::to_string(v) + " placed at a wrong level");
        in_group[v] = g;
        covered[v] = 1;
      }
    }
    for (int32_t l = p.dsp_levels; l < p.num_levels; ++l) {
      const int64_t idx = p.rap_index(g, l);
      for (int32_t i = p.rap_offsets[idx]; i < p.rap_offsets[idx + 1]; ++i) {
        const int32_t v = p.rap_nodes[i];
        for (int32_t e = c.fanin_offsets[v]; e < c.fanin_offsets[v + 1]; ++e) {
          const int32_t u = c.fanins[e];
          if (level[u] >= p.dsp_levels && in_group[u] != g) {
            fail("fanin " + std::to_string(u) + " of RAP node " + std::to_string(v) +
                 " is not available in group " + std::to_string(g));
          }
        }
      }
    }
  }
  for (int32_t v = 0; v < n; ++v) {
    if (!covered[v]) fail("node " + std::to_string(v) + " is not simulated");
  }
}

}  // namespace simpart
