#include "simpart/levelize.hpp"

#include <algorithm>
#include <stdexcept>

namespace simpart {

Levels bucket_by_level(std::vector<int32_t> level) {
  Levels out;
  out.num_levels = level.empty() ? 0 : *std::max_element(level.begin(), level.end()) + 1;
  out.offsets.assign(static_cast<size_t>(out.num_levels) + 1, 0);
  for (int32_t l : level) ++out.offsets[l + 1];
  for (int32_t l = 0; l < out.num_levels; ++l) out.offsets[l + 1] += out.offsets[l];
  out.nodes.resize(level.size());
  std::vector<int32_t> cursor(out.offsets.begin(), out.offsets.end() - 1);
  for (int32_t v = 0; v < static_cast<int32_t>(level.size()); ++v) {
    out.nodes[cursor[level[v]]++] = v;
  }
  out.level = std::move(level);
  return out;
}

Levels levelize(const Circuit& c) {
  const int32_t n = c.num_nodes();
  std::vector<int32_t> level(n, 0), pending(n), queue;
  queue.reserve(n);
  for (int32_t v = 0; v < n; ++v) {
    pending[v] = c.fanin_count(v);
    if (pending[v] == 0) queue.push_back(v);
  }
  for (size_t head = 0; head < queue.size(); ++head) {
    const int32_t v = queue[head];
    for (int32_t e = c.fanout_offsets[v]; e < c.fanout_offsets[v + 1]; ++e) {
      const int32_t u = c.fanouts[e];
      level[u] = std::max(level[u], level[v] + 1);
      if (--pending[u] == 0) queue.push_back(u);
    }
  }
  if (static_cast<int32_t>(queue.size()) != n) {
    throw std::runtime_error("circuit is not acyclic");
  }
  return bucket_by_level(std::move(level));
}

}  // namespace simpart
