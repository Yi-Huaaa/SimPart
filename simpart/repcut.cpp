#include "simpart/repcut.hpp"

#include <libmtkahypar.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "simpart/levelize.hpp"
#include "simpart/timer.hpp"

namespace simpart {

namespace {

// For every node, the sorted ids of the sinks (cones) it reaches, stored in a
// shared pool. Sinks are numbered in the order (level descending, index
// ascending), which is the order RepCut discovers the cones in.
struct ConeMembership {
  std::vector<int64_t> begin;  // per node
  std::vector<int32_t> size;   // per node
  std::vector<int32_t> pool;
  int32_t num_cones = 0;

  const int32_t* data(int32_t v) const { return pool.data() + begin[v]; }
};

// Nodes ordered by level descending, then index ascending.
std::vector<int32_t> reverse_level_order(const Levels& levels) {
  std::vector<int32_t> order;
  order.reserve(levels.nodes.size());
  for (int32_t l = levels.num_levels - 1; l >= 0; --l) {
    order.insert(order.end(), levels.nodes.begin() + levels.offsets[l],
                 levels.nodes.begin() + levels.offsets[l + 1]);
  }
  return order;
}

ConeMembership compute_cones(const Circuit& c, const std::vector<int32_t>& order) {
  ConeMembership m;
  const int32_t n = c.num_nodes();
  m.begin.assign(n, 0);
  m.size.assign(n, 0);
  m.pool.reserve(n);
  std::vector<int32_t> merged;
  for (int32_t v : order) {
    const int64_t begin = static_cast<int64_t>(m.pool.size());
    if (c.fanout_count(v) == 0) {
      m.pool.push_back(m.num_cones++);
    } else if (c.fanout_count(v) == 1) {
      const int32_t u = c.fanouts[c.fanout_offsets[v]];
      // Copy by index: push_back may reallocate the pool.
      for (int32_t i = 0; i < m.size[u]; ++i) {
        const int32_t cone = m.pool[m.begin[u] + i];
        m.pool.push_back(cone);
      }
    } else {
      merged.clear();
      for (int32_t e = c.fanout_offsets[v]; e < c.fanout_offsets[v + 1]; ++e) {
        const int32_t u = c.fanouts[e];
        merged.insert(merged.end(), m.data(u), m.data(u) + m.size[u]);
      }
      std::sort(merged.begin(), merged.end());
      merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
      m.pool.insert(m.pool.end(), merged.begin(), merged.end());
    }
    m.begin[v] = begin;
    m.size[v] = static_cast<int32_t>(static_cast<int64_t>(m.pool.size()) - begin);
  }
  return m;
}

struct Hypergraph {
  std::vector<int32_t> cluster_size;  // clusters = maximal sets of nodes with equal cone sets
  std::vector<int32_t> cluster_rep;  // a representative node of each cluster
  std::vector<mt_kahypar_hypernode_weight_t> vertex_weights;  // vertex c <-> cone c
  std::vector<size_t> hyperedge_offsets;
  std::vector<mt_kahypar_hyperedge_id_t> hyperedge_pins;
  std::vector<mt_kahypar_hyperedge_weight_t> hyperedge_weights;
};

int checked_weight(double w) {
  if (!(w <= double(INT_MAX))) throw std::runtime_error("hypergraph weight overflows int");
  return static_cast<int>(w);
}

// Builds RepCut's proxy hypergraph. Clusters and hyperedges are numbered in the
// order of the original implementation: by smallest cone id, then by the first
// member in (level descending, index ascending) order.
Hypergraph build_hypergraph(const Circuit& c, const std::vector<int32_t>& order,
                            const ConeMembership& m) {
  const int32_t n = c.num_nodes();
  struct SetHash {
    const ConeMembership* m;
    size_t operator()(int32_t v) const {
      return std::hash<std::string_view>()(std::string_view(
          reinterpret_cast<const char*>(m->data(v)), sizeof(int32_t) * m->size[v]));
    }
  };
  struct SetEqual {
    const ConeMembership* m;
    bool operator()(int32_t a, int32_t b) const {
      return m->size[a] == m->size[b] && std::equal(m->data(a), m->data(a) + m->size[a], m->data(b));
    }
  };
  std::unordered_map<int32_t, int32_t, SetHash, SetEqual> clusters(n / 4 + 16, SetHash{&m},
                                                                  SetEqual{&m});
  Hypergraph h;
  std::vector<int32_t> discovered_rep, discovered_size;
  for (int32_t v : order) {
    auto [it, inserted] = clusters.try_emplace(v, static_cast<int32_t>(discovered_rep.size()));
    if (inserted) {
      discovered_rep.push_back(v);
      discovered_size.push_back(0);
    }
    ++discovered_size[it->second];
  }

  // Renumber clusters by (smallest cone id, discovery order).
  const auto num_clusters = static_cast<int32_t>(discovered_rep.size());
  std::vector<int32_t> rank(num_clusters);
  std::iota(rank.begin(), rank.end(), 0);
  std::stable_sort(rank.begin(), rank.end(), [&](int32_t a, int32_t b) {
    return m.data(discovered_rep[a])[0] < m.data(discovered_rep[b])[0];
  });
  std::vector<int32_t> new_id(num_clusters);
  for (int32_t i = 0; i < num_clusters; ++i) new_id[rank[i]] = i;
  h.cluster_size.resize(num_clusters);
  h.cluster_rep.resize(num_clusters);
  for (int32_t d = 0; d < num_clusters; ++d) {
    h.cluster_size[new_id[d]] = discovered_size[d];
    h.cluster_rep[new_id[d]] = discovered_rep[d];
  }

  // Vertices: clusters reaching a single cone (each contains that cone's sink).
  // Hyperedges: clusters shared by several cones, connecting exactly those cones.
  std::vector<double> incident_weight(m.num_cones, 0.0);
  std::vector<int32_t> incident_count(m.num_cones, 0), exclusive_size(m.num_cones, 0);
  h.hyperedge_offsets.push_back(0);
  for (int32_t k = 0; k < num_clusters; ++k) {
    const int32_t rep = h.cluster_rep[k];
    const int32_t* cones = m.data(rep);
    const int32_t num = m.size[rep];
    if (num == 1) {
      exclusive_size[cones[0]] = h.cluster_size[k];
      continue;
    }
    for (int32_t i = 0; i < num; ++i) {
      h.hyperedge_pins.push_back(cones[i]);
      incident_weight[cones[i]] += h.cluster_size[k];
      ++incident_count[cones[i]];
    }
    h.hyperedge_offsets.push_back(h.hyperedge_pins.size());
    h.hyperedge_weights.push_back(checked_weight(100.0 * h.cluster_size[k]));
  }
  // Vertex weight: own nodes plus the average weight of the incident hyperedges.
  h.vertex_weights.resize(m.num_cones);
  for (int32_t v = 0; v < m.num_cones; ++v) {
    double w = exclusive_size[v];
    if (incident_count[v] > 0) w += incident_weight[v] / incident_count[v];
    h.vertex_weights[v] = checked_weight(std::round(w * 100.0));
  }
  return h;
}

std::vector<int32_t> run_mt_kahypar(const Hypergraph& h, const RepCutOptions& o) {
  const auto num_vertices = static_cast<mt_kahypar_hypernode_id_t>(h.vertex_weights.size());
  const auto num_hyperedges = static_cast<mt_kahypar_hyperedge_id_t>(h.hyperedge_weights.size());
  std::vector<int32_t> block(num_vertices);
  if (num_vertices <= static_cast<mt_kahypar_hypernode_id_t>(o.num_partitions)) {
    std::iota(block.begin(), block.end(), 0);  // every sink gets its own block
    return block;
  }
  if (num_hyperedges == 0) {
    // Disjoint cones: nothing to cut, balance the vertex weights greedily.
    std::vector<int32_t> order(num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return h.vertex_weights[a] > h.vertex_weights[b];
    });
    std::vector<int64_t> load(o.num_partitions, 0);
    for (int32_t v : order) {
      const auto p = std::min_element(load.begin(), load.end()) - load.begin();
      block[v] = static_cast<int32_t>(p);
      load[p] += h.vertex_weights[v];
    }
    return block;
  }

  static int initialized_threads = 0;
  if (initialized_threads != o.num_threads) {
    mt_kahypar_initialize_thread_pool(o.num_threads, true);
    initialized_threads = o.num_threads;
  }
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT);
  mt_kahypar_set_partitioning_parameters(context, o.num_partitions, o.imbalance, KM1);
  mt_kahypar_set_context_parameter(context, VERBOSE, "0");
  mt_kahypar_set_seed(o.seed);

  mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
      DEFAULT, num_vertices, num_hyperedges, h.hyperedge_offsets.data(), h.hyperedge_pins.data(),
      h.hyperedge_weights.data(), h.vertex_weights.data());
  mt_kahypar_partitioned_hypergraph_t partitioned = mt_kahypar_partition(hypergraph, context);
  std::vector<mt_kahypar_partition_id_t> result(num_vertices);
  mt_kahypar_get_partition(partitioned, result.data());
  mt_kahypar_free_partitioned_hypergraph(partitioned);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_context(context);

  std::copy(result.begin(), result.end(), block.begin());
  return block;
}

// Every node is replicated into each block that owns one of the cones it belongs to.
Partition assemble_partition(const Circuit& c, const Levels& levels, const ConeMembership& m,
                             const std::vector<int32_t>& block, int32_t k) {
  const int32_t n = c.num_nodes();
  Partition p;
  p.num_levels = levels.num_levels;
  p.dsp_levels = 0;
  p.num_groups = k;
  const int32_t num_levels = p.num_levels;
  p.rap_offsets.assign(static_cast<size_t>(k) * num_levels + 1, 0);

  std::vector<int32_t> stamp(k, -1);
  auto for_each_block = [&](int32_t v, auto&& fn) {
    for (int32_t i = 0; i < m.size[v]; ++i) {
      const int32_t b = block[m.data(v)[i]];
      if (stamp[b] != v) {
        stamp[b] = v;
        fn(b);
      }
    }
  };
  for (int32_t v = 0; v < n; ++v) {
    for_each_block(v, [&](int32_t b) { ++p.rap_offsets[p.rap_index(b, levels.level[v]) + 1]; });
  }
  for (size_t i = 1; i < p.rap_offsets.size(); ++i) p.rap_offsets[i] += p.rap_offsets[i - 1];
  p.rap_nodes.resize(p.rap_offsets.back());
  std::vector<int32_t> cursor(p.rap_offsets.begin(), p.rap_offsets.end() - 1);
  std::fill(stamp.begin(), stamp.end(), -1);
  for (int32_t v = 0; v < n; ++v) {
    for_each_block(v, [&](int32_t b) { p.rap_nodes[cursor[p.rap_index(b, levels.level[v])]++] = v; });
  }
  return p;
}

}  // namespace

RepCutResult repcut_partition(const Circuit& circuit, const RepCutOptions& options) {
  if (options.num_partitions < 1) throw std::invalid_argument("number of partitions must be positive");
  if (options.num_threads < 1) throw std::invalid_argument("number of threads must be positive");
  RepCutResult r;

  Timer timer;
  Levels levels = levelize(circuit);
  r.levelize_ms = timer.elapsed_ms();

  timer.restart();
  const std::vector<int32_t> order = reverse_level_order(levels);
  const ConeMembership cones = compute_cones(circuit, order);
  const Hypergraph hypergraph = build_hypergraph(circuit, order, cones);
  r.build_ms = timer.elapsed_ms();
  r.num_vertices = static_cast<int64_t>(hypergraph.vertex_weights.size());
  r.num_hyperedges = static_cast<int64_t>(hypergraph.hyperedge_weights.size());
  r.num_pins = static_cast<int64_t>(hypergraph.hyperedge_pins.size());

  timer.restart();
  const std::vector<int32_t> block = run_mt_kahypar(hypergraph, options);
  r.partition_ms = timer.elapsed_ms();

  timer.restart();
  r.partition = assemble_partition(circuit, levels, cones, block, options.num_partitions);
  r.assemble_ms = timer.elapsed_ms();
  r.level = std::move(levels.level);
  return r;
}

}  // namespace simpart
