#include "simpart/simpart.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#include "simpart/cuda_utils.hpp"
#include "simpart/timer.hpp"

namespace simpart {

namespace {

constexpr int kBlockSize = 1024;

int grid_size(int64_t n) { return static_cast<int>((n + kBlockSize - 1) / kBlockSize); }

// Streaming majority vote over group ids in [0, k). Ties go to the group that
// first reached the maximal count, i.e. the order of the votes matters.
template <int MaxK>
struct MajorityVote {
  int32_t count[MaxK];
  int32_t best = 0;
  int32_t best_count = 0;

  __device__ explicit MajorityVote(int32_t k) {
    for (int32_t i = 0; i < k; ++i) count[i] = 0;
  }
  __device__ void add(int32_t g) {
    const int32_t c = ++count[g];
    if (c > best_count) {
      best_count = c;
      best = g;
    }
  }
};

// Decrements the unresolved-fanin counter of every fanout of v and appends the
// fanouts whose fanins are now all resolved to the next BFS frontier.
__device__ __forceinline__ void release_fanouts(int32_t v, const int32_t* __restrict__ fanout_offsets,
                                                const int32_t* __restrict__ fanouts,
                                                int32_t* pending, int32_t* next,
                                                int32_t* next_size) {
  for (int32_t e = fanout_offsets[v]; e < fanout_offsets[v + 1]; ++e) {
    const int32_t u = fanouts[e];
    if (atomicSub(&pending[u], 1) == 1) next[atomicAdd(next_size, 1)] = u;
  }
}

__global__ void init_pending_kernel(int32_t num_nodes, const int32_t* __restrict__ fanin_offsets,
                                    int32_t* pending) {
  const int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < num_nodes) pending[v] = fanin_offsets[v + 1] - fanin_offsets[v];
}

// Algorithm 1, step 1: each PI takes the majority of (successor index mod k).
template <int MaxK>
__global__ void group_sources_kernel(int32_t num_pis, int32_t k,
                                     const int32_t* __restrict__ fanout_offsets,
                                     const int32_t* __restrict__ fanouts, int32_t* group,
                                     int32_t* level, int32_t* pending, int32_t* next,
                                     int32_t* next_size) {
  const int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= num_pis) return;
  MajorityVote<MaxK> vote(k);
  for (int32_t e = fanout_offsets[v]; e < fanout_offsets[v + 1]; ++e) vote.add(fanouts[e] % k);
  group[v] = vote.best;
  level[v] = 0;
  release_fanouts(v, fanout_offsets, fanouts, pending, next, next_size);
}

// Algorithm 1, step 2: every node of the current BFS level takes the majority
// group of its fanins, which were all resolved in earlier levels.
template <int MaxK>
__global__ void group_frontier_kernel(const int32_t* __restrict__ frontier, int32_t frontier_size,
                                      int32_t cur_level, int32_t k,
                                      const int32_t* __restrict__ fanin_offsets,
                                      const int32_t* __restrict__ fanins,
                                      const int32_t* __restrict__ fanout_offsets,
                                      const int32_t* __restrict__ fanouts, int32_t* group,
                                      int32_t* level, int32_t* pending, int32_t* next,
                                      int32_t* next_size) {
  const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= frontier_size) return;
  const int32_t v = frontier[t];
  MajorityVote<MaxK> vote(k);
  for (int32_t e = fanin_offsets[v]; e < fanin_offsets[v + 1]; ++e) vote.add(group[fanins[e]]);
  group[v] = vote.best;
  level[v] = cur_level;
  release_fanouts(v, fanout_offsets, fanouts, pending, next, next_size);
}

// GPU majority-based grouping and levelization (Algorithm 1).
template <int MaxK>
void majority_grouping(const Circuit& c, int32_t k, std::vector<int32_t>& group,
                       std::vector<int32_t>& level) {
  const int32_t n = c.num_nodes();
  group.clear();
  level.clear();
  if (n == 0) return;
  DeviceBuffer<int32_t> fanin_offsets(c.fanin_offsets), fanins(c.fanins);
  DeviceBuffer<int32_t> fanout_offsets(c.fanout_offsets), fanouts(c.fanouts);
  DeviceBuffer<int32_t> d_group(n), d_level(n), pending(n), queue_a(n), queue_b(n), next_size(1);

  init_pending_kernel<<<grid_size(n), kBlockSize>>>(n, fanin_offsets.data(), pending.data());
  SIMPART_CUDA_CHECK_LAUNCH();
  SIMPART_CUDA_CHECK(cudaMemset(next_size.data(), 0, sizeof(int32_t)));

  int32_t* frontier = queue_a.data();
  int32_t* next = queue_b.data();
  int64_t visited = c.num_pis;
  if (c.num_pis > 0) {
    group_sources_kernel<MaxK><<<grid_size(c.num_pis), kBlockSize>>>(
        c.num_pis, k, fanout_offsets.data(), fanouts.data(), d_group.data(), d_level.data(),
        pending.data(), next, next_size.data());
    SIMPART_CUDA_CHECK_LAUNCH();
  }
  for (int32_t cur_level = 1;; ++cur_level) {
    int32_t size = 0;
    SIMPART_CUDA_CHECK(cudaMemcpy(&size, next_size.data(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (size == 0) break;
    std::swap(frontier, next);
    SIMPART_CUDA_CHECK(cudaMemset(next_size.data(), 0, sizeof(int32_t)));
    group_frontier_kernel<MaxK><<<grid_size(size), kBlockSize>>>(
        frontier, size, cur_level, k, fanin_offsets.data(), fanins.data(), fanout_offsets.data(),
        fanouts.data(), d_group.data(), d_level.data(), pending.data(), next, next_size.data());
    SIMPART_CUDA_CHECK_LAUNCH();
    visited += size;
  }
  if (visited != n) throw std::runtime_error("circuit is not acyclic");
  group = d_group.download();
  level = d_level.download();
}

void majority_grouping_dispatch(const Circuit& c, int32_t k, std::vector<int32_t>& group,
                                std::vector<int32_t>& level) {
  if (k <= 16) return majority_grouping<16>(c, k, group, level);
  if (k <= 32) return majority_grouping<32>(c, k, group, level);
  if (k <= 64) return majority_grouping<64>(c, k, group, level);
  if (k <= 128) return majority_grouping<128>(c, k, group, level);
  if (k <= 256) return majority_grouping<256>(c, k, group, level);
  if (k <= 512) return majority_grouping<512>(c, k, group, level);
  return majority_grouping<1024>(c, k, group, level);
}

}  // namespace

Partition build_hybrid_partition(const Circuit& c, const std::vector<int32_t>& level,
                                 const std::vector<int32_t>& group, int32_t num_groups,
                                 int64_t gamma) {
  const int32_t n = c.num_nodes();
  Partition p;
  p.num_groups = num_groups;
  p.num_levels = n == 0 ? 0 : *std::max_element(level.begin(), level.end()) + 1;

  std::vector<int32_t> level_offsets(static_cast<size_t>(p.num_levels) + 1, 0);
  for (int32_t l : level) ++level_offsets[l + 1];
  for (int32_t l = 0; l < p.num_levels; ++l) level_offsets[l + 1] += level_offsets[l];

  // Replication level: the last level, scanning backward from the POs, that
  // holds more than gamma nodes. It and all levels before it form the DSP region.
  p.dsp_levels = p.num_levels;
  while (p.dsp_levels > 0 &&
         level_offsets[p.dsp_levels] - level_offsets[p.dsp_levels - 1] <= gamma) {
    --p.dsp_levels;
  }

  // DSP region: one disjoint partition per level (Algorithm 2, lines 7-12).
  p.dsp_offsets.assign(level_offsets.begin(), level_offsets.begin() + p.dsp_levels + 1);
  p.dsp_nodes.resize(p.dsp_offsets.back());
  {
    std::vector<int32_t> cursor(p.dsp_offsets.begin(), p.dsp_offsets.end() - 1);
    for (int32_t v = 0; v < n; ++v) {
      if (level[v] < p.dsp_levels) p.dsp_nodes[cursor[level[v]]++] = v;
    }
  }

  // RAP region: the fanin cone of every group, cut at the replication level
  // (Algorithm 2, lines 2-6). A node reached from several groups is replicated.
  const int32_t r = p.rap_levels();
  p.rap_offsets.assign(static_cast<size_t>(num_groups) * r + 1, 0);
  if (r == 0) return p;

  std::vector<std::vector<int32_t>> sinks(num_groups);
  for (int32_t v = 0; v < n; ++v) {
    if (c.fanout_count(v) == 0 && level[v] >= p.dsp_levels) sinks[group[v]].push_back(v);
  }
  std::vector<int32_t> mark(n, -1), stack;
  std::vector<int64_t> members;  // (level << 32) | node, sorted to get the per-level lists
  for (int32_t g = 0; g < num_groups; ++g) {
    members.clear();
    for (int32_t s : sinks[g]) {
      mark[s] = g;
      stack.push_back(s);
    }
    while (!stack.empty()) {
      const int32_t v = stack.back();
      stack.pop_back();
      members.push_back((int64_t(level[v]) << 32) | v);
      for (int32_t e = c.fanin_offsets[v]; e < c.fanin_offsets[v + 1]; ++e) {
        const int32_t u = c.fanins[e];
        if (level[u] >= p.dsp_levels && mark[u] != g) {
          mark[u] = g;
          stack.push_back(u);
        }
      }
    }
    std::sort(members.begin(), members.end());
    for (int64_t key : members) {
      const auto l = static_cast<int32_t>(key >> 32);
      ++p.rap_offsets[p.rap_index(g, l) + 1];
      p.rap_nodes.push_back(static_cast<int32_t>(key & 0xffffffff));
    }
  }
  for (size_t i = 1; i < p.rap_offsets.size(); ++i) p.rap_offsets[i] += p.rap_offsets[i - 1];
  return p;
}

SimPartResult simpart_partition(const Circuit& c, const SimPartOptions& options) {
  const int32_t k = options.num_partitions;
  if (k < 1 || k > SimPartOptions::kMaxPartitions) {
    throw std::invalid_argument("number of partitions must be in [1, " +
                                std::to_string(SimPartOptions::kMaxPartitions) + "]");
  }
  if (options.gamma < 0) throw std::invalid_argument("gamma must be non-negative");

  SimPartResult result;
  Timer timer;
  majority_grouping_dispatch(c, k, result.group, result.level);
  result.grouping_ms = timer.elapsed_ms();

  timer.restart();
  result.partition = build_hybrid_partition(c, result.level, result.group, k, options.gamma);
  result.hybrid_ms = timer.elapsed_ms();
  return result;
}

}  // namespace simpart
