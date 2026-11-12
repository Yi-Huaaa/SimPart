#include "simpart/cpu_simulator.hpp"

#include <atomic>
#include <stdexcept>

#include "simpart/gate_eval.hpp"
#include "simpart/levelize.hpp"

namespace simpart {

namespace {

// Replicated nodes are written by several threads (always with the same
// value), so RAP groups access node values through relaxed atomics. On x86
// these compile to plain loads and stores.
struct RelaxedLoad {
  uint64_t* values;
  uint64_t operator()(int32_t u) const {
    return std::atomic_ref<uint64_t>(values[u]).load(std::memory_order_relaxed);
  }
};

inline void relaxed_store(uint64_t& dst, uint64_t value) {
  std::atomic_ref<uint64_t>(dst).store(value, std::memory_order_relaxed);
}

inline void eval_node(const Circuit& c, const Patterns& p, int32_t round, int32_t v,
                      uint64_t* values, uint64_t* po_values) {
  const GateType type = c.types[v];
  uint64_t value;
  if (type == GateType::PI) {
    value = p.words[int64_t(round) * c.num_pis + v];
  } else {
    const int32_t begin = c.fanin_offsets[v];
    value = eval_gate(type, c.fanins.data() + begin, c.fanin_offsets[v + 1] - begin,
                      RelaxedLoad{values});
    if (type == GateType::PO) {
      relaxed_store(po_values[int64_t(round) * c.num_pos + (v - c.num_pis)], value);
    }
  }
  relaxed_store(values[v], value);
}

void check_patterns(const Circuit& c, const Patterns& p) {
  if (p.num_pis != c.num_pis) {
    throw std::invalid_argument("patterns do not match the number of PIs of the circuit");
  }
}

}  // namespace

SimulationResult simulate_reference(const Circuit& c, const Patterns& p) {
  check_patterns(c, p);
  const Levels levels = levelize(c);
  SimulationResult out;
  out.node_values.assign(c.num_nodes(), 0);
  out.po_values.assign(static_cast<size_t>(p.num_rounds) * c.num_pos, 0);
  for (int32_t r = 0; r < p.num_rounds; ++r) {
    for (int32_t v : levels.nodes) eval_node(c, p, r, v, out.node_values.data(), out.po_values.data());
  }
  return out;
}

CpuSimulator::CpuSimulator(const Circuit& circuit, const Partition& partition,
                           const Patterns& patterns, int num_threads)
    : circuit_(circuit),
      partition_(partition),
      patterns_(patterns),
      num_threads_(num_threads),
      values_(circuit.num_nodes(), 0),
      po_values_(static_cast<size_t>(patterns.num_rounds) * circuit.num_pos, 0) {
  check_patterns(circuit, patterns);
  if (num_threads < 1) throw std::invalid_argument("number of threads must be positive");
}

void CpuSimulator::run() {
  const Circuit& c = circuit_;
  const Partition& p = partition_;
  uint64_t* values = values_.data();
  uint64_t* po_values = po_values_.data();
  const int32_t rap_levels = p.rap_levels();

  for (int32_t r = 0; r < patterns_.num_rounds; ++r) {
    for (int32_t l = 0; l < p.dsp_levels; ++l) {
#pragma omp parallel for num_threads(num_threads_) schedule(static)
      for (int32_t i = p.dsp_offsets[l]; i < p.dsp_offsets[l + 1]; ++i) {
        eval_node(c, patterns_, r, p.dsp_nodes[i], values, po_values);
      }
    }
#pragma omp parallel for num_threads(num_threads_) schedule(static)
    for (int32_t g = 0; g < p.num_groups; ++g) {
      const int64_t begin = p.rap_offsets[int64_t(g) * rap_levels];
      const int64_t end = p.rap_offsets[int64_t(g + 1) * rap_levels];
      for (int64_t i = begin; i < end; ++i) eval_node(c, patterns_, r, p.rap_nodes[i], values, po_values);
    }
  }
}

}  // namespace simpart
