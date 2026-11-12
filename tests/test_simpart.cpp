#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/cpu_simulator.hpp"
#include "simpart/gpu_simulator.hpp"
#include "simpart/levelize.hpp"
#include "simpart/partition.hpp"
#include "simpart/patterns.hpp"
#include "simpart/simpart.hpp"
#ifdef SIMPART_WITH_REPCUT
#include "simpart/repcut.hpp"
#endif

using namespace simpart;

namespace {

// The 4-gate example of benchmark/simple_circuits/simple.ckt:
// PIs 0-2, PO 3, gates 4 (INV), 5 (NOR), 6 (OR), 7 (AND).
constexpr const char* kSimpleCircuit = R"(3 1 4 7
7 0 1 5 1
4 0 1 5 2
5 0 1 3 1
6 0 2 7 2 4 1
1 0 1 6 1
0 0 1 6 2
2 0 1 7 1
0 5 2 1
)";

std::vector<int32_t> fanins_of(const Circuit& c, int32_t v) {
  return {c.fanins.begin() + c.fanin_offsets[v], c.fanins.begin() + c.fanin_offsets[v + 1]};
}

const std::string kBench = SIMPART_BENCHMARK_DIR;

const std::vector<std::string> kCircuits = {
    "simple_circuits/simple",        "simple_circuits/oneGate1",
    "simple_circuits/oneGate2",      "simple_circuits/linear_chain1",
    "simple_circuits/linear_chain2", "simple_circuits/independent_graph1",
    "simple_circuits/ring1",         "simple_circuits/RepCutOriginal",
    "c_circuits/c17",                "c_circuits/c432",
    "c_circuits/c499",               "c_circuits/c880",
    "c_circuits/c1355",              "c_circuits/c1908",
    "c_circuits/c2670",              "c_circuits/c3540",
    "c_circuits/c5315",              "c_circuits/c6288",
    "c_circuits/c7552",              "large_circuits/random1",
};

struct Testcase {
  std::string name;
  Circuit circuit;
  Patterns patterns;
  SimulationResult reference;
};

const std::vector<Testcase>& testcases() {
  static const std::vector<Testcase> cases = [] {
    std::vector<Testcase> out;
    for (const auto& name : kCircuits) {
      Testcase t;
      t.name = name;
      t.circuit = read_circuit(kBench + "/" + name + ".ckt");
      t.patterns = read_patterns(kBench + "/" + name + ".ptn", t.circuit.num_pis);
      t.reference = simulate_reference(t.circuit, t.patterns);
      out.push_back(std::move(t));
    }
    return out;
  }();
  return cases;
}

void check_gpu(const Testcase& t, const Partition& p) {
  GpuSimulator sim(t.circuit, p, t.patterns);
  for (Executor e : {Executor::Stream, Executor::Graph}) {
    INFO("executor = " << to_string(e));
    for (int repeat = 0; repeat < 2; ++repeat) {
      sim.run(e);
      CHECK(sim.po_values() == t.reference.po_values);
      CHECK(sim.node_values() == t.reference.node_values);
    }
  }
}

void check_cpu(const Testcase& t, const Partition& p, int threads) {
  CpuSimulator sim(t.circuit, p, t.patterns, threads);
  sim.run();
  const SimulationResult r = sim.result();
  CHECK(r.po_values == t.reference.po_values);
  CHECK(r.node_values == t.reference.node_values);
}

}  // namespace

TEST_CASE("parse a circuit") {
  const Circuit c = parse_circuit(kSimpleCircuit);
  CHECK(c.num_nodes() == 8);
  CHECK(c.num_edges() == 8);
  CHECK(c.types[0] == GateType::PI);
  CHECK(c.types[3] == GateType::PO);
  CHECK(c.types[4] == GateType::INV);
  CHECK(c.types[5] == GateType::NOR);
  CHECK(c.types[6] == GateType::OR);
  CHECK(c.types[7] == GateType::AND);
  // Fanins keep the order of the file.
  CHECK(fanins_of(c, 5) == std::vector<int32_t>{7, 4});
  CHECK(fanins_of(c, 6) == std::vector<int32_t>{1, 0});
  CHECK(fanins_of(c, 7) == std::vector<int32_t>{6, 2});
  CHECK(c.fanout_count(6) == 2);
}

TEST_CASE("reject malformed circuits") {
  CHECK_THROWS(parse_circuit("1 1 0 2\n0 0 1 1 0\n0 0 1 1 1\n"));    // PO with two fanins
  CHECK_THROWS(parse_circuit("1 1 1 2\n0 0 1 2 0\n2 0 1 1 0\n9\n"));  // bad gate type
  CHECK_THROWS(parse_circuit("1 1 1 2\n0 0 1 2 0\n"));                // truncated
  CHECK_THROWS(parse_circuit("1 1 1 2\n0 0 1 5 0\n2 0 1 1 0\n0\n"));  // node out of range
}

TEST_CASE("levelize and simulate a small circuit") {
  const Circuit c = parse_circuit(kSimpleCircuit);
  const Levels levels = levelize(c);
  CHECK(levels.num_levels == 5);
  CHECK(levels.level == std::vector<int32_t>{0, 0, 0, 4, 2, 3, 1, 2});

  // 3 patterns, PI words 2, 4, 1:  OR = 6, INV = ~6, AND = 0, NOR = 6.
  const Patterns p = parse_patterns("3\n2 4 1\n", c.num_pis);
  CHECK(p.num_rounds == 1);
  CHECK(p.round_mask(0) == 0b111);
  const SimulationResult r = simulate_reference(c, p);
  CHECK(r.po_values == std::vector<uint64_t>{6});
  CHECK(r.node_values[4] == ~uint64_t(6));
}

TEST_CASE("SimPart partitions are valid and simulate correctly") {
  for (const auto& t : testcases()) {
    const Levels levels = levelize(t.circuit);
    for (int32_t k : {1, 3, 16}) {
      for (int64_t gamma : {int64_t(0), int64_t(1), int64_t(4), int64_t(64), int64_t(1024),
                            SimPartOptions::kInfinity}) {
        INFO(t.name << ", k = " << k << ", gamma = " << gamma);
        const SimPartResult r = simpart_partition(t.circuit, {k, gamma});
        CHECK(r.level == levels.level);
        for (int32_t g : r.group) CHECK((g >= 0 && g < k));
        CHECK_NOTHROW(validate_partition(t.circuit, r.level, r.partition));
        if (gamma == 0) CHECK(r.partition.dsp_levels == levels.num_levels);
        if (gamma == SimPartOptions::kInfinity) CHECK(r.partition.dsp_levels == 0);
        check_gpu(t, r.partition);
        check_cpu(t, r.partition, 4);
      }
    }
  }
}

TEST_CASE("majority-based grouping is deterministic") {
  const Testcase& t = testcases().back();
  const SimPartResult a = simpart_partition(t.circuit, {16, 1024});
  const SimPartResult b = simpart_partition(t.circuit, {16, 1024});
  CHECK(a.group == b.group);
  CHECK(a.partition.rap_nodes == b.partition.rap_nodes);
}

TEST_CASE("RAP levels wider than a thread block") {
  // gamma = inf on random1 yields RAP levels with more than 1024 nodes per group.
  const Testcase& t = testcases().back();
  const SimPartResult r = simpart_partition(t.circuit, {1, SimPartOptions::kInfinity});
  int32_t widest = 0;
  for (int32_t l = 0; l < r.partition.num_levels; ++l) {
    const int64_t i = r.partition.rap_index(0, l);
    widest = std::max(widest, r.partition.rap_offsets[i + 1] - r.partition.rap_offsets[i]);
  }
  CHECK(widest > GpuSimulator::kBlockSize);
  check_gpu(t, r.partition);
}

#ifdef SIMPART_WITH_REPCUT
TEST_CASE("RepCut partitions are valid and simulate correctly") {
  for (const auto& t : testcases()) {
    for (int32_t k : {2, 16}) {
      INFO(t.name << ", k = " << k);
      const RepCutResult r = repcut_partition(t.circuit, {k, 4});
      CHECK(r.partition.dsp_levels == 0);
      CHECK_NOTHROW(validate_partition(t.circuit, r.level, r.partition));
      check_gpu(t, r.partition);
      check_cpu(t, r.partition, 1);
      check_cpu(t, r.partition, 4);
    }
  }
}
#endif
