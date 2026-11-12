// simpart: partition a circuit with SimPart (or the RepCut baseline) and
// simulate it on the GPU (or the CPU), reporting runtime and replication.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "simpart/circuit.hpp"
#include "simpart/cpu_simulator.hpp"
#include "simpart/cuda_utils.hpp"
#include "simpart/gpu_simulator.hpp"
#include "simpart/partition.hpp"
#include "simpart/patterns.hpp"
#include "simpart/simpart.hpp"
#include "simpart/timer.hpp"
#ifdef SIMPART_WITH_REPCUT
#include "simpart/repcut.hpp"
#endif

#ifndef SIMPART_VERSION
#define SIMPART_VERSION "unknown"
#endif

namespace {

using namespace simpart;

constexpr const char* kUsage = R"(usage: simpart <circuit.ckt> <patterns.ptn> [options]

Partitioning
  -p, --partitioner NAME  simpart (default) or repcut
  -k, --partitions N      number of partitions k (default: 16)
  -g, --gamma N           SimPart replication threshold (default: 1024);
                          0 gives pure DSP, 'inf' gives pure RAP
  -t, --threads N         CPU threads of Mt-KaHyPar and the CPU simulator (default: 16)

Simulation
  -s, --simulator NAME    graph:  GPU, conditional CUDA graph (default)
                          stream: GPU, one stream launch per kernel
                          cpu:    CPU, OpenMP over the partitions
                          none:   partition only

Measurement
  -r, --runs N            timed repetitions of partitioning and simulation (default: 10)
      --part-runs N       timed repetitions of partitioning (overrides --runs)
      --sim-runs N        timed repetitions of simulation (overrides --runs)
  -w, --warmup N          untimed warm-up repetitions before timing (default: 1)
      --verify            check the simulation results against a sequential CPU simulation
      --csv FILE          append the results as one CSV row to FILE
  -h, --help              show this message
)";

enum class SimulatorKind { Graph, Stream, Cpu, None };

struct Options {
  std::string circuit_path, patterns_path;
  std::string partitioner = "simpart";
  int32_t partitions = 16;
  int64_t gamma = 1024;
  int32_t threads = 16;
  SimulatorKind simulator = SimulatorKind::Graph;
  int runs = 10;
  std::optional<int> part_runs, sim_runs;
  int warmup = 1;
  bool verify = false;
  std::string csv_path;
};

[[noreturn]] void usage_error(const std::string& msg) {
  std::cerr << "simpart: " << msg << "\n\n" << kUsage;
  std::exit(2);
}

int64_t parse_int(const std::string& s, const std::string& flag, int64_t min_value) {
  try {
    size_t pos = 0;
    const long long v = std::stoll(s, &pos);
    if (pos != s.size() || v < min_value) throw std::invalid_argument(s);
    return v;
  } catch (const std::exception&) {
    usage_error("invalid value '" + s + "' for " + flag);
  }
}

Options parse_options(int argc, char** argv) {
  Options o;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      std::exit(0);
    } else if (arg == "-p" || arg == "--partitioner") {
      o.partitioner = value();
      if (o.partitioner != "simpart" && o.partitioner != "repcut") {
        usage_error("unknown partitioner '" + o.partitioner + "'");
      }
    } else if (arg == "-k" || arg == "--partitions") {
      o.partitions = static_cast<int32_t>(parse_int(value(), arg, 1));
    } else if (arg == "-g" || arg == "--gamma") {
      const std::string v = value();
      o.gamma = (v == "inf") ? SimPartOptions::kInfinity : parse_int(v, arg, 0);
    } else if (arg == "-t" || arg == "--threads") {
      o.threads = static_cast<int32_t>(parse_int(value(), arg, 1));
    } else if (arg == "-s" || arg == "--simulator") {
      const std::string v = value();
      if (v == "graph") o.simulator = SimulatorKind::Graph;
      else if (v == "stream") o.simulator = SimulatorKind::Stream;
      else if (v == "cpu") o.simulator = SimulatorKind::Cpu;
      else if (v == "none") o.simulator = SimulatorKind::None;
      else usage_error("unknown simulator '" + v + "'");
    } else if (arg == "-r" || arg == "--runs") {
      o.runs = static_cast<int>(parse_int(value(), arg, 1));
    } else if (arg == "--part-runs") {
      o.part_runs = static_cast<int>(parse_int(value(), arg, 1));
    } else if (arg == "--sim-runs") {
      o.sim_runs = static_cast<int>(parse_int(value(), arg, 1));
    } else if (arg == "-w" || arg == "--warmup") {
      o.warmup = static_cast<int>(parse_int(value(), arg, 0));
    } else if (arg == "--verify") {
      o.verify = true;
    } else if (arg == "--csv") {
      o.csv_path = value();
    } else if (!arg.empty() && arg[0] == '-') {
      usage_error("unknown option '" + arg + "'");
    } else {
      positional.push_back(arg);
    }
  }
  if (positional.size() != 2) usage_error("expected a circuit file and a pattern file");
  o.circuit_path = positional[0];
  o.patterns_path = positional[1];
#ifndef SIMPART_WITH_REPCUT
  if (o.partitioner == "repcut") usage_error("this build has no RepCut support (Mt-KaHyPar not found)");
#endif
  return o;
}

std::string gamma_string(int64_t gamma) {
  return gamma == SimPartOptions::kInfinity ? "inf" : std::to_string(gamma);
}

const char* simulator_name(SimulatorKind s) {
  switch (s) {
    case SimulatorKind::Graph: return "graph";
    case SimulatorKind::Stream: return "stream";
    case SimulatorKind::Cpu: return "cpu";
    case SimulatorKind::None: return "none";
  }
  return "?";
}

// Runs `fn` warmup + runs times and returns the timings of the timed runs.
std::vector<double> measure(int warmup, int runs, const std::function<void()>& fn) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> times;
  for (int i = 0; i < runs; ++i) {
    Timer t;
    fn();
    times.push_back(t.elapsed_ms());
  }
  return times;
}

struct Report {
  // Partitioning
  Summary part;
  std::string part_detail;
  double grouping_ms = -1, hybrid_ms = -1;   // SimPart stages
  double build_ms = -1, assemble_ms = -1;    // RepCut hypergraph construction / cone expansion
  int64_t hg_vertices = -1, hg_hyperedges = -1;
  // Simulation
  Summary sim;
  double sim_setup_ms = -1;
  std::string verified = "skipped";
};

std::string fmt(double v, int precision = 3) {
  if (v < 0) return "";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, v);
  return buf;
}

void append_csv(const Options& o, const Circuit& c, const Partition& p, const ReplicationStats& rep,
                const Report& r) {
  const bool exists = std::filesystem::exists(o.csv_path);
  std::ofstream out(o.csv_path, std::ios::app);
  if (!out) throw std::runtime_error("cannot open " + o.csv_path);
  if (!exists) {
    out << "circuit,partitioner,k,gamma,threads,simulator,nodes,edges,levels,dsp_levels,"
           "part_runs,part_ms,part_std_ms,grouping_ms,hybrid_ms,build_ms,assemble_ms,"
           "hg_vertices,hg_hyperedges,node_ratio,edge_ratio,sim_runs,sim_setup_ms,sim_ms,"
           "sim_std_ms,verified\n";
  }
  auto opt_int = [](int64_t v) { return v < 0 ? std::string() : std::to_string(v); };
  out << std::filesystem::path(o.circuit_path).filename().string() << ',' << o.partitioner << ','
      << o.partitions << ',' << (o.partitioner == "simpart" ? gamma_string(o.gamma) : "") << ','
      << o.threads << ',' << simulator_name(o.simulator) << ',' << c.num_nodes() << ','
      << c.num_edges() << ',' << p.num_levels << ',' << p.dsp_levels << ',' << r.part.count << ','
      << fmt(r.part.mean) << ',' << fmt(r.part.stddev) << ',' << fmt(r.grouping_ms) << ','
      << fmt(r.hybrid_ms) << ',' << fmt(r.build_ms) << ',' << fmt(r.assemble_ms) << ','
      << opt_int(r.hg_vertices) << ',' << opt_int(r.hg_hyperedges) << ','
      << fmt(rep.node_ratio, 6) << ',' << fmt(rep.edge_ratio, 6) << ',' << r.sim.count << ','
      << fmt(r.sim_setup_ms) << ',' << (r.sim.count ? fmt(r.sim.mean) : "") << ','
      << (r.sim.count ? fmt(r.sim.stddev) : "") << ',' << r.verified << '\n';
}

// Compares simulation outputs with the sequential reference.
std::string verify(const Circuit& c, const Patterns& patterns, const std::vector<uint64_t>& po_values,
                   const std::vector<uint64_t>& node_values) {
  const SimulationResult ref = simulate_reference(c, patterns);
  size_t bad_po = 0, bad_nodes = 0;
  for (size_t i = 0; i < ref.po_values.size(); ++i) bad_po += (po_values[i] != ref.po_values[i]);
  for (size_t i = 0; i < ref.node_values.size(); ++i) bad_nodes += (node_values[i] != ref.node_values[i]);
  std::cout << "verify      ";
  if (bad_po == 0 && bad_nodes == 0) {
    std::cout << "PASSED (" << patterns.num_rounds << " rounds x " << c.num_pos << " POs, "
              << c.num_nodes() << " node values of the last round)\n";
    return "passed";
  }
  std::cout << "FAILED (" << bad_po << " wrong PO values, " << bad_nodes << " wrong node values)\n";
  return "failed";
}

int run(const Options& o) {
  // Create the CUDA context up front so that it is not charged to partitioning.
  SIMPART_CUDA_CHECK(cudaFree(nullptr));
  int device = 0;
  cudaDeviceProp prop{};
  SIMPART_CUDA_CHECK(cudaGetDevice(&device));
  SIMPART_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "SimPart " << SIMPART_VERSION << " | GPU: " << prop.name << " (sm_" << prop.major
            << prop.minor << ")\n";

  Timer io_timer;
  const Circuit circuit = read_circuit(o.circuit_path);
  const Patterns patterns = read_patterns(o.patterns_path, circuit.num_pis);
  std::cout << "circuit     " << std::filesystem::path(o.circuit_path).filename().string() << ": "
            << circuit.num_nodes() << " nodes, " << circuit.num_edges() << " edges, "
            << circuit.num_pis << " PIs, " << circuit.num_pos << " POs\n"
            << "patterns    " << patterns.num_patterns << " patterns (" << patterns.num_rounds
            << " rounds), files read in " << fmt(io_timer.elapsed_ms(), 1) << " ms\n";

  const int part_runs = o.part_runs.value_or(o.runs);
  const int sim_runs = o.sim_runs.value_or(o.runs);
  Report report;
  Partition partition;

  if (o.partitioner == "simpart") {
    SimPartOptions so;
    so.num_partitions = o.partitions;
    so.gamma = o.gamma;
    SimPartResult result;
    std::vector<double> grouping, hybrid;
    const auto times = measure(o.warmup, part_runs, [&] {
      result = simpart_partition(circuit, so);
      grouping.push_back(result.grouping_ms);
      hybrid.push_back(result.hybrid_ms);
    });
    grouping.erase(grouping.begin(), grouping.begin() + o.warmup);
    hybrid.erase(hybrid.begin(), hybrid.begin() + o.warmup);
    report.part = summarize(times);
    report.grouping_ms = summarize(grouping).mean;
    report.hybrid_ms = summarize(hybrid).mean;
    partition = std::move(result.partition);
    std::cout << "partitioner SimPart, k=" << o.partitions << ", gamma=" << gamma_string(o.gamma) << "\n"
              << "  time      " << fmt(report.part.mean) << " ms (grouping " << fmt(report.grouping_ms)
              << " + hybrid " << fmt(report.hybrid_ms) << "), mean of " << part_runs
              << " runs, std " << fmt(report.part.stddev) << "\n";
  } else {
#ifdef SIMPART_WITH_REPCUT
    RepCutOptions ro;
    ro.num_partitions = o.partitions;
    ro.num_threads = o.threads;
    RepCutResult result;
    std::vector<double> build, part, assemble;
    measure(o.warmup, part_runs, [&] {
      result = repcut_partition(circuit, ro);
      build.push_back(result.hypergraph_ms());
      part.push_back(result.partition_ms);
      assemble.push_back(result.assemble_ms);
    });
    for (auto* v : {&build, &part, &assemble}) v->erase(v->begin(), v->begin() + o.warmup);
    report.part = summarize(part);
    report.build_ms = summarize(build).mean;
    report.assemble_ms = summarize(assemble).mean;
    report.hg_vertices = result.num_vertices;
    report.hg_hyperedges = result.num_hyperedges;
    partition = std::move(result.partition);
    std::cout << "partitioner RepCut (Mt-KaHyPar, " << o.threads << " threads), k=" << o.partitions
              << "\n"
              << "  hypergraph " << result.num_vertices << " vertices, " << result.num_hyperedges
              << " hyperedges, " << result.num_pins << " pins, built in " << fmt(report.build_ms)
              << " ms\n"
              << "  time      " << fmt(report.part.mean) << " ms (Mt-KaHyPar), mean of "
              << part_runs << " runs, std " << fmt(report.part.stddev) << "; cone expansion "
              << fmt(report.assemble_ms) << " ms\n";
#endif
  }

  const ReplicationStats rep = replication_stats(circuit, partition);
  std::cout << "  layout    " << partition.num_levels << " levels: DSP [0, " << partition.dsp_levels
            << "), RAP [" << partition.dsp_levels << ", " << partition.num_levels << ") x "
            << partition.num_groups << " groups (" << rep.rap_nodes << " node instances)\n"
            << "  replication R|V| = " << fmt(rep.node_ratio, 5) << ", R|E| = " << fmt(rep.edge_ratio, 5)
            << "\n";

  std::vector<uint64_t> po_values, node_values;
  if (o.simulator == SimulatorKind::Graph || o.simulator == SimulatorKind::Stream) {
    const Executor executor = o.simulator == SimulatorKind::Graph ? Executor::Graph : Executor::Stream;
    GpuSimulator sim(circuit, partition, patterns);
    if (executor == Executor::Graph) report.sim_setup_ms = sim.prepare_graph();
    report.sim = summarize(measure(o.warmup, sim_runs, [&] { sim.run(executor); }));
    std::cout << "simulator   GPU, " << (executor == Executor::Graph ? "conditional CUDA graph" : "stream launches");
    if (executor == Executor::Graph) std::cout << " (graph setup " << fmt(report.sim_setup_ms) << " ms)";
    std::cout << "\n  time      " << fmt(report.sim.mean) << " ms, mean of " << sim_runs
              << " runs, std " << fmt(report.sim.stddev) << "\n";
    if (o.verify) {
      po_values = sim.po_values();
      node_values = sim.node_values();
    }
  } else if (o.simulator == SimulatorKind::Cpu) {
    CpuSimulator sim(circuit, partition, patterns, o.threads);
    report.sim = summarize(measure(o.warmup, sim_runs, [&] { sim.run(); }));
    std::cout << "simulator   CPU, " << o.threads << " threads\n"
              << "  time      " << fmt(report.sim.mean) << " ms, mean of " << sim_runs
              << " runs, std " << fmt(report.sim.stddev) << "\n";
    if (o.verify) {
      const SimulationResult r = sim.result();
      po_values = r.po_values;
      node_values = r.node_values;
    }
  }
  if (o.verify && o.simulator != SimulatorKind::None) {
    report.verified = verify(circuit, patterns, po_values, node_values);
  }
  if (!o.csv_path.empty()) append_csv(o, circuit, partition, rep, report);
  return report.verified == "failed" ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  try {
    return run(options);
  } catch (const std::exception& e) {
    std::cerr << "simpart: error: " << e.what() << "\n";
    return 1;
  }
}
