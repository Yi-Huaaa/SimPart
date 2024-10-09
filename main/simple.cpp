#include <chrono>
#include <iostream>

#include <fsim/fsim.hpp>

#define num_thread 8 // define by user
#define run_seq
// #define run_tf
// #define run_omp
// #define visualization
// #define print_output
#define print_output_arr_format

double round_to(double value, double precision = 1.0) {
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {

  // from user's perspective
  fsim::Simulator simulator;
  size_t mode;
  std::chrono::duration<double> duration_read_data, duration_SEQ, duration_TF,
      duration_OMP;

  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string outputfile_path(argv[4]);

  auto start = std::chrono::steady_clock::now();
  simulator.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;

#ifdef run_seq
  mode = 0; // mode: (0,1,2) = (SEQUENTIAL, PARALLEL_TF, PARALLEL_openMP);
  start = std::chrono::steady_clock::now();
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), 1);
  end = std::chrono::steady_clock::now();
  duration_SEQ = end - start;
#endif

#ifdef run_tf
  // PARALLEL Version - TF
  mode = 1;
  start = std::chrono::steady_clock::now();
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);
  end = std::chrono::steady_clock::now();
  duration_TF = end - start;
#endif

#ifdef run_omp
  // PARALLEL Version - OMP
  mode = 2;
  start = std::chrono::steady_clock::now();
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);
  end = std::chrono::steady_clock::now();
  duration_OMP = end - start;
#endif

#ifdef visualization
  // write file: visualization_graph
  simulator.visualization_graph(outputfile_path);
#endif

  // print cost time
  // duration_read_data, duration_SEQ, duration_TF, duration_OMP
  std::cout << "Execution Time:\n";
  std::cout << "Read Data: "
            << round_to((duration_read_data.count()) * 1000, 0.001) << " ms\n";
  std::cout << "SEQUENTIAL: " << round_to((duration_SEQ.count()) * 1000, 0.001)
            << " ms\n";
  std::cout << "PARALLEL_TF: " << round_to((duration_TF.count()) * 1000, 0.001)
            << " ms\n";
  std::cout << "PARALLEL_OMP (#td=" << num_thread
            << "): " << round_to((duration_OMP.count()) * 1000, 0.001)
            << " ms\n";

#ifdef print_output
  // Output results
  auto &g_pi_results = simulator.g_pi_results();
  auto &g_gate_results = simulator.g_gate_results();
  auto &g_po_results = simulator.g_po_results();

  std::cout << "User space resutls:" << std::endl;
  for (size_t piid = 0; piid < g_pi_results.size(); piid++) {
    auto &pii = simulator.pi(piid);
    std::cout << "G-result of PI " << piid << " (level=" << pii.level()
              << "): " << g_pi_results[piid] << '\n';
  }
  for (size_t gid = 0; gid < g_gate_results.size(); gid++) {
    auto &gate = simulator.gate(gid);
    std::cout << "G-result of Gate " << gid << " (level=" << gate.level()
              << "): " << g_gate_results[gid] << '\n';
  }
  for (size_t poid = 0; poid < g_po_results.size(); poid++) {
    auto &poo = simulator.gate(poid);
    std::cout << "G-result of PO " << poid << " (level=" << poo.level()
              << "): " << g_po_results[poid] << '\n';
  }
#endif

#ifdef print_output_arr_format
  // Output results
  auto &g_pi_results = simulator.g_pi_results();
  auto &g_gate_results = simulator.g_gate_results();
  auto &g_po_results = simulator.g_po_results();

  for (size_t piid = 0; piid < g_pi_results.size(); piid++) {
    // auto &pii = simulator.pi(piid);
    std::cout << "arr[" << piid << "] = " << g_pi_results[piid] << '\n';
  }
  for (size_t gid = 0; gid < g_gate_results.size(); gid++) {
    // auto &gate = simulator.gate(gid);
    std::cout << "arr[" << gid + (simulator.num_PIs()) << "] = " << g_gate_results[gid] << '\n';
  }
  for (size_t poid = 0; poid < g_po_results.size(); poid++) {
    // auto &poo = simulator.gate(poid);
    std::cout << "arr[" << poid + (simulator.num_PIs()+simulator.num_gates()) 
              << "] = " << g_po_results[poid] << '\n';
  }
#endif

  return 0;
}