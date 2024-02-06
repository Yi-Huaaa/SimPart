#include <iostream>
#include <chrono>

#include <fsim/fsim.hpp>


#define num_thread 16// define by user
#define run_seq
#define run_tf
#define run_omp
#define visualization
// #define print_output

int main(int argc, char *argv[]) {

  // from user's perspective
  fsim::Simulator simulator;
  size_t mode;
  std::chrono::duration<double> duration_read_data, duration_SEQ, duration_TF, duration_OMP;

  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string outputfile_path(argv[4]);

  auto start = std::chrono::high_resolution_clock::now();
    simulator.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::high_resolution_clock::now();
  duration_read_data = end - start;
  
#ifdef run_seq
  mode = 0; // mode: (0,1,2) = (SEQUENTIAL, PARALLEL_TF, PARALLEL_openMP);
  start = std::chrono::high_resolution_clock::now();
    simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), 1);
  end = std::chrono::high_resolution_clock::now();
  duration_SEQ = end - start;
#endif

#ifdef run_tf
  // PARALLEL Version - TF
  mode = 1;
  start = std::chrono::high_resolution_clock::now();
    simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);
  end = std::chrono::high_resolution_clock::now();
  duration_TF = end - start;
#endif

#ifdef run_omp
  // PARALLEL Version - OMP
  mode = 2;
  start = std::chrono::high_resolution_clock::now();
    simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);
  end = std::chrono::high_resolution_clock::now();
  duration_OMP = end - start;

#endif

#ifdef visualization
  // write file: visualization_graph
  simulator.visualization_graph(outputfile_path);
#endif 

  // print cost time
  // duration_read_data, duration_SEQ, duration_TF, duration_OMP
  std::cout << "Execution Time:\n";
  std::cout << "Read Data: " << duration_read_data.count() << " seconds\n";
  std::cout << "SEQUENTIAL: " << duration_SEQ.count() << " seconds\n";
  std::cout << "PARALLEL_TF: " << duration_TF.count() << " seconds\n";
  std::cout << "PARALLEL_OMP (#td=" << num_thread << "): " << duration_OMP.count() << " seconds\n";

  
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

  return 0;
}
