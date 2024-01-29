#include <iostream>

#include <fsim/fsim.hpp>

#define num_thread 4 // define by user

int main(int argc, char *argv[]) {

  // TODO: from user's perspective
  fsim::Simulator simulator;
  size_t mode;
  
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);

  simulator.read(ckt_path, flst_path, ptn_path);
  mode = 0; // mode: (0,1,2) = (SEQUENTIAL, PARALLEL_TF, PARALLEL_openMP); 
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), 1);

  // Output results

  auto& g_gate_results = simulator.g_gate_results();
  std::cout << "User space resutls:" << std::endl;
  for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
    auto& gate = simulator.gate(gid);
    std::cout << "G-result of Gate " << gid << " (level=" << gate.level() << "): " << g_gate_results[gid] << '\n';
  }

  // PARALLEL Version - TF
  mode = 1;
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);


  // PARALLEL Version - TF
  mode = 2;
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)), num_thread);


  return 0;
}
