#include <iostream>

#include <fsim/fsim.hpp>

int main(int argc, char *argv[]) {

  // TODO: from user's perspective
  fsim::Simulator simulator;
  

  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);

  simulator.read(ckt_path, flst_path, ptn_path);
  size_t mode = 0; // mode: (0, 1) = (SEQUENTIAL, PARALLEL); 
  simulator.run(fsim::Simulator::Mode(static_cast<size_t>(mode)));

  // TODO:
  auto& g_gate_results = simulator.g_gate_results();

  std::cout << "User space resutls:" << std::endl;
  for(size_t gid = 0; gid<g_gate_results.size(); gid++) {
    auto& gate = simulator.gate(gid);
    std::cout << "G-result of Gate " << gid << " (level=" << gate.level() << "): "
              << g_gate_results[gid] << '\n';
  }

  // // user to access gates in the simulator
  // for(auto& gate : simulator.gates()) {
  //     // std::cout << gate.output_value() << '\n';
  // }

  return 0;
}
