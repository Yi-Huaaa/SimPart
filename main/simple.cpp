#include <iostream>

#include <fsim/fsim.hpp>

int main(int argc, char *argv[]) {

  // TODO: from user's perspective
  fsim::Simulator simulator;

  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);

  simulator.read(ckt_path, flst_path, ptn_path);
  simulator.run();

  // // user to access gates in the simulator
  // for(auto& gate : simulator.gates()) {
  //     // std::cout << gate.output_value() << '\n';
  // }

  return 0;
}
