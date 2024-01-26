#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <fsim/fsim.hpp>

/*
// TODO:
design corner circuits - especially those with strange structures

1. one node
2. two nodes with edge
3. two nodes without edges

parameterize your unittest as a function that takes execution mode as an example

e.g.,
void test_simple1(execution_mode, num_threads)


Unittests: 
1. test different modes
2. test wether levelized success 
3. 


*/

TEST_CASE("Simple.1" * doctest::timeout(300)) {

  std::istringstream ickt("\
  3\
  1\
  4\
  7\
  7 2 1 5 0\
  4 2 1 5 1\
  5 2 1 3 0\
  6 2 2 7 1 4 0\
  1 2 1 6 0\
  0 2 1 6 1\
  2 2 1 7 0\
  0 5 2 1\
  ");
  std::istringstream iflst("\
  2\
  2 1\
  4 0\
  ");
  std::istringstream iptn("\
  3\
  2\
  4\
  1\
  ");

  fsim::Simulator sim;

  sim.read(ickt, iflst, iptn);

  // TODO:
  //REQUIRE(sim.num_gates() == ???);
  //REQUIRE(sim.num_wires() == ???);

  sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

  auto& g_gate_results = sim.g_gate_results();
  std::vector<size_t> answer = {2,4,1,6,0,1,6,6};

  //std::cout << "Unittests - simple:\n";
  for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
    auto& gate = sim.gate(gid);
    //std::cout << "Unittests G-result of Gate " << gid << " (level=" << gate.level() << "): "
    //          << g_gate_results[gid] << '\n';
    REQUIRE(g_gate_results[gid] == answer[gid]); // 1 -> ???
  }
}
