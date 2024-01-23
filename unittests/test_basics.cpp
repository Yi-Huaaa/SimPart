#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <fsim/fsim.hpp>

// TODO: include and write many unittests...
// make test
// you can study how taskflow enables unittest

TEST_CASE("Type" * doctest::timeout(300)) {
  int i=1;
  REQUIRE(i == 1);
}

TEST_CASE("Simple.1" * doctest::timeout(300)) {
  std::cout << "Unit test, Simple.1\n";

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

  sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

  auto& g_gate_results = sim.g_gate_results();
  std::vector<size_t> answer = {2,4,1,6,0,1,6,6};

  std::cout << "Unittests - simple:\n";
  for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
    auto& gate = sim.gate(gid);
    std::cout << "Unittests G-result of Gate " << gid << " (level=" << gate.level() << "): "
              << g_gate_results[gid] << '\n';
    REQUIRE(g_gate_results[gid] == answer[gid]); // 1 -> ???
  }
}
