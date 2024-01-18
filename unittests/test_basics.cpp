#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

// TODO: include and write many unittests...
// make test
// you can study how taskflow enables unittest

TEST_CASE("Type" * doctest::timeout(300)) {
  int i=1;
  REQUIRE(i == 1);
}

TEST_CASE("Simple.1" * doctest::timeout(300)) {

  const std::string ckt = ".....";
  const std::string flst = ".....";
  const std::string ptn = ".....";

  std::istringstream ickt(ckt);
  std::istringstream iflst(flst);
  std::istringstream iptn(ptn);

  fsim::Simulator sim;

  sim.read(ickt, iflst, iptn);

  sim.run(fsim::Simulator::Mode::SEQUENTIAL);

  for(size_t gid = 0; gid<g_gate_results.size(); gid++) {
    auto& gate = simulator.gate(gid);
    std::cout << "G-result of Gate " << gid << " (level=" << gate.level() << "): "
              << g_gate_results[gid] << '\n';
    REQUIRE( g_gate_results[gid] == ???);
  }

}
