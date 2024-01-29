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

/**
 * graph TEST_CASE
 * PPSFP_simple
 * one PI
 * one PO
 * one PI - one PO
 * one node: two PIs - one gate (two inputs gate) - one PO
 * one linear chain: one PI - (gate (INV))^n - one PO (n = 1)
 * one linear chain: one PI - (gate (INV))^n - one PO (n = 10)
 * two independent linear chain: (one PI - (gate (INV))^n - one PO)^2 (n = 10)
 * Ring
 * test FFR
 * X-style circuit
 * 
*/

/** 
 * For all testcases: 
 * test `read`, `run`, `output`
 * 
*/

/**
 * test different mode
 * 
 * 
*/

TEST_CASE("PPSFP_simple.1" * doctest::timeout(300)) {
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
  // check var

  sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

  auto& g_pi_results = sim.g_pi_results();
  auto& g_gate_results = sim.g_gate_results();
  auto& g_po_results = sim.g_po_results();
  std::vector<size_t> pi_ans = {2,4,1};
  std::vector<size_t> gate_answer = {6,0,1,6};
  std::vector<size_t> po_answer = {6};

  for(size_t piid = 0; piid < g_pi_results.size(); piid++) {
    auto& pi = sim.pi(piid);
    REQUIRE_MESSAGE(g_pi_results[piid] == pi_ans[piid], "PI result mismatch");
  }
  for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
    auto& gate = sim.gate(gid);
    REQUIRE_MESSAGE(g_gate_results[gid] == gate_answer[gid], "Gate result mismatch");
  }
  for(size_t poid = 0; poid < g_po_results.size(); poid++) {
    auto& po = sim.po(poid);
    REQUIRE_MESSAGE(g_po_results[poid] == po_answer[poid], "PO result mismatch");
  }
}
