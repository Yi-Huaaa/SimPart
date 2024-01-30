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

// define test cases
std::vector<std::vector<std::string>> test_cases {
 // CML
 {
  "\
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
  ",
  "\
  2\
  2 1\
  4 0\
  ",
  "\
  3\
  2\
  4\
  1\
  "
 },
  // oneGate   
 {
  "\
  2\
  1\
  1\
  3\
  1 2 1 3 0\
  0 2 1 3 1\
  3 2 1 2 0\
  1\
  ",
  "\
  3\
  1 1\
  2 0\
  0 0\
  ",
  "\
  2\
  1\
  ",
 },


};

// helper functions
void _read_test_case(fsim::Simulator& sim, size_t test_case_id) {
  std::istringstream ickt(test_cases[test_case_id][0]);
  std::istringstream iflst(test_cases[test_case_id][1]);
  std::istringstream iptn(test_cases[test_case_id][2]);
  sim.read(ickt, iflst, iptn);
}

void _check_num_vars(fsim::Simulator& sim, std::vector<size_t>& ans) {
  // check var
  REQUIRE_MESSAGE(sim.num_PIs() == ans[0], "_num_PIs mismatch");
  REQUIRE_MESSAGE(sim.num_gates() == ans[1], "_num_gates mismatch");
  REQUIRE_MESSAGE(sim.num_POs() == ans[2], "_num_POs mismatch");
  REQUIRE_MESSAGE(sim.num_wires() == ans[3], "_num_wires mismatch");
  REQUIRE_MESSAGE(sim.num_rounds() == ans[4], "_num_rounds mismatch");
}


void _check_good_case_results(fsim::Simulator& sim, std::vector<size_t>& pi_ans, std::vector<size_t>& gate_ans, std::vector<size_t>& po_ans) {
  auto& g_pi_results = sim.g_pi_results();
  auto& g_gate_results = sim.g_gate_results();
  auto& g_po_results = sim.g_po_results();

  for(size_t piid = 0; piid < g_pi_results.size(); piid++) {
    REQUIRE_MESSAGE(g_pi_results[piid] == pi_ans[piid], "PI result mismatch");
  }
  for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
    REQUIRE_MESSAGE(g_gate_results[gid] == gate_ans[gid], "Gate result mismatch");
  }
  for(size_t poid = 0; poid < g_po_results.size(); poid++) {
    REQUIRE_MESSAGE(g_po_results[poid] == po_ans[poid], "PO result mismatch");
  }
}


TEST_CASE("Check num vars (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);

  std::vector<size_t> ans = {3, 4, 1, 7, 1};
  _check_num_vars(sim, ans);
}

TEST_CASE("Check tpg sort order (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  // check topological sort results
  // _gate_idx_to_tpg_order_idx -> this is index transfer from (1) original index (readin from stream) to (2) after topologicacl sort the new index of each gate
  std::vector<size_t> tpg_res = {0, 1, 2, 7, 5, 6, 3, 4};
  std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();
  for (size_t i = 0; i < gate_tpg_order.size(); i++) {
    REQUIRE_MESSAGE(gate_tpg_order[i] == tpg_res[i], "topological sort results order mismatch");
  }
}

TEST_CASE("Check levelization (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  // check levelization
  // _num_gates_per_level
  std::vector<size_t> num_gate_per_lvl_res = {3, 1, 2, 1, 1};
  std::vector<size_t> num_gate_per_lvl = sim.num_gates_per_level();
  REQUIRE_MESSAGE(num_gate_per_lvl.size() == num_gate_per_lvl_res.size(), "num_gate_per_lvl.size()");
  for (size_t i = 0; i < num_gate_per_lvl.size(); i++) {
    REQUIRE_MESSAGE(num_gate_per_lvl[i] == num_gate_per_lvl_res[i], "num_gates_per_level results order mismatch");
  }  

  // check _PIs, _gates, _POs levelization
  // check PI
  std::vector<size_t> pi_lvl_ans = {0, 0, 0};
  for (size_t i = 0; i < sim.num_PIs(); i++) {
    REQUIRE_MESSAGE(sim.pi(i).level() == pi_lvl_ans[i], "PI level mismatch");
  }
  // check gate
  std::vector<size_t> gate_lvl_ans = {1, 2, 2, 3};
  for (size_t i = 0; i < sim.num_gates(); i++) {
    REQUIRE_MESSAGE(sim.gate(i).level() == gate_lvl_ans[i], "gate level mismatch");
  }
  // check PO
  std::vector<size_t> po_lvl_ans = {4, };
  for (size_t i = 0; i < sim.num_POs(); i++) {
    REQUIRE_MESSAGE(sim.po(i).level() == po_lvl_ans[i], "PO level mismatch");
  }
}

TEST_CASE("Check faults initialization (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();

  // _faults; 
  auto& flts = sim.faults();
  REQUIRE_MESSAGE(flts.size() == 2, "_fault.size() mismatch");
  REQUIRE_MESSAGE(flts[0].gate_with_fault() == gate_tpg_order[2], "_fault init (gate_with_fault) mismatch");
  REQUIRE_MESSAGE(flts[1].gate_with_fault() == gate_tpg_order[4], "_fault init (gate_with_fault) mismatch");
  REQUIRE_MESSAGE(flts[0].gate_SA_fault_val() == -1, "_fault init (gate_SA_fault_val) mismatch"); // -1 -> means SA_1 fault
  REQUIRE_MESSAGE(flts[1].gate_SA_fault_val() == 0, "_fault init (gate_SA_fault_val) mismatch");
}

TEST_CASE("Check patterns initialization (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();

  // _faults; 
  auto& ptns = sim.patterns();
  REQUIRE_MESSAGE(ptns.size() == 1, "_patterns.size() mismatch");
  std::vector<std::vector<size_t>> ptn_val_ans;
  ptn_val_ans.resize(1);  ptn_val_ans[0] = {2, 4, 1};
  
  for (size_t i = 0; i < ptns.size(); i++) {
    for (size_t j = 0; j < (ptns[i].output_value().size()); j++) {
      size_t j_idx = gate_tpg_order[j];
      REQUIRE_MESSAGE((ptns[i].output_value())[j_idx] == ptn_val_ans[i][j_idx], "_patterns value mismatch");
    }
  }
}

TEST_CASE("Check good case simulation (sequential) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

  // good simulation resutls check
  std::vector<size_t> pi_ans = {2,4,1};
  std::vector<size_t> gate_ans = {6,0,1,6};
  std::vector<size_t> po_ans = {6};
  _check_good_case_results(sim, pi_ans, gate_ans, po_ans);
}

TEST_CASE("Check good case simulation (parallel_tf) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  size_t num_threads = 4;
  sim.run(fsim::Simulator::Mode::PARALLEL_TF, num_threads);

  // good simulation resutls check
  std::vector<size_t> pi_ans = {2,4,1};
  std::vector<size_t> gate_ans = {6,0,1,6};
  std::vector<size_t> po_ans = {6};
  _check_good_case_results(sim, pi_ans, gate_ans, po_ans);
}

TEST_CASE("Check good case simulation (parallel_omp) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  size_t num_threads = 4;
  sim.run(fsim::Simulator::Mode::PARALLEL_OMP, num_threads);

  // good simulation resutls check
  std::vector<size_t> pi_ans = {2,4,1};
  std::vector<size_t> gate_ans = {6,0,1,6};
  std::vector<size_t> po_ans = {6};
  _check_good_case_results(sim, pi_ans, gate_ans, po_ans);
}

TEST_CASE("Check fault to pattern table (sequential) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

  // check the fault table
  std::vector<size_t> found_flst_2_ptn_ans = {1, 0, 1, 0};
  std::vector<size_t> ff2p = sim.found_fault_to_pattern();
  for(size_t i = 0; i < ff2p.size(); i++) {
    REQUIRE_MESSAGE(ff2p[i] == found_flst_2_ptn_ans[i], "found fault mismatch");
  }
}

TEST_CASE("Check fault to pattern table (parallel_tf) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  size_t num_threads = 4;
  sim.run(fsim::Simulator::Mode::PARALLEL_TF, num_threads);

  // check the fault table
  std::vector<size_t> found_flst_2_ptn_ans = {1, 0, 1, 0};
  std::vector<size_t> ff2p = sim.found_fault_to_pattern();
  for(size_t i = 0; i < ff2p.size(); i++) {
    REQUIRE_MESSAGE(ff2p[i] == found_flst_2_ptn_ans[i], "found fault mismatch");
  }
}

TEST_CASE("Check fault to pattern table (parallel_omp) (CML)" * doctest::timeout(300)) {
  // CML testcase
  fsim::Simulator sim;
  size_t testcase = 0;
  _read_test_case(sim, testcase);
  
  size_t num_threads = 4;
  sim.run(fsim::Simulator::Mode::PARALLEL_OMP, num_threads);

  // check the fault table
  std::vector<size_t> found_flst_2_ptn_ans = {1, 0, 1, 0};
  std::vector<size_t> ff2p = sim.found_fault_to_pattern();
  for(size_t i = 0; i < ff2p.size(); i++) {
    REQUIRE_MESSAGE(ff2p[i] == found_flst_2_ptn_ans[i], "found fault mismatch");
  }
}


// TEST_CASE("oneGate.1" * doctest::timeout(300)) {  
//   fsim::Simulator sim;
//   size_t testcase = 0;
//   _read_test_case(sim, testcase);

//   std::vector<size_t> ans = {1, 1, 1, 2, 1,};
//   // _check_num_vars(sim, ans);
//   printf("!!! %ld, %ld, %ld, %ld, %ld, \n", sim.num_PIs(),  sim.num_gates(),  sim.num_POs(),  sim.num_wires(),  sim.num_rounds());

// }

// TEST_CASE("oneGate.2" * doctest::timeout(300)) {  
// /**
//  * PI0 -- AND  -- PO
//  * PI1 --| 
// */


// }


// TEST_CASE("linear_chain.1" * doctest::timeout(300)) {
// // linear chain: one linear chain: one PI - (gate (INV))^n - one PO (n = 10)
// /**
//  * PI0 -- INV -- INV -- (... all INV) -- PO
// */


// }

// TEST_CASE("linear_chain.2" * doctest::timeout(300)) {
// /**
//  * PI0 -- AND -- AND -- AND --- AND --- AND --- PO
//  * 
// */

// }


// TEST_CASE("independent_graph.1" * doctest::timeout(300)) {
// // linear chain: 2 independent graphs: (one PI - (gate (INV))^n - one PO)^2 (n = 3)


// }


// TEST_CASE("ring1" * doctest::timeout(300)) {
// // a ring 
// /**
//  *        / --- INV --- INV --- \
//  * PI ---                          AND --- PO
//  *        \ --- INV --- INV --- /
// */

// }


// todo 
// TEST_CASE("X_style" * doctest::timeout(300)) {
// // 階層性電路

// }


// -----------------------------


TEST_CASE("simple_fully_tests.1" * doctest::timeout(300)) {
  // CML testcase
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

  // check var
  REQUIRE_MESSAGE(sim.num_PIs() == 3, "_num_PIs mismatch");
  REQUIRE_MESSAGE(sim.num_gates() == 4, "_num_gates mismatch");
  REQUIRE_MESSAGE(sim.num_POs() == 1, "_num_POs mismatch");
  REQUIRE_MESSAGE(sim.num_wires() == 7, "_num_wires mismatch");
  REQUIRE_MESSAGE(sim.num_rounds() == 1, "_num_rounds mismatch");

  // check topological sort results
  // _gate_idx_to_tpg_order_idx -> this is index transfer from (1) original index (readin from stream) to (2) after topologicacl sort the new index of each gate
  std::vector<size_t> tpg_res = {0, 1, 2, 7, 5, 6, 3, 4};
  std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();
  for (size_t i = 0; i < gate_tpg_order.size(); i++) {
    REQUIRE_MESSAGE(gate_tpg_order[i] == tpg_res[i], "topological sort results order mismatch");
  }

  // check levelization
  // _num_gates_per_level
  std::vector<size_t> num_gate_per_lvl_res = {3, 1, 2, 1, 1};
  std::vector<size_t> num_gate_per_lvl = sim.num_gates_per_level();
  REQUIRE_MESSAGE(num_gate_per_lvl.size() == num_gate_per_lvl_res.size(), "num_gate_per_lvl.size()");
  for (size_t i = 0; i < num_gate_per_lvl.size(); i++) {
    REQUIRE_MESSAGE(num_gate_per_lvl[i] == num_gate_per_lvl_res[i], "num_gates_per_level results order mismatch");
  }  

  // check _PIs, _gates, _POs levelization
  // check PI
  std::vector<size_t> pi_lvl_ans = {0, 0, 0};
  for (size_t i = 0; i < sim.num_PIs(); i++) {
    REQUIRE_MESSAGE(sim.pi(i).level() == pi_lvl_ans[i], "PI level mismatch");
  }
  // check gate
  std::vector<size_t> gate_lvl_ans = {1, 2, 2, 3};
  for (size_t i = 0; i < sim.num_gates(); i++) {
    // auto& gate = sim.gate(i);
    REQUIRE_MESSAGE(sim.gate(i).level() == gate_lvl_ans[i], "gate level mismatch");
  }
  // check PO
  std::vector<size_t> po_lvl_ans = {4, };
  for (size_t i = 0; i < sim.num_POs(); i++) {
    // auto& po = sim.po(i);
    REQUIRE_MESSAGE(sim.po(i).level() == po_lvl_ans[i], "PO level mismatch");
  }

  // check faults init; check patterns init
  // _faults; 
  auto& flts = sim.faults();
  REQUIRE_MESSAGE(flts.size() == 2, "_fault.size() mismatch");
  REQUIRE_MESSAGE(flts[0].gate_with_fault() == gate_tpg_order[2], "_fault init (gate_with_fault) mismatch");
  REQUIRE_MESSAGE(flts[1].gate_with_fault() == gate_tpg_order[4], "_fault init (gate_with_fault) mismatch");
  REQUIRE_MESSAGE(flts[0].gate_SA_fault_val() == -1, "_fault init (gate_SA_fault_val) mismatch"); // -1 -> means SA_1 fault
  REQUIRE_MESSAGE(flts[1].gate_SA_fault_val() == 0, "_fault init (gate_SA_fault_val) mismatch");
  // _patterns;
  auto& ptns = sim.patterns();
  REQUIRE_MESSAGE(ptns.size() == 1, "_patterns.size() mismatch");
  std::vector<std::vector<size_t>> ptn_val_ans;
  ptn_val_ans.resize(1);  ptn_val_ans[0] = {2, 4, 1};
  
  for (size_t i = 0; i < ptns.size(); i++) {
    for (size_t j = 0; j < (ptns[i].output_value().size()); j++) {
      size_t j_idx = gate_tpg_order[j];
      REQUIRE_MESSAGE((ptns[i].output_value())[j_idx] == ptn_val_ans[i][j_idx], "_patterns value mismatch");
    }
  }
  // test gate inputs gate (skip) -> use dump_graph to check this

  // Simulation, check different modes
  for (size_t m = 0; m < 3; m++) {
    if (m == 1) { // mode == OMP
      sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);
    } else if (m == 2) { // mode == OMP
      sim.run(fsim::Simulator::Mode::PARALLEL_TF, 4);
    } else { // mode == OMP
      sim.run(fsim::Simulator::Mode::PARALLEL_OMP, 4);
    }
    // good simulation resutls check 
    auto& g_pi_results = sim.g_pi_results();
    auto& g_gate_results = sim.g_gate_results();
    auto& g_po_results = sim.g_po_results();
    std::vector<size_t> pi_ans = {2,4,1};
    std::vector<size_t> gate_ans = {6,0,1,6};
    std::vector<size_t> po_ans = {6};

    for(size_t piid = 0; piid < g_pi_results.size(); piid++) {
      REQUIRE_MESSAGE(g_pi_results[piid] == pi_ans[piid], "PI result mismatch");
    }
    for(size_t gid = 0; gid < g_gate_results.size(); gid++) {
      REQUIRE_MESSAGE(g_gate_results[gid] == gate_ans[gid], "Gate result mismatch");
    }
    for(size_t poid = 0; poid < g_po_results.size(); poid++) {
      REQUIRE_MESSAGE(g_po_results[poid] == po_ans[poid], "PO result mismatch");
    }

    // check the fault table
    std::vector<size_t> found_flst_2_ptn_ans = {1, 0, 1, 0};
    std::vector<size_t> ff2p = sim.found_fault_to_pattern();
    for(size_t i = 0; i < ff2p.size(); i++) {
      REQUIRE_MESSAGE(ff2p[i] == found_flst_2_ptn_ans[i], "found fault mismatch");
    }  
  }

  // todo: bad simulation resutls check (check the results of fault simulation )
}