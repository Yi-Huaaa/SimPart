/* trunk-ignore-all(clang-format) */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>
#include <fsim/fsim.hpp>

// define test cases
std::vector<std::vector<std::string>> test_cases{
  // CML
  {"\
  3\
  1\
  4\
  7\
  7 0 1 5 1\
  4 0 1 5 2\
  5 0 1 3 1\
  6 0 2 7 2 4 1\
  1 0 1 6 1\
  0 0 1 6 2\
  2 0 1 7 1\
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
  "},
  // oneGate1
  {
  "\
  1\
  1\
  1\
  2\
  0 0 1 2 1\
  2 0 1 1 1\
  0\
  ",
  "\
  2\
  1 1\
  0 0\
  ",
  "\
  2\
  1\
  ",
  },
  // OneGate2
  {
  "\
  2\
  1\
  1\
  3\
  1 0 1 3 1\
  0 0 1 3 2\
  3 0 1 2 1\
  1\
  ",
  "\
  3\
  1 1\
  2 0\
  0 0\
  ",
  "\
  4\
  6 10 \
  "
  },
  // linear_chain1
  {
  "\
  1\
  1\
  10\
  11\
  3 0 1 2 1\
  2 0 1 1 1\
  4 0 1 3 1\
  5 0 1 4 1\
  8 0 1 5 1\
  6 0 1 8 1\
  9 0 1 6 1\
  7 0 1 9 1\
  10 0 1 7 1\
  11 0 1 10 1\
  0 0 1 11 1\
  0 0 0 0 0 0 0 0 0 0\
  ",
  "\
  5\
  0 1\
  9 0\
  4 0\
  7 1\
  3 1\
  ",
  "\
  2\
  1\
  ",
  },
  // linear_chain2
  {
  "\
  6\
  1\
  5\
  11\
  0 0 1 7 1\
  9 0 1 7 2\
  7 0 1 6 1\
  1 0 1 9 1\
  8 0 1 9 2\
  2 0 1 8 1\
  10 0 1 8 2\
  3 0 1 10 1\
  11 0 1 10 2\
  5 0 1 11 1\
  4 0 1 11 2\
  1 1 1 1 1\
  ",
  "\
  5\
  8 0\
  10 0\
  1 0\
  7 1\
  9 1\
  ",
  "\
  4\
  13 9 15 11 13 12\
  ",  
  },  
  // independent_graph1
{
  "\
  6\
  3\
  8\
  14\
  13 0 2 10 2 9 1\
  10 0 1 9 2\
  9 0 1 6 1\
  14 0 1 10 1\
  1 0 1 13 1\
  11 0 1 13 2\
  0 0 1 11 1\
  2 0 1 14 1\
  15 0 1 12 1\
  12 0 1 7 1\
  3 0 1 15 1\
  5 0 1 16 1\
  4 0 1 16 2\
  16 0 1 8 1\
  3 5 0 0 2 0 0 4\
  ",
  "\
  2\
  9 0\
  5 1\
  ",
  "\
  3\
  7 5 3 2 6 1\
  ",
  },
  // ring1
  {
  "\
  2\
  1\
  6\
  8\
  4 0 1 5 1\
  3 0 1 5 2\
  5 0 1 2 1\
  6 0 1 3 1\
  8 0 2 6 1 7 1\
  7 0 1 4 1\
  1 0 1 8 1\
  0 0 1 8 2\
  0 0 3 0 0 3\
  ",
  "\
  2\
  1 0\
  6 0\
  ",
  "\
  3\
  2 6\
  ",
  }
};

#define size_t_bits 64

/** Answers */

// answers check num vars (num_PIs, num_POs, num_gates, num_wires, num_rounds, )
std::vector<std::vector<size_t>> num_vars_ans = {
  {3, 1, 4, 7, 1, }, // simple
  {1, 1, 1, 2, 1,}, // oneGate1
  {2, 1, 1, 3, 1, }, // oneGate2
  {1, 1, 10, 11, 1, }, // linear_chain1
  {6, 1, 5, 11, 1, }, // linear_chain2
  {6, 3, 8, 14, 1, }, // independent_graph1
  {2, 1, 6, 8, 1, }, // ring1
};

// asnwers for Check tpg sort order
std::vector<std::vector<size_t>> tpg_ans = {
  {0, 1, 2, 7, 5, 6, 3, 4}, // simple
  {0, 2, 1,}, // oneGate1
  {0, 1, 3, 2, }, // oneGate2
  {0, 11, 10, 9, 8, 7, 5, 3, 6, 4, 2, 1, }, // linear_chain1
  {0, 1, 2, 3, 4, 5, 11, 10, 8, 9, 7, 6, }, // linear_chain2
  {0, 1, 2, 3, 4, 5, 14, 15, 16, 13, 12, 6, 11, 10, 7, 8, 9, }, // independent_graph1
  {0, 1, 8, 5, 6, 7, 3, 4, 2, }, // ring1
};
// answers for check levelization
std::vector<std::vector<size_t>> num_gate_per_lvl_ans = {
  {3, 1, 2, 1, 1, }, // simple
  {1, 1, 1, },  // oneGate1
  {2, 1, 1, },  // oneGate2
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, }, // linear_chain1
  {6, 1, 1, 1, 1, 1, 1, }, // linear_chain2
  {6, 4, 2, 1, 1, 3, }, // independent_graph1
  {2, 1, 2, 2, 1, 1, }, // ring1
};
std::vector<std::vector<size_t>> pi_lvl_ans = {
  {0, 0, 0, }, // simple
  {0, }, // oneGate1
  {0, 0, }, // oneGate2
  {0, }, // linear_chain1
  {0, 0, 0, 0, 0, 0, }, // linear_chain2
  {0, 0, 0, 0, 0, 0, }, // independent_graph1
  {0, 0, }, // ring1
};
std::vector<std::vector<size_t>> gate_lvl_ans = {
  {1, 2, 2, 3, }, // simple
  {1, }, // oneGate1
  {1, }, // oneGate2
  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, }, // linear_chain1
  {1, 2, 3, 4, 5, }, // linear_chain2
  {1, 1, 1, 1, 2, 2, 3, 4, }, // independent_graph1
  {1, 2, 2, 3, 3, 4, }, // ring1
};
std::vector<std::vector<size_t>> po_lvl_ans = {
  {4,}, // simple
  {2,}, // oneGate1
  {2, }, // oneGate2
  {11, }, // linear_chain1
  {6, }, // linear_chain2
  {5, 5, 5, }, // independent_graph1
  {5, }, // ring1  
};
// answers for faults initialization
std::vector<size_t> fault_size_ans = {
  2, // simple
  2, // oneGate1
  3, // oneGate2
  5, // linear_chain1
  5, // linear_chain2
  2, // independent_graph1
  2, // ring1  
  };
std::vector<std::vector<size_t>> fault_ans = {
  {2, 1, 4, 0, }, // simple
  {1, 1, 0, 0, }, // oneGate1
  {1, 1, 2, 0, 0, 0, }, // oneGate2
  {0, 1, 9, 0, 4, 0, 7, 1, 3, 1, }, // linear_chain1
  {8, 0, 10, 0, 1, 0, 7, 1, 9, 1, }, // linear_chain2
  {9, 0, 5, 1, }, // independent_graph1
  {1, 0, 6, 0,}, // ring1  
};

// answers for patterns initialization
std::vector<size_t> ptn_size_ans = {
  3, // simple
  1, // oneGate1
  2, // oneGate2
  1, // linear_chain1
  6, // linear_chain2
  3, // independent_graph1
  3, // ring1  
};
std::vector<std::vector<size_t>> ptn_val_ans = {
  {2, 4, 1, }, // simple
  {1, }, // oneGate1
  {6, 10, }, // oneGate2
  {1, }, // linear_chain1
  {13, 9, 15, 11, 13, 12, }, // linear_chain2
  {7, 5, 3, 2, 6, 1, }, // independent_graph1
  {2, 6, }, // ring1  
};
// answers for simulation outputs of PIs, gates, POs
std::vector<std::vector<size_t>> pi_ans = {
  {2, 4, 1, }, // simple
  {1, }, // oneGate1
  {6, 10, }, // oneGate2
  {1, }, // linear_chain1
  {13, 9, 15, 11, 13, 12, }, // linear_chain2
  {7, 5, 3, 2, 6, 1, }, // independent_graph1
  {2, 6, }, // ring1  
};
std::vector<std::vector<size_t>> gate_ans = {
  {6, 0, 1, 6}, // simple
  {2,}, // oneGate1
  {2, }, // oneGate2
  {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, }, // linear_chain1
  {12, 8, 8, 8, 8,}, // linear_chain2
  {0, 4, 5, 7, 5, 2, 2, 7, }, // independent_graph1
  {4, 3, 3, 4, 4, 0, }, // ring1  
};
std::vector<std::vector<size_t>> po_ans = {
  {6, }, // simple
  {2, }, // oneGate1
  {2, }, // oneGate2
  {1, }, // linear_chain1
  {8, }, // linear_chain2
  {7, 2, 7,}, // independent_graph1
  {0, }, // ring1  
};
// answers for found_fault_2_pattern_tables
std::vector<std::vector<size_t>> found_flst_2_ptn_ans = {
  {1, 0, 1, 0}, // simple
  {1, 0, 1, 0}, // oneGate1
  {1, 0, 1, 0, 1, 0, }, // oneGate2
  {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, }, // linear_chain1
  {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, }, // linear_chain2
  {1, 0, 1, 0, }, // independent_graph1
  {0, 0, 1, 0, }, // ring1  
};

/** helper functions */
void _read_test_case(fsim::Simulator &sim, size_t test_case_id) {
  std::istringstream ickt(test_cases[test_case_id][0]);
  std::istringstream iflst(test_cases[test_case_id][1]);
  std::istringstream iptn(test_cases[test_case_id][2]);
  sim.read(ickt, iflst, iptn);
}

void _check_num_vars(fsim::Simulator &sim, std::vector<size_t> &num_vars_ans_t) {
  // check var
  REQUIRE_MESSAGE(sim.num_PIs() == num_vars_ans_t[0], "_num_PIs mismatch");
  REQUIRE_MESSAGE(sim.num_POs() == num_vars_ans_t[1], "_num_POs mismatch");
  REQUIRE_MESSAGE(sim.num_gates() == num_vars_ans_t[2], "_num_gates mismatch");
  REQUIRE_MESSAGE(sim.num_wires() == num_vars_ans_t[3], "_num_wires mismatch");
  REQUIRE_MESSAGE(sim.num_rounds() == num_vars_ans_t[4], "_num_rounds mismatch");
}

void _check_good_case_results(fsim::Simulator &sim,
                              std::vector<size_t> &pi_ans_t,
                              std::vector<size_t> &gate_ans_t,
                              std::vector<size_t> &po_ans_t) {
  auto &g_pi_results = sim.g_pi_results();
  auto &g_gate_results = sim.g_gate_results();
  auto &g_po_results = sim.g_po_results();

  for (size_t piid = 0; piid < g_pi_results.size(); piid++) {
    REQUIRE_MESSAGE(g_pi_results[piid] == pi_ans_t[piid], "PI result mismatch");
  }
  for (size_t gid = 0; gid < g_gate_results.size(); gid++) {
    REQUIRE_MESSAGE(g_gate_results[gid] == gate_ans_t[gid],
                    "Gate result mismatch");
  }
  for (size_t poid = 0; poid < g_po_results.size(); poid++) {
    REQUIRE_MESSAGE(g_po_results[poid] == po_ans_t[poid], "PO result mismatch");
  }
}

void _check_tpg_sort_order(std::vector<size_t> res, std::vector<size_t> ans) {
  for (size_t i = 0; i < res.size(); i++) {
    REQUIRE_MESSAGE(res[i] == ans[i],
                    "topological sort results order mismatch");
  }
}

void _check_num_gate_per_lvl_ans(std::vector<size_t> res,
                                 std::vector<size_t> num_vars_ans_t) {
  for (size_t i = 0; i < res.size(); i++) {
    REQUIRE_MESSAGE(res[i] == num_vars_ans_t[i],
                    "num_gates_per_level results order mismatch");
  }
}

void _check_pi_gate_po_level_ans(fsim::Simulator &sim,
                                 std::vector<size_t> pi_lvl_ans_t,
                                 std::vector<size_t> gate_lvl_ans_t,
                                 std::vector<size_t> po_lvl_ans_t) {
  // check PI
  for (size_t i = 0; i < sim.num_PIs(); i++) {
    REQUIRE_MESSAGE(sim.pi(i).level() == pi_lvl_ans_t[i], "PI level mismatch");
  }
  // check gate
  for (size_t i = 0; i < sim.num_gates(); i++) {
    REQUIRE_MESSAGE(sim.gate(i).level() == gate_lvl_ans_t[i],
                    "gate level mismatch");
  }
  // check PO
  for (size_t i = 0; i < sim.num_POs(); i++) {
    REQUIRE_MESSAGE(sim.po(i).level() == po_lvl_ans_t[i], "PO level mismatch");
  }
}

void _check_faults(fsim::Simulator &sim, std::vector<size_t> gate_tpg_order,
                   size_t fault_size_ans_t, std::vector<size_t> fault_ans_t) {
  auto &flts = sim.faults();
  REQUIRE_MESSAGE(flts.size() == fault_size_ans_t, "_fault.size() mismatch");

  for (size_t i = 0; i < flts.size(); i++) {
    size_t gate_with_fault_ans = gate_tpg_order[fault_ans_t[2 * i]];
    size_t gate_SA_fault_val_ans = (fault_ans_t[2 * i + 1] == 1) ? (-1) : (0);
    REQUIRE_MESSAGE(flts[i].gate_with_fault() == gate_with_fault_ans,
                    "_fault init (gate_with_fault) mismatch");
    REQUIRE_MESSAGE(
        flts[i].gate_SA_fault_val() == gate_SA_fault_val_ans,
        "_fault init (gate_SA_fault_val) mismatch"); // -1 -> means SA_1 fault
  }
}

void _check_patterns(fsim::Simulator &sim, std::vector<size_t> gate_tpg_order,
                     size_t ptn_size_ans_t, std::vector<size_t> ptn_val_ans_t) {
  auto &ptns = sim.patterns();
  REQUIRE_MESSAGE(ptns.size() == ptn_size_ans_t, "_patterns.size() mismatch");

  for (size_t i = 0; i < ptns.size(); i++) {
    for (size_t j = 0; j < (ptns[i].output_value().size()); j++) {
      size_t j_idx = gate_tpg_order[j];
      REQUIRE_MESSAGE((ptns[i].output_value())[j_idx] == ptn_val_ans_t[j_idx],
                      "_patterns value mismatch");
    }
  }
}

void _check_found_fault_2_pattern(
    fsim::Simulator &sim, std::vector<size_t> found_fault_2_pattern_ans) {

  std::vector<size_t> found_fault_2_pattern_res = sim.found_fault_to_pattern();
  for (size_t i = 0; i < found_fault_2_pattern_res.size(); i++) {
    REQUIRE_MESSAGE(found_fault_2_pattern_res[i] ==
                        found_fault_2_pattern_ans[i],
                    "found fault mismatch");
  }
}

/** TEST_CASE, now: (CML, OneGate) */
TEST_CASE("Check num vars (All testcases)" * doctest::timeout(300)) {
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);
    _check_num_vars(sim, num_vars_ans[t]);
  }
}

TEST_CASE("Check tpg sort order (All testcases)" * doctest::timeout(300)) {
  // check topological sort results
  // _gate_idx_to_tpg_order_idx -> this is index transfer from (1) original
  // index (readin from stream) to (2) after topologicacl sort the new index of
  // each gate

  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();
    _check_tpg_sort_order(gate_tpg_order, tpg_ans[t]);
  }
}

TEST_CASE("Check levelization (All testcases)" * doctest::timeout(300)) {
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    // _num_gates_per_level
    std::vector<size_t> num_gate_per_lvl = sim.num_gates_per_level();
    REQUIRE_MESSAGE(num_gate_per_lvl.size() == num_gate_per_lvl_ans[t].size(),
                    "num_gate_per_lvl.size()");
    _check_num_gate_per_lvl_ans(num_gate_per_lvl, num_gate_per_lvl_ans[t]);

    // check _PIs, _gates, _POs levelization
    _check_pi_gate_po_level_ans(sim, pi_lvl_ans[t], gate_lvl_ans[t],
                                po_lvl_ans[t]);
  }
}

TEST_CASE("Check faults initialization (All testcases)" * doctest::timeout(300)) {
  // _faults;
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();

    _check_faults(sim, gate_tpg_order, fault_size_ans[t], fault_ans[t]);
  }
}

TEST_CASE("Check patterns initialization (All testcases)" * doctest::timeout(300)) {
  for (size_t i = 0; i < ptn_size_ans.size(); i++) {
    ptn_size_ans[i] = ((ptn_size_ans[i] + size_t_bits - 1) / size_t_bits);
  }

  // _patterns;
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    std::vector<size_t> gate_tpg_order = sim.gate_idx_to_tpg_order_idx();

    _check_patterns(sim, gate_tpg_order, ptn_size_ans[t], ptn_val_ans[t]);
  }
}

TEST_CASE("Check good case simulation (sequential) (All testcases)" * doctest::timeout(300)) {
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    sim.run(fsim::Simulator::Mode::SEQUENTIAL, 1);

    // good simulation resutls check
    _check_good_case_results(sim, pi_ans[t], gate_ans[t], po_ans[t]);
  }
}

TEST_CASE("Check good case simulation (parallel_tf) (All testcases)" * doctest::timeout(300)) {
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    size_t num_threads = 4;
    sim.run(fsim::Simulator::Mode::PARALLEL_TF, num_threads);

    // good simulation resutls check
    _check_good_case_results(sim, pi_ans[t], gate_ans[t], po_ans[t]);
  }
}

TEST_CASE("Check good case simulation (parallel_omp) (All testcases)" * doctest::timeout(300)) {
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    size_t num_threads = 4;
    sim.run(fsim::Simulator::Mode::PARALLEL_OMP, num_threads);

    // good simulation resutls check
    _check_good_case_results(sim, pi_ans[t], gate_ans[t], po_ans[t]);
  }
}

TEST_CASE("Check fault to pattern table (sequential) (All testcases)" * doctest::timeout(300)) {
  size_t num_threads = 1;
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    sim.run(fsim::Simulator::Mode::SEQUENTIAL, num_threads);

    // check the fault table
    _check_found_fault_2_pattern(sim, found_flst_2_ptn_ans[t]);
  }
}

TEST_CASE("Check fault to pattern table (parallel_tf) (All testcases)" *
          doctest::timeout(300)) {
  size_t num_threads = 4;
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    sim.run(fsim::Simulator::Mode::PARALLEL_TF, num_threads);

    // check the fault table
    _check_found_fault_2_pattern(sim, found_flst_2_ptn_ans[t]);
  }
}

TEST_CASE("Check fault to pattern table (parallel_omp) (All testcases)" *
          doctest::timeout(300)) {
  size_t num_threads = 4;
  for (size_t t = 0; t < test_cases.size(); t++) {
    fsim::Simulator sim;
    _read_test_case(sim, t);

    sim.run(fsim::Simulator::Mode::PARALLEL_OMP, num_threads);

    // check the fault table
    _check_found_fault_2_pattern(sim, found_flst_2_ptn_ans[t]);
  }
}






// todo
// TEST_CASE("X_style" * doctest::timeout(300)) {
// // 階層性電路

// }

// todo: bad simulation resutls check (check the results of fault simulation )
