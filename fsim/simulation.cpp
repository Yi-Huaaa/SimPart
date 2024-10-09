#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

#include "simulation.hpp"

void fsim::Simulator::read(const std::string &ckt_path,
                           const std::string &flst_path,
                           const std::string &ptn_path) {
  /* Ref:
   * https://stackoverflow.com/questions/38950008/using-string-literals-without-using-namespace-std*/
  // using namespace std::literals::string_literals;
  // using namespace std::literals;
  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path);

  // TODO (study): study operator bool (cpp reference)
  if (!ckt) {
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  }
  if (!flst) {
    throw std::runtime_error("cannot open fault file "s + flst_path);
  }
  if (!ptn) {
    throw std::runtime_error("cannot open pattern file "s + ptn_path);
  }

  read(ckt, flst, ptn);
}

void fsim::Simulator::read(std::istream &ckt, std::istream &flst,
                           std::istream &ptn) {
  // gate
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires;
#ifdef FSIM_DEBUG_PRINT_GRAPH
  std::cout << "Read and construct ckt" << std::endl;
  std::cout << _num_PIs << ", " << _num_POs << ", " << _num_inner_gates << ", "
            << _num_wires << std::endl;
#endif
  _read_graph(ckt);

  // fault
  flst >> _num_fault;
#ifdef FSIM_DEBUG_PRINT_FAULTS
  std::cout << "Read and construct flst" << std::endl;
  std::cout << _num_fault << std::endl;
#endif
  _read_fault(flst);

  // pattern
  ptn >> _num_pattern;
#ifdef FSIM_DEBUG_PRINT_PATTERNS
  std::cout << "Read and construct ptn" << std::endl;
  std::cout << _num_pattern << std::endl;
#endif
  _read_pattern(ptn);
}

void fsim::Simulator::run(Mode mode, const size_t num_threads) {

  switch (mode) {
  case Mode::SEQUENTIAL:
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    std::cout << "run _run_sequential\n";
#endif
    _run_sequential();
    break;
  case Mode::PARALLEL_TF:
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    std::cout << "run _run_parallel_tf\n";
#endif
    _run_parallel_tf();
    break;
  case Mode::PARALLEL_OMP:
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    std::cout << "run _run_parallel_omp\n";
#endif
    _run_parallel_omp(num_threads);
    break;
  }
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
  size_t m = static_cast<size_t>(mode);
  std::cout << "Mode: " << m << std::endl;
  print_found_fault_to_pattern(_found_fault_to_pattern);
#endif
}

void fsim::Simulator::_shift_to_correct_answer(std::vector<size_t> &results,
                                               const size_t bits,
                                               const size_t num_shift_gates) {

  for (size_t i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
}

void fsim::Simulator::_run_sequential() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_sequential();" << '\n';
#endif

  // Simulation
  _g_pi_results.resize(_num_PIs);
  _g_gate_results.resize(_num_inner_gates);
  _g_po_results.resize(_num_POs);
  _b_pi_results.resize(_num_PIs);
  _b_gate_results.resize(_num_inner_gates);
  _b_po_results.resize(_num_POs);
  _found_fault_to_pattern.resize(2 * _faults.size());

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "Run Good case" << std::endl;
#endif
    _run_good_case(_patterns[rd], num_testcases_this_round);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _faults.size(); j++) {
      _run_bad_case(_faults[j], _patterns[rd], num_testcases_this_round);

      size_t found_fault = 0;
      for (size_t i = 0; i < _num_POs; i++) {
        if (_g_po_results[i] != _b_po_results[i]) {
          found_fault = 1;
          break;
        }
      }
      // Record whether fault can be found
      _found_fault_to_pattern[2 * j] = found_fault;
      // Record which pattern found the fault
      _found_fault_to_pattern[2 * j + 1] = rd;
    }
  }
}

void fsim::Simulator::_run_good_case(const Pattern pattern, const size_t bits) {
  // Simulation
  for (size_t i = 0; i < _num_PIs; i++) {
    _run_gate(_PIs[i], pattern, i, 0, 0);
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _run_gate(_gates[i], pattern, i, 0,
              0); // here: i is redundant, no need to be worry about the
                  // correctness of its val
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _run_gate(_POs[i], pattern, i, 0, 0);
  }

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _g_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _g_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _g_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_g_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_g_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_g_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls seq:" << std::endl;
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void fsim::Simulator::_run_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, const Pattern pattern,
                                    const size_t bits) {
  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);

  // fault simulation
  for (size_t i = 0; i < _num_PIs; i++) {
    size_t SA_fault = ((i + pi_idx_accum) == wrong_gate);
    _run_gate(_PIs[i], pattern, i, SA_fault, fault_val);
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    size_t SA_fault = ((i + gate_idx_accum) == wrong_gate);
    _run_gate(_gates[i], pattern, i, SA_fault, fault_val);
  }
  for (size_t i = 0; i < _num_POs; i++) {
    size_t SA_fault = ((i + po_idx_accum) == wrong_gate);
    _run_gate(_POs[i], pattern, i, SA_fault, fault_val);
  }

  // Duplicated the answer of BAD_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _b_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _b_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _b_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_b_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_b_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_b_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "BAD resutls seq:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void fsim::Simulator::_run_parallel_tf() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel taskflow();" << '\n';
#endif

  // Simulation
  _g_pi_results.resize(_num_PIs);
  _g_gate_results.resize(_num_inner_gates);
  _g_po_results.resize(_num_POs);
  _b_pi_results.resize(_num_PIs);
  _b_gate_results.resize(_num_inner_gates);
  _b_po_results.resize(_num_POs);
  _found_fault_to_pattern.resize(2 * _faults.size());
  _which_fault_pattern.resize(2); // [0]: which_fault, [1]: which_pattern

  // todo: make the emplace of good cases here
  // taskflow task assign: emplace
  tf::Taskflow taskflow_bad_cases("_bad_case");
  _emplace_bad_case_tf(taskflow_bad_cases);

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "TF: Run Good case" << std::endl;
#endif
    //
    _run_good_case_tf(_patterns[rd], num_testcases_this_round);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _faults.size(); j++) {
      _which_fault_pattern[0] = j;  // _which_fault
      _which_fault_pattern[1] = rd; // _which_pattern
      // run
      _run_bad_case_tf(taskflow_bad_cases, 
                                      num_testcases_this_round);

      size_t found_fault = 0;
      for (size_t i = 0; i < _num_POs; i++) {
        if (_g_po_results[i] != _b_po_results[i]) {
          found_fault = 1;
          break;
        }
      }
      // Record whether fault can be found
      _found_fault_to_pattern[2 * j] = found_fault;
      // Record which pattern found the fault
      _found_fault_to_pattern[2 * j + 1] = rd;
    }
  }
}

void fsim::Simulator::_run_good_case_tf(const Pattern pattern,
                                        const size_t bits) {
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);
  // taskflow task assign
  // TODO: executor should be centralized - best practice
  tf::Taskflow taskflow("_good_case");

  // Simulation
  std::vector<tf::Task> tasks;
  tasks.resize((_num_PIs + _num_inner_gates + _num_POs));

  for (size_t i = 0; i < _PIs.size(); i++) {
    tasks[i + pi_idx_accum] = taskflow.emplace(
        [this, i, &pattern]() { _run_gate(_PIs[i], pattern, i, 0, 0); });
  }
  for (size_t i = 0; i < _gates.size(); i++) {
    tasks[i + gate_idx_accum] = taskflow.emplace(
        [this, i, &pattern]() { _run_gate(_gates[i], pattern, i, 0, 0); });
  }
  for (size_t i = 0; i < _POs.size(); i++) {
    tasks[i + po_idx_accum] = taskflow.emplace(
        [this, i, &pattern]() { _run_gate(_POs[i], pattern, i, 0, 0); });
  }

  for (size_t i = 0; i < _gates.size(); i++) {              // now_gate
    for (size_t j = 0; j < _gates[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _gates[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      size_t pre_idx = pre_gate->_idx;
      tasks[i + gate_idx_accum].succeed(tasks[pre_idx]);
    }
  }

  for (size_t i = 0; i < _POs.size(); i++) {              // now_gate
    for (size_t j = 0; j < _POs[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _POs[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      size_t pre_idx = pre_gate->_idx;
      tasks[i + po_idx_accum].succeed(tasks[pre_idx]);
    }
  }

  _executor.run(taskflow).wait();

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _g_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _g_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _g_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_g_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_g_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_g_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls:" << std::endl;
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void fsim::Simulator::_emplace_bad_case_tf(tf::Taskflow &taskflow) {
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);

#ifdef TF_parallel_for
  // fault simulation
  std::vector<tf::Task> tasks;
  tasks.resize(_num_gates_per_level.size());
  size_t tasks_idx = 0;
  int N = 0;

  // #pragma omp parallel for
  tasks[tasks_idx] =
      taskflow.for_each_index(0, N = _num_PIs, 1, [this, pi_idx_accum](int i) {
        size_t SA_fault = ((i + pi_idx_accum) ==
                           _faults[_which_fault_pattern[0]]._gate_with_fault);
        _run_gate(_PIs[i], _patterns[_which_fault_pattern[1]], i, SA_fault,
                  _faults[_which_fault_pattern[0]]._gate_SA_fault_val);
      });
  tasks_idx++;

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  size_t accum = 0;
  for (size_t i = 1; i < (_num_gates_per_level.size() - 1); i++) {
    tasks[tasks_idx] = taskflow.for_each_index(
        0, N = _num_gates_per_level[i], 1,
        [this, gate_idx_accum, accum](int j) {
          size_t idx = accum + j;
          size_t SA_fault = ((idx + gate_idx_accum) ==
                             _faults[_which_fault_pattern[0]]._gate_with_fault);
          _run_gate(_gates[idx], _patterns[_which_fault_pattern[1]], idx,
                    SA_fault,
                    _faults[_which_fault_pattern[0]]
                        ._gate_SA_fault_val); // here: idx is redundant, no need
                                              // to be worry about the
                                              // correctness of its val
        });
    tasks_idx++;
    accum += _num_gates_per_level[i];
  }

  tasks[tasks_idx] =
      taskflow.for_each_index(0, N = _num_POs, 1, [this, po_idx_accum](int i) {
        size_t SA_fault = ((i + po_idx_accum) ==
                           _faults[_which_fault_pattern[0]]._gate_with_fault);
        _run_gate(
            _POs[i], _patterns[_which_fault_pattern[1]], i, SA_fault,
            _faults[_which_fault_pattern[0]]
                ._gate_SA_fault_val); // here: i is redundant, no need to be
                                      // worry about the correctness of its val
      });
  tasks_idx++;

  // tf: succeed
  for (size_t i = 0; i < (tasks.size() - 1); i++) {
    tasks[i + 1].succeed(tasks[i]);
  }

#else

  std::vector<tf::Task> tasks;
  tasks.resize((_num_PIs + _num_inner_gates + _num_POs));

  for (size_t i = 0; i < _PIs.size(); i++) {
    tasks[i + pi_idx_accum] = taskflow.emplace([this, i, pi_idx_accum]() {
      size_t SA_fault = ((i + pi_idx_accum) ==
                         _faults[_which_fault_pattern[0]]._gate_with_fault);
      _run_gate(_PIs[i], _patterns[_which_fault_pattern[1]], i, SA_fault,
                _faults[_which_fault_pattern[0]]._gate_SA_fault_val);
    });
  }

  for (size_t i = 0; i < _gates.size(); i++) {
    tasks[i + gate_idx_accum] = taskflow.emplace([this, i, gate_idx_accum]() {
      size_t SA_fault = ((i + gate_idx_accum) ==
                         _faults[_which_fault_pattern[0]]._gate_with_fault);
      _run_gate(_gates[i], _patterns[_which_fault_pattern[1]], i, SA_fault,
                _faults[_which_fault_pattern[0]]._gate_SA_fault_val);
    });
  }

  for (size_t i = 0; i < _POs.size(); i++) {
    tasks[i + po_idx_accum] = taskflow.emplace([this, i, po_idx_accum]() {
      size_t SA_fault = ((i + po_idx_accum) ==
                         _faults[_which_fault_pattern[0]]._gate_with_fault);
      _run_gate(_POs[i], _patterns[_which_fault_pattern[1]], i, SA_fault,
                _faults[_which_fault_pattern[0]]._gate_SA_fault_val);
    });
  }

  for (size_t i = 0; i < _gates.size(); i++) {              // now_gate
    for (size_t j = 0; j < _gates[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _gates[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      size_t pre_idx = pre_gate->_idx;
      tasks[i + gate_idx_accum].succeed(tasks[pre_idx]);
    }
  }

  for (size_t i = 0; i < _POs.size(); i++) {              // now_gate
    for (size_t j = 0; j < _POs[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _POs[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      size_t pre_idx = pre_gate->_idx;
      tasks[i + po_idx_accum].succeed(tasks[pre_idx]);
    }
  }

#endif
}

void fsim::Simulator::_run_bad_case_tf(tf::Taskflow &taskflow,
                                       const size_t bits) {

  // emplace only need to be done once outside the for loop of faults

  // run
  // printf("fault = %ld, pattern = %ld\n", _which_fault_pattern[0],
  // _which_fault_pattern[1]);
  _executor.run(taskflow).wait();

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _b_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _b_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _b_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_b_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_b_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_b_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "BAD resutls TF:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void fsim::Simulator::_run_parallel_omp(const size_t num_threads) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel omp();" << '\n';
#endif

  // Simulation
  _g_pi_results.resize(_num_PIs);
  _g_gate_results.resize(_num_inner_gates);
  _g_po_results.resize(_num_POs);
  _b_pi_results.resize(_num_PIs);
  _b_gate_results.resize(_num_inner_gates);
  _b_po_results.resize(_num_POs);
  _found_fault_to_pattern.resize(2 * _faults.size());

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "OMP: Run Good case" << std::endl;
#endif
    _run_good_case_omp(_patterns[rd], num_testcases_this_round, num_threads);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _faults.size(); j++) {
      _run_bad_case_omp(_faults[j], _patterns[rd], num_testcases_this_round,
                        num_threads);

      size_t found_fault = 0;
      for (size_t i = 0; i < _num_POs; i++) {
        if (_g_po_results[i] != _b_po_results[i]) {
          found_fault = 1;
          break;
        }
      }
      // Record whether fault can be found
      _found_fault_to_pattern[2 * j] = found_fault;
      // Record which pattern found the fault
      _found_fault_to_pattern[2 * j + 1] = rd;
    }
  }
}

void fsim::Simulator::_run_good_case_omp(const Pattern pattern,
                                         const size_t bits,
                                         const size_t num_threads) {

  omp_set_num_threads(num_threads);

// Simulation
#pragma omp parallel for
  for (size_t i = 0; i < _num_PIs; i++) {
    _run_gate(_PIs[i], pattern, i, 0, 0);
  }

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  size_t accum = 0;
  for (size_t i = 1; i < (_num_gates_per_level.size() - 1); i++) {
#pragma omp parallel for
    for (size_t j = 0; j < _num_gates_per_level[i]; j++) {
      size_t idx = accum + j;
      _run_gate(_gates[idx], pattern, idx, 0,
                0); // here: idx is redundant, no need to be worry about the
                    // correctness of its val
      // printf("debug: omp_get_thread_num = %d (#threads = %ld, i = %ld, j =
      // %ld)\n", omp_get_thread_num(), omp_get_num_threads(), i, j); // debug_
    }
    accum += _num_gates_per_level[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < _num_POs; i++) {
    _run_gate(_POs[i], pattern, i, 0,
              0); // here: i is redundant, no need to be worry about the
                  // correctness of its val
  }

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _g_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _g_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _g_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_g_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_g_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_g_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls omp:" << std::endl;
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void fsim::Simulator::_run_bad_case_omp(const Fault<FAULT_INDEX_TYPE> &fault,
                                        const Pattern pattern,
                                        const size_t bits,
                                        const size_t num_threads) {
  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);

  omp_set_num_threads(num_threads);
// fault simulation
#pragma omp parallel for
  for (size_t i = 0; i < _num_PIs; i++) {
    size_t SA_fault = ((i + pi_idx_accum) == wrong_gate);
    _run_gate(_PIs[i], pattern, i, SA_fault, fault_val);
  }

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  size_t accum = 0;
  for (size_t i = 1; i < (_num_gates_per_level.size() - 1); i++) {
#pragma omp parallel for
    for (size_t j = 0; j < _num_gates_per_level[i]; j++) {
      size_t idx = accum + j;
      size_t SA_fault = ((idx + gate_idx_accum) == wrong_gate);
      _run_gate(_gates[idx], pattern, idx, SA_fault,
                fault_val); // here: idx is redundant, no need to be worry about
                            // the correctness of its val
    }
    accum += _num_gates_per_level[i];
  }

#pragma omp parallel for
  for (size_t i = 0; i < _num_POs; i++) {
    size_t SA_fault = ((i + po_idx_accum) == wrong_gate);
    _run_gate(_POs[i], pattern, i, SA_fault,
              fault_val); // here: i is redundant, no need to be worry about the
                          // correctness of its val
  }

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _b_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _b_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _b_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_b_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_b_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_b_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "BAD resutls omp:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

// Construct functions
void fsim::Simulator::_read_graph(std::istream &ckt) {
  size_t sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  // for topological sort
  std::vector<std::vector<size_t>> adj;
  adj.resize(sum_pi_gates_pos);
  std::vector<size_t> indegree;
  indegree.resize(sum_pi_gates_pos);

  // map table for recording gate input order
  // record for each to_gate, its from_gates's are which order (#_inputs, A/A1,
  // B/A2, A3/S, A4)
  std::vector<std::vector<size_t>> gate_input_order_table;
  gate_input_order_table.resize(sum_pi_gates_pos);
  for (size_t i = 0; i < sum_pi_gates_pos; i++) {
    gate_input_order_table[i].resize(5);
  }

  /** Levelized the input gates before constructing the circuit graph */
  // read file, construct adjacent list, construct indegree array
  for (size_t i = 0; i < _num_wires; i++) {
    size_t gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;
    /**
     * from parser.py: "Z": 0, "ZN": 0, "A": 1, "A1": 1, "B": 2, "A2": 2, "A3":
     * 3, "A4": 4, "S": 5
     * * pin_Y must be 0;
     * * pin_tmp must be:
     * "A": 1, "A1": 1, "B": 2, "A2": 2, "A3": 3, "A4": 4, "S": 5
     * idx:
     * A/A1, B/A2, A3, A4, S
     *    1,    2,  3,  4, 5
     * re-idx:
     *    1,    2,  3,  4, 3
     */

    for (size_t j = 0; j < num_post_gates; j++) {
      size_t gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp); // for MUX
      adj[gate_0].push_back(gate_tmp);
      indegree[gate_tmp]++;

      gate_input_order_table[gate_tmp][pin_tmp] = gate_0;
      gate_input_order_table[gate_tmp]
                            [0]++; // calculate how many inputs for this to_gate
    }
  }

#ifdef FSIM_DEBUG_PRINT_GRAPH
  // copy `indegree` vector since we need its original value later
  std::vector<size_t> indegree_orig;
  indegree_orig = indegree;
#endif

  // Kahn's algorithm, topological sort
  std::queue<size_t> source;
  std::vector<size_t> order;

  // since only PIs will be indegree == 0
  for (size_t i = 0; i < _num_PIs; i++) {
    source.push(i);
  }

  while (!source.empty()) {
    size_t first = source.front();
    source.pop(); // queue pop front element
    order.push_back(first);
    for (size_t i = 0; i < adj[first].size(); i++) {
      indegree[adj[first][i]]--;
      if (indegree[adj[first][i]] == 0) {
        source.push(adj[first][i]);
      }
    }
  }

  // make vector of order's order becomes [PIs, gates, POs]
  // remove the PO in the order
  order.erase(std::remove_if(order.begin(), order.end(),
                             [this](const size_t &x) {
                               return ((x >= _num_PIs) &&
                                       (x < (_num_PIs + _num_POs)));
                             }),
              order.end());

  for (size_t i = 0; i < _num_POs; i++) {
    order.push_back(_num_PIs + i); // push_back PO's index
  }

#ifdef FSIM_DEBUG_PRINT_GRAPH
  print_topological(adj, indegree_orig, order);
  print_gate_input_order_table(gate_input_order_table);
#endif

  /** Constructing circuit graph */
  // read gate type array
  std::vector<GateType> gate_type;
  gate_type.resize(sum_pi_gates_pos);
  for (size_t i = 0; i < _num_PIs; i++) {
    gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs + _num_POs); i < (sum_pi_gates_pos); i++) {
    size_t t; // type
    ckt >> t;
    gate_type[i] = static_cast<GateType>(t);
  }

  // mapping gate_idx to gate_order_index
  /**
   * CML example:
   * order[3] = 6
   * _gate_idx_to_tpg_order_idx[6] = 3
   * _gate_idx_to_tpg_order_idx: 0, 1, 2, 7, 5, 6, 3, 4
   * */
  _gate_idx_to_tpg_order_idx.resize(sum_pi_gates_pos);
  for (size_t i = 0; i < order.size(); i++) {
    _gate_idx_to_tpg_order_idx[order[i]] = i;
  }

  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);
  // PIs are stored in the order specified by the `order` array
  _PIs.resize(_num_PIs);
  for (size_t i = 0; i < _num_PIs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + pi_idx_accum];
    _PIs[i]._idx = i + pi_idx_accum;
    _PIs[i]._type = gate_type[gate_id];
    _PIs[i]._level = default_level;
    _PIs[i]._output_value = default_output_value;
  }

  // gates are stored in the order specified by the `order` array
  _gates.resize(_num_inner_gates);
  for (size_t i = 0; i < _num_inner_gates; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + gate_idx_accum];
    _gates[i]._idx = i + gate_idx_accum;
    _gates[i]._type = gate_type[gate_id];
    _gates[i]._level = default_level;
    _gates[i]._output_value = default_output_value;
  }

  // POs are stored in the order specified by the `order` array
  _POs.resize(_num_POs);
  for (size_t i = 0; i < _num_POs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + po_idx_accum];
    _POs[i]._idx = i + po_idx_accum;
    _POs[i]._type = gate_type[gate_id];
    _POs[i]._level = default_level;
    _POs[i]._output_value = default_output_value;
  }

  // // give the input gates number
  // for (size_t i = 0; i < adj.size(); i++) {
  //   for (size_t j = 0; j < adj[i].size(); j++) {
  //     size_t from_gate_id = i, to_gate_id = adj[i][j];
  //     if (from_gate_id < _num_PIs) { // PI -> gate
  //       PI &from_gate =
  //           _PIs[_gate_idx_to_tpg_order_idx[from_gate_id] - pi_idx_accum];
  //       Gate &to_gate =
  //           _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
  //       to_gate._inputs.push_back(&from_gate);
  //     } else if ((to_gate_id >= (_num_PIs)) &&
  //                (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
  //       Gate &from_gate =
  //           _gates[_gate_idx_to_tpg_order_idx[from_gate_id] -
  //           gate_idx_accum];
  //       PO &to_gate =
  //           _POs[_gate_idx_to_tpg_order_idx[to_gate_id] - po_idx_accum];
  //       to_gate._inputs.push_back(&from_gate);
  //     } else { // gate -> gate
  //       Gate &from_gate =
  //           _gates[_gate_idx_to_tpg_order_idx[from_gate_id] -
  //           gate_idx_accum];
  //       Gate &to_gate =
  //           _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
  //       to_gate._inputs.push_back(&from_gate);
  //     }
  //   }
  // }

  // give the input gates number - v2
  for (size_t i = _num_PIs; i < gate_input_order_table.size(); i++) {
    for (size_t j = 1; j < (gate_input_order_table[i][0] + 1);
         j++) { // how many inputs for this gate
      size_t from_gate_id = gate_input_order_table[i][j];
      size_t to_gate_id = i;
      if (from_gate_id < _num_PIs) { // PI -> gate
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _PIs[_gate_idx_to_tpg_order_idx[from_gate_id] - pi_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
            _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
        to_gate._inputs.push_back(&from_gate);
      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
            _POs[_gate_idx_to_tpg_order_idx[to_gate_id] - po_idx_accum];
        to_gate._inputs.push_back(&from_gate);
      } else { // gate -> gate
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
            _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
        to_gate._inputs.push_back(&from_gate);
      }
    }
  }

  /** levelize */
  // levelize PIs -> set by the default_level (= 0), note: all PIs belond to the
  // level_0 , making the parallel in omp becomes easier levelize gates
  _max_level = 0;
  _max_gate_size_of_levels = 0;
  for (size_t i = 0; i < _num_inner_gates; i++) {
    size_t ret_level = ((_gates[i]._inputs[0])->_level) + 1;
    for (size_t j = 1; j < _gates[i]._inputs.size(); j++) {
      size_t tmp_level = ((_gates[i]._inputs[j])->_level) + 1;
      if (ret_level < tmp_level) {
        ret_level = tmp_level;
      }
    }
    _gates[i]._level = ret_level;
    _max_level = (_max_level < ret_level) ? (ret_level) : (_max_level);
  }

  // levelize POs -> all POs belond to the level_max
  // , making the parallel in omp becomes easier
  for (size_t i = 0; i < _num_POs; i++) {
    _POs[i]._level = _max_level + 1;
  }
  _max_level++;

  _num_gates_per_level.resize(_max_level + 1);

  for (size_t i = 0; i < _num_PIs; i++) {
    _num_gates_per_level[_PIs[i]._level]++;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _num_gates_per_level[_gates[i]._level]++;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _num_gates_per_level[_POs[i]._level]++;
  }
  // find the _max_gate_size_of_levels
  _max_gate_size_of_levels = *std::max_element(_num_gates_per_level.begin(),
                                               _num_gates_per_level.end());

#ifdef FSIM_DEBUG_PRINT_GRAPH
  printf("_max_level = %ld, _max_gate_size_of_levels = %ld\n", _max_level,
         _max_gate_size_of_levels);

  std::cout << "\n_PIs = [\n";
  for (size_t i = 0; i < _num_PIs; i++) {
    print_ppg(_PIs[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_gates = [\n";
  for (size_t i = 0; i < _num_inner_gates; i++) {
    print_ppg(_gates[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_POs = [\n";
  for (size_t i = 0; i < _num_POs; i++) {
    print_ppg(_POs[i]);
  }
  std::cout << "]\n";
  print_num_gates_per_level();
#endif
}

void fsim::Simulator::_read_fault(std::istream &flst) {
  _faults.resize(_num_fault);
  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
#ifdef FSIM_DEBUG_PRINT_FAULTS
    std::cout << wrong_gate << ", " << fault_value << "\n";
#endif
    _faults[i]._gate_with_fault = _gate_idx_to_tpg_order_idx[wrong_gate];
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<size_t>::max() : (0);
  }

#ifdef FSIM_DEBUG_PRINT_FAULTS
  print_faults(_faults);
#endif
}

void fsim::Simulator::_read_pattern(std::istream &ptn) {
  _num_rounds = (_num_pattern + SIZE_T_BITS - 1) / SIZE_T_BITS; // ceiling

  _patterns.resize(_num_rounds);
  for (size_t i = 0; i < _num_rounds; i++) {
    _patterns[i]._value.resize(_num_PIs);
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = _gate_idx_to_tpg_order_idx[i];
      ptn >> _patterns[idx]._value[pi];
    }
  }

#ifdef FSIM_DEBUG_PRINT_PATTERNS
  print_patterns(_patterns, _num_rounds, _num_PIs);
#endif
}

// simulation for gates
void fsim::Simulator::_apply_INV(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

void fsim::Simulator::_apply_AND(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::_apply_OR(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::_apply_XOR(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::_apply_NAND(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void fsim::Simulator::_apply_NOR(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void fsim::Simulator::_apply_XNOR(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void fsim::Simulator::_apply_MUX(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  size_t a = gate._inputs[0]->_output_value;
  size_t b = gate._inputs[1]->_output_value;
  size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | (a & (!s)));

  gate._output_value = (ret);
}

void fsim::Simulator::_apply_CLKBUF(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}

void fsim::Simulator::_apply_PI(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate, const Pattern pattern,
                                const size_t pi) {
  gate._output_value = pattern._value[pi];
}

void fsim::Simulator::_apply_PO(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

void fsim::Simulator::_run_gate(ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate, const Pattern pattern,
                                const size_t pi, const size_t SA_fault,
                                const size_t fault_val) {

  if (SA_fault) {
    gate._output_value = fault_val;
    return;
  }
  switch (gate._type) {
  case GateType::INV:
    _apply_INV(gate);
    break;
  case GateType::AND:
    _apply_AND(gate);
    break;
  case GateType::OR:
    _apply_OR(gate);
    break;
  case GateType::XOR:
    _apply_XOR(gate);
    break;
  case GateType::NAND:
    _apply_NAND(gate);
    break;
  case GateType::NOR:
    _apply_NOR(gate);
    break;
  case GateType::XNOR:
    _apply_XNOR(gate);
    break;
  case GateType::MUX:
    _apply_MUX(gate);
    break;
  case GateType::CLKBUF:
    _apply_CLKBUF(gate);
    break;
  case GateType::PI:
    _apply_PI(gate, pattern, pi);
    break;
  case GateType::PO:
    _apply_PO(gate);
    break;
  case GateType::MAX_GATE_TYPE:
    break;
  }
}

// Function to convert GateType to string
std::string fsim::Simulator::gateTypeToString(GateType type) {
  switch (type) {
  case GateType::INV:
    return "INV";
  case GateType::AND:
    return "AND";
  case GateType::OR:
    return "OR";
  case GateType::XOR:
    return "XOR";
  case GateType::NAND:
    return "NAND";
  case GateType::NOR:
    return "NOR";
  case GateType::XNOR:
    return "XNOR";
  case GateType::MUX:
    return "MUX";
  case GateType::CLKBUF:
    return "CLKBUF";
  case GateType::PI:
    return "PI";
  case GateType::PO:
    return "PO";
  default:
    return "UNKNOWN";
  }
}

// output file visualization graph
void fsim::Simulator::visualization_graph(const std::string &outputFile) {
  using std::string_literals::operator""s;
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);

  // Open output file
  std::ofstream output_file(outputFile);

  // Check if the output file is open
  if (!output_file.is_open()) {
    std::cerr << "Error opening output file: " << outputFile << std::endl;
    return;
  }

  // write file
  output_file << "digraph G {\n";
  output_file << "\tsubgraph cluster_0 {\n";
  output_file << "\t\tcolor=black;\n";
  output_file << "\t\tnode [style=filled,color=lightgrey];\n";

  // gates
  for (size_t i = 0; i < _gates.size(); i++) {              // now_gate
    for (size_t j = 0; j < _gates[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _gates[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> now_gate = _gates[i];

      std::string pre_s;
      if (pre_gate->_type == GateType::PI) {
        pre_s =
            "PI_"s + std::to_string(pre_gate->_idx - pi_idx_accum) + " -> "s;
      } else {
        pre_s = "Gate_"s + std::to_string(pre_gate->_idx - gate_idx_accum) +
                "_"s + gateTypeToString(pre_gate->_type) + " -> "s;
      }

      std::string now_s;
      now_s = "Gate_"s + std::to_string(now_gate._idx - gate_idx_accum) + "_"s +
              gateTypeToString(now_gate._type) + ";"s;

      output_file << "\t\t" << pre_s << now_s << "\n";
    }
  }

  // POs
  for (size_t i = 0; i < _POs.size(); i++) {              // now_gate
    for (size_t j = 0; j < _POs[i]._inputs.size(); j++) { // pre_gate
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> *pre_gate =
          _POs[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> now_gate = _POs[i];

      std::string pre_s;
      if (pre_gate->_type == GateType::PI) {
        pre_s =
            "PI_"s + std::to_string(pre_gate->_idx - pi_idx_accum) + " -> "s;
      } else {
        pre_s = "Gate_"s + std::to_string(pre_gate->_idx - gate_idx_accum) +
                "_"s + gateTypeToString(pre_gate->_type) + " -> "s;
      }

      std::string now_s;
      now_s = "PO_"s + std::to_string(now_gate._idx - po_idx_accum) + ";"s;

      output_file << "\t\t" << pre_s << now_s << "\n";
    }
  }

  output_file << "\t}\n\n";

  for (size_t i = 0; i < _num_PIs; i++) {
    output_file << "\tPI_"s << std::to_string(i) << "  [shape=Msquare];\n"s;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    output_file << "\tPO_"s << std::to_string(i) << "  [shape=Msquare];\n"s;
  }
  output_file << "}\n\n";

  // Close files
  output_file.close();
}

// debug: print functions
void fsim::Simulator::print_topological(
    const std::vector<std::vector<size_t>> &adj,
    const std::vector<size_t> &indegree,
    const std::vector<size_t> &order) const {
  std::cout << "\n=====\n\nadj:\n";
  for (size_t i = 0; i < adj.size(); i++) {
    std::cout << i << " = [";
    for (size_t v_len = 0; v_len < adj[i].size(); v_len++) {
      std::cout << adj[i][v_len] << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";

  std::cout << "Indegree:\n";
  for (size_t i = 0; i < indegree.size(); i++) {
    std::cout << i << " = [" << indegree[i] << "]\n";
  }

  std::cout << "Order:\n";
  for (size_t i = 0; i < order.size(); i++) {
    std::cout << order[i] << ", ";
  }
  std::cout << "\n";
}

void fsim::Simulator::print_ppg(const ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate) const {
  std::cout << "\n=====\n\n";
  std::cout << "mem_id: " << &gate << "\n";
  std::cout << "idx: " << gate._idx << "\n";
  std::cout << "type: " << static_cast<size_t>(gate._type) << "\n";
  std::cout << "level: " << gate._level << "\n";
  std::cout << "input_gates: [";
  for (size_t i = 0; i < gate._inputs.size(); i++) {
    std::cout << gate._inputs[i] << ", ";
  }
  std::cout << "]\n";
  std::cout << "output_value: " << gate._output_value << "\n";
}

void fsim::Simulator::print_num_gates_per_level() const {
  std::cout << "\n=====\n\n";
  std::cout << "_num_gates_per_level: \n";
  for (size_t i = 0; i < _num_gates_per_level.size(); i++) {
    std::cout << "level_" << i << ": " << _num_gates_per_level[i] << "\n";
  }
  std::cout << "\n=====\n\n";
}


void fsim::Simulator::print_faults(const std::vector<Fault<FAULT_INDEX_TYPE>>  &faults) const {
  std::cout << "\n=====\n\n"
            << "FSIM_DEBUG_PRINT_FAULTS\n";
  for (size_t i = 0; i < _num_fault; i++) {
    std::cout << "_gate_with_fault = " << faults[i]._gate_with_fault << ", SA_"
              << faults[i]._gate_SA_fault_val << "\n";
  }
  std::cout << "\n=====\n\n";
}


void fsim::Simulator::print_patterns(const std::vector<Pattern> &patterns,
                                     const size_t round,
                                     const size_t num_PIs) const {
  std::cout << "\n=====\n\n";
  for (size_t i = 0; i < round; i++) {
    std::cout << "[" << SIZE_T_BITS * i << ", " << SIZE_T_BITS * (i + 1)
              << "] bits = [\n";
    for (size_t j = 0; j < num_PIs; j++) {
      print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
    }
    std::cout << "]\n";
  }
}

/**
 * `print_bits_stack` from stackoverflow:
 * https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
 */

void fsim::Simulator::print_bits_stack(const size_t size,
                                       const void *const ptr) const {
  unsigned char *b = (unsigned char *)ptr;
  unsigned char byte;
  int i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      std::cout << static_cast<unsigned>(byte);
    }
  }
  std::cout << "\n";
}

void fsim::Simulator::print_simulation_results(
    const std::vector<size_t> &pi_results,
    const std::vector<size_t> &gate_results,
    const std::vector<size_t> &po_results) const {
  for (size_t i = 0; i < pi_results.size(); i++) {
    std::cout << "PI_" << i << ".output = " << pi_results[i] << "\n";
  }
  for (size_t i = 0; i < gate_results.size(); i++) {
    std::cout << "Gate_" << i << ".output = " << gate_results[i] << "\n";
  }
  for (size_t i = 0; i < po_results.size(); i++) {
    std::cout << "PO_" << i << ".output = " << po_results[i] << "\n";
  }
  std::cout << "\n\n";
}

void fsim::Simulator::print_found_fault_to_pattern(
    const std::vector<size_t> &found_fault_to_pattern) const {
  for (size_t i = 0; i < (2 * _num_fault); i += 2) {
    if (found_fault_to_pattern[i] == 1) {
      std::cout << "fault_" << (i / 2) << " is found, with Pattern_"
                << found_fault_to_pattern[i + 1] << "\n";
    } else {
      std::cout << "fault_" << i << " cannot be found\n";
    }
  }
  std::cout << "\n=====\n\n";
}

void fsim::Simulator::print_gate_input_order_table(
    std::vector<std::vector<size_t>> input_table) {
  std::cout << "\ngate_input_order_table:\n";
  for (size_t i = 0; i < input_table.size(); i++) {
    std::cout << "Gate_" << i << ": ";
    std::cout << "num_of_inputs = " << input_table[i][0]
              << ", input gate_idx: ";
    for (size_t j = 0; j < input_table[i][0];
         j++) { // how many inputs for this gate
      std::cout << input_table[i][j + 1] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "]\n";
  std::cout << "\n=====\n\n";
}