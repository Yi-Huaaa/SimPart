#ifndef SIMULATION_H
#define SIMULATION_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


#include "../taskflow/taskflow/taskflow.hpp"
#include "../taskflow/taskflow/algorithm/for_each.hpp"
#include "base_ppg.hpp"
#include "fault.hpp"
#include "gate.hpp"
#include "pattern.hpp"
#include "pi.hpp"
#include "po.hpp"

// fsim: Taskflow using the for_each_loop ver w/ this define 
// #define TF_parallel_for

// // fsim print for check
// #define FSIM_DEBUG_PRINT_SIMULATION
// #define FSIM_DEBUG_PRINT_GRAPH
// #define FSIM_DEBUG_PRINT_FAULTS
// #define FSIM_DEBUG_PRINT_PATTERNS
// #define FSIM_DEBUG_PRINT_FAULT_TABLE


#define ELEMENT_INDEX_TYPE size_t
#define ELEMENT_LEVEL_TYPE size_t
#define FAULT_INDEX_TYPE size_t


namespace fsim {

constexpr auto SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;

class Simulator {

  friend class CPUPartitioner;
  friend class CUDAPartitioner;

public:
  enum class Mode { SEQUENTIAL = 0, PARALLEL_TF, PARALLEL_OMP };

  void read(const std::string &ckt, const std::string &flst,
            const std::string &ptn);
  void read(const std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run(Mode mode, const size_t num_threads);

  // for unittest
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);

  // testing
  void dump_graph(std::ostream &) const;

  const std::vector<size_t> &g_pi_results() const { return _g_pi_results; }
  const std::vector<size_t> &g_gate_results() const { return _g_gate_results; }
  const std::vector<size_t> &g_po_results() const { return _g_po_results; }

  // return gates, faults, patterns
  // single
  const ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &pi(size_t piid) const { return _PIs[piid]; }
  const ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &gate(size_t gid) const { return _gates[gid]; }
  const ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &po(size_t poid) const { return _POs[poid]; }
  // all of them
  const std::vector<ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE>> &gates() const { return _gates; }
  
  const std::vector<Fault<>> &faults() const { return _faults; }
  const std::vector<Pattern> &patterns() const { return _patterns; }

  // accessor
  // var
  size_t num_PIs() const { return _PIs.size(); }
  size_t num_gates() const { return _gates.size(); }
  size_t num_POs() const { return _POs.size(); }
  size_t num_wires() const { return _num_wires; }
  size_t num_pos() const { return _POs.size(); }
  size_t num_rounds() const { return _num_rounds; }
  const std::vector<size_t> &gate_idx_to_tpg_order_idx() const {
    return _gate_idx_to_tpg_order_idx;
  }
  const std::vector<size_t> &num_gates_per_level() const {
    return _num_gates_per_level;
  }
  const std::vector<size_t> &found_fault_to_pattern() const {
    return _found_fault_to_pattern;
  }

  // input of the "Graph_viz online"
  std::string gateTypeToString(GateType type);
  void visualization_graph(const std::string &outputFile);

private:
  // Read
  tf::Executor _executor;

  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  std::vector<size_t> _gate_idx_to_tpg_order_idx;
  std::vector<ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE>> _PIs;
  std::vector<ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE>> _gates;
  std::vector<ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE>> _POs;

  size_t _num_fault; // for read file
  std::vector<Fault<FAULT_INDEX_TYPE>> _faults; 

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // used for run parallel
  std::vector<size_t> _num_gates_per_level; // records number of gates per level
  size_t _max_level; // the max number of levels
  size_t _max_gate_size_of_levels;  // the max number of gates in all levels
  

  // Construct
  void _read_graph(std::istream &ckt);
  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

  // Simulation
  std::vector<size_t> _g_pi_results;
  std::vector<size_t> _g_gate_results;
  std::vector<size_t> _g_po_results;
  std::vector<size_t> _b_pi_results;
  std::vector<size_t> _b_gate_results;
  std::vector<size_t> _b_po_results;
  std::vector<size_t> _found_fault_to_pattern;



  // Mode: _run_sequential, _run_parallel
  // sequential
  void _run_sequential();
  void _run_good_case(const Pattern pattern, const size_t bits);
  void _run_bad_case(const Fault<FAULT_INDEX_TYPE> &fault, const Pattern pattern,
                     const size_t bits);
  // parallel-tf
  void _run_parallel_tf();
  void _run_good_case_tf(const Pattern pattern, const size_t bits);
  // _which_fault_pattern: private indices for TF to emplace work for only 1 time
  std::vector<size_t> _which_fault_pattern; // [0]: which_fault, [1]: which_pattern
  void _emplace_bad_case_tf(tf::Taskflow &taskflow);
  void _run_bad_case_tf(tf::Taskflow &taskflow, const size_t bits);

  // parallel-omp
  void _run_parallel_omp(const size_t num_threads);
  void _run_good_case_omp(const Pattern pattern, const size_t bits,
                          const size_t num_threads);
  void _run_bad_case_omp(const Fault<FAULT_INDEX_TYPE> &fault, const Pattern pattern,
                         const size_t bits, const size_t num_threads);

  void _apply_INV(ElementBase<> &gate);
  void _apply_AND(ElementBase<> &gate);
  void _apply_OR(ElementBase<> &gate);
  void _apply_XOR(ElementBase<> &gate);
  void _apply_NAND(ElementBase<> &gate);
  void _apply_NOR(ElementBase<> &gate);
  void _apply_XNOR(ElementBase<> &gate);
  void _apply_MUX(ElementBase<> &gate);
  void _apply_CLKBUF(ElementBase<> &gate);
  void _apply_PI(ElementBase<> &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(ElementBase<> &gate);

  void _run_gate(ElementBase<> &gate, const Pattern pattern, const size_t pi,
                 const size_t SA_fault, const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results, const size_t bits,
                                const size_t num_shift_gates);

  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
                         const std::vector<size_t> &indegree,
                         const std::vector<size_t> &order) const;
  void print_ppg(const ElementBase<> &gate) const;
  void print_num_gates_per_level() const;
  void print_faults(const std::vector<Fault<FAULT_INDEX_TYPE>> &faults) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                      const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_simulation_results(const std::vector<size_t> &pi_results,
                                const std::vector<size_t> &gate_results,
                                const std::vector<size_t> &po_results) const;
  void print_found_fault_to_pattern(
      const std::vector<size_t> &_found_fault_to_pattern) const;
  void print_gate_input_order_table(std::vector<std::vector<size_t>> input_table);
};

} // namespace fsim

#endif