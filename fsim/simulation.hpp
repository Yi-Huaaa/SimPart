#ifndef SIMULATION_H
#define SIMULATION_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "base_ppg.hpp"
#include "pi.hpp"
#include "po.hpp"
#include "gate.hpp"
#include "fault.hpp"
#include "pattern.hpp"
#include "../taskflow/taskflow/taskflow.hpp"

// fsim print for check
#define FSIM_DEBUG_PRINT_SIMULATION
#define FSIM_DEBUG_PRINT_GRAPH
#define FSIM_DEBUG_PRINT_FAULTS
#define FSIM_DEBUG_PRINT_PATTERNS
#define FSIM_DEBUG_PRINT_FAULT_TABLE

namespace fsim {

constexpr auto SIZE_T_BITS  = sizeof(size_t) * CHAR_BIT;

class Simulator {

public:
  
  enum class Mode {
    SEQUENTIAL = 0,
    PARALLEL_TF,
    PARALLEL_OMP
  };

  void read(const std::string &ckt, const std::string &flst,
            const std::string &ptn);
  void read(const std::istream& ckt, std::istream& flst, std::istream& ptn);
  void run(Mode mode, const size_t num_threads);

  // for unittest
  void read(std::istream& ckt, std::istream& flst, std::istream& ptn);

  // TODO: for visualization - graph 
  // (v) add warning flags "-Wall -Wextra -Wfatal-errors -Wshadow" to your CMake
  // (v) add make flag during the cmake to make the unittests run directly
  // add more unittests to run all of your program: for exmaple: different mode testing
  void dump_graph(std::ostream&) const;


  const std::vector<size_t>& g_pi_results() const { return _g_pi_results; }
  const std::vector<size_t>& g_gate_results() const { return _g_gate_results; }
  const std::vector<size_t>& g_po_results() const { return _g_po_results; }
  
  // return gates, faults, patterns
  // single
  const PI& pi(size_t piid) const { return _PIs[piid]; }
  const Gate& gate(size_t gid) const { return _gates[gid];}
  const PO& po(size_t poid) const { return _POs[poid];}
  // all of them
  const std::vector<Gate>& gates() const { return _gates; }
  const std::vector<Fault>& faults() { return _faults; }
  const std::vector<Pattern>& patterns() { return _patterns;}   

  // accessor
  // var 
  size_t num_PIs() const { return _PIs.size(); }
  size_t num_gates() const { return _gates.size(); }
  size_t num_POs() const { return _POs.size(); }
  size_t num_wires() const { return _num_wires; }
  size_t num_pos() const { return _POs.size(); }
  size_t num_rounds() const { return _num_rounds; }
  const std::vector<size_t> gate_idx_to_tpg_order_idx() const  { return _gate_idx_to_tpg_order_idx; }
  const std::vector<size_t> num_gates_per_level() const  { return _num_gates_per_level; }
  const std::vector<size_t> found_fault_to_pattern () const { return _found_fault_to_pattern; }
  

private:
  
  // Read
  tf::Executor _executor;

  
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  std::vector<size_t> _gate_idx_to_tpg_order_idx;
  std::vector<Gate> _gates;
  std::vector<PI> _PIs;
  std::vector<PO> _POs;
  
  size_t _num_fault; // for read file
  std::vector<Fault> _faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // used for run parallel
  std::vector<size_t> _num_gates_per_level; // records number of gates per level 

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
  void _run_bad_case(const Fault &fault, const Pattern pattern, const size_t bits);
  // parallel-tf
  // todo: check whether taskflow use 4 threads runs parallelly
  void _run_parallel_tf();
  void _run_good_case_tf(const Pattern pattern, const size_t bits);
  void _run_bad_case_tf(const Fault &fault, const Pattern pattern, const size_t bits);
  // parallel-omp
  void _run_parallel_omp(const size_t num_threads);
  void _run_good_case_omp(const Pattern pattern, const size_t bits, const size_t num_threads);
  void _run_bad_case_omp(const Fault &fault, const Pattern pattern, const size_t bits, const size_t num_threads);


  void _apply_INV(BasePPG &gate);
  void _apply_AND(BasePPG &gate);
  void _apply_OR(BasePPG &gate);
  void _apply_XOR(BasePPG &gate);
  void _apply_NAND(BasePPG &gate);
  void _apply_NOR(BasePPG &gate);
  void _apply_PI(BasePPG &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(BasePPG &gate);
  
  void _run_gate(BasePPG &gate, const Pattern pattern, const size_t pi, const size_t SA_fault,
                const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results, const size_t bits, const size_t num_shift_gates);

  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
                         const std::vector<size_t> &indegree,
                         const std::vector<size_t> &order) const;
  void print_ppg(const BasePPG &gate) const;
  void print_num_gates_per_level() const;
  void print_faults(const std::vector<Fault> &faults) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                     const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_simulation_results(const std::vector<size_t> &pi_results, const std::vector<size_t> &gate_results, const std::vector<size_t> &po_results) const;
  void print_found_fault_to_pattern(const std::vector<size_t> &_found_fault_to_pattern) const;
};

} // namespace fsim

#endif