#ifndef SIMULATION_H
#define SIMULATION_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fault.hpp"
#include "gate.hpp"
#include "pattern.hpp"

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
  
  // for testing
  void read(std::istream& ckt, std::istream& flst, std::istream& pin);

  // TODO (parallel): add a level-based parallel simulation
  void run(Mode mode);

  const std::vector<size_t>& g_gate_results() const {
    return _g_gate_results;
  }
  
  const Gate& gate(size_t gid) const {
    return _gates[gid];
  }
  

  // // accessor
  // size_t num_gates() const { return _gates.size(); }
  // const std::vector<Gate>& gates() const { return _gates; }

private:
  
  // Read
  void _read_files(std::ifstream &ckt);
  
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires, _total_gates;
  std::vector<size_t> _gate_idx_to_tpg_order_idx;
  std::vector<Gate> _gates;
  
  size_t _num_fault; // for read file
  std::vector<Fault> _faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _pattern;

  // Construct
  void _read_graph(std::ifstream &ckt);
  void _read_fault(std::ifstream &flst);
  void _read_pattern(std::ifstream &ptn);

  // Simulation
  std::vector<size_t> _g_gate_results; 
  std::vector<size_t> _b_gate_results;
  std::vector<size_t> _found_fault_to_pattern;

  // Mode: _run_sequential, _run_parallel
  void _run_sequential();
  void _run_parallel_tf();
  void _run_parallel_omp();
  void _run_good_case(const Pattern pattern, const size_t bits);
  void _run_bad_case(const Fault &fault, const Pattern pattern, const size_t bits);

  void _apply_INV(Gate &gate);
  void _apply_AND(Gate &gate);
  void _apply_OR(Gate &gate);
  void _apply_XOR(Gate &gate);
  void _apply_NAND(Gate &gate);
  void _apply_NOR(Gate &gate);
  void _apply_PI(Gate &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(Gate &gate);
  
  void _run_gate(Gate &gate, const Pattern pattern, const size_t pi, const size_t SA_fault,
                const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results, const size_t bits);

  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
                         const std::vector<size_t> &indegree,
                         const std::vector<size_t> &order) const;
  void print_gate(const Gate &gate) const;
  void print_faults(const std::vector<Fault> &_faults) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                     const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_simulation_results(const std::vector<size_t> &results) const;
  void print_found_fault_to_pattern(const std::vector<size_t> &_found_fault_to_pattern) const;
};

} // namespace fsim

#endif
