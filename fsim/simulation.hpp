#ifndef SIMULATION_H
#define SIMULATION_H
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fault.hpp"
#include "gate.hpp"
#include "pattern.hpp"

//#define SIZE_T_BITS (sizeof(size_t) * CHAR_BIT)
//#define SIMPLE_TESTCASE_BITS 3

// Print for check
// TODO: rename
// #define FSIM_DEBUG_PRINT_SIMULATION
#define print_simulation
#define print_graph
#define print_faults
#define print_patterns
#define print_fault_table

namespace fsim {

constexpr auto SIZE_T_BITS  = sizeof(size_t) * CHAR_BIT;
constexpr auto SIMPLE_TESTCASE_BITS = 3;

class Simulator {

public:
  
  enum class Mode {
    SEQUENTIAL = 0,
    PARALLEL
  };

  void read(const std::string &ckt, const std::string &flst,
            const std::string &ptn);
  
  // for testing
  void read(std::istream& ckt, std::istream& flst, std::istream& pin);

  // TODO: add a level-based parallel simulation
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

  // TODO: _num_gates, _num_wires
  size_t _num_PIs, _num_POs, _num_gate, _num_wire, _total_gates;
  std::vector<size_t> _gate_id_to_order_idx;  // _gate2tpgorder
  std::vector<Gate> _gates;
  
  // TODO: reduce redundancies (we don't need another data members for _num_fault,
  // as it can be queried by _faults.size())
  size_t _num_fault;
  std::vector<Fault> _faults;

  // TODO: naming...
  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _round;       // ceiling(_num_pattern/SIZE_T_BITS) => num_rounds ???1
  std::vector<Pattern> _pattern;

  // Construct
  // TODO: rename...
  void _read_graph(std::ifstream &ckt);
  void _read_fault(std::ifstream &flst);
  void _read_pattern(std::ifstream &ptn);

  // Simulation
  // TODO: local variable for testcase_bit...???
  size_t _testcase_bits;
  std::vector<size_t> _g_gate_results; 
  std::vector<size_t> _b_gate_results;
  std::vector<size_t> _found_fault_table; // fault2pattern???

  // TODO: private members and functions should be prefixed with '_'
  void _run_good_case(Pattern &pattern, size_t bits);
  void _run_bad_case(Fault &fault, Pattern &pattern, size_t bits);

  void _apply_INV(Gate &gate);
  void AND(Gate &gate);
  void OR(Gate &gate);
  void XOR(Gate &gate);
  void NAND(Gate &gate);
  void NOR(Gate &gate);


  void run_PI(Gate &gate, Pattern &pattern, size_t pi);
  void run_PO(Gate &gate);
  void run_gate(Gate &gate, Pattern &pattern, size_t pi, size_t SA_fault,
                size_t fault_val);
  void shift_to_correct_answer(std::vector<size_t> &results, size_t bits);

  // print for debug
  // TODO: define const method if not changing in-class members
  void print_topological(const std::vector<std::vector<size_t>> &adj,
                         const std::vector<size_t> &indegree,
                         const std::vector<size_t> &order) const;
  void print_gate(const Gate &gate) const;
  void print_fault(const std::vector<Fault> &_faults) const;
  void print_pattern(const std::vector<Pattern> &patterns, const size_t round,
                     const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_check_simulation(const std::vector<size_t> &results) const;
  void print_found_fault_table(const std::vector<size_t> &_found_fault_table) const;
};

} // namespace fsim

#endif
