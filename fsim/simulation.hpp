#ifndef SIMULATION_H
#define SIMULATION_H
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fault.hpp"
#include "gate.hpp"
#include "pattern.hpp"

#define SIZE_T_BITS (sizeof(size_t) * CHAR_BIT)
#define SIMPLE_TESTCASE_BITS 3

// Print for check
#define print_simulation
#define print_graph
#define print_faults
#define print_patterns
#define print_fault_table

namespace fsim {

class Simulator {

public:
  void read(const std::string &ckt, const std::string &flst,
            const std::string &ptn);
  void run();

  // // accessor
  // size_t num_gates() const { return _gates.size(); }
  // const std::vector<Gate>& gates() const { return _gates; }

private:
  // Read
  size_t _num_PIs, _num_POs, _num_gate, _num_wire, _total_gates;
  std::vector<size_t> _gate_id_to_order_idx;
  std::vector<Gate> _gates;

  size_t _num_fault;
  std::vector<Fault> _faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _round;       // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _pattern;

  // Construct
  void construct_graph(std::ifstream &ckt);
  void construct_fault(std::ifstream &flst);
  void construct_pattern(std::ifstream &ptn);

  // Simulation
  size_t _testcase_bits;
  std::vector<size_t> _good_result;
  std::vector<size_t> _bad_result;
  std::vector<size_t> _found_fault_table;

  void good_case(Pattern &pattern, size_t bits);
  void bad_case(Fault &fault, Pattern &pattern, size_t bits);

  void INV(Gate &gate);
  void AND(Gate &gate);
  void OR(Gate &gate);
  void XOR(Gate &gate);
  void NAND(Gate &gate);
  void NOR(Gate &gate);
  void PI(Gate &gate, Pattern &pattern, size_t pi);
  void PO(Gate &gate);
  void run_gate(Gate &gate, Pattern &pattern, size_t pi, size_t SA_fault,
                size_t fault_val);
  void shift_to_correct_answer(std::vector<size_t> &results, size_t bits);

  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
                         const std::vector<size_t> &indegree,
                         const std::vector<size_t> &order);
  void print_gate(const Gate &gate);
  void print_fault(const std::vector<Fault> &_faults);
  void print_pattern(const std::vector<Pattern> &patterns, const size_t round,
                     const size_t num_PIs);
  void print_bits_stack(const size_t size, const void *const ptr);
  void print_check_simulation(const std::vector<size_t> &results);
  void print_found_fault_table(const std::vector<size_t> &_found_fault_table);
};

} // namespace fsim

#endif
