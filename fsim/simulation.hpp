#ifndef SIMULATION_H
#define SIMULATION_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>


#include "fault.hpp"
#include "pattern.hpp"
#include "gate.hpp"

#define SIZE_T_BITS (sizeof(size_t) * CHAR_BIT)
#define simple_testcase_bits 3

// Print for check
#define print_simulation
#define print_graph
#define print_faults
#define print_patterns
#define print_found_fault_table


/* Functions */
void INV (Gate *gate);
void AND (Gate *gate);
void OR (Gate *gate);
void XOR (Gate *gate);
void NAND (Gate *gate);
void NOR (Gate *gate);
void PI (Gate *gate, int PI_i, Pattern *pattern);
void PO (Gate *gate);



unsigned int *good_case (Gate **gates, Pattern* pattern, int total_gates, int bits);
void print_check_simulation(unsigned int *results, int total_gates);
int SA_faults (Gate **gates, Fault faults, Pattern* pattern, int total_gates, int num_POs, unsigned int *good_results, int bits);



// TODO:
namespace fsim {

  class Simulator {

    public:

      void read(const std::string& ckt, const std::string& flst, const std::string& ptn);
      void run();

      // // accessor
      // size_t num_gates() const { return _gates.size(); }
      // const std::vector<Gate>& gates() const { return _gates; }

    private:
      
      // 
      size_t _num_PIs, _num_POs, _num_gate, _num_wire, _total_gates;
      std::vector<size_t> _gate_id_to_order_idx;
      std::vector<Gate> _gates;
      
      size_t _num_fault;
      std::vector<Fault> _faults;

      size_t _num_pattern; // total number of patterns that need to be tested
      size_t _round; // ceiling(_num_pattern/SIZE_T_BITS)
      std::vector<Pattern> _pattern;
      
      // construct
      void construct_graph(std::ifstream &ckt);
      void construct_fault(std::ifstream &flst);
      void construct_pattern(std::ifstream &ptn);

      // print
      void print_topological (std::vector<std::vector <size_t>> &adj, std::vector<size_t> &indegree, std::vector<size_t> &order);
      void print_gate(Gate& gate);
      void print_fault(std::vector<Fault> &_faults);
      void print_pattern (std::vector<Pattern> &patterns, size_t round, size_t num_PIs);
      void printBits_stack(size_t const size, void const * const ptr);
      void print_check_simulation (std::vector<size_t> &results);
      
      
      // simulation
      size_t _testcase_bits;
      std::vector<size_t> good_result;

      void good_case (Pattern pattern, size_t bits);

      void INV (Gate &gate, size_t SA_fault, size_t fault_val);
      void AND (Gate &gate, size_t SA_fault, size_t fault_val);
      void OR (Gate &gate, size_t SA_fault, size_t fault_val);
      void XOR (Gate &gate, size_t SA_fault, size_t fault_val);
      void NAND (Gate &gate, size_t SA_fault, size_t fault_val);
      void NOR (Gate &gate, size_t SA_fault, size_t fault_val);
      void PI (Gate &gate, Pattern &pattern, size_t PI_i, size_t SA_fault, size_t fault_val);
      void PO (Gate &gate);
      void run_gate (Gate &gate, Pattern &pattern, size_t P_i, size_t SA_fault, size_t fault_val);
      void shift_to_correct_answer (std::vector<size_t> &results, size_t bits);
  };


}


#endif
