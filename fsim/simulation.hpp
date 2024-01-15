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

#define print_graph


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

// Print for check
#define print_simulation
// #define print_graph

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
      
      size_t _num_PIs, _num_POs, _num_gate, _num_wire, _total_gates;
      std::vector<size_t> _gate_id_to_order_idx;
      std::vector<Gate> _gates;
      void construct_graph(std::ifstream &ckt);
      void print_topological (std::vector<std::vector <size_t>> &adj, std::vector<size_t> &indegree, std::vector<size_t> &order);
      void print_gate(Gate& gate);
  };


}


#endif
