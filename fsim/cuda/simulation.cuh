#ifndef CUDA_SIMULATION_H
#define CUDA_SIMULATION_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "base_ppg.cuh"
#include "fault.cuh"
#include "gate.cuh"
#include "pattern.cuh"
#include "pi.cuh"
#include "po.cuh"

// CUDA functions 
// testing functions 
__global__ void helloCUDA();
__global__ void _run_gate_gpu(size_t *inputs);



// defiine for debug
#define FSIM_DEBUG_PRINT_GRAPH



class CudaSimulator {

public:
 
  // testing functions 
  void do_hello () {
    printf("???\n");
    helloCUDA <<< 1, 1 >>> ();
    
    _init();
    _run_gate_gpu <<< 1, 1 >>> (inputs_gpu);
    cudaDeviceSynchronize();

    printf("check\n");
    size_t num = 10;
    for (size_t i = 0; i < num; i++) {
      printf("%ld, ", inputs_gpu[i]);
    } printf("\n");
  }

// =======================================================================

  // cuda simulator functions 
  enum class Mode { CUDA=0 };

  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  // void run(Mode mode, const size_t num_threads);            
  
  
private: 
  
  // testig functions 
  // store data on unified memory 
  size_t *inputs_gpu;
  void _init () {
    size_t num  = 10;
    cudaMallocManaged(&inputs_gpu, num*sizeof(size_t));
    for (size_t i = 0; i < num; i++) {
      inputs_gpu[i] = i;
    }
  }
  
// =======================================================================
  // gpu functions
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t *_gate_idx_to_tpg_order_idx;
  PI *_PIs;
  Gate *_gates;
  PO *_POs;

  size_t _max_level; // the max number of levels
  size_t _max_gate_size_of_levels;  // the max number of gates in all levels
  size_t *_num_gates_per_level; // records number of gates per level

  // Construct
  void _read_graph(std::istream &ckt);
  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
    const std::vector<size_t> &indegree,
    const std::vector<size_t> &order) const;
  void print_gate_input_order_table(std::vector<std::vector<size_t>> input_table);
  void print_num_gates_per_level() const;
  void print_ppg(const ElementBase &gate) const;
};


#endif