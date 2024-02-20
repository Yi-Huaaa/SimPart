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
__global__ void _run_gate(ElementBase *gate, const Pattern pattern, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level);
// simulation for gates
__device__ __forceinline__ void _apply_INV(ElementBase &gate);
__device__ __forceinline__ void _apply_AND(ElementBase &gate);
__device__ __forceinline__ void _apply_OR(ElementBase &gate);
__device__ __forceinline__ void _apply_XOR(ElementBase &gate);
__device__ __forceinline__ void _apply_NAND(ElementBase &gate);
__device__ __forceinline__ void _apply_NOR(ElementBase &gate);
__device__ __forceinline__ void _apply_XNOR(ElementBase &gate);
__device__ __forceinline__ void _apply_MUX(ElementBase &gate);
__device__ __forceinline__ void _apply_CLKBUF(ElementBase &gate);
__device__ __forceinline__ void _apply_PI(ElementBase &gate, const Pattern pattern, const size_t pi);
__device__ __forceinline__ void _apply_PO(ElementBase &gate);


// print for check
// #define FSIM_DEBUG_PRINT_SIMULATION
// #define FSIM_DEBUG_PRINT_GRAPH
// #define FSIM_DEBUG_PRINT_FAULTS
// #define FSIM_DEBUG_PRINT_PATTERNS
// #define FSIM_DEBUG_PRINT_FAULT_TABLE


class CudaSimulator {

// constexpr size_t SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;
const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

// friend functions 
friend __global__ void _run_gate(ElementBase *gate, const Pattern pattern, const size_t pi, const size_t SA_fault, const size_t fault_val, const size_t max_gate_num_per_level);

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
  void run(Mode mode, const size_t num_threads);
  
  // input of the "Graph_viz online"
  std::string gateTypeToString(GateType type);
  void visualization_graph(const std::string &outputFile);
  
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
  size_t _size_of_num_gates_per_level; // _num_gates_per_level.size()

  size_t _num_fault; // for read file
  Fault *_faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  Pattern *_patterns;

  // Construct
  void _read_graph(std::istream &ckt);
  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

  // Simulation
  size_t *_g_pi_results;
  size_t *_g_gate_results;
  size_t *_g_po_results;
  size_t *_b_pi_results;
  size_t *_b_gate_results;
  size_t *_b_po_results;
  size_t *_found_fault_to_pattern;

  // parallel-cuda
  void _run_parallel_cuda();
  void _run_good_case_cuda(const Pattern pattern, const size_t bits);
  void _run_bad_case_cuda(const Fault &fault, const Pattern pattern, const size_t bits);
  void _shift_to_correct_answer(size_t *results, const size_t bits, const size_t num_shift_gates);

  
  


  // print for debug
  void print_topological(const std::vector<std::vector<size_t>> &adj,
    const std::vector<size_t> &indegree,
    const std::vector<size_t> &order) const;
  void print_gate_input_order_table(std::vector<std::vector<size_t>> input_table);
  void print_num_gates_per_level() const;
  void print_ppg(const ElementBase &gate) const;
  void print_faults(const Fault *faults) const;
  void print_patterns(const Pattern *patterns, const size_t round, const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_found_fault_to_pattern( const size_t *found_fault_to_pattern) const;
  void print_simulation_results(const size_t *pi_results, const size_t *gate_results, const size_t *po_results) const;
};


#endif