#ifndef CUDA_SIMULATION_H
#define CUDA_SIMULATION_H
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <list>
#include <cublas_v2.h>
#include <set>

#include "base_ppg.cuh"
#include "fault.cuh"
#include "gate.cuh"
#include "pattern.cuh"
#include "pi.cuh"
#include "po.cuh"
#include "FFR.cuh"


// CUDA functions 
__global__ void _run_gate(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                        size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                        size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                        size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                        const size_t rd, const size_t fault_num, const size_t bad_case, 
                        const size_t max_gate_num_per_level, const size_t fin_num_gates, const size_t _num_PIs);

__global__ void _compare_g_b_results(const size_t j, const size_t _num_POs, const size_t rd, size_t *_g_po_results_gpu, size_t *_b_po_results_gpu, size_t *_found_fault_to_pattern_gpu);
__global__ void _run_gate_graph(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                size_t *_params_gpu, size_t *_order_gpu);


__device__ __forceinline__ void _apply_INV(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_AND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_OR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_XOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_NAND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_NOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_XNOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_MUX(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_CLKBUF(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__device__ __forceinline__ void _apply_PI(size_t gate_idx, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val);
__device__ __forceinline__ void _apply_PO(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
__global__ void _shift_to_correct_answer_gpu(size_t *outputs, size_t *_pi_gate_po_output_res_gpu, const size_t bits, const size_t SIZE_T_BITS, const size_t pre_gates_accum, const size_t num);

__global__ void _set_params(size_t *_params_gpu_idx_address, 
                            size_t var_0, size_t var_1, size_t var_2, 
                            size_t var_3, size_t var_4, size_t var_5, size_t var_6);
__global__ void _update_params_gpu (size_t *_params_gpu_idx_address, size_t _num_paras, size_t limit_sz, 
                                    size_t new_rd, size_t new_fault_num, size_t new_bad_case);

__global__ void _run_gate_graph_mul_levels(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                    size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                    size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                    size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                    size_t *_params_gpu, size_t *_order_gpu, size_t *_num_gates_per_level_gpu);


// for FSIM_DEBUG_GPU_INPUTS
__global__ void print_pi_gate_po_gate_type_gpu(size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_num_inputs_gpu, size_t num);
__global__ void print_patterns_gpu(size_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs);
__global__ void print_fault_gate_idx_gpu(size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, size_t _num_faults, 
                                        size_t * _FFR_levelization_gpu, size_t s, size_t n);
__global__ void _show_computational_results (size_t *a, size_t start, size_t end);


// print for check
// #define FSIM_DEBUG_PRINT_SIMULATION // print all of gate values after the fault simulation
// #define FSIM_DEBUG_PRINT_GRAPH
// #define FSIM_DEBUG_PRINT_FAULTS
// #define FSIM_DEBUG_PRINT_PATTERNS
// #define FSIM_DEBUG_PRINT_FAULT_TABLE
// #define FSIM_DEBUG_GPU_INPUTS

// #define CUDA_GRAPH_CHECK
#define FSIM_LEVELIZAED_PARTITION // levelization 
// #define FSIM_CHECK_PARTITION_RESULTS
// #define FSIM_CHECK_CONSTRUCT_GATE_KNS // show the processes of constructing gates' KNs


// forward declaration 
class CUDAPartitioner;



class CUDASimulator {

  // constexpr size_t SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;
  const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;
  const size_t SIZE_T_MAX = std::numeric_limits<size_t>::max();
  const size_t NULL_gate_idx = SIZE_T_MAX;

  friend class CUDAPartitioner;

public:
  // cuda simulator functions 
  enum class Mode { CUDA=0, CUDA_GRAPH };

  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run(Mode mode, const size_t num_threads);
  
  // input of the "Graph_viz online"
  std::string gateTypeToString(GateType type);
  void visualization_graph(const std::string &outputFile);
  
private:   
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

  size_t _num_faults; // for read file
  Fault *_faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  Pattern *_patterns;

  std::vector<size_t> _order; // implementation order after topological sort

  // Construct
  void _read_graph(std::istream &ckt);
  void _partition(std::vector<std::vector<size_t>> &adj_topol,
                  std::vector<std::vector<size_t>> &inv_adj_topol,
                  std::vector<std::vector<size_t>> &gates_per_level,
                  std::vector<size_t> &level_of_gates) ;
  void _find_sets(const size_t gate_idx, const size_t start_level, const size_t end_level, 
                  std::set<size_t> &visited,
                  std::vector<std::vector<size_t>> &adj_topol,
                  std::vector<std::vector<size_t>> &inv_adj_topol,
                  std::vector<size_t> &level_of_gates, 
                  std::vector<std::vector<std::vector<size_t>>> &partition_sets);
  void _find_sub_sets(const size_t set_idx, const size_t start_level, const size_t end_level, 
                    std::vector<std::vector<size_t>> &adj_topol,
                    std::vector<std::vector<size_t>> &inv_adj_topol,
                    std::vector<size_t> &level_of_gates,
                    std::vector<std::vector<std::vector<size_t>>> &partition_sets,
                    std::vector<std::vector<std::vector<size_t>>> &partition_sets_2);
  bool _can_merge(std::vector<std::vector<size_t>> &sub_set);
  void _split_tmp_subsets(std::vector<std::vector<size_t>> &sub_set, 
                          std::vector<size_t> &added_num_gates);
                    
  void _merge_small_sets(const size_t start_level, const size_t end_level, 
                        std::vector<std::vector<std::vector<size_t>>> &partition_sets_2,
                        std::vector<std::vector<std::vector<size_t>>> &_partition_results);

  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

  const size_t _tail_ended_OPT_threshold = 1024;
  const size_t _head_front_OPT_threshold = 1024;
  const size_t _head_front_OPT_threshold_log = 10;
  size_t _total_num_levels;
  
  // compute number of FFRs in each level 
  std::vector<size_t> num_FFRs_in_this_level;

  // Simulation
  size_t *_gate_inputs_list;
  size_t *_gate_inputs_list_start;
  size_t _sum_inputs_gates;
  size_t *_found_fault_to_pattern;
  size_t *_order_cpu;
  size_t _num_total_gates_after_head_end_opt;

// GPU pointers
  // CSR format: 
  size_t *_order_gpu;
  size_t *_pi_gate_po_gate_type_gpu;
  size_t *_pi_gate_po_output_res_gpu;
  size_t *_pi_gate_po_num_inputs_gpu;
  
  size_t *_gate_inputs_list_gpu;
  size_t *_gate_inputs_list_start_gpu;

  // Pattern *_patterns_gpu;
  size_t *_patterns_gpu;

  // Fault *_faults_gpu;
  size_t *_fault_gate_idx_gpu;
  size_t *_fault_SA_fault_val_gpu;

  // bad results 
  size_t *_g_pi_results_gpu; 
  size_t *_g_gate_results_gpu; 
  size_t *_g_po_results_gpu; 
  size_t *_found_fault_to_pattern_gpu; 
  
  // bad results 
  size_t *_b_pi_results_gpu; 
  size_t *_b_gate_results_gpu; 
  size_t *_b_po_results_gpu;   

  // memory
  size_t *_params_gpu; // used for constructing cuda_graph parameters
  size_t _num_const_paras = 7; // ask more memory than we actually need
  void _gpu_ask_copy_mem();
  void _free_all();
  bool _pull_up_head_frt_optimization = false; // init: false
  bool _pull_up_tail_end_optimization = false; // init: false
  size_t _start_head_frt_level = 21; // head-front
  size_t _end_head_frt_level; // head-front --> simply set before _start_tail_end_level
  size_t _start_tail_end_level; // tail-end
  size_t *_num_gates_per_level_gpu; 


  // parallel-cuda
  void _run_parallel_cuda();
  void _run_good_case_cuda(const size_t rd, const size_t bits);
  void _run_bad_case_cuda(const size_t rd, const size_t fault_num, const size_t bits);
  void _shift_to_correct_answer(size_t *results, const size_t bits, const size_t num_shift_gates);

  // parallel-cuda-graph
  void _construct_push_single_lvl_gate_KN(cudaStream_t stream1, cudaGraph_t graph,
                                  const size_t i, const size_t params_gpu_idx_idx, 
                                  const size_t params_gpu_idx_accum, 
                                  const size_t num_func_inputs,
                                  const size_t num_gates_this_level,
                                  const size_t accum, 
                                  const size_t num_blocks, 
                                  const size_t num_threads,                                   
                                  size_t **params_gpu_idx,
                                  void **kernelArgs_gate, 
                                  cudaGraphNode_t *kernelNode_gate, 
                                  cudaKernelNodeParams *kernelNode_gate_Params);

void _construct_push_multi_lvl_gate_KN(cudaStream_t stream1, cudaGraph_t graph, 
                                  const size_t level, 
                                  const size_t params_gpu_idx_idx, 
                                  const size_t params_gpu_idx_accum, 
                                  const size_t num_func_inputs,
                                  const size_t accum, 
                                  const size_t num_blocks, 
                                  const size_t num_threads, 
                                  const size_t start_level, 
                                  const size_t end_level,                                   
                                  size_t **params_gpu_idx,
                                  void **kernelArgs_gate, 
                                  cudaGraphNode_t *kernelNode_gate, 
                                  cudaKernelNodeParams *kernelNode_gate_Params);

  void _construct_cuda_graph(cudaStream_t stream1, cudaGraph_t graph, 
                            cudaGraphNode_t *kernelNode_PI, 
                            cudaKernelNodeParams *kernelNode_PI_Params,
                            cudaGraphNode_t *kernelNode_gate, 
                            cudaKernelNodeParams *kernelNode_gate_Params, 
                            cudaGraphNode_t *kernelNode_PO, 
                            cudaKernelNodeParams *kernelNode_PO_Params,
                            void **kernelArgs_PI,
                            void **kernelArgs_gate,
                            void **kernelArgs_PO,
                            size_t **params_gpu_idx,
                            size_t num_func_inputs);
  void _run_parallel_cuda_graph();

  void _run_good_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance);
  void _run_bad_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance);
  void _update_cuda_graph_params(cudaStream_t &stream1, size_t rd, size_t fault_num, size_t bad_case);

  

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