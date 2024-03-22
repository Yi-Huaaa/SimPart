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
// testing functions 
__global__ void helloCUDA();
__global__ void _run_gate_gpu(size_t *inputs);
// __global__ void _run_gate(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, const size_t rd, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t idx_accum, 
//   size_t pi_g_po_idx_accum, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);
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
                                size_t *_params_gpu);

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

__global__ void _set_params(size_t *_params_gpu, size_t rd, size_t fault_val, size_t idx_accum_sa, 
                            size_t wrong_gate, size_t bad_case, size_t max_gate_num_per_level, 
                            size_t idx_accum, size_t pi_g_po_idx_accum, 
                            size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);
__global__ void _update_params_gpu (size_t *_params_gpu_idx_address, size_t _num_paras, size_t limit_sz, 
                                    size_t new_rd, size_t new_fault_num, size_t new_bad_case);


// partition
__global__ void _run_gate_graph_partition(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                          size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                          size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                          size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                          size_t *_params_gpu,
                                          size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx);

__global__ void print_merged_FFRs_gpu (size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx, size_t sum_pi_gates_pos, size_t aa);

// for FSIM_DEBUG_GPU_INPUTS
__global__ void print_pi_gate_po_gate_type_gpu(size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_num_inputs_gpu, size_t num);
__global__ void print_patterns_gpu(size_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs);
__global__ void print_fault_gate_idx_gpu(size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, size_t _num_faults);


// print for check
// #define FSIM_DEBUG_PRINT_SIMULATION
// #define FSIM_DEBUG_PRINT_SIMULATION_GPU
// #define FSIM_DEBUG_PRINT_GRAPH
// #define FSIM_DEBUG_PRINT_FAULTS
// #define FSIM_DEBUG_PRINT_PATTERNS
// #define FSIM_DEBUG_PRINT_FAULT_TABLE
// #define FSIM_DEBUG_GPU_INPUTS

// #define CUDA_GRAPH_CHECK
// #define TEST_cuGraphExecNodeSetParams

// #define FSIM_LEVELIZAED_PARTITION

// #define FSIM_CHECK_FFR
#define FSIM_PARTITION_FFR // partition_FFR
// #define FSIM_PARTITION_CORRECTNESS



class CUDASimulator {

// constexpr size_t SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;
const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

// friend functions 
// friend __global__ void _run_gate(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, const size_t rd, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t idx_accum, size_t pi_g_po_idx_accum, size_t _num_inner_gates, size_t _num_POs);
friend __global__ void _run_gate(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                const size_t rd, const size_t fault_num, const size_t bad_case, 
                                const size_t max_gate_num_per_level, const size_t idx_accum);
                                
friend __global__ void _shift_to_correct_answer_gpu(size_t *outputs, size_t *_pi_gate_po_output_res_gpu, const size_t bits, const size_t SIZE_T_BITS, const size_t pre_gates_accum, const size_t num);
friend __global__ void _compare_g_b_results(const size_t j, const size_t _num_POs, const size_t rd, size_t *_g_po_results_gpu, size_t *_b_po_results_gpu, size_t *_found_fault_to_pattern_gpu);
friend __global__ void _run_gate_graph(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                size_t *_params_gpu);
friend __global__ void _change_params(size_t *_params_gpu, size_t rd, size_t fault_val, size_t idx_accum_sa, 
                              size_t wrong_gate, size_t bad_case, size_t max_gate_num_per_level, 
                              size_t idx_accum, size_t pi_g_po_idx_accum, 
                              size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);

friend __global__ void _run_gate_graph_partition(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                                size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                                size_t *_params_gpu,
                                                size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx);


public:
  // cuda simulator functions 
  enum class Mode { CUDA=0, CUDA_GRAPH, CUDA_GRAPH_PARTITION };

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

  size_t _num_faults; // for read file
  Fault *_faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  Pattern *_patterns;

  // Construct
  void _read_graph(std::istream &ckt);
  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

// ---- FFR parition
  void _find_split_merge_FFRs();
  void _DFS(std::vector<bool> &visited, size_t gate_idx, 
            std::vector<std::list<size_t>> &bucket_list, size_t depth,
            std::queue<size_t> &source_queue);

  void _merge_FFRs();  
  size_t depth_constraint_0 = 4;
  size_t width_constraint_0 = 512; 
  size_t _largest_FFR_level;
  // store the FFR on each level ->  which FFRs are on this FFR-level 
  std::vector<std::list<size_t>> FFR_levelization; 
  // compute number of FFRs in each level 
  std::vector<size_t> num_FFRs_in_this_level;
  bool _valid_and_merged_FFR(std::vector<std::list<size_t>> &bucket_lists_for_FFRs,
                            size_t FFR_level,
                            std::vector<std::vector<size_t>> &merged_FFR, 
                            bool &merged_FFR_full);
  // after merge
  size_t _num_FFRs;
  // merged_FFRs: used to store FFR, store the gate_idx merged in this FFR
  std::vector<std::vector<std::vector<std::vector<size_t>>>> merged_FFRs; 
  std::vector<size_t> merged_FFRs_level; // store the level of each FFRs after merged
  // used to help construct the PARTITION CUDA GRAPHs
  std::vector<std::set<size_t>> merged_FFRs_dependency; // recoed the "inputs" of each merged FFRs

  
// ---- FFR parition

  
  // Simulation
  size_t *_gate_inputs_list;
  size_t *_gate_inputs_list_start;
  size_t sum_inputs_gates;
  size_t *_found_fault_to_pattern;


// GPU pointers
  // CSR format: 
  // PI *_PIs_gpu; 
  // Gate *_gates_gpu; 
  // PO *_POs_gpu;
  // 3/20: new
  size_t *_pi_gate_po_gate_type_gpu;
  size_t *_pi_gate_po_output_res_gpu;
  size_t *_pi_gate_po_num_inputs_gpu;
  
  size_t *_gate_inputs_list_gpu;
  // size_t *_gate_inputs_list_from_gpu; // 0: PI, 1: PO
  size_t *_gate_inputs_list_start_gpu;

  // Pattern *_patterns_gpu;
  size_t *_patterns_gpu;

  // Fault *_faults_gpu;
  size_t *_fault_gate_idx_gpu;
  size_t *_fault_SA_fault_val_gpu;
  // 3/20: new

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
  size_t _num_paras = 6;
  void _gpu_ask_copy_mem();
  void _free_all();

// ---- FFR parition
  size_t *merged_FFRs_c; // CPU
  size_t *merged_FFRs_idx_c; // CPU
  size_t *merged_FFRs_gpu; // GPU
  size_t *merged_FFRs_gpu_idx; // GPU
  void _gpu_ask_copy_mem_partition();
// ---- FFR parition

// TESTING
// #ifdef TEST_cuGraphExecNodeSetParams  
size_t *_params_gpu_tmptmp; // used for testing --> DELETE!
void test_cuGraphExecNodeSetParams_cpu();
// #endif
// TESTING


  // parallel-cuda
  void _run_parallel_cuda();
  void _run_good_case_cuda(const size_t rd, const size_t bits);
  void _run_bad_case_cuda(const size_t rd, const size_t fault_num, const size_t bits);
  void _shift_to_correct_answer(size_t *results, const size_t bits, const size_t num_shift_gates);

  // parallel-cuda-graph
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
  void _update_cuda_graph_params(cudaStream_t &stream1, cudaGraphExec_t instance, 
                                size_t rd, size_t fault_num, size_t bad_case,
                                cudaGraphNode_t *kernelNode_PI, 
                                cudaKernelNodeParams *kernelNode_PI_Params,
                                cudaGraphNode_t *kernelNode_gate, 
                                cudaKernelNodeParams *kernelNode_gate_Params, 
                                cudaGraphNode_t *kernelNode_PO, 
                                cudaKernelNodeParams *kernelNode_PO_Params,
                                size_t start_idx, size_t end_idx);
 
  
  // parallel-cuda-graph-partition
  void _run_parallel_cuda_graph_partition();
  void _construct_cuda_graph_partition(cudaStream_t stream1, cudaGraph_t graph, 
                                      cudaGraphNode_t *kernelNode_PI, 
                                      cudaKernelNodeParams *kernelNode_PI_Params,
                                      cudaGraphNode_t *kernelNode_gate, 
                                      cudaKernelNodeParams *kernelNode_gate_Params, 
                                      cudaGraphNode_t *kernelNode_PO, 
                                      cudaKernelNodeParams *kernelNode_PO_Params,
                                      void **kernelArgs_PI,
                                      void **kernelArgs_gate,
                                      void **kernelArgs_PO,
                                      size_t **params_gpu_idx, size_t num_func_inputs);
  void _run_good_case_cuda_graph_partition(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance);
  void _run_bad_case_cuda_graph_partition(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance);  
                        
  
  
  

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