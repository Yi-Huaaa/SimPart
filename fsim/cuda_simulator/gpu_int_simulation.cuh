#ifndef GALPS_INT_GPU_SIMULATION_H
#define GALPS_INT_GPU_SIMULATION_H
#include <chrono>
#include <thread>
#include <assert.h>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility> 
#include <algorithm>
#include <cstddef>  // For int and SIZE_MAX
#include <cstdio>

#include <fsim/fsim.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <set>


// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// constexpr
constexpr int _NUM_THREADS = 1024;


// GPU functions pre-declarations
__device__ __forceinline__ void _apply_INV(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_AND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_OR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_XOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_NAND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_NOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_XNOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_MUX(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_CLKBUF(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_PI(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, const size_t pattern_val);
__device__ __forceinline__ void _apply_PO(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
__global__ void _run_gate_level(const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                          const int *_gateIdx_in_each_level_gpu, 
                          const int *_per_level_of_group_start_accum_level_gpu, const int k_level, 
                          const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          const size_t *_patterns_gpu, const size_t rd, 
                          const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                          const size_t fault_num, const size_t bad_case, 
                          const int num_gates_per_level, 
                          const int _num_PIs) ;
__global__ void _run_gate_part(const int _num_PIs,
                              const int _start_level,
                              const int _total_num_levels,
                              const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                              const int *_numGates_per_level_gpu_of_groups_gpu,
                              const int *_cones_partitioned_gpu, 
                              const int *_per_level_of_group_start_accum_gpu, 
                              const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                              const size_t *_patterns_gpu, const size_t rd, 
                              const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                              const size_t fault_num, const size_t bad_case);                          
__global__ void _write_and_shift_to_array_gpu(const int _num_PIs, 
                                              const int _num_inner_gates, 
                                              const int _num_POs, 
                                              const size_t bits, 
                                              const size_t *_pi_gate_po_output_res_gpu, const int total_num_gates,
                                              size_t *outputs_PIs, size_t *outputs_Gates, size_t *outputs_POs);
__global__ void _run_gate_level_1115(const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                          const int *_gateIdx_in_each_level_gpu, 
                          const int *_per_level_of_group_start_accum_level_gpu, const int *level_counter_gpu, 
                          const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          const size_t *_patterns_gpu, const size_t rd, 
                          const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                          const size_t fault_num, const size_t bad_case, 
                          const int *num_gates_per_level, 
                          const int _num_PIs);

// print function 
__global__  void _print_simulation_results(const int _num_PIs, 
                                          const int _num_inner_gates, 
                                          const int _num_POs, 
                                          const size_t *_g_pi_results_gpu,
                                          const size_t *_g_gate_results_gpu,
                                          const size_t *_g_po_results_gpu);                                                


// Forward declaration of CUDAPartitioner
class CUDAPartitioner;


class GALPS_GPUSimulator {
  const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

  friend class CUDAPartitioner;

public:
  // // Simulation - (1) levelization (2) cones 
  // By adopting partitioning, we have no need to:
    // (1) re-launching kernel level by level 
    // (2) doing grid level synchronization
  void run_gpu_simulator_cones_part_gpu(const int k,
                                        const int num_PIs, 
                                        const int num_inner_gates,
                                        const int num_POs, 
                                        const int sum_pi_gates_pos,
                                        const int num_pattern, 
                                        const size_t num_rounds,
                                        const int num_fault, 
                                        const int *_pi_gate_po_gate_type_gpu,
                                        const size_t *_patterns_gpu,
                                        const int *_fault_gate_idx_gpu,
                                        const size_t *_fault_SA_fault_val_gpu,
                                        size_t *_pi_gate_po_output_res_gpu,
                                        const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                        const int *_gateIdx_in_each_level_gpu, 
                                        const int *_per_level_of_group_start_accum_level_gpu,
                                        const int threshold_0_level,
                                        const int *_numGates_per_level_gpu_of_groups_gpu,
                                        const int *_per_level_of_group_start_accum_gpu,
                                        const int *_cones_partitioned_gpu,
                                        const int _total_num_levels,
                                        const int *_invAdj_gpu,
                                        const int *_invAdj_index_table_gpu,
                                        const std::vector<Pattern> _patterns,
                                        size_t *_g_pi_results_gpu,
                                        size_t *_g_gate_results_gpu,
                                        size_t *_g_po_results_gpu,
                                        size_t *_b_pi_results_gpu,
                                        size_t *_b_gate_results_gpu,
                                        size_t *_b_po_results_gpu,
                                        int *_found_fault_to_pattern_gpu,
                                        const size_t NUM_SIMULATION_RDS) {
    // Get vars   
    _k = k;       
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    _threshold_0_level = threshold_0_level;
    // printf("Get inside run_gpu_simulator_cones_part_gpu\n");
    
    // Run simulation
    for (size_t rd = 0; rd < NUM_SIMULATION_RDS; rd++) {
      // Levelization part 
      if (_threshold_0_level != -1) {
        _run_cones_gates_level_gpu(_gateIdx_in_each_level,
                                  _per_level_of_group_start_accum_level_gpu,
                                  _gateIdx_in_each_level_gpu, 
                                  _threshold_0_level+1,
                                  _invAdj_gpu,
                                  _invAdj_index_table_gpu,
                                  _pi_gate_po_gate_type_gpu, 
                                  _patterns_gpu,
                                  _patterns,
                                  _fault_gate_idx_gpu,
                                  _fault_SA_fault_val_gpu,
                                  _g_pi_results_gpu,
                                  _g_gate_results_gpu,
                                  _g_po_results_gpu,
                                  _b_pi_results_gpu,
                                  _b_gate_results_gpu,
                                  _b_po_results_gpu,
                                  _found_fault_to_pattern_gpu,
                                  _pi_gate_po_output_res_gpu);      
      }

      // Partitioned part 
      _run_cones_gates_part_gpu(_numGates_per_level_gpu_of_groups_gpu,
                                _per_level_of_group_start_accum_gpu,
                                _cones_partitioned_gpu, 
                                _threshold_0_level+1, 
                                _total_num_levels,
                                _invAdj_gpu,
                                _invAdj_index_table_gpu,
                                _pi_gate_po_gate_type_gpu, 
                                _patterns_gpu,
                                _patterns,
                                _fault_gate_idx_gpu,
                                _fault_SA_fault_val_gpu,
                                _g_pi_results_gpu,
                                _g_gate_results_gpu,
                                _g_po_results_gpu,
                                _b_pi_results_gpu,
                                _b_gate_results_gpu,
                                _b_po_results_gpu,
                                _found_fault_to_pattern_gpu,
                                _pi_gate_po_output_res_gpu);
    } // NUM_SIMULATION_RDS
  }

  void run_gpu_simulator_cones_part_gpu_cuda_graph(const int k,
                                                  const int num_PIs, 
                                                  const int num_inner_gates,
                                                  const int num_POs, 
                                                  const int sum_pi_gates_pos,
                                                  const int num_pattern, 
                                                  const size_t num_rounds,
                                                  const int num_fault, 
                                                  const int *_pi_gate_po_gate_type_gpu,
                                                  const size_t *_patterns_gpu,
                                                  const int *_fault_gate_idx_gpu,
                                                  const size_t *_fault_SA_fault_val_gpu,
                                                  size_t *_pi_gate_po_output_res_gpu,
                                                  const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                                  const int *_gateIdx_in_each_level_gpu, 
                                                  const int *_per_level_of_group_start_accum_level_gpu,
                                                  const int threshold_0_level,
                                                  const int *_numGates_per_level_gpu_of_groups_gpu,
                                                  const int *_per_level_of_group_start_accum_gpu,
                                                  const int *_cones_partitioned_gpu,
                                                  const int _total_num_levels,
                                                  const int *_invAdj_gpu,
                                                  const int *_invAdj_index_table_gpu,
                                                  const std::vector<Pattern> _patterns,
                                                  size_t *_g_pi_results_gpu,
                                                  size_t *_g_gate_results_gpu,
                                                  size_t *_g_po_results_gpu,
                                                  size_t *_b_pi_results_gpu,
                                                  size_t *_b_gate_results_gpu,
                                                  size_t *_b_po_results_gpu,
                                                  int *_found_fault_to_pattern_gpu,
                                                  const size_t NUM_SIMULATION_RDS) {
    // Get vars   
    _k = k;       
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    _threshold_0_level = threshold_0_level;
    // printf("Get inside run_gpu_simulator_cones_part_gpu_cuda_graph\n");
    
    // Run simulation
    for (size_t rd = 0; rd < NUM_SIMULATION_RDS; rd++) {
      _run_cones_gates_level_rep_cuda_graph(_numGates_per_level_gpu_of_groups_gpu,
                                            _per_level_of_group_start_accum_gpu,
                                            _cones_partitioned_gpu,
                                            _total_num_levels,
                                            _gateIdx_in_each_level,
                                            _per_level_of_group_start_accum_level_gpu,
                                            _gateIdx_in_each_level_gpu, 
                                            _threshold_0_level+1,
                                            _invAdj_gpu,
                                            _invAdj_index_table_gpu,
                                            _pi_gate_po_gate_type_gpu, 
                                            _patterns_gpu,
                                            _patterns,
                                            _fault_gate_idx_gpu,
                                            _fault_SA_fault_val_gpu,
                                            _g_pi_results_gpu,
                                            _g_gate_results_gpu,
                                            _g_po_results_gpu,
                                            _b_pi_results_gpu,
                                            _b_gate_results_gpu,
                                            _b_po_results_gpu,
                                            _found_fault_to_pattern_gpu,
                                            _pi_gate_po_output_res_gpu);  
    } // NUM_SIMULATION_RDS
  }  

private:


  // void _run_level_gates_gpu();
  // vars
  int _k;
  int _num_PIs; 
  int _num_inner_gates;
  int _num_POs; 
  int _sum_pi_gates_pos;
  int _num_pattern; 
  size_t _num_rounds; 
  int _num_fault; 
  int _threshold_0_level; // levelize ending level (included)

  // original level by level 
  void _run_cones_gates_level_gpu(const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                const int *_per_level_of_group_start_accum_level_gpu,
                                const int *_gateIdx_in_each_level_gpu,
                                const int _end_level,
                                const int *_invAdj_gpu,
                                const int *_invAdj_index_table_gpu,
                                const int *_pi_gate_po_gate_type_gpu,
                                const size_t *_patterns_gpu,
                                const std::vector<Pattern> _patterns,
                                const int *_fault_gate_idx_gpu,
                                const size_t *_fault_SA_fault_val_gpu,
                                size_t *_g_pi_results_gpu,
                                size_t *_g_gate_results_gpu,
                                size_t *_g_po_results_gpu,
                                size_t *_b_pi_results_gpu,
                                size_t *_b_gate_results_gpu,
                                size_t *_b_po_results_gpu,
                                int *_found_fault_to_pattern_gpu,
                                size_t *_pi_gate_po_output_res_gpu);   
  void _run_cones_good_case_level_gpu(const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                    const int *_per_level_of_group_start_accum_level_gpu,
                                    const int *_gateIdx_in_each_level_gpu,
                                    const int _end_level,
                                    const int *_invAdj_gpu,
                                    const int *_invAdj_index_table_gpu,
                                    const int *_pi_gate_po_gate_type_gpu,
                                    const size_t *_patterns_gpu,
                                    const int *_fault_gate_idx_gpu,
                                    const size_t *_fault_SA_fault_val_gpu,
                                    const size_t bits,
                                    const size_t rd, 
                                    size_t *_g_pi_results_gpu,
                                    size_t *_g_gate_results_gpu,
                                    size_t *_g_po_results_gpu,
                                    size_t *_pi_gate_po_output_res_gpu);
  void _run_cones_bad_case_level_gpu(std::vector<ElementBase<int, int>> &_Gates,
                                  const Fault<int> &fault, 
                                  const Pattern pattern,
                                  const size_t bits,
                                  std::vector<size_t> &_b_pi_results,
                                  std::vector<size_t> &_b_gate_results,
                                  std::vector<size_t> &_b_po_results);

  // partitioning                                
  void _run_cones_gates_part_gpu(const int *_numGates_per_level_gpu_of_groups_gpu,
                                const int *_per_level_of_group_start_accum_gpu,
                                const int *_cones_partitioned_gpu,
                                const int _start_level,
                                const int _total_num_levels,
                                const int *_invAdj_gpu,
                                const int *_invAdj_index_table_gpu,
                                const int *_pi_gate_po_gate_type_gpu,
                                const size_t *_patterns_gpu,
                                const std::vector<Pattern> _patterns,    
                                const int *_fault_gate_idx_gpu,
                                const size_t *_fault_SA_fault_val_gpu,                                                   
                                size_t *_g_pi_results_gpu,
                                size_t *_g_gate_results_gpu,
                                size_t *_g_po_results_gpu,
                                size_t *_b_pi_results_gpu,
                                size_t *_b_gate_results_gpu,
                                size_t *_b_po_results_gpu,
                                int *_found_fault_to_pattern_gpu,
                                size_t *_pi_gate_po_output_res_gpu);
  void _run_cones_good_case_part_gpu(const int *_numGates_per_level_gpu_of_groups_gpu,
                                    const int *_invAdj_gpu,
                                    const int *_invAdj_index_table_gpu,  
                                    const int *_cones_partitioned_gpu,
                                    const int *_per_level_of_group_start_accum_gpu,
                                    const int _start_level, 
                                    const int _total_num_levels,
                                    const int *_pi_gate_po_gate_type_gpu,
                                    const size_t *_patterns_gpu,
                                    const int *_fault_gate_idx_gpu,
                                    const size_t *_fault_SA_fault_val_gpu,
                                    const size_t bits,
                                    const size_t rd, 
                                    size_t *_g_pi_results_gpu,
                                    size_t *_g_gate_results_gpu,
                                    size_t *_g_po_results_gpu,
                                    size_t *_pi_gate_po_output_res_gpu);

  void _run_cones_gates_level_rep_cuda_graph(const int *_numGates_per_level_gpu_of_groups_gpu,
                                              const int *_per_level_of_group_start_accum_gpu,
                                              const int *_cones_partitioned_gpu,
                                              const int _total_num_levels,
                                              const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                              const int *_per_level_of_group_start_accum_level_gpu,
                                              const int *_gateIdx_in_each_level_gpu,
                                              const int _end_level,
                                              const int *_invAdj_gpu,
                                              const int *_invAdj_index_table_gpu,
                                              const int *_pi_gate_po_gate_type_gpu,
                                              const size_t *_patterns_gpu,
                                              const std::vector<Pattern> _patterns,
                                              const int *_fault_gate_idx_gpu,
                                              const size_t *_fault_SA_fault_val_gpu,
                                              size_t *_g_pi_results_gpu,
                                              size_t *_g_gate_results_gpu,
                                              size_t *_g_po_results_gpu,
                                              size_t *_b_pi_results_gpu,
                                              size_t *_b_gate_results_gpu,
                                              size_t *_b_po_results_gpu,
                                              int *_found_fault_to_pattern_gpu,
                                              size_t *_pi_gate_po_output_res_gpu);
    void _run_cones_gates_level_rep_cuda_graph_CORRECT_VERSION(const int *_numGates_per_level_gpu_of_groups_gpu,
                                                                  const int *_per_level_of_group_start_accum_gpu,
                                                                  const int *_cones_partitioned_gpu,
                                                                  const int _total_num_levels,
                                                                  const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                                                                  const int *_per_level_of_group_start_accum_level_gpu,
                                                                  const int *_gateIdx_in_each_level_gpu,
                                                                  const int _end_level,
                                                                  const int *_invAdj_gpu,
                                                                  const int *_invAdj_index_table_gpu,
                                                                  const int *_pi_gate_po_gate_type_gpu,
                                                                  const size_t *_patterns_gpu,
                                                                  const std::vector<Pattern> _patterns,
                                                                  const int *_fault_gate_idx_gpu,
                                                                  const size_t *_fault_SA_fault_val_gpu,
                                                                  size_t *_g_pi_results_gpu,
                                                                  size_t *_g_gate_results_gpu,
                                                                  size_t *_g_po_results_gpu,
                                                                  size_t *_b_pi_results_gpu,
                                                                  size_t *_b_gate_results_gpu,
                                                                  size_t *_b_po_results_gpu,
                                                                  int *_found_fault_to_pattern_gpu,
                                                                  size_t *_pi_gate_po_output_res_gpu);

    void construct_cuda_graph(
                            cudaStream_t stream1, cudaGraph_t graph, 
                            cudaGraphNode_t *while_node, 
                            cudaGraph_t *bodyGraph,
                            cudaGraphNode_t *dsp_nodes, 
                            cudaGraphNode_t *rap_node, 
                            cudaKernelNodeParams *dsp_nodes_params, 
                            cudaKernelNodeParams *rsp_node_params,
                            void **kernelArgs_dsp,
                            void **kernelArgs_rap,      
                            const int *_numGates_per_level_gpu_of_groups_gpu,
                            const int *_per_level_of_group_start_accum_gpu,
                            const int *_cones_partitioned_gpu,
                            const int _total_num_levels,
                            const std::vector<std::vector<int>> &_gateIdx_in_each_level,
                            const int *_per_level_of_group_start_accum_level_gpu,
                            const int *_gateIdx_in_each_level_gpu,
                            const int _end_level,
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu,
                            const int *_pi_gate_po_gate_type_gpu,
                            const size_t *_patterns_gpu,
                            const std::vector<Pattern> _patterns,
                            const int *_fault_gate_idx_gpu,
                            const size_t *_fault_SA_fault_val_gpu,
                            size_t *_g_pi_results_gpu,
                            size_t *_g_gate_results_gpu,
                            size_t *_g_po_results_gpu,
                            size_t *_b_pi_results_gpu,
                            size_t *_b_gate_results_gpu,
                            size_t *_b_po_results_gpu,
                            int *_found_fault_to_pattern_gpu,
                            size_t *_pi_gate_po_output_res_gpu);

};

#endif  // GALPS_INT_GPU_SIMULATION_H