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
#include "./gpu_int_simulation.cuh"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// System includes
#include <cassert>
#include <cstdio>

// CUDA runtime

// helper functions and utilities to work with CUDA
// #include <helper_cuda.h>
// #include <helper_functions.h>

// #define GPU_PART_DEBUG_PRINT_SIMULATION // print simulation resutls 
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



void GALPS_GPUSimulator::_run_cones_gates_level_gpu(const std::vector<std::vector<int>> &_gateIdx_in_each_level,
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
                                                    size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_cones_gates_level_gpu();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
    _run_cones_good_case_level_gpu(_gateIdx_in_each_level,
                                _per_level_of_group_start_accum_level_gpu,
                                _gateIdx_in_each_level_gpu, 
                                _end_level,
                                _invAdj_gpu, 
                                _invAdj_index_table_gpu, 
                                _pi_gate_po_gate_type_gpu, 
                                _patterns_gpu,
                                _fault_gate_idx_gpu, 
                                _fault_SA_fault_val_gpu,
                                num_testcases_this_round,  // bits 
                                rd, 
                                _g_pi_results_gpu, _g_gate_results_gpu, _g_po_results_gpu,
                                _pi_gate_po_output_res_gpu);
    // // bad simulation (fault simulation)
    // for (size_t j = 0; j < _faults.size(); j++) {
    //   _run_cones_bad_case_level_gpu(_Gates, _faults[j], _patterns[rd], 
    //                       num_testcases_this_round,
    //                       _b_pi_results, _b_gate_results, _b_po_results);

    //   size_t found_fault = 0;
    //   for (size_t i = 0; i < _num_POs; i++) {
    //     if (_g_po_results[i] != _b_po_results[i]) {
    //       found_fault = 1;
    //       break;
    //     }
    //   }
    //   // Record whether fault can be found
    //   _found_fault_to_pattern[2 * j] = found_fault;
    //   // Record which pattern found the fault
    //   _found_fault_to_pattern[2 * j + 1] = rd;
    // }
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    if (rd == 0) {
      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
      
      std::cout << "GOOD resutls ans:" << std::endl;
      _print_simulation_results <<< 1, 1 >>> (_num_PIs, _num_inner_gates, _num_POs,
                                              _g_pi_results_gpu,
                                              _g_gate_results_gpu,
                                              _g_po_results_gpu);
      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: _print_simulation_results cudaDeviceSynchronize - Failure");
    }
#endif
  }
}


void GALPS_GPUSimulator::_run_cones_good_case_level_gpu(const std::vector<std::vector<int>> &_gateIdx_in_each_level,
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
                                                        size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    printf("Getting inside _run_cones_good_case_level_gpu\n");
#endif 

  const size_t fault_num = 0; const size_t bad_case = 0;
  int num_blocks, num_threads;

  for (int level = 0; level < _end_level; level++) {
    const int num_gates_per_level = (_gateIdx_in_each_level[level].size());
    num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                  (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                  (1);
    num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                  (_NUM_THREADS) : 
                  (num_gates_per_level);

    _run_gate_level <<< num_blocks, num_threads >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
                                                    _gateIdx_in_each_level_gpu, 
                                                    _per_level_of_group_start_accum_level_gpu, level,
                                                    _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                    _patterns_gpu, rd, 
                                                    _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                    fault_num, bad_case, 
                                                    num_gates_per_level, 
                                                    _num_PIs);


#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif      
  }
}

void GALPS_GPUSimulator::_run_cones_bad_case_level_gpu(std::vector<ElementBase<int, int>> &_Gates,
                                                const Fault<int> &fault, 
                                                const Pattern pattern,
                                                const size_t bits,
                                                std::vector<size_t> &_b_pi_results,
                                                std::vector<size_t> &_b_gate_results,
                                                std::vector<size_t> &_b_po_results) {
  printf("_run_cones_bad_case_level_gpu\n");

}



// -------------------------------------------------



void GALPS_GPUSimulator::_run_cones_gates_part_gpu(const int *_numGates_per_level_gpu_of_groups_gpu,
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
                                              size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_cones_gates_part_gpu();\n";
#endif

  // Simulation

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
    _run_cones_good_case_part_gpu(_numGates_per_level_gpu_of_groups_gpu,
                                  _invAdj_gpu, 
                                  _invAdj_index_table_gpu, 
                                  _cones_partitioned_gpu, 
                                  _per_level_of_group_start_accum_gpu,
                                  _start_level, 
                                  _total_num_levels,
                                  _pi_gate_po_gate_type_gpu, 
                                  _patterns_gpu,
                                  _fault_gate_idx_gpu, 
                                  _fault_SA_fault_val_gpu,
                                  num_testcases_this_round, // bits 
                                  rd, 
                                  _g_pi_results_gpu, _g_gate_results_gpu, _g_po_results_gpu,
                                  _pi_gate_po_output_res_gpu);
    
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    if (rd == 0) {
      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
      
      std::cout << "GOOD resutls ans:" << std::endl;
      _print_simulation_results <<< 1, 1 >>> (_num_PIs, _num_inner_gates, _num_POs,
                                              _g_pi_results_gpu,
                                              _g_gate_results_gpu,
                                              _g_po_results_gpu);
      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: _print_simulation_results cudaDeviceSynchronize - Failure");
    }
#endif
  }
}


void GALPS_GPUSimulator::_run_cones_good_case_part_gpu(const int *_numGates_per_level_gpu_of_groups_gpu,
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
                                                       size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    printf("Getting inside _run_cones_good_case_part_gpu\n");
#endif 

  // Copy data to the device and launch kernels in different streams
  const size_t fault_num = 0; const size_t bad_case = 0;
  // printf("_start_level = %d, _total_num_levels = %d\n", _start_level, _total_num_levels); // remove 

  _run_gate_part <<< _k, _NUM_THREADS >>> (_num_PIs, _start_level, _total_num_levels, 
                                          _invAdj_gpu, _invAdj_index_table_gpu,
                                          _numGates_per_level_gpu_of_groups_gpu,
                                          _cones_partitioned_gpu, 
                                          _per_level_of_group_start_accum_gpu, 
                                          _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                          _patterns_gpu, rd, 
                                          _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                          fault_num, bad_case);

  cudaCheckErrors("CUDA: _run_gate_part launch- Failure");
  cudaDeviceSynchronize(); // remove


  // shift and copy answer to the good results
  int num_blocks = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                (_sum_pi_gates_pos + _NUM_THREADS - 1)/_NUM_THREADS : 
                (1);//1;
  int num_threads = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                (_NUM_THREADS) : 
                (_sum_pi_gates_pos);    
  _write_and_shift_to_array_gpu <<< num_blocks, num_threads >>> (_num_PIs, 
                                                                _num_inner_gates, 
                                                                _num_POs,
                                                                bits, 
                                                                _pi_gate_po_output_res_gpu,
                                                                _sum_pi_gates_pos,
                                                                _g_pi_results_gpu,
                                                                _g_gate_results_gpu,
                                                                _g_po_results_gpu);
  cudaCheckErrors("CUDA: _write_and_shift_to_array_gpu launch- Failure");
                           

#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
#endif  
}



// -------------------------------------
// -------------------------------------
// -------------------------------------
// -------------------------------------
// -------------------------------------
// -------------------------------------




void GALPS_GPUSimulator::_run_cones_gates_level_rep_cuda_graph_CORRECT_VERSION(const int *_numGates_per_level_gpu_of_groups_gpu,
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
                                                              size_t *_pi_gate_po_output_res_gpu) {
  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    const size_t fault_num = 0; const size_t bad_case = 0;
    int num_blocks, num_threads;

    for (int level = 0; level < _end_level; level++) {
      // printf("DSP level = %d\n", level);
      const int num_gates_per_level = (_gateIdx_in_each_level[level].size());
      num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                    (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                    (1);
      num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                    (_NUM_THREADS) : 
                    (num_gates_per_level);

      _run_gate_level <<< num_blocks, num_threads >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
                                                      _gateIdx_in_each_level_gpu, 
                                                      _per_level_of_group_start_accum_level_gpu, level,
                                                      _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                      _patterns_gpu, rd, 
                                                      _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                      fault_num, bad_case, 
                                                      num_gates_per_level, 
                                                      _num_PIs);   
    }


  // _start_level = _end_level;
  _run_gate_part <<< _k, _NUM_THREADS >>> (_num_PIs, _end_level, _total_num_levels, 
                                          _invAdj_gpu, _invAdj_index_table_gpu,
                                          _numGates_per_level_gpu_of_groups_gpu,
                                          _cones_partitioned_gpu, 
                                          _per_level_of_group_start_accum_gpu, 
                                          _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                          _patterns_gpu, rd, 
                                          _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                          fault_num, bad_case);


    const size_t bits = num_testcases_this_round;
    // shift and copy answer to the good results
    num_blocks = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                  (_sum_pi_gates_pos + _NUM_THREADS - 1)/_NUM_THREADS : 
                  (1);//1;
    num_threads = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                  (_NUM_THREADS) : 
                  (_sum_pi_gates_pos);    
    _write_and_shift_to_array_gpu <<< num_blocks, num_threads >>> (_num_PIs, 
                                                                  _num_inner_gates, 
                                                                  _num_POs,
                                                                  bits, 
                                                                  _pi_gate_po_output_res_gpu,
                                                                  _sum_pi_gates_pos,
                                                                  _g_pi_results_gpu,
                                                                  _g_gate_results_gpu,
                                                                  _g_po_results_gpu);
  }
}


__global__ void testing() {
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  
  printf("t_idx = %d\n", t_idx);
}

double hhh(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}
void dumpCudaGraphToDot(cudaGraph_t graph, const char* fileName) {
  // Use the DOT print function to dump the graph
  cudaError_t err = cudaGraphDebugDotPrint(graph, fileName, cudaGraphDebugDotFlagsVerbose);

  if (err != cudaSuccess) {
      std::cerr << "Failed to dump CUDA graph to DOT file: " << cudaGetErrorString(err) << std::endl;
  } else {
      std::cout << "CUDA Graph dumped to " << fileName << std::endl;
  }
}


// __global__ void doWhileLoopKernel(char *dPtr, cudaGraphConditionalHandle handle, int *level_counter_gpu, 
//                                   int *gateIdx_in_each_level_copy_gpu, 
//                                   int *num_gates_per_level_gpu, int *d_numBlocks, int *d_numThreads, int *bb_m, int*tt_m)
// {
//     if (--(*dPtr) == 0) {
//       cudaGraphSetConditional(handle, 0);
//     }
//     (*level_counter_gpu)++;
//     (*num_gates_per_level_gpu) = gateIdx_in_each_level_copy_gpu[*level_counter_gpu];

//     (*bb_m)  = ((*num_gates_per_level_gpu) > 1024) ? 
//               ((*num_gates_per_level_gpu) + 1024 - 1)/1024 : 
//               (1);
//     (*tt_m) = ((*num_gates_per_level_gpu) > 1024) ? 
//               (1024) : 
//               ((*num_gates_per_level_gpu));


//     // printf("GPU: counter = %d, level_counter_gpu = %d, (*num_gates_per_level_gpu) = %d, bb_m = %d, tt_m = %d\n", 
//     //       *dPtr, *level_counter_gpu, *num_gates_per_level_gpu, *bb_m, *tt_m);
//     // for (int i = 0; i < 77; i++) {
//     //   printf("gateIdx_in_each_level_copy_gpu[%d] = %d\n", i, gateIdx_in_each_level_copy_gpu[i]);
//     // }
// }

__global__ void doWhileLoopKernel(char *dPtr, cudaGraphConditionalHandle handle) {
  // if (--(*dPtr) == 0) {
  //   cudaGraphSetConditional(handle, 0);
  // }
  // printf("GPU: counter = %d\n", *dPtr);
  if (*dPtr <= 0) {
    cudaGraphSetConditional(handle, 0);  
    // printf("GPU: Breaking loop, counter = %d\n", *dPtr);
  } else {
    --(*dPtr);  // 更新计数器
    // printf("GPU: counter decremented to %d\n", *dPtr);
  }  
}

void GALPS_GPUSimulator::construct_cuda_graph(
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
                                            size_t *_pi_gate_po_output_res_gpu) {

  // construct dsp nodes
  int num_blocks, num_threads = 0;
  const int rd = 0, fault_num = 0, bad_case = 0;
  for (int level = 0; level < _end_level; level++) {
    const int num_gates_per_level = (_gateIdx_in_each_level[level].size());
    num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                  (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                  (1);
    num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                  (_NUM_THREADS) : 
                  (num_gates_per_level);

    kernelArgs_dsp[15*level+0] = (void *)&_invAdj_gpu;
    kernelArgs_dsp[15*level+1] = (void *)&_invAdj_index_table_gpu;
    kernelArgs_dsp[15*level+2] = (void *)&_gateIdx_in_each_level_gpu;
    kernelArgs_dsp[15*level+3] = (void *)&_per_level_of_group_start_accum_level_gpu;
    kernelArgs_dsp[15*level+4] = (void *)&level;
    kernelArgs_dsp[15*level+5] = (void *)&_pi_gate_po_gate_type_gpu;
    kernelArgs_dsp[15*level+6] = (void *)&_pi_gate_po_output_res_gpu;
    kernelArgs_dsp[15*level+7] = (void *)&_patterns_gpu;
    kernelArgs_dsp[15*level+8] = (void *)&rd;
    kernelArgs_dsp[15*level+9] = (void *)&_fault_gate_idx_gpu;
    kernelArgs_dsp[15*level+10] = (void *)&_fault_SA_fault_val_gpu;
    kernelArgs_dsp[15*level+11] = (void *)&fault_num;
    kernelArgs_dsp[15*level+12] = (void *)&bad_case;
    kernelArgs_dsp[15*level+13] = (void *)&num_gates_per_level;
    kernelArgs_dsp[15*level+14] = (void *)&_num_PIs;

    dsp_nodes_params[level].func = (void *)_run_gate_level;
    dsp_nodes_params[level].gridDim = dim3(num_blocks, 1, 1);
    dsp_nodes_params[level].blockDim = dim3(num_threads, 1, 1);
    dsp_nodes_params[level].sharedMemBytes = 0;
    dsp_nodes_params[level].kernelParams = (void **)&(kernelArgs_dsp[15*level]);
    dsp_nodes_params[level].extra = NULL;  
    cudaCheckErrors("dsp_nodes_params failed");
    
    // push node: gate
    // cudaGraphAddKernelNode(&dsp_nodes[level], graph, NULL, 0, &dsp_nodes_params[level]);
    cudaGraphAddKernelNode(&dsp_nodes[level], *bodyGraph, NULL, 0, &dsp_nodes_params[level]);
    
    cudaCheckErrors("Adding kernelNode_gate failed");
  }


  // construct rap nodes
  kernelArgs_rap[0] = (void *)&_num_PIs;
  kernelArgs_rap[1] = (void *)&_end_level;
  kernelArgs_rap[2] = (void *)&_total_num_levels;
  kernelArgs_rap[3] = (void *)&_invAdj_gpu;
  kernelArgs_rap[4] = (void *)&_invAdj_index_table_gpu;
  kernelArgs_rap[5] = (void *)&_numGates_per_level_gpu_of_groups_gpu;
  kernelArgs_rap[6] = (void *)&_cones_partitioned_gpu;
  kernelArgs_rap[7] = (void *)&_per_level_of_group_start_accum_gpu;
  kernelArgs_rap[8] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_rap[9] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_rap[10] = (void *)&_patterns_gpu;
  kernelArgs_rap[11] = (void *)&rd;
  kernelArgs_rap[12] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_rap[13] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_rap[14] = (void *)&fault_num;
  kernelArgs_rap[15] = (void *)&bad_case;

  rsp_node_params->func = (void *)_run_gate_part;
  rsp_node_params->gridDim = dim3(_k, 1, 1); // Replace `_k` with your desired block configuration
  rsp_node_params->blockDim = dim3(_NUM_THREADS, 1, 1); // Replace `_NUM_THREADS` with your desired thread configuration
  rsp_node_params->sharedMemBytes = 0;
  rsp_node_params->kernelParams = (void **)(kernelArgs_rap);
  rsp_node_params->extra = NULL;

  // Add kernel node to the main graph
  cudaGraphAddKernelNode(rap_node, *bodyGraph, NULL, 0, rsp_node_params);
  cudaCheckErrors("Adding rap_node failed");
  

  // for (int level = 0; level < _end_level-1; level++) {
  //   cudaGraphAddDependencies(graph, &dsp_nodes[level], &dsp_nodes[level+1], 1);
  //   cudaCheckErrors("cudaGraphAddDependencies dsp_nodes[level] -> dsp_nodes[level+1]");  
  // }
  // cudaGraphAddDependencies(graph, &dsp_nodes[_end_level-1], rap_node, 1);
  // cudaCheckErrors("cudaGraphAddDependencies dsp_nodes[_end_level-1] -> rap_node");
}



void GALPS_GPUSimulator::_run_cones_gates_level_rep_cuda_graph(
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
                                                size_t *_pi_gate_po_output_res_gpu) {
  // printf("Starting CUDA Graph...\n");

  cudaStream_t stream1;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaCheckErrors("cudaStreamCreateWithFlags failure");

  
  cudaGraph_t graph; // Create the parent graph
  cudaGraphCreate(&graph, 0);
  cudaGraph_t subgraph; // Create the subgraph (loop body)
  cudaGraphCreate(&subgraph, 0);
  // cudaGraph_t     graph;
  // cudaGraphCreate(&graph, 0); // create the graph
  cudaCheckErrors("cudaGraphCreate failure");
  cudaGraphExec_t graphExec;
  cudaGraphNode_t *dsp_nodes = (cudaGraphNode_t *)malloc(_end_level*sizeof(cudaGraphNode_t));
  cudaGraphNode_t rap_node;
  cudaCheckErrors("cudaGraphNode_t failure");

  cudaKernelNodeParams *dsp_nodes_params = (cudaKernelNodeParams *)malloc(_end_level*sizeof(cudaKernelNodeParams));
  cudaKernelNodeParams rsp_node_params {0};
  cudaCheckErrors("cudaKernelNodeParams failure");

  void **kernelArgs_dsp = (void **)malloc(_end_level*15*sizeof(void *));
  void **kernelArgs_rap = (void **)malloc(16*sizeof(void *));
  cudaCheckErrors("kernelArgs_dsp failure");

  // Construct conditional node 
  char *dPtr;
  cudaMalloc((void**)&dPtr, 1);
  cudaMemset(dPtr, _num_rounds, 1);
  cudaGraphNode_t while_node;
                        
  // Create a conditional while node
  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

  cudaGraphNodeParams while_params = { cudaGraphNodeTypeConditional };
  while_params.conditional.handle = handle;
  while_params.conditional.type = cudaGraphCondTypeWhile;
  while_params.conditional.size = 1;  
  cudaGraphAddNode(&while_node, graph, NULL, 0, &while_params);

  cudaGraph_t bodyGraph = while_params.conditional.phGraph_out[0];

  construct_cuda_graph(stream1, graph, &while_node, &bodyGraph, 
                      dsp_nodes, &rap_node, 
                      dsp_nodes_params, &rsp_node_params, 
                      kernelArgs_dsp, kernelArgs_rap,
                      _numGates_per_level_gpu_of_groups_gpu,
                      _per_level_of_group_start_accum_gpu,
                      _cones_partitioned_gpu,
                      _total_num_levels,
                      _gateIdx_in_each_level,
                      _per_level_of_group_start_accum_level_gpu,
                      _gateIdx_in_each_level_gpu,
                      _end_level,
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
                   
  // doWhileLoopKernel
  cudaGraphNode_t dw_node;
  cudaCheckErrors("dowhile_node failure");
  cudaKernelNodeParams dw_node_params {0};
  cudaCheckErrors("dw_node_params failure");
  void **kernelArgs_dw = (void **)malloc(2*sizeof(void *));
  cudaCheckErrors("kernelArgs_dw failure");
  // construct rap nodes
  kernelArgs_dw[0] = (void *)&dPtr;
  kernelArgs_dw[1] = (void *)&handle;
  dw_node_params.func = (void *)doWhileLoopKernel;
  dw_node_params.gridDim = dim3(1, 1, 1); // Replace `_k` with your desired block configuration
  dw_node_params.blockDim = dim3(1, 1, 1); // Replace `_NUM_THREADS` with your desired thread configuration
  dw_node_params.sharedMemBytes = 0;
  dw_node_params.kernelParams = (void **)(kernelArgs_dw);
  dw_node_params.extra = NULL;

  // Add kernel node to the main graph
  cudaGraphAddKernelNode(&dw_node, bodyGraph, NULL, 0, &dw_node_params);
  cudaCheckErrors("Adding dw_node failed");

  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
  // cudaGraphLaunch(graphExec, stream1);
  // cudaDeviceSynchronize();


  // auto start1115 = std::chrono::steady_clock::now();

    cudaGraphLaunch(graphExec, stream1);

  cudaDeviceSynchronize();

  // auto end1115 = std::chrono::steady_clock::now();
  // std::chrono::duration<double>  duration_simulation = (end1115 - start1115);
  // std::cout << "1115run_simulator: " <<  hhh(((duration_simulation.count()))*1000, 0.001) << "\n";

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);

}










// ---------------------------------------------------------------------------------------------------------



// GPU functions

__device__ __forceinline__ void _apply_INV(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_AND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_OR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_XOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_NAND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_NOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_XNOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_MUX(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  const size_t a = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  const size_t b = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+1]];
  const size_t s = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+2]];

  size_t ret = ((s & b) | ( a & (!s)));

  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_CLKBUF(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_PI(const int gate_idx, const int *_invAdj_gpu, 
                                          const int *_invAdj_index_table_gpu, 
                                          size_t *_pi_gate_po_output_res_gpu, 
                                          const size_t pattern_val) {
  _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
  // printf("_apply_PI: gate_idx = %d, pattern_val = %lu, %lu\n", gate_idx, pattern_val, _pi_gate_po_output_res_gpu[gate_idx]);
}

__device__ __forceinline__ void _apply_PO(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
}

__global__ void _run_gate_level(const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                          const int *_gateIdx_in_each_level_gpu, 
                          const int *_per_level_of_group_start_accum_level_gpu, const int k_level, 
                          const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          const size_t *_patterns_gpu, const size_t rd, 
                          const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                          const size_t fault_num, const size_t bad_case, 
                          const int num_gates_per_level, 
                          const int _num_PIs) {  
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (num_gates_per_level > t_idx) {
    int k_accum_gates = _per_level_of_group_start_accum_level_gpu[k_level];
    int real_g_idx = _gateIdx_in_each_level_gpu[t_idx + k_accum_gates];

    int SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    
    // If have fault 
    if (SA_fault) {
      _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
      return;
    }

    int type = _pi_gate_po_gate_type_gpu[real_g_idx];

    switch (type) { 
      case 0:
      _apply_INV(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 1:
      _apply_AND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 2:
      _apply_OR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 3:
      _apply_XOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 4:
      _apply_NAND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 5:
      _apply_NOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 6:
      _apply_XNOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 7:
      _apply_MUX(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 8:
      _apply_CLKBUF(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 9:
      _apply_PI(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
      break;
    case 10:
      _apply_PO(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 11:
      break;
    }      
  }
}

__global__ void _run_gate_level_1115(const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                                      const int *_gateIdx_in_each_level_gpu, 
                                      const int *_per_level_of_group_start_accum_level_gpu, const int *level_counter_gpu, 
                                      const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                      const size_t *_patterns_gpu, const size_t rd, 
                                      const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                                      const size_t fault_num, const size_t bad_case, 
                                      const int *num_gates_per_level, 
                                      const int _num_PIs) {  
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  // if (t_idx == 0)
  //   printf("b = %d, t = %d, *level_counter_gpu = %d, *num_gates_per_level = %d\n", blockIdx.x, threadIdx.x, *level_counter_gpu, *num_gates_per_level);
  if (*num_gates_per_level > t_idx) {

    int k_accum_gates = _per_level_of_group_start_accum_level_gpu[*level_counter_gpu];
    int real_g_idx = _gateIdx_in_each_level_gpu[t_idx + k_accum_gates];
    
    // if (t_idx == ((*num_gates_per_level)-1)) {
    //   printf("*level_counter_gpu = %d, t_idx = %d, blockDim.x = %d\n", *level_counter_gpu, t_idx, blockDim.x);
    // } 

    int type = _pi_gate_po_gate_type_gpu[real_g_idx];
  

    switch (type) { 
      case 0:
      _apply_INV(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 1:
      _apply_AND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 2:
      _apply_OR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 3:
      _apply_XOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 4:
      _apply_NAND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 5:
      _apply_NOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 6:
      _apply_XNOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 7:
      _apply_MUX(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 8:
      _apply_CLKBUF(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 9:
      _apply_PI(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
      break;
    case 10:
      _apply_PO(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
      break;
    case 11:
      break;
    }      
  }  
}
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
                              const size_t fault_num, const size_t bad_case) {  
                                
  // int t_idx = threadIdx.x;
  // int acc = 0;
  // if (t_idx == 0) {
  //   const int tmp_level = _start_level+4;
  //   for (int level = tmp_level; level < _total_num_levels ; level++) {
  //     const int k = blockIdx.x;
  //     const int k_level = k*_total_num_levels+level;
  //     const int num_gates_this_level = _numGates_per_level_gpu_of_groups_gpu[k_level];    
  //     // printf("level = %d, %d\n", level, num_gates_this_level);
  //     acc += num_gates_this_level;
  //   }
  //   if (acc != 0) {
  //     printf("%d, %d\n", tmp_level, _total_num_levels);
  //     printf("blockIdx.x = %d, acc = %d\n", blockIdx.x, acc);
  //   }
  // }

  // __syncthreads();


  int t_idx = threadIdx.x;
  for (int level = _start_level; level < _total_num_levels ; level++) {
    const int k = blockIdx.x;
    const int k_level = k*_total_num_levels+level;
    const int num_gates_this_level = _numGates_per_level_gpu_of_groups_gpu[k_level];
    const int gateRounds = (num_gates_this_level + blockDim.x - 1)/blockDim.x;
    
    for (int gateRd = 0; gateRd < gateRounds; gateRd++) {
      if (num_gates_this_level > t_idx) {
        const int k_accum_gates = _per_level_of_group_start_accum_gpu[k_level];
        const int real_g_idx = _cones_partitioned_gpu[t_idx+k_accum_gates+gateRd*blockDim.x];

        const int SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
        
        // If have fault 
        if (SA_fault) {
          _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
          return;
        }

        int type = _pi_gate_po_gate_type_gpu[real_g_idx];
        
        switch (type) { 
          case 0:
            _apply_INV(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 1:
            _apply_AND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 2:
            _apply_OR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 3:
            _apply_XOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 4:
            _apply_NAND(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 5:
            _apply_NOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 6:
            _apply_XNOR(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 7:
            _apply_MUX(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 8:
            _apply_CLKBUF(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 9:
            _apply_PI(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
            break;
          case 10:
            _apply_PO(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
            break;
          case 11:
            break;
        } // switch 

      } // if 
    } // gateRD

    __syncthreads();
  } // level 
}



__global__ void _write_and_shift_to_array_gpu(const int _num_PIs, 
                                              const int _num_inner_gates, 
                                              const int _num_POs, 
                                              const size_t bits, 
                                              const size_t *_pi_gate_po_output_res_gpu, const int total_num_gates,
                                              size_t *outputs_PIs, size_t *outputs_Gates, size_t *outputs_POs) {

  int gate_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (gate_idx < total_num_gates) {
    size_t *outputs = (gate_idx < _num_PIs) ? outputs_PIs 
                      : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? outputs_POs 
                      : outputs_Gates;

    int index_accum = (gate_idx < _num_PIs) ? 0 
                    : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? _num_PIs 
                    : (_num_PIs+_num_POs);

    outputs[gate_idx-index_accum] = (_pi_gate_po_output_res_gpu[gate_idx] << (SIZE_T_BITS - bits));
    outputs[gate_idx-index_accum] >>= (SIZE_T_BITS - bits);
  } 
}


// print functions ------------------

__global__  void _print_simulation_results(const int _num_PIs, 
                                          const int _num_inner_gates, 
                                          const int _num_POs, 
                                          const size_t *_g_pi_results_gpu,
                                          const size_t *_g_gate_results_gpu,
                                          const size_t *_g_po_results_gpu){
  for (int i = 0; i < _num_PIs; i++) {
    printf("PI_%d.output = %lu\n", i, _g_pi_results_gpu[i]);
  }
  for (int i = 0; i < _num_inner_gates; i++) {
    printf("Gate_%d.output = %lu\n", i, _g_gate_results_gpu[i]);
  }
  for (int i = 0; i < _num_POs; i++) {
    printf("PO_%d.output = %lu\n", i, _g_po_results_gpu[i]);
  }
}