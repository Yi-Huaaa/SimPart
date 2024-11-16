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


__global__ void doWhileLoopKernel(char *dPtr, cudaGraphConditionalHandle handle, int *level_counter_gpu, 
                                  int *gateIdx_in_each_level_copy_gpu, 
                                  int *num_gates_per_level_gpu, int *d_numBlocks, int *d_numThreads, int *bb_m, int*tt_m)
{
    if (--(*dPtr) == 0) {
      cudaGraphSetConditional(handle, 0);
    }
    (*level_counter_gpu)++;
    (*num_gates_per_level_gpu) = gateIdx_in_each_level_copy_gpu[*level_counter_gpu];

    (*bb_m)  = ((*num_gates_per_level_gpu) > 1024) ? 
              ((*num_gates_per_level_gpu) + 1024 - 1)/1024 : 
              (1);
    (*tt_m) = ((*num_gates_per_level_gpu) > 1024) ? 
              (1024) : 
              ((*num_gates_per_level_gpu));


    // printf("GPU: counter = %d, level_counter_gpu = %d, (*num_gates_per_level_gpu) = %d, bb_m = %d, tt_m = %d\n", 
    //       *dPtr, *level_counter_gpu, *num_gates_per_level_gpu, *bb_m, *tt_m);
    // for (int i = 0; i < 77; i++) {
    //   printf("gateIdx_in_each_level_copy_gpu[%d] = %d\n", i, gateIdx_in_each_level_copy_gpu[i]);
    // }
}



void GALPS_GPUSimulator::simpleDoWhileGraph(const int *_numGates_per_level_gpu_of_groups_gpu,
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

  // Allocate device memory for dynamic block/thread config
  int *d_numBlocks, *d_numThreads, *gateIdx_in_each_level_copy_gpu, *num_gates_per_level_gpu;
  int num_blocks, num_threads, biNum = 0;
  cudaMalloc(&d_numBlocks, sizeof(int));
  cudaMalloc(&d_numThreads, sizeof(int));
  cudaMalloc(&num_gates_per_level_gpu, sizeof(int));
  cudaMalloc(&gateIdx_in_each_level_copy_gpu, _gateIdx_in_each_level.size()*sizeof(int));

  int *bb_m, *tt_m;  
  cudaMallocManaged(&bb_m, 1*sizeof(int));
  cudaMallocManaged(&tt_m, 1*sizeof(int));
  std::vector<int> tmp; 
  for (size_t level = 0; level < _gateIdx_in_each_level.size(); level++) {
    tmp.push_back(_gateIdx_in_each_level[level].size());
    biNum = (biNum > _gateIdx_in_each_level[level].size()) ? (biNum) : (_gateIdx_in_each_level[level].size());
  }
  cudaMemcpy(gateIdx_in_each_level_copy_gpu, tmp.data(), 
                  tmp.size()*sizeof(int), cudaMemcpyHostToDevice);
  num_blocks  = (biNum > _NUM_THREADS) ? 
                (biNum + _NUM_THREADS - 1)/_NUM_THREADS : 
                (1);
  num_threads = (biNum > _NUM_THREADS) ? 
                (_NUM_THREADS) : 
                (biNum);
  *bb_m = (_gateIdx_in_each_level[0].size() > _NUM_THREADS) ? 
          (_gateIdx_in_each_level[0].size() + _NUM_THREADS - 1)/_NUM_THREADS : 
          (1);;
  *tt_m = (_gateIdx_in_each_level[0].size() > _NUM_THREADS) ? 
          (_NUM_THREADS) : 
          (_gateIdx_in_each_level[0].size());
  // printf("num_blocks = %d, num_threads = %d\n", num_blocks, num_threads);
  // Allocate device memory for dynamic block/thread config

  



  cudaGraph_t     graph;
  cudaGraphExec_t graphExec;
  cudaGraphNode_t node;

  char *dPtr;
  cudaMalloc((void**)&dPtr, 1);
  int *level_counter_gpu;
  cudaMalloc((void**)&level_counter_gpu, 1*sizeof(int));

  // printf("simpleDoWhileGraph: Building graph...\n");
  cudaGraphCreate(&graph, 0);

  cudaGraphConditionalHandle handle;
  cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

  cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
  cParams.conditional.handle = handle;
  cParams.conditional.type   = cudaGraphCondTypeWhile;
  cParams.conditional.size   = 1;
  cudaGraphAddNode(&node, graph, NULL, 0, &cParams);

  cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

  cudaStream_t captureStream;
  cudaStreamCreate(&captureStream);
  
  cudaStreamBeginCaptureToGraph(captureStream, bodyGraph, nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed);
  _run_gate_level_1115 <<< num_blocks, num_threads, 0, captureStream >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
  // _run_gate_level_1115 <<< *bb_m, *tt_m, 0, captureStream >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
                                                                    _gateIdx_in_each_level_gpu, 
                                                                    _per_level_of_group_start_accum_level_gpu, level_counter_gpu,
                                                                    _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                                    _patterns_gpu, 0, 
                                                                    _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                                    0, 0, 
                                                                    num_gates_per_level_gpu, 
                                                                    _num_PIs);   
  doWhileLoopKernel<<<1, 1, 0, captureStream>>>(dPtr, handle, level_counter_gpu, 
                                                gateIdx_in_each_level_copy_gpu, 
                                                num_gates_per_level_gpu, d_numBlocks, d_numThreads, bb_m, tt_m);
  cudaStreamEndCapture(captureStream, nullptr);
  cudaStreamDestroy(captureStream);

  // cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // cudaMemset(dPtr, _end_level, 1);
  // printf("Host: Launching graph with loop counter set to %d\n", _end_level);

  cudaGraphNode_t kernelNode_PO;
  cudaKernelNodeParams kernelNode_PO_Params {0};
  cudaCheckErrors("cudaGraphCreate failure");

  const size_t num_func_inputs = 16; // partition kernel 的 input 數量
  void **kernelArgs_PO = (void **)malloc(num_func_inputs*sizeof(void *));


  const int rd = 0, fault_num = 0, bad_case = 0;
  kernelArgs_PO[0] = (void *)&_num_PIs;
  kernelArgs_PO[1] = (void *)&_end_level;
  kernelArgs_PO[2] = (void *)&_total_num_levels;
  kernelArgs_PO[3] = (void *)&_invAdj_gpu;
  kernelArgs_PO[4] = (void *)&_invAdj_index_table_gpu;
  kernelArgs_PO[5] = (void *)&_numGates_per_level_gpu_of_groups_gpu;
  kernelArgs_PO[6] = (void *)&_cones_partitioned_gpu;
  kernelArgs_PO[7] = (void *)&_per_level_of_group_start_accum_gpu;
  kernelArgs_PO[8] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PO[9] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PO[10] = (void *)&_patterns_gpu;
  kernelArgs_PO[11] = (void *)&rd;
  kernelArgs_PO[12] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PO[13] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PO[14] = (void *)&fault_num;
  kernelArgs_PO[15] = (void *)&bad_case;

  kernelNode_PO_Params.func = (void *)_run_gate_part;
  kernelNode_PO_Params.gridDim = dim3(_k, 1, 1); // Replace `_k` with your desired block configuration
  kernelNode_PO_Params.blockDim = dim3(_NUM_THREADS, 1, 1); // Replace `_NUM_THREADS` with your desired thread configuration
  kernelNode_PO_Params.sharedMemBytes = 0;
  kernelNode_PO_Params.kernelParams = (void **)(kernelArgs_PO);
  kernelNode_PO_Params.extra = NULL;

  // Add kernel node to the main graph
  cudaGraphAddKernelNode(&kernelNode_PO, graph, NULL, 0, &kernelNode_PO_Params);
  cudaCheckErrors("Adding kernelNode_PO failed");
  cudaGraphAddDependencies(graph, &node, &kernelNode_PO, 1);
  cudaCheckErrors("cudaGraphAddDependencies node -> kernelNode_PO");  


  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // cudaMemset(dPtr, _end_level, 1);
  // // printf("Host: Launching graph with loop counter set to %d\n", _end_level);

  // cudaGraphLaunch(graphExec, 0);

  // cudaDeviceSynchronize();

  // cudaGraphExecDestroy(graphExec);
  // cudaGraphDestroy(graph);
  // cudaFree(dPtr);

  // cudaError_t err = cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);




  cudaStream_t stream1115;
  cudaStreamCreate(&stream1115);
  cudaGraphLaunch(graphExec, stream1115);
  cudaDeviceSynchronize();

  auto start1115 = std::chrono::steady_clock::now();

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    cudaMemset(dPtr, _end_level, 1);
    cudaMemset(level_counter_gpu, 0, 1*sizeof(int)); // init
    *bb_m = (_gateIdx_in_each_level[0].size() > _NUM_THREADS) ? 
            (_gateIdx_in_each_level[0].size() + _NUM_THREADS - 1)/_NUM_THREADS : 
            (1);;
    *tt_m = (_gateIdx_in_each_level[0].size() > _NUM_THREADS) ? 
            (_NUM_THREADS) : 
            (_gateIdx_in_each_level[0].size());  
  
    // printf("Host: Launching graph with loop counter set to %d\n", _end_level);
  
    cudaGraphLaunch(graphExec, stream1115);
    cudaDeviceSynchronize();

    // _run_gate_part <<< _k, _NUM_THREADS, 0, stream1115 >>> (_num_PIs, _end_level, _total_num_levels, 
    //                                         _invAdj_gpu, _invAdj_index_table_gpu,
    //                                         _numGates_per_level_gpu_of_groups_gpu,
    //                                         _cones_partitioned_gpu, 
    //                                         _per_level_of_group_start_accum_gpu, 
    //                                         _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
    //                                         _patterns_gpu, rd, 
    //                                         _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
    //                                         0, 0);


      const size_t bits = num_testcases_this_round;
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
  }

  cudaDeviceSynchronize();

  auto end1115 = std::chrono::steady_clock::now();
  std::chrono::duration<double>  duration_simulation = (end1115 - start1115);
  std::cout << "1115run_simulator: " <<  hhh(((duration_simulation.count()))*1000, 0.001) << "\n";

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaFree(dPtr);
  cudaFree(d_numBlocks);
  cudaFree(d_numThreads);
  cudaFree(gateIdx_in_each_level_copy_gpu);
  cudaFree(num_gates_per_level_gpu);



  // int *testB, *testT;
  // cudaMalloc(&testB, sizeof(int));
  // cudaMalloc(&testT, sizeof(int));


  // for (int i = 0; i < 10; i++) {
  //   int newB = i+1, newT = 4*i;
  //   cudaMemset(testB, newB, 1*sizeof(int)); // init
  //   cudaMemset(testT, newT, 1*sizeof(int)); // init
  //   testing <<< *testB, *testT >>> ();
  // }

  







  // printf("simpleDoWhileGraph: Complete\n\n");
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

  simpleDoWhileGraph(_numGates_per_level_gpu_of_groups_gpu,
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
  // capturedWhileGraph();







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