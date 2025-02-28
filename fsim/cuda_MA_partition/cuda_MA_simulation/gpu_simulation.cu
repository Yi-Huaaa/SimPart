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
#include "./gpu_simulation.cuh"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

// System includes
#include <cassert>
#include <cstdio>

// constexpr
constexpr int _NUM_THREADS = 1024;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)



void GALPS_MA_GPUSimulator::_run_gates_DSP_gpu(const int _total_num_levels,
                                              const std::vector<int> &_numGates_per_level,
                                              const int *_numGates_per_level_gpu,
                                              const int *_invAdj_gpu,
                                              const int *_invAdj_index_table_gpu,
                                              const int *_pi_gate_po_gate_type_gpu,
                                              const size_t *_patterns_gpu,
                                              const std::vector<Pattern> _patterns,
                                              const int *_fault_gate_idx_gpu,
                                              const size_t *_fault_SA_fault_val_gpu,
                                              size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_gates_DSP_gpu();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
            // if (rd==0) {
              // printf("rd = %lu, _num_rounds= %lu %d\n", rd, _num_rounds, _total_num_levels);
              // if (_sum_pi_gates_pos <= 409184) {
              //   printf("PASS: #gates = %d\n", _sum_pi_gates_pos);
              // } else {
              //   printf("NO: #gates = %d\n", _sum_pi_gates_pos);
              // }
              // for (int i = 0; i < _total_num_levels; i++) {
              //   printf("level %d: %d\n", i, _numGates_per_level[i]);
              // }
            // }
    _run_cones_good_case_DSP_gpu(_total_num_levels, 
                                  _numGates_per_level, 
                                  _numGates_per_level_gpu,
                                  _invAdj_gpu, 
                                  _invAdj_index_table_gpu, 
                                  _pi_gate_po_gate_type_gpu, 
                                  _patterns_gpu,
                                  _fault_gate_idx_gpu, 
                                  _fault_SA_fault_val_gpu,
                                  num_testcases_this_round,  // bits 
                                  rd, 
                                  _pi_gate_po_output_res_gpu);
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    if (rd == 0) {
      // shift and copy answer to the good results
      int num_blocks = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                       (_sum_pi_gates_pos + _NUM_THREADS - 1)/_NUM_THREADS : 
                       (1);//1;
      int num_threads = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                        (_NUM_THREADS) : 
                        (_sum_pi_gates_pos);    
      _write_and_shift_to_array_gpu <<< num_blocks, num_threads >>> (num_testcases_this_round, 
                                                                    _pi_gate_po_output_res_gpu,
                                                                    _sum_pi_gates_pos);
      cudaCheckErrors("CUDA: _write_and_shift_to_array_gpu launch- Failure");

      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
      
      std::cout << "GOOD resutls ans:" << std::endl;
      _print_simulation_results <<< 1, 1 >>> (_pi_gate_po_output_res_gpu, _sum_pi_gates_pos);
      cudaDeviceSynchronize();
      cudaCheckErrors("CUDA: _print_simulation_results cudaDeviceSynchronize - Failure");
    }
#endif
  }
}


void GALPS_MA_GPUSimulator::_run_cones_good_case_DSP_gpu(const int _total_num_levels,
                                                        const std::vector<int> &_numGates_per_level,
                                                        const int *_numGates_per_level_gpu,
                                                        const int *_invAdj_gpu,
                                                        const int *_invAdj_index_table_gpu,
                                                        const int *_pi_gate_po_gate_type_gpu,
                                                        const size_t *_patterns_gpu,
                                                        const int *_fault_gate_idx_gpu,
                                                        const size_t *_fault_SA_fault_val_gpu,
                                                        const size_t bits,
                                                        const size_t rd, 
                                                        size_t *_pi_gate_po_output_res_gpu) {
  // printf("Getting inside _run_cones_good_case_DSP_gpu\n");

  const size_t fault_num = 0; const size_t bad_case = 0;
  int num_blocks, num_threads;
  int num_accumGates = 0;

  for (int level = 0; level < _total_num_levels; level++) {
    const int num_gates_per_level = (_numGates_per_level[level]);
    num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                  (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                  (1);
    num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                  (_NUM_THREADS) : 
                  (num_gates_per_level);

    // printf("\nnum_gates_per_level = %d, num_blocks = %d, num_threads = %d\n", num_gates_per_level, num_blocks, num_threads);
    _run_gate_DSP <<< num_blocks, num_threads >>> (num_accumGates, _numGates_per_level_gpu, 
                                                    _invAdj_gpu, _invAdj_index_table_gpu, 
                                                    _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                    _patterns_gpu, rd, 
                                                    _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                    fault_num, bad_case, 
                                                    num_gates_per_level, 
                                                    _num_PIs);
    num_accumGates += num_gates_per_level; 
    
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif      
  }
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

__global__ void _run_gate_DSP(const int num_accumGates, const int *_numGates_per_level_gpu,
                              const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                              const int *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                              const size_t *_patterns_gpu, const size_t rd, 
                              const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                              const size_t fault_num, const size_t bad_case, 
                              const int num_gates_per_level, 
                              const int _num_PIs) {  
  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (num_gates_per_level > t_idx) {
    int real_g_idx = num_accumGates+t_idx;

    int SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    
    // If have fault 
    if (SA_fault) {
      _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
      return;
    }

    int type = _pi_gate_po_gate_type_gpu[real_g_idx];
    // printf("t_idx = %d, num_gates_per_level = %d, num_accumGates = %d, real_g_idx = %d, type = %d\n", t_idx, num_gates_per_level, num_accumGates, real_g_idx, type);

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

__global__ void _write_and_shift_to_array_gpu(const size_t bits, 
                                              size_t *_pi_gate_po_output_res_gpu, 
                                              const int total_num_gates) {

  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (t_idx < total_num_gates) {
    // printf("pre: _pi_gate_po_output_res_gpu[%d] = %lu\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
    _pi_gate_po_output_res_gpu[t_idx] = (_pi_gate_po_output_res_gpu[t_idx] << (SIZE_T_BITS - bits));
    _pi_gate_po_output_res_gpu[t_idx] >>= (SIZE_T_BITS - bits);
    // printf("post: _pi_gate_po_output_res_gpu[%d] = %lu\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
  } 
}


// print functions ------------------

__global__  void _print_simulation_results(const size_t *_pi_gate_po_output_res_gpu, 
                                            const int total_num_gates){
  for (int i = 0; i < total_num_gates; i++)
    printf("gate_%d.output = %lu\n", i, _pi_gate_po_output_res_gpu[i]);
}