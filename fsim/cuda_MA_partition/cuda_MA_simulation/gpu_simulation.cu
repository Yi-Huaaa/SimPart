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
constexpr int _NUM_THREADS = 512;
constexpr int _NUM_GATES_PER_CL = 32;
constexpr uint32_t UINT32T_BITS = std::numeric_limits<uint32_t>::digits;

// #define PRINT_SIMULATION_OUTPUTS_DSP
// #define PRINT_SIMULATION_OUTPUTS_CACHE

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
                                              const uint32_t *_patterns_gpu,
                                              const std::vector<Pattern> _patterns,
                                              const int *_fault_gate_idx_gpu,
                                              const size_t *_fault_SA_fault_val_gpu,
                                              uint32_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_gates_DSP_gpu();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (UINT32T_BITS * (rd + 1))))
            ? (UINT32T_BITS)
            : (_num_pattern % UINT32T_BITS);
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
#ifdef PRINT_SIMULATION_OUTPUTS_DSP
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
                                                        const uint32_t *_patterns_gpu,
                                                        const int *_fault_gate_idx_gpu,
                                                        const size_t *_fault_SA_fault_val_gpu,
                                                        const size_t bits,
                                                        const size_t rd, 
                                                        uint32_t *_pi_gate_po_output_res_gpu) {
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
    
// #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
// #endif      
  }
}

void GALPS_MA_GPUSimulator::_run_gates_MA_gpu(const int _total_num_levels, 
                                              const std::vector<int> &_numGates_per_level,
                                              const int *_numGates_per_level_gpu,
                                              const int *_invAdj_gpu,
                                              const int *_invAdj_index_table_gpu,
                                              const int *_pi_gate_po_gate_type_gpu,
                                              const uint32_t *_patterns_gpu,
                                              const std::vector<Pattern> _patterns,
                                              const int *_fault_gate_idx_gpu,
                                              const size_t *_fault_SA_fault_val_gpu,
                                              uint32_t *_pi_gate_po_output_res_gpu,
                                              const int *_st_ld_CLIdxs_gpu,
                                              const int *_st_ld_positi_gpu,
                                              const int *_st_ld_CLs_index_table_gpu,
                                              uint32_t *_gpu_sync,
                                              const uint32_t *_num_needed_blocks_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_gates_MA_gpu();\n";
#endif

  // Simulation
  // for (size_t rd = 0; rd < 1; rd++) {
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (UINT32T_BITS * (rd + 1))))
            ? (UINT32T_BITS)
            : (_num_pattern % UINT32T_BITS);
    _run_cones_good_case_MA_gpu(_total_num_levels, 
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
                                  _pi_gate_po_output_res_gpu,
                                  _st_ld_CLIdxs_gpu,
                                  _st_ld_positi_gpu,
                                  _st_ld_CLs_index_table_gpu,
                                  _gpu_sync, _num_needed_blocks_gpu);
#ifdef PRINT_SIMULATION_OUTPUTS_CACHE
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

void GALPS_MA_GPUSimulator::_run_cones_good_case_MA_gpu(const int _total_num_levels,
                                                        const std::vector<int> &_numGates_per_level,
                                                        const int *_numGates_per_level_gpu,
                                                        const int *_invAdj_gpu,
                                                        const int *_invAdj_index_table_gpu,
                                                        const int *_pi_gate_po_gate_type_gpu,
                                                        const uint32_t *_patterns_gpu,
                                                        const int *_fault_gate_idx_gpu,
                                                        const size_t *_fault_SA_fault_val_gpu,
                                                        const size_t bits,
                                                        const size_t rd, 
                                                        uint32_t *_pi_gate_po_output_res_gpu, 
                                                        const int *_st_ld_CLIdxs_gpu,
                                                        const int *_st_ld_positi_gpu,
                                                        const int *_st_ld_CLs_index_table_gpu,
                                                        uint32_t *_gpu_sync,
                                                        const uint32_t *_num_needed_blocks_gpu) {
  // printf("Getting inside _run_cones_good_case_MA_gpu\n");
  const size_t fault_num = 0; const size_t bad_case = 0;
  // printf("num_blocks = %d, num_threads = %d\n", _used_num_blocks, _num_threads);

  int *bibicheck; 
  // cudaMalloc((void**)&bibicheck, _sum_pi_gates_pos*sizeof(int));
  // cudaMemset(bibicheck, 0, _sum_pi_gates_pos*sizeof(int));

  _run_gate_MA <<< _used_num_blocks, _num_threads >>> (_used_num_blocks, _gpu_sync, _sum_pi_gates_pos, 
                                                      _numGates_per_level_gpu, _total_num_levels, 
                                                      _invAdj_gpu, _invAdj_index_table_gpu, 
                                                      _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                      _patterns_gpu, rd, 
                                                      _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                      fault_num, bad_case, 
                                                      _num_PIs, 
                                                      _st_ld_CLIdxs_gpu, _st_ld_positi_gpu, 
                                                      _st_ld_CLs_index_table_gpu,
                                                      _num_needed_blocks_gpu,
                                                      bibicheck);

#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif  

  // cudaDeviceSynchronize();
  // std::vector<int> bibicheck_cpu; 
  // bibicheck_cpu.resize(_sum_pi_gates_pos, 0);
  // cudaMemcpyAsync(bibicheck_cpu.data(), bibicheck, 
  //                 _sum_pi_gates_pos*sizeof(int), cudaMemcpyDeviceToHost);
  // cudaCheckErrors("CUDA: bibicheck - cudaMemcpyAsync - Failure");  
  // cudaDeviceSynchronize();
  // for (int i = 0 ; i < _sum_pi_gates_pos; i++) {
  //   if (bibicheck_cpu[i] != 1) {
  //     printf("bibicheck_cpu[%d] = %d\n", i, bibicheck_cpu[i]);

  //   }
  // }


}









// ----------------------------------------------------- PURE DSP STARTS -----------------------------------------------------

// GPU functions
__device__ __forceinline__ void _apply_INV(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_AND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_OR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];

  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_XOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_NAND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_NOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_XNOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_MUX(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  const uint32_t a = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  const uint32_t b = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+1]];
  const uint32_t s = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+2]];

  uint32_t ret = ((s & b) | ( a & (!s)));

  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_CLKBUF(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  uint32_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_PI(const int gate_idx, const int *_invAdj_gpu, 
                                          const int *_invAdj_index_table_gpu, 
                                          uint32_t *_pi_gate_po_output_res_gpu, 
                                          const uint32_t pattern_val) {
  _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
  // printf("_apply_PI: gate_idx = %d, pattern_val = %lu, %lu\n", gate_idx, pattern_val, _pi_gate_po_output_res_gpu[gate_idx]);
}

__device__ __forceinline__ void _apply_PO(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
}

__global__ void _run_gate_DSP(const int num_accumGates, const int *_numGates_per_level_gpu,
                              const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                              const int *_pi_gate_po_gate_type_gpu, uint32_t *_pi_gate_po_output_res_gpu, 
                              const uint32_t *_patterns_gpu, const size_t rd, 
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
    // printf("gate_idx = %d\n", real_g_idx);
  }
}

// ----------------------------------------------------- PURE DSP END -----------------------------------------------------


// ----------------------------------------------------- MA_DSP STARTS -----------------------------------------------------
// GPU functions
__device__ __forceinline__ int _get_CLIdx(int gateIdx) {
  return gateIdx/_NUM_GATES_PER_CL;
}

__device__ __forceinline__ uint32_t _get_bits(const uint32_t *value, const int p) {
  return (*value >> (p*8)) & 0xFF;
}

__device__ __forceinline__ int _get_g_c(const uint32_t *g_c_table, const int CLIdx) {
  return _get_bits(&g_c_table[(CLIdx&8191)/4], CLIdx&3); // 0: global, 1: cache
}

__device__ __forceinline__ int _get_pos(const int gate_idx, const uint32_t *pos_table, const int CLIdx) {
  int clpos = _get_bits(&pos_table[(CLIdx&8191)/4], CLIdx&3); // 0: global, 1: cache
  return (clpos<<5)+gate_idx&31;
}

__device__ __forceinline__ uint32_t _get_val(const int gate_idx, 
                                            const uint32_t *_pi_gate_po_output_res_gpu, const uint32_t *cache, 
                                            const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int CLIdx = _get_CLIdx(gate_idx);
  const int g_c = _get_g_c(g_c_table, CLIdx);
  const int pos = _get_pos(gate_idx, pos_table, CLIdx);
  const uint32_t ret = (g_c == 0) 
                    ? (_pi_gate_po_output_res_gpu[gate_idx])
                    : (cache[pos]);
  // printf("_get_val: gate_idx = %d, CLIdx = %d, g_c = %u, pos = %u, val = %u\n", gate_idx, CLIdx, g_c, pos, ret);
  return ret;
}

__device__ __forceinline__ void _wb_val(const int gate_idx, const uint32_t wb_val, 
                                      uint32_t *_pi_gate_po_output_res_gpu, uint32_t *cache, 
                                      const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int CLIdx = _get_CLIdx(gate_idx);
  const int g_c = _get_g_c(g_c_table, CLIdx);
  const int pos = _get_pos(gate_idx, pos_table, CLIdx);
  uint32_t *wb = (g_c == 0) ? (_pi_gate_po_output_res_gpu) : (cache);
  const uint32_t wb_pos = (g_c == 0) ? (gate_idx) : (pos);
  wb[wb_pos] = wb_val;
}

__device__ __forceinline__ void _wb_neg_val(const int gate_idx, const uint32_t wb_val, 
                                            uint32_t *_pi_gate_po_output_res_gpu, uint32_t *cache, 
                                            const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int CLIdx = _get_CLIdx(gate_idx);
  const int g_c = _get_g_c(g_c_table, CLIdx);
  const int pos = _get_pos(gate_idx, pos_table, CLIdx);
  uint32_t *wb = (g_c == 0) ? (_pi_gate_po_output_res_gpu) : (cache);
  const uint32_t wb_pos = (g_c == 0) ? (gate_idx) : (pos);
  wb[wb_pos] = ~wb_val;
}


__device__ __forceinline__ void _apply_INV_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  _wb_neg_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_AND_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret &= now_gate_val; 
  }
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_OR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret |= now_gate_val; 
  }
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_XOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret ^= now_gate_val; 
  }
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_NAND_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                                uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret &= now_gate_val; 
  }
  _wb_neg_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);  
}

__device__ __forceinline__ void _apply_NOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret |= now_gate_val; 
  }
  _wb_neg_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_XNOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                                uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const int e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  
  for (int n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const uint32_t now_gate_val = _get_val(_invAdj_gpu[n_loc], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
    ret ^= now_gate_val; 
  }
  _wb_neg_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_MUX_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  const uint32_t a = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  const uint32_t b = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  const uint32_t s = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  uint32_t ret = ((s & b) | ( a & (!s)));
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_CLKBUF_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                                  uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void _apply_PI_MA(const int gate_idx, 
                                            uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table,
                                            uint32_t *_pi_gate_po_output_res_gpu, 
                                            const uint32_t pattern_val) {
  const int CLIdx = _get_CLIdx(gate_idx);
  int g_c = _get_g_c(g_c_table, CLIdx);
  int pos = _get_pos(gate_idx, pos_table, CLIdx);
  uint32_t ret = (g_c == 0) ? (pattern_val) : (cache[pos]);
  uint32_t *wb = (g_c == 0) ? (cache) : (_pi_gate_po_output_res_gpu);
  uint32_t wb_pos = (g_c == 0) ? (pos) : (gate_idx);
  wb[wb_pos] = ret; 
  // printf("CLIdx = %d, g_c = %u, pos = %u, val = %u, wb[%u] = %u\n", CLIdx, g_c, pos, ret, pattern_val, wb_pos, wb[wb_pos]);
}

__device__ __forceinline__ void _apply_PO_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                            uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table) {
  const int s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  uint32_t ret = _get_val(_invAdj_gpu[s_loc+0], _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
  _wb_val(gate_idx, ret, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
}

__device__ __forceinline__ void __syncthreadsAllBlocks(uint32_t *_gpu_sync, 
                                                      const int l, 
                                                      const int num_blocks) {
  // sync threads in the same block 
  __syncthreads(); 

  // centralized sync threads in all blocks 
  if (threadIdx.x == 0) {
    atomicAdd(_gpu_sync, 1);  
  }

  // Busy-wait: wait for all blocks to finish
  while (atomicAdd(_gpu_sync, 0) < (l + 1) * num_blocks) {
    __threadfence();  
  }
}

__device__ __forceinline__ void __syncthreadsAllBlocks_early_ret(uint32_t *_gpu_sync, const uint32_t acc) {
  // sync threads in the same block 
  __syncthreads(); 

  // centralized sync threads in all blocks 
  if (threadIdx.x == 0) {
    atomicAdd(_gpu_sync, 1);
  }

  // Busy-wait: wait for all blocks to finish
  while (atomicAdd(_gpu_sync, 0) < acc) {
    __threadfence();  
  }
}

// get store/load index 
__device__ __forceinline__ int _getSLIdx(const int *_st_ld_CLs_index_table_gpu, const int l, const int cases) {
  return _st_ld_CLs_index_table_gpu[4*l+cases];
}

__device__ void _run_gate_MA_thd(const int real_g_idx, const int num_gates_per_level, const int num_accumGates, 
                                const int SA_fault, 
                                const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                                const int *_pi_gate_po_gate_type_gpu, uint32_t *_pi_gate_po_output_res_gpu, 
                                const uint32_t *_patterns_gpu, const size_t rd, 
                                const int _num_PIs,
                                const int *_st_ld_CLIdxs_gpu,
                                const int *_st_ld_positi_gpu,
                                const int *_st_ld_CLs_index_table_gpu,
                                uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table,
                                int *bibicheck) {
  if (real_g_idx < (num_gates_per_level+num_accumGates)) {
    // bibicheck[real_g_idx]++;
    int type = _pi_gate_po_gate_type_gpu[real_g_idx];
    switch (type) { 
      case 0:
      _apply_INV_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 1:
      _apply_AND_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 2:
      _apply_OR_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 3:
      _apply_XOR_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 4:
      _apply_NAND_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 5:
      _apply_NOR_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 6:
      _apply_XNOR_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 7:
      _apply_MUX_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 8:
      _apply_CLKBUF_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 9:
      // _apply_PI_MA(real_g_idx, cache, g_c_table, pos_table, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
      break;
    case 10:
      _apply_PO_MA(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, cache, g_c_table, pos_table);
      break;
    case 11:
      break;
    }
  }
}

__device__ __forceinline__ int _CLIdx_to_idx(const int CLIdx, const int lane_idx) {
  return (CLIdx<<5)+lane_idx;
}

// __device__ __forceinline__ void _modify_bits(uint32_t *a, const int p, const int val) {
//   uint32_t mask = 0xFF << (p * 8);  // Mask for the 8-bit segment
//   uint32_t value = (val & 0xFF) << (p * 8);  // Shift val into position
//   *a = (*a & ~mask) | value; // Clear the target bits and set new value
// }

__device__ __forceinline__ void _modify_bits(uint32_t *a, const int p, const int val) {
  uint32_t mask = 0xFF << (p * 8);
  uint32_t value = (val & 0xFF) << (p * 8);
  // *a = (*a & ~mask) | value; // Clear the target bits and set new value
  uint32_t old_val, new_val;
  do {
    old_val = *a;
    new_val = (old_val & ~mask) | value;
  } while (atomicCAS(a, old_val, new_val) != old_val);
}


__global__ void _run_gate_MA(const int num_blocks, uint32_t *_gpu_sync, const int _sum_pi_gates_pos, 
                            const int *_numGates_per_level_gpu, const int _total_num_levels,
                            const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                            const int *_pi_gate_po_gate_type_gpu, uint32_t *_pi_gate_po_output_res_gpu, 
                            const uint32_t *_patterns_gpu, const size_t rd, 
                            const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                            const size_t fault_num, const size_t bad_case, 
                            const int _num_PIs,
                            const int *_st_ld_CLIdxs_gpu,
                            const int *_st_ld_positi_gpu,
                            const int *_st_ld_CLs_index_table_gpu,
                            const uint32_t *_num_needed_blocks_gpu,
                            int * bibicheck) {
  // shared memory allocation 
  __shared__ uint32_t shared_mem[48 * 1024 / sizeof(uint32_t)]; // 48 KB shared memory as int array
  uint32_t *cache = shared_mem;                   // First 32 KB (8192 elements)
  uint32_t *g_c_table = shared_mem + 8192;        // Next 8 KB (starting from element 8192)
  uint32_t *pos_table = shared_mem + 8192 + 2048; // Last 8 KB (starting from element 8192 + 2048)

  // starts 
  const int t_idx = blockDim.x*blockIdx.x+threadIdx.x;
  const int w_idx = threadIdx.x>>5;   // 你是該 block 中的哪一個 warp
  const int numWarpsPerBlk = blockDim.x>>5;
  const int lane_idx = threadIdx.x&31; // warp 中的第幾個 thread
  int num_accumGates = 0; 
  const int SA_fault = 0; 

  for (int l = 0; l < _total_num_levels; l++) {
    // if (blockIdx.x < 4) {
    //   // ST/LD cachelines 
    //   const int bi = 1; 
    //   // ST
    //   const int st_s = _getSLIdx(_st_ld_CLs_index_table_gpu, l, 0);
    //   const int st_e = _getSLIdx(_st_ld_CLs_index_table_gpu, l, 1);
    //   const int numST = st_e-st_s;
    //   const int STrounds = (numST + numWarpsPerBlk - 1)/numWarpsPerBlk;
    //   const int bi2 = (bi < STrounds) ? (bi) : (STrounds);
    //   for (int rd = 0; rd < bi2; rd++) {
    //     const int acc_idx = rd*numWarpsPerBlk+w_idx;
    //     if (acc_idx < numST) {
    //       // ST
    //       const int STCLIdx = _st_ld_CLIdxs_gpu[st_s+acc_idx];
    //       const int STCLPos = _st_ld_positi_gpu[st_s+acc_idx];
    //       if (STCLIdx != -1) {
    //         _pi_gate_po_output_res_gpu[_CLIdx_to_idx(STCLIdx, lane_idx)] = cache[_CLIdx_to_idx(STCLPos, lane_idx)];
    //         if (lane_idx == 0) {
    //           _modify_bits(&g_c_table[(STCLIdx&8191)>>2], STCLIdx&3, 0); // update g_c_table
    //         }
    //       }
    //     } // if (acc_idx < numST)
    //   } // for loop: STrounds

    //   // LD
    //   const int ld_s = _getSLIdx(_st_ld_CLs_index_table_gpu, l, 2);
    //   const int ld_e = _getSLIdx(_st_ld_CLs_index_table_gpu, l, 3);  
    //   const int numLD = ld_e-ld_s;
    //   const int LDrounds = (numLD + numWarpsPerBlk - 1)/numWarpsPerBlk;
    //   const int bi3 = (bi < LDrounds) ? (bi) : (LDrounds);
    //   for (int rd = 0; rd < bi3; rd++) {
    //     const int acc_idx = rd*numWarpsPerBlk+w_idx;
    //     if (acc_idx < numLD) {
    //       // LD
    //       const int LDCLIdx = _st_ld_CLIdxs_gpu[ld_s+acc_idx];
    //       const int LDCLPos = _st_ld_positi_gpu[ld_e+acc_idx];
    //       cache[_CLIdx_to_idx(LDCLPos, lane_idx)] = _pi_gate_po_output_res_gpu[_CLIdx_to_idx(LDCLIdx, lane_idx)];
    //       if (lane_idx == 0) {
    //         _modify_bits(&g_c_table[(LDCLIdx&8191)>>2], LDCLIdx&3, 1); // update g_c_table
    //         _modify_bits(&pos_table[(LDCLIdx&8191)>>2], LDCLIdx&3, LDCLPos); // update pos_table
    //       }
    //     } // if (acc_idx < numLD)
    //   } // for loop: LDrounds  
    // // } // if branch 
    // __syncthreads();

    // Do works, since one block may need to do multiple w_rounds of block 
    const int totalNumThreads = num_blocks*blockDim.x;
    const int w_rounds = (_numGates_per_level_gpu[l] + totalNumThreads - 1)/totalNumThreads;
    for (int rd = 0; rd < w_rounds; rd++) {
      const int rdAccum = rd*totalNumThreads;
      const int real_g_idx = num_accumGates+rdAccum+t_idx;
      _run_gate_MA_thd(real_g_idx, _numGates_per_level_gpu[l], num_accumGates, SA_fault, 
                        _invAdj_gpu, _invAdj_index_table_gpu,
                        _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                        _patterns_gpu, rd, 
                        _num_PIs, 
                        _st_ld_CLIdxs_gpu, 
                        _st_ld_positi_gpu, 
                        _st_ld_CLs_index_table_gpu, 
                        cache, g_c_table, pos_table,
                        bibicheck);
    }
    num_accumGates += _numGates_per_level_gpu[l];
    // sync
    __syncthreadsAllBlocks(&_gpu_sync[3], l, num_blocks);
  } // for loop: _total_num_levels 

  // // Last WB POs
  // const int st_s_f = _getSLIdx(_st_ld_CLs_index_table_gpu, _total_num_levels, 0);
  // const int st_e_f = _getSLIdx(_st_ld_CLs_index_table_gpu, _total_num_levels, 1);
  // const int numST_f = st_e_f-st_s_f;

  // if (w_idx < numST_f) {
  //   const int CLIdx = _st_ld_CLIdxs_gpu[st_s_f+w_idx];
  //   const int CLPos = _st_ld_positi_gpu[st_s_f+w_idx];
  //   if (_CLIdx_to_idx(CLIdx, lane_idx) < _sum_pi_gates_pos && CLIdx != -1) {
  //     _pi_gate_po_output_res_gpu[_CLIdx_to_idx(CLIdx, lane_idx)] = cache[_CLIdx_to_idx(CLPos, lane_idx)];
  //   }
  // } // if (w_idx < numST_f) 
}



// ----------------------------------------------------- MA_DSP ENDS -----------------------------------------------------




__global__ void _write_and_shift_to_array_gpu(const size_t bits, 
                                              uint32_t *_pi_gate_po_output_res_gpu, 
                                              const int total_num_gates) {

  int t_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (t_idx < total_num_gates) {
    // printf("pre: _pi_gate_po_output_res_gpu[%d] = %u\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
    _pi_gate_po_output_res_gpu[t_idx] = (_pi_gate_po_output_res_gpu[t_idx] << (UINT32T_BITS - bits));
    _pi_gate_po_output_res_gpu[t_idx] >>= (UINT32T_BITS - bits);
    // printf("post: _pi_gate_po_output_res_gpu[%d] = %lu\n", t_idx, _pi_gate_po_output_res_gpu[t_idx]);
  } 
}


// ----------------------------------------------------- PRINT FUNCTIONS -----------------------------------------------------

__global__  void _print_simulation_results(const uint32_t *_pi_gate_po_output_res_gpu, 
                                            const int total_num_gates){
  for (int i = 0; i < total_num_gates; i++)
    printf("gate_%d.output = %u\n", i, _pi_gate_po_output_res_gpu[i]);
}