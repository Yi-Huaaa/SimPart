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
#include <cstddef>  // For size_t and SIZE_MAX
#include <cstdio>

#include <fsim/fsim.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <set>

#include "./gpu_sizet_simulation.cuh"

// #define GPU_PART_DEBUG_PRINT_SIMULATION // print simulation resutls 


void GALPS_SIZET_GPUSimulator::_run_cones_gates_level_gpu(const std::vector<size_t> &_numGates_per_level_gpu_of_groups,
                                                          const size_t *_per_level_of_group_start_accum_gpu,
                                                          const size_t *_cones_partitioned_gpu,
                                                          const size_t _total_num_levels,
                                                          const size_t *_invAdj_gpu,
                                                          const size_t *_invAdj_index_table_gpu,
                                                          const size_t *_pi_gate_po_gate_type_gpu,
                                                          const size_t *_patterns_gpu,
                                                          const std::vector<Pattern> _patterns,
                                                          const size_t *_fault_gate_idx_gpu,
                                                          const size_t *_fault_SA_fault_val_gpu,
                                                          size_t *_g_pi_results_gpu,
                                                          size_t *_g_gate_results_gpu,
                                                          size_t *_g_po_results_gpu,
                                                          size_t *_b_pi_results_gpu,
                                                          size_t *_b_gate_results_gpu,
                                                          size_t *_b_po_results_gpu,
                                                          size_t *_found_fault_to_pattern_gpu,
                                                          size_t *_pi_gate_po_output_res_gpu) {
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
std::cout << "execute simulation._run_cones_gates_level_gpu();\n";
#endif

  // for (size_t g = 0; g < _numGates_per_level_gpu_of_groups.size(); g+=_total_num_levels) {
  //   printf("g = %lu\n", g/_total_num_levels);
  //   for (size_t level = 0; level < _total_num_levels; level++) {
  //     printf("\tlevel_%d__sz = %lu, over 1024 = %lu\n", 
  //               level, 
  //               _numGates_per_level_gpu_of_groups[g*_total_num_levels+level],
  //               (_numGates_per_level_gpu_of_groups[g*_total_num_levels+level]+1024-1)/1024);
  //   }
  // }

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    _run_cones_good_case_level_gpu(_numGates_per_level_gpu_of_groups,
                            _per_level_of_group_start_accum_gpu,
                            _cones_partitioned_gpu, 
                            _total_num_levels,
                            _invAdj_gpu, 
                            _invAdj_index_table_gpu, 
                            _pi_gate_po_gate_type_gpu, 
                            _patterns_gpu,
                            _fault_gate_idx_gpu, 
                            _fault_SA_fault_val_gpu,
                            num_testcases_this_round, 
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

void GALPS_SIZET_GPUSimulator::_run_cones_good_case_level_gpu(const std::vector<size_t> &_numGates_per_level_gpu_of_groups,
                                                              const size_t *_per_level_of_group_start_accum_gpu,
                                                              const size_t *_cones_partitioned_gpu,
                                                              const size_t _total_num_levels,
                                                              const size_t *_invAdj_gpu,
                                                              const size_t *_invAdj_index_table_gpu,
                                                              const size_t *_pi_gate_po_gate_type_gpu,
                                                              const size_t *_patterns_gpu,
                                                              const size_t *_fault_gate_idx_gpu,
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
  size_t num_blocks, num_threads;

  for (size_t level = 0; level < _total_num_levels; level++) {
    const size_t num_gates_per_level = (_numGates_per_level_gpu_of_groups[level]);
    num_blocks  = (num_gates_per_level > _NUM_THREADS_SIZET) ? 
                  (num_gates_per_level + _NUM_THREADS_SIZET - 1)/_NUM_THREADS_SIZET : 
                  (1);
    num_threads = (num_gates_per_level > _NUM_THREADS_SIZET) ? 
                  (_NUM_THREADS_SIZET) : 
                  (num_gates_per_level);
    // printf("level = %lu, num_blocks = %lu, num_threads = %lu, num_gates_per_level = %lu\n", 
    //         level, num_blocks, num_threads, num_gates_per_level);

    _run_gate <<< num_blocks, num_threads >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
                                              _cones_partitioned_gpu, 
                                              _per_level_of_group_start_accum_gpu, level,
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

  // shift and copy answer to the good results
  num_blocks = (_sum_pi_gates_pos > _NUM_THREADS_SIZET) ? 
                (_sum_pi_gates_pos + _NUM_THREADS_SIZET - 1)/_NUM_THREADS_SIZET : 
                (1);//1;
  num_threads = (_sum_pi_gates_pos > _NUM_THREADS_SIZET) ? 
                (_NUM_THREADS_SIZET) : 
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

#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
#endif  
}

// =========



// -------------------------------------------------

void GALPS_SIZET_GPUSimulator::_run_cones_gates_part_gpu(const size_t *_numGates_per_level_gpu_of_groups_gpu,
                                                        const size_t *_per_level_of_group_start_accum_gpu,
                                                        const size_t *_cones_partitioned_gpu,
                                                        const size_t _start_level, 
                                                        const size_t _total_num_levels,
                                                        const size_t *_invAdj_gpu,
                                                        const size_t *_invAdj_index_table_gpu,
                                                        const size_t *_pi_gate_po_gate_type_gpu,
                                                        const size_t *_patterns_gpu,
                                                        const std::vector<Pattern> _patterns,    
                                                        const size_t *_fault_gate_idx_gpu,
                                                        const size_t *_fault_SA_fault_val_gpu,                                                   
                                                        size_t *_g_pi_results_gpu,
                                                        size_t *_g_gate_results_gpu,
                                                        size_t *_g_po_results_gpu,
                                                        size_t *_b_pi_results_gpu,
                                                        size_t *_b_gate_results_gpu,
                                                        size_t *_b_po_results_gpu,
                                                        size_t *_found_fault_to_pattern_gpu,
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
      
      // cudaDeviceSynchronize(); // remove
      // cudaCheckErrors("CUDA: bibi cudaDeviceSynchronize - Failure"); // remove
    
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


void GALPS_SIZET_GPUSimulator::_run_cones_good_case_part_gpu(const size_t *_numGates_per_level_gpu_of_groups_gpu,
                                                            const size_t *_invAdj_gpu,
                                                            const size_t *_invAdj_index_table_gpu,  
                                                            const size_t *_cones_partitioned_gpu,
                                                            const size_t *_per_level_of_group_start_accum_gpu,
                                                            const size_t _start_level,
                                                            const size_t _total_num_levels,
                                                            const size_t *_pi_gate_po_gate_type_gpu,
                                                            const size_t *_patterns_gpu,
                                                            const size_t *_fault_gate_idx_gpu,
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
  // printf("_start_level = %lu, _total_num_levels = %lu\n", _start_level, _total_num_levels); // remove 

  _run_gate_part <<< _k, _NUM_THREADS_SIZET >>> (_num_PIs, _start_level, _total_num_levels, 
                                          _invAdj_gpu, _invAdj_index_table_gpu,
                                          _numGates_per_level_gpu_of_groups_gpu,
                                          _cones_partitioned_gpu, 
                                          _per_level_of_group_start_accum_gpu, 
                                          _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                          _patterns_gpu, rd, 
                                          _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                          fault_num, bad_case);

  // cudaCheckErrors("CUDA: _run_gate_part launch- Failure");
  // cudaDeviceSynchronize(); // remove


  // shift and copy answer to the good results
  size_t num_blocks = (_sum_pi_gates_pos > _NUM_THREADS_SIZET) ? 
                (_sum_pi_gates_pos + _NUM_THREADS_SIZET - 1)/_NUM_THREADS_SIZET : 
                (1);//1;
  size_t num_threads = (_sum_pi_gates_pos > _NUM_THREADS_SIZET) ? 
                (_NUM_THREADS_SIZET) : 
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


// ---------------------------------------------------------------------------------------------------------

// GPU functions
__device__ __forceinline__ void _apply_INV(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_AND(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_OR(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_XOR(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_NAND(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_NOR(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_XNOR(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  const size_t e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  
  for (size_t n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
    const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
}

__device__ __forceinline__ void _apply_MUX(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  const size_t a = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  const size_t b = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+1]];
  const size_t s = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+2]];

  size_t ret = ((s & b) | ( a & (!s)));

  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_CLKBUF(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
  size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_PI(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val) {
  _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
}

__device__ __forceinline__ void _apply_PO(const size_t gate_idx, const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
  const size_t s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
}

__global__ void _run_gate(const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, 
                          const size_t *_cones_partitioned_gpu, 
                          const size_t *_per_level_of_group_start_accum_gpu, const size_t k_level, 
                          const size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          const size_t *_patterns_gpu, const size_t rd, 
                          const size_t *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                          const size_t fault_num, const size_t bad_case, 
                          const size_t num_gates_per_level, 
                          const size_t _num_PIs) {  
  size_t t_idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (num_gates_per_level > t_idx) {
    size_t k_accum_gates = _per_level_of_group_start_accum_gpu[k_level];
    size_t real_g_idx = _cones_partitioned_gpu[t_idx + k_accum_gates]; // tmp

    size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    
    // If have fault 
    if (SA_fault) {
      _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
      return;
    }

    size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
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
  __syncthreads();
}

__global__ void _write_and_shift_to_array_gpu(const size_t _num_PIs, 
                                              const size_t _num_inner_gates, 
                                              const size_t _num_POs, 
                                              const size_t bits, 
                                              const size_t *_pi_gate_po_output_res_gpu, const size_t total_num_gates,
                                              size_t *outputs_PIs, size_t *outputs_Gates, size_t *outputs_POs) {

  size_t gate_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (gate_idx < total_num_gates) {
    size_t *outputs = (gate_idx < _num_PIs) ? outputs_PIs 
              : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? outputs_POs 
              : outputs_Gates;

    size_t index_accum = (gate_idx < _num_PIs) ? 0 
                            : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? _num_PIs 
                            : (_num_PIs+_num_POs);

    outputs[gate_idx-index_accum] = (_pi_gate_po_output_res_gpu[gate_idx] << (SIZE_T_BITS - bits));
    outputs[gate_idx-index_accum] >>= (SIZE_T_BITS - bits);
  } 
}


__global__ void _run_gate_part(const size_t _num_PIs,
                              const size_t _start_level, 
                              const size_t _total_num_levels,
                              const size_t *_invAdj_gpu, const size_t *_invAdj_index_table_gpu, 
                              const size_t *_numGates_per_level_gpu_of_groups_gpu,
                              const size_t *_cones_partitioned_gpu, 
                              const size_t *_per_level_of_group_start_accum_gpu, 
                              const size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                              const size_t *_patterns_gpu, const size_t rd, 
                              const size_t *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                              const size_t fault_num, const size_t bad_case) {  
                                
  size_t t_idx = threadIdx.x;

  for (size_t level = _start_level; level < _total_num_levels ; level++) {
    const size_t k = blockIdx.x;
    const size_t k_level = k*_total_num_levels+level;
    const size_t num_gates_this_level = _numGates_per_level_gpu_of_groups_gpu[k_level];
    const size_t gateRounds = (num_gates_this_level + blockDim.x - 1)/blockDim.x;
    
    for (size_t gateRd = 0; gateRd < gateRounds; gateRd++) {
      if (num_gates_this_level > t_idx) {
        const size_t k_accum_gates = _per_level_of_group_start_accum_gpu[k_level];
        const size_t real_g_idx = _cones_partitioned_gpu[t_idx+k_accum_gates+gateRd*blockDim.x];

        const size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
        
        // If have fault 
        if (SA_fault) {
          _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
          return;
        }

        size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
        
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

// ---------------------------------------------------------------------------------------------------------



// print functions ------------------
__global__  void _print_simulation_results(const size_t _num_PIs, 
                                          const size_t _num_inner_gates, 
                                          const size_t _num_POs, 
                                          const size_t *_g_pi_results_gpu,
                                          const size_t *_g_gate_results_gpu,
                                          const size_t *_g_po_results_gpu){
  for (size_t i = 0; i < _num_PIs; i++) {
    printf("PI_%lu.output = %lu\n", i, _g_pi_results_gpu[i]);
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    printf("Gate_%lu.output = %lu\n", i, _g_gate_results_gpu[i]);
  }
  for (size_t i = 0; i < _num_POs; i++) {
    printf("PO_%lu.output = %lu\n", i, _g_po_results_gpu[i]);
  }
}

