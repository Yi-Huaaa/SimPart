#ifndef GALPS_GPU_SIMULATION_H
#define GALPS_GPU_SIMULATION_H
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


#define GPU_PART_DEBUG_PRINT_SIMULATION


// GPU functions pre-declarations
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_INV(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_AND(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_OR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_XOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_NAND(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_NOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_XNOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_MUX(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_CLKBUF(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_PI(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_PO(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__global__ void _run_gate(const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, 
                          size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          size_t *_patterns_gpu, const size_t rd, 
                          size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                          const size_t fault_num, const size_t bad_case, 
                          const size_t max_num_gates, const size_t finished_num_gates, const size_t _num_PIs); 



// Forward declaration of CUDAPartitioner
class CUDAPartitioner;


template<typename IndexType = size_t>
class GALPS_GPUSimulator {
  const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

  friend class CUDAPartitioner;

public:
  // Simulation - (1) levelization (2) cones 
  void run_gpu_simulator_cones_gpu(const IndexType num_PIs, 
                                  const IndexType num_inner_gates,
                                  const IndexType num_POs, 
                                  const IndexType sum_pi_gates_pos,
                                  const IndexType num_pattern, 
                                  const size_t num_rounds,
                                  const IndexType num_fault, 
                                  const IndexType *_pi_gate_po_gate_type_gpu,
                                  const size_t *_patterns_gpu,
                                  const IndexType *_fault_gate_idx_gpu,
                                  const size_t *_fault_SA_fault_val_gpu,
                                  const size_t *_pi_gate_po_output_res_gpu,
                                  const IndexType *_numGates_per_level_gpu_of_groups_gpu,
                                  const IndexType *_total_numGates_partitioned_gpu,
                                  const IndexType *_per_level_of_group_start_accum_gpu,
                                  const IndexType *_cones_partitioned_gpu,
                                  const IndexType *_total_num_levels_gpu,
                                  const IndexType *_level_of_gates_gpu,
                                  size_t *_g_pi_results_gpu,
                                  size_t *_g_gate_results_gpu,
                                  size_t *_g_po_results_gpu,
                                  size_t *_b_pi_results_gpu,
                                  size_t *_b_gate_results_gpu,
                                  size_t *_b_po_results_gpu,
                                  int *_found_fault_to_pattern_gpu) {
    // Get vars          
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    
    // printf("Hello world\n");

    // run simulation
    _run_cones_gates_gpu(_Gates, 
                        _g_pi_results,
                        _g_gate_results,
                        _g_po_results,
                        _b_pi_results,
                        _b_gate_results,
                        _b_po_results,
                        _found_fault_to_pattern,
                        _TS_order_gpu, 
                      
                      );
  }
  // void run_gpu_simulator_level();


private:


  // void _run_level_gates_gpu();
  
  // vars
  IndexType _num_PIs; 
  IndexType _num_inner_gates;
  IndexType _num_POs; 
  IndexType _sum_pi_gates_pos;
  IndexType _num_pattern; 
  size_t _num_rounds; 
  IndexType _num_fault; 

  std::vector<Pattern> _patterns;
  std::vector<Fault<IndexType>> _faults;
  std::vector<std::vector<std::vector<IndexType>>> _cones_partitioned;


  // IndexType _max_level;  TODO -> get this by using TP results
  // IndexType _total_num_levels;  TODO -> get this by using TP results  
  // note: this tow number should be gpu params -> stored on GPU

  void _run_cones_gates_gpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                            std::vector<size_t> &_g_pi_results,
                            std::vector<size_t> &_g_gate_results,
                            std::vector<size_t> &_g_po_results,
                            std::vector<size_t> &_b_pi_results,
                            std::vector<size_t> &_b_gate_results,
                            std::vector<size_t> &_b_po_results,
                            std::vector<size_t> &_found_fault_to_pattern);  

  void _run_cones_good_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                            const Pattern pattern, 
                            const size_t bits,
                            std::vector<size_t> &_g_pi_results,
                            std::vector<size_t> &_g_gate_results,
                            std::vector<size_t> &_g_po_results);

  void _run_cones_bad_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                          const Fault<IndexType> &fault, 
                          const Pattern pattern,
                          const size_t bits,
                          std::vector<size_t> &_b_pi_results,
                          std::vector<size_t> &_b_gate_results,
                          std::vector<size_t> &_b_po_results);                            
};


template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_run_cones_gates_gpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                                      std::vector<size_t> &_g_pi_results,
                                                      std::vector<size_t> &_g_gate_results,
                                                      std::vector<size_t> &_g_po_results,
                                                      std::vector<size_t> &_b_pi_results,
                                                      std::vector<size_t> &_b_gate_results,
                                                      std::vector<size_t> &_b_po_results,
                                                      std::vector<size_t> &_found_fault_to_pattern) {
  #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    std::cout << "execute simulation._run_cones_gates_gpu();\n";
  #endif
  
    // Simulation
    for (size_t rd = 0; rd < _num_rounds; rd++) {
      size_t num_testcases_this_round =
          ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
              ? (SIZE_T_BITS)
              : (_num_pattern % SIZE_T_BITS);
  #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
      // std::cout << "Run Good case with threads = " << _num_threads << std::endl;
  #endif
      _run_cones_good_case(_Gates, _patterns[rd], num_testcases_this_round, 
                          _g_pi_results, _g_gate_results, _g_po_results);
          
      // // bad simulation (fault simulation)
      // for (size_t j = 0; j < _faults.size(); j++) {
      //   _run_cones_bad_case(_Gates, _faults[j], _patterns[rd], 
      //                       num_testcases_this_round,
      // _b_pi_results, _b_gate_results, _b_po_results);
  
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
      std::cout << "GOOD resutls ans:" << std::endl;
      // print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
    }
  #endif
    }
  }
  
  template<typename IndexType>
  void GALPS_GPUSimulator<IndexType>::_run_cones_good_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                                          const Pattern pattern, 
                                                          const size_t bits,
                                                          std::vector<size_t> &_g_pi_results,
                                                          std::vector<size_t> &_g_gate_results,
                                                          std::vector<size_t> &_g_po_results) {
    printf("Getting inside _run_cones_good_case\n");
  
  //   omp_set_num_threads(_num_threads);  // Set the number of threads for OpenMP
  
  //   // Parallelize across cones (c)
  //   // #pragma omp parallel for
  //   // #pragma omp parallel for if (_cones_partitioned.size() > 1) schedule(dynamic)
  //   for (size_t c = 0; c < _cones_partitioned.size(); c++) {
  //     for (size_t lvl = 0; lvl < _cones_partitioned[c].size(); lvl++) {
  //       // Parallelize across gates within the same cone and level
  //       #pragma omp parallel for
  //       // #pragma omp parallel for schedule(dynamic)
  //       // #pragma omp parallel for if (_cones_partitioned[c][lvl].size()) schedule(dynamic)
  //       for (size_t g = 0; g < _cones_partitioned[c][lvl].size(); g++) {
  //         int gateIdx = _cones_partitioned[c][lvl][g];
          
  //         // Select element (PI, PO, or gate)
  //         ElementBase<IndexType, IndexType> &tmp = _Gates[gateIdx];
  
  //         // Execute the gate operation
  //         _run_gate(tmp, pattern, gateIdx, 0, 0);
  //       }
  //     }
  //   }
    
  //   // Duplicated the answer of good_case into a new memory
  //   _write_to_array(_Gates, _g_pi_results, _g_gate_results, _g_po_results, bits);
  
  // #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  //   std::cout << "CONES: GOOD resutls ans:\n";
  //   // print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
  // #endif
  }
  
  template<typename IndexType>
  void GALPS_GPUSimulator<IndexType>::_run_cones_bad_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                            const Fault<IndexType> &fault, 
                                            const Pattern pattern,
                                            const size_t bits,
                                            std::vector<size_t> &_b_pi_results,
                                            std::vector<size_t> &_b_gate_results,
                                            std::vector<size_t> &_b_po_results) {
    printf("Getting inside _run_cones_bad_case\n");
  //   omp_set_num_threads(_num_threads);  // Set the number of threads for OpenMP
  
  //   IndexType wrong_gate = fault._gate_with_fault;
  //   size_t fault_val = fault._gate_SA_fault_val;
  
  //   // Parallelize across cones (c)
  //   #pragma omp parallel for
  //   for (size_t c = 0; c < _cones_partitioned.size(); c++) {
  //    for (size_t lvl = 0; lvl < _cones_partitioned[c].size(); lvl++) {
  //       // Parallelize across gates within the same cone and level
  //       #pragma omp parallel for
  //       for (size_t g = 0; g < _cones_partitioned[c][lvl].size(); g++) {
  //         int thread_id = omp_get_thread_num();
  //         int gateIdx = _cones_partitioned[c][lvl][g];
  
  //         // Calculate SA_fault using a ternary operator (avoiding if-else)
  //         IndexType SA_fault = (gateIdx < _num_PIs) ? ((gateIdx) == wrong_gate) :
  //                           (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx+_num_PIs+_num_POs) == wrong_gate) :
  //                           ((gateIdx + _num_PIs) == wrong_gate);
  
  //         // Select element (PI, PO, or gate)
  //         ElementBase<IndexType, IndexType> &tmp = _Gates[gateIdx];
  
  //         // // Execute the gate operation
  //         // _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
  //       }
  //     }
  //   }
  
  //   // Duplicated the answer of good_case into a new memory
  //   _write_to_array(_Gates, _b_pi_results, _b_gate_results, _b_po_results, bits);
  
  // #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
  //   std::cout << "CONES: BAD resutls ans:\n";
  //   // print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
  // #endif
  }



  // GPU functions
  template<typename IndexType>
  __device__ __forceinline__ void _apply_INV(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0];
    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }


  template<typename IndexType>
  __device__ __forceinline__ void _apply_AND(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret &= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_OR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret |= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_XOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret ^= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_NAND(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret &= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_NOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret |= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_XNOR(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_adj_gpu[n_loc]];
      ret ^= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_MUX(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _adj_index_table_gpu[2*gate_idx+1]; 

    const size_t a = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
    const size_t b = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+1]];
    const size_t s = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+2]];

    size_t ret = ((s & b) | ( a & (!s)));

    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_CLKBUF(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[gate_idx];
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_PI(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val) {
    _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_PO(const size_t gate_idx, const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _adj_index_table_gpu[2*gate_idx+0]; 

    _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_adj_gpu[s_loc+0]];
  }




  // template<typename IndexType>
  // __global__ void _run_gate(const IndexType *_adj_gpu, const IndexType *_adj_index_table_gpu, 
  //                           size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
  //                           size_t *_patterns_gpu, const size_t rd, 
  //                           size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
  //                           const size_t fault_num, const size_t bad_case, 
  //                           const size_t max_num_gates, const size_t finished_num_gates, const size_t _num_PIs) {  
  //   size_t t_idx = threadIdx.x;
  //   size_t b_idx = blockIdx.x;
  //   size_t real_g_idx = blockDim.x * b_idx + t_idx + finished_num_gates; // tmp
  //   if (real_g_idx > (max_num_gates+finished_num_gates))
  //     return;
    
  //   size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    
  //   if (SA_fault) {
  //     _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
  //     return;
  //   }

  //   size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
  //   size_t num_inputs = _pi_gate_po_num_inputs_gpu[real_g_idx];
  //   switch (type) { 
  //     case 0:
  //       _apply_INV<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 1:
  //       _apply_AND<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 2:
  //       _apply_OR<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 3:
  //       _apply_XOR<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 4:
  //       _apply_NAND<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 5:
  //       _apply_NOR<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 6:
  //       _apply_XNOR<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 7:
  //       _apply_MUX<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 8:
  //       _apply_CLKBUF<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 9:
  //       _apply_PI<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
  //       break;
  //     case 10:
  //       _apply_PO<IndexType>(real_g_idx, _adj_gpu, _adj_index_table_gpu, _pi_gate_po_output_res_gpu);
  //       break;
  //     case 11:
  //       break;
  //     }
  // }





#endif  // GALPS_GPU_SIMULATION_H
