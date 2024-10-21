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
const int _NUM_THREADS = 1024;

// #define GPU_PART_DEBUG_PRINT_SIMULATION
// #define GPU_PART_DEBUG_PRINT_SIMULATION_RESULTS // print simulation resutls 

// GPU functions pre-declarations
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_INV(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_AND(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_OR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_XOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_NAND(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_NOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_XNOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_MUX(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_CLKBUF(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_PI(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val);
template<typename IndexType = size_t>
__device__ __forceinline__ void _apply_PO(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu);
template<typename IndexType>
__global__ void _run_gate(const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, 
                          const IndexType *_cones_partitioned_gpu,
                          const IndexType *_per_level_of_group_start_accum_gpu, const IndexType k_level, 
                          const IndexType *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          const size_t *_patterns_gpu, const size_t rd, 
                          const IndexType *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                          const size_t fault_num, const size_t bad_case, 
                          const IndexType num_gates_per_level, 
                          const IndexType _num_PIs);
  template<typename IndexType>
  __global__ void _write_and_shift_to_array_gpu(const IndexType _num_PIs, 
                                                const IndexType _num_inner_gates, 
                                                const IndexType _num_POs, 
                                                const size_t bits, 
                                                const size_t *_pi_gate_po_output_res_gpu, const IndexType total_num_gates,
                                                size_t *outputs_PIs, size_t *outputs_Gates, size_t *outputs_POs);


// print function 
template<typename IndexType>
__global__  void _print_simulation_results(const IndexType _num_PIs, 
                                          const IndexType _num_inner_gates, 
                                          const IndexType _num_POs, 
                                          const size_t *_g_pi_results_gpu,
                                          const size_t *_g_gate_results_gpu,
                                          const size_t *_g_po_results_gpu);                                                


// Forward declaration of CUDAPartitioner
class CUDAPartitioner;


template<typename IndexType = size_t>
class GALPS_GPUSimulator {
  const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

  friend class CUDAPartitioner;

public:
  // Simulation - (1) levelization (2) cones 
  void run_gpu_simulator_cones_gpu(const IndexType k,
                                  const IndexType num_PIs, 
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
                                  size_t *_pi_gate_po_output_res_gpu,
                                  const std::vector<IndexType> &_numGates_per_level_gpu_of_groups,
                                  const IndexType *_per_level_of_group_start_accum_gpu,
                                  const IndexType *_cones_partitioned_gpu,
                                  const IndexType _total_num_levels,
                                  const IndexType *_invAdj_gpu,
                                  const IndexType *_invAdj_index_table_gpu,
                                  const std::vector<Pattern> _patterns,
                                  size_t *_g_pi_results_gpu,
                                  size_t *_g_gate_results_gpu,
                                  size_t *_g_po_results_gpu,
                                  size_t *_b_pi_results_gpu,
                                  size_t *_b_gate_results_gpu,
                                  size_t *_b_po_results_gpu,
                                  IndexType *_found_fault_to_pattern_gpu) {
    // Get vars   
    _k = k;       
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    
        

    // run simulation
    _run_cones_gates_gpu(_numGates_per_level_gpu_of_groups,
                        _per_level_of_group_start_accum_gpu,
                        _cones_partitioned_gpu, 
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
  }
  // void run_gpu_simulator_level();


private:


  // void _run_level_gates_gpu();
  
  // vars
  IndexType _k;
  IndexType _num_PIs; 
  IndexType _num_inner_gates;
  IndexType _num_POs; 
  IndexType _sum_pi_gates_pos;
  IndexType _num_pattern; 
  size_t _num_rounds; 
  IndexType _num_fault; 


  void _run_cones_gates_gpu(const std::vector<IndexType> &_numGates_per_level_gpu_of_groups,
                            const IndexType *_per_level_of_group_start_accum_gpu,
                            const IndexType *_cones_partitioned_gpu,
                            const IndexType _total_num_levels,
                            const IndexType *_invAdj_gpu,
                            const IndexType *_invAdj_index_table_gpu,
                            const IndexType *_pi_gate_po_gate_type_gpu,
                            const size_t *_patterns_gpu,
                            const std::vector<Pattern> _patterns,
                            const IndexType *_fault_gate_idx_gpu,
                            const size_t *_fault_SA_fault_val_gpu,                            
                            size_t *_g_pi_results_gpu,
                            size_t *_g_gate_results_gpu,
                            size_t *_g_po_results_gpu,
                            size_t *_b_pi_results_gpu,
                            size_t *_b_gate_results_gpu,
                            size_t *_b_po_results_gpu,
                            IndexType *_found_fault_to_pattern_gpu,
                            size_t *_pi_gate_po_output_res_gpu);  

  void _run_cones_good_case_gpu(const std::vector<IndexType> &_numGates_per_level_gpu_of_groups,
                                const IndexType *_per_level_of_group_start_accum_gpu,
                                const IndexType *_cones_partitioned_gpu,
                                const IndexType _total_num_levels,
                                const IndexType *_invAdj_gpu,
                                const IndexType *_invAdj_index_table_gpu,
                                const IndexType *_pi_gate_po_gate_type_gpu,
                                const size_t *_patterns_gpu,
                                const IndexType *_fault_gate_idx_gpu,
                                const size_t *_fault_SA_fault_val_gpu,
                                const size_t bits,
                                const size_t rd, 
                                size_t *_g_pi_results_gpu,
                                size_t *_g_gate_results_gpu,
                                size_t *_g_po_results_gpu,
                                size_t *_pi_gate_po_output_res_gpu);

  void _run_cones_bad_case_gpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                          const Fault<IndexType> &fault, 
                          const Pattern pattern,
                          const size_t bits,
                          std::vector<size_t> &_b_pi_results,
                          std::vector<size_t> &_b_gate_results,
                          std::vector<size_t> &_b_po_results);                            
};


template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_run_cones_gates_gpu(const std::vector<IndexType> &_numGates_per_level_gpu_of_groups,
                                                        const IndexType *_per_level_of_group_start_accum_gpu,
                                                        const IndexType *_cones_partitioned_gpu,
                                                        const IndexType _total_num_levels,
                                                        const IndexType *_invAdj_gpu,
                                                        const IndexType *_invAdj_index_table_gpu,
                                                        const IndexType *_pi_gate_po_gate_type_gpu,
                                                        const size_t *_patterns_gpu,
                                                        const std::vector<Pattern> _patterns,    
                                                        const IndexType *_fault_gate_idx_gpu,
                                                        const size_t *_fault_SA_fault_val_gpu,                                                   
                                                        size_t *_g_pi_results_gpu,
                                                        size_t *_g_gate_results_gpu,
                                                        size_t *_g_po_results_gpu,
                                                        size_t *_b_pi_results_gpu,
                                                        size_t *_b_gate_results_gpu,
                                                        size_t *_b_po_results_gpu,
                                                        IndexType *_found_fault_to_pattern_gpu,
                                                        size_t *_pi_gate_po_output_res_gpu) {
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
      _run_cones_good_case_gpu(_numGates_per_level_gpu_of_groups,
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
      //   _run_cones_bad_case_gpu(_Gates, _faults[j], _patterns[rd], 
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
  
  template<typename IndexType>
  void GALPS_GPUSimulator<IndexType>::_run_cones_good_case_gpu(const std::vector<IndexType> &_numGates_per_level_gpu_of_groups,
                                                              const IndexType *_per_level_of_group_start_accum_gpu,
                                                              const IndexType *_cones_partitioned_gpu,
                                                              const IndexType _total_num_levels,
                                                              const IndexType *_invAdj_gpu,
                                                              const IndexType *_invAdj_index_table_gpu,
                                                              const IndexType *_pi_gate_po_gate_type_gpu,
                                                              const size_t *_patterns_gpu,
                                                              const IndexType *_fault_gate_idx_gpu,
                                                              const size_t *_fault_SA_fault_val_gpu,
                                                              const size_t bits,
                                                              const size_t rd, 
                                                              size_t *_g_pi_results_gpu,
                                                              size_t *_g_gate_results_gpu,
                                                              size_t *_g_po_results_gpu,
                                                              size_t *_pi_gate_po_output_res_gpu) {
  #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
      printf("Getting inside _run_cones_good_case_gpu\n");
  #endif 
    // Create multiple streams using an array
    const IndexType numStreams = _k;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i) {
      cudaStreamCreate(&streams[i]);
    }

    // Copy data to the device and launch kernels in different streams
    for (IndexType k = 0; k < numStreams; k++) {
      const size_t fault_num = 0; const size_t bad_case = 0;
      IndexType num_blocks, num_threads;

      for (IndexType level = 0; level < _total_num_levels ; level++) {
        const IndexType num_gates_per_level = (_numGates_per_level_gpu_of_groups[k*_total_num_levels+level]);
        num_blocks  = (num_gates_per_level > _NUM_THREADS) ? 
                      (num_gates_per_level + _NUM_THREADS - 1)/_NUM_THREADS : 
                      (1);
        num_threads = (num_gates_per_level > _NUM_THREADS) ? 
                      (_NUM_THREADS) : 
                      (num_gates_per_level);
        // printf("k = %d, level = %d, num_blocks = %d, num_threads = %d\n", k, level, num_blocks, num_threads);

        _run_gate <<< num_blocks, num_threads >>> (_invAdj_gpu, _invAdj_index_table_gpu, 
                                                  _cones_partitioned_gpu, 
                                                  _per_level_of_group_start_accum_gpu, k*_total_num_levels+level,
                                                  _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                                  _patterns_gpu, rd, 
                                                  _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                                  fault_num, bad_case, 
                                                  num_gates_per_level, 
                                                  _num_PIs);
#ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    cudaDeviceSynchronize();
    printf("---\n");
#endif      
      }
    }


    // Synchronize all streams
    for (int i = 0; i < numStreams; ++i) {
      cudaStreamSynchronize(streams[i]);
    }

    // Cleanup: destroy streams
    for (int i = 0; i < numStreams; ++i) {
      cudaStreamDestroy(streams[i]);
    }

    // shift and copy answer to the good results
    if (bits < SIZE_T_BITS) {
      IndexType num_blocks = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                    (_sum_pi_gates_pos + _NUM_THREADS - 1)/_NUM_THREADS : 
                    (1);//1;
      IndexType num_threads = (_sum_pi_gates_pos > _NUM_THREADS) ? 
                    (_NUM_THREADS) : 
                    (_sum_pi_gates_pos);    
      _write_and_shift_to_array_gpu<IndexType> <<< num_blocks, num_threads >>> (_num_PIs, 
                                                                                _num_inner_gates, 
                                                                                _num_POs,
                                                                                bits, 
                                                                                _pi_gate_po_output_res_gpu,
                                                                                _sum_pi_gates_pos,
                                                                                _g_pi_results_gpu,
                                                                                _g_gate_results_gpu,
                                                                                _g_po_results_gpu);
    }

  #ifdef GPU_PART_DEBUG_PRINT_SIMULATION
    cudaDeviceSynchronize();  
    printf("---\n");
  #endif  
  }
  
  template<typename IndexType>
  void GALPS_GPUSimulator<IndexType>::_run_cones_bad_case_gpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                            const Fault<IndexType> &fault, 
                                            const Pattern pattern,
                                            const size_t bits,
                                            std::vector<size_t> &_b_pi_results,
                                            std::vector<size_t> &_b_gate_results,
                                            std::vector<size_t> &_b_po_results) {
    // size_t num_blocks, num_threads;
    // const size_t bad_case = 1;
    // size_t accum = 0;

    // // fault simulation
    // // Simulation
    // num_blocks = (_num_PIs > _NUM_THREADS) ? ((_num_PIs + _NUM_THREADS - 1) >> 10) : (1); // 1;  
    // num_threads = (_num_PIs > _NUM_THREADS) ? (_NUM_THREADS) : (_num_PIs); // _num_PIs;
    // // printf("num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
    
    // _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
    //                                           _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
    //                                           _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
    //                                           _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
    //                                           rd, fault_num, bad_case, _num_PIs, accum, _num_PIs);

    // // since the level_0 belongs to PI, the level_MAX belongs to POs
    // accum += _num_PIs;
    // for (size_t i = 1; i < (_total_num_levels - 1); i++) {
    //   num_blocks = (_num_gates_per_level[i] > _NUM_THREADS) ? (_num_gates_per_level[i] + _NUM_THREADS - 1) >> 10 : (1);//1;
    //   num_threads = (_num_gates_per_level[i] > _NUM_THREADS) ? (_NUM_THREADS) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    //   _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
    //                                             _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
    //                                             _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
    //                                             _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
    //                                             rd, fault_num, bad_case, _num_gates_per_level[i], accum, _num_PIs);   
    //   accum += _num_gates_per_level[i];
    // }

    // // POs
    // num_blocks = (_num_POs > _NUM_THREADS) ? ((_num_POs + _NUM_THREADS - 1) >> 10) : (1); // 1;
    // num_threads = (_num_POs > _NUM_THREADS) ? (_NUM_THREADS) : (_num_POs);// _num_POs; //_num_POs;
    // _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
    //                                           _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
    //                                           _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
    //                                           _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
    //                                           rd, fault_num, bad_case, _num_POs, accum, _num_PIs);
                                              

    // // shift and copy answer to the good results
    // if (bits < SIZE_T_BITS) {
    //   num_blocks = (_num_PIs > _NUM_THREADS) ? ((_num_PIs + _NUM_THREADS - 1) >> 10) : (1); // 1;  
    //   num_threads = (_num_PIs > _NUM_THREADS) ? (_NUM_THREADS) : (_num_PIs);     
    //   _write_and_shift_to_array_gpu<IndexType> <<< num_blocks, num_threads >>> (_b_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
      
    //   num_blocks = (_num_inner_gates > _NUM_THREADS) ? (_num_inner_gates + _NUM_THREADS - 1) >> 10 : (1);//1;
    //   num_threads = (_num_inner_gates > _NUM_THREADS) ? (_NUM_THREADS) : (_num_inner_gates);    
    //   _write_and_shift_to_array_gpu<IndexType> <<< num_blocks, num_threads >>> (_b_gate_results_gpu,  _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

    //   num_blocks = (_num_POs > _NUM_THREADS) ? ((_num_POs + _NUM_THREADS - 1) >> 10) : (1); // 1;
    //   num_threads = (_num_POs > _NUM_THREADS) ? (_NUM_THREADS) : (_num_POs);    
    //   _write_and_shift_to_array_gpu<IndexType> <<< num_blocks, num_threads >>> (_b_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
    // } 
  }



  // GPU functions
  template<typename IndexType>
  __device__ __forceinline__ void _apply_INV(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0];
    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }


  template<typename IndexType>
  __device__ __forceinline__ void _apply_AND(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret &= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_OR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret |= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_XOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret ^= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_NAND(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret &= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_NOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret |= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_XNOR(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    size_t ret = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    
    for (IndexType n_loc = s_loc+1; n_loc < e_loc; n_loc++) {
      const size_t now_gate_val = _pi_gate_po_output_res_gpu[_invAdj_gpu[n_loc]];
      ret ^= now_gate_val; 
    }
    _pi_gate_po_output_res_gpu[gate_idx] = ~ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_MUX(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 
    const IndexType e_loc = _invAdj_index_table_gpu[2*gate_idx+1]; 

    const size_t a = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
    const size_t b = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+1]];
    const size_t s = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+2]];

    size_t ret = ((s & b) | ( a & (!s)));

    _pi_gate_po_output_res_gpu[gate_idx] = ret;
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_CLKBUF(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[gate_idx];
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_PI(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val) {
    _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
  }

  template<typename IndexType>
  __device__ __forceinline__ void _apply_PO(const IndexType gate_idx, const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, size_t *_pi_gate_po_output_res_gpu) {
    const IndexType s_loc = _invAdj_index_table_gpu[2*gate_idx+0]; 

    _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_invAdj_gpu[s_loc+0]];
  }




  template<typename IndexType>
  __global__ void _run_gate(const IndexType *_invAdj_gpu, const IndexType *_invAdj_index_table_gpu, 
                            const IndexType *_cones_partitioned_gpu, 
                            const IndexType *_per_level_of_group_start_accum_gpu, const IndexType k_level, 
                            const IndexType *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                            const size_t *_patterns_gpu, const size_t rd, 
                            const IndexType *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                            const size_t fault_num, const size_t bad_case, 
                            const IndexType num_gates_per_level, 
                            const IndexType _num_PIs) {  
    IndexType t_idx = blockDim.x*blockIdx.x + threadIdx.x;
    IndexType k_accum_gates = _per_level_of_group_start_accum_gpu[k_level];
    IndexType real_g_idx = _cones_partitioned_gpu[t_idx + k_accum_gates]; // tmp

    // printf("Hello world, real_g_idx = %d\n", real_g_idx);

    if (num_gates_per_level > t_idx) {
      IndexType SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
      
      // If have fault 
      if (SA_fault) {
        _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
        return;
      }

      IndexType type = _pi_gate_po_gate_type_gpu[real_g_idx];
      switch (type) { 
        case 0:
          _apply_INV<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 1:
          _apply_AND<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 2:
          _apply_OR<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 3:
          _apply_XOR<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 4:
          _apply_NAND<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 5:
          _apply_NOR<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 6:
          _apply_XNOR<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 7:
          _apply_MUX<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 8:
          _apply_CLKBUF<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 9:
          _apply_PI<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
          break;
        case 10:
          _apply_PO<IndexType>(real_g_idx, _invAdj_gpu, _invAdj_index_table_gpu, _pi_gate_po_output_res_gpu);
          break;
        case 11:
          break;
        }      
    }
  }


  template<typename IndexType>
  __global__ void _write_and_shift_to_array_gpu(const IndexType _num_PIs, 
                                                const IndexType _num_inner_gates, 
                                                const IndexType _num_POs, 
                                                const size_t bits, 
                                                const size_t *_pi_gate_po_output_res_gpu, const IndexType total_num_gates,
                                                size_t *outputs_PIs, size_t *outputs_Gates, size_t *outputs_POs) {
  
    IndexType gate_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (gate_idx < total_num_gates) {
      size_t *outputs = (gate_idx < _num_PIs) ? outputs_PIs 
               : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? outputs_POs 
               : outputs_Gates;

      IndexType index_accum = (gate_idx < _num_PIs) ? 0 
                              : ((gate_idx < (_num_PIs + _num_POs)) && (gate_idx >= _num_PIs)) ? _num_PIs 
                              : (_num_PIs+_num_POs);

      outputs[gate_idx-index_accum] = (_pi_gate_po_output_res_gpu[gate_idx] << (SIZE_T_BITS - bits));
      outputs[gate_idx-index_accum] >>= (SIZE_T_BITS - bits);
    } 
  }


// print functions ------------------
  template<typename IndexType>
  __global__  void _print_simulation_results(const IndexType _num_PIs, 
                                            const IndexType _num_inner_gates, 
                                            const IndexType _num_POs, 
                                            const size_t *_g_pi_results_gpu,
                                            const size_t *_g_gate_results_gpu,
                                            const size_t *_g_po_results_gpu){
    for (IndexType i = 0; i < _num_PIs; i++) {
      printf("PI_%d.output = %lu\n", i, _g_pi_results_gpu[i]);
    }
    for (IndexType i = 0; i < _num_inner_gates; i++) {
      printf("Gate_%d.output = %lu\n", i, _g_gate_results_gpu[i]);
    }
    for (IndexType i = 0; i < _num_POs; i++) {
      printf("PO_%d.output = %lu\n", i, _g_po_results_gpu[i]);
    }
  }

#endif  // GALPS_GPU_SIMULATION_H
