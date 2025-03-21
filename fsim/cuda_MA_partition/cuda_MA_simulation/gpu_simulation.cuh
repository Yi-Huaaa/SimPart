#ifndef GALPS_MA_GPU_SIMULATION_H
#define GALPS_MA_GPU_SIMULATION_H
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

// constexpr uint32_t UINT32T_BITS = std::numeric_limits<uint32_t>::digits;


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


// GPU functions pre-declarations
__device__ __forceinline__ void _apply_INV(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_AND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_OR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_XOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_NAND(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_NOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_XNOR(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_MUX(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_CLKBUF(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__device__ __forceinline__ void _apply_PI(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu, const uint32_t pattern_val);
__device__ __forceinline__ void _apply_PO(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu);
__global__ void _run_gate_DSP(const int num_accumGates, const int *_numGates_per_level_gpu,
                              const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, 
                              const int *_pi_gate_po_gate_type_gpu, uint32_t *_pi_gate_po_output_res_gpu, 
                              const uint32_t *_patterns_gpu, const size_t rd, 
                              const int *_fault_gate_idx_gpu, const size_t *_fault_SA_fault_val_gpu,
                              const size_t fault_num, const size_t bad_case, 
                              const int num_gates_per_level, 
                              const int _num_PIs);
__device__ __forceinline__ void _apply_INV_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_AND_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_OR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                            uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_XOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_NAND_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_NOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_XNOR_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_MUX_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                              uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_CLKBUF_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                                  uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _apply_PI_MA(const int gate_idx, 
                                            uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table,
                                            uint32_t *_pi_gate_po_output_res_gpu, 
                                            const uint32_t pattern_va);
__device__ __forceinline__ void _apply_PO_MA(const int gate_idx, const int *_invAdj_gpu, const int *_invAdj_index_table_gpu, uint32_t *_pi_gate_po_output_res_gpu,
                                          uint32_t *cache, const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ int _getSLIdx(const int *_st_ld_CLs_index_table_gpu, const int l, const int cases);
__device__ __forceinline__ void _modify_bits(uint32_t *a, const int p, const int val);
__device__ __forceinline__ int _get_CLIdx(int gateIdx);
__device__ __forceinline__ uint32_t _get_bits(const uint32_t *value, const int p);
__device__ __forceinline__ int _get_g_c(const uint32_t *g_c_table, const int CLIdx);
__device__ __forceinline__ int _get_pos(const int gate_idx, const uint32_t *pos_table, const int CLIdx);
__device__ __forceinline__ uint32_t _get_val(const int gate_idx, 
                                            const uint32_t *_pi_gate_po_output_res_gpu, const uint32_t *cache, 
                                            const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _wb_val(const int gate_idx, const uint32_t wb_val, 
                                      uint32_t *_pi_gate_po_output_res_gpu, uint32_t *cache, 
                                      const uint32_t *g_c_table, const uint32_t *pos_table);
__device__ __forceinline__ void _wb_neg_val(const int gate_idx, const uint32_t wb_val, 
                                            uint32_t *_pi_gate_po_output_res_gpu, uint32_t *cache, 
                                            const uint32_t *g_c_table, const uint32_t *pos_table);                                      
__device__ __forceinline__ void __syncthreadsAllBlocks(uint32_t *_gpu_sync, 
                                                      const int l, 
                                                      const int num_blocks);
__device__ __forceinline__ void __syncthreadsAllBlocks_early_ret(uint32_t *_gpu_sync, const uint32_t acc);
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
                                int *bibicheck);
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
                          int* bibicheck);
__global__ void _write_and_shift_to_array_gpu(const size_t bits, 
                                              uint32_t *_pi_gate_po_output_res_gpu, 
                                              const int total_num_gates);

// print function 
__global__ void _print_simulation_results(const uint32_t *_pi_gate_po_output_res_gpu, const int total_num_gates);


// Forward declaration of CUDAMAPartitioner
class CUDAMAPartitioner;

class GALPS_MA_GPUSimulator {
  // const uint32_t UINT32T_BITS = std::numeric_limits<uint32_t>::digits;

  friend class CUDAMAPartitioner;

public:
  void run_gpu_simulator_DSP_gpu(const int num_PIs, 
                                const int num_inner_gates,
                                const int num_POs, 
                                const int sum_pi_gates_pos,
                                const int num_pattern, 
                                const size_t num_rounds,
                                const int num_fault, 
                                const int *_pi_gate_po_gate_type_gpu,
                                const uint32_t *_patterns_gpu,
                                const int *_fault_gate_idx_gpu,
                                const size_t *_fault_SA_fault_val_gpu,
                                uint32_t *_pi_gate_po_output_res_gpu,
                                std::vector<int> &_numGates_per_level,
                                const int *_numGates_per_level_gpu,
                                const int _total_num_levels,
                                const int *_invAdj_gpu,
                                const int *_invAdj_index_table_gpu,
                                const std::vector<Pattern> _patterns,
                                const size_t NUM_SIMULATION_RDS) {
    // Get vars   
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    // printf("Get inside run_gpu_simulator_DSP_gpu\n");
    
    // Run simulation - Levelization part 
    for (size_t rd = 0; rd < NUM_SIMULATION_RDS; rd++) {
      _run_gates_DSP_gpu(_total_num_levels, 
                        _numGates_per_level,
                        _numGates_per_level_gpu,
                        _invAdj_gpu,
                        _invAdj_index_table_gpu,
                        _pi_gate_po_gate_type_gpu, 
                        _patterns_gpu,
                        _patterns,
                        _fault_gate_idx_gpu,
                        _fault_SA_fault_val_gpu,
                        _pi_gate_po_output_res_gpu);
    } // NUM_SIMULATION_RDS
  }

  void run_gpu_simulator_MA_gpu(const int num_PIs, 
                                const int num_inner_gates,
                                const int num_POs, 
                                const int sum_pi_gates_pos,
                                const int num_pattern, 
                                const size_t num_rounds,
                                const int num_fault, 
                                const int *_pi_gate_po_gate_type_gpu,
                                const uint32_t *_patterns_gpu,
                                const int *_fault_gate_idx_gpu,
                                const size_t *_fault_SA_fault_val_gpu,
                                uint32_t *_pi_gate_po_output_res_gpu,
                                std::vector<int> &_numGates_per_level,
                                const int *_numGates_per_level_gpu,
                                const int _total_num_levels,
                                const int *_invAdj_gpu,
                                const int *_invAdj_index_table_gpu,
                                const std::vector<Pattern> _patterns,
                                const size_t NUM_SIMULATION_RDS,
                                const int *_st_ld_CLIdxs_gpu,
                                const int *_st_ld_positi_gpu,
                                const int *_st_ld_CLs_index_table_gpu,
                                uint32_t *_gpu_sync,
                                const int used_num_blocks,
                                const int num_threads,
                                const uint32_t *_num_needed_blocks_gpu) {
    // Get vars   
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;
    _used_num_blocks = used_num_blocks; 
    _num_threads = num_threads;
    // printf("Get inside run_gpu_simulator_MA_gpu\n");
    
    // Run simulation
    for (size_t rd = 0; rd < NUM_SIMULATION_RDS; rd++) {
      _run_gates_MA_gpu(_total_num_levels, 
                        _numGates_per_level,
                        _numGates_per_level_gpu,
                        _invAdj_gpu,
                        _invAdj_index_table_gpu,
                        _pi_gate_po_gate_type_gpu, 
                        _patterns_gpu,
                        _patterns,
                        _fault_gate_idx_gpu,
                        _fault_SA_fault_val_gpu,
                        _pi_gate_po_output_res_gpu, 
                        _st_ld_CLIdxs_gpu, 
                        _st_ld_positi_gpu, 
                        _st_ld_CLs_index_table_gpu,
                        _gpu_sync,
                        _num_needed_blocks_gpu);
    } // NUM_SIMULATION_RDS
  }


private:
  // vars
  int _num_PIs; 
  int _num_inner_gates;
  int _num_POs; 
  int _sum_pi_gates_pos;
  int _num_pattern; 
  size_t _num_rounds; 
  int _num_fault; 
  int _used_num_blocks;
  int _num_threads;

  // DSP simulation
  void _run_gates_DSP_gpu(const int _total_num_levels, 
                          const std::vector<int> &_numGates_per_level,
                          const int *_numGates_per_level_gpu,
                          const int *_invAdj_gpu,
                          const int *_invAdj_index_table_gpu,
                          const int *_pi_gate_po_gate_type_gpu,
                          const uint32_t *_patterns_gpu,
                          const std::vector<Pattern> _patterns,
                          const int *_fault_gate_idx_gpu,
                          const size_t *_fault_SA_fault_val_gpu,
                          uint32_t *_pi_gate_po_output_res_gpu);   
  void _run_cones_good_case_DSP_gpu(const int _total_num_levels, 
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
                                    uint32_t *_pi_gate_po_output_res_gpu);
  // MA simulation
  void _run_gates_MA_gpu(const int _total_num_levels, 
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
                        const uint32_t *_num_needed_blocks_gpu);
  void _run_cones_good_case_MA_gpu(const int _total_num_levels, 
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
                                    const uint32_t *_num_needed_blocks_gpu); 
                        
};

#endif  // GALPS_MA_GPU_SIMULATION_H