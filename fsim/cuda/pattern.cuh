#ifndef CUDA_PATTERN_H
#define CUDA_PATTERN_H

#include <iostream>
#include <stdio.h>
#include <vector>


// Forward declaration
class CUDASimulator;
class ElementBase;
class Fault;
class CUDAPartitioner;

class Pattern {

  friend class CUDASimulator;
  friend class CUDAPartitioner;

  friend __global__ void _run_gate(PI*_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, const size_t rd, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t idx_accum, size_t pi_g_po_idx_accum, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);
  friend __global__ void _run_gate_graph(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, size_t *_params_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __global__ void _run_gate_graph_partition(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, size_t *_params_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx);
  // simulation for gates
  friend __device__ __forceinline__ void _apply_PI(ElementBase &gate, const Pattern pattern, const size_t pi, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);
  

public:
  // accessor
  // plain old data type can simply be returned as a copy
  // std::vector<size_t> output_value() const { return _value;}
  

private:
  // std::vector<size_t> _value;
  size_t * _value;
};

#endif