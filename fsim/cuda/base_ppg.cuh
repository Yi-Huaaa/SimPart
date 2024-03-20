#ifndef CUDA_BASE_PPG_H
#define CUDA_BASE_PPG_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#define default_level 0
#define default_output_value 0


// Forward declaration
class CUDASimulator;
class Pattern;
class Fault;
class PI;
class Gate;
class PO;
class FFR;


// TOOD: ElementType
enum class GateType { 
  INV = 0, 
  AND, // 1 
  OR, // 2
  XOR,  // 3
  NAND, // 4
  NOR, // 5
  XNOR, // 6
  MUX, // 7
  CLKBUF, // 8
  PI,  // 9
  PO,  // 10
  MAX_GATE_TYPE
};


class ElementBase {

  friend class CUDASimulator;
  
  // friend __global__ void _run_gate(PI*_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, const size_t rd, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t idx_accum, size_t pi_g_po_idx_accum, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs);
  friend __global__ void _shift_to_correct_answer_gpu(size_t *outputs, size_t *_pi_gate_po_output_res_gpu, const size_t bits, const size_t SIZE_T_BITS, const size_t pre_gates_accum, const size_t num);
  friend __global__ void _run_gate_graph(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, size_t *_params_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __global__ void _run_gate_graph_partition(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, size_t *_params_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx);

  friend __global__ void _run_gate(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                GateType *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                const size_t rd, const size_t fault_num, const size_t bad_case, 
                                const size_t max_gate_num_per_level, const size_t fin_num_gates, const size_t _num_PIs);

  // simulation for gates
  friend __device__ __forceinline__ void _apply_INV(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_AND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_OR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_XOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_NAND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_NOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_XNOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_MUX(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_CLKBUF(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);
  friend __device__ __forceinline__ void _apply_PI(size_t gate_idx, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val);
  friend __device__ __forceinline__ void _apply_PO(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu);


public:
  // accessor
  // plain old data type can simply be returned as a copy
  size_t idx() const { return _idx; }
  size_t level() const { return _level; }
  size_t output_value() const { return _output_value; }

  
private:
  GateType _type;
  // std::vector<ElementBase *> _inputs;
  ElementBase **_inputs;

  size_t _num_inputs; // store how many input gates
  
  size_t _idx;
  size_t _level;
  size_t _output_value;
  // size_t _output_val_start_idx; 
  // size_t _output_val_end_idx; 
};

#endif