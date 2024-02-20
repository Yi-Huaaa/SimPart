#ifndef CUDA_PATTERN_H
#define CUDA_PATTERN_H

#include <iostream>
#include <stdio.h>
#include <vector>


// Forward declaration
class CudaSimulator;
class ElementBase;
class Fault;

class Pattern {

  friend class CudaSimulator;

  friend __global__ void _run_gate(ElementBase *gate, const Pattern pattern, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level);
  // simulation for gates
  friend __device__ __forceinline__ void _apply_PI(ElementBase &gate, const Pattern pattern, const size_t pi);
  

public:
  // accessor
  // plain old data type can simply be returned as a copy
  // std::vector<size_t> output_value() const { return _value;}
  

private:
  // std::vector<size_t> _value;
  size_t * _value;
};

#endif