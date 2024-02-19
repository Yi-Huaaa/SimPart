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

  friend __global__ void _run_gate(ElementBase *gate, const Pattern pattern, const size_t pi, const size_t SA_fault, const size_t fault_val);
  // simulation for gates
  friend __device__ void _apply_INV(ElementBase &gate);
  friend __device__ void _apply_AND(ElementBase &gate);
  friend __device__ void _apply_OR(ElementBase &gate);
  friend __device__ void _apply_XOR(ElementBase &gate);
  friend __device__ void _apply_NAND(ElementBase &gate);
  friend __device__ void _apply_NOR(ElementBase &gate);
  friend __device__ void _apply_XNOR(ElementBase &gate);
  friend __device__ void _apply_MUX(ElementBase &gate);
  friend __device__ void _apply_CLKBUF(ElementBase &gate);
  friend __device__ void _apply_PI(ElementBase &gate, const Pattern pattern, const size_t pi);
  friend __device__ void _apply_PO(ElementBase &gate);

public:
  // accessor
  // plain old data type can simply be returned as a copy
  // std::vector<size_t> output_value() const { return _value;}
  

private:
  // std::vector<size_t> _value;
  size_t * _value;
};

#endif