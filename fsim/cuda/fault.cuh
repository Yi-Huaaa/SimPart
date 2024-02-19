#ifndef CUDA_FAULT_H
#define CUDA_FAULT_H

#include <cstddef>

// Forward declaration
class CudaSimulator;
class ElementBase;
class Pattern;


class Fault {

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
  size_t gate_with_fault() const { return _gate_with_fault;}
  size_t gate_SA_fault_val() const { return _gate_SA_fault_val;}


private:
  size_t _gate_with_fault;
  size_t _gate_SA_fault_val; // 0 or 1
};

#endif
