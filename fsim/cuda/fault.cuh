#ifndef CUDA_FAULT_H
#define CUDA_FAULT_H

#include <cstddef>

// Forward declaration
class CUDASimulator;
class ElementBase;
class Pattern;
class CUDAPartitioner;


class Fault {

  friend class CUDASimulator;
  friend class CUDAPartitioner;

public:
  // accessor
  size_t gate_with_fault() const { return _gate_with_fault;}
  size_t gate_SA_fault_val() const { return _gate_SA_fault_val;}


private:
  size_t _gate_with_fault;
  size_t _gate_SA_fault_val; // 0 or 1
};

#endif
