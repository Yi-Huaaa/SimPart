#ifndef CUDA_FAULT_H
#define CUDA_FAULT_H

#include <cstddef>

class CudaSimulator;

class Fault {

  friend class CudaSimulator;


public:
  // accessor
  size_t gate_with_fault() const { return _gate_with_fault;}
  size_t gate_SA_fault_val() const { return _gate_SA_fault_val;}


private:
  size_t _gate_with_fault;
  size_t _gate_SA_fault_val; // 0 or 1
};

#endif
