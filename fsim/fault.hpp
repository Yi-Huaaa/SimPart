#ifndef FAULT_H
#define FAULT_H

#include <cstddef>

namespace fsim {
class Simulator;
class CUDAPartitioner;
class CUDAMAPartitioner;
}


template<typename IndexType = size_t>
class Fault {

  friend class fsim::Simulator;
  friend class CPUPartitioner;
  friend class CUDAPartitioner;
  friend class CUDAMAPartitioner;
  
  // This is the correct syntax for declaring a template class as a friend
  template<typename T> 
  friend class GALPS_CPUSimulator;
  friend class GALPS_GPUSimulator;
    
public:
  // accessor
  // plain old data type can simply be returned as a copy
  size_t gate_with_fault() const { return _gate_with_fault;}
  size_t gate_SA_fault_val() const { return _gate_SA_fault_val;}


private:
  IndexType _gate_with_fault;
  size_t _gate_SA_fault_val; // 0 or 1
};

#endif
