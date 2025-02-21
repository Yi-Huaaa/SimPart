#ifndef PATTERN_H
#define PATTERN_H

#include <iostream>
#include <stdio.h>
#include <vector>

namespace fsim {
class Simulator;
class CUDAPartitioner;
class CUDAMAPartitioner;
}

class Pattern {

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
  std::vector<size_t> output_value() const { return _value;}

private:
  std::vector<size_t> _value;
};

#endif