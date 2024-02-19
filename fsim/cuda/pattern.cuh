#ifndef CUDA_PATTERN_H
#define CUDA_PATTERN_H

#include <iostream>
#include <stdio.h>
#include <vector>


class Pattern {

  friend class CudaSimulator;

public:
  // accessor
  // plain old data type can simply be returned as a copy
  // std::vector<size_t> output_value() const { return _value;}
  

private:
  // std::vector<size_t> _value;
  size_t * _value;
};

#endif