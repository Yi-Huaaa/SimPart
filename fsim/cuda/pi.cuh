#ifndef CUDA_PI_H
#define CUDA_PI_H

#include <iostream>
#include <fstream>
#include <string>

#include "./base_ppg.cuh"

// forward declaration 
class CUDASimulator;
class CUDAPartitioner;


class PI : public ElementBase {

  friend class CUDASimulator;
  friend class CUDAPartitioner;
 
public:

  // accessor

private:

};


#endif
