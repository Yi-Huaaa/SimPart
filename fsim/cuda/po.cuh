#ifndef CUDA_PO_H
#define CUDA_PO_H

#include <iostream>
#include <fstream>
#include <string>

#include "./base_ppg.cuh"

// forward declaration 
class CUDASimulator;
class CUDAPartitioner;

class PO : public ElementBase {

  friend class CUDASimulator;
  friend class CUDAPartitioner;
 
public:

  // accessor

private:

};


#endif
