#ifndef CUDA_SIMULATION_H
#define CUDA_SIMULATION_H

#include <stdio.h>
#include "../simulation.hpp"

// CUDA functions 
__global__ void helloCUDA();


// fsim namespace
namespace fsim {

class CudaSimulator : public Simulator {
public:
  void do_hello() {
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
  }

  void run(Mode mode, const size_t num_threads) {
    helloCUDA<<<1, 2>>>();
    cudaDeviceSynchronize();
  }

  private: 
  
};

  

} // namespace fsim

#endif