#include <stdio.h>
#include <fsim/cuda/fsim_cuda.cuh>


int main() {
  // user perspective
  fsim::CudaSimulator simulator;
  simulator.do_hello();

  simulator.run(fsim::Simulator::Mode::CUDA, 1);
  return 0;
}