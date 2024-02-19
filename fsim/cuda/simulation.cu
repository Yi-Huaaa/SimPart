#include <stdio.h>

#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "simulation.cuh"

__global__ void helloCUDA() {
  size_t t_idx = threadIdx.x, b_idx = blockIdx.x;
  printf("Hello, CUDA!, b_idx = %ld, t_idx = %ld\n", b_idx, t_idx);
}


__global__ void _run_gate_gpu(size_t *inputs){
  size_t num  = 10;
  for (size_t i = 0; i < num; i++) {
    inputs[i] = i+10;
  }

}