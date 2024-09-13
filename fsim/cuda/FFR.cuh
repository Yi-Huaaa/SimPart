#ifndef CUDA_FFR_H
#define CUDA_FFR_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#define FSIM_FFR_MAX_WIDTH 32 // Adjustable parameter 
#define FSIM_FFR_MAX_DEPTH 32 // Adjustable parameter 


// Forward declaration
  class CUDASimulator;
  class CUDAPartitioner;


class FFR_CPU {

  friend class CUDASimulator;
  friend class CUDAPartitioner;
  

public:


private:
  // todo: 將 _input_FFRs 以及 _FFR_gates 這些會變動大小的東西拆出去外面寫，額外有個 array 才會比較好寫
  // FFR **_input_FFRs; // input FFRs of this FFR
  // ElementBase *_FFR_gates[FSIM_FFR_MAX_DEPTH]; // gates belongs to this FFR

  size_t _level_MAX;
  size_t _level_min;
  size_t _width_MAX;
  size_t _depth;

};

class FFR_CUDA {

  friend class CUDASimulator;

public:


private:
  // todo: 將 _input_FFRs 以及 _FFR_gates 這些會變動大小的東西拆出去外面寫，額外有個 array 才會比較好寫
  // FFR **_input_FFRs; // input FFRs of this FFR
  // ElementBase *_FFR_gates[FSIM_FFR_MAX_DEPTH]; // gates belongs to this FFR

  size_t _level_MAX;
  size_t _level_min;
  size_t _width_MAX;
  size_t _depth;

};

#endif