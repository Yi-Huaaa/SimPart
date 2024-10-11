#ifndef GALPS_GPU_SIMULATION_H
#define GALPS_GPU_SIMULATION_H

#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <list>
#include <cublas_v2.h>
#include <set>

#include <fsim/fsim.hpp>

// Testing functions 
__global__ void BIBI_TRYTRY(const int a);
__device__ void bibi_trytry(const int a);

// Simulation functions 

// Forward declaration of CUDAPartitioner
class CUDAPartitioner;


template<typename IndexType = size_t>
class GALPS_GPUSimulator {
  const size_t SIZE_T_BITS = std::numeric_limits<size_t>::digits;

  friend class CUDAPartitioner;

public:
  // Testing functions 
  void HAHAHA(const int a);
  
  // Simulation
  enum class Mode { GPU_SIMUTOR_LEVELIZE = 0, 
                    GPU_SIMUTOR_RUNCONES, };

  // Simulation functions 
  void run_simulation(Mode mode, 
                      const IndexType sum_pi_gates_pos,
                      const IndexType num_PIs,
                      const IndexType num_inner_gates,
                      const IndexType num_POs,
                      const IndexType num_rounds,
                      const IndexType num_faults,
                      const IndexType total_num_levels,
                      const size_t num_threads);
  
private:
  // Testing functions 
  void _hahaha(const int a);

  // Simulation functions
  void _set_params(const IndexType sum_pi_gates_pos,
                  const IndexType num_PIs,
                  const IndexType num_inner_gates,
                  const IndexType num_POs,
                  const IndexType num_rounds,
                  const IndexType num_faults,
                  const IndexType total_num_levels); 

  void _run_levelized_cpu(const size_t num_threads);
  void _run_cones_gates_cpu(const size_t num_threads);
  void _run_levelized_gpu();
  void _run_cones_gates_gpu();
  
  
  // Simulation Variables
  IndexType _sum_pi_gates_pos;
  IndexType _num_PIs;
  IndexType _num_inner_gates;
  IndexType _num_POs;
  IndexType _num_rounds;
  IndexType _num_faults;
  IndexType _total_num_levels;
};



// Templated function (CPU functions) definitions inside the header
// Since I need to use the template here 

// Testing functions 
template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::HAHAHA(const int a) {
  _hahaha(a);
  BIBI_TRYTRY <<< 1, 1 >>> (a);
  cudaDeviceSynchronize();
}  

template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_hahaha(const int a) {
  printf("BBIBI void GALPS_GPUSimulator<IndexType>::_hahaha() a = %d\n\n\n", a);
}



// Simulation functions 
template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_set_params(const IndexType sum_pi_gates_pos,
                                    const IndexType num_PIs,
                                    const IndexType num_inner_gates,
                                    const IndexType num_POs,
                                    const IndexType num_rounds,
                                    const IndexType num_faults,
                                    const IndexType total_num_levels) {
  _sum_pi_gates_pos = sum_pi_gates_pos;
  _num_PIs = num_PIs;
  _num_inner_gates = num_inner_gates;
  _num_POs = num_POs;
  _num_rounds = num_rounds;
  _num_faults = num_faults;
  _total_num_levels = total_num_levels;
}

template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::run_simulation(Mode mode, 
                                                  const IndexType sum_pi_gates_pos,
                                                  const IndexType num_PIs,
                                                  const IndexType num_inner_gates,
                                                  const IndexType num_POs,
                                                  const IndexType num_rounds,
                                                  const IndexType num_faults,
                                                  const IndexType total_num_levels,
                                                  const size_t num_threads) {
  // Copy parameters 
  _set_params(sum_pi_gates_pos,
              num_PIs,
              num_inner_gates,
              num_POs,
              num_rounds,
              num_faults,
              total_num_levels);



  // Run simulation 
  switch (mode) {
    case Mode::GPU_SIMUTOR_LEVELIZE: {
      _run_levelized_gpu();
      break;
    }
    case Mode::GPU_SIMUTOR_RUNCONES: {
      _run_cones_gates_gpu();
      break;
    }
    default: {
      printf("No SUCH CASES\n");
      return; 
    }
  }
}


template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_run_levelized_gpu() {
  printf("void GALPS_GPUSimulator<IndexType>::_run_levelized_gpu()\n");
}

template<typename IndexType>
void GALPS_GPUSimulator<IndexType>::_run_cones_gates_gpu() {
  printf("void GALPS_GPUSimulator<IndexType>::_run_cones_gates_gpu()\n");
}


#endif  // GALPS_GPU_SIMULATION_H
