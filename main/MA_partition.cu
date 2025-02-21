#include <iostream>
#include <chrono>
#include <memory>
#include <vector>
#include <iostream>

// #include <fsim/cuda_partition/fsim_cuda_partition.cuh>
#include <fsim/cuda_MA_partition/fsim_cuda_MA_partition.cuh>

constexpr size_t TIMES = 1;
#define NUM_THREADS 16
#define SIMU_MODE 1

constexpr std::chrono::duration<double> chro_zero() {
  return std::chrono::duration<double>::zero();
}

double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cerr << "Usage: " << argv[0] << " <ckt_path> <flst_path> <ptn_path> <hypergraph_path> <outputfile_path> <num_simulation_rds>" << std::endl;
    return 1;
  }

  // Init recording time 
  std::chrono::duration<double> dur_sim = chro_zero();

  // Inputs 
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string hypergraph_path(argv[4]);
  std::string outputfile_path(argv[5]);

  // New input for number of simulation rounds
  size_t _NUM_SIMULATION_RDS = std::atoi(argv[6]);

  // user perspective
  CUDAMAPartitioner cudaMAPartitioner;

  // Read data and Run cudaMAPartitioner
  cudaMAPartitioner.read(ckt_path, flst_path, ptn_path);
  
  // Prepare Simulator
  cudaMAPartitioner.prepare_gpu_simulation();

  // pre-run
  cudaMAPartitioner.run(NUM_THREADS, _NUM_SIMULATION_RDS);
  
  // // Run Simulator
  // for (size_t t = 0; t < TIMES; t++) {
  //   auto start = std::chrono::steady_clock::now();
  //   cudaMAPartitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(SIMU_MODE)), NUM_THREADS, _NUM_SIMULATION_RDS);
  //   auto end = std::chrono::steady_clock::now();
  //   dur_sim += (end - start);
  // }

  // cudaMAPartitioner.freeMem();

  // std::cout << "run_simulator: " <<  round_to(((dur_sim.count()/TIMES))*1000, 0.001) << "\n";


  return 0;
}