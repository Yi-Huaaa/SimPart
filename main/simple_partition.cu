#include <iostream>
#include <chrono>

#include <fsim/cuda_partition/fsim_cuda_partition.cuh>

#include <memory>
#include <vector>
#include <iostream>

constexpr size_t TIMES = 1;

// #define CPU_PRE_CONSTRUCT_HYPERGRAH
// #define CPU_PARTITIONER_VERSION
  #ifdef CPU_PARTITIONER_VERSION
    #define PRE_CONSTRUCT_HG 0 // 0: construct hypergraph / 1: load pre-constructed hypergraph 
    #define VERSION 1 // VERSION can be 0, 1, 2, 3
    #if VERSION == 0   // Without Partitioner / CPU Simulation
        #define MODE 0
        #define NUM_THREADS 16
    #elif VERSION == 1
        #define MODE 1
        #define NUM_THREADS 16
    #elif VERSION == 2   // RepCut Partitioner
        #define MODE 2
        #define NUM_THREADS 16
    #else // VERSION == 3   // MT-RepCut Partitioner
        #define MODE 3
        #define NUM_THREADS 16
    #endif
    #define SIMU_MODE 0   // MODE 0: CPU Simulator / 1: GPU Simulator 
  #endif
  
#ifndef CPU_PARTITIONER_VERSION
  #define GPU_PARTITIONER_VERSION
  #define NUM_THREADS 16
  #define SIMU_MODE 1  // MODE 0: CPU Simulator / 1: GPU Simulator 
#endif   // #ifndef CPU_PARTITIONER_VERSION


// Both Definition 
#define SHOW_ALL_TIME
// #define OUTPUT_VISUALIZATION_GRAPH_TO_FILE


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
  std::chrono::duration<double> duration_simulation = chro_zero();

  // Inputs 
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string hypergraph_path(argv[4]);
  std::string outputfile_path(argv[5]);

  // New input for number of simulation rounds
  size_t _NUM_SIMULATION_RDS = std::atoi(argv[6]);



#ifdef CPU_PARTITIONER_VERSION
  // user perspective
  CPUPartitioner cpuPartitioner;

  // Read data and Run cpuPartitioner
  cpuPartitioner.read(CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
                      ckt_path, flst_path, ptn_path, hypergraph_path, 
                      NUM_THREADS, PRE_CONSTRUCT_HG);

  // Prepare Simulator
  switch (SIMU_MODE) {
    case 0: { // Prepare for CPU simulation
        cpuPartitioner.prepare_cpu_simulation(PRE_CONSTRUCT_HG, CPUPartitioner::Mode(static_cast<size_t>(MODE)));
        break;
    }
    case 1: { // Prepare for GPU simulation
        cpuPartitioner.prepare_gpu_simulation(CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
                                              PRE_CONSTRUCT_HG);
        break;
    }
  }

  // pre-run
  // cpuPartitioner.run(CPUPartitioner::SIMU_Mode(static_cast<size_t>(SIMU_MODE)), 
  //                   CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
  //                   NUM_THREADS_RUN, _NUM_SIMULATION_RDS);    
  // // Run Simulator
  // for (size_t t = 0; t < TIMES; t++) {
  //   auto start = std::chrono::steady_clock::now();
  //     cpuPartitioner.run(CPUPartitioner::SIMU_Mode(static_cast<size_t>(SIMU_MODE)), 
  //                         CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
  //                         NUM_THREADS_RUN, _NUM_SIMULATION_RDS);
  //   auto end = std::chrono::steady_clock::now();
  //   duration_simulation += (end - start);
  // }

  // Run Simulator
  for (size_t t = 0; t < TIMES; t++) {
    auto start = std::chrono::steady_clock::now();
      cpuPartitioner.run(CPUPartitioner::SIMU_Mode(static_cast<size_t>(SIMU_MODE)), 
                          CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
                          NUM_THREADS, _NUM_SIMULATION_RDS);
    auto end = std::chrono::steady_clock::now();
    duration_simulation += (end - start);
  }  

#endif  // CPU_PARTITIONER_VERSION


#ifdef GPU_PARTITIONER_VERSION
  // user perspective
  CUDAPartitioner cudaPartitioner;

  // Read data and Run cudaPartitioner
    cudaPartitioner.read(ckt_path, flst_path, ptn_path);
  
  // Prepare Simulator
  switch (SIMU_MODE) {
    case 0: {
      // printf("run GPU_PARTIOR_CPU_SIMUTOR");
      cudaPartitioner.prepare_cpu_simulation();
      break;
    }
    case 1: {
      // printf("run GPU_PARTIOR_GPU_SIMUTOR");
      cudaPartitioner.prepare_gpu_simulation();
      // cudaPartitioner.prepare_gpu_simulation_new_version();
      break;
    }
  }

  // pre-run
  cudaPartitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(SIMU_MODE)), NUM_THREADS, _NUM_SIMULATION_RDS);
  
  // Run Simulator
  for (size_t t = 0; t < TIMES; t++) {
    auto start = std::chrono::steady_clock::now();
    cudaPartitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(SIMU_MODE)), NUM_THREADS, _NUM_SIMULATION_RDS);
    auto end = std::chrono::steady_clock::now();
    duration_simulation += (end - start);
  }

  cudaPartitioner.freeMem();
#endif  // GPU_PARTITIONER_VERSION

    
// #ifdef OUTPUT_VISUALIZATION_GRAPH_TO_FILE
//   // Output visualization graph
//   cpuPartitioner.visualization_graph(outputfile_path);
// #endif  // OUTPUT_VISUALIZATION_GRAPH_TO_FILE
  
  // Show time 
#ifdef SHOW_ALL_TIME
  std::cout << "run_simulator: " <<  round_to(((duration_simulation.count()/TIMES))*1000, 0.001) << "\n";
#endif // SHOW_ALL_TIME




// #ifdef CPU_PRE_CONSTRUCT_HYPERGRAH
//   // user perspective
//   CPUPartitioner cpuPartitioner;
//   // Read data and Run cpuPartitioner
//   cpuPartitioner.read_pre_construct_HG(CPUPartitioner::Mode(3), 
//                                       ckt_path, flst_path, ptn_path, hypergraph_path, 8);
// #endif 
  


  // cpuPartitioner.KaHyParExample();
  // cpuPartitioner.mt_KaHyParExample();

  return 0;
}