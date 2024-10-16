#include <iostream>
#include <chrono>

#include <fsim/cuda_partition/fsim_cuda_partition.cuh>

#include <memory>
#include <vector>
#include <iostream>

#define CPU_PARTITIONER_VERSION
  #ifdef CPU_PARTITIONER_VERSION
    #define VERSION 6 // VERSION can be 0, 1, 2, 3, 4, 5, 6, 7
    #if VERSION == 0
        #define MODE 0
        #define NUM_THREADS 1
    #elif VERSION == 1
        #define MODE 1
        #define NUM_THREADS 8
    #elif VERSION == 2
        #define MODE 2
        #define NUM_THREADS 1
    #elif VERSION == 3
        #define MODE 3
        #define NUM_THREADS 8
    #elif VERSION == 4
        #define MODE 4
        #define NUM_THREADS 1
    #elif VERSION == 5
        #define MODE 5
        #define NUM_THREADS 1
    #elif VERSION == 6
        #define MODE 6
        #define NUM_THREADS 1                
    #else // VERSION == 7
        #define MODE 7
        #define NUM_THREADS 8
    #endif
  #endif
  
#ifndef CPU_PARTITIONER_VERSION
  #define GPU_PARTITIONER_VERSION
  #define NUM_THREADS 8
  #define MODE 1
#endif   // #ifndef CPU_PARTITIONER_VERSION


// Both Definition 
#define SHOW_ALL_TIME
#define TIMES 1
// #define OUTPUT_VISUALIZATION_GRAPH_TO_FILE


constexpr std::chrono::duration<double> chro_zero() {
  return std::chrono::duration<double>::zero();
}

double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // Init recording time 
  std::chrono::duration<double> duration_read_data = chro_zero();
  std::chrono::duration<double> duration_Partition = chro_zero();

  // Inputs 
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string outputfile_path(argv[4]);


#ifdef CPU_PARTITIONER_VERSION
  // user perspective
  CPUPartitioner cpuPartitioner;
  printf("RUN MODE = %d, with NUM_THREADS = %d\n", MODE, NUM_THREADS);

  // Read data and Run cpuPartitioner
  auto start = std::chrono::steady_clock::now();
    cpuPartitioner.read(CPUPartitioner::Mode(static_cast<size_t>(MODE)), 
                        ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  
    // Run Simulator
    for (size_t t = 0; t < TIMES; t++) {
      start = std::chrono::steady_clock::now();
        cpuPartitioner.run(CPUPartitioner::Mode(static_cast<size_t>(MODE)), NUM_THREADS);
      end = std::chrono::steady_clock::now();
      duration_Partition += (end - start);
    }
#endif  // CPU_PARTITIONER_VERSION


#ifdef GPU_PARTITIONER_VERSION
  printf("GPU_PARTITIONER_VERSION\n");
  
  // user perspective
  CUDAPartitioner cudaPartitioner;

  // Read data and Run cudaPartitioner
  auto start = std::chrono::steady_clock::now();
    cudaPartitioner.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  
  // Run Simulator
  for (size_t t = 0; t < TIMES; t++) {
    start = std::chrono::steady_clock::now();
    cudaPartitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(MODE)), NUM_THREADS);
    end = std::chrono::steady_clock::now();
    duration_Partition += (end - start);
  }

  cudaPartitioner.freeMem();
#endif  // GPU_PARTITIONER_VERSION

    
// #ifdef OUTPUT_VISUALIZATION_GRAPH_TO_FILE
//   // Output visualization graph
//   cpuPartitioner.visualization_graph(outputfile_path);
// #endif  // OUTPUT_VISUALIZATION_GRAPH_TO_FILE
  
  // Show time 
#ifdef SHOW_ALL_TIME
  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << "run_simulator: " <<  round_to(((duration_Partition.count()/TIMES))*1000, 0.001) << " ms\n";
#endif // SHOW_ALL_TIME


  // cpuPartitioner.KaHyParExample();
  // cpuPartitioner.mt_KaHyParExample();

  return 0;
}