#include <iostream>
#include <chrono>

#include <fsim/cuda_partition/fsim_cuda_partition.cuh>

// #define CPU_PARTITIONER_VERSION
  #ifdef CPU_PARTITIONER_VERSION

  // #define SEQUENTAIL_VERSION
  #ifdef SEQUENTAIL_VERSION
    #define NUM_THREADS 1 // define by user
  #else 
    #define PARALLEL_OMP_VERSION
    #define NUM_THREADS 8 // define by user
  #endif // #ifdef SEQUENTAIL_VERSION
  #endif   // #define CPU_PARTITIONER_VERSION


#ifndef CPU_PARTITIONER_VERSION
  #define GPU_PARTITIONER_VERSION
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

  // Read data and Run cpuPartitioner
  auto start = std::chrono::steady_clock::now();
    cpuPartitioner.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  
    // Run Simulator
  #ifdef SEQUENTAIL_VERSION
    printf("RUN SEQUENTAIL_VERSION\n");
  #else
    printf("RUN PARALLEL_OMP_VERSION, with NUM_THREADS = %d\n", NUM_THREADS);
  #endif   // SEQUENTAIL_VERSION
    for (size_t t = 0; t < TIMES; t++) {
      start = std::chrono::steady_clock::now();
  #ifdef SEQUENTAIL_VERSION
      size_t mode = 0;
  #else
      size_t mode = 1;
  #endif   // SEQUENTAIL_VERSION  
        cpuPartitioner.run(CPUPartitioner::Mode(static_cast<size_t>(mode)), NUM_THREADS);
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
    size_t mode = 0;
    cudaPartitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(mode)));
    end = std::chrono::steady_clock::now();
    duration_Partition += (end - start);
  }
#endif  // GPU_PARTITIONER_VERSION

    
// #ifdef OUTPUT_VISUALIZATION_GRAPH_TO_FILE
//   // Output visualization graph
//   cpuPartitioner.visualization_graph(outputfile_path);
// #endif  // OUTPUT_VISUALIZATION_GRAPH_TO_FILE
  
  // Show time 
#ifdef SHOW_ALL_TIME
  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << "run_CUDA: " <<  round_to(((duration_Partition.count()/TIMES))*1000, 0.001) << " ms\n";
#else 
  std::cout << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << round_to((duration_Partition.count())*1000, 0.001) << " ms\n";
#endif  // SHOW_ALL_TIME

  return 0;
}