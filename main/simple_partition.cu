#include <iostream>
#include <chrono>

#include <fsim/cuda_partition/partition.cuh>


#define SHOW_ALL_TIME
#define TIMES 1
// #define SEQUENTAIL_VERSION
#ifdef SEQUENTAIL_VERSION
  #define NUM_THREADS 1 // define by user
#else 
  #define PARALLEL_OMP_VERSION
  #define NUM_THREADS 8 // define by user
#endif


constexpr std::chrono::duration<double> chro_zero() {
  return std::chrono::duration<double>::zero();
}

double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // user perspective
  CUDAPartitioner partitioner;

  // Init recording time 
  std::chrono::duration<double> duration_read_data = chro_zero();
  std::chrono::duration<double> duration_CUDAPartition = chro_zero();

  // Inputs 
  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string outputfile_path(argv[4]);

  // Read data and Run partitioner
  auto start = std::chrono::steady_clock::now();
    partitioner.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  
  // Run Simulator
#ifdef SEQUENTAIL_VERSION
  printf("RUN SEQUENTAIL_VERSION\n");
#else
  printf("RUN PARALLEL_OMP_VERSION, with NUM_THREADS = %d\n", NUM_THREADS);
#endif
  for (size_t t = 0; t < TIMES; t++) {
    start = std::chrono::steady_clock::now();
#ifdef SEQUENTAIL_VERSION
    size_t mode = 0;
#else
    size_t mode = 1;
#endif      
      partitioner.run(CUDAPartitioner::Mode(static_cast<size_t>(mode)), NUM_THREADS);
    end = std::chrono::steady_clock::now();
    duration_CUDAPartition += (end - start);
  }
  
  // // Output visualization graph
  // partitioner.visualization_graph(outputfile_path);
  
  // Show time 
#ifdef SHOW_ALL_TIME
  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << "run_CUDA: " <<  round_to(((duration_CUDAPartition.count()/TIMES))*1000, 0.001) << " ms\n";
#else 
  std::cout << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << round_to((duration_CUDAPartition.count())*1000, 0.001) << " ms\n";
#endif 


  return 0;
}