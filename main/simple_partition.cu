#include <iostream>
#include <chrono>

#include <fsim/cuda_partition/partition.cuh>

constexpr std::chrono::duration<double> chro_zero() {
  return std::chrono::duration<double>::zero();
}

#define TIMES 1

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

  // Read data
  auto start = std::chrono::steady_clock::now();
    partitioner.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  
  // Run partitioner 
  for (size_t t = 0; t < TIMES; t++) {
    start = std::chrono::steady_clock::now();
      partitioner.run();
    end = std::chrono::steady_clock::now();
    duration_CUDAPartition += (end - start);
  }
  
  // Output visualization graph
  partitioner.visualization_graph(outputfile_path);
  
  // Show time 
  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << "run_CUDA: " <<  round_to(((duration_CUDAPartition.count()/TIMES))*1000, 0.001) << " ms\n";



  return 0;
}