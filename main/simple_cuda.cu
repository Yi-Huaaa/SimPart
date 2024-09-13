#include <iostream>
#include <chrono>

#include <fsim/cuda/fsim_cuda.cuh>

// #define run_cuda
// #define run_cuda_graph
// #define visualization
#define TIMES 1
#define FSIM_SHOWALL
// #define FSIM_SHOW_ONLY_RUNTIME_OF_RUN


double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // user perspective
  CUDASimulator simulator;

  size_t mode;
  std::chrono::duration<double> duration_read_data, duration_CUDA, duration_CUDA_GRAPH;

  std::string ckt_path(argv[1]);
  std::string flst_path(argv[2]);
  std::string ptn_path(argv[3]);
  std::string outputfile_path(argv[4]);

  // read data and construct graph/fault/pattern
  auto start = std::chrono::steady_clock::now();
    simulator.read(ckt_path, flst_path, ptn_path);
  auto end = std::chrono::steady_clock::now();
  duration_read_data = end - start;
  

#ifdef run_cuda
  mode = 0;
  for (size_t t = 0; t < TIMES; t++) {
    start = std::chrono::steady_clock::now();
      simulator.run(CUDASimulator::Mode(static_cast<size_t>(mode)), 1);
    end = std::chrono::steady_clock::now();
    duration_CUDA += (end - start);
  }
#endif
  
#ifdef run_cuda_graph
  mode = 1;
  for (size_t t = 0; t < TIMES; t++) {
    // printf("t = %ld\n", t);
    start = std::chrono::steady_clock::now();
      simulator.run(CUDASimulator::Mode(static_cast<size_t>(mode)), 1);
    end = std::chrono::steady_clock::now();
    duration_CUDA_GRAPH += (end - start);
  }

#endif
  


  // write file: visualization_graph
#ifdef visualization
  simulator.visualization_graph(outputfile_path);
#endif 

#ifdef FSIM_SHOWALL
  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  #ifdef run_cuda
    std::cout << "run_CUDA: " <<  round_to(((duration_CUDA.count()/TIMES))*1000, 0.001) << " ms\n";
  #endif
  #ifdef run_cuda_graph
    std::cout << "run_CUDA_GRAPH: " <<  round_to(((duration_CUDA_GRAPH.count()/TIMES))*1000, 0.001) << " ms\n";
  #endif
  #ifdef run_cuda_graph_partition
    std::cout << "run_CUDA_GRAPH_PART: " <<  round_to(((duration_CUDA_GRAPH_PARTITION.count()/TIMES))*1000, 0.001) << " ms\n";
  #endif
#endif

#ifdef FSIM_SHOW_ONLY_RUNTIME_OF_RUN
  #ifdef run_cuda
    std::cout << round_to(((duration_CUDA.count()/TIMES))*1000, 0.001) << "\n";
  #endif
  #ifdef run_cuda_graph
    std::cout << round_to(((duration_CUDA_GRAPH.count()/TIMES))*1000, 0.001) << "\n";
  #endif
#endif

  return 0;
}