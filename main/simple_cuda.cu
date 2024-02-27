#include <iostream>
#include <chrono>

#include <fsim/cuda/fsim_cuda.cuh>

// #define run_cuda
#define run_cuda_graph
// #define visualization
// #define print_output
// #define test_makefile

double round_to(double value, double precision = 1.0){
  return std::round(value / precision) * precision;
}

int main(int argc, char *argv[]) {
  // user perspective
  CUDASimulator simulator;

#ifdef test_makefile
  simulator.do_hello();
#endif

  size_t mode;
  std::chrono::duration<double> duration_read_data, duration_CUDA;

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
  start = std::chrono::steady_clock::now();
    simulator.run(CUDASimulator::Mode(static_cast<size_t>(mode)), 1);
  end = std::chrono::steady_clock::now();
  duration_CUDA = end - start;
#endif
  
#ifdef run_cuda_graph
  mode = 1;
  start = std::chrono::steady_clock::now();
    simulator.run(CUDASimulator::Mode(static_cast<size_t>(mode)), 1);
  end = std::chrono::steady_clock::now();
  duration_CUDA = end - start;
#endif


#ifdef visualization
  // write file: visualization_graph
  simulator.visualization_graph(outputfile_path);
#endif 

  std::cout << "Execution Time:\n";
  std::cout << "read_data: " << round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  std::cout << "run_CUDA: " <<  round_to((duration_CUDA.count())*1000, 0.001) << " ms\n";
  
  return 0;
}