#ifndef CUDA_SIMULATION_H
#define CUDA_SIMULATION_H


// CUDA functions 

__global__ void helloCUDA();
__global__ void _run_gate_gpu(size_t *inputs);


class CudaSimulator {


public:
 
  void do_hello () {
    printf("???\n");
    helloCUDA <<< 1, 1 >>> ();
    
    _init();
    _run_gate_gpu <<< 1, 1 >>> (_gates_gpu);
    cudaDeviceSynchronize();

    printf("check\n");
    size_t num = 10;
    for (size_t i = 0; i < num; i++) {
      printf("%ld, ", _gates_gpu[i]);
    } printf("\n");
  }
  void read(std::string ckt_path, std::string flst_path, std::string ptn_path){
    printf("help\n");
  }

  
  
private: 
  

  void _init () {
    size_t num  = 10;
    cudaMallocManaged(&_gates_gpu, num*sizeof(size_t));
    for (size_t i = 0; i < num; i++) {
      _gates_gpu[i] = i;
    }
  }
  // store data on unified memory 
  size_t *_gates_gpu;


};


#endif