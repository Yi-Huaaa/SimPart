#ifndef CUDA_PARTITION_H
#define CUDA_PARTITION_H

#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <list>
#include <cublas_v2.h>
#include <set>

/* CPU */
#include <fsim/fsim.hpp>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// constexpr
constexpr int _k = 5; // based on the kernel concurrency
constexpr int QUEUE_SIZE = 1024; // 

// Define the enum class
enum class gpuGateType : int {
  INV = 0,
  AND = 1,
  OR = 2,
  XOR = 3,
  NAND = 4,
  NOR = 5,
  XNOR = 6,
  MUX = 7,
  CLKBUF = 8,
  PI = 9,
  PO = 10,
  MAX_GATE_TYPE = 11
};


/* GPU functions */
__global__ void _init_visited(const int _sum_pi_gates_pos, int *_visited);
__global__ void _frist_level_enqueue(const int _num_PIs, 
                                    int *_queue_data, int *_queue_head, 
                                    int *_queue_tail, int *_visited);
__global__ void _frist_level (const int _num_PIs, 
                              const int *_adj_gpu,
                              const int *_adj_index_table_gpu,
                              const int *_invAdj_gpu,
                              const int *_invAdj_index_table_gpu,                              
                              int *_queue_data, int *_queue_head, int *_queue_tail, 
                              int *_visited, 
                              int *_partitionIndex);
__global__ void _GPU_partitioner (int *_queue_data, int *_queue_head, int *_queue_tail, int *_queue_size,
                                 const int *_adj_gpu, 
                                 const int *_adj_index_table_gpu, 
                                 const int *_invAdj_gpu,
                                 const int *_invAdj_index_table_gpu,
                                 int *_visited,
                                 int *_partitionIndex);
__device__ void _enqueue(const int g_idx, int *_queue_head, int *_queue_tail, int *_visited);
__device__ int _dequeue(int *_queue_data, int *_queue_head);
__device__ bool _canFirstLevelEnqueue(const int _num_PIs,
                                      const int g_idx, 
                                      const int *_invAdj_gpu,
                                      const int *_invAdj_index_table_gpu);
__device__ bool _canEnqueue(const int g_idx, 
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu, 
                            int *_visited);
// For print
__global__ void _gpu_print_read_graph(const int _sum_pi_gates_pos, 
                                      const int _szOfAdj, 
                                      const int *_adj_gpu, 
                                      const int *_invAdj_gpu, 
                                      const int *_adj_index_table_gpu, 
                                      const int *_invAdj_index_table_gpu);
__global__ void _print_first_level (const int _num_PIs, 
                                    const int *_partitionIndex);
__global__ void _print_queue(int *_queue_data, int *_queue_head, int *_queue_tail);
__global__ void _print_GPU_partitioner(const int _sum_pi_gates_pos, 
                                      const int *_partitionIndex);
__global__ void _check_visited(const int sz, const int *_visited, const int mode);


/* Forward declaration */
class CUDASimulator;



class CUDAPartitioner {
  
  // Declare friend class 
  friend class CUDASimulator;

public:

  enum class Mode { PARALLEL_CUDA = 0, CONES_PARALLEL_CUDA };

  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run(Mode mode);

private:
  // Basic private members 
  int _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  int _sum_pi_gates_pos;
  
  // Host memory 
  // Read Graph 
  std::vector<GateType> _gate_type;
  int _szOfAdj; // size of _adj
  int *_adj; // fromGate -> toGate
  int *_invAdj; // toGate -> fromGate
  // TODO: _adj_index_table and _invAdj_index_table table size can be reduced 
  int *_adj_index_table;
  int *_invAdj_index_table;
  // int *_gate_input_order_table;  // toGate -> fromGate
  
  // Faults and Patterns 
  int _num_fault; // for read file
  std::vector<Fault> _faults;

  int _num_pattern; // total number of patterns that need to be tested
  int _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // For levelization and Find cones
  int _max_level;
  int _total_num_levels;


  // GPU memory 
  int *_adj_gpu; 
  int *_invAdj_gpu; 
  int *_adj_index_table_gpu;
  int *_invAdj_index_table_gpu;
  int *_partitionIndex;
  int *_visited;

  const int _NUM_THREADS = 1024;
  
  // New Queue for processing nodes on GPU
  int *_queue_data;
  int *_queue_head;
  int *_queue_tail;
  int *_queue_size;

  /* Read files - CPU function */
  void _read_graph(std::istream &ckt);
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);

  /* GPU function */
  void _ask_gpu_memory();
  void _call_gpu_partitioner();
  int _get_queue_size();




  // void _free_devc_memory();
  // void _levelization();
  // void _partition(); 
  // void _find_all_cones();








  // --------------------------------------------------------------------------

  // Print for check 
  std::string gateTypeToString(GateType type) const;
  void print_patterns(const std::vector<Pattern> &patterns, const int round,
                      const int num_PIs) const;
  void print_bits_stack(const int size, const void *const ptr) const;
  gpuGateType convertGateTypeToGpu(GateType gate_type);
  void _print_read_graph(int *gate_input_order_table_readIn) const;
  void _check_graph_connection_CPU();

};

#endif