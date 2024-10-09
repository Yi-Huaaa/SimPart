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


#define FAULT_INDEX_TYPE size_t
#define FAULT_DATA_TYPE  size_t 



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
constexpr int _k = 2; // based on the kernel concurrency
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

  enum class Mode { CONES_PARALLEL_CUDA = 0, };

  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run(Mode mode, const size_t num_threads);

private:
  // Basic private members 
  int _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  int _sum_pi_gates_pos;
  std::vector<PI> _PIs;
  std::vector<Gate> _gates;
  std::vector<PO> _POs;

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
  std::vector<Fault<FAULT_INDEX_TYPE, FAULT_DATA_TYPE>> _faults;

  int _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // For levelization and Find cones
  std::vector<int> _queue_data_cpu; 
  std::vector<int> _partitionIndex_cpu;
  int _queue_tail_cpu;

  std::vector<int> _sources;
  std::vector<int> _sinks; 
  int _max_level;
  int _total_num_levels;
  std::vector<int> _level_of_gates;
  std::vector<std::vector<std::set<int>>> _cones_set;
  std::vector<std::vector<std::vector<int>>> _cones_partitioned;
  
  // Simulation
  std::vector<int> _g_pi_results;
  std::vector<int> _g_gate_results;
  std::vector<int> _g_po_results;
  std::vector<int> _b_pi_results;
  std::vector<int> _b_gate_results;
  std::vector<int> _b_po_results;
  std::vector<int> _found_fault_to_pattern;
  int _pi_idx_accum, _po_idx_accum, _gate_idx_accum;





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
  void _read_graph(std::istream &ckt, int*gate_input_order_table_readIn);
  void _construct_graph(int *gate_input_order_table_readIn);
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);

  /* GPU function */
  void _ask_gpu_memory();
  void _call_gpu_partitioner();
  int _get_queue_size();

  /* Construct cones */
  void _construct_partitioned_groups();
  void _get_topological_sort_resutls();
  void _copy_TS_resutls_to_Host();
  void _levelized();
  void _construct_groups();
  void _construct_cones(std::vector<int> &sinks,
                      std::vector<std::set<int>> &_cones_set_g,
                      std::vector<bool> &visited_cpu);

  // TODO: free memory 



  // CPU Simulation
  void _run_cones_gates(const size_t num_threads);
  void _run_cones_good_case(const Pattern pattern, 
                            const size_t bits, 
                            const size_t num_threads);
  void _run_cones_bad_case(const Fault<FAULT_INDEX_TYPE, FAULT_DATA_TYPE> &fault,
                          const Pattern pattern,
                          const size_t bits, const size_t num_threads);
  void _write_to_array(std::vector<int> &pi_output, 
                      std::vector<int> &gate_output,
                      std::vector<int> &po_output,
                      const size_t bits);
    void _apply_INV(ElementBase<> &gate);
    void _apply_AND(ElementBase<> &gate);
    void _apply_OR(ElementBase<> &gate);
    void _apply_XOR(ElementBase<> &gate);
    void _apply_NAND(ElementBase<> &gate);
    void _apply_NOR(ElementBase<> &gate);
    void _apply_XNOR(ElementBase<> &gate);
    void _apply_MUX(ElementBase<> &gate);
    void _apply_CLKBUF(ElementBase<> &gate);
    void _apply_PI(ElementBase<> &gate, const Pattern pattern, const size_t pi);
    void _apply_PO(ElementBase<> &gate);
    template <typename IndexType, typename LevelType, typename OutputValueType>
    void _run_gate(ElementBase<IndexType, LevelType, OutputValueType> &gate, 
                   const Pattern pattern, 
                   const size_t pi, 
                   const size_t SA_fault, 
                   const OutputValueType fault_val);
    
    void _shift_to_correct_answer(std::vector<int> &results, const size_t bits,
                                  const size_t num_shift_gates);
  
  
  





  // --------------------------------------------------------------------------

  // Print for check 
  std::string gateTypeToString(GateType type) const;
  void print_patterns(const std::vector<Pattern> &patterns, const int round,
                      const int num_PIs) const;
  void print_bits_stack(const int size, const void *const ptr) const;
  gpuGateType convertGateTypeToGpu(GateType gate_type);
  void _print_read_graph(int *gate_input_order_table_readIn) const;
  void _check_graph_connection_CPU();
  void _print_copy_TS_resutls_to_Host();
  void _print_levelized();
  void _print_construct_groups(const std::vector<std::vector<int>> &sinks_groups);
  void _count_duplications();
  // Simulation
  void print_simulation_results(const std::vector<int> &pi_results,
                                const std::vector<int> &gate_results,
                                const std::vector<int> &po_results) const;
  void print_ppg(const ElementBase<> &gate) const;

};

#endif