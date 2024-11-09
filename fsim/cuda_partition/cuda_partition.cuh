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

/* Simulator */
#include <fsim/cpu_simulator/galps_cpu_simulator.cuh>
#include <fsim/cuda_simulator/galps_cuda_simulator.cuh>


#define ELEMENT_INDEX_TYPE_PART int
#define ELEMENT_LEVEL_TYPE_PART int

#define FAULT_INDEX_TYPE_PART int


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
constexpr int _k = 16; // based on the kernel concurrency
constexpr int _THRESHOLD_0 = 1024; // until which level should use levelization 


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
__global__ void _init_visited(const int _sum_pi_gates_pos, int *_visited_gpu);
__global__ void _frist_level_enqueue(const int _num_PIs, 
                                    int *_queue_data, int *_queue_head, 
                                    int *_queue_tail, int *_visited_gpu);
__global__ void _frist_level (const int _num_PIs, 
                              const int *_adj_gpu,
                              const int *_adj_index_table_gpu,
                              const int *_invAdj_gpu,
                              const int *_invAdj_index_table_gpu,                              
                              int *_queue_data, int *_queue_head, int *_queue_tail, 
                              int *_visited_gpu, 
                              int *_partitionIndex_gpu);
__global__ void _GPU_partitioner (int *_queue_data, int *_queue_head, int *_queue_tail, int *_queue_size,
                                 const int *_adj_gpu, 
                                 const int *_adj_index_table_gpu, 
                                 const int *_invAdj_gpu,
                                 const int *_invAdj_index_table_gpu,
                                 int *_visited_gpu,
                                 int *_partitionIndex_gpu);
__device__ void _enqueue(const int g_idx, int *_queue_head, int *_queue_tail, int *_visited_gpu);
__device__ int _dequeue(int *_queue_data, int *_queue_head);
__device__ bool _canFirstLevelEnqueue(const int _num_PIs,
                                      const int g_idx, 
                                      const int *_invAdj_gpu,
                                      const int *_invAdj_index_table_gpu);
__device__ bool _canEnqueue(const int g_idx, 
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu, 
                            int *_visited_gpu);



// For print
__global__ void _gpu_print_read_graph(const int _sum_pi_gates_pos, 
                                      const int _szOfAdj, 
                                      const int *_adj_gpu, 
                                      const int *_invAdj_gpu, 
                                      const int *_adj_index_table_gpu, 
                                      const int *_invAdj_index_table_gpu);
__global__ void _print_first_level (const int _num_PIs, 
                                    const int *_partitionIndex_gpu);
__global__ void _print_queue(int *_queue_data, int *_queue_head, int *_queue_tail);
__global__ void _print_full_queue(int *_queue_data, int *_queue_tail);
__global__ void _print_GPU_partitioner(const int _sum_pi_gates_pos, 
                                      const int *_partitionIndex_gpu);
__global__ void _check_visited(const int sz, const int *_visited_gpu, const int mode);

template<typename IndexType = int>
__global__ void _check_levelize_gpu(const IndexType *_max_level_gpu, 
                                    const IndexType *_total_num_levels_gpu, 
                                    const IndexType _sum_pi_gates_pos, 
                                    const IndexType *_level_of_gates_gpu);
template<typename IndexType>
__global__ void _print_construct_cones_partitioned_gpu(IndexType *_total_numGates_partitioned_gpu,
                                                      IndexType *_cones_partitioned_gpu);
template<typename IndexType>                                                      
__global__ void print_pi_gate_po_gate_type_gpu(const IndexType *_pi_gate_po_gate_type_gpu, const IndexType num);
__global__ void print_patterns_gpu(size_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs);
template<typename IndexType>
__global__ void print_fault_gate_idx_gpu(IndexType *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, IndexType _num_fault);
__global__ void check_cones_partitioned_gpu_visited(int *visited, 
                                                    const int *_cones_partitioned_gpu, 
                                                    const int num1,
                                                    const int total_numGates_partitioned);
__global__ void print_ask_gpu_simulation_memory(const size_t sz1, 
                                                const int *_per_level_of_group_start_accum_gpu,
                                                const size_t sz2,
                                                const int *_cones_partitioned_gpu);



/* Forward declaration */
class CUDASimulator;
template class GALPS_CPUSimulator<ELEMENT_INDEX_TYPE_PART>;

class CUDAPartitioner {
  
  // Declare friend class 
  friend class CUDASimulator;
  friend class GALPS_CPUSimulator<ELEMENT_INDEX_TYPE_PART>;

public:

  enum class Mode { GPU_PARTIOR_CPU_SIMUTOR = 0, GPU_PARTIOR_GPU_SIMUTOR = 1, };

  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);

  // preparation 
  void prepare_cpu_simulation();
  void prepare_gpu_simulation();

  void run(Mode mode, const size_t num_threads, const size_t NUM_SIMULATION_RDS);
  void freeMem() {
    _free();
  }

private:
  // Basic private members 
  int _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  int _sum_pi_gates_pos;
  // Include PIs, POs, Gates, 
  std::vector<ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART>> _Gates;

  // Host memory 
  // Read Graph 
  std::vector<GateType> _gate_type;
  int _szOfAdj; // size of _adj
  int *_adj; // fromGate -> toGate
  int *_invAdj; // toGate -> fromGate
  int *_adj_index_table;
  int *_invAdj_index_table;
  
  // Faults and Patterns 
  int _num_fault; // for read file
  std::vector<Fault<FAULT_INDEX_TYPE_PART>> _faults;

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
  std::vector<std::vector<int>> _gateIdx_in_each_level;
  std::vector<std::vector<std::set<int>>> _cones_set;
  std::vector<std::vector<std::vector<int>>> _cones_partitioned;
  int _threshold_0_level;
  
  // CPU Simulation
  std::vector<size_t> _g_pi_results;
  std::vector<size_t> _g_gate_results;
  std::vector<size_t> _g_po_results;
  std::vector<size_t> _b_pi_results;
  std::vector<size_t> _b_gate_results;
  std::vector<size_t> _b_po_results;
  std::vector<size_t> _found_fault_to_pattern;

  // GPU partitioner memory: 
  int *_adj_gpu; 
  int *_invAdj_gpu; 
  int *_adj_index_table_gpu;
  int *_invAdj_index_table_gpu;
  int *_partitionIndex_gpu;
  int *_visited_gpu;

  const int _NUM_THREADS = 1024;


  
  // GPU partitioner: New Queue for processing nodes on GPU
  int *_queue_data;
  int *_queue_head;
  int *_queue_tail;
  int *_queue_size;


  // GPU simulation pre-process
  std::vector<int> _numGates_per_level_gpu_of_groups; // partitioned cones CPU
  int *_numGates_per_level_gpu_of_groups_gpu;
  
  int *_total_numGates_partitioned_gpu;
  
  std::vector<int> _per_level_of_group_start_accum_tmp;
  int *_per_level_of_group_start_accum_gpu; // start point of each level of each group in _total_numGates_partitioned_gpu
  
  std::vector<int> _cones_partitioned_tmp;
  int *_cones_partitioned_gpu; // partitioned results 

  std::vector<int> _per_level_of_group_start_accum_level_tmp;
  int *_per_level_of_group_start_accum_level_gpu;
  
  std::vector<int> _gateIdx_in_each_level_tmp;
  int *_gateIdx_in_each_level_gpu; 
  

  // GPU Simulation: 
  int *_max_level_gpu;
  int *_total_num_levels_gpu;
  // for levelization
  int *_level_of_gates_gpu;

  int *_pi_gate_po_gate_type_gpu;
  size_t *_patterns_gpu;
  int *_fault_gate_idx_gpu;
  size_t *_fault_SA_fault_val_gpu;
  size_t *_pi_gate_po_output_res_gpu;
  size_t *_g_pi_results_gpu;
  size_t *_g_gate_results_gpu;
  size_t *_g_po_results_gpu;
  size_t *_b_pi_results_gpu;
  size_t *_b_gate_results_gpu;
  size_t *_b_po_results_gpu;
  int *_found_fault_to_pattern_gpu;


  /* Read files - CPU function */
  void _read_graph(std::istream &ckt);
  void _construct_graph();
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);

  /* GPU function */
  void _ask_gpu_memory_part();
  void _call_gpu_partitioner();
  int _get_queue_size();

  /* Construct cones */
  // GPU Simulator preparation - CPU
  void _construct_partitioned_groups(const int simulator);
    void _get_topological_sort_resutls();
      void _copy_TS_resutls_to_Host();
      void _levelized();
    void _construct_groups();
      void _get_sinks_group(std::vector<std::vector<int>> &sinks_groups);
      void _construct_cones(std::vector<int> &sinks,
                          std::vector<std::set<int>> &_cones_set_g,
                          std::vector<bool> &visited_cpu);
  // GPU Simulator preparation - GPU
  void _ask_gpu_simulation_memory();
  void _move_GateType_h2d();
  void _move_patterns_h2d();
  void _move_faults___h2d();




  void _free() {
    // Host
    free(_adj);
    free(_invAdj);
    free(_adj_index_table);
    free(_invAdj_index_table);

    // Device
    cudaFree(_adj_gpu);
    cudaFree(_invAdj_gpu);
    cudaFree(_adj_index_table_gpu);
    cudaFree(_invAdj_index_table_gpu);
    cudaFree(_partitionIndex_gpu);
    cudaFree(_visited_gpu);
    cudaFree(_queue_data);
    cudaFree(_queue_head);
    cudaFree(_queue_tail);
    cudaFree(_queue_size);
    cudaFree(_cones_partitioned_gpu);
    cudaFree(_max_level_gpu);
    cudaFree(_total_num_levels_gpu);
    cudaFree(_level_of_gates_gpu);
    cudaFree(_per_level_of_group_start_accum_gpu);
    cudaFree(_total_numGates_partitioned_gpu);
    cudaFree(_pi_gate_po_gate_type_gpu);
    cudaFree(_patterns_gpu);
    cudaFree(_fault_gate_idx_gpu);
    cudaFree(_fault_SA_fault_val_gpu);
    cudaFree(_pi_gate_po_output_res_gpu);
    cudaFree(_g_pi_results_gpu);
    cudaFree(_g_gate_results_gpu);
    cudaFree(_g_po_results_gpu);
    cudaFree(_b_pi_results_gpu);
    cudaFree(_b_gate_results_gpu);
    cudaFree(_b_po_results_gpu);
    cudaFree(_found_fault_to_pattern_gpu);
    cudaFree(_per_level_of_group_start_accum_level_gpu);
    cudaFree(_gateIdx_in_each_level_gpu);
  }


  // CPU Simulation
  void _ask_simulation_memory() {
    _g_pi_results.resize(_num_PIs);
    _g_gate_results.resize(_num_inner_gates);
    _g_po_results.resize(_num_POs);
    _b_pi_results.resize(_num_PIs);
    _b_gate_results.resize(_num_inner_gates);
    _b_po_results.resize(_num_POs);
    _found_fault_to_pattern.resize(2 * _faults.size());
  }

  


  // --------------------------------------------------------------------------

  // Print for check 
  std::string _gateTypeToString(GateType type) const;
  void _print_patterns(const std::vector<Pattern> &patterns, const int round,
                      const int num_PIs) const;
  void _print_bits_stack(const int size, const void *const ptr) const;
  gpuGateType _convertGateTypeToGpu(GateType gate_type);
  void _print_read_graph() const;
  void _check_graph_connection_CPU();
  void _print_copy_TS_resutls_to_Host();
  void _print_levelized();
  void _print_construct_groups(const std::vector<std::vector<int>> &sinks_groups);
  void _count_duplications(const int simulator);
  // Simulation
  void _print_simulation_results(const std::vector<size_t> &pi_results,
                                const std::vector<size_t> &gate_results,
                                const std::vector<size_t> &po_results) const;
  void _print_ppg(const ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) const;

  double _round_to(double value, double precision = 1.0){
    return std::round(value / precision) * precision;
  }  

};

#endif