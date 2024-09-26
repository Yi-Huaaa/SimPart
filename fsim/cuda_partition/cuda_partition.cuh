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
__device__ const char* gpuGateTypeToString(gpuGateType type);
// For  levelization                                                   
__device__ void enqueue(int *queue, int *tail, int element, 
                        const int capacity);
__device__ int dequeue(int *queue, int *head, int tail, 
                      const int capacity);                                                  
__global__ void bfs_queue_levelize(const size_t _num_PIs, const size_t _sum_pi_gates_pos, 
                                  int *_max_level_devc, int *_total_num_levels_devc,
                                  int *_level_of_gates_devc,
                                  int *_gate_output_order_table_devc,
                                  int *_gate_output_order_table_index_devc);


// For print                                 
__global__ void _print_malloc_transfer_devc_memory(const size_t _sum_pi_gates_pos, 
                                                  const int *_gate_input_order_table_devc, 
                                                  const gpuGateType *_gate_type_devc);
__global__ void _print_levelization(int *_max_level_devc, int *_total_num_levels_devc,
                                    const size_t _sum_pi_gates_pos, 
                                    int *_level_of_gates_devc);



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
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t _sum_pi_gates_pos;
  
  // Host memory 
  // Read Graph 
  size_t _szOfAdj; // size of _gate_output_order_table
  int *_gate_output_order_table; // fromGate -> toGate
  int *_gate_output_order_table_index; 
  int *_gate_input_order_table;  // toGate -> fromGate
  std::vector<GateType> _gate_type;
  
  // Faults and Patterns 
  size_t _num_fault; // for read file
  std::vector<Fault> _faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // For levelization and Find cones
  size_t _max_level;
  size_t _total_num_levels;
  std::vector<size_t> _level_of_gates;
  std::vector<std::vector<size_t>> _gateIdx_in_each_level;
  std::vector<std::vector<std::vector<size_t>>> _cones;


  // Device memory 
  int *_max_level_devc, *_total_num_levels_devc;
  int *_gate_output_order_table_devc; // fromGate -> toGate
  int *_gate_output_order_table_index_devc; 
  int *_gate_input_order_table_devc; 
  gpuGateType* _gate_type_devc;
  int *_level_of_gates_devc;
  int *_gateIdx_in_each_level_devc;












  // // Simulation
  // std::vector<size_t> _g_pi_results;
  // std::vector<size_t> _g_gate_results;
  // std::vector<size_t> _g_po_results;
  // std::vector<size_t> _b_pi_results;
  // std::vector<size_t> _b_gate_results;
  // std::vector<size_t> _b_po_results;
  // std::vector<size_t> _found_fault_to_pattern;
  // size_t _pi_idx_accum, _po_idx_accum, _gate_idx_accum;















  /* Read files - CPU function */
  void _read_graph(std::istream &ckt);
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);
  void _levelization();

  /* GPU function */
  void _malloc_transfer_devc_memory();
  void _free_devc_memory();
  gpuGateType convertGateTypeToGpu(GateType gate_type);


  // void _free_devc_memory();
  // void _levelization();
  // void _partition(); 
  // void _find_all_cones();








  // --------------------------------------------------------------------------

  // Print for check 
  std::string gateTypeToString(GateType type) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                      const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;

};

#endif