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

// CPU
#include <fsim/fsim.hpp>


// Forward declaration
class CUDASimulator;

// // constexpr
// constexpr auto SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;

class CUDAPartitioner {
  
  // Declare friend class 
  friend class CUDASimulator;

public:
  enum class Mode { PARALLEL_CUDA = 0, CONES_PARALLEL_CUDA };

  // // accessor
  // void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  // void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  // void run(Mode mode, const size_t num_threads);

private:
  // // Basic private members 
  // size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  // size_t _sum_pi_gates_pos;
  // std::vector<PI> _PIs;
  // std::vector<Gate> _gates;
  // std::vector<PO> _POs;
  // size_t *_gate_input_order_table;
  
  // // Fault and Pattern 
  // size_t _num_fault; // for read file
  // std::vector<Fault> _faults;

  // size_t _num_pattern; // total number of patterns that need to be tested
  // size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  // std::vector<Pattern> _patterns;

  // // For levelization and Find cones
  // std::vector<size_t> _sources; 
  // std::vector<size_t> _sinks; 
  // size_t _max_level;
  // size_t _total_num_levels;
  // std::vector<size_t> _level_of_gates;
  // std::vector<std::vector<size_t>> _gateIdx_in_each_level;
  // std::vector<std::vector<std::vector<size_t>>> _cones;

  // // Simulation
  // std::vector<size_t> _g_pi_results;
  // std::vector<size_t> _g_gate_results;
  // std::vector<size_t> _g_po_results;
  // std::vector<size_t> _b_pi_results;
  // std::vector<size_t> _b_gate_results;
  // std::vector<size_t> _b_po_results;
  // std::vector<size_t> _found_fault_to_pattern;
  // size_t _pi_idx_accum, _po_idx_accum, _gate_idx_accum;


  // void _GPU_partition();
  // void _GPU_memory();
  // void _GPU_BFS_find_cones();
  // void _GPU_construct_duplication_table();
  // void _GPU_merge();

};

#endif