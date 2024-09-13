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

#include "../cuda/base_ppg.cuh"
#include "../cuda/fault.cuh"
#include "../cuda/gate.cuh"
#include "../cuda/pattern.cuh"
#include "../cuda/pi.cuh"
#include "../cuda/po.cuh"
#include "../cuda/simulation.cuh"



// Forward declaration

class CUDAPartitioner {
  // Declare friend class 
  // friend class CUDASimulator;

public:
  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run();
  void visualization_graph(const std::string &outputFile);
  

private:
  // gpu functions
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t *_gate_idx_to_tpg_order_idx;
  PI *_PIs;
  Gate *_gates;
  PO *_POs;

  size_t _max_level; // the max number of levels
  size_t _max_gate_size_of_levels;  // the max number of gates in all levels
  size_t *_num_gates_per_level; // records number of gates per level
  size_t _size_of_num_gates_per_level; // _num_gates_per_level.size()

  size_t _num_faults; // for read file
  Fault *_faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  Pattern *_patterns;

  std::vector<size_t> _order; // implementation order after topological sort

  // Construct
  void _read_graph(std::istream &ckt);
  void _partition(std::vector<std::vector<size_t>> &adj_topol,
                  std::vector<std::vector<size_t>> &inv_adj_topol,
                  std::vector<std::vector<size_t>> &gates_per_level,
                  std::vector<size_t> &level_of_gates) ;
  void _find_sets(const size_t gate_idx, const size_t start_level, const size_t end_level, 
                  std::set<size_t> &visited,
                  std::vector<std::vector<size_t>> &adj_topol,
                  std::vector<std::vector<size_t>> &inv_adj_topol,
                  std::vector<size_t> &level_of_gates, 
                  std::vector<std::vector<std::vector<size_t>>> &partition_sets);
  void _find_sub_sets(const size_t set_idx, const size_t start_level, const size_t end_level, 
                    std::vector<std::vector<size_t>> &adj_topol,
                    std::vector<std::vector<size_t>> &inv_adj_topol,
                    std::vector<size_t> &level_of_gates,
                    std::vector<std::vector<std::vector<size_t>>> &partition_sets,
                    std::vector<std::vector<std::vector<size_t>>> &partition_sets_2);
  bool _can_merge(std::vector<std::vector<size_t>> &sub_set);
  void _split_tmp_subsets(std::vector<std::vector<size_t>> &sub_set, 
                          std::vector<size_t> &added_num_gates);
                    
  void _merge_small_sets(const size_t start_level, const size_t end_level, 
                        std::vector<std::vector<std::vector<size_t>>> &partition_sets_2,
                        std::vector<std::vector<std::vector<size_t>>> &_partition_results);

  void _read_fault(std::istream &flst);
  void _read_pattern(std::istream &ptn);

  const size_t _tail_ended_OPT_threshold = 1024;
  const size_t _head_front_OPT_threshold = 1024;
  const size_t _head_front_OPT_threshold_log = 10;
  size_t _total_num_levels;
  
  // compute number of FFRs in each level 
  std::vector<size_t> num_FFRs_in_this_level;

  // Simulation
  size_t *_gate_inputs_list;
  size_t *_gate_inputs_list_start;
  size_t _sum_inputs_gates;
  size_t *_found_fault_to_pattern;
  size_t *_order_cpu;
  size_t _num_total_gates_after_head_end_opt;

};

#endif