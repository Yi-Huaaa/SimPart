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
  friend class CUDASimulator;

public:
  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run();
  void visualization_graph(const std::string &outputFile);
  

private:
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t _sum_pi_gates_pos;
  
  size_t *_gate_input_order_table;
  
  std::vector<size_t> _sources; 
  std::vector<size_t> _sinks; 
  size_t _max_level;
  size_t _total_num_levels;
  std::vector<size_t> _level_of_gates;
  std::vector<std::vector<size_t>> _gateIdx_in_each_level;



  // For merge
  // const size_t _merge_threshold = 11; // adjustable  
  const size_t _merge_bound = 2;

  // Construct
  void _read_graph(std::istream &ckt, 
                  std::vector<std::vector<size_t>> &adj,
                  std::vector<std::vector<size_t>> &invAdj);

                  
  void _CPU_levelization(std::vector<std::vector<size_t>> &adj);
  

  void _CPU_partition(std::vector<std::vector<size_t>> &adj);
  bool _gateHasCone(std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    const size_t gateIdx);
  bool _gateHasPostGates(std::vector<std::vector<size_t>> &adj, 
                        const size_t gateIdx, const size_t startLevel);         
  void _getConeIdices(const size_t conesSz, const size_t gateIdx, const size_t startLevel, 
                      bool newCone, 
                      std::vector<std::vector<size_t>> &adj, 
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones);
  void _updateConIdxTable(bool newCone, const size_t gateIdx, 
                          std::vector<size_t> &coneIdxToSinkGateIdx);
  void _update_cones(bool newCone, const size_t gateIdx, 
                    const size_t startLevel, 
                    std::vector<std::vector<std::vector<size_t>>> &cones,
                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    size_t &coneWidth);
  void _CPU_find_all_cones(const size_t startLevel, 
                          std::vector<std::vector<size_t>> &adj, 
                          std::vector<std::vector<std::vector<size_t>>> &cones,
                          std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                          std::vector<size_t> &coneIdxToSinkGateIdx,
                          int &stopLevel);
void _CPU_construct_duplication_table(std::vector<size_t> &coneIdxToSinkGateIdx,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable,
                      const size_t startLevel, const int stopLevel, 
                      const size_t numLevels);

  void _CPU_merge_cones();

  void _GPU_partition();
  void _GPU_memory();
  void _GPU_BFS_find_cones();
  void _GPU_construct_duplication_table();
  void _GPU_merge();

};

#endif