#include <chrono>
#include <thread>
#include <assert.h>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <cmath>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <unordered_set>
#include <utility> 
#include <algorithm>
#include <cstddef>  // For size_t and SIZE_MAX

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "partition.cuh"

#define SIZE_T_MAX SIZE_MAX 

// define print for check 
#define CPU_PARTITIONER_PRINT_FOR_CHECK


// Implement your own remove_if function
template <typename ForwardIterator, typename UnaryPredicate>
ForwardIterator remove_if(ForwardIterator first, ForwardIterator last, UnaryPredicate pred) {
    ForwardIterator result = first;
    while (first != last) {
        if (!pred(*first)) {
            *result = std::move(*first);
            ++result;
        }
        ++first;
    }
    return result;
}

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


// CUDAPartitioner fucntions
void CUDAPartitioner::read(const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {
  printf("Get inside CUDAPartitioner::read first\n");

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path);

  if (!ckt)
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  if (!flst)
    throw std::runtime_error("cannot open fault file "s + flst_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);

  read(ckt, flst, ptn);
}

void CUDAPartitioner::read(std::istream &ckt, std::istream &flst, std::istream &ptn) {
  printf("Get inside CUDAPartitioner::read second\n");

  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 
  printf("_num_PIs = %lu, _num_POs = %lu, _num_inner_gates = %lu, _num_wires = %lu\n", 
          _num_PIs, _num_POs, _num_inner_gates, _num_wires);
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<size_t>> invAdj;
  _read_graph(ckt, adj, invAdj);

  /* Construct GPU */
  _CPU_levelization(adj);
  _CPU_partition(adj);
}


void CUDAPartitioner::_read_graph(std::istream &ckt, 
                                  std::vector<std::vector<size_t>> &adj,
                                  std::vector<std::vector<size_t>> &invAdj) {
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  adj.resize(_sum_pi_gates_pos);
  invAdj.resize(_sum_pi_gates_pos);

  // Map table for recording gate input _order
  // Record for each to_gate, its from_gates's are which _order (#_inputs, A/A1, B/A2, A3/S, A4)
  
  size_t sz = 5*_sum_pi_gates_pos*sizeof(size_t);
  _gate_input_order_table = (size_t*)malloc(sz);
  memset(_gate_input_order_table, 0, sz);

  // Construct adjacent list
  for (size_t i = 0; i < _num_wires; i++) {
    size_t gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (size_t j = 0; j < num_post_gates; j++) {
      size_t gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      adj[gate_0].push_back(gate_tmp);
      invAdj[gate_tmp].push_back(gate_0);

      _gate_input_order_table[gate_tmp*5+pin_tmp] = gate_0;
      _gate_input_order_table[gate_tmp*5+0]++;
    }
  }
  
  // Find all _sources, actually == PIs
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    if (_gate_input_order_table[i*5] == 0) {
      _sources.push_back(i);
    }
  }  

  printf("Check _gate_input_order_table\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("_gate_input_order_table[%lu] = ", i);
    for (size_t j = 0; j < 5; j++) {
      printf("%lu, ", _gate_input_order_table[i*5+j]);
    }
    printf("\n");
  }

  // ---------------------------------------------------------------  


  // Read GateType
  // Note: GateType will not utilize during the step of partition
  std::vector<GateType> gate_type;
  gate_type.resize(_sum_pi_gates_pos);
  for (size_t i = 0; i < _num_PIs; i++) {
    gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs + _num_POs); i < (_sum_pi_gates_pos); i++) {
    size_t t; // type
    ckt >> t;
    gate_type[i] = static_cast<GateType>(t);
  }
}


// CPU version first
void CUDAPartitioner::_CPU_levelization(std::vector<std::vector<size_t>> &adj) {
  printf("_sources.size() = %lu\n", _sources.size());

  _max_level = 0;
  _level_of_gates.resize(_sum_pi_gates_pos, 0);

  for (size_t fromGate = 0; fromGate < _sources.size(); fromGate++) {
    size_t sourceIdx = _sources[fromGate];
    std::queue<size_t> postGates; 
    postGates.push(sourceIdx);
    _level_of_gates[sourceIdx] = 0;

    while (!postGates.empty()) {
      size_t startGateIdx = postGates.front(); 
      postGates.pop();

      for (size_t j = 0; j < adj[startGateIdx].size(); j++) {
        size_t postGateIdx = adj[startGateIdx][j];
        bool update = (_level_of_gates[startGateIdx] >= _level_of_gates[postGateIdx]);
        _level_of_gates[postGateIdx] = (update)  
                                    ? (_level_of_gates[startGateIdx]+1) 
                                    : (_level_of_gates[postGateIdx]);
        _max_level = (_level_of_gates[postGateIdx] > _max_level) 
                  ? (_level_of_gates[postGateIdx]) 
                  : (_max_level);
        if (update) {
          postGates.push(postGateIdx);
        }
      }
    }
  }

  // Find all _sinks 
  for (size_t i = 0; i < adj.size(); i ++) {
    if (adj[i].size() == 0) {
      _sinks.push_back(i);
    }
  }

  _total_num_levels = _max_level + 1;

  // Construct _gateIdx_in_each_level
  _gateIdx_in_each_level.resize(_total_num_levels);
  for (size_t gateIdx = 0; gateIdx < _level_of_gates.size(); gateIdx++) {
    size_t gateLvl = _level_of_gates[gateIdx];
    _gateIdx_in_each_level[gateLvl].push_back(gateIdx);
  }


#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%lu, level = %lu\n", i, _level_of_gates[i]);
  }
  for (size_t i = 0; i < _gateIdx_in_each_level.size(); i++) {
    printf("level_%lu: ", i);
    for (size_t j = 0; j < _gateIdx_in_each_level[i].size(); j++) {
      printf("%lu, ", _gateIdx_in_each_level[i][j]);
    } printf("\n");
  }
  printf("_max_level = %lu, _total_num_levels = %lu\n", 
          _max_level, _total_num_levels);
#endif
}

void CUDAPartitioner::_CPU_partition(std::vector<std::vector<size_t>> &adj) {
  
  bool stop = false; 
  size_t startLevel = _max_level; 

  while(!stop) {
    printf("start _CPU_partition\n");

    std::vector<std::vector<std::vector<size_t>>> cones;
    std::vector<std::vector<size_t>> gateBelongsToWhichCones; 
    std::vector<size_t> coneIdxToSinkGateIdx;
    std::vector<std::vector<std::vector<size_t>>> duplicationTable; 

    int stopLevel = -1;
    _CPU_find_all_cones(startLevel, adj, cones, 
                        gateBelongsToWhichCones, 
                        coneIdxToSinkGateIdx, 
                        stopLevel);
    // Construct  
    const size_t numLevels = startLevel - stopLevel;
    _CPU_construct_duplication_table(coneIdxToSinkGateIdx, 
                                    gateBelongsToWhichCones, 
                                    duplicationTable, 
                                    startLevel, stopLevel,  
                                    numLevels);
    


    // Merge cones 
    _CPU_merge_cones();

    // Update startLevel & conti 
    stop = (stopLevel == -1) ? (true) : (false);
    startLevel = stopLevel;
  }
}

bool CUDAPartitioner::_gateHasCone(std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                                  const size_t gateIdx) {
  return (gateBelongsToWhichCones[gateIdx].size() > 0);
}

// gate has postGates in this time during the construction of its cones 
bool CUDAPartitioner::_gateHasPostGates(std::vector<std::vector<size_t>> &adj, 
                                      const size_t gateIdx, const size_t startLevel) {
  if ((_level_of_gates[gateIdx] == startLevel) || 
      (adj[gateIdx].size() == 0)) {
    return false; 
  }
  
  // All of its post gates's level are larger than startLevel, which means that they have been processed  
  size_t i = 0; 
  while(i < adj[gateIdx].size()) {
    size_t postGateIdx = adj[gateIdx][i];
    if (startLevel >= _level_of_gates[postGateIdx]) {
      return true; 
    }
    i++;
  }

  return false;
}

void CUDAPartitioner::_getConeIdices(const size_t conesSz, const size_t gateIdx, 
                                    const size_t startLevel, bool newCone, 
                                    std::vector<std::vector<size_t>> &adj, 
                                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones) {
  if (newCone) {
    gateBelongsToWhichCones[gateIdx].push_back(conesSz);
  } else {
    std::unordered_set<size_t> unique_numbers;
    for (size_t postGate = 0; postGate < adj[gateIdx].size(); postGate++) {
      size_t postGateIdx = adj[gateIdx][postGate];
      if (startLevel >= _level_of_gates[postGateIdx]) {
        for (size_t c = 0; c < gateBelongsToWhichCones[postGateIdx].size(); c++) {
          size_t postGateCone = gateBelongsToWhichCones[postGateIdx][c];
          unique_numbers.insert(postGateCone);
        }
      }
    }
    // Update ret
    for (const size_t& conIdx : unique_numbers) {
      gateBelongsToWhichCones[gateIdx].push_back(conIdx);
    }
  }

  // printf("\tgateIdx = %lu: Final gateBelongsToWhichCones = [", gateIdx);
  // for (size_t i = 0; i < gateBelongsToWhichCones[gateIdx].size(); i++) {
  //   printf("%lu, ", gateBelongsToWhichCones[gateIdx][i]);
  // } printf("]\n");
}

void CUDAPartitioner::_updateConIdxTable(bool newCone, const size_t gateIdx, 
                      std::vector<size_t> &coneIdxToSinkGateIdx) {
  if (newCone) {
    coneIdxToSinkGateIdx.push_back(gateIdx);
  }
}

void CUDAPartitioner::_update_cones(bool newCone, const size_t gateIdx, 
                                    const size_t startLevel, 
                                    std::vector<std::vector<std::vector<size_t>>> &cones,
                                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones, 
                                    size_t &coneWidth) {
  // If no cone, construct a new cone 
  if (newCone) {
    cones.push_back(std::vector<std::vector<size_t>>());
  }

  const size_t coneLevel = startLevel - _level_of_gates[gateIdx]; 
  // Push the gate to all of the cones it belongs to 
  for (size_t cIdx = 0; cIdx < gateBelongsToWhichCones[gateIdx].size(); cIdx++) {
    size_t coneIdx = gateBelongsToWhichCones[gateIdx][cIdx];
    while(coneLevel >= cones[coneIdx].size()) {
      cones[coneIdx].push_back(std::vector<size_t>());
    }
    cones[coneIdx][coneLevel].push_back(gateIdx);
    coneWidth = (cones[coneIdx][coneLevel].size() > coneWidth) 
                ? (cones[coneIdx][coneLevel].size()) 
                : (coneWidth);
  }
}

/* Doing
For finding all cones: needing to find all cones (two kinds)
1. those cones start from the start merged level 
2. those cones are not starting from the start merged level 
*/
void CUDAPartitioner::_CPU_find_all_cones(const size_t startLevel, 
                      std::vector<std::vector<size_t>> &adj, 
                      std::vector<std::vector<std::vector<size_t>>> &cones,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<size_t> &coneIdxToSinkGateIdx,
                      int &stopLevel) {
  
  // Backward, traverse all gates and find its cones 
  // A gate can belong to multiple cones 

  gateBelongsToWhichCones.resize(_sum_pi_gates_pos);

  for (int level = startLevel; level > stopLevel; level--) {
    size_t coneWidth = 0;
    for (size_t g = 0; g < _gateIdx_in_each_level[level].size(); g++) {
      size_t gateIdx = _gateIdx_in_each_level[level][g];
      bool hasCone = _gateHasCone(gateBelongsToWhichCones, gateIdx);
      bool hasPostGate = _gateHasPostGates(adj, gateIdx, startLevel);
      bool newCone = (!hasCone && !hasPostGate);
      // printf("gateIdx = %lu, newCone = %d\n", gateIdx, newCone);

      // It can have multiple cone Idices 
      _getConeIdices(cones.size(), gateIdx, startLevel, 
                    newCone, adj, gateBelongsToWhichCones);
      _updateConIdxTable(newCone, gateIdx, coneIdxToSinkGateIdx);

      // Update cone
      _update_cones(newCone, gateIdx, startLevel, cones, 
                    gateBelongsToWhichCones, coneWidth); 
    }

    // printf("coneWidth = %lu, _merge_bound = %lu, level = %d\n", coneWidth, _merge_bound, level);
    // Ensure that 
    if (coneWidth > _merge_bound) {
      // update stopLevel
      stopLevel = level; // we can only merge to (level+1)
      break;
    }   
  }
#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  printf("startLevel = %lu (include), stopLevel = %d (exclude)\n", 
        startLevel, stopLevel);
  printf("Check Cones: cones.size() = %lu\n", cones.size());
  for (size_t c = 0; c < cones.size(); c++) {
    printf("Cone_%lu:\n", c);
    for (size_t level = 0; level < cones[c].size(); level++) {
      printf("\tlevel_%lu: ", level);
      for (size_t g = 0; g < cones[c][level].size(); g++) {
        printf("%lu, ", cones[c][level][g]);
      } 
      printf("\n");
    }
  }
  printf("gateBelongsToWhichCones:\n");
  for (size_t i = 0; i < gateBelongsToWhichCones.size(); i++) {
    printf("gateBelongsToWhichCones[%lu]: ", i);
    for (size_t j = 0; j < gateBelongsToWhichCones[i].size(); j++) {
      printf("%lu, ", gateBelongsToWhichCones[i][j]);
    } printf("\n");
  }

  printf("coneIdxToSinkGateIdx:\n");
  for (size_t i = 0; i < coneIdxToSinkGateIdx.size(); i++) {
    printf("cone_%lu, sinkGateIdx = %lu\n", i, coneIdxToSinkGateIdx[i]);
  }
#endif 
}

void CUDAPartitioner::_CPU_construct_duplication_table(std::vector<size_t> &coneIdxToSinkGateIdx,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable, 
                      const size_t startLevel, const int stopLevel, 
                      const size_t numLevels) {
  size_t numCones = coneIdxToSinkGateIdx.size();
  printf("Get inside _CPU_construct_duplication_table, numCones = %lu, numLevels = %lu\n", numCones, numLevels);
  printf("stopLevel = %d\n", stopLevel);
  
  std::vector<std::vector<size_t>> usedGates;
  usedGates.resize(numLevels); 

  for (size_t gateIdx = 0; gateIdx < gateBelongsToWhichCones.size(); gateIdx++) {
    bool hasCone = _gateHasCone(gateBelongsToWhichCones, gateIdx);
    bool lvlInCone = (stopLevel == -1) ? (true) : (_level_of_gates[gateIdx] > stopLevel);
    if (hasCone && lvlInCone) {
      const size_t coneLevel = startLevel - _level_of_gates[gateIdx]; 
      printf("gateIdx = %lu, hasCone = %d, coneLevel = %lu\n", gateIdx, hasCone, coneLevel);
      usedGates[coneLevel].push_back(gateIdx);
    }
  }
  
  // Init duplicationTable
  duplicationTable.resize(numLevels);
  for (size_t lvl = 0; lvl < numLevels; lvl++) {
    duplicationTable[lvl].resize(usedGates[lvl].size());
    for (size_t g = 0; g < usedGates[lvl].size(); g++) {
      duplicationTable[lvl][g].resize(usedGates[lvl].size(), 0);
    }
  }

  // Update duplicationTable
  for (size_t lvl = 0; lvl < numLevels; lvl++) {
    for (size_t gate = 0; gate < usedGates[lvl].size(); gate++) {
      size_t gateIdx = usedGates[lvl][gate];
      printf("gateIdx = %lu, ", gateIdx);
      for (size_t c0 = 0; c0 < gateBelongsToWhichCones[gateIdx].size(); c0++) {
        for (size_t c1 = 0; c1 < gateBelongsToWhichCones[gateIdx].size(); c1++) {
          size_t cone0 = gateBelongsToWhichCones[gateIdx][c0];
          size_t cone1 = gateBelongsToWhichCones[gateIdx][c1];
          printf("cone0 = %lu, cone1 = %lu\n", cone0, cone1);
          if (cone0 != cone1) {
            duplicationTable[lvl][cone0][cone1]++;
            duplicationTable[lvl][cone1][cone0]++;
          }
        }
      }
    }  
  }


#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  printf("usedGates:\n");
  for (size_t lvl = 0; lvl < usedGates.size(); lvl++) {
    printf("lvl = %lu\n", lvl);
    for (size_t g = 0; g < usedGates[lvl].size(); g++) {
      printf("%lu, ", usedGates[lvl][g]);
    }
    printf("\n");
  }

  printf("duplicationTable:\n");
  for (size_t lvl = 0; lvl < duplicationTable.size(); lvl++) {
    printf("lvl = %lu\n", lvl);
    for (size_t g1 = 0; g1 < duplicationTable[lvl].size(); g1++) {
      for (size_t g2 = 0; g2 < duplicationTable[lvl][g1].size(); g2++) {
        printf("%lu, ", duplicationTable[lvl][g1][g2]);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
#endif 
}

void CUDAPartitioner::_CPU_merge_cones() {
}


void CUDAPartitioner::run() {
  printf("Get inside CUDAPartitioner::run\n");

}

// ---------------------------------------------------------------


void CUDAPartitioner::_GPU_partition() {
  
  // // Ask GPU memory 
  // _GPU_memory();

  // // BFS for finding all of the cones 
  // _GPU_BFS_find_cones();

  // // Construct duplication table
  // _GPU_construct_duplication_table();

  // // Iteratively merge steps 
  // for (size_t i = 0; i < _merge_threshold; i++) {
  //   _GPU_merge();
  // }
}

void CUDAPartitioner::_GPU_memory() {
  /* we need to know: 
    1. for each gate, which level it belongs to
    2. for each gate, which partition it belongs to 
  */

}
void CUDAPartitioner::_GPU_BFS_find_cones() {

}


void CUDAPartitioner::_GPU_construct_duplication_table() {

}

void CUDAPartitioner::_GPU_merge() {

}

















// output file visualization graph
void CUDAPartitioner::visualization_graph(const std::string &outputFile) {
  using std::string_literals::operator""s;

  // Open output file
  std::ofstream output_file(outputFile);

  // Check if the output file is open
  if (!output_file.is_open()) {
    std::cerr << "Error opening output file: " << outputFile << std::endl;
    return;
  }

  // write file
  output_file << "Hello\n";


  // Close files
  output_file.close();
}
