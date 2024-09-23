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
#include <cstdio>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "partition.cuh"

#define SIZE_T_MAX SIZE_MAX 

// #define CUDAPAR_OMP_MODE
#ifdef CUDAPAR_OMP_MODE
  #define num_threads 8
#endif 

// #define CPU_PARTITIONER_PRINT_FOR_CHECK
// #define PART_DEBUG_PRINT_GRAPH
// #define PART_DEBUG_PRINT_PATTERNS
// #define PART_DEBUG_PRINT_FAULT_TABLE
// #define PART_DEBUG_PRINT_SIMULATION

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
  // printf("Get inside CUDAPartitioner::read first\n");

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
  // printf("Get inside CUDAPartitioner::read second\n");

  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 
  // printf("_num_PIs = %lu, _num_POs = %lu, _num_inner_gates = %lu, _num_wires = %lu\n", 
  //         _num_PIs, _num_POs, _num_inner_gates, _num_wires);
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<size_t>> invAdj;
  _read_graph(ckt, adj, invAdj);

  /* Partitioning */
  std::vector<std::vector<std::vector<size_t>>> cones;
  _CPU_levelization(adj);
  _CPU_partition(adj, cones);

  /* Construct circuit, Read Pattern, Read Fault */
  _construct_graph(ckt);
  _read_fault(flst);
  _read_pattern(ptn);
}

void CUDAPartitioner::_read_fault(std::istream &flst) {
  flst >> _num_fault;
  _faults.resize(_num_fault);

  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
    // std::cout << wrong_gate << ", " << fault_value << "\n";

    _faults[i]._gate_with_fault = wrong_gate;
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<size_t>::max() : (0);
  }
}

void CUDAPartitioner::_read_pattern(std::istream &ptn) {
  ptn >> _num_pattern;

  _num_rounds = (_num_pattern + SIZE_T_BITS - 1) / SIZE_T_BITS; // ceiling

  _patterns.resize(_num_rounds);
  for (size_t i = 0; i < _num_rounds; i++) {
    _patterns[i]._value.resize(_num_PIs);
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = i;
      ptn >> _patterns[idx]._value[pi];
    }
  }

#ifdef PART_DEBUG_PRINT_PATTERNS
  print_patterns(_patterns, _num_rounds, _num_PIs);
#endif 
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
  assert(_sources.size() == _num_PIs && "_sources.size() != _num_PIs");
  
  // printf("_sources.size() = %lu\n", _sources.size());
  // for (size_t i = 0; i < _num_PIs; i++) {
  //   if (adj[i].size() == 0) {
  //     printf("adj[%lu].size() = %lu;;; gates = ", i, adj[i].size());
  //     for (size_t j = 0; j < adj[i].size(); j++) {
  //       printf("%lu, ", adj[i][j]);
  //     }
  //     printf("\n");
  //   }
  // }
  // printf("\n");

  // for (size_t i = 0; i < _num_PIs; i++) {
  //   printf("invAdj[%lu].size() = %lu;;; gates = ", i, invAdj[i].size());
  //   for (size_t j = 0; j < invAdj[i].size(); j++) {
  //     printf("%lu, ", invAdj[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");




#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  printf("Check _gate_input_order_table\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("_gate_input_order_table[%lu] = ", i);
    for (size_t j = 0; j < 5; j++) {
      printf("%lu, ", _gate_input_order_table[i*5+j]);
    }
    printf("\n");
  }
#endif 
}


void CUDAPartitioner::_construct_graph(std::istream &ckt) {
  // Read GateType
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

  // printf("gate_type:\n");
  // for (size_t i = 0; i < gate_type.size(); i++) {
  //   std::cout << "Gate_" << i << ", gate_type = " << gateTypeToString(gate_type[i]) << "\n";
  // }

  _pi_idx_accum = 0; 
  _po_idx_accum = _num_PIs;
  _gate_idx_accum = (_num_PIs + _num_POs);

  // PIs are stored in the order specified by the `order` array
  _PIs.resize(_num_PIs);
  for (size_t i = 0; i < _num_PIs; i++) {
    size_t gate_id = i;
    _PIs[i]._idx = gate_id;
    _PIs[i]._type = gate_type[gate_id];
    _PIs[i]._level = _level_of_gates[i];
    _PIs[i]._output_value = 0;
  }

  // gates are stored in the order specified by the `order` array
  _gates.resize(_num_inner_gates);
  for (size_t i = 0; i < _num_inner_gates; i++) {
    size_t gate_id = i + _gate_idx_accum;
    _gates[i]._idx = gate_id;
    _gates[i]._type = gate_type[gate_id];
    _gates[i]._level = _level_of_gates[i];
    _gates[i]._output_value = 0;
  }

  // POs are stored in the order specified by the `order` array
  _POs.resize(_num_POs);
  for (size_t i = 0; i < _num_POs; i++) {
    size_t gate_id = i+_po_idx_accum;
    _POs[i]._idx = gate_id;
    _POs[i]._type = gate_type[gate_id];
    _POs[i]._level = _level_of_gates[i];
    _POs[i]._output_value = 0;
  }

  // Give the input gates number
  for (size_t gateIdx = _num_PIs; gateIdx < _sum_pi_gates_pos; gateIdx++) {
    for (size_t j = 1; j < (_gate_input_order_table[gateIdx*5] + 1); j++) { // how many inputs for this gate
      size_t from_gate_id = _gate_input_order_table[gateIdx*5+j];
      size_t to_gate_id = gateIdx;

      if (from_gate_id < _num_PIs) { // PI -> gate
        PI &from_gate =
            _PIs[from_gate_id - _pi_idx_accum];
        Gate &to_gate =
            _gates[to_gate_id - _gate_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("PI_%lu (%lu) -> Gate_%lu (%lu)\n", from_gate_id - _pi_idx_accum, from_gate_id, to_gate_id - _gate_idx_accum, to_gate_id);
      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        Gate &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        PO &to_gate =
            _POs[to_gate_id - _po_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("Gate_%lu (%lu) -> PO_%lu (%lu)\n", from_gate_id - _gate_idx_accum, from_gate_id, to_gate_id - _po_idx_accum, to_gate_id);
      } else { // gate -> gate
        Gate &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        Gate &to_gate =
            _gates[to_gate_id - _gate_idx_accum];
        // printf("Gate_%lu (%lu) -> Gate_%lu (%lu)\n", from_gate_id - _gate_idx_accum, from_gate_id, to_gate_id - _gate_idx_accum, to_gate_id);
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
      }
    }
  }

#ifdef PART_DEBUG_PRINT_GRAPH
  std::cout << "\n_PIs = [\n";
  for (size_t i = 0; i < _num_PIs; i++) {
    print_ppg(_PIs[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_gates = [\n";
  for (size_t i = 0; i < _num_inner_gates; i++) {
    print_ppg(_gates[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_POs = [\n";
  for (size_t i = 0; i < _num_POs; i++) {
    print_ppg(_POs[i]);
  }
  std::cout << "]\n";
#endif
  
}

void CUDAPartitioner::_CPU_levelization(std::vector<std::vector<size_t>> &adj) {
  // printf("start _CPU_levelization\n");
#ifdef CUDAPAR_OMP_MODE
  omp_set_num_threads(num_threads);
#endif 

  _max_level = 0;
  _level_of_gates.resize(_sum_pi_gates_pos, 0);

  // Parallelizing the outer loop with OpenMP
  #pragma omp parallel for shared(adj) reduction(max:_max_level)
  for (size_t fromGate = 0; fromGate < _sources.size(); fromGate++) {
    size_t sourceIdx = _sources[fromGate];
    std::queue<size_t> postGates; 
    postGates.push(sourceIdx);
    _level_of_gates[sourceIdx] = 0;
    // printf("fromGate = %lu, sourceIdx = %lu\n", fromGate, sourceIdx);

    while (!postGates.empty()) {
      size_t startGateIdx = postGates.front(); 
      postGates.pop();
      int tid = omp_get_thread_num();
      // printf("\tstartGateIdx = %lu %lu (tid = %d)\n", startGateIdx, adj[startGateIdx].size(), tid);

      for (size_t j = 0; j < adj[startGateIdx].size(); j++) {
        size_t postGateIdx = adj[startGateIdx][j];
        bool update = (_level_of_gates[startGateIdx] >= _level_of_gates[postGateIdx]);
        _level_of_gates[postGateIdx] = (update)  
                                        ? (_level_of_gates[startGateIdx]+1) 
                                        : (_level_of_gates[postGateIdx]);

        // Update max level in a thread-safe way
        _max_level = std::max(_max_level, _level_of_gates[postGateIdx]);

        if (update) {
          postGates.push(postGateIdx);
        }
      }
    }
  }

  // Note: Since we use OpenMP here, we need to add "omp critical" here
  // But this will make the program run slower, so we remove CPU parallel here 

  // Find all _sinks
  // #pragma omp parallel for
  for (size_t i = 0; i < adj.size(); i++) {
    if (adj[i].size() == 0) {
      // #pragma omp critical
      // {
        _sinks.push_back(i);
      // }
    }
  }

  _total_num_levels = _max_level + 1;

  // Construct _gateIdx_in_each_level
  _gateIdx_in_each_level.resize(_total_num_levels);
  // #pragma omp parallel for
  for (size_t gateIdx = 0; gateIdx < _level_of_gates.size(); gateIdx++) {
    size_t gateLvl = _level_of_gates[gateIdx];
    // #pragma omp critical
    // {
      _gateIdx_in_each_level[gateLvl].push_back(gateIdx);
    // }
  }

  // printf("_max_level = %lu, _total_num_levels = %lu\n", 
  //         _max_level, _total_num_levels);


#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%lu, level = %lu\n", i, _level_of_gates[i]);
  }

  for (size_t i = 0; i < _gateIdx_in_each_level.size(); i++) {
    printf("level_%lu: size() == %lu\n", i, _gateIdx_in_each_level[i].size());
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


void CUDAPartitioner::_CPU_partition(std::vector<std::vector<size_t>> &adj,
                      std::vector<std::vector<std::vector<size_t>>> &cones) {
  
  // printf("start _CPU_partition\n");
  
  bool stop = false; 
  size_t startLevel = _max_level; 

  while(!stop) {
    std::vector<std::vector<size_t>> gateBelongsToWhichCones; 
    std::vector<size_t> coneIdxToSinkGateIdx;
    std::vector<std::vector<std::vector<size_t>>> duplicationTable; 
    std::vector<std::vector<std::vector<size_t>>> outputCones;

    int stopLevel = -1;
    _CPU_find_all_cones(startLevel, adj, cones, 
                        gateBelongsToWhichCones, 
                        coneIdxToSinkGateIdx, 
                        stopLevel);
    // // Construct  
    // const size_t numLevels = startLevel - stopLevel;
    // _CPU_construct_duplication_table(coneIdxToSinkGateIdx, 
    //                                 gateBelongsToWhichCones, 
    //                                 duplicationTable, 
    //                                 startLevel, stopLevel,  
    //                                 numLevels);
    
    // // Merge cones 
    // _CPU_merge_cones(cones, duplicationTable, outputCones);


    // Update startLevel & conti 
    stop = (stopLevel == -1) ? (true) : (false);
    startLevel = stopLevel;
  }
  _CPU_show_cones_size(cones);
}

void CUDAPartitioner::_CPU_show_cones_size(
                      std::vector<std::vector<std::vector<size_t>>> &cones) {
  // printf("cones.size() = %lu\n", cones.size());
  size_t maxWidth = 0, conWithMaxWidth = 0;
  for (size_t c = 0; c < cones.size(); c++) {
    for (size_t lvl = 0; lvl < cones[c].size(); lvl++) {
      bool update = cones[c][lvl].size() > maxWidth;
      maxWidth = (update) 
                ? (cones[c][lvl].size()) : (maxWidth);
      conWithMaxWidth = (update) ? (c) : (conWithMaxWidth);
    }
  }
  // printf("cone_%lu, maxWidth = %lu\n", conWithMaxWidth, maxWidth);

  // for (size_t c = 0; c < cones.size(); c++) {
  //   size_t maxWidth = 0;
  //   for (size_t level = 0; level < cones[c].size(); level++) {
  //     printf("\tlevel_%lu: ", level);
  //     for (size_t g = 0; g < cones[c][level].size(); g++) {
  //       printf("%lu, ", cones[c][level][g]);
  //     } 
  //     printf("\n");
  //   }
  // }  
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
* Note: this version without considering the "_merge_bound"
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

      // It can have multiple cone Idices 
      _getConeIdices(cones.size(), gateIdx, startLevel, 
                    newCone, adj, gateBelongsToWhichCones);
      _updateConIdxTable(newCone, gateIdx, coneIdxToSinkGateIdx);

      // Update cone
      _update_cones(newCone, gateIdx, startLevel, cones, 
                    gateBelongsToWhichCones, coneWidth); 
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

void CUDAPartitioner::_CPU_merge_cones(std::vector<std::vector<std::vector<size_t>>> &cones,
                              std::vector<std::vector<std::vector<size_t>>> &duplicationTable,
                              std::vector<std::vector<std::vector<size_t>>> &outputCones) {
  printf("Get _CPU_merge_cones\n");
  // std::vector<bool> canMerge;
  // canMerge.resize(??);  

  // Steps:
  // for (size_t mr = 0; mr < _mergedRound; mr++) {
  //   _updateCanMerge();
  //   for (m in all possible merge) {
  //     if (canMerge[m]) {
  //       update for each cone, which cone it should merge with (); // construct a table to record
  //     }
  //   }
  
  //   // After merging 
  //   update duplicationTable();   
  // }



  // // After all of the merge steps finished
  // construct the outputCones();
  // // note: before finishing deciding all of the cones should merge with which cones, 
  // // do not construct the outputCones


}


// This is run simulation 
void CUDAPartitioner::run(Mode mode, const size_t num_threads) {
  switch (mode) {
    case Mode::SEQUENTIAL:
      // std::cout << "run _run_gates\n";
      assert(num_threads == 1 && "RUN SEQUENTIAL num_threads setting ERROR");
      _run_gates(num_threads);
      break;
    case Mode::PARALLEL_OMP:
      std::cout << "run _run_parallel_omp\n";
      _run_gates(num_threads);
      break;
    }
#ifdef PART_DEBUG_PRINT_FAULT_TABLE
  size_t m = static_cast<size_t>(mode);
  std::cout << "Mode: " << m << std::endl;
#endif
}

void CUDAPartitioner::_run_gates(const size_t num_threads) {
#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_gates();" << '\n';
#endif

  // printf("_num_rounds = %lu\n", _num_rounds);

  // Simulation
  _g_pi_results.resize(_num_PIs);
  _g_gate_results.resize(_num_inner_gates);
  _g_po_results.resize(_num_POs);
  _b_pi_results.resize(_num_PIs);
  _b_gate_results.resize(_num_inner_gates);
  _b_po_results.resize(_num_POs);
  _found_fault_to_pattern.resize(2 * _faults.size());

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

#ifdef PART_DEBUG_PRINT_SIMULATION
    std::cout << "Run Good case with threads = " << num_threads << std::endl;
#endif

    _run_good_case(_patterns[rd], num_testcases_this_round, num_threads);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _faults.size(); j++) {
      _run_bad_case(_faults[j], _patterns[rd], 
                    num_testcases_this_round, num_threads);

      size_t found_fault = 0;
      for (size_t i = 0; i < _num_POs; i++) {
        if (_g_po_results[i] != _b_po_results[i]) {
          found_fault = 1;
          break;
        }
      }
      // Record whether fault can be found
      _found_fault_to_pattern[2 * j] = found_fault;
      // Record which pattern found the fault
      _found_fault_to_pattern[2 * j + 1] = rd;
    }    


#ifdef PART_DEBUG_PRINT_SIMULATION
    if (rd == 0) {
      std::cout << "GOOD resutls ans:" << std::endl;
      print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
    }
#endif
  }
}


void CUDAPartitioner::_run_good_case(const Pattern pattern, const size_t bits, const size_t num_threads) {
  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  for (size_t lvl = 0; lvl < _gateIdx_in_each_level.size(); lvl++) {
    // printf("lvl = %lu\n", lvl);
    // Parallelize only the second loop (gate index loop)
    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < _gateIdx_in_each_level[lvl].size(); g++) {
      int thread_id = omp_get_thread_num();
      size_t gateIdx = _gateIdx_in_each_level[lvl][g];
      // printf("\tthread_id = %d, gateIdx = %lu\n", thread_id, gateIdx);

      // Select the correct element (PI, PO, or gate)
      ElementBase &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase&>(_PIs[gateIdx - _pi_idx_accum]) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase&>(_gates[gateIdx - _gate_idx_accum]) :
                          static_cast<ElementBase&>(_POs[gateIdx - _po_idx_accum]);

      // Execute the gate operation
      _run_gate(tmp, pattern, gateIdx, 0, 0);
    }
  }

  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _g_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _g_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _g_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_g_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_g_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_g_po_results, bits, _num_POs);
  }

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls ans:\n";
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void CUDAPartitioner::_run_bad_case(const Fault &fault, const Pattern pattern,
                                    const size_t bits, const size_t num_threads) {
  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;

  for (size_t lvl = 0; lvl < _gateIdx_in_each_level.size(); lvl++) {
    // Parallelize only the second loop (gate index loop)
    #pragma omp parallel for schedule(dynamic)
    for (size_t g = 0; g < _gateIdx_in_each_level[lvl].size(); g++) {
      int thread_id = omp_get_thread_num();
      size_t gateIdx = _gateIdx_in_each_level[lvl][g];

      // Calculate SA_fault using a ternary operator (avoiding if-else)
      size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx + _pi_idx_accum) == wrong_gate) :
                        (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx + _gate_idx_accum) == wrong_gate) :
                                                            ((gateIdx + _po_idx_accum) == wrong_gate);

      // Select the correct element (PI, PO, or gate)
      ElementBase &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase&>(_PIs[gateIdx - _pi_idx_accum]) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase&>(_gates[gateIdx - _gate_idx_accum]) :
                          static_cast<ElementBase&>(_POs[gateIdx - _po_idx_accum]);

      // Execute the gate operation
      _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
    }
  }

  // Duplicated the answer of BAD_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    _b_pi_results[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _b_gate_results[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _b_po_results[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(_b_pi_results, bits, _num_PIs);
    _shift_to_correct_answer(_b_gate_results, bits, _num_inner_gates);
    _shift_to_correct_answer(_b_po_results, bits, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "BAD resutls ans:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

// Simulation for gates
void CUDAPartitioner::_apply_INV(ElementBase &gate) {
  ElementBase *pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

void CUDAPartitioner::_apply_AND(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_OR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;
  
  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_XOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_NAND(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_NOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_XNOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_MUX(ElementBase &gate) {
  size_t a = gate._inputs[0]->_output_value;
  size_t b = gate._inputs[1]->_output_value;
  size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | (a & (!s)));

  gate._output_value = (ret);
}

void CUDAPartitioner::_apply_CLKBUF(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}

void CUDAPartitioner::_apply_PI(ElementBase &gate, const Pattern pattern,
                                const size_t pi) {
  gate._output_value = pattern._value[pi];
}

void CUDAPartitioner::_apply_PO(ElementBase &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  ElementBase *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

void CUDAPartitioner::_run_gate(ElementBase &gate, const Pattern pattern,
                                const size_t pi, const size_t SA_fault,
                                const size_t fault_val) {

  if (SA_fault) {
    gate._output_value = fault_val;
    return;
  }
  // std::cout << "\tgateType = " << gateTypeToString(gate._type) << "\n";
  // std::cout << "\tgateIdx = " << gate._idx << "\n";

  switch (gate._type) {
  case GateType::INV:
    _apply_INV(gate);
    break;
  case GateType::AND:
    _apply_AND(gate);
    break;
  case GateType::OR:
    _apply_OR(gate);
    break;
  case GateType::XOR:
    _apply_XOR(gate);
    break;
  case GateType::NAND:
    _apply_NAND(gate);
    break;
  case GateType::NOR:
    _apply_NOR(gate);
    break;
  case GateType::XNOR:
    _apply_XNOR(gate);
    break;
  case GateType::MUX:
    _apply_MUX(gate);
    break;
  case GateType::CLKBUF:
    _apply_CLKBUF(gate);
    break;
  case GateType::PI:
    _apply_PI(gate, pattern, pi);
    break;
  case GateType::PO:
    _apply_PO(gate);
    break;
  case GateType::MAX_GATE_TYPE:
    break;
  }
}

void CUDAPartitioner::_shift_to_correct_answer(std::vector<size_t> &results,
                                               const size_t bits,
                                               const size_t num_shift_gates) {

  for (size_t i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
}




// -----------------------------------------------------------------------

// Output file visualization graph
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

void CUDAPartitioner::print_ppg(const ElementBase &gate) const {
  std::cout << "\n=====\n\n";
  std::cout << "mem_id: " << &gate << "\n";
  std::cout << "idx: " << gate._idx << "\n";
  std::cout << "type: " << static_cast<size_t>(gate._type) << "\n";
  std::cout << "level: " << gate._level << "\n";
  std::cout << "input_gates: [";
  for (size_t i = 0; i < gate._inputs.size(); i++) {
    std::cout << gate._inputs[i] << ", ";
  }
  std::cout << "]\n";
  std::cout << "output_value: " << gate._output_value << "\n";
}

void CUDAPartitioner::print_patterns(const std::vector<Pattern> &patterns,
                                    const size_t round,
                                    const size_t num_PIs) const {
  std::cout << "\n=====\n\n";
    for (size_t i = 0; i < round; i++) {
      std::cout << "[" << SIZE_T_BITS * i << ", " << SIZE_T_BITS * (i + 1)
                << "] bits = [\n";
      for (size_t j = 0; j < num_PIs; j++) {
        print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
      }
    std::cout << "]\n";
  }
}

void CUDAPartitioner::print_bits_stack(const size_t size,
                                      const void *const ptr) const {
  unsigned char *b = (unsigned char *)ptr;
  unsigned char byte;
  int i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      std::cout << static_cast<unsigned>(byte);
    }
  }
  std::cout << "\n";
}

void CUDAPartitioner::print_simulation_results(
                      const std::vector<size_t> &pi_results,
                      const std::vector<size_t> &gate_results,
                      const std::vector<size_t> &po_results) const {
  for (size_t i = 0; i < pi_results.size(); i++) {
    std::cout << "PI_" << i << ".output = " << pi_results[i] << "\n";
  }
  for (size_t i = 0; i < gate_results.size(); i++) {
    std::cout << "Gate_" << i << ".output = " << gate_results[i] << "\n";
  }
  for (size_t i = 0; i < po_results.size(); i++) {
    std::cout << "PO_" << i << ".output = " << po_results[i] << "\n";
  }
  std::cout << "\n\n";
}

// Function to convert GateType to string
std::string CUDAPartitioner::gateTypeToString(GateType type) const {
  switch (type) {
  case GateType::INV:
    return "INV"; // 0
  case GateType::AND:
    return "AND"; // 1
  case GateType::OR:
    return "OR"; // 2
  case GateType::XOR:
    return "XOR"; // 3
  case GateType::NAND:
    return "NAND"; // 4
  case GateType::NOR:
    return "NOR"; // 5
  case GateType::XNOR:
    return "XNOR"; // 6
  case GateType::MUX:
    return "MUX"; // 7
  case GateType::CLKBUF:
    return "CLKBUF"; // 8
  case GateType::PI:
    return "PI"; // 9
  case GateType::PO:
    return "PO"; // 10
  default:
    return "UNKNOWN"; // 11
  }
}