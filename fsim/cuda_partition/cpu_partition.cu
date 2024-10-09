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

#include "cpu_partition.cuh"

#define SIZE_T_MAX SIZE_MAX 

// #define CPU_PAR_OMP_MODE
#ifdef CPU_PAR_OMP_MODE
  #define num_threads 8
#endif 

// #define CPU_PARTITIONER_PRINT_FOR_CHECK
// #define PART_DEBUG_PRINT_GRAPH
// #define PART_DEBUG_PRINT_PATTERNS
// #define PART_DEBUG_PRINT_FAULT_TABLE
// #define PART_DEBUG_PRINT_SIMULATION
// #define PART_DEBUG_REPCUT_PRINT_FIND_CLUSTERS
// #define PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS

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

// CPUPartitioner fucntions
void CPUPartitioner::read(Mode mode, const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {
  // printf("Get inside CPUPartitioner::read first\n");

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path);

  if (!ckt)
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  if (!flst)
    throw std::runtime_error("cannot open fault file "s + flst_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);

  read(mode, ckt, flst, ptn);
}

void CPUPartitioner::read(Mode mode, std::istream &ckt, std::istream &flst, std::istream &ptn) {
  auto start = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_0;
  std::chrono::duration<double> duration_1;


  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<size_t>> invAdj;
  _read_graph(ckt, adj, invAdj);
  
  // levelization
  _levelization(adj);

  // whether partitioning
  switch (mode) {
    case Mode::SEQUENTIAL:
      break;
    case Mode::PARALLEL_OMP:
      break;
    case Mode::CONES_SEQUENTIAL:
      /* Partitioning */
      _partition(adj);
      break;
    case Mode::CONES_PARALLEL_OMP:
      /* Partitioning */
      _partition(adj);
      break;
    case Mode::REPCUT_SEQUENTIAL:
      /* Partitioning */
      _partition(adj);
      _RepCut(duration_0, duration_1);
      break;
    case Mode::REPCUT_PARALLEL_OMP:
      /* Partitioning */
      _partition(adj);
      _RepCut(duration_0, duration_1);
      break;      
    default:
      printf("read: No SUCH CASES\n");
      return; 
  }



  /* Construct circuit, Read Pattern, Read Fault */
  _construct_graph(ckt);
  _read_fault(flst);
  _read_pattern(ptn);
  _ask_simulation_memory();

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_read_data = end - start;
  std::cout << "Inside read_data: " << _round_to((duration_0.count())*1000, 0.001) << " ms\n";
  std::cout << "Inside read_data: " << _round_to((duration_1.count())*1000, 0.001) << " ms\n";
  std::cout << "Inside read_data: " << _round_to((duration_read_data.count())*1000, 0.001) << " ms\n";
  


  _print_num_PIs_Gates_POs(mode);

}


void CPUPartitioner::_read_fault(std::istream &flst) {
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

void CPUPartitioner::_read_pattern(std::istream &ptn) {
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

void CPUPartitioner::_ask_simulation_memory() {
  _g_pi_results.resize(_num_PIs);
  _g_gate_results.resize(_num_inner_gates);
  _g_po_results.resize(_num_POs);
  _b_pi_results.resize(_num_PIs);
  _b_gate_results.resize(_num_inner_gates);
  _b_po_results.resize(_num_POs);
  _found_fault_to_pattern.resize(2 * _faults.size());
}

void CPUPartitioner::_read_graph(std::istream &ckt, 
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
  
#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK

  printf("_sources.size() = %lu\n", _sources.size());
  for (size_t i = 0; i < _num_PIs; i++) {
    if (adj[i].size() == 0) {
      printf("adj[%lu].size() = %lu;;; gates = ", i, adj[i].size());
      for (size_t j = 0; j < adj[i].size(); j++) {
        printf("%lu, ", adj[i][j]);
      }
      printf("\n");
    }
  }
  printf("\n");

  for (size_t i = 0; i < _num_PIs; i++) {
    printf("invAdj[%lu].size() = %lu;;; gates = ", i, invAdj[i].size());
    for (size_t j = 0; j < invAdj[i].size(); j++) {
      printf("%lu, ", invAdj[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Check _gate_input_order_table\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("_gate_input_order_table[%lu] = ", i);
    for (size_t j = 0; j < 5; j++) {
      printf("%lu, ", _gate_input_order_table[i*5+j]);
    }
  }
#endif 
}


void CPUPartitioner::_construct_graph(std::istream &ckt) {
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
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _PIs[from_gate_id - _pi_idx_accum];
            ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
            _gates[to_gate_id - _gate_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("PI_%lu (%lu) -> Gate_%lu (%lu)\n", from_gate_id - _pi_idx_accum, from_gate_id, to_gate_id - _gate_idx_accum, to_gate_id);
      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
            _POs[to_gate_id - _po_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("Gate_%lu (%lu) -> PO_%lu (%lu)\n", from_gate_id - _gate_idx_accum, from_gate_id, to_gate_id - _po_idx_accum, to_gate_id);
      } else { // gate -> gate
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE> &to_gate =
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

void CPUPartitioner::_levelization(std::vector<std::vector<size_t>> &adj) {
  // printf("start _levelization\n");
#ifdef CPU_PAR_OMP_MODE
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

void CPUPartitioner::_partition(std::vector<std::vector<size_t>> &adj) {
  // printf("start _partition\n");
  
  bool stop = false; 
  size_t startLevel = _max_level; 

  while(!stop) {
    std::vector<std::vector<size_t>> gateBelongsToWhichCones; 
    std::vector<size_t> coneIdxToSinkGateIdx;
    std::vector<std::vector<std::vector<size_t>>> duplicationTable; 
    std::vector<std::vector<std::vector<size_t>>> outputCones;

    int stopLevel = -1;
    _find_all_cones(startLevel, adj, 
                    gateBelongsToWhichCones, 
                    coneIdxToSinkGateIdx, 
                    stopLevel);

    // Update startLevel & conti 
    stop = (stopLevel == -1) ? (true) : (false);
    startLevel = stopLevel;
  }
}


bool CPUPartitioner::_gateHasCone(std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                                  const size_t gateIdx) {
  return (gateBelongsToWhichCones[gateIdx].size() > 0);
}

// gate has postGates in this time during the construction of its _cones 
bool CPUPartitioner::_gateHasPostGates(std::vector<std::vector<size_t>> &adj, 
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

void CPUPartitioner::_getConeIdices(const size_t conesSz, const size_t gateIdx, 
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

void CPUPartitioner::_updateConIdxTable(bool newCone, const size_t gateIdx, 
                      std::vector<size_t> &coneIdxToSinkGateIdx) {
  if (newCone) {
    coneIdxToSinkGateIdx.push_back(gateIdx);
  }
}

void CPUPartitioner::_update_cones(bool newCone, const size_t gateIdx, 
                                    const size_t startLevel, 
                                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones, 
                                    size_t &coneWidth) {
  // If no cone, construct a new cone 
  if (newCone) {
    _cones.push_back(std::vector<std::vector<size_t>>());
  }

  const size_t coneLevel = startLevel - _level_of_gates[gateIdx]; 
  // Push the gate to all of the _cones it belongs to 
  for (size_t cIdx = 0; cIdx < gateBelongsToWhichCones[gateIdx].size(); cIdx++) {
    size_t coneIdx = gateBelongsToWhichCones[gateIdx][cIdx];
    while(coneLevel >= _cones[coneIdx].size()) {
      _cones[coneIdx].push_back(std::vector<size_t>());
    }
    _cones[coneIdx][coneLevel].push_back(gateIdx);
    coneWidth = (_cones[coneIdx][coneLevel].size() > coneWidth) 
                ? (_cones[coneIdx][coneLevel].size()) 
                : (coneWidth);
  }
}

/* Doing
For finding all _cones: needing to find all _cones (two kinds)
1. those _cones start from the start merged level 
2. those _cones are not starting from the start merged level 
* Note: this version without considering the "_merge_bound"
*/
void CPUPartitioner::_find_all_cones(const size_t startLevel, 
                      std::vector<std::vector<size_t>> &adj, 
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<size_t> &coneIdxToSinkGateIdx,
                      int &stopLevel) {
  
  // Backward, traverse all gates and find its _cones 
  // A gate can belong to multiple _cones 

  gateBelongsToWhichCones.resize(_sum_pi_gates_pos);

  for (int level = startLevel; level > stopLevel; level--) {
    size_t coneWidth = 0;
    
    for (size_t g = 0; g < _gateIdx_in_each_level[level].size(); g++) {
      size_t gateIdx = _gateIdx_in_each_level[level][g];
      bool hasCone = _gateHasCone(gateBelongsToWhichCones, gateIdx);
      bool hasPostGate = _gateHasPostGates(adj, gateIdx, startLevel);
      bool newCone = (!hasCone && !hasPostGate);

      // It can have multiple cone Idices 
      _getConeIdices(_cones.size(), gateIdx, startLevel, 
                    newCone, adj, gateBelongsToWhichCones);
      _updateConIdxTable(newCone, gateIdx, coneIdxToSinkGateIdx);

      // Update cone
      _update_cones(newCone, gateIdx, startLevel, 
                    gateBelongsToWhichCones, coneWidth); 
    }

  }
  
#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  printf("startLevel = %lu (include), stopLevel = %d (exclude)\n", 
        startLevel, stopLevel);
  printf("Check Cones: _cones.size() = %lu\n", _cones.size());
  for (size_t c = 0; c < _cones.size(); c++) {
    printf("Cone_%lu:\n", c);
    for (size_t level = 0; level < _cones[c].size(); level++) {
      printf("\tlevel_%lu: ", level);
      for (size_t g = 0; g < _cones[c][level].size(); g++) {
        printf("%lu, ", _cones[c][level][g]);
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

void CPUPartitioner::_construct_duplication_table(std::vector<size_t> &coneIdxToSinkGateIdx,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable, 
                      const size_t startLevel, const int stopLevel, 
                      const size_t numLevels) {
  size_t numCones = coneIdxToSinkGateIdx.size();
  printf("Get inside _construct_duplication_table, numCones = %lu, numLevels = %lu\n", numCones, numLevels);
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


void CPUPartitioner::_RepCut(std::chrono::duration<double> &duration_0, std::chrono::duration<double> &duration_1) {
  auto start = std::chrono::steady_clock::now();
  auto end0 = std::chrono::steady_clock::now();
  auto end1 = std::chrono::steady_clock::now();


  _find_clusters();

  if (_can_run_kahypar) {
    _construct_hyperGraph();
end0 = std::chrono::steady_clock::now();
duration_0 = end0 - start;    
    std::vector<size_t> vertex_to_partition;
    _run_KaHyPar(vertex_to_partition);
    _get_KHP_partitioning_res(vertex_to_partition);
end1 = std::chrono::steady_clock::now();
duration_1 = end1 - start;        
  } else {
    _RepCut_output_cones = _cones;
  }
}

/* For those gates who belong to the same cones belong to the same cluster */
void CPUPartitioner::_find_clusters() {
  _gate_to_cones.resize(_sum_pi_gates_pos);

  // Get _gate_to_cones
  for (size_t c = 0; c < _cones.size(); c++) {
    for (size_t level = 0; level < _cones[c].size(); level++) {
      for (size_t g = 0; g < _cones[c][level].size(); g++) {
        size_t gateIdx = _cones[c][level][g];
        _gate_to_cones[gateIdx].push_back(c);
      } 
    }
  }  

  std::vector<bool> get_cluster;  // tmp 
  get_cluster.resize(_sum_pi_gates_pos, false);
  _gate_to_cluster.resize(_sum_pi_gates_pos);

  // Get _gate_to_cluster
  for (size_t gate0_Idx = 0; gate0_Idx < _sum_pi_gates_pos; gate0_Idx++) {
    if (!get_cluster[gate0_Idx]) {
      std::vector<size_t> tmp;
      tmp.push_back(gate0_Idx);
      _clusters.push_back(tmp);
      _gate_to_cluster[gate0_Idx] = _clusters.size()-1;
      get_cluster[gate0_Idx] = true;
    }

    size_t cluster_0_idx = _gate_to_cluster[gate0_Idx];

    for (size_t gate1_Idx = 0; gate1_Idx < _sum_pi_gates_pos; gate1_Idx++) {
      if (!get_cluster[gate1_Idx] && (_gate_to_cones[gate0_Idx] == _gate_to_cones[gate1_Idx])) {
        _clusters[cluster_0_idx].push_back(gate1_Idx);
        _gate_to_cluster[gate1_Idx] = cluster_0_idx;
        get_cluster[gate1_Idx] = true;        
      }
    }
  }

  // Get _is_sink_clusters, _KaHyPar_num_hyperedges, and _KaHyPar_num_vertices
  _is_sink_clusters.resize(_clusters.size(), false);

  for (size_t c = 0; c < _cones.size(); c++) {
    size_t lvl = 0;
    size_t sz = _cones[c][lvl].size();

    while (sz == 0) {
      lvl++;
      sz = _cones[c][lvl].size();
    }

    size_t sinkIdx = _cones[c][lvl][0];
    if (!_is_sink_clusters[_gate_to_cluster[sinkIdx]]) {
      _is_sink_clusters[_gate_to_cluster[sinkIdx]] = true;
      _KaHyPar_num_vertices++;
    }
  }

  
  if (_clusters.size() != 1) {
    _KaHyPar_num_hyperedges = _clusters.size() - _KaHyPar_num_vertices;

    // Get _to_hypergraph_vertex_idx, _is_sink_clusters_idx, _non_sink_clusters_idx
    for (size_t i = 0; i < _clusters.size(); i++) {
      if (_is_sink_clusters[i]) {
        _to_hypergraph_vertex_idx.push_back(_is_sink_clusters_idx.size());
        _is_sink_clusters_idx.push_back(i); // vertex
      } else {
        _to_hypergraph_vertex_idx.push_back(_non_sink_clusters_idx.size());
        _non_sink_clusters_idx.push_back(i); // hyperedge
      }
    }
  }

  if (_clusters.size() == 1 || _KaHyPar_num_vertices == 0 || _KaHyPar_num_hyperedges == 0) {
    _can_run_kahypar = false;
    // // DEBUG 
    // int a0 = (_clusters.size() == 1) ? (1) : (0);
    // int a1 = (_KaHyPar_num_vertices == 0) ? (1) : (0);
    // int a2 = (_KaHyPar_num_hyperedges == 0) ? (1) : (0);
    // printf("_can_run_kahypar is false since %d %d %d\n", a0, a1, a2);
    // // DEBUG 
  }

#ifdef PART_DEBUG_REPCUT_PRINT_FIND_CLUSTERS
  _show_cones_size();
  _print_find_clusters();
#endif
}


void CPUPartitioner::_construct_hyperGraph() {
  _construct_hyperedge_vertices();

  assert(_vectices_weight.size() == _KaHyPar_num_vertices && 
        "_vectices_weight.size() == _KaHyPar_num_vertices ERROR");
  assert(_hyperedges_weight.size() == _KaHyPar_num_hyperedges && 
        "_hyperedges_weight.size() == _KaHyPar_num_hyperedges ERROR");
  assert(_hyperedges_eptr.size() == (_KaHyPar_num_hyperedges+1) && "_hyperedges_eptr ERROR");
}

void CPUPartitioner::_construct_hyperedge_vertices() {
  /* To know for each hyperedge, it connects to which vertices */

  _vertexIdx_to_sinkGateIdx.resize(_is_sink_clusters_idx.size()); 

  // Get hyperedge_include_which_vertices
  std::vector<std::set<size_t>> hyperedge_include_which_vertices;
  hyperedge_include_which_vertices.resize(_non_sink_clusters_idx.size()); 
  _hyperedges_weight.resize(_non_sink_clusters_idx.size());

  for (size_t c = 0; c < _cones.size(); c++) {
    size_t level = 0;
    size_t sz = _cones[c][level].size();

    while (sz == 0) {
      level++;
      sz = _cones[c][level].size();
    }

    size_t sinkIdx = _cones[c][level][0];
    size_t conSymbolCluster = _gate_to_cluster[sinkIdx];
    size_t vertexIdx = _to_hypergraph_vertex_idx[conSymbolCluster];
    _vertexIdx_to_sinkGateIdx[vertexIdx] = sinkIdx;

    for (size_t lvl = 0; lvl < _cones[c].size(); lvl++) {
      for (size_t g = 0; g < _cones[c][lvl].size(); g++) {
        size_t gateIdx = _cones[c][lvl][g];
        size_t clusterIdx = _gate_to_cluster[gateIdx];
        if (!_is_sink_clusters[clusterIdx]) { // is hyperegde
          size_t hyperedgeIdx = _to_hypergraph_vertex_idx[clusterIdx];
          hyperedge_include_which_vertices[hyperedgeIdx].insert(vertexIdx);
          
          // Update _hyperedges_weight
          _hyperedges_weight[hyperedgeIdx] = _clusters[clusterIdx].size();
        } 
      }
    }
  }
  
  /* Get _hyperedges_weight, _hyperedges_eind, _hyperedges_eptr */
  for (size_t hyperedge = 0; hyperedge < hyperedge_include_which_vertices.size(); hyperedge++) {
    // Update _hyperedges_eptr
    _hyperedges_eptr.push_back(_hyperedges_eind.size());
    for (const auto& vertexIdx : hyperedge_include_which_vertices[hyperedge]) {
      // Update _hyperedges_eind
      _hyperedges_eind.push_back(vertexIdx);
    }
  } 
  // Update _hyperedges_eptr
  _hyperedges_eptr.push_back(_hyperedges_eind.size()); // final one 
  

  // Get vertex_include_which_hyperedges
  std::vector<std::set<size_t>> vertex_include_which_hyperedges;
  vertex_include_which_hyperedges.resize(_is_sink_clusters_idx.size());

  for (size_t hyperedgeIdx = 0; hyperedgeIdx < hyperedge_include_which_vertices.size(); hyperedgeIdx++) {
    for (const auto& vertexIdx : hyperedge_include_which_vertices[hyperedgeIdx]) {
      vertex_include_which_hyperedges[vertexIdx].insert(hyperedgeIdx);
    }
  }  

  /* Get _vectices_weight */
  std::vector<double> tmp_vectices_weight;
  _compute_vertex_weight(tmp_vectices_weight, 
                        vertex_include_which_hyperedges);

  /* Normalize double to size_t*/
  _multiple_100(tmp_vectices_weight);
  
#ifdef PART_DEBUG_REPCUT_PRINT_FIND_CLUSTERS
  print_hyperedge_vertices_details(hyperedge_include_which_vertices, 
                                  vertex_include_which_hyperedges);
#endif 
}

void CPUPartitioner::_compute_vertex_weight(std::vector<double> &tmp_vectices_weight, 
                      std::vector<std::set<size_t>> &vertex_include_which_hyperedges) {

  tmp_vectices_weight.resize(_is_sink_clusters_idx.size());
  for (size_t cl = 0; cl < _is_sink_clusters_idx.size(); cl++) {
    size_t clusterIdx = _is_sink_clusters_idx[cl];
    size_t vertexIdx = _to_hypergraph_vertex_idx[clusterIdx];
    double sumOfYita = 0.0;

    for (const auto& hyperedgeIdx : vertex_include_which_hyperedges[vertexIdx]) {
      sumOfYita += _hyperedges_weight[hyperedgeIdx];
      // printf("sumOfYita = %.3lf\n", sumOfYita);
    }

    if (vertex_include_which_hyperedges[vertexIdx].size() != 0) {
      sumOfYita /= double(vertex_include_which_hyperedges[vertexIdx].size());
    }

    sumOfYita += _clusters[clusterIdx].size();
    tmp_vectices_weight[vertexIdx] = sumOfYita;
  }
}

void CPUPartitioner::_multiple_100(std::vector<double> &tmp_vectices_weight) {
  for (size_t i = 0; i < tmp_vectices_weight.size(); i++) {
    _vectices_weight.push_back(static_cast<size_t>(std::round(tmp_vectices_weight[i] * 100)));
  }

  for (size_t i = 0; i < _hyperedges_weight.size(); i++) {
    _hyperedges_weight[i] *= 100;
  }
}

void CPUPartitioner::_run_KaHyPar(std::vector<size_t> &vertex_to_partition) {
  // printf("_run_KaHyPar\n");

  kahypar_context_t* context = kahypar_context_new();
  kahypar_configure_context_from_file(context, "/home/ychung79/kahypar/config/repcut.ini");
  
  kahypar_set_seed(context, -1);

  /* Settings of #V and #he of H */
  const kahypar_hypernode_id_t num_vertices = _KaHyPar_num_vertices;
  const kahypar_hyperedge_id_t num_hyperedges = _KaHyPar_num_hyperedges;
  
  assert(_vectices_weight.size() == _KaHyPar_num_vertices 
        && "_vectices_weight.size() == _KaHyPar_num_vertices ERROR"); 
  assert(_hyperedges_weight.size() == _KaHyPar_num_hyperedges 
        && "_hyperedges_weight.size() == _KaHyPar_num_hyperedges ERROR"); 

  /* Inputs of vertex weights - bibi */
  std::unique_ptr<kahypar_hypernode_weight_t[]> vertex_weights = 
              std::make_unique<kahypar_hypernode_weight_t[]>(_vectices_weight.size());
  for (size_t i = 0; i < (_vectices_weight.size()); i++) {
    vertex_weights[i] = _vectices_weight[i];
  }

  /* Input of hyperedges */
  std::unique_ptr<kahypar_hyperedge_weight_t[]> hyperedge_weights = 
        std::make_unique<kahypar_hyperedge_weight_t[]>(num_hyperedges);
  for (size_t i = 0; i < (num_hyperedges); i++) {
    hyperedge_weights[i] = _hyperedges_weight[i];
  }

  // eptr
  std::unique_ptr<size_t[]> hyperedge_indices = 
        std::make_unique<size_t[]>(_hyperedges_eptr.size());
  for (size_t i = 0; i < _hyperedges_eptr.size(); i++) {
    hyperedge_indices[i] = _hyperedges_eptr[i];
  }
  
  // eind
  std::unique_ptr<kahypar_hyperedge_id_t[]> hyperedges = 
          std::make_unique<kahypar_hyperedge_id_t[]>(_hyperedges_eind.size());
  for (size_t i = 0; i < _hyperedges_eind.size(); i++) {
    hyperedges[i] = _hyperedges_eind[i];
  }

  // write back 
  kahypar_hyperedge_weight_t objective = 0;

  std::vector<kahypar_partition_id_t> partition(num_vertices, -1);

  kahypar_partition(num_vertices, num_hyperedges,
       	            _imbalance, _k,
               	    vertex_weights.get(), hyperedge_weights.get(),
               	    hyperedge_indices.get(), hyperedges.get(),
       	            &objective, context, partition.data());

  for(size_t i = 0; i < num_vertices; i++) {
    vertex_to_partition.push_back(size_t(partition[i]));
  }

  kahypar_context_free(context);

#ifdef PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS
  for(size_t i = 0; i < vertex_to_partition.size(); i++) {
    std::cout << i << ":" << vertex_to_partition[i] << std::endl;
  }  
#endif
}

void CPUPartitioner::_get_KHP_partitioning_res(std::vector<size_t> &vertex_to_partition) {
  _RepCut_output_cones_set.resize(_k);
  _RepCut_output_cones.resize(_k);
  for (size_t i = 0; i < _k_sizet; i++) {
    _RepCut_output_cones_set[i].resize(_total_num_levels);
    _RepCut_output_cones[i].resize(_total_num_levels);
  }

  for (size_t i = 0; i < _is_sink_clusters_idx.size(); i++) {
    size_t clusterIdx = _is_sink_clusters_idx[i];
    size_t vertexIdx = _to_hypergraph_vertex_idx[clusterIdx];
    size_t partitionIdx = vertex_to_partition[vertexIdx];
    size_t sinkIdx = _vertexIdx_to_sinkGateIdx[vertexIdx];

    // traverse the cone of the sinkGate
    for (size_t c = 0; c < _gate_to_cones[sinkIdx].size(); c++) {
      size_t coneIdx = _gate_to_cones[sinkIdx][c];
      for (size_t lvl = 0; lvl < _cones[coneIdx].size(); lvl++) {
        for (size_t g = 0; g < _cones[coneIdx][lvl].size(); g++) {
          size_t gateIdx = _cones[coneIdx][lvl][g];
          _RepCut_output_cones_set[partitionIdx][lvl].insert(gateIdx);
        }
      }
    }
  }

  for (size_t i = 0; i < _RepCut_output_cones_set.size(); ++i) {
    for (size_t lvl = 0; lvl < _RepCut_output_cones_set[i].size(); ++lvl) {
      for (const auto& element : _RepCut_output_cones_set[i][lvl]) {
        _RepCut_output_cones[i][lvl].push_back(element);
      }
    }
  }

#ifdef PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS
  _print_RepCut_output_cones();
#endif
}

// --------------------------------------------------------------------------------------

// Run the simulation 
void CPUPartitioner::run(Mode mode, const size_t num_threads) {
  switch (mode) {
    case Mode::SEQUENTIAL: {
      std::cout << "run SEQUENTIAL\n";
      assert(num_threads == 1 && "RUN SEQUENTIAL num_threads num_threads > 1");
      _run_fsim(num_threads);
      break;
    }
    case Mode::PARALLEL_OMP: {
      std::cout << "run PARALLEL_OMP\n";
      _run_fsim(num_threads);
      break;
    }
    case Mode::CONES_SEQUENTIAL: {
      std::cout << "run CONES_SEQUENTIAL\n";
      assert(num_threads == 1 && "RUN CONES_SEQUENTIAL num_threads num_threads > 1");
      _run_cones_gates(num_threads);
      break;
    }
    case Mode::CONES_PARALLEL_OMP: {
      std::cout << "run CONES_PARALLEL_OMP\n";
      _run_cones_gates(num_threads);
      break;
    }
    case Mode::REPCUT_SEQUENTIAL: {
      std::cout << "run REPCUT_SEQUENTIAL\n";
      if (_can_run_kahypar) {
        printf("_can_run_kahypar TRUE\n");
      } else {
        printf("_can_run_kahypar FALSE\n");
      }
      assert(num_threads == 1 && "RUN REPCUT_SEQUENTIAL num_threads > 1");
      auto start = std::chrono::steady_clock::now();      
      _run_RepCut_cones_gates(num_threads);
      auto end = std::chrono::steady_clock::now();      
      std::chrono::duration<double> duration_0 = end - start;

      start = std::chrono::steady_clock::now();      
      const size_t new_num_threads = 8;
      _run_RepCut_cones_gates(new_num_threads);
      end = std::chrono::steady_clock::now();      
      std::chrono::duration<double> duration_1 = end - start;
      
      std::cout << "BIBI run_CUDA SEQ: " << _round_to((duration_0.count())*1000, 0.001) << " ms\n";
      std::cout << "BIBI run_CUDA OMP: " << _round_to((duration_1.count())*1000, 0.001) << " ms\n";
      break;
    }
    case Mode::REPCUT_PARALLEL_OMP: {
      std::cout << "run REPCUT_PARALLEL_OMP\n";
      if (_can_run_kahypar) {
        printf("_can_run_kahypar TRUE\n");
      } else {
        printf("_can_run_kahypar FALSE\n");
      }
      _run_RepCut_cones_gates(num_threads);
      break;      
    }
    default:
      printf("No SUCH CASES\n");
      return; 
  }
}

void CPUPartitioner::_run_fsim(const size_t num_threads) {
#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_fsim();" << '\n';
#endif

  // printf("_num_rounds = %lu\n", _num_rounds);

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

#ifdef PART_DEBUG_PRINT_SIMULATION
    std::cout << "Run Good case with threads = " << num_threads << std::endl;
#endif
    _run_good_case(_patterns[rd], num_testcases_this_round, num_threads);
    
    // // bad simulation (fault simulation)
    // for (size_t j = 0; j < _faults.size(); j++) {
    //   _run_bad_case(_faults[j], _patterns[rd], 
    //                 num_testcases_this_round, num_threads);

    //   size_t found_fault = 0;
    //   for (size_t i = 0; i < _num_POs; i++) {
    //     if (_g_po_results[i] != _b_po_results[i]) {
    //       found_fault = 1;
    //       break;
    //     }
    //   }
    //   // Record whether fault can be found
    //   _found_fault_to_pattern[2 * j] = found_fault;
    //   // Record which pattern found the fault
    //   _found_fault_to_pattern[2 * j + 1] = rd;
    // }
#ifdef PART_DEBUG_PRINT_SIMULATION
  if (rd == 0) {
    std::cout << "GOOD resutls ans:" << std::endl;
    print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
  }
#endif
  }
}

void CPUPartitioner::_run_good_case(const Pattern pattern, const size_t bits, const size_t num_threads) {
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
      ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                          static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

      // Execute the gate operation
      _run_gate(tmp, pattern, gateIdx, 0, 0);
    }
  }

  // Duplicated the answer of good_case into a new memory
  _write_to_array(_g_pi_results, _g_gate_results, _g_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls ans:\n";
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void CPUPartitioner::_run_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, 
                                    const Pattern pattern,
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
      ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                          static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

      // Execute the gate operation
      _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
    }
  }

  // Duplicated the answer of good_case into a new memory
  _write_to_array(_b_pi_results, _b_gate_results, _b_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "BAD resutls ans:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void CPUPartitioner::_run_cones_gates(const size_t num_threads) {
#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_cones_gates();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
#ifdef PART_DEBUG_PRINT_SIMULATION
    std::cout << "Run Good case with threads = " << num_threads << std::endl;
#endif
    _run_cones_good_case(_patterns[rd], num_testcases_this_round, num_threads);
    
    // // bad simulation (fault simulation)
    // for (size_t j = 0; j < _faults.size(); j++) {
    //   _run_cones_bad_case(_faults[j], _patterns[rd], 
    //                       num_testcases_this_round, num_threads);

    //   size_t found_fault = 0;
    //   for (size_t i = 0; i < _num_POs; i++) {
    //     if (_g_po_results[i] != _b_po_results[i]) {
    //       found_fault = 1;
    //       break;
    //     }
    //   }
    //   // Record whether fault can be found
    //   _found_fault_to_pattern[2 * j] = found_fault;
    //   // Record which pattern found the fault
    //   _found_fault_to_pattern[2 * j + 1] = rd;
    // }
#ifdef PART_DEBUG_PRINT_SIMULATION
  if (rd == 0) {
    std::cout << "GOOD resutls ans:" << std::endl;
    print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
  }
#endif
  }
}

void CPUPartitioner::_run_cones_good_case(const Pattern pattern, 
                                          const size_t bits, 
                                          const size_t num_threads) {

  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  // Parallelize across cones (c)
  #pragma omp parallel for
  for (size_t c = 0; c < _cones.size(); c++) {
    for (int lvl = _cones[c].size()-1; lvl > -1; lvl--) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for
      for (size_t g = 0; g < _cones[c][lvl].size(); g++) {
        size_t gateIdx = _cones[c][lvl][g];
        // printf("c = %ld, lvl = %d, g = %ld, gateIdx = %ld\n", c, lvl, g, gateIdx);
        
        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                            static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, 0, 0);
      }
    }
  }
  
  // Duplicated the answer of good_case into a new memory
  _write_to_array(_g_pi_results, _g_gate_results, _g_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: GOOD resutls ans:\n";
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}


void CPUPartitioner::_run_cones_bad_case(const Fault<FAULT_INDEX_TYPE> &fault, 
                                        const Pattern pattern,
                                        const size_t bits, 
                                        const size_t num_threads) {
  // printf("Getting inside _run_cones_bad_case\n");
  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;

  // Parallelize across cones (c)
  #pragma omp parallel for
  for (size_t c = 0; c < _cones.size(); c++) {
    for (int lvl = _cones[c].size()-1; lvl > -1; lvl--) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for
      for (size_t g = 0; g < _cones[c][lvl].size(); g++) {
        int thread_id = omp_get_thread_num();
        size_t gateIdx = _cones[c][lvl][g];

        // Calculate SA_fault using a ternary operator (avoiding if-else)
        size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx + _pi_idx_accum) == wrong_gate) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx + _gate_idx_accum) == wrong_gate) :
                                                              ((gateIdx + _po_idx_accum) == wrong_gate);

        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                            static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
      }
    }
  }

  // Duplicated the answer of good_case into a new memory
  _write_to_array(_b_pi_results, _b_gate_results, _b_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: BAD resutls ans:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void CPUPartitioner::_run_RepCut_cones_gates(const size_t num_threads) {
#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_RepCut_cones_gates();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
#ifdef PART_DEBUG_PRINT_SIMULATION
    std::cout << "Run Good case with threads = " << num_threads << std::endl;
#endif
    _run_RepCut_cones_good_case(_patterns[rd], num_testcases_this_round, num_threads);
    
    // // bad simulation (fault simulation)
    // for (size_t j = 0; j < _faults.size(); j++) {
    //   _run_RepCut_cones_bad_case(_faults[j], _patterns[rd], 
    //                       num_testcases_this_round, num_threads);

    //   size_t found_fault = 0;
    //   for (size_t i = 0; i < _num_POs; i++) {
    //     if (_g_po_results[i] != _b_po_results[i]) {
    //       found_fault = 1;
    //       break;
    //     }
    //   }
    //   // Record whether fault can be found
    //   _found_fault_to_pattern[2 * j] = found_fault;
    //   // Record which pattern found the fault
    //   _found_fault_to_pattern[2 * j + 1] = rd;
    // }
#ifdef PART_DEBUG_PRINT_SIMULATION
  if (rd == 0) {
    std::cout << "GOOD resutls ans:" << std::endl;
    print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
  }
#endif
  }
}

void CPUPartitioner::_run_RepCut_cones_good_case(const Pattern pattern, 
                                          const size_t bits, 
                                          const size_t num_threads) {

  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  // Parallelize across cones (c)
  #pragma omp parallel for
  for (size_t c = 0; c < _RepCut_output_cones.size(); c++) {
    for (int lvl = _RepCut_output_cones[c].size()-1; lvl > -1; lvl--) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for
      for (const auto& gateIdx : _RepCut_output_cones[c][lvl]) {
        
        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                            static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, 0, 0);
      }
    }
  }
  
  // Duplicated the answer of good_case into a new memory
  _write_to_array(_g_pi_results, _g_gate_results, _g_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: GOOD resutls ans:\n";
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}


void CPUPartitioner::_run_RepCut_cones_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, 
                                                const Pattern pattern,
                                                const size_t bits, const size_t num_threads) {
  // printf("Getting inside _run_RepCut_cones_bad_case\n");
  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;

  // Parallelize across cones (c)
  #pragma omp parallel for
  for (size_t c = 0; c < _RepCut_output_cones.size(); c++) {
    for (int lvl = _RepCut_output_cones[c].size()-1; lvl > -1; lvl--) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for
      for (const auto& gateIdx : _RepCut_output_cones[c][lvl]) {

        // Calculate SA_fault using a ternary operator (avoiding if-else)
        size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx + _pi_idx_accum) == wrong_gate) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx + _gate_idx_accum) == wrong_gate) :
                                                              ((gateIdx + _po_idx_accum) == wrong_gate);

        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = (gateIdx < _num_PIs) ? static_cast<ElementBase<>&>(_PIs[gateIdx - _pi_idx_accum]) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? static_cast<ElementBase<>&>(_gates[gateIdx - _gate_idx_accum]) :
                            static_cast<ElementBase<>&>(_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
      }
    }
  }

  // Duplicated the answer of good_case into a new memory
  _write_to_array(_b_pi_results, _b_gate_results, _b_po_results, bits);

#ifdef PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: BAD resutls ans:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void CPUPartitioner::_write_to_array(std::vector<size_t> &pi_output, 
                                    std::vector<size_t> &gate_output,
                                    std::vector<size_t> &po_output,
                                    const size_t bits) {
  // Duplicated the answer of good_case into a new memory
  for (size_t i = 0; i < _num_PIs; i++) {
    pi_output[i] = _PIs[i]._output_value;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    gate_output[i] = _gates[i]._output_value;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    po_output[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(pi_output, bits, _num_PIs);
    _shift_to_correct_answer(gate_output, bits, _num_inner_gates);
    _shift_to_correct_answer(po_output, bits, _num_POs);
  }  
}



// Simulation for gates
void CPUPartitioner::_apply_INV(ElementBase<> &gate) {
  ElementBase<> *pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

void CPUPartitioner::_apply_AND(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CPUPartitioner::_apply_OR(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;
  
  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CPUPartitioner::_apply_XOR(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CPUPartitioner::_apply_NAND(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CPUPartitioner::_apply_NOR(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CPUPartitioner::_apply_XNOR(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CPUPartitioner::_apply_MUX(ElementBase<> &gate) {
  size_t a = gate._inputs[0]->_output_value;
  size_t b = gate._inputs[1]->_output_value;
  size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | (a & (!s)));

  gate._output_value = (ret);
}

void CPUPartitioner::_apply_CLKBUF(ElementBase<> &gate) {
  ElementBase<> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}

void CPUPartitioner::_apply_PI(ElementBase<> &gate, const Pattern pattern,
                                const size_t pi) {
  gate._output_value = pattern._value[pi];
}

void CPUPartitioner::_apply_PO(ElementBase<> &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  ElementBase<> *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

void CPUPartitioner::_run_gate(ElementBase<> &gate, const Pattern pattern,
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

void CPUPartitioner::_shift_to_correct_answer(std::vector<size_t> &results,
                                               const size_t bits,
                                               const size_t num_shift_gates) {
  for (size_t i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
}




// -----------------------------------------------------------------------

// Output file visualization graph
void CPUPartitioner::visualization_graph(const std::string &outputFile) {
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

void CPUPartitioner::print_ppg(const ElementBase<> &gate) const {
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

void CPUPartitioner::print_patterns(const std::vector<Pattern> &patterns,
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

void CPUPartitioner::print_bits_stack(const size_t size,
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

void CPUPartitioner::print_simulation_results(
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
std::string CPUPartitioner::gateTypeToString(GateType type) const {
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

void CPUPartitioner::_show_cones_size() {
  // printf("_cones.size() = %lu\n", _cones.size());
  size_t maxWidth = 0, conWithMaxWidth = 0;
  for (size_t c = 0; c < _cones.size(); c++) {
    for (size_t lvl = 0; lvl < _cones[c].size(); lvl++) {
      bool update = _cones[c][lvl].size() > maxWidth;
      maxWidth = (update) 
                ? (_cones[c][lvl].size()) : (maxWidth);
      conWithMaxWidth = (update) ? (c) : (conWithMaxWidth);
    }
  }
  printf("cone_%lu, maxWidth = %lu\n", conWithMaxWidth, maxWidth);

  for (size_t c = 0; c < _cones.size(); c++) {
    printf("Cones_%lu:\n", c);
    for (size_t level = 0; level < _cones[c].size(); level++) {
      printf("\tlevel_%lu: ", level);
      for (size_t g = 0; g < _cones[c][level].size(); g++) {
        printf("%lu, ", _cones[c][level][g]);
      } 
      printf("\n");
    }
    printf("\n");
  }  
  printf("\n");
}

void CPUPartitioner::_print_find_clusters() const {

  printf("Check find_clusters\n");
  printf("_gate_to_cones:\n");
  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    printf("Gate[%lu] belongs to cones: ", g);
    for (size_t c = 0; c < _gate_to_cones[g].size(); c++) {
      printf("%lu, ", _gate_to_cones[g][c]);
    }
    printf("\n");
  }
  printf("\n");

  printf("_gate_to_cluster:\n");
  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    printf("Gate[%lu] belongs to cluster: %lu\n", g, _gate_to_cluster[g]);
  }
  printf("\n");

  printf("_clusters (sz = %lu)\n", _clusters.size());
  for (size_t cl = 0; cl < _clusters.size(); cl++) {
      printf("Cluster %lu: ", cl);
      for (size_t g = 0; g < _clusters[cl].size(); g++) {
      printf("%lu, ", _clusters[cl][g]);
    }
    printf("\n");
  }
  printf("\n");


  if (_can_run_kahypar) {
    printf("_is_sink_clusters:\n");
    for (size_t i = 0; i < _is_sink_clusters.size(); i ++) {
      if (_is_sink_clusters[i]) {
        printf("cluster %lu: true\n", i);
      } else {
        printf("cluster %lu: false\n", i);
      }
    }
    printf("\n");
  
    
    printf("_KaHyPar_num_hyperedges = %lu, _KaHyPar_num_vertices = %lu\n", 
            _KaHyPar_num_hyperedges, _KaHyPar_num_vertices);
  
    printf("_is_sink_clusters_idx: ");
    for (size_t i = 0; i < _is_sink_clusters_idx.size(); i++) {
      printf("%lu, ", _is_sink_clusters_idx[i]);
    }        
    printf("\n");  
  
    printf("_non_sink_clusters_idx: ");
    for (size_t i = 0; i < _non_sink_clusters_idx.size(); i++) {
      printf("%lu, ", _non_sink_clusters_idx[i]);
    }        
    printf("\n");
  
    for (size_t i = 0; i < _clusters.size(); i++) {
      if (_is_sink_clusters[i]) {
        printf("cluster %lu, is vertex on HG, with vertex index = %lu\n", i,  _to_hypergraph_vertex_idx[i]);
      } else {
        printf("cluster %lu, is hyperegde on HG, with hyperegde index = %lu\n", i,  _to_hypergraph_vertex_idx[i]);
      }
    }
    printf("\n");
  }
}


void CPUPartitioner::print_hyperedge_vertices_details(
                    const std::vector<std::set<size_t>> &hyperedge_include_which_vertices,
                    const std::vector<std::set<size_t>> &vertex_include_which_hyperedges) const {

  printf("hyperedge_include_which_vertices:\n");
  for (size_t hyperedge = 0; hyperedge < hyperedge_include_which_vertices.size(); hyperedge++) {
    printf("hyperedge %lu includes vertex indices: ", hyperedge);
    for (const auto& vertexIdx : hyperedge_include_which_vertices[hyperedge]) {
      printf("%lu, ", vertexIdx);
    }
    printf("\n");
  }
  printf("\n");


  printf("vertex_include_which_hyperedges:\n");
  for (size_t vertexIdx = 0; vertexIdx < vertex_include_which_hyperedges.size(); vertexIdx++) {
    printf("vertexIdx %lu includes vertex indices: ", vertexIdx);
    for (const auto& hyperedge : vertex_include_which_hyperedges[vertexIdx]) {
      printf("%lu, ", hyperedge);
    }
    printf("\n");
  }
  printf("\n");

  printf("_hyperedges_weight: \n");
  for (size_t i = 0; i < _hyperedges_weight.size(); i++) {
    printf("hyperedge %lu, weight = %lu\n", i, _hyperedges_weight[i]);
  }
  printf("\n");

  printf("_hyperedges_eptr: = [ ");
  for (size_t i = 0; i < _hyperedges_eptr.size(); i++) {
    printf("%lu, ", _hyperedges_eptr[i]);
  } printf("]\n\n");

  printf("_hyperedges_eind = [ ");
  for (size_t i = 0; i < _hyperedges_eind.size(); i++) {
    printf("%lu, ", _hyperedges_eind[i]);
  } printf("]\n\n");

  printf("_vectices_weight\n");
  for (size_t i = 0; i < _vectices_weight.size(); i++) {
    printf("vertex %lu, with weight = %lu\n", i, _vectices_weight[i]);
  } 
  printf("\n");

}

void CPUPartitioner::_print_RepCut_output_cones() {
  _show_cones_size();
  printf("_RepCut_output_cones:\n\n");
  for (size_t i = 0; i < _RepCut_output_cones.size(); ++i) {
    printf("_RepCut_output_cones %lu\n", i);
    for (size_t lvl = 0; lvl < _RepCut_output_cones[i].size(); ++lvl) {
      printf("\tlvl = %lu, gates: ", lvl);
      for (const auto& element : _RepCut_output_cones[i][lvl]) {
        printf("%lu, ", element);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}


void CPUPartitioner::_print_num_PIs_Gates_POs(Mode mode) {
  std::vector<size_t> ret;
  _traverse_cones_for_print(mode, ret);

  printf("new_num_PIs = %lu, new_num_inner_gates = %lu, new_num_POs = %lu\n", 
          ret[0], ret[1], ret[2]);
}

void CPUPartitioner::_traverse_cones_for_print(Mode mode, std::vector<size_t> &ret) {
  ret.resize(3, 0);
  // const size_t b0 = 0;
  const size_t b1 = _num_PIs;
  const size_t b2 = _num_PIs + _num_POs;
  const size_t b3 = _sum_pi_gates_pos;

  if (mode == Mode::SEQUENTIAL || mode == Mode::PARALLEL_OMP) {
    ret[0] = _num_PIs;
    ret[2] = _num_POs;
    ret[1] = _num_inner_gates;
  } else if (mode == Mode::CONES_SEQUENTIAL || mode == Mode::CONES_PARALLEL_OMP) {
    for (size_t c = 0; c < _cones.size(); c++) {
      for (size_t lvl = 0; lvl < _cones[c].size(); lvl++) {
        for (size_t g = 0; g < _cones[c][lvl].size(); g++) {
          size_t gateIdx = _cones[c][lvl][g];
          if (gateIdx < b1) { // PIs
            ret[0]++;
          } else if (gateIdx >= b1 && gateIdx < b2) { // POs
            ret[2]++;
          } else if (gateIdx >= b2 && gateIdx < b3) {
            ret[1]++;
          } else {
            printf("_traverse_cones_for_print (mode 3, 4): Gate Index: No SUCH CASES\n");
            exit(1);
          }
        }
      }
    }
  } else if (mode == Mode::REPCUT_SEQUENTIAL || mode == Mode::REPCUT_PARALLEL_OMP) {
    for (size_t c = 0; c < _RepCut_output_cones.size(); c++) {
      for (size_t lvl = 0; lvl < _RepCut_output_cones[c].size(); lvl++) {
        for (size_t g = 0; g < _RepCut_output_cones[c][lvl].size(); g++) {
          size_t gateIdx = _RepCut_output_cones[c][lvl][g];
          if (gateIdx < b1) { // PIs
            ret[0]++;
          } else if (gateIdx >= b1 && gateIdx < b2) { // POs
            ret[2]++;
          } else if (gateIdx >= b2 && gateIdx < b3) {
            ret[1]++;
          } else {
            printf("_traverse_cones_for_print (mode 4, 5): Gate Index: No SUCH CASES\n");
            exit(1);
          }
        }
      }
    }    
  } else {
    printf("read: No SUCH CASES\n");
    exit(1);
  }
}




/* 
  This is the example code revised from KaHypar. 
  The hypergraph is the graph from RepCut
*/
void CPUPartitioner::KaHyParExample() {
  kahypar_context_t* context = kahypar_context_new();
  kahypar_configure_context_from_file(context, "/home/ychung79/kahypar/config/repcut.ini");
  
  kahypar_set_seed(context, -1);

  /* Settings of #V and #he of H */
  const kahypar_hypernode_id_t num_vertices = 5;
  const kahypar_hyperedge_id_t num_hyperedges = 5;

  /* Inputs of vertex weights - bibi */
  std::unique_ptr<kahypar_hypernode_weight_t[]> vertex_weights = 
              std::make_unique<kahypar_hypernode_weight_t[]>(num_vertices);
  double tmp0 [num_vertices] = {2.50, 2.67, 4.33, 2.50, 2.50, };
  for (size_t i = 0; i < (num_vertices); i++) {
    vertex_weights[i] = tmp0[i];
  }

  /* Input of hyperedges */
  std::unique_ptr<kahypar_hyperedge_weight_t[]> hyperedge_weights = 
        std::make_unique<kahypar_hyperedge_weight_t[]>(num_hyperedges);
  size_t tmp1 [num_hyperedges] = {2, 1, 2, 1, 2};
  for (size_t i = 0; i < (num_hyperedges); i++) {
    hyperedge_weights[i] = tmp1[i];
  }

  // eptr
  std::unique_ptr<size_t[]> hyperedge_indices = 
        std::make_unique<size_t[]>(num_hyperedges+1);
  size_t tmp2 [num_hyperedges+1] = {0, 2, 5, 7, 10, 12};
  for (size_t i = 0; i < (num_hyperedges+1); i++) {
    hyperedge_indices[i] = tmp2[i];
  }
  
  // eind
  size_t tmp3Sz = 12; 
  std::unique_ptr<kahypar_hyperedge_id_t[]> hyperedges = 
          std::make_unique<kahypar_hyperedge_id_t[]>(tmp3Sz);
  size_t tmp3 [tmp3Sz] = {0, 1, 0, 1, 2, 1, 2, 2, 3, 4, 3, 4, };
  for (size_t i = 0; i < tmp3Sz; i++) {
    hyperedges[i] = tmp3[i];
  }

  // write back 
  kahypar_hyperedge_weight_t objective = 0;

  std::vector<kahypar_partition_id_t> partition(num_vertices, -1);

  kahypar_partition(num_vertices, num_hyperedges,
       	            _imbalance, _k,
               	    vertex_weights.get(), hyperedge_weights.get(),
               	    hyperedge_indices.get(), hyperedges.get(),
       	            &objective, context, partition.data());

  for(int i = 0; i != num_vertices; ++i) {
    std::cout << i << ":" << partition[i] << std::endl;
  }

  kahypar_context_free(context);
}


