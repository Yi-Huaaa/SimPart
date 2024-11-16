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
#include <cstdlib>

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
// #define PART_DEBUG_PRINT_SIMULATION // show the output answers
// #define PART_DEBUG_REPCUT_PRINT_FIND_CLUSTERS
// #define PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS

// #define CPU_PRE_CONSTRUCTION_OF_HYPERGRAPH // run pre-construct hypergraph 

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
void CPUPartitioner::read(Mode mode, 
                          const std::string &ckt_path, 
                          const std::string &flst_path, 
                          const std::string &ptn_path,
                          const std::string &hyg_path,
                          const size_t num_threads,
                          const int hypergraph_construct_mode) {
  // printf("Get inside CPUPartitioner::read first\n");

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path), hyg(hyg_path);

  if (!ckt)
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  if (!flst)
    throw std::runtime_error("cannot open fault file "s + flst_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);

  read(mode, ckt, flst, ptn, hyg, num_threads, hypergraph_construct_mode);
}

void CPUPartitioner::read(Mode mode, 
                          std::istream &ckt, 
                          std::istream &flst, 
                          std::istream &ptn, 
                          std::istream &hyg, 
                          const size_t num_threads,
                          const int hypergraph_construct_mode) {


  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 

  _read_graph(ckt);

  /* Partitioning */                              
  std::chrono::duration<double> duration_KHP;
  std::chrono::duration<double> duration_load_data; // need to be minus 
  auto start = std::chrono::steady_clock::now();

  // levelization
  _levelization();

  // whether partitioning
  switch (mode) {
    case Mode::SEQUENTIAL: 
    case Mode::PARALLEL_OMP: { // 0 1
      break;
    } 
    case Mode::REPCUT_PARALLEL_OMP: { // 2
      _partition();
      _get_hypergraph(hypergraph_construct_mode, hyg, duration_load_data);
      _RepCut(duration_KHP);
      break;  
    }
    case Mode::MT_REPCUT_PARALLEL_OMP: { // 3
      _partition();
      _get_hypergraph(hypergraph_construct_mode, hyg, duration_load_data);
      _RepCut_MT(duration_KHP, num_threads);
      break;  
    }    
    default: {
      printf("read: No SUCH CASES\n");
      return; 
    }
  }

  /* Construct circuit, Read Pattern, Read Fault */
  _read_fault(flst);
  _read_pattern(ptn);

  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> duration_total_RC_HGC = (hypergraph_construct_mode == 1) ? 
                                                        (end - start - duration_load_data) :
                                                        (end - start);
  duration_KHP = (hypergraph_construct_mode == 1) ? 
                 (duration_KHP) : 
                 (duration_total_RC_HGC - duration_load_data);


  std::cout << "duration_load_data (aka hypergraph construction time): " << _round_to((duration_load_data.count())*1000, 0.001) << "\n";
  std::cout << "duration_KHP_Partition: " << _round_to((duration_KHP.count())*1000, 0.001) << "\n";
  std::cout << "duration_RepCut_Partition (Total time): " << _round_to((duration_total_RC_HGC.count())*1000, 0.001) << "\n";

  // // 1107 for exp
  // _preconstruct_hypergraph_print_flag = true;
  // if (_preconstruct_hypergraph_print_flag) {
  //   print_preconstruct_hyperedge_vertices_details();
  // }
  // // 1107 for exp
}


void CPUPartitioner::read_pre_construct_HG(Mode mode, 
                                            const std::string &ckt_path, 
                                            const std::string &flst_path, 
                                            const std::string &ptn_path,
                                            const std::string &hyg_path,
                                            const size_t num_threads) {
  // printf("Get inside CPUPartitioner::read first\n");

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path), hyg(hyg_path);

  if (!ckt)
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  if (!flst)
    throw std::runtime_error("cannot open fault file "s + flst_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);
  if (!ptn)
    throw std::runtime_error("cannot open pattern file "s + ptn_path);

  read_pre_construct_HG(mode, ckt, flst, ptn, hyg, num_threads);
}


void CPUPartitioner::read_pre_construct_HG(Mode mode, 
                                          std::istream &ckt, 
                                          std::istream &flst, 
                                          std::istream &ptn, 
                                          std::istream &hyg, 
                                          const size_t num_threads) {


  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 

  _read_graph(ckt);
  printf("read_pre_construct_HG\n");

  /* Partitioning */                              
  _levelization();
  _partition();
  _preconstruct_hypergraph_print_flag = true;
  std::chrono::duration<double> duration_load_data; 
  _get_hypergraph(0, hyg, duration_load_data);
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

void CPUPartitioner::_read_graph(std::istream &ckt) {
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  _adj.resize(_sum_pi_gates_pos);
  _invAdj.resize(_sum_pi_gates_pos);

  // Construct adjacent list
  for (size_t i = 0; i < _num_wires; i++) {
    size_t gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (size_t j = 0; j < num_post_gates; j++) {
      size_t gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      _adj[gate_0].push_back(gate_tmp);
      _invAdj[gate_tmp].push_back(gate_0);
    }
  }
  
  // Find all _sources, actually == PIs
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    if (_invAdj[i].size() == 0) { // inputs == 0
      _sources.push_back(i);
    }
  }
  assert(_sources.size() == _num_PIs && "_sources.size() != _num_PIs");

  // Read GateType
  _gate_type.resize(_sum_pi_gates_pos);
  for (size_t i = 0; i < _num_PIs; i++) {
    _gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    _gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs + _num_POs); i < (_sum_pi_gates_pos); i++) {
    size_t t; // type
    ckt >> t;
    _gate_type[i] = static_cast<GateType>(t);
  }

#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  printf("_sources.size() = %lu\n", _sources.size());
  for (size_t i = 0; i < _num_PIs; i++) {
    if (_adj[i].size() == 0) {
      printf("_adj[%lu].size() = %lu;;; gates = ", i, _adj[i].size());
      for (size_t j = 0; j < _adj[i].size(); j++) {
        printf("%lu, ", _adj[i][j]);
      }
      printf("\n");
    }
  }
  printf("\n");

  for (size_t i = 0; i < _num_PIs; i++) {
    printf("_invAdj[%lu].size() = %lu;;; gates = ", i, _invAdj[i].size());
    for (size_t j = 0; j < _invAdj[i].size(); j++) {
      printf("%lu, ", _invAdj[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("_gate_type:\n");
  for (size_t i = 0; i < _gate_type.size(); i++) {
    std::cout << "Gate_" << i << ", _gate_type = " << gateTypeToString(_gate_type[i]) << "\n";
  }  
  
#endif 
}


void CPUPartitioner::_construct_graph() {
  _Gates.resize(_sum_pi_gates_pos);

  for (size_t toGate = 0; toGate < _invAdj.size(); toGate++) {
    size_t toGateIdx = toGate;
    _Gates[toGate]._idx = toGateIdx;
    _Gates[toGate]._type = _gate_type[toGate];
    _Gates[toGate]._level = _level_of_gates[toGate];
    _Gates[toGate]._output_value = 0;

    for (size_t fmGate = 0; fmGate < _invAdj[toGate].size(); fmGate++) {
      size_t fmGateIdx = _invAdj[toGate][fmGate];
      ElementBase<> &from_gate =
        _Gates[fmGateIdx];
      ElementBase<> &to_gate =
        _Gates[toGateIdx];
      to_gate._inputs.push_back(&from_gate);
    }
  }

  // printf("_invAdj.size() = %lu, _sum_pi_gates_pos = %lu\n", _invAdj.size(), _sum_pi_gates_pos);

#ifdef PART_DEBUG_PRINT_GRAPH
  std::cout << "\n_Gates = [\n";
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    print_ppg(_Gates[i]);
  }
  std::cout << "]\n";
#endif  
}

void CPUPartitioner::_levelization() {
  // printf("start _levelization\n");
#ifdef CPU_PAR_OMP_MODE
  omp_set_num_threads(num_threads);
#endif 

  _max_level = 0;
  _level_of_gates.resize(_sum_pi_gates_pos, 0);

  // Parallelizing the outer loop with OpenMP
  #pragma omp parallel for shared(_adj) reduction(max:_max_level)
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
      // printf("\tstartGateIdx = %lu %lu (tid = %d)\n", startGateIdx, _adj[startGateIdx].size(), tid);

      for (size_t j = 0; j < _adj[startGateIdx].size(); j++) {
        size_t postGateIdx = _adj[startGateIdx][j];
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
  for (size_t i = 0; i < _adj.size(); i++) {
    if (_adj[i].size() == 0) {
      #pragma omp critical
      {
        _sinks.push_back(i);
      }
    }
  }

  _total_num_levels = _max_level + 1;

  // Construct _gateIdx_in_each_level
  _gateIdx_in_each_level.resize(_total_num_levels);
  // #pragma omp parallel for
  for (size_t gateIdx = 0; gateIdx < _level_of_gates.size(); gateIdx++) {
    size_t gateLvl = _level_of_gates[gateIdx];
    #pragma omp critical
    {
      _gateIdx_in_each_level[gateLvl].push_back(gateIdx);
    }
  }

  // printf("_max_level = %lu, _total_num_levels = %lu\n", 
  //         _max_level, _total_num_levels);


#ifdef CPU_PARTITIONER_PRINT_FOR_CHECK
  // for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
  //   printf("gate_%lu, level = %lu\n", i, _level_of_gates[i]);
  // }

  // for (size_t i = 0; i < _gateIdx_in_each_level.size(); i++) {
  //   printf("level_%lu: size() == %lu\n", i, _gateIdx_in_each_level[i].size());
  // }

  // for (size_t i = _gateIdx_in_each_level.size()-5; i < _gateIdx_in_each_level.size(); i++) {
  //   printf("level_%lu: ", i);
  //   for (size_t j = 0; j < _gateIdx_in_each_level[i].size(); j++) {
  //     printf("%lu, ", _gateIdx_in_each_level[i][j]);
  //   } printf("\n");
  // }
  printf("_max_level = %lu, _total_num_levels = %lu\n", 
          _max_level, _total_num_levels);
#endif
}

void CPUPartitioner::_partition() {
  // printf("start _partition\n");
  
  bool stop = false; 
  size_t startLevel = _max_level; 

  while(!stop) {
    std::vector<std::vector<size_t>> gateBelongsToWhichCones; 
    std::vector<size_t> coneIdxToSinkGateIdx;
    std::vector<std::vector<std::vector<size_t>>> duplicationTable; 
    std::vector<std::vector<std::vector<size_t>>> outputCones;

    int stopLevel = -1;
    _find_all_cones(startLevel, 
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
bool CPUPartitioner::_gateHasPostGates(const size_t gateIdx, const size_t startLevel) {
  if ((_level_of_gates[gateIdx] == startLevel) || 
      (_adj[gateIdx].size() == 0)) {
    return false; 
  }
  
  // All of its post gates's level are larger than startLevel, which means that they have been processed  
  size_t i = 0; 
  while(i < _adj[gateIdx].size()) {
    size_t postGateIdx = _adj[gateIdx][i];
    if (startLevel >= _level_of_gates[postGateIdx]) {
      return true; 
    }
    i++;
  }

  return false;
}

void CPUPartitioner::_getConeIdices(const size_t conesSz, const size_t gateIdx, 
                                    const size_t startLevel, bool newCone, 
                                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones) {
  if (newCone) {
    gateBelongsToWhichCones[gateIdx].push_back(conesSz);
  } else {
    std::unordered_set<size_t> unique_numbers;
    for (size_t postGate = 0; postGate < _adj[gateIdx].size(); postGate++) {
      size_t postGateIdx = _adj[gateIdx][postGate];
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
      bool hasPostGate = _gateHasPostGates(gateIdx, startLevel);
      bool newCone = (!hasCone && !hasPostGate);

      // It can have multiple cone Idices 
      _getConeIdices(_cones.size(), gateIdx, startLevel, 
                    newCone, gateBelongsToWhichCones);
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

void CPUPartitioner::_load_constructed_hypergraph(std::istream &hyg) {
  // printf("Get inside _load_constructed_hypergraph\n");
  // Load _hyperedge_include_which_vertices
  size_t sz0, sz1, vertexIdx, element; 
  hyg >> sz0;
  _hyperedge_include_which_vertices.resize(sz0);
  for (size_t i = 0; i < sz0; i++) {
    hyg >> sz1;
    for (size_t j = 0; j < sz1; j++) {
      hyg >> vertexIdx;
      _hyperedge_include_which_vertices[i].insert(vertexIdx);
    }
  }

  // Load _vertex_include_which_hyperedges
  std::vector<std::set<size_t>> _vertex_include_which_hyperedges;
  hyg >> sz0;
  _vertex_include_which_hyperedges.resize(sz0);
  for (size_t i = 0; i < sz0; i++) {
    hyg >> sz1;
    for (size_t j = 0; j < sz1; j++) {
      hyg >> element;
      _vertex_include_which_hyperedges[i].insert(element);
    }
  }

  // Load _hyperedges_weight
  hyg >> sz0;
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _hyperedges_weight.push_back(element);
  }

  // Load _hyperedges_eptr
  hyg >> sz0;
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _hyperedges_eptr.push_back(element);
  }

  // Load _hyperedges_eind
  hyg >> sz0;
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _hyperedges_eind.push_back(element);
  }  
  
  // Load _vectices_weight
  hyg >> sz0;  
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _vectices_weight.push_back(element);
  }  

  // Load _vertexIdx_to_sinkGateIdx
  hyg >> sz0;  
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _vertexIdx_to_sinkGateIdx.push_back(element);
  }  
  
  // Load _sinkGateIdx_to_vertexIdx
  hyg >> sz0;  
  for (size_t i = 0; i < sz0; i++) {
    hyg >> element;
    _sinkGateIdx_to_vertexIdx.push_back(element);
  }  


#ifdef CPU_PRE_CONSTRUCTION_OF_HYPERGRAPH
  print_preconstruct_hyperedge_vertices_details();
#endif
}

void CPUPartitioner::_get_hypergraph(const int mode, std::istream &hyg,
                                    std::chrono::duration<double> &duration_load_data) {
  // printf("_get_hypergraph\n");
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  switch (mode) {
    case 0: { // construct hypergraph 
      start = std::chrono::steady_clock::now();
      _find_clusters();
      if (_can_run_kahypar) {
        _construct_hyperGraph();
      } 
      end = std::chrono::steady_clock::now();
      break;
    }
    case 1: { // load pre-construct hypergraph 
      start = std::chrono::steady_clock::now();
      size_t tmp;
      hyg >> tmp; 
      if (tmp == 1) {
        _load_constructed_hypergraph(hyg);
        _can_run_kahypar = true; 
      } else { // tmp == 0
        _find_clusters();
        _can_run_kahypar = false; 
      }
      end = std::chrono::steady_clock::now();
      break;
    }
    default:
      printf("No SUCH CASES\n");
      return; 
  }
  duration_load_data = end - start;
}

void CPUPartitioner::_RepCut(std::chrono::duration<double> &duration_KHP) {
  auto start = std::chrono::steady_clock::now();
  if (_can_run_kahypar) {
    _run_KaHyPar();
  } 
  auto end0 = std::chrono::steady_clock::now();
  duration_KHP = end0 - start;
}

void CPUPartitioner::_RepCut_MT(std::chrono::duration<double> &duration_KHP, 
                                const size_t num_threads) {
  auto start = std::chrono::steady_clock::now();
  if (_can_run_kahypar) {
    _run_mtKaHyPar(num_threads);
  } 
  auto end0 = std::chrono::steady_clock::now();
  duration_KHP = end0 - start;
}


/* For those gates who belong to the same cones belong to the same cluster */
void CPUPartitioner::_find_clusters() {
  // printf("_find_clusters\n");
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

  // std::vector<bool> get_cluster;  // tmp 
  // get_cluster.resize(_sum_pi_gates_pos, false);
  // _gate_to_cluster.resize(_sum_pi_gates_pos);

  // // Get _gate_to_cluster
  // for (size_t gate0_Idx = 0; gate0_Idx < _sum_pi_gates_pos; gate0_Idx++) {
  //   if (!get_cluster[gate0_Idx]) {
  //     std::vector<size_t> tmp;
  //     tmp.push_back(gate0_Idx);
  //     _clusters.push_back(tmp);
  //     _gate_to_cluster[gate0_Idx] = _clusters.size()-1;
  //     get_cluster[gate0_Idx] = true;
  //   }

  //   size_t cluster_0_idx = _gate_to_cluster[gate0_Idx];

  //   for (size_t gate1_Idx = 0; gate1_Idx < _sum_pi_gates_pos; gate1_Idx++) {
  //     if (!get_cluster[gate1_Idx] && (_gate_to_cones[gate0_Idx] == _gate_to_cones[gate1_Idx])) {
  //       _clusters[cluster_0_idx].push_back(gate1_Idx);
  //       _gate_to_cluster[gate1_Idx] = cluster_0_idx;
  //       get_cluster[gate1_Idx] = true;        
  //     }
  //   }
  // }

  // 1108 new ver
  std::vector<bool> get_cluster;  // tmp 
  get_cluster.resize(_sum_pi_gates_pos, false);
  _gate_to_cluster.resize(_sum_pi_gates_pos);

  for (size_t c = 0; c < _cones.size(); c++) {
    for (size_t level = 0; level < _cones[c].size(); level++) {
      for (size_t g = 0; g < _cones[c][level].size(); g++) {
        size_t gateIdx1 = _cones[c][level][g];
        
        if (!get_cluster[gateIdx1]) {
          std::vector<size_t> tmp;
          tmp.push_back(gateIdx1);
          _clusters.push_back(tmp);
          _gate_to_cluster[gateIdx1] = _clusters.size()-1;
          get_cluster[gateIdx1] = true;
        }
        size_t cluster_0_idx = _gate_to_cluster[gateIdx1];

        for (size_t level1 = 0; level1 < _cones[c].size(); level1++) {
          for (size_t g1 = 0; g1 < _cones[c][level1].size(); g1++) {
            size_t gateIdx2 = _cones[c][level1][g1];
            if (gateIdx1 != gateIdx2) {
              if (!get_cluster[gateIdx2] && 
                  (_gate_to_cones[gateIdx1] == _gate_to_cones[gateIdx2])) {
                _clusters[cluster_0_idx].push_back(gateIdx2);
                _gate_to_cluster[gateIdx2] = cluster_0_idx;
                get_cluster[gateIdx2] = true;        
              }
            }
          }
        }

      } 
    }
  }
  // 1108 new ver


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
  _sinkGateIdx_to_vertexIdx.resize(_sum_pi_gates_pos, 0);

  // Get _hyperedge_include_which_vertices
  _hyperedge_include_which_vertices.resize(_non_sink_clusters_idx.size()); 
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
    _sinkGateIdx_to_vertexIdx[sinkIdx] = vertexIdx;

    for (size_t lvl = 0; lvl < _cones[c].size(); lvl++) {
      for (size_t g = 0; g < _cones[c][lvl].size(); g++) {
        size_t gateIdx = _cones[c][lvl][g];
        size_t clusterIdx = _gate_to_cluster[gateIdx];
        if (!_is_sink_clusters[clusterIdx]) { // is hyperegde
          size_t hyperedgeIdx = _to_hypergraph_vertex_idx[clusterIdx];
          _hyperedge_include_which_vertices[hyperedgeIdx].insert(vertexIdx);
          
          // Update _hyperedges_weight
          _hyperedges_weight[hyperedgeIdx] = _clusters[clusterIdx].size();
        } 
      }
    }
  }
  
  /* Get _hyperedges_weight, _hyperedges_eind, _hyperedges_eptr */
  for (size_t hyperedge = 0; hyperedge < _hyperedge_include_which_vertices.size(); hyperedge++) {
    // Update _hyperedges_eptr
    _hyperedges_eptr.push_back(_hyperedges_eind.size());
    for (const auto& vertexIdx : _hyperedge_include_which_vertices[hyperedge]) {
      // Update _hyperedges_eind
      _hyperedges_eind.push_back(vertexIdx);
    }
  } 
  // Update _hyperedges_eptr
  _hyperedges_eptr.push_back(_hyperedges_eind.size()); // final one 
  

  // Get _vertex_include_which_hyperedges
  // std::vector<std::set<size_t>> _vertex_include_which_hyperedges;
  _vertex_include_which_hyperedges.resize(_is_sink_clusters_idx.size());

  for (size_t hyperedgeIdx = 0; hyperedgeIdx < _hyperedge_include_which_vertices.size(); hyperedgeIdx++) {
    for (const auto& vertexIdx : _hyperedge_include_which_vertices[hyperedgeIdx]) {
      _vertex_include_which_hyperedges[vertexIdx].insert(hyperedgeIdx);
    }
  }  

  /* Get _vectices_weight */
  std::vector<double> tmp_vectices_weight;
  _compute_vertex_weight(tmp_vectices_weight, 
                        _vertex_include_which_hyperedges);

  /* Normalize double to size_t*/
  _multiple_100(tmp_vectices_weight);
  
#ifdef PART_DEBUG_REPCUT_PRINT_FIND_CLUSTERS
  print_hyperedge_vertices_details();
#endif 

  if (_preconstruct_hypergraph_print_flag) {
    print_preconstruct_hyperedge_vertices_details();
  }
}

void CPUPartitioner::_compute_vertex_weight(std::vector<double> &tmp_vectices_weight, 
                      std::vector<std::set<size_t>> &_vertex_include_which_hyperedges) {

  tmp_vectices_weight.resize(_is_sink_clusters_idx.size());
  for (size_t cl = 0; cl < _is_sink_clusters_idx.size(); cl++) {
    size_t clusterIdx = _is_sink_clusters_idx[cl];
    size_t vertexIdx = _to_hypergraph_vertex_idx[clusterIdx];
    double sumOfYita = 0.0;

    for (const auto& hyperedgeIdx : _vertex_include_which_hyperedges[vertexIdx]) {
      sumOfYita += _hyperedges_weight[hyperedgeIdx];
      // printf("sumOfYita = %.3lf\n", sumOfYita);
    }

    if (_vertex_include_which_hyperedges[vertexIdx].size() != 0) {
      sumOfYita /= double(_vertex_include_which_hyperedges[vertexIdx].size());
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

void CPUPartitioner::_run_KaHyPar() {
  // printf("_run_KaHyPar\n");

  kahypar_context_t* context = kahypar_context_new();
  kahypar_configure_context_from_file(context, "/home/ychung79/kahypar/config/repcut.ini");
  
  kahypar_set_seed(context, -1);

  /* Settings of #V and #he of H */
  const kahypar_hypernode_id_t num_vertices = _vectices_weight.size();
  const kahypar_hyperedge_id_t num_hyperedges = _hyperedges_weight.size();
  // printf("_vectices_weight.size() = %lu, _hyperedges_weight.size() = %lu\n", 
  //         _vectices_weight.size(), _hyperedges_weight.size());

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
    _vertex_to_partition.push_back(size_t(partition[i]));
  }

  kahypar_context_free(context);

#ifdef PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS
  printf("KaHyPar results\n");
  for(size_t i = 0; i < _vertex_to_partition.size(); i++) {
    std::cout << i << ":" << _vertex_to_partition[i] << std::endl;
  }  
#endif
}

void CPUPartitioner::_run_mtKaHyPar(const size_t num_threads) {
  // printf("_run_mtKaHyPar\n");
  
  /* Construct context */
  mt_kahypar_initialize_thread_pool(
    num_threads, 
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into two blocks
  // with an allowed _imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
                                        _k_sizet /* number of blocks */, 
                                        _imbalance /* _imbalance parameter */,
                                        KM1 /* objective function */);

  mt_kahypar_set_seed(42 /* seed */);

  /* Construct hypergraph */
  // Settings of #V and #he of H
  const mt_kahypar_hypernode_id_t num_vertices = _vectices_weight.size();
  const mt_kahypar_hyperedge_id_t num_hyperedges = _hyperedges_weight.size();
  

  // Define hyperedges: each hyperedge connects multiple vertices
  std::vector<std::vector<mt_kahypar_hypernode_id_t>> hyperedges;
  for (const auto& vertex_set : _hyperedge_include_which_vertices) {
    std::vector<mt_kahypar_hypernode_id_t> hyperedge(vertex_set.begin(), vertex_set.end());
    hyperedges.push_back(hyperedge);  
  }


  // Flatten the hyperedges array into a single vector of vertices
  std::vector<mt_kahypar_hypernode_id_t> edge_vector;
  std::vector<size_t> edge_offsets;  // Offsets for where each hyperedge starts in the flattened array
  for (const auto& edge : hyperedges) {
    edge_offsets.push_back(edge_vector.size());  // Mark the start of each hyperedge
    edge_vector.insert(edge_vector.end(), edge.begin(), edge.end());  // Insert all vertices in the hyperedge
  }
  edge_offsets.push_back(edge_vector.size());  // Final offset to mark the end

  // // Optional: Weights for hyperedges and vertices (can be nullptr if not used)
  // Inputs for vertex weights (converted from KaHyPar example)
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights = 
              std::make_unique<mt_kahypar_hypernode_weight_t[]>(num_vertices);
  for (size_t i = 0; i < num_vertices; i++) {
    vertex_weights[i] = _vectices_weight[i];
  }

  // Inputs for hyperedge weights (converted from KaHyPar example)
  std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> hyperedge_weights = 
        std::make_unique<mt_kahypar_hyperedge_weight_t[]>(num_hyperedges);
  for (size_t i = 0; i < (num_hyperedges); i++) {
    hyperedge_weights[i] = _hyperedges_weight[i];
  }

  // Define the preset type (e.g., DEFAULT)
  mt_kahypar_preset_type_t preset = DEFAULT;  // Choose a preset type (e.g., DEFAULT or another preset)

  // Now, create the hypergraph in memory using the correct API
  mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
    preset,  // Preset type
    num_vertices,  // Number of vertices
    num_hyperedges,  // Number of hyperedges
    edge_offsets.data(),  // Array of hyperedge indices (offsets)
    edge_vector.data(),   // Flattened array of hyperedges (vertices)
    hyperedge_weights.get(),  // Hyperedge weights (nullptr if unweighted)
    vertex_weights.get()      // Vertex weights (nullptr if unweighted)
  );

  // Partition the hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);
  // Extract the partition information (which block each vertex belongs to)
  std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, partition.get());

  /* Extract block weights */
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(_k);  // Since we have _k blocks
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  /* Compute Metrics */
  const double imbalance_2 = mt_kahypar_imbalance(partitioned_hg, context);
  const double km1_2 = mt_kahypar_km1(partitioned_hg);

  for(size_t i = 0; i < num_vertices; i++) {
    _vertex_to_partition.push_back(size_t(partition[i]));
  }


  /* Cleanup */
  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);

#ifdef PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS
  printf("MTKaHyPar results\n");

  /* Output Results */
  std::cout << "BIBI:" << std::endl;
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance_2 << std::endl;
  std::cout << "Km1               = " << km1_2 << std::endl;
  std::cout << "Weight of Block 0 = " << block_weights[0] << std::endl;
  std::cout << "Weight of Block 1 = " << block_weights[1] << std::endl;

  // Output the partition results (which partition each vertex belongs to)
  std::cout << "Partitioned Vertices:" << std::endl;
  for (mt_kahypar_hypernode_id_t i = 0; i < num_vertices; ++i) {
    std::cout << "Vertex " << i << " is in partition " << partition[i] << std::endl;
  }  

  for(size_t i = 0; i < _vertex_to_partition.size(); i++) {
    std::cout << i << ":" << _vertex_to_partition[i] << std::endl;
  }  
#endif
}


void CPUPartitioner::_construct_cones(std::vector<size_t> &sinks,
                                       std::vector<std::set<size_t>> &cones_set_g,
                                       std::vector<bool> &visited_cpu) {

  // printf("sinks.size() = %lu\n", sinks.size());
  for (size_t sink = 0; sink < sinks.size(); sink++) {
    size_t sinkIdx = sinks[sink];
    std::queue<size_t> preGates; 
    preGates.push(sinkIdx);
    // used to check the correctness 
    visited_cpu[sinkIdx] = true;
    // used to check the correctness 

    size_t sourceLevel = _level_of_gates[sinkIdx];
    cones_set_g[sourceLevel].insert(sinkIdx);


    while (!preGates.empty()) {
      size_t startGateIdx = preGates.front(); 
      preGates.pop();
      visited_cpu[startGateIdx] = true;

      for (size_t j = 0; j < _invAdj[startGateIdx].size(); j++) {
        size_t preGateIdx = _invAdj[startGateIdx][j];
        size_t preGateLevel = _level_of_gates[preGateIdx];
        cones_set_g[preGateLevel].insert(preGateIdx);
        preGates.push(preGateIdx);
      }
    }
  }
}

void CPUPartitioner::_get_sinks_group(std::vector<std::vector<size_t>> &sinks_groups) {
  // Get the group of sinks 
  sinks_groups.resize(_k_sizet);

  // Get sinks 
  for (size_t sinkIdx = 0; sinkIdx < _sum_pi_gates_pos; sinkIdx++) {
    size_t outputSize = _adj[sinkIdx].size();
    if (outputSize == 0) {
      size_t vertexIdx = _sinkGateIdx_to_vertexIdx[sinkIdx];
      size_t sinkGroup = _vertex_to_partition[vertexIdx];
      sinks_groups[sinkGroup].push_back(sinkIdx);
    }
  }
}

void CPUPartitioner::_construct_groups(){
  std::vector<std::vector<size_t>> sinks_groups;
  _get_sinks_group(sinks_groups);

  // used to check the correctness 
  std::vector<bool> visited_cpu;
  visited_cpu.resize(_sum_pi_gates_pos, false);
  // used to check the correctness 

  // Construct cones 
  _RepCut_output_cones_set.resize(_k_sizet);
  _RepCut_output_cones.resize(_k_sizet);
  for (size_t g = 0; g < sinks_groups.size(); g++) {
    // Push in NULL cone
    _RepCut_output_cones_set[g].resize(_total_num_levels);
    _RepCut_output_cones[g].resize(_total_num_levels);
    _construct_cones(sinks_groups[g], _RepCut_output_cones_set[g], visited_cpu);
  }
}

void CPUPartitioner::_construct_partitioned_groups() {
  // Construct group and show constructed group 
  _construct_groups();

  // Construct _RepCut_output_cones
  for (size_t i = 0; i < _RepCut_output_cones_set.size(); ++i) {
    for (size_t lvl = 0; lvl < _RepCut_output_cones_set[i].size(); ++lvl) {
      for (const auto& element : _RepCut_output_cones_set[i][lvl]) {
        size_t w_level = _max_level - lvl;
        _RepCut_output_cones[i][w_level].push_back(element);
      }
    }
  }
}

void CPUPartitioner::_get_KHP_partitioning_res(const int hypergraph_construct_mode) {
  // printf("hypergraph_construct_mode = %d\n", hypergraph_construct_mode);

  switch (hypergraph_construct_mode) {
    case 0: { // construct the hypergraph in the code
      if (!_can_run_kahypar) {
        _RepCut_output_cones = _cones;
      } else {
        _RepCut_output_cones_set.resize(_k);
        _RepCut_output_cones.resize(_k);
        for (size_t i = 0; i < _k_sizet; i++) {
          _RepCut_output_cones_set[i].resize(_total_num_levels);
          _RepCut_output_cones[i].resize(_total_num_levels);
        }
      
        for (size_t i = 0; i < _is_sink_clusters_idx.size(); i++) {
          size_t clusterIdx = _is_sink_clusters_idx[i];
          size_t vertexIdx = _to_hypergraph_vertex_idx[clusterIdx];
          size_t partitionIdx = _vertex_to_partition[vertexIdx];
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
      }
      break;
    }
    case 1: { // load pre-computed hypergraph 
      if (!_can_run_kahypar) {
        _RepCut_output_cones = _cones;
      } else {
        _construct_partitioned_groups();
      }
      break;
    }
    default:
      printf("No SUCH CASES\n");
      return; 
  }

#ifdef PART_DEBUG_REPCUT_PRINT_KAHYPAR_RESULTS
  _print_RepCut_output_cones();
#endif
}

// --------------------------------------------------------------------------------------

void CPUPartitioner::prepare_cpu_simulation(const int hypergraph_construct_mode, Mode mode) {
  auto start = std::chrono::steady_clock::now();
  _get_KHP_partitioning_res(hypergraph_construct_mode);

  _construct_graph();
  _ask_simulation_memory();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  // std::cout << "prepare_CPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";  

  _count_duplications(mode);
}


void CPUPartitioner::_ask_gpu_memory_simu_1(Mode mode){
  // prepare _invAdj_gpu and _invAdj_index_table_gpu
  std::vector<size_t> invAdj_tmp;
  std::vector<size_t> invAdj_index_tmp;
  size_t accum = 0;
  for (size_t i = 0; i < _invAdj.size(); i++) {
    invAdj_index_tmp.push_back(accum);
    for (size_t j = 0; j < _invAdj[i].size(); j++) {
      invAdj_tmp.push_back(_invAdj[i][j]);
      accum++;
    }
    invAdj_index_tmp.push_back(accum);
  }

  cudaMalloc((void**)&_invAdj_gpu, (invAdj_tmp.size())*sizeof(size_t));
  cudaMalloc((void**)&_invAdj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(size_t));
  cudaCheckErrors("CUDA: _invAdj_gpu  cudaMalloc - Failure");
  // printf("invAdj_index_tmp.size() = %lu, 2*_sum_pi_gates_pos = %lu\n", 
  //         invAdj_index_tmp.size(), 2*_sum_pi_gates_pos);

  cudaMemcpyAsync(_invAdj_gpu, invAdj_tmp.data(), 
                   (invAdj_tmp.size())*sizeof(size_t), cudaMemcpyHostToDevice);  
  cudaMemcpyAsync(_invAdj_index_table_gpu, invAdj_index_tmp.data(), 
                   (invAdj_index_tmp.size())*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _invAdj_gpu cudaMemcpyAsync - Failure");

  // prepare _numGates_per_level_gpu_of_groups and _cones_partitioned_gpu
  std::vector<size_t> cones_partitioned_tmp;
  std::vector<size_t> per_level_of_group_start_accum_tmp;
  switch (mode) {
    case Mode::SEQUENTIAL: 
    case Mode::PARALLEL_OMP: {
      accum = 0;
      per_level_of_group_start_accum_tmp.push_back(accum);

      for (size_t lvl = 0; lvl < _gateIdx_in_each_level.size(); lvl++) {
        size_t in_accum = 0;
        for (size_t g = 0; g < _gateIdx_in_each_level[lvl].size(); g++) {
          size_t element = _gateIdx_in_each_level[lvl][g];
          cones_partitioned_tmp.push_back(element);
          in_accum++;
          accum++;
        }
        _numGates_per_level_gpu_of_groups.push_back(in_accum);
        per_level_of_group_start_accum_tmp.push_back(accum);
      }

      break;
    }
    case Mode::REPCUT_PARALLEL_OMP: 
    case Mode::MT_REPCUT_PARALLEL_OMP: {
      accum = 0;
      per_level_of_group_start_accum_tmp.push_back(accum);

      for (size_t i = 0; i < _RepCut_output_cones.size(); ++i) {
        for (int lvl = _RepCut_output_cones[i].size()-1; lvl > -1; lvl--) {
          size_t in_accum = 0;
          for (const auto& element : _RepCut_output_cones[i][lvl]) {
            cones_partitioned_tmp.push_back(element);
            in_accum++;
            accum++;
          }
          _numGates_per_level_gpu_of_groups.push_back(in_accum);
          per_level_of_group_start_accum_tmp.push_back(accum);
          // printf("accum = %lu, i = %lu, _RepCut_output_cones.size() = %lu\n", accum, i, _RepCut_output_cones.size());
        }
      }
      break;
    }
    default:
      printf("No SUCH CASES\n");
      return; 
  }

  // for (size_t i = 0; i < cones_partitioned_tmp.size(); i++) {
  //   printf("cones_partitioned_tmp[%lu] = %lu\n", i, cones_partitioned_tmp[i]);
  // }
  // for (size_t i = 0; i < per_level_of_group_start_accum_tmp.size(); i++) {
  //   printf("per_level_of_group_start_accum_tmp[%lu] = %lu\n", i, per_level_of_group_start_accum_tmp[i]);
  // }  
// =====================================================================================
  size_t pis = 0, pos = 0, gatess = 0;
  for (size_t i = 0; i < cones_partitioned_tmp.size(); i++) {
    if (cones_partitioned_tmp[i] < _num_PIs) {
      pis++;
    } else if (cones_partitioned_tmp[i] < (_num_PIs+_num_POs) && cones_partitioned_tmp[i] >= _num_PIs) {
      pos++;
    } else {
      gatess++;
    }
  }
  // printf("cones_partitioned_tmp.size() = %lu, pis = %lu, pos = %lu, gatess = %lu\n", 
  //         cones_partitioned_tmp.size(), pis, pos, gatess);
// =====================================================================================

  cudaMalloc((void**)&_cones_partitioned_gpu, (cones_partitioned_tmp.size())*sizeof(size_t));
  cudaMalloc((void**)&_per_level_of_group_start_accum_gpu, (_k*_total_num_levels+1)*sizeof(size_t));
  cudaMalloc((void**)&_numGates_per_level_gpu_of_groups_gpu, (_numGates_per_level_gpu_of_groups.size())*sizeof(size_t));
  cudaCheckErrors("CUDA: _per_level_of_group_start_accum_gpu cudaMalloc - Failure");

  // printf("_total_num_levels = %lu, (_k*_total_num_levels+1) = %lu, per_level_of_group_start_accum_tmp.data() = %lu\n", 
  //       _total_num_levels, 
  //       (_k*_total_num_levels+1), 
  //       per_level_of_group_start_accum_tmp.data());

  cudaMemcpyAsync(_cones_partitioned_gpu, cones_partitioned_tmp.data(), 
                (cones_partitioned_tmp.size())*sizeof(size_t), cudaMemcpyHostToDevice);  
  cudaMemcpyAsync(_per_level_of_group_start_accum_gpu, per_level_of_group_start_accum_tmp.data(), 
                (per_level_of_group_start_accum_tmp.size())*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(_numGates_per_level_gpu_of_groups_gpu, _numGates_per_level_gpu_of_groups.data(), 
                  (_numGates_per_level_gpu_of_groups.size())*sizeof(size_t), cudaMemcpyHostToDevice);                 
  cudaCheckErrors("CUDA: _ask_gpu_memory_simu_1 cudaMemcpyAsync - Failure");
}

void CPUPartitioner::_ask_gpu_memory_simu_2(){
  cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, _sum_pi_gates_pos*sizeof(size_t));
  cudaMalloc((void**)&_patterns_gpu, _num_rounds*_num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_fault_gate_idx_gpu, _num_fault*sizeof(size_t));
  cudaMalloc((void**)&_fault_SA_fault_val_gpu, _num_fault*sizeof(size_t));
  
  cudaMalloc((void**)&_pi_gate_po_output_res_gpu, _sum_pi_gates_pos*sizeof(size_t));
  cudaMalloc((void**)&_g_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_g_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_g_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_b_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_b_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_b_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_found_fault_to_pattern_gpu, 2*_num_fault*sizeof(size_t));  
}

void CPUPartitioner::_move_GateType_h2d() {
  // pi_gate_po_gate_type_gpu
  std::vector<size_t>pi_gate_po_gate_type;

  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    pi_gate_po_gate_type.push_back(static_cast<size_t>(_gate_type[g]));
  }

  cudaMemcpyAsync(_pi_gate_po_gate_type_gpu, pi_gate_po_gate_type.data(), 
                  (pi_gate_po_gate_type.size())*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _pi_gate_po_gate_type_gpu cudaMemcpy failure"); 
}

void CPUPartitioner::_move_patterns_h2d() {
  std::vector<size_t> patterns_cpu;
  for (size_t i = 0; i < _num_rounds; i++) {
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      patterns_cpu.push_back(_patterns[i]._value[pi]);
    }
  }

  cudaMemcpy(_patterns_gpu, patterns_cpu.data(), 
            (_num_rounds*_num_PIs)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _patterns_gpu cudaMemcpy failure");
}

void CPUPartitioner::_move_faults_h2d() {
  std::vector<size_t> fault_gate_idx;
  std::vector<size_t> fault_SA_fault_val;

  for (size_t i = 0; i < _num_fault; i++) {
    fault_gate_idx.push_back(_faults[i]._gate_with_fault);
    fault_SA_fault_val.push_back(_faults[i]._gate_SA_fault_val);
  }

  cudaMemcpy(_fault_gate_idx_gpu, fault_gate_idx.data(), _num_fault*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_fault_SA_fault_val_gpu, fault_SA_fault_val.data(), _num_fault*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _fault_gate_idx_gpu OR _fault_SA_fault_val_gpu cudaMemcpy failure");
}


void CPUPartitioner::prepare_gpu_simulation(Mode mode, const int hypergraph_construct_mode) {
  // printf("CPUPartitioner::prepare_gpu_simulation\n");
  auto start = std::chrono::steady_clock::now();
  _get_KHP_partitioning_res(hypergraph_construct_mode);

  _ask_gpu_memory_simu_1(mode);
  _ask_gpu_memory_simu_2();
  _move_GateType_h2d();
  _move_patterns_h2d();
  _move_faults_h2d();

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  // std::cout << "prepare_GPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";

  _count_duplications(mode);
}



// Run the simulation 
void CPUPartitioner::run(SIMU_Mode simu_mode, Mode mode, const size_t num_threads, const size_t NUM_SIMULATION_RDS) {
  switch (simu_mode) {
    case SIMU_Mode::CPU: {
      _run_cpu_simulator(mode, num_threads);
      break;
    }
    case SIMU_Mode::GPU: {
      _run_gpu_simulator(mode, NUM_SIMULATION_RDS);
      break;      
    }
    default: {
      printf("No SUCH CASES\n");
      return;
    }
  }
  // printf("_num_PIs = %lu, _num_POs = %lu, _num_inner_gates = %lu, _sum_pi_gates_pos = %lu\n", 
  //         _num_PIs, _num_POs, _num_inner_gates, _sum_pi_gates_pos);
}

void CPUPartitioner::_run_gpu_simulator(Mode mode, const size_t NUM_SIMULATION_RDS) {
  switch (mode) {
    case Mode::SEQUENTIAL: 
    case Mode::PARALLEL_OMP: {
      GALPS_SIZET_GPUSimulator gpuSimulator;
      gpuSimulator.run_gpu_simulator_level_sizet_gpu(_k_sizet, _num_PIs, _num_inner_gates, _num_POs, 
                                                    _sum_pi_gates_pos, 
                                                    _num_pattern, _num_rounds, _num_fault,
                                                    _pi_gate_po_gate_type_gpu, 
                                                    _patterns_gpu,
                                                    _fault_gate_idx_gpu,
                                                    _fault_SA_fault_val_gpu,
                                                    _pi_gate_po_output_res_gpu,
                                                    _numGates_per_level_gpu_of_groups,
                                                    _per_level_of_group_start_accum_gpu,
                                                    _cones_partitioned_gpu,
                                                    _total_num_levels,
                                                    _invAdj_gpu,
                                                    _invAdj_index_table_gpu,
                                                    _patterns, 
                                                    _g_pi_results_gpu,
                                                    _g_gate_results_gpu,
                                                    _g_po_results_gpu,
                                                    _b_pi_results_gpu,
                                                    _b_gate_results_gpu,
                                                    _b_po_results_gpu,
                                                    _found_fault_to_pattern_gpu,
                                                    NUM_SIMULATION_RDS);
      cudaDeviceSynchronize(); 
      break;
    }
    case Mode::REPCUT_PARALLEL_OMP: 
    case Mode::MT_REPCUT_PARALLEL_OMP: {
      GALPS_SIZET_GPUSimulator gpuSimulator;
      gpuSimulator.run_gpu_simulator_cones_part_sizet_gpu(_k_sizet, _num_PIs, _num_inner_gates, _num_POs, 
                                                          _sum_pi_gates_pos, 
                                                          _num_pattern, _num_rounds, _num_fault,
                                                          _pi_gate_po_gate_type_gpu, 
                                                          _patterns_gpu,
                                                          _fault_gate_idx_gpu,
                                                          _fault_SA_fault_val_gpu,
                                                          _pi_gate_po_output_res_gpu,
                                                          _numGates_per_level_gpu_of_groups,
                                                          _numGates_per_level_gpu_of_groups_gpu, 
                                                          _per_level_of_group_start_accum_gpu,
                                                          _cones_partitioned_gpu,
                                                          _total_num_levels,
                                                          _invAdj_gpu,
                                                          _invAdj_index_table_gpu,
                                                          _patterns, 
                                                          _g_pi_results_gpu,
                                                          _g_gate_results_gpu,
                                                          _g_po_results_gpu,
                                                          _b_pi_results_gpu,
                                                          _b_gate_results_gpu,
                                                          _b_po_results_gpu,
                                                          _found_fault_to_pattern_gpu,
                                                          NUM_SIMULATION_RDS);
      cudaDeviceSynchronize();                                               
      break;      
    }

    default: {
      printf("No SUCH CASES\n");
      return;
    }
  }


  cudaDeviceSynchronize();
}




void CPUPartitioner::_run_cpu_simulator(Mode mode, const size_t num_threads) {
  switch (mode) {
    case Mode::SEQUENTIAL: {
      // std::cout << "run SEQUENTIAL\n";
      assert(num_threads == 1 && "RUN SEQUENTIAL num_threads num_threads > 1");
      _run_fsim(num_threads);
      break;
    }
    case Mode::PARALLEL_OMP: {
      // std::cout << "run PARALLEL_OMP\n";
      _run_fsim(num_threads);
      break;
    }
    case Mode::REPCUT_PARALLEL_OMP: {
      _run_RepCut_cones_gates(num_threads);
      break;      
    }
    case Mode::MT_REPCUT_PARALLEL_OMP: {
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
    // #pragma omp parallel for schedule(dynamic)
    #pragma omp parallel for
    for (size_t g = 0; g < _gateIdx_in_each_level[lvl].size(); g++) {
      int thread_id = omp_get_thread_num();
      size_t gateIdx = _gateIdx_in_each_level[lvl][g];
      // printf("\tthread_id = %d, gateIdx = %lu\n", thread_id, gateIdx);

      // Select the correct element (PI, PO, or gate)
      ElementBase<> &tmp = _Gates[gateIdx];

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
    // #pragma omp parallel for schedule(dynamic)
    #pragma omp parallel for
    for (size_t g = 0; g < _gateIdx_in_each_level[lvl].size(); g++) {
      int thread_id = omp_get_thread_num();
      size_t gateIdx = _gateIdx_in_each_level[lvl][g];

      // Calculate SA_fault using a ternary operator (avoiding if-else)
      size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx) == wrong_gate) :
                        (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx+_num_PIs+_num_POs) == wrong_gate) :
                        ((gateIdx+_num_PIs) == wrong_gate);

      // Select the correct element (PI, PO, or gate)
      ElementBase<> &tmp = _Gates[gateIdx];

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
      ElementBase<> &tmp = _Gates[gateIdx];
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
        size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx) == wrong_gate) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx+_num_PIs+_num_POs) == wrong_gate) :
                          ((gateIdx + _num_PIs) == wrong_gate);

        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = _Gates[gateIdx];

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
      // #pragma omp parallel for
      for (const auto& gateIdx : _RepCut_output_cones[c][lvl]) {
        // printf("c = %lu, lvl = %d, gateIdx = %lu\n", c, lvl, gateIdx);
        
        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = _Gates[gateIdx];

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
        size_t SA_fault = (gateIdx < _num_PIs) ? ((gateIdx) == wrong_gate) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx+_num_PIs+_num_POs) == wrong_gate) :
                          ((gateIdx + _num_PIs) == wrong_gate);

        // Select the correct element (PI, PO, or gate)
        ElementBase<> &tmp = _Gates[gateIdx];

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
  // Duplicated the answer into a new memory
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    if (i < _num_PIs) { // PI
      pi_output[i] = _Gates[i]._output_value;
    } else if (i >= _num_PIs && i < (_num_PIs+_num_POs)) { // PO
      po_output[i-_num_PIs] = _Gates[i]._output_value;
    } else { // gate
      gate_output[i-(_num_PIs+_num_POs)] = _Gates[i]._output_value;
    }
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


void CPUPartitioner::print_hyperedge_vertices_details() const {

  printf("_hyperedge_include_which_vertices:\n");
  for (size_t hyperedge = 0; hyperedge < _hyperedge_include_which_vertices.size(); hyperedge++) {
    printf("hyperedge %lu includes vertex indices: ", hyperedge);
    for (const auto& vertexIdx : _hyperedge_include_which_vertices[hyperedge]) {
      printf("%lu, ", vertexIdx);
    }
    printf("\n");
  }
  printf("\n");


  printf("_vertex_include_which_hyperedges:\n");
  for (size_t vertexIdx = 0; vertexIdx < _vertex_include_which_hyperedges.size(); vertexIdx++) {
    printf("vertexIdx %lu includes vertex indices: ", vertexIdx);
    for (const auto& hyperedge : _vertex_include_which_hyperedges[vertexIdx]) {
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

void CPUPartitioner::print_preconstruct_hyperedge_vertices_details() const {
  printf("_hyperedge_include_which_vertices:\n");
  for (size_t hyperedge = 0; hyperedge < _hyperedge_include_which_vertices.size(); hyperedge++) {
    printf("%lu  ", hyperedge);
    for (const auto& vertexIdx : _hyperedge_include_which_vertices[hyperedge]) {
      printf("%lu ", vertexIdx);
    }
    printf("\n");
  }
  printf("\n");


  printf("_vertex_include_which_hyperedges:\n");
  for (size_t vertexIdx = 0; vertexIdx < _vertex_include_which_hyperedges.size(); vertexIdx++) {
    printf("%lu  ", vertexIdx);
    for (const auto& hyperedge : _vertex_include_which_hyperedges[vertexIdx]) {
      printf("%lu ", hyperedge);
    }
    printf("\n");
  }
  printf("\n");

  printf("_hyperedges_weight: \n");
  for (size_t i = 0; i < _hyperedges_weight.size(); i++) {
    printf("%lu  %lu\n", i, _hyperedges_weight[i]);
  }
  printf("\n");

  printf("_hyperedges_eptr:\n");
  for (size_t i = 0; i < _hyperedges_eptr.size(); i++) {
    printf("%lu\n", _hyperedges_eptr[i]);
  } printf("\n");

  printf("_hyperedges_eind:\n");
  for (size_t i = 0; i < _hyperedges_eind.size(); i++) {
    printf("%lu\n", _hyperedges_eind[i]);
  } printf("\n");

  printf("_vectices_weight\n");
  for (size_t i = 0; i < _vectices_weight.size(); i++) {
    printf("%lu  %lu\n", i, _vectices_weight[i]);
  } 
  printf("\n");

  printf("_vertexIdx_to_sinkGateIdx:\n");
  for (size_t i = 0; i < _vertexIdx_to_sinkGateIdx.size(); i++) {
    printf("%lu  %lu\n", i, _vertexIdx_to_sinkGateIdx[i]);
  } 
  printf("\n");


  printf("_sinkGateIdx_to_vertexIdx:\n");
  for (size_t i = 0; i < _sinkGateIdx_to_vertexIdx.size(); i++) {
    printf("%lu  %lu\n", i, _sinkGateIdx_to_vertexIdx[i]);
  } 
  printf("\n");  


  // printf("=============================\n");  
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
  } else if (mode == Mode::REPCUT_PARALLEL_OMP) {
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

void CPUPartitioner::_count_duplications(Mode mode) {
  switch (mode) {
    case Mode::SEQUENTIAL: 
    case Mode::PARALLEL_OMP: { 

      // printf("_count_duplications:\n");
      // printf("_part_PIs = %lu, %lu\n", _num_PIs, _num_PIs);
      // printf("_part_Gates = %lu, %lu\n", _num_inner_gates, _num_inner_gates);
      // printf("_part_POs = %lu, %lu\n", _num_POs, _num_POs);
      // printf("_part_sum_pi_gates_pos = %lu, %lu\n", _sum_pi_gates_pos, _sum_pi_gates_pos);

      printf("_count_duplications:\n");
      printf("_part_PIs = %lu\n", _num_PIs);
      printf("_part_Gates = %lu\n", _num_inner_gates);
      printf("_part_POs = %lu\n", _num_POs);
      printf("_part_sum_pi_gates_pos = %lu\n", _sum_pi_gates_pos);
      break;  
    }
    case Mode::REPCUT_PARALLEL_OMP: 
    case Mode::MT_REPCUT_PARALLEL_OMP: { // GPU simulator 
      std::vector<bool> visited_cpu;
      visited_cpu.resize(_sum_pi_gates_pos, false);
      std::vector<size_t> numGates; 
      numGates.resize(4, 0);
      // printf("_RepCut_output_cones.size() = %lu\n", _RepCut_output_cones.size());

      for (size_t c = 0; c < _RepCut_output_cones.size(); c++) {
        for (size_t lvl = 0; lvl < _RepCut_output_cones[c].size(); lvl++) {
          for (size_t g = 0; g < _RepCut_output_cones[c][lvl].size(); g++) {
            size_t gateIdx = _RepCut_output_cones[c][lvl][g];
            visited_cpu[gateIdx] = true;

            if (gateIdx < _num_PIs) { // PIs
              numGates[0]++;
            } else if (gateIdx >= _num_PIs && gateIdx < (_num_PIs + _num_POs)) { // POs
              numGates[2]++;
            } else {
              numGates[1]++;
            }
          }
        }
      }  

      numGates[3] = numGates[0]+numGates[1]+numGates[2];
    
      size_t accum = 0;
      for (size_t i = 0; i < visited_cpu.size(); i++) {
        if (!visited_cpu[i]) {
          accum++;
        }
      }
      
      // printf("Partitioned:\n");
      // printf("_part_PIs = %d, %d\n", numGates[0], _num_PIs);
      // printf("_part_Gates = %d, %d\n", numGates[1], _num_inner_gates);
      // printf("_part_POs = %d, %d\n", numGates[2], _num_POs);
      // printf("_part_sum_pi_gates_pos = %d, %d\n", numGates[3], _sum_pi_gates_pos);
      // printf("_part_visited_cpu: accum = %d (should == 0)\n", accum);

      printf("Partitioned:\n");
      printf("_part_PIs = %lu\n", numGates[0]);
      printf("_part_Gates = %lu\n", numGates[1]);
      printf("_part_POs = %lu\n", numGates[2]);
      printf("_part_sum_pi_gates_pos = %lu\n", numGates[3]);
      printf("_part_visited_cpu: accum = %lu (should == 0)\n", accum);      
      break;
    }
    default: {
      printf("No SUCH CASES\n");
      return; 
    }
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

void CPUPartitioner::mt_KaHyParExample() {
  const double imbalance = 0.015;
  const size_t k = 2;
  // BIBI
  /* Construct context */
  // Initialize Mt-KaHyPar
  mt_kahypar_initialize_thread_pool(
    std::thread::hardware_concurrency() /* use all available cores */,
    true /* activate interleaved NUMA allocation policy */ );

  // Setup partitioning context
  mt_kahypar_context_t* context = mt_kahypar_context_new();
  mt_kahypar_load_preset(context, DEFAULT /* corresponds to MT-KaHyPar-D */);
  // In the following, we partition a hypergraph into two blocks
  // with an allowed imbalance of 3% and optimize the connective metric (KM1)
  mt_kahypar_set_partitioning_parameters(context,
                                          k /* number of blocks */, 
                                          imbalance /* imbalance parameter */,
                                          KM1 /* objective function */);
  mt_kahypar_set_seed(42 /* seed */);

  /* Construct hypergraph */
  // Number of vertices
  const mt_kahypar_hypernode_id_t num_vertices = 5;  // Example number of vertices

  // Number of hyperedges
  const mt_kahypar_hyperedge_id_t num_hyperedges = 5;  // Example number of hyperedges

  // Define hyperedges: each hyperedge connects multiple vertices
  const std::vector<std::vector<mt_kahypar_hypernode_id_t>> hyperedges = {
    {0, 1, },      // Hyperedge 0 connects vertices 0, 1, 
    {0, 1, 2, },   // Hyperedge 1 connects vertices 0, 1, 2, 
    {1, 2, },      // Hyperedge 2 connects vertices 1, 2, 
    {2, 3, 4, },   // Hyperedge 3 connects vertices 2, 3, 4, 
    {3, 4, }       // Hyperedge 4 connects vertices 3, 4, 
  };

  // Flatten the hyperedges array into a single vector of vertices
  std::vector<mt_kahypar_hypernode_id_t> edge_vector;
  std::vector<size_t> edge_offsets;  // Offsets for where each hyperedge starts in the flattened array
  for (const auto& edge : hyperedges) {
    edge_offsets.push_back(edge_vector.size());  // Mark the start of each hyperedge
    edge_vector.insert(edge_vector.end(), edge.begin(), edge.end());  // Insert all vertices in the hyperedge
  }
  edge_offsets.push_back(edge_vector.size());  // Final offset to mark the end

  // // Optional: Weights for hyperedges and vertices (can be nullptr if not used)
    // const mt_kahypar_hyperedge_weight_t* hyperedge_weights = nullptr;  // No weights for hyperedges
    // const mt_kahypar_hypernode_weight_t* vertex_weights = nullptr;     // No weights for vertices
  // Inputs for vertex weights (converted from KaHyPar example)
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> vertex_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(num_vertices);
  double tmp0[num_vertices] = {250, 267, 433, 250, 250};  // Example vertex weights
  for (size_t i = 0; i < num_vertices; i++) {
    vertex_weights[i] = static_cast<mt_kahypar_hypernode_weight_t>(tmp0[i]);
  }

  // Inputs for hyperedge weights (converted from KaHyPar example)
  std::unique_ptr<mt_kahypar_hyperedge_weight_t[]> hyperedge_weights =
    std::make_unique<mt_kahypar_hyperedge_weight_t[]>(num_hyperedges);
  size_t tmp1[num_hyperedges] = {2, 1, 2, 1, 2};  // Example hyperedge weights
  for (size_t i = 0; i < num_hyperedges; i++) {
    hyperedge_weights[i] = static_cast<mt_kahypar_hyperedge_weight_t>(tmp1[i]);
  }

  // Define the preset type (e.g., DEFAULT)
  mt_kahypar_preset_type_t preset = DEFAULT;  // Choose a preset type (e.g., DEFAULT or another preset)

  // Now, create the hypergraph in memory using the correct API
  mt_kahypar_hypergraph_t hypergraph = mt_kahypar_create_hypergraph(
    preset,  // Preset type
    num_vertices,  // Number of vertices
    num_hyperedges,  // Number of hyperedges
    edge_offsets.data(),  // Array of hyperedge indices (offsets)
    edge_vector.data(),   // Flattened array of hyperedges (vertices)
    hyperedge_weights.get(),  // Hyperedge weights (nullptr if unweighted)
    vertex_weights.get()      // Vertex weights (nullptr if unweighted)
  );

  // Partition the hypergraph
  mt_kahypar_partitioned_hypergraph_t partitioned_hg =
    mt_kahypar_partition(hypergraph, context);

  // Extract the partition information (which block each vertex belongs to)
  std::unique_ptr<mt_kahypar_partition_id_t[]> partition =
    std::make_unique<mt_kahypar_partition_id_t[]>(mt_kahypar_num_hypernodes(hypergraph));
  mt_kahypar_get_partition(partitioned_hg, partition.get());

  /* Extract block weights */
  std::unique_ptr<mt_kahypar_hypernode_weight_t[]> block_weights =
    std::make_unique<mt_kahypar_hypernode_weight_t[]>(2);  // Since we have 2 blocks
  mt_kahypar_get_block_weights(partitioned_hg, block_weights.get());

  /* Compute Metrics */
  const double imbalance_2 = mt_kahypar_imbalance(partitioned_hg, context);
  const double km1_2 = mt_kahypar_km1(partitioned_hg);

  /* Output Results */
  std::cout << "BIBI:" << std::endl;
  std::cout << "Partitioning Results:" << std::endl;
  std::cout << "Imbalance         = " << imbalance_2 << std::endl;
  std::cout << "Km1               = " << km1_2 << std::endl;
  std::cout << "Weight of Block 0 = " << block_weights[0] << std::endl;
  std::cout << "Weight of Block 1 = " << block_weights[1] << std::endl;

  // Output the partition results (which partition each vertex belongs to)
  std::cout << "Partitioned Vertices:" << std::endl;
  for (mt_kahypar_hypernode_id_t i = 0; i < num_vertices; ++i) {
    std::cout << "Vertex " << i << " is in partition " << partition[i] << std::endl;
  }

  /* Cleanup */
  mt_kahypar_free_context(context);
  mt_kahypar_free_hypergraph(hypergraph);
  mt_kahypar_free_partitioned_hypergraph(partitioned_hg);
}

