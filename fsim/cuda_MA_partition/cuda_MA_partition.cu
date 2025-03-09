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
#include <cstddef>  // For int and SIZE_MAX
#include <cstdio>
#include <set>
#include <ranges>


#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "cuda_MA_partition.cuh"
#include <fsim/cuda_MA_partition/cuda_MA_simulation/gpu_simulation.cuh>

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


inline int getCLIdx(int gateIdx) {
  return gateIdx/_NUM_GATES_PER_CL;
}


// Read files
void CUDAMAPartitioner::read(const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {
  // printf("Get inside CUDAMAPartitioner::read first\n");

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

void CUDAMAPartitioner::read(std::istream &ckt, std::istream &flst, std::istream &ptn) {
  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;
  
  _read_graph(ckt);
  _read_fault(flst);
  _read_pattern(ptn);

  // auto start = std::chrono::steady_clock::now();
  // auto end = std::chrono::steady_clock::now();
  // std::chrono::duration<double> duration_GPU_partition = end - start;
  // std::cout << "duration_GPU_partition: " << _round_to((duration_GPU_partition.count())*1000, 0.001) << "\n";
}

void CUDAMAPartitioner::_read_graph(std::istream &ckt) {
  // Map table for recording gate input order
  // Record for each to_gate, its from_gates's are which order (#_inputs, A/A1, B/A2, A3/S, A4)
  _szOfAdj = 0;
  std::vector<std::vector<int>> adj;
  std::vector<std::vector<int>> invAdj;
  adj.resize(_sum_pi_gates_pos);
  invAdj.resize(_sum_pi_gates_pos);

  // Construct adjacent list
  for (int i = 0; i < _num_wires; i++) {
    int gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (int j = 0; j < num_post_gates; j++) {
      int gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      adj[gate_0].push_back(gate_tmp); 
      invAdj[gate_tmp].push_back(gate_0); 
      _szOfAdj++;
    }
  }

  _levelized(adj, invAdj);

  std::vector<std::vector<int>> newIdxAdj;
  std::vector<std::vector<int>> newIdxInvAdj;
  newIdxAdj.resize(_sum_pi_gates_pos);
  newIdxInvAdj.resize(_sum_pi_gates_pos);

  // Update adj, invAdj according to reIndex
  for (size_t i = 0; i < adj.size(); i++) {
    int oriFromGateIdx = i;
    for (size_t j = 0; j < adj[i].size(); j++) {
      int oriToGateIdx = adj[i][j];
      int newFromGateIdx = _newIndex[oriFromGateIdx];
      int newToGateIdx = _newIndex[oriToGateIdx];
      newIdxAdj[newFromGateIdx].push_back(newToGateIdx);
    }
  }
  for (size_t i = 0; i < invAdj.size(); i++) {
    int oriToGateIdx = i;
    for (size_t j = 0; j < invAdj[i].size(); j++) {
      int oriFromGateIdx = invAdj[i][j];
      int newToGateIdx = _newIndex[oriToGateIdx];
      int newFromGateIdx = _newIndex[oriFromGateIdx];
      newIdxInvAdj[newToGateIdx].push_back(newFromGateIdx);
    }
  }

  _level_of_gates.clear();
  _level_of_gates.resize(_sum_pi_gates_pos, 0);
  
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    // traverse its inputs 
    for (size_t fmGate = 0; fmGate < newIdxInvAdj[i].size(); fmGate++) {
      int fmGateIdx = newIdxInvAdj[i][fmGate];
      int new_level = _level_of_gates[fmGateIdx]+1;
      
      // Update _level_of_gates[i]
      _level_of_gates[i] = (new_level > _level_of_gates[i]) ? 
                           (new_level) : 
                           (_level_of_gates[i]);
    }
  }
  
  _numGates_per_level.resize(_max_level+1, 0);
  for (size_t i = 0; i < _level_of_gates.size(); i++) {
    int level = _level_of_gates[i];
    _numGates_per_level[level]++;
  }

  // _print_levelized();

  // Construct _adj and _invAdj
  _adj = (int*)malloc(_szOfAdj*sizeof(int));
  _adj_index_table = (int*)malloc(2*_sum_pi_gates_pos*sizeof(int));
  memset(_adj, 0, _szOfAdj*sizeof(int));
  memset(_adj_index_table, 0, 2*_sum_pi_gates_pos*sizeof(int));

  _invAdj = (int*)malloc(_szOfAdj*sizeof(int));
  _invAdj_index_table = (int*)malloc(2*_sum_pi_gates_pos*sizeof(int));
  memset(_invAdj, 0, _szOfAdj*sizeof(int));
  memset(_invAdj_index_table, 0, 2*_sum_pi_gates_pos*sizeof(int));
  
  int accum = 0;
  for (size_t fromGate = 0; fromGate < newIdxAdj.size(); fromGate++) {
    // Update _adj_index_table
    _adj_index_table[2*fromGate+0] = accum;

    for (size_t toGate = 0; toGate < newIdxAdj[fromGate].size(); toGate++) {
      int toGateIdx = newIdxAdj[fromGate][toGate];
      
      // Update _adj
      _adj[accum] = toGateIdx;
      accum++;
    }
    // Update _adj_index_table
    _adj_index_table[2*fromGate+1] = accum;
  }

  accum = 0;
  for (size_t toGate = 0; toGate < newIdxInvAdj.size(); toGate++) {
    // Update _invAdj_index_table
    _invAdj_index_table[2*toGate+0] = accum;

    for (size_t fromGate = 0; fromGate < newIdxInvAdj[toGate].size(); fromGate++) {
      int toGateIdx = newIdxInvAdj[toGate][fromGate];
      
      // Update _invAdj
      _invAdj[accum] = toGateIdx;
      accum++;
    }
    // Update _invAdj_index_table
    _invAdj_index_table[2*toGate+1] = accum;
  }

  // Read GateType according to new index
  _gate_type.resize(_sum_pi_gates_pos);
  for (int i = 0; i < _num_PIs; i++) {
    int newGateIdx = _newIndex[i];
    _gate_type[newGateIdx] = GateType::PI;
  }
  for (int i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    int newGateIdx = _newIndex[i];
    _gate_type[newGateIdx] = GateType::PO;
  }
  for (int i = (_num_PIs + _num_POs); i < (_sum_pi_gates_pos); i++) {
    int type;
    ckt >> type;
    int newGateIdx = _newIndex[i];
    _gate_type[newGateIdx] = static_cast<GateType>(type);
  }

  // _print_read_graph();
  
#ifdef MA_PAR
  // printf("_max_level = %d\n", _max_level);
  _MA_simu_preparation();
#endif 
}

void CUDAMAPartitioner::_read_fault(std::istream &flst) {
  flst >> _num_fault;
  _faults.resize(_num_fault);

  for (int i = 0; i < _num_fault; i++) {
    int wrong_gate;
    size_t fault_value;
    flst >> wrong_gate >> fault_value;

    _faults[i]._gate_with_fault = wrong_gate;
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<int>::max() : (0);
  }
}

void CUDAMAPartitioner::_read_pattern(std::istream &ptn) {
  ptn >> _num_pattern;

  _num_rounds = (_num_pattern + UINT32T_BITS - 1) / UINT32T_BITS; // ceiling
  _num_rounds = (_sum_pi_gates_pos > 40) ? (_num_rounds/2) : (_num_rounds); // 這行只是為了從 size_t -> uint32_t
  // printf("_num_rounds = %lu, _num_pattern = %d, gates = %d\n", _num_rounds, _num_pattern, _sum_pi_gates_pos);

  _patterns.resize(_num_rounds);
  for (size_t i = 0; i < _num_rounds; i++) {
    _patterns[i]._value.resize(_num_PIs);
    for (int pi = 0; pi < _num_PIs; pi++) {
      int idx = i;
      ptn >> _patterns[idx]._value[pi];
    }
  }
}

void CUDAMAPartitioner::_construct_num_CLs_weight_tables() {
  // prepare _num_CLs
  _num_CLs = (_sum_pi_gates_pos + _NUM_GATES_PER_CL - 1)/_NUM_GATES_PER_CL;

  // prepare _weight_table
  _weight_table.resize(_num_CLs*_total_num_levels, 0);
  
  // init _weight_table
  for (int gateIdx = 0; gateIdx < _sum_pi_gates_pos; gateIdx++) {
    int num_output_gates = _adj_index_table[2*gateIdx+1] - _adj_index_table[2*gateIdx+0];
    // printf("gateIdx = %d, num_output_gates = %d\n", gateIdx, num_output_gates);
    int which_CL = getCLIdx(gateIdx); 
    _weight_table[which_CL] += num_output_gates;
  }

  // update _weight_table
  int fin_level = 0, acc_g = 0;
  bool level_update = false; 
  // TODO: this for loop can be run in parallel 
  for (int gateIdx = 0; gateIdx < _sum_pi_gates_pos; gateIdx++) {
    // printf("gateIdx = %d\n", gateIdx);
    // reduce 
    for (int fmGate = _invAdj_index_table[2*gateIdx+0]; fmGate < _invAdj_index_table[2*gateIdx+1]; fmGate++) {
      int fmGateIdx = _invAdj[fmGate];
      int fmGateCLIdx = getCLIdx(fmGateIdx); 
      _weight_table[fin_level*_num_CLs+fmGateCLIdx]--;
      // printf("\t\tfmGateIdx = %d, fmGateCLIdx = %d (limit: %d), _weight_table[%d] = %d\n", 
      //         fmGateIdx, fmGateCLIdx, _num_CLs, 
      //         fin_level*_num_CLs+fmGateCLIdx, _weight_table[fin_level*_num_CLs+fmGateCLIdx]);
    }
    acc_g++; // accumulate one finished gate in this fin_level

    level_update = ((acc_g == _numGates_per_level[fin_level]) && (fin_level < (_total_num_levels-1)))
                  ? (true) : (false);
    if (level_update) {
      // printf("level_update!\n");
      fin_level++; // update fin_level 
      for (int i = 0; i < _num_CLs; i++) { // init the weight table for the next level 
        // printf("\t\tupdate _weight_table[%d]\n", fin_level*_num_CLs+i);
        _weight_table[fin_level*_num_CLs+i] = _weight_table[(fin_level-1)*_num_CLs+i];
      }
      level_update = false; // re-init level_update back to false
      acc_g = 0; // re-init back to zero
      // printf("\n\n");
    }
  }

  // // print weight table
  // printf("\n\n_sum_pi_gates_pos = %d, _weight_table = [\n", _sum_pi_gates_pos);
  // for (int l = 0; l < _total_num_levels; l++) {
  //   printf("\tlevel_%d: ", l);
  //   for (int i = l*_num_CLs; i < (l+1)*_num_CLs; i++) {
  //     printf("%d, ", _weight_table[i]);
  //   }
  //   printf("\n");
  // }
  // printf("]\n");

  // // test the correctness of weight table 
  // bool all_zeros = true; 
  // for (int l = _total_num_levels-1; l < _total_num_levels; l++) {
  //   for (int i = l*_num_CLs; i < (l+1)*_num_CLs; i++) {
  //     all_zeros = (_weight_table[i] == 0) ? (true) : (false);
  //   }
  // }
  // if (!all_zeros) {
  //   printf("QQ FAIL ALL ZERO!!!\n");
  // } else {
  //   printf("PASS ALL ZERO!!!\n");
  // }
}

void CUDAMAPartitioner::_needed_st_ld_CLs_per_level(std::vector<bool> &POs_in_this_CL) {
  _needed_CLs_per_level.resize(_total_num_levels);
  int acc_g = 0;
  for (int l = 0; l < _total_num_levels; l++) {
    // printf("level = %d\n", l);
    for (int g = 0; g < _numGates_per_level[l]; g++) {
      int gateIdx = acc_g+g;
      int which_CL = getCLIdx(gateIdx);
      _needed_CLs_per_level[l].insert(which_CL);
      // printf("\tlevel %d, gateIdx = %d, insert CL_%d\n", 
      //         l, gateIdx, which_CL);
      for (int fmGate = _invAdj_index_table[2*gateIdx+0]; 
               fmGate < _invAdj_index_table[2*gateIdx+1]; 
               fmGate++) {
        int fmGateIdx = _invAdj[fmGate];
        which_CL = getCLIdx(fmGateIdx);
        _needed_CLs_per_level[l].insert(which_CL);
        // printf("\tlevel %d, gateIdx = %d, fmGateIdx = %d, insert CL_%d\n", 
        //         l, gateIdx, fmGateIdx, which_CL);
      }
    }
    acc_g += _numGates_per_level[l];
  } 

  // printf("_numGates_per_level:\n");
  // int acc = 0;
  // for (size_t i = 0; i < _numGates_per_level.size(); i++) {
  //   acc += _numGates_per_level[i];
  //   printf("\tlevel %lu: _numGates_per_level[%lu] = %d (acc = %d)\n", i, i, _numGates_per_level[i], acc);
  // }

  assert(_needed_CLs_per_level.size() == _total_num_levels && "_needed_CLs_per_level.size() != _total_num_levels");
  // printf("\n\n_num_CLs = %d\n", _num_CLs);


  // printf("\n\nst_ld_CLs:\n");
  // for (int l = 0; l < _total_num_levels; l++) {
  //   printf("_needed_CLs_per_level[%d].size() = %lu\n", l, _needed_CLs_per_level[l].size());
  // }

  // for (int l = 0; l < _total_num_levels; l++) {
  //   printf("\tlevel %d CLs: \n\t\t", l);
  //   std::cout << "Iterating needed_CLs_per_level_l: ";
  //   for (const int &val : _needed_CLs_per_level[l]) {
  //       std::cout << val << " ";
  //   }
  //   std::cout << std::endl;
  // }

  POs_in_this_CL.resize(_num_CLs, false);
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    if (_gate_type[i] == GateType::PO) {
      int which_CL = getCLIdx(i);
      POs_in_this_CL[which_CL] = true; 
    }
  }

  // for (size_t i = 0; i < POs_in_this_CL.size(); i++) {
  //   if (POs_in_this_CL[i]) {
  //     printf("POs_in_this_CL[%lu] = true\n", i);
  //   }
  // }
}

void CUDAMAPartitioner::_sorting(const int l, 
                                const std::set<int> &needed_CLs_per_level_l, 
                                const std::set<int> &cache_set,
                                std::set<int> &erase_set, std::set<int> &insert_set) {
  // step 0. 取聯集                                
  std::set<int> unionSet;
  // printf("needed_CLs_per_level_l.size() = %lu %lu\n", needed_CLs_per_level_l.size(), cache_set.size());
  // std::cout << "Iterating needed_CLs_per_level_l: ";
  // for (const int &val : needed_CLs_per_level_l) {
  //     std::cout << val << " ";
  // }
  // std::cout << std::endl;
    
  // std::cout << "Iterating cache_set: ";
  // for (const int &val : cache_set) {
  //     std::cout << val << " ";
  // }
  // std::cout << std::endl;
  
  std::set_union(needed_CLs_per_level_l.begin(), needed_CLs_per_level_l.end(),
                cache_set.begin(), cache_set.end(),
                std::inserter(unionSet, unionSet.begin()));

  // step 1: 構建 (值, index) pair
  std::vector<std::pair<int, int>> indexed_cache;
  indexed_cache.reserve(unionSet.size());  

  // printf("_weight_table[%d] = [", l); 
  // for (int i = 0; i < 256; i++) {
  //   printf("%d, ", _weight_table[l*_num_CLs+i]); 
  // }
  // printf("]\n"); 

  for (const int CLIdx : unionSet) {
    int CL_weight = _weight_table[l*_num_CLs+CLIdx];
    indexed_cache.emplace_back(CL_weight, CLIdx);
    // printf("push in CLIdx = %d, CL_weight = %d\n", CLIdx, CL_weight);
  }  

  // // Print each pair (value, index)
  // printf("\tindexed_cache: ");
  // for (const auto& p : indexed_cache) {
  //   std::cout << "(" << p.first << ", " << p.second << ") ";
  // }
  // std::cout << std::endl << std::endl;

  // step 2. sort from large to small 
  std::sort(
    indexed_cache.begin(), indexed_cache.end(),
    [](const auto &lhs, const auto &rhs) {
      return lhs.first > rhs.first; // Sort based on the first element of the pair
    });
  std::set<int> remin_set;
  assert(indexed_cache.size() >= _NUM_CLS_PER_BLOCK && "assert(indexed_cache.size() >= _NUM_CLS_PER_BLOCK");
  for (size_t i = 0; i < _NUM_CLS_PER_BLOCK; i++) { // 留下 256 個需要被留下的
    remin_set.insert(indexed_cache[i].second);
  }

  // // Print each pair (value, index)
  // printf("\tindexed_cache: ");
  // for (const auto& p : indexed_cache) {
  //   std::cout << "(" << p.first << ", " << p.second << ") ";
  // }
  // std::cout << std::endl << std::endl;
  
  // printf("\tremin_set: ");
  // for (const int& p : remin_set) {
  //   std::cout << p << ", ";
  // }
  // std::cout << std::endl << std::endl;

  // step 3. 取兩種差集
  // Compute A - B (elements in a but not in b)
  std::set_difference(cache_set.begin(), cache_set.end(), remin_set.begin(), remin_set.end(),
                      std::inserter(erase_set, erase_set.begin()));

  // Compute B - A (elements in b but not in a)
  std::set_difference(remin_set.begin(), remin_set.end(), cache_set.begin(), cache_set.end(),
                      std::inserter(insert_set, insert_set.begin()));

  assert(erase_set.size() == insert_set.size() && "erase_set.size() == insert_set.size()");

  // printf("\terase_set (sz = %lu): ", erase_set.size());
  // for (const int& p : erase_set) {
  //   std::cout << p << ", ";
  // }
  // std::cout << std::endl << std::endl;

  // printf("\tinsert_set (sz = %lu): ", insert_set.size());
  // for (const int& p : insert_set) {
  //   std::cout << p << ", ";
  // }
  // std::cout << std::endl << std::endl;
}

void CUDAMAPartitioner::update_st(const int l, std::vector<bool> &POs_in_this_CL,
                                  const std::set<int> &erase_set, 
                                  const std::vector<int> &cache_index)  {
  _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size()); // starts 

  // weight table != 0 才要寫回去
  for (const int& CLIdx : erase_set) {
    auto it = std::find(cache_index.begin(), cache_index.end(), CLIdx);
    assert(it != cache_index.end() && "it != cache_index.end()"); // 保險起見放的而已
    int posIdx = std::distance(cache_index.begin(), it); // 要被踢掉的 CLIdx 在 cache_index 中的位置
    if ((_weight_table[l*_num_CLs+CLIdx] == 0) && (POs_in_this_CL[CLIdx] == false)) {
      // update _st_ld_positi
      _st_ld_positi.push_back(posIdx);
      // update _st_ld_CLIdxs
      _st_ld_CLIdxs.push_back(-1);       
      // printf("(X) No need to WB posIdx = %d, CLIdx = %d\n", posIdx, CLIdx);
    } else {
      // update _st_ld_positi
      _st_ld_positi.push_back(posIdx);
      // update _st_ld_CLIdxs
      _st_ld_CLIdxs.push_back(CLIdx); 
      // printf("(V) Need to WB posIdx = %d, CLIdx = %d\n", posIdx, CLIdx);
    }
  }
  _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size()); // ends
}

void CUDAMAPartitioner::update_ld(const int l, 
                                  const std::set<int> &insert_set, 
                                  std::vector<int> &cache_index) {
  _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size()); // starts 

    assert((_st_ld_CLs_index_table[4*l+1]-_st_ld_CLs_index_table[4*l+0]) == insert_set.size() 
          && "(_st_ld_CLs_index_table[4*l+1]-_st_ld_CLs_index_table[4*l+0]) == insert_set.size() ");
    const int starts = _st_ld_CLs_index_table[4*l+0]; int i = 0;
    for (const int &newCLIdx : insert_set) {
      int posIdx = _st_ld_positi[starts+i]; i++; 
      // update _st_ld_positi
      _st_ld_positi.push_back(posIdx);
      // update _st_ld_CLIdxs
      _st_ld_CLIdxs.push_back(newCLIdx); 
      // printf("posIdx = %d, newCLIdx = %d\n", posIdx, newCLIdx);
      // update cache_index
      cache_index[posIdx] = newCLIdx;
    }

  _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size()); // ends
}

void CUDAMAPartitioner::update_cache_set(std::set<int> &erase_set, 
                                        std::set<int> &insert_set, 
                                        std::set<int> &cache_set) {
  for (const int&clIdx : erase_set) {
    cache_set.erase(clIdx);
  }
  for (const int&clIdx : insert_set) {
    cache_set.insert(clIdx);
  }
}

void CUDAMAPartitioner::_st_ld_CLs_per_level(std::vector<bool> &POs_in_this_CL) {
  // init 
  // _g_c_table.resize(_total_num_levels*(_NUM_CLS_PER_BLOCK/32)); 
  // _position_table.resize(_total_num_levels*(_NUM_CLS_PER_BLOCK/4)); 

  std::set<int> cache_set; // 紀錄 cahce 當下有個 CLs，用 sets 取聯集和差集比較快
  std::vector<int> cache_index; 
  
  // 一開始先把 256 條都放進去
  for (int i = 0; i < _NUM_CLS_PER_BLOCK; i++) {
    cache_set.insert(i);
    cache_index.push_back(i); 
  }

  int diffSetSz = 0;
  int level = 0;
  while (diffSetSz == 0 && level < _total_num_levels) {
    std::set<int> diffSet;
    std::set_difference(_needed_CLs_per_level[level].begin(), _needed_CLs_per_level[level].end(),
                        cache_set.begin(), cache_set.end(),
                        std::inserter(diffSet, diffSet.begin()));
    diffSetSz = diffSet.size();
    // printf("level %d PASS, diffSet.size() = %lu (_needed_CLs_per_level[%d].size() = %lu)\n", 
    //       level, diffSet.size(), level, _needed_CLs_per_level[level].size());
    if (diffSetSz == 0) {
      level++;
    }
  } 
  printf("level = %d\n", level);

  bool all_in_one_cache = (level == _total_num_levels);
  if (all_in_one_cache) { // 所有東西都能被塞進 cache line 裡面
    // printf("level = %d, _total_num_levels = %d, _num_CLs = %d\n", level, _total_num_levels, _num_CLs);
    
    // init _st_ld_CLs_index_table: level 0
    // st: 
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    // ld: 
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    assert(_num_CLs <= _NUM_CLS_PER_BLOCK && "assert(_num_CLs <= _NUM_CLS_PER_BLOCK");
    int CLs = (_num_CLs < _NUM_CLS_PER_BLOCK) ? (_num_CLs) : (_NUM_CLS_PER_BLOCK);
    for (int i = 0; i < CLs; i++) {
      _st_ld_CLIdxs.push_back(i);
      _st_ld_positi.push_back(i);
    }
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());

    for (int l = 1; l < (level-1); l++) {
      // st
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      // ld
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());    
    }

    // last level:  need to WB
    // st
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    for (int i = 0; i < CLs; i++) {
      if (POs_in_this_CL[i]) {
        _st_ld_CLIdxs.push_back(i);
        _st_ld_positi.push_back(i);        
      }
    }
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    // ld
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());    
  } else {
    // init _st_ld_CLs_index_table: level 0
    // st: 
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    // ld: 
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
    for (int i = 0; i < _NUM_CLS_PER_BLOCK; i++) {
      _st_ld_CLIdxs.push_back(i);
      _st_ld_positi.push_back(i);
    }
    _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());

    for (int l = 1; l < level; l++) {
      // st
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      // ld
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());
      _st_ld_CLs_index_table.push_back(_st_ld_CLIdxs.size());    
    }

    // printf("!?level = %d\n", level);
    for (int l = level; l < _total_num_levels; l++) {
      // printf("level %d, with cache_set.size() = %lu, _needed_CLs_per_level[%d].size() = %lu\n", l, cache_set.size(), l, _needed_CLs_per_level[l].size());
      std::set<int> erase_set;
      std::set<int> insert_set;
      _sorting(l, _needed_CLs_per_level[l], cache_set, erase_set, insert_set);
      // printf("\nst-st-st-st-st-st-st-st-st-st-st-st-st-st-st-st-st-st-st\n");
      update_st(l, POs_in_this_CL, erase_set, cache_index);
      // printf("\nld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld-ld\n");
      update_ld(l, insert_set, cache_index);
      update_cache_set(erase_set, insert_set, cache_set);
      // printf("After update_cache_set: cache_set.size() = %lu\n", cache_set.size());
      assert(cache_set.size() == 256 && "assert(cache_set.size() == 256");
    }
  }
    
  // std::vector<int> &a = _st_ld_CLs_index_table;
  // printf("_st_ld_CLIdxs (sz = %lu) = [\n", a.size());
  // for (size_t i = 0; i < a.size()/4; i++) {
  //   // st
  //   printf("\tlevel %lu st: ", i);
  //   for (int j = a[4*i+0]; j < a[4*i+1]; j++) {
  //     printf("(pos, CLIdx) = (%d, %d), ", _st_ld_positi[j], _st_ld_CLIdxs[j]);
  //   }
  //   printf("\n");
  //   // ld 
  //   printf("\tlevel %lu ld: ", i);
  //   for (int j = a[4*i+2]; j < a[4*i+3]; j++) {
  //     printf("(pos, CLIdx) = (%d, %d), ", _st_ld_positi[j], _st_ld_CLIdxs[j]);
  //   }
  //   printf("\n");
  // }
  // printf("]\n");
}

void CUDAMAPartitioner::_update_st_ld_CLs_per_level() {
  
  std::vector<bool> POs_in_this_CL;
  _needed_st_ld_CLs_per_level(POs_in_this_CL);
  
  // 這裡要確定每一層需要重新 st/ld 哪些 CLs
  // for each level: 1. st start, 2. st end, 3. ld start, 4. ld end
  _st_ld_CLs_per_level(POs_in_this_CL); 
}

void CUDAMAPartitioner::_MA_simu_preparation() {
  _construct_num_CLs_weight_tables();
  _update_st_ld_CLs_per_level();
}




/* Prepare for GPU simulation */
void CUDAMAPartitioner::_levelized(std::vector<std::vector<int>> &adj, 
                                   std::vector<std::vector<int>> &invAdj) {
  // Kahn's algorithm, topological sort
  std::vector<int> indegree; 
  
  for (int fromGate = 0; fromGate < _sum_pi_gates_pos; fromGate++) {
    indegree.push_back(invAdj[fromGate].size());
  }

  std::queue<int> source;
  std::vector<int> order;

  // Since only PIs will be indegree == 0
  for (int i = 0; i < _num_PIs; i++) {
    source.push(i);
  }

  while (!source.empty()) {
    int first = source.front();
    source.pop();
    order.push_back(first);
    for (size_t i = 0; i < adj[first].size(); i++) {
      int toGateIdx = adj[first][i];
      indegree[toGateIdx]--;
      if (indegree[toGateIdx] == 0) {
        source.push(toGateIdx);
      }
    }
  }
  
  // ------- levelization -------
  _level_of_gates.resize(_sum_pi_gates_pos, 0);
  _max_level = 0;
  
  for (size_t i = 0; i < order.size(); i++) {
    int gateIdx = order[i];
    
    // traverse its inputs 
    for (size_t fmGate = 0; fmGate < invAdj[gateIdx].size(); fmGate++) {
      int fmGateIdx = invAdj[gateIdx][fmGate];
      int new_level = _level_of_gates[fmGateIdx]+1;
      
      // Update _level_of_gates[gateIdx]
      _level_of_gates[gateIdx] = (new_level > _level_of_gates[gateIdx]) ? 
                                (new_level) : 
                                (_level_of_gates[gateIdx]);
      _max_level = (_level_of_gates[gateIdx] > _max_level) ? 
                  (_level_of_gates[gateIdx]) : 
                  (_max_level);
    }
  } 
  _total_num_levels = _max_level + 1;

  // re-index based on the results of topological sort
  _newIndex.resize(_sum_pi_gates_pos);
  for (size_t i = 0; i < order.size(); i++) {
    _newIndex[order[i]] = i;
  } 
}

void CUDAMAPartitioner::_ask_gpu_simulation_memory() {
  // Memory allocation
  cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, _sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_patterns_gpu, _num_rounds*_num_PIs*sizeof(uint32_t));
  cudaMalloc((void**)&_fault_gate_idx_gpu, _num_fault*sizeof(int));
  cudaMalloc((void**)&_fault_SA_fault_val_gpu, _num_fault*sizeof(size_t));
  
  cudaMalloc((void**)&_pi_gate_po_output_res_gpu, _sum_pi_gates_pos*sizeof(uint32_t));
  cudaMalloc((void**)&_numGates_per_level_gpu, _numGates_per_level.size()*sizeof(int));
  cudaCheckErrors("CUDA: ask gpu simulation memory - cudaMalloc - Failure");

  cudaMemcpyAsync(_numGates_per_level_gpu, _numGates_per_level.data(), 
                  _numGates_per_level.size()*sizeof(int), cudaMemcpyHostToDevice); 
  cudaCheckErrors("CUDA: ask gpu simulation memory - cudaMemcpyAsync - Failure");
}

void CUDAMAPartitioner::_move_GateType_h2d() {
  std::vector<int>pi_gate_po_gate_type;

  for (int g = 0; g < _sum_pi_gates_pos; g++) {
    pi_gate_po_gate_type.push_back(static_cast<int>(_gate_type[g]));
  }

  cudaMemcpyAsync(_pi_gate_po_gate_type_gpu, pi_gate_po_gate_type.data(), 
                _sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _pi_gate_po_gate_type_gpu cudaMemcpy failure");

#ifdef GPU_PREPARE_SIMULATION_PRINT_CHECK
  cudaDeviceSynchronize();
  print_pi_gate_po_gate_type_gpu <<< 1, 1 >>> (_pi_gate_po_gate_type_gpu, _sum_pi_gates_pos);
  cudaCheckErrors("CUDA: print_pi_gate_po_gate_type_gpu failure");
  cudaDeviceSynchronize();
#endif 
}


void CUDAMAPartitioner::_move_patterns_h2d() {
  std::vector<uint32_t> patterns_cpu;
  for (size_t i = 0; i < _num_rounds; i++) {
    for (int pi = 0; pi < _num_PIs; pi++) {
      patterns_cpu.push_back(_patterns[i]._value[pi]);
      // printf("_patterns[%lu]._value[%d] = %u\n", i, pi, _patterns[i]._value[pi]);
    }
  }

  cudaMemcpy(_patterns_gpu, patterns_cpu.data(), 
            (_num_rounds*_num_PIs)*sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _patterns_gpu cudaMemcpy failure");

#ifdef GPU_PREPARE_SIMULATION_PRINT_CHECK
  cudaDeviceSynchronize(); 
  print_patterns_gpu <<< 1, 1 >>> (_patterns_gpu, _num_rounds, _num_pattern);
  cudaCheckErrors("CUDA: print_patterns_gpu failure");
  cudaDeviceSynchronize(); 
#endif
}

void CUDAMAPartitioner::_move_faults___h2d() {
  std::vector<int> fault_gate_idx;
  std::vector<size_t> fault_SA_fault_val;

  for (int i = 0; i < _num_fault; i++) {
    fault_gate_idx.push_back(_faults[i]._gate_with_fault);
    fault_SA_fault_val.push_back(_faults[i]._gate_SA_fault_val);
  }

  cudaMemcpy(_fault_gate_idx_gpu, fault_gate_idx.data(), _num_fault*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_fault_SA_fault_val_gpu, fault_SA_fault_val.data(), _num_fault*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _fault_gate_idx_gpu OR _fault_SA_fault_val_gpu cudaMemcpy failure");

#ifdef GPU_PREPARE_SIMULATION_PRINT_CHECK
  cudaDeviceSynchronize();
  print_fault_gate_idx_gpu <<< 1, 1 >>> (_fault_gate_idx_gpu, _fault_SA_fault_val_gpu, _num_fault);
  cudaCheckErrors("CUDA: print_patterns_gpu failure");
  cudaDeviceSynchronize();
#endif
}

void CUDAMAPartitioner::_move_st_ld____h2d() {
  cudaMalloc((void**)&_st_ld_CLIdxs_gpu, _st_ld_CLIdxs.size()*sizeof(int));
  cudaMalloc((void**)&_st_ld_positi_gpu, _st_ld_positi.size()*sizeof(int));
  cudaMalloc((void**)&_st_ld_CLs_index_table_gpu, _st_ld_CLs_index_table.size()*sizeof(int));
  // cudaMalloc((void**)&_g_c_table, ((_num_CLs + 32 - 1)/32)*sizeof(uint32_t));
  // cudaMalloc((void**)&_position_table, ((_num_CLs + 8 - 1)/8)*sizeof(uint32_t));
  // printf("_num_CLs = %d, _g_c_table size = %d, _position_table size = %d\n", _num_CLs, ((_num_CLs + 32 - 1)/32), (_num_CLs + 8 - 1)/8);
  cudaCheckErrors("CUDA: Asking _move_st_ld____h2d cudaMalloc - Failure");

  cudaMemcpyAsync(_st_ld_CLIdxs_gpu, _st_ld_CLIdxs.data(), 
                  _st_ld_CLIdxs.size()*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_st_ld_positi_gpu, _st_ld_positi.data(), 
                  _st_ld_positi.size()*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_st_ld_CLs_index_table_gpu, _st_ld_CLs_index_table.data(), 
                  _st_ld_CLs_index_table.size()*sizeof(int), cudaMemcpyHostToDevice); 
  cudaCheckErrors("CUDA: Asking _move_st_ld____h2d cudaMemcpyAsync - Failure");
}

void CUDAMAPartitioner::prepare_gpu_simulation() {
  auto start = std::chrono::steady_clock::now();
  
  // Copy data from Host to Device for GPU simulation 
  cudaMalloc((void**)&_adj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_invAdj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_adj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_invAdj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaCheckErrors("CUDA: Asking partitioner memory - Failure");

  cudaMemcpyAsync(_adj_gpu, _adj, _szOfAdj*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_invAdj_gpu, _invAdj, _szOfAdj*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_adj_index_table_gpu, _adj_index_table, 
                  2*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_invAdj_index_table_gpu, _invAdj_index_table, 
                  2*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice); 
  _ask_gpu_simulation_memory();

  _move_GateType_h2d();
  _move_patterns_h2d();
  _move_faults___h2d();

#ifdef MA_PAR
  _move_st_ld____h2d();
#endif 

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  std::cout << "prepare_GPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";
}


// Simulation functions 
void CUDAMAPartitioner::run(const size_t num_threads, const size_t NUM_SIMULATION_RDS) {  
  GALPS_MA_GPUSimulator gpuSimulator; 
  gpuSimulator.run_gpu_simulator_DSP_gpu(_num_PIs, _num_inner_gates, _num_POs, 
                                          _sum_pi_gates_pos, 
                                          _num_pattern, _num_rounds, _num_fault,
                                          _pi_gate_po_gate_type_gpu, 
                                          _patterns_gpu,
                                          _fault_gate_idx_gpu,
                                          _fault_SA_fault_val_gpu,
                                          _pi_gate_po_output_res_gpu,
                                          _numGates_per_level,
                                          _numGates_per_level_gpu,
                                          _total_num_levels,
                                          _invAdj_gpu,
                                          _invAdj_index_table_gpu,
                                          _patterns, 
                                          NUM_SIMULATION_RDS);
  cudaDeviceSynchronize();
}

void CUDAMAPartitioner::run_MA(const size_t num_threads, const size_t NUM_SIMULATION_RDS) {  
  GALPS_MA_GPUSimulator gpuSimulator;
  gpuSimulator.run_gpu_simulator_MA_gpu(_num_PIs, _num_inner_gates, _num_POs, 
                                        _sum_pi_gates_pos, 
                                        _num_pattern, _num_rounds, _num_fault,
                                        _pi_gate_po_gate_type_gpu, 
                                        _patterns_gpu,
                                        _fault_gate_idx_gpu,
                                        _fault_SA_fault_val_gpu,
                                        _pi_gate_po_output_res_gpu,
                                        _numGates_per_level,
                                        _numGates_per_level_gpu,
                                        _total_num_levels,
                                        _invAdj_gpu,
                                        _invAdj_index_table_gpu,
                                        _patterns, 
                                        NUM_SIMULATION_RDS, 
                                        _st_ld_CLIdxs_gpu, 
                                        _st_ld_positi_gpu, 
                                        _st_ld_CLs_index_table_gpu);
  cudaDeviceSynchronize();
}







// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------
// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------
// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------
// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------
// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------
void CUDAMAPartitioner::_print_patterns(const std::vector<Pattern> &patterns,
                                        const int round,
                                        const int num_PIs) const {
  std::cout << "\n=====\n\n";
    for (int i = 0; i < round; i++) {
      std::cout << "[" << UINT32T_BITS * i << ", " << UINT32T_BITS * (i + 1)
                << "] bits = [\n";
      for (int j = 0; j < num_PIs; j++) {
        _print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
      }
    std::cout << "]\n";
  }
}

void CUDAMAPartitioner::_print_bits_stack(const int size,
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

// Function to convert GateType to gpuGateType
gpuGateType CUDAMAPartitioner::_convertGateTypeToGpu(GateType gate_type) {
  switch (gate_type) {
    case GateType::INV: return gpuGateType::INV; // 0
    case GateType::AND: return gpuGateType::AND; // 1
    case GateType::OR: return gpuGateType::OR;
    case GateType::XOR: return gpuGateType::XOR;
    case GateType::NAND: return gpuGateType::NAND;
    case GateType::NOR: return gpuGateType::NOR;
    case GateType::XNOR: return gpuGateType::XNOR;
    case GateType::MUX: return gpuGateType::MUX;
    case GateType::CLKBUF: return gpuGateType::CLKBUF;
    case GateType::PI: return gpuGateType::PI;
    case GateType::PO: return gpuGateType::PO;
    default: return gpuGateType::MAX_GATE_TYPE; // Handle default case
  }
}

// Function to convert GateType to string
std::string CUDAMAPartitioner::_gateTypeToString(GateType type) const {
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

void CUDAMAPartitioner::_print_read_graph() const {
  printf("Get inside _print_read_graph:\n");

  printf("_sum_pi_gates_pos = %d, _szOfAdj = %d\n", 
          _sum_pi_gates_pos, _szOfAdj);

  printf("_gate_type:\n");
  for (size_t i = 0; i < _gate_type.size(); i++) {
    std::cout << "Gate_" << i << ", _gate_type = " << _gateTypeToString(_gate_type[i]) << "\n";
  } printf("\n");
  
  printf("\n---------\n");

  printf("_adj:\n");
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%d's output = ", i);
    for (int j = _adj_index_table[2*i+0]; 
                j < _adj_index_table[2*i+1]; 
                j++) {
      printf("%d, ", _adj[j]);
    }
    printf("\n");
  }
  printf("\n");  
  
  printf("\n---------\n");
  
  printf("_invAdj:\n");
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%d's output = ", i);
    for (int j = _invAdj_index_table[2*i+0]; 
                j < _invAdj_index_table[2*i+1]; 
                j++) {
      printf("%d, ", _invAdj[j]);
    }
    printf("\n");
  }
  printf("\n");  

  for (size_t i = 0; i < _numGates_per_level.size(); i++) {
    printf("level%lu: %d gates\n", i, _numGates_per_level[i]);
  }
  printf("===============================\n");
}

void CUDAMAPartitioner::_print_levelized(){
  printf("_level_of_gates:\n");
  for (size_t i = 0; i < _level_of_gates.size(); i++) {
    printf("_level_of_gates[%lu] = %d\n", i, _level_of_gates[i]);
  }
  printf("\n");
}


__global__ void print_patterns_gpu (uint32_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs) {
  for (size_t i = 0; i < _num_rounds; i++) {
    printf("round %lu: ", i);
    for (size_t j = 0; j < _num_PIs; j++) {
      uint32_t p = _patterns_gpu[_num_PIs*i+j];
      printf("%u, ", p);
    }
    printf("\n");
  }
  printf("\n");
}
