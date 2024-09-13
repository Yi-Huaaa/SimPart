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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "simulation.cuh"

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


// CUDASimulator fucntions
void CUDASimulator::read(const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path);

  if (!ckt) {
    throw std::runtime_error("cannot open circut file "s + ckt_path);
  }
  if (!flst) {
    throw std::runtime_error("cannot open fault file "s + flst_path);
  }
  if (!ptn) {
    throw std::runtime_error("cannot open pattern file "s + ptn_path);
  }

  read(ckt, flst, ptn);
}

void CUDASimulator::read(std::istream &ckt, std::istream &flst, std::istream &ptn) {
  // gate
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires;
  #ifdef FSIM_DEBUG_PRINT_GRAPH
    std::cout << "Read and construct ckt" << std::endl;
    std::cout << _num_PIs << ", " << _num_POs << ", " << _num_inner_gates << ", "
  << _num_wires << std::endl;
  #endif
  _read_graph(ckt);

  // fault
  flst >> _num_faults;
  #ifdef FSIM_DEBUG_PRINT_FAULTS
    std::cout << "Read and construct flst" << std::endl;
    std::cout << _num_faults << std::endl;
  #endif
  _read_fault(flst);

  // pattern
  ptn >> _num_pattern;
  #ifdef FSIM_DEBUG_PRINT_PATTERNS
    std::cout << "Read and construct ptn" << std::endl;
    std::cout << _num_pattern << std::endl;
  #endif
  _read_pattern(ptn);
}

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


// Construct functions
void CUDASimulator::_read_graph(std::istream &ckt) {
  size_t sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  // for topological sort
  std::vector<std::vector<size_t>> adj;
  adj.resize(sum_pi_gates_pos);
  std::vector<size_t> indegree;
  indegree.resize(sum_pi_gates_pos);

  // map table for recording gate input _order
  // record for each to_gate, its from_gates's are which _order (#_inputs, A/A1, B/A2, A3/S, A4)
  std::vector<std::vector<size_t>> gate_input_order_table; 
  gate_input_order_table.resize(sum_pi_gates_pos);
  for (size_t i = 0; i < sum_pi_gates_pos; i++) {
    gate_input_order_table[i].resize(5);
  }

  /** Levelized the input gates before constructing the circuit graph */
  // read file, construct adjacent list, construct indegree array
  for (size_t i = 0; i < _num_wires; i++) {
    size_t gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (size_t j = 0; j < num_post_gates; j++) {
      size_t gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      adj[gate_0].push_back(gate_tmp);
      indegree[gate_tmp]++;

      gate_input_order_table[gate_tmp][pin_tmp] = gate_0;
      gate_input_order_table[gate_tmp][0] ++; // calculate how many inputs for this to_gate
    }
  }

#ifdef FSIM_DEBUG_PRINT_GRAPH
  // copy `indegree` vector since we need its original value later
  std::vector<size_t> indegree_orig;
  indegree_orig = indegree;
#endif

  // Kahn's algorithm, topological sort
  std::queue<size_t> source;

  // since only PIs will be indegree == 0
  for (size_t i = 0; i < _num_PIs; i++) {
    source.push(i);
  }

  while (!source.empty()) {
    size_t first = source.front();
    source.pop(); // queue pop front element
    _order.push_back(first);
    for (size_t i = 0; i < adj[first].size(); i++) {
      indegree[adj[first][i]]--;
      if (indegree[adj[first][i]] == 0) {
        source.push(adj[first][i]);
      }
    }
  }
  
  // // make vector of _order's _order becomes [PIs, gates, POs] 
  // // remove the PO in the _order
  // // since  error: namespace "std" has no member "remove_if, therefore, we implement the 'remove_if' by ourselves 
  _order.erase(std::remove_if(_order.begin(), _order.end(),
                        [this](const size_t &x) {
                            return ((x >= _num_PIs) && (x < (_num_PIs + _num_POs)));
                        }),
              _order.end());

  for (size_t i = 0; i < _num_POs; i++) {
    _order.push_back(_num_PIs + i); // push_back PO's index
  }         

#ifdef FSIM_DEBUG_PRINT_GRAPH
  print_topological(adj, indegree_orig, _order);
  print_gate_input_order_table(gate_input_order_table);
#endif

  /** Constructing circuit graph */
  // read gate type array
  std::vector<GateType> gate_type;
  gate_type.resize(sum_pi_gates_pos);
  for (size_t i = 0; i < _num_PIs; i++) {
    gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs + _num_POs); i < (sum_pi_gates_pos); i++) {
    size_t t; // type
    ckt >> t;
    gate_type[i] = static_cast<GateType>(t);
  }

  // mapping gate_idx to gate_order_index
  /**
   * CML example:
   * _order[3] = 6
   * _gate_idx_to_tpg_order_idx[6] = 3
   * _gate_idx_to_tpg_order_idx: 0, 1, 2, 7, 5, 6, 3, 4
   * */
  _gate_idx_to_tpg_order_idx = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  memset(_gate_idx_to_tpg_order_idx, 0, sum_pi_gates_pos*sizeof(size_t));
  for (size_t i = 0; i < _order.size(); i++) {
    _gate_idx_to_tpg_order_idx[_order[i]] = i;
  }

  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);
  // PIs are stored in the _order specified by the `_order` array
  _PIs = (PI*)malloc(_num_PIs*sizeof(PI)); memset(_PIs, 0, _num_PIs*sizeof(PI));
  for (size_t i = 0; i < _num_PIs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = _order[i + pi_idx_accum];
    _PIs[i]._idx = i + pi_idx_accum;
    _PIs[i]._type = gate_type[gate_id];
    _PIs[i]._level = default_level;
    _PIs[i]._output_value = default_output_value;
    _PIs[i]._num_inputs = 0;
  }

  // gates are stored in the _order specified by the `_order` array
  _gates = (Gate *)malloc(_num_inner_gates*sizeof(Gate)); 
  memset(_gates, 0, _num_inner_gates*sizeof(Gate));
  for (size_t i = 0; i < _num_inner_gates; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = _order[i + gate_idx_accum];
    _gates[i]._idx = i + gate_idx_accum;
    _gates[i]._type = gate_type[gate_id];
    _gates[i]._level = default_level;
    _gates[i]._output_value = default_output_value;
  }

  // POs are stored in the _order specified by the `_order` array
  _POs = (PO *)malloc(_num_POs*sizeof(PO)); memset(_POs, 0, _num_POs*sizeof(PO));

  for (size_t i = 0; i < _num_POs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = _order[i + po_idx_accum];
    _POs[i]._idx = i + po_idx_accum;
    _POs[i]._type = gate_type[gate_id];
    _POs[i]._level = default_level;
    _POs[i]._output_value = default_output_value;
  }

  // give the input gates number - v2
  // ask memory -> note: remember to minus the gate_idx_accum (po_idx_accum)
  for (size_t i = _num_PIs; i < (sum_pi_gates_pos); i++) {
    if ( _gate_idx_to_tpg_order_idx[i] >= (_num_PIs+_num_inner_gates) ) { // PO
      size_t gate_idx = _gate_idx_to_tpg_order_idx[i]-po_idx_accum;
      size_t num_inputs = gate_input_order_table[i][0];
      _POs[gate_idx]._num_inputs = num_inputs;
      _POs[gate_idx]._inputs = (ElementBase **)malloc(num_inputs*sizeof(ElementBase*));
      memset(_POs[gate_idx]._inputs, 0, num_inputs*sizeof(ElementBase*));
    } else { // gate
      size_t gate_idx = _gate_idx_to_tpg_order_idx[i]-gate_idx_accum;
      size_t num_inputs = gate_input_order_table[i][0];
      _gates[gate_idx]._num_inputs = num_inputs;
      _gates[gate_idx]._inputs = (ElementBase **)malloc(num_inputs*sizeof(ElementBase*));
      memset(_gates[gate_idx]._inputs, 0, num_inputs*sizeof(ElementBase*));
    }
  }

  std::vector<size_t> index_accum;
  index_accum.resize(sum_pi_gates_pos, 0);
  size_t to_gate_idx;

  std::vector<std::vector<size_t>> adj_topol;     // adj after topological sort (reindex)
  std::vector<std::vector<size_t>> inv_adj_topol; // inv_adj after topological sort (reindex)
  adj_topol.resize(sum_pi_gates_pos);
  inv_adj_topol.resize(sum_pi_gates_pos);

  for (size_t i = _num_PIs; i < gate_input_order_table.size(); i++) {
    for (size_t j = 1; j < (gate_input_order_table[i][0]+1); j++) { // how many inputs for this gate
      size_t from_gate_id = gate_input_order_table[i][j];
      size_t to_gate_id = i;
      if (from_gate_id < _num_PIs) { // PI -> gate
        PI &from_gate =
            _PIs[_gate_idx_to_tpg_order_idx[from_gate_id] - pi_idx_accum];
        Gate &to_gate =
            _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
        to_gate_idx = to_gate._idx;
        // to_gate._inputs.push_back(&from_gate);
        to_gate._inputs[index_accum[to_gate_idx]] = &from_gate;
        index_accum[to_gate_idx]++;

        // add up adj; inv_adj
        // printf("from_gate._idx = %lu, to_gate._idx = %lu\n", from_gate._idx, to_gate._idx);
        adj_topol[from_gate._idx].push_back(to_gate._idx);
        inv_adj_topol[to_gate._idx].push_back(from_gate._idx);
      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        Gate &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        PO &to_gate =
            _POs[_gate_idx_to_tpg_order_idx[to_gate_id] - po_idx_accum];
        to_gate_idx = to_gate._idx;
        to_gate._inputs[index_accum[to_gate_idx]] = &from_gate;
        index_accum[to_gate_idx]++;
        
        // add up adj; inv_adj
        // printf("from_gate._idx = %lu, to_gate._idx = %lu\n", from_gate._idx, to_gate._idx);
        adj_topol[from_gate._idx].push_back(to_gate._idx);
        inv_adj_topol[to_gate._idx].push_back(from_gate._idx);
      } else { // gate -> gate
        Gate &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        Gate &to_gate =
            _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
        to_gate_idx = to_gate._idx;
        to_gate._inputs[index_accum[to_gate_idx]] = &from_gate;
        index_accum[to_gate_idx]++;

        // add up adj; inv_adj
        // printf("from_gate._idx = %lu, to_gate._idx = %lu\n", from_gate._idx, to_gate._idx);
        adj_topol[from_gate._idx].push_back(to_gate._idx);
        inv_adj_topol[to_gate._idx].push_back(from_gate._idx);
      }
    }
  }

  // // check adj_topol inv_adj_topol
  // size_t mmm = 0;
  // for (size_t i = 0; i < sum_pi_gates_pos; i++) {
  //   mmm = (adj_topol[i].size()>mmm) ? (adj_topol[i].size()) : (mmm);
  //   printf("adj_topol[%lu].size() = %lu\n", i, adj_topol[i].size());
  // } 
  // printf("mmm = %lu\n", mmm);

  // size_t aaa = 0;
  // for (size_t i = 0; i < _num_inner_gates; i++) {
  //   for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
  //     size_t in_gate = (_gates[i]._inputs[j])->_idx;
  //     if (in_gate == 0) {
  //       aaa ++;
  //     }
  //   }
  // }
  // printf("aaa = %lu\n", aaa);

  _sum_inputs_gates = 0;
  for (size_t i = 0; i < gate_input_order_table.size(); i++){
    _sum_inputs_gates += gate_input_order_table[i][0];
  }

  _gate_inputs_list = (size_t*)malloc(_sum_inputs_gates*sizeof(size_t));  // memset(input, 0, N*sizeof(float));
  _gate_inputs_list_start = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  memset(_gate_inputs_list, 0, _sum_inputs_gates*sizeof(size_t));
  memset(_gate_inputs_list_start, 0, sum_pi_gates_pos*sizeof(size_t));
  
  // construct _gate_inputs_list_start
  for (size_t i = 0; i < _num_inner_gates; i++) {
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
      _gate_inputs_list_start[i+1+gate_idx_accum] = _gates[i]._num_inputs + _gate_inputs_list_start[i+gate_idx_accum];
    }
  }
  for (size_t i = 0; i < (_num_POs-1); i++) { // since the last one no need to add
    for (size_t j = 0; j < _POs[i]._num_inputs; j++) {
      _gate_inputs_list_start[i+1+po_idx_accum] = _gate_inputs_list_start[i+po_idx_accum]+1;// PO only have 1 input
    }
  }

#ifdef FSIM_DEBUG_PRINT_GRAPH
  printf("_gate_inputs_list_start:\n");
  for (size_t i = 0; i < sum_pi_gates_pos; i++) {
    printf("_gate_inputs_list_start[%ld] = %ld\n", i, _gate_inputs_list_start[i]);
  }
#endif

  // construct '_gate_inputs_list'
  size_t idx_accum = 0;
  for (size_t i = 0; i < _num_inner_gates; i++) {
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
      _gate_inputs_list[idx_accum] = (_gates[i]._inputs[j])->_idx;
      idx_accum++;
    }
  }
  for (size_t i = 0; i < _num_POs; i++) {
    for (size_t j = 0; j < _POs[i]._num_inputs; j++) {
      _gate_inputs_list[idx_accum] = (_POs[i]._inputs[j])->_idx;
      idx_accum++;
    }
  }

#ifdef FSIM_DEBUG_PRINT_GRAPH
  printf("_gate_inputs_list:\n");
  for (size_t i = 0; i < _sum_inputs_gates; i++) {
    printf("%ld, ", _gate_inputs_list[i]);
  } printf("\n");
#endif  
  
  /** levelize */
  // levelize PIs -> set by the default_level (= 0), note: all PIs belond to the
  // level_0 , making the parallel in omp becomes easier levelize gates
  _max_level = 0;
  for (size_t i = 0; i < _num_inner_gates; i++) {
    size_t ret_level = ((_gates[i]._inputs[0])->_level) + 1;
    for (size_t j = 1; j < _gates[i]._num_inputs; j++) {
      size_t tmp_level = ((_gates[i]._inputs[j])->_level) + 1;
      if (ret_level < tmp_level) {
        ret_level = tmp_level;
      }
    }
    _gates[i]._level = ret_level;
    _max_level = (_max_level < ret_level) ? (ret_level) : (_max_level);
  }


  // levelize POs -> all POs belond to the level_max
  // , making the parallel in omp becomes easier
  for (size_t i = 0; i < _num_POs; i++) {
    _POs[i]._level = _max_level + 1;
  }
  _max_level++;

  // _num_gates_per_level.resize(_max_level + 1);
  // cudaMallocManaged(&_num_gates_per_level, (_max_level + 1)*sizeof(size_t));
  _total_num_levels = _max_level + 1;
  _num_gates_per_level = (size_t *)malloc((_max_level + 1)*sizeof(size_t));
  memset(_num_gates_per_level, 0, ((_max_level + 1))*sizeof(size_t));

  for (size_t i = 0; i < _num_PIs; i++) {
    _num_gates_per_level[_PIs[i]._level]++;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _num_gates_per_level[_gates[i]._level]++;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _num_gates_per_level[_POs[i]._level]++;
  }
  
  std::vector<std::vector<size_t>> gates_per_level; // each level contains which gate_idx
  std::vector<size_t> level_of_gates; // each gate belongs to which level 
  gates_per_level.resize(_total_num_levels);
  level_of_gates.resize(_num_PIs + _num_inner_gates + _num_POs);

  for (size_t i = 0; i < _num_PIs; i++) {
    size_t lvl = _PIs[i]._level;
    gates_per_level[lvl].push_back(_PIs[i]._idx);
    level_of_gates[_PIs[i]._idx] = lvl;
    assert(lvl == 0 && "PI lvl ERROR\n");
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    size_t lvl = _gates[i]._level;
    gates_per_level[lvl].push_back(_gates[i]._idx);
    level_of_gates[_gates[i]._idx] = lvl;
    assert(lvl > 0 && "gate 0 lvl ERROR\n");
    assert(lvl < (_total_num_levels-1) && "gate -1 lvl ERROR\n");
  }
  for (size_t i = 0; i < _num_POs; i++) {
    size_t lvl = _POs[i]._level;
    gates_per_level[lvl].push_back(_POs[i]._idx);
    level_of_gates[_POs[i]._idx] = lvl;
    assert(lvl == (_total_num_levels-1) && "PO lvl ERROR\n");
  }

  // printf("gates_per_level, %lu %lu %Lu\n", _num_PIs, _num_inner_gates, _num_POs);
  // for (size_t lvl = 0; lvl < _total_num_levels; lvl++) {
  //   printf("lvl_%lu, size() = %lu\n", lvl, gates_per_level[lvl].size());;
  // }

#ifdef FSIM_DEBUG_PRINT_GRAPH
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
  print_num_gates_per_level();
#endif

  // tailed-end optimization: check how many layers can be merged
  // i == (_total_num_levels-1): PO, i == 0: PI
  for (size_t i = _total_num_levels-2; i > 0; i--) {
    if (_num_gates_per_level[i] <= _tail_ended_OPT_threshold) {
      _start_tail_end_level = i;
      // printf("i = %lu, %lu\n", i, _num_gates_per_level[i]);
    } else {
      break;
    }
  }
  _pull_up_tail_end_optimization = (((_total_num_levels - _start_tail_end_level) > 1)) 
                                          ? (true) : (false);
  // _pull_up_tail_end_optimization = false;
  // printf("_start_tail_end_level = %lu %d\n", _start_tail_end_level, _pull_up_tail_end_optimization);

  _pull_up_head_frt_optimization = true;
  // _pull_up_head_frt_optimization = false; 
  if (!_pull_up_head_frt_optimization) {
    _order_cpu = (size_t *)malloc((_num_PIs+_num_inner_gates+_num_POs)*sizeof(size_t));
    memset(_order_cpu, 0, ((_num_PIs+_num_inner_gates+_num_POs))*sizeof(size_t));
    size_t accum = 0;
    for (size_t i = 0; i < gates_per_level.size(); i++) {
      for (size_t j = 0; j < gates_per_level[i].size(); j++) {
        _order_cpu[accum] = gates_per_level[i][j];
        accum++;
      }
    }
    assert(accum == (_num_PIs+_num_inner_gates+_num_POs) && "accum ERROR");
  } else {
    // head-front optimization 
    _partition(adj_topol, inv_adj_topol, gates_per_level, level_of_gates);
  }

}

void CUDASimulator::_partition(std::vector<std::vector<size_t>> &adj_topol,
                              std::vector<std::vector<size_t>> &inv_adj_topol,
                              std::vector<std::vector<size_t>> &gates_per_level,
                              std::vector<size_t> &level_of_gates) {
  
  // for (size_t lvl = 0; lvl < _start_tail_end_level; lvl++) {
  //   printf("lvl %lu.size() = %lu\n", lvl, _num_gates_per_level[lvl]);
  // }
  // printf("Note: only show before lvl__start_tail_end_level\n");

  // Decide which levels are going to merged into one kernel node 
  // note: level_0: PIs
  // vectors: (0, 1, 2): (which set, which level, contains gates index)
  std::vector<std::vector<std::vector<size_t>>> partition_sets;

  // _find_sets
  std::set<size_t> visited; 

  // Note: start_level == 0: PI level
  _end_head_frt_level = _start_tail_end_level;
  assert(_start_head_frt_level != _end_head_frt_level);
  size_t start_level = _start_head_frt_level; // TODO: need to change dynamically
  size_t end_level = _end_head_frt_level; // TODO: need to change dynamically 
  assert(start_level != 0 && "PI cannot be the start_level\n");
  assert(end_level <= _start_tail_end_level && "end_level cannot equal to _start_tail_end_level\n");

  for (size_t lvl = start_level; lvl < end_level; lvl++) {
    // printf("Hello %lu, lvl = %lu\n", gates_per_level[lvl].size(), lvl);
    for (size_t g = 0; g < gates_per_level[lvl].size(); g++) {
      size_t gate_idx = gates_per_level[lvl][g];
      if (visited.find(gate_idx) == visited.end()) {
        // printf("Find sets starts, gate_idx = %lu (set_idx = %lu)\n", gate_idx, partition_sets.size());
        _find_sets(gate_idx, start_level, end_level, visited, adj_topol, inv_adj_topol, level_of_gates, partition_sets);
      }
    }
  }


#ifdef FSIM_CHECK_PARTITION_RESULTS
  // check the _find_sets
  printf("partition_sets (%lu) =\n", partition_sets.size());
  for (size_t set_idx = 0; set_idx < partition_sets.size(); set_idx++) {
    size_t total_size = 0;
    bool print = false;
    for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
      if (partition_sets[set_idx][lvl].size() >= 1024) {
        print = true;
      }  
      total_size += partition_sets[set_idx][lvl].size();
    } 
    if (print) {
      printf("partition_sets[%lu] (total_size() = %lu): ", set_idx, total_size);
      for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
        printf("%lu, ", partition_sets[set_idx][lvl].size());
      } printf("\n");
    }

  } printf("\n");
#endif


  std::vector<std::vector<std::vector<size_t>>> partition_sets_2;
  // split large sets
  for (size_t set_idx = 0; set_idx < partition_sets.size(); set_idx++) {
    bool need_to_split = false;
    for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
      if (partition_sets[set_idx][lvl].size() >= 1024) {
        need_to_split = true;
      }
    }
    
    if (need_to_split) {
      // printf("need_to_split: partition_sets[%lu]\n", set_idx);
      // for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
      //   printf("partition_sets[%lu][%lu].size() = %lu\n", set_idx, lvl, partition_sets[set_idx][lvl].size());
      // }

      _find_sub_sets(set_idx, start_level, end_level, adj_topol, inv_adj_topol, 
                    level_of_gates, partition_sets, partition_sets_2);
      // printf("partition_sets_2.size() = %lu\n", partition_sets_2.size());
    } else {
      // directly copy 
      size_t set_idx_2 = partition_sets_2.size();
      partition_sets_2.push_back(std::vector<std::vector<size_t>>());
      for (size_t lvl = 0; lvl < (end_level - start_level); lvl++) {
        partition_sets_2[set_idx_2].push_back(partition_sets[set_idx][lvl]);
      }
    }
  }


#ifdef FSIM_CHECK_PARTITION_SUBSETS_RESULTS
  std::vector<size_t> aa_0;
  std::vector<size_t> aa_1;

  for (size_t set_idx = 0; set_idx < partition_sets.size(); set_idx++) {
    for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
      for (size_t ele = 0; ele < partition_sets[set_idx][lvl].size(); ele++) {
        aa_0.push_back(partition_sets[set_idx][lvl][ele]);
      }
    }
  }

  for (size_t set_idx = 0; set_idx < partition_sets_2.size(); set_idx++) {
    for (size_t lvl = 0; lvl < partition_sets_2[set_idx].size(); lvl++) {
      for (size_t ele = 0; ele < partition_sets_2[set_idx][lvl].size(); ele++) {
        aa_1.push_back(partition_sets_2[set_idx][lvl][ele]);
      }
    }
  }

  size_t replication = aa_1.size() - aa_0.size(); 
  printf("replication = %lu\n", replication);


  sort(aa_0.begin(), aa_0.end());
  sort(aa_1.begin(), aa_1.end());

  auto it_0 = unique(aa_0.begin(), aa_0.end()); 
  aa_0.erase(it_0, aa_0.end()); 
  auto it_1 = unique(aa_1.begin(), aa_1.end()); 
  aa_1.erase(it_1, aa_1.end()); 
  printf("aa_0.size() = %lu, aa_1.size() = %lu (Delta: %lu)\n", aa_0.size(), aa_1.size(), aa_0.size() - aa_1.size());
    
  for (size_t i = 0; i < aa_0.size(); i++) {
    if (aa_0[i] != aa_1[i]) {
      printf("aa_0[%lu] = %lu, aa_1[%lu] = %lu\n", i, aa_0[i], i, aa_1[i]);
      exit(1);
    }
  }
  // ＴＯＤＯ：
  // (V) 1. 目前的 bugs: 找出來的 partition_2 有少 gates 
  // (V) 2. 把小的東東做 merge
  // 3. 接 CUDA Graph 接圖
  // 加油（滾滾）
#endif

  // merge small sets
  std::vector<std::vector<std::vector<size_t>>> _partition_results;
  _merge_small_sets(start_level, end_level, partition_sets_2, _partition_results);

  // need to update: 
  /*
  1. _num_gates_per_level_gpu -->>>> update _num_gates_per_level (this is an CPU array)
  2. _order_gpu_ -->>>> update `_order_cpu` array 
  */

  // Update _num_total_gates_after_head_end_opt
  _num_total_gates_after_head_end_opt = 0;
  for (size_t lvl = 0; lvl < start_level; lvl++) {
    _num_total_gates_after_head_end_opt += _num_gates_per_level[lvl];
  } 
  _num_total_gates_after_head_end_opt += _partition_results.size()
                                        *(end_level-start_level)*_head_front_OPT_threshold;
  for (size_t lvl = end_level; lvl < _total_num_levels; lvl++) {
    _num_total_gates_after_head_end_opt += _num_gates_per_level[lvl];
  }                                        

  // Update _num_gates_per_level_gpu
  for (size_t lvl = start_level; lvl < end_level; lvl++) {
    _num_gates_per_level[lvl] = _partition_results.size()*_head_front_OPT_threshold;
    // gates_per_level --> no need to update!!!
  }

  // note: _num_total_gates_after_head_end_opt contains lots of NULL-gates
  // printf("_num_total_gates_after_head_end_opt = %lu (ori = %lu)\n", 
  //       _num_total_gates_after_head_end_opt,
  //       _num_PIs+_num_inner_gates+_num_POs);

  // Update _order_gpu
  _order_cpu = (size_t *)malloc(_num_total_gates_after_head_end_opt*sizeof(size_t));
  memset(_order_cpu, 0, (_num_total_gates_after_head_end_opt)*sizeof(size_t));
  size_t accum = 0;
  for (size_t lvl = 0; lvl < start_level; lvl++) {
    for (size_t g = 0; g < gates_per_level[lvl].size(); g++) {
      _order_cpu[accum] = gates_per_level[lvl][g];
      accum++;
    }
  }

  size_t num_null_gate = 0;
  // simply use: _partition_results[0].size() as the bound, since all of them are the same
  assert(_partition_results[0].size() == (end_level-start_level));
  for (size_t lvl = 0; lvl < _partition_results[0].size(); lvl++) {
    for (size_t set = 0; set < _partition_results.size(); set++) {
      for (size_t ele = 0; ele < _partition_results[set][lvl].size(); ele++) {
        _order_cpu[accum] = _partition_results[set][lvl][ele];
        accum++;
      }
      // NULL gate
      for (size_t ele = _partition_results[set][lvl].size(); ele < _head_front_OPT_threshold; ele++) {
        _order_cpu[accum] = NULL_gate_idx; 
        accum++;
        num_null_gate++;
      }
    }
  }

  for (size_t lvl = end_level; lvl < _total_num_levels; lvl++) {
    for (size_t g = 0; g < gates_per_level[lvl].size(); g++) {
      _order_cpu[accum] = gates_per_level[lvl][g];
      accum++;
    }
  }
  
  // printf("accum = %lu, _num_total_gates_after_head_end_opt = %lu, num_null_gate = %lu\n", 
  //       accum, _num_total_gates_after_head_end_opt, num_null_gate);
  assert(accum == _num_total_gates_after_head_end_opt && "accum ERROR");
  printf("_num_inner_gates = %lu, _num_total_gates_after_head_end_opt = %lu, num_null_gate = %lu, \n", 
        _num_inner_gates, _num_total_gates_after_head_end_opt-_num_PIs-_num_POs, num_null_gate);  
  // note: Have checked that all gates are included 
}

void CUDASimulator::_find_sets(const size_t gate_idx, const size_t start_level, const size_t end_level, 
                              std::set<size_t> &visited,
                              std::vector<std::vector<size_t>> &adj_topol,
                              std::vector<std::vector<size_t>> &inv_adj_topol,
                              std::vector<size_t> &level_of_gates, 
                              std::vector<std::vector<std::vector<size_t>>> &partition_sets) {

  size_t set_idx = partition_sets.size();
  partition_sets.push_back(std::vector<std::vector<size_t>>());
  for (size_t lvl = 0; lvl < (end_level - start_level); lvl++) {
    partition_sets[set_idx].push_back(std::vector<size_t>());
  }

  std::queue<size_t> gate_in_same_set_candidates;
  gate_in_same_set_candidates.push(gate_idx);

  while (!gate_in_same_set_candidates.empty()) {
    size_t cur_idx = gate_in_same_set_candidates.front();
    gate_in_same_set_candidates.pop();

    if (visited.find(cur_idx) != visited.end()) {
      continue;
    }
    visited.insert(cur_idx);

    // add adj: --> 
    for (auto idx : adj_topol[cur_idx]) {
      if (level_of_gates[idx] < end_level) {
        gate_in_same_set_candidates.push(idx);
      }
    }
    
    // add inv-adj: <-- 
    for (auto idx : inv_adj_topol[cur_idx]) {
      if (level_of_gates[idx] >= start_level) {
        gate_in_same_set_candidates.push(idx);
      }
    }

    size_t lvl = level_of_gates[cur_idx] - start_level;
    assert(lvl < (end_level - start_level) && "lvl out of bound LAR");
    partition_sets[set_idx][lvl].push_back(cur_idx);
  }
}

void CUDASimulator::_find_sub_sets(const size_t set_idx, const size_t start_level, const size_t end_level, 
                                  std::vector<std::vector<size_t>> &adj_topol,
                                  std::vector<std::vector<size_t>> &inv_adj_topol,
                                  std::vector<size_t> &level_of_gates,
                                  std::vector<std::vector<std::vector<size_t>>> &partition_sets,
                                  std::vector<std::vector<std::vector<size_t>>> &partition_sets_2) {

  std::vector<std::set<size_t>> visited;
  visited.resize(end_level-start_level);
  size_t total_gates = 0; 
  for (size_t lvl = 0; lvl < partition_sets[set_idx].size(); lvl++) {
    total_gates += partition_sets[set_idx][lvl].size();
  } 

  // find end gates
  std::vector<std::set<std::pair<size_t, bool>>> end_gates;
  end_gates.resize(end_level - start_level);
  // search from the last level 
  for (int lvl = partition_sets[set_idx].size()-1; lvl > -1; lvl--) {
    for (size_t g = 0; g < partition_sets[set_idx][lvl].size(); g++) {
      bool end_p = true;
      size_t gate_idx = partition_sets[set_idx][lvl][g];
      // add adj: -->
      for (auto idx : adj_topol[gate_idx]) {
        size_t output_lvl = level_of_gates[idx];
        if (output_lvl < end_level) {
          end_p = false;
          break;
        }
      } 

      if (end_p) {
        end_gates[lvl].insert({gate_idx, false});
      }      
    }
  }

  size_t total_end_gates = 0;
  for (size_t i = 0; i < end_gates.size(); i++) {
    total_end_gates += end_gates[i].size();
  }
  // printf("end_gates.size() = %lu, total_gates = %lu\n", total_end_gates, total_gates);

  size_t find_end_gates = 0;

  // printf("ORI: partition_sets_2.size() = %lu\n", partition_sets_2.size());

  while(find_end_gates < total_end_gates) {
    // printf("find_end_gates = %lu, total_end_gates = %lu\n", find_end_gates, total_end_gates);
    std::vector<std::vector<size_t>> sub_set; 
    sub_set.resize(end_level - start_level);

    bool should_break = false;

    for (int lvl = end_gates.size() - 1; lvl > -1 && !should_break; lvl--) {
      for (auto g = end_gates[lvl].begin(); g != end_gates[lvl].end() && !should_break; ++g) {
          // size_t source_idx = partition_sets[set_idx][lvl][g]; 
          size_t source_idx = g->first;

        // printf("source_idx = %lu (%d %lu %lu %lu)\n", source_idx, lvl, g, partition_sets[set_idx].size()-1, partition_sets[set_idx][lvl].size());
        if (visited[lvl].find(source_idx) != visited[lvl].end()) {
          continue;
        }
    
        std::set<size_t> visited_2; // visited inside 
        std::vector<size_t> added_num_gates; 
        added_num_gates.resize(end_level-start_level, 0);

        std::queue<size_t> gate_in_same_set_candidates;
        gate_in_same_set_candidates.push(source_idx);

        while (!gate_in_same_set_candidates.empty()) {
          size_t cur_idx = gate_in_same_set_candidates.front();
          gate_in_same_set_candidates.pop();

          size_t lvl_cur_idx = level_of_gates[cur_idx] - start_level;

    
          // skip the iteraiton if visited (or visited_2)
          if ((visited[lvl_cur_idx].find(cur_idx) != visited[lvl_cur_idx].end()) || 
              (visited_2.find(cur_idx) != visited_2.end()) ) {
            continue;
          }

          visited_2.insert(cur_idx);
          assert(cur_idx < (_num_inner_gates+_num_PIs) && "cur_idx out of bound");
      
          // add inv-adj: <-- 
          for (auto idx : inv_adj_topol[cur_idx]) {
            if (level_of_gates[idx] >= start_level) {
              gate_in_same_set_candidates.push(idx);
            }
          }
          
          
          sub_set[lvl_cur_idx].push_back(cur_idx);
          assert(lvl_cur_idx < sub_set.size() && "lvl_cur_idx size bomb\n");
          
          assert(lvl_cur_idx < (added_num_gates.size()));
          added_num_gates[lvl_cur_idx]++;
          assert(added_num_gates[added_num_gates.size()-1] < 2 && "added_num_gates[added_num_gates.size()-1] should == 1");
        }

        if (!_can_merge(sub_set)) {
          // printf("before _split_tmp_subsets\n");
          // for (size_t ii = 0; ii < (end_level - start_level); ii++) {
            // printf("sub_set[%lu].size() = %lu\n", ii, sub_set[ii].size());
          // }
          _split_tmp_subsets(sub_set, added_num_gates);
          // printf("after _split_tmp_subsets\n");
          // for (size_t ii = 0; ii < (end_level - start_level); ii++) {
            // printf("sub_set[%lu].size() = %lu\n", ii, sub_set[ii].size());
          // }
          should_break = true;
          
          // update visited
          size_t start_clear_level = end_level-start_level-1;
          for (int ii = (end_level-start_level-1); ii > -1; ii--) {
            size_t found_eg = 0;
            for (auto gg = end_gates[ii].begin(); gg != end_gates[ii].end(); ++gg) {
              if (gg->second){
                found_eg++;
              }
            }

            if (found_eg != end_gates[ii].size()) {
              start_clear_level = ii;
              // printf("0: start_clear_level = %lu (%lu)\n", ii, partition_sets[set_idx][ii].size());
              break;
            } 
          }
          if (sub_set[start_clear_level].size() == _head_front_OPT_threshold) {
            start_clear_level--;
          }
          // printf("1: start_clear_level = %lu\n", start_clear_level);
          for (int ii = start_clear_level; ii > -1; ii--) {
            visited[ii].clear();
          }
          // update visited
        } else {
          // update visited: copy the elements in visited_2 to visited
          // since there is no duplication between visited and visited_2
          // printf("_can_merge\n");
          // printf("before merge\n");
          // for (size_t ii = 0; ii < (end_level - start_level); ii++) {
          //   printf("sub_set[%lu].size() = %lu\n", ii, sub_set[ii].size());
          // }
          for (auto it = visited_2.begin(); it != visited_2.end(); ++it) {
            size_t lvl_tmp = level_of_gates[*it] - start_level;
            visited[lvl_tmp].insert(*it);

            // update find_end_gates
            auto end_gate_it = end_gates[lvl_tmp].find({*it, false});
            if (end_gate_it != end_gates[lvl_tmp].end()) {
              end_gates[lvl_tmp].erase(end_gate_it);
              end_gates[lvl_tmp].insert({*it, true});
              find_end_gates++;
            }
          }
          // printf("after merge\n");
          // for (size_t ii = 0; ii < (end_level - start_level); ii++) {
          //   printf("sub_set[%lu].size() = %lu\n", ii, sub_set[ii].size());
          // }
        }
      }
    }

    // printf("push out something\n");
    // for (size_t lvl = 0; lvl < sub_set.size(); lvl++) {
    //   printf("sub_set[%lu].size() = %lu\n", lvl, sub_set[lvl].size());
    // } 
    partition_sets_2.push_back(sub_set);
    sub_set.clear();
    sub_set.resize(end_level - start_level);
  }

  // printf("FIN: partition_sets_2.size() = %lu\n\n\n", partition_sets_2.size());
}

bool CUDASimulator::_can_merge(std::vector<std::vector<size_t>> &sub_set) {
  // printf("Check _can_merge()\n");
  bool ret = true;
  for (size_t lvl = 0; lvl < sub_set.size(); lvl++) {
    if (sub_set[lvl].size() > _head_front_OPT_threshold) {
      ret = false;
    }
  }
  return ret;
}

void CUDASimulator::_split_tmp_subsets(std::vector<std::vector<size_t>> &sub_set, 
                                      std::vector<size_t> &added_num_gates) {
  // merge need to skip the duplication 
  for (size_t i = 0; i < added_num_gates.size(); i++) {
    for (size_t j = 0; j < added_num_gates[i]; j++) {
      sub_set[i].pop_back();
    }
  }
  assert(added_num_gates[added_num_gates.size()-1] < 2 && "added_num_gates[added_num_gates.size()-1] should == 1");
}


void CUDASimulator::_merge_small_sets(const size_t start_level, const size_t end_level, 
                                    std::vector<std::vector<std::vector<size_t>>> &partition_sets_2,
                                    std::vector<std::vector<std::vector<size_t>>> &_partition_results) {
  
  // for (size_t set_idx = 0; set_idx < partition_sets_2.size(); set_idx++) {
  //   printf("set_idx_%lu: ", set_idx);
  //   for (size_t lvl = 0; lvl < partition_sets_2[set_idx].size(); lvl++) {
  //     printf("%lu, ", partition_sets_2[set_idx][lvl].size());
  //   } printf("\n");
  // }

  size_t merged_sets = 0;

  std::vector<bool> this_set_is_merged;
  this_set_is_merged.resize(partition_sets_2.size(), false);
  // printf("this_set_is_merged.size = %lu %lu\n", this_set_is_merged.size(), partition_sets_2.size());

  while (merged_sets < partition_sets_2.size()) {
    // printf("Find a new merge_set_%lu\n", _partition_results.size());
    // init 
    std::vector<std::vector<size_t>> tmp_set; 
    tmp_set.resize(end_level - start_level);
    assert(tmp_set.size() ==  partition_sets_2[0].size());

    for (size_t set_idx = 0; set_idx < partition_sets_2.size(); set_idx++) {
      if (this_set_is_merged[set_idx] == true) {
        continue;
      }

      // std::cout << "try to merge set_idx_" << set_idx << " (been merged: " << this_set_is_merged[set_idx] << ")\n";
      
      // get a set that haven't been merged
      bool can_merged = true;

      // check whether this set can be merged or not
      for (size_t lvl = 0; lvl < partition_sets_2[set_idx].size(); lvl ++) {
        size_t after_merged_sz = tmp_set[lvl].size() + partition_sets_2[set_idx][lvl].size();
        if (after_merged_sz > _head_front_OPT_threshold) {
          can_merged = false;
          break;
        }
      }

      // merge
      if (can_merged) { // push in 
        for (size_t lvl = 0; lvl < partition_sets_2[set_idx].size(); lvl++) {
          tmp_set[lvl].insert(tmp_set[lvl].end(), 
                              partition_sets_2[set_idx][lvl].begin(), 
                              partition_sets_2[set_idx][lvl].end());
        }
        this_set_is_merged[set_idx] = true;
        merged_sets++;
      } 
    }
    
    // push out 
    // printf("Push something out, merged_sets = %lu\n", merged_sets);
    _partition_results.push_back(tmp_set);

    // printf("_partition_results[%lu]: ", _partition_results.size()-1);
    // for (size_t lvl = 0; lvl < _partition_results[_partition_results.size()-1].size(); lvl ++) {
    //   printf("%lu, ", _partition_results[_partition_results.size()-1][lvl].size());
    // } printf("\n");
  }
  // printf("FIN: _partition_results.size() = %lu\n", _partition_results.size());
}


void CUDASimulator::_read_fault(std::istream &flst) {
  // _faults.resize(_num_faults);
  // cudaMallocManaged(&_faults, _num_faults*sizeof(Fault));
  _faults = (Fault *)malloc(_num_faults*sizeof(Fault));
  for (size_t i = 0; i < _num_faults; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
#ifdef FSIM_DEBUG_PRINT_FAULTS
    std::cout << wrong_gate << ", " << fault_value << "\n";
#endif
    _faults[i]._gate_with_fault = _gate_idx_to_tpg_order_idx[wrong_gate];
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<size_t>::max() : (0);
  }

#ifdef FSIM_DEBUG_PRINT_FAULTS
  print_faults(_faults);
#endif
}

void CUDASimulator::_read_pattern(std::istream &ptn) {
  _num_rounds = (_num_pattern + SIZE_T_BITS - 1) / SIZE_T_BITS; // ceiling

  // _patterns.resize(_num_rounds);
  // cudaMallocManaged(&_patterns, _num_rounds*sizeof(Pattern));
  _patterns = (Pattern *)malloc(_num_rounds*sizeof(Pattern));

  for (size_t i = 0; i < _num_rounds; i++) {
    // _patterns[i]._value.resize(_num_PIs);
    // cudaMallocManaged(&_patterns[i]._value, _num_PIs*sizeof(size_t));
    _patterns[i]._value = (size_t *)malloc(_num_PIs*sizeof(size_t));
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = _gate_idx_to_tpg_order_idx[i];
      ptn >> _patterns[idx]._value[pi];
    }
  }
  
#ifdef FSIM_DEBUG_PRINT_PATTERNS
  print_patterns(_patterns, _num_rounds, _num_PIs);
#endif
}

void CUDASimulator::run(Mode mode, const size_t num_threads) {
  switch (mode) {
    case Mode::CUDA:
      {
      #ifdef FSIM_DEBUG_PRINT_SIMULATION
        std::cout << "run _CUDA\n";
        size_t m0 = static_cast<size_t>(mode);
        std::cout << "Mode: " << m0 << std::endl;
      #endif
      _run_parallel_cuda();
      break;
      }
    case Mode::CUDA_GRAPH:
      {
      #ifdef FSIM_DEBUG_PRINT_SIMULATION
        std::cout << "run _CUDA_GRAPH\n";
        size_t m1 = static_cast<size_t>(mode);
        std::cout << "Mode: " << m1 << std::endl;
      #endif
      _run_parallel_cuda_graph();
      break;
      }
  }
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    size_t m3 = static_cast<size_t>(mode);
    std::cout << "Mode: " << m3 << std::endl;
    print_found_fault_to_pattern(_found_fault_to_pattern);
#endif
}

void CUDASimulator::_run_parallel_cuda() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel cuda();" << '\n';
#endif

  // ask memory 
  _gpu_ask_copy_mem();

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "CUDA: Run Good case" << std::endl;
#endif

    _run_good_case_cuda(rd, num_testcases_this_round);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_faults; j++) {
      _run_bad_case_cuda(rd, j, num_testcases_this_round);
      _compare_g_b_results <<< 1, 1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
    }
  }
  
  // free mem
  _free_all();
  
  for (size_t i = 0; i < _num_POs; i++) {
    free(_POs[i]._inputs);
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    free(_gates[i]._inputs);
  }
  for (size_t i = 0; i < _num_rounds; i++) {
    free(_patterns[i]._value);
  }
  free(_gate_idx_to_tpg_order_idx);
  free(_PIs);
  free(_gates);
  free(_POs);
  free(_num_gates_per_level);
  free(_faults);
  free(_patterns);
  free(_gate_inputs_list);
  free(_gate_inputs_list_start); 
}

void CUDASimulator::_run_good_case_cuda(const size_t rd, const size_t bits) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda\n");
#endif  
  
  size_t num_blocks, num_threads;
  const size_t fault_num = 0; const size_t bad_case = 0;
  size_t accum = 0;

  // Simulation
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
  // printf("num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  
  _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                            _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                            _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                            _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                            rd, fault_num, bad_case, _num_PIs, accum, _num_PIs);


#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  accum += _num_PIs;
  for (size_t i = 1; i < (_total_num_levels - 1); i++) {
    
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                              _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                              _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                              _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                              rd, fault_num, bad_case, _num_gates_per_level[i], accum, _num_PIs);

#ifdef FSIM_DEBUG_PRINT_SIMULATION
    cudaDeviceSynchronize();
    printf("---\n");
#endif      
    accum += _num_gates_per_level[i];
  }

  // POs
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);// _num_POs; //_num_POs;
  _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                            _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                            _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                            _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                            rd, fault_num, bad_case, _num_POs, accum, _num_PIs);


#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  

  // shift and copy answer to the good results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_gate_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  
}

void CUDASimulator::_run_bad_case_cuda(const size_t rd, const size_t fault_num, const size_t bits) {    
  size_t num_blocks, num_threads;
  const size_t bad_case = 1;
  size_t accum = 0;

  // fault simulation
  // Simulation
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
  // printf("num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  
  _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                            _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                            _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                            _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                            rd, fault_num, bad_case, _num_PIs, accum, _num_PIs);

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  accum += _num_PIs;
  for (size_t i = 1; i < (_total_num_levels - 1); i++) {
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                              _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                              _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                              _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                              rd, fault_num, bad_case, _num_gates_per_level[i], accum, _num_PIs);   
    accum += _num_gates_per_level[i];
  }

  // POs
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);// _num_POs; //_num_POs;
  _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
                                            _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
                                            _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
                                            _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
                                            rd, fault_num, bad_case, _num_POs, accum, _num_PIs);
                                            

  // shift and copy answer to the good results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_gate_results_gpu,  _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
  } 
}

__global__ void _set_params(size_t *_params_gpu_idx_address, 
                            size_t var_0, size_t var_1, size_t var_2, 
                            size_t var_3, size_t var_4, size_t var_5, size_t var_6) {

  _params_gpu_idx_address[0] = var_0;
  _params_gpu_idx_address[1] = var_1;
  _params_gpu_idx_address[2] = var_2;
  _params_gpu_idx_address[3] = var_3;
  _params_gpu_idx_address[4] = var_4;
  _params_gpu_idx_address[5] = var_5;
  _params_gpu_idx_address[6] = var_6;
}

void CUDASimulator::_construct_push_single_lvl_gate_KN(cudaStream_t stream1, cudaGraph_t graph, 
                                              const size_t i, const size_t params_gpu_idx_idx, 
                                              const size_t params_gpu_idx_accum, 
                                              const size_t num_func_inputs,
                                              const size_t num_gates_this_level,
                                              const size_t accum, 
                                              const size_t num_blocks, 
                                              const size_t num_threads, 
                                              size_t **params_gpu_idx,
                                              void **kernelArgs_gate, 
                                              cudaGraphNode_t *kernelNode_gate, 
                                              cudaKernelNodeParams *kernelNode_gate_Params) {
  
  params_gpu_idx[params_gpu_idx_idx] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_gate[(i-1)*num_func_inputs+0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+5] = (void *)&_patterns_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_gate[(i-1)*num_func_inputs+8] = (void *)&params_gpu_idx[params_gpu_idx_idx];
  kernelArgs_gate[(i-1)*num_func_inputs+9] = (void *)&_order_gpu;

  // construct gate node
  // size_t num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);
  // size_t num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);
  
  size_t rd = 0;
  size_t fault_num = 0;
  size_t bad_case = 0;
  size_t reserved_space = 0;

  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[params_gpu_idx_idx], rd, fault_num, bad_case, 
                                        num_gates_this_level, accum, _num_PIs, reserved_space);
  cudaDeviceSynchronize();
  cudaCheckErrors("gate _set_params cudaDeviceSynchronize failed");
      
  // Adding nodes
  kernelNode_gate_Params[(i-1)].func = (void *)_run_gate_graph;
  kernelNode_gate_Params[(i-1)].gridDim = dim3(num_blocks, 1, 1);
  kernelNode_gate_Params[(i-1)].blockDim = dim3(num_threads, 1, 1);
  kernelNode_gate_Params[(i-1)].sharedMemBytes = 0;
  kernelNode_gate_Params[(i-1)].kernelParams = (void **)&(kernelArgs_gate[(i-1)*num_func_inputs]);
  kernelNode_gate_Params[(i-1)].extra = NULL;  
  cudaCheckErrors("kernelNode_gate_Params failed");
  
  // push node: gate
  cudaGraphAddKernelNode(&kernelNode_gate[(i-1)], graph, NULL, 0, &kernelNode_gate_Params[(i-1)]);
  cudaCheckErrors("Adding kernelNode_gate failed");
#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
  printf("s Construct gate_KN_%lu\n", i-1);
#endif
}

void CUDASimulator::_construct_push_multi_lvl_gate_KN(cudaStream_t stream1, cudaGraph_t graph, 
                                              const size_t level, 
                                              const size_t params_gpu_idx_idx, 
                                              const size_t params_gpu_idx_accum, 
                                              const size_t num_func_inputs,
                                              const size_t accum, 
                                              const size_t num_blocks, 
                                              const size_t num_threads, 
                                              const size_t start_level, 
                                              const size_t end_level, 
                                              size_t **params_gpu_idx,
                                              void **kernelArgs_gate, 
                                              cudaGraphNode_t *kernelNode_gate, 
                                              cudaKernelNodeParams *kernelNode_gate_Params) {
  
  params_gpu_idx[params_gpu_idx_idx] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_gate[(level-1)*num_func_inputs+0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+5] = (void *)&_patterns_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+8] = (void *)&params_gpu_idx[params_gpu_idx_idx];
  kernelArgs_gate[(level-1)*num_func_inputs+9] = (void *)&_order_gpu;
  kernelArgs_gate[(level-1)*num_func_inputs+10] = (void *)&_num_gates_per_level_gpu;

  // construct gate node
  size_t rd = 0;
  size_t fault_num = 0;
  size_t bad_case = 0;
  
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[params_gpu_idx_idx], rd, fault_num, bad_case, 
                                        start_level, accum, end_level, _num_PIs);
  cudaDeviceSynchronize();
  cudaCheckErrors("tail gate _set_params cudaDeviceSynchronize failed");
      
  // Adding nodes
  kernelNode_gate_Params[(level-1)].func = (void *)_run_gate_graph_mul_levels;
  kernelNode_gate_Params[(level-1)].gridDim = dim3(num_blocks, 1, 1);
  kernelNode_gate_Params[(level-1)].blockDim = dim3(num_threads, 1, 1);
  kernelNode_gate_Params[(level-1)].sharedMemBytes = 0;
  kernelNode_gate_Params[(level-1)].kernelParams = (void **)&(kernelArgs_gate[(level-1)*num_func_inputs]);
  kernelNode_gate_Params[(level-1)].extra = NULL;  
  cudaCheckErrors("tail kernelNode_gate_Params failed");

  // push node: gate
  cudaGraphAddKernelNode(&kernelNode_gate[(level-1)], graph, NULL, 0, &kernelNode_gate_Params[(level-1)]);
  cudaCheckErrors("tail Adding kernelNode_gate failed");
#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
  printf("m Construct gate_KN_%lu (ACT level: %lu - %lu)\n", level-1, start_level, end_level-1);
#endif 
}

void CUDASimulator::_construct_cuda_graph(cudaStream_t stream1, cudaGraph_t graph, 
                                        cudaGraphNode_t *kernelNode_PI, 
                                        cudaKernelNodeParams *kernelNode_PI_Params,
                                        cudaGraphNode_t *kernelNode_gate, 
                                        cudaKernelNodeParams *kernelNode_gate_Params, 
                                        cudaGraphNode_t *kernelNode_PO, 
                                        cudaKernelNodeParams *kernelNode_PO_Params,
                                        void **kernelArgs_PI,
                                        void **kernelArgs_gate,
                                        void **kernelArgs_PO,
                                        size_t **params_gpu_idx,
                                        size_t num_func_inputs) {
  size_t num_blocks, num_threads;
  size_t params_gpu_idx_accum = 0, params_gpu_idx_idx = 0;

  // construct PI node
  params_gpu_idx[params_gpu_idx_idx] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PI[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PI[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PI[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PI[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PI[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PI[5] = (void *)&_patterns_gpu;
  kernelArgs_PI[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PI[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PI[8] = (void *)&params_gpu_idx[params_gpu_idx_idx];
  kernelArgs_PI[9] = (void *)&_order_gpu;

  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;

  size_t rd = 0;
  size_t fault_num = 0;
  size_t bad_case = 0;
  size_t max_gate_num_per_level = _num_PIs;
  size_t fin_num_gates = 0;
  size_t reserved_space = 0;
  
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[params_gpu_idx_idx], rd, fault_num, bad_case, 
                                        max_gate_num_per_level, fin_num_gates, _num_PIs, reserved_space);
  cudaDeviceSynchronize();
  cudaCheckErrors("PI _set_params cudaDeviceSynchronize failed");

  // Adding PI node, as head node of graph
  kernelNode_PI_Params->func = (void *)_run_gate_graph;
  kernelNode_PI_Params->gridDim = dim3(num_blocks, 1, 1);
  kernelNode_PI_Params->blockDim = dim3(num_threads, 1, 1);
  kernelNode_PI_Params->sharedMemBytes = 0;
  kernelNode_PI_Params->kernelParams = (void **)(kernelArgs_PI);
  kernelNode_PI_Params->extra = NULL;  

  // push node: PI
  cudaGraphAddKernelNode(kernelNode_PI, graph, NULL, 0, kernelNode_PI_Params);
  cudaCheckErrors("Adding kernelNode_PI failed");

  // construct gate nodes
  size_t accum = _num_PIs;

  size_t end_level = (_pull_up_tail_end_optimization) 
                    ? (_start_tail_end_level) : (_total_num_levels - 1);

  size_t num_reduced_KNs_in_head_front_opt = (_end_head_frt_level-_start_head_frt_level)-1;

  if (_pull_up_head_frt_optimization) {
    // before head_front optimization
    for (size_t i = 1; i < _start_head_frt_level; i++) {
      params_gpu_idx_accum += _num_const_paras;
      params_gpu_idx_idx++;

#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS      
      printf("Act_level_%lu, ", i);
#endif 

      num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);
      num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);
      
      _construct_push_single_lvl_gate_KN(stream1, graph, i, params_gpu_idx_idx, 
                                params_gpu_idx_accum,
                                num_func_inputs, 
                                _num_gates_per_level[i], 
                                accum, 
                                num_blocks, num_threads, 
                                params_gpu_idx,
                                kernelArgs_gate, kernelNode_gate, kernelNode_gate_Params);
      
      accum += _num_gates_per_level[i];
    }

    // head_front_single KN
    params_gpu_idx_accum += _num_const_paras;
    params_gpu_idx_idx++;

#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
    printf("Act_level_%lu, ", _start_head_frt_level);
#endif

    
    num_blocks = (_num_gates_per_level[_start_head_frt_level] > _head_front_OPT_threshold) ? 
                (_num_gates_per_level[_start_head_frt_level] + _head_front_OPT_threshold - 1) >> _head_front_OPT_threshold_log : 
                (_head_front_OPT_threshold);
    num_threads = _head_front_OPT_threshold;

    _construct_push_multi_lvl_gate_KN(stream1, graph, _start_head_frt_level,
                                    params_gpu_idx_idx, 
                                    params_gpu_idx_accum,
                                    num_func_inputs, accum, 
                                    num_blocks, num_threads, 
                                    _start_head_frt_level, _end_head_frt_level, 
                                    params_gpu_idx,
                                    kernelArgs_gate, kernelNode_gate, kernelNode_gate_Params);
    
    for (size_t j = _start_head_frt_level; j < _end_head_frt_level; j++) {
      accum += _num_gates_per_level[j];
    }
    
    // after head_front optimization
    for (size_t i = _end_head_frt_level; i < end_level; i++) {
      size_t act_level = i;
#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
      printf("Act_level_%lu, ", act_level);
#endif
      
      params_gpu_idx_accum += _num_const_paras;
      params_gpu_idx_idx++;
      num_blocks = (_num_gates_per_level[act_level] > 1024) ? 
                  (_num_gates_per_level[act_level] + 1024 - 1) >> 10 : (1);
      num_threads = (_num_gates_per_level[act_level] > 1024) ? 
                  (1024) : (_num_gates_per_level[act_level]);
      
      _construct_push_single_lvl_gate_KN(stream1, graph, i-num_reduced_KNs_in_head_front_opt, params_gpu_idx_idx, 
                                params_gpu_idx_accum,
                                num_func_inputs, 
                                _num_gates_per_level[act_level],
                                accum, 
                                num_blocks, num_threads, 
                                params_gpu_idx,
                                kernelArgs_gate, kernelNode_gate, kernelNode_gate_Params);
      
      accum += _num_gates_per_level[act_level];
    }
  } else {
    // w/o head front OPT
    // i == 0: PI, i == (_total_num_levels-1): PO
    for (size_t i = 1; i < end_level; i++) {
      params_gpu_idx_accum += _num_const_paras;
      params_gpu_idx_idx++;

#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
    printf("Act_level_%lu, ", i);
#endif 

      num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);
      num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);
      
      _construct_push_single_lvl_gate_KN(stream1, graph, i, params_gpu_idx_idx, 
                                params_gpu_idx_accum,
                                num_func_inputs, 
                                _num_gates_per_level[i],
                                accum, 
                                num_blocks, num_threads, 
                                params_gpu_idx,
                                kernelArgs_gate, kernelNode_gate, kernelNode_gate_Params);
      
      accum += _num_gates_per_level[i];
    }
  }

  // printf("num_reduced_KNs_in_head_front_opt = %lu\n", num_reduced_KNs_in_head_front_opt);
  // tail-end optimization 
  if (_pull_up_tail_end_optimization) {
    params_gpu_idx_accum += _num_const_paras;
    params_gpu_idx_idx++;
#ifdef FSIM_CHECK_CONSTRUCT_GATE_KNS
    printf("Act_level_%lu, ", end_level);
#endif 

    num_blocks = 1;
    num_threads = _tail_ended_OPT_threshold;

    size_t level_tail = (_pull_up_head_frt_optimization) ? 
        (end_level-num_reduced_KNs_in_head_front_opt) : (end_level) ;

    _construct_push_multi_lvl_gate_KN(stream1, graph, level_tail,
                                    params_gpu_idx_idx, 
                                    params_gpu_idx_accum,
                                    num_func_inputs, accum, 
                                    num_blocks, num_threads, 
                                    end_level, _total_num_levels-1,
                                    params_gpu_idx,
                                    kernelArgs_gate, kernelNode_gate, kernelNode_gate_Params);
    
    for (size_t j = end_level; j < (_total_num_levels-1); j++) {
      accum += _num_gates_per_level[j];
    }
  }

  if (_pull_up_head_frt_optimization) {
    assert(accum == (_num_total_gates_after_head_end_opt-_num_POs) && "tail-end accum ERROR (w/ head-end OPT)\n");
  } else {
    assert(accum == (_num_PIs+_num_inner_gates) && "tail-end accum ERROR (w/o head_end OPT)\n");
  }
  // printf("_total_num_levels = %lu\n", _total_num_levels);


  // construct PO node
  end_level++; 
  params_gpu_idx_accum += _num_const_paras;
  params_gpu_idx_idx++;
  params_gpu_idx[params_gpu_idx_idx] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PO[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PO[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PO[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PO[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PO[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PO[5] = (void *)&_patterns_gpu;
  kernelArgs_PO[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PO[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PO[8] = (void *)&params_gpu_idx[params_gpu_idx_idx]; 
  kernelArgs_PO[9] = (void *)&_order_gpu;

  
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);

  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[params_gpu_idx_idx], rd, fault_num, bad_case, 
                                        _num_POs, accum, _num_PIs, reserved_space);                                      
  cudaDeviceSynchronize();  
  cudaCheckErrors("PO _set_params cudaDeviceSynchronize failed");

  // Adding PO node, as head node of graph
  kernelNode_PO_Params->func = (void *)_run_gate_graph;
  kernelNode_PO_Params->gridDim = dim3(num_blocks, 1, 1);
  kernelNode_PO_Params->blockDim = dim3(num_threads, 1, 1);
  kernelNode_PO_Params->sharedMemBytes = 0;
  kernelNode_PO_Params->kernelParams = (void **)(kernelArgs_PO);
  kernelNode_PO_Params->extra = NULL;  

  // push node: PO
  cudaGraphAddKernelNode(kernelNode_PO, graph, NULL, 0, kernelNode_PO_Params);
  cudaCheckErrors("Adding kernelNode_PO failed");


  // // Construct dependency
  cudaGraphAddDependencies(graph, kernelNode_PI, &kernelNode_gate[0], 1);
  cudaCheckErrors("cudaGraphAddDependencies PI -> gate[0]");
  size_t end_dep = (_pull_up_tail_end_optimization) ? 
          (end_level-2) : (end_level-3);
  end_dep = (_pull_up_head_frt_optimization) ? 
          (end_dep - (_end_head_frt_level-_start_head_frt_level) + 1) : (end_dep);
  // printf("end_dep = %lu\n", end_dep);
  for (size_t i = 0; i < end_dep; i++) {
    // printf("%lu -> %lu\n", i, i+1);
    cudaGraphAddDependencies(graph, &kernelNode_gate[i], &kernelNode_gate[i+1], 1);
    cudaCheckErrors("cudaGraphAddDependencies gate[i] -> gate[i+1]");
  }
  cudaGraphAddDependencies(graph, &kernelNode_gate[end_dep], kernelNode_PO, 1);
  cudaCheckErrors("cudaGraphAddDependencies gate -> PO");


#ifdef CUDA_GRAPH_CHECK
  cudaDeviceSynchronize();
  // check nodes
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  cudaGraphGetNodes(graph, nodes, &numNodes);
  cudaCheckErrors("Graph get nodes failed");
  printf("!!! Number of the nodes in the graph = %zu\n", numNodes);
#endif
}

__global__ void _update_params_gpu(size_t *_params_gpu_idx_address, size_t _num_paras, size_t limit_sz, 
                                  size_t new_rd, size_t new_fault_num, size_t new_bad_case) {
  // _params_gpu_idx_address
  size_t t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t address_start = t_idx*_num_paras;
  // printf("address_start = %ld\n", address_start);

  if (address_start > (limit_sz))
    return;

  // printf("t_idx = %ld, %ld, %ld, %ld\n", t_idx, address_start+0, address_start+1, address_start+2);
  _params_gpu_idx_address[address_start+0] = new_rd; // rd
  _params_gpu_idx_address[address_start+1] = new_fault_num; 
  _params_gpu_idx_address[address_start+2] = new_bad_case; // wrong_gate
}

void CUDASimulator::_update_cuda_graph_params(cudaStream_t &stream1, size_t rd, size_t fault_num, size_t bad_case) {
  // TODOL after adding head_front OPT, here -> need to revise
  size_t num_blocks = (_pull_up_tail_end_optimization) ? 
                      (_start_tail_end_level+3) : (_max_level+1);
  size_t limit_sz = (_pull_up_tail_end_optimization) ? 
                    (_num_const_paras*(_start_tail_end_level+3)+1) : (_num_const_paras*(_max_level+1));
  _update_params_gpu <<< num_blocks, 1, 0, stream1 >>> (_params_gpu, _num_const_paras, limit_sz, 
                                                        rd, fault_num, bad_case);  

  // TODO: the `cudaDeviceSynchronize` here is redundant (?) 
  cudaCheckErrors("GPU _update_params_gpu failed");
  cudaDeviceSynchronize();
  cudaCheckErrors("update cudaDeviceSynchronize failed");
}


void CUDASimulator::_run_parallel_cuda_graph() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel cuda_graph();" << '\n';
#endif

  // ask memory 
  _gpu_ask_copy_mem();
  cudaCheckErrors("_gpu_ask_copy_mem failure");
  
  // construct graph
  cudaStream_t stream1;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaCheckErrors("cudaStreamCreateWithFlags failure");
  cudaGraph_t graph; // main graph
  cudaGraphCreate(&graph, 0); // create the graph
  cudaCheckErrors("cudaGraphCreate failure");

  // construct_graph
  cudaGraphNode_t kernelNode_PI;
  cudaKernelNodeParams kernelNode_PI_Params {0};
  cudaGraphNode_t *kernelNode_gate = (cudaGraphNode_t *)malloc((_total_num_levels - 2)*sizeof(cudaGraphNode_t));
  cudaKernelNodeParams *kernelNode_gate_Params = (cudaKernelNodeParams *)malloc((_total_num_levels - 2)*sizeof(cudaKernelNodeParams));
  cudaGraphNode_t kernelNode_PO;
  cudaKernelNodeParams kernelNode_PO_Params {0};
  
  size_t num_func_inputs = 11; // ask a larger memory size, corresponding to the size of tail-end function 
  void **kernelArgs_PI = (void **)malloc(num_func_inputs*sizeof(void *));
  void **kernelArgs_gate = (void **)malloc((num_func_inputs*(_total_num_levels-2))*sizeof(void *));
  void **kernelArgs_PO = (void **)malloc(num_func_inputs*sizeof(void *));
  size_t **params_gpu_idx = (size_t **)malloc((_total_num_levels)*sizeof(void *));
  memset(params_gpu_idx, 0, (_total_num_levels)*sizeof(void *));

  _construct_cuda_graph(stream1, graph, 
                        &kernelNode_PI, &kernelNode_PI_Params,
                        kernelNode_gate, kernelNode_gate_Params,
                        &kernelNode_PO, &kernelNode_PO_Params, 
                        kernelArgs_PI, kernelArgs_gate, kernelArgs_PO, params_gpu_idx, num_func_inputs);
  

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaCheckErrors("Graph instantiation failed");
  

  size_t fault_num, bad_case;

  for (size_t rd = 0; rd < _num_rounds; rd++) {
  // for (size_t rd = 0; rd < 1; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // update to good case parameters
    fault_num = 0;
    bad_case = 0;   

    // update KernelNodes params (good case)
    cudaCheckErrors("a failed");
    _update_cuda_graph_params(stream1, rd, fault_num, bad_case);
    cudaCheckErrors("_update_cuda_graph_params failed");

    _run_good_case_cuda_graph(num_testcases_this_round, stream1, instance);
    
    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_faults; j++) {
      // update to bad case parameters
      fault_num = j;
      bad_case = 1;

      // update KernelNodes params (bad case)
      _update_cuda_graph_params(stream1, rd, fault_num, bad_case);
      cudaCheckErrors("_update_cuda_graph_params failed");

      _run_bad_case_cuda_graph(num_testcases_this_round, stream1, instance);
      _compare_g_b_results <<< 1, 1, 0, stream1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
    }
  }

  // // TEST the correctness of the answer
  // cudaDeviceSynchronize();
  // cudaStreamSynchronize(stream1);
  // cudaDeviceSynchronize();
  // cudaStreamSynchronize(stream1);
  // _show_computational_results <<< 1, 1, 0, stream1 >>> (_pi_gate_po_output_res_gpu, 0, _num_PIs+_num_inner_gates+_num_POs);
  // cudaStreamSynchronize(stream1);
  // cudaDeviceSynchronize();
  // // TEST the correctness of the answer


  // // free mem
  _free_all();  
  free(kernelNode_gate);
  free(kernelNode_gate_Params);
  free(_gate_inputs_list);
  free(_gate_inputs_list_start); 
  free(kernelArgs_PI);
  free(kernelArgs_gate);
  free(kernelArgs_PO);
  free(params_gpu_idx);
  cudaStreamDestroy(stream1);
  cudaGraphDestroy(graph);

  for (size_t i = 0; i < _num_POs; i++) {
    free(_POs[i]._inputs);
  }

  for (size_t i = 0; i < _num_inner_gates; i++) {
    free(_gates[i]._inputs);
  }

  for (size_t i = 0; i < _num_rounds; i++) {
    free(_patterns[i]._value);
  }

  free(_gate_idx_to_tpg_order_idx);
  free(_PIs);
  free(_gates);
  free(_POs);
  free(_num_gates_per_level);
  free(_faults);
  free(_patterns);
}

void CUDASimulator::_run_good_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda_graph\n");
#endif  
  
  cudaGraphLaunch(instance, stream1);
  cudaCheckErrors("Graph launch failed (_run_good_case_cuda_graph)");  
  cudaStreamSynchronize(stream1);
  cudaCheckErrors("cudaStreamSynchronize (_run_good_case_cuda_graph) failed");
  cudaDeviceSynchronize();


#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaCheckErrors("Graph launch failed (_run_good_case_cuda_graph)");  
  cudaDeviceSynchronize();
#endif

  size_t num_blocks, num_threads;
  // shift and copy answer to the good results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_gate_results_gpu,  _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
  }
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif
}

void CUDASimulator::_run_bad_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance) {  
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_bad_case_cuda_graph\n");
#endif 

  cudaGraphLaunch(instance, stream1);
  cudaStreamSynchronize(stream1);    

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaCheckErrors("Graph launch failed (_run_bad_case_cuda_graph)");
  cudaDeviceSynchronize();
#endif

  size_t num_blocks, num_threads;
  // shift and copy answer to the bad results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_gate_results_gpu,  _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
  }
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif
}



__global__ void _shift_to_correct_answer_gpu(size_t *outputs, size_t *_pi_gate_po_output_res_gpu, const size_t bits, const size_t SIZE_T_BITS, const size_t pre_gates_accum, const size_t num) {
  size_t gate_idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (gate_idx > num) 
    return;
  
  gate_idx += pre_gates_accum;
  outputs[gate_idx] = (_pi_gate_po_output_res_gpu[gate_idx] << (SIZE_T_BITS - bits));
  outputs[gate_idx] >>= (SIZE_T_BITS - bits);
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("gate_idx = %ld, outputs[%ld] = %ld\n", gate_idx, gate_idx, outputs[gate_idx]);
#endif
}


__global__ void _run_gate(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                          size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                          size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                          size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                          const size_t rd, const size_t fault_num, const size_t bad_case, 
                          const size_t max_gate_num_per_level, const size_t fin_num_gates, const size_t _num_PIs) {  
  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t real_g_idx = blockDim.x * b_idx + t_idx + fin_num_gates; // tmp
  if (real_g_idx > (max_gate_num_per_level+fin_num_gates))
    return;
  
  size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
  
  if (SA_fault) {
    _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
    return;
  }

  size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
  size_t num_inputs = _pi_gate_po_num_inputs_gpu[real_g_idx];
  switch (type) { 
    case 0:
      _apply_INV(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 1:
      _apply_AND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 2:
      _apply_OR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 3:
      _apply_XOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 4:
      _apply_NAND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 5:
      _apply_NOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 6:
      _apply_XNOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 7:
      _apply_MUX(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 8:
      _apply_CLKBUF(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 9:
      _apply_PI(real_g_idx, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
    break;
    case 10:
      _apply_PO(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
    break;
    case 11:
    break;
    }
}


__global__ void _run_gate_graph(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                size_t *_params_gpu, size_t *_order_gpu) {  

 // parameters
  const size_t rd = _params_gpu[0];
  const size_t fault_num = _params_gpu[1];
  const size_t bad_case = _params_gpu[2];
  const size_t max_gate_num_per_level = _params_gpu[3];
  const size_t fin_num_gates = _params_gpu[4];
  const size_t _num_PIs = _params_gpu[5];
  // const size_t reserved_space = _params_gpu[6]; // a reserved space

  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t tb_idx = blockDim.x * b_idx + t_idx;
  size_t real_g_idx = _order_gpu[tb_idx + fin_num_gates]; // tmp

  if ((max_gate_num_per_level) > tb_idx) {
    size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    if (SA_fault) {
      _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
    } else {
      size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
      size_t num_inputs = _pi_gate_po_num_inputs_gpu[real_g_idx];

      switch (type) { 
        case 0:
          _apply_INV(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 1:
          _apply_AND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 2:
          _apply_OR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 3:
          _apply_XOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 4:
          _apply_NAND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 5:
          _apply_NOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 6:
          _apply_XNOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 7:
          _apply_MUX(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 8:
          _apply_CLKBUF(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 9:
          _apply_PI(real_g_idx, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
        break;
        case 10:
          _apply_PO(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
        break;
        case 11: // NULL gate
        break;
      }
    }
  }
}

__global__ void _run_gate_graph_mul_levels(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                    size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                    size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                    size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                    size_t *_params_gpu, size_t *_order_gpu, size_t *_num_gates_per_level_gpu) {

  // parameters
  const size_t rd = _params_gpu[0];
  const size_t fault_num = _params_gpu[1];
  const size_t bad_case = _params_gpu[2];
  const size_t start_level = _params_gpu[3]; // merged_KN, start level index
  size_t fin_num_gates = _params_gpu[4];
  const size_t end_level = _params_gpu[5]; // merged_KN, end_level index
  const size_t _num_PIs = _params_gpu[6]; 
  
  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t tb_idx = blockDim.x * b_idx + t_idx;

  for (size_t lvl = start_level; lvl < end_level; lvl++) { 
    if (_num_gates_per_level_gpu[lvl] > tb_idx) { // tb_idx smaller than the #gates in this level 
      size_t gate_idx = _order_gpu[tb_idx + fin_num_gates];                                           
      size_t SA_fault = (((gate_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
      // note: in GPU, PIs, gates, POs are leveraged in the same array, so we don't need to consider the 
      // index accumulation for PI, gate, PO on GPU (which is different from CPU)
      if (SA_fault) {
        _pi_gate_po_output_res_gpu[gate_idx] = _fault_SA_fault_val_gpu[fault_num];
      } else {
        size_t type = (gate_idx == 18446744073709551615) ? 
                      (11) : (_pi_gate_po_gate_type_gpu[gate_idx]);
        size_t num_inputs = (gate_idx == 18446744073709551615) ? 
                      (0) : (_pi_gate_po_num_inputs_gpu[gate_idx]);
  
        switch (type) { 
          case 0:
            _apply_INV(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 1:
            _apply_AND(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 2:
            _apply_OR(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 3:
            _apply_XOR(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 4:
            _apply_NAND(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 5:
            _apply_NOR(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 6:
            _apply_XNOR(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 7:
            _apply_MUX(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 8:
            _apply_CLKBUF(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 9:
            _apply_PI(gate_idx, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+gate_idx]);
          break;
          case 10:
            _apply_PO(gate_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
          break;
          case 11:
          break;
        }
      } 
    }
    fin_num_gates += _num_gates_per_level_gpu[lvl]; // update the starting point
    __syncthreads();
  }
}

// simulation for gates
__device__ __forceinline__ void _apply_INV(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx]; // pass in real index
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];

  _pi_gate_po_output_res_gpu[gate_idx] = ~(_pi_gate_po_output_res_gpu[pre_gate_idx]);
}

__device__ __forceinline__ void _apply_AND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret &= now_gate_val;     
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_OR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_XOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ret;
}

__device__ __forceinline__ void _apply_NAND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret &= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);
}

__device__ __forceinline__ void _apply_NOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret |= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);   
}

__device__ __forceinline__ void _apply_XNOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

  for (size_t i = 1; i < num_inputs; i++) {
    size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
    ret ^= now_gate_val; 
  }
  _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);
}

__device__ __forceinline__ void _apply_MUX(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  size_t a = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+0]];
  size_t b = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+1]];
  size_t s = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+2]];

  size_t ret = ((s & b) | ( a & (!s)));

  _pi_gate_po_output_res_gpu[gate_idx] = (ret);
}

__device__ __forceinline__ void _apply_CLKBUF(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];
}

__device__ __forceinline__ void _apply_PI(size_t gate_idx, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val) {
  _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
}

__device__ __forceinline__ void _apply_PO(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
  _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];
}

__global__ void _show_computational_results (size_t *a, size_t start, size_t end) {
  printf("_show_computational_results\n");

  for (size_t i = start; i < end; i++) {
    printf("arr[%lu] = %lu\n", i, a[i]);
  }
  printf("\n");
}

__global__ void _compare_g_b_results(const size_t j, const size_t _num_POs, const size_t rd, 
  size_t *_g_po_results_gpu, size_t *_b_po_results_gpu, size_t *_found_fault_to_pattern_gpu) {

  size_t found_fault = 0;
  for (size_t i = 0; i < _num_POs; i++) {
    if (_g_po_results_gpu[i] != _b_po_results_gpu[i]) {
      found_fault = 1;
    break;
    }
  }
  // Record whether fault can be found
  _found_fault_to_pattern_gpu[2 * j] = found_fault;
  // Record which pattern found the fault
  _found_fault_to_pattern_gpu[2 * j + 1] = rd;
}

// output file visualization graph
void CUDASimulator::visualization_graph(const std::string &outputFile) {
  using std::string_literals::operator""s;
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);

  // Open output file
  std::ofstream output_file(outputFile);

  // Check if the output file is open
  if (!output_file.is_open()) {
    std::cerr << "Error opening output file: " << outputFile << std::endl;
    return;
  }

  // write file
  output_file << "digraph G {\n";
  output_file << "\tsubgraph cluster_0 {\n";
  output_file << "\t\tcolor=black;\n";
  output_file << "\t\tnode [style=filled,color=lightgrey];\n";

  // gates
  for (size_t i = 0; i < _num_inner_gates; i++) {              // now_gate
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) { // pre_gate
      ElementBase *pre_gate = _gates[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      Gate now_gate = _gates[i];

      std::string pre_s;
      if (pre_gate->_type == GateType::PI) {
        pre_s =
            "PI_"s + std::to_string(pre_gate->_idx - pi_idx_accum) + " -> "s;
      } else {
        pre_s = "Gate_"s + std::to_string(pre_gate->_idx - gate_idx_accum) +
                "_"s + gateTypeToString(pre_gate->_type) + " -> "s;
      }

      std::string now_s;
      now_s = "Gate_"s + std::to_string(now_gate._idx - gate_idx_accum) + "_"s +
              gateTypeToString(now_gate._type) + ";"s;

      output_file << "\t\t" << pre_s << now_s << "\n";
    }
  }

  // POs
  for (size_t i = 0; i < _num_POs; i++) {              // now_gate
    for (size_t j = 0; j < _POs[i]._num_inputs; j++) { // pre_gate
      ElementBase *pre_gate = _POs[i]._inputs[j]; // pre_gate can be PI, Gate, PO
      PO now_gate = _POs[i];

      std::string pre_s;
      if (pre_gate->_type == GateType::PI) {
        pre_s =
            "PI_"s + std::to_string(pre_gate->_idx - pi_idx_accum) + " -> "s;
      } else {
        pre_s = "Gate_"s + std::to_string(pre_gate->_idx - gate_idx_accum) +
                "_"s + gateTypeToString(pre_gate->_type) + " -> "s;
      }

      std::string now_s;
      now_s = "PO_"s + std::to_string(now_gate._idx - po_idx_accum) + ";"s;

      output_file << "\t\t" << pre_s << now_s << "\n";
    }
  }

  output_file << "\t}\n\n";

  for (size_t i = 0; i < _num_PIs; i++) {
    output_file << "\tPI_"s << std::to_string(i) << "  [shape=Msquare];\n"s;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    output_file << "\tPO_"s << std::to_string(i) << "  [shape=Msquare];\n"s;
  }
  output_file << "}\n\n";

  // Close files
  output_file.close();
}

// Function to convert GateType to string
std::string CUDASimulator::gateTypeToString(GateType type) {
  switch (type) {
  case GateType::INV:
    return "INV";
  case GateType::AND:
    return "AND";
  case GateType::OR:
    return "OR";
  case GateType::XOR:
    return "XOR";
  case GateType::NAND:
    return "NAND";
  case GateType::NOR:
    return "NOR";
  case GateType::XNOR:
    return "XNOR";
  case GateType::MUX:
    return "MUX";
  case GateType::CLKBUF:
    return "CLKBUF";
  case GateType::PI:
    return "PI";
  case GateType::PO:
    return "PO";
  default:
    return "UNKNOWN";
  }
}

void CUDASimulator::_gpu_ask_copy_mem() {
  size_t sum_pi_gates_pos = _num_PIs+_num_inner_gates+_num_POs;

  cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, (sum_pi_gates_pos)*sizeof(size_t));
  cudaMalloc((void**)&_pi_gate_po_output_res_gpu, (sum_pi_gates_pos)*sizeof(size_t));
  // _pi_gate_po_num_inputs_gpu: records the #inputs for each gate (PI, PO)
  cudaMalloc((void**)&_pi_gate_po_num_inputs_gpu, (sum_pi_gates_pos)*sizeof(size_t));
  cudaMalloc((void**)&_patterns_gpu, (_num_rounds*_num_PIs)*sizeof(size_t));
  cudaMalloc((void**)&_fault_gate_idx_gpu, _num_faults*sizeof(size_t));
  cudaMalloc((void**)&_fault_SA_fault_val_gpu, _num_faults*sizeof(size_t));
  
  cudaMalloc((void**)&_g_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_g_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_g_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_b_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_b_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_b_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_found_fault_to_pattern_gpu, 2*_num_faults*sizeof(size_t));
  
  cudaMalloc((void**)&_gate_inputs_list_gpu, _sum_inputs_gates*sizeof(size_t));
  cudaMalloc((void**)&_gate_inputs_list_start_gpu, sum_pi_gates_pos*sizeof(size_t));

  cudaMalloc((void**)&_num_gates_per_level_gpu, _total_num_levels*sizeof(size_t));
  size_t orders_size = (_pull_up_head_frt_optimization) ? 
            (_num_total_gates_after_head_end_opt) : (sum_pi_gates_pos);
  cudaMalloc((void**)&_order_gpu, orders_size*sizeof(size_t));


#ifdef FSIM_LEVELIZAED_PARTITION
  cudaMalloc((void**)&_params_gpu, (_num_const_paras*_total_num_levels)*sizeof(size_t));
#endif

  // copy memory 
  cudaMemcpy(_gate_inputs_list_gpu, _gate_inputs_list, _sum_inputs_gates*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("_gate_inputs_list_gpu failure");
  cudaMemcpy(_gate_inputs_list_start_gpu, _gate_inputs_list_start, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("_gate_inputs_list_start_gpu failure");

  // prepare data for CSR format: 
  // pi_gate_po_gate_type_gpu
  size_t *pi_gate_po_gate_type = (size_t *)malloc(sum_pi_gates_pos*sizeof(size_t));
  size_t *pi_gate_po_num_inputs = (size_t *)malloc(sum_pi_gates_pos*sizeof(size_t));
  memset(pi_gate_po_gate_type, 0, (sum_pi_gates_pos)*sizeof(size_t));
  memset(pi_gate_po_num_inputs, 0, (sum_pi_gates_pos)*sizeof(size_t));
  for (size_t i = 0; i < _num_PIs; i++) {
    pi_gate_po_gate_type[i] = static_cast<size_t>(_PIs[i]._type);
    pi_gate_po_num_inputs[i] = _PIs[i]._num_inputs;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    pi_gate_po_gate_type[i+_num_PIs] = static_cast<size_t>(_gates[i]._type);
    pi_gate_po_num_inputs[i+_num_PIs] = _gates[i]._num_inputs;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    pi_gate_po_gate_type[i+_num_PIs+_num_inner_gates] = static_cast<size_t>(_POs[i]._type);
    pi_gate_po_num_inputs[i+_num_PIs+_num_inner_gates] = _POs[i]._num_inputs;
  }

  // _patterns_gpu
  size_t *_patterns_cpu = (size_t *)malloc((_num_rounds*_num_PIs)*sizeof(size_t));
  for (size_t i = 0; i < _num_rounds; i++) {
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = _num_PIs*i + pi;
      _patterns_cpu[idx] = _patterns[i]._value[pi];
    }
  }
  
  // _fault_gate_idx_gpu; _fault_SA_fault_val_gpu;
  size_t *fault_gate_idx_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
  size_t *fault_SA_fault_val_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
  for (size_t i = 0; i < _num_faults; i++) {
    fault_gate_idx_cpu[i]     = _faults[i]._gate_with_fault;
    fault_SA_fault_val_cpu[i] = _faults[i]._gate_SA_fault_val;
  }
  cudaCheckErrors("fault_gate_idx cpy failure");


  cudaMemcpy(_pi_gate_po_gate_type_gpu, pi_gate_po_gate_type, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 0 failure");
  cudaMemcpy(_pi_gate_po_num_inputs_gpu, pi_gate_po_num_inputs, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 1 failure");
  cudaMemcpy(_patterns_gpu, _patterns_cpu, (_num_rounds*_num_PIs)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 2 failure");
  cudaMemcpy(_fault_gate_idx_gpu, fault_gate_idx_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 3 failure");
  cudaMemcpy(_fault_SA_fault_val_gpu, fault_SA_fault_val_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 4 failure");
  cudaMemcpy(_num_gates_per_level_gpu, _num_gates_per_level, _total_num_levels*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 5 failure");

  cudaMemcpy(_order_gpu, _order_cpu, orders_size*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy 6 failure");
  
  
  free(pi_gate_po_gate_type);
  free(pi_gate_po_num_inputs);  
  free(_patterns_cpu);
  free(fault_gate_idx_cpu);
  free(fault_SA_fault_val_cpu);
  free(_order_cpu);
  
#ifdef FSIM_DEBUG_GPU_INPUTS
  // check 
  // _pi_gate_po_gate_type_gpu
  print_pi_gate_po_gate_type_gpu <<< 1, 1 >>> (_pi_gate_po_gate_type_gpu, _pi_gate_po_num_inputs_gpu, sum_pi_gates_pos);

  // _patterns_gpu
  print_patterns_gpu <<< 1, 1 >>> (_patterns_gpu, _num_rounds, _num_PIs);

  // _fault_gate_idx_gpu && _FFR_levelization_gpu
  print_fault_gate_idx_gpu <<< 1, 1 >>> (_fault_gate_idx_gpu, _fault_SA_fault_val_gpu, _num_faults, 
                                        _FFR_levelization_gpu, 0, _total_num_levels-_start_tail_end_level);

#endif
}

__global__ void print_pi_gate_po_gate_type_gpu(size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_num_inputs_gpu, size_t num) {
  printf("print_pi_gate_po_gate_type_gpu (num = %ld): [\n", num);
  for (size_t i = 0; i < num; i++) {
    size_t type = _pi_gate_po_gate_type_gpu[i];
    switch (type) {
      case 0:
        printf("i = %ld, type_INV\n", i);
        break;
      case 1:
        printf("i = %ld, type_AND\n", i);
        break;
      case 2:
        printf("i = %ld, type_OR\n", i);
        break;
      case 3:
        printf("i = %ld, type_XOR\n", i);
        break;
      case 4:
        printf("i = %ld, type_NAND\n", i);
        break;
      case 5:
        printf("i = %ld, type_NOR\n", i);
        break;
      case 6:
        printf("i = %ld, type_XNOR\n", i);
        break;
      case 7:
        printf("i = %ld, type_MUX\n", i);
        break;
      case 8:
        printf("i = %ld, type_CLKBUF\n", i);
        break;
      case 9:
        printf("i = %ld, type_PI\n", i);
        break;
      case 10:
        printf("i = %ld, type_PO\n", i);
        break;
      case 11:
        printf("i = %ld, type_UNKNOWN\n", i);
        break;
    }
  }
  printf("]\n");

  printf("_pi_gate_po_num_inputs_gpu =[\n");  
  for (size_t i = 0; i < num; i++) {
    printf("%ld, ", _pi_gate_po_num_inputs_gpu[i]);
  } printf("]\n");
}
__global__ void print_patterns_gpu(size_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs) {
  printf("print_patterns_gpu (_num_rounds = %ld, _num_PIs = %ld): [\n", _num_rounds, _num_PIs);
  for (size_t i = 0; i < _num_rounds; i++) {
    printf("RD_%ld: [", i);
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = _num_PIs*i + pi;
      printf("%ld, ", _patterns_gpu[idx]);
    } printf("]\n");
  } printf("]\n");
}
__global__ void print_fault_gate_idx_gpu(size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, size_t _num_faults, 
                                          size_t * _FFR_levelization_gpu, size_t s, size_t n) {
  printf("print_fault_gate_idx_gpu (_num_faults = %ld): [\n", _num_faults);
  for (size_t i = 0; i < _num_faults; i++) {
    printf("gate_w_f = %ld, SA_val = %ld\n", _fault_gate_idx_gpu[i], _fault_SA_fault_val_gpu[i]);
  } printf("]\n");

  printf("_FFR_levelization_gpu =\n");
  for (size_t i = s; i < n; i++) {
    printf("!_FFR_levelization_gpu[%ld] = %ld\n", i, _FFR_levelization_gpu[i]);
  } printf("]\n");
}

void CUDASimulator::_free_all() {  
  cudaFree(_pi_gate_po_gate_type_gpu);
  cudaFree(_pi_gate_po_output_res_gpu);

  cudaFree(_patterns_gpu);
  cudaFree(_patterns_gpu);

  cudaFree(_fault_gate_idx_gpu);
  cudaFree(_fault_SA_fault_val_gpu);

  cudaFree(_g_pi_results_gpu);
  cudaFree(_g_gate_results_gpu);
  cudaFree(_g_po_results_gpu);
  cudaFree(_b_pi_results_gpu);
  cudaFree(_b_gate_results_gpu);
  cudaFree(_b_po_results_gpu);
  cudaFree(_found_fault_to_pattern_gpu);
  
  cudaFree(_gate_inputs_list_gpu);
  cudaFree(_gate_inputs_list_start_gpu); 

  cudaFree(_params_gpu);
}

// debug: print functions
void CUDASimulator::print_topological(
  const std::vector<std::vector<size_t>> &adj,
  const std::vector<size_t> &indegree,
  const std::vector<size_t> &order) const {
  std::cout << "\n=====\n\nadj:\n";
  for (size_t i = 0; i < adj.size(); i++) {
    std::cout << i << " = [";
    for (size_t v_len = 0; v_len < adj[i].size(); v_len++) {
      std::cout << adj[i][v_len] << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";

  std::cout << "Indegree:\n";
  for (size_t i = 0; i < indegree.size(); i++) {
    std::cout << i << " = [" << indegree[i] << "]\n";
  }

  std::cout << "Order:\n";
  for (size_t i = 0; i < order.size(); i++) {
    std::cout << order[i] << ", ";
  }
  std::cout << "\n";
}

void CUDASimulator::print_gate_input_order_table(std::vector<std::vector<size_t>> input_table) {
  std::cout << "\ngate_input_order_table:\n";
  for (size_t i = 0; i < input_table.size(); i++) {
    std::cout << "Gate_" << i << ": ";
    std::cout << "num_of_inputs = " << input_table[i][0] << ", input gate_idx: ";
    for (size_t j = 0; j < input_table[i][0]; j++) { // how many inputs for this gate
      std::cout << input_table[i][j+1] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "]\n";
  std::cout << "\n=====\n\n";
}

void CUDASimulator::print_num_gates_per_level() const {
  std::cout << "\n=====\n\n";
  std::cout << "_num_gates_per_level: \n";
  for (size_t i = 0; i < _total_num_levels; i++) {
    std::cout << "level_" << i << ": " << _num_gates_per_level[i] << "\n";
  }
  std::cout << "\n=====\n\n";
}

void CUDASimulator::print_ppg(const ElementBase &gate) const {
  std::cout << "\n=====\n\n";
  std::cout << "mem_id: " << &gate << "\n";
  std::cout << "idx: " << gate._idx << "\n";
  std::cout << "type: " << static_cast<size_t>(gate._type) << "\n";
  std::cout << "level: " << gate._level << "\n";
  std::cout << "num_inputs: " << gate._num_inputs << "\n";
  std::cout << "input_gates: [";
  for (size_t i = 0; i < gate._num_inputs; i++) {
    std::cout << gate._inputs[i] << ", ";
  }
  std::cout << "]\n";
  std::cout << "output_value: " << gate._output_value << "\n";
}

void CUDASimulator::print_faults(const Fault *faults) const {
  std::cout << "\n=====\n\n"
            << "FSIM_DEBUG_PRINT_FAULTS\n";
  for (size_t i = 0; i < _num_faults; i++) {
    std::cout << "_gate_with_fault = " << faults[i]._gate_with_fault << ", SA_"
              << faults[i]._gate_SA_fault_val << "\n";
  }
  std::cout << "\n=====\n\n";
}

void CUDASimulator::print_patterns(const Pattern *patterns,
    const size_t round,
    const size_t num_PIs) const {
  std::cout << "\n=====\n\n";

  for (size_t i = 0; i < round; i++) {
    std::cout << "[" << SIZE_T_BITS * i << ", " << SIZE_T_BITS * (i + 1) << "] bits = [\n";
    for (size_t j = 0; j < num_PIs; j++) {
      print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
    }
    std::cout << "]\n";
  }
}

void CUDASimulator::print_bits_stack(const size_t size, const void *const ptr) const {
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
void CUDASimulator::print_found_fault_to_pattern(const size_t *found_fault_to_pattern) const {
  for (size_t i = 0; i < (2 * _num_faults); i += 2) {
    if (found_fault_to_pattern[i] == 1) {
      std::cout << "fault_" << (i / 2) << " is found, with Pattern_"
                << found_fault_to_pattern[i + 1] << "\n";
    } else {
      std::cout << "fault_" << i << " cannot be found\n";
    }
  }
  std::cout << "\n=====\n\n";
}

void CUDASimulator::print_simulation_results(const size_t *pi_results, const size_t *gate_results, const size_t *po_results) const {
  for (size_t i = 0; i < _num_PIs; i++) {
    std::cout << "PI_" << i << ".output = " << pi_results[i] << "\n";
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    std::cout << "Gate_" << i << ".output = " << gate_results[i] << "\n";
  }
  for (size_t i = 0; i < _num_POs; i++) {
    std::cout << "PO_" << i << ".output = " << po_results[i] << "\n";
  }
  std::cout << "\n\n";
}