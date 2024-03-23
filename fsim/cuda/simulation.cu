#include <stdio.h>

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



// testing functions
__global__ void helloCUDA() {
  size_t t_idx = threadIdx.x, b_idx = blockIdx.x;
  printf("Hello, CUDA!, b_idx = %ld, t_idx = %ld\n", b_idx, t_idx);
}

__global__ void _run_gate_gpu(size_t *inputs){
  size_t num  = 10;
  for (size_t i = 0; i < num; i++) {
    inputs[i] = i+10;
  }
}

// CUDASimulator fucntions
void CUDASimulator::read(const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {

  using std::string_literals::operator""s;

  std::ifstream ckt(ckt_path), flst(flst_path), ptn(ptn_path);

  // TODO (study): study operator bool (cpp reference)
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

  // map table for recording gate input order
  // record for each to_gate, its from_gates's are which order (#_inputs, A/A1, B/A2, A3/S, A4)
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
  std::vector<size_t> order;

  // since only PIs will be indegree == 0
  for (size_t i = 0; i < _num_PIs; i++) {
    source.push(i);
  }

  while (!source.empty()) {
    size_t first = source.front();
    source.pop(); // queue pop front element
    order.push_back(first);
    for (size_t i = 0; i < adj[first].size(); i++) {
      indegree[adj[first][i]]--;
      if (indegree[adj[first][i]] == 0) {
        source.push(adj[first][i]);
      }
    }
  }
  
  // // make vector of order's order becomes [PIs, gates, POs] 
  // // remove the PO in the order
  // // since  error: namespace "std" has no member "remove_if, therefore, we implement the 'remove_if' by ourselves 
  order.erase(remove_if(order.begin(), order.end(),
                        [this](const size_t &x) {
                            return ((x >= _num_PIs) && (x < (_num_PIs + _num_POs)));
                        }),
              order.end());

  for (size_t i = 0; i < _num_POs; i++) {
    order.push_back(_num_PIs + i); // push_back PO's index
  }              

#ifdef FSIM_DEBUG_PRINT_GRAPH
  print_topological(adj, indegree_orig, order);
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
   * order[3] = 6
   * _gate_idx_to_tpg_order_idx[6] = 3
   * _gate_idx_to_tpg_order_idx: 0, 1, 2, 7, 5, 6, 3, 4
   * */
  // _gate_idx_to_tpg_order_idx.resize(sum_pi_gates_pos);
  _gate_idx_to_tpg_order_idx = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  memset(_gate_idx_to_tpg_order_idx, 0, sum_pi_gates_pos*sizeof(size_t));
  for (size_t i = 0; i < order.size(); i++) {
    _gate_idx_to_tpg_order_idx[order[i]] = i;
  }

  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);
  // PIs are stored in the order specified by the `order` array
  // _PIs.resize(_num_PIs);
  _PIs = (PI*)malloc(_num_PIs*sizeof(PI)); memset(_PIs, 0, _num_PIs*sizeof(PI));
  for (size_t i = 0; i < _num_PIs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + pi_idx_accum];
    _PIs[i]._idx = i + pi_idx_accum;
    _PIs[i]._type = gate_type[gate_id];
    _PIs[i]._level = default_level;
    _PIs[i]._output_value = default_output_value;
    _PIs[i]._num_inputs = 0;
  }

  // gates are stored in the order specified by the `order` array
  // _gates.resize(_num_inner_gates);
  // cudaMallocManaged(&_gates, _num_inner_gates*sizeof(Gate));
  _gates = (Gate *)malloc(_num_inner_gates*sizeof(Gate)); 
  memset(_gates, 0, _num_inner_gates*sizeof(Gate));
  for (size_t i = 0; i < _num_inner_gates; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + gate_idx_accum];
    _gates[i]._idx = i + gate_idx_accum;
    _gates[i]._type = gate_type[gate_id];
    _gates[i]._level = default_level;
    _gates[i]._output_value = default_output_value;
  }

  // POs are stored in the order specified by the `order` array
  // _POs.resize(_num_POs);
  // cudaMallocManaged(&_POs, _num_POs*sizeof(PO));
  _POs = (PO *)malloc(_num_POs*sizeof(PO)); memset(_POs, 0, _num_POs*sizeof(PO));

  for (size_t i = 0; i < _num_POs; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i + po_idx_accum];
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
      // cudaMallocManaged(&_POs[gate_idx]._inputs, num_inputs*sizeof(ElementBase*));
      _POs[gate_idx]._inputs = (ElementBase **)malloc(num_inputs*sizeof(ElementBase*));
      memset(_POs[gate_idx]._inputs, 0, num_inputs*sizeof(ElementBase*));
    } else { // gate
      size_t gate_idx = _gate_idx_to_tpg_order_idx[i]-gate_idx_accum;
      size_t num_inputs = gate_input_order_table[i][0];
      _gates[gate_idx]._num_inputs = num_inputs;
      // cudaMallocManaged(&_gates[gate_idx]._inputs, num_inputs*sizeof(ElementBase*));
      _gates[gate_idx]._inputs = (ElementBase **)malloc(num_inputs*sizeof(ElementBase*));
      memset(_gates[gate_idx]._inputs, 0, num_inputs*sizeof(ElementBase*));
    }
  }

  std::vector<size_t> index_accum;
  index_accum.resize(sum_pi_gates_pos, 0);
  size_t to_gate_idx;

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

      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        Gate &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        PO &to_gate =
            _POs[_gate_idx_to_tpg_order_idx[to_gate_id] - po_idx_accum];
        // to_gate._inputs.push_back(&from_gate);
        to_gate_idx = to_gate._idx;
        to_gate._inputs[index_accum[to_gate_idx]] = &from_gate;
        index_accum[to_gate_idx]++;
      } else { // gate -> gate
        Gate &from_gate =
            _gates[_gate_idx_to_tpg_order_idx[from_gate_id] - gate_idx_accum];
        Gate &to_gate =
            _gates[_gate_idx_to_tpg_order_idx[to_gate_id] - gate_idx_accum];
        // to_gate._inputs.push_back(&from_gate);
        to_gate_idx = to_gate._idx;
        to_gate._inputs[index_accum[to_gate_idx]] = &from_gate;
        index_accum[to_gate_idx]++;
      }
    }
  }

  // new// new// new// new// new// new// new// new// new

  sum_inputs_gates = 0;
  for (size_t i = 0; i < gate_input_order_table.size(); i++){
    sum_inputs_gates += gate_input_order_table[i][0];
  }

  _gate_inputs_list = (size_t*)malloc(sum_inputs_gates*sizeof(size_t));  // memset(input, 0, N*sizeof(float));
  _gate_inputs_list_start = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  memset(_gate_inputs_list, 0, sum_inputs_gates*sizeof(size_t));
  memset(_gate_inputs_list_start, 0, sum_pi_gates_pos*sizeof(size_t));
  
  // construct _gate_inputs_list_start
  for (size_t i = 0; i < _num_inner_gates; i++) {
    // printf(":) %ld, = %ld\n", i, _gates[i]._num_inputs);
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
      _gate_inputs_list_start[i+1+gate_idx_accum] = _gates[i]._num_inputs + _gate_inputs_list_start[i+gate_idx_accum];
    }
  }
  for (size_t i = 0; i < (_num_POs-1); i++) { // since the last one no need to add
    // printf(":) %ld, = %ld\n", i, _gates[i]._num_inputs);
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
  for (size_t i = 0; i < sum_inputs_gates; i++) {
    printf("%ld, ", _gate_inputs_list[i]);
  } printf("\n");
#endif  
  // new// new// new// new// new// new// new// new// new
  
  /** levelize */
  // levelize PIs -> set by the default_level (= 0), note: all PIs belond to the
  // level_0 , making the parallel in omp becomes easier levelize gates
  _max_level = 0;
  _max_gate_size_of_levels = 0;
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
  _num_gates_per_level = (size_t *)malloc((_max_level + 1)*sizeof(size_t));
  memset(_num_gates_per_level, 0, ((_max_level + 1))*sizeof(size_t));
  _size_of_num_gates_per_level = _max_level + 1;
  // size_t debug_ii = 0; 

  for (size_t i = 0; i < _num_PIs; i++) {
    _num_gates_per_level[_PIs[i]._level]++;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _num_gates_per_level[_gates[i]._level]++;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _num_gates_per_level[_POs[i]._level]++;
  }
  // for (size_t i = 0; i < _max_level + 1; i++) {
  //   debug_ii+=_num_gates_per_level[i];
  // } printf("debug_ii = %ld\n", debug_ii);

  // find the _max_gate_size_of_levels
  // _max_gate_size_of_levels = *std::max_element(_num_gates_per_level.begin(), _num_gates_per_level.end());
  for (size_t i = 0; i < _max_level + 1; i++) {
    _max_gate_size_of_levels = (_num_gates_per_level[i] > _max_gate_size_of_levels) ? (_num_gates_per_level[i]) : (_max_gate_size_of_levels);
  }

#ifdef FSIM_PARTITION_FFR
  // todo: use "SMART POINTER"  / learn: difference between: "smart pointer," "unique pointer," and "shared pointer".
  // note: in Java, every pointer is a smart poiniter, so after new, we have no need to free the memory 

  _find_split_merge_FFRs();
#endif 


#ifdef FSIM_DEBUG_PRINT_GRAPH
  printf("_max_level = %ld, _max_gate_size_of_levels = %ld\n", _max_level, _max_gate_size_of_levels);
  
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
}

void CUDASimulator::_DFS(std::vector<bool> &visited, size_t gate_idx, std::vector<std::list<size_t>> &bucket_list, size_t depth, std::queue<size_t> &source_queue) {
  // check depth: after adding this gate into this FFR
  if (depth >= depth_constraint_0) { // EXCEED
    source_queue.push(gate_idx);
    return;
  }

  // check width: after adding this gate into this FFR
  size_t width = (depth >= bucket_list.size()) ? (1) : (bucket_list[depth].size()+1);
  if (width >= width_constraint_0) {
    source_queue.push(gate_idx);
    return;
  }
  
  visited[gate_idx] = true;
  if (depth >= bucket_list.size()) {
    std::list<size_t> tmp;
    tmp.push_back(gate_idx);
    bucket_list.push_back(tmp);
  } else {
    bucket_list[depth].push_front(gate_idx);
  }

  for (size_t i = 0; i < _gates[gate_idx]._num_inputs; i++){
    ElementBase *pre_gate = _gates[gate_idx]._inputs[i];
    size_t tmp_idx = (pre_gate->_idx - _num_PIs);
    if ((pre_gate->_type != GateType::PI) && (visited[tmp_idx] == false)) {
      _DFS(visited, tmp_idx, bucket_list, depth+1, source_queue);
    }
  }
}

void CUDASimulator::_find_split_merge_FFRs() { // find fanout-free FFRs
  // check the nodes with inputs larger than 2
  std::vector<size_t> gate_num_outputs;
  gate_num_outputs.resize(_num_inner_gates, 0);
  for (size_t i = 0; i < _num_inner_gates; i++) {
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
      if ((_gates[i]._inputs[j])->_type != GateType::PI) {
        gate_num_outputs[((_gates[i]._inputs[j])->_idx)-_num_PIs]++;
      }
    }
  }

  std::queue<size_t> source_queue; // source node to traverse gates
  for (size_t i = 0; i < gate_num_outputs.size(); i++) {
    if (gate_num_outputs[i] != 1) {
      source_queue.push(i);
    }
  }
  
  // back DFS
  std::vector<bool> visited(_num_inner_gates, false);
  std::vector<std::vector<std::list<size_t>>> bucket_lists_for_all_FFRs;
  while(source_queue.size() != 0 && visited[source_queue.front()] == false) {  
    std::vector<std::list<size_t>> bucket_list; // temparol
    size_t depth = 0;
    _DFS(visited, source_queue.front(), bucket_list, depth, source_queue);
    bucket_lists_for_all_FFRs.push_back(bucket_list);
    source_queue.pop();
  }

#ifdef FSIM_CHECK_FFR
  // check the queue utilization
  size_t aaccum = 0;
  for (size_t i = 0; i < _num_inner_gates; i++)
    aaccum += visited[i];
  bool answer = (aaccum == _num_inner_gates);
  printf("aaccum = %ld, #gates = %ld, ", aaccum, _num_inner_gates);
  printf("answer: %s", answer ? "True\n" : "False QAQ\n");
#endif
  
  // map gate to its FFR
  std::vector<size_t> gate_to_FFR; gate_to_FFR.resize(_num_inner_gates, 0);
  for (size_t i = 0; i < bucket_lists_for_all_FFRs.size(); i++) { // which FFR
    for (size_t j = 0; j < bucket_lists_for_all_FFRs[i].size(); j++) { // which level in the FFR
      for (auto it = bucket_lists_for_all_FFRs[i][j].begin(); it != bucket_lists_for_all_FFRs[i][j].end(); ++it) {
        gate_to_FFR[*it] = i; // which FFR
      }
    }
  }
  
#ifdef FSIM_CHECK_FFR
  printf("Fanout-Free Regions (FFRs = %ld): \n", bucket_lists_for_all_FFRs.size()); 

  printf("\nCheck FFR bucket_list:\n");
  for (size_t i = 0; i < bucket_lists_for_all_FFRs.size(); i++) {
    printf("Region[%ld] (size = %ld):\n", i, bucket_lists_for_all_FFRs[i].size());
    for (size_t j = 0; j < bucket_lists_for_all_FFRs[i].size(); j++) {
      printf("level %ld: ", j);
      for (auto it = bucket_lists_for_all_FFRs[i][j].begin(); it != bucket_lists_for_all_FFRs[i][j].end(); ++it) {
        printf("%ld, ", (*it)+_num_PIs);
      } printf("\n");
    } printf("\n");
  } printf("\n\n");

#endif 

  // levelized the FFRs
  // forward traversal of _gates (according to the topological sort results of gates) 
  // help to assign the FFR's level
  _largest_FFR_level = 0; // MAX number of level for FFRs
  std::vector<size_t> level_of_FFRs; // the level of each FFR 
  level_of_FFRs.resize(bucket_lists_for_all_FFRs.size(), 0);
  // traverse all gates
  size_t accum_idx = 0;
  // level_0: PI, level_last: PO
  for (size_t i = 1; i < (_size_of_num_gates_per_level-1); i++) {
    for (size_t j = 0; j < _num_gates_per_level[i]; j++) {
      size_t now_gate_idx = accum_idx+j;
      size_t FFR_idx = gate_to_FFR[now_gate_idx];
      for (size_t pg = 0; pg < _gates[now_gate_idx]._num_inputs; pg++) {
        if ((_gates[now_gate_idx]._inputs[pg])->_type != GateType::PI){
          size_t pre_gate_FFR_idx = gate_to_FFR[((_gates[now_gate_idx]._inputs[pg])->_idx) - _num_PIs];
          if (pre_gate_FFR_idx != FFR_idx) { // not in the same FFR
            size_t ret_FFR_level = level_of_FFRs[pre_gate_FFR_idx]+1; // next level of pregate's FFR's level
            level_of_FFRs[FFR_idx] = (ret_FFR_level > level_of_FFRs[FFR_idx]) ? (ret_FFR_level) : (level_of_FFRs[FFR_idx]);
            _largest_FFR_level = (ret_FFR_level > _largest_FFR_level) ? (ret_FFR_level) : (_largest_FFR_level);
          }
        }
      }
    }
    accum_idx += _num_gates_per_level[i];
  }
  _largest_FFR_level++;

#ifdef FSIM_CHECK_FFR
  printf("_largest_FFR_level = %ld\n", _largest_FFR_level);
#endif

  FFR_levelization.resize(_largest_FFR_level);
  for (size_t i = 0; i < level_of_FFRs.size(); i++) {
    FFR_levelization[level_of_FFRs[i]].push_back(i);
  }

#ifdef FSIM_CHECK_FFR
  printf("Check level_of_FFRs\n");
  for (size_t i = 0; i < level_of_FFRs.size(); i++) {
    printf("FFR_%ld, level = %ld\n", i, level_of_FFRs[i]);
  }
#endif 

  // vector: each level
  // vector: each depth 
  // queue: each FFR idx
  // FFR_levelization_with_depth: the candidate FFRs on the same level
  // size (deep) as key sorted into differnt queue for the smae level
  std::vector<std::vector<std::queue<size_t>>> FFR_levelization_with_depth; 
  // init
  FFR_levelization_with_depth.resize(_largest_FFR_level);
  for (size_t i = 0; i < _largest_FFR_level; i++) {
    FFR_levelization_with_depth[i].resize(depth_constraint_0);
  }
  // add item
  for (size_t i = 0; i < level_of_FFRs.size(); i++) {
    // level, depth 
    size_t lvl = level_of_FFRs[i];
    size_t depth = (bucket_lists_for_all_FFRs[i].size())-1;
    FFR_levelization_with_depth[lvl][depth].push(i);
  }


#ifdef FSIM_CHECK_FFR
  printf("FFR_levelization:\n");
  for (size_t i = 0; i < _largest_FFR_level; i++) {
    printf("level_%ld: ", i);
    for (auto it = FFR_levelization[i].begin(); it != FFR_levelization[i].end(); ++it) {
      printf("%ld, ", (*it));
    } printf("\n");
  }

  printf("\nCheck the FFR_levelization_with_depth:\n");
  std::queue<size_t> tmp;
  for (size_t i = 0; i < FFR_levelization_with_depth.size(); i++) {
    printf("Level_%ld\n", i);
    for (size_t j = 0; j < FFR_levelization_with_depth[i].size(); j++) {
      printf("lvl %ld: ", j);
      while (!FFR_levelization_with_depth[i][j].empty()) {
        size_t element = FFR_levelization_with_depth[i][j].front(); // Access the front element
        FFR_levelization_with_depth[i][j].pop(); // Remove the front element
        tmp.push(element);
        printf("%ld, ", element);
      } printf("\n");
      while (!tmp.empty()) {
        size_t element = tmp.front();
        tmp.pop();
        FFR_levelization_with_depth[i][j].push(element);
      }
    } printf("\n\n");
  }
#endif
  // compute number of FFRs in each level 
  num_FFRs_in_this_level.resize(_largest_FFR_level, 0);
  for (size_t level = 0; level < _largest_FFR_level; level++) {
    for (size_t d = 0; d < depth_constraint_0; d++) {
      num_FFRs_in_this_level[level] += FFR_levelization_with_depth[level][d].size();
    }
  }

  // merge the FFR according to the level, merging the FFRs on the same level
  // vector: level, 
  merged_FFRs.resize(_largest_FFR_level);
  for (size_t level = 0; level < _largest_FFR_level; level++) { // all levels
    // printf("level = %zu\n", level);

    size_t num_merged_FFRs = 0;
    std::vector<std::vector<size_t>> merged_FFR; merged_FFR.resize(depth_constraint_0);
    while (num_FFRs_in_this_level[level] > num_merged_FFRs) {
      
      // printf("num_FFRs_in_this_level[%ld] = %ld (w/ num_merged_FFRs = %ld)\n", level, num_FFRs_in_this_level[level], num_merged_FFRs);
      bool merged_FFR_full = false; 
      size_t debug_check_FFR = 0;
      // update, pick a new candidate FFR to merge
      for (int sl = (depth_constraint_0-1); sl > -1 ; sl--) { // pick from the largest FFR
        // printf("level = %zu, sl = %zu\n", level, sl);
        size_t num_FFRs_in_this_sub_level = FFR_levelization_with_depth[level][sl].size();
        while (num_FFRs_in_this_sub_level--) {
          // printf("sl = %ld, num_FFRs_in_this_sub_level = %ld\n", sl, num_FFRs_in_this_sub_level);
          // pick candidate FFR to merge
          size_t candidate_FFR_idx = FFR_levelization_with_depth[level][sl].front();
          FFR_levelization_with_depth[level][sl].pop();
          // // if valid then merge -> in the same function 
          if (_valid_and_merged_FFR(bucket_lists_for_all_FFRs[candidate_FFR_idx], sl+1, merged_FFR, merged_FFR_full)) {
            num_merged_FFRs++;        
          } else {
            // push_back 
            FFR_levelization_with_depth[level][sl].push(candidate_FFR_idx);
          }
          // test whether the merged box is full 
          if (merged_FFR_full || (num_merged_FFRs == num_FFRs_in_this_level[level])) { // merged_FFR_full or all FFRs have been merged
            // printf("inside merged_FFR_full || (num_merged_FFRs == num_FFRs_in_this_level[level]\n");
            merged_FFRs[level].push_back(merged_FFR);
            debug_check_FFR++;
            merged_FFR.clear(); merged_FFR.resize(depth_constraint_0);
            merged_FFRs_level.push_back(level+1); // since level 0 is PI
            merged_FFR_full = false;
          }
        }
      }
      bool need_to_push_in = false;
      // 至少有一層大於 0 就要 push 出去了
      for (size_t ii = 0; ii < depth_constraint_0; ii++) {
        need_to_push_in |= (merged_FFR[ii].size() > 0);
      }
      // if ((debug_check_FFR == 0)) {
      //   printf("need_to_push_in = %d\n", need_to_push_in);
      //   exit(1);
      // }
      if (need_to_push_in) {
        // printf("Inside here, merged_FFRs[%ld].size() = %ld, merged_FFR.size() = %ld\n", level, merged_FFRs[level].size(), merged_FFR.size());
        merged_FFRs[level].push_back(merged_FFR);
        merged_FFR.clear(); merged_FFR.resize(depth_constraint_0);
        merged_FFRs_level.push_back(level+1); // since level 0 is PI
        merged_FFR_full = false;
      }
    }
  }

  _num_FFRs = 0; 
  for (size_t i = 0; i < _largest_FFR_level; i++) {
    // printf("i = %zu, _num_FFRs = %zu\n", i, _num_FFRs);
    _num_FFRs += merged_FFRs[i].size();
  }
  // printf("_largest_FFR_level = %ld\n", _largest_FFR_level);
  
#ifdef FSIM_CHECK_FFR
  printf("PP _num_FFRs = %ld, merged_FFRs_level.size() = %ld\n", _num_FFRs, merged_FFRs_level.size());
  for (size_t i = 0; i < merged_FFRs_level.size(); i++) {
    printf("merged_FFRs_level[%ld]'s level = %ld\n", i, merged_FFRs_level[i]);
  }
#endif

  // we have re-levelization the merged_FFRs by using merged_FFRs_level
  // find the dependency of FFRs
  merged_FFRs_dependency.resize(_num_FFRs+2);
  
  // map gate to its merged_FFR
  std::vector<size_t> gate_to_merged_FFR; gate_to_merged_FFR.resize((_num_PIs+_num_inner_gates+_num_POs), 0);
  // PIs belong to FFR_0
  // gates
  // printf("_num_PIs+_num_inner_gates+_num_POs = %ld\n", _num_PIs+_num_inner_gates+_num_POs);
  for (size_t lvl = 0; lvl < merged_FFRs.size(); lvl++) { // level
    for (size_t ffr = 0; ffr < merged_FFRs[lvl].size(); ffr++) {
      for (size_t sub_lvl = 0; sub_lvl < merged_FFRs[lvl][ffr].size(); sub_lvl++) {
        for (size_t g = 0; g < merged_FFRs[lvl][ffr][sub_lvl].size(); g++) {
          size_t gate_idx = (merged_FFRs[lvl][ffr][sub_lvl][g])+_num_PIs;
          // printf("gate_idx = %ld, gate_idx+_num_PIs = %ld, lvl+1= %ld\n", gate_idx, gate_idx+_num_PIs, lvl+1);
          gate_to_merged_FFR[gate_idx] = lvl+1; // since the first level node only PIs node
        }
      }
    }
  }
  // POs: belong to FFR_{_num_FFRs+1}
  for (size_t i = 0; i < _num_POs; i++) {
    gate_to_merged_FFR[i+_num_PIs+_num_inner_gates] = _num_FFRs+1; // the last FFR
  }
    
#ifdef FSIM_CHECK_FFR
  printf("gate_to_merged_FFR:\n");
  for (size_t i = 0; i < gate_to_merged_FFR.size(); i++) {
    printf("gate_%ld's new level = %ld\n", i, gate_to_merged_FFR[i]);
  } printf("\n");
#endif

  // start to compute the dependency
  // PIs 不會有 inputs 因此可以直接略過
  // gates 會有 input 需要加上 dependency
  for (size_t i = (_num_PIs); i < (_num_PIs+_num_inner_gates); i++) {
    size_t gate_belong_FFR = gate_to_merged_FFR[i];
    for (size_t inp = 0; inp < _gates[i-_num_PIs]._num_inputs; inp++) {
      size_t pre_gate_idx = ((_gates[i-_num_PIs]._inputs[inp])->_idx);
      size_t pre_gate_belong_FFR = gate_to_merged_FFR[pre_gate_idx];
      if (gate_belong_FFR != pre_gate_belong_FFR) {
        merged_FFRs_dependency[gate_belong_FFR].insert(pre_gate_belong_FFR);
        // printf("pre_gate_idx = %ld (FFR_Node_%ld), gate_idx = %ld (FFR_Node_%ld)\n", pre_gate_idx, pre_gate_belong_FFR, i, gate_belong_FFR);
      }
    }
  }
  
  // printf("===\n");
  
  // POs 會有 input 需要加上 dependency
  for (size_t i = (_num_PIs+_num_inner_gates); i < (_num_PIs+_num_inner_gates+_num_POs); i++) {
    size_t gate_belong_FFR = gate_to_merged_FFR[i];
    size_t pre_gate_idx = ((_POs[i-_num_PIs-_num_inner_gates]._inputs[0])->_idx);
    size_t pre_gate_belong_FFR = gate_to_merged_FFR[pre_gate_idx];
    if (gate_belong_FFR != pre_gate_belong_FFR) {
      merged_FFRs_dependency[gate_belong_FFR].insert(pre_gate_belong_FFR);
      // printf("pre_gate_idx = %ld (FFR_Node_%ld), gate_idx = %ld (FFR_Node_%ld)\n", pre_gate_idx, pre_gate_belong_FFR, i, gate_belong_FFR);
    }
  }

  // 0323 compute the #threads and blocks in each block
  max_number_of_gates_of_FFR.resize(_num_FFRs+2);
  size_t accum = 0;

  // PI
  max_number_of_gates_of_FFR[accum] = size_t(int((_num_PIs+31) & (-32))); // round to 32x
  accum++;

  // merged_FFRs
  size_t to_set_max_threads = 0; 
  for (size_t lvl = 0; lvl < merged_FFRs.size(); lvl++) {
    for (size_t ffr = 0; ffr < merged_FFRs[lvl].size(); ffr++) {
      for (int sl = merged_FFRs[lvl][ffr].size()-1; sl > -1 ; sl--) { // sub-level 
          to_set_max_threads = (merged_FFRs[lvl][ffr][sl].size() > to_set_max_threads) 
                            ? (merged_FFRs[lvl][ffr][sl].size()) : (to_set_max_threads);
      }
      max_number_of_gates_of_FFR[accum] = size_t(int((to_set_max_threads+31) & (-32))); // round to 32x
      to_set_max_threads = 0;
      accum++;
    }
  }

  // PO
  max_number_of_gates_of_FFR[accum] = size_t(int((_num_POs+31) & (-32))); // round to 32x
  accum++;

#ifdef FSIM_CHECK_FFR  
  printf("merged_FFRs_dependency:\n");
  for (size_t i = 0; i < _num_FFRs+2; ++i) {
    printf("merged_FFR_%zu dependencies: ", i);
    for (auto it = merged_FFRs_dependency[i].begin(); it != merged_FFRs_dependency[i].end(); ++it) {
        printf("%zu ", *it);
    }
    printf("\n");
  }  

  printf("_num_FFRs+2 = %zu, accum = %zu\n, max_number_of_gates_of_FFR: \n", _num_FFRs+2, accum);
  for (size_t i = 0; i < max_number_of_gates_of_FFR.size(); i++) {
    printf("max_number_of_gates_of_FFR[%zu] = %zu\n", i, max_number_of_gates_of_FFR[i]);
  }

#endif  
#ifdef FSIM_PARTITION_CORRECTNESS
  // only uesd for testing the partition algo
  std::vector<bool> test_merge; test_merge.resize(_num_inner_gates, 0);
  for (size_t level = 0; level < _largest_FFR_level; level++) { // all levels
    for (size_t i = 0; i < merged_FFRs[level].size(); i++) {
      for (size_t sl = 0; sl < depth_constraint_0; sl++) { // each sub-level in the FFRs
        for (size_t gate = 0; gate < merged_FFRs[level][i][sl].size(); gate++) {
          size_t ele = merged_FFRs[level][i][sl][gate];
          test_merge[ele] = true;
        } 
      } 
    } 
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    if (test_merge[i] == false) {
      printf("QQ: %ld\n", i);
    }
  }
  // only uesd for testing the partition algo
#endif
#ifdef FSIM_CHECK_FFR
  printf("Check _valid_and_merged_FFR results\n");
  for (size_t level = 0; level < _largest_FFR_level; level++) { // all levels
    printf("merged_FFRs.size() = %ld\n", merged_FFRs.size());
    for (size_t i = 0; i < merged_FFRs[level].size(); i++) {
      printf("FFR_%ld (merged_FFRs[%ld].size() = %ld):\n", i, level, merged_FFRs[level].size());
      for (size_t sl = 0; sl < depth_constraint_0; sl++) { // each sub-level in the FFRs
        printf("sub_level_%ld: ", sl);
        for (size_t gate = 0; gate < merged_FFRs[level][i][sl].size(); gate++) {
          printf("%ld, ", merged_FFRs[level][i][sl][gate]+_num_PIs);
        } printf("\n");
      } printf("\n");
    } printf("--- \n");
  }
  printf("==============================\n\n\n\n\n");
#endif 
}

bool CUDASimulator::_valid_and_merged_FFR(std::vector<std::list<size_t>> &bucket_lists_for_FFRs,
                                          size_t FFR_level,
                                          std::vector<std::vector<size_t>> &merged_FFR, 
                                          bool &merged_FFR_full) {
  
  
  // check the validation
  // check all possible solutions
  // size_t valid_start = 0, valid_zone_range = 1;
  // printf("Into _valid_and_merged_FFR, FFR_level = %ld\n", FFR_level);
  bool ret = false;
  std::vector<size_t> insert_level;
  int previous_merge_level = -1;
  for (int sl = 0; sl < int(FFR_level); sl++) { // 將 FFR 的 level 塞進去
    ret = false;
    for (int i = previous_merge_level+1; i < (int(depth_constraint_0) - (int(FFR_level) - sl) + 1); i++) {
      // printf("merged_FFR[%ld].size() = %ld, bucket_lists_for_FFRs[sl].size() = %ld\n", i, merged_FFR[i].size(), bucket_lists_for_FFRs[sl].size());
      if ((merged_FFR[i].size() + bucket_lists_for_FFRs[sl].size()) 
                  <= width_constraint_0) { // merged successfully 
        // record which layer
        insert_level.push_back(i);
        previous_merge_level = i;
        ret = true;
        break;
      }
    }
    // if one layer cannot be merged into any level of this merged_FFR
    if (ret == false) {
      return ret;
    }
  }

  // printf("Come to merge\n");
  // merge
  for (size_t i = 0; i < insert_level.size(); i++) {
    while(!bucket_lists_for_FFRs[i].empty()) {
      size_t element = bucket_lists_for_FFRs[i].front();
      bucket_lists_for_FFRs[i].pop_front();
      // printf("element = %ld\n", element);
      merged_FFR[insert_level[i]].push_back(element);
    }
  }

  // check: merged_FFR_full
  merged_FFR_full = true;
  for (size_t i = 0; i < depth_constraint_0; i++) {
    merged_FFR_full = merged_FFR_full && (merged_FFR[i].size() == width_constraint_0);
  }
  // if (merged_FFR_full == true)
  //   printf("in_v merged_FFR_full = %B\n", merged_FFR_full);
  
  return ret;
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
    case Mode::CUDA_GRAPH_PARTITION:
      {
      #ifdef FSIM_DEBUG_PRINT_SIMULATION
        std::cout << "run _CUDA_GRAPH_PARTITION\n";
        size_t m2 = static_cast<size_t>(mode);
        std::cout << "Mode: " << m2 << std::endl; 
      #endif
      _run_parallel_cuda_graph_partition();
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
  
  // // free mem
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
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    
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

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  accum += _num_PIs;
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
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

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  
}



__global__ void _set_params(size_t *_params_gpu_idx_address, 
                            size_t rd, size_t fault_num, size_t bad_case, 
                            size_t max_gate_num_per_level, 
                            size_t fin_num_gates, size_t _num_PIs) {

  _params_gpu_idx_address[0] = rd; // need update
  _params_gpu_idx_address[1] = fault_num; // need update
  _params_gpu_idx_address[2] = bad_case; // need update
  _params_gpu_idx_address[3] = max_gate_num_per_level;
  _params_gpu_idx_address[4] = fin_num_gates;
  _params_gpu_idx_address[5] = _num_PIs;
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
  size_t params_gpu_idx_accum = 0;
  // size_t debug_kk = 0;

  // construct PI node
  params_gpu_idx[0] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PI[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PI[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PI[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PI[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PI[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PI[5] = (void *)&_patterns_gpu;
  kernelArgs_PI[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PI[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PI[8] = (void *)&params_gpu_idx[0];
  
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
  // printf("PI: num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  // debug_kk+=num_threads;

  size_t rd = 0;
  size_t fault_num = 0;
  size_t bad_case = 0;
  size_t max_gate_num_per_level = _num_PIs;
  size_t fin_num_gates = 0;
  
  
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[0], rd, fault_num, bad_case, max_gate_num_per_level, fin_num_gates, _num_PIs);
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
  // void *kernelArgs_gate[7];
  size_t accum = _num_PIs;

  // printf("max_num_level = %ld\n", _max_level);
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    params_gpu_idx_accum += _num_paras;
    params_gpu_idx[i] = _params_gpu+params_gpu_idx_accum;
    kernelArgs_gate[(i-1)*num_func_inputs+0] = (void *)&_gate_inputs_list_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+1] = (void *)&_gate_inputs_list_start_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+2] = (void *)&_pi_gate_po_gate_type_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+3] = (void *)&_pi_gate_po_output_res_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+4] = (void *)&_pi_gate_po_num_inputs_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+5] = (void *)&_patterns_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+6] = (void *)&_fault_gate_idx_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+7] = (void *)&_fault_SA_fault_val_gpu;
    kernelArgs_gate[(i-1)*num_func_inputs+8] = (void *)&params_gpu_idx[i];
    assert(((i-1)*num_func_inputs+8) < num_func_inputs*(_size_of_num_gates_per_level-2));  
    // printf("((i-1)*num_func_inputs+8) = %ld, limit = %ld\n", ((i-1)*num_func_inputs+8), num_func_inputs*(_size_of_num_gates_per_level-2));

    // construct gate node
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    // printf("gateKernel_%ld (_num_gates_per_level[%ld] = %ld): num_blocks = %ld, num_threads = %ld\n", i, i, _num_gates_per_level[i], num_blocks, num_threads);
    // debug_kk+=num_threads;

    max_gate_num_per_level = _num_gates_per_level[i];
    fin_num_gates = accum;
    _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[i], rd, fault_num, bad_case, max_gate_num_per_level, fin_num_gates, _num_PIs);
    cudaDeviceSynchronize();
    cudaCheckErrors("gate _set_params cudaDeviceSynchronize failed");
        
    // Adding nodes
    kernelNode_gate_Params[i-1].func = (void *)_run_gate_graph;
    kernelNode_gate_Params[i-1].gridDim = dim3(num_blocks, 1, 1);
    kernelNode_gate_Params[i-1].blockDim = dim3(num_threads, 1, 1);
    kernelNode_gate_Params[i-1].sharedMemBytes = 0;
    kernelNode_gate_Params[i-1].kernelParams = (void **)&(kernelArgs_gate[(i-1)*num_func_inputs]);
    kernelNode_gate_Params[i-1].extra = NULL;  
    cudaCheckErrors("tmptmp failed");
    assert((i-1) < _size_of_num_gates_per_level - 2);
    // printf("i-1 = %ld, _size_of_num_gates_per_level - 2 = %ld\n", i-1, _size_of_num_gates_per_level - 2);
    
    // push node: gate
    cudaGraphAddKernelNode(&kernelNode_gate[i-1], graph, NULL, 0, &kernelNode_gate_Params[i-1]);
    cudaCheckErrors("Adding kernelNode_gate failed");

    accum += _num_gates_per_level[i];
  }

  // construct PO node
  params_gpu_idx_accum += _num_paras;

  params_gpu_idx[_size_of_num_gates_per_level - 1] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PO[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PO[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PO[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PO[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PO[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PO[5] = (void *)&_patterns_gpu;
  kernelArgs_PO[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PO[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PO[8] = (void *)&params_gpu_idx[_size_of_num_gates_per_level - 1]; 
  
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);
  // printf("PO: num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  // debug_kk+=num_threads;
  // printf("debug_kk = %ld, sum = %ld\n", debug_kk, _num_PIs+_num_inner_gates+_num_POs);

  max_gate_num_per_level = _num_POs;
  fin_num_gates = accum;
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[_size_of_num_gates_per_level - 1], rd, fault_num, bad_case, max_gate_num_per_level, fin_num_gates, _num_PIs);
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

  // Construct dependency
  cudaGraphAddDependencies(graph, kernelNode_PI, &kernelNode_gate[0], 1);
  for (size_t i = 0; i < (_size_of_num_gates_per_level - 3); i++) {
    cudaGraphAddDependencies(graph, &kernelNode_gate[i], &kernelNode_gate[i+1], 1);
  }
  cudaGraphAddDependencies(graph, &kernelNode_gate[_size_of_num_gates_per_level-3], kernelNode_PO, 1);


#ifdef CUDA_GRAPH_CHECK
  cudaDeviceSynchronize();
  // print_params_gpu_idx_address <<< 1, 1, 0, stream1 >>> (_params_gpu, _size_of_num_gates_per_level, _num_paras);
  // check nodes
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  cudaGraphGetNodes(graph, nodes, &numNodes);
  cudaCheckErrors("Graph get nodes failed");
  printf("!!! Number of the nodes in the graph = %zu\n", numNodes);
#endif
}

__global__ void _update_params_gpu (size_t *_params_gpu_idx_address, size_t _num_paras, size_t limit_sz, 
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

  // _params_gpu_idx_address[0] = rd; // need update
  // _params_gpu_idx_address[1] = fault_num; // need update
  // _params_gpu_idx_address[2] = bad_case; // need update
  // _params_gpu_idx_address[3] = max_gate_num_per_level;
  // _params_gpu_idx_address[4] = fin_num_gates;
  // _params_gpu_idx_address[5] = _num_PIs;
}

void CUDASimulator::_update_cuda_graph_params(cudaStream_t &stream1, cudaGraphExec_t instance, 
                                              size_t rd, size_t fault_num, size_t bad_case,
                                              cudaGraphNode_t *kernelNode_PI, 
                                              cudaKernelNodeParams *kernelNode_PI_Params,
                                              cudaGraphNode_t *kernelNode_gate, 
                                              cudaKernelNodeParams *kernelNode_gate_Params, 
                                              cudaGraphNode_t *kernelNode_PO, 
                                              cudaKernelNodeParams *kernelNode_PO_Params,
                                              size_t start_idx, size_t end_idx) {


  cudaCheckErrors("so sad failed");
  _update_params_gpu <<< _max_level+1, 1, 0, stream1 >>> (_params_gpu, _num_paras, 
                                                        (_num_paras*(_max_level+1)), 
                                                        rd, fault_num, bad_case);  

  cudaCheckErrors("GPU _update_params_gpu failed");
  cudaDeviceSynchronize();
  cudaCheckErrors("update cudaDeviceSynchronize failed");

  // __host__ cudaGraphExecKernelNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams )
  // update PI node
  // cudaGraphKernelNodeSetParam
  cudaGraphExecKernelNodeSetParams(instance, *kernelNode_PI, kernelNode_PI_Params);  
  cudaCheckErrors("0; cudaGraphExecKernelNodeSetParams failed");
  // cudaGraphKernelNodeSetParams(*kernelNode_PI, kernelNode_PI_Params);  
  // cudaCheckErrors("0; cudaGraphExecKernelNodeSetParams failed");
  
#ifdef FSIM_LEVELIZAED_PARTITION
  // update gate node
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    cudaGraphExecKernelNodeSetParams(instance, kernelNode_gate[i-1], &kernelNode_gate_Params[i-1]);  
    cudaCheckErrors("1; cudaGraphExecKernelNodeSetParams failed");
  }
#endif

#ifdef FSIM_PARTITION_FFR
  for (size_t i = start_idx; i < end_idx; i++) {
    cudaGraphExecKernelNodeSetParams(instance, kernelNode_gate[i], &kernelNode_gate_Params[i]);  
    cudaCheckErrors("1: cudaGraphExecKernelNodeSetParams failed");
  }
#endif

  // update PO node
  cudaGraphExecKernelNodeSetParams(instance, *kernelNode_PO, kernelNode_PO_Params);  
  cudaCheckErrors("2; cudaGraphExecKernelNodeSetParams failed");
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
  // cudaDeviceSynchronize();
  // cudaCheckErrors("cudaDeviceSynchronize failure");
  
  // construct_graph
  cudaGraphNode_t kernelNode_PI;
  cudaKernelNodeParams kernelNode_PI_Params {0};
  cudaGraphNode_t *kernelNode_gate = (cudaGraphNode_t *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaGraphNode_t));
  cudaKernelNodeParams *kernelNode_gate_Params = (cudaKernelNodeParams *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaKernelNodeParams));
  // memset(kernelNode_gate, 0, (_size_of_num_gates_per_level - 2)*sizeof(cudaGraphNode_t));
  // memset(kernelNode_gate_Params, 0, (_size_of_num_gates_per_level - 2)*sizeof(cudaKernelNodeParams));
  cudaGraphNode_t kernelNode_PO;
  cudaKernelNodeParams kernelNode_PO_Params {0};
  
  size_t num_func_inputs = 9;
  void **kernelArgs_PI = (void **)malloc(num_func_inputs*sizeof(void *));
  void **kernelArgs_gate = (void **)malloc(num_func_inputs*(_size_of_num_gates_per_level-2)*sizeof(void *));
  void **kernelArgs_PO = (void **)malloc(num_func_inputs*sizeof(void *));
  size_t **params_gpu_idx = (size_t **)malloc((_size_of_num_gates_per_level)*sizeof(void *));
  memset(params_gpu_idx, 0, (_size_of_num_gates_per_level)*sizeof(void *));


  _construct_cuda_graph(stream1, graph, 
                        &kernelNode_PI, &kernelNode_PI_Params,
                        kernelNode_gate, kernelNode_gate_Params,
                        &kernelNode_PO, &kernelNode_PO_Params, 
                        kernelArgs_PI, kernelArgs_gate, kernelArgs_PO, params_gpu_idx, num_func_inputs);
  
  // printf("debug! %d, %d, %p\n", kernelNode_PI_Params.gridDim.x, kernelNode_PI_Params.blockDim.x, kernelNode_PI_Params.kernelParams[4]);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaCheckErrors("Graph instantiation failed");
  

  size_t fault_num, bad_case;

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "CUDA GRAPH: Run Good case" << std::endl;
#endif
    
    // update to good case parameters
    // wrong_gate = 0;
    fault_num = 0;
    bad_case = 0;   
    // printf("RD = %ld (_num_rounds = %ld)\n", rd, _num_rounds) ;
    // update KernelNodes params (good case)
    cudaCheckErrors("WTF failed");
    _update_cuda_graph_params(stream1, instance, rd, fault_num, bad_case,
                              &kernelNode_PI, &kernelNode_PI_Params,
                              kernelNode_gate, kernelNode_gate_Params,
                              &kernelNode_PO, &kernelNode_PO_Params, 0, _size_of_num_gates_per_level - 2);

    cudaCheckErrors("_update_cuda_graph_params failed");

    _run_good_case_cuda_graph(num_testcases_this_round, stream1, instance);

    
    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_faults; j++) {
      // update to bad case parameters
      // wrong_gate = _faults[j]._gate_with_fault;
      fault_num = j;
      bad_case = 1;
      // printf("RD = %ld, j = %ld\n", rd, j);
      // update KernelNodes params (bad case)
      _update_cuda_graph_params(stream1, instance, rd, fault_num, bad_case,
                                &kernelNode_PI, &kernelNode_PI_Params,
                                kernelNode_gate, kernelNode_gate_Params,
                                &kernelNode_PO, &kernelNode_PO_Params, 0, _size_of_num_gates_per_level - 2);

      _run_bad_case_cuda_graph(num_testcases_this_round, stream1, instance);
      _compare_g_b_results <<< 1, 1, 0, stream1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
    }
  }
  
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


#ifdef TEST_cuGraphExecNodeSetParams 
  test_cuGraphExecNodeSetParams_cpu();
#endif
}

void CUDASimulator::_run_good_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda_graph\n");
#endif  
  
  cudaGraphLaunch(instance, stream1);
  cudaCheckErrors("Graph launch failed (_run_good_case_cuda_graph)");  
  // printf("\n");
  cudaStreamSynchronize(stream1);
  cudaCheckErrors("cudaStreamSynchronize (_run_good_case_cuda_graph) failed");
  // cudaDeviceSynchronize();
  // cudaCheckErrors("cudaDeviceSynchronize (_run_good_case_cuda_graph) failed");

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

void CUDASimulator::_construct_cuda_graph_partition(cudaStream_t stream1, cudaGraph_t graph, 
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
  // printf("insides _construct_cuda_graph_partition\n");
  size_t num_blocks, num_threads;
  size_t params_gpu_idx_accum = 0;

  // construct PI node
  params_gpu_idx[0] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PI[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PI[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PI[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PI[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PI[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PI[5] = (void *)&_patterns_gpu;
  kernelArgs_PI[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PI[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PI[8] = (void *)&params_gpu_idx[0];
  kernelArgs_PI[9] = (void *)&merged_FFRs_gpu;
  kernelArgs_PI[10] = (void *)&merged_FFRs_gpu_idx;

  num_blocks = (max_number_of_gates_of_FFR[0] > 1024) ? ((max_number_of_gates_of_FFR[0] + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (max_number_of_gates_of_FFR[0] > 1024) ? (1024) : (max_number_of_gates_of_FFR[0]); // _num_PIs;

  size_t rd = 0;
  size_t fault_num = 0;
  size_t bad_case = 0;
  size_t max_gate_num_per_level = _num_PIs;
  size_t fin_num_gates = 0;
  size_t kernelNode_idx = 0;

  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[0], rd, fault_num, bad_case, kernelNode_idx, depth_constraint_0, _num_PIs);
  cudaDeviceSynchronize();
  cudaCheckErrors("PI _set_params cudaDeviceSynchronize failed");


  // Adding PI node, as head node of graph
  kernelNode_PI_Params->func = (void *)_run_gate_graph_partition;
  kernelNode_PI_Params->gridDim = dim3(num_blocks, 1, 1);
  kernelNode_PI_Params->blockDim = dim3(num_threads, 1, 1);
  kernelNode_PI_Params->sharedMemBytes = 0;
  kernelNode_PI_Params->kernelParams = (void **)(kernelArgs_PI);
  kernelNode_PI_Params->extra = NULL;  

  // push node: PI
  cudaGraphAddKernelNode(kernelNode_PI, graph, NULL, 0, kernelNode_PI_Params);
  cudaCheckErrors("Adding kernelNode_PI failed");


  // construct FFR nodes
  for (size_t i = 0; i < _num_FFRs; i++) {
    kernelNode_idx++;

    params_gpu_idx_accum += _num_paras;
    params_gpu_idx[kernelNode_idx] = _params_gpu+params_gpu_idx_accum;
    kernelArgs_gate[i*num_func_inputs+0] = (void *)&_gate_inputs_list_gpu;
    kernelArgs_gate[i*num_func_inputs+1] = (void *)&_gate_inputs_list_start_gpu;
    kernelArgs_gate[i*num_func_inputs+2] = (void *)&_pi_gate_po_gate_type_gpu;
    kernelArgs_gate[i*num_func_inputs+3] = (void *)&_pi_gate_po_output_res_gpu;
    kernelArgs_gate[i*num_func_inputs+4] = (void *)&_pi_gate_po_num_inputs_gpu;
    kernelArgs_gate[i*num_func_inputs+5] = (void *)&_patterns_gpu;
    kernelArgs_gate[i*num_func_inputs+6] = (void *)&_fault_gate_idx_gpu;
    kernelArgs_gate[i*num_func_inputs+7] = (void *)&_fault_SA_fault_val_gpu;
    kernelArgs_gate[i*num_func_inputs+8] = (void *)&params_gpu_idx[kernelNode_idx];
    kernelArgs_gate[i*num_func_inputs+9] = (void *)&merged_FFRs_gpu;
    kernelArgs_gate[i*num_func_inputs+10] = (void *)&merged_FFRs_gpu_idx;

    // construct gate node
    num_blocks = (max_number_of_gates_of_FFR[i+1] > 1024) ? ((max_number_of_gates_of_FFR[i+1] + 1024 - 1) >> 10) : (1);
    num_threads = (max_number_of_gates_of_FFR[i+1] > 1024) ? (1024) : (max_number_of_gates_of_FFR[i+1]);
    // num_blocks = 1;
    // num_threads = width_constraint_0;

    _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[kernelNode_idx], rd, fault_num, bad_case, kernelNode_idx, depth_constraint_0, _num_PIs);
    cudaDeviceSynchronize();
    cudaCheckErrors("FFR _set_params cudaDeviceSynchronize failed");

    // Adding nodes
    kernelNode_gate_Params[i].func = (void *)_run_gate_graph_partition;
    kernelNode_gate_Params[i].gridDim = dim3(num_blocks, 1, 1);
    kernelNode_gate_Params[i].blockDim = dim3(num_threads, 1, 1);
    kernelNode_gate_Params[i].sharedMemBytes = 0;
    kernelNode_gate_Params[i].kernelParams = (void **)&(kernelArgs_gate[(i)*num_func_inputs]);
    kernelNode_gate_Params[i].extra = NULL;  

    // push node: gate
    cudaGraphAddKernelNode(&kernelNode_gate[i], graph, NULL, 0, &kernelNode_gate_Params[i]);
    cudaCheckErrors("Adding kernelNode_FFR failed");
  }

  // construct PO node
  kernelNode_idx++;
  assert(kernelNode_idx == (_num_FFRs+1));
  // printf("kernelNode_idx = %zu, _num_FFRs+1 = %zu\n", kernelNode_idx, _num_FFRs+1);

  params_gpu_idx_accum += _num_paras;

  params_gpu_idx[kernelNode_idx] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PO[0] = (void *)&_gate_inputs_list_gpu;
  kernelArgs_PO[1] = (void *)&_gate_inputs_list_start_gpu;
  kernelArgs_PO[2] = (void *)&_pi_gate_po_gate_type_gpu;
  kernelArgs_PO[3] = (void *)&_pi_gate_po_output_res_gpu;
  kernelArgs_PO[4] = (void *)&_pi_gate_po_num_inputs_gpu;
  kernelArgs_PO[5] = (void *)&_patterns_gpu;
  kernelArgs_PO[6] = (void *)&_fault_gate_idx_gpu;
  kernelArgs_PO[7] = (void *)&_fault_SA_fault_val_gpu;
  kernelArgs_PO[8] = (void *)&params_gpu_idx[kernelNode_idx]; 
  kernelArgs_PO[9] = (void *)&merged_FFRs_gpu; 
  kernelArgs_PO[10] = (void *)&merged_FFRs_gpu_idx; 

  num_blocks = (max_number_of_gates_of_FFR[_num_FFRs+1] > 1024) ? ((max_number_of_gates_of_FFR[_num_FFRs+1] + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (max_number_of_gates_of_FFR[_num_FFRs+1] > 1024) ? (1024) : (max_number_of_gates_of_FFR[_num_FFRs+1]); // _num_PIs;
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[kernelNode_idx], rd, fault_num, bad_case, kernelNode_idx, depth_constraint_0, _num_PIs);
  cudaDeviceSynchronize();  
  cudaCheckErrors("PO _set_params cudaDeviceSynchronize failed");

  // Adding PO node, as head node of graph
  kernelNode_PO_Params->func = (void *)_run_gate_graph_partition;
  kernelNode_PO_Params->gridDim = dim3(num_blocks, 1, 1);
  kernelNode_PO_Params->blockDim = dim3(num_threads, 1, 1);
  kernelNode_PO_Params->sharedMemBytes = 0;
  kernelNode_PO_Params->kernelParams = (void **)(kernelArgs_PO);
  kernelNode_PO_Params->extra = NULL;  

  // push node: PO
  cudaGraphAddKernelNode(kernelNode_PO, graph, NULL, 0, kernelNode_PO_Params);
  cudaCheckErrors("Adding kernelNode_PO failed");

  // Construct dependency by using `merged_FFRs_dependency`
  // 他是一個 vector, 存下每個 node 有哪些 previous FFR
  for (size_t i = 0; i < merged_FFRs_dependency.size(); i++) {
    for (auto it = merged_FFRs_dependency[i].begin(); it != merged_FFRs_dependency[i].end(); ++it) {
      size_t pre_FFR_idx = *it;
      if (pre_FFR_idx == 0) { // from PI
        if (i == (_num_FFRs+1)) { // to PO
          cudaGraphAddDependencies(graph, kernelNode_PI, kernelNode_PO, 1);
          cudaCheckErrors("Adding cudaGraphAddDependencies PI -> PO failed");
          // printf("PI (%zu) -> PO (%zu)\n", pre_FFR_idx, i);
        } else { // to FFR
          cudaGraphAddDependencies(graph, kernelNode_PI, &kernelNode_gate[i-1], 1);
          cudaCheckErrors("Adding cudaGraphAddDependencies PI -> FFR failed");
          // printf("PI (%zu) -> FFR_%zu\n", pre_FFR_idx, i);
        }
      } else { // from FFR
        if (i == (_num_FFRs+1)) { // to PO
          cudaGraphAddDependencies(graph, &kernelNode_gate[pre_FFR_idx-1], kernelNode_PO, 1);
          cudaCheckErrors("Adding cudaGraphAddDependencies FFR -> PO failed");
          // printf("FFR_%zu -> PO (%zu)\n", pre_FFR_idx, i);
        } else { // to FFR
          cudaGraphAddDependencies(graph, &kernelNode_gate[pre_FFR_idx-1], &kernelNode_gate[i-1], 1);
          cudaCheckErrors("Adding cudaGraphAddDependencies FFR -> FFR failed");
          // printf("FFR_%zu -> FFR_%zu\n", pre_FFR_idx, i);
        }
      }
    }
  }    

#ifdef CUDA_GRAPH_CHECK
  cudaDeviceSynchronize();
  // print_params_gpu_idx_address <<< 1, 1, 0, stream1 >>> (_params_gpu, _size_of_num_gates_per_level, _num_paras);
  // check nodes
  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  cudaGraphGetNodes(graph, nodes, &numNodes);
  cudaCheckErrors("Graph get nodes failed");
  printf("!!! Number of the nodes in the graph = %zu\n", numNodes);
#endif
}

void CUDASimulator::_run_parallel_cuda_graph_partition() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel _run_parallel_cuda_graph_partition();" << '\n';
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
  cudaGraphNode_t *kernelNode_gate = (cudaGraphNode_t *)malloc((_num_FFRs)*sizeof(cudaGraphNode_t));
  cudaKernelNodeParams *kernelNode_gate_Params = (cudaKernelNodeParams *)malloc((_num_FFRs)*sizeof(cudaKernelNodeParams));
  cudaGraphNode_t kernelNode_PO;
  cudaKernelNodeParams kernelNode_PO_Params {0};
  

  size_t num_func_inputs = 11;
  void **kernelArgs_PI = (void **)malloc(num_func_inputs*sizeof(void *));
  void **kernelArgs_gate = (void **)malloc(num_func_inputs*(_num_FFRs)*sizeof(void *));
  void **kernelArgs_PO = (void **)malloc(num_func_inputs*sizeof(void *));
  size_t **params_gpu_idx = (size_t **)malloc((_num_FFRs+2)*sizeof(void *));
  memset(params_gpu_idx, 0, (_num_FFRs+2)*sizeof(void *));


  _construct_cuda_graph_partition(stream1, graph, 
                        &kernelNode_PI, &kernelNode_PI_Params,
                        kernelNode_gate, kernelNode_gate_Params,
                        &kernelNode_PO, &kernelNode_PO_Params, 
                        kernelArgs_PI, kernelArgs_gate, kernelArgs_PO, params_gpu_idx, num_func_inputs);
  
  
  // printf("debug! %d, %d, %p\n", kernelNode_PI_Params.gridDim.x, kernelNode_PI_Params.blockDim.x, kernelNode_PI_Params.kernelParams[4]);
  
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaCheckErrors("Graph instantiation failed");
  
  size_t fault_num, bad_case;

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "CUDA GRAPH: Run Good case" << std::endl;
#endif
    
    // update to good case parameters
    // wrong_gate = 0;
    fault_num = 0;
    bad_case = 0; 
    // printf("RD = %ld (_num_rounds = %ld)\n", rd, _num_rounds) ;
    // update KernelNodes params (good case)
    _update_cuda_graph_params(stream1, instance, rd, fault_num, bad_case,
                              &kernelNode_PI, &kernelNode_PI_Params,
                              kernelNode_gate, kernelNode_gate_Params,
                              &kernelNode_PO, &kernelNode_PO_Params, 0, _num_FFRs);

    _run_good_case_cuda_graph(num_testcases_this_round, stream1, instance);

    
    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_faults; j++) {
      // update to bad case parameters
      // wrong_gate = _faults[j]._gate_with_fault;
      // fault_val = _faults[j]._gate_SA_fault_val;
      fault_num = j;
      bad_case = 1;
      // printf("RD = %ld, j = %ld\n", rd, j);
      // // update KernelNodes params (bad case)
      _update_cuda_graph_params(stream1, instance, rd, fault_num, bad_case,
                                &kernelNode_PI, &kernelNode_PI_Params,
                                kernelNode_gate, kernelNode_gate_Params,
                                &kernelNode_PO, &kernelNode_PO_Params, 0, _num_FFRs);

      _run_bad_case_cuda_graph(num_testcases_this_round, stream1, instance);
      _compare_g_b_results <<< 1, 1, 0, stream1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
      // printf("-------------------\n");
    } // printf("-------------------\n\n\n");
  }

  // free mem
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

  cudaFree(_gate_idx_to_tpg_order_idx);
  cudaFree(_PIs);
  cudaFree(_gates);
  cudaFree(_POs);
  cudaFree(_num_gates_per_level);
  cudaFree(_faults);
  cudaFree(_patterns);
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

#ifdef FSIM_DEBUG_PRINT_SIMULATION_GPU  
  printf("real_g_idx = %ld (fin_num_gates = %ld), type = %ld, num_inputs = %ld\n", real_g_idx, fin_num_gates, _pi_gate_po_gate_type_gpu[real_g_idx], _pi_gate_po_num_inputs_gpu[real_g_idx]);
  if (SA_fault) {
    printf("SA_fault = %ld, real_g_idx = %ld, wrong_gate = %ld\n", SA_fault, real_g_idx, _fault_gate_idx_gpu[fault_num]);
  } else {
    printf("No fault in this test\n");
  }
#endif
  
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
                                size_t *_params_gpu) {  

  // parameters
  const size_t rd = _params_gpu[0];
  const size_t fault_num = _params_gpu[1];
  const size_t bad_case = _params_gpu[2];
  const size_t max_gate_num_per_level = _params_gpu[3];
  const size_t fin_num_gates = _params_gpu[4];
  const size_t _num_PIs = _params_gpu[5];

  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t real_g_idx = blockDim.x * b_idx + t_idx + fin_num_gates; // tmp

  if ((max_gate_num_per_level+fin_num_gates) > real_g_idx) {
    size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
    if (SA_fault) {
      _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
    } else {
      size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
      size_t num_inputs = _pi_gate_po_num_inputs_gpu[real_g_idx];
      // if (real_g_idx > 763) 
      //   printf("real_g_idx = %ld, num_inputs = %ld\n", real_g_idx, num_inputs);

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
  }
  __syncthreads();
}

__global__ void _run_gate_graph_partition(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
                                          size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
                                          size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
                                          size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
                                          size_t *_params_gpu,
                                          size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx) {
    
  // parameters
  const size_t rd = _params_gpu[0];
  const size_t fault_num = _params_gpu[1];
  const size_t bad_case = _params_gpu[2];
  const size_t kernelNode_idx = _params_gpu[3]; // note: this includes PI, FFRs, PO 
  const size_t depth_constraint_0 = _params_gpu[4]; // means the dpeth of the FFRs, this would be different from FFRs
  const size_t _num_PIs = _params_gpu[5];

  // size_t t_idx = threadIdx.x;
  // size_t b_idx = blockIdx.x;
  size_t tb_idx = blockIdx.x*blockDim.x + threadIdx.x;

  for (size_t sl = 0; sl < depth_constraint_0; sl++) {
    // 要記錄的是：
    // 1. 你是第幾個 FFR -> 協助得知：你要從 merged_FFRs_gpu 的哪個位置開始取 gate_idx
    // 2. 每個 FFR 每層 有幾個人
    size_t idx_idx = kernelNode_idx*depth_constraint_0 + sl;
    size_t start_gate_idx = merged_FFRs_gpu_idx[idx_idx];
    size_t num_gates_in_this_sublevel = merged_FFRs_gpu_idx[idx_idx+1] - merged_FFRs_gpu_idx[idx_idx];

    if (num_gates_in_this_sublevel > tb_idx) {
      size_t real_g_idx = merged_FFRs_gpu[start_gate_idx+tb_idx];
      size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
      // if (tb_idx == 0)
      //   printf("sl = %ld, tb_idx = %ld (real_g_idx = %ld), idx_idx = %ld, start_gate_idx = %ld, num_gates_in_this_sublevel = %ld\n", sl, t_idx, real_g_idx, idx_idx, start_gate_idx, num_gates_in_this_sublevel);

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
          case 11:
          break;
        }
      }
    }
    // __syncwarp();
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
  // printf("sum_pi_gates_pos = %ld, sum_inputs_gates = %ld\n", sum_pi_gates_pos, sum_inputs_gates);

  cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, (sum_pi_gates_pos)*sizeof(size_t));
  cudaMalloc((void**)&_pi_gate_po_output_res_gpu, (sum_pi_gates_pos)*sizeof(size_t));
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
  
  cudaMalloc((void**)&_gate_inputs_list_gpu, sum_inputs_gates*sizeof(size_t));
  cudaMalloc((void**)&_gate_inputs_list_start_gpu, sum_pi_gates_pos*sizeof(size_t));

#ifdef FSIM_LEVELIZAED_PARTITION
  cudaMalloc((void**)&_params_gpu, _num_paras*(_max_level+1)*sizeof(size_t));
#endif
#ifdef FSIM_PARTITION_FFR
  cudaMalloc((void**)&_params_gpu, _num_paras*(_num_FFRs+2)*sizeof(size_t));
#endif 

  // copy memory 
  cudaMemcpy(_gate_inputs_list_gpu, _gate_inputs_list, sum_inputs_gates*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_gate_inputs_list_start_gpu, _gate_inputs_list_start, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("_gate_inputs_list_start_gpu failure");

  // prepare data for CSR format,
  // todo: move this part to the _read -> replace the origin C++ code
  // pi_gate_po_output_res_gpu // memset: 0 -> can skip
  
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
  
  // _fault_gate_idx_gpu
  // _fault_SA_fault_val_gpu
  size_t *fault_gate_idx_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
  size_t *fault_SA_fault_val_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
  for (size_t i = 0; i < _num_faults; i++) {
    fault_gate_idx_cpu[i]     = _faults[i]._gate_with_fault;
    fault_SA_fault_val_cpu[i] = _faults[i]._gate_SA_fault_val;
    // printf("_faults[i]._gate_with_fault = %ld\n", _faults[i]._gate_with_fault);
  }
  cudaCheckErrors("AA_ failure");

  cudaMemcpy(_pi_gate_po_gate_type_gpu, pi_gate_po_gate_type, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("AA0- failure");
  cudaMemcpy(_pi_gate_po_num_inputs_gpu, pi_gate_po_num_inputs, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("AA1 failure");
  cudaMemcpy(_patterns_gpu, _patterns_cpu, (_num_rounds*_num_PIs)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("AA2 failure");
  cudaMemcpy(_fault_gate_idx_gpu, fault_gate_idx_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("AA3 failure");
  cudaMemcpy(_fault_SA_fault_val_gpu, fault_SA_fault_val_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("AA4 failure");
  
  free(pi_gate_po_gate_type);
  free(pi_gate_po_num_inputs);  
  free(_patterns_cpu);
  free(fault_gate_idx_cpu);
  free(fault_SA_fault_val_cpu);
  


#ifdef FSIM_DEBUG_GPU_INPUTS
  // check 
  // _pi_gate_po_gate_type_gpu
  print_pi_gate_po_gate_type_gpu <<< 1, 1 >>> (_pi_gate_po_gate_type_gpu, _pi_gate_po_num_inputs_gpu, sum_pi_gates_pos);

  // _patterns_gpu
  print_patterns_gpu <<< 1, 1 >>> (_patterns_gpu, _num_rounds, _num_PIs);

  // _fault_gate_idx_gpu
  print_fault_gate_idx_gpu <<< 1, 1 >>> (_fault_gate_idx_gpu, _fault_SA_fault_val_gpu, _num_faults);
  // 
#endif

#ifdef FSIM_PARTITION_FFR
  _gpu_ask_copy_mem_partition();
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
__global__ void print_fault_gate_idx_gpu(size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, size_t _num_faults) {
  printf("print_fault_gate_idx_gpu (_num_faults = %ld): [\n", _num_faults);
  for (size_t i = 0; i < _num_faults; i++) {
    printf("gate_w_f = %ld, SA_val = %ld\n", _fault_gate_idx_gpu[i], _fault_SA_fault_val_gpu[i]);
  } printf("]\n");  
}




void CUDASimulator::_gpu_ask_copy_mem_partition() {
  size_t sum_pi_gates_pos = _num_PIs+_num_inner_gates+_num_POs;

  merged_FFRs_c = (size_t*)malloc((sum_pi_gates_pos)*sizeof(size_t));
  merged_FFRs_idx_c = (size_t*)malloc((1+(_num_FFRs+2)*depth_constraint_0)*sizeof(size_t));
  memset(merged_FFRs_c, 0, (sum_pi_gates_pos)*sizeof(size_t));
  memset(merged_FFRs_idx_c, 0, (1+(_num_FFRs+2)*depth_constraint_0)*sizeof(size_t));
  
  // set memory 
  // merged_FFRs_c
  size_t accum_idx = 0;
  // PIs
  for (size_t i = 0; i < _num_PIs; i++) {
    merged_FFRs_c[accum_idx] = _PIs[i]._idx;
    // printf("PI: merged_FFRs_c[%ld] = %ld\n", accum_idx, merged_FFRs_c[accum_idx]);
    accum_idx++;
  }  
  // merged_FFRs
  for (size_t lvl = 0; lvl < merged_FFRs.size(); lvl++) {
    for (size_t ffr = 0; ffr < merged_FFRs[lvl].size(); ffr++) {
      for (int sl = merged_FFRs[lvl][ffr].size()-1; sl > -1 ; sl--) { // sub-level 
        for (size_t gate = 0; gate < merged_FFRs[lvl][ffr][sl].size(); gate++) {
          merged_FFRs_c[accum_idx] = (merged_FFRs[lvl][ffr][sl][gate]+_num_PIs);
          // printf("GA: merged_FFRs_c[%ld] = %ld\n", accum_idx, merged_FFRs_c[accum_idx]);
          assert(accum_idx < (_num_inner_gates+_num_PIs));
          assert(merged_FFRs[lvl][ffr][sl][gate] < (_num_inner_gates+_num_PIs));
          accum_idx++;
        }
      }
    }
  }
  // POs
  for (size_t i = 0; i < _num_POs; i++) {
    merged_FFRs_c[accum_idx] = _POs[i]._idx;
    // printf("PO: merged_FFRs_c[%ld] = %ld\n", accum_idx, merged_FFRs_c[accum_idx]);
    accum_idx++;
  }

#ifdef FSIM_CHECK_FFR  
  printf("(1+(_num_FFRs+2)*depth_constraint_0) = %ld, _num_FFRs = %ld\n", (1+(_num_FFRs+2)*depth_constraint_0), _num_FFRs);
  printf("merged_FFRs_c : \n");
  for (size_t i = 0; i < (sum_pi_gates_pos); i++) {
    printf("%ld, ", merged_FFRs_c[i]);
  } printf("\n");
  assert(sum_pi_gates_pos == accum_idx);
#endif  

  // ----
  // // merged_FFRs_idx_c
  size_t *tmp = (size_t*)malloc((((_num_FFRs+2)*depth_constraint_0))*sizeof(size_t));
  memset(tmp, 0, (((_num_FFRs+2)*depth_constraint_0))*sizeof(size_t));
  accum_idx = 0;
  // PI: 
  tmp[accum_idx] = _num_PIs;
  accum_idx++;
  for (size_t i = 1; i < depth_constraint_0; i++) {
    tmp[accum_idx] = 0;
    accum_idx++;
  }
  // merged_FFRs
  for (size_t lvl = 0; lvl < merged_FFRs.size(); lvl++) {
    for (size_t ffr = 0; ffr < merged_FFRs[lvl].size(); ffr++) {
      for (int sl = merged_FFRs[lvl][ffr].size()-1; sl > -1 ; sl--) { // sub-level 
        tmp[accum_idx] = merged_FFRs[lvl][ffr][sl].size();
        accum_idx++;
      }
    }
  }  
  // POs
  tmp[accum_idx] = _num_POs;
  accum_idx++;
  for (size_t i = 1; i < depth_constraint_0; i++) {
    tmp[accum_idx] = 0;
    accum_idx++;
  }

#ifdef FSIM_CHECK_FFR
  printf("tmp: ");
  for (size_t i = 0; i < ((_num_FFRs+2)*depth_constraint_0); i++) {
    printf("%ld, ", tmp[i]);
  } printf("\n");
#endif

  for (size_t i = 0; i < ((_num_FFRs+2)*depth_constraint_0); i++) {
    merged_FFRs_idx_c[i+1] = tmp[i] + merged_FFRs_idx_c[i];
    // printf("pre = %ld, tmp = %ld, sum = %ld\n", merged_FFRs_idx_c[i], tmp[i], merged_FFRs_idx_c[i+1]);
  }

#ifdef FSIM_CHECK_FFR
  printf("merged_FFRs_idx_c :\n");
  for (size_t i = 0; i < (1+(_num_FFRs+2)*depth_constraint_0); i++) {
    printf("%ld, ", merged_FFRs_idx_c[i]);
  } printf("\n");
#endif  

  // printf("accum_idx = %ld, _num_FFRs*(depth_constraint_0) = %ld\n", accum_idx, _num_FFRs*(depth_constraint_0));
  cudaMalloc((void**)&merged_FFRs_gpu, sum_pi_gates_pos*sizeof(size_t));
  cudaCheckErrors("cudaMalloc failed");
  assert(merged_FFRs_gpu!=NULL);
  cudaMemcpy(merged_FFRs_gpu, merged_FFRs_c, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy failed");
  

  cudaMalloc((void**)&merged_FFRs_gpu_idx, (1+(_num_FFRs+2)*depth_constraint_0)*sizeof(size_t));
  cudaCheckErrors("cudaMalloc failed");
  assert(merged_FFRs_gpu_idx!=NULL);
  cudaMemcpy(merged_FFRs_gpu_idx, merged_FFRs_idx_c, (1+(_num_FFRs+2)*depth_constraint_0)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy failed");
  
#ifdef FSIM_CHECK_FFR
  print_merged_FFRs_gpu <<< 1, 1 >>> (merged_FFRs_gpu, merged_FFRs_gpu_idx, sum_pi_gates_pos, 1+(_num_FFRs+2)*depth_constraint_0);
#endif

  cudaDeviceSynchronize();
  free(merged_FFRs_c);
  free(merged_FFRs_idx_c);
}

__global__ void print_merged_FFRs_gpu (size_t *merged_FFRs_gpu, size_t *merged_FFRs_gpu_idx, size_t sum_pi_gates_pos, size_t aa) {
  printf("GPU: merged_FFRs_c : \n");
  for (size_t i = 0; i < (sum_pi_gates_pos); i++) {
    printf("%ld, ", merged_FFRs_gpu[i]);
  } printf("\n");
  
  printf("GPU: merged_FFRs_idx_c : \n");
  for (size_t i = 0; i < aa; i++) {
    printf("%ld, ", merged_FFRs_gpu_idx[i]);
  } printf("\n");
}

void CUDASimulator::_free_all() {  
  // 03/20
  cudaFree(_pi_gate_po_gate_type_gpu);
  cudaFree(_pi_gate_po_output_res_gpu);

  cudaFree(_patterns_gpu);
  cudaFree(_patterns_gpu);

  cudaFree(_fault_gate_idx_gpu);
  cudaFree(_fault_SA_fault_val_gpu);
  // 03/20

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
  
#ifdef FSIM_PARTITION_FFR
  cudaFree(merged_FFRs_gpu);
  cudaFree(merged_FFRs_gpu_idx);
#endif 
}




#ifdef TEST_cuGraphExecNodeSetParams
  __global__ void _set_params_test(size_t *_params_gpu_tmptmp, size_t num, size_t rd){
    for (size_t i = 0; i < num; i++) {
      _params_gpu_tmptmp[i] = rd;
    }
  }

  __global__ void test_cuGraphExecNodeSetParams_gpu (size_t *_params_gpu_tmptmp, size_t num, size_t rd) {
    printf("GPU: ");
    for (size_t i = 0; i < num; i++) {
      printf("%ld, ", _params_gpu_tmptmp[i]);
    } printf("\n");
  }

  void CUDASimulator::test_cuGraphExecNodeSetParams_cpu () {
    // construct graph
    cudaStream_t stream1;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaGraph_t graph; // main graph
    cudaGraphCreate(&graph, 0); // create the graph
    cudaCheckErrors("cudaGraphCreate failure");
    
    size_t num = 10;
    size_t rd = 100;
    
    // asking for memory 
    cudaMalloc((void**)&_params_gpu_tmptmp, num*sizeof(size_t));

    void *kernelArgs[3];
    kernelArgs[0] = (void *)&_params_gpu_tmptmp;
    kernelArgs[1] = (void *)&num;
    kernelArgs[2] = (void *)&rd;

    
    cudaGraphNode_t kernelNode_test;
    cudaKernelNodeParams kernelNode_test_Params {0};
    size_t num_blocks = 1;
    size_t num_threads = 1;
    _set_params_test <<< 1, 1, 0, stream1 >>> (_params_gpu_tmptmp, num, rd);
    // cudaStreamSynchronize(stream1);    
    cudaDeviceSynchronize();

    // Adding node
    kernelNode_test_Params.func = (void *)test_cuGraphExecNodeSetParams_gpu;
    kernelNode_test_Params.gridDim = dim3(num_blocks, 1, 1);
    kernelNode_test_Params.blockDim = dim3(num_threads, 1, 1);
    kernelNode_test_Params.sharedMemBytes = 0;
    kernelNode_test_Params.kernelParams = (void **)(kernelArgs);
    kernelNode_test_Params.extra = NULL;  

    // push node
    cudaGraphAddKernelNode(&kernelNode_test, graph, NULL, 0, &kernelNode_test_Params);
    // __host__ ​cudaError_t cudaGraphAddKernelNode ( cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const cudaKernelNodeParams* pNodeParams )

    cudaCheckErrors("Adding kernelNode_test failed");

    // Instantiate cudaGraph
    cudaGraphExec_t instance;
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    cudaCheckErrors("Graph instantiation failed");

    // changed rd
    for (size_t rd_cpu = 0; rd_cpu < 5; rd_cpu++) {
      printf("CPU: rd = %ld\n", rd_cpu);
      rd = rd_cpu;
      
      // change var
      _set_params_test <<< 1, 1, 0, stream1 >>> (_params_gpu_tmptmp, num, rd);
      cudaStreamSynchronize(stream1);    
      // __host__ ​cudaError_t cudaGraphExecKernelNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams )
      cudaGraphExecKernelNodeSetParams(instance, kernelNode_test, &kernelNode_test_Params);

      cudaGraphLaunch(instance, stream1);
      cudaStreamSynchronize(stream1);    
      cudaCheckErrors("Graph launch failed");
      printf("======\n\n");
      cudaDeviceSynchronize();  
    }

  }
#endif  










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
  for (size_t i = 0; i < _size_of_num_gates_per_level; i++) {
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