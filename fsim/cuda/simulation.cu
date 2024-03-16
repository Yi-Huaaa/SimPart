#include <stdio.h>

#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <unordered_set>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "simulation.cuh"

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
  flst >> _num_fault;
  #ifdef FSIM_DEBUG_PRINT_FAULTS
    std::cout << "Read and construct flst" << std::endl;
    std::cout << _num_fault << std::endl;
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
  cudaMallocManaged(&_gate_idx_to_tpg_order_idx, sum_pi_gates_pos*sizeof(size_t));
  for (size_t i = 0; i < order.size(); i++) {
    _gate_idx_to_tpg_order_idx[order[i]] = i;
  }

  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
         po_idx_accum = (_num_PIs + _num_inner_gates);
  // PIs are stored in the order specified by the `order` array
  // _PIs.resize(_num_PIs);
  cudaMallocManaged(&_PIs, _num_PIs*sizeof(PI));
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
  cudaMallocManaged(&_gates, _num_inner_gates*sizeof(Gate));
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
  cudaMallocManaged(&_POs, _num_POs*sizeof(PO));

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
      cudaMallocManaged(&_POs[gate_idx]._inputs, num_inputs*sizeof(ElementBase*));
    } else { // gate
      size_t gate_idx = _gate_idx_to_tpg_order_idx[i]-gate_idx_accum;
      size_t num_inputs = gate_input_order_table[i][0];
      _gates[gate_idx]._num_inputs = num_inputs;
      cudaMallocManaged(&_gates[gate_idx]._inputs, num_inputs*sizeof(ElementBase*));
    }
  }

  std::vector<size_t> index_accum;
  index_accum.resize(sum_pi_gates_pos);
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
  // _gate_inputs_list_from = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  _gate_inputs_list_start = (size_t*)malloc(sum_pi_gates_pos*sizeof(size_t));
  
  memset(_gate_inputs_list, 0, sum_inputs_gates*sizeof(size_t));
  // memset(_gate_inputs_list_from, 0, sum_inputs_gates*sizeof(size_t));
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

  // construct '_gate_inputs_list' and '_gate_inputs_list_from'
  size_t idx_accum = 0;
  for (size_t i = 0; i < _num_inner_gates; i++) {
    for (size_t j = 0; j < _gates[i]._num_inputs; j++) {
      _gate_inputs_list[idx_accum] = (_gates[i]._inputs[j])->_idx;
      // size_t pi_or_gate = static_cast<size_t>(_gates[i]._inputs[j]->_type);
      // _gate_inputs_list_from[idx_accum] = (pi_or_gate == 9) ? (0) : (1); // 0: from PI, 1: from Gate
      idx_accum++;
    }
  }
  for (size_t i = 0; i < _num_POs; i++) {
    for (size_t j = 0; j < _POs[i]._num_inputs; j++) {
      _gate_inputs_list[idx_accum] = (_POs[i]._inputs[j])->_idx;
      // size_t pi_or_gate = static_cast<size_t>(_POs[i]._inputs[j]->_type);
      // _gate_inputs_list_from[idx_accum] = (pi_or_gate == 9) ? (0) : (1); // 0: from PI, 1: from Gate
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
  cudaMallocManaged(&_num_gates_per_level, (_max_level + 1)*sizeof(size_t));
  _size_of_num_gates_per_level = _max_level + 1;

  for (size_t i = 0; i < _num_PIs; i++) {
    _num_gates_per_level[_PIs[i]._level]++;
  }
  for (size_t i = 0; i < _num_inner_gates; i++) {
    _num_gates_per_level[_gates[i]._level]++;
  }
  for (size_t i = 0; i < _num_POs; i++) {
    _num_gates_per_level[_POs[i]._level]++;
  }
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

#ifdef FSIM_LEVELIZAED_PARTITION
  _levelized_partition();

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
  gate_num_outputs.resize(_num_inner_gates);
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
  std::vector<size_t> gate_to_FFR; gate_to_FFR.resize(_num_inner_gates);
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
  size_t largest_FFR_level = 0; // MAX number of level for FFRs
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
            largest_FFR_level = (ret_FFR_level > largest_FFR_level) ? (ret_FFR_level) : (largest_FFR_level);
          }
        }
      }
    }
    accum_idx += _num_gates_per_level[i];
  }
  largest_FFR_level++;

#ifdef FSIM_CHECK_FFR
  printf("largest_FFR_level = %ld\n", largest_FFR_level);
#endif

  std::vector<std::list<size_t>> FFR_levelization; FFR_levelization.resize(largest_FFR_level);
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
  FFR_levelization_with_depth.resize(largest_FFR_level);
  for (size_t i = 0; i < largest_FFR_level; i++) {
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
  // printf("FFR_levelization:\n");
  // for (size_t i = 0; i < largest_FFR_level; i++) {
  //   printf("level_%ld: ", i);
  //   for (auto it = FFR_levelization[i].begin(); it != FFR_levelization[i].end(); ++it) {
  //     printf("%ld, ", (*it));
  //   } printf("\n");
  // }

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
  std::vector<size_t> num_FFRs_in_this_level; num_FFRs_in_this_level.resize(largest_FFR_level, 0);
  for (size_t level = 0; level < largest_FFR_level; level++) {
    for (size_t d = 0; d < depth_constraint_0; d++) {
      num_FFRs_in_this_level[level] += FFR_levelization_with_depth[level][d].size();
    }
  }

  // merge the FFR according to the level, merging the FFRs on the same level
  // vector: level, 
  std::vector<std::vector<std::vector<std::vector<size_t>>>> merged_FFRs; // used to store FFR, store the gate_idx merged in this FFR
  merged_FFRs.resize(largest_FFR_level);
  for (size_t level = 0; level < largest_FFR_level; level++) { // all levels
    size_t num_merged_FFRs = 0;
    std::vector<std::vector<size_t>> merged_FFR; merged_FFR.resize(depth_constraint_0);
    while (num_FFRs_in_this_level[level] > num_merged_FFRs) {
      
      // printf("num_FFRs_in_this_level[%ld] = %ld (w/ num_merged_FFRs = %ld)\n", level, num_FFRs_in_this_level[level], num_merged_FFRs);
      bool merged_FFR_full = false; 
      // update, pick a new candidate FFR to merge
      for (int sl = (depth_constraint_0-1); sl > -1 ; sl--) { // pick from the largest FFR
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
            // printf("Inside here\n");
            merged_FFRs[level].push_back(merged_FFR);
            merged_FFR.clear(); merged_FFR.resize(depth_constraint_0);
            merged_FFR_full = false;
          }
        }
      }
      if (!merged_FFR.empty()) {
        merged_FFRs[level].push_back(merged_FFR);
        merged_FFR.clear(); merged_FFR.resize(depth_constraint_0);
        merged_FFR_full = false;
      }

    }
    // printf("---\n\n");
  }

#ifdef FSIM_PARTITION_CORRECTNESS
  // only uesd for testing the partition algo
  std::vector<bool> test_merge; test_merge.resize(_num_inner_gates);
  for (size_t level = 0; level < largest_FFR_level; level++) { // all levels
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
  printf("Check num_FFRs_in_this_level\n"); // compute number of FFRs in each level 
  for (size_t level = 0; level < largest_FFR_level; level++) {
    printf("num_FFRs_in_this_level[%ld] = %ld\n", level, num_FFRs_in_this_level[level]);
  }

  printf("Check _valid_and_merged_FFR results\n");
  for (size_t level = 0; level < largest_FFR_level; level++) { // all levels
    printf("merged_FFRs.size() = %ld\n", merged_FFRs.size());
    for (size_t i = 0; i < merged_FFRs[level].size(); i++) {
      printf("FFR_%ld:\n", i);
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
  // _faults.resize(_num_fault);
  cudaMallocManaged(&_faults, _num_fault*sizeof(Fault));
  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
#ifdef FSIM_DEBUG_PRINT_FAULTS
    std::cout << wrong_gate << ", " << fault_value << "\n";
#endif
    _faults[i]._gate_with_fault = _gate_idx_to_tpg_order_idx[wrong_gate];
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<size_t>::max() : (0);
  }

  // todo: sort the faults 

#ifdef FSIM_DEBUG_PRINT_FAULTS
  print_faults(_faults);
#endif
}

void CUDASimulator::_read_pattern(std::istream &ptn) {
  _num_rounds = (_num_pattern + SIZE_T_BITS - 1) / SIZE_T_BITS; // ceiling

  // _patterns.resize(_num_rounds);
  cudaMallocManaged(&_patterns, _num_rounds*sizeof(Pattern));

  for (size_t i = 0; i < _num_rounds; i++) {
    // _patterns[i]._value.resize(_num_PIs);
    cudaMallocManaged(&_patterns[i]._value, _num_PIs*sizeof(size_t));
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

// #ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
//     std::cout << "run _run_parallel_cuda\n";
// #endif
//     _run_parallel_cuda();
// #ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
//   size_t m = static_cast<size_t>(mode);
//   std::cout << "Mode: " << m << std::endl;
//   // print_found_fault_to_pattern(_found_fault_to_pattern);
// #endif

  switch (mode) {
  case Mode::CUDA:
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    std::cout << "run _CUDA\n";
    size_t m = static_cast<size_t>(mode);
    std::cout << "Mode: " << m << std::endl;
#endif
    _run_parallel_cuda();
    break;
  case Mode::CUDA_GRAPH:
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
  std::cout << "run _CUDA_GRAPH\n";
  size_t m = static_cast<size_t>(mode);
  std::cout << "Mode: " << m << std::endl;
#endif  
    _run_parallel_cuda_graph();
    break;

  }
  #ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    size_t m = static_cast<size_t>(mode);
    std::cout << "Mode: " << m << std::endl;
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
    for (size_t j = 0; j < _num_fault; j++) {
      _run_bad_case_cuda(rd, j, num_testcases_this_round);
      _compare_g_b_results <<< 1, 1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
    }
  }
  
  // free mem
  // _free_all();
}

void CUDASimulator::_run_good_case_cuda(const size_t rd, const size_t bits) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda\n");
#endif  
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
  po_idx_accum = (_num_PIs + _num_inner_gates);
  
  size_t num_blocks, num_threads;

// Simulation
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
  // printf("num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, 0, 0, 0, 0, _num_PIs, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 0, pi_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
  
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  size_t accum = 0;
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, 0, 0, 0, 0, _num_gates_per_level[i], _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, accum, gate_idx_accum, _num_PIs, _num_inner_gates, _num_POs);

#ifdef FSIM_DEBUG_PRINT_SIMULATION
    cudaDeviceSynchronize();
    printf("---\n");
#endif      
    accum += _num_gates_per_level[i];
  }

  // POs
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);// _num_POs; //_num_POs;
  _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, 0, 0, 0, 0, _num_POs, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 0, po_idx_accum, _num_PIs, _num_inner_gates, _num_POs);

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  

  // shift and copy answer to the good results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_pi_results_gpu, _PIs_gpu, bits, SIZE_T_BITS);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_gate_results_gpu, _gates_gpu, bits, SIZE_T_BITS);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_po_results_gpu, _POs_gpu, bits, SIZE_T_BITS);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  
}

void CUDASimulator::_run_bad_case_cuda(const size_t rd, const size_t fault_num, const size_t bits) {  
  size_t wrong_gate = _faults[fault_num]._gate_with_fault;
  size_t fault_val = _faults[fault_num]._gate_SA_fault_val;
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("wrong_gate = %ld, fault_val = %ld\n", wrong_gate, fault_val);
#endif
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
  po_idx_accum = (_num_PIs + _num_inner_gates);
  size_t bad_case = 1;
  size_t idx_accum_sa = pi_idx_accum;
  size_t num_blocks, num_threads;

  // fault simulation
  // size_t SA_fault = ((i + pi_idx_accum) == wrong_gate);
  idx_accum_sa = pi_idx_accum;
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
  _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, fault_val, idx_accum_sa, wrong_gate, bad_case, _num_PIs, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 0, pi_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  idx_accum_sa = gate_idx_accum;
  size_t accum = 0;
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
    _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, fault_val, idx_accum_sa, wrong_gate, bad_case, _num_gates_per_level[i], _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, accum, gate_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    cudaDeviceSynchronize();
    printf("---\n");
#endif      
    accum += _num_gates_per_level[i];
  }

  // POs
  idx_accum_sa = po_idx_accum;
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);// _num_POs; //_num_POs;
  _run_gate <<< num_blocks, num_threads >>> (_PIs_gpu, _gates_gpu, _POs_gpu, _patterns_gpu, rd, fault_val, idx_accum_sa, wrong_gate, bad_case, _num_POs, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 0, po_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  

  // shift and copy answer to the good results
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_pi_results_gpu, _PIs_gpu, bits, SIZE_T_BITS);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_gate_results_gpu, _gates_gpu, bits, SIZE_T_BITS);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_po_results_gpu, _POs_gpu, bits, SIZE_T_BITS);
  }

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();  
  printf("---\n");
#endif  
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


__global__ void _set_params(size_t *_params_gpu_idx_address, size_t rd, size_t fault_val, size_t idx_accum_sa, 
  size_t wrong_gate, size_t bad_case, size_t max_gate_num_per_level, 
  size_t idx_accum, size_t pi_g_po_idx_accum, 
  size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {

  _params_gpu_idx_address[0] = rd;
  _params_gpu_idx_address[1] = fault_val;
  _params_gpu_idx_address[2] = idx_accum_sa;
  _params_gpu_idx_address[3] = wrong_gate;
  _params_gpu_idx_address[4] = bad_case;
  _params_gpu_idx_address[5] = max_gate_num_per_level;
  _params_gpu_idx_address[6] = idx_accum;
  _params_gpu_idx_address[7] = pi_g_po_idx_accum;
  _params_gpu_idx_address[8] = _num_PIs;
  _params_gpu_idx_address[9] = _num_inner_gates;
  _params_gpu_idx_address[10] = _num_POs;
}

__global__ void print_params_gpu_idx_address (size_t *_params_gpu_idx_address, const size_t _size_of_num_gates_per_level, size_t idx_accum) {
  for (size_t i = 0; i < (_size_of_num_gates_per_level); i++) {
    printf("level_%ld:\n", i);
    for (size_t j = 0; j < idx_accum; j++) {
      size_t idx = i*idx_accum+j;
      printf("%ld, ", _params_gpu_idx_address[idx]);
    }
    printf("\n");
  }
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
                                        size_t **params_gpu_idx) {
  size_t num_blocks, num_threads;
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
  po_idx_accum = (_num_PIs + _num_inner_gates);
  size_t params_gpu_idx_accum = 0;

  // void *kernelArgs[7];
  params_gpu_idx[0] = _params_gpu+params_gpu_idx_accum;
  kernelArgs_PI[0] = (void *)&_PIs_gpu;
  kernelArgs_PI[1] = (void *)&_gates_gpu; 
  kernelArgs_PI[2] = (void *)&_POs_gpu; 
  kernelArgs_PI[3] = (void *)&_patterns_gpu; 
  kernelArgs_PI[4] = (void *)&params_gpu_idx[0];
  kernelArgs_PI[5] = (void *)&_gate_inputs_list_gpu; 
  kernelArgs_PI[6] = (void *)&_gate_inputs_list_start_gpu; 

  // construct PI node
  // cudaGraphNode_t kernelNode_PI;
  // cudaKernelNodeParams kernelNode_PI_Params {0};
  num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
  num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;

  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[0], 0, 0, 0, 0, 0, _num_PIs, 0, pi_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
  cudaDeviceSynchronize();

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
  size_t accum = 0;
  // cudaGraphNode_t *kernelNode_gate = (cudaGraphNode_t *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaGraphNode_t));
  // cudaKernelNodeParams *kernelNode_gate_Params = (cudaKernelNodeParams *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaKernelNodeParams));

  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    params_gpu_idx_accum += _num_paras;
    params_gpu_idx[i] = _params_gpu+params_gpu_idx_accum;

    kernelArgs_gate[(i-1)*7+0] = (void *)&_PIs_gpu;
    kernelArgs_gate[(i-1)*7+1] = (void *)&_gates_gpu; 
    kernelArgs_gate[(i-1)*7+2] = (void *)&_POs_gpu; 
    kernelArgs_gate[(i-1)*7+3] = (void *)&_patterns_gpu; 
    kernelArgs_gate[(i-1)*7+4] = (void *)&params_gpu_idx[i]; 
    kernelArgs_gate[(i-1)*7+5] = (void *)&_gate_inputs_list_gpu; 
    kernelArgs_gate[(i-1)*7+6] = (void *)&_gate_inputs_list_start_gpu; 
      
    // construct gate node
    num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];

    // todo: this only need to launch 1 time
    _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[i], 0, 0, 0, 0, 0, _num_gates_per_level[i], accum, gate_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
    cudaDeviceSynchronize();
        
    // Adding nodes
    kernelNode_gate_Params[i-1].func = (void *)_run_gate_graph;
    kernelNode_gate_Params[i-1].gridDim = dim3(num_blocks, 1, 1);
    kernelNode_gate_Params[i-1].blockDim = dim3(num_threads, 1, 1);
    kernelNode_gate_Params[i-1].sharedMemBytes = 0;
    kernelNode_gate_Params[i-1].kernelParams = (void **)&(kernelArgs_gate[(i-1)*7]);
    kernelNode_gate_Params[i-1].extra = NULL;  

    // push node: gate
    cudaGraphAddKernelNode(&kernelNode_gate[i-1], graph, NULL, 0, &kernelNode_gate_Params[i-1]);
    cudaCheckErrors("Adding kernelNode_gate failed");

    accum += _num_gates_per_level[i];
  }

  
  // construct PO node
  params_gpu_idx_accum += _num_paras;
  params_gpu_idx[_size_of_num_gates_per_level - 1] = _params_gpu+params_gpu_idx_accum;
  
  kernelArgs_PO[0] = (void *)&_PIs_gpu;
  kernelArgs_PO[1] = (void *)&_gates_gpu; 
  kernelArgs_PO[2] = (void *)&_POs_gpu; 
  kernelArgs_PO[3] = (void *)&_patterns_gpu; 
  kernelArgs_PO[4] = (void *)&params_gpu_idx[_size_of_num_gates_per_level - 1]; 
  kernelArgs_PO[5] = (void *)&_gate_inputs_list_gpu; 
  kernelArgs_PO[6] = (void *)&_gate_inputs_list_start_gpu; 

  // construct PO node
  // cudaGraphNode_t kernelNode_PO;
  // cudaKernelNodeParams kernelNode_PO_Params {0};
  num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
  num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);
  // _params_gpu, rd, 0, 0, 0, 0, _num_PIs, 0, pi_idx_accum, _num_PIs, _num_inner_gates, _num_POs
  _set_params <<< 1, 1, 0, stream1 >>> (params_gpu_idx[_size_of_num_gates_per_level - 1], 0, 0, 0, 0, 0, _num_POs, 0, po_idx_accum, _num_PIs, _num_inner_gates, _num_POs);
  cudaDeviceSynchronize();  

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
  size_t new_rd, size_t new_wrong_gate, size_t new_fault_val, size_t new_bad_case) {
  // _params_gpu_idx_address
  size_t gate_idx = blockIdx.x*blockDim.x + threadIdx.x;
  size_t address_start = gate_idx*_num_paras;

  if (address_start > (limit_sz))
    return;

  _params_gpu_idx_address[address_start+0] = new_rd; // rd
  _params_gpu_idx_address[address_start+1] = new_fault_val; // fault_val
  _params_gpu_idx_address[address_start+3] = new_wrong_gate; // wrong_gate
  _params_gpu_idx_address[address_start+4] = new_bad_case; // bad_case
  // printf("gate_idx = %ld, new_rd = %ld, new_wrong_gate = %ld, new_fault_val = %ld, new_bad_case = %ld\n", gate_idx, new_rd, new_wrong_gate, new_fault_val, new_bad_case);

}

void CUDASimulator::_update_cuda_graph_params(cudaStream_t stream1, cudaGraphExec_t instance, 
                                              size_t rd, size_t wrong_gate, size_t fault_val, size_t bad_case,
                                              cudaGraphNode_t *kernelNode_PI, 
                                              cudaKernelNodeParams *kernelNode_PI_Params,
                                              cudaGraphNode_t *kernelNode_gate, 
                                              cudaKernelNodeParams *kernelNode_gate_Params, 
                                              cudaGraphNode_t *kernelNode_PO, 
                                              cudaKernelNodeParams *kernelNode_PO_Params) {

  
  size_t sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;
  size_t num_blocks = (sum_pi_gates_pos > 1024) ? ((sum_pi_gates_pos + 1024 - 1) >> 10) : (1); // 1;  
  size_t num_threads = (sum_pi_gates_pos > 1024) ? (1024) : (sum_pi_gates_pos); // _num_PIs;
  _update_params_gpu <<< num_blocks, num_threads, 0, stream1 >>> (_params_gpu, _num_paras, 
                                                                  (_num_paras*(_max_level+1)), 
                                                                  rd, wrong_gate, fault_val, bad_case);  

  cudaDeviceSynchronize();

  // __host__ cudaGraphExecKernelNodeSetParams ( cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const cudaKernelNodeParams* pNodeParams )
  // update PI node
  cudaGraphExecKernelNodeSetParams(instance, *kernelNode_PI, kernelNode_PI_Params);  
  cudaCheckErrors("cudaGraphExecKernelNodeSetParams failed");

  // update gate node
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    cudaGraphExecKernelNodeSetParams(instance, kernelNode_gate[i-1], &kernelNode_gate_Params[i-1]);  
    cudaCheckErrors("cudaGraphExecKernelNodeSetParams failed");
  }

  // update PO node
  cudaGraphExecKernelNodeSetParams(instance, *kernelNode_PO, kernelNode_PO_Params);  
  cudaCheckErrors("cudaGraphExecKernelNodeSetParams failed");
}



void CUDASimulator::_run_parallel_cuda_graph() {
  #ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel cuda_graph();" << '\n';
#endif

  // ask memory 
  _gpu_ask_copy_mem();
  
  // construct graph
  cudaStream_t stream1;
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaGraph_t graph; // main graph
  cudaGraphCreate(&graph, 0); // create the graph
  cudaCheckErrors("cudaGraphCreate failure");
  
  // construct_graph
  cudaGraphNode_t kernelNode_PI;
  cudaKernelNodeParams kernelNode_PI_Params {0};
  cudaGraphNode_t *kernelNode_gate = (cudaGraphNode_t *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaGraphNode_t));
  cudaKernelNodeParams *kernelNode_gate_Params = (cudaKernelNodeParams *)malloc((_size_of_num_gates_per_level - 2)*sizeof(cudaKernelNodeParams));
  cudaGraphNode_t kernelNode_PO;
  cudaKernelNodeParams kernelNode_PO_Params {0};
  // void *kernelArgs_PI[7];
  // void *kernelArgs_gate[7]; // 7*num_inner_gate
  // void *kernelArgs_PO[7];
  void **kernelArgs_PI = (void **)malloc(7*sizeof(void *));
  void **kernelArgs_gate = (void **)malloc(7*_num_inner_gates*sizeof(void *));
  void **kernelArgs_PO = (void **)malloc(7*sizeof(void *));
  size_t **params_gpu_idx = (size_t **)malloc((_num_PIs+_num_inner_gates+_num_POs)*sizeof(size_t));


  _construct_cuda_graph(stream1, graph, 
                        &kernelNode_PI, &kernelNode_PI_Params,
                        kernelNode_gate, kernelNode_gate_Params,
                        &kernelNode_PO, &kernelNode_PO_Params, 
                        kernelArgs_PI, kernelArgs_gate, kernelArgs_PO, params_gpu_idx);
  
  // printf("debug! %d, %d, %p\n", kernelNode_PI_Params.gridDim.x, kernelNode_PI_Params.blockDim.x, kernelNode_PI_Params.kernelParams[4]);

  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaCheckErrors("Graph instantiation failed");
  

  size_t wrong_gate, fault_val, bad_case;

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
    wrong_gate = 0;
    fault_val = 0;
    bad_case = 0;    
    // update KernelNodes params (good case)
    _update_cuda_graph_params(stream1, instance, rd, wrong_gate, fault_val, bad_case,
                              &kernelNode_PI, &kernelNode_PI_Params,
                              kernelNode_gate, kernelNode_gate_Params,
                              &kernelNode_PO, &kernelNode_PO_Params);
    
    _run_good_case_cuda_graph(num_testcases_this_round, stream1, instance);

    
    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_fault; j++) {
      // update to bad case parameters
      wrong_gate = _faults[j]._gate_with_fault;
      fault_val = _faults[j]._gate_SA_fault_val;
      bad_case = 1;
      // update KernelNodes params (bad case)
      _update_cuda_graph_params(stream1, instance, rd, wrong_gate, fault_val, bad_case,
                                &kernelNode_PI, &kernelNode_PI_Params,
                                kernelNode_gate, kernelNode_gate_Params,
                                &kernelNode_PO, &kernelNode_PO_Params);

      _run_bad_case_cuda_graph(num_testcases_this_round, stream1, instance);
      _compare_g_b_results <<< 1, 1, 0, stream1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
    }
  }
  
  // // free mem
  _free_all();
  free(kernelNode_gate);
  free(kernelNode_gate_Params);

#ifdef TEST_cuGraphExecNodeSetParams 
  test_cuGraphExecNodeSetParams_cpu();
#endif
}


void CUDASimulator::_run_good_case_cuda_graph(const size_t bits, cudaStream_t stream1, cudaGraphExec_t instance) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda_graph\n");
#endif  
  
  cudaGraphLaunch(instance, stream1);
  cudaStreamSynchronize(stream1);
  
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaCheckErrors("Graph launch failed (_run_good_case_cuda_graph)");  
  cudaDeviceSynchronize();
#endif

  // shift and copy answer to the good results
  size_t num_blocks, num_threads;
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? (_num_PIs >> 10) : (1);
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads, 0, stream1 >>> (_g_pi_results_gpu, _PIs_gpu, bits, SIZE_T_BITS);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates >> 10) : (1); 
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_gate_results_gpu, _gates_gpu, bits, SIZE_T_BITS);

    num_blocks = (_num_POs > 1024) ? (_num_POs >> 10) : (1);
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads, 0, stream1 >>> (_g_po_results_gpu, _POs_gpu, bits, SIZE_T_BITS);
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

  // shift and copy answer to the bad results
  size_t num_blocks, num_threads;
  if (bits < SIZE_T_BITS) {
    num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
    num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_pi_results_gpu, _PIs_gpu, bits, SIZE_T_BITS);
    
    num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
    num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_gate_results_gpu, _gates_gpu, bits, SIZE_T_BITS);

    num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
    num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
    _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_b_po_results_gpu, _POs_gpu, bits, SIZE_T_BITS);
  }
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  cudaDeviceSynchronize();
#endif
}


__global__ void _shift_to_correct_answer_gpu(size_t *outputs, ElementBase *input_arr, const size_t bits, const size_t SIZE_T_BITS) {
  size_t gate_idx = blockDim.x*blockIdx.x + threadIdx.x;
  outputs[gate_idx] =  (input_arr[gate_idx]._output_value << (SIZE_T_BITS - bits));
  outputs[gate_idx] >>= (SIZE_T_BITS - bits);
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("gate_idx = %ld, outputs[%ld] = %ld\n", gate_idx, gate_idx, outputs[gate_idx]);
#endif
}


__global__ void _run_gate(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, const size_t rd, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case, const size_t max_gate_num_per_level, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t idx_accum, size_t pi_g_po_idx_accum, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t gate_idx = blockDim.x * b_idx + t_idx + idx_accum; // tmp
  size_t real_g_idx = (gate_idx+pi_g_po_idx_accum); // used to judge is PI, Gate, PO
  
  // printf("gate_idx = %ld\n", gate_idx);
  // printf("real_g_idx = %ld\n", real_g_idx);
  
  if (gate_idx > (max_gate_num_per_level+idx_accum))
    return;
  Pattern &pattern = _patterns_gpu[rd];

  // idx_accum: pi_idx_accum, gate_idx_accum, po_idx_accum, 
  size_t SA_fault = (((real_g_idx) == wrong_gate) & bad_case);
  // printf("SA_fault = %ld, real_g_idx = %ld, wrong_gate= %ld\n", SA_fault, real_g_idx, wrong_gate);

  ElementBase *gate;
  if (real_g_idx < _num_PIs) { // PI
    gate = &(_PIs_gpu[gate_idx]);
  } else if (real_g_idx < (_num_PIs+_num_inner_gates)) { // Gate
    gate = &(_gates_gpu[gate_idx]);
  } else { // PO
    gate = &(_POs_gpu[gate_idx]);
  }
  
  if (SA_fault) {
    gate[0]._output_value = fault_val;
    return;
  }

  switch (gate[0]._type) { 
    case GateType::INV:
      _apply_INV(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::AND:
      _apply_AND(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::OR:
      _apply_OR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::XOR:
      _apply_XOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::NAND:
      _apply_NAND(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::NOR:
      _apply_NOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::XNOR:
      _apply_XNOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::MUX:
      _apply_MUX(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::CLKBUF:
      _apply_CLKBUF(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::PI:
      _apply_PI(gate[0], pattern, gate_idx, _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::PO:
      _apply_PO(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::MAX_GATE_TYPE:
    break;
    }
}

__global__ void _run_gate_graph(PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, Pattern *_patterns_gpu, size_t *_params_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
  // parameters
  const size_t rd = _params_gpu[0];
  const size_t fault_val = _params_gpu[1];
  const size_t idx_accum_sa = _params_gpu[2];
  const size_t wrong_gate = _params_gpu[3];
  const size_t bad_case = _params_gpu[4];
  const size_t max_gate_num_per_level = _params_gpu[5];
  const size_t idx_accum = _params_gpu[6];
  const size_t pi_g_po_idx_accum = _params_gpu[7];
  const size_t _num_PIs = _params_gpu[8];
  const size_t _num_inner_gates = _params_gpu[9];
  const size_t _num_POs = _params_gpu[10];

  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t gate_idx = blockDim.x * b_idx + t_idx + idx_accum; // tmp
  size_t real_g_idx = (gate_idx+pi_g_po_idx_accum); // used to judge is PI, Gate, PO

  if (gate_idx > (max_gate_num_per_level+idx_accum))
    return;
  Pattern &pattern = _patterns_gpu[rd];

   
  // if (real_g_idx < 3) {
  //   printf("insides _run_gate_graph\n");
  //   for (size_t i = 0; i < 11; i ++) {
  //     printf("_params_gpu[%ld] = %ld (real_g_idx = %ld)\n", i, _params_gpu[i], real_g_idx);
  //   } printf("\n");
  // }
  
  // idx_accum: pi_idx_accum, gate_idx_accum, po_idx_accum, 
  size_t SA_fault = (((real_g_idx) == wrong_gate) & bad_case);

  ElementBase *gate;
  if (real_g_idx < _num_PIs) { // PI
    gate = &(_PIs_gpu[gate_idx]);
  } else if (real_g_idx < (_num_PIs+_num_inner_gates)) { // Gate
    gate = &(_gates_gpu[gate_idx]);
  } else { // PO
    gate = &(_POs_gpu[gate_idx]);
  }
  
  if (SA_fault) {
    gate[0]._output_value = fault_val;
    return;
  }

  switch (gate[0]._type) { 
    case GateType::INV:
      _apply_INV(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::AND:
      _apply_AND(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::OR:
      _apply_OR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::XOR:
      _apply_XOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::NAND:
      _apply_NAND(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::NOR:
      _apply_NOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::XNOR:
      _apply_XNOR(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::MUX:
      _apply_MUX(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::CLKBUF:
      _apply_CLKBUF(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::PI:
      _apply_PI(gate[0], pattern, gate_idx, _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::PO:
      _apply_PO(gate[0], _PIs_gpu, _gates_gpu, _POs_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu, real_g_idx, _num_PIs, _num_inner_gates, _num_POs);
    break;
    case GateType::MAX_GATE_TYPE:
    break;
    }
}

// simulation for gates
__device__ __forceinline__ void _apply_INV(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  // gate idx need to minus _num_PIs
  
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI
  // ElementBase *pre_gate = gate._inputs[0];
  // size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

__device__ __forceinline__ void _apply_AND(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    // now_gate = gate._inputs[i];
    // ret &= (now_gate)->_output_value;
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret &= now_gate_val; 
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_OR(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret |= now_gate_val; 

    // now_gate = gate._inputs[i];
    // ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_XOR(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret ^= now_gate_val; 

    // now_gate = gate._inputs[i];
    // ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_NAND(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret &= now_gate_val; 

    // now_gate = gate._inputs[i];
    // ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_NOR(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI
  
  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret |= now_gate_val; 

    // now_gate = gate._inputs[i];
    // ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_XNOR(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *now_gate = gate._inputs[0];
  // size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    size_t now_gate_idx = _gate_inputs_list_gpu[start_location+i];
    size_t now_gate_val = (now_gate_idx >= (_num_PIs)) ? (_gates_gpu[now_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[now_gate_idx]._output_value); 
    ret ^= now_gate_val; 

    // now_gate = gate._inputs[i];
    // ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_MUX(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  
  size_t a = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI
  pre_gate_idx = _gate_inputs_list_gpu[start_location+1];
  size_t b = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI
  pre_gate_idx = _gate_inputs_list_gpu[start_location+2];
  size_t s = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // size_t a = gate._inputs[0]->_output_value;
  // size_t b = gate._inputs[1]->_output_value;
  // size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | ( a & (!s)));

  gate._output_value = (ret);
}

__device__ __forceinline__ void _apply_CLKBUF(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}


__device__ __forceinline__ void _apply_PI(ElementBase &gate, const Pattern pattern,
                                const size_t pi, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  gate._output_value = pattern._value[pi];
}

__device__ __forceinline__ void _apply_PO(ElementBase &gate, PI *_PIs_gpu, Gate *_gates_gpu, PO *_POs_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, size_t real_g_idx, size_t _num_PIs, size_t _num_inner_gates, size_t _num_POs) {
  // It outputs its previous gate's output_value therefore there is no SA fault
  size_t start_location = _gate_inputs_list_start_gpu[gate._idx];
  size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];
  size_t ret = (pre_gate_idx >= (_num_PIs)) ? (_gates_gpu[pre_gate_idx-_num_PIs]._output_value) : (_PIs_gpu[pre_gate_idx]._output_value); // from Gate, from PI

  // ElementBase *pre_gate = gate._inputs[0];
  gate._output_value = ret;
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



// todo: this can be merged into the __device__code for acceleration
void CUDASimulator::_shift_to_correct_answer(size_t *results, const size_t bits, const size_t num_shift_gates) {
  for (size_t i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
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

  cudaMalloc((void**)&_PIs_gpu, _num_PIs*sizeof(PI));
  cudaMalloc((void**)&_gates_gpu, _num_inner_gates*sizeof(Gate));
  cudaMalloc((void**)&_POs_gpu, _num_POs*sizeof(PO));
  cudaMalloc((void**)&_faults_gpu, _num_fault*sizeof(Fault));
  cudaMalloc((void**)&_patterns_gpu, _num_rounds*sizeof(Pattern));

  cudaMalloc((void**)&_g_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_g_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_g_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_b_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_b_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_b_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_found_fault_to_pattern_gpu, 2*_num_fault*sizeof(size_t));
  
  cudaMalloc((void**)&_gate_inputs_list_gpu, sum_inputs_gates*sizeof(size_t));
  cudaMalloc((void**)&_gate_inputs_list_start_gpu, sum_pi_gates_pos*sizeof(size_t));

  cudaMalloc((void**)&_params_gpu, _num_paras*(_max_level+1)*sizeof(size_t));
  

  // copy memory 
  cudaMemcpy(_PIs_gpu, _PIs, _num_PIs*sizeof(PI), cudaMemcpyHostToDevice);
  cudaMemcpy(_gates_gpu, _gates, _num_inner_gates*sizeof(Gate), cudaMemcpyHostToDevice);
  cudaMemcpy(_POs_gpu, _POs, _num_POs*sizeof(PO), cudaMemcpyHostToDevice);
  cudaMemcpy(_faults_gpu, _faults, _num_fault*sizeof(Fault), cudaMemcpyHostToDevice);
  cudaMemcpy(_patterns_gpu, _patterns, _num_rounds*sizeof(Pattern), cudaMemcpyHostToDevice);
  
  cudaMemcpy(_gate_inputs_list_gpu, _gate_inputs_list, sum_inputs_gates*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemcpy(_gate_inputs_list_start_gpu, _gate_inputs_list_start, sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
}

void CUDASimulator::_free_all() {
  // CPU 
  // free(_gate_idx_to_tpg_order_idx);
  // free(_PIs);
  // free(_POs);
  // free(_num_gates_per_level);
  // free(_faults);
  // free(_patterns);
  // free(_gate_inputs_list);
  // free(_gate_inputs_list_start);
  // free(_found_fault_to_pattern);

  // GPU
  cudaFree(_PIs_gpu);
  cudaFree(_gates_gpu);
  cudaFree(_POs_gpu);
  cudaFree(_faults_gpu);
  cudaFree(_patterns_gpu);

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
  for (size_t i = 0; i < _num_fault; i++) {
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
  for (size_t i = 0; i < (2 * _num_fault); i += 2) {
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