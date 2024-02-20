#include <stdio.h>

#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

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

// cudasimulator fucntions
void CudaSimulator::read(const std::string &ckt_path, const std::string &flst_path, const std::string &ptn_path) {

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

void CudaSimulator::read(std::istream &ckt, std::istream &flst, std::istream &ptn) {
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
void CudaSimulator::_read_graph(std::istream &ckt) {
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

void CudaSimulator::_read_fault(std::istream &flst) {
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

#ifdef FSIM_DEBUG_PRINT_FAULTS
  print_faults(_faults);
#endif
}

void CudaSimulator::_read_pattern(std::istream &ptn) {
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

void CudaSimulator::run(Mode mode, const size_t num_threads) {

#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
    std::cout << "run _run_parallel_cuda\n";
#endif
    _run_parallel_cuda();
#ifdef FSIM_DEBUG_PRINT_FAULT_TABLE
  size_t m = static_cast<size_t>(mode);
  std::cout << "Mode: CUDA\n";
  // print_found_fault_to_pattern(_found_fault_to_pattern);
#endif
}

void CudaSimulator::_run_parallel_cuda() {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_parallel cuda();" << '\n';
#endif

  // // Simulation
  cudaMallocManaged(&_g_pi_results, _num_PIs*sizeof(size_t));
  cudaMallocManaged(&_g_gate_results, _num_inner_gates*sizeof(size_t));
  cudaMallocManaged(&_g_po_results, _num_POs*sizeof(size_t));
  cudaMallocManaged(&_b_pi_results, _num_PIs*sizeof(size_t));
  cudaMallocManaged(&_b_gate_results, _num_inner_gates*sizeof(size_t));
  cudaMallocManaged(&_b_po_results, _num_POs*sizeof(size_t));
  cudaMallocManaged(&_found_fault_to_pattern, 2*_num_fault*sizeof(size_t));

  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);

    // good simulation
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    std::cout << "CUDA: Run Good case" << std::endl;
#endif
    _run_good_case_cuda(_patterns[rd], num_testcases_this_round);

    // bad simulation (fault simulation)
    for (size_t j = 0; j < _num_fault; j++) {
      _run_bad_case_cuda(_faults[j], _patterns[rd], num_testcases_this_round);

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
  }
}


void CudaSimulator::_run_good_case_cuda(const Pattern pattern, const size_t bits) {
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("insides _run_good_case_cuda\n");
#endif  
  // todo: adjustable
  size_t num_blocks, num_threads;

// Simulation
  num_blocks = 1; // 1;
  num_threads = _num_PIs; //_num_PIs;
  _run_gate <<< num_blocks, num_threads >>> (_PIs, pattern, 0, 0, 0, 0);
  cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  size_t accum = 0;
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    num_blocks = 1;
    num_threads = _num_gates_per_level[i];
    _run_gate <<< num_blocks, num_threads >>> (_gates+accum, pattern, 0, 0, 0, 0);
    cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    printf("---\n");
#endif      
    accum += _num_gates_per_level[i];
  }

  // POs
  num_blocks = 1;
  num_threads = _num_POs;
  _run_gate <<< num_blocks, num_threads >>> (_POs, pattern, 0, 0, 0, 0);
  cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("---\n");
#endif  

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

#ifdef FSIM_DEBUG_PRINT_SIMULATION
  std::cout << "GOOD resutls omp:" << std::endl;
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}

void CudaSimulator::_run_bad_case_cuda(const Fault &fault, const Pattern pattern, const size_t bits) {
  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("wrong_gate = %ld, fault_val = %ld\n", wrong_gate, fault_val);
#endif
  size_t pi_idx_accum = 0, gate_idx_accum = _num_PIs,
  po_idx_accum = (_num_PIs + _num_inner_gates);
  size_t bad_case = 1;
  size_t idx_accum_sa = pi_idx_accum;

  // fault simulation
  // size_t SA_fault = ((i + pi_idx_accum) == wrong_gate);
  idx_accum_sa = pi_idx_accum;
  _run_gate <<< 1, _num_PIs >>>(_PIs, pattern, fault_val, idx_accum_sa, wrong_gate, bad_case);
  cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("---\n");
#endif  

  // since the level_0 belongs to PI, the level_MAX belongs to POs
  idx_accum_sa = gate_idx_accum;
  size_t accum = 0;
  for (size_t i = 1; i < (_size_of_num_gates_per_level - 1); i++) {
    // note: here be carefil, the index will be updated by usnig pointer addition!!!
    _run_gate <<< 1, _num_gates_per_level[i] >>>(_gates+accum, pattern, fault_val, idx_accum_sa+accum, wrong_gate, bad_case);
    cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
    printf("---\n");
#endif      
    accum += _num_gates_per_level[i];
  }

  // POs
  idx_accum_sa = po_idx_accum;
  _run_gate <<< 1, _num_POs >>>(_POs, pattern, fault_val, idx_accum_sa, wrong_gate, bad_case);
  cudaDeviceSynchronize();
#ifdef FSIM_DEBUG_PRINT_SIMULATION
  printf("---\n");
#endif  

    // Duplicated the answer of good_case into a new memory
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
    std::cout << "BAD resutls cuda:\n";
    print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
  #endif
}

__global__ void _run_gate(ElementBase *gate, const Pattern pattern, const size_t fault_val, size_t idx_accum_sa, size_t wrong_gate, size_t bad_case) {

  size_t t_idx = threadIdx.x;
  size_t b_idx = blockIdx.x;
  size_t gate_idx = blockDim.x * b_idx + t_idx; // tmp

  // idx_accum: pi_idx_accum, gate_idx_accum, po_idx_accum, 
  size_t SA_fault = (((gate_idx + idx_accum_sa) == wrong_gate) & bad_case);

  if (SA_fault) {
    gate[gate_idx]._output_value = fault_val;
    return;
  }
  // printf("_run_gate: t_idx = %ld, b_idx = %ld, %ld :)))\n", t_idx, b_idx, gate[t_idx]._num_inputs);
  switch (gate[gate_idx]._type) {
    case GateType::INV:
      _apply_INV(gate[gate_idx]);
    break;
    case GateType::AND:
      _apply_AND(gate[gate_idx]);
    break;
    case GateType::OR:
      _apply_OR(gate[gate_idx]);
    break;
    case GateType::XOR:
      _apply_XOR(gate[gate_idx]);
    break;
    case GateType::NAND:
      _apply_NAND(gate[gate_idx]);
    break;
    case GateType::NOR:
      _apply_NOR(gate[gate_idx]);
    break;
    case GateType::XNOR:
      _apply_XNOR(gate[gate_idx]);
    break;
    case GateType::MUX:
      _apply_MUX(gate[gate_idx]);
    break;
    case GateType::CLKBUF:
      _apply_CLKBUF(gate[gate_idx]);
    break;
    case GateType::PI:
      _apply_PI(gate[gate_idx], pattern, gate_idx);
    break;
    case GateType::PO:
      _apply_PO(gate[gate_idx]);
    break;
    case GateType::MAX_GATE_TYPE:
    break;
    }
}

// simulation for gates
__device__ __forceinline__ void _apply_INV(ElementBase &gate) {
  ElementBase *pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

__device__ __forceinline__ void _apply_AND(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_OR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_XOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

__device__ __forceinline__ void _apply_NAND(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_NOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_XNOR(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._num_inputs; i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

__device__ __forceinline__ void _apply_MUX(ElementBase &gate) {
  size_t a = gate._inputs[0]->_output_value;
  size_t b = gate._inputs[1]->_output_value;
  size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | ( a & (!s)));

  gate._output_value = (ret);
}

__device__ __forceinline__ void _apply_CLKBUF(ElementBase &gate) {
  ElementBase *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}


__device__ __forceinline__ void _apply_PI(ElementBase &gate, const Pattern pattern,
                                const size_t pi) {
  gate._output_value = pattern._value[pi];
}

__device__ __forceinline__ void _apply_PO(ElementBase &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  ElementBase *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

// todo: this can be merged into the __device__code for acceleration
void CudaSimulator::_shift_to_correct_answer(size_t *results, const size_t bits, const size_t num_shift_gates) {
  for (size_t i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
}

// output file visualization graph
void CudaSimulator::visualization_graph(const std::string &outputFile) {
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
std::string CudaSimulator::gateTypeToString(GateType type) {
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













// debug: print functions
void CudaSimulator::print_topological(
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

void CudaSimulator::print_gate_input_order_table(std::vector<std::vector<size_t>> input_table) {
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

void CudaSimulator::print_num_gates_per_level() const {
  std::cout << "\n=====\n\n";
  std::cout << "_num_gates_per_level: \n";
  for (size_t i = 0; i < _size_of_num_gates_per_level; i++) {
    std::cout << "level_" << i << ": " << _num_gates_per_level[i] << "\n";
  }
  std::cout << "\n=====\n\n";
}

void CudaSimulator::print_ppg(const ElementBase &gate) const {
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

void CudaSimulator::print_faults(const Fault *faults) const {
  std::cout << "\n=====\n\n"
            << "FSIM_DEBUG_PRINT_FAULTS\n";
  for (size_t i = 0; i < _num_fault; i++) {
    std::cout << "_gate_with_fault = " << faults[i]._gate_with_fault << ", SA_"
              << faults[i]._gate_SA_fault_val << "\n";
  }
  std::cout << "\n=====\n\n";
}

void CudaSimulator::print_patterns(const Pattern *patterns,
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

void CudaSimulator::print_bits_stack(const size_t size, const void *const ptr) const {
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
void CudaSimulator::print_found_fault_to_pattern(const size_t *found_fault_to_pattern) const {
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

void CudaSimulator::print_simulation_results(const size_t *pi_results, const size_t *gate_results, const size_t *po_results) const {
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
