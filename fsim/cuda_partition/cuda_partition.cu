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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "cuda_partition.cuh"

// #define CPU_GPU_READ_GRAPH_PRINT_FOR_CHECK
// #define CPU_COPY_TS_RESULTS_BACK_TO_HOST
// #define CPU_PRINT_LEVELIZED
// #define CPU_PRINT_CONSTRUCT_GROUPS
// #define GPU_PART_CHECK_FOR_VISITED
#define CHECK_CONSTRUCT_PARTITIONS_DUPLICATIONS

// #define CPU_PART_DEBUG_PRINT_SIMULATION // show the output answers
// #define CPU_PART_DEBUG_PRINT_GRAPH


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

// CPU fucntions
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
  // Read gate 
  ckt >> _num_PIs >> _num_POs >> _num_inner_gates >> _num_wires; 
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;
  
  int *gate_input_order_table_readIn;
  gate_input_order_table_readIn = 
      (int*)malloc(5*_sum_pi_gates_pos*sizeof(int));
  memset(gate_input_order_table_readIn, 0, 5*_sum_pi_gates_pos*sizeof(int));

  _read_graph(ckt, gate_input_order_table_readIn);
  _read_fault(flst);
  _read_pattern(ptn);

  /* Partitioning */
  _ask_gpu_memory();
  _call_gpu_partitioner();
  _construct_partitioned_groups();

  /* For simulation */
  _construct_graph(gate_input_order_table_readIn);
  _ask_simulation_memory();
}

void CUDAPartitioner::_read_graph(std::istream &ckt, int*gate_input_order_table_readIn) {

  // Map table for recording gate input _order
  // Record for each to_gate, its from_gates's are which _order (#_inputs, A/A1, B/A2, A3/S, A4)
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

      gate_input_order_table_readIn[gate_tmp*5+pin_tmp] = gate_0;
      gate_input_order_table_readIn[gate_tmp*5+0]++;
    }
  }
  
  // Read GateType
  _gate_type.resize(_sum_pi_gates_pos);
  for (int i = 0; i < _num_PIs; i++) {
    _gate_type[i] = GateType::PI;
  }
  for (int i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    _gate_type[i] = GateType::PO;
  }
  for (int i = (_num_PIs + _num_POs); i < (_sum_pi_gates_pos); i++) {
    int type;
    ckt >> type;
    _gate_type[i] = static_cast<GateType>(type);
  }

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
  for (size_t fromGate = 0; fromGate < adj.size(); fromGate++) {
    // Update _adj_index_table
    _adj_index_table[2*fromGate+0] = accum;

    for (size_t toGate = 0; toGate < adj[fromGate].size(); toGate++) {
      int toGateIdx = adj[fromGate][toGate];
      
      // Update _adj
      _adj[accum] = toGateIdx;
      accum++;
    }
    // Update _adj_index_table
    _adj_index_table[2*fromGate+1] = accum;
  }

  accum = 0;
  for (size_t toGate = 0; toGate < invAdj.size(); toGate++) {
    // Update _invAdj_index_table
    _invAdj_index_table[2*toGate+0] = accum;

    for (size_t fromGate = 0; fromGate < invAdj[toGate].size(); fromGate++) {
      int toGateIdx = invAdj[toGate][fromGate];
      
      // Update _invAdj
      _invAdj[accum] = toGateIdx;
      accum++;
    }
    // Update _invAdj_index_table
    _invAdj_index_table[2*toGate+1] = accum;
  }

  // _check_graph_connection_CPU(); 

#ifdef CPU_GPU_READ_GRAPH_PRINT_FOR_CHECK
  _print_read_graph(gate_input_order_table_readIn);
#endif 
}

void CUDAPartitioner::_construct_graph(int *gate_input_order_table_readIn) {
  _pi_idx_accum = 0; 
  _po_idx_accum = _num_PIs;
  _gate_idx_accum = (_num_PIs + _num_POs);

  // PIs are stored in the order specified by the `order` array
  _PIs.resize(_num_PIs);
  for (int i = 0; i < _num_PIs; i++) {
    int gate_id = i;
    _PIs[i]._idx = gate_id;
    _PIs[i]._type = _gate_type[gate_id];
    _PIs[i]._level = _level_of_gates[i];
    _PIs[i]._output_value = 0;
  }

  // gates are stored in the order specified by the `order` array
  _gates.resize(_num_inner_gates);
  for (int i = 0; i < _num_inner_gates; i++) {
    int gate_id = i + _gate_idx_accum;
    _gates[i]._idx = gate_id;
    _gates[i]._type = _gate_type[gate_id];
    _gates[i]._level = _level_of_gates[i];
    _gates[i]._output_value = 0;
  }

  // POs are stored in the order specified by the `order` array
  _POs.resize(_num_POs);
  for (int i = 0; i < _num_POs; i++) {
    int gate_id = i+_po_idx_accum;
    _POs[i]._idx = gate_id;
    _POs[i]._type = _gate_type[gate_id];
    _POs[i]._level = _level_of_gates[i];
    _POs[i]._output_value = 0;
  }

  // Give the input gates number
  for (int gateIdx = _num_PIs; gateIdx < _sum_pi_gates_pos; gateIdx++) {
    for (int j = 1; j < (gate_input_order_table_readIn[gateIdx*5] + 1); j++) { // how many inputs for this gate
      int from_gate_id = gate_input_order_table_readIn[gateIdx*5+j];
      int to_gate_id = gateIdx;

      if (from_gate_id < _num_PIs) { // PI -> gate
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &from_gate =
            _PIs[from_gate_id - _pi_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &to_gate =
            _gates[to_gate_id - _gate_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("PI_%lu (%lu) -> Gate_%lu (%lu)\n", from_gate_id - _pi_idx_accum, from_gate_id, to_gate_id - _gate_idx_accum, to_gate_id);
      } else if ((to_gate_id >= (_num_PIs)) &&
                 (to_gate_id < (_num_PIs + _num_POs))) { // gate -> PO
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &to_gate =
            _POs[to_gate_id - _po_idx_accum];
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
        // printf("Gate_%lu (%lu) -> PO_%lu (%lu)\n", from_gate_id - _gate_idx_accum, from_gate_id, to_gate_id - _po_idx_accum, to_gate_id);
      } else { // gate -> gate
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &from_gate =
            _gates[from_gate_id - _gate_idx_accum];
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &to_gate =
            _gates[to_gate_id - _gate_idx_accum];
        // printf("Gate_%lu (%lu) -> Gate_%lu (%lu)\n", from_gate_id - _gate_idx_accum, from_gate_id, to_gate_id - _gate_idx_accum, to_gate_id);
        to_gate._inputs.push_back(&from_gate);
        to_gate._level = _level_of_gates[to_gate_id];
      }
    }
  }
#ifdef CPU_PART_DEBUG_PRINT_GRAPH
  std::cout << "\n_PIs = [\n";
  for (int i = 0; i < _num_PIs; i++) {
    print_ppg(_PIs[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_gates = [\n";
  for (int i = 0; i < _num_inner_gates; i++) {
    print_ppg(_gates[i]);
  }
  std::cout << "]\n";

  std::cout << "\n_POs = [\n";
  for (int i = 0; i < _num_POs; i++) {
    print_ppg(_POs[i]);
  }
  std::cout << "]\n";
#endif  

}


void CUDAPartitioner::_read_fault(std::istream &flst) {
  flst >> _num_fault;
  _faults.resize(_num_fault);

  for (int i = 0; i < _num_fault; i++) {
    FAULT_INDEX_TYPE_PART wrong_gate;
    size_t fault_value;
    flst >> wrong_gate >> fault_value;

    _faults[i]._gate_with_fault = wrong_gate;
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<int>::max() : (0);
  }
}

void CUDAPartitioner::_read_pattern(std::istream &ptn) {
  ptn >> _num_pattern;

  _num_rounds = (_num_pattern + SIZE_T_BITS - 1) / SIZE_T_BITS; // ceiling

  _patterns.resize(_num_rounds);
  for (size_t i = 0; i < _num_rounds; i++) {
    _patterns[i]._value.resize(_num_PIs);
    for (int pi = 0; pi < _num_PIs; pi++) {
      int idx = i;
      ptn >> _patterns[idx]._value[pi];
    }
  }
}


// GPU functions
void CUDAPartitioner::_ask_gpu_memory() {
  // For partitioner
  cudaMalloc((void**)&_adj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_invAdj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_adj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_invAdj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_partitionIndex, _sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_visited, _sum_pi_gates_pos*sizeof(int));
  
  // Ask queue memory 
  // Note: we cudaMalloc a large size queue -> actually array, 22 is got by tuning
  cudaMalloc((void**)&_queue_data, 28*_szOfAdj*sizeof(int));
  // printf("BIBIBI _szOfAdj = %d\n", 28*_szOfAdj);
  cudaMalloc((void**)&_queue_head, 1*sizeof(int));
  cudaMalloc((void**)&_queue_tail, 1*sizeof(int));
  cudaMalloc((void**)&_queue_size, 1*sizeof(int));
  
  cudaCheckErrors("CUDA: Asking partitioner memory - Failure");

  
  cudaMemcpyAsync(_adj_gpu, _adj, 
                  _szOfAdj*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(_invAdj_gpu, _invAdj, 
                  _szOfAdj*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(_adj_index_table_gpu, _adj_index_table, 
                  2*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(_invAdj_index_table_gpu, _invAdj_index_table, 
                  2*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice);


  // Init queue                
  cudaMemset(_queue_data, 0, _szOfAdj*sizeof(int)); 
  cudaMemset(_queue_head, 0, 1*sizeof(int)); 
  cudaMemset(_queue_tail, 0, 1*sizeof(int)); 
  cudaMemset(_queue_size, 0, 1*sizeof(int)); 
  cudaCheckErrors("CUDA: partitioner cudaMemcpyAsync - Failure");

#ifdef CPU_GPU_READ_GRAPH_PRINT_FOR_CHECK
  _gpu_print_read_graph <<< 1, 1 >>> (_sum_pi_gates_pos, _szOfAdj, 
                                      _adj_gpu, _invAdj_gpu, 
                                      _adj_index_table_gpu, 
                                      _invAdj_index_table_gpu);
  cudaCheckErrors("CUDA: _gpu_print_read_graph - Failure");

  cudaDeviceSynchronize();
#endif
}

__global__ void _init_visited(const int _sum_pi_gates_pos, int *_visited) {
  int t_idx = blockIdx.x*blockDim.x+ threadIdx.x;
  if (t_idx < _sum_pi_gates_pos) {
    _visited[t_idx] = 0;
  }
}

__device__ void _enqueue(const int g_idx, int *_queue_data, 
                         int *_queue_tail, 
                         int *_visited) {
  int pos = atomicAdd(_queue_tail, 1);
  _queue_data[pos] = g_idx;
  _visited[g_idx] = 1;
  // printf("\t_enqueue, pos = %d, g_idx = %d\n", pos, g_idx);
}

__device__ int _dequeue(int *_queue_data, int *_queue_head) {
  int pos = atomicAdd(_queue_head, 1);
  int gateIdx = _queue_data[pos];
  return gateIdx; 
}

__device__ bool _canFirstLevelEnqueue(const int _num_PIs,
                                      const int g_idx, 
                                      const int *_invAdj_gpu,
                                      const int *_invAdj_index_table_gpu) {
  bool ret = true;
  // traverse its inputs 
  for (int fmGate = _invAdj_index_table_gpu[2*g_idx+0]; 
      fmGate < _invAdj_index_table_gpu[2*g_idx+1]; fmGate++) {
    int fmGateIdx = _invAdj_gpu[fmGate];
    bool tmp = (_num_PIs > fmGateIdx) ? (true) : (false);
    ret &= tmp;
  }
  return ret; 
}

__device__ bool _canEnqueue(const int g_idx, 
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu, 
                            int *_visited) {
  int ret = 0;
  // traverse its inputs 
  for (int fmGate = _invAdj_index_table_gpu[2*g_idx+0]; 
      fmGate < _invAdj_index_table_gpu[2*g_idx+1]; fmGate++) {
    int fmGateIdx = _invAdj_gpu[fmGate];
    ret += _visited[fmGateIdx];
  }

  // printf("\t_canEnqueue: t_idx = %d g_idx = %d, ret = %d, %d\n", 
  //       blockIdx.x*blockDim.x+threadIdx.x, g_idx, ret, _invAdj_index_table_gpu[2*g_idx+1] - _invAdj_index_table_gpu[2*g_idx+0]);
  return (ret == (_invAdj_index_table_gpu[2*g_idx+1] - _invAdj_index_table_gpu[2*g_idx+0])); 
}

__global__ void _frist_level_enqueue(const int _num_PIs, 
                                    int *_queue_data, int *_queue_head, 
                                    int *_queue_tail, int *_visited) {
  const int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (t_idx < _num_PIs) {
    _enqueue(t_idx, _queue_data, _queue_tail, _visited);
  }
}

template<const int _k>
__global__ void _frist_level(const int _num_PIs,
                            const int *_adj_gpu,
                            const int *_adj_index_table_gpu,
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu,
                            int *_queue_data, int *_queue_head, int *_queue_tail, 
                            int *_visited,
                            int *_partitionIndex) {

  const int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (t_idx < _num_PIs) {
    // const int g_idx = t_idx;
    const int g_idx = _dequeue(_queue_data, _queue_head);
    // printf("t_idx = %d, g_idx = %d\n", t_idx, g_idx);
    
    int maxGroupAccum = 0;
    int maxGroupIdx = 0; 

    int groupIdxAccum [_k];
    for (int i = 0; i < _k; i++) {
      groupIdxAccum[i] = 0;
    }
    
    // traverse its outputs 
    for (int toGate = _adj_index_table_gpu[2*g_idx+0]; 
        toGate < _adj_index_table_gpu[2*g_idx+1]; toGate++) {
      int toGateIdx = _adj_gpu[toGate];
      int toGateMod = toGateIdx%_k;
      groupIdxAccum[toGateMod]++;
      maxGroupIdx = (groupIdxAccum[toGateMod] > maxGroupAccum) ? 
                    (toGateMod) : 
                    (maxGroupIdx);
      maxGroupAccum = (groupIdxAccum[toGateMod] > maxGroupAccum) ? 
                      (groupIdxAccum[toGateMod]) : 
                      (maxGroupAccum);

      // Update queue 
      if (_canEnqueue(toGateIdx, _invAdj_gpu, _invAdj_index_table_gpu, _visited)) {
        _enqueue(toGateIdx, _queue_data, _queue_tail, _visited);
        // printf("\t_enqueue t_idx = %d, toGateIdx = %d\n", t_idx, toGateIdx);
      }
    }

    // Update _partitionIndex
    _partitionIndex[g_idx] = maxGroupIdx;
    // printf("t_idx = %d, _partitionIndex[%d] = %d\n", t_idx, g_idx, _partitionIndex[g_idx]);
  }
}


__global__ void _GPU_partitioner(int *_queue_data, int *_queue_head, int *_queue_tail, int *_queue_size, 
                                 const int *_adj_gpu, 
                                 const int *_adj_index_table_gpu, 
                                 const int *_invAdj_gpu,
                                 const int *_invAdj_index_table_gpu, 
                                 int *_visited,                                
                                 int *_partitionIndex) {
  int t_idx = threadIdx.x + blockIdx.x * blockDim.x;
  // printf("_queue_size[0] = %d\n", _queue_size[0]);
  // __syncthreads();
  
  if (t_idx < *_queue_size) {
    const int g_idx = _dequeue(_queue_data, _queue_head);
    // printf("t_idx = %d, g_idx = %d\n", t_idx, g_idx);
    
    int maxGroupAccum = 0;
    int maxGroupIdx = 0; 

    int groupIdxAccum [_k];
    for (int i = 0; i < _k; i++) {
      groupIdxAccum[i] = 0;
    }

    // traverse its inputs 
    for (int fmGate = _invAdj_index_table_gpu[2*g_idx+0]; 
        fmGate < _invAdj_index_table_gpu[2*g_idx+1]; fmGate++) {
      int fmGateIdx = _invAdj_gpu[fmGate];
      int fmGateGroup = _partitionIndex[fmGateIdx];

      groupIdxAccum[fmGateGroup]++;
      bool upd = groupIdxAccum[fmGateGroup] > maxGroupAccum;
      maxGroupIdx = (upd) ? 
                    (fmGateGroup) : 
                    (maxGroupIdx);
      maxGroupAccum = (upd) ? 
                      (groupIdxAccum[fmGateGroup]) : 
                      (maxGroupAccum);
    }

    // Update _partitionIndex
    _partitionIndex[g_idx] = maxGroupIdx;
    // printf("\tt_idx = %d g_idx = %d, _partitionIndex[%d] %d\n", t_idx, g_idx, g_idx, _partitionIndex[g_idx]);

    
    // Update queue
    // traverse its outputs 
    for (int toGate = _adj_index_table_gpu[2*g_idx+0]; 
        toGate < _adj_index_table_gpu[2*g_idx+1]; toGate++) {
        int toGateIdx = _adj_gpu[toGate];
      
        // Update queue 
        if (_canEnqueue(toGateIdx, _invAdj_gpu, _invAdj_index_table_gpu, _visited)) {
          _enqueue(toGateIdx, _queue_data, _queue_tail, _visited);
          // printf("\t_enqueue t_idx = %d, toGateIdx = %d\n", t_idx, toGateIdx);
        }
    }
  }
}

int CUDAPartitioner::_get_queue_size() {
  int head, tail;
  cudaMemcpy(&head, _queue_head, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tail, _queue_tail, sizeof(int), cudaMemcpyDeviceToHost);
  
  int size = tail - head;
  cudaMemcpy(_queue_size, &size, sizeof(int), cudaMemcpyHostToDevice);

  // printf("_get_queue_size head = %d, tail = %d, q_sz = %d\n", head, tail, size);

  return size;
}

void CUDAPartitioner::_call_gpu_partitioner() {
  const int num_threads = _NUM_THREADS;
  int num_blocks = (_sum_pi_gates_pos + _NUM_THREADS - 1) / _NUM_THREADS;
  int q_sz = 0;
  // printf("_sum_pi_gates_pos = %d\n", _sum_pi_gates_pos);

  // _visited init 
  _init_visited <<< num_blocks, num_threads >>> (_sum_pi_gates_pos, _visited);
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  // For PIs level 
  num_blocks = (_num_PIs + _NUM_THREADS - 1) / _NUM_THREADS;
  _frist_level_enqueue <<< num_blocks, num_threads >>> (_num_PIs, _queue_data, _queue_head, _queue_tail, _visited);
  
  q_sz = _get_queue_size();
  

  _frist_level<_k> <<< num_blocks, num_threads >>> (_num_PIs,
                                                    _adj_gpu, 
                                                    _adj_index_table_gpu, 
                                                    _invAdj_gpu, 
                                                    _invAdj_index_table_gpu,
                                                    _queue_data, _queue_head, _queue_tail, 
                                                    _visited,
                                                    _partitionIndex);
  cudaCheckErrors("CUDA: _frist_level - Failure");
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  q_sz = _get_queue_size();
  
  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  // _print_first_level <<< 1, 1 >>> (_num_PIs, _partitionIndex);
  // cudaCheckErrors("CUDA: _print_first_level - Failure");
  // _print_queue <<< 1, 1 >>> (_queue_data, _queue_head, _queue_tail);
  // cudaCheckErrors("CUDA: _print_queue - Failure");
  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize _print_first_level - Failure");

  // ---------------------------
  // For the remaining levels 
  while (q_sz > 0) {
    num_blocks = (q_sz + _NUM_THREADS - 1) / _NUM_THREADS;
    _GPU_partitioner <<< num_blocks, num_threads >>> (_queue_data, _queue_head, _queue_tail, _queue_size, 
                                                      _adj_gpu, 
                                                      _adj_index_table_gpu, 
                                                      _invAdj_gpu, 
                                                      _invAdj_index_table_gpu, 
                                                      _visited,
                                                      _partitionIndex);
                                                      
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");

    // Update _num_plms_gpu
    q_sz = _get_queue_size();
    
    cudaDeviceSynchronize();
    cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  }

#ifdef GPU_PART_CHECK_FOR_VISITED  
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  _check_visited <<< 1, 1 >>> (_sum_pi_gates_pos, _visited, 1);
  // printf("_num_PIs = %d, _num_POs = %d, _num_inner_gates = %d\n", _num_PIs, _num_POs, _num_inner_gates);
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
#endif 

  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  // _print_GPU_partitioner <<< 1, 1 >>> (_sum_pi_gates_pos, _partitionIndex);
  // cudaCheckErrors("CUDA: _print_GPU_partitioner - Failure");    
  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");

}


void CUDAPartitioner::_construct_partitioned_groups() {
  // Topological sort on CPU:
    // Copy topological sort results from GPU to CPU
    // Using topological sort results from GPU to levelize 
  _get_topological_sort_resutls();  

  // construct group and show constructed group 
  _construct_groups();

#ifdef CHECK_CONSTRUCT_PARTITIONS_DUPLICATIONS
  _count_duplications();
#endif 
}


void CUDAPartitioner::_copy_TS_resutls_to_Host() {
  _queue_data_cpu.resize(28*_szOfAdj);
  _partitionIndex_cpu.resize(_sum_pi_gates_pos); 
  _queue_tail_cpu = 0;

  cudaMemcpyAsync(_queue_data_cpu.data(), _queue_data, 
                  28*_szOfAdj*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(_partitionIndex_cpu.data(), _partitionIndex, 
                  _sum_pi_gates_pos*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(&_queue_tail_cpu, _queue_tail, 
                  1*sizeof(int), cudaMemcpyDeviceToHost);

#ifdef CPU_COPY_TS_RESULTS_BACK_TO_HOST
  _print_copy_TS_resutls_to_Host();
#endif 
}

void CUDAPartitioner::_levelized() {
  _level_of_gates.resize(_sum_pi_gates_pos, 0);
  _max_level = 0;
  
  for (int i = 0; i < _queue_tail_cpu; i++) {
    int gateIdx = _queue_data_cpu[i];
    // travverse its inputs 
    for (int fmGate = _invAdj_index_table[2*gateIdx+0]; 
        fmGate < _invAdj_index_table[2*gateIdx+1];
        fmGate++) {
      int fmGateIdx = _invAdj[fmGate];
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

#ifdef CPU_PRINT_LEVELIZED
  _print_levelized();
#endif 
}


void CUDAPartitioner::_get_topological_sort_resutls() {
  _copy_TS_resutls_to_Host();  
  _levelized();
}

void CUDAPartitioner::_construct_cones(std::vector<int> &sinks,
                                       std::vector<std::set<int>> &_cones_set_g,
                                       std::vector<bool> &visited_cpu) {

  // printf("sinks.size() = %lu\n", sinks.size());
  for (size_t sink = 0; sink < sinks.size(); sink++) {
    int sinkIdx = sinks[sink];
    std::queue<int> preGates; 
    preGates.push(sinkIdx);
    // used to check the correctness 
    visited_cpu[sinkIdx] = true;
    // used to check the correctness 

    int sourceLevel = _level_of_gates[sinkIdx];
    _cones_set_g[sourceLevel].insert(sinkIdx);


    while (!preGates.empty()) {
      int startGateIdx = preGates.front(); 
      preGates.pop();
      visited_cpu[startGateIdx] = true;

      for (int j = _invAdj_index_table[2*startGateIdx+0]; 
          j < _invAdj_index_table[2*startGateIdx+1]; 
          j++) {
        int preGateIdx = _invAdj[j];
        int preGateLevel = _level_of_gates[preGateIdx];
        _cones_set_g[preGateLevel].insert(preGateIdx);
        preGates.push(preGateIdx);
        // used to check the correctness 
        // used to check the correctness 
      }
    }
  }
}

void CUDAPartitioner::_construct_groups(){
  // Get the group of sinks 
  std::vector<std::vector<int>> sinks_groups;
  sinks_groups.resize(_k);

  for (int sinkIdx = 0; sinkIdx < _sum_pi_gates_pos; sinkIdx++) {
    int outputSize = _adj_index_table[2*sinkIdx+1] - _adj_index_table[2*sinkIdx+0];
    if (outputSize == 0) {
      int sinkGroup = _partitionIndex_cpu[sinkIdx];
      sinks_groups[sinkGroup].push_back(sinkIdx);
    }
  }

  // used to check the correctness 
  std::vector<bool> visited_cpu;
  visited_cpu.resize(_sum_pi_gates_pos, false);
  // used to check the correctness 

  // Construct cones 
  _cones_set.resize(_k);
  _cones_partitioned.resize(_k);
  for (size_t g = 0; g < sinks_groups.size(); g++) {
    // Push in NULL cone
    _cones_set[g].resize(_total_num_levels);
    _cones_partitioned[g].resize(_total_num_levels);
    _construct_cones(sinks_groups[g], _cones_set[g], visited_cpu);
  }

  // Construct _cones_partitioned
  for (size_t i = 0; i < _cones_set.size(); ++i) {
    for (size_t lvl = 0; lvl < _cones_set[i].size(); ++lvl) {
      for (const auto& element : _cones_set[i][lvl]) {
        _cones_partitioned[i][lvl].push_back(element);
      }
    }
  }  

#ifdef GPU_PART_CHECK_FOR_VISITED
  int accum = 0;
  for (size_t i = 0; i < visited_cpu.size(); i++) {
    if (!visited_cpu[i]) {
      accum++;
    }
  }
  printf("BIBI2: visited_cpu: accum = %d\n\n", accum);
#endif 

#ifdef CPU_PRINT_CONSTRUCT_GROUPS
  _print_construct_groups(sinks_groups);
#endif
}


// Simulation functions 
void CUDAPartitioner::run(Mode mode, const size_t num_threads) {  
  int m = static_cast<int>(mode);

  switch (mode) {
    case Mode::CONES_PARALLEL_CUDA:
      std::cout << "Mode: " << m << " (run CONES_PARALLEL_CUDA)\n";
      _run_cones_gates(num_threads);
      break;   
    default:
      printf("No SUCH CASES\n");
      return; 
  }
}

void CUDAPartitioner::_run_cones_gates(const size_t num_threads) {
#ifdef CPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "execute simulation._run_cones_gates();\n";
#endif

  // Simulation
  for (size_t rd = 0; rd < _num_rounds; rd++) {
    size_t num_testcases_this_round =
        ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
            ? (SIZE_T_BITS)
            : (_num_pattern % SIZE_T_BITS);
#ifdef CPU_PART_DEBUG_PRINT_SIMULATION
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
#ifdef CPU_PART_DEBUG_PRINT_SIMULATION
  if (rd == 0) {
    std::cout << "GOOD resutls ans:" << std::endl;
    print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);        
  }
#endif
  }
}

void CUDAPartitioner::_run_cones_good_case(const Pattern pattern, 
                                          const size_t bits, 
                                          const size_t num_threads) {

  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP
  // printf("num_threads = %lu, _cones_partitioned.size() = %lu\n", 
  //         num_threads, _cones_partitioned.size());

  // Parallelize across cones (c)
  #pragma omp parallel for if (_cones_partitioned.size() > 1) schedule(dynamic)
  for (size_t c = 0; c < _cones_partitioned.size(); c++) {
    for (size_t lvl = 0; lvl < _cones_partitioned[c].size(); lvl++) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for schedule(dynamic)
      for (size_t g = 0; g < _cones_partitioned[c][lvl].size(); g++) {
        int gateIdx = _cones_partitioned[c][lvl][g];
        
        // Select the correct element (PI, PO, or gate)
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &tmp = 
                  (gateIdx < _num_PIs) ? (_PIs[gateIdx - _pi_idx_accum]) :
                  (gateIdx >= (_num_PIs + _num_POs)) ? (_gates[gateIdx - _gate_idx_accum]) :
                  (_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, 0, 0);
      }
    }
  }
  
  // Duplicated the answer of good_case into a new memory
  _write_to_array(_g_pi_results, _g_gate_results, _g_po_results, bits);

#ifdef CPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: GOOD resutls ans:\n";
  print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
#endif
}


void CUDAPartitioner::_run_cones_bad_case(const Fault<FAULT_INDEX_TYPE_PART> &fault, 
                                          const Pattern pattern,
                                          const size_t bits, const size_t num_threads) {
  // printf("Getting inside _run_cones_bad_case\n");
  omp_set_num_threads(num_threads);  // Set the number of threads for OpenMP

  FAULT_INDEX_TYPE_PART wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;

  // Parallelize across cones (c)
  #pragma omp parallel for
  for (size_t c = 0; c < _cones_partitioned.size(); c++) {
   for (size_t lvl = 0; lvl < _cones_partitioned[c].size(); lvl++) {
      // Parallelize across gates within the same cone and level
      #pragma omp parallel for
      for (size_t g = 0; g < _cones_partitioned[c][lvl].size(); g++) {
        int thread_id = omp_get_thread_num();
        int gateIdx = _cones_partitioned[c][lvl][g];

        // Calculate SA_fault using a ternary operator (avoiding if-else)
        FAULT_INDEX_TYPE_PART SA_fault = (gateIdx < _num_PIs) ? ((gateIdx + _pi_idx_accum) == wrong_gate) :
                          (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx + _gate_idx_accum) == wrong_gate) :
                          ((gateIdx + _po_idx_accum) == wrong_gate);

        // Select the correct element (PI, PO, or gate)
        ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &tmp = (gateIdx < _num_PIs) ? (_PIs[gateIdx - _pi_idx_accum]) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? (_gates[gateIdx - _gate_idx_accum]) :
                            (_POs[gateIdx - _po_idx_accum]);

        // Execute the gate operation
        _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);

      }
    }
  }

  // Duplicated the answer of good_case into a new memory
  _write_to_array(_b_pi_results, _b_gate_results, _b_po_results, bits);

#ifdef CPU_PART_DEBUG_PRINT_SIMULATION
  std::cout << "CONES: BAD resutls ans:\n";
  print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
#endif
}

void CUDAPartitioner::_write_to_array(std::vector<size_t> &pi_output, 
                                    std::vector<size_t> &gate_output,
                                    std::vector<size_t> &po_output,
                                    const size_t bits) {
  // Duplicated the answer of good_case into a new memory
  for (int i = 0; i < _num_PIs; i++) {
    pi_output[i] = _PIs[i]._output_value;
  }
  for (int i = 0; i < _num_inner_gates; i++) {
    gate_output[i] = _gates[i]._output_value;
  }
  for (int i = 0; i < _num_POs; i++) {
    po_output[i] = _POs[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    _shift_to_correct_answer(pi_output, bits, _num_PIs);
    _shift_to_correct_answer(gate_output, bits, _num_inner_gates);
    _shift_to_correct_answer(po_output, bits, _num_POs);
  }  
}

// Simulation for gates
void CUDAPartitioner::_apply_INV(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

void CUDAPartitioner::_apply_AND(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_OR(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;
  
  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_XOR(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void CUDAPartitioner::_apply_NAND(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_NOR(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_XNOR(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void CUDAPartitioner::_apply_MUX(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  size_t a = gate._inputs[0]->_output_value;
  size_t b = gate._inputs[1]->_output_value;
  size_t s = gate._inputs[2]->_output_value;

  size_t ret = ((s & b) | (a & (!s)));

  gate._output_value = (ret);
}

void CUDAPartitioner::_apply_CLKBUF(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  gate._output_value = ret;
}

void CUDAPartitioner::_apply_PI(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate, const Pattern pattern,
                                const size_t pi) {
  gate._output_value = pattern._value[pi];
}

void CUDAPartitioner::_apply_PO(ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}


void CUDAPartitioner::_run_gate(ElementBase<ELEMENT_LEVEL_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate, 
                                const Pattern pattern,
                                const ELEMENT_INDEX_TYPE_PART pi, 
                                const FAULT_INDEX_TYPE_PART SA_fault, 
                                const size_t fault_val) {

  if (SA_fault == 1) {
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
                                               const ELEMENT_INDEX_TYPE_PART num_shift_gates) {
  for (ELEMENT_INDEX_TYPE_PART i = 0; i < num_shift_gates; i++) {
    results[i] <<= (SIZE_T_BITS - bits);
    results[i] >>= (SIZE_T_BITS - bits);
  }
}


















// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------

void CUDAPartitioner::print_patterns(const std::vector<Pattern> &patterns,
                                    const int round,
                                    const int num_PIs) const {
  std::cout << "\n=====\n\n";
    for (int i = 0; i < round; i++) {
      std::cout << "[" << SIZE_T_BITS * i << ", " << SIZE_T_BITS * (i + 1)
                << "] bits = [\n";
      for (int j = 0; j < num_PIs; j++) {
        print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
      }
    std::cout << "]\n";
  }
}

void CUDAPartitioner::print_bits_stack(const int size,
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
gpuGateType CUDAPartitioner::convertGateTypeToGpu(GateType gate_type) {
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


// Device-side function to convert gpuGateType to string
__device__ const char* gpuGateTypeToString(gpuGateType type) {
  switch (type) {
    case gpuGateType::INV:
      return "INV";
    case gpuGateType::AND:
      return "AND";
    case gpuGateType::OR:
      return "OR";
    case gpuGateType::XOR:
      return "XOR";
    case gpuGateType::NAND:
      return "NAND";
    case gpuGateType::NOR:
      return "NOR";
    case gpuGateType::XNOR:
      return "XNOR";
    case gpuGateType::MUX:
      return "MUX";
    case gpuGateType::CLKBUF:
      return "CLKBUF";
    case gpuGateType::PI:
      return "PI";
    case gpuGateType::PO:
      return "PO";
    default:
      return "UNKNOWN";
  }
}

void CUDAPartitioner::_print_read_graph(int *gate_input_order_table_readIn) const {
  printf("Get inside _print_read_graph:\n");

  printf("_sum_pi_gates_pos = %d, _szOfAdj = %d\n", 
          _sum_pi_gates_pos, _szOfAdj);

  printf("gate_input_order_table_readIn:\n");
  for (int g = 0; g < _sum_pi_gates_pos; g++) {
    printf("gateIdx_%d, with Inputs (%d) = [", g, gate_input_order_table_readIn[g*5+0]);
    for (int j = 1; j < gate_input_order_table_readIn[g*5+0]+1; j++) {
      printf("%d, ", gate_input_order_table_readIn[g*5+j]);
    }
    printf("]\n");
  }

  printf("_gate_type:\n");
  for (size_t i = 0; i < _gate_type.size(); i++) {
    std::cout << "Gate_" << i << ", _gate_type = " << gateTypeToString(_gate_type[i]) << "\n";
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
  printf("===============================\n");
}

__global__ void _gpu_print_read_graph(const int _sum_pi_gates_pos, 
                                      const int _szOfAdj, 
                                      const int *_adj_gpu, 
                                      const int *_invAdj_gpu, 
                                      const int *_adj_index_table_gpu, 
                                      const int *_invAdj_index_table_gpu) {
  printf("Get inside _gpu_print_read_graph:\n");
  printf("_sum_pi_gates_pos = %d, _szOfAdj = %d\n", 
          _sum_pi_gates_pos, _szOfAdj);

  printf("_adj_gpu:\n");
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%d's output = ", i);
    for (int j = _adj_index_table_gpu[2*i+0]; 
                j < _adj_index_table_gpu[2*i+1]; 
                j++) {
      printf("%d, ", _adj_gpu[j]);
    }
    printf("\n");
  }
  printf("\n");  
  
  printf("\n---------\n");
  
  printf("_invAdj_gpu:\n");
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%d's output = ", i);
    for (int j = _invAdj_index_table_gpu[2*i+0]; 
                j < _invAdj_index_table_gpu[2*i+1]; 
                j++) {
      printf("%d, ", _invAdj_gpu[j]);
    }
    printf("\n");
  }
  printf("\n");  
  printf("===============================\n");
      
}

__global__ void _print_first_level(const int _num_PIs, 
                                   const int *_partitionIndex) {
  printf("Get Insdies _print_first_level\n");
  
  for (int i = 0; i < _num_PIs; i++) {
    const int piIdx = i;
    const int groupIdx = _partitionIndex[piIdx];
    printf("piIdx = %d, groupIdx = %d\n", piIdx, groupIdx);
  }
  printf("\n");
}

__global__ void _print_queue(int *_queue_data, int *_queue_head, int *_queue_tail) {
  int start = *_queue_head;
  int endpt = *_queue_tail;

  printf("Get Insdies _print_queue, start = %d, endpt = %d\n", start, endpt);

  printf("queue = [");
  for (int i = start; i < endpt; i++) {
    printf("%d, ", _queue_data[i]);
  } printf("]\n");
  printf("\n");
}

__global__ void _print_GPU_partitioner(const int _sum_pi_gates_pos, 
                                      const int *_partitionIndex) {
                  
  printf("Get Insdies _print_GPU_partitioner\n");
  
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    const int gateIdx = i;
    const int groupIdx = _partitionIndex[gateIdx];
    printf("gateIdx = %d, groupIdx = %d\n", gateIdx, groupIdx);
  }
}

__global__ void _check_visited(const int sz, const int *_visited, const int mode) {
  
  int ret = 0;
  for (int i = 0; i < sz; i++) {
    if (_visited[i] == 0) {
      ret++;
      printf("check_visited == false, %d\n", i);
    }
  }

  switch (mode) {
    case 0: {
      printf("check_visited, check init\n");
      if (ret != sz) {
        printf("check_visited: ret != sz; ret = %d, sz = %d\n", ret, sz);
      } else {
        printf("check_visited: ret == sz\n");
      }
      break;
    }
    case 1: {
      if (ret != 0) {
        printf("check_visited FAILED: ret = %d, sz = %d\n", ret, sz);
      } else {
        printf("check_visited PASSED\n");
      }
      break;
    }
  }  
}

void CUDAPartitioner::_print_copy_TS_resutls_to_Host() {
  printf("_queue_tail_cpu = %d\n", _queue_tail_cpu);
  
  printf("_queue_data_cpu: [");
  for (int i = 0; i < _queue_tail_cpu; i++) {
    printf("%d, ", _queue_data_cpu[i]);
  } printf("]\n");

  printf("_partitionIndex_cpu:\n");
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%i, group = %d\n", i, _partitionIndex_cpu[i]);
  } printf("\n");
  printf("\n");
}

void CUDAPartitioner::_print_levelized(){
  printf("_level_of_gates:\n");
  for (size_t i = 0; i < _level_of_gates.size(); i++) {
    printf("_level_of_gates[%lu] = %d\n", i, _level_of_gates[i]);
  }
  printf("\n");
}

void CUDAPartitioner::_print_construct_groups(const std::vector<std::vector<int>> &sinks_groups) {
  for (size_t i = 0; i < sinks_groups.size(); i++) {
    printf("Group Idx = %lu with sinkIdx: ", i);
    for (size_t j = 0; j < sinks_groups[i].size(); j++) {
      printf("%d, ", sinks_groups[i][j]);
    } printf("\n");
  } printf("\n");
  printf("\n");

  printf("_cones_partitioned:\n\n");
  for (size_t i = 0; i < _cones_partitioned.size(); i++) {
    printf("_cones_partitioned %lu\n", i);
    for (size_t lvl = 0; lvl < _cones_partitioned[i].size(); lvl++) {
      printf("\tlvl = %lu, gates: ", lvl);
      for (const auto& element : _cones_partitioned[i][lvl]) {
        printf("%d, ", element);
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

void CUDAPartitioner::_count_duplications() {
  std::vector<bool> visited_cpu;
  visited_cpu.resize(_sum_pi_gates_pos, false);
  std::vector<size_t> numGates; 
  numGates.resize(4, 0);

  for (size_t i = 0; i < _cones_partitioned.size(); i++) {
    for (size_t lvl = 0; lvl < _cones_partitioned[i].size(); lvl++) {
      // for (const auto& element : _cones_partitioned[i][lvl]) {
      for (size_t g = 0; g < _cones_partitioned[i][lvl].size(); g++) {
        int element = _cones_partitioned[i][lvl][g];
        visited_cpu[element] = true; 
        if (element < _num_PIs) { // is PI
          numGates[0]++;
        } else if (element >= _num_PIs && element < (_num_PIs+_num_POs)) { // is PO
          numGates[2]++;
        } else { // is Gate
          numGates[1]++;
        }
      }
    }
  }

  numGates[3] = numGates[0]+numGates[1]+numGates[2];

  int accum = 0;
  for (size_t i = 0; i < visited_cpu.size(); i++) {
    if (!visited_cpu[i]) {
      accum++;
    }
  }

  printf("Partitioned:\n");
  printf("_part_PIs = %lu, %d\n", numGates[0], _num_PIs);
  printf("_part_Gates = %lu, %d\n", numGates[1], _num_inner_gates);
  printf("_part_POs = %lu, %d\n", numGates[2], _num_POs);
  printf("_part_sum_pi_gates_pos = %lu, %d\n", numGates[3], _sum_pi_gates_pos);
  printf("_part_visited_cpu: accum = %d\n", accum);
}


void CUDAPartitioner::_check_graph_connection_CPU() {
  std::vector<bool> visited_cpu;
  visited_cpu.resize(_sum_pi_gates_pos, false);

  std::vector<int> sources;
  for (int i = 0; i < _num_PIs; i++) {
    sources.push_back(i);
  }

  for (size_t fromGate = 0; fromGate < sources.size(); fromGate++) {
    int sourceIdx = sources[fromGate];
    std::queue<int> postGates; 
    postGates.push(sourceIdx);
    visited_cpu[sourceIdx] = true; 

    while (!postGates.empty()) {
      int startGateIdx = postGates.front(); 
      postGates.pop();
      visited_cpu[startGateIdx] = true; 
      // printf("\tstartGateIdx = %d %d (tid = %d)\n", startGateIdx, adj[startGateIdx].size(), tid);

      for (int j = _adj_index_table[2*startGateIdx+0]; 
          j < _adj_index_table[2*startGateIdx+1]; 
          j++) {
        int postGateIdx = _adj[j];
        postGates.push(postGateIdx);
      }
    }
  }

  int accum = 0;
  for (size_t i = 0; i < visited_cpu.size(); i++) {
    if (!visited_cpu[i]) {
      accum++;
    }
  }

  printf("BIBI, accum = %d\n", accum);


  // REVERSE VERSION
  std::vector<bool> visited_cpu_back;
  visited_cpu_back.resize(_sum_pi_gates_pos, false);

  std::vector<int> sinks;
  for (int s = 0; s < _sum_pi_gates_pos; s++) {
    int outputSize = _adj_index_table[2*s+1] - _adj_index_table[2*s+0];
    if (outputSize == 0) {
      sinks.push_back(s);
    }
  }

  for (size_t sink = 0; sink < sinks.size(); sink++) {
    int sinkIdx = sinks[sink];
    std::queue<int> preGates; 
    preGates.push(sinkIdx);
    visited_cpu_back[sinkIdx] = true; 

    while (!preGates.empty()) {
      int startGateIdx = preGates.front(); 
      preGates.pop();
      visited_cpu_back[startGateIdx] = true; 

      for (int j = _invAdj_index_table[2*startGateIdx+0]; 
          j < _invAdj_index_table[2*startGateIdx+1]; 
          j++) {
        int preGatesIdx = _invAdj[j];
        preGates.push(preGatesIdx);
      }
    }
  }

  int accum2 = 0;
  for (size_t i = 0; i < visited_cpu_back.size(); i++) {
    if (!visited_cpu_back[i]) {
      accum2++;
    }
  }

  printf("BIBI, back: accum2 = %d\n", accum2);

  // Check the correctness of _adj_index_table and _invAdj_index_table
  for (int g = 0; g < _sum_pi_gates_pos; g++) {
    int fmGateIdx = g;
    for (int toGate = _adj_index_table[2*fmGateIdx+0]; 
      toGate < _adj_index_table[2*fmGateIdx+1]; 
      toGate++) {
      int toGateIdx = _adj[toGate];
      bool found = false;

      for (int fmGate2 = _invAdj_index_table[2*toGateIdx+0]; 
        fmGate2 < _invAdj_index_table[2*toGateIdx+1]; 
        fmGate2++) {
        int fmGate2Idx = _invAdj[fmGate2];
        if (fmGate2Idx == fmGateIdx) {
          found = true; 
          break;
        }
      }

      if (!found) {
        printf("NOT FOUND, with fmGateIdx = %d, toGateIdx = %d\n", fmGateIdx, toGateIdx);
      }
    }
  }

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


void CUDAPartitioner::print_ppg(const ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) const {
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





