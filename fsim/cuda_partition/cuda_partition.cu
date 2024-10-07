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
  _read_graph(ckt);
  _read_fault(flst);
  _read_pattern(ptn);

  /* Partitioning */
  _ask_gpu_memory();
  _call_gpu_partitioner();
}

void CUDAPartitioner::_read_graph(std::istream &ckt) {
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  // Map table for recording gate input _order
  // Record for each to_gate, its from_gates's are which _order (#_inputs, A/A1, B/A2, A3/S, A4)
  int *gate_input_order_table_readIn = 
      (int*)malloc(5*_sum_pi_gates_pos*sizeof(int));
  memset(gate_input_order_table_readIn, 0, 5*_sum_pi_gates_pos*sizeof(int));

  _szOfAdj = 0;
  std::vector<std::vector<int>> adj;
  adj.resize(_sum_pi_gates_pos);

  // Construct adjacent list
  for (int i = 0; i < _num_wires; i++) {
    int gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (int j = 0; j < num_post_gates; j++) {
      int gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      adj[gate_0].push_back(gate_tmp); 
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

  // Construct _adj
  // printf("_szOfAdj = %d\n", _szOfAdj);
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
  for (int toGate = 0; toGate < _sum_pi_gates_pos; toGate++) {
    // Update _invAdj_index_table
    _invAdj_index_table[2*toGate+0] = accum;
    
    for (int i = 0; i < gate_input_order_table_readIn[toGate*5+0]; i++) {
      int fromGateIdx = gate_input_order_table_readIn[toGate*5+i+1];

      // Update 
      _invAdj[accum] = fromGateIdx;
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

void CUDAPartitioner::_check_graph_connection_CPU() {
  std::vector<bool> _visited_cpu;
  _visited_cpu.resize(_sum_pi_gates_pos, false);

  std::vector<int> _sources;
  for (int i = 0; i < _num_PIs; i++) {
    _sources.push_back(i);
  }
  std::vector<int> _level_of_gates;
  _level_of_gates.resize(_sum_pi_gates_pos, 0);
  
  for (size_t fromGate = 0; fromGate < _sources.size(); fromGate++) {
    int sourceIdx = _sources[fromGate];
    std::queue<int> postGates; 
    postGates.push(sourceIdx);
    _visited_cpu[sourceIdx] = true; 
    _level_of_gates[sourceIdx] = 0;

    while (!postGates.empty()) {
      int startGateIdx = postGates.front(); 
      postGates.pop();
      _visited_cpu[startGateIdx] = true; 
      // printf("\tstartGateIdx = %d %d (tid = %d)\n", startGateIdx, adj[startGateIdx].size(), tid);

      for (int j = _adj_index_table[2*startGateIdx+0]; 
          j < _adj_index_table[2*startGateIdx+1]; 
          j++) {
        // for (int j = 0; j < _adj[startGateIdx].size(); j++) {
        int postGateIdx = _adj[j];
        bool update = (_level_of_gates[startGateIdx] >= _level_of_gates[postGateIdx]);
        _level_of_gates[postGateIdx] = (update)  
                                        ? (_level_of_gates[startGateIdx]+1) 
                                        : (_level_of_gates[postGateIdx]);

        if (update) {
          postGates.push(postGateIdx);
        }
      }
    }
  }

  int accum = 0;
  for (size_t i = 0; i < _visited_cpu.size(); i++) {
    if (!_visited_cpu[i]) {
      accum++;
    }
  }

  printf("BIBI, accum = %d\n", accum);
}

void CUDAPartitioner::_read_fault(std::istream &flst) {
  flst >> _num_fault;
  _faults.resize(_num_fault);

  for (int i = 0; i < _num_fault; i++) {
    int wrong_gate, fault_value;
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
  for (int i = 0; i < _num_rounds; i++) {
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

  // TODO: 
  // 3. construct new cones
  // 4. connect with simulator 

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
      int fmGateMod = fmGateIdx%_k;
      groupIdxAccum[fmGateMod]++;
      bool upd = groupIdxAccum[fmGateMod] > maxGroupAccum;
      maxGroupIdx = (upd) ? 
                    (fmGateMod) : 
                    (maxGroupIdx);
      maxGroupAccum = (upd) ? 
                      (groupIdxAccum[fmGateMod]) : 
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
  printf("_sum_pi_gates_pos = %d\n", _sum_pi_gates_pos);

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

  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  _check_visited <<< 1, 1 >>> (_sum_pi_gates_pos, _visited, 1);
  // printf("_num_PIs = %d, _num_POs = %d, _num_inner_gates = %d\n", _num_PIs, _num_POs, _num_inner_gates);
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");


  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  _print_GPU_partitioner <<< 1, 1 >>> (_sum_pi_gates_pos, _partitionIndex);
  cudaCheckErrors("CUDA: _print_GPU_partitioner - Failure");    
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");

}



// Simulation functions 
void CUDAPartitioner::run(Mode mode) {
  int m = static_cast<int>(mode);

  switch (mode) {
    case Mode::PARALLEL_CUDA:
      std::cout << "Mode: " << m << " (run PARALLEL_CUDA)\n";
      // _run_gates(num_threads);
      break;
    case Mode::CONES_PARALLEL_CUDA:
      std::cout << "Mode: " << m << " (run CONES_PARALLEL_CUDA)\n";
      // _run_gates(num_threads);
      break;   
    default:
      printf("No SUCH CASES\n");
      return; 
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