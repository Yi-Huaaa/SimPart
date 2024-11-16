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
// #define CHECK_CONSTRUCT_PARTITIONS_DUPLICATIONS

// For simulation 
// #define CPU_PART_DEBUG_PRINT_GRAPH // print _construct_graph
// #define CPU_PART_DEBUG_PRINT_SIMULATION // show the output answers
// #define GPU_PREPARE_SIMULATION_PRINT_CHECK


#define CPU_PARTITIONER_PRINT_FOR_CHECK

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
  
  _read_graph(ckt);
  _read_fault(flst);
  _read_pattern(ptn);

  /* Partitioning */
  auto start = std::chrono::steady_clock::now();
  _ask_gpu_memory_part();
  _call_gpu_partitioner();

  // Topological sort on CPU:
    // Copy topological sort results from GPU to CPU
    // Using topological sort results from GPU to levelize 
  _get_topological_sort_resutls();  

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_GPU_partition = end - start;

  std::cout << "duration_GPU_partition: " << _round_to((duration_GPU_partition.count())*1000, 0.001) << "\n";
}

void CUDAPartitioner::_read_graph(std::istream &ckt) {

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
  _print_read_graph();
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

// --------------------------------------------------------------------------------------------------

/* GPU Partitioner */
void CUDAPartitioner:: _ask_gpu_memory_part() {
  // For partitioner
  cudaMalloc((void**)&_adj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_invAdj_gpu, _szOfAdj*sizeof(int));
  cudaMalloc((void**)&_adj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_invAdj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_partitionIndex_gpu, _sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_visited_gpu, _sum_pi_gates_pos*sizeof(int));
  
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
  
  // Init visited gpu
  cudaMemset(_visited_gpu, 0, _sum_pi_gates_pos*sizeof(int)); 
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

__global__ void _init_visited(const int _sum_pi_gates_pos, int *_visited_gpu) {
  int t_idx = blockIdx.x*blockDim.x+ threadIdx.x;
  if (t_idx < _sum_pi_gates_pos) {
    _visited_gpu[t_idx] = 0;
  }
}

__device__ void _enqueue(const int g_idx, int *_queue_data, 
                         int *_queue_tail, 
                         int *_visited_gpu) {
  int pos = atomicAdd(_queue_tail, 1);
  _queue_data[pos] = g_idx;
  _visited_gpu[g_idx] = 1;
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
                            int *_visited_gpu) {
  int ret = 0;
  // traverse its inputs 
  for (int fmGate = _invAdj_index_table_gpu[2*g_idx+0]; 
      fmGate < _invAdj_index_table_gpu[2*g_idx+1]; fmGate++) {
    int fmGateIdx = _invAdj_gpu[fmGate];
    ret += _visited_gpu[fmGateIdx];
  }

  // printf("\t_canEnqueue: t_idx = %d g_idx = %d, ret = %d, %d\n", 
  //       blockIdx.x*blockDim.x+threadIdx.x, g_idx, ret, _invAdj_index_table_gpu[2*g_idx+1] - _invAdj_index_table_gpu[2*g_idx+0]);
  return (ret == (_invAdj_index_table_gpu[2*g_idx+1] - _invAdj_index_table_gpu[2*g_idx+0])); 
}

__global__ void _frist_level_enqueue(const int _num_PIs, 
                                    int *_queue_data, int *_queue_head, 
                                    int *_queue_tail, int *_visited_gpu) {
  const int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (t_idx < _num_PIs) {
    _enqueue(t_idx, _queue_data, _queue_tail, _visited_gpu);
  }
}

template<const int _k>
__global__ void _frist_level(const int _num_PIs,
                            const int *_adj_gpu,
                            const int *_adj_index_table_gpu,
                            const int *_invAdj_gpu,
                            const int *_invAdj_index_table_gpu,
                            int *_queue_data, int *_queue_head, int *_queue_tail, 
                            int *_visited_gpu,
                            int *_partitionIndex_gpu) {

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
      if (_canEnqueue(toGateIdx, _invAdj_gpu, _invAdj_index_table_gpu, _visited_gpu)) {
        _enqueue(toGateIdx, _queue_data, _queue_tail, _visited_gpu);
        // printf("\t_enqueue t_idx = %d, toGateIdx = %d\n", t_idx, toGateIdx);
      }
    }

    // int maxGroupIdx = int((static_cast<double>(toGateIdx/_num_PIs))*8);
    

    // Update _partitionIndex_gpu
    _partitionIndex_gpu[g_idx] = maxGroupIdx;
    // printf("t_idx = %d, _partitionIndex_gpu[%d] = %d\n", t_idx, g_idx, _partitionIndex_gpu[g_idx]);
  }
}


__global__ void _GPU_partitioner(int *_queue_data, int *_queue_head, int *_queue_tail, int *_queue_size, 
                                 const int *_adj_gpu, 
                                 const int *_adj_index_table_gpu, 
                                 const int *_invAdj_gpu,
                                 const int *_invAdj_index_table_gpu, 
                                 int *_visited_gpu,                                
                                 int *_partitionIndex_gpu) {
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
      int fmGateGroup = _partitionIndex_gpu[fmGateIdx];

      groupIdxAccum[fmGateGroup]++;
      bool upd = groupIdxAccum[fmGateGroup] > maxGroupAccum;
      maxGroupIdx = (upd) ? 
                    (fmGateGroup) : 
                    (maxGroupIdx);
      maxGroupAccum = (upd) ? 
                      (groupIdxAccum[fmGateGroup]) : 
                      (maxGroupAccum);
    }

    // Update _partitionIndex_gpu
    _partitionIndex_gpu[g_idx] = maxGroupIdx;
    // printf("\tt_idx = %d g_idx = %d, _partitionIndex_gpu[%d] %d\n", t_idx, g_idx, g_idx, _partitionIndex_gpu[g_idx]);

    
    // Update queue
    // traverse its outputs 
    for (int toGate = _adj_index_table_gpu[2*g_idx+0]; 
        toGate < _adj_index_table_gpu[2*g_idx+1]; toGate++) {
        int toGateIdx = _adj_gpu[toGate];
      
        // Update queue 
        if (_canEnqueue(toGateIdx, _invAdj_gpu, _invAdj_index_table_gpu, _visited_gpu)) {
          _enqueue(toGateIdx, _queue_data, _queue_tail, _visited_gpu);
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
  // int num_blocks = (_sum_pi_gates_pos + _NUM_THREADS - 1) / _NUM_THREADS;
  int q_sz = 0;
  // printf("_sum_pi_gates_pos = %d\n", _sum_pi_gates_pos);

  // // _visited_gpu init -> changed to use cudaMemset
  // _init_visited <<< num_blocks, num_threads >>> (_sum_pi_gates_pos, _visited_gpu);
  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  // For PIs level 
  int num_blocks = (_num_PIs + _NUM_THREADS - 1) / _NUM_THREADS;
  _frist_level_enqueue <<< num_blocks, num_threads >>> (_num_PIs, _queue_data, _queue_head, _queue_tail, _visited_gpu);
  
  q_sz = _get_queue_size();
  

  _frist_level<_k> <<< num_blocks, num_threads >>> (_num_PIs,
                                                    _adj_gpu, 
                                                    _adj_index_table_gpu, 
                                                    _invAdj_gpu, 
                                                    _invAdj_index_table_gpu,
                                                    _queue_data, _queue_head, _queue_tail, 
                                                    _visited_gpu,
                                                    _partitionIndex_gpu);
  cudaCheckErrors("CUDA: _frist_level - Failure");
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  q_sz = _get_queue_size();
  
  // cudaDeviceSynchronize();
  // cudaCheckErrors("CUDA: cudaDeviceSynchronize _frist_level - Failure");

  // _print_first_level <<< 1, 1 >>> (_num_PIs, _partitionIndex_gpu);
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
                                                      _visited_gpu,
                                                      _partitionIndex_gpu);
                                                      
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
  _check_visited <<< 1, 1 >>> (_sum_pi_gates_pos, _visited_gpu, 1);
  // printf("_num_PIs = %d, _num_POs = %d, _num_inner_gates = %d\n", _num_PIs, _num_POs, _num_inner_gates);
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");

  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");
  _print_GPU_partitioner <<< 1, 1 >>> (_sum_pi_gates_pos, _partitionIndex_gpu);
  cudaCheckErrors("CUDA: _print_GPU_partitioner - Failure");    
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize - Failure");

  _print_full_queue <<< 1, 1 >>> (_queue_data, _queue_tail);
  cudaCheckErrors("CUDA: _print_full_queue - Failure");
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA: cudaDeviceSynchronize _print_first_level - Failure");  
#endif 
}


// --------------------------------------------------------------------------------------------------

/* Prepare for simulation */
void CUDAPartitioner::_copy_TS_resutls_to_Host() {
  _queue_data_cpu.resize(28*_szOfAdj);
  _partitionIndex_cpu.resize(_sum_pi_gates_pos); 
  _queue_tail_cpu = 0;

  cudaMemcpyAsync(_queue_data_cpu.data(), _queue_data, 
                  28*_szOfAdj*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(_partitionIndex_cpu.data(), _partitionIndex_gpu, 
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
    // traverse its inputs 
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


  // Construct _gateIdx_in_each_level
  _gateIdx_in_each_level.resize(_total_num_levels);
  for (int gateIdx = 0; gateIdx < _level_of_gates.size(); gateIdx++) {
    int gateLvl = _level_of_gates[gateIdx];
    _gateIdx_in_each_level[gateLvl].push_back(gateIdx);
  }

  _threshold_0_level = _max_level;
  while(_threshold_0_level != -1 && 
        _gateIdx_in_each_level[_threshold_0_level].size() <= _THRESHOLD_0) {
    _threshold_0_level--;
  }  


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

void CUDAPartitioner::_get_sinks_group(std::vector<std::vector<int>> &sinks_groups) {
  // Get the group of sinks 
  sinks_groups.resize(_k);

  // Get sinks 
  for (int sinkIdx = 0; sinkIdx < _sum_pi_gates_pos; sinkIdx++) {
    int outputSize = _adj_index_table[2*sinkIdx+1] - _adj_index_table[2*sinkIdx+0];
    if (outputSize == 0) {
      int sinkGroup = _partitionIndex_cpu[sinkIdx];
      sinks_groups[sinkGroup].push_back(sinkIdx);
    }
  }
}

void CUDAPartitioner::_construct_groups(){
  std::vector<std::vector<int>> sinks_groups;
  _get_sinks_group(sinks_groups);

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

void CUDAPartitioner::_construct_graph() {
  _Gates.resize(_sum_pi_gates_pos);

  for (int toGate = 0; toGate < _sum_pi_gates_pos; toGate++) {
    int toGateIdx = toGate;
    _Gates[toGate]._idx = toGateIdx;
    _Gates[toGate]._type = _gate_type[toGate];
    _Gates[toGate]._level = _level_of_gates[toGate];
    _Gates[toGate]._output_value = 0;

    // for (int fmGate = 0; fmGate < _invAdj[toGate].size(); fmGate++) {
    for (int fmGate = _invAdj_index_table[2*toGate+0]; 
        fmGate < _invAdj_index_table[2*toGate+1]; 
        fmGate++) {
      int fmGateIdx = _invAdj[fmGate];
      ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &from_gate =
        _Gates[fmGateIdx];
      ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &to_gate =
        _Gates[toGateIdx];
      to_gate._inputs.push_back(&from_gate);
    }
  }

#ifdef CPU_PART_DEBUG_PRINT_GRAPH
  std::cout << "\n_Gates = [\n";
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    _print_ppg(_Gates[i]);
  }
  std::cout << "]\n";
#endif  
}

void CUDAPartitioner::_construct_partitioned_groups(const int simulator) {
  // construct group and show constructed group 
  _construct_groups();

  
  switch (simulator) {
    case 0: { // CPU simulator
      // Construct _cones_partitioned
      for (size_t i = 0; i < _cones_set.size(); ++i) {
        for (size_t lvl = 0; lvl < _cones_set[i].size(); ++lvl) {
          for (const auto& element : _cones_set[i][lvl]) {
            _cones_partitioned[i][lvl].push_back(element);
          }
        }
      }  

      // _count_duplications(simulator);
      break;  
    }
    case 1: { // GPU simulator 
      // Prepare for the part of levelization
        // construct _per_level_of_group_start_accum_level_tmp
        // construct _gateIdx_in_each_level_tmp
      int accum = 0;
      _per_level_of_group_start_accum_level_tmp.push_back(accum);

      for (size_t level = 0; level < _gateIdx_in_each_level.size(); level++) {
        for (size_t gate = 0; gate < _gateIdx_in_each_level[level].size(); gate++) {
          int gateIdx = _gateIdx_in_each_level[level][gate];
          _gateIdx_in_each_level_tmp.push_back(gateIdx);
          accum++;
        }
        _per_level_of_group_start_accum_level_tmp.push_back(accum);
      }
   

      // Prepare for the part of partitioned groups
        // construct _cones_partitioned_tmp
        // construct _numGates_per_level_gpu_of_groups
        // construct _per_level_of_group_start_accum_tmp
      accum = 0;
      _per_level_of_group_start_accum_tmp.push_back(accum);

      for (size_t i = 0; i < _cones_set.size(); ++i) {
        for (size_t lvl = 0; lvl < _cones_set[i].size(); ++lvl) {
          int accum2 = 0;
          for (const auto& element : _cones_set[i][lvl]) {
            _cones_partitioned_tmp.push_back(element);
            accum2++;
            accum++;         
          }
          _numGates_per_level_gpu_of_groups.push_back(accum2);
          _per_level_of_group_start_accum_tmp.push_back(accum);
        }
      }


      //   printf("_numGates_per_level_gpu_of_groups = [");
      //   for (size_t i = 0; i < _numGates_per_level_gpu_of_groups.size(); i++) {
      //     printf("%d, ", _numGates_per_level_gpu_of_groups[i]);
      //   } printf("]\n");

      //   printf("_per_level_of_group_start_accum_tmp = [");
      //   for (size_t i = 0; i < _per_level_of_group_start_accum_tmp.size(); i++) {
      //     printf("%d, ", _per_level_of_group_start_accum_tmp[i]);
      //   } printf("]\n");

      //   printf("_cones_partitioned_tmp = [");
      //   for (size_t i = 0; i < _cones_partitioned_tmp.size(); i++) {
      //     printf("%d, ", _cones_partitioned_tmp[i]);
      //   } printf("]\n");

      //   std::vector<bool> vv; vv.resize(_sum_pi_gates_pos, false);
      //   for (size_t i = 0; i < _cones_partitioned_tmp.size(); i++) {
      //     vv[_cones_partitioned_tmp[i]] = true;
      //   }
      //   size_t acc = 0;
      //   for (size_t i = 0; i < vv.size(); i++) {
      //     if (!vv[i]) {
      //       acc++;
      //     }
      //   } 
      //   printf("haha acc = %lu (should == 0)\n", acc);
        


      break;
    }
    default: {
      printf("No SUCH CASES\n");
      return; 
    }
  }

}

void CUDAPartitioner::prepare_cpu_simulation() {
  auto start = std::chrono::steady_clock::now();
  const int simulator = 0;
  _construct_partitioned_groups(simulator);
  // Note: _construct_graph need to be run after _construct_partitioned_groups
  _construct_graph();
  _ask_simulation_memory();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  // std::cout << "prepare_CPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";

  _count_duplications(simulator);
}


// ----------------


void CUDAPartitioner::_ask_gpu_simulation_memory() {
  // Memory allocation
  cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, _sum_pi_gates_pos*sizeof(int));
  cudaMalloc((void**)&_patterns_gpu, _num_rounds*_num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_fault_gate_idx_gpu, _num_fault*sizeof(int));
  cudaMalloc((void**)&_fault_SA_fault_val_gpu, _num_fault*sizeof(size_t));
  
  cudaMalloc((void**)&_pi_gate_po_output_res_gpu, _sum_pi_gates_pos*sizeof(size_t));
  cudaMalloc((void**)&_g_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_g_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_g_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_b_pi_results_gpu, _num_PIs*sizeof(size_t));
  cudaMalloc((void**)&_b_gate_results_gpu, _num_inner_gates*sizeof(size_t));
  cudaMalloc((void**)&_b_po_results_gpu, _num_POs*sizeof(size_t));
  cudaMalloc((void**)&_found_fault_to_pattern_gpu, 2*_num_fault*sizeof(int));  

  cudaMalloc((void**)&_per_level_of_group_start_accum_gpu, (_per_level_of_group_start_accum_tmp.size())*sizeof(int));
  cudaMalloc((void**)&_cones_partitioned_gpu, (_cones_partitioned_tmp.size())*sizeof(int));
  cudaMalloc((void**)&_numGates_per_level_gpu_of_groups_gpu, (_numGates_per_level_gpu_of_groups.size())*sizeof(int));
  
  cudaMalloc((void**)&_per_level_of_group_start_accum_level_gpu, (_per_level_of_group_start_accum_level_tmp.size())*sizeof(int));
  cudaMalloc((void**)&_gateIdx_in_each_level_gpu, (_gateIdx_in_each_level_tmp.size())*sizeof(int));
  cudaCheckErrors("CUDA: ask gpu simulation memory - cudaMalloc - Failure");

  cudaMemcpyAsync(_per_level_of_group_start_accum_gpu, _per_level_of_group_start_accum_tmp.data(), 
                  (_per_level_of_group_start_accum_tmp.size())*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_cones_partitioned_gpu, _cones_partitioned_tmp.data(), 
                  (_cones_partitioned_tmp.size())*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_numGates_per_level_gpu_of_groups_gpu, _numGates_per_level_gpu_of_groups.data(), 
                  (_numGates_per_level_gpu_of_groups.size())*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_per_level_of_group_start_accum_level_gpu, _per_level_of_group_start_accum_level_tmp.data(), 
                  (_per_level_of_group_start_accum_level_tmp.size())*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpyAsync(_gateIdx_in_each_level_gpu, _gateIdx_in_each_level_tmp.data(), 
                  (_gateIdx_in_each_level_tmp.size())*sizeof(int), cudaMemcpyHostToDevice);                   
  cudaCheckErrors("CUDA: ask gpu simulation memory - cudaMemcpyAsync - Failure");
}

void CUDAPartitioner::_move_GateType_h2d() {
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

void CUDAPartitioner::_move_patterns_h2d() {
  std::vector<size_t> patterns_cpu;
  for (size_t i = 0; i < _num_rounds; i++) {
    for (int pi = 0; pi < _num_PIs; pi++) {
      patterns_cpu.push_back(_patterns[i]._value[pi]);
    }
  }

  cudaMemcpy(_patterns_gpu, patterns_cpu.data(), 
            (_num_rounds*_num_PIs)*sizeof(size_t), cudaMemcpyHostToDevice);
  cudaCheckErrors("CUDA: _patterns_gpu cudaMemcpy failure");

#ifdef GPU_PREPARE_SIMULATION_PRINT_CHECK
  cudaDeviceSynchronize();
  print_patterns_gpu <<< 1, 1 >>> (_patterns_gpu, _num_rounds, _num_PIs);
  cudaCheckErrors("CUDA: print_patterns_gpu failure");
  cudaDeviceSynchronize();
#endif
}

void CUDAPartitioner::_move_faults___h2d() {
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

void CUDAPartitioner::prepare_gpu_simulation() {
  auto start = std::chrono::steady_clock::now();
  const int simulator = 1;
  _construct_partitioned_groups(simulator);
  // Copy data from Host to Device for GPU simulation 
  _ask_gpu_simulation_memory();
    // cudaDeviceSynchronize();
    // print_ask_gpu_simulation_memory <<< 1, 1 >>> (_per_level_of_group_start_accum_tmp.size(), 
    //                                               _per_level_of_group_start_accum_gpu,
    //                                               _cones_partitioned_tmp.size(), 
    //                                               _cones_partitioned_gpu);
    // cudaCheckErrors("CUDA: print_ask_gpu_simulation_memory - Failure");
    // cudaDeviceSynchronize();

  _move_GateType_h2d();
  _move_patterns_h2d();
  _move_faults___h2d(); 
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  // std::cout << "prepare_GPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";
  _count_duplications(simulator);
}


// --------------------------------------------------------------------------------------------------

// Simulation functions 
void CUDAPartitioner::run(Mode mode, const size_t num_threads, const size_t NUM_SIMULATION_RDS) {  
  // int m = static_cast<int>(mode);

  switch (mode) {
    case Mode::GPU_PARTIOR_CPU_SIMUTOR: {
      // std::cout << "Mode: " << m << " (run GPU_PARTIOR_CPU_SIMUTOR)\n";
      GALPS_CPUSimulator<ELEMENT_INDEX_TYPE_PART> cpuSimulator;
      cpuSimulator.run_cpu_simulator_cones(num_threads, 
                                          _num_PIs, _num_inner_gates, _num_POs, 
                                          _sum_pi_gates_pos, 
                                          _num_pattern, _num_rounds, _num_fault,
                                          _patterns, _faults, _Gates,
                                          _cones_partitioned,
                                          _g_pi_results,
                                          _g_gate_results,
                                          _g_po_results,
                                          _b_pi_results,
                                          _b_gate_results,
                                          _b_po_results,
                                          _found_fault_to_pattern);
      break;  
    }

    case Mode::GPU_PARTIOR_GPU_SIMUTOR: {
      // std::cout << "Mode: " << m << " (run GPU_PARTIOR_GPU_SIMUTOR)\n";
      GALPS_GPUSimulator gpuSimulator;
      // gpuSimulator.run_gpu_simulator_cones_part_gpu(_k, _num_PIs, _num_inner_gates, _num_POs, 
      //                                               _sum_pi_gates_pos, 
      //                                               _num_pattern, _num_rounds, _num_fault,
      //                                               _pi_gate_po_gate_type_gpu, 
      //                                               _patterns_gpu,
      //                                               _fault_gate_idx_gpu,
      //                                               _fault_SA_fault_val_gpu,
      //                                               _pi_gate_po_output_res_gpu,
      //                                               _gateIdx_in_each_level,
      //                                               _gateIdx_in_each_level_gpu,
      //                                               _per_level_of_group_start_accum_level_gpu,
      //                                               _threshold_0_level,
      //                                               _numGates_per_level_gpu_of_groups_gpu,
      //                                               _per_level_of_group_start_accum_gpu,
      //                                               _cones_partitioned_gpu,
      //                                               _total_num_levels,
      //                                               _invAdj_gpu,
      //                                               _invAdj_index_table_gpu,
      //                                               _patterns, 
      //                                               _g_pi_results_gpu,
      //                                               _g_gate_results_gpu,
      //                                               _g_po_results_gpu,
      //                                               _b_pi_results_gpu,
      //                                               _b_gate_results_gpu,
      //                                               _b_po_results_gpu,
      //                                               _found_fault_to_pattern_gpu,
      //                                               NUM_SIMULATION_RDS);
      gpuSimulator.run_gpu_simulator_cones_part_gpu_cuda_graph(_k, _num_PIs, _num_inner_gates, _num_POs, 
                                                              _sum_pi_gates_pos, 
                                                              _num_pattern, _num_rounds, _num_fault,
                                                              _pi_gate_po_gate_type_gpu, 
                                                              _patterns_gpu,
                                                              _fault_gate_idx_gpu,
                                                              _fault_SA_fault_val_gpu,
                                                              _pi_gate_po_output_res_gpu,
                                                              _gateIdx_in_each_level,
                                                              _gateIdx_in_each_level_gpu,
                                                              _per_level_of_group_start_accum_level_gpu,
                                                              _threshold_0_level,
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
}

// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------

void CUDAPartitioner::_print_patterns(const std::vector<Pattern> &patterns,
                                    const int round,
                                    const int num_PIs) const {
  std::cout << "\n=====\n\n";
    for (int i = 0; i < round; i++) {
      std::cout << "[" << SIZE_T_BITS * i << ", " << SIZE_T_BITS * (i + 1)
                << "] bits = [\n";
      for (int j = 0; j < num_PIs; j++) {
        _print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
      }
    std::cout << "]\n";
  }
}

void CUDAPartitioner::_print_bits_stack(const int size,
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
gpuGateType CUDAPartitioner::_convertGateTypeToGpu(GateType gate_type) {
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
std::string CUDAPartitioner::_gateTypeToString(GateType type) const {
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

void CUDAPartitioner::_print_read_graph() const {
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
                                   const int *_partitionIndex_gpu) {
  printf("Get Insdies _print_first_level\n");
  
  for (int i = 0; i < _num_PIs; i++) {
    const int piIdx = i;
    const int groupIdx = _partitionIndex_gpu[piIdx];
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

__global__ void _print_full_queue(int *_queue_data, int *_queue_tail) {
  int start = 0;
  int endpt = *_queue_tail;

  printf("Get Insdies _print_full_queue, start = %d, endpt = %d\n", start, endpt);

  printf("queue = [");
  for (int i = start; i < endpt; i++) {
    printf("%d, ", _queue_data[i]);
  } printf("]\n");
  printf("\n");
}

__global__ void _print_GPU_partitioner(const int _sum_pi_gates_pos, 
                                      const int *_partitionIndex_gpu) {
                  
  printf("Get Insdies _print_GPU_partitioner\n");
  
  for (int i = 0; i < _sum_pi_gates_pos; i++) {
    const int gateIdx = i;
    const int groupIdx = _partitionIndex_gpu[gateIdx];
    printf("gateIdx = %d, groupIdx = %d\n", gateIdx, groupIdx);
  }
}

__global__ void _check_visited(const int sz, const int *_visited_gpu, const int mode) {
  
  int ret = 0;
  for (int i = 0; i < sz; i++) {
    if (_visited_gpu[i] == 0) {
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

  printf("_threshold_0_level = %d\n", _threshold_0_level);
  for (size_t i = 0; i < _gateIdx_in_each_level.size(); i++) {
    printf("level_%lu, size = %lu\n", i, _gateIdx_in_each_level[i].size());
  }  
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

void CUDAPartitioner::_count_duplications(const int simulator) {
  switch (simulator) {
    case 0: { // CPU simulator
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
        printf("_part_visited_cpu: accum = %d (should == 0)\n", accum);
      break;  
    }
    case 1: { // GPU simulator 
      std::vector<bool> visited_cpu;
      visited_cpu.resize(_sum_pi_gates_pos, false);
      std::vector<int> numGates; 
      numGates.resize(4, 0);

      // Levelize
      for (int level = 0; level < (_threshold_0_level+1); level++) {
        for (size_t gate = 0; gate < _gateIdx_in_each_level[level].size(); gate++) {
          int element = _gateIdx_in_each_level[level][gate];
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

      // Partitioning
      for (int k = 0; k < _k; k++) {
        size_t s_loc = _per_level_of_group_start_accum_tmp[k*_total_num_levels+_threshold_0_level+1];
        size_t e_loc = _per_level_of_group_start_accum_tmp[(k+1)*_total_num_levels];
        // printf("k = %d, s_loc = %lu, e_loc = %lu\n", k, s_loc, e_loc);
        for (size_t i = s_loc; i < e_loc; i++) {
          int element = _cones_partitioned_tmp[i];
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

      numGates[3] = numGates[0]+numGates[1]+numGates[2];
    
      int accum = 0;
      for (size_t i = 0; i < visited_cpu.size(); i++) {
        if (!visited_cpu[i]) {
          accum++;
        }
      }
      
      printf("Partitioned:\n");
      printf("_part_PIs = %d, %d\n", numGates[0], _num_PIs);
      printf("_part_Gates = %d, %d\n", numGates[1], _num_inner_gates);
      printf("_part_POs = %d, %d\n", numGates[2], _num_POs);
      if (numGates[3] == _sum_pi_gates_pos) {
        printf("The same! _sum_pi_gates_pos = %d\n", _sum_pi_gates_pos);
      } else {
        printf("_part_sum_pi_gates_pos = %d, %d\n", numGates[3], _sum_pi_gates_pos);
      }
      printf("_part_visited_cpu: accum = %d (should == 0)\n", accum);
      printf("_threshold_0_level = %d, _total_num_levels = %d\n", 
              _threshold_0_level, _total_num_levels);

      break;
    }
    default: {
      printf("No SUCH CASES\n");
      return; 
    }
  }  
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

void CUDAPartitioner::_print_simulation_results(
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


void CUDAPartitioner::_print_ppg(const ElementBase<ELEMENT_INDEX_TYPE_PART, ELEMENT_LEVEL_TYPE_PART> &gate) const {
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


template<typename IndexType>
__global__ void _check_levelize_gpu(const IndexType *_max_level_gpu, 
                                    const IndexType *_total_num_levels_gpu, 
                                    const IndexType _sum_pi_gates_pos, 
                                    const IndexType *_level_of_gates_gpu) {
  printf("Get inside _check_levelize_gpu\n");
  printf("_level_of_gates_gpu:\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%lu, level = %d\n", i, _level_of_gates_gpu[i]);
  }
  printf("_max_level_gpu = %d, _total_num_levels_gpu = %d\n", *_max_level_gpu, *_total_num_levels_gpu);
  printf("\n");  
}



template<typename IndexType>
__global__ void _print_construct_cones_partitioned_gpu(IndexType *_total_numGates_partitioned_gpu,
                                                      IndexType *_cones_partitioned_gpu) {
  printf("_cones_partitioned_gpu = [\n");
  for (IndexType i = 0; i < *_total_numGates_partitioned_gpu; i++) {
    printf("%d, ", _cones_partitioned_gpu[i]);
  }                                                      
  printf("]\n\n");
}

template<typename IndexType>
__global__ void print_pi_gate_po_gate_type_gpu(const IndexType *_pi_gate_po_gate_type_gpu, const IndexType num) {
  printf("print_pi_gate_po_gate_type_gpu (num = %d): [\n", num);
  for (IndexType i = 0; i < num; i++) {
    IndexType type = _pi_gate_po_gate_type_gpu[i];
    switch (type) {
      case 0:
        printf("i = %d, type_INV\n", i);
        break;
      case 1:
        printf("i = %d, type_AND\n", i);
        break;
      case 2:
        printf("i = %d, type_OR\n", i);
        break;
      case 3:
        printf("i = %d, type_XOR\n", i);
        break;
      case 4:
        printf("i = %d, type_NAND\n", i);
        break;
      case 5:
        printf("i = %d, type_NOR\n", i);
        break;
      case 6:
        printf("i = %d, type_XNOR\n", i);
        break;
      case 7:
        printf("i = %d, type_MUX\n", i);
        break;
      case 8:
        printf("i = %d, type_CLKBUF\n", i);
        break;
      case 9:
        printf("i = %d, type_PI\n", i);
        break;
      case 10:
        printf("i = %d, type_PO\n", i);
        break;
      case 11:
        printf("i = %d, type_UNKNOWN\n", i);
        break;
    }
  }
  printf("]\n");
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

template<typename IndexType>
__global__ void print_fault_gate_idx_gpu(IndexType *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, IndexType _num_fault) {
  printf("print_fault_gate_idx_gpu (_num_fault = %d): [\n", _num_fault);
  for (size_t i = 0; i < _num_fault; i++) {
    printf("gate_w_f = %d, SA_val = %ld\n", _fault_gate_idx_gpu[i], _fault_SA_fault_val_gpu[i]);
  } printf("]\n");
}



__global__ void check_cones_partitioned_gpu_visited(int *visited, 
                                                    const int *_cones_partitioned_gpu, 
                                                    const int num1, 
                                                    const int total_numGates_partitioned) {
  for (int i = 0; i < 1498565; i++) {
    visited[i] = 0;
  }

  for (int i = 0; i < total_numGates_partitioned; i++) {
    int gateIdx = _cones_partitioned_gpu[i];
    visited[gateIdx] = 1; 
  }

  printf("num1 = %d, 1498565\n", num1);
  size_t miss_accum = 0;
  for (int i = 0; i < num1; i++) {
    if (visited[i] == 0) {
      miss_accum++;
    }
  }
  printf("GPU miss_accum = %lu\n", miss_accum);
}



__global__ void print_ask_gpu_simulation_memory(const size_t sz1, 
                                                const int *_per_level_of_group_start_accum_gpu,
                                                const size_t sz2,
                                                const int *_cones_partitioned_gpu) {
  printf("_per_level_of_group_start_accum_gpu :[");
  for (int i = 0; i < sz1; i++) {
    printf("%d, ", _per_level_of_group_start_accum_gpu[i]);  
  } printf("]\n");

  printf("_cones_partitioned_gpu :[");
  for (int i = 0; i < sz2; i++) {
    printf("%d, ", _cones_partitioned_gpu[i]);  
  } printf("]\n");  
}