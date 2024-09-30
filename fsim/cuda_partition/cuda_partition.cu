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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "cuda_partition.cuh"

// #define CPU_READ_GRAPH_PRINT_FOR_CHECK
// #define CUDA_READ_GRAPH_PRINT_FOR_CHECK
#define CUDA_LEVELIZATION_PRINT_FOR_CHECK

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
  // _read_fault(flst);
  // _read_pattern(ptn);

  /* Partitioning */
  _malloc_transfer_devc_memory();
  _levelization();
  // // _partition();
}

void CUDAPartitioner::_read_graph(std::istream &ckt) {
  _sum_pi_gates_pos = _num_PIs + _num_inner_gates + _num_POs;

  // Map table for recording gate input _order
  // Record for each to_gate, its from_gates's are which _order (#_inputs, A/A1, B/A2, A3/S, A4)
  _gate_input_order_table = (int*)malloc(5*_sum_pi_gates_pos*sizeof(int));
  memset(_gate_input_order_table, 0, 5*_sum_pi_gates_pos*sizeof(int));

  _szOfAdj = 0;
  std::vector<std::vector<size_t>> adj;
  adj.resize(_sum_pi_gates_pos);

  // Construct adjacent list
  for (size_t i = 0; i < _num_wires; i++) {
    int gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;

    for (int j = 0; j < num_post_gates; j++) {
      int gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      pin_tmp = (pin_tmp == 5) ? (3) : (pin_tmp);
      adj[gate_0].push_back(gate_tmp); 
      _szOfAdj++;

      _gate_input_order_table[gate_tmp*5+pin_tmp] = gate_0;
      _gate_input_order_table[gate_tmp*5+0]++;
    }
  }
  
  // Read GateType
  _gate_type.resize(_sum_pi_gates_pos);
  for (size_t i = 0; i < _num_PIs; i++) {
    _gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs + _num_POs); i++) {
    _gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs + _num_POs); i < (_sum_pi_gates_pos); i++) {
    size_t type;
    ckt >> type;
    _gate_type[i] = static_cast<GateType>(type);
  }


  // Construct _gate_output_order_table
  // printf("_szOfAdj = %lu\n", _szOfAdj);
  _gate_output_order_table = (int*)malloc(_szOfAdj*sizeof(int));
  _gate_output_order_table_index = 
                    (int*)malloc(2*_sum_pi_gates_pos*sizeof(int));
  memset(_gate_output_order_table, 0, 
        _szOfAdj*sizeof(int));
  memset(_gate_output_order_table_index, 0, 
        2*_sum_pi_gates_pos*sizeof(int));
      
  size_t accum = 0;
  for (size_t fromGate = 0; fromGate < adj.size(); fromGate++) {
    // Update _gate_output_order_table_index
    _gate_output_order_table_index[2*fromGate+0] = accum;

    for (size_t toGate = 0; toGate < adj[fromGate].size(); toGate++) {
      size_t toGateIdx = adj[fromGate][toGate];
      
      // Update _gate_output_order_table
      _gate_output_order_table[accum] = toGateIdx;
      accum++;
    }
    // Update _gate_output_order_table_index
    _gate_output_order_table_index[2*fromGate+1] = accum;
  }

#ifdef CPU_READ_GRAPH_PRINT_FOR_CHECK
  printf("_gate_input_order_table:\n");
  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    printf("gateIdx_%lu, with Inputs = [", g, _gate_input_order_table[g*5+0]);
    for (size_t j = 1; j < _gate_input_order_table[g*5+0]+1; j++) {
      printf("%d, ", _gate_input_order_table[g*5+j]);
    }
    printf("]\n");
  }

  printf("_gate_type:\n");
  for (size_t i = 0; i < _gate_type.size(); i++) {
    std::cout << "Gate_" << i << ", _gate_type = " << gateTypeToString(_gate_type[i]) << "\n";
  } printf("\n");
  
  printf("_gate_output_order_table:\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("gate_%lu's output = ", i);
    for (size_t j = _gate_output_order_table_index[2*i+0]; 
                j < _gate_output_order_table_index[2*i+1]; 
                j++) {
      printf("%d, ", _gate_output_order_table[j]);
    }
    printf("\n");
  }
  printf("\n");  
#endif 
}


void CUDAPartitioner::_read_fault(std::istream &flst) {
  flst >> _num_fault;
  _faults.resize(_num_fault);

  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;

    _faults[i]._gate_with_fault = wrong_gate;
    _faults[i]._gate_SA_fault_val =
        (fault_value) ? std::numeric_limits<size_t>::max() : (0);
  }
}

void CUDAPartitioner::_read_pattern(std::istream &ptn) {
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


// GPU functions

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

void CUDAPartitioner::_malloc_transfer_devc_memory() {
  // printf("Get Inside _malloc_transfer_devc_memory\n");

  /* Allocate memory on the GPU */
  // For levelize and partition
  cudaMalloc(&_gate_output_order_table_devc, _szOfAdj*sizeof(int));
  cudaMalloc(&_gate_output_order_table_index_devc, 2*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc(&_gate_input_order_table_devc, 5*_sum_pi_gates_pos*sizeof(int));
  cudaMalloc(&_gate_type_devc, _sum_pi_gates_pos*sizeof(gpuGateType));
  cudaMalloc(&_level_of_gates_devc, _sum_pi_gates_pos*sizeof(int));
  cudaMalloc(&_max_level_devc, 1*sizeof(int));
  cudaMalloc(&_total_num_levels_devc, 1*sizeof(int));
  
  // TODO: _gateIdx_in_each_level_devc needs a table to record the level by level start/end points 
  // cudaMalloc(&_gateIdx_in_each_level_devc, _sum_pi_gates_pos*sizeof(int));

  // Convert each GateType to gpuGateType
  std::vector<gpuGateType> host_gpu_gate_type(_sum_pi_gates_pos);
  for (size_t i = 0; i < _gate_type.size(); ++i){
    host_gpu_gate_type[i] = convertGateTypeToGpu(_gate_type[i]);
  }

  
  /* Copy the data from host to device */
  // For levelize and partition
  cudaMemcpy(_gate_output_order_table_devc, _gate_output_order_table, 
            _szOfAdj*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_gate_output_order_table_index_devc, _gate_output_order_table_index, 
            2*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice);            
  cudaMemcpy(_gate_input_order_table_devc, _gate_input_order_table, 
            5*_sum_pi_gates_pos*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_gate_type_devc, host_gpu_gate_type.data(), 
            _sum_pi_gates_pos*sizeof(gpuGateType), cudaMemcpyHostToDevice);

    
#ifdef CUDA_READ_GRAPH_PRINT_FOR_CHECK
  // check the memory copy's correctness 
  printf("Before Inside CUDA_READ_GRAPH_PRINT_FOR_CHECK\n");
  _print_malloc_transfer_devc_memory <<< 1, 1 >>> (_sum_pi_gates_pos, 
                                                  _gate_input_order_table_devc, 
                                                  _gate_type_devc);
  cudaDeviceSynchronize();
  cudaCheckErrors("_print_malloc_transfer_devc_memory failed");
#endif 
}


void CUDAPartitioner::_levelization() {
  size_t numThreads = (_num_PIs > 1024) ? (1024) : (_num_PIs);
  bfs_queue_levelize <<< 1, numThreads, (_num_PIs+4)*sizeof(int) >>> (_num_PIs, _sum_pi_gates_pos, 
                        _max_level_devc, _total_num_levels_devc, 
                        _level_of_gates_devc, 
                        _gate_output_order_table_devc, _gate_output_order_table_index_devc);
  cudaDeviceSynchronize();
  cudaCheckErrors("bfs_queue_levelize failed");

#ifdef CUDA_LEVELIZATION_PRINT_FOR_CHECK
  _print_levelization <<< 1, 1 >>> (_max_level_devc, _total_num_levels_devc, 
                                    _sum_pi_gates_pos, _level_of_gates_devc);
  cudaDeviceSynchronize();
  cudaCheckErrors("CUDA_LEVELIZATION_PRINT_FOR_CHECK failed");
#endif 
}


__device__ void enqueue(int *queue, int *tail, int element, 
                        const int capacity) {

  // Atomically increment the tail pointer and get the old value
  size_t position = atomicAdd(tail, 1) % capacity;
  queue[position] = element;
}

__device__ int dequeue(int *queue, int *head, int tail, 
                          const int capacity) {

  // Atomically increment the head pointer and get the old value
  size_t position = atomicAdd(head, 1) % capacity;

  // Ensure we don't dequeue beyond what has been enqueued
  if (position < tail) {
    return queue[position];
  } else {
    // Queue is empty
    queue[position] = -1;
    return -1;
  }
}

// NOTE: here ERROR -> WA for
__global__ void bfs_queue_levelize(const size_t _num_PIs, const size_t _sum_pi_gates_pos, 
                                int *_max_level_devc, int *_total_num_levels_devc,
                                int *_level_of_gates_devc,
                                int *_gate_output_order_table_devc,
                                int *_gate_output_order_table_index_devc) {
  extern __shared__ int queue[];    // Shared memory for the queue
  __shared__ int head, tail, numOfJobs; // Shared variables for the queue head and tail
  const int capacity = _num_PIs;  // Queue capacity (shared memory limit)

  int t_idx = threadIdx.x;

  // Initialize the queue with primary inputs (PIs)
  int rounds = (_num_PIs + blockDim.x - 1) / blockDim.x;
  for (int rd = 0; rd < rounds; rd++) {
    int gateIdx = rd * blockDim.x + t_idx;
    if (gateIdx < _num_PIs) { 
      queue[gateIdx] = gateIdx;
      _level_of_gates_devc[gateIdx] = 0; // Initialize PIs to level 0
    }
  }

  head = 0;           // Queue head pointer
  tail = _num_PIs;     // Queue tail pointer
  _max_level_devc[0] = 0;  // Initialize the maximum level
  _total_num_levels_devc[0] = 0;
  numOfJobs = _num_PIs;
  int maxlevelOfEachThread = 0; 
  __syncthreads();  

  // BFS levelization loop
  while (numOfJobs > 0) {
    int getJob = atomicSub(&numOfJobs, 1);
    if (getJob <= 0) {
      atomicAdd(&numOfJobs, 1);
      continue;
    } 

    // Dequeue the next gate from the queue
    int gate = dequeue(queue, &head, tail, capacity);

    // Traverse the adjacency list of the current gate (fan-out nodes)
    for (int j = _gate_output_order_table_index_devc[2*gate+0]; 
            j < _gate_output_order_table_index_devc[2*gate+1]; 
            j++) {
      int postGateIdx = _gate_output_order_table_devc[j];
      int new_level = _level_of_gates_devc[gate] + 1;

      // Use atomicMax to ensure correct level update only if the new level is greater
      int old_level = atomicMax(&_level_of_gates_devc[postGateIdx], new_level);
      // printf("\tt_idx = %d, gate_%d, postGateIdx_%d (%d %d), numOfJobs = %d\n", 
      //         t_idx, gate, postGateIdx, new_level, old_level, numOfJobs);

      if (new_level > old_level) {
        enqueue(queue, &tail, postGateIdx, capacity);
        atomicAdd(&numOfJobs, 1);
        // printf("\t\tt_idx = %d, Enqueuing postGateIdx_%d, update gate_%d to level = %d, numOfJobs = %d\n", 
        //       t_idx, postGateIdx, 
        //       postGateIdx, _level_of_gates_devc[postGateIdx],
        //       numOfJobs);  // Add this for debugging
      }

      // Track the maximum level for this thread
      maxlevelOfEachThread = max(maxlevelOfEachThread, new_level);
    }

    // Atomically update the global max level across threads
    atomicMax(&_max_level_devc[0], maxlevelOfEachThread);
  }

  __syncthreads();

  // Update the total number of levels (max level + 1)
  if (t_idx == 0) {
    _total_num_levels_devc[0] = _max_level_devc[0] + 1;
  } 
}












void CUDAPartitioner::_free_devc_memory() {
  cudaFree(_gate_input_order_table_devc);
  cudaFree(_gate_type_devc);
  cudaFree(_level_of_gates_devc);
  cudaFree(_max_level_devc);
  cudaFree(_total_num_levels_devc);
}



// Simulation functions 
void CUDAPartitioner::run(Mode mode) {
  size_t m = static_cast<size_t>(mode);

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

void CUDAPartitioner::print_bits_stack(const size_t size,
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

__global__ void _print_malloc_transfer_devc_memory(const size_t _sum_pi_gates_pos, 
                                                  const int *_gate_input_order_table_devc, 
                                                  const gpuGateType *_gate_type_devc) {
  printf("_gate_input_order_table_devc:\n");
  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    printf("gateIdx_%lu, with Inputs (%d) = [", g, _gate_input_order_table_devc[g*5+0]);
    for (size_t j = 1; j < _gate_input_order_table_devc[g*5+0]+1; j++) {
      printf("%d, ", _gate_input_order_table_devc[g*5+j]);
    }
    printf("]\n");
  }

  // Print _gate_type_devc:
  printf("_gate_type_devc:\n");
  for (size_t i = 0; i < _sum_pi_gates_pos; i++) {
    printf("Gate_%lu, _gate_type = %s\n", i, gpuGateTypeToString(_gate_type_devc[i]));
  }
  printf("\n");
}

__global__ void _print_levelization(int *_max_level_devc, int *_total_num_levels_devc,
                                    const size_t _sum_pi_gates_pos, 
                                    int *_level_of_gates_devc) {
  printf("_max_level_devc = %d, _total_num_levels_devc = %d\n", 
          _max_level_devc[0], _total_num_levels_devc[0]);

  printf("_level_of_gates_devc:\n");
  for (size_t g = 0; g < _sum_pi_gates_pos; g++) {
    int level = _level_of_gates_devc[g];
    printf("gate_%lu, level = %d\n", g, level);
  }
}

