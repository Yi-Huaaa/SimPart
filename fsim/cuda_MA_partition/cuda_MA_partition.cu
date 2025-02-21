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

#include "cuda_MA_partition.cuh"


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
  _levelized();

  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_GPU_partition = end - start;

  std::cout << "duration_GPU_partition: " << _round_to((duration_GPU_partition.count())*1000, 0.001) << "\n";
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

  _print_read_graph();
}

void CUDAMAPartitioner::_read_fault(std::istream &flst) {
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

void CUDAMAPartitioner::_read_pattern(std::istream &ptn) {
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

/* Prepare for GPU simulation */
void CUDAMAPartitioner::_levelized() {
  // Kahn's algorithm, topological sort
  std::vector<int> indegree; 
  
  for (size_t fromGate = 0; fromGate < _sum_pi_gates_pos; fromGate++) {
    int numIndegree = _invAdj_index_table[2*fromGate+1] - _invAdj_index_table[2*fromGate+0]; 
    indegree.push_back(numIndegree);
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
    for (int i = _adj_index_table[2*first+0]; 
             i < _adj_index_table[2*first+1]; 
             i++) {
      int toGateIdx = _adj[i];
      indegree[toGateIdx]--;
      if (indegree[toGateIdx] == 0) {
        source.push(toGateIdx);
      }
    }
  }

  // ------- levelization -------
  _level_of_gates.resize(_sum_pi_gates_pos, 0);
  _max_level = 0;
  
  for (int i = 0; i < order.size(); i++) {
    int gateIdx = order[i];
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

  _print_levelized();
}

// void cudaMAPartitioner::_construct_partitioned_groups(const int simulator) {
//   // // construct group and show constructed group 
//   // _construct_groups();

  
//   // switch (simulator) {
//   //   case 0: { // CPU simulator
//   //     // Construct _cones_partitioned
//   //     for (size_t i = 0; i < _cones_set.size(); ++i) {
//   //       for (size_t lvl = 0; lvl < _cones_set[i].size(); ++lvl) {
//   //         for (const auto& element : _cones_set[i][lvl]) {
//   //           _cones_partitioned[i][lvl].push_back(element);
//   //         }
//   //       }
//   //     }  

//   //     // _count_duplications(simulator);
//   //     break;  
//   //   }
//   //   case 1: { // GPU simulator 
//   //     // Prepare for the part of levelization
//   //       // construct _per_level_of_group_start_accum_level_tmp
//   //       // construct _gateIdx_in_each_level_tmp
//   //     int accum = 0;
//   //     _per_level_of_group_start_accum_level_tmp.push_back(accum);

//   //     for (size_t level = 0; level < _gateIdx_in_each_level.size(); level++) {
//   //       for (size_t gate = 0; gate < _gateIdx_in_each_level[level].size(); gate++) {
//   //         int gateIdx = _gateIdx_in_each_level[level][gate];
//   //         _gateIdx_in_each_level_tmp.push_back(gateIdx);
//   //         accum++;
//   //       }
//   //       _per_level_of_group_start_accum_level_tmp.push_back(accum);
//   //     }
   

//   //     // Prepare for the part of partitioned groups
//   //       // construct _cones_partitioned_tmp
//   //       // construct _numGates_per_level_gpu_of_groups
//   //       // construct _per_level_of_group_start_accum_tmp
//   //     accum = 0;
//   //     _per_level_of_group_start_accum_tmp.push_back(accum);

//   //     for (size_t i = 0; i < _cones_set.size(); ++i) {
//   //       for (size_t lvl = 0; lvl < _cones_set[i].size(); ++lvl) {
//   //         int accum2 = 0;
//   //         for (const auto& element : _cones_set[i][lvl]) {
//   //           _cones_partitioned_tmp.push_back(element);
//   //           accum2++;
//   //           accum++;         
//   //         }
//   //         _numGates_per_level_gpu_of_groups.push_back(accum2);
//   //         _per_level_of_group_start_accum_tmp.push_back(accum);
//   //       }
//   //     }


//   //     //   printf("_numGates_per_level_gpu_of_groups = [");
//   //     //   for (size_t i = 0; i < _numGates_per_level_gpu_of_groups.size(); i++) {
//   //     //     printf("%d, ", _numGates_per_level_gpu_of_groups[i]);
//   //     //   } printf("]\n");

//   //     //   printf("_per_level_of_group_start_accum_tmp = [");
//   //     //   for (size_t i = 0; i < _per_level_of_group_start_accum_tmp.size(); i++) {
//   //     //     printf("%d, ", _per_level_of_group_start_accum_tmp[i]);
//   //     //   } printf("]\n");

//   //     //   printf("_cones_partitioned_tmp = [");
//   //     //   for (size_t i = 0; i < _cones_partitioned_tmp.size(); i++) {
//   //     //     printf("%d, ", _cones_partitioned_tmp[i]);
//   //     //   } printf("]\n");

//   //     //   std::vector<bool> vv; vv.resize(_sum_pi_gates_pos, false);
//   //     //   for (size_t i = 0; i < _cones_partitioned_tmp.size(); i++) {
//   //     //     vv[_cones_partitioned_tmp[i]] = true;
//   //     //   }
//   //     //   size_t acc = 0;
//   //     //   for (size_t i = 0; i < vv.size(); i++) {
//   //     //     if (!vv[i]) {
//   //     //       acc++;
//   //     //     }
//   //     //   } 
//   //     //   printf("haha acc = %lu (should == 0)\n", acc);
        


//   //     break;
//   //   }
//   //   default: {
//   //     printf("No SUCH CASES\n");
//   //     return; 
//   //   }
//   // }

// }

void CUDAMAPartitioner::prepare_gpu_simulation() {
  printf("hellow world 1!\n");
  auto start = std::chrono::steady_clock::now();
  const int simulator = 1;
  
  // _construct_partitioned_groups(simulator);
  // // Copy data from Host to Device for GPU simulation 
  
  // cudaMalloc((void**)&_adj_gpu, _szOfAdj*sizeof(int));
  // cudaMalloc((void**)&_invAdj_gpu, _szOfAdj*sizeof(int));
  // cudaMalloc((void**)&_adj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  // cudaMalloc((void**)&_invAdj_index_table_gpu, 2*_sum_pi_gates_pos*sizeof(int));
  // cudaMalloc((void**)&_visited_gpu, _sum_pi_gates_pos*sizeof(int));
  // cudaCheckErrors("CUDA: Asking partitioner memory - Failure");


  // // _ask_gpu_simulation_memory();

  // // _move_GateType_h2d();
  // // _move_patterns_h2d();
  // // _move_faults___h2d(); 
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> duration_prepare = end - start;
  std::cout << "prepare_GPU_simulation: " << _round_to((duration_prepare.count())*1000, 0.001) << "\n";
  // _count_duplications(simulator);
}

// Simulation functions 
void CUDAMAPartitioner::run(Mode mode, const size_t num_threads, const size_t NUM_SIMULATION_RDS) {  
  // // int m = static_cast<int>(mode);

  // switch (mode) {
  //   case Mode::GPU_PARTIOR_CPU_SIMUTOR: {
  //     // std::cout << "Mode: " << m << " (run GPU_PARTIOR_CPU_SIMUTOR)\n";
  //     GALPS_CPUSimulator<ELEMENT_INDEX_TYPE_PART> cpuSimulator;
  //     cpuSimulator.run_cpu_simulator_cones(num_threads, 
  //                                         _num_PIs, _num_inner_gates, _num_POs, 
  //                                         _sum_pi_gates_pos, 
  //                                         _num_pattern, _num_rounds, _num_fault,
  //                                         _patterns, _faults, _Gates,
  //                                         _cones_partitioned,
  //                                         _g_pi_results,
  //                                         _g_gate_results,
  //                                         _g_po_results,
  //                                         _b_pi_results,
  //                                         _b_gate_results,
  //                                         _b_po_results,
  //                                         _found_fault_to_pattern);
  //     break;  
  //   }

  //   case Mode::GPU_PARTIOR_GPU_SIMUTOR: {
  //     // std::cout << "Mode: " << m << " (run GPU_PARTIOR_GPU_SIMUTOR)\n";
  //     GALPS_GPUSimulator gpuSimulator;
  //     gpuSimulator.run_gpu_simulator_cones_part_gpu_cuda_graph(_k, _num_PIs, _num_inner_gates, _num_POs, 
  //                                                             _sum_pi_gates_pos, 
  //                                                             _num_pattern, _num_rounds, _num_fault,
  //                                                             _pi_gate_po_gate_type_gpu, 
  //                                                             _patterns_gpu,
  //                                                             _fault_gate_idx_gpu,
  //                                                             _fault_SA_fault_val_gpu,
  //                                                             _pi_gate_po_output_res_gpu,
  //                                                             _gateIdx_in_each_level,
  //                                                             _gateIdx_in_each_level_gpu,
  //                                                             _per_level_of_group_start_accum_level_gpu,
  //                                                             _threshold_0_level,
  //                                                             _numGates_per_level_gpu_of_groups_gpu,
  //                                                             _per_level_of_group_start_accum_gpu,
  //                                                             _cones_partitioned_gpu,
  //                                                             _total_num_levels,
  //                                                             _invAdj_gpu,
  //                                                             _invAdj_index_table_gpu,
  //                                                             _patterns, 
  //                                                             _g_pi_results_gpu,
  //                                                             _g_gate_results_gpu,
  //                                                             _g_po_results_gpu,
  //                                                             _b_pi_results_gpu,
  //                                                             _b_gate_results_gpu,
  //                                                             _b_po_results_gpu,
  //                                                             _found_fault_to_pattern_gpu,
  //                                                             NUM_SIMULATION_RDS);
  //     cudaDeviceSynchronize();
  //     break;
  //   }

  //   default: {
  //     printf("No SUCH CASES\n");
  //     return; 
  //   }
  // }
}

// --------------- PRINT FUNCTIONS FOR CHECK THE CORRECTNESS ---------------

void CUDAMAPartitioner::_print_patterns(const std::vector<Pattern> &patterns,
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
  printf("===============================\n");
}

void CUDAMAPartitioner::_print_levelized(){
  printf("_level_of_gates:\n");
  for (size_t i = 0; i < _level_of_gates.size(); i++) {
    printf("_level_of_gates[%lu] = %d\n", i, _level_of_gates[i]);
  }
  printf("\n");

  for (size_t i = 0; i < _gateIdx_in_each_level.size(); i++) {
    printf("level_%lu, size = %lu\n", i, _gateIdx_in_each_level[i].size());
  }  
}
