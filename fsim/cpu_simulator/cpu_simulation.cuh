#ifndef GALPS_CPU_SIMULATION_H
#define GALPS_CPU_SIMULATION_H

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


#include <fsim/fsim.hpp>


// #define CPU_PART_DEBUG_PRINT_SIMULATION  // print answer for check 


// Forward declaration of CUDAPartitioner
class CUDAPartitioner;


template<typename IndexType = size_t>
class GALPS_CPUSimulator {

  friend class CUDAPartitioner;

public:

  void run_cpu_simulator_cones(const size_t num_threads,
            const size_t num_PIs, 
            const size_t num_inner_gates,
            const size_t num_POs, 
            const size_t sum_pi_gates_pos,
            const size_t num_pattern, 
            const size_t num_rounds,
            const size_t num_fault, 
            const std::vector<Pattern> &_patterns,
            const std::vector<Fault<IndexType>> &_faults,
            std::vector<ElementBase<IndexType, IndexType>> &_Gates,
            const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned,
            std::vector<size_t> &_g_pi_results,
            std::vector<size_t> &_g_gate_results,
            std::vector<size_t> &_g_po_results,
            std::vector<size_t> &_b_pi_results,
            std::vector<size_t> &_b_gate_results,
            std::vector<size_t> &_b_po_results,
            std::vector<size_t> &_found_fault_to_pattern) {
    // Get vars          
    _num_threads = num_threads;
    _num_PIs = num_PIs;
    _num_inner_gates = num_inner_gates;
    _num_POs = num_POs;
    _sum_pi_gates_pos = sum_pi_gates_pos;
    _num_pattern = num_pattern;
    _num_rounds = num_rounds;
    _num_fault = num_fault;

    // run simulation
    _run_cones_gates_cpu(_Gates, 
                        _g_pi_results,
                        _g_gate_results,
                        _g_po_results,
                        _b_pi_results,
                        _b_gate_results,
                        _b_po_results,
                        _found_fault_to_pattern,
                        _cones_partitioned, 
                        _patterns, 
                        _faults);
  }

private:
  // vars
  IndexType _num_PIs; 
  IndexType _num_inner_gates;
  IndexType _num_POs; 
  IndexType _sum_pi_gates_pos;
  IndexType _num_pattern; 
  size_t _num_rounds; 
  IndexType _num_fault; 
  size_t _num_threads;


  void _run_cones_gates_cpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                            std::vector<size_t> &_g_pi_results,
                            std::vector<size_t> &_g_gate_results,
                            std::vector<size_t> &_g_po_results,
                            std::vector<size_t> &_b_pi_results,
                            std::vector<size_t> &_b_gate_results,
                            std::vector<size_t> &_b_po_results,
                            std::vector<size_t> &_found_fault_to_pattern,
                            const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned,
                            const std::vector<Pattern> &_patterns,
                            const std::vector<Fault<IndexType>> &_faults);

  void _run_cones_good_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                            const Pattern pattern, 
                            const size_t bits,
                            std::vector<size_t> &_g_pi_results,
                            std::vector<size_t> &_g_gate_results,
                            std::vector<size_t> &_g_po_results,
                            const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned);
  void _run_cones_bad_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                          const Fault<IndexType> &fault, 
                          const Pattern pattern,
                          const size_t bits,
                          std::vector<size_t> &_b_pi_results,
                          std::vector<size_t> &_b_gate_results,
                          std::vector<size_t> &_b_po_results,
                          const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned);
  void _write_to_array(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                      std::vector<size_t> &pi_output, 
                      std::vector<size_t> &gate_output,
                      std::vector<size_t> &po_output,
                      const size_t bits);

  void _apply_INV(ElementBase<IndexType, IndexType> &gate);
  void _apply_AND(ElementBase<IndexType, IndexType> &gate);
  void _apply_OR(ElementBase<IndexType, IndexType> &gate);
  void _apply_XOR(ElementBase<IndexType, IndexType> &gate);
  void _apply_NAND(ElementBase<IndexType, IndexType> &gate);
  void _apply_NOR(ElementBase<IndexType, IndexType> &gate);
  void _apply_XNOR(ElementBase<IndexType, IndexType> &gate);
  void _apply_MUX(ElementBase<IndexType, IndexType> &gate);
  void _apply_CLKBUF(ElementBase<IndexType, IndexType> &gate);
  void _apply_PI(ElementBase<IndexType, IndexType> &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(ElementBase<IndexType, IndexType> &gate);
  void _run_gate(ElementBase<IndexType, IndexType> &gate, 
                  const Pattern pattern, 
                  const IndexType pi, 
                  const IndexType SA_fault, 
                  const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results,
                                const size_t bits,
                                const IndexType num_shift_gates);                                                 

  // print functions
  std::string gateTypeToString(GateType type) const;
  void print_simulation_results(const std::vector<size_t> &pi_results,
                                const std::vector<size_t> &gate_results,
                                const std::vector<size_t> &po_results) const;
};



// Functions 

template<typename IndexType>
void GALPS_CPUSimulator<IndexType>::_run_cones_gates_cpu(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                                      std::vector<size_t> &_g_pi_results,
                                                      std::vector<size_t> &_g_gate_results,
                                                      std::vector<size_t> &_g_po_results,
                                                      std::vector<size_t> &_b_pi_results,
                                                      std::vector<size_t> &_b_gate_results,
                                                      std::vector<size_t> &_b_po_results,
                                                      std::vector<size_t> &_found_fault_to_pattern,
                                                      const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned,
                                                      const std::vector<Pattern> &_patterns,
                                                      const std::vector<Fault<IndexType>> &_faults) {
  #ifdef CPU_PART_DEBUG_PRINT_SIMULATION
    std::cout << "execute simulation._run_cones_gates_cpu();\n";
  #endif
  
    // Simulation
    for (size_t rd = 0; rd < _num_rounds; rd++) {
      size_t num_testcases_this_round =
          ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
              ? (SIZE_T_BITS)
              : (_num_pattern % SIZE_T_BITS);
  #ifdef CPU_PART_DEBUG_PRINT_SIMULATION
      std::cout << "Run Good case with threads = " << _num_threads << std::endl;
  #endif
      _run_cones_good_case(_Gates, _patterns[rd], num_testcases_this_round, 
                          _g_pi_results, _g_gate_results, _g_po_results, _cones_partitioned);
          
      // // bad simulation (fault simulation)
      // for (size_t j = 0; j < _faults.size(); j++) {
      //   _run_cones_bad_case(_Gates, _faults[j], _patterns[rd], 
      //                       num_testcases_this_round,
      // _b_pi_results, _b_gate_results, _b_po_results, _cones_partitioned);
  
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
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_run_cones_good_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                                          const Pattern pattern, 
                                                          const size_t bits,
                                                          std::vector<size_t> &_g_pi_results,
                                                          std::vector<size_t> &_g_gate_results,
                                                          std::vector<size_t> &_g_po_results,
                                                          const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned) {
  
    omp_set_num_threads(_num_threads);  // Set the number of threads for OpenMP
    // Parallelize across cones (c)
    // #pragma omp parallel for
    // #pragma omp parallel for if (_cones_partitioned.size() > 1) schedule(dynamic)
    for (size_t c = 0; c < _cones_partitioned.size(); c++) {
      for (size_t lvl = 0; lvl < _cones_partitioned[c].size(); lvl++) {
        // Parallelize across gates within the same cone and level
        #pragma omp parallel for
        // #pragma omp parallel for schedule(dynamic)
        // #pragma omp parallel for if (_cones_partitioned[c][lvl].size()) schedule(dynamic)
        for (size_t g = 0; g < _cones_partitioned[c][lvl].size(); g++) {
          int gateIdx = _cones_partitioned[c][lvl][g];
          
          // Select element (PI, PO, or gate)
          ElementBase<IndexType, IndexType> &tmp = _Gates[gateIdx];
  
          // Execute the gate operation
          _run_gate(tmp, pattern, gateIdx, 0, 0);
        }
      }
    }
    
    // Duplicated the answer of good_case into a new memory
    _write_to_array(_Gates, _g_pi_results, _g_gate_results, _g_po_results, bits);
  
  #ifdef CPU_PART_DEBUG_PRINT_SIMULATION
    std::cout << "CONES: GOOD resutls ans:\n";
    print_simulation_results(_g_pi_results, _g_gate_results, _g_po_results);
  #endif
  }
  
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_run_cones_bad_case(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                            const Fault<IndexType> &fault, 
                                            const Pattern pattern,
                                            const size_t bits,
                                            std::vector<size_t> &_b_pi_results,
                                            std::vector<size_t> &_b_gate_results,
                                            std::vector<size_t> &_b_po_results,
                                            const std::vector<std::vector<std::vector<IndexType>>> &_cones_partitioned) {
    // printf("Getting inside _run_cones_bad_case\n");
    omp_set_num_threads(_num_threads);  // Set the number of threads for OpenMP
  
    IndexType wrong_gate = fault._gate_with_fault;
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
          IndexType SA_fault = (gateIdx < _num_PIs) ? ((gateIdx) == wrong_gate) :
                            (gateIdx >= (_num_PIs + _num_POs)) ? ((gateIdx+_num_PIs+_num_POs) == wrong_gate) :
                            ((gateIdx + _num_PIs) == wrong_gate);
  
          // Select element (PI, PO, or gate)
          ElementBase<IndexType, IndexType> &tmp = _Gates[gateIdx];
  
          // Execute the gate operation
          _run_gate(tmp, pattern, gateIdx, SA_fault, fault_val);
        }
      }
    }
  
    // Duplicated the answer of good_case into a new memory
    _write_to_array(_Gates, _b_pi_results, _b_gate_results, _b_po_results, bits);
  
  #ifdef CPU_PART_DEBUG_PRINT_SIMULATION
    std::cout << "CONES: BAD resutls ans:\n";
    print_simulation_results(_b_pi_results, _b_gate_results, _b_po_results);
  #endif
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_write_to_array(std::vector<ElementBase<IndexType, IndexType>> &_Gates,
                                      std::vector<size_t> &pi_output, 
                                      std::vector<size_t> &gate_output,
                                      std::vector<size_t> &po_output,
                                      const size_t bits) {
    for (int i = 0; i < _sum_pi_gates_pos; i++) {
      if (i < _num_PIs) { // PI
        pi_output[i] = _Gates[i]._output_value;
      } else if (i >= _num_PIs && i < (_num_PIs+_num_POs)) { // PO
        po_output[i-_num_PIs] = _Gates[i]._output_value;
      } else { // gate
        gate_output[i-(_num_PIs+_num_POs)] = _Gates[i]._output_value;
      }
    }
  
    if (bits < SIZE_T_BITS) {
      _shift_to_correct_answer(pi_output, bits, _num_PIs);
      _shift_to_correct_answer(gate_output, bits, _num_inner_gates);
      _shift_to_correct_answer(po_output, bits, _num_POs);
    }  
  }
  
  // Simulation for gates
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_INV(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *pre_gate = gate._inputs[0];
    size_t ret = pre_gate->_output_value;
    gate._output_value = ~ret;
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_AND(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret &= (now_gate)->_output_value;
    }
    gate._output_value = ret;
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_OR(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
    
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret |= (now_gate)->_output_value;
    }

    gate._output_value = ret;


    for (size_t i = 0; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
    }    
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_XOR(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret ^= (now_gate)->_output_value;
    }
    gate._output_value = ret;
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_NAND(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret &= (now_gate)->_output_value;
    }
    gate._output_value = ~(ret);
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_NOR(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret |= (now_gate)->_output_value;
    }
    gate._output_value = ~(ret);
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_XNOR(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret ^= (now_gate)->_output_value;
    }
    gate._output_value = ~(ret);
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_MUX(ElementBase<IndexType, IndexType> &gate) {
    size_t a = gate._inputs[0]->_output_value;
    size_t b = gate._inputs[1]->_output_value;
    size_t s = gate._inputs[2]->_output_value;
  
    size_t ret = ((s & b) | (a & (!s)));
  
    gate._output_value = (ret);
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_CLKBUF(ElementBase<IndexType, IndexType> &gate) {
    ElementBase<IndexType, IndexType> *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;
  
    gate._output_value = ret;
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_PI(ElementBase<IndexType, IndexType> &gate, 
                                                const Pattern pattern,
                                                const size_t pi) {
    gate._output_value = pattern._value[pi];
  }
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_apply_PO(ElementBase<IndexType, IndexType> &gate) {
    // It outputs its previous gate's output_value if there is no SA fault
    ElementBase<IndexType, IndexType> *pre_gate = gate._inputs[0];
    gate._output_value = pre_gate->_output_value;
  }
  
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_run_gate(ElementBase<IndexType, IndexType> &gate, 
                                                const Pattern pattern,
                                                const IndexType pi, 
                                                const IndexType SA_fault, 
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
  
  
  template<typename IndexType>
  void GALPS_CPUSimulator<IndexType>::_shift_to_correct_answer(std::vector<size_t> &results,
                                                 const size_t bits,
                                                 const IndexType num_shift_gates) {
    for (IndexType i = 0; i < num_shift_gates; i++) {
      results[i] <<= (SIZE_T_BITS - bits);
      results[i] >>= (SIZE_T_BITS - bits);
    }
  }


// print functions ------------------
template<typename IndexType>
void GALPS_CPUSimulator<IndexType>::print_simulation_results(
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

// Function to convert GateType to string
template<typename IndexType>
std::string GALPS_CPUSimulator<IndexType>::gateTypeToString(GateType type) const {
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


#endif  // GALPS_CPU_SIMULATION_H
