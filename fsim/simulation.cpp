#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <climits>

#include "simulation.hpp"

// -----


void fsim::Simulator::read(const std::string& ckt_path, const std::string& flst_path, const std::string& ptn_path) {
  // gate
  std::cout << "Read and construct ckt" << std::endl;
  std::ifstream ckt(ckt_path);
  ckt >> _num_PIs >> _num_POs >> _num_gate >> _num_wire;
  _total_gates = _num_PIs+_num_POs+_num_gate;
  std::cout << _num_PIs << ", " << _num_POs << ", " << _num_gate << ", " << _num_wire << ", " << _total_gates << std::endl;
  construct_graph(ckt);

  // fault
  std::cout << "Read and construct flst" << std::endl;
  std::ifstream flst(flst_path);
  flst >> _num_fault;
  std::cout << _num_fault << std::endl;
  construct_fault(flst);

  // pattern
  std::cout << "Read and construct ptn" << std::endl;
  std::ifstream ptn(ptn_path);
  ptn >> _num_pattern;
  std::cout << _num_pattern << std::endl;
  construct_pattern(ptn);

}

void fsim::Simulator::run() {
  std::cout << "execute simulation.run();" << std::endl;

  // Simulation
  // unsigned int team = (num_pattern+deal_bits-1)/deal_bits;
  // int *found_fault_table = (int *)malloc((2*num_fault)*sizeof(int)); memset(found_fault_table, 0, (2*num_fault)*sizeof(int));
    
  for (size_t i = 0; i < _round; i++) {
    // todo!!!! -> need to set the _testcase_bits
    size_t _testcase_bits = simple_testcase_bits;
    
    // good simulation
    std::cout << "Run Good case" << std::endl;
    good_case(_pattern[i], _testcase_bits);
    
    // Faults simulation 
    for (int j = 0; j < _num_fault; j++) {
        // // record whether the fault can be found
        // found_fault_table[2*j] = SA_faults(gates, faults[j], patterns[i], total_gates, num_POs, good_results, simple_testcase_bits);
        // // record which pattern found the fault
        // found_fault_table[2*j+1] = i;
    }
    // free(good_results);
  }  
}

void fsim::Simulator::good_case (Pattern pattern, size_t bits) {
  // Simulation
  for (size_t i = 0; i < _total_gates; i++) {
    run_gate(_gates[i], pattern, i, 0, 0);
  }
  
  // Duplicated the answer of good_case into a new memory
  good_result.resize(_total_gates);
  // size_t *good_results = (size_t *)malloc(_total_gates*sizeof(size_t));
  for (int i = 0; i < _total_gates; i++) {
      good_result[i] = _gates[i]._output_value;
  }

  if (bits < SIZE_T_BITS){
    shift_to_correct_answer(good_result, bits);
  }
  
#ifdef print_simulation
  std::cout << "GOOD resutls\n" << std::endl;
  print_check_simulation(good_result);
#endif
    
}


// Construct functions
void fsim::Simulator::construct_graph(std::ifstream &ckt) {
  std::vector<std::vector <size_t>> adj;
  adj.resize(_total_gates);
  std::vector<size_t> indegree;
  indegree.resize(_total_gates);  
  
  /** Levelized the input gates before constructing the circuit graph */
  // read file, constrcut adjacent list, construct indegree array
  for (size_t i = 0; i < _num_wire; i++) {
    size_t gate_0, pin_Y, num_post_gates;
    ckt >> gate_0 >> pin_Y >> num_post_gates;
    
    for (size_t j = 0; j < num_post_gates; j++) {
      size_t gate_tmp, pin_tmp;
      ckt >> gate_tmp >> pin_tmp;
      adj[gate_0].push_back(gate_tmp);
      indegree[gate_tmp]++;
    }
  }

  // copy `indegree` vector since we need its original value later
#ifdef print_graph
  std::vector<size_t> indegree_orig;
  indegree_orig = indegree;
#endif
  
  // Kahn's algorithm, topological sort
  std::queue <size_t> source;
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
  
#ifdef print_graph
  print_topological (adj, indegree_orig, order);
#endif

  /** Constructing circuit graph */
  // scanf gate type array
  std::vector<size_t> gate_type;
  gate_type.resize(_total_gates);
  // size_t *gate_type = (size_t*)malloc(_total_gates*sizeof(size_t));
  for (size_t i = 0; i < _num_PIs; i++) {
    gate_type[i] = PI_type_enum;
  }
  for (size_t i = _num_PIs; i < (_num_PIs+_num_POs); i++) {
    gate_type[i] = PO_type_enum;
  }
  for (size_t i = (_num_PIs+_num_POs); i < (_total_gates); i++) {
    ckt >> gate_type[i];
  }    
  
  // mapping gate_idx to gate_order_index
  /**
   * CML example:
   * order[3] = 6
   * _gate_id_to_order_idx[6] = 3
   * _gate_id_to_order_idx: 0, 1, 2, 7, 5, 6, 3, 4
   * */ 
  _gate_id_to_order_idx.resize(_total_gates);
  for (size_t i = 0; i < order.size(); i++) {
    _gate_id_to_order_idx[order[i]] = i;
  }

  // gates are stored in the order specified by the `order` array
  _gates.resize(_total_gates);
  for (size_t i = 0; i < _total_gates; i++) {
    // `gate_id` is the ID assigned in the input file
    size_t gate_id = order[i];
    size_t type = gate_type[gate_id];
    _gates[i]._type = static_cast<GateType>(type);
    _gates[i]._level = default_level;
    _gates[i]._output_value = default_output_value;
  }

  // give the input gates number 
  for (size_t i = 0; i < adj.size(); i++) {
    for (size_t j = 0; j < adj[i].size(); j++) {
      size_t from_gate_id = i, to_gate_id = adj[i][j];
      Gate& from_gate = _gates[_gate_id_to_order_idx[from_gate_id]];
      Gate& to_gate = _gates[_gate_id_to_order_idx[to_gate_id]];
      to_gate._inputs.push_back(&from_gate);
    }
  }

#ifdef print_graph
  for (size_t i = 0; i < _total_gates; i++) {
    print_gate(_gates[i]);
  }
#endif

} 

void fsim::Simulator::construct_fault(std::ifstream &flst){
  _faults.resize(_num_fault);
  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
    std::cout << wrong_gate << ", " << fault_value << std::endl;
    _faults[i]._gate_with_fault = _gate_id_to_order_idx[wrong_gate];
    _faults[i]._gate_SA_fault_val = (fault_value) ? (SIZE_MAX) : (0);
  }  

#ifdef print_faults 
  print_fault(_faults);
#endif

}

// #define deal_bits 32


void fsim::Simulator::construct_pattern(std::ifstream &ptn){
  _round = (_num_pattern+SIZE_T_BITS-1)/SIZE_T_BITS; // ceiling
  
  _pattern.resize(_round);
  for (size_t i = 0; i < _round; i++) {
    _pattern[i]._value.resize(_num_PIs);
    for (size_t pi = 0; pi < _num_PIs; pi++) {
      size_t idx = _gate_id_to_order_idx[i];
      ptn >> _pattern[i]._value[pi];
    }
  }

#ifdef print_patterns    
  print_pattern(_pattern, _round, _num_PIs);
#endif
  
}

// simulation for gates
void fsim::Simulator::INV (Gate& gate, size_t SA_fault, size_t fault_val) {
    if (SA_fault) {
      gate._output_value = fault_val;
    } else {
      Gate* pre_gate = gate._inputs[0];
      size_t ret = pre_gate->_output_value;
      gate._output_value = ~ret;
    }
}

void fsim::Simulator::AND (Gate& gate, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    Gate *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;

    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret &= (now_gate)->_output_value;
    }
    gate._output_value = ret;
  }
}

void fsim::Simulator::OR (Gate& gate, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    Gate *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;

    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret |= (now_gate)->_output_value;
    }
    gate._output_value = ret;
  }
}

void fsim::Simulator::XOR (Gate& gate, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    Gate *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;

    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret ^= (now_gate)->_output_value;
    }
    gate._output_value = ret;
  }
}

void fsim::Simulator::NAND (Gate& gate, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    Gate *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;

    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret &= (now_gate)->_output_value;
    }
    gate._output_value = ~(ret);
  }
}

void fsim::Simulator::NOR (Gate& gate, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    Gate *now_gate = gate._inputs[0];
    size_t ret = (now_gate)->_output_value;

    for (size_t i = 1; i < gate._inputs.size(); i++) {
      now_gate = gate._inputs[i];
      ret |= (now_gate)->_output_value;
    }
    gate._output_value = ~(ret);
  }
}

void fsim::Simulator::PI (Gate &gate, Pattern &pattern, size_t PI_i, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
  } else {
    gate._output_value = pattern._value[PI_i];
  }
}

void fsim::Simulator::PO (Gate &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  Gate *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

void fsim::Simulator::run_gate (Gate &gate, Pattern &pattern, size_t P_i, size_t SA_fault, size_t fault_val) {
  switch (gate._type) {
    case GateType::INV:
      INV(gate, SA_fault, fault_val);
      break;
    case GateType::AND:
      AND(gate, SA_fault, fault_val);
      break;
    case GateType::OR:
      OR(gate, SA_fault, fault_val);
      break;
    case GateType::XOR:
      XOR(gate, SA_fault, fault_val);
      break;
    case GateType::NAND:
      NAND(gate, SA_fault, fault_val);
      break;
    case GateType::NOR:
      NOR(gate, SA_fault, fault_val);
      break;
    case GateType::PI:
      PI(gate, pattern, P_i, SA_fault, fault_val);
      break;
    case GateType::PO:
      PO(gate);
      break;
  }
}

void fsim::Simulator::shift_to_correct_answer (std::vector<size_t> &results, size_t bits){
  for (size_t i = 0; i < _total_gates; i++) {
    results[i]<<=(SIZE_T_BITS-bits);
    results[i]>>=(SIZE_T_BITS-bits);
    // printf("!results[%ld] = %ld\n", i, results[i]);
  }
}


// Print functions
void fsim::Simulator::print_topological (std::vector<std::vector <size_t>> &adj, std::vector<size_t> &indegree, std::vector<size_t> &order) {
  printf("\n=====\n\n");
  printf("adj:\n");
  for (size_t i = 0; i < adj.size(); i++) {
    printf("%ld = [", i);
    for (size_t v_len = 0; v_len < adj[i].size(); v_len++) {
      printf("%ld, ", adj[i][v_len]);
    } printf("]\n");
  } printf("\n");
  
  printf("indegree:\n");
  for (size_t i = 0; i < indegree.size(); i++) {
    printf("%ld = [%ld]\n", i, indegree[i]);
  }

  printf("Order:\n");
  for (size_t i = 0; i < order.size(); i++) {
    printf("%ld, ", order[i]);
  } printf("\n");
}

void fsim::Simulator::print_gate(Gate& gate) {
  printf("\n=====\n\n");
  printf("id: %p\n", &gate);
  printf("type: %ld\n", static_cast<size_t>(gate._type));
  printf("level: %ld\n", gate._level);
  printf("input_gates: [");
  for (size_t i = 0; i < gate._inputs.size(); i++) {
    printf("%p, ", gate._inputs[i]);
  }
  printf("]\n");
  printf("output_value: %ld\n", gate._output_value);
}

void fsim::Simulator::print_fault (std::vector<Fault> &_faults) {
  std::cout << "\n=====\n\n" << "print_faults" << std::endl;
  for (size_t i = 0; i < _num_fault; i++) {
    printf("_gate_with_fault = %ld, SA_%ld\n", _faults[i]._gate_with_fault, _faults[i]._gate_SA_fault_val);
  }
  std::cout << "\n=====\n\n" << std::endl;
}

void fsim::Simulator::print_pattern (std::vector<Pattern> &patterns, size_t round, size_t num_PIs) {
  printf("\n=====\n\n");
  for (unsigned int i = 0; i < round; i++) {
    printf("[%ld, %ld] bits = [\n", SIZE_T_BITS*i, SIZE_T_BITS*(i+1));
    for (int j = 0; j < num_PIs; j++) {
      printBits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
    }
    printf("]\n");
  }    
}

void fsim::Simulator::printBits_stack(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

void fsim::Simulator::print_check_simulation (std::vector<size_t> &results) {
  for (size_t i = 0; i < _total_gates; i++) {
    printf("gate_%ld.output = %ld\n", i, results[i]);
  }
}



