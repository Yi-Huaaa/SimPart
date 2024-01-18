#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <climits>

#include "simulation.hpp"


void fsim::Simulator::read(const std::string& ckt_path, const std::string& flst_path, const std::string& ptn_path) {
  // gate
  std::cout << "Read and construct ckt" << std::endl;
  std::ifstream ckt(ckt_path);
  ckt >> _num_PIs >> _num_POs >> _num_gate >> _num_wire;
  _total_gates = _num_PIs + _num_POs + _num_gate;
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
  _found_fault_table.resize(2*_num_fault);

  for (size_t i = 0; i < _round; i++) {
    // todo!!!! -> need to set the _testcase_bits
    size_t _testcase_bits = SIMPLE_TESTCASE_BITS;
    
    // good simulation
    std::cout << "Run Good case" << std::endl;
    good_case(_pattern[i], _testcase_bits);
    
    // Faults simulation 
    for (size_t j = 0; j < _num_fault; j++) {
      bad_case(_faults[j], _pattern[i], _testcase_bits);

      size_t found_fault = 0;
      for (size_t i = (_total_gates-1); i > (_total_gates-_num_POs-1); i--) {
        if (_good_result[i] != _bad_result[i]) {
          found_fault = 1;
        }
      }
      // Record whether fault can be found
      _found_fault_table[2*j] = found_fault;
      // Record which pattern found the fault
      _found_fault_table[2*j+1] = i;
    }
  }

#ifdef print_fault_table
  print_found_fault_table (_found_fault_table);
#endif 
}

void fsim::Simulator::good_case (Pattern &pattern, size_t bits) {
  // Simulation
  for (size_t i = 0; i < _total_gates; i++) {
    run_gate(_gates[i], pattern, i, 0, 0);
  }
  
  // Duplicated the answer of good_case into a new memory
  _good_result.resize(_total_gates);
  for (size_t i = 0; i < _total_gates; i++) {
    _good_result[i] = _gates[i]._output_value;
  }

  if (bits < SIZE_T_BITS) {
    shift_to_correct_answer(_good_result, bits);
  }
  
#ifdef print_simulation
  std::cout << "GOOD resutls:" << std::endl;
  print_check_simulation(_good_result);
#endif
  
}


void fsim::Simulator::bad_case (Fault &fault, Pattern &pattern, size_t bits) {
  size_t wrong_gate = fault._gate_with_fault;
  size_t fault_val = fault._gate_SA_fault_val;
  
  // fault simulation
  for (size_t i = 0; i < _total_gates; i++) {
    size_t SA_fault = (i == wrong_gate);
    run_gate(_gates[i], pattern, i, SA_fault, fault_val);
  }

  // Duplicated the answer of BAD_case into a new memory
  _bad_result.resize(_total_gates);
  for (size_t i = 0; i < _total_gates; i++) {
    _bad_result[i] = _gates[i]._output_value;
  }
  
  if (bits < SIZE_T_BITS) {
    shift_to_correct_answer(_bad_result, bits);
  }

#ifdef print_simulation
  std::cout << "BAD resutls:" << std::endl;
  print_check_simulation(_bad_result);
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


#ifdef print_graph
  // copy `indegree` vector since we need its original value later
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
  std::vector<GateType> gate_type;
  gate_type.resize(_total_gates);
  for (size_t i = 0; i < _num_PIs; i++) {
    gate_type[i] = GateType::PI;
  }
  for (size_t i = _num_PIs; i < (_num_PIs+_num_POs); i++) {
    gate_type[i] = GateType::PO;
  }
  for (size_t i = (_num_PIs+_num_POs); i < (_total_gates); i++) {
    size_t t; // type
    ckt >> t;
    gate_type[i] = static_cast<GateType>(t);
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
    _gates[i]._type = gate_type[gate_id];
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

void fsim::Simulator::construct_fault(std::ifstream &flst) {
  _faults.resize(_num_fault);
  for (size_t i = 0; i < _num_fault; i++) {
    size_t wrong_gate, fault_value;
    flst >> wrong_gate >> fault_value;
#ifdef print_faults 
    std::cout << wrong_gate << ", " << fault_value << std::endl;
#endif
    _faults[i]._gate_with_fault = _gate_id_to_order_idx[wrong_gate];
    _faults[i]._gate_SA_fault_val = (fault_value) ? (SIZE_MAX) : (0);
  }

#ifdef print_faults 
  print_fault(_faults);
#endif

}


void fsim::Simulator::construct_pattern(std::ifstream &ptn) {
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
void fsim::Simulator::INV (Gate& gate) {
  Gate* pre_gate = gate._inputs[0];
  size_t ret = pre_gate->_output_value;
  gate._output_value = ~ret;
}

void fsim::Simulator::AND (Gate& gate) {
  Gate *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::OR (Gate& gate) {
  Gate *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::XOR (Gate& gate) {
  Gate *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret ^= (now_gate)->_output_value;
  }
  gate._output_value = ret;
}

void fsim::Simulator::NAND (Gate& gate) {
  Gate *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret &= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void fsim::Simulator::NOR (Gate& gate) {
  Gate *now_gate = gate._inputs[0];
  size_t ret = (now_gate)->_output_value;

  for (size_t i = 1; i < gate._inputs.size(); i++) {
    now_gate = gate._inputs[i];
    ret |= (now_gate)->_output_value;
  }
  gate._output_value = ~(ret);
}

void fsim::Simulator::PI (Gate &gate, Pattern &pattern, size_t pi) {
  gate._output_value = pattern._value[pi];
}

void fsim::Simulator::PO (Gate &gate) {
  // It outputs its previous gate's output_value if there is no SA fault
  Gate *pre_gate = gate._inputs[0];
  gate._output_value = pre_gate->_output_value;
}

void fsim::Simulator::run_gate (Gate &gate, Pattern &pattern, size_t pi, size_t SA_fault, size_t fault_val) {
  if (SA_fault) {
    gate._output_value = fault_val;
    return;
  }
  switch (gate._type) {
    case GateType::INV:
      INV(gate);
      break;
    case GateType::AND:
      AND(gate);
      break;
    case GateType::OR:
      OR(gate);
      break;
    case GateType::XOR:
      XOR(gate);
      break;
    case GateType::NAND:
      NAND(gate);
      break;
    case GateType::NOR:
      NOR(gate);
      break;
    case GateType::PI:
      PI(gate, pattern, pi);
      break;
    case GateType::PO:
      PO(gate);
      break;
  }
}

void fsim::Simulator::shift_to_correct_answer (std::vector<size_t> &results, size_t bits) {
  for (size_t i = 0; i < _total_gates; i++) {
    results[i]<<=(SIZE_T_BITS-bits);
    results[i]>>=(SIZE_T_BITS-bits);
  }
}


// print functions
void fsim::Simulator::print_topological (const std::vector<std::vector <size_t>> &adj, const std::vector<size_t> &indegree, const std::vector<size_t> &order) {
  std::cout << "\n=====\n\nadj:" << std::endl;
  for (size_t i = 0; i < adj.size(); i++) {
    std::cout << i << " = [";
    for (size_t v_len = 0; v_len < adj[i].size(); v_len++) {
      std::cout << adj[i][v_len] << ", ";
    } std::cout << "]\n";
  } std::cout << "\n";
  
  std::cout << "Indegree:\n";
  for (size_t i = 0; i < indegree.size(); i++) {
    std::cout << i << " = [" << indegree[i] << "]" << std::endl;
  }

  std::cout << "Order:\n";
  for (size_t i = 0; i < order.size(); i++) {
    std::cout << order[i] << ", ";
  } std::cout << std::endl;
}

void fsim::Simulator::print_gate(const Gate& gate) {
  std::cout << "\n=====\n\n";
  std::cout << "id: " << &gate << std::endl;
  std::cout << "type: " << static_cast<size_t>(gate._type) << "\n";
  std::cout << "level: " << gate._level << "\n";
  std::cout << "input_gates: [";
  for (size_t i = 0; i < gate._inputs.size(); i++) {
    std::cout << gate._inputs[i] << ", ";
  }
  std::cout << "]\n";
  std::cout << "output_value: " << gate._output_value << "\n";
}

void fsim::Simulator::print_fault (const std::vector<Fault> &_faults) {
  std::cout << "\n=====\n\n" << "print_faults" << std::endl;
  for (size_t i = 0; i < _num_fault; i++) {
    std::cout << "_gate_with_fault = " << _faults[i]._gate_with_fault << ", SA_" << _faults[i]._gate_SA_fault_val << "\n";
  }
  std::cout << "\n=====\n\n" << std::endl;
}

void fsim::Simulator::print_pattern (const std::vector<Pattern> &patterns, const size_t round, const size_t num_PIs) {
  std::cout << "\n=====\n\n" << std::endl;
  for (unsigned int i = 0; i < round; i++) {
    std::cout << "[" << SIZE_T_BITS*i << ", " << SIZE_T_BITS*(i+1) << "] bits = [\n";
    for (int j = 0; j < num_PIs; j++) {
      print_bits_stack(sizeof(patterns[i]._value[j]), &patterns[i]._value[j]);
    }
    std::cout << "]\n";
  }    
}


/**
 * `print_bits_stack` from stackoverflow: 
 * https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
*/

void fsim::Simulator::print_bits_stack(const size_t size, const void* const ptr) {
  unsigned char *b = (unsigned char*) ptr;
  unsigned char byte;
  int i, j;
  
  for (i = size-1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      std::cout << byte;
    }
  }
  std::cout << "\n";
}

void fsim::Simulator::print_check_simulation (const std::vector<size_t> &results) {
  for (size_t i = 0; i < _total_gates; i++) {
    std::cout << "gate_" << i << ".output = " << results[i] << "\n";
  } 
  std::cout << "\n" << std::endl;
}

void fsim::Simulator::print_found_fault_table (const std::vector<size_t> &_found_fault_table) {
  std::cout << "\n=====\n\n" << std::endl;
  for (size_t i = 0; i < (2*_num_fault); i+=2) {
    if (_found_fault_table[i] == 1) {
      std::cout << "fault_" << (i/2) << " is found, with Pattern_" << _found_fault_table[i+1] << "\n";
    } else {
      std::cout << "fault_" << i << " cannot be found\n";
    }
  }
}


