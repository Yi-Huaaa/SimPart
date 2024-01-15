#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>

#include "simulation.hpp"



// void INV (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *pre_gate = gate->input_gates[0];
//         unsigned int b = pre_gate->output_value;
//         gate->output_value = ~b;
//     }
// }


// void AND (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *now_gate = gate->input_gates[0];
//         unsigned int ret = (now_gate)->output_value;

//         for (int i = 1; i < gate->num_of_inputs; i++) {
//             now_gate = gate->input_gates[i];
//             ret &= (now_gate)->output_value;
//         }
//         gate->output_value = ret;
//     }
// }


// void OR (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *now_gate = gate->input_gates[0];
//         unsigned int ret = (now_gate)->output_value;

//         for (int i = 1; i < gate->num_of_inputs; i++) {
//             now_gate = gate->input_gates[i];
//             ret |= (now_gate)->output_value;
//         }
//         gate->output_value = ret;        
//     }

// }


// void XOR (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *now_gate = gate->input_gates[0];
//         unsigned int ret = (now_gate)->output_value;

//         for (int i = 1; i < gate->num_of_inputs; i++) {
//             now_gate = gate->input_gates[i];
//             ret ^= (now_gate)->output_value;
//         }
//         gate->output_value = ret;        
//     }

// }


// void NAND (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *now_gate = gate->input_gates[0];
//         unsigned int ret = (now_gate)->output_value;

//         for (int i = 1; i < gate->num_of_inputs; i++) {
//             now_gate = gate->input_gates[i];
//             ret &= (now_gate)->output_value;
//         }
//         gate->output_value = ~(ret);
//     }
// }


// void NOR (Gate *gate, int SA_fault, int fault_val) {
//     if (SA_fault) {
//         gate->output_value = fault_val;
//     } else {
//         Gate *now_gate = gate->input_gates[0];
//         unsigned int ret = (now_gate)->output_value;

//         for (int i = 1; i < gate->num_of_inputs; i++) {
//             now_gate = gate->input_gates[i];
//             ret |= (now_gate)->output_value;
//         }
//         gate->output_value = ~(ret);
//     }
// }


// void PI (Gate *gate, int PI_i, Pattern *pattern, int SA_fault, int fault_val) {
//   if (SA_fault) {
//     gate->output_value = fault_val;
//     // printf("debug2: inside here, fault_val = %d\n", fault_val);
//   } else {
//     gate->output_value = pattern->value[PI_i];    
//   }
// }

// void PO (Gate *gate) {
//   // It outputs its previous gate's output_value if there is no SA fault
//   Gate *pre_gate = gate->input_gates[0];
//   gate->output_value = pre_gate->output_value;
// }


// void run_gate (Gate *gate, Pattern *pattern, int P_i, int SA_fault, int fault_val) {
//   switch (gate->type) {
//     case 0:
//       INV(gate, SA_fault, fault_val);
//       break;
//     case 1:
//       AND(gate, SA_fault, fault_val);
//       break;
//     case 2:
//       OR(gate, SA_fault, fault_val);
//       break;
//     case 3:
//       XOR(gate, SA_fault, fault_val);
//       break;
//     case 4:
//       NAND(gate, SA_fault, fault_val);
//       break;
//     case 5:
//       NOR(gate, SA_fault, fault_val);
//       break;
//     case PI_type_enum:
//       PI(gate, P_i, pattern, SA_fault, fault_val);
//       break;
//     case PO_type_enum:
//       PO(gate);
//       break;
//   }
// }

// void shift_to_correct_answer (unsigned int *results, int total_gates, int bits){
//   for (int i = 0; i < total_gates; i++) {
//     results[i]<<=(deal_bits-bits);
//     results[i]>>=(deal_bits-bits);
//   }
// }


// unsigned int *good_case (Gate **gates, Pattern* pattern, int total_gates, int bits) {
//   // Simulation
//   for (int i = 0; i < total_gates; i++) {
//       run_gate(gates[i], pattern, i, 0, 0);
//   }
  
//   // Duplicated the answer of good_case into a new memory
//   unsigned int *good_results = (unsigned int *)malloc(total_gates*sizeof(unsigned int));
//   for (int i = 0; i < total_gates; i++) {
//       good_results[i] = gates[i]->output_value;
//   }

//   if (bits < deal_bits){
//       shift_to_correct_answer(good_results, total_gates, bits);
//   }
  
// #ifdef print_simulation
//   printf("GOOD resutls\n");
//   print_check_simulation(good_results, total_gates);
// #endif
    
//   return good_results;
// }


// int SA_faults (Gate **gates, Fault faults, Pattern* pattern, int total_gates, int num_POs, unsigned int *good_results, int bits) {
    
//   int wrong_gate = faults.w_gate;
//   int fault_val = faults.SA_fault_value;
//   // Simulation
//   for (int i = 0; i < total_gates; i++) {
//       int SA_fault = (i == wrong_gate);
//       // printf("debug1:i = %d, SA_fault = %d\n", i, SA_fault);
//       run_gate(gates[i], pattern, i, SA_fault, fault_val);
//   }
//   // Duplicated the answer of BAD_case into a new memory
//   unsigned int *bad_results = (unsigned int *)malloc(total_gates*sizeof(unsigned int));
//   for (int i = 0; i < total_gates; i++) {
//       bad_results[i] = gates[i]->output_value;
//   }
  
//   if (bits < deal_bits){
//       shift_to_correct_answer(bad_results, total_gates, bits);
//   }

// #ifdef print_simulation
//   printf("BAD resutls\n");
//   print_check_simulation(bad_results, total_gates);
// #endif

//   // Record whether fault can be found
//   int found_fault = 0;
//   for (int i = (total_gates-1); i > (total_gates-num_POs-1); i--) {
//       if (good_results[i] != bad_results[i]) {
//           found_fault = 1;
//       }
//   }
//   // printf("found_fault = %d\n", found_fault);

//   free(bad_results);
//   return found_fault;
// }


void print_check_simulation(unsigned int *results, int total_gates) {
  for (int i = 0; i < total_gates; i++) {
      printf("gate_%d.output = %u\n", i, results[i]);
  }
}

// -----


void fsim::Simulator::read(const std::string& ckt_path, const std::string& flst, const std::string& ptn) {
  // std::cout << "execute simulation.read()" << std::endl;
  std::ifstream ckt(ckt_path);

  ckt >> _num_PIs >> _num_POs >> _num_gate >> _num_wire;
  _total_gates = _num_PIs+_num_POs+_num_gate;
  std::cout << _num_PIs << ", " << _num_POs << ", " << _num_gate << ", " << _num_wire << ", " << _total_gates << std::endl;
  construct_graph(ckt);

}

void fsim::Simulator::run() {
  // std::cout << "execute simulation.run()" << std::endl;
}


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