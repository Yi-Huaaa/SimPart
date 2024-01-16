#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <iostream>

#include <fsim/fsim.hpp>

#define simple_testcase_bits 3


int main(int argc, char* argv[]) {
    
    // TODO: from user's perspective
    fsim::Simulator simulator;

    std::string ckt_path(argv[1]);
    std::string flst_path(argv[2]);
    std::string ptn_path(argv[3]);

    simulator.read(ckt_path, flst_path, ptn_path);
    // simulator.run();

    // // user to access gates in the simulator
    // for(auto& gate : simulator.gates()) {
    //     // std::cout << gate.output_value() << '\n';
    // }


//     //
    
//     /* Readin data - circit */ 
//     FILE *input_file_circuit = fopen(argv[1], "r");
//     assert(input_file_circuit != NULL);
//     int num_PIs, num_POs, num_gate, num_wire;
//     assert(fscanf(input_file_circuit, "%d%d%d%d", &num_PIs, &num_POs, &num_gate, &num_wire) == 4);
//     int total_gates = num_PIs+num_POs+num_gate;

//     /* Constrcut graph (gate) */
//     int *gate_id_to_order_idx = (int*)malloc(total_gates*sizeof(int));
//     Gate **gates = construct_graph(input_file_circuit, total_gates, num_wire, num_PIs, num_POs, gate_id_to_order_idx);
//     fclose(input_file_circuit);



//     /* Readin data - faults*/ 
//     FILE *input_file_faults = fopen(argv[2], "r");
//     assert(input_file_faults != NULL);
//     int num_fault;
//     assert(fscanf(input_file_faults, "%d", &num_fault) == 1);

//     /* Constrcut faults */
//     Fault *faults = construct_fault(input_file_faults, num_fault, gate_id_to_order_idx);
//     fclose(input_file_faults);    



//     /* Readin data - patterns */ 
//     FILE *input_file_patterns = fopen(argv[3], "r");
//     assert(input_file_patterns != NULL);
//     // todo: delete the num_of_PIs here
//     int num_of_PIs, num_pattern;
//     assert(fscanf(input_file_patterns, "%d%d", &num_of_PIs, &num_pattern) == 2);

//     /* Constrcut faults */
//     Pattern** patterns = construct_pattern(input_file_patterns, num_of_PIs, num_pattern, gate_id_to_order_idx);
//     fclose(input_file_patterns);



//     // Simulation
//     unsigned int team = (num_pattern+deal_bits-1)/deal_bits;
//     int *found_fault_table = (int *)malloc((2*num_fault)*sizeof(int)); memset(found_fault_table, 0, (2*num_fault)*sizeof(int));
    
//     for (unsigned int i = 0; i < team; i++) {
//         // good simulation
//         unsigned int *good_results = good_case(gates, patterns[i], total_gates, simple_testcase_bits);
//         // Faults simulation 
//         for (int j = 0; j < num_fault; j++) {
//             // record whether the fault can be found
//             found_fault_table[2*j] = SA_faults(gates, faults[j], patterns[i], total_gates, num_POs, good_results, simple_testcase_bits);
//             // record which pattern found the fault
//             found_fault_table[2*j+1] = i;
//         }
//         free(good_results);
//     }

// #ifdef print_found_fault_table
//     printf_found_fault_table (found_fault_table, num_fault);
// #endif

//     /* Free data */
//     for (int i = 0; i < total_gates; i++) {
//         free_gate(gates[i]);
//     }
//     free(gates);
//     free(faults);
//     for (unsigned int i = 0; i < team; i++) {
//         free_pattern(patterns[i]);
//     }
//     free(patterns);
//     free(gate_id_to_order_idx);
//     free(found_fault_table);

    return 0;
}

