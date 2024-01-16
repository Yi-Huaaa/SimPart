#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "fault.hpp"


// void print_fault (Fault *faults, int num_fault) {
//     printf("\n=====\n\n");
//     printf("Fault array, this order follows the order after topological sort, wiht %lu == 1\n", unsigned_int_MAX);
//     printf("faults = [\n");
//     for (int i = 0; i < num_fault; i++) {
//         printf("w_gate = %d, SA_%u\n", faults[i].w_gate, faults[i].SA_fault_value);
//     }
//     printf("]\n");
// }

// Fault *construct_fault (FILE *input_file, int num_fault, int *gate_id_to_order_idx) {
//     Fault *faults = (Fault*)malloc(num_fault*sizeof(Fault));

//     for (int i = 0; i < num_fault; i++) {
//         int wrong_gate, fault_value;
//         assert(fscanf(input_file, "%d%d", &wrong_gate, &fault_value) == 2);
//         faults[i].w_gate = gate_id_to_order_idx[wrong_gate];
//         faults[i].SA_fault_value = (fault_value) ? (unsigned_int_MAX) : (0);
    
//     }
// #ifdef print_faults    
//     print_fault(faults, num_fault);
// #endif

//     return faults;
// }
