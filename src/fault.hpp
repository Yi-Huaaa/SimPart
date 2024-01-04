#ifndef FAULT_H
#define FAULT_H

#include <stdio.h>

#define unsigned_int_MAX 4294967295

typedef struct Fault {
    int w_gate; // gate with SA fault
    int SA_fault_value;
} Fault;

Fault *construct_fault (FILE *input_file, int num_fault, int *gate_id_to_order_idx);
void print_fault (Fault *faults, int num_fault);

// Print for check
#define print_faults

#endif