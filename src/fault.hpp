#ifndef FAULT_H
#define FAULT_H

#include <stdio.h>

#include "print_check.hpp"

#define unsigned_int_MAX 4294967295

typedef struct Fault {
    int w_gate; // gate with SA fault
    int SA_fault_value;
} Fault;

Fault *construct_fault (FILE *input_file, int num_fault);
void print_fault (Fault *faults, int num_fault);

// Print for check
#define print_faults

#endif