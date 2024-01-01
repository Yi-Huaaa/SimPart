#ifndef FAULT_H
#define FAULT_H

typedef struct Fault {
    int wire;
    int SA_fault;
} Fault;

Fault *construct_fault (FILE *input_file, int num_wire, int num_fault);
void print_fault (Fault *faults, int num_fault);

#define print_faults

#endif