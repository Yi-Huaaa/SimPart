#ifndef SIMULATION_H
#define SIMULATION_H


#include "graph.hpp"
#include "fault.hpp"
#include "pattern.hpp"


/* Functions */
void INV (Gate *gate);
void AND (Gate *gate);
void OR (Gate *gate);
void XOR (Gate *gate);
void NAND (Gate *gate);
void NOR (Gate *gate);
void PI (Gate *gate, int PI_i, Pattern *pattern);
void PO (Gate *gate);



unsigned int *good_case (Gate **gates, Fault *faults, Pattern* pattern,
                int num_PIs, int num_POs, int num_gate, 
                int num_wire, int num_fault, int num_pattern);

void print_check_good_simulation(unsigned int *good_results, int num_gate);

#define check_simulation_good

#endif
