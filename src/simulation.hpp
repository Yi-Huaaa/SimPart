#ifndef SIMULATION_H
#define SIMULATION_H

#include "print_check.hpp"
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



unsigned int *good_case (Gate **gates, Pattern* pattern, int num_gate, int bits);
void print_check_simulation(unsigned int *results, int num_gate);
int SA_faults (Gate **gates, Fault faults, Pattern* pattern, int num_gate, int num_POs, unsigned int *good_results, int bits);

// Print for check
#define print_simulation

#endif
