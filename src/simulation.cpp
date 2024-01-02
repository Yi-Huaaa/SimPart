#include <stdio.h>
#include <stdlib.h>

#include "simulation.hpp"


void INV (Gate *gate) {
    Gate *pre_gate = gate->input_gates[0];
    unsigned int b = pre_gate->output_value;
    gate->output_value = ~b;
}

void AND (Gate *gate) {
    Gate *now_gate = gate->input_gates[0];
    unsigned int ret = (now_gate)->output_value;

    for (int i = 1; i < gate->num_of_inputs; i++) {
        now_gate = gate->input_gates[i];
        ret &= (now_gate)->output_value;
    }
    gate->output_value = ret;
}

void OR (Gate *gate) {
    Gate *now_gate = gate->input_gates[0];
    unsigned int ret = (now_gate)->output_value;

    for (int i = 1; i < gate->num_of_inputs; i++) {
        now_gate = gate->input_gates[i];
        ret |= (now_gate)->output_value;
    }
    gate->output_value = ret;
}

void XOR (Gate *gate) {
    Gate *now_gate = gate->input_gates[0];
    unsigned int ret = (now_gate)->output_value;

    for (int i = 1; i < gate->num_of_inputs; i++) {
        now_gate = gate->input_gates[i];
        ret ^= (now_gate)->output_value;
    }
    gate->output_value = ret;
}

void NAND (Gate *gate) {
    Gate *now_gate = gate->input_gates[0];
    unsigned int ret = (now_gate)->output_value;

    for (int i = 1; i < gate->num_of_inputs; i++) {
        now_gate = gate->input_gates[i];
        ret &= (now_gate)->output_value;
    }
    gate->output_value = ~(ret);
}

void NOR (Gate *gate) {
    Gate *now_gate = gate->input_gates[0];
    unsigned int ret = (now_gate)->output_value;

    for (int i = 1; i < gate->num_of_inputs; i++) {
        now_gate = gate->input_gates[i];
        ret |= (now_gate)->output_value;
    }
    gate->output_value = ~(ret);
}

void PI (Gate *gate, int PI_i, Pattern *pattern) {
    gate->output_value = pattern->value[PI_i];
}

void PO (Gate *gate) {
    // It outputs its previous gate's output_value if there is no SA fault
    Gate *pre_gate = gate->input_gates[0];
    gate->output_value = pre_gate->output_value;
}

void run_gate (Gate *gate, Pattern *pattern, int P_i) {
    switch (gate->type) {
        case 0:
            INV(gate);
            break;
        case 1:
            AND(gate);
            break;
        case 2:
            OR(gate);
            break;
        case 3:
            XOR(gate);
            break;
        case 4:
            NAND(gate);
            break;
        case 5:
            NOR(gate);
            break;
        case PI_type_enum:
            PI(gate, P_i, pattern);
            break;
        case PO_type_enum:
            PO(gate);
            break;
    }
}


unsigned int *good_case (Gate **gates, Fault *faults, Pattern* pattern,
                int num_PIs, int num_POs, int num_gate, 
                int num_wire, int num_fault, int num_pattern) {
    
    for (int i = 0; i < num_gate; i++) {
        run_gate(gates[i], pattern, i);
    }
    
    // Duplicated the answer of good_case into a new memory
    unsigned int *good_results = (unsigned int *)malloc(num_gate*sizeof(unsigned int));
    for (int i = 0; i < num_gate; i++) {
        good_results[i] = gates[i]->output_value;
    }
    
#ifdef check_simulation_good
    print_check_good_simulation(good_results, num_gate);
#endif

    return good_results;
}

void print_check_good_simulation(unsigned int *good_results, int num_gate) {
    for (int i = 0; i < num_gate; i++) {
        printf("gate_%d.output = %u\n", i, good_results[i]);
    }
}