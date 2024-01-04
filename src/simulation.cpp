#include <stdio.h>
#include <stdlib.h>

#include "simulation.hpp"


void INV (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *pre_gate = gate->input_gates[0];
        unsigned int b = pre_gate->output_value;
        gate->output_value = ~b;
    }
}


void AND (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *now_gate = gate->input_gates[0];
        unsigned int ret = (now_gate)->output_value;

        for (int i = 1; i < gate->num_of_inputs; i++) {
            now_gate = gate->input_gates[i];
            ret &= (now_gate)->output_value;
        }
        gate->output_value = ret;
    }
}


void OR (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *now_gate = gate->input_gates[0];
        unsigned int ret = (now_gate)->output_value;

        for (int i = 1; i < gate->num_of_inputs; i++) {
            now_gate = gate->input_gates[i];
            ret |= (now_gate)->output_value;
        }
        gate->output_value = ret;        
    }

}


void XOR (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *now_gate = gate->input_gates[0];
        unsigned int ret = (now_gate)->output_value;

        for (int i = 1; i < gate->num_of_inputs; i++) {
            now_gate = gate->input_gates[i];
            ret ^= (now_gate)->output_value;
        }
        gate->output_value = ret;        
    }

}


void NAND (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *now_gate = gate->input_gates[0];
        unsigned int ret = (now_gate)->output_value;

        for (int i = 1; i < gate->num_of_inputs; i++) {
            now_gate = gate->input_gates[i];
            ret &= (now_gate)->output_value;
        }
        gate->output_value = ~(ret);
    }
}


void NOR (Gate *gate, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
    } else {
        Gate *now_gate = gate->input_gates[0];
        unsigned int ret = (now_gate)->output_value;

        for (int i = 1; i < gate->num_of_inputs; i++) {
            now_gate = gate->input_gates[i];
            ret |= (now_gate)->output_value;
        }
        gate->output_value = ~(ret);
    }
}


void PI (Gate *gate, int PI_i, Pattern *pattern, int SA_fault, int fault_val) {
    if (SA_fault) {
        gate->output_value = fault_val;
        // printf("debug2: inside here, fault_val = %d\n", fault_val);
    } else {
        gate->output_value = pattern->value[PI_i];    
    }
}

void PO (Gate *gate) {
    // It outputs its previous gate's output_value if there is no SA fault
    Gate *pre_gate = gate->input_gates[0];
    gate->output_value = pre_gate->output_value;
}


void run_gate (Gate *gate, Pattern *pattern, int P_i, int SA_fault, int fault_val) {
    switch (gate->type) {
        case 0:
            INV(gate, SA_fault, fault_val);
            break;
        case 1:
            AND(gate, SA_fault, fault_val);
            break;
        case 2:
            OR(gate, SA_fault, fault_val);
            break;
        case 3:
            XOR(gate, SA_fault, fault_val);
            break;
        case 4:
            NAND(gate, SA_fault, fault_val);
            break;
        case 5:
            NOR(gate, SA_fault, fault_val);
            break;
        case PI_type_enum:
            PI(gate, P_i, pattern, SA_fault, fault_val);
            break;
        case PO_type_enum:
            PO(gate);
            break;
    }
}

void shift_to_correct_answer (unsigned int *results, int total_gates, int bits){
    for (int i = 0; i < total_gates; i++) {
        results[i]<<=(deal_bits-bits);
        results[i]>>=(deal_bits-bits);
    }
}


unsigned int *good_case (Gate **gates, Pattern* pattern, int total_gates, int bits) {
    // Simulation
    for (int i = 0; i < total_gates; i++) {
        run_gate(gates[i], pattern, i, 0, 0);
    }
    
    // Duplicated the answer of good_case into a new memory
    unsigned int *good_results = (unsigned int *)malloc(total_gates*sizeof(unsigned int));
    for (int i = 0; i < total_gates; i++) {
        good_results[i] = gates[i]->output_value;
    }

    if (bits < deal_bits){
        shift_to_correct_answer(good_results, total_gates, bits);
    }
    
#ifdef print_simulation
    printf("GOOD resutls\n");
    print_check_simulation(good_results, total_gates);
#endif
    
    return good_results;
}


int SA_faults (Gate **gates, Fault faults, Pattern* pattern, int total_gates, int num_POs, unsigned int *good_results, int bits) {
    
    int wrong_gate = faults.w_gate;
    int fault_val = faults.SA_fault_value;
    // Simulation
    for (int i = 0; i < total_gates; i++) {
        int SA_fault = (i == wrong_gate);
        // printf("debug1:i = %d, SA_fault = %d\n", i, SA_fault);
        run_gate(gates[i], pattern, i, SA_fault, fault_val);
    }
    // Duplicated the answer of BAD_case into a new memory
    unsigned int *bad_results = (unsigned int *)malloc(total_gates*sizeof(unsigned int));
    for (int i = 0; i < total_gates; i++) {
        bad_results[i] = gates[i]->output_value;
    }
    
    if (bits < deal_bits){
        shift_to_correct_answer(bad_results, total_gates, bits);
    }

#ifdef print_simulation
    printf("BAD resutls\n");
    print_check_simulation(bad_results, total_gates);
#endif

    // Record whether fault can be found
    int found_fault = 0;
    for (int i = (total_gates-1); i > (total_gates-num_POs-1); i--) {
        if (good_results[i] != bad_results[i]) {
            found_fault = 1;
        }
    }
    // printf("found_fault = %d\n", found_fault);

    free(bad_results);
    return found_fault;
}


void print_check_simulation(unsigned int *results, int total_gates) {
    for (int i = 0; i < total_gates; i++) {
        printf("gate_%d.output = %u\n", i, results[i]);
    }
}