#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
// #include <omp.h>

#include "graph.hpp"
#include "fault.hpp"
// #include "pattern.hpp"
#include "simulation.hpp"



void PPSFP () {

}


int main(int argc, char* argv[]){
    
    /* Readin data - circit */ 
    FILE *input_file_circuit = fopen(argv[1], "r");
    assert(input_file_circuit != NULL);
    int num_PIs, num_POs, num_gate, num_wire;
    assert(fscanf(input_file_circuit, "%d%d%d%d", &num_PIs, &num_POs, &num_gate, &num_wire) == 4);
    
    /* Constrcut graph (gate) */
    Gate **gates = construct_graph(input_file_circuit, num_gate+num_PIs+num_POs, num_wire, num_PIs, num_POs);
    fclose(input_file_circuit);

    /* Readin data - faults*/ 
    FILE *input_file_faults = fopen(argv[2], "r");
    assert(input_file_faults != NULL);
    int num_fault;
    assert(fscanf(input_file_faults, "%d", &num_fault) == 1);

    /* Constrcut faults */
    Fault *faults = construct_fault(input_file_faults, num_wire, num_fault);
    fclose(input_file_faults);    


    // /* Readin data - patterns */ 
    // FILE *input_file_patterns = fopen(argv[3], "r");
    // assert(input_file_patterns != NULL);
    // int num_pattern;
    // assert(fscanf(input_file_patterns, "%d", &num_pattern) == 1);

    // /* Constrcut faults */
    // construt_pattern (input_file_patterns, num_pattern);

    // fclose(input_file_patterns);


    // Good cases


    // /* PPSFP */
    // for (auto f = 0; f < faults; f++) {
    //     for (auto p = 0; p < patterns; p++) {
    //         PPSFP ();
    //     }
    // }


    /* Free data */
    for (int i = 0; i < num_gate; i++) {
        free_gate(gates[i]);
    }
    free(gates);
    free(faults);

    return 0;
}
