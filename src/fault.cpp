#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "fault.hpp"


void print_fault (Fault *faults, int num_fault) {
    printf("faults = [\n");
    for (int i = 0; i < num_fault; i++) {
        printf("wire = %d, SA_%d\n", faults[i].wire, faults[i].SA_fault);
    }
    printf("]\n");
}

Fault *construct_fault (FILE *input_file, int num_wire, int num_fault) {
    Fault *faults = (Fault*)malloc(num_fault*sizeof(Fault));

    for (int i = 0; i < num_fault; i++) {
        int w, f;
        assert(fscanf(input_file, "%d%d", &w, &f) == 2);
        faults[i].wire = w;
        faults[i].SA_fault = f;
    }
#ifdef print_faults    
    print_fault(faults, num_fault);
#endif

    return faults;
}
