#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "pattern.hpp"


Pattern *new_pattern (FILE *input_file, int num_PIs, int *gate_id_to_order_idx) {
    Pattern *pattern = (Pattern *)malloc(sizeof(Pattern));
    pattern->num_PIs = num_PIs;
    pattern->value = (unsigned int *)malloc(num_PIs*sizeof(unsigned int));
    for (int i = 0; i < num_PIs; i++) {
        assert(fscanf(input_file, "%u", &(pattern->value[gate_id_to_order_idx[i]])) == 1);
        // printf("i = %d, pattern->value[%d] = %d\n", i, gate_id_to_order_idx[i], pattern->value[gate_id_to_order_idx[i]]);
    }
    return pattern;
}

Pattern **construct_pattern (FILE *input_file, int num_PIs, int num_pattern, int *gate_id_to_order_idx) {
    unsigned int team = (num_pattern+deal_bits-1)/deal_bits; // ceiling
    Pattern **patterns = (Pattern **)malloc(team*sizeof(Pattern *));
    for (unsigned int i = 0; i < team; i++) {
        patterns[i] = new_pattern(input_file, num_PIs, gate_id_to_order_idx);
    }

#ifdef print_patterns    
    print_pattern(patterns, team, num_PIs);
#endif
    
    return patterns;
}

void free_pattern (Pattern *patterns) {
    free(patterns->value);
    free(patterns);
}

void print_pattern (Pattern **patterns, unsigned int team, int num_PIs) {
    printf("\n=====\n\n");
    for (unsigned int i = 0; i < team; i++) {
        printf("[%u, %u] bits = [\n", deal_bits*i, deal_bits*(i+1));
        for (int j = 0; j < num_PIs; j++) {
            // printf("%b, ", (patterns[i]->value[j]));
            printBits_stack(sizeof(patterns[i]->value[j]), &patterns[i]->value[j]);
        }
        printf("]\n");
    }    
}

void printBits_stack(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    
    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

void printf_found_fault_table (int *found_fault_table, int num_fault) {
    printf("\n=====\n\n");
    for (int i = 0; i < (2*num_fault); i+=2) {
        if (found_fault_table[i] == 1) {
            printf("fault_%d is found, with Pattern_%d\n", i, found_fault_table[i+1]);
        } else {
            printf("fault_%d cannot be found\n", i);
        }
    }
}