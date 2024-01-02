#ifndef PATTERN_H
#define PATTERN_H

#include <stdio.h>

#define deal_bits 32

typedef struct Pattern {
    int num_PIs;
    unsigned int *value;

} Pattern;

// Functions 
Pattern ** construct_pattern (FILE *input_file, int num_of_PIs, int num_patterns);
void print_pattern (Pattern **patterns, unsigned int team, int num_PIs);
void free_pattern (Pattern *patterns);



/**
 * `printBits_stack` from stackoverflow: 
 * https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
*/
void printBits_stack(size_t const size, void const * const ptr);

#define print_patterns

#endif