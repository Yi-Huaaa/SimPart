#ifndef GRAPH_H
#define GRAPH_H

#include <string.h>
#include <vector>

using namespace std;

/**
 * Gate Type table:
 * (INV: 0, AND: 1, OR: 2, XOR: 3, NAND: 4, NOR: 5, PI: 6, PO: 7)
 * 
*/
typedef struct Gate {
    int type;
    int level; // gate level after topological sort

    int num_of_inputs;
    Gate **input_gates; // 該 gate 的 input gates, 屬於下家存上家
    int output_value; // 該 Gate 的 output value

} Gate;

#define PI_type_enum 6
#define PO_type_enum 7
#define default_level 0
#define default_output_value 0

// functions

Gate *new_gate (int type, int num_of_inputs);
Gate **construct_graph (FILE *input_file, int num_gate, int num_wire, int num_PIs, int num_POs);
void free_gate (Gate *gate);

void print_topological (vector<vector <int>> &adj, vector<int> &indegree, vector<int> &order);
void print_gate (Gate *gate);


// For printf checking
#define print_check_graph


#endif
