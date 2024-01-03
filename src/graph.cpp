#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <queue>

#include "graph.hpp"
using namespace std;

// note: function Gate *new_gate does not init input gates, only malloc the needed memory
Gate *new_gate (int type, int num_of_inputs) {
    Gate *gate = (Gate *)malloc(sizeof(Gate));
    gate->type = type;
    gate->level = default_level;
    gate->num_of_inputs = num_of_inputs;
    gate->input_gates = (Gate **)malloc(num_of_inputs*sizeof(Gate *));
    gate->output_value = default_output_value;
    return gate;
}

void free_gate (Gate *gate) {
    free(gate->input_gates);
    free(gate);
}

Gate **construct_graph (FILE *input_file, int num_gate, int num_wire, int num_PIs, int num_POs) {
    vector<vector <int>> adj;
    adj.resize(num_gate);
    vector<int> indegree, indegree_orig;
    indegree.resize(num_gate);
    
    /** Levelized the input gates before constructing the circuit graph */
    // read file, constrcut adjacent list, construct indegree array
    for (int i = 0; i < num_wire; i++) {
        int gate_0, pin_Y, post_num_gates;
        assert(fscanf(input_file, "%d%d%d", &gate_0, &pin_Y, &post_num_gates) == 3);
        for (int j = 0; j < post_num_gates; j++) {
            int gate_tmp, pin_tmp;
            assert(fscanf(input_file, "%d%d", &gate_tmp, &pin_tmp) == 2);
            adj[gate_0].push_back(gate_tmp);
            indegree[gate_tmp]++;
        }
    }

    // copy `indegree` vector since we need its original value later
    indegree_orig = indegree;
    
    // Kahn's algorithm, topological sort
    queue <int> source;
    vector<int> order;
    
    // since only PIs will be indegree == 0
    for (int i = 0; i < num_PIs; i++) {
        source.push(i);
    }

    while (!source.empty()) {
        int first = source.front();
        source.pop(); // queue pop front element
        order.push_back(first);
        for (size_t i = 0; i < adj[first].size(); i++) {
            indegree[adj[first][i]]--;
            if (indegree[adj[first][i]] == 0) {
                source.push(adj[first][i]);
            }
        }
    }
    
#ifdef print_graph
    print_topological (adj, indegree_orig, order);
#endif

    /** Constructing circuit graph */
    // scanf gate type array
    int *gate_type = (int*)malloc(num_gate*sizeof(int));
    for (int i = 0; i < num_PIs; i++) {
        gate_type[i] = PI_type_enum;
    }
    for (int i = num_PIs; i < (num_PIs+num_POs); i++) {
        gate_type[i] = PO_type_enum;
    }
    for (int i = (num_PIs+num_POs); i < (num_gate); i++) {
        assert(fscanf(input_file, "%d", &gate_type[i]) == 1);
    }    
    
    // mapping gate_idx to gate_order_index
    /**
     * CML example:
     * order[3] = 6
     * gate_id_to_order_idx[6] = 3
     * gate_id_to_order_idx: 0, 1, 2, 7, 5, 6, 3, 4
     * */ 

    int *gate_id_to_order_idx = (int*)malloc(num_gate*sizeof(int));
    for (size_t i = 0; i < order.size(); i++) {
        gate_id_to_order_idx[order[i]] = i;
    }

    // gates are stored in the order specified by the `order` array
    Gate **gates = (Gate **)malloc(num_gate*sizeof(Gate *));
    for (int i = 0; i < num_gate; i++) {
        // `gate_id` is the ID assigned in the input file
        int gate_id = order[i];
        int type = gate_type[gate_id];
        int num_of_inputs = indegree_orig[gate_id];
        gates[i] = new_gate(type, num_of_inputs);
    }

    // keep track of the next index for the `input_gates` array, given a gate ID
    int *input_gates_next_idx = (int*)malloc(num_gate*sizeof(int));
    memset(input_gates_next_idx, 0, num_gate*sizeof(int));

    for (size_t i = 0; i < adj.size(); i++) {
        for (size_t j = 0; j < adj[i].size(); j++) {
            int from_gate_id = i, to_gate_id = adj[i][j];
            Gate *from_gate = gates[gate_id_to_order_idx[from_gate_id]];
            Gate *to_gate = gates[gate_id_to_order_idx[to_gate_id]];
            to_gate->input_gates[input_gates_next_idx[to_gate_id]++] = from_gate;
        }
    }

    // free everything
    free(gate_type);
    free(gate_id_to_order_idx);
    free(input_gates_next_idx);

#ifdef print_graph
    for (int i = 0; i < num_gate; i++) {
        print_gate(gates[i]);
    }
#endif

    return gates;
}


void print_topological (vector<vector <int>> &adj, vector<int> &indegree, vector<int> &order) {
    printf("adj:\n");
    for (size_t i = 0; i < adj.size(); i++) {
        printf("%ld = [", i);
        for (size_t v_len = 0; v_len < adj[i].size(); v_len++) {
            printf("%d, ", adj[i][v_len]);
        } printf("]\n");
    } printf("\n");
    
    printf("indegree:\n");
    for (size_t i = 0; i < indegree.size(); i++) {
        printf("%ld = [%d]\n", i, indegree[i]);
    }

    printf("Order:\n");
    for (size_t i = 0; i < order.size(); i++) {
        printf("%d, ", order[i]);
    } printf("\n");
}


void print_gate(Gate *gate) {
    printf("=====\n");
    printf("id: %p\n", gate);
    printf("type: %d\n", gate->type);
    printf("level: %d\n", gate->level);
    printf("num_of_inputs: %d\n", gate->num_of_inputs);
    printf("input_gates: [");
    for (int i = 0; i < gate->num_of_inputs; i++) {
        printf("%p, ", gate->input_gates[i]);
    }
    printf("]\n");
    printf("output_value: %u\n", gate->output_value);
}