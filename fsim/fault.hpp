#ifndef FAULT_H
#define FAULT_H

#include <stdio.h>
#define print_faults

// typedef struct Fault {
//     int w_gate; // gate with SA fault
//     int SA_fault_value;
// } Fault;

// Fault *construct_fault (FILE *input_file, int num_fault, int *gate_id_to_order_idx);
// void print_fault (Fault *faults, int num_fault);

namespace fsim {
  class Simulator;
}

class Fault {

  friend class fsim::Simulator;

  public:

    // accessor

  private:
    size_t _gate_with_fault;
    size_t _gate_SA_fault_val; // 0 or 1

};

#endif