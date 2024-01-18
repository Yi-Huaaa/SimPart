#ifndef FAULT_H
#define FAULT_H

#include <cstddef>

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