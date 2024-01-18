#ifndef GATE_H
#define GATE_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <vector>

#include "simulation.hpp"


#define default_level 0
#define default_output_value 0

namespace fsim {
  class Simulator;
}

// TODO: Gate
// How to access? GateType::INV
// Minimize the use of delete / new  /malloc /free
enum class GateType {
  INV = 0,
  AND,
  OR,
  XOR,
  NAND,
  NOR,
  PI,
  PO,
  MAX_GATE_TYPE
};

class Gate {

  friend class fsim::Simulator;
 
  public:

    // // accessor
    // GateType type() {
    //   return _type;
    // }
    // unsigned output_value() { return _output_value; }
    // size_t level() { return _level; } 

  private:
    GateType _type;
    std::vector<Gate *> _inputs;
    
    size_t _level; 
    size_t _output_value;

};

#endif