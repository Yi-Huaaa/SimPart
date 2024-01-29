#ifndef BASE_PPG_H
#define BASE_PPG_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <vector>


#define default_level 0
#define default_output_value 0

namespace fsim {
class Simulator;
}

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

class BasePPG {

  friend class fsim::Simulator;

public:

  // accessor
  // plain old data type can simply be returned as a copy
  size_t level() const {
    return _level;
  }

private:
  GateType _type;
  std::vector<BasePPG *> _inputs;
  
  size_t _idx;
  size_t _level; 
  size_t _output_value;

};


#endif
