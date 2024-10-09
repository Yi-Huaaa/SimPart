#ifndef GATE_H
#define GATE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <vector>
#include "./base_ppg.hpp"

#define default_level 0
#define default_output_value 0

namespace fsim {
class Simulator;
}

// enum class GateType {
//   INV = 0,
//   AND,
//   OR,
//   XOR,
//   NAND,
//   NOR,
//   PI,
//   PO,
//   MAX_GATE_TYPE
// };

class Gate : public ElementBase<> {

  friend class fsim::Simulator;
  friend class PI;
  friend class PO; 
  friend class CPUPartitioner;
  friend class CUDAPartitioner;
  
public:

  // // // accessor
  // // GateType type() {
  // //   return _type;
  // // }
  // // unsigned output_value() { return _output_value; }
  // // size_t level() { return _level; } 

  // // plain old data type can simply be returned as a copy
  // size_t level() const {
  //   return _level;
  // }

private:
  // GateType _type;
  // std::vector<Gate *> _inputs;
  // std::vector<PI *> _pi_inputs;
  
  // size_t _idx;
  // size_t _level; 
  // size_t _output_value;

};

#endif