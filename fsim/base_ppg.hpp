#ifndef BASE_PPG_H
#define BASE_PPG_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <variant>
#include <vector>

#define default_level 0
#define default_output_value 0

namespace fsim {
class Simulator;
}

// TODO: ElementType
enum class GateType { 
  INV = 0, 
  AND, // 1 
  OR, // 2
  XOR,  // 3
  NAND, // 4
  NOR, // 5
  XNOR, // 6
  MUX, // 7
  CLKBUF, // 8
  PI,  // 30
  PO,  // 31
  MAX_GATE_TYPE
};

template<typename IndexType = size_t, typename LevelType = size_t>
class ElementBase {

  friend class fsim::Simulator;
  friend class CPUPartitioner;
  friend class CUDAPartitioner;

  // This is the correct syntax for declaring a template class as a friend
  template<typename T> 
  friend class GALPS_CPUSimulator;
  friend class GALPS_GPUSimulator;

public:
  // accessor
  IndexType idx() const { return _idx; }
  LevelType level() const { return _level; }
  size_t output_value() const { return _output_value; }

private:
  GateType _type;
  std::vector<ElementBase *> _inputs;

  IndexType _idx;
  LevelType _level;
  size_t _output_value;

  // TODO: std::optional
  //       std::variant<int, float, std::string> storage
};

#endif
