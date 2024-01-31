#ifndef BASE_PPG_H
#define BASE_PPG_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#define default_level 0
#define default_output_value 0

namespace fsim {
class Simulator;
}

// TOOD: ElementType
enum class GateType { INV = 0, AND, OR, XOR, NAND, NOR, PI, PO, MAX_GATE_TYPE };

// TODO: ElementBase
class BasePPG {

  friend class fsim::Simulator;

public:
  // accessor
  // plain old data type can simply be returned as a copy
  size_t idx() const { return _idx; }
  size_t level() const { return _level; }
  size_t output_value() const { return _output_value; }

private:
  GateType _type;
  std::vector<BasePPG *> _inputs;

  size_t _idx;
  size_t _level;
  size_t _output_value;

  // TODO: std::optional
  //       std::variant<int, float, std::string> storage
};

#endif