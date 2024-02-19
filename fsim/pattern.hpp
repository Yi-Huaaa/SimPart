#ifndef PATTERN_H
#define PATTERN_H

#include <iostream>
#include <stdio.h>
#include <vector>

namespace fsim {
class Simulator;
}

class Pattern {

  friend class fsim::Simulator;

public:
  // accessor
  // plain old data type can simply be returned as a copy
  std::vector<size_t> output_value() const { return _value;}

private:
  std::vector<size_t> _value;
};

#endif