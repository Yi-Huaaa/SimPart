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

private:
  std::vector<size_t> _value;
};

#endif