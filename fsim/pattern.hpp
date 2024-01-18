#ifndef PATTERN_H
#define PATTERN_H

#include <stdio.h>
#include <iostream>
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