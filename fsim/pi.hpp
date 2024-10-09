#ifndef PI_H
#define PI_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <vector>
#include "./base_ppg.hpp"

// forward declaration 
namespace fsim {
class Simulator;
}

class PI : public ElementBase<> {

  friend class fsim::Simulator;
  friend class CPUPartitioner;
  friend class CUDAPartitioner;
   
public:

  // accessor

private:

};


#endif
