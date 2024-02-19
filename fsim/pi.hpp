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
class CudaSimulator;
}

class PI : public ElementBase {

  friend class fsim::Simulator;
  friend class fsim::CudaSimulator;
 
public:

  // accessor

private:

};


#endif
