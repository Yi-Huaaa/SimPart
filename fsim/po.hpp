#ifndef PO_H
#define PO_H

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

class PO : public ElementBase<> {

  friend class fsim::Simulator;
  friend class CPUPartitioner;
  friend class CUDAPartitioner;
 
public:

  // accessor

private:

};


#endif
