#ifndef CUDA_GATE_H
#define CUDA_GATE_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <vector>
#include "./base_ppg.cuh"

#define default_level 0
#define default_output_value 0


class CudaSimulator;
class ElementBase;
class Pattern;
class Fault;

class Gate : public ElementBase {

  friend class CudaSimulator;
  friend class PI;
  friend class PO; 
  

public:


private:

};

#endif