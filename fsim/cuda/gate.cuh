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


class CUDASimulator;
class ElementBase;
class Pattern;
class Fault;
class CUDAPartitioner;

class Gate : public ElementBase {

  friend class CUDASimulator;
  friend class PI;
  friend class PO; 
  friend class CUDAPartitioner;
  

public:


private:

};

#endif