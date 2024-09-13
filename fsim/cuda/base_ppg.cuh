#ifndef CUDA_BASE_PPG_H
#define CUDA_BASE_PPG_H

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#define default_level 0
#define default_output_value 0


// Forward declaration
class CUDASimulator;
class Pattern;
class Fault;
class PI;
class Gate;
class PO;
class FFR;
class CUDAPartitioner;


// TOOD: ElementType
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
  PI,  // 9
  PO,  // 10
  MAX_GATE_TYPE
};


class ElementBase {

  friend class CUDASimulator;
  friend class CUDAPartitioner;


public:
  // accessor
  // plain old data type can simply be returned as a copy
  size_t idx() const { return _idx; }
  size_t level() const { return _level; }
  size_t output_value() const { return _output_value; }

  
private:
  GateType _type;
  // std::vector<ElementBase *> _inputs;
  ElementBase **_inputs;

  size_t _num_inputs; // store how many input gates
  
  size_t _idx;
  size_t _level;
  size_t _output_value;
  // size_t _output_val_start_idx; 
  // size_t _output_val_end_idx; 
};

#endif