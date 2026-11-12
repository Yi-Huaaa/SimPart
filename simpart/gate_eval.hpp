#pragma once

#include <cstdint>

#include "simpart/circuit.hpp"

#if defined(__CUDACC__)
#define SIMPART_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define SIMPART_HOST_DEVICE inline
#endif

namespace simpart {

// Evaluates a (non-PI) node on 64 bit-packed patterns. `input(i)` returns the
// value of the i-th fanin (fanins in CSR order, 0 <= i < num_fanins).
template <typename Input>
SIMPART_HOST_DEVICE uint64_t eval_gate(GateType type, int32_t num_fanins, const Input& input) {
  uint64_t r = input(0);
  switch (type) {
    case GateType::INV: return ~r;
    case GateType::BUF:
    case GateType::PO: return r;
    case GateType::AND:
    case GateType::NAND:
      for (int32_t i = 1; i < num_fanins; ++i) r &= input(i);
      return type == GateType::NAND ? ~r : r;
    case GateType::OR:
    case GateType::NOR:
      for (int32_t i = 1; i < num_fanins; ++i) r |= input(i);
      return type == GateType::NOR ? ~r : r;
    case GateType::XOR:
    case GateType::XNOR:
      for (int32_t i = 1; i < num_fanins; ++i) r ^= input(i);
      return type == GateType::XNOR ? ~r : r;
    case GateType::MUX: {
      const uint64_t b = input(1);
      const uint64_t s = input(2);
      return (s & b) | (~s & r);
    }
    case GateType::PI: break;  // PIs are assigned from the patterns, never evaluated
  }
  return 0;
}

}  // namespace simpart
