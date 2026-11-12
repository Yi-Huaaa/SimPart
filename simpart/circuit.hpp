#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simpart {

// Gate types. Codes 0-8 are the type codes of the .ckt format; PI and PO are
// implied by the node index (see Circuit).
enum class GateType : uint8_t {
  INV = 0,
  AND = 1,
  OR = 2,
  XOR = 3,
  NAND = 4,
  NOR = 5,
  XNOR = 6,
  MUX = 7,  // inputs in fanin order (a, b, s): out = s ? b : a
  BUF = 8,  // CLKBUF in the original netlists
  PI = 9,
  PO = 10,
};

const char* to_string(GateType type);

// A combinational circuit stored as a directed acyclic graph in CSR form.
//
// Node indices follow the .ckt convention:
//   [0, num_pis)                      primary inputs
//   [num_pis, num_pis + num_pos)      primary outputs (single-input buffers)
//   [num_pis + num_pos, num_nodes())  gates
// Fanins of a node are kept in the order they appear in the input file.
struct Circuit {
  int32_t num_pis = 0;
  int32_t num_pos = 0;
  int32_t num_gates = 0;
  std::vector<GateType> types;           // per node
  std::vector<int32_t> fanin_offsets;    // size num_nodes() + 1
  std::vector<int32_t> fanins;           // size num_edges()
  std::vector<int32_t> fanout_offsets;   // size num_nodes() + 1
  std::vector<int32_t> fanouts;          // size num_edges()

  int32_t num_nodes() const { return num_pis + num_pos + num_gates; }
  int64_t num_edges() const { return static_cast<int64_t>(fanins.size()); }
  int32_t fanin_count(int32_t v) const { return fanin_offsets[v + 1] - fanin_offsets[v]; }
  int32_t fanout_count(int32_t v) const { return fanout_offsets[v + 1] - fanout_offsets[v]; }
  bool is_po(int32_t v) const { return v >= num_pis && v < num_pis + num_pos; }
};

// Parses a circuit in the .ckt format (see benchmark/README.md). Throws
// std::runtime_error on malformed input.
Circuit parse_circuit(std::string_view text);
Circuit read_circuit(const std::string& path);

// Reads a whole file into memory. Throws if the file cannot be read.
std::string read_file(const std::string& path);

}  // namespace simpart
