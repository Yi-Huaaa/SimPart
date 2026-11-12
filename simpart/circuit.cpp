#include "simpart/circuit.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "simpart/tokenizer.hpp"

namespace simpart {

const char* to_string(GateType type) {
  switch (type) {
    case GateType::INV: return "INV";
    case GateType::AND: return "AND";
    case GateType::OR: return "OR";
    case GateType::XOR: return "XOR";
    case GateType::NAND: return "NAND";
    case GateType::NOR: return "NOR";
    case GateType::XNOR: return "XNOR";
    case GateType::MUX: return "MUX";
    case GateType::BUF: return "BUF";
    case GateType::PI: return "PI";
    case GateType::PO: return "PO";
  }
  return "UNKNOWN";
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return std::move(ss).str();
}

namespace {

constexpr uint64_t kMaxIndex = std::numeric_limits<int32_t>::max();

void check_fanin_count(const Circuit& c, int32_t v) {
  const int32_t n = c.fanin_count(v);
  const GateType t = c.types[v];
  bool ok = true;
  switch (t) {
    case GateType::PI: ok = (n == 0); break;
    case GateType::PO:
    case GateType::INV:
    case GateType::BUF: ok = (n == 1); break;
    case GateType::MUX: ok = (n == 3); break;
    default: ok = (n >= 1); break;
  }
  if (!ok) {
    throw std::runtime_error("node " + std::to_string(v) + " of type " + to_string(t) + " has " +
                             std::to_string(n) + " fanin(s)");
  }
}

// Builds a CSR adjacency (offsets, targets) grouped by key[e], preserving the
// relative order of edges with the same key.
void build_csr(int32_t num_nodes, const std::vector<int32_t>& key, const std::vector<int32_t>& value,
               std::vector<int32_t>& offsets, std::vector<int32_t>& targets) {
  offsets.assign(static_cast<size_t>(num_nodes) + 1, 0);
  for (int32_t k : key) ++offsets[k + 1];
  for (int32_t v = 0; v < num_nodes; ++v) offsets[v + 1] += offsets[v];
  targets.resize(key.size());
  std::vector<int32_t> cursor(offsets.begin(), offsets.end() - 1);
  for (size_t e = 0; e < key.size(); ++e) targets[cursor[key[e]]++] = value[e];
}

}  // namespace

Circuit parse_circuit(std::string_view text) {
  detail::Tokenizer tok(text);
  Circuit c;
  c.num_pis = static_cast<int32_t>(tok.expect("number of PIs", kMaxIndex));
  c.num_pos = static_cast<int32_t>(tok.expect("number of POs", kMaxIndex));
  c.num_gates = static_cast<int32_t>(tok.expect("number of gates", kMaxIndex));
  const uint64_t num_wires = tok.expect("number of wires", kMaxIndex);
  const uint64_t n64 = uint64_t(c.num_pis) + uint64_t(c.num_pos) + uint64_t(c.num_gates);
  if (n64 > kMaxIndex) throw std::runtime_error("too many nodes");
  const int32_t n = static_cast<int32_t>(n64);

  // Wires: <driver> <driver pin> <#sinks> (<sink> <sink pin>)*
  std::vector<int32_t> src, dst;
  src.reserve(n);
  dst.reserve(n);
  for (uint64_t w = 0; w < num_wires; ++w) {
    const auto driver = static_cast<int32_t>(tok.expect("wire driver", n - 1));
    tok.expect("wire driver pin");
    const uint64_t num_sinks = tok.expect("wire fanout count", kMaxIndex);
    for (uint64_t s = 0; s < num_sinks; ++s) {
      const auto sink = static_cast<int32_t>(tok.expect("wire sink", n - 1));
      tok.expect("wire sink pin");
      src.push_back(driver);
      dst.push_back(sink);
    }
    if (src.size() > kMaxIndex) throw std::runtime_error("too many edges");
  }

  // Gate types of nodes [num_pis + num_pos, n).
  c.types.assign(n, GateType::PI);
  for (int32_t v = c.num_pis; v < c.num_pis + c.num_pos; ++v) c.types[v] = GateType::PO;
  for (int32_t v = c.num_pis + c.num_pos; v < n; ++v) {
    c.types[v] = static_cast<GateType>(tok.expect("gate type", uint64_t(GateType::BUF)));
  }
  if (!tok.at_end()) throw std::runtime_error("trailing data after gate types");

  build_csr(n, dst, src, c.fanin_offsets, c.fanins);
  build_csr(n, src, dst, c.fanout_offsets, c.fanouts);
  for (int32_t v = 0; v < n; ++v) check_fanin_count(c, v);
  return c;
}

Circuit read_circuit(const std::string& path) {
  try {
    return parse_circuit(read_file(path));
  } catch (const std::exception& e) {
    throw std::runtime_error("failed to read circuit " + path + ": " + e.what());
  }
}

}  // namespace simpart
