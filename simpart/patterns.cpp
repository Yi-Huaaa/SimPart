#include "simpart/patterns.hpp"

#include <limits>
#include <stdexcept>

#include "simpart/circuit.hpp"
#include "simpart/tokenizer.hpp"

namespace simpart {

Patterns parse_patterns(std::string_view text, int32_t num_pis) {
  detail::Tokenizer tok(text);
  Patterns p;
  p.num_pis = num_pis;
  p.num_patterns = static_cast<int64_t>(
      tok.expect("number of patterns", std::numeric_limits<int32_t>::max()));
  p.num_rounds = static_cast<int32_t>((p.num_patterns + Patterns::kBitsPerRound - 1) /
                                      Patterns::kBitsPerRound);
  p.words.resize(static_cast<size_t>(p.num_rounds) * num_pis);
  for (auto& w : p.words) w = tok.expect("pattern word");
  return p;
}

Patterns read_patterns(const std::string& path, int32_t num_pis) {
  try {
    return parse_patterns(read_file(path), num_pis);
  } catch (const std::exception& e) {
    throw std::runtime_error("failed to read patterns " + path + ": " + e.what());
  }
}

}  // namespace simpart
