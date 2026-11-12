#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace simpart {

// Input stimulus. Patterns are bit-packed 64 per word: one simulation round
// evaluates 64 patterns at once, and round r stores one word per PI.
struct Patterns {
  static constexpr int kBitsPerRound = 64;

  int64_t num_patterns = 0;
  int32_t num_pis = 0;
  int32_t num_rounds = 0;       // ceil(num_patterns / 64)
  std::vector<uint64_t> words;  // words[r * num_pis + pi]

  // Mask of the bits of round r that correspond to real patterns.
  uint64_t round_mask(int32_t r) const {
    const int64_t remaining = num_patterns - int64_t(r) * kBitsPerRound;
    return remaining >= kBitsPerRound ? ~uint64_t(0) : ((uint64_t(1) << remaining) - 1);
  }
};

// Parses a .ptn file: <#patterns> followed by ceil(#patterns/64) rounds of
// num_pis decimal 64-bit words.
Patterns parse_patterns(std::string_view text, int32_t num_pis);
Patterns read_patterns(const std::string& path, int32_t num_pis);

}  // namespace simpart
