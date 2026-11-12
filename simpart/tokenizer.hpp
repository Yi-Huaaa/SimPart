#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simpart::detail {

// Minimal whitespace-separated reader of non-negative decimal integers.
class Tokenizer {
 public:
  explicit Tokenizer(std::string_view text) : p_(text.data()), end_(text.data() + text.size()) {}

  // Reads the next integer; returns false at end of input.
  bool next(uint64_t& value) {
    while (p_ != end_ && is_space(*p_)) ++p_;
    if (p_ == end_) return false;
    if (*p_ < '0' || *p_ > '9') {
      throw std::runtime_error(std::string("unexpected character '") + *p_ + "'");
    }
    uint64_t v = 0;
    while (p_ != end_ && *p_ >= '0' && *p_ <= '9') {
      const uint64_t d = static_cast<uint64_t>(*p_ - '0');
      if (v > (UINT64_MAX - d) / 10) throw std::runtime_error("integer overflow");
      v = v * 10 + d;
      ++p_;
    }
    if (p_ != end_ && !is_space(*p_)) {
      throw std::runtime_error(std::string("unexpected character '") + *p_ + "'");
    }
    value = v;
    return true;
  }

  // Reads the next integer and checks that it does not exceed max_value.
  uint64_t expect(const char* what, uint64_t max_value = UINT64_MAX) {
    uint64_t v = 0;
    if (!next(v)) throw std::runtime_error(std::string("unexpected end of input reading ") + what);
    if (v > max_value) {
      throw std::runtime_error(std::string("value ") + std::to_string(v) + " out of range for " +
                               what);
    }
    return v;
  }

  bool at_end() {
    while (p_ != end_ && is_space(*p_)) ++p_;
    return p_ == end_;
  }

 private:
  static bool is_space(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
  const char* p_;
  const char* end_;
};

}  // namespace simpart::detail
