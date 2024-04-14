#pragma once
#include "libint2_includes.hpp"
class AtomInfo {
public:
  size_t                      nbf;
  size_t                      nbf_lo;
  size_t                      nbf_hi;
  int                         atomic_number;
  std::string                 symbol;
  std::vector<libint2::Shell> shells; // shells corresponding to this atom
};