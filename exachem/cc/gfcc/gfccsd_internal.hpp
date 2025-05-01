/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/gfcc/gf_diis.hpp"
#include "exachem/cc/gfcc/gf_guess.hpp"
#include "exachem/common/chemenv.hpp"
#include "exachem/common/cutils.hpp"

using namespace tamm;

#include <filesystem>
namespace fs = std::filesystem;

#define GF_PGROUPS 1
#define GF_IN_SG 0
#define GF_GS_SG 0

namespace exachem::cc::gfcc {

template<typename... Ts>
std::string gfacc_str(Ts&&... args) {
  std::string res;
  (res.append(args), ...);
  res.append("\n");
  return res;
}

template<typename T>
T find_closest(T w, std::vector<T>& wlist) {
  double diff = std::abs(wlist[0] - w);
  int    idx  = 0;
  for(size_t c = 1; c < wlist.size(); c++) {
    double cdiff = std::abs(wlist[c] - w);
    if(cdiff < diff) {
      idx  = c;
      diff = cdiff;
    }
  }

  return wlist[idx];
}

}; // namespace exachem::cc::gfcc