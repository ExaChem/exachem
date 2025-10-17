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

template<typename T>
class GFCCSDInternal {
public:
  // Constructor
  GFCCSDInternal() = default;

  // Destructor
  virtual ~GFCCSDInternal() = default;

  GFCCSDInternal(const GFCCSDInternal&)            = default;
  GFCCSDInternal& operator=(const GFCCSDInternal&) = default;
  GFCCSDInternal(GFCCSDInternal&&)                 = default;
  GFCCSDInternal& operator=(GFCCSDInternal&&)      = default;

  /**
   * @brief Helper function to create formatted strings for GF-CCSD output
   */
  template<typename... Ts>
  static std::string gfacc_str(Ts&&... args) {
    std::string res;
    (res.append(args), ...);
    res.append("\n");
    return res;
  }

  /**
   * @brief Find the closest value to w in wlist
   */
  virtual T find_closest(T w, std::vector<T>& wlist) {
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
};

// Backward compatibility wrapper functions
template<typename... Ts>
std::string gfacc_str(Ts&&... args) {
  GFCCSDInternal<double> gfccsd_internal;
  return gfccsd_internal.gfacc_str(std::forward<Ts>(args)...);
}

template<typename T>
T find_closest(T w, std::vector<T>& wlist) {
  GFCCSDInternal<T> gfccsd_internal;
  return gfccsd_internal.find_closest(w, wlist);
}

}; // namespace exachem::cc::gfcc