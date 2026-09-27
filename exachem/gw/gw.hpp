/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"
#include "exachem/scf/scf_main.hpp"

#include <cstdarg>
#include <cstdio>

using namespace tamm;

namespace exachem::gw {

// printf-style formatting into a std::string (used for the fixed-width GW output tables)
inline std::string gw_strfmt(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  va_list args2;
  va_copy(args2, args);
  const int n = std::vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  std::string out(n > 0 ? n : 0, '\0');
  if(n > 0) std::vsnprintf(out.data(), n + 1, fmt, args2);
  va_end(args2);
  return out;
}

// Per-spin-channel bookkeeping for the QP window.
// All orbital counts are spatial; index 0 is alpha, index 1 is beta.
struct GWData {
  int              ipol{1};  // number of spin channels (1: unpolarized, 2: polarized)
  int              nmo{0};   // number of spatial MOs (per channel)
  int              maxev{1}; // number of GW cycles to run (1 for G0W0)
  std::vector<int> nocc, nvir;
  std::vector<int> noqp, nvqp, nqp; // #occupied / #virtual / total QP states solved
  std::vector<int> lo, hi;          // QP window is the MO range [lo, hi)

  void print() const;
};

GWData gw_pars(const ChemEnv& chem_env, bool mrank);

// Groups near-degenerate states of the QP window into clusters.
std::vector<int> gw_findclusters(const std::vector<double>& vals, int nqp);

// Applies the scissor shift to the states outside the QP window.
void gw_scissor(const std::vector<double>& oldevals, std::vector<double>& newevals, int noqp,
                int nvqp, int nocc, int lo, int hi, int nmo, const std::string& spin, bool mrank);

void gw_print_iter(int inewton, double ein, double eout, double elower, double eupper,
                   bool bracket);

// Spatial MO index space of one spin channel with "occ" / "virt" subspaces; tile boundaries
// are aligned at nocc so occ/virt labels can be used in contractions.
TiledIndexSpace gw_setupMOIS(int nmo, int nocc, tamm::Tile tilesize);

void gw_driver(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::gw
