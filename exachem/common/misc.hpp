/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <iostream>

#include "libint2_includes.hpp"

struct BasisFunctionCount {
  int s_count = 0;
  int p_count = 0;
  int d_count = 0;
  int f_count = 0;
  int g_count = 0;
  int h_count = 0;
  int i_count = 0;

  int sp_count      = 0;
  int spd_count     = 0;
  int spdf_count    = 0;
  int spdfg_count   = 0;
  int spdfgh_count  = 0;
  int spdfghi_count = 0;

  // s_type = 0, p_type = 1, d_type = 2,
  // f_type = 3, g_type = 4, h_type = 5, i_type = 6
  /*For spherical
  n = 2*type + 1
  s=1,p=3,d=5,f=7,g=9,h=11,i=13
  For cartesian
  n = (type+1)*(type+2)/2
  s=1,p=3,d=6,f=10,g=15,h=21,i=28
  */
  void compute(const bool is_spherical, libint2::Shell& s) {
    // for (const auto& s: shells) {
    for(const auto& c: s.contr) {
      const int n_sph  = 2 * c.l + 1;
      const int n_cart = ((c.l + 1) * (c.l + 2)) / 2;
      if(c.l == 0) s_count += n_sph;
      if(c.l == 1) p_count += n_sph;
      if(c.l == 2) {
        if(is_spherical) d_count += n_sph;
        else d_count += n_cart;
      }
      if(c.l == 3) {
        if(is_spherical) f_count += n_sph;
        else f_count += n_cart;
      }
      if(c.l == 4) {
        if(is_spherical) g_count += n_sph;
        else g_count += n_cart;
      }
      if(c.l == 5) {
        if(is_spherical) h_count += n_sph;
        else h_count += n_cart;
      }
      if(c.l == 6) {
        if(is_spherical) i_count += n_sph;
        else i_count += n_cart;
      }
    }
    // }

    sp_count      = s_count + p_count;
    spd_count     = sp_count + d_count;
    spdf_count    = spd_count + f_count;
    spdfg_count   = spdf_count + g_count;
    spdfgh_count  = spdfg_count + h_count;
    spdfghi_count = spdfgh_count + i_count;
  }

  void print() {
    std::cout << "s = " << s_count << ", p = " << p_count << ", d = " << d_count
              << ", f = " << f_count << ", g = " << g_count << ", h = " << h_count << std::endl;
    std::cout << "sp = " << sp_count << ", spd = " << spd_count << ", spdf = " << spdf_count
              << ", spdfg = " << spdfg_count << ", spdfgh = " << spdfgh_count << std::endl;
  }
};

struct AtomInfo {
  size_t                      nbf;
  size_t                      nbf_lo;
  size_t                      nbf_hi;
  int                         atomic_number;
  std::string                 symbol;
  std::vector<libint2::Shell> shells; // shells corresponding to this atom
};

struct BasisSetMap {
  size_t                nbf;
  size_t                natoms;
  size_t                nshells;
  std::vector<AtomInfo> atominfo; // natoms
  // map from atoms to the corresponding shells
  std::vector<std::vector<long>> atom2shell;
  // shell2atom(nshells) - the given atom a shell is assigned to.
  std::vector<long> shell2atom;
  // bf2shell(nbf)    - the given shell a function is assigned to.
  std::vector<size_t> bf2shell;
  // shell2bf(nshells) - First basis function on a given shell.
  std::vector<size_t> shell2bf;
  // bf2atom(nbf)     - the given atom a function is assigned to.
  std::vector<size_t> bf2atom;
  // nbf_atom(natoms)     - number of basis functions on a given atom.
  std::vector<size_t> nbf_atom;
  // nshells_atom(natoms)   - number of shells on a given atom.
  std::vector<size_t> nshells_atom;
  // first_bf_atom(natoms) - First basis function on a given atom.
  std::vector<size_t> first_bf_atom;
  // first_bf_shell(nshells)    - number of basis functions on a given shell.
  std::vector<size_t> first_bf_shell;
  // first_shell_atom(natoms) - First shell on a given atom.
  std::vector<size_t> first_shell_atom;
  // gaussian basis function components
  std::map<size_t, std::string> bf_comp;
};
