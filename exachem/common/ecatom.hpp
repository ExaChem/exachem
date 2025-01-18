/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <iostream>
#include <vector>

#include "exachem/common/atom_info.hpp"
#include "exachem/common/libint2_includes.hpp"
#include "exachem/common/txt_utils.hpp"
#include "tamm/tamm.hpp"
using namespace tamm;
using libint2::Atom;

class ECAtom: public AtomInfo {
public:
  Atom                atom;
  std::string         esymbol;
  std::string         basis;
  bool                has_ecp{false};
  int                 ecp_nelec{};
  std::vector<double> ecp_coeffs{};
  std::vector<double> ecp_exps{};
  std::vector<int>    ecp_ams{};
  std::vector<int>    ecp_ns{};
  static int          get_atomic_number(std::string element_symbol);
  static std::string  get_symbol(const int atomic_number);
};
