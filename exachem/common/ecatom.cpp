/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ecatom.hpp"
int ECAtom::get_atomic_number(std::string element_symbol) {
  int Z = -1;
  for(const auto& e: libint2::chemistry::get_element_info()) {
    auto es = element_symbol;
    es.erase(std::remove_if(std::begin(es), std::end(es), [](auto d) { return std::isdigit(d); }),
             es.end());
    if(txt_utils::strequal_case(e.symbol, es)) {
      Z = e.Z;
      break;
    }
  }
  if(Z == -1) {
    tamm_terminate("INPUT FILE ERROR: element symbol \"" + element_symbol + "\" is not recognized");
  }

  return Z;
}