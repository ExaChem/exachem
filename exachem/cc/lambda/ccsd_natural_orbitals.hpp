/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"
#include <tamm/tamm.hpp>

using namespace tamm;

namespace exachem::cc::ccsd_lambda {

template<typename T>
class CCSD_Natural_Orbitals {
public:
  // Constructor
  CCSD_Natural_Orbitals() = default;

  // Destructor
  virtual ~CCSD_Natural_Orbitals() = default;

  // Copy constructor and assignment
  CCSD_Natural_Orbitals(const CCSD_Natural_Orbitals&)            = default;
  CCSD_Natural_Orbitals& operator=(const CCSD_Natural_Orbitals&) = default;
  CCSD_Natural_Orbitals(CCSD_Natural_Orbitals&&)                 = default;
  CCSD_Natural_Orbitals& operator=(CCSD_Natural_Orbitals&&)      = default;

  /**
   * @brief Compute CCSD natural orbitals from 1-RDM
   */
  virtual void ccsd_natural_orbitals(ChemEnv& chem_env, std::vector<int>& cc_rdm,
                                     std::string files_prefix, std::string files_dir,
                                     Scheduler& sch, ExecutionContext& ec, TiledIndexSpace& MO,
                                     TiledIndexSpace& AO_opt, Tensor<T>& gamma1,
                                     ExecutionHW ex_hw = ExecutionHW::CPU);
};

// Explicit instantiation declaration (definition in .cpp) - prevents implicit instantiation
extern template class CCSD_Natural_Orbitals<double>;

} // namespace exachem::cc::ccsd_lambda
