/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/fcidump.hpp"

namespace exachem::fci {
template<typename T>
class FCI {
public:
  FCI()          = default;
  virtual ~FCI() = default;

  // Copy ctor / assignment
  FCI(const FCI&)            = default;
  FCI& operator=(const FCI&) = default;

  // Move ctor / assignment
  FCI(FCI&&) noexcept            = default;
  FCI& operator=(FCI&&) noexcept = default;

  // generate fcidump and return files prefix
  virtual std::string generate_fcidump(ChemEnv& chem_env, tamm::ExecutionContext& ec,
                                       const TiledIndexSpace& MSO, tamm::Tensor<T>& lcao,
                                       tamm::Tensor<T>& d_f1, tamm::Tensor<T>& full_v2,
                                       ExecutionHW ex_hw = ExecutionHW::CPU);

  // top-level driver
  virtual void driver(tamm::ExecutionContext& ec, ChemEnv& chem_env);
};

// Backward-compatible free functions that forward to the template class
template<typename T>
std::string generate_fcidump(ChemEnv& chem_env, tamm::ExecutionContext& ec,
                             const TiledIndexSpace& MSO, tamm::Tensor<T>& lcao,
                             tamm::Tensor<T>& d_f1, tamm::Tensor<T>& full_v2,
                             ExecutionHW ex_hw = ExecutionHW::CPU);

// backward-compatible non-templated declaration (for old callers)
void fci_driver(tamm::ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::fci
