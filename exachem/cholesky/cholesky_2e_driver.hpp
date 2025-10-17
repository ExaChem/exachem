/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccse_tensors.hpp"
#include "exachem/cc/diis.hpp"
#include "exachem/cholesky/v2tensors.hpp"
#include "exachem/scf/scf_main.hpp"

namespace exachem::cholesky_2e {

template<typename T>
class Cholesky_2E_Driver {
public:
  Cholesky_2E_Driver()          = default;
  virtual ~Cholesky_2E_Driver() = default;

  // Copy constructor and copy assignment operator
  Cholesky_2E_Driver(const Cholesky_2E_Driver&)            = default;
  Cholesky_2E_Driver& operator=(const Cholesky_2E_Driver&) = default;

  // Move constructor and move assignment operator
  Cholesky_2E_Driver(Cholesky_2E_Driver&&)            = default;
  Cholesky_2E_Driver& operator=(Cholesky_2E_Driver&&) = default;

  virtual void cholesky_2e_driver(ExecutionContext& ec, ChemEnv& chem_env);
};

// Backward compatibility wrapper function
void cholesky_2e_driver(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::cholesky_2e
