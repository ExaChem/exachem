/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/chemenv.hpp"
#include "exachem/common/cutils.hpp"
#include "exachem/common/system_data.hpp"
#include "exachem/scf/scf_iter.hpp"
#include "exachem/scf/scf_outputs.hpp"

namespace exachem::scf {

class SCFGradients {
public:
  // Public constructors so this engine can be instantiated directly
  SCFGradients() = default;
  SCFGradients(ExecutionContext& exc, ChemEnv& chem_env);

  virtual ~SCFGradients() = default;

  // Delete copy operations
  SCFGradients(const SCFGradients&)            = delete;
  SCFGradients& operator=(const SCFGradients&) = delete;

  // Allow move operations
  SCFGradients(SCFGradients&&) noexcept            = default;
  SCFGradients& operator=(SCFGradients&&) noexcept = default;

  virtual void scf_gradients(ExecutionContext& exc, ChemEnv& chem_env, Matrix& SchwarzK,
                             SCFData& scf_data, ScalapackInfo& scalapack_info,
                             GauXC::XCIntegrator<Matrix>& xc_integrator);

protected:
  SCFIter<TensorType>  scf_iter;
  SCFGuess<TensorType> scf_guess;
};

} // namespace exachem::scf
