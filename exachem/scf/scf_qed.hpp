/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <cctype>

#include "exachem/scf/scf_common.hpp"

namespace exachem::scf {

class SCFQed {
public:
  void qed_functionals_setup(std::vector<double>& params, ChemEnv& chem_env);

  template<typename TensorType>
  void compute_QED_1body(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                         TAMMTensors& ttensors);

  template<typename TensorType>
  void compute_QED_2body(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& scf_vars,
                         TAMMTensors& ttensors);

  template<typename TensorType>
  void compute_qed_emult_ints(ExecutionContext& ec, ChemEnv& chem_env, const SCFVars& spvars,
                              TAMMTensors& ttensors);
};
} // namespace exachem::scf
