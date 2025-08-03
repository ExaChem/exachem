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

template<typename T>
class DefaultSCFQed {
public:
  virtual ~DefaultSCFQed() = default;
  virtual void qed_functionals_setup(std::vector<double>& params, ChemEnv& chem_env);

  virtual void compute_QED_1body(ExecutionContext& ec, ChemEnv& chem_env, const SCFData& scf_data,
                                 TAMMTensors<T>& ttensors);

  virtual void compute_QED_2body(ExecutionContext& ec, ChemEnv& chem_env, const SCFData& scf_data,
                                 TAMMTensors<T>& ttensors);

  virtual void compute_qed_emult_ints(ExecutionContext& ec, ChemEnv& chem_env,
                                      const SCFData& spvars, TAMMTensors<T>& ttensors);
};

template<typename T>
class SCFQed: public DefaultSCFQed<T> {
public:
  using DefaultSCFQed<T>::qed_functionals_setup;
  using DefaultSCFQed<T>::compute_QED_1body;
  using DefaultSCFQed<T>::compute_QED_2body;
  using DefaultSCFQed<T>::compute_qed_emult_ints;
};
} // namespace exachem::scf
