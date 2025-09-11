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
class SCFQed {
public:
  SCFQed() = default;
  // Virtual destructor because this is a polymorphic class
  virtual ~SCFQed() = default;

  SCFQed(const SCFQed&)                = default;
  SCFQed& operator=(const SCFQed&)     = default;
  SCFQed(SCFQed&&) noexcept            = default;
  SCFQed& operator=(SCFQed&&) noexcept = default;

  // Virtual interface functions
  virtual void qed_functionals_setup(std::vector<double>& params, const ChemEnv& chem_env) const;

  virtual void compute_QED_1body(ExecutionContext& ec, const ChemEnv& chem_env,
                                 const SCFData& scf_data, TAMMTensors<T>& ttensors) const;

  virtual void compute_QED_2body(ExecutionContext& ec, const ChemEnv& chem_env,
                                 const SCFData& scf_data, TAMMTensors<T>& ttensors) const;

  virtual void compute_qed_emult_ints(ExecutionContext& ec, const ChemEnv& chem_env,
                                      const SCFData& scf_data, TAMMTensors<T>& ttensors) const;
};

} // namespace exachem::scf
