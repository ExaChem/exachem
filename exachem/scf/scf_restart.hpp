/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/cutils.hpp"
#include "exachem/common/ec_molden.hpp"
#include "exachem/scf/scf_outputs.hpp"
#include "exachem/scf/scf_tensors.hpp"

namespace exachem::scf {
template<typename T>
class SCFRestart {
public:
  SCFRestart()          = default;
  virtual ~SCFRestart() = default;

  SCFRestart(const SCFRestart&)            = default;
  SCFRestart& operator=(const SCFRestart&) = default;

  SCFRestart(SCFRestart&&) noexcept            = default;
  SCFRestart& operator=(SCFRestart&&) noexcept = default;

  virtual void run(const ExecutionContext& ec, ChemEnv& chem_env,
                   const std::string& files_prefix) const;
  virtual void run(ExecutionContext& ec, const ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                   TAMMTensors<T>& ttensors, EigenTensors& etensors,
                   const std::string& files_prefix) const;
};
} // namespace exachem::scf
