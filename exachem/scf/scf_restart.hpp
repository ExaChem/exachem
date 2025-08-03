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
using namespace tamm;

namespace exachem::scf {
class DefaultSCFRestart {
public:
  virtual ~DefaultSCFRestart() = default;
  virtual void run(const ExecutionContext& ec, ChemEnv& chem_env, std::string files_prefix);
  virtual void run(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                   TAMMTensors<T>& ttensors, EigenTensors& etensors, std::string files_prefix);
};

class SCFRestart: public DefaultSCFRestart {
public:
  void run(const ExecutionContext& ec, ChemEnv& chem_env, std::string files_prefix) {
    DefaultSCFRestart::run(ec, chem_env, files_prefix);
  }

  void run(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
           TAMMTensors<T>& ttensors, EigenTensors& etensors, std::string files_prefix) override {
    DefaultSCFRestart::run(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix);
  }
};
} // namespace exachem::scf
