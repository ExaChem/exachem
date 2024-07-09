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
#include "exachem/scf/scf_eigen_tensors.hpp"
#include "exachem/scf/scf_outputs.hpp"
#include "exachem/scf/scf_tamm_tensors.hpp"
using namespace tamm;

namespace exachem::scf {
class SCFRestart: public SCFIO {
public:
  void operator()(const ExecutionContext& ec, ChemEnv& chem_env, std::string files_prefix);
  void operator()(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                  TAMMTensors& ttensors, EigenTensors& etensors, std::string files_prefix);
};
} // namespace exachem::scf
