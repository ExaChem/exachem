#pragma once

#include "common/chemenv.hpp"
#include "common/cutils.hpp"
#include "common/ec_molden.hpp"
#include "scf/scf_eigen_tensors.hpp"
#include "scf/scf_outputs.hpp"
#include "scf/scf_tamm_tensors.hpp"
using namespace tamm;

namespace exachem::scf {
class SCFRestart: public SCFIO {
public:
  void operator()(const ExecutionContext& ec, ChemEnv& chem_env, std::string files_prefix);
  void operator()(ExecutionContext& ec, ChemEnv& chem_env, ScalapackInfo& scalapack_info,
                  TAMMTensors& ttensors, EigenTensors& etensors, std::string files_prefix);
};
} // namespace exachem::scf
