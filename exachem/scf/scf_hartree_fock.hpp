/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/chemenv.hpp"
#include "exachem/common/ec_basis.hpp"
#include "exachem/common/ec_molden.hpp"
#include "exachem/common/system_data.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "exachem/scf/scf_iter.hpp"
#include "exachem/scf/scf_outputs.hpp"
#include "exachem/scf/scf_restart.hpp"
#include "exachem/scf/scf_taskmap.hpp"
#include <variant>

#define SCF_THROTTLE_RESOURCES 1

// using VarType = std::variant<TypeA, TypeB>;

namespace exachem::scf {

class SCFHartreeFock {
private:
  void scf_hf(ExecutionContext& exc, ChemEnv& chem_env);

public:
  SCFHartreeFock() = default;
  SCFHartreeFock(ExecutionContext& exc, ChemEnv& chem_env) { initialize(exc, chem_env); };
  void operator()(ExecutionContext& exc, ChemEnv& chem_env) { initialize(exc, chem_env); };
  void initialize(ExecutionContext& exc, ChemEnv& chem_env);
};
} // namespace exachem::scf
