#pragma once
#include "common/chemenv.hpp"
#include "common/ec_basis.hpp"
#include "common/ec_molden.hpp"
#include "common/system_data.hpp"
#include "scf_compute.hpp"
#include "scf_iter.hpp"
#include "scf_outputs.hpp"
#include "scf_restart.hpp"
#include "scf_taskmap.hpp"
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
