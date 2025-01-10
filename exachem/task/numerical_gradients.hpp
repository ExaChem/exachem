/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/ec_task.hpp"

namespace exachem::task {

Matrix compute_numerical_gradients(ExecutionContext& ec, ChemEnv& chem_env,
                                   const std::vector<Atom>&   atoms,
                                   const std::vector<ECAtom>& ec_atoms, const std::string ec_arg2);
Matrix compute_gradients(ExecutionContext& ec, ChemEnv& chem_env, const std::vector<Atom>& atoms,
                         const std::vector<ECAtom>& ec_atoms, const std::string ec_arg2);

double compute_energy(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2);

} // namespace exachem::task
