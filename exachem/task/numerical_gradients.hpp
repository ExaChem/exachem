/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/ec_task.hpp"

namespace exachem::task {

Matrix compute_numerical_gradients(ExecutionContext& ec, ChemEnv& chem_env,
                                   std::vector<Atom>& atoms, std::vector<ECAtom>& ec_atoms,
                                   std::string ec_arg2);
void   compute_gradients(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                         std::vector<ECAtom>& ec_atoms, std::string ec_arg2);

double compute_energy(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2);

} // namespace exachem::task
