/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/numerical_gradients.hpp"

namespace exachem::task {

void geometry_optimizer(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                        std::vector<ECAtom>& ec_atoms, std::string ec_arg2);

} // namespace exachem::task