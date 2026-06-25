/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <string>
#include <vector>

#include "exachem/common/chemenv.hpp"

namespace exachem::integrations::ase {

/**
 * Run ExaChem as the compute backend for ASE (Atomic Simulation Environment).
 *
 * ASE's SocketIOCalculator speaks the i-PI protocol over a UNIX domain socket,
 * so this hands off to the i-PI backend with the UNIX-socket configuration ASE
 * expects.
 */
void run(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
         std::vector<ECAtom>& ec_atoms, std::string ec_arg2);

} // namespace exachem::integrations::ase
