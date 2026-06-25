/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/integrations/ase/ase_backend.hpp"
#include "exachem/integrations/ipi/ipi_backend.hpp"

namespace exachem::integrations::ase {

void run(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
         std::vector<ECAtom>& ec_atoms, std::string ec_arg2) {
  // Connect to ASE over a UNIX domain socket. The ASE SocketIOCalculator on the
  // other end must be configured to listen on this same path.
  exachem::integrations::ipi::run("ase_exachem_socket", 0, 0, "/tmp/ipi_", ec, chem_env, atoms,
                                  ec_atoms, ec_arg2);
}

} // namespace exachem::integrations::ase
