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
#include "exachem/integrations/ipi/sockets.hpp"

namespace exachem::integrations::ipi {

/**
 * Run ExaChem as the compute backend for an i-PI server: connect over a socket
 * and service its STATUS/INIT/POSDATA/GETFORCE/EXIT message loop, returning the
 * energy and forces for each requested geometry.
 *
 * @param host           Hostname (INET) or socket name (UNIX).
 * @param port           Port number (INET mode only).
 * @param inet_mode      >0 for an INET socket, 0 for a UNIX domain socket.
 * @param sockets_prefix Prefix path for UNIX sockets.
 */
void run(const char* host, int port, int inet_mode, std::string sockets_prefix,
         ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
         std::vector<ECAtom>& ec_atoms, std::string ec_arg2);

} // namespace exachem::integrations::ipi
