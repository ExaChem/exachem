/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/integrations/ipi/ipi_backend.hpp"
#include "exachem/gradients/ec_gradients.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

static bool msg_is(std::string_view msg, std::string_view ref) {
  return msg.substr(0, ref.size()) == ref;
}

namespace exachem::integrations::ipi {

void run(const char* host, int port, int inet_mode, std::string sockets_prefix,
         ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
         std::vector<ECAtom>& ec_atoms, std::string ec_arg2) {
  const int rank = ec.pg().rank().value();

  int sockfd = -1;
  if(rank == 0) {
    int iport = port;
    open_socket(&sockfd, &inet_mode, &iport, host, sockets_prefix.c_str());
  }

  bool isinit   = false;
  bool has_data = false;

  int                 natoms = chem_env.atoms.size();
  std::vector<double> coords, forces;
  double              cell[9], icell[9];
  double              virial[9] = {};
  double              energy    = 0.0;

  while(true) {
    char msg[13] = {};

    // Read the next message header. A clean socket close at this point (read
    // returns 0) means the driver is done without sending an explicit EXIT
    // (e.g. ASE closes its SocketIOCalculator), so treat it as a graceful exit
    // rather than a fatal error. Use a sentinel header so all ranks agree.
    if(rank == 0) {
      int n = read(sockfd, msg, 12);
      while(n > 0 && n < 12) {
        int nr = read(sockfd, msg + n, 12 - n);
        if(nr <= 0) {
          n = nr;
          break;
        }
        n += nr;
      }
      if(n <= 0) std::strcpy(msg, "EXIT");
    }
    ec.pg().broadcast(msg, 12, 0);

    // ----------------------------------------------------------------
    if(msg_is(msg, "STATUS")) {
      if(rank == 0) {
        int len = 12;
        if(has_data) writebuffer(&sockfd, "HAVEDATA    ", &len);
        else if(isinit) writebuffer(&sockfd, "READY       ", &len);
        else writebuffer(&sockfd, "NEEDINIT    ", &len);
      }

      // ----------------------------------------------------------------
    }
    else if(msg_is(msg, "INIT")) {
      // Read replica ID and optional init string (just consume them)
      if(rank == 0) {
        int32_t rid;
        int     len4 = sizeof(int32_t);
        readbuffer(&sockfd, (char*) &rid, &len4);

        int32_t str_len;
        readbuffer(&sockfd, (char*) &str_len, &len4);
        if(str_len > 0) {
          std::vector<char> init_str(str_len + 1, 0);
          readbuffer(&sockfd, init_str.data(), &str_len);
        }
      }
      isinit = true;

      // ----------------------------------------------------------------
    }
    else if(msg_is(msg, "POSDATA")) {
      if(rank == 0) {
        int len9d = 9 * sizeof(double);
        readbuffer(&sockfd, (char*) cell, &len9d);
        readbuffer(&sockfd, (char*) icell, &len9d);

        int32_t n;
        int     len4 = sizeof(int32_t);
        readbuffer(&sockfd, (char*) &n, &len4);
        natoms = (int) n;
        coords.resize(3 * natoms);
        int lencoords = 3 * natoms * sizeof(double);
        readbuffer(&sockfd, (char*) coords.data(), &lencoords);
      }

      ec.pg().broadcast(&natoms, 1, 0);
      coords.resize(3 * natoms);
      ec.pg().broadcast(coords.data(), 3 * natoms, 0);

      // --- compute energy and forces for the requested geometry ---
      forces.resize(3 * natoms, 0.0);
      Eigen::Map<Eigen::RowVectorXd> new_geometry(coords.data(), coords.size());
      chem_env.update_geometry(atoms, ec_atoms, new_geometry);
      Matrix gradient_matrix =
        exachem::gradients::ECGradients::compute_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);

      energy = chem_env.get_task_energy(ec, chem_env);
      std::copy(gradient_matrix.data(), gradient_matrix.data() + gradient_matrix.size(),
                forces.data());
      for(auto& f: forces) f = -f;
      // ------------------------

      has_data = true;

      // ----------------------------------------------------------------
    }
    else if(msg_is(msg, "GETFORCE")) {
      if(rank == 0) {
        int len12 = 12;
        writebuffer(&sockfd, "FORCEREADY  ", &len12);

        int len8 = sizeof(double);
        writebuffer(&sockfd, (char*) &energy, &len8);
        int32_t n    = natoms;
        int     len4 = sizeof(int32_t);
        writebuffer(&sockfd, (char*) &n, &len4);
        int lenforces = 3 * natoms * sizeof(double);
        writebuffer(&sockfd, (char*) forces.data(), &lenforces);
        int len9d = 9 * sizeof(double);
        writebuffer(&sockfd, (char*) virial, &len9d);

        int32_t extra = 0;
        writebuffer(&sockfd, (char*) &extra, &len4);
      }

      // Reset state - resets isinit here to re-trigger
      // INIT on the next step (for replica ID tracking).
      // For simple gas-phase MD you can skip that; keep it for
      // correctness with multi-replica runs.
      isinit   = false;
      has_data = false;

      // ----------------------------------------------------------------
    }
    else if(msg_is(msg, "EXIT")) {
      if(rank == 0) std::cout << std::endl << " i-PI driver: received EXIT, done" << std::endl;
      break;
    }
    else {
      // Any unrecognised message means the server is done
      if(rank == 0)
        std::cout << std::endl
                  << " i-PI driver: unknown message '" << msg << "', exiting" << std::endl;
      break;
    }
  }

  if(rank == 0) close(sockfd);
}

} // namespace exachem::integrations::ipi
