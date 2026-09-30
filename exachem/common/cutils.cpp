/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/cutils.hpp"

// Nbf, % of nodes, % of Nbf, nnodes from input file, (% of nodes, % of nbf) for scalapack
ProcGroupData get_spg_data(ExecutionContext& ec, const size_t N, const int node_p, const int nbf_p,
                           const int node_inp) {
  ProcGroupData pgdata;
  pgdata.ppn = ec.ppn();

  const int ppn    = pgdata.ppn;
  const int nnodes = ec.nnodes();

  int spg_guessranks = std::ceil((nbf_p / 100.0) * N);
  if(node_p > 0) spg_guessranks = std::ceil((node_p / 100.0) * nnodes);
  int spg_nnodes = spg_guessranks / ppn;
  if(spg_guessranks % ppn > 0 || spg_nnodes == 0) spg_nnodes++;
  if(spg_nnodes > nnodes) spg_nnodes = nnodes;
  int spg_nranks = spg_nnodes * ppn;

  int user_nnodes = static_cast<int>(node_inp / 100) * nnodes;
  if(user_nnodes > spg_nnodes) {
    spg_nnodes = user_nnodes;
    spg_nranks = spg_nnodes * ppn;
  }
  pgdata.spg_nnodes = spg_nnodes;
  pgdata.spg_nranks = spg_nranks;

  return pgdata;
}
