/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_restart.hpp"
#include "exachem/scf/scf_matrix.hpp"

// Originally scf_restart_test
void exachem::scf::SCFRestart::operator()(const ExecutionContext& ec, ChemEnv& chem_env,
                                          std::string files_prefix) {
  bool restart = chem_env.ioptions.scf_options.restart || chem_env.ioptions.scf_options.noscf;

  if(!restart) return;
  const auto rank   = ec.pg().rank();
  const bool is_uhf = (chem_env.sys_data.is_unrestricted);

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";
  std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  std::string densityfile_beta  = files_prefix + ".beta.density";
  bool        status            = false;

  if(rank == 0) {
    status = fs::exists(movecsfile_alpha) && fs::exists(densityfile_alpha);
    if(is_uhf) status = status && fs::exists(movecsfile_beta) && fs::exists(densityfile_beta);
  }
  ec.pg().barrier();
  ec.pg().broadcast(&status, 0);
  std::string fnf = movecsfile_alpha + "; " + densityfile_alpha;
  if(is_uhf) fnf = fnf + "; " + movecsfile_beta + "; " + densityfile_beta;
  if(!status) tamm_terminate("Error reading one or all of the files: [" + fnf + "]");
}

void exachem::scf::SCFRestart::operator()(ExecutionContext& ec, ChemEnv& chem_env,
                                          ScalapackInfo& scalapack_info, TAMMTensors& ttensors,
                                          EigenTensors& etensors, std::string files_prefix) {
  const auto N      = chem_env.sys_data.nbf_orig;
  const auto Northo = N - chem_env.sys_data.n_lindep;
  EXPECTS(Northo == chem_env.sys_data.nbf);

  SCFIO::rw_md_disk(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix, true);
}
