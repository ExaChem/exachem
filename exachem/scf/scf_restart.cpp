/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_restart.hpp"

// Originally scf_restart_test
void exachem::scf::DefaultSCFRestart::run(const ExecutionContext& ec, ChemEnv& chem_env,
                                          std::string files_prefix) {
  const bool restart = chem_env.ioptions.scf_options.restart || chem_env.ioptions.scf_options.noscf;
  const auto rank    = ec.pg().rank();

  if(restart) {
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
    if(!status)
      tamm_terminate("\n [SCF restart] Error reading one or all of the files: [" + fnf + "]");

    const int orig_ts = chem_env.run_context["ao_tilesize"];
    if(orig_ts != chem_env.is_context.ao_tilesize) {
      std::string err_msg =
        "\n[ERROR] Restarting a calculation requires the AO tilesize to be the same\n";
      err_msg += "  - current tilesize (" + std::to_string(chem_env.is_context.ao_tilesize);
      err_msg += ") does not match the original tilesize (" + std::to_string(orig_ts) + ")";
      tamm_terminate(err_msg);
    }
  }
  else {
    chem_env.run_context["ao_tilesize"] = chem_env.is_context.ao_tilesize;
    if(chem_env.scf_context.do_df)
      chem_env.run_context["dfao_tilesize"] = chem_env.is_context.dfao_tilesize;
    if(rank == 0)
      chem_env.write_run_context(); // write here as well in case we kill an SCF run midway
  }
}

void exachem::scf::DefaultSCFRestart::run(ExecutionContext& ec, ChemEnv& chem_env,
                                          ScalapackInfo& scalapack_info, TAMMTensors<T>& ttensors,
                                          EigenTensors& etensors, std::string files_prefix) {
  const auto N      = chem_env.sys_data.nbf_orig;
  const auto Northo = N - chem_env.sys_data.n_lindep;
  EXPECTS(Northo == chem_env.sys_data.nbf);
  SCFIO<TensorType> scf_io;
  scf_io.rw_md_disk(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix, true);
}
