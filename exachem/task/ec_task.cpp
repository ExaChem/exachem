/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/ec_task.hpp"

namespace exachem::task {

void execute_task(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2) {
  const auto task       = chem_env.ioptions.task_options;
  const auto input_file = chem_env.input_file;

  print_geometry(ec, chem_env);

  if(task.sinfo) chem_env.sinfo();
  else if(task.scf) {
    scf::scf_driver(ec, chem_env);
    Tensor<TensorType>::deallocate(chem_env.scf_context.C_AO, chem_env.scf_context.F_AO);
    if(chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::deallocate(chem_env.scf_context.C_beta_AO,
                                     chem_env.scf_context.F_beta_AO);
  }
#if defined(ENABLE_CC)
  else if(task.mp2) mp2::cd_mp2(ec, chem_env);
  else if(task.cd_2e) cholesky_2e::cholesky_2e_driver(ec, chem_env);
  else if(task.ccsd) { cc::ccsd::cd_ccsd(ec, chem_env); }
  else if(task.ccsd_t) cc::ccsd_t::ccsd_t_driver(ec, chem_env);
  else if(task.cc2) cc2::cd_cc2_driver(ec, chem_env);
  else if(task.ccsd_lambda) cc::ccsd_lambda::ccsd_lambda_driver(ec, chem_env);
  else if(task.eom_ccsd) cc::eom::eom_ccsd_driver(ec, chem_env);
  else if(task.ducc) cc::ducc::ducc_driver(ec, chem_env);
#if defined(EC_COMPLEX)
  else if(task.fci || task.fcidump) fci::fci_driver(ec, chem_env);
  else if(task.gfccsd) cc::gfcc::gfccsd_driver(ec, chem_env);
  else if(task.rteom_ccsd) rteom_cc::ccsd::rt_eom_cd_ccsd_driver(ec, chem_env);
#endif

#endif

  else
    tamm_terminate(
      "\n[ERROR] Unsupported task specified (or) code for the specified task is not built");
}

} // namespace exachem::task
