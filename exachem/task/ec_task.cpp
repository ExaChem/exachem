/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/ec_task.hpp"

namespace exachem::task {

void ec_execute_task(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2) {
  const auto task       = chem_env.ioptions.task_options;
  const auto input_file = chem_env.input_file;

  if(ec.print()) {
    std::cout << std::endl << std::string(60, '-') << std::endl;
    for(auto ecatom: chem_env.ec_atoms) {
      std::cout << std::setw(3) << std::left << ecatom.esymbol << " " << std::right << std::setw(14)
                << std::fixed << std::setprecision(10) << ecatom.atom.x << " " << std::right
                << std::setw(14) << std::fixed << std::setprecision(10) << ecatom.atom.y << " "
                << std::right << std::setw(14) << std::fixed << std::setprecision(10)
                << ecatom.atom.z << std::endl;
    }
  }

  if(task.sinfo) chem_env.sinfo();
  else if(task.scf) {
    scf::scf_driver(ec, chem_env);
    Tensor<TensorType>::deallocate(chem_env.C_AO, chem_env.F_AO);
    if(chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::deallocate(chem_env.C_beta_AO, chem_env.F_beta_AO);
  }
#if defined(ENABLE_CC)
  else if(task.mp2) mp2::cd_mp2(ec, chem_env);
  else if(task.cd_2e) cholesky_2e::cholesky_decomp_2e(ec, chem_env);
  else if(task.ccsd) { cc::ccsd::cd_ccsd(ec, chem_env); }
  else if(task.ccsd_t) cc::ccsd_t::ccsd_t_driver(ec, chem_env);
  else if(task.cc2) cc2::cd_cc2_driver(ec, chem_env);
  else if(task.ccsd_lambda) cc::ccsd_lambda::ccsd_lambda_driver(ec, chem_env);
  else if(task.eom_ccsd) cc::eom::eom_ccsd_driver(ec, chem_env);
  else if(task.ducc) cc::ducc::ducc_driver(ec, chem_env);
#if !defined(USE_UPCXX) and defined(EC_COMPLEX)
  else if(task.fci || task.fcidump) fci::fci_driver(ec, chem_env);
  else if(task.gfccsd) cc::gfcc::gfccsd_driver(ec, chem_env);
  else if(task.rteom_ccsd) rteom_cc::ccsd::rt_eom_cd_ccsd_driver(ec, chem_env);
#endif

#endif

  else
    tamm_terminate(
      "[ERROR] Unsupported task specified (or) code for the specified task is not built");
}

} // namespace exachem::task