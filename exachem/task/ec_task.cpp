/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/ec_task.hpp"

namespace exachem::task {

void check_task_options(ExecutionContext& ec, ChemEnv& chem_env) {
  // if user sets it to empty vector
  if(chem_env.ioptions.task_options.operation.empty())
    chem_env.ioptions.task_options.operation = {"energy"};

  const TaskOptions& task = chem_env.ioptions.task_options;

  for(auto x: task.operation) {
    if(!(txt_utils::strequal_case(x, "energy") || txt_utils::strequal_case(x, "gradient") ||
         txt_utils::strequal_case(x, "optimize"))) {
      tamm_terminate("ERROR: unsupported task operation [" + x + "] specified");
    }
  }

  const std::vector<bool> tvec = {task.sinfo,
                                  task.scf,
                                  task.fci,
                                  task.fcidump,
                                  task.mp2,
                                  task.gw,
                                  task.cd_2e,
                                  task.cc2,
                                  task.ducc,
                                  task.ccsd,
                                  task.ccsdt,
                                  task.ccsd_t,
                                  task.ccsd_lambda,
                                  task.eom_ccsd,
                                  task.gfccsd,
                                  task.rteom_cc2,
                                  task.rteom_ccsd,
                                  task.dlpno_ccsd.first,
                                  task.dlpno_ccsd_t.first};
  if(std::count(tvec.begin(), tvec.end(), true) > 1)
    tamm_terminate("[INPUT FILE ERROR] only a single task can be enabled at once!");

  const std::vector<bool> tvec_nograd = {
    task.sinfo,     task.gw,         task.fci,         task.cd_2e,    task.ducc,  task.fcidump,
    task.rteom_cc2, task.rteom_ccsd, task.ccsd_lambda, task.eom_ccsd, task.gfccsd};
  const std::vector<std::string> task_nograd = {
    "sinfo",     "gw",         "fci",         "cd_2e",    "ducc",  "fcidump",
    "rteom_cc2", "rteom_ccsd", "ccsd_lambda", "eom_ccsd", "gfccsd"};

  if(task.operation[0] == "gradient") {
    for(size_t i = 0; i < tvec_nograd.size(); ++i) {
      if(tvec_nograd[i])
        tamm_terminate("[INPUT FILE ERROR] gradients not defined for task [" + task_nograd[i] +
                       "]!");
    }
  }

  // TODO: avoid redefining. originally defined in task options parser
  const std::vector<string> valid_tasks{
    "sinfo",  "scf",       "fci",        "fcidump",    "mp2",         "gw",          "cd_2e",
    "cc2",    "ducc",      "ccsd",       "ccsdt",      "ccsd_t",      "ccsd_lambda", "eom_ccsd",
    "gfccsd", "rteom_cc2", "rteom_ccsd", "dlpno_ccsd", "dlpno_ccsd_t"};

  for(size_t i = 0; i < tvec.size(); ++i) {
    if(tvec[i]) {
      std::string task_string = valid_tasks[i];
      txt_utils::to_upper(task_string);
      chem_env.task_string = task_string;
    }
  }

#if !defined(USE_MACIS)
  if(task.fci) tamm_terminate("Full CI integration not enabled!");
#endif
}

void execute_task(ExecutionContext& ec, ChemEnv& chem_env, std::string ec_arg2) {
  const auto task       = chem_env.ioptions.task_options;
  const auto input_file = chem_env.input_file;

  // TODO: This is redundant if multiple tasks for same geometry are executed.
  SCFOptions& scf_options   = chem_env.ioptions.scf_options;
  chem_env.ec_basis         = ECBasis(ec, scf_options.basis, scf_options.basisfile,
                                      scf_options.gaussian_type, chem_env.atoms, chem_env.ec_atoms);
  chem_env.shells           = chem_env.ec_basis.shells;
  chem_env.sys_data.has_ecp = chem_env.ec_basis.has_ecp;

  if(chem_env.atoms.size() <= 30) exachem::task::geometry_analysis(ec, chem_env);

  print_geometry(ec, chem_env);

  // Check task options. Needed when multiple tasks are supported
  check_task_options(ec, chem_env);

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
  else if(task.ccsd) { cc::ccsd::cd_ccsd_driver(ec, chem_env); }
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
