/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/eom/eomccsd_opt.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

void exachem::cc::eom::eom_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  CCContext& cc_context      = chem_env.cc_context;
  cc_context.keep.fvt12_full = true;
  cc_context.compute.set(true, true); // compute ft12 and v2 in full
  exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env);

  TiledIndexSpace&           MO        = chem_env.is_context.MSO;
  Tensor<T>                  d_f1      = chem_env.cd_context.d_f1;
  Tensor<T>                  cholVpr   = chem_env.cd_context.cholV2;
  Tensor<T>                  d_t1      = chem_env.cc_context.d_t1_full;
  Tensor<T>                  d_t2      = chem_env.cc_context.d_t2_full;
  cholesky_2e::V2Tensors<T>& v2tensors = chem_env.cd_context.v2tensors;

  free_tensors(cholVpr);

  if(chem_env.ioptions.ccsd_options.eom_nroots <= 0)
    tamm_terminate("EOMCCSD: nroots should be greater than 1");

  std::string eom_type = chem_env.ioptions.ccsd_options.eom_type;

  // EOMCCSD Routine
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  if(eom_type == "right")
    right_eomccsd_driver<T>(chem_env, ec, MO, d_t1, d_t2, d_f1, v2tensors,
                            chem_env.cd_context.p_evl_sorted);

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  auto ccsd_time = std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << std::endl
              << "Time taken for " << eom_type << "-Eigenstate EOMCCSD: " << std::fixed
              << std::setprecision(2) << ccsd_time << " secs" << std::endl;
  }

  v2tensors.deallocate();
  free_tensors(d_t1, d_t2, d_f1);

  ec.flush_and_sync();
}
