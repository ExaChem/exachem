/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/canonical/ccsd_canonical.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::cc::ducc {

// Member function implementation
void DUCCDriver::execute(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  CCContext& cc_context      = chem_env.cc_context;
  cc_context.keep.fvt12_full = true;
  cc_context.compute.set(true, true); // compute ft12 and v2 in full

  const bool do_hubbard = chem_env.sys_data.is_hubbard;
  if(do_hubbard) { exachem::cc::ccsd_canonical::ccsd_canonical_driver(ec, chem_env); }
  else { exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env); }

  TiledIndexSpace&           MO        = chem_env.is_context.MSO;
  Tensor<T>                  d_f1      = chem_env.cd_context.d_f1;
  Tensor<T>                  cholVpr   = chem_env.cd_context.cholV2;
  Tensor<T>                  d_t1      = chem_env.cc_context.d_t1_full;
  Tensor<T>                  d_t2      = chem_env.cc_context.d_t2_full;
  cholesky_2e::V2Tensors<T>& v2tensors = chem_env.cd_context.v2tensors;

  if(!do_hubbard) free_tensors(cholVpr);

  IndexVector       occ_int_vec;
  IndexVector       virt_int_vec;
  int               pos;
  std::stringstream qfstr;

  exachem::cc::ducc::internal::DUCCInternal<double> ducc_internal;
  ducc_internal.DUCC_T_CCSD_Driver(chem_env, ec, MO, d_t1, d_t2, d_f1, v2tensors, occ_int_vec,
                                   virt_int_vec, pos, qfstr);

  v2tensors.deallocate();
  free_tensors(d_t1, d_t2, d_f1);

  print_memory_usage<T>(ec.pg().rank().value(), "DUCC Memory Stats");

  ec.flush_and_sync();
}

// Wrapper function for backward compatibility
void ducc_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  DUCCDriver driver;
  driver.execute(ec, chem_env);
}

} // namespace exachem::cc::ducc
