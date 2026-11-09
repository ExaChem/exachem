/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/fci/fci.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"
#if defined(USE_MACIS)
#include "exachem/fci/macis.hpp"
#endif

using namespace tamm;

namespace exachem::fci {
template<typename T>
void FCI<T>::driver(ExecutionContext& ec, ChemEnv& chem_env) {
  chem_env.cd_context.keep_movecs_so = true;
  exachem::cholesky_2e::Cholesky_2E_Driver<T> cholesky_driver;
  cholesky_driver.cholesky_2e_driver(ec, chem_env);

  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  chem_env.cc_context.init_filenames(files_prefix);

  const TiledIndexSpace& MO      = chem_env.is_context.MSO;
  const TiledIndexSpace& CI      = chem_env.is_context.CI;
  const TiledIndexSpace  N       = MO("all");
  Tensor<T>              d_f1    = cd_context.d_f1;
  Tensor<T>              cholVpr = cd_context.cholV2;

  ec.pg().barrier();

  const auto [cindex]     = CI.labels<1>("all");
  const auto [p, q, r, s] = MO.labels<4>("all");

  Tensor<T> full_v2{N, N, N, N};
  Tensor<T>::allocate(&ec, full_v2);

  ExecutionHW ex_hw = ec.exhw();

  // clang-format off
  Scheduler sch{ec};
  sch(full_v2(p, r, q, s)  = cholVpr(p, r, cindex) * cholVpr(q, s, cindex)).execute(ex_hw);
  // clang-format on

  free_tensors(cholVpr);

  Tensor<T> lcao = cd_context.movecs_so;

  const TiledIndexSpace& AO = lcao.tiled_index_spaces()[0];
  Tensor<T>              hcore{AO, AO};
  Tensor<T>::allocate(&ec, hcore);
  read_from_disk(ec, hcore, chem_env.get_files_prefix("", "scf") + ".hcore");

  fcidump::generate_fcidump(chem_env, ec, hcore, lcao, full_v2, ex_hw);
  free_tensors(hcore);
#if defined(USE_MACIS)
  if(chem_env.ioptions.task_options.fci)
    macis_driver(ec, chem_env, chem_env.get_files_prefix("", "fci"));
#endif

  free_tensors(lcao, d_f1, full_v2);

  ec.flush_and_sync();
}

// Explicit instantiation for double fci_driver free function usage
void fci_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  FCI<double> fci;
  fci.driver(ec, chem_env);
}
} // namespace exachem::fci
