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
// Implementations for template class FCI
template<typename T>
std::string FCI<T>::generate_fcidump(ChemEnv& chem_env, ExecutionContext& ec,
                                     const TiledIndexSpace& MSO, Tensor<T>& lcao, Tensor<T>& d_f1,
                                     Tensor<T>& full_v2, ExecutionHW ex_hw) {
  // int nactv = chem_env.ioptions.fci_options.nactive;
  Scheduler sch{ec};

  std::cout.precision(15);

  const auto [z1, z2] = MSO.labels<2>("all");

  // Transform d_f1 from Fock operator to the one-electron operator representation
  TiledIndexSpace AO  = lcao.tiled_index_spaces()[0];
  const auto [mu, nu] = AO.labels<2>("all");

  Tensor<T> hcore{AO, AO};
  Tensor<T> hcore_mo{{MSO, MSO}, {1, 1}};
  Tensor<T>::allocate(&ec, hcore, hcore_mo);

  SystemData&       sys_data     = chem_env.sys_data;
  const std::string out_fp       = chem_env.workspace_dir;
  const std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type + "/fci";
  const std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  const std::string hcorefile = files_dir + "/../scf/" + sys_data.output_file_prefix + ".hcore";
  read_from_disk(hcore, hcorefile);

  Tensor<T> tmp{MSO, AO};
  // clang-format off
  sch.allocate(tmp)
    (tmp(z1,nu) = lcao(mu,z1) * hcore(mu,nu))
    (hcore_mo(z1,z2) = tmp(z1,nu) * lcao(nu,z2))
    .deallocate(tmp,hcore).execute();
  // clang-format on

  std::vector<int> symvec(sys_data.nbf_orig);
  if(sys_data.is_unrestricted) symvec.resize(2 * sys_data.nbf_orig);
  std::fill(symvec.begin(), symvec.end(), 1);

  // write fcidump file
  const std::string fcid_file = files_prefix + ".fcidump";
  fcidump::write_fcidump_file(chem_env, hcore_mo, full_v2, symvec, fcid_file);

  free_tensors(hcore_mo);
  return files_prefix;
}

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

  const std::string files = generate_fcidump(chem_env, ec, MO, lcao, d_f1, full_v2, ex_hw);
#if defined(USE_MACIS)
  if(chem_env.ioptions.task_options.fci) macis_driver(ec, chem_env, files);
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
