/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/fci/fci.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

using namespace tamm;

namespace exachem::fci {

template<typename T>
std::string generate_fcidump(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MSO,
                             Tensor<T>& lcao, Tensor<T>& d_f1, Tensor<T>& full_v2,
                             ExecutionHW ex_hw) {
  // int nactv = chem_env.ioptions.fci_options.nactive;
  Scheduler sch{ec};

  std::cout.precision(15);
  // const auto rank = ec.pg().rank();

  auto [z1, z2] = MSO.labels<2>("all");

  // Transform d_f1 from Fock operator to the one-electron operator representation
  TiledIndexSpace AO = lcao.tiled_index_spaces()[0];
  auto [mu, nu]      = AO.labels<2>("all");

  Tensor<T> hcore{AO, AO};
  Tensor<T> hcore_mo{{MSO, MSO}, {1, 1}};
  Tensor<T>::allocate(&ec, hcore, hcore_mo);

  SystemData& sys_data     = chem_env.sys_data;
  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type + "/fci";
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  std::string hcorefile = files_dir + "/../scf/" + sys_data.output_file_prefix + ".hcore";
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
  std::string fcid_file = files_prefix + ".fcidump";
  fcidump::write_fcidump_file(chem_env, hcore_mo, full_v2, symvec, fcid_file);

  free_tensors(hcore_mo);
  return files_prefix;
}

void fci_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  auto rank = ec.pg().rank();

  scf::scf_driver(ec, chem_env);

  // double              hf_energy      = chem_env.hf_energy;
  libint2::BasisSet   shells         = chem_env.shells;
  Tensor<T>           C_AO           = chem_env.C_AO;
  Tensor<T>           C_beta_AO      = chem_env.C_beta_AO;
  Tensor<T>           F_AO           = chem_env.F_AO;
  Tensor<T>           F_beta_AO      = chem_env.F_beta_AO;
  TiledIndexSpace     AO_opt         = chem_env.AO_opt;
  TiledIndexSpace     AO_tis         = chem_env.AO_tis;
  std::vector<size_t> shell_tile_map = chem_env.shell_tile_map;

  SystemData   sys_data     = chem_env.sys_data;
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = cholesky_2e::setupMOIS(ec, chem_env);

  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";

  ExecutionHW ex_hw = ec.exhw();

  bool ccsd_restart = ccsd_options.readt || (fs::exists(f1file) && fs::exists(v2file));

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cholesky_2e::cholesky_2e_driver<T>(chem_env, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO,
                                       shells, shell_tile_map, ccsd_restart, cholfile);

  TiledIndexSpace N = MO("all");

  if(ccsd_restart) {
    read_from_disk(d_f1, f1file);
    read_from_disk(cholVpr, v2file);
    ec.pg().barrier();
  }

  else if(ccsd_options.writet) {
    // fs::remove_all(files_dir);
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);

    write_to_disk(d_f1, f1file);
    write_to_disk(cholVpr, v2file);

    if(rank == 0) {
      std::ofstream out(cholfile, std::ios::out);
      if(!out) cerr << "Error opening file " << cholfile << endl;
      out << chol_count << std::endl;
      out.close();
    }
  }

  ec.pg().barrier();

  auto [cindex]     = CI.labels<1>("all");
  auto [p, q, r, s] = MO.labels<4>("all");

  Tensor<T> full_v2{N, N, N, N};
  Tensor<T>::allocate(&ec, full_v2);

  // clang-format off
  Scheduler sch{ec};
  sch(full_v2(p, r, q, s)  = cholVpr(p, r, cindex) * cholVpr(q, s, cindex)).execute(ex_hw);
  // clang-format off

  free_tensors(cholVpr);

  files_prefix = generate_fcidump(chem_env, ec, MO, lcao, d_f1, full_v2, ex_hw);
  #if defined(USE_MACIS)
  if(options_map.task_options.fci)
    macis_driver(ec, sys_data, files_prefix);
  #endif
  
  free_tensors(lcao, d_f1, full_v2);

  ec.flush_and_sync();
  // delete ec;
}
} // namespace exachem::fci
