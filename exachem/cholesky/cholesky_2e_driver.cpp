/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cholesky/cholesky_2e_driver.hpp"
using namespace exachem::scf;

void exachem::cholesky_2e::cholesky_decomp_2e(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  auto rank = ec.pg().rank();

  scf::scf_driver(ec, chem_env);

  libint2::BasisSet   shells         = chem_env.shells;
  Tensor<T>           C_AO           = chem_env.C_AO;
  Tensor<T>           C_beta_AO      = chem_env.C_beta_AO;
  Tensor<T>           F_AO           = chem_env.F_AO;
  Tensor<T>           F_beta_AO      = chem_env.F_beta_AO;
  TiledIndexSpace     AO_opt         = chem_env.AO_opt;
  TiledIndexSpace     AO_tis         = chem_env.AO_tis;
  std::vector<size_t> shell_tile_map = chem_env.shell_tile_map;

  SystemData  sys_data     = chem_env.sys_data;
  CCSDOptions ccsd_options = chem_env.ioptions.ccsd_options;
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

  bool cd_restart = fs::exists(f1file) && fs::exists(v2file) && fs::exists(cholfile);

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cholesky_2e::cholesky_2e_driver<T>(chem_env, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO,
                                       shells, shell_tile_map, cd_restart, cholfile);
  free_tensors(lcao);

  if(!cd_restart && ccsd_options.writet) {
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

  free_tensors(d_f1, cholVpr);

  ec.flush_and_sync();
  // delete ec;
} // End of cholesky_decomp_2e

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
exachem::cholesky_2e::cholesky_2e_driver(ChemEnv& chem_env, ExecutionContext& ec,
                                         TiledIndexSpace& MO, TiledIndexSpace& AO, Tensor<T> C_AO,
                                         Tensor<T> F_AO, Tensor<T> C_beta_AO, Tensor<T> F_beta_AO,
                                         libint2::BasisSet&   shells,
                                         std::vector<size_t>& shell_tile_map, bool readv2,
                                         std::string cholfile, bool is_dlpno, bool is_mso) {
  SystemData& sys_data        = chem_env.sys_data;
  CDOptions   cd_options      = chem_env.ioptions.cd_options;
  auto        diagtol         = cd_options.diagtol; // tolerance for the max. diagonal
  cd_options.max_cvecs_factor = 2 * std::abs(std::log10(diagtol));
  // TODO
  tamm::Tile max_cvecs = cd_options.max_cvecs_factor * sys_data.nbf;

  std::cout << std::defaultfloat;
  auto rank = ec.pg().rank();
  if(rank == 0) cd_options.print();

  TiledIndexSpace N = MO("all");

  Tensor<T> d_f1{{N, N}, {1, 1}};
  Tensor<T> lcao{AO, N};
  Tensor<T>::allocate(&ec, d_f1, lcao);

  auto      hf_t1      = std::chrono::high_resolution_clock::now();
  TAMM_SIZE chol_count = 0;

  // std::tie(V2) =
  Tensor<T> cholVpr;
  auto      itile_size = chem_env.ioptions.cd_options.itilesize;

  sys_data.freeze_atomic    = chem_env.ioptions.ccsd_options.freeze_atomic;
  sys_data.n_frozen_core    = chem_env.get_nfcore();
  sys_data.n_frozen_virtual = chem_env.ioptions.ccsd_options.freeze_virtual;
  bool do_freeze            = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;

  std::string out_fp    = chem_env.workspace_dir;
  std::string files_dir = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string lcaofile  = files_dir + "/" + sys_data.output_file_prefix + ".lcao";

  auto skip_cd = cd_options.skip_cd;

  if(!readv2 && !skip_cd.first) {
    exachem::cholesky_2e::two_index_transform(chem_env, ec, C_AO, F_AO, C_beta_AO, F_beta_AO, d_f1,
                                              shells, lcao, is_dlpno || !is_mso);
    if(!is_dlpno)
      cholVpr = exachem::cholesky_2e::cholesky_2e(chem_env, ec, MO, AO, chol_count, max_cvecs,
                                                  shells, lcao, is_mso);
    write_to_disk<TensorType>(lcao, lcaofile);
  }
  else {
    if(!skip_cd.first) {
      std::ifstream in(cholfile, std::ios::in);
      int           rstatus = 0;
      if(in.is_open()) rstatus = 1;
      if(rstatus == 1) in >> chol_count;
      else tamm_terminate("Error reading " + cholfile);

      if(rank == 0)
        cout << "Number of cholesky vectors to be read from disk = " << chol_count << endl;
    }
    else {
      chol_count = skip_cd.second;
      if(rank == 0)
        cout << endl
             << "Skipping Cholesky Decomposition... using user provided cholesky vector count of "
             << chol_count << endl
             << endl;
    }

    if(!is_dlpno) exachem::cholesky_2e::update_sysdata(ec, chem_env, MO, is_mso);

    IndexSpace      chol_is{range(0, chol_count)};
    TiledIndexSpace CI{chol_is, static_cast<tamm::Tile>(itile_size)};

    TiledIndexSpace N = MO("all");
    cholVpr = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
    if(!is_dlpno) Tensor<TensorType>::allocate(&ec, cholVpr);
    // Scheduler{ec}(cholVpr()=0).execute();
    if(!skip_cd.first) read_from_disk(lcao, lcaofile);
  }

  auto   hf_t2 = std::chrono::high_resolution_clock::now();
  double cholesky_2e_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  if(rank == 0 && !skip_cd.first)
    std::cout << std::endl
              << "Total Time taken for Cholesky Decomposition: " << std::fixed
              << std::setprecision(2) << cholesky_2e_time << " secs" << std::endl
              << std::endl;

  Tensor<T>::deallocate(C_AO, F_AO);
  if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO, F_beta_AO);

  IndexSpace      chol_is{range(0, chol_count)};
  TiledIndexSpace CI{chol_is, static_cast<tamm::Tile>(itile_size)};

  chem_env.cd_context.num_chol_vectors                   = chol_count;
  sys_data.results["output"]["CD"]["n_cholesky_vectors"] = chol_count;

  sys_data.results["output"]["CD"]["diagtol"] = chem_env.ioptions.cd_options.diagtol;

  if(rank == 0) sys_data.print();

  if(do_freeze) {
    TiledIndexSpace N_eff = MO("all");
    Tensor<T>       d_f1_new{{N_eff, N_eff}, {1, 1}};
    Tensor<T>::allocate(&ec, d_f1_new);
    if(rank == 0) {
      Matrix f1_eig     = tamm_to_eigen_matrix(d_f1);
      Matrix f1_new_eig = exachem::cholesky_2e::reshape_mo_matrix(chem_env, f1_eig);
      eigen_to_tamm_tensor(d_f1_new, f1_new_eig);
      f1_new_eig.resize(0, 0);
    }
    Tensor<T>::deallocate(d_f1);
    d_f1 = d_f1_new;
  }

  if(!readv2 && chem_env.ioptions.scf_options.mos_txt) {
    Scheduler   sch{ec};
    std::string hcorefile = files_dir + "/scf/" + sys_data.output_file_prefix + ".hcore";
    Tensor<T>   hcore{AO, AO};
    Tensor<T>   hcore_mo{MO, MO};
    Tensor<T>::allocate(&ec, hcore, hcore_mo);
    read_from_disk(hcore, hcorefile);

    auto [mu, nu]   = AO.labels<2>("all");
    auto [mo1, mo2] = MO.labels<2>("all");

    Tensor<T> tmp{MO, AO};
    // clang-format off
    sch.allocate(tmp)
        (tmp(mo1,nu) = lcao(mu,mo1) * hcore(mu,nu))
        (hcore_mo(mo1,mo2) = tmp(mo1,nu) * lcao(nu,mo2))
        .deallocate(tmp,hcore).execute();
    // clang-format on

    ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
    std::string      mop_dir   = files_dir + "/mos_txt/";
    std::string      mofprefix = mop_dir + out_fp;
    if(!fs::exists(mop_dir)) fs::create_directories(mop_dir);

    Tensor<T> d_v2        = setupV2<T>(ec, MO, CI, cholVpr, chol_count);
    Tensor<T> d_f1_dense  = to_dense_tensor(ec_dense, d_f1);
    Tensor<T> lcao_dense  = to_dense_tensor(ec_dense, lcao);
    Tensor<T> d_v2_dense  = to_dense_tensor(ec_dense, d_v2);
    Tensor<T> hcore_dense = to_dense_tensor(ec_dense, hcore_mo);

    Tensor<T>::deallocate(hcore_mo, d_v2);

    print_dense_tensor(d_v2_dense, mofprefix + ".v2_mo");
    print_dense_tensor(lcao_dense, mofprefix + ".ao2mo");
    print_dense_tensor(d_f1_dense, mofprefix + ".fock_mo");
    print_dense_tensor(hcore_dense, mofprefix + ".hcore_mo");

    Tensor<T>::deallocate(hcore_dense, d_f1_dense, lcao_dense, d_v2_dense);
  }

  return std::make_tuple(cholVpr, d_f1, lcao, chol_count, max_cvecs, CI);
} // END of cholesky_2e_driver

using T = double;

template std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
exachem::cholesky_2e::cholesky_2e_driver<T>(ChemEnv& chem_env, ExecutionContext& ec,
                                            TiledIndexSpace& MO, TiledIndexSpace& AO,
                                            Tensor<T> C_AO, Tensor<T> F_AO, Tensor<T> C_beta_AO,
                                            Tensor<T> F_beta_AO, libint2::BasisSet& shells,
                                            std::vector<size_t>& shell_tile_map, bool readv2,
                                            std::string cholfile, bool is_dlpno, bool is_mso);
