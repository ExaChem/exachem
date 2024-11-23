/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/context/cd_context.hpp"
#include "exachem/common/context/is_context.hpp"
using namespace exachem::scf;

void exachem::cholesky_2e::cholesky_2e_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  if(!chem_env.scf_context.scf_converged) scf::scf_driver(ec, chem_env);

  libint2::BasisSet& shells    = chem_env.shells;
  Tensor<T>          C_AO      = chem_env.scf_context.C_AO;
  Tensor<T>          C_beta_AO = chem_env.scf_context.C_beta_AO;
  Tensor<T>          F_AO      = chem_env.scf_context.F_AO;
  Tensor<T>          F_beta_AO = chem_env.scf_context.F_beta_AO;

  TiledIndexSpace& AO = chem_env.is_context.AO_opt;

  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  std::cout << std::defaultfloat;
  auto        rank       = ec.pg().rank();
  SystemData& sys_data   = chem_env.sys_data;
  CDOptions   cd_options = chem_env.ioptions.cd_options;

  if(rank == 0) {
    // ccsd_options.print();
    cd_options.print();
    std::cout << std::endl
              << "- #occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << std::endl;
  }

  std::string files_dir    = chem_env.get_files_dir();
  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  cd_context.init_filenames(files_prefix);

  const std::string cholfile = cd_context.cv_count_file;
  bool              is_dlpno = cd_context.is_dlpno;
  bool              is_mso   = cd_context.is_mso;

  // TODO: MO here is MSO, rename MO to MSO
  TiledIndexSpace MO;
  TAMM_SIZE       total_orbitals;
  if(ccsd_options.nactive > 0) {
    // Only DUCC uses this right now
    std::tie(MO, total_orbitals) =
      cholesky_2e::setupMOIS(ec, chem_env, false, ccsd_options.nactive);
    // TODO: Implement check for UHF
    if(ccsd_options.nactive > sys_data.n_vir_alpha && sys_data.is_restricted)
      tamm_terminate("[DUCC ERROR]: nactive > n_vir_alpha");
  }
  else std::tie(MO, total_orbitals) = cholesky_2e::setupMOIS(ec, chem_env);
  chem_env.is_context.MSO = MO;

  if(ccsd_options.skip_ccsd) {
    Tensor<T>::deallocate(C_AO, F_AO);
    if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO, F_beta_AO);
    return;
  }

  bool&      readv2    = cd_context.readv2;
  const bool fmv_exist = (fs::exists(cd_context.f1file) && fs::exists(cd_context.v2file) &&
                          fs::exists(cd_context.movecs_so_file));
  readv2               = (ccsd_options.readt || ccsd_options.writet) && fmv_exist;

  sys_data.freeze_atomic    = ccsd_options.freeze_atomic;
  sys_data.n_frozen_core    = chem_env.get_nfcore();
  sys_data.n_frozen_virtual = ccsd_options.freeze_virtual;
  const bool do_freeze      = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;

  if(readv2) {
    // if(chem_env.run_context.contains("mso_tilesize"))
    const int orig_ts = chem_env.run_context["mso_tilesize"];
    if(orig_ts != chem_env.is_context.mso_tilesize) {
      std::string err_msg =
        "\n[ERROR] Restarting a calculation requires the MO tilesize to be the same\n";
      err_msg += "  - current tilesize (" + std::to_string(chem_env.is_context.mso_tilesize);
      err_msg += ") does not match the original tilesize (" + std::to_string(orig_ts) + ")";
      tamm_terminate(err_msg);
    }
  }
  else {
    chem_env.run_context["do_freeze"]    = do_freeze;
    chem_env.run_context["mso_tilesize"] = chem_env.is_context.mso_tilesize;
    if(chem_env.ioptions.task_options.ccsd_t) {
      // not used currently, triples can be restarted with a different tilesize.
      chem_env.run_context["mso_tilesize_triples"] = chem_env.is_context.mso_tilesize_triples;
    }
    if(rank == 0)
      chem_env.write_run_context(); // write here as well in case a run gets killed midway.
  }

  if(rank == 0) {
    std::cout << std::endl
              << "- Tilesize for the MSO space: " << chem_env.is_context.mso_tilesize << std::endl;
  }

  auto diagtol                = cd_options.diagtol; // tolerance for the max. diagonal
  cd_options.max_cvecs_factor = 2 * std::abs(std::log10(diagtol));
  // TODO
  cd_context.max_cvecs = cd_options.max_cvecs_factor * sys_data.nbf;

  int& chol_count = cd_context.num_chol_vecs;

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  Tensor<T>& cholVpr    = cd_context.cholV2;
  auto       itile_size = cd_options.itilesize;

  auto skip_cd = cd_options.skip_cd;

  if(!readv2 && !skip_cd.first) {
    TiledIndexSpace N    = MO("all");
    cd_context.d_f1      = Tensor<T>{{N, N}, {1, 1}};
    cd_context.movecs_so = Tensor<T>{AO, N};
    Tensor<T>& movecs_so = cd_context.movecs_so;
    Tensor<T>::allocate(&ec, cd_context.d_f1, movecs_so);

    exachem::cholesky_2e::two_index_transform(chem_env, ec, C_AO, F_AO, C_beta_AO, F_beta_AO,
                                              cd_context.d_f1, shells, movecs_so,
                                              is_dlpno || !is_mso);
    if(!is_dlpno) exachem::cholesky_2e::cholesky_2e<T>(ec, chem_env);

    MO = chem_env.is_context.MSO; // modified if freezing
    if(do_freeze) {
      TiledIndexSpace N_eff = MO("all");
      Tensor<T>       d_f1_new{{N_eff, N_eff}, {1, 1}};
      Tensor<T>::allocate(&ec, d_f1_new);
      if(rank == 0) {
        Matrix f1_eig     = tamm_to_eigen_matrix(cd_context.d_f1);
        Matrix f1_new_eig = exachem::cholesky_2e::reshape_mo_matrix(chem_env, f1_eig);
        eigen_to_tamm_tensor(d_f1_new, f1_new_eig);
        f1_new_eig.resize(0, 0);
      }
      Tensor<T>::deallocate(cd_context.d_f1);
      cd_context.d_f1 = d_f1_new;
    }

    if(ccsd_options.writet) {
      if(!fs::exists(files_dir)) fs::create_directories(files_dir);

      write_to_disk<TensorType>(movecs_so, cd_context.movecs_so_file);
      write_to_disk(cd_context.d_f1, cd_context.f1file);
      write_to_disk(cholVpr, cd_context.v2file);

      if(rank == 0) {
        std::ofstream out(cholfile, std::ios::out);
        if(!out) cerr << "Error opening file " << cholfile << std::endl;
        out << chol_count << std::endl;
        out.close();
      }
    }
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
    chem_env.is_context.MSO = MO; // modified if freezing

    IndexSpace      chol_is{range(0, chol_count)};
    TiledIndexSpace CI{chol_is, static_cast<tamm::Tile>(itile_size)};
    chem_env.is_context.CI = CI;

    TiledIndexSpace N = MO("all");

    cd_context.d_f1      = Tensor<T>{{N, N}, {1, 1}};
    cd_context.movecs_so = Tensor<T>{AO, N};
    cholVpr = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
    if(!is_dlpno) Tensor<TensorType>::allocate(&ec, cholVpr);
    Tensor<TensorType>::allocate(&ec, cd_context.d_f1, cd_context.movecs_so);

    if(readv2) {
      if(!fmv_exist) {
        std::string fnf = cd_context.f1file + "; " + cd_context.movecs_so_file;
        if(!is_dlpno) fnf = fnf + "; " + cd_context.v2file;
        tamm_terminate("\n [Cholesky restart] Error reading one or all of the files: [" + fnf +
                       "]");
      }
      if(do_freeze != chem_env.run_context["do_freeze"]) {
        tamm_terminate("\n [Cholesky Error] Restart requires the freezing options to be the same");
      }
      read_from_disk(cd_context.movecs_so, cd_context.movecs_so_file);
      read_from_disk(cd_context.d_f1, cd_context.f1file);
      if(!is_dlpno) read_from_disk(cholVpr, cd_context.v2file);
      ec.pg().barrier();
    }
  }

  Tensor<T>& movecs_so = cd_context.movecs_so;
  if(!cd_context.keep_movecs_so) free_tensors(movecs_so);

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
  chem_env.is_context.CI = CI;

  // cd_context.num_chol_vecs                   = chol_count;
  sys_data.results["output"]["CD"]["n_cholesky_vectors"] = chol_count;

  sys_data.results["output"]["CD"]["diagtol"] = chem_env.ioptions.cd_options.diagtol;

  if(rank == 0) sys_data.print();

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
        (tmp(mo1,nu) = movecs_so(mu,mo1) * hcore(mu,nu))
        (hcore_mo(mo1,mo2) = tmp(mo1,nu) * movecs_so(nu,mo2))
        .deallocate(tmp,hcore).execute();
    // clang-format on

    ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
    std::string      mop_dir   = files_dir + "/mos_txt/";
    std::string      mofprefix = mop_dir + chem_env.workspace_dir;
    if(!fs::exists(mop_dir)) fs::create_directories(mop_dir);

    Tensor<T> d_v2            = setupV2<T>(ec, MO, CI, cholVpr, chol_count);
    Tensor<T> d_f1_dense      = to_dense_tensor(ec_dense, cd_context.d_f1);
    Tensor<T> movecs_so_dense = to_dense_tensor(ec_dense, movecs_so);
    Tensor<T> d_v2_dense      = to_dense_tensor(ec_dense, d_v2);
    Tensor<T> hcore_dense     = to_dense_tensor(ec_dense, hcore_mo);

    Tensor<T>::deallocate(hcore_mo, d_v2);

    print_dense_tensor(d_v2_dense, mofprefix + ".v2_mo");
    print_dense_tensor(movecs_so_dense, mofprefix + ".movecs_so");
    print_dense_tensor(d_f1_dense, mofprefix + ".fock_mo");
    print_dense_tensor(hcore_dense, mofprefix + ".hcore_mo");

    Tensor<T>::deallocate(hcore_dense, d_f1_dense, movecs_so_dense, d_v2_dense);
  }

} // END of cholesky_2e_driver
