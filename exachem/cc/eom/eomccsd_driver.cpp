/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cholesky/cholesky_2e_driver.hpp"
#include "eomccsd_opt.hpp"
void exachem::cc::eom::eom_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T = double;

  auto rank = ec.pg().rank();
  scf::scf_driver(ec, chem_env);

  double              hf_energy      = chem_env.hf_energy;
  libint2::BasisSet   shells         = chem_env.shells;
  Tensor<T>           C_AO           = chem_env.C_AO;
  Tensor<T>           C_beta_AO      = chem_env.C_beta_AO;
  Tensor<T>           F_AO           = chem_env.F_AO;
  Tensor<T>           F_beta_AO      = chem_env.F_beta_AO;
  TiledIndexSpace     AO_opt         = chem_env.AO_opt;
  TiledIndexSpace     AO_tis         = chem_env.AO_tis;
  std::vector<size_t> shell_tile_map = chem_env.shell_tile_map;
  bool                scf_conv       = chem_env.no_scf;

  SystemData& sys_data = chem_env.sys_data;
  // CCSDOptions ccsd_options = chem_env.ioptions.ccsd_options;
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;
  auto         debug        = ccsd_options.debug;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = cholesky_2e::setupMOIS(ec, chem_env);

  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string t1file       = files_prefix + ".t1amp";
  std::string t2file       = files_prefix + ".t2amp";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";
  std::string ccsdstatus   = files_prefix + ".ccsdstatus";

  const bool is_rhf = sys_data.is_restricted;

  bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                              fs::exists(f1file) && fs::exists(v2file)));

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cholesky_2e::cholesky_2e_driver<T>(chem_env, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO,
                                       shells, shell_tile_map, ccsd_restart, cholfile);
  free_tensors(lcao);

  // if(ccsd_options.writev) ccsd_options.writet = true;

  TiledIndexSpace N = MO("all");

  std::vector<T>         p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  if(is_rhf)
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors_cs(
      ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);
  else
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors(
      ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

  if(ccsd_restart) {
    read_from_disk(d_f1, f1file);
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    read_from_disk(cholVpr, v2file);
    ec.pg().barrier();
    p_evl_sorted = tamm::diagonal(d_f1);
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

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

  std::string fullV2file = files_prefix + ".fullV2";
  // t1file = files_prefix+".fullT1amp";
  // t2file = files_prefix+".fullT2amp";

  bool computeTData = true; // not needed
  // if(ccsd_options.writev)
  //     computeTData = computeTData && !fs::exists(fullV2file);
  //&& !fs::exists(t1file) && !fs::exists(t2file);

  Tensor<T> dt1_full, dt2_full;
  if(computeTData && is_rhf) setup_full_t1t2(ec, MO, dt1_full, dt2_full);

  double residual = 0, corr_energy = 0;

  if(is_rhf)
    std::tie(residual, corr_energy) = exachem::cc::ccsd::cd_ccsd_cs_driver<T>(
      chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s, p_evl_sorted,
      cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
  else
    std::tie(residual, corr_energy) =
      cd_ccsd_os_driver<T>(chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s,
                           d_t2s, p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);

  if(computeTData && is_rhf) {
    // if(ccsd_options.writev) {
    //     write_to_disk(dt1_full,t1file);
    //     write_to_disk(dt2_full,t2file);
    //     free_tensors(dt1_full, dt2_full);
    // }
    free_tensors(d_t1, d_t2); // free t1_aa, t2_abab
    d_t1 = dt1_full;          // need full T1,T2
    d_t2 = dt2_full;          // need full T1,T2
  }

  ccsd_stats(ec, hf_energy, residual, corr_energy, ccsd_options.threshold);

  if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    if(rank == 0) {
      std::ofstream out(ccsdstatus, std::ios::out);
      if(!out) cerr << "Error opening file " << ccsdstatus << endl;
      out << 1 << std::endl;
      out.close();
    }
  }

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    if(is_rhf)
      std::cout << std::endl
                << "Time taken for Closed Shell Cholesky CCSD: " << std::fixed
                << std::setprecision(2) << ccsd_time << " secs" << std::endl;
    else
      std::cout << std::endl
                << "Time taken for Open Shell Cholesky CCSD: " << std::fixed << std::setprecision(2)
                << ccsd_time << " secs" << std::endl;
  }

  cc_print(chem_env, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }

  cholesky_2e::V2Tensors<T> v2tensors;
  if(computeTData && !v2tensors.exist_on_disk(files_prefix)) {
    v2tensors = cholesky_2e::setupV2Tensors<T>(ec, cholVpr, ec.exhw());
    if(ccsd_options.writet) { v2tensors.write_to_disk(files_prefix); }
  }
  else {
    v2tensors.allocate(ec, MO);
    v2tensors.read_from_disk(files_prefix);
  }

  free_tensors(cholVpr);

  if(ccsd_options.eom_nroots <= 0) tamm_terminate("EOMCCSD: nroots should be greater than 1");

  std::string eom_type = ccsd_options.eom_type;

  // EOMCCSD Routine
  cc_t1 = std::chrono::high_resolution_clock::now();

  if(eom_type == "right")
    right_eomccsd_driver<T>(chem_env, ec, MO, d_t1, d_t2, d_f1, v2tensors, p_evl_sorted);

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time = std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << std::endl
              << "Time taken for " << eom_type << "-Eigenstate EOMCCSD: " << std::fixed
              << std::setprecision(2) << ccsd_time << " secs" << std::endl;
  }

  v2tensors.deallocate();
  free_tensors(d_t1, d_t2, d_f1);

  ec.flush_and_sync();
  // delete ec;
}
