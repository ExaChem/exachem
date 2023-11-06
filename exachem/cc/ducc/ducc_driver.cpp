/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cc/ccsd/cd_ccsd_os_ann.hpp"

#include <filesystem>
namespace fs = std::filesystem;

template<typename T>
void DUCC_T_CCSD_Driver(SystemData sys_data, ExecutionContext& ec, const TiledIndexSpace& MO,
                        Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1, V2Tensors<T>& v2tensors,
                        size_t nactv, ExecutionHW ex_hw);

void ducc_driver(std::string filename, OptionsMap options_map) {
  using T = double;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank = ec.pg().rank();

  auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt,
        AO_tis, scf_conv] = hartree_fock_driver<T>(ec, filename, options_map);

  CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;
  auto         debug        = ccsd_options.debug;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  const int nactv = ccsd_options.nactive;

  const bool is_rhf = sys_data.is_restricted;

  // TODO: Implement check for UHF
  if(nactv > sys_data.n_vir_alpha && is_rhf) tamm_terminate("[DUCC ERROR]: nactive > n_vir_alpha");

  auto [MO, total_orbitals] = setupMOIS(sys_data, false, nactv);

  std::string out_fp       = sys_data.output_file_prefix + "." + ccsd_options.basis;
  std::string files_dir    = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string t1file       = files_prefix + ".t1amp";
  std::string t2file       = files_prefix + ".t2amp";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";
  std::string ccsdstatus   = files_prefix + ".ccsdstatus";

  bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                              fs::exists(f1file) && fs::exists(v2file)));

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cd_svd_driver<T>(sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells,
                     shell_tile_map, ccsd_restart, cholfile);
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

  ExecutionHW ex_hw = ec.exhw();

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
    std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
      sys_data, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s, p_evl_sorted,
      cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
  else
    std::tie(residual, corr_energy) =
      cd_ccsd_os_driver<T>(sys_data, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s,
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

  cc_print(sys_data, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }

  V2Tensors<T> v2tensors;
  if(computeTData && !v2tensors.exist_on_disk(files_prefix)) {
    v2tensors = setupV2Tensors<T>(ec, cholVpr, ex_hw);
    if(ccsd_options.writet) { v2tensors.write_to_disk(files_prefix); }
  }
  else {
    v2tensors.allocate(ec, MO);
    v2tensors.read_from_disk(files_prefix);
  }

  free_tensors(cholVpr);

  DUCC_T_CCSD_Driver<T>(sys_data, ec, MO, d_t1, d_t2, d_f1, v2tensors, nactv, ex_hw);

  v2tensors.deallocate();
  free_tensors(d_t1, d_t2, d_f1);
  ec.flush_and_sync();
  // delete ec;
}
