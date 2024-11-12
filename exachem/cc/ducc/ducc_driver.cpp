/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

#include <filesystem>
namespace fs = std::filesystem;

namespace exachem::cc::ducc {

void ducc_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  const bool  is_rhf       = chem_env.sys_data.is_restricted;
  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  CCContext& cc_context = chem_env.cc_context;
  cc_context.init_filenames(files_prefix);
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  auto              debug      = ccsd_options.debug;
  bool              scf_conv   = chem_env.scf_context.no_scf;
  std::string       t1file     = cc_context.t1file;
  std::string       t2file     = cc_context.t2file;
  const std::string ccsdstatus = cc_context.ccsdstatus;

  bool ccsd_restart = ccsd_options.readt ||
                      ((fs::exists(t1file) && fs::exists(t2file) && fs::exists(cd_context.f1file) &&
                        fs::exists(cd_context.v2file)));

  if(ccsd_options.writev) ccsd_options.writet = true;

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& CI      = chem_env.is_context.CI;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr = chem_env.cd_context.cholV2;

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
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    p_evl_sorted = tamm::diagonal(d_f1);
  }

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ExecutionHW ex_hw = ec.exhw();

  ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

  std::string fullV2file = cd_context.fullV2file;
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

  ccsd_stats(ec, chem_env.scf_context.hf_energy, residual, corr_energy, ccsd_options.threshold);

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
    v2tensors = cholesky_2e::setupV2Tensors<T>(ec, cholVpr, ex_hw);
    if(ccsd_options.writet) { v2tensors.write_to_disk(files_prefix); }
  }
  else {
    v2tensors.allocate(ec, MO);
    v2tensors.read_from_disk(files_prefix);
  }

  free_tensors(cholVpr);

  DUCC_T_CCSD_Driver<T>(chem_env, ec, MO, d_t1, d_t2, d_f1, v2tensors, ccsd_options.nactive, ex_hw);

  v2tensors.deallocate();
  free_tensors(d_t1, d_t2, d_f1);
  ec.flush_and_sync();
  // delete ec;
}
} // namespace exachem::cc::ducc
