/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/cc2/cd_cc2_os.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

#include <filesystem>
namespace fs = std::filesystem;

void exachem::cc::ccsd::cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  const bool is_rhf = chem_env.sys_data.is_restricted;

  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  CCContext& cc_context = chem_env.cc_context;
  cc_context.init_filenames(files_prefix);
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  if(ccsd_options.skip_ccsd) {
    TiledIndexSpace& MO = chem_env.is_context.MSO;
    cholesky_2e::update_sysdata(ec, chem_env, MO);
    cd_context.d_f1 = {{MO, MO}, {1, 1}};
    Tensor<T>::allocate(&ec, cd_context.d_f1);
    if(rank == 0) chem_env.sys_data.print();
    return;
  }

  auto        debug      = ccsd_options.debug;
  bool        scf_conv   = chem_env.scf_context.no_scf;
  std::string t1file     = cc_context.t1file;
  std::string t2file     = cc_context.t2file;
  const bool  ccsdstatus = cc_context.is_converged(chem_env.run_context, "ccsd");

  bool ccsd_restart = ccsd_options.readt ||
                      ((fs::exists(t1file) && fs::exists(t2file) && fs::exists(cd_context.f1file) &&
                        fs::exists(cd_context.v2file)));

  const bool use_subgroup = cc_context.use_subgroup;

  int nsranks = 0;
  if(use_subgroup) {
    nsranks = chem_env.sys_data.nbf / 15;
    if(nsranks < 1) nsranks = 1;
    int ga_cnn = ec.nnodes();
    if(nsranks > ga_cnn) nsranks = ga_cnn;
    nsranks = nsranks * ec.ppn();

    cc_context.sub_pg = ProcGroup::create_subgroup(ec.pg(), nsranks);
    cc_context.sub_ec = nullptr;

    if(cc_context.sub_pg.is_valid()) {
      cc_context.sub_ec =
        new ExecutionContext(cc_context.sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
    }
  }

  else cc_context.sub_ec = &ec;

  // Scheduler sub_sch{*sub_ec};

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& CI      = chem_env.is_context.CI;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = cd_context.d_f1;
  Tensor<T>        cholVpr = cd_context.cholV2;

  std::vector<T>&        p_evl_sorted = cd_context.p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  if(is_rhf)
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) =
      setupTensors_cs(ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && ccsdstatus && scf_conv);
  else
    std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) =
      setupTensors(ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && ccsdstatus && scf_conv);

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

  ccsd_restart = ccsd_restart && ccsdstatus && scf_conv;

  bool computeTData = cc_context.compute.fvt12_full;

  Tensor<T> dt1_full, dt2_full;
  if(computeTData && is_rhf) setup_full_t1t2(ec, MO, dt1_full, dt2_full);

  double residual = 0, corr_energy = 0;

  if(is_rhf) {
    if(ccsd_restart && use_subgroup) {
      if(cc_context.sub_pg.is_valid()) {
        const int ppn = ec.ppn();
        if(rank == 0)
          std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                    << std::endl;
        if(cc_context.task_cc2) {
          std::tie(residual, corr_energy) = exachem::cc2::cc2_cs::cd_cc2_cs_driver<T>(
            chem_env, *(cc_context.sub_ec), MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s,
            d_t1s, d_t2s, p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix,
            computeTData);
        }

        else {
          std::tie(residual, corr_energy) = exachem::cc::ccsd::cd_ccsd_cs_driver<T>(
            chem_env, *(cc_context.sub_ec), MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s,
            d_t1s, d_t2s, p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix,
            computeTData);
        }
      }
      ec.pg().barrier();
    }
    else {
      if(cc_context.task_cc2) {
        std::tie(residual, corr_energy) = exachem::cc2::cc2_cs::cd_cc2_cs_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
      }
      else {
        std::tie(residual, corr_energy) = exachem::cc::ccsd::cd_ccsd_cs_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, dt1_full, dt2_full, ccsd_restart, files_prefix, computeTData);
      }
    }
  }
  else {
    if(ccsd_restart && use_subgroup) {
      if(cc_context.sub_pg.is_valid()) {
        const int ppn = ec.ppn();
        if(rank == 0)
          std::cout << "Executing with " << nsranks << " ranks (" << nsranks / ppn << " nodes)"
                    << std::endl;

        if(cc_context.task_cc2) {
          std::tie(residual, corr_energy) = exachem::cc2::cc2_os::cd_cc2_os_driver<T>(
            chem_env, *(cc_context.sub_ec), MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s,
            d_t1s, d_t2s, p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
        }
        else {
          std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
            chem_env, *(cc_context.sub_ec), MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s,
            d_t1s, d_t2s, p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
        }
      }
      ec.pg().barrier();
    }
    else {
      if(cc_context.task_cc2) {
        std::tie(residual, corr_energy) = exachem::cc2::cc2_os::cd_cc2_os_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
      }
      else {
        std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
          chem_env, ec, MO, CI, d_t1, d_t2, d_f1, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s,
          p_evl_sorted, cholVpr, ccsd_restart, files_prefix, computeTData);
      }
    }
  }

  std::string task_str = "CCSD";
  if(cc_context.task_cc2) task_str = "CC2";
  ccsd_stats(ec, chem_env.scf_context.hf_energy, residual, corr_energy, ccsd_options.threshold,
             task_str);

  if(ccsd_options.writet && !ccsdstatus) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    chem_env.run_context["ccsd"]["converged"] = true;
  }
  else if(!ccsdstatus) chem_env.run_context["ccsd"]["converged"] = false;
  if(rank == 0) chem_env.write_run_context();

  if(!cc_context.task_cc2) task_str = "Cholesky CCSD";
  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    if(is_rhf)
      std::cout << std::endl
                << "Time taken for Closed Shell " << task_str << ": " << std::fixed
                << std::setprecision(2) << ccsd_time << " secs" << std::endl;
    else
      std::cout << std::endl
                << "Time taken for Open Shell " << task_str << ": " << std::fixed
                << std::setprecision(2) << ccsd_time << " secs" << std::endl;
  }

  cc_print(chem_env, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }

  ExecutionHW ex_hw = ec.exhw();

  cholesky_2e::V2Tensors<T>& v2tensors = cd_context.v2tensors;
  if(computeTData) {
    bool compute_fullv2 = cc_context.compute.v2_full;
    if(compute_fullv2 && (ccsd_options.writet || ccsd_options.readt)) {
      if(v2tensors.exist_on_disk(files_prefix)) {
        v2tensors.allocate(ec, MO);
        v2tensors.read_from_disk(files_prefix);
        compute_fullv2 = false;
      }
    }
    if(compute_fullv2) {
      v2tensors = cholesky_2e::setupV2Tensors<T>(ec, cholVpr, ex_hw);
      if(ccsd_options.writet) { v2tensors.write_to_disk(files_prefix); }
    }
  }

  cc_context.d_t1 = d_t1;
  cc_context.d_t2 = d_t2;

  if(!cc_context.keep.fvt12_full) {
    free_tensors(d_f1, d_t1, d_t2, cholVpr);
    if(computeTData && is_rhf) { free_tensors(dt1_full, dt2_full); }
    if(cc_context.compute.v2_full) v2tensors.deallocate();
  }
  else {
    if(is_rhf) {
      free_tensors(d_t1, d_t2);        // free t1_aa, t2_abab
      cc_context.d_t1_full = dt1_full; // need full T1,T2
      cc_context.d_t2_full = dt2_full; // need full T1,T2
    }
    else {
      cc_context.d_t1_full = d_t1;
      cc_context.d_t2_full = d_t2;
    }
  }

  ec.flush_and_sync();
  // delete ec;
}
