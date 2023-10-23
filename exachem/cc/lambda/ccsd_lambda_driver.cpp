/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ccsd_lambda.hpp"

#include <filesystem>
namespace fs = std::filesystem;

template<typename T>
void compute_rdm(std::vector<int>&, std::string, Scheduler&, TiledIndexSpace&, Tensor<T>, Tensor<T>,
                 Tensor<T>, Tensor<T>);

void ccsd_lambda_driver(std::string filename, OptionsMap options_map) {
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

  auto [MO, total_orbitals] = setupMOIS(sys_data);

  std::string out_fp       = sys_data.output_file_prefix + "." + ccsd_options.basis;
  std::string files_dir    = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
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
    cd_svd_driver<T>(sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells,
                     shell_tile_map, ccsd_restart, cholfile);

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

  //    free_tensors(cholVpr);

  auto [l_r1, l_r2, d_y1, d_y2, l_r1s, l_r2s, d_y1s, d_y2s] =
    setupLambdaTensors<T>(ec, MO, ccsd_options.ndiis);

  cc_t1 = std::chrono::high_resolution_clock::now();
  std::tie(residual, corr_energy) =
    lambda_ccsd_driver<T>(sys_data, ec, MO, CI, d_t1, d_t2, d_f1, v2tensors, cholVpr, l_r1, l_r2,
                          d_y1, d_y2, l_r1s, l_r2s, d_y1s, d_y2s, p_evl_sorted);
  cc_t2 = std::chrono::high_resolution_clock::now();

  if(rank == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < ccsd_options.threshold) {
      std::cout << " CCSD Lambda Iterations converged" << std::endl;
    }
  }

  ccsd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time taken for CCSD Lambda: " << std::fixed << std::setprecision(2) << ccsd_time
              << " secs" << std::endl
              << std::endl;

  v2tensors.deallocate();
  free_tensors(d_f1, l_r1, l_r2, cholVpr);
  free_vec_tensors(l_r1s, l_r2s, d_y1s, d_y2s);

  cc_t1 = std::chrono::high_resolution_clock::now();

  Tensor<T> DipX_ao{AO_opt, AO_opt};
  Tensor<T> DipY_ao{AO_opt, AO_opt};
  Tensor<T> DipZ_ao{AO_opt, AO_opt};
  Tensor<T> DipX_mo{{N, N}, {1, 1}};
  Tensor<T> DipY_mo{{N, N}, {1, 1}};
  Tensor<T> DipZ_mo{{N, N}, {1, 1}};

  Scheduler sch{ec};
  // Initialize the tensors
  // clang-format off
  sch.allocate(DipX_ao, DipY_ao, DipZ_ao, DipX_mo, DipY_mo, DipZ_mo)
    (DipX_ao() = 0)
    (DipY_ao() = 0)
    (DipZ_ao() = 0)
  .execute();
  // clang-format on

  SCFVars spvars;
  compute_shellpair_list(ec, shells, spvars);
  std::tie(spvars.shell_tile_map, spvars.AO_tiles, spvars.AO_opttiles) =
    compute_AO_tiles(ec, sys_data, shells);

  auto atoms = sys_data.options_map.options.atoms;
  compute_dipole_ints(ec, spvars, DipX_ao, DipY_ao, DipZ_ao, atoms, shells,
                      libint2::Operator::emultipole1);

  auto [mu, nu, ku]     = AO_opt.labels<3>("all");
  auto [mo1, mo2]       = MO.labels<2>("all");
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  // AO2MO transormation
  Tensor<T> lcao_t{mo1, mu};
  Tensor<T> tmp1{nu, mo2};
  Tensor<T> dens{mu, nu};
  Tensor<T> dipole_mx{}, dipole_my{}, dipole_mz{};

  sch.allocate(dipole_mx, dipole_my, dipole_mz, lcao_t, dens, tmp1)(lcao_t(mo1, mu) = lcao(mu, mo1))
    .execute();

  std::string densityfile_alpha = files_dir + "/scf/" + out_fp + ".alpha.density";
  read_from_disk<T>(dens, densityfile_alpha);

  // compute electronic dipole moments
  sch(dipole_mx() = dens() * DipX_ao())(dipole_my() = dens() * DipY_ao())(dipole_mz() =
                                                                            dens() * DipZ_ao())
    .deallocate(dens)
    .execute(ex_hw);

  auto dmx = get_scalar(dipole_mx);
  auto dmy = get_scalar(dipole_my);
  auto dmz = get_scalar(dipole_mz);

  // compute nuclear dipole moments
  for(int i = 0; i < atoms.size(); i++) {
    dmx += atoms[i].x * atoms[i].atomic_number;
    dmy += atoms[i].y * atoms[i].atomic_number;
    dmz += atoms[i].z * atoms[i].atomic_number;
  }

  const double au2debye      = 2.541766;
  double       total_dip_val = 0;

  auto print_dm = [&](std::string mname) {
    if(rank == 0) {
      std::cout << std::fixed << std::setprecision(8) << std::right;
      std::cout << std::endl
                << mname << " dipole moments / hartree & Debye" << std::endl
                << std::string(40, '-') << std::endl;
      std::cout << "X" << std::showpos << std::setw(18) << dmx << "\t" << dmx * au2debye
                << std::endl;
      std::cout << "Y" << std::showpos << std::setw(18) << dmy << "\t" << dmy * au2debye
                << std::endl;
      std::cout << "Z" << std::showpos << std::setw(18) << dmz << "\t" << dmz * au2debye
                << std::endl;
      double val    = std::sqrt(dmx * dmx + dmy * dmy + dmz * dmz);
      total_dip_val = val;
      std::cout << "Total" << std::showpos << std::setw(14) << val << "\t" << val * au2debye
                << std::endl;
    }
  };

  print_dm("SCF");

  auto dip_ao2mo = [&, nu = nu, mu = mu, mo1 = mo1, mo2 = mo2, lcao = lcao](Tensor<T>& dip_ao,
                                                                            Tensor<T>& dip_mo) {
    sch(tmp1(nu, mo2) = dip_ao(mu, nu) * lcao(mu, mo2))(dip_mo(mo1, mo2) =
                                                          lcao_t(mo1, nu) * tmp1(nu, mo2))
      .execute(ex_hw);
  };

  dip_ao2mo(DipX_ao, DipX_mo);
  dip_ao2mo(DipY_ao, DipY_mo);
  dip_ao2mo(DipZ_ao, DipZ_mo);
  sch.deallocate(lcao, lcao_t, tmp1, DipX_ao, DipY_ao, DipZ_ao).execute();

  tmp1 = Tensor<T>{h1, p1};
  Tensor<T> tmp2{{p1, h1}, {1, 1}}, tmp3{{h1, h2}, {1, 1}}, tmp4{{h2, h1}, {1, 1}},
    tmp5{{p3, p1}, {1, 1}};
  sch.allocate(tmp1, tmp2, tmp3, tmp4, tmp5).execute();

  auto compute_dm = [&, h1 = h1, h2 = h2, h3 = h3, p1 = p1, p2 = p2, p3 = p3, d_y1 = d_y1,
                     d_y2 = d_y2](Tensor<T>& dipole_m, Tensor<T>& dipole_i) {
    // clang-format off
    sch
      (dipole_m()     =        dipole_i(h1,p1)   * d_t1(p1,h1)                         )
      (dipole_m()    +=        d_y1(h1,p1)       * dipole_i(p1,h1)                     )
    //(dipole_m()    += -1.  * y1(h1,p1)         * dipole_i(h2,h1) * t1(p1,h2)         )
      (tmp1(h1,p1)    =        dipole_i(h2,h1)   * d_t1(p1,h2)                         )
      (dipole_m()    += -1.  * d_y1(h1,p1)       * tmp1(h1,p1)                         )
    //(dipole_m()    +=        d_y1(h1,p1)       * dipole_i(p1,p2) * t1(p2,h1)         )
      (tmp2(p1,h1)    =        dipole_i(p1,p2)   * d_t1(p2,h1)                         )
      (dipole_m()    +=        d_y1(h1,p1)       * tmp2(p1,h1)                         )
    //(dipole_m()    +=        d_y1(h1,p1)       * dipole_i(h2,p2) * t2(p1,p2, h1, h2) )
      (tmp2(p1,h1)    =        dipole_i(h2,p2)   * d_t2(p1,p2,h1,h2)                   )
      (dipole_m()    +=        d_y1(h1,p1)       * tmp2(p1,h1)                         )
    //(dipole_m()    += -1.  * d_y1(h1,p1)       * dipole_i(h2,p2) * t1(p1,h2) * t1(p2,h1) )
      (tmp3(h1,h2)    =        d_y1(h1,p1)       * d_t1(p1,h2)                         )
      (tmp4(h2,h1)    = -1.  * dipole_i(h2,p2)   * d_t1(p2,h1)                         )
      (dipole_m()    +=        tmp3(h1,h2)       * tmp4(h2,h1)                         )
    //(dipole_m()    += -0.5 * y2(h3,h2,p1,p2)   * dipole_i(h1,h3) * t2(p1,p2,h1,h2)   )
      (tmp3(h3,h1)    =        d_y2(h3,h2,p1,p2) * d_t2(p1,p2,h1,h2)                   )
      (dipole_m()    += -0.5 * tmp3(h3,h1)       * dipole_i(h1,h3)                     )
    //(dipole_m()    +=  0.5 * y2(h1,h2,p3,p2)   * dipole_i(p3,p1) * t2(p1,p2,h1,h2)   )
      (tmp5(p1,p3)    =        d_y2(h1,h2,p3,p2) * d_t2(p1,p2,h1,h2)                   )
      (dipole_m()    +=  0.5 * tmp5(p1,p3)       * dipole_i(p3,p1)                     )
    //(dipole_m()    += -0.5 * y2(h3,h2,p3,p2)   * dipole_i(h1,p1) * t1(p3, h1) * t2(p1,p2,h3,h2) )
      (tmp5(p1,p3)    =        d_y2(h3,h2,p3,p2) * d_t2(p1,p2,h3,h2)                   )
      (tmp2(p3,h1)    =        tmp5(p1,p3)       * dipole_i(h1,p1)                     )
      (dipole_m()    += -0.5 * tmp2(p3,h1)       * d_t1(p3,h1)                         )
    //(dipole_m()    += -0.5 * y2(h3,h2,p3,p2)   * dipole_i(h1,p1) * t1(p1,h3) * t2(p3,p2,h1,h2) )
      (tmp3(h3,h1)    =        d_y2(h3,h2,p3,p2) * d_t2(p3,p2,h1,h2)                   )
      (tmp1(h3,p1)    =        tmp3(h3,h1)       * dipole_i(h1,p1)                     )
      (dipole_m()    += -0.5 * tmp1(h3,p1)       * d_t1(p1,h3)                         )
      .execute(ex_hw);
    // clang-format on
  };

  compute_dm(dipole_mx, DipX_mo);
  compute_dm(dipole_my, DipY_mo);
  compute_dm(dipole_mz, DipZ_mo);

  dmx += get_scalar(dipole_mx);
  dmy += get_scalar(dipole_my);
  dmz += get_scalar(dipole_mz);

  print_dm("CCSD");

  cc_t2     = std::chrono::high_resolution_clock::now();
  ccsd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::fixed << std::setprecision(3) << std::endl
              << "Time taken to compute dipole moments: " << std::fixed << std::setprecision(2)
              << ccsd_time << " secs" << std::endl;

  if(rank == 0) {
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["X"]               = dmx;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Y"]               = dmy;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Z"]               = dmz;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Total"]           = total_dip_val;
    sys_data.results["output"]["CCSD_Lambda"]["performance"]["total_time"] = ccsd_time;
    write_json_data(sys_data, "CCSD_Lambda");
  }

  compute_rdm(ccsd_options.cc_rdm, files_prefix, sch, MO, d_t1, d_t2, d_y1, d_y2);

  sch
    .deallocate(d_t1, d_t2, d_y1, d_y2, tmp1, tmp2, tmp3, tmp4, tmp5, dipole_mx, dipole_my,
                dipole_mz, DipX_mo, DipY_mo, DipZ_mo)
    .execute();

  ec.flush_and_sync();
  // delete ec;
}
