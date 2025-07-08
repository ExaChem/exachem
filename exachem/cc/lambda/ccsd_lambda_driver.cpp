/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/lambda/ccsd_lambda.hpp"
// #include "exachem/common/termcolor.hpp"
#include "exachem/scf/scf_guess.hpp"
#include <filesystem>

namespace fs = std::filesystem;

template<typename T>
void ccsd_natural_orbitals(ChemEnv& chem_env, std::vector<int>&, std::string, std::string,
                           Scheduler&, ExecutionContext&, TiledIndexSpace&, TiledIndexSpace&,
                           Tensor<T>&, ExecutionHW);

template<typename T>
Tensor<T> compute_1rdm(std::vector<int>&, std::string, Scheduler&, TiledIndexSpace&, Tensor<T>,
                       Tensor<T>, Tensor<T>, Tensor<T>);

template<typename T>
Tensor<T> compute_2rdm(std::vector<int>&, std::string, Scheduler&, TiledIndexSpace&, Tensor<T>,
                       Tensor<T>, Tensor<T>, Tensor<T>);

void exachem::cc::ccsd_lambda::ccsd_lambda_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T   = double;
  auto rank = ec.pg().rank();

  chem_env.cd_context.keep_movecs_so = true;
  CCContext& cc_context              = chem_env.cc_context;
  cc_context.keep.fvt12_full         = true;
  cc_context.compute.set(true, true); // compute ft12 and v2 in full
  exachem::cc::ccsd::cd_ccsd_driver(ec, chem_env);

  SystemData& sys_data  = chem_env.sys_data;
  std::string files_dir = chem_env.get_files_dir();
  ExecutionHW ex_hw     = ec.exhw();

  TiledIndexSpace&           MO           = chem_env.is_context.MSO;
  TiledIndexSpace&           CI           = chem_env.is_context.CI;
  TiledIndexSpace            N            = MO("all");
  Tensor<T>                  d_f1         = chem_env.cd_context.d_f1;
  Tensor<T>                  cholVpr      = chem_env.cd_context.cholV2;
  Tensor<T>                  d_t1         = chem_env.cc_context.d_t1_full;
  Tensor<T>                  d_t2         = chem_env.cc_context.d_t2_full;
  CCSDOptions&               ccsd_options = chem_env.ioptions.ccsd_options;
  auto                       debug        = ccsd_options.debug;
  cholesky_2e::V2Tensors<T>& v2tensors    = chem_env.cd_context.v2tensors;

  std::string files_prefix = chem_env.get_files_prefix();

  //    free_tensors(cholVpr);
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  auto [l_r1, l_r2, d_y1, d_y2, l_r1s, l_r2s, d_y1s, d_y2s] =
    setupLambdaTensors<T>(ec, MO, ccsd_options.ndiis);

  double residual{};
  double corr_energy{};

  std::tie(residual, corr_energy) =
    lambda_ccsd_driver<T>(chem_env, ec, MO, CI, d_t1, d_t2, d_f1, v2tensors, cholVpr, l_r1, l_r2,
                          d_y1, d_y2, l_r1s, l_r2s, d_y1s, d_y2s, chem_env.cd_context.p_evl_sorted);
  auto cc_t2 = std::chrono::high_resolution_clock::now();

  if(rank == 0) {
    std::cout << std::string(66, '-') << std::endl;
    if(residual < ccsd_options.threshold) {
      std::cout << " CCSD Lambda Iterations converged" << std::endl;
    }
  }

  auto ccsd_lambda_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << std::endl
              << "Time taken for CCSD Lambda: " << std::fixed << std::setprecision(2)
              << ccsd_lambda_time << " secs" << std::endl
              << std::endl;
    sys_data.results["output"]["CCSD_Lambda"]["performance"]["total_time"] = ccsd_lambda_time;
  }

  v2tensors.deallocate();
  free_tensors(d_f1, cholVpr);
  free_tensors(l_r1, l_r2);
  free_vec_tensors(l_r1s, l_r2s, d_y1s, d_y2s);

  cc_t1 = std::chrono::high_resolution_clock::now();

  TiledIndexSpace& AO_opt = chem_env.is_context.AO_opt;

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

  scf::SCFData    spvars;
  scf::SCFCompute scf_compute;
  scf_compute.compute_shellpair_list(ec, chem_env.shells, spvars);
  std::tie(spvars.shell_tile_map, spvars.AO_tiles, spvars.AO_opttiles) =
    scf_compute.compute_AO_tiles(ec, chem_env, chem_env.shells);

  auto          atoms = chem_env.atoms;
  scf::SCFGuess scf_guess;
  scf_guess.compute_dipole_ints(ec, spvars, DipX_ao, DipY_ao, DipZ_ao, atoms, chem_env.shells,
                                libint2::Operator::emultipole1);

  auto [mu, nu, ku]     = AO_opt.labels<3>("all");
  auto [mo1, mo2]       = MO.labels<2>("all");
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  Tensor<T> lcao = chem_env.cd_context.movecs_so;

  // AO2MO transormation
  Tensor<T> lcao_t{mo1, mu};
  Tensor<T> tmp1{nu, mo2};
  Tensor<T> dens{mu, nu};
  Tensor<T> dipole_mx{}, dipole_my{}, dipole_mz{};

  sch.allocate(dipole_mx, dipole_my, dipole_mz, lcao_t, dens, tmp1)(lcao_t(mo1, mu) = lcao(mu, mo1))
    .execute();

  std::string densityfile_alpha =
    files_dir + "/scf/" + sys_data.output_file_prefix + ".alpha.density";
  scf::SCFIO<TensorType> scf_output;
  scf_output.rw_mat_disk(dens, densityfile_alpha, debug, true); // read density

  // compute electronic dipole moments
  sch(dipole_mx() = dens() * DipX_ao())(dipole_my() = dens() * DipY_ao())(dipole_mz() =
                                                                            dens() * DipZ_ao())
    .deallocate(dens)
    .execute(ex_hw);

  auto dmx = get_scalar(dipole_mx);
  auto dmy = get_scalar(dipole_my);
  auto dmz = get_scalar(dipole_mz);

  // compute nuclear dipole moments
  for(size_t i = 0; i < atoms.size(); i++) {
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

  cc_t2 = std::chrono::high_resolution_clock::now();
  auto dip_ttime =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::fixed << std::setprecision(3) << std::endl
              << "Time taken to compute dipole moments: " << std::fixed << std::setprecision(2)
              << dip_ttime << " secs" << std::endl;

  if(rank == 0) {
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["X"]     = dmx;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Y"]     = dmy;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Z"]     = dmz;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["Total"] = total_dip_val;
    sys_data.results["output"]["CCSD_Lambda"]["dipole"]["performance"]["total_time"] = dip_ttime;
    chem_env.write_json_data();
  }

  if(!ccsd_options.cc_rdm.empty()) {
    if(ccsd_options.cc_rdm[0] == 1) {
      Tensor<T> gamma1;
      gamma1 = compute_1rdm<T>(ccsd_options.cc_rdm, files_prefix, sch, MO, d_t1, d_t2, d_y1, d_y2);
      ccsd_natural_orbitals(chem_env, ccsd_options.cc_rdm, files_prefix, files_dir, sch, ec, MO,
                            AO_opt, gamma1, ex_hw);
    }
    auto rdm_val = (ccsd_options.cc_rdm.size() == 2) ? ccsd_options.cc_rdm[1]
                                                     : ccsd_options.cc_rdm[0];
    if(rdm_val == 2) {
      Tensor<T> gamma2;
      gamma2 = compute_2rdm<T>(ccsd_options.cc_rdm, files_prefix, sch, MO, d_t1, d_t2, d_y1, d_y2);
    }
  }

  sch
    .deallocate(d_t1, d_t2, d_y1, d_y2, tmp1, tmp2, tmp3, tmp4, tmp5, dipole_mx, dipole_my,
                dipole_mz, DipX_mo, DipY_mo, DipZ_mo)
    .execute();

  ec.flush_and_sync();
  // delete ec;
}
