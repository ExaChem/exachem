/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"

namespace exachem::cc::ducc::internal {

// Member function implementation
template<typename T>
void DUCCInternal<T>::DUCC_T_CCSD_Driver(ChemEnv& chem_env, ExecutionContext& ec,
                                         const TiledIndexSpace& MO, Tensor<T>& t1, Tensor<T>& t2,
                                         Tensor<T>& f1, cholesky_2e::V2Tensors<T>& v2tensors,
                                         IndexVector& occ_int_vec, IndexVector& virt_int_vec,
                                         const int pos, std::stringstream& qfstr,
                                         QflowTimers& qflow_timers) {
  Scheduler   sch{ec};
  ExecutionHW ex_hw = ec.exhw();
  const auto  rank  = ec.pg().rank();

  SystemData& sys_data = chem_env.sys_data;
  // const size_t nelectrons_alpha = sys_data.nelectrons_alpha;
  const size_t nactoa     = chem_env.ioptions.ccsd_options.nactive_oa;
  const size_t nactob     = chem_env.ioptions.ccsd_options.nactive_ob;
  const size_t nactva     = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb     = chem_env.ioptions.ccsd_options.nactive_vb;
  const int    ducc_lvl   = chem_env.ioptions.ccsd_options.ducc_lvl;
  const bool   noprint    = chem_env.ioptions.ccsd_options.noprint;
  const bool   ducc_print = rank == 0 && !noprint;

  const TiledIndexSpace& O  = MO("occ");
  const TiledIndexSpace& Oi = MO("occ_int");
  // const TiledIndexSpace& V  = MO("virt");
  const TiledIndexSpace& Vi = MO("virt_int");
  // const TiledIndexSpace& N = MO("all");
  // const TiledIndexSpace& Vai = MO("virt_alpha_int");
  // const TiledIndexSpace& Vbi = MO("virt_beta_int");
  // const TiledIndexSpace& Vae = MO("virt_alpha_ext");
  // const TiledIndexSpace& Vbe = MO("virt_beta_ext");

  auto [z1, z2, z3, z4]     = MO.labels<4>("all");
  auto [h1, h2, h3, h4]     = MO.labels<4>("occ");
  auto [p1, p2, p3, p4]     = MO.labels<4>("virt");
  auto [h1i, h2i, h3i, h4i] = MO.labels<4>("occ_int");
  // auto [h1e, h2e, h3e, h4e] = MO.labels<4>("occ_ext");
  auto [p1i, p2i, p3i, p4i] = MO.labels<4>("virt_int");
  // auto [p1ai,p2ai] = MO.labels<2>("virt_alpha_int");
  // auto [p1bi,p2bi] = MO.labels<2>("virt_beta_int");

  std::cout.precision(15);

  std::string files_dir    = chem_env.get_files_dir("", "ducc");
  std::string files_prefix = chem_env.get_files_prefix("", "ducc");
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string ftij_file   = files_prefix + ".ftij";
  std::string ftia_file   = files_prefix + ".ftia";
  std::string ftab_file   = files_prefix + ".ftab";
  std::string vtijkl_file = files_prefix + ".vtijkl";
  std::string vtijka_file = files_prefix + ".vtijka";
  std::string vtaijb_file = files_prefix + ".vtaijb";
  std::string vtijab_file = files_prefix + ".vtijab";
  std::string vtiabc_file = files_prefix + ".vtiabc";
  std::string vtabcd_file = files_prefix + ".vtabcd";

  if(ducc_print) {
    std::cout << std::endl << "Executing DUCC routine" << std::endl;
    std::cout << "======================" << std::endl;
  }

  // COMPUTE <H>
  Tensor<T> deltaoo{{O, O}, {1, 1}};
  sch.allocate(deltaoo).execute();
  sch(deltaoo() = 0).execute();
  init_diagonal(ec, deltaoo());

  Tensor<T> adj_scalar{};
  Tensor<T> total_shift{};
  Tensor<T> oei{{O, O}, {1, 1}};
  sch.allocate(adj_scalar, total_shift, oei).execute();

  // clang-format off
  sch(oei(h1, h2)  = f1(h1, h2))
     (oei(h1, h2) += -0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h3, h1, h4, h2))
     (oei(h1, h2) += -0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h1, h3, h2, h4))
     (oei(h1, h2) += 0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h1, h3, h4, h2))
     (oei(h1, h2) += 0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h3, h1, h2, h4))
     .execute(ex_hw);
  // clang-format on

  // clang-format off
  sch(oei(h1, h2) += 0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h3, h1, h4, h2))
     (oei(h1, h2) += -0.25 * deltaoo(h3, h4) * v2tensors.v2ijkl(h1, h3, h4, h2))
     (adj_scalar() = deltaoo(h1, h2) * oei(h1, h2))
     .execute(ex_hw);
  // clang-format on

  auto rep_energy      = chem_env.scf_context.nuc_repl_energy;
  auto full_scf_energy = chem_env.scf_context.hf_energy;
  auto bare_energy     = get_scalar(adj_scalar) + rep_energy;
  auto core_energy     = full_scf_energy - bare_energy;

  if(ducc_print) {
    std::cout << "Number of active occupied alpha = " << nactoa << std::endl;
    std::cout << "Number of active occupied beta  = " << nactob << std::endl;
    std::cout << "Number of active virtual alpha  = " << nactva << std::endl;
    std::cout << "Number of active virtual beta   = " << nactvb << std::endl;
    std::cout << "ducc_lvl = " << ducc_lvl << std::endl;
    std::cout << std::endl
              << "Full SCF Energy: " << std::setprecision(12) << full_scf_energy << std::endl;
    std::cout << "Bare SCF Energy: " << std::setprecision(12) << bare_energy << std::endl;
    std::cout << "Frozen Core Energy: " << std::setprecision(12) << core_energy << std::endl;
  }

  // Initialize shift with the full SCF energy and repulsion energy.
  // clang-format off
  sch(total_shift() = full_scf_energy)
     (total_shift() -= rep_energy)
     .execute(ex_hw);
  // clang-format on

  // Allocate the transformed arrays
  Tensor<T> ftij{{Oi, Oi}, {1, 1}};
  Tensor<T> ftia{{Oi, Vi}, {1, 1}};
  Tensor<T> ftab{{Vi, Vi}, {1, 1}};
  Tensor<T> vtijkl{{Oi, Oi, Oi, Oi}, {2, 2}};
  Tensor<T> vtijka{{Oi, Oi, Oi, Vi}, {2, 2}};
  Tensor<T> vtaijb{{Vi, Oi, Oi, Vi}, {2, 2}};
  Tensor<T> vtijab{{Oi, Oi, Vi, Vi}, {2, 2}};
  Tensor<T> vtiabc{{Oi, Vi, Vi, Vi}, {2, 2}};
  Tensor<T> vtabcd{{Vi, Vi, Vi, Vi}, {2, 2}};
  sch.allocate(ftij, vtijkl).execute();
  if(nactva > 0) { sch.allocate(ftia, ftab, vtijka, vtaijb, vtijab, vtiabc, vtabcd).execute(); }

  // Zero the transformed arrays
  // clang-format off
  sch(ftij(h1i, h2i) = 0)
     (vtijkl(h1i, h2i, h3i, h4i) = 0).execute();
  // clang-format on
  if(nactva > 0) {
    // clang-format off
    sch (ftia(h1i,p1i) = 0)
        (ftab(p1i,p2i) = 0)
        (vtijka(h1i,h2i,h3i,p1i) = 0)
        (vtaijb(p1i,h1i,h2i,p2i) = 0)
        (vtijab(h1i,h2i,p1i,p2i) = 0)
        (vtiabc(h1i,p1i,p2i,p3i) = 0)
        (vtabcd(p1i,p2i,p3i,p4i) = 0).execute();
    // clang-format on
  }

  // Print norms of original T1/T2 operators
  const auto t1_full_norm = tamm::norm(t1);
  const auto t2_full_norm = tamm::norm(t2);

  // Zero T_int components
  // This means that T1 and T2 are no longer the full T1 and T2.
  // clang-format off
  sch(t1(p1i, h1i) = 0)
     (t2(p1i, p2i, h1i, h2i) = 0).execute();
  // clang-format on

  const auto t1_ext_norm = tamm::norm(t1);
  const auto t2_ext_norm = tamm::norm(t2);

  auto [maxt1ext, maxblockid1, maxcoord1] = tamm::max_element(t1);
  auto [mint1ext, minblockid1, mincoord1] = tamm::min_element(t1);
  const auto maxabst1ext                  = std::max(std::abs(maxt1ext), std::abs(mint1ext));
  auto [maxt2ext, maxblockid2, maxcoord2] =
    tamm::max_element(t2); // T2 is antisymmetric so max and min are the same.

  const bool is_qflow = chem_env.ioptions.task_options.ducc.second == "qflow";

  if(is_qflow && ducc_lvl == 0) {
    qfstr << "Norm of T1 (Full): " << t1_full_norm << std::endl;
    qfstr << "Norm of T2 (Full): " << t2_full_norm << std::endl;
    qfstr << "Norm of T1 (Ext): " << t1_ext_norm << std::endl;
    qfstr << "Norm of T2 (Ext): " << t2_ext_norm << std::endl;
    qfstr << "Max abs of T1 (Ext): " << maxabst1ext << std::endl;
    qfstr << "Max abs of T2 (Ext): " << maxt2ext << std::endl;
    qfstr << "Norm of T1 (Int): " << sqrt(t1_full_norm * t1_full_norm - t1_ext_norm * t1_ext_norm)
          << std::endl;
    qfstr << "Norm of T2 (Int): " << sqrt(t2_full_norm * t2_full_norm - t2_ext_norm * t2_ext_norm)
          << std::endl;
  }

  if(ducc_print) {
    std::cout << "Norm of T1 (Full): " << t1_full_norm << std::endl;
    std::cout << "Norm of T2 (Full): " << t2_full_norm << std::endl;
    std::cout << "Norm of T1 (Ext): " << t1_ext_norm << std::endl;
    std::cout << "Norm of T2 (Ext): " << t2_ext_norm << std::endl;
    std::cout << "Max abs of T1 (Ext): " << maxabst1ext << std::endl;
    std::cout << "Max abs of T2 (Ext): " << maxt2ext << std::endl;
    std::cout << "Norm of T1 (Int): "
              << sqrt(t1_full_norm * t1_full_norm - t1_ext_norm * t1_ext_norm) << std::endl;
    std::cout << "Norm of T2 (Int): "
              << sqrt(t2_full_norm * t2_full_norm - t2_ext_norm * t2_ext_norm) << std::endl;
  }

  // Tensors to read/write from/to disk
  std::vector<Tensor<T>>   ducc_tensors = {ftij,   ftia,   ftab,   vtijkl, vtijka,
                                           vtaijb, vtijab, vtiabc, vtabcd};
  std::vector<std::string> dt_files     = {ftij_file,   ftia_file,   ftab_file,
                                           vtijkl_file, vtijka_file, vtaijb_file,
                                           vtijab_file, vtiabc_file, vtabcd_file};

  bool drestart       = chem_env.ioptions.ccsd_options.writet;
  bool dtensors_exist = ducc_tensors_exist(ducc_tensors, dt_files, drestart);

  // TODO: Extend this logic to new active space variables
  if(drestart && dtensors_exist) {
    const size_t orig_nactva = chem_env.run_context["ducc"]["nactive_va"];
    if(orig_nactva != nactva) { dtensors_exist = false; }
    const int orig_ducc_lvl = chem_env.run_context["ducc"]["ducc_lvl"];
    if(orig_ducc_lvl > ducc_lvl) { dtensors_exist = false; }
    if(!(drestart && dtensors_exist)) { reset_ducc_runcontext(ec, chem_env); }
  }
  else { reset_ducc_runcontext(ec, chem_env); }

  chem_env.run_context["ducc"]["ducc_lvl"] = ducc_lvl;

  const bool is_l0_done = chem_env.run_context["ducc"]["level0"];
  const bool is_l1_done = chem_env.run_context["ducc"]["level1"];
  const bool is_l2_done = chem_env.run_context["ducc"]["level2"];

  auto cc_t1 = std::chrono::high_resolution_clock::now();
  // Bare Hamiltonian
  if(drestart && dtensors_exist && is_l0_done) {
    if(!is_l1_done && !is_l2_done) {
      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 0, drestart, true);
      sch(total_shift() = chem_env.run_context["ducc"]["total_shift"]).execute();
    }
  }
  else {
    H_0(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1,
        v2tensors, ex_hw);
    ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 0, drestart);
    chem_env.run_context["ducc"]["level0"]      = true;
    chem_env.run_context["ducc"]["level1"]      = false;
    chem_env.run_context["ducc"]["level2"]      = false;
    chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
    if(ducc_print) chem_env.write_run_context();
  }
  // TODO Setup a print statement here.
  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ducc_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(ducc_print)
    std::cout << std::endl
              << "DUCC: Time taken to compute Bare Hamiltonian: " << std::fixed
              << std::setprecision(2) << ducc_time << " secs" << std::endl;

  double ducc_total_time = ducc_time;

  Tensor<T> eob_temp{{O, O}, {1, 1}};
  sch.allocate(eob_temp).execute();

  // clang-format off
  sch(adj_scalar() = 0.0)
    (adj_scalar() += 1.0 * deltaoo(h1i, h2i) * ftij(h1i, h2i))
    (eob_temp(h1i, h2i) = 1.0 * deltaoo(h3i, h4i) * vtijkl(h3i, h1i, h4i, h2i))
    (adj_scalar() -= 0.25 * deltaoo(h1i, h2i) * eob_temp(h1i, h2i))
    (eob_temp(h1i, h2i) = 1.0 * deltaoo(h3i, h4i) * vtijkl(h3i, h1i, h2i, h4i))
    (adj_scalar() -= -0.25 * deltaoo(h1i, h2i) * eob_temp(h1i, h2i))
    .execute(ex_hw);
  // clang-format on

  const auto bas_energy = get_scalar(adj_scalar) + rep_energy;
  if(ducc_print) {
    std::cout << "Bare Active Space SCF Energy: " << std::setprecision(12) << bas_energy
              << std::endl;
  }

  // Level 1
  if(ducc_lvl > 0) {
    cc_t1 = std::chrono::high_resolution_clock::now();

    if(dtensors_exist && is_l1_done) {
      if(!is_l2_done) {
        ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 1, drestart, true);
        sch(total_shift() = chem_env.run_context["ducc"]["total_shift"]).execute();
      }
    }
    else {
      F_1(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1,
          t1, t2, ex_hw);

      // clang-format off
      sch(adj_scalar() = 0.0)
         (adj_scalar() += 1.0 * f1(h1, p2) * t1(p2, h1))
         (adj_scalar() += 1.0 * t1(p2, h1) * f1(h1, p2))
         (total_shift() += adj_scalar())
         .execute();
      // clang-format on

      V_1(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd,
          v2tensors, t1, t2, ex_hw);

      // clang-format off
      sch(adj_scalar() = 0.0)
         (adj_scalar() += (1.0 / 4.0) * v2tensors.v2ijab(h1, h3, p2, p4) * t2(p2, p4, h1, h3))
         (adj_scalar() += (1.0 / 4.0) * t2(p4, p2, h3, h1) * v2tensors.v2ijab(h1, h3, p2, p4))
         (total_shift() += adj_scalar())
         .execute();
      // clang-format on

      sch(adj_scalar() = 0.0).execute();
      F_2(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1,
          t1, t2, adj_scalar, ex_hw);
      sch(total_shift() += adj_scalar()).execute();

      // energy = get_scalar(adj_scalar);
      // if(ducc_print)std::cout << "FC F2: " << std::setprecision(12) << energy << std::endl;

      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 1, drestart);

      chem_env.run_context["ducc"]["level1"]      = true;
      chem_env.run_context["ducc"]["level2"]      = false;
      chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
      if(ducc_print) chem_env.write_run_context();
    }
    cc_t2     = std::chrono::high_resolution_clock::now();
    ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    ducc_total_time += ducc_time;
    if(ducc_print)
      std::cout
        << std::endl
        << "DUCC: Time taken to compute single commutator of F and V and double commutator of F: "
        << std::fixed << std::setprecision(2) << ducc_time << " secs" << std::endl;
  }

  // Level 2
  if(ducc_lvl > 1) {
    cc_t1 = std::chrono::high_resolution_clock::now();
    if(drestart && dtensors_exist && is_l2_done) {
      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 2, drestart, true);
      sch(total_shift() = chem_env.run_context["ducc"]["total_shift"]).execute();
    }
    else {
      sch(adj_scalar() = 0.0).execute();
      V_2(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd,
          v2tensors, t1, t2, adj_scalar, ex_hw);
      sch(total_shift() += adj_scalar()).execute();

      // energy = get_scalar(adj_scalar);
      // if(ducc_print)std::cout << "FC V2: " << std::setprecision(12) << energy << std::endl;

      // TODO: Enable when we want to do partial restarts within this level
      // ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 2, drestart);

      sch(adj_scalar() = 0.0).execute();
      F_3(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1,
          t1, t2, adj_scalar, ex_hw);
      sch(total_shift() += adj_scalar()).execute();

      // energy = get_scalar(adj_scalar);
      // if(ducc_print)std::cout << "FC F3: " << std::setprecision(12) << energy << std::endl;

      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 2, drestart);
      chem_env.run_context["ducc"]["level2"]      = true;
      chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
      if(ducc_print) chem_env.write_run_context();
    }
    cc_t2     = std::chrono::high_resolution_clock::now();
    ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    ducc_total_time += ducc_time;
    if(ducc_print)
      std::cout << std::endl
                << "DUCC: Time taken to compute double commutator of V and triple commutator of F: "
                << std::fixed << std::setprecision(2) << ducc_time << " secs" << std::endl;
  }

  cc_t1 = std::chrono::high_resolution_clock::now();

  const auto fc_scalar = get_scalar(total_shift) - full_scf_energy + rep_energy;
  if(ducc_print) {
    std::cout << "Fully Contracted Scalar: " << std::setprecision(12) << fc_scalar << std::endl;
  }

  // Transform ft from Fock operator to one-electron operator.
  // clang-format off
  if (nactva > 0) {
    sch(ftab(p1i,p2i) +=  1.0 * deltaoo(h1i,h2i) * vtaijb(p1i,h1i,h2i,p2i))
       (ftia(h3i,p1i) += -0.5 * deltaoo(h1i,h2i) * vtijka(h1i,h3i,h2i,p1i))
       (ftia(h3i,p1i) +=  0.5 * deltaoo(h1i,h2i) * vtijka(h3i,h1i,h2i,p1i))
       (ftij(h3i,h4i) += -0.25 * deltaoo(h1i,h2i) * vtijkl(h3i,h1i,h4i,h2i))
       (ftij(h3i,h4i) +=  0.25 * deltaoo(h1i,h2i) * vtijkl(h3i,h1i,h2i,h4i))
       (ftij(h3i,h4i) +=  0.25 * deltaoo(h1i,h2i) * vtijkl(h1i,h3i,h4i,h2i))
       (ftij(h3i,h4i) += -0.25 * deltaoo(h1i,h2i) * vtijkl(h1i,h3i,h2i,h4i))
      //  .deallocate(deltaoo).execute(ex_hw);
       .execute(ex_hw);
  } else {
    sch(ftij(h3i,h4i)   += -0.25 * deltaoo(h1i,h2i) * vtijkl(h3i,h1i,h4i,h2i))
       (ftij(h3i,h4i)   +=  0.25 * deltaoo(h1i,h2i) * vtijkl(h3i,h1i,h2i,h4i))
       (ftij(h3i,h4i)   +=  0.25 * deltaoo(h1i,h2i) * vtijkl(h1i,h3i,h4i,h2i))
       (ftij(h3i,h4i)   += -0.25 * deltaoo(h1i,h2i) * vtijkl(h1i,h3i,h2i,h4i))
      //  .deallocate(deltaoo).execute(ex_hw);
       .execute(ex_hw);
  }
  // clang-format on

  // clang-format off
  sch(adj_scalar() = 0.0)
    (adj_scalar() += 1.0 * deltaoo(h1i, h2i) * ftij(h1i, h2i))
    (eob_temp(h1i, h2i) = 1.0 * deltaoo(h3i, h4i) * vtijkl(h3i, h1i, h4i, h2i))
    (adj_scalar() += 0.25 * deltaoo(h1i, h2i) * eob_temp(h1i, h2i))
    (eob_temp(h1i, h2i) = 1.0 * deltaoo(h3i, h4i) * vtijkl(h3i, h1i, h2i, h4i))
    (adj_scalar() += -0.25 * deltaoo(h1i, h2i) * eob_temp(h1i, h2i))
    (total_shift() -= adj_scalar())
    .deallocate(deltaoo, eob_temp)
    .execute(ex_hw);
  // clang-format on

  cc_t2     = std::chrono::high_resolution_clock::now();
  ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  ducc_total_time += ducc_time;
  qflow_timers.ducc_time += ducc_total_time;

  auto new_energy = get_scalar(adj_scalar);
  auto shift      = get_scalar(total_shift);

  // Total Shift = SCF-Bare-AS + Fully Contracted Terms - SCF-DUCC-AS + Froz. Core + Out-of-AS SCF
  // Energy Full SCF = SCF-Bare-AS + Froz. Core + Out-of-AS SCF Energy Total Shift = Full SCF +
  // Fully Contracted Terms - SCF-DUCC-AS
  if(ducc_print && ducc_lvl > 0) {
    std::cout << std::endl
              << "DUCC SCF energy: " << std::setprecision(12) << new_energy + rep_energy
              << std::endl;
    std::cout << "Total Energy Shift: " << std::setprecision(12) << shift << std::endl;
  }

  if(ducc_print) {
    sys_data.results["output"]["DUCC"]["performance"]["total_time"] = ducc_total_time;
    std::cout << std::endl
              << "DUCC: Total compute time: " << std::fixed << std::setprecision(2)
              << ducc_total_time << " secs" << std::endl
              << std::endl;
  }

  // PRINT STATEMENTS
  // X1-X9 below are only used for the default (non-QFlow) DUCC task
  if(chem_env.ioptions.task_options.ducc.second != "qflow") {
    cc_t1 = std::chrono::high_resolution_clock::now();
    print_ducc_results(chem_env, ec, files_prefix, ducc_print, nactoa, nactva, ftij, ftia, ftab,
                       vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd);
    cc_t2     = std::chrono::high_resolution_clock::now();
    ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(ducc_print)
      std::cout << "DUCC: Time to write results: " << std::fixed << std::setprecision(2)
                << ducc_time << " secs" << std::endl;
  } // task_options.ducc.second != "qflow"

// qflow
#if defined(USE_NWQSIM)
  if(chem_env.ioptions.task_options.ducc.second == "qflow" && ducc_lvl != 0)
    DUCC_T_QFLOW_Driver(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc,
                        vtabcd, ex_hw, shift, occ_int_vec, virt_int_vec, pos, qfstr, qflow_timers);
#endif
  free_tensors(ftij, vtijkl, adj_scalar, total_shift, oei);
  if(nactva > 0) { free_tensors(ftia, ftab, vtijka, vtaijb, vtijab, vtiabc, vtabcd); }

  if(ducc_print) chem_env.write_json_data();
}

} // namespace exachem::cc::ducc::internal

// Explicit template instantiation for DUCCInternal class
template class exachem::cc::ducc::internal::DUCCInternal<double>;
