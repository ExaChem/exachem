/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"

namespace exachem::cc::ducc {
using namespace exachem::cc::ducc::internal;

template<typename T>
void DUCC_T_CCSD_Driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                        Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1,
                        cholesky_2e::V2Tensors<T>& v2tensors, IndexVector& occ_int_vec,
                        IndexVector& virt_int_vec, string& pos_str) {
  Scheduler   sch{ec};
  ExecutionHW ex_hw = ec.exhw();

  SystemData& sys_data = chem_env.sys_data;
  // const size_t nelectrons_alpha = sys_data.nelectrons_alpha;
  const size_t nactoa   = chem_env.ioptions.ccsd_options.nactive_oa;
  const size_t nactob   = chem_env.ioptions.ccsd_options.nactive_ob;
  const size_t nactva   = chem_env.ioptions.ccsd_options.nactive_va;
  const size_t nactvb   = chem_env.ioptions.ccsd_options.nactive_vb;
  const int    ducc_lvl = chem_env.ioptions.ccsd_options.ducc_lvl;

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

  const auto rank = ec.pg().rank();
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

  if(rank == 0) {
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

  if(rank == 0) {
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

  // Zero T_int components
  // This means that T1 and T2 are no longer the full T1 and T2.
  // clang-format off
  sch(t1(p1i, h1i) = 0)
     (t2(p1i, p2i, h1i, h2i) = 0).execute();
  // clang-format off

  // Tensors to read/write from/to disk
  std::vector<Tensor<T>>   ducc_tensors = {ftij,   ftia,   ftab,   vtijkl, vtijka,
                                           vtaijb, vtijab, vtiabc, vtabcd};
  std::vector<std::string> dt_files     = {ftij_file,   ftia_file,   ftab_file,
                                           vtijkl_file, vtijka_file, vtaijb_file,
                                           vtijab_file, vtiabc_file, vtabcd_file};

  bool      drestart       = chem_env.ioptions.ccsd_options.writet;
  bool      dtensors_exist = ducc_tensors_exist(ducc_tensors, dt_files, drestart);

  // TODO: Extend this logic to new active space variables
  if(drestart && dtensors_exist) {
    const size_t orig_nactva = chem_env.run_context["ducc"]["nactive_va"];
    if(orig_nactva != nactva) { dtensors_exist = false; }
    const int orig_ducc_lvl = chem_env.run_context["ducc"]["ducc_lvl"];
    if(orig_ducc_lvl > ducc_lvl) { dtensors_exist = false; }
    if(!(drestart && dtensors_exist)) { exachem::cc::ducc::internal::reset_ducc_runcontext(ec, chem_env); }
  }
  else { exachem::cc::ducc::internal::reset_ducc_runcontext(ec, chem_env); }

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
    H_0(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1, v2tensors, ex_hw);
    ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 0, drestart);
    chem_env.run_context["ducc"]["level0"]      = true;
    chem_env.run_context["ducc"]["level1"]      = false;
    chem_env.run_context["ducc"]["level2"]      = false;
    chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
    if(ec.print()) chem_env.write_run_context();
  }
  // TODO Setup a print statement here.
  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ducc_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
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

  if(rank == 0) {
    std::cout << "Bare Active Space SCF Energy: " << std::setprecision(12)
              << get_scalar(adj_scalar) + rep_energy << std::endl;
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
      // if(rank == 0)std::cout << "FC F2: " << std::setprecision(12) << energy << std::endl;

      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 1, drestart);

      chem_env.run_context["ducc"]["level1"]      = true;
      chem_env.run_context["ducc"]["level2"]      = false;
      chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
      if(ec.print()) chem_env.write_run_context();
    }
    cc_t2     = std::chrono::high_resolution_clock::now();
    ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    ducc_total_time += ducc_time;
    if(rank == 0)
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
      // if(rank == 0)std::cout << "FC V2: " << std::setprecision(12) << energy << std::endl;

      // TODO: Enable when we want to do partial restarts within this level
      // ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 2, drestart);

      sch(adj_scalar() = 0.0).execute();
      F_3(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc, vtabcd, f1,
          t1, t2, adj_scalar, ex_hw);
      sch(total_shift() += adj_scalar()).execute();

      // energy = get_scalar(adj_scalar);
      // if(rank == 0)std::cout << "FC F3: " << std::setprecision(12) << energy << std::endl;

      ducc_tensors_io(ec, chem_env, ducc_tensors, dt_files, 2, drestart);
      chem_env.run_context["ducc"]["level2"]      = true;
      chem_env.run_context["ducc"]["total_shift"] = get_scalar(total_shift);
      if(ec.print()) chem_env.write_run_context();
    }
    cc_t2     = std::chrono::high_resolution_clock::now();
    ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    ducc_total_time += ducc_time;
    if(rank == 0)
      std::cout << std::endl
                << "DUCC: Time taken to compute double commutator of V and triple commutator of F: "
                << std::fixed << std::setprecision(2) << ducc_time << " secs" << std::endl;
  }

  if(rank == 0) {
    std::cout << "Fully Contracted Scalar: " << std::setprecision(12)
              << get_scalar(total_shift) - full_scf_energy + rep_energy << std::endl;
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
    .deallocate(deltaoo)
    .execute(ex_hw);
  // clang-format on

  auto new_energy = get_scalar(adj_scalar);
  auto shift      = get_scalar(total_shift);

  // Total Shift = SCF-Bare-AS + Fully Contracted Terms - SCF-DUCC-AS + Froz. Core + Out-of-AS SCF
  // Energy Full SCF = SCF-Bare-AS + Froz. Core + Out-of-AS SCF Energy Total Shift = Full SCF +
  // Fully Contracted Terms - SCF-DUCC-AS
  if(rank == 0 && ducc_lvl > 0) {
    std::cout << std::endl
              << "DUCC SCF energy: " << std::setprecision(12) << new_energy + rep_energy
              << std::endl;
    std::cout << "Total Energy Shift: " << std::setprecision(12) << shift << std::endl;
  }

  if(rank == 0) {
    sys_data.results["output"]["DUCC"]["performance"]["total_time"] = ducc_total_time;
    std::cout << std::endl
              << "DUCC: Total compute time: " << std::fixed << std::setprecision(2)
              << ducc_total_time << " secs" << std::endl
              << std::endl;
  }

  // PRINT STATEMENTS
  // TODO: Everything is assuming closed shell. For open shell calculations,
  //       formats starting from the tensor contractions to printing must be reconsidered.
  cc_t1 = std::chrono::high_resolution_clock::now();
  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense,
                            MemoryManagerKind::ga}; // create ec_dense once
  // const auto       nelectrons       = sys_data.nelectrons;
  auto print_blockstr = [](std::string filename, std::string val, bool append = false) {
    if(!filename.empty()) {
      std::ofstream tos;
      if(append) tos.open(filename + ".txt", std::ios::app);
      else tos.open(filename + ".txt", std::ios::out);
      if(!tos) std::cerr << "Error opening file " << filename << std::endl;
      tos << val << std::endl;
      tos.close();
    }
  };
  const std::string results_file = files_prefix + ".ducc.results";

  if(rank == 0) {
    print_blockstr(results_file, "Begin IJ Block");
    // std::cout << "Begin IJ Block" << std::endl;
  }

  Tensor<T>                                X1       = to_dense_tensor(ec_dense, ftij);
  std::function<bool(std::vector<size_t>)> dp_cond1 = [&](std::vector<size_t> cond) {
    if(cond[0] < nactoa && cond[1] < nactoa && cond[0] <= cond[1]) return true;
    // if(cond[0] < nelectrons_alpha && cond[1] < nelectrons_alpha && cond[0] <= cond[1]) return
    // true;
    return false;
  };
  print_dense_tensor(X1, dp_cond1, results_file, true);
  if(rank == 0) {
    print_blockstr(results_file, "End IJ Block", true);
    T first_val                                         = tamm::get_tensor_element(X1, {0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X1"] = first_val;
  }
  Tensor<T>::deallocate(X1);

  if(nactva > 0) {
    if(rank == 0) print_blockstr(results_file, "Begin IA Block", true);
    Tensor<T>                                X2       = to_dense_tensor(ec_dense, ftia);
    std::function<bool(std::vector<size_t>)> dp_cond2 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactoa && cond[1] < nactva) return true;
      return false;
    };
    print_dense_tensor(X2, dp_cond2, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IA Block", true);
      T first_val                                         = tamm::get_tensor_element(X2, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X2"] = first_val;
    }
    Tensor<T>::deallocate(X2);

    if(rank == 0) print_blockstr(results_file, "Begin AB Block", true);
    Tensor<T>                                X3       = to_dense_tensor(ec_dense, ftab);
    std::function<bool(std::vector<size_t>)> dp_cond3 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactva && cond[1] < nactva && cond[0] <= cond[1]) return true;
      return false;
    };
    print_dense_tensor(X3, dp_cond3, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End AB Block", true);
      T first_val                                         = tamm::get_tensor_element(X3, {0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X3"] = first_val;
    }
    Tensor<T>::deallocate(X3);
  }

  if(rank == 0) print_blockstr(results_file, "Begin IJKL Block", true);
  Tensor<T>                                X4       = to_dense_tensor(ec_dense, vtijkl);
  std::function<bool(std::vector<size_t>)> dp_cond4 = [&](std::vector<size_t> cond) {
    if(cond[0] < nactoa && cond[2] < nactoa && nactoa <= cond[1] && nactoa <= cond[3]) return true;
    return false;
  };
  print_dense_tensor(X4, dp_cond4, results_file, true);
  if(rank == 0) {
    print_blockstr(results_file, "End IJKL Block", true);
    T first_val = tamm::get_tensor_element(X4, {0, 0, 0, 0});
    sys_data.results["output"]["DUCC"]["results"]["X4"] = first_val;
  }
  Tensor<T>::deallocate(X4);

  if(nactva > 0) {
    if(rank == 0) print_blockstr(results_file, "Begin IJAB Block", true);
    Tensor<T>                                X5       = to_dense_tensor(ec_dense, vtijab);
    std::function<bool(std::vector<size_t>)> dp_cond5 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactoa && nactoa <= cond[1] && cond[2] < nactva && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X5, dp_cond5, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IJAB Block", true);
      T first_val = tamm::get_tensor_element(X5, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X5"] = first_val;
    }
    Tensor<T>::deallocate(X5);

    if(rank == 0) print_blockstr(results_file, "Begin ABCD Block", true);
    Tensor<T>                                X6       = to_dense_tensor(ec_dense, vtabcd);
    std::function<bool(std::vector<size_t>)> dp_cond6 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactva && cond[2] < nactva && nactva <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X6, dp_cond6, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End ABCD Block", true);
      T first_val = tamm::get_tensor_element(X6, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X6"] = first_val;
    }
    Tensor<T>::deallocate(X6);

    if(rank == 0) print_blockstr(results_file, "Begin AIJB Block", true);
    Tensor<T>                                X7         = to_dense_tensor(ec_dense, vtaijb);
    std::function<bool(std::vector<size_t>)> dp_cond7_1 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactva && cond[2] < nactoa && nactoa <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_1, results_file, true);

    std::function<bool(std::vector<size_t>)> dp_cond7_2 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactva && nactoa <= cond[1] && nactoa <= cond[2] && cond[3] < nactva)
        return true;
      return false;
    };
    print_dense_tensor(X7, dp_cond7_2, results_file, true);
    if(rank == 0) print_blockstr(results_file, "End AIJB Block", true);
    if(rank == 0) {
      T first_val = tamm::get_tensor_element(X7, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X7"] = first_val;
    }
    Tensor<T>::deallocate(X7);

    if(rank == 0) print_blockstr(results_file, "Begin IJKA Block", true);
    Tensor<T>                                X8       = to_dense_tensor(ec_dense, vtijka);
    std::function<bool(std::vector<size_t>)> dp_cond8 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactoa && cond[2] < nactoa && nactoa <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X8, dp_cond8, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IJKA Block", true);
      T first_val = tamm::get_tensor_element(X8, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X8"] = first_val;
    }
    Tensor<T>::deallocate(X8);

    if(rank == 0) print_blockstr(results_file, "Begin IABC Block", true);
    Tensor<T>                                X9       = to_dense_tensor(ec_dense, vtiabc);
    std::function<bool(std::vector<size_t>)> dp_cond9 = [&](std::vector<size_t> cond) {
      if(cond[0] < nactoa && cond[2] < nactva && nactva <= cond[1] && nactva <= cond[3])
        return true;
      return false;
    };
    print_dense_tensor(X9, dp_cond9, results_file, true);
    if(rank == 0) {
      print_blockstr(results_file, "End IABC Block", true);
      T first_val = tamm::get_tensor_element(X9, {0, 0, 0, 0});
      sys_data.results["output"]["DUCC"]["results"]["X9"] = first_val;
    }
    Tensor<T>::deallocate(X9);
  }

  cc_t2     = std::chrono::high_resolution_clock::now();
  ducc_time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << "DUCC: Time to write results: " << std::fixed << std::setprecision(2) << ducc_time
              << " secs" << std::endl;

// qflow
#if defined(USE_NWQSIM)
  if(chem_env.ioptions.task_options.ducc.second == "qflow")
    DUCC_T_QFLOW_Driver(sch, chem_env, MO, ftij, ftia, ftab, vtijkl, vtijka, vtaijb, vtijab, vtiabc,
                        vtabcd, ex_hw, shift, occ_int_vec, virt_int_vec, pos_str);
#endif
  free_tensors(ftij, vtijkl, adj_scalar, total_shift, oei);
  if(nactva > 0) { free_tensors(ftia, ftab, vtijka, vtaijb, vtijab, vtiabc, vtabcd); }

  if(rank == 0) chem_env.write_json_data();
}

using T = double;
template void DUCC_T_CCSD_Driver<T>(ChemEnv& chem_env, ExecutionContext& ec,
                                    const TiledIndexSpace& MO, Tensor<T>& t1, Tensor<T>& t2,
                                    Tensor<T>& f1, cholesky_2e::V2Tensors<T>& v2tensors,
                                    IndexVector& occ_int_vec, IndexVector& virt_int_vec,
                                    string& pos_str);
} // namespace exachem::cc::ducc
