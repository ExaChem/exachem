/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

/*
 * The code in this file is adapted from:
 * https://github.com/wavefunction91/MACIS/blob/master/tests/standalone_driver.cxx
 *
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See https://github.com/wavefunction91/MACIS/blob/master/LICENSE.txt for details
 */

#pragma once

#include <spdlog/cfg/env.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <iomanip>
#include <iostream>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/util/cas.hpp>
#include <macis/util/detail/rdm_files.hpp>
#include <macis/util/fcidump.hpp>
#include <macis/util/fock_matrices.hpp>
#include <macis/util/memory.hpp>
#include <macis/util/moller_plesset.hpp>
#include <macis/util/mpi.hpp>
#include <macis/util/transform.hpp>
#include <macis/wavefunction_io.hpp>
#include <map>
#include <sparsexx/io/write_dist_mm.hpp>

using macis::NumActive;
using macis::NumCanonicalOccupied;
using macis::NumCanonicalVirtual;
using macis::NumElectron;
using macis::NumInactive;
using macis::NumOrbital;
using macis::NumVirtual;

enum class Job { CI, MCSCF };

enum class CIExpansion { CAS, ASCI };

std::map<std::string, Job> job_map = {{"CI", Job::CI}, {"MCSCF", Job::MCSCF}};

std::map<std::string, CIExpansion> ci_exp_map = {{"CAS", CIExpansion::CAS},
                                                 {"ASCI", CIExpansion::ASCI}};

template<typename T>
T vec_sum(const std::vector<T>& x) {
  return std::accumulate(x.begin(), x.end(), T(0));
}

void macis_driver(ExecutionContext& ec, SystemData& sys_data, const std::string files_prefix) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  std::cout << std::scientific << std::setprecision(12);
  spdlog::cfg::load_env_levels();
  spdlog::set_pattern("[%n] %v");

  constexpr size_t nwfn_bits = 64;

  auto world_comm = ec.pg().comm();
  auto world_rank = macis::comm_rank(world_comm);
  auto world_size = macis::comm_size(world_comm);

  // Create Logger
  auto console = world_rank ? spdlog::null_logger_mt("CI") : spdlog::stdout_color_mt("CI");

  auto ci_options = sys_data.options_map.fci_options;

  // Required Keywords
  auto nalpha        = ci_options.nalpha;
  auto nbeta         = ci_options.nbeta;
  auto fcidump_fname = files_prefix + ".fcidump";

  if(nalpha != nbeta) tamm_terminate("INPUT FILE ERROR: [FCI] NALPHA != BETA");

  // Read FCIDUMP File
  size_t norb  = macis::read_fcidump_norb(fcidump_fname);
  size_t norb2 = norb * norb;
  size_t norb3 = norb2 * norb;
  size_t norb4 = norb2 * norb2;

  // console->info("norb = {}", norb);

  // XXX: Consider reading this into shared memory to avoid replication
  std::vector<double> T(norb2), V(norb4);
  auto                E_core = macis::read_fcidump_core(fcidump_fname);
  macis::read_fcidump_1body(fcidump_fname, T.data(), norb);
  macis::read_fcidump_2body(fcidump_fname, V.data(), norb);

  // Set up job
  std::string job_str = ci_options.job;
  Job         job;
  try {
    job = job_map.at(job_str);
  } catch(...) { tamm_terminate("INPUT FILE ERROR: [FCI] Job Not Recognized"); }

  std::string ciexp_str = ci_options.expansion;
  CIExpansion ci_exp;
  try {
    ci_exp = ci_exp_map.at(ciexp_str);
  } catch(...) { tamm_terminate("INPUT FILE ERROR: [FCI] CI Expansion Not Recognized"); }

  // Set up active space
  size_t n_inactive = ci_options.ninactive;

  if(n_inactive >= norb) tamm_terminate("INPUT FILE ERROR: [FCI] NINACTIVE >= NORB");

  size_t n_active = ci_options.nactive;
  if(n_active != norb - n_inactive)
    tamm_terminate("INPUT FILE ERROR: [FCI] NACTIVE != NORB - NINACTIVE");

  if(n_inactive + n_active > norb)
    tamm_terminate("INPUT FILE ERROR: [FCI] NINACTIVE + NACTIVE > NORB");

  size_t n_virtual = norb - n_active - n_inactive;

  // Misc optional files
  std::string rdm_fname = ""; // ci_options.rdm_fname
  std::string fci_out_fname{};

  if(n_active > nwfn_bits / 2) tamm_terminate("[FCI] Not Enough Bits");

  // MCSCF Settings
  macis::MCSCFSettings mcscf_settings;
  mcscf_settings.max_macro_iter     = ci_options.max_macro_iter;
  mcscf_settings.max_orbital_step   = ci_options.max_orbital_step;
  mcscf_settings.orb_grad_tol_mcscf = ci_options.orb_grad_tol_mcscf;
  mcscf_settings.enable_diis        = ci_options.enable_diis;
  mcscf_settings.diis_start_iter    = ci_options.diis_start_iter;
  mcscf_settings.diis_nkeep         = ci_options.diis_nkeep;
  mcscf_settings.ci_res_tol         = ci_options.ci_res_tol;
  mcscf_settings.ci_max_subspace    = ci_options.ci_max_subspace;
  mcscf_settings.ci_matel_tol       = ci_options.ci_matel_tol;

  // ASCI Settings
  macis::ASCISettings asci_settings;
  std::string         asci_wfn_fname, asci_wfn_out_fname;
  double              asci_E0         = 0.0;
  bool                compute_asci_E0 = true;

  bool mp2_guess = false;
  // OPT_KEYWORD("MCSCF.MP2_GUESS", mp2_guess, bool );

  if(ec.print()) {
    console->info("[Wavefunction Data]:");
    console->info("  * JOB     = {}", job_str);
    console->info("  * CIEXP   = {}", ciexp_str);
    console->info("  * FCIDUMP = {}", fcidump_fname);
    if(fci_out_fname.size()) console->info("  * FCIDUMP_OUT = {}", fci_out_fname);
    console->info("  * MP2_GUESS = {}", mp2_guess);

    console->debug("READ {} 1-body integrals and {} 2-body integrals", T.size(), V.size());
    console->info("ECORE  = {:.12f}", E_core);
    console->debug("TSUM  = {:.12f}", vec_sum(T));
    console->debug("VSUM  = {:.12f}", vec_sum(V));
    console->info("TMEM   = {:.2e} GiB", macis::to_gib(T));
    console->info("VMEM   = {:.2e} GiB", macis::to_gib(V));
  }

  // Setup printing
  bool print_davidson    = ci_options.print_davidson;
  bool print_ci          = ci_options.print_ci;
  bool print_mcscf       = ci_options.print_mcscf;
  bool print_diis        = ci_options.print_diis;
  bool print_asci_search = ci_options.print_asci_search;

  if(world_rank or not print_davidson) spdlog::null_logger_mt("davidson");
  if(world_rank or not print_ci) spdlog::null_logger_mt("ci_solver");
  if(world_rank or not print_mcscf) spdlog::null_logger_mt("mcscf");
  if(world_rank or not print_diis) spdlog::null_logger_mt("diis");
  if(world_rank or not print_asci_search) spdlog::null_logger_mt("asci_search");

  // MP2 Guess Orbitals
  if(mp2_guess) {
    console->info("Calculating MP2 Natural Orbitals");
    size_t nocc_canon = n_inactive + nalpha;
    size_t nvir_canon = norb - nocc_canon;

    // Compute MP2 Natural Orbitals
    std::vector<double> MP2_RDM(norb * norb, 0.0);
    std::vector<double> W_occ(norb);
    macis::mp2_natural_orbitals(NumOrbital(norb), NumCanonicalOccupied(nocc_canon),
                                NumCanonicalVirtual(nvir_canon), T.data(), norb, V.data(), norb,
                                W_occ.data(), MP2_RDM.data(), norb);

    // Transform Hamiltonian
    macis::two_index_transform(norb, norb, T.data(), norb, MP2_RDM.data(), norb, T.data(), norb);
    macis::four_index_transform(norb, norb, V.data(), norb, MP2_RDM.data(), norb, V.data(), norb);
  }

  // Copy integrals into active subsets
  std::vector<double> T_active(n_active * n_active);
  std::vector<double> V_active(n_active * n_active * n_active * n_active);

  // Compute active-space Hamiltonian and inactive Fock matrix
  std::vector<double> F_inactive(norb2);
  macis::active_hamiltonian(NumOrbital(norb), NumActive(n_active), NumInactive(n_inactive),
                            T.data(), norb, V.data(), norb, F_inactive.data(), norb,
                            T_active.data(), n_active, V_active.data(), n_active);

  console->debug("FINACTIVE_SUM = {:.12f}", vec_sum(F_inactive));
  console->debug("VACTIVE_SUM   = {:.12f}", vec_sum(V_active));
  console->debug("TACTIVE_SUM   = {:.12f}", vec_sum(T_active));

  // Compute Inactive energy
  auto E_inactive =
    macis::inactive_energy(NumInactive(n_inactive), T.data(), norb, F_inactive.data(), norb);
  console->info("E(inactive) = {:.12f}", E_inactive);

  // Storage for active RDMs
  std::vector<double> active_ordm(n_active * n_active);
  std::vector<double> active_trdm(active_ordm.size() * active_ordm.size());

  double E0 = 0;

  // CI
  if(job == Job::CI) {
    using generator_t = macis::DoubleLoopHamiltonianGenerator<nwfn_bits>;
    if(ci_exp == CIExpansion::CAS) {
      std::vector<double> C_local;
      // TODO: VERIFY MPI + CAS
      E0 = macis::CASRDMFunctor<generator_t>::rdms(
        mcscf_settings, NumOrbital(n_active), nalpha, nbeta, T_active.data(), V_active.data(),
        active_ordm.data(), active_trdm.data(), C_local, world_comm);
      E0 += E_inactive + E_core;

      if(ci_options.print_state_char.first) {
        auto det_logger = world_rank ? spdlog::null_logger_mt("determinants")
                                     : spdlog::stdout_color_mt("determinants");
        det_logger->info("Print leading determinants > {:.12f}",
                         ci_options.print_state_char.second);
        auto dets = macis::generate_hilbert_space<generator_t::nbits>(n_active, nalpha, nbeta);
        for(size_t i = 0; i < dets.size(); ++i) {
          if(std::abs(C_local[i]) > ci_options.print_state_char.second) {
            det_logger->info("{:>16.12f}   {}", C_local[i], macis::to_canonical_string(dets[i]));
          }
        }
      }
    }
    else {
      generator_t ham_gen(
        macis::matrix_span<double>(T_active.data(), n_active, n_active),
        macis::rank4_span<double>(V_active.data(), n_active, n_active, n_active, n_active));

      std::vector<macis::wfn_t<nwfn_bits>> dets;
      std::vector<double>                  C;
      if(asci_wfn_fname.size()) {
        // Read wave function from standard file
        console->info("Reading Guess Wavefunction From {}", asci_wfn_fname);
        macis::read_wavefunction(asci_wfn_fname, dets, C);
        // std::cout << dets[0].to_ullong() << std::endl;
        if(compute_asci_E0) {
          console->info("*  Calculating E0");
          E0 = 0;
          for(auto ii = 0; ii < dets.size(); ++ii) {
            double tmp = 0.0;
            for(auto jj = 0; jj < dets.size(); ++jj) {
              tmp += ham_gen.matrix_element(dets[ii], dets[jj]) * C[jj];
            }
            E0 += C[ii] * tmp;
          }
        }
        else {
          console->info("*  Reading E0");
          E0 = asci_E0 - E_core - E_inactive;
        }
      }
      else {
        // HF Guess
        console->info("Generating HF Guess for ASCI");
        dets = {macis::canonical_hf_determinant<nwfn_bits>(nalpha, nalpha)};
        // std::cout << dets[0].to_ullong() << std::endl;
        E0 = ham_gen.matrix_element(dets[0], dets[0]);
        C  = {1.0};
      }
      console->info("ASCI Guess Size = {}", dets.size());
      console->info("ASCI E0 = {:.10e}", E0 + E_core + E_inactive);

      auto asci_st          = hrt_t::now();
      std::tie(E0, dets, C) = macis::asci_grow(asci_settings, mcscf_settings, E0, std::move(dets),
                                               std::move(C), ham_gen, n_active, world_comm);
      if(asci_settings.max_refine_iter) {
        std::tie(E0, dets, C) = macis::asci_refine(asci_settings, mcscf_settings, E0,
                                                   std::move(dets), std::move(C), ham_gen, n_active,
                                                   world_comm);
      }
      E0 += E_inactive + E_core;
      auto  asci_en  = hrt_t::now();
      dur_t asci_dur = asci_en - asci_st;
      console->info("* ASCI_DUR = {:.2e} ms", asci_dur.count());

      if(asci_wfn_out_fname.size() and !world_rank) {
        console->info("Writing ASCI Wavefunction to {}", asci_wfn_out_fname);
        macis::write_wavefunction(asci_wfn_out_fname, n_active, dets, C);
      }

      // Dump Hamiltonian
      if(0) {
        auto H = macis::make_dist_csr_hamiltonian<int64_t>(world_comm, dets.begin(), dets.end(),
                                                           ham_gen, 1e-16);
        sparsexx::write_dist_mm("ham.mtx", H, 1);
      }
    }

    // MCSCF
  }
  else if(job == Job::MCSCF) {
    // Possibly read active RDMs
    if(rdm_fname.size()) {
      console->info("  * RDMFILE = {}", rdm_fname);
      std::vector<double> full_ordm(norb2), full_trdm(norb4);
      macis::read_rdms_binary(rdm_fname, norb, full_ordm.data(), norb, full_trdm.data(), norb);
      macis::active_submatrix_1body(NumActive(n_active), NumInactive(n_inactive), full_ordm.data(),
                                    norb, active_ordm.data(), n_active);
      macis::active_subtensor_2body(NumActive(n_active), NumInactive(n_inactive), full_trdm.data(),
                                    norb, active_trdm.data(), n_active);

      // Compute CI energy from RDMs
      double ERDM = blas::dot(active_ordm.size(), active_ordm.data(), 1, T_active.data(), 1);
      ERDM += blas::dot(active_trdm.size(), active_trdm.data(), 1, V_active.data(), 1);
      console->info("E(RDM)  = {:.12f} Eh", ERDM + E_inactive + E_core);
    }

    // CASSCF
    E0 = macis::casscf_diis(mcscf_settings, NumElectron(nalpha), NumElectron(nbeta),
                            NumOrbital(norb), NumInactive(n_inactive), NumActive(n_active),
                            NumVirtual(n_virtual), E_core, T.data(), norb, V.data(), norb,
                            active_ordm.data(), n_active, active_trdm.data(), n_active, world_comm);
  }

  console->info("E(CI)  = {:.12f} Eh", E0);

  if(fci_out_fname.size())
    macis::write_fcidump(fci_out_fname, norb, T.data(), norb, V.data(), norb, E_core);
}
