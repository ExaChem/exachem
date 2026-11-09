#include "exachem/embedding/embedding_driver.hpp"
#include "exachem/common/termcolor.hpp"

#include "exachem/mp2/cd_mp2.hpp"
#include "exachem/scf/scf_data.hpp"
#include "exachem/scf/scf_engine.hpp"
#include "exachem/scf/scf_guess.hpp"

#include "exachem/cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/fcidump.hpp"

#if defined(ENABLE_CC)
#include "exachem/cc/ccsd/canonical/ccsd_canonical.hpp"
#include "exachem/cc/ccsd/canonical/ccsd_cs.hpp"
#include "exachem/cc/ccsd/canonical/ccsd_os.hpp"
#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"
#include "exachem/cc/ccsd/qed/cd_qed_ccsd_cs.hpp"
#include "exachem/cc/ccsd_t/ccsd_t_fused_driver.hpp"
// #include "exachem/cc/ccsdt/cd_ccsdt_os.hpp"
#include "exachem/cc/cc2/cd_cc2.hpp"
#include "exachem/cc/ducc/ducc-t_ccsd.hpp"
#include "exachem/cc/eom/eomccsd_opt.hpp"
#include "exachem/cc/lambda/ccsd_lambda.hpp"
#endif

using exachem::scf::SCFCompute;
using exachem::scf::SCFData;
using exachem::scf::SCFEngine;
using exachem::scf::SCFGuess;
using exachem::scf::SCFIter;
using tamm::Tensor;
using T = double;

namespace exachem::embedding {

void embedding_driver(ExecutionContext& ec, ChemEnv& chem_env) { embedding(ec, chem_env); }

void embedding(ExecutionContext& ec, ChemEnv& chem_env) {
  auto       rank  = ec.pg().rank();
  const bool rank0 = rank == 0;

  // Print header
  std::string header(28, ' ');
  header += "EMBEDDING MODULE";
  if(rank0) std::cout << "\n" << header << "\n";

  // Get embedding options
  const std::string&      projector        = chem_env.ioptions.embedding_options.projector;
  const double            lambda           = chem_env.ioptions.embedding_options.lambda;
  const bool              freeze_projected = chem_env.ioptions.embedding_options.freeze_projected;
  std::vector<int>&       n_acc_occ_mos = chem_env.ioptions.embedding_options.nactive_occ_orbitals;
  std::vector<int>&       n_acc_vir_mos = chem_env.ioptions.embedding_options.nactive_vir_orbitals;
  const std::string&      partition     = chem_env.ioptions.embedding_options.partition;
  const std::vector<int>& active_atoms  = chem_env.ioptions.embedding_options.active_atoms;
  const std::unordered_map<std::string, std::unordered_set<std::string>> active_orbitals =
    chem_env.ioptions.embedding_options.active_orbitals;
  const double avas_occ_threshold = chem_env.ioptions.embedding_options.avas_occ_threshold;
  const double avas_vir_threshold = chem_env.ioptions.embedding_options.avas_vir_threshold;
  const std::vector<std::string>& high_level  = chem_env.ioptions.embedding_options.high_level;
  const bool                      huzinaga    = projector == "HUZINAGA";
  const double                    pao_thresh1 = chem_env.ioptions.embedding_options.pao_thresh1;
  const double                    pao_thresh2 = chem_env.ioptions.embedding_options.pao_thresh2;

  // Get energy of full system
  SCFEngine scf_engine(ec, chem_env);
  scf_engine.run(ec, chem_env);
  double energy_ab_low = chem_env.scf_context.hf_energy;

  const int                   N        = chem_env.shells.nbf();
  const int                   NMO      = chem_env.sys_data.nbf;
  std::vector<libint2::Atom>& atoms    = chem_env.atoms;
  SCFData&                    scf_data = scf_engine.scf_data;
  scf::TAMMTensors<T>&        ttensors = scf_data.ttensors;
  scf::EigenTensors&          etensors = scf_data.etensors;
  std::streambuf*             orig_buf = std::cout.rdbuf();

  // Compute Overlap and Core Hamiltonian
  std::cout.rdbuf(NULL);
  scf_engine.scf_compute.compute_shellpair_list(ec, chem_env.shells, scf_data);
  scf_engine.setup_tiled_index_space(ec, chem_env);
  scf_engine.scf_compute.compute_hamiltonian(ec, scf_data, chem_env, ttensors, etensors);
  std::cout.rdbuf(orig_buf);
  ec.pg().barrier();

  // Overlap Matrix Square Root
  Matrix S1_eigen(N, N);
  tamm::tamm_to_eigen_tensor(scf_data.ttensors.S1, S1_eigen);
  Matrix S12_eigen = sqrtm(S1_eigen);

  // Save C_AO matrix from scf_context and get occupied orbitals
  const bool          is_unrestricted  = chem_env.sys_data.is_unrestricted;
  const int           nelectrons_alpha = chem_env.sys_data.nelectrons_alpha;
  const int           nelectrons_beta  = chem_env.sys_data.nelectrons_beta;
  const int           nvirtual_alpha   = chem_env.sys_data.n_vir_alpha;
  const int           nvirtual_beta    = chem_env.sys_data.n_vir_beta;
  Matrix              scratch(N, NMO);
  std::vector<Matrix> C_occ;
  std::vector<Matrix> C_vir;
  tamm::tamm_to_eigen_tensor(chem_env.scf_context.C_AO, scratch);
  Matrix Sm1 = scratch * scratch.transpose();
  C_occ.push_back(scratch.leftCols(nelectrons_alpha));
  C_vir.push_back(scratch.rightCols(NMO - nelectrons_alpha));
  if(is_unrestricted) {
    tamm::tamm_to_eigen_tensor(chem_env.scf_context.C_beta_AO, scratch);
    C_occ.push_back(scratch.leftCols(nelectrons_beta));
    C_vir.push_back(scratch.rightCols(NMO - nelectrons_beta));
  }

  // Get active orbitals
  std::vector<int> indicesToKeep;
  auto             atom2shell = chem_env.shells.atom2shell(atoms);
  auto             shell2bf   = chem_env.shells.shell2bf();
  auto             shell_type = [](size_t nbf) -> std::string {
    switch(nbf) {
      case 1: return "s";
      case 3: return "p";
      case 5: return "d";
      case 7: return "f";
      default: return "";
    }
  };
  // Select indices for the active orbitals
  if(!active_orbitals.empty()) {
    for(const auto& [iatom_str, target_shells]: active_orbitals) {
      int iatom = std::stoi(iatom_str);
      for(auto ish: atom2shell[iatom]) {
        auto&       shell = chem_env.shells[ish];
        std::string type  = shell_type(shell.size());
        if(!target_shells.contains(type)) { continue; }
        for(size_t ibf = shell2bf[ish]; ibf < shell2bf[ish] + shell.size(); ++ibf) {
          indicesToKeep.push_back(ibf);
        }
      }
    }
  }
  else { // Use active_atoms
    for(auto iatom: active_atoms) {
      for(auto ish: atom2shell[iatom]) {
        for(size_t ibf = shell2bf[ish]; ibf < shell2bf[ish] + chem_env.shells[ish].size(); ibf++) {
          indicesToKeep.push_back(ibf);
        }
      }
    }
  }

  if(n_acc_occ_mos.empty()) {
    n_acc_occ_mos.push_back(0);
    if(is_unrestricted) n_acc_occ_mos.push_back(0);
  }
  if(n_acc_vir_mos.empty()) {
    n_acc_vir_mos.push_back(0);
    if(is_unrestricted) n_acc_vir_mos.push_back(0);
  }
  std::vector<Matrix> C_occ_active;
  std::vector<Matrix> C_uno_active;
  if(partition == "AVAS") {
    C_occ_active.push_back(
      avas(S1_eigen, C_occ[0], indicesToKeep, avas_occ_threshold, n_acc_occ_mos[0], rank0));
    C_uno_active.push_back(
      avas(S1_eigen, C_vir[0], indicesToKeep, avas_vir_threshold, n_acc_vir_mos[0], rank0));
    if(is_unrestricted) {
      C_occ_active.push_back(
        avas(S1_eigen, C_occ[1], indicesToKeep, avas_occ_threshold, n_acc_occ_mos[1], rank0));
      C_uno_active.push_back(
        avas(S1_eigen, C_vir[1], indicesToKeep, avas_vir_threshold, n_acc_vir_mos[1], rank0));
    }
  }
  else if(partition == "SPADE") {
    C_occ_active.push_back(spade(S12_eigen, C_occ[0], indicesToKeep, n_acc_occ_mos[0], rank0));
    C_uno_active.push_back(
      paos(S1_eigen, C_occ[0], indicesToKeep, rank0, pao_thresh1, pao_thresh2));
    if(is_unrestricted) {
      C_occ_active.push_back(spade(S12_eigen, C_occ[1], indicesToKeep, n_acc_occ_mos[1], rank0));
      C_uno_active.push_back(
        paos(S1_eigen, C_occ[1], indicesToKeep, rank0, pao_thresh1, pao_thresh2));
    }
  }

  if(rank0) {
    std::cout << "Number of active occupied MOs: " << C_occ_active[0].cols();
    if(is_unrestricted) { std::cout << ", " << C_occ_active[1].cols() << "\n"; }
    else { std::cout << "\n"; }
    std::cout << "Number of active unoccupied MOs: " << C_uno_active[0].cols();
    if(is_unrestricted) { std::cout << ", " << C_uno_active[1].cols() << "\n"; }
    else { std::cout << "\n"; }
  }

  // Make Full and Subsystem densities
  std::vector<Matrix> Dens_ab, Dens_a, Dens_b, Q_a, Q_b;
  std::vector<int>    n_occ_a;
  double              coeff = is_unrestricted ? 1.0 : 2.0;
  int                 nelectrons_active{0};
  int                 nspin{(int) C_occ.size()};
  for(int ispin = 0; ispin < nspin; ispin++) {
    n_occ_a.push_back(C_occ_active[ispin].cols());
    Dens_ab.push_back(coeff * C_occ[ispin] * C_occ[ispin].transpose());
    Dens_a.push_back(coeff * C_occ_active[ispin] * C_occ_active[ispin].transpose());
    Dens_b.push_back(Dens_ab[ispin] - Dens_a[ispin]);
    Q_a.push_back(C_uno_active[ispin] * C_uno_active[ispin].transpose());
    Q_b.push_back(Sm1 - C_occ[ispin] * C_occ[ispin].transpose() - Q_a[ispin]);
    nelectrons_active += (int) coeff * n_occ_a[ispin];
  }

  // Save Fock Matrices from scf_context
  std::vector<Matrix> F_AO{Matrix(N, N)};
  tamm::tamm_to_eigen_tensor(chem_env.scf_context.F_AO, F_AO[0]);
  if(is_unrestricted) {
    F_AO.push_back(Matrix(N, N));
    tamm::tamm_to_eigen_tensor(chem_env.scf_context.F_beta_AO, F_AO[1]);
  }

  // Update C and D tensors with active occupied orbitals
  // TODO: USE_SCALAPACK CASE
  ttensors.D_alpha = {scf_data.tAO, scf_data.tAO};
  ttensors.C_alpha = {scf_data.tAO, scf_data.tAO_ortho};
  Tensor<TensorType>::allocate(&ec, ttensors.D_alpha, ttensors.C_alpha);
  etensors.C_alpha                      = Eigen::MatrixXd::Zero(N, NMO);
  etensors.C_alpha.leftCols(n_occ_a[0]) = C_occ_active[0];
  // tamm::eigen_to_tamm_tensor(ttensors.C_alpha, temp);
  if(rank0) tamm::eigen_to_tamm_tensor(ttensors.D_alpha, Dens_a[0]);
  if(is_unrestricted) {
    ttensors.D_beta = {scf_data.tAO, scf_data.tAO};
    ttensors.C_beta = {scf_data.tAO, scf_data.tAO_ortho};
    Tensor<TensorType>::allocate(&ec, ttensors.D_beta, ttensors.C_beta);
    etensors.C_beta                      = Eigen::MatrixXd::Zero(N, NMO);
    etensors.C_beta.leftCols(n_occ_a[1]) = C_occ_active[1];
    // tamm::eigen_to_tamm_tensor(ttensors.C_beta, temp);
    if(rank0) tamm::eigen_to_tamm_tensor(ttensors.D_beta, Dens_a[1]);
  }
  ec.pg().barrier();

  // Write Updated Tensors to disk (for a noscf calculation)
  scf_engine.scf_output.rw_md_disk(ec, chem_env, scf_data.ttensors, scf_data.etensors,
                                   scf_engine.files_prefix, false);

  // Deallocate unneeded tensors
  Tensor<TensorType>::deallocate(chem_env.scf_context.C_AO, chem_env.scf_context.F_AO);
  if(chem_env.sys_data.is_unrestricted)
    Tensor<TensorType>::deallocate(chem_env.scf_context.C_beta_AO, chem_env.scf_context.F_beta_AO);
  S12_eigen.resize(0, 0);
  for(int ispin = 0; ispin < nspin; ispin++) {
    // C_occ[ispin].resize(0,0);
    C_occ_active[ispin].resize(0, 0);
  }

  // Get energy of system A (low-level) with noscf option and adjusted charge
  // We also suppress printing
  std::cout.rdbuf(NULL);
  chem_env.ioptions.scf_options.noscf = true;
  chem_env.ioptions.scf_options.charge += chem_env.sys_data.nelectrons - nelectrons_active;
  scf::scf_driver(ec, chem_env);
  std::cout.rdbuf(orig_buf);
  // SCFEngine scf_engine_a(ec, chem_env);
  // scf_engine_a.run(ec, chem_env);
  double energy_a_low = chem_env.scf_context.hf_energy;

  // Build Embedding Potential
  coeff = 1.0 / coeff;
  std::vector<Matrix> Vembedding{Matrix(N, N)};
  scratch.resize(N, N);
  tamm::tamm_to_eigen_tensor(chem_env.scf_context.F_AO, scratch);
  Vembedding[0] = F_AO[0] - scratch;
  if(huzinaga) {
    scratch = (F_AO[0] * Dens_b[0] * S1_eigen);
    Vembedding[0] -= coeff * (scratch + scratch.transpose());
    Vembedding[0] += coeff * coeff * S1_eigen * Dens_b[0] * scratch;
    scratch = (F_AO[0] * Q_b[0] * S1_eigen);
    Vembedding[0] -= (scratch + scratch.transpose());
    Vembedding[0] += S1_eigen * Q_b[0] * scratch;
    Vembedding[0] += lambda * (S1_eigen * Dens_b[0] * S1_eigen);
    Vembedding[0] += lambda * (S1_eigen * Q_b[0] * S1_eigen);
  }
  else { Vembedding[0] += lambda * (S1_eigen * Dens_b[0] * S1_eigen); }
  if(is_unrestricted) {
    tamm::tamm_to_eigen_tensor(chem_env.scf_context.F_beta_AO, scratch);
    Vembedding.push_back(F_AO[1] - scratch);
    if(huzinaga) {
      scratch = (F_AO[1] * Dens_b[1] * S1_eigen);
      Vembedding[1] -= coeff * (scratch + scratch.transpose());
      Vembedding[1] += coeff * coeff * S1_eigen * Dens_b[1] * scratch;
      scratch = (F_AO[1] * Q_b[1] * S1_eigen);
      Vembedding[1] -= (scratch + scratch.transpose());
      Vembedding[1] += S1_eigen * Q_b[1] * scratch;
      Vembedding[1] += lambda * (S1_eigen * Dens_b[1] * S1_eigen);
      Vembedding[1] += lambda * (S1_eigen * Q_b[1] * S1_eigen);
    }
    else { Vembedding[1] += lambda * (S1_eigen * Dens_b[1] * S1_eigen); }
  }
  // Update the energy from the noscf calculation with the embedding potential
  energy_a_low += Vembedding[0].cwiseProduct(Dens_a[0]).sum();
  if(is_unrestricted) energy_a_low += Vembedding[1].cwiseProduct(Dens_a[1]).sum();

  // Print energies up to this point
  if(rank0) {
    std::cout << std::fixed << std::setprecision(10) << std::endl
              << "      Energy AB(low): " << std::right << std::setw(20) << energy_ab_low
              << std::endl
              << "       Energy A(low): " << std::right << std::setw(20) << energy_a_low
              << std::endl;
  }

  // Write Embedding Potential to Disk (will be read during next SCF)
  if(rank0) tamm::eigen_to_tamm_tensor(scf_data.ttensors.H1, Vembedding[0]);
  ec.pg().barrier();
  scf_engine.scf_output.rw_mat_disk(ec, scf_data.ttensors.H1, ".vembedding_alpha", false, false);
  if(is_unrestricted) {
    if(rank0) tamm::eigen_to_tamm_tensor(scf_data.ttensors.H1, Vembedding[1]);
    ec.pg().barrier();
    scf_engine.scf_output.rw_mat_disk(ec, scf_data.ttensors.H1, ".vembedding_beta", false, false);
  }
  ec.pg().barrier();

  // Deallocate tensors
  Tensor<TensorType>::deallocate(ttensors.H1, ttensors.S1, ttensors.T1, ttensors.V1,
                                 ttensors.C_alpha, ttensors.D_alpha, chem_env.scf_context.C_AO,
                                 chem_env.scf_context.F_AO);
  if(is_unrestricted)
    Tensor<TensorType>::deallocate(ttensors.C_beta, ttensors.D_beta, chem_env.scf_context.C_beta_AO,
                                   chem_env.scf_context.F_beta_AO);
  scratch.resize(0, 0);
  // S1_eigen.resize(0,0);
  for(int ispin = 0; ispin < nspin; ispin++) {
    Dens_a[ispin].resize(0, 0);
    Dens_b[ispin].resize(0, 0);
    Dens_ab[ispin].resize(0, 0);
  }

  // Adjust SCF options
  const std::vector<std::string> dft{"PBE", "SCAN", "R2SCAN", "BLYP", "PBE0"};
  bool&                          is_ks     = chem_env.sys_data.is_ks;
  bool&                          use_ksref = chem_env.ioptions.embedding_options.use_ksref;
  const std::vector<std::string> correlated_methods{"MP2",     "CC2",   "CCSD", "CCSD_CANONICAL",
                                                    "CCSD(T)", "CCSDT", "DUCC", "QFLOW"};
  // If high level is HF, clear out the xc for scf
  if(high_level[0] == "HF") {
    is_ks = false;
    chem_env.ioptions.scf_options.xc_type.clear();
  }
  // If high level is a correlated method based on a HF reference determinant, clear out the xc for
  // scf
  else if(std::find(correlated_methods.begin(), correlated_methods.end(), high_level[0]) !=
            correlated_methods.end() &&
          !use_ksref) {
    is_ks = false;
    chem_env.ioptions.scf_options.xc_type.clear();
  }
  // If high level is DFT, replace xc with that for the high level DFT
  for(auto& ihigh: high_level) {
    if(ihigh.rfind("XC_", 0) == 0 || std::find(dft.begin(), dft.end(), ihigh) != dft.end()) {
      is_ks = true;
      chem_env.ioptions.scf_options.xc_type.push_back(ihigh);
    }
  }
  chem_env.ioptions.scf_options.noscf           = false;
  chem_env.ioptions.scf_options.restart         = true;
  chem_env.ioptions.scf_options.read_vembedding = true;
  if(freeze_projected) {
    int nfreeze = nelectrons_alpha - n_occ_a[0];
    nfreeze += nvirtual_alpha - C_uno_active[0].cols();
    if(is_unrestricted) {
      nfreeze = std::min(
        nfreeze, (int) (nelectrons_beta - n_occ_a[1] + nvirtual_beta - C_uno_active[1].cols()));
    }
    chem_env.ioptions.ccsd_options.freeze_virtual = nfreeze;
  }

  // Run system A(high-level)
  double energy_a_high = 0.0;
  scf::scf_driver(ec, chem_env);
  // If high level is HF or DFT, set energy_a_high to the converged SCF energy
  if(high_level[0] == "HF" || std::find(dft.begin(), dft.end(), high_level[0]) != dft.end()) {
    energy_a_high = chem_env.scf_context.hf_energy;
  }
  // If high level is a correlated method and based on a KS reference determinant, run a one-shot HF
  else if(std::find(correlated_methods.begin(), correlated_methods.end(), high_level[0]) !=
            correlated_methods.end() &&
          use_ksref) {
    chem_env.ioptions.scf_options.lshift = 0.0;
    chem_env.ioptions.scf_options.noscf  = true;
    chem_env.sys_data.is_ks              = false;
    scf::scf_driver(ec, chem_env);
    chem_env.scf_context.skip_scf = true;
  }
  // If high level is a correlated method and based on a HF reference determinant, we already have
  // what we need
  else if(std::find(correlated_methods.begin(), correlated_methods.end(), high_level[0]) !=
            correlated_methods.end() &&
          !use_ksref) {
    chem_env.scf_context.skip_scf = true;
    ec.pg().barrier();
  }
  // Will skip SCF in subsequent calls to correlated methods
  if(rank0)
    std::cout << std::fixed << std::setprecision(10) << "Energy A(mean-field): " << std::right
              << std::setw(20) << chem_env.scf_context.hf_energy << std::endl
              << std::endl;

  if(high_level[0] == "MP2") {
    mp2::cd_mp2(ec, chem_env);
    energy_a_high = chem_env.mp2_context.mp2_total_energy;
  }
  else if(high_level[0] == "CC2") {
    cc2::cd_cc2_driver(ec, chem_env);
    energy_a_high = chem_env.cc_context.cc2_total_energy;
  }
  else if(high_level[0] == "CCSD") {
    cc::ccsd::cd_ccsd_driver(ec, chem_env);
    energy_a_high = chem_env.cc_context.ccsd_total_energy;
  }
  else if(high_level[0] == "CCSD_CANONICAL") {
    cc::ccsd_canonical::ccsd_canonical_driver(ec, chem_env);
    energy_a_high = chem_env.cc_context.ccsd_total_energy;
  }
  else if(high_level[0] == "CCSD(T)") {
    cc::ccsd_t::ccsd_t_driver(ec, chem_env);
    energy_a_high = chem_env.cc_context.ccsd_pt_total_energy;
  }
  /*
  else if(high_level[0] == "CCSDT") {
    cc::ccsdt::cd_ccsdt(ec, chem_env);
    energy_a_high = chem_env.cc_context.ccsdt_total_energy;
  }
  */
  else if(high_level[0] == "DUCC") {
    chem_env.ioptions.task_options.ducc.first = true;
    if(high_level.size() > 1) {
      if(high_level[1] == "QFLOW") {
        chem_env.ioptions.task_options.ducc.second = "qflow";
#if defined(USE_NWQSIM)
        cc::ducc::ducc_qflow_driver(ec, chem_env);
#else
        if(rank0)
          std::cout << std::endl
                    << "QFLOW is not enabled. Please rebuild with USE_NWQSIM=ON to use DUCC QFLOW."
                    << std::endl;
#endif
      }
      else { tamm_terminate("Invalid high_level option"); }
    }
    else {
      chem_env.ioptions.task_options.ducc.second = "default";
      cc::ducc::ducc_driver(ec, chem_env);
    }
  }
  else if(high_level[0] == "FCIDUMP") {
    if(rank0) {
      std::cout << "Calling FCIDUMP and exiting - skipping the high-level calculation" << std::endl;
      std::cout << "Will write <i| hcore+Vembed |j> and <ij| 1/r |kl> for MSOs in the active space"
                << std::endl;
    }
    fci_driver_embedding(ec, chem_env, Vembedding, is_unrestricted, rank0);
    ec.pg().barrier();
  }
  else if(high_level[0] != "HF" && high_level[0] != "FCIDUMP" &&
          std::find(correlated_methods.begin(), correlated_methods.end(), high_level[0]) ==
            correlated_methods.end() &&
          std::find(dft.begin(), dft.end(), high_level[0]) == dft.end()) {
    tamm_terminate("Invalid high-level specification: " + high_level[0]);
  }

  // Deallocate tensors from SCF context
  if(chem_env.scf_context.C_AO.is_allocated())
    Tensor<TensorType>::deallocate(chem_env.scf_context.C_AO);
  if(chem_env.scf_context.F_AO.is_allocated())
    Tensor<TensorType>::deallocate(chem_env.scf_context.F_AO);
  if(is_unrestricted) {
    if(chem_env.scf_context.C_beta_AO.is_allocated())
      Tensor<TensorType>::deallocate(chem_env.scf_context.C_beta_AO);
    if(chem_env.scf_context.F_beta_AO.is_allocated())
      Tensor<TensorType>::deallocate(chem_env.scf_context.F_beta_AO);
  }

  // Print final results
  if(rank0) {
    std::cout << std::endl
              << std::endl
              << std::fixed << std::setprecision(10) << "Energy AB     : " << std::right
              << std::setw(20) << energy_ab_low << std::endl
              << "Energy A(low) : " << std::right << std::setw(20) << energy_a_low << std::endl
              << "Energy A(high): " << std::right << std::setw(20) << energy_a_high << std::endl
              << "Total Energy  : " << std::right << std::setw(20)
              << energy_ab_low + (energy_a_high - energy_a_low) << std::endl;
  }
}

Matrix spade(const Matrix& S12, const Matrix& C_occ, const std::vector<int> indicesToKeep,
             const int n_acc_mos, bool rank0) {
  Matrix              C_occ_ortho  = S12 * C_occ;
  Matrix              C_occ_active = C_occ_ortho(indicesToKeep, Eigen::placeholders::all);
  Matrix              CTC          = C_occ_active.transpose() * C_occ_active;
  size_t              Nocc         = CTC.rows();
  std::vector<double> eigvals(Nocc);

  lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Nocc, CTC.data(), Nocc, eigvals.data());
  Matrix eigenvectors = CTC.transpose();
  size_t n_act_mos    = 0;
  // seeded below any possible gap so the first delta always wins. syevd returns ascending
  // eigenvalues, so a degenerate spectrum would otherwise leave n_act_mos unassigned
  double maxdelta = -1.0;
  if(n_acc_mos == 0) {
    for(size_t ival = 0; ival < eigvals.size() - 1; ival++) {
      double delta = (eigvals[ival + 1] - eigvals[ival]);
      if(delta > maxdelta) {
        maxdelta  = delta;
        n_act_mos = Nocc - ival - 1;
      }
    }
  }
  else { n_act_mos = n_acc_mos; }
  if(rank0) {
    std::cout << std::setprecision(10) << std::endl;
    std::cout << "*** SPADE ALGORITHM ***" << std::endl;
    for(size_t ival = 0; ival < eigvals.size(); ival++) { std::cout << eigvals[ival] << std::endl; }
    std::cout << std::endl;
  }
  return C_occ * eigenvectors.rightCols(n_act_mos);
}

Matrix avas(const Matrix& S1_eigen, const Matrix& C_occ, const std::vector<int> target_indices,
            const double avas_threshold, const int n_acc_mos, bool rank0) {
  Matrix              sigma = S1_eigen(target_indices, target_indices);
  Matrix              S21   = S1_eigen(target_indices, Eigen::placeholders::all);
  Matrix              P     = S21.transpose() * sigma.inverse() * S21;
  Matrix              SA    = C_occ.transpose() * P * C_occ;
  size_t              Nocc  = SA.rows();
  std::vector<double> eigvals(Nocc);
  lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Nocc, SA.data(), Nocc, eigvals.data());
  Matrix eigenvectors = SA.transpose();
  size_t n_act_mos    = 0;
  if(n_acc_mos == 0) {
    for(double val: eigvals)
      if(val > avas_threshold) n_act_mos++;
  }
  else { n_act_mos = n_acc_mos; }
  if(rank0) {
    std::cout << std::setprecision(10) << std::endl;
    std::cout << "*** AVAS ALGORITHM ***" << std::endl;
    for(auto x: eigvals) std::cout << x << std::endl;
    std::cout << std::endl;
  }
  return C_occ * eigenvectors.rightCols(n_act_mos);
}

Matrix paos(const Matrix& S, const Matrix& C_occ, const std::vector<int> indicesToKeep, bool rank0,
            double pao_thresh1, double pao_thresh2) {
  // double    pao_thresh1 = 0.01;
  // double    pao_thresh2 = 0.00001;
  const int Npao1 = indicesToKeep.size();
  const int N     = S.rows();
  Matrix    c_pao = (Eigen::MatrixXd::Identity(N, N) -
                  C_occ * C_occ.transpose() * S)(Eigen::placeholders::all, indicesToKeep);

  Matrix           SC = S * c_pao;
  std::vector<int> active_paos;
  for(size_t ipao = 0; ipao < (size_t) Npao1; ipao++) {
    double norb = c_pao(Eigen::placeholders::all, ipao).dot(SC.col(ipao));
    if(std::abs(norb) > pao_thresh1) { active_paos.push_back(ipao); }
  }

  Matrix              c_pao_prime = c_pao(Eigen::placeholders::all, active_paos);
  Matrix              CSC         = c_pao_prime.transpose() * S * c_pao_prime;
  const int           Npaos       = c_pao_prime.cols();
  std::vector<double> eigvals(Npaos);
  lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, Npaos, CSC.data(), Npaos, eigvals.data());
  Matrix eigenvectors = CSC.transpose();
  active_paos.clear();
  for(size_t ipao = 0; ipao < Npaos; ipao++) {
    if(eigvals[ipao] > pao_thresh2) {
      eigenvectors.col(ipao) *= 1.0 / std::sqrt(eigvals[ipao]);
      active_paos.push_back(ipao);
    }
  }
  return c_pao_prime * eigenvectors(Eigen::placeholders::all, active_paos);
}

Matrix sqrtm(const Matrix& mat) {
  size_t              N    = mat.rows();
  Matrix              vecs = mat;
  std::vector<double> vals(N);
  lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, N, vecs.data(), N, vals.data());
  for(size_t ival = 0; ival < N; ival++) {
    if(vals[ival] < 0.0) { vecs.row(ival) *= 0.0; }
    else { vecs.row(ival) *= std::sqrt(std::sqrt(vals[ival])); }
  }
  return vecs.transpose() * vecs;
}

void permute_orbitals(ChemEnv& chem_env, const Matrix& C_occ, const Matrix& S, Tensor<double>& C_AO,
                      const int nocc, const int nvirtual) {
  int    N   = S.rows();
  int    NMO = chem_env.sys_data.nbf;
  Matrix C(N, NMO);
  tamm::tamm_to_eigen_tensor(C_AO, C);
  Matrix Overlap = C_occ.transpose() * S * C.rightCols(nvirtual);
  Overlap        = Overlap.cwiseAbs();
  Eigen::VectorXi indices(NMO);
  for(int iocc = 0; iocc < nocc; iocc++) { indices(iocc) = iocc; }
  for(int ivirtual = 0, jvirtual = 0, kvirtual = nocc; ivirtual < nvirtual; ivirtual++) {
    double maxcoeff = Overlap.col(ivirtual).maxCoeff();
    if(maxcoeff > 0.1) {
      indices(NMO - jvirtual - 1) = nocc + ivirtual;
      jvirtual++;
    }
    else {
      indices(kvirtual) = nocc + ivirtual;
      kvirtual++;
    }
  }
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
  perm.indices() = indices;
  C              = C * perm;
  tamm::eigen_to_tamm_tensor(C_AO, C);
}

void print_eigs(ExecutionContext& ec, Tensor<double>& F_tamm, Tensor<double>& C_tamm, size_t N,
                size_t NMO) {
  Matrix F(N, N), C(N, NMO);
  tamm::tamm_to_eigen_tensor(F_tamm, F);
  tamm::tamm_to_eigen_tensor(C_tamm, C);
  Matrix FC = F * C;
  std::cout << "EIGVALS" << std::fixed << std::setprecision(10) << std::endl;
  for(size_t imo = 0; imo < NMO; imo++) {
    double eigval = FC.col(imo).dot(C.col(imo));
    std::cout << eigval << ", ";
    if(imo + 1 % 5 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
}

void fci_driver_embedding(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Matrix>& Vembedding,
                          const bool is_unrestricted, bool rank0) {
  chem_env.cd_context.keep_movecs_so = true;
  cholesky_2e::cholesky_2e_driver(ec, chem_env);
  std::string files_prefix = chem_env.get_files_prefix();
  CDContext&  cd_context   = chem_env.cd_context;
  chem_env.cc_context.init_filenames(files_prefix);
  const TiledIndexSpace& MO      = chem_env.is_context.MSO;
  const TiledIndexSpace& CI      = chem_env.is_context.CI;
  const TiledIndexSpace  N_MSO   = MO("all");
  Tensor<T>              d_f1    = cd_context.d_f1;
  Tensor<T>              cholVpr = cd_context.cholV2;
  const auto [cindex]            = CI.labels<1>("all");
  const auto [p, q, r, s]        = MO.labels<4>("all");
  Tensor<T> full_v2{N_MSO, N_MSO, N_MSO, N_MSO};
  Tensor<T>::allocate(&ec, full_v2);
  ExecutionHW ex_hw = ec.exhw();
  Scheduler   sch{ec};
  sch(full_v2(p, r, q, s) = cholVpr(p, r, cindex) * cholVpr(q, s, cindex)).execute(ex_hw);
  Tensor<T>              lcao = cd_context.movecs_so;
  const TiledIndexSpace& AO   = lcao.tiled_index_spaces()[0];
  const auto [mu, nu]         = AO.labels<2>("all");
  Tensor<T> hcore{AO, AO};
  Tensor<T> hcore_eff_alpha{AO, AO};
  Tensor<T> hcore_eff_beta{AO, AO};
  Tensor<T>::allocate(&ec, hcore);
  Tensor<T>::allocate(&ec, hcore_eff_alpha);
  Tensor<T>::allocate(&ec, hcore_eff_beta);
  read_from_disk(ec, hcore, chem_env.get_files_prefix("", "scf") + ".hcore");
  if(rank0) tamm::eigen_to_tamm_tensor(hcore_eff_alpha, Vembedding[0]);
  ec.pg().barrier();
  sch(hcore_eff_alpha(mu, nu) += hcore(mu, nu)).execute(ec.exhw());
  if(is_unrestricted) {
    if(rank0) tamm::eigen_to_tamm_tensor(hcore_eff_beta, Vembedding[1]);
    ec.pg().barrier();
    sch(hcore_eff_beta(mu, nu) += hcore(mu, nu)).execute(ec.exhw());
  }
  else sch(hcore_eff_beta(mu, nu) = hcore_eff_alpha(mu, nu)).execute(ec.exhw());
  generate_fcidump_embedding(chem_env, ec, hcore_eff_alpha, hcore_eff_beta, lcao, full_v2, ex_hw);
  free_tensors(hcore, hcore_eff_alpha, hcore_eff_beta, lcao, full_v2, cholVpr, d_f1);
  int nspin{(int) Vembedding.size()};
  for(int ispin = 0; ispin < nspin; ispin++) { Vembedding[ispin].resize(0, 0); }
}

template<typename T>
void generate_fcidump_embedding(ChemEnv& chem_env, ExecutionContext& ec,
                                const Tensor<T>& hcore_eff_alpha, const Tensor<T>& hcore_eff_beta,
                                const Tensor<T>& lcao, const Tensor<T>& full_v2, ExecutionHW ex_hw,
                                std::string filename) {
  SystemData&  sys_data    = chem_env.sys_data;
  const size_t n_occ_alpha = sys_data.n_occ_alpha;
  const size_t n_occ_beta  = sys_data.n_occ_beta;
  const size_t n_vir_alpha = sys_data.n_vir_alpha;
  const size_t n_vir_beta  = sys_data.n_vir_beta;
  const size_t nmo_occ     = n_occ_alpha + n_occ_beta;
  const size_t nmo_vir     = n_vir_alpha + n_vir_beta;
  if(filename.empty()) filename = chem_env.get_files_prefix("", "fci") + ".fcidump";
  const fs::path files_dir = fs::path(filename).parent_path();
  if(!files_dir.empty() && !fs::exists(files_dir)) fs::create_directories(files_dir);
  const TiledIndexSpace& AO  = lcao.tiled_index_spaces()[0];
  const TiledIndexSpace& MSO = lcao.tiled_index_spaces()[1];
  const auto [mu, nu]        = AO.labels<2>("all");
  const auto [z1, z2]        = MSO.labels<2>("all");
  // Transform the core Hamiltonian to the one-electron integrals in the MO basis
  Tensor<T> hcore_eff_alpha_mo{{MSO, MSO}, {1, 1}};
  Tensor<T> hcore_eff_beta_mo{{MSO, MSO}, {1, 1}};
  Tensor<T> hcore_eff_mo{{MSO, MSO}, {1, 1}};
  Tensor<T> tmp_alpha{MSO, AO};
  Tensor<T> tmp_beta{MSO, AO};
  Scheduler sch{ec};
  // clang-format off
  sch.allocate(hcore_eff_alpha_mo, tmp_alpha)
    (tmp_alpha(z1,nu) = lcao(mu,z1) * hcore_eff_alpha(mu,nu))
    (hcore_eff_alpha_mo(z1,z2) = tmp_alpha(z1,nu) * lcao(nu,z2))
    .deallocate(tmp_alpha).execute(ex_hw);
  sch.allocate(hcore_eff_beta_mo, tmp_beta)
    (tmp_beta(z1,nu) = lcao(mu,z1) * hcore_eff_beta(mu,nu))
    (hcore_eff_beta_mo(z1,z2) = tmp_beta(z1,nu) * lcao(nu,z2))
    .deallocate(tmp_beta).execute(ex_hw);
  Matrix hcore_eff_alpha_mo_eig(nmo_occ + nmo_vir, nmo_occ + nmo_vir);
  Matrix hcore_eff_beta_mo_eig(nmo_occ + nmo_vir, nmo_occ + nmo_vir);
  tamm::tamm_to_eigen_tensor(hcore_eff_alpha_mo, hcore_eff_alpha_mo_eig);
  tamm::tamm_to_eigen_tensor(hcore_eff_beta_mo, hcore_eff_beta_mo_eig);
  // Create one generalized hcore_eff_mo_eig in the basis {alpha_o, beta_o, alpha_v, beta_v}
  // Inspired by the generalized Fock matrix construction in cholesky::TwoIndexTransform::two_index_transform
  // Note that hcore_eff_alpha_mo_eig and hcore_eff_beta_mo_eig are each represented in the full MSO basis, so we select the relevant blocks
  // clang-format on 
  Matrix hcore_eff_mo_eig; 
  hcore_eff_mo_eig.setZero(nmo_occ + nmo_vir, nmo_occ + nmo_vir);
  /*
        hcore_MO   = |  hcore_MO_alpha_oo           0             hcore_MO_alpha_ov            0         |
                     |      0               hcore_MO_beta_oo            0             hcore_MO_beta_ov   |
                     |  hcore_MO_alpha_vo           0             hcore_MO_alpha_vv            0         |
                     |      0               hcore_MO_beta_vo            0             hcore_MO_beta_vv   |
  */
  hcore_eff_mo_eig.block(0,       0,        n_occ_alpha, n_occ_alpha) = hcore_eff_alpha_mo_eig.block(0,       0,        n_occ_alpha, n_occ_alpha); // alpha_oo
  hcore_eff_mo_eig.block(0,       nmo_occ,  n_occ_alpha, n_vir_alpha) = hcore_eff_alpha_mo_eig.block(0,       nmo_occ,  n_occ_alpha, n_vir_alpha); // alpha_ov
  hcore_eff_mo_eig.block(nmo_occ, 0,        n_vir_alpha, n_occ_alpha) = hcore_eff_alpha_mo_eig.block(nmo_occ, 0,        n_vir_alpha, n_occ_alpha); // alpha_vo
  hcore_eff_mo_eig.block(nmo_occ, nmo_occ,  n_vir_alpha, n_vir_alpha) = hcore_eff_alpha_mo_eig.block(nmo_occ, nmo_occ,  n_vir_alpha, n_vir_alpha); // alpha_vv

  hcore_eff_mo_eig.block(n_occ_alpha,         n_occ_alpha,         n_occ_beta, n_occ_beta) = hcore_eff_beta_mo_eig.block(n_occ_alpha,         n_occ_alpha,         n_occ_beta, n_occ_beta); // beta_oo
  hcore_eff_mo_eig.block(n_occ_alpha,         nmo_occ+n_vir_alpha, n_occ_beta, n_vir_beta) = hcore_eff_beta_mo_eig.block(n_occ_alpha,         nmo_occ+n_vir_alpha, n_occ_beta, n_vir_beta); // beta_ov
  hcore_eff_mo_eig.block(nmo_occ+n_vir_alpha, n_occ_alpha,         n_vir_beta, n_occ_beta) = hcore_eff_beta_mo_eig.block(nmo_occ+n_vir_alpha, n_occ_alpha,         n_vir_beta, n_occ_beta); // beta_vo
  hcore_eff_mo_eig.block(nmo_occ+n_vir_alpha, nmo_occ+n_vir_alpha, n_vir_beta, n_vir_beta) = hcore_eff_beta_mo_eig.block(nmo_occ+n_vir_alpha, nmo_occ+n_vir_alpha, n_vir_beta, n_vir_beta); // beta_vv
  // One entry per orbital in the MO space (linear dependencies and frozen orbitals excluded)
  Tensor<T>::allocate(&ec, hcore_eff_mo);
  tamm::eigen_to_tamm_tensor(hcore_eff_mo, hcore_eff_mo_eig);
  const size_t     nmso = MSO.max_num_indices();
  std::vector<int> symvec(sys_data.is_unrestricted ? nmso : nmso / 2, 1);
  fcidump::write_fcidump_file(chem_env, hcore_eff_mo, full_v2, symvec, filename);
  free_tensors(hcore_eff_alpha_mo, hcore_eff_beta_mo, hcore_eff_mo);
  hcore_eff_alpha_mo_eig.resize(0, 0);
  hcore_eff_beta_mo_eig.resize(0, 0);
  hcore_eff_mo_eig.resize(0, 0);
}

} // namespace exachem::embedding
