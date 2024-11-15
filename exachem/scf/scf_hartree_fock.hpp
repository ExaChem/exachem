/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/chemenv.hpp"
#include "exachem/common/ec_basis.hpp"
#include "exachem/common/ec_molden.hpp"
#include "exachem/common/system_data.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "exachem/scf/scf_iter.hpp"
#include "exachem/scf/scf_outputs.hpp"
#include "exachem/scf/scf_qed.hpp"
#include "exachem/scf/scf_restart.hpp"
#include "exachem/scf/scf_taskmap.hpp"
#include <variant>

namespace exachem::scf {

class SCFHartreeFock {
private:
  bool        is_spherical;
  std::string out_fp;
  std::string files_dir;
  std::string files_prefix;

  SCFCompute scf_compute;
  SCFQed     scf_qed;
  SCFIter    scf_iter;
  SCFGuess   scf_guess;
  SCFRestart scf_restart;
  SCFIO      scf_output;
  ECMolden   ec_molden;
  SCFVars    scf_vars; // init vars

  double ehf = 0.0; // initialize Hartree-Fock energy

  EigenTensors etensors;
  TAMMTensors  ttensors;

  bool is_conv = true;

  std::vector<libecpint::ECP>           ecps;
  std::vector<libecpint::GaussianShell> libecp_shells;

  std::string ortho_file, ortho_jfile;

  std::string schwarz_matfile;
  Matrix      SchwarzK;

  std::string movecsfile_alpha, movecsfile_beta;
  std::string qed_dx_file, qed_dy_file, qed_dz_file, qed_qxx_file;
  std::string vcx_alpha_file, vcx_beta_file;
  std::string hcore_file;
  // SCF main loop
  TensorType   ediis;
  size_t       nbumps, ndiis;
  const size_t nbumps_max = 3;
  double       conve;
  double       convd;
  // double enuc;

  Tensor<TensorType> Fa_global;
  Tensor<TensorType> Fb_global;

  IndexSpace         AO_ortho;
  TiledIndexSpace    tAO_ortho;
  Tensor<TensorType> C_alpha_tamm;
  Tensor<TensorType> C_beta_tamm;

  IndexSpace       dfCocc;
  std::vector<int> s1vec, s2vec, ntask_vec;
  bool             is_3c_init         = false;
  bool             do_density_fitting = false;

  double        rmsd  = 1.0;
  double        ediff = 0.0;
  int           iter  = 0;
  IndexSpace    AO;
  ScalapackInfo scalapack_info;
  /* Indices  representing a specific basis function or atomic orbital */
  tamm::TiledIndexLabel mu, nu, ku;
  tamm::TiledIndexLabel mup, nup, kup;

  ProcGroupData pgdata;

#if defined(USE_GAUXC)
  std::shared_ptr<GauXC::XCIntegrator<Matrix>> gauxc_integrator_ptr;
  double                                       xHF;
  double                                       gauxc_exc = 0;
#else
  const double xHF = 1.;
#endif

  void scf_hf(ExecutionContext& exc, ChemEnv& chem_env);
  void initialize_variables(ExecutionContext& exc, ChemEnv& chem_env);
  void reset_tolerences(ExecutionContext& exc, ChemEnv& chem_env);
  void write_dplot_data(ExecutionContext& exc, ChemEnv& chem_env);
  void qed_tensors_1e(ExecutionContext& ec, ChemEnv& chem_env);
  void setup_libecpint(ExecutionContext& exc, ChemEnv& chem_env);
  void scf_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env);
  void declare_main_tensors(ExecutionContext& ec, ChemEnv& chem_env);
  void setup_tiled_index_space(ExecutionContext& exc, ChemEnv& chem_env);
  void process_molden_data(ExecutionContext& exc, ChemEnv& chem_env);
  void setup_density_fitting(ExecutionContext& exc, ChemEnv& chem_env);

  double calculate_diis_error(bool is_uhf);
  void   reset_fock_and_save_last_density(ExecutionContext& exc, ChemEnv& chem_env);
  void   handle_energy_bumps(ExecutionContext& exc, ChemEnv& chem_env);
  void   print_write_iteration(ExecutionContext& exc, ChemEnv& chem_env, double loop_time);
  bool   check_convergence(ExecutionContext& exc, ChemEnv& chem_env);

  void compute_fock_matrix(ExecutionContext& ec, ChemEnv& chem_env, bool is_uhf,
                           const bool do_schwarz_screen, const size_t& max_nprim4,
                           std::vector<size_t>& shell2bf);
  void deallocate_main_tensors(ExecutionContext& exc, ChemEnv& chem_env);
  void scf_final_io(ExecutionContext& ec, ChemEnv& chem_env);
  void update_movecs(ExecutionContext& ec, ChemEnv& chem_env);
#if defined(USE_GAUXC)
  GauXC::XCIntegrator<Matrix> get_gauxc_integrator(ExecutionContext& ec, ChemEnv& chem_env);
  void                        add_snk_contribution(ExecutionContext& exc, ChemEnv& chem_env,
                                                   GauXC::XCIntegrator<Matrix>& gauxc_integrator);
  void                        compute_update_xc(ExecutionContext& ec, ChemEnv& chem_env,
                                                GauXC::XCIntegrator<Matrix>& gauxc_integrator);
#endif

public:
  SCFHartreeFock() = default;
  SCFHartreeFock(ExecutionContext& exc, ChemEnv& chem_env) { initialize(exc, chem_env); };
  void operator()(ExecutionContext& exc, ChemEnv& chem_env) { initialize(exc, chem_env); };
  void initialize(ExecutionContext& exc, ChemEnv& chem_env);
};
} // namespace exachem::scf
