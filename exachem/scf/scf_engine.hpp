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

class SCFEngine {
public:
  // Public constructors so this engine can be instantiated directly
  SCFEngine() = default;
  SCFEngine(ExecutionContext& exc, ChemEnv& chem_env);

protected:
  SCFCompute<TensorType> scf_compute;
  SCFQed<TensorType>     scf_qed;
  SCFIter<TensorType>    scf_iter;
  SCFGuess<TensorType>   scf_guess;
  SCFRestart<TensorType> scf_restart;
  SCFIO<TensorType>      scf_output;
  ECMolden               ec_molden;
  SCFData                scf_data;

  std::string files_prefix;
  enum class FileType {
    Schwarz,
    AlphaMovecs,
    BetaMovecs,
    Hcore,
    VxcAlpha,
    VxcBeta,
    QEDDx,
    QEDDy,
    QEDDz,
    QEDQxx,
    Ortho,
    OrthoJson
  };

  std::unordered_map<FileType, std::string> ext_type = {
    {FileType::Schwarz, ".schwarz"},        {FileType::AlphaMovecs, ".alpha.movecs"},
    {FileType::BetaMovecs, ".beta.movecs"}, {FileType::Hcore, ".hcore"},
    {FileType::VxcAlpha, ".vxc_alpha"},     {FileType::VxcBeta, ".vxc_beta"},
    {FileType::QEDDx, ".QED_Dx"},           {FileType::QEDDy, ".QED_Dy"},
    {FileType::QEDDz, ".QED_Dz"},           {FileType::QEDQxx, ".QED_Qxx"},
    {FileType::Ortho, ".orthogonalizer"},   {FileType::OrthoJson, ".orthogonalizer.json"}};

  std::map<FileType, std::string> fname;

  struct SCFIterationState {
    double     ehf        = 0.0;
    double     ediff      = 0.0;
    TensorType ediis      = 0.0;
    size_t     ndiis      = 0;
    size_t     nbumps     = 0;
    size_t     iter       = 0;
    double     rmsd       = 0.0;
    size_t     nbumps_max = 3;
    bool       is_conv    = true;
    bool       is_3c_init = false; // whether 3-center integrals are initialized
  };

  ScalapackInfo scalapack_info;

#if defined(USE_GAUXC)
  SCFGauxc<TensorType>                         scf_gauxc;
  std::shared_ptr<GauXC::XCIntegrator<Matrix>> gauxc_integrator_ptr;
  double                                       xHF;
  double                                       gauxc_exc = 0;
#else
  double xHF = 1.;
#endif

  // All methods are virtual and can be overridden
  virtual void reset_tolerences(ExecutionContext& exc, ChemEnv& chem_env);
  virtual void write_dplot_data(ExecutionContext& exc, ChemEnv& chem_env);
  virtual void qed_tensors_1e(ExecutionContext& ec, const ChemEnv& chem_env);
  virtual void setup_libecpint(ExecutionContext& exc, ChemEnv& chem_env,
                               std::vector<libecpint::ECP>&           ecps,
                               std::vector<libecpint::GaussianShell>& libecp_shells);
  virtual void scf_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env);
  virtual void declare_main_tensors(ExecutionContext& ec, const ChemEnv& chem_env);
  virtual void setup_tiled_index_space(ExecutionContext& exc, ChemEnv& chem_env);
  virtual void process_molden_data(ExecutionContext& exc, ChemEnv& chem_env);
  virtual void setup_density_fitting(ExecutionContext& exc, ChemEnv& chem_env);

  virtual double calculate_diis_error(bool is_uhf, size_t ndiis) const;
  virtual void   reset_fock_and_save_last_density(ExecutionContext& exc, ChemEnv& chem_env);
  virtual void   handle_energy_bumps(ExecutionContext& exc, const ChemEnv& chem_env,
                                     SCFIterationState& scf_state);
  virtual void   print_write_iteration(ExecutionContext& exc, ChemEnv& chem_env, double loop_time,
                                       SCFIterationState& scf_state);
  virtual bool   check_convergence(ExecutionContext& exc, const ChemEnv& chem_env,
                                   SCFIterationState& scf_state);
  inline void    set_basis_purity(const ChemEnv& chem_env, libint2::BasisSet& basis) const {
       basis.set_pure(chem_env.ioptions.scf_options.gaussian_type == "spherical");
  }
  virtual void compute_fock_matrix(ExecutionContext& ec, const ChemEnv& chem_env, bool is_uhf,
                                   const bool do_schwarz_screen, Matrix& SchwarzK,
                                   const size_t& max_nprim4, std::vector<size_t>& shell2bf,
                                   bool& is_3c_init);
  virtual void deallocate_main_tensors(ExecutionContext& exc, const ChemEnv& chem_env);
  virtual void scf_final_io(ExecutionContext& ec, const ChemEnv& chem_env);
  virtual void setup_file_paths(const ChemEnv& chem_env);
  virtual void initialize_scf_vars_and_tensors(ChemEnv& chem_env);
  virtual std::tuple<Tensor<TensorType>, Tensor<TensorType>, TiledIndexSpace>
  update_movecs(ExecutionContext& ec, ChemEnv& chem_env);

#if defined(USE_GAUXC)
  virtual GauXC::XCIntegrator<Matrix> get_gauxc_integrator(ExecutionContext& ec,
                                                           const ChemEnv&    chem_env);
  virtual void add_snk_contribution(ExecutionContext& exc, const ChemEnv& chem_env,
                                    GauXC::XCIntegrator<Matrix>& gauxc_integrator);
  virtual void compute_update_xc(ExecutionContext& ec, const ChemEnv& chem_env,
                                 GauXC::XCIntegrator<Matrix>& gauxc_integrator, double& ehf);
#endif

public:
  virtual ~SCFEngine() = default;

  // Delete copy operations (expensive due to complex state)
  SCFEngine(const SCFEngine&)            = delete;
  SCFEngine& operator=(const SCFEngine&) = delete;

  // Allow move operations (efficient for transferring state)
  SCFEngine(SCFEngine&&) noexcept            = default;
  SCFEngine& operator=(SCFEngine&&) noexcept = default;

  // Main entry point, can be overridden
  virtual void run(ExecutionContext& exc, ChemEnv& chem_env);
};
} // namespace exachem::scf
