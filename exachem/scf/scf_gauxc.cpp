/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_gauxc.hpp"

#if defined(USE_GAUXC)

std::tuple<std::shared_ptr<GauXC::XCIntegrator<Matrix>>, double>
exachem::scf::gauxc::setup_gauxc(ExecutionContext& ec, const ChemEnv& chem_env,
                                 const SCFVars& scf_vars) {
  const SystemData& sys_data    = chem_env.sys_data;
  const SCFOptions& scf_options = chem_env.ioptions.scf_options;

  const std::vector<libint2::Atom>& atoms  = chem_env.atoms;
  const libint2::BasisSet&          shells = chem_env.shells;

  const bool is_rhf = sys_data.is_restricted;
  // const bool is_ks      = sys_data.is_ks;
  // const bool is_qed     = sys_data.is_qed;
  // const bool do_qed     = sys_data.do_qed;
  auto rank = ec.pg().rank();

  auto gc1 = std::chrono::high_resolution_clock::now();

  auto gauxc_mol   = make_gauxc_molecule(atoms);
  auto gauxc_basis = make_gauxc_basis(shells, scf_options.xc_basis_tol);
  auto polar       = is_rhf ? ExchCXX::Spin::Unpolarized : ExchCXX::Spin::Polarized;

  auto xc_radang_size = scf_options.xc_radang_size;
  auto xc_grid_str    = scf_options.xc_grid_type;
  std::transform(xc_grid_str.begin(), xc_grid_str.end(), xc_grid_str.begin(), ::tolower);
  std::map<std::string, GauXC::AtomicGridSizeDefault> grid_map = {
    {"fine", GauXC::AtomicGridSizeDefault::FineGrid},
    {"ultrafine", GauXC::AtomicGridSizeDefault::UltraFineGrid},
    {"superfine", GauXC::AtomicGridSizeDefault::SuperFineGrid},
    {"gm3", GauXC::AtomicGridSizeDefault::GM3},
    {"gm5", GauXC::AtomicGridSizeDefault::GM5}};
  const bool use_custom_grid = xc_radang_size.first > 0 && xc_radang_size.second > 0;

  // This options are set to get good accuracy.
  // [Unpruned, Robust, Treutler]
  auto xc_pruning_str = scf_options.xc_pruning_scheme;
  std::transform(xc_pruning_str.begin(), xc_pruning_str.end(), xc_pruning_str.begin(), ::tolower);
  std::map<std::string, GauXC::PruningScheme> pruning_map = {
    {"robust", GauXC::PruningScheme::Robust},
    {"treutler", GauXC::PruningScheme::Treutler},
    {"unpruned", GauXC::PruningScheme::Unpruned}};

  auto xc_radquad_str = scf_options.xc_rad_quad;
  std::transform(xc_radquad_str.begin(), xc_radquad_str.end(), xc_radquad_str.begin(), ::tolower);
  std::map<std::string, GauXC::RadialQuad> radquad_map = {
    {"mk", GauXC::RadialQuad::MuraKnowles},
    {"ta", GauXC::RadialQuad::TreutlerAldrichs},
    {"mhl", GauXC::RadialQuad::MurrayHandyLaming}};

  auto gauxc_molgrid =
    use_custom_grid
      ? GauXC::MolGridFactory::create_default_molgrid(
          gauxc_mol, pruning_map.at(xc_pruning_str),
          GauXC::BatchSize((size_t) scf_options.xc_batch_size), radquad_map.at(xc_radquad_str),
          GauXC::RadialSize(xc_radang_size.first), GauXC::AngularSize(xc_radang_size.second))
      : GauXC::MolGridFactory::create_default_molgrid(
          gauxc_mol, pruning_map.at(xc_pruning_str),
          GauXC::BatchSize((size_t) scf_options.xc_batch_size), radquad_map.at(xc_radquad_str),
          grid_map.at(xc_grid_str));

  auto gauxc_molmeta = std::make_shared<GauXC::MolMeta>(gauxc_mol);

  auto xc_space_str = scf_options.xc_exec_space;
  std::transform(xc_space_str.begin(), xc_space_str.end(), xc_space_str.begin(), ::tolower);

#ifdef GAUXC_HAS_DEVICE
  double gauxc_gpu_pool = 0.1;
  if(const char* tamm_gpu_pool_char = std::getenv("TAMM_GPU_POOL")) {
    int tamm_gpu_pool = std::atoi(tamm_gpu_pool_char);
    gauxc_gpu_pool    = 0.9 - tamm_gpu_pool / 100.0;
  }

  std::map<std::string, GauXC::ExecutionSpace> exec_space_map = {
    {"host", GauXC::ExecutionSpace::Host}, {"device", GauXC::ExecutionSpace::Device}};
  auto lb_exec_space  = exec_space_map.at(xc_space_str);
  auto int_exec_space = exec_space_map.at(xc_space_str);
  auto gauxc_rt       = txt_utils::strequal_case(xc_space_str, "device")
                          ? GauXC::DeviceRuntimeEnvironment(ec.pg().comm(), gauxc_gpu_pool)
                          : GauXC::RuntimeEnvironment(ec.pg().comm());
#else
  auto gauxc_rt       = GauXC::RuntimeEnvironment(ec.pg().comm());
  auto lb_exec_space  = GauXC::ExecutionSpace::Host;
  auto int_exec_space = GauXC::ExecutionSpace::Host;
#endif

  // Set the load balancer
  GauXC::LoadBalancerFactory lb_factory(lb_exec_space, scf_options.xc_lb_kernel);
  auto gauxc_lb = lb_factory.get_shared_instance(gauxc_rt, gauxc_mol, gauxc_molgrid, gauxc_basis);

  // Modify the weighting algorithm from the input [Becke, SSF, LKO]
  auto xc_weight_str = scf_options.xc_weight_scheme;
  std::transform(xc_weight_str.begin(), xc_weight_str.end(), xc_weight_str.begin(), ::tolower);
  std::map<std::string, GauXC::XCWeightAlg> weight_map = {{"ssf", GauXC::XCWeightAlg::SSF},
                                                          {"becke", GauXC::XCWeightAlg::Becke},
                                                          {"lko", GauXC::XCWeightAlg::LKO}};

  GauXC::MolecularWeightsSettings mw_settings = {weight_map.at(xc_weight_str), false};
  GauXC::MolecularWeightsFactory  mw_factory(int_exec_space, scf_options.xc_mw_kernel, mw_settings);
  auto                            mw = mw_factory.get_instance();
  mw.modify_weights(*gauxc_lb);

  std::vector<std::string>       xc_vector = scf_options.xc_type;
  std::vector<ExchCXX::XCKernel> kernels   = {};
  std::vector<double>            params(2049, 0.0);
  // int                            kernel_id = -1;

  // TODO: Refactor DFT code path when we eventually enable GauXC by default.
  // is_ks=false, so we setup, but do not run DFT.
  for(std::string& xcfunc: xc_vector) {
    std::transform(xcfunc.begin(), xcfunc.end(), xcfunc.begin(), ::toupper);
    if(rank == 0) std::cout << "Functional: " << xcfunc << std::endl;

    // First try few functionals defined in ExchCXX
    if(txt_utils::strequal_case(xcfunc, "BLYP")) {
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("B88"), polar));
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("LYP"), polar));
    }
    else if(txt_utils::strequal_case(xcfunc, "PBE")) {
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("PBE_X"), polar));
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("PBE_C"), polar));
      // SCAN and R2SCAN might not be implemented in current version, fallback to LibXC
    }
    else if(txt_utils::strequal_case(xcfunc, "SCAN")) {
      if(ExchCXX::kernel_map.key_exists("SCAN_X")) {
        kernels.push_back(
          ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("SCAN_X"), polar));
        kernels.push_back(
          ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("SCAN_C"), polar));
      }
      else {
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string("MGGA_X_SCAN"), polar));
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string("MGGA_C_SCAN"), polar));
      }
    }
    else if(txt_utils::strequal_case(xcfunc, "R2SCAN")) {
      if(ExchCXX::kernel_map.key_exists("R2SCAN_X")) {
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::Backend::builtin,
                                            ExchCXX::kernel_map.value("R2SCAN_X"), polar));
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::Backend::builtin,
                                            ExchCXX::kernel_map.value("R2SCAN_C"), polar));
      }
      else {
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string("MGGA_X_R2SCAN"), polar));
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string("MGGA_C_R2SCAN"), polar));
      }
      // Try using the builtin backend
    }
    else if(ExchCXX::kernel_map.key_exists(xcfunc)) {
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value(xcfunc), polar));
      // Fallback to LibXC
    }
    else {
      kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string(xcfunc), polar));
      if(txt_utils::strequal_case(xcfunc, "HYB_GGA_X_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_GGA_XC_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_MGGA_XC_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_MGGA_X_QED")) {
        // kernel_id = kernels.size() - 1;
      }
    }
  }

  // QED functionals need lambdas and omegas
  // SCFQed scf_qed
  // if(is_qed && kernel_id > -1) scf_qed.qed_functionals_setup(params, chem_env);
  // gauxc_integrator.set_ext_params( params );

  // Setup dummy functional for snK calculations and no XC functional
  bool dummy_xc{false};
  if(scf_options.snK && kernels.size() < 1) {
    kernels.push_back(
      ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value("PBE_X"), polar));
    dummy_xc = true;
  }

  GauXC::functional_type gauxc_func = GauXC::functional_type(kernels);

  // Initialize GauXC integrator
  GauXC::XCIntegratorFactory<Matrix> integrator_factory(
    int_exec_space, "Replicated", scf_options.xc_int_kernel, scf_options.xc_red_kernel,
    scf_options.xc_lwd_kernel);

  auto gc2     = std::chrono::high_resolution_clock::now();
  auto gc_time = std::chrono::duration_cast<std::chrono::duration<double>>((gc2 - gc1)).count();

  double xHF = dummy_xc ? 1.0 : (gauxc_func.is_hyb() ? gauxc_func.hyb_exx() : 0.0);

  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2) << "GauXC setup time: " << gc_time << "s\n";
  return std::make_tuple(integrator_factory.get_shared_instance(gauxc_func, gauxc_lb), xHF);
};

GauXC::Molecule exachem::scf::gauxc::make_gauxc_molecule(const std::vector<libint2::Atom>& atoms) {
  GauXC::Molecule mol;
  mol.resize(atoms.size());
  std::transform(atoms.begin(), atoms.end(), mol.begin(), [](const libint2::Atom& atom) {
    GauXC::Atom gauxc_atom(GauXC::AtomicNumber(atom.atomic_number), atom.x, atom.y, atom.z);
    return gauxc_atom;
  });
  return mol;
}

GauXC::BasisSet<double> exachem::scf::gauxc::make_gauxc_basis(const libint2::BasisSet& basis,
                                                              const double             basis_tol) {
  using shell_t = GauXC::Shell<double>;
  using prim_t  = typename shell_t::prim_array;
  using cart_t  = typename shell_t::cart_array;

  GauXC::BasisSet<double> gauxc_basis;
  for(const auto& shell: basis) {
    prim_t prim_array, coeff_array;
    cart_t origin;

    std::copy(shell.alpha.begin(), shell.alpha.end(), prim_array.begin());
    std::copy(shell.contr[0].coeff.begin(), shell.contr[0].coeff.end(), coeff_array.begin());
    std::copy(shell.O.begin(), shell.O.end(), origin.begin());

#if defined(GAUXC_HAS_DEVICE)
    gauxc_basis.emplace_back(GauXC::PrimSize(shell.alpha.size()),
                             GauXC::AngularMomentum(shell.contr[0].l), GauXC::SphericalType(false),
                             prim_array, coeff_array, origin, false);
#else
    gauxc_basis.emplace_back(
      GauXC::PrimSize(shell.alpha.size()), GauXC::AngularMomentum(shell.contr[0].l),
      GauXC::SphericalType(shell.contr[0].pure), prim_array, coeff_array, origin, false);
#endif
  }
  // gauxc_basis.generate_shell_to_ao();
  for(auto& sh: gauxc_basis) sh.set_shell_tolerance(basis_tol);

  return gauxc_basis;
}

template<typename TensorType>
void exachem::scf::gauxc::compute_exx(ExecutionContext& ec, ChemEnv& chem_env, SCFVars& scf_vars,
                                      exachem::scf::TAMMTensors&   ttensors,
                                      exachem::scf::EigenTensors&  etensors,
                                      GauXC::XCIntegrator<Matrix>& xc_integrator) {
  const SystemData& sys_data    = chem_env.sys_data;
  const SCFOptions& scf_options = chem_env.ioptions.scf_options;

  Scheduler              sch{ec};
  const TiledIndexSpace& tAO = scf_vars.tAO;
  auto [mu, nu]              = tAO.labels<2>("all");

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  auto       rank0  = ec.pg().rank() == 0;
  auto       factor = is_rhf ? 0.5 * scf_vars.xHF : scf_vars.xHF;

  GauXC::IntegratorSettingsSNLinK sn_link_settings;
  sn_link_settings.energy_tol = scf_options.xc_snK_etol;
  sn_link_settings.k_tol      = scf_options.xc_snK_ktol;

#ifdef GAUXC_HAS_DEVICE
  exachem::scf::SCFCompute scf_compute;
  Matrix&                  D_alpha = etensors.D_alpha_cart;
  Matrix&                  K_alpha = etensors.VXC_alpha_cart;
  Matrix&                  D_beta  = etensors.D_beta_cart;
  Matrix&                  K_beta  = etensors.VXC_beta_cart;
  const libint2::BasisSet& shells  = chem_env.shells;
  scf_compute.compute_sdens_to_cdens<TensorType>(shells, etensors.D_alpha, D_alpha, etensors);
  if(is_uhf)
    scf_compute.compute_sdens_to_cdens<TensorType>(shells, etensors.D_beta, D_beta, etensors);
#else
  Matrix& D_alpha   = etensors.D_alpha;
  Matrix& K_alpha   = etensors.G_alpha;
  Matrix& D_beta    = etensors.D_beta;
  Matrix& K_beta    = etensors.G_beta;
#endif

  K_alpha = xc_integrator.eval_exx(factor * D_alpha, sn_link_settings);
#ifdef GAUXC_HAS_DEVICE
  scf_compute.compute_cpot_to_spot<TensorType>(shells, etensors.G_alpha, K_alpha, etensors);
  K_alpha.resize(0, 0);
  D_alpha.resize(0, 0);
#endif
  if(rank0) eigen_to_tamm_tensor(ttensors.F_alpha_tmp, etensors.G_alpha);
  ec.pg().barrier();

  // clang-format off
  sch
    (ttensors.F_alpha(mu, nu) -= 0.5*ttensors.F_alpha_tmp(mu, nu))
    (ttensors.F_alpha(mu, nu) -= 0.5*ttensors.F_alpha_tmp(nu, mu)).execute();
  // clang-format on

  if(is_uhf) {
    K_beta = xc_integrator.eval_exx(factor * D_beta, sn_link_settings);
#ifdef GAUXC_HAS_DEVICE
    scf_compute.compute_cpot_to_spot<TensorType>(shells, etensors.G_beta, K_beta, etensors);
    K_beta.resize(0, 0);
    D_beta.resize(0, 0);
#endif

    if(rank0) eigen_to_tamm_tensor(ttensors.F_beta_tmp, etensors.G_beta);
    ec.pg().barrier();

    // clang-format off
    sch
      (ttensors.F_beta(mu, nu) -= 0.5*ttensors.F_beta_tmp(mu, nu))
      (ttensors.F_beta(mu, nu) -= 0.5*ttensors.F_beta_tmp(nu, mu)).execute();
    // clang-format on
  }
}

template<typename TensorType>
TensorType exachem::scf::gauxc::compute_xcf(ExecutionContext& ec, ChemEnv& chem_env,
                                            exachem::scf::TAMMTensors&   ttensors,
                                            exachem::scf::EigenTensors&  etensors,
                                            GauXC::XCIntegrator<Matrix>& xc_integrator) {
  SystemData& sys_data = chem_env.sys_data;

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  auto       rank0  = ec.pg().rank() == 0;

  double EXC{};

#ifdef GAUXC_HAS_DEVICE
  exachem::scf::SCFCompute scf_compute;
  Matrix&                  D_alpha   = etensors.D_alpha_cart;
  Matrix&                  vxc_alpha = etensors.VXC_alpha_cart;
  Matrix&                  D_beta    = etensors.D_beta_cart;
  Matrix&                  vxc_beta  = etensors.VXC_beta_cart;
  const libint2::BasisSet& shells    = chem_env.shells;
  scf_compute.compute_sdens_to_cdens<TensorType>(shells, etensors.D_alpha, D_alpha, etensors);
  if(is_uhf)
    scf_compute.compute_sdens_to_cdens<TensorType>(shells, etensors.D_beta, D_beta, etensors);
#else
  Matrix& D_alpha   = etensors.D_alpha;
  Matrix& vxc_alpha = etensors.G_alpha;
  Matrix& D_beta    = etensors.D_beta;
  Matrix& vxc_beta  = etensors.G_beta;
#endif

  if(is_rhf) {
    std::tie(EXC, vxc_alpha) = xc_integrator.eval_exc_vxc(0.5 * D_alpha);
#ifdef GAUXC_HAS_DEVICE
    scf_compute.compute_cpot_to_spot<TensorType>(shells, etensors.G_alpha, vxc_alpha, etensors);
    vxc_alpha.resize(0, 0);
    D_alpha.resize(0, 0);
#endif
    if(rank0) eigen_to_tamm_tensor(ttensors.VXC_alpha, etensors.G_alpha);
  }
  else if(is_uhf) {
    std::tie(EXC, vxc_alpha, vxc_beta) =
      xc_integrator.eval_exc_vxc((D_alpha + D_beta), (D_alpha - D_beta));
#ifdef GAUXC_HAS_DEVICE
    scf_compute.compute_cpot_to_spot<TensorType>(shells, etensors.G_alpha, vxc_alpha, etensors);
    scf_compute.compute_cpot_to_spot<TensorType>(shells, etensors.G_beta, vxc_beta, etensors);
    vxc_alpha.resize(0, 0);
    vxc_beta.resize(0, 0);
    D_alpha.resize(0, 0);
    D_beta.resize(0, 0);
#endif

    if(rank0) {
      eigen_to_tamm_tensor(ttensors.VXC_alpha, etensors.G_alpha);
      eigen_to_tamm_tensor(ttensors.VXC_beta, etensors.G_beta);
    }
  }
  ec.pg().barrier();

  return EXC;
}

template double exachem::scf::gauxc::compute_xcf<double>(
  ExecutionContext& ec, ChemEnv& chem_env, exachem::scf::TAMMTensors& ttensors,
  exachem::scf::EigenTensors& etensors, GauXC::XCIntegrator<Matrix>& xc_integrator);

template void exachem::scf::gauxc::compute_exx<double>(ExecutionContext& ec, ChemEnv& chem_env,
                                                       SCFVars&                     scf_vars,
                                                       exachem::scf::TAMMTensors&   ttensors,
                                                       exachem::scf::EigenTensors&  etensors,
                                                       GauXC::XCIntegrator<Matrix>& xc_integrator);

#endif
