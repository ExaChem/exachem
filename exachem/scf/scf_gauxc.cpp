#include "scf/scf_gauxc.hpp"

#if defined(USE_GAUXC)

std::tuple<std::shared_ptr<GauXC::XCIntegrator<Matrix>>, double>
exachem::scf::gauxc::setup_gauxc(ExecutionContext& ec, const ChemEnv& chem_env,
                                 const SCFVars& scf_vars) {
  size_t            batch_size  = 512;
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
  auto gauxc_basis = make_gauxc_basis(shells);
  auto gauxc_rt    = GauXC::RuntimeEnvironment(ec.pg().comm());
  auto polar       = is_rhf ? ExchCXX::Spin::Unpolarized : ExchCXX::Spin::Polarized;
  auto grid_type   = GauXC::AtomicGridSizeDefault::UltraFineGrid;
  auto xc_grid_str = scf_options.xc_grid_type;

  std::transform(xc_grid_str.begin(), xc_grid_str.end(), xc_grid_str.begin(), ::tolower);
  if(xc_grid_str == "fine") grid_type = GauXC::AtomicGridSizeDefault::FineGrid;
  else if(xc_grid_str == "superfine") grid_type = GauXC::AtomicGridSizeDefault::SuperFineGrid;

  // This options are set to get good accuracy.
  // [Unpruned, Robust, Treutler]
  auto gauxc_molgrid = GauXC::MolGridFactory::create_default_molgrid(
    gauxc_mol, GauXC::PruningScheme::Robust, GauXC::BatchSize(batch_size),
    GauXC::RadialQuad::MuraKnowles, grid_type);
  auto gauxc_molmeta = std::make_shared<GauXC::MolMeta>(gauxc_mol);

  std::string lb_exec_space_str  = "HOST";
  std::string int_exec_space_str = "HOST";

#ifdef GAUXC_HAS_DEVICE
  std::map<std::string, GauXC::ExecutionSpace> exec_space_map = {
    {"HOST", GauXC::ExecutionSpace::Host}, {"DEVICE", GauXC::ExecutionSpace::Device}};

  auto lb_exec_space  = exec_space_map.at(lb_exec_space_str);
  auto int_exec_space = exec_space_map.at(int_exec_space_str);
#else
  auto lb_exec_space  = GauXC::ExecutionSpace::Host;
  auto int_exec_space = GauXC::ExecutionSpace::Host;
#endif

  // Set the load balancer
  GauXC::LoadBalancerFactory lb_factory(lb_exec_space, "Replicated");
  auto gauxc_lb = lb_factory.get_shared_instance(gauxc_rt, gauxc_mol, gauxc_molgrid, gauxc_basis);

  // Modify the weighting algorithm from the input [Becke, SSF, LKO]
  GauXC::MolecularWeightsSettings mw_settings = {GauXC::XCWeightAlg::LKO, false};
  GauXC::MolecularWeightsFactory  mw_factory(int_exec_space, "Default", mw_settings);
  auto                            mw = mw_factory.get_instance();
  mw.modify_weights(*gauxc_lb);

  std::vector<std::string>       xc_vector = scf_options.xc_type;
  std::vector<ExchCXX::XCKernel> kernels   = {};
  std::vector<double>            params(2049, 0.0);
  int                            kernel_id = -1;

  // TODO: Refactor DFT code path when we eventually enable GauXC by default.
  // is_ks=false, so we setup, but do not run DFT.
  for(std::string& xcfunc: xc_vector) {
    std::transform(xcfunc.begin(), xcfunc.end(), xcfunc.begin(), ::toupper);
    if(rank == 0) std::cout << "Functional: " << xcfunc << std::endl;

    try {
      // Try to setup using the builtin backend.
      kernels.push_back(
        ExchCXX::XCKernel(ExchCXX::Backend::builtin, ExchCXX::kernel_map.value(xcfunc), polar));
    } catch(...) {
      // If the above failed, setup with LibXC backend
      kernels.push_back(ExchCXX::XCKernel(ExchCXX::libxc_name_string(xcfunc), polar));
      if(txt_utils::strequal_case(xcfunc, "HYB_GGA_X_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_GGA_XC_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_MGGA_XC_QED") ||
         txt_utils::strequal_case(xcfunc, "HYB_MGGA_X_QED"))
        kernel_id = kernels.size() - 1;
    }
  }

  // QED functionals need lambdas and omegas
  // SCFQed scf_qed
  // if(is_qed && kernel_id > -1) scf_qed.qed_functionals_setup(params, chem_env);
  // gauxc_integrator.set_ext_params( params );

  GauXC::functional_type gauxc_func = GauXC::functional_type(kernels);

  // Initialize GauXC integrator
  GauXC::XCIntegratorFactory<Matrix> integrator_factory(int_exec_space, "Replicated", "Default",
                                                        "Default", "Default");
  auto                               gc2 = std::chrono::high_resolution_clock::now();
  auto gc_time = std::chrono::duration_cast<std::chrono::duration<double>>((gc2 - gc1)).count();

  double xHF = gauxc_func.is_hyb() ? gauxc_func.hyb_exx() : 0.0;

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

GauXC::BasisSet<double> exachem::scf::gauxc::make_gauxc_basis(const libint2::BasisSet& basis) {
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

    gauxc_basis.emplace_back(
      GauXC::PrimSize(shell.alpha.size()), GauXC::AngularMomentum(shell.contr[0].l),
      GauXC::SphericalType(shell.contr[0].pure), prim_array, coeff_array, origin, false);
  }
  // gauxc_basis.generate_shell_to_ao();
  return gauxc_basis;
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

  double  EXC{};
  Matrix& vxc_alpha = etensors.G_alpha;
  Matrix& vxc_beta  = etensors.G_beta;

  if(is_rhf) {
    std::tie(EXC, vxc_alpha) = xc_integrator.eval_exc_vxc(0.5 * etensors.D_alpha);
    if(rank0) eigen_to_tamm_tensor(ttensors.VXC_alpha, vxc_alpha);
  }
  else if(is_uhf) {
    std::tie(EXC, vxc_alpha, vxc_beta) = xc_integrator.eval_exc_vxc(
      (etensors.D_alpha + etensors.D_beta), (etensors.D_alpha - etensors.D_beta));
    if(rank0) {
      eigen_to_tamm_tensor(ttensors.VXC_alpha, vxc_alpha);
      eigen_to_tamm_tensor(ttensors.VXC_beta, vxc_beta);
    }
  }
  ec.pg().barrier();

  return EXC;
}

template double exachem::scf::gauxc::compute_xcf<double>(
  ExecutionContext& ec, ChemEnv& chem_env, exachem::scf::TAMMTensors& ttensors,
  exachem::scf::EigenTensors& etensors, GauXC::XCIntegrator<Matrix>& xc_integrator);

#endif
