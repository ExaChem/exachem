/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

//*******************************************************
// Compute intial guess -> D
// for each iter:
// 1. 2 body fock procedure -> computes G (combined JK)
// 2. [EXC, VXC] = xc_integrator.eval_exc_vxc(D)
// 3. F = H + G
// 4. F += VXC
// 5. E = 0.5 * Tr((H+F) * D)
// 6. E += EXC
// 7. diagonalize F -> updates D
// 8. E += enuc, print E
//*******************************************************

#include "scf_main.hpp"

std::tuple<SystemData, double, libint2::BasisSet, std::vector<size_t>, Tensor<TensorType>,
           Tensor<TensorType>, Tensor<TensorType>, Tensor<TensorType>, TiledIndexSpace,
           TiledIndexSpace, bool>
hartree_fock(ExecutionContext& exc, const string filename, OptionsMap options_map) {
  using libint2::Atom;
  using libint2::BasisSet;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  /*** Setup options ***/

  SystemData sys_data{options_map, options_map.scf_options.scf_type};
  SCFOptions scf_options = sys_data.options_map.scf_options;

  std::string basis         = scf_options.basis;
  int         charge        = scf_options.charge;
  int         multiplicity  = scf_options.multiplicity;
  int         maxiter       = scf_options.maxiter;
  double      conve         = scf_options.conve;
  double      convd         = scf_options.convd;
  bool        debug         = scf_options.debug;
  auto        restart       = scf_options.restart;
  bool        is_spherical  = (scf_options.gaussian_type == "spherical");
  bool        sad           = scf_options.sad;
  auto        iter          = 0;
  auto        rank          = exc.pg().rank();
  const bool  molden_exists = !scf_options.moldenfile.empty();
  const int   restart_size  = scf_options.restart_size;

  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  const bool is_ks  = sys_data.is_ks;
  // const bool is_rohf = sys_data.is_restricted_os;
  if(is_ks && is_uhf) tamm_terminate("UKS-DFT is currently not supported!");

  bool molden_file_valid = false;
  if(molden_exists) {
    molden_file_valid = std::filesystem::exists(scf_options.moldenfile);
    if(!molden_file_valid)
      tamm_terminate("ERROR: moldenfile provided: " + scf_options.moldenfile + " does not exist");
  }

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  // Initialize the Libint integrals library
  libint2::initialize(false);
  // libint2::Shell::do_enforce_unit_normalization(false);

  // Create the basis set
  std::string basis_set_file = std::string(DATADIR) + "/basis/" + basis + ".g94";

  int basis_file_exists = 0;
  if(rank == 0) basis_file_exists = std::filesystem::exists(basis_set_file);
  exc.pg().broadcast(&basis_file_exists, 0);

  if(!basis_file_exists)
    tamm_terminate("ERROR: basis set file " + basis_set_file + " does not exist");

  // If starting guess is from a molden file, read the geometry.
  if(molden_file_valid) read_geom_molden(sys_data, sys_data.options_map.options.atoms);

  auto atoms    = sys_data.options_map.options.atoms;
  auto ec_atoms = sys_data.options_map.options.ec_atoms;

  libint2::BasisSet shells;
  {
    std::vector<libint2::Shell> bset_vec;
    for(int i = 0; i < atoms.size(); i++) {
      const auto        Z = atoms[i].atomic_number;
      libint2::BasisSet ashells(ec_atoms[i].basis, {atoms[i]});
      bset_vec.insert(bset_vec.end(), ashells.begin(), ashells.end());
    }
    libint2::BasisSet bset(bset_vec);
    shells = std::move(bset);
  }

  if(is_spherical) shells.set_pure(true);
  else shells.set_pure(false); // use cartesian gaussians

  const size_t N      = shells.nbf();
  auto         nnodes = exc.nnodes();

  sys_data.nbf      = N;
  sys_data.nbf_orig = N;

  std::string out_fp    = options_map.options.output_file_prefix + "." + scf_options.basis;
  std::string files_dir = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type + "/scf";
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  // If using molden file, read the exponents and coefficients and renormalize shells.
  // This modifies the existing basisset object.
  if(molden_exists && molden_file_valid) {
    shells = read_basis_molden(sys_data, shells);
    shells = renormalize_libint_shells(sys_data, shells);
    if(is_spherical) shells.set_pure(true);
    else shells.set_pure(false); // use cartesian gaussians
  }

#if SCF_THROTTLE_RESOURCES
  ProcGroupData pgdata = get_spg_data(exc, N, -1, 50, scf_options.nnodes, -1, 4);
  auto [t_nnodes, hf_nnodes, ppn, hf_nranks, sca_nnodes, sca_nranks] = pgdata.unpack();

#if defined(USE_UPCXX)
  bool         in_new_team = (rank < hf_nranks);
  upcxx::team* gcomm       = exc.pg().team();
  upcxx::team* hf_comm =
    new upcxx::team(gcomm->split(in_new_team ? 0 : upcxx::team::color_none, rank.value()));
#else
  int ranks[hf_nranks];
  for(int i = 0; i < hf_nranks; i++) ranks[i] = i;
  auto      gcomm = exc.pg().comm();
  MPI_Group wgroup;
  MPI_Comm_group(gcomm, &wgroup);
  MPI_Group hfgroup;
  MPI_Group_incl(wgroup, hf_nranks, ranks, &hfgroup);
  MPI_Comm hf_comm;
  MPI_Comm_create(gcomm, hfgroup, &hf_comm);
#endif

#if defined(USE_SCALAPACK)
  MPI_Comm scacomm = get_scalapack_comm(exc, sca_nranks);
#endif
#endif

  if(rank == 0) {
    cout << endl;
#if SCF_THROTTLE_RESOURCES
    cout << "Number of nodes, processes per node used for SCF calculation: " << hf_nnodes << ", "
         << ppn << endl;
#endif
#if defined(USE_SCALAPACK)
    cout << "Number of nodes, processes per node, total processes used for Scalapack operations: "
         << sca_nnodes << ", " << sca_nranks / sca_nnodes << ", " << sca_nranks << endl;
#endif
    scf_options.print();
  }

  SCFVars scf_vars; // init vars

  // Compute Nuclear repulsion energy.
  auto [nelectrons, enuc] = compute_NRE(exc, atoms);
  // Might be actually useful to store?
  sys_data.results["output"]["SCF"]["nucl_rep_energy"] = enuc;

  // Compute number of electrons.
  nelectrons -= charge;
  sys_data.nelectrons = nelectrons;
  EXPECTS((nelectrons + scf_options.multiplicity - 1) % 2 == 0);

  sys_data.nelectrons_alpha = (nelectrons + scf_options.multiplicity - 1) / 2;
  sys_data.nelectrons_beta  = nelectrons - sys_data.nelectrons_alpha;

  if(rank == 0) {
    std::cout << std::endl << "Number of basis functions = " << N << std::endl;
    std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
    std::cout << "  # of alpha electrons    = " << sys_data.nelectrons_alpha << std::endl;
    std::cout << "  # of beta electons      = " << sys_data.nelectrons_beta << std::endl;
    std::cout << std::endl
              << "Nuclear repulsion energy  = " << std::setprecision(15) << enuc << std::endl
              << std::endl;
  }

  // Compute non-negligible shell-pair list
  compute_shellpair_list(exc, shells, scf_vars);

  // Setup tiled index spaces
  IndexSpace AO{range(0, N)};
  recompute_tilesize(sys_data.options_map.scf_options.AO_tilesize, N,
                     sys_data.options_map.scf_options.force_tilesize, rank == 0);
  std::tie(scf_vars.shell_tile_map, scf_vars.AO_tiles, scf_vars.AO_opttiles) =
    compute_AO_tiles(exc, sys_data, shells);
  scf_vars.tAO                                       = {AO, scf_vars.AO_opttiles};
  scf_vars.tAOt                                      = {AO, scf_vars.AO_tiles};
  std::tie(scf_vars.mu, scf_vars.nu, scf_vars.ku)    = scf_vars.tAO.labels<3>("all");
  std::tie(scf_vars.mup, scf_vars.nup, scf_vars.kup) = scf_vars.tAOt.labels<3>("all");

  auto mu = scf_vars.mu, nu = scf_vars.nu, ku = scf_vars.ku;
  auto mup = scf_vars.mup, nup = scf_vars.nup, kup = scf_vars.kup;

  Scheduler schg{exc};
  // Fock matrices allocated on world group
  Tensor<TensorType> Fa_global{scf_vars.tAO, scf_vars.tAO};
  Tensor<TensorType> Fb_global{scf_vars.tAO, scf_vars.tAO};
  schg.allocate(Fa_global);
  if(is_uhf) schg.allocate(Fb_global);
  schg.execute();

  // If a fitting basis is provided, perform the necessary setup
  const auto dfbasisname        = scf_options.dfbasis;
  bool       is_3c_init         = false;
  bool       do_density_fitting = false;

  if(!dfbasisname.empty()) do_density_fitting = true;
  if(do_density_fitting) {
    scf_vars.dfbs = BasisSet(dfbasisname, atoms);
    if(is_spherical) scf_vars.dfbs.set_pure(true);
    else scf_vars.dfbs.set_pure(false); // use cartesian gaussians

    if(rank == 0) cout << "density-fitting basis set rank = " << scf_vars.dfbs.nbf() << endl;
// compute DFBS non-negligible shell-pair list
#if 0
      {
        //TODO: Doesn't work to screen - revisit
        std::tie(scf_vars.dfbs_shellpair_list, scf_vars.dfbs_shellpair_data) = compute_shellpairs(scf_vars.dfbs);
        size_t nsp = 0;
        for (auto& sp : scf_vars.dfbs_shellpair_list) {
          nsp += sp.second.size();
        }
        if(rank==0) std::cout << "# of {all,non-negligible} DFBS shell-pairs = {"
                  << scf_vars.dfbs.size() * (scf_vars.dfbs.size() + 1) / 2 << "," << nsp << "}"
                  << endl;
      }
#endif

    sys_data.ndf  = scf_vars.dfbs.nbf();
    scf_vars.dfAO = IndexSpace{range(0, sys_data.ndf)};
    recompute_tilesize(sys_data.options_map.scf_options.dfAO_tilesize, sys_data.ndf,
                       sys_data.options_map.scf_options.force_tilesize, rank == 0);
    std::tie(scf_vars.df_shell_tile_map, scf_vars.dfAO_tiles, scf_vars.dfAO_opttiles) =
      compute_AO_tiles(exc, sys_data, scf_vars.dfbs, true);

    scf_vars.tdfAO  = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_opttiles};
    scf_vars.tdfAOt = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_tiles};
  }
  std::unique_ptr<DFFockEngine> dffockengine(
    do_density_fitting ? new DFFockEngine(shells, scf_vars.dfbs) : nullptr);
  // End setup for fitting basis

  auto   hf_t2 = std::chrono::high_resolution_clock::now();
  double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "Time for initial setup: " << hf_time << " secs" << endl;

  double ehf = 0.0; // initialize Hartree-Fock energy

  exc.pg().barrier();

  EigenTensors etensors;
  TAMMTensors  ttensors;

  const bool scf_conv = restart && scf_options.noscf;
  bool       is_conv  = true;

  scf_restart_test(exc, sys_data, filename, restart, files_prefix);

  std::string movecsfile_alpha  = files_prefix + ".alpha.movecs";
  std::string densityfile_alpha = files_prefix + ".alpha.density";
  std::string movecsfile_beta   = files_prefix + ".beta.movecs";
  std::string densityfile_beta  = files_prefix + ".beta.density";

#if SCF_THROTTLE_RESOURCES
  if(rank < hf_nranks) {
#if defined(USE_UPCXX)
    ProcGroup pg = ProcGroup::create_coll(*hf_comm);
#else
    EXPECTS(hf_comm != MPI_COMM_NULL);
    ProcGroup pg = ProcGroup::create_coll(hf_comm);
#endif
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
#else
  // TODO: Fix - create ec_m, when throttle is disabled
  ExecutionContext& ec  = exc;
#endif
    // ProcGroup pg_l = ProcGroup::create_coll(MPI_COMM_SELF);
    // ExecutionContext ec_l{pg_l, DistributionKind::nw, MemoryManagerKind::local};

    ScalapackInfo scalapack_info;
#if defined(USE_SCALAPACK)
    setup_scalapack_info(sys_data, scalapack_info, scacomm);
#endif

#if defined(USE_GAUXC)
    /*** =========================== ***/
    /*** Setup GauXC types           ***/
    /*** =========================== ***/
    size_t batch_size  = 512;
    auto   gauxc_mol   = gauxc_util::make_gauxc_molecule(atoms);
    auto   gauxc_basis = gauxc_util::make_gauxc_basis(shells);
    auto   gauxc_rt    = GauXC::RuntimeEnvironment(ec.pg().comm());

    // This options are set to get good accuracy.
    // TODO: Modify these from the input [Unpruned, Robust, Treutler]
    auto gauxc_molgrid = GauXC::MolGridFactory::create_default_molgrid(
      gauxc_mol, GauXC::PruningScheme::Treutler, GauXC::BatchSize(batch_size),
      GauXC::RadialQuad::MuraKnowles, GauXC::AtomicGridSizeDefault::UltraFineGrid);
    auto gauxc_molmeta = std::make_shared<GauXC::MolMeta>(gauxc_mol);

    // Set the load balancer
    GauXC::LoadBalancerFactory lb_factory(GauXC::ExecutionSpace::Host, "Replicated");
    auto gauxc_lb = lb_factory.get_shared_instance(gauxc_rt, gauxc_mol, gauxc_molgrid, gauxc_basis);

    // TODO: Modify the weighting algorithm from the input [Becke, SSF, LKO]
    GauXC::MolecularWeightsSettings mw_settings = {GauXC::XCWeightAlg::SSF, false};
    GauXC::MolecularWeightsFactory  mw_factory(GauXC::ExecutionSpace::Host, "Default", mw_settings);
    auto                            mw = mw_factory.get_instance();
    mw.modify_weights(*gauxc_lb);

    std::vector<std::string>       xc_vector = scf_options.xc_type;
    std::vector<ExchCXX::XCKernel> kernels   = {};
    std::vector<double>            params(2049, 0.0);
    int                            kernel_id = -1;

    // TODO: Refactor DFT code path when we eventually enable GauXC by default.
    // is_ks=false, so we setup, but do not run DFT.
    if(xc_vector.empty()) xc_vector = {"PBE0"};
    for(std::string& xcfunc: xc_vector) {
      std::transform(xcfunc.begin(), xcfunc.end(), xcfunc.begin(), ::toupper);
      if(rank == 0) cout << "Functional: " << xcfunc << endl;

      try {
        // Try to setup using the builtin backend.
        kernels.push_back(ExchCXX::XCKernel(ExchCXX::Backend::builtin,
                                            ExchCXX::kernel_map.value(xcfunc),
                                            ExchCXX::Spin::Unpolarized));
      } catch(...) {
        // If the above failed, setup with LibXC backend
        kernels.push_back(
          ExchCXX::XCKernel(ExchCXX::libxc_name_string(xcfunc), ExchCXX::Spin::Unpolarized));
        if(strequal_case(xcfunc, "HYB_GGA_X_QED") || strequal_case(xcfunc, "HYB_GGA_XC_QED") ||
           strequal_case(xcfunc, "HYB_MGGA_XC_QED") || strequal_case(xcfunc, "HYB_MGGA_X_QED"))
          kernel_id = kernels.size() - 1;
      }
    }

    GauXC::functional_type gauxc_func = GauXC::functional_type(kernels);

    // Initialize GauXC integrator
    GauXC::XCIntegratorFactory<Matrix> integrator_factory(GauXC::ExecutionSpace::Host, "Replicated",
                                                          "Default", "Default", "Default");
    auto gauxc_integrator = integrator_factory.get_instance(gauxc_func, gauxc_lb);

    // TODO
    const double xHF = is_ks ? (gauxc_func.is_hyb() ? gauxc_func.hyb_exx() : 0.) : 1.;
    if(rank == 0) cout << "HF exch = " << xHF << endl;
#else
  const double      xHF = 1.;
#endif

    ec.pg().barrier();

    Scheduler sch{ec};

    const TiledIndexSpace& tAO  = scf_vars.tAO;
    const TiledIndexSpace& tAOt = scf_vars.tAOt;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/
    compute_hamiltonian<TensorType>(ec, scf_vars, atoms, shells, ttensors, etensors);

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    std::string ortho_file  = files_prefix + ".orthogonalizer";
    std::string ortho_jfile = ortho_file + ".json";

    if(N >= restart_size && fs::exists(ortho_file)) {
      if(rank == 0) {
        cout << "Reading orthogonalizer from disk ..." << endl << endl;
        auto jX           = json_from_file(ortho_jfile);
        auto Xdims        = jX["ortho_dims"].get<std::vector<int>>();
        sys_data.n_lindep = sys_data.nbf_orig - Xdims[1];
      }
      ec.pg().broadcast(&sys_data.n_lindep, 0);
      sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep; // Compute Northo

      scf_vars.tAO_ortho = TiledIndexSpace{IndexSpace{range(0, (size_t) (sys_data.nbf))},
                                           sys_data.options_map.scf_options.AO_tilesize};

#if defined(USE_SCALAPACK)
      {
        const tamm::Tile _mb = scf_options.scalapack_nb; //(scalapack_info.blockcyclic_dist)->mb();
        scf_vars.tN_bc       = TiledIndexSpace{IndexSpace{range(sys_data.nbf_orig)}, _mb};
        scf_vars.tNortho_bc  = TiledIndexSpace{IndexSpace{range(sys_data.nbf)}, _mb};
        if(scacomm != MPI_COMM_NULL) {
          ttensors.X_alpha = {scf_vars.tN_bc, scf_vars.tNortho_bc};
          ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
          Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_alpha);
          read_from_disk<TensorType>(ttensors.X_alpha, ortho_file);
          if(is_uhf) {
            ttensors.X_beta = {scf_vars.tN_bc, scf_vars.tNortho_bc};
            ttensors.X_beta.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
            Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_beta);
            Scheduler{scalapack_info.ec}(ttensors.X_beta() = ttensors.X_alpha()).execute();
            // read_from_disk<TensorType>(ttensors.X_beta,ortho_file);
          }
        }
      }
#else
    ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
    if(is_uhf) ttensors.X_beta = {scf_vars.tAO, scf_vars.tAO_ortho};
    sch.allocate(ttensors.X_alpha).execute();
    if(is_uhf) sch.allocate(ttensors.X_beta).execute();
    read_from_disk<TensorType>(ttensors.X_alpha, ortho_file);
    if(is_uhf) read_from_disk<TensorType>(ttensors.X_beta, ortho_file);
#endif
    }
    else {
      compute_orthogonalizer(ec, sys_data, scf_vars, scalapack_info, ttensors);
      // sys_data.nbf = sys_data.nbf_orig - sys_data.n_lindep;
      if(rank == 0) {
        json jX;
        jX["ortho_dims"] = {sys_data.nbf_orig, sys_data.nbf};
        json_to_file(jX, ortho_jfile);
      }
      if(N >= restart_size) {
#if defined(USE_SCALAPACK)
        if(scacomm != MPI_COMM_NULL) write_to_disk<TensorType>(ttensors.X_alpha, ortho_file);
#else
      write_to_disk<TensorType>(ttensors.X_alpha, ortho_file);
#endif
      }
    }

#if defined(USE_SCALAPACK)
    if(scacomm != MPI_COMM_NULL) {
      ttensors.F_BC = {scf_vars.tN_bc, scf_vars.tN_bc};
      ttensors.F_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.F_BC);
    }
#endif

    etensors.D = Matrix::Zero(N, N);
    etensors.G = Matrix::Zero(N, N);
    if(is_uhf) {
      etensors.D_beta = Matrix::Zero(N, N);
      etensors.G_beta = Matrix::Zero(N, N);
    }

    // pre-compute data for Schwarz bounds
    std::string schwarz_matfile = files_prefix + ".schwarz";
    Matrix      SchwarzK;
    if(N >= restart_size && fs::exists(schwarz_matfile)) {
      if(rank == 0) cout << "Read Schwarz matrix from disk... " << endl;
      SchwarzK = read_scf_mat<TensorType>(schwarz_matfile);
    }
    else {
      // if(rank == 0) cout << "pre-computing data for Schwarz bounds... " << endl;
      SchwarzK = compute_schwarz_ints<>(ec, scf_vars, shells);
      if(rank == 0) write_scf_mat<TensorType>(SchwarzK, schwarz_matfile);
    }

    hf_t1 = std::chrono::high_resolution_clock::now();

    ttensors.ehf_tamm = Tensor<TensorType>{};
    ttensors.F_dummy  = {tAOt, tAOt}; // not allocated

    ttensors.ehf_tmp     = {tAO, tAO};
    ttensors.F_alpha     = {tAO, tAO};
    ttensors.D_tamm      = {tAO, tAO};
    ttensors.D_diff      = {tAO, tAO};
    ttensors.D_last_tamm = {tAO, tAO};
    ttensors.F_alpha_tmp = {tAO, tAO};
    ttensors.FD_tamm     = {tAO, tAO};
    ttensors.FDS_tamm    = {tAO, tAO};

    // XXX: Enable only for DFT
    ttensors.VXC = {tAO, tAO};

    if(is_uhf) {
      ttensors.ehf_beta_tmp     = {tAO, tAO};
      ttensors.F_beta           = {tAO, tAO};
      ttensors.D_beta_tamm      = {tAO, tAO};
      ttensors.D_last_beta_tamm = {tAO, tAO};
      ttensors.F_beta_tmp       = {tAO, tAO};
      ttensors.FD_beta_tamm     = {tAO, tAO};
      ttensors.FDS_beta_tamm    = {tAO, tAO};
    }

#if defined(USE_UPCXX_DISTARRAY)
    ec.set_memory_manager_cache(1);
#endif

    Tensor<TensorType>::allocate(&ec, ttensors.F_alpha, ttensors.D_tamm, ttensors.D_last_tamm,
                                 ttensors.D_diff, ttensors.F_alpha_tmp, ttensors.ehf_tmp,
                                 ttensors.ehf_tamm, ttensors.FD_tamm, ttensors.FDS_tamm);
    if(is_uhf)
      Tensor<TensorType>::allocate(
        &ec, ttensors.F_beta, ttensors.D_beta_tamm, ttensors.D_last_beta_tamm, ttensors.F_beta_tmp,
        ttensors.ehf_beta_tmp, ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm);

    // XXX: Only allocate for DFT
    if(is_ks) Tensor<TensorType>::allocate(&ec, ttensors.VXC);

    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    const libint2::BasisSet& obs        = shells;
    auto                     max_nprim  = obs.max_nprim();
    auto                     max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto                     shell2bf   = obs.shell2bf();

    if(restart) {
      scf_restart(ec, sys_data, filename, etensors, files_prefix);
      if(rank == 0) {
        eigen_to_tamm_tensor(ttensors.X_alpha, etensors.C);                // Xa = Ca;
        if(is_uhf) eigen_to_tamm_tensor(ttensors.X_beta, etensors.C_beta); // Xb = Cb;
      }
    }
    else if(molden_exists) {
      auto N      = sys_data.nbf_orig;
      auto Northo = sys_data.nbf;

      etensors.C.setZero(N, Northo);
      if(is_uhf) etensors.C_beta.setZero(N, Northo);

      if(rank == 0) {
        cout << endl << "Reading from molden file provided ..." << endl;
        if(molden_file_valid) {
          read_molden<TensorType>(sys_data, shells, etensors.C, etensors.C_beta);
        }
      }

      // compute density
      if(rank == 0) {
        if(is_rhf) {
          auto C_occ = etensors.C.leftCols(sys_data.nelectrons_alpha);
          etensors.D = 2.0 * C_occ * C_occ.transpose();
        }
        else if(is_uhf) {
          auto C_occ      = etensors.C.leftCols(sys_data.nelectrons_alpha);
          etensors.D      = C_occ * C_occ.transpose();
          C_occ           = etensors.C_beta.leftCols(sys_data.nelectrons_beta);
          etensors.D_beta = C_occ * C_occ.transpose();
        }
      }

      if(rank == 0) {
        eigen_to_tamm_tensor(ttensors.X_alpha, etensors.C);                // Xa = Ca;
        if(is_uhf) eigen_to_tamm_tensor(ttensors.X_beta, etensors.C_beta); // Xb = Cb;
      }

      ec.pg().barrier();
    }
    else {
// TODO: WIP
#if 0
        if(sad) {
          if(rank==0) cout << "SAD enabled" << endl;

          compute_sad_guess<TensorType>(ec, sys_data, atoms, shells, basis, 
                                       is_spherical, etensors, charge, multiplicity); 
          compute_2bf<TensorType>(ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK,
                                         max_nprim4,shells, ttensors, etensors, false, do_density_fitting);
          sch
            (ttensors.F_alpha()  = ttensors.H1())
            (ttensors.F_alpha() += ttensors.F_alpha_tmp())
            .execute();
          Matrix Fa_eig = tamm_to_eigen_matrix(ttensors.F_alpha);
          Matrix X_eig  = tamm_to_eigen_matrix(ttensors.X_alpha);
          Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_a(X_eig.transpose() * Fa_eig * X_eig);
          auto C_alpha = X_eig * eig_solver_guess_a.eigenvectors();
          auto C_occ_a = C_alpha.leftCols(sys_data.nelectrons_alpha);
          if(is_rhf) 
            etensors.D = 2.0 * C_occ_a * C_occ_a.transpose();
          if(is_uhf) {
            etensors.D = C_occ_a * C_occ_a.transpose();
            sch
              (ttensors.F_beta()  = ttensors.H1())
              (ttensors.F_beta() += ttensors.F_beta_tmp())
              .execute();
            Matrix Fb_eig = tamm_to_eigen_matrix(ttensors.F_beta);
            Eigen::SelfAdjointEigenSolver<Matrix> eig_solver_guess_b(X_eig.transpose() * Fb_eig * X_eig);
            auto C_beta  = X_eig * eig_solver_guess_b.eigenvectors();
            auto C_occ_b = C_beta.leftCols(sys_data.nelectrons_beta);
            etensors.D_beta = C_occ_b * C_occ_b.transpose();
          }
        }
        else
#endif
      {
        auto [s1vec, s2vec, ntask_vec] = compute_initial_guess_taskinfo<TensorType>(
          ec, sys_data, scf_vars, atoms, shells, basis, is_spherical, etensors, ttensors, charge,
          multiplicity);

        auto [s1_all, s2_all, ntasks_all] =
          gather_task_vectors<TensorType>(ec, s1vec, s2vec, ntask_vec);

        int tmdim = 0;
        if(rank == 0) {
          Loads dummyLoads;
          /***generate load balanced task map***/
          readLoads(s1_all, s2_all, ntasks_all, dummyLoads);
          simpleLoadBal(dummyLoads, ec.pg().size().value());
          tmdim = std::max(dummyLoads.maxS1, dummyLoads.maxS2);
          etensors.taskmap.resize(tmdim + 1, tmdim + 1);
          for(int i = 0; i < tmdim + 1; i++) {
            for(int j = 0; j < tmdim + 1; j++) {
              // value in this array is the rank that executes task i,j
              // -1 indicates a task i,j that can be skipped
              etensors.taskmap(i, j) = -1;
            }
          }
          createTaskMap(etensors.taskmap, dummyLoads);
        }

        ec.pg().broadcast(&tmdim, 0);
        if(rank != 0) etensors.taskmap.resize(tmdim + 1, tmdim + 1);
        ec.pg().broadcast(etensors.taskmap.data(), etensors.taskmap.size(), 0);

        compute_initial_guess<TensorType>(ec, scalapack_info, sys_data, scf_vars, atoms, shells,
                                          basis, is_spherical, etensors, ttensors, charge,
                                          multiplicity);

        etensors.taskmap.resize(0, 0);
        if(rank == 0) {
          write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
          write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
          if(is_uhf) {
            write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
            write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
          }
        }
        ec.pg().barrier();
      } // initial guess
    }

    hf_t2   = std::chrono::high_resolution_clock::now();
    hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0)
      std::cout << std::fixed << std::setprecision(2)
                << "Total Time to compute initial guess: " << hf_time << " secs" << endl;

    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/
    double rmsd  = 1.0;
    double ediff = 0.0;

    if(rank == 0) {
      eigen_to_tamm_tensor(ttensors.D_tamm, etensors.D);
      if(is_uhf) { eigen_to_tamm_tensor(ttensors.D_beta_tamm, etensors.D_beta); }
    }

    ec.pg().broadcast(etensors.D.data(), etensors.D.size(), 0);
    if(is_uhf) ec.pg().broadcast(etensors.D_beta.data(), etensors.D_beta.size(), 0);

    if(rank == 0 && scf_options.debug && N < restart_size) {
      Matrix S(sys_data.nbf_orig, sys_data.nbf_orig);
      tamm_to_eigen_tensor(ttensors.S1, S);
      if(is_rhf) cout << "debug #electrons       = " << (etensors.D * S).trace() << endl;
      if(is_uhf) {
        cout << "debug #alpha electrons = " << (etensors.D * S).trace() << endl;
        cout << "debug #beta  electrons = " << (etensors.D_beta * S).trace() << endl;
      }
    }

    // Setup tiled index spaces when a fitting basis is provided
    IndexSpace dfCocc{range(0, sys_data.nelectrons_alpha)};
    scf_vars.tdfCocc             = {dfCocc, sys_data.options_map.scf_options.dfAO_tilesize};
    std::tie(scf_vars.dCocc_til) = scf_vars.tdfCocc.labels<1>("all");
    if(do_density_fitting) {
      std::tie(scf_vars.d_mu, scf_vars.d_nu, scf_vars.d_ku)    = scf_vars.tdfAO.labels<3>("all");
      std::tie(scf_vars.d_mup, scf_vars.d_nup, scf_vars.d_kup) = scf_vars.tdfAOt.labels<3>("all");

      ttensors.Zxy_tamm   = Tensor<TensorType>{scf_vars.tdfAO, tAO, tAO}; // ndf,n,n
      ttensors.xyK_tamm   = Tensor<TensorType>{tAO, tAO, scf_vars.tdfAO}; // n,n,ndf
      ttensors.C_occ_tamm = Tensor<TensorType>{tAO, scf_vars.tdfCocc};    // n,nocc
      Tensor<TensorType>::allocate(&ec, ttensors.xyK_tamm, ttensors.C_occ_tamm);
    }

    if(rank == 0) {
      std::cout << std::endl << std::endl;
      std::cout << " SCF iterations" << endl;
      std::cout << std::string(65, '-') << endl;
      std::string sph = " Iter     Energy            E-Diff       RMSD          Time(s)";
      if(scf_conv) sph = " Iter     Energy            E-Diff       Time(s)";
      std::cout << sph << endl;
      std::cout << std::string(65, '-') << endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    /*** Generate task mapping ***/

    // Collect task info
    auto [s1vec, s2vec, ntask_vec] = compute_2bf_taskinfo<TensorType>(
      ec, sys_data, scf_vars, obs, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4, shells,
      ttensors, etensors, do_density_fitting);

    auto [s1_all, s2_all, ntasks_all] =
      gather_task_vectors<TensorType>(ec, s1vec, s2vec, ntask_vec);

    int tmdim = 0;
    if(rank == 0) {
      Loads dummyLoads;
      /***generate load balanced task map***/
      readLoads(s1_all, s2_all, ntasks_all, dummyLoads);
      simpleLoadBal(dummyLoads, ec.pg().size().value());
      tmdim = std::max(dummyLoads.maxS1, dummyLoads.maxS2);
      etensors.taskmap.resize(tmdim + 1, tmdim + 1);
      // value in this array is the rank that executes task i,j
      // -1 indicates a task i,j that can be skipped
      etensors.taskmap.setConstant(-1);
      // cout<<"creating task map"<<endl;
      createTaskMap(etensors.taskmap, dummyLoads);
      // cout<<"task map creation completed"<<endl;
    }

    ec.pg().broadcast(&tmdim, 0);
    if(rank != 0) etensors.taskmap.resize(tmdim + 1, tmdim + 1);
    ec.pg().broadcast(etensors.taskmap.data(), etensors.taskmap.size(), 0);

    if(restart || molden_exists) {
      sch(ttensors.F_alpha_tmp() = 0).execute();

      if(is_uhf) { sch(ttensors.F_beta_tmp() = 0).execute(); }
      // F1 = H1 + F_alpha_tmp
      compute_2bf<TensorType>(ec, scalapack_info, sys_data, scf_vars, obs, do_schwarz_screen,
                              shell2bf, SchwarzK, max_nprim4, shells, ttensors, etensors,
                              is_3c_init, do_density_fitting, xHF);

      TensorType gauxc_exc = 0.;
#if defined(USE_GAUXC)
      if(is_ks) {
        gauxc_exc = gauxc_util::compute_xcf<TensorType>(ec, ttensors, etensors, gauxc_integrator);
      }
#endif

      // ehf = D * (H1+F1);
      if(is_rhf) {
        // clang-format off
        sch (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_alpha_tmp(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            .execute();
        // clang-format on
      }
      if(is_uhf) {
        // clang-format off
        sch (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_alpha_tmp(mu,nu))
            (ttensors.ehf_tamm()      = 1.0 * ttensors.D_tamm() * ttensors.ehf_tmp())
            (ttensors.ehf_tmp(mu,nu)  = 2.0 * ttensors.H1(mu,nu))
            (ttensors.ehf_tmp(mu,nu) += 1.0 * ttensors.F_beta_tmp(mu,nu))
            (ttensors.ehf_tamm()     += 1.0 * ttensors.D_beta_tamm() * ttensors.ehf_tmp())
            .execute();
        // clang-format on
      }

      ehf = 0.5 * get_scalar(ttensors.ehf_tamm) + enuc + gauxc_exc;
      if(rank == 0)
        std::cout << std::setprecision(18) << "Total HF energy after restart: " << ehf << std::endl;
    }

    // SCF main loop
    do {
      const auto loop_start = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      double ehf_last = ehf;

      // clang-format off
      sch (ttensors.F_alpha_tmp() = 0)
          (ttensors.D_last_tamm(mu,nu) = ttensors.D_tamm(mu,nu))
          .execute();
      // clang-format on

      if(is_uhf) {
        // clang-format off
        sch (ttensors.F_beta_tmp() = 0)
            (ttensors.D_last_beta_tamm(mu,nu) = ttensors.D_beta_tamm(mu,nu))
            .execute();
        // clang-format on
      }

      // auto D_tamm_nrm = norm(ttensors.D_tamm);
      // if(rank==0) cout << std::setprecision(18) << "norm of D_tamm: " << D_tamm_nrm << endl;

      // build a new Fock matrix
      compute_2bf<TensorType>(ec, scalapack_info, sys_data, scf_vars, obs, do_schwarz_screen,
                              shell2bf, SchwarzK, max_nprim4, shells, ttensors, etensors,
                              is_3c_init, do_density_fitting, xHF);

      std::tie(ehf, rmsd) = scf_iter_body<TensorType>(ec, scalapack_info, iter, sys_data, scf_vars,
                                                      ttensors, etensors,
#if defined(USE_GAUXC)
                                                      gauxc_integrator,
#endif
                                                      scf_conv);

      ehf += enuc;
      // compute difference with last iteration
      ediff = ehf - ehf_last;

      const auto loop_stop = std::chrono::high_resolution_clock::now();
      const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

      if(rank == 0) {
        std::cout << std::setw(4) << iter << "  " << std::setw(10);
        if(debug) {
          std::cout << std::fixed << std::setprecision(18) << ehf;
          std::cout << std::scientific << std::setprecision(18);
        }
        else {
          std::cout << std::fixed << std::setprecision(10) << ehf;
          std::cout << std::scientific << std::setprecision(2);
        }
        std::cout << ' ' << std::scientific << std::setw(12) << ediff;
        if(!scf_conv) std::cout << ' ' << std::setw(12) << rmsd << ' ';
        std::cout << ' ' << std::setw(10) << std::fixed << std::setprecision(1) << loop_time << ' '
                  << endl;

        sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)] = {
          {"energy", ehf}, {"e_diff", ediff}, {"rmsd", rmsd}};
        sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)]["performance"] = {
          {"total_time", loop_time}};
      }

      // if(rank==0) cout << "D at the end of iteration: " << endl << std::setprecision(6) <<
      // etensors.D << endl;
      if(rank == 0 && (scf_options.writem % iter == 0 || scf_options.writem == 1)) {
        write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
        write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
        if(is_uhf) {
          write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
          write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
        }
      }

      if(iter >= maxiter) {
        is_conv = false;
        break;
      }

      if(scf_conv) break;

      if(debug) print_energies(ec, ttensors, sys_data, scf_vars, debug);

    } while((fabs(ediff) > conve) || (fabs(rmsd) > convd)); // SCF main loop

    if(rank == 0) {
      std::cout.precision(13);
      if(is_conv) cout << endl << "** Total SCF energy = " << ehf << endl;
      else {
        cout << endl << std::string(50, '*') << endl;
        cout << std::string(10, ' ') << "ERROR: SCF calculation does not converge!!!" << endl;
        cout << std::string(50, '*') << endl;
      }
    }

    if(is_ks) { // or rohf
      sch(ttensors.F_alpha_tmp() = 0).execute();
      if(is_uhf) sch(ttensors.F_beta_tmp() = 0).execute();

      // build a new Fock matrix
      compute_2bf<TensorType>(ec, scalapack_info, sys_data, scf_vars, obs, do_schwarz_screen,
                              shell2bf, SchwarzK, max_nprim4, shells, ttensors, etensors,
                              is_3c_init, do_density_fitting, 1.0);

      // clang-format off
      sch (ttensors.F_alpha()  = ttensors.H1())
          (ttensors.F_alpha() += ttensors.F_alpha_tmp())
          .execute();
      // clang-format on
      if(is_uhf) {
        // clang-format off
        sch (ttensors.F_beta()   = ttensors.H1())
            (ttensors.F_beta()  += ttensors.F_beta_tmp())
            .execute();
        // clang-format on
      }
    }

    for(auto x: ttensors.ehf_tamm_hist) Tensor<TensorType>::deallocate(x);

    for(auto x: ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.D_hist) Tensor<TensorType>::deallocate(x);

    if(is_uhf) {
      for(auto x: ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
      for(auto x: ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
      for(auto x: ttensors.D_beta_hist) Tensor<TensorType>::deallocate(x);
    }

    if(rank == 0)
      std::cout << std::endl
                << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;
    print_energies(ec, ttensors, sys_data, scf_vars);

    if(rank == 0 && !scf_conv) {
      cout << "writing orbitals and density to file... ";
      write_scf_mat<TensorType>(etensors.C, movecsfile_alpha);
      write_scf_mat<TensorType>(etensors.D, densityfile_alpha);
      if(is_uhf) {
        write_scf_mat<TensorType>(etensors.C_beta, movecsfile_beta);
        write_scf_mat<TensorType>(etensors.D_beta, densityfile_beta);
      }
      cout << "done." << endl;
    }

    if(rank == 0 && scf_options.mulliken_analysis && is_conv) {
      Matrix S = tamm_to_eigen_matrix(ttensors.S1);
      print_mulliken(options_map, shells, etensors.D, etensors.D_beta, S, is_uhf);
    }
    // copy to fock matrices allocated on world group
    sch(Fa_global(mu, nu) = ttensors.F_alpha(mu, nu));
    if(is_uhf) sch(Fb_global(mu, nu) = ttensors.F_beta(mu, nu));
    sch.execute();

    if(do_density_fitting) Tensor<TensorType>::deallocate(ttensors.xyK_tamm, ttensors.C_occ_tamm);

    write_to_disk<TensorType>(ttensors.H1, files_prefix + ".hcore");

    Tensor<TensorType>::deallocate(ttensors.H1, ttensors.S1, ttensors.T1, ttensors.V1,
                                   ttensors.F_alpha_tmp, ttensors.ehf_tmp, ttensors.ehf_tamm,
                                   ttensors.F_alpha, ttensors.D_tamm, ttensors.D_diff,
                                   ttensors.D_last_tamm, ttensors.FD_tamm, ttensors.FDS_tamm);

    if(is_uhf)
      Tensor<TensorType>::deallocate(
        ttensors.F_beta, ttensors.D_beta_tamm, ttensors.D_last_beta_tamm, ttensors.F_beta_tmp,
        ttensors.ehf_beta_tmp, ttensors.FD_beta_tamm, ttensors.FDS_beta_tamm);

    if(is_ks) {
      if(rank == 0) etensors.VXC = tamm_to_eigen_matrix(ttensors.VXC);
      Tensor<TensorType>::deallocate(ttensors.VXC);
    }

#if SCF_THROTTLE_RESOURCES
    ec.flush_and_sync();
#endif

#if defined(USE_SCALAPACK)
#if defined(USE_UPCXX)
    abort(); // Not supported currently in UPC++
#endif
    if(scalapack_info.comm != MPI_COMM_NULL) {
      Tensor<TensorType>::deallocate(ttensors.F_BC, ttensors.X_alpha);
      if(is_uhf) Tensor<TensorType>::deallocate(ttensors.X_beta);
      scalapack_info.ec.flush_and_sync();
    }
// Free up created comms / groups
// MPI_Comm_free( &scalapack_comm );
// MPI_Group_free( &scalapack_group );
// MPI_Group_free( &world_group );
#else
  sch.deallocate(ttensors.X_alpha);
  if(is_uhf) sch.deallocate(ttensors.X_beta);
  sch.execute();
#endif

#if SCF_THROTTLE_RESOURCES

  } // end scaled down process group

#if defined(USE_UPCXX)
  hf_comm->destroy();
#endif

#endif

  // C,F1 is not allocated for ranks > hf_nranks
  exc.pg().barrier();
  exc.pg().broadcast(&is_conv, 0);

  if(!is_conv) { tamm_terminate("Please check SCF input parameters"); }

  // F, C are not deallocated.
  sys_data.n_occ_alpha = sys_data.nelectrons_alpha;
  sys_data.n_occ_beta  = sys_data.nelectrons_beta;
  sys_data.n_vir_alpha = sys_data.nbf_orig - sys_data.n_occ_alpha - sys_data.n_lindep;
  sys_data.n_vir_beta  = sys_data.nbf_orig - sys_data.n_occ_beta - sys_data.n_lindep;

  exc.pg().broadcast(&ehf, 0);
  exc.pg().broadcast(&sys_data.nbf, 0);
  exc.pg().broadcast(&sys_data.n_lindep, 0);
  exc.pg().broadcast(&sys_data.n_occ_alpha, 0);
  exc.pg().broadcast(&sys_data.n_vir_alpha, 0);
  exc.pg().broadcast(&sys_data.n_occ_beta, 0);
  exc.pg().broadcast(&sys_data.n_vir_beta, 0);

  sys_data.update();
  if(rank == 0 && debug) sys_data.print();
  // sys_data.input_molecule = getfilename(filename);
  sys_data.scf_energy = ehf;
  // iter not broadcasted, but fine since only rank 0 writes to json
  if(rank == 0) {
    sys_data.results["output"]["SCF"]["final_energy"] = ehf;
    sys_data.results["output"]["SCF"]["n_iterations"] = iter;
  }

  IndexSpace      AO_ortho{range(0, (size_t) (sys_data.nbf_orig - sys_data.n_lindep))};
  TiledIndexSpace tAO_ortho{AO_ortho, sys_data.options_map.scf_options.AO_tilesize};

  Tensor<TensorType> C_alpha_tamm{scf_vars.tAO, tAO_ortho};
  Tensor<TensorType> C_beta_tamm{scf_vars.tAO, tAO_ortho};
  ttensors.VXC = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};

#if defined(USE_UPCXX_DISTARRAY)
  exc.set_memory_manager_cache(1);
#endif

  schg.allocate(C_alpha_tamm);
  if(is_uhf) schg.allocate(C_beta_tamm);
  if(is_ks) schg.allocate(ttensors.VXC);
  schg.execute();
#if defined(USE_UPCXX_DISTARRAY)
  exc.set_memory_manager_cache(); // resets cache to pg.size().value();
#endif

  if(rank == 0) {
    eigen_to_tamm_tensor(C_alpha_tamm, etensors.C);
    if(is_uhf) eigen_to_tamm_tensor(C_beta_tamm, etensors.C_beta);
    if(is_ks) eigen_to_tamm_tensor(ttensors.VXC, etensors.VXC);
  }
  if(is_ks) {
    write_to_disk<TensorType>(ttensors.VXC, files_prefix + ".vxc");
    schg.deallocate(ttensors.VXC);
  }

  exc.pg().barrier();

  return std::make_tuple(sys_data, ehf, shells, scf_vars.shell_tile_map, C_alpha_tamm, Fa_global,
                         C_beta_tamm, Fb_global, scf_vars.tAO, scf_vars.tAOt, scf_conv);
}

void scf(std::string filename, OptionsMap options_map) {
  using T = double;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank = ec.pg().rank();

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt,
        AO_tis, scf_conv] = hartree_fock(ec, filename, options_map);

  Tensor<T>::deallocate(C_AO, F_AO);
  if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO, F_beta_AO);

  if(rank == 0) {
    sys_data.output_file_prefix                   = options_map.options.output_file_prefix;
    sys_data.input_molecule                       = sys_data.output_file_prefix;
    sys_data.results["input"]["molecule"]["name"] = sys_data.output_file_prefix;
    write_json_data(sys_data, "SCF");
  }

  auto hf_t2 = std::chrono::high_resolution_clock::now();

  double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

  ec.flush_and_sync();

  if(rank == 0)
    std::cout << std::endl
              << "Total Time taken for Hartree-Fock: " << std::fixed << std::setprecision(2)
              << hf_time << " secs" << std::endl;
}
