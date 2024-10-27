/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_hartree_fock.hpp"
#include "exachem/common/cutils.hpp"
#include "exachem/common/ec_dplot.hpp"
#include "exachem/common/options/parser_utils.hpp"
#include <functional>

void exachem::scf::SCFHartreeFock::initialize(ExecutionContext& exc, ChemEnv& chem_env) {
  scf_hf(exc, chem_env);
}

void exachem::scf::SCFHartreeFock::scf_hf(ExecutionContext& exc, ChemEnv& chem_env) {
  SCFCompute scf_compute;
  SCFQed     scf_qed;
  SCFIter    scf_iter;
  SCFGuess   scf_guess;
  SCFRestart scf_restart;
  SCFIO      scf_output;
  ECMolden   ec_molden;
  SCFVars    scf_vars; // init vars

  bool is_spherical = (chem_env.ioptions.scf_options.gaussian_type == "spherical");
  auto iter         = 0;
  auto rank         = exc.pg().rank();

  auto hf_t1 = std::chrono::high_resolution_clock::now();

  ec_molden.check_molden(chem_env.ioptions.scf_options.moldenfile);
  if(ec_molden.molden_file_valid) ec_molden.read_geom_molden(chem_env);

  const int N                                               = chem_env.shells.nbf();
  chem_env.sys_data.nbf                                     = N;
  chem_env.sys_data.nbf_orig                                = N;
  chem_env.sys_data.results["output"]["system_info"]["nbf"] = chem_env.sys_data.nbf;

  std::string out_fp       = chem_env.workspace_dir;
  std::string files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type + "/scf";
  std::string files_prefix = files_dir + "/" + chem_env.sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  if(ec_molden.molden_exists && ec_molden.molden_file_valid) {
    chem_env.shells = ec_molden.read_basis_molden(chem_env);
    chem_env.shells = ec_molden.renormalize_libint_shells(chem_env.shells);
    if(is_spherical) chem_env.shells.set_pure(true);
    else chem_env.shells.set_pure(false); // use cartesian gaussians
  }

  // -------------Everythin related to Basis Sets-----------------------------

#if SCF_THROTTLE_RESOURCES
  ProcGroupData pgdata = get_spg_data(exc, N, -1, 50, chem_env.ioptions.scf_options.nnodes);
  auto [t_nnodes, hf_nnodes, ppn, hf_nranks] = pgdata.unpack();

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
  MPI_Group_free(&wgroup);
  MPI_Group_free(&hfgroup);
#endif

#endif

  if(rank == 0) {
    std::cout << std::endl;
#if SCF_THROTTLE_RESOURCES
    std::cout << "Number of nodes, processes per node used for SCF calculation: " << hf_nnodes
              << ", " << ppn << std::endl;
#endif
    chem_env.ioptions.common_options.print();
    chem_env.ioptions.scf_options.print();
  }

  // SCFVars scf_vars; // init vars
  scf_vars.lshift = chem_env.ioptions.scf_options.lshift;

  const double fock_precision =
    std::min(chem_env.ioptions.scf_options.tol_sch, 1e-2 * chem_env.ioptions.scf_options.conve);
  if(fock_precision < chem_env.ioptions.scf_options.tol_sch) {
    chem_env.ioptions.scf_options.tol_sch = fock_precision;
    if(rank == 0) cout << "Resetting tol_sch to " << fock_precision << endl;
  }
#if defined(USE_GAUXC)
  if(chem_env.ioptions.scf_options.snK) {
    if(chem_env.ioptions.scf_options.xc_snK_etol > chem_env.ioptions.scf_options.conve) {
      chem_env.ioptions.scf_options.xc_snK_etol = chem_env.ioptions.scf_options.conve;
      if(rank == 0)
        cout << "Resetting xc_snK_etol to " << chem_env.ioptions.scf_options.conve << endl;
    }
    if(chem_env.ioptions.scf_options.xc_snK_ktol > fock_precision) {
      chem_env.ioptions.scf_options.xc_snK_ktol = fock_precision;
      if(rank == 0) cout << "Resetting xc_snK_ktol to " << fock_precision << endl;
    }
  }
  if(chem_env.ioptions.scf_options.xc_basis_tol > chem_env.ioptions.scf_options.conve) {
    chem_env.ioptions.scf_options.xc_basis_tol = fock_precision;
    if(rank == 0)
      cout << "Resetting xc_basis_tol to " << chem_env.ioptions.scf_options.conve << endl;
  }
#endif

  // Compute Nuclear repulsion energy.
  auto [nelectrons, enuc] = scf_compute.compute_NRE(exc, chem_env.atoms);
  // Might be actually useful to store?
  chem_env.sys_data.results["output"]["SCF"]["nucl_rep_energy"] = enuc;

  // Compute number of electrons.
  nelectrons -= chem_env.ioptions.scf_options.charge;

  // Account for ECPs
  // if(chem_env.sys_data.has_ecp)
  //  for(auto k: chem_env.ec_atoms)
  //    if(k.has_ecp)
  //      nelectrons -= k.ecp_nelec;
  chem_env.sys_data.nelectrons = nelectrons;
  if((nelectrons + chem_env.ioptions.scf_options.multiplicity - 1) % 2 != 0) {
    std::string err_msg =
      "[ERROR] Number of electrons (" + std::to_string(nelectrons) + ") " + "and multiplicity (" +
      std::to_string(chem_env.ioptions.scf_options.multiplicity) + ") " + " not compatible!";
    tamm_terminate(err_msg);
  }

  chem_env.sys_data.nelectrons_alpha =
    (nelectrons + chem_env.ioptions.scf_options.multiplicity - 1) / 2;
  chem_env.sys_data.nelectrons_beta = nelectrons - chem_env.sys_data.nelectrons_alpha;
  if(rank == 0) {
    std::cout << std::endl << "Number of basis functions = " << N << std::endl;
    std::cout << std::endl << "Total number of shells = " << chem_env.shells.size() << std::endl;
    std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
    std::cout << "  # of alpha electrons    = " << chem_env.sys_data.nelectrons_alpha << std::endl;
    std::cout << "  # of beta electons      = " << chem_env.sys_data.nelectrons_beta << std::endl;
    std::cout << std::endl
              << "Nuclear repulsion energy  = " << std::setprecision(15) << enuc << std::endl
              << std::endl;

    chem_env.sys_data.results["output"]["system_info"]["nshells"]          = chem_env.shells.size();
    chem_env.sys_data.results["output"]["system_info"]["nelectrons_total"] = nelectrons;
    chem_env.sys_data.results["output"]["system_info"]["nelectrons_alpha"] =
      chem_env.sys_data.nelectrons_alpha;
    chem_env.sys_data.results["output"]["system_info"]["nelectrons_beta"] =
      chem_env.sys_data.nelectrons_beta;
    chem_env.write_sinfo();
  }

  // Compute non-negligible shell-pair list

  scf_compute.compute_shellpair_list(exc, chem_env.shells, scf_vars);

  // Setup tiled index spaces
  IndexSpace AO{range(0, N)};
  scf_compute.recompute_tilesize(chem_env.ioptions.scf_options.AO_tilesize, N,
                                 chem_env.ioptions.scf_options.force_tilesize, rank == 0);
  std::tie(scf_vars.shell_tile_map, scf_vars.AO_tiles, scf_vars.AO_opttiles) =
    scf_compute.compute_AO_tiles(exc, chem_env, chem_env.shells);
  scf_vars.tAO                                       = {AO, scf_vars.AO_opttiles};
  scf_vars.tAOt                                      = {AO, scf_vars.AO_tiles};
  std::tie(scf_vars.mu, scf_vars.nu, scf_vars.ku)    = scf_vars.tAO.labels<3>("all");
  std::tie(scf_vars.mup, scf_vars.nup, scf_vars.kup) = scf_vars.tAOt.labels<3>("all");

  scf_vars.tAO_occ_a = TiledIndexSpace{range(0, chem_env.sys_data.nelectrons_alpha),
                                       chem_env.ioptions.scf_options.AO_tilesize};
  scf_vars.tAO_occ_b = TiledIndexSpace{range(0, chem_env.sys_data.nelectrons_beta),
                                       chem_env.ioptions.scf_options.AO_tilesize};

  std::tie(scf_vars.mu_oa, scf_vars.nu_oa) = scf_vars.tAO_occ_a.labels<2>("all");
  std::tie(scf_vars.mu_ob, scf_vars.nu_ob) = scf_vars.tAO_occ_b.labels<2>("all");

  auto mu = scf_vars.mu, nu = scf_vars.nu, ku = scf_vars.ku;
  auto mup = scf_vars.mup, nup = scf_vars.nup, kup = scf_vars.kup;

  Scheduler schg{exc};
  // Fock matrices allocated on world group
  Tensor<TensorType> Fa_global{scf_vars.tAO, scf_vars.tAO};
  Tensor<TensorType> Fb_global{scf_vars.tAO, scf_vars.tAO};
  schg.allocate(Fa_global);
  if(chem_env.sys_data.is_unrestricted) schg.allocate(Fb_global);
  schg.execute();

  // If a fitting basis is provided, perform the necessary setup

  bool is_3c_init         = false;
  bool do_density_fitting = false;

  if(!chem_env.ioptions.scf_options.dfbasis.empty()) do_density_fitting = true;
  scf_vars.do_dens_fit = do_density_fitting;
  if(do_density_fitting) {
    scf_vars.dfbs = libint2::BasisSet(chem_env.ioptions.scf_options.dfbasis, chem_env.atoms);
    if(is_spherical) scf_vars.dfbs.set_pure(true);
    else scf_vars.dfbs.set_pure(false); // use cartesian gaussians

    if(rank == 0) cout << "density-fitting basis set rank = " << scf_vars.dfbs.nbf() << endl;

    chem_env.sys_data.ndf = scf_vars.dfbs.nbf();
    scf_vars.dfAO         = IndexSpace{range(0, chem_env.sys_data.ndf)};
    scf_compute.recompute_tilesize(chem_env.ioptions.scf_options.dfAO_tilesize,
                                   chem_env.sys_data.ndf,
                                   chem_env.ioptions.scf_options.force_tilesize, rank == 0);
    std::tie(scf_vars.df_shell_tile_map, scf_vars.dfAO_tiles, scf_vars.dfAO_opttiles) =
      scf_compute.compute_AO_tiles(exc, chem_env, scf_vars.dfbs, true);

    scf_vars.tdfAO  = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_opttiles};
    scf_vars.tdfAOt = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_tiles};
    chem_env.sys_data.results["output"]["system_info"]["ndf"] = chem_env.sys_data.ndf;
  }
  std::unique_ptr<DFFockEngine> dffockengine(
    do_density_fitting ? new DFFockEngine(chem_env.shells, scf_vars.dfbs) : nullptr);
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

  bool is_conv = true;
  // const bool scf_vars.do_load_bal = scf_vars.do_scf_vars.do_load_bal;

  // This is originally scf_restart_test
  scf_restart(exc, chem_env, files_prefix);

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
    setup_scalapack_info(ec, chem_env, scalapack_info, pgdata);
    MPI_Comm scacomm = scalapack_info.comm;
#endif

#if defined(USE_GAUXC)
    double                                       xHF;
    std::shared_ptr<GauXC::XCIntegrator<Matrix>> gauxc_integrator_ptr;
    if(chem_env.sys_data.is_ks || chem_env.sys_data.do_snK)
      std::tie(gauxc_integrator_ptr, xHF) = scf::gauxc::setup_gauxc(ec, chem_env, scf_vars);
    else xHF = 1.0;
    auto gauxc_integrator = (chem_env.sys_data.is_ks || chem_env.sys_data.do_snK)
                              ? GauXC::XCIntegrator<Matrix>(std::move(*gauxc_integrator_ptr))
                              : GauXC::XCIntegrator<Matrix>();
    scf_vars.xHF          = xHF;
    if(rank == 0) cout << "HF exch = " << xHF << endl;
#else
  const double      xHF = 1.;
#endif

    chem_env.sys_data.results["output"]["SCF"]["xHF"] = xHF;

    // Compute SPH<->CART transformation
    scf_compute.compute_trafo(chem_env.shells, etensors);

    scf_vars.direct_df = do_density_fitting && chem_env.ioptions.scf_options.direct_df;
    if(scf_vars.direct_df && xHF != 0.0 && !chem_env.sys_data.do_snK) {
      if(rank == 0) {
        cout << "[Warning] Direct DF cannot be used without snK and xHF != 0.0" << endl;
        cout << "Falling back to in-core DF" << endl;
      }
      scf_vars.direct_df = false;
    }

    // SETUP LibECPint
    std::vector<libecpint::ECP>           ecps;
    std::vector<libecpint::GaussianShell> libecp_shells;
    if(chem_env.sys_data.has_ecp) {
      for(auto shell: chem_env.shells) {
        std::array<double, 3>    O = {shell.O[0], shell.O[1], shell.O[2]};
        libecpint::GaussianShell newshell(O, shell.contr[0].l);
        for(size_t iprim = 0; iprim < shell.alpha.size(); iprim++)
          newshell.addPrim(shell.alpha[iprim], shell.contr[0].coeff[iprim]);
        libecp_shells.push_back(newshell);
      }

      for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
        if(chem_env.ec_atoms[i].has_ecp) {
          int maxam = *std::max_element(chem_env.ec_atoms[i].ecp_ams.begin(),
                                        chem_env.ec_atoms[i].ecp_ams.end());
          std::replace(chem_env.ec_atoms[i].ecp_ams.begin(), chem_env.ec_atoms[i].ecp_ams.end(), -1,
                       maxam + 1);

          std::array<double, 3> O = {chem_env.atoms[i].x, chem_env.atoms[i].y, chem_env.atoms[i].z};
          libecpint::ECP        newecp(O.data());
          for(size_t iprim = 0; iprim < chem_env.ec_atoms[i].ecp_coeffs.size(); iprim++) {
            newecp.addPrimitive(
              chem_env.ec_atoms[i].ecp_ns[iprim], chem_env.ec_atoms[i].ecp_ams[iprim],
              chem_env.ec_atoms[i].ecp_exps[iprim], chem_env.ec_atoms[i].ecp_coeffs[iprim], true);
          }
          ecps.push_back(newecp);
        }
      }
    }

    ec.pg().barrier();

    Scheduler sch{ec};

    const TiledIndexSpace& tAO  = scf_vars.tAO;
    const TiledIndexSpace& tAOt = scf_vars.tAOt;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/
    scf_compute.compute_hamiltonian<TensorType>(ec, scf_vars, chem_env, ttensors, etensors);
    if(chem_env.sys_data.is_qed) {
      ttensors.QED_Dx    = {tAO, tAO};
      ttensors.QED_Dy    = {tAO, tAO};
      ttensors.QED_Dz    = {tAO, tAO};
      ttensors.QED_Qxx   = {tAO, tAO};
      ttensors.QED_Qxy   = {tAO, tAO};
      ttensors.QED_Qxz   = {tAO, tAO};
      ttensors.QED_Qyy   = {tAO, tAO};
      ttensors.QED_Qyz   = {tAO, tAO};
      ttensors.QED_Qzz   = {tAO, tAO};
      ttensors.QED_1body = {tAO, tAO};
      ttensors.QED_2body = {tAO, tAO};
      Tensor<TensorType>::allocate(&ec, ttensors.QED_Dx, ttensors.QED_Dy, ttensors.QED_Dz,
                                   ttensors.QED_Qxx, ttensors.QED_Qxy, ttensors.QED_Qxz,
                                   ttensors.QED_Qyy, ttensors.QED_Qyz, ttensors.QED_Qzz,
                                   ttensors.QED_1body, ttensors.QED_2body);

      scf_qed.compute_qed_emult_ints<TensorType>(ec, chem_env, scf_vars, ttensors);
      if(chem_env.sys_data.do_qed)
        scf_qed.compute_QED_1body<TensorType>(ec, chem_env, scf_vars, ttensors);
    }

    if(chem_env.sys_data.has_ecp) {
      Tensor<TensorType> ECP{tAO, tAO};
      Tensor<TensorType>::allocate(&ec, ECP);
      scf_guess.compute_ecp_ints(ec, scf_vars, ECP, libecp_shells, ecps);
      sch(ttensors.H1() += ECP()).deallocate(ECP).execute();
    }

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    std::string ortho_file  = files_prefix + ".orthogonalizer";
    std::string ortho_jfile = ortho_file + ".json";
    if(N >= chem_env.ioptions.scf_options.restart_size && fs::exists(ortho_file)) {
      if(rank == 0) {
        cout << "Reading orthogonalizer from disk ..." << endl << endl;
        auto jX                    = ParserUtils::json_from_file(ortho_jfile);
        auto Xdims                 = jX["ortho_dims"].get<std::vector<int>>();
        chem_env.sys_data.n_lindep = chem_env.sys_data.nbf_orig - Xdims[1];
      }
      ec.pg().broadcast(&chem_env.sys_data.n_lindep, 0);
      chem_env.sys_data.nbf =
        chem_env.sys_data.nbf_orig - chem_env.sys_data.n_lindep; // Compute Northo

      scf_vars.tAO_ortho = TiledIndexSpace{IndexSpace{range(0, (size_t) (chem_env.sys_data.nbf))},
                                           chem_env.ioptions.scf_options.AO_tilesize};

#if defined(USE_SCALAPACK)
      {
        const tamm::Tile _mb =
          chem_env.ioptions.scf_options.scalapack_nb; //(scalapack_info.blockcyclic_dist)->mb();
        scf_vars.tN_bc      = TiledIndexSpace{IndexSpace{range(chem_env.sys_data.nbf_orig)}, _mb};
        scf_vars.tNortho_bc = TiledIndexSpace{IndexSpace{range(chem_env.sys_data.nbf)}, _mb};
        if(scacomm != MPI_COMM_NULL) {
          ttensors.X_alpha = {scf_vars.tN_bc, scf_vars.tNortho_bc};
          ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
          Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.X_alpha);
          scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                             chem_env.ioptions.scf_options.debug, true);
        }
      }
#else
    ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
    sch.allocate(ttensors.X_alpha).execute();
    scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                       chem_env.ioptions.scf_options.debug, true);
#endif
    }
    else {
      scf_compute.compute_orthogonalizer(ec, chem_env, scf_vars, scalapack_info, ttensors);
      // chem_env.sys_data.nbf = chem_env.sys_data.nbf_orig - chem_env.sys_data.n_lindep;
      if(rank == 0) {
        json jX;
        jX["ortho_dims"] = {chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf};
        ParserUtils::json_to_file(jX, ortho_jfile);
      }
      if(N >= chem_env.ioptions.scf_options.restart_size) {
#if defined(USE_SCALAPACK)
        if(scacomm != MPI_COMM_NULL)
          scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                             chem_env.ioptions.scf_options.debug);
#else
      scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                         chem_env.ioptions.scf_options.debug);
#endif
      }
    }

#if defined(USE_SCALAPACK)
    if(scacomm != MPI_COMM_NULL) {
      ttensors.F_BC = {scf_vars.tN_bc, scf_vars.tN_bc};
      ttensors.F_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      ttensors.C_alpha_BC = {scf_vars.tN_bc, scf_vars.tNortho_bc};
      ttensors.C_alpha_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.F_BC, ttensors.C_alpha_BC);
      if(chem_env.sys_data.is_unrestricted) {
        ttensors.C_beta_BC = {scf_vars.tN_bc, scf_vars.tNortho_bc};
        ttensors.C_beta_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
        Tensor<TensorType>::allocate(&scalapack_info.ec, ttensors.C_beta_BC);
      }
    }
#endif

    if(!do_density_fitting || scf_vars.direct_df || chem_env.sys_data.is_ks ||
       chem_env.sys_data.do_snK) {
      // needed for 4c HF, direct_df, KS, or snK
      etensors.D_alpha = Matrix::Zero(N, N);
      etensors.G_alpha = Matrix::Zero(N, N);
      if(chem_env.sys_data.is_unrestricted) {
        etensors.D_beta = Matrix::Zero(N, N);
        etensors.G_beta = Matrix::Zero(N, N);
      }
    }

    // pre-compute data for Schwarz bounds

    std::string schwarz_matfile = files_prefix + ".schwarz";
    Matrix      SchwarzK;

    if(!do_density_fitting || scf_vars.direct_df) {
      if(N >= chem_env.ioptions.scf_options.restart_size && fs::exists(schwarz_matfile)) {
        if(rank == 0) cout << "Read Schwarz matrix from disk ... " << endl;

        SchwarzK = scf_output.read_scf_mat<TensorType>(schwarz_matfile);
      }
      else {
        // if(rank == 0) cout << "pre-computing data for Schwarz bounds... " << endl;
        SchwarzK = scf_compute.compute_schwarz_ints<>(ec, scf_vars, chem_env.shells);
        if(rank == 0) scf_output.write_scf_mat<TensorType>(SchwarzK, schwarz_matfile);
      }
    }
    hf_t1 = std::chrono::high_resolution_clock::now();

    ttensors.ehf_tamm = Tensor<TensorType>{};
    ttensors.F_dummy  = {tAOt, tAOt}; // not allocated

    ttensors.ehf_tmp      = {tAO, tAO};
    ttensors.F_alpha      = {tAO, tAO};
    ttensors.D_alpha      = {tAO, tAO};
    ttensors.D_diff       = {tAO, tAO};
    ttensors.D_last_alpha = {tAO, tAO};
    ttensors.F_alpha_tmp  = {tAO, tAO};
    ttensors.FD_alpha     = {tAO, tAO};
    ttensors.FDS_alpha    = {tAO, tAO};

    ttensors.C_alpha  = {tAO, scf_vars.tAO_ortho};
    ttensors.C_occ_a  = {tAO, scf_vars.tAO_occ_a};
    ttensors.C_occ_aT = {scf_vars.tAO_occ_a, tAO};

    // TODO: Enable only for DFT
    ttensors.VXC_alpha = {tAO, tAO};
    ttensors.VXC_beta  = {tAO, tAO};

    if(chem_env.sys_data.is_unrestricted) {
      ttensors.C_beta       = {tAO, scf_vars.tAO_ortho};
      ttensors.C_occ_b      = {tAO, scf_vars.tAO_occ_b};
      ttensors.C_occ_bT     = {scf_vars.tAO_occ_b, tAO};
      ttensors.ehf_beta_tmp = {tAO, tAO};
      ttensors.F_beta       = {tAO, tAO};
      ttensors.D_beta       = {tAO, tAO};
      ttensors.D_last_beta  = {tAO, tAO};
      ttensors.F_beta_tmp   = {tAO, tAO};
      ttensors.FD_beta      = {tAO, tAO};
      ttensors.FDS_beta     = {tAO, tAO};
    }

    Tensor<TensorType>::allocate(&ec, ttensors.F_alpha, ttensors.C_alpha, ttensors.C_occ_a,
                                 ttensors.C_occ_aT, ttensors.D_alpha, ttensors.D_last_alpha,
                                 ttensors.D_diff, ttensors.F_alpha_tmp, ttensors.ehf_tmp,
                                 ttensors.ehf_tamm, ttensors.FD_alpha, ttensors.FDS_alpha);
    if(chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::allocate(&ec, ttensors.F_beta, ttensors.C_beta, ttensors.C_occ_b,
                                   ttensors.C_occ_bT, ttensors.D_beta, ttensors.D_last_beta,
                                   ttensors.F_beta_tmp, ttensors.ehf_beta_tmp, ttensors.FD_beta,
                                   ttensors.FDS_beta);

    if(chem_env.sys_data.is_ks) Tensor<TensorType>::allocate(&ec, ttensors.VXC_alpha);
    if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::allocate(&ec, ttensors.VXC_beta);

    // Setup tiled index spaces when a fitting basis is provided
    IndexSpace dfCocc{range(0, chem_env.sys_data.nelectrons_alpha)};
    scf_vars.tdfCocc             = {dfCocc, chem_env.ioptions.scf_options.dfAO_tilesize};
    std::tie(scf_vars.dCocc_til) = scf_vars.tdfCocc.labels<1>("all");
    if(do_density_fitting) {
      std::tie(scf_vars.d_mu, scf_vars.d_nu, scf_vars.d_ku)    = scf_vars.tdfAO.labels<3>("all");
      std::tie(scf_vars.d_mup, scf_vars.d_nup, scf_vars.d_kup) = scf_vars.tdfAOt.labels<3>("all");

      ttensors.xyZ = Tensor<TensorType>{tAO, tAO, scf_vars.tdfAO};       // n,n,ndf
      ttensors.xyK = Tensor<TensorType>{tAO, tAO, scf_vars.tdfAO};       // n,n,ndf
      ttensors.Vm1 = Tensor<TensorType>{scf_vars.tdfAO, scf_vars.tdfAO}; // ndf, ndf
      if(!scf_vars.direct_df) Tensor<TensorType>::allocate(&ec, ttensors.xyK);

      scf_iter.init_ri<TensorType>(ec, chem_env, scalapack_info, scf_vars, etensors, ttensors);
    }
    const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    const libint2::BasisSet& obs        = chem_env.shells;
    auto                     max_nprim  = obs.max_nprim();
    auto                     max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto                     shell2bf   = obs.shell2bf();
    if(chem_env.ioptions.scf_options.restart || chem_env.ioptions.scf_options.noscf) {
      // This was originally scf_restart.restart()
      scf_restart(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix);
      if(!do_density_fitting || scf_vars.direct_df || chem_env.sys_data.is_ks ||
         chem_env.sys_data.do_snK) {
        tamm_to_eigen_tensor(ttensors.D_alpha, etensors.D_alpha);
        if(chem_env.sys_data.is_unrestricted) {
          tamm_to_eigen_tensor(ttensors.D_beta, etensors.D_beta);
        }
      }
      ec.pg().barrier();
    }
    else if(ec_molden.molden_exists) {
      auto N      = chem_env.sys_data.nbf_orig;
      auto Northo = chem_env.sys_data.nbf;

      etensors.C_alpha.setZero(N, Northo);
      if(chem_env.sys_data.is_unrestricted) etensors.C_beta.setZero(N, Northo);

      if(rank == 0) {
        cout << endl << "Reading from molden file provided ..." << endl;
        if(ec_molden.molden_file_valid) {
          ec_molden.read_molden<TensorType>(chem_env, etensors.C_alpha, etensors.C_beta);
        }
      }

      scf_compute.compute_density<TensorType>(ec, chem_env, scf_vars, scalapack_info, ttensors,
                                              etensors);
      // X=C?

      ec.pg().barrier();
    }
    else {
      std::vector<int> s1vec, s2vec, ntask_vec;

      if(rank == 0) cout << "Superposition of Atomic Density Guess ..." << endl;

      scf_guess.compute_sad_guess<TensorType>(ec, chem_env, scf_vars, scalapack_info, etensors,
                                              ttensors);

      ec.pg().barrier();
    }

    hf_t2   = std::chrono::high_resolution_clock::now();
    hf_time = std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0 && !chem_env.ioptions.scf_options.noscf && !chem_env.ioptions.scf_options.restart)
      std::cout << std::fixed << std::setprecision(2)
                << "Total Time to compute initial guess: " << hf_time << " secs" << endl;

    /*** =========================== ***/
    /*** main iterative loop         ***/
    /*** =========================== ***/
    double rmsd   = 1.0;
    double ediff  = 0.0;
    bool   no_scf = chem_env.ioptions.scf_options.noscf;
    if(rank == 0 && chem_env.ioptions.scf_options.debug &&
       N < chem_env.ioptions.scf_options.restart_size) {
      Matrix S(chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf_orig);
      tamm_to_eigen_tensor(ttensors.S1, S);
      if(chem_env.sys_data.is_restricted)
        cout << "debug #electrons       = " << (int) std::round((etensors.D_alpha * S).trace())
             << endl;
      if(chem_env.sys_data.is_unrestricted) {
        cout << "debug #alpha electrons = " << (int) std::round((etensors.D_alpha * S).trace())
             << endl;
        cout << "debug #beta  electrons = " << (int) std::round((etensors.D_beta * S).trace())
             << endl;
      }
    }
    if(rank == 0 && !no_scf) {
      std::cout << std::endl << std::endl;
      std::cout << " SCF iterations" << endl;
      std::cout << std::string(77, '-') << endl;
      std::string sph =
        " Iter     Energy            E-Diff       RMSD        |[F,P]|^2       Time(s)";
      std::cout << sph << endl;
      std::cout << std::string(77, '-') << endl;
    }

    std::cout << std::fixed << std::setprecision(2);

    /*** Generate task mapping ***/

    if(!do_density_fitting && scf_vars.do_load_bal) {
      // Collect task info
      auto [s1vec, s2vec, ntask_vec] = scf_iter.compute_2bf_taskinfo<TensorType>(
        ec, chem_env, scf_vars, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4, ttensors,
        etensors, do_density_fitting);

      auto [s1_all, s2_all, ntasks_all] =
        gather_task_vectors<TensorType>(ec, s1vec, s2vec, ntask_vec);

      int tmdim = 0;
      if(rank == 0) {
        Loads dummyLoads;
        /***generate load balanced task map***/
        dummyLoads.readLoads(s1_all, s2_all, ntasks_all);
        dummyLoads.simpleLoadBal(ec.pg().size().value());
        tmdim = std::max(dummyLoads.maxS1, dummyLoads.maxS2);
        etensors.taskmap.resize(tmdim + 1, tmdim + 1);
        // value in this array is the rank that executes task i,j
        // -1 indicates a task i,j that can be skipped
        etensors.taskmap.setConstant(-1);
        // cout<<"creating task map"<<endl;
        dummyLoads.createTaskMap(etensors.taskmap);
        // cout<<"task map creation completed"<<endl;
      }
      ec.pg().broadcast(&tmdim, 0);
      if(rank != 0) etensors.taskmap.resize(tmdim + 1, tmdim + 1);
      ec.pg().broadcast(etensors.taskmap.data(), etensors.taskmap.size(), 0);
    }

    if(chem_env.ioptions.scf_options.noscf) {
      SystemData& sys_data = chem_env.sys_data;

      // clang-format off
      sch (ttensors.F_alpha_tmp() = 0)
          (ttensors.D_last_alpha(mu,nu) = ttensors.D_alpha(mu,nu))
          .execute();
      // clang-format on

      if(chem_env.sys_data.is_unrestricted) {
        // clang-format off
        sch (ttensors.F_beta_tmp() = 0)
            (ttensors.D_last_beta(mu,nu) = ttensors.D_beta(mu,nu))
            .execute();
        // clang-format on
      }

      // F_alpha = H1 + F_alpha_tmp
      scf_iter.compute_2bf<TensorType>(ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen,
                                       shell2bf, SchwarzK, max_nprim4, ttensors, etensors,
                                       is_3c_init, do_density_fitting, xHF);

      // Add QED contribution
      if(chem_env.sys_data.do_qed) {
        scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, ttensors);
      }

#if defined(USE_GAUXC)
      // Add snK contribution
      if(chem_env.sys_data.do_snK) {
        const auto snK_start = std::chrono::high_resolution_clock::now();
        scf::gauxc::compute_exx<TensorType>(ec, chem_env, scf_vars, ttensors, etensors,
                                            gauxc_integrator);
        const auto snK_stop = std::chrono::high_resolution_clock::now();
        const auto snK_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((snK_stop - snK_start)).count();
        auto debug = chem_env.ioptions.scf_options.debug;
        if(rank == 0 && debug)
          std::cout << std::fixed << std::setprecision(2) << "snK: " << snK_time << "s, ";
      }
#endif

      if(sys_data.is_restricted) {
        // clang-format off
        sch
          (ttensors.ehf_tmp(mu,nu)  = ttensors.H1(mu,nu))
          (ttensors.ehf_tmp(mu,nu) += ttensors.F_alpha(mu,nu))
          (ttensors.ehf_tamm()      = 0.5 * ttensors.D_alpha() * ttensors.ehf_tmp())
          .execute();
        // clang-format on
      }

      if(sys_data.is_unrestricted) {
        // clang-format off
        sch
          (ttensors.ehf_tmp(mu,nu)  = ttensors.H1(mu,nu))
          (ttensors.ehf_tmp(mu,nu) += ttensors.F_alpha(mu,nu))
          (ttensors.ehf_tamm()      = 0.5 * ttensors.D_alpha() * ttensors.ehf_tmp())
          (ttensors.ehf_tmp(mu,nu)  = ttensors.H1(mu,nu))
          (ttensors.ehf_tmp(mu,nu) += ttensors.F_beta(mu,nu))
          (ttensors.ehf_tamm()     += 0.5 * ttensors.D_beta()  * ttensors.ehf_tmp())
          .execute();
        // clang-format on
      }

      ehf = get_scalar(ttensors.ehf_tamm);

#if defined(USE_GAUXC)
      const bool is_ks     = sys_data.is_ks;
      double     gauxc_exc = 0;
      if(is_ks) {
        const auto xcf_start = std::chrono::high_resolution_clock::now();
        gauxc_exc =
          scf::gauxc::compute_xcf<TensorType>(ec, chem_env, ttensors, etensors, gauxc_integrator);

        const auto xcf_stop = std::chrono::high_resolution_clock::now();
        const auto xcf_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((xcf_stop - xcf_start)).count();
        auto debug = chem_env.ioptions.scf_options.debug;
        if(rank == 0 && debug)
          std::cout << std::fixed << std::setprecision(2) << "xcf: " << xcf_time << "s, ";
        if(sys_data.is_qed && !sys_data.do_qed) { scf_vars.eqed = gauxc_exc; }
      }

      ehf += gauxc_exc;
      scf_vars.exc = gauxc_exc;

      if(is_ks) {
        sch(ttensors.F_alpha() += ttensors.VXC_alpha());
        if(sys_data.is_unrestricted) {
          // clang-format off
          sch
            (ttensors.F_alpha() += ttensors.VXC_beta())
            (ttensors.F_beta()  += ttensors.VXC_alpha())
            (ttensors.F_beta()  += -1.0 * ttensors.VXC_beta());
          // clang-format on
        }
        sch.execute();
      }
#endif

      ehf += enuc;
      if(rank == 0)
        std::cout << std::setprecision(18) << "Total HF energy after restart: " << ehf << std::endl;
    }

    // SCF main loop
    TensorType   ediis;
    size_t       nbumps     = 0;
    const size_t nbumps_max = 3;
    double       conve      = chem_env.ioptions.scf_options.conve;
    double       convd      = chem_env.ioptions.scf_options.convd;
    const bool   is_uhf     = chem_env.sys_data.is_unrestricted;

    do {
      if(chem_env.ioptions.scf_options.noscf) break;

      const auto loop_start = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      double ehf_last = ehf;

      // clang-format off
      sch (ttensors.F_alpha_tmp() = 0)
          (ttensors.D_last_alpha(mu,nu) = ttensors.D_alpha(mu,nu))
          .execute();
      // clang-format on

      if(chem_env.sys_data.is_unrestricted) {
        // clang-format off
        sch (ttensors.F_beta_tmp() = 0)
            (ttensors.D_last_beta(mu,nu) = ttensors.D_beta(mu,nu))
            .execute();
        // clang-format on
      }

      // auto D_tamm_nrm = norm(ttensors.D_alpha);
      // if(rank==0) cout << std::setprecision(18) << "norm of D_tamm: " << D_tamm_nrm << endl;

      // build a new Fock matrix
      scf_iter.compute_2bf<TensorType>(ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen,
                                       shell2bf, SchwarzK, max_nprim4, ttensors, etensors,
                                       is_3c_init, do_density_fitting, xHF);

      // Add QED contribution
      if(chem_env.sys_data.do_qed) {
        scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, ttensors);
      }

      std::tie(ehf, rmsd) = scf_iter.scf_iter_body<TensorType>(ec, chem_env, scalapack_info, iter,
                                                               scf_vars, ttensors, etensors
#if defined(USE_GAUXC)
                                                               ,
                                                               gauxc_integrator
#endif
      );

      // DIIS error
      size_t ndiis = ttensors.diis_hist.size();
      ediis        = pow(tamm::norm(ttensors.diis_hist[ndiis - 1]), 2);
      if(is_uhf) ediis += pow(tamm::norm(ttensors.diis_beta_hist[ndiis - 1]), 2);

      ehf += enuc;
      // compute difference with last iteration
      ediff = ehf - ehf_last;

      if(ediff > 0.0) nbumps += 1;

      if(nbumps > nbumps_max && ndiis >= (size_t) chem_env.ioptions.scf_options.diis_hist) {
        nbumps         = 0;
        scf_vars.idiis = 0;
        if(rank == 0) std::cout << "Resetting DIIS" << std::endl;
        for(auto x: ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
        for(auto x: ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
        ttensors.diis_hist.clear();
        ttensors.fock_hist.clear();
        if(is_uhf) {
          for(auto x: ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
          for(auto x: ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
          ttensors.diis_beta_hist.clear();
          ttensors.fock_beta_hist.clear();
        }
      };

      const auto loop_stop = std::chrono::high_resolution_clock::now();
      const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

      if(rank == 0) {
        std::cout << std::setw(4) << iter << "  " << std::setw(10);
        if(chem_env.ioptions.scf_options.debug) {
          std::cout << std::fixed << std::setprecision(18) << ehf;
          std::cout << std::scientific << std::setprecision(18);
        }
        else {
          std::cout << std::fixed << std::setprecision(10) << ehf;
          std::cout << std::scientific << std::setprecision(2);
        }
        std::cout << ' ' << std::scientific << std::setw(12) << ediff;
        std::cout << ' ' << std::setw(12) << rmsd;
        std::cout << ' ' << std::setw(12) << ediis << ' ';
        std::cout << ' ' << std::setw(10) << std::fixed << std::setprecision(1) << loop_time << ' '
                  << endl;

        chem_env.sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)] = {
          {"energy", ehf}, {"e_diff", ediff}, {"rmsd", rmsd}, {"ediis", ediis}};
        chem_env.sys_data.results["output"]["SCF"]["iter"][std::to_string(iter)]["performance"] = {
          {"total_time", loop_time}};
      }

      // if(rank==0) cout << "D at the end of iteration: " << endl << std::setprecision(6) <<
      // etensors.D_alpha << endl;
      if(iter % chem_env.ioptions.scf_options.writem == 0 ||
         chem_env.ioptions.scf_options.writem == 1) {
        scf_output.rw_md_disk(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix);
      }

      if(iter >= chem_env.ioptions.scf_options.maxiter) {
        is_conv = false;
        break;
      }

      if(chem_env.ioptions.scf_options.debug)
        scf_output.print_energies(ec, chem_env, ttensors, etensors, scf_vars, scalapack_info);

      // Reset lshift to input option.
      if(fabs(ediff) > 1e-2) scf_vars.lshift = chem_env.ioptions.scf_options.lshift;

    } while((fabs(ediff) > conve) || (fabs(rmsd) > convd) ||
            (fabs(ediis) > 10.0 * conve)); // SCF main loop

    if(rank == 0) {
      std::cout.precision(13);
      if(is_conv) cout << endl << "** Total SCF energy = " << ehf << endl;
      else {
        cout << endl << std::string(50, '*') << endl;
        cout << std::string(10, ' ') << "ERROR: SCF calculation does not converge!!!" << endl;
        cout << std::string(50, '*') << endl;
      }
    }

    auto dplot_opt = chem_env.ioptions.dplot_options;
    if(dplot_opt.cube) {
      // if(dplot_opt.density == "spin") // TODO
      // else plot total density by default when cube=true
      /* else */ EC_DPLOT::write_dencube(ec, chem_env, etensors.D_alpha, etensors.D_beta,
                                         files_prefix);
#if defined(USE_SCALAPACK)
      if(scalapack_info.comm != MPI_COMM_NULL) {
        tamm::from_block_cyclic_tensor(ttensors.C_alpha_BC, ttensors.C_alpha);
        if(is_uhf) tamm::from_block_cyclic_tensor(ttensors.C_beta_BC, ttensors.C_beta);
      }
      tamm_to_eigen_tensor(ttensors.C_alpha, etensors.C_alpha);
      if(is_uhf) tamm_to_eigen_tensor(ttensors.C_beta, etensors.C_beta);
#else
    if(rank != 0) {
           etensors.C_alpha.resize(chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf);
           if(is_uhf) { etensors.C_beta.resize(chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf); }
    }
    ec.pg().broadcast(etensors.C_alpha.data(), etensors.C_alpha.size(), 0);
    if(is_uhf) { ec.pg().broadcast(etensors.C_beta.data(), etensors.C_beta.size(), 0); }
#endif
      if(dplot_opt.orbitals > 0) {
        for(int iorb = chem_env.sys_data.nelectrons_alpha - 1;
            iorb >= std::max(0, chem_env.sys_data.nelectrons_alpha - dplot_opt.orbitals); iorb--) {
          EC_DPLOT::write_mocube(ec, chem_env, etensors.C_alpha, iorb, "alpha", files_prefix);
        }
        for(int iorb = chem_env.sys_data.nelectrons_alpha;
            iorb < std::min(chem_env.sys_data.nbf,
                            chem_env.sys_data.nelectrons_alpha + dplot_opt.orbitals);
            iorb++) {
          EC_DPLOT::write_mocube(ec, chem_env, etensors.C_alpha, iorb, "alpha", files_prefix);
        }
        if(is_uhf) {
          for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
              iorb >= std::max(0, chem_env.sys_data.nelectrons_beta - dplot_opt.orbitals); iorb--) {
            EC_DPLOT::write_mocube(ec, chem_env, etensors.C_beta, iorb, "beta", files_prefix);
          }
          for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
              iorb < std::min(chem_env.sys_data.nbf,
                              chem_env.sys_data.nelectrons_beta + dplot_opt.orbitals);
              iorb++) {
            EC_DPLOT::write_mocube(ec, chem_env, etensors.C_beta, iorb, "beta", files_prefix);
          }
        }
      }
    }

    if(chem_env.sys_data.is_ks) { // or rohf
      sch(ttensors.F_alpha_tmp() = 0).execute();
      if(chem_env.sys_data.is_unrestricted) sch(ttensors.F_beta_tmp() = 0).execute();

      auto xHF_adjust = xHF;
      // TODO: skip for non-CC methods
      if(!chem_env.ioptions.task_options.scf) xHF_adjust = 1.0;
      // build a new Fock matrix
      scf_iter.compute_2bf<TensorType>(ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen,
                                       shell2bf, SchwarzK, max_nprim4, ttensors, etensors,
                                       is_3c_init, do_density_fitting, xHF_adjust);

      // Add QED contribution;
      // CHECK

      if(chem_env.sys_data.do_qed) {
        scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, ttensors);
      }
    }
    else if(scf_vars.lshift > 0) {
      // Remove level shift from Fock matrix
      double lval = chem_env.sys_data.is_restricted ? 0.5 * scf_vars.lshift : scf_vars.lshift;
      // clang-format off
      sch
      (ttensors.ehf_tmp(mu,ku) = ttensors.S1(mu,nu) * ttensors.D_last_alpha(nu,ku))
      (ttensors.F_alpha(mu,ku) += lval * ttensors.ehf_tmp(mu,nu) * ttensors.S1(nu,ku))
      .execute();
      // clang-format on

      if(is_uhf) {
        // clang-format off
        sch
        (ttensors.ehf_tmp(mu,ku) = ttensors.S1(mu,nu) * ttensors.D_last_beta(nu,ku))
        (ttensors.F_beta(mu,ku) += lval * ttensors.ehf_tmp(mu,nu) * ttensors.S1(nu,ku))
        .execute();
        // clang-format on
      }
    }

    for(auto x: ttensors.ehf_tamm_hist) Tensor<TensorType>::deallocate(x);

    for(auto x: ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.D_hist) Tensor<TensorType>::deallocate(x);

    if(chem_env.sys_data.is_unrestricted) {
      for(auto x: ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
      for(auto x: ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
      for(auto x: ttensors.D_beta_hist) Tensor<TensorType>::deallocate(x);
    }
    if(rank == 0)
      std::cout << std::endl
                << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;
    scf_output.print_energies(ec, chem_env, ttensors, etensors, scf_vars, scalapack_info);

    if(!chem_env.ioptions.scf_options.noscf) {
      if(rank == 0) cout << "writing orbitals and density to disk ... ";
      scf_output.rw_md_disk(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix);
      if(rank == 0) cout << "done." << endl;
    }

    if(rank == 0 && chem_env.ioptions.scf_options.mulliken_analysis && is_conv) {
      Matrix S = tamm_to_eigen_matrix(ttensors.S1);
      scf_output.print_mulliken(chem_env, etensors.D_alpha, etensors.D_beta, S);
    }
    // copy to fock matrices allocated on world group
    sch(Fa_global(mu, nu) = ttensors.F_alpha(mu, nu));
    if(chem_env.sys_data.is_unrestricted) sch(Fb_global(mu, nu) = ttensors.F_beta(mu, nu));
    sch.execute();

    if(do_density_fitting) {
      if(scf_vars.direct_df) { Tensor<TensorType>::deallocate(ttensors.Vm1); }
      else { Tensor<TensorType>::deallocate(ttensors.xyK); }
    }

    scf_output.rw_mat_disk<TensorType>(ttensors.H1, files_prefix + ".hcore",
                                       chem_env.ioptions.scf_options.debug);

    Tensor<TensorType>::deallocate(ttensors.H1, ttensors.S1, ttensors.T1, ttensors.V1,
                                   ttensors.F_alpha_tmp, ttensors.ehf_tmp, ttensors.ehf_tamm,
                                   ttensors.F_alpha, ttensors.C_alpha, ttensors.C_occ_a,
                                   ttensors.C_occ_aT, ttensors.D_alpha, ttensors.D_diff,
                                   ttensors.D_last_alpha, ttensors.FD_alpha, ttensors.FDS_alpha);

    if(chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::deallocate(ttensors.F_beta, ttensors.C_beta, ttensors.C_occ_b,
                                     ttensors.C_occ_bT, ttensors.D_beta, ttensors.D_last_beta,
                                     ttensors.F_beta_tmp, ttensors.ehf_beta_tmp, ttensors.FD_beta,
                                     ttensors.FDS_beta);
    if(chem_env.sys_data.is_ks) {
      // write vxc to disk
      scf_output.rw_mat_disk<TensorType>(ttensors.VXC_alpha, files_prefix + ".vxc_alpha",
                                         chem_env.ioptions.scf_options.debug);
      if(chem_env.sys_data.is_unrestricted)
        scf_output.rw_mat_disk<TensorType>(ttensors.VXC_beta, files_prefix + ".vxc_beta",
                                           chem_env.ioptions.scf_options.debug);
      Tensor<TensorType>::deallocate(ttensors.VXC_alpha);
      if(chem_env.sys_data.is_unrestricted) Tensor<TensorType>::deallocate(ttensors.VXC_beta);
    }
    if(chem_env.sys_data.is_qed) {
      scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dx, files_prefix + ".QED_Dx",
                                         chem_env.ioptions.scf_options.debug);
      scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dy, files_prefix + ".QED_Dy",
                                         chem_env.ioptions.scf_options.debug);
      scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dz, files_prefix + ".QED_Dz",
                                         chem_env.ioptions.scf_options.debug);
      scf_output.rw_mat_disk<TensorType>(ttensors.QED_Qxx, files_prefix + ".QED_Qxx",
                                         chem_env.ioptions.scf_options.debug);
      Tensor<TensorType>::deallocate(ttensors.QED_Dx, ttensors.QED_Dy, ttensors.QED_Dz,
                                     ttensors.QED_Qxx, ttensors.QED_Qxy, ttensors.QED_Qxz,
                                     ttensors.QED_Qyy, ttensors.QED_Qyz, ttensors.QED_Qzz,
                                     ttensors.QED_1body, ttensors.QED_2body);
    }
#if SCF_THROTTLE_RESOURCES
    ec.flush_and_sync();
#endif

#if defined(USE_SCALAPACK)
#if defined(USE_UPCXX)
    abort(); // Not supported currently in UPC++
#endif
    if(scalapack_info.comm != MPI_COMM_NULL) {
      Tensor<TensorType>::deallocate(ttensors.F_BC, ttensors.X_alpha, ttensors.C_alpha_BC);
      if(chem_env.sys_data.is_unrestricted) Tensor<TensorType>::deallocate(ttensors.C_beta_BC);
      scalapack_info.ec.flush_and_sync();
      MPI_Comm_free(&scacomm); // frees scalapack_info.comm
      scalapack_info.ec.pg().destroy_coll();
    }
// Free created comms / groups
// MPI_Comm_free( &scalapack_comm );
// MPI_Group_free( &scalapack_group );
// MPI_Group_free( &world_group );
#else
  sch.deallocate(ttensors.X_alpha);
  sch.execute();
#endif

#if SCF_THROTTLE_RESOURCES
    ec.pg().destroy_coll();
  } // end scaled down process group

#if defined(USE_UPCXX)
  hf_comm->destroy();
#else
  if(hf_comm != MPI_COMM_NULL) MPI_Comm_free(&hf_comm);
#endif

#endif

  // C,F1 is not allocated for ranks > hf_nranks
  exc.pg().barrier();
  exc.pg().broadcast(&is_conv, 0);

  if(!is_conv) { tamm_terminate("Please check SCF input parameters"); }

  // F, C are not deallocated.
  chem_env.sys_data.n_occ_alpha = chem_env.sys_data.nelectrons_alpha;
  chem_env.sys_data.n_occ_beta  = chem_env.sys_data.nelectrons_beta;
  chem_env.sys_data.n_vir_alpha =
    chem_env.sys_data.nbf_orig - chem_env.sys_data.n_occ_alpha - chem_env.sys_data.n_lindep;
  chem_env.sys_data.n_vir_beta =
    chem_env.sys_data.nbf_orig - chem_env.sys_data.n_occ_beta - chem_env.sys_data.n_lindep;

  exc.pg().broadcast(&ehf, 0);
  exc.pg().broadcast(&chem_env.sys_data.nbf, 0);
  exc.pg().broadcast(&chem_env.sys_data.n_lindep, 0);
  exc.pg().broadcast(&chem_env.sys_data.n_occ_alpha, 0);
  exc.pg().broadcast(&chem_env.sys_data.n_vir_alpha, 0);
  exc.pg().broadcast(&chem_env.sys_data.n_occ_beta, 0);
  exc.pg().broadcast(&chem_env.sys_data.n_vir_beta, 0);

  chem_env.sys_data.results["output"]["system_info"]["nbf_orig"] = chem_env.sys_data.nbf_orig;

  chem_env.sys_data.update();
  if(rank == 0 && chem_env.ioptions.scf_options.debug) chem_env.sys_data.print();

  // iter not broadcasted, but fine since only rank 0 writes to json
  if(rank == 0) {
    chem_env.sys_data.results["output"]["SCF"]["final_energy"] = ehf;
    chem_env.sys_data.results["output"]["SCF"]["n_iterations"] = iter;
  }

  IndexSpace AO_ortho{range(0, (size_t) (chem_env.sys_data.nbf_orig - chem_env.sys_data.n_lindep))};
  TiledIndexSpace    tAO_ortho{AO_ortho, chem_env.ioptions.scf_options.AO_tilesize};
  Tensor<TensorType> C_alpha_tamm{scf_vars.tAO, tAO_ortho};
  Tensor<TensorType> C_beta_tamm{scf_vars.tAO, tAO_ortho};
  ttensors.VXC_alpha = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};
  if(chem_env.sys_data.is_unrestricted)
    ttensors.VXC_beta = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};

  schg.allocate(C_alpha_tamm);
  if(chem_env.sys_data.is_unrestricted) schg.allocate(C_beta_tamm);
  if(chem_env.sys_data.is_ks) schg.allocate(ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted) schg.allocate(ttensors.VXC_beta);
  schg.execute();

  std::string movecsfile_alpha = files_prefix + ".alpha.movecs";
  std::string movecsfile_beta  = files_prefix + ".beta.movecs";

  scf_output.rw_mat_disk<TensorType>(C_alpha_tamm, movecsfile_alpha,
                                     chem_env.ioptions.scf_options.debug, true);
  if(chem_env.sys_data.is_unrestricted)
    scf_output.rw_mat_disk<TensorType>(C_beta_tamm, movecsfile_beta,
                                       chem_env.ioptions.scf_options.debug, true);

  if(rank == 0 && chem_env.ioptions.scf_options.molden) {
    Matrix C_a = tamm_to_eigen_matrix(C_alpha_tamm);
    if(chem_env.sys_data.is_unrestricted)
      std::cout << "[MOLDEN] molden write for UHF unsupported!" << std::endl;
    else ec_molden.write_molden(chem_env, C_a, etensors.eps_a, files_prefix);
  }

  if(chem_env.sys_data.is_ks) schg.deallocate(ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted)
    schg.deallocate(ttensors.VXC_beta);
  if(chem_env.sys_data.is_ks) schg.execute();

  exc.pg().barrier();

  chem_env.update(ehf, chem_env.shells, scf_vars.shell_tile_map, C_alpha_tamm, Fa_global,
                  C_beta_tamm, Fb_global, scf_vars.tAO, scf_vars.tAOt, tAO_ortho,
                  chem_env.ioptions.scf_options.noscf);
} // END of scf_hf(ExecutionContext& exc, ChemEnv& chem_env)
