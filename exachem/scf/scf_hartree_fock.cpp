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
  initialize_variables(exc, chem_env);
  scf_hf(exc, chem_env);
}

void exachem::scf::SCFHartreeFock::initialize_variables(ExecutionContext& exc, ChemEnv& chem_env) {
  is_spherical               = (chem_env.ioptions.scf_options.gaussian_type == "spherical");
  chem_env.sys_data.nbf      = chem_env.shells.nbf();
  chem_env.sys_data.nbf_orig = chem_env.sys_data.nbf;
  chem_env.sys_data.results["output"]["system_info"]["nbf"] = chem_env.sys_data.nbf;

  /* ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Related File Names ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ */
  out_fp       = chem_env.workspace_dir;
  files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type + "/scf";
  files_prefix = files_dir + "/" + chem_env.sys_data.output_file_prefix;

  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  ortho_file  = files_prefix + ".orthogonalizer";
  ortho_jfile = ortho_file + ".json";

  schwarz_matfile = files_prefix + ".schwarz";

  movecsfile_alpha = files_prefix + ".alpha.movecs";
  movecsfile_beta  = files_prefix + ".beta.movecs";

  hcore_file = files_prefix + ".hcore";

  vcx_alpha_file = files_prefix + ".vxc_alpha";
  vcx_beta_file  = files_prefix + ".vxc_beta";

  qed_dx_file  = files_prefix + ".QED_Dx";
  qed_dy_file  = files_prefix + ".QED_Dy";
  qed_dz_file  = files_prefix + ".QED_Dz";
  qed_qxx_file = files_prefix + ".QED_Qxx";

  /* ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Related File Names ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ */
  const int N = chem_env.shells.nbf();

  // SCFVars scf_vars; // init vars
  scf_vars.lshift = chem_env.ioptions.scf_options.lshift;

  if(!chem_env.ioptions.scf_options.dfbasis.empty()) do_density_fitting = true;
  scf_vars.do_dens_fit       = do_density_fitting;
  chem_env.scf_context.do_df = do_density_fitting;

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

  // SCF main loop
  nbumps = 0;
  conve  = chem_env.ioptions.scf_options.conve;
  convd  = chem_env.ioptions.scf_options.convd;

  reset_tolerences(exc, chem_env);

} // initialize_variables

void exachem::scf::SCFHartreeFock::reset_tolerences(ExecutionContext& exc, ChemEnv& chem_env) {
  auto         rank = exc.pg().rank();
  const double fock_precision =
    std::min(chem_env.ioptions.scf_options.tol_sch, 1e-2 * chem_env.ioptions.scf_options.conve);

  if(fock_precision < chem_env.ioptions.scf_options.tol_sch) {
    chem_env.ioptions.scf_options.tol_sch = fock_precision;
    if(rank == 0) std::cout << "Resetting tol_sch to " << fock_precision << std::endl;
  }

#if defined(USE_GAUXC)
  if(chem_env.ioptions.scf_options.snK) {
    if(chem_env.ioptions.scf_options.xc_snK_etol > chem_env.ioptions.scf_options.conve) {
      chem_env.ioptions.scf_options.xc_snK_etol = chem_env.ioptions.scf_options.conve;
      if(rank == 0)
        std::cout << "Resetting xc_snK_etol to " << chem_env.ioptions.scf_options.conve
                  << std::endl;
    }
    if(chem_env.ioptions.scf_options.xc_snK_ktol > fock_precision) {
      chem_env.ioptions.scf_options.xc_snK_ktol = fock_precision;
      if(rank == 0) std::cout << "Resetting xc_snK_ktol to " << fock_precision << std::endl;
    }
  }
  if(chem_env.ioptions.scf_options.xc_basis_tol > chem_env.ioptions.scf_options.conve) {
    chem_env.ioptions.scf_options.xc_basis_tol = fock_precision;
    if(rank == 0)
      std::cout << "Resetting xc_basis_tol to " << chem_env.ioptions.scf_options.conve << std::endl;
  }
#endif
} // reset_tolerences

void exachem::scf::SCFHartreeFock::write_dplot_data(ExecutionContext& ec, ChemEnv& chem_env) {
  auto       dplot_opt = chem_env.ioptions.dplot_options;
  const bool is_uhf    = chem_env.sys_data.is_unrestricted;
  auto       rank      = ec.pg().rank();

  // if(dplot_opt.density == "spin") // TODO
  // else plot total density by default when cube=true
  /* else */ EC_DPLOT::write_dencube(ec, chem_env, etensors.D_alpha, etensors.D_beta, files_prefix);
#if defined(USE_SCALAPACK)
  if(scalapack_info.pg.is_valid()) {
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
        iorb <
        std::min(chem_env.sys_data.nbf, chem_env.sys_data.nelectrons_alpha + dplot_opt.orbitals);
        iorb++) {
      EC_DPLOT::write_mocube(ec, chem_env, etensors.C_alpha, iorb, "alpha", files_prefix);
    }
    if(is_uhf) {
      for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
          iorb >= std::max(0, chem_env.sys_data.nelectrons_beta - dplot_opt.orbitals); iorb--) {
        EC_DPLOT::write_mocube(ec, chem_env, etensors.C_beta, iorb, "beta", files_prefix);
      }
      for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
          iorb <
          std::min(chem_env.sys_data.nbf, chem_env.sys_data.nelectrons_beta + dplot_opt.orbitals);
          iorb++) {
        EC_DPLOT::write_mocube(ec, chem_env, etensors.C_beta, iorb, "beta", files_prefix);
      }
    }
  }

} // write_dplot_data

void exachem::scf::SCFHartreeFock::qed_tensors_1e(ExecutionContext& ec, ChemEnv& chem_env) {
  const TiledIndexSpace& tAO = scf_vars.tAO;
  ttensors.QED_Dx            = {tAO, tAO};
  ttensors.QED_Dy            = {tAO, tAO};
  ttensors.QED_Dz            = {tAO, tAO};
  ttensors.QED_Qxx           = {tAO, tAO};
  ttensors.QED_Qxy           = {tAO, tAO};
  ttensors.QED_Qxz           = {tAO, tAO};
  ttensors.QED_Qyy           = {tAO, tAO};
  ttensors.QED_Qyz           = {tAO, tAO};
  ttensors.QED_Qzz           = {tAO, tAO};
  ttensors.QED_1body         = {tAO, tAO};
  ttensors.QED_2body         = {tAO, tAO};

  Tensor<TensorType>::allocate(&ec, ttensors.QED_Dx, ttensors.QED_Dy, ttensors.QED_Dz,
                               ttensors.QED_Qxx, ttensors.QED_Qxy, ttensors.QED_Qxz,
                               ttensors.QED_Qyy, ttensors.QED_Qyz, ttensors.QED_Qzz,
                               ttensors.QED_1body, ttensors.QED_2body);
  scf_qed.compute_qed_emult_ints<TensorType>(ec, chem_env, scf_vars, ttensors);
  if(chem_env.sys_data.do_qed)
    scf_qed.compute_QED_1body<TensorType>(ec, chem_env, scf_vars, ttensors);
} // end of initialize_qed_tensors

void exachem::scf::SCFHartreeFock::setup_libecpint(ExecutionContext& exc, ChemEnv& chem_env) {
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

} // setup_libECPint

void exachem::scf::SCFHartreeFock::scf_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env) {
  const int N    = chem_env.shells.nbf();
  auto      rank = ec.pg().rank();
  Scheduler sch{ec};
  if(N >= chem_env.ioptions.scf_options.restart_size && fs::exists(ortho_file)) {
    if(rank == 0) {
      std::cout << "Reading orthogonalizer from disk ..." << std::endl << std::endl;
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
      if(scalapack_info.pg.is_valid()) {
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

    if(rank == 0) {
      json jX;
      jX["ortho_dims"] = {chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf};
      ParserUtils::json_to_file(jX, ortho_jfile);
    }

    if(N >= chem_env.ioptions.scf_options.restart_size) {
#if defined(USE_SCALAPACK)
      if(scalapack_info.pg.is_valid())
        scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                           chem_env.ioptions.scf_options.debug);
#else
      scf_output.rw_mat_disk<TensorType>(ttensors.X_alpha, ortho_file,
                                         chem_env.ioptions.scf_options.debug);
#endif
    }
  }

#if defined(USE_SCALAPACK)
  if(scalapack_info.pg.is_valid()) {
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

} // scf_orthogonalizer

void exachem::scf::SCFHartreeFock::declare_main_tensors(ExecutionContext& ec, ChemEnv& chem_env) {
  const TiledIndexSpace& tAO  = scf_vars.tAO;
  const TiledIndexSpace& tAOt = scf_vars.tAOt;

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

  if(do_density_fitting) {
    ttensors.xyZ = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO, scf_vars.tdfAO}; // n,n,ndf
    ttensors.xyK = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO, scf_vars.tdfAO}; // n,n,ndf
    ttensors.Vm1 = Tensor<TensorType>{scf_vars.tdfAO, scf_vars.tdfAO};             // ndf, ndf
    if(!scf_vars.direct_df) Tensor<TensorType>::allocate(&ec, ttensors.xyK);
  }

  // Setup tiled index spaces when a fitting basis is provided
  dfCocc                       = {range(0, chem_env.sys_data.nelectrons_alpha)};
  scf_vars.tdfCocc             = {dfCocc, chem_env.ioptions.scf_options.dfAO_tilesize};
  std::tie(scf_vars.dCocc_til) = scf_vars.tdfCocc.labels<1>("all");

} // declare main tensors

void exachem::scf::SCFHartreeFock::deallocate_main_tensors(ExecutionContext& ec,
                                                           ChemEnv&          chem_env) {
  for(auto x: ttensors.ehf_tamm_hist) Tensor<TensorType>::deallocate(x);

  for(auto x: ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
  for(auto x: ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
  for(auto x: ttensors.D_hist) Tensor<TensorType>::deallocate(x);

  if(chem_env.sys_data.is_unrestricted) {
    for(auto x: ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: ttensors.D_beta_hist) Tensor<TensorType>::deallocate(x);
  }
  if(do_density_fitting) {
    if(scf_vars.direct_df) { Tensor<TensorType>::deallocate(ttensors.Vm1); }
    else { Tensor<TensorType>::deallocate(ttensors.xyK); }
  }
  if(chem_env.sys_data.is_ks) {
    Tensor<TensorType>::deallocate(ttensors.VXC_alpha);
    if(chem_env.sys_data.is_unrestricted) Tensor<TensorType>::deallocate(ttensors.VXC_beta);
  }
  if(chem_env.sys_data.is_qed) {
    Tensor<TensorType>::deallocate(ttensors.QED_Dx, ttensors.QED_Dy, ttensors.QED_Dz,
                                   ttensors.QED_Qxx, ttensors.QED_Qxy, ttensors.QED_Qxz,
                                   ttensors.QED_Qyy, ttensors.QED_Qyz, ttensors.QED_Qzz,
                                   ttensors.QED_1body, ttensors.QED_2body);
  }

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

} // deallocate_main_tensors

void exachem::scf::SCFHartreeFock::scf_final_io(ExecutionContext& ec, ChemEnv& chem_env) {
  auto rank = ec.pg().rank();

  if(!chem_env.ioptions.scf_options.noscf) {
    if(rank == 0) cout << "writing orbitals and density to disk ... ";
    scf_output.rw_md_disk(ec, chem_env, scalapack_info, ttensors, etensors, files_prefix);
    if(rank == 0) cout << "done." << endl;
  }

  scf_output.rw_mat_disk<TensorType>(ttensors.H1, hcore_file, chem_env.ioptions.scf_options.debug);
  if(chem_env.sys_data.is_ks) {
    // write vxc to disk
    scf_output.rw_mat_disk<TensorType>(ttensors.VXC_alpha, vcx_alpha_file,
                                       chem_env.ioptions.scf_options.debug);
    if(chem_env.sys_data.is_unrestricted)
      scf_output.rw_mat_disk<TensorType>(ttensors.VXC_beta, vcx_beta_file,
                                         chem_env.ioptions.scf_options.debug);
  }
  if(chem_env.sys_data.is_qed) {
    scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dx, qed_dx_file,
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dy, qed_dy_file,
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(ttensors.QED_Dz, qed_dz_file,
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(ttensors.QED_Qxx, qed_qxx_file,
                                       chem_env.ioptions.scf_options.debug);
  }
} // scf_final_io

void exachem::scf::SCFHartreeFock::setup_tiled_index_space(ExecutionContext& exc,
                                                           ChemEnv&          chem_env) {
  // Setup tiled index spaces
  // auto      rank = exc.pg().rank();
  const int N = chem_env.shells.nbf();
  AO          = {range(0, N)};
  scf_compute.recompute_tilesize(exc, chem_env);
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
} // setup_tiled_index_space

#if defined(USE_GAUXC)
GauXC::XCIntegrator<Matrix> exachem::scf::SCFHartreeFock::get_gauxc_integrator(ExecutionContext& ec,
                                                                               ChemEnv& chem_env) {
  auto rank = ec.pg().rank();
  if(chem_env.sys_data.is_ks || chem_env.sys_data.do_snK)
    std::tie(gauxc_integrator_ptr, xHF) = scf::gauxc::setup_gauxc(ec, chem_env, scf_vars);
  else xHF = 1.0;
  auto gauxc_integrator = (chem_env.sys_data.is_ks || chem_env.sys_data.do_snK)
                            ? GauXC::XCIntegrator<Matrix>(std::move(*gauxc_integrator_ptr))
                            : GauXC::XCIntegrator<Matrix>();
  scf_vars.xHF          = xHF;
  if(rank == 0) std::cout << "HF exch = " << xHF << std::endl;
  return gauxc_integrator;
} // get_gauxc_integrator

void exachem::scf::SCFHartreeFock::add_snk_contribution(
  ExecutionContext& ec, ChemEnv& chem_env, GauXC::XCIntegrator<Matrix>& gauxc_integrator) {
  auto rank = ec.pg().rank();
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
} // add_nk_contribution

void exachem::scf::SCFHartreeFock::compute_update_xc(
  ExecutionContext& ec, ChemEnv& chem_env, GauXC::XCIntegrator<Matrix>& gauxc_integrator) {
  Scheduler  sch{ec};
  auto       rank  = ec.pg().rank();
  const bool is_ks = chem_env.sys_data.is_ks;
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
    if(chem_env.sys_data.is_qed && !chem_env.sys_data.do_qed) { scf_vars.eqed = gauxc_exc; }
  }

  ehf += gauxc_exc;
  scf_vars.exc = gauxc_exc;

  if(is_ks) {
    sch(ttensors.F_alpha() += ttensors.VXC_alpha());
    if(chem_env.sys_data.is_unrestricted) {
      // clang-format off
      sch
        (ttensors.F_alpha() += ttensors.VXC_beta())
        (ttensors.F_beta()  += ttensors.VXC_alpha())
        (ttensors.F_beta()  += -1.0 * ttensors.VXC_beta());
      // clang-format on
    }
    sch.execute();
  }

} // compute_update_xc

#endif

void exachem::scf::SCFHartreeFock::process_molden_data(ExecutionContext& ec, ChemEnv& chem_env) {
  ec_molden.check_molden(chem_env.ioptions.scf_options.moldenfile);
  if(ec_molden.molden_file_valid) ec_molden.read_geom_molden(chem_env);
  if(ec_molden.molden_exists && ec_molden.molden_file_valid) {
    chem_env.shells = ec_molden.read_basis_molden(chem_env);
    chem_env.shells = ec_molden.renormalize_libint_shells(chem_env.shells);
    if(is_spherical) chem_env.shells.set_pure(true);
    else chem_env.shells.set_pure(false); // use cartesian gaussians
  }

} // process_molden_data

void exachem::scf::SCFHartreeFock::setup_density_fitting(ExecutionContext& exc, ChemEnv& chem_env) {
  auto rank = exc.pg().rank();
  if(do_density_fitting) {
    scf_vars.dfbs = libint2::BasisSet(chem_env.ioptions.scf_options.dfbasis, chem_env.atoms);
    if(is_spherical) scf_vars.dfbs.set_pure(true);
    else scf_vars.dfbs.set_pure(false); // use cartesian gaussians

    if(rank == 0) cout << "density-fitting basis set rank = " << scf_vars.dfbs.nbf() << endl;

    chem_env.sys_data.ndf = scf_vars.dfbs.nbf();
    scf_vars.dfAO         = IndexSpace{range(0, chem_env.sys_data.ndf)};
    scf_compute.recompute_tilesize(exc, chem_env, true);
    std::tie(scf_vars.df_shell_tile_map, scf_vars.dfAO_tiles, scf_vars.dfAO_opttiles) =
      scf_compute.compute_AO_tiles(exc, chem_env, scf_vars.dfbs, true);

    scf_vars.tdfAO  = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_opttiles};
    scf_vars.tdfAOt = TiledIndexSpace{scf_vars.dfAO, scf_vars.dfAO_tiles};
    chem_env.sys_data.results["output"]["system_info"]["ndf"] = chem_env.sys_data.ndf;
  }
  std::unique_ptr<DFFockEngine> dffockengine(
    do_density_fitting ? new DFFockEngine(chem_env.shells, scf_vars.dfbs) : nullptr);
  // End setup for fitting basis

} // setup_density_fitting

double exachem::scf::SCFHartreeFock::calculate_diis_error(bool is_uhf) {
  double lediis = pow(tamm::norm(ttensors.diis_hist[ndiis - 1]), 2);
  if(is_uhf) { lediis += pow(tamm::norm(ttensors.diis_beta_hist[ndiis - 1]), 2); }
  return lediis;
} // calculate_diis_energy

void exachem::scf::SCFHartreeFock::reset_fock_and_save_last_density(ExecutionContext& exc,
                                                                    ChemEnv&          chem_env) {
  // clang-format off
  Scheduler sch{exc};
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
} // reset_fock_density

void exachem::scf::SCFHartreeFock::handle_energy_bumps(ExecutionContext& exc, ChemEnv& chem_env) {
  const bool is_uhf = chem_env.sys_data.is_unrestricted;
  auto       rank   = exc.pg().rank();
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
  }
} // handle_energy_bumps()

void exachem::scf::SCFHartreeFock::print_write_iteration(ExecutionContext& exc, ChemEnv& chem_env,
                                                         double loop_time) {
  auto rank = exc.pg().rank();
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
  if(iter % chem_env.ioptions.scf_options.writem == 0 ||
     chem_env.ioptions.scf_options.writem == 1) {
    scf_output.rw_md_disk(exc, chem_env, scalapack_info, ttensors, etensors, files_prefix);
  }
  if(chem_env.ioptions.scf_options.debug)
    scf_output.print_energies(exc, chem_env, ttensors, etensors, scf_vars, scalapack_info);

} // print_energy_iteration

bool exachem::scf::SCFHartreeFock::check_convergence(ExecutionContext& exc, ChemEnv& chem_env) {
  if(iter >= chem_env.ioptions.scf_options.maxiter) {
    is_conv = false;
    return false;
  }
  if((fabs(ediff) > conve) || (fabs(rmsd) > convd) || (fabs(ediis) > 10.0 * conve)) return true;
  else return false;
} // check_convergence

void exachem::scf::SCFHartreeFock::compute_fock_matrix(ExecutionContext& ec, ChemEnv& chem_env,
                                                       bool is_uhf, const bool do_schwarz_screen,
                                                       const size_t&        max_nprim4,
                                                       std::vector<size_t>& shell2bf) {
  Scheduler sch{ec};

  if(chem_env.sys_data.is_ks) { // or rohf
    sch(ttensors.F_alpha_tmp() = 0).execute();
    if(chem_env.sys_data.is_unrestricted) sch(ttensors.F_beta_tmp() = 0).execute();

    auto xHF_adjust = xHF;
    // TODO: skip for non-CC methods
    if(!chem_env.ioptions.task_options.scf) xHF_adjust = 1.0;
    // build a new Fock matrix
    scf_iter.compute_2bf<TensorType>(ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen,
                                     shell2bf, SchwarzK, max_nprim4, ttensors, etensors, is_3c_init,
                                     do_density_fitting, xHF_adjust);

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

} // compute_fock_matrix

void exachem::scf::SCFHartreeFock::update_movecs(ExecutionContext& ec, ChemEnv& chem_env) {
  auto      rank = ec.pg().rank();
  Scheduler schg{ec};
  AO_ortho     = {range(0, (size_t) (chem_env.sys_data.nbf_orig - chem_env.sys_data.n_lindep))};
  tAO_ortho    = {AO_ortho, chem_env.ioptions.scf_options.AO_tilesize};
  C_alpha_tamm = {scf_vars.tAO, tAO_ortho};
  C_beta_tamm  = {scf_vars.tAO, tAO_ortho};
  ttensors.VXC_alpha = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};
  if(chem_env.sys_data.is_unrestricted)
    ttensors.VXC_beta = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};

  schg.allocate(C_alpha_tamm);
  if(chem_env.sys_data.is_unrestricted) schg.allocate(C_beta_tamm);
  if(chem_env.sys_data.is_ks) schg.allocate(ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted) schg.allocate(ttensors.VXC_beta);
  schg.execute();

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

  ec.pg().barrier();

} // update_wavefunctions

/* ↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓
 */

void exachem::scf::SCFHartreeFock::scf_hf(ExecutionContext& exc, ChemEnv& chem_env) {
  auto rank = exc.pg().rank();

  auto hf_t1 = std::chrono::high_resolution_clock::now();
  process_molden_data(exc, chem_env);

  const int N = chem_env.shells.nbf();

  pgdata = get_spg_data(exc, N, -1, 50, chem_env.ioptions.scf_options.nnodes);
  auto [hf_nnodes, ppn, hf_nranks] = pgdata.unpack();
  if(rank == 0) {
    std::cout << "\n Number of nodes, processes per node used for SCF calculation: " << hf_nnodes
              << ", " << ppn << std::endl;
  }

  // pg is valid only for first hf_nranks
  ProcGroup pg = ProcGroup::create_subgroup(exc.pg(), hf_nranks);

  if(rank == 0) {
    chem_env.ioptions.common_options.print();
    chem_env.ioptions.scf_options.print();
  }

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
  setup_tiled_index_space(exc, chem_env);

  mu = scf_vars.mu, nu = scf_vars.nu, ku = scf_vars.ku;
  mup = scf_vars.mup, nup = scf_vars.nup, kup = scf_vars.kup;

  Scheduler schg{exc};
  // Fock matrices allocated on world group
  Fa_global = {scf_vars.tAO, scf_vars.tAO};
  Fb_global = {scf_vars.tAO, scf_vars.tAO};
  schg.allocate(Fa_global);
  if(chem_env.sys_data.is_unrestricted) schg.allocate(Fb_global);
  schg.execute();

  setup_density_fitting(exc, chem_env);

  auto   hf_t2 = std::chrono::high_resolution_clock::now();
  double hf_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "Time for initial setup: " << hf_time << " secs" << endl;

  exc.pg().barrier();

  // const bool scf_vars.do_load_bal = scf_vars.do_scf_vars.do_load_bal;

  // This is originally scf_restart_test
  scf_restart(exc, chem_env, files_prefix);

  if(pg.is_valid()) {
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

#if defined(USE_SCALAPACK)
    setup_scalapack_info(ec, chem_env, scalapack_info, pgdata);
#endif

#if defined(USE_GAUXC)
    GauXC::XCIntegrator<Matrix> gauxc_integrator = get_gauxc_integrator(ec, chem_env);
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

    setup_libecpint(ec, chem_env);

    ec.pg().barrier();

    Scheduler sch{ec};

    const TiledIndexSpace& tAO = scf_vars.tAO;
    // const TiledIndexSpace& tAOt = scf_vars.tAOt;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/
    scf_compute.compute_hamiltonian<TensorType>(ec, scf_vars, chem_env, ttensors, etensors);
    if(chem_env.sys_data.is_qed) qed_tensors_1e(ec, chem_env);
    if(chem_env.sys_data.has_ecp) {
      Tensor<TensorType> ECP{tAO, tAO};
      Tensor<TensorType>::allocate(&ec, ECP);
      scf_guess.compute_ecp_ints(ec, scf_vars, ECP, libecp_shells, ecps);
      sch(ttensors.H1() += ECP()).deallocate(ECP).execute();
    }

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    scf_orthogonalizer(ec, chem_env);

    // pre-compute data for Schwarz bounds

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

    declare_main_tensors(ec, chem_env);

    if(do_density_fitting) {
      std::tie(scf_vars.d_mu, scf_vars.d_nu, scf_vars.d_ku)    = scf_vars.tdfAO.labels<3>("all");
      std::tie(scf_vars.d_mup, scf_vars.d_nup, scf_vars.d_kup) = scf_vars.tdfAOt.labels<3>("all");
      scf_iter.init_ri<TensorType>(ec, chem_env, scalapack_info, scf_vars, etensors, ttensors);
    }
    // const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;

    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)

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
    if(rank == 0 && !chem_env.ioptions.scf_options.noscf) {
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
    const bool          do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    size_t              max_nprim         = chem_env.shells.max_nprim();
    const size_t        max_nprim4        = max_nprim * max_nprim * max_nprim * max_nprim;
    std::vector<size_t> shell2bf          = chem_env.shells.shell2bf();

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
      add_snk_contribution(ec, chem_env, gauxc_integrator);
#endif

      if(chem_env.sys_data.is_restricted) {
        // clang-format off
        sch
          (ttensors.ehf_tmp(mu,nu)  = ttensors.H1(mu,nu))
          (ttensors.ehf_tmp(mu,nu) += ttensors.F_alpha(mu,nu))
          (ttensors.ehf_tamm()      = 0.5 * ttensors.D_alpha() * ttensors.ehf_tmp())
          .execute();
        // clang-format on
      }

      if(chem_env.sys_data.is_unrestricted) {
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
      compute_update_xc(ec, chem_env, gauxc_integrator);
#endif

      ehf += enuc;
      if(rank == 0)
        std::cout << std::setprecision(18) << "Total HF energy after restart: " << ehf << std::endl;
    }

    // SCF main loop

    const bool is_uhf                 = chem_env.sys_data.is_unrestricted;
    bool       scf_main_loop_continue = true;

    do {
      if(chem_env.ioptions.scf_options.noscf) break;

      const auto loop_start = std::chrono::high_resolution_clock::now();
      ++iter;

      // Save a copy of the energy and the density
      double ehf_last = ehf;
      // resetting the Fock matrix and saving the last density.
      reset_fock_and_save_last_density(ec, chem_env);

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
      ndiis = ttensors.diis_hist.size();
      ediis = calculate_diis_error(is_uhf);

      ehf += enuc;
      // compute difference with last iteration
      ediff = ehf - ehf_last;

      handle_energy_bumps(ec, chem_env);

      const auto loop_stop = std::chrono::high_resolution_clock::now();
      const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

      print_write_iteration(ec, chem_env, loop_time);

      // if(rank==0) cout << "D at the end of iteration: " << endl << std::setprecision(6) <<
      // etensors.D_alpha << endl;
      scf_main_loop_continue = check_convergence(ec, chem_env);
      if(!is_conv) break;

      // Reset lshift to input option.
      if(fabs(ediff) > 1e-2) scf_vars.lshift = chem_env.ioptions.scf_options.lshift;

    } while(scf_main_loop_continue); // SCF main loop

    if(rank == 0) {
      std::cout.precision(13);
      if(is_conv) cout << endl << "** Total SCF energy = " << ehf << endl;
      else {
        cout << endl << std::string(50, '*') << endl;
        cout << std::string(10, ' ') << "ERROR: SCF calculation does not converge!!!" << endl;
        cout << std::string(50, '*') << endl;
      }
    }
    if(chem_env.ioptions.dplot_options.cube) write_dplot_data(ec, chem_env);

    compute_fock_matrix(ec, chem_env, is_uhf, do_schwarz_screen, max_nprim4, shell2bf);

    sch(Fa_global(mu, nu) = ttensors.F_alpha(mu, nu));
    if(chem_env.sys_data.is_unrestricted) sch(Fb_global(mu, nu) = ttensors.F_beta(mu, nu));
    sch.execute();
    if(rank == 0)
      std::cout << std::endl
                << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;
    scf_output.print_energies(ec, chem_env, ttensors, etensors, scf_vars, scalapack_info);

    if(rank == 0 && chem_env.ioptions.scf_options.mulliken_analysis && is_conv) {
      Matrix S = tamm_to_eigen_matrix(ttensors.S1);
      scf_output.print_mulliken(chem_env, etensors.D_alpha, etensors.D_beta, S);
    }

    scf_final_io(ec, chem_env);
    deallocate_main_tensors(ec, chem_env);

    ec.flush_and_sync();

#if defined(USE_SCALAPACK)
    if(scalapack_info.pg.is_valid()) {
      Tensor<TensorType>::deallocate(ttensors.F_BC, ttensors.X_alpha, ttensors.C_alpha_BC);
      if(chem_env.sys_data.is_unrestricted) Tensor<TensorType>::deallocate(ttensors.C_beta_BC);
      scalapack_info.ec.flush_and_sync();
      scalapack_info.ec.pg().destroy_coll();
    }
#else
    sch.deallocate(ttensors.X_alpha);
    sch.execute();
#endif

    ec.pg().destroy_coll();
  } // end scf subgroup

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

  update_movecs(exc, chem_env);

  chem_env.is_context.AO_opt   = scf_vars.tAO;
  chem_env.is_context.AO_tis   = scf_vars.tAOt;
  chem_env.is_context.AO_ortho = tAO_ortho;

  // chem_env.scf_context.scf_converged = true;

  chem_env.scf_context.update(ehf, scf_vars.shell_tile_map, C_alpha_tamm, Fa_global, C_beta_tamm,
                              Fb_global, chem_env.ioptions.scf_options.noscf);
} // END of scf_hf(ExecutionContext& exc, ChemEnv& chem_env)
