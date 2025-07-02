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

exachem::scf::DefaultSCFEngine::DefaultSCFEngine(ExecutionContext& exc, ChemEnv& chem_env) {
  chem_env.sys_data.nbf                                     = chem_env.shells.nbf();
  chem_env.sys_data.nbf_orig                                = chem_env.sys_data.nbf;
  chem_env.sys_data.results["output"]["system_info"]["nbf"] = chem_env.sys_data.nbf;

  setup_file_paths(chem_env);

  initialize_scf_vars_and_tensors(chem_env);

  // SCF main loop

  reset_tolerences(exc, chem_env);

} // initialize_variables

void exachem::scf::DefaultSCFEngine::initialize_scf_vars_and_tensors(ChemEnv& chem_env) {
  const int N     = chem_env.shells.nbf();
  scf_vars.lshift = chem_env.ioptions.scf_options.lshift;
  if(!chem_env.ioptions.scf_options.dfbasis.empty()) scf_vars.do_dens_fit = true;
  chem_env.scf_context.do_df = scf_vars.do_dens_fit;

  if(!scf_vars.do_dens_fit || scf_vars.direct_df || chem_env.sys_data.is_ks ||
     chem_env.sys_data.do_snK) {
    // needed for 4c HF, direct_df, KS, or snK
    scf_vars.etensors.D_alpha = Matrix::Zero(N, N);
    scf_vars.etensors.G_alpha = Matrix::Zero(N, N);
    if(chem_env.sys_data.is_unrestricted) {
      scf_vars.etensors.D_beta = Matrix::Zero(N, N);
      scf_vars.etensors.G_beta = Matrix::Zero(N, N);
    }
  }
}

void exachem::scf::DefaultSCFEngine::setup_file_paths(const ChemEnv& chem_env) {
  std::string files_dir = chem_env.workspace_dir + chem_env.ioptions.scf_options.scf_type + "/scf";
  files_prefix          = files_dir + "/" + chem_env.sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  for(const auto& [tag, ext]: ext_type) { fname[tag] = files_prefix + ext; }
}

void exachem::scf::DefaultSCFEngine::reset_tolerences(ExecutionContext& exc, ChemEnv& chem_env) {
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
  if(chem_env.ioptions.scf_options.xc_basis_tol > chem_env.ioptions.scf_options.conve &&
     (chem_env.sys_data.is_ks || chem_env.sys_data.do_snK)) {
    chem_env.ioptions.scf_options.xc_basis_tol = chem_env.ioptions.scf_options.conve;
    if(rank == 0)
      std::cout << std::endl
                << "Resetting xc_basis_tol to " << chem_env.ioptions.scf_options.conve << std::endl;
  }
#endif
} // reset_tolerences

void exachem::scf::DefaultSCFEngine::write_dplot_data(ExecutionContext& ec, ChemEnv& chem_env) {
  auto       dplot_opt = chem_env.ioptions.dplot_options;
  const bool is_uhf    = chem_env.sys_data.is_unrestricted;
  auto       rank      = ec.pg().rank();

  // if(dplot_opt.density == "spin") // TODO
  // else plot total density by default when cube=true
  /* else */ EC_DPLOT::write_dencube(ec, chem_env, scf_vars.etensors.D_alpha,
                                     scf_vars.etensors.D_beta, files_prefix);
#if defined(USE_SCALAPACK)
  if(scalapack_info.pg.is_valid()) {
    tamm::from_block_cyclic_tensor(scf_vars.ttensors.C_alpha_BC, scf_vars.ttensors.C_alpha);
    if(is_uhf)
      tamm::from_block_cyclic_tensor(scf_vars.ttensors.C_beta_BC, scf_vars.ttensors.C_beta);
  }
  tamm_to_eigen_tensor(scf_vars.ttensors.C_alpha, scf_vars.etensors.C_alpha);
  if(is_uhf) tamm_to_eigen_tensor(scf_vars.ttensors.C_beta, scf_vars.etensors.C_beta);
#else
  if(rank != 0) {
    scf_vars.etensors.C_alpha.resize(chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf);
    if(is_uhf) {
      scf_vars.etensors.C_beta.resize(chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf);
    }
  }
  ec.pg().broadcast(scf_vars.etensors.C_alpha.data(), scf_vars.etensors.C_alpha.size(), 0);
  if(is_uhf) {
    ec.pg().broadcast(scf_vars.etensors.C_beta.data(), scf_vars.etensors.C_beta.size(), 0);
  }
#endif
  if(dplot_opt.orbitals > 0) {
    for(int iorb = chem_env.sys_data.nelectrons_alpha - 1;
        iorb >= std::max(0, chem_env.sys_data.nelectrons_alpha - dplot_opt.orbitals); iorb--) {
      EC_DPLOT::write_mocube(ec, chem_env, scf_vars.etensors.C_alpha, iorb, "alpha", files_prefix);
    }
    for(int iorb = chem_env.sys_data.nelectrons_alpha;
        iorb <
        std::min(chem_env.sys_data.nbf, chem_env.sys_data.nelectrons_alpha + dplot_opt.orbitals);
        iorb++) {
      EC_DPLOT::write_mocube(ec, chem_env, scf_vars.etensors.C_alpha, iorb, "alpha", files_prefix);
    }
    if(is_uhf) {
      for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
          iorb >= std::max(0, chem_env.sys_data.nelectrons_beta - dplot_opt.orbitals); iorb--) {
        EC_DPLOT::write_mocube(ec, chem_env, scf_vars.etensors.C_beta, iorb, "beta", files_prefix);
      }
      for(int iorb = chem_env.sys_data.nelectrons_beta - 1;
          iorb <
          std::min(chem_env.sys_data.nbf, chem_env.sys_data.nelectrons_beta + dplot_opt.orbitals);
          iorb++) {
        EC_DPLOT::write_mocube(ec, chem_env, scf_vars.etensors.C_beta, iorb, "beta", files_prefix);
      }
    }
  }

} // write_dplot_data

void exachem::scf::DefaultSCFEngine::qed_tensors_1e(ExecutionContext& ec, ChemEnv& chem_env) {
  const TiledIndexSpace& tAO  = scf_vars.tAO;
  scf_vars.ttensors.QED_Dx    = {tAO, tAO};
  scf_vars.ttensors.QED_Dy    = {tAO, tAO};
  scf_vars.ttensors.QED_Dz    = {tAO, tAO};
  scf_vars.ttensors.QED_Qxx   = {tAO, tAO};
  scf_vars.ttensors.QED_Qxy   = {tAO, tAO};
  scf_vars.ttensors.QED_Qxz   = {tAO, tAO};
  scf_vars.ttensors.QED_Qyy   = {tAO, tAO};
  scf_vars.ttensors.QED_Qyz   = {tAO, tAO};
  scf_vars.ttensors.QED_Qzz   = {tAO, tAO};
  scf_vars.ttensors.QED_1body = {tAO, tAO};
  scf_vars.ttensors.QED_2body = {tAO, tAO};

  Tensor<TensorType>::allocate(
    &ec, scf_vars.ttensors.QED_Dx, scf_vars.ttensors.QED_Dy, scf_vars.ttensors.QED_Dz,
    scf_vars.ttensors.QED_Qxx, scf_vars.ttensors.QED_Qxy, scf_vars.ttensors.QED_Qxz,
    scf_vars.ttensors.QED_Qyy, scf_vars.ttensors.QED_Qyz, scf_vars.ttensors.QED_Qzz,
    scf_vars.ttensors.QED_1body, scf_vars.ttensors.QED_2body);
  scf_qed.compute_qed_emult_ints<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors);
  if(chem_env.sys_data.do_qed)
    scf_qed.compute_QED_1body<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors);
} // end of initialize_qed_tensors

void exachem::scf::DefaultSCFEngine::setup_libecpint(
  ExecutionContext& exc, ChemEnv& chem_env, std::vector<libecpint::ECP>& ecps,
  std::vector<libecpint::GaussianShell>& libecp_shells) {
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

void exachem::scf::DefaultSCFEngine::scf_orthogonalizer(ExecutionContext& ec, ChemEnv& chem_env) {
  const int N    = chem_env.shells.nbf();
  auto      rank = ec.pg().rank();
  Scheduler sch{ec};
  if(N >= chem_env.ioptions.scf_options.restart_size && fs::exists(fname[FileType::Ortho]) &&
     fs::exists(fname[FileType::OrthoJson])) {
    if(rank == 0) {
      std::cout << "Reading orthogonalizer from disk ..." << std::endl << std::endl;
      auto jX                    = ParserUtils::json_from_file(fname[FileType::OrthoJson]);
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
        scf_vars.ttensors.X_alpha = {scf_vars.tN_bc, scf_vars.tNortho_bc};
        scf_vars.ttensors.X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
        Tensor<TensorType>::allocate(&scalapack_info.ec, scf_vars.ttensors.X_alpha);
        scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.X_alpha, fname[FileType::Ortho],
                                           chem_env.ioptions.scf_options.debug, true);
      }
    }
#else
    scf_vars.ttensors.X_alpha = {scf_vars.tAO, scf_vars.tAO_ortho};
    sch.allocate(scf_vars.ttensors.X_alpha).execute();
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.X_alpha, fname[FileType::Ortho],
                                       chem_env.ioptions.scf_options.debug, true);
#endif
  }
  else {
    scf_compute.compute_orthogonalizer(ec, chem_env, scf_vars, scalapack_info, scf_vars.ttensors);

    if(rank == 0) {
      json jX;
      jX["ortho_dims"] = {chem_env.sys_data.nbf_orig, chem_env.sys_data.nbf};
      ParserUtils::json_to_file(jX, fname[FileType::OrthoJson]);
    }

    if(N >= chem_env.ioptions.scf_options.restart_size) {
#if defined(USE_SCALAPACK)
      if(scalapack_info.pg.is_valid())
        scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.X_alpha, fname[FileType::Ortho],
                                           chem_env.ioptions.scf_options.debug);
#else
      scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.X_alpha, fname[FileType::Ortho],
                                         chem_env.ioptions.scf_options.debug);
#endif
    }
  }

#if defined(USE_SCALAPACK)
  if(scalapack_info.pg.is_valid()) {
    scf_vars.ttensors.F_BC = {scf_vars.tN_bc, scf_vars.tN_bc};
    scf_vars.ttensors.F_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    scf_vars.ttensors.C_alpha_BC = {scf_vars.tN_bc, scf_vars.tNortho_bc};
    scf_vars.ttensors.C_alpha_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
    Tensor<TensorType>::allocate(&scalapack_info.ec, scf_vars.ttensors.F_BC,
                                 scf_vars.ttensors.C_alpha_BC);
    if(chem_env.sys_data.is_unrestricted) {
      scf_vars.ttensors.C_beta_BC = {scf_vars.tN_bc, scf_vars.tNortho_bc};
      scf_vars.ttensors.C_beta_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      Tensor<TensorType>::allocate(&scalapack_info.ec, scf_vars.ttensors.C_beta_BC);
    }
  }
#endif

} // scf_orthogonalizer

void exachem::scf::DefaultSCFEngine::declare_main_tensors(ExecutionContext& ec, ChemEnv& chem_env) {
  const TiledIndexSpace& tAO  = scf_vars.tAO;
  const TiledIndexSpace& tAOt = scf_vars.tAOt;

  scf_vars.ttensors.ehf_tamm = Tensor<TensorType>{};
  scf_vars.ttensors.F_dummy  = {tAOt, tAOt}; // not allocated

  scf_vars.ttensors.ehf_tmp      = {tAO, tAO};
  scf_vars.ttensors.F_alpha      = {tAO, tAO};
  scf_vars.ttensors.D_alpha      = {tAO, tAO};
  scf_vars.ttensors.D_diff       = {tAO, tAO};
  scf_vars.ttensors.D_last_alpha = {tAO, tAO};
  scf_vars.ttensors.F_alpha_tmp  = {tAO, tAO};
  scf_vars.ttensors.FD_alpha     = {tAO, tAO};
  scf_vars.ttensors.FDS_alpha    = {tAO, tAO};

  scf_vars.ttensors.C_alpha  = {tAO, scf_vars.tAO_ortho};
  scf_vars.ttensors.C_occ_a  = {tAO, scf_vars.tAO_occ_a};
  scf_vars.ttensors.C_occ_aT = {scf_vars.tAO_occ_a, tAO};

  // TODO: Enable only for DFT
  scf_vars.ttensors.VXC_alpha = {tAO, tAO};
  scf_vars.ttensors.VXC_beta  = {tAO, tAO};

  if(chem_env.sys_data.is_unrestricted) {
    scf_vars.ttensors.C_beta       = {tAO, scf_vars.tAO_ortho};
    scf_vars.ttensors.C_occ_b      = {tAO, scf_vars.tAO_occ_b};
    scf_vars.ttensors.C_occ_bT     = {scf_vars.tAO_occ_b, tAO};
    scf_vars.ttensors.ehf_beta_tmp = {tAO, tAO};
    scf_vars.ttensors.F_beta       = {tAO, tAO};
    scf_vars.ttensors.D_beta       = {tAO, tAO};
    scf_vars.ttensors.D_last_beta  = {tAO, tAO};
    scf_vars.ttensors.F_beta_tmp   = {tAO, tAO};
    scf_vars.ttensors.FD_beta      = {tAO, tAO};
    scf_vars.ttensors.FDS_beta     = {tAO, tAO};
  }
  Tensor<TensorType>::allocate(
    &ec, scf_vars.ttensors.F_alpha, scf_vars.ttensors.C_alpha, scf_vars.ttensors.C_occ_a,
    scf_vars.ttensors.C_occ_aT, scf_vars.ttensors.D_alpha, scf_vars.ttensors.D_last_alpha,
    scf_vars.ttensors.D_diff, scf_vars.ttensors.F_alpha_tmp, scf_vars.ttensors.ehf_tmp,
    scf_vars.ttensors.ehf_tamm, scf_vars.ttensors.FD_alpha, scf_vars.ttensors.FDS_alpha);
  if(chem_env.sys_data.is_unrestricted)
    Tensor<TensorType>::allocate(&ec, scf_vars.ttensors.F_beta, scf_vars.ttensors.C_beta,
                                 scf_vars.ttensors.C_occ_b, scf_vars.ttensors.C_occ_bT,
                                 scf_vars.ttensors.D_beta, scf_vars.ttensors.D_last_beta,
                                 scf_vars.ttensors.F_beta_tmp, scf_vars.ttensors.ehf_beta_tmp,
                                 scf_vars.ttensors.FD_beta, scf_vars.ttensors.FDS_beta);

  if(chem_env.sys_data.is_ks) Tensor<TensorType>::allocate(&ec, scf_vars.ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted)
    Tensor<TensorType>::allocate(&ec, scf_vars.ttensors.VXC_beta);

  if(scf_vars.do_dens_fit) {
    scf_vars.ttensors.xyZ =
      Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO, scf_vars.tdfAO}; // n,n,ndf
    scf_vars.ttensors.xyK =
      Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO, scf_vars.tdfAO};           // n,n,ndf
    scf_vars.ttensors.Vm1 = Tensor<TensorType>{scf_vars.tdfAO, scf_vars.tdfAO}; // ndf, ndf
    if(!scf_vars.direct_df) Tensor<TensorType>::allocate(&ec, scf_vars.ttensors.xyK);
  }

  // Setup tiled index spaces when a fitting basis is provided
  //  dfCocc;
  IndexSpace dfCocc            = {range(0, chem_env.sys_data.nelectrons_alpha)};
  scf_vars.tdfCocc             = {dfCocc, chem_env.ioptions.scf_options.dfAO_tilesize};
  std::tie(scf_vars.dCocc_til) = scf_vars.tdfCocc.labels<1>("all");

} // declare main tensors

void exachem::scf::DefaultSCFEngine::deallocate_main_tensors(ExecutionContext& ec,
                                                             ChemEnv&          chem_env) {
  for(auto x: scf_vars.ttensors.ehf_tamm_hist) Tensor<TensorType>::deallocate(x);

  for(auto x: scf_vars.ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
  for(auto x: scf_vars.ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
  for(auto x: scf_vars.ttensors.D_hist) Tensor<TensorType>::deallocate(x);

  if(chem_env.sys_data.is_unrestricted) {
    for(auto x: scf_vars.ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: scf_vars.ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: scf_vars.ttensors.D_beta_hist) Tensor<TensorType>::deallocate(x);
  }
  if(scf_vars.do_dens_fit) {
    if(scf_vars.direct_df) { Tensor<TensorType>::deallocate(scf_vars.ttensors.Vm1); }
    else { Tensor<TensorType>::deallocate(scf_vars.ttensors.xyK); }
  }
  if(chem_env.sys_data.is_ks) {
    Tensor<TensorType>::deallocate(scf_vars.ttensors.VXC_alpha);
    if(chem_env.sys_data.is_unrestricted)
      Tensor<TensorType>::deallocate(scf_vars.ttensors.VXC_beta);
  }
  if(chem_env.sys_data.is_qed) {
    Tensor<TensorType>::deallocate(
      scf_vars.ttensors.QED_Dx, scf_vars.ttensors.QED_Dy, scf_vars.ttensors.QED_Dz,
      scf_vars.ttensors.QED_Qxx, scf_vars.ttensors.QED_Qxy, scf_vars.ttensors.QED_Qxz,
      scf_vars.ttensors.QED_Qyy, scf_vars.ttensors.QED_Qyz, scf_vars.ttensors.QED_Qzz,
      scf_vars.ttensors.QED_1body, scf_vars.ttensors.QED_2body);
  }

  Tensor<TensorType>::deallocate(
    scf_vars.ttensors.H1, scf_vars.ttensors.S1, scf_vars.ttensors.T1, scf_vars.ttensors.V1,
    scf_vars.ttensors.F_alpha_tmp, scf_vars.ttensors.ehf_tmp, scf_vars.ttensors.ehf_tamm,
    scf_vars.ttensors.F_alpha, scf_vars.ttensors.C_alpha, scf_vars.ttensors.C_occ_a,
    scf_vars.ttensors.C_occ_aT, scf_vars.ttensors.D_alpha, scf_vars.ttensors.D_diff,
    scf_vars.ttensors.D_last_alpha, scf_vars.ttensors.FD_alpha, scf_vars.ttensors.FDS_alpha);

  if(chem_env.sys_data.is_unrestricted)
    Tensor<TensorType>::deallocate(scf_vars.ttensors.F_beta, scf_vars.ttensors.C_beta,
                                   scf_vars.ttensors.C_occ_b, scf_vars.ttensors.C_occ_bT,
                                   scf_vars.ttensors.D_beta, scf_vars.ttensors.D_last_beta,
                                   scf_vars.ttensors.F_beta_tmp, scf_vars.ttensors.ehf_beta_tmp,
                                   scf_vars.ttensors.FD_beta, scf_vars.ttensors.FDS_beta);

} // deallocate_main_tensors

void exachem::scf::DefaultSCFEngine::scf_final_io(ExecutionContext& ec, ChemEnv& chem_env) {
  auto rank = ec.pg().rank();

  if(!chem_env.ioptions.scf_options.noscf) {
    if(rank == 0) cout << "writing orbitals and density to disk ... ";
    scf_output.rw_md_disk(ec, chem_env, scalapack_info, scf_vars.ttensors, scf_vars.etensors,
                          files_prefix);
    if(rank == 0) cout << "done." << endl;
  }

  scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.H1, fname[FileType::Hcore],
                                     chem_env.ioptions.scf_options.debug);
  if(chem_env.sys_data.is_ks) {
    // write vxc to disk
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.VXC_alpha, fname[FileType::VxcAlpha],
                                       chem_env.ioptions.scf_options.debug);
    if(chem_env.sys_data.is_unrestricted)
      scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.VXC_beta, fname[FileType::VxcBeta],
                                         chem_env.ioptions.scf_options.debug);
  }
  if(chem_env.sys_data.is_qed) {
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.QED_Dx, fname[FileType::QEDDx],
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.QED_Dy, fname[FileType::QEDDy],
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.QED_Dz, fname[FileType::QEDDz],
                                       chem_env.ioptions.scf_options.debug);
    scf_output.rw_mat_disk<TensorType>(scf_vars.ttensors.QED_Qxx, fname[FileType::QEDQxx],
                                       chem_env.ioptions.scf_options.debug);
  }
} // scf_final_io

void exachem::scf::DefaultSCFEngine::setup_tiled_index_space(ExecutionContext& exc,
                                                             ChemEnv&          chem_env) {
  // Setup tiled index spaces
  IndexSpace AO;
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
GauXC::XCIntegrator<Matrix>
exachem::scf::DefaultSCFEngine::get_gauxc_integrator(ExecutionContext& ec, ChemEnv& chem_env) {
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

void exachem::scf::DefaultSCFEngine::add_snk_contribution(
  ExecutionContext& ec, ChemEnv& chem_env, GauXC::XCIntegrator<Matrix>& gauxc_integrator) {
  auto rank = ec.pg().rank();
  // Add snK contribution
  if(chem_env.sys_data.do_snK) {
    const auto snK_start = std::chrono::high_resolution_clock::now();
    scf::gauxc::compute_exx<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors,
                                        scf_vars.etensors, gauxc_integrator);
    const auto snK_stop = std::chrono::high_resolution_clock::now();
    const auto snK_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((snK_stop - snK_start)).count();
    auto debug = chem_env.ioptions.scf_options.debug;
    if(rank == 0 && debug)
      std::cout << std::fixed << std::setprecision(2) << "snK: " << snK_time << "s, ";
  }
} // add_nk_contribution

void exachem::scf::DefaultSCFEngine::compute_update_xc(
  ExecutionContext& ec, ChemEnv& chem_env, GauXC::XCIntegrator<Matrix>& gauxc_integrator,
  double& ehf) {
  Scheduler  sch{ec};
  auto       rank  = ec.pg().rank();
  const bool is_ks = chem_env.sys_data.is_ks;
  if(is_ks) {
    const auto xcf_start = std::chrono::high_resolution_clock::now();
    gauxc_exc            = scf::gauxc::compute_xcf<TensorType>(ec, chem_env, scf_vars.ttensors,
                                                    scf_vars.etensors, gauxc_integrator);

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
    sch(scf_vars.ttensors.F_alpha() += scf_vars.ttensors.VXC_alpha());
    if(chem_env.sys_data.is_unrestricted) {
      // clang-format off
      sch
        (scf_vars.ttensors.F_alpha() += scf_vars.ttensors.VXC_beta())
        (scf_vars.ttensors.F_beta()  += scf_vars.ttensors.VXC_alpha())
        (scf_vars.ttensors.F_beta()  += -1.0 * scf_vars.ttensors.VXC_beta());
      // clang-format on
    }
    sch.execute();
  }

} // compute_update_xc

#endif

void exachem::scf::DefaultSCFEngine::process_molden_data(ExecutionContext& ec, ChemEnv& chem_env) {
  ec_molden.check_molden(chem_env.ioptions.scf_options.moldenfile);
  if(ec_molden.molden_file_valid) ec_molden.read_geom_molden(chem_env);
  if(ec_molden.molden_exists && ec_molden.molden_file_valid) {
    chem_env.shells = ec_molden.read_basis_molden(chem_env);
    chem_env.shells = ec_molden.renormalize_libint_shells(chem_env.shells);
    // if(chem_env.ioptions.scf_options.gaussian_type == "spherical")
    // chem_env.shells.set_pure(true); else chem_env.shells.set_pure(false); // use cartesian
    // gaussians
    set_basis_purity(chem_env, chem_env.shells);
  }

} // process_molden_data

void exachem::scf::DefaultSCFEngine::setup_density_fitting(ExecutionContext& exc,
                                                           ChemEnv&          chem_env) {
  auto rank = exc.pg().rank();
  if(scf_vars.do_dens_fit) {
    scf_vars.dfbs = libint2::BasisSet(chem_env.ioptions.scf_options.dfbasis, chem_env.atoms);

    //  if(chem_env.ioptions.scf_options.gaussian_type == "spherical") scf_vars.dfbs.set_pure(true);
    //     else scf_vars.dfbs.set_pure(false); // use cartesian gaussians
    set_basis_purity(chem_env, scf_vars.dfbs);
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
    scf_vars.do_dens_fit ? new DFFockEngine(chem_env.shells, scf_vars.dfbs) : nullptr);
  // End setup for fitting basis

} // setup_density_fitting

double exachem::scf::DefaultSCFEngine::calculate_diis_error(bool is_uhf, size_t ndiis) {
  double lediis = pow(tamm::norm(scf_vars.ttensors.diis_hist[ndiis - 1]), 2);
  if(is_uhf) { lediis += pow(tamm::norm(scf_vars.ttensors.diis_beta_hist[ndiis - 1]), 2); }
  return lediis;
} // calculate_diis_energy

void exachem::scf::DefaultSCFEngine::reset_fock_and_save_last_density(ExecutionContext& exc,
                                                                      ChemEnv&          chem_env) {
  // clang-format off
  Scheduler sch{exc};
sch (scf_vars.ttensors.F_alpha_tmp() = 0)
          (scf_vars.ttensors.D_last_alpha(scf_vars.mu,scf_vars.nu) = scf_vars.ttensors.D_alpha(scf_vars.mu,scf_vars.nu))
          .execute();
  // clang-format on

  if(chem_env.sys_data.is_unrestricted) {
    // clang-format off
        sch (scf_vars.ttensors.F_beta_tmp() = 0)
            (scf_vars.ttensors.D_last_beta(scf_vars.mu,scf_vars.nu) = scf_vars.ttensors.D_beta(scf_vars.mu,scf_vars.nu))
            .execute();
    // clang-format on
  }
} // reset_fock_density

void exachem::scf::DefaultSCFEngine::handle_energy_bumps(ExecutionContext& exc, ChemEnv& chem_env,
                                                         SCFIterationState& scf_state) {
  const bool is_uhf = chem_env.sys_data.is_unrestricted;
  auto       rank   = exc.pg().rank();

  if(scf_state.ediff > 0.0) scf_state.nbumps += 1;

  if(scf_state.nbumps > scf_state.nbumps_max &&
     scf_state.ndiis >= (size_t) chem_env.ioptions.scf_options.diis_hist) {
    scf_state.nbumps = 0;
    scf_vars.idiis   = 0;
    if(rank == 0) std::cout << "Resetting DIIS" << std::endl;
    for(auto x: scf_vars.ttensors.diis_hist) Tensor<TensorType>::deallocate(x);
    for(auto x: scf_vars.ttensors.fock_hist) Tensor<TensorType>::deallocate(x);
    scf_vars.ttensors.diis_hist.clear();
    scf_vars.ttensors.fock_hist.clear();
    if(is_uhf) {
      for(auto x: scf_vars.ttensors.diis_beta_hist) Tensor<TensorType>::deallocate(x);
      for(auto x: scf_vars.ttensors.fock_beta_hist) Tensor<TensorType>::deallocate(x);
      scf_vars.ttensors.diis_beta_hist.clear();
      scf_vars.ttensors.fock_beta_hist.clear();
    }
  }
} // handle_energy_bumps()

void exachem::scf::DefaultSCFEngine::print_write_iteration(ExecutionContext& exc, ChemEnv& chem_env,
                                                           double             loop_time,
                                                           SCFIterationState& scf_state) {
  auto rank = exc.pg().rank();
  if(rank == 0) {
    std::cout << std::setw(4) << std::right << scf_state.iter << "  " << std::setw(10);
    if(chem_env.ioptions.scf_options.debug) {
      std::cout << std::fixed << std::setprecision(18) << scf_state.ehf;
      std::cout << std::scientific << std::setprecision(18);
    }
    else {
      std::cout << std::fixed << std::setprecision(10) << scf_state.ehf;
      std::cout << std::scientific << std::setprecision(2);
    }
    std::cout << ' ' << std::scientific << std::setw(12) << scf_state.ediff;
    std::cout << ' ' << std::setw(12) << scf_state.rmsd;
    std::cout << ' ' << std::setw(12) << scf_state.ediis << ' ';
    std::cout << ' ' << std::setw(10) << std::fixed << std::setprecision(1) << loop_time << ' '
              << endl;

    auto& iter_info =
      chem_env.sys_data.results["output"]["SCF"]["iter"][std::to_string(scf_state.iter)];
    auto set_iter = [&](const std::string& key, auto value) { iter_info[key] = value; };
    set_iter("energy", scf_state.ehf);
    set_iter("e_diff", scf_state.ediff);
    set_iter("rmsd", scf_state.rmsd);
    set_iter("ediis", scf_state.ediis);
    iter_info["performance"] = {{"total_time", loop_time}};
  }
  if(scf_state.iter % chem_env.ioptions.scf_options.writem == 0 ||
     chem_env.ioptions.scf_options.writem == 1) {
    scf_output.rw_md_disk(exc, chem_env, scalapack_info, scf_vars.ttensors, scf_vars.etensors,
                          files_prefix);
  }
  if(chem_env.ioptions.scf_options.debug)
    scf_output.print_energies(exc, chem_env, scf_vars.ttensors, scf_vars.etensors, scf_vars,
                              scalapack_info);

} // print_energy_iteration

bool exachem::scf::DefaultSCFEngine::check_convergence(ExecutionContext& exc, ChemEnv& chem_env,
                                                       SCFIterationState& scf_state) {
  double conve = chem_env.ioptions.scf_options.conve;
  double convd = chem_env.ioptions.scf_options.convd;
  if(scf_state.iter >= static_cast<size_t>(chem_env.ioptions.scf_options.maxiter)) {
    scf_state.is_conv = false;
    return false;
  }
  if((fabs(scf_state.ediff) > conve) || (fabs(scf_state.rmsd) > convd) ||
     (fabs(scf_state.ediis) > 10.0 * conve))
    return true;
  else return false;
} // check_convergence

void exachem::scf::DefaultSCFEngine::compute_fock_matrix(ExecutionContext& ec, ChemEnv& chem_env,
                                                         bool is_uhf, const bool do_schwarz_screen,
                                                         Matrix& SchwarzK, const size_t& max_nprim4,
                                                         std::vector<size_t>& shell2bf,
                                                         bool&                is_3c_init) {
  Scheduler sch{ec};
  if(chem_env.sys_data.is_ks) { // or rohf
    sch(scf_vars.ttensors.F_alpha_tmp() = 0).execute();
    if(chem_env.sys_data.is_unrestricted) sch(scf_vars.ttensors.F_beta_tmp() = 0).execute();

    auto xHF_adjust = xHF;
    // TODO: skip for non-CC methods
    if(!chem_env.ioptions.task_options.scf) xHF_adjust = 1.0;
    // build a new Fock matrix
    scf_iter.compute_2bf<TensorType>(
      ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4,
      scf_vars.ttensors, scf_vars.etensors, is_3c_init, scf_vars.do_dens_fit, xHF_adjust);

    // Add QED contribution;
    // CHECK

    if(chem_env.sys_data.do_qed) {
      scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors);
    }
  }
  else if(scf_vars.lshift > 0) {
    // Remove level shift from Fock matrix
    double lval = chem_env.sys_data.is_restricted ? 0.5 * scf_vars.lshift : scf_vars.lshift;
    // clang-format off
        sch
        (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.ku) = scf_vars.ttensors.S1(scf_vars.mu,scf_vars.nu) * scf_vars.ttensors.D_last_alpha(scf_vars.nu,scf_vars.ku))
        (scf_vars.ttensors.F_alpha(scf_vars.mu,scf_vars.ku) += lval * scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu) * scf_vars.ttensors.S1(scf_vars.nu,scf_vars.ku))
        .execute();
    // clang-format on

    if(is_uhf) {
      // clang-format off
          sch
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.ku) = scf_vars.ttensors.S1(scf_vars.mu,scf_vars.nu) * scf_vars.ttensors.D_last_beta(scf_vars.nu,scf_vars.ku))
          (scf_vars.ttensors.F_beta(scf_vars.mu,scf_vars.ku) += lval * scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu) * scf_vars.ttensors.S1(scf_vars.nu,scf_vars.ku))
          .execute();
      // clang-format on
    }
  }

} // compute_fock_matrix

std::tuple<Tensor<TensorType>, Tensor<TensorType>, TiledIndexSpace>
exachem::scf::DefaultSCFEngine::update_movecs(ExecutionContext& ec, ChemEnv& chem_env) {
  auto               rank = ec.pg().rank();
  IndexSpace         AO_ortho;
  TiledIndexSpace    tAO_ortho;
  Tensor<TensorType> C_alpha_tamm, C_beta_tamm;
  Scheduler          schg{ec};
  AO_ortho     = {range(0, (size_t) (chem_env.sys_data.nbf_orig - chem_env.sys_data.n_lindep))};
  tAO_ortho    = {AO_ortho, chem_env.ioptions.scf_options.AO_tilesize};
  C_alpha_tamm = {scf_vars.tAO, tAO_ortho};
  C_beta_tamm  = {scf_vars.tAO, tAO_ortho};
  scf_vars.ttensors.VXC_alpha = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};
  if(chem_env.sys_data.is_unrestricted)
    scf_vars.ttensors.VXC_beta = Tensor<TensorType>{scf_vars.tAO, scf_vars.tAO};

  schg.allocate(C_alpha_tamm);
  if(chem_env.sys_data.is_unrestricted) schg.allocate(C_beta_tamm);
  if(chem_env.sys_data.is_ks) schg.allocate(scf_vars.ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted)
    schg.allocate(scf_vars.ttensors.VXC_beta);
  schg.execute();

  scf_output.rw_mat_disk<TensorType>(C_alpha_tamm, fname[FileType::AlphaMovecs],
                                     chem_env.ioptions.scf_options.debug, true);
  if(chem_env.sys_data.is_unrestricted)
    scf_output.rw_mat_disk<TensorType>(C_beta_tamm, fname[FileType::BetaMovecs],
                                       chem_env.ioptions.scf_options.debug, true);

  if(rank == 0 && chem_env.ioptions.scf_options.molden) {
    Matrix C_a = tamm_to_eigen_matrix(C_alpha_tamm);
    if(chem_env.sys_data.is_unrestricted)
      std::cout << "[MOLDEN] molden write for UHF unsupported!" << std::endl;
    else ec_molden.write_molden(chem_env, C_a, scf_vars.etensors.eps_a, files_prefix);
  }

  if(chem_env.sys_data.is_ks) schg.deallocate(scf_vars.ttensors.VXC_alpha);
  if(chem_env.sys_data.is_ks && chem_env.sys_data.is_unrestricted)
    schg.deallocate(scf_vars.ttensors.VXC_beta);
  schg.execute();

  ec.pg().barrier();

  return std::make_tuple(C_alpha_tamm, C_beta_tamm, tAO_ortho);
} // update_movecs

/* ↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓↑↓
 */

void exachem::scf::DefaultSCFEngine::run(ExecutionContext& exc, ChemEnv& chem_env) {
  auto              rank = exc.pg().rank();
  SCFIterationState scf_state;
  auto              hf_t1 = std::chrono::high_resolution_clock::now();
  process_molden_data(exc, chem_env);

  const int N = chem_env.shells.nbf();

  ProcGroupData pgdata = get_spg_data(exc, N, -1, 50, chem_env.ioptions.scf_options.nnodes);
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
    auto print = [](const std::string& label, auto value) {
      std::cout << label << value << std::endl;
    };
    print("\nNumber of basis functions = ", N);
    print("\nTotal number of shells = ", chem_env.shells.size());
    print("\nTotal number of electrons = ", nelectrons);
    print("  # of alpha electrons    = ", chem_env.sys_data.nelectrons_alpha);
    print("  # of beta electons      = ", chem_env.sys_data.nelectrons_beta);
    std::cout << std::endl
              << "Nuclear repulsion energy  = " << std::setprecision(15) << enuc << std::endl
              << std::endl;

    auto& sys_info               = chem_env.sys_data.results["output"]["system_info"];
    sys_info["nshells"]          = chem_env.shells.size();
    sys_info["nelectrons_total"] = nelectrons;
    sys_info["nelectrons_alpha"] = chem_env.sys_data.nelectrons_alpha;
    sys_info["nelectrons_beta"]  = chem_env.sys_data.nelectrons_beta;
    chem_env.write_sinfo();
  }

  // Compute non-negligible shell-pair list

  scf_compute.compute_shellpair_list(exc, chem_env.shells, scf_vars);
  setup_tiled_index_space(exc, chem_env);

  Scheduler schg{exc};
  // Fock matrices allocated on world group

  Tensor<TensorType> Fa_global, Fb_global;

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
  scf_restart.run(exc, chem_env, files_prefix);

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
    scf_compute.compute_trafo(chem_env.shells, scf_vars.etensors);

    scf_vars.direct_df = scf_vars.do_dens_fit && chem_env.ioptions.scf_options.direct_df;
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
    setup_libecpint(ec, chem_env, ecps, libecp_shells);

    ec.pg().barrier();

    Scheduler sch{ec};

    const TiledIndexSpace& tAO = scf_vars.tAO;
    // const TiledIndexSpace& tAOt = scf_vars.tAOt;

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/
    scf_compute.compute_hamiltonian<TensorType>(ec, scf_vars, chem_env, scf_vars.ttensors,
                                                scf_vars.etensors);
    if(chem_env.sys_data.is_qed) qed_tensors_1e(ec, chem_env);
    if(chem_env.sys_data.has_ecp) {
      Tensor<TensorType> ECP{tAO, tAO};
      Tensor<TensorType>::allocate(&ec, ECP);
      scf_guess.compute_ecp_ints(ec, scf_vars, ECP, libecp_shells, ecps);
      sch(scf_vars.ttensors.H1() += ECP()).deallocate(ECP).execute();
    }

    /*** =========================== ***/
    /*** build initial-guess density ***/
    /*** =========================== ***/

    scf_orthogonalizer(ec, chem_env);

    // pre-compute data for Schwarz bounds
    Matrix SchwarzK;
    if(!scf_vars.do_dens_fit || scf_vars.direct_df) {
      if(N >= chem_env.ioptions.scf_options.restart_size && fs::exists(fname[FileType::Schwarz])) {
        if(rank == 0) cout << "Read Schwarz matrix from disk ... " << endl;

        SchwarzK = scf_output.read_scf_mat<TensorType>(fname[FileType::Schwarz]);
      }
      else {
        // if(rank == 0) cout << "pre-computing data for Schwarz bounds... " << endl;
        SchwarzK = scf_compute.compute_schwarz_ints<>(ec, scf_vars, chem_env.shells);
        if(rank == 0) scf_output.write_scf_mat<TensorType>(SchwarzK, fname[FileType::Schwarz]);
      }
    }
    hf_t1 = std::chrono::high_resolution_clock::now();

    declare_main_tensors(ec, chem_env);

    if(scf_vars.do_dens_fit) {
      std::tie(scf_vars.d_mu, scf_vars.d_nu, scf_vars.d_ku)    = scf_vars.tdfAO.labels<3>("all");
      std::tie(scf_vars.d_mup, scf_vars.d_nup, scf_vars.d_kup) = scf_vars.tdfAOt.labels<3>("all");
      scf_iter.init_ri<TensorType>(ec, chem_env, scalapack_info, scf_vars, scf_vars.etensors,
                                   scf_vars.ttensors);
    }
    // const auto do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;

    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)

    if(chem_env.ioptions.scf_options.restart || chem_env.ioptions.scf_options.noscf) {
      // This was originally scf_restart.restart()
      scf_restart.run(ec, chem_env, scalapack_info, scf_vars.ttensors, scf_vars.etensors,
                      files_prefix);
      if(!scf_vars.do_dens_fit || scf_vars.direct_df || chem_env.sys_data.is_ks ||
         chem_env.sys_data.do_snK) {
        tamm_to_eigen_tensor(scf_vars.ttensors.D_alpha, scf_vars.etensors.D_alpha);
        if(chem_env.sys_data.is_unrestricted) {
          tamm_to_eigen_tensor(scf_vars.ttensors.D_beta, scf_vars.etensors.D_beta);
        }
      }
      ec.pg().barrier();
    }
    else if(ec_molden.molden_exists) {
      auto N      = chem_env.sys_data.nbf_orig;
      auto Northo = chem_env.sys_data.nbf;

      scf_vars.etensors.C_alpha.setZero(N, Northo);
      if(chem_env.sys_data.is_unrestricted) scf_vars.etensors.C_beta.setZero(N, Northo);

      if(rank == 0) {
        cout << endl << "Reading from molden file provided ..." << endl;
        if(ec_molden.molden_file_valid) {
          ec_molden.read_molden<TensorType>(chem_env, scf_vars.etensors.C_alpha,
                                            scf_vars.etensors.C_beta);
        }
      }

      scf_compute.compute_density<TensorType>(ec, chem_env, scf_vars, scalapack_info,
                                              scf_vars.ttensors, scf_vars.etensors);
      // X=C?

      ec.pg().barrier();
    }
    else {
      if(rank == 0) cout << "Superposition of Atomic Density Guess ..." << endl;
      scf_guess.compute_sad_guess<TensorType>(ec, chem_env, scf_vars, scalapack_info,
                                              scf_vars.etensors, scf_vars.ttensors);

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
      tamm_to_eigen_tensor(scf_vars.ttensors.S1, S);
      if(chem_env.sys_data.is_restricted)
        cout << "debug #electrons       = "
             << (int) std::round((scf_vars.etensors.D_alpha * S).trace()) << endl;
      if(chem_env.sys_data.is_unrestricted) {
        cout << "debug #alpha electrons = "
             << (int) std::round((scf_vars.etensors.D_alpha * S).trace()) << endl;
        cout << "debug #beta  electrons = "
             << (int) std::round((scf_vars.etensors.D_beta * S).trace()) << endl;
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
    const bool do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
    size_t     max_nprim         = chem_env.shells.max_nprim();
    // const size_t        max_nprim4        = max_nprim * max_nprim * max_nprim * max_nprim;
    const size_t        max_nprim4 = static_cast<size_t>(std::pow(max_nprim, 4));
    std::vector<size_t> shell2bf   = chem_env.shells.shell2bf();

    if(!scf_vars.do_dens_fit && scf_vars.do_load_bal) {
      // Collect task info
      auto [s1vec, s2vec, ntask_vec] = scf_iter.compute_2bf_taskinfo<TensorType>(
        ec, chem_env, scf_vars, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4,
        scf_vars.ttensors, scf_vars.etensors, scf_vars.do_dens_fit);

      auto [s1_all, s2_all, ntasks_all] =
        gather_task_vectors<TensorType>(ec, s1vec, s2vec, ntask_vec);

      int tmdim = 0;
      if(rank == 0) {
        Loads dummyLoads;
        /***generate load balanced task map***/
        dummyLoads.readLoads(s1_all, s2_all, ntasks_all);
        dummyLoads.simpleLoadBal(ec.pg().size().value());
        tmdim = std::max(dummyLoads.maxS1, dummyLoads.maxS2);
        scf_vars.etensors.taskmap.resize(tmdim + 1, tmdim + 1);
        // value in this array is the rank that executes task i,j
        // -1 indicates a task i,j that can be skipped
        scf_vars.etensors.taskmap.setConstant(-1);
        // cout<<"creating task map"<<endl;
        dummyLoads.createTaskMap(scf_vars.etensors.taskmap);
        // cout<<"task map creation completed"<<endl;
      }
      ec.pg().broadcast(&tmdim, 0);
      if(rank != 0) scf_vars.etensors.taskmap.resize(tmdim + 1, tmdim + 1);
      ec.pg().broadcast(scf_vars.etensors.taskmap.data(), scf_vars.etensors.taskmap.size(), 0);
    }

    if(chem_env.ioptions.scf_options.noscf) {
      // clang-format off
      sch (scf_vars.ttensors.F_alpha_tmp() = 0)
          (scf_vars.ttensors.D_last_alpha(scf_vars.mu,scf_vars.nu) = scf_vars.ttensors.D_alpha(scf_vars.mu,scf_vars.nu))
          .execute();
      // clang-format on

      if(chem_env.sys_data.is_unrestricted) {
        // clang-format off
        sch (scf_vars.ttensors.F_beta_tmp() = 0)
            (scf_vars.ttensors.D_last_beta(scf_vars.mu,scf_vars.nu) = scf_vars.ttensors.D_beta(scf_vars.mu,scf_vars.nu))
            .execute();
        // clang-format on
      }

      // F_alpha = H1 + F_alpha_tmp
      scf_iter.compute_2bf<TensorType>(
        ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4,
        scf_vars.ttensors, scf_vars.etensors, scf_state.is_3c_init, scf_vars.do_dens_fit, xHF);

      // Add QED contribution
      if(chem_env.sys_data.do_qed) {
        scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors);
      }

#if defined(USE_GAUXC)
      add_snk_contribution(ec, chem_env, gauxc_integrator);
#endif

      if(chem_env.sys_data.is_restricted) {
        // clang-format off
        sch
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu)  = scf_vars.ttensors.H1(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu) += scf_vars.ttensors.F_alpha(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tamm()      = 0.5 * scf_vars.ttensors.D_alpha() * scf_vars.ttensors.ehf_tmp())
          .execute();
        // clang-format on
      }

      if(chem_env.sys_data.is_unrestricted) {
        // clang-format off
        sch
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu)  = scf_vars.ttensors.H1(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu) += scf_vars.ttensors.F_alpha(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tamm()      = 0.5 * scf_vars.ttensors.D_alpha() * scf_vars.ttensors.ehf_tmp())
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu)  = scf_vars.ttensors.H1(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tmp(scf_vars.mu,scf_vars.nu) += scf_vars.ttensors.F_beta(scf_vars.mu,scf_vars.nu))
          (scf_vars.ttensors.ehf_tamm()     += 0.5 * scf_vars.ttensors.D_beta()  * scf_vars.ttensors.ehf_tmp())
          .execute();
        // clang-format on
      }

      scf_state.ehf = get_scalar(scf_vars.ttensors.ehf_tamm);

#if defined(USE_GAUXC)
      compute_update_xc(ec, chem_env, gauxc_integrator, scf_state.ehf);
#endif

      scf_state.ehf += enuc;
      if(rank == 0)
        std::cout << std::setprecision(18) << "Total HF energy after restart: " << scf_state.ehf
                  << std::endl;
    }

    // SCF main loop

    const bool is_uhf                 = chem_env.sys_data.is_unrestricted;
    bool       scf_main_loop_continue = true;

    do {
      if(chem_env.ioptions.scf_options.noscf) break;

      const auto loop_start = std::chrono::high_resolution_clock::now();
      ++scf_state.iter;

      // Save a copy of the energy and the density
      double ehf_last = scf_state.ehf;
      // resetting the Fock matrix and saving the last density.
      reset_fock_and_save_last_density(ec, chem_env);

      // auto D_tamm_nrm = norm(scf_vars.ttensors.D_alpha);
      // if(rank==0) cout << std::setprecision(18) << "norm of D_tamm: " << D_tamm_nrm << endl;

      // build a new Fock matrix
      scf_iter.compute_2bf<TensorType>(
        ec, chem_env, scalapack_info, scf_vars, do_schwarz_screen, shell2bf, SchwarzK, max_nprim4,
        scf_vars.ttensors, scf_vars.etensors, scf_state.is_3c_init, scf_vars.do_dens_fit, xHF);

      // Add QED contribution
      if(chem_env.sys_data.do_qed) {
        scf_qed.compute_QED_2body<TensorType>(ec, chem_env, scf_vars, scf_vars.ttensors);
      }

      std::tie(scf_state.ehf, scf_state.rmsd) = scf_iter.scf_iter_body<TensorType>(
        ec, chem_env, scalapack_info, scf_state.iter, scf_vars, scf_vars.ttensors, scf_vars.etensors
#if defined(USE_GAUXC)
        ,
        gauxc_integrator
#endif
      );

      // DIIS error
      scf_state.ndiis = scf_vars.ttensors.diis_hist.size();
      scf_state.ediis = calculate_diis_error(is_uhf, scf_state.ndiis);

      scf_state.ehf += enuc;
      // compute difference with last iteration

      scf_state.ediff = scf_state.ehf - ehf_last;
      handle_energy_bumps(ec, chem_env, scf_state);

      const auto loop_stop = std::chrono::high_resolution_clock::now();
      const auto loop_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((loop_stop - loop_start)).count();

      print_write_iteration(ec, chem_env, loop_time, scf_state);

      // if(rank==0) cout << "D at the end of iteration: " << endl << std::setprecision(6) <<
      // scf_vars.etensors.D_alpha << endl;
      scf_main_loop_continue = check_convergence(ec, chem_env, scf_state);
      if(!scf_state.is_conv) break;

      // Reset lshift to input option.
      if(fabs(scf_state.ediff) > 1e-2) scf_vars.lshift = chem_env.ioptions.scf_options.lshift;

    } while(scf_main_loop_continue); // SCF main loop

    if(rank == 0) {
      std::cout.precision(13);
      if(scf_state.is_conv) cout << endl << "** Total SCF energy = " << scf_state.ehf << endl;
      else {
        cout << endl << std::string(50, '*') << endl;
        cout << std::string(10, ' ') << "ERROR: SCF calculation does not converge!!!" << endl;
        cout << std::string(50, '*') << endl;
      }
    }
    if(chem_env.ioptions.dplot_options.cube) write_dplot_data(ec, chem_env);

    compute_fock_matrix(ec, chem_env, is_uhf, do_schwarz_screen, SchwarzK, max_nprim4, shell2bf,
                        scf_state.is_3c_init);

    sch(Fa_global(scf_vars.mu, scf_vars.nu) = scf_vars.ttensors.F_alpha(scf_vars.mu, scf_vars.nu));
    if(chem_env.sys_data.is_unrestricted)
      sch(Fb_global(scf_vars.mu, scf_vars.nu) = scf_vars.ttensors.F_beta(scf_vars.mu, scf_vars.nu));
    sch.execute();
    if(rank == 0)
      std::cout << std::endl
                << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;
    scf_output.print_energies(ec, chem_env, scf_vars.ttensors, scf_vars.etensors, scf_vars,
                              scalapack_info);

    if(rank == 0 && chem_env.ioptions.scf_options.mulliken_analysis && scf_state.is_conv) {
      Matrix S = tamm_to_eigen_matrix(scf_vars.ttensors.S1);
      scf_output.print_mulliken(chem_env, scf_vars.etensors.D_alpha, scf_vars.etensors.D_beta, S);
    }

    scf_final_io(ec, chem_env);
    deallocate_main_tensors(ec, chem_env);

    ec.flush_and_sync();

#if defined(USE_SCALAPACK)
    if(scalapack_info.pg.is_valid()) {
      Tensor<TensorType>::deallocate(scf_vars.ttensors.F_BC, scf_vars.ttensors.X_alpha,
                                     scf_vars.ttensors.C_alpha_BC);
      if(chem_env.sys_data.is_unrestricted)
        Tensor<TensorType>::deallocate(scf_vars.ttensors.C_beta_BC);
      scalapack_info.ec.flush_and_sync();
      scalapack_info.ec.pg().destroy_coll();
    }
#else
    sch.deallocate(scf_vars.ttensors.X_alpha);
    sch.execute();
#endif

    ec.pg().destroy_coll();
  } // end scf subgroup

  // C,F1 is not allocated for ranks > hf_nranks
  exc.pg().barrier();
  exc.pg().broadcast(&scf_state.is_conv, 0);

  if(!scf_state.is_conv) { tamm_terminate("Please check SCF input parameters"); }

  // F, C are not deallocated.
  chem_env.sys_data.n_occ_alpha = chem_env.sys_data.nelectrons_alpha;
  chem_env.sys_data.n_occ_beta  = chem_env.sys_data.nelectrons_beta;
  chem_env.sys_data.n_vir_alpha =
    chem_env.sys_data.nbf_orig - chem_env.sys_data.n_occ_alpha - chem_env.sys_data.n_lindep;
  chem_env.sys_data.n_vir_beta =
    chem_env.sys_data.nbf_orig - chem_env.sys_data.n_occ_beta - chem_env.sys_data.n_lindep;

  exc.pg().broadcast(&scf_state.ehf, 0);
  exc.pg().broadcast(&enuc, 0);
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
    chem_env.sys_data.results["output"]["SCF"]["final_energy"] = scf_state.ehf;
    chem_env.sys_data.results["output"]["SCF"]["n_iterations"] = scf_state.iter;
  }

  auto [C_alpha_tamm, C_beta_tamm, tAO_ortho] = update_movecs(exc, chem_env);

  chem_env.is_context.AO_opt   = scf_vars.tAO;
  chem_env.is_context.AO_tis   = scf_vars.tAOt;
  chem_env.is_context.AO_ortho = tAO_ortho;

  // chem_env.scf_context.scf_converged = true;

  chem_env.scf_context.update(scf_state.ehf, enuc, scf_vars.shell_tile_map, C_alpha_tamm, Fa_global,
                              C_beta_tamm, Fb_global, chem_env.ioptions.scf_options.noscf);
} // END of scf_hf(ExecutionContext& exc, ChemEnv& chem_env)
