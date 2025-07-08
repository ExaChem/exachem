/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/libint2_includes.hpp"
#include "exachem/scf/scf_tensors.hpp"
#include <libecpint.hpp>

using namespace tamm;
using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t =
  std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>; // in same order as

namespace exachem::scf {

class SCFData {
public:
  // diis
  int                      idiis        = 0;
  bool                     switch_diis  = false;
  double                   exc          = 0.0;
  double                   eqed         = 0.0;
  bool                     do_snK       = false;
  bool                     do_dens_fit  = false;
  bool                     direct_df    = false;
  bool                     do_load_bal  = false;
  bool                     lshift_reset = false;
  double                   lshift       = 0;
  double                   xHF          = 1.0;
  libecpint::ECPIntegrator ecp_factory;

  // AO spaces
  tamm::TiledIndexSpace   tAO, tAO_ortho, tAO_occ_a, tAO_occ_b, tAOt; // tAO_ld
  std::vector<tamm::Tile> AO_tiles;
  std::vector<tamm::Tile> AO_opttiles;
  std::vector<size_t>     shell_tile_map;

  // AO spaces for BC representation
  tamm::TiledIndexSpace tN_bc;
  tamm::TiledIndexSpace tNortho_bc;

  // Tensors for AO and SCF calculations
  EigenTensors etensors; // Holds Eigen-based tensors for SCF and AO operations
  TAMMTensors  ttensors; //  Holds TAMM-based tensors for SCF and AO operations

  // AO labels
  // tamm::TiledIndexLabel mu_ld, nu_ld;
  tamm::TiledIndexLabel mu, nu, ku;
  tamm::TiledIndexLabel mup, nup, kup;

  tamm::TiledIndexLabel mu_oa, nu_oa;
  tamm::TiledIndexLabel mu_ob, nu_ob;

  // DF spaces
  libint2::BasisSet     dfbs;
  tamm::IndexSpace      dfAO;
  std::vector<Tile>     dfAO_tiles;
  std::vector<Tile>     dfAO_opttiles;
  std::vector<size_t>   df_shell_tile_map;
  tamm::TiledIndexSpace tdfAO, tdfAOt;
  tamm::TiledIndexSpace tdfCocc;
  tamm::TiledIndexLabel dCocc_til;

  // DF labels
  tamm::TiledIndexLabel d_mu, d_nu, d_ku;
  tamm::TiledIndexLabel d_mup, d_nup, d_kup;

  // shellpair list
  shellpair_list_t obs_shellpair_list;        // shellpair list for OBS
  shellpair_list_t dfbs_shellpair_list;       // shellpair list for DFBS
  shellpair_list_t minbs_shellpair_list;      // shellpair list for minBS
  shellpair_list_t obs_shellpair_list_atom;   // shellpair list for OBS for specfied atom
  shellpair_list_t minbs_shellpair_list_atom; // shellpair list for minBS for specfied atom

  // shellpair data
  shellpair_data_t obs_shellpair_data;        // shellpair data for OBS
  shellpair_data_t dfbs_shellpair_data;       // shellpair data for DFBS
  shellpair_data_t minbs_shellpair_data;      // shellpair data for minBS
  shellpair_data_t obs_shellpair_data_atom;   // shellpair data for OBS for specfied atom
  shellpair_data_t minbs_shellpair_data_atom; // shellpair data for minBS for specfied atom
};
} // namespace exachem::scf
