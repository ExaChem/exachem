/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "common/system_data.hpp"
#include "scf/scf_main.hpp"
#include "tamm/eigen_utils.hpp"

#if defined(USE_UPCXX)
#include "tamm/ga_over_upcxx.hpp"
#endif

using namespace tamm;
using TAMM_GA_SIZE = int64_t;

bool cd_debug = false;

#if 0
std::tuple<Index, Index, size_t, size_t> get_shell_ids(const std::vector<size_t>& shell_tile_map,
                                                       const IndexVector&         AO_tiles,
                                                       IndexVector& maxblockid, size_t bfu,
                                                       size_t bfv) {
  auto mi0  = maxblockid[0];
  auto mi1  = maxblockid[1];
  auto s1rs = 0l;
  auto s1re = shell_tile_map[mi0];
  if(mi0 > 0) s1rs = shell_tile_map[mi0 - 1] + 1;
  auto s2rs = 0l;
  auto s2re = shell_tile_map[mi1];
  if(mi1 > 0) s2rs = shell_tile_map[mi1 - 1] + 1;

  Index s1               = 0;
  Index s2               = 0;
  auto  curshelloffset_i = 0U;
  auto  curshelloffset_j = 0U;

  for(auto x1 = s1rs; x1 <= s1re; ++x1) {
    if(bfu >= curshelloffset_i && bfu < curshelloffset_i + AO_tiles[x1]) {
      s1 = x1;
      break;
    }
    curshelloffset_i += AO_tiles[x1];
  }

  for(auto x1 = s2rs; x1 <= s2re; ++x1) {
    if(bfv >= curshelloffset_j && bfv < curshelloffset_j + AO_tiles[x1]) {
      s2 = x1;
      break;
    }
    curshelloffset_j += AO_tiles[x1];
  }

  return std::make_tuple(s1, s2, curshelloffset_i, curshelloffset_j);
}
#endif

#if !defined(USE_UPCXX)
int64_t ac_fetch_add(int ga_ac, int64_t index, int64_t amount) {
  auto ret = NGA_Read_inc64(ga_ac, &index, amount);
  return ret;
}
#endif

std::tuple<TiledIndexSpace, TAMM_SIZE> setup_mo_red(SystemData sys_data, bool triples = false) {
  // TAMM_SIZE nao = sys_data.nbf;
  TAMM_SIZE n_occ_alpha = sys_data.n_occ_alpha;
  TAMM_SIZE n_vir_alpha = sys_data.n_vir_alpha;

  Tile tce_tile = sys_data.options_map.ccsd_options.tilesize;
  if(!triples) {
    if((tce_tile < static_cast<Tile>(sys_data.nbf / 10) || tce_tile < 50) &&
       !sys_data.options_map.ccsd_options.force_tilesize) {
      tce_tile = static_cast<Tile>(sys_data.nbf / 10);
      if(tce_tile < 50) tce_tile = 50; // 50 is the default tilesize for CCSD.
      if(ProcGroup::world_rank() == 0)
        std::cout << std::endl << "Resetting CCSD tilesize to: " << tce_tile << std::endl;
    }
  }
  else tce_tile = sys_data.options_map.ccsd_options.ccsdt_tilesize;

  const TAMM_SIZE total_orbitals = sys_data.nbf;

  // Construction of tiled index space MO
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {{"occ", {range(0, n_occ_alpha)}}, {"virt", {range(n_occ_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_occ_alpha / est_nt + (x < (n_occ_alpha % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_vir_alpha / est_nt + (x < (n_vir_alpha % est_nt)));

  TiledIndexSpace MO{MO_IS, mo_tiles};

  return std::make_tuple(MO, total_orbitals);
}

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(SystemData sys_data, bool triples = false,
                                                 int nactv = 0) {
  TAMM_SIZE n_occ_alpha = sys_data.n_occ_alpha;
  TAMM_SIZE n_occ_beta  = sys_data.n_occ_beta;

  Tile tce_tile      = sys_data.options_map.ccsd_options.tilesize;
  bool balance_tiles = sys_data.options_map.ccsd_options.balance_tiles;
  if(!triples) {
    if((tce_tile < static_cast<Tile>(sys_data.nbf / 10) || tce_tile < 50 || tce_tile > 100) &&
       !sys_data.options_map.ccsd_options.force_tilesize) {
      tce_tile = static_cast<Tile>(sys_data.nbf / 10);
      if(tce_tile < 50) tce_tile = 50;   // 50 is the default tilesize for CCSD.
      if(tce_tile > 100) tce_tile = 100; // 100 is the max tilesize for CCSD.
      if(ProcGroup::world_rank() == 0)
        std::cout << std::endl << "Resetting CCSD tilesize to: " << tce_tile << std::endl;
    }
  }
  else {
    balance_tiles = false;
    tce_tile      = sys_data.options_map.ccsd_options.ccsdt_tilesize;
  }

  TAMM_SIZE nmo         = sys_data.nmo;
  TAMM_SIZE n_vir_alpha = sys_data.n_vir_alpha;
  TAMM_SIZE n_vir_beta  = sys_data.n_vir_beta;
  TAMM_SIZE nocc        = sys_data.nocc;

  const TAMM_SIZE total_orbitals = nmo;

  // Construction of tiled index space MO
  TAMM_SIZE  virt_alpha_int = nactv;
  TAMM_SIZE  virt_beta_int  = virt_alpha_int;
  TAMM_SIZE  virt_alpha_ext = n_vir_alpha - nactv;
  TAMM_SIZE  virt_beta_ext  = total_orbitals - (nocc + nactv + n_vir_alpha);
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {
      {"occ", {range(0, nocc)}},
      {"occ_alpha", {range(0, n_occ_alpha)}},
      {"occ_beta", {range(n_occ_alpha, nocc)}},
      {"virt", {range(nocc, total_orbitals)}},
      {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
      {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}},
      {"virt_alpha_int", {range(nocc, nocc + nactv)}},
      {"virt_beta_int", {range(nocc + n_vir_alpha, nocc + nactv + n_vir_alpha)}},
      {"virt_int",
       {range(nocc, nocc + nactv), range(nocc + n_vir_alpha, nocc + nactv + n_vir_alpha)}},
      {"virt_alpha_ext", {range(nocc + nactv, nocc + n_vir_alpha)}},
      {"virt_beta_ext", {range(nocc + nactv + n_vir_alpha, total_orbitals)}},
      {"virt_ext",
       {range(nocc + nactv, nocc + n_vir_alpha),
        range(nocc + nactv + n_vir_alpha, total_orbitals)}},
    },
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  if(!balance_tiles) {
    tamm::Tile est_nt    = n_occ_alpha / tce_tile;
    tamm::Tile last_tile = n_occ_alpha % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = n_occ_beta / tce_tile;
    last_tile = n_occ_beta % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    // est_nt = n_vir_alpha/tce_tile;
    // last_tile = n_vir_alpha%tce_tile;
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
    // if(last_tile>0) mo_tiles.push_back(last_tile);
    est_nt    = virt_alpha_int / tce_tile;
    last_tile = virt_alpha_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = virt_alpha_ext / tce_tile;
    last_tile = virt_alpha_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    // est_nt = n_vir_beta/tce_tile;
    // last_tile = n_vir_beta%tce_tile;
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
    // if(last_tile>0) mo_tiles.push_back(last_tile);
    est_nt    = virt_beta_int / tce_tile;
    last_tile = virt_beta_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = virt_beta_ext / tce_tile;
    last_tile = virt_beta_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
  }
  else {
    tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(n_occ_alpha / est_nt + (x < (n_occ_alpha % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(n_occ_beta / est_nt + (x < (n_occ_beta % est_nt)));

    // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_alpha / est_nt + (x<(n_vir_alpha %
    // est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_alpha_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(virt_alpha_int / est_nt + (x < (virt_alpha_int % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_alpha_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(virt_alpha_ext / est_nt + (x < (virt_alpha_ext % est_nt)));

    // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_beta / est_nt + (x<(n_vir_beta %
    // est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_beta_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(virt_beta_int / est_nt + (x < (virt_beta_int % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * virt_beta_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(virt_beta_ext / est_nt + (x < (virt_beta_ext % est_nt)));
  }

  TiledIndexSpace MO{MO_IS, mo_tiles}; //{ova,ova,ovb,ovb}};

  return std::make_tuple(MO, total_orbitals);
}

void update_sysdata(SystemData& sys_data, TiledIndexSpace& MO, bool is_mso = true) {
  const bool do_freeze      = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;
  TAMM_SIZE  total_orbitals = sys_data.nmo;
  if(do_freeze) {
    sys_data.nbf -= (sys_data.n_frozen_core + sys_data.n_frozen_virtual);
    sys_data.n_occ_alpha -= sys_data.n_frozen_core;
    sys_data.n_vir_alpha -= sys_data.n_frozen_virtual;
    if(is_mso) {
      sys_data.n_occ_beta -= sys_data.n_frozen_core;
      sys_data.n_vir_beta -= sys_data.n_frozen_virtual;
    }
    sys_data.update();
    if(!is_mso) std::tie(MO, total_orbitals) = setup_mo_red(sys_data);
    else std::tie(MO, total_orbitals) = setupMOIS(sys_data);
  }
}

Matrix reshape_mo_matrix(SystemData sys_data, Matrix& emat) {
  const int noa   = sys_data.n_occ_alpha;
  const int nob   = sys_data.n_occ_beta;
  const int nva   = sys_data.n_vir_alpha;
  const int nvb   = sys_data.n_vir_beta;
  const int nocc  = sys_data.nocc;
  const int N_eff = sys_data.nmo;

  const int n_frozen_core    = sys_data.n_frozen_core;
  const int n_frozen_virtual = sys_data.n_frozen_virtual;

  Matrix    cvec(N_eff, N_eff);
  const int block2_off     = 2 * n_frozen_core + noa;
  const int block3_off     = 2 * n_frozen_core + nocc;
  const int last_block_off = block3_off + n_frozen_virtual + nva;

  cvec.block(0, 0, noa, noa)          = emat.block(n_frozen_core, n_frozen_core, noa, noa);
  cvec.block(0, noa, noa, nob)        = emat.block(n_frozen_core, block2_off, noa, nob);
  cvec.block(0, nocc, noa, nva)       = emat.block(n_frozen_core, block3_off, noa, nva);
  cvec.block(0, nocc + nva, noa, nvb) = emat.block(n_frozen_core, last_block_off, noa, nvb);

  cvec.block(noa, 0, nob, noa)          = emat.block(block2_off, n_frozen_core, nob, noa);
  cvec.block(noa, noa, nob, nob)        = emat.block(block2_off, block2_off, nob, nob);
  cvec.block(noa, nocc, nob, nva)       = emat.block(block2_off, block3_off, nob, nva);
  cvec.block(noa, nocc + nva, nob, nvb) = emat.block(block2_off, last_block_off, nob, nvb);

  cvec.block(nocc, 0, nva, noa)          = emat.block(block3_off, n_frozen_core, nva, noa);
  cvec.block(nocc, noa, nva, nob)        = emat.block(block3_off, block2_off, nva, nob);
  cvec.block(nocc, nocc, nva, nva)       = emat.block(block3_off, block3_off, nva, nva);
  cvec.block(nocc, nocc + nva, nva, nvb) = emat.block(block3_off, last_block_off, nva, nvb);

  cvec.block(nocc + nva, 0, nvb, noa)    = emat.block(last_block_off, n_frozen_core, nvb, noa);
  cvec.block(nocc + nva, noa, nvb, nob)  = emat.block(last_block_off, block2_off, nvb, nob);
  cvec.block(nocc + nva, nocc, nvb, nva) = emat.block(last_block_off, block3_off, nvb, nva);
  cvec.block(nocc + nva, nocc + nva, nvb, nvb) =
    emat.block(last_block_off, last_block_off, nvb, nvb);
  emat.resize(0, 0);

  return cvec;
}

template<typename TensorType>
Tensor<TensorType> cd_svd(SystemData& sys_data, ExecutionContext& ec, TiledIndexSpace& tMO,
                          TiledIndexSpace& tAO, TAMM_SIZE& chol_count, const TAMM_GA_SIZE max_cvecs,
                          libint2::BasisSet& shells, Tensor<TensorType>& lcao, bool is_mso = true) {
  using libint2::Atom;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  double           diagtol    = sys_data.options_map.cd_options.diagtol;
  const tamm::Tile itile_size = sys_data.options_map.cd_options.itilesize;
  // const TAMM_GA_SIZE northo         = sys_data.nbf;
  const TAMM_GA_SIZE nao = sys_data.nbf_orig;

  auto rank = ec.pg().rank();

  TAMM_GA_SIZE N = tMO("all").max_num_indices();

  Matrix lcao_eig(nao, N);
  lcao_eig.setZero();
  tamm_to_eigen_tensor(lcao, lcao_eig);
  TensorType* k_movecs_sorted = lcao_eig.data();

  //
  // Cholesky decomposition
  //
  if(rank == 0) { cout << "Begin Cholesky Decomposition ... " << endl; }
  auto cd_t1 = std::chrono::high_resolution_clock::now();

  // Step A. Initialization
  int64_t iproc = rank.value();
  int64_t ndim  = 3;
  auto    nbf   = nao;
  int64_t count = 0; // Initialize chol vector count

#if !defined(USE_UPCXX)
  int g_chol_mo = 0;
#endif

#ifdef CD_SVD_THROTTLE
  int64_t cd_nranks = std::abs(std::log10(diagtol)) * nbf; // max cores
  auto nnodes = ec.nnodes() auto ppn       = ec.ppn();
  int                            cd_nnodes = cd_nranks / ppn;
  if(cd_nranks % ppn > 0 || cd_nnodes == 0) cd_nnodes++;
  if(cd_nnodes > nnodes) cd_nnodes = nnodes;
  cd_nranks = cd_nnodes * ppn;
  if(rank == 0)
    cout << "Total # of mpi ranks used for Cholesky decomposition: " << cd_nranks << endl
         << "  --> Number of nodes, mpi ranks per node: " << cd_nnodes << ", " << ppn << endl;
#endif

  int64_t dimsmo[3];
  int64_t chnkmo[3];
#if !defined(USE_UPCXX)
  int     nblockmo32[GA_MAX_DIM];
  int64_t nblockmo[GA_MAX_DIM];
#endif
  int64_t              size_map;
  std::vector<int64_t> k_map;
  int                  ga_eltype = C_DBL;

  auto create_map = [&](auto& dims, auto& nblock) {
    std::vector<int64_t> k_map(size_map);
    auto                 mi = 0;
    for(auto count_dim = 0; count_dim < 2; count_dim++) {
      auto size_blk = dims[count_dim] / nblock[count_dim];
      for(auto i = 0; i < nblock[count_dim]; i++) {
        k_map[mi] = size_blk * i;
        mi++;
      }
    }
    k_map[mi] = 0;
    return k_map;
  };

#if defined(USE_UPCXX)
  upcxx::team& team_ref = upcxx::world();
  upcxx::team* team     = &team_ref;
#else
  int ga_pg_default = GA_Pgroup_get_default();
  int ga_pg         = ga_pg_default;
#endif

#ifdef CD_SVD_THROTTLE
  const bool throttle_cd = ec.pg().size() > cd_nranks;

  if(iproc < cd_nranks) { // throttle

    if(throttle_cd) {
#if defined(USE_UPCXX)
      team = new upcxx::team(
        team->split((team->rank_me() < cd_nranks ? 0 : upcxx::team::color_none), 0));
#else
      int ranks[cd_nranks];
      for(int i = 0; i < cd_nranks; i++) ranks[i] = i;
      ga_pg = GA_Pgroup_create(ranks, cd_nranks);
      GA_Pgroup_set_default(ga_pg);
#endif
    }
#endif

    int64_t dims[3] = {nbf, nbf, max_cvecs};
    int64_t chnk[3] = {-1, -1, max_cvecs};
    int     nblock32[GA_MAX_DIM];
    int64_t nblock[GA_MAX_DIM];

#if defined(USE_UPCXX)
    ga_over_upcxx<TensorType>* g_chol = new ga_over_upcxx<TensorType>(3, dims, chnk, *team);
    g_chol->zero();

    int64_t dims2[3] = {nbf, nbf, 1};
    int64_t chnk2[3] = {-1, -1, 1};
#else
  int g_test        = NGA_Create64(ga_eltype, ndim, dims, const_cast<char*>("CholVecTmp"), chnk);
  NGA_Nblock(g_test, nblock32);
  NGA_Destroy(g_test);

  for(auto x = 0; x < GA_MAX_DIM; x++) nblock[x] = nblock32[x];

  size_map = nblock[0] + nblock[1] + nblock[2];

  k_map = create_map(dims, nblock);

  int g_chol =
    NGA_Create_irreg64(ga_eltype, 3, dims, const_cast<char*>("CholX"), nblock, &k_map[0]);

  GA_Zero(g_chol);

  int64_t dims2[2]   = {nbf, nbf};
  int64_t nblock2[2] = {nblock[0], nblock[1]};
#endif

    // TODO: Check k_map;
#if defined(USE_UPCXX)
    ga_over_upcxx<TensorType>* g_d = new ga_over_upcxx<TensorType>(3, dims2, chnk2, *team);
    ga_over_upcxx<TensorType>* g_r = new ga_over_upcxx<TensorType>(3, dims2, chnk2, *team);
    g_d->zero();
#else
  int     g_d =
    NGA_Create_irreg64(ga_eltype, 2, dims2, const_cast<char*>("ERI Diag"), nblock2, &k_map[0]);
  int g_r =
    NGA_Create_irreg64(ga_eltype, 2, dims2, const_cast<char*>("ERI Res"), nblock2, &k_map[0]);

  int64_t lo_b[GA_MAX_DIM]; // The lower limits of blocks of B
  int64_t lo_r[GA_MAX_DIM]; // The lower limits of blocks of R
  int64_t lo_d[GA_MAX_DIM]; // The lower limits of blocks of D
  int64_t hi_b[GA_MAX_DIM]; // The upper limits of blocks of B
  int64_t hi_r[GA_MAX_DIM]; // The upper limits of blocks of R
  int64_t hi_d[GA_MAX_DIM]; // The upper limits of blocks of D
  int64_t ld_b[GA_MAX_DIM]; // The leading dims of blocks of B
  int64_t ld_r[GA_MAX_DIM]; // The leading dims of blocks of R
  int64_t ld_d[GA_MAX_DIM]; // The leading dims of blocks of D

  // Distribution Check
  NGA_Distribution64(g_chol, iproc, lo_b, hi_b);
  NGA_Distribution64(g_d, iproc, lo_d, hi_d);
  NGA_Distribution64(g_r, iproc, lo_r, hi_r);
#endif

    auto shell2bf = map_shell_to_basis_function(shells);
    auto bf2shell = map_basis_function_to_shell(shells);

    auto cd_t2 = std::chrono::high_resolution_clock::now();
    auto cd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
    if(iproc == 0) {
      std::cout << endl
                << std::fixed << std::setprecision(2) << "Setup time: " << cd_time << " secs"
                << endl;
    }

    // Step B. Compute the diagonal
    Engine      engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
    const auto& buf = engine.results();

    for(size_t s1 = 0; s1 != shells.size(); ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1        = shells[s1].size();
#if defined(USE_UPCXX)
      for(size_t s2 = 0; s2 != shells.size(); ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2        = shells[s2].size();

        if(g_d->coord_is_local(bf1_first, bf2_first, 0, 0)) {
#else
    decltype(bf1_first) lo_d0 = lo_d[0];
    decltype(bf1_first) hi_d0 = hi_d[0];
    if(lo_d0 <= bf1_first && bf1_first <= hi_d0) {
      for(size_t s2 = 0; s2 != shells.size(); ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2        = shells[s2].size();

        decltype(bf2_first) lo_d1 = lo_d[1];
        decltype(bf2_first) hi_d1 = hi_d[1];
        if(lo_d1 <= bf2_first && bf2_first <= hi_d1) {
#endif

          // TODO: Screening
          engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
          const auto* buf_1212 = buf[0];
          if(buf_1212 == nullptr) continue; // if all integrals screened out, skip to next quartet

          std::vector<TensorType> k_eri(n1 * n2);
          for(decltype(n1) f1 = 0; f1 != n1; ++f1) {
            // const auto bf1 = f1 + bf1_first;
            for(decltype(n2) f2 = 0; f2 != n2; ++f2) {
              // const auto bf2 = f2 + bf2_first;
              auto f1212          = f1 * n2 * n1 * n2 + f2 * n1 * n2 + f1 * n2 + f2;
              k_eri[f1 * n2 + f2] = buf_1212[f1212];
              //// cout << f1212 << " " << s1 << s2 << "(" << bf1 << bf2 << "|" << bf1 << bf2 << ")
              ///= " << DiagInt(bf1, bf2) << endl;
            }
          }
          int64_t ibflo[2] = {cd_ncast<size_t>(bf1_first), cd_ncast<size_t>(bf2_first)};
          int64_t ibfhi[2] = {cd_ncast<size_t>(bf1_first + n1 - 1),
                              cd_ncast<size_t>(bf2_first + n2 - 1)};
#if defined(USE_UPCXX)
          int64_t ld[4] = {cd_ncast<size_t>(n1), cd_ncast<size_t>(n2), 1, 1};
          g_d->put(ibflo[0], ibflo[1], 0, 0, ibfhi[0], ibfhi[1], 0, 0, &k_eri[0], ld);
#else
          int64_t     ld[1]    = {cd_ncast<size_t>(n2)};
          const void* from_buf = &k_eri[0];
          NGA_Put64(g_d, ibflo, ibfhi, const_cast<void*>(from_buf), ld);
#endif
        } // if s2
      }   // s2
#if !defined(USE_UPCXX)
    } //#if s1
#endif
  } // s1

  auto cd_t3 = std::chrono::high_resolution_clock::now();
  cd_time    = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t3 - cd_t2)).count();
  if(iproc == 0) {
    std::cout << endl
              << "Time for computing the diagonal: " << std::fixed << std::setprecision(2)
              << cd_time << " secs" << endl;
  }

// Step C. Find the coordinates of the maximum element of the diagonal.
#if defined(USE_UPCXX)
  int64_t indx_d0[4];
#else
    int64_t indx_d0[GA_MAX_DIM];
#endif
  TensorType val_d0;
#if defined(USE_UPCXX)
  g_d->maximum(val_d0, indx_d0[0], indx_d0[1], indx_d0[2], indx_d0[3]);
#else
    NGA_Select_elem64(g_d, const_cast<char*>("max"), &val_d0, indx_d0);
#endif

#if defined(USE_UPCXX)
  int64_t lo_x[4]; // The lower limits of blocks
  int64_t hi_x[4]; // The upper limits of blocks
  int64_t ld_x[4]; // The leading dims of blocks
#else
    int64_t lo_x[GA_MAX_DIM]; // The lower limits of blocks
    int64_t hi_x[GA_MAX_DIM]; // The upper limits of blocks
    int64_t ld_x[GA_MAX_DIM]; // The leading dims of blocks
#endif

  // Step D. Start the while loop
  while(val_d0 > diagtol && count < max_cvecs) {
#if defined(USE_UPCXX)
    g_r->zero();
#else
      NGA_Zero(g_r);
#endif
    auto bfu   = indx_d0[0];
    auto bfv   = indx_d0[1];
    auto s1    = bf2shell[bfu];
    auto n1    = shells[s1].size();
    auto s2    = bf2shell[bfv];
    auto n2    = shells[s2].size();
    auto n12   = n1 * n2;
    auto f1    = bfu - shell2bf[s1];
    auto f2    = bfv - shell2bf[s2];
    auto ind12 = f1 * n2 + f2;

    for(size_t s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3        = shells[s3].size();

#if defined(USE_UPCXX)
      for(decltype(s3) s4 = 0; s4 != shells.size(); ++s4) {
        auto bf4_first = shell2bf[s4];
        auto n4        = shells[s4].size();

        if(g_r->coord_is_local(bf3_first, bf4_first, 0, 0)) {
#else
        decltype(bf3_first) lo_r0 = lo_r[0];
        decltype(bf3_first) hi_r0 = hi_r[0];
        if(lo_r0 <= bf3_first && bf3_first <= hi_r0) {
          for(decltype(s3) s4 = 0; s4 != shells.size(); ++s4) {
            auto bf4_first = shell2bf[s4];
            auto n4        = shells[s4].size();

            decltype(bf4_first) lo_r1 = lo_r[1];
            decltype(bf4_first) hi_r1 = hi_r[1];
            if(lo_r1 <= bf4_first && bf4_first <= hi_r1) {
#endif
          engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
          const auto* buf_3412 = buf[0];
          if(buf_3412 == nullptr) continue; // if all integrals screened out, skip to next quartet

          std::vector<TensorType> k_eri(n3 * n4);
          for(decltype(n3) f3 = 0; f3 != n3; ++f3) {
            for(decltype(n4) f4 = 0; f4 != n4; ++f4) {
              auto f3412          = f3 * n4 * n12 + f4 * n12 + ind12;
              k_eri[f3 * n4 + f4] = buf_3412[f3412];
            }
          }

          int64_t ibflo[2] = {cd_ncast<size_t>(bf3_first), cd_ncast<size_t>(bf4_first)};
          int64_t ibfhi[2] = {cd_ncast<size_t>(bf3_first + n3 - 1),
                              cd_ncast<size_t>(bf4_first + n4 - 1)};
#if defined(USE_UPCXX)
          int64_t ld[4] = {cd_ncast<size_t>(n3), cd_ncast<size_t>(n4), 1, 1}; // n3
          g_r->put(ibflo[0], ibflo[1], 0, 0, ibfhi[0], ibfhi[1], 0, 0, &k_eri[0], ld);
#else
              const void* fbuf = &k_eri[0];
              // TODO
              int64_t ld[1] = {cd_ncast<size_t>(n4)}; // n3
              NGA_Put64(g_r, ibflo, ibfhi, const_cast<void*>(fbuf), ld);
#endif
        } // if s4
      }   // s4
#if !defined(USE_UPCXX)
    } // if s3
#endif
  } // s3

#if defined(USE_UPCXX)
  upcxx::barrier(*team);
#else
        NGA_Sync();
#endif

  // Step F. Update the residual
  lo_x[0] = indx_d0[0];
  lo_x[1] = indx_d0[1];
  lo_x[2] = 0;
  lo_x[3] = 0;
  hi_x[0] = indx_d0[0];
  hi_x[1] = indx_d0[1];
  hi_x[2] = count; // count>0? count : 0;
  hi_x[3] = 0;
  ld_x[0] = 1;
#if defined(USE_UPCXX)
  ld_x[1] = 1;
  ld_x[2] = hi_x[2] + 1;
  ld_x[3] = 1;
#else
        ld_x[1] = hi_x[2] + 1;
#endif

  TensorType *            indx_b, *indx_d, *indx_r;
  std::vector<TensorType> k_elems(max_cvecs);
  TensorType*             k_row = &k_elems[0];
#if defined(USE_UPCXX)
  g_chol->get(lo_x[0], lo_x[1], lo_x[2], lo_x[3], hi_x[0], hi_x[1], hi_x[2], hi_x[3], k_row, ld_x);

  auto g_r_iter    = g_r->local_chunks_begin();
  auto g_r_end     = g_r->local_chunks_end();
  auto g_chol_iter = g_chol->local_chunks_begin();
  auto g_chol_end  = g_chol->local_chunks_end();

  while(g_r_iter != g_r_end && g_chol_iter != g_chol_end) {
    ga_over_upcxx_chunk<TensorType>* g_r_chunk    = *g_r_iter;
    ga_over_upcxx_chunk<TensorType>* g_chol_chunk = *g_chol_iter;
    assert(g_r_chunk->same_coord(g_chol_chunk) && g_r_chunk->same_size_or_smaller(g_chol_chunk));

    ga_over_upcxx_chunk_view<TensorType> g_chol_view = g_chol_chunk->local_view();
    ga_over_upcxx_chunk_view<TensorType> g_r_view    = g_r_chunk->local_view();

    for(int64_t icount = 0; icount < count; icount++) {
      for(int64_t i = 0; i < g_r_view.get_chunk_size(0); i++) {
        for(int64_t j = 0; j < g_r_view.get_chunk_size(1); j++) {
          g_r_view.subtract(i, j, 0, 0, g_chol_view.read(i, j, icount, 0) * k_row[icount]);
        }
      }
    }

    g_r_iter++;
    g_chol_iter++;
  }
  assert(g_r_iter == g_r_end && g_chol_iter == g_chol_end);

  g_r_iter    = g_r->local_chunks_begin();
  g_r_end     = g_r->local_chunks_end();
  g_chol_iter = g_chol->local_chunks_begin();
  g_chol_end  = g_chol->local_chunks_end();

  while(g_r_iter != g_r_end && g_chol_iter != g_chol_end) {
    ga_over_upcxx_chunk<TensorType>* g_r_chunk    = *g_r_iter;
    ga_over_upcxx_chunk<TensorType>* g_chol_chunk = *g_chol_iter;
    assert(g_r_chunk->same_coord(g_chol_chunk) && g_r_chunk->same_size_or_smaller(g_chol_chunk));

    ga_over_upcxx_chunk_view<TensorType> g_r_view    = g_r_chunk->local_view();
    ga_over_upcxx_chunk_view<TensorType> g_chol_view = g_chol_chunk->local_view();

    for(auto i = 0; i < g_r_view.get_chunk_size(0); i++) {
      for(auto j = 0; j < g_r_view.get_chunk_size(1); j++) {
        auto tmp = g_r_view.read(i, j, 0, 0) / sqrt(val_d0);
        g_chol_view.write(i, j, count, 0, tmp);
      }
    }
    g_r_iter++;
    g_chol_iter++;
  }
  assert(g_r_iter == g_r_end && g_chol_iter == g_chol_end);
#else
        NGA_Get64(g_chol, lo_x, hi_x, k_row, ld_x);
        NGA_Access64(g_r, lo_r, hi_r, &indx_r, ld_r);
        NGA_Access64(g_chol, lo_b, hi_b, &indx_b, ld_b);

        for(decltype(count) icount = 0; icount < count; icount++) {
          for(int64_t i = 0; i <= hi_r[0] - lo_r[0]; i++) {
            for(int64_t j = 0; j <= hi_r[1] - lo_r[1]; j++) {
              indx_r[i * ld_r[0] + j] -=
                indx_b[icount + j * ld_b[1] + i * ld_b[1] * ld_b[0]] * k_row[icount];
            }
          }
        }

        NGA_Release64(g_chol, lo_b, hi_b);
        NGA_Release_update64(g_r, lo_r, hi_r);

        // Step G. Compute the new Cholesky vector
        NGA_Access64(g_r, lo_r, hi_r, &indx_r, ld_r);
        NGA_Access64(g_chol, lo_b, hi_b, &indx_b, ld_b);

        for(auto i = 0; i <= hi_r[0] - lo_r[0]; i++) {
          for(auto j = 0; j <= hi_r[1] - lo_r[1]; j++) {
            auto tmp = indx_r[i * ld_r[0] + j] / sqrt(val_d0);
            indx_b[count + j * ld_b[1] + i * ld_b[1] * ld_b[0]] = tmp;
          }
        }

        NGA_Release_update64(g_chol, lo_b, hi_b);
        NGA_Release64(g_r, lo_r, hi_r);
#endif

  // Step H. Increment count
  count++;

  // Step I. Update the diagonal
#if defined(USE_UPCXX)
  auto g_d_iter = g_d->local_chunks_begin();
  auto g_d_end  = g_d->local_chunks_end();
  g_chol_iter   = g_chol->local_chunks_begin();
  g_chol_end    = g_chol->local_chunks_end();

  while(g_d_iter != g_d_end && g_chol_iter != g_chol_end) {
    ga_over_upcxx_chunk<TensorType>* g_d_chunk    = *g_d_iter;
    ga_over_upcxx_chunk<TensorType>* g_chol_chunk = *g_chol_iter;
    assert(g_d_chunk->same_coord(g_chol_chunk) && g_d_chunk->same_size_or_smaller(g_chol_chunk));

    ga_over_upcxx_chunk_view<TensorType> g_chol_view = g_chol_chunk->local_view();
    ga_over_upcxx_chunk_view<TensorType> g_d_view    = g_d_chunk->local_view();

    for(auto i = 0; i < g_d_view.get_chunk_size(0); i++) {
      for(auto j = 0; j < g_d_view.get_chunk_size(1); j++) {
        auto tmp = g_chol_view.read(i, j, count - 1, 0);
        g_d_view.subtract(i, j, 0, 0, tmp * tmp);
      }
    }
    g_d_iter++;
    g_chol_iter++;
  }
  assert(g_d_iter == g_d_end && g_chol_iter == g_chol_end);
#else
        NGA_Access64(g_d, lo_d, hi_d, &indx_d, ld_d);
        NGA_Access64(g_chol, lo_b, hi_b, &indx_b, ld_b);

        for(auto i = 0; i <= hi_d[0] - lo_d[0]; i++) {
          for(auto j = 0; j <= hi_d[1] - lo_d[1]; j++) {
            auto tmp = indx_b[count - 1 + j * ld_b[1] + i * ld_b[1] * ld_b[0]];
            indx_d[i * ld_d[0] + j] -= tmp * tmp;
          }
        }

        NGA_Release64(g_chol, lo_b, hi_b);
        NGA_Release_update64(g_d, lo_d, hi_d);
#endif

  // Step J. Find the coordinates of the maximum element of the diagonal.
#if defined(USE_UPCXX)
  g_d->maximum(val_d0, indx_d0[0], indx_d0[1], indx_d0[2], indx_d0[3]);
#else
        NGA_Select_elem64(g_d, const_cast<char*>("max"), &val_d0, indx_d0);
#endif
}

if(iproc == 0) cout << "Number of cholesky vectors = " << count << endl;
#if defined(USE_UPCXX)
g_r->destroy();
g_d->destroy();
#else
      NGA_Destroy(g_r);
      NGA_Destroy(g_d);
#endif

auto cd_t4 = std::chrono::high_resolution_clock::now();
cd_time    = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t4 - cd_t3)).count();
if(iproc == 0) {
  std::cout << endl
            << "Time to compute cholesky vectors: " << std::fixed << std::setprecision(2) << cd_time
            << " secs" << endl;
}

update_sysdata(sys_data, tMO, is_mso);

TAMM_GA_SIZE N_eff = tMO("all").max_num_indices();

#if defined(USE_UPCXX)
dimsmo[0] = N;
dimsmo[1] = N;
dimsmo[2] = count;
#else
      dimsmo[0]   = N_eff;
      dimsmo[1]   = N_eff;
      dimsmo[2]   = count;
#endif
chnkmo[0] = -1;
chnkmo[1] = -1;
chnkmo[2] = count;

#if defined(USE_UPCXX)
ga_over_upcxx<TensorType>* g_chol_mo = new ga_over_upcxx<TensorType>(3, dimsmo, chnkmo, *team);
g_chol_mo->zero();
#else
      int g_test2 = NGA_Create64(ga_eltype, 3, dimsmo, const_cast<char*>("CholVecMOTmp"), chnkmo);
      NGA_Nblock(g_test2, nblockmo32);
      NGA_Destroy(g_test2);

      for(auto x = 0; x < GA_MAX_DIM; x++) nblockmo[x] = nblockmo32[x];

      size_map = nblockmo[0] + nblockmo[1] + nblockmo[2];
      k_map    = create_map(dimsmo, nblockmo);
      g_chol_mo =
        NGA_Create_irreg64(ga_eltype, 3, dimsmo, const_cast<char*>("CholXMO"), nblockmo, &k_map[0]);
      GA_Zero(g_chol_mo);
#endif

std::vector<TensorType> k_pj(N* nbf);
std::vector<TensorType> k_pq(N* N);

std::vector<TensorType> k_ij(nbf* nbf);
std::vector<TensorType> k_eval_r(nbf);

// #define DO_SVD 0
#if defined(DO_SVD)
auto svdtol = 1e-8; // TODO same as diagtol ?
#endif

cd_t1            = std::chrono::high_resolution_clock::now();
double cvpr_time = 0;

#if defined(USE_UPCXX)
atomic_counter_over_upcxx ga_ac(*team);
#else
      char    name[]        = "atomic-counter";
      int64_t num_counters_ = 1;
      int64_t init_val      = 0;
      int64_t size          = num_counters_;
      int     ga_ac         = NGA_Create_config64(MT_C_LONGLONG, 1, &size, name, nullptr, ga_pg);
      EXPECTS(ga_ac != 0);
      if(GA_Pgroup_nodeid(ga_pg) == 0) {
        int64_t   lo[1] = {0};
        int64_t   hi[1] = {num_counters_ - 1};
        int64_t   ld    = -1;
        long long buf[num_counters_];
        for(int i = 0; i < num_counters_; i++) { buf[i] = init_val; }
        NGA_Put64(ga_ac, lo, hi, buf, &ld);
      }
      GA_Pgroup_sync(ga_pg);
#endif

int64_t taskcount = 0;
#if defined(USE_UPCXX)
int64_t next = ga_ac.fetch_add(1);

/*
 * Necessary for progress. The atomic counter is stored on rank 0, and rank 0
 * may just proceed straight in to the loop below preventing others from
 * making progress in their fetch-adds?
 */
upcxx::barrier(*team);
#else
      int64_t next = ac_fetch_add(ga_ac, 0, 1);
#endif

const bool do_freeze = (sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0);

for(decltype(count) kk = 0; kk < count; kk++) {
  if(next == taskcount) {
    int64_t lo_ao[3] = {0, 0, kk};
    int64_t hi_ao[3] = {nbf - 1, nbf - 1, kk};
#if defined(USE_UPCXX)
    int64_t ld_ao[4] = {nbf, nbf, 1, 1};
#else
          int64_t ld_ao[2] = {nbf, 1};
#endif

#if defined(USE_UPCXX)
    g_chol->get(lo_ao[0], lo_ao[1], lo_ao[2], 0, hi_ao[0], hi_ao[1], hi_ao[2], 0, &k_ij[0], ld_ao);
#else
          NGA_Get64(g_chol, lo_ao, hi_ao, &k_ij[0], ld_ao);
#endif

#if defined(DO_SVD)
    // uplotri
    for(auto i = 0; i < nbf; i++)
      for(auto j = i + 1; j < nbf; j++) k_ij[i * nbf + j] = 0;

    // TODO
    LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', (BLA_LAPACK_INT) nbf, &k_ij[0], (BLA_LAPACK_INT) nbf,
                   &k_eval_r[0]);

    auto m = 0;
    for(auto i = 0; i < nbf; i++) {
      if(fabs(k_eval_r[i]) <= svdtol) continue;
      k_eval_r[m] = k_eval_r[i];
      for(auto x = 0; x < nbf; x++) k_ij[m * nbf + x] = k_ij[i * nbf + x];
      m++;
    }

    std::vector<TensorType> k_ij_tmp(nbf * m);
    for(auto i = 0; i < nbf; i++)
      for(auto j = 0; j < m; j++) k_ij_tmp[i * m + j] = k_ij[j * nbf + i];

    g_num += m;
    std::vector<TensorType> k_pi(N * m);
    std::vector<TensorType> k_qj(N * m);

    blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, N, m, nbf, 1.0,
               k_movecs_sorted, N, &k_ij[0], nbf, 0, &k_pi[0], N);

    for(auto x = 0; x < N * m; x++) k_qj[x] = k_pi[x];

    for(auto i = 0; i < N; i++) {
      auto sf = k_eval_r[i];
      for(auto j = 0; j < m; j++) k_pi[i * m + j] *= sf;
    }

    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::Trans, N, N, m, 1, &k_pi[0], N,
               &k_qj[0], N, 0, &k_pq[0], N);

#else

          //---------Two-Step-Contraction----
          auto cvpr_t1 = std::chrono::high_resolution_clock::now();
          blas::gemm(blas::Layout::RowMajor, blas::Op::Trans, blas::Op::NoTrans, N, nbf, nbf, 1,
                     k_movecs_sorted, N, &k_ij[0], nbf, 0, &k_pj[0], nbf);

          blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, N, nbf, 1,
                     &k_pj[0], nbf, k_movecs_sorted, N, 0, &k_pq[0], N);

          auto cvpr_t2 = std::chrono::high_resolution_clock::now();
          cvpr_time +=
            std::chrono::duration_cast<std::chrono::duration<double>>((cvpr_t2 - cvpr_t1)).count();

#endif

    int64_t lo_mo[3] = {0, 0, kk};
    int64_t hi_mo[3] = {N_eff - 1, N_eff - 1, kk};
#ifdef USE_UPCXX
    int64_t ld_mo[4] = {N_eff, N_eff, 1, 1};
#else
          int64_t ld_mo[2] = {N_eff, 1};
#endif

    if(do_freeze) {
      Matrix emat = Eigen::Map<Matrix>(k_pq.data(), N, N);
      k_pq.clear();

      Matrix cvec = reshape_mo_matrix(sys_data, emat);

      k_pq.resize(N_eff * N_eff);
      Eigen::Map<Matrix>(k_pq.data(), N_eff, N_eff) = cvec;
      cvec.resize(0, 0);
    }

#if defined(USE_UPCXX)
    g_chol_mo->put(lo_mo[0], lo_mo[1], lo_mo[2], 0, hi_mo[0], hi_mo[1], hi_mo[2], 0, &k_pq[0],
                   ld_mo);
    next = ga_ac.fetch_add(1);
#else
          NGA_Put64(g_chol_mo, lo_mo, hi_mo, &k_pq[0], ld_mo);
          next = ac_fetch_add(ga_ac, 0, 1);
#endif
  }
  taskcount++;

#if defined(USE_UPCXX)
  upcxx::progress();
#endif
}

#if defined(USE_UPCXX)
upcxx::barrier(*team);
#else
      GA_Pgroup_sync(ga_pg);
#endif

#if defined(USE_UPCXX)
g_chol->destroy();
#else
      NGA_Destroy(ga_ac);
      NGA_Destroy(g_chol);
#endif
k_pj.clear();
k_pq.clear();
k_ij.clear();
k_eval_r.clear();
k_eval_r.shrink_to_fit();

cd_t2   = std::chrono::high_resolution_clock::now();
cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
if(rank == 0) {
  std::cout << endl
            << "Time for constructing MO-based CholVpr tensor: " << std::fixed
            << std::setprecision(2) << cd_time << " secs" << endl;
  std::cout << "  --> Time for 2-step contraction: " << std::fixed << std::setprecision(2)
            << cvpr_time << " secs" << endl;
}

#ifdef CD_SVD_THROTTLE
#if defined(USE_UPCXX)
team = &team_ref;
#else
if(throttle_cd) GA_Pgroup_set_default(ga_pg_default);
#endif

} // end throttle
#endif

ec.pg().barrier();
#ifdef CD_SVD_THROTTLE
ec.pg().broadcast(&count, 1, 0);
#endif

cd_t1 = std::chrono::high_resolution_clock::now();

#ifdef CD_SVD_THROTTLE

dimsmo[0] = N;
dimsmo[1] = N;
dimsmo[2] = count;
chnkmo[0] = -1;
chnkmo[1] = -1;
chnkmo[2] = count;

#if defined(USE_UPCXX)
ga_over_upcxx<TensorType>* g_chol_mo_copy = new ga_over_upcxx<TensorType>(3, dimsmo, chnkmo, *team);
g_chol_mo_copy->zero();
#else
int g_test_mo = NGA_Create64(ga_eltype, 3, dimsmo, const_cast<char*>("CholVecMOTmp"), chnkmo);
NGA_Nblock(g_test_mo, nblockmo32);
NGA_Destroy(g_test_mo);

for(auto x = 0; x < GA_MAX_DIM; x++) nblockmo[x] = nblockmo32[x];

size_map = nblockmo[0] + nblockmo[1] + nblockmo[2];
k_map = create_map(dimsmo, nblockmo);
int g_chol_mo_copy =
  NGA_Create_irreg64(ga_eltype, 3, dimsmo, const_cast<char*>("CholXMOCopy"), nblockmo, &k_map[0]);
GA_Zero(g_chol_mo_copy);
#endif

if(iproc < cd_nranks) { // throttle
#if defined(USE_UPCXX)
  g_chol_mo->copy(g_chol_mo_copy);
  g_chol_mo->destroy();
#else
  GA_Pgroup_set_default(ga_pg);
  GA_Copy(g_chol_mo, g_chol_mo_copy);
  NGA_Destroy(g_chol_mo);
  GA_Pgroup_sync(ga_pg);
  GA_Pgroup_set_default(ga_pg_default);
#endif
}

ec.pg().barrier();
#else
#if defined(USE_UPCXX)
      ga_over_upcxx<TensorType>* g_chol_mo_copy = g_chol_mo;
#else
      int g_chol_mo_copy = g_chol_mo;
#endif
#endif

IndexSpace      CIp{range(0, count)};
TiledIndexSpace tCIp{CIp, static_cast<tamm::Tile>(itile_size)};

Tensor<TensorType> CholVpr_tamm{{tMO, tMO, tCIp},
                                {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
Tensor<TensorType>::allocate(&ec, CholVpr_tamm);

// convert g_chol_mo_copy to CholVpr_tamm
auto lambdacv = [&](const IndexVector& bid) {
  const IndexVector blockid = internal::translate_blockid(bid, CholVpr_tamm());

  auto block_dims   = CholVpr_tamm.block_dims(blockid);
  auto block_offset = CholVpr_tamm.block_offsets(blockid);

  const tamm::TAMM_SIZE dsize = CholVpr_tamm.block_size(blockid);

  int64_t lo[3] = {cd_ncast<size_t>(block_offset[0]), cd_ncast<size_t>(block_offset[1]),
                   cd_ncast<size_t>(block_offset[2])};
  int64_t hi[3] = {cd_ncast<size_t>(block_offset[0] + block_dims[0] - 1),
                   cd_ncast<size_t>(block_offset[1] + block_dims[1] - 1),
                   cd_ncast<size_t>(block_offset[2] + block_dims[2] - 1)};
#if defined(USE_UPCXX)
  int64_t ld[4] = {cd_ncast<size_t>(block_dims[0]), cd_ncast<size_t>(block_dims[1]),
                   cd_ncast<size_t>(block_dims[2]), 1};

  upcxx::progress();
#else
        int64_t ld[2] = {cd_ncast<size_t>(block_dims[1]), cd_ncast<size_t>(block_dims[2])};
#endif
  std::vector<TensorType> sbuf(dsize);
#if defined(USE_UPCXX)
  g_chol_mo_copy->get(lo[0], lo[1], lo[2], 0, hi[0], hi[1], hi[2], 0, &sbuf[0], ld);
#else
        NGA_Get64(g_chol_mo_copy, lo, hi, &sbuf[0], ld);
#endif

  CholVpr_tamm.put(blockid, sbuf);
#if defined(USE_UPCXX)
  upcxx::progress();
#endif
};

block_for(ec, CholVpr_tamm(), lambdacv);

#if defined(USE_UPCXX)
g_chol_mo_copy->destroy();
#else
      NGA_Destroy(g_chol_mo_copy);
#endif

cd_t2   = std::chrono::high_resolution_clock::now();
cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
if(rank == 0)
  std::cout << std::endl
            << "Time for CholVpr (GA->TAMM) conversion: " << std::fixed << std::setprecision(2)
            << cd_time << " secs" << endl;

#if defined(USE_UPCXX)
ga_ac.destroy();
#endif
#if 0
Tensor<TensorType> CholVuv_opt{tAO, tAO, tCIp};
Tensor<TensorType>::allocate(&ec, CholVuv_opt);

cd_t1 = std::chrono::high_resolution_clock::now();

// Contraction 1
Tensor<TensorType> CholVpv_tamm{tMO, tAO, tCIp};
Tensor<TensorType>::allocate(&ec, CholVpv_tamm);
Scheduler{ec}(CholVpv_tamm(pmo, mu, cindexp) = CTiled_tamm(nu, pmo) * CholVuv_opt(nu, mu, cindexp))
  .execute();
Tensor<TensorType>::deallocate(CholVuv_opt);

cd_t2   = std::chrono::high_resolution_clock::now();
cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
if(rank == 0)
  std::cout << std::endl << "Time for computing CholVpr: " << std::fixed << std::setprecision(2) << cd_time << " secs" << std::endl;

// Contraction 2
cd_t1 = std::chrono::high_resolution_clock::now();

Tensor<TensorType> CholVpr_tamm{{tMO, tMO, tCIp},
                                {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
// clang-format off
Scheduler{ec}
  .allocate(CholVpr_tamm)
  (CholVpr_tamm(pmo,rmo,cindexp) += CTiled_tamm(mu, rmo) * CholVpv_tamm(pmo, mu,cindexp))
  .deallocate(CholVpv_tamm)
  .execute();
// clang-format on

cd_t2   = std::chrono::high_resolution_clock::now();
cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
if(rank == 0)
  std::cout << std::endl << "Time for computing CholVpr: " << std::fixed << std::setprecision(2) << cd_time << " secs" << std::endl;

#endif

chol_count = count;
return CholVpr_tamm;
}
