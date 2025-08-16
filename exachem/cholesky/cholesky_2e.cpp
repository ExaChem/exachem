/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cholesky/cholesky_2e.hpp"
using namespace exachem::scf;
bool cd_debug = false;
#define CD_USE_PGAS_API
#define CD_THROTTLE

template<typename T>
auto cd_tensor_zero(Tensor<T>& tens) {
#if !defined(USE_UPCXX)
  NGA_Zero(tens.ga_handle());
#endif
}

namespace exachem::cholesky_2e {

int get_ts_recommendation(ExecutionContext& ec, ChemEnv& chem_env) {
  int nranks   = ec.nnodes() * ec.ppn();
  int ts_guess = chem_env.ioptions.ccsd_options.tilesize;

  SystemData& sys_data = chem_env.sys_data;
  const auto  o_alpha  = sys_data.n_occ_alpha;
  const auto  v_alpha  = sys_data.n_vir_alpha;
  const auto  o_beta   = sys_data.n_occ_beta;
  const auto  v_beta   = sys_data.n_vir_beta;

  int           max_ts = 180; // gpu_mem=64
  tamm::meminfo minfo  = ec.mem_info();
  if(ec.has_gpu()) {
    const auto ranks_per_gpu = ec.ppn() / ec.gpn();
    const auto gpu_mem       = minfo.gpu_mem_per_device;
    if(gpu_mem <= 8) max_ts = 100;
    else if(gpu_mem <= 12) max_ts = 120;
    else if(gpu_mem <= 16) max_ts = 130;
    else if(gpu_mem <= 24) max_ts = 145;
    else if(gpu_mem <= 32) max_ts = 155;
    else if(gpu_mem <= 40) max_ts = 160;
    else if(gpu_mem <= 48) max_ts = 170;

    while((6 * (std::pow(max_ts, 4) * 8 / (1024 * 1024 * 1024.0)) * ranks_per_gpu) >=
          0.95 * gpu_mem) {
      max_ts -= 5;
    }
  }

  std::vector<int> tilesizes;
  for(int i = ts_guess; i <= max_ts; i += 5) { tilesizes.push_back(i); }

  int ts_guess_ = tilesizes[0];
  int ts_max_   = tilesizes[0];

  for(int ts: tilesizes) {
    int nblocks =
      std::ceil(static_cast<double>(v_alpha) / ts) * std::ceil(static_cast<double>(o_alpha) / ts) *
      std::ceil(static_cast<double>(v_beta) / ts) * std::ceil(static_cast<double>(o_beta) / ts);

    ts_max_ = ts;

    if((nblocks * 1.0 / nranks) < 0.31 || ts_max_ >= v_alpha + 10 || nblocks == 1) {
      ts_max_ = ts_guess_;
      break;
    }

    ts_guess_ = ts;
  }

  return ts_max_;
}

std::tuple<TiledIndexSpace, TAMM_SIZE> setup_mo_red(ExecutionContext& ec, ChemEnv& chem_env,
                                                    bool triples) {
  SystemData& sys_data    = chem_env.sys_data;
  TAMM_SIZE   n_occ_alpha = sys_data.n_occ_alpha;
  TAMM_SIZE   n_vir_alpha = sys_data.n_vir_alpha;

  const int rank         = ec.pg().rank().value();
  auto      ccsd_options = chem_env.ioptions.ccsd_options;

  const std::string jkey = "tilesize";
  bool              user_ts{false};
  if(chem_env.jinput.contains("CC")) user_ts = chem_env.jinput["CC"].contains(jkey) ? true : false;

  Tile tce_tile = ccsd_options.tilesize;
  if(!triples) {
    if(!user_ts && ec.has_gpu()) {
      tce_tile = static_cast<Tile>(sys_data.nbf / 10);
      if(tce_tile < 50) tce_tile = 50;        // 50 is the default tilesize
      else if(tce_tile > 140) tce_tile = 140; // 140 is the max tilesize
      if(rank == 0)
        std::cout << std::endl
                  << "**** Resetting tilesize for the MO space to: " << tce_tile << std::endl;
    }
  }
  else tce_tile = ccsd_options.ccsdt_tilesize;

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

  if(rank == 0 && ccsd_options.debug) { cout << endl << "MO Tiles = " << mo_tiles << endl; }

  return std::make_tuple(MO, total_orbitals);
}

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(ExecutionContext& ec, ChemEnv& chem_env,
                                                 bool triples) {
  const int   rank         = ec.pg().rank().value();
  SystemData& sys_data     = chem_env.sys_data;
  auto        ccsd_options = chem_env.ioptions.ccsd_options;
  auto        task_options = chem_env.ioptions.task_options;

  TAMM_SIZE total_orbitals = sys_data.nmo;
  TAMM_SIZE nocc           = sys_data.nocc;
  TAMM_SIZE n_occ_alpha    = sys_data.n_occ_alpha;
  TAMM_SIZE n_occ_beta     = sys_data.n_occ_beta;
  TAMM_SIZE n_vir_alpha    = sys_data.n_vir_alpha;
  TAMM_SIZE n_vir_beta     = sys_data.n_vir_beta;

  // Active space sizes
  // 'int' (internal) are orbitals in the active space
  // 'ext' (external) are orbitals outside the active space
  // xxx_int + xxx_ext = xxx
  TAMM_SIZE occ_alpha_int = 0;
  TAMM_SIZE occ_beta_int  = 0;
  TAMM_SIZE vir_alpha_int = 0;
  TAMM_SIZE vir_beta_int  = 0;
  if(task_options.ducc.first) {
    occ_alpha_int = ccsd_options.nactive_oa;
    occ_beta_int  = ccsd_options.nactive_ob;
    vir_alpha_int = ccsd_options.nactive_va;
    vir_beta_int  = ccsd_options.nactive_vb;
  }
  else {
    if(ccsd_options.nactive_oa > 0 || ccsd_options.nactive_ob > 0 || ccsd_options.nactive_va > 0 ||
       ccsd_options.nactive_vb > 0) {
      if(rank == 0) std::cout << std::endl << "**** Ignoring nactive options" << std::endl;
    }
  }
  TAMM_SIZE occ_alpha_ext = n_occ_alpha - occ_alpha_int;
  TAMM_SIZE occ_beta_ext  = n_occ_beta - occ_beta_int;
  TAMM_SIZE vir_alpha_ext = n_vir_alpha - vir_alpha_int;
  TAMM_SIZE vir_beta_ext  = n_vir_beta - vir_beta_int;

  Tile tce_tile      = ccsd_options.tilesize;
  bool balance_tiles = ccsd_options.balance_tiles;

  // Check if active space is allowed:
  if(task_options.ducc.first) {
    if(n_occ_alpha != n_occ_beta)
      tamm_terminate("[DUCC ERROR]: DUCC is only for closed-shell calculations");
    if(occ_alpha_int > n_occ_alpha) tamm_terminate("[DUCC ERROR]: nactive_oa > n_occ_alpha");
    if(occ_beta_int > n_occ_beta) tamm_terminate("[DUCC ERROR]: nactive_ob > n_occ_beta");
    if(vir_alpha_int > n_vir_alpha) tamm_terminate("[DUCC ERROR]: nactive_va > n_vir_alpha");
    if(vir_beta_int > n_vir_beta) tamm_terminate("[DUCC ERROR]: nactive_vb > n_vir_beta");
    if(occ_alpha_int == 0 || occ_beta_int == 0)
      tamm_terminate("[DUCC ERROR]: nactive_oa/nactive_ob cannot be 0");
    if(occ_alpha_int != occ_beta_int) tamm_terminate("[DUCC ERROR]: nactive_oa != nactive_ob");
    if(vir_alpha_int != vir_beta_int) tamm_terminate("[DUCC ERROR]: nactive_va != nactive_vb");
  }

  const std::string jkey = "tilesize";
  bool              user_ts{false};
  if(chem_env.jinput.contains("CC")) user_ts = chem_env.jinput["CC"].contains(jkey) ? true : false;

  if(!triples) {
    if(!user_ts && ec.has_gpu()) {
      tce_tile = get_ts_recommendation(ec, chem_env);
      if(rank == 0)
        std::cout << std::endl
                  << "**** Resetting tilesize for the MSO space to: " << tce_tile << std::endl;
    }
    chem_env.is_context.mso_tilesize = tce_tile;
  }
  else {
    balance_tiles                            = false;
    tce_tile                                 = ccsd_options.ccsdt_tilesize;
    chem_env.is_context.mso_tilesize_triples = tce_tile;
  }

  // | occ_alpha | occ_beta | virt_alpha | virt_beta |
  // | occ_alpha_ext | occ_alpha_int | occ_beta_ext | occ_beta_int | --> (cont.)
  //    --> | vir_alpha_int | vir_alpha_ext | vir_beta_int |vir_beta_ext |
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {
      {"occ", {range(0, nocc)}},
      {"occ_alpha", {range(0, n_occ_alpha)}},
      {"occ_beta", {range(n_occ_alpha, nocc)}},
      {"virt", {range(nocc, total_orbitals)}},
      {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
      {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}},
      // Active-space index spaces
      {"occ_alpha_ext", {range(0, occ_alpha_ext)}},
      {"occ_beta_ext", {range(n_occ_alpha, n_occ_alpha + occ_beta_ext)}},
      {"occ_ext", {range(0, occ_alpha_ext), range(n_occ_alpha, n_occ_alpha + occ_beta_ext)}},
      {"occ_alpha_int", {range(occ_alpha_ext, n_occ_alpha)}},
      {"occ_beta_int", {range(n_occ_alpha + occ_beta_ext, nocc)}},
      {"occ_int", {range(occ_alpha_ext, n_occ_alpha), range(n_occ_alpha + occ_beta_ext, nocc)}},
      {"virt_alpha_int", {range(nocc, nocc + vir_alpha_int)}},
      {"virt_beta_int", {range(nocc + n_vir_alpha, nocc + n_vir_alpha + vir_beta_int)}},
      {"virt_int",
       {range(nocc, nocc + vir_alpha_int),
        range(nocc + n_vir_alpha, nocc + n_vir_alpha + vir_beta_int)}},
      {"virt_alpha_ext", {range(nocc + vir_alpha_int, nocc + n_vir_alpha)}},
      {"virt_beta_ext", {range(nocc + n_vir_alpha + vir_beta_int, total_orbitals)}},
      {"virt_ext",
       {range(nocc + vir_alpha_int, nocc + n_vir_alpha),
        range(nocc + n_vir_alpha + vir_beta_int, total_orbitals)}},
      // All spin index spaces
      {"all_alpha", {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
      {"all_beta", {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}},
    },
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  if(!balance_tiles) {
    // tamm::Tile est_nt    = n_occ_alpha / tce_tile;
    // tamm::Tile last_tile = n_occ_alpha % tce_tile;
    // for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    // if(last_tile > 0) mo_tiles.push_back(last_tile);
    tamm::Tile est_nt    = occ_alpha_ext / tce_tile;
    tamm::Tile last_tile = occ_alpha_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = occ_alpha_int / tce_tile;
    last_tile = occ_alpha_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    // est_nt    = n_occ_beta / tce_tile;
    // last_tile = n_occ_beta % tce_tile;
    // for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    // if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = occ_beta_ext / tce_tile;
    last_tile = occ_beta_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = occ_beta_int / tce_tile;
    last_tile = occ_beta_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    // est_nt = n_vir_alpha/tce_tile;
    // last_tile = n_vir_alpha%tce_tile;
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
    // if(last_tile>0) mo_tiles.push_back(last_tile);
    est_nt    = vir_alpha_int / tce_tile;
    last_tile = vir_alpha_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = vir_alpha_ext / tce_tile;
    last_tile = vir_alpha_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    // est_nt = n_vir_beta/tce_tile;
    // last_tile = n_vir_beta%tce_tile;
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
    // if(last_tile>0) mo_tiles.push_back(last_tile);
    est_nt    = vir_beta_int / tce_tile;
    last_tile = vir_beta_int % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
    est_nt    = vir_beta_ext / tce_tile;
    last_tile = vir_beta_ext % tce_tile;
    for(tamm::Tile x = 0; x < est_nt; x++) mo_tiles.push_back(tce_tile);
    if(last_tile > 0) mo_tiles.push_back(last_tile);
  }
  else {
    // tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
    // for(tamm::Tile x = 0; x < est_nt; x++)
    //   mo_tiles.push_back(n_occ_alpha / est_nt + (x < (n_occ_alpha % est_nt)));

    tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * occ_alpha_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(occ_alpha_ext / est_nt + (x < (occ_alpha_ext % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * occ_alpha_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(occ_alpha_int / est_nt + (x < (occ_alpha_int % est_nt)));

    // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
    // for(tamm::Tile x = 0; x < est_nt; x++)
    //   mo_tiles.push_back(n_occ_beta / est_nt + (x < (n_occ_beta % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * occ_beta_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(occ_beta_ext / est_nt + (x < (occ_beta_ext % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * occ_beta_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(occ_beta_int / est_nt + (x < (occ_beta_int % est_nt)));

    // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_alpha / est_nt + (x<(n_vir_alpha %
    // est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * vir_alpha_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(vir_alpha_int / est_nt + (x < (vir_alpha_int % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * vir_alpha_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(vir_alpha_ext / est_nt + (x < (vir_alpha_ext % est_nt)));

    // est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
    // for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_beta / est_nt + (x<(n_vir_beta %
    // est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * vir_beta_int / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(vir_beta_int / est_nt + (x < (vir_beta_int % est_nt)));

    est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * vir_beta_ext / tce_tile));
    for(tamm::Tile x = 0; x < est_nt; x++)
      mo_tiles.push_back(vir_beta_ext / est_nt + (x < (vir_beta_ext % est_nt)));
  }

  TiledIndexSpace MO{MO_IS, mo_tiles}; //{ova,ova,ovb,ovb}};

  if(rank == 0 && ccsd_options.debug) { cout << endl << "MO Tiles = " << mo_tiles << endl; }

  return std::make_tuple(MO, total_orbitals);
}

void update_sysdata(ExecutionContext& ec, ChemEnv& chem_env, TiledIndexSpace& MO, bool is_mso) {
  SystemData& sys_data       = chem_env.sys_data;
  const bool  do_freeze      = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;
  TAMM_SIZE   total_orbitals = sys_data.nmo;
  if(do_freeze) {
    sys_data.nbf -= (sys_data.n_frozen_core + sys_data.n_frozen_virtual);
    sys_data.n_occ_alpha -= sys_data.n_frozen_core;
    sys_data.n_vir_alpha -= sys_data.n_frozen_virtual;
    if(is_mso) {
      sys_data.n_occ_beta -= sys_data.n_frozen_core;
      sys_data.n_vir_beta -= sys_data.n_frozen_virtual;
    }
    sys_data.update();
    if(!is_mso) std::tie(MO, total_orbitals) = setup_mo_red(ec, chem_env);
    else { std::tie(MO, total_orbitals) = cholesky_2e::setupMOIS(ec, chem_env); }
  }
}

// reshape F/lcao after freezing
Matrix reshape_mo_matrix(ChemEnv& chem_env, Matrix& emat, bool is_lcao) {
  SystemData& sys_data = chem_env.sys_data;
  const int   noa      = sys_data.n_occ_alpha;
  const int   nob      = sys_data.n_occ_beta;
  const int   nva      = sys_data.n_vir_alpha;
  const int   nvb      = sys_data.n_vir_beta;
  const int   nocc     = sys_data.nocc;
  const int   N_eff    = sys_data.nmo;
  const int   nbf      = sys_data.nbf_orig;

  const int n_frozen_core    = sys_data.n_frozen_core;
  const int n_frozen_virtual = sys_data.n_frozen_virtual;

  Matrix cvec;
  if(!is_lcao) cvec.resize(N_eff, N_eff); // MOxMO
  else cvec.resize(nbf, N_eff);           // AOxMO

  const int block2_off     = 2 * n_frozen_core + noa;
  const int block3_off     = 2 * n_frozen_core + nocc;
  const int last_block_off = block3_off + n_frozen_virtual + nva;

  if(is_lcao) {
    cvec.block(0, 0, nbf, noa)          = emat.block(0, n_frozen_core, nbf, noa);
    cvec.block(0, noa, nbf, nob)        = emat.block(0, block2_off, nbf, nob);
    cvec.block(0, nocc, nbf, nva)       = emat.block(0, block3_off, nbf, nva);
    cvec.block(0, nocc + nva, nbf, nvb) = emat.block(0, last_block_off, nbf, nvb);
  }
  else {
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
  }

  emat.resize(0, 0);
  return cvec;
}

template<typename TensorType>
void cholesky_2e(ExecutionContext& ec, ChemEnv& chem_env) {
  TiledIndexSpace& tMO = chem_env.is_context.MSO;
  TiledIndexSpace& tAO = chem_env.is_context.AO_opt;

  libint2::BasisSet&  shells = chem_env.shells;
  Tensor<TensorType>& lcao   = chem_env.cd_context.movecs_so;
  bool                is_mso = chem_env.cd_context.is_mso;

  using libint2::Atom;
  using libint2::Engine;
  using libint2::Operator;
  using libint2::Shell;

  const int max_cvecs = chem_env.cd_context.max_cvecs;

  SystemData&      sys_data   = chem_env.sys_data;
  const auto       cd_options = chem_env.ioptions.cd_options;
  const auto       write_cv   = cd_options.write_cv;
  const double     diagtol    = cd_options.diagtol;
  const tamm::Tile itile_size = cd_options.itilesize;
  // const TAMM_GA_SIZE northo      = sys_data.nbf;
  const TAMM_GA_SIZE nao = sys_data.nbf_orig;

  SCFData                scf_data; // init vars
  SCFCompute<TensorType> scf_compute;
  std::tie(scf_data.shell_tile_map, scf_data.AO_tiles, scf_data.AO_opttiles) =
    scf_compute.compute_AO_tiles(ec, chem_env, shells);
  scf_compute.compute_shellpair_list(ec, shells, scf_data);
  auto [obs_shellpair_list, obs_shellpair_data] = scf_compute.compute_shellpairs(shells);

  IndexSpace AO{range(0, shells.nbf())};
  scf_data.tAO    = {AO, scf_data.AO_opttiles};
  Matrix SchwarzK = scf_compute.compute_schwarz_ints(ec, scf_data, shells);

  auto shell2bf = BasisSetMap::map_shell_to_basis_function(shells);
  auto bf2shell = BasisSetMap::map_basis_function_to_shell(shells);

  std::vector<size_t>     shell_tile_map = scf_data.shell_tile_map;
  std::vector<tamm::Tile> AO_tiles       = scf_data.AO_tiles;

  // TiledIndexSpace tAOt{tAO.index_space(), AO_tiles};

  TAMM_GA_SIZE N = tMO("all").max_num_indices();
  IndexSpace   CI{range(0, max_cvecs)};
  // need CI dim to be a single tile for cholesky
  TiledIndexSpace tCI{CI, static_cast<Tile>(max_cvecs)};

  Matrix lcao_eig(nao, N);
  lcao_eig.setZero();
  tamm_to_eigen_tensor(lcao, lcao_eig);

  auto rank = ec.pg().rank().value();

  // Cholesky decomposition
  if(rank == 0) {
    cout << endl << "    Begin Cholesky Decomposition" << endl;
    cout << std::string(45, '-') << endl;
  }

  auto   cd_t1   = std::chrono::high_resolution_clock::now();
  auto   cd_t2   = cd_t1;
  double cd_time = 0;

  const auto nbf   = nao;
  int64_t    count = 0; // Initialize cholesky vector count

  std::string out_fp       = chem_env.workspace_dir;
  const auto  files_dir    = out_fp + chem_env.ioptions.scf_options.scf_type;
  const auto  files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  const auto  chol_ao_file = files_prefix + ".chol_ao";
  const auto  diag_ao_file = files_prefix + ".diag_ao";

  int64_t cd_nranks = /* std::abs(std::log10(diagtol)) */ nbf / 3; // max cores
  auto    nnodes    = ec.nnodes();
  auto    ppn       = ec.ppn();
  int     cd_nnodes = cd_nranks / ppn;
  if(cd_nranks % ppn > 0 || cd_nnodes == 0) cd_nnodes++;
  if(cd_nnodes > nnodes) cd_nnodes = nnodes;
  cd_nranks = cd_nnodes * ppn;
  if(rank == 0) {
    cout << "Total # of mpi ranks used for Cholesky decomposition: " << cd_nranks << endl
         << "  --> Number of nodes, mpi ranks per node: " << cd_nnodes << ", " << ppn << endl;
  }

#if defined(CD_THROTTLE)
  ProcGroup pg_cd = ProcGroup::create_subgroup(ec.pg(), cd_nranks);
#endif

  Tensor<TensorType> g_d_tamm{tAO, tAO};
  Tensor<TensorType> g_r_tamm{tAO, tAO};
  Tensor<TensorType> g_chol_tamm{tAO, tAO, tCI};

  double cd_mem_req       = sum_tensor_sizes(g_d_tamm, g_r_tamm, g_chol_tamm);
  auto   check_cd_mem_req = [&](const std::string mstep) {
    if(ec.print())
      std::cout << "- CPU memory required for " << mstep << ": " << std::fixed
                << std::setprecision(2) << cd_mem_req << " GiB" << std::endl;
    check_memory_requirements(ec, cd_mem_req);
  };
  check_cd_mem_req("computing cholesky vectors");

  g_r_tamm.set_dense();
  g_d_tamm.set_dense();
  g_chol_tamm.set_dense();

  bool cd_throttle = true;
  if(rank >= cd_nranks) cd_throttle = false;

#if defined(CD_THROTTLE)
  if(cd_throttle) {
    ExecutionContext ec_cd{pg_cd, DistributionKind::nw, MemoryManagerKind::ga};
#else
  ExecutionContext& ec_cd = ec;
#endif

    ExecutionContext ec_dense{ec_cd.pg(), DistributionKind::dense, MemoryManagerKind::ga};

    std::vector<int64_t> lo_x(4, -1); // The lower limits of blocks
    std::vector<int64_t> hi_x(4, -2); // The upper limits of blocks
    std::vector<int64_t> ld_x(4);     // The leading dims of blocks

    Tensor<TensorType>::allocate(&ec_dense, g_d_tamm, g_r_tamm, g_chol_tamm);

#if !defined(USE_UPCXX)
    // cd_tensor_zero(g_d_tamm);
    // cd_tensor_zero(g_r_tamm);
    // cd_tensor_zero(g_chol_tamm);

    auto write_chol_vectors = [&]() {
      write_to_disk(g_d_tamm, diag_ao_file);
      write_to_disk(g_chol_tamm, chol_ao_file);
      if(rank == 0) {
        chem_env.run_context["cholesky_2e"]["num_chol_vecs"] = count;
        chem_env.write_run_context();
        cout << endl << "- Number of cholesky vectors written to disk = " << count << endl;
      }
    };

    const int g_chol = g_chol_tamm.ga_handle();
    const int g_d    = g_d_tamm.ga_handle();
    const int g_r    = g_r_tamm.ga_handle();

#if defined(CD_USE_PGAS_API)
    std::vector<int64_t> lo_b(g_chol_tamm.num_modes(), -1); // The lower limits of blocks of B
    std::vector<int64_t> hi_b(g_chol_tamm.num_modes(), -2); // The upper limits of blocks of B
    std::vector<int64_t> ld_b(g_chol_tamm.num_modes());     // The leading dims of blocks of B

    std::vector<int64_t> lo_r(g_r_tamm.num_modes(), -1); // The lower limits of blocks of R
    std::vector<int64_t> hi_r(g_r_tamm.num_modes(), -2); // The upper limits of blocks of R
    std::vector<int64_t> ld_r(g_r_tamm.num_modes());     // The leading dims of blocks of R

    std::vector<int64_t> lo_d(g_d_tamm.num_modes(), -1); // The lower limits of blocks of D
    std::vector<int64_t> hi_d(g_d_tamm.num_modes(), -2); // The upper limits of blocks of D
    std::vector<int64_t> ld_d(g_d_tamm.num_modes());     // The leading dims of blocks of D

    // Distribution Check
    NGA_Distribution64(g_chol, rank, lo_b.data(), hi_b.data());
    NGA_Distribution64(g_d, rank, lo_d.data(), hi_d.data());
    NGA_Distribution64(g_r, rank, lo_r.data(), hi_r.data());

    bool has_gc_data = (lo_b[0] >= 0 && hi_b[0] >= 0);
    bool has_gd_data = (lo_d[0] >= 0 && hi_d[0] >= 0);
    bool has_gr_data = (lo_r[0] >= 0 && hi_r[0] >= 0);
#endif
#endif

    ec_dense.pg().barrier();

    cd_t1 = std::chrono::high_resolution_clock::now();
    /* Compute the diagonal
      g_d_tamm stores the diagonal integrals, i.e. (uv|uv)'s
      ScrCol temporarily stores all (uv|rs)'s with fixed r and s
    */
    Engine       engine(Operator::coulomb, max_nprim(shells), max_l(shells), 0);
    const double engine_precision = chem_env.ioptions.scf_options.tol_int;

    // Compute diagonal without primitive screening
    engine.set_precision(0.0);
    const auto& buf = engine.results();

    bool cd_restart = write_cv.first && fs::exists(diag_ao_file) && fs::exists(chol_ao_file);

    auto compute_diagonals = [&](const IndexVector& blockid) {
      auto bi0 = blockid[0];
      auto bi1 = blockid[1];

      const TAMM_SIZE         size       = g_d_tamm.block_size(blockid);
      auto                    block_dims = g_d_tamm.block_dims(blockid);
      std::vector<TensorType> dbuf(size);

      auto bd1 = block_dims[1];

      size_t s1range_start = 0;
      auto   s1range_end   = shell_tile_map[bi0];
      if(bi0 > 0) s1range_start = shell_tile_map[bi0 - 1] + 1;

      for(auto s1 = s1range_start; s1 <= s1range_end; ++s1) {
        auto n1 = shells[s1].size();

        size_t s2range_start = 0;
        auto   s2range_end   = shell_tile_map[bi1];
        if(bi1 > 0) s2range_start = shell_tile_map[bi1 - 1] + 1;

        for(size_t s2 = s2range_start; s2 <= s2range_end; ++s2) {
          if(s2 > s1) {
            auto s2spl = scf_data.obs_shellpair_list[s2];
            if(std::find(s2spl.begin(), s2spl.end(), s1) == s2spl.end()) continue;
          }
          else {
            auto s2spl = scf_data.obs_shellpair_list[s1];
            if(std::find(s2spl.begin(), s2spl.end(), s2) == s2spl.end()) continue;
          }

          auto n2 = shells[s2].size();

          // compute shell pair; return is the pointer to the buffer
          engine.compute(shells[s1], shells[s2], shells[s1], shells[s2]);
          const auto* buf_1212 = buf[0];
          if(buf_1212 == nullptr) continue;

          auto curshelloffset_i = 0U;
          auto curshelloffset_j = 0U;
          for(auto x = s1range_start; x < s1; x++) curshelloffset_i += AO_tiles[x];
          for(auto x = s2range_start; x < s2; x++) curshelloffset_j += AO_tiles[x];

          auto dimi = curshelloffset_i + AO_tiles[s1];
          auto dimj = curshelloffset_j + AO_tiles[s2];

          for(size_t i = curshelloffset_i; i < dimi; i++) {
            for(size_t j = curshelloffset_j; j < dimj; j++) {
              auto f1           = i - curshelloffset_i;
              auto f2           = j - curshelloffset_j;
              auto f1212        = f1 * n2 * n1 * n2 + f2 * n1 * n2 + f1 * n2 + f2;
              dbuf[i * bd1 + j] = buf_1212[f1212];
            }
          }
        }
      }
      g_d_tamm.put(blockid, dbuf);
    };
    // for(auto blockid: g_d_tamm.loop_nest()) {
    //   if(g_d_tamm.is_local_block(blockid)) {
    //     compute_diagonals(blockid);
    //   }
    // }
    if(!cd_restart) block_for(ec_dense, g_d_tamm(), compute_diagonals);

    cd_t2   = std::chrono::high_resolution_clock::now();
    cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
    if(rank == 0 && !cd_restart) {
      std::cout << endl
                << "- Time for computing the diagonal: " << std::fixed << std::setprecision(2)
                << cd_time << " secs" << endl;
    }

#if !defined(USE_UPCXX)
    if(cd_restart) {
      cd_t1 = std::chrono::high_resolution_clock::now();

      read_from_disk(g_d_tamm, diag_ao_file);
      read_from_disk(g_chol_tamm, chol_ao_file);

      count = chem_env.run_context["cholesky_2e"]["num_chol_vecs"];

      if(rank == 0)
        cout << endl << "- [CD restart] Number of cholesky vectors read = " << count << endl;

      cd_t2   = std::chrono::high_resolution_clock::now();
      cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
      if(rank == 0) {
        std::cout << "- [CD restart] Time for reading the diagonal and cholesky vectors: "
                  << std::fixed << std::setprecision(2) << cd_time << " secs" << endl;
      }
    }
#endif

    auto cd_t3 = std::chrono::high_resolution_clock::now();

    auto [val_d0, blkid, eoff]  = tamm::max_element(g_d_tamm);
    auto                 blkoff = g_d_tamm.block_offsets(blkid);
    std::vector<int64_t> indx_d0(g_d_tamm.num_modes());
    indx_d0[0] = (int64_t) blkoff[0] + (int64_t) eoff[0];
    indx_d0[1] = (int64_t) blkoff[1] + (int64_t) eoff[1];

    // Reset Engine precision
    const double schwarz_tol = chem_env.ioptions.scf_options.tol_sch;
    engine.set_precision(engine_precision);

    std::vector<TensorType> k_row(max_cvecs);

    while(val_d0 > diagtol && count < max_cvecs) {
      auto bfu        = indx_d0[0];
      auto bfv        = indx_d0[1];
      auto s1         = bf2shell[bfu];
      auto s2         = bf2shell[bfv];
      auto n2         = shells[s2].size();
      auto f1         = bfu - shell2bf[s1];
      auto f2         = bfv - shell2bf[s2];
      auto ind12      = f1 * n2 + f2;
      auto schwarz_12 = SchwarzK(s1, s2);

#if !defined(USE_UPCXX)
      cd_tensor_zero(g_r_tamm);
#endif

#if !defined(CD_USE_PGAS_API)
      auto n1          = shells[s1].size();
      auto n12         = n1 * n2;
      auto compute_eri = [&](const IndexVector& blockid) {
        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        const TAMM_SIZE         size       = g_r_tamm.block_size(blockid);
        auto                    block_dims = g_r_tamm.block_dims(blockid);
        std::vector<TensorType> dbuf(size);

        auto bd1 = block_dims[1];

        auto s3range_start = 0l;
        auto s3range_end   = shell_tile_map[bi0];
        if(bi0 > 0) s3range_start = shell_tile_map[bi0 - 1] + 1;

        auto s4range_start = 0l;
        auto s4range_end   = shell_tile_map[bi1];
        if(bi1 > 0) s4range_start = shell_tile_map[bi1 - 1] + 1;

        auto curshelloffset_i = 0U;
        for(Index s3 = s3range_start; s3 <= s3range_end; ++s3) {
          auto n3   = shells[s3].size();
          auto dimi = curshelloffset_i + AO_tiles[s3];

          auto s3spl     = obs_shellpair_list.at(s3);
          auto sp34_iter = s3spl.begin();

          auto curshelloffset_j = 0U;
          for(Index s4 = s4range_start; s4 <= s4range_end; ++s4) {
            if(s4 > s3) {
              auto s4spl = scf_data.obs_shellpair_list[s4];
              if(std::find(s4spl.begin(), s4spl.end(), s3) == s4spl.end()) {
                curshelloffset_j += AO_tiles[s4];
                continue;
              }
            }
            else {
              if(std::find(sp34_iter, s3spl.end(), s4) == s3spl.end()) {
                curshelloffset_j += AO_tiles[s4];
                continue;
              }
            }
            // const auto* sp34 = sp34_iter->get();
            ++sp34_iter;

            if(schwarz_12 * SchwarzK(s3, s4) < schwarz_tol) {
              curshelloffset_j += AO_tiles[s4];
              continue;
            }

            // compute shell pair; return is the pointer to the buffer
            engine.compute(shells[s3], shells[s4], shells[s1], shells[s2]);
            const auto* buf_3412 = buf[0];
            if(buf_3412 == nullptr) {
              curshelloffset_j += AO_tiles[s4];
              continue; // if all integrals screened out, skip to next quartet
            }

            auto n4   = shells[s4].size();
            auto dimj = curshelloffset_j + AO_tiles[s4];

            for(size_t i = curshelloffset_i; i < dimi; i++) {
              for(size_t j = curshelloffset_j; j < dimj; j++) {
                auto f3           = i - curshelloffset_i;
                auto f4           = j - curshelloffset_j;
                auto f3412        = f3 * n4 * n12 + f4 * n12 + ind12;
                auto x            = buf_3412[f3412];
                dbuf[i * bd1 + j] = x;
              }
            }
            curshelloffset_j += AO_tiles[s4];
          }
          curshelloffset_i += AO_tiles[s3];
        }
        g_r_tamm.put(blockid, dbuf);
      };
      // for(auto blockid: g_r_tamm.loop_nest()) {
      //   if(g_r_tamm.is_local_block(blockid))
      //    {  compute_eri(blockid); }
      // }
      block_for(ec_dense, g_r_tamm(), compute_eri);
      ec_dense.pg().barrier();

#else
    for(size_t s3 = 0; s3 != shells.size(); ++s3) {
      auto bf3_first = shell2bf[s3]; // first basis function in this shell
      auto n3        = shells[s3].size();
#if !defined(USE_UPCXX)
      if(!(lo_r[0] <= cd_ncast<size_t>(bf3_first) && cd_ncast<size_t>(bf3_first) <= hi_r[0]))
        continue;
      for(size_t s4 = 0; s4 != shells.size(); ++s4) {
#else
      for(size_t s4 = 0; s4 != shells.size(); ++s4) {
#endif
        auto bf4_first = shell2bf[s4];
        auto n4        = shells[s4].size();
        if(schwarz_12 * SchwarzK(s3, s4) < schwarz_tol) continue;

#if defined(USE_UPCXX)
        if(g_r_tamm.is_local_element(0, 0, bf3_first, bf4_first)) {
          double factor = 1.0;
#else
        if(lo_r[1] <= cd_ncast<size_t>(bf4_first) && cd_ncast<size_t>(bf4_first) <= hi_r[1]) {
#endif
          // Switching shell order allows unit-stride access
          engine.compute(shells[s1], shells[s2], shells[s3], shells[s4]);
          const auto* buf_3412 = buf[0];
          if(buf_3412 == nullptr) continue; // if all integrals screened out, skip to next quartet

          std::vector<TensorType> k_eri(buf_3412 + ind12 * n3 * n4,
                                        buf_3412 + ind12 * n3 * n4 + n3 * n4);

          int64_t ibflo[4] = {0, 0, cd_ncast<size_t>(bf3_first), cd_ncast<size_t>(bf4_first)};
          int64_t ibfhi[4] = {0, 0, cd_ncast<size_t>(bf3_first + n3 - 1),
                              cd_ncast<size_t>(bf4_first + n4 - 1)};

#ifdef USE_UPCXX
          g_r_tamm.put_raw(ibflo, ibfhi, k_eri.data());
#else
          int64_t ld[1] = {cd_ncast<size_t>(n4)};
          NGA_Put64(g_r, &ibflo[2], &ibfhi[2], k_eri.data(), ld);
#endif
        }
      }
    }
    ec_dense.pg().barrier();
#endif

#ifndef USE_UPCXX
      lo_x[0] = indx_d0[0];
      lo_x[1] = indx_d0[1];
      lo_x[2] = 0;
      lo_x[3] = 0;
      hi_x[0] = indx_d0[0];
      hi_x[1] = indx_d0[1];
      hi_x[2] = count;
      hi_x[3] = 0;
      ld_x[0] = 1;
      ld_x[1] = hi_x[2] + 1;
#else
    lo_x[0] = 0;
    lo_x[1] = indx_d0[0];
    lo_x[2] = indx_d0[1];
    lo_x[3] = 0;
    hi_x[0] = 0;
    hi_x[1] = indx_d0[0];
    hi_x[2] = indx_d0[1];
    hi_x[3] = count;
#endif

#if !defined(CD_USE_PGAS_API)
      auto update_diagonals = [&](const IndexVector& blockid) {
        auto bi0 = blockid[0];
        auto bi1 = blockid[1];

        IndexVector             rdblockid  = {bi0, bi1};
        const TAMM_SIZE         size       = g_chol_tamm.block_size(blockid);
        auto                    block_dims = g_chol_tamm.block_dims(blockid);
        const TAMM_SIZE         rdsize     = g_r_tamm.block_size(rdblockid);
        std::vector<TensorType> cbuf(size);
        std::vector<TensorType> rbuf(rdsize);
        std::vector<TensorType> dbuf(rdsize);

        g_r_tamm.get(rdblockid, rbuf);
        g_d_tamm.get(rdblockid, dbuf);
        g_chol_tamm.get(blockid, cbuf);

        auto bd0 = block_dims[1];
        auto bd1 = block_dims[2];

        auto s3range_start = 0l;
        auto s3range_end   = shell_tile_map[bi0];
        if(bi0 > 0) s3range_start = shell_tile_map[bi0 - 1] + 1;

        for(Index s3 = s3range_start; s3 <= s3range_end; ++s3) {
          auto n3 = shells[s3].size();

          auto s4range_start = 0l;
          auto s4range_end   = shell_tile_map[bi1];
          if(bi1 > 0) s4range_start = shell_tile_map[bi1 - 1] + 1;

          for(Index s4 = s4range_start; s4 <= s4range_end; ++s4) {
            if(s4 > s3) {
              auto s2spl = obs_shellpair_list[s4];
              if(std::find(s2spl.begin(), s2spl.end(), s3) == s2spl.end()) continue;
            }
            else {
              auto s2spl = obs_shellpair_list[s3];
              if(std::find(s2spl.begin(), s2spl.end(), s4) == s2spl.end()) continue;
            }

            auto n4 = shells[s4].size();

            auto curshelloffset_i = 0U;
            auto curshelloffset_j = 0U;
            for(auto x = s3range_start; x < s3; x++) curshelloffset_i += AO_tiles[x];
            for(auto x = s4range_start; x < s4; x++) curshelloffset_j += AO_tiles[x];

            auto dimi = curshelloffset_i + AO_tiles[s3];
            auto dimj = curshelloffset_j + AO_tiles[s4];

            for(size_t i = curshelloffset_i; i < dimi; i++) {
              for(size_t j = curshelloffset_j; j < dimj; j++) {
                for(decltype(count) icount = 0; icount < count; icount++) {
                  rbuf[i * bd0 + j] -= cbuf[icount + j * bd1 + i * bd0 * bd1] * k_row[icount];
                }
                auto vtmp                             = rbuf[i * bd0 + j] / sqrt(val_d0);
                cbuf[count + j * bd1 + i * bd1 * bd0] = vtmp;
                dbuf[i * bd0 + j] -= vtmp * vtmp;
              }
            }
          }
        }

        g_r_tamm.put(rdblockid, rbuf);
        g_d_tamm.put(rdblockid, dbuf);
        g_chol_tamm.put(blockid, cbuf);
      };
      // for(auto blockid: g_chol_tamm.loop_nest()) {
      //   if(g_chol_tamm.is_local_block(blockid))
      //    {  update_diagonals(blockid); }
      // }
      block_for(ec_dense, g_chol_tamm(), update_diagonals);

      count++;

#else

#if defined(USE_UPCXX)
    g_chol_tamm.get_raw_contig(lo_x.data(), hi_x.data(), k_row.data());

    auto left  = g_r_tamm.access_local_buf();
    auto right = g_chol_tamm.access_local_buf();
    auto n     = g_r_tamm.local_buf_size();
    for(size_t icount = 0; icount < count; icount++)
      for(size_t i = 0, k = icount; i < n; i++, k += max_cvecs)
        *(left + i) -= *(right + k) * k_row[icount];

    for(size_t i = 0, k = count; i < n; i++, k += max_cvecs) {
      auto tmp     = *(left + i) / sqrt(val_d0);
      *(right + k) = tmp;
    }

    left = g_d_tamm.access_local_buf();
    n    = g_d_tamm.local_buf_size();
    for(size_t i = 0, k = count; i < n; i++, k += max_cvecs) {
      auto tmp = *(right + k);
      *(left + i) -= tmp * tmp;
    }

    count++;
#else
    TensorType *indx_b, *indx_d, *indx_r;

    {
      NGA_Get64(g_chol, lo_x.data(), hi_x.data(), k_row.data(), ld_x.data());

      if(has_gr_data) NGA_Access64(g_r, lo_r.data(), hi_r.data(), &indx_r, ld_r.data());
      if(has_gc_data) NGA_Access64(g_chol, lo_b.data(), hi_b.data(), &indx_b, ld_b.data());
      if(has_gd_data) NGA_Access64(g_d, lo_d.data(), hi_d.data(), &indx_d, ld_d.data());

      if(has_gr_data) {
        int64_t nj = hi_r[1] - lo_r[1] + 1;
        // Implemented as a safeguard and for performance?
        if(nj != ld_r[0] || count < 50) {
          for(int64_t i = 0; i <= hi_r[0] - lo_r[0]; i++) {
            int64_t ild = i * ld_r[0];
            int64_t ildld = i * ld_b[1] * ld_b[0];
            for(int64_t j = 0; j <= hi_r[1] - lo_r[1]; j++) {
              int64_t nij = ild + j;
              for(decltype(count) icount = 0; icount < count; icount++) {
                indx_r[nij] -= indx_b[icount + j * ld_b[1] + ildld] * k_row[icount];
              }
            }
          }
        }
        else {
          int64_t nij = ld_b[0] * (hi_r[0] - lo_r[0] + 1);
          blas::gemv(blas::Layout::RowMajor, blas::Op::NoTrans, nij, (int64_t) count, -1.0, indx_b,
                     (int64_t) ld_b[1], k_row.data(), 1, 1.0, indx_r, 1);
        }
      }

      if(has_gc_data) {
        double value = 1.0 / sqrt(val_d0);
        int64_t nij = ld_r[0] * (hi_r[0] - lo_r[0] + 1);
        blas::scal(nij, value, indx_r, 1);
        blas::copy(nij, indx_r, 1, indx_b + count, ld_b[1]);
        for(auto ij = 0; ij < nij; ij++) { indx_d[ij] -= indx_r[ij] * indx_r[ij]; }
      }

      if(has_gc_data) NGA_Release_update64(g_chol, lo_b.data(), hi_b.data());
      if(has_gd_data) NGA_Release_update64(g_d, lo_d.data(), hi_d.data());
      if(has_gr_data) NGA_Release_update64(g_r, lo_r.data(), hi_r.data());
    }
    count++;
#endif
#endif

      std::tie(val_d0, blkid, eoff) = tamm::max_element(g_d_tamm);
      blkoff                        = g_d_tamm.block_offsets(blkid);
      indx_d0[0]                    = (int64_t) blkoff[0] + (int64_t) eoff[0];
      indx_d0[1]                    = (int64_t) blkoff[1] + (int64_t) eoff[1];

#if !defined(USE_UPCXX)
      // Restart
      if(write_cv.first && count % write_cv.second == 0 && nbf > 1000) { write_chol_vectors(); }
#endif

    } // while

    if(rank == 0)
      std::cout << endl << "- Total number of cholesky vectors = " << count << std::endl;

#if !defined(USE_UPCXX)
    if(write_cv.first && nbf > 1000) write_chol_vectors();
#endif

    write_to_disk(g_chol_tamm, chol_ao_file);
    Tensor<TensorType>::deallocate(g_d_tamm, g_r_tamm, g_chol_tamm);

    auto cd_t4 = std::chrono::high_resolution_clock::now();
    cd_time    = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t4 - cd_t3)).count();
    if(rank == 0) {
      std::cout << endl
                << "- Time to compute cholesky vectors: " << std::fixed << std::setprecision(2)
                << cd_time << " secs" << endl
                << endl;
    }

#if defined(CD_THROTTLE)
    ec_cd.flush_and_sync();
    ec_dense.flush_and_sync();
    ec_cd.pg().destroy_coll();
  } // if(cd_throttle)

  ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
#endif

  ec.pg().barrier();
  ec.pg().broadcast(&count, 0);

  Tensor<TensorType> g_chol_tamm1{tAO, tAO, tCI};
  g_chol_tamm1.set_dense();
  Tensor<TensorType>::allocate(&ec_dense, g_chol_tamm1);
  read_from_disk(g_chol_tamm1, chol_ao_file);
  fs::remove(chol_ao_file);

  update_sysdata(ec, chem_env, tMO, is_mso);

  const bool do_freeze = (sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0);

  IndexSpace         CIp{range(0, count)};
  TiledIndexSpace    tCIp{CIp, static_cast<tamm::Tile>(itile_size)};
  Tensor<TensorType> g_chol_ao_tamm{tAO, tAO, tCIp};

  cd_mem_req -= sum_tensor_sizes(g_d_tamm, g_r_tamm);
  cd_mem_req += sum_tensor_sizes(g_chol_ao_tamm);
  check_cd_mem_req("resizing the ao cholesky tensor");

  Tensor<TensorType>::allocate(&ec, g_chol_ao_tamm);

  // Convert g_chol_tamm(nD with max_cvecs) to g_chol_ao_tamm(1D with chol_count)
  auto lambdacv = [&](const IndexVector& bid) {
    const IndexVector blockid = internal::translate_blockid(bid, g_chol_ao_tamm());

    auto block_dims   = g_chol_ao_tamm.block_dims(blockid);
    auto block_offset = g_chol_ao_tamm.block_offsets(blockid);

    const tamm::TAMM_SIZE dsize = g_chol_ao_tamm.block_size(blockid);

    int64_t lo[4] = {0, cd_ncast<size_t>(block_offset[0]), cd_ncast<size_t>(block_offset[1]),
                     cd_ncast<size_t>(block_offset[2])};
    int64_t hi[4] = {0, cd_ncast<size_t>(block_offset[0] + block_dims[0] - 1),
                     cd_ncast<size_t>(block_offset[1] + block_dims[1] - 1),
                     cd_ncast<size_t>(block_offset[2] + block_dims[2] - 1)};

    std::vector<TensorType> sbuf(dsize);
#ifdef USE_UPCXX
    g_chol_tamm1.get_raw(lo, hi, sbuf.data());
#else
    int64_t   ld[2]  = {cd_ncast<size_t>(block_dims[1]), cd_ncast<size_t>(block_dims[2])};
    const int g_chol = g_chol_tamm1.ga_handle();
    NGA_Get64(g_chol, &lo[1], &hi[1], sbuf.data(), ld);
#endif

    g_chol_ao_tamm.put(blockid, sbuf);
  };

  block_for(ec, g_chol_ao_tamm(), lambdacv);

  Scheduler sch{ec};

  if(do_freeze) {
    Matrix lcao_new;
    if(rank == 0) lcao_new = reshape_mo_matrix(chem_env, lcao_eig, true);
    sch.deallocate(lcao).execute();
    lcao = Tensor<TensorType>{tAO, tMO};
    sch.allocate(lcao).execute();
    if(rank == 0) eigen_to_tamm_tensor(lcao, lcao_new);
  }

  Tensor<TensorType>::deallocate(g_chol_tamm1);
  ec_dense.flush_and_sync();

  Tensor<TensorType> CholVpr_tmp{tMO, tAO, tCIp};
  Tensor<TensorType> CholVpr_tamm{{tMO, tMO, tCIp},
                                  {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
  Tensor<TensorType>::allocate(&ec, CholVpr_tmp);

  cd_mem_req -= sum_tensor_sizes(g_chol_tamm);
  cd_mem_req += sum_tensor_sizes(CholVpr_tmp);
  check_cd_mem_req("ao2mo transformation");

  auto [mu, nu]   = tAO.labels<2>("all");
  auto [pmo, rmo] = tMO.labels<2>("all");
  auto cindexp    = tCIp.label("all");

  cd_t1 = std::chrono::high_resolution_clock::now();

  // clang-format off
  // Contraction 1
  sch(CholVpr_tmp(pmo, mu, cindexp) = lcao(nu, pmo) * g_chol_ao_tamm(nu, mu, cindexp))
  .deallocate(g_chol_ao_tamm).execute(ec.exhw());

  cd_mem_req -= sum_tensor_sizes(g_chol_ao_tamm);
  cd_mem_req += sum_tensor_sizes(CholVpr_tamm);
  check_cd_mem_req("the 2-step contraction");

  // Contraction 2
  sch.allocate(CholVpr_tamm)
  (CholVpr_tamm(pmo, rmo, cindexp) = lcao(mu, rmo) * CholVpr_tmp(pmo, mu, cindexp))
  .deallocate(CholVpr_tmp)
  .execute(ec.exhw());
  // clang-format on

  cd_t2   = std::chrono::high_resolution_clock::now();
  cd_time = std::chrono::duration_cast<std::chrono::duration<double>>((cd_t2 - cd_t1)).count();
  if(rank == 0) {
    std::cout << endl
              << "- Time for ao to mo transform: " << std::fixed << std::setprecision(2) << cd_time
              << " secs" << endl;
  }

  if(rank == 0) {
    cout << endl << "   End Cholesky Decomposition" << endl;
    cout << std::string(45, '-') << endl;
  }

  chem_env.cd_context.num_chol_vecs = count;
  chem_env.cd_context.cholV2        = CholVpr_tamm;
}

template void cholesky_2e<double>(ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::cholesky_2e