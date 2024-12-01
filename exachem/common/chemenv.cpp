/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/chemenv.hpp"

int ChemEnv::get_nfcore() {
  if(ioptions.ccsd_options.freeze_atomic) {
    int nfcore = 0;
    for(size_t i = 0; i < atoms.size(); ++i) {
      const auto Z = atoms[i].atomic_number;
      if(Z >= 3 && Z <= 10) nfcore += 1;
      else if(Z >= 11 && Z <= 18) nfcore += 5;
      else if(Z >= 19 && Z <= 36) nfcore += 9;
      else if(Z >= 37 && Z <= 54) nfcore += 18;
      else if(Z >= 55 && Z <= 86) nfcore += 27;
      else if(Z >= 87 && Z <= 118) nfcore += 43;
    }
    return nfcore;
  }

  return ioptions.ccsd_options.freeze_core;
}

void ChemEnv::read_run_context() {
  std::string files_prefix = get_files_prefix();
  std::string json_file    = files_prefix + ".runcontext.json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    std::ifstream jread(json_file);
    jread >> run_context;
  }
  // else {
  //   std::string err_msg = "\n[ERROR] " + json_file + " does not exist!"
  //   tamm_terminate(err_msg);
  // }
}

void ChemEnv::write_run_context() {
  std::string files_dir = get_files_dir();
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string files_prefix = get_files_prefix();
  std::string json_file    = files_prefix + ".runcontext.json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    // std::ifstream jread(json_file);
    // jread >> run_context;
    std::filesystem::remove(json_file);
  }

  // std::cout << std::endl << std::endl << run_context.dump() << std::endl;
  std::ofstream res_file(json_file);
  res_file << std::setw(2) << run_context << std::endl;
}

void ChemEnv::write_sinfo() {
  std::string basis = ioptions.scf_options.basis;

  std::string out_fp       = workspace_dir;
  std::string files_dir    = out_fp + ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  json results;

  results["molecule"]["name"]  = sys_data.input_molecule;
  results["molecule"]["basis"] = jinput["basis"];

  // for(size_t i = 0; i < atoms.size(); i++) {
  //   ECAtom& iatom = ec_atoms[i];
  //   if(iatom.basis != basis) results["molecule"]["basis"][iatom.esymbol] = iatom.basis;
  // }

  results["molecule"]["nbf"]              = shells.nbf();
  results["molecule"]["nshells"]          = shells.size();
  results["molecule"]["nelectrons"]       = sys_data.nelectrons;
  results["molecule"]["nelectrons_alpha"] = sys_data.nelectrons_alpha;
  results["molecule"]["nelectrons_beta"]  = sys_data.nelectrons_beta;

  std::string json_file   = files_prefix + ".sinfo.json";
  bool        json_exists = std::filesystem::exists(json_file);
  if(json_exists) std::filesystem::remove(json_file);

  std::ofstream res_file(json_file);
  res_file << std::setw(2) << results << std::endl;
}

void ChemEnv::write_json_data(const std::string cmodule) {
  auto  scf     = ioptions.scf_options;
  auto  cd      = ioptions.cd_options;
  auto  ccsd    = ioptions.ccsd_options;
  json& results = sys_data.results;

  auto str_bool = [=](const bool val) {
    if(val) return "true";
    return "false";
  };

  results["input"]["molecule"]["name"] = sys_data.input_molecule;
  results["input"]["geometry"]         = jinput["geometry"];
  results["input"]["basis"]            = jinput["basis"];
  results["input"]["common"]           = jinput["common"];

  // SCF options
  results["input"]["SCF"]["charge"]       = scf.charge;
  results["input"]["SCF"]["multiplicity"] = scf.multiplicity;
  results["input"]["SCF"]["lshift"]       = scf.lshift;
  results["input"]["SCF"]["tol_int"]      = scf.tol_int;
  results["input"]["SCF"]["tol_sch"]      = scf.tol_sch;
  results["input"]["SCF"]["tol_lindep"]   = scf.tol_lindep;
  results["input"]["SCF"]["conve"]        = scf.conve;
  results["input"]["SCF"]["convd"]        = scf.convd;
  results["input"]["SCF"]["diis_hist"]    = scf.diis_hist;
  results["input"]["SCF"]["AO_tilesize"]  = scf.AO_tilesize;
  results["input"]["SCF"]["damp"]         = scf.damp;
  results["input"]["SCF"]["debug"]        = str_bool(scf.debug);
  results["input"]["SCF"]["restart"]      = str_bool(scf.restart);
  results["input"]["SCF"]["noscf"]        = str_bool(scf.noscf);
  results["input"]["SCF"]["scf_type"]     = scf.scf_type;
  results["input"]["SCF"]["direct_df"]    = str_bool(scf.direct_df);
  if(!scf.dfbasis.empty()) results["input"]["SCF"]["dfAO_tilesize"] = scf.dfAO_tilesize;

  if(!scf.xc_type.empty() || scf.snK) {
    results["input"]["SCF"]["DFT"]["snK"]               = str_bool(scf.snK);
    results["input"]["SCF"]["DFT"]["xc_type"]           = scf.xc_type;
    results["input"]["SCF"]["DFT"]["xc_grid_type"]      = scf.xc_grid_type;
    results["input"]["SCF"]["DFT"]["xc_pruning_scheme"] = scf.xc_pruning_scheme;
    results["input"]["SCF"]["DFT"]["xc_rad_quad"]       = scf.xc_rad_quad;
    results["input"]["SCF"]["DFT"]["xc_weight_scheme"]  = scf.xc_weight_scheme;
    results["input"]["SCF"]["DFT"]["xc_exec_space"]     = scf.xc_exec_space;
    results["input"]["SCF"]["DFT"]["xc_basis_tol"]      = scf.xc_basis_tol;
    results["input"]["SCF"]["DFT"]["xc_batch_size"]     = scf.xc_batch_size;
    results["input"]["SCF"]["DFT"]["xc_snK_etol"]       = scf.xc_snK_etol;
    results["input"]["SCF"]["DFT"]["xc_snK_ktol"]       = scf.xc_snK_ktol;
    results["input"]["SCF"]["DFT"]["xc_lb_kernel"]      = scf.xc_lb_kernel;
    results["input"]["SCF"]["DFT"]["xc_mw_kernel"]      = scf.xc_mw_kernel;
    results["input"]["SCF"]["DFT"]["xc_int_kernel"]     = scf.xc_int_kernel;
    results["input"]["SCF"]["DFT"]["xc_red_kernel"]     = scf.xc_red_kernel;
    results["input"]["SCF"]["DFT"]["xc_lwd_kernel"]     = scf.xc_lwd_kernel;
    results["input"]["SCF"]["DFT"]["xc_radang_size"]    = scf.xc_radang_size;
  }

  if(scf.scalapack_np_row > 0 && scf.scalapack_np_col > 0) {
    results["input"]["SCF"]["scalapack_nb"]     = scf.scalapack_nb;
    results["input"]["SCF"]["scalapack_np_row"] = scf.scalapack_np_row;
    results["input"]["SCF"]["scalapack_np_col"] = scf.scalapack_np_col;
  }

  // QED
  results["input"]["SCF"]["lambdas"] = scf.qed_lambdas;
  results["input"]["SCF"]["polvecs"] = scf.qed_polvecs;
  results["input"]["SCF"]["omegas"]  = scf.qed_omegas;
  results["input"]["SCF"]["volumes"] = scf.qed_volumes;

  if(cmodule == "CD" || cmodule == "CCSD") {
    // CD options
    results["input"]["CD"]["diagtol"]          = cd.diagtol;
    results["input"]["CD"]["itilesize"]        = cd.itilesize;
    results["input"]["CD"]["skip_cd"]          = cd.skip_cd;
    results["input"]["CD"]["write_cv"]         = cd.write_cv;
    results["input"]["CD"]["max_cvecs_factor"] = cd.max_cvecs_factor;
  }

  results["input"]["CCSD"]["threshold"] = ccsd.threshold;

  if(cmodule == "CCSD") {
    // CCSD options
    results["input"][cmodule]["tilesize"]      = ccsd.tilesize;
    results["input"][cmodule]["lshift"]        = ccsd.lshift;
    results["input"][cmodule]["ndiis"]         = ccsd.ndiis;
    results["input"][cmodule]["readt"]         = str_bool(ccsd.readt);
    results["input"][cmodule]["writet"]        = str_bool(ccsd.writet);
    results["input"][cmodule]["writet_iter"]   = ccsd.writet_iter;
    results["input"][cmodule]["ccsd_maxiter"]  = ccsd.ccsd_maxiter;
    results["input"][cmodule]["nactive"]       = ccsd.nactive;
    results["input"][cmodule]["debug"]         = ccsd.debug;
    results["input"][cmodule]["profile_ccsd"]  = ccsd.profile_ccsd;
    results["input"][cmodule]["balance_tiles"] = str_bool(ccsd.balance_tiles);

    results["input"][cmodule]["freeze"]["atomic"]  = ccsd.freeze_atomic;
    results["input"][cmodule]["freeze"]["core"]    = ccsd.freeze_core;
    results["input"][cmodule]["freeze"]["virtual"] = ccsd.freeze_virtual;
  }

  if(cmodule == "CCSD(T)" || cmodule == "CCSD_T") {
    // CCSD(T) options
    results["input"][cmodule]["skip_ccsd"]      = ccsd.skip_ccsd;
    results["input"][cmodule]["cache_size"]     = ccsd.cache_size;
    results["input"][cmodule]["ccsdt_tilesize"] = ccsd.ccsdt_tilesize;
  }

  if(cmodule == "DLPNO-CCSD") {
    // DLPNO-CCSD options
    results["input"][cmodule]["localize"]         = str_bool(ccsd.localize);
    results["input"][cmodule]["skip_dlpno"]       = str_bool(ccsd.skip_dlpno);
    results["input"][cmodule]["max_pnos"]         = ccsd.max_pnos;
    results["input"][cmodule]["keep_npairs"]      = ccsd.keep_npairs;
    results["input"][cmodule]["TCutEN"]           = ccsd.TCutEN;
    results["input"][cmodule]["TCutPNO"]          = ccsd.TCutPNO;
    results["input"][cmodule]["TCutPre"]          = ccsd.TCutPre;
    results["input"][cmodule]["TCutPairs"]        = ccsd.TCutPairs;
    results["input"][cmodule]["TCutDO"]           = ccsd.TCutDO;
    results["input"][cmodule]["TCutDOij"]         = ccsd.TCutDOij;
    results["input"][cmodule]["TCutDOPre"]        = ccsd.TCutDOPre;
    results["input"][cmodule]["dlpno_dfbasis"]    = ccsd.dlpno_dfbasis;
    results["input"][cmodule]["doubles_opt_eqns"] = ccsd.doubles_opt_eqns;
  }

  if(cmodule == "DUCC") {
    // DUCC options
    results["input"]["DUCC"]["nactive"] = ccsd.nactive;
  }

  if(cmodule == "EOMCCSD") {
    // EOMCCSD options
    results["input"][cmodule]["eom_type"]      = ccsd.eom_type;
    results["input"][cmodule]["eom_nroots"]    = ccsd.eom_nroots;
    results["input"][cmodule]["eom_microiter"] = ccsd.eom_microiter;
    results["input"][cmodule]["eom_threshold"] = ccsd.eom_threshold;
  }

  if(cmodule == "RT-EOMCCS" || cmodule == "RT-EOMCCSD") {
    // RT-EOMCC options
    results["input"]["RT-EOMCC"]["pcore"]         = ccsd.pcore;
    results["input"]["RT-EOMCC"]["ntimesteps"]    = ccsd.ntimesteps;
    results["input"]["RT-EOMCC"]["rt_microiter"]  = ccsd.rt_microiter;
    results["input"]["RT-EOMCC"]["rt_step_size"]  = ccsd.rt_step_size;
    results["input"]["RT-EOMCC"]["rt_multiplier"] = ccsd.rt_multiplier;
  }

  if(cmodule == "GFCCSD") {
    // GFCCSD options
    results["input"][cmodule]["gf_ngmres"]            = ccsd.gf_ngmres;
    results["input"][cmodule]["gf_maxiter"]           = ccsd.gf_maxiter;
    results["input"][cmodule]["gf_threshold"]         = ccsd.gf_threshold;
    results["input"][cmodule]["gf_nprocs_poi"]        = ccsd.gf_nprocs_poi;
    results["input"][cmodule]["gf_damping_factor"]    = ccsd.gf_damping_factor;
    results["input"][cmodule]["gf_omega_min_ip"]      = ccsd.gf_omega_min_ip;
    results["input"][cmodule]["gf_omega_max_ip"]      = ccsd.gf_omega_max_ip;
    results["input"][cmodule]["gf_omega_min_ip_e"]    = ccsd.gf_omega_min_ip_e;
    results["input"][cmodule]["gf_omega_max_ip_e"]    = ccsd.gf_omega_max_ip_e;
    results["input"][cmodule]["gf_omega_delta"]       = ccsd.gf_omega_delta;
    results["input"][cmodule]["gf_omega_delta_e"]     = ccsd.gf_omega_delta_e;
    results["input"][cmodule]["gf_extrapolate_level"] = ccsd.gf_extrapolate_level;
  }

  results["input"]["task"][task_string] = "true";
  results["input"]["task"]["operation"] = ioptions.task_options.operation;

  std::string l_module = cmodule;
  txt_utils::to_lower(l_module);

  std::string files_dir = workspace_dir + ioptions.scf_options.scf_type + "/json";
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string files_prefix = files_dir + "/" + sys_data.output_file_prefix;
  std::string json_file    = files_prefix + "." + l_module + ".json";
  bool        json_exists  = std::filesystem::exists(json_file);
  if(json_exists) {
    // std::ifstream jread(json_file);
    // jread >> results;
    std::filesystem::remove(json_file);
  }

  // std::cout << std::endl << std::endl << results.dump() << std::endl;
  std::ofstream res_file(json_file);
  res_file << std::setw(2) << results << std::endl;
}

void ChemEnv::sinfo() {
  SCFOptions& scf_options = ioptions.scf_options;

  // TODO: dont create ec
  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank   = ec.pg().rank();
  std::string      basis  = scf_options.basis;
  int              charge = scf_options.charge;

  const int N = shells.nbf();

  auto nelectrons = 0;
  for(size_t i = 0; i < atoms.size(); ++i) nelectrons += atoms[i].atomic_number;
  nelectrons -= charge;

  // sys_data.nelectrons = nelectrons;
  if((nelectrons + scf_options.multiplicity - 1) % 2 != 0) {
    std::string err_msg = "[ERROR] Number of electrons (" + std::to_string(nelectrons) + ") " +
                          "and multiplicity (" + std::to_string(scf_options.multiplicity) + ") " +
                          " not compatible!";
    tamm_terminate(err_msg);
  }

  int nelectrons_alpha = (nelectrons + scf_options.multiplicity - 1) / 2;
  int nelectrons_beta  = nelectrons - nelectrons_alpha;

  if(rank == 0) {
    std::cout << std::endl << "Number of basis functions = " << N << std::endl;
    std::cout << std::endl << "Total number of shells = " << shells.size() << std::endl;
    std::cout << std::endl << "Total number of electrons = " << nelectrons << std::endl;
    std::cout << "  # of alpha electrons    = " << nelectrons_alpha << std::endl;
    std::cout << "  # of beta electons      = " << nelectrons_beta << std::endl << std::flush;
  }

  if(rank == 0) write_sinfo();

  ec.flush_and_sync();
}

Matrix ChemEnv::compute_shellblock_norm(const libint2::BasisSet& obs, const Matrix& A) {
  const auto nsh = obs.size();
  Matrix     Ash = Matrix::Zero(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for(size_t s1 = 0; s1 != nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size  = obs[s1].size();
    for(size_t s2 = 0; s2 != nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size  = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size).lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}
