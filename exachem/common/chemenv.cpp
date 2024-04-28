#include "chemenv.hpp"

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

void ChemEnv::write_sinfo() {
  std::string basis = ioptions.scf_options.basis;

  std::string out_fp       = workspace_dir;
  std::string files_dir    = out_fp + ioptions.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + sys_data.output_file_prefix;
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);

  json results;

  results["molecule"]["name"]         = sys_data.input_molecule;
  results["molecule"]["basis"]["all"] = basis;

  for(size_t i = 0; i < atoms.size(); i++) {
    ECAtom& iatom = ec_atoms[i];
    if(iatom.basis != basis) results["molecule"]["basis"][iatom.esymbol] = iatom.basis;
  }

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

  results["input"]["molecule"]["name"]     = sys_data.input_molecule;
  results["input"]["molecule"]["basisset"] = scf.basis;
  // results["input"]["molecule"]["gaussian_type"]  = scf.gaussian_type;
  results["input"]["molecule"]["geometry_units"] = scf.geom_units;
  // SCF options
  results["input"]["SCF"]["tol_int"]        = scf.tol_int;
  results["input"]["SCF"]["tol_sch"]        = scf.tol_sch;
  results["input"]["SCF"]["tol_lindep"]     = scf.tol_lindep;
  results["input"]["SCF"]["conve"]          = scf.conve;
  results["input"]["SCF"]["convd"]          = scf.convd;
  results["input"]["SCF"]["diis_hist"]      = scf.diis_hist;
  results["input"]["SCF"]["AO_tilesize"]    = scf.AO_tilesize;
  results["input"]["SCF"]["force_tilesize"] = str_bool(scf.force_tilesize);
  results["input"]["SCF"]["scf_type"]       = scf.scf_type;
  results["input"]["SCF"]["multiplicity"]   = scf.multiplicity;
  results["input"]["SCF"]["lambdas"]        = scf.qed_lambdas;
  results["input"]["SCF"]["polvecs"]        = scf.qed_polvecs;
  results["input"]["SCF"]["omegas"]         = scf.qed_omegas;
  results["input"]["SCF"]["volumes"]        = scf.qed_volumes;

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
    results["input"][cmodule]["ndiis"]         = ccsd.ndiis;
    results["input"][cmodule]["readt"]         = str_bool(ccsd.readt);
    results["input"][cmodule]["writet"]        = str_bool(ccsd.writet);
    results["input"][cmodule]["ccsd_maxiter"]  = ccsd.ccsd_maxiter;
    results["input"][cmodule]["balance_tiles"] = str_bool(ccsd.balance_tiles);
  }

  if(cmodule == "CCSD(T)" || cmodule == "CCSD_T") {
    // CCSD(T) options
    results["input"][cmodule]["skip_ccsd"]      = ccsd.skip_ccsd;
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

  std::string l_module = cmodule;
  txt_utils::to_lower(l_module);

  std::string files_dir =
    sys_data.output_file_prefix + "_files/" + ioptions.scf_options.scf_type + "/json";
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

  libint2::initialize(false);

  std::string basis_set_file = std::string(DATADIR) + "/basis/" + basis + ".g94";

  int basis_file_exists = 0;
  if(rank == 0) basis_file_exists = std::filesystem::exists(basis_set_file);
  ec.pg().broadcast(&basis_file_exists, 0);

  if(!basis_file_exists)
    tamm_terminate("ERROR: basis set file " + basis_set_file + " does not exist");

  libint2::BasisSet shells;
  {
    std::vector<libint2::Shell> bset_vec;
    for(size_t i = 0; i < atoms.size(); i++) {
      // const auto        Z = atoms[i].atomic_number;
      libint2::BasisSet ashells(ec_atoms[i].basis, {atoms[i]});
      bset_vec.insert(bset_vec.end(), ashells.begin(), ashells.end());
    }
    libint2::BasisSet bset(bset_vec);
    shells = std::move(bset);
  }

  shells.set_pure(true);

  const int N = shells.nbf();

  auto nelectrons = 0;
  for(size_t i = 0; i < atoms.size(); ++i) nelectrons += atoms[i].atomic_number;
  nelectrons -= charge;

  // sys_data.nelectrons = nelectrons;
  EXPECTS((nelectrons + scf_options.multiplicity - 1) % 2 == 0);

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
