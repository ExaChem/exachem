/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "json_data.hpp"

void SystemData::update(bool spin_orbital) {
  EXPECTS(nbf == n_occ_alpha + n_vir_alpha); // lin-deps
  // EXPECTS(nbf_orig == n_occ_alpha + n_vir_alpha + n_lindep + n_frozen_core + n_frozen_virtual);
  nocc = n_occ_alpha + n_occ_beta;
  nvir = n_vir_alpha + n_vir_beta;
  // EXPECTS(nelectrons == n_occ_alpha + n_occ_beta);
  // EXPECTS(nelectrons == nelectrons_alpha+nelectrons_beta);
  if(spin_orbital) nmo = n_occ_alpha + n_vir_alpha + n_occ_beta + n_vir_beta; // lin-deps
  else nmo = n_occ_alpha + n_vir_alpha;
}

void SystemData::print() {
  std::cout << std::endl << "----------------------------" << std::endl;
  std::cout << "scf_type = " << scf_type_string << std::endl;
  if(is_restricted) std::cout << "Closed-Shell SCF" << std::endl;
  if(is_unrestricted) std::cout << "Open-Shell SCF" << std::endl;
  if(is_restricted_os) std::cout << "Restricted Open-Shell SCF" << std::endl;
  if(is_ks) std::cout << "KS-DFT Enabled" << std::endl;
  if(is_qed && do_qed && is_ks) { std::cout << "QED-KS Enabled" << std::endl; }
  else if(is_qed && is_ks) { std::cout << "QEDFT Enabled" << std::endl; }
  else if(is_qed) { std::cout << "QED-HF Enabled" << std::endl; }

  std::cout << "nbf = " << nbf << std::endl;
  std::cout << "nbf_orig = " << nbf_orig << std::endl;
  std::cout << "n_lindep = " << n_lindep << std::endl;

  std::cout << "focc = " << focc << std::endl;
  std::cout << "nmo = " << nmo << std::endl;
  std::cout << "nocc = " << nocc << std::endl;
  if(nact) std::cout << "nact = " << nact << std::endl;
  std::cout << "nvir = " << nvir << std::endl;

  std::cout << "n_occ_alpha = " << n_occ_alpha << std::endl;
  std::cout << "n_vir_alpha = " << n_vir_alpha << std::endl;
  std::cout << "n_occ_beta = " << n_occ_beta << std::endl;
  std::cout << "n_vir_beta = " << n_vir_beta << std::endl;

  std::cout << "nelectrons = " << nelectrons << std::endl;
  if(nelectrons_active) std::cout << "nelectrons_active = " << nelectrons_active << std::endl;
  std::cout << "nelectrons_alpha = " << nelectrons_alpha << std::endl;
  std::cout << "nelectrons_beta = " << nelectrons_beta << std::endl;
  std::cout << "n_frozen_core = " << n_frozen_core << std::endl;
  std::cout << "n_frozen_virtual = " << n_frozen_virtual << std::endl;
  std::cout << "----------------------------" << std::endl;
}

SystemData::SystemData(OptionsMap options_map_, const std::string scf_type_string):
  options_map(options_map_), scf_type_string(scf_type_string) {
  results          = json::object();
  is_restricted    = false;
  is_unrestricted  = false;
  is_restricted_os = false;
  is_ks            = false;
  is_qed           = false;
  do_qed           = false;
  if(scf_type_string == "restricted") {
    focc          = 1;
    is_restricted = true;
  }
  else if(scf_type_string == "unrestricted") {
    focc            = 2;
    is_unrestricted = true;
  }
  else if(scf_type_string == "restricted_os") {
    focc             = -1;
    is_restricted_os = true;
  }
  else tamm_terminate("ERROR: unrecognized scf_type [" + scf_type_string + "] provided");
  if(!options_map_.scf_options.xc_type.empty()) { is_ks = true; }

  if(!options_map_.scf_options.qed_volumes.empty()) {
    for(auto x: options_map_.scf_options.qed_volumes) {
      options_map_.scf_options.qed_lambdas.push_back(sqrt(1.0 / x));
    }
  }
  qed_nmodes = options_map_.scf_options.qed_lambdas.size();
  if(qed_nmodes > 0) {
    is_qed = true;
    do_qed = true;
  }
  if(is_qed && is_ks) {
    for(auto x: options_map_.scf_options.xc_type) {
      if(strequal_case(x, "gga_xc_qed") || strequal_case(x, "mgga_xc_qed") ||
         strequal_case(x, "hyb_gga_xc_qed") || strequal_case(x, "hyb_mgga_xc_qed")) {
        if(options_map_.scf_options.qed_omegas.size() != qed_nmodes) {
          tamm_terminate("ERROR: (m)gga_xc_qed needs qed_omegas for each qed_nmode");
        }
        break;
      }
      else if(strequal_case(x, "gga_x_qed") || strequal_case(x, "mgga_x_qed") ||
              strequal_case(x, "hyb_gga_x_qed") || strequal_case(x, "hyb_mgga_x_qed")) {
        do_qed = false;
        break;
      }
    }
  }
}

void write_json_data(SystemData& sys_data, const std::string cmodule) {
  auto options = sys_data.options_map;
  auto scf     = options.scf_options;
  auto cd      = options.cd_options;
  auto ccsd    = options.ccsd_options;

  json& results = sys_data.results;

  auto str_bool = [=](const bool val) {
    if(val) return "true";
    return "false";
  };

  results["input"]["molecule"]["name"]           = sys_data.input_molecule;
  results["input"]["molecule"]["basisset"]       = scf.basis;
  results["input"]["molecule"]["gaussian_type"]  = scf.gaussian_type;
  results["input"]["molecule"]["geometry_units"] = scf.geom_units;
  // SCF options
  results["input"]["SCF"]["tol_int"]        = scf.tol_int;
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
    results["input"]["CD"]["write_cv"]         = cd.write_cv;
    results["input"]["CD"]["write_vcount"]     = cd.write_vcount;
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
  to_lower(l_module);

  std::string out_fp = sys_data.output_file_prefix + "." + sys_data.options_map.ccsd_options.basis;
  std::string files_dir = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type + "/json";
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  std::string files_prefix = files_dir + "/" + out_fp;
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
