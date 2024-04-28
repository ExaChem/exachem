#include "parse_ccsd_options.hpp"

ParseCCSDOptions::ParseCCSDOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseCCSDOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseCCSDOptions::parse_check(json& jinput) {
  // clang-format off
    const std::vector<string> valid_cc{
      "CCSD(T)",  "DLPNO",        "EOMCCSD",        "RT-EOMCC",     "GFCCSD",
      "comments", "threshold",    "force_tilesize", "tilesize",     "computeTData",
      "lshift",   "ndiis",        "ccsd_maxiter",   "freeze",       "PRINT",
      "readt",    "writet",       "writev",         "writet_iter",  "debug",
      "nactive",  "profile_ccsd", "balance_tiles",  "ext_data_path"};
  // clang-format on
  for(auto& el: jinput["CC"].items()) {
    if(std::find(valid_cc.begin(), valid_cc.end(), el.key()) == valid_cc.end())
      tamm_terminate("INPUT FILE ERROR: Invalid CC option [" + el.key() + "] in the input file");
  }
}

void ParseCCSDOptions::parse(ChemEnv& chem_env) {
  json         jcc        = chem_env.jinput["CC"];
  CCSDOptions& cc_options = chem_env.ioptions.ccsd_options;
  parse_option<int>(cc_options.ndiis, jcc, "ndiis");
  parse_option<int>(cc_options.nactive, jcc, "nactive");
  parse_option<int>(cc_options.ccsd_maxiter, jcc, "ccsd_maxiter");
  parse_option<double>(cc_options.lshift, jcc, "lshift");
  parse_option<double>(cc_options.threshold, jcc, "threshold");
  parse_option<int>(cc_options.tilesize, jcc, "tilesize");
  parse_option<bool>(cc_options.debug, jcc, "debug");
  parse_option<bool>(cc_options.readt, jcc, "readt");
  parse_option<bool>(cc_options.writet, jcc, "writet");
  parse_option<bool>(cc_options.writev, jcc, "writev");
  parse_option<int>(cc_options.writet_iter, jcc, "writet_iter");
  parse_option<bool>(cc_options.balance_tiles, jcc, "balance_tiles");
  parse_option<bool>(cc_options.profile_ccsd, jcc, "profile_ccsd");
  parse_option<bool>(cc_options.force_tilesize, jcc, "force_tilesize");
  parse_option<string>(cc_options.ext_data_path, jcc, "ext_data_path");
  parse_option<bool>(cc_options.computeTData, jcc, "computeTData");

  json jcc_print = jcc["PRINT"];
  parse_option<bool>(cc_options.ccsd_diagnostics, jcc_print, "ccsd_diagnostics");
  parse_option<std::vector<int>>(cc_options.cc_rdm, jcc_print, "rdm");
  parse_option<std::pair<bool, double>>(cc_options.tamplitudes, jcc_print, "tamplitudes");

  json jcc_freeze = jcc["freeze"];
  parse_option<bool>(cc_options.freeze_atomic, jcc_freeze, "atomic");
  parse_option<int>(cc_options.freeze_core, jcc_freeze, "core");
  parse_option<int>(cc_options.freeze_virtual, jcc_freeze, "virtual");

  // RT-EOMCC
  json jrt_eom = jcc["RT-EOMCC"];
  parse_option<int>(cc_options.pcore, jrt_eom, "pcore");
  parse_option<int>(cc_options.ntimesteps, jrt_eom, "ntimesteps");
  parse_option<int>(cc_options.rt_microiter, jrt_eom, "rt_microiter");
  parse_option<double>(cc_options.rt_threshold, jrt_eom, "rt_threshold");
  parse_option<double>(cc_options.rt_step_size, jrt_eom, "rt_step_size");
  parse_option<double>(cc_options.rt_multiplier, jrt_eom, "rt_multiplier");
  parse_option<double>(cc_options.secent_x, jrt_eom, "secent_x");
  parse_option<double>(cc_options.h_red, jrt_eom, "h_red");
  parse_option<double>(cc_options.h_inc, jrt_eom, "h_inc");
  parse_option<double>(cc_options.h_max, jrt_eom, "h_max");

  // DLPNO
  json jdlpno = jcc["DLPNO"];
  parse_option<int>(cc_options.max_pnos, jdlpno, "max_pnos");
  parse_option<size_t>(cc_options.keep_npairs, jdlpno, "keep_npairs");
  parse_option<bool>(cc_options.localize, jdlpno, "localize");
  parse_option<bool>(cc_options.skip_dlpno, jdlpno, "skip_dlpno");
  parse_option<string>(cc_options.dlpno_dfbasis, jdlpno, "df_basisset");
  parse_option<double>(cc_options.TCutDO, jdlpno, "TCutDO");
  parse_option<double>(cc_options.TCutEN, jdlpno, "TCutEN");
  parse_option<double>(cc_options.TCutPNO, jdlpno, "TCutPNO");
  parse_option<double>(cc_options.TCutTNO, jdlpno, "TCutTNO");
  parse_option<double>(cc_options.TCutPre, jdlpno, "TCutPre");
  parse_option<double>(cc_options.TCutPairs, jdlpno, "TCutPairs");
  parse_option<double>(cc_options.TCutDOij, jdlpno, "TCutDOij");
  parse_option<double>(cc_options.TCutDOPre, jdlpno, "TCutDOPre");
  parse_option<std::vector<int>>(cc_options.doubles_opt_eqns, jdlpno, "doubles_opt_eqns");

  json jccsd_t = jcc["CCSD(T)"];
  parse_option<bool>(cc_options.skip_ccsd, jccsd_t, "skip_ccsd");
  parse_option<int>(cc_options.cache_size, jccsd_t, "cache_size");
  parse_option<int>(cc_options.ccsdt_tilesize, jccsd_t, "ccsdt_tilesize");

  json jeomccsd = jcc["EOMCCSD"];
  parse_option<int>(cc_options.eom_nroots, jeomccsd, "eom_nroots");
  parse_option<int>(cc_options.eom_microiter, jeomccsd, "eom_microiter");
  parse_option<string>(cc_options.eom_type, jeomccsd, "eom_type");
  parse_option<double>(cc_options.eom_threshold, jeomccsd, "eom_threshold");

  json jgfcc = jcc["GFCCSD"];
  // clang-format off
  parse_option<bool>(cc_options.gf_ip      , jgfcc, "gf_ip");
  parse_option<bool>(cc_options.gf_ea      , jgfcc, "gf_ea");
  parse_option<bool>(cc_options.gf_os      , jgfcc, "gf_os");
  parse_option<bool>(cc_options.gf_cs      , jgfcc, "gf_cs");
  parse_option<bool>(cc_options.gf_restart , jgfcc, "gf_restart");
  parse_option<bool>(cc_options.gf_profile , jgfcc, "gf_profile");
  parse_option<bool>(cc_options.gf_itriples, jgfcc, "gf_itriples");

  parse_option<int>   (cc_options.gf_ndiis            , jgfcc, "gf_ndiis");
  parse_option<int>   (cc_options.gf_ngmres           , jgfcc, "gf_ngmres");
  parse_option<int>   (cc_options.gf_maxiter          , jgfcc, "gf_maxiter");
  parse_option<int>   (cc_options.gf_nprocs_poi       , jgfcc, "gf_nprocs_poi");
  parse_option<double>(cc_options.gf_damping_factor   , jgfcc, "gf_damping_factor");
  parse_option<double>(cc_options.gf_eta              , jgfcc, "gf_eta");
  parse_option<double>(cc_options.gf_lshift           , jgfcc, "gf_lshift");
  parse_option<bool>  (cc_options.gf_preconditioning  , jgfcc, "gf_preconditioning");
  parse_option<double>(cc_options.gf_threshold        , jgfcc, "gf_threshold");
  parse_option<double>(cc_options.gf_omega_min_ip     , jgfcc, "gf_omega_min_ip");
  parse_option<double>(cc_options.gf_omega_max_ip     , jgfcc, "gf_omega_max_ip");
  parse_option<double>(cc_options.gf_omega_min_ip_e   , jgfcc, "gf_omega_min_ip_e");
  parse_option<double>(cc_options.gf_omega_max_ip_e   , jgfcc, "gf_omega_max_ip_e");
  parse_option<double>(cc_options.gf_omega_min_ea     , jgfcc, "gf_omega_min_ea");
  parse_option<double>(cc_options.gf_omega_max_ea     , jgfcc, "gf_omega_max_ea");
  parse_option<double>(cc_options.gf_omega_min_ea_e   , jgfcc, "gf_omega_min_ea_e");
  parse_option<double>(cc_options.gf_omega_max_ea_e   , jgfcc, "gf_omega_max_ea_e");
  parse_option<double>(cc_options.gf_omega_delta      , jgfcc, "gf_omega_delta");
  parse_option<double>(cc_options.gf_omega_delta_e    , jgfcc, "gf_omega_delta_e");
  parse_option<int>   (cc_options.gf_extrapolate_level, jgfcc, "gf_extrapolate_level");
  parse_option<int>   (cc_options.gf_analyze_level    , jgfcc, "gf_analyze_level");
  parse_option<int>   (cc_options.gf_analyze_num_omega, jgfcc, "gf_analyze_num_omega");
  parse_option<int>   (cc_options.gf_p_oi_range       , jgfcc, "gf_p_oi_range");

  parse_option<std::vector<size_t>>(cc_options.gf_orbitals     , jgfcc, "gf_orbitals");
  parse_option<std::vector<double>>(cc_options.gf_analyze_omega, jgfcc, "gf_analyze_omega");
  // clang-format on
  if(cc_options.gf_p_oi_range != 0) {
    if(cc_options.gf_p_oi_range != 1 && cc_options.gf_p_oi_range != 2)
      tamm_terminate("gf_p_oi_range can only be one of 1 or 2");
  }

  update_common_options(chem_env);
}

void ParseCCSDOptions::update_common_options(ChemEnv& chem_env) {
  CCSDOptions&   cc_options     = chem_env.ioptions.ccsd_options;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  cc_options.debug         = common_options.debug;
  cc_options.maxiter       = common_options.maxiter;
  cc_options.basis         = common_options.basis;
  cc_options.dfbasis       = common_options.dfbasis;
  cc_options.basisfile     = common_options.basisfile;
  cc_options.gaussian_type = common_options.gaussian_type;
  cc_options.geom_units    = common_options.geom_units;
  cc_options.file_prefix   = common_options.file_prefix;
  cc_options.ext_data_path = common_options.ext_data_path;
}
