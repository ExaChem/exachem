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

void ParseCCSDOptions::print(ChemEnv& chem_env) {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CCSD Options" << std::endl;
  std::cout << "{" << std::endl;

  CCSDOptions& cc_options = chem_env.ioptions.ccsd_options;
  std::cout << " cache_size           = " << cc_options.cache_size << std::endl;
  std::cout << " ccsdt_tilesize       = " << cc_options.ccsdt_tilesize << std::endl;

  std::cout << " ndiis                = " << cc_options.ndiis << std::endl;
  std::cout << " threshold            = " << cc_options.threshold << std::endl;
  std::cout << " tilesize             = " << cc_options.tilesize << std::endl;
  if(cc_options.nactive > 0)
    std::cout << " nactive              = " << cc_options.nactive << std::endl;
  if(cc_options.pcore > 0) std::cout << " pcore                = " << cc_options.pcore << std::endl;
  std::cout << " ccsd_maxiter         = " << cc_options.ccsd_maxiter << std::endl;
  txt_utils::print_bool(" freeze_atomic        ", cc_options.freeze_atomic);
  std::cout << " freeze_core          = " << cc_options.freeze_core << std::endl;
  std::cout << " freeze_virtual       = " << cc_options.freeze_virtual << std::endl;
  if(cc_options.lshift != 0)
    std::cout << " lshift               = " << cc_options.lshift << std::endl;
  if(cc_options.gf_nprocs_poi > 0)
    std::cout << " gf_nprocs_poi        = " << cc_options.gf_nprocs_poi << std::endl;
  txt_utils::print_bool(" readt               ", cc_options.readt);
  txt_utils::print_bool(" writet              ", cc_options.writet);
  txt_utils::print_bool(" writev              ", cc_options.writev);
  // txt_utils::print_bool(" computeTData        ", computeTData);
  std::cout << " writet_iter          = " << cc_options.writet_iter << std::endl;
  txt_utils::print_bool(" profile_ccsd        ", cc_options.profile_ccsd);
  txt_utils::print_bool(" balance_tiles       ", cc_options.balance_tiles);

  if(!cc_options.dlpno_dfbasis.empty())
    std::cout << " dlpno_dfbasis        = " << cc_options.dlpno_dfbasis << std::endl;
  if(!cc_options.doubles_opt_eqns.empty()) {
    std::cout << " doubles_opt_eqns        = [";
    for(auto x: cc_options.doubles_opt_eqns) std::cout << x << ",";
    std::cout << "]" << std::endl;
  }

  if(!cc_options.ext_data_path.empty()) {
    std::cout << " ext_data_path   = " << cc_options.ext_data_path << std::endl;
  }

  if(cc_options.eom_nroots > 0) {
    std::cout << " eom_nroots           = " << cc_options.eom_nroots << std::endl;
    std::cout << " eom_microiter        = " << cc_options.eom_microiter << std::endl;
    std::cout << " eom_threshold        = " << cc_options.eom_threshold << std::endl;
  }

  if(cc_options.gf_p_oi_range > 0) {
    std::cout << " gf_p_oi_range        = " << cc_options.gf_p_oi_range << std::endl;
    txt_utils::print_bool(" gf_ip               ", cc_options.gf_ip);
    txt_utils::print_bool(" gf_ea               ", cc_options.gf_ea);
    txt_utils::print_bool(" gf_os               ", cc_options.gf_os);
    txt_utils::print_bool(" gf_cs               ", cc_options.gf_cs);
    txt_utils::print_bool(" gf_restart          ", cc_options.gf_restart);
    txt_utils::print_bool(" gf_profile          ", cc_options.gf_profile);
    txt_utils::print_bool(" gf_itriples         ", cc_options.gf_itriples);
    std::cout << " gf_ndiis             = " << cc_options.gf_ndiis << std::endl;
    std::cout << " gf_ngmres            = " << cc_options.gf_ngmres << std::endl;
    std::cout << " gf_maxiter           = " << cc_options.gf_maxiter << std::endl;
    std::cout << " gf_eta               = " << cc_options.gf_eta << std::endl;
    std::cout << " gf_lshift            = " << cc_options.gf_lshift << std::endl;
    std::cout << " gf_preconditioning   = " << cc_options.gf_preconditioning << std::endl;
    std::cout << " gf_damping_factor    = " << cc_options.gf_damping_factor << std::endl;

    // std::cout << " gf_omega       = " << gf_omega <<std::endl;
    std::cout << " gf_threshold         = " << cc_options.gf_threshold << std::endl;
    std::cout << " gf_omega_min_ip      = " << cc_options.gf_omega_min_ip << std::endl;
    std::cout << " gf_omega_max_ip      = " << cc_options.gf_omega_max_ip << std::endl;
    std::cout << " gf_omega_min_ip_e    = " << cc_options.gf_omega_min_ip_e << std::endl;
    std::cout << " gf_omega_max_ip_e    = " << cc_options.gf_omega_max_ip_e << std::endl;
    std::cout << " gf_omega_min_ea      = " << cc_options.gf_omega_min_ea << std::endl;
    std::cout << " gf_omega_max_ea      = " << cc_options.gf_omega_max_ea << std::endl;
    std::cout << " gf_omega_min_ea_e    = " << cc_options.gf_omega_min_ea_e << std::endl;
    std::cout << " gf_omega_max_ea_e    = " << cc_options.gf_omega_max_ea_e << std::endl;
    std::cout << " gf_omega_delta       = " << cc_options.gf_omega_delta << std::endl;
    std::cout << " gf_omega_delta_e     = " << cc_options.gf_omega_delta_e << std::endl;
    if(!cc_options.gf_orbitals.empty()) {
      std::cout << " gf_orbitals        = [";
      for(auto x: cc_options.gf_orbitals) std::cout << x << ",";
      std::cout << "]" << std::endl;
    }
    if(cc_options.gf_analyze_level > 0) {
      std::cout << " gf_analyze_level     = " << cc_options.gf_analyze_level << std::endl;
      std::cout << " gf_analyze_num_omega = " << cc_options.gf_analyze_num_omega << std::endl;
      std::cout << " gf_analyze_omega     = [";
      for(auto x: cc_options.gf_analyze_omega) std::cout << x << ",";
      std::cout << "]" << std::endl;
    }
    if(cc_options.gf_extrapolate_level > 0)
      std::cout << " gf_extrapolate_level = " << cc_options.gf_extrapolate_level << std::endl;
  }

  txt_utils::print_bool(" debug               ", cc_options.debug);
  std::cout << "}" << std::endl;
}