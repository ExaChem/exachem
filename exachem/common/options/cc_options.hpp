class CCSDOptions: public COptions {
public:
  CCSDOptions(): COptions() { initialize(); }

  CCSDOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  CCSDOptions(COptions o): COptions(o) { initialize(); }

  CCSDOptions(COptions o, json& jinput): COptions(o) {
    initialize();
    parse(jinput);
  }
  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  CCSDOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void initialize() {
    threshold      = 1e-6;
    force_tilesize = false;
    tilesize       = 40;
    ndiis          = 5;
    lshift         = 0;
    nactive        = 0;
    ccsd_maxiter   = 50;
    freeze_core    = 0;
    freeze_virtual = 0;
    balance_tiles  = true;
    profile_ccsd   = false;

    writet       = false;
    writev       = false;
    writet_iter  = ndiis;
    readt        = false;
    computeTData = false;

    localize      = false;
    skip_dlpno    = false;
    keep_npairs   = 1;
    max_pnos      = 1;
    dlpno_dfbasis = "";
    TCutEN        = 0.97;
    TCutPNO       = 1.0e-6;
    TCutTNO       = 1.0e-6;
    TCutPre       = 1.0e-3;
    TCutPairs     = 1.0e-3;
    TCutDO        = 1e-2;
    TCutDOij      = 1e-7;
    TCutDOPre     = 3e-2;

    cache_size     = 8;
    skip_ccsd      = false;
    ccsdt_tilesize = 32;

    eom_nroots    = 1;
    eom_threshold = 1e-6;
    eom_type      = "right";
    eom_microiter = ccsd_maxiter;

    pcore         = 0;
    ntimesteps    = 10;
    rt_microiter  = 20;
    rt_threshold  = 1e-6;
    rt_multiplier = 0.5;
    rt_step_size  = 0.025;
    secent_x      = 0.1;
    h_red         = 0.5;
    h_inc         = 1.2;
    h_max         = 0.25;

    gf_ip       = true;
    gf_ea       = false;
    gf_os       = false;
    gf_cs       = true;
    gf_restart  = false;
    gf_itriples = false;
    gf_profile  = false;

    gf_p_oi_range      = 0; // 1-number of occupied, 2-all MOs
    gf_ndiis           = 10;
    gf_ngmres          = 10;
    gf_maxiter         = 500;
    gf_eta             = 0.01;
    gf_lshift          = 1.0;
    gf_preconditioning = true;
    gf_damping_factor  = 1.0;
    gf_nprocs_poi      = 0;
    // gf_omega          = -0.4; //a.u (range min to max)
    gf_threshold         = 1e-2;
    gf_omega_min_ip      = -0.8;
    gf_omega_max_ip      = -0.4;
    gf_omega_min_ip_e    = -2.0;
    gf_omega_max_ip_e    = 0;
    gf_omega_min_ea      = 0.0;
    gf_omega_max_ea      = 0.1;
    gf_omega_min_ea_e    = 0.0;
    gf_omega_max_ea_e    = 2.0;
    gf_omega_delta       = 0.01;
    gf_omega_delta_e     = 0.002;
    gf_extrapolate_level = 0;
    gf_analyze_level     = 0;
    gf_analyze_num_omega = 0;
  }

  int  tilesize;
  bool force_tilesize;
  int  ndiis;
  int  writet_iter;
  bool readt, writet, writev, gf_restart, gf_ip, gf_ea, gf_os, gf_cs, gf_itriples, gf_profile,
    balance_tiles, computeTData;
  bool                    profile_ccsd;
  double                  lshift;
  double                  threshold;
  bool                    ccsd_diagnostics{false};
  std::pair<bool, double> tamplitudes{false, 0.05};
  std::vector<int>        cc_rdm{};

  int  nactive;
  int  ccsd_maxiter;
  bool freeze_atomic{false};
  int  freeze_core;
  int  freeze_virtual;

  // RT-EOMCC
  int    pcore;      // pth core orbital
  int    ntimesteps; // number of time steps
  int    rt_microiter;
  double rt_threshold;
  double rt_step_size;
  double rt_multiplier;
  double secent_x; // secent scale factor
  double h_red;    // time-step reduction factor
  double h_inc;    // time-step increase factor
  double h_max;    // max time-step factor

  // CCSD(T)
  bool skip_ccsd;
  int  cache_size;
  int  ccsdt_tilesize;

  // DLPNO
  bool             localize;
  bool             skip_dlpno;
  int              max_pnos;
  size_t           keep_npairs;
  double           TCutEN;
  double           TCutPNO;
  double           TCutTNO;
  double           TCutPre;
  double           TCutPairs;
  double           TCutDO;
  double           TCutDOij;
  double           TCutDOPre;
  std::string      dlpno_dfbasis;
  std::vector<int> doubles_opt_eqns;

  // EOM
  int    eom_nroots;
  int    eom_microiter;
  string eom_type;
  double eom_threshold;

  // GF
  int    gf_p_oi_range;
  int    gf_ndiis;
  int    gf_ngmres;
  int    gf_maxiter;
  double gf_eta;
  double gf_lshift;
  bool   gf_preconditioning;
  int    gf_nprocs_poi;
  double gf_damping_factor;
  // double gf_omega;
  double              gf_threshold;
  double              gf_omega_min_ip;
  double              gf_omega_max_ip;
  double              gf_omega_min_ip_e;
  double              gf_omega_max_ip_e;
  double              gf_omega_min_ea;
  double              gf_omega_max_ea;
  double              gf_omega_min_ea_e;
  double              gf_omega_max_ea_e;
  double              gf_omega_delta;
  double              gf_omega_delta_e;
  int                 gf_extrapolate_level;
  int                 gf_analyze_level;
  int                 gf_analyze_num_omega;
  std::vector<double> gf_analyze_omega;
  // Force processing of specified orbitals first
  std::vector<size_t> gf_orbitals;

  void parse(json& jinput) {
    json                      jcc = jinput["CC"];
    const std::vector<string> valid_cc{
      "CCSD(T)",  "DLPNO",        "EOMCCSD",        "RT-EOMCC",     "GFCCSD",
      "comments", "threshold",    "force_tilesize", "tilesize",     "computeTData",
      "lshift",   "ndiis",        "ccsd_maxiter",   "freeze",       "PRINT",
      "readt",    "writet",       "writev",         "writet_iter",  "debug",
      "nactive",  "profile_ccsd", "balance_tiles",  "ext_data_path"};
    for(auto& el: jcc.items()) {
      if(std::find(valid_cc.begin(), valid_cc.end(), el.key()) == valid_cc.end())
        tamm_terminate("INPUT FILE ERROR: Invalid CC option [" + el.key() + "] in the input file");
    }
    // clang-format off
  parse_option<int>   (ndiis         , jcc, "ndiis");
  parse_option<int>   (nactive       , jcc, "nactive");
  parse_option<int>   (ccsd_maxiter  , jcc, "ccsd_maxiter");
  parse_option<double>(lshift        , jcc, "lshift");
  parse_option<double>(threshold     , jcc, "threshold");
  parse_option<int>   (tilesize      , jcc, "tilesize");
  parse_option<bool>  (debug         , jcc, "debug");
  parse_option<bool>  (readt         , jcc, "readt");
  parse_option<bool>  (writet        , jcc, "writet");
  parse_option<bool>  (writev        , jcc, "writev");
  parse_option<int>   (writet_iter   , jcc, "writet_iter");
  parse_option<bool>  (balance_tiles , jcc, "balance_tiles");
  parse_option<bool>  (profile_ccsd  , jcc, "profile_ccsd");
  parse_option<bool>  (force_tilesize, jcc, "force_tilesize");
  parse_option<string>(ext_data_path , jcc, "ext_data_path");
  parse_option<bool>  (computeTData  , jcc, "computeTData");

  json jcc_print = jcc["PRINT"];
  parse_option<bool> (ccsd_diagnostics, jcc_print, "ccsd_diagnostics");
  parse_option<std::vector<int>>(cc_rdm, jcc_print, "rdm");
  parse_option<std::pair<bool, double>>(tamplitudes, jcc_print, "tamplitudes");

  json jcc_freeze = jcc["freeze"];
  parse_option<bool>(freeze_atomic,  jcc_freeze, "atomic");
  parse_option<int> (freeze_core,    jcc_freeze, "core");
  parse_option<int> (freeze_virtual, jcc_freeze, "virtual");

  //RT-EOMCC
  json jrt_eom = jcc["RT-EOMCC"];
  parse_option<int>   (pcore        , jrt_eom, "pcore");
  parse_option<int>   (ntimesteps   , jrt_eom, "ntimesteps");
  parse_option<int>   (rt_microiter , jrt_eom, "rt_microiter");
  parse_option<double>(rt_threshold , jrt_eom, "rt_threshold");
  parse_option<double>(rt_step_size , jrt_eom, "rt_step_size");
  parse_option<double>(rt_multiplier, jrt_eom, "rt_multiplier");
  parse_option<double>(secent_x     , jrt_eom, "secent_x");
  parse_option<double>(h_red        , jrt_eom, "h_red");
  parse_option<double>(h_inc        , jrt_eom, "h_inc");
  parse_option<double>(h_max        , jrt_eom, "h_max");

  // DLPNO
  json jdlpno = jcc["DLPNO"];
  parse_option<int>   (max_pnos     , jdlpno, "max_pnos");
  parse_option<size_t>(keep_npairs  , jdlpno, "keep_npairs");
  parse_option<bool>  (localize     , jdlpno, "localize");
  parse_option<bool>  (skip_dlpno   , jdlpno, "skip_dlpno");
  parse_option<string>(dlpno_dfbasis, jdlpno, "df_basisset");
  parse_option<double>(TCutDO       , jdlpno, "TCutDO");
  parse_option<double>(TCutEN       , jdlpno, "TCutEN");
  parse_option<double>(TCutPNO      , jdlpno, "TCutPNO");
  parse_option<double>(TCutTNO      , jdlpno, "TCutTNO");
  parse_option<double>(TCutPre      , jdlpno, "TCutPre");
  parse_option<double>(TCutPairs    , jdlpno, "TCutPairs");
  parse_option<double>(TCutDOij     , jdlpno, "TCutDOij");
  parse_option<double>(TCutDOPre    , jdlpno, "TCutDOPre");
  parse_option<std::vector<int>>(doubles_opt_eqns, jdlpno, "doubles_opt_eqns");

  json jccsd_t = jcc["CCSD(T)"];
  parse_option<bool>(skip_ccsd    , jccsd_t, "skip_ccsd");
  parse_option<int>(cache_size    , jccsd_t, "cache_size");
  parse_option<int>(ccsdt_tilesize, jccsd_t, "ccsdt_tilesize");

  json jeomccsd = jcc["EOMCCSD"];
  parse_option<int>   (eom_nroots   , jeomccsd, "eom_nroots");
  parse_option<int>   (eom_microiter, jeomccsd, "eom_microiter");
  parse_option<string>(eom_type     , jeomccsd, "eom_type");
  parse_option<double>(eom_threshold, jeomccsd, "eom_threshold");
    // clang-format on
    std::vector<string> etlist{"right", "left", "RIGHT", "LEFT"};
    if(std::find(std::begin(etlist), std::end(etlist), string(eom_type)) == std::end(etlist))
      tamm_terminate("INPUT FILE ERROR: EOMCC type can only be one of [left,right]");

    json jgfcc = jcc["GFCCSD"];
    // clang-format off
  parse_option<bool>(gf_ip      , jgfcc, "gf_ip");
  parse_option<bool>(gf_ea      , jgfcc, "gf_ea");
  parse_option<bool>(gf_os      , jgfcc, "gf_os");
  parse_option<bool>(gf_cs      , jgfcc, "gf_cs");
  parse_option<bool>(gf_restart , jgfcc, "gf_restart");
  parse_option<bool>(gf_profile , jgfcc, "gf_profile");
  parse_option<bool>(gf_itriples, jgfcc, "gf_itriples");

  parse_option<int>   (gf_ndiis            , jgfcc, "gf_ndiis");
  parse_option<int>   (gf_ngmres           , jgfcc, "gf_ngmres");
  parse_option<int>   (gf_maxiter          , jgfcc, "gf_maxiter");
  parse_option<int>   (gf_nprocs_poi       , jgfcc, "gf_nprocs_poi");
  parse_option<double>(gf_damping_factor   , jgfcc, "gf_damping_factor");
  parse_option<double>(gf_eta              , jgfcc, "gf_eta");
  parse_option<double>(gf_lshift           , jgfcc, "gf_lshift");
  parse_option<bool>  (gf_preconditioning  , jgfcc, "gf_preconditioning");
  parse_option<double>(gf_threshold        , jgfcc, "gf_threshold");
  parse_option<double>(gf_omega_min_ip     , jgfcc, "gf_omega_min_ip");
  parse_option<double>(gf_omega_max_ip     , jgfcc, "gf_omega_max_ip");
  parse_option<double>(gf_omega_min_ip_e   , jgfcc, "gf_omega_min_ip_e");
  parse_option<double>(gf_omega_max_ip_e   , jgfcc, "gf_omega_max_ip_e");
  parse_option<double>(gf_omega_min_ea     , jgfcc, "gf_omega_min_ea");
  parse_option<double>(gf_omega_max_ea     , jgfcc, "gf_omega_max_ea");
  parse_option<double>(gf_omega_min_ea_e   , jgfcc, "gf_omega_min_ea_e");
  parse_option<double>(gf_omega_max_ea_e   , jgfcc, "gf_omega_max_ea_e");
  parse_option<double>(gf_omega_delta      , jgfcc, "gf_omega_delta");
  parse_option<double>(gf_omega_delta_e    , jgfcc, "gf_omega_delta_e");
  parse_option<int>   (gf_extrapolate_level, jgfcc, "gf_extrapolate_level");
  parse_option<int>   (gf_analyze_level    , jgfcc, "gf_analyze_level");
  parse_option<int>   (gf_analyze_num_omega, jgfcc, "gf_analyze_num_omega");
  parse_option<int>   (gf_p_oi_range       , jgfcc, "gf_p_oi_range");

  parse_option<std::vector<size_t>>(gf_orbitals     , jgfcc, "gf_orbitals");
  parse_option<std::vector<double>>(gf_analyze_omega, jgfcc, "gf_analyze_omega");
    // clang-format on
    if(gf_p_oi_range != 0) {
      if(gf_p_oi_range != 1 && gf_p_oi_range != 2)
        tamm_terminate("gf_p_oi_range can only be one of 1 or 2");
    }

  } // end of parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

  void print() {
    std::cout << std::defaultfloat;
    std::cout << std::endl << "CCSD COptions" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << " cache_size           = " << cache_size << std::endl;
    std::cout << " ccsdt_tilesize       = " << ccsdt_tilesize << std::endl;

    std::cout << " ndiis                = " << ndiis << std::endl;
    std::cout << " threshold            = " << threshold << std::endl;
    std::cout << " tilesize             = " << tilesize << std::endl;
    if(nactive > 0) std::cout << " nactive              = " << nactive << std::endl;
    if(pcore > 0) std::cout << " pcore                = " << pcore << std::endl;
    std::cout << " ccsd_maxiter         = " << ccsd_maxiter << std::endl;
    caseChange.print_bool(" freeze_atomic        ", freeze_atomic);
    std::cout << " freeze_core          = " << freeze_core << std::endl;
    std::cout << " freeze_virtual       = " << freeze_virtual << std::endl;
    if(lshift != 0) std::cout << " lshift               = " << lshift << std::endl;
    if(gf_nprocs_poi > 0) std::cout << " gf_nprocs_poi        = " << gf_nprocs_poi << std::endl;
    caseChange.print_bool(" readt               ", readt);
    caseChange.print_bool(" writet              ", writet);
    caseChange.print_bool(" writev              ", writev);
    // caseChange.print_bool(" computeTData        ", computeTData);
    std::cout << " writet_iter          = " << writet_iter << std::endl;
    caseChange.print_bool(" profile_ccsd        ", profile_ccsd);
    caseChange.print_bool(" balance_tiles       ", balance_tiles);

    if(!dlpno_dfbasis.empty())
      std::cout << " dlpno_dfbasis        = " << dlpno_dfbasis << std::endl;
    if(!doubles_opt_eqns.empty()) {
      std::cout << " doubles_opt_eqns        = [";
      for(auto x: doubles_opt_eqns) std::cout << x << ",";
      std::cout << "]" << std::endl;
    }

    if(!ext_data_path.empty()) { std::cout << " ext_data_path   = " << ext_data_path << std::endl; }

    if(eom_nroots > 0) {
      std::cout << " eom_nroots           = " << eom_nroots << std::endl;
      std::cout << " eom_microiter        = " << eom_microiter << std::endl;
      std::cout << " eom_threshold        = " << eom_threshold << std::endl;
    }

    if(gf_p_oi_range > 0) {
      std::cout << " gf_p_oi_range        = " << gf_p_oi_range << std::endl;
      caseChange.print_bool(" gf_ip               ", gf_ip);
      caseChange.print_bool(" gf_ea               ", gf_ea);
      caseChange.print_bool(" gf_os               ", gf_os);
      caseChange.print_bool(" gf_cs               ", gf_cs);
      caseChange.print_bool(" gf_restart          ", gf_restart);
      caseChange.print_bool(" gf_profile          ", gf_profile);
      caseChange.print_bool(" gf_itriples         ", gf_itriples);
      std::cout << " gf_ndiis             = " << gf_ndiis << std::endl;
      std::cout << " gf_ngmres            = " << gf_ngmres << std::endl;
      std::cout << " gf_maxiter           = " << gf_maxiter << std::endl;
      std::cout << " gf_eta               = " << gf_eta << std::endl;
      std::cout << " gf_lshift            = " << gf_lshift << std::endl;
      std::cout << " gf_preconditioning   = " << gf_preconditioning << std::endl;
      std::cout << " gf_damping_factor    = " << gf_damping_factor << std::endl;

      // std::cout << " gf_omega       = " << gf_omega <<std::endl;
      std::cout << " gf_threshold         = " << gf_threshold << std::endl;
      std::cout << " gf_omega_min_ip      = " << gf_omega_min_ip << std::endl;
      std::cout << " gf_omega_max_ip      = " << gf_omega_max_ip << std::endl;
      std::cout << " gf_omega_min_ip_e    = " << gf_omega_min_ip_e << std::endl;
      std::cout << " gf_omega_max_ip_e    = " << gf_omega_max_ip_e << std::endl;
      std::cout << " gf_omega_min_ea      = " << gf_omega_min_ea << std::endl;
      std::cout << " gf_omega_max_ea      = " << gf_omega_max_ea << std::endl;
      std::cout << " gf_omega_min_ea_e    = " << gf_omega_min_ea_e << std::endl;
      std::cout << " gf_omega_max_ea_e    = " << gf_omega_max_ea_e << std::endl;
      std::cout << " gf_omega_delta       = " << gf_omega_delta << std::endl;
      std::cout << " gf_omega_delta_e     = " << gf_omega_delta_e << std::endl;
      if(!gf_orbitals.empty()) {
        std::cout << " gf_orbitals        = [";
        for(auto x: gf_orbitals) std::cout << x << ",";
        std::cout << "]" << std::endl;
      }
      if(gf_analyze_level > 0) {
        std::cout << " gf_analyze_level     = " << gf_analyze_level << std::endl;
        std::cout << " gf_analyze_num_omega = " << gf_analyze_num_omega << std::endl;
        std::cout << " gf_analyze_omega     = [";
        for(auto x: gf_analyze_omega) std::cout << x << ",";
        std::cout << "]" << std::endl;
      }
      if(gf_extrapolate_level > 0)
        std::cout << " gf_extrapolate_level = " << gf_extrapolate_level << std::endl;
    }

    caseChange.print_bool(" debug               ", debug);
    std::cout << "}" << std::endl;
  }
};
