class FCIOptions: public COptions {
public:
  FCIOptions(): COptions() { initialize(); }

  FCIOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  FCIOptions(COptions o): COptions(o) { initialize(); }

  FCIOptions(COptions o, json& jinput): COptions(o) {
    initialize();
    parse(jinput);
  }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  FCIOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void initialize() {
    job       = "CI";
    expansion = "CAS";

    // MCSCF
    max_macro_iter     = 100;
    max_orbital_step   = 0.5;
    orb_grad_tol_mcscf = 5e-6;
    enable_diis        = true;
    diis_start_iter    = 3;
    diis_nkeep         = 10;
    ci_res_tol         = 1e-8;
    ci_max_subspace    = 20;
    ci_matel_tol       = std::numeric_limits<double>::epsilon();

    // PRINT
    print_mcscf = true;
  }

  int         nalpha{}, nbeta{}, nactive{}, ninactive{};
  std::string job, expansion;

  // MCSCF
  bool   enable_diis;
  int    max_macro_iter, diis_start_iter, diis_nkeep, ci_max_subspace;
  double max_orbital_step, orb_grad_tol_mcscf, ci_res_tol, ci_matel_tol;

  // FCIDUMP

  // PRINT
  bool print_davidson{}, print_ci{}, print_mcscf{}, print_diis{}, print_asci_search{};
  std::pair<bool, double> print_state_char{false, 1e-2};

  void parse(json& jinput) {
    json                           jfci = jinput["FCI"];
    const std::vector<std::string> valid_fci{"nalpha",   "nbeta", "nactive",   "ninactive",
                                             "comments", "job",   "expansion", "FCIDUMP",
                                             "MCSCF",    "PRINT"};

    for(auto& el: jfci.items()) {
      if(std::find(valid_fci.begin(), valid_fci.end(), el.key()) == valid_fci.end())
        tamm_terminate("INPUT FILE ERROR: Invalid FCI option [" + el.key() + "] in the input file");
    }

    // clang-format off
  parse_option<int>(nalpha,       jfci, "nalpha");
  parse_option<int>(nbeta,        jfci, "nbeta");
  parse_option<int>(nactive,      jfci, "nactive");
  parse_option<int>(ninactive,    jfci, "ninactive");
  parse_option<std::string>(job,       jfci, "job");
  parse_option<std::string>(expansion, jfci, "expansion");


  // MCSCF
  json jmcscf = jfci["MCSCF"];
  parse_option<int>   (max_macro_iter,     jmcscf, "max_macro_iter");
  parse_option<double>(max_orbital_step,   jmcscf, "max_orbital_step");
  parse_option<double>(orb_grad_tol_mcscf, jmcscf, "orb_grad_tol_mcscf");
  parse_option<bool>  (enable_diis,        jmcscf, "enable_diis");
  parse_option<int>   (diis_start_iter,    jmcscf, "diis_start_iter");
  parse_option<int>   (diis_nkeep,         jmcscf, "diis_nkeep");
  parse_option<double>(ci_res_tol,         jmcscf, "ci_res_tol");
  parse_option<int>   (ci_max_subspace,    jmcscf, "ci_max_subspace");
  parse_option<double>(ci_matel_tol,       jmcscf, "ci_matel_tol");

  // FCIDUMP
  json jfcidump = jfci["FCIDUMP"];
  // parse_option<bool>(fcidump,    jfcidump, "fcidump");

  // PRINT
  json jprint = jfci["PRINT"];
  parse_option<bool>(print_davidson,    jprint, "davidson");
  parse_option<bool>(print_ci,          jprint, "ci");
  parse_option<bool>(print_mcscf,       jprint, "mcscf");
  parse_option<bool>(print_diis,        jprint, "diis");
  parse_option<bool>(print_asci_search, jprint, "asci_search");
  parse_option<std::pair<bool, double>>(print_state_char, jprint, "state_char");
}//end parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_){
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

};