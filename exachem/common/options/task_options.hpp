
class TaskOptions: public COptions {
public:
  TaskOptions() = default;
  TaskOptions(COptions o): COptions(o) {}
  TaskOptions(COptions o, json& jinput): COptions(o) { parse(jinput); }
  TaskOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    parse(jinput, ec_atoms, atoms_);
  }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    parse(jinput, ec_atoms, atoms_);
  }

  TaskOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  bool                         sinfo{false};
  bool                         scf{false};
  bool                         mp2{false};
  bool                         gw{false};
  bool                         cc2{false};
  bool                         fci{false};
  bool                         fcidump{false};
  bool                         ducc{false};
  bool                         cd_2e{false};
  bool                         ccsd{false};
  bool                         ccsd_sf{false};
  bool                         ccsd_t{false};
  bool                         ccsd_lambda{false};
  bool                         eom_ccsd{false};
  bool                         rteom_cc2{false};
  bool                         rteom_ccsd{false};
  bool                         gfccsd{false};
  std::pair<bool, std::string> dlpno_ccsd{false, ""};
  std::pair<bool, std::string> dlpno_ccsd_t{false, ""};

  void parse(json& jinput) {
    json                      jtask = jinput["TASK"];
    const std::vector<string> valid_tasks{
      "sinfo",  "scf",         "fci",          "fcidump",   "mp2",        "gw",      "cd_2e",
      "cc2",    "dlpno_ccsd",  "dlpno_ccsd_t", "ducc",      "ccsd",       "ccsd_sf", "ccsd_t",
      "gfccsd", "ccsd_lambda", "eom_ccsd",     "rteom_cc2", "rteom_ccsd", "comments"};

    for(auto& el: jtask.items()) {
      if(std::find(valid_tasks.begin(), valid_tasks.end(), el.key()) == valid_tasks.end())
        tamm_terminate("INPUT FILE ERROR: Invalid TASK option [" + el.key() +
                       "] in the input file");
    }

    // clang-format off
  parse_option<bool>(sinfo       , jtask, "sinfo");
  parse_option<bool>(scf         , jtask, "scf");
  parse_option<bool>(mp2         , jtask, "mp2");
  parse_option<bool>(gw          , jtask, "gw");
  parse_option<bool>(cc2         , jtask, "cc2");
  parse_option<bool>(fci         , jtask, "fci");  
  parse_option<bool>(fcidump     , jtask, "fcidump");  
  parse_option<bool>(cd_2e       , jtask, "cd_2e");
  parse_option<bool>(ducc        , jtask, "ducc");
  parse_option<bool>(ccsd        , jtask, "ccsd");
  parse_option<bool>(ccsd_sf     , jtask, "ccsd_sf");
  parse_option<bool>(ccsd_t      , jtask, "ccsd_t");
  parse_option<bool>(ccsd_lambda , jtask, "ccsd_lambda");
  parse_option<bool>(eom_ccsd    , jtask, "eom_ccsd");
  parse_option<bool>(rteom_cc2   , jtask, "rteom_cc2");
  parse_option<bool>(rteom_ccsd  , jtask, "rteom_ccsd");
  parse_option<bool>(gfccsd      , jtask, "gfccsd");

  parse_option<std::pair<bool, std::string>>(dlpno_ccsd  , jtask, "dlpno_ccsd");
  parse_option<std::pair<bool, std::string>>(dlpno_ccsd_t, jtask, "dlpno_ccsd_t");
    // clang-format on
  } // end of parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

  void print() {
    //  std::cout << std::defaultfloat;
    // std::cout  <<  std::endl << "Common COptions" <<  std::endl;
    // std::cout  << "{" <<  std::endl;
    // std::cout  << " maxiter    = " << maxiter <<  std::endl;
    // std::cout  << " basis      = " << basis << " ";
    // std::cout  << gaussian_type;
    // std::cout  <<  std::endl;
    // if(!dfbasis.empty()) std::cout  << " dfbasis    = " << dfbasis <<  std::endl;
    // if(!basisfile.empty()) std::cout  << " basisfile  = " << basisfile <<  std::endl;
    // std::cout  << " geom_units = " << geom_units <<  std::endl;
    // caseChange.print_bool(" debug     ", debug);
    // if(!output_file_prefix.empty())
    //   std::cout  << " output_file_prefix    = " << output_file_prefix <<  std::endl;
    // std::cout  << "}" <<  std::endl;

    COptions::print();

    std::cout << std::endl << "Task COptions" << std::endl;
    std::cout << "{" << std::endl;
    caseChange.print_bool(" sinfo        ", sinfo);
    caseChange.print_bool(" scf          ", scf);
    caseChange.print_bool(" mp2          ", mp2);
    caseChange.print_bool(" gw           ", gw);
    caseChange.print_bool(" cc2          ", cc2);
    caseChange.print_bool(" fci          ", fci);
    caseChange.print_bool(" fcidump      ", fcidump);
    caseChange.print_bool(" cd_2e        ", cd_2e);
    caseChange.print_bool(" ducc         ", ducc);
    caseChange.print_bool(" ccsd         ", ccsd);
    caseChange.print_bool(" ccsd_sf      ", ccsd_sf);
    caseChange.print_bool(" ccsd_lambda  ", ccsd_lambda);
    caseChange.print_bool(" eom_ccsd     ", eom_ccsd);
    caseChange.print_bool(" rteom_cc2    ", rteom_cc2);
    caseChange.print_bool(" rteom_ccsd   ", rteom_ccsd);
    caseChange.print_bool(" gfccsd       ", gfccsd);
    std::cout << " dlpno_ccsd:  " << dlpno_ccsd.first << ", " << dlpno_ccsd.second << "\n";
    std::cout << " dlpno_ccsd_t " << dlpno_ccsd_t.first << ", " << dlpno_ccsd_t.second << "\n";
    std::cout << "}" << std::endl;
  }
};