
class SCFOptions: public COptions {
public:
  SCFOptions(): COptions() { initialize(); }

  SCFOptions(COptions o): COptions(o) { initialize(); }

  SCFOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  SCFOptions(COptions o, json& jinput): COptions(o) {
    initialize();
    parse(jinput);
  }

  /*
  Even though all the test runs went through, this function call operator does not
  initializes inherited member variables correctly. (left as an example)
  */
  // void operator()(COptions o, json& jinput) {
  //   initialize();
  //   parse(jinput);
  //   std::cout<<"--------------------"<<termcolor::bold<<termcolor::red<<"GN-Debug"<<termcolor::reset<<"----------------------------------\n";
  //   COptions::print();
  //   std::cout<<"--------------------"<<termcolor::bold<<termcolor::red<<"GN-Debug"<<termcolor::reset<<"----------------------------------\n";
  // }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  SCFOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }
  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void initialize() {
    charge           = 0;
    multiplicity     = 1;
    lshift           = 0;
    tol_int          = 1e-22;
    tol_sch          = 1e-10;
    tol_lindep       = 1e-5;
    conve            = 1e-8;
    convd            = 1e-7;
    diis_hist        = 10;
    AO_tilesize      = 30;
    dfAO_tilesize    = 50;
    restart_size     = 2000;
    restart          = false;
    noscf            = false;
    sad              = true;
    force_tilesize   = false;
    riscf            = 0; // 0 for JK, 1 for J, 2 for K
    riscf_str        = "JK";
    moldenfile       = "";
    n_lindep         = 0;
    scf_type         = "restricted";
    xc_type          = {}; // pbe0
    xc_grid_type     = "UltraFine";
    damp             = 100;
    nnodes           = 1;
    writem           = 1;
    scalapack_nb     = 256;
    scalapack_np_row = 0;
    scalapack_np_col = 0;
    qed_omegas       = {};
    qed_volumes      = {};
    qed_lambdas      = {};
    qed_polvecs      = {};
  }

  int         charge;
  int         multiplicity;
  double      lshift;     // level shift factor, +ve value b/w 0 and 1
  double      tol_int;    // tolerance for integral primitive screening
  double      tol_sch;    // tolerance for schwarz screening
  double      tol_lindep; // tolerance for linear dependencies
  double      conve;      // energy convergence
  double      convd;      // density convergence
  int         diis_hist;  // number of diis history entries
  uint32_t    AO_tilesize;
  uint32_t    dfAO_tilesize;
  bool        restart; // Read movecs from disk
  bool        noscf;   // only recompute energy from movecs
  bool        sad;
  bool        force_tilesize;
  int         restart_size; // read/write orthogonalizer, schwarz, etc matrices when N>=restart_size
  int         scalapack_nb;
  int         riscf;
  int         nnodes;
  int         scalapack_np_row;
  int         scalapack_np_col;
  std::string riscf_str;
  std::string moldenfile;
  int         n_lindep;
  int         writem;
  int         damp; // density mixing parameter
  std::string scf_type;
  std::string xc_grid_type;

  std::map<std::string, std::tuple<int, int>> guess_atom_options;

  std::vector<std::string> xc_type;
  // mos_txt: write lcao, mo transformed core H, fock, and v2 to disk as text files.
  bool                             mos_txt{false};
  bool                             mulliken_analysis{false};
  std::pair<bool, double>          mo_vectors_analysis{false, 0.15};
  std::vector<double>              qed_omegas;
  std::vector<double>              qed_volumes;
  std::vector<double>              qed_lambdas;
  std::vector<std::vector<double>> qed_polvecs;

  void parse(json& jinput) {
    // clang-format off
  json jscf = jinput["SCF"];
  parse_option<int>   (charge          , jscf, "charge");
  parse_option<int>   (multiplicity    , jscf, "multiplicity");
  parse_option<double>(lshift          , jscf, "lshift");
  parse_option<double>(tol_int         , jscf, "tol_int");
  parse_option<double>(tol_sch         , jscf, "tol_sch");
  parse_option<double>(tol_lindep      , jscf, "tol_lindep");
  parse_option<double>(conve           , jscf, "conve");
  parse_option<double>(convd           , jscf, "convd");
  parse_option<int>   (diis_hist       , jscf, "diis_hist");
  parse_option<bool>  (force_tilesize  , jscf, "force_tilesize");
  parse_option<uint32_t>(AO_tilesize   , jscf, "tilesize");
  parse_option<uint32_t>(dfAO_tilesize , jscf, "df_tilesize");
  parse_option<int>   (damp            , jscf, "damp");
  parse_option<int>   (writem          , jscf, "writem");
  parse_option<int>   (nnodes          , jscf, "nnodes");
  parse_option<bool>  (restart         , jscf, "restart");
  parse_option<bool>  (noscf           , jscf, "noscf");
  parse_option<bool>  (debug           , jscf, "debug");
  parse_option<std::string>(moldenfile      , jscf, "moldenfile");
  parse_option<std::string>(scf_type        , jscf, "scf_type");
  parse_option<std::string>(xc_grid_type    , jscf, "xc_grid_type");
  parse_option<std::vector<std::string>>(xc_type, jscf, "xc_type");
  parse_option<int>   (n_lindep        , jscf, "n_lindep");
  parse_option<int>   (restart_size    , jscf, "restart_size");
  parse_option<int>   (scalapack_nb    , jscf, "scalapack_nb");
  parse_option<int>   (scalapack_np_row, jscf, "scalapack_np_row");
  parse_option<int>   (scalapack_np_col, jscf, "scalapack_np_col");
  parse_option<std::string>(ext_data_path   , jscf, "ext_data_path");
  parse_option<std::vector<double>>(qed_omegas , jscf, "qed_omegas");
  parse_option<std::vector<double>>(qed_lambdas, jscf, "qed_lambdas");
  parse_option<std::vector<double>>(qed_volumes, jscf, "qed_volumes");
  parse_option<std::vector<std::vector<double>>>(qed_polvecs, jscf, "qed_polvecs");


  json jscf_guess  = jscf["guess"];
  json jguess_atom_options = jscf_guess["atom_options"];
  parse_option<bool>  (sad, jscf_guess, "sad");

  for(auto& [element_symbol, atom_opt]: jguess_atom_options.items()) {
    guess_atom_options[element_symbol] = atom_opt;
  }  

  json jscf_analysis = jscf["PRINT"];
  parse_option<bool> (mos_txt          , jscf_analysis, "mos_txt");
  parse_option<bool> (mulliken_analysis, jscf_analysis, "mulliken");
  parse_option<std::pair<bool, double>>(mo_vectors_analysis, jscf_analysis, "mo_vectors");
    // clang-format on
    std::string riscf_str;
    parse_option<std::string>(riscf_str, jscf, "riscf");
    if(riscf_str == "J") riscf = 1;
    else if(riscf_str == "K") riscf = 2;
    // clang-format off
  const std::vector<std::string> valid_scf{"charge", "multiplicity", "lshift", "tol_int", "tol_sch",
    "tol_lindep", "conve", "convd", "diis_hist","force_tilesize","tilesize","df_tilesize",
    "damp","writem","nnodes","restart","noscf","moldenfile", "guess",
    "debug","scf_type","xc_type", "xc_grid_type", "n_lindep","restart_size","scalapack_nb","riscf",
    "scalapack_np_row","scalapack_np_col","ext_data_path","PRINT",
    "qed_omegas","qed_lambdas","qed_volumes","qed_polvecs","comments"};
    // clang-format on

    for(auto& el: jscf.items()) {
      if(std::find(valid_scf.begin(), valid_scf.end(), el.key()) == valid_scf.end())
        tamm_terminate("INPUT FILE ERROR: Invalid SCF option [" + el.key() + "] in the input file");
    }
    if(nnodes < 1 || nnodes > 100) {
      tamm_terminate("INPUT FILE ERROR: SCF option nnodes should be a number between 1 and 100");
    }
    {
      auto xc_grid_str = xc_grid_type;
      xc_grid_str.erase(remove_if(xc_grid_str.begin(), xc_grid_str.end(), isspace),
                        xc_grid_str.end());
      xc_grid_type = xc_grid_str;
      std::transform(xc_grid_str.begin(), xc_grid_str.end(), xc_grid_str.begin(), ::tolower);
      if(xc_grid_str != "fine" && xc_grid_str != "ultrafine" && xc_grid_str != "superfine")
        tamm_terminate("INPUT FILE ERROR: SCF option xc_grid_type should be one of [Fine, "
                       "UltraFine, SuperFine]");
    }
  } // end of parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

  void print() {
    std::cout << std::defaultfloat;
    std::cout << std::endl << "SCF COptions" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << " charge       = " << charge << std::endl;
    std::cout << " multiplicity = " << multiplicity << std::endl;
    std::cout << " level shift  = " << lshift << std::endl;
    std::cout << " tol_int      = " << tol_int << std::endl;
    std::cout << " tol_sch      = " << tol_sch << std::endl;
    std::cout << " tol_lindep   = " << tol_lindep << std::endl;
    std::cout << " conve        = " << conve << std::endl;
    std::cout << " convd        = " << convd << std::endl;
    std::cout << " diis_hist    = " << diis_hist << std::endl;
    std::cout << " AO_tilesize  = " << AO_tilesize << std::endl;
    std::cout << " writem       = " << writem << std::endl;
    std::cout << " damp         = " << damp << std::endl;
    if(!moldenfile.empty()) {
      std::cout << " moldenfile   = " << moldenfile << std::endl;
      // std::cout << " n_lindep = " << n_lindep <<  std::endl;
    }

    std::cout << " scf_type     = " << scf_type << std::endl;

    // QED
    if(!qed_omegas.empty()) {
      std::cout << " qed_omegas  = [";
      for(auto x: qed_omegas) { std::cout << x << ","; }
      std::cout << "\b]" << std::endl;
    }

    if(!qed_lambdas.empty()) {
      std::cout << " qed_lambdas  = [";
      for(auto x: qed_lambdas) { std::cout << x << ","; }
      std::cout << "\b]" << std::endl;
    }

    if(!qed_volumes.empty()) {
      std::cout << " qed_volumes  = [";
      for(auto x: qed_volumes) { std::cout << x << ","; }
      std::cout << "\b]" << std::endl;
    }

    if(!qed_polvecs.empty()) {
      std::cout << " qed_polvecs  = [";
      for(auto x: qed_polvecs) {
        std::cout << "[";
        for(auto y: x) { std::cout << y << ","; }
        std::cout << "\b],";
      }
      std::cout << "\b]" << std::endl;
    }

    if(!xc_type.empty()) {
      std::cout << " xc_type      = [ ";
      for(auto xcfunc: xc_type) { std::cout << " \"" << xcfunc << "\","; }
      std::cout << "\b ]" << std::endl;
      std::cout << " xc_grid_type = " << xc_grid_type << std::endl;
    }

    if(scalapack_np_row > 0 && scalapack_np_col > 0) {
      std::cout << " scalapack_np_row = " << scalapack_np_row << std::endl;
      std::cout << " scalapack_np_col = " << scalapack_np_col << std::endl;
      if(scalapack_nb > 1) std::cout << " scalapack_nb = " << scalapack_nb << std::endl;
    }
    std::cout << " restart_size = " << restart_size << std::endl;
    caseChange.print_bool(" restart     ", restart);
    caseChange.print_bool(" debug       ", debug);
    if(restart) caseChange.print_bool(" noscf       ", noscf);
    // caseChange.print_bool(" sad         ", sad);
    if(mulliken_analysis || mos_txt || mo_vectors_analysis.first) {
      std::cout << " PRINT {" << std::endl;
      if(mos_txt) std::cout << std::boolalpha << "  mos_txt             = " << mos_txt << std::endl;
      if(mulliken_analysis)
        std::cout << std::boolalpha << "  mulliken_analysis   = " << mulliken_analysis << std::endl;
      if(mo_vectors_analysis.first) {
        std::cout << "  mo_vectors_analysis = [" << std::boolalpha << mo_vectors_analysis.first;
        std::cout << "," << mo_vectors_analysis.second << "]" << std::endl;
      }
      std::cout << " }" << std::endl;
    }
    std::cout << "}" << std::endl;
  }
};