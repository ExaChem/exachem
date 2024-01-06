class COptions {
public:
  COptions() { initialize(); }

  COptions(json& jinput, std::vector<ECAtom>& ec_atoms_, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms_, atoms_);
  }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms_, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms_, atoms_);
  }

  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  COptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }
  void initialize() {
    maxiter       = 50;
    debug         = false;
    basis         = "sto-3g";
    geom_units    = "angstrom";
    gaussian_type = "spherical";
  }

  bool                       debug;
  int                        maxiter;
  std::string                basis;
  std::string                dfbasis{};
  std::string                basisfile{}; // supports only ECPs for now
  std::string                gaussian_type;
  std::string                geom_units;
  std::string                output_file_prefix{};
  std::string                ext_data_path{};
  std::vector<libint2::Atom> atoms;
  std::vector<ECAtom>        ec_atoms;
  txtutils                   caseChange;

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms_, std::vector<Atom>& atoms_) {
    parse_option<std::string>(geom_units, jinput["geometry"], "units");

    const std::vector<std::string> valid_sections{
      "geometry", "basis", "common", "SCF", "CD", "GW", "CC", "FCI", "TASK", "comments"};
    for(auto& el: jinput.items()) {
      if(std::find(valid_sections.begin(), valid_sections.end(), el.key()) == valid_sections.end())
        tamm_terminate("INPUT FILE ERROR: Invalid section [" + el.key() + "] in the input file");
    }

    // basis
    json jbasis = jinput["basis"];
    parse_option<std::string>(basis, jbasis, "basisset", false);
    parse_option<std::string>(basisfile, jbasis, "basisfile");
    // parse_option<std::string>(gaussian_type, jbasis, "gaussian_type");
    parse_option<std::string>(dfbasis, jbasis, "df_basisset");
    txtutils caseChange;
    caseChange.to_lower(basis);
    const std::vector<std::string> valid_basis{"comments", "basisset", "basisfile",
                                               /*"gaussian_type",*/ "df_basisset", "atom_basis"};
    for(auto& el: jbasis.items()) {
      if(std::find(valid_basis.begin(), valid_basis.end(), el.key()) == valid_basis.end())
        tamm_terminate("INPUT FILE ERROR: Invalid basis section option [" + el.key() +
                       "] in the input file");
    }

    json                               jatom_basis = jinput["basis"]["atom_basis"];
    std::map<std::string, std::string> atom_basis_map;
    for(auto& [element_symbol, basis_string]: jatom_basis.items()) {
      atom_basis_map[element_symbol] = basis_string;
    }

    for(size_t i = 0; i < ec_atoms_.size(); i++) {
      const auto es      = ec_atoms_[i].esymbol; // element_symbol
      ec_atoms_[i].basis = basis;
      if(atom_basis_map.find(es) != atom_basis_map.end()) ec_atoms_[i].basis = atom_basis_map[es];
    }

    // common
    json jcommon = jinput["common"];
    parse_option<int>(maxiter, jcommon, "maxiter");
    parse_option<bool>(debug, jcommon, "debug");
    parse_option<std::string>(output_file_prefix, jcommon, "output_file_prefix");

    const std::vector<std::string> valid_common{"comments", "maxiter", "debug",
                                                "output_file_prefix"};
    for(auto& el: jcommon.items()) {
      if(std::find(valid_common.begin(), valid_common.end(), el.key()) == valid_common.end()) {
        tamm_terminate("INPUT FILE ERROR: Invalid common section option [" + el.key() +
                       "] in the input file");
      }
    }
    ec_atoms = ec_atoms_;
    atoms    = atoms_;
  } // end of parse

  void print() {
    std::cout << std::defaultfloat;
    std::cout << std::endl << "Common COptions" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << " maxiter    = " << maxiter << std::endl;
    std::cout << " basis      = " << basis << " ";
    std::cout << gaussian_type;
    std::cout << std::endl;
    if(!dfbasis.empty()) std::cout << " dfbasis    = " << dfbasis << std::endl;
    if(!basisfile.empty()) std::cout << " basisfile  = " << basisfile << std::endl;
    std::cout << " geom_units = " << geom_units << std::endl;
    caseChange.print_bool(" debug     ", debug);
    if(!output_file_prefix.empty())
      std::cout << " output_file_prefix    = " << output_file_prefix << std::endl;
    std::cout << "}" << std::endl;
  }
};