
class CDOptions: public COptions {
public:
  CDOptions(): COptions() { initialize(); }

  CDOptions(COptions o): COptions(o) { initialize(); }

  CDOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  CDOptions(COptions o, json& jinput): COptions(o) {
    initialize();
    parse(jinput);
  }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  CDOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void initialize() {
    diagtol      = 1e-5;
    write_cv     = false;
    write_vcount = 5000;
    // At most 8*ao CholVec's. For vast majority cases, this is way
    // more than enough. For very large basis, it can be increased.
    max_cvecs_factor = 12;
    itilesize        = 1000;
  }

  double diagtol;
  int    itilesize;
  int    max_cvecs_factor;
  // write to disk after every count number of vectors are computed.
  // enabled only if write_cv=true and nbf>1000
  bool write_cv;
  int  write_vcount;

  void parse(json& jinput) {
    json jcd = jinput["CD"];
    parse_option<bool>(debug, jcd, "debug");
    parse_option<int>(itilesize, jcd, "itilesize");
    parse_option<double>(diagtol, jcd, "diagtol");
    parse_option<bool>(write_cv, jcd, "write_cv");
    parse_option<int>(write_vcount, jcd, "write_vcount");
    parse_option<int>(max_cvecs_factor, jcd, "max_cvecs");

    parse_option<string>(ext_data_path, jcd, "ext_data_path");

    const std::vector<string> valid_cd{"comments", "debug",        "itilesize", "diagtol",
                                       "write_cv", "write_vcount", "max_cvecs", "ext_data_path"};
    for(auto& el: jcd.items()) {
      if(std::find(valid_cd.begin(), valid_cd.end(), el.key()) == valid_cd.end())
        tamm_terminate("INPUT FILE ERROR: Invalid CD option [" + el.key() + "] in the input file");
    }

  } // parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

  void print() {
    std::cout << std::defaultfloat;
    std::cout << std::endl << "CD COptions" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << std::boolalpha << " debug            = " << debug << std::endl;
    std::cout << std::boolalpha << " write_cv         = " << write_cv << std::endl;
    std::cout << " diagtol          = " << diagtol << std::endl;
    std::cout << " write_vcount     = " << write_vcount << std::endl;
    std::cout << " itilesize        = " << itilesize << std::endl;
    std::cout << " max_cvecs_factor = " << max_cvecs_factor << std::endl;
    std::cout << "}" << std::endl;
  }
};