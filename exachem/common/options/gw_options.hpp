
class GWOptions: public COptions {
public:
  GWOptions(): COptions() { initialize(); }

  GWOptions(COptions o): COptions(o) { initialize(); }

  GWOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_): COptions() {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  GWOptions(COptions o, json& jinput): COptions(o) {
    initialize();
    parse(jinput);
  }

  void operator()(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize();
    parse(jinput, ec_atoms, atoms_);
  }

  GWOptions(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }
  void operator()(ECParse& fparse) {
    initialize();
    parse(fparse.jinput, fparse.ec_atoms, fparse.atoms);
  }

  void initialize() {
    cdbasis   = "";
    ngl       = 200;
    noqpa     = 1;
    noqpb     = 1;
    nvqpa     = 0;
    nvqpb     = 0;
    ieta      = 0.01;
    evgw      = false;
    evgw0     = false;
    core      = false;
    maxnewton = 15;
    maxev     = 0;
    minres    = false;
    method    = "sdgw";
  }
  int    ngl;       // Number of Gauss-Legendre quadrature points
  int    noqpa;     // Number of Occupied QP energies ALPHA spi
  int    noqpb;     // Number of Occupied QP energies BETA spin
  int    nvqpa;     // Number of Virtual QP energies ALPHA spin
  int    nvqpb;     // Number of Virtual QP energies BETA spin
  double ieta;      // Imaginary infinitesimal value
  bool   evgw;      // Do an evGW self-consistent calculation
  bool   evgw0;     // Do an evGW_0 self-consistent calculation
  bool   core;      // If true, start counting from the core
  int    maxnewton; // Maximum number of Newton steps per QP
  int    maxev;     // Maximum number of evGW or evGW_0 cycles
  bool   minres;    // Use MINRES solver
  string method;    // Method to use [cdgw,sdgw]
  string cdbasis;   // Name of the CD basis set

  void parse(json& jinput) {
    // clang-format off
  json jgw = jinput["GW"];
  parse_option<bool>  (debug,     jgw, "debug");
  parse_option<int>   (ngl,       jgw, "ngl");
  parse_option<int>   (noqpa,     jgw, "noqpa");
  parse_option<int>   (noqpb,     jgw, "noqpb");
  parse_option<int>   (nvqpa,     jgw, "nvqpa");
  parse_option<int>   (nvqpb,     jgw, "nvqpb");
  parse_option<double>(ieta,      jgw, "ieta");
  parse_option<bool>  (evgw,      jgw, "evgw");
  parse_option<bool>  (evgw0,     jgw, "evgw0");
  parse_option<bool>  (core,      jgw, "core");
  parse_option<int>   (maxev,     jgw, "maxev");
  parse_option<bool>  (minres,    jgw, "minres");
  parse_option<int>   (maxnewton, jgw, "maxnewton");
  parse_option<std::string>(method,    jgw, "method");
  parse_option<std::string>(cdbasis,   jgw, "cdbasis");
  parse_option<std::string>(ext_data_path, jgw, "ext_data_path");
    // clang-format on
    std::vector<std::string> gwlist{"cdgw", "sdgw", "CDGW", "SDGW"};
    if(std::find(std::begin(gwlist), std::end(gwlist), string(method)) == std::end(gwlist))
      tamm_terminate("INPUT FILE ERROR: GW method can only be one of [sdgw,cdgw]");

  } // end of parse

  void parse(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    COptions::parse(jinput, ec_atoms, atoms_);
    parse(jinput);
  }

  void print() {
    std::cout << std::defaultfloat;
    std::cout << std::endl << "GW COptions" << std::endl;
    std::cout << "{" << std::endl;
    std::cout << " ngl       = " << ngl << std::endl;
    std::cout << " noqpa     = " << noqpa << std::endl;
    std::cout << " noqpb     = " << noqpb << std::endl;
    std::cout << " nvqpa     = " << nvqpa << std::endl;
    std::cout << " nvqp/b     = " << nvqpb << std::endl;
    std::cout << " ieta      = " << ieta << std::endl;
    std::cout << " maxnewton = " << maxnewton << std::endl;
    std::cout << " maxev     = " << maxev << std::endl;
    std::cout << " method    = " << method << std::endl;
    std::cout << " cdbasis   = " << cdbasis << std::endl;
    caseChange.print_bool(" evgw     ", evgw);
    caseChange.print_bool(" evgw0    ", evgw0);
    caseChange.print_bool(" core     ", core);
    caseChange.print_bool(" minres   ", minres);
    caseChange.print_bool(" debug    ", debug);
    std::cout << "}" << std::endl;
  }
};