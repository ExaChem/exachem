// clang-format off
#include "common/parser_options.hpp"
#include "common_options.hpp"
#include "cc_options.hpp"
#include "cd_options.hpp"
#include "fci_options.hpp"
#include "gw_options.hpp"
#include "scf_options.hpp"
#include "task_options.hpp"
// clang-format on

class ECOptions {
public:
  ECOptions() = default;
  ECOptions(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    initialize(jinput, ec_atoms, atoms_);
  }

  ECOptions(ECParse& parse) { initialize(parse.jinput, parse.ec_atoms, parse.atoms); }

  void operator()(ECParse& parse) { initialize(parse.jinput, parse.ec_atoms, parse.atoms); }

  void initialize(json& jinput, std::vector<ECAtom>& ec_atoms, std::vector<Atom>& atoms_) {
    options(jinput, ec_atoms, atoms_);
    scf_options(jinput, ec_atoms, atoms_);
    cd_options(jinput, ec_atoms, atoms_);
    gw_options(jinput, ec_atoms, atoms_);
    ccsd_options(jinput, ec_atoms, atoms_);
    fci_options(jinput, ec_atoms, atoms_);
    task_options(jinput, ec_atoms, atoms_);
  }

  COptions    options;
  SCFOptions  scf_options;
  CDOptions   cd_options;
  GWOptions   gw_options;
  CCSDOptions ccsd_options;
  FCIOptions  fci_options;
  TaskOptions task_options;
};