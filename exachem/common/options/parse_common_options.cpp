#include "parse_common_options.hpp"

void ParseCOptions::operator()(ChemEnv& chem_env) { parse(chem_env); }

ParseCOptions::ParseCOptions(ChemEnv& chem_env) { parse(chem_env); }

void ParseCOptions::parse(ChemEnv& chem_env) {
  json           jinput         = chem_env.jinput;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  parse_option<std::string>(common_options.geom_units, jinput["geometry"], "units");

  // basis

  parse_option<std::string>(common_options.basis, jinput["basis"], "basisset", false);
  parse_option<std::string>(common_options.basisfile, jinput["basis"], "basisfile");
  // parse_option<std::string>(gaussian_type, jbasis, "gaussian_type");
  parse_option<std::string>(common_options.dfbasis, jinput["basis"], "df_basisset");

  txt_utils::to_lower(common_options.basis);

  json                               jatom_basis = jinput["basis"]["atom_basis"];
  std::map<std::string, std::string> atom_basis_map;
  for(auto& [element_symbol, basis_string]: jatom_basis.items()) {
    atom_basis_map[element_symbol] = basis_string;
  }

  for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
    const auto es              = chem_env.ec_atoms[i].esymbol; // element_symbol
    chem_env.ec_atoms[i].basis = common_options.basis;
    if(atom_basis_map.find(es) != atom_basis_map.end())
      chem_env.ec_atoms[i].basis = atom_basis_map[es];
  }

  // common
  parse_option<int>(common_options.maxiter, jinput["common"], "maxiter");
  parse_option<bool>(common_options.debug, jinput["common"], "debug");
  parse_option<std::string>(common_options.file_prefix, jinput["common"], "file_prefix");
}

void ParseCOptions::print(ChemEnv& chem_env) {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "Common Options" << std::endl;
  std::cout << "{" << std::endl;

  CommonOptions& common_options = chem_env.ioptions.common_options;
  std::cout << " maxiter    = " << common_options.maxiter << std::endl;
  std::cout << " basis      = " << common_options.basis << " ";
  std::cout << common_options.gaussian_type;
  std::cout << std::endl;
  if(!common_options.dfbasis.empty())
    std::cout << " dfbasis    = " << common_options.dfbasis << std::endl;
  if(!common_options.basisfile.empty())
    std::cout << " basisfile  = " << common_options.basisfile << std::endl;
  std::cout << " geom_units = " << common_options.geom_units << std::endl;
  txt_utils::print_bool(" debug     ", common_options.debug);
  if(!common_options.file_prefix.empty())
    std::cout << " file_prefix    = " << common_options.file_prefix << std::endl;
  std::cout << "}" << std::endl;
}