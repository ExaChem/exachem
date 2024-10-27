/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_common_options.hpp"

void ParseCommonOptions::operator()(ChemEnv& chem_env) { parse(chem_env); }

ParseCommonOptions::ParseCommonOptions(ChemEnv& chem_env) { parse(chem_env); }

void ParseCommonOptions::parse(ChemEnv& chem_env) {
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
  parse_option<std::string>(common_options.output_dir, jinput["common"], "output_dir");

  // parse cube options here for now
  parse_option<bool>(chem_env.ioptions.dplot_options.cube, jinput["DPLOT"], "cube");
  parse_option<std::string>(chem_env.ioptions.dplot_options.density, jinput["DPLOT"], "density");
  parse_option<int>(chem_env.ioptions.dplot_options.orbitals, jinput["DPLOT"], "orbitals");
}
