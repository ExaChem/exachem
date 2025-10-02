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

std::map<std::string, std::string> get_atom_basis_map(ChemEnv& chem_env, const json& jatom_basis) {
  std::map<std::string, std::string> atom_basis_map;
  for(auto& [element_symbol, basis_string]: jatom_basis.items()) {
    std::string _basis_string = basis_string;
    txt_utils::to_lower(_basis_string);
    if(_basis_string.find("aug") == 0) { _basis_string = std::string("ec-") + _basis_string; }
    std::replace(_basis_string.begin(), _basis_string.end(), ' ', '_');
    atom_basis_map[element_symbol] = _basis_string;
  }
  return atom_basis_map;
}

void ParseCommonOptions::parse(ChemEnv& chem_env) {
  json           jinput         = chem_env.jinput;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  parse_option<std::string>(common_options.geom_units, jinput["geometry"], "units");
  parse_option<int>(common_options.natoms_max, jinput["geometry"]["analysis"], "natoms_max");

  // basis

  parse_option<std::string>(common_options.basis, jinput["basis"], "basisset", false);
  // parse_option<std::string>(gaussian_type, jbasis, "gaussian_type");
  parse_option<std::string>(common_options.dfbasis, jinput["basis"], "df_basisset");

  txt_utils::to_lower(common_options.basis);
  std::replace(common_options.basis.begin(), common_options.basis.end(), ' ', '_');
  if(common_options.basis.find("aug") == 0) {
    common_options.basis = std::string("ec-") + common_options.basis;
  }

  json                               jatom_basis    = jinput["basis"]["atom_basis"];
  std::map<std::string, std::string> atom_basis_map = get_atom_basis_map(chem_env, jatom_basis);
  for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
    const auto es              = chem_env.ec_atoms[i].esymbol; // element_symbol
    chem_env.ec_atoms[i].basis = common_options.basis;
    if(atom_basis_map.find(es) != atom_basis_map.end())
      chem_env.ec_atoms[i].basis = atom_basis_map[es];
  }

  json                               jatom_ecp    = jinput["basis"]["atom_ecp"];
  std::map<std::string, std::string> atom_ecp_map = get_atom_basis_map(chem_env, jatom_ecp);
  for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
    const auto es = chem_env.ec_atoms[i].esymbol; // element_symbol
    if(atom_ecp_map.find(es) != atom_ecp_map.end())
      chem_env.ec_atoms[i].ecp_basis = atom_ecp_map[es];
  }

  // common
  if(jinput.contains("common")) {
    parse_option<int>(common_options.maxiter, jinput["common"], "maxiter");
    parse_option<bool>(common_options.debug, jinput["common"], "debug");
    parse_option<std::string>(common_options.file_prefix, jinput["common"], "file_prefix");
    parse_option<std::string>(common_options.output_dir, jinput["common"], "output_dir");
  }

  // parse cube options here for now
  if(jinput.contains("DPLOT")) {
    parse_option<bool>(chem_env.ioptions.dplot_options.cube, jinput["DPLOT"], "cube");
    parse_option<std::string>(chem_env.ioptions.dplot_options.density, jinput["DPLOT"], "density");
    parse_option<int>(chem_env.ioptions.dplot_options.orbitals, jinput["DPLOT"], "orbitals");
  }
}
