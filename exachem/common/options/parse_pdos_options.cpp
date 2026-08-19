/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_pdos_options.hpp"

ParsePDOSOptions::ParsePDOSOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParsePDOSOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParsePDOSOptions::parse_check(json& jinput) {
  if(!jinput.contains("PDOS")) return;
  // clang-format off
  const std::vector<std::string> valid_pdos{"emin", "emax", "npoints", "fwhm", "distribution", "modified"};
  // clang-format on

  for(auto& el: jinput["PDOS"].items()) {
    if(std::find(valid_pdos.begin(), valid_pdos.end(), el.key()) == valid_pdos.end())
      tamm_terminate("INPUT FILE ERROR: Invalid PDOS option [" + el.key() + "] in the input file");
  }
}

void ParsePDOSOptions::parse(ChemEnv& chem_env) {
  if(!chem_env.jinput.contains("PDOS")) return;
  json         jpdos        = chem_env.jinput["PDOS"];
  PDOSOptions& pdos_options = chem_env.ioptions.pdos_options;

  parse_option<double>(pdos_options.emin, jpdos, "emin");
  parse_option<double>(pdos_options.emax, jpdos, "emax");
  parse_option<double>(pdos_options.fwhm, jpdos, "fwhm");
  parse_option<bool>(pdos_options.do_mod, jpdos, "modified");
  parse_option<std::string>(pdos_options.distribution, jpdos, "distribution");

  parse_option<size_t>(pdos_options.npoints, jpdos, "npoints");

  if(pdos_options.npoints > 0) pdos_options.do_pdos = true;
}