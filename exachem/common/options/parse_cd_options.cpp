/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_cd_options.hpp"

void ParseCDOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

ParseCDOptions::ParseCDOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseCDOptions::parse_check(json& jinput) {
  if(!jinput.contains("CD")) return;
  const std::vector<string> valid_cd{"comments", "debug",   "itilesize", "diagtol",
                                     "write_cv", "skip_cd", "max_cvecs", "ext_data_path"};
  for(auto& el: jinput["CD"].items()) {
    if(std::find(valid_cd.begin(), valid_cd.end(), el.key()) == valid_cd.end())
      tamm_terminate("INPUT FILE ERROR: Invalid CD option [" + el.key() + "] in the input file");
  }
}

void ParseCDOptions::parse(ChemEnv& chem_env) {
  if(!chem_env.jinput.contains("CD")) return;
  json jcd = chem_env.jinput["CD"];
  parse_option<bool>(chem_env.ioptions.cd_options.debug, jcd, "debug");
  parse_option<int>(chem_env.ioptions.cd_options.itilesize, jcd, "itilesize");
  parse_option<double>(chem_env.ioptions.cd_options.diagtol, jcd, "diagtol");
  parse_option<std::pair<bool, int>>(chem_env.ioptions.cd_options.skip_cd, jcd, "skip_cd");
  parse_option<std::pair<bool, int>>(chem_env.ioptions.cd_options.write_cv, jcd, "write_cv");
  parse_option<int>(chem_env.ioptions.cd_options.max_cvecs_factor, jcd, "max_cvecs");
  parse_option<string>(chem_env.ioptions.cd_options.ext_data_path, jcd, "ext_data_path");
}
