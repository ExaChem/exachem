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
  const std::vector<string> valid_cd{"comments", "debug",        "itilesize", "diagtol",
                                     "write_cv", "write_vcount", "max_cvecs", "ext_data_path"};
  for(auto& el: jinput["CD"].items()) {
    if(std::find(valid_cd.begin(), valid_cd.end(), el.key()) == valid_cd.end())
      tamm_terminate("INPUT FILE ERROR: Invalid CD option [" + el.key() + "] in the input file");
  }
}

void ParseCDOptions::parse(ChemEnv& chem_env) {
  json jcd = chem_env.jinput["CD"];
  parse_option<bool>(chem_env.ioptions.cd_options.debug, jcd, "debug");
  parse_option<int>(chem_env.ioptions.cd_options.itilesize, jcd, "itilesize");
  parse_option<double>(chem_env.ioptions.cd_options.diagtol, jcd, "diagtol");
  parse_option<bool>(chem_env.ioptions.cd_options.write_cv, jcd, "write_cv");
  parse_option<int>(chem_env.ioptions.cd_options.write_vcount, jcd, "write_vcount");
  parse_option<int>(chem_env.ioptions.cd_options.max_cvecs_factor, jcd, "max_cvecs");
  parse_option<string>(chem_env.ioptions.cd_options.ext_data_path, jcd, "ext_data_path");
}

void ParseCDOptions::print(ChemEnv& chem_env) {
  std::cout << std::defaultfloat;
  std::cout << std::endl << "CD Options" << std::endl;
  std::cout << "{" << std::endl;
  std::cout << std::boolalpha << " debug            = " << chem_env.ioptions.cd_options.debug
            << std::endl;
  std::cout << std::boolalpha << " write_cv         = " << chem_env.ioptions.cd_options.write_cv
            << std::endl;
  std::cout << " diagtol          = " << chem_env.ioptions.cd_options.diagtol << std::endl;
  std::cout << " write_vcount     = " << chem_env.ioptions.cd_options.write_vcount << std::endl;
  std::cout << " itilesize        = " << chem_env.ioptions.cd_options.itilesize << std::endl;
  std::cout << " max_cvecs_factor = " << chem_env.ioptions.cd_options.max_cvecs_factor << std::endl;
  std::cout << "}" << std::endl;
}