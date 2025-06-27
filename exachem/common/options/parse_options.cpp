/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/options/parse_options.hpp"

ECOptionParser::ECOptionParser(ChemEnv& chem_env) {
  initialize(chem_env);
  parse_all_options(chem_env);
}

void ECOptionParser::parse_n_check(std::string_view filename, json& jinput) {
  if(std::filesystem::path(std::string(filename)).extension() != ".json") {
    tamm_terminate("ERROR: Input file provided [" + std::string(filename) +
                   "] must be a json file");
  }

  auto is = std::ifstream(std::string(filename));

  auto jsax =
    nlohmann::detail::json_sax_dom_parser<json, nlohmann::detail::string_input_adapter_type>(jinput,
                                                                                             false);
  bool parse_result = json::sax_parse(is, &jsax);
  if(!parse_result) tamm_terminate("Error parsing input file");

  const std::vector<std::string> valid_sections{
    "geometry", "basis", "common", "DPLOT", "SCF", "CD", "GW", "CC", "FCI", "TASK", "comments"};
  for(auto& el: jinput.items()) {
    if(std::find(valid_sections.begin(), valid_sections.end(), el.key()) == valid_sections.end())
      tamm_terminate("INPUT FILE ERROR: Invalid section [" + el.key() + "] in the input file");
  }

  json jbasis = jinput["basis"];

  const std::vector<std::string> valid_basis{"comments", "basisset", "basisfile",
                                             /*"gaussian_type",*/ "df_basisset", "atom_basis"};
  for(auto& el: jbasis.items()) {
    if(std::find(valid_basis.begin(), valid_basis.end(), el.key()) == valid_basis.end())
      tamm_terminate("INPUT FILE ERROR: Invalid basis section option [" + el.key() +
                     "] in the input file");
  }

  const std::vector<std::string> valid_common{"comments", "maxiter", "debug", "file_prefix",
                                              "output_dir"};
  if(jinput.contains("common")) {
    for(auto& el: jinput["common"].items()) {
      if(std::find(valid_common.begin(), valid_common.end(), el.key()) == valid_common.end()) {
        tamm_terminate("INPUT FILE ERROR: Invalid common section option [" + el.key() +
                       "] in the input file");
      }
    }
  }
}

void ECOptionParser::initialize(ChemEnv& chem_env) {
  json jinput;
  parse_n_check(chem_env.input_file, jinput);

  chem_env.jinput = jinput;

  std::vector<string> geometry;
  // std::vector<string> geom_bohr;
  constexpr double ang2bohr = exachem::constants::ang2bohr;
  std::string      geom_units{"angstrom"};

  parse_option<string>(geom_units, jinput["geometry"], "units");
  parse_option<std::vector<string>>(geometry, jinput["geometry"], "coordinates", false);
  size_t natom = geometry.size();

  chem_env.ec_atoms.resize(natom);
  chem_env.atoms.resize(natom);
  // geom_bohr.resize(natom);

  double convert_units = (geom_units == "angstrom") ? ang2bohr : 1.0;

  for(size_t i = 0; i < natom; i++) {
    std::istringstream iss(geometry[i]);
    std::string        element_symbol;
    double             x, y, z;
    iss >> element_symbol >> x >> y >> z;
    // geom_bohr[i] = element_symbol;

    const auto Z                 = ECAtom::get_atomic_number(element_symbol);
    chem_env.atoms[i]            = {Z, x * convert_units, y * convert_units, z * convert_units};
    chem_env.ec_atoms[i].atom    = chem_env.atoms[i];
    chem_env.ec_atoms[i].esymbol = element_symbol;

    if(txt_utils::strequal_case(element_symbol.substr(0, 2), "bq")) {
      chem_env.ec_atoms[i].is_bq = true;
    }

    // ss_bohr << std::setw(3) << std::left << geom_bohr[i] << " " << std::right << std::setw(14)
    //         << std::fixed << std::setprecision(10) << atoms[i].x << " " << std::right
    //         << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].y << " "
    //         << std::right << std::setw(14) << std::fixed << std::setprecision(10) << atoms[i].z;
    // geom_bohr[i]     = ss_bohr.str();
  }
  // jgeom_bohr["geometry_bohr"] = geom_bohr;
}

void ECOptionParser::parse_all_options(ChemEnv& chem_env) {
  ParseCommonOptions parse_common_options(chem_env);
  ParseSCFOptions    parse_scf_options(chem_env);
  ParseCDOptions     parse_cd_options(chem_env);
  ParseGWOptions     parse_gw_options(chem_env);
  ParseCCSDOptions   parse_ccsd_options(chem_env);
  ParseFCIOptions    parse_fci_options(chem_env);
  ParseTaskOptions   parse_task_options(chem_env);
  IniSystemData      ini_sys_data(chem_env);
}
