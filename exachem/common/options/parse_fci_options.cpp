/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_fci_options.hpp"

ParseFCIOptions::ParseFCIOptions(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseFCIOptions::operator()(ChemEnv& chem_env) {
  parse_check(chem_env.jinput);
  parse(chem_env);
}

void ParseFCIOptions::parse_check(json& jinput) {
  if(!jinput.contains("FCI")) return;
  // clang-format off
    const std::vector<std::string> valid_fci{"nalpha",   "nbeta", "nactive",   "ninactive",
                                             "comments", "job",   "expansion", "FCIDUMP",
                                             "MCSCF",    "PRINT"};

    for(auto& el: jinput["FCI"].items()) {
      if(std::find(valid_fci.begin(), valid_fci.end(), el.key()) == valid_fci.end())
        tamm_terminate("INPUT FILE ERROR: Invalid FCI option [" + el.key() + "] in the input file");
    }
  // clang-format on    
}

void ParseFCIOptions::parse(ChemEnv& chem_env){
  if(!chem_env.jinput.contains("FCI")) return;
  json jfci = chem_env.jinput["FCI"];
  FCIOptions& fci_options = chem_env.ioptions.fci_options;
  update_common_options(chem_env);

  // clang-format off
  parse_option<int>(fci_options.nalpha,       jfci, "nalpha");
  parse_option<int>(fci_options.nbeta,        jfci, "nbeta");
  parse_option<int>(fci_options.nactive,      jfci, "nactive");
  parse_option<int>(fci_options.ninactive,    jfci, "ninactive");
  parse_option<std::string>(fci_options.job,       jfci, "job");
  parse_option<std::string>(fci_options.expansion, jfci, "expansion");


  // MCSCF
  json jmcscf = jfci["MCSCF"];
  parse_option<int>   (fci_options.max_macro_iter,     jmcscf, "max_macro_iter");
  parse_option<double>(fci_options.max_orbital_step,   jmcscf, "max_orbital_step");
  parse_option<double>(fci_options.orb_grad_tol_mcscf, jmcscf, "orb_grad_tol_mcscf");
  parse_option<bool>  (fci_options.enable_diis,        jmcscf, "enable_diis");
  parse_option<int>   (fci_options.diis_start_iter,    jmcscf, "diis_start_iter");
  parse_option<int>   (fci_options.diis_nkeep,         jmcscf, "diis_nkeep");
  parse_option<double>(fci_options.ci_res_tol,         jmcscf, "ci_res_tol");
  parse_option<int>   (fci_options.ci_max_subspace,    jmcscf, "ci_max_subspace");
  parse_option<double>(fci_options.ci_matel_tol,       jmcscf, "ci_matel_tol");

  // FCIDUMP
  json jfcidump = jfci["FCIDUMP"];
  // parse_option<bool>(fci_options.fcidump,    jfcidump, "fcidump");

  // PRINT
  json jprint = jfci["PRINT"];
  parse_option<bool>(fci_options.print_davidson,    jprint, "davidson");
  parse_option<bool>(fci_options.print_ci,          jprint, "ci");
  parse_option<bool>(fci_options.print_mcscf,       jprint, "mcscf");
  parse_option<bool>(fci_options.print_diis,        jprint, "diis");
  parse_option<bool>(fci_options.print_asci_search, jprint, "asci_search");
  parse_option<std::pair<bool, double>>(fci_options.print_state_char, jprint, "state_char");
  // clang-format on
}

void ParseFCIOptions::update_common_options(ChemEnv& chem_env) {
  FCIOptions&    fci_options    = chem_env.ioptions.fci_options;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  fci_options.debug         = common_options.debug;
  fci_options.maxiter       = common_options.maxiter;
  fci_options.basis         = common_options.basis;
  fci_options.dfbasis       = common_options.dfbasis;
  fci_options.gaussian_type = common_options.gaussian_type;
  fci_options.geom_units    = common_options.geom_units;
  fci_options.file_prefix   = common_options.file_prefix;
  fci_options.output_dir    = common_options.output_dir;
  fci_options.ext_data_path = common_options.ext_data_path;
}
