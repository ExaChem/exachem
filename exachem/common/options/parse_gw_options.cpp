/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "parse_gw_options.hpp"

ParseGWOptions::ParseGWOptions(ChemEnv& chem_env) { parse(chem_env); }

void ParseGWOptions::parse(ChemEnv& chem_env) {
  // clang-format off
  json jgw = chem_env.jinput["GW"];
  GWOptions& gw_options = chem_env.ioptions.gw_options;
  update_common_options(chem_env);

  parse_option<bool>  (gw_options.debug,     jgw, "debug");
  parse_option<int>   (gw_options.ngl,       jgw, "ngl");
  parse_option<int>   (gw_options.noqpa,     jgw, "noqpa");
  parse_option<int>   (gw_options.noqpb,     jgw, "noqpb");
  parse_option<int>   (gw_options.nvqpa,     jgw, "nvqpa");
  parse_option<int>   (gw_options.nvqpb,     jgw, "nvqpb");
  parse_option<double>(gw_options.ieta,      jgw, "ieta");
  parse_option<bool>  (gw_options.evgw,      jgw, "evgw");
  parse_option<bool>  (gw_options.evgw0,     jgw, "evgw0");
  parse_option<bool>  (gw_options.core,      jgw, "core");
  parse_option<int>   (gw_options.maxev,     jgw, "maxev");
  parse_option<bool>  (gw_options.minres,    jgw, "minres");
  parse_option<int>   (gw_options.maxnewton, jgw, "maxnewton");
  parse_option<std::string>(gw_options.method,    jgw, "method");
  parse_option<std::string>(gw_options.cdbasis,   jgw, "cdbasis");
  parse_option<std::string>(gw_options.ext_data_path, jgw, "ext_data_path");
  // clang-format on
  std::vector<std::string> gwlist{"cdgw", "sdgw", "CDGW", "SDGW"};
  if(std::find(std::begin(gwlist), std::end(gwlist), string(gw_options.method)) == std::end(gwlist))
    tamm_terminate("INPUT FILE ERROR: GW method can only be one of [sdgw,cdgw]");
}

void ParseGWOptions::update_common_options(ChemEnv& chem_env) {
  GWOptions&     gw_options     = chem_env.ioptions.gw_options;
  CommonOptions& common_options = chem_env.ioptions.common_options;

  gw_options.debug         = common_options.debug;
  gw_options.maxiter       = common_options.maxiter;
  gw_options.basis         = common_options.basis;
  gw_options.dfbasis       = common_options.dfbasis;
  gw_options.basisfile     = common_options.basisfile;
  gw_options.gaussian_type = common_options.gaussian_type;
  gw_options.geom_units    = common_options.geom_units;
  gw_options.file_prefix   = common_options.file_prefix;
  gw_options.output_dir    = common_options.output_dir;

  gw_options.ext_data_path = common_options.ext_data_path;
}
