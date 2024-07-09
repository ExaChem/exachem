/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
// clang-format off
#include "parser_utils.hpp"
#include "parse_common_options.hpp"
#include "parse_cd_options.hpp"
#include "parse_ccsd_options.hpp"
#include "parse_fci_options.hpp"
#include "parse_gw_options.hpp"
#include "parse_scf_options.hpp"
#include "parse_task_options.hpp"
#include "exachem/common/initialize_system_data.hpp"
// clang-format on

// This class populates all the input options.
class ECOptionParser: public ParserUtils {
private:
  void parse_n_check(std::string_view filename, json& jinput);

public:
  ECOptionParser() = default;
  ECOptionParser(ChemEnv& chem_env);
  void initialize(ChemEnv& chem_env);
  void parse_all_options(ChemEnv& chem_env);
};
