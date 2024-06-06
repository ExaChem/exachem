/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "common/chemenv.hpp"
#include "common/options/input_options.hpp"
#include "common/options/parser_utils.hpp"

class ParseFCIOptions: public ParserUtils {
private:
  void parse_check(json& jinput);
  void parse(ChemEnv& chem_env);
  void update_common_options(ChemEnv& chem_env);

public:
  ParseFCIOptions() = default;
  ParseFCIOptions(ChemEnv& chem_env);
  void operator()(ChemEnv& chem_env);
  // void print(ChemEnv& chem_env);
};

// Have to add print() functionls