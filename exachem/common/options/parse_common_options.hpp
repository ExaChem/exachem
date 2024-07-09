/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/chemenv.hpp"
#include "exachem/common/options/parser_utils.hpp"

class ParseCommonOptions: public ParserUtils {
private:
  void parse(ChemEnv& chem_env);

public:
  ParseCommonOptions() = default;
  ParseCommonOptions(ChemEnv& chem_env);
  void operator()(ChemEnv& chem_env);
  void print(ChemEnv& chem_env);
};
