/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "common/chemenv.hpp"
#include "common/txt_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;
using json   = nlohmann::ordered_json;

class IniSystemData {
public:
  IniSystemData(ChemEnv& chem_env) { initialize(chem_env); }
  void operator()(ChemEnv& chem_env) { initialize(chem_env); }
  void initialize(ChemEnv& chem_env);
};
