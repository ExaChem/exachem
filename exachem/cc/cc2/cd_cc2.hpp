/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/cc2/cd_cc2_os.hpp"

namespace exachem::cc2 {
void cd_cc2_driver(ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::cc2
