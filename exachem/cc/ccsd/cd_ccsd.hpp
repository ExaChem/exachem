/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"

namespace exachem::cc::ccsd {

void cd_ccsd_driver(tamm::ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::cc::ccsd
