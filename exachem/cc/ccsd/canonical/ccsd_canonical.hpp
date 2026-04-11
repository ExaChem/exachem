/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cholesky/cholesky_2e_driver.hpp"
using namespace tamm;

namespace exachem::cc::ccsd_canonical {
void ccsd_canonical_driver(ExecutionContext& ec, ChemEnv& chem_env);
}; // namespace exachem::cc::ccsd_canonical
