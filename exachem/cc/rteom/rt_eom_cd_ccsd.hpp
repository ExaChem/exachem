/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include "exachem/common/chemenv.hpp"

namespace exachem::rteom_cc::ccsd {
void rt_eom_cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::rteom_cc::ccsd
