/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

//*******************************************************
// Compute intial guess -> D
// for each iter:
// 1. 2 body fock procedure -> computes G (combined JK)
// 2. [EXC, VXC] = xc_integrator.eval_exc_vxc(D)
// 3. F = H + G
// 4. F += VXC
// 5. E = 0.5 * Tr((H+F) * D)
// 6. E += EXC
// 7. diagonalize F -> updates D
// 8. E += enuc, print E
//*******************************************************

#pragma once

// standard C++ headers
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "common/chemenv.hpp"
#include "scf/scf_taskmap.hpp"

using libint2::BasisSet;

#if defined(USE_GAUXC)
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/xc_integrator/integrator_factory.hpp>
#endif

#include <filesystem>
namespace fs = std::filesystem;

#define SCF_THROTTLE_RESOURCES 1

namespace exachem::scf {
void scf(ExecutionContext& ec, ChemEnv& chem_env);
void scf_driver(ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::scf
