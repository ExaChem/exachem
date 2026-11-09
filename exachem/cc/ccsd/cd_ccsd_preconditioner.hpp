/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"

namespace exachem::cc::ccsd {

// Solve the singles/doubles Sylvester equations A_F z = rhs, preserving the input layout.
// Retained for future noncanonical-gauge support; current CCSD callbacks use Jacobi.
template<typename T>
std::vector<Tensor<T>> sylvester_preconditioner(ExecutionContext& ec, ChemEnv& chem_env,
                                                Tensor<T>& d_e, const Tensor<T>& fock,
                                                const std::vector<Tensor<T>>& rhs);

} // namespace exachem::cc::ccsd
