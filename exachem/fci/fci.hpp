/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/fcidump.hpp"

namespace exachem::fci {

template<typename T>
std::string generate_fcidump(ChemEnv& chem_env, tamm::ExecutionContext& ec,
                             const TiledIndexSpace& MSO, tamm::Tensor<T>& lcao,
                             tamm::Tensor<T>& d_f1, tamm::Tensor<T>& full_v2,
                             ExecutionHW ex_hw = ExecutionHW::CPU);

void fci_driver(tamm::ExecutionContext& ec, ChemEnv& chem_env);
} // namespace exachem::fci
