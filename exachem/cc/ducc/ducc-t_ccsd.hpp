/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cholesky/cholesky_2e.hpp"
#include "exachem/cholesky/v2tensors.hpp"

namespace exachem::cc::ducc {
void ducc_driver(ExecutionContext& ec, ChemEnv& chem_env);

template<typename T>
void DUCC_T_CCSD_Driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                        Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1,
                        cholesky_2e::V2Tensors<T>& v2tensors, size_t nactv,
                        ExecutionHW ex_hw = ExecutionHW::CPU);
} // namespace exachem::cc::ducc
