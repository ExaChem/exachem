/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cc/ccse_tensors.hpp"
#include "cc/diis.hpp"
#include "cholesky/v2tensors.hpp"
#include "scf/scf_main.hpp"

namespace exachem::cholesky_2e {

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, TAMM_SIZE, tamm::Tile, TiledIndexSpace>
cholesky_2e_driver(ChemEnv& chem_env, ExecutionContext& ec, TiledIndexSpace& MO,
                   TiledIndexSpace& AO, Tensor<T> C_AO, Tensor<T> F_AO, Tensor<T> C_beta_AO,
                   Tensor<T> F_beta_AO, libint2::BasisSet& shells,
                   std::vector<size_t>& shell_tile_map, bool readv2 = false,
                   std::string cholfile = "", bool is_dlpno = false, bool is_mso = true);

void cholesky_decomp_2e(ExecutionContext& ec, ChemEnv& chem_env);

} // namespace exachem::cholesky_2e
