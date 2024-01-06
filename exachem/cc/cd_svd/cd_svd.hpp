/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "common/system_data.hpp"
#include "scf/scf_main.hpp"
#include "tamm/eigen_utils.hpp"

#if defined(USE_UPCXX)
#include "tamm/ga_over_upcxx.hpp"
#endif

using namespace tamm;

using TAMM_GA_SIZE = int64_t;

std::tuple<TiledIndexSpace, TAMM_SIZE> setup_mo_red(SystemData sys_data, bool triples = false);

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(SystemData sys_data, bool triples = false,
                                                 int nactv = 0);

void update_sysdata(SystemData& sys_data, TiledIndexSpace& MO, bool is_mso = true);

// reshape F/lcao after freezing
Matrix reshape_mo_matrix(SystemData sys_data, Matrix& emat, bool is_lcao = false);

template<typename TensorType>
Tensor<TensorType> cd_svd(SystemData& sys_data, ExecutionContext& ec, TiledIndexSpace& tMO,
                          TiledIndexSpace& tAO, TAMM_SIZE& chol_count, const TAMM_GA_SIZE max_cvecs,
                          libint2::BasisSet& shells, Tensor<TensorType>& lcao, bool is_mso = true);
