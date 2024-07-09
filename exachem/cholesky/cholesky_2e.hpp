/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "cholesky/cholesky_2e_driver.hpp"
#include "exachem/common/ec_basis.hpp"
#include "exachem/common/system_data.hpp"
#include "exachem/scf/scf_compute.hpp"
#include "tamm/eigen_utils.hpp"
#if defined(USE_UPCXX)
#include "tamm/ga_over_upcxx.hpp"
#endif

using namespace tamm;

using TAMM_GA_SIZE = int64_t;

namespace exachem::cholesky_2e {
std::tuple<TiledIndexSpace, TAMM_SIZE> setup_mo_red(ChemEnv& chem_env, bool triples = false);

std::tuple<TiledIndexSpace, TAMM_SIZE> setupMOIS(ChemEnv& chem_env, bool triples = false,
                                                 int nactv = 0);

void update_sysdata(ChemEnv& chem_env, TiledIndexSpace& MO, bool is_mso = true);

// reshape F/lcao after freezing
Matrix reshape_mo_matrix(ChemEnv& chem_env, Matrix& emat, bool is_lcao = false);

template<typename TensorType>
Tensor<TensorType> cholesky_2e(ChemEnv& chem_env, ExecutionContext& ec, TiledIndexSpace& tMO,
                               TiledIndexSpace& tAO, TAMM_SIZE& chol_count,
                               const TAMM_GA_SIZE max_cvecs, libint2::BasisSet& shells,
                               Tensor<TensorType>& lcao, bool is_mso = true);
} // namespace exachem::cholesky_2e
