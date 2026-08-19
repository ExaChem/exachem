/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/ccsd/ccsd_util.hpp"

namespace exachem::cc::ccsd_os {

using namespace tamm;

/**
 * @brief Preconditioner action function for the CCSD residuals:
 *
 * This function computes the action of the preconditioner on a given vector.
 *
 * Input: Fock matrix, a vector of same dimensions as the expected precoditioned vector.
 *
 * Note: There are some other things we need to pass in bases on the way you written ExaChem, I am
 * not sure what is the best way to write down down them here.
 *
 * Output: The preconditioner action on the input vector (Defined as My).
 */
template<typename T>
std::vector<Tensor<T>> preconditioner_action(Scheduler& sch, const TiledIndexSpace& MO,
                                             const TensorMap<T>& f, const TensorMap<T>& eri,
                                             const std::vector<Tensor<T>>& preconditoned_vector);

/**
 * @brief Preconditioner inverse application function for the CCSD residuals:
 *
 * Purpose: Use the preconditioner action function in GMRES to solve the linear system and return
 * precondtioned solution.
 *
 * Input: Preconditioner action function, the vector one wants to be preconditioned.
 * Note: In the context of CC, the vectors to be preconditioned,
 *  - J*delta_x
 *  - residual function
 *
 * Note: For GMRES,
 *  - Matrix vector product function: Preconditioner action function
 *  - Right hand side: The vector one wants to be preconditioned
 *  - Krylov subspace dimension: krylov_dims_precond (Should not be the same as krylov_dims in
 * general)
 *
 * Output: The preconditioned vector.
 */
template<typename T>
std::vector<Tensor<T>>
preconditioner_inverse_application(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
                                   Scheduler& sch, const TiledIndexSpace& MO, const TensorMap<T>& f,
                                   const TensorMap<T>&           eri,
                                   const std::vector<Tensor<T>>& vector_to_be_preconditioned,
                                   int krylov_dims_precond, double gmres_tol);

} // namespace exachem::cc::ccsd_os
