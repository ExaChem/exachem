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
 * @brief Selects how M^{-1} is applied.
 *  - GMRES:    general approach, solves M y = v iteratively using preconditioner_action.
 *  - Diagonal: exact inverse of the orbital-energy-denominator preconditioner (canonical MOs).
 */
enum class PreconditionerType { GMRES, Diagonal };

/**
 * @brief Preconditioner action function for the CCSD residuals:
 *
 * This function computes the action of the preconditioner on a given vector.
 *
 * Input: Fock matrix, a vector of same dimensions as the expected precoditioned vector.
 *
 * Output: The preconditioner action on the input vector (Defined as My).
 */
template<typename T>
std::vector<Tensor<T>> preconditioner_action(Scheduler& sch, const TiledIndexSpace& MO,
                                             const TensorMap<T>& f, const TensorMap<T>& eri,
                                             const std::vector<Tensor<T>>& preconditoned_vector);

/**
 * @brief Preconditioner inverse application (GMRES version).
 *
 * Purpose: Use the preconditioner action function in GMRES to solve the linear system and return
 * precondtioned solution.
 *
 * Note: For GMRES,
 *  - Matrix vector product function: Preconditioner action function
 *  - Right hand side: The vector one wants to be preconditioned
 *  - Krylov subspace dimension: krylov_dims_precond (Should not be the same as krylov_dims in
 * general)
 *
 * Output: The preconditioned vector (new tensors, caller deallocates).
 */
template<typename T>
std::vector<Tensor<T>>
preconditioner_inverse_application(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
                                   Scheduler& sch, const TiledIndexSpace& MO, const TensorMap<T>& f,
                                   const TensorMap<T>&           eri,
                                   const std::vector<Tensor<T>>& vector_to_be_preconditioned,
                                   int krylov_dims_precond, double gmres_tol);

/**
 * @brief Diagonal preconditioner inverse application (orbital energy denominators).
 *
 * Valid for canonical orbitals (f_oo, f_vv diagonal, f_ov = 0). Applies M^{-1} element-wise:
 *  - singles: y(a,i)     / (e_a - e_i + shift)
 *  - doubles: y(a,b,i,j) / (e_a + e_b - e_i - e_j + shift)
 *
 * Input: vector = {singles (V,O), doubles (V,V,O,O)}.
 * @param eps_occ  orbital energies of the occupied space, ordered like the "occ" tiled index space
 * @param eps_virt orbital energies of the virtual space, ordered like the "virt" tiled index space
 * @param shift    optional level shift added to every denominator
 *
 * Output: new tensors (input untouched, caller deallocates).
 */
template<typename T>
std::vector<Tensor<T>> initialize_diagonal_preconditioner(
  ExecutionContext& ec, const std::vector<Tensor<T>>& vector_template,
  const std::vector<T>& eps_occ, const std::vector<T>& eps_virt, T shift = T(0));

template<typename T>
std::vector<Tensor<T>> diagonal_preconditioner_inverse_application(
  ExecutionContext& ec, const std::vector<Tensor<T>>& vector_to_be_preconditioned,
  const std::vector<Tensor<T>>& inverse_denominators);

/**
 * @brief Dispatcher: applies M^{-1} using the selected preconditioner type.
 * GMRES-only arguments (chem_env, d_e, sch, MO, f, eri, krylov_dims_precond, gmres_tol) are ignored
 * for PreconditionerType::Diagonal, and inverse_denominators are ignored for
 * PreconditionerType::GMRES.
 */
template<typename T>
std::vector<Tensor<T>> apply_preconditioner_inverse(
  PreconditionerType type, ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e, Scheduler& sch,
  const TiledIndexSpace& MO, const TensorMap<T>& f, const TensorMap<T>& eri,
  const std::vector<Tensor<T>>& vector_to_be_preconditioned, int krylov_dims_precond,
  double gmres_tol, const std::vector<Tensor<T>>& inverse_denominators);

} // namespace exachem::cc::ccsd_os
