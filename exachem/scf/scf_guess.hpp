/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "scf_common.hpp"

namespace scf_guess {

/// computes orbital occupation numbers for a subshell of size \c size created
/// by smearing
/// no more than \c ne electrons (corresponds to spherical averaging)
///
/// @param[in,out] occvec occupation vector, increments by \c size on return
/// @param[in] size the size of the subshell
/// @param[in,out] ne the number of electrons, on return contains the number of
/// "remaining" electrons
template<typename Real>
void subshell_occvec(Real& occvec, size_t size, size_t& ne);

/// @brief computes average orbital occupancies in the ground state of a neutral
///        atoms
/// @return occupation vector corresponding to the ground state electronic
///         configuration of a neutral atom with atomic number \c Z
///         corresponding to the orbital ordering.
template<typename Real = double>
const std::vector<Real> compute_ao_occupation_vector(size_t Z);

} // namespace scf_guess

// computes Superposition-Of-Atomic-Densities guess for the molecular density matrix
// in minimal basis; occupies subshells by smearing electrons evenly over the orbitals
Matrix compute_soad(const std::vector<Atom>& atoms);

template<typename TensorType>
void compute_dipole_ints(ExecutionContext& ec, const SCFVars& spvars, Tensor<TensorType>& tensorX,
                         Tensor<TensorType>& tensorY, Tensor<TensorType>& tensorZ,
                         std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                         libint2::Operator otype);

template<typename TensorType>
void compute_1body_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                        std::vector<libint2::Atom>& atoms, libint2::BasisSet& shells,
                        libint2::Operator otype);

template<typename TensorType>
void compute_pchg_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                       std::vector<std::pair<double, std::array<double, 3>>>& q,
                       libint2::BasisSet& shells, libint2::Operator otype);

template<typename TensorType>
void compute_ecp_ints(ExecutionContext& ec, const SCFVars& scf_vars, Tensor<TensorType>& tensor1e,
                      std::vector<libecpint::GaussianShell>& shells,
                      std::vector<libecpint::ECP>&           ecps);

template<typename TensorType>
void scf_diagonalize(Scheduler& sch, const SystemData& sys_data, ScalapackInfo& scalapack_info,
                     TAMMTensors& ttensors, EigenTensors& etensors);

template<typename TensorType>
void compute_sad_guess(ExecutionContext& ec, ScalapackInfo& scalapack_info, SystemData& sys_data,
                       SCFVars& scf_vars, const std::vector<libint2::Atom>& atoms,
                       const libint2::BasisSet& shells, const std::string& basis, bool is_spherical,
                       EigenTensors& etensors, TAMMTensors& ttensors, int charge, int multiplicity);
