/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/lambda/ccsd_natural_orbitals.hpp"

namespace exachem::cc::ccsd_lambda {

template<typename T>
void CCSD_Natural_Orbitals<T>::ccsd_natural_orbitals(ChemEnv& chem_env, std::vector<int>& cc_rdm,
                                                     std::string files_prefix,
                                                     std::string files_dir, Scheduler& sch,
                                                     ExecutionContext& ec, TiledIndexSpace& MO,
                                                     TiledIndexSpace& AO_opt, Tensor<T>& gamma1,
                                                     ExecutionHW ex_hw) {
  // About:
  // Routine computes and returns the CCSD natural orbitals. This is only done for RHF references.

  // Note: Eigenvectors seem to come out normalized, but this has not been thoroughly checked.

  // TODO: - Put in checks for imaginary eigenvalues
  //       - Put in checks for normalization
  //       - Update for UHF and ROHF references

  // Objects:
  //              gamma1 : 1-RDM in the spin-orbital basis
  //             gamma1a : 1-RDM in the orbital basis (alpha portion of gamma1)
  //         gamma1a_eig : Eigen form of gamma1a
  //          gamma1_dim : Dimension of gamma1 (i.e. the number of spin orbitals)
  //             deltaoo : An (Occ x Occ) Matrix with 1's on the diagonal. Used to add the HF
  //                       contribution for which the occupied spin orbitals have an occupation
  //                       of 1.
  //          noocc_vals : Eigenvalues for the diagonalization of gamma1a
  //                 nev : Number of eigenvalues
  //              occnum : Real component of Eigenvalues
  //       occnum_sorted : Sorted real component of Eigenvalues (large to small)
  //  noocc_sorted_order : Sorting order
  //            natorbs1 : Eigenvectors for the diagonalization of gamma1a
  //             natorbs : Sorted eigenvectors for the diagonalization of gamma1a

  Matrix         natorbs;
  Matrix         natspinorbs;
  std::vector<T> occnum;
  std::vector<T> occnum_sorted;

  const auto gamma1_dim = MO("all").max_num_indices(); // # of spin orbitals

  const auto rank     = sch.ec().pg().rank();
  const auto [mu, nu] = AO_opt.labels<2>("all");
  const auto [pa, qa] = MO.labels<2>("all_alpha");
  const auto [i, j]   = MO.labels<2>("occ");
  const auto [ia]     = MO.labels<1>("occ_alpha");

  Matrix gamma1a_eig = Matrix::Zero(gamma1_dim / 2, gamma1_dim / 2);
}

// Explicit template instantiation
template class CCSD_Natural_Orbitals<double>;

} // namespace exachem::cc::ccsd_lambda
