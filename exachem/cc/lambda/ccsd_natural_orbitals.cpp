/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/ccsd_util.hpp"
// #include "exachem/cc/ducc/ducc_driver.cpp"
#include <tamm/tamm.hpp>

using namespace tamm;

template<typename T>
void ccsd_natural_orbitals(ChemEnv& chem_env, std::vector<int>& cc_rdm, std::string files_prefix,
                           std::string files_dir, Scheduler& sch, ExecutionContext& ec,
                           TiledIndexSpace& MO, TiledIndexSpace& AO_opt, Tensor<T>& gamma1,
                           ExecutionHW ex_hw = ExecutionHW::CPU) {
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

  auto rank     = sch.ec().pg().rank();
  auto [mu, nu] = AO_opt.labels<2>("all");
  auto [pa, qa] = MO.labels<2>("all_alpha");
  auto [i, j]   = MO.labels<2>("occ");
  auto [ia]     = MO.labels<1>("occ_alpha");

  Matrix         natorbs;
  Matrix         natspinorbs;
  std::vector<T> occnum;
  std::vector<T> occnum_sorted;

  const auto gamma1_dim  = MO("all").max_num_indices(); // # of spin orbitals
  Matrix     gamma1a_eig = Matrix::Zero(gamma1_dim / 2, gamma1_dim / 2);
  Matrix     C_alpha_eig = Matrix::Zero(gamma1_dim / 2, gamma1_dim / 2);

  if(rank == 0)
    std::cout << "\nComputing CCSD Natural Orbitals"
              << "\n"
              << std::endl;

  //################################################################################
  // ADD THE HARTREE-FOCK CONTRIBUTION TO THE CC 1-RMD MATRIX
  //################################################################################
  // gamma1 is the CC 1-RDM matrix and does not contain the HF contribution
  // when computed using compute_1rdm.

  Tensor<T> deltaoo{{j, i}};
  sch.allocate(deltaoo).execute();

  sch(deltaoo() = 0).execute();
  init_diagonal(sch.ec(), deltaoo());

  sch(gamma1(i, j) += 1.0 * deltaoo(i, j)).execute();

  sch.deallocate(deltaoo).execute();
  //################################################################################
  // SPIN-ORBITAL TO ORBITAL 1-RDM
  //################################################################################
  // Since TAMM uses the oa|ob|va|vb tiling, using the full spin-orbital representation is not
  // straight forward. For example, how do you identify the ordering of eigenvalues/vectors for
  // degenerate orbitals and for both alpha and beta spin (remembering that ordering of
  // eigenvalues/vectors is not for all spin orbitals but has to be seperated for alpha and beta
  // spins).

  // So the alpha and beta parts should be seperated. For RHF, these are the same and so
  // the alpha contribution is doubled in order to get the 'orbital' occupation numbers while
  // easily symmetrizing the 1-RMD density matrix (alpha part). The eigenvectors will be used to
  // treat the alpha and beta parts later in a consistent manner.

  Tensor<T> gamma1a{{pa, qa}};
  sch.allocate(gamma1a).execute();

  // 1-RDM is symmetrized so the left and right eigenvalues/vectors are the same.
  // Saves on having to compute both left and right.
  sch(gamma1a(pa, qa) = gamma1(pa, qa))(gamma1a(pa, qa) += gamma1(qa, pa)).execute();

  tamm_to_eigen_tensor(gamma1a, gamma1a_eig);

  sch.deallocate(gamma1).execute();

  //################################################################################
  // SOLVE EIGENVALUES AND EIGENVECTORS
  //################################################################################

  Eigen::EigenSolver<Matrix> gamma1mat(gamma1a_eig);
  auto                       noocc_vals = gamma1mat.eigenvalues();

  const auto nev = noocc_vals.rows();
  occnum.resize(nev);
  occnum_sorted.resize(nev);
  for(auto x = 0; x < nev; x++) occnum[x] = real(noocc_vals(x));

  // Sort eigenvalues
  std::vector<size_t> noocc_sorted_order = exachem::scf::sort_indexes(occnum, true);
  for(auto x = 0; x < nev; x++) occnum_sorted[x] = occnum[noocc_sorted_order[x]];

  // Print occupation numbers
  if(rank == 0) std::cout << "\nNatural Orbital Occupation" << std::endl;
  if(rank == 0) std::cout << "--------------------------" << std::endl;
  for(auto x = 0; x < nev; x++)
    if(rank == 0)
      std::cout << std::setprecision(8) << x + 1 << "  " << occnum_sorted[x] << std::endl;
  if(rank == 0) std::cout << "--------------------------" << std::endl;

  // Sort eigenvectors in same order as eigenvalues
  auto natorbs1 = gamma1mat.eigenvectors();
  assert(natorbs1.rows() == nev && natorbs1.cols() == nev);
  natorbs.resize(nev, nev);
  natorbs.setZero();

  for(auto x = 0; x < nev; x++) natorbs.col(x) = natorbs1.col(noocc_sorted_order[x]).real();

  // Transform the MO transformation matrix to a TAMM tensor
  Tensor<T> natorbs_TAMM{{pa, qa}};
  sch.allocate(natorbs_TAMM).execute();
  eigen_to_tamm_tensor(natorbs_TAMM, natorbs);

  //################################################################################
  // TRANSFORM C AND FORM DENSITY FOR NATURAL ORBITAL BASIS
  //################################################################################
  Tensor<T> C_alpha_AO2{mu, chem_env.AO_ortho};
  Tensor<T> C_alpha_AOMO{mu, pa};
  Tensor<T> C_alpha_NO_AOMO{mu, pa};
  Tensor<T> alpha_density{mu, nu};
  sch.allocate(C_alpha_AO2, C_alpha_AOMO, C_alpha_NO_AOMO, alpha_density).execute();

  exachem::scf::SCFIO scf_output;
  scf_output.rw_mat_disk<T>(
    C_alpha_AO2, files_dir + "/scf/" + chem_env.sys_data.output_file_prefix + ".alpha.movecs",
    false, true); // read C

  // Movecs were written as AOxAO, so they need to be read as such, then transfered to AOxMO.
  // You can surpass this if natorbs_TAMM is defined {{mu, qa}}.
  tamm_to_eigen_tensor(C_alpha_AO2, C_alpha_eig);
  eigen_to_tamm_tensor(C_alpha_AOMO, C_alpha_eig);

  // clang-format off
  sch(alpha_density(mu, nu) = 0.0)
     (C_alpha_NO_AOMO(mu, qa) = C_alpha_AOMO(mu, pa) * natorbs_TAMM(pa, qa))
     (alpha_density(mu, nu) = 2.0 * C_alpha_NO_AOMO(mu, ia) * C_alpha_NO_AOMO(nu, ia))
     .execute();
  // clang-format on

  tamm_to_eigen_tensor(C_alpha_NO_AOMO, C_alpha_eig);
  eigen_to_tamm_tensor(C_alpha_AO2, C_alpha_eig);

  if(rank == 0)
    std::cout << "\nCCSD natural orbitals movecs written to " + files_dir + "/scf/" +
                   chem_env.sys_data.output_file_prefix + +".ccsd.alpha.movecs"
              << std::endl;
  if(rank == 0)
    std::cout << "\nCCSD natural orbitals density written to " + files_dir + "/scf/" +
                   chem_env.sys_data.output_file_prefix + +".ccsd.alpha.density"
              << std::endl;
  scf_output.rw_mat_disk<TensorType>(
    C_alpha_AO2, files_dir + "/scf/" + chem_env.sys_data.output_file_prefix + +".ccsd.alpha.movecs",
    false, false);
  scf_output.rw_mat_disk<TensorType>(alpha_density,
                                     files_dir + "/scf/" + chem_env.sys_data.output_file_prefix +
                                       ".ccsd.alpha.density",
                                     false, false);

  sch.deallocate(C_alpha_AO2, C_alpha_AOMO, C_alpha_NO_AOMO, alpha_density).execute();
}

using T = double;
template void ccsd_natural_orbitals<T>(ChemEnv& chem_env, std::vector<int>& cc_rdm,
                                       std::string files_prefix, std::string files_dir,
                                       Scheduler& sch, ExecutionContext& ec, TiledIndexSpace& MO,
                                       TiledIndexSpace& AO_opt, Tensor<T>& gamma1,
                                       ExecutionHW ex_hw);
