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
  // Routine computes and returns the CCSD natural orbitals for RHF and UHF references.
  //   RHF : One set of natural orbitals from the spin-summed 1-RDM (occupations 0-2).
  //   UHF : Separate alpha and beta natural spin-orbitals (occupations 0-1).
  // Frozen core and frozen virtual orbitals are left as they are in the SCF movecs (occupation
  // 1 and 0 per spin). Only the active block is rotated, and the full movecs is written.

  // Objects:
  //              gamma1 : 1-RDM in the spin-orbital basis (active orbitals only)
  //             gamma1a : Alpha block of gamma1 (RHF: spin-summed orbital 1-RDM)
  //             gamma1b : Beta block of gamma1 (UHF only)
  //     gamma1a/b_eig   : Eigen form of gamma1a/b
  //         nmo_a/nmo_b : Number of active alpha/beta orbitals
  //          n_fc, n_fv : Number of frozen core/virtual orbitals
  //              northo : Number of orbitals in the SCF movecs (frozen + active)
  //                dfac : Density factor (2 for RHF, 1 per spin for UHF)
  //             deltaoo : An (Occ x Occ) Matrix with 1's on the diagonal. Used to add the HF
  //                       contribution for which the occupied spin orbitals have an occupation
  //                       of 1.
  //          noocc_vals : Eigenvalues for the diagonalization of gamma1a/b
  //                 nev : Number of eigenvalues
  //              occnum : Eigenvalues
  //       occnum_sorted : Sorted eigenvalues (large to small)
  //  noocc_sorted_order : Sorting order
  //            natorbs1 : Eigenvectors for the diagonalization of gamma1a/b
  //             natorbs : Sorted eigenvectors for the diagonalization of gamma1a/b
  //  occ_alpha/occ_beta : Per-spin occupations over all orbitals (frozen + active), for printing

  // TODO:
  // 1. Ensure the proper handling when there are linear dependencies at the SCF level.

  const SystemData& sys_data = chem_env.sys_data;
  const bool        is_uhf   = sys_data.is_unrestricted; // else RHF
  const int         n_fc     = sys_data.n_frozen_core;
  const int         n_fv     = sys_data.n_frozen_virtual;
  const int         nao      = sys_data.nbf_orig;
  const int         northo   = chem_env.is_context.AO_ortho("all").max_num_indices();
  const int         nmo_a    = sys_data.n_occ_alpha + sys_data.n_vir_alpha; // active only
  const int         nmo_b    = sys_data.n_occ_beta + sys_data.n_vir_beta;   // active only
  const T           dfac     = is_uhf ? 1.0 : 2.0;
  EXPECTS(n_fc + nmo_a + n_fv == northo);

  const auto rank     = sch.ec().pg().rank();
  const auto [mu, nu] = AO_opt.labels<2>("all");
  const auto [pa, qa] = MO.labels<2>("all_alpha");
  const auto [pb, qb] = MO.labels<2>("all_beta");
  const auto [i, j]   = MO.labels<2>("occ");
  const auto [ia]     = MO.labels<1>("occ_alpha");
  const auto [ib]     = MO.labels<1>("occ_beta");

  Matrix gamma1a_eig = Matrix::Zero(nmo_a, nmo_a);
  Matrix gamma1b_eig;

  const std::string scf_prefix = files_dir + "/scf/" + sys_data.output_file_prefix;

  if(rank == 0) std::cout << "\nComputing CCSD Natural Orbitals" << std::endl << std::endl;

  // ################################################################################
  //  ADD THE HARTREE-FOCK CONTRIBUTION TO THE CC 1-RMD MATRIX
  // ################################################################################
  //  gamma1 is the CC 1-RDM matrix and does not contain the HF contribution
  //  when computed using compute_1rdm. Frozen core orbitals are not part of gamma1;
  //  their contribution is added to the AO density below.

  Tensor<T> deltaoo{{j, i}};
  sch.allocate(deltaoo).execute();

  sch(deltaoo() = 0).execute();
  init_diagonal(sch.ec(), deltaoo());

  sch(gamma1(i, j) += 1.0 * deltaoo(i, j)).execute();

  sch.deallocate(deltaoo).execute();
  // ################################################################################
  //  SPIN-ORBITAL TO ORBITAL 1-RDM
  // ################################################################################
  //  Since TAMM uses the oa|ob|va|vb tiling, using the full spin-orbital representation is not
  //  straight forward. For example, how do you identify the ordering of eigenvalues/vectors for
  //  degenerate orbitals and for both alpha and beta spin (remembering that ordering of
  //  eigenvalues/vectors is not for all spin orbitals but has to be seperated for alpha and beta
  //  spins).

  // So the alpha and beta parts are seperated. The 1-RDM is symmetrized so the left and right
  // eigenvalues/vectors are the same. Saves on having to compute both left and right.
  //   RHF : The alpha and beta parts are the same, so the alpha part is doubled in order to get
  //         the 'orbital' occupation numbers. gamma1(pa,qa) + gamma1(qa,pa) does both at once
  //         (x2 for beta = alpha, x1/2 for symmetrization).
  //   UHF : The alpha and beta parts differ, so each block is symmetrized on its own and
  //         diagonalized separately. The alpha-beta blocks of gamma1 are zero.

  Tensor<T> gamma1a{{pa, qa}};
  Tensor<T> gamma1b{{pb, qb}};
  sch.allocate(gamma1a).execute();
  if(is_uhf) sch.allocate(gamma1b).execute();

  if(!is_uhf) {
    sch(gamma1a(pa, qa) = gamma1(pa, qa))(gamma1a(pa, qa) += gamma1(qa, pa)).execute();
  }
  else {
    // clang-format off
    sch(gamma1a(pa, qa)  = 0.5 * gamma1(pa, qa))
       (gamma1a(pa, qa) += 0.5 * gamma1(qa, pa))
       (gamma1b(pb, qb)  = 0.5 * gamma1(pb, qb))
       (gamma1b(pb, qb) += 0.5 * gamma1(qb, pb))
       .execute();
    // clang-format on
  }

  tamm_to_eigen_tensor(gamma1a, gamma1a_eig);
  if(is_uhf) {
    gamma1b_eig = Matrix::Zero(nmo_b, nmo_b);
    tamm_to_eigen_tensor(gamma1b, gamma1b_eig);
  }

  sch.deallocate(gamma1, gamma1a).execute();
  if(is_uhf) sch.deallocate(gamma1b).execute();

  // ################################################################################
  //  SOLVE EIGENVALUES AND EIGENVECTORS
  // ################################################################################
  //  gamma1a/b are symmetric, so the self-adjoint solver is used. It gives real eigenvalues
  //  and orthonormal eigenvectors, also for degenerate eigenvalues.

  auto solve_natorbs = [&](Matrix& gamma_eig) {
    Eigen::SelfAdjointEigenSolver<Matrix> gamma1mat(gamma_eig);
    auto                                  noocc_vals = gamma1mat.eigenvalues();

    const auto     nev = noocc_vals.rows();
    std::vector<T> occnum(nev);
    std::vector<T> occnum_sorted(nev);
    for(auto x = 0; x < nev; x++) occnum[x] = noocc_vals(x);

    // Sort eigenvalues
    std::vector<size_t> noocc_sorted_order = exachem::scf::SCFUtil::sort_indexes(occnum, true);
    for(auto x = 0; x < nev; x++) occnum_sorted[x] = occnum[noocc_sorted_order[x]];

    // Sort eigenvectors in same order as eigenvalues
    auto natorbs1 = gamma1mat.eigenvectors();
    assert(natorbs1.rows() == nev && natorbs1.cols() == nev);
    Matrix natorbs = Matrix::Zero(nev, nev);

    for(auto x = 0; x < nev; x++) natorbs.col(x) = natorbs1.col(noocc_sorted_order[x]);

    return std::make_tuple(natorbs, occnum_sorted);
  };

  auto [natorbs_a, occnum_a] = solve_natorbs(gamma1a_eig);
  Matrix         natorbs_b;
  std::vector<T> occnum_b;
  if(is_uhf) std::tie(natorbs_b, occnum_b) = solve_natorbs(gamma1b_eig);

  // Per-spin occupations over all orbitals: frozen core, active, frozen virtual.
  // For RHF the orbital occupation is split evenly between alpha and beta.
  std::vector<T> occ_alpha(northo, 0.0);
  std::vector<T> occ_beta(northo, 0.0);
  for(auto x = 0; x < n_fc; x++) occ_alpha[x] = occ_beta[x] = 1.0;
  for(auto x = 0; x < nmo_a; x++) occ_alpha[n_fc + x] = is_uhf ? occnum_a[x] : 0.5 * occnum_a[x];
  for(auto x = 0; x < nmo_b; x++) occ_beta[n_fc + x] = is_uhf ? occnum_b[x] : 0.5 * occnum_a[x];

  // Print occupation numbers
  if(rank == 0) {
    const std::string dashes(38, '-');
    T                 sum_alpha = 0.0;
    T                 sum_beta  = 0.0;

    std::cout << std::fixed << std::setprecision(8) << std::noshowpos << std::right;
    std::cout << "\nNatural Orbital Occupation" << std::endl;
    std::cout << dashes << std::endl;
    std::cout << std::setw(6) << "#" << std::setw(16) << "Alpha" << std::setw(16) << "Beta"
              << std::endl;
    std::cout << dashes << std::endl;
    for(auto x = 0; x < northo; x++) {
      if(n_fv > 0 && x == n_fc + nmo_a)
        std::cout << dashes << std::endl << "v Frozen v" << std::endl;
      std::cout << std::setw(6) << x + 1 << std::setw(16) << occ_alpha[x] << std::setw(16)
                << occ_beta[x] << std::endl;
      if(n_fc > 0 && x == n_fc - 1) std::cout << "^ Frozen ^" << std::endl << dashes << std::endl;
      sum_alpha += occ_alpha[x];
      sum_beta += occ_beta[x];
    }
    std::cout << dashes << std::endl;
    std::cout << std::setw(6) << "Sum" << std::setw(16) << sum_alpha << std::setw(16) << sum_beta
              << std::endl;
    std::cout << dashes << std::endl;

    // Write occupation numbers to the results json:
    // numbered the same as the printed table (frozen orbitals included).
    auto& no_json =
      chem_env.sys_data.results["output"]["CCSD_Lambda"]["natural_orbital_occupation"];
    for(auto x = 0; x < northo; x++) {
      no_json["alpha"][std::to_string(x + 1)] = occ_alpha[x];
      no_json["beta"][std::to_string(x + 1)]  = occ_beta[x];
    }
    chem_env.write_json_data();
  }

  // ################################################################################
  //  TRANSFORM C AND FORM DENSITY FOR NATURAL ORBITAL BASIS
  // ################################################################################
  //  Done once for alpha (RHF and UHF) and once for beta (UHF only).
  //  Frozen core and frozen virtual columns of the SCF movecs are not rotated.

  exachem::scf::SCFIO<TensorType> scf_output;

  auto write_natorbs = [&, mu = mu, nu = nu](const std::string& spin, Matrix& natorbs,
                                             const TiledIndexLabel& p, const TiledIndexLabel& q,
                                             const TiledIndexLabel& io, const int nmo) {
    Matrix C_eig     = Matrix::Zero(nao, northo); // all orbitals (frozen + active)
    Matrix C_act_eig = Matrix::Zero(nao, nmo);    // active orbitals only

    // Transform the MO transformation matrix to a TAMM tensor
    Tensor<T> natorbs_TAMM{{p, q}};
    Tensor<T> C_AO2{mu, chem_env.is_context.AO_ortho};
    Tensor<T> C_AOMO{mu, p};
    Tensor<T> C_NO_AOMO{mu, p};
    Tensor<T> density{mu, nu};
    sch.allocate(natorbs_TAMM, C_AO2, C_AOMO, C_NO_AOMO, density).execute();
    eigen_to_tamm_tensor(natorbs_TAMM, natorbs);

    scf_output.rw_mat_disk(ec, C_AO2, scf_prefix + "." + spin + ".movecs", false,
                           true); // read C

    // Movecs were written as AO x (all orbitals), so they need to be read as such. Only the
    // active columns are transfered to AO x (active MO).
    tamm_to_eigen_tensor(C_AO2, C_eig);
    C_act_eig = C_eig.middleCols(n_fc, nmo);
    eigen_to_tamm_tensor(C_AOMO, C_act_eig);

    // clang-format off
    sch(density(mu, nu) = 0.0)
       (C_NO_AOMO(mu, q) = C_AOMO(mu, p) * natorbs_TAMM(p, q))
       (density(mu, nu) = dfac * C_NO_AOMO(mu, io) * C_NO_AOMO(nu, io))
       .execute();
    // clang-format on

    // Frozen core electrons are not in gamma1, so their contribution is added here.
    if(n_fc > 0) {
      TiledIndexSpace tFC{IndexSpace{range(0, (size_t) n_fc)}, static_cast<tamm::Tile>(n_fc)};
      const auto      ifc      = tFC.label("all");
      Matrix          C_fc_eig = C_eig.leftCols(n_fc);

      Tensor<T> C_FC{mu, tFC};
      sch.allocate(C_FC).execute();
      eigen_to_tamm_tensor(C_FC, C_fc_eig);

      sch(density(mu, nu) += dfac * C_FC(mu, ifc) * C_FC(nu, ifc)).deallocate(C_FC).execute();
    }

    // Put the rotated active columns back; frozen columns are left as they are.
    tamm_to_eigen_tensor(C_NO_AOMO, C_act_eig);
    C_eig.middleCols(n_fc, nmo) = C_act_eig;
    eigen_to_tamm_tensor(C_AO2, C_eig);

    if(rank == 0)
      std::cout << "\nCCSD natural orbitals movecs written to " + scf_prefix + ".ccsd." + spin +
                     ".movecs"
                << std::endl;
    if(rank == 0)
      std::cout << "\nCCSD natural orbitals density written to " + scf_prefix + ".ccsd." + spin +
                     ".density"
                << std::endl;
    scf_output.rw_mat_disk(ec, C_AO2, scf_prefix + ".ccsd." + spin + ".movecs", false, false);
    scf_output.rw_mat_disk(ec, density, scf_prefix + ".ccsd." + spin + ".density", false, false);

    sch.deallocate(natorbs_TAMM, C_AO2, C_AOMO, C_NO_AOMO, density).execute();
  };

  write_natorbs("alpha", natorbs_a, pa, qa, ia, nmo_a);
  if(is_uhf) write_natorbs("beta", natorbs_b, pb, qb, ib, nmo_b);
}

// Explicit template instantiation
template class CCSD_Natural_Orbitals<double>;

} // namespace exachem::cc::ccsd_lambda
