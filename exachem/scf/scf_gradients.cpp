/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/scf/scf_gradients.hpp"

void exachem::scf::SCFGradients::scf_gradients(ExecutionContext& ec, ChemEnv& chem_env,
                                               Matrix& SchwarzK, SCFData& scf_data,
                                               ScalapackInfo&               scalapack_info,
                                               GauXC::XCIntegrator<Matrix>& xc_integrator) {
  const int           rank              = ec.pg().rank().value();
  const bool          do_schwarz_screen = SchwarzK.cols() != 0 && SchwarzK.rows() != 0;
  size_t              max_nprim         = chem_env.shells.max_nprim();
  std::vector<size_t> shell2bf          = chem_env.shells.shell2bf();
  const size_t        max_nprim4        = static_cast<size_t>(std::pow(max_nprim, 4));

  auto do_t1 = std::chrono::high_resolution_clock::now();

  auto      atoms  = chem_env.atoms;
  const int natoms = atoms.size();

  Scheduler sch{ec};

  // libint2::operator_traits<Operator::X>::nopers = 1 for X=nuclear,kinetic,overlap
  const int  nopers{1};
  const auto nresults = nopers * libint2::num_geometrical_derivatives(natoms, 1);

  auto&  ttensors = scf_data.ttensors;
  Matrix grad_1body, grad_2body;
  Matrix grad_nuc_repl;
  Matrix grad_pulay;

  const bool is_uhf = chem_env.sys_data.is_unrestricted;

  ttensors.T_deriv.resize(nresults);
  ttensors.V_deriv.resize(nresults);
  ttensors.S_deriv.resize(nresults);
  for(unsigned int i = 0; i < nresults; ++i) {
    ttensors.T_deriv[i] = {scf_data.tAO, scf_data.tAO};
    ttensors.V_deriv[i] = {scf_data.tAO, scf_data.tAO};
    ttensors.S_deriv[i] = {scf_data.tAO, scf_data.tAO};
    sch.allocate(ttensors.T_deriv[i], ttensors.V_deriv[i], ttensors.S_deriv[i]);
  }
  sch.execute();

  scf_guess.compute_1body_ints_deriv(ec, chem_env, 1, scf_data, ttensors.T_deriv,
                                     libint2::Operator::kinetic);
  scf_guess.compute_1body_ints_deriv(ec, chem_env, 1, scf_data, ttensors.V_deriv,
                                     libint2::Operator::nuclear);
  scf_guess.compute_1body_ints_deriv(ec, chem_env, 1, scf_data, ttensors.S_deriv,
                                     libint2::Operator::overlap);
  auto do_t2   = std::chrono::high_resolution_clock::now();
  auto do_time = std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2) << std::endl
              << "Time to compute 1-e integral gradients T, V, S: " << do_time << " secs"
              << std::endl;

  // auto [mu, nu] = scf_data.tAO.labels<2>("all");

  using T = double;
  // one-body contributions to the gradients
  grad_1body = Matrix::Zero(natoms, 3);
  {
    Tensor<T> dens_tmp{scf_data.tAO, scf_data.tAO};
    Tensor<T> tv_tmp{scf_data.tAO, scf_data.tAO};
    Tensor<T> grad_contrib{};
    sch.allocate(dens_tmp, tv_tmp, grad_contrib);
    sch(dens_tmp() = 0.5 * ttensors.D_alpha());
    if(is_uhf) sch(dens_tmp() += 0.5 * ttensors.D_beta());
    sch.execute();
    for(auto atom = 0, i = 0; atom != natoms; ++atom) {
      for(auto xyz = 0; xyz != 3; ++xyz, ++i) {
        // clang-format off
          sch(tv_tmp()  = ttensors.T_deriv[i]())
             (tv_tmp() += ttensors.V_deriv[i]())
             (grad_contrib() = 2.0 * tv_tmp() * dens_tmp())
             .execute();
        // clang-format on
        grad_1body(atom, xyz) += tamm::get_scalar(grad_contrib);
      }
    }
    sch.deallocate(dens_tmp, tv_tmp, grad_contrib).execute();
  }

  for(unsigned int i = 0; i < nresults; ++i) {
    sch.deallocate(ttensors.T_deriv[i], ttensors.V_deriv[i]);
  }
  sch.execute();

  do_t1 = std::chrono::high_resolution_clock::now();

  scf_iter.compute_2bf_deriv(ec, chem_env, scalapack_info, scf_data, do_schwarz_screen, shell2bf,
                             SchwarzK, max_nprim4, scf_data.ttensors, scf_data.etensors,
                             /*scf_state.is_3c_init*/ false, scf_data.do_dens_fit, scf_data.xHF);

  do_t2   = std::chrono::high_resolution_clock::now();
  do_time = std::chrono::duration_cast<std::chrono::duration<double>>((do_t2 - do_t1)).count();

  if(rank == 0)
    std::cout << std::fixed << std::setprecision(2)
              << "Time to compute 2-e integral gradients: " << do_time << " secs" << std::endl;

  // Pulay contributions to the gradients
  Tensor<T> Wab_tamm{scf_data.tAO, scf_data.tAO};
  Tensor<T> grad_contrib{};
  sch.allocate(Wab_tamm, grad_contrib).execute();

  grad_pulay = Matrix::Zero(natoms, 3);
  if(rank == 0) {
    typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> DiagonalMatrix;
    Matrix                                                                Wab;
    {
      auto&          eps_a = scf_data.etensors.eps_a;
      Matrix         evals_a(Eigen::Map<const Eigen::VectorXd>(eps_a.data(), eps_a.size()));
      DiagonalMatrix evals_occ(evals_a.topRows(chem_env.sys_data.nelectrons_alpha));
      Matrix         C_occ_a = tamm_to_eigen_matrix(scf_data.ttensors.C_occ_a);
      Matrix         Wa      = C_occ_a * evals_occ * C_occ_a.transpose();
      Matrix         Wb      = Wa;
      if(is_uhf) {
        auto&          eps_b = scf_data.etensors.eps_b;
        Matrix         evals_b(Eigen::Map<const Eigen::VectorXd>(eps_b.data(), eps_b.size()));
        DiagonalMatrix evals_occ_b(evals_b.topRows(chem_env.sys_data.nelectrons_beta));
        Matrix         C_occ_b = tamm_to_eigen_matrix(scf_data.ttensors.C_occ_b);
        Wb                     = C_occ_b * evals_occ_b * C_occ_b.transpose();
      }

      Wab = Wa + Wb;
      // if(is_uhf) Wab += Wb;
    }
    eigen_to_tamm_tensor(Wab_tamm, Wab);
  }
  ec.pg().barrier();

  for(auto atom = 0, i = 0; atom != natoms; ++atom) {
    for(auto xyz = 0; xyz != 3; ++xyz, ++i) {
      sch(grad_contrib() = ttensors.S_deriv[i]() * Wab_tamm()).execute();
      grad_pulay(atom, xyz) -= tamm::get_scalar(grad_contrib);
    }
  }

  sch.deallocate(Wab_tamm, grad_contrib).execute();
  for(unsigned int i = 0; i < nresults; ++i) { sch.deallocate(ttensors.S_deriv[i]); }
  sch.execute();

  // compute nuclear repulsion contributions to the gradients
  grad_nuc_repl = Matrix::Zero(natoms, 3);
  // nuclear repulsion contribution to the gradients
  for(auto a1 = 1; a1 != natoms; ++a1) {
    const auto& atom1 = atoms[a1];
    for(auto a2 = 0; a2 < a1; ++a2) {
      const auto& atom2 = atoms[a2];

      auto x12   = atom1.x - atom2.x;
      auto y12   = atom1.y - atom2.y;
      auto z12   = atom1.z - atom2.z;
      auto r12_2 = x12 * x12 + y12 * y12 + z12 * z12;
      auto r12   = sqrt(r12_2);
      auto r12_3 = r12 * r12_2;

      auto z1z2_over_r12_3 = atom1.atomic_number * atom2.atomic_number / r12_3;

      auto fx = -x12 * z1z2_over_r12_3;
      auto fy = -y12 * z1z2_over_r12_3;
      auto fz = -z12 * z1z2_over_r12_3;
      grad_nuc_repl(a1, 0) += fx;
      grad_nuc_repl(a1, 1) += fy;
      grad_nuc_repl(a1, 2) += fz;
      grad_nuc_repl(a2, 0) -= fx;
      grad_nuc_repl(a2, 1) -= fy;
      grad_nuc_repl(a2, 2) -= fz;
    }
  }

  grad_2body = Matrix::Zero(natoms, 3);

  for(auto atom = 0, i = 0; atom != natoms; ++atom) {
    for(auto xyz = 0; xyz != 3; ++xyz, ++i) {
      auto grad_contrib =
        scf_data.etensors.Ga_deriv[i].cwiseProduct(scf_data.etensors.D_alpha * 0.5).sum();
      if(is_uhf)
        grad_contrib +=
          scf_data.etensors.Gb_deriv[i].cwiseProduct(scf_data.etensors.D_beta * 0.5).sum();
      grad_2body(atom, xyz) += grad_contrib;
    }
  }

  Matrix scf_gradients = grad_1body + grad_pulay + grad_2body + grad_nuc_repl;
  if(chem_env.sys_data.is_ks) {
    std::vector<T> exc_grad_vec;
    SCFGauxc<T>    scf_gauxc;
    exc_grad_vec =
      scf_gauxc.compute_exc_grad(ec, chem_env, scf_data.ttensors, scf_data.etensors, xc_integrator);
    Matrix exc_grad = Eigen::Map<const Matrix>(exc_grad_vec.data(), natoms, 3);
    scf_gradients += exc_grad;
  }

  if(ec.print() && chem_env.ioptions.scf_options.debug) {
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "** 1-body contributions to the gradients = \n";
    for(int atom = 0; atom != natoms; ++atom) {
      for(int xyz = 0; xyz != 3; ++xyz) std::cout << grad_1body(atom, xyz) << " ";
      std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "** Pulay contributions to the gradients = \n";
    for(int atom = 0; atom != natoms; ++atom) {
      for(int xyz = 0; xyz != 3; ++xyz) std::cout << grad_pulay(atom, xyz) << " ";
      std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "** 2-body contributions to the gradients = \n";
    for(int atom = 0; atom != natoms; ++atom) {
      for(int xyz = 0; xyz != 3; ++xyz) std::cout << grad_2body(atom, xyz) << " ";
      std::cout << "\n";
    }
    std::cout << std::endl;

    std::cout << "** Nuclear repulsion contributions to the gradients = \n";
    for(int atom = 0; atom != natoms; ++atom) {
      for(int xyz = 0; xyz != 3; ++xyz) std::cout << grad_nuc_repl(atom, xyz) << " ";
      std::cout << "\n";
    }

    std::cout << std::endl;

    std::cout << "** Hartree-Fock contributions to the gradients = \n";
    for(int atom = 0; atom != natoms; ++atom) {
      for(int xyz = 0; xyz != 3; ++xyz) std::cout << scf_gradients(atom, xyz) << " ";
      std::cout << "\n";
    }

    std::cout << std::endl;
    std::cout << std::defaultfloat;
  }

  chem_env.scf_context.scf_gradients = scf_gradients;
}