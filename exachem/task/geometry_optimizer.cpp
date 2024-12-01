/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/task/geometry_optimizer.hpp"

namespace exachem::task {

void update_geometry(std::vector<Atom>& atoms, std::vector<ECAtom>& ec_atoms,
                     const Eigen::RowVectorXd& new_geometry) {
  int c = 0;
  for(size_t i = 0; i < atoms.size(); i++) {
    atoms[i].x         = new_geometry(0, c++);
    ec_atoms[i].atom.x = atoms[i].x;

    atoms[i].y         = new_geometry(0, c++);
    ec_atoms[i].atom.y = atoms[i].y;

    atoms[i].z         = new_geometry(0, c++);
    ec_atoms[i].atom.z = atoms[i].z;
  }
}

void geometry_optimizer(ExecutionContext& ec, ChemEnv& chem_env, std::vector<Atom>& atoms,
                        std::vector<ECAtom>& ec_atoms, std::string ec_arg2) {
  using RowVectorXd = Eigen::RowVectorXd;

  const int    max_steps = 20;
  const double grad_tol  = 4.5e-4;

  const int   natoms3 = atoms.size() * 3;
  RowVectorXd geometry(natoms3);

  int c = 0;
  for(size_t i = 0; i < atoms.size(); i++) {
    geometry(c++) = atoms[i].x;
    geometry(c++) = atoms[i].y;
    geometry(c++) = atoms[i].z;
  }

  Matrix      gradient_matrix = compute_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);
  RowVectorXd gradients = Eigen::Map<RowVectorXd>(gradient_matrix.data(), gradient_matrix.size());

  double curr_energy = chem_env.task_energy; // original task energy
  double prev_energy = curr_energy;

  for(int iter = 1; iter < max_steps; iter++) {
    if(ec.print()) {
      std::cout << std::endl
                << "[Optimization] Step " << iter - 1 << ": Energy = " << std::fixed
                << std::setprecision(10) << curr_energy
                << ", Energy Delta = " << std::setprecision(2) << std::scientific
                << (curr_energy - prev_energy) << ", Gradient norm = " << std::fixed
                << std::setprecision(6) << gradients.norm() << std::endl;
    }

    if(gradients.norm() < grad_tol) {
      if(ec.print()) {
        std::cout << std::endl << "Optimization converged in " << iter << " steps" << std::endl;
        std::cout << std::endl << std::setw(34) << "Optimized geometry" << std::endl;
        print_geometry(ec, chem_env);
      }
      return;
    }

    // Initial Hessian approximation
    Matrix H = Matrix::Identity(natoms3, natoms3);

    RowVectorXd search_direction = -H * gradients;

    const double step_length  = 1.0; // TODO: implement line search
    RowVectorXd  new_geometry = geometry + step_length * search_direction;

    update_geometry(atoms, ec_atoms, new_geometry);

    gradient_matrix = compute_gradients(ec, chem_env, atoms, ec_atoms, ec_arg2);
    RowVectorXd new_gradients =
      Eigen::Map<RowVectorXd>(gradient_matrix.data(), gradient_matrix.size());

    RowVectorXd s = new_geometry - geometry;   // Step vector
    RowVectorXd y = new_gradients - gradients; // Gradient difference

    // Update Hessian approximation using BFGS formula
    double s_dot_y = s.dot(y);
    if(s_dot_y > 1e-10) { // Avoid division by zero or instability
      RowVectorXd Hy = H * y;
      // H += (s * s.transpose()) / s_dot_y - (Hy * Hy.transpose()) / (y.transpose() * Hy);
      H += (s.transpose() * s) / s_dot_y - (Hy.transpose() * Hy) / (Hy * y.transpose());
    }

    // Update for next iteration
    geometry  = new_geometry;
    gradients = new_gradients;

    prev_energy = curr_energy;
    curr_energy = chem_env.task_energy;

  } // max_steps

  if(ec.print()) {
    std::cout << "Geometry optimizer did not converge in " << max_steps << " steps" << std::endl;
    std::cout << std::endl << std::setw(34) << "Last geometry produced" << std::endl;
    print_geometry(ec, chem_env);
  }
}

} // namespace exachem::task