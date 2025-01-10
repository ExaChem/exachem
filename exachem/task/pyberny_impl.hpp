/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * This file contains a C++ implementation of the PyBerny Molecular structure optimizer
 * (https://github.com/jhrmnn/pyberny), licensed under the Mozilla Public License, v. 2.0.
 * You can obtain a copy of the MPL-2.0 license at http://mozilla.org/MPL/2.0
 *
 * This file is distributed under the Mozilla Public License, v. 2.0.
 *
 */

#include "exachem/task/internal_coordinates.hpp"

// double angstrom = 1 / 0.52917721092;
// std::vector<double> atom_radii_ = {
//   0.38, 0.32, 1.34, 0.9, 0.82, 0.77, 0.75, 0.73,
//   0.71, 0.69, 1.54, 1.3, 1.18, 1.11, 1.06, 1.02,
//   0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27,
//   1.39, 1.25, 1.26, 1.21, 1.38, 1.31, 1.26, 1.22,
//   1.19, 1.16, 1.14, 1.1, 2.11, 1.92, 1.62, 1.48,
//   1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48,
//   1.44, 1.41, 1.38, 1.35, 1.33, 1.3, 2.25, 1.98,
//   1.69
// };

namespace exachem::task {
class OptPoint {
public:
  Eigen::VectorXd q;
  double          E;
  Eigen::VectorXd g;

  OptPoint() {}

  OptPoint(Eigen::VectorXd _q) { q = _q; }

  OptPoint(Eigen::VectorXd _q, double _E) {
    q = _q;
    E = _E;
  }

  OptPoint(Eigen::VectorXd _q, double _E, Eigen::VectorXd _g) {
    q = _q;
    E = _E;
    g = _g;
  }
};

std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
           Eigen::MatrixXd>
optimizer_berny(ExecutionContext& ec, ChemEnv& chem_env, double energy, Eigen::MatrixXd gradients,
                bool first, OptPoint best, OptPoint previous, OptPoint predicted,
                OptPoint interpolated_, OptPoint future, double trust, Eigen::MatrixXd hessian,
                exachem::task::InternalCoordinates int_coords);

class Pyberny {
public:
  // declaring variables
  OptPoint                           best;
  OptPoint                           previous;
  OptPoint                           predicted;
  OptPoint                           interpolated;
  OptPoint                           future;
  Eigen::MatrixXd                    hessian;
  exachem::task::InternalCoordinates int_coords;
  bool                               first = true;
  double                             trust = 0.3;

  // constructor
  Pyberny(ExecutionContext& ec, ChemEnv& chem_env) {
    int_coords = exachem::task::InternalCoords(ec, chem_env);
  }

  // optimizer function, returns next geometry
  Eigen::RowVectorXd step(ExecutionContext& ec, ChemEnv& chem_env, double curr_energy,
                          Eigen::MatrixXd gradients) {
    std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
               Eigen::MatrixXd>
      resultant;

    Eigen::RowVectorXd new_geometry;

    resultant = exachem::task::optimizer_berny(ec, chem_env, curr_energy, gradients, first, best,
                                               previous, predicted, interpolated, future, trust,
                                               hessian, int_coords);

    first        = false;
    new_geometry = std::get<0>(resultant);
    best         = std::get<1>(resultant);
    previous     = std::get<2>(resultant);
    predicted    = std::get<3>(resultant);
    interpolated = std::get<4>(resultant);
    future       = std::get<5>(resultant);
    trust        = std::get<6>(resultant);
    hessian      = std::get<7>(resultant);

    return new_geometry;
  }
};

} // namespace exachem::task