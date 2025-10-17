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

namespace exachem::task {
// Forward declarations for free functions implemented in pyberny_impl.cpp
// For now they are commented out to avoid them being used anywhere else other than pyberny_impl.cpp
// std::tuple<std::vector<int>, Eigen::MatrixXd> get_init_geometry(ExecutionContext& ec, ChemEnv&
// chem_env); Eigen::MatrixXd dist_diff(int num_atoms, Eigen::MatrixXd coords); Eigen::MatrixXd
// calculate_rho(int num_atoms, Eigen::MatrixXd coords, std::vector<int> atomic_numbers); double
// bond_weight(Eigen::MatrixXd rho, int i, int j); std::tuple<double, std::tuple<Eigen::VectorXd,
// Eigen::VectorXd>> bond_eval(Eigen::MatrixXd coords, int i, int j); double
// bond_eval(Eigen::MatrixXd coords, int i, int j, bool _); double angle_center(Eigen::VectorXd ijk,
// int j); std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>
// angle_eval_impl(Eigen::MatrixXd coords, int i, int j, int k); double
// angle_eval_impl(Eigen::MatrixXd coords, bool _, int i, int j, int k); double
// angle_weight(Eigen::MatrixXd rho, Eigen::MatrixXd coords, int i, int j, int k); double
// torsion_weight(Eigen::MatrixXd rho, Eigen::MatrixXd coords, int i, int j, int k, int l); double
// torsion_center(Eigen::MatrixXd ijk, int j, int k); std::tuple<double, std::tuple<Eigen::VectorXd,
// Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>> torsion_eval(Eigen::MatrixXd coords, int i,
// int j, int k, int l); double torsion_eval(Eigen::MatrixXd coords, int i, int j, int k, int l,
// bool _); Eigen::MatrixXd b_matrix(Eigen::MatrixXd geom, InternalCoordinates int_coords);
// Eigen::MatrixXd pinv(ExecutionContext& ec, const Eigen::MatrixXd& A);
// Eigen::MatrixXd hessian_guess(ExecutionContext& ec, Eigen::MatrixXd geom, Eigen::MatrixXd rho,
// InternalCoordinates int_coords); Eigen::MatrixXd update_hessian(Eigen::MatrixXd hessian,
// Eigen::VectorXd dq, Eigen::VectorXd dg); Eigen::VectorXd eval_geom(ExecutionContext& ec,
// InternalCoordinates coords, Eigen::MatrixXd geom, const Eigen::VectorXd* q_template);
// Eigen::VectorXd eval_geom_(InternalCoordinates coords, Eigen::MatrixXd geom);
// double update_trust(double trust, double dE, double dE_predicted, Eigen::VectorXd dq);
// std::vector<double> g(double y0, double y1, double g0, double g1, double c);
// std::pair<double, double> fit_cubic(double y0, double y1, double g0, double g1);
// std::pair<double, double> quart_min(const std::vector<double>& p);
// std::pair<double, double> fit_quartic(double y0, double y1, double g0, double g1);
// std::tuple<double, double> linear_search(double e0, double e1, double g0, double g1);
// double find_root(ExecutionContext& ec, std::function<double(double)> f, double lim);
// std::tuple<Eigen::VectorXd, double, bool> quadratic_step(ExecutionContext& ec, Eigen::VectorXd g,
// Eigen::MatrixXd H, Eigen::VectorXd w, double trust); double rms(const Eigen::MatrixXd A);
// std::tuple<Eigen::VectorXd, Eigen::MatrixXd> update_geom(ExecutionContext& ec, Eigen::MatrixXd
// geom, Eigen::VectorXd q, Eigen::VectorXd dq, Eigen::MatrixXd B_inv, InternalCoordinates
// int_coords); bool is_converged(Eigen::VectorXd forces, Eigen::VectorXd step, bool on_sphere);
class OptPoint {
public:
  Eigen::VectorXd q;
  double          E;
  Eigen::VectorXd g;

  OptPoint() {}

  explicit OptPoint(const Eigen::VectorXd& _q): q(_q), E(0.0) {}

  OptPoint(const Eigen::VectorXd& _q, double _E): q(_q), E(_E) {}

  OptPoint(const Eigen::VectorXd& _q, double _E, const Eigen::VectorXd& _g): q(_q), E(_E), g(_g) {}

  // Destructor
  ~OptPoint() = default;

  // Copy constructor
  OptPoint(const OptPoint&) = default;

  // Move constructor
  OptPoint(OptPoint&&) noexcept = default;

  // Copy assignment
  OptPoint& operator=(const OptPoint&) = default;

  // Move assignment
  OptPoint& operator=(OptPoint&&) noexcept = default;
};

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
    int_coords = exachem::task::InternalCoords(ec, chem_env, false);
  }

  std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
             Eigen::MatrixXd>
  optimizer_berny(ExecutionContext& ec, ChemEnv& chem_env, double energy, Eigen::MatrixXd gradients,
                  bool first, OptPoint best, OptPoint previous, OptPoint predicted,
                  OptPoint interpolated_, OptPoint future, double trust, Eigen::MatrixXd hessian,
                  exachem::task::InternalCoordinates int_coords);

  // Destructor
  ~Pyberny() = default;

  // Copy constructor
  Pyberny(const Pyberny&) = default;

  // Move constructor
  Pyberny(Pyberny&&) noexcept = default;

  // Copy assignment
  Pyberny& operator=(const Pyberny&) = default;

  // Move assignment
  Pyberny& operator=(Pyberny&&) noexcept = default;

  // optimizer function, returns next geometry
  Eigen::RowVectorXd step(ExecutionContext& ec, ChemEnv& chem_env, double curr_energy,
                          Eigen::MatrixXd gradients) {
    std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
               Eigen::MatrixXd>
      resultant;

    Eigen::RowVectorXd new_geometry;

    resultant = optimizer_berny(ec, chem_env, curr_energy, gradients, first, best, previous,
                                predicted, interpolated, future, trust, hessian, int_coords);

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
