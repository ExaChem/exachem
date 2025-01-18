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

#include "exachem/task/pyberny_impl.hpp"

using exachem::task::InternalCoordinate;
using exachem::task::InternalCoordinates;

namespace exachem::task {

double angstrom_impl = 1 / 0.52917721092;

std::vector<double> atom_radii_impl = {
  0.38, 0.32, 1.34, 0.9,  0.82, 0.77, 0.75, 0.73, 0.71, 0.69, 1.54, 1.3,  1.18, 1.11, 1.06,
  1.02, 0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31,
  1.26, 1.22, 1.19, 1.16, 1.14, 1.1,  2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35,
  1.31, 1.53, 1.48, 1.44, 1.41, 1.38, 1.35, 1.33, 1.3,  2.25, 1.98, 1.69};

std::tuple<std::vector<int>, Eigen::MatrixXd> get_init_geometry(ExecutionContext& ec,
                                                                ChemEnv&          chem_env) {
  std::vector<int> atomic_numbers;
  Eigen::MatrixXd  geometry(chem_env.ec_atoms.size(), 3);

  // for each atom
  for(size_t i = 0; i < chem_env.ec_atoms.size(); i++) {
    atomic_numbers.push_back(chem_env.ec_atoms[i].atom.atomic_number);
    double x = chem_env.ec_atoms[i].atom.x;
    double y = chem_env.ec_atoms[i].atom.y;
    double z = chem_env.ec_atoms[i].atom.z;

    geometry(i, 0) = x;
    geometry(i, 1) = y;
    geometry(i, 2) = z;
  }

  std::tuple<std::vector<int>, Eigen::MatrixXd> out = std::make_tuple(atomic_numbers, geometry);

  return out;
}

Eigen::MatrixXd dist_diff(int num_atoms, Eigen::MatrixXd coords) {
  Eigen::MatrixXd                               dist(num_atoms, num_atoms);
  std::vector<std::vector<std::vector<double>>> diff(
    num_atoms, std::vector<std::vector<double>>(num_atoms, std::vector<double>(3, 0.0)));

  // Calculate distances and differences
  for(int i = 0; i < num_atoms; i++) {
    for(int j = 0; j < num_atoms; j++) {
      if(i == j) continue; // Skip the diagonal

      // Calculate the vector difference
      for(int k = 0; k < 3; k++) { diff[i][j][k] = coords(i, k) - coords(j, k); }

      // Calculate the distance
      double distance = 0.0;
      for(int k = 0; k < 3; k++) { distance += diff[i][j][k] * diff[i][j][k]; }
      dist(i, j) = std::sqrt(distance);
    }
  }

  return dist;
}

Eigen::MatrixXd calculate_rho(int num_atoms, Eigen::MatrixXd coords,
                              std::vector<int> atomic_numbers) {
  Eigen::MatrixXd rho_matrix(num_atoms, num_atoms);

  std::vector<double> radii;
  for(int i = 0; i < num_atoms; i++) { radii.push_back(atom_radii_impl[atomic_numbers[i] - 1]); }

  // Calculate the distance matrix
  Eigen::MatrixXd dist = dist_diff(num_atoms, coords);

  for(int i = 0; i < num_atoms; i++) {
    for(int j = 0; j < num_atoms; j++) {
      if(i != j) {
        double factor    = -dist(i, j) / (radii[i] + radii[j]) + 1;
        rho_matrix(i, j) = std::exp(factor);
      }
    }
  }

  return rho_matrix;
}

double bond_weight(Eigen::MatrixXd rho, int i, int j) { return rho(i, j); }

std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd>> bond_eval(Eigen::MatrixXd coords,
                                                                           int i, int j) {
  Eigen::VectorXd v  = (coords.row(i) - coords.row(j)) * angstrom_impl;
  double          r  = v.norm();
  Eigen::VectorXd f1 = v / r;
  Eigen::VectorXd f2 = -v / r;
  for(auto& item: f1) {
    if(std::isnan(item)) { item = 0.0; }
  }
  for(auto& item: f2) {
    if(std::isnan(item)) { item = 0.0; }
  }
  return std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd>>(
    r, std::tuple<Eigen::VectorXd, Eigen::VectorXd>(f1, f2));
}

double bond_eval(Eigen::MatrixXd coords, int i, int j, bool _) {
  Eigen::VectorXd v = (coords.row(i) - coords.row(j)) * angstrom_impl;
  double          r = v.norm();
  return r;
}

double angle_center(Eigen::VectorXd ijk, int j) { return round(2 * ijk(j)); }

std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>
angle_eval_impl(Eigen::MatrixXd coords, int i, int j, int k) {
  auto v1 = (coords.row(i) - coords.row(j)) * angstrom_impl;
  auto v2 = (coords.row(k) - coords.row(j)) * angstrom_impl;

  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());
  if(std::isnan(dot_product)) { dot_product = 0.0; }
  if(dot_product < -1) { dot_product = -1; }
  else if(dot_product > 1) { dot_product = 1; }
  double phi = acos(dot_product);

  std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> grad;

  if(abs(phi) > acos(-1.0) - 1e-6) {
    double pi_phi     = acos(-1.0) - phi;
    double v1_norm    = v1.norm();
    double v2_norm    = v2.norm();
    double sq_v1_norm = v1_norm * v1_norm;
    double sq_v2_norm = v2_norm * v2_norm;

    double term1 = (2 * sq_v1_norm);
    double term2 = (1 / v1_norm - 1 / v2_norm) * pi_phi;
    double term3 = (2 * sq_v2_norm);

    grad = std::make_tuple((pi_phi / term1 * v1.array()).matrix().eval(),
                           (term2 / (2 * v1_norm * v1.array())).matrix().eval(),
                           (pi_phi / term3 * v2.array()).matrix().eval());
  }
  else {
    grad = std::make_tuple(
      (1 / tan(phi) * v1.array() / (pow(v1.norm(), 2)) -
       v2.array() / (v1.norm() * v2.norm() * sin(phi)))
        .matrix()
        .eval(),
      ((v1.array() + v2.array()) / (v1.norm() * v2.norm() * sin(phi)) -
       1 / tan(phi) * (v1.array() / pow(v1.norm(), 2) + v2.array() / pow(v2.norm(), 2)))
        .matrix()
        .eval(),
      (1 / tan(phi) * v2.array() / (pow(v2.norm(), 2)) -
       v1.array() / (v1.norm() * v2.norm() * sin(phi)))
        .matrix()
        .eval());
  }

  return std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>(phi,
                                                                                           grad);
}

double angle_eval_impl(Eigen::MatrixXd coords, bool _, int i, int j, int k) {
  auto v1 = (coords.row(i) - coords.row(j)) * angstrom_impl;
  auto v2 = (coords.row(k) - coords.row(j)) * angstrom_impl;

  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());
  if(std::isnan(dot_product)) { dot_product = 0.0; }
  if(dot_product < -1) { dot_product = -1; }
  else if(dot_product > 1) { dot_product = 1; }
  double phi = acos(dot_product);

  return phi;
}

double angle_weight(Eigen::MatrixXd rho, Eigen::MatrixXd coords, int i, int j, int k) {
  double f = 0.12;
  return sqrt(rho(i, j) * rho(j, k)) *
         (f + (1 - f) * (sin(angle_eval_impl(coords, false, i, j, k))));
}

double torsion_weight(Eigen::MatrixXd rho, Eigen::MatrixXd coords, int i, int j, int k, int l) {
  double f   = 0.12;
  double th1 = angle_eval_impl(coords, false, i, j, k);
  double th2 = angle_eval_impl(coords, false, j, k, l);
  return (pow((rho(i, j) * rho(j, k) * rho(k, l)), (1.0 / 3.0)) * (f + (1 - f) * sin(th1)) *
          (f + (1 - f) * sin(th2)));
}

double torsion_center(Eigen::MatrixXd ijk, int j, int k) {
  double comp = ijk(j, k);
  return round(comp);
}

std::tuple<double, std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>
torsion_eval(Eigen::MatrixXd coords, int i, int j, int k, int l) {
  Eigen::VectorXd v1 = (coords.row(i) - coords.row(j));
  Eigen::VectorXd v2 = (coords.row(l) - coords.row(k));
  Eigen::Vector3d w  = (coords.row(k) - coords.row(j));

  auto            ew = w / w.norm();
  Eigen::Vector3d a1 = v1 - v1.dot(ew) * ew;
  Eigen::Vector3d a2 = v2 - v2.dot(ew) * ew;
  Eigen::Matrix3d mat;
  mat.col(0) = v2;
  mat.col(1) = v1;
  mat.col(2) = w;
  double det = mat.determinant();
  int    sgn = (det > 0) - (det < 0);

  double dot_product = a1.dot(a2) / a1.norm() * a2.norm();
  if(dot_product < -1) { dot_product = -1; }
  else if(dot_product > 1) { dot_product = 1; }
  double phi = acos(dot_product) * sgn;

  std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> grad;

  if(abs(phi) > acos(-1.0) - 1e-6) {
    Eigen::VectorXd g = w.cross(a1);
    g                 = g / g.norm();
    double A          = v1.dot(ew) / w.norm();
    double B          = v2.dot(ew) / w.norm();
    grad = std::make_tuple(g / (g.norm() * a1.norm()), -((1 - A) / a1.norm() - B / a2.norm()) * g,
                           -((1 - B) / a2.norm() - A / a1.norm()) * g, g / (g.norm() * a2.norm()));
  }
  else if(abs(phi) < 1e-6) {
    Eigen::VectorXd g = w.cross(a1);
    g                 = g / g.norm();
    double A          = v1.dot(ew) / w.norm();
    double B          = v2.dot(ew) / w.norm();
    grad = std::make_tuple(g / (g.norm() * a1.norm()), -((1 - A) / a1.norm() + B / a2.norm()) * g,
                           ((1 + B) / a2.norm() - A / a1.norm()) * g, -g / (g.norm() * a2.norm()));
  }
  else {
    double A = v1.dot(ew) / w.norm();
    double B = v2.dot(ew) / w.norm();
    grad     = std::make_tuple(
          1 / tan(phi) * a1 / a1.norm() * a1.norm() - a2 / (a1.norm() * a2.norm() * sin(phi)),
          ((1 - A) * a2 - B * a1) / (a1.norm() * a2.norm() * sin(phi)) -
            1 / tan(phi) * ((1 - A) * a1 / a1.norm() * a1.norm() - B * a2 / a2.norm() * a2.norm()),
          ((1 + B) * a1 + A * a2) / (a1.norm() * a2.norm() * sin(phi)) -
            1 / tan(phi) * ((1 + B) * a2 / pow(a2.norm(), 2) + A * a1 / pow(a1.norm(), 2)),
          1 / tan(phi) * a2 / pow(a2.norm(), 2) - a1 / (a1.norm() * a2.norm() * sin(phi)));
  }

  return std::tuple<double,
                    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>>(
    phi, grad);
}

double torsion_eval(Eigen::MatrixXd coords, int i, int j, int k, int l, bool _) {
  Eigen::VectorXd v1 = (coords.row(i) - coords.row(j)) * angstrom_impl;
  Eigen::VectorXd v2 = (coords.row(l) - coords.row(k)) * angstrom_impl;
  Eigen::Vector3d w  = (coords.row(k) - coords.row(j)) * angstrom_impl;

  auto            ew = w / w.norm();
  Eigen::Vector3d a1 = v1 - v1.dot(ew) * ew;
  Eigen::Vector3d a2 = v2 - v2.dot(ew) * ew;
  Eigen::Matrix3d mat;
  mat.col(0) = v2;
  mat.col(1) = v1;
  mat.col(2) = w;
  double det = mat.determinant();
  int    sgn = (det > 0) - (det < 0);

  double dot_product = a1.dot(a2) / a1.norm() * a2.norm();
  if(dot_product < -1) { dot_product = -1; }
  else if(dot_product > 1) { dot_product = 1; }
  double phi = acos(dot_product) * sgn;

  return phi;
}

auto b_matrix(Eigen::MatrixXd geom, InternalCoordinates int_coords) {
  std::vector<std::vector<Eigen::Vector3d>> B(
    int_coords.size(), std::vector<Eigen::Vector3d>(geom.size() / 3, Eigen::Vector3d::Zero()));

  for(int i = 0; i < int_coords.size(); i++) {
    if(int_coords[i].type == "Bond") {
      auto            grad   = bond_eval(geom, int_coords[i].i, int_coords[i].j);
      auto            _grad  = std::get<1>(grad);
      Eigen::Vector3d atom_i = std::get<0>(_grad);
      Eigen::Vector3d atom_j = std::get<1>(_grad);
      B[i][int_coords[i].i] += atom_i;
      B[i][int_coords[i].j] += atom_j;
    }
    else if(int_coords[i].type == "Angle") {
      auto grad  = angle_eval_impl(geom, int_coords[i].i, int_coords[i].j, int_coords[i].k);
      auto _grad = std::get<1>(grad);
      Eigen::Vector3d atom_i = std::get<0>(_grad);
      Eigen::Vector3d atom_j = std::get<1>(_grad);
      Eigen::Vector3d atom_k = std::get<2>(_grad);
      B[i][int_coords[i].i] += atom_i;
      B[i][int_coords[i].j] += atom_j;
      B[i][int_coords[i].k] += atom_k;
    }
    else if(int_coords[i].type == "Torsion") {
      auto grad =
        torsion_eval(geom, int_coords[i].i, int_coords[i].j, int_coords[i].k, int_coords[i].l);
      auto            _grad  = std::get<1>(grad);
      Eigen::Vector3d atom_i = std::get<0>(_grad);
      Eigen::Vector3d atom_j = std::get<1>(_grad);
      Eigen::Vector3d atom_k = std::get<2>(_grad);
      Eigen::Vector3d atom_l = std::get<3>(_grad);
      B[i][int_coords[i].i] += atom_i;
      B[i][int_coords[i].j] += atom_j;
      B[i][int_coords[i].k] += atom_k;
      B[i][int_coords[i].l] += atom_l;
    }
  }

  Eigen::MatrixXd b_mat(int_coords.size(), geom.size());
  for(int i = 0; i < int_coords.size(); i++) {
    for(int j = 0; j < geom.size() / 3; j++) {
      b_mat(i, j * 3)       = B[i][j][0];
      b_mat(i, (j * 3) + 1) = B[i][j][1];
      b_mat(i, (j * 3) + 2) = B[i][j][2];
    }
  }

  for(int i = 0; i < int_coords.size(); i++) {
    for(int j = 0; j < geom.size(); j++) {
      if(std::isnan(b_mat(i, j)) || b_mat(i, j) > 1e25 || b_mat(i, j) < -1e25) {
        b_mat(i, j) = 0.0;
      }
    }
  }

  return b_mat;
}

Eigen::MatrixXd pinv(ExecutionContext& ec, const Eigen::MatrixXd& A) {
  // Perform Singular Value Decomposition
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::VectorXd                   D = svd.singularValues();
  Eigen::MatrixXd                   U = svd.matrixU();
  Eigen::MatrixXd                   V = svd.matrixV();

  const double thre = 1e3;

  Eigen::VectorXd gaps = D.head(D.size() - 1).array() / D.tail(D.size() - 1).array();

  int n = D.size();
  for(int i = 0; i < gaps.size(); ++i) {
    if(gaps(i) > thre) {
      n = i + 1;
      break;
    }
  }

  for(int i = n; i < D.size(); ++i) { D(i) = 0; }

  for(int i = 0; i < n; ++i) { D(i) = 1 / D(i); }

  Eigen::MatrixXd D_inv = D.asDiagonal();
  return U * D_inv * V.transpose();
}

Eigen::MatrixXd hessian_guess(ExecutionContext& ec, Eigen::MatrixXd geom, Eigen::MatrixXd rho,
                              InternalCoordinates int_coords) {
  Eigen::MatrixXd H(int_coords.size(), int_coords.size());

  for(int i = 0; i < int_coords.size(); i++) {
    for(int j = 0; j < int_coords.size(); j++) { H(i, j) = 0.0; }
  }

  for(int i = 0; i < int_coords.size(); i++) {
    double comp = int_coords[i].hessian_component(rho);
    H(i, i)     = comp;
  }

  return H;
}

Eigen::MatrixXd update_hessian(Eigen::MatrixXd hessian, Eigen::VectorXd dq, Eigen::VectorXd dg) {
  // gradients is an nx3 matrix

  // Compute dH1: outer product of dg with itself, divided by the dot product of dq and dg
  double          dq_dot_dg = dq.dot(dg);
  Eigen::MatrixXd dH1       = (dg * dg.transpose()) / dq_dot_dg;

  // Compute dH2: matrix multiplication with H, outer product of dq, and division by dot product
  double          dq_dot_H_dq = dq.dot(hessian * dq);
  Eigen::MatrixXd dH2         = (hessian * (dq * dq.transpose()) * hessian) / dq_dot_H_dq;

  // Compute dH as the difference
  Eigen::MatrixXd dH = dH1 - dH2;

  // Return the updated Hessian
  return hessian + dH;
}

Eigen::VectorXd eval_geom(ExecutionContext& ec, InternalCoordinates coords, Eigen::MatrixXd geom,
                          const Eigen::VectorXd* q_template = nullptr) {
  int size = coords.size();

  Eigen::VectorXd q(size);

  for(int i = 0; i < size; i++) {
    if(coords[i].type == "Bond") { q[i] = bond_eval(geom, coords[i].i, coords[i].j, false); }
    else if(coords[i].type == "Angle") {
      q[i] = angle_eval_impl(geom, false, coords[i].i, coords[i].j, coords[i].k);
    }
    else if(coords[i].type == "Torsion") {
      q[i] = torsion_eval(geom, coords[i].i, coords[i].j, coords[i].k, coords[i].l, false);
    }
  }

  if(q_template == nullptr) { return q; }

  std::vector<InternalCoordinate> swapped;
  std::set<InternalCoordinate>    candidates;

  for(int i = 0; i < size; i++) {
    if(coords[i].type != "Torsion") { continue; }
    else {
      double diff = q(i) - (*q_template)(i);
      if(abs(abs(diff) - 2 * acos(-1.0)) < acos(-1.0) / 2) {
        if(diff >= 0) { q(i) -= 2 * acos(-1.0); }
        else { q(i) += 2 * acos(-1.0); }
      }
      else if(abs(abs(diff) - acos(-1.0)) < acos(-1.0) / 2) {
        if(diff >= 0) { q(i) -= acos(-1.0); }
        else { q(i) += acos(-1.0); }
        swapped.push_back(coords[i]);
        candidates.insert(
          InternalCoordinate(coords[i].i, coords[i].j, coords[i].k, (*coords[i].angle0).value));
        candidates.insert(
          InternalCoordinate(coords[i].j, coords[i].k, coords[i].l, (*coords[i].angle1).value));
      }
    }
  }

  for(int i = 0; i < size; i++) {
    // if i is not an angle and not in candidates
    if(coords[i].type != "Angle" ||
       std::find(candidates.begin(), candidates.end(), coords[i]) == candidates.end()) {
      continue;
    }
    else {
      bool condition_met = true;
      for(int j = 0; j < size; j++) {
        if(coords[j].type == "Torsion") {
          if((coords[i].i == coords[j].i && coords[i].j == coords[j].j &&
              coords[i].k == coords[j].k) ||
             (coords[i].i == coords[j].j && coords[i].j == coords[j].k &&
              coords[i].k == coords[j].l)) {
            bool all_angles_in_candidates = true;

            if(std::find(candidates.begin(), candidates.end(), *(coords[j].angle0)) ==
                 candidates.end() ||
               std::find(candidates.begin(), candidates.end(), *(coords[j].angle1)) ==
                 candidates.end()) {
              all_angles_in_candidates = false;
            }

            if(!(std::find(swapped.begin(), swapped.end(), coords[j]) != swapped.end() ||
                 all_angles_in_candidates)) {
              condition_met = false;
              break;
            }
          }
        }
      }
      if(condition_met) { q(i) = 2 * acos(-1.0) - q(i); }
    }
  }

  return q;
}

Eigen::VectorXd eval_geom_(InternalCoordinates coords, Eigen::MatrixXd geom) {
  // Calculate q (evaluating each coordinate in InternalCoordinates)
  Eigen::VectorXd q(coords.size());
  for(int i = 0; i < coords.size(); ++i) {
    if(coords[i].type == "Bond") { q[i] = bond_eval(geom, coords[i].i, coords[i].j, false); }
    else if(coords[i].type == "Angle") {
      q[i] = angle_eval_impl(geom, false, coords[i].i, coords[i].j, coords[i].k);
    }
    else if(coords[i].type == "Torsion") {
      q[i] = torsion_eval(geom, coords[i].i, coords[i].j, coords[i].k, coords[i].l, false);
    }
  }

  return q;
}

double update_trust(double trust, double dE, double dE_predicted, Eigen::VectorXd dq) {
  double r;

  if(dE != 0.0) { r = dE / dE_predicted; }
  else { r = 1.0; }

  if(r < 0.25) { return dq.norm() / 4; }
  else if(r > 0.75 && std::abs(dq.norm() - trust) < 1e-10) { return 2 * trust; }
  else { return trust; }
}

std::vector<double> g(double y0, double y1, double g0, double g1, double c) {
  double a = c + 3 * (y0 - y1) + 2 * g0 + g1;
  double b = -2 * c - 4 * (y0 - y1) - 3 * g0 - g1;
  return {a, b, c, g0, y0};
}

std::pair<double, double> fit_cubic(double y0, double y1, double g0, double g1) {
  double              a = 2 * (y0 - y1) + g0 + g1;
  double              b = -3 * (y0 - y1) - 2 * g0 - g1;
  std::vector<double> p = {a, b, g0, y0};

  // Derivative of the cubic polynomial p
  std::vector<double> p_prime = {3 * p[0], 2 * p[1], p[2]};

  // Find the roots of p_prime, which is a quadratic equation
  std::vector<double> r;
  double              discriminant = p_prime[1] * p_prime[1] - 4 * p_prime[0] * p_prime[2];
  if(discriminant >= 0) {
    r.push_back((-p_prime[1] + std::sqrt(discriminant)) / (2 * p_prime[0]));
    r.push_back((-p_prime[1] - std::sqrt(discriminant)) / (2 * p_prime[0]));
  }

  if(r.empty()) { return {-100.0, -100.0}; }

  std::sort(r.begin(), r.end());

  double minim, maxim;
  if(p[0] > 0) {
    maxim = r[0];
    minim = r[1];
  }
  else {
    minim = r[0];
    maxim = r[1];
  }

  if(0 < maxim && maxim < 1 && std::abs(minim - 0.5) > std::abs(maxim - 0.5)) {
    return {-100.0, -100.0};
  }

  // Evaluate the polynomial at the minimum point
  double result = p[0] * minim * minim + p[1] * minim + p[2];
  return {minim, result};
}

std::pair<double, double> quart_min(const std::vector<double>& p) {
  std::vector<double> p_prime = {3 * p[0], 2 * p[1], p[2]}; // First derivative

  // Find the roots of the quadratic equation p_prime
  std::vector<double> r;
  double              discriminant = p_prime[1] * p_prime[1] - 4 * p_prime[0] * p_prime[2];
  if(discriminant >= 0) {
    r.push_back((-p_prime[1] + std::sqrt(discriminant)) / (2 * p_prime[0]));
    r.push_back((-p_prime[1] - std::sqrt(discriminant)) / (2 * p_prime[0]));
  }

  double minim;
  if(r.size() == 0) { return {-100.0, 0.0}; }
  else if(r.size() == 1) { minim = r[0]; }
  else { minim = (r[0] == *std::max_element(r.begin(), r.end())) ? r[1] : r[0]; }

  // Evaluate the polynomial at the minimum
  double result =
    p[0] * minim * minim * minim + p[1] * minim * minim + p[2] * minim + p[3] * minim + p[4];

  return {minim, result};
}

std::pair<double, double> fit_quartic(double y0, double y1, double g0, double g1) {
  double D =
    -((g0 + g1) * (g0 + g1)) - 2 * g0 * g1 + 6 * (y1 - y0) * (g0 + g1) - 6 * (y1 - y0) * (y1 - y0);
  if(D < 1e-11) { return {-100.0, -100.0}; }

  double              m     = -5 * g0 - g1 - 6 * y0 + 6 * y1;
  double              sqrtD = std::sqrt(2 * D);
  std::vector<double> p1    = g(y0, y1, g0, g1, 0.5 * (m + sqrtD));
  std::vector<double> p2    = g(y0, y1, g0, g1, 0.5 * (m - sqrtD));

  if(p1[0] < 0 && p2[0] < 0) { return {-100.0, -100.0}; }

  auto [minim1, minval1] = quart_min(p1);
  auto [minim2, minval2] = quart_min(p2);

  return (minval1 < minval2) ? std::make_pair(minim1, minval1) : std::make_pair(minim2, minval2);
}

std::tuple<double, double> linear_search(double e0, double e1, double g0, double g1) {
  auto [t, E] = fit_quartic(e0, e1, g0, g1);

  if(t == -100.0 || t < -1.0 || t > 2.0) {
    auto cubic = fit_cubic(e0, e1, g0, g1);
    t          = std::get<0>(cubic);
    E          = std::get<1>(cubic);
    if(t == -100.0 || t < 0.0 || t > 1.0) {
      if(e0 <= e1) { return std::tuple<double, double>(0, e0); }
      else { return std::tuple<double, double>(1, e1); }
    }
  }

  return std::tuple<double, double>(t, E);
}

double find_root(ExecutionContext& ec, std::function<double(double)> f, double lim) {
  double d = 1.0;
  for(int i = 0; i < 1000; i++) {
    double val = f(lim - d);
    if(val > 0) { break; }
    d /= 2; // find d so that f(lim-d) > 0
  }

  double x   = lim - d; // initial guess
  double dx  = 1e-10;   // step for numerical derivative
  double fx  = f(x);
  double err = std::abs(fx);

  for(int j = 0; j < 1000; j++) {
    double fxpdx = f(x + dx);
    double dxf   = (fxpdx - fx) / dx;
    x -= fx / dxf;
    fx             = f(x);
    double err_new = std::abs(fx);
    if(err_new >= err) { return x; }
    err = err_new;
  }

  throw std::runtime_error("Cannot find root during geometry optimization");
}

std::tuple<Eigen::VectorXd, double, bool> quadratic_step(ExecutionContext& ec, Eigen::VectorXd g,
                                                         Eigen::MatrixXd H, Eigen::VectorXd w,
                                                         double trust) {
  // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigSolver_(H);

  // Eigenvalue decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_solver((H + H.transpose()) / 2);
  Eigen::VectorXd                                ev = eig_solver.eigenvalues();

  // Construct RFO matrix (rfo)
  Eigen::MatrixXd rfo(H.rows() + 1, H.cols() + 1);
  rfo.topLeftCorner(H.rows(), H.cols()) = H;
  rfo.topRightCorner(H.rows(), 1)       = g;
  rfo.bottomLeftCorner(1, H.cols())     = g.transpose();
  rfo(H.rows(), H.cols())               = 0;

  // Eigenvalue decomposition of the RFO matrix
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> rfo_eig_solver((rfo + rfo.transpose()) / 2);
  Eigen::VectorXd                                D = rfo_eig_solver.eigenvalues();
  Eigen::MatrixXd                                V = rfo_eig_solver.eigenvectors();

  double l = D(0);

  Eigen::VectorXd dq_temp(V.rows() - 1);
  for(int i = 0; i < V.rows() - 1; i++) { dq_temp(i) = V(i, 0); }

  Eigen::VectorXd dq = dq_temp / V(V.rows() - 1, 0);

  bool on_sphere = false;
  if(dq.norm() <= trust) { on_sphere = false; }
  else {
    // Define the steplength function
    auto steplength = [&H, &g, trust](double l) -> double {
      Eigen::MatrixXd I      = Eigen::MatrixXd::Identity(H.rows(), H.cols());
      Eigen::VectorXd result = (l * I - H).ldlt().solve(g);
      return result.norm() - trust;
    };

    // Minimization on sphere (find root of the steplength function)
    l         = find_root(ec, steplength, ev(0)); // minimization on sphere
    dq        = (l * Eigen::MatrixXd::Identity(H.rows(), H.cols()) - H).ldlt().solve(g);
    on_sphere = true;
  }

  // Predicted energy change (dE)
  double dE = g.dot(dq) + 0.5 * dq.transpose() * H * dq;

  return std::tuple<Eigen::VectorXd, double, bool>(dq, dE, on_sphere);
}

double rms(const Eigen::MatrixXd A) {
  double var = 0;
  for(int i = 0; i < A.rows(); i++) {
    for(int j = 0; j < A.cols(); j++) { var += pow(A(i, j), 2); }
  }
  return sqrt(var) / A.size();
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> update_geom(ExecutionContext& ec, Eigen::MatrixXd geom,
                                                         Eigen::VectorXd q, Eigen::VectorXd dq,
                                                         Eigen::MatrixXd     B_inv,
                                                         InternalCoordinates int_coords) {
  double                                                       thre = 1e-6;
  std::tuple<Eigen::MatrixXd, Eigen::VectorXd, double, double> keep_first;
  bool                                                         converged  = false;
  Eigen::MatrixXd                                              coords_new = geom;

  for(int i = 0; i < 20; i++) {
    Eigen::MatrixXd B_inv_dq = (B_inv * dq) / angstrom_impl;

    for(int j = 0; j < B_inv_dq.size() / 3; j++) {
      coords_new(j, 0) += B_inv_dq(j * 3);
      coords_new(j, 1) += B_inv_dq(j * 3 + 1);
      coords_new(j, 2) += B_inv_dq(j * 3 + 2);
    }

    coords_new = (coords_new).eval();

    auto dcart_rms = rms(coords_new - geom);
    geom           = coords_new;

    auto q_new = eval_geom(ec, int_coords, coords_new, &q);

    double dq_rms = rms(q_new - q);
    dq            = dq - (q_new - q);
    q             = q_new;
    if(dcart_rms < thre) {
      converged = true;
      break;
    }
    if(i == 0) { keep_first = std::make_tuple(coords_new, q, dcart_rms, dq_rms); }
  }
  if(!converged) {
    coords_new = std::get<0>(keep_first);
    q          = std::get<1>(keep_first);
    if(ec.print()) { std::cout << "Failed to converge in update_geom" << std::endl; }
  }

  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> output;
  output = std::make_tuple(q, coords_new);

  return output;
}

bool is_converged(Eigen::VectorXd forces, Eigen::VectorXd step, bool on_sphere) { // N
  // Define convergence criteria
  std::vector<std::tuple<std::string, double, double>> criteria = {
    {"Gradient RMS", rms(forces), 0.15e-3},
    {"Gradient maximum",
     *std::max_element(forces.begin(), forces.end(),
                       [](double a, double b) { return std::abs(a) < std::abs(b); }),
     0.45e-3}};

  if(on_sphere) { criteria.push_back({"Minimization on sphere", false, 0.0}); }
  else {
    criteria.push_back({"Step RMS", rms(step), 1.2e-3});
    criteria.push_back(
      {"Step maximum",
       *std::max_element(step.begin(), step.end(),
                         [](double a, double b) { return std::abs(a) < std::abs(b); }),
       1.8e-3});
  }

  bool all_matched = true;

  for(const auto& crit: criteria) {
    bool        result = false;
    std::string msg;

    if(std::get<2>(crit) != 0.0) { result = std::get<1>(crit) < std::get<2>(crit); }
    else { result = std::get<1>(crit) == false; }

    if(!result) { all_matched = false; }
  }

  return all_matched;
}

std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
           Eigen::MatrixXd>
optimizer_berny(ExecutionContext& ec, ChemEnv& chem_env, double energy, Eigen::MatrixXd gradients,
                bool first, OptPoint best, OptPoint previous, OptPoint predicted,
                OptPoint interpolated, OptPoint future, double trust, Eigen::MatrixXd hessian,
                InternalCoordinates int_coords) {
  // initial geometry setup
  std::tuple<std::vector<int>, Eigen::MatrixXd> init_geom      = get_init_geometry(ec, chem_env);
  std::vector<int>                              atomic_numbers = std::get<0>(init_geom);
  Eigen::MatrixXd                               geom           = std::get<1>(init_geom);
  int                                           num_atoms      = atomic_numbers.size();

  // calculating rho and weights
  Eigen::MatrixXd rho = calculate_rho(num_atoms, geom, atomic_numbers);
  Eigen::VectorXd weights(int_coords.size());
  for(int i = 0; i < int_coords.size(); i++) {
    InternalCoordinate coord = int_coords[i];
    if(coord.type == "Bond") { weights(i) = bond_weight(rho, coord.i, coord.j); }
    else if(coord.type == "Angle") {
      weights(i) = angle_weight(rho, geom, coord.i, coord.j, coord.k);
    }
    else if(coord.type == "Torsion") {
      weights(i) = torsion_weight(rho, geom, coord.i, coord.j, coord.k, coord.l);
    }
  }

  // calculating the b_matrix and associated objects
  Eigen::MatrixXd b           = b_matrix(geom, int_coords);
  auto            b_t         = b.transpose();
  Eigen::MatrixXd b_tdot      = b * b_t;
  auto            b_tdot_pinv = pinv(ec, b_tdot);
  auto            b_inv       = b_t * b_tdot_pinv;
  auto            b_inv_t     = b_inv.transpose();
  auto            product     = b_inv_t * gradients.transpose();

  OptPoint current;
  future  = OptPoint(eval_geom(ec, int_coords, geom));
  current = OptPoint(future.q, energy, product);

  if(!first) {
    hessian      = update_hessian(hessian, current.q - best.q, current.g - best.g);
    trust        = update_trust(trust, current.E - previous.E, predicted.E - interpolated.E,
                                predicted.q - interpolated.q);
    auto dq      = best.q - current.q;
    auto [t, E]  = linear_search(current.E, best.E, current.g.dot(dq), best.g.dot(dq));
    interpolated = OptPoint(current.q + t * dq, E, current.g + t * (best.g - current.g));
  }
  else {
    hessian      = hessian_guess(ec, geom, rho, int_coords);
    interpolated = OptPoint(current.q, current.E, current.g);
  }

  if(trust < 1e-6) { throw std::runtime_error("The trust radius got too small, check forces?"); }

  auto            proj     = b * b_inv;
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(int_coords.size(), int_coords.size());
  Eigen::MatrixXd H_proj   = proj * hessian * proj + 1000 * (identity - proj);
  auto            temp     = proj * interpolated.g;
  auto [dq, dE, on_sphere] = quadratic_step(ec, temp, H_proj, weights, trust);
  predicted                = OptPoint(interpolated.q + dq, interpolated.E + dE);
  dq                       = predicted.q - current.q;

  Eigen::VectorXd                              q;
  std::tuple<Eigen::VectorXd, Eigen::MatrixXd> updated_q_geom =
    update_geom(ec, geom, current.q, predicted.q - current.q, b_inv, int_coords);
  q        = std::get<0>(updated_q_geom);
  geom     = std::get<1>(updated_q_geom);
  future   = OptPoint(q);
  previous = current;
  if(first || current.E < best.E) { best = current; }

  first = false;

  bool _converged = is_converged(current.g, future.q, on_sphere);
  if(_converged) {}

  Eigen::RowVectorXd flat(geom.size());
  int                idx = 0;
  for(int i = 0; i < geom.rows(); i++) {
    for(int j = 0; j < geom.cols(); j++) {
      flat[idx] = geom(i, j);
      idx++;
    }
  }

  // convert new geometry to bohr and return
  std::tuple<Eigen::RowVectorXd, OptPoint, OptPoint, OptPoint, OptPoint, OptPoint, double,
             Eigen::MatrixXd>
    resultant =
      std::make_tuple(flat, best, previous, predicted, interpolated, future, trust, hessian);

  return resultant;
}

std::stringstream print_bond_heading(std::stringstream ss) {
  ss << std::endl << std::string(60, '-') << std::endl;
  ss << std::setw(25) << "Bond Lengths" << std::endl << std::endl;

  ss << std::setw(3) << std::left << "i"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
     << " " << std::right << std::setw(20) << std::setw(3) << std::left << "Length (Angstroms)"
     << " " << std::right << std::setw(20) << std::setw(3) << std::left << "Length (Bohr)"
     << " " << std::right << std::setw(20) << std::endl;

  return ss;
}

std::stringstream print_angle_heading(std::stringstream ss) {
  ss << std::endl << std::string(60, '-') << std::endl;
  ss << std::setw(18) << "Bond Angles" << std::endl << std::endl;

  ss << std::setw(3) << std::left << "i"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
     << " " << std::right << std::setw(13) << std::setw(3) << std::left << "Angle (degrees)"
     << " " << std::right << std::setw(14) << std::endl;

  return ss;
}

std::stringstream print_torsion_heading(std::stringstream ss) {
  ss << std::endl << std::string(60, '-') << std::endl;
  ss << std::setw(20) << "Torsional Angles" << std::endl << std::endl;

  ss << std::setw(3) << std::left << "i"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "j"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "k"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "l"
     << " " << std::right << std::setw(14) << std::setw(3) << std::left << "Angle (degrees)"
     << " " << std::right << std::setw(14) << std::endl;

  return ss;
}

} // namespace exachem::task
