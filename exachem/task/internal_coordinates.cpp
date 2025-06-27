/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * This file contains a C++ version of the Internal Coordinates implementation
 * from the PyBerny Molecular structure optimizer (https://github.com/jhrmnn/pyberny),
 * licensed under the Mozilla Public License, v. 2.0.
 * You can obtain a copy of the MPL-2.0 license at http://mozilla.org/MPL/2.0
 *
 * This file is distributed under the Mozilla Public License, v. 2.0.
 *
 */

#include "exachem/task/internal_coordinates.hpp"
#include <queue>
#include <unordered_set>

std::vector<double> atom_radii_int = {
  0.38, 0.32, 1.34, 0.9,  0.82, 0.77, 0.75, 0.73, 0.71, 0.69, 1.54, 1.3,  1.18, 1.11, 1.06,
  1.02, 0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31,
  1.26, 1.22, 1.19, 1.16, 1.14, 1.1,  2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35,
  1.31, 1.53, 1.48, 1.44, 1.41, 1.38, 1.35, 1.33, 1.3,  2.25, 1.98, 1.69, 0.00, 0.00, 0.00,
  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.6,  1.5,  1.38, 1.46, 1.59,
  1.28, 1.37, 1.28, 1.44, 1.49, 1.48, 1.47, 1.46, 0.00, 0.00, 1.45};

namespace exachem::task {

InternalCoordinate::InternalCoordinate(int _i, int _j, double _value) {
  type  = "Bond";
  i     = _i;
  j     = _j;
  value = _value;
}

InternalCoordinate::InternalCoordinate(int _i, int _j, int _k, double _value) {
  type  = "Angle";
  i     = _i;
  j     = _j;
  k     = _k;
  value = _value;
}

InternalCoordinate::InternalCoordinate(int _i, int _j, int _k, int _l, double _value,
                                       InternalCoordinate* _angle0, InternalCoordinate* _angle1) {
  type   = "Torsion";
  i      = _i;
  j      = _j;
  k      = _k;
  l      = _l;
  value  = _value;
  angle0 = _angle0;
  angle1 = _angle1;
}

bool InternalCoordinate::operator<(const InternalCoordinate other) const { return false; }

bool InternalCoordinate::operator==(const InternalCoordinate& other) const {
  if(other.type != type) { return false; }
  else {
    if(type == "Bond" && (other.i != i || other.j != j)) { return false; }
    else if(type == "Angle" && (other.i != i || other.j != j || other.k != k)) { return false; }
    else if(type == "Torsion" && (other.i != i || other.j != j || other.k != k || other.l != l)) {
      return false;
    }
    else { return true; }
  }
}

double InternalCoordinate::hessian_component(Eigen::MatrixXd rho) {
  if(type == "Bond") { return 0.45 * rho(i, j); }
  else if(type == "Angle") { return 0.15 * (rho(i, j) * rho(j, k)); }
  else if(type == "Torsion") { return 0.005 * (rho(i, j) * rho(j, k)) * rho(k, l); }
  return -1.0;
}

InternalCoordinate InternalCoordinates::operator[](int idx) { return coords[idx]; }

int InternalCoordinates::size() { return coords.size(); }

void InternalCoordinates::push_back(InternalCoordinate coord) { coords.push_back(coord); }

// function to print all internal coordinates
void InternalCoordinates::print(ExecutionContext& ec) {
  // bonds, angles, and torsions are computed in blocks
  // meaning that a while loop checking the type can be used

  if(ec.print()) {
    const double      ang2bohr = exachem::constants::ang2bohr;
    std::stringstream ss;

    ss << "Printing Internal Coordinates" << std::endl;

    // bond lengths
    bool print_bond    = false;
    bool print_angle   = false;
    bool print_torsion = false;

    int num_bonds    = 0;
    int num_angles   = 0;
    int num_torsions = 0;

    for(size_t i = 0; i < coords.size(); i++) {
      if(coords[i].type == "Bond") {
        if(print_bond == false) {
          ss << std::endl << std::string(60, '-') << std::endl;
          ss << std::setw(28) << "Bond Lengths" << std::endl << std::endl;

          ss << std::setw(6) << std::left << "i"
             << " " << std::setw(6) << std::left << "j"
             << " " << std::right << std::setw(20) << "Length (Angstroms)"
             << " " << std::right << std::setw(17) << "Length (Bohr)" << std::endl;
          print_bond = true;
        }
        ss << std::setw(6) << std::left << coords[i].i << " " << std::setw(6) << std::left
           << coords[i].j << " " << std::right << std::setw(16) << std::fixed
           << std::setprecision(10) << coords[i].value << " " << std::right << std::setw(20)
           << coords[i].value * ang2bohr << std::endl;
        num_bonds++;
      }
      else if(coords[i].type == "Angle") {
        if(print_angle == false) {
          ss << std::endl << "Number of Bonds: " << num_bonds << std::endl;
          ss << std::endl << std::string(60, '-') << std::endl;
          ss << std::setw(22) << "Bond Angles" << std::endl << std::endl;

          ss << std::setw(6) << std::left << "i"
             << " " << std::setw(6) << std::left << "j"
             << " " << std::setw(6) << std::left << "k"
             << " " << std::right << std::setw(12) << "Angle (degrees)" << std::endl;
          print_angle = true;
        }
        ss << std::setw(6) << std::left << coords[i].i << " " << std::setw(6) << std::left
           << coords[i].j << " " << std::setw(6) << std::left << coords[i].k << " " << std::fixed
           << std::setw(10) << std::setprecision(4) << std::right
           << coords[i].value * 180 / acos(-1.0) << " " << std::endl; // j center, i0, j1
        num_angles++;
      }
      else if(coords[i].type == "Torsion") {
        if(print_torsion == false) {
          ss << std::endl << "Number of Angles: " << num_angles << std::endl;
          ss << std::endl << std::string(60, '-') << std::endl;
          ss << std::setw(30) << "Torsional Angles" << std::endl << std::endl;

          ss << std::setw(6) << std::left << "i"
             << " " << std::setw(6) << std::left << "j"
             << " " << std::setw(6) << std::left << "k"
             << " " << std::setw(6) << std::left << "l"
             << " " << std::right << std::setw(12) << "Angle (degrees)" << std::endl;
          print_torsion = true;
        }
        ss << std::setw(6) << std::left << coords[i].i << " " << std::setw(6) << std::left
           << coords[i].j << " " << std::setw(6) << std::left << coords[i].k << " " << std::setw(6)
           << std::left << coords[i].l << " " << std::fixed << std::setw(10) << std::setprecision(4)
           << std::right << coords[i].value * 180 / acos(-1.0) << " "
           << std::endl; // j center, i0, j1
        num_torsions++;
      }
    }

    ss << std::endl;
    if(!print_torsion) { ss << "Number of Angles: " << num_angles << std::endl; }

    ss << "Number of Torsional Angles: " << num_torsions << std::endl;

    std::cout << ss.str() << std::endl;
  }
}

double angle_eval(Eigen::MatrixXd coords, bool _, int i, int j, int k) {
  Eigen::RowVectorXd v1 = (coords.row(i) - coords.row(j)) * exachem::constants::ang2bohr;
  Eigen::RowVectorXd v2 = (coords.row(k) - coords.row(j)) * exachem::constants::ang2bohr;

  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());
  if(dot_product < -1) { dot_product = -1; }
  else if(dot_product > 1) { dot_product = 1; }
  double phi = acos(dot_product);
  return phi;
}

std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd> get_clusters(const Eigen::MatrixXd& C) {
  int n = C.rows();
  std::vector<std::vector<int>>
    clusters; // Vector to hold the clusters (each cluster is a list of node indices)
  std::unordered_set<int> nonassigned; // Set to track unassigned nodes

  // Initialize the nonassigned set with all node indices
  for(int i = 0; i < n; ++i) { nonassigned.insert(i); }

  // Process each unassigned node
  while(!nonassigned.empty()) {
    std::vector<int> current_cluster; // Vector to hold the current cluster
    std::queue<int>  q;               // Queue for BFS

    // Start BFS from the first unassigned node
    int start_node = *nonassigned.begin();
    q.push(start_node);
    nonassigned.erase(start_node);

    while(!q.empty()) {
      int node = q.front();
      q.pop();

      current_cluster.push_back(node);

      // Check the neighbors of the current node
      for(int neighbor = 0; neighbor < n; ++neighbor) {
        if(C(node, neighbor) != 0 && nonassigned.find(neighbor) != nonassigned.end()) {
          q.push(neighbor);
          nonassigned.erase(neighbor); // Mark as assigned
        }
      }
    }

    // Add the current cluster to the clusters list
    clusters.push_back(current_cluster);
  }

  // Create the new cluster matrix C
  Eigen::MatrixXd cluster_matrix = Eigen::MatrixXd::Zero(n, n);
  for(const auto& cluster: clusters) {
    for(int i: cluster) {
      for(int j: cluster) {
        cluster_matrix(i, j) = 1.0; // Set the connections between nodes in the same cluster
      }
    }
  }

  return std::make_pair(clusters, cluster_matrix);
}

// give the InternalCoordinates object a print function
// derived from the printing functions in geometry_analysis
// add a flag to include torsional angles
// look into why pyberny wasn't calculating them
InternalCoordinates InternalCoords(ExecutionContext& ec, ChemEnv& chem_env, bool torsions) {
  InternalCoordinates coords;
  auto                data_mat  = exachem::task::process_geometry(ec, chem_env);
  int                 num_atoms = data_mat.size();
  auto bonds = exachem::task::process_bond_lengths(ec, num_atoms, data_mat, atom_radii_int);
  auto apuv  = exachem::task::calculate_atom_pair_unit_vector(data_mat, bonds, num_atoms);

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

  std::tuple<std::vector<int>, Eigen::MatrixXd> matrix_geom_tuple =
    std::make_tuple(atomic_numbers, geometry);

  auto matrix_geom = std::get<1>(matrix_geom_tuple);

  // calculating bonds
  // first calculating the base bond matrix

  Eigen::MatrixXd bondmatrix(num_atoms, num_atoms);
  for(int i = 0; i < num_atoms; i++) {
    for(int j = 0; j < num_atoms; j++) {
      double length = exachem::task::single_bond_length_optimize(ec, num_atoms, data_mat, i, j, 1.3,
                                                                 atom_radii_int);
      if(i != j && length != 0.0) { bondmatrix(i, j) = 1.0; }
      else { bondmatrix(i, j) = 0.0; }
    }
  }

  std::tuple<std::vector<std::vector<int>>, Eigen::MatrixXd> cluster_pair =
    get_clusters(bondmatrix);
  auto fragments = std::get<0>(cluster_pair);
  auto C         = std::get<1>(cluster_pair);
  auto C_total   = C;

  double shift = 0.0;
  while(!((C_total.array() == 1.0).all())) {
    for(int i = 0; i < num_atoms; i++) {
      for(int j = 0; j < num_atoms; j++) {
        if(bondmatrix(i, j) != 1.0) {
          double length = exachem::task::single_bond_length_optimize(ec, num_atoms, data_mat, i, j,
                                                                     3e50, atom_radii_int);
          if((!C_total(i, j)) && (length < atom_radii_int[data_mat[i][0] - 1] +
                                             atom_radii_int[data_mat[j][0] - 1] + shift)) {
            bondmatrix(i, j) = 1.0;
          }
        }
      }
    }
    auto temp_cluster_pair = get_clusters(bondmatrix);
    C_total                = std::get<1>(temp_cluster_pair);
    shift++;
  }

  // calculating bonds based on the bond matrix
  for(int i = 0; i < num_atoms; i++) {
    for(int j = i; j < num_atoms; j++) {
      if(bondmatrix(i, j) == 1.0) {
        double length = exachem::task::single_bond_length_optimize(ec, num_atoms, data_mat, i, j,
                                                                   3e50, atom_radii_int);
        coords.push_back(InternalCoordinate(
          i, j, length / exachem::constants::ang2bohr)); // converting to angstrom
      }
    }
  }

  // calculating angles
  for(int i = 0; i < num_atoms; i++) {     // center atom
    for(int j = 0; j < num_atoms; j++) {   // atom 1
      for(int k = i; k < num_atoms; k++) { // atom 2
        if(i != j && i != k && j != k && bondmatrix(i, j) == 1.0 && bondmatrix(j, k) == 1.0 &&
           exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k) != 0.0 &&
           angle_eval(matrix_geom, false, i, j, k) > acos(-1.0) / 4) {
          double angle = exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k);

          coords.push_back(InternalCoordinate(i, j, k, angle));
        }
      }
    }
  }

  // calculating torsions
  if(torsions) {
    std::set<std::vector<int>> used_indices;

    for(int i = 0; i < num_atoms; i++) {
      for(int j = 0; j < num_atoms; j++) {
        if(i != j && bonds[i][j] != 0) {
          for(int k = 0; k < num_atoms; k++) {
            if(i != k && j != k && bonds[j][k] != 0.0 && bonds[i][j] != 0.0 && bonds[j][k] != 0.0 &&
               exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k) !=
                 acos(-1.0)) {
              for(int l = 0; l < num_atoms; l++) {
                std::vector<int> current = {i, j, k, l};
                used_indices.insert(current);
                std::vector<int> reverse = {l, k, j, i};

                if(i != l && j != l && k != l && bonds[k][l] != 0.0 &&
                   exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, j, k, l) !=
                     acos(-1.0) &&
                   used_indices.find(reverse) != used_indices.end()) {
                  // std::vector<double> e_ij      = apuv[i][j];
                  // std::vector<double> e_jk      = apuv[j][k];
                  // std::vector<double> e_kl      = apuv[k][l];
                  // auto                cproduct0 = exachem::task::cross_product(e_ij, e_jk);
                  // auto                cproduct1 = exachem::task::cross_product(e_jk, e_kl);
                  // double              numerator = exachem::task::dot_product(cproduct0,
                  // cproduct1); double              divisor0 =
                  //   sin(exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k));
                  // double divisor1 =
                  //   sin(exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, j, k, l));

                  // double divisor   = divisor0 * divisor1;
                  // double cos_value = numerator / divisor;
                  // double value;
                  // if(cos_value < -1.0) value = acos(-1.0);
                  // else if(cos_value > 1.0) value = acos(1);
                  // else value = acos(cos_value);

                  // double cross_x = cproduct0[1] * cproduct1[2] - cproduct0[2] * cproduct1[1];
                  // double cross_y = cproduct0[2] * cproduct1[0] - cproduct0[0] * cproduct1[2];
                  // double cross_z = cproduct0[0] * cproduct1[1] - cproduct0[1] * cproduct1[0];
                  // double norm    = cross_x * cross_x + cross_y * cross_y + cross_z * cross_z;
                  // cross_x /= norm;
                  // cross_y /= norm;
                  // cross_z /= norm;
                  // double sign = 1.0;
                  // double dot  = cross_x * e_jk[0] + cross_y * e_jk[1] + cross_z * e_jk[2];
                  // if(dot < 0.0) sign = -1.0;

                  // double current_torsional_angle = value * sign;

                  double current_torsional_angle = exachem::task::single_torsional_angle(
                    ec, data_mat, num_atoms, bonds, i, j, k, l);

                  if(current_torsional_angle != 0.0 && current_torsional_angle != acos(-1.0)) {
                    InternalCoordinate coord1(
                      i, j, k,
                      exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k));
                    InternalCoordinate coord2(
                      j, k, l,
                      exachem::task::specific_bond_angle(ec, num_atoms, bonds, apuv, j, k, l));

                    // dihedral angles are disabled because they are not computed with pyberny

                    coords.push_back(
                      InternalCoordinate(i, j, k, l, current_torsional_angle, &coord1, &coord2));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return coords;
}

} // namespace exachem::task
