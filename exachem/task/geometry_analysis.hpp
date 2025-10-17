/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/chemenv.hpp"

namespace exachem::task {
class GeometryAnalyzer {
public:
  static std::vector<double>       atom_mass;
  static std::vector<double>       atom_radii;
  void                             print_geometry(ExecutionContext& ec, ChemEnv& chem_env);
  void                             geometry_analysis(ExecutionContext& ec, ChemEnv& chem_env);
  std::vector<std::vector<double>> process_geometry(ExecutionContext& ec, ChemEnv& chem_env);
  std::vector<std::vector<double>> process_bond_lengths(ExecutionContext& ec, int num_atoms,
                                                        std::vector<std::vector<double>>& data_mat);
  std::vector<std::vector<double>> process_bond_lengths(ExecutionContext& ec, int num_atoms,
                                                        std::vector<std::vector<double>>& data_mat,
                                                        std::vector<double> atom_radii_arg);
  std::vector<std::vector<std::vector<double>>>
                      calculate_atom_pair_unit_vector(std::vector<std::vector<double>>& data,
                                                      std::vector<std::vector<double>>& bonds, int& num_atoms);
  double              single_bond_length(ExecutionContext& ec, int num_atoms,
                                         std::vector<std::vector<double>>& data_mat, int i, int j);
  double              single_bond_length_optimize(ExecutionContext& ec, int num_atoms,
                                                  std::vector<std::vector<double>>& data_mat, int i, int j,
                                                  double threshold, std::vector<double> atom_radii_arg);
  double              specific_bond_angle(ExecutionContext& ec, const int& num_atoms,
                                          const std::vector<std::vector<double>>&              bonds,
                                          const std::vector<std::vector<std::vector<double>>>& apuv,
                                          const int& i, const int& j, const int& k);
  double              single_torsional_angle(const ExecutionContext&                 ec,
                                             const std::vector<std::vector<double>>& data, const int& num_atoms,
                                             const std::vector<std::vector<double>>& bonds, const int& i,
                                             const int& j, const int& k, const int& l);
  std::vector<double> cross_product(std::vector<double>& atom0, std::vector<double>& atom1);
  double              dot_product(std::vector<double>& atom0, std::vector<double>& atom1);
  // Eigen overload for internal torsion computation
  Eigen::Vector3d     cross_product(Eigen::Vector3d a, Eigen::Vector3d b);
  std::vector<double> center_of_mass(ExecutionContext&                       ec,
                                     const std::vector<std::vector<double>>& data_mat,
                                     const int&                              num_atoms);
  std::vector<std::vector<double>>
  moment_of_inertia(ExecutionContext& ec, const std::vector<std::vector<double>>& data_mat,
                    const int& num_atoms, const std::vector<double>& com);
  std::vector<double> principle_moments_of_inertia(ExecutionContext&                       ec,
                                                   const std::vector<std::vector<double>>& mot);
  void                print_com(ExecutionContext& ec, std::vector<double>& com);
  void                print_mot(ExecutionContext& ec, const std::vector<std::vector<double>>& mot);
  void print_pmots(ExecutionContext& ec, std::vector<double>& pmots, const int& num_atoms);
  std::vector<std::vector<double>> z_matrix(ExecutionContext& ec, ChemEnv& chem_env);
  void cartesian_from_z_matrix(ExecutionContext& ec, const ChemEnv& chem_env,
                               const std::vector<std::vector<double>> zmatrix);
};

// Free function wrappers (maintain older call sites expecting these names)
inline void print_geometry(ExecutionContext& ec, ChemEnv& chem_env) {
  GeometryAnalyzer ga;
  ga.print_geometry(ec, chem_env);
}
inline std::vector<std::vector<double>> process_geometry(ExecutionContext& ec, ChemEnv& chem_env) {
  GeometryAnalyzer ga;
  return ga.process_geometry(ec, chem_env);
}
inline double specific_bond_angle(ExecutionContext& ec, const int& num_atoms,
                                  const std::vector<std::vector<double>>&              bonds,
                                  const std::vector<std::vector<std::vector<double>>>& apuv,
                                  const int& i, const int& j, const int& k) {
  GeometryAnalyzer ga;
  return ga.specific_bond_angle(ec, num_atoms, bonds, apuv, i, j, k);
}
} // namespace exachem::task
