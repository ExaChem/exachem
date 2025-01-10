/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/common/chemenv.hpp"

namespace exachem::task {

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
                                        const std::vector<std::vector<std::vector<double>>>& apuv, const int& i,
                                        const int& j, const int& k);
std::vector<double> cross_product(std::vector<double>& atom0, std::vector<double>& atom1);
double              dot_product(std::vector<double>& atom0, std::vector<double>& atom1);

} // namespace exachem::task
