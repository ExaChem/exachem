/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2025 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/chemenv.hpp"
#include "exachem/common/libint2_includes.hpp"
#include <fstream>
#include <iostream>

class ECNWChem {
public:
  ECNWChem()               = default;
  bool nwmovecs_exists     = false;
  bool nwmovecs_file_valid = false;
  bool check_nwmovecs(std::string moldenfile);
  void write_nwmovecs(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a,
                      std::string files_prefix);
  void write_nwmovecs(ChemEnv& chem_env, Matrix& C_a, std::vector<double>& eps_a, Matrix& C_b,
                      std::vector<double>& eps_b, std::string files_prefix);

  void reorder_nwchem_orbitals(const bool is_spherical, const libint2::BasisSet& shells,
                               Matrix& nw_mat, Matrix& ec_mat);

  void reorder_ec_orbitals(const bool is_spherical, const libint2::BasisSet& shells, Matrix& nw_mat,
                           Matrix& ec_mat);

  void read_nwmovecs(ChemEnv& chem_env, Matrix& C_alpha, Matrix& C_beta, std::vector<double>& eps_a,
                     std::vector<double>& eps_b);

  template<typename T>
  void write_record(std::ofstream& file, T* data, int32_t length);

  template<typename T>
  void read_record(std::ifstream& file, T* data);
};
