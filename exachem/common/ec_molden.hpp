/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "atom_info.hpp"
#include "chemenv.hpp"
#include "libint2_includes.hpp"
#include <iostream>

class ECMolden {
private:
  std::string read_option(std::string line);
  bool        is_comment(const std::string line);
  bool        is_in_line(const std::string str, const std::string line);

public:
  ECMolden()             = default;
  bool molden_exists     = false;
  bool molden_file_valid = false;

  template<typename T>
  void reorder_molden_orbitals(const bool is_spherical, std::vector<AtomInfo>& atominfo,
                               Matrix& smat, Matrix& dmat, const bool reorder_cols = true,
                               const bool reorder_rows = true);

  // TODO: is this needed? - currently does not make a difference
  libint2::BasisSet renormalize_libint_shells(libint2::BasisSet& shells);

  void read_geom_molden(ChemEnv& chem_env);

  bool check_molden(std::string moldenfile);

  // TODO: is this needed? - currently does not make a difference
  libint2::BasisSet read_basis_molden(const ChemEnv& chem_env);

  template<typename T>
  void read_molden(ChemEnv&                                                           chem_env,
                   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_alpha,
                   Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_beta);
};
