/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "json_data.hpp"
#include "misc.hpp"
#include <iostream>

template<typename T>
void reorder_molden_orbitals(const bool is_spherical, std::vector<AtomInfo>& atominfo, Matrix& smat,
                             Matrix& dmat, const bool reorder_cols = true,
                             const bool reorder_rows = true);

// TODO: is this needed? - currently does not make a difference
libint2::BasisSet renormalize_libint_shells(const SystemData& sys_data, libint2::BasisSet& shells);

void read_geom_molden(const SystemData& sys_data, std::vector<libint2::Atom>& atoms);

// TODO: is this needed? - currently does not make a difference
libint2::BasisSet read_basis_molden(const SystemData& sys_data, libint2::BasisSet& shells);

template<typename T>
void read_molden(const SystemData& sys_data, libint2::BasisSet& shells,
                 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_alpha,
                 Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C_beta);
