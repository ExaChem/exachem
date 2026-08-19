/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/atom_info.hpp"
#include "exachem/common/chemenv.hpp"
#include "exachem/common/libint2_includes.hpp"
#include <iostream>

class ECPDOS {
private:
  std::vector<double> gaussian_smearing(std::vector<double>& x, double x0, double fwhm);
  std::vector<double> lorentzian_smearing(std::vector<double>& x, double x0, double fwhm);

public:
  ECPDOS() = default;
  void write_pdos(ChemEnv& chem_env, Matrix& S, Matrix& C_a, std::vector<double>& eps_a,
                  std::string files_prefix);

  void write_pdos(ChemEnv& chem_env, Matrix& S, Matrix& C_a, std::vector<double>& eps_a,
                  Matrix& C_b, std::vector<double>& eps_b, std::string files_prefix);
};