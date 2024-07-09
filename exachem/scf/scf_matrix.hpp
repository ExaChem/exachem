/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/common/ec_molden.hpp"
#include <string>

using TensorType = double;
namespace exachem::scf {
class SCFMatrix {
public:
  template<typename T>
  Matrix read_scf_mat(std::string matfile);
  template<typename T>
  void write_scf_mat(Matrix& C, std::string matfile);
};
} // namespace exachem::scf
