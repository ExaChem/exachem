#pragma once

#include "common/ec_molden.hpp"
#include <string>

using TensorType = double;

class SCFMatrix {
public:
  template<typename T>
  Matrix read_scf_mat(std::string matfile);
  template<typename T>
  void write_scf_mat(Matrix& C, std::string matfile);
};