/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/eigen_utils.hpp"
namespace exachem::scf {
class EigenTensors {
public:
  Matrix C_alpha, C_beta, C_occ; // allocated only on rank 0 when scalapack is not used
  Matrix VXC_alpha, VXC_beta;    // allocated only on rank 0 when DFT is enabled
  Matrix G_alpha, D_alpha;       // allocated on all ranks for 4c HF, only on rank 0 otherwise.
  Matrix G_beta, D_beta; // allocated on all ranks for 4c HF, only D_beta on rank 0 otherwise.
  Matrix D_alpha_cart, D_beta_cart;
  Matrix VXC_alpha_cart, VXC_beta_cart;
  std::vector<double>                   eps_a, eps_b;
  Eigen::Vector<double, Eigen::Dynamic> dfNorm; // Normalization coefficients for DF basis
  std::vector<Matrix>                   trafo_ctos, trafo_stoc;
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    taskmap; // on all ranks for 4c HF only
};
} // namespace exachem::scf
