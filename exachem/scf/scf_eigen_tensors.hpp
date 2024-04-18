#pragma once

#include "tamm/eigen_utils.hpp"

class EigenTensors {
public:
  Matrix C_alpha, C_beta, C_occ; // allocated only on rank 0 when scalapack is not used
  Matrix VXC_alpha, VXC_beta;    // allocated only on rank 0 when DFT is enabled
  Matrix G_alpha, D_alpha;       // allocated on all ranks for 4c HF, only on rank 0 otherwise.
  Matrix G_beta, D_beta; // allocated on all ranks for 4c HF, only D_beta on rank 0 otherwise.
  Eigen::Vector<double, Eigen::Dynamic> dfNorm; // Normalization coefficients for DF basis
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    taskmap; // on all ranks for 4c HF only
};
