/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"
using namespace tamm;
using TensorType = double;

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

template<typename T>
class TAMMTensors {
public:
  std::vector<Tensor<T>> ehf_tamm_hist;

  std::vector<Tensor<T>> diis_hist;
  std::vector<Tensor<T>> fock_hist;
  std::vector<Tensor<T>> D_hist;

  std::vector<Tensor<T>> diis_beta_hist;
  std::vector<Tensor<T>> fock_beta_hist;
  std::vector<Tensor<T>> D_beta_hist;

  Tensor<T> ehf_tamm;
  Tensor<T> ehf_tmp;
  Tensor<T> ehf_beta_tmp;

  Tensor<T> H1;      // core hamiltonian
  Tensor<T> S1;      // overlap ints
  Tensor<T> T1;      // kinetic ints
  Tensor<T> V1;      // nuclear ints
  Tensor<T> QED_Dx;  // dipole ints
  Tensor<T> QED_Dy;  // dipole ints
  Tensor<T> QED_Dz;  // dipole ints
  Tensor<T> QED_Qxx; // quadrupole ints
  Tensor<T> QED_Qxy; // quadrupole ints
  Tensor<T> QED_Qxz; // quadrupole ints
  Tensor<T> QED_Qyy; // quadrupole ints
  Tensor<T> QED_Qyz; // quadrupole ints
  Tensor<T> QED_Qzz; // quadrupole ints
  Tensor<T> QED_1body;
  Tensor<T> QED_2body;
  Tensor<T> QED_energy;

  Tensor<T> X_alpha;
  Tensor<T> F_alpha; // H1+F_alpha_tmp
  Tensor<T> F_beta;
  Tensor<T> F_alpha_tmp; // computed via call to compute_2bf(...)
  Tensor<T> F_beta_tmp;
  Tensor<T> F_BC; // block-cyclic Fock matrix used in the scalapack code path
  // not allocated, shell tiled. tensor structure used to identify shell blocks in compute_2bf
  Tensor<T> F_dummy;
  Tensor<T> VXC_alpha;
  Tensor<T> VXC_beta;

  Tensor<T> C_alpha;
  Tensor<T> C_beta;
  Tensor<T> C_occ_a;
  Tensor<T> C_occ_b;
  Tensor<T> C_occ_aT;
  Tensor<T> C_occ_bT;

  Tensor<T> C_alpha_BC;
  Tensor<T> C_beta_BC;

  Tensor<T> D_alpha;
  Tensor<T> D_beta;
  Tensor<T> D_diff;
  Tensor<T> D_last_alpha;
  Tensor<T> D_last_beta;

  Tensor<T> FD_alpha;
  Tensor<T> FDS_alpha;
  Tensor<T> FD_beta;
  Tensor<T> FDS_beta;

  // DF
  Tensor<T> xyK; // n,n,ndf
  Tensor<T> xyZ; // n,n,ndf
  Tensor<T> Vm1; // ndf,ndf
};

} // namespace exachem::scf
