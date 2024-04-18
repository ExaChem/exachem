#pragma once

#include "tamm/tamm.hpp"
using namespace tamm;
using TensorType = double;

class TAMMTensors {
public:
  std::vector<Tensor<TensorType>> ehf_tamm_hist;

  std::vector<Tensor<TensorType>> diis_hist;
  std::vector<Tensor<TensorType>> fock_hist;
  std::vector<Tensor<TensorType>> D_hist;

  std::vector<Tensor<TensorType>> diis_beta_hist;
  std::vector<Tensor<TensorType>> fock_beta_hist;
  std::vector<Tensor<TensorType>> D_beta_hist;

  Tensor<TensorType> ehf_tamm;
  Tensor<TensorType> ehf_tmp;
  Tensor<TensorType> ehf_beta_tmp;

  Tensor<TensorType> H1; // core hamiltonian
  Tensor<TensorType> S1; // overlap ints
  Tensor<TensorType> T1; // kinetic ints
  Tensor<TensorType> V1; // nuclear ints

  Tensor<TensorType> X_alpha;
  Tensor<TensorType> F_alpha; // H1+F_alpha_tmp
  Tensor<TensorType> F_beta;
  Tensor<TensorType> F_alpha_tmp; // computed via call to compute_2bf(...)
  Tensor<TensorType> F_beta_tmp;
  Tensor<TensorType> F_BC; // block-cyclic Fock matrix used in the scalapack code path
  // not allocated, shell tiled. tensor structure used to identify shell blocks in compute_2bf
  Tensor<TensorType> F_dummy;
  Tensor<TensorType> VXC_alpha;
  Tensor<TensorType> VXC_beta;

  Tensor<TensorType> C_alpha;
  Tensor<TensorType> C_beta;
  Tensor<TensorType> C_occ_a;
  Tensor<TensorType> C_occ_b;
  Tensor<TensorType> C_occ_aT;
  Tensor<TensorType> C_occ_bT;

  Tensor<TensorType> C_alpha_BC;
  Tensor<TensorType> C_beta_BC;

  Tensor<TensorType> D_alpha;
  Tensor<TensorType> D_beta;
  Tensor<TensorType> D_diff;
  Tensor<TensorType> D_last_alpha;
  Tensor<TensorType> D_last_beta;

  Tensor<TensorType> FD_alpha;
  Tensor<TensorType> FDS_alpha;
  Tensor<TensorType> FD_beta;
  Tensor<TensorType> FDS_beta;

  // DF
  Tensor<TensorType> xyK; // n,n,ndf
  Tensor<TensorType> xyZ; // n,n,ndf
  Tensor<TensorType> Vm1; // ndf,ndf
};
