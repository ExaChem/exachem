/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <algorithm>
#include <complex>
using namespace tamm;

template<typename T>
void gf_guess_ip(ExecutionContext& ec, const TiledIndexSpace& MO, const TAMM_SIZE nocc,
                 double omega, double gf_eta, int pi, std::vector<T>& p_evl_sorted_occ,
                 Tensor<T>& t2v2_o, Tensor<std::complex<T>>& x1, Tensor<std::complex<T>>& Minv,
                 bool opt = false) {
  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& O = MO("occ");

  ComplexTensor guessM{O, O};
  ComplexTensor::allocate(&ec, guessM);

  Scheduler sch{ec};
  sch(guessM() = t2v2_o()).execute();

  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  CMatrix guessM_eig(nocc, nocc);

  tamm_to_eigen_tensor(guessM, guessM_eig);
  ComplexTensor::deallocate(guessM);

  double denominator = 0.0;
  // TODO, update diagonal in TAMM, and inverse too
  for(TAMM_SIZE i = 0; i < nocc; i++) {
    denominator = omega - p_evl_sorted_occ[i];
    if(denominator > 0 && denominator < 0.5) { denominator += 0.5; }
    else if(denominator < 0 && denominator > -0.5) { denominator += -0.5; }
    guessM_eig(i, i) += std::complex<T>(denominator, -1.0 * gf_eta);
  }

  CMatrix guessMI = guessM_eig.inverse();

  if(opt) {
    Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> x1e(nocc);
    for(TAMM_SIZE i = 0; i < nocc; i++) x1e(i) = guessMI(i, pi);

    eigen_to_tamm_tensor(x1, x1e);
  }
  else {
    Eigen::Tensor<std::complex<T>, 2, Eigen::RowMajor> x1e(nocc, 1);
    for(TAMM_SIZE i = 0; i < nocc; i++) x1e(i, 0) = guessMI(i, pi);

    eigen_to_tamm_tensor(x1, x1e);
  }
  eigen_to_tamm_tensor(Minv, guessMI);
}

template<typename T>
void gf_guess_ea(ExecutionContext& ec, const TiledIndexSpace& MO, const TAMM_SIZE nvir,
                 double omega, double gf_eta, int pi, std::vector<T>& p_evl_sorted_vir,
                 Tensor<T>& t2v2_v, Tensor<std::complex<T>>& y1, Tensor<std::complex<T>>& Minv,
                 bool opt = false) {
  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& V = MO("virt");

  ComplexTensor guessM{V, V};
  ComplexTensor::allocate(&ec, guessM);

  Scheduler sch{ec};
  // sch(guessM() = t2v2_v()).execute();
  sch(guessM() = 0).execute();

  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  CMatrix guessM_eig(nvir, nvir);

  tamm_to_eigen_tensor(guessM, guessM_eig);
  ComplexTensor::deallocate(guessM);

  double denominator = 0.0;
  double shift       = 0.0;
  // TODO, update diagonal in TAMM, and inverse too
  for(TAMM_SIZE i = 0; i < nvir; i++) {
    denominator = omega - p_evl_sorted_vir[i];
    // if (denominator > 0 && denominator < 0.5) {
    //     shift = 0.5;
    // }
    // else if (denominator < 0 && denominator > -0.5) {
    //     shift = -0.5;
    // }
    guessM_eig(i, i) += std::complex<T>(denominator + shift, gf_eta);
  }

  CMatrix guessMI = guessM_eig.inverse();

  if(opt) {
    Eigen::Tensor<std::complex<T>, 1, Eigen::RowMajor> y1e(nvir);
    for(TAMM_SIZE i = 0; i < nvir; i++) y1e(i) = guessMI(i, pi);

    eigen_to_tamm_tensor(y1, y1e);
  }
  else {
    Eigen::Tensor<std::complex<T>, 2, Eigen::RowMajor> y1e(nvir, 1);
    for(TAMM_SIZE i = 0; i < nvir; i++) y1e(i, 0) = guessMI(i, pi);

    eigen_to_tamm_tensor(y1, y1e);
  }
  eigen_to_tamm_tensor(Minv, guessMI);
}
