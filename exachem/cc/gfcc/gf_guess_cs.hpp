/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <algorithm>
#include <complex>
using namespace tamm;

template<typename T>
void gf_guess(ExecutionContext& ec, const TiledIndexSpace& MO, const TAMM_SIZE noab, double omega,
              double gf_eta, int pi, std::vector<T>& p_evl_sorted_occ, Tensor<T>& t2v2,
              Tensor<std::complex<T>>& x1, Tensor<std::complex<T>>& Minv) {
  using ComplexTensor = Tensor<std::complex<T>>;

  const TiledIndexSpace& O = MO("occ");

  ComplexTensor guessM{O, O};
  ComplexTensor::allocate(&ec, guessM);

  Scheduler sch{ec};
  sch(guessM() = t2v2()).execute();

  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  CMatrix guessM_eig(noab, noab);

  tamm_to_eigen_tensor(guessM, guessM_eig);
  ComplexTensor::deallocate(guessM);

  double denominator = 0.0;
  double shift       = 0.0;
  // TODO, update diagonal in TAMM, and inverse too
  for(TAMM_SIZE i = 0; i < noab; i++) {
    denominator = omega - p_evl_sorted_occ[i];
    if(denominator > 0 && denominator < 0.5) { shift = 0.5; }
    else if(denominator < 0 && denominator > -0.5) { shift = -0.5; }
    guessM_eig(i, i) += std::complex<T>(denominator + shift, gf_eta);
  }

  CMatrix guessMI = guessM_eig.inverse();

  Eigen::Tensor<std::complex<T>, 2, Eigen::RowMajor> x1e(noab, 1);
  for(TAMM_SIZE i = 0; i < noab; i++) x1e(i, 0) = guessMI(i, pi);

  eigen_to_tamm_tensor(x1, x1e);
  eigen_to_tamm_tensor(Minv, guessMI);
}
