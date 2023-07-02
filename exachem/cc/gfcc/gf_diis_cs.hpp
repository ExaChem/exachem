/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "ga/ga.h"
#include "tamm/eigen_utils.hpp"
#include "tamm/tamm.hpp"

namespace tamm {

/**
 * @brief DIIS routine
 * @tparam T Type of element in each tensor
 * @param ec Execution context in which this function invoked
 * @param[in] d_rs Vector of R tensors
 * @param[in] d_ts Vector of T tensors
 * @param[out] d_t Vector of T tensors produced by DIIS
 * @pre d_rs.size() == d_ts.size()
 * @pre 0<=i<d_rs.size(): d_rs[i].size() == d_t.size()
 * @pre 0<=i<d_ts.size(): d_ts[i].size() == d_t.size()
 */
template<typename T>
inline void gf_diis_a(ExecutionContext& ec, const TiledIndexSpace& MO, int ndiis,
                      Tensor<std::complex<T>>& e1_a, Tensor<std::complex<T>>& e2_aaa,
                      Tensor<std::complex<T>>& e2_bab, Tensor<std::complex<T>>& xx1_a,
                      Tensor<std::complex<T>>& xx2_aaa, Tensor<std::complex<T>>& xx2_bab,
                      Tensor<std::complex<T>>& x1_a, Tensor<std::complex<T>>& x2_aaa,
                      Tensor<std::complex<T>>& x2_bab, const TiledIndexSpace& diis_tis,
                      const TiledIndexSpace& unit_tis) {
  using ComplexTensor = Tensor<std::complex<T>>;

  TiledIndexSpace        o_alpha, v_alpha, o_beta, v_beta;
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles   = O.num_tiles();
  const int vtiles   = V.num_tiles();
  const int oabtiles = otiles / 2;
  const int vabtiles = vtiles / 2;

  o_alpha = {MO("occ"), range(oabtiles)};
  v_alpha = {MO("virt"), range(vabtiles)};
  o_beta  = {MO("occ"), range(oabtiles, otiles)};
  v_beta  = {MO("virt"), range(vabtiles, vtiles)};

  auto [p1_va]        = v_alpha.labels<1>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  auto [dh1, dh2]     = diis_tis.labels<2>("all");
  auto [u1]           = unit_tis.labels<1>("all");

  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  CMatrix A = CMatrix::Zero(ndiis + 1, ndiis + 1);
  CMatrix b = CMatrix::Zero(ndiis + 1, 1);

  Scheduler     sch{ec};
  ComplexTensor tmp{diis_tis, diis_tis};

  auto e1_conj_a   = tamm::conj(e1_a);
  auto e2_conj_aaa = tamm::conj(e2_aaa);
  auto e2_conj_bab = tamm::conj(e2_bab);

  // clang-format off
  sch.allocate(tmp)
    (tmp(dh1,dh2)  = e1_conj_a(h1_oa,dh1) * e1_a(h1_oa,dh2))
    (tmp(dh1,dh2) += e2_conj_aaa(p1_va,h1_oa,h2_oa,dh1) * e2_aaa(p1_va,h1_oa,h2_oa,dh2))
    (tmp(dh1,dh2) += 2.0 * e2_conj_bab(p1_vb,h1_oa,h2_ob,dh1) * e2_bab(p1_vb,h1_oa,h2_ob,dh2))
    .deallocate(e1_conj_a,e2_conj_aaa,e2_conj_bab) 
    .execute();
  // clang-format on

  tamm_to_eigen_tensor(tmp, A);
  sch.deallocate(tmp).execute();

  for(auto i = 0; i < ndiis; i++) {
    A(i, ndiis) = std::complex<T>(-1.0, 0.0);
    A(ndiis, i) = std::complex<T>(-1.0, 0.0);
  }

  b(ndiis, 0) = std::complex<T>(-1.0, 0.0);

  // TODO: Solve AX = B
  CMatrix       coeff = A.lu().solve(b);
  ComplexTensor coeff_tamm{diis_tis, unit_tis};
  ComplexTensor::allocate(&ec, coeff_tamm);
  eigen_to_tamm_tensor(coeff_tamm, coeff);

  // clang-format off
  sch 
    (x1_a(h1_oa,u1) = coeff_tamm(dh1,u1) * xx1_a(h1_oa,dh1))
    (x2_aaa(p1_va,h1_oa,h2_oa,u1) = coeff_tamm(dh1,u1) * xx2_aaa(p1_va,h1_oa,h2_oa,dh1))
    (x2_bab(p1_vb,h1_oa,h2_ob,u1) = coeff_tamm(dh1,u1) * xx2_bab(p1_vb,h1_oa,h2_ob,dh1))
    .deallocate(coeff_tamm).execute();
  // clang-format on
}

template<typename T>
inline void gf_diis_b(ExecutionContext& ec, const TiledIndexSpace& MO, int ndiis,
                      Tensor<std::complex<T>>& e1_b, Tensor<std::complex<T>>& e2_bbb,
                      Tensor<std::complex<T>>& e2_aba, Tensor<std::complex<T>>& xx1_b,
                      Tensor<std::complex<T>>& xx2_bbb, Tensor<std::complex<T>>& xx2_aba,
                      Tensor<std::complex<T>>& x1_b, Tensor<std::complex<T>>& x2_bbb,
                      Tensor<std::complex<T>>& x2_aba, const TiledIndexSpace& diis_tis,
                      const TiledIndexSpace& unit_tis) {
  using ComplexTensor = Tensor<std::complex<T>>;

  TiledIndexSpace        o_alpha, v_alpha, o_beta, v_beta;
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles   = O.num_tiles();
  const int vtiles   = V.num_tiles();
  const int oabtiles = otiles / 2;
  const int vabtiles = vtiles / 2;

  o_alpha = {MO("occ"), range(oabtiles)};
  v_alpha = {MO("virt"), range(vabtiles)};
  o_beta  = {MO("occ"), range(oabtiles, otiles)};
  v_beta  = {MO("virt"), range(vabtiles, vtiles)};

  auto [p1_va]        = v_alpha.labels<1>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  auto [dh1, dh2]     = diis_tis.labels<2>("all");
  auto [u1]           = unit_tis.labels<1>("all");

  using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  CMatrix A = CMatrix::Zero(ndiis + 1, ndiis + 1);
  CMatrix b = CMatrix::Zero(ndiis + 1, 1);

  Scheduler     sch{ec};
  ComplexTensor tmp{diis_tis, diis_tis};

  auto e1_conj_b   = tamm::conj(e1_b);
  auto e2_conj_aba = tamm::conj(e2_aba);
  auto e2_conj_bbb = tamm::conj(e2_bbb);

  // clang-format off
  sch.allocate(tmp)
    (tmp(dh1,dh2)  = e1_conj_b(h1_ob,dh1) * e1_b(h1_ob,dh2))
    (tmp(dh1,dh2) += 2.0 * e2_conj_aba(p1_va,h1_ob,h2_oa,dh1) * e2_aba(p1_va,h1_ob,h2_oa,dh2))
    (tmp(dh1,dh2) += e2_conj_bbb(p1_vb,h1_ob,h2_ob,dh1) * e2_bbb(p1_vb,h1_ob,h2_ob,dh2))
    .deallocate(e1_conj_b,e2_conj_bbb,e2_conj_aba)
    .execute();
  // clang-format on

  tamm_to_eigen_tensor(tmp, A);
  sch.deallocate(tmp).execute();

  for(auto i = 0; i < ndiis; i++) {
    A(i, ndiis) = std::complex<T>(-1.0, 0.0);
    A(ndiis, i) = std::complex<T>(-1.0, 0.0);
  }

  b(ndiis, 0) = std::complex<T>(-1.0, 0.0);

  // TODO: Solve AX = B
  CMatrix       coeff = A.lu().solve(b);
  ComplexTensor coeff_tamm{diis_tis, unit_tis};
  ComplexTensor::allocate(&ec, coeff_tamm);
  eigen_to_tamm_tensor(coeff_tamm, coeff);

  // clang-format off
  sch 
    (x1_b(h1_ob,u1) = coeff_tamm(dh1,u1) * xx1_b(h1_ob,dh1))
    (x2_aba(p1_va,h1_ob,h2_oa,u1) = coeff_tamm(dh1,u1) * xx2_aba(p1_va,h1_ob,h2_oa,dh1))
    (x2_bbb(p1_vb,h1_ob,h2_ob,u1) = coeff_tamm(dh1,u1) * xx2_bbb(p1_vb,h1_ob,h2_ob,dh1))
    .deallocate(coeff_tamm).execute();
  // clang-format on
}

} // namespace tamm
