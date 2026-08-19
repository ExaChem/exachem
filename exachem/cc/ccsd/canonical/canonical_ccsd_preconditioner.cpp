/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/canonical/canonical_ccsd_preconditioner.hpp"
#include "exachem/cc/krylov_solvers.hpp"

namespace exachem::cc::ccsd_os {

using namespace tamm;
using namespace exachem::cc::solvers;

/**
 * @brief Preconditioning a linear system. (Note: This is system dependent)
 *
 * This involves 2 steps:
 *  1. Function to compute the preconditioner action on a vector.
 *  2. Use the preconditioner action function in GMRES to solve the linear system and return
 * precondtioned solution.
 *
 * Note: When we solve A*x = b via GMRES, the convergence rate of depends on the condition number of
 * A.
 *  - If A is ill-conditioned, GMRES may converge slowly or not at all.
 *  - Preconditioning is a technique to improve the convergence of iterative solvers by transforming
 * the original system into an equivalent one with better spectral properties.
 *
 * A good preconditioner:
 * - Diagonal approximation: M = diag(A), where M is the preconditioner.
 *
 * I derived the preconditioner action for the CCSD residuals in the following way:
 *  - System: J*delta_x = - residual function
 *  - Precondioned system M^{-1}*J*delta_x = - M^{-1}*residual function
 *
 *  - Here, the preconditoned vectors we want are
 *    a) y_1 = M^{-1}*J*delta_x
 *    b) y_2 = M^{-1}*residual function
 *
 *  - The action of the preconditoner is
 *    a) For singles: M_1*y = f_vv(a,c) * y(c,i) - f_oo(k,i) * y(a,k)
 *    b) For doubles: M_2*y = f_vv(a,c) * y(c,b,i,j) + f_vv(b,d) * y(a,d,i,j) - f_oo(i,k) *
 * y(a,b,k,j) - f_oo(j,l) * y(a,b,i,l)
 *
 *  - Let M = {M_1, M_2}, then we use GMRES to solve
 *    a) M*y_1 = J*delta_x (GMRES: MVPFun = M*y_1, rhs = J*delta_x)
 *    b) M*y_2 = residual function (GMRES: MVPFun = M*y_2, rhs = residual function)
 *
 * Note: The idea of using GMRES is to avoid explicit inversion of the preconditioner.
 */

template<typename T>
std::vector<Tensor<T>> preconditioner_action(Scheduler& sch, const TiledIndexSpace& MO,
                                             const TensorMap<T>& f, const TensorMap<T>& eri,
                                             const std::vector<Tensor<T>>& preconditoned_vector) {
  const auto& y_1 = preconditoned_vector.at(0);
  const auto& y_2 = preconditoned_vector.at(1);

  const TiledIndexSpace O = MO("occ");
  const TiledIndexSpace V = MO("virt");
  const TiledIndexSpace N = MO("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  TiledIndexLabel aa, ba, ca, da;
  TiledIndexLabel ia, ja, ka, la;
  TiledIndexLabel ab, bb, cb, db;
  TiledIndexLabel ib, jb, kb, lb;
  TiledIndexLabel a, b, c, d;
  TiledIndexLabel i, j, k, l;

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");
  std::tie(a, b, c, d)     = V.labels<4>("all");
  std::tie(i, j, k, l)     = O.labels<4>("all");

  Tensor<T> M_1_y_1{{V, O}, {1, 1}};
  Tensor<T> M_2_y_2{{V, V, O, O}, {2, 2}};
  Tensor<T> f_full{{N, N}, {1, 1}};
  sch.allocate(M_1_y_1, M_2_y_2, f_full).execute();

  // clang-format off
  sch
    (f_full(ia, ja) = f.at("aa_oo")(ia, ja))
    (f_full(aa, ba) = f.at("aa_vv")(aa, ba))
    (f_full(aa, ia) = f.at("aa_ov")(ia, aa))
    (f_full(ia, aa) = f.at("aa_ov")(ia, aa))
    (f_full(ib, jb) = f.at("bb_oo")(ib, jb))
    (f_full(ab, bb) = f.at("bb_vv")(ab, bb))
    (f_full(ab, ib) = f.at("bb_ov")(ib, ab))
    (f_full(ib, bb) = f.at("bb_ov")(ib, ab))
    .execute();

  sch
  // Singles preconditioner action
  (M_1_y_1(a, i) = 0.0 )
  (M_1_y_1(a, i) += f_full(a, c) * y_1(c, i))
  (M_1_y_1(a, i) -= f_full(k, i) * y_1(a, k))

  // // Extra terms from Nick's derivtion for singles
  // (M_1_y_1(aa, ia) -= 0.5 * eri.at("aaaa_vooo")(ca, ia, ka, la) * y_2(aa, ca, la, ka))
  // (M_1_y_1(aa, ia) +=       eri.at("baab_vooo")(cb, ia, la, kb) * y_2(aa, cb, la, kb))
  // (M_1_y_1(ab, ib) -= 0.5 * eri.at("bbbb_vooo")(cb, ib, kb, lb) * y_2(ab, cb, lb, kb))
  // (M_1_y_1(ab, ib) -=       eri.at("abab_vooo")(ca, ib, ka, lb) * y_2(ca, ab, ka, lb))

  // Doubles preconditioner action
  (M_2_y_2(a, b, i, j) = 0.0)
  (M_2_y_2(a, b, i, j) += f_full(a, c) * y_2(c, b, i, j))
  (M_2_y_2(a, b, i, j) += f_full(b, d) * y_2(a, d, i, j))
  (M_2_y_2(a, b, i, j) -= f_full(i, k) * y_2(a, b, k, j))
  (M_2_y_2(a, b, i, j) -= f_full(j, l) * y_2(a, b, i, l))

  // // Extra terms from Nick's derivtion for doubles
  // (M_2_y_2(aa, ba, ia, ja) += 0.5 * eri.at("aaaa_oooo")(la, ka, ia, ja) * y_2(aa, ba, la, ka))
  // (M_2_y_2(ab, bb, ib, jb) += 0.5 * eri.at("bbbb_oooo")(lb, kb, ib, jb) * y_2(ab, bb, lb, kb))
  // (M_2_y_2(aa, bb, ia, jb) += 0.5 * eri.at("abab_oooo")(la, kb, ia, jb) * y_2(aa, bb, la, kb))
  
  
  
  .deallocate(f_full)
  .execute(sch.ec().exhw());
  // clang-format on

  std::vector<Tensor<T>> My;
  My.push_back(M_1_y_1);
  My.push_back(M_2_y_2);

  return My;
}

template<typename T>
std::vector<Tensor<T>>
preconditioner_inverse_application(ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e,
                                   Scheduler& sch, const TiledIndexSpace& MO, const TensorMap<T>& f,
                                   const TensorMap<T>&           eri,
                                   const std::vector<Tensor<T>>& vector_to_be_preconditioned,
                                   int krylov_dims_precond, double gmres_tol) {
  auto precond_action =
    [&](const std::vector<Tensor<T>>& preconditoned_vector) -> std::vector<Tensor<T>> {
    return preconditioner_action<T>(sch, MO, f, eri, preconditoned_vector);
  };

  return gmres_solver(ec, chem_env, d_e, precond_action, vector_to_be_preconditioned,
                      krylov_dims_precond, gmres_tol);
}

template std::vector<Tensor<double>>
preconditioner_action<double>(Scheduler&, const TiledIndexSpace&, const TensorMap<double>&,
                              const TensorMap<double>&, const std::vector<Tensor<double>>&);

template std::vector<Tensor<double>>
preconditioner_inverse_application<double>(ExecutionContext&, ChemEnv&, Tensor<double>&, Scheduler&,
                                           const TiledIndexSpace&, const TensorMap<double>&,
                                           const TensorMap<double>&,
                                           const std::vector<Tensor<double>>&, int, double);

} // namespace exachem::cc::ccsd_os
