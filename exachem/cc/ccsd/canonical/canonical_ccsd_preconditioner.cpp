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
 * Two implementations of M^{-1} are provided so they can be compared:
 *
 * (1) GMRES: general. Solve M*y = v iteratively with preconditioner_action.
 * (2) Diagonal: for canonical MOs M is diagonal, so M^{-1} is an element-wise division by the
 *     orbital energy denominators. No GMRES, no contractions.
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
 *
 * Diagonal case (canonical orbitals: f_oo = diag(e_i), f_vv = diag(e_a), f_ov = 0):
 *    a) M_1*y(a,i)     = (e_a - e_i) * y(a,i)
 *    b) M_2*y(a,b,i,j) = (e_a + e_b - e_i - e_j) * y(a,b,i,j)
 *  so M^{-1} is a pointwise division by the same denominators.
 */

// ---------------------------------------------------------------------------------------------
// (1) GMRES-based preconditioner
// ---------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------
// (2) Diagonal (orbital energy denominator) preconditioner
// ---------------------------------------------------------------------------------------------

namespace {

template<typename T>
std::vector<Tensor<T>> allocate_like(ExecutionContext&             ec,
                                     const std::vector<Tensor<T>>& reference) {
  std::vector<Tensor<T>> result;
  result.reserve(reference.size());
  for(const auto& tensor: reference) {
    Tensor<T> allocated{tensor.tiled_index_spaces()};
    Tensor<T>::allocate(&ec, allocated);
    result.push_back(allocated);
  }
  return result;
}

// Builds the inverse orbital-energy denominator for each tensor block.
//  rank 2: (V,O)     -> e_a - e_i
//  rank 4: (V,V,O,O) -> e_a + e_b - e_i - e_j
template<typename T>
void fill_inverse_denominator(ExecutionContext& ec, Tensor<T>& result,
                              const std::vector<T>& eps_occ, const std::vector<T>& eps_virt,
                              T shift) {
  constexpr T tiny = static_cast<T>(1e-10);
  auto        safe = [&](T d) {
    d += shift;
    return (std::abs(d) < tiny) ? (d < 0 ? -tiny : tiny) : d;
  };

  block_for(ec, result(), [&](IndexVector blockid) {
    const auto     size = result.block_size(blockid);
    std::vector<T> buf(size);

    const auto dims = result.block_dims(blockid);
    const auto off  = result.block_offsets(blockid);

    size_t c = 0;
    if(dims.size() == 2) {
      for(size_t a = 0; a < dims[0]; ++a)
        for(size_t i = 0; i < dims[1]; ++i, ++c)
          buf[c] = T(1) / safe(eps_virt[off[0] + a] - eps_occ[off[1] + i]);
    }
    else if(dims.size() == 4) {
      for(size_t a = 0; a < dims[0]; ++a)
        for(size_t b = 0; b < dims[1]; ++b)
          for(size_t i = 0; i < dims[2]; ++i)
            for(size_t j = 0; j < dims[3]; ++j, ++c)
              buf[c] = T(1) / safe(eps_virt[off[0] + a] + eps_virt[off[1] + b] -
                                   eps_occ[off[2] + i] - eps_occ[off[3] + j]);
    }
    result.put(blockid, buf);
  });
}

} // namespace

template<typename T>
std::vector<Tensor<T>> initialize_diagonal_preconditioner(
  ExecutionContext& ec, const std::vector<Tensor<T>>& vector_template,
  const std::vector<T>& eps_occ, const std::vector<T>& eps_virt, T shift) {
  auto inverse_denominators = allocate_like(ec, vector_template);
  for(auto& denominator: inverse_denominators)
    fill_inverse_denominator(ec, denominator, eps_occ, eps_virt, shift);
  return inverse_denominators;
}

template<typename T>
std::vector<Tensor<T>> diagonal_preconditioner_inverse_application(
  ExecutionContext& ec, const std::vector<Tensor<T>>& vector_to_be_preconditioned,
  const std::vector<Tensor<T>>& inverse_denominators) {
  auto result = allocate_like(ec, vector_to_be_preconditioned);
  auto sch    = Scheduler{ec};
  for(size_t i = 0; i < result.size(); ++i) {
    const auto tis = result[i].tiled_index_spaces();
    if(result[i].num_modes() == 2) {
      auto [a]    = tis[0].template labels<1>("all");
      auto [iocc] = tis[1].template labels<1>("all");
      sch(result[i](a, iocc) =
            vector_to_be_preconditioned[i](a, iocc) * inverse_denominators[i](a, iocc));
    }
    else {
      auto [a]    = tis[0].template labels<1>("all");
      auto [b]    = tis[1].template labels<1>("all");
      auto [iocc] = tis[2].template labels<1>("all");
      auto [jocc] = tis[3].template labels<1>("all");
      sch(result[i](a, b, iocc, jocc) = vector_to_be_preconditioned[i](a, b, iocc, jocc) *
                                        inverse_denominators[i](a, b, iocc, jocc));
    }
  }
  sch.execute();
  return result;
}

// ---------------------------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------------------------

template<typename T>
std::vector<Tensor<T>> apply_preconditioner_inverse(
  PreconditionerType type, ExecutionContext& ec, ChemEnv& chem_env, Tensor<T>& d_e, Scheduler& sch,
  const TiledIndexSpace& MO, const TensorMap<T>& f, const TensorMap<T>& eri,
  const std::vector<Tensor<T>>& vector_to_be_preconditioned, int krylov_dims_precond,
  double gmres_tol, const std::vector<Tensor<T>>& inverse_denominators) {
  if(type == PreconditionerType::Diagonal) {
    return diagonal_preconditioner_inverse_application<T>(ec, vector_to_be_preconditioned,
                                                          inverse_denominators);
  }
  return preconditioner_inverse_application<T>(ec, chem_env, d_e, sch, MO, f, eri,
                                               vector_to_be_preconditioned, krylov_dims_precond,
                                               gmres_tol);
}

// ---------------------------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------------------------

template std::vector<Tensor<double>>
preconditioner_action<double>(Scheduler&, const TiledIndexSpace&, const TensorMap<double>&,
                              const TensorMap<double>&, const std::vector<Tensor<double>>&);

template std::vector<Tensor<double>>
preconditioner_inverse_application<double>(ExecutionContext&, ChemEnv&, Tensor<double>&, Scheduler&,
                                           const TiledIndexSpace&, const TensorMap<double>&,
                                           const TensorMap<double>&,
                                           const std::vector<Tensor<double>>&, int, double);

template std::vector<Tensor<double>>
initialize_diagonal_preconditioner<double>(ExecutionContext&, const std::vector<Tensor<double>>&,
                                           const std::vector<double>&, const std::vector<double>&,
                                           double);

template std::vector<Tensor<double>> diagonal_preconditioner_inverse_application<double>(
  ExecutionContext&, const std::vector<Tensor<double>>&, const std::vector<Tensor<double>>&);

template std::vector<Tensor<double>> apply_preconditioner_inverse<double>(
  PreconditionerType, ExecutionContext&, ChemEnv&, Tensor<double>&, Scheduler&,
  const TiledIndexSpace&, const TensorMap<double>&, const TensorMap<double>&,
  const std::vector<Tensor<double>>&, int, double, const std::vector<Tensor<double>>&);

} // namespace exachem::cc::ccsd_os
