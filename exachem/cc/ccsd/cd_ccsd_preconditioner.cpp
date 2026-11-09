/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_preconditioner.hpp"
#include "exachem/cc/krylov_solvers.hpp"

namespace exachem::cc::ccsd {

template<typename T>
std::vector<Tensor<T>> sylvester_preconditioner(ExecutionContext& ec, ChemEnv& chem_env,
                                                Tensor<T>& d_e, const Tensor<T>& fock,
                                                const std::vector<Tensor<T>>& rhs) {
  using namespace exachem::cc::solvers;
  auto                  sch    = Scheduler{ec};
  const auto&           spaces = rhs[1].tiled_index_spaces();
  const TiledIndexSpace Va = spaces[0], Vb = spaces[1], Oa = spaces[2], Ob = spaces[3];
  auto [a, c]         = Va.labels<2>("all");
  auto [b, d]         = Vb.labels<2>("all");
  auto [i, k]         = Oa.labels<2>("all");
  auto [j, l]         = Ob.labels<2>("all");
  const double shift  = chem_env.ioptions.ccsd_options.lshift;
  auto         action = [&](const std::vector<Tensor<T>>& y) {
    auto result = zeroslike(ec, y);
    // clang-format off
    sch
      (result[0](a, i) = fock(a, c) * y[0](c, i))
      (result[0](a, i) -= fock(k, i) * y[0](a, k))
      (result[0](a, i) += shift * y[0](a, i))
      (result[1](a, b, i, j) = fock(a, c) * y[1](c, b, i, j))
      (result[1](a, b, i, j) += fock(b, d) * y[1](a, d, i, j))
      (result[1](a, b, i, j) -= fock(k, i) * y[1](a, b, k, j))
      (result[1](a, b, i, j) -= fock(l, j) * y[1](a, b, i, l))
      (result[1](a, b, i, j) += (2.0 * shift) * y[1](a, b, i, j))
      .execute(ec.exhw());
    // clang-format on
    return result;
  };
  // GMRES owns its RHS; keep the caller's residual scratch intact.
  auto work   = tensor_copy(ec, rhs);
  auto result = gmres_solver(ec, chem_env, d_e, action, std::move(work), 10,
                             chem_env.ioptions.ccsd_options.solvers.newton_krylov.gmres_tol);
  for(auto& t: work) sch.deallocate(t);
  sch.execute();
  return result;
}

template std::vector<Tensor<double>>
sylvester_preconditioner<double>(ExecutionContext&, ChemEnv&, Tensor<double>&,
                                 const Tensor<double>&, const std::vector<Tensor<double>>&);

} // namespace exachem::cc::ccsd
