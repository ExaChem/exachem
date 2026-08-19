/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_preconditioner.hpp"

namespace exachem::cc::ccsd {

template<typename T>
std::vector<Tensor<T>> preconditioner_action_os(Scheduler& sch, const TiledIndexSpace& MO,
                                                CCSE_Tensors<T>& f1_oo, CCSE_Tensors<T>& f1_vv,
                                                const std::vector<Tensor<T>>& preconditoned_vector,
                                                ExecutionHW                   exhw) {
  const auto& y_1 = preconditoned_vector.at(0); // singles (V,O)
  const auto& y_2 = preconditoned_vector.at(1); // doubles (V,V,O,O)

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

  auto [h1_oa, h2_oa] = Oa.labels<2>("all");
  auto [p1_va, p2_va] = Va.labels<2>("all");
  auto [h1_ob, h2_ob] = Ob.labels<2>("all");
  auto [p1_vb, p2_vb] = Vb.labels<2>("all");
  auto [a, b, c, d]   = V.labels<4>("all");
  auto [i, j, k, l]   = O.labels<4>("all");

  Tensor<T> M_1_y_1{{V, O}, {1, 1}};
  Tensor<T> M_2_y_2{{V, V, O, O}, {2, 2}};
  Tensor<T> f_full{{N, N}, {1, 1}};
  sch.allocate(M_1_y_1, M_2_y_2, f_full).execute();

  // clang-format off
  sch
    (f_full(h1_oa, h2_oa) = f1_oo("aa")(h1_oa, h2_oa))
    (f_full(p1_va, p2_va) = f1_vv("aa")(p1_va, p2_va))
    (f_full(h1_ob, h2_ob) = f1_oo("bb")(h1_ob, h2_ob))
    (f_full(p1_vb, p2_vb) = f1_vv("bb")(p1_vb, p2_vb))
    .execute();

  sch
    (M_1_y_1(a, i)        = 0.0)
    (M_1_y_1(a, i)       += f_full(a, c) * y_1(c, i))
    (M_1_y_1(a, i)       -= f_full(k, i) * y_1(a, k))
    (M_2_y_2(a, b, i, j)  = 0.0)
    (M_2_y_2(a, b, i, j) += f_full(a, c) * y_2(c, b, i, j))
    (M_2_y_2(a, b, i, j) += f_full(b, d) * y_2(a, d, i, j))
    (M_2_y_2(a, b, i, j) -= f_full(i, k) * y_2(a, b, k, j))
    (M_2_y_2(a, b, i, j) -= f_full(j, l) * y_2(a, b, i, l))
    .deallocate(f_full)
    .execute(exhw);
  // clang-format on

  return {M_1_y_1, M_2_y_2};
}

template<typename T>
std::vector<Tensor<T>> preconditioner_action_cs(Scheduler& sch, const TiledIndexSpace& MO,
                                                CCSE_Tensors<T>& f1_oo, CCSE_Tensors<T>& f1_vv,
                                                const std::vector<Tensor<T>>& preconditoned_vector,
                                                ExecutionHW                   exhw) {
  const auto& y_1 = preconditoned_vector.at(0); // aa singles  (Va,Oa)
  const auto& y_2 = preconditoned_vector.at(1); // abab doubles (Va,Vb,Oa,Ob)

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

  auto [aa, ba, ca, da] = Va.labels<4>("all");
  auto [ia, ja, ka, la] = Oa.labels<4>("all");
  auto [ab, bb, cb, db] = Vb.labels<4>("all");
  auto [ib, jb, kb, lb] = Ob.labels<4>("all");

  Tensor<T> M_1_y_1{{Va, Oa}, {1, 1}};
  Tensor<T> M_2_y_2{{Va, Vb, Oa, Ob}, {2, 2}};
  Tensor<T> f_full{{N, N}, {1, 1}};
  sch.allocate(M_1_y_1, M_2_y_2, f_full).execute();

  // clang-format off
  sch
    (f_full(ia, ja) = f1_oo("aa")(ia, ja))
    (f_full(aa, ba) = f1_vv("aa")(aa, ba))
    (f_full(ib, jb) = f1_oo("bb")(ib, jb))
    (f_full(ab, bb) = f1_vv("bb")(ab, bb))
    .execute();

  sch
    // Singles (aa) preconditioner action
    (M_1_y_1(aa, ia)  = 0.0)
    (M_1_y_1(aa, ia) += f_full(aa, ca) * y_1(ca, ia))
    (M_1_y_1(aa, ia) -= f_full(ka, ia) * y_1(aa, ka))
    // Doubles (abab) preconditioner action
    (M_2_y_2(aa, ab, ia, ib)  = 0.0)
    (M_2_y_2(aa, ab, ia, ib) += f_full(aa, ca) * y_2(ca, ab, ia, ib))
    (M_2_y_2(aa, ab, ia, ib) += f_full(ab, cb) * y_2(aa, cb, ia, ib))
    (M_2_y_2(aa, ab, ia, ib) -= f_full(ia, ka) * y_2(aa, ab, ka, ib))
    (M_2_y_2(aa, ab, ia, ib) -= f_full(ib, kb) * y_2(aa, ab, ia, kb))
    .deallocate(f_full)
    .execute(exhw);
  // clang-format on

  return {M_1_y_1, M_2_y_2};
}

template std::vector<Tensor<double>>
preconditioner_action_os<double>(Scheduler&, const TiledIndexSpace&, CCSE_Tensors<double>&,
                                 CCSE_Tensors<double>&, const std::vector<Tensor<double>>&,
                                 ExecutionHW);

template std::vector<Tensor<double>>
preconditioner_action_cs<double>(Scheduler&, const TiledIndexSpace&, CCSE_Tensors<double>&,
                                 CCSE_Tensors<double>&, const std::vector<Tensor<double>>&,
                                 ExecutionHW);

} // namespace exachem::cc::ccsd
