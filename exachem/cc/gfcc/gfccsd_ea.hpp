/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "exachem/cc/gfcc/gfccsd_internal.hpp"

namespace exachem::cc::gfcc {
#if 0
  template<typename T>
  void gfccsd_driver_ea_a(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO,
    Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
    Tensor<T>& f1, Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
    Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
    Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
    Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
    Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
    Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
    Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
    Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
    Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
    Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
    Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
    Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
    Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
    std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt,
    const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI,
    const TiledIndexSpace& unit_tis, string files_prefix, string levelstr, double gf_omega);

  template<typename T>
  void gfccsd_driver_ea_b(
      ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO,
      Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
      Tensor<T>& f1, Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
      Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
      Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
      Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
      Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
      Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
      Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
      Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
      Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
      Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
      Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
      Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
      Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
      std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt,
      const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI,
      const TiledIndexSpace& unit_tis, string files_prefix, string levelstr, double gf_omega);
#endif
} // namespace exachem::cc::gfcc

template<typename T>
void gfccsd_y1_a(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_a,
                 const Tensor<T>& t1_a, const Tensor<T>& t1_b, const Tensor<T>& t2_aaaa,
                 const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& y1_a, const Tensor<std::complex<T>>& y2_aaa,
                 const Tensor<std::complex<T>>& y2_bab, const Tensor<T>& f1, const Tensor<T>& iy1_a,
                 const Tensor<T>& iy1_2_1_a, const Tensor<T>& iy1_2_1_b,
                 const Tensor<T>& v2ijab_aaaa, const Tensor<T>& v2ijab_abab,
                 const Tensor<T>& cholOV_a, const Tensor<T>& cholOV_b, const Tensor<T>& cholVV_a,
                 const TiledIndexSpace& CI, const TiledIndexSpace& gf_tis, bool has_tis,
                 bool debug = false) {
  using ComplexTensor = Tensor<std::complex<T>>;
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h5_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] =
    o_alpha.labels<10>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h5_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] =
    o_beta.labels<10>("all");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va, p7_va, p8_va, p9_va, p10_va] =
    v_alpha.labels<10>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p5_vb, p6_vb, p7_vb, p8_vb, p9_vb, p10_vb] =
    v_beta.labels<10>("all");
  auto [u1] = gf_tis.labels<1>("all");

  /* spin implicit contraction
  sch
    ( i0(p1)         =  0                                             )
    ( i0(p2)        +=  1.0 * y1(p6) * iy1(p2,p6)                  )
    ( i0(p2)        +=  1.0 * y2(h6,p2,p7) * iy1_2_1(h6,p7)        )
    (   i1_1(p5,cind)  =  1.0 * y2(p4,p5,h3) * cholVpr(h3,p4,cind) )
    ( i0(p2)        += -1.0 * i1_1(p5,cind) * cholVpr(p2,p5)    )
    (   i1_2(h3)       =  0.5 * y2(h4,p5,p6) * v2ijab(h3,h4,p5,p6) )
    ( i0(p2)        += -1.0 * d_t1(p2,h3) * i1_2(h3)               );
  */

  /* intermediates needed
  iy1_a,iy1_b,
  iy1_2_1_a,iy1_2_1_b,
  cholOV_a,cholOV_b,cholVV_a,cholVV_b,
  v2ijab_aaaa,v2ijab_abab,v2ijab_bbbb
  */

  ComplexTensor i1_1_a;
  ComplexTensor i1_2_a;

  if(has_tis) {
    i1_1_a = {v_alpha, CI, gf_tis};
    i1_2_a = {o_alpha, gf_tis};
  }
  else {
    i1_1_a = {v_alpha, CI};
    i1_2_a = {o_alpha};
  }

  sch.allocate(i1_1_a, i1_2_a);
  // clang-format off
  if(has_tis) {
    sch( i0_a(p1_va,u1)        =  0                                                                         )
    ( i0_a(p2_va,u1)          +=  1.0 * y1_a(p6_va,u1) * iy1_a(p2_va,p6_va)                                 )
    ( i0_a(p2_va,u1)          +=  1.0 * y2_aaa(h6_oa,p2_va,p7_va,u1) * iy1_2_1_a(h6_oa,p7_va)               )
    ( i0_a(p2_va,u1)          +=  1.0 * y2_bab(h6_ob,p2_va,p7_vb,u1) * iy1_2_1_b(h6_ob,p7_vb)               )
    (   i1_1_a(p5_va,cind,u1)  =  1.0 * y2_aaa(h3_oa,p4_va,p5_va,u1) * cholOV_a(h3_oa,p4_va,cind)           )
    (   i1_1_a(p5_va,cind,u1) += -1.0 * y2_bab(h3_ob,p5_va,p4_vb,u1) * cholOV_b(h3_ob,p4_vb,cind)           )
    ( i0_a(p2_va,u1)          += -1.0 * i1_1_a(p5_va,cind,u1) * cholVV_a(p2_va,p5_va,cind)                  )
    (   i1_2_a(h3_oa,u1)       =  0.5 * y2_aaa(h4_oa,p5_va,p6_va,u1) * v2ijab_aaaa(h3_oa,h4_oa,p5_va,p6_va) )
    (   i1_2_a(h3_oa,u1)      +=  1.0 * y2_bab(h4_ob,p5_va,p6_vb,u1) * v2ijab_abab(h3_oa,h4_ob,p5_va,p6_vb) )
    ( i0_a(p2_va,u1)          += -1.0 * t1_a(p2_va,h3_oa) * i1_2_a(h3_oa,u1)                                );
  }
  else {
    sch( i0_a(p1_va)        =  0                                                                      )
    ( i0_a(p2_va)          +=  1.0 * y1_a(p6_va) * iy1_a(p2_va,p6_va)                                 )
    ( i0_a(p2_va)          +=  1.0 * y2_aaa(h6_oa,p2_va,p7_va) * iy1_2_1_a(h6_oa,p7_va)               )
    ( i0_a(p2_va)          +=  1.0 * y2_bab(h6_ob,p2_va,p7_vb) * iy1_2_1_b(h6_ob,p7_vb)               )
    (   i1_1_a(p5_va,cind)  =  1.0 * y2_aaa(h3_oa,p4_va,p5_va) * cholOV_a(h3_oa,p4_va,cind)           )
    (   i1_1_a(p5_va,cind) += -1.0 * y2_bab(h3_ob,p5_va,p4_vb) * cholOV_b(h3_ob,p4_vb,cind)           )
    ( i0_a(p2_va)          += -1.0 * i1_1_a(p5_va,cind) * cholVV_a(p2_va,p5_va,cind)                  )
    (   i1_2_a(h3_oa)       =  0.5 * y2_aaa(h4_oa,p5_va,p6_va) * v2ijab_aaaa(h3_oa,h4_oa,p5_va,p6_va) )
    (   i1_2_a(h3_oa)      +=  1.0 * y2_bab(h4_ob,p5_va,p6_vb) * v2ijab_abab(h3_oa,h4_ob,p5_va,p6_vb) )
    ( i0_a(p2_va)          += -1.0 * t1_a(p2_va,h3_oa) * i1_2_a(h3_oa)                                );
  }
  // clang-format on
  sch.deallocate(i1_1_a, i1_2_a);

  if(debug) sch.execute();
}

template<typename T>
void gfccsd_y1_b(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_b,
                 const Tensor<T>& t1_a, const Tensor<T>& t1_b, const Tensor<T>& t2_aaaa,
                 const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& y1_b, const Tensor<std::complex<T>>& y2_bbb,
                 const Tensor<std::complex<T>>& y2_aba, const Tensor<T>& f1, const Tensor<T>& iy1_b,
                 const Tensor<T>& iy1_2_1_a, const Tensor<T>& iy1_2_1_b,
                 const Tensor<T>& v2ijab_bbbb, const Tensor<T>& v2ijab_abab,
                 const Tensor<T>& cholOV_a, const Tensor<T>& cholOV_b, const Tensor<T>& cholVV_b,
                 const TiledIndexSpace& CI, const TiledIndexSpace& gf_tis, bool has_tis,
                 bool debug = false) {
  using ComplexTensor = Tensor<std::complex<T>>;
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h5_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] =
    o_alpha.labels<10>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h5_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] =
    o_beta.labels<10>("all");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va, p7_va, p8_va, p9_va, p10_va] =
    v_alpha.labels<10>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p5_vb, p6_vb, p7_vb, p8_vb, p9_vb, p10_vb] =
    v_beta.labels<10>("all");
  auto [u1] = gf_tis.labels<1>("all");

  ComplexTensor i1_1_b;
  ComplexTensor i1_2_b;

  if(has_tis) {
    i1_1_b = {v_beta, CI, gf_tis};
    i1_2_b = {o_beta, gf_tis};
  }
  else {
    i1_1_b = {v_beta, CI};
    i1_2_b = {o_beta};
  }

  sch.allocate(i1_1_b, i1_2_b);
  // clang-format off
  if(has_tis) {
    sch( i0_b(p1_vb,u1)        =  0                                                                         )
    ( i0_b(p2_vb,u1)          +=  1.0 * y1_b(p6_vb,u1) * iy1_b(p2_vb,p6_vb)                                 )
    ( i0_b(p2_vb,u1)          +=  1.0 * y2_bbb(h6_ob,p2_vb,p7_vb,u1) * iy1_2_1_b(h6_ob,p7_vb)               )
    ( i0_b(p2_vb,u1)          +=  1.0 * y2_aba(h6_oa,p2_vb,p7_va,u1) * iy1_2_1_a(h6_oa,p7_va)               )
    (   i1_1_b(p5_vb,cind,u1)  =  1.0 * y2_bbb(h3_ob,p4_vb,p5_vb,u1) * cholOV_b(h3_ob,p4_vb,cind)           )
    (   i1_1_b(p5_vb,cind,u1) += -1.0 * y2_aba(h3_oa,p5_vb,p4_va,u1) * cholOV_a(h3_oa,p4_va,cind)           )
    ( i0_b(p2_vb,u1)          += -1.0 * i1_1_b(p5_vb,cind,u1) * cholVV_b(p2_vb,p5_vb,u1)                    )
    (   i1_2_b(h3_ob,u1)       =  0.5 * y2_bbb(h4_ob,p5_vb,p6_vb,u1) * v2ijab_bbbb(h3_ob,h4_ob,p5_vb,p6_vb) )
    (   i1_2_b(h3_ob,u1)      +=  1.0 * y2_aba(h4_oa,p5_vb,p6_va,u1) * v2ijab_abab(h4_oa,h3_ob,p6_va,p5_vb) )
    ( i0_b(p2_vb,u1)          += -1.0 * t1_a(p2_vb,h3_ob) * i1_2_b(h3_ob,u1)                                );
  }
  else {
    sch( i0_b(p1_vb)        =  0                                                                      )
    ( i0_b(p2_vb)          +=  1.0 * y1_b(p6_vb) * iy1_b(p2_vb,p6_vb)                                 )
    ( i0_b(p2_vb)          +=  1.0 * y2_bbb(h6_ob,p2_vb,p7_vb) * iy1_2_1_b(h6_ob,p7_vb)               )
    ( i0_b(p2_vb)          +=  1.0 * y2_aba(h6_oa,p2_vb,p7_va) * iy1_2_1_a(h6_oa,p7_va)               )
    (   i1_1_b(p5_vb,cind)  =  1.0 * y2_bbb(h3_ob,p4_vb,p5_vb) * cholOV_b(h3_ob,p4_vb,cind)           )
    (   i1_1_b(p5_vb,cind) += -1.0 * y2_aba(h3_oa,p5_vb,p4_va) * cholOV_a(h3_oa,p4_va,cind)           )
    ( i0_b(p2_vb)          += -1.0 * i1_1_b(p5_vb,cind) * cholVV_b(p2_vb,p5_vb)                    )
    (   i1_2_b(h3_ob)       =  0.5 * y2_bbb(h4_ob,p5_vb,p6_vb) * v2ijab_bbbb(h3_ob,h4_ob,p5_vb,p6_vb) )
    (   i1_2_b(h3_ob)      +=  1.0 * y2_aba(h4_oa,p5_vb,p6_va) * v2ijab_abab(h4_oa,h3_ob,p6_va,p5_vb) )
    ( i0_b(p2_vb)          += -1.0 * t1_a(p2_vb,h3_ob) * i1_2_b(h3_ob)                                );
  }
  // clang-format on
  sch.deallocate(i1_1_b, i1_2_b);

  if(debug) sch.execute();
}

template<typename T>
void gfccsd_y2_a(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_aaa,
                 Tensor<std::complex<T>>& i0_bab, const Tensor<T>& t1_a, const Tensor<T>& t1_b,
                 const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& y1_a, const Tensor<std::complex<T>>& y2_aaa,
                 const Tensor<std::complex<T>>& y2_bab, const Tensor<T>& f1,
                 const Tensor<T>& iy1_1_a, const Tensor<T>& iy1_1_b, const Tensor<T>& iy1_2_1_a,
                 const Tensor<T>& iy1_2_1_b, const Tensor<T>& iy2_a, const Tensor<T>& iy2_b,
                 const Tensor<T>& iy3_1_aaaa, const Tensor<T>& iy3_1_baba,
                 const Tensor<T>& iy3_1_abba, const Tensor<T>& iy3_1_2_a,
                 const Tensor<T>& iy3_1_2_b, const Tensor<T>& iy3_aaaa, const Tensor<T>& iy3_bbbb,
                 const Tensor<T>& iy3_baba, const Tensor<T>& iy3_baab, const Tensor<T>& iy3_abba,
                 const Tensor<T>& iy4_1_aaaa, const Tensor<T>& iy4_1_baab,
                 const Tensor<T>& iy4_1_baba, const Tensor<T>& iy4_1_bbbb,
                 const Tensor<T>& iy4_1_abba, const Tensor<T>& iy4_2_aaaa,
                 const Tensor<T>& iy4_2_abba, const Tensor<T>& iy5_aaaa, const Tensor<T>& iy5_baab,
                 const Tensor<T>& iy5_bbbb, const Tensor<T>& iy5_baba, const Tensor<T>& iy5_abba,
                 const Tensor<T>& iy6_a, const Tensor<T>& iy6_b, const Tensor<T>& cholOO_a,
                 const Tensor<T>& cholOO_b, const Tensor<T>& cholOV_a, const Tensor<T>& cholOV_b,
                 const Tensor<T>& cholVV_a, const Tensor<T>& cholVV_b, const Tensor<T>& v2ijab_aaaa,
                 const Tensor<T>& v2ijab_abab, const TiledIndexSpace& CI,
                 const TiledIndexSpace& gf_tis, bool has_tis, bool debug = false) {
  using ComplexTensor = Tensor<std::complex<T>>;
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [u1]                = gf_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h5_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] =
    o_alpha.labels<10>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h5_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] =
    o_beta.labels<10>("all");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va, p7_va, p8_va, p9_va, p10_va] =
    v_alpha.labels<10>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p5_vb, p6_vb, p7_vb, p8_vb, p9_vb, p10_vb] =
    v_beta.labels<10>("all");

  /* spin implicit contraction
  sch
    (   i2_1(p4,cind)  =  1.0 * y1(p5) * cholVpr(p4,p5,cind) )
    ( i0_temp(h2,p3,p4)  = -1.0 * cholVpr(p3,h2,cind) * i2_1(p4,cind) )
    ( i0(h2,p3,p4)  = -1.0 * y2(h8,p3,p4) * iy2(h8,h2) )
    ( i0_temp(h2,p3,p4) +=  1.0 * y2(h2,p3,p8) * iy1_1(p4,p8) )
    ( i0_temp(h2,p3,p4) +=  1.0 * y2(h7,p3,p8) * iy3(h7,p4,h2,p8) )
    (   i2_3(h2,p3,p6,cind)  =  1.0 * y2(h2,p5,p6) * cholVpr(p3,p5,cind) )
    ( i0(h2,p3,p4) +=  1.0 * i2_3(h2,p3,p6,cind) * cholVpr(p4,p6,cind) )

    (   i2_4(h9,p3,h2)  = -0.5 * y1(p6) * v2iajb(h9,p3,h2,p6) )
    (   i2_4(h9,p3,h2) += -0.5 * y2(h2,p3,p5) * iy1_2_1(h9,p5) )
    (     i2_4_1(h8,h9,h1,p10)  =  1.0 * v2ijka(h8,h9,h1,p10) )
    (     i2_4_1(h8,h9,h1,p10) +=  1.0 * iy4(h8,h9,h1,p10) )
    (   i2_4(h9,p3,h2) +=  0.5 * y2(h8,p3,p10) * iy4(h8,h9,h2,p10) )
    (     i2_4_2(h9,p7,h2,cind)  =  1.0 * y2(h2,p6,p7) * cholVpr(h9,p6,cind) )
    (   i2_4(h9,p3,h2) +=  0.5 * i2_4_2(h9,p7,h2,cind) * cholVpr(p3,p7,cind) )
    (       i2_4_3_1(h8,h9,h1,p10)  =  1.0 * v2ijka(h8,h9,h1,p10) )
    (       i2_4_3_1(h8,h9,h1,p10)  = -1.0 * iy4(h8,h9,h1,p10) )
    (     i2_4_3(h9,h10,h2)  = -0.5 * y1(p7) * i2_4_3_1(h9,h10,h2,p7) )
    (     i2_4_3(h9,h10,h2) +=  0.25 * y2(h2,p7,p8) * v2ijab(h9,h10,p7,p8) )
    (   i2_4(h9,p3,h2) += -0.5 * d_t1(p3,h10) * i2_4_3(h9,h10,h2) )
    (   i2_4(h9,p3,h2) +=  0.5 * y1(p7) * iy3_1(h9,p3,h1,p7) )
    ( i0_temp(h2,p3,p4) += -2.0 * d_t1(p3,h9) * i2_4(h9,p4,h2) )
    ( i0_temp(h2,p3,p4) +=  1.0 * i2_1(p4,cind) * iy3_1_2(h1,p3,cind) )
    (   i2_5(h5)  =  1.0 * y1(p9) * iy1_2_1(h5,p9) )
    (   i2_5(h5) +=  0.5 * y2(h6,p7,p8) * v2ijab(h5,h6,p7,p8) )
    ( i0(h1,p3,p4) += -1.0 * d_t2(p3,p4,h1,h5) * i2_5(h5) )
    (   i2_6(h5,h6,h3)  = -0.5 * y1(p7) * i2_4_3_1(h5,h6,h2,p7) )
    (   i2_6(h5,h6,h3) +=  0.25 * y2(h2,p7,p8) * v2ijab(h5,h6,p7,p8) )
    ( i0(h2,p3,p4) +=  1.0 * d_t2(p3,p4,h5,h6) * i2_6(h5,h6,h2) )
    ( i0_temp(h1,p3,p4) +=  1.0 * i2_1(p4,cind) * iy6(h1,p3,cind) )
    (     i2_7_2(h6,cind)  =  1.0 * y1(p7) * cholOV(h6,p7,cind) )
    (   i2_7(h6,p3,p5) += -1.0 * i2_7_2(h6,cind) * cholVV(p3,p5,cind) )
    ( i0_temp(h1,p3,p4) +=  1.0 * t2(p3,p5,h1,h6) * i2_7(h6,p4,p5) )
    ( i0_temp(h1,p3,p4) +=  1.0 * y2(h7,p4,p8) * iy5(h7,p3,h1,p8) )
    ( i0(h2,p3,p4) +=  1.0 * i0_temp(h2,p3,p4) )
    ( i0(h2,p3,p4) += -1.0 * i0_temp(h2,p4,p3) )
  */

  ComplexTensor i2_1_a;
  ComplexTensor i2_3_aaa;
  ComplexTensor i2_3_bab;
  ComplexTensor i2_4_1_a;
  ComplexTensor i2_4_aaa;
  ComplexTensor i2_4_abb;
  ComplexTensor i2_4_bab;
  ComplexTensor i2_4_2_aaa;
  ComplexTensor i2_4_2_abb;
  ComplexTensor i2_4_2_bab;
  ComplexTensor i2_4_3_aaa;
  ComplexTensor i2_4_3_bab;
  ComplexTensor i2_5_a;
  ComplexTensor i2_7_aaa;
  ComplexTensor i2_7_bab;
  ComplexTensor i2_7_abb;
  ComplexTensor i0_temp_aaa;
  ComplexTensor i0_temp_bba;
  ComplexTensor i0_temp_bab;

  if(has_tis) {
    i2_1_a      = {v_alpha, CI, gf_tis};
    i2_3_aaa    = {o_alpha, v_alpha, v_alpha, CI, gf_tis};
    i2_3_bab    = {o_beta, v_alpha, v_beta, CI, gf_tis};
    i2_4_1_a    = {o_alpha, CI, gf_tis};
    i2_4_aaa    = {o_alpha, v_alpha, o_alpha, gf_tis};
    i2_4_abb    = {o_alpha, v_beta, o_beta, gf_tis};
    i2_4_bab    = {o_beta, v_alpha, o_beta, gf_tis};
    i2_4_2_aaa  = {o_alpha, v_alpha, o_alpha, CI, gf_tis};
    i2_4_2_abb  = {o_alpha, v_beta, o_beta, CI, gf_tis};
    i2_4_2_bab  = {o_beta, v_alpha, o_beta, CI, gf_tis};
    i2_4_3_aaa  = {o_alpha, o_alpha, o_alpha, gf_tis};
    i2_4_3_bab  = {o_beta, o_alpha, o_beta, gf_tis};
    i2_5_a      = {o_alpha, gf_tis};
    i2_7_aaa    = {o_alpha, v_alpha, v_alpha, gf_tis};
    i2_7_bab    = {o_beta, v_alpha, v_beta, gf_tis};
    i2_7_abb    = {o_alpha, v_beta, v_beta, gf_tis};
    i0_temp_aaa = {o_alpha, v_alpha, v_alpha, gf_tis};
    i0_temp_bba = {o_beta, v_beta, v_alpha, gf_tis};
    i0_temp_bab = {o_beta, v_alpha, v_beta, gf_tis};
  }
  else {
    i2_1_a      = {v_alpha, CI};
    i2_3_aaa    = {o_alpha, v_alpha, v_alpha, CI};
    i2_3_bab    = {o_beta, v_alpha, v_beta, CI};
    i2_4_1_a    = {o_alpha, CI};
    i2_4_aaa    = {o_alpha, v_alpha, o_alpha};
    i2_4_abb    = {o_alpha, v_beta, o_beta};
    i2_4_bab    = {o_beta, v_alpha, o_beta};
    i2_4_2_aaa  = {o_alpha, v_alpha, o_alpha, CI};
    i2_4_2_abb  = {o_alpha, v_beta, o_beta, CI};
    i2_4_2_bab  = {o_beta, v_alpha, o_beta, CI};
    i2_4_3_aaa  = {o_alpha, o_alpha, o_alpha};
    i2_4_3_bab  = {o_beta, o_alpha, o_beta};
    i2_5_a      = {o_alpha};
    i2_7_aaa    = {o_alpha, v_alpha, v_alpha};
    i2_7_bab    = {o_beta, v_alpha, v_beta};
    i2_7_abb    = {o_alpha, v_beta, v_beta};
    i0_temp_aaa = {o_alpha, v_alpha, v_alpha};
    i0_temp_bba = {o_beta, v_beta, v_alpha};
    i0_temp_bab = {o_beta, v_alpha, v_beta};
  }

  /* intermediates needed
  iy1_1_a,iy1_1_b,iy2_a,iy2_b,
  iy3_1_aaaa,iy3_1_abba,iy3_1_baba,iy3_1_bbbb,iy3_1_baab,iy3_1_abab,
  iy3_1_2_a,iy3_1_2_b,
  iy3_aaaa,iy3_baab,iy3_abba,iy3_bbbb,iy3_baba,iy3_abab,
  cholOO_a,cholOO_b,
  iy4_1_aaaa,iy4_1_baab,iy4_1_baba,iy4_1_bbbb,iy4_1_abba,iy4_1_abab,
  iy4_2_aaaa,iy4_2_abba,iy4_2_bbbb,iy4_2_baab,
  iy5_aaaa,iy5_baab,iy5_baba,iy5_abba,iy5_bbbb,iy5_abab,
  iy6_a,iy6_b
  */

  sch.allocate(i2_1_a, i2_3_aaa, i2_3_bab, i2_4_1_a, i2_4_aaa, i2_4_abb, i2_4_bab, i2_4_2_aaa,
               i2_4_2_abb, i2_4_2_bab, i2_4_3_aaa, i2_4_3_bab, i2_5_a, i2_7_aaa, i2_7_bab, i2_7_abb,
               i0_temp_aaa, i0_temp_bba, i0_temp_bab);
  if(has_tis) {
    // clang-format off
    sch(   i2_1_a(p5_va,cind,u1)  =  1.0 * y1_a(p4_va,u1) * cholVV_a(p4_va,p5_va,cind) )
      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1)  = -1.0 * cholOV_a(h2_oa,p3_va,cind) * i2_1_a(p4_va,cind,u1) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va,u1)  = -1.0 * cholOV_b(h2_ob,p3_vb,cind) * i2_1_a(p4_va,cind,u1) )

      ( i0_aaa(h2_oa,p3_va,p4_va,u1)  = -1.0 * y2_aaa(h8_oa,p3_va,p4_va,u1) * iy2_a(h8_oa,h2_oa) )
      ( i0_bab(h2_ob,p3_va,p4_vb,u1)  = -1.0 * y2_bab(h8_ob,p3_va,p4_vb,u1) * iy2_b(h8_ob,h2_ob) )

      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1) +=  1.0 * y2_aaa(h2_oa,p3_va,p8_va,u1) * iy1_1_a(p4_va,p8_va) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb,u1)  =  1.0 * y2_bab(h2_ob,p3_va,p8_vb,u1) * iy1_1_b(p4_vb,p8_vb) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va,u1) += -1.0 * y2_bab(h2_ob,p8_va,p3_vb,u1) * iy1_1_a(p4_va,p8_va) )

      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1) += -1.0 * y2_aaa(h7_oa,p3_va,p8_va,u1) * iy3_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1) += -1.0 * y2_bab(h7_ob,p3_va,p8_vb,u1) * iy3_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb,u1) += -1.0 * y2_aaa(h7_oa,p3_va,p8_va,u1) * iy3_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb,u1) += -1.0 * y2_bab(h7_ob,p3_va,p8_vb,u1) * iy3_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va,u1) +=  1.0 * y2_bab(h7_ob,p8_va,p3_vb,u1) * iy3_baba(h7_ob,p4_va,h2_ob,p8_va) )
      
      (   i2_3_aaa(h2_oa,p3_va,p6_va,cind,u1)  =  1.0 * y2_aaa(h2_oa,p5_va,p6_va,u1) * cholVV_a(p3_va,p5_va,cind) )
      (   i2_3_bab(h2_ob,p3_va,p6_vb,cind,u1)  =  1.0 * y2_bab(h2_ob,p5_va,p6_vb,u1) * cholVV_a(p3_va,p5_va,cind) )
      ( i0_aaa(h2_oa,p3_va,p4_va,u1) +=  1.0 * i2_3_aaa(h2_oa,p3_va,p6_va,cind,u1) * cholVV_a(p4_va,p6_va,cind) )
      ( i0_bab(h2_ob,p3_va,p4_vb,u1) +=  1.0 * i2_3_bab(h2_ob,p3_va,p6_vb,cind,u1) * cholVV_b(p4_vb,p6_vb,cind) )

      (     i2_4_1_a(h6_oa,cind,u1)  =  1.0 * y1_a(p7_va,u1) * cholOV_a(h6_oa,p7_va,cind) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1)  = -0.5 * i2_1_a(p3_va,cind,u1) * cholOO_a(h9_oa,h2_oa,cind)   )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) +=  0.5 * i2_4_1_a(h9_oa,cind,u1) * cholOV_a(h2_oa,p3_va,cind) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1)  =  0.5 * i2_4_1_a(h9_oa,cind,u1) * cholOV_b(h2_ob,p3_vb,cind) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1)  = -0.5 * i2_1_a(p3_va,cind,u1) * cholOO_b(h9_ob,h2_ob,cind)   )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) += -0.5 * y2_aaa(h2_oa,p3_va,p5_va,u1) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1) +=  0.5 * y2_bab(h2_ob,p5_va,p3_vb,u1) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) += -0.5 * y2_bab(h2_ob,p3_va,p5_vb,u1) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) +=  0.5 * y2_aaa(h8_oa,p3_va,p10_va,u1) * iy4_1_aaaa(h8_oa,h9_oa,h2_oa,p10_va) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) +=  0.5 * y2_bab(h8_ob,p3_va,p10_vb,u1) * iy4_1_baab(h8_ob,h9_oa,h2_oa,p10_vb) )      
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1) += -0.5 * y2_bab(h8_ob,p10_va,p3_vb,u1) * iy4_1_baba(h8_ob,h9_oa,h2_ob,p10_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) +=  0.5 * y2_bab(h8_ob,p3_va,p10_vb,u1) * iy4_1_bbbb(h8_ob,h9_ob,h2_ob,p10_vb) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) +=  0.5 * y2_aaa(h8_oa,p3_va,p10_va,u1) * iy4_1_abba(h8_oa,h9_ob,h2_ob,p10_va) )
      (     i2_4_2_aaa(h9_oa,p7_va,h2_oa,cind,u1)  =  1.0 * y2_aaa(h2_oa,p6_va,p7_va,u1) * cholOV_a(h9_oa,p6_va,cind) )
      (     i2_4_2_abb(h9_oa,p7_vb,h2_ob,cind,u1)  =  1.0 * y2_bab(h2_ob,p6_va,p7_vb,u1) * cholOV_a(h9_oa,p6_va,cind) )
      (     i2_4_2_bab(h9_ob,p7_va,h2_ob,cind,u1)  = -1.0 * y2_bab(h2_ob,p7_va,p6_vb,u1) * cholOV_b(h9_ob,p6_vb,cind) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) +=  0.5 * i2_4_2_aaa(h9_oa,p7_va,h2_oa,cind,u1) * cholVV_a(p3_va,p7_va,cind) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1) +=  0.5 * i2_4_2_abb(h9_oa,p7_vb,h2_ob,cind,u1) * cholVV_b(p3_vb,p7_vb,cind) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) +=  0.5 * i2_4_2_bab(h9_ob,p7_va,h2_ob,cind,u1) * cholVV_a(p3_va,p7_va,cind) )
      (     i2_4_3_aaa(h9_oa,h10_oa,h2_oa,u1)  = -0.5 * y1_a(p7_va,u1) * iy4_2_aaaa(h9_oa,h10_oa,h2_oa,p7_va) )
      (     i2_4_3_bab(h9_ob,h10_oa,h2_ob,u1)  =  0.5 * y1_a(p7_va,u1) * iy4_2_abba(h10_oa,h9_ob,h2_ob,p7_va) )
      (     i2_4_3_aaa(h9_oa,h10_oa,h2_oa,u1) +=  0.25 * y2_aaa(h2_oa,p7_va,p8_va,u1) * v2ijab_aaaa(h9_oa,h10_oa,p7_va,p8_va) )   
      (     i2_4_3_bab(h9_ob,h10_oa,h2_ob,u1) += -0.5  * y2_bab(h2_ob,p7_va,p8_vb,u1) * v2ijab_abab(h10_oa,h9_ob,p7_va,p8_vb) )         
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aaa(h9_oa,h10_oa,h2_oa,u1) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1) +=  0.5 * t1_b(p3_vb,h10_ob) * i2_4_3_bab(h10_ob,h9_oa,h2_ob,u1) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_bab(h9_ob,h10_oa,h2_ob,u1) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa,u1) +=  0.5 * y1_a(p7_va,u1) * iy3_1_aaaa(h9_oa,p3_va,h2_oa,p7_va) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob,u1) +=  0.5 * y1_a(p7_va,u1) * iy3_1_abba(h9_oa,p3_vb,h2_ob,p7_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob,u1) +=  0.5 * y1_a(p7_va,u1) * iy3_1_baba(h9_ob,p3_va,h2_ob,p7_va) )      
      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_aaa(h9_oa,p4_va,h2_oa,u1) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb,u1) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_abb(h9_oa,p4_vb,h2_ob,u1) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va,u1) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_bab(h9_ob,p4_va,h2_ob,u1) )
 
      ( i0_temp_aaa(h2_oa,p3_va,p4_va,u1) +=  1.0 * i2_1_a(p4_va,cind,u1) * iy3_1_2_a(h2_oa,p3_va,cind) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va,u1) +=  1.0 * i2_1_a(p4_va,cind,u1) * iy3_1_2_b(h2_ob,p3_vb,cind) )

      (   i2_5_a(h5_oa,u1)  =  1.0 * y1_a(p9_va,u1) * iy1_2_1_a(h5_oa,p9_va) )
      (   i2_5_a(h5_oa,u1) +=  0.5 * y2_aaa(h6_oa,p7_va,p8_va,u1) * v2ijab_aaaa(h5_oa,h6_oa,p7_va,p8_va) )
      (   i2_5_a(h5_oa,u1) +=  1.0 * y2_bab(h6_ob,p7_va,p8_vb,u1) * v2ijab_abab(h5_oa,h6_ob,p7_va,p8_vb) )     
      ( i0_aaa(h1_oa,p3_va,p4_va,u1) += -1.0 * t2_aaaa(p3_va,p4_va,h1_oa,h5_oa) * i2_5_a(h5_oa,u1) )
      ( i0_bab(h1_ob,p3_va,p4_vb,u1) +=  1.0 * t2_abab(p3_va,p4_vb,h5_oa,h1_ob) * i2_5_a(h5_oa,u1) )
      
      ( i0_aaa(h2_oa,p3_va,p4_va,u1) +=  1.0 * t2_aaaa(p3_va,p4_va,h5_oa,h6_oa) * i2_4_3_aaa(h5_oa,h6_oa,h2_oa,u1) )
      ( i0_bab(h2_ob,p3_va,p4_vb,u1) += -2.0 * t2_abab(p3_va,p4_vb,h5_oa,h6_ob) * i2_4_3_bab(h6_ob,h5_oa,h2_ob,u1) )

      ( i0_temp_aaa(h1_oa,p3_va,p4_va,u1) += 1.0 * i2_1_a(p4_va,cind,u1) * iy6_a(h1_oa,p3_va,cind) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va,u1) += 1.0 * i2_1_a(p4_va,cind,u1) * iy6_b(h1_ob,p3_vb,cind) )
      (   i2_7_aaa(h6_oa,p3_va,p5_va,u1)  = -1.0 * i2_4_1_a(h6_oa,cind,u1) * cholVV_a(p3_va,p5_va,cind) )
      (   i2_7_abb(h6_oa,p3_vb,p5_vb,u1)  = -1.0 * i2_4_1_a(h6_oa,cind,u1) * cholVV_b(p3_vb,p5_vb,cind) )
      ( i0_temp_aaa(h1_oa,p3_va,p4_va,u1) +=  1.0 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i2_7_aaa(h6_oa,p4_va,p5_va,u1) )
      ( i0_temp_bab(h1_ob,p3_va,p4_vb,u1) += -1.0 * t2_abab(p3_va,p5_vb,h6_oa,h1_ob) * i2_7_abb(h6_oa,p4_vb,p5_vb,u1) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va,u1) +=  1.0 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i2_7_aaa(h6_oa,p4_va,p5_va,u1) )
      
      ( i0_temp_aaa(h1_oa,p3_va,p4_va,u1) += -1.0 * y2_aaa(h7_oa,p4_va,p8_va,u1) * iy5_aaaa(h7_oa,p3_va,h1_oa,p8_va) )
      ( i0_temp_aaa(h1_oa,p3_va,p4_va,u1) += -1.0 * y2_bab(h7_ob,p4_va,p8_vb,u1) * iy5_baab(h7_ob,p3_va,h1_oa,p8_vb) )      
      ( i0_temp_bab(h1_ob,p3_va,p4_vb,u1) +=  1.0 * y2_bab(h7_ob,p8_va,p4_vb,u1) * iy5_baba(h7_ob,p3_va,h1_ob,p8_va) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va,u1) += -1.0 * y2_aaa(h7_oa,p4_va,p8_va,u1) * iy5_abba(h7_oa,p3_vb,h1_ob,p8_va) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va,u1) += -1.0 * y2_bab(h7_ob,p4_va,p8_vb,u1) * iy5_bbbb(h7_ob,p3_vb,h1_ob,p8_vb) )

      ( i0_aaa(h2_oa,p3_va,p4_va,u1) +=  1.0 * i0_temp_aaa(h2_oa,p3_va,p4_va,u1) )
      ( i0_bab(h2_ob,p3_va,p4_vb,u1) +=  1.0 * i0_temp_bab(h2_ob,p3_va,p4_vb,u1) )
      ( i0_aaa(h2_oa,p3_va,p4_va,u1) += -1.0 * i0_temp_aaa(h2_oa,p4_va,p3_va,u1) )
      ( i0_bab(h2_ob,p3_va,p4_vb,u1) += -1.0 * i0_temp_bba(h2_ob,p4_vb,p3_va,u1) );
    // clang-format on
  }
  else {
    // clang-format off
    sch(i2_1_a(p5_va,cind)  =  1.0 * y1_a(p4_va) * cholVV_a(p4_va,p5_va,cind) )
      ( i0_temp_aaa(h2_oa,p3_va,p4_va)  = -1.0 * cholOV_a(h2_oa,p3_va,cind) * i2_1_a(p4_va,cind) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va)  = -1.0 * cholOV_b(h2_ob,p3_vb,cind) * i2_1_a(p4_va,cind) )

      ( i0_aaa(h2_oa,p3_va,p4_va)  = -1.0 * y2_aaa(h8_oa,p3_va,p4_va) * iy2_a(h8_oa,h2_oa) )
      ( i0_bab(h2_ob,p3_va,p4_vb)  = -1.0 * y2_bab(h8_ob,p3_va,p4_vb) * iy2_b(h8_ob,h2_ob) )

      ( i0_temp_aaa(h2_oa,p3_va,p4_va) +=  1.0 * y2_aaa(h2_oa,p3_va,p8_va) * iy1_1_a(p4_va,p8_va) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb)  =  1.0 * y2_bab(h2_ob,p3_va,p8_vb) * iy1_1_b(p4_vb,p8_vb) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va) += -1.0 * y2_bab(h2_ob,p8_va,p3_vb) * iy1_1_a(p4_va,p8_va) )

      ( i0_temp_aaa(h2_oa,p3_va,p4_va) += -1.0 * y2_aaa(h7_oa,p3_va,p8_va) * iy3_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aaa(h2_oa,p3_va,p4_va) += -1.0 * y2_bab(h7_ob,p3_va,p8_vb) * iy3_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb) += -1.0 * y2_aaa(h7_oa,p3_va,p8_va) * iy3_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb) += -1.0 * y2_bab(h7_ob,p3_va,p8_vb) * iy3_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va) +=  1.0 * y2_bab(h7_ob,p8_va,p3_vb) * iy3_baba(h7_ob,p4_va,h2_ob,p8_va) )
      
      (   i2_3_aaa(h2_oa,p3_va,p6_va,cind)  =  1.0 * y2_aaa(h2_oa,p5_va,p6_va) * cholVV_a(p3_va,p5_va,cind) )
      (   i2_3_bab(h2_ob,p3_va,p6_vb,cind)  =  1.0 * y2_bab(h2_ob,p5_va,p6_vb) * cholVV_a(p3_va,p5_va,cind) )
      ( i0_aaa(h2_oa,p3_va,p4_va) +=  1.0 * i2_3_aaa(h2_oa,p3_va,p6_va,cind) * cholVV_a(p4_va,p6_va,cind) )
      ( i0_bab(h2_ob,p3_va,p4_vb) +=  1.0 * i2_3_bab(h2_ob,p3_va,p6_vb,cind) * cholVV_b(p4_vb,p6_vb,cind) )

      (     i2_4_1_a(h6_oa,cind)  =  1.0 * y1_a(p7_va) * cholOV_a(h6_oa,p7_va,cind) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa)  = -0.5 * i2_1_a(p3_va,cind) * cholOO_a(h9_oa,h2_oa,cind)   )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) +=  0.5 * i2_4_1_a(h9_oa,cind) * cholOV_a(h2_oa,p3_va,cind) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob)  =  0.5 * i2_4_1_a(h9_oa,cind) * cholOV_b(h2_ob,p3_vb,cind) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob)  = -0.5 * i2_1_a(p3_va,cind) * cholOO_b(h9_ob,h2_ob,cind)   )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) += -0.5 * y2_aaa(h2_oa,p3_va,p5_va) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob) +=  0.5 * y2_bab(h2_ob,p5_va,p3_vb) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) += -0.5 * y2_bab(h2_ob,p3_va,p5_vb) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) +=  0.5 * y2_aaa(h8_oa,p3_va,p10_va) * iy4_1_aaaa(h8_oa,h9_oa,h2_oa,p10_va) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) +=  0.5 * y2_bab(h8_ob,p3_va,p10_vb) * iy4_1_baab(h8_ob,h9_oa,h2_oa,p10_vb) )      
      (   i2_4_abb(h9_oa,p3_vb,h2_ob) += -0.5 * y2_bab(h8_ob,p10_va,p3_vb) * iy4_1_baba(h8_ob,h9_oa,h2_ob,p10_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) +=  0.5 * y2_bab(h8_ob,p3_va,p10_vb) * iy4_1_bbbb(h8_ob,h9_ob,h2_ob,p10_vb) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) +=  0.5 * y2_aaa(h8_oa,p3_va,p10_va) * iy4_1_abba(h8_oa,h9_ob,h2_ob,p10_va) )
      (     i2_4_2_aaa(h9_oa,p7_va,h2_oa,cind)  =  1.0 * y2_aaa(h2_oa,p6_va,p7_va) * cholOV_a(h9_oa,p6_va,cind) )
      (     i2_4_2_abb(h9_oa,p7_vb,h2_ob,cind)  =  1.0 * y2_bab(h2_ob,p6_va,p7_vb) * cholOV_a(h9_oa,p6_va,cind) )
      (     i2_4_2_bab(h9_ob,p7_va,h2_ob,cind)  = -1.0 * y2_bab(h2_ob,p7_va,p6_vb) * cholOV_b(h9_ob,p6_vb,cind) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) +=  0.5 * i2_4_2_aaa(h9_oa,p7_va,h2_oa,cind) * cholVV_a(p3_va,p7_va,cind) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob) +=  0.5 * i2_4_2_abb(h9_oa,p7_vb,h2_ob,cind) * cholVV_b(p3_vb,p7_vb,cind) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) +=  0.5 * i2_4_2_bab(h9_ob,p7_va,h2_ob,cind) * cholVV_a(p3_va,p7_va,cind) )
      (     i2_4_3_aaa(h9_oa,h10_oa,h2_oa)  = -0.5 * y1_a(p7_va) * iy4_2_aaaa(h9_oa,h10_oa,h2_oa,p7_va) )
      (     i2_4_3_bab(h9_ob,h10_oa,h2_ob)  =  0.5 * y1_a(p7_va) * iy4_2_abba(h10_oa,h9_ob,h2_ob,p7_va) )
      (     i2_4_3_aaa(h9_oa,h10_oa,h2_oa) +=  0.25 * y2_aaa(h2_oa,p7_va,p8_va) * v2ijab_aaaa(h9_oa,h10_oa,p7_va,p8_va) )   
      (     i2_4_3_bab(h9_ob,h10_oa,h2_ob) += -0.5  * y2_bab(h2_ob,p7_va,p8_vb) * v2ijab_abab(h10_oa,h9_ob,p7_va,p8_vb) )         
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aaa(h9_oa,h10_oa,h2_oa) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob) +=  0.5 * t1_b(p3_vb,h10_ob) * i2_4_3_bab(h10_ob,h9_oa,h2_ob) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_bab(h9_ob,h10_oa,h2_ob) )
      (   i2_4_aaa(h9_oa,p3_va,h2_oa) +=  0.5 * y1_a(p7_va) * iy3_1_aaaa(h9_oa,p3_va,h2_oa,p7_va) )
      (   i2_4_abb(h9_oa,p3_vb,h2_ob) +=  0.5 * y1_a(p7_va) * iy3_1_abba(h9_oa,p3_vb,h2_ob,p7_va) )
      (   i2_4_bab(h9_ob,p3_va,h2_ob) +=  0.5 * y1_a(p7_va) * iy3_1_baba(h9_ob,p3_va,h2_ob,p7_va) )      
      ( i0_temp_aaa(h2_oa,p3_va,p4_va) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_aaa(h9_oa,p4_va,h2_oa) )
      ( i0_temp_bab(h2_ob,p3_va,p4_vb) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_abb(h9_oa,p4_vb,h2_ob) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_bab(h9_ob,p4_va,h2_ob) )
 
      ( i0_temp_aaa(h2_oa,p3_va,p4_va) +=  1.0 * i2_1_a(p4_va,cind) * iy3_1_2_a(h2_oa,p3_va,cind) )
      ( i0_temp_bba(h2_ob,p3_vb,p4_va) +=  1.0 * i2_1_a(p4_va,cind) * iy3_1_2_b(h2_ob,p3_vb,cind) )

      (   i2_5_a(h5_oa)  =  1.0 * y1_a(p9_va) * iy1_2_1_a(h5_oa,p9_va) )
      (   i2_5_a(h5_oa) +=  0.5 * y2_aaa(h6_oa,p7_va,p8_va) * v2ijab_aaaa(h5_oa,h6_oa,p7_va,p8_va) )
      (   i2_5_a(h5_oa) +=  1.0 * y2_bab(h6_ob,p7_va,p8_vb) * v2ijab_abab(h5_oa,h6_ob,p7_va,p8_vb) )     
      ( i0_aaa(h1_oa,p3_va,p4_va) += -1.0 * t2_aaaa(p3_va,p4_va,h1_oa,h5_oa) * i2_5_a(h5_oa) )
      ( i0_bab(h1_ob,p3_va,p4_vb) +=  1.0 * t2_abab(p3_va,p4_vb,h5_oa,h1_ob) * i2_5_a(h5_oa) )
      
      ( i0_aaa(h2_oa,p3_va,p4_va) +=  1.0 * t2_aaaa(p3_va,p4_va,h5_oa,h6_oa) * i2_4_3_aaa(h5_oa,h6_oa,h2_oa) )
      ( i0_bab(h2_ob,p3_va,p4_vb) += -2.0 * t2_abab(p3_va,p4_vb,h5_oa,h6_ob) * i2_4_3_bab(h6_ob,h5_oa,h2_ob) )

      ( i0_temp_aaa(h1_oa,p3_va,p4_va) += 1.0 * i2_1_a(p4_va,cind) * iy6_a(h1_oa,p3_va,cind) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va) += 1.0 * i2_1_a(p4_va,cind) * iy6_b(h1_ob,p3_vb,cind) )
      (   i2_7_aaa(h6_oa,p3_va,p5_va)  = -1.0 * i2_4_1_a(h6_oa,cind) * cholVV_a(p3_va,p5_va,cind) )
      (   i2_7_abb(h6_oa,p3_vb,p5_vb)  = -1.0 * i2_4_1_a(h6_oa,cind) * cholVV_b(p3_vb,p5_vb,cind) )
      ( i0_temp_aaa(h1_oa,p3_va,p4_va) +=  1.0 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i2_7_aaa(h6_oa,p4_va,p5_va) )
      ( i0_temp_bab(h1_ob,p3_va,p4_vb) += -1.0 * t2_abab(p3_va,p5_vb,h6_oa,h1_ob) * i2_7_abb(h6_oa,p4_vb,p5_vb) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va) +=  1.0 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i2_7_aaa(h6_oa,p4_va,p5_va) )
      
      ( i0_temp_aaa(h1_oa,p3_va,p4_va) += -1.0 * y2_aaa(h7_oa,p4_va,p8_va) * iy5_aaaa(h7_oa,p3_va,h1_oa,p8_va) )
      ( i0_temp_aaa(h1_oa,p3_va,p4_va) += -1.0 * y2_bab(h7_ob,p4_va,p8_vb) * iy5_baab(h7_ob,p3_va,h1_oa,p8_vb) )      
      ( i0_temp_bab(h1_ob,p3_va,p4_vb) +=  1.0 * y2_bab(h7_ob,p8_va,p4_vb) * iy5_baba(h7_ob,p3_va,h1_ob,p8_va) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va) += -1.0 * y2_aaa(h7_oa,p4_va,p8_va) * iy5_abba(h7_oa,p3_vb,h1_ob,p8_va) )
      ( i0_temp_bba(h1_ob,p3_vb,p4_va) += -1.0 * y2_bab(h7_ob,p4_va,p8_vb) * iy5_bbbb(h7_ob,p3_vb,h1_ob,p8_vb) )

      ( i0_aaa(h2_oa,p3_va,p4_va) +=  1.0 * i0_temp_aaa(h2_oa,p3_va,p4_va) )
      ( i0_bab(h2_ob,p3_va,p4_vb) +=  1.0 * i0_temp_bab(h2_ob,p3_va,p4_vb) )
      ( i0_aaa(h2_oa,p3_va,p4_va) += -1.0 * i0_temp_aaa(h2_oa,p4_va,p3_va) )
      ( i0_bab(h2_ob,p3_va,p4_vb) += -1.0 * i0_temp_bba(h2_ob,p4_vb,p3_va) );
    // clang-format on
  }
  sch.deallocate(i2_1_a, i2_3_aaa, i2_3_bab, i2_4_1_a, i2_4_aaa, i2_4_abb, i2_4_bab, i2_4_2_aaa,
                 i2_4_2_abb, i2_4_2_bab, i2_4_3_aaa, i2_4_3_bab, i2_5_a, i2_7_aaa, i2_7_bab,
                 i2_7_abb, i0_temp_aaa, i0_temp_bba, i0_temp_bab);

  if(debug) sch.execute();
}

template<typename T>
void gfccsd_y2_b(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_bbb,
                 Tensor<std::complex<T>>& i0_aba, const Tensor<T>& t1_a, const Tensor<T>& t1_b,
                 const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& y1_b, const Tensor<std::complex<T>>& y2_bbb,
                 const Tensor<std::complex<T>>& y2_aba, const Tensor<T>& f1,
                 const Tensor<T>& iy1_1_a, const Tensor<T>& iy1_1_b, const Tensor<T>& iy1_2_1_a,
                 const Tensor<T>& iy1_2_1_b, const Tensor<T>& iy2_a, const Tensor<T>& iy2_b,
                 const Tensor<T>& iy3_1_bbbb, const Tensor<T>& iy3_1_abab,
                 const Tensor<T>& iy3_1_baab, const Tensor<T>& iy3_1_2_a,
                 const Tensor<T>& iy3_1_2_b, const Tensor<T>& iy3_aaaa, const Tensor<T>& iy3_bbbb,
                 const Tensor<T>& iy3_abab, const Tensor<T>& iy3_baab, const Tensor<T>& iy3_abba,
                 const Tensor<T>& iy4_1_aaaa, const Tensor<T>& iy4_1_baab,
                 const Tensor<T>& iy4_1_bbbb, const Tensor<T>& iy4_1_abba,
                 const Tensor<T>& iy4_1_abab, const Tensor<T>& iy4_2_bbbb,
                 const Tensor<T>& iy4_2_baab, const Tensor<T>& iy5_aaaa, const Tensor<T>& iy5_baab,
                 const Tensor<T>& iy5_bbbb, const Tensor<T>& iy5_abab, const Tensor<T>& iy5_abba,
                 const Tensor<T>& iy6_a, const Tensor<T>& iy6_b, const Tensor<T>& cholOO_a,
                 const Tensor<T>& cholOO_b, const Tensor<T>& cholOV_a, const Tensor<T>& cholOV_b,
                 const Tensor<T>& cholVV_a, const Tensor<T>& cholVV_b, const Tensor<T>& v2ijab_bbbb,
                 const Tensor<T>& v2ijab_abab, const TiledIndexSpace& CI,
                 const TiledIndexSpace& gf_tis, bool has_tis, bool debug = false) {
  using ComplexTensor = Tensor<std::complex<T>>;
  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [u1]                = gf_tis.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  // const int obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int vbtiles = MO("virt_beta").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [cind] = CI.labels<1>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h5_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] =
    o_alpha.labels<10>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h5_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] =
    o_beta.labels<10>("all");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p6_va, p7_va, p8_va, p9_va, p10_va] =
    v_alpha.labels<10>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p5_vb, p6_vb, p7_vb, p8_vb, p9_vb, p10_vb] =
    v_beta.labels<10>("all");

  ComplexTensor i2_1_b;
  ComplexTensor i2_3_bbb;
  ComplexTensor i2_3_aba;
  ComplexTensor i2_4_1_b;
  ComplexTensor i2_4_bbb;
  ComplexTensor i2_4_baa;
  ComplexTensor i2_4_aba;
  ComplexTensor i2_4_2_bbb;
  ComplexTensor i2_4_2_baa;
  ComplexTensor i2_4_2_aba;
  ComplexTensor i2_4_3_bbb;
  ComplexTensor i2_4_3_aba;
  ComplexTensor i2_5_b;
  ComplexTensor i2_7_bbb;
  ComplexTensor i2_7_aba;
  ComplexTensor i2_7_baa;
  ComplexTensor i0_temp_bbb;
  ComplexTensor i0_temp_aab;
  ComplexTensor i0_temp_aba;

  if(has_tis) {
    i2_1_b      = {v_beta, CI, gf_tis};
    i2_3_bbb    = {o_beta, v_beta, v_beta, CI, gf_tis};
    i2_3_aba    = {o_alpha, v_beta, v_alpha, CI, gf_tis};
    i2_4_1_b    = {o_beta, CI, gf_tis};
    i2_4_bbb    = {o_beta, v_beta, o_beta, gf_tis};
    i2_4_baa    = {o_beta, v_alpha, o_alpha, gf_tis};
    i2_4_aba    = {o_alpha, v_beta, o_alpha, gf_tis};
    i2_4_2_bbb  = {o_beta, v_beta, o_beta, CI, gf_tis};
    i2_4_2_baa  = {o_beta, v_alpha, o_alpha, CI, gf_tis};
    i2_4_2_aba  = {o_alpha, v_beta, o_alpha, CI, gf_tis};
    i2_4_3_bbb  = {o_beta, o_beta, o_beta, gf_tis};
    i2_4_3_aba  = {o_alpha, o_beta, o_alpha, gf_tis};
    i2_5_b      = {o_beta, gf_tis};
    i2_7_bbb    = {o_beta, v_beta, v_beta, gf_tis};
    i2_7_aba    = {o_alpha, v_beta, v_alpha, gf_tis};
    i2_7_baa    = {o_beta, v_alpha, v_alpha, gf_tis};
    i0_temp_bbb = {o_beta, v_beta, v_beta, gf_tis};
    i0_temp_aab = {o_alpha, v_alpha, v_beta, gf_tis};
    i0_temp_aba = {o_alpha, v_beta, v_alpha, gf_tis};
  }
  else {
    i2_1_b      = {v_beta, CI};
    i2_3_bbb    = {o_beta, v_beta, v_beta, CI};
    i2_3_aba    = {o_alpha, v_beta, v_alpha, CI};
    i2_4_1_b    = {o_beta, CI};
    i2_4_bbb    = {o_beta, v_beta, o_beta};
    i2_4_baa    = {o_beta, v_alpha, o_alpha};
    i2_4_aba    = {o_alpha, v_beta, o_alpha};
    i2_4_2_bbb  = {o_beta, v_beta, o_beta, CI};
    i2_4_2_baa  = {o_beta, v_alpha, o_alpha, CI};
    i2_4_2_aba  = {o_alpha, v_beta, o_alpha, CI};
    i2_4_3_bbb  = {o_beta, o_beta, o_beta};
    i2_4_3_aba  = {o_alpha, o_beta, o_alpha};
    i2_5_b      = {o_beta};
    i2_7_bbb    = {o_beta, v_beta, v_beta};
    i2_7_aba    = {o_alpha, v_beta, v_alpha};
    i2_7_baa    = {o_beta, v_alpha, v_alpha};
    i0_temp_bbb = {o_beta, v_beta, v_beta};
    i0_temp_aab = {o_alpha, v_alpha, v_beta};
    i0_temp_aba = {o_alpha, v_beta, v_alpha};
  }

  sch.allocate(i2_1_b, i2_3_bbb, i2_3_aba, i2_4_1_b, i2_4_bbb, i2_4_baa, i2_4_aba, i2_4_2_bbb,
               i2_4_2_baa, i2_4_2_aba, i2_4_3_bbb, i2_4_3_aba, i2_5_b, i2_7_bbb, i2_7_aba, i2_7_baa,
               i0_temp_bbb, i0_temp_aab, i0_temp_aba);

  if(has_tis) {
    // clang-format off
    sch(   i2_1_b(p4_vb,cind,u1)  =  1.0 * y1_b(p5_vb,u1) * cholVV_b(p4_vb,p5_vb,cind) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1)  = -1.0 * cholOV_b(h2_ob,p3_vb,cind) * i2_1_b(p4_vb,cind,u1) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb,u1)  = -1.0 * cholOV_a(h2_oa,p3_va,cind) * i2_1_b(p4_vb,cind,u1) )

      ( i0_bbb(h2_ob,p3_vb,p4_vb,u1)  = -1.0 * y2_bbb(h8_ob,p3_vb,p4_vb,u1) * iy2_b(h8_ob,h2_ob) )
      ( i0_aba(h2_oa,p3_vb,p4_va,u1)  = -1.0 * y2_aba(h8_oa,p3_vb,p4_va,u1) * iy2_a(h8_oa,h2_oa) )

      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * y2_bbb(h2_ob,p3_vb,p8_vb,u1) * iy1_1_b(p4_vb,p8_vb) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va,u1)  =  1.0 * y2_aba(h2_oa,p3_vb,p8_va,u1) * iy1_1_a(p4_va,p8_va) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb,u1) += -1.0 * y2_aba(h2_oa,p8_vb,p3_va,u1) * iy1_1_b(p4_vb,p8_vb) )

      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * y2_bbb(h7_ob,p3_vb,p8_vb,u1) * iy3_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * y2_aba(h7_oa,p3_vb,p8_va,u1) * iy3_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va,u1) +=  1.0 * y2_aba(h7_oa,p3_vb,p8_va,u1) * iy3_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va,u1) +=  1.0 * y2_bbb(h7_ob,p3_vb,p8_vb,u1) * iy3_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb,u1) += -1.0 * y2_aba(h7_oa,p8_vb,p3_va,u1) * iy3_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
      
      (   i2_3_bbb(h2_ob,p3_vb,p6_vb,cind,u1)  =  1.0 * y2_bbb(h2_ob,p5_vb,p6_vb,u1) * cholVV_b(p3_vb,p5_vb,cind) )
      (   i2_3_aba(h2_oa,p3_vb,p6_va,cind,u1)  =  1.0 * y2_aba(h2_oa,p5_vb,p6_va,u1) * cholVV_b(p3_vb,p5_vb,cind) )
      ( i0_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * i2_3_bbb(h2_ob,p3_vb,p6_vb,cind,u1) * cholVV_b(p4_vb,p6_vb,cind) )
      ( i0_aba(h2_oa,p3_vb,p4_va,u1) +=  1.0 * i2_3_aba(h2_oa,p3_vb,p6_va,cind,u1) * cholVV_a(p4_va,p6_va,cind) )

      (     i2_4_1_b(h6_ob,cind,u1)  =  1.0 * y1_b(p7_vb,u1) * cholOV_b(h6_ob,p7_vb,cind) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1)  = -0.5 * i2_1_b(p3_vb,cind,u1) * cholOO_b(h9_ob,h2_ob,cind)   )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) +=  0.5 * i2_4_1_b(h9_ob,cind,u1) * cholOV_b(h2_ob,p3_vb,cind) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1)  =  0.5 * i2_4_1_b(h9_ob,cind,u1) * cholOV_a(h2_oa,p3_va,cind) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1)  = -0.5 * i2_1_b(p3_vb,cind,u1) * cholOO_a(h9_oa,h2_oa,cind)   )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) += -0.5 * y2_bbb(h2_ob,p3_vb,p5_vb,u1) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1) +=  0.5 * y2_aba(h2_oa,p5_vb,p3_va,u1) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1) += -0.5 * y2_aba(h2_oa,p3_vb,p5_va,u1) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) +=  0.5 * y2_bbb(h8_ob,p3_vb,p10_vb,u1) * iy4_1_bbbb(h8_ob,h9_ob,h2_ob,p10_vb) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) +=  0.5 * y2_aba(h8_oa,p3_vb,p10_va,u1) * iy4_1_abba(h8_oa,h9_ob,h2_ob,p10_va) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1) += -0.5 * y2_aba(h8_oa,p10_vb,p3_va,u1) * iy4_1_abab(h8_oa,h9_ob,h2_oa,p10_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1) +=  0.5 * y2_aba(h8_oa,p3_vb,p10_va,u1) * iy4_1_aaaa(h8_oa,h9_oa,h2_oa,p10_va) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1) +=  0.5 * y2_bbb(h8_ob,p3_vb,p10_vb,u1) * iy4_1_baab(h8_ob,h9_oa,h2_oa,p10_vb) )
      (     i2_4_2_bbb(h9_ob,p7_vb,h2_ob,cind,u1)  =  1.0 * y2_bbb(h2_ob,p6_vb,p7_vb,u1) * cholOV_b(h9_ob,p6_vb,cind) )
      (     i2_4_2_baa(h9_ob,p7_va,h2_oa,cind,u1)  =  1.0 * y2_aba(h2_oa,p6_vb,p7_va,u1) * cholOV_b(h9_ob,p6_vb,cind) )
      (     i2_4_2_aba(h9_oa,p7_vb,h2_oa,cind,u1)  = -1.0 * y2_aba(h2_oa,p7_vb,p6_va,u1) * cholOV_a(h9_oa,p6_va,cind) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) +=  0.5 * i2_4_2_bbb(h9_ob,p7_vb,h2_ob,cind,u1) * cholVV_b(p3_vb,p7_vb,cind) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1) +=  0.5 * i2_4_2_baa(h9_ob,p7_va,h2_oa,cind,u1) * cholVV_a(p3_va,p7_va,cind) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1) +=  0.5 * i2_4_2_aba(h9_oa,p7_vb,h2_oa,cind,u1) * cholVV_b(p3_vb,p7_vb,cind) )
      (     i2_4_3_bbb(h9_ob,h10_ob,h2_ob,u1)  = -0.5 * y1_b(p7_vb,u1) * iy4_2_bbbb(h9_ob,h10_ob,h2_ob,p7_vb) )
      (     i2_4_3_aba(h9_oa,h10_ob,h2_oa,u1)  =  0.5 * y1_b(p7_vb,u1) * iy4_2_baab(h10_ob,h9_oa,h2_oa,p7_vb) )
      (     i2_4_3_bbb(h9_ob,h10_ob,h2_ob,u1) +=  0.25 * y2_bbb(h2_ob,p7_vb,p8_vb,u1) * v2ijab_bbbb(h9_ob,h10_ob,p7_vb,p8_vb) )   
      (     i2_4_3_aba(h9_oa,h10_ob,h2_oa,u1) += -0.5  * y2_aba(h2_oa,p7_vb,p8_va,u1) * v2ijab_abab(h9_oa,h10_ob,p8_va,p7_vb) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) += -0.5 * t1_b(p3_vb,h10_ob) * i2_4_3_bbb(h9_ob,h10_ob,h2_ob,u1) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1) +=  0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aba(h10_oa,h9_ob,h2_oa,u1) )
      (   i2_4_aba(h9_oa,p3_va,h2_oa,u1) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aba(h9_oa,h10_ob,h2_oa,u1) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob,u1) +=  0.5 * y1_b(p7_vb,u1) * iy3_1_bbbb(h9_ob,p3_vb,h2_ob,p7_vb) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa,u1) +=  0.5 * y1_b(p7_vb,u1) * iy3_1_baab(h9_ob,p3_va,h2_oa,p7_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa,u1) +=  0.5 * y1_b(p7_vb,u1) * iy3_1_abab(h9_oa,p3_vb,h2_oa,p7_vb) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_bbb(h9_ob,p4_vb,h2_ob,u1) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va,u1) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_baa(h9_ob,p4_va,h2_oa,u1) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb,u1) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_aba(h9_oa,p4_vb,h2_oa,u1) )
 
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * i2_1_b(p4_vb,cind,u1) * iy3_1_2_b(h2_ob,p3_vb,cind) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb,u1) +=  1.0 * i2_1_b(p4_vb,cind,u1) * iy3_1_2_a(h2_oa,p3_va,cind) )

      (   i2_5_b(h5_ob,u1)  =  1.0 * y1_b(p9_vb,u1) * iy1_2_1_b(h5_ob,p9_vb) )
      (   i2_5_b(h5_ob,u1) +=  0.5 * y2_bbb(h6_ob,p7_vb,p8_vb,u1) * v2ijab_bbbb(h5_ob,h6_ob,p7_vb,p8_vb) )
      (   i2_5_b(h5_ob,u1) +=  1.0 * y2_aba(h6_oa,p7_vb,p8_va,u1) * v2ijab_abab(h6_oa,h5_ob,p8_va,p7_vb) )
      ( i0_bbb(h1_ob,p3_vb,p4_vb,u1) += -1.0 * t2_bbbb(p3_vb,p4_vb,h1_ob,h5_ob) * i2_5_b(h5_ob,u1) )
      ( i0_aba(h1_oa,p3_vb,p4_va,u1) +=  1.0 * t2_abab(p4_va,p3_vb,h1_oa,h5_ob) * i2_5_b(h5_ob,u1) )
      
      ( i0_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * t2_bbbb(p3_vb,p4_vb,h5_ob,h6_ob) * i2_4_3_bbb(h5_ob,h6_ob,h2_ob,u1) )
      ( i0_aba(h2_oa,p3_vb,p4_va,u1) += -2.0 * t2_abab(p4_va,p3_vb,h6_oa,h5_ob) * i2_4_3_aba(h6_oa,h5_ob,h2_oa,u1) )

      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb,u1) +=  1.0 * i2_1_b(p4_vb,cind,u1) * iy6_b(h1_ob,p3_vb,cind) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb,u1) +=  1.0 * i2_1_b(p4_vb,cind,u1) * iy6_a(h1_oa,p3_va,cind) )      
      (   i2_7_bbb(h6_ob,p3_vb,p5_vb,u1)  = -1.0 * i2_4_1_b(h6_ob,cind,u1) * cholVV_b(p3_vb,p5_vb,cind) )
      (   i2_7_baa(h6_ob,p3_va,p5_va,u1)  = -1.0 * i2_4_1_b(h6_ob,cind,u1) * cholVV_a(p3_va,p5_va,cind) )
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb,u1) +=  1.0 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i2_7_bbb(h6_ob,p4_vb,p5_vb,u1) )
      ( i0_temp_aba(h1_oa,p3_vb,p4_va,u1) += -1.0 * t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i2_7_baa(h6_ob,p4_va,p5_va,u1) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb,u1) +=  1.0 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i2_7_bbb(h6_ob,p4_vb,p5_vb,u1) )
      
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb,u1) +=  1.0 * y2_bbb(h7_ob,p4_vb,p8_vb,u1) * iy5_bbbb(h7_ob,p3_vb,h1_ob,p8_vb) )
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb,u1) +=  1.0 * y2_aba(h7_oa,p4_vb,p8_va,u1) * iy5_abba(h7_oa,p3_vb,h1_ob,p8_va) )
      ( i0_temp_aba(h1_oa,p3_vb,p4_va,u1) += -1.0 * y2_aba(h7_oa,p8_vb,p4_va,u1) * iy5_abab(h7_oa,p3_vb,h1_oa,p8_vb) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb,u1) +=  1.0 * y2_bbb(h7_ob,p4_vb,p8_vb,u1) * iy5_baab(h7_ob,p3_va,h1_oa,p8_vb) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb,u1) +=  1.0 * y2_aba(h7_oa,p4_vb,p8_va,u1) * iy5_aaaa(h7_oa,p3_va,h1_oa,p8_va) )

      ( i0_bbb(h2_ob,p3_vb,p4_vb,u1) +=  1.0 * i0_temp_bbb(h2_ob,p3_vb,p4_vb,u1) )
      ( i0_aba(h2_oa,p3_vb,p4_va,u1) +=  1.0 * i0_temp_aba(h2_oa,p3_vb,p4_va,u1) )
      ( i0_bbb(h2_ob,p3_vb,p4_vb,u1) += -1.0 * i0_temp_bbb(h2_ob,p4_vb,p3_vb,u1) )
      ( i0_aba(h2_oa,p3_vb,p4_va,u1) += -1.0 * i0_temp_aab(h2_oa,p4_va,p3_vb,u1) );
    // clang-format on
  }

  else {
    // clang-format off
      sch(i2_1_b(p4_vb,cind)  =  1.0 * y1_b(p5_vb) * cholVV_b(p4_vb,p5_vb,cind) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb)  = -1.0 * cholOV_b(h2_ob,p3_vb,cind) * i2_1_b(p4_vb,cind) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb)  = -1.0 * cholOV_a(h2_oa,p3_va,cind) * i2_1_b(p4_vb,cind) )

      ( i0_bbb(h2_ob,p3_vb,p4_vb)  = -1.0 * y2_bbb(h8_ob,p3_vb,p4_vb) * iy2_b(h8_ob,h2_ob) )
      ( i0_aba(h2_oa,p3_vb,p4_va)  = -1.0 * y2_aba(h8_oa,p3_vb,p4_va) * iy2_a(h8_oa,h2_oa) )

      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * y2_bbb(h2_ob,p3_vb,p8_vb) * iy1_1_b(p4_vb,p8_vb) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va)  =  1.0 * y2_aba(h2_oa,p3_vb,p8_va) * iy1_1_a(p4_va,p8_va) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb) += -1.0 * y2_aba(h2_oa,p8_vb,p3_va) * iy1_1_b(p4_vb,p8_vb) )

      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * y2_bbb(h7_ob,p3_vb,p8_vb) * iy3_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * y2_aba(h7_oa,p3_vb,p8_va) * iy3_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va) +=  1.0 * y2_aba(h7_oa,p3_vb,p8_va) * iy3_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va) +=  1.0 * y2_bbb(h7_ob,p3_vb,p8_vb) * iy3_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb) += -1.0 * y2_aba(h7_oa,p8_vb,p3_va) * iy3_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
      
      (   i2_3_bbb(h2_ob,p3_vb,p6_vb,cind)  =  1.0 * y2_bbb(h2_ob,p5_vb,p6_vb) * cholVV_b(p3_vb,p5_vb,cind) )
      (   i2_3_aba(h2_oa,p3_vb,p6_va,cind)  =  1.0 * y2_aba(h2_oa,p5_vb,p6_va) * cholVV_b(p3_vb,p5_vb,cind) )
      ( i0_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * i2_3_bbb(h2_ob,p3_vb,p6_vb,cind) * cholVV_b(p4_vb,p6_vb,cind) )
      ( i0_aba(h2_oa,p3_vb,p4_va) +=  1.0 * i2_3_aba(h2_oa,p3_vb,p6_va,cind) * cholVV_a(p4_va,p6_va,cind) )

      (     i2_4_1_b(h6_ob,cind)  =  1.0 * y1_b(p7_vb) * cholOV_b(h6_ob,p7_vb,cind) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob)  = -0.5 * i2_1_b(p3_vb,cind) * cholOO_b(h9_ob,h2_ob,cind)   )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) +=  0.5 * i2_4_1_b(h9_ob,cind) * cholOV_b(h2_ob,p3_vb,cind) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa)  =  0.5 * i2_4_1_b(h9_ob,cind) * cholOV_a(h2_oa,p3_va,cind) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa)  = -0.5 * i2_1_b(p3_vb,cind) * cholOO_a(h9_oa,h2_oa,cind)   )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) += -0.5 * y2_bbb(h2_ob,p3_vb,p5_vb) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa) +=  0.5 * y2_aba(h2_oa,p5_vb,p3_va) * iy1_2_1_b(h9_ob,p5_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa) += -0.5 * y2_aba(h2_oa,p3_vb,p5_va) * iy1_2_1_a(h9_oa,p5_va) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) +=  0.5 * y2_bbb(h8_ob,p3_vb,p10_vb) * iy4_1_bbbb(h8_ob,h9_ob,h2_ob,p10_vb) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) +=  0.5 * y2_aba(h8_oa,p3_vb,p10_va) * iy4_1_abba(h8_oa,h9_ob,h2_ob,p10_va) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa) += -0.5 * y2_aba(h8_oa,p10_vb,p3_va) * iy4_1_abab(h8_oa,h9_ob,h2_oa,p10_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa) +=  0.5 * y2_aba(h8_oa,p3_vb,p10_va) * iy4_1_aaaa(h8_oa,h9_oa,h2_oa,p10_va) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa) +=  0.5 * y2_bbb(h8_ob,p3_vb,p10_vb) * iy4_1_baab(h8_ob,h9_oa,h2_oa,p10_vb) )
      (     i2_4_2_bbb(h9_ob,p7_vb,h2_ob,cind)  =  1.0 * y2_bbb(h2_ob,p6_vb,p7_vb) * cholOV_b(h9_ob,p6_vb,cind) )
      (     i2_4_2_baa(h9_ob,p7_va,h2_oa,cind)  =  1.0 * y2_aba(h2_oa,p6_vb,p7_va) * cholOV_b(h9_ob,p6_vb,cind) )
      (     i2_4_2_aba(h9_oa,p7_vb,h2_oa,cind)  = -1.0 * y2_aba(h2_oa,p7_vb,p6_va) * cholOV_a(h9_oa,p6_va,cind) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) +=  0.5 * i2_4_2_bbb(h9_ob,p7_vb,h2_ob,cind) * cholVV_b(p3_vb,p7_vb,cind) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa) +=  0.5 * i2_4_2_baa(h9_ob,p7_va,h2_oa,cind) * cholVV_a(p3_va,p7_va,cind) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa) +=  0.5 * i2_4_2_aba(h9_oa,p7_vb,h2_oa,cind) * cholVV_b(p3_vb,p7_vb,cind) )
      (     i2_4_3_bbb(h9_ob,h10_ob,h2_ob)  = -0.5 * y1_b(p7_vb) * iy4_2_bbbb(h9_ob,h10_ob,h2_ob,p7_vb) )
      (     i2_4_3_aba(h9_oa,h10_ob,h2_oa)  =  0.5 * y1_b(p7_vb) * iy4_2_baab(h10_ob,h9_oa,h2_oa,p7_vb) )
      (     i2_4_3_bbb(h9_ob,h10_ob,h2_ob) +=  0.25 * y2_bbb(h2_ob,p7_vb,p8_vb) * v2ijab_bbbb(h9_ob,h10_ob,p7_vb,p8_vb) )   
      (     i2_4_3_aba(h9_oa,h10_ob,h2_oa) += -0.5  * y2_aba(h2_oa,p7_vb,p8_va) * v2ijab_abab(h9_oa,h10_ob,p8_va,p7_vb) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) += -0.5 * t1_b(p3_vb,h10_ob) * i2_4_3_bbb(h9_ob,h10_ob,h2_ob) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa) +=  0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aba(h10_oa,h9_ob,h2_oa) )
      (   i2_4_aba(h9_oa,p3_va,h2_oa) += -0.5 * t1_a(p3_va,h10_oa) * i2_4_3_aba(h9_oa,h10_ob,h2_oa) )
      (   i2_4_bbb(h9_ob,p3_vb,h2_ob) +=  0.5 * y1_b(p7_vb) * iy3_1_bbbb(h9_ob,p3_vb,h2_ob,p7_vb) )
      (   i2_4_baa(h9_ob,p3_va,h2_oa) +=  0.5 * y1_b(p7_vb) * iy3_1_baab(h9_ob,p3_va,h2_oa,p7_vb) )
      (   i2_4_aba(h9_oa,p3_vb,h2_oa) +=  0.5 * y1_b(p7_vb) * iy3_1_abab(h9_oa,p3_vb,h2_oa,p7_vb) )
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_bbb(h9_ob,p4_vb,h2_ob) )
      ( i0_temp_aba(h2_oa,p3_vb,p4_va) += -2.0 * t1_b(p3_vb,h9_ob) * i2_4_baa(h9_ob,p4_va,h2_oa) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb) += -2.0 * t1_a(p3_va,h9_oa) * i2_4_aba(h9_oa,p4_vb,h2_oa) )
 
      ( i0_temp_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * i2_1_b(p4_vb,cind) * iy3_1_2_b(h2_ob,p3_vb,cind) )
      ( i0_temp_aab(h2_oa,p3_va,p4_vb) +=  1.0 * i2_1_b(p4_vb,cind) * iy3_1_2_a(h2_oa,p3_va,cind) )

      (   i2_5_b(h5_ob)  =  1.0 * y1_b(p9_vb) * iy1_2_1_b(h5_ob,p9_vb) )
      (   i2_5_b(h5_ob) +=  0.5 * y2_bbb(h6_ob,p7_vb,p8_vb) * v2ijab_bbbb(h5_ob,h6_ob,p7_vb,p8_vb) )
      (   i2_5_b(h5_ob) +=  1.0 * y2_aba(h6_oa,p7_vb,p8_va) * v2ijab_abab(h6_oa,h5_ob,p8_va,p7_vb) )
      ( i0_bbb(h1_ob,p3_vb,p4_vb) += -1.0 * t2_bbbb(p3_vb,p4_vb,h1_ob,h5_ob) * i2_5_b(h5_ob) )
      ( i0_aba(h1_oa,p3_vb,p4_va) +=  1.0 * t2_abab(p4_va,p3_vb,h1_oa,h5_ob) * i2_5_b(h5_ob) )
      
      ( i0_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * t2_bbbb(p3_vb,p4_vb,h5_ob,h6_ob) * i2_4_3_bbb(h5_ob,h6_ob,h2_ob) )
      ( i0_aba(h2_oa,p3_vb,p4_va) += -2.0 * t2_abab(p4_va,p3_vb,h6_oa,h5_ob) * i2_4_3_aba(h6_oa,h5_ob,h2_oa) )

      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb) +=  1.0 * i2_1_b(p4_vb,cind) * iy6_b(h1_ob,p3_vb,cind) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb) +=  1.0 * i2_1_b(p4_vb,cind) * iy6_a(h1_oa,p3_va,cind) )      
      (   i2_7_bbb(h6_ob,p3_vb,p5_vb)  = -1.0 * i2_4_1_b(h6_ob,cind) * cholVV_b(p3_vb,p5_vb,cind) )
      (   i2_7_baa(h6_ob,p3_va,p5_va)  = -1.0 * i2_4_1_b(h6_ob,cind) * cholVV_a(p3_va,p5_va,cind) )
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb) +=  1.0 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i2_7_bbb(h6_ob,p4_vb,p5_vb) )
      ( i0_temp_aba(h1_oa,p3_vb,p4_va) += -1.0 * t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i2_7_baa(h6_ob,p4_va,p5_va) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb) +=  1.0 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i2_7_bbb(h6_ob,p4_vb,p5_vb) )
      
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb) +=  1.0 * y2_bbb(h7_ob,p4_vb,p8_vb) * iy5_bbbb(h7_ob,p3_vb,h1_ob,p8_vb) )
      ( i0_temp_bbb(h1_ob,p3_vb,p4_vb) +=  1.0 * y2_aba(h7_oa,p4_vb,p8_va) * iy5_abba(h7_oa,p3_vb,h1_ob,p8_va) )
      ( i0_temp_aba(h1_oa,p3_vb,p4_va) += -1.0 * y2_aba(h7_oa,p8_vb,p4_va) * iy5_abab(h7_oa,p3_vb,h1_oa,p8_vb) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb) +=  1.0 * y2_bbb(h7_ob,p4_vb,p8_vb) * iy5_baab(h7_ob,p3_va,h1_oa,p8_vb) )
      ( i0_temp_aab(h1_oa,p3_va,p4_vb) +=  1.0 * y2_aba(h7_oa,p4_vb,p8_va) * iy5_aaaa(h7_oa,p3_va,h1_oa,p8_va) )

      ( i0_bbb(h2_ob,p3_vb,p4_vb) +=  1.0 * i0_temp_bbb(h2_ob,p3_vb,p4_vb) )
      ( i0_aba(h2_oa,p3_vb,p4_va) +=  1.0 * i0_temp_aba(h2_oa,p3_vb,p4_va) )
      ( i0_bbb(h2_ob,p3_vb,p4_vb) += -1.0 * i0_temp_bbb(h2_ob,p4_vb,p3_vb) )
      ( i0_aba(h2_oa,p3_vb,p4_va) += -1.0 * i0_temp_aab(h2_oa,p4_va,p3_vb) );
    // clang-format on
  }

  sch.deallocate(i2_1_b, i2_3_bbb, i2_3_aba, i2_4_1_b, i2_4_bbb, i2_4_baa, i2_4_aba, i2_4_2_bbb,
                 i2_4_2_baa, i2_4_2_aba, i2_4_3_bbb, i2_4_3_aba, i2_5_b, i2_7_bbb, i2_7_aba,
                 i2_7_baa, i0_temp_bbb, i0_temp_aab, i0_temp_aba);

  if(debug) sch.execute();
}
