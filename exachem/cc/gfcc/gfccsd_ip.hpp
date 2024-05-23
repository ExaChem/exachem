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
void gfccsd_x1_a(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_a,
                 const Tensor<T>& t1_a, const Tensor<T>& t1_b, const Tensor<T>& t2_aaaa,
                 const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& x1_a, const Tensor<std::complex<T>>& x2_aaa,
                 const Tensor<std::complex<T>>& x2_bab, const Tensor<T>& f1,
                 const Tensor<T>& ix2_2_a, const Tensor<T>& ix1_1_1_a, const Tensor<T>& ix1_1_1_b,
                 const Tensor<T>& ix2_6_3_aaaa, const Tensor<T>& ix2_6_3_abab,
                 const TiledIndexSpace& gf_tis, bool has_tis, bool debug = false) {
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

  auto [p7_va]               = v_alpha.labels<1>("all");
  auto [p7_vb]               = v_beta.labels<1>("all");
  auto [h1_oa, h6_oa, h8_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h6_ob, h8_ob] = o_beta.labels<3>("all");
  auto [u1]                  = gf_tis.labels<1>("all");

  if(has_tis) {
    // clang-format off
    sch
      ( i0_a(h1_oa,u1)  =  0 )
      ( i0_a(h1_oa,u1) += -1   * x1_a(h6_oa,u1) * ix2_2_a(h6_oa,h1_oa) )
      ( i0_a(h1_oa,u1) +=        x2_aaa(p7_va,h1_oa,h6_oa,u1) * ix1_1_1_a(h6_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h1_oa,h6_ob,u1) * ix1_1_1_b(h6_ob,p7_vb) )
      ( i0_a(h1_oa,u1) +=  0.5 * x2_aaa(p7_va,h6_oa,h8_oa,u1) * ix2_6_3_aaaa(h6_oa,h8_oa,h1_oa,p7_va) )
      ( i0_a(h1_oa,u1) +=        x2_bab(p7_vb,h6_oa,h8_ob,u1) * ix2_6_3_abab(h6_oa,h8_ob,h1_oa,p7_vb) );
    // clang-format on
  }
  else {
    // clang-format off
    sch
      ( i0_a(h1_oa)  =  0 )
      ( i0_a(h1_oa) += -1   * x1_a(h6_oa) * ix2_2_a(h6_oa,h1_oa) )
      ( i0_a(h1_oa) +=        x2_aaa(p7_va,h1_oa,h6_oa) * ix1_1_1_a(h6_oa,p7_va) )
      ( i0_a(h1_oa) +=        x2_bab(p7_vb,h1_oa,h6_ob) * ix1_1_1_b(h6_ob,p7_vb) )
      ( i0_a(h1_oa) +=  0.5 * x2_aaa(p7_va,h6_oa,h8_oa) * ix2_6_3_aaaa(h6_oa,h8_oa,h1_oa,p7_va) )
      ( i0_a(h1_oa) +=        x2_bab(p7_vb,h6_oa,h8_ob) * ix2_6_3_abab(h6_oa,h8_ob,h1_oa,p7_vb) );
    // clang-format on
  }
  // if(debug) sch.execute();
}

template<typename T>
void gfccsd_x1_b(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_b,
                 const Tensor<T>& t1_a, const Tensor<T>& t1_b, const Tensor<T>& t2_aaaa,
                 const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& x1_b, const Tensor<std::complex<T>>& x2_bbb,
                 const Tensor<std::complex<T>>& x2_aba, const Tensor<T>& f1,
                 const Tensor<T>& ix2_2_b, const Tensor<T>& ix1_1_1_a, const Tensor<T>& ix1_1_1_b,
                 const Tensor<T>& ix2_6_3_bbbb, const Tensor<T>& ix2_6_3_baba,
                 const TiledIndexSpace& gf_tis, bool has_tis, bool debug = false) {
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

  auto [p7_va]               = v_alpha.labels<1>("all");
  auto [p7_vb]               = v_beta.labels<1>("all");
  auto [h1_oa, h6_oa, h8_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h6_ob, h8_ob] = o_beta.labels<3>("all");
  auto [u1]                  = gf_tis.labels<1>("all");

  // clang-format off
  if(has_tis) {
    sch
    ( i0_b(h1_ob,u1)  =  0 )
    ( i0_b(h1_ob,u1) += -1   * x1_b(h6_ob,u1) * ix2_2_b(h6_ob,h1_ob) )
    ( i0_b(h1_ob,u1) +=        x2_bbb(p7_vb,h1_ob,h6_ob,u1) * ix1_1_1_b(h6_ob,p7_vb) )
    ( i0_b(h1_ob,u1) +=        x2_aba(p7_va,h1_ob,h6_oa,u1) * ix1_1_1_a(h6_oa,p7_va) )
    ( i0_b(h1_ob,u1) +=  0.5 * x2_bbb(p7_vb,h6_ob,h8_ob,u1) * ix2_6_3_bbbb(h6_ob,h8_ob,h1_ob,p7_vb) )
    ( i0_b(h1_ob,u1) +=        x2_aba(p7_va,h6_ob,h8_oa,u1) * ix2_6_3_baba(h6_ob,h8_oa,h1_ob,p7_va) );
  }
  else {
    sch
    ( i0_b(h1_ob)  =  0 )
    ( i0_b(h1_ob) += -1   * x1_b(h6_ob) * ix2_2_b(h6_ob,h1_ob) )
    ( i0_b(h1_ob) +=        x2_bbb(p7_vb,h1_ob,h6_ob) * ix1_1_1_b(h6_ob,p7_vb) )
    ( i0_b(h1_ob) +=        x2_aba(p7_va,h1_ob,h6_oa) * ix1_1_1_a(h6_oa,p7_va) )
    ( i0_b(h1_ob) +=  0.5 * x2_bbb(p7_vb,h6_ob,h8_ob) * ix2_6_3_bbbb(h6_ob,h8_ob,h1_ob,p7_vb) )
    ( i0_b(h1_ob) +=        x2_aba(p7_va,h6_ob,h8_oa) * ix2_6_3_baba(h6_ob,h8_oa,h1_ob,p7_va) );      
  }
  // clang-format on
  // if(debug) sch.execute();
}

template<typename T>
void gfccsd_x2_a(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_aaa,
                 Tensor<std::complex<T>>& i0_bab, const Tensor<T>& t1_a, const Tensor<T>& t1_b,
                 const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& x1_a, const Tensor<std::complex<T>>& x2_aaa,
                 const Tensor<std::complex<T>>& x2_bab, const Tensor<T>& f1,
                 const Tensor<T>& ix2_1_aaaa, const Tensor<T>& ix2_1_abab, const Tensor<T>& ix2_2_a,
                 const Tensor<T>& ix2_2_b, const Tensor<T>& ix2_3_a, const Tensor<T>& ix2_3_b,
                 const Tensor<T>& ix2_4_aaaa, const Tensor<T>& ix2_4_abab,
                 const Tensor<T>& ix2_5_aaaa, const Tensor<T>& ix2_5_abba,
                 const Tensor<T>& ix2_5_abab, const Tensor<T>& ix2_5_bbbb,
                 const Tensor<T>& ix2_5_baab, const Tensor<T>& ix2_6_2_a,
                 const Tensor<T>& ix2_6_2_b, const Tensor<T>& ix2_6_3_aaaa,
                 const Tensor<T>& ix2_6_3_abba, const Tensor<T>& ix2_6_3_abab,
                 const Tensor<T>& ix2_6_3_bbbb, const Tensor<T>& ix2_6_3_baab,
                 const Tensor<T>& v2ijab_aaaa, const Tensor<T>& v2ijab_abab,
                 const Tensor<T>& v2ijab_bbbb, const TiledIndexSpace& gf_tis, bool has_tis,
                 bool debug = false) {
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

  auto [p3_va, p4_va, p5_va, p8_va, p9_va]                = v_alpha.labels<5>("all");
  auto [p3_vb, p4_vb, p5_vb, p8_vb, p9_vb]                = v_beta.labels<5>("all");
  auto [h1_oa, h2_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] = o_alpha.labels<7>("all");
  auto [h1_ob, h2_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] = o_beta.labels<7>("all");

  ComplexTensor i_6_aaa;
  ComplexTensor i_6_bab;
  ComplexTensor i_10_a;
  ComplexTensor i_11_aaa;
  ComplexTensor i_11_bab;
  ComplexTensor i_11_bba;
  ComplexTensor i0_temp_aaa;
  ComplexTensor i0_temp_bab;
  ComplexTensor i0_temp_bba;
  ComplexTensor i_6_temp_aaa;
  ComplexTensor i_6_temp_bab;
  ComplexTensor i_6_temp_bba;

  if(has_tis) {
    i_6_aaa      = {o_alpha, o_alpha, o_alpha, gf_tis};
    i_6_bab      = {o_beta, o_alpha, o_beta, gf_tis};
    i_10_a       = {v_alpha, gf_tis};
    i_11_aaa     = {o_alpha, o_alpha, v_alpha, gf_tis};
    i_11_bab     = {o_beta, o_alpha, v_beta, gf_tis};
    i_11_bba     = {o_beta, o_beta, v_alpha, gf_tis};
    i0_temp_aaa  = {v_alpha, o_alpha, o_alpha, gf_tis};
    i0_temp_bab  = {v_beta, o_alpha, o_beta, gf_tis};
    i0_temp_bba  = {v_beta, o_beta, o_alpha, gf_tis};
    i_6_temp_aaa = {o_alpha, o_alpha, o_alpha, gf_tis};
    i_6_temp_bab = {o_beta, o_alpha, o_beta, gf_tis};
    i_6_temp_bba = {o_beta, o_beta, o_alpha, gf_tis};
  }
  else {
    i_6_aaa      = {o_alpha, o_alpha, o_alpha};
    i_6_bab      = {o_beta, o_alpha, o_beta};
    i_10_a       = {v_alpha};
    i_11_aaa     = {o_alpha, o_alpha, v_alpha};
    i_11_bab     = {o_beta, o_alpha, v_beta};
    i_11_bba     = {o_beta, o_beta, v_alpha};
    i0_temp_aaa  = {v_alpha, o_alpha, o_alpha};
    i0_temp_bab  = {v_beta, o_alpha, o_beta};
    i0_temp_bba  = {v_beta, o_beta, o_alpha};
    i_6_temp_aaa = {o_alpha, o_alpha, o_alpha};
    i_6_temp_bab = {o_beta, o_alpha, o_beta};
    i_6_temp_bba = {o_beta, o_beta, o_alpha};
  }
  //  sch
  //    .allocate(i_6,i_10,i_11,i0_temp,i_6_temp)
  //    ( i0(p4,h1,h2)                  =  0                                               )
  //    ( i0(p4,h1,h2)                 +=         x1(h9)        * ix2_1(h9,p4,h1,h2)       )
  //    ( i0_temp(p3,h1,h2)             =         x2(p3,h1,h8)  * ix2_2(h8,h2)             )
  //    ( i0(p3,h1,h2)                 += -1    * i0_temp(p3,h1,h2)                        )
  //    ( i0(p3,h2,h1)                 +=         i0_temp(p3,h1,h2)                        )
  //    ( i0(p4,h1,h2)                 +=         x2(p8,h1,h2)  * ix2_3(p4,p8)             )
  //    ( i0(p3,h1,h2)                 +=  0.5  * x2(p3,h9,h10) * ix2_4(h9,h10,h1,h2)      ) //O4V
  //    ( i0_temp(p4,h1,h2)             =         x2(p8,h1,h7)  * ix2_5(h7,p4,h2,p8)       ) //O3V2
  //    ( i0(p4,h1,h2)                 += -1    * i0_temp(p4,h1,h2)                        )
  //    ( i0(p4,h2,h1)                 +=         i0_temp(p4,h1,h2)                        )
  //    (   i_6(h10,h1,h2)              = -1    * x1(h8)        * ix2_6_1(h8,h10,h1,h2)    )
  //    (   i_6(h10,h1,h2)             +=         x2(p5,h1,h2)  * ix2_6_2(h10,p5)          )
  //    (   i_6_temp(h10,h2,h1)         =         x2(p9,h1,h8)  * ix2_6_3(h8,h10,h2,p9)    ) //O4V
  //    (   i_6(h10,h1,h2)             += -1    * i_6_temp(h10,h1,h2)                      )
  //    (   i_6(h10,h2,h1)             +=         i_6_temp(h10,h1,h2)                      )
  //    ( i0(p3,h1,h2)                  +=         d_t1(p3,h10)       * i_6(h10,h1,h2)       )

  //    (   i_10(p5)                    =  0.5  * x2(p8,h6,h7)  * v2(h6,h7,p5,p8)          )
  //    ( i0(p3,h1,h2)                 +=  1.0  * d_t2(p3,p5,h1,h2)  * i_10(p5)              )

  //    (   i_11(h6,h1,p5)              =         x2(p8,h1,h7)  * v2(h6,h7,p5,p8)          ) //O3V2
  //    ( i0_temp(p3,h2,h1)             =         d_t2(p3,p5,h1,h6)  * i_11(h6,h2,p5)        )
  //    //O3V2 ( i0(p3,h1,h2)                 +=         i0_temp(p3,h1,h2)                        )
  //    ( i0(p3,h2,h1)                 += -1    * i0_temp(p3,h1,h2)                        )
  //    .deallocate(i_6,i_10,i_11,i0_temp,i_6_temp);//.execute();

  sch.allocate(i_6_aaa, i_6_bab, i_10_a, i_11_aaa, i_11_bab, i_11_bba, i0_temp_aaa, i0_temp_bab,
               i0_temp_bba, i_6_temp_aaa, i_6_temp_bab, i_6_temp_bba);

  if(has_tis) {
    // clang-format off
    sch( i0_aaa(p4_va,h1_oa,h2_oa,u1)    =  0 )
      ( i0_bab(p4_vb,h1_oa,h2_ob,u1)    =  0 )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa,u1)      +=  x1_a(h9_oa,u1) * ix2_1_aaaa(h9_oa,p4_va,h1_oa,h2_oa) ) 
      ( i0_bab(p4_vb,h1_oa,h2_ob,u1)      +=  x1_a(h9_oa,u1) * ix2_1_abab(h9_oa,p4_vb,h1_oa,h2_ob) )
      
      ( i0_temp_aaa(p3_va,h1_oa,h2_oa,u1)  =       x2_aaa(p3_va,h1_oa,h8_oa,u1) * ix2_2_a(h8_oa,h2_oa) )
      ( i0_temp_bab(p3_vb,h1_oa,h2_ob,u1)  =       x2_bab(p3_vb,h1_oa,h8_ob,u1) * ix2_2_b(h8_ob,h2_ob) )
      ( i0_temp_bba(p3_vb,h1_ob,h2_oa,u1)  =  -1 * x2_bab(p3_vb,h8_oa,h1_ob,u1) * ix2_2_a(h8_oa,h2_oa) )
  
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa,u1) +=       x2_aaa(p8_va,h1_oa,h7_oa,u1) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) ) //O3V2
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa,u1) +=       x2_bab(p8_vb,h1_oa,h7_ob,u1) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob,u1) +=       x2_bab(p8_vb,h1_oa,h7_ob,u1) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob,u1) +=       x2_aaa(p8_va,h1_oa,h7_oa,u1) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_bba(p4_vb,h1_ob,h2_oa,u1) +=  -1 * x2_bab(p8_vb,h7_oa,h1_ob,u1) * ix2_5_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
  
      (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)   =  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aaa(h6_oa,h1_oa,p5_va,u1)  +=  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)   =  x2_bab(p8_vb,h1_oa,h7_ob,u1) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb,u1)  +=  x2_aaa(p8_va,h1_oa,h7_oa,u1) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_bba(h6_ob,h1_ob,p5_va,u1)   =  x2_bab(p8_vb,h7_oa,h1_ob,u1) * v2ijab_abab(h7_oa,h6_ob,p5_va,p8_vb) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa,u1) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aaa(h6_oa,h2_oa,p5_va,u1) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa,u1) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb,u1) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob,u1) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb,u1) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aaa(h6_oa,h2_oa,p5_va,u1) )
      ( i0_temp_bba(p3_vb,h2_ob,h1_oa,u1) +=      t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i_11_bba(h6_ob,h2_ob,p5_va,u1) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa,u1) += -1 * i0_temp_aaa(p3_va,h1_oa,h2_oa,u1) )
      ( i0_aaa(p3_va,h2_oa,h1_oa,u1) +=      i0_temp_aaa(p3_va,h1_oa,h2_oa,u1) )
      ( i0_bab(p3_vb,h1_oa,h2_ob,u1) += -1 * i0_temp_bab(p3_vb,h1_oa,h2_ob,u1) )
      ( i0_bab(p3_vb,h2_oa,h1_ob,u1) +=      i0_temp_bba(p3_vb,h1_ob,h2_oa,u1) )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa,u1) +=  x2_aaa(p8_va,h1_oa,h2_oa,u1) * ix2_3_a(p4_va,p8_va) )
      ( i0_bab(p4_vb,h1_oa,h2_ob,u1) +=  x2_bab(p8_vb,h1_oa,h2_ob,u1) * ix2_3_b(p4_vb,p8_vb) )
  
      ( i0_aaa(p3_va,h1_oa,h2_oa,u1) +=  0.5 * x2_aaa(p3_va,h9_oa,h10_oa,u1) * ix2_4_aaaa(h9_oa,h10_oa,h1_oa,h2_oa) ) //O4V
      ( i0_bab(p3_vb,h1_oa,h2_ob,u1) +=        x2_bab(p3_vb,h9_oa,h10_ob,u1) * ix2_4_abab(h9_oa,h10_ob,h1_oa,h2_ob) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1)  = -1 * x1_a(h8_oa,u1) * ix2_4_aaaa(h8_oa,h10_oa,h1_oa,h2_oa) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob,u1)  = -1 * x1_a(h8_oa,u1) * ix2_4_abab(h8_oa,h10_ob,h1_oa,h2_ob) )
      
      (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1) +=  x2_aaa(p5_va,h1_oa,h2_oa,u1) * ix2_6_2_a(h10_oa,p5_va) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob,u1) +=  x2_bab(p5_vb,h1_oa,h2_ob,u1) * ix2_6_2_b(h10_ob,p5_vb) )
      
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1)  =  x2_aaa(p9_va,h2_oa,h8_oa,u1) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) +=  x2_bab(p9_vb,h2_oa,h8_ob,u1) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) ) 
  
      (   i_6_temp_bab(h10_ob,h1_oa,h2_ob,u1)  =  -1 * x2_bab(p9_vb,h8_oa,h2_ob,u1) * ix2_6_3_abab(h8_oa,h10_ob,h1_oa,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1)  =       x2_bab(p9_vb,h2_oa,h8_ob,u1) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1) +=       x2_aaa(p9_va,h2_oa,h8_oa,u1) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa,u1) += -1 * i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) )
      (   i_6_aaa(h10_oa,h2_oa,h1_oa,u1) +=      i_6_temp_aaa(h10_oa,h1_oa,h2_oa,u1) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob,u1) += -1 * i_6_temp_bab(h10_ob,h1_oa,h2_ob,u1) )
      (   i_6_bab(h10_ob,h2_oa,h1_ob,u1) +=      i_6_temp_bba(h10_ob,h1_ob,h2_oa,u1) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa,u1)  +=  t1_a(p3_va,h10_oa) * i_6_aaa(h10_oa,h1_oa,h2_oa,u1) )
      ( i0_bab(p3_vb,h1_oa,h2_ob,u1)  +=  t1_b(p3_vb,h10_ob) * i_6_bab(h10_ob,h1_oa,h2_ob,u1) )
  
      (   i_10_a(p5_va,u1)  =  0.5 * x2_aaa(p8_va,h6_oa,h7_oa,u1) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_10_a(p5_va,u1) +=        x2_bab(p8_vb,h6_oa,h7_ob,u1) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      ( i0_aaa(p3_va,h1_oa,h2_oa,u1) +=      t2_aaaa(p3_va,p5_va,h1_oa,h2_oa) * i_10_a(p5_va,u1) )
      ( i0_bab(p3_vb,h1_oa,h2_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h1_oa,h2_ob) * i_10_a(p5_va,u1) );
    // clang-format on
  }
  else {
    // clang-format off
    sch( i0_aaa(p4_va,h1_oa,h2_oa)    =  0 )
      ( i0_bab(p4_vb,h1_oa,h2_ob)    =  0 )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa)      +=  x1_a(h9_oa) * ix2_1_aaaa(h9_oa,p4_va,h1_oa,h2_oa) ) 
      ( i0_bab(p4_vb,h1_oa,h2_ob)      +=  x1_a(h9_oa) * ix2_1_abab(h9_oa,p4_vb,h1_oa,h2_ob) )
      
      ( i0_temp_aaa(p3_va,h1_oa,h2_oa)  =       x2_aaa(p3_va,h1_oa,h8_oa) * ix2_2_a(h8_oa,h2_oa) )
      ( i0_temp_bab(p3_vb,h1_oa,h2_ob)  =       x2_bab(p3_vb,h1_oa,h8_ob) * ix2_2_b(h8_ob,h2_ob) )
      ( i0_temp_bba(p3_vb,h1_ob,h2_oa)  =  -1 * x2_bab(p3_vb,h8_oa,h1_ob) * ix2_2_a(h8_oa,h2_oa) )
  
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa) +=       x2_aaa(p8_va,h1_oa,h7_oa) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) ) //O3V2
      ( i0_temp_aaa(p4_va,h1_oa,h2_oa) +=       x2_bab(p8_vb,h1_oa,h7_ob) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob) +=       x2_bab(p8_vb,h1_oa,h7_ob) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) )
      ( i0_temp_bab(p4_vb,h1_oa,h2_ob) +=       x2_aaa(p8_va,h1_oa,h7_oa) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_bba(p4_vb,h1_ob,h2_oa) +=  -1 * x2_bab(p8_vb,h7_oa,h1_ob) * ix2_5_abab(h7_oa,p4_vb,h2_oa,p8_vb) )
  
      (   i_11_aaa(h6_oa,h1_oa,p5_va)   =  x2_aaa(p8_va,h1_oa,h7_oa) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aaa(h6_oa,h1_oa,p5_va)  +=  x2_bab(p8_vb,h1_oa,h7_ob) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb)   =  x2_bab(p8_vb,h1_oa,h7_ob) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bab(h6_ob,h1_oa,p5_vb)  +=  x2_aaa(p8_va,h1_oa,h7_oa) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_bba(h6_ob,h1_ob,p5_va)   =  x2_bab(p8_vb,h7_oa,h1_ob) * v2ijab_abab(h7_oa,h6_ob,p5_va,p8_vb) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aaa(h6_oa,h2_oa,p5_va) )
      ( i0_temp_aaa(p3_va,h2_oa,h1_oa) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bab(h6_ob,h2_oa,p5_vb) )
      ( i0_temp_bab(p3_vb,h2_oa,h1_ob) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aaa(h6_oa,h2_oa,p5_va) )
      ( i0_temp_bba(p3_vb,h2_ob,h1_oa) +=      t2_abab(p5_va,p3_vb,h1_oa,h6_ob) * i_11_bba(h6_ob,h2_ob,p5_va) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa) += -1 * i0_temp_aaa(p3_va,h1_oa,h2_oa) )
      ( i0_aaa(p3_va,h2_oa,h1_oa) +=      i0_temp_aaa(p3_va,h1_oa,h2_oa) )
      ( i0_bab(p3_vb,h1_oa,h2_ob) += -1 * i0_temp_bab(p3_vb,h1_oa,h2_ob) )
      ( i0_bab(p3_vb,h2_oa,h1_ob) +=      i0_temp_bba(p3_vb,h1_ob,h2_oa) )
  
      ( i0_aaa(p4_va,h1_oa,h2_oa) +=  x2_aaa(p8_va,h1_oa,h2_oa) * ix2_3_a(p4_va,p8_va) )
      ( i0_bab(p4_vb,h1_oa,h2_ob) +=  x2_bab(p8_vb,h1_oa,h2_ob) * ix2_3_b(p4_vb,p8_vb) )
  
      ( i0_aaa(p3_va,h1_oa,h2_oa) +=  0.5 * x2_aaa(p3_va,h9_oa,h10_oa) * ix2_4_aaaa(h9_oa,h10_oa,h1_oa,h2_oa) ) //O4V
      ( i0_bab(p3_vb,h1_oa,h2_ob) +=        x2_bab(p3_vb,h9_oa,h10_ob) * ix2_4_abab(h9_oa,h10_ob,h1_oa,h2_ob) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa)  = -1 * x1_a(h8_oa) * ix2_4_aaaa(h8_oa,h10_oa,h1_oa,h2_oa) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob)  = -1 * x1_a(h8_oa) * ix2_4_abab(h8_oa,h10_ob,h1_oa,h2_ob) )
      
      (   i_6_aaa(h10_oa,h1_oa,h2_oa) +=  x2_aaa(p5_va,h1_oa,h2_oa) * ix2_6_2_a(h10_oa,p5_va) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob) +=  x2_bab(p5_vb,h1_oa,h2_ob) * ix2_6_2_b(h10_ob,p5_vb) )
      
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa)  =  x2_aaa(p9_va,h2_oa,h8_oa) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
      (   i_6_temp_aaa(h10_oa,h1_oa,h2_oa) +=  x2_bab(p9_vb,h2_oa,h8_ob) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) ) 
  
      (   i_6_temp_bab(h10_ob,h1_oa,h2_ob)  =  -1 * x2_bab(p9_vb,h8_oa,h2_ob) * ix2_6_3_abab(h8_oa,h10_ob,h1_oa,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa)  =       x2_bab(p9_vb,h2_oa,h8_ob) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
      (   i_6_temp_bba(h10_ob,h1_ob,h2_oa) +=       x2_aaa(p9_va,h2_oa,h8_oa) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
      (   i_6_aaa(h10_oa,h1_oa,h2_oa) += -1 * i_6_temp_aaa(h10_oa,h1_oa,h2_oa) )
      (   i_6_aaa(h10_oa,h2_oa,h1_oa) +=      i_6_temp_aaa(h10_oa,h1_oa,h2_oa) )
      (   i_6_bab(h10_ob,h1_oa,h2_ob) += -1 * i_6_temp_bab(h10_ob,h1_oa,h2_ob) )
      (   i_6_bab(h10_ob,h2_oa,h1_ob) +=      i_6_temp_bba(h10_ob,h1_ob,h2_oa) )
      
      ( i0_aaa(p3_va,h1_oa,h2_oa)  +=  t1_a(p3_va,h10_oa) * i_6_aaa(h10_oa,h1_oa,h2_oa) )
      ( i0_bab(p3_vb,h1_oa,h2_ob)  +=  t1_b(p3_vb,h10_ob) * i_6_bab(h10_ob,h1_oa,h2_ob) )

      (   i_10_a(p5_va)  =  0.5 * x2_aaa(p8_va,h6_oa,h7_oa) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_10_a(p5_va) +=        x2_bab(p8_vb,h6_oa,h7_ob) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      ( i0_aaa(p3_va,h1_oa,h2_oa) +=      t2_aaaa(p3_va,p5_va,h1_oa,h2_oa) * i_10_a(p5_va) )
      ( i0_bab(p3_vb,h1_oa,h2_ob) += -1 * t2_abab(p5_va,p3_vb,h1_oa,h2_ob) * i_10_a(p5_va) );
    // clang-format on
  }
  sch.deallocate(i_6_aaa, i_6_bab, i_10_a, i_11_aaa, i_11_bab, i_11_bba, i0_temp_aaa, i0_temp_bab,
                 i0_temp_bba, i_6_temp_aaa, i_6_temp_bab, i_6_temp_bba);
  // if(debug) sch.execute();
}

template<typename T>
void gfccsd_x2_b(/* ExecutionContext& ec, */
                 Scheduler& sch, const TiledIndexSpace& MO, Tensor<std::complex<T>>& i0_bbb,
                 Tensor<std::complex<T>>& i0_aba, const Tensor<T>& t1_a, const Tensor<T>& t1_b,
                 const Tensor<T>& t2_aaaa, const Tensor<T>& t2_bbbb, const Tensor<T>& t2_abab,
                 const Tensor<std::complex<T>>& x1_b, const Tensor<std::complex<T>>& x2_bbb,
                 const Tensor<std::complex<T>>& x2_aba, const Tensor<T>& f1,
                 const Tensor<T>& ix2_1_bbbb, const Tensor<T>& ix2_1_baba, const Tensor<T>& ix2_2_a,
                 const Tensor<T>& ix2_2_b, const Tensor<T>& ix2_3_a, const Tensor<T>& ix2_3_b,
                 const Tensor<T>& ix2_4_bbbb, const Tensor<T>& ix2_4_abab,
                 const Tensor<T>& ix2_5_aaaa, const Tensor<T>& ix2_5_abba,
                 const Tensor<T>& ix2_5_baba, const Tensor<T>& ix2_5_bbbb,
                 const Tensor<T>& ix2_5_baab, const Tensor<T>& ix2_6_2_a,
                 const Tensor<T>& ix2_6_2_b, const Tensor<T>& ix2_6_3_aaaa,
                 const Tensor<T>& ix2_6_3_abba, const Tensor<T>& ix2_6_3_baba,
                 const Tensor<T>& ix2_6_3_bbbb, const Tensor<T>& ix2_6_3_baab,
                 const Tensor<T>& v2ijab_aaaa, const Tensor<T>& v2ijab_abab,
                 const Tensor<T>& v2ijab_bbbb, const TiledIndexSpace& gf_tis, bool has_tis,
                 bool debug = false) {
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

  auto [p3_va, p4_va, p5_va, p8_va, p9_va]                = v_alpha.labels<5>("all");
  auto [p3_vb, p4_vb, p5_vb, p8_vb, p9_vb]                = v_beta.labels<5>("all");
  auto [h1_oa, h2_oa, h6_oa, h7_oa, h8_oa, h9_oa, h10_oa] = o_alpha.labels<7>("all");
  auto [h1_ob, h2_ob, h6_ob, h7_ob, h8_ob, h9_ob, h10_ob] = o_beta.labels<7>("all");

  ComplexTensor i_6_bbb;
  ComplexTensor i_6_aba;
  ComplexTensor i_10_b;
  ComplexTensor i_11_bbb;
  ComplexTensor i_11_aba;
  ComplexTensor i_11_aab;
  ComplexTensor i0_temp_bbb;
  ComplexTensor i0_temp_aba;
  ComplexTensor i0_temp_aab;
  ComplexTensor i_6_temp_bbb;
  ComplexTensor i_6_temp_aba;
  ComplexTensor i_6_temp_aab;

  if(has_tis) {
    i_6_bbb      = {o_beta, o_beta, o_beta, gf_tis};
    i_6_aba      = {o_alpha, o_beta, o_alpha, gf_tis};
    i_10_b       = {v_beta, gf_tis};
    i_11_bbb     = {o_beta, o_beta, v_beta, gf_tis};
    i_11_aba     = {o_alpha, o_beta, v_alpha, gf_tis};
    i_11_aab     = {o_alpha, o_alpha, v_beta, gf_tis};
    i0_temp_bbb  = {v_beta, o_beta, o_beta, gf_tis};
    i0_temp_aba  = {v_alpha, o_beta, o_alpha, gf_tis};
    i0_temp_aab  = {v_alpha, o_alpha, o_beta, gf_tis};
    i_6_temp_bbb = {o_beta, o_beta, o_beta, gf_tis};
    i_6_temp_aba = {o_alpha, o_beta, o_alpha, gf_tis};
    i_6_temp_aab = {o_alpha, o_alpha, o_beta, gf_tis};
  }
  else {
    i_6_bbb      = {o_beta, o_beta, o_beta};
    i_6_aba      = {o_alpha, o_beta, o_alpha};
    i_10_b       = {v_beta};
    i_11_bbb     = {o_beta, o_beta, v_beta};
    i_11_aba     = {o_alpha, o_beta, v_alpha};
    i_11_aab     = {o_alpha, o_alpha, v_beta};
    i0_temp_bbb  = {v_beta, o_beta, o_beta};
    i0_temp_aba  = {v_alpha, o_beta, o_alpha};
    i0_temp_aab  = {v_alpha, o_alpha, o_beta};
    i_6_temp_bbb = {o_beta, o_beta, o_beta};
    i_6_temp_aba = {o_alpha, o_beta, o_alpha};
    i_6_temp_aab = {o_alpha, o_alpha, o_beta};
  }
  //  sch
  //    .allocate(i_6,i_10,i_11,i0_temp,i_6_temp)
  //    ( i0(p4,h1,h2)                  =  0                                               )
  //    ( i0(p4,h1,h2)                 +=         x1(h9)        * ix2_1(h9,p4,h1,h2)       )
  //    ( i0_temp(p3,h1,h2)             =         x2(p3,h1,h8)  * ix2_2(h8,h2)             )
  //    ( i0(p3,h1,h2)                 += -1    * i0_temp(p3,h1,h2)                        )
  //    ( i0(p3,h2,h1)                 +=         i0_temp(p3,h1,h2)                        )
  //    ( i0(p4,h1,h2)                 +=         x2(p8,h1,h2)  * ix2_3(p4,p8)             )
  //    ( i0(p3,h1,h2)                 +=  0.5  * x2(p3,h9,h10) * ix2_4(h9,h10,h1,h2)      ) //O4V
  //    ( i0_temp(p4,h1,h2)             =         x2(p8,h1,h7)  * ix2_5(h7,p4,h2,p8)       ) //O3V2
  //    ( i0(p4,h1,h2)                 += -1    * i0_temp(p4,h1,h2)                        )
  //    ( i0(p4,h2,h1)                 +=         i0_temp(p4,h1,h2)                        )
  //    (   i_6(h10,h1,h2)              = -1    * x1(h8)        * ix2_6_1(h8,h10,h1,h2)    )
  //    (   i_6(h10,h1,h2)             +=         x2(p5,h1,h2)  * ix2_6_2(h10,p5)          )
  //    (   i_6_temp(h10,h2,h1)         =         x2(p9,h1,h8)  * ix2_6_3(h8,h10,h2,p9)    ) //O4V
  //    (   i_6(h10,h1,h2)             += -1    * i_6_temp(h10,h1,h2)                      )
  //    (   i_6(h10,h2,h1)             +=         i_6_temp(h10,h1,h2)                      )
  //    ( i0(p3,h1,h2)                  +=         d_t1(p3,h10)       * i_6(h10,h1,h2)       )

  //    (   i_10(p5)                    =  0.5  * x2(p8,h6,h7)  * v2(h6,h7,p5,p8)          )
  //    ( i0(p3,h1,h2)                 +=  1.0  * d_t2(p3,p5,h1,h2)  * i_10(p5)              )

  //    (   i_11(h6,h1,p5)              =         x2(p8,h1,h7)  * v2(h6,h7,p5,p8)          ) //O3V2
  //    ( i0_temp(p3,h2,h1)             =         d_t2(p3,p5,h1,h6)  * i_11(h6,h2,p5)        )
  //    //O3V2 ( i0(p3,h1,h2)                 +=         i0_temp(p3,h1,h2)                        )
  //    ( i0(p3,h2,h1)                 += -1    * i0_temp(p3,h1,h2)                        )
  //    .deallocate(i_6,i_10,i_11,i0_temp,i_6_temp);//.execute();

  sch.allocate(i_6_bbb, i_6_aba, i_10_b, i_11_bbb, i_11_aba, i_11_aab, i0_temp_bbb, i0_temp_aba,
               i0_temp_aab, i_6_temp_bbb, i_6_temp_aba, i_6_temp_aab);

  if(has_tis) {
    // clang-format off
    sch( i0_aba(p4_va,h1_ob,h2_oa,u1)    =  0 )
      ( i0_bbb(p4_vb,h1_ob,h2_ob,u1)    =  0 )

      ( i0_aba(p4_va,h1_ob,h2_oa,u1)      +=  x1_b(h9_ob,u1) * ix2_1_baba(h9_ob,p4_va,h1_ob,h2_oa) )
      ( i0_bbb(p4_vb,h1_ob,h2_ob,u1)      +=  x1_b(h9_ob,u1) * ix2_1_bbbb(h9_ob,p4_vb,h1_ob,h2_ob) )

      ( i0_temp_bbb(p3_vb,h1_ob,h2_ob,u1)  =       x2_bbb(p3_vb,h1_ob,h8_ob,u1) * ix2_2_b(h8_ob,h2_ob) )
      ( i0_temp_aba(p3_va,h1_ob,h2_oa,u1)  =       x2_aba(p3_va,h1_ob,h8_oa,u1) * ix2_2_a(h8_oa,h2_oa) )
      ( i0_temp_aab(p3_va,h1_oa,h2_ob,u1)  =  -1 * x2_aba(p3_va,h8_ob,h1_oa,u1) * ix2_2_b(h8_ob,h2_ob) )

      ( i0_temp_bbb(p4_vb,h1_ob,h2_ob,u1) +=       x2_bbb(p8_vb,h1_ob,h7_ob,u1) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) ) //O3V2
      ( i0_temp_bbb(p4_vb,h1_ob,h2_ob,u1) +=       x2_aba(p8_va,h1_ob,h7_oa,u1) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_aba(p4_va,h1_ob,h2_oa,u1) +=       x2_aba(p8_va,h1_ob,h7_oa,u1) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aba(p4_va,h1_ob,h2_oa,u1) +=       x2_bbb(p8_vb,h1_ob,h7_ob,u1) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_aab(p4_va,h1_oa,h2_ob,u1) +=  -1 * x2_aba(p8_va,h7_ob,h1_oa,u1) * ix2_5_baba(h7_ob,p4_va,h2_ob,p8_va) )

      (   i_11_aba(h6_oa,h1_ob,p5_va,u1)   =  x2_aba(p8_va,h1_ob,h7_oa,u1) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aba(h6_oa,h1_ob,p5_va,u1)  +=  x2_bbb(p8_vb,h1_ob,h7_ob,u1) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bbb(h6_ob,h1_ob,p5_vb,u1)   =  x2_bbb(p8_vb,h1_ob,h7_ob,u1) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bbb(h6_ob,h1_ob,p5_vb,u1)  +=  x2_aba(p8_va,h1_ob,h7_oa,u1) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_aab(h6_oa,h1_oa,p5_vb,u1)   =  x2_aba(p8_va,h7_ob,h1_oa,u1) * v2ijab_abab(h6_oa,h7_ob,p8_va,p5_vb) )

      ( i0_temp_bbb(p3_vb,h2_ob,h1_ob,u1) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bbb(h6_ob,h2_ob,p5_vb,u1) )
      ( i0_temp_bbb(p3_vb,h2_ob,h1_ob,u1) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aba(h6_oa,h2_ob,p5_va,u1) ) 
      ( i0_temp_aba(p3_va,h2_ob,h1_oa,u1) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aba(h6_oa,h2_ob,p5_va,u1) )
      ( i0_temp_aba(p3_va,h2_ob,h1_oa,u1) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bbb(h6_ob,h2_ob,p5_vb,u1) )
      ( i0_temp_aab(p3_va,h2_oa,h1_ob,u1) +=      t2_abab(p3_va,p5_vb,h6_oa,h1_ob) * i_11_aab(h6_oa,h2_oa,p5_vb,u1) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob,u1) += -1 * i0_temp_bbb(p3_vb,h1_ob,h2_ob,u1) )
      ( i0_bbb(p3_vb,h2_ob,h1_ob,u1) +=      i0_temp_bbb(p3_vb,h1_ob,h2_ob,u1) )
      ( i0_aba(p3_va,h1_ob,h2_oa,u1) += -1 * i0_temp_aba(p3_va,h1_ob,h2_oa,u1) )
      ( i0_aba(p3_va,h2_ob,h1_oa,u1) +=      i0_temp_aab(p3_va,h1_oa,h2_ob,u1) )

      ( i0_bbb(p4_vb,h1_ob,h2_ob,u1) +=  x2_bbb(p8_vb,h1_ob,h2_ob,u1) * ix2_3_b(p4_vb,p8_vb) )
      ( i0_aba(p4_va,h1_ob,h2_oa,u1) +=  x2_aba(p8_va,h1_ob,h2_oa,u1) * ix2_3_a(p4_va,p8_va) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob,u1) +=  0.5 * x2_bbb(p3_vb,h9_ob,h10_ob,u1) * ix2_4_bbbb(h9_ob,h10_ob,h1_ob,h2_ob) ) 
      ( i0_aba(p3_va,h1_ob,h2_oa,u1) +=        x2_aba(p3_va,h9_ob,h10_oa,u1) * ix2_4_abab(h10_oa,h9_ob,h2_oa,h1_ob) )

      (   i_6_aba(h10_oa,h1_ob,h2_oa,u1)  = -1 * x1_b(h8_ob,u1) * ix2_4_abab(h10_oa,h8_ob,h2_oa,h1_ob) )
      (   i_6_bbb(h10_ob,h1_ob,h2_ob,u1)  = -1 * x1_b(h8_ob,u1) * ix2_4_bbbb(h8_ob,h10_ob,h1_ob,h2_ob) )

      (   i_6_aba(h10_oa,h1_ob,h2_oa,u1) +=  x2_aba(p5_va,h1_ob,h2_oa,u1) * ix2_6_2_a(h10_oa,p5_va) )
      (   i_6_bbb(h10_ob,h1_ob,h2_ob,u1) +=  x2_bbb(p5_vb,h1_ob,h2_ob,u1) * ix2_6_2_b(h10_ob,p5_vb) )

      (   i_6_temp_bbb(h10_ob,h1_ob,h2_ob,u1)  =  x2_bbb(p9_vb,h2_ob,h8_ob,u1) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
      (   i_6_temp_bbb(h10_ob,h1_ob,h2_ob,u1) +=  x2_aba(p9_va,h2_ob,h8_oa,u1) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
      (   i_6_temp_aba(h10_oa,h1_ob,h2_oa,u1)  =  -1 * x2_aba(p9_va,h8_ob,h2_oa,u1) * ix2_6_3_baba(h8_ob,h10_oa,h1_ob,p9_va) )
      (   i_6_temp_aab(h10_oa,h1_oa,h2_ob,u1)  =       x2_aba(p9_va,h2_ob,h8_oa,u1) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
      (   i_6_temp_aab(h10_oa,h1_oa,h2_ob,u1) +=       x2_bbb(p9_vb,h2_ob,h8_ob,u1) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) )

      (   i_6_bbb(h10_ob,h1_ob,h2_ob,u1) += -1 * i_6_temp_bbb(h10_ob,h1_ob,h2_ob,u1) )
      (   i_6_bbb(h10_ob,h2_ob,h1_ob,u1) +=      i_6_temp_bbb(h10_ob,h1_ob,h2_ob,u1) )
      (   i_6_aba(h10_oa,h1_ob,h2_oa,u1) += -1 * i_6_temp_aba(h10_oa,h1_ob,h2_oa,u1) )
      (   i_6_aba(h10_oa,h2_ob,h1_oa,u1) +=      i_6_temp_aab(h10_oa,h1_oa,h2_ob,u1) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob,u1)  +=  t1_b(p3_vb,h10_ob) * i_6_bbb(h10_ob,h1_ob,h2_ob,u1) )
      ( i0_aba(p3_va,h1_ob,h2_oa,u1)  +=  t1_a(p3_va,h10_oa) * i_6_aba(h10_oa,h1_ob,h2_oa,u1) )

      (   i_10_b(p5_vb,u1)  =  0.5 * x2_bbb(p8_vb,h6_ob,h7_ob,u1) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_10_b(p5_vb,u1) +=        x2_aba(p8_va,h6_ob,h7_oa,u1) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      ( i0_bbb(p3_vb,h1_ob,h2_ob,u1) +=      t2_bbbb(p3_vb,p5_vb,h1_ob,h2_ob) * i_10_b(p5_vb,u1) )
      ( i0_aba(p3_va,h1_ob,h2_oa,u1) += -1 * t2_abab(p3_va,p5_vb,h2_oa,h1_ob) * i_10_b(p5_vb,u1) );
    // clang-format on
  }
  else {
    // clang-format off
    sch( i0_aba(p4_va,h1_ob,h2_oa)    =  0 )
      ( i0_bbb(p4_vb,h1_ob,h2_ob)    =  0 )

      ( i0_aba(p4_va,h1_ob,h2_oa)      +=  x1_b(h9_ob) * ix2_1_baba(h9_ob,p4_va,h1_ob,h2_oa) )
      ( i0_bbb(p4_vb,h1_ob,h2_ob)      +=  x1_b(h9_ob) * ix2_1_bbbb(h9_ob,p4_vb,h1_ob,h2_ob) )

      ( i0_temp_bbb(p3_vb,h1_ob,h2_ob)  =       x2_bbb(p3_vb,h1_ob,h8_ob) * ix2_2_b(h8_ob,h2_ob) )
      ( i0_temp_aba(p3_va,h1_ob,h2_oa)  =       x2_aba(p3_va,h1_ob,h8_oa) * ix2_2_a(h8_oa,h2_oa) )
      ( i0_temp_aab(p3_va,h1_oa,h2_ob)  =  -1 * x2_aba(p3_va,h8_ob,h1_oa) * ix2_2_b(h8_ob,h2_ob) )

      ( i0_temp_bbb(p4_vb,h1_ob,h2_ob) +=       x2_bbb(p8_vb,h1_ob,h7_ob) * ix2_5_bbbb(h7_ob,p4_vb,h2_ob,p8_vb) ) //O3V2
      ( i0_temp_bbb(p4_vb,h1_ob,h2_ob) +=       x2_aba(p8_va,h1_ob,h7_oa) * ix2_5_abba(h7_oa,p4_vb,h2_ob,p8_va) )
      ( i0_temp_aba(p4_va,h1_ob,h2_oa) +=       x2_aba(p8_va,h1_ob,h7_oa) * ix2_5_aaaa(h7_oa,p4_va,h2_oa,p8_va) )
      ( i0_temp_aba(p4_va,h1_ob,h2_oa) +=       x2_bbb(p8_vb,h1_ob,h7_ob) * ix2_5_baab(h7_ob,p4_va,h2_oa,p8_vb) )
      ( i0_temp_aab(p4_va,h1_oa,h2_ob) +=  -1 * x2_aba(p8_va,h7_ob,h1_oa) * ix2_5_baba(h7_ob,p4_va,h2_ob,p8_va) )

      (   i_11_aba(h6_oa,h1_ob,p5_va)   =  x2_aba(p8_va,h1_ob,h7_oa) * v2ijab_aaaa(h6_oa,h7_oa,p5_va,p8_va) )
      (   i_11_aba(h6_oa,h1_ob,p5_va)  +=  x2_bbb(p8_vb,h1_ob,h7_ob) * v2ijab_abab(h6_oa,h7_ob,p5_va,p8_vb) )
      (   i_11_bbb(h6_ob,h1_ob,p5_vb)   =  x2_bbb(p8_vb,h1_ob,h7_ob) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_11_bbb(h6_ob,h1_ob,p5_vb)  +=  x2_aba(p8_va,h1_ob,h7_oa) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      (   i_11_aab(h6_oa,h1_oa,p5_vb)   =  x2_aba(p8_va,h7_ob,h1_oa) * v2ijab_abab(h6_oa,h7_ob,p8_va,p5_vb) )

      ( i0_temp_bbb(p3_vb,h2_ob,h1_ob) += -1 * t2_bbbb(p3_vb,p5_vb,h1_ob,h6_ob) * i_11_bbb(h6_ob,h2_ob,p5_vb) )
      ( i0_temp_bbb(p3_vb,h2_ob,h1_ob) += -1 * t2_abab(p5_va,p3_vb,h6_oa,h1_ob) * i_11_aba(h6_oa,h2_ob,p5_va) ) 
      ( i0_temp_aba(p3_va,h2_ob,h1_oa) += -1 * t2_aaaa(p3_va,p5_va,h1_oa,h6_oa) * i_11_aba(h6_oa,h2_ob,p5_va) )
      ( i0_temp_aba(p3_va,h2_ob,h1_oa) += -1 * t2_abab(p3_va,p5_vb,h1_oa,h6_ob) * i_11_bbb(h6_ob,h2_ob,p5_vb) )
      ( i0_temp_aab(p3_va,h2_oa,h1_ob) +=      t2_abab(p3_va,p5_vb,h6_oa,h1_ob) * i_11_aab(h6_oa,h2_oa,p5_vb) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob) += -1 * i0_temp_bbb(p3_vb,h1_ob,h2_ob) )
      ( i0_bbb(p3_vb,h2_ob,h1_ob) +=      i0_temp_bbb(p3_vb,h1_ob,h2_ob) )
      ( i0_aba(p3_va,h1_ob,h2_oa) += -1 * i0_temp_aba(p3_va,h1_ob,h2_oa) )
      ( i0_aba(p3_va,h2_ob,h1_oa) +=      i0_temp_aab(p3_va,h1_oa,h2_ob) )

      ( i0_bbb(p4_vb,h1_ob,h2_ob) +=  x2_bbb(p8_vb,h1_ob,h2_ob) * ix2_3_b(p4_vb,p8_vb) )
      ( i0_aba(p4_va,h1_ob,h2_oa) +=  x2_aba(p8_va,h1_ob,h2_oa) * ix2_3_a(p4_va,p8_va) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob) +=  0.5 * x2_bbb(p3_vb,h9_ob,h10_ob) * ix2_4_bbbb(h9_ob,h10_ob,h1_ob,h2_ob) ) 
      ( i0_aba(p3_va,h1_ob,h2_oa) +=        x2_aba(p3_va,h9_ob,h10_oa) * ix2_4_abab(h10_oa,h9_ob,h2_oa,h1_ob) )

      (   i_6_aba(h10_oa,h1_ob,h2_oa)  = -1 * x1_b(h8_ob) * ix2_4_abab(h10_oa,h8_ob,h2_oa,h1_ob) )
      (   i_6_bbb(h10_ob,h1_ob,h2_ob)  = -1 * x1_b(h8_ob) * ix2_4_bbbb(h8_ob,h10_ob,h1_ob,h2_ob) )

      (   i_6_aba(h10_oa,h1_ob,h2_oa) +=  x2_aba(p5_va,h1_ob,h2_oa) * ix2_6_2_a(h10_oa,p5_va) )
      (   i_6_bbb(h10_ob,h1_ob,h2_ob) +=  x2_bbb(p5_vb,h1_ob,h2_ob) * ix2_6_2_b(h10_ob,p5_vb) )

      (   i_6_temp_bbb(h10_ob,h1_ob,h2_ob)  =  x2_bbb(p9_vb,h2_ob,h8_ob) * ix2_6_3_bbbb(h8_ob,h10_ob,h1_ob,p9_vb) )
      (   i_6_temp_bbb(h10_ob,h1_ob,h2_ob) +=  x2_aba(p9_va,h2_ob,h8_oa) * ix2_6_3_abba(h8_oa,h10_ob,h1_ob,p9_va) )
  
      (   i_6_temp_aba(h10_oa,h1_ob,h2_oa)  =  -1 * x2_aba(p9_va,h8_ob,h2_oa) * ix2_6_3_baba(h8_ob,h10_oa,h1_ob,p9_va) )
      (   i_6_temp_aab(h10_oa,h1_oa,h2_ob)  =       x2_aba(p9_va,h2_ob,h8_oa) * ix2_6_3_aaaa(h8_oa,h10_oa,h1_oa,p9_va) )
      (   i_6_temp_aab(h10_oa,h1_oa,h2_ob) +=       x2_bbb(p9_vb,h2_ob,h8_ob) * ix2_6_3_baab(h8_ob,h10_oa,h1_oa,p9_vb) )

      (   i_6_bbb(h10_ob,h1_ob,h2_ob) += -1 * i_6_temp_bbb(h10_ob,h1_ob,h2_ob) )
      (   i_6_bbb(h10_ob,h2_ob,h1_ob) +=      i_6_temp_bbb(h10_ob,h1_ob,h2_ob) )
      (   i_6_aba(h10_oa,h1_ob,h2_oa) += -1 * i_6_temp_aba(h10_oa,h1_ob,h2_oa) )
      (   i_6_aba(h10_oa,h2_ob,h1_oa) +=      i_6_temp_aab(h10_oa,h1_oa,h2_ob) )

      ( i0_bbb(p3_vb,h1_ob,h2_ob)  +=  t1_b(p3_vb,h10_ob) * i_6_bbb(h10_ob,h1_ob,h2_ob) )
      ( i0_aba(p3_va,h1_ob,h2_oa)  +=  t1_a(p3_va,h10_oa) * i_6_aba(h10_oa,h1_ob,h2_oa) )

      (   i_10_b(p5_vb)  =  0.5 * x2_bbb(p8_vb,h6_ob,h7_ob) * v2ijab_bbbb(h6_ob,h7_ob,p5_vb,p8_vb) )
      (   i_10_b(p5_vb) +=        x2_aba(p8_va,h6_ob,h7_oa) * v2ijab_abab(h7_oa,h6_ob,p8_va,p5_vb) )
      ( i0_bbb(p3_vb,h1_ob,h2_ob) +=      t2_bbbb(p3_vb,p5_vb,h1_ob,h2_ob) * i_10_b(p5_vb) )
      ( i0_aba(p3_va,h1_ob,h2_oa) += -1 * t2_abab(p3_va,p5_vb,h2_oa,h1_ob) * i_10_b(p5_vb) );
    // clang-format on
  }
  sch.deallocate(i_6_bbb, i_6_aba, i_10_b, i_11_bbb, i_11_aba, i_11_aab, i0_temp_bbb, i0_temp_aba,
                 i0_temp_aab, i_6_temp_bbb, i_6_temp_aba, i_6_temp_aab);
  // if(debug) sch.execute();
}
