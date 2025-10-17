/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/rteom/rt_eom_cd_ccsd.hpp"
#include "exachem/cc/ccsd/ccsd_util.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

using namespace tamm;
namespace exachem::rteom_cc::ccsd {

using CCEType = std::complex<double>;

template<typename T>
void RT_EOM_CD_CCSD<T>::ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO,
                                  const TiledIndexSpace& CI, Tensor<T>& de, CCSE_Tensors<T>& t1,
                                  CCSE_Tensors<T>& t2, std::vector<CCSE_Tensors<T>>& f1_se,
                                  std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind]                = CI.labels<1>("all");
  const auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  const auto [p1_vb, p2_vb, p3_vb] = v_beta.labels<3>("all");
  const auto [h3_oa, h4_oa, h6_oa] = o_alpha.labels<3>("all");
  const auto [h3_ob, h4_ob, h6_ob] = o_beta.labels<3>("all");

  Tensor<T> t1_aa   = t1("aa");
  Tensor<T> t1_bb   = t1("bb");
  Tensor<T> t2_aaaa = t2("aaaa");
  Tensor<T> t2_abab = t2("abab");
  Tensor<T> t2_bbbb = t2("bbbb");

  // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
  // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
  (_a01V(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h4_oa, p3_va, cind), 
  "_a01V(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h4_oa, p3_va, cind)")
  (_a02("aa")(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h6_oa, p3_va, cind), 
  "_a02( aa )(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h6_oa, p3_va, cind)")
  (_a03("aa")(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
  "_a03( aa )(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
  (_a03("aa")(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
  "_a03( aa )(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
  (_a01V(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h4_ob, p3_vb, cind), 
  "_a01V(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h4_ob, p3_vb, cind)")
  (_a02("bb")(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h6_ob, p3_vb, cind), 
  "_a02( bb )(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h6_ob, p3_vb, cind)")
  (_a03("bb")(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
  "_a03( bb )(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
  (_a03("bb")(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
  "_a03( bb )(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
  (de()                            =  0.5 * _a01V() * _a01V(), 
  "de()                            =  0.5 * _a01V() * _a01V()")
  (de()                           += -0.5 * _a02("aa")(h4_oa, h6_oa, cind) * _a02("aa")(h6_oa, h4_oa, cind), 
  "de()                           += -0.5 * _a02( aa )(h4_oa, h6_oa, cind) * _a02( aa )(h6_oa, h4_oa, cind)")
  (de()                           += -0.5 * _a02("bb")(h4_ob, h6_ob, cind) * _a02("bb")(h6_ob, h4_ob, cind), 
  "de()                           += -0.5 * _a02( bb )(h4_ob, h6_ob, cind) * _a02( bb )(h6_ob, h4_ob, cind)")
  (de()                           +=  0.5 * _a03("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h4_oa, p1_va, cind), 
  "de()                           +=  0.5 * _a03( aa )(h4_oa, p1_va, cind) * chol3d_ov( aa )(h4_oa, p1_va, cind)")
  (de()                           +=  0.5 * _a03("bb")(h4_ob, p1_vb, cind) * chol3d_ov("bb")(h4_ob, p1_vb, cind), 
  "de()                           +=  0.5 * _a03( bb )(h4_ob, p1_vb, cind) * chol3d_ov( bb )(h4_ob, p1_vb, cind)")
  (de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov("aa")(h3_oa, p1_va),
  "de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov( aa )(h3_oa, p1_va)") // NEW TERM
  (de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov("bb")(h3_ob, p1_vb),
  "de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov( bb )(h3_ob, p1_vb)") // NEW TERM
  ;
  // clang-format on
}

template<typename T>
void RT_EOM_CD_CCSD<T>::ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO,
                                   const TiledIndexSpace& CI, CCSE_Tensors<T>& r1_vo,
                                   CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                                   std::vector<CCSE_Tensors<T>>& f1_se,
                                   std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind]                       = CI.labels<1>("all");
  const auto [p2]                         = MO.labels<1>("virt");
  const auto [h1]                         = MO.labels<1>("occ");
  const auto [p1_va, p2_va, p3_va]        = v_alpha.labels<3>("all");
  const auto [p1_vb, p2_vb, p3_vb]        = v_beta.labels<3>("all");
  const auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha.labels<4>("all");
  const auto [h1_ob, h2_ob, h3_ob, h7_ob] = o_beta.labels<4>("all");

  Tensor<T> i0_aa = r1_vo("aa");
  Tensor<T> i0_bb = r1_vo("bb");

  Tensor<T> t1_aa   = t1("aa");
  Tensor<T> t1_bb   = t1("bb");
  Tensor<T> t2_aaaa = t2("aaaa");
  Tensor<T> t2_abab = t2("abab");
  Tensor<T> t2_bbbb = t2("bbbb");

  // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
  // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vo     = f1_se[2];
  auto f1_vv     = f1_se[3];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vo = chol3d_se[2];
  auto chol3d_vv = chol3d_se[3];

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1_vo("aa")(p2_va, h1_oa),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1_vo( aa )(p2_va, h1_oa)")
    (i0_bb(p2_vb, h1_ob)             =  1.0 * f1_vo("bb")(p2_vb, h1_ob),
    "i0_bb(p2_vb, h1_ob)             =  1.0 * f1_vo( bb )(p2_vb, h1_ob)")
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind), 
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a01("bb")(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov("bb")(h2_ob, p1_vb, cind), 
    "_a01( bb )(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov( bb )(h2_ob, p1_vb, cind)")                 // ovm
    (_a02V(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov("aa")(h3_oa, p3_va, cind), 
    "_a02V(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov( aa )(h3_oa, p3_va, cind)")                 // ovm
    (_a02V(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov("bb")(h3_ob, p3_vb, cind), 
    "_a02V(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov( bb )(h3_ob, p3_vb, cind)")                 // ovm
    (_a06("aa")(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
    "_a06( aa )(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
    (_a06("aa")(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
    "_a06( aa )(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
    (_a06("bb")(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
    "_a06( bb )(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
    (_a06("bb")(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
    "_a06( bb )(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa), 
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("bb")(h2_ob, h1_ob)        = -1.0 * f1_oo("bb")(h2_ob, h1_ob), 
    "_a04( bb )(h2_ob, h1_ob)        = -1.0 * f1_oo( bb )(h2_ob, h1_ob)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind), 
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04("bb")(h2_ob, h1_ob)       +=  1.0 * chol3d_ov("bb")(h2_ob, p1_vb, cind) * _a06("bb")(p1_vb, h1_ob, cind), 
    "_a04( bb )(h2_ob, h1_ob)       +=  1.0 * chol3d_ov( bb )(h2_ob, p1_vb, cind) * _a06( bb )(p1_vb, h1_ob, cind)")   // o2vm
    (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va), 
    "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (_a04("bb")(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov("bb")(h2_ob, p1_vb), 
    "_a04( bb )(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04("bb")(h2_ob, h1_ob), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04( bb )(h2_ob, h1_ob)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo("aa")(p1_va, h2_oa, cind) * _a02V(cind), 
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                      // ovm
    (i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo("bb")(p1_vb, h2_ob, cind) * _a02V(cind), 
    "i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo( bb )(p1_vb, h2_ob, cind) * _a02V(cind)")                      // ovm
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h3_oa, p1_va, cind) * _a01("aa")(h2_oa, h3_oa, cind), 
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h3_oa, p1_va, cind) * _a01( aa )(h2_oa, h3_oa, cind)")   // o2vm
    (_a05("bb")(h2_ob, p1_vb)        = -1.0 * chol3d_ov("bb")(h3_ob, p1_vb, cind) * _a01("bb")(h2_ob, h3_ob, cind), 
    "_a05( bb )(h2_ob, p1_vb)        = -1.0 * chol3d_ov( bb )(h3_ob, p1_vb, cind) * _a01( bb )(h2_ob, h3_ob, cind)")   // o2vm
    (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va), 
    "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (_a05("bb")(h2_ob, p1_vb)       +=  1.0 * f1_ov("bb")(h2_ob, p1_vb), 
    "_a05( bb )(h2_ob, p1_vb)       +=  1.0 * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h2_oa, p1_va), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h2_oa, p1_va)")         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05("aa")(h2_oa, p1_va), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05( aa )(h2_oa, p1_va)")         // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05("bb")(h2_ob, p1_vb), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05( bb )(h2_ob, p1_vb)")         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05("bb")(h2_ob, p1_vb), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05( bb )(h2_ob, p1_vb)")         // o2v
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind), 
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
    (i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv("bb")(p2_vb, p1_vb, cind) * _a06("bb")(p1_vb, h1_ob, cind), 
    "i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv( bb )(p2_vb, p1_vb, cind) * _a06( bb )(p1_vb, h1_ob, cind)")   // ov2m
    (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind), 
    "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (_a06("bb")(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv("bb")(p2_vb, p1_vb, cind), 
    "_a06( bb )(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv( bb )(p2_vb, p1_vb, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind), 
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (i0_bb(p1_vb, h2_ob)            += -1.0 * _a06("bb")(p1_vb, h2_ob, cind) * _a02V(cind), 
    "i0_bb(p1_vb, h2_ob)            += -1.0 * _a06( bb )(p1_vb, h2_ob, cind) * _a02V(cind)")                           // ovm
    (_a06("aa")(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V(cind), 
    "_a06( aa )(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V(cind)")                                      // ovm
    (_a06("bb")(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V(cind), 
    "_a06( bb )(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V(cind)")                                      // ovm
    (_a06("aa")(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h3_oa, cind), 
    "_a06( aa )(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h3_oa, cind)")                   // o2vm
    (_a06("bb")(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01("bb")(h2_ob, h3_ob, cind), 
    "_a06( bb )(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01( bb )(h2_ob, h3_ob, cind)")                   // o2vm
    (_a01("aa")(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h3_oa, h1_oa, cind), 
    "_a01( aa )(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h3_oa, h1_oa, cind)")                                    // o2m
    (_a01("bb")(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo("bb")(h3_ob, h1_ob, cind), 
    "_a01( bb )(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo( bb )(h3_ob, h1_ob, cind)")                                    // o2m        
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h3_oa, h1_oa, cind) * _a06("aa")(p2_va, h3_oa, cind), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h3_oa, h1_oa, cind) * _a06( aa )(p2_va, h3_oa, cind)")        // o2vm
  //  (i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo("aa")(h7_oa, h1_oa), 
  //  "i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo( aa )(h7_oa, h1_oa)") // MOVED ABOVE         // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv("aa")(p2_va, p3_va), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv( aa )(p2_va, p3_va)")                        // ov2
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01("bb")(h3_ob, h1_ob, cind) * _a06("bb")(p2_vb, h3_ob, cind), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01( bb )(h3_ob, h1_ob, cind) * _a06( bb )(p2_vb, h3_ob, cind)")        // o2vm
  //  (i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo("bb")(h7_ob, h1_ob), 
  //  "i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo( bb )(h7_ob, h1_ob)") // MOVED ABOVE         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv("bb")(p2_vb, p3_vb), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv( bb )(p2_vb, p3_vb)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void RT_EOM_CD_CCSD<T>::ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO,
                                   const TiledIndexSpace& CI, CCSE_Tensors<T>& r2,
                                   CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                                   std::vector<CCSE_Tensors<T>>& f1_se,
                                   std::vector<CCSE_Tensors<T>>& chol3d_se,
                                   CCSE_Tensors<T>&              i0tmp) {
  const auto [cind]                                     = CI.labels<1>("all");
  const auto [p3, p4]                                   = MO.labels<2>("virt");
  const auto [h1, h2]                                   = MO.labels<2>("occ");
  const auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha.labels<6>("all");
  const auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] = v_beta.labels<6>("all");
  const auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha.labels<6>("all");
  const auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] = o_beta.labels<6>("all");

  Tensor<T> i0_aaaa = r2("aaaa");
  Tensor<T> i0_abab = r2("abab");
  Tensor<T> i0_bbbb = r2("bbbb");

  const Tensor<T> t1_aa   = t1("aa");
  const Tensor<T> t1_bb   = t1("bb");
  const Tensor<T> t2_aaaa = t2("aaaa");
  const Tensor<T> t2_abab = t2("abab");
  const Tensor<T> t2_bbbb = t2("bbbb");

  // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
  // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vo     = f1_se[2];
  auto f1_vv     = f1_se[3];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vo = chol3d_se[2];
  auto chol3d_vv = chol3d_se[3];

  // clang-format off
  sch 
    (_a017("aa")(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
    "_a017( aa )(p3_va, h2_oa, cind)            = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (_a017("bb")(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
    "_a017( bb )(p3_vb, h2_ob, cind)            = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a017("bb")(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
    "_a017( bb )(p3_vb, h2_ob, cind)           += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (_a017("aa")(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
    "_a017( aa )(p3_va, h2_oa, cind)           += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a006("aa")(h4_oa, h1_oa)                  = -1.0   * chol3d_ov("aa")(h4_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind), 
    "_a006( aa )(h4_oa, h1_oa)                  = -1.0   * chol3d_ov( aa )(h4_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
    (_a006("bb")(h4_ob, h1_ob)                  = -1.0   * chol3d_ov("bb")(h4_ob, p2_vb, cind) * _a017("bb")(p2_vb, h1_ob, cind), 
    "_a006( bb )(h4_ob, h1_ob)                  = -1.0   * chol3d_ov( bb )(h4_ob, p2_vb, cind) * _a017( bb )(p2_vb, h1_ob, cind)")
    (_a007V(cind)                               =  1.0   * chol3d_ov("aa")(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa), 
    "_a007V(cind)                               =  1.0   * chol3d_ov( aa )(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa)")
    (_a007V(cind)                              +=  1.0   * chol3d_ov("bb")(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob), 
    "_a007V(cind)                              +=  1.0   * chol3d_ov( bb )(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob)")
    (_a009("aa")(h3_oa, h2_oa, cind)            =  1.0   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa), 
    "_a009( aa )(h3_oa, h2_oa, cind)            =  1.0   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a009("bb")(h3_ob, h2_ob, cind)            =  1.0   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob), 
    "_a009( bb )(h3_ob, h2_ob, cind)            =  1.0   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob)")
    (_a021("aa")(p3_va, p1_va, cind)            = -0.5   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa), 
    "_a021( aa )(p3_va, p1_va, cind)            = -0.5   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa)")
    (_a021("bb")(p3_vb, p1_vb, cind)            = -0.5   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob), 
    "_a021( bb )(p3_vb, p1_vb, cind)            = -0.5   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob)")
    (_a021("aa")(p3_va, p1_va, cind)           +=  0.5   * chol3d_vv("aa")(p3_va, p1_va, cind), 
    "_a021( aa )(p3_va, p1_va, cind)           +=  0.5   * chol3d_vv( aa )(p3_va, p1_va, cind)")
    (_a021("bb")(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_vv("bb")(p3_vb, p1_vb, cind), 
    "_a021( bb )(p3_vb, p1_vb, cind)           +=  0.5   * chol3d_vv( bb )(p3_vb, p1_vb, cind)")
    (_a017("aa")(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021("aa")(p3_va, p2_va, cind), 
    "_a017( aa )(p3_va, h2_oa, cind)           += -2.0   * t1_aa(p2_va, h2_oa) * _a021( aa )(p3_va, p2_va, cind)")
    (_a017("bb")(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021("bb")(p3_vb, p2_vb, cind), 
    "_a017( bb )(p3_vb, h2_ob, cind)           += -2.0   * t1_bb(p2_vb, h2_ob) * _a021( bb )(p3_vb, p2_vb, cind)")
    (_a008("aa")(h3_oa, h1_oa, cind)            =  1.0   * _a009("aa")(h3_oa, h1_oa, cind), 
    "_a008( aa )(h3_oa, h1_oa, cind)            =  1.0   * _a009( aa )(h3_oa, h1_oa, cind)")
    (_a008("bb")(h3_ob, h1_ob, cind)            =  1.0   * _a009("bb")(h3_ob, h1_ob, cind), 
    "_a008( bb )(h3_ob, h1_ob, cind)            =  1.0   * _a009( bb )(h3_ob, h1_ob, cind)")
    (_a009("aa")(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_oo("aa")(h3_oa, h1_oa, cind), 
    "_a009( aa )(h3_oa, h1_oa, cind)           +=  1.0   * chol3d_oo( aa )(h3_oa, h1_oa, cind)")
    (_a009("bb")(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_oo("bb")(h3_ob, h1_ob, cind), 
    "_a009( bb )(h3_ob, h1_ob, cind)           +=  1.0   * chol3d_oo( bb )(h3_ob, h1_ob, cind)")

    (_a001("aa")(p4_va, p2_va)                  = -2.0   * _a021("aa")(p4_va, p2_va, cind) * _a007V(cind), 
    "_a001( aa )(p4_va, p2_va)                  = -2.0   * _a021( aa )(p4_va, p2_va, cind) * _a007V(cind)")
    (_a001("bb")(p4_vb, p2_vb)                  = -2.0   * _a021("bb")(p4_vb, p2_vb, cind) * _a007V(cind), 
    "_a001( bb )(p4_vb, p2_vb)                  = -2.0   * _a021( bb )(p4_vb, p2_vb, cind) * _a007V(cind)")
    (_a001("aa")(p4_va, p2_va)                 += -1.0   * _a017("aa")(p4_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind), 
    "_a001( aa )(p4_va, p2_va)                 += -1.0   * _a017( aa )(p4_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a001("bb")(p4_vb, p2_vb)                 += -1.0   * _a017("bb")(p4_vb, h2_ob, cind) * chol3d_ov("bb")(h2_ob, p2_vb, cind), 
    "_a001( bb )(p4_vb, p2_vb)                 += -1.0   * _a017( bb )(p4_vb, h2_ob, cind) * chol3d_ov( bb )(h2_ob, p2_vb, cind)")
    (_a006("aa")(h4_oa, h1_oa)                 +=  1.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a007V(cind), 
    "_a006( aa )(h4_oa, h1_oa)                 +=  1.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a007V(cind)")
    (_a006("bb")(h4_ob, h1_ob)                 +=  1.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a007V(cind), 
    "_a006( bb )(h4_ob, h1_ob)                 +=  1.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a007V(cind)")
    (_a006("aa")(h4_oa, h1_oa)                 += -1.0   * _a009("aa")(h3_oa, h1_oa, cind) * _a008("aa")(h4_oa, h3_oa, cind), 
    "_a006( aa )(h4_oa, h1_oa)                 += -1.0   * _a009( aa )(h3_oa, h1_oa, cind) * _a008( aa )(h4_oa, h3_oa, cind)")
    (_a006("bb")(h4_ob, h1_ob)                 += -1.0   * _a009("bb")(h3_ob, h1_ob, cind) * _a008("bb")(h4_ob, h3_ob, cind), 
    "_a006( bb )(h4_ob, h1_ob)                 += -1.0   * _a009( bb )(h3_ob, h1_ob, cind) * _a008( bb )(h4_ob, h3_ob, cind)")
    (_a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009("aa")(h4_oa, h1_oa, cind) * _a009("aa")(h3_oa, h2_oa, cind), 
    "_a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009( aa )(h4_oa, h1_oa, cind) * _a009( aa )(h3_oa, h2_oa, cind)") 
    (_a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009("aa")(h4_oa, h1_oa, cind) * _a009("bb")(h3_ob, h2_ob, cind), 
    "_a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009( aa )(h4_oa, h1_oa, cind) * _a009( bb )(h3_ob, h2_ob, cind)")
    (_a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009("bb")(h4_ob, h1_ob, cind) * _a009("bb")(h3_ob, h2_ob, cind), 
    "_a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009( bb )(h4_ob, h1_ob, cind) * _a009( bb )(h3_ob, h2_ob, cind)") 
    (_a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a021("aa")(p4_va, p1_va, cind), 
    "_a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a021( aa )(p4_va, p1_va, cind)")
    (_a020("abab")(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a021("aa")(p4_va, p1_va, cind), 
    "_a020( abab )(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a021( aa )(p4_va, p1_va, cind)")
    (_a020("baba")(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009("aa")(h4_oa, h1_oa, cind) * _a021("bb")(p4_vb, p1_vb, cind), 
    "_a020( baba )(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009( aa )(h4_oa, h1_oa, cind) * _a021( bb )(p4_vb, p1_vb, cind)")
    (_a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009("bb")(h4_ob, h1_ob, cind) * _a021("bb")(p4_vb, p1_vb, cind), 
    "_a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009( bb )(h4_ob, h1_ob, cind) * _a021( bb )(p4_vb, p1_vb, cind)")

    (_a017("aa")(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo("aa")(h3_oa, h2_oa, cind), 
    "_a017( aa )(p3_va, h2_oa, cind)           +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo( aa )(h3_oa, h2_oa, cind)")
    (_a017("bb")(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo("bb")(h3_ob, h2_ob, cind), 
    "_a017( bb )(p3_vb, h2_ob, cind)           +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo( bb )(h3_ob, h2_ob, cind)")
    (_a017("aa")(p3_va, h2_oa, cind)           += -1.0   * chol3d_vo("aa")(p3_va, h2_oa, cind), 
    "_a017( aa )(p3_va, h2_oa, cind)           += -1.0   * chol3d_vo( aa )(p3_va, h2_oa, cind)")
    (_a017("bb")(p3_vb, h2_ob, cind)           += -1.0   * chol3d_vo("bb")(p3_vb, h2_ob, cind), 
    "_a017( bb )(p3_vb, h2_ob, cind)           += -1.0   * chol3d_vo( bb )(p3_vb, h2_ob, cind)")

    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017("aa")(p3_va, h1_oa, cind) * _a017("aa")(p4_va, h2_oa, cind), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017( aa )(p3_va, h1_oa, cind) * _a017( aa )(p4_va, h2_oa, cind)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017("bb")(p3_vb, h1_ob, cind) * _a017("bb")(p4_vb, h2_ob, cind), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017( bb )(p3_vb, h1_ob, cind) * _a017( bb )(p4_vb, h2_ob, cind)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017("aa")(p3_va, h1_oa, cind) * _a017("bb")(p4_vb, h2_ob, cind), 
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017( aa )(p3_va, h1_oa, cind) * _a017( bb )(p4_vb, h2_ob, cind)")

    (_a022("aaaa")(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021("aa")(p3_va,p2_va,cind) * _a021("aa")(p4_va,p1_va,cind), 
    "_a022( aaaa )(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021( aa )(p3_va,p2_va,cind) * _a021( aa )(p4_va,p1_va,cind)")
    (_a022("abab")(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021("aa")(p3_va,p2_va,cind) * _a021("bb")(p4_vb,p1_vb,cind), 
    "_a022( abab )(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021( aa )(p3_va,p2_va,cind) * _a021( bb )(p4_vb,p1_vb,cind)")
    (_a022("bbbb")(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021("bb")(p3_vb,p2_vb,cind) * _a021("bb")(p4_vb,p1_vb,cind), 
    "_a022( bbbb )(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021( bb )(p3_vb,p2_vb,cind) * _a021( bb )(p4_vb,p1_vb,cind)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022("aaaa")(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022( aaaa )(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022("bbbb")(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022( bbbb )(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022("abab")(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022( abab )(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004("aaaa")(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
    "_a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004( aaaa )(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004("abab")(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob), 
    "_a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob) +=  0.25  * _a004( abab )(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)") 
    (_a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004("bbbb")(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
    "_a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob) += -0.125 * _a004( bbbb )(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019("abab")(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob), 
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019( abab )(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob)")
    (_a020("aaaa")(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa), 
    "_a020( aaaa )(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)") 
    (_a020("baab")(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob), 
    "_a020( baab )(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)") 
    (_a020("abba")(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob), 
    "_a020( abba )(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob)")
    (_a020("bbbb")(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob), 
    "_a020( bbbb )(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
    (_a020("baba")(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004("abab")(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob), 
    "_a020( baba )(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004( abab )(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
    (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020("baba")(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob), 
    "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020( baba )(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob)")
    (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020("abab")(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob), 
    "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020( abab )(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob), 
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa), 
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob), 
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob), 
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")

    (_a001("aa")(p4_va, p1_va)                 += -1.0   * f1_vv("aa")(p4_va, p1_va), 
    "_a001( aa )(p4_va, p1_va)                 += -1.0   * f1_vv( aa )(p4_va, p1_va)")
    (_a001("bb")(p4_vb, p1_vb)                 += -1.0   * f1_vv("bb")(p4_vb, p1_vb), 
    "_a001( bb )(p4_vb, p1_vb)                 += -1.0   * f1_vv( bb )(p4_vb, p1_vb)")
    (_a001("aa")(p4_va, p1_va)                 +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va), 
    "_a001( aa )(p4_va, p1_va)                 +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a001("bb")(p4_vb, p1_vb)                 +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov("bb")(h1_ob, p1_vb), 
    "_a001( bb )(p4_vb, p1_vb)                 +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov( bb )(h1_ob, p1_vb)") // NEW TERM
    (_a006("aa")(h9_oa, h1_oa)                 +=  1.0   * f1_oo("aa")(h9_oa, h1_oa), 
    "_a006( aa )(h9_oa, h1_oa)                 +=  1.0   * f1_oo( aa )(h9_oa, h1_oa)")
    (_a006("bb")(h9_ob, h1_ob)                 +=  1.0   * f1_oo("bb")(h9_ob, h1_ob), 
    "_a006( bb )(h9_ob, h1_ob)                 +=  1.0   * f1_oo( bb )(h9_ob, h1_ob)")
    (_a006("aa")(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov("aa")(h9_oa, p8_va), 
    "_a006( aa )(h9_oa, h1_oa)                 +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov( aa )(h9_oa, p8_va)")
    (_a006("bb")(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov("bb")(h9_ob, p8_vb), 
    "_a006( bb )(h9_ob, h1_ob)                 +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov( bb )(h9_ob, p8_vb)")

    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001("aa")(p4_va, p2_va), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001( aa )(p4_va, p2_va)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001("bb")(p4_vb, p2_vb), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001( bb )(p4_vb, p2_vb)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p4_vb, p2_vb), 
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p4_vb, p2_vb)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001("aa")(p4_va, p2_va), 
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001( aa )(p4_va, p2_va)")

    (i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006("aa")(h3_oa, h2_oa), 
    "i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006( aa )(h3_oa, h2_oa)")
    (i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006("bb")(h3_ob, h2_ob), 
    "i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006( bb )(h3_ob, h2_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006("aa")(h3_oa, h2_oa), 
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006( aa )(h3_oa, h2_oa)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006("bb")(h3_ob, h2_ob), 
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006( bb )(h3_ob, h2_ob)")

    (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa)  =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa), 
    "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa)  =  1.0   * i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)")        
    (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa), 
    "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) +=  1.0   * i0_aaaa(p4_va, p3_va, h2_oa, h1_oa)")        
    (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa), 
    "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)")        
    (i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa), 
    "i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa) += -1.0   * i0_aaaa(p4_va, p3_va, h1_oa, h2_oa)")
    (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob)  =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob), 
    "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob)  =  1.0   * i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)")
    (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob), 
    "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) +=  1.0   * i0_bbbb(p4_vb, p3_vb, h2_ob, h1_ob)")
    (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob), 
    "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)") 
    (i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob), 
    "i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob) += -1.0   * i0_bbbb(p4_vb, p3_vb, h1_ob, h2_ob)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  1.0   * i0tmp("aaaa")(p3_va, p4_va, h1_oa, h2_oa), 
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  1.0   * i0tmp( aaaa )(p3_va, p4_va, h1_oa, h2_oa)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  1.0   * i0tmp("bbbb")(p3_vb, p4_vb, h1_ob, h2_ob), 
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  1.0   * i0tmp( bbbb )(p3_vb, p4_vb, h1_ob, h2_ob)")
    ;
  // clang-format on
}

template<typename T>
void RT_EOM_CD_CCSD<T>::scale_complex_ip(Tensor<T> tensor, double alpha_real, double alpha_imag) {
  if constexpr(tamm::internal::is_complex_v<T>) {
    std::function<T(T)> func = [&](T a) { return T(alpha_real * a.real(), alpha_imag * a.imag()); };
    apply_ewise_ip(tensor(), func);
  }
  else tamm_terminate("[scale_complex_ip] Tensor type T not complex ...");
}

template<typename T>
void RT_EOM_CD_CCSD<T>::td_iteration_print(ChemEnv& chem_env, int iter, CCEType energy,
                                           CCEType x1_1, CCEType x1_2, CCEType x2_1, CCEType x2_2,
                                           CCEType x2_3, double time) {
  std::cout.width(6);
  std::cout << std::right << iter + 1 << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(13) << energy.real() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(13) << energy.imag() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << x1_1.real() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << x1_2.real() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << x2_1.real() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << x2_2.real() << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << x2_3.real() << " ";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << std::string(5, ' ') << time;
  std::cout << std::endl;
}

template<typename T>
void RT_EOM_CD_CCSD<T>::debug_full_rt(Scheduler& sch, const TiledIndexSpace& MO,
                                      CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& r2_vvoo) {
  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const TiledIndexSpace  o_alpha = MO("occ_alpha");
  const TiledIndexSpace  v_alpha = MO("virt_alpha");
  const TiledIndexSpace  o_beta  = MO("occ_beta");
  const TiledIndexSpace  v_beta  = MO("virt_beta");

  auto [p1_va, p2_va, p3_va, p4_va] = v_alpha.labels<4>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta.labels<4>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha.labels<4>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob] = o_beta.labels<4>("all");

  Tensor<T> d_r1{{V, O}, {1, 1}};
  Tensor<T> d_r2{{V, V, O, O}, {2, 2}};
  Tensor<T> r2_baba{v_beta, v_alpha, o_beta, o_alpha};
  Tensor<T> r2_abba{v_alpha, v_beta, o_beta, o_alpha};
  Tensor<T> r2_baab{v_beta, v_alpha, o_alpha, o_beta};

  // clang-format off
  sch.allocate(d_r1,d_r2,r2_baba,r2_abba,r2_baab)
    (d_r1() = 0) (d_r2() = 0)
    (r2_baba() = 0) (r2_abba() = 0) (r2_baab() = 0)
    (r2_baba(p2_vb,p1_va,h4_ob,h3_oa) =        r2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob))
    (r2_abba(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * r2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob))
    (r2_baab(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * r2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob))
    
    (d_r1(p2_va, h1_oa)                = r1_vo("aa")(p2_va, h1_oa))
    (d_r1(p2_vb, h1_ob)                = r1_vo("bb")(p2_vb, h1_ob))
    (d_r2(p3_va, p4_va, h2_oa, h1_oa)  = r2_vvoo("aaaa")(p3_va, p4_va, h2_oa, h1_oa))
    (d_r2(p3_vb, p4_vb, h2_ob, h1_ob)  = r2_vvoo("bbbb")(p3_vb, p4_vb, h2_ob, h1_ob))
    (d_r2(p3_va, p4_vb, h2_oa, h1_ob)  = r2_vvoo("abab")(p3_va, p4_vb, h2_oa, h1_ob))

    (d_r2(p1_vb,p2_va,h3_ob,h4_oa) = r2_baba(p1_vb,p2_va,h3_ob,h4_oa))
    (d_r2(p1_va,p2_vb,h3_ob,h4_oa) = r2_abba(p1_va,p2_vb,h3_ob,h4_oa))
    (d_r2(p1_vb,p2_va,h3_oa,h4_ob) = r2_baab(p1_vb,p2_va,h3_oa,h4_ob))    
    .execute();
  // clang-format on

  T r1_norm = tamm::norm(d_r1);
  T r2_norm = tamm::norm(d_r2);

  if(sch.ec().print()) {
    // std::cout << std::string(70, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(13) << r1_norm << ", ";
    std::cout << std::fixed << std::setprecision(13) << r2_norm << std::endl;
    // std::cout << std::string(70, '-') << std::endl;
  }

  sch.deallocate(d_r1, d_r2, r2_baba, r2_abba, r2_baab).execute();
}

template<typename T>
void RT_EOM_CD_CCSD<T>::complex_copy_swap(ExecutionContext& ec, Tensor<T> src, Tensor<T> dest,
                                          bool update) {
  if constexpr(tamm::internal::is_complex_v<T>) {
    Tensor<T> tensor = dest;

    auto copy_lambda = [&](const IndexVector& bid) {
      const IndexVector     blockid = internal::translate_blockid(bid, tensor());
      const tamm::TAMM_SIZE dsize   = tensor.block_size(blockid);

      std::vector<T> sbuf(dsize);
      std::vector<T> dbuf(dsize, 0);
      src.get(blockid, sbuf);
      if(update) dest.get(blockid, dbuf);

      for(TAMM_SIZE i = 0; i < dsize; i++) {
        T val(sbuf[i].imag(), sbuf[i].real());
        if(update) dbuf[i] += val;
        else dbuf[i] = val;
      }
      dest.put(blockid, dbuf);
    };

    block_for(ec, tensor(), copy_lambda);
  }
  else tamm_terminate("[complex_copy_swap] Tensor type T not complex");
}

template<typename T>
void RT_EOM_CD_CCSD<T>::rt_eom_cd_ccsd(ChemEnv& chem_env, ExecutionContext& ec,
                                       const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                                       Tensor<T>& d_f1, std::vector<T>& p_evl_sorted,
                                       Tensor<T>& cv3d, bool cc_restart, std::string rt_eom_fp) {
  SystemData& sys_data = chem_env.sys_data;
  // int    maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  // int    ndiis       = chem_env.ioptions.ccsd_options.ndiis;
  const double thresh      = chem_env.ioptions.ccsd_options.rt_threshold;
  const bool   writet      = chem_env.ioptions.ccsd_options.writet;
  const int    writet_iter = chem_env.ioptions.ccsd_options.writet_iter;
  // double zshiftl     = chem_env.ioptions.ccsd_options.lshift;
  const bool profile = chem_env.ioptions.ccsd_options.profile_ccsd;
  // T    residual = 0.0;
  // T    energy   = 0.0;
  // int    niter       = 0;

  std::cout.precision(15);

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

  const auto [cind]                       = CI.labels<1>("all");
  const auto [p1_va, p2_va, p3_va, p4_va] = v_alpha.labels<4>("all");
  const auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta.labels<4>("all");
  const auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha.labels<4>("all");
  const auto [h1_ob, h2_ob, h3_ob, h4_ob] = o_beta.labels<4>("all");

  Tensor<T> d_e{};

  CCSE_Tensors<T> f1_oo{MO, {O, O}, "f1_oo", {"aa", "bb"}};
  CCSE_Tensors<T> f1_ov{MO, {O, V}, "f1_ov", {"aa", "bb"}};
  CCSE_Tensors<T> f1_vo{MO, {V, O}, "f1_vo", {"aa", "bb"}};
  CCSE_Tensors<T> f1_vv{MO, {V, V}, "f1_vv", {"aa", "bb"}};

  CCSE_Tensors<T> chol3d_oo{MO, {O, O, CI}, "chol3d_oo", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_ov{MO, {O, V, CI}, "chol3d_ov", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_vo{MO, {V, O, CI}, "chol3d_vo", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_vv{MO, {V, V, CI}, "chol3d_vv", {"aa", "bb"}};

  std::vector<CCSE_Tensors<T>> f1_se{f1_oo, f1_ov, f1_vo, f1_vv};
  std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo, chol3d_ov, chol3d_vo, chol3d_vv};

  CCSE_Tensors<T> t1_vo{MO, {V, O}, "t1", {"aa", "bb"}};
  CCSE_Tensors<T> t2_vvoo{MO, {V, V, O, O}, "t2", {"aaaa", "abab", "bbbb"}};

  CCSE_Tensors<T> t1_vo_aux{MO, {V, O}, "t1_aux", {"aa", "bb"}};
  CCSE_Tensors<T> t2_vvoo_aux{MO, {V, V, O, O}, "t2_aux", {"aaaa", "abab", "bbbb"}};

  CCSE_Tensors<T> t1_vo_old{MO, {V, O}, "t1_old", {"aa", "bb"}};
  CCSE_Tensors<T> t2_vvoo_old{MO, {V, V, O, O}, "t2_old", {"aaaa", "abab", "bbbb"}};

  CCSE_Tensors<T> r1_vo   = CCSE_Tensors<T>{MO, {V, O}, "r1", {"aa", "bb"}};
  CCSE_Tensors<T> r2_vvoo = CCSE_Tensors<T>{MO, {V, V, O, O}, "r2", {"aaaa", "abab", "bbbb"}};

  CCSE_Tensors<T> r1_vo_old = CCSE_Tensors<T>{MO, {V, O}, "r1_old", {"aa", "bb"}};
  CCSE_Tensors<T> r2_vvoo_old =
    CCSE_Tensors<T>{MO, {V, V, O, O}, "r2_old", {"aaaa", "abab", "bbbb"}};

  _a004 = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004", {"aaaa", "abab", "bbbb"}};

  // Energy intermediates
  _a01V = {CI};
  _a02  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02", {"aa", "bb"}};
  _a03  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03", {"aa", "bb"}};

  // T1
  _a02V = {CI};
  _a01  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01", {"aa", "bb"}};
  _a04  = CCSE_Tensors<T>{MO, {O, O}, "_a04", {"aa", "bb"}};
  _a05  = CCSE_Tensors<T>{MO, {O, V}, "_a05", {"aa", "bb"}};
  _a06  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06", {"aa", "bb"}};

  // T2
  _a007V = {CI};
  _a001  = CCSE_Tensors<T>{MO, {V, V}, "_a001", {"aa", "bb"}};
  _a006  = CCSE_Tensors<T>{MO, {O, O}, "_a006", {"aa", "bb"}};
  _a008  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008", {"aa", "bb"}};
  _a009  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009", {"aa", "bb"}};
  _a017  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017", {"aa", "bb"}};
  _a021  = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021", {"aa", "bb"}};

  _a019 = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"aaaa", "abab", "bbbb"}};
  _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"aaaa", "abab", "bbbb"}};
  _a020 =
    CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "abab", "baab", "abba", "baba", "bbbb"}};

  CCSE_Tensors<CCEType> i0_t2_tmp{MO, {V, V, O, O}, "i0_t2_tmp", {"aaaa", "bbbb"}};

  auto total_ccsd_mem = sum_tensor_sizes(cv3d, d_e, _a01V) +
                        CCSE_Tensors<T>::sum_tensor_sizes_list(r1_vo, r2_vvoo, t1_vo, t2_vvoo) +
                        CCSE_Tensors<T>::sum_tensor_sizes_list(
                          r1_vo_old, r2_vvoo_old, t1_vo_aux, t2_vvoo_aux, t1_vo_old, t2_vvoo_old) +
                        CCSE_Tensors<T>::sum_tensor_sizes_list(
                          f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo, chol3d_vv) +
                        CCSE_Tensors<T>::sum_tensor_sizes_list(_a02, _a03);

  // Intermediates
  auto total_ccsd_mem_tmp =
    sum_tensor_sizes(_a02V, _a007V) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(i0_t2_tmp, _a01, _a04, _a05, _a06, _a001, _a004, _a006,
                                           _a008, _a009, _a017, _a019, _a020, _a021, _a022);

  /* if(!cc_restart) */ total_ccsd_mem += total_ccsd_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for RT-EOM-Cholesky CCSD calculation: "
              << std::setprecision(2) << total_ccsd_mem << " GiB" << std::endl;
  }

  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  sch.allocate(d_e, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                 chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  CCSE_Tensors<T>::allocate_list(sch, r1_vo_old, r2_vvoo_old, t1_vo_aux, t2_vvoo_aux, t1_vo_old,
                                 t2_vvoo_old);
  CCSE_Tensors<T>::allocate_list(sch, _a02, _a03);
  sch.execute();

  const int pcore = chem_env.ioptions.ccsd_options.pcore - 1; // 0-based indexing
  if(pcore >= 0) {
    const auto timer_start = std::chrono::high_resolution_clock::now();

    Tensor<T>       d_f1_ut{d_f1, 2, {1, 1}};
    Tensor<T>       cv3d_ut{cv3d, 2, {1, 1}};
    TiledIndexSpace mo_ut     = cv3d_ut.tiled_index_spaces()[0];
    TiledIndexSpace cv3d_occ  = mo_ut("occ");
    TiledIndexSpace cv3d_virt = mo_ut("virt");

    auto [h1, h2, h3, h4] = cv3d_occ.labels<4>("all");
    auto [p1, p2, p3, p4] = cv3d_virt.labels<4>("all");

    TiledIndexSpace cv3d_utis{
      mo_ut, range(sys_data.nmo - sys_data.n_vir_beta, sys_data.nmo - sys_data.n_vir_beta + 1)};
    auto [c1, c2] = cv3d_utis.labels<2>("all");

    // clang-format off
    sch
      (d_f1_ut(h1,h2) += -1.0 * cv3d_ut(h1,h2,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(h1,h2) +=  1.0 * cv3d_ut(h1,c1,cind) * cv3d_ut(h2,c2,cind))
      (d_f1_ut(p1,p2) += -1.0 * cv3d_ut(p1,p2,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(p1,p2) +=  1.0 * cv3d_ut(p1,c1,cind) * cv3d_ut(p2,c2,cind))
      (d_f1_ut(p1,h1) += -1.0 * cv3d_ut(p1,h1,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(p1,h1) +=  1.0 * cv3d_ut(p1,c1,cind) * cv3d_ut(h1,c2,cind))
      (d_f1_ut(h1,p1) += -1.0 * cv3d_ut(h1,p1,cind) * cv3d_ut(c1,c2,cind))
      (d_f1_ut(h1,p1) +=  1.0 * cv3d_ut(h1,c1,cind) * cv3d_ut(p1,c2,cind));
    // clang-format on
    sch.execute(exhw);

    p_evl_sorted = tamm::diagonal(d_f1);

    const auto timer_end = std::chrono::high_resolution_clock::now();
    auto       f1_rctime =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();
    if(ec.print())
      std::cout << "Time to reconstruct Fock matrix: " << f1_rctime << " secs" << std::endl
                << std::endl;
  }

  // clang-format off
  sch
    (f1_oo("aa")(h3_oa,h4_oa)           =  d_f1(h3_oa,h4_oa))
    (f1_ov("aa")(h3_oa,p2_va)           =  d_f1(h3_oa,p2_va))
    (f1_vo("aa")(p1_va,h4_oa)           =  d_f1(p1_va,h4_oa))
    (f1_vv("aa")(p1_va,p2_va)           =  d_f1(p1_va,p2_va))
    (f1_oo("bb")(h3_ob,h4_ob)           =  d_f1(h3_ob,h4_ob))
    (f1_ov("bb")(h3_ob,p1_vb)           =  d_f1(h3_ob,p1_vb))
    (f1_vo("bb")(p1_vb,h3_ob)           =  d_f1(p1_vb,h3_ob))
    (f1_vv("bb")(p1_vb,p2_vb)           =  d_f1(p1_vb,p2_vb))
    (chol3d_oo("aa")(h3_oa,h4_oa,cind)  =  cv3d(h3_oa,h4_oa,cind))
    (chol3d_ov("aa")(h3_oa,p2_va,cind)  =  cv3d(h3_oa,p2_va,cind))
    (chol3d_vo("aa")(p1_va,h4_oa,cind)  =  cv3d(p1_va,h4_oa,cind))
    (chol3d_vv("aa")(p1_va,p2_va,cind)  =  cv3d(p1_va,p2_va,cind))
    (chol3d_oo("bb")(h3_ob,h4_ob,cind)  =  cv3d(h3_ob,h4_ob,cind))
    (chol3d_ov("bb")(h3_ob,p1_vb,cind)  =  cv3d(h3_ob,p1_vb,cind))
    (chol3d_vo("bb")(p1_vb,h3_ob,cind)  =  cv3d(p1_vb,h3_ob,cind))
    (chol3d_vv("bb")(p1_vb,p2_vb,cind)  =  cv3d(p1_vb,p2_vb,cind))
    .execute();
  // clang-format on

  // allocate all intermediates
  sch.allocate(_a02V, _a007V);
  CCSE_Tensors<T>::allocate_list(sch, _a004, i0_t2_tmp, _a01, _a04, _a05, _a06, _a001, _a006, _a008,
                                 _a009, _a017, _a019, _a020, _a021, _a022);
  sch.execute();

  // clang-format off
  sch
    (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("aa")(p2_va, h3_oa, cind))
    (_a004("abab")(p1_va, p2_vb, h4_oa, h3_ob) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind))
    (_a004("bbbb")(p1_vb, p2_vb, h4_ob, h3_ob) = 1.0 * chol3d_vo("bb")(p1_vb, h4_ob, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind));
  // clang-format on

  sch.execute(exhw);

  const int    ntimesteps    = chem_env.ioptions.ccsd_options.ntimesteps;
  const int    rt_microiter  = chem_env.ioptions.ccsd_options.rt_microiter;
  const double rt_multiplier = chem_env.ioptions.ccsd_options.rt_multiplier;
  const double rt_step_size  = chem_env.ioptions.ccsd_options.rt_step_size;
  const double scale_factor  = rt_multiplier * rt_step_size;

  CCSE_Tensors<T>::initialize(sch, 0, t1_vo, t2_vvoo, t1_vo_aux, t2_vvoo_aux, t1_vo_old,
                              t2_vvoo_old);
  sch.execute();

  if(ec.print()) {
    std::cout << "Number of RT-EOM-CC time steps          : " << ntimesteps << std::endl;
    std::cout << "Max number of Euler iterations          : " << rt_microiter << std::endl;
    std::cout << "Length of RT-EOM-CC time step (in au)   : " << rt_step_size << std::endl;
    std::cout << "Implicit Euler eomcc amplitude threshold: " << thresh << std::endl << std::endl;
  }

  int         ts_start  = 0;
  std::string dcdt_file = rt_eom_fp + ".dcdt";
  std::string ts_file   = rt_eom_fp + ".restart_ts";

  if(cc_restart) {
    bool ts_file_exists = fs::exists(ts_file);
    if(ts_file_exists) {
      if(ec.print()) {
        std::ifstream in(ts_file);
        if(in.is_open()) in >> ts_start;
      }
      ec.pg().broadcast(&ts_start, 0);
      // else tamm_terminate("[RT-EOM-CC] restart file " + ts_file + " is missing ");

      if(ts_start < ntimesteps) {
        if(t1_vo.exist_on_disk(rt_eom_fp)) {
          t1_vo.read_from_disk(rt_eom_fp);
          t2_vvoo.read_from_disk(rt_eom_fp);
          t1_vo_old.read_from_disk(rt_eom_fp);
          t2_vvoo_old.read_from_disk(rt_eom_fp);
        }
        if(ec.print()) std::cout << "Restarting from Timestep " << ts_start + 1 << std::endl;
      }
    }
  }

  for(int titer = ts_start; titer < ntimesteps; titer++) {
    if(ec.print()) std::cout << std::right << "Timestep " << titer + 1 << std::endl;

    // step 1
    ccsd_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);
    sch.execute(exhw, profile);

    if(ec.print()) {
      T                 energy = get_scalar(d_e);
      std::stringstream dcdt_out; // dcdt
      dcdt_out << std::fixed << std::setprecision(6) << (titer * rt_step_size)
               << std::string(3, ' ') << std::setprecision(13) << energy.real()
               << std::string(3, ' ') << energy.imag() << std::string(5, ' ') << titer + 1
               << std::endl;

      std::ofstream out(dcdt_file, std::ios::out | std::ofstream::app);
      out << dcdt_out.str();
      out.close();

      const std::string iter_str = std::to_string(titer + 1);
      sys_data.results["output"]["RT-EOMCCSD"]["dcdt"]["timestep"][iter_str]["time_au"] =
        titer * rt_step_size;
      sys_data.results["output"]["RT-EOMCCSD"]["dcdt"]["timestep"][iter_str]["energy"] = {
        {"real", energy.real()}, {"imag", energy.imag()}};
    }

    // CCSE_Tensors<T>::initialize(sch, 0, r1_vo, r2_vvoo, r1_vo_old, r2_vvoo_old);

    // step 2 (t_old = t)
    CCSE_Tensors<T>::copy(sch, t1_vo, t1_vo_old);
    CCSE_Tensors<T>::copy(sch, t2_vvoo, t2_vvoo_old);
    sch.execute();

    // step 3
    ccsd_t1_os(sch, MO, CI, r1_vo_old, t1_vo_old, t2_vvoo_old, f1_se, chol3d_se);
    ccsd_t2_os(sch, MO, CI, r2_vvoo_old, t1_vo_old, t2_vvoo_old, f1_se, chol3d_se, i0_t2_tmp);
    sch.execute(exhw, profile);

    // if(ec.print() && debug) std::cout << "Step 3 debug r1,r2 old norm ..." << std::endl;
    // if(debug) debug_full_rt(sch,MO,r1_vo_old,r2_vvoo_old);

    // microiter loop
    for(int li = 0; li < rt_microiter; li++) {
      const auto mt_start = std::chrono::high_resolution_clock::now();

      // step 4
      // t-aux = t_old
      CCSE_Tensors<T>::copy(sch, t1_vo_old, t1_vo_aux);
      CCSE_Tensors<T>::copy(sch, t2_vvoo_old, t2_vvoo_aux);
      sch.execute();

      // r1=r2=0
      // step 5
      ccsd_t1_os(sch, MO, CI, r1_vo, t1_vo, t2_vvoo, f1_se, chol3d_se);
      ccsd_t2_os(sch, MO, CI, r2_vvoo, t1_vo, t2_vvoo, f1_se, chol3d_se, i0_t2_tmp);
      sch.execute(exhw, profile);

      // step 6 (r += r_old)
      CCSE_Tensors<T>::copy(sch, r1_vo_old, r1_vo, true);
      CCSE_Tensors<T>::copy(sch, r2_vvoo_old, r2_vvoo, true);
      sch.execute();

      // step 7
      scale_complex_ip(r1_vo("aa"), scale_factor, -scale_factor);
      scale_complex_ip(r1_vo("bb"), scale_factor, -scale_factor);
      scale_complex_ip(r2_vvoo("aaaa"), scale_factor, -scale_factor);
      scale_complex_ip(r2_vvoo("abab"), scale_factor, -scale_factor);
      scale_complex_ip(r2_vvoo("bbbb"), scale_factor, -scale_factor);

      // step 8 (t_aux += r)
      // CCSE_Tensors<T>::copy(sch, r1_vo, t1_vo_aux, true);
      // CCSE_Tensors<T>::copy(sch, r2_vvoo, t2_vvoo_aux, true);
      // sch.execute();
      complex_copy_swap(ec, r1_vo("aa"), t1_vo_aux("aa"));
      complex_copy_swap(ec, r1_vo("bb"), t1_vo_aux("bb"));
      complex_copy_swap(ec, r2_vvoo("aaaa"), t2_vvoo_aux("aaaa"));
      complex_copy_swap(ec, r2_vvoo("abab"), t2_vvoo_aux("abab"));
      complex_copy_swap(ec, r2_vvoo("bbbb"), t2_vvoo_aux("bbbb"));

      // step 9
      ccsd_e_os(sch, MO, CI, d_e, t1_vo_aux, t2_vvoo_aux, f1_se, chol3d_se);
      sch.execute(exhw, profile);

      // step 10
      const T t1_aa_norm   = tamm::norm(t1_vo("aa"));
      const T t1_bb_norm   = tamm::norm(t1_vo("bb"));
      const T t2_aaaa_norm = tamm::norm(t2_vvoo("aaaa"));
      const T t2_abab_norm = tamm::norm(t2_vvoo("abab"));
      const T t2_bbbb_norm = tamm::norm(t2_vvoo("bbbb"));

      const T t1_aux_aa_norm   = tamm::norm(t1_vo_aux("aa"));
      const T t1_aux_bb_norm   = tamm::norm(t1_vo_aux("bb"));
      const T t2_aux_aaaa_norm = tamm::norm(t2_vvoo_aux("aaaa"));
      const T t2_aux_abab_norm = tamm::norm(t2_vvoo_aux("abab"));
      const T t2_aux_bbbb_norm = tamm::norm(t2_vvoo_aux("bbbb"));

      const T x1_1 = std::abs(t1_aux_aa_norm - t1_aa_norm);
      const T x1_2 = std::abs(t1_aux_bb_norm - t1_bb_norm);
      const T x2_1 = std::abs(t2_aux_aaaa_norm - t2_aaaa_norm);
      const T x2_2 = std::abs(t2_aux_abab_norm - t2_abab_norm);
      const T x2_3 = std::abs(t2_aux_bbbb_norm - t2_bbbb_norm);

      // step 11 (t = t_aux)
      CCSE_Tensors<T>::copy(sch, t1_vo_aux, t1_vo);
      CCSE_Tensors<T>::copy(sch, t2_vvoo_aux, t2_vvoo);
      sch.execute();

      const auto mt_end = std::chrono::high_resolution_clock::now();
      auto       mi_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((mt_end - mt_start)).count();

      if(ec.print())
        td_iteration_print(chem_env, li, get_scalar(d_e), x1_1, x1_2, x2_1, x2_2, x2_3, mi_time);

      // step 12
      if((x1_1.real() < thresh) && (x1_2.real() < thresh) && (x2_1.real() < thresh) &&
         (x2_2.real() < thresh) && (x2_3.real() < thresh))
        break;
    } // microiter loop

    if(writet && ((titer + 1) % writet_iter == 0)) {
      t1_vo.write_to_disk(rt_eom_fp);
      t2_vvoo.write_to_disk(rt_eom_fp);
      t1_vo_old.write_to_disk(rt_eom_fp);
      t2_vvoo_old.write_to_disk(rt_eom_fp);
      if(ec.print()) {
        std::ofstream out(ts_file, std::ios::out);
        out << titer + 1 << std::endl;
        out.close();
      }
    }

  } // end timestep loop

  if(ec.print()) {
    if(ts_start < ntimesteps) chem_env.write_json_data();

    if(profile) {
      std::string   profile_csv = rt_eom_fp + "_profile.csv";
      std::ofstream pds(profile_csv, std::ios::out);
      if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
      pds << ec.get_profile_header() << std::endl;
      pds << ec.get_profile_data().str() << std::endl;
      pds.close();
    }
  }

  sch.deallocate(_a02V, _a007V);
  CCSE_Tensors<T>::deallocate_list(sch, _a004, i0_t2_tmp, _a01, _a04, _a05, _a06, _a001, _a006,
                                   _a008, _a009, _a017, _a019, _a020, _a021, _a022);

  CCSE_Tensors<T>::deallocate_list(sch, _a02, _a03);
  CCSE_Tensors<T>::deallocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  CCSE_Tensors<T>::deallocate_list(sch, r1_vo_old, r2_vvoo_old, t1_vo_aux, t2_vvoo_aux, t1_vo_old,
                                   t2_vvoo_old);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                   chol3d_vv);
  sch.deallocate(d_e, _a01V).execute();
}

void rt_eom_cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T             = double;
  using ComplexTensor = Tensor<CCEType>;

  const auto  rank     = ec.pg().rank();
  SystemData& sys_data = chem_env.sys_data;

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  std::string files_dir    = chem_env.get_files_dir();
  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  CCContext& cc_context = chem_env.cc_context;
  cc_context.init_filenames(files_prefix);
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  const auto debug = ccsd_options.debug;

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& CI      = chem_env.is_context.CI;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = cd_context.d_f1;
  Tensor<T>        cholVpr = cd_context.cholV2;

  const bool is_rhf = sys_data.is_restricted;
  if(is_rhf) tamm_terminate("RHF not supported");

  bool cc_restart = ccsd_options.readt ||
                    (fs::exists(cd_context.f1file) && fs::exists(cd_context.v2file));
  Scheduler sch{ec};

  ComplexTensor d_f1_c{{d_f1.tiled_index_spaces()}, {1, 1}};
  ComplexTensor cholVpr_c{{cholVpr.tiled_index_spaces()}, {1, 1}};
  sch.allocate(cholVpr_c, d_f1_c)(d_f1_c() = d_f1())(cholVpr_c() = cholVpr())
    .deallocate(d_f1, cholVpr)
    .execute();

  std::vector<CCEType> p_evl_sorted = tamm::diagonal(d_f1_c);

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  cc_restart = cc_restart && ccsd_options.writet;
  files_dir += "/rteom/";
  if(!fs::exists(files_dir)) fs::create_directories(files_dir);
  files_prefix = files_dir + sys_data.output_file_prefix;

  RT_EOM_CD_CCSD<CCEType> rteom_obj;
  rteom_obj.rt_eom_cd_ccsd(chem_env, ec, MO, CI, d_f1_c, p_evl_sorted, cholVpr_c, cc_restart,
                           files_prefix);

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) {
    std::cout << std::endl
              << "Time taken for RT-EOM-Cholesky CCSD: " << ccsd_time << " secs" << std::endl;
  }

  free_tensors(d_f1_c, cholVpr_c);

  ec.flush_and_sync();
  // delete ec;
}
} // namespace exachem::rteom_cc::ccsd
