/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2026 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/rteom/rt_eom_cd_ccsd.hpp"
#include "exachem/cc/ccsd/ccsd_util.hpp"
#include "exachem/cc/rteom/rt_eom_common.hpp"
#include "exachem/cholesky/cholesky_2e_driver.hpp"

using namespace tamm;
namespace exachem::rteom_cc::ccsd {

using CCEType = std::complex<double>;

template<typename T>
void RT_EOM_CD_CCSD<T>::ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO,
                                  const TiledIndexSpace& CI, Tensor<T>& de, Tensor<T>& t1_aa,
                                  Tensor<T>& t1_bb, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb,
                                  Tensor<T>& t2_abab, std::vector<CCSE_Tensors<T>>& f1_se,
                                  std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind]                = CI.labels<1>("all");
  const auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  const auto [p1_vb, p2_vb, p3_vb] = v_beta.labels<3>("all");
  const auto [h3_oa, h4_oa, h6_oa] = o_alpha.labels<3>("all");
  const auto [h3_ob, h4_ob, h6_ob] = o_beta.labels<3>("all");

  // f1_se{f1_oo,f1_ov,f1_vo,f1_vv}
  // chol3d_se{chol3d_oo,chol3d_ov,chol3d_vo,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
  (t2_aaaa()                       = 0)
  (t2_bbbb()                       = 0)  
  (t2_abab()                       = 0)
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
                                   const TiledIndexSpace& CI, Tensor<T>& i0_aa, Tensor<T>& i0_bb,
                                   Tensor<T>& t1_aa, Tensor<T>& t1_bb, Tensor<T>& t2_aaaa,
                                   Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
                                   std::vector<CCSE_Tensors<T>>& f1_se,
                                   std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind]                       = CI.labels<1>("all");
  const auto [p2]                         = MO.labels<1>("virt");
  const auto [h1]                         = MO.labels<1>("occ");
  const auto [p1_va, p2_va, p3_va]        = v_alpha.labels<3>("all");
  const auto [p1_vb, p2_vb, p3_vb]        = v_beta.labels<3>("all");
  const auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha.labels<4>("all");
  const auto [h1_ob, h2_ob, h3_ob, h7_ob] = o_beta.labels<4>("all");

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
 sch(t2_aaaa()                        = 0)
    (t2_abab()                        = 0)
    (t2_bbbb()                        = 0)     
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
                                   const TiledIndexSpace& CI, Tensor<T>& i0_aaaa,
                                   Tensor<T>& i0_bbbb, Tensor<T>& i0_abab, Tensor<T>& t1_aa,
                                   Tensor<T>& t1_bb, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb,
                                   Tensor<T>& t2_abab, std::vector<CCSE_Tensors<T>>& f1_se,
                                   std::vector<CCSE_Tensors<T>>& chol3d_se,
                                   CCSE_Tensors<T>&              i0tmp) {
  const auto [cind]                                     = CI.labels<1>("all");
  const auto [p3, p4]                                   = MO.labels<2>("virt");
  const auto [h1, h2]                                   = MO.labels<2>("occ");
  const auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha.labels<6>("all");
  const auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] = v_beta.labels<6>("all");
  const auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha.labels<6>("all");
  const auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] = o_beta.labels<6>("all");

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
    std::function<T(T)> func = [&](T a) {
      T val(alpha_real * a.real(), alpha_imag * a.imag());
      return val;
    };
    apply_ewise_ip(tensor(), func);
  }
  else tamm_terminate("[scale_complex_ip] T not complex ...");
}

template<typename T>
void RT_EOM_CD_CCSD<T>::td_iteration_print(ChemEnv& chem_env, int iter, double rt_step_size,
                                           CCEType energy, CCEType x1_1, CCEType x1_2, CCEType x2_1,
                                           CCEType x2_2, CCEType x2_3, double time) {
  std::cout.width(6);
  std::cout << std::right << iter + 1 << std::string(3, ' ');
  std::cout << std::fixed << std::setprecision(6) << rt_step_size << std::string(5, ' ');
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
      const IndexVector blockid = internal::translate_blockid(bid, tensor());

      auto block_dims   = tensor.block_dims(blockid);
      auto block_offset = tensor.block_offsets(blockid);

      const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);

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
  else tamm_terminate("[complex_copy_swap] Tensors provided are not complex");
}

template<typename T>
void RT_EOM_CD_CCSD<T>::secant_update_ip(ExecutionContext& ec, Tensor<T> t, Tensor<T> t_upper,
                                         Tensor<T> f_upper, Tensor<T> delta_t, Tensor<T> deno) {
  // all five tensors share the same tiled index spaces, hence the same block structure
  const T zero{};

  auto update_lambda = [&](const IndexVector& bid) {
    const IndexVector     blockid = internal::translate_blockid(bid, t());
    const tamm::TAMM_SIZE bsize   = t.block_size(blockid);

    std::vector<T> tbuf(bsize);
    std::vector<T> tu_buf(bsize);
    std::vector<T> fu_buf(bsize);
    std::vector<T> dt_buf(bsize);
    std::vector<T> de_buf(bsize);
    t_upper.get(blockid, tu_buf);
    f_upper.get(blockid, fu_buf);
    delta_t.get(blockid, dt_buf);
    deno.get(blockid, de_buf);

    for(TAMM_SIZE i = 0; i < bsize; i++) {
      if(de_buf[i] != zero) tbuf[i] = tu_buf[i] - fu_buf[i] * dt_buf[i] / de_buf[i];
      else tbuf[i] = zero;
    }
    t.put(blockid, tbuf);
  };

  block_for(ec, t(), update_lambda);
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
  const bool profile = chem_env.ioptions.ccsd_options.profile;
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

  auto [cind]                       = CI.labels<1>("all");
  auto [p1_va, p2_va, p3_va, p4_va] = v_alpha.labels<4>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta.labels<4>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha.labels<4>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob] = o_beta.labels<4>("all");

  auto [i, j, k, l] = MO.labels<4>("occ");
  auto [a, b, c, d] = MO.labels<4>("virt");

  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

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

  //  CCSE_Tensors<T> t1_vo{MO, {V, O}, "t1", {"aa", "bb"}};
  //  CCSE_Tensors<T> t2_vvoo{MO, {V, V, O, O}, "t2", {"aaaa", "abab", "bbbb"}};

  //  CCSE_Tensors<T> t1_vo_aux{MO, {V, O}, "t1_aux", {"aa", "bb"}};
  //  CCSE_Tensors<T> t2_vvoo_aux{MO, {V, V, O, O}, "t2_aux", {"aaaa", "abab", "bbbb"}};

  //  CCSE_Tensors<T> t1_vo_old{MO, {V, O}, "t1_old", {"aa", "bb"}};
  //  CCSE_Tensors<T> t2_vvoo_old{MO, {V, V, O, O}, "t2_old", {"aaaa", "abab", "bbbb"}};

  //  CCSE_Tensors<T> r1_vo   = CCSE_Tensors<T>{MO, {V, O}, "r1", {"aa", "bb"}};
  //  CCSE_Tensors<T> r2_vvoo = CCSE_Tensors<T>{MO, {V, V, O, O}, "r2", {"aaaa", "abab", "bbbb"}};

  //  CCSE_Tensors<T> r1_vo_old = CCSE_Tensors<T>{MO, {V, O}, "r1_old", {"aa", "bb"}};
  //  CCSE_Tensors<T> r2_vvoo_old =
  //   CCSE_Tensors<T>{MO, {V, V, O, O}, "r2_old", {"aaaa", "abab", "bbbb"}};

  Tensor<T> t1_aa{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb{{p1_vb, h1_ob}, {1, 1}};
  Tensor<T> t1_aa_aux{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_aux{{p1_vb, h1_ob}, {1, 1}};
  Tensor<T> t1_aa_old{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_old{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> t1_aa_aux_copy{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_aux_copy{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> t1_aa_diff{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_diff{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> t1_aa_lower{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_lower{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> t1_aa_upper{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> t1_bb_upper{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> delta_t1_aa{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> delta_t1_bb{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> f1_aa_upper{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> f1_bb_upper{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> deno_aa{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> deno_bb{{p1_vb, h1_ob}, {1, 1}};

  //
  Tensor<T> t2_aaaa{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_aux{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_aux{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_aux{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_aux_copy{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_aux_copy{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_aux_copy{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_diff{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_diff{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_diff{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_old{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_old{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_old{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_lower{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_lower{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_lower{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t2_aaaa_upper{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> t2_bbbb_upper{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> t2_abab_upper{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> f2_aaaa_upper{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> f2_bbbb_upper{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> f2_abab_upper{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> delta_t2_aaaa{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> delta_t2_bbbb{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> delta_t2_abab{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  //

  Tensor<T> r1_aa{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> r1_aa_old{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb_old{{p1_vb, h1_ob}, {1, 1}};
  Tensor<T> r1_aa_old_copy{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb_old_copy{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> r1_aa_upper_copy{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb_upper_copy{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> r1_aa_lower{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb_lower{{p1_vb, h1_ob}, {1, 1}};
  Tensor<T> r1_aa_upper{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> r1_bb_upper{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> delta_r1_aa{{p1_va, h1_oa}, {1, 1}};
  Tensor<T> delta_r1_bb{{p1_vb, h1_ob}, {1, 1}};

  Tensor<T> r2_aaaa{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> r2_aaaa_old{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb_old{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab_old{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> r2_aaaa_old_copy{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb_old_copy{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab_old_copy{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> r2_aaaa_upper_copy{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb_upper_copy{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab_upper_copy{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> r2_aaaa_lower{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb_lower{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab_lower{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> r2_aaaa_upper{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> r2_bbbb_upper{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> r2_abab_upper{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> delta_r2_aaaa{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> delta_r2_bbbb{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> delta_r2_abab{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> deno_aaaa{{p1_va, p2_va, h1_oa, h2_oa}, {2, 2}};
  Tensor<T> deno_bbbb{{p1_vb, p2_vb, h1_ob, h2_ob}, {2, 2}};
  Tensor<T> deno_abab{{p1_va, p2_vb, h1_oa, h2_ob}, {2, 2}};

  Tensor<T> t1_full_diff{{V, O}, {1, 1}};

  sch
    .allocate(t1_aa, t1_bb, t1_aa_aux, t1_bb_aux, t1_aa_aux_copy, t1_bb_aux_copy, t1_aa_diff,
              t1_bb_diff, t1_aa_old, t1_bb_old, t2_aaaa, t2_bbbb, t2_abab, t2_aaaa_aux, t2_bbbb_aux,
              t2_abab_aux, t2_aaaa_aux_copy, t2_bbbb_aux_copy, t2_abab_aux_copy, t2_aaaa_diff,
              t2_bbbb_diff, t2_abab_diff, t2_aaaa_old, t2_bbbb_old, t2_abab_old, r1_aa, r1_bb,
              r1_aa_old, r1_bb_old, r1_aa_old_copy, r1_bb_old_copy, r1_aa_upper_copy,
              r1_bb_upper_copy, r2_aaaa, r2_bbbb, r2_abab, r2_aaaa_old, r2_bbbb_old, r2_abab_old,
              r2_aaaa_old_copy, r2_bbbb_old_copy, r2_abab_old_copy, r2_aaaa_upper_copy,
              r2_bbbb_upper_copy, r2_abab_upper_copy, t1_aa_lower, t1_bb_lower, t1_aa_upper,
              t1_bb_upper, t2_aaaa_lower, t2_bbbb_lower, t2_abab_lower, t2_aaaa_upper,
              t2_bbbb_upper, t2_abab_upper, r1_aa_lower, r1_bb_lower, r1_aa_upper, r1_bb_upper,
              r2_aaaa_lower, r2_bbbb_lower, r2_abab_lower, r2_aaaa_upper, r2_bbbb_upper,
              r2_abab_upper, delta_r1_aa, delta_r1_bb, delta_r2_aaaa, delta_r2_bbbb, delta_r2_abab,
              delta_t1_aa, delta_t1_bb, delta_t2_aaaa, delta_t2_bbbb, delta_t2_abab, f1_aa_upper,
              f1_bb_upper, f2_aaaa_upper, f2_bbbb_upper, f2_abab_upper, deno_aa, deno_bb, deno_aaaa,
              deno_bbbb, deno_abab, t1_full_diff)
    .execute();

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

  // All tensors below are allocated upfront and live for the entire time propagation,
  // so the peak memory is the sum of all of them.
  auto total_ccsd_mem =
    sum_tensor_sizes(cv3d, d_e, _a01V) +
    // T1: t1, aux, aux_copy, diff, old, lower, upper, delta, f1_upper, deno, t1_full_diff
    sum_tensor_sizes(t1_aa, t1_bb, t1_aa_aux, t1_bb_aux, t1_aa_aux_copy, t1_bb_aux_copy, t1_aa_diff,
                     t1_bb_diff, t1_aa_old, t1_bb_old, t1_aa_lower, t1_bb_lower, t1_aa_upper,
                     t1_bb_upper, delta_t1_aa, delta_t1_bb, f1_aa_upper, f1_bb_upper, deno_aa,
                     deno_bb, t1_full_diff) +
    // T2: t2, aux, aux_copy, diff, old, lower, upper, delta, f2_upper, deno
    sum_tensor_sizes(t2_aaaa, t2_bbbb, t2_abab, t2_aaaa_aux, t2_bbbb_aux, t2_abab_aux,
                     t2_aaaa_aux_copy, t2_bbbb_aux_copy, t2_abab_aux_copy, t2_aaaa_diff,
                     t2_bbbb_diff, t2_abab_diff, t2_aaaa_old, t2_bbbb_old, t2_abab_old,
                     t2_aaaa_lower, t2_bbbb_lower, t2_abab_lower, t2_aaaa_upper, t2_bbbb_upper,
                     t2_abab_upper, delta_t2_aaaa, delta_t2_bbbb, delta_t2_abab, f2_aaaa_upper,
                     f2_bbbb_upper, f2_abab_upper, deno_aaaa, deno_bbbb, deno_abab) +
    // R1: r1, old, old_copy, upper_copy, lower, upper, delta
    sum_tensor_sizes(r1_aa, r1_bb, r1_aa_old, r1_bb_old, r1_aa_old_copy, r1_bb_old_copy,
                     r1_aa_upper_copy, r1_bb_upper_copy, r1_aa_lower, r1_bb_lower, r1_aa_upper,
                     r1_bb_upper, delta_r1_aa, delta_r1_bb) +
    // R2: r2, old, old_copy, upper_copy, lower, upper, delta
    sum_tensor_sizes(r2_aaaa, r2_bbbb, r2_abab, r2_aaaa_old, r2_bbbb_old, r2_abab_old,
                     r2_aaaa_old_copy, r2_bbbb_old_copy, r2_abab_old_copy, r2_aaaa_upper_copy,
                     r2_bbbb_upper_copy, r2_abab_upper_copy, r2_aaaa_lower, r2_bbbb_lower,
                     r2_abab_lower, r2_aaaa_upper, r2_bbbb_upper, r2_abab_upper, delta_r2_aaaa,
                     delta_r2_bbbb, delta_r2_abab) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov,
                                           chol3d_vo, chol3d_vv) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(_a02, _a03);

  // Intermediates
  auto total_ccsd_mem_tmp =
    sum_tensor_sizes(_a02V, _a007V) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(i0_t2_tmp, _a01, _a04, _a05, _a06, _a001, _a004, _a006,
                                           _a008, _a009, _a017, _a019, _a020, _a021, _a022);

  total_ccsd_mem += total_ccsd_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for RT-EOM-Cholesky CCSD calculation: "
              << std::setprecision(2) << total_ccsd_mem << " GiB" << std::endl;
  }

  sch.allocate(d_e, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                 chol3d_vv);
  //  CCSE_Tensors<T>::allocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  //  CCSE_Tensors<T>::allocate_list(sch, r1_vo_old, r2_vvoo_old, t1_vo_aux, t2_vvoo_aux, t1_vo_old,
  //                                 t2_vvoo_old);
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

  const int ntimesteps   = chem_env.ioptions.ccsd_options.ntimesteps;
  const int rt_microiter = chem_env.ioptions.ccsd_options.rt_microiter;
  // const double rt_multiplier = chem_env.ioptions.ccsd_options.rt_multiplier;
  double rt_step_size = chem_env.ioptions.ccsd_options.rt_step_size;
  // const double scale_factor  = rt_multiplier * rt_step_size;

  const double h_max = chem_env.ioptions.ccsd_options.rt_step_max;
  //  double h_min = 0.001;
  const double scx       = chem_env.ioptions.ccsd_options.rt_secant_x;
  const double h_red_fac = chem_env.ioptions.ccsd_options.rt_step_dec;
  const double h_inc_fac = chem_env.ioptions.ccsd_options.rt_step_inc;

  // step 1: Initialize

  // clang-format off
  sch(t1_aa()         = 0)
    (t1_bb()         = 0)
    (t1_aa_aux()     = 0)
    (t1_bb_aux()     = 0)
    (t1_aa_diff()    = 0)
    (t1_bb_diff()    = 0)
    (t1_aa_old()     = 0)
    (t1_bb_old()     = 0)
    (t2_aaaa()       = 0)
    (t2_bbbb()       = 0)
    (t2_abab()       = 0)
    (t2_aaaa_aux()   = 0)
    (t2_bbbb_aux()   = 0)
    (t2_abab_aux()   = 0)
    (t2_aaaa_diff()  = 0)
    (t2_bbbb_diff()  = 0)
    (t2_abab_diff()  = 0)
    (t2_aaaa_old()   = 0)
    (t2_bbbb_old()   = 0)
    (t2_abab_old()   = 0);
  // clang-format on

  sch.execute();

  if(ec.print()) {
    std::cout << "Number of RT-EOM-CC time steps          : " << ntimesteps << std::endl;
    std::cout << "Max number of micro-iterations          : " << rt_microiter << std::endl;
    std::cout << "Initial time step size (in au)          : " << std::fixed << std::setprecision(3)
              << rt_step_size << std::endl;
    std::cout << "Max time step size (in au)              : " << h_max << std::endl;
    std::cout << "Step size decrease / increase factors   : " << std::setprecision(2) << h_red_fac
              << " / " << h_inc_fac << std::endl;
    std::cout << "Secant initial-guess scaling            : " << std::setprecision(3) << scx
              << std::endl;
    std::cout << "Secant amplitude threshold              : " << std::scientific
              << std::setprecision(0) << thresh << std::endl
              << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6); // restore stream defaults
  }

  int         ts_start   = 0;
  double      total_time = 0.0; //
  std::string dcdt_file  = rt_eom_fp + ".dcdt";
  std::string ts_file    = rt_eom_fp + ".restart_ts";

  std::string t1_aa_file     = rt_eom_fp + ".t1_aa";
  std::string t1_bb_file     = rt_eom_fp + ".t1_bb";
  std::string t1_aa_old_file = rt_eom_fp + ".t1_aa_old";
  std::string t1_bb_old_file = rt_eom_fp + ".t1_bb_old";

  std::string t2_aaaa_file     = rt_eom_fp + ".t2_aaaa";
  std::string t2_bbbb_file     = rt_eom_fp + ".t2_bbbb";
  std::string t2_abab_file     = rt_eom_fp + ".t2_abab";
  std::string t2_aaaa_old_file = rt_eom_fp + ".t2_aaaa_old";
  std::string t2_bbbb_old_file = rt_eom_fp + ".t2_bbbb_old";
  std::string t2_abab_old_file = rt_eom_fp + ".t2_abab_old";

  // restart tensors are read/written as one group so a checkpoint is committed all-or-nothing
  std::vector<Tensor<T>>   restart_tensors{t1_aa,   t1_bb,   t1_aa_old,   t1_bb_old,   t2_aaaa,
                                         t2_bbbb, t2_abab, t2_aaaa_old, t2_bbbb_old, t2_abab_old};
  std::vector<std::string> restart_files{
    t1_aa_file,   t1_bb_file,   t1_aa_old_file,   t1_bb_old_file,   t2_aaaa_file,
    t2_bbbb_file, t2_abab_file, t2_aaaa_old_file, t2_bbbb_old_file, t2_abab_old_file};

  if(cc_restart) {
    bool ts_file_exists = fs::exists(ts_file);
    if(ts_file_exists) {
      if(ec.print()) {
        std::ifstream in(ts_file);
        if(in.is_open()) in >> ts_start >> total_time >> rt_step_size;
      }
      ec.pg().broadcast(&ts_start, 0);
      // else tamm_terminate("[RT-EOM-CC] restart file " + ts_file + " is missing ");

      if(ts_start < ntimesteps) {
        if(fs::exists(t1_aa_file)) { read_from_disk(ec, restart_tensors, restart_files, profile); }
        if(ec.print()) std::cout << "Restarting from Timestep " << ts_start + 1 << std::endl;
      }
    }
  }

  // step 2: time-propagation starts

  for(int titer = ts_start; titer < ntimesteps; titer++) {
    if(ec.print()) std::cout << std::right << "Timestep " << titer + 1 << std::endl;

    total_time += rt_step_size;

    // step 3: compute the cumulant with the available T, this goes to DCDT file

    ccsd_e_os(sch, MO, CI, d_e, t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f1_se, chol3d_se);
    sch.execute(exhw, profile);

    if(ec.print()) {
      T                 energy = get_scalar(d_e);
      std::stringstream dcdt_out; // dcdt
      dcdt_out << titer + 1 << "\t" << std::fixed << std::setprecision(6)
               << (total_time - rt_step_size) << std::string(3, ' ') << std::setprecision(13)
               << energy.real() << std::string(3, ' ') << energy.imag() << std::string(5, ' ')
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

    // step 4 (t_old = t); keep a copy of the current T as T_old

    // clang-format off
    sch(t1_aa_old()   = 0)
       (t1_bb_old()   = 0)
       (t2_aaaa_old() = 0)
       (t2_bbbb_old() = 0)
       (t2_abab_old() = 0)        
       (t1_aa_old()   += t1_aa())
       (t1_bb_old()   += t1_bb())
       (t2_aaaa_old() += t2_aaaa())
       (t2_bbbb_old() += t2_bbbb())
       (t2_abab_old() += t2_abab());
    // clang-format on

    sch.execute();

    // step 5 (R_old = CC_R(T_old); Compute R_old using the T_old

    ccsd_t1_os(sch, MO, CI, r1_aa_old, r1_bb_old, t1_aa_old, t1_bb_old, t2_aaaa_old, t2_bbbb_old,
               t2_abab_old, f1_se, chol3d_se);
    ccsd_t2_os(sch, MO, CI, r2_aaaa_old, r2_bbbb_old, r2_abab_old, t1_aa_old, t1_bb_old,
               t2_aaaa_old, t2_bbbb_old, t2_abab_old, f1_se, chol3d_se, i0_t2_tmp);
    sch.execute(exhw, profile);

    // step 6: Compute T_aux; (T_aux = T_old+i*0.5*h*R_old)

    // keep a copy of R_old before modifying

    // clang-format off
    sch(r1_aa_old_copy()  = 0)
       (r1_aa_old_copy() += r1_aa_old())
       (r1_bb_old_copy()  = 0)	
       (r1_bb_old_copy() += r1_bb_old())
       (r2_aaaa_old_copy()= 0)
       (r2_bbbb_old_copy()= 0) 
       (r2_abab_old_copy()= 0)  
       (r2_aaaa_old_copy()  += r2_aaaa_old())
       (r2_bbbb_old_copy()  += r2_bbbb_old())
       (r2_abab_old_copy()  += r2_abab_old());
    // clang-format on

    sch.execute();

    double scale_factor1 = 0.5 * rt_step_size;

    scale_complex_ip(r1_aa_old, scale_factor1, -scale_factor1);
    scale_complex_ip(r1_bb_old, scale_factor1, -scale_factor1);
    scale_complex_ip(r2_aaaa_old, scale_factor1, -scale_factor1);
    scale_complex_ip(r2_bbbb_old, scale_factor1, -scale_factor1);
    scale_complex_ip(r2_abab_old, scale_factor1, -scale_factor1);

    // clang-format off
    sch (t1_aa_aux()  = 0)
	      (t1_bb_aux()  = 0)
        (t2_aaaa_aux() = 0)
        (t2_bbbb_aux() = 0)
        (t2_abab_aux() = 0)	     
	      (t1_aa_aux()   += t1_aa_old())
        (t1_bb_aux()     += t1_bb_old())
        (t2_aaaa_aux()   += t2_aaaa_old())
        (t2_bbbb_aux()   += t2_bbbb_old())
        (t2_abab_aux()   += t2_abab_old());
    // clang-format on

    sch.execute();

    // (t_aux += r_old)

    complex_copy_swap(ec, r1_aa_old, t1_aa_aux);
    complex_copy_swap(ec, r1_bb_old, t1_bb_aux);
    complex_copy_swap(ec, r2_aaaa_old, t2_aaaa_aux);
    complex_copy_swap(ec, r2_bbbb_old, t2_bbbb_aux);
    complex_copy_swap(ec, r2_abab_old, t2_abab_aux);

    // step 7: Compute guess Ts for the micro-iterations

    double lower_factor = 1.0 - scx;
    double upper_factor = 1.0 + scx;

    double lower_scale_factor = lower_factor * rt_step_size;
    // double lower_scale_factor =lower_factor*0.0005;

    // clang-format off
    sch(r1_aa_old()    = 0)
       (r1_aa_old()   += r1_aa_old_copy())
       (r1_bb_old()    = 0)
       (r1_bb_old()   += r1_bb_old_copy())
       (r2_aaaa_old()  = 0)
       (r2_aaaa_old() += r2_aaaa_old_copy())
       (r2_bbbb_old()  = 0)
       (r2_bbbb_old() += r2_bbbb_old_copy())
       (r2_abab_old()  = 0)
       (r2_abab_old() += r2_abab_old_copy());
    // clang-format on

    sch.execute();

    scale_complex_ip(r1_aa_old, lower_scale_factor, -lower_scale_factor);
    scale_complex_ip(r1_bb_old, lower_scale_factor, -lower_scale_factor);
    scale_complex_ip(r2_aaaa_old, lower_scale_factor, -lower_scale_factor);
    scale_complex_ip(r2_bbbb_old, lower_scale_factor, -lower_scale_factor);
    scale_complex_ip(r2_abab_old, lower_scale_factor, -lower_scale_factor);

    // clang-format off
    sch (t1_aa_lower()      = 0)
	      (t1_bb_lower()      = 0)
        (t2_aaaa_lower()    = 0)
        (t2_bbbb_lower()    = 0)
        (t2_abab_lower()    = 0) 	
        (t1_aa_lower()     += t1_aa_old())
        (t1_bb_lower()     += t1_bb_old())
        (t2_aaaa_lower()   += t2_aaaa_old())
        (t2_bbbb_lower()   += t2_bbbb_old())
        (t2_abab_lower()   += t2_abab_old());
    // clang-format on
    sch.execute();

    // (t_lower += r_old)

    complex_copy_swap(ec, r1_aa_old, t1_aa_lower);
    complex_copy_swap(ec, r1_bb_old, t1_bb_lower);
    complex_copy_swap(ec, r2_aaaa_old, t2_aaaa_lower);
    complex_copy_swap(ec, r2_bbbb_old, t2_bbbb_lower);
    complex_copy_swap(ec, r2_abab_old, t2_abab_lower);

    double upper_scale_factor = upper_factor * rt_step_size;

    // double upper_scale_factor =upper_factor*0.0005;

    // clang-format off
    sch(r1_aa_old()    = 0)
       (r1_aa_old()   += r1_aa_old_copy())
       (r1_bb_old()    = 0)
       (r1_bb_old()   += r1_bb_old_copy())
       (r2_aaaa_old()  = 0)
       (r2_aaaa_old() += r2_aaaa_old_copy())
       (r2_bbbb_old()  = 0)
       (r2_bbbb_old() += r2_bbbb_old_copy())
       (r2_abab_old()  = 0)
       (r2_abab_old() += r2_abab_old_copy());
    // clang-format on
    sch.execute();

    scale_complex_ip(r1_aa_old, upper_scale_factor, -upper_scale_factor);
    scale_complex_ip(r1_bb_old, upper_scale_factor, -upper_scale_factor);
    scale_complex_ip(r2_aaaa_old, upper_scale_factor, -upper_scale_factor);
    scale_complex_ip(r2_bbbb_old, upper_scale_factor, -upper_scale_factor);
    scale_complex_ip(r2_abab_old, upper_scale_factor, -upper_scale_factor);

    // clang-format off
  sch(t1_aa_upper()   = 0)
     (t1_bb_upper()   = 0)
     (t2_aaaa_upper() = 0) 
     (t2_bbbb_upper() = 0)
     (t2_abab_upper() = 0)      
     (t1_aa_upper()   += t1_aa_old())
     (t1_bb_upper()   += t1_bb_old())
     (t2_aaaa_upper() += t2_aaaa_old())
     (t2_bbbb_upper() += t2_bbbb_old())
     (t2_abab_upper() += t2_abab_old());
    // clang-format on
    sch.execute();

    // (t_upper += r_old)

    complex_copy_swap(ec, r1_aa_old, t1_aa_upper);
    complex_copy_swap(ec, r1_bb_old, t1_bb_upper);
    complex_copy_swap(ec, r2_aaaa_old, t2_aaaa_upper);
    complex_copy_swap(ec, r2_bbbb_old, t2_bbbb_upper);
    complex_copy_swap(ec, r2_abab_old, t2_abab_upper);

    // sch(r1_aa_lower() = 0)
    //    (r1_bb_lower() = 0)
    //    (r2_aaaa_lower() = 0)
    //    (r2_bbbb_lower() = 0)
    //    (r2_abab_lower() = 0)
    //    (r1_aa_upper() = 0)
    //    (r1_bb_upper() = 0)
    //    (r2_aaaa_upper() = 0)
    //    (r2_bbbb_upper() = 0)
    //    (r2_abab_upper() = 0);
    //    sch.execute();

    // step 8: Compute guess residual with guess Ts
    //(R_lower = CC_R(T_lower); Compute R_lower using the T_lower
    //(R_upper = CC_R(T_upper); Compute R_upper using the T_upper
    //
    ccsd_t1_os(sch, MO, CI, r1_aa_lower, r1_bb_lower, t1_aa_lower, t1_bb_lower, t2_aaaa_lower,
               t2_bbbb_lower, t2_abab_lower, f1_se, chol3d_se);
    ccsd_t2_os(sch, MO, CI, r2_aaaa_lower, r2_bbbb_lower, r2_abab_lower, t1_aa_lower, t1_bb_lower,
               t2_aaaa_lower, t2_bbbb_lower, t2_abab_lower, f1_se, chol3d_se, i0_t2_tmp);
    ccsd_t1_os(sch, MO, CI, r1_aa_upper, r1_bb_upper, t1_aa_upper, t1_bb_upper, t2_aaaa_upper,
               t2_bbbb_upper, t2_abab_upper, f1_se, chol3d_se);
    ccsd_t2_os(sch, MO, CI, r2_aaaa_upper, r2_bbbb_upper, r2_abab_upper, t1_aa_upper, t1_bb_upper,
               t2_aaaa_upper, t2_bbbb_upper, t2_abab_upper, f1_se, chol3d_se, i0_t2_tmp);
    sch.execute(exhw, profile);

    // std::cout << std::fixed << "rt_step_size before entering micro-iter" << std::endl;
    // std::cout << std::fixed << std::setprecision(6) << rt_step_size << std::endl;

    int li = 0;
    // step 9: Micro-iteration starts here
    // microiter loop
    for(li = 0; li < rt_microiter; li++) {
      const auto mt_start = std::chrono::high_resolution_clock::now();

      // std::cout << "time-step fixed during the micro-iter at" << std::endl;
      // std::cout << std::fixed << std::setprecision(6) << rt_step_size << std::endl;

      // step 10: Compute delta R
      // Delta R = R_lower- R_upper

      // clang-format off
      sch(delta_r1_aa()     = 0)
        (delta_r1_bb()     = 0)
        (delta_r2_aaaa()   = 0)
        (delta_r2_bbbb()   = 0) 
        (delta_r2_abab()   = 0)   
        (delta_r1_aa()     = r1_aa_lower())
        (delta_r1_aa()    += -1.0*r1_aa_upper())
        (delta_r1_bb()     = r1_bb_lower())
        (delta_r1_bb()    += -1.0*r1_bb_upper())
        (delta_r2_aaaa()  += r2_aaaa_lower())
        (delta_r2_aaaa()  += -1.0*r2_aaaa_upper())
        (delta_r2_bbbb()  += r2_bbbb_lower())
        (delta_r2_bbbb()  += -1.0*r2_bbbb_upper())
        (delta_r2_abab()  += r2_abab_lower())
        (delta_r2_abab()  += -1.0*r2_abab_upper());
      // clang-format on
      sch.execute();

      // step 11: Computer delta T,
      // Delta T = T_upper - T_lower

      // clang-format off
      sch(delta_t1_aa()    = 0)
        (delta_t1_bb()     = 0)
        (delta_t2_aaaa()   = 0)
        (delta_t2_bbbb()   = 0) 
        (delta_t2_abab()   = 0)     
        (delta_t1_aa()    += t1_aa_upper())
        (delta_t1_aa()    += -1.0*t1_aa_lower())
        (delta_t1_bb()    += t1_bb_upper())
        (delta_t1_bb()    += -1.0*t1_bb_lower())
        (delta_t2_aaaa()  += t2_aaaa_upper())
        (delta_t2_aaaa()  += -1.0*t2_aaaa_lower())
        (delta_t2_bbbb()  += t2_bbbb_upper())
        (delta_t2_bbbb()  += -1.0*t2_bbbb_lower())
        (delta_t2_abab()  += t2_abab_upper())
        (delta_t2_abab()  += -1.0*t2_abab_lower());
      // clang-format on
      sch.execute();

      // keep a copy of R_upper before modifying

      // clang-format off
      sch(r1_aa_upper_copy()  = 0)
        (r1_aa_upper_copy() += r1_aa_upper())
        (r1_bb_upper_copy()  = 0)
        (r1_bb_upper_copy() += r1_bb_upper())
        (r2_aaaa_upper_copy()= 0)
        (r2_bbbb_upper_copy()= 0)
        (r2_abab_upper_copy()= 0)
        (r2_aaaa_upper_copy()  += r2_aaaa_upper())
        (r2_bbbb_upper_copy()  += r2_bbbb_upper())
        (r2_abab_upper_copy()  += r2_abab_upper());
      // clang-format on
      sch.execute();

      // step 12: compute f_upper
      //  f_upper = t_upper-[t_aux+i*0.5*h*R_upper]

      //   double scale_factor2 =0.5*rt_step_size;

      scale_complex_ip(r1_aa_upper, scale_factor1, -scale_factor1);
      scale_complex_ip(r1_bb_upper, scale_factor1, -scale_factor1);
      scale_complex_ip(r2_aaaa_upper, scale_factor1, -scale_factor1);
      scale_complex_ip(r2_bbbb_upper, scale_factor1, -scale_factor1);
      scale_complex_ip(r2_abab_upper, scale_factor1, -scale_factor1);

      // std::cout << "scale_factor1 and rt_step_size" << std::endl;
      // std::cout << scale_factor1 << "\t" << rt_step_size << std::endl;

      // clang-format off
      sch(t1_aa_aux_copy()   = 0)
        (t1_bb_aux_copy()    = 0)
        (t2_aaaa_aux_copy()  = 0)
        (t2_bbbb_aux_copy()  = 0)
        (t2_abab_aux_copy()  = 0)
        (t1_aa_aux_copy()   += t1_aa_aux())
        (t1_bb_aux_copy()   += t1_bb_aux())
        (t2_aaaa_aux_copy() += t2_aaaa_aux())
        (t2_bbbb_aux_copy() += t2_bbbb_aux())
        (t2_abab_aux_copy() += t2_abab_aux());
      // clang-format on
      sch.execute();

      // (t_aux_copy =t_aux+r_upper)

      complex_copy_swap(ec, r1_aa_upper, t1_aa_aux_copy);
      complex_copy_swap(ec, r1_bb_upper, t1_bb_aux_copy);
      complex_copy_swap(ec, r2_aaaa_upper, t2_aaaa_aux_copy);
      complex_copy_swap(ec, r2_bbbb_upper, t2_bbbb_aux_copy);
      complex_copy_swap(ec, r2_abab_upper, t2_abab_aux_copy);

      // Compute f_upper

      // clang-format off
      sch(f1_aa_upper()     = 0)
	       (f1_bb_upper()     = 0)
         (f2_aaaa_upper()   = 0)
         (f2_bbbb_upper()   = 0)
         (f2_abab_upper()   = 0) 	 
         (f1_aa_upper()     = t1_aa_upper())
         (f1_aa_upper()    += -1.0*t1_aa_aux_copy())
         (f1_bb_upper()     = t1_bb_upper())
         (f1_bb_upper()    += -1.0*t1_bb_aux_copy())
         (f2_aaaa_upper()   = t2_aaaa_upper())
         (f2_aaaa_upper()  += -1.0*t2_aaaa_aux_copy())
         (f2_bbbb_upper()   = t2_bbbb_upper())
         (f2_bbbb_upper()  += -1.0*t2_bbbb_aux_copy())
         (f2_abab_upper()   = t2_abab_upper())
         (f2_abab_upper()  += -1.0*t2_abab_aux_copy());
      // clang-format on
      sch.execute();

      // step 13 Compute the denominator
      //  deno = \delta T +i*0.5*h*\delta R

      scale_complex_ip(delta_r1_aa, scale_factor1, -scale_factor1);
      scale_complex_ip(delta_r1_bb, scale_factor1, -scale_factor1);
      scale_complex_ip(delta_r2_aaaa, scale_factor1, -scale_factor1);
      scale_complex_ip(delta_r2_bbbb, scale_factor1, -scale_factor1);
      scale_complex_ip(delta_r2_abab, scale_factor1, -scale_factor1);

      // clang-format off
      sch (deno_aa()   = 0)
          (deno_bb()   = 0)
          (deno_aaaa() = 0)
          (deno_bbbb() = 0) 
          (deno_abab() = 0)  
          (deno_aa()   += delta_t1_aa())
          (deno_bb()   += delta_t1_bb())
          (deno_aaaa() += delta_t2_aaaa())
          (deno_bbbb() += delta_t2_bbbb())
          (deno_abab() += delta_t2_abab());
      // clang-format on
      sch.execute();

      // deno = deno+delta_r

      complex_copy_swap(ec, delta_r1_aa, deno_aa);
      complex_copy_swap(ec, delta_r1_bb, deno_bb);
      complex_copy_swap(ec, delta_r2_aaaa, deno_aaaa);
      complex_copy_swap(ec, delta_r2_bbbb, deno_bbbb);
      complex_copy_swap(ec, delta_r2_abab, deno_abab);

      // step 14
      secant_update_ip(ec, t1_aa, t1_aa_upper, f1_aa_upper, delta_t1_aa, deno_aa);
      secant_update_ip(ec, t1_bb, t1_bb_upper, f1_bb_upper, delta_t1_bb, deno_bb);

      secant_update_ip(ec, t2_aaaa, t2_aaaa_upper, f2_aaaa_upper, delta_t2_aaaa, deno_aaaa);
      secant_update_ip(ec, t2_bbbb, t2_bbbb_upper, f2_bbbb_upper, delta_t2_bbbb, deno_bbbb);
      secant_update_ip(ec, t2_abab, t2_abab_upper, f2_abab_upper, delta_t2_abab, deno_abab);

      // step 15
      //  compute the cumulant
      //    std::cout << "computing the cumulant" << std::endl;

      ccsd_e_os(sch, MO, CI, d_e, t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f1_se, chol3d_se);
      sch.execute(exhw, profile);

      /*      const T t1_aa_norm   = tamm::norm(t1_aa);
            const T t1_bb_norm   = tamm::norm(t1_bb);
            const T t2_aaaa_norm = tamm::norm(t2_aaaa);
            const T t2_abab_norm = tamm::norm(t2_abab);
            const T t2_bbbb_norm = tamm::norm(t2_bbbb);

            const T t1_aa_upper_norm   = tamm::norm(t1_aa_upper);
            const T t1_bb_upper_norm   = tamm::norm(t1_bb_upper);
            const T t2_aaaa_upper_norm = tamm::norm(t2_aaaa_upper);
            const T t2_abab_upper_norm = tamm::norm(t2_abab_upper);
            const T t2_bbbb_upper_norm = tamm::norm(t2_bbbb_upper);

            const T x1_1 = std::abs(t1_aa_upper_norm - t1_aa_norm);
            const T x1_2 = std::abs(t1_bb_upper_norm - t1_bb_norm);
            const T x2_1 = std::abs(t2_aaaa_upper_norm - t2_aaaa_norm);
            const T x2_2 = std::abs(t2_abab_upper_norm - t2_abab_norm);
            const T x2_3 = std::abs(t2_bbbb_upper_norm - t2_bbbb_norm);
      */

      // clang-format off
      sch (t1_aa_diff()    = t1_aa_upper())
          (t1_aa_diff()   += -1.0*t1_aa())
          (t1_bb_diff()    = t1_bb_upper())
          (t1_bb_diff()   += -1.0*t1_bb())	
          (t2_aaaa_diff()  = t2_aaaa_upper())
          (t2_aaaa_diff() += -1.0*t2_aaaa())
          (t2_bbbb_diff()  = t2_bbbb_upper())
          (t2_bbbb_diff() += -1.0*t2_bbbb()) 
          (t2_abab_diff()  = t2_abab_upper())
          (t2_abab_diff() += -1.0*t2_abab());
      
      if(debug) {
        sch(t1_full_diff()             = 0)
          (t1_full_diff(p3_va, h4_oa) += t1_aa_diff(p3_va, h4_oa))
          (t1_full_diff(p3_vb, h4_ob) += t1_bb_diff(p3_vb, h4_ob));
      }
      // clang-format on

      sch.execute();

      double x1_1 = std::abs(tamm::norm(t1_aa_diff));
      double x1_2 = std::abs(tamm::norm(t1_bb_diff));
      double x2_1 = std::abs(tamm::norm(t2_aaaa_diff));
      double x2_2 = std::abs(tamm::norm(t2_bbbb_diff));
      double x2_3 = std::abs(tamm::norm(t2_abab_diff));

      // double x1_1_r  =0.0;
      // double x1_1_im = 0.0;
      // double x1_2_r  = 0.0;
      // double x1_2_im = 0.0;
      // double x2_1_r  = 0.0;
      // double x2_1_im =0.0;
      // double x2_2_r  = 0.0;
      // double x2_2_im = 0.0;
      // double x2_3_r  =0.0;
      // double x2_3_im = 0.0;

      if(ec.print() && debug) {
        const auto de_energy        = get_scalar(d_e);
        auto       t1_full_diff_eig = tamm_to_eigen_matrix(t1_full_diff);

        std::vector<double> real_parts(sys_data.nocc * sys_data.nvir);
        std::vector<double> imag_parts(sys_data.nocc * sys_data.nvir);

        int counter = 0;

        for(int i = 0; i < sys_data.nvir; i++) {
          for(int j = 0; j < sys_data.nocc; j++) {
            std::complex<double> value = t1_full_diff_eig(i, j);

            real_parts[counter] = std::real(value);
            imag_parts[counter] = std::imag(value);

            counter++;
          }
        }
        const double threshold = 1e-12;
        std::cout << "Real Parts:" << std::endl;
        for(const auto& val: real_parts) {
          if(std::abs(val) > threshold) {
            std::cout << std::fixed << std::setprecision(10) << val << " ";
          }
        }
        std::cout << std::endl;

        std::cout << "Imaginary Parts:" << std::endl;
        for(const auto& val: imag_parts) {
          if(std::abs(val) > threshold) {
            std::cout << std::fixed << std::setprecision(10) << val << " ";
          }
        }
        std::cout << std::endl;

        // Calculate the Euclidean norm of the real parts
        double norm_real_parts = 0.0;
        for(const auto& val: real_parts) { norm_real_parts += val * val; }
        norm_real_parts = std::sqrt(norm_real_parts);

        // Calculate the Euclidean norm of the imaginary parts
        double norm_imag_parts = 0.0;
        for(const auto& val: imag_parts) { norm_imag_parts += val * val; }
        norm_imag_parts = std::sqrt(norm_imag_parts);

        // Print the results
        std::cout << "Norm of real parts: " << std::fixed << std::setprecision(10)
                  << norm_real_parts << std::endl;
        std::cout << "Norm of imaginary parts: " << std::fixed << std::setprecision(10)
                  << norm_imag_parts << std::endl;

        std::cout.width(6);
        std::cout << std::right << li + 1 << std::string(3, ' ');
        std::cout << std::fixed << std::setprecision(6) << rt_step_size << std::string(5, ' ');
        std::cout << std::fixed << std::setprecision(13) << de_energy << std::string(3, ' ');
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::endl;
      }

      // clang-format off
      sch(r1_aa_upper()    = 0)
         (r1_aa_upper()   += r1_aa_upper_copy())
         (r1_bb_upper()    = 0)
         (r1_bb_upper()   += r1_bb_upper_copy())
         (r2_aaaa_upper()  = 0)
         (r2_aaaa_upper() += r2_aaaa_upper_copy())
         (r2_bbbb_upper()  = 0)
         (r2_bbbb_upper() += r2_bbbb_upper_copy())
         (r2_abab_upper()  = 0)
         (r2_abab_upper() += r2_abab_upper_copy());
      // clang-format on
      sch.execute();

      //      std::cout << "t1_aa_norm" << std:: endl;
      //      std::cout << t1_aa_norm << std:: endl;

      // clang-format off
      sch(t1_aa_lower()   = 0)
         (t1_aa_lower()  += t1_aa_upper())
	       (t1_bb_lower()   = 0)
         (t1_bb_lower()  += t1_bb_upper())
         (t2_aaaa_lower() = 0)
	       (t2_aaaa_lower() += t2_aaaa_upper())
	       (t2_bbbb_lower() = 0)
         (t2_bbbb_lower() += t2_bbbb_upper())
	       (t2_abab_lower() = 0)
         (t2_abab_lower() += t2_abab_upper())
	       (t1_aa_upper()   = 0)
         (t1_aa_upper()  += t1_aa())
	       (t1_bb_upper()   = 0)
         (t1_bb_upper()  += t1_bb())
	       (t2_aaaa_upper() = 0)
         (t2_aaaa_upper() += t2_aaaa())
	       (t2_bbbb_upper() = 0)
         (t2_bbbb_upper() += t2_bbbb())
	       (t2_abab_upper()  = 0)
         (t2_abab_upper() += t2_abab())
	       (r1_aa_lower()   = 0)
         (r1_aa_lower()   = r1_aa_upper())
	       (r1_bb_lower()   = 0)
         (r1_bb_lower()   = r1_bb_upper())
	       (r2_aaaa_lower() = 0)
         (r2_aaaa_lower() = r2_aaaa_upper())
	       (r2_bbbb_lower() = 0)
         (r2_bbbb_lower() += r2_bbbb_upper())
	       (r2_abab_lower() = 0)
         (r2_abab_lower() += r2_abab_upper());
      // clang-format on
      sch.execute();

      ccsd_t1_os(sch, MO, CI, r1_aa_upper, r1_bb_upper, t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab,
                 f1_se, chol3d_se);
      ccsd_t2_os(sch, MO, CI, r2_aaaa_upper, r2_bbbb_upper, r2_abab_upper, t1_aa, t1_bb, t2_aaaa,
                 t2_bbbb, t2_abab, f1_se, chol3d_se, i0_t2_tmp);
      sch.execute(exhw, profile);

      const auto mt_end = std::chrono::high_resolution_clock::now();
      auto       mi_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((mt_end - mt_start)).count();

      if(ec.print()) {
        td_iteration_print(chem_env, li, rt_step_size, get_scalar(d_e), x1_1, x1_2, x2_1, x2_2,
                           x2_3, mi_time);
      }

      // step 16
      // variable time-steps
      // std::cout << "for the next time-step, it will be printed again before next time starts" <<
      // std::endl; std::cout << std::fixed << std::setprecision(6) << rt_step_size1 << std::endl;
      //     if (li >= 6) {
      //          rt_step_size1 = h_red_fac * rt_step_size;
      //            } else if (li <= 2) {
      //            rt_step_size1 = h_inc_fac * rt_step_size;
      //             }

      //    if (rt_step_size > h_max) {
      //            rt_step_size1 = h_max;
      //             }
      //  std::cout << "li" << li+1 << std::endl;
      //  std::cout << "rt_step_size1" << li+1 << std::endl;

      // ec.pg().broadcast(&rt_step_size, 0);

      // step 17
      if((x1_1 < thresh) && (x1_2 < thresh) && (x2_1 < thresh) && (x2_2 < thresh) &&
         (x2_3 < thresh)) {
        li++;
        break;
      }

    } // microiter loop
      //

    //      li = li+1;

    if(ec.print() && debug)
      std::cout << "Number of micro-iterations to converge: " << li << std::endl;
    if(li >= 6) {
      rt_step_size = h_red_fac * rt_step_size;
      if(ec.print())
        std::cout << "rt_step_size is decreased for the next time-step by a factor of: "
                  << h_red_fac << std::endl;
    }
    else if(li <= 3) {
      if(ec.print())
        std::cout << "rt_step_size is increased for the next time-step by a factor of: "
                  << h_inc_fac << std::endl;
      rt_step_size = h_inc_fac * rt_step_size;
    }

    if(rt_step_size > h_max) {
      if(ec.print()) std::cout << "rt_step_size reached the maximum value:" << std::endl;
      rt_step_size = h_max;
    }

    // rt_step_size = rt_step_size1;

    // std::cout << "for the next time-step, it is:" << std::endl;
    // std::cout << std::fixed << std::setprecision(6) << rt_step_size << std::endl;

    // step 18

    if(writet && ((titer + 1) % writet_iter == 0)) {
      // only advance the restart timestep once the tensors are committed; on failure the
      // previous checkpoint (tensors + timestep file) stays consistent
      const bool restart_written = write_to_disk(ec, restart_tensors, restart_files, profile);
      if(ec.print()) {
        if(!restart_written)
          std::cout << "[RT-EOM-CC] WARNING: failed to write restart tensors at timestep "
                    << titer + 1 << "; keeping the previous checkpoint" << std::endl;
        else {
          std::ofstream out(ts_file, std::ios::out);
          out << titer + 1 << " " << total_time << " " << rt_step_size << std::endl;
          out.close();
        }
      }
    }

  } // end timestep loop
    //

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
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                   chol3d_vv);
  sch.deallocate(d_e, _a01V);
  sch.deallocate(t1_aa, t1_bb, t1_aa_old, t1_bb_old, t1_aa_aux, t1_bb_aux);
  sch.deallocate(t2_aaaa, t2_bbbb, t2_abab, t2_aaaa_old, t2_bbbb_old, t2_abab_old, t2_aaaa_aux,
                 t2_bbbb_aux, t2_abab_aux);
  sch.deallocate(r1_aa, r1_bb, r1_aa_old, r1_bb_old, r2_aaaa, r2_bbbb, r2_abab, r2_aaaa_old,
                 r2_bbbb_old, r2_abab_old);
  sch.deallocate(t1_aa_aux_copy, t1_bb_aux_copy, t2_aaaa_aux_copy, t2_bbbb_aux_copy,
                 t2_abab_aux_copy);
  sch.deallocate(t1_aa_diff, t1_bb_diff, t2_aaaa_diff, t2_bbbb_diff, t2_abab_diff, t1_full_diff);
  sch.deallocate(r1_aa_old_copy, r1_bb_old_copy, r2_aaaa_old_copy, r2_bbbb_old_copy,
                 r2_abab_old_copy);
  sch.deallocate(r1_aa_upper_copy, r1_bb_upper_copy, r2_aaaa_upper_copy, r2_bbbb_upper_copy,
                 r2_abab_upper_copy);
  sch.deallocate(t1_aa_lower, t1_bb_lower, t2_aaaa_lower, t2_bbbb_lower, t2_abab_lower);
  sch.deallocate(t1_aa_upper, t1_bb_upper, t2_aaaa_upper, t2_bbbb_upper, t2_abab_upper);
  sch.deallocate(r1_aa_lower, r1_bb_lower, r2_aaaa_lower, r2_bbbb_lower, r2_abab_lower);
  sch.deallocate(r1_aa_upper, r1_bb_upper, r2_aaaa_upper, r2_bbbb_upper, r2_abab_upper);
  sch.deallocate(delta_r1_aa, delta_r1_bb, delta_r2_aaaa, delta_r2_bbbb, delta_r2_abab);
  sch.deallocate(delta_t1_aa, delta_t1_bb, delta_t2_aaaa, delta_t2_bbbb, delta_t2_abab);
  sch.deallocate(f1_aa_upper, f1_bb_upper, f2_aaaa_upper, f2_bbbb_upper, f2_abab_upper);
  sch.deallocate(deno_aa, deno_bb, deno_aaaa, deno_bbbb, deno_abab);
  sch.execute();
}

void rt_eom_cd_ccsd_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T             = double;
  using ComplexTensor = Tensor<CCEType>;

  const auto  rank     = ec.pg().rank();
  SystemData& sys_data = chem_env.sys_data;

  // phase 1: reference (RHF) SCF; phase 2: core-hole (UHF, noscf) SCF + cholesky
  const json ref_scf_results = rteom_reference_scf(ec, chem_env);
  cholesky_2e::cholesky_2e_driver(ec, chem_env);
  // the core-hole SCF pass only sets up the environment; report the reference SCF instead
  chem_env.sys_data.results["output"]["SCF"] = ref_scf_results;

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
