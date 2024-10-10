/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_cc2_os.hpp"

namespace cc2_os {

using CCEType = double;
CCSE_Tensors<CCEType> _a021_os;
TiledIndexSpace       o_alpha_os, v_alpha_os, o_beta_os, v_beta_os;

Tensor<CCEType>       _a01V_os, _a02V_os, _a007V_os;
CCSE_Tensors<CCEType> _a01_os, _a02_os, _a03_os, _a04_os, _a05_os, _a06_os, _a001_os, _a004_os,
  _a006_os, _a008_os, _a009_os, _a017_os, _a019_os, _a020_os; //_a022
};                                                            // namespace cc2_os

template<typename T>
void cc2_os::cc2_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                      Tensor<T>& de, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                      std::vector<CCSE_Tensors<T>>& f1_se,
                      std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind]                = CI.labels<1>("all");
  auto [p1_va, p2_va, p3_va] = v_alpha_os.labels<3>("all");
  auto [p1_vb, p2_vb, p3_vb] = v_beta_os.labels<3>("all");
  auto [h3_oa, h4_oa, h6_oa] = o_alpha_os.labels<3>("all");
  auto [h3_ob, h4_ob, h6_ob] = o_beta_os.labels<3>("all");

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
  (_a01V_os(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h4_oa, p3_va, cind), 
  "_a01V_os(cind)                     = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h4_oa, p3_va, cind)")
  (_a02_os("aa")(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov("aa")(h6_oa, p3_va, cind), 
  "_a02_os( aa )(h4_oa, h6_oa, cind)  = t1_aa(p3_va, h4_oa) * chol3d_ov( aa )(h6_oa, p3_va, cind)")
  (_a03_os("aa")(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
  "_a03_os( aa )(h4_oa, p2_va, cind)  = t2_aaaa(p1_va, p2_va, h3_oa, h4_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
  (_a03_os("aa")(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
  "_a03_os( aa )(h4_oa, p2_va, cind) += t2_abab(p2_va, p1_vb, h4_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
  (_a01V_os(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h4_ob, p3_vb, cind), 
  "_a01V_os(cind)                    += t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h4_ob, p3_vb, cind)")
  (_a02_os("bb")(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov("bb")(h6_ob, p3_vb, cind), 
  "_a02_os( bb )(h4_ob, h6_ob, cind)  = t1_bb(p3_vb, h4_ob) * chol3d_ov( bb )(h6_ob, p3_vb, cind)")
  (_a03_os("bb")(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind), 
  "_a03_os( bb )(h4_ob, p2_vb, cind)  = t2_bbbb(p1_vb, p2_vb, h3_ob, h4_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
  (_a03_os("bb")(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind), 
  "_a03_os( bb )(h4_ob, p2_vb, cind) += t2_abab(p1_va, p2_vb, h3_oa, h4_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
  (de()                            =  0.5 * _a01V_os() * _a01V_os(), 
  "de()                            =  0.5 * _a01V_os() * _a01V_os()")
  (de()                           += -0.5 * _a02_os("aa")(h4_oa, h6_oa, cind) * _a02_os("aa")(h6_oa, h4_oa, cind), 
  "de()                           += -0.5 * _a02_os( aa )(h4_oa, h6_oa, cind) * _a02_os( aa )(h6_oa, h4_oa, cind)")
  (de()                           += -0.5 * _a02_os("bb")(h4_ob, h6_ob, cind) * _a02_os("bb")(h6_ob, h4_ob, cind), 
  "de()                           += -0.5 * _a02_os( bb )(h4_ob, h6_ob, cind) * _a02_os( bb )(h6_ob, h4_ob, cind)")
  (de()                           +=  0.5 * _a03_os("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h4_oa, p1_va, cind), 
  "de()                           +=  0.5 * _a03_os( aa )(h4_oa, p1_va, cind) * chol3d_ov( aa )(h4_oa, p1_va, cind)")
  (de()                           +=  0.5 * _a03_os("bb")(h4_ob, p1_vb, cind) * chol3d_ov("bb")(h4_ob, p1_vb, cind), 
  "de()                           +=  0.5 * _a03_os( bb )(h4_ob, p1_vb, cind) * chol3d_ov( bb )(h4_ob, p1_vb, cind)")
  (de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov("aa")(h3_oa, p1_va),
  "de()                           +=  1.0 * t1_aa(p1_va, h3_oa) * f1_ov( aa )(h3_oa, p1_va)") // NEW TERM
  (de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov("bb")(h3_ob, p1_vb),
  "de()                           +=  1.0 * t1_bb(p1_vb, h3_ob) * f1_ov( bb )(h3_ob, p1_vb)") // NEW TERM
  ;
  // clang-format on
}

template<typename T>
void cc2_os::cc2_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                       CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                       std::vector<CCSE_Tensors<T>>& f1_se,
                       std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind]                       = CI.labels<1>("all");
  auto [p2]                         = MO.labels<1>("virt");
  auto [h1]                         = MO.labels<1>("occ");
  auto [p1_va, p2_va, p3_va]        = v_alpha_os.labels<3>("all");
  auto [p1_vb, p2_vb, p3_vb]        = v_beta_os.labels<3>("all");
  auto [h1_oa, h2_oa, h3_oa, h7_oa] = o_alpha_os.labels<4>("all");
  auto [h1_ob, h2_ob, h3_ob, h7_ob] = o_beta_os.labels<4>("all");

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
    (_a01_os("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind), 
    "_a01_os( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a01_os("bb")(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov("bb")(h2_ob, p1_vb, cind), 
    "_a01_os( bb )(h2_ob, h1_ob, cind)  =  1.0 * t1_bb(p1_vb, h1_ob) * chol3d_ov( bb )(h2_ob, p1_vb, cind)")                 // ovm
    (_a02V_os(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov("aa")(h3_oa, p3_va, cind), 
    "_a02V_os(cind)                     =  1.0 * t1_aa(p3_va, h3_oa) * chol3d_ov( aa )(h3_oa, p3_va, cind)")                 // ovm
    (_a02V_os(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov("bb")(h3_ob, p3_vb, cind), 
    "_a02V_os(cind)                    +=  1.0 * t1_bb(p3_vb, h3_ob) * chol3d_ov( bb )(h3_ob, p3_vb, cind)")                 // ovm
    (_a06_os("aa")(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
    "_a06_os( aa )(p1_va, h1_oa, cind)  =  1.0 * t2_aaaa(p1_va, p3_va, h2_oa, h1_oa) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
    (_a06_os("aa")(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
    "_a06_os( aa )(p1_va, h1_oa, cind) += -1.0 * t2_abab(p1_va, p3_vb, h1_oa, h2_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
    (_a06_os("bb")(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov("aa")(h2_oa, p3_va, cind), 
    "_a06_os( bb )(p1_vb, h1_ob, cind)  = -1.0 * t2_abab(p3_va, p1_vb, h2_oa, h1_ob) * chol3d_ov( aa )(h2_oa, p3_va, cind)") // o2v2m
    (_a06_os("bb")(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov("bb")(h2_ob, p3_vb, cind), 
    "_a06_os( bb )(p1_vb, h1_ob, cind) +=  1.0 * t2_bbbb(p1_vb, p3_vb, h2_ob, h1_ob) * chol3d_ov( bb )(h2_ob, p3_vb, cind)") // o2v2m
    (_a04_os("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa), 
    "_a04_os( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04_os("bb")(h2_ob, h1_ob)        = -1.0 * f1_oo("bb")(h2_ob, h1_ob), 
    "_a04_os( bb )(h2_ob, h1_ob)        = -1.0 * f1_oo( bb )(h2_ob, h1_ob)") // MOVED TERM
    (_a04_os("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06_os("aa")(p1_va, h1_oa, cind), 
    "_a04_os( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06_os( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04_os("bb")(h2_ob, h1_ob)       +=  1.0 * chol3d_ov("bb")(h2_ob, p1_vb, cind) * _a06_os("bb")(p1_vb, h1_ob, cind), 
    "_a04_os( bb )(h2_ob, h1_ob)       +=  1.0 * chol3d_ov( bb )(h2_ob, p1_vb, cind) * _a06_os( bb )(p1_vb, h1_ob, cind)")   // o2vm
    (_a04_os("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va), 
    "_a04_os( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (_a04_os("bb")(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov("bb")(h2_ob, p1_vb), 
    "_a04_os( bb )(h2_ob, h1_ob)       += -1.0 * t1_bb(p1_vb, h1_ob) * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_os("aa")(h2_oa, h1_oa), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04_os( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04_os("bb")(h2_ob, h1_ob), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p2_vb, h2_ob) * _a04_os( bb )(h2_ob, h1_ob)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo("aa")(p1_va, h2_oa, cind) * _a02V_os(cind), 
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_vo( aa )(p1_va, h2_oa, cind) * _a02V_os(cind)")                      // ovm
    (i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo("bb")(p1_vb, h2_ob, cind) * _a02V_os(cind), 
    "i0_bb(p1_vb, h2_ob)            +=  1.0 * chol3d_vo( bb )(p1_vb, h2_ob, cind) * _a02V_os(cind)")                      // ovm
    (_a05_os("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h3_oa, p1_va, cind) * _a01_os("aa")(h2_oa, h3_oa, cind), 
    "_a05_os( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h3_oa, p1_va, cind) * _a01_os( aa )(h2_oa, h3_oa, cind)")   // o2vm
    (_a05_os("bb")(h2_ob, p1_vb)        = -1.0 * chol3d_ov("bb")(h3_ob, p1_vb, cind) * _a01_os("bb")(h2_ob, h3_ob, cind), 
    "_a05_os( bb )(h2_ob, p1_vb)        = -1.0 * chol3d_ov( bb )(h3_ob, p1_vb, cind) * _a01_os( bb )(h2_ob, h3_ob, cind)")   // o2vm
    (_a05_os("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va), 
    "_a05_os( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (_a05_os("bb")(h2_ob, p1_vb)       +=  1.0 * f1_ov("bb")(h2_ob, p1_vb), 
    "_a05_os( bb )(h2_ob, p1_vb)       +=  1.0 * f1_ov( bb )(h2_ob, p1_vb)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05_os("aa")(h2_oa, p1_va), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_aaaa(p1_va, p2_va, h2_oa, h1_oa) * _a05_os( aa )(h2_oa, p1_va)")         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05_os("aa")(h2_oa, p1_va), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_abab(p1_va, p2_vb, h2_oa, h1_ob) * _a05_os( aa )(h2_oa, p1_va)")         // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05_os("bb")(h2_ob, p1_vb), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a05_os( bb )(h2_ob, p1_vb)")         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05_os("bb")(h2_ob, p1_vb), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t2_bbbb(p1_vb, p2_vb, h2_ob, h1_ob) * _a05_os( bb )(h2_ob, p1_vb)")         // o2v
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06_os("aa")(p1_va, h1_oa, cind), 
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06_os( aa )(p1_va, h1_oa, cind)")   // ov2m
    (i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv("bb")(p2_vb, p1_vb, cind) * _a06_os("bb")(p1_vb, h1_ob, cind), 
    "i0_bb(p2_vb, h1_ob)            += -1.0 * chol3d_vv( bb )(p2_vb, p1_vb, cind) * _a06_os( bb )(p1_vb, h1_ob, cind)")   // ov2m
    (_a06_os("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind), 
    "_a06_os( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (_a06_os("bb")(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv("bb")(p2_vb, p1_vb, cind), 
    "_a06_os( bb )(p2_vb, h2_ob, cind) += -1.0 * t1_bb(p1_vb, h2_ob) * chol3d_vv( bb )(p2_vb, p1_vb, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06_os("aa")(p1_va, h2_oa, cind) * _a02V_os(cind), 
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06_os( aa )(p1_va, h2_oa, cind) * _a02V_os(cind)")                           // ovm
    (i0_bb(p1_vb, h2_ob)            += -1.0 * _a06_os("bb")(p1_vb, h2_ob, cind) * _a02V_os(cind), 
    "i0_bb(p1_vb, h2_ob)            += -1.0 * _a06_os( bb )(p1_vb, h2_ob, cind) * _a02V_os(cind)")                           // ovm
    (_a06_os("aa")(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V_os(cind), 
    "_a06_os( aa )(p2_va, h3_oa, cind) += -1.0 * t1_aa(p2_va, h3_oa) * _a02V_os(cind)")                                      // ovm
    (_a06_os("bb")(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V_os(cind), 
    "_a06_os( bb )(p2_vb, h3_ob, cind) += -1.0 * t1_bb(p2_vb, h3_ob) * _a02V_os(cind)")                                      // ovm
    (_a06_os("aa")(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_os("aa")(h2_oa, h3_oa, cind), 
    "_a06_os( aa )(p2_va, h3_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01_os( aa )(h2_oa, h3_oa, cind)")                   // o2vm
    (_a06_os("bb")(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01_os("bb")(h2_ob, h3_ob, cind), 
    "_a06_os( bb )(p2_vb, h3_ob, cind) +=  1.0 * t1_bb(p2_vb, h2_ob) * _a01_os( bb )(h2_ob, h3_ob, cind)")                   // o2vm
    (_a01_os("aa")(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h3_oa, h1_oa, cind), 
    "_a01_os( aa )(h3_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h3_oa, h1_oa, cind)")                                    // o2m
    (_a01_os("bb")(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo("bb")(h3_ob, h1_ob, cind), 
    "_a01_os( bb )(h3_ob, h1_ob, cind) +=  1.0 * chol3d_oo( bb )(h3_ob, h1_ob, cind)")                                    // o2m        
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_os("aa")(h3_oa, h1_oa, cind) * _a06_os("aa")(p2_va, h3_oa, cind), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01_os( aa )(h3_oa, h1_oa, cind) * _a06_os( aa )(p2_va, h3_oa, cind)")        // o2vm
  //  (i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo("aa")(h7_oa, h1_oa), 
  //  "i0_aa(p2_va, h1_oa)         += -1.0 * t1_aa(p2_va, h7_oa) * f1_oo( aa )(h7_oa, h1_oa)") // MOVED ABOVE         // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv("aa")(p2_va, p3_va), 
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p3_va, h1_oa) * f1_vv( aa )(p2_va, p3_va)")                        // ov2
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01_os("bb")(h3_ob, h1_ob, cind) * _a06_os("bb")(p2_vb, h3_ob, cind), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * _a01_os( bb )(h3_ob, h1_ob, cind) * _a06_os( bb )(p2_vb, h3_ob, cind)")        // o2vm
  //  (i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo("bb")(h7_ob, h1_ob), 
  //  "i0_bb(p2_vb, h1_ob)         += -1.0 * t1_bb(p2_vb, h7_ob) * f1_oo( bb )(h7_ob, h1_ob)") // MOVED ABOVE         // o2v
    (i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv("bb")(p2_vb, p3_vb), 
    "i0_bb(p2_vb, h1_ob)            +=  1.0 * t1_bb(p3_vb, h1_ob) * f1_vv( bb )(p2_vb, p3_vb)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void cc2_os::cc2_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                       CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                       std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se,
                       CCSE_Tensors<T>& i0tmp, Tensor<T>& d_f1, Tensor<T>& cv3d, Tensor<T>& res_2) {
  auto [cind]                                     = CI.labels<1>("all");
  auto [p3, p4, a, b, c, d, e]                    = MO.labels<7>("virt");
  auto [h1, h2, i, j, k, l, m]                    = MO.labels<7>("occ");
  auto [p1_va, p2_va, p3_va, p4_va, p5_va, p8_va] = v_alpha_os.labels<6>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb, p6_vb, p8_vb] = v_beta_os.labels<6>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa, h7_oa, h9_oa] = o_alpha_os.labels<6>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob, h8_ob, h9_ob] = o_beta_os.labels<6>("all");

  Tensor<T> i0_aaaa = r2("aaaa");
  Tensor<T> i0_abab = r2("abab");
  Tensor<T> i0_bbbb = r2("bbbb");

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

  // auto hw   = sch.ec().exhw();
  // auto rank = sch.ec().pg().rank();

  Tensor<T> ph{{a, i, cind}, {1, 1}};
  Tensor<T> ph_1{{a, i, cind}, {1, 1}};
  Tensor<T> ph_2{{a, i, cind}, {1, 1}};
  Tensor<T> ph_3{{a, i, cind}, {1, 1}};
  // Tensor<T> ph_4{{a, i, cind}, {1, 1}};
  Tensor<T> pp{{a, b}, {1, 1}};
  Tensor<T> hh{{i, j, cind}, {1, 1}};
  Tensor<T> hh_1{{i, j}, {1, 1}};

  Tensor<T> t_1{{a, i}, {1, 1}};
  Tensor<T> t_2{{a, b, i, j}, {2, 2}};

  Tensor<T> t2_baba{v_beta_os, v_alpha_os, o_beta_os, o_alpha_os};
  Tensor<T> t2_abba{v_alpha_os, v_beta_os, o_beta_os, o_alpha_os};
  Tensor<T> t2_baab{v_beta_os, v_alpha_os, o_alpha_os, o_beta_os};

  sch.allocate(t2_baba, t2_abba, t2_baab, t_1, t_2).execute();

  // clang-format off
  sch
    (t2_baba(p2_vb,p1_va,h4_ob,h3_oa) =        t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_abba(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_baab(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))

    (t_1(p1_va,h3_oa)             = t1_aa(p1_va,h3_oa))
    (t_1(p1_vb,h3_ob)             = t1_bb(p1_vb,h3_ob))
    (t_2(p1_va,p2_va,h3_oa,h4_oa) = t2_aaaa(p1_va,p2_va,h3_oa,h4_oa))
    (t_2(p1_va,p2_vb,h3_oa,h4_ob) = t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t_2(p1_vb,p2_vb,h3_ob,h4_ob) = t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob))

    (t_2(p1_vb,p2_va,h3_ob,h4_oa) = t2_baba(p1_vb,p2_va,h3_ob,h4_oa))
    (t_2(p1_va,p2_vb,h3_ob,h4_oa) = t2_abba(p1_va,p2_vb,h3_ob,h4_oa))
    (t_2(p1_vb,p2_va,h3_oa,h4_ob) = t2_baab(p1_vb,p2_va,h3_oa,h4_ob))
    .deallocate(t2_baba,t2_abba,t2_baab)
    .execute();
  // clang-format on

  Tensor<T> i0{{a, b, i, j}, {2, 2}};
  Tensor<T> pphh{{a, b, i, j}, {2, 2}};
  sch.allocate(ph, ph_1, ph_2, ph_3, hh, hh_1, pp, i0, pphh).execute();

  // clang-format off
  sch(i0(a, b, i, j)                 =                     res_2(a, b, i, j))
      (pp(b, c)                       =                            d_f1(b, c))
      (pp(b, c)                      += -1.0*   d_f1(k, c) *        t_1(b, k))      
      (pphh(a, b, i, j)               = pp(b, c)           *  t_2(a, c, i, j))
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                += -1.0*                pphh(b, a, i, j))
      (hh_1(k, j)                     = d_f1(k, c)         *        t_1(c, j))
      (hh_1(k, j)                    +=                            d_f1(k, j))
      (pphh(a, b, i, j)               = hh_1(k, j)         *  t_2(a, b, i, k))
      (i0(a, b, i, j)                += -1.0*                pphh(a, b, i, j))
      (i0(a, b, i, j)                +=                      pphh(a, b, j, i))
      (ph(b, j, cind)                 = cv3d(b, c, cind)   *        t_1(c, j))  // 4
      (ph_2(b, j, cind)               =                        ph(b, j, cind))
      (pphh(a, b, i, j)               = ph(b, j, cind)     * cv3d(a, i, cind))
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                +=-1.0                * pphh(b, a, i, j))
      (i0(a, b, i, j)                +=-1.0                * pphh(a, b, j, i))
      (i0(a, b, i, j)                +=                      pphh(b, a, j, i))
      (ph_1(b, i, cind)               = cv3d(k, i, cind)   *        t_1(b, k))
      (ph_3(b, i, cind)               =                      ph_1(b, i, cind))
      (pphh(a, b, i, j)               = ph(a, j, cind)     * ph_1(b, i, cind))
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, j, i))
      (i0(a, b, i, j)                += -1.0               * pphh(b, a, i, j))
      (i0(a, b, i, j)                +=                      pphh(b, a, j, i))
      (pphh(a, b, i, j)               = ph(a, i, cind)     *   ph(b, j, cind))  //8
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                +=-1.0                * pphh(a, b, j, i))
      (pphh(a, b, i, j)               = ph_1(b, j, cind)   * cv3d(a, i, cind))   // 5
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, i, j))
      (i0(a, b, i, j)                +=                      pphh(a, b, j, i))
      (i0(a, b, i, j)                +=                      pphh(b, a, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(b, a, j, i))
      (pphh(a, b, i, j)               = ph_1(a, i, cind)   * ph_1(b, j, cind))   //9
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, j, i))
      (hh(k, j, cind)                 = cv3d(k, c, cind)   *        t_1(c, j))
      (ph(b, j, cind)                 = hh(k, j, cind)     *        t_1(b, k))
      (pphh(a, b, i, j)               = cv3d(a, i, cind)   *   ph(b, j, cind))
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, i, j))
      (i0(a, b, i, j)                +=                      pphh(a, b, j, i))
      (i0(a, b, i, j)                +=                      pphh(b, a, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(b, a, j, i))
      (pphh(a, b, i, j)               = ph(b, j, cind)     * ph_2(a, i, cind))   //11
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, i, j))
      (i0(a, b, i, j)                +=                      pphh(a, b, j, i))
      (i0(a, b, i, j)                +=                      pphh(b, a, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(b, a, j, i))
      (pphh(a, b, i, j)               = ph(a, i, cind)     * ph_3(b, j, cind))  //12
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, j, i))
      (i0(a, b, i, j)                += -1.0               * pphh(b, a, i, j))
      (i0(a, b, i, j)                +=                      pphh(b, a, j, i))
      (pphh(a, b, i, j)               = ph(a, i, cind)     *   ph(b, j, cind))  // 13
      (i0(a, b, i, j)                +=                      pphh(a, b, i, j))
      (i0(a, b, i, j)                += -1.0               * pphh(a, b, j, i))
    ;

 sch(i0_aaaa(p3_va,p4_va, h1_oa, h2_oa) = i0(p3_va,p4_va, h1_oa, h2_oa))
    (i0_abab(p3_va,p4_vb, h1_oa, h2_ob) = i0(p3_va,p4_vb, h1_oa, h2_ob))
    (i0_bbbb(p3_vb,p4_vb, h1_ob, h2_ob) = i0(p3_vb,p4_vb, h1_ob, h2_ob));

  // clang-format on
  sch.deallocate(ph_2, ph_3, hh, pp, hh_1, i0, pphh, t_1, t_2).execute();
}

template<typename T>
std::tuple<double, double> cc2_os::cd_cc2_os_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, Tensor<T>& d_r1, Tensor<T>& d_r2,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, bool cc2_restart,
  std::string cc2_fp, bool computeTData) {
  SystemData& sys_data    = chem_env.sys_data;
  int         maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  int         ndiis       = chem_env.ioptions.ccsd_options.ndiis;
  double      thresh      = chem_env.ioptions.ccsd_options.threshold;
  bool        writet      = chem_env.ioptions.ccsd_options.writet;
  int         writet_iter = chem_env.ioptions.ccsd_options.writet_iter;
  double      zshiftl     = chem_env.ioptions.ccsd_options.lshift;
  bool        profile     = chem_env.ioptions.ccsd_options.profile_ccsd;
  double      residual    = 0.0;
  double      energy      = 0.0;
  int         niter       = 0;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);

  std::string t1file = cc2_fp + ".t1amp";
  std::string t2file = cc2_fp + ".t2amp";

  std::cout.precision(15);

  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const int              otiles  = O.num_tiles();
  const int              vtiles  = V.num_tiles();
  const int              oatiles = MO("occ_alpha").num_tiles();
  // const int              obtiles = MO("occ_beta").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();
  // const int              vbtiles = MO("virt_beta").num_tiles();

  o_alpha_os = {MO("occ"), range(oatiles)};
  v_alpha_os = {MO("virt"), range(vatiles)};
  o_beta_os  = {MO("occ"), range(oatiles, otiles)};
  v_beta_os  = {MO("virt"), range(vatiles, vtiles)};

  auto [cind]                       = CI.labels<1>("all");
  auto [p1_va, p2_va, p3_va, p4_va] = v_alpha_os.labels<4>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta_os.labels<4>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha_os.labels<4>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob] = o_beta_os.labels<4>("all");

  Tensor<T>             d_e{};
  CCSE_Tensors<CCEType> r1_vo, r2_vvoo; // r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb;

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

  r1_vo   = CCSE_Tensors<T>{MO, {V, O}, "r1", {"aa", "bb"}};
  r2_vvoo = CCSE_Tensors<T>{MO, {V, V, O, O}, "r2", {"aaaa", "abab", "bbbb"}};

  _a004_os = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004_os", {"aaaa", "abab", "bbbb"}};

  // Energy intermediates
  _a01V_os = {CI};
  _a02_os  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02_os", {"aa", "bb"}};
  _a03_os  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03_os", {"aa", "bb"}};

  // T1
  _a02V_os = {CI};
  _a01_os  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01_os", {"aa", "bb"}};
  _a04_os  = CCSE_Tensors<T>{MO, {O, O}, "_a04_os", {"aa", "bb"}};
  _a05_os  = CCSE_Tensors<T>{MO, {O, V}, "_a05_os", {"aa", "bb"}};
  _a06_os  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06_os", {"aa", "bb"}};

  // T2
  _a007V_os = {CI};
  _a001_os  = CCSE_Tensors<T>{MO, {V, V}, "_a001_os", {"aa", "bb"}};
  _a006_os  = CCSE_Tensors<T>{MO, {O, O}, "_a006_os", {"aa", "bb"}};
  _a008_os  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008_os", {"aa", "bb"}};
  _a009_os  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009_os", {"aa", "bb"}};
  _a017_os  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017_os", {"aa", "bb"}};
  _a021_os  = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021_os", {"aa", "bb"}};

  _a019_os = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019_os", {"aaaa", "abab", "bbbb"}};
  // _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"aaaa", "abab", "bbbb"}};
  _a020_os =
    CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020_os", {"aaaa", "abab", "baab", "abba", "baba", "bbbb"}};

  CCSE_Tensors<CCEType> i0_t2_tmp{MO, {V, V, O, O}, "i0_t2_tmp", {"aaaa", "bbbb"}};

  double total_cc2_mem = sum_tensor_sizes(d_t1, d_t2, d_f1, d_r1, d_r2, cv3d, d_e, _a01V_os) +
                         CCSE_Tensors<T>::sum_tensor_sizes_list(r1_vo, r2_vvoo, t1_vo, t2_vvoo) +
                         CCSE_Tensors<T>::sum_tensor_sizes_list(
                           f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo, chol3d_vv) +
                         CCSE_Tensors<T>::sum_tensor_sizes_list(_a02_os, _a03_os);

  for(size_t ri = 0; ri < d_r1s.size(); ri++)
    total_cc2_mem += sum_tensor_sizes(d_r1s[ri], d_r2s[ri], d_t1s[ri], d_t2s[ri]);

  // Intermediates
  // const double v4int_size = CCSE_Tensors<T>::sum_tensor_sizes_list(_a022);
  double total_cc2_mem_tmp =
    sum_tensor_sizes(_a02V_os, _a007V_os) /*+ v4int_size */ +
    CCSE_Tensors<T>::sum_tensor_sizes_list(i0_t2_tmp, _a01_os, _a04_os, _a05_os, _a06_os, _a001_os,
                                           _a004_os, _a006_os, _a008_os, _a009_os, _a017_os,
                                           _a019_os, _a020_os, _a021_os);

  if(!cc2_restart) total_cc2_mem += total_cc2_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for Open Shell Cholesky CC2 calculation: " << std::fixed
              << std::setprecision(2) << total_cc2_mem << " GiB" << std::endl;
  }
  check_memory_requirements(ec, total_cc2_mem);

  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  sch.allocate(d_e, _a01V_os);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                 chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  CCSE_Tensors<T>::allocate_list(sch, _a02_os, _a03_os);
  sch.execute();

  const int pcore = chem_env.ioptions.ccsd_options.pcore - 1; // 0-based indexing
  if(pcore >= 0) {
    const auto timer_start = std::chrono::high_resolution_clock::now();

    TiledIndexSpace mo_ut{IndexSpace{range(0, MO.max_num_indices())}, 1};
    TiledIndexSpace cv3d_occ{mo_ut, range(0, O.max_num_indices())};
    TiledIndexSpace cv3d_virt{mo_ut, range(O.max_num_indices(), MO.max_num_indices())};

    auto [h1, h2, h3, h4] = cv3d_occ.labels<4>("all");
    auto [p1, p2, p3, p4] = cv3d_virt.labels<4>("all");

    Tensor<T> d_f1_ut = redistribute_tensor<T>(d_f1, (TiledIndexSpaceVec){mo_ut, mo_ut});
    Tensor<T> cv3d_ut = redistribute_tensor<T>(cv3d, (TiledIndexSpaceVec){mo_ut, mo_ut, CI});

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

    sch.deallocate(d_f1, cv3d_ut).execute();

    d_f1 = redistribute_tensor<T>(d_f1_ut, (TiledIndexSpaceVec){MO, MO}, {1, 1});
    sch.deallocate(d_f1_ut).execute();

    p_evl_sorted = tamm::diagonal(d_f1);

    const auto timer_end = std::chrono::high_resolution_clock::now();
    auto       f1_rctime =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();
    if(ec.print())
      std::cout << "Time to reconstruct Fock matrix: " << std::fixed << std::setprecision(2)
                << f1_rctime << " secs" << std::endl;
  }

  print_ccsd_header(ec.print(), "CC2");

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
    ;
  // clang-format on

  if(!cc2_restart) {
    // allocate all intermediates
    sch.allocate(_a02V_os, _a007V_os);
    CCSE_Tensors<T>::allocate_list(sch, _a004_os, i0_t2_tmp, _a01_os, _a04_os, _a05_os, _a06_os,
                                   _a001_os, _a006_os, _a008_os, _a009_os, _a017_os, _a019_os,
                                   _a020_os, _a021_os); // _a022
    sch.execute();
    // clang-format off
    sch
      (_a004_os("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("aa")(p2_va, h3_oa, cind))
      (_a004_os("abab")(p1_va, p2_vb, h4_oa, h3_ob) = 1.0 * chol3d_vo("aa")(p1_va, h4_oa, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind))
      (_a004_os("bbbb")(p1_vb, p2_vb, h4_ob, h3_ob) = 1.0 * chol3d_vo("bb")(p1_vb, h4_ob, cind) * chol3d_vo("bb")(p2_vb, h3_ob, cind));
    // clang-format on
    sch.execute(exhw);
    //-------------------------------------------------
    //  auto [cind]   = CI.labels<1>("all");
    auto [a, b, c, d, e] = MO.labels<5>("virt");
    auto [i, j, k, l, m] = MO.labels<5>("occ");
    Tensor<T> res_2{{a, b, i, j}, {2, 2}};
    Tensor<T> pphh{{a, b, i, j}, {2, 2}};
    sch.allocate(res_2, pphh).execute();
    // clang-format off
    sch(pphh(a, b, i, j)       = cv3d(a, i, cind)         * cv3d(b, j, cind))
      (res_2(a, b, i, j)       = pphh(a, b, i, j))
      (res_2(a, b, i, j)      += -1.0*pphh(a, b, j, i));
    // clang-format on

    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;

        sch((d_t1s[off])() = d_t1())((d_t2s[off])() = d_t2()).execute();

        // TODO:UPDATE FOR DIIS
        // clang-format off
        sch
          (t1_vo("aa")(p1_va,h3_oa)                 = d_t1(p1_va,h3_oa))
          (t1_vo("bb")(p1_vb,h3_ob)                 = d_t1(p1_vb,h3_ob))
          (t2_vvoo("aaaa")(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
          (t2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
          (t2_vvoo("bbbb")(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob))
          .execute();
        // clang-format on
        cc2_os::cc2_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);
        cc2_os::cc2_t1_os(sch, MO, CI, /*d_r1,*/ r1_vo, t1_vo, t2_vvoo, f1_se, chol3d_se);
        cc2_os::cc2_t2_os(sch, MO, CI, /*d_r2,*/ r2_vvoo, t1_vo, t2_vvoo, f1_se, chol3d_se,
                          i0_t2_tmp, d_f1, cv3d, res_2);
        // clang-format off
        sch
          (d_r1(p2_va, h1_oa)                = r1_vo("aa")(p2_va, h1_oa))
          (d_r1(p2_vb, h1_ob)                = r1_vo("bb")(p2_vb, h1_ob))
          (d_r2(p3_va, p4_va, h2_oa, h1_oa)  = r2_vvoo("aaaa")(p3_va, p4_va, h2_oa, h1_oa))
          (d_r2(p3_vb, p4_vb, h2_ob, h1_ob)  = r2_vvoo("bbbb")(p3_vb, p4_vb, h2_ob, h1_ob))
          (d_r2(p3_va, p4_vb, h2_oa, h1_ob)  = r2_vvoo("abab")(p3_va, p4_vb, h2_oa, h1_ob))
          ;
        // clang-format on

        sch.execute(exhw, profile);

        std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2, d_e, d_r1_residual,
                                          d_r2_residual, p_evl_sorted, zshiftl, n_occ_alpha,
                                          n_occ_beta);

        update_r2(ec, d_r2());
        // clang-format off
        sch
          ((d_r1s[off])() = d_r1())
          ((d_r2s[off])() = d_r2())
          .execute();
        // clang-format on

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

        if(writet && (((iter + 1) % writet_iter == 0) /*|| (residual < thresh)*/)) {
          write_to_disk(d_t1, t1file);
          write_to_disk(d_t2, t2file);
        }

        if(residual < thresh) {
          Tensor<T> t2_copy{{V, V, O, O}, {2, 2}};
          // clang-format off
          sch.allocate(t2_copy)
            (t2_copy()                     =  1.0 * d_t2())
            (d_t2(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
            (d_t2(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
            (d_t2(p2_vb,p1_va,h4_ob,h3_oa) =  1.0 * t2_copy(p1_va,p2_vb,h3_oa,h4_ob))
            .deallocate(t2_copy)
            .execute();
          // clang-format on
          if(writet) {
            write_to_disk(d_t1, t1file);
            write_to_disk(d_t2, t2file);
            if(computeTData && chem_env.ioptions.ccsd_options.writev) {
              fs::copy_file(t1file, cc2_fp + ".fullT1amp", fs::copy_options::update_existing);
              fs::copy_file(t2file, cc2_fp + ".fullT2amp", fs::copy_options::update_existing);
            }
          }
          break;
        }
      }

      if(residual < thresh || titer + ndiis >= maxiter) { break; }
      if(ec.pg().rank() == 0) {
        std::cout << " MICROCYCLE DIIS UPDATE:";
        std::cout.width(21);
        std::cout << std::right << std::min(titer + ndiis, maxiter) + 1;
        std::cout.width(21);
        std::cout << std::right << "5" << std::endl;
      }

      std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
      std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
      std::vector<Tensor<T>>              next_t{d_t1, d_t2};
      diis<T>(ec, rs, ts, next_t);
    }

    if(profile && ec.print()) {
      std::string   profile_csv = cc2_fp + "_profile.csv";
      std::ofstream pds(profile_csv, std::ios::out);
      if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
      pds << ec.get_profile_header() << std::endl;
      pds << ec.get_profile_data().str() << std::endl;
      pds.close();
    }

    sch.deallocate(_a02V_os, _a007V_os, d_r1_residual, d_r2_residual);
    CCSE_Tensors<T>::deallocate_list(sch, _a004_os, i0_t2_tmp, _a01_os, _a04_os, _a05_os, _a06_os,
                                     _a001_os, _a006_os, _a008_os, _a009_os, _a017_os, _a019_os,
                                     _a020_os, _a021_os); //_a022

  } // no restart
  else {
    // clang-format off
    sch
      (d_e()=0)
      (t1_vo("aa")(p1_va,h3_oa) = d_t1(p1_va,h3_oa))
      (t1_vo("bb")(p1_vb,h3_ob) = d_t1(p1_vb,h3_ob))
      (t2_vvoo("aaaa")(p1_va,p2_va,h3_oa,h4_oa) = d_t2(p1_va,p2_va,h3_oa,h4_oa))
      (t2_vvoo("abab")(p1_va,p2_vb,h3_oa,h4_ob) = d_t2(p1_va,p2_vb,h3_oa,h4_ob))
      (t2_vvoo("bbbb")(p1_vb,p2_vb,h3_ob,h4_ob) = d_t2(p1_vb,p2_vb,h3_ob,h4_ob));
    // clang-format on

    cc2_os::cc2_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);

    sch.execute(exhw, profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  chem_env.cc_context.cc2_correlation_energy = energy;
  chem_env.cc_context.cc2_total_energy       = chem_env.hf_energy + energy;

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CC2"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CC2"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CC2"]["final_energy"]["total"] =
      chem_env.cc_context.cc2_total_energy;

    chem_env.write_json_data("CC2");
  }

  CCSE_Tensors<T>::deallocate_list(sch, _a02_os, _a03_os);
  CCSE_Tensors<T>::deallocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                   chol3d_vv);
  sch.deallocate(d_e, _a01V_os).execute();

  return std::make_tuple(residual, energy);
}

using T = double;
template std::tuple<double, double> cc2_os::cd_cc2_os_driver<T>(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, Tensor<T>& d_r1, Tensor<T>& d_r2,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, bool cc2_restart,
  std::string out_fp, bool computeTData);
