/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_os_ann.hpp"

using CCEType = double;
CCSE_Tensors<CCEType> _a021_os;
Tensor<CCEType>       a22_abab_os, a22_aaaa_os, a22_bbbb_os;
TiledIndexSpace       o_alpha_os, v_alpha_os, o_beta_os, v_beta_os;

Tensor<CCEType>       _a01V_os, _a02V_os, _a007V_os;
CCSE_Tensors<CCEType> _a01_os, _a02_os, _a03_os, _a04_os, _a05_os, _a06_os, _a001_os, _a004_os,
  _a006_os, _a008_os, _a009_os, _a017_os, _a019_os, _a020_os; //_a022

template<typename T>
void ccsd_e_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2, std::vector<CCSE_Tensors<T>>& f1_se,
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
void ccsd_t1_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                CCSE_Tensors<T>& r1_vo, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
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
void ccsd_t2_os(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                CCSE_Tensors<T>& r2, CCSE_Tensors<T>& t1, CCSE_Tensors<T>& t2,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se,
                CCSE_Tensors<T>& i0tmp) {
  auto [cind]                                     = CI.labels<1>("all");
  auto [p3, p4]                                   = MO.labels<2>("virt");
  auto [h1, h2]                                   = MO.labels<2>("occ");
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

  auto hw = sch.ec().exhw();
  // auto rank = sch.ec().pg().rank();

  // "_a022( aaaa )(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021_os( aa )(p3_va,p2_va,cind) *
  // _a021_os( aa )(p4_va,p1_va,cind)")
  // "_a022( bbbb )(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021_os( bb )(p3_vb,p2_vb,cind) *
  // _a021_os( bb )(p4_vb,p1_vb,cind)"
  // "_a022( abab )(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021_os( aa )(p3_va,p2_va,cind) *
  // _a021_os( bb )(p4_vb,p1_vb,cind)")

  int   a22_flag = 0;
  auto& oprof    = tamm::OpProfiler::instance();

  auto compute_v4_term = [=, &a22_flag, &oprof](const IndexVector& cblkid, span<T> cbuf) {
    Tensor<T>        a22_tmp;
    LabeledTensor<T>*lhsp_{nullptr}, *rhs1p_{nullptr}, *rhs2p_{nullptr};

    auto [cind]                       = CI.labels<1>("all");
    auto [p1_va, p2_va, p3_va, p4_va] = v_alpha_os.labels<4>("all");
    auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta_os.labels<4>("all");

    if(a22_flag == 1) {
      a22_tmp = {v_alpha_os, v_alpha_os, v_alpha_os, v_alpha_os};
      lhsp_   = new LabeledTensor<T>{a22_tmp, {p3_va, p4_va, p2_va, p1_va}};
      rhs1p_  = new LabeledTensor<T>{_a021_os("aa"), {p3_va, p2_va, cind}};
      rhs2p_  = new LabeledTensor<T>{_a021_os("aa"), {p4_va, p1_va, cind}};
    }
    else if(a22_flag == 2) {
      a22_tmp = {v_beta_os, v_beta_os, v_beta_os, v_beta_os};
      lhsp_   = new LabeledTensor<T>{a22_tmp, {p3_vb, p4_vb, p2_vb, p1_vb}};
      rhs1p_  = new LabeledTensor<T>{_a021_os("bb"), {p3_vb, p2_vb, cind}};
      rhs2p_  = new LabeledTensor<T>{_a021_os("bb"), {p4_vb, p1_vb, cind}};
    }
    else if(a22_flag == 3) {
      a22_tmp = {v_alpha_os, v_beta_os, v_alpha_os, v_beta_os};
      lhsp_   = new LabeledTensor<T>{a22_tmp, {p3_va, p4_vb, p2_va, p1_vb}};
      rhs1p_  = new LabeledTensor<T>{_a021_os("aa"), {p3_va, p2_va, cind}};
      rhs2p_  = new LabeledTensor<T>{_a021_os("bb"), {p4_vb, p1_vb, cind}};
    }
    else { tamm_terminate("[CCSD-OS] This line should be unreachable"); }

    LabeledTensor<T>& lhs_  = *lhsp_;
    LabeledTensor<T>& rhs1_ = *rhs1p_;
    LabeledTensor<T>& rhs2_ = *rhs2p_;

    // mult op constructor
    auto lhs_lbls  = lhs_.labels();
    auto rhs1_lbls = rhs1_.labels();
    auto rhs2_lbls = rhs2_.labels();

    IntLabelVec lhs_int_labels_;
    IntLabelVec rhs1_int_labels_;
    IntLabelVec rhs2_int_labels_;

    auto labels{lhs_lbls};
    labels.insert(labels.end(), rhs1_lbls.begin(), rhs1_lbls.end());
    labels.insert(labels.end(), rhs2_lbls.begin(), rhs2_lbls.end());

    internal::update_labels(labels);

    lhs_lbls  = IndexLabelVec(labels.begin(), labels.begin() + lhs_.labels().size());
    rhs1_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size(),
                              labels.begin() + lhs_.labels().size() + rhs1_.labels().size());
    rhs2_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size() + rhs1_.labels().size(),
                              labels.begin() + lhs_.labels().size() + rhs1_.labels().size() +
                                rhs2_.labels().size());
    lhs_.set_labels(lhs_lbls);
    rhs1_.set_labels(rhs1_lbls);
    rhs2_.set_labels(rhs2_lbls);

    // fillin_int_labels
    std::map<TileLabelElement, int> primary_labels_map;
    int                             cnt = -1;
    for(const auto& lbl: lhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: rhs1_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: rhs2_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
    for(const auto& lbl: lhs_.labels()) {
      lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    for(const auto& lbl: rhs1_.labels()) {
      rhs1_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    for(const auto& lbl: rhs2_.labels()) {
      rhs2_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
    }
    // todo: validate

    using TensorElType1 = T;
    using TensorElType2 = T;
    using TensorElType3 = T;

    // determine set of all labels for do_work
    IndexLabelVec all_labels{lhs_.labels()};
    all_labels.insert(all_labels.end(), rhs1_.labels().begin(), rhs1_.labels().end());
    all_labels.insert(all_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());
    // LabelLoopNest loop_nest{all_labels};

    // execute-bufacc
    IndexLabelVec lhs_labels{lhs_.labels()};
    IndexLabelVec rhs1_labels{rhs1_.labels()};
    IndexLabelVec rhs2_labels{rhs2_.labels()};
    IndexLabelVec all_rhs_labels{rhs1_.labels()};
    all_rhs_labels.insert(all_rhs_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());

    // compute the reduction labels
    std::sort(lhs_labels.begin(), lhs_labels.end());
    auto unique_labels = internal::unique_entries_by_primary_label(all_rhs_labels);
    std::sort(unique_labels.begin(), unique_labels.end());
    IndexLabelVec reduction_labels; //{reduction.begin(), reduction.end()};
    std::set_difference(unique_labels.begin(), unique_labels.end(), lhs_labels.begin(),
                        lhs_labels.end(), std::back_inserter(reduction_labels));

    std::vector<int> rhs1_map_output;
    std::vector<int> rhs2_map_output;
    std::vector<int> rhs1_map_reduction;
    std::vector<int> rhs2_map_reduction;
    // const auto&      lhs_lbls = lhs_.labels();
    for(auto& lbl: rhs1_labels) {
      auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
      if(it_out != lhs_lbls.end()) rhs1_map_output.push_back(it_out - lhs_lbls.begin());
      else rhs1_map_output.push_back(-1);

      // auto it_red = std::find(reduction.begin(), reduction.end(), lbl);
      auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
      if(it_red != reduction_labels.end())
        rhs1_map_reduction.push_back(it_red - reduction_labels.begin());
      else rhs1_map_reduction.push_back(-1);
    }

    for(auto& lbl: rhs2_labels) {
      auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
      if(it_out != lhs_lbls.end()) rhs2_map_output.push_back(it_out - lhs_lbls.begin());
      else rhs2_map_output.push_back(-1);

      auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
      if(it_red != reduction_labels.end())
        rhs2_map_reduction.push_back(it_red - reduction_labels.begin());
      else rhs2_map_reduction.push_back(-1);
    }

    auto ctensor = lhs_.tensor();
    auto atensor = rhs1_.tensor();
    auto btensor = rhs2_.tensor();

    std::vector<AddBuf<TensorElType1, TensorElType2, TensorElType3>*> add_bufs;

    // compute blockids from the loop indices. itval is the loop index
    // execute_bufacc(ec, hw);
    LabelLoopNest lhs_loop_nest{lhs_.labels()};
    IndexVector   translated_ablockid, translated_bblockid, translated_cblockid;
    auto          it    = lhs_loop_nest.begin();
    auto          itval = *it;
    for(; it != lhs_loop_nest.end(); ++it) {
      itval = *it;
      // auto        it   = ivec.begin();
      IndexVector c_block_id{itval};
      translated_cblockid = internal::translate_blockid(c_block_id, lhs_);
      if(translated_cblockid == cblkid) break;
    }

    // execute
    // const auto& ldist = lhs_.tensor().distribution();
    // for(const auto& lblockid: lhs_loop_nest) {
    //   const auto translated_lblockid = internal::translate_blockid(lblockid, lhs_);
    //   if(lhs_.tensor().is_non_zero(translated_lblockid) &&
    //       std::get<0>(ldist.locate(translated_lblockid)) == rank) {
    //     lambda(lblockid);
    //   }

    const size_t csize = ctensor.block_size(translated_cblockid);
    // std::vector<TensorElType1> cbuf(csize, 0);
    memset(cbuf.data(), 0x00, csize * sizeof(TensorElType1));
    const auto& cdims = ctensor.block_dims(translated_cblockid);

    SizeVec cdims_sz;
    for(const auto v: cdims) { cdims_sz.push_back(v); }

    AddBuf<TensorElType1, TensorElType2, TensorElType3>* ab{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    TensorElType2* th_a{nullptr};
    TensorElType3* th_b{nullptr};
    auto&          thandle = GPUStreamPool::getInstance().getStream();

    ab =
      new AddBuf<TensorElType1, TensorElType2, TensorElType3>{th_a, th_b, {}, translated_cblockid};
#else
    gpuStream_t thandle{};
    ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{ctensor, {}, translated_cblockid};
#endif
    add_bufs.push_back(ab);

    // LabelLoopNest inner_loop{reduction_lbls};
    LabelLoopNest inner_loop{reduction_labels};

    // int loop_counter = 0;

    TensorElType1* cbuf_dev_ptr{nullptr};
    TensorElType1* cbuf_tmp_dev_ptr{nullptr};
    auto&          memHostPool = tamm::RMMMemoryManager::getInstance().getHostMemoryPool();

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    auto& memDevicePool = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

    if(hw == ExecutionHW::GPU) {
      cbuf_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));
      cbuf_tmp_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));

      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
    }
#endif

    for(const auto& inner_it_val: inner_loop) { // k

      IndexVector a_block_id(rhs1_.labels().size());

      for(size_t i = 0; i < rhs1_map_output.size(); i++) {
        if(rhs1_map_output[i] != -1) { a_block_id[i] = itval[rhs1_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs1_map_reduction.size(); i++) {
        if(rhs1_map_reduction[i] != -1) { a_block_id[i] = inner_it_val[rhs1_map_reduction[i]]; }
      }

      const auto translated_ablockid = internal::translate_blockid(a_block_id, rhs1_);
      if(!atensor.is_non_zero(translated_ablockid)) continue;

      IndexVector b_block_id(rhs2_.labels().size());

      for(size_t i = 0; i < rhs2_map_output.size(); i++) {
        if(rhs2_map_output[i] != -1) { b_block_id[i] = itval[rhs2_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs2_map_reduction.size(); i++) {
        if(rhs2_map_reduction[i] != -1) { b_block_id[i] = inner_it_val[rhs2_map_reduction[i]]; }
      }

      const auto translated_bblockid = internal::translate_blockid(b_block_id, rhs2_);
      if(!btensor.is_non_zero(translated_bblockid)) continue;

      // compute block size and allocate buffers for abuf and bbuf
      const size_t asize = atensor.block_size(translated_ablockid);
      const size_t bsize = btensor.block_size(translated_bblockid);

      TensorElType2* abuf{nullptr};
      TensorElType3* bbuf{nullptr};
      abuf = static_cast<TensorElType2*>(memHostPool.allocate(asize * sizeof(TensorElType2)));
      bbuf = static_cast<TensorElType3*>(memHostPool.allocate(bsize * sizeof(TensorElType3)));

      atensor.get(translated_ablockid, {abuf, asize});
      btensor.get(translated_bblockid, {bbuf, bsize});

      const auto& adims = atensor.block_dims(translated_ablockid);
      const auto& bdims = btensor.block_dims(translated_bblockid);

      // changed cscale from 0 to 1 to aggregate on cbuf
      T cscale{1};

      SizeVec adims_sz, bdims_sz;
      for(const auto v: adims) { adims_sz.push_back(v); }
      for(const auto v: bdims) { bdims_sz.push_back(v); }

      // A*B
      {
        AddBuf<TensorElType1, TensorElType2, TensorElType3>* abptr{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        if(hw == ExecutionHW::GPU) {
          abptr = ab;
          ab->ta_ =
            static_cast<TensorElType2*>(memDevicePool.allocate(asize * sizeof(TensorElType2)));
          ab->tb_ =
            static_cast<TensorElType3*>(memDevicePool.allocate(bsize * sizeof(TensorElType3)));
        }
        else abptr = add_bufs[0];
#else
        abptr = add_bufs[0];
#endif
        abptr->abuf_ = abuf;
        abptr->bbuf_ = bbuf;

        {
          TimerGuard tg_dgemm{&oprof.multOpDgemmTime};
          kernels::block_multiply<T, TensorElType1, TensorElType2, TensorElType3>(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
            abptr->ta_, abptr->tb_,
#endif
            thandle, 1.0, abptr->abuf_, adims_sz, rhs1_int_labels_, abptr->bbuf_, bdims_sz,
            rhs2_int_labels_, cscale, cbuf.data(), cdims_sz, lhs_int_labels_, hw, false,
            cbuf_dev_ptr, cbuf_tmp_dev_ptr);
        }

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
        if(hw == ExecutionHW::GPU) {
          memDevicePool.deallocate(ab->ta_, asize * sizeof(TensorElType2));
          memDevicePool.deallocate(ab->tb_, bsize * sizeof(TensorElType3));
        }
#endif
      } // A * B

      memHostPool.deallocate(abuf, asize * sizeof(TensorElType2));
      memHostPool.deallocate(bbuf, bsize * sizeof(TensorElType3));
    } // end of reduction loop

    // add the computed update to the tensor
    {
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
      // copy to host
      if(hw == ExecutionHW::GPU) {
        TensorElType1* cbuf_tmp{nullptr};
        cbuf_tmp = static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
        std::memset(cbuf_tmp, 0, csize * sizeof(TensorElType1));
        {
          TimerGuard tg_copy{&oprof.multOpCopyTime};
          gpuMemcpyAsync<TensorElType1>(cbuf_tmp, cbuf_dev_ptr, csize, gpuMemcpyDeviceToHost,
                                        thandle);
        }
        // cbuf+=cbuf_tmp
        gpuStreamSynchronize(thandle);
        blas::axpy(csize, TensorElType1{1}, cbuf_tmp, 1, cbuf.data(), 1);

        memHostPool.deallocate(cbuf_tmp, csize * sizeof(TensorElType1));

        memDevicePool.deallocate(static_cast<void*>(cbuf_dev_ptr), csize * sizeof(TensorElType1));
        memDevicePool.deallocate(static_cast<void*>(cbuf_tmp_dev_ptr),
                                 csize * sizeof(TensorElType1));
      }
#endif
    }

    for(auto& ab: add_bufs) delete ab;
    add_bufs.clear();

    delete lhsp_;
    delete rhs1p_;
    delete rhs2p_;
  };

  a22_aaaa_os = Tensor<T>{{v_alpha_os, v_alpha_os, v_alpha_os, v_alpha_os}, compute_v4_term};
  a22_abab_os = Tensor<T>{{v_alpha_os, v_beta_os, v_alpha_os, v_beta_os}, compute_v4_term};
  a22_bbbb_os = Tensor<T>{{v_beta_os, v_beta_os, v_beta_os, v_beta_os}, compute_v4_term};

  // clang-format off
  sch
    (_a017_os("aa")(p3_va, h2_oa, cind)         = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov("aa")(h3_oa, p1_va, cind),
    "_a017_os( aa )(p3_va, h2_oa, cind)         = -1.0   * t2_aaaa(p1_va, p3_va, h3_oa, h2_oa) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (_a017_os("bb")(p3_vb, h2_ob, cind)         = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind),
    "_a017_os( bb )(p3_vb, h2_ob, cind)         = -1.0   * t2_bbbb(p1_vb, p3_vb, h3_ob, h2_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a017_os("bb")(p3_vb, h2_ob, cind)        += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov("aa")(h3_oa, p1_va, cind),
    "_a017_os( bb )(p3_vb, h2_ob, cind)        += -1.0   * t2_abab(p1_va, p3_vb, h3_oa, h2_ob) * chol3d_ov( aa )(h3_oa, p1_va, cind)")
    (_a017_os("aa")(p3_va, h2_oa, cind)        += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov("bb")(h3_ob, p1_vb, cind),
    "_a017_os( aa )(p3_va, h2_oa, cind)        += -1.0   * t2_abab(p3_va, p1_vb, h2_oa, h3_ob) * chol3d_ov( bb )(h3_ob, p1_vb, cind)")
    (_a006_os("aa")(h4_oa, h1_oa)               = -1.0   * chol3d_ov("aa")(h4_oa, p2_va, cind) * _a017_os("aa")(p2_va, h1_oa, cind),
    "_a006_os( aa )(h4_oa, h1_oa)               = -1.0   * chol3d_ov( aa )(h4_oa, p2_va, cind) * _a017_os( aa )(p2_va, h1_oa, cind)")
    (_a006_os("bb")(h4_ob, h1_ob)               = -1.0   * chol3d_ov("bb")(h4_ob, p2_vb, cind) * _a017_os("bb")(p2_vb, h1_ob, cind),
    "_a006_os( bb )(h4_ob, h1_ob)               = -1.0   * chol3d_ov( bb )(h4_ob, p2_vb, cind) * _a017_os( bb )(p2_vb, h1_ob, cind)")
    (_a007V_os(cind)                            =  1.0   * chol3d_ov("aa")(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa),
    "_a007V_os(cind)                            =  1.0   * chol3d_ov( aa )(h4_oa, p1_va, cind) * t1_aa(p1_va, h4_oa)")
    (_a007V_os(cind)                           +=  1.0   * chol3d_ov("bb")(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob),
    "_a007V_os(cind)                           +=  1.0   * chol3d_ov( bb )(h4_ob, p1_vb, cind) * t1_bb(p1_vb, h4_ob)")
    (_a009_os("aa")(h3_oa, h2_oa, cind)         =  1.0   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009_os( aa )(h3_oa, h2_oa, cind)         =  1.0   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a009_os("bb")(h3_ob, h2_ob, cind)         =  1.0   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob),
    "_a009_os( bb )(h3_ob, h2_ob, cind)         =  1.0   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p1_vb, h2_ob)")
    (_a021_os("aa")(p3_va, p1_va, cind)         = -0.5   * chol3d_ov("aa")(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa),
    "_a021_os( aa )(p3_va, p1_va, cind)         = -0.5   * chol3d_ov( aa )(h3_oa, p1_va, cind) * t1_aa(p3_va, h3_oa)")
    (_a021_os("bb")(p3_vb, p1_vb, cind)         = -0.5   * chol3d_ov("bb")(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob),
    "_a021_os( bb )(p3_vb, p1_vb, cind)         = -0.5   * chol3d_ov( bb )(h3_ob, p1_vb, cind) * t1_bb(p3_vb, h3_ob)")
    (_a021_os("aa")(p3_va, p1_va, cind)        +=  0.5   * chol3d_vv("aa")(p3_va, p1_va, cind),
    "_a021_os( aa )(p3_va, p1_va, cind)        +=  0.5   * chol3d_vv( aa )(p3_va, p1_va, cind)")
    (_a021_os("bb")(p3_vb, p1_vb, cind)        +=  0.5   * chol3d_vv("bb")(p3_vb, p1_vb, cind),
    "_a021_os( bb )(p3_vb, p1_vb, cind)        +=  0.5   * chol3d_vv( bb )(p3_vb, p1_vb, cind)")
    (_a017_os("aa")(p3_va, h2_oa, cind)        += -2.0   * t1_aa(p2_va, h2_oa) * _a021_os("aa")(p3_va, p2_va, cind),
    "_a017_os( aa )(p3_va, h2_oa, cind)        += -2.0   * t1_aa(p2_va, h2_oa) * _a021_os( aa )(p3_va, p2_va, cind)")
    (_a017_os("bb")(p3_vb, h2_ob, cind)        += -2.0   * t1_bb(p2_vb, h2_ob) * _a021_os("bb")(p3_vb, p2_vb, cind),
    "_a017_os( bb )(p3_vb, h2_ob, cind)        += -2.0   * t1_bb(p2_vb, h2_ob) * _a021_os( bb )(p3_vb, p2_vb, cind)")
    (_a008_os("aa")(h3_oa, h1_oa, cind)         =  1.0   * _a009_os("aa")(h3_oa, h1_oa, cind),
    "_a008_os( aa )(h3_oa, h1_oa, cind)         =  1.0   * _a009_os( aa )(h3_oa, h1_oa, cind)")
    (_a008_os("bb")(h3_ob, h1_ob, cind)         =  1.0   * _a009_os("bb")(h3_ob, h1_ob, cind),
    "_a008_os( bb )(h3_ob, h1_ob, cind)         =  1.0   * _a009_os( bb )(h3_ob, h1_ob, cind)")
    (_a009_os("aa")(h3_oa, h1_oa, cind)        +=  1.0   * chol3d_oo("aa")(h3_oa, h1_oa, cind),
    "_a009_os( aa )(h3_oa, h1_oa, cind)        +=  1.0   * chol3d_oo( aa )(h3_oa, h1_oa, cind)")
    (_a009_os("bb")(h3_ob, h1_ob, cind)        +=  1.0   * chol3d_oo("bb")(h3_ob, h1_ob, cind),
    "_a009_os( bb )(h3_ob, h1_ob, cind)        +=  1.0   * chol3d_oo( bb )(h3_ob, h1_ob, cind)")

    (_a001_os("aa")(p4_va, p2_va)                  = -2.0   * _a021_os("aa")(p4_va, p2_va, cind) * _a007V_os(cind),
    "_a001_os( aa )(p4_va, p2_va)                  = -2.0   * _a021_os( aa )(p4_va, p2_va, cind) * _a007V_os(cind)")
    (_a001_os("bb")(p4_vb, p2_vb)                  = -2.0   * _a021_os("bb")(p4_vb, p2_vb, cind) * _a007V_os(cind),
    "_a001_os( bb )(p4_vb, p2_vb)                  = -2.0   * _a021_os( bb )(p4_vb, p2_vb, cind) * _a007V_os(cind)")
    (_a001_os("aa")(p4_va, p2_va)                 += -1.0   * _a017_os("aa")(p4_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a001_os( aa )(p4_va, p2_va)                 += -1.0   * _a017_os( aa )(p4_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a001_os("bb")(p4_vb, p2_vb)                 += -1.0   * _a017_os("bb")(p4_vb, h2_ob, cind) * chol3d_ov("bb")(h2_ob, p2_vb, cind),
    "_a001_os( bb )(p4_vb, p2_vb)                 += -1.0   * _a017_os( bb )(p4_vb, h2_ob, cind) * chol3d_ov( bb )(h2_ob, p2_vb, cind)")
    (_a006_os("aa")(h4_oa, h1_oa)                 +=  1.0   * _a009_os("aa")(h4_oa, h1_oa, cind) * _a007V_os(cind),
    "_a006_os( aa )(h4_oa, h1_oa)                 +=  1.0   * _a009_os( aa )(h4_oa, h1_oa, cind) * _a007V_os(cind)")
    (_a006_os("bb")(h4_ob, h1_ob)                 +=  1.0   * _a009_os("bb")(h4_ob, h1_ob, cind) * _a007V_os(cind),
    "_a006_os( bb )(h4_ob, h1_ob)                 +=  1.0   * _a009_os( bb )(h4_ob, h1_ob, cind) * _a007V_os(cind)")
    (_a006_os("aa")(h4_oa, h1_oa)                 += -1.0   * _a009_os("aa")(h3_oa, h1_oa, cind) * _a008_os("aa")(h4_oa, h3_oa, cind),
    "_a006_os( aa )(h4_oa, h1_oa)                 += -1.0   * _a009_os( aa )(h3_oa, h1_oa, cind) * _a008_os( aa )(h4_oa, h3_oa, cind)")
    (_a006_os("bb")(h4_ob, h1_ob)                 += -1.0   * _a009_os("bb")(h3_ob, h1_ob, cind) * _a008_os("bb")(h4_ob, h3_ob, cind),
    "_a006_os( bb )(h4_ob, h1_ob)                 += -1.0   * _a009_os( bb )(h3_ob, h1_ob, cind) * _a008_os( bb )(h4_ob, h3_ob, cind)")
    (_a019_os("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009_os("aa")(h4_oa, h1_oa, cind) * _a009_os("aa")(h3_oa, h2_oa, cind),
    "_a019_os( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa)  =  0.25  * _a009_os( aa )(h4_oa, h1_oa, cind) * _a009_os( aa )(h3_oa, h2_oa, cind)")
    (_a019_os("abab")(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009_os("aa")(h4_oa, h1_oa, cind) * _a009_os("bb")(h3_ob, h2_ob, cind),
    "_a019_os( abab )(h4_oa, h3_ob, h1_oa, h2_ob)  =  0.25  * _a009_os( aa )(h4_oa, h1_oa, cind) * _a009_os( bb )(h3_ob, h2_ob, cind)")
    (_a019_os("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009_os("bb")(h4_ob, h1_ob, cind) * _a009_os("bb")(h3_ob, h2_ob, cind),
    "_a019_os( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob)  =  0.25  * _a009_os( bb )(h4_ob, h1_ob, cind) * _a009_os( bb )(h3_ob, h2_ob, cind)")
    (_a020_os("aaaa")(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009_os("aa")(h4_oa, h1_oa, cind) * _a021_os("aa")(p4_va, p1_va, cind),
    "_a020_os( aaaa )(p4_va, h4_oa, p1_va, h1_oa)  = -2.0   * _a009_os( aa )(h4_oa, h1_oa, cind) * _a021_os( aa )(p4_va, p1_va, cind)")
    (_a020_os("abab")(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009_os("bb")(h4_ob, h1_ob, cind) * _a021_os("aa")(p4_va, p1_va, cind),
    "_a020_os( abab )(p4_va, h4_ob, p1_va, h1_ob)  = -2.0   * _a009_os( bb )(h4_ob, h1_ob, cind) * _a021_os( aa )(p4_va, p1_va, cind)")
    (_a020_os("baba")(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009_os("aa")(h4_oa, h1_oa, cind) * _a021_os("bb")(p4_vb, p1_vb, cind),
    "_a020_os( baba )(p4_vb, h4_oa, p1_vb, h1_oa)  = -2.0   * _a009_os( aa )(h4_oa, h1_oa, cind) * _a021_os( bb )(p4_vb, p1_vb, cind)")
    (_a020_os("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009_os("bb")(h4_ob, h1_ob, cind) * _a021_os("bb")(p4_vb, p1_vb, cind),
    "_a020_os( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob)  = -2.0   * _a009_os( bb )(h4_ob, h1_ob, cind) * _a021_os( bb )(p4_vb, p1_vb, cind)")

    (_a017_os("aa")(p3_va, h2_oa, cind)        +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo("aa")(h3_oa, h2_oa, cind),
    "_a017_os( aa )(p3_va, h2_oa, cind)        +=  1.0   * t1_aa(p3_va, h3_oa) * chol3d_oo( aa )(h3_oa, h2_oa, cind)")
    (_a017_os("bb")(p3_vb, h2_ob, cind)        +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo("bb")(h3_ob, h2_ob, cind),
    "_a017_os( bb )(p3_vb, h2_ob, cind)        +=  1.0   * t1_bb(p3_vb, h3_ob) * chol3d_oo( bb )(h3_ob, h2_ob, cind)")
    (_a017_os("aa")(p3_va, h2_oa, cind)        += -1.0   * chol3d_vo("aa")(p3_va, h2_oa, cind),
    "_a017_os( aa )(p3_va, h2_oa, cind)        += -1.0   * chol3d_vo( aa )(p3_va, h2_oa, cind)")
    (_a017_os("bb")(p3_vb, h2_ob, cind)        += -1.0   * chol3d_vo("bb")(p3_vb, h2_ob, cind),
    "_a017_os( bb )(p3_vb, h2_ob, cind)        += -1.0   * chol3d_vo( bb )(p3_vb, h2_ob, cind)")

    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017_os("aa")(p3_va, h1_oa, cind) * _a017_os("aa")(p4_va, h2_oa, cind),
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)        =  0.5   * _a017_os( aa )(p3_va, h1_oa, cind) * _a017_os( aa )(p4_va, h2_oa, cind)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017_os("bb")(p3_vb, h1_ob, cind) * _a017_os("bb")(p4_vb, h2_ob, cind),
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)        =  0.5   * _a017_os( bb )(p3_vb, h1_ob, cind) * _a017_os( bb )(p4_vb, h2_ob, cind)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017_os("aa")(p3_va, h1_oa, cind) * _a017_os("bb")(p4_vb, h2_ob, cind),
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)        =  1.0   * _a017_os( aa )(p3_va, h1_oa, cind) * _a017_os( bb )(p4_vb, h2_ob, cind)").execute(hw);



    // sch(_a022("aaaa")(p3_va,p4_va,p2_va,p1_va)  =  1.0   * _a021_os("aa")(p3_va,p2_va,cind) * _a021_os("aa")(p4_va,p1_va,cind),
    // "_a022( aaaa )(p3_va,p4_va,p2_va,p1_va)     =  1.0   * _a021_os( aa )(p3_va,p2_va,cind) * _a021_os( aa )(p4_va,p1_va,cind)")
    // (_a022("abab")(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021_os("aa")(p3_va,p2_va,cind) * _a021_os("bb")(p4_vb,p1_vb,cind),
    // "_a022( abab )(p3_va,p4_vb,p2_va,p1_vb)     =  1.0   * _a021_os( aa )(p3_va,p2_va,cind) * _a021_os( bb )(p4_vb,p1_vb,cind)")
    // (_a022("bbbb")(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021_os("bb")(p3_vb,p2_vb,cind) * _a021_os("bb")(p4_vb,p1_vb,cind),
    // "_a022( bbbb )(p3_vb,p4_vb,p2_vb,p1_vb)     =  1.0   * _a021_os( bb )(p3_vb,p2_vb,cind) * _a021_os( bb )(p4_vb,p1_vb,cind)")
    // (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022("aaaa")(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa),
    // "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a022( aaaa )(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa)")
    // (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022("bbbb")(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob),
    // "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a022( bbbb )(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob)")
    // (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022("abab")(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    // "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a022( abab )(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)");

    a22_flag = 1;

    sch(i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * a22_aaaa_os(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa),
       "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * a22_aaaa_os(p3_va, p4_va, p2_va, p1_va) * t2_aaaa(p2_va,p1_va,h1_oa,h2_oa)").execute(hw);

    a22_flag = 2;

    sch(i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * a22_bbbb_os(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob),
       "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * a22_bbbb_os(p3_vb, p4_vb, p2_vb, p1_vb) * t2_bbbb(p2_vb,p1_vb,h1_ob,h2_ob)").execute(hw);

    a22_flag = 3;

    sch(i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * a22_abab_os(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
       "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * a22_abab_os(p3_va, p4_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)").execute(hw);

    sch(_a019_os("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) += -0.125 * _a004_os("aaaa")(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a019_os( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa)    += -0.125 * _a004_os( aaaa )(p1_va, p2_va, h3_oa, h4_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a019_os("abab")(h4_oa, h3_ob, h1_oa, h2_ob)    +=  0.25  * _a004_os("abab")(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019_os( abab )(h4_oa, h3_ob, h1_oa, h2_ob)    +=  0.25  * _a004_os( abab )(p1_va, p2_vb, h4_oa, h3_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (_a019_os("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob)    += -0.125 * _a004_os("bbbb")(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob),
    "_a019_os( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob)    += -0.125 * _a004_os( bbbb )(p1_vb, p2_vb, h3_ob, h4_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019_os("aaaa")(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa),
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a019_os( aaaa )(h4_oa, h3_oa, h1_oa, h2_oa) * t2_aaaa(p3_va, p4_va, h4_oa, h3_oa)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019_os("bbbb")(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob),
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a019_os( bbbb )(h4_ob, h3_ob, h1_ob, h2_ob) * t2_bbbb(p3_vb, p4_vb, h4_ob, h3_ob)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019_os("abab")(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob),
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       +=  4.0   * _a019_os( abab )(h4_oa, h3_ob, h1_oa, h2_ob) * t2_abab(p3_va, p4_vb, h4_oa, h3_ob)")
    (_a020_os("aaaa")(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004_os("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020_os( aaaa )(p1_va, h3_oa, p4_va, h2_oa) +=  0.5   * _a004_os( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020_os("baab")(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004_os("aaaa")(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020_os( baab )(p1_vb, h3_oa, p4_va, h2_ob)  = -0.5   * _a004_os( aaaa )(p2_va, p4_va, h3_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020_os("abba")(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004_os("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob),
    "_a020_os( abba )(p1_va, h3_ob, p4_vb, h2_oa)  = -0.5   * _a004_os( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_abab(p1_va,p2_vb,h2_oa,h1_ob)")
    (_a020_os("bbbb")(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004_os("bbbb")(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob),
    "_a020_os( bbbb )(p1_vb, h3_ob, p4_vb, h2_ob) +=  0.5   * _a004_os( bbbb )(p2_vb, p4_vb, h3_ob, h1_ob) * t2_bbbb(p1_vb,p2_vb,h1_ob,h2_ob)")
    (_a020_os("baba")(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004_os("abab")(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob),
    "_a020_os( baba )(p1_vb, h7_oa, p6_vb, h2_oa) +=  1.0   * _a004_os( abab )(p5_va, p6_vb, h7_oa, h8_ob) * t2_abab(p5_va,p1_vb,h2_oa,h8_ob)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020_os("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa),
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       +=  1.0   * _a020_os( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020_os("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob),
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -1.0   * _a020_os( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020_os("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob),
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       +=  1.0   * _a020_os( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020_os("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob),
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -1.0   * _a020_os( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
    (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020_os("baba")(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob),
    "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020_os( baba )(p1_vb, h7_oa, p6_vb, h2_oa) * t2_abab(p3_va, p6_vb, h7_oa, h4_ob)")
    (i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020_os("abab")(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob),
    "i0_abab(p3_va, p1_vb, h2_oa, h4_ob)       +=  1.0   * _a020_os( abab )(p3_va, h8_ob, p5_va, h4_ob) * t2_abab(p5_va, p1_vb, h2_oa, h8_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020_os("bbbb")(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob),
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       +=  1.0   * _a020_os( bbbb )(p4_vb, h4_ob, p1_vb, h1_ob) * t2_abab(p3_va, p1_vb, h2_oa, h4_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020_os("baab")(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa),
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * _a020_os( baab )(p4_vb, h4_oa, p1_va, h1_ob) * t2_aaaa(p3_va, p1_va, h4_oa, h2_oa)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020_os("aaaa")(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob),
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       +=  1.0   * _a020_os( aaaa )(p4_va, h4_oa, p1_va, h1_oa) * t2_abab(p1_va, p3_vb, h4_oa, h2_ob)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020_os("abba")(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob),
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * _a020_os( abba )(p4_va, h4_ob, p1_vb, h1_oa) * t2_bbbb(p3_vb, p1_vb, h4_ob, h2_ob)")

    (_a001_os("aa")(p4_va, p1_va)              += -1.0   * f1_vv("aa")(p4_va, p1_va),
    "_a001_os( aa )(p4_va, p1_va)              += -1.0   * f1_vv( aa )(p4_va, p1_va)")
    (_a001_os("bb")(p4_vb, p1_vb)              += -1.0   * f1_vv("bb")(p4_vb, p1_vb),
    "_a001_os( bb )(p4_vb, p1_vb)              += -1.0   * f1_vv( bb )(p4_vb, p1_vb)")
    (_a001_os("aa")(p4_va, p1_va)              +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "_a001_os( aa )(p4_va, p1_va)              +=  1.0   * t1_aa(p4_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a001_os("bb")(p4_vb, p1_vb)              +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov("bb")(h1_ob, p1_vb),
    "_a001_os( bb )(p4_vb, p1_vb)              +=  1.0   * t1_bb(p4_vb, h1_ob) * f1_ov( bb )(h1_ob, p1_vb)") // NEW TERM
    (_a006_os("aa")(h9_oa, h1_oa)              +=  1.0   * f1_oo("aa")(h9_oa, h1_oa),
    "_a006_os( aa )(h9_oa, h1_oa)              +=  1.0   * f1_oo( aa )(h9_oa, h1_oa)")
    (_a006_os("bb")(h9_ob, h1_ob)              +=  1.0   * f1_oo("bb")(h9_ob, h1_ob),
    "_a006_os( bb )(h9_ob, h1_ob)              +=  1.0   * f1_oo( bb )(h9_ob, h1_ob)")
    (_a006_os("aa")(h9_oa, h1_oa)              +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov("aa")(h9_oa, p8_va),
    "_a006_os( aa )(h9_oa, h1_oa)              +=  1.0   * t1_aa(p8_va, h1_oa) * f1_ov( aa )(h9_oa, p8_va)")
    (_a006_os("bb")(h9_ob, h1_ob)              +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov("bb")(h9_ob, p8_vb),
    "_a006_os( bb )(h9_ob, h1_ob)              +=  1.0   * t1_bb(p8_vb, h1_ob) * f1_ov( bb )(h9_ob, p8_vb)")

    (i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001_os("aa")(p4_va, p2_va),
    "i0_aaaa(p3_va, p4_va, h1_oa, h2_oa)       += -0.5   * t2_aaaa(p3_va, p2_va, h1_oa, h2_oa) * _a001_os( aa )(p4_va, p2_va)")
    (i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001_os("bb")(p4_vb, p2_vb),
    "i0_bbbb(p3_vb, p4_vb, h1_ob, h2_ob)       += -0.5   * t2_bbbb(p3_vb, p2_vb, h1_ob, h2_ob) * _a001_os( bb )(p4_vb, p2_vb)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001_os("bb")(p4_vb, p2_vb),
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p2_vb, h1_oa, h2_ob) * _a001_os( bb )(p4_vb, p2_vb)")
    (i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001_os("aa")(p4_va, p2_va),
    "i0_abab(p4_va, p3_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p2_va, p3_vb, h1_oa, h2_ob) * _a001_os( aa )(p4_va, p2_va)")

    (i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006_os("aa")(h3_oa, h2_oa),
    "i0_aaaa(p3_va, p4_va, h2_oa, h1_oa)       += -0.5   * t2_aaaa(p3_va, p4_va, h3_oa, h1_oa) * _a006_os( aa )(h3_oa, h2_oa)")
    (i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006_os("bb")(h3_ob, h2_ob),
    "i0_bbbb(p3_vb, p4_vb, h2_ob, h1_ob)       += -0.5   * t2_bbbb(p3_vb, p4_vb, h3_ob, h1_ob) * _a006_os( bb )(h3_ob, h2_ob)")
    (i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006_os("aa")(h3_oa, h2_oa),
    "i0_abab(p3_va, p4_vb, h2_oa, h1_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h3_oa, h1_ob) * _a006_os( aa )(h3_oa, h2_oa)")
    (i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006_os("bb")(h3_ob, h2_ob),
    "i0_abab(p3_va, p4_vb, h1_oa, h2_ob)       += -1.0   * t2_abab(p3_va, p4_vb, h1_oa, h3_ob) * _a006_os( bb )(h3_ob, h2_ob)")

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
std::tuple<double, double>
cd_ccsd_os_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                  const TiledIndexSpace& CI, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1,
                  Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
                  std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
                  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
                  bool ccsd_restart, std::string ccsd_fp, bool computeTData) {
  auto cc_t1 = std::chrono::high_resolution_clock::now();

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

  std::string t1file = ccsd_fp + ".t1amp";
  std::string t2file = ccsd_fp + ".t2amp";

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

  double total_ccsd_mem = sum_tensor_sizes(d_t1, d_t2, d_f1, d_r1, d_r2, cv3d, d_e, _a01V_os) +
                          CCSE_Tensors<T>::sum_tensor_sizes_list(r1_vo, r2_vvoo, t1_vo, t2_vvoo) +
                          CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo, f1_ov, f1_vo, f1_vv,
                                                                 chol3d_oo, chol3d_ov, chol3d_vo,
                                                                 chol3d_vv) +
                          CCSE_Tensors<T>::sum_tensor_sizes_list(_a02_os, _a03_os);

  for(size_t ri = 0; ri < d_r1s.size(); ri++)
    total_ccsd_mem += sum_tensor_sizes(d_r1s[ri], d_r2s[ri], d_t1s[ri], d_t2s[ri]);

  // Intermediates
  // const double v4int_size = CCSE_Tensors<T>::sum_tensor_sizes_list(_a022);
  double total_ccsd_mem_tmp =
    sum_tensor_sizes(_a02V_os, _a007V_os) /*+ v4int_size */ +
    CCSE_Tensors<T>::sum_tensor_sizes_list(i0_t2_tmp, _a01_os, _a04_os, _a05_os, _a06_os, _a001_os,
                                           _a004_os, _a006_os, _a008_os, _a009_os, _a017_os,
                                           _a019_os, _a020_os, _a021_os);

  if(!ccsd_restart) total_ccsd_mem += total_ccsd_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for Open Shell Cholesky CCSD calculation: "
              << std::fixed << std::setprecision(2) << total_ccsd_mem << " GiB" << std::endl;
    // std::cout << " (V^4 intermediate size: " << std::fixed << std::setprecision(2) << v4int_size
    //           << " GiB)" << std::endl;
  }
  check_memory_requirements(ec, total_ccsd_mem);

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

  print_ccsd_header(ec.print());

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

  if(!ccsd_restart) {
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
        ccsd_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);
        ccsd_t1_os(sch, MO, CI, /*d_r1,*/ r1_vo, t1_vo, t2_vvoo, f1_se, chol3d_se);
        ccsd_t2_os(sch, MO, CI, /*d_r2,*/ r2_vvoo, t1_vo, t2_vvoo, f1_se, chol3d_se, i0_t2_tmp);
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

        if(writet && ((iter + 1) % writet_iter == 0)) {
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

    if(writet) {
      write_to_disk(d_t1, t1file);
      write_to_disk(d_t2, t2file);
      if(computeTData && chem_env.ioptions.ccsd_options.writev) {
        fs::copy_file(t1file, ccsd_fp + ".fullT1amp", fs::copy_options::update_existing);
        fs::copy_file(t2file, ccsd_fp + ".fullT2amp", fs::copy_options::update_existing);
      }
    }

    if(profile && ec.print()) {
      std::string   profile_csv = ccsd_fp + "_profile.csv";
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

    ccsd_e_os(sch, MO, CI, d_e, t1_vo, t2_vvoo, f1_se, chol3d_se);

    sch.execute(exhw, profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  chem_env.cc_context.ccsd_correlation_energy = energy;
  chem_env.cc_context.ccsd_total_energy       = chem_env.hf_energy + energy;

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CCSD"]["final_energy"]["total"] =
      chem_env.cc_context.ccsd_total_energy;
    sys_data.results["output"]["CCSD"]["performance"]["total_time"] = ccsd_time;

    chem_env.write_json_data("CCSD");
  }

  CCSE_Tensors<T>::deallocate_list(sch, _a02_os, _a03_os);
  CCSE_Tensors<T>::deallocate_list(sch, r1_vo, r2_vvoo, t1_vo, t2_vvoo);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vo, f1_vv, chol3d_oo, chol3d_ov, chol3d_vo,
                                   chol3d_vv);
  sch.deallocate(d_e, _a01V_os).execute();

  return std::make_tuple(residual, energy);
}

using T = double;
template std::tuple<double, double>
cd_ccsd_os_driver<T>(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                     const TiledIndexSpace& CI, Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1,
                     Tensor<T>& d_r1, Tensor<T>& d_r2, std::vector<Tensor<T>>& d_r1s,
                     std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
                     std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d,
                     bool ccsd_restart, std::string out_fp, bool computeTData);
