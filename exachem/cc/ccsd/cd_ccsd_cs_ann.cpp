/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/cd_ccsd_cs_ann.hpp"
#include "exachem/cc/krylov_solvers.hpp"

using namespace exachem::cc::solvers;

template<typename T>

void exachem::cc::ccsd::CD_CCSD_CS<T>::ccsd_amp_update(Scheduler& sch, const Tensor<T>& t2_abab,
                                                       const Tensor<T>& t2_aaaa) {
  const auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  const auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");

  // clang-format off
    sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
    (t2_aaaa_temp() = t2_aaaa(),
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa),
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa),
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)");
  // clang-format on
}

template<typename T>
void exachem::cc::ccsd::CD_CCSD_CS<T>::ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO,
                                                 const TiledIndexSpace& CI, Tensor<T>& de,
                                                 const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                                                 const Tensor<T>&              t2_aaaa,
                                                 std::vector<CCSE_Tensors<T>>& f1_se,
                                                 std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind] = CI.labels<1>("all");

  const auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  const auto [p1_vb]        = v_beta.labels<1>("all");
  const auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  const auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
    (_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (_a02("aa")(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a02( aa )(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a03("aa")(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a03( aa )(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de()  =  2.0 * _a01V() * _a01V(),
    "de()  =  2.0 * _a01V() * _a01V()")
    (de() += -1.0 * _a02("aa")(h1_oa, h2_oa, cind) * _a02("aa")(h2_oa, h1_oa, cind),
    "de() += -1.0 * _a02( aa )(h1_oa, h2_oa, cind) * _a02( aa )(h2_oa, h1_oa, cind)")
    (de() +=  1.0 * _a03("aa")(h1_oa, p1_va, cind) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "de() +=  1.0 * _a03( aa )(h1_oa, p1_va, cind) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    ;
  // clang-format on
}

template<typename T>
void exachem::cc::ccsd::CD_CCSD_CS<T>::ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO,
                                                  const TiledIndexSpace& CI, Tensor<T>& i0_aa,
                                                  const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                                                  std::vector<CCSE_Tensors<T>>& f1_se,
                                                  std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind] = CI.labels<1>("all");
  const auto [p2]   = MO.labels<1>("virt");
  const auto [h1]   = MO.labels<1>("occ");

  const auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  const auto [p1_vb]        = v_beta.labels<1>("all");
  const auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  const auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov( aa )(h1_oa, p2_va)")
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01("aa")(h1_oa, h1_oa, cind))
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h1_oa, p1_va, cind) * _a01( aa )(h2_oa, h1_oa, cind)")      // o2vm
    (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va),
    "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06("aa")(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a06( aa )(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov( aa )(h2_oa, p2_va, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa),
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a02V(cind)")                      // ovm
    (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h1_oa, p2_va),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h1_oa, p2_va)")
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
    (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (_a06("aa")(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    "_a06( aa )(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind)")                                      // ovm
    (_a06("aa")(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a06( aa )(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h1_oa, cind)")                   // o2vm
    (_a01("aa")(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a01( aa )(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h2_oa, h1_oa, cind)")                                    // o2m
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h2_oa, h1_oa, cind) * _a06("aa")(p2_va, h2_oa, cind),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h2_oa, h1_oa, cind) * _a06( aa )(p2_va, h2_oa, cind)")        // o2vm
    // (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo("aa")(h2_oa, h1_oa), // MOVED ABOVE
    // "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo( aa )(h2_oa, h1_oa)")                        // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv("aa")(p2_va, p1_va),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv( aa )(p2_va, p1_va)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void exachem::cc::ccsd::CD_CCSD_CS<T>::ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO,
                                                  const TiledIndexSpace& CI, Tensor<T>& i0_abab,
                                                  const Tensor<T>& t1_aa, Tensor<T>& t2_abab,
                                                  Tensor<T>&                    t2_aaaa,
                                                  std::vector<CCSE_Tensors<T>>& f1_se,
                                                  std::vector<CCSE_Tensors<T>>& chol3d_se) {
  const auto [cind]   = CI.labels<1>("all");
  const auto [p3, p4] = MO.labels<2>("virt");
  const auto [h1, h2] = MO.labels<2>("occ");

  const auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  const auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
  const auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
  const auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (_a017("aa")(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)               = -1.0  * chol3d_ov("aa")(h2_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind),
    "_a006( aa )(h2_oa, h1_oa)               = -1.0  * chol3d_ov( aa )(h2_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
    (_a007V(cind)                            =  2.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    "_a007V(cind)                            =  2.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa)")
    (_a009("aa")(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009( aa )(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a021("aa")(p2_va, p1_va, cind)         = -0.5  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    "_a021( aa )(p2_va, p1_va, cind)         = -0.5  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa)")
    (_a021("aa")(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a021( aa )(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv( aa )(p2_va, p1_va, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021("aa")(p1_va, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021( aa )(p1_va, p2_va, cind)")
    (_a008("aa")(h2_oa, h1_oa, cind)         =  1.0  * _a009("aa")(h2_oa, h1_oa, cind),
    "_a008( aa )(h2_oa, h1_oa, cind)         =  1.0  * _a009( aa )(h2_oa, h1_oa, cind)")
    (_a009("aa")(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a009( aa )(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo( aa )(h2_oa, h1_oa, cind)")
    .exact_copy(_a009("bb")(h2_ob,h1_ob,cind),_a009("aa")(h2_ob,h1_ob,cind))
    .exact_copy(_a021("bb")(p2_vb,p1_vb,cind),_a021("aa")(p2_vb,p1_vb,cind))
    (_a001("aa")(p1_va, p2_va)               = -2.0  * _a021("aa")(p1_va, p2_va, cind) * _a007V(cind),
    "_a001( aa )(p1_va, p2_va)               = -2.0  * _a021( aa )(p1_va, p2_va, cind) * _a007V(cind)")
    (_a001("aa")(p1_va, p2_va)              += -1.0  * _a017("aa")(p1_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a001( aa )(p1_va, p2_va)              += -1.0  * _a017( aa )(p1_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)              +=  1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a007V(cind),
    "_a006( aa )(h2_oa, h1_oa)              +=  1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a007V(cind)")
    (_a006("aa")(h3_oa, h1_oa)              += -1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a008("aa")(h3_oa, h2_oa, cind),
    "_a006( aa )(h3_oa, h1_oa)              += -1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a008( aa )(h3_oa, h2_oa, cind)")
    (_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009("aa")(h2_oa, h1_oa, cind) * _a009("bb")(h1_ob, h2_ob, cind),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009( aa )(h2_oa, h1_oa, cind) * _a009( bb )(h1_ob, h2_ob, cind)")
    (_a020("aaaa")(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a021("aa")(p2_va, p1_va, cind),
    "_a020( aaaa )(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a021( aa )(p2_va, p1_va, cind)")
    .exact_copy(_a020("baba")(p2_vb, h2_oa, p1_vb, h1_oa),_a020("aaaa")(p2_vb, h2_oa, p1_vb, h1_oa))
    (_a020("aaaa")(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004("aaaa")(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020( aaaa )(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004( aaaa )(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020("baab")(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004("aaaa")(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020( baab )(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004( aaaa )(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004("abab")(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob),
    "_a020( baba )(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004( abab )(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob)")
    (_a017("aa")(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo("aa")(h1_oa, h2_oa, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo( aa )(h1_oa, h2_oa, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a001("aa")(p2_va, p1_va)                 += -1.0  * f1_vv("aa")(p2_va, p1_va),
    "_a001( aa )(p2_va, p1_va)                 += -1.0  * f1_vv( aa )(p2_va, p1_va)")
    (_a001("aa")(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "_a001( aa )(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * f1_oo("aa")(h2_oa, h1_oa),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * f1_oo( aa )(h2_oa, h1_oa)")
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)")
    .exact_copy(_a017("bb")(p1_vb, h1_ob, cind), _a017("aa")(p1_vb, h1_ob, cind))
    .exact_copy(_a006("bb")(h1_ob, h2_ob), _a006("aa")(h1_ob, h2_ob))
    .exact_copy(_a001("bb")(p1_vb, p2_vb), _a001("aa")(p1_vb, p2_vb))
    .exact_copy(_a021("bb")(p1_vb, p2_vb, cind), _a021("aa")(p1_vb, p2_vb, cind))
    .exact_copy(_a020("bbbb")(p1_vb, h1_ob, p2_vb, h2_ob), _a020("aaaa")(p1_vb, h1_ob, p2_vb, h2_ob))

    (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020("bbbb")(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    "i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020(bbbb)(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob)")
    (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baab")(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    "i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baab)(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baba)(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob)")
    .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017("aa")(p1_va, h1_oa, cind) * _a017("bb")(p1_vb, h2_ob, cind),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017( aa )(p1_va, h1_oa, cind) * _a017( bb )(p1_vb, h2_ob, cind)");

    // V^4 term: stored in _a022 when the driver allocated it (CC.store_v4), otherwise a lazy
    // tensor whose blocks are computed from _a021 whenever the contraction below reads them.
    if(a22_abab.is_allocated()) {
      sch(_a022("abab")(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021("aa")(p1_va,p2_va,cind) * _a021("bb")(p2_vb,p1_vb,cind),
      "_a022( abab )(p1_va,p2_vb,p2_va,p1_vb)          =  1.0  * _a021( aa )(p1_va,p2_va,cind) * _a021( bb )(p2_vb,p1_vb,cind)");
    }
    else {
      a22_abab = Tensor<T>::lazy(sch.ec(), {p1_va, p2_vb, p2_va, p1_vb},
                                 1.0 * _a021("aa")(p1_va, p2_va, cind) * _a021("bb")(p2_vb, p1_vb, cind));
    }

    sch
    (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)");


    sch(_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004("abab")(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004( abab )(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p1_vb, p2_vb),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p1_vb, p2_vb)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001("aa")(p1_va, p2_va),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001( aa )(p1_va, p2_va)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("aa")(h1_oa, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( aa )(h1_oa, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("bb")(h1_ob, h2_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( bb )(h1_ob, h2_ob)")
    ;
  // clang-format on
}

template<typename T>
std::tuple<double, double> exachem::cc::ccsd::CD_CCSD_CS<T>::cd_ccsd_cs_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, Tensor<T> dt1_full,
  Tensor<T> dt2_full, bool ccsd_restart, std::string ccsd_fp, bool computeTData) {
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  SystemData&  sys_data    = chem_env.sys_data;
  const int    maxiter     = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  const int    ndiis       = chem_env.ioptions.ccsd_options.ndiis;
  const double thresh      = chem_env.ioptions.ccsd_options.threshold;
  const bool   writet      = chem_env.ioptions.ccsd_options.writet;
  const int    writet_iter = chem_env.ioptions.ccsd_options.writet_iter;
  const double zshiftl     = chem_env.ioptions.ccsd_options.lshift;
  const bool   profile     = chem_env.ioptions.ccsd_options.profile;
  double       residual    = 0.0;
  double       energy      = 0.0;
  int          niter       = 0;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_vir_alpha = static_cast<TAMM_SIZE>(sys_data.n_vir_alpha);

  const std::string t1file = ccsd_fp + ".t1amp";
  const std::string t2file = ccsd_fp + ".t2amp";

  std::cout.precision(15);

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [cind]              = CI.labels<1>("all");

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

  const auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  const auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  const auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
  const auto [h3_ob, h4_ob] = o_beta.labels<2>("all");

  Tensor<T> d_e{};

  Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, {2, 2}};

  CCSE_Tensors<T> f1_oo{MO, {O, O}, "f1_oo", {"aa", "bb"}};
  CCSE_Tensors<T> f1_ov{MO, {O, V}, "f1_ov", {"aa", "bb"}};
  CCSE_Tensors<T> f1_vv{MO, {V, V}, "f1_vv", {"aa", "bb"}};

  CCSE_Tensors<T> chol3d_oo{MO, {O, O, CI}, "chol3d_oo", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_ov{MO, {O, V, CI}, "chol3d_ov", {"aa", "bb"}};
  CCSE_Tensors<T> chol3d_vv{MO, {V, V, CI}, "chol3d_vv", {"aa", "bb"}};

  std::vector<CCSE_Tensors<T>> f1_se{f1_oo, f1_ov, f1_vv};
  std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo, chol3d_ov, chol3d_vv};

  _a01V = {CI};
  _a02  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02", {"aa"}};
  _a03  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03", {"aa"}};
  _a004 = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004", {"aaaa", "abab"}};

  t2_aaaa_temp = {v_alpha, v_alpha, o_alpha, o_alpha};
  i0_temp      = {v_beta, v_alpha, o_beta, o_alpha};

  // Intermediates
  // T1
  _a02V = {CI};
  _a01  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01", {"aa"}};
  _a04  = CCSE_Tensors<T>{MO, {O, O}, "_a04", {"aa"}};
  _a05  = CCSE_Tensors<T>{MO, {O, V}, "_a05", {"aa", "bb"}};
  _a06  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06", {"aa"}};

  // T2
  _a007V = {CI};
  _a001  = CCSE_Tensors<T>{MO, {V, V}, "_a001", {"aa", "bb"}};
  _a006  = CCSE_Tensors<T>{MO, {O, O}, "_a006", {"aa", "bb"}};

  _a008 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008", {"aa"}};
  _a009 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009", {"aa", "bb"}};
  _a017 = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017", {"aa", "bb"}};
  _a021 = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021", {"aa", "bb"}};

  _a019               = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"abab"}};
  const bool store_v4 = chem_env.ioptions.ccsd_options.store_v4;
  if(store_v4) _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"abab"}};
  _a020 = CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}};

  double total_ccsd_mem =
    sum_tensor_sizes(t1_aa, t2_aaaa, t2_abab, d_f1, r1_aa, r2_abab, cv3d, d_e, i0_temp,
                     t2_aaaa_temp, _a01V) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(_a02, _a03);

  for(size_t ri = 0; ri < d_r1s.size(); ri++)
    total_ccsd_mem += sum_tensor_sizes(d_r1s[ri], d_r2s[ri], d_t1s[ri], d_t2s[ri]);

  // Intermediates
  const double v4int_size         = store_v4 ? CCSE_Tensors<T>::sum_tensor_sizes_list(_a022) : 0.0;
  double       total_ccsd_mem_tmp = sum_tensor_sizes(_a02V, _a007V) + v4int_size +
                              CCSE_Tensors<T>::sum_tensor_sizes_list(_a01, _a04, _a05, _a06, _a001,
                                                                     _a004, _a006, _a008, _a009,
                                                                     _a017, _a019, _a020, _a021);

  if(!ccsd_restart) total_ccsd_mem += total_ccsd_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Estimated system RAM required for Closed Shell Cholesky CCSD calculation: "
              << std::fixed << std::setprecision(2) << total_ccsd_mem << " GiB" << std::endl;
    if(store_v4)
      std::cout << " (V^4 intermediate size: " << std::fixed << std::setprecision(2) << v4int_size
                << " GiB)" << std::endl;
  }
  check_memory_requirements(ec, total_ccsd_mem);

  print_ccsd_header(ec.print());

  Scheduler   sch{ec};
  ExecutionHW exhw = ec.exhw();

  sch.allocate(t2_aaaa);
  sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, _a02, _a03);

  // clang-format off
  sch
    (chol3d_oo("aa")(h3_oa,h4_oa,cind) = cv3d(h3_oa,h4_oa,cind))
    (chol3d_ov("aa")(h3_oa,p2_va,cind) = cv3d(h3_oa,p2_va,cind))
    (chol3d_vv("aa")(p1_va,p2_va,cind) = cv3d(p1_va,p2_va,cind))
    (chol3d_oo("bb")(h3_ob,h4_ob,cind) = cv3d(h3_ob,h4_ob,cind))
    (chol3d_ov("bb")(h3_ob,p1_vb,cind) = cv3d(h3_ob,p1_vb,cind))
    (chol3d_vv("bb")(p1_vb,p2_vb,cind) = cv3d(p1_vb,p2_vb,cind))

    (f1_oo("aa")(h3_oa,h4_oa) = d_f1(h3_oa,h4_oa))
    (f1_ov("aa")(h3_oa,p2_va) = d_f1(h3_oa,p2_va))
    (f1_vv("aa")(p1_va,p2_va) = d_f1(p1_va,p2_va))
    (f1_oo("bb")(h3_ob,h4_ob) = d_f1(h3_ob,h4_ob))
    (f1_ov("bb")(h3_ob,p1_vb) = d_f1(h3_ob,p1_vb))
    (f1_vv("bb")(p1_vb,p2_vb) = d_f1(p1_vb,p2_vb));
  // clang-format on

  // for restart, this copy is extra for new CCSD calculations
  ccsd_amp_update(sch, t2_abab, t2_aaaa);

  sch.execute();

  const auto& sv = chem_env.ioptions.ccsd_options.solvers;

  // Enter the iterative block for a fresh run OR whenever Newton-Krylov is selected (NK also
  // resumes from restart amplitudes). A converged non-NK restart falls through to the single energy
  // evaluation in the else branch below.
  if(sv.solver_type == "newton_krylov" || !ccsd_restart) {
    // allocate all intermediates
    sch.allocate(_a02V, _a007V);
    CCSE_Tensors<T>::allocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                   _a017, _a019, _a020, _a021);
    if(store_v4) {
      CCSE_Tensors<T>::allocate_list(sch, _a022);
      a22_abab = _a022("abab");
    }
    sch.execute();

    // clang-format off
    sch
        (r1_aa() = 0)
        (r2_abab() = 0)
        (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_ov("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h3_oa, p2_va, cind))
        .exact_copy(_a004("abab")(p1_va, p1_vb, h3_oa, h3_ob), _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob))
        ;
    // clang-format on

    sch.execute(exhw);

    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    // MP2 initial guess (skipped when amplitude files exist). Closed-shell: only the abab doubles
    // are independent; T2_abab^(1) = (ai|bj) / D from _a004("abab"); T1 stays zero.
    bool ccsd_nk_restart =
      chem_env.ioptions.ccsd_options.readt ||
      ((fs::exists(t1file) && fs::exists(t2file) && fs::exists(chem_env.cd_context.f1file) &&
        fs::exists(chem_env.cd_context.v2file)));

    if(!ccsd_nk_restart) {
      sch(r2_abab(p1_va, p1_vb, h3_oa, h3_ob) = _a004("abab")(p1_va, p1_vb, h3_oa, h3_ob))
        .execute(exhw);
      jacobi_cs(ec, r2_abab, t2_abab, 0.0, false, p_evl_sorted, n_occ_alpha, n_vir_alpha);
    }

    // ===================== Newton-Krylov solver path =====================
    // Reuses the generic exachem::cc::solvers machinery; closed-shell specifics are the reduced
    // aa/abab residual and Jacobi preconditioner.
    chem_env.cc_context.ccsd_iter = -1;

    std::vector<Tensor<T>> d_t{t1_aa, t2_abab};
    std::vector<Tensor<T>> d_r{r1_aa, r2_abab};

    // Residual F(x): x = {t1_aa, t2_abab}. Derive t2_aaaa, evaluate the closed-shell T1/T2
    // residuals and the energy. r1_aa/r2_abab are the (reduced) closed-shell residuals directly.
    auto residual_fun = [&](const std::vector<Tensor<T>>& x) -> std::vector<Tensor<T>>& {
      // ccsd_t2_cs takes t2_abab by non-const ref (reads only); use handle copies (shared data) so
      // the const solver-vector elements bind without mutating them.
      Tensor<T> x_t1 = x.at(0); // t1_aa
      Tensor<T> x_t2 = x.at(1); // t2_abab
      ccsd_amp_update(sch, x_t2, t2_aaaa);
      ccsd_t1_cs(sch, MO, CI, r1_aa, x_t1, x_t2, f1_se, chol3d_se);
      ccsd_t2_cs(sch, MO, CI, r2_abab, x_t1, x_t2, t2_aaaa, f1_se, chol3d_se);
      ccsd_e_cs(sch, MO, CI, d_e, x_t1, x_t2, t2_aaaa, f1_se, chol3d_se);
      sch.execute(exhw, profile);
      return d_r;
    };

    auto preconditioner = [&](std::vector<Tensor<T>> r) {
      auto result = zeroslike(ec, r);
      jacobi_cs(ec, r[0], result[0], -1.0 * zshiftl, false, p_evl_sorted, n_occ_alpha, n_vir_alpha);
      jacobi_cs(ec, r[1], result[1], -2.0 * zshiftl, false, p_evl_sorted, n_occ_alpha, n_vir_alpha);
      scale_tensors(result, -1.0);
      ec.pg().barrier();
      return result;
    };

    if(sv.solver_type == "newton_krylov") {
      std::tie(residual, energy) =
        preconditioned_newton_krylov_solver(ec, chem_env, residual_fun, preconditioner, d_e, d_t);
      // keep t2_aaaa consistent with the converged t2_abab for downstream use
      ccsd_amp_update(sch, t2_abab, t2_aaaa);
      sch.execute(exhw);
    }
    else {
      for(int titer = 0; titer < maxiter; titer += ndiis) {
        for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
          const auto timer_start = std::chrono::high_resolution_clock::now();

          niter   = iter;
          int off = iter - titer;
          // clang-format off
            sch
               ((d_t1s[off])()  = t1_aa())
               ((d_t2s[off])()  = t2_abab())
               .execute();
          // clang-format on

          ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se);
          ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

          sch.execute(exhw, profile);

          std::tie(residual, energy) = rest_cs(ec, MO, r1_aa, r2_abab, t1_aa, t2_abab, d_e,
                                               d_r1_residual, d_r2_residual, p_evl_sorted, zshiftl,
                                               n_occ_alpha, n_vir_alpha);

          update_r2(ec, r2_abab());
          // clang-format off
        sch ((d_r1s[off])() = r1_aa())
            ((d_r2s[off])() = r2_abab())
            .execute();
          // clang-format on

          ccsd_amp_update(sch, t2_abab, t2_aaaa); // update t2_aaaa after jacobi
          CD_CCSD_CS<T>::ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);
          sch.execute(exhw, profile);
          energy = get_scalar(d_e);

          const auto timer_end = std::chrono::high_resolution_clock::now();
          auto       iter_time =
            std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
              .count();

          iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

          if(writet && ((iter + 1) % writet_iter == 0)) {
            write_to_disk(ec, t1_aa, t1file);
            write_to_disk(ec, t2_abab, t2file);
          }

          if(residual < thresh) { break; }
        }

        if(residual < thresh || titer + ndiis >= maxiter) { break; }
        if(ec.pg().rank() == 0) {
          std::cout << " MICROCYCLE DIIS UPDATE:";
          std::cout.width(21);
          std::cout << std::right << std::min(titer + ndiis, maxiter) + 1 << std::endl;
        }

        std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
        std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s};
        std::vector<Tensor<T>>              next_t{t1_aa, t2_abab};
        diis<T>(ec, rs, ts, next_t);

        // update t2_aaaa after diis
        ccsd_amp_update(sch, t2_abab, t2_aaaa);
        sch.execute();
      }
    }

    if(writet) {
      write_to_disk(ec, t1_aa, t1file);
      write_to_disk(ec, t2_abab, t2file);
    }

    if(profile && ec.print()) {
      std::string   profile_csv = ccsd_fp + "_profile.csv";
      std::ofstream pds(profile_csv, std::ios::out);
      if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
      pds << ec.get_profile_header() << std::endl;
      pds << ec.get_profile_data().str() << std::endl;
      pds.close();
    }

    // deallocate all intermediates
    sch.deallocate(_a02V, _a007V, d_r1_residual, d_r2_residual);
    CCSE_Tensors<T>::deallocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                     _a017, _a019, _a020, _a021);
    if(store_v4) CCSE_Tensors<T>::deallocate_list(sch, _a022);

  } // no restart
  else {
    ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

    sch.execute(exhw, profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  chem_env.cc_context.ccsd_correlation_energy = energy;
  chem_env.cc_context.ccsd_total_energy       = chem_env.scf_context.hf_energy + energy;

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CCSD"]["final_energy"]["total"] =
      chem_env.cc_context.ccsd_total_energy;
    sys_data.results["output"]["CCSD"]["performance"]["total_time"] = ccsd_time;
    chem_env.write_json_data();
  }

  sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::deallocate_list(sch, _a02, _a03);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  sch.execute();

  if(computeTData) {
    Tensor<T> d_t1 = dt1_full;
    Tensor<T> d_t2 = dt2_full;

    // IndexVector perm1 = {1,0,3,2};
    // IndexVector perm2 = {0,1,3,2};
    // IndexVector perm3 = {1,0,2,3};

    // t1_aa, t1_bb, t2_aaaa, t2_abab, t2_bbbb,t2_baba, t2_abba, t2_baab
    Tensor<T> t1_bb{v_beta, o_beta};
    Tensor<T> t2_bbbb{v_beta, v_beta, o_beta, o_beta};
    Tensor<T> t2_baba{v_beta, v_alpha, o_beta, o_alpha};
    Tensor<T> t2_abba{v_alpha, v_beta, o_beta, o_alpha};
    Tensor<T> t2_baab{v_beta, v_alpha, o_alpha, o_beta};

    // clang-format off
    sch.allocate(t1_bb,t2_bbbb,t2_baba,t2_abba,t2_baab)
    .exact_copy(t1_bb(p1_vb,h3_ob),  t1_aa(p1_vb,h3_ob))
    .exact_copy(t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob),  t2_aaaa(p1_vb,p2_vb,h3_ob,h4_ob)).execute();

    // .exact_copy(t2_baba(p1_vb,p2_va,h3_ob,h4_oa),  t2_abab(p1_vb,p2_va,h3_ob,h4_oa),true,1.0,perm)
    // .exact_copy(t2_abba(p1_va,p2_vb,h3_ob,h4_oa),  t2_abab(p1_va,p2_vb,h3_ob,h4_oa),true,-1.0)
    // .exact_copy(t2_baab(p1_vb,p2_va,h3_oa,h4_ob),  t2_abab(p1_vb,p2_va,h3_oa,h4_ob),true,-1.0)

    // sch.exact_copy(t2_baba,t2_abab,true, 1.0,perm1);
    // sch.exact_copy(t2_abba,t2_abab,true,-1.0,perm2);
    // sch.exact_copy(t2_baab,t2_abab,true,-1.0,perm3);

    sch
    (t2_baba(p2_vb,p1_va,h4_ob,h3_oa) =        t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_abba(p1_va,p2_vb,h4_ob,h3_oa) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (t2_baab(p2_vb,p1_va,h3_oa,h4_ob) = -1.0 * t2_abab(p1_va,p2_vb,h3_oa,h4_ob))

    (d_t1(p1_va,h3_oa)             = t1_aa(p1_va,h3_oa))
    (d_t1(p1_vb,h3_ob)             = t1_bb(p1_vb,h3_ob))
    (d_t2(p1_va,p2_va,h3_oa,h4_oa) = t2_aaaa(p1_va,p2_va,h3_oa,h4_oa))
    (d_t2(p1_va,p2_vb,h3_oa,h4_ob) = t2_abab(p1_va,p2_vb,h3_oa,h4_ob))
    (d_t2(p1_vb,p2_vb,h3_ob,h4_ob) = t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob))

    (d_t2(p1_vb,p2_va,h3_ob,h4_oa) = t2_baba(p1_vb,p2_va,h3_ob,h4_oa))
    (d_t2(p1_va,p2_vb,h3_ob,h4_oa) = t2_abba(p1_va,p2_vb,h3_ob,h4_oa))
    (d_t2(p1_vb,p2_va,h3_oa,h4_ob) = t2_baab(p1_vb,p2_va,h3_oa,h4_ob))
    .deallocate(t1_bb,t2_bbbb,t2_baba,t2_abba,t2_baab)
    .execute();
    // clang-format on
  }

  sch.deallocate(t2_aaaa).execute();

  return std::make_tuple(residual, energy);
}

template class exachem::cc::ccsd::CD_CCSD_CS<double>;
