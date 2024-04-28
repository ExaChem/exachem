/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cd_cc2_cs.hpp"

namespace cc2_cs {

using CCEType = double;
CCSE_Tensors<CCEType> _a021;
TiledIndexSpace       o_alpha, v_alpha, o_beta, v_beta;

Tensor<CCEType>       _a01V, _a02V, _a007V;
CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009, _a017,
  _a019, _a020; //_a022

Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only
};                                     // namespace cc2_cs

template<typename T>
void cc2_cs::cc2_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                      Tensor<T>& de, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                      const Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                      std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
    (t2_aaaa_temp() = t2_aaaa(), 
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa), 
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa), 
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)")

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
  //    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
  //    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM: zero for closed-shell case 
    ;
  // clang-format on
}

template<typename T>
void cc2_cs::cc2_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                       Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                       std::vector<CCSE_Tensors<T>>& f1_se,
                       std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");
  auto [p2]   = MO.labels<1>("virt");
  auto [h1]   = MO.labels<1>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

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
  // h terms do not have contributions for closed-shell case 
  // h i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va), 
  // h i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va)   
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind), 
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind), 
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01("aa")(h1_oa, h1_oa, cind))
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * _a01("aa")(h2_oa, h1_oa, cind), 
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h1_oa, p1_va, cind) * _a01( aa )(h2_oa, h1_oa, cind)")      // o2vm
    //h _a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va),
    //h _a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va) // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06("aa")(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov("aa")(h2_oa, p2_va, cind), 
    "_a06( aa )(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov( aa )(h2_oa, p2_va, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa), 
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind), 
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    //h _a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    //h _a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va) // NEW TERM
    (i0_aa(p2_va, h1_oa)            =  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa), 
    "i0_aa(p2_va, h1_oa)            =  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
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
void cc2_cs::cc2_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                       Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab,
                       Tensor<T>& t2_aaaa, std::vector<CCSE_Tensors<T>>& f1_se,
                       std::vector<CCSE_Tensors<T>>& chol3d_se, Tensor<T>& d_f1, Tensor<T>& cv3d,
                       Tensor<T>& res_2) {
  auto [cind]                  = CI.labels<1>("all");
  auto [p3, p4, a, b, c, d, e] = MO.labels<7>("virt");
  auto [h1, h2, i, j, k, l, m] = MO.labels<7>("occ");

  auto [p1_va, p2_va, p3_va, p4_va] = v_alpha.labels<4>("all");
  auto [p1_vb, p2_vb, p3_vb, p4_vb] = v_beta.labels<4>("all");
  auto [h1_oa, h2_oa, h3_oa, h4_oa] = o_alpha.labels<4>("all");
  auto [h1_ob, h2_ob, h3_ob, h4_ob] = o_beta.labels<4>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];
  // auto hw     = sch.ec().exhw();
  // auto rank = sch.ec().pg().rank();

  Tensor<T> ph{{a, i, cind}, {1, 1}};
  Tensor<T> ph_1{{a, i, cind}, {1, 1}};
  Tensor<T> ph_2{{a, i, cind}, {1, 1}};
  Tensor<T> ph_3{{a, i, cind}, {1, 1}};
  // Tensor<T> ph_4{{a, i, cind}, {1, 1}};
  Tensor<T> hh{{i, j, cind}, {1, 1}};

  Tensor<T> d_t1{{a, i}, {1, 1}};
  Tensor<T> d_t2{{a, b, i, j}, {2, 2}};

  Tensor<T> t1_bb{v_beta, o_beta};
  Tensor<T> t2_bbbb{v_beta, v_beta, o_beta, o_beta};
  Tensor<T> t2_baba{v_beta, v_alpha, o_beta, o_alpha};
  Tensor<T> t2_abba{v_alpha, v_beta, o_beta, o_alpha};
  Tensor<T> t2_baab{v_beta, v_alpha, o_alpha, o_beta};

  // clang-format off
  sch.allocate(t1_bb,t2_bbbb,t2_baba,t2_abba,t2_baab, d_t1, d_t2)
  .exact_copy(t1_bb(p1_vb,h3_ob),  t1_aa(p1_vb,h3_ob))
  .exact_copy(t2_bbbb(p1_vb,p2_vb,h3_ob,h4_ob),  t2_aaaa(p1_vb,p2_vb,h3_ob,h4_ob)).execute();

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

  Tensor<T> i0{{a, b, i, j}, {2, 2}};
  Tensor<T> pphh{{a, b, i, j}, {2, 2}};
  sch.allocate(ph, ph_1, ph_2, ph_3, hh, i0, pphh).execute();

  // clang-format off
  sch(i0(a, b, i, j)          =                            res_2(a, b, i, j))
     (pphh(a, b, i, j)        = d_f1(a, c)          *       d_t2(b, c, j, i))
     (i0(a, b, i, j)         +=                             pphh(a, b, i, j))
     (i0(a, b, i, j)         += -1.0                *       pphh(b, a, i, j))
     (pphh(a, b, i, j)        = d_f1(k, i)          *       d_t2(b, a, j, k))
     (i0(a, b, i, j)         += -1.0                *       pphh(a, b, i, j))
     (i0(a, b, i, j)         +=                             pphh(a, b, j, i))
     (ph(b, j, cind)          = cv3d(b, c, cind)    *             d_t1(c, j))  // 4
     (ph_2(b, j, cind)        = ph(b, j, cind)                              )
     (pphh(a, b, i, j)        = ph(b, j, cind)      *       cv3d(a, i, cind))
      (i0(a, b, i, j)        +=                             pphh(a, b, i, j))
      (i0(a, b, i, j)        +=-1.0                 *       pphh(b, a, i, j))
      (i0(a, b, i, j)        +=-1.0                 *       pphh(a, b, j, i))
      (i0(a, b, i, j)        +=                             pphh(b, a, j, i))
      (ph_1(b, i, cind)       = cv3d(k, i, cind)    *             d_t1(b, k))
      (ph_3(b, i, cind)       =                             ph_1(b, i, cind))
      (pphh(a, b, i, j)       = ph(a, j, cind)      *       ph_1(b, i, cind))
      (i0(a, b, i, j)        +=                             pphh(a, b, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, j, i))
      (i0(a, b, i, j)        += -1.0                *       pphh(b, a, i, j))
      (i0(a, b, i, j)        +=                             pphh(b, a, j, i))
      (pphh(a, b, i, j)       = ph(a, i, cind)      *         ph(b, j, cind))  //8
      (i0(a, b, i, j)        += pphh(a, b, i, j))
      (i0(a, b, i, j)        +=-1.0                 *       pphh(a, b, j, i))
      (pphh(a, b, i, j)       = ph_1(b, j, cind)    *       cv3d(a, i, cind))   // 5
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, i, j))
      (i0(a, b, i, j)        +=                             pphh(a, b, j, i))
      (i0(a, b, i, j)        +=                             pphh(b, a, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(b, a, j, i))
      (pphh(a, b, i, j)       = ph_1(a, i, cind)    *       ph_1(b, j, cind))   //9
      (i0(a, b, i, j)        += pphh(a, b, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, j, i)) 
      (hh(k, j, cind)         = cv3d(k, c, cind)    *             d_t1(c, j))
      (ph(b, j, cind)         = hh(k, j, cind)      *             d_t1(b, k))
      (pphh(a, b, i, j)       = cv3d(a, i, cind)    *         ph(b, j, cind))
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, i, j))
      (i0(a, b, i, j)        +=                             pphh(a, b, j, i)) 
      (i0(a, b, i, j)        +=                             pphh(b, a, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(b, a, j, i))
      (pphh(a, b, i, j)       = ph(b, j, cind)      *       ph_2(a, i, cind))   //11
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, i, j))
      (i0(a, b, i, j)        +=                             pphh(a, b, j, i))
      (i0(a, b, i, j)        +=                             pphh(b, a, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(b, a, j, i))
      (pphh(a, b, i, j)       = ph(a, i, cind)      *       ph_3(b, j, cind))  //12
      (i0(a, b, i, j)        +=                             pphh(a, b, i, j))
      (i0(a, b, i, j)        += -1.0                *       pphh(a, b, j, i))
      (i0(a, b, i, j)        += -1.0                *       pphh(b, a, i, j))
      (i0(a, b, i, j)        +=                             pphh(b, a, j, i))
      (pphh(a, b, i, j)       = ph(a, i, cind)      *         ph(b, j, cind))  // 13
        (i0(a, b, i, j)      +=                             pphh(a, b, i, j))
        (i0(a, b, i, j)      += -1.0                *       pphh(a, b, j, i));

    // f(k, c)t1(c, i)*t2(a, b, k, j)  is not added as f(k, c) is zero for RHF case
    // f(k, c) t1(a, k)*t2(c, b, i, j) is also not added for RHF case
  // clang-format on
  sch(i0_abab(p3_va, p4_vb, h1_oa, h2_ob) = i0(p3_va, p4_vb, h1_oa, h2_ob));
  sch.deallocate(ph_2, ph_3, hh, i0, pphh).execute();
}

template<typename T>
std::tuple<double, double> cc2_cs::cd_cc2_cs_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, Tensor<T> dt1_full,
  Tensor<T> dt2_full, bool cc2_restart, std::string cd_cc2_fp, bool computeTData) {
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
  const TAMM_SIZE n_vir_alpha = static_cast<TAMM_SIZE>(sys_data.n_vir_alpha);

  std::string t1file = cd_cc2_fp + ".t1amp";
  std::string t2file = cd_cc2_fp + ".t2amp";

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

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
  auto [h3_ob, h4_ob] = o_beta.labels<2>("all");

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

  _a019 = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"abab"}};
  // _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"abab"}};
  _a020 = CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}};

  double total_cc2_mem =
    sum_tensor_sizes(t1_aa, t2_aaaa, t2_abab, d_f1, r1_aa, r2_abab, cv3d, d_e, i0_temp,
                     t2_aaaa_temp, _a01V) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv) +
    CCSE_Tensors<T>::sum_tensor_sizes_list(_a02, _a03);

  for(size_t ri = 0; ri < d_r1s.size(); ri++)
    total_cc2_mem += sum_tensor_sizes(d_r1s[ri], d_r2s[ri], d_t1s[ri], d_t2s[ri]);

  // Intermediates
  // const double v4int_size         = CCSE_Tensors<T>::sum_tensor_sizes_list(_a022);
  double total_cc2_mem_tmp = sum_tensor_sizes(_a02V, _a007V) + /*v4int_size +*/
                             CCSE_Tensors<T>::sum_tensor_sizes_list(_a01, _a04, _a05, _a06, _a001,
                                                                    _a004, _a006, _a008, _a009,
                                                                    _a017, _a019, _a020, _a021);

  if(!cc2_restart) total_cc2_mem += total_cc2_mem_tmp;

  if(ec.print()) {
    std::cout << std::endl
              << "Total CPU memory required for Closed Shell Cholesky CC2 calculation: "
              << std::fixed << std::setprecision(2) << total_cc2_mem << " GiB" << std::endl;
  }
  check_memory_requirements(ec, total_cc2_mem);

  print_ccsd_header(ec.print(), "CC2");

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

  sch.execute();

  if(!cc2_restart) {
    // allocate all intermediates
    sch.allocate(_a02V, _a007V);
    CCSE_Tensors<T>::allocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                   _a017, _a019, _a020, _a021); //_a022
    sch.execute();

    // clang-format off
    sch
      (r1_aa() = 0)
      (r2_abab() = 0);
      // (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_ov("aa")(h4_oa, p1_va, cind) * chol3d_ov("aa")(h3_oa, p2_va, cind))
      //.exact_copy(_a004("abab")(p1_va, p1_vb, h3_oa, h3_ob), _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob))
    // clang-format on

    sch.execute(exhw);

    //--------------------------------------
    auto [cind]                  = CI.labels<1>("all");
    auto [p3, p4, a, b, c, d, e] = MO.labels<7>("virt");
    auto [h1, h2, i, j, k, l, m] = MO.labels<7>("occ");
    Tensor<T> res_2{{a, b, i, j}, {2, 2}};
    Tensor<T> pphh{{a, b, i, j}, {2, 2}};
    sch.allocate(res_2, pphh).execute();

    // clang-format off
    sch(pphh(a, b, i, j)  = cv3d(a, i, cind) * cv3d(b, j, cind))
      (res_2(a, b, i, j)  = pphh(a, b, i, j))
      (res_2(a, b, i, j) += -1.0*pphh(a, b, j, i));
    // clang-format on

    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

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

        cc2_cs::cc2_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);
        cc2_cs::cc2_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se);
        cc2_cs::cc2_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se, d_f1,
                          cv3d, res_2);

        sch.execute(exhw, profile);

        std::tie(residual, energy) = rest_cs(ec, MO, r1_aa, r2_abab, t1_aa, t2_abab, d_e,
                                             d_r1_residual, d_r2_residual, p_evl_sorted, zshiftl,
                                             n_occ_alpha, n_vir_alpha);

        update_r2(ec, r2_abab());
        // clang-format off
        sch((d_r1s[off])() = r1_aa())
            ((d_r2s[off])() = r2_abab())
            .execute();
        // clang-format on

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

        if(writet && (((iter + 1) % writet_iter == 0) || (residual < thresh))) {
          write_to_disk(t1_aa, t1file);
          write_to_disk(t2_abab, t2file);
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
    }

    if(profile && ec.print()) {
      std::string   profile_csv = cd_cc2_fp + "_profile.csv";
      std::ofstream pds(profile_csv, std::ios::out);
      if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
      pds << ec.get_profile_header() << std::endl;
      pds << ec.get_profile_data().str() << std::endl;
      pds.close();
    }

    // deallocate all intermediates
    sch.deallocate(_a02V, _a007V, d_r1_residual, d_r2_residual);
    CCSE_Tensors<T>::deallocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                     _a017, _a019, _a020, _a021); //_a022

  } // no restart
  else {
    cc2_cs::cc2_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

    sch.execute(exhw, profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  sys_data.cc2_corr_energy = energy;

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CC2"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CC2"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CC2"]["final_energy"]["total"]       = sys_data.scf_energy + energy;
    chem_env.write_json_data("CC2");
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

using T = double;
template std::tuple<double, double> cc2_cs::cd_cc2_cs_driver<T>(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& d_f1, Tensor<T>& r1_aa, Tensor<T>& r2_abab,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<T>& p_evl_sorted, Tensor<T>& cv3d, Tensor<T> dt1_full,
  Tensor<T> dt2_full, bool cc2_restart, std::string out_fp, bool computeTData);
