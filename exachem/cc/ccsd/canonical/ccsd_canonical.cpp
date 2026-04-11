/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/ccsd/canonical/ccsd_canonical.hpp"
#include "exachem/cc/ccsd/ccsd_util.hpp"
using namespace tamm;

namespace exachem::cc::ccsd_canonical {

template<typename T>
void ccsd_e(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
            const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
  //                                        SpinPosition::lower};

  Tensor<T> i1{{O, V}, {1, 1}};

  TiledIndexLabel p1, p2, p3, p4, p5;
  TiledIndexLabel h3, h4, h5, h6;

  std::tie(p1, p2, p3, p4, p5) = MO.labels<5>("virt");
  std::tie(h3, h4, h5, h6)     = MO.labels<4>("occ");

  // clang-format off
  sch.allocate(i1)
    (i1(h6, p5) = f1(h6, p5))
    (i1(h6, p5) += 0.5 * t1(p3, h4) * v2(h4, h6, p3, p5))
    (de() = 0)
    (de() += t1(p5, h6) * i1(h6, p5))
    (de() += 0.25 * t2(p1, p2, h3, h4) * v2(h3, h4, p1, p2))
    .deallocate(i1);
  // clang-format on
}

template<typename T>
void ccsd_t1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
             const Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
  //                                        SpinPosition::lower};

  // std::vector<SpinPosition> {2,2}{SpinPosition::upper,SpinPosition::upper,
  //                                        SpinPosition::lower,SpinPosition::lower};

  Tensor<T> t1_2_1{{O, O}, {1, 1}};
  Tensor<T> t1_2_2_1{{O, V}, {1, 1}};
  Tensor<T> t1_3_1{{V, V}, {1, 1}};
  Tensor<T> t1_5_1{{O, V}, {1, 1}};
  Tensor<T> t1_6_1{{O, O, O, V}, {2, 2}};

  TiledIndexLabel p2, p3, p4, p5, p6, p7;
  TiledIndexLabel h1, h4, h5, h6, h7, h8;

  std::tie(p2, p3, p4, p5, p6, p7) = MO.labels<6>("virt");
  std::tie(h1, h4, h5, h6, h7, h8) = MO.labels<6>("occ");

  // clang-format off
  sch
    .allocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1)
    (t1_2_1(h7, h1) = 0)
    (t1_3_1(p2, p3)  = 0)
    ( i0(p2,h1)            =        f1(p2,h1))
    ( t1_2_1(h7,h1)        =        f1(h7,h1))
    ( t1_2_2_1(h7,p3)      =        f1(h7,p3))
    ( t1_2_2_1(h7,p3)     += -1   * t1(p5,h6)       * v2(h6,h7,p3,p5))
    ( t1_2_1(h7,h1)       +=        t1(p3,h1)       * t1_2_2_1(h7,p3))
    ( t1_2_1(h7,h1)       += -1   * t1(p4,h5)       * v2(h5,h7,h1,p4))
    ( t1_2_1(h7,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2(h5,h7,p3,p4))
    ( i0(p2,h1)           += -1   * t1(p2,h7)       * t1_2_1(h7,h1))
    ( t1_3_1(p2,p3)        =        f1(p2,p3))
    ( t1_3_1(p2,p3)       += -1   * t1(p4,h5)       * v2(h5,p2,p3,p4))
    ( i0(p2,h1)           +=        t1(p3,h1)       * t1_3_1(p2,p3))
    ( i0(p2,h1)           += -1   * t1(p3,h4)       * v2(h4,p2,h1,p3))
    ( t1_5_1(h8,p7)        =        f1(h8,p7))
    ( t1_5_1(h8,p7)       +=        t1(p5,h6)       * v2(h6,h8,p5,p7))
    ( i0(p2,h1)           +=        t2(p2,p7,h1,h8) * t1_5_1(h8,p7))
    ( t1_6_1(h4,h5,h1,p3)  =        v2(h4,h5,h1,p3))
    ( t1_6_1(h4,h5,h1,p3) += -1   * t1(p6,h1)       * v2(h4,h5,p3,p6))
    ( i0(p2,h1)           += -0.5 * t2(p2,p3,h4,h5) * t1_6_1(h4,h5,h1,p3))
    ( i0(p2,h1)           += -0.5 * t2(p3,p4,h1,h5) * v2(h5,p2,p3,p4))
  .deallocate(t1_2_1, t1_2_2_1, t1_3_1, t1_5_1, t1_6_1);
  // clang-format on
}

template<typename T>
void ccsd_t2(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
             Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
  //                                        SpinPosition::lower};

  // std::vector<SpinPosition> {2,2}{SpinPosition::upper,SpinPosition::upper,
  //                                        SpinPosition::lower,SpinPosition::lower};

  Tensor<T> i0_temp{{V, V, O, O}, {2, 2}};
  Tensor<T> t2_temp{{V, V, O, O}, {2, 2}};
  Tensor<T> t2_2_1{{O, V, O, O}, {2, 2}};
  Tensor<T> t2_2_1_temp{{O, V, O, O}, {2, 2}};
  Tensor<T> t2_2_2_1{{O, O, O, O}, {2, 2}};
  Tensor<T> t2_2_2_1_temp{{O, O, O, O}, {2, 2}};
  Tensor<T> t2_2_2_2_1{{O, O, O, V}, {2, 2}};
  Tensor<T> t2_2_4_1{{O, V}, {1, 1}};
  Tensor<T> t2_2_5_1{{O, O, O, V}, {2, 2}};
  Tensor<T> t2_4_1{{O, O}, {1, 1}};
  Tensor<T> t2_4_2_1{{O, V}, {1, 1}};
  Tensor<T> t2_5_1{{V, V}, {1, 1}};
  Tensor<T> t2_6_1{{O, O, O, O}, {2, 2}};
  Tensor<T> t2_6_1_temp{{O, O, O, O}, {2, 2}};
  Tensor<T> t2_6_2_1{{O, O, O, V}, {2, 2}};
  Tensor<T> t2_7_1{{O, V, O, V}, {2, 2}};
  Tensor<T> vt1t1_1{{O, V, O, O}, {2, 2}};
  Tensor<T> vt1t1_1_temp{{O, V, O, O}, {2, 2}};

  TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9;
  TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11;

  std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9)           = MO.labels<9>("virt");
  std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11) = MO.labels<11>("occ");

  // clang-format off
  sch.allocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
            t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
            t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp)
  (i0(p3, p4, h1, h2) = v2(p3, p4, h1, h2))
  (t2_4_1(h9, h1) = 0)
  (t2_5_1(p3, p5) = 0)
  (t2_2_1(h10, p3, h1, h2) = v2(h10, p3, h1, h2))

  (t2_2_2_1(h10, h11, h1, h2) = -1 * v2(h10, h11, h1, h2))
  (t2_2_2_2_1(h10, h11, h1, p5) = v2(h10, h11, h1, p5))
  (t2_2_2_2_1(h10, h11, h1, p5) += -0.5 * t1(p6, h1) * v2(h10, h11, p5, p6))

  // (t2_2_2_1(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
  // (t2_2_2_1(h10, h11, h2, h1) += -1 * t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5)) //perm symm
  (t2_2_2_1_temp(h10, h11, h1, h2) = 0)
  (t2_2_2_1_temp(h10, h11, h1, h2) += t1(p5, h1) * t2_2_2_2_1(h10, h11, h2, p5))
  (t2_2_2_1(h10, h11, h1, h2) += t2_2_2_1_temp(h10, h11, h1, h2))
  (t2_2_2_1(h10, h11, h2, h1) += -1 * t2_2_2_1_temp(h10, h11, h1, h2)) //perm symm

  (t2_2_2_1(h10, h11, h1, h2) += -0.5 * t2(p7, p8, h1, h2) * v2(h10, h11, p7, p8))
  (t2_2_1(h10, p3, h1, h2) += 0.5 * t1(p3, h11) * t2_2_2_1(h10, h11, h1, h2))
  
  (t2_2_4_1(h10, p5) = f1(h10, p5))
  (t2_2_4_1(h10, p5) += -1 * t1(p6, h7) * v2(h7, h10, p5, p6))
  (t2_2_1(h10, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_2_4_1(h10, p5))
  (t2_2_5_1(h7, h10, h1, p9) = v2(h7, h10, h1, p9))
  (t2_2_5_1(h7, h10, h1, p9) += t1(p5, h1) * v2(h7, h10, p5, p9))

  // (t2_2_1(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
  // (t2_2_1(h10, p3, h2, h1) += -1 * t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9)) //perm symm
  (t2_2_1_temp(h10, p3, h1, h2) = 0)
  (t2_2_1_temp(h10, p3, h1, h2) += t2(p3, p9, h1, h7) * t2_2_5_1(h7, h10, h2, p9))
  (t2_2_1(h10, p3, h1, h2) += t2_2_1_temp(h10, p3, h1, h2))
  (t2_2_1(h10, p3, h2, h1) += -1 * t2_2_1_temp(h10, p3, h1, h2)) //perm symm

  // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
  // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
  // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  (t2_temp(p1, p2, h3, h4) = 0)
  (t2_temp(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
  (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
  (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
  (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
  (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

  (t2_2_1(h10, p3, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(h10, p3, p5, p6))
  // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
  // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
  // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
  (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
  (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perm
  (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perm
  
  // (i0(p3, p4, h1, h2) += -1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
  // (i0(p4, p3, h1, h2) += 1 * t1(p3, h10) * t2_2_1(h10, p4, h1, h2)) //perm sym
  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += t1(p3, h10) * t2_2_1(h10, p4, h1, h2))
  (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
  (i0(p4, p3, h1, h2) += i0_temp(p3, p4, h1, h2)) //perm sym


  //  (i0(p3, p4, h1, h2) += -1 * t1(p5, h1) * v2(p3, p4, h2, p5))
  //  (i0(p3, p4, h2, h1) += 1 * t1(p5, h1) * v2(p3, p4, h2, p5)) //perm sym
  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += t1(p5, h1) * v2(p3, p4, h2, p5))
  (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
  (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym

  (t2_4_1(h9, h1) = f1(h9, h1))
  (t2_4_2_1(h9, p8) = f1(h9, p8))
  (t2_4_2_1(h9, p8) += t1(p6, h7) * v2(h7, h9, p6, p8))
  (t2_4_1(h9, h1) += t1(p8, h1) * t2_4_2_1(h9, p8))
  (t2_4_1(h9, h1) += -1 * t1(p6, h7) * v2(h7, h9, h1, p6))
  (t2_4_1(h9, h1) += -0.5 * t2(p6, p7, h1, h8) * v2(h8, h9, p6, p7))

  // (i0(p3, p4, h1, h2) += -1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
  // (i0(p3, p4, h2, h1) += 1 * t2(p3, p4, h1, h9) * t2_4_1(h9, h2)) //perm sym
  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += t2(p3, p4, h1, h9) * t2_4_1(h9, h2))
  (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
  (i0(p3, p4, h2, h1) += i0_temp(p3, p4, h1, h2)) //perm sym


  (t2_5_1(p3, p5) = f1(p3, p5))
  (t2_5_1(p3, p5) += -1 * t1(p6, h7) * v2(h7, p3, p5, p6))
  (t2_5_1(p3, p5) += -0.5 * t2(p3, p6, h7, h8) * v2(h7, h8, p5, p6))

  // (i0(p3, p4, h1, h2) += 1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
  // (i0(p4, p3, h1, h2) += -1 * t2(p3, p5, h1, h2) * t2_5_1(p4, p5)) //perm sym
  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h2) * t2_5_1(p4, p5))
  (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
  (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm sym

  (t2_6_1(h9, h11, h1, h2) = -1 * v2(h9, h11, h1, h2))
  (t2_6_2_1(h9, h11, h1, p8) = v2(h9, h11, h1, p8))
  (t2_6_2_1(h9, h11, h1, p8) += 0.5 * t1(p6, h1) * v2(h9, h11, p6, p8))
  
  // (t2_6_1(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
  // (t2_6_1(h9, h11, h2, h1) += -1 * t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8)) //perm symm
  (t2_6_1_temp(h9, h11, h1, h2) = 0)
  (t2_6_1_temp(h9, h11, h1, h2) += t1(p8, h1) * t2_6_2_1(h9, h11, h2, p8))
  (t2_6_1(h9, h11, h1, h2) += t2_6_1_temp(h9, h11, h1, h2))
  (t2_6_1(h9, h11, h2, h1) += -1 * t2_6_1_temp(h9, h11, h1, h2)) //perm symm

  (t2_6_1(h9, h11, h1, h2) += -0.5 * t2(p5, p6, h1, h2) * v2(h9, h11, p5, p6))
  (i0(p3, p4, h1, h2) += -0.5 * t2(p3, p4, h9, h11) * t2_6_1(h9, h11, h1, h2))

  (t2_7_1(h6, p3, h1, p5) = v2(h6, p3, h1, p5))
  (t2_7_1(h6, p3, h1, p5) += -1 * t1(p7, h1) * v2(h6, p3, p5, p7))
  (t2_7_1(h6, p3, h1, p5) += -0.5 * t2(p3, p7, h1, h8) * v2(h6, h8, p5, p7))

  // (i0(p3, p4, h1, h2) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
  // (i0(p3, p4, h2, h1) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //4 perms
  // (i0(p4, p3, h1, h2) += 1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm
  // (i0(p4, p3, h2, h1) += -1 * t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5)) //perm

  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += t2(p3, p5, h1, h6) * t2_7_1(h6, p4, h2, p5))
  (i0(p3, p4, h1, h2) += -1 * i0_temp(p3, p4, h1, h2))
  (i0(p3, p4, h2, h1) +=  1 * i0_temp(p3, p4, h1, h2)) //4 perms
  (i0(p4, p3, h1, h2) +=  1 * i0_temp(p3, p4, h1, h2)) //perm
  (i0(p4, p3, h2, h1) += -1 * i0_temp(p3, p4, h1, h2)) //perm

  //(vt1t1_1(h5, p3, h1, h2) = 0)
  //(vt1t1_1(h5, p3, h1, h2) += -2 * t1(p6, h1) * v2(h5, p3, h2, p6))
  //(vt1t1_1(h5, p3, h2, h1) += 2 * t1(p6, h1) * v2(h5, p3, h2, p6)) //perm symm
  (vt1t1_1_temp()=0)
  (vt1t1_1_temp(h5, p3, h1, h2) += t1(p6, h1) * v2(h5, p3, h2, p6))
  (vt1t1_1(h5, p3, h1, h2) = -2 * vt1t1_1_temp(h5, p3, h1, h2))
  (vt1t1_1(h5, p3, h2, h1) += 2 * vt1t1_1_temp(h5, p3, h1, h2)) //perm symm

  // (i0(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
  // (i0(p4, p3, h1, h2) += 0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2)) //perm symm
  (i0_temp(p3, p4, h1, h2) = 0)
  (i0_temp(p3, p4, h1, h2) += -0.5 * t1(p3, h5) * vt1t1_1(h5, p4, h1, h2))
  (i0(p3, p4, h1, h2) += i0_temp(p3, p4, h1, h2))
  (i0(p4, p3, h1, h2) += -1 * i0_temp(p3, p4, h1, h2)) //perm symm

  // (t2(p1, p2, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4))
  // (t2(p1, p2, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
  // (t2(p2, p1, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  // (t2(p2, p1, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perm
  (t2(p1, p2, h3, h4) += t2_temp(p1, p2, h3, h4))
  (t2(p1, p2, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //4 perms
  (t2(p2, p1, h3, h4) += -1 * t2_temp(p1, p2, h3, h4)) //perm
  (t2(p2, p1, h4, h3) += t2_temp(p1, p2, h3, h4)) //perm

  (i0(p3, p4, h1, h2) += 0.5 * t2(p5, p6, h1, h2) * v2(p3, p4, p5, p6))
  
  // (t2(p1, p2, h3, h4) += -0.5 * t1(p1, h3) * t1(p2, h4))
  // (t2(p1, p2, h4, h3) += 0.5 * t1(p1, h3) * t1(p2, h4)) //4 perms
  // (t2(p2, p1, h3, h4) += 0.5 * t1(p1, h3) * t1(p2, h4)) //perms
  // (t2(p2, p1, h4, h3) += -0.5 * t1(p1, h3) * t1(p2, h4)) //perms
  (t2(p1, p2, h3, h4) += -1 * t2_temp(p1, p2, h3, h4))
  (t2(p1, p2, h4, h3) += t2_temp(p1, p2, h3, h4)) //4 perms
  (t2(p2, p1, h3, h4) += t2_temp(p1, p2, h3, h4)) //perms
  (t2(p2, p1, h4, h3) += -1 * t2_temp(p1, p2, h3, h4)) //perms

  .deallocate(t2_2_1, t2_2_2_1, t2_2_2_2_1, t2_2_4_1, t2_2_5_1, t2_4_1, t2_4_2_1,
            t2_5_1, t2_6_1, t2_6_2_1, t2_7_1, vt1t1_1,vt1t1_1_temp,t2_2_2_1_temp,
            t2_2_1_temp,i0_temp,t2_temp,t2_6_1_temp);
  // sch.execute();
  // clang-format on
}

template<typename TensorType>
void update_ints2e_hubbard(Tensor<TensorType> tensor, const double U_val) {
  LabeledTensor<TensorType> ltensor = tensor();

  ExecutionContext& ec = get_ec(ltensor);
  EXPECTS(tensor.num_modes() == 4);

  auto update_diagonal = [&](const IndexVector& bid) {
    const IndexVector       blockid = internal::translate_blockid(bid, ltensor);
    const tamm::TAMM_SIZE   dsize   = tensor.block_size(blockid);
    std::vector<TensorType> dbuf(dsize);
    tensor.get(blockid, dbuf);
    auto block_dims   = tensor.block_dims(blockid);
    auto block_offset = tensor.block_offsets(blockid);

    size_t c = 0;
    for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
      for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
        for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++) {
          for(size_t l = block_offset[3]; l < block_offset[3] + block_dims[3]; l++, c++) {
            if(i == j && i == k && i == l) { dbuf[c] = U_val; }
          }
        }
      }
    }
    tensor.put(blockid, dbuf);
  };
  block_for(ec, ltensor, update_diagonal);

  return;
}

template<typename T>
std::tuple<double, double>
ccsd_v2_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_t1,
               Tensor<T>& d_t2, Tensor<T>& d_f1, Tensor<T>& d_v2, Tensor<T>& d_r1, Tensor<T>& d_r2,
               std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s,
               std::vector<Tensor<T>>& d_t1s, std::vector<Tensor<T>>& d_t2s,
               std::vector<T>& p_evl_sorted, bool ccsd_restart = false, std::string ccsd_fp = "") {
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

  Tensor<T> d_e{};
  Tensor<T>::allocate(&ec, d_e);
  Scheduler sch{ec};

  print_ccsd_header(ec.print());

  if(!ccsd_restart) {
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;

        sch((d_t1s[off])() = d_t1())((d_t2s[off])() = d_t2()).execute();

        ccsd_canonical::ccsd_t1(sch, MO, d_r1, d_t1, d_t2, d_f1, d_v2);
        ccsd_canonical::ccsd_t2(sch, MO, d_r2, d_t1, d_t2, d_f1, d_v2);

        sch.execute(ec.exhw(), profile);

        std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2, d_e, d_r1_residual,
                                          d_r2_residual, p_evl_sorted, zshiftl, n_occ_alpha,
                                          n_occ_beta);

        update_r2(ec, d_r2());

        sch((d_r1s[off])() = d_r1())((d_r2s[off])() = d_r2()).execute();

        ccsd_canonical::ccsd_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);
        sch.execute(ec.exhw(), profile);
        energy = get_scalar(d_e);

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

        if(writet && ((iter + 1) % writet_iter == 0)) {
          write_to_disk(d_t1, t1file);
          write_to_disk(d_t2, t2file);
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
      std::vector<Tensor<T>>              next_t{d_t1, d_t2};
      diis<T>(ec, rs, ts, next_t);
    }

    if(writet) {
      write_to_disk(d_t1, t1file);
      write_to_disk(d_t2, t2file);
    }

    Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  } // no restart
  else {
    ccsd_canonical::ccsd_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);

    sch.execute(ec.exhw(), profile);

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

  return std::make_tuple(residual, energy);
}

void ccsd_canonical_driver(ExecutionContext& ec, ChemEnv& chem_env) {
  using T        = double;
  auto      rank = ec.pg().rank();
  Scheduler sch{ec};

  const bool do_hubbard = chem_env.sys_data.is_hubbard;
  if(do_hubbard) {
    chem_env.cd_context.do_cholesky    = false;
    chem_env.cd_context.keep_movecs_so = true;
  }

  cholesky_2e::cholesky_2e_driver(ec, chem_env);

  std::string files_prefix = chem_env.get_files_prefix();

  CDContext& cd_context = chem_env.cd_context;
  CCContext& cc_context = chem_env.cc_context;
  cc_context.init_filenames(files_prefix);
  CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

  auto        debug      = ccsd_options.debug;
  bool        scf_conv   = chem_env.scf_context.no_scf;
  std::string t1file     = cc_context.t1file;
  std::string t2file     = cc_context.t2file;
  const bool  ccsdstatus = cc_context.is_converged(chem_env.run_context, "ccsd");

  bool ccsd_restart = ccsd_options.readt ||
                      (fs::exists(t1file) && fs::exists(t2file) && fs::exists(cd_context.f1file));
  if(!do_hubbard) ccsd_restart = ccsd_restart && fs::exists(cd_context.v2file);

  TiledIndexSpace& MO      = chem_env.is_context.MSO;
  TiledIndexSpace& CI      = chem_env.is_context.CI;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr = chem_env.cd_context.cholV2;

  std::vector<T>&        p_evl_sorted = cd_context.p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) =
    setupTensors(ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && ccsdstatus && scf_conv);

  if(ccsd_restart) {
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    p_evl_sorted = tamm::diagonal(d_f1);
  }

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ccsd_restart = ccsd_restart && ccsdstatus && scf_conv;

  std::string fullV2file = cd_context.fullV2file;

  Tensor<T>& d_v2 = cd_context.d_v2;

  if(do_hubbard) { // hubbard
    auto             cc_h1 = std::chrono::high_resolution_clock::now();
    TiledIndexSpace& mo    = chem_env.is_context.MSO;
    auto             ao    = chem_env.is_context.AO_opt;
    const Tensor<T>  lcao  = chem_env.cd_context.movecs_so; // aoxmo
    auto [h1, h2, h3, h4]  = mo.labels<4>("occ");
    auto [p1, p2, p3, p4]  = mo.labels<4>("virt");
    // auto [p, q, r, s]     = mo.labels<4>("all");
    auto [p, r]           = mo.labels<2>("all_alpha");
    auto [q, s]           = mo.labels<2>("all_beta");
    auto [a1, a2, a3, a4] = ao.labels<4>("all");

    Tensor<T> ints2e_ao{ao, ao, ao, ao};
    sch.allocate(ints2e_ao)(ints2e_ao() = 0.0).execute();
    update_ints2e_hubbard(ints2e_ao, chem_env.ioptions.scf_options.hub_U_val);

    if(chem_env.ioptions.ccsd_options.debug) {
      const auto f1_norm   = tamm::norm(chem_env.cd_context.d_f1);
      const auto lcao_norm = tamm::norm(chem_env.cd_context.movecs_so);
      if(ec.print()) {
        std::cout << "Norm of LCAO matrix: " << lcao_norm << std::endl;
        std::cout << "Norm of MO Fock matrix: " << f1_norm << std::endl;
      }
    }

    auto create_v2_hubbard = [&](Tensor<T>& fullv2) {
      Tensor<T> temp1{mo, ao, ao, ao};
      Tensor<T> temp2{mo, mo, ao, ao};
      Tensor<T> temp3{mo, mo, mo, ao};
      Tensor<T> v2tmp{{mo, mo, mo, mo}, {2, 2}};

      // clang-format off
        sch.allocate(temp1,temp2)
            (temp1(p,a2,a3,a4) = lcao(a1,p)*ints2e_ao(a1,a2,a3,a4))
            .deallocate(ints2e_ao)
            (temp2(p,q,a3,a4) = lcao(a2,q) * temp1(p,a2,a3,a4))
            .deallocate(temp1).allocate(temp3)
            (temp3(p,q,r,a4) = lcao(a3,r) * temp2(p,q,a3,a4))
            .deallocate(temp2).allocate(fullv2)
            (fullv2(p,q,r,s) = lcao(a4,s) * temp3(p,q,r,a4))
            .deallocate(temp3)
            .execute(ec.exhw());

        //anti-sym
        sch.allocate(v2tmp)
           (v2tmp(p,q,r,s) = fullv2(p,q,r,s))
           (fullv2(q,p,s,r) = v2tmp(p,q,r,s))
           (fullv2(p,q,s,r) += -1.0*v2tmp(p,q,r,s))
           (fullv2(q,p,r,s) += -1.0*v2tmp(p,q,r,s))
          .deallocate(v2tmp)
          .execute();
      // clang-format on

      if(chem_env.ioptions.ccsd_options.debug) {
        auto v2norm = tamm::norm(fullv2);
        if(ec.print()) std::cout << "v2 mo norm: " << v2norm << std::endl;
      }
    };

    Tensor<T> fullv2{{mo, mo, mo, mo}, {2, 2}};
    create_v2_hubbard(fullv2);
    cd_context.d_v2 = fullv2;

    sch.deallocate(lcao).execute();

    auto cc_h2 = std::chrono::high_resolution_clock::now();

    double hub_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((cc_h2 - cc_h1)).count();
    if(rank == 0)
      std::cout << std::endl
                << "Hubbard Setup Time: " << std::fixed << std::setprecision(2) << hub_time
                << " secs" << std::endl;
  }

  if(!fs::exists(fullV2file)) {
    if(!do_hubbard)
      d_v2 = cholesky_2e::setupV2<T>(ec, MO, CI, cholVpr, cd_context.num_chol_vecs, ec.exhw());
    // if(ccsd_options.writet) {
    //   write_to_disk(d_v2, fullV2file);
    // }
  }
  /*else {
    d_v2 = Tensor<T>{{N, N, N, N}, {2, 2}};
    Tensor<T>::allocate(&ec, d_v2);
    read_from_disk(d_v2, fullV2file);
  }*/

  if(!do_hubbard) free_tensors(cholVpr);

  auto [residual, corr_energy] = ccsd_canonical::ccsd_v2_driver<T>(
    chem_env, ec, MO, d_t1, d_t2, d_f1, d_v2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s, p_evl_sorted,
    ccsd_restart, files_prefix);

  ccsd_stats(ec, chem_env.scf_context.hf_energy, residual, corr_energy, ccsd_options.threshold);

  if(ccsd_options.writet && !ccsdstatus) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    chem_env.run_context["ccsd"]["converged"] = true;
  }
  else if(!ccsdstatus) chem_env.run_context["ccsd"]["converged"] = false;
  if(rank == 0) chem_env.write_run_context();

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time taken for Canonical CCSD: " << std::fixed << std::setprecision(2)
              << ccsd_time << " secs" << std::endl;

  cc_print(chem_env, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }

  chem_env.cc_context.d_t1_full = d_t1;
  chem_env.cc_context.d_t2_full = d_t2;

  bool                       computeTData = cc_context.compute.fvt12_full;
  cholesky_2e::V2Tensors<T>& v2tensors    = cd_context.v2tensors;

  if(computeTData) {
    bool compute_fullv2 = cc_context.compute.v2_full;
    if(compute_fullv2 && (ccsd_options.writet || ccsd_options.readt)) {
      if(v2tensors.exist_on_disk(files_prefix)) {
        v2tensors.allocate(ec, MO);
        v2tensors.read_from_disk(files_prefix);
        compute_fullv2 = false;
      }
    }
    if(compute_fullv2) {
      v2tensors = cholesky_2e::setupV2Tensors_from_fullV2<T>(ec, d_v2, ec.exhw());
      if(ccsd_options.writet) { v2tensors.write_to_disk(files_prefix); }
    }
  }

  free_tensors(d_v2); // deallocate the single fullV2 tensor
  if(!cc_context.keep.fvt12_full) free_tensors(d_t1, d_t2, d_f1);

  ec.flush_and_sync();
  // delete ec;
}
}; // namespace exachem::cc::ccsd_canonical
