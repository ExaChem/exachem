/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/eom/eomccsd_opt.hpp"

#include <filesystem>
namespace fs = std::filesystem;
using namespace exachem::scf;
template<typename T>
void eomccsd_x1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
                const Tensor<T>& t2, const Tensor<T>& x1, const Tensor<T>& x2, const Tensor<T>& f1,
                exachem::cholesky_2e::V2Tensors<T>& v2tensors, EOM_X1Tensors<T>& x1tensors) {
  auto [h1, h3, h4, h5, h6, h8] = MO.labels<6>("occ");
  auto [p2, p3, p4, p5, p6, p7] = MO.labels<6>("virt");

  Tensor<T> i_1 = x1tensors.i_1;
  //  Tensor<T> i_1_1 = x1tensors.i_1_1;
  Tensor<T> i_2   = x1tensors.i_2;
  Tensor<T> i_3   = x1tensors.i_3;
  Tensor<T> i_4   = x1tensors.i_4;
  Tensor<T> i_5_1 = x1tensors.i_5_1;
  Tensor<T> i_5   = x1tensors.i_5;
  Tensor<T> i_5_2 = x1tensors.i_5_2;
  Tensor<T> i_6   = x1tensors.i_6;
  Tensor<T> i_7   = x1tensors.i_7;
  Tensor<T> i_8   = x1tensors.i_8;

  // clang-format off
  sch
    //  ( i0(p2,h1)           =  0                                        )
    //  (   i_1(h6,h1)        =        f1(h6,h1)                          )
    //  (     i_1_1(h6,p7)    =        f1(h6,p7)                          )
    //  (     i_1_1(h6,p7)   +=        t1(p4,h5)       * v2tensors.v2ijab(h5,h6,p4,p7)  )
    //  (   i_1(h6,h1)       +=        t1(p7,h1)       * i_1_1(h6,p7)     )
    //  (   i_1(h6,h1)       += -1   * t1(p3,h4)       * v2tensors.v2ijka(h4,h6,h1,p3)  )
    //  (   i_1(h6,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2tensors.v2ijab(h5,h6,p3,p4)  )
     ( i0(p2,h1)           = -1   * x1(p2,h6)       * i_1(h6,h1)       )
    //  (   i_2(p2,p6)        =        f1(p2,p6)                          )
    //  (   i_2(p2,p6)       +=        t1(p3,h4)       * v2tensors.v2iabc(h4,p2,p3,p6)  )
     ( i0(p2,h1)          +=        x1(p6,h1)       * i_2(p2,p6)       )
     ( i0(p2,h1)          += -1   * x1(p4,h3)       * v2tensors.v2iajb(h3,p2,h1,p4)  )
    //  (   i_3(h6,p7)        =        f1(h6,p7)                          )
    //  (   i_3(h6,p7)       +=        t1(p3,h4)       * v2tensors.v2ijab(h4,h6,p3,p7)  )
     ( i0(p2,h1)          +=        x2(p2,p7,h1,h6) * i_3(h6,p7)       )
    //  (   i_4(h6,h8,h1,p7)  =        v2tensors.v2ijka(h6,h8,h1,p7)                    )
    //  (   i_4(h6,h8,h1,p7) +=        t1(p3,h1)       * v2tensors.v2ijab(h6,h8,p3,p7)  )
     ( i0(p2,h1)          += -0.5 * x2(p2,p7,h6,h8) * i_4(h6,h8,h1,p7) )
     ( i0(p2,h1)          += -0.5 * x2(p4,p5,h1,h3) * v2tensors.v2iabc(h3,p2,p4,p5)  )
    //  (     i_5_1(h8,p3)    =        f1(h8,p3)                          )
    //  (     i_5_1(h8,p3)   += -1   * t1(p4,h5)       * v2tensors.v2ijab(h5,h8,p3,p4)  )
     (   i_5(h8,h1)        =        x1(p3,h1)       * i_5_1(h8,p3)     )
     (   i_5(h8,h1)       += -1   * x1(p5,h4)       * v2tensors.v2ijka(h4,h8,h1,p5)  )
     (   i_5(h8,h1)       += -0.5 * x2(p5,p6,h1,h4) * v2tensors.v2ijab(h4,h8,p5,p6)  )
     (     i_5_2(h8,p3)    = -1   * x1(p6,h5)       * v2tensors.v2ijab(h5,h8,p3,p6)  )
     (   i_5(h8,h1)       +=        t1(p3,h1)       * i_5_2(h8,p3)     )
     ( i0(p2,h1)          += -1   * t1(p2,h8)       * i_5(h8,h1)       )
     (   i_6(p2,p3)        =        x1(p5,h4)       * v2tensors.v2iabc(h4,p2,p3,p5)  )
     ( i0(p2,h1)          += -1   * t1(p3,h1)       * i_6(p2,p3)       )
     (   i_7(h4,p3)        =        x1(p6,h5)       * v2tensors.v2ijab(h4,h5,p3,p6)  )
     ( i0(p2,h1)          +=        t2(p2,p3,h1,h4) * i_7(h4,p3)       )
     (   i_8(h4,h5,h1,p3)  =        x1(p6,h1)       * v2tensors.v2ijab(h4,h5,p3,p6)  )
     ( i0(p2,h1)          +=  0.5 * t2(p2,p3,h4,h5) * i_8(h4,h5,h1,p3) )
    ;
  // clang-format on
}

template<typename T>
void eomccsd_x2(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
                const Tensor<T>& t2, const Tensor<T>& x1, const Tensor<T>& x2, const Tensor<T>& f1,
                exachem::cholesky_2e::V2Tensors<T>& v2tensors, EOM_X2Tensors<T>& x2tensors) {
  TiledIndexLabel h1, h2, h5, h6, h7, h8, h9, h10;
  TiledIndexLabel p3, p4, p5, p6, p7, p8, p9;

  std::tie(h1, h2, h5, h6, h7, h8, h9, h10) = MO.labels<8>("occ");
  std::tie(p3, p4, p5, p6, p7, p8, p9)      = MO.labels<7>("virt");

  Tensor<T> i_1        = x2tensors.i_1;
  Tensor<T> i_2        = x2tensors.i_2;
  Tensor<T> i_3        = x2tensors.i_3;
  Tensor<T> i_4        = x2tensors.i_4;
  Tensor<T> i_5        = x2tensors.i_5;
  Tensor<T> i_6_1      = x2tensors.i_6_1;
  Tensor<T> i_6        = x2tensors.i_6;
  Tensor<T> i_6_2      = x2tensors.i_6_2;
  Tensor<T> i_6_3      = x2tensors.i_6_3;
  Tensor<T> i_6_4      = x2tensors.i_6_4;
  Tensor<T> i_6_4_1    = x2tensors.i_6_4_1;
  Tensor<T> i_6_5      = x2tensors.i_6_5;
  Tensor<T> i_6_6      = x2tensors.i_6_6;
  Tensor<T> i_6_7      = x2tensors.i_6_7;
  Tensor<T> i_7        = x2tensors.i_7;
  Tensor<T> i_8_1      = x2tensors.i_8_1;
  Tensor<T> i_8        = x2tensors.i_8;
  Tensor<T> i_8_2      = x2tensors.i_8_2;
  Tensor<T> i_9        = x2tensors.i_9;
  Tensor<T> i_9_1      = x2tensors.i_9_1;
  Tensor<T> i_10       = x2tensors.i_10;
  Tensor<T> i_11       = x2tensors.i_11;
  Tensor<T> i0_temp    = x2tensors.i0_temp;
  Tensor<T> i_6_temp   = x2tensors.i_6_temp;
  Tensor<T> i_6_4_temp = x2tensors.i_6_4_temp;
  Tensor<T> i_9_temp   = x2tensors.i_9_temp;
  //  Tensor<T> i_1_1      = x2tensors.i_1_1     ;
  //  Tensor<T> i_1_2      = x2tensors.i_1_2     ;
  //  Tensor<T> i_1_3      = x2tensors.i_1_3     ;
  //  Tensor<T> i_2_1      = x2tensors.i_2_1     ;
  //  Tensor<T> i_4_1      = x2tensors.i_4_1     ;
  //  Tensor<T> i_6_1_1    = x2tensors.i_6_1_1   ;
  //  Tensor<T> i_1_temp   = x2tensors.i_1_temp  ;
  //  Tensor<T> i_4_temp   = x2tensors.i_4_temp  ;
  //  Tensor<T> i_6_1_temp = x2tensors.i_6_1_temp;

  // clang-format off
  sch
  //   ( i0(p4,p3,h1,h2)               =  0                                               )
    //  (   i_1(h9,p3,h1,h2)            =         v2tensors.v2ijka(h1,h2,h9,p3)                         )
    //  (     i_1_1(h9,p3,h1,p5)        =         v2tensors.v2iajb(h9,p3,h1,p5)                          )
    //  (     i_1_1(h9,p3,h1,p5)       += -0.5  * t1(p6,h1)        * v2tensors.v2iabc(h9,p3,p5,p6)      )
    //  (   i_1_temp(h9,p3,h1,h2)       =         t1(p5,h1)        * i_1_1(h9,p3,h2,p5)    )
    //  (   i_1(h9,p3,h1,h2)           += -1    * i_1_temp(h9,p3,h1,h2)                    )
    //  (   i_1(h9,p3,h2,h1)           +=         i_1_temp(h9,p3,h1,h2)                    ) //P(h1/h2)
    //  (     i_1_2(h9,p8)              =         f1(h9,p8)                                )
    //  (     i_1_2(h9,p8)             +=         t1(p6,h7)        * v2tensors.v2ijab(h7,h9,p6,p8)       )
    //  (   i_1(h9,p3,h1,h2)           += -1    * t2(p3,p8,h1,h2)  * i_1_2(h9,p8)          )
    //  (     i_1_3(h6,h9,h1,p5)        =         v2tensors.v2ijka(h6,h9,h1,p5)                          )
    //  (     i_1_3(h6,h9,h1,p5)       += -1    * t1(p7,h1)        * v2tensors.v2ijab(h6,h9,p5,p7)       )
    //  (   i_1_temp(h9,p3,h1,h2)       =         t2(p3,p5,h1,h6)  * i_1_3(h6,h9,h2,p5)    )
    //  (   i_1(h9,p3,h1,h2)           +=         i_1_temp(h9,p3,h1,h2)                    )
    //  (   i_1(h9,p3,h2,h1)           += -1    * i_1_temp(h9,p3,h1,h2)                    ) //P(h1/h2)
    //  (   i_1(h9,p3,h1,h2)           +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2iabc(h9,p3,p5,p6)       )
     ( i0_temp(p3,p4,h1,h2)          =         x1(p3,h9)        * i_1(h9,p4,h1,h2)      )
     ( i0(p3,p4,h1,h2)               = -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p4,p3,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
     ( i0_temp(p3,p4,h1,h2)          =         x1(p5,h1)        * v2tensors.v2iabc(h2,p5,p3,p4)       )
     ( i0(p3,p4,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
    //  (   i_2(h8,h1)                  =         f1(h8,h1)                                )
    //  (     i_2_1(h8,p9)              =         f1(h8,p9)                                )
    //  (     i_2_1(h8,p9)             +=         t1(p6,h7)        * v2tensors.v2ijab(h7,h8,p6,p9)        )
    //  (   i_2(h8,h1)                 +=         t1(p9,h1)        * i_2_1(h8,p9)          )
    //  (   i_2(h8,h1)                 += -1    * t1(p5,h6)        * v2tensors.v2ijka(h6,h8,h1,p5)       )
    //  (   i_2(h8,h1)                 += -0.5  * t2(p5,p6,h1,h7)  * v2tensors.v2ijab(h7,h8,p5,p6)       )
     ( i0_temp(p3,p4,h1,h2)          =         x2(p3,p4,h1,h8)  * i_2(h8,h2)            )
     ( i0(p3,p4,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
    //  (   i_3(p3,p8)                  =         f1(p3,p8)                                )
    //  (   i_3(p3,p8)                 +=         t1(p5,h6)        * v2tensors.v2iabc(h6,p3,p5,p8)       )
    //  (   i_3(p3,p8)                 +=  0.5  * t2(p3,p5,h6,h7)  * v2tensors.v2ijab(h6,h7,p5,p8)       )
     ( i0_temp(p3,p4,h1,h2)          =         x2(p3,p8,h1,h2)  * i_3(p4,p8)            )
     ( i0(p3,p4,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     )
     ( i0(p4,p3,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
    //  (   i_4(h9,h10,h1,h2)           =         v2tensors.v2ijkl(h9,h10,h1,h2)                         )
    //  (     i_4_1(h9,h10,h1,p5)       =         v2tensors.v2ijka(h9,h10,h1,p5)                         )
    //  (     i_4_1(h9,h10,h1,p5)      += -0.5  * t1(p6,h1)        * v2tensors.v2ijab(h9,h10,p5,p6)      )
    //  (   i_4_temp(h9,h10,h1,h2)      =         t1(p5,h1)        * i_4_1(h9,h10,h2,p5)   )
    //  (   i_4(h9,h10,h1,h2)          += -1    * i_4_temp(h9,h10,h1,h2)                   )
    //  (   i_4(h9,h10,h2,h1)          +=         i_4_temp(h9,h10,h1,h2)                   ) //P(h1/h2)
    //  (   i_4(h9,h10,h1,h2)          +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2ijab(h9,h10,p5,p6)      )
     ( i0(p3,p4,h1,h2)              +=  0.5  * x2(p3,p4,h9,h10) * i_4(h9,h10,h1,h2)     )
    //  (   i_5(h7,p3,h1,p8)            =         v2tensors.v2iajb(h7,p3,h1,p8)                          )
    //  (   i_5(h7,p3,h1,p8)           +=         t1(p5,h1)        * v2tensors.v2iabc(h7,p3,p5,p8)       )
     ( i0_temp(p3,p4,h1,h2)          =         x2(p3,p8,h1,h7)  * i_5(h7,p4,h2,p8)      )
     ( i0(p3,p4,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
     ( i0(p4,p3,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
     ( i0(p4,p3,h2,h1)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2,p3/p4)
     ( i0(p3,p4,h1,h2)              +=  0.5  * x2(p5,p6,h1,h2)  * v2tensors.v2abcd(p3,p4,p5,p6)       )
    //  (     i_6_1(h8,h10,h1,h2)       =         v2tensors.v2ijkl(h8,h10,h1,h2)                         )
    //  (       i_6_1_1(h8,h10,h1,p5)   =         v2tensors.v2ijka(h8,h10,h1,p5)                         )
    //  (       i_6_1_1(h8,h10,h1,p5)  += -0.5  * t1(p6,h1)        * v2tensors.v2ijab(h8,h10,p5,p6)      )
    //  (     i_6_1_temp(h8,h10,h1,h2)  =         t1(p5,h1)        * i_6_1_1(h8,h10,h2,p5) )
    //  (     i_6_1(h8,h10,h1,h2)      += -1    * i_6_1_temp(h8,h10,h1,h2)                 )
    //  (     i_6_1(h8,h10,h2,h1)      +=         i_6_1_temp(h8,h10,h1,h2)                 ) //P(h1/h2)
    //  (     i_6_1(h8,h10,h1,h2)      +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2ijab(h8,h10,p5,p6)      )
     (   i_6(h10,p3,h1,h2)           = -1    * x1(p3,h8)        * i_6_1(h8,h10,h1,h2)   )
     (   i_6_temp(h10,p3,h1,h2)      =         x1(p6,h1)        * v2tensors.v2iajb(h10,p3,h2,p6)      )
     (   i_6(h10,p3,h1,h2)          +=         i_6_temp(h10,p3,h1,h2)                   )
     (   i_6(h10,p3,h2,h1)          += -1    * i_6_temp(h10,p3,h1,h2)                   ) //P(h1/h2)
    //  (     i_6_2(h10,p5)             =         f1(h10,p5)                               )
    //  (     i_6_2(h10,p5)            += -1    * t1(p6,h7)        * v2tensors.v2ijab(h7,h10,p5,p6)      )
     (   i_6(h10,p3,h1,h2)          +=         x2(p3,p5,h1,h2)  * i_6_2(h10,p5)         )
    //  (     i_6_3(h8,h10,h1,p9)       =         v2tensors.v2ijka(h8,h10,h1,p9)                         )
    //  (     i_6_3(h8,h10,h1,p9)      +=         t1(p5,h1)        * v2tensors.v2ijab(h8,h10,p5,p9)      )
     (   i_6_temp(h10,p3,h1,h2)      =         x2(p3,p9,h1,h8)  * i_6_3(h8,h10,h2,p9)   )
     (   i_6(h10,p3,h1,h2)          += -1    * i_6_temp(h10,p3,h1,h2)                   )
     (   i_6(h10,p3,h2,h1)          +=         i_6_temp(h10,p3,h1,h2)                   ) //P(h1/h2)
     (   i_6(h10,p3,h1,h2)          += -0.5  * x2(p6,p7,h1,h2)  * v2tensors.v2iabc(h10,p3,p6,p7)      )
     (     i_6_4_temp(h9,h10,h1,h2)  =  0.5  * x1(p7,h1)        * v2tensors.v2ijka(h9,h10,h2,p7)      )
     (     i_6_4(h9,h10,h1,h2)       =         i_6_4_temp(h9,h10,h1,h2)                 )
     (     i_6_4(h9,h10,h2,h1)      += -1    * i_6_4_temp(h9,h10,h1,h2)                 ) //P(h1/h2)
     (     i_6_4(h9,h10,h1,h2)      += -0.25 * x2(p7,p8,h1,h2)  * v2tensors.v2ijab(h9,h10,p7,p8)      )
     (       i_6_4_1(h9,h10,h1,p5)   = -1    * x1(p8,h1)        * v2tensors.v2ijab(h9,h10,p5,p8)      )
     (     i_6_4_temp(h9,h10,h1,h2)  =  0.5  * t1(p5,h1)        * i_6_4_1(h9,h10,h2,p5) )
     (     i_6_4(h9,h10,h1,h2)      +=         i_6_4_temp(h9,h10,h1,h2)                 )
     (     i_6_4(h9,h10,h2,h1)      += -1    * i_6_4_temp(h9,h10,h1,h2)                 ) //P(h1/h2)
     (   i_6(h10,p3,h1,h2)          +=         t1(p3,h9)        * i_6_4(h9,h10,h1,h2)   )
     (     i_6_5(h10,p3,h1,p5)       =         x1(p7,h1)        * v2tensors.v2iabc(h10,p3,p5,p7)      )
     (   i_6_temp(h10,p3,h1,h2)      =         t1(p5,h1)        * i_6_5(h10,p3,h2,p5)   )
     (   i_6(h10,p3,h1,h2)          += -1    * i_6_temp(h10,p3,h1,h2)                   )
     (   i_6(h10,p3,h2,h1)          +=         i_6_temp(h10,p3,h1,h2)                   ) //P(h1/h2)
     (     i_6_6(h10,p5)             = -1    * x1(p8,h7)        * v2tensors.v2ijab(h7,h10,p5,p8)      )
     (   i_6(h10,p3,h1,h2)          +=         t2(p3,p5,h1,h2)  * i_6_6(h10,p5)         )
     (     i_6_7(h6,h10,h1,p5)       =         x1(p8,h1)        * v2tensors.v2ijab(h6,h10,p5,p8)      )
     (   i_6_temp(h10,p3,h1,h2)      =         t2(p3,p5,h1,h6)  * i_6_7(h6,h10,h2,p5)   )
     (   i_6(h10,p3,h1,h2)          +=         i_6_temp(h10,p3,h1,h2)                   )
     (   i_6(h10,p3,h2,h1)          += -1    * i_6_temp(h10,p3,h1,h2)                   ) //P(h1/h2)
     ( i0_temp(p3,p4,h1,h2)          =         t1(p3,h10)       * i_6(h10,p4,h1,h2)     )
     ( i0(p3,p4,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     )
     ( i0(p4,p3,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
     (   i_7(p3,p4,h1,p5)            =         x1(p6,h1)        * v2tensors.v2abcd(p3,p4,p5,p6)       )
     ( i0_temp(p3,p4,h1,h2)          =         t1(p5,h1)        * i_7(p3,p4,h2,p5)      )
     ( i0(p3,p4,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
    //  (     i_8_1(h5,p9)              =         f1(h5,p9)                                )
    //  (     i_8_1(h5,p9)             += -1    * t1(p6,h7)        * v2tensors.v2ijab(h5,h7,p6,p9)       )
     (   i_8(h5,h1)                  =         x1(p9,h1)        * i_8_1(h5,p9)          )
     (   i_8(h5,h1)                 +=         x1(p7,h6)        * v2tensors.v2ijka(h5,h6,h1,p7)       )
     (   i_8(h5,h1)                 +=  0.5  * x2(p7,p8,h1,h6)  * v2tensors.v2ijab(h5,h6,p7,p8)       )
     (     i_8_2(h5,p6)              =         x1(p8,h7)        * v2tensors.v2ijab(h5,h7,p6,p8)       )
     (   i_8(h5,h1)                 +=         t1(p6,h1)        * i_8_2(h5,p6)          )
     ( i0_temp(p3,p4,h1,h2)          =         t2(p3,p4,h1,h5)  * i_8(h5,h2)            )
     ( i0(p3,p4,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
     (   i_9_temp(h5,h6,h1,h2)       =  0.5  * x1(p7,h1)        * v2tensors.v2ijka(h5,h6,h2,p7)       )
     (   i_9(h5,h6,h1,h2)            = -1    * i_9_temp(h5,h6,h1,h2)                    )
     (   i_9(h5,h6,h2,h1)           +=         i_9_temp(h5,h6,h1,h2)                    ) //P(h1/h2)
     (   i_9(h5,h6,h1,h2)           +=  0.25 * x2(p7,p8,h1,h2)  * v2tensors.v2ijab(h5,h6,p7,p8)       )
     (     i_9_1(h5,h6,h1,p7)        =         x1(p8,h1)        * v2tensors.v2ijab(h5,h6,p7,p8)       )
     (   i_9_temp(h5,h6,h1,h2)       =  0.5  * t1(p7,h1)        * i_9_1(h5,h6,h2,p7)    )
     (   i_9(h5,h6,h1,h2)           +=         i_9_temp(h5,h6,h1,h2)                    )
     (   i_9(h5,h6,h2,h1)           += -1    * i_9_temp(h5,h6,h1,h2)                    ) //P(h1/h2)
     ( i0(p3,p4,h1,h2)              +=         t2(p3,p4,h5,h6)  * i_9(h5,h6,h1,h2)      )
     (   i_10(p3,p5)                 =         x1(p7,h6)        * v2tensors.v2iabc(h6,p3,p5,p7)       )
     (   i_10(p3,p5)                +=  0.5  * x2(p3,p8,h6,h7)  * v2tensors.v2ijab(h6,h7,p5,p8)       )
     ( i0_temp(p3,p4,h1,h2)          =         t2(p3,p5,h1,h2)  * i_10(p4,p5)           )
     ( i0(p3,p4,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     )
     ( i0(p4,p3,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
     (   i_11(h6,p3,h1,p5)           =         x1(p7,h1)        * v2tensors.v2iabc(h6,p3,p5,p7)       )
     (   i_11(h6,p3,h1,p5)          +=         x2(p3,p8,h1,h7)  * v2tensors.v2ijab(h6,h7,p5,p8)       )
     ( i0_temp(p3,p4,h1,h2)          =         t2(p3,p5,h1,h6)  * i_11(h6,p4,h2,p5)     )
     ( i0(p3,p4,h1,h2)              +=         i0_temp(p3,p4,h1,h2)                     )
     ( i0(p3,p4,h2,h1)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2)
     ( i0(p4,p3,h1,h2)              += -1    * i0_temp(p3,p4,h1,h2)                     ) //P(p3/p4)
     ( i0(p4,p3,h2,h1)              +=         i0_temp(p3,p4,h1,h2)                     ) //P(h1/h2,p3/p4)
    ;
  // clang-format on
}

template<typename T>
void right_eomccsd_driver(ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO,
                          Tensor<T>& t1, Tensor<T>& t2, Tensor<T>& f1,
                          exachem::cholesky_2e::V2Tensors<T>& v2tensors,
                          std::vector<T>                      p_evl_sorted) {
  SystemData&     sys_data    = chem_env.sys_data;
  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);

  // EOMCCSD Variables
  CCSDOptions ccsd_options = chem_env.ioptions.ccsd_options;
  int         nroots       = ccsd_options.eom_nroots;
  int         maxeomiter   = ccsd_options.maxiter;
  //    int eomsolver        = 1; //INDICATES WHICH SOLVER TO USE. (LATER IMPLEMENTATION)
  double eomthresh = ccsd_options.eom_threshold;
  //    double x2guessthresh = 0.6; //THRESHOLD FOR X2 INITIAL GUESS (LATER IMPLEMENTATION)
  int        microeomiter = ccsd_options.eom_microiter; // Number of iterations in a microcycle
  const bool profile      = ccsd_options.profile_ccsd;

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  std::cout.precision(15);

  Scheduler sch{ec};
  /// @todo: make it a tamm tensor
  auto populate_vector_of_tensors = [&](std::vector<Tensor<T>>& vec, bool is2D = true) {
    for(size_t x = 0; x < vec.size(); x++) {
      if(is2D) vec[x] = Tensor<T>{{V, O}, {1, 1}};
      else vec[x] = Tensor<T>{{V, V, O, O}, {2, 2}};
      Tensor<T>::allocate(&ec, vec[x]);
    }
  };

  // FOR JACOBI STEP
  //   double zshiftl = 0;
  //   bool transpose=false;

  // INITIAL GUESS WILL MOVE TO HERE
  // TO DO: NXTRIALS IS SET TO NROOTS BECAUSE WE ONLY ALLOW #NROOTS# INITIAL GUESSES
  //       WHEN THE EOM_GUESS ROUTINE IS UPDATED TO BE LIKE THE INITIAL GUESS IN TCE
  //       THEN THE FOLLOWING LINE WILL BE "INT NXTRIALS = INITVECS", WHERE INITVEC
  //       IS THE NUMBER OF INITIAL VECTORS. THE {X1,X2} AND {XP1,XP2} TENSORS WILL
  //       BE OF DIMENSION (ninitvecs + NROOTS*(MICROEOMITER-1)) FOR THE FIRST
  //       MICROCYCLE AND (NROOTS*MICROEOMITER) FOR THE REMAINING.

  int        ninitvecs = nroots;
  const auto hbardim   = ninitvecs + nroots * (microeomiter - 1);

  TiledIndexSpace hbar_tis = {IndexSpace{range(0, hbardim)}};
  Matrix          hbar     = Matrix::Zero(hbardim, hbardim);
  Matrix          hbar_right;

  Tensor<T> u1{{V, O}, {1, 1}};
  Tensor<T> u2{{V, V, O, O}, {2, 2}};
  Tensor<T> uu2{{V, V, O, O}, {2, 2}};
  Tensor<T> uuu2{{V, V, O, O}, {2, 2}};
  Tensor<T>::allocate(&ec, u1, u2, uu2, uuu2);

  using std::vector;

  vector<Tensor<T>> x1(hbardim);
  populate_vector_of_tensors(x1);
  vector<Tensor<T>> x2(hbardim);
  populate_vector_of_tensors(x2, false);
  vector<Tensor<T>> xp1(hbardim);
  populate_vector_of_tensors(xp1);
  vector<Tensor<T>> xp2(hbardim);
  populate_vector_of_tensors(xp2, false);
  vector<Tensor<T>> xc1(nroots);
  populate_vector_of_tensors(xc1);
  vector<Tensor<T>> xc2(nroots);
  populate_vector_of_tensors(xc2, false);
  vector<Tensor<T>> r1(nroots);
  populate_vector_of_tensors(r1);
  vector<Tensor<T>> r2(nroots);
  populate_vector_of_tensors(r2, false);

  Tensor<T> d_r1{};
  Tensor<T> oscalar{};
  Tensor<T>::allocate(&ec, d_r1, oscalar);
  bool       convflag = false;
  double     au2ev    = 27.2113961;
  const bool mrank    = (ec.pg().rank() == 0);

  //################################################################################
  //  CALL THE EOM_GUESS ROUTINE (EXTERNAL ROUTINE)
  //################################################################################
  auto cc_t1 = std::chrono::high_resolution_clock::now();

  eom_guess_opt(ec, MO, hbar_tis, nroots, n_occ_alpha, n_occ_beta, p_evl_sorted, x1);

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(mrank)
    std::cout << std::endl
              << "Time taken for EOM GUESS: " << std::fixed << std::setprecision(2) << time
              << " secs" << std::endl;

  EOM_X1Tensors<T> x1tensors;
  EOM_X2Tensors<T> x2tensors;
  x1tensors.allocate(ec, MO);
  x2tensors.allocate(ec, MO);

  ExecutionHW exhw = ec.exhw();

  //#################################################
  //  Compute intermediates independent of X1/X2
  //#################################################

  cc_t1 = std::chrono::high_resolution_clock::now();

  auto [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10] = MO.labels<10>("occ");
  auto [p1, p2, p3, p4, p5, p6, p7, p8, p9]      = MO.labels<9>("virt");

  { // X1
    Tensor<T> i_1   = x1tensors.i_1;
    Tensor<T> i_1_1 = x1tensors.i_1_1;
    Tensor<T> i_2   = x1tensors.i_2;
    Tensor<T> i_3   = x1tensors.i_3;
    Tensor<T> i_4   = x1tensors.i_4;
    Tensor<T> i_5_1 = x1tensors.i_5_1;

    // clang-format off
    sch.allocate(i_1_1)
      (   i_1(h6,h1)        =        f1(h6,h1)                          )
      (     i_1_1(h6,p7)    =        f1(h6,p7)                          )
      (     i_1_1(h6,p7)   +=        t1(p4,h5)       * v2tensors.v2ijab(h5,h6,p4,p7)  )
      (   i_1(h6,h1)       +=        t1(p7,h1)       * i_1_1(h6,p7)     )
      (   i_1(h6,h1)       += -1   * t1(p3,h4)       * v2tensors.v2ijka(h4,h6,h1,p3)  )
      (   i_1(h6,h1)       += -0.5 * t2(p3,p4,h1,h5) * v2tensors.v2ijab(h5,h6,p3,p4)  )
      (   i_2(p2,p6)        =        f1(p2,p6)                          )
      (   i_2(p2,p6)       +=        t1(p3,h4)       * v2tensors.v2iabc(h4,p2,p3,p6)  )
      (   i_3(h6,p7)        =        f1(h6,p7)                          )
      (   i_3(h6,p7)       +=        t1(p3,h4)       * v2tensors.v2ijab(h4,h6,p3,p7)  )
      (   i_4(h6,h8,h1,p7)  =        v2tensors.v2ijka(h6,h8,h1,p7)                    )
      (   i_4(h6,h8,h1,p7) +=        t1(p3,h1)       * v2tensors.v2ijab(h6,h8,p3,p7)  )
      (     i_5_1(h8,p3)    =        f1(h8,p3)                          )
      (     i_5_1(h8,p3)   += -1   * t1(p4,h5)       * v2tensors.v2ijab(h5,h8,p3,p4)  )
      .deallocate(i_1_1)
      .execute(exhw);
    // clang-format on
  }

  { // X2
    Tensor<T> i_1        = x2tensors.i_1;
    Tensor<T> i_1_1      = x2tensors.i_1_1;
    Tensor<T> i_1_2      = x2tensors.i_1_2;
    Tensor<T> i_1_3      = x2tensors.i_1_3;
    Tensor<T> i_2        = x2tensors.i_2;
    Tensor<T> i_2_1      = x2tensors.i_2_1;
    Tensor<T> i_3        = x2tensors.i_3;
    Tensor<T> i_4        = x2tensors.i_4;
    Tensor<T> i_4_1      = x2tensors.i_4_1;
    Tensor<T> i_5        = x2tensors.i_5;
    Tensor<T> i_6_1      = x2tensors.i_6_1;
    Tensor<T> i_6_1_1    = x2tensors.i_6_1_1;
    Tensor<T> i_6_2      = x2tensors.i_6_2;
    Tensor<T> i_6_3      = x2tensors.i_6_3;
    Tensor<T> i_8_1      = x2tensors.i_8_1;
    Tensor<T> i_1_temp   = x2tensors.i_1_temp;
    Tensor<T> i_4_temp   = x2tensors.i_4_temp;
    Tensor<T> i_6_1_temp = x2tensors.i_6_1_temp;

    // clang-format off
    sch.allocate(i_1_1,i_1_2,i_1_3,i_2_1,i_4_1,i_6_1_1,i_1_temp,i_4_temp,i_6_1_temp)
      (   i_1(h9,p3,h1,h2)            =         v2tensors.v2ijka(h1,h2,h9,p3)                         )
      (     i_1_1(h9,p3,h1,p5)        =         v2tensors.v2iajb(h9,p3,h1,p5)                          )
      (     i_1_1(h9,p3,h1,p5)       += -0.5  * t1(p6,h1)        * v2tensors.v2iabc(h9,p3,p5,p6)      )
      (   i_1_temp(h9,p3,h1,h2)       =         t1(p5,h1)        * i_1_1(h9,p3,h2,p5)    )     
      (   i_1(h9,p3,h1,h2)           += -1    * i_1_temp(h9,p3,h1,h2)                    )
      (   i_1(h9,p3,h2,h1)           +=         i_1_temp(h9,p3,h1,h2)                    ) //P(h1/h2)
      (     i_1_2(h9,p8)              =         f1(h9,p8)                                )
      (     i_1_2(h9,p8)             +=         t1(p6,h7)        * v2tensors.v2ijab(h7,h9,p6,p8)       )
      (   i_1(h9,p3,h1,h2)           += -1    * t2(p3,p8,h1,h2)  * i_1_2(h9,p8)          )
      (     i_1_3(h6,h9,h1,p5)        =         v2tensors.v2ijka(h6,h9,h1,p5)                          )
      (     i_1_3(h6,h9,h1,p5)       += -1    * t1(p7,h1)        * v2tensors.v2ijab(h6,h9,p5,p7)       )
      (   i_1_temp(h9,p3,h1,h2)       =         t2(p3,p5,h1,h6)  * i_1_3(h6,h9,h2,p5)    )
      (   i_1(h9,p3,h1,h2)           +=         i_1_temp(h9,p3,h1,h2)                    )
      (   i_1(h9,p3,h2,h1)           += -1    * i_1_temp(h9,p3,h1,h2)                    ) //P(h1/h2)
      (   i_1(h9,p3,h1,h2)           +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2iabc(h9,p3,p5,p6)       )
      (   i_2(h8,h1)                  =         f1(h8,h1)                                )
      (     i_2_1(h8,p9)              =         f1(h8,p9)                                )
      (     i_2_1(h8,p9)             +=         t1(p6,h7)        * v2tensors.v2ijab(h7,h8,p6,p9)        )
      (   i_2(h8,h1)                 +=         t1(p9,h1)        * i_2_1(h8,p9)          )
      (   i_2(h8,h1)                 += -1    * t1(p5,h6)        * v2tensors.v2ijka(h6,h8,h1,p5)       )
      (   i_2(h8,h1)                 += -0.5  * t2(p5,p6,h1,h7)  * v2tensors.v2ijab(h7,h8,p5,p6)       )
      (   i_3(p3,p8)                  =         f1(p3,p8)                                )
      (   i_3(p3,p8)                 +=         t1(p5,h6)        * v2tensors.v2iabc(h6,p3,p5,p8)       )
      (   i_3(p3,p8)                 +=  0.5  * t2(p3,p5,h6,h7)  * v2tensors.v2ijab(h6,h7,p5,p8)       )
      (   i_4(h9,h10,h1,h2)           =         v2tensors.v2ijkl(h9,h10,h1,h2)                         )
      (     i_4_1(h9,h10,h1,p5)       =         v2tensors.v2ijka(h9,h10,h1,p5)                         )
      (     i_4_1(h9,h10,h1,p5)      += -0.5  * t1(p6,h1)        * v2tensors.v2ijab(h9,h10,p5,p6)      )
      (   i_4_temp(h9,h10,h1,h2)      =         t1(p5,h1)        * i_4_1(h9,h10,h2,p5)   )
      (   i_4(h9,h10,h1,h2)          += -1    * i_4_temp(h9,h10,h1,h2)                   )
      (   i_4(h9,h10,h2,h1)          +=         i_4_temp(h9,h10,h1,h2)                   ) //P(h1/h2)
      (   i_4(h9,h10,h1,h2)          +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2ijab(h9,h10,p5,p6)      )
      (   i_5(h7,p3,h1,p8)            =         v2tensors.v2iajb(h7,p3,h1,p8)                          )
      (   i_5(h7,p3,h1,p8)           +=         t1(p5,h1)        * v2tensors.v2iabc(h7,p3,p5,p8)       )
      (     i_6_1(h8,h10,h1,h2)       =         v2tensors.v2ijkl(h8,h10,h1,h2)                         )
      (       i_6_1_1(h8,h10,h1,p5)   =         v2tensors.v2ijka(h8,h10,h1,p5)                         )
      (       i_6_1_1(h8,h10,h1,p5)  += -0.5  * t1(p6,h1)        * v2tensors.v2ijab(h8,h10,p5,p6)      )
      (     i_6_1_temp(h8,h10,h1,h2)  =         t1(p5,h1)        * i_6_1_1(h8,h10,h2,p5) )
      (     i_6_1(h8,h10,h1,h2)      += -1    * i_6_1_temp(h8,h10,h1,h2)                 )
      (     i_6_1(h8,h10,h2,h1)      +=         i_6_1_temp(h8,h10,h1,h2)                 ) //P(h1/h2)
      (     i_6_1(h8,h10,h1,h2)      +=  0.5  * t2(p5,p6,h1,h2)  * v2tensors.v2ijab(h8,h10,p5,p6)      )
      (     i_6_2(h10,p5)             =         f1(h10,p5)                               )
      (     i_6_2(h10,p5)            += -1    * t1(p6,h7)        * v2tensors.v2ijab(h7,h10,p5,p6)      )
      (     i_6_3(h8,h10,h1,p9)       =         v2tensors.v2ijka(h8,h10,h1,p9)                         )
      (     i_6_3(h8,h10,h1,p9)      +=         t1(p5,h1)        * v2tensors.v2ijab(h8,h10,p5,p9)      )
      (     i_8_1(h5,p9)              =         f1(h5,p9)                                )
      (     i_8_1(h5,p9)             += -1    * t1(p6,h7)        * v2tensors.v2ijab(h5,h7,p6,p9)       )
      .deallocate(i_1_1,i_1_2,i_1_3,i_2_1,i_4_1,i_6_1_1,i_1_temp,i_4_temp,i_6_1_temp)
      .execute(exhw);
    // clang-format on
  }

  cc_t2 = std::chrono::high_resolution_clock::now();
  time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(mrank)
    std::cout << std::endl
              << "Time taken for computing intermediates: " << std::fixed << std::setprecision(2)
              << time << " secs" << std::endl;

  //################################################################################
  //  PRINT THE HEADER FOR THE EOM ITERATIONS
  //################################################################################

  if(mrank) {
    std::cout << std::endl << std::endl;
    std::cout << " No. of initial right vectors " << ninitvecs << std::endl;
    std::cout << std::endl;
    std::cout << " EOM-CCSD right-hand side iterations" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << "     Residuum \t      Omega / hartree \t Omega / eV \t Time(s)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
  }

  //################################################################################
  //  MAIN ITERATION LOOP
  //################################################################################

  for(int iter = 0; iter < maxeomiter;) {
    int nxtrials    = 0;
    int newnxtrials = ninitvecs;

    for(int micro = 0; micro < microeomiter; iter++, micro++) {
      cc_t1                  = std::chrono::high_resolution_clock::now();
      const auto timer_start = cc_t1;
      if(mrank) {
        std::cout << std::endl;
        std::cout << " Iteration " << iter + 1 << " using " << newnxtrials << " trial vectors"
                  << std::endl;
        // std::cout << " " << std::string(40, '-') << std::endl;
        sys_data
          .results["output"]["EOMCCSD"]["iter"][std::to_string(iter + 1)]["num_trial_vectors"] =
          newnxtrials;
      }
      for(int root = nxtrials; root < newnxtrials; root++) {
        eomccsd_x1(sch, MO, xp1.at(root), t1, t2, x1.at(root), x2.at(root), f1, v2tensors,
                   x1tensors);
        eomccsd_x2(sch, MO, xp2.at(root), t1, t2, x1.at(root), x2.at(root), f1, v2tensors,
                   x2tensors);
      }
      sch.execute(exhw);

      if(mrank && profile) {
        cc_t2 = std::chrono::high_resolution_clock::now();
        time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        std::cout << std::string(4, ' ') << "- Time taken for X1/X2 Calls: " << std::fixed
                  << std::setprecision(2) << time << " secs" << std::endl;
      }

      //################################################################################
      //  UPDATE HBAR: ELEMENTS FOR THE NEWEST X AND XP VECTORS ARE COMPUTED
      //################################################################################

      if(mrank && profile) cc_t1 = std::chrono::high_resolution_clock::now();
      if(micro == 0) {
        for(int ivec = 0; ivec < newnxtrials; ivec++) {
          for(int jvec = 0; jvec < newnxtrials; jvec++) {
            // clang-format off
            sch(d_r1() = xp1.at(jvec)() * x1.at(ivec)())
               (d_r1() += xp2.at(jvec)() * x2.at(ivec)())
              .execute(exhw);
            // clang-format on

            hbar(ivec, jvec) = get_scalar(d_r1);
          }
        }
      }
      else {
        for(int ivec = 0; ivec < newnxtrials; ivec++) {
          for(int jvec = nxtrials; jvec < newnxtrials; jvec++) {
            // clang-format off
            sch(d_r1() = xp1.at(jvec)() * x1.at(ivec)())
               (d_r1() += xp2.at(jvec)() * x2.at(ivec)())
              .execute(exhw);
            // clang-format on

            hbar(ivec, jvec) = get_scalar(d_r1);
          }
        }

        for(int ivec = nxtrials; ivec < newnxtrials; ivec++) {
          for(int jvec = 0; jvec < nxtrials; jvec++) {
            // clang-format off
            sch(d_r1() = xp1.at(jvec)() * x1.at(ivec)())
               (d_r1() += xp2.at(jvec)() * x2.at(ivec)())
              .execute(exhw);
            // clang-format on

            hbar(ivec, jvec) = get_scalar(d_r1);
          }
        }
      }

      if(mrank && profile) {
        cc_t2 = std::chrono::high_resolution_clock::now();
        time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        std::cout << std::string(4, ' ') << "- Time taken for HBAR Products: " << std::fixed
                  << std::setprecision(2) << time << " secs" << std::endl;
      }

      //################################################################################
      //  DIAGONALIZE HBAR
      //################################################################################

      if(mrank && profile) cc_t1 = std::chrono::high_resolution_clock::now();
      Eigen::EigenSolver<Matrix> hbardiag(hbar.block(0, 0, newnxtrials, newnxtrials));
      auto                       omegar1 = hbardiag.eigenvalues();

      const auto     nev = omegar1.rows();
      std::vector<T> omegar(nev);
      for(auto x = 0; x < nev; x++) omegar[x] = real(omegar1(x));

      //################################################################################
      //  SORT THE EIGENVECTORS AND CORRESPONDING EIGENVALUES
      //################################################################################

      std::vector<size_t> omegar_sorted_order = SCFUtil::sort_indexes(omegar);
      std::sort(omegar.begin(), omegar.end());

      auto hbar_right1 = hbardiag.eigenvectors();
      assert(hbar_right1.rows() == nev && hbar_right1.cols() == nev);
      hbar_right.resize(nev, nev);
      hbar_right.setZero();

      for(auto x = 0; x < nev; x++)
        hbar_right.col(x) = hbar_right1.col(omegar_sorted_order[x]).real();

      if(mrank && profile) {
        cc_t2 = std::chrono::high_resolution_clock::now();
        time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
        std::cout << std::string(4, ' ')
                  << "- Time taken for Diag. and Sort Products: " << std::fixed
                  << std::setprecision(2) << time << " secs" << std::endl;
      }

      nxtrials = newnxtrials;

      for(auto root = 0; root < nroots; root++) {
        if(mrank && profile && root == 0) cc_t1 = std::chrono::high_resolution_clock::now();
        //################################################################################
        //  FORM RESIDUAL VECTORS
        //################################################################################

        sch(r1.at(root)() = 0)(r2.at(root)() = 0).execute();

        for(int i = 0; i < nxtrials; i++) {
          T omegar_hbar_scalar = -1 * omegar[root] * hbar_right(i, root);

          // clang-format off
          sch(r1.at(root)() += omegar_hbar_scalar * x1.at(i)())
             (r2.at(root)() += omegar_hbar_scalar * x2.at(i)())
             (r1.at(root)() += hbar_right(i, root) * xp1.at(i)())
             (r2.at(root)() += hbar_right(i, root) * xp2.at(i)())
            .execute();
          // clang-format on
        }

        T xresidual =
          sqrt(norm(r1.at(root)) * norm(r1.at(root)) + norm(r2.at(root)) * norm(r2.at(root)));

        T newsc = 1 / xresidual;

        if(mrank && profile && root == 0) {
          cc_t2 = std::chrono::high_resolution_clock::now();
          time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
          std::cout << std::string(4, ' ') << "- Time taken for R1/R2 (root 0): " << std::fixed
                    << std::setprecision(2) << time << " secs" << std::endl;
        }

        //################################################################################
        //  EXPAND ITERATIVE SPACE WITH NEW ORTHONORMAL VECTORS
        //################################################################################
        if(mrank && profile && root == 0) cc_t1 = std::chrono::high_resolution_clock::now();
        if(xresidual > eomthresh) {
          int ivec = newnxtrials;
          newnxtrials++;

          if(newnxtrials <= hbardim) {
            // If we can overwrite r1 and r2 here we can replace x1.at(ivec) in the next few lines
            jacobi(ec, r1.at(root), x1.at(ivec), 0.0, false, p_evl_sorted, n_occ_alpha, n_occ_beta);
            jacobi(ec, r2.at(root), x2.at(ivec), 0.0, false, p_evl_sorted, n_occ_alpha, n_occ_beta);

            scale_ip(x1.at(ivec), newsc);
            scale_ip(x2.at(ivec), newsc);

            // clang-format off
            sch(u1() = x1.at(ivec)())
               (u2() = x2.at(ivec)()).execute();

            for(int jvec = 0; jvec < ivec; jvec++) {
              sch(oscalar() = x1.at(ivec)() * x1.at(jvec)())
                 (oscalar() += x2.at(ivec)() * x2.at(jvec)())
                 (u1() += -1.0 * oscalar() * x1.at(jvec)())
                 (u2() += -1.0 * oscalar() * x2.at(jvec)());
            }

            sch(x1.at(ivec)() = u1())
               (x2.at(ivec)() = u2()).execute(exhw);
            // clang-format on

            T newsc = 1 / sqrt(norm(x1.at(ivec)()) * norm(x1.at(ivec)()) +
                               norm(x2.at(ivec)()) * norm(x2.at(ivec)()));

            scale_ip(x1.at(ivec), newsc);
            scale_ip(x2.at(ivec), newsc);
          }
        }
        if(mrank && profile && root == 0) {
          cc_t2 = std::chrono::high_resolution_clock::now();
          time = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
          std::cout << std::string(4, ' ') << "- Time taken for Expanding (root 0): " << std::fixed
                    << std::setprecision(2) << time << " secs" << std::endl;
        }
        if(mrank) {
          std::cout.precision(13);
          // print residual, energy (hartree), energy (eV)
          std::cout << "   " << xresidual << "   " << omegar[root] << "    " << omegar[root] * au2ev
                    << std::endl;
          const std::string rootstr = "root" + std::to_string(root + 1);
          sys_data
            .results["output"]["EOMCCSD"]["iter"][std::to_string(iter + 1)][rootstr]["residual"] =
            xresidual;
          sys_data
            .results["output"]["EOMCCSD"]["iter"][std::to_string(iter + 1)][rootstr]["energy"] =
            omegar[root];
        }
      } // root loop

      const auto timer_end = std::chrono::high_resolution_clock::now();
      auto       iter_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
          .count();

      if(mrank) {
        std::cout << " Iteration " << iter + 1 << " time: ";
        std::cout << std::fixed << std::setprecision(2) << iter_time << " secs" << std::endl;
        sys_data.results["output"]["EOMCCSD"]["iter"][std::to_string(iter + 1)]["performance"]
                        ["total_time"] = iter_time;
        chem_env.write_json_data();
      }

      //################################################################################
      //  CHECK CONVERGENCE
      //################################################################################
      if(nxtrials == newnxtrials) {
        if(mrank) {
          std::cout << std::string(62, '-') << std::endl;
          std::cout << " Iterations converged" << std::endl;
        }
        convflag = true;
        break;
      }

    } // END MICRO

    if(convflag) break;

    //################################################################################
    //  FORM INITAL VECTORS FOR NEWEST MICRO INTERATIONS
    //################################################################################
    if(mrank) {
      std::cout << " END OF MICROITERATIONS: COLLAPSING WITH NEW INITAL VECTORS" << std::endl;
    }

    ninitvecs = nroots;

    for(auto root = 0; root < nroots; root++) {
      sch(xc1.at(root)() = 0)(xc2.at(root)() = 0).execute();

      for(int i = 0; i < nxtrials; i++) {
        T hbr_scalar = hbar_right(i, root);
        // clang-format off
        sch(xc1.at(root)() += hbr_scalar * x1.at(i)())
           (xc2.at(root)() += hbr_scalar * x2.at(i)())
          .execute();
        // clang-format on
      }
    }

    for(auto root = 0; root < nroots; root++) {
      // clang-format off
      sch(x1.at(root)() = xc1.at(root)())
         (x2.at(root)() = xc2.at(root)());
      // clang-format on
    }
    sch.execute();

  } // end convergence loop

  x1tensors.deallocate();
  x2tensors.deallocate();

  Tensor<T>::deallocate(u1, u2, uu2, uuu2, d_r1, oscalar);
  free_vec_tensors(x1, x2, xp1, xp2, xc1, xc2, r1, r2);
}

using T = double;
template void right_eomccsd_driver<T>(ChemEnv& chem_env, ExecutionContext& ec,
                                      const TiledIndexSpace& MO, Tensor<T>& t1, Tensor<T>& t2,
                                      Tensor<T>& f1, exachem::cholesky_2e::V2Tensors<T>& v2tensors,
                                      std::vector<T> p_evl_sorted);
