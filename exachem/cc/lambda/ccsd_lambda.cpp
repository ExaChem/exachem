/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "ccsd_lambda.hpp"

void exachem::cc::ccsd_lambda::iteration_print_lambda(ChemEnv& chem_env, const ProcGroup& pg,
                                                      int iter, double residual, double time) {
  if(pg.rank() == 0) {
    std::cout << std::setw(4) << std::right << iter + 1 << "     ";
    std::cout << std::setprecision(13) << std::setw(16) << std::left << residual << "  ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(5, ' ') << time << std::endl;

    chem_env.sys_data.results["output"]["CCSD_Lambda"]["iter"][std::to_string(iter + 1)] = {
      {"residual", residual}};
    chem_env.sys_data.results["output"]["CCSD_Lambda"]["iter"][std::to_string(iter + 1)]
                             ["performance"] = {{"total_time", time}};
  }
}

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
           std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
exachem::cc::ccsd_lambda::setupLambdaTensors(ExecutionContext& ec, TiledIndexSpace& MO,
                                             size_t ndiis) {
  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  auto rank = ec.pg().rank();

  Tensor<T> d_r1{{O, V}, {1, 1}};
  Tensor<T> d_r2{{O, O, V, V}, {2, 2}};
  Tensor<T> d_y1{{O, V}, {1, 1}};
  Tensor<T> d_y2{{O, O, V, V}, {2, 2}};

  Tensor<T>::allocate(&ec, d_r1, d_r2, d_y1, d_y2);
  // clang-format off
  Scheduler{ec}
    (d_y1() = 0)
    (d_y2() = 0)
    (d_r1() = 0)
    (d_r2() = 0)
  .execute();
  // clang-format on

  if(rank == 0) {
    std::cout << std::endl << std::endl;
    std::cout << " Lambda CCSD iterations" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    std::cout << "  Iter     Residuum \t      Time(s)" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
  }

  std::vector<Tensor<T>> d_r1s, d_r2s, d_y1s, d_y2s;

  for(size_t i = 0; i < ndiis; i++) {
    d_r1s.push_back(Tensor<T>{{O, V}, {1, 1}});
    d_r2s.push_back(Tensor<T>{{O, O, V, V}, {2, 2}});

    d_y1s.push_back(Tensor<T>{{O, V}, {1, 1}});
    d_y2s.push_back(Tensor<T>{{O, O, V, V}, {2, 2}});
    Tensor<T>::allocate(&ec, d_r1s[i], d_r2s[i], d_y1s[i], d_y2s[i]);
  }

  return std::make_tuple(d_r1, d_r2, d_y1, d_y2, d_r1s, d_r2s, d_y1s, d_y2s);
}

template<typename T>
void exachem::cc::ccsd_lambda::lambda_ccsd_y1(Scheduler& sch, const TiledIndexSpace& MO,
                                              const TiledIndexSpace& CI, Tensor<T>& i0,
                                              const Tensor<T>& t1, const Tensor<T>& t2,
                                              const Tensor<T>& y1, const Tensor<T>& y2,
                                              const Tensor<T>&           f1,
                                              cholesky_2e::V2Tensors<T>& v2tensors, Tensor<T>& cv3d,
                                              Y1Tensors<T>& y1tensors) {
  // const TiledIndexSpace& O = MO("occ");
  // const TiledIndexSpace& V = MO("virt");
  auto [cind] = CI.labels<1>("all");

  TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
  TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

  std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)      = MO.labels<11>("virt");
  std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

  Tensor<T> i_1      = y1tensors.i_1;
  Tensor<T> i_1_1    = y1tensors.i_1_1;
  Tensor<T> i_2      = y1tensors.i_2;
  Tensor<T> i_2_1    = y1tensors.i_2_1;
  Tensor<T> i_3      = y1tensors.i_3;
  Tensor<T> i_3_1    = y1tensors.i_3_1;
  Tensor<T> i_3_1_1  = y1tensors.i_3_1_1;
  Tensor<T> i_3_2    = y1tensors.i_3_2;
  Tensor<T> i_3_3    = y1tensors.i_3_3;
  Tensor<T> i_3_4    = y1tensors.i_3_4;
  Tensor<T> i_4      = y1tensors.i_4;
  Tensor<T> i_4_1    = y1tensors.i_4_1;
  Tensor<T> i_4_1_1  = y1tensors.i_4_1_1;
  Tensor<T> i_4_2    = y1tensors.i_4_2;
  Tensor<T> i_4_3    = y1tensors.i_4_3;
  Tensor<T> i_4_4    = y1tensors.i_4_4;
  Tensor<T> i_5      = y1tensors.i_5;
  Tensor<T> i_6      = y1tensors.i_6;
  Tensor<T> i_6_1    = y1tensors.i_6_1;
  Tensor<T> i_6_2    = y1tensors.i_6_2;
  Tensor<T> i_7      = y1tensors.i_7;
  Tensor<T> i_8      = y1tensors.i_8;
  Tensor<T> i_9      = y1tensors.i_9;
  Tensor<T> i_10     = y1tensors.i_10;
  Tensor<T> i_11     = y1tensors.i_11;
  Tensor<T> i_11_1   = y1tensors.i_11_1;
  Tensor<T> i_11_1_1 = y1tensors.i_11_1_1;
  Tensor<T> i_11_2   = y1tensors.i_11_2;
  Tensor<T> i_11_3   = y1tensors.i_11_3;
  Tensor<T> i_12     = y1tensors.i_12;
  Tensor<T> i_12_1   = y1tensors.i_12_1;
  Tensor<T> i_13     = y1tensors.i_13;
  Tensor<T> i_13_1   = y1tensors.i_13_1;

  Tensor<T> tmp_1  = y1tensors.tmp_1;
  Tensor<T> tmp_2  = y1tensors.tmp_2;
  Tensor<T> tmp_4  = y1tensors.tmp_4;
  Tensor<T> tmp_5  = y1tensors.tmp_5;
  Tensor<T> tmp_6  = y1tensors.tmp_6;
  Tensor<T> tmp_7  = y1tensors.tmp_7;
  Tensor<T> tmp_8  = y1tensors.tmp_8;
  Tensor<T> tmp_9  = y1tensors.tmp_9;
  Tensor<T> tmp_11 = y1tensors.tmp_11;
  Tensor<T> tmp_12 = y1tensors.tmp_12;
  Tensor<T> tmp_13 = y1tensors.tmp_13;
  Tensor<T> tmp_14 = y1tensors.tmp_14;
  Tensor<T> tmp_15 = y1tensors.tmp_15;
  Tensor<T> tmp_16 = y1tensors.tmp_16;
  Tensor<T> tmp_19 = y1tensors.tmp_19;
  Tensor<T> tmp_20 = y1tensors.tmp_20;

  // clang-format off
  sch
    ( i0(h2,p1)                     =         f1(h2,p1)                                               )
    (   i_1(h2,h7)                  =         f1(h2,h7)                                               )
    (     i_1_1(h2,p3)              =         f1(h2,p3)                                               )
    (     i_1_1(h2,p3)             +=         t1(p5,h6)            * v2tensors.v2ijab(h2,h6,p3,p5)    )
    (   i_1(h2,h7)                 +=         t1(p3,h7)            * i_1_1(h2,p3)                     )
    (   i_1(h2,h7)                 +=         t1(p3,h4)            * v2tensors.v2ijka(h2,h4,h7,p3)    )
    (   i_1(h2,h7)                 += -0.5  * t2(p3,p4,h6,h7)      * v2tensors.v2ijab(h2,h6,p3,p4)    )
    ( i0(h2,p1)                    += -1.0  * y1(h7,p1)            * i_1(h2,h7)                       )
    (   i_2(p7,p1)                  =         f1(p7,p1)                                               )
  //(   i_2(p7,p1)                 += -1.0  * t1(p3,h4)            * v2tensors.v2iabc(h4,p7,p1,p3)    )
    (  tmp_1(p7,h4,cind)            =         cv3d(p7,p3,cind)     * t1(p3,h4)                        ) 
    (   i_2(p7,p1)                 += -1.0  * tmp_1(p7,h4,cind)    * cv3d(h4,p1,cind)                 )
    (  tmp_2(cind)                  =         t1(p3,h4)            * cv3d(h4,p3,cind)                 )
    (   i_2(p7,p1)                 +=         cv3d(p7,p1,cind)     * tmp_2(cind)                      )        
    (     i_2_1(h4,p1)              =         t1(p5,h6)            * v2tensors.v2ijab(h4,h6,p1,p5)    )
    (   i_2(p7,p1)                 += -1    * t1(p7,h4)            * i_2_1(h4,p1)                     )
    ( i0(h2,p1)                    +=         y1(h2,p7)            * i_2(p7,p1)                       )
    ( i0(h2,p1)                    += -1    * y1(h4,p3)            * v2tensors.v2iajb(h2,p3,h4,p1)    )
    (   i_3(p9,h11)                 =         f1(p9,h11)                                              )
    (     i_3_1(h10,h11)            =         f1(h10,h11)                                             )
    (       i_3_1_1(h10,p3)         =         f1(h10,p3)                                              )
    (       i_3_1_1(h10,p3)        += -1    * t1(p7,h8)            * v2tensors.v2ijab(h8,h10,p3,p7)   ) 
    (     i_3_1(h10,h11)           +=         t1(p3,h11)           * i_3_1_1(h10,p3)                  )
    (     i_3_1(h10,h11)           += -1    * t1(p5,h6)            * v2tensors.v2ijka(h6,h10,h11,p5)  )
    (     i_3_1(h10,h11)           +=  0.5  * t2(p3,p4,h6,h11)     * v2tensors.v2ijab(h6,h10,p3,p4)   )
    (   i_3(p9,h11)                += -1    * t1(p9,h10)           * i_3_1(h10,h11)                   )
    (     i_3_2(p9,p7)              =         f1(p9,p7)                                               )
  //(  i_3_2(p9,p7)                +=         t1(p5,h6)            * v2tensors.v2iabc(h6,p9,p5,p7)    )
  //(  tmp_3(cind)                  =         cv3d(h6,p5,cind)     * t1(p5,h6)                        )
    (     i_3_2(p9,p7)             +=         tmp_2(cind)          * cv3d (p9,p7,cind)                ) 
    (   tmp_4(p5,p7,cind)           =         t1(p5,h6)            * cv3d(h6,p7,cind)                 )
    (     i_3_2(p9,p7)             += -1.0  * cv3d(p9,p5,cind)     * tmp_4(p5,p7,cind)                )      
    (   i_3(p9,h11)                +=         t1(p7,h11)           * i_3_2(p9,p7)                     )
    (   i_3(p9,h11)                += -1    * t1(p3,h4)            * v2tensors.v2iajb(h4,p9,h11,p3)   )
    (     i_3_3(h5,p4)              =         f1(h5,p4)                                               )
    (     i_3_3(h5,p4)             +=         t1(p7,h8)            * v2tensors.v2ijab(h5,h8,p4,p7)    )
    (   i_3(p9,h11)                +=         t2(p4,p9,h5,h11)     * i_3_3(h5,p4)                     )
    (     i_3_4(h5,h6,h11,p4)       =         v2tensors.v2ijka(h5,h6,h11,p4)                          )
    (     i_3_4(h5,h6,h11,p4)      += -1.0  * t1(p7,h11)           * v2tensors.v2ijab(h5,h6,p4,p7)    )
    (   i_3(p9,h11)                +=  0.5  * t2(p4,p9,h5,h6)      * i_3_4(h5,h6,h11,p4)              )
  //(   i_3(p9,h11)                +=  0.5  * t2(p3,p4,h6,h11)     * v2tensors.v2iabc(h6,p9,p3,p4)    )
    (  tmp_5(p4,h11,cind)           =         t2(p3,p4,h6,h11)     * cv3d(h6,p3,cind)                 )
    (   i_3(p9,h11)                +=  0.5  * cv3d(p9,p4,cind)     * tmp_5(p4,h11,cind)               ) 
    (  tmp_6(p3,h11,cind)           =         t2(p3,p4,h6,h11)     * cv3d(h6,p4,cind)                 )
    (   i_3(p9,h11)                += -0.5  * cv3d(p9,p3,cind)     * tmp_6(p3,h11,cind)               )
    ( i0(h2,p1)                    +=         y2(h2,h11,p1,p9)     * i_3(p9,h11)                      )      
    (   i_4(h2,p9,h11,h12)          =         v2tensors.v2ijka(h11,h12,h2,p9)                         )
    (     i_4_1(h2,h7,h11,h12)      =         v2tensors.v2ijkl(h2,h7,h11,h12)                         )
    (       i_4_1_1(h2,h7,h12,p3)   =         v2tensors.v2ijka(h2,h7,h12,p3)                          )
    (       i_4_1_1(h2,h7,h12,p3)  += -0.5  * t1(p5,h12)           * v2tensors.v2ijab(h2,h7,p3,p5)    )
    (     i_4_1(h2,h7,h11,h12)     += -2    * t1(p3,h11)           * i_4_1_1(h2,h7,h12,p3)            )
    (     i_4_1(h2,h7,h11,h12)     +=  0.5  * t2(p3,p4,h11,h12)    * v2tensors.v2ijab(h2,h7,p3,p4)    )
    (   i_4(h2,p9,h11,h12)         += -1    * t1(p9,h7)            * i_4_1(h2,h7,h11,h12)             )
    (     i_4_2(h2,p9,h12,p3)       =         v2tensors.v2iajb(h2,p9,h12,p3)                          )
  //(     i_4_2(h2,p9,h12,p3)      += -0.5  * t1(p5,h12)           * v2tensors.v2iabc(h2,p9,p3,p5)    )
    (   tmp_7(p9,h12,cind)          =         cv3d(p9,p5,cind)     * t1(p5,h12)                       )
    (        i_4_2(h2,p9,h12,p3)   += -0.5  * tmp_7(p9,h12,cind)   * cv3d(h2,p3,cind)                 )
    (   tmp_8(h2,h12,cind)          =         cv3d(h2,p5,cind)     * t1(p5,h12)                       )
    (     i_4_2(h2,p9,h12,p3)      +=  0.5  * tmp_8(h2,h12,cind)   * cv3d(p9,p3,cind)                 )
    (   i_4(h2,p9,h11,h12)         += -2.0  * t1(p3,h11)           * i_4_2(h2,p9,h12,p3)              )
    (     i_4_3(h2,p5)              =         f1(h2,p5)                                               )
    (     i_4_3(h2,p5)             +=         t1(p7,h8)            * v2tensors.v2ijab(h2,h8,p5,p7)    )
    (   i_4(h2,p9,h11,h12)         +=         t2(p5,p9,h11,h12)    * i_4_3(h2,p5)                     )
    (     i_4_4(h2,h6,h12,p4)       =         v2tensors.v2ijka(h2,h6,h12,p4)                          )
    (     i_4_4(h2,h6,h12,p4)      += -1.0  * t1(p7,h12)           * v2tensors.v2ijab(h2,h6,p4,p7)    )
    (   i_4(h2,p9,h11,h12)         += -2.0  * t2(p4,p9,h6,h11)     * i_4_4(h2,h6,h12,p4)              )
  //(   i_4(h2,p9,h11,h12)         +=  0.5  * t2(p3,p4,h11,h12)    * v2tensors.v2iabc(h2,p9,p3,p4)    )
    (   tmp_9(h2,p9,p3,p4)          =  0.5  * cv3d(h2,p3,cind)     * cv3d(p9,p4,cind)                 )
    (   tmp_9(h2,p9,p3,p4)         += -0.5  * cv3d(h2,p4,cind)     * cv3d(p9,p3,cind)                 ) // *t2(p3,p4,h11,h12))
  //(   i_4(h2,p9,h11,h12)         +=  0.5  * tmp_9(h2,p9,p3,p4)   * t2(p3,p4,h11,h12)                ) // cv3d(p9,p4,cind)) 
  //(   tmp_10(h2,p9,p3,p4)         =         cv3d(h2,p4,cind)     * cv3d(p9,p3,cind)                 ) // *t2(p3,p4,h11,h12))
    (   i_4(h2,p9,h11,h12)         +=         tmp_9(h2,p9,p3,p4)   * t2(p3,p4,h11,h12)                )
    ( i0(h2,p1)                    += -0.5  * y2(h11,h12,p1,p9)    * i_4(h2,p9,h11,h12)               )
  //need to reorder
  //(   i_5(p5,p8,h7,p1)            = -1.0  * v2tensors.v2iabc(h7,p1,p5,p8)                           )
  //(   i_5(p5,p8,h7,p1)            = -1.0  * cv3d(h7,p5,cind)     * cv3d(p1,p8,cind)                 )
  //(   i_5(p5,p8,h7,p1)           +=         cv3d(h7,p8,cind)     * cv3d(p1,p5,cind)                 )
  //( i0(h2,p1)                    +=  0.5  * y2(h2,h7,p5,p8)      * i_5(p5,p8,h7,p1)                 ) 
    (  tmp_19(h2,p5,cind)           =         y2(h2,h7,p5,p8)      * cv3d(p8,h7,cind)                 )
    (  i0(h2,p1)                   +=  0.5  * tmp_19(h2,p5,cind)   * cv3d(p5,p1,cind)                 )
    (  tmp_20(h2,p8,cind)           =         y2(h2,h7,p5,p8)      * cv3d(p5,h7,cind)                 )
    (  i0(h2,p1)                   += -0.5  * tmp_20(h2,p8,cind)   * cv3d(p8,p1,cind)                 )  
  //need to refactorize
  //(  i_5(p5,p8,h7,p1)             =         t1(p3,h7)            * v2tensors.v2abcd(p5,p8,p3,p1)    )
  //(  i0(h2,p1)                   +=  0.5  * y2(h7,h2,p5,p8)      * i_5(p5,p8,h7,p1)                 )
    (  tmp_11(p5,p8,p1,p3)          =         cv3d(p8,p1,cind)     * cv3d(p5,p3,cind)                 ) // y2(h7,h2,p5,p8)*cv3d(p8,p1,cind))
    (  tmp_11(p5,p8,p1,p3)         += -1.0  * cv3d(p8,p3,cind)     * cv3d(p5,p1,cind)                 )
    (  tmp_12(h7,h2,p1,p3)          =         y2(h7,h2,p5,p8)      * tmp_11(p5,p8,p1,p3)              ) // tmp_11(h2,h7,p5,p1,cind)*cv3d(p5,p3,cind))
    ( i0(h2,p1)                    +=  0.5  * tmp_12(h7,h2,p1,p3)  * t1(p3,h7)                        )
    (   i_6(p9,h10)                 =         t1(p9,h10)                                              )
    (   i_6(p9,h10)                +=         t2(p3,p9,h5,h10)     * y1(h5,p3)                        )
    (     i_6_1(h6,h10)             =         t1(p5,h10)           * y1(h6,p5)                        )
    (     i_6_1(h6,h10)            +=  0.5  * t2(p3,p4,h5,h10)     * y2(h5,h6,p3,p4)                  )
    (   i_6(p9,h10)                += -1    * t1(p9,h6)            * i_6_1(h6,h10)                    )
    (     i_6_2(h5,h6,h10,p3)       =         t1(p7,h10)           * y2(h5,h6,p3,p7)                  )
    (   i_6(p9,h10)                += -0.5  * t2(p3,p9,h5,h6)      * i_6_2(h5,h6,h10,p3)              )
    ( i0(h2,p1)                    +=         i_6(p9,h10)          * v2tensors.v2ijab(h2,h10,p1,p9)   )
    (   i_7(h2,h3)                  =         t1(p4,h3)            * y1(h2,p4)                        )
    (   i_7(h2,h3)                 +=  0.5  * t2(p4,p5,h3,h6)      * y2(h2,h6,p4,p5)                  )
    ( i0(h2,p1)                    += -1    * i_7(h2,h3)           * f1(h3,p1)                        )
    (   i_8(h6,h8)                  =         t1(p3,h8)            * y1(h6,p3)                        )
    (   i_8(h6,h8)                 +=  0.5  * t2(p3,p4,h5,h8)      * y2(h5,h6,p3,p4)                  )
    ( i0(h2,p1)                    +=         i_8(h6,h8)           * v2tensors.v2ijka(h2,h8,h6,p1)    )
    (   i_9(p7,p8)                  =         t1(p7,h4)            * y1(h4,p8)                        )
    (   i_9(p7,p8)                 +=  0.5  * t2(p3,p7,h5,h6)      * y2(h5,h6,p3,p8)                  )
  //( i0(h2,p1)                    +=         i_9(p7,p8)           * v2tensors.v2iabc(h2,p8,p1,p7)    )
    ( tmp_13(cind)                  =         cv3d(p8,p7,cind)     * i_9(p7,p8)                       )
    (  i0(h2,p1)                   +=         tmp_13(cind)         * cv3d(h2,p1,cind)                 ) 
    ( tmp_14(h2,p8,cind)            =         cv3d(h2,p7,cind)     * i_9(p7,p8)                       )
    ( i0(h2,p1)                    += -1.0  * tmp_14(h2,p8,cind)   * cv3d(p8,p1,cind)                 )        
    (   i_10(h2,h6,h4,p5)           =         t1(p3,h4)            * y2(h2,h6,p3,p5)                  )
    ( i0(h2,p1)                    +=         i_10(h2,h6,h4,p5)    * v2tensors.v2iajb(h4,p5,h6,p1)    )
    (   i_11(h2,p9,h6,h12)          = -1    * t2(p3,p9,h6,h12)     * y1(h2,p3)                        )
    (     i_11_1(h2,h10,h6,h12)     = -1    * t2(p3,p4,h6,h12)     * y2(h2,h10,p3,p4)                 )
    (       i_11_1_1(h2,h10,h6,p5)  =         t1(p7,h6)            * y2(h2,h10,p5,p7)                 )
    (     i_11_1(h2,h10,h6,h12)    +=  2    * t1(p5,h12)           * i_11_1_1(h2,h10,h6,p5)           )
    (   i_11(h2,p9,h6,h12)         += -0.5  * t1(p9,h10)           * i_11_1(h2,h10,h6,h12)            )
    (     i_11_2(h2,h5,h6,p3)       =         t1(p7,h6)            * y2(h2,h5,p3,p7)                  )
    (   i_11(h2,p9,h6,h12)         +=  2    * t2(p3,p9,h5,h12)     * i_11_2(h2,h5,h6,p3)              )
    (     i_11_3(h2,h12)            =         t2(p3,p4,h5,h12)     * y2(h2,h5,p3,p4)                  )
    (   i_11(h2,p9,h6,h12)         += -1    * t1(p9,h6)            * i_11_3(h2,h12)                   )
    ( i0(h2,p1)                    +=  0.5  * i_11(h2,p9,h6,h12)   * v2tensors.v2ijab(h6,h12,p1,p9)   )
    (   i_12(h2,h7,h6,h8)           = -1    * t2(p3,p4,h6,h8)      * y2(h2,h7,p3,p4)                  )
    (     i_12_1(h2,h7,h6,p3)       =         t1(p5,h6)            * y2(h2,h7,p3,p5)                  )
    (   i_12(h2,h7,h6,h8)          +=  2    * t1(p3,h8)            * i_12_1(h2,h7,h6,p3)              )
    ( i0(h2,p1)                    +=  0.25 * i_12(h2,h7,h6,h8)    * v2tensors.v2ijka(h6,h8,h7,p1)    )
    (   i_13(h2,p8,h6,p7)           =         t2(p3,p8,h5,h6)      * y2(h2,h5,p3,p7)                  )
    (     i_13_1(h2,h4,h6,p7)       =         t1(p5,h6)            * y2(h2,h4,p5,p7)                  )
    (   i_13(h2,p8,h6,p7)          += -1    * t1(p8,h4)            * i_13_1(h2,h4,h6,p7)              )
  //( i0(h2,p1)                    +=         i_13(h2,p8,h6,p7)    * v2tensors.v2iabc(h6,p7,p1,p8)    ) 
    ( tmp_15(h2,h6,cind)            =         i_13(h2,p8,h6,p7)    * cv3d(p7,p8,cind)                 )
    ( i0(h2,p1)                    +=         tmp_15(h2,h6,cind)   * cv3d(h6,p1,cind)                 )
    ( tmp_16(h2,p7,cind)            =         i_13(h2,p8,h6,p7)    * cv3d(h6,p8,cind)                 )
    ( i0(h2,p1)                    += -1.0  * tmp_16(h2,p7,cind)   * cv3d(p7,p1,cind)                 );
  // clang-format on
}

template<typename T>
void exachem::cc::ccsd_lambda::lambda_ccsd_y2(
  Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& i0,
  const Tensor<T>& t1, Tensor<T>& t2, const Tensor<T>& y1, Tensor<T>& y2, const Tensor<T>& f1,
  cholesky_2e::V2Tensors<T>& v2tensors, Tensor<T>& cv3d, Y2Tensors<T>& y2tensors) {
  // const TiledIndexSpace& O = MO("occ");
  // const TiledIndexSpace& V = MO("virt");
  auto [cind] = CI.labels<1>("all");

  TiledIndexLabel p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
  TiledIndexLabel h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;

  std::tie(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)      = MO.labels<11>("virt");
  std::tie(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12) = MO.labels<12>("occ");

  Tensor<T> i_1    = y2tensors.i_1;
  Tensor<T> i_2    = y2tensors.i_2;
  Tensor<T> i_3    = y2tensors.i_3;
  Tensor<T> i_3_1  = y2tensors.i_3_1;
  Tensor<T> i_4    = y2tensors.i_4;
  Tensor<T> i_4_1  = y2tensors.i_4_1;
  Tensor<T> i_5    = y2tensors.i_5;
  Tensor<T> i_5_1  = y2tensors.i_5_1;
  Tensor<T> i_6    = y2tensors.i_6;
  Tensor<T> i_7    = y2tensors.i_7;
  Tensor<T> i_8    = y2tensors.i_8;
  Tensor<T> i_9    = y2tensors.i_9;
  Tensor<T> i_10   = y2tensors.i_10;
  Tensor<T> i_11   = y2tensors.i_11;
  Tensor<T> i_12   = y2tensors.i_12;
  Tensor<T> i_12_1 = y2tensors.i_12_1;
  Tensor<T> i_13   = y2tensors.i_13;
  Tensor<T> i_13_1 = y2tensors.i_13_1;

  Tensor<T> tmp_1  = y2tensors.tmp_1;
  Tensor<T> tmp_2  = y2tensors.tmp_2;
  Tensor<T> tmp_3  = y2tensors.tmp_3;
  Tensor<T> tmp_4  = y2tensors.tmp_4;
  Tensor<T> tmp_5  = y2tensors.tmp_5;
  Tensor<T> tmp_6  = y2tensors.tmp_6;
  Tensor<T> tmp_7  = y2tensors.tmp_7;
  Tensor<T> tmp_8  = y2tensors.tmp_8;
  Tensor<T> tmp_9  = y2tensors.tmp_9;
  Tensor<T> tmp_10 = y2tensors.tmp_10;
  Tensor<T> tmp_11 = y2tensors.tmp_11;
  Tensor<T> tmp_12 = y2tensors.tmp_12;
  Tensor<T> tmp_13 = y2tensors.tmp_13;

  // clang-format off
  sch
    ( i0(h3,h4,p1,p2)           =         v2tensors.v2ijab(h3,h4,p1,p2)                        )
    (   i_1(h3,p1)              =         f1(h3,p1)                                            )
    (   i_1(h3,p1)             +=         t1(p5,h6)           * v2tensors.v2ijab(h3,h6,p1,p5)  )
    ( i0(h3,h4,p1,p2)          +=         y1(h3,p1)           * i_1(h4,p2)                     )
    ( i0(h3,h4,p2,p1)          += -1.0  * y1(h3,p1)           * i_1(h4,p2)                     ) //P(p1/p2)
    ( i0(h4,h3,p1,p2)          += -1.0  * y1(h3,p1)           * i_1(h4,p2)                     ) //P(h3/h4)
    ( i0(h4,h3,p2,p1)          +=         y1(h3,p1)           * i_1(h4,p2)                     ) //P(p1/p2,h3/h4)
    (   i_2(h3,h4,h7,p1)        =         v2tensors.v2ijka(h3,h4,h7,p1)                        )
    (   i_2(h3,h4,h7,p1)       += -1    * t1(p5,h7)           * v2tensors.v2ijab(h3,h4,p1,p5)  )
    ( i0(h3,h4,p1,p2)          += -1    * y1(h7,p1)           * i_2(h3,h4,h7,p2)               )
    ( i0(h3,h4,p2,p1)          +=         y1(h7,p1)           * i_2(h3,h4,h7,p2)               ) //P(p1/p2)
  //( i0(h3,h4,p1,p2)          += -1    * y1(h3,p5)           * v2tensors.v2iabc(h4,p5,p1,p2)  )
    ( tmp_1(h3,p2,cind)         =         y1(h3,p5)           * cv3d(p5,p2,cind)               )
    ( i0(h3,h4,p1,p2)          += -1.0  * tmp_1(h3,p2,cind)   * cv3d(h4,p1,cind)               )
    ( tmp_2(h3,p1,cind)         =         y1(h3,p5)           * cv3d(p5,p1,cind)               )
    ( i0(h3,h4,p1,p2)          +=         tmp_2(h3,p1,cind)   * cv3d(h4,p2,cind)               ) 
  //( i0(h4,h3,p1,p2)          +=         y1(h3,p5)           * v2tensors.v2iabc(h4,p5,p1,p2)  ) //P(h3/h4)
    ( tmp_3(h3,p2,cind)         =         y1(h3,p5)           * cv3d(p5,p2,cind)               )
    ( i0(h4,h3,p1,p2)          +=         tmp_3(h3,p2,cind)   * cv3d(h4,p1,cind)               )
    ( tmp_4(h3,p1,cind)         =         y1(h3,p5)           * cv3d(p5,p1,cind)               )
    ( i0(h4,h3,p1,p2)          += -1.0  * tmp_4(h3,p1,cind)   * cv3d(h4,p2,cind)               )
    (   i_3(h3,h9)              =         f1(h3,h9)                                            )
    (     i_3_1(h3,p5)          =         f1(h3,p5)                                            )
    (     i_3_1(h3,p5)         +=         t1(p7,h8)           * v2tensors.v2ijab(h3,h8,p5,p7)  )
    (   i_3(h3,h9)             +=         t1(p5,h9)           * i_3_1(h3,p5)                   )
    (   i_3(h3,h9)             +=         t1(p5,h6)           * v2tensors.v2ijka(h3,h6,h9,p5)  )
    (   i_3(h3,h9)             += -0.5  * t2(p5,p6,h8,h9)     * v2tensors.v2ijab(h3,h8,p5,p6)  )
    ( i0(h3,h4,p1,p2)          += -1    * y2(h3,h9,p1,p2)     * i_3(h4,h9)                     )
    ( i0(h4,h3,p1,p2)          +=         y2(h3,h9,p1,p2)     * i_3(h4,h9)                     ) //P(h3/h4)
    (   i_4(p10,p1)             =         f1(p10,p1)                                           )
  //(   i_4(p10,p1)            += -1    * t1(p5,h6)           * v2tensors.v2iabc(h6,p10,p1,p5) )
    ( tmp_5(h6,p10,cind)       =          t1(p5,h6)           * cv3d(p10,p5,cind)              )
    (  i_4(p10,p1)             += -1.0  * tmp_5(h6,p10,cind)  * cv3d(h6,p1,cind)               )
    ( tmp_6(cind)               =         t1(p5,h6)           * cv3d(h6,p5,cind)               )
    (  i_4(p10,p1)             +=         tmp_6(cind)         * cv3d(p10,p1,cind)              )    
    (   i_4(p10,p1)            +=  0.5  * t2(p6,p10,h7,h8)    * v2tensors.v2ijab(h7,h8,p1,p6)  )
    (     i_4_1(h6,p1)          =         t1(p7,h8)           * v2tensors.v2ijab(h6,h8,p1,p7)  )
    (   i_4(p10,p1)            += -1    * t1(p10,h6)          * i_4_1(h6,p1)                   )
    ( i0(h3,h4,p1,p2)          +=         y2(h3,h4,p1,p10)    * i_4(p10,p2)                    )
    ( i0(h3,h4,p2,p1)          += -1    * y2(h3,h4,p1,p10)    * i_4(p10,p2)                    ) //P(p1/p2)
    (   i_5(h3,h4,h9,h10)       =         v2tensors.v2ijkl(h3,h4,h9,h10)                       )
    (     i_5_1(h3,h4,h10,p5)   =         v2tensors.v2ijka(h3,h4,h10,p5)                       )
    (     i_5_1(h3,h4,h10,p5)  += -0.5  * t1(p7,h10)          * v2tensors.v2ijab(h3,h4,p5,p7)  )
    (   i_5(h3,h4,h9,h10)      += -2    * t1(p5,h9)           * i_5_1(h3,h4,h10,p5)            )
    (   i_5(h3,h4,h9,h10)      +=  0.5  * t2(p5,p6,h9,h10)    * v2tensors.v2ijab(h3,h4,p5,p6)  )
    ( i0(h3,h4,p1,p2)          +=  0.5  * y2(h9,h10,p1,p2)    * i_5(h3,h4,h9,h10)              )
    (   i_6(h3,p7,h9,p1)        =         v2tensors.v2iajb(h3,p7,h9,p1)                        )
  //(   i_6(h3,p7,h9,p1)       += -1    * t1(p5,h9)           * v2tensors.v2iabc(h3,p7,p1,p5)  )
    ( tmp_7(h9,p7,cind)         =         t1(p5,h9)           * cv3d(p7,p5,cind)               )
    (   i_6(h3,p7,h9,p1)       += -1.0  * tmp_7(h9,p7,cind)   * cv3d(h3,p1,cind)               ) 
    ( tmp_8(h9,h3,cind)         =         t1(p5,h9)           * cv3d(h3,p5,cind)               )
    (   i_6(h3,p7,h9,p1)       +=         tmp_8(h9,h3,cind)   * cv3d(p7,p1,cind)               )
    (   i_6(h3,p7,h9,p1)       += -1    * t2(p6,p7,h8,h9)     * v2tensors.v2ijab(h3,h8,p1,p6)  )
  //( i0(h3,h4,p1,p2)          += -1    * y2(h3,h9,p1,p7)     * i_6(h4,p7,h9,p2)               )
  //( i0(h3,h4,p2,p1)          +=         y2(h3,h9,p1,p7)     * i_6(h4,p7,h9,p2)               ) //P(p1/p2)
  //( i0(h4,h3,p1,p2)          +=         y2(h3,h9,p1,p7)     * i_6(h4,p7,h9,p2)               ) //P(h3/h4)
  //( i0(h4,h3,p2,p1)          += -1    * y2(h3,h9,p1,p7)     * i_6(h4,p7,h9,p2)               ) //P(p1/p2,h3/h3)
    ( tmp_10(h3,h4,p1,p2)       =         y2(h3,h9,p1,p7)     * i_6(h4,p7,h9,p2)               )
    ( i0(h3,h4,p1,p2)          += -1    * tmp_10(h3,h4,p1,p2)                                  )
    ( i0(h3,h4,p2,p1)          +=         tmp_10(h3,h4,p1,p2)                                  )
    ( i0(h4,h3,p1,p2)          +=         tmp_10(h3,h4,p1,p2)                                  )
    ( i0(h4,h3,p2,p1)          += -1    * tmp_10(h3,h4,p1,p2)                                  )
  //( i0(h3,h4,p1,p2)          +=  0.5  * y2(h3,h4,p5,p6)     * v2tensors.v2abcd(p5,p6,p1,p2)  )
  //( tmp_9(h3,h4,p5,p2,cind)   =         y2(h3,h4,p5,p6)     * cv3d(p6,p2,cind)               )
  //( i0(h3,h4,p1,p2)          +=  0.5  * tmp_9(h3,h4,p5,p2,cind) * cv3d(p5,p1,cind)           )
  //( tmp_10(h3,h4,p5,p1,cind)  =         y2(h3,h4,p5,p6)     * cv3d(p6,p1,cind)               )
  //( i0(h3,h4,p1,p2)          += -0.5  * tmp_10(h3,h4,p5,p1,cind)*cv3d(p5,p2,cind)            )
    ( tmp_9(p5,p6,p1,p2)        =  0.5  * cv3d(p5,p1,cind)    * cv3d(p6,p2,cind)               )
    ( tmp_9(p5,p6,p1,p2)       += -0.5  * cv3d(p5,p2,cind)    * cv3d(p6,p1,cind)               )
    ( i0(h3,h4,p1,p2)          +=         y2(h3,h4,p5,p6)     * tmp_9(p5,p6,p1,p2)             )
    (   i_7(h3,h9)              =         t1(p5,h9)           * y1(h3,p5)                      )
    (   i_7(h3,h9)             += -0.5  * t2(p5,p6,h7,h9)     * y2(h3,h7,p5,p6)                )
    ( i0(h3,h4,p1,p2)          +=         i_7(h3,h9)          * v2tensors.v2ijab(h4,h9,p1,p2)  )
    ( i0(h4,h3,p1,p2)          += -1    * i_7(h3,h9)          * v2tensors.v2ijab(h4,h9,p1,p2)  ) //P(h3/h4)
    (   i_8(h3,h4,h5,p1)        = -1    * t1(p6,h5)           * y2(h3,h4,p1,p6)                )
    ( i0(h3,h4,p1,p2)          +=         i_8(h3,h4,h5,p1)    * f1(h5,p2)                      )
    ( i0(h3,h4,p1,p2)          += -1    * i_8(h3,h4,h5,p1)    * f1(h5,p2)                      ) //P(p1/p2)
    (   i_9(h3,h7,h6,p1)        =         t1(p5,h6)           * y2(h3,h7,p1,p5)                )
  //( i0(h3,h4,p1,p2)          +=         i_9(h3,h7,h6,p1)    * v2tensors.v2ijka(h4,h6,h7,p2)  )
  //( i0(h3,h4,p2,p1)          += -1    * i_9(h3,h7,h6,p1)    * v2tensors.v2ijka(h4,h6,h7,p2)  ) //P(p1/p2)
  //( i0(h4,h3,p1,p2)          += -1    * i_9(h3,h7,h6,p1)    * v2tensors.v2ijka(h4,h6,h7,p2)  ) //P(h3/h4)
  //( i0(h4,h3,p2,p1)          +=         i_9(h3,h7,h6,p1)    * v2tensors.v2ijka(h4,h6,h7,p2)  ) //P(p1/p2,h3/h3)
    ( tmp_13(h3,h4,p1,p2)       =         i_9(h3,h7,h6,p1)    * v2tensors.v2ijka(h4,h6,h7,p2)  )     
    ( i0(h3,h4,p1,p2)          +=         tmp_13(h3,h4,p1,p2)                                  )
    ( i0(h3,h4,p2,p1)          += -1    * tmp_13(h3,h4,p1,p2)                                  )
    ( i0(h4,h3,p1,p2)          += -1    * tmp_13(h3,h4,p1,p2)                                  )
    ( i0(h4,h3,p2,p1)          +=         tmp_13(h3,h4,p1,p2)                                  )
    (   i_10(h3,h4,h6,p7)       = -1    * t1(p5,h6)           * y2(h3,h4,p5,p7)                )
  //( i0(h3,h4,p1,p2)          +=         i_10(h3,h4,h6,p7)   * v2tensors.v2iabc(h6,p7,p1,p2)  )
  //( tmp_11(h3,h4,h6,p2,cind)  =         i_10(h3,h4,h6,p7)   * cv3d(p7,p2,cind )              )
  //( i0(h3,h4,p1,p2)          +=         tmp_11(h3,h4,h6,p2,cind) * cv3d(h6,p1,cind)          )
  //( tmp_12(h3,h4,h6,p1,cind)  =         i_10(h3,h4,h6,p7)   * cv3d(p7,p1,cind )              )
  //( i0(h3,h4,p1,p2)          += -1.0  * tmp_12(h3,h4,h6,p1,cind) * cv3d(h6,p2,cind)          )
    ( tmp_11(h6,p7,p1,p2)       =         cv3d(h6,p1,cind)    * cv3d(p7,p2,cind)               )
    ( tmp_11(h6,p7,p1,p2)      += -1.0  * cv3d(h6,p2,cind)    * cv3d(p7,p1,cind)               )
    ( i0(h3,h4,p1,p2)          +=         i_10(h3,h4,h6,p7)   * tmp_11(h6,p7,p1,p2)            )
    (   i_11(p6,p1)             =         t2(p5,p6,h7,h8)     * y2(h7,h8,p1,p5)                )
    ( i0(h3,h4,p1,p2)          += -0.5  * i_11(p6,p1)         * v2tensors.v2ijab(h3,h4,p2,p6)  )
    ( i0(h3,h4,p2,p1)          +=  0.5  * i_11(p6,p1)         * v2tensors.v2ijab(h3,h4,p2,p6)  ) //P(p1/p2)
    (   i_12(h3,h4,h8,h9)       =         t2(p5,p6,h8,h9)     * y2(h3,h4,p5,p6)                )
    (     i_12_1(h3,h4,h8,p5)   = -1    * t1(p7,h8)           * y2(h3,h4,p5,p7)                )
    (   i_12(h3,h4,h8,h9)      +=  2    * t1(p5,h9)           * i_12_1(h3,h4,h8,p5)            )
    ( i0(h3,h4,p1,p2)          +=  0.25 * i_12(h3,h4,h8,h9)   * v2tensors.v2ijab(h8,h9,p1,p2)  )
    (     i_13_1(h3,h6,h8,p1)   =         t1(p7,h8)           * y2(h3,h6,p1,p7)                )
    (   i_13(h3,p5,h8,p1)       =         t1(p5,h6)           * i_13_1(h3,h6,h8,p1)            )
  //( i0(h3,h4,p1,p2)          += -1    * i_13(h3,p5,h8,p1)   * v2tensors.v2ijab(h4,h8,p2,p5)  )
  //( i0(h3,h4,p2,p1)          +=         i_13(h3,p5,h8,p1)   * v2tensors.v2ijab(h4,h8,p2,p5)  ) //P(p1/p2)
  //( i0(h4,h3,p1,p2)          +=         i_13(h3,p5,h8,p1)   * v2tensors.v2ijab(h4,h8,p2,p5)  ) //P(h3/h4)
  //( i0(h4,h3,p2,p1)          += -1    * i_13(h3,p5,h8,p1)   * v2tensors.v2ijab(h4,h8,p2,p5)  ) //P(p1/p2,h3/h4)
    ( tmp_12(h3,h4,p1,p2)       =         i_13(h3,p5,h8,p1)   * v2tensors.v2ijab(h4,h8,p2,p5)  )
    ( i0(h3,h4,p1,p2)          += -1    * tmp_12 (h3,h4,p1,p2)                                 )
    ( i0(h3,h4,p2,p1)          +=         tmp_12 (h3,h4,p1,p2)                                 ) 
    ( i0(h4,h3,p1,p2)          +=         tmp_12 (h3,h4,p1,p2)                                 ) 
    ( i0(h4,h3,p2,p1)          += -1    * tmp_12 (h3,h4,p1,p2)                                 );
  // clang-format on
}

template<typename T>
std::tuple<double, double> exachem::cc::ccsd_lambda::lambda_ccsd_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, cholesky_2e::V2Tensors<T>& v2tensors,
  Tensor<T>& cv3d, Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_y1, Tensor<T>& d_y2,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_y1s,
  std::vector<Tensor<T>>& d_y2s, std::vector<T>& p_evl_sorted) {
  // TODO: LAMBDA DOES NOT HAVE THE SAME ITERATION CONVERGENCE
  //       PROTOCOL AND NEEDS TO BE UPDATED.

  double      zshiftl  = 0.0;
  SystemData& sys_data = chem_env.sys_data;
  int         maxiter  = chem_env.ioptions.ccsd_options.ccsd_maxiter;
  int         ndiis    = chem_env.ioptions.ccsd_options.ndiis;
  double      thresh   = chem_env.ioptions.ccsd_options.threshold;
  bool        profile  = chem_env.ioptions.ccsd_options.profile_ccsd;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);
  std::cout.precision(15);

  // auto [cind] = CI.labels<1>("all");

  double residual = 0.0;
  double energy   = 0.0;

  Scheduler sch{ec};

  Y1Tensors<T> y1tensors;
  Y2Tensors<T> y2tensors;
  y1tensors.allocate(ec, MO, CI);
  y2tensors.allocate(ec, MO, CI);

  Tensor<T> d_e{}, d_r1_residual{}, d_r2_residual{};
  Tensor<T>::allocate(&ec, d_e, d_r1_residual, d_r2_residual);

  for(int titer = 0; titer < maxiter; titer += ndiis) {
    for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
      const auto timer_start = std::chrono::high_resolution_clock::now();
      int        off         = iter - titer;

      sch((d_y1s[off])() = d_y1())((d_y2s[off])() = d_y2()).execute();

      lambda_ccsd_y1(sch, MO, CI, d_r1, d_t1, d_t2, d_y1, d_y2, d_f1, v2tensors, cv3d, y1tensors);
      lambda_ccsd_y2(sch, MO, CI, d_r2, d_t1, d_t2, d_y1, d_y2, d_f1, v2tensors, cv3d, y2tensors);

      sch.execute(sch.ec().exhw(), profile);

      std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_y1, d_y2, d_e, d_r1_residual,
                                        d_r2_residual, p_evl_sorted, zshiftl, n_occ_alpha,
                                        n_occ_beta, true);

      update_r2(ec, d_r2());

      sch((d_r1s[off])() = d_r1())((d_r2s[off])() = d_r2()).execute();

      const auto timer_end = std::chrono::high_resolution_clock::now();
      auto       iter_time =
        std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
          .count();

      iteration_print_lambda(chem_env, ec.pg(), iter, residual, iter_time);

      if(residual < thresh) { break; }
    }

    if(residual < thresh || titer + ndiis >= maxiter) { break; }
    if(ec.pg().rank() == 0) {
      std::cout << " MICROCYCLE DIIS UPDATE:";
      std::cout.width(21);
      std::cout << std::right << std::min(titer + ndiis, maxiter) + 1 << std::endl;
    }

    std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s};
    std::vector<std::vector<Tensor<T>>> ys{d_y1s, d_y2s};
    std::vector<Tensor<T>>              next_y{d_y1, d_y2};
    diis<T>(ec, rs, ys, next_y);
  }

  y1tensors.deallocate();
  y2tensors.deallocate();
  Tensor<T>::deallocate(d_e, d_r1_residual, d_r2_residual);

  return std::make_tuple(residual, energy);
} // End of lambda_ccsd_driver

using T = double;
template std::tuple<double, double> exachem::cc::ccsd_lambda::lambda_ccsd_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
  Tensor<T>& d_t1, Tensor<T>& d_t2, Tensor<T>& d_f1, cholesky_2e::V2Tensors<T>& v2tensors,
  Tensor<T>& cv3d, Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_y1, Tensor<T>& d_y2,
  std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_y1s,
  std::vector<Tensor<T>>& d_y2s, std::vector<T>& p_evl_sorted);

template std::tuple<Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, std::vector<Tensor<T>>,
                    std::vector<Tensor<T>>, std::vector<Tensor<T>>, std::vector<Tensor<T>>>
exachem::cc::ccsd_lambda::setupLambdaTensors(ExecutionContext& ec, TiledIndexSpace& MO,
                                             size_t ndiis);
