/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include <tamm/tamm.hpp>

using namespace tamm;

template<typename T>
Tensor<T> compute_1rdm(std::vector<int>& cc_rdm, std::string files_prefix, Scheduler& sch,
                       TiledIndexSpace& MO, Tensor<T> d_t1, Tensor<T> d_t2, Tensor<T> d_y1,
                       Tensor<T> d_y2) {
  auto [mo1, mo2]      = MO.labels<2>("all");
  auto [i, j, k, l, m] = MO.labels<5>("occ");
  auto [a, b, c, d, e] = MO.labels<5>("virt");

  auto ex_hw = sch.ec().exhw();
  auto rank  = sch.ec().pg().rank();

  // 1-RDM
  Tensor<T> gamma1{{mo1, mo2}, {1, 1}};
  Tensor<T> hh_1{{j, i}, {1, 1}};
  Tensor<T> pp_1{{a, d}, {1, 1}};

  sch.allocate(gamma1, hh_1, pp_1).execute();

  // D_a^i =  l_i^a
  // D_i^a =  t_a^i + l_m^e*t_ae^im
  //         -(l_m^e*t_e^i + 1/2*l_mn^ef*t_ef^in)=i1_m^i * t_a^m
  //         +(-1/2 * l_mn^ef*t_e^i)=i2_mn^if * t_af^mn
  // D_i^j = -l_j^e*t_e^i - 1/2*l_jm^ef*t_ef^im
  // D_a^b =  l_m^a*t_b^m + 1/2*l_mn^ae*t_be^mn
  // {a,b,e,f} -> {p1, p2, p3, p4}
  // {i,j,m,n} -> {h1, h2, h3, h4}

  // clang-format off
  sch(gamma1(i, a)      =        d_y1(i, a)                              )
     (gamma1(a, i)      =        d_t1(a, i)                              )
     (gamma1(a, i)     +=        d_y1(j, b)          * d_t2(b, a, j, i)  )
     (hh_1(j, i)        =        d_y1(j, b)          * d_t1(b, i)        )
     (gamma1(a, i)     += -1.0*  hh_1(j, i)          * d_t1(a, j)        )        
     (hh_1(l, i)        =  d_t2(c, d, k, i)          * d_y2(k, l, c, d)  )
     (gamma1(a, i)     += -0.5* hh_1(l, i)           * d_t1(a, l)        )
     (pp_1(a, d)        = d_y2(k, l, c, d)           * d_t2(c, a, k, l)  )
     (gamma1(a, i)     += -0.5*pp_1(a, d)            * d_t1(d, i)        )
     (gamma1(i, j)      = -1.0*d_t1(a, j)            * d_y1(i, a)        )
     (gamma1(a, b)      =  d_y1(i, b)                * d_t1(a, i)        ) 
     (gamma1(i, j)     += -0.5* d_y2(k, i, c, b)     * d_t2(c, b, k, j)  ) 
     (gamma1(a, b)     += 0.5 * d_y2(k, j, c, b)     * d_t2(c, a, k, j)  )  
     .execute(ex_hw);
  // clang-format on

  sch.deallocate(hh_1, pp_1).execute();
  if(!cc_rdm.empty()) {
    ExecutionContext ec_dense{sch.ec().pg(), DistributionKind::dense, MemoryManagerKind::ga};
    if(cc_rdm[0] == 1) {
      if(rank == 0)
        std::cout << "Printing 1-RDM to file: " << files_prefix << ".1_RDM.txt" << std::endl;
      Tensor<T> gamma1_dense = to_dense_tensor(ec_dense, gamma1);
      print_dense_tensor(gamma1_dense, files_prefix + ".1_RDM");
      Tensor<T>::deallocate(gamma1_dense);
    }
  }

  //--------------------------------------------------------------------------------------------------
  // cross checking of 1-RDM, by computing 1st order response property in terms of density matrices
  //----------------------------------------------------------------------------------------------------
  // Tensor<T> dipole_mx_new{}, dipole_my_new{}, dipole_mz_new{};
  // sch.allocate(dipole_mx_new, dipole_my_new, dipole_mz_new).execute();
  // sch(dipole_my_new() = gamma1(mo1, mo2)*DipY_mo(mo2, mo1))
  //       .execute(ex_hw);
  // auto dmy_new = get_scalar(dipole_my_new);
  // if(rank == 0){
  // std::cout << std::fixed << std::setprecision(8) << dmy_new << std::endl;
  // }

  return gamma1;
}

// 2-RDM
template<typename T>
Tensor<T> compute_2rdm(std::vector<int>& cc_rdm, std::string files_prefix, Scheduler& sch,
                       TiledIndexSpace& MO, Tensor<T> d_t1, Tensor<T> d_t2, Tensor<T> d_y1,
                       Tensor<T> d_y2) {
  auto [mo1, mo2, mo3, mo4] = MO.labels<4>("all");
  auto [i, j, k, l, m]      = MO.labels<5>("occ");
  auto [a, b, c, d, e]      = MO.labels<5>("virt");

  auto ex_hw = sch.ec().exhw();
  auto rank  = sch.ec().pg().rank();

  Tensor<T> gamma2{{mo1, mo2, mo3, mo4}, {2, 2}};
  Tensor<T> pphh{{a, b, i, j}, {2, 2}};
  Tensor<T> hhhp{{i, j, k, a}, {2, 2}};
  Tensor<T> pp{{a, b}, {1, 1}};
  Tensor<T> hh{{i, j}, {1, 1}};
  Tensor<T> ph{{a, i}, {1, 1}};
  Tensor<T> hpph{{i, a, b, j}, {2, 2}};
  Tensor<T> phhp{{a, i, j, b}, {2, 2}};
  Tensor<T> hhhh{{i, j, k, l}, {2, 2}};
  Tensor<T> hhpp{{i, j, a, b}, {2, 2}};
  Tensor<T> pppp{{a, b, c, d}, {2, 2}};
  Tensor<T> pphp{{a, b, i, c}, {2, 2}};
  Tensor<T> phpp{{a, i, b, c}, {2, 2}};
  Tensor<T> hhph{{i, j, a, k}, {2, 2}};

  sch.allocate(gamma2, pphh, hhhp, pp, hh, ph, hpph, phhp, hhhh, hhpp, pppp, pphp, phpp, hhph)
    .execute();

  // clang-format off
  sch (gamma2(a, b, c, d)        = 0.5 * d_y2(i, j, c, d)     * d_t2(a, b, i, j))  //1 
      (gamma2(i, j, k, l)       += 0.5 * d_y2(i, j, c, d)     * d_t2(c, d, k, l))  //2
      (pphh(a, b, i, j)          = d_t1(a, i)                 * d_t1(b, j)      )  //3
      (gamma2(a, b, c, d)       += d_y2(i, j, c, d)           * pphh(a, b, i, j))
      (hhhp(i, j, k, d)          = d_y2(i, j, c, d)           * d_t1(c, k)      )  //4
      (gamma2(i, j, k, l)       += hhhp(i, j, k, d)           * d_t1(d, l)      )
      (gamma2(b, c, a, k)       += d_y1(l, a)                 * d_t2(b, c, l, k))  //5
      (gamma2(i, c, j, k)       += -1.0*d_y1(i, b)            * d_t2(b, c, j, k))  //6
      (pp(a, c)                  = d_y1(k, c)                 * d_t1(a, k)      )  //7
      (gamma2(a, b, c, i)       += pp(a, c)                   * d_t1(b, i)      )
      //p(a/b)
      (pp(b, c)                  = d_y1(k, c)                 * d_t1(b, k)      )
      (gamma2(a, b, c, i)       += -1.0* pp(b, c)             * d_t1(a, i)      )
      //8
      (hh(i, j)                  = d_y1(i, a)                 * d_t1(a, j)      )
      (gamma2(i, b, j, k)       += -1.0* hh(i, j)             * d_t1(b, k)      )
      //p(jk)
      (hh(i, k)                  = d_y1(i, a)                 * d_t1(a, k)      )
      (gamma2(i, b, j, k)       += hh(i, k)                   * d_t1(b, j)      )
      //9
      (pp(a, b)                  =  d_y2(i, k, d, b)          * d_t2(d, a, i, k))
      (gamma2(a, c, b, j)       += 0.5*pp(a, b)               * d_t1(c, j)      )
      //p(ac)
      (pp(c, b)                  = d_y2(i, k, d, b)           * d_t2(d, c, i, k))
      (gamma2(a, c, b, j)       +=-0.5* pp(c, b)              * d_t1(a, j)      )
      //10
      (hh(k, i)                  = d_t2(d, a, l, i)           * d_y2(l, k, d, a))
      (gamma2(k, b, i, j)       +=-0.5* hh(k, i)              * d_t1(b, j)      ) 
      //p(ij)      
      (hh(k, j)                  = d_t2(d, a, l, j)           * d_y2(l, k, d, a))
      (gamma2(k, b, i, j)       += 0.5*hh(k, j)               * d_t1(b, i)      )
      //11
      (hpph(k, b, c, j)         = d_t2(b, d, j, l)            * d_y2(k, l, c, d))
      (gamma2(a, b, c, j)      += hpph(k, b, c, j)            * d_t1(a, k)      )
      // p(ab)      
      (hpph(k, a, c, j)         = d_t2(a, d, j, l)            * d_y2(k, l, c, d))
      (gamma2(a, b, c, j)      += -1.0*hpph(k, a, c, j)       * d_t1(b, k)      )
      //12
      (hpph(k, b, c, j)         = d_t2(b, d, j, l)            * d_y2(k, l, c, d))
      (gamma2(k, b, i, j)      +=-1.0*hpph(k, b, c, j)        * d_t1(c, i)      )
      //! p(ij)
      (hpph(k, b, c, i)         = d_t2(b, d, i, l)            * d_y2(k, l, c, d))
      (gamma2(k, b, i, j)      += hpph(k, b, c, i)            * d_t1(c, j)      )
      //13
      (hhph(k, l, c, j)         = d_y2(k, l, c, d)            * d_t1(d, j)      )
      (gamma2(a, b, c, j)      += -0.5*hhph(k, l, c, j)       * d_t2(a, b, k, l))
      //14
      (hhhh(k, l, i, j)         = d_t2(c, d, i, j)            * d_y2(k, l, c, d))
      (gamma2(k, b, i, j)      += 0.5*hhhh(k, l, i, j)        * d_t1(b, l)      )           
      //15
      (hhph(i, j, a, l)         = d_y2(i, j, a, d)            * d_t1(d, l)      )
      (hpph(i, c, a, l)         = hhph(i, j, a, l)            * d_t1(c, j)      )
      (gamma2(b, c, a, l)      += -1.0*hpph(i, c, a, l)       * d_t1(b, i)      )
      //16
      (hhph(i, k, a, l)          = d_y2(i, k, a, d)           * d_t1(d, l)      )
      (hpph(i, c, a, l)          = hhph(i, k, a, l)           * d_t1(c, k)      )
      (gamma2(i, c, j, l)       += hpph(i, c, a, l)           * d_t1(a, j)      )
      //17
      (gamma2(b, j, a, c)       += d_y2(i, j, a, c)           * d_t1(b, i)      )
      //18
      (gamma2(i, k, j, c)       += -1.0 * d_y2(i, k, a, c)    * d_t1(a, j)      )
      //19
      (gamma2(a, b, i, j)       += d_t2(a, b, i, j)                             )
      //20 
      (gamma2(a, b, i, j)   += 0.5 * d_t1(a, i)               * d_t1(b, j)      )
      //! p(ij)
      (gamma2(a, b, i, j)     +=-0.5 * d_t1(a, j)             * d_t1(b, i)      )
      //! P(ab)
      (gamma2(a, b, i, j)     +=-0.5 * d_t1(b, i)             * d_t1(a, j)      )
      //! p(ij) & p(ab)
      (gamma2(a, b, i, j)     += 0.5 * d_t1(b, j)             * d_t1(a, i)      )
      //21
      (ph(a, i)               = d_y1(k, c)                    * d_t2(c, a, k, i))
      (gamma2(a, b, i, j)    += ph(a, i)                      * d_t1(b, j)      )
      //! p(ij)
      (ph(a, j)                = d_y1(k, c)                   * d_t2(c, a, k, j))
      (gamma2(a, b, i, j)     +=-1.0*ph(a, j)                 * d_t1(b, i)      )
      //! p(ab)
      (ph(b, i)                = d_y1(k, c)                   * d_t2(c, b, k, i))
      (gamma2(a, b, i, j)     +=-1.0*ph(b, i)                 * d_t1(a, j)      )
      //!p(ij) & p(ab)
      (ph(b, j)                = d_y1(k, c)                   * d_t2(c, b, k, j))
      (gamma2(a, b, i, j)     += ph(b, j)                     * d_t1(a, i)      )
      //22
      (hh(k, i)                = d_t1(c, i)                   * d_y1(k, c)      )
      (gamma2(a, b, i, j)     += -1.0*hh(k, i)                * d_t2(a, b, k, j))
      // ! p(ij)
      (hh(k, j)              = d_t1(c, j)                    * d_y1(k, c)      )
      (gamma2(a, b, i, j)   += hh(k, j)                      * d_t2(a, b, k, i))
      //23
      (pp(a, c)              = d_t1(a, k)                     * d_y1(k, c)      )
      (gamma2(a, b, i, j)    += -1.0*pp(a, c)                 * d_t2(c, b, i, j))
      //! p(ab)
      (pp(b, c)               = d_t1(b, k)                    * d_y1(k, c)      )
      (gamma2(a, b, i, j)    += pp(b, c)                      * d_t2(c, a, i, j))
      //24
      (pp(a, c)               = d_y1(k, c)                    * d_t1(a, k)      )
      (ph(a, i)               = pp(a, c)                      * d_t1(c, i)      )
      (gamma2(a, b, i, j)    +=-1.0*ph(a, i)                  * d_t1(b, j)      )
      //! p(ij)
      (pp(a, c)               = d_y1(k, c)                    * d_t1(a, k)      )
      (ph(a, j)               = pp(a, c)                      * d_t1(c, j)      )
      (gamma2(a, b, i, j)    += ph(a, j)                      * d_t1(b, i)      )
      //! p(ab)
      (pp(b, c)               = d_y1(k, c)                    * d_t1(b, k)      )
      (ph(b, i)               = pp(b, c)                      * d_t1(c, i)      )
      (gamma2(a, b, i, j)    += ph(b, i)                      * d_t1(a, j)      )
      //!  p(ij|ab)
      (pp(b, c)               = d_y1(k, c)                    * d_t1(b, k)      )
      (ph(b, j)               = pp(b, c)                      * d_t1(c, j)      )
      (gamma2(a, b, i, j)    +=-1.0*ph(b, j)                  * d_t1(a, i)      )
      //25
      (phhp(a, l, i, d)       = d_y2(k, l, c, d)              * d_t2(c, a, k, i))
      (gamma2(a, b, i, j)    += 0.5*phhp(a, l, i, d)          * d_t2(b, d, j, l))
      //! p(ij)
      (phhp(a, l, j, d)       = d_y2(k, l, c, d)              * d_t2(c, a, k, j))
      (gamma2(a, b, i, j)    += -0.5*phhp(a, l, j, d)         * d_t2(b, d, i, l))
      //! p(ab)
      (phhp(b, l, i, d)       = d_y2(k, l, c, d)              * d_t2(c, b, k, i))
      (gamma2(a, b, i, j)    += -0.5*phhp(b, l, i, d)         * d_t2(a, d, j, l))
      // ! P(ij)  & p(ab)
      (phhp(b, l, j, d)       = d_y2(k, l, c, d)              * d_t2(c, b, k, j))
      (gamma2(a, b, i, j)    += 0.5*phhp(b, l, j, d)          * d_t2(a, d, i, l))
      //26
      (hh(l, i)               = d_y2(k, l, c, d)              * d_t2(c, d, k, i))
      (gamma2(a, b, i, j)    +=-0.5*hh(l, i)                  * d_t2(a, b, l, j))
      //! p(ij)
      (hh(l, j)               = d_y2(k, l, c, d)              * d_t2(c, d, k, j))
      (gamma2(a, b, i, j)    += 0.5*hh(l, j)                  * d_t2(a, b, l, i))
      //27
      (pp(a, d)               = d_y2(k, l, c, d)              * d_t2(c, a, k, l))
      (gamma2(a, b, i, j)    +=-0.5*pp(a, d)                  * d_t2(d, b, i, j))
      //! p(ab)
      (pp(b, d)               = d_y2(k, l, c, d)              * d_t2(c, b, k, l))
      (gamma2(a, b, i, j)    += 0.5*pp(b, d)                  * d_t2(d, a, i, j))
      //28
      (hhhh(l, k, i, j)       = d_y2(l, k, d, c)              * d_t2(d, c, i, j))
      (gamma2(a, b, i, j)    += 0.25*hhhh(l, k, i, j)         * d_t2(a, b, l, k))
      //29
      (hh(l, i)               = d_t2(c, d, k, i)              * d_y2(k, l, c, d))
      (ph(a, i)               = hh(l, i)                      * d_t1(a, l))
      (gamma2(a, b, i, j)    +=-0.5*ph(a, i)                  * d_t1(b, j))
      //! p(ij)
      (hh(l, j)               = d_t2(c, d, k, j)              * d_y2(k, l, c, d))
      (ph(a, j)               = hh(l, j)                      * d_t1(a, l))
      (gamma2(a, b, i, j)    += 0.5*ph(a, j)                  * d_t1(b, i))
      //! p(ab)
      (hh(l, i)               = d_t2(c, d, k, i)              * d_y2(k, l, c, d))
      (ph(b, i)               = hh(l, i)                      * d_t1(b, l))
      (gamma2(a, b, i, j)    += 0.5*ph(b, i)                  * d_t1(a, j))
      //! p(ij) & p(ab)
      (hh(l, j)                = d_t2(c, d, k, j)             * d_y2(k, l, c, d))
      (ph(b, j)                = hh(l, j)                     * d_t1(b, l)      )
      (gamma2(a, b, i, j)     += -0.5*ph(b, j)                * d_t1(a, i)      )
      //30
      (pp(a, d)                = d_y2(k, l, c, d)             * d_t2(c, a, k, l))
      (ph(a, i)                = pp(a, d)                     * d_t1(d, i)      )
      (gamma2(a, b, i, j)     += -0.5*ph(a, i)                * d_t1(b, j)      )
      //! p(ij)
      (pp(a, d)                = d_y2(k, l, c, d)             * d_t2(c, a, k, l))
      (ph(a, j)                = pp(a, d)                     * d_t1(d, j)      )
      (gamma2(a, b, i, j)     += 0.5*ph(a, j)                 * d_t1(b, i)      )
      //! p(ab)
      (pp(b, d)                = d_y2(k, l, c, d)             * d_t2(c, b, k, l))
      (ph(b, i)                = pp(b, d)                     * d_t1(d, i)      )
      (gamma2(a, b, i, j)     += 0.5*ph(b, i)                 * d_t1(a, j)      )
      //! p(ij) & p(ab)
      (pp(b, d)                = d_y2(k, l, c, d)             * d_t2(c, b, k, l))
      (ph(b, j)                = pp(b, d)                     * d_t1(d, j)      )
      (gamma2(a, b, i, j)     +=-0.5*ph(b, j)                 * d_t1(a, i)      )
      //31
      (hhhp(l, k, i, c)        = d_y2(l, k, d, c)             * d_t1(d, i)      )
      (hhpp(i, k, a, c)        = hhhp(l, k, i, c)             * d_t1(a, l)      )
      (gamma2(a, b, i, j)     += -1.0*hhpp(i, k, a, c)        * d_t2(b, c, j, k))
      // ! p(ij)
      (hhhp(l, k, j, c)        = d_y2(l, k, d, c)             * d_t1(d, j)      )
      (hhpp(j, k, a, c)        = hhhp(l, k, j, c)             * d_t1(a, l)      )
      (gamma2(a, b, i, j)     += hhpp(j, k, a, c)             * d_t2(b, c, i, k))
      // ! p(ab)
      (hhhp(l, k, i, c)        = d_y2(l, k, d, c)             * d_t1(d, i)      )
      (hhpp(i, k, b, c)        = hhhp(l, k, i, c)             * d_t1(b, l)      )
      (gamma2(a, b, i, j)     += hhpp(i, k, b, c)             * d_t2(a, c, j, k))
      // ! p(ij) & p(ab)
      (hhhp(l, k, j, c)        = d_y2(l, k, d, c)             * d_t1(d, j)      )
      (hhpp(j, k, b, c)        = hhhp(l, k, j, c)             * d_t1(b, l)      )
      (gamma2(a, b, i, j)     += -1.0*hhpp(j, k, b, c)        * d_t2(a, c, i, k))
      //32
      (hhhp(k, l, i, d)        = d_y2(k, l, c, d)             * d_t1(c, i)      )
      (hhhh(k, l, i, j)        = hhhp(k, l, i, d)             * d_t1(d, j)      )
      (gamma2(a, b, i, j)     += 0.5*hhhh(k, l, i, j)         * d_t2(a, b, k, l))
      //33
      (phpp(a, l, c, d)        = d_y2(k, l, c, d)             * d_t1(a, k)      )
      (pppp(a, b, c, d)        = phpp(a, l, c, d)             * d_t1(b, l)      )
      (gamma2(a, b, i, j)     += 0.5*pppp(a, b, c, d)         * d_t2(c, d, i, j))
      //34
      (hhhp(k, l, i, d)        = d_y2(k, l, c, d)             * d_t1(c, i)      )
      (phhp(a, l, i, d)        = hhhp(k, l, i, d)             * d_t1(a, k)      )
      (pphp(a, b, i, d)        = phhp(a, l, i, d)             * d_t1(b, l)      )
      (gamma2(a, b, i, j)     += pphp(a, b, i, d)             * d_t1(d, j)      )
      //35
      (gamma2(i, b, a, j)     += d_y1(i, a)                   * d_t1(b, j)      )
      //36
      (gamma2(i, b, a, j)     += d_y2(k, i, c, a)             * d_t2(c, b, k, j))
      //37
      (pphh(c, b, k, j)        =       d_t1(c, k)             * d_t1(b, j)      )
      (gamma2(i, c, a, j)     +=-1.0 * pphh(c, b, k, j)       * d_y2(i, k, a, b))
      //38
      (gamma2(i, j, a, b)     +=      d_y2(i, j, a, b)                          )

      .execute(ex_hw);
  // clang-format on

  if(!cc_rdm.empty()) {
    ExecutionContext ec_dense{sch.ec().pg(), DistributionKind::dense, MemoryManagerKind::ga};
    auto             rdm_val = (cc_rdm.size() == 2) ? cc_rdm[1] : cc_rdm[0];
    if(rdm_val == 2) {
      if(rank == 0)
        std::cout << "Printing 2-RDM to file: " << files_prefix << ".2_RDM.txt" << std::endl;
      Tensor<T> gamma2_dense = to_dense_tensor(ec_dense, gamma2);
      print_dense_tensor(gamma2_dense, files_prefix + ".2_RDM");
      Tensor<T>::deallocate(gamma2_dense);
    }
  }

  sch.deallocate(pphh, hhhp, pp, hh, ph, hpph, phhp, hhhh, hhpp, pppp, pphp, phpp, hhph).execute();

  return gamma2;
}

using T = double;
template Tensor<T> compute_1rdm<T>(std::vector<int>& cc_rdm, std::string files_prefix,
                                   Scheduler& sch, TiledIndexSpace& MO, Tensor<T> d_t1,
                                   Tensor<T> d_t2, Tensor<T> d_y1, Tensor<T> d_y2);

template Tensor<T> compute_2rdm<T>(std::vector<int>& cc_rdm, std::string files_prefix,
                                   Scheduler& sch, TiledIndexSpace& MO, Tensor<T> d_t1,
                                   Tensor<T> d_t2, Tensor<T> d_y1, Tensor<T> d_y2);
