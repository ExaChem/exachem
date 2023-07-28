/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023 NWChemEx-Project.
 * Copyright 2023 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "cc/ccsd/ccsd_util.hpp"

using namespace tamm;

namespace cc2_canonical {

template<typename T>
void cc2_e(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& de, const Tensor<T>& t1,
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
void cc2_t1(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
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
void cc2_t2(Scheduler& sch, const TiledIndexSpace& MO, Tensor<T>& i0, const Tensor<T>& t1,
            Tensor<T>& t2, const Tensor<T>& f1, const Tensor<T>& v2) {
  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  // std::vector<SpinPosition> {1,1}{SpinPosition::upper,
  //                                        SpinPosition::lower};

  // std::vector<SpinPosition> {2,2}{SpinPosition::upper,SpinPosition::upper,
  //                                        SpinPosition::lower,SpinPosition::lower};

  Tensor<T>       pp{{V, V}, {1, 1}};
  Tensor<T>       hh{{O, O}, {1, 1}};
  Tensor<T>       pphh{{V, V, O, O}, {2, 2}};
  Tensor<T>       phhh{{V, O, O, O}, {2, 2}};
  Tensor<T>       pphp{{V, V, O, V}, {2, 2}};
  Tensor<T>       hphp{{O, V, O, V}, {2, 2}};
  Tensor<T>       hphh{{O, V, O, O}, {2, 2}};
  Tensor<T>       hhhh{{O, O, O, O}, {2, 2}};
  Tensor<T>       hhhp{{O, O, O, V}, {2, 2}};
  TiledIndexLabel a, b, c, d, e, f;
  TiledIndexLabel i, j, k, l, m, n;

  std::tie(a, b, c, d, e, f) = MO.labels<6>("virt");
  std::tie(i, j, k, l, m, n) = MO.labels<6>("occ");

  // clang-format off
  sch.allocate(pp, hh, pphh, phhh, pphp, hphp, hphh, hhhh, hhhp)
  // V_(a, b, i, j)     we follow <left out,right out || left in, right in> notation 
  (i0(a, b, i, j)            =           v2(a, b, i, j))        
  (pphh(a, b, i, j)          =           f1(b, c)               *      t2(a, c, i, j)) 
  // +p(a/b)[f(b, c)+f(k, c)*t1(b, k)]*t2(a, c, i, j)
  (i0(a, b, i,  j)          +=           pphh(a, b, i, j)) 
  (i0(a, b, i,  j)          +=-1.0*      pphh(b, a, i, j)) 
  (pp(b, c)                  =           f1(k, c)               *      t1(b, k))
  (pphh(a, b, i, j)          =           pp(b, c)               *      t2(a, c, i, j))
  (i0(a, b, i, j)           += -1.0*     pphh(a, b, i, j))
  (i0(a, b, i, j)           +=  1.0*     pphh(b, a, i, j))                                   // p(a/b)  
  (pphh(a, b, i, j)          =           f1(k, j)               *      t2(a, b, i, k)) 
  (i0(a, b, i, j)           +=  -1.0*    pphh(a, b, i, j))  
  // -p(i/j)[f(k, j)+ f(k, c)*t1(c, j)]*t2(a, b, i, k)
  (i0(a, b, i, j)           +=   1.0*    pphh(a, b, j, i)) 
  (hh(k, j)                  =           f1(k, c)               *      t1(c, j))
  (pphh(a, b, i, j)          =           hh(k, j)               *      t2(a, b, i, k)) 
  (i0(a, b, i, j)           += -1.0                             *      pphh(a, b, i, j))
  (i0(a, b, i, j)           +=           pphh(a, b, j, i)) 
  (pphh(a, b, i, j)          =           v2(a, b, i, c)         *      t1(c, j))   
  (i0(a, b, i, j)           +=           pphh(a, b, i, j)                      )             // p(i/j)v2(a, b, i, c)*t1(c, j))
  (i0(a, b, i, j)           += -1.0*     pphh(a, b, j, i)                      )
  (pphh(a, b, i, j)          =           v2(a, k, i, j)         *      t1(b, k))  
  (i0(a, b, i, j)           += -1.0*     pphh(a, b, i, j)                      )             // v2(a, k, i, j)&t1(b, k))
  (i0(a, b, i, j)           +=           pphh(b, a, i, j)                      )             // p(a/b)
  (phhh(a, l, i, j)          =           v2(k, l, i, j)         *      t1(a, k))             // v2(k, l, i, j)*t1(a, k)*t1(b, l)
  (i0(a, b, i, j)           +=           phhh(a, l, i, j)       *      t1(b, l))
  (pphp(a, b, i, d)          =           v2(a, b, c, d)         *      t1(c, i))  
  // v(a, b, c, d)*t1(c, i)*t1(d, j)
  (pphh(a, b, i, j)          =           pphp(a, b, i, d)       *      t1(d, j))             
  (i0(a, b, i, j)           +=           pphh(a, b, i, j))
  // -p(ij/ab)  v2(a, k, i, c)*t1(c, j)*t1(b, k)
  (phhh(a, k, i, j)          =           v2(a, k, i, c)         *      t1(c, j))            
  (pphh(a, b, i, j)          =           phhh(a, k, i, j)       *      t1(b, k))
  (i0(a, b, i, j)           +=  -1.0*    pphh(a, b, i, j))
  (i0(a, b, i, j)           +=   1.0*    pphh(a, b, j, i))                                 // p(ij)
  (i0(a, b, i, j)           +=   1.0*    pphh(b, a, i, j))                                 // p(ab)
  (i0(a, b, i, j)           +=  -1.0*    pphh(b, a, j, i))                                 // p(ij/ab)
  //-p(ab)v2(k, a, c, d)*t1(c, j)*t1(b, k)*t1(d, i)
  (hphp(k, a, j, d)          =           v2(k, a, c, d)         *      t1(c, j))          
  (hphh(k, a, j, i)          =           hphp(k, a, j, d)       *      t1(d, i))
  (pphh(a, b, i, j)          =           hphh(k, a, j, i)       *      t1(b, k))
  (i0(a, b, i, j)           += -1.0*     pphh(a, b, i, j))
  (i0(a, b, i, j)           +=  1.0*     pphh(b, a, i, j))
  // p(ij)v2(k, l, c, j)*t1(c, i)*t1(a, k)*t1(b, l)
  (hhhh(k, l, i, j)          =           v2(k, l, c, j)          *      t1(c, i))          
  (phhh(a, l, i, j)          =           hhhh(k, l, i, j)        *      t1(a, k))
  (pphh(a, b, i, j)          =           phhh(a, l, i, j)        *      t1(b, l))
  (i0(a, b, i, j)           +=           pphh(a, b, i, j))
  (i0(a, b, i, j)           += -1.0*     pphh(a, b, j, i))
  // v2(k, l, c, d)*t1(c, i)*t1(d, j)*t1(a, k)*t1(b, l)
  (hhhp(k, l, i, d)          =           v2(k, l, c, d)          *      t1(c, i))         
  (hhhh(k, l, i, j)          =           hhhp(k, l, i, d)        *      t1(d, j))
  (phhh(a, l, i, j)          =           hhhh(k, l, i, j)        *      t1(a, k))
  (i0(a, b, i, j)           +=           phhh(a, l, i, j)        *      t1(b, l))   
  .deallocate(pp, hh, pphh, phhh, pphp, hphp, hphh, hhhh, hhhp);
  // sch.execute();
  // clang-format on
}

template<typename T>
std::tuple<double, double>
cc2_v2_driver(SystemData sys_data, ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_t1,
              Tensor<T>& d_t2, Tensor<T>& d_f1, Tensor<T>& d_v2, Tensor<T>& d_r1, Tensor<T>& d_r2,
              std::vector<Tensor<T>>& d_r1s, std::vector<Tensor<T>>& d_r2s,
              std::vector<Tensor<T>>& d_t1s, std::vector<Tensor<T>>& d_t2s,
              std::vector<T>& p_evl_sorted, bool ccsd_restart = false, std::string out_fp = "") {
  int    maxiter     = sys_data.options_map.ccsd_options.ccsd_maxiter;
  int    ndiis       = sys_data.options_map.ccsd_options.ndiis;
  double thresh      = sys_data.options_map.ccsd_options.threshold;
  bool   writet      = sys_data.options_map.ccsd_options.writet;
  int    writet_iter = sys_data.options_map.ccsd_options.writet_iter;
  double zshiftl     = sys_data.options_map.ccsd_options.lshift;
  bool   profile     = sys_data.options_map.ccsd_options.profile_ccsd;
  double residual    = 0.0;
  double energy      = 0.0;
  int    niter       = 0;

  const TAMM_SIZE n_occ_alpha = static_cast<TAMM_SIZE>(sys_data.n_occ_alpha);
  const TAMM_SIZE n_occ_beta  = static_cast<TAMM_SIZE>(sys_data.n_occ_beta);

  std::string t1file = out_fp + ".t1amp";
  std::string t2file = out_fp + ".t2amp";

  std::cout.precision(15);

  Tensor<T> d_e{};
  Tensor<T>::allocate(&ec, d_e);
  Scheduler sch{ec};

  if(!ccsd_restart) {
    Tensor<T> d_r1_residual{}, d_r2_residual{};
    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;

        sch((d_t1s[off])() = d_t1())((d_t2s[off])() = d_t2()).execute();

        cc2_canonical::cc2_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);
        cc2_canonical::cc2_t1(sch, MO, d_r1, d_t1, d_t2, d_f1, d_v2);
        cc2_canonical::cc2_t2(sch, MO, d_r2, d_t1, d_t2, d_f1, d_v2);

        sch.execute(ec.exhw(), profile);

        std::tie(residual, energy) = rest(ec, MO, d_r1, d_r2, d_t1, d_t2, d_e, d_r1_residual,
                                          d_r2_residual, p_evl_sorted, zshiftl, n_occ_alpha,
                                          n_occ_beta);

        update_r2(ec, d_r2());

        sch((d_r1s[off])() = d_r1())((d_r2s[off])() = d_r2()).execute();

        const auto timer_end = std::chrono::high_resolution_clock::now();
        auto       iter_time =
          std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
            .count();

        iteration_print(sys_data, ec.pg(), iter, residual, energy, iter_time);

        if(writet && (((iter + 1) % writet_iter == 0) || (residual < thresh))) {
          write_to_disk(d_t1, t1file);
          write_to_disk(d_t2, t2file);
        }

        if(residual < thresh) { break; }
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
    Tensor<T>::deallocate(d_r1_residual, d_r2_residual);

  } // no restart
  else {
    cc2_canonical::cc2_e(sch, MO, d_e, d_t1, d_t2, d_f1, d_v2);

    sch.execute(ec.exhw(), profile);

    energy   = get_scalar(d_e);
    residual = 0.0;
  }

  sys_data.ccsd_corr_energy = energy;

  if(ec.pg().rank() == 0) {
    sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
    sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
    sys_data.results["output"]["CCSD"]["final_energy"]["total"] = sys_data.scf_energy + energy;

    write_json_data(sys_data, "CCSD");
  }

  return std::make_tuple(residual, energy);
}

}; // namespace cc2_canonical

void cc2_canonical_driver(std::string filename, OptionsMap options_map) {
  using T = double;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
  auto             rank = ec.pg().rank();

  auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt,
        AO_tis, scf_conv] = hartree_fock_driver<T>(ec, filename, options_map);

  CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;
  auto         debug        = ccsd_options.debug;
  if(rank == 0) ccsd_options.print();

  if(rank == 0)
    cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

  auto [MO, total_orbitals] = setupMOIS(sys_data);

  std::string out_fp    = sys_data.output_file_prefix + "." + ccsd_options.basis;
  std::string files_dir = out_fp + "_files/" + sys_data.options_map.scf_options.scf_type + "/cc2";
  std::string files_prefix = /*out_fp;*/ files_dir + "/" + out_fp;
  std::string f1file       = files_prefix + ".f1_mo";
  std::string t1file       = files_prefix + ".t1amp";
  std::string t2file       = files_prefix + ".t2amp";
  std::string v2file       = files_prefix + ".cholv2";
  std::string cholfile     = files_prefix + ".cholcount";
  std::string ccsdstatus   = files_prefix + ".ccsdstatus";

  bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                              fs::exists(f1file) && fs::exists(v2file)));

  // deallocates F_AO, C_AO
  auto [cholVpr, d_f1, lcao, chol_count, max_cvecs, CI] =
    cd_svd_driver<T>(sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells,
                     shell_tile_map, ccsd_restart, cholfile);
  free_tensors(lcao);

  if(ccsd_options.writev) ccsd_options.writet = true;

  TiledIndexSpace N = MO("all");

  std::vector<T>         p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_t1, d_t2;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

  std::tie(p_evl_sorted, d_t1, d_t2, d_r1, d_r2, d_r1s, d_r2s, d_t1s, d_t2s) = setupTensors(
    ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

  if(ccsd_restart) {
    read_from_disk(d_f1, f1file);
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
    }
    read_from_disk(cholVpr, v2file);
    ec.pg().barrier();
    p_evl_sorted = tamm::diagonal(d_f1);
  }

  else if(ccsd_options.writet) {
    // fs::remove_all(files_dir);
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);

    write_to_disk(d_f1, f1file);
    write_to_disk(cholVpr, v2file);

    if(rank == 0) {
      std::ofstream out(cholfile, std::ios::out);
      if(!out) cerr << "Error opening file " << cholfile << endl;
      out << chol_count << std::endl;
      out.close();
    }
  }

  if(rank == 0 && debug) {
    print_vector(p_evl_sorted, files_prefix + ".eigen_values.txt");
    cout << "Eigen values written to file: " << files_prefix + ".eigen_values.txt" << endl << endl;
  }

  ec.pg().barrier();

  auto cc_t1 = std::chrono::high_resolution_clock::now();

  ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

  std::string fullV2file = files_prefix + ".fullV2";

  Tensor<T> d_v2;
  if(!fs::exists(fullV2file)) {
    d_v2 = setupV2<T>(ec, MO, CI, cholVpr, chol_count, ec.exhw());
    if(ccsd_options.writet) {
      write_to_disk(d_v2, fullV2file, true);
      // Tensor<T>::deallocate(d_v2);
    }
  }
  else {
    d_v2 = Tensor<T>{{N, N, N, N}, {2, 2}};
    Tensor<T>::allocate(&ec, d_v2);
    read_from_disk(d_v2, fullV2file);
  }

  free_tensors(cholVpr);

  auto [residual, corr_energy] =
    cc2_canonical::cc2_v2_driver<T>(sys_data, ec, MO, d_t1, d_t2, d_f1, d_v2, d_r1, d_r2, d_r1s,
                                    d_r2s, d_t1s, d_t2s, p_evl_sorted, ccsd_restart, files_prefix);

  ccsd_stats(ec, hf_energy, residual, corr_energy, ccsd_options.threshold);

  if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
    // write_to_disk(d_t1,t1file);
    // write_to_disk(d_t2,t2file);
    if(rank == 0) {
      std::ofstream out(ccsdstatus, std::ios::out);
      if(!out) cerr << "Error opening file " << ccsdstatus << endl;
      out << 1 << std::endl;
      out.close();
    }
  }

  auto   cc_t2 = std::chrono::high_resolution_clock::now();
  double cc2_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0)
    std::cout << std::endl
              << "Time taken for spin-orbital CC2: " << std::fixed << std::setprecision(2)
              << cc2_time << " secs" << std::endl;

  cc_print(sys_data, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
  }
  free_tensors(d_t1, d_t2, d_f1, d_v2);

  ec.flush_and_sync();
  // delete ec;
}
