/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "qed_ccsd_os.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_1.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_2.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_3.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_4.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_5.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_6.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_7.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_resid_8.hpp"
#include "residuals/qed_ccsd_os/qed_ccsd_os_tmps.hpp"

template<typename T>
void exachem::cc::qed_ccsd_os::residuals(
  Scheduler& sch, const TiledIndexSpace& MO, const TensorMap<T>& f, const TensorMap<T>& eri,
  const TensorMap<T>& dp, const double w0, const TensorMap<T>& t1, const TensorMap<T>& t2,
  const double t0_1p, const TensorMap<T>& t1_1p, const TensorMap<T>& t2_1p, const double t0_2p,
  const TensorMap<T>& t1_2p, const TensorMap<T>& t2_2p, Tensor<T>& energy, TensorMap<T>& r1,
  TensorMap<T>& r2, Tensor<T>& r0_1p, TensorMap<T>& r1_1p, TensorMap<T>& r2_1p, Tensor<T>& r0_2p,
  TensorMap<T>& r1_2p, TensorMap<T>& r2_2p) {
  // build intermediates
  TensorMap<T> tmps, scalars;
  build_tmps(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p,
             t2_2p);

  sch.execute();

  // Residuals
  resid_1(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_2(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_3(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_4(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_5(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_6(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_7(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);
  resid_8(sch, MO, tmps, scalars, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p,
          energy, r1, r2, r0_1p, r1_1p, r2_1p, r0_2p, r1_2p, r2_2p);

  sch.execute();

  // Deallocate temporary tensors
  for(auto& [name, tmp]: tmps) {
    if(tmp.is_allocated()) sch.deallocate(tmp);
  }
  for(auto& [name, scalar]: scalars) sch.deallocate(scalar);

  sch.execute();
}

namespace exachem::cc::qed_ccsd_os {

template<typename T>
std::tuple<double, double> ccsd_v2_driver(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<T>& d_t1,
  Tensor<T>& d_t2, Tensor<T>& d_t1_1p, Tensor<T>& d_t2_1p, Tensor<T>& d_t1_2p, Tensor<T>& d_t2_2p,
  Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_r0_1p, Tensor<T>& d_r1_1p, Tensor<T>& d_r2_1p,
  Tensor<T>& d_r0_2p, Tensor<T>& d_r1_2p, Tensor<T>& d_r2_2p, std::vector<Tensor<T>>& d_r1s,
  std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_r1_1ps, std::vector<Tensor<T>>& d_r2_1ps,
  std::vector<Tensor<T>>& d_r1_2ps, std::vector<Tensor<T>>& d_r2_2ps, std::vector<Tensor<T>>& d_t1s,
  std::vector<Tensor<T>>& d_t2s, std::vector<Tensor<T>>& d_t1_1ps, std::vector<Tensor<T>>& d_t2_1ps,
  std::vector<Tensor<T>>& d_t1_2ps, std::vector<Tensor<T>>& d_t2_2ps, TensorMap<T>& f,
  TensorMap<T>& eri, TensorMap<T>& dp, double w0, std::vector<T>& p_evl_sorted,
  bool ccsd_restart = false, std::string ccsd_fp = "") {
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

  std::string t1file           = ccsd_fp + ".t1amp";
  std::string t2file           = ccsd_fp + ".t2amp";
  std::string t1_1pfile        = ccsd_fp + ".t1_1pamp";
  std::string t2_1pfile        = ccsd_fp + ".t2_1pamp";
  std::string t1_2pfile        = ccsd_fp + ".t1_2pamp";
  std::string t2_2pfile        = ccsd_fp + ".t2_2pamp";
  std::string t0_1p_t0_2p_file = ccsd_fp + ".t0_1p_t0_2p.txt";

  std::cout.precision(15);

  Tensor<T> d_e{};
  Tensor<T>::allocate(&ec, d_e);
  Scheduler sch{ec};

  // Allocate all tensors
  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const int              otiles  = O.num_tiles();
  const int              vtiles  = V.num_tiles();
  const int              oatiles = MO("occ_alpha").num_tiles();
  const int              vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  TiledIndexLabel aa, ba, ca, da;
  TiledIndexLabel ia, ja, ka, la;
  TiledIndexLabel ab, bb, cb, db;
  TiledIndexLabel ib, jb, kb, lb;

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");

  double t0_1p = 0.0;
  double t0_2p = 0.0;

  if(fs::exists(t0_1p_t0_2p_file)) {
    std::ifstream inFile(t0_1p_t0_2p_file);
    if(inFile.is_open()) {
      std::string label;

      inFile >> label >> t0_1p;
      inFile >> label >> t0_2p;
      inFile.close();
    }
  }
  print_ccsd_header(ec.print());

  if(!ccsd_restart) {
    TensorMap<T> t1, t2, t1_1p, t2_1p, t1_2p, t2_2p;

    t1["aa"]   = declare<T>(MO, "aa_vo");
    t1["bb"]   = declare<T>(MO, "bb_vo");
    t2["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    t2["abab"] = declare<T>(MO, "abab_vvoo");
    t2["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(t1.at("aa"), t1.at("bb"), t2.at("aaaa"), t2.at("abab"), t2.at("bbbb"));

    t1_1p["aa"]   = declare<T>(MO, "aa_vo");
    t1_1p["bb"]   = declare<T>(MO, "bb_vo");
    t2_1p["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    t2_1p["abab"] = declare<T>(MO, "abab_vvoo");
    t2_1p["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(t1_1p.at("aa"), t1_1p.at("bb"), t2_1p.at("aaaa"), t2_1p.at("abab"),
                 t2_1p.at("bbbb"));

    t1_2p["aa"]   = declare<T>(MO, "aa_vo");
    t1_2p["bb"]   = declare<T>(MO, "bb_vo");
    t2_2p["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    t2_2p["abab"] = declare<T>(MO, "abab_vvoo");
    t2_2p["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(t1_2p.at("aa"), t1_2p.at("bb"), t2_2p.at("aaaa"), t2_2p.at("abab"),
                 t2_2p.at("bbbb"));

    TensorMap<T> r1, r2, r1_1p, r2_1p, r1_2p, r2_2p;

    r1["aa"]   = declare<T>(MO, "aa_vo");
    r1["bb"]   = declare<T>(MO, "bb_vo");
    r2["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    r2["abab"] = declare<T>(MO, "abab_vvoo");
    r2["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(r1.at("aa"), r1.at("bb"), r2.at("aaaa"), r2.at("abab"), r2.at("bbbb"));

    r1_1p["aa"]   = declare<T>(MO, "aa_vo");
    r1_1p["bb"]   = declare<T>(MO, "bb_vo");
    r2_1p["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    r2_1p["abab"] = declare<T>(MO, "abab_vvoo");
    r2_1p["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(r1_1p.at("aa"), r1_1p.at("bb"), r2_1p.at("aaaa"), r2_1p.at("abab"),
                 r2_1p.at("bbbb"));

    r1_2p["aa"]   = declare<T>(MO, "aa_vo");
    r1_2p["bb"]   = declare<T>(MO, "bb_vo");
    r2_2p["aaaa"] = declare<T>(MO, "aaaa_vvoo");
    r2_2p["abab"] = declare<T>(MO, "abab_vvoo");
    r2_2p["bbbb"] = declare<T>(MO, "bbbb_vvoo");
    sch.allocate(r1_2p.at("aa"), r1_2p.at("bb"), r2_2p.at("aaaa"), r2_2p.at("abab"),
                 r2_2p.at("bbbb"));

    sch.execute(ec.exhw(), profile);

    Tensor<T> d_r1_residual{}, d_r2_residual{};

    Tensor<T> d_r1_1p_residual{};
    Tensor<T> d_r2_1p_residual{};

    Tensor<T> d_r1_2p_residual{};
    Tensor<T> d_r2_2p_residual{};

    Tensor<T>::allocate(&ec, d_r1_residual, d_r2_residual, d_r1_1p_residual, d_r2_1p_residual,
                        d_r1_2p_residual, d_r2_2p_residual);

    for(int titer = 0; titer < maxiter; titer += ndiis) {
      for(int iter = titer; iter < std::min(titer + ndiis, maxiter); iter++) {
        const auto timer_start = std::chrono::high_resolution_clock::now();

        niter   = iter;
        int off = iter - titer;

        // clang-format off
                    sch     ((d_t1s[off])()    = d_t1())
                            ((d_t2s[off])()    = d_t2())
                            ((d_t1_1ps[off])() = d_t1_1p())
                            ((d_t2_1ps[off])() = d_t2_1p())
                            ((d_t1_2ps[off])() = d_t1_2p())
                            ((d_t2_2ps[off])() = d_t2_2p()).execute();
        // clang-format on
        // clang-format off
                    sch
                    (   t1.at("aa")(aa,ia)    = d_t1(aa,ia))
                    (t1_1p.at("aa")(aa,ia) = d_t1_1p(aa, ia))
                    (t1_2p.at("aa")(aa,ia) = d_t1_2p(aa, ia))

                    (   t1.at("bb")(ab,ib)    = d_t1(ab,ib))
                    (t1_1p.at("bb")(ab,ib) = d_t1_1p(ab, ib))
                    (t1_2p.at("bb")(ab,ib) = d_t1_2p(ab, ib))


                    (   t2.at("aaaa")(aa,ba,ia,ja)    = d_t2(aa,ba,ia,ja))
                    (t2_1p.at("aaaa")(aa,ba,ia,ja) = d_t2_1p(aa, ba, ia, ja))
                    (t2_2p.at("aaaa")(aa,ba,ia,ja) = d_t2_2p(aa, ba, ia, ja))

                    (   t2.at("abab")(aa,bb,ia,jb)    = d_t2(aa,bb,ia,jb))
                    (t2_1p.at("abab")(aa,bb,ia,jb) = d_t2_1p(aa, bb, ia, jb))
                    (t2_2p.at("abab")(aa,bb,ia,jb) = d_t2_2p(aa, bb, ia, jb))

                    (   t2.at("bbbb")(ab,bb,ib,jb)    = d_t2(ab,bb,ib,jb))
                    (t2_1p.at("bbbb")(ab,bb,ib,jb) = d_t2_1p(ab, bb, ib, jb))
                    (t2_2p.at("bbbb")(ab,bb,ib,jb) = d_t2_2p(ab, bb, ib, jb))
                    .execute();
        // clang-format on

        // modified energy equation
        qed_ccsd_os::residuals(sch, MO, f, eri, dp, w0, t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p,
                               t2_2p, d_e, r1, r2, d_r0_1p, r1_1p, r2_1p, d_r0_2p, r1_2p, r2_2p);
        sch.execute(ec.exhw(), profile);

        // copy residuals to d_r1, d_r2, d_r1_1p, d_r2_1p, d_r1_2p, d_r2_2p

        // clang-format off
                    sch
                    (   d_r1(aa, ia)    = r1.at("aa")(aa, ia))
                    (d_r1_1p(aa, ia) = r1_1p.at("aa")(aa, ia))
                    (d_r1_2p(aa, ia) = r1_2p.at("aa")(aa, ia))

                    (   d_r1(ab, ib)    = r1.at("bb")(ab, ib))
                    (d_r1_1p(ab, ib) = r1_1p.at("bb")(ab, ib))
                    (d_r1_2p(ab, ib) = r1_2p.at("bb")(ab, ib))

                    (   d_r2(aa, ba, ia, ja)    = r2.at("aaaa")(aa, ba, ia, ja))
                    (d_r2_1p(aa, ba, ia, ja) = r2_1p.at("aaaa")(aa, ba, ia, ja))
                    (d_r2_2p(aa, ba, ia, ja) = r2_2p.at("aaaa")(aa, ba, ia, ja))

                    (   d_r2(aa, bb, ia, jb)    = r2.at("abab")(aa, bb, ia, jb))
                    (d_r2_1p(aa, bb, ia, jb) = r2_1p.at("abab")(aa, bb, ia, jb))
                    (d_r2_2p(aa, bb, ia, jb) = r2_2p.at("abab")(aa, bb, ia, jb))

                    (   d_r2(ab, bb, ib, jb)    = r2.at("bbbb")(ab, bb, ib, jb))
                    (d_r2_1p(ab, bb, ib, jb) = r2_1p.at("bbbb")(ab, bb, ib, jb))
                    (d_r2_2p(ab, bb, ib, jb) = r2_2p.at("bbbb")(ab, bb, ib, jb))
                    .execute(ec.exhw(), profile);
        // clang-format on

        double r0_1p_val = get_scalar(d_r0_1p);
        double r0_2p_val = get_scalar(d_r0_2p);
        if(fabs(w0) > 1e-12) {
          t0_1p += -r0_1p_val / w0;
          t0_2p += -r0_2p_val / (4 * w0);
        }
        else {
          t0_1p = 0.0;
          t0_2p = 0.0;
        }

        r0_1p_val = 0.50 * std::sqrt(r0_1p_val * r0_1p_val);
        r0_2p_val = 0.50 * std::sqrt(r0_2p_val * r0_2p_val);
        tamm::scale_ip(d_r1_2p, 0.5);
        tamm::scale_ip(d_r2_2p, 0.5);
        std::tie(residual, energy) = rest_qed(
          ec, MO, d_r1, d_r2, d_r1_1p, d_r2_1p, d_r1_2p, d_r2_2p, d_t1, d_t2, d_t1_1p, d_t2_1p,
          d_t1_2p, d_t2_2p, d_e, d_r1_residual, d_r2_residual, d_r1_1p_residual, d_r2_1p_residual,
          d_r1_2p_residual, d_r2_2p_residual, p_evl_sorted, zshiftl, w0, n_occ_alpha, n_occ_beta);
        residual = std::max({residual, r0_1p_val, r0_2p_val});

        // clang-format off
                    sch     ((d_r1s[off])()    = d_r1())
                            ((d_r2s[off])()    = d_r2())
                            ((d_r1_1ps[off])() = d_r1_1p())
                            ((d_r2_1ps[off])()  = d_r2_1p())
                            ((d_r1_2ps[off])() = d_r1_2p())
                            ((d_r2_2ps[off])() = d_r2_2p()).execute();
                    // clang-format off

                    const auto timer_end = std::chrono::high_resolution_clock::now();
                    auto       iter_time =
                            std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start))
                                    .count();

                    iteration_print(chem_env, ec.pg(), iter, residual, energy, iter_time);

                    if(writet && (((iter + 1) % writet_iter == 0) || (residual < thresh))) {
                        std::ofstream outFile(t0_1p_t0_2p_file,std::ios::out);
                        if (outFile.is_open()) {
                            outFile << std::fixed << std::setprecision(12);
                            outFile << "d_t0_1p: " << t0_1p << "\n";
                            outFile << "d_t0_2p: " << t0_2p << "\n";
                            outFile.close();
                        }
                        write_to_disk(d_t1, t1file);
                        write_to_disk(d_t2, t2file);
                        write_to_disk(d_t1_1p, t1_1pfile);
                        write_to_disk(d_t2_1p, t2_1pfile);
                        write_to_disk(d_t1_2p, t1_2pfile);
                        write_to_disk(d_t2_2p, t2_2pfile);
                    }

                    if(residual < thresh) { break; }
                }

                if(residual < thresh || titer + ndiis >= maxiter) { break; }
                if(ec.pg().rank() == 0) {
                    std::cout << " MICROCYCLE DIIS UPDATE:";
                    std::cout.width(21);
                    std::cout << std::right << std::min(titer + ndiis, maxiter) + 1 << std::endl;
                }

                std::vector<std::vector<Tensor<T>>> rs{d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps};
                std::vector<std::vector<Tensor<T>>> ts{d_t1s, d_t2s, d_t1_1ps, d_t2_1ps, d_t1_2ps, d_t2_2ps};
                std::vector<Tensor<T>>              next_t{d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p};
                diis<T>(ec, rs, ts, next_t);
            }

            if(profile && ec.print()) {
                std::string   profile_csv = ccsd_fp + "_profile.csv";
                std::ofstream pds(profile_csv, std::ios::out);
                if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
                pds << ec.get_profile_header() << std::endl;
                pds << ec.get_profile_data().str() << std::endl;
                pds.close();
            }

            Tensor<T>::deallocate(d_r1_residual, d_r2_residual, d_r1_1p_residual, d_r2_1p_residual, d_r1_2p_residual, d_r2_2p_residual);

            // deallocate tensors
            for (auto& [name, t] : t1)     sch.deallocate(t);
            for (auto& [name, t] : t2)     sch.deallocate(t);
            for (auto& [name, t] : t1_1p)  sch.deallocate(t);
            for (auto& [name, t] : t2_1p)  sch.deallocate(t);
            for (auto& [name, t] : t1_2p)  sch.deallocate(t);
            for (auto& [name, t] : t2_2p)  sch.deallocate(t);
            for (auto& [name, t] : r1)    sch.deallocate(t);
            for (auto& [name, t] : r2)    sch.deallocate(t);
            for (auto& [name, t] : r1_1p) sch.deallocate(t);
            for (auto& [name, t] : r2_1p) sch.deallocate(t);
            for (auto& [name, t] : r1_2p) sch.deallocate(t);
            for (auto& [name, t] : r2_2p) sch.deallocate(t);

        } // no restart
        else {
            sch.execute(ec.exhw(), profile);

            energy   = get_scalar(d_e);
            residual = 0.0;
        }

        chem_env.cc_context.ccsd_correlation_energy = energy;
        chem_env.cc_context.ccsd_total_energy = chem_env.scf_context.hf_energy + energy;

        auto   cc_t2 = std::chrono::high_resolution_clock::now();
        double ccsd_time =
                std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

        if(ec.pg().rank() == 0) {
            sys_data.results["output"]["CCSD"]["n_iterations"]                = niter + 1;
            sys_data.results["output"]["CCSD"]["final_energy"]["correlation"] = energy;
            sys_data.results["output"]["CCSD"]["final_energy"]["total"] = chem_env.cc_context.ccsd_total_energy;
            sys_data.results["output"]["CCSD"]["performance"]["total_time"] = ccsd_time;
            chem_env.write_json_data();
        }

        return std::make_tuple(residual, energy);
    }
};

void  exachem::cc::qed_ccsd_os::qed_driver(ExecutionContext& ec, ChemEnv& chem_env) {

    using T = double;
    using namespace scf;

    auto rank = ec.pg().rank();

    chem_env.cd_context.keep_movecs_so=true;
    cholesky_2e::cholesky_2e_driver(ec,chem_env);

    SystemData& sys_data    = chem_env.sys_data;

    std::string files_dir    = chem_env.get_files_dir();
    std::string files_prefix    = chem_env.get_files_prefix();

    CDContext& cd_context = chem_env.cd_context;
    CCContext& cc_context = chem_env.cc_context;
    cc_context.init_filenames(files_prefix);
    CCSDOptions& ccsd_options = chem_env.ioptions.ccsd_options;

    auto         debug        = ccsd_options.debug;
    bool         scf_conv       = chem_env.scf_context.no_scf;
    std::string t1file = cc_context.t1file;
    std::string t2file = cc_context.t2file;
    const bool ccsdstatus = cc_context.is_converged(chem_env.run_context, "ccsd");

    bool ccsd_restart = ccsd_options.readt || ((fs::exists(t1file) && fs::exists(t2file) &&
                                                fs::exists(cd_context.f1file) && fs::exists(cd_context.v2file)));

    Scheduler sch{ec};

    TiledIndexSpace& MO = chem_env.is_context.MSO;
    TiledIndexSpace& AO_opt = chem_env.is_context.AO_opt;
    Tensor<T>  QED_Dx{AO_opt, AO_opt};
    Tensor<T>  QED_Dy{AO_opt, AO_opt};
    Tensor<T>  QED_Dz{AO_opt, AO_opt};

    Tensor<T>  QED_Dx_new{AO_opt, AO_opt};
    Tensor<T>  QED_Dy_new{AO_opt, AO_opt};
    Tensor<T>  QED_Dz_new{AO_opt, AO_opt};

    sch.allocate(QED_Dx, QED_Dy, QED_Dz, QED_Dx_new, QED_Dy_new, QED_Dz_new).execute();

    scf::SCFIO<TensorType> scf_output;
    // read QED Dx,Dy,Dz
    scf_output.rw_mat_disk(QED_Dx, files_dir + "/scf/" + sys_data.output_file_prefix + ".QED_Dx", debug, true);
    scf_output.rw_mat_disk(QED_Dy, files_dir + "/scf/" + sys_data.output_file_prefix + ".QED_Dy", debug, true);
    scf_output.rw_mat_disk(QED_Dz, files_dir + "/scf/" + sys_data.output_file_prefix + ".QED_Dz", debug, true);


    SCFOptions& scf_options = chem_env.ioptions.scf_options;

    double omega = static_cast<double>(scf_options.qed_omegas[0]);
    double lambda = static_cast<double>(scf_options.qed_lambdas[0]);
    const std::vector<std::vector<double>> polvecs = scf_options.qed_polvecs;

    double pol_x = polvecs[0][0];
    double pol_y = polvecs[0][1];
    double pol_z = polvecs[0][2];

    double lambda_x = lambda*pol_x;
    double lambda_y = lambda*pol_y;
    double lambda_z = lambda*pol_z;

    //  std::vector<double> lambda_cav = {0.0, 0.0, 0.1};
    //  double lambda_x = lambda_cav[0];
    //  double lambda_y = lambda_cav[1];
    //  double lambda_z = lambda_cav[2];

    double lambda_x2 = lambda_x*lambda_x;
    double lambda_y2 = lambda_y*lambda_y;
    double lambda_z2 = lambda_z*lambda_z;

    double coupling_factor_x = lambda_x * std::sqrt(omega / 2.0);
    double coupling_factor_y = lambda_y * std::sqrt(omega / 2.0);
    double coupling_factor_z = lambda_z * std::sqrt(omega / 2.0);

    if(rank == 0){
        std::cout << "--------- QED-CCSD --------" << std::endl;


        std::cout << "lambda_x: "  << std::fixed  << std::setprecision(4) <<  lambda_x << std::endl;
        std::cout << "lambda_y: "  << std::fixed  << std::setprecision(4) <<  lambda_y << std::endl;
        std::cout << "lambda_z: "  << std::fixed  << std::setprecision(4) <<  lambda_z << std::endl;
//        std::cout << "lambda_x2: " << std::fixed  << std::setprecision(4) << lambda_x2 << std::endl;
//        std::cout << "lambda_y2: " << std::fixed  << std::setprecision(4) << lambda_y2 << std::endl;
//        std::cout << "lambda_z2: " << std::fixed  << std::setprecision(4) << lambda_z2 << std::endl;
        std::cout << "omega: " << std::fixed << std::setprecision(6) << omega << std::endl;
        std::cout << "coupling_factor_x: " << std::fixed << std::setprecision(6) << coupling_factor_x << std::endl;
        std::cout << "coupling_factor_y: " << std::fixed << std::setprecision(6) << coupling_factor_y << std::endl;
        std::cout << "coupling_factor_z: " << std::fixed << std::setprecision(6) << coupling_factor_z << std::endl;
        std::cout << "----------------------------" << std::endl;
    }

    // computing operator d_pq= d_e+d_n/N, its expectation value

    Tensor<T> dipole_x_exp{}, dipole_y_exp{}, dipole_z_exp{};
    sch.allocate(dipole_x_exp, dipole_y_exp, dipole_z_exp)
            .execute();

    scf::SCFData    spvars;
    scf::SCFCompute<T> scf_compute;
    scf_compute.compute_shellpair_list(ec, chem_env.shells, spvars);
    std::tie(spvars.shell_tile_map, spvars.AO_tiles, spvars.AO_opttiles) =
            scf_compute.compute_AO_tiles(ec, chem_env, chem_env.shells);

    auto          atoms = chem_env.atoms;
    scf::SCFGuess<T> scf_guess;

    Tensor<T> S1{AO_opt, AO_opt};
    sch.allocate(S1).execute();
    scf_guess.compute_1body_ints(ec, spvars, S1, atoms, chem_env.shells,
                                 libint2::Operator::overlap);

    auto [mu, nu, ku, lam, si] = AO_opt.labels<5>("all");
    auto [p, q, r, s]     = MO.labels<4>("all");

    // check SCF dipole by dumped SCF density

    Tensor<T> dens{AO_opt, AO_opt};
    //Tensor<T> temp1{MO, AO_opt};
    sch.allocate(dens)
            .execute();

    std::string densityfile_alpha =
            files_dir + "/scf/" + sys_data.output_file_prefix + ".alpha.density";
    scf_output.rw_mat_disk(dens, densityfile_alpha, debug, true); // read density


    // clang-format off
    sch(dipole_x_exp()  = 0)
            (dipole_y_exp()   = 0)
            (dipole_z_exp()   = 0)
            (dipole_x_exp()  += dens() * QED_Dx())
            (dipole_y_exp()  += dens() * QED_Dy())
            (dipole_z_exp()  += dens() * QED_Dz());
  // clang-format on
  sch.execute(ec.exhw());

  double dipole_x_elec = get_scalar(dipole_x_exp);
  double dipole_y_elec = get_scalar(dipole_y_exp);
  double dipole_z_elec = get_scalar(dipole_z_exp);

  double d_nuc_x = 0.0;
  double d_nuc_y = 0.0;
  double d_nuc_z = 0.0;

  for(size_t i = 0; i < atoms.size(); i++) {
    d_nuc_x += (atoms[i].x * atoms[i].atomic_number) / sys_data.nelectrons;
    d_nuc_y += (atoms[i].y * atoms[i].atomic_number) / sys_data.nelectrons;
    d_nuc_z += (atoms[i].z * atoms[i].atomic_number) / sys_data.nelectrons;
  }

  double dipole_total_x = dipole_x_elec + d_nuc_x;
  double dipole_total_y = dipole_y_elec + d_nuc_y;
  double dipole_total_z = dipole_z_elec + d_nuc_z;

  if(rank == 0) {
    std::cout << "----------------------------" << std::endl;
    std::cout << "Electronic dipole expectation value using SCF density: " << std::endl;
    std::cout << "dipole_x_elec: " << std::fixed << std::setprecision(8) << dipole_x_elec
              << std::endl;
    std::cout << "dipole_y_elec: " << std::fixed << std::setprecision(8) << dipole_y_elec
              << std::endl;
    std::cout << "dipole_z_elec: " << std::fixed << std::setprecision(8) << dipole_z_elec
              << std::endl;

    std::cout << std::endl
              << "d_n/N value: d_n is nuclear dipole moment and N is the total number of electrons "
              << std::endl;

    std::cout << "d_nuc_x: " << d_nuc_x << std::endl;
    std::cout << "d_nuc_y: " << d_nuc_y << std::endl;
    std::cout << "d_nuc_z: " << d_nuc_z << std::endl;

    std::cout << std::endl << "<d_e+d_n/N> expectation values: " << std::endl;

    std::cout << "dipole_total_x: " << std::fixed << std::setprecision(8) << dipole_total_x
              << std::endl;
    std::cout << "dipole_total_y: " << std::fixed << std::setprecision(8) << dipole_total_y
              << std::endl;
    std::cout << "dipole_total_z: " << std::fixed << std::setprecision(8) << dipole_total_z
              << std::endl;
    std::cout << "----------------------------" << std::endl;
  }

  // clang-format off
    sch(QED_Dx_new(lam, si)      = QED_Dx(lam, si))  // D-<D>= d_e+dn/N-<de+d_n/N>
            (QED_Dx_new(lam, si)  += d_nuc_x*S1(lam, si))
            (QED_Dx_new(lam, si)  -= dipole_total_x*S1(lam, si))
            (QED_Dy_new(lam, si)   = QED_Dy(lam, si))
            (QED_Dy_new(lam, si)  += d_nuc_y*S1(lam, si))
            (QED_Dy_new(lam, si)  -= dipole_total_y*S1(lam, si))
            (QED_Dz_new(lam, si)   = QED_Dz(lam, si));
    (QED_Dz_new(lam, si)  += d_nuc_z*S1(lam, si));
    (QED_Dz_new(lam, si)  -= dipole_total_z*S1(lam, si));
  // clang-format on
  sch.execute();

  // Scaled AO to MO transformation of dipole
  Tensor<T> dip_ints{{MO, MO}, {1, 1}};
  Tensor<T> array1{MO, AO_opt};
  Tensor<T> dip{{MO, MO}, {1, 1}};

  Tensor<T> lcao = chem_env.cd_context.movecs_so;

  sch.allocate(dip_ints, array1, dip).execute();

  // clang-format off
    sch(array1(p, nu)  = QED_Dx_new(mu, nu)*lcao(mu, p))
            (dip_ints(p, q)  = coupling_factor_x*array1(p, nu)*lcao(nu, q))
            (array1(p, nu)   = QED_Dy_new(mu, nu)*lcao(mu, p))
            (dip_ints(p, q) += coupling_factor_y*array1(p, nu)*lcao(nu, q))
            (array1(p, nu)   = QED_Dz_new(mu, nu)*lcao(mu, p))
            (dip_ints(p, q) += coupling_factor_z*array1(p, nu)*lcao(nu, q))
            (dip(p, q)       =-1.0*dip_ints(p, q));
  // clang-format on
  sch.execute(ec.exhw());

  TiledIndexSpace& CI      = chem_env.is_context.CI;
  TiledIndexSpace  N       = MO("all");
  Tensor<T>        d_f1    = chem_env.cd_context.d_f1;
  Tensor<T>        cholVpr = chem_env.cd_context.cholV2;

  std::vector<T>         p_evl_sorted;
  Tensor<T>              d_r1, d_r2, d_r0_1p, d_r0_2p, d_r1_1p, d_r2_1p, d_r1_2p, d_r2_2p;
  Tensor<T>              d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p;
  std::vector<Tensor<T>> d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps;
  std::vector<Tensor<T>> d_t1s, d_t2s, d_t1_1ps, d_t2_1ps, d_t1_2ps, d_t2_2ps;

  std::tie(p_evl_sorted, d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p, d_r1, d_r2, d_r0_1p,
           d_r1_1p, d_r2_1p, d_r0_2p, d_r1_2p, d_r2_2p, d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps,
           d_r2_2ps, d_t1s, d_t2s, d_t1_1ps, d_t2_1ps, d_t1_2ps, d_t2_2ps) =
    setupTensors_qed(ec, MO, d_f1, ccsd_options.ndiis, ccsd_restart && ccsdstatus && scf_conv);

  if(ccsd_restart) {
    if(fs::exists(t1file) && fs::exists(t2file)) {
      read_from_disk(d_t1, t1file);
      read_from_disk(d_t2, t2file);
      read_from_disk(d_t1_1p, chem_env.cc_context.t1_1pfile);
      read_from_disk(d_t2_1p, chem_env.cc_context.t2_1pfile);
      read_from_disk(d_t1_2p, chem_env.cc_context.t1_2pfile);
      read_from_disk(d_t2_2p, chem_env.cc_context.t2_2pfile);
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

  const TiledIndexSpace& O       = MO("occ");
  const TiledIndexSpace& V       = MO("virt");
  const int              otiles  = O.num_tiles();
  const int              vtiles  = V.num_tiles();
  const int              oatiles = MO("occ_alpha").num_tiles();
  const int              vatiles = MO("virt_alpha").num_tiles();

  const TiledIndexSpace Oa = {MO("occ"), range(oatiles)};
  const TiledIndexSpace Va = {MO("virt"), range(vatiles)};
  const TiledIndexSpace Ob = {MO("occ"), range(oatiles, otiles)};
  const TiledIndexSpace Vb = {MO("virt"), range(vatiles, vtiles)};

  TiledIndexLabel aa, ba, ca, da;
  TiledIndexLabel ia, ja, ka, la;
  TiledIndexLabel ab, bb, cb, db;
  TiledIndexLabel ib, jb, kb, lb;

  std::tie(aa, ba, ca, da) = Va.labels<4>("all");
  std::tie(ab, bb, cb, db) = Vb.labels<4>("all");
  std::tie(ia, ja, ka, la) = Oa.labels<4>("all");
  std::tie(ib, jb, kb, lb) = Ob.labels<4>("all");

  // one body integrals
  TensorMap<T>             f; // fock
  std::vector<std::string> f_blocks = {
    "aa_oo", "aa_ov", "aa_vo", "aa_vv", "bb_oo", "bb_ov", "bb_vo", "bb_vv",
  };
  for(const auto& block: f_blocks) {
    f[block] = declare<T>(MO, block);
    sch.allocate(f.at(block));
  }
  // clang-format off
    sch
    (f.at("aa_oo")(ia,ja) = d_f1(ia,ja)) (f.at("aa_ov")(ia,aa) = d_f1(ia,aa))
    (f.at("aa_vo")(aa,ia) = d_f1(aa,ia)) (f.at("aa_vv")(aa,ba) = d_f1(aa,ba))
    (f.at("bb_oo")(ib,jb) = d_f1(ib,jb)) (f.at("bb_ov")(ib,ab) = d_f1(ib,ab))
    (f.at("bb_vo")(ab,ib) = d_f1(ab,ib)) (f.at("bb_vv")(ab,bb) = d_f1(ab,bb))
    .execute(ec.exhw());
  // clang-format on

  TensorMap<T>             dp; // dipole
  std::vector<std::string> dp_blocks = {
    "aa_oo", "aa_ov", "aa_vo", "aa_vv", "bb_oo", "bb_ov", "bb_vo", "bb_vv",
  };
  for(const auto& block: dp_blocks) {
    dp[block] = declare<T>(MO, block);
    sch.allocate(dp.at(block));
  }

  // clang-format off
    sch
    (dp.at("aa_oo")(ia,ja) = dip(ia,ja)) (dp.at("aa_ov")(ia,aa) = dip(ia,aa))
    (dp.at("aa_vo")(aa,ia) = dip(aa,ia)) (dp.at("aa_vv")(aa,ba) = dip(aa,ba))
    (dp.at("bb_oo")(ib,jb) = dip(ib,jb)) (dp.at("bb_ov")(ib,ab) = dip(ib,ab))
    (dp.at("bb_vo")(ab,ib) = dip(ab,ib)) (dp.at("bb_vv")(ab,bb) = dip(ab,bb))
    .execute(ec.exhw());
  // clang-format on

  Tensor<T> dipole_mo_x{{MO, MO}, {1, 1}};
  Tensor<T> dipole_mo_y{{MO, MO}, {1, 1}};
  Tensor<T> dipole_mo_z{{MO, MO}, {1, 1}};
  Tensor<T> d_nuc_x_mo{{MO, MO}, {1, 1}};
  Tensor<T> d_nuc_y_mo{{MO, MO}, {1, 1}};
  Tensor<T> d_nuc_z_mo{{MO, MO}, {1, 1}};
  Tensor<T> temp_x{AO_opt, MO};
  Tensor<T> temp_y{AO_opt, MO};
  Tensor<T> temp_z{AO_opt, MO};
  Tensor<T> temp_ao_nuc_x{AO_opt, AO_opt};
  Tensor<T> temp_ao_nuc_y{AO_opt, AO_opt};
  Tensor<T> temp_ao_nuc_z{AO_opt, AO_opt};

  sch
    .allocate(dipole_mo_x, dipole_mo_y, dipole_mo_z, d_nuc_x_mo, d_nuc_y_mo, d_nuc_z_mo, temp_x,
              temp_y, temp_z, temp_ao_nuc_x, temp_ao_nuc_y, temp_ao_nuc_z)
    .execute();

  sch(temp_x(mu, p) = QED_Dx(mu, nu) * lcao(nu, p))(
    dipole_mo_x(p, q) = lcao(mu, p) * temp_x(mu, q))(temp_y(mu, p) = QED_Dy(mu, nu) * lcao(nu, p))(
    dipole_mo_y(p, q) = lcao(mu, p) * temp_y(mu, q))(temp_z(mu, p) = QED_Dz(mu, nu) * lcao(nu, p))(
    dipole_mo_z(p, q) = lcao(mu, p) * temp_z(mu, q))(temp_ao_nuc_x(mu, nu) = d_nuc_x * S1(mu, nu))(
    temp_x(mu, p) = temp_ao_nuc_x(mu, nu) * lcao(nu, p))(
    d_nuc_x_mo(p, q) = lcao(mu, p) * temp_x(mu, q))(temp_ao_nuc_y(mu, nu) = d_nuc_y * S1(mu, nu))(
    temp_y(mu, p) = temp_ao_nuc_y(mu, nu) * lcao(nu, p))(
    d_nuc_y_mo(p, q) = lcao(mu, p) * temp_y(mu, q))(temp_ao_nuc_z(mu, nu) = d_nuc_z * S1(mu, nu))(
    temp_z(mu, p) = temp_ao_nuc_z(mu, nu) * lcao(nu, p))(d_nuc_z_mo(p, q) =
                                                           lcao(mu, p) * temp_z(mu, q))
    .execute(ec.exhw());

  TiledIndexLabel Q;
  std::tie(Q) = CI.labels<1>("all");

  //    Tensor<T>& d_v2 = chem_env.cd_context.d_v2;
  //    d_v2 = cholesky_2e::setupV2<T>(ec, MO, CI, cholVpr, cd_context.num_chol_vecs, ec.exhw());

  // clang-format off
    auto build_eri_block = [&](Tensor<T>& v2_block,
                               const TiledIndexLabel& p, const TiledIndexLabel& q,
                               const TiledIndexLabel& r, const TiledIndexLabel& s){
        sch
                (v2_block(p, q, r, s)          = cholVpr(p, r, Q) * cholVpr(q, s, Q))
                (v2_block(p, q, r, s)         -= cholVpr(p, s, Q) * cholVpr(q, r, Q))
//                (v2_block(p, q, r, s)          = d_v2(p, q, r, s))

                (v2_block(p, q, r, s)         += lambda_x2*dipole_mo_x(p, r)*dipole_mo_x(q, s))
                (v2_block(p, q, r, s)         -= lambda_x2*dipole_mo_x(p, s)*dipole_mo_x(q, r))

                (v2_block(p, q, r, s)         -= lambda_x2*dipole_mo_x(p, r)*d_nuc_x_mo(q, s))
                (v2_block(p, q, r, s)         += lambda_x2*dipole_mo_x(p, s)*d_nuc_x_mo(q, r))

                (v2_block(p, q, r, s)         -= lambda_x2*d_nuc_x_mo(p, r)*dipole_mo_x(q, s))
                (v2_block(p, q, r, s)         += lambda_x2*d_nuc_x_mo(p, s)*dipole_mo_x(q, r))

                (v2_block(p, q, r, s)         += lambda_x2*d_nuc_x_mo(p, r)*d_nuc_x_mo(q, s))
                (v2_block(p, q, r, s)         -= lambda_x2*d_nuc_x_mo(p, s)*d_nuc_x_mo(q, r))

                (v2_block(p, q, r, s)         += lambda_y2*dipole_mo_y(p, r)*dipole_mo_y(q, s))
                (v2_block(p, q, r, s)         -= lambda_y2*dipole_mo_y(p, s)*dipole_mo_y(q, r))

                (v2_block(p, q, r, s)         -= lambda_y2*dipole_mo_y(p, r)*d_nuc_y_mo(q, s))
                (v2_block(p, q, r, s)         += lambda_y2*dipole_mo_y(p, s)*d_nuc_y_mo(q, r))

                (v2_block(p, q, r, s)         -= lambda_y2*d_nuc_y_mo(p, r)*dipole_mo_y(q, s))
                (v2_block(p, q, r, s)         += lambda_y2*d_nuc_y_mo(p, s)*dipole_mo_y(q, r))

                (v2_block(p, q, r, s)         += lambda_y2*d_nuc_y_mo(p, r)*d_nuc_y_mo(q, s))
                (v2_block(p, q, r, s)         -= lambda_y2*d_nuc_y_mo(p, s)*d_nuc_y_mo(q, r))

                (v2_block(p, q, r, s)         += lambda_z2*dipole_mo_z(p, r)*dipole_mo_z(q, s))
                (v2_block(p, q, r, s)         -= lambda_z2*dipole_mo_z(p, s)*dipole_mo_z(q, r))

                (v2_block(p, q, r, s)         -= lambda_z2*dipole_mo_z(p, r)*d_nuc_z_mo(q, s))
                (v2_block(p, q, r, s)         += lambda_z2*dipole_mo_z(p, s)*d_nuc_z_mo(q, r))

                (v2_block(p, q, r, s)         -= lambda_z2*d_nuc_z_mo(p, r)*dipole_mo_z(q, s))
                (v2_block(p, q, r, s)         += lambda_z2*d_nuc_z_mo(p, s)*dipole_mo_z(q, r))

                (v2_block(p, q, r, s)         += lambda_z2*d_nuc_z_mo(p, r)*d_nuc_z_mo(q, s))
                (v2_block(p, q, r, s)         -= lambda_z2*d_nuc_z_mo(p, s)*d_nuc_z_mo(q, r));
    };
  // clang-format on

  // print_ccsd_header(ec.print());

  TensorMap<T>             eri; // electron repulsion integrals
  std::vector<std::string> eri_blocks = {
    "aaaa_oooo", "aaaa_oovo", "aaaa_oovv", "aaaa_vooo", "aaaa_vovo", "aaaa_vovv", "aaaa_vvoo",
    "aaaa_vvvo", "aaaa_vvvv", "abab_oooo", "abab_oovo", "abab_oovv", "abab_vooo", "abab_vovo",
    "abab_vovv", "abab_vvoo", "abab_vvvo", "abab_vvvv", "abba_oovo", "abba_vovo", "abba_vvvo",
    "baab_vooo", "baab_vovo", "baab_vovv", "baba_vovo", "bbbb_oooo", "bbbb_oovo", "bbbb_oovv",
    "bbbb_vooo", "bbbb_vovo", "bbbb_vovv", "bbbb_vvoo", "bbbb_vvvo", "bbbb_vvvv"};

  for(const auto& block: eri_blocks) {
    eri[block] = declare<T>(MO, block);
    sch.allocate(eri.at(block));

    TiledIndexLabel p, q, r, s;
    if(block[0] == 'a' && block[0 + 5] == 'o') p = ia;
    if(block[0] == 'b' && block[0 + 5] == 'o') p = ib;
    if(block[0] == 'a' && block[0 + 5] == 'v') p = aa;
    if(block[0] == 'b' && block[0 + 5] == 'v') p = ab;

    if(block[1] == 'a' && block[1 + 5] == 'o') q = ja;
    if(block[1] == 'b' && block[1 + 5] == 'o') q = jb;
    if(block[1] == 'a' && block[1 + 5] == 'v') q = ba;
    if(block[1] == 'b' && block[1 + 5] == 'v') q = bb;

    if(block[2] == 'a' && block[2 + 5] == 'o') r = ka;
    if(block[2] == 'b' && block[2 + 5] == 'o') r = kb;
    if(block[2] == 'a' && block[2 + 5] == 'v') r = ca;
    if(block[2] == 'b' && block[2 + 5] == 'v') r = cb;

    if(block[3] == 'a' && block[3 + 5] == 'o') s = la;
    if(block[3] == 'b' && block[3 + 5] == 'o') s = lb;
    if(block[3] == 'a' && block[3 + 5] == 'v') s = da;
    if(block[3] == 'b' && block[3 + 5] == 'v') s = db;

    build_eri_block(eri.at(block), p, q, r, s);
  }
  sch.execute(ec.exhw());

  free_tensors(QED_Dx, QED_Dy, QED_Dz, QED_Dx_new, QED_Dy_new, QED_Dz_new, dip_ints, array1,
               dipole_x_exp, dipole_y_exp, dipole_z_exp);
  free_tensors(dipole_mo_x, dipole_mo_y, dipole_mo_z, d_nuc_x_mo, d_nuc_y_mo, d_nuc_z_mo, temp_x,
               temp_y, temp_z, temp_ao_nuc_x, temp_ao_nuc_y, temp_ao_nuc_z);
  free_tensors(lcao, S1);
  free_tensors(cholVpr);
  //    free_tensors(d_v2);

  auto [residual, corr_energy] = exachem::cc::qed_ccsd_os::ccsd_v2_driver<T>(
    chem_env, ec, MO, d_t1, d_t2, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p, d_r1, d_r2, d_r0_1p, d_r1_1p,
    d_r2_1p, d_r0_2p, d_r1_2p, d_r2_2p, d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps, d_t1s,
    d_t2s, d_t1_1ps, d_t2_1ps, d_t1_2ps, d_t2_2ps, f, eri, dp, omega, p_evl_sorted, ccsd_restart,
    files_prefix);

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
              << "Time taken for spin-orbital canonical QED-CCSD: " << std::fixed
              << std::setprecision(2) << ccsd_time << " secs" << std::endl;

  cc_print(chem_env, d_t1, d_t2, files_prefix);

  if(!ccsd_restart) {
    free_tensors(d_r1, d_r2, d_r0_1p, d_r1_1p, d_r2_1p, d_r0_2p, d_r1_2p, d_r2_2p);
    free_vec_tensors(d_r1s, d_r2s, d_r1_1ps, d_r2_1ps, d_r1_2ps, d_r2_2ps, d_t1s, d_t2s, d_t1_1ps,
                     d_t2_1ps, d_t1_2ps, d_t2_2ps);
  }
  free_tensors(d_t1, d_t2, d_f1, d_t1_1p, d_t2_1p, d_t1_2p, d_t2_2p, dip);
  // v2_se.deallocate(sch);

  for(auto& [block, tensor]: f) sch.deallocate(tensor);
  for(auto& [block, tensor]: dp) sch.deallocate(tensor);
  for(auto& [block, tensor]: eri) sch.deallocate(tensor);
  sch.execute(ec.exhw());
  ec.flush_and_sync();
}

template std::tuple<double, double> exachem::cc::qed_ccsd_os::ccsd_v2_driver<double>(
  ChemEnv& chem_env, ExecutionContext& ec, const TiledIndexSpace& MO, Tensor<double>& d_t1,
  Tensor<double>& d_t2, Tensor<double>& d_t1_1p, Tensor<double>& d_t2_1p, Tensor<double>& d_t1_2p,
  Tensor<double>& d_t2_2p, Tensor<double>& d_r1, Tensor<double>& d_r2, Tensor<double>& d_r0_1p,
  Tensor<double>& d_r1_1p, Tensor<double>& d_r2_1p, Tensor<double>& d_r0_2p,
  Tensor<double>& d_r1_2p, Tensor<double>& d_r2_2p, std::vector<Tensor<double>>& d_r1s,
  std::vector<Tensor<double>>& d_r2s, std::vector<Tensor<double>>& d_r1_1ps,
  std::vector<Tensor<double>>& d_r2_1ps, std::vector<Tensor<double>>& d_r1_2ps,
  std::vector<Tensor<double>>& d_r2_2ps, std::vector<Tensor<double>>& d_t1s,
  std::vector<Tensor<double>>& d_t2s, std::vector<Tensor<double>>& d_t1_1ps,
  std::vector<Tensor<double>>& d_t2_1ps, std::vector<Tensor<double>>& d_t1_2ps,
  std::vector<Tensor<double>>& d_t2_2ps, TensorMap<double>& f, TensorMap<double>& eri,
  TensorMap<double>& dp, double w0, std::vector<double>& p_evl_sorted, bool ccsd_restart = false,
  std::string ccsd_fp = "");

template void exachem::cc::qed_ccsd_os::residuals<double>(
  Scheduler& sch, const TiledIndexSpace& MO, const TensorMap<double>& f,
  const TensorMap<double>& eri, const TensorMap<double>& dp, const double w0,
  const TensorMap<double>& t1, const TensorMap<double>& t2, const double t0_1p,
  const TensorMap<double>& t1_1p, const TensorMap<double>& t2_1p, const double t0_2p,
  const TensorMap<double>& t1_2p, const TensorMap<double>& t2_2p, Tensor<double>& energy,
  TensorMap<double>& r1, TensorMap<double>& r2, Tensor<double>& r0_1p, TensorMap<double>& r1_1p,
  TensorMap<double>& r2_1p, Tensor<double>& r0_2p, TensorMap<double>& r1_2p,
  TensorMap<double>& r2_2p);