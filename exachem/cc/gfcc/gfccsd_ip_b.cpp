/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/gfcc/gfccsd_ip.hpp"

namespace exachem::cc::gfcc {

#if 0
  template<typename T>
  void gfccsd_driver_ip_b(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO,
    Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
    Tensor<T>& f1, Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
    Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
    Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
    Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
    Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
    Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
    Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
    Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba, Tensor<T>& v2ijab_aaaa,
    Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, std::vector<T>& p_evl_sorted_occ,
    std::vector<T>& p_evl_sorted_virt, const TAMM_SIZE nocc,
    const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& unit_tis, string files_prefix,
    string levelstr, double gf_omega) {
    using ComplexTensor = Tensor<std::complex<T>>;
  
    const CCSDOptions&     ccsd_options = chem_env.ioptions.ccsd_options;
    auto debug = ccsd_options.debug;
    const int          gf_nprocs_poi      = ccsd_options.gf_nprocs_poi;
    const size_t       gf_maxiter         = ccsd_options.gf_maxiter;
    const double       gf_eta             = ccsd_options.gf_eta;
    const double       gf_lshift          = ccsd_options.gf_lshift;
    const double       gf_threshold       = ccsd_options.gf_threshold;
    const bool         gf_preconditioning = ccsd_options.gf_preconditioning;
    // const double gf_damping_factor    = ccsd_options.gf_damping_factor;
    std::vector<size_t> gf_orbitals = ccsd_options.gf_orbitals;
  
    ProcGroup& sub_pg = chem_env.cc_context.sub_pg;
    ExecutionContext& sub_ec = (*chem_env.cc_context.sub_ec);
  
    const int noa = chem_env.sys_data.n_occ_alpha;
    const int nob = chem_env.sys_data.n_occ_beta;
    const int total_orbitals = chem_env.sys_data.nmo;
  
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");
    // auto [u1] = unit_tis.labels<1>("all");
  
    const int otiles  = O.num_tiles();
    const int vtiles  = V.num_tiles();
    const int oatiles = MO("occ_alpha").num_tiles();
    // const int obtiles = MO("occ_beta").num_tiles();
    const int vatiles = MO("virt_alpha").num_tiles();
    // const int vbtiles = MO("virt_beta").num_tiles();
  
    TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;
  
    o_alpha = {MO("occ"), range(oatiles)};
    v_alpha = {MO("virt"), range(vatiles)};
    o_beta  = {MO("occ"), range(oatiles, otiles)};
    v_beta  = {MO("virt"), range(vatiles, vtiles)};
  
    auto [p1_va]        = v_alpha.labels<1>("all");
    auto [p1_vb]        = v_beta.labels<1>("all");
    auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
    auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  
    std::cout.precision(15);
  
    Scheduler gsch{gec};
    auto      rank = gec.pg().rank();
  
    std::stringstream gfo;
    gfo << std::fixed << std::setprecision(2) << gf_omega;
  
    // PRINT THE HEADER FOR GF-CCSD ITERATIONS
    if(rank == 0) {
      std::stringstream gfp;
      gfp << std::endl
          << std::string(55, '-') << std::endl
          << "GF-CCSD (w = " << gfo.str() << ") " << std::endl;
      std::cout << gfp.str() << std::flush;
    }
  
    ComplexTensor dtmp_bbb{v_beta, o_beta, o_beta};
    ComplexTensor dtmp_aba{v_alpha, o_beta, o_alpha};
    ComplexTensor::allocate(&gec, dtmp_bbb, dtmp_aba);
  
    // double au2ev = 27.2113961;
  
    std::string dtmp_bbb_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_bbb.l" + levelstr;
    std::string dtmp_aba_file = files_prefix + ".W" + gfo.str() + ".r_dtmp_aba.l" + levelstr;
  
    if(fs::exists(dtmp_bbb_file) && fs::exists(dtmp_aba_file)) {
      read_from_disk(dtmp_bbb, dtmp_bbb_file);
      read_from_disk(dtmp_aba, dtmp_aba_file);
    }
    else {
      ComplexTensor DEArr_IP{V, O, O};
  
      double       denominator  = 0.0;
      const double lshift       = 1.00000000;
      auto         DEArr_lambda = [&](const IndexVector& bid) {
        const IndexVector            blockid = internal::translate_blockid(bid, DEArr_IP());
        const TAMM_SIZE              size    = DEArr_IP.block_size(blockid);
        std::vector<std::complex<T>> buf(size);
  
        auto   block_dims   = DEArr_IP.block_dims(blockid);
        auto   block_offset = DEArr_IP.block_offsets(blockid);
        size_t c            = 0;
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
              denominator =
                gf_omega + p_evl_sorted_virt[i] - p_evl_sorted_occ[j] - p_evl_sorted_occ[k];
              if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * lshift; }
              else if(denominator > 0.0 && denominator < 1.0) { denominator += lshift; }
              buf[c] = 1.0 / std::complex<T>(denominator, -1.0 * gf_eta);
              // if(i==0&&j==0) {
              //   cout << " omega,k,Eocc,denom_inv,buf : " << gf_omega << "," << k << "," <<
              //   p_evl_sorted_occ[k] << "," << denominator << "," << buf[c] << endl;
              // }
            }
          }
        }
        DEArr_IP.put(blockid, buf);
      };
  
      gsch.allocate(DEArr_IP).execute();
      if(sub_pg.is_valid()) {
        Scheduler sub_sch{sub_ec};
        sub_sch(DEArr_IP() = 0).execute();
        block_for(sub_ec, DEArr_IP(), DEArr_lambda);
        sub_sch(dtmp_bbb() = 0)(dtmp_aba() = 0)(dtmp_bbb(p1_vb, h1_ob, h2_ob) =
                                                  DEArr_IP(p1_vb, h1_ob, h2_ob))(
          dtmp_aba(p1_va, h1_ob, h2_oa) = DEArr_IP(p1_va, h1_ob, h2_oa))
          .execute();
      }
      gec.pg().barrier();
      gsch.deallocate(DEArr_IP).execute();
  
      write_to_disk(dtmp_bbb, dtmp_bbb_file);
      write_to_disk(dtmp_aba, dtmp_aba_file);
    }
  
    //------------------------
    auto nranks     = gec.pg().size().value();
  
    const size_t        num_oi           = nob;
    size_t              num_pi_processed = 0;
    std::vector<size_t> pi_tbp;
    // Check pi's already processed
    for(size_t pi = noa; pi < noa + num_oi; pi++) {
      std::string x1_b_wpi_file = files_prefix + ".x1_b.W" + gfo.str() + ".oi" + std::to_string(pi);
      std::string x2_bbb_wpi_file =
        files_prefix + ".x2_bbb.W" + gfo.str() + ".oi" + std::to_string(pi);
      std::string x2_aba_wpi_file =
        files_prefix + ".x2_aba.W" + gfo.str() + ".oi" + std::to_string(pi);
  
      if(fs::exists(x1_b_wpi_file) && fs::exists(x2_bbb_wpi_file) && fs::exists(x2_aba_wpi_file))
        num_pi_processed++;
      else pi_tbp.push_back(pi);
    }
  
    size_t num_pi_remain = num_oi - num_pi_processed;
    if(num_pi_remain == 0) {
      gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
      return;
    }
  
    EXPECTS(num_pi_remain == pi_tbp.size());
    const int  ppn       = gec.ppn();
    const int  nnodes    = gec.nnodes();
    int        subranks  = std::floor(nranks / num_pi_remain);
    const bool no_pg     = (subranks == 0 || subranks == 1);
    int        sub_nodes = 0;
  
    if(no_pg) {
      subranks  = nranks;
      sub_nodes = nnodes;
    }
    else {
      int sub_nodes = subranks / ppn;
      if(subranks % ppn > 0 || sub_nodes == 0) sub_nodes++;
      if(sub_nodes > nnodes) sub_nodes = nnodes;
      subranks = sub_nodes * ppn;
    }
  
    if(gf_nprocs_poi > 0) {
      if(nnodes > 1 && gf_nprocs_poi % ppn != 0)
        tamm_terminate("[ERROR] gf_nprocs_poi should be a muliple of user mpi ranks per node");
      if(nnodes == 1) {
        // TODO: This applies only when using GA's PR runtime
        int ga_num_pr = 1;
        if(const char* ga_npr = std::getenv("GA_NUM_PROGRESS_RANKS_PER_NODE")) {
          ga_num_pr = std::atoi(ga_npr);
        }
        if(ga_num_pr > 1)
          tamm_terminate("[ERROR] use of multiple GA progress ranks for a single node gfccsd "
                         "calculation is not allowed");
      }
      subranks  = gf_nprocs_poi;
      sub_nodes = subranks / ppn;
    }
    if(sub_nodes == 0) sub_nodes++;
  
    int num_oi_can_bp = nnodes / sub_nodes;
    if(nnodes % sub_nodes > 0) num_oi_can_bp++;
    // when using 1 node
    if(gf_nprocs_poi > 0 && nnodes == 1) {
      num_oi_can_bp = ppn / gf_nprocs_poi;
      if(ppn % gf_nprocs_poi > 0) num_oi_can_bp++;
    }
  
    if(rank == 0) {
      cout << "Total number of process groups = " << num_oi_can_bp << endl;
      cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
           << num_oi_can_bp << endl;
      cout << "No of processes used to compute each orbital = " << subranks << endl;
      // ofs_profile << "No of processes used to compute each orbital = " << subranks << endl;
    }
  
    ///////////////////////////
    //                       //
    //  MAIN ITERATION LOOP  //
    //        (beta)         //
    ///////////////////////////
    auto cc_t1 = std::chrono::high_resolution_clock::now();

#if GF_PGROUPS
    ProcGroup        pg = ProcGroup::create_subgroups(gec.pg(),subranks);
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    Scheduler        sch{ec};
#else
    Scheduler&        sch = gsch;
    ExecutionContext& ec  = gec;
#endif
  
    int64_t root_ppi = 0;
    for(size_t pib = 0; pib < num_pi_remain; pib += num_oi_can_bp) {
      size_t piv_end = num_oi_can_bp;
      if(pib + num_oi_can_bp > num_pi_remain) piv_end = pib + num_oi_can_bp - num_pi_remain;
      for(size_t piv = 0; piv < piv_end; piv++) {
        size_t pi = pi_tbp[piv + pib];
#if GF_PGROUPS
        if((rank >= piv * subranks && rank < (piv * subranks + subranks)) || no_pg) {
          if(!no_pg) root_ppi = piv * subranks; // root of sub-group
#endif
  
          auto gf_t1 = std::chrono::high_resolution_clock::now();
  
          bool          gf_conv = false;
          ComplexTensor Minv{{O, O}, {1, 1}};
          ComplexTensor x1{O};
          Tensor<T>     B1{O};
  
          ComplexTensor Hx1_b{o_beta};
          ComplexTensor Hx2_bbb{v_beta, o_beta, o_beta};
          ComplexTensor Hx2_aba{v_alpha, o_beta, o_alpha};
          ComplexTensor d1_b{o_beta};
          ComplexTensor d2_bbb{v_beta, o_beta, o_beta};
          ComplexTensor d2_aba{v_alpha, o_beta, o_alpha};
          ComplexTensor Minv_b{o_beta, o_beta};
          ComplexTensor dx1_b{o_beta};
          ComplexTensor dx2_bbb{v_beta, o_beta, o_beta};
          ComplexTensor dx2_aba{v_alpha, o_beta, o_alpha};
          ComplexTensor x1_b{o_beta};
          ComplexTensor x2_bbb{v_beta, o_beta, o_beta};
          ComplexTensor x2_aba{v_alpha, o_beta, o_alpha};
          Tensor<T>     B1_b{o_beta};
  
          sch
            .allocate(Hx1_b, Hx2_bbb, Hx2_aba, d1_b, d2_bbb, d2_aba, dx1_b, dx2_bbb, dx2_aba, x1,
                      x1_b, x2_bbb, x2_aba, Minv, Minv_b, B1, B1_b)
            .execute();
  
          int    total_iter     = 0;
          double gf_t_guess     = 0.0;
          double gf_t_x1_tot    = 0.0;
          double gf_t_x2_tot    = 0.0;
          double gf_t_res_tot   = 0.0;
          double gf_t_res_tot_1 = 0.0;
          double gf_t_res_tot_2 = 0.0;
          double gf_t_upd_tot   = 0.0;
          double gf_t_dis_tot   = 0.0;
  
          std::string x1_b_wpi_file =
            files_prefix + ".x1_b.W" + gfo.str() + ".oi" + std::to_string(pi);
          std::string x2_bbb_wpi_file =
            files_prefix + ".x2_bbb.W" + gfo.str() + ".oi" + std::to_string(pi);
          std::string x2_aba_wpi_file =
            files_prefix + ".x2_aba.W" + gfo.str() + ".oi" + std::to_string(pi);
  
          if(!(fs::exists(x1_b_wpi_file) && fs::exists(x2_bbb_wpi_file) &&
               fs::exists(x2_aba_wpi_file))) {
            ComplexTensor e1_b{o_beta, diis_tis};
            ComplexTensor e2_bbb{v_beta, o_beta, o_beta, diis_tis};
            ComplexTensor e2_aba{v_alpha, o_beta, o_alpha, diis_tis};
            ComplexTensor xx1_b{o_beta, diis_tis};
            ComplexTensor xx2_bbb{v_beta, o_beta, o_beta, diis_tis};
            ComplexTensor xx2_aba{v_alpha, o_beta, o_alpha, diis_tis};
  
            // clang-format off
            sch
              .allocate( e1_b, e2_bbb, e2_aba,
                        xx1_b,xx2_bbb,xx2_aba)
              (x1()      = 0)
              (Minv()    = 0)
              (Hx1_b()   = 0)
              (Hx2_bbb() = 0)
              (Hx2_aba() = 0)
              (d1_b()    = 0)
              (d2_bbb()  = 0)
              (d2_aba()  = 0)
              (x1_b()    = 0)
              (x2_bbb()  = 0)
              (x2_aba()  = 0)
              (Minv_b()  = 0)
              .execute();
            // clang-format on
  
            LabelLoopNest loop_nest{B1().labels()};
            sch(B1() = 0).execute();
  
            for(const IndexVector& bid: loop_nest) {
              const IndexVector blockid = internal::translate_blockid(bid, B1());
  
              const TAMM_SIZE         size = B1.block_size(blockid);
              std::vector<TensorType> buf(size);
              B1.get(blockid, buf);
              auto block_dims   = B1.block_dims(blockid);
              auto block_offset = B1.block_offsets(blockid);
              auto dim          = block_dims[0];
              auto offset       = block_offset[0];
              if(pi >= offset && pi < offset + dim) buf[pi - offset] = 1.0;
              B1.put(blockid, buf);
            }
  
            sch(B1_b(h1_ob) = B1(h1_ob)).execute();
            ec.pg().barrier();
  
            auto gf_t_guess1 = std::chrono::high_resolution_clock::now();
  
            gf_guess_ip(ec, MO, nocc, gf_omega, gf_eta, pi, p_evl_sorted_occ, t2v2_o, x1, Minv, true);
  
            // clang-format off
            sch
              (x1_b(h1_ob) = x1(h1_ob))
              (Minv_b(h1_ob,h2_ob)  = Minv(h1_ob,h2_ob))
              .execute();
            // clang-format on
  
            auto gf_t_guess2 = std::chrono::high_resolution_clock::now();
            gf_t_guess =
              std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_guess2 - gf_t_guess1))
                .count();
  
            for(size_t iter = 0; iter < gf_maxiter; iter += ndiis) {
              for(size_t micro = iter; micro < std::min(iter + ndiis, gf_maxiter); micro++) {
                total_iter    = micro;
                auto gf_t_ini = std::chrono::high_resolution_clock::now();
  
                gfccsd_x1_b(sch, MO, Hx1_b, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_b, x2_bbb,
                            x2_aba, f1, ix2_2_b, ix1_1_1_a, ix1_1_1_b, ix2_6_3_bbbb, ix2_6_3_baba,
                            unit_tis, false);
  
                auto gf_t_x1 = std::chrono::high_resolution_clock::now();
                gf_t_x1_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x1 - gf_t_ini))
                    .count();
  
                gfccsd_x2_b(sch, MO, Hx2_bbb, Hx2_aba, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, x1_b,
                            x2_bbb, x2_aba, f1, ix2_1_bbbb, ix2_1_baba, ix2_2_a, ix2_2_b, ix2_3_a,
                            ix2_3_b, ix2_4_bbbb, ix2_4_abab, ix2_5_aaaa, ix2_5_abba, ix2_5_baba,
                            ix2_5_bbbb, ix2_5_baab, ix2_6_2_a, ix2_6_2_b, ix2_6_3_aaaa, ix2_6_3_abba,
                            ix2_6_3_baba, ix2_6_3_bbbb, ix2_6_3_baab, v2ijab_aaaa, v2ijab_abab,
                            v2ijab_bbbb, unit_tis, false);
  
                sch.execute(); // TODO: not needed if not profiling
                auto gf_t_x2 = std::chrono::high_resolution_clock::now();
                gf_t_x2_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_x2 - gf_t_x1))
                    .count();
  
                // clang-format off
                sch
                  (d1_b() = -1.0 * Hx1_b())
                  (d1_b(h1_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x1_b(h1_ob))
                  (d1_b() += B1_b())
                  (d2_bbb() = -1.0 * Hx2_bbb())
                  (d2_bbb(p1_vb,h1_ob,h2_ob) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_bbb(p1_vb,h1_ob,h2_ob))
                  (d2_aba() = -1.0 * Hx2_aba())
                  (d2_aba(p1_va,h1_ob,h2_oa) -= std::complex<double>(gf_omega,-1.0*gf_eta) * x2_aba(p1_va,h1_ob,h2_oa))
                  .execute();
                // clang-format on
  
                auto gf_t_res_1 = std::chrono::high_resolution_clock::now();
                gf_t_res_tot_1 +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_1 - gf_t_x2))
                    .count();
  
                auto d1_b_norm   = norm(d1_b);
                auto d2_bbb_norm = norm(d2_bbb);
                auto d2_aba_norm = norm(d2_aba);
  
                auto gf_t_res_2 = std::chrono::high_resolution_clock::now();
                gf_t_res_tot_2 +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_2 - gf_t_res_1))
                    .count();
  
                double gf_residual =
                  sqrt(d1_b_norm * d1_b_norm + d2_bbb_norm * d2_bbb_norm + d2_aba_norm * d2_aba_norm)
                    .real();
  
                if(debug && rank == root_ppi)
                  cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                       << "): #iter " << total_iter << ", residual = " << std::fixed
                       << std::setprecision(6) << gf_residual << std::endl;
                auto gf_t_res = std::chrono::high_resolution_clock::now();
                gf_t_res_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res - gf_t_x2))
                    .count();
  
                if(gf_residual < gf_threshold) {
                  gf_conv = true;
                  break;
                }
  
                // JACOBI
                auto gindx = micro - iter;
  
                TiledIndexSpace hist_tis{diis_tis, range(gindx, gindx + 1)};
                auto [dh1] = hist_tis.labels<1>("all");
  
                // clang-format off
                sch
                  (dx1_b(h1_ob) = Minv_b(h1_ob,h2_ob) * d1_b(h2_ob))
                  (dx2_bbb(p1_vb,h1_ob,h2_ob) = dtmp_bbb(p1_vb,h1_ob,h2_ob) * d2_bbb(p1_vb,h1_ob,h2_ob))
                  (dx2_aba(p1_va,h1_ob,h2_oa) = dtmp_aba(p1_va,h1_ob,h2_oa) * d2_aba(p1_va,h1_ob,h2_oa))
                  (x1_b(h1_ob) += gf_damping_factor * dx1_b(h1_ob))
                  (x2_bbb(p1_vb,h1_ob,h2_ob) += gf_damping_factor * dx2_bbb(p1_vb,h1_ob,h2_ob))
                  (x2_aba(p1_va,h1_ob,h2_oa) += gf_damping_factor * dx2_aba(p1_va,h1_ob,h2_oa))
                  (e1_b(h1_ob,dh1) = d1_b(h1_ob))
                  (e2_bbb(p1_vb,h1_ob,h2_ob,dh1) = d2_bbb(p1_vb,h1_ob,h2_ob))
                  (e2_aba(p1_va,h1_ob,h2_oa,dh1) = d2_aba(p1_va,h1_ob,h2_oa))
                  (xx1_b(h1_ob,dh1) = x1_b(h1_ob))
                  (xx2_bbb(p1_vb,h1_ob,h2_ob,dh1) = x2_bbb(p1_vb,h1_ob,h2_ob))
                  (xx2_aba(p1_va,h1_ob,h2_oa,dh1) = x2_aba(p1_va,h1_ob,h2_oa))
                  .execute();
                // clang-format on
  
                auto gf_t_upd = std::chrono::high_resolution_clock::now();
                gf_t_upd_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_upd - gf_t_res))
                    .count();
  
              } // end micro
  
              if(gf_conv || iter + ndiis >= gf_maxiter) { break; }
  
              auto gf_t_tmp = std::chrono::high_resolution_clock::now();
  
              // DIIS
              ret_gf_diis_b(ec, MO, ndiis, e1_b, e2_bbb, e2_aba, xx1_b, xx2_bbb, xx2_aba, x1_b,
                            x2_bbb, x2_aba, diis_tis, unit_tis, true);
  
              auto gf_t_dis = std::chrono::high_resolution_clock::now();
              gf_t_dis_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_dis - gf_t_tmp))
                  .count();
            } // end iter
  
            write_to_disk(x1_b, x1_b_wpi_file);
            write_to_disk(x2_bbb, x2_bbb_wpi_file);
            write_to_disk(x2_aba, x2_aba_wpi_file);
  
            // sch.deallocate(e1,e2,xx1,xx2).execute();
            sch.deallocate(e1_b, e2_bbb, e2_aba, xx1_b, xx2_bbb, xx2_aba).execute();
          }
          else { gf_conv = true; }
  
          if(!gf_conv) { //&& rank == root_ppi)
            std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
            tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
          }
  
          auto   gf_t2 = std::chrono::high_resolution_clock::now();
          double gftime =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
          if(rank == root_ppi) {
            std::string gf_stats;
            gf_stats = gfacc_str("R-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                                 ") = ", std::to_string(gftime),
                                 " secs, #iter = ", std::to_string(total_iter));
  
            // if(debug){
            gf_stats += gfacc_str("|----------initial guess  : ", std::to_string(gf_t_guess));
            gf_stats += gfacc_str("|----------x1 contraction : ", std::to_string(gf_t_x1_tot));
            gf_stats += gfacc_str("|----------x2 contraction : ", std::to_string(gf_t_x2_tot));
            gf_stats += gfacc_str("|----------computing res. : ", std::to_string(gf_t_res_tot));
            gf_stats +=
              gfacc_str("           |----------misc. contr. : ", std::to_string(gf_t_res_tot_1));
            gf_stats +=
              gfacc_str("           |----------compt. norm  : ", std::to_string(gf_t_res_tot_2));
            gf_stats += gfacc_str("|----------updating x1/x2 : ", std::to_string(gf_t_upd_tot));
            gf_stats += gfacc_str("|----------diis update    : ", std::to_string(gf_t_dis_tot));
            //}
            std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
          }
  
          sch
            .deallocate(Hx1_b, Hx2_bbb, Hx2_aba, d1_b, d2_bbb, d2_aba, dx1_b, dx2_bbb, dx2_aba, x1,
                        x1_b, x2_bbb, x2_aba, Minv, Minv_b, B1, B1_b)
            .execute();

#if GF_PGROUPS
        }
#endif
      } // end pi batch
      ec.pg().barrier();
    } // end all remaining pi

#if GF_PGROUPS
    ec.flush_and_sync();
    // MemoryManagerGA::destroy_coll(mgr);
    pg.destroy_coll();
#endif
    gec.pg().barrier();
  
    auto   cc_t2 = std::chrono::high_resolution_clock::now();
    double time  = std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) {
      std::cout << "Total R-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
                << std::endl;
      std::cout << std::string(55, '-') << std::endl;
    }
  
    gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
  } // gfccsd_driver_ip_b
  
  
template void gfccsd_driver_ip_b<double>(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO,
    Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
    Tensor<T>& f1, Tensor<T>& t2v2_o, Tensor<T>& lt12_o_a, Tensor<T>& lt12_o_b, Tensor<T>& ix1_1_1_a,
    Tensor<T>& ix1_1_1_b, Tensor<T>& ix2_1_aaaa, Tensor<T>& ix2_1_abab, Tensor<T>& ix2_1_bbbb,
    Tensor<T>& ix2_1_baba, Tensor<T>& ix2_2_a, Tensor<T>& ix2_2_b, Tensor<T>& ix2_3_a,
    Tensor<T>& ix2_3_b, Tensor<T>& ix2_4_aaaa, Tensor<T>& ix2_4_abab, Tensor<T>& ix2_4_bbbb,
    Tensor<T>& ix2_5_aaaa, Tensor<T>& ix2_5_abba, Tensor<T>& ix2_5_abab, Tensor<T>& ix2_5_bbbb,
    Tensor<T>& ix2_5_baab, Tensor<T>& ix2_5_baba, Tensor<T>& ix2_6_2_a, Tensor<T>& ix2_6_2_b,
    Tensor<T>& ix2_6_3_aaaa, Tensor<T>& ix2_6_3_abba, Tensor<T>& ix2_6_3_abab,
    Tensor<T>& ix2_6_3_bbbb, Tensor<T>& ix2_6_3_baab, Tensor<T>& ix2_6_3_baba, Tensor<T>& v2ijab_aaaa,
    Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, std::vector<T>& p_evl_sorted_occ,
    std::vector<T>& p_evl_sorted_virt, const TAMM_SIZE nocc,
    const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& unit_tis, string files_prefix,
    string levelstr, double gf_omega);
#endif

} // namespace exachem::cc::gfcc