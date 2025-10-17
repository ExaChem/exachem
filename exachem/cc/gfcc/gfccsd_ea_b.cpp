/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/gfcc/gfccsd_ea_b.hpp"
namespace exachem::cc::gfcc {

#if 0
  template<typename T>
  void exachem::cc::gfcc::GFCCSD_EA_B_Driver<T>::gfccsd_driver_ea_b(
    ExecutionContext& gec, ChemEnv& chem_env, const TiledIndexSpace& MO,
    Tensor<T>& t1_a, Tensor<T>& t1_b, Tensor<T>& t2_aaaa, Tensor<T>& t2_bbbb, Tensor<T>& t2_abab,
    Tensor<T>& f1, Tensor<T>& t2v2_v, Tensor<T>& lt12_v_a, Tensor<T>& lt12_v_b, Tensor<T>& iy1_1_a,
    Tensor<T>& iy1_1_b, Tensor<T>& iy1_2_1_a, Tensor<T>& iy1_2_1_b, Tensor<T>& iy1_a,
    Tensor<T>& iy1_b, Tensor<T>& iy2_a, Tensor<T>& iy2_b, Tensor<T>& iy3_1_aaaa,
    Tensor<T>& iy3_1_bbbb, Tensor<T>& iy3_1_abab, Tensor<T>& iy3_1_baba, Tensor<T>& iy3_1_baab,
    Tensor<T>& iy3_1_abba, Tensor<T>& iy3_1_2_a, Tensor<T>& iy3_1_2_b, Tensor<T>& iy3_aaaa,
    Tensor<T>& iy3_bbbb, Tensor<T>& iy3_abab, Tensor<T>& iy3_baba, Tensor<T>& iy3_baab,
    Tensor<T>& iy3_abba, Tensor<T>& iy4_1_aaaa, Tensor<T>& iy4_1_baab, Tensor<T>& iy4_1_baba,
    Tensor<T>& iy4_1_bbbb, Tensor<T>& iy4_1_abba, Tensor<T>& iy4_1_abab, Tensor<T>& iy4_2_aaaa,
    Tensor<T>& iy4_2_baab, Tensor<T>& iy4_2_bbbb, Tensor<T>& iy4_2_abba, Tensor<T>& iy5_aaaa,
    Tensor<T>& iy5_abab, Tensor<T>& iy5_baab, Tensor<T>& iy5_bbbb, Tensor<T>& iy5_baba,
    Tensor<T>& iy5_abba, Tensor<T>& iy6_a, Tensor<T>& iy6_b, Tensor<T>& v2ijab_aaaa,
    Tensor<T>& v2ijab_abab, Tensor<T>& v2ijab_bbbb, Tensor<T>& cholOO_a, Tensor<T>& cholOO_b,
    Tensor<T>& cholOV_a, Tensor<T>& cholOV_b, Tensor<T>& cholVV_a, Tensor<T>& cholVV_b,
    std::vector<T>& p_evl_sorted_occ, std::vector<T>& p_evl_sorted_virt,
    const TAMM_SIZE nocc, const TAMM_SIZE nvir, size_t& nptsi, const TiledIndexSpace& CI,
    const TiledIndexSpace& unit_tis, std::string files_prefix, std::string levelstr, double gf_omega) {
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
  
    const int nva = chem_env.sys_data.n_vir_alpha;
    const int nvb = chem_env.sys_data.n_vir_beta;
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
  
    auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
    auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
    auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  
    std::cout.precision(15);
  
    // Create instance of GFCCSD_EA_B for calling gfccsd_y1_b and gfccsd_y2_b
    GFCCSD_EA_B<T> gfccsd_ea_b_instance;
  
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
  
    ComplexTensor dtmp_bbb{o_beta, v_beta, v_beta};
    ComplexTensor dtmp_aba{o_alpha, v_beta, v_alpha};
    ComplexTensor::allocate(&gec, dtmp_bbb, dtmp_aba);
  
    // double au2ev = 27.2113961;
  
    const std::string dtmp_bbb_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_bbb.l" + levelstr;
    const std::string dtmp_aba_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_aba.l" + levelstr;
  
    if(fs::exists(dtmp_bbb_file) && fs::exists(dtmp_aba_file)) {
      read_from_disk(dtmp_bbb, dtmp_bbb_file);
      read_from_disk(dtmp_aba, dtmp_aba_file);
    }
    else {
      ComplexTensor DEArr_EA{O, V, V};
  
      double       denominator     = 0.0;
      const double lshift          = 1.00000000;
      auto         DEArr_EA_lambda = [&](const IndexVector& bid) {
        const IndexVector            blockid = internal::translate_blockid(bid, DEArr_EA());
        const TAMM_SIZE              size    = DEArr_EA.block_size(blockid);
        std::vector<std::complex<T>> buf(size);
  
        auto   block_dims   = DEArr_EA.block_dims(blockid);
        auto   block_offset = DEArr_EA.block_offsets(blockid);
        size_t c            = 0;
        for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0]; i++) {
          for(size_t j = block_offset[1]; j < block_offset[1] + block_dims[1]; j++) {
            for(size_t k = block_offset[2]; k < block_offset[2] + block_dims[2]; k++, c++) {
              denominator =
                gf_omega + p_evl_sorted_occ[i] - p_evl_sorted_virt[j] - p_evl_sorted_virt[k];
              if(denominator < 0.0 && denominator > -1.0) { denominator += -1.0 * lshift; }
              else if(denominator > 0.0 && denominator < 1.0) { denominator += lshift; }
              buf[c] = 1.0 / std::complex<T>(denominator, gf_eta);
            }
          }
        }
        DEArr_EA.put(blockid, buf);
      };
  
      gsch.allocate(DEArr_EA).execute();
      if(sub_pg.is_valid()) {
        Scheduler sub_sch{sub_ec};
        sub_sch(DEArr_EA() = 0).execute();
        block_for(sub_ec, DEArr_EA(), DEArr_EA_lambda);
        // clang-format off
        sub_sch      
          (dtmp_bbb() = 0)
          (dtmp_aba() = 0)
          (dtmp_bbb(h1_ob,p1_vb,p2_vb) = DEArr_EA(h1_ob,p1_vb,p2_vb))
          (dtmp_aba(h1_oa,p1_vb,p2_va) = DEArr_EA(h1_oa,p1_vb,p2_va))
          .execute();
        // clang-format on
      }
      gec.pg().barrier();
      gsch.deallocate(DEArr_EA).execute();
  
      write_to_disk(dtmp_bbb, dtmp_bbb_file);
      write_to_disk(dtmp_aba, dtmp_aba_file);
    }
  
    //------------------------
    auto nranks     = gec.pg().size().value();
  
    const size_t        num_oi           = nvb;
    size_t              num_pi_processed = 0;
    std::vector<size_t> pi_tbp;
    // Check pi's already processed
    for(size_t pi = nva; pi < nva + num_oi; pi++) {
      const std::string y1_b_wpi_file = files_prefix + ".y1_b.w" + gfo.str() + ".oi" + std::to_string(pi);
      const std::string y2_bbb_wpi_file =
        files_prefix + ".y2_bbb.w" + gfo.str() + ".oi" + std::to_string(pi);
      const std::string y2_aba_wpi_file =
        files_prefix + ".y2_aba.w" + gfo.str() + ".oi" + std::to_string(pi);
  
      if(fs::exists(y1_b_wpi_file) && fs::exists(y2_bbb_wpi_file) && fs::exists(y2_aba_wpi_file))
        num_pi_processed++;
      else pi_tbp.push_back(pi);
    }
  
    size_t num_pi_remain = num_oi - num_pi_processed;
    if(num_pi_remain == 0) {
      gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
      return;
    }
    EXPECTS(num_pi_remain == pi_tbp.size());
    // if(num_pi_remain == 0) num_pi_remain = 1;
    int        subranks = std::floor(nranks / num_pi_remain);
    const bool no_pg    = (subranks == 0 || subranks == 1);
    if(no_pg) subranks = nranks;
    if(gf_nprocs_poi > 0) subranks = gf_nprocs_poi;
  
    // Figure out how many orbitals in pi_tbp can be processed with subranks
    // TODO: gf_nprocs_pi must be a multiple of total #ranks for best performance.
    size_t num_oi_can_bp = std::ceil(nranks / (1.0 * subranks));
    if(num_pi_remain < num_oi_can_bp) {
      num_oi_can_bp = num_pi_remain;
      subranks      = std::floor(nranks / num_pi_remain);
      if(no_pg) subranks = nranks;
    }
  
    if(rank == 0) {
      std::cout << "Total, remaining orbitals, batch size = " << num_oi << ", " << num_pi_remain << ", "
           << num_oi_can_bp << std::endl;
      std::cout << "No of processes used to compute each orbital = " << subranks << std::endl;
      // ofs_profile << "No of processes used to compute each orbital = " << subranks << std::endl;
    }
  
    ///////////////////////////
    //                       //
    //  MAIN ITERATION LOOP  //
    //        (beta)        //
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
          ComplexTensor Minv{{V, V}, {1, 1}};
          ComplexTensor y1{V};
          Tensor<T>     B1{V};
  
          ComplexTensor Hy1_b{v_beta};
          ComplexTensor Hy2_bbb{o_beta, v_beta, v_beta};
          ComplexTensor Hy2_aba{o_alpha, v_beta, v_alpha};
          ComplexTensor d1_b{v_beta};
          ComplexTensor d2_bbb{o_beta, v_beta, v_beta};
          ComplexTensor d2_aba{o_alpha, v_beta, v_alpha};
          ComplexTensor Minv_b{v_beta, v_beta};
          ComplexTensor dy1_b{v_beta};
          ComplexTensor dy2_bbb{o_beta, v_beta, v_beta};
          ComplexTensor dy2_aba{o_alpha, v_beta, v_alpha};
          ComplexTensor y1_b{v_beta};
          ComplexTensor y2_bbb{o_beta, v_beta, v_beta};
          ComplexTensor y2_aba{o_alpha, v_beta, v_alpha};
          Tensor<T>     B1_b{v_beta};
  
          sch
            .allocate(Hy1_b, Hy2_bbb, Hy2_aba, d1_b, d2_bbb, d2_aba, dy1_b, dy2_bbb, dy2_aba, y1,
                      y1_b, y2_bbb, y2_aba, Minv, Minv_b, B1, B1_b)
            .execute();
  
          int    total_iter     = 0;
          double gf_t_guess     = 0.0;
          double gf_t_y1_tot    = 0.0;
          double gf_t_y2_tot    = 0.0;
          double gf_t_res_tot   = 0.0;
          double gf_t_res_tot_1 = 0.0;
          double gf_t_res_tot_2 = 0.0;
          double gf_t_upd_tot   = 0.0;
          double gf_t_dis_tot   = 0.0;
  
          const std::string y1_b_wpi_file =
            files_prefix + ".y1_b.w" + gfo.str() + ".oi" + std::to_string(pi);
          const std::string y2_bbb_wpi_file =
            files_prefix + ".y2_bbb.w" + gfo.str() + ".oi" + std::to_string(pi);
          const std::string y2_aba_wpi_file =
            files_prefix + ".y2_aba.w" + gfo.str() + ".oi" + std::to_string(pi);
  
          if(!(fs::exists(y1_b_wpi_file) && fs::exists(y2_bbb_wpi_file) &&
               fs::exists(y2_aba_wpi_file))) {
            ComplexTensor e1_b{v_beta, diis_tis};
            ComplexTensor e2_bbb{o_beta, v_beta, v_beta, diis_tis};
            ComplexTensor e2_aba{o_alpha, v_beta, v_alpha, diis_tis};
            ComplexTensor yy1_b{v_beta, diis_tis};
            ComplexTensor yy2_bbb{o_beta, v_beta, v_beta, diis_tis};
            ComplexTensor yy2_aba{o_alpha, v_beta, v_alpha, diis_tis};
  
            // clang-format off
            sch
              .allocate(e1_b, e2_bbb, e2_aba, 
                      yy1_b,yy2_bbb,yy2_aba)
              (y1()      = 0)
              (Minv()    = 0)
              (Hy1_b()   = 0)
              (Hy2_bbb() = 0)
              (Hy2_aba() = 0)
              (d1_b()    = 0)
              (d2_bbb()  = 0)
              (d2_aba()  = 0)
              (y1_b()    = 0)
              (y2_bbb()  = 0)
              (y2_aba()  = 0)
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
  
            sch(B1_b(p1_vb) = B1(p1_vb)).execute();
            ec.pg().barrier();
  
            auto gf_t_guess1 = std::chrono::high_resolution_clock::now();
  
            gf_guess_ea(ec, MO, nvir, gf_omega, gf_eta, pi, p_evl_sorted_virt, t2v2_v, y1, Minv,
                        true);
  
            // clang-format off
            sch
              (y1_b(p1_vb) = y1(p1_vb))
              (Minv_b(p1_vb,p2_vb)  = Minv(p1_vb,p2_vb))
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
  
                gfccsd_ea_b_instance.gfccsd_y1_b(sch, MO, Hy1_b, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_b, y2_bbb,
                            y2_aba, f1, iy1_b, iy1_2_1_a, iy1_2_1_b, v2ijab_bbbb, v2ijab_abab,
                            cholOV_a, cholOV_b, cholVV_b, CI, unit_tis, false);
  
                auto gf_t_y1 = std::chrono::high_resolution_clock::now();
                gf_t_y1_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_y1 - gf_t_ini))
                    .count();
  
                gfccsd_ea_b_instance.gfccsd_y2_b(sch, MO, Hy2_bbb, Hy2_aba, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_b,
                            y2_bbb, y2_aba, f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b,
                            iy3_1_bbbb, iy3_1_abab, iy3_1_baab, iy3_1_2_a, iy3_1_2_b, iy3_aaaa,
                            iy3_bbbb, iy3_abab, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab,
                            iy4_1_bbbb, iy4_1_abba, iy4_1_abab, iy4_2_bbbb, iy4_2_baab, iy5_aaaa,
                            iy5_baab, iy5_bbbb, iy5_abab, iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b,
                            cholOV_a, cholOV_b, cholVV_a, cholVV_b, v2ijab_bbbb, v2ijab_abab, CI,
                            unit_tis, false);
  
                sch.execute(); // TODO: not needed if not profiling
                auto gf_t_y2 = std::chrono::high_resolution_clock::now();
                gf_t_y2_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_y2 - gf_t_y1))
                    .count();
  
                // clang-format off
                sch
                  (d1_b() =  1.0 * Hy1_b())
                  (d1_b(p1_vb) -= std::complex<double>(gf_omega,gf_eta) * y1_b(p1_vb))
                  (d1_b() += B1_b())
                  (d2_bbb() =  1.0 * Hy2_bbb())
                  (d2_bbb(h1_ob,p1_vb,p2_vb) -= std::complex<double>(gf_omega,gf_eta) * y2_bbb(h1_ob,p1_vb,p2_vb))
                  (d2_aba() =  1.0 * Hy2_aba())
                  (d2_aba(h1_oa,p1_vb,p2_va) -= std::complex<double>(gf_omega,gf_eta) * y2_aba(h1_oa,p1_vb,p2_va))
                  .execute();
                // clang-format on
  
                auto gf_t_res_1 = std::chrono::high_resolution_clock::now();
                gf_t_res_tot_1 +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_1 - gf_t_y2))
                    .count();
  
                auto d1_b_norm   = norm(d1_b);
                auto d2_bbb_norm = norm(d2_bbb);
                auto d2_aba_norm = norm(d2_aba);
  
                auto gf_t_res_2 = std::chrono::high_resolution_clock::now();
                gf_t_res_tot_2 +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res_2 - gf_t_res_1))
                    .count();
  
                double gf_residual = sqrt(d1_b_norm * d1_b_norm + 0.5 * d2_bbb_norm * d2_bbb_norm +
                                          d2_aba_norm * d2_aba_norm)
                                       .real();
  
                if(debug && rank == root_ppi)
                  std::cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                       << "): #iter " << total_iter << ", residual = " << std::fixed
                       << std::setprecision(6) << gf_residual << std::endl;
                auto gf_t_res = std::chrono::high_resolution_clock::now();
                gf_t_res_tot +=
                  std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_res - gf_t_y2))
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
                  (dy1_b(p1_vb) = Minv_b(p1_vb,p2_vb) * d1_b(p2_vb))
                  (dy2_bbb(h1_ob,p1_vb,p2_vb) = dtmp_bbb(h1_ob,p1_vb,p2_vb) * d2_bbb(h1_ob,p1_vb,p2_vb))
                  (dy2_aba(h1_oa,p1_vb,p2_va) = dtmp_aba(h1_oa,p1_vb,p2_va) * d2_aba(h1_oa,p1_vb,p2_va))
                  (y1_b(p1_vb) += gf_damping_factor * dy1_b(p1_vb))
                  (y2_bbb(h1_ob,p1_vb,p2_vb) += gf_damping_factor * dy2_bbb(h1_ob,p1_vb,p2_vb))
                  (y2_aba(h1_oa,p1_vb,p2_va) += gf_damping_factor * dy2_aba(h1_oa,p1_vb,p2_va))
                  (e1_b(p1_vb,dh1) = d1_b(p1_vb))
                  (e2_bbb(h1_ob,p1_vb,p2_vb,dh1) = d2_bbb(h1_ob,p1_vb,p2_vb))
                  (e2_aba(h1_oa,p1_vb,p2_va,dh1) = d2_aba(h1_oa,p1_vb,p2_va))
                  (yy1_b(p1_vb,dh1) = y1_b(p1_vb))
                  (yy2_bbb(h1_ob,p1_vb,p2_vb,dh1) = y2_bbb(h1_ob,p1_vb,p2_vb))
                  (yy2_aba(h1_oa,p1_vb,p2_va,dh1) = y2_aba(h1_oa,p1_vb,p2_va))
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
              adv_gf_diis_b(ec, MO, ndiis, e1_b, e2_bbb, e2_aba, yy1_b, yy2_bbb, yy2_aba, y1_b,
                            y2_bbb, y2_aba, diis_tis, unit_tis, true);
  
              auto gf_t_dis = std::chrono::high_resolution_clock::now();
              gf_t_dis_tot +=
                std::chrono::duration_cast<std::chrono::duration<double>>((gf_t_dis - gf_t_tmp))
                  .count();
            } // end iter
  
            write_to_disk(y1_b, y1_b_wpi_file);
            write_to_disk(y2_bbb, y2_bbb_wpi_file);
            write_to_disk(y2_aba, y2_aba_wpi_file);
  
            sch.deallocate(e1_b, e2_bbb, e2_aba, yy1_b, yy2_bbb, yy2_aba).execute();
          }
          else { gf_conv = true; }
  
          if(!gf_conv) { // && rank == root_ppi
            const std::string error_string = gfo.str() + "," + std::to_string(pi) + ".";
            tamm_terminate("ERROR: GF-CCSD does not converge for w,oi = " + error_string);
          }
  
          auto   gf_t2 = std::chrono::high_resolution_clock::now();
          double gftime =
            std::chrono::duration_cast<std::chrono::duration<double>>((gf_t2 - gf_t1)).count();
          if(rank == root_ppi) {
            std::string gf_stats;
            gf_stats = gfacc_str("A-GF-CCSD Time for w,oi (", gfo.str(), ",", std::to_string(pi),
                                 ") = ", std::to_string(gftime),
                                 " secs, #iter = ", std::to_string(total_iter));
  
            // if(debug){
            gf_stats += gfacc_str("|----------initial guess  : ", std::to_string(gf_t_guess));
            gf_stats += gfacc_str("|----------y1 contraction : ", std::to_string(gf_t_y1_tot));
            gf_stats += gfacc_str("|----------y2 contraction : ", std::to_string(gf_t_y2_tot));
            gf_stats += gfacc_str("|----------computing res. : ", std::to_string(gf_t_res_tot));
            gf_stats +=
              gfacc_str("           |----------misc. contr. : ", std::to_string(gf_t_res_tot_1));
            gf_stats +=
              gfacc_str("           |----------compt. norm  : ", std::to_string(gf_t_res_tot_2));
            gf_stats += gfacc_str("|----------updating y1/y2 : ", std::to_string(gf_t_upd_tot));
            gf_stats += gfacc_str("|----------diis update    : ", std::to_string(gf_t_dis_tot));
            //}
            std::cout << std::fixed << std::setprecision(6) << gf_stats << std::flush;
          }
  
          sch
            .deallocate(Hy1_b, Hy2_bbb, Hy2_aba, d1_b, d2_bbb, d2_aba, dy1_b, dy2_bbb, dy2_aba, y1_b,
                        y2_bbb, y2_aba, y1, Minv, Minv_b, B1, B1_b)
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
      std::cout << "Total A-GF-CCSD Time (w = " << gfo.str() << ") = " << time << " secs"
                << std::endl;
      std::cout << std::string(55, '-') << std::endl;
    }
  
    gsch.deallocate(dtmp_bbb, dtmp_aba).execute();
  } // gfccsd_driver_ea_b

// Explicit template instantiation
template class GFCCSD_EA_B_Driver<double>;
#endif

} // namespace exachem::cc::gfcc