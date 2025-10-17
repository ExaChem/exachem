/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#include "exachem/cc/gfcc/gfccsd_ea_a.hpp"

namespace exachem::cc::gfcc {

#if 0
  template<typename T>
  void GFCCSD_EA_A_Driver<T>::gfccsd_driver_ea_a(
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
    const TiledIndexSpace& unit_tis, string files_prefix, string levelstr, double gf_omega) {
  
    using ComplexTensor  = Tensor<std::complex<T>>;
    using VComplexTensor = std::vector<Tensor<std::complex<T>>>;
    using CMatrix = Eigen::Matrix<std::complex<T>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  
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
  
    const auto [p1_va, p2_va] = v_alpha.labels<2>("all");
    const auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
    const auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
    const auto [h1_ob, h2_ob] = o_beta.labels<2>("all");
  
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
  
    ComplexTensor dtmp_aaa{o_alpha, v_alpha, v_alpha};
    ComplexTensor dtmp_bab{o_beta, v_alpha, v_beta};
    ComplexTensor::allocate(&gec, dtmp_aaa, dtmp_bab);
  
    // double au2ev = 27.2113961;
  
  const   std::string dtmp_aaa_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_aaa.l" + levelstr;
    const std::string dtmp_bab_file = files_prefix + ".W" + gfo.str() + ".a_dtmp_bab.l" + levelstr;
  
    if(fs::exists(dtmp_aaa_file) && fs::exists(dtmp_bab_file)) {
      read_from_disk(dtmp_aaa, dtmp_aaa_file);
      read_from_disk(dtmp_bab, dtmp_bab_file);
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
        sub_sch(dtmp_aaa() = 0)(dtmp_bab() = 0)(dtmp_aaa(h1_oa, p1_va, p2_va) =
                                                  DEArr_EA(h1_oa, p1_va, p2_va))(
          dtmp_bab(h1_ob, p1_va, p2_vb) = DEArr_EA(h1_ob, p1_va, p2_vb))
          .execute();
      }
      gec.pg().barrier();
      gsch.deallocate(DEArr_EA).execute();
      write_to_disk(dtmp_aaa, dtmp_aaa_file);
      write_to_disk(dtmp_bab, dtmp_bab_file);
    }
  
    //------------------------
    auto nranks     = gec.pg().size().value();
  
    const size_t        num_oi           = nva;
    size_t              num_pi_processed = 0;
    std::vector<size_t> pi_tbp;
    // Check pi's already processed
    for(size_t pi = 0; pi < num_oi; pi++) {
      const std::string y1_a_conv_wpi_file =
        files_prefix + ".y1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
      const std::string y2_aaa_conv_wpi_file =
        files_prefix + ".y2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
      const std::string y2_bab_conv_wpi_file =
        files_prefix + ".y2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);
  
      if(fs::exists(y1_a_conv_wpi_file) && fs::exists(y2_aaa_conv_wpi_file) &&
         fs::exists(y2_bab_conv_wpi_file))
        num_pi_processed++;
      else pi_tbp.push_back(pi);
    }
  
    size_t num_pi_remain = num_oi - num_pi_processed;
    if(num_pi_remain == 0) {
      gsch.deallocate(dtmp_aaa, dtmp_bab).execute();
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
    //        (alpha)        //
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
  
          ComplexTensor Hy1_a{v_alpha};
          ComplexTensor Hy2_aaa{o_alpha, v_alpha, v_alpha};
          ComplexTensor Hy2_bab{o_beta, v_alpha, v_beta};
          ComplexTensor y1_a{v_alpha};
          ComplexTensor y2_aaa{o_alpha, v_alpha, v_alpha};
          ComplexTensor y2_bab{o_beta, v_alpha, v_beta};
          Tensor<T>     B1_a{v_alpha};
  
          // if(rank==0) std::cout << "allocate B" << std::endl;
          sch.allocate(B1, B1_a).execute();
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
  
          sch(B1_a(p1_va) = B1(p1_va)).deallocate(B1).execute();
          ec.pg().barrier();
  
          sch.allocate(Hy1_a, Hy2_aaa, Hy2_bab, y1_a, y2_aaa, y2_bab).execute();
  
          double gf_t_guess     = 0.0;
          double gf_t_y1_tot    = 0.0;
          double gf_t_y2_tot    = 0.0;
          double gf_t_res_tot   = 0.0;
          double gf_t_res_tot_1 = 0.0;
          double gf_t_res_tot_2 = 0.0;
          double gf_t_upd_tot   = 0.0;
          double gf_t_dis_tot   = 0.0;
          size_t gf_iter        = 0;
  
          const std::string y1_a_inter_wpi_file =
            const std::string files_prefix + ".y1_a.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
          const std::string y2_aaa_inter_wpi_file =
            const std::string files_prefix + ".y2_aaa.inter.w" + gfo.str() + ".oi" + std::to_string(pi);
          const std::string y2_bab_inter_wpi_file =
            const std::string files_prefix + ".y2_bab.inter.w" + gfo.str() + ".oi" + std::to_string(pi);

          if(fs::exists(y1_a_inter_wpi_file) && fs::exists(y2_aaa_inter_wpi_file) &&
             fs::exists(y2_bab_inter_wpi_file)) {
            read_from_disk(y1_a, y1_a_inter_wpi_file);
            read_from_disk(y2_aaa, y2_aaa_inter_wpi_file);
            read_from_disk(y2_bab, y2_bab_inter_wpi_file);
          }
          else {
            // clang-format off
            sch
              .allocate(y1,Minv)
              (y1()      = 0)
              (Minv()    = 0)
              .execute();
            // clang-format on
  
            gf_guess_ea(ec, MO, nvir, gf_omega, gf_eta, pi, p_evl_sorted_virt, t2v2_v, y1, Minv,
                        true);
  
            sch(y1_a(p1_va) = y1(p1_va)).deallocate(y1, Minv).execute();
          }
  
          // GMRES
  
          ComplexTensor tmp{};
          sch.allocate(tmp).execute();
  
          do {
            int64_t gmres_hist = ccsd_options.gf_ngmres;
  
            gf_iter++;
  
            ComplexTensor  r1_a{v_alpha};
            ComplexTensor  r2_aaa{o_alpha, v_alpha, v_alpha};
            ComplexTensor  r2_bab{o_beta, v_alpha, v_beta};
            VComplexTensor Q1_a;
            VComplexTensor Q2_aaa;
            VComplexTensor Q2_bab;
  
            gfccsd_y1_a(sch, MO, Hy1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_a, y2_aaa, y2_bab,
                        f1, iy1_a, iy1_2_1_a, iy1_2_1_b, v2ijab_aaaa, v2ijab_abab, cholOV_a, cholOV_b,
                        cholVV_a, CI, unit_tis, false);
  
            gfccsd_y2_a(sch, MO, Hy2_aaa, Hy2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, y1_a,
                        y2_aaa, y2_bab, f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a, iy2_b,
                        iy3_1_aaaa, iy3_1_baba, iy3_1_abba, iy3_1_2_a, iy3_1_2_b, iy3_aaaa, iy3_bbbb,
                        iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab, iy4_1_baba, iy4_1_bbbb,
                        iy4_1_abba, iy4_2_aaaa, iy4_2_abba, iy5_aaaa, iy5_baab, iy5_bbbb, iy5_baba,
                        iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b, cholOV_a, cholOV_b, cholVV_a,
                        cholVV_b, v2ijab_aaaa, v2ijab_abab, CI, unit_tis, false);
  
            // clang-format off
            sch 
              .allocate(r1_a,  r2_aaa,  r2_bab)
              (r1_a()   =  1.0 * Hy1_a())
              (r2_aaa() =  1.0 * Hy2_aaa())
              (r2_bab() =  1.0 * Hy2_bab())
              (r1_a(p1_va)               -= std::complex<double>(gf_omega,gf_eta) * y1_a(p1_va))
              (r2_aaa(h1_oa,p1_va,p2_va) -= std::complex<double>(gf_omega,gf_eta) * y2_aaa(h1_oa,p1_va,p2_va))
              (r2_bab(h1_ob,p1_va,p2_vb) -= std::complex<double>(gf_omega,gf_eta) * y2_bab(h1_ob,p1_va,p2_vb))
              (r1_a() += B1_a())
              .execute();
            // clang-format on
  
            auto r1_a_norm   = norm(r1_a);
            auto r2_aaa_norm = norm(r2_aaa);
            auto r2_bab_norm = norm(r2_bab);
  
            auto gf_residual = sqrt(r1_a_norm * r1_a_norm + 0.5 * r2_aaa_norm * r2_aaa_norm +
                                    r2_bab_norm * r2_bab_norm);
  
            if(rank == root_ppi && debug) {
              std::cout << std::fixed << std::setprecision(2) << "w,oi (" << gfo.str() << "," << pi
                   << "): #iter " << gf_iter << "(" << gf_maxiter << "), residual = " << std::fixed
                   << std::setprecision(6) << gf_residual << std::endl;
            }
  
            if(std::abs(gf_residual) < gf_threshold || gf_iter > gf_maxiter) {
              sch.deallocate(r1_a, r2_aaa, r2_bab).execute();
              free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
              Q1_a.clear();
              Q2_aaa.clear();
              Q2_bab.clear();
              if(std::abs(gf_residual) < gf_threshold) gf_conv = true;
              break;
            }
  
            tamm::scale_ip(r1_a, 1.0 / gf_residual);
            tamm::scale_ip(r2_aaa, 1.0 / gf_residual);
            tamm::scale_ip(r2_bab, 1.0 / gf_residual);
            Q1_a.push_back(r1_a);
            Q2_aaa.push_back(r2_aaa);
            Q2_bab.push_back(r2_bab);
            CMatrix cn = CMatrix::Zero(gmres_hist, 1);
            CMatrix sn = CMatrix::Zero(gmres_hist, 1);
            CMatrix H  = CMatrix::Zero(gmres_hist + 1, gmres_hist);
            CMatrix b  = CMatrix::Zero(gmres_hist + 1, 1);
            b(0, 0)    = gf_residual;
  
            for(auto k = 0; k < gmres_hist; k++) {
              ComplexTensor q1_a{v_alpha};
              ComplexTensor q2_aaa{o_alpha, v_alpha, v_alpha};
              ComplexTensor q2_bab{o_beta, v_alpha, v_beta};
  
              gfccsd_y1_a(sch, MO, Hy1_a, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k], Q2_aaa[k],
                          Q2_bab[k], f1, iy1_a, iy1_2_1_a, iy1_2_1_b, v2ijab_aaaa, v2ijab_abab,
                          cholOV_a, cholOV_b, cholVV_a, CI, unit_tis, false);
  
              gfccsd_y2_a(sch, MO, Hy2_aaa, Hy2_bab, t1_a, t1_b, t2_aaaa, t2_bbbb, t2_abab, Q1_a[k],
                          Q2_aaa[k], Q2_bab[k], f1, iy1_1_a, iy1_1_b, iy1_2_1_a, iy1_2_1_b, iy2_a,
                          iy2_b, iy3_1_aaaa, iy3_1_baba, iy3_1_abba, iy3_1_2_a, iy3_1_2_b, iy3_aaaa,
                          iy3_bbbb, iy3_baba, iy3_baab, iy3_abba, iy4_1_aaaa, iy4_1_baab, iy4_1_baba,
                          iy4_1_bbbb, iy4_1_abba, iy4_2_aaaa, iy4_2_abba, iy5_aaaa, iy5_baab,
                          iy5_bbbb, iy5_baba, iy5_abba, iy6_a, iy6_b, cholOO_a, cholOO_b, cholOV_a,
                          cholOV_b, cholVV_a, cholVV_b, v2ijab_aaaa, v2ijab_abab, CI, unit_tis,
                          false);
  
              // clang-format off
              sch 
                .allocate(q1_a,q2_aaa,q2_bab)
                (q1_a()   =  -1.0 * Hy1_a())
                (q2_aaa() =  -1.0 * Hy2_aaa())
                (q2_bab() =  -1.0 * Hy2_bab())
                (q1_a(p1_va)               += std::complex<double>(gf_omega,gf_eta) * Q1_a[k](p1_va))
                (q2_aaa(h1_oa,p1_va,p2_va) += std::complex<double>(gf_omega,gf_eta) * Q2_aaa[k](h1_oa,p1_va,p2_va))
                (q2_bab(h1_ob,p1_va,p2_vb) += std::complex<double>(gf_omega,gf_eta) * Q2_bab[k](h1_ob,p1_va,p2_vb))
                .execute();
              // clang-format on
  
              // Arnoldi iteration or G-S orthogonalization
              for(auto j = 0; j <= k; j++) {
                auto conj_a   = tamm::conj(Q1_a[j]);
                auto conj_aaa = tamm::conj(Q2_aaa[j]);
                auto conj_bab = tamm::conj(Q2_bab[j]);
  
                // clang-format off
                sch
                  (tmp()  = 1.0 * conj_a(p1_va) * q1_a(p1_va))
                  (tmp() += 0.5 * conj_aaa(h1_oa,p1_va,p2_va) * q2_aaa(h1_oa,p1_va,p2_va))
                  (tmp() += 1.0 * conj_bab(h1_ob,p1_va,p2_vb) * q2_bab(h1_ob,p1_va,p2_vb))
                  (q1_a()   -= tmp() * Q1_a[j]())
                  (q2_aaa() -= tmp() * Q2_aaa[j]())
                  (q2_bab() -= tmp() * Q2_bab[j]())
                  .deallocate(conj_a,conj_aaa,conj_bab)
                  .execute();
                // clang-format on
  
                H(j, k) = get_scalar(tmp);
              } // j loop
  
              r1_a_norm   = norm(q1_a);
              r2_aaa_norm = norm(q2_aaa);
              r2_bab_norm = norm(q2_bab);
  
              H(k + 1, k) = sqrt(r1_a_norm * r1_a_norm + 0.5 * r2_aaa_norm * r2_aaa_norm +
                                 r2_bab_norm * r2_bab_norm);
  
              // if(std::abs(H(k+1,k))<1e-16) {
              //   gmres_hist = k+1;
              //   sch.deallocate(q1_a,q2_aaa,q2_bab).execute();
              //   break;
              // }
              std::complex<double> scaling = 1.0 / H(k + 1, k);
  
              CMatrix Hsub = H.block(0, 0, k + 2, k + 1);
              CMatrix bsub = b.block(0, 0, k + 2, 1);
  
              // apply givens rotation
              for(auto i = 0; i < k; i++) {
                auto temp   = cn(i, 0) * H(i, k) + sn(i, 0) * H(i + 1, k);
                H(i + 1, k) = -sn(i, 0) * H(i, k) + cn(i, 0) * H(i + 1, k);
                H(i, k)     = temp;
              }
  
              // if(std::abs(H(k,k))<1e-16){
              //   cn(k,0) = std::complex<double>(0,0);
              //   sn(k,0) = std::complex<double>(1,0);
              // }
              // else{
              auto t   = sqrt(H(k, k) * H(k, k) + H(k + 1, k) * H(k + 1, k));
              cn(k, 0) = abs(H(k, k)) / t;
              sn(k, 0) = cn(k, 0) * H(k + 1, k) / H(k, k);
              // }
  
              H(k, k)     = cn(k, 0) * H(k, k) + sn(k, 0) * H(k + 1, k);
              H(k + 1, k) = std::complex<double>(0, 0);
  
              b(k + 1, 0) = -sn(k, 0) * b(k, 0);
              b(k, 0)     = cn(k, 0) * b(k, 0);
  
              if(rank == root_ppi && debug)
                std::cout << "k: " << k << ", error: " << std::abs(b(k + 1, 0)) << std::endl;
  
              // normalization
              if(std::abs(b(k + 1, 0)) > 1e-2) {
                tamm::scale_ip(q1_a, scaling);
                tamm::scale_ip(q2_aaa, scaling);
                tamm::scale_ip(q2_bab, scaling);
                Q1_a.push_back(q1_a);
                Q2_aaa.push_back(q2_aaa);
                Q2_bab.push_back(q2_bab);
              }
              else {
                gmres_hist = k + 1;
                sch.deallocate(q1_a, q2_aaa, q2_bab).execute();
                break;
              }
            } // k loop
  
            // solve a least square problem in the subspace
            CMatrix Hsub = H.block(0, 0, gmres_hist, gmres_hist);
            CMatrix bsub = b.block(0, 0, gmres_hist, 1);
            CMatrix y    = Hsub.householderQr().solve(bsub);
  
            if(rank == 0) std::cout << "residual: " << (bsub - Hsub * y).norm() << std::endl;
  
            for(auto i = 0; i < gmres_hist; i++) {
              // clang-format off
              sch
                (y1_a(p1_va)               += y(i,0) * Q1_a[i](p1_va))
                (y2_aaa(h1_oa,p1_va,p2_va) += y(i,0) * Q2_aaa[i](h1_oa,p1_va,p2_va))
                (y2_bab(h1_ob,p1_va,p2_vb) += y(i,0) * Q2_bab[i](h1_ob,p1_va,p2_vb));
              // clang-format on
            }
            sch.execute();
  
            write_to_disk(y1_a, y1_a_inter_wpi_file);
            write_to_disk(y2_aaa, y2_aaa_inter_wpi_file);
            write_to_disk(y2_bab, y2_bab_inter_wpi_file);
  
            free_vec_tensors(Q1_a, Q2_aaa, Q2_bab);
            Q1_a.clear();
            Q2_aaa.clear();
            Q2_bab.clear();
  
          } while(true);
  
          // deallocate memory
          sch.deallocate(tmp).execute();
  
          if(gf_conv) {
            const std::string y1_a_conv_wpi_file =
              files_prefix + ".y1_a.w" + gfo.str() + ".oi" + std::to_string(pi);
            const std::string y2_aaa_conv_wpi_file =
              files_prefix + ".y2_aaa.w" + gfo.str() + ".oi" + std::to_string(pi);
            const std::string y2_bab_conv_wpi_file =
              files_prefix + ".y2_bab.w" + gfo.str() + ".oi" + std::to_string(pi);
            write_to_disk(y1_a, y1_a_conv_wpi_file);
            write_to_disk(y2_aaa, y2_aaa_conv_wpi_file);
            write_to_disk(y2_bab, y2_bab_conv_wpi_file);
            fs::remove(y1_a_inter_wpi_file);
            fs::remove(y2_aaa_inter_wpi_file);
            fs::remove(y2_bab_inter_wpi_file);
          }
  
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
                                 " secs, #iter = ", std::to_string(gf_iter));
  
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
  
          sch.deallocate(Hy1_a, Hy2_aaa, Hy2_bab, y1_a, y2_aaa, y2_bab, B1_a).execute();

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
  
    gsch.deallocate(dtmp_aaa, dtmp_bab).execute();
  } // gfccsd_driver_ea_a
#endif

// Explicit template instantiation
template class GFCCSD_EA_A_Driver<double>;

} // namespace exachem::cc::gfcc
